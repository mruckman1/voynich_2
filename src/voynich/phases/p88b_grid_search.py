"""
Phase 88b — Parameter grid search over generalized Naibbe.

Addresses the "you only tested Greshko's defaults" counterargument: run a
targeted grid over N_TABLES, table weights, output alphabet size, and
affix length, measuring all three diagnostics per config. If any config
reaches tachygraphy-specific thresholds (MI ≥ 1.284× or freq-conn ρ ≥
0.5), the Phase 88 verdict is updated.

Dependency chain:
    data/reference/greshko/nathist_book16.txt
    data/corpus/ (Voynich corpus)
    src/voynich/phases/p88_naibbe_generalized.py (helpers)
        -> results/p88b_grid_search.json
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import load_corpus
from voynich.core.stats import cosine_similarity, entropy_curve
from voynich.phases.p88_naibbe_generalized import (
    _convert,
    _cross_boundary_ratio_plain,
    _freq_conn_rho_plain,
    _load_latin_file,
    _match_corpus_size,
    _PLAINTEXT_ALPHABET,
    encode_bigrams,
    make_grammar_targeting_length,
    make_tables,
    MAX_ORDER,
)


# ---------------------------------------------------------------------------
# Grid definition: 10 diverse configurations
# ---------------------------------------------------------------------------

@dataclass
class GridConfig:
    name: str
    n_tables: int
    weights: List[int]
    output_alpha_size: int
    affix_lo: float
    affix_hi: float


GRID: List[GridConfig] = [
    GridConfig('baseline_greshko',     6, [5, 2, 2, 2, 1, 1], 20, 2.0, 3.0),
    GridConfig('single_table',         1, [1],                 20, 2.0, 3.0),
    GridConfig('two_tables_equal',     2, [1, 1],              20, 2.0, 3.0),
    GridConfig('three_tables_equal',   3, [1, 1, 1],           20, 2.0, 3.0),
    GridConfig('ten_tables_equal',    10, [1] * 10,            20, 2.0, 3.0),
    GridConfig('extreme_skew',         6, [10, 1, 1, 1, 1, 1], 20, 2.0, 3.0),
    GridConfig('alpha_15',             6, [5, 2, 2, 2, 1, 1], 15, 2.0, 3.0),
    GridConfig('alpha_26',             6, [5, 2, 2, 2, 1, 1], 26, 2.0, 3.0),
    GridConfig('short_affixes',        6, [5, 2, 2, 2, 1, 1], 20, 1.0, 2.0),
    GridConfig('long_affixes',         6, [5, 2, 2, 2, 1, 1], 20, 3.0, 4.0),
]

N_SEEDS_PER_CONFIG = 20


# ---------------------------------------------------------------------------
# Per-run measurement
# ---------------------------------------------------------------------------

def _build_cum_weights(weights: List[int]) -> List[float]:
    total = sum(weights)
    cum, running = [], 0.0
    for w in weights:
        running += w / total
        cum.append(running)
    return cum


def _run_config_seed(
    config: GridConfig,
    seed: int,
    latin_text: str,
    voy_chars_full: str,
    voy_shift_full: np.ndarray,
    voy_shift_b: np.ndarray,
    latin_curve: Dict[int, float],
) -> Dict[str, Any]:
    """One (config, seed) encoding + diagnostics."""
    rng = random.Random(seed)

    out_alpha = sorted(rng.sample(
        list("abcdefghijklmnopqrstuvwxyz"),
        config.output_alpha_size,
    ))
    half = config.output_alpha_size // 2
    prefix_chars = out_alpha[:half]
    suffix_chars = out_alpha[half:]

    pre_slots, _, _, _ = make_grammar_targeting_length(
        prefix_chars, rng, lo=config.affix_lo, hi=config.affix_hi)
    suf_slots, _, _, _ = make_grammar_targeting_length(
        suffix_chars, rng, lo=config.affix_lo, hi=config.affix_hi)

    tables = make_tables(pre_slots, suf_slots, config.n_tables, rng)
    cum_weights = _build_cum_weights(config.weights)
    tokens = encode_bigrams(latin_text, tables, rng, cum_weights=cum_weights)
    cipher_chars = "".join(tokens)

    sample = _match_corpus_size(cipher_chars, len(voy_chars_full))
    curve = entropy_curve(sample, max_order=MAX_ORDER)

    orders = list(range(MAX_ORDER + 1))
    shift = np.array([curve.get(k, 0.0) - latin_curve.get(k, 0.0) for k in orders])
    cos_full = float(cosine_similarity(shift, voy_shift_full))
    cos_b = float(cosine_similarity(shift, voy_shift_b))

    mi = _cross_boundary_ratio_plain(tokens)
    fc = _freq_conn_rho_plain(tokens, max_types=2000)

    return {
        'seed': seed,
        'h1': round(float(curve.get(1, 0.0)), 4),
        'cos_full': round(cos_full, 4),
        'cos_b': round(cos_b, 4),
        'mi_ratio': round(float(mi['ratio']), 4),
        'rho': round(float(fc['rho']), 4),
        'n_tokens': len(tokens),
    }


# ---------------------------------------------------------------------------
# Aggregation + thresholds
# ---------------------------------------------------------------------------

MI_THRESHOLD = 1.284        # tachygraphic-specific
RHO_THRESHOLD = 0.5         # tachygraphic-specific (0.618 is Voynich)


@dataclass
class ConfigSummary:
    name: str
    params: Dict[str, Any]
    n_seeds: int
    mean_cos_full: float
    std_cos_full: float
    mean_cos_b: float
    std_cos_b: float
    mean_mi_ratio: float
    std_mi_ratio: float
    max_mi_ratio: float
    mean_rho: float
    std_rho: float
    max_rho: float
    mean_h1: float
    hits_mi_threshold: int
    hits_rho_threshold: int
    runs: List[Dict[str, Any]]


@dataclass
class GridSearchResult:
    timestamp: str
    runtime_seconds: float
    grid_size: int
    n_seeds_per_config: int
    mi_threshold: float
    rho_threshold: float
    voynich_full_mi_ratio: float
    voynich_full_rho: float
    configs: List[ConfigSummary]
    n_configs_hitting_mi: int
    n_configs_hitting_rho: int
    best_mi_config: str
    best_mi_value: float
    best_rho_config: str
    best_rho_value: float
    overall_verdict: str


def run_grid_search() -> None:
    t0 = time.time()
    rd = str(_results_dir())

    print("=" * 60)
    print("Phase 88b: Generalized Naibbe Parameter Grid Search")
    print("=" * 60)

    # ── 1. Load Voynich + Latin + prior Phase 88 results ───────────────
    print("\n  1. Loading references ...")

    corpus = load_corpus(verbose=False)
    voy_text_full = corpus.get_text()
    voy_tokens_full = voy_text_full.split()
    voy_chars_full = "".join(voy_tokens_full)
    voy_curve_full = entropy_curve(voy_chars_full, max_order=MAX_ORDER)

    voy_text_b = corpus.get_text(language='B')
    voy_chars_b = "".join(voy_text_b.split())
    voy_curve_b = entropy_curve(voy_chars_b, max_order=MAX_ORDER)

    latin_path = os.path.join('data', 'reference', 'greshko', 'nathist_book16.txt')
    latin_text = _load_latin_file(latin_path)
    latin_chars = latin_text.replace(" ", "")
    latin_curve = entropy_curve(latin_chars, max_order=MAX_ORDER)

    orders = list(range(MAX_ORDER + 1))
    voy_shift_full = np.array([voy_curve_full.get(k, 0.0) - latin_curve.get(k, 0.0)
                               for k in orders])
    voy_shift_b = np.array([voy_curve_b.get(k, 0.0) - latin_curve.get(k, 0.0)
                            for k in orders])

    # Pull Voynich ratios from phase88 output for reference
    p88_path = os.path.join(rd, 'p88_naibbe_generalized.json')
    voy_mi_full = 1.448
    voy_rho_full = 0.615
    if os.path.exists(p88_path):
        with open(p88_path) as f:
            d = json.load(f)
        voy_mi_full = d.get('voynich_full_cross_boundary_ratio', voy_mi_full)
        voy_rho_full = d.get('voynich_full_freq_conn_rho', voy_rho_full)

    print(f"    Voynich MI ratio (full): {voy_mi_full:.4f}")
    print(f"    Voynich freq-conn ρ (full): {voy_rho_full:+.4f}")
    print(f"    Targets: MI ≥ {MI_THRESHOLD} (tachygraphic) ; ρ ≥ {RHO_THRESHOLD}")

    # ── 2. Grid loop ───────────────────────────────────────────────────
    print(f"\n  2. Running grid ({len(GRID)} configs × {N_SEEDS_PER_CONFIG} seeds = "
          f"{len(GRID) * N_SEEDS_PER_CONFIG} total runs) ...")

    summaries: List[ConfigSummary] = []

    for ci, config in enumerate(GRID):
        print(f"\n  [{ci + 1}/{len(GRID)}] config='{config.name}' "
              f"N={config.n_tables} weights={config.weights} "
              f"alpha={config.output_alpha_size} affix=[{config.affix_lo},{config.affix_hi}]")
        runs = []
        for si in range(N_SEEDS_PER_CONFIG):
            seed = 70000 + ci * 1000 + si
            try:
                r = _run_config_seed(
                    config, seed,
                    latin_text, voy_chars_full,
                    voy_shift_full, voy_shift_b, latin_curve,
                )
                runs.append(r)
            except Exception as e:
                print(f"    [seed {seed}] ERROR: {e}")
                continue

        if not runs:
            print("    (no successful runs)")
            continue

        cos_full_vals = [r['cos_full'] for r in runs]
        cos_b_vals = [r['cos_b'] for r in runs]
        mi_vals = [r['mi_ratio'] for r in runs]
        rho_vals = [r['rho'] for r in runs]
        h1_vals = [r['h1'] for r in runs]

        hits_mi = sum(1 for v in mi_vals if v >= MI_THRESHOLD)
        hits_rho = sum(1 for v in rho_vals if v >= RHO_THRESHOLD)

        summary = ConfigSummary(
            name=config.name,
            params={
                'n_tables': config.n_tables,
                'weights': config.weights,
                'output_alpha_size': config.output_alpha_size,
                'affix_lo': config.affix_lo,
                'affix_hi': config.affix_hi,
            },
            n_seeds=len(runs),
            mean_cos_full=round(float(np.mean(cos_full_vals)), 4),
            std_cos_full=round(float(np.std(cos_full_vals)), 4),
            mean_cos_b=round(float(np.mean(cos_b_vals)), 4),
            std_cos_b=round(float(np.std(cos_b_vals)), 4),
            mean_mi_ratio=round(float(np.mean(mi_vals)), 4),
            std_mi_ratio=round(float(np.std(mi_vals)), 4),
            max_mi_ratio=round(float(np.max(mi_vals)), 4),
            mean_rho=round(float(np.mean(rho_vals)), 4),
            std_rho=round(float(np.std(rho_vals)), 4),
            max_rho=round(float(np.max(rho_vals)), 4),
            mean_h1=round(float(np.mean(h1_vals)), 4),
            hits_mi_threshold=hits_mi,
            hits_rho_threshold=hits_rho,
            runs=runs,
        )
        summaries.append(summary)
        print(f"    cos_full={summary.mean_cos_full:+.4f}±{summary.std_cos_full:.4f}  "
              f"MI={summary.mean_mi_ratio:.4f}±{summary.std_mi_ratio:.4f} "
              f"(max {summary.max_mi_ratio:.4f}, hits {hits_mi}/{summary.n_seeds})  "
              f"ρ={summary.mean_rho:+.4f}±{summary.std_rho:.4f} "
              f"(max {summary.max_rho:+.4f}, hits {hits_rho}/{summary.n_seeds})")

    # ── 3. Overall ──────────────────────────────────────────────────────
    print("\n  3. Aggregating ...")

    n_cfg_mi = sum(1 for s in summaries if s.mean_mi_ratio >= MI_THRESHOLD)
    n_cfg_rho = sum(1 for s in summaries if s.mean_rho >= RHO_THRESHOLD)

    best_mi = max(summaries, key=lambda s: s.mean_mi_ratio)
    best_rho = max(summaries, key=lambda s: s.mean_rho)

    if n_cfg_mi > 0 and n_cfg_rho > 0:
        overall = (f"GRID_REFUTES: {n_cfg_mi} config(s) reach MI ≥ {MI_THRESHOLD} "
                   f"and {n_cfg_rho} reach ρ ≥ {RHO_THRESHOLD} — token-adjacency "
                   "discriminators are NOT robust. Naibbe family matches Voynich on "
                   "all three diagnostics under parameter tuning.")
    elif n_cfg_mi > 0:
        overall = (f"GRID_SHIFTS_MI: {n_cfg_mi} config(s) reach MI ≥ {MI_THRESHOLD}. "
                   f"Freq-conn ρ still insufficient (best {best_rho.mean_rho:+.4f}). "
                   "Section 4.4 weakens; Section 5.1 holds.")
    elif n_cfg_rho > 0:
        overall = (f"GRID_SHIFTS_RHO: {n_cfg_rho} config(s) reach ρ ≥ {RHO_THRESHOLD}. "
                   f"MI still insufficient (best {best_mi.mean_mi_ratio:.4f}). "
                   "Section 5.1 weakens; Section 4.4 holds.")
    else:
        overall = (f"GRID_CONFIRMS_PHASE88: No config (n={len(summaries)}) reaches "
                   f"MI ≥ {MI_THRESHOLD} or ρ ≥ {RHO_THRESHOLD}. "
                   f"Best MI = {best_mi.mean_mi_ratio:.4f} ({best_mi.name}); "
                   f"best ρ = {best_rho.mean_rho:+.4f} ({best_rho.name}). "
                   "Token-adjacency diagnostics are robust to Naibbe parameter "
                   "variation. Phase 88 NAIBBE_1_OF_3 verdict holds under grid search.")

    result = GridSearchResult(
        timestamp=time.strftime('%Y-%m-%dT%H:%M:%S'),
        runtime_seconds=round(time.time() - t0, 2),
        grid_size=len(GRID),
        n_seeds_per_config=N_SEEDS_PER_CONFIG,
        mi_threshold=MI_THRESHOLD,
        rho_threshold=RHO_THRESHOLD,
        voynich_full_mi_ratio=round(voy_mi_full, 4),
        voynich_full_rho=round(voy_rho_full, 4),
        configs=summaries,
        n_configs_hitting_mi=n_cfg_mi,
        n_configs_hitting_rho=n_cfg_rho,
        best_mi_config=best_mi.name,
        best_mi_value=best_mi.mean_mi_ratio,
        best_rho_config=best_rho.name,
        best_rho_value=best_rho.mean_rho,
        overall_verdict=overall,
    )

    out_path = os.path.join(rd, 'p88b_grid_search.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"{'config':22s} {'MI':>9s} {'(max)':>9s} {'ρ':>9s} {'(max)':>9s}  hits_MI  hits_ρ")
    for s in summaries:
        print(f"{s.name:22s} {s.mean_mi_ratio:>9.4f} {s.max_mi_ratio:>9.4f} "
              f"{s.mean_rho:>+9.4f} {s.max_rho:>+9.4f}  "
              f"{s.hits_mi_threshold:>5d}/{s.n_seeds:<2d} "
              f"{s.hits_rho_threshold:>5d}/{s.n_seeds:<2d}")

    print(f"\nBest MI : {best_mi.name} = {best_mi.mean_mi_ratio:.4f} "
          f"(Voynich {voy_mi_full:.4f})")
    print(f"Best ρ  : {best_rho.name} = {best_rho.mean_rho:+.4f} "
          f"(Voynich {voy_rho_full:+.4f})")
    print(f"\nOVERALL VERDICT: {overall}")
    print(f"\n  -> {out_path}")
    print(f"  Runtime: {result.runtime_seconds:.1f}s")
