"""
Phase 77: Timm-Schinner Self-Citation Discriminator Test
=========================================================
Close the paper's acknowledged gap: "Timm & Schinner (2020)'s
self-citation algorithm has not been tested on either measure."

Two tests, no new methodology:
  Test A: Entropy shift — compute cosine vs Voynich shift vector
  Test B: Cross-boundary MI — compute word-final→word-initial MI ratio

Connect Phase 27.1's generator to Phase 55A/B's discriminators.

Dependency chain:
    results/entropy_shift_cipher.json    (Phase 19.2: Latin H0-H6, Voynich shift)
    results/phase55_entropy_extended.json (Phase 55A: existing ranking)
    results/phase55_currier_voynich.json  (Phase 55B: Voynich MI reference)
        -> results/p77_timm_schinner.json
"""

import json
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import load_corpus
from voynich.core.stats import cosine_similarity, entropy_curve
from voynich.phases.gibberish_typology import (
    _build_voynich_char_freq,
    _build_voynich_word_lengths,
    _generate_timm_schinner,
)
from voynich.phases.currier_selfcorr import measure_cross_boundary_mi


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def _convert(obj: Any) -> Any:
    if hasattr(obj, '__dataclass_fields__'):
        return {k: _convert(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _convert(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_convert(item) for item in obj]
    if isinstance(obj, float) and (obj != obj):
        return None
    if isinstance(obj, set):
        return sorted(obj)
    if isinstance(obj, (bool, int, float, str, type(None))):
        return obj
    return str(obj)


def _safe_load(path: str) -> Dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _save_json(rd: str, filename: str, data: Any) -> str:
    path = os.path.join(rd, filename)
    with open(path, 'w') as f:
        json.dump(_convert(data), f, indent=2)
    return path


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class TimmSchinnnerResult:
    phase: str = "77"
    step: str = "77.1"
    experiment: str = "timm_schinner_discriminator"
    # Generation
    n_configs: int = 0
    n_seeds: int = 0
    n_corpora: int = 0
    # Test A: Entropy shift
    default_cosine: float = 0.0
    default_cosine_ci: List[float] = field(default_factory=list)
    best_grid_cosine: float = 0.0
    best_grid_params: Dict[str, Any] = field(default_factory=dict)
    entropy_per_config: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # Test B: Cross-boundary MI
    default_mi_ratio: float = 0.0
    default_mi_ci: List[float] = field(default_factory=list)
    best_grid_mi_ratio: float = 0.0
    best_grid_mi_params: Dict[str, Any] = field(default_factory=dict)
    mi_per_config: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # References
    voynich_mi_ratio: float = 0.0
    tachygraphy_mi_ratio: float = 0.0
    schinner_mi_ratio: float = 0.0
    tachygraphy_cosine: float = 0.0
    # Ranking
    rank_among_mechanisms: int = 0
    n_mechanisms: int = 0
    updated_ranking: List[Dict[str, Any]] = field(default_factory=list)
    # Discrimination
    discriminated_from_tachygraphy: bool = False
    mi_above_null_level: bool = False
    passes_entropy: bool = False
    passes_mi: bool = False
    passes_both: bool = False
    # Gates
    gate_a1: bool = False   # Entropy shift CI does not overlap tachygraphy from above
    gate_a2: bool = False   # Self-citation cosine < 0.5 (discriminated)
    gate_b1: bool = False   # MI ratio < 1.10 (below null level)
    gate_b2: bool = False   # MI ratio CI does not contain Voynich's 1.450
    gates_passed: int = 0
    gate_passed: bool = False
    verdict: str = "UNKNOWN"
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_ts_test() -> TimmSchinnnerResult:
    """Phase 77: Test Timm-Schinner self-citation on entropy shift + MI."""
    t0 = time.time()
    rd = str(_results_dir())

    print("Phase 77 — Timm-Schinner Self-Citation Discriminator Test")
    print("=" * 58)

    # --- Load reference data ---
    print("  Loading reference data...")

    # Phase 19.2: Latin entropy curve and Voynich shift vector
    cipher_data = _safe_load(os.path.join(rd, 'entropy_shift_cipher.json'))
    if not cipher_data:
        print("  ERROR: entropy_shift_cipher.json not found. Run phase19 first.")
        result = TimmSchinnnerResult(verdict='MISSING_DATA',
                                     runtime_seconds=time.time() - t0)
        _save_json(rd, 'p77_timm_schinner.json', asdict(result))
        return result

    # Extract Latin curve and Voynich shift from Phase 19.2 data
    # The cipher data has mechanism profiles with entropy curves
    orders = list(range(7))  # H0 through H6

    # Get Latin reference curve (key: 'latin_entropy_curve')
    latin_curve_data = cipher_data.get('latin_entropy_curve', {})
    if not latin_curve_data:
        latin_curve_data = cipher_data.get('latin_curve', {})

    # Get Voynich shift vector (key: 'observed_shift_vector')
    voynich_shift_data = cipher_data.get('observed_shift_vector', [])
    if not voynich_shift_data:
        voynich_shift_data = cipher_data.get('voynich_shift', [])
    if not voynich_shift_data:
        voynich_shift_data = cipher_data.get('shift_vector', [])

    # Convert to usable format
    latin_curve = {}
    if isinstance(latin_curve_data, dict):
        for k, v in latin_curve_data.items():
            try:
                latin_curve[int(k)] = float(v)
            except (ValueError, TypeError):
                pass
    elif isinstance(latin_curve_data, list):
        for i, v in enumerate(latin_curve_data):
            latin_curve[i] = float(v)

    voynich_shift = []
    if isinstance(voynich_shift_data, list):
        voynich_shift = [float(v) for v in voynich_shift_data]
    elif isinstance(voynich_shift_data, dict):
        voynich_shift = [float(voynich_shift_data.get(str(i), 0)) for i in orders]

    if not latin_curve or not voynich_shift:
        print("  WARNING: Could not extract Latin curve or Voynich shift.")
        print("  Attempting to reconstruct from existing data...")
        # Load Phase 55 extended results for existing mechanism data
        ext_data = _safe_load(os.path.join(rd, 'phase55_entropy_extended.json'))
        if ext_data:
            latin_curve = {}
            voynich_shift = ext_data.get('voynich_shift', [])
            latin_ref = ext_data.get('latin_curve', {})
            if latin_ref:
                for k, v in latin_ref.items():
                    try:
                        latin_curve[int(k)] = float(v)
                    except (ValueError, TypeError):
                        pass

    print(f"  Latin curve orders: {sorted(latin_curve.keys())}")
    print(f"  Voynich shift length: {len(voynich_shift)}")

    # Phase 55B: Voynich MI reference
    currier_data = _safe_load(os.path.join(rd, 'phase55_currier_voynich.json'))
    voynich_mi_ratio = currier_data.get('ratio', currier_data.get('weighted_ratio', 1.450))
    print(f"  Voynich MI ratio: {voynich_mi_ratio:.3f}")

    # Phase 55 existing ranking
    ext_data = _safe_load(os.path.join(rd, 'phase55_entropy_extended.json'))
    existing_ranking = ext_data.get('ranking', ext_data.get('updated_ranking', []))

    # Load Voynich corpus for character frequencies
    print("  Loading Voynich corpus...")
    corpus = load_corpus(verbose=False)
    all_tokens = corpus.get_tokens()
    voynich_text = ' '.join(all_tokens)

    char_freq = _build_voynich_char_freq(voynich_text)
    word_lengths = _build_voynich_word_lengths(all_tokens)

    print(f"  Voynich tokens: {len(all_tokens)}")
    print(f"  Unique chars: {len(char_freq)}")

    # --- Parameter grid ---
    P_COPY = [0.6, 0.7, 0.8]
    P_MUTATE = [0.05, 0.10, 0.15]
    BUFFER_SIZE = [50, 100, 200]
    N_SEEDS = 20
    N_TOKENS = len(all_tokens)  # Match Voynich corpus size

    n_configs = len(P_COPY) * len(P_MUTATE) * len(BUFFER_SIZE)
    n_corpora = n_configs * N_SEEDS
    print(f"  Grid: {n_configs} configs × {N_SEEDS} seeds = {n_corpora} corpora")

    # --- Generate and test ---
    entropy_by_config: Dict[str, List[float]] = {}
    mi_by_config: Dict[str, List[float]] = {}

    config_idx = 0
    for p_copy in P_COPY:
        for p_mutate in P_MUTATE:
            for buffer_size in BUFFER_SIZE:
                config_idx += 1
                config_key = f"pc{p_copy}_pm{p_mutate}_bs{buffer_size}"
                is_default = (p_copy == 0.7 and p_mutate == 0.10
                              and buffer_size == 100)
                tag = " (DEFAULT)" if is_default else ""

                print(f"\n  [{config_idx}/{n_configs}] "
                      f"p_copy={p_copy}, p_mutate={p_mutate}, "
                      f"buffer={buffer_size}{tag}")

                cosines = []
                ratios = []

                for seed in range(N_SEEDS):
                    # Generate corpus
                    tokens = _generate_timm_schinner(
                        char_freq=char_freq,
                        word_lengths=word_lengths,
                        p_copy=p_copy,
                        p_mutate=p_mutate,
                        buffer_size=buffer_size,
                        n_tokens=N_TOKENS,
                        seed=seed,
                    )

                    # Test A: Entropy shift
                    if latin_curve and voynich_shift:
                        text = ' '.join(tokens)
                        curve = entropy_curve(text, max_order=6)
                        shift = [curve.get(k, 0.0) - latin_curve.get(k, 0.0)
                                 for k in orders]
                        shift_arr = np.array(shift)
                        voynich_arr = np.array(voynich_shift[:len(orders)])
                        cos = float(cosine_similarity(shift_arr, voynich_arr))
                        cosines.append(cos)

                    # Test B: Cross-boundary MI
                    mi_result = measure_cross_boundary_mi(tokens)
                    ratio = mi_result.get('ratio', mi_result.get('weighted_ratio', 0))
                    ratios.append(ratio)

                # Aggregate
                entropy_by_config[config_key] = cosines
                mi_by_config[config_key] = ratios

                if cosines:
                    mean_cos = float(np.mean(cosines))
                    print(f"    Entropy cosine: {mean_cos:.4f} "
                          f"± {float(np.std(cosines)):.4f}")
                mean_ratio = float(np.mean(ratios))
                print(f"    MI ratio: {mean_ratio:.4f} "
                      f"± {float(np.std(ratios)):.4f}")

    # --- Aggregate results per config ---
    print("\n  Aggregating results...")
    entropy_results: Dict[str, Dict[str, Any]] = {}
    mi_results: Dict[str, Dict[str, Any]] = {}

    for config_key in entropy_by_config:
        cosines = entropy_by_config[config_key]
        if cosines:
            entropy_results[config_key] = {
                'mean_cosine': float(np.mean(cosines)),
                'std_cosine': float(np.std(cosines)),
                'ci_low': float(np.percentile(cosines, 2.5)),
                'ci_high': float(np.percentile(cosines, 97.5)),
                'n_seeds': len(cosines),
            }

        ratios = mi_by_config[config_key]
        mi_results[config_key] = {
            'mean_ratio': float(np.mean(ratios)),
            'std_ratio': float(np.std(ratios)),
            'ci_low': float(np.percentile(ratios, 2.5)),
            'ci_high': float(np.percentile(ratios, 97.5)),
            'n_seeds': len(ratios),
        }

    # --- Default configuration results ---
    default_key = "pc0.7_pm0.1_bs100"
    default_entropy = entropy_results.get(default_key, {})
    default_mi = mi_results.get(default_key, {})

    default_cosine = default_entropy.get('mean_cosine', 0.0)
    default_cosine_ci = [default_entropy.get('ci_low', 0.0),
                         default_entropy.get('ci_high', 0.0)]
    default_mi_ratio = default_mi.get('mean_ratio', 0.0)
    default_mi_ci = [default_mi.get('ci_low', 0.0),
                     default_mi.get('ci_high', 0.0)]

    # --- Best configuration ---
    best_entropy_key = max(entropy_results,
                           key=lambda k: entropy_results[k]['mean_cosine']) \
        if entropy_results else default_key
    best_mi_key = max(mi_results,
                      key=lambda k: mi_results[k]['mean_ratio']) \
        if mi_results else default_key

    best_grid_cosine = entropy_results.get(best_entropy_key, {}).get(
        'mean_cosine', 0.0)
    best_grid_mi = mi_results.get(best_mi_key, {}).get('mean_ratio', 0.0)

    # Parse config keys back to params
    def _parse_config(key):
        parts = key.split('_')
        return {
            'p_copy': float(parts[0][2:]),
            'p_mutate': float(parts[1][2:]),
            'buffer_size': int(parts[2][2:]),
        }

    print(f"\n  Default config: cosine={default_cosine:.4f}, "
          f"MI={default_mi_ratio:.4f}")
    print(f"  Best entropy config: {best_entropy_key} "
          f"cosine={best_grid_cosine:.4f}")
    print(f"  Best MI config: {best_mi_key} "
          f"MI={best_grid_mi:.4f}")

    # --- Reference values ---
    tachygraphy_cosine = 0.820
    tachygraphy_mi = 1.284
    schinner_mi = 1.044

    # --- Update ranking ---
    print("\n  Updating mechanism ranking...")
    updated_ranking = []
    for entry in existing_ranking:
        if isinstance(entry, dict):
            updated_ranking.append({
                'mechanism': entry.get('mechanism', entry.get('name', '?')),
                'cosine': entry.get('cosine', entry.get('mean_cosine', 0.0)),
            })
        elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
            updated_ranking.append({
                'mechanism': entry[0],
                'cosine': entry[1],
            })

    # Add self-citation
    updated_ranking.append({
        'mechanism': 'self_citation_default',
        'cosine': default_cosine,
        'cosine_ci': default_cosine_ci,
        'mi_ratio': default_mi_ratio,
        'mi_ci': default_mi_ci,
    })
    updated_ranking.append({
        'mechanism': 'self_citation_best_grid',
        'cosine': best_grid_cosine,
        'mi_ratio': best_grid_mi,
    })

    updated_ranking.sort(key=lambda x: -x.get('cosine', 0))
    n_mechanisms = len(updated_ranking)

    ts_rank = next(
        (i + 1 for i, r in enumerate(updated_ranking)
         if r['mechanism'] == 'self_citation_default'),
        n_mechanisms)

    print(f"  Self-citation rank: {ts_rank}/{n_mechanisms}")
    print(f"  Updated ranking (top 5):")
    for i, r in enumerate(updated_ranking[:5]):
        marker = " <--" if 'self_citation' in r.get('mechanism', '') else ""
        print(f"    {i + 1}. {r['mechanism']}: {r.get('cosine', 0):.4f}{marker}")

    # --- Discrimination tests ---
    # A1: CI does not overlap tachygraphy from above
    gate_a1 = default_cosine_ci[1] < tachygraphy_cosine if default_cosine_ci[1] else True
    # A2: Cosine < 0.5 (clearly discriminated)
    gate_a2 = default_cosine < 0.5
    # B1: MI ratio < 1.10 (null level)
    gate_b1 = default_mi_ratio < 1.10
    # B2: MI CI does not contain Voynich's ratio
    gate_b2 = default_mi_ci[1] < voynich_mi_ratio if default_mi_ci[1] else True

    passes_entropy = default_cosine > 0.50
    passes_mi = default_mi_ratio > 1.10
    passes_both = passes_entropy and passes_mi
    discriminated = gate_a1 and gate_a2
    mi_above_null = not gate_b1

    gates_passed = sum([gate_a1, gate_a2, gate_b1, gate_b2])

    print(f"\n  Gates:")
    print(f"    A1 (entropy CI < tachygraphy): "
          f"{'PASS' if gate_a1 else 'FAIL'} "
          f"(CI high={default_cosine_ci[1]:.4f} vs tachy={tachygraphy_cosine})")
    print(f"    A2 (cosine < 0.5): "
          f"{'PASS' if gate_a2 else 'FAIL'} ({default_cosine:.4f})")
    print(f"    B1 (MI ratio < 1.10): "
          f"{'PASS' if gate_b1 else 'FAIL'} ({default_mi_ratio:.4f})")
    print(f"    B2 (MI CI < Voynich {voynich_mi_ratio:.3f}): "
          f"{'PASS' if gate_b2 else 'FAIL'} "
          f"(CI high={default_mi_ci[1]:.4f})")
    print(f"    Total: {gates_passed}/4")

    # --- Verdict ---
    if gate_a1 and gate_a2 and gate_b1:
        verdict = 'SELF_CITATION_ELIMINATED'
    elif gate_a1 and gate_a2:
        verdict = 'ENTROPY_DISCRIMINATED_MI_SURVIVES'
    elif gate_b1 and gate_b2:
        verdict = 'MI_DISCRIMINATED_ENTROPY_SURVIVES'
    elif passes_both:
        verdict = 'SELF_CITATION_VIABLE'
    else:
        verdict = 'PARTIAL_DISCRIMINATION'

    print(f"\n  Verdict: {verdict}")

    # --- Comparison table ---
    print(f"\n  Mechanism Comparison:")
    print(f"    {'Mechanism':<25} {'Cosine':>8} {'MI Ratio':>10}")
    print(f"    {'-'*25} {'-'*8} {'-'*10}")
    print(f"    {'Tachygraphy':<25} {tachygraphy_cosine:>8.3f} "
          f"{tachygraphy_mi:>10.3f}")
    print(f"    {'Self-citation (default)':<25} {default_cosine:>8.3f} "
          f"{default_mi_ratio:>10.3f}")
    print(f"    {'Self-citation (best)':<25} {best_grid_cosine:>8.3f} "
          f"{best_grid_mi:>10.3f}")
    print(f"    {'Schinner stochastic':<25} {'~0.95':>8} "
          f"{schinner_mi:>10.3f}")
    print(f"    {'Voynich (reference)':<25} {'1.000':>8} "
          f"{voynich_mi_ratio:>10.3f}")

    # --- Build result ---
    result = TimmSchinnnerResult(
        n_configs=n_configs,
        n_seeds=N_SEEDS,
        n_corpora=n_corpora,
        default_cosine=round(default_cosine, 4),
        default_cosine_ci=[round(c, 4) for c in default_cosine_ci],
        best_grid_cosine=round(best_grid_cosine, 4),
        best_grid_params=_parse_config(best_entropy_key),
        entropy_per_config={k: {kk: round(vv, 4) for kk, vv in v.items()
                                if isinstance(vv, float)}
                            for k, v in entropy_results.items()},
        default_mi_ratio=round(default_mi_ratio, 4),
        default_mi_ci=[round(c, 4) for c in default_mi_ci],
        best_grid_mi_ratio=round(best_grid_mi, 4),
        best_grid_mi_params=_parse_config(best_mi_key),
        mi_per_config={k: {kk: round(vv, 4) for kk, vv in v.items()
                           if isinstance(vv, float)}
                       for k, v in mi_results.items()},
        voynich_mi_ratio=round(voynich_mi_ratio, 4),
        tachygraphy_mi_ratio=tachygraphy_mi,
        schinner_mi_ratio=schinner_mi,
        tachygraphy_cosine=tachygraphy_cosine,
        rank_among_mechanisms=ts_rank,
        n_mechanisms=n_mechanisms,
        updated_ranking=updated_ranking[:15],  # Top 15 for JSON size
        discriminated_from_tachygraphy=discriminated,
        mi_above_null_level=mi_above_null,
        passes_entropy=passes_entropy,
        passes_mi=passes_mi,
        passes_both=passes_both,
        gate_a1=gate_a1,
        gate_a2=gate_a2,
        gate_b1=gate_b1,
        gate_b2=gate_b2,
        gates_passed=gates_passed,
        gate_passed=gates_passed >= 3,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 1),
    )

    path = _save_json(rd, 'p77_timm_schinner.json', asdict(result))
    print(f"\n  Saved: {path}")
    print(f"  Runtime: {result.runtime_seconds:.1f}s")

    return result
