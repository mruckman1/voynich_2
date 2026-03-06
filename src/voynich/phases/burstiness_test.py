"""
Phase 18.1 – Burstiness / Spatial Autocorrelation Test
=======================================================

Discriminates H1 (Procedural Hoax) from H2/H3 (meaningful text) by
measuring whether word recurrences are bursty (clustered in local
stretches) or Poisson-uniform (memoryless, as a table generator would
produce).

Metric: coefficient of variation (CV) of inter-arrival gaps for
mid-frequency token types.

  Poisson / hoax  → CV ≈ 1.0
  Natural text    → CV >> 1.0  (topical clustering)

Dependency chain:
    (none — reads corpus directly)
        -> burstiness_test.json
"""

import json
import math
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as sp_stats

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import load_corpus
from voynich.core.reference import load_reference_corpus
from voynich.core.stats import bootstrap_ci, coefficient_of_variation


# ---------------------------------------------------------------------------
# JSON serialiser (project convention — duplicated per module)
# ---------------------------------------------------------------------------

def _convert(obj: Any) -> Any:
    if hasattr(obj, '__dataclass_fields__'):
        return {k: _convert(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _convert(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_convert(item) for item in obj]
    if isinstance(obj, float) and (obj != obj):  # NaN
        return None
    if isinstance(obj, (bool, int, float, str, type(None))):
        return obj
    return str(obj)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class BurstinessResult:
    n_qualifying_types: int
    voynich_mean_cv: float
    voynich_median_cv: float
    voynich_cv_std: float
    latin_mean_cv: Optional[float]
    occitan_mean_cv: Optional[float]
    null_mean_cv: float
    null_cv_ci: List[float]                  # [lower, upper] 95 % CI
    poisson_ks_statistic: float
    weibull_ks_statistic: float
    best_fit_distribution: str               # 'poisson' or 'weibull'
    top_bursty_types: List[Dict[str, Any]]   # top-10 most bursty tokens
    hypothesis_support: Dict[str, float]     # H1 / H2 / H3 scores 0-1
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


def _build_token_positions(tokens: List[str]) -> Dict[str, List[int]]:
    """Map each token type to the list of corpus positions where it occurs."""
    positions: Dict[str, List[int]] = {}
    for i, tok in enumerate(tokens):
        positions.setdefault(tok, []).append(i)
    return positions


def _compute_gaps(positions: List[int]) -> List[int]:
    """Inter-arrival gaps from a sorted list of positions."""
    return [positions[i + 1] - positions[i] for i in range(len(positions) - 1)]


def _filter_mid_frequency(
    token_positions: Dict[str, List[int]],
    top_n_exclude: int = 20,
    min_occurrences: int = 5,
) -> Dict[str, List[int]]:
    """Keep mid-frequency types: exclude top-N by count and types with < min_occurrences."""
    ranked = sorted(token_positions.items(), key=lambda kv: -len(kv[1]))
    excluded_top = {tok for tok, _ in ranked[:top_n_exclude]}
    return {
        tok: pos
        for tok, pos in token_positions.items()
        if tok not in excluded_top and len(pos) >= min_occurrences
    }


def _corpus_cv_stats(tokens: List[str], top_n_exclude: int = 20, min_occ: int = 5) -> Dict[str, Any]:
    """Compute per-type CV of inter-arrival gaps, return summary statistics."""
    positions = _build_token_positions(tokens)
    filtered = _filter_mid_frequency(positions, top_n_exclude, min_occ)

    per_type_cv: List[Tuple[str, float]] = []
    all_gaps: List[int] = []

    for tok, pos_list in filtered.items():
        gaps = _compute_gaps(pos_list)
        if len(gaps) < 2:
            continue
        cv = coefficient_of_variation([float(g) for g in gaps])
        per_type_cv.append((tok, cv))
        all_gaps.extend(gaps)

    if not per_type_cv:
        return {'mean_cv': 0.0, 'median_cv': 0.0, 'std_cv': 0.0,
                'n_types': 0, 'per_type': [], 'all_gaps': []}

    cvs = [cv for _, cv in per_type_cv]
    return {
        'mean_cv': float(np.mean(cvs)),
        'median_cv': float(np.median(cvs)),
        'std_cv': float(np.std(cvs)),
        'n_types': len(per_type_cv),
        'per_type': sorted(per_type_cv, key=lambda x: -x[1]),
        'all_gaps': all_gaps,
    }


def _fit_distributions(gaps: List[int]) -> Dict[str, float]:
    """Fit Poisson (geometric inter-arrival) and Weibull to gap data; return KS statistics."""
    if len(gaps) < 10:
        return {'poisson_ks': 1.0, 'weibull_ks': 1.0}

    arr = np.array(gaps, dtype=float)

    # Geometric (Poisson process inter-arrivals): parameter p = 1/mean
    mean_gap = float(np.mean(arr))
    p_geom = 1.0 / mean_gap if mean_gap > 0 else 0.5
    ks_geom, _ = sp_stats.kstest(arr, 'geom', args=(p_geom,))

    # Weibull fit
    try:
        shape, _, scale = sp_stats.weibull_min.fit(arr, floc=0)
        ks_weibull, _ = sp_stats.kstest(arr, 'weibull_min', args=(shape, 0, scale))
    except Exception:
        ks_weibull = 1.0

    return {'poisson_ks': float(ks_geom), 'weibull_ks': float(ks_weibull)}


def _null_cv(tokens: List[str], n_perms: int = 50, seed: int = 42) -> List[float]:
    """Shuffled-null mean CV values (n_perms shuffles)."""
    rng = np.random.default_rng(seed)
    results = []
    for _ in range(n_perms):
        shuffled = list(tokens)
        rng.shuffle(shuffled)
        stats = _corpus_cv_stats(shuffled)
        results.append(stats['mean_cv'])
    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_burstiness_test() -> None:
    """Phase 18.1: spatial autocorrelation / burstiness test."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE 18.1: Burstiness / Spatial Autocorrelation Test")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Voynich corpus ──────────────────────────────────────────────
    print("\n  1. Loading Voynich corpus …")
    corpus = load_corpus(verbose=False)
    tokens_a = corpus.get_tokens(language='A', paragraph_only=True)
    print(f"     {len(tokens_a):,} Language-A paragraph tokens")

    voynich_stats = _corpus_cv_stats(tokens_a)
    n_types = voynich_stats['n_types']
    v_mean_cv = voynich_stats['mean_cv']
    v_median_cv = voynich_stats['median_cv']
    v_std_cv = voynich_stats['std_cv']
    print(f"     {n_types} qualifying types  |  mean CV = {v_mean_cv:.3f}  |  median CV = {v_median_cv:.3f}")

    # ── 2. Distribution fitting ────────────────────────────────────────
    print("\n  2. Fitting gap distributions …")
    fit = _fit_distributions(voynich_stats['all_gaps'])
    poisson_ks = fit['poisson_ks']
    weibull_ks = fit['weibull_ks']
    best_fit = 'poisson' if poisson_ks < weibull_ks else 'weibull'
    print(f"     Poisson (geom) KS = {poisson_ks:.4f}  |  Weibull KS = {weibull_ks:.4f}  →  best fit: {best_fit}")

    # ── 3. Reference corpora ──────────────────────────────────────────
    print("\n  3. Reference corpora …")
    latin_cv: Optional[float] = None
    occitan_cv: Optional[float] = None
    try:
        ref = load_reference_corpus(languages=['latin'], verbose=False)
        latin_tokens = ref.get_combined_tokens('latin')
        if latin_tokens:
            latin_stats = _corpus_cv_stats(latin_tokens)
            latin_cv = latin_stats['mean_cv']
            print(f"     Latin: {len(latin_tokens):,} tokens  |  mean CV = {latin_cv:.3f}")
    except Exception as e:
        print(f"     WARNING: Latin corpus unavailable ({e})")

    try:
        ref_oc = load_reference_corpus(languages=['occitan'], verbose=False)
        occitan_tokens = ref_oc.get_combined_tokens('occitan')
        if occitan_tokens:
            oc_stats = _corpus_cv_stats(occitan_tokens)
            occitan_cv = oc_stats['mean_cv']
            print(f"     Occitan: {len(occitan_tokens):,} tokens  |  mean CV = {occitan_cv:.3f}")
    except Exception:
        print("     WARNING: Occitan corpus unavailable")

    # ── 4. Shuffled null ──────────────────────────────────────────────
    print("\n  4. Computing shuffled null (50 permutations) …")
    null_cvs = _null_cv(tokens_a, n_perms=50)
    null_mean = float(np.mean(null_cvs))
    null_lo = float(np.percentile(null_cvs, 2.5))
    null_hi = float(np.percentile(null_cvs, 97.5))
    print(f"     Null mean CV = {null_mean:.3f}  |  95 % CI = [{null_lo:.3f}, {null_hi:.3f}]")

    # ── 5. Top bursty types ───────────────────────────────────────────
    top_bursty = [
        {'token': tok, 'cv': round(cv, 4), 'count': len(_build_token_positions(tokens_a).get(tok, []))}
        for tok, cv in voynich_stats['per_type'][:10]
    ]

    # ── 6. Hypothesis scoring ─────────────────────────────────────────
    print("\n  5. Scoring hypotheses …")
    # H1: supported if CV ≈ 1.0 (Poisson)
    h1 = _sigmoid(-(v_mean_cv - 1.0) / 0.3)
    # H2: supported if burstiness is comparable to Latin
    ref_cv = latin_cv if latin_cv is not None else 1.8
    h2 = _sigmoid((v_mean_cv - ref_cv + 0.3) / 0.4)
    # H3: bursty like a lexical corpus
    h3 = _sigmoid((v_mean_cv - 1.5) / 0.4)

    # Normalise so they sum to 1
    total = h1 + h2 + h3
    if total > 0:
        h1, h2, h3 = h1 / total, h2 / total, h3 / total

    hypothesis_support = {'H1': round(h1, 4), 'H2': round(h2, 4), 'H3': round(h3, 4)}
    print(f"     H1={h1:.3f}  H2={h2:.3f}  H3={h3:.3f}")

    # ── Verdict ───────────────────────────────────────────────────────
    if v_mean_cv < 1.2:
        verdict = (f"NEAR-POISSON: mean CV = {v_mean_cv:.3f} — token recurrence is nearly "
                   "memoryless, consistent with a procedural hoax (H1).")
    elif v_mean_cv > null_hi:
        verdict = (f"BURSTY: mean CV = {v_mean_cv:.3f} significantly exceeds shuffled null "
                   f"CI [{null_lo:.3f}, {null_hi:.3f}] — topical clustering detected, "
                   "inconsistent with H1. Supports H2/H3.")
    else:
        verdict = (f"MARGINALLY BURSTY: mean CV = {v_mean_cv:.3f} within null CI — "
                   "burstiness signal is weak.")

    print(f"\n  Verdict: {verdict}")

    # ── Save ──────────────────────────────────────────────────────────
    result = BurstinessResult(
        n_qualifying_types=n_types,
        voynich_mean_cv=round(v_mean_cv, 4),
        voynich_median_cv=round(v_median_cv, 4),
        voynich_cv_std=round(v_std_cv, 4),
        latin_mean_cv=round(latin_cv, 4) if latin_cv is not None else None,
        occitan_mean_cv=round(occitan_cv, 4) if occitan_cv is not None else None,
        null_mean_cv=round(null_mean, 4),
        null_cv_ci=[round(null_lo, 4), round(null_hi, 4)],
        poisson_ks_statistic=round(poisson_ks, 4),
        weibull_ks_statistic=round(weibull_ks, 4),
        best_fit_distribution=best_fit,
        top_bursty_types=top_bursty,
        hypothesis_support=hypothesis_support,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'burstiness_test.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)

    print(f"\n  -> {out_path}")
