"""
Phase 59, Investigation 5: Cross-Boundary MI Under CVC Decode
===============================================================
Phase 55 measured Currier's self-correlation at 1.45× on the Voynich
corpus.  CVC decode absorbs coda markers into preceding syllables,
changing what constitutes a "word boundary."  If codas are correctly
absorbed, cross-boundary MI should DECREASE because the phonotactic
constraint leaking across boundaries is now captured within CVC syllables.

Dependency chain:
    results/coda_table.json           (Phase 57.1)
    results/combined_refine.json      (Phase 15)
    results/phase55_currier_voynich.json (Phase 55 baseline)
        -> results/cvc_cross_mi.json
"""

import json
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import build_eva_to_triple_lookup, load_corpus, tokenize_eva_chars
from voynich.phases.coda_markers import (
    build_coda_table,
    decode_corpus_cvc,
    decode_corpus_cv_strip,
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
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class MiMeasurement:
    """MI measurement result."""
    name: str
    mi: float
    ratio: float
    n_pairs: int


@dataclass
class CvcCrossMiResult:
    """Full Investigation 5 output."""
    phase: str = "59"
    investigation: str = "5"
    experiment: str = "cvc_cross_mi"
    # Measurements
    eva_raw: Optional[MiMeasurement] = None      # Raw EVA tokens
    cv_decoded: Optional[MiMeasurement] = None    # CV decoded Latin chars
    cvc_decoded: Optional[MiMeasurement] = None   # CVC decoded Latin chars
    # Phase 55 baselines
    phase55_mi: float = 0.0
    phase55_ratio: float = 0.0
    # Comparisons
    cvc_decreased: bool = False
    decrease_fraction: float = 0.0
    cvc_above_null: bool = False
    # Gates
    g1_cvc_lower: bool = False         # CVC MI < CV MI
    g2_above_null: bool = False        # CVC MI still > 0
    g3_partial_absorption: bool = False # decrease 10-50%
    gates_passed: int = 0
    gate_passed: bool = False
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Decoded-token MI measurement
# ---------------------------------------------------------------------------

def measure_decoded_mi(decoded_tokens: List[str]) -> Dict:
    """Compute cross-boundary MI on decoded (Latin character) tokens.

    Uses the DECODED characters (not EVA characters) to measure
    P(first_char_of_token_N+1 | last_char_of_token_N).
    """
    pairs: List[Tuple[str, str]] = []
    for i in range(len(decoded_tokens) - 1):
        w1 = decoded_tokens[i]
        w2 = decoded_tokens[i + 1]
        if w1 and w2 and w1 != '?' and w2 != '?':
            pairs.append((w1[-1], w2[0]))

    n_pairs = len(pairs)
    if n_pairs == 0:
        return {'mi': 0.0, 'ratio': 1.0, 'n_pairs': 0}

    joint = Counter(pairs)
    last_counts = Counter(p[0] for p in pairs)
    first_counts = Counter(p[1] for p in pairs)

    mi = 0.0
    for (last, first), count in joint.items():
        p_joint = count / n_pairs
        p_last = last_counts[last] / n_pairs
        p_first = first_counts[first] / n_pairs
        if p_joint > 0 and p_last > 0 and p_first > 0:
            mi += p_joint * np.log2(p_joint / (p_last * p_first))

    weighted_ratio = 0.0
    for (last, first), count in joint.items():
        p_first_given_last = count / last_counts[last]
        p_first = first_counts[first] / n_pairs
        if p_first > 0:
            weighted_ratio += count * (p_first_given_last / p_first)
    if n_pairs > 0:
        weighted_ratio /= n_pairs

    return {
        'mi': round(float(mi), 6),
        'ratio': round(float(weighted_ratio), 6),
        'n_pairs': n_pairs,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def run_cvc_mi():
    """Investigation 5: Cross-boundary MI under CVC decode."""
    t0 = time.time()
    print("=" * 70)
    print("Phase 59, Investigation 5: Cross-Boundary MI Under CVC Decode")
    print("=" * 70)

    rd = str(_results_dir())

    # Load corpus and decode
    print("\n  Loading corpus ...")
    eva_to_triple = build_eva_to_triple_lookup()
    refine_data = _safe_load(os.path.join(rd, 'combined_refine.json'))
    assignment = refine_data.get('best_assignment', {})

    corpus = load_corpus(verbose=False)
    all_tokens = corpus.get_tokens()
    coda_table = build_coda_table('primary')

    # 1. Raw EVA MI (using existing function)
    print("  Measuring raw EVA cross-boundary MI ...")
    eva_mi = measure_cross_boundary_mi(all_tokens)
    eva_result = MiMeasurement(
        name='eva_raw',
        mi=eva_mi['mi'],
        ratio=eva_mi['ratio'],
        n_pairs=eva_mi['n_pairs'],
    )
    print(f"    EVA MI:    {eva_mi['mi']:.6f} bits, ratio={eva_mi['ratio']:.4f}")

    # 2. CV decoded MI
    print("  Measuring CV-decoded cross-boundary MI ...")
    cv_decoded = decode_corpus_cv_strip(all_tokens, assignment, eva_to_triple, coda_table)
    cv_mi = measure_decoded_mi(cv_decoded)
    cv_result = MiMeasurement(
        name='cv_decoded',
        mi=cv_mi['mi'],
        ratio=cv_mi['ratio'],
        n_pairs=cv_mi['n_pairs'],
    )
    print(f"    CV MI:     {cv_mi['mi']:.6f} bits, ratio={cv_mi['ratio']:.4f}")

    # 3. CVC decoded MI
    print("  Measuring CVC-decoded cross-boundary MI ...")
    cvc_decoded = decode_corpus_cvc(all_tokens, assignment, eva_to_triple, coda_table)
    cvc_mi = measure_decoded_mi(cvc_decoded)
    cvc_result = MiMeasurement(
        name='cvc_decoded',
        mi=cvc_mi['mi'],
        ratio=cvc_mi['ratio'],
        n_pairs=cvc_mi['n_pairs'],
    )
    print(f"    CVC MI:    {cvc_mi['mi']:.6f} bits, ratio={cvc_mi['ratio']:.4f}")

    # Load Phase 55 baseline
    phase55_data = _safe_load(os.path.join(rd, 'phase55_currier_voynich.json'))
    phase55_mi = phase55_data.get('mi', 0.0)
    phase55_ratio = phase55_data.get('ratio', 0.0)
    print(f"\n  Phase 55 baseline:")
    print(f"    MI:    {phase55_mi:.6f} bits, ratio={phase55_ratio:.4f}")

    # Comparisons
    cvc_decreased = cvc_mi['mi'] < cv_mi['mi']
    decrease_frac = ((cv_mi['mi'] - cvc_mi['mi']) / cv_mi['mi']
                     if cv_mi['mi'] > 0 else 0.0)
    cvc_above_null = cvc_mi['mi'] > 0

    print(f"\n  Comparison:")
    print(f"    CVC MI {'<' if cvc_decreased else '>='} CV MI: "
          f"{'DECREASED' if cvc_decreased else 'NOT DECREASED'}")
    print(f"    Decrease fraction: {decrease_frac:.1%}")
    print(f"    CVC MI > 0: {cvc_above_null}")

    # Summary table
    print(f"\n  {'Source':<16} {'MI (bits)':>10} {'Ratio':>8} {'Pairs':>8}")
    print(f"  {'-'*16} {'-'*10} {'-'*8} {'-'*8}")
    for name, mi_val, ratio, pairs in [
        ('EVA raw', eva_mi['mi'], eva_mi['ratio'], eva_mi['n_pairs']),
        ('CV decoded', cv_mi['mi'], cv_mi['ratio'], cv_mi['n_pairs']),
        ('CVC decoded', cvc_mi['mi'], cvc_mi['ratio'], cvc_mi['n_pairs']),
        ('Phase 55', phase55_mi, phase55_ratio, '-'),
    ]:
        print(f"  {name:<16} {mi_val:>10.6f} {ratio:>8.4f} {str(pairs):>8}")

    # Gates
    g1 = cvc_decreased
    g2 = cvc_above_null
    g3 = 0.10 <= decrease_frac <= 0.50
    gates_passed = sum([g1, g2, g3])

    print(f"\n  Validation Gates:")
    print(f"    G1 CVC MI < CV MI:          {'PASS' if g1 else 'FAIL'}")
    print(f"    G2 CVC MI > 0 (still real): {'PASS' if g2 else 'FAIL'}")
    print(f"    G3 decrease 10-50%:         {'PASS' if g3 else 'FAIL'} ({decrease_frac:.1%})")
    print(f"    Gates passed: {gates_passed}/3")

    result = CvcCrossMiResult(
        eva_raw=eva_result,
        cv_decoded=cv_result,
        cvc_decoded=cvc_result,
        phase55_mi=phase55_mi,
        phase55_ratio=phase55_ratio,
        cvc_decreased=cvc_decreased,
        decrease_fraction=round(decrease_frac, 4),
        cvc_above_null=cvc_above_null,
        g1_cvc_lower=g1,
        g2_above_null=g2,
        g3_partial_absorption=g3,
        gates_passed=gates_passed,
        gate_passed=gates_passed >= 2,
        runtime_seconds=round(time.time() - t0, 2),
    )

    path = _save_json(rd, 'cvc_cross_mi.json', result)
    print(f"\n  Saved: {path}")
    print(f"  Investigation 5 completed in {time.time() - t0:.1f}s")
