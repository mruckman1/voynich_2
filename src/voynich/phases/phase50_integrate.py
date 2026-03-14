"""
Phase 50 Integration: WFST Validation + Word-Level LM
=======================================================
Combine results from four tracks to determine Phase 49 validity
and whether word-level LM rescoring improves decoding.

Dependency chain:
    permuted_table_null.json    (Track A)
    word_lm_rescore.json        (Track B)
    null_battery_50.json        (Track C)
    size_matched_langid.json    (Track D)
        -> phase50_integrate.json
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from voynich.core._paths import results_dir as _results_dir


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _convert(obj: Any) -> Any:
    if hasattr(obj, '__dataclass_fields__'):
        return {k: _convert(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _convert(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_convert(item) for item in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return _convert(obj.tolist())
    if isinstance(obj, float) and (obj != obj):
        return None
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
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class Phase50IntegrateResult:
    track_a_verdict: str
    track_a_full_selectivity: float
    track_b_verdict: str
    track_b_viterbi_rate: float
    track_b_improvement: float
    track_c_verdict: str
    track_c_passed: int
    track_d_verdict: str
    track_d_top_language: str
    overall_verdict: str
    overall_rationale: str
    validations: List[Dict]
    n_validations_passed: int
    gate_passed: bool
    progression_table: List[Dict]
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Main integration function
# ---------------------------------------------------------------------------

def run_phase50_integrate() -> Dict[str, Any]:
    """Phase 50 Integration: Combine all four tracks and render verdict."""
    t0 = time.time()
    rd = _results_dir()

    print("=" * 70)
    print("Phase 50 Integration: WFST Validation + Word-Level LM")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load all track results
    # ------------------------------------------------------------------
    print("\n--- Step 1: Loading track results ---")

    track_a = _safe_load(os.path.join(rd, 'permuted_table_null.json'))
    track_b = _safe_load(os.path.join(rd, 'word_lm_rescore.json'))
    track_c = _safe_load(os.path.join(rd, 'null_battery_50.json'))
    track_d = _safe_load(os.path.join(rd, 'size_matched_langid.json'))

    print(f"  Track A loaded: {'yes' if track_a else 'no'}")
    print(f"  Track B loaded: {'yes' if track_b else 'no'}")
    print(f"  Track C loaded: {'yes' if track_c else 'no'}")
    print(f"  Track D loaded: {'yes' if track_d else 'no'}")

    # ------------------------------------------------------------------
    # 2. Extract key metrics
    # ------------------------------------------------------------------
    print("\n--- Step 2: Extracting key metrics ---")

    # Track A
    a_verdict = track_a.get('verdict', 'UNKNOWN')
    a_full_selectivity = track_a.get('full_selectivity', 0.0)
    print(f"  Track A verdict: {a_verdict}")
    print(f"  Track A full selectivity: {a_full_selectivity:.2f}")

    # Track B
    b_verdict = track_b.get('verdict', 'UNKNOWN')
    b_viterbi_rate = track_b.get('viterbi_rate', 0.0)
    b_stage1_rate = track_b.get('stage1_rate', 0.0)
    b_improvement = b_viterbi_rate - b_stage1_rate
    b_scramble_selectivity = track_b.get('scramble_selectivity', 0.0)
    b_bigram_z = track_b.get('bigram_z_score', 0.0)
    print(f"  Track B verdict: {b_verdict}")
    print(f"  Track B viterbi rate: {b_viterbi_rate:.4f}")
    print(f"  Track B improvement: {b_improvement:+.4f}")

    # Track C
    c_verdict = track_c.get('verdict', 'UNKNOWN')
    c_passed = track_c.get('tests_passed', 0)
    c_total = track_c.get('tests_total', 5)
    print(f"  Track C verdict: {c_verdict}")
    print(f"  Track C passed: {c_passed}/{c_total}")

    # Track D
    d_verdict = track_d.get('verdict', 'UNKNOWN')
    d_top_language = track_d.get('top_language', 'unknown')
    d_margin = track_d.get('margin', 0.0)
    print(f"  Track D verdict: {d_verdict}")
    print(f"  Track D top language: {d_top_language}")

    # ------------------------------------------------------------------
    # 3. Decision tree
    # ------------------------------------------------------------------
    print("\n--- Step 3: Decision tree ---")

    if a_verdict == 'ARTIFACT':
        overall_verdict = 'ED1_INVALIDATED'
        overall_rationale = (
            f'Track A verdict ARTIFACT: full selectivity {a_full_selectivity:.2f}x '
            f'indicates table is no better than random permutation + ED1 + char LM.'
        )
    elif a_verdict == 'MARGINAL':
        if b_verdict == 'WORD_LM_IMPROVES':
            overall_verdict = 'WORD_LM_PARTIAL'
            overall_rationale = (
                f'Track A MARGINAL (selectivity {a_full_selectivity:.2f}x) but '
                f'Track B word-level LM improved viterbi rate by {b_improvement:+.4f}. '
                f'Word LM partially compensates for weak table signal.'
            )
        else:
            overall_verdict = 'INSUFFICIENT_SIGNAL'
            overall_rationale = (
                f'Track A MARGINAL (selectivity {a_full_selectivity:.2f}x) and '
                f'Track B {b_verdict} — neither table nor word LM provides strong signal.'
            )
    elif a_verdict == 'CONFIRMED_CORE_ONLY':
        overall_verdict = 'TABLE_CORE_ONLY'
        overall_rationale = (
            f'Track A CONFIRMED_CORE_ONLY: confirmed triples carry signal '
            f'(partial selectivity < 1.2) but full table is genuine '
            f'(full selectivity {a_full_selectivity:.2f}x). '
            f'Focus improvements on the {track_a.get("n_free_triples", "?")} free triples.'
        )
    elif a_verdict == 'GENUINE':
        if b_verdict == 'WORD_LM_IMPROVES':
            overall_verdict = 'TABLE_PLUS_WORD_LM'
            overall_rationale = (
                f'Track A GENUINE (selectivity {a_full_selectivity:.2f}x) and '
                f'Track B word LM improves by {b_improvement:+.4f}. '
                f'Both table and sequential LM contribute signal.'
            )
        else:
            overall_verdict = 'TABLE_GENUINE_WORDLM_NEUTRAL'
            overall_rationale = (
                f'Track A GENUINE (selectivity {a_full_selectivity:.2f}x) but '
                f'Track B word LM is {b_verdict}. '
                f'Table carries the signal; word-level LM adds nothing.'
            )
    else:
        overall_verdict = 'UNKNOWN'
        overall_rationale = f'Track A verdict {a_verdict} not in decision tree.'

    print(f"  Overall verdict: {overall_verdict}")
    print(f"  Rationale: {overall_rationale}")

    # ------------------------------------------------------------------
    # 4. Validation battery (6 tests)
    # ------------------------------------------------------------------
    print("\n--- Step 4: Validation battery ---")

    validations: List[Dict] = []

    # V1: Track A full_selectivity > 1.0
    v1_pass = a_full_selectivity > 1.0
    validations.append({
        'test_id': 'V1',
        'description': 'Track A full selectivity > 1.0 (table better than random)',
        'metric': 'full_selectivity',
        'threshold': '>1.0',
        'value': f'{a_full_selectivity:.4f}',
        'passed': v1_pass,
    })

    # V2: Track B viterbi_rate > stage1_rate
    v2_pass = b_viterbi_rate > b_stage1_rate
    validations.append({
        'test_id': 'V2',
        'description': 'Track B viterbi_rate > stage1_rate (word LM helps)',
        'metric': 'viterbi_improvement',
        'threshold': '>0',
        'value': f'{b_improvement:+.4f}',
        'passed': v2_pass,
    })

    # V3: Track C tests_passed >= 3
    v3_pass = c_passed >= 3
    validations.append({
        'test_id': 'V3',
        'description': 'Track C null battery: >= 3/5 tests pass',
        'metric': 'tests_passed',
        'threshold': '>=3',
        'value': f'{c_passed}/{c_total}',
        'passed': v3_pass,
    })

    # V4: Track D top language is Latin or Italian
    v4_pass = d_top_language in ('latin', 'italian')
    validations.append({
        'test_id': 'V4',
        'description': 'Track D top language is Latin or Italian',
        'metric': 'top_language',
        'threshold': 'latin or italian',
        'value': d_top_language,
        'passed': v4_pass,
    })

    # V5: Track B scramble_selectivity > 1.05
    v5_pass = b_scramble_selectivity > 1.05
    validations.append({
        'test_id': 'V5',
        'description': 'Track B scramble selectivity > 1.05 (sequential structure)',
        'metric': 'scramble_selectivity',
        'threshold': '>1.05',
        'value': f'{b_scramble_selectivity:.4f}',
        'passed': v5_pass,
    })

    # V6: Track B bigram_z_score > 2.0
    v6_pass = b_bigram_z > 2.0
    validations.append({
        'test_id': 'V6',
        'description': 'Track B bigram z-score > 2.0 (CC bigrams above chance)',
        'metric': 'bigram_z_score',
        'threshold': '>2.0',
        'value': f'{b_bigram_z:.4f}',
        'passed': v6_pass,
    })

    n_validations_passed = sum(1 for v in validations if v['passed'])
    gate_passed = n_validations_passed >= 3

    for v in validations:
        marker = 'PASS' if v['passed'] else 'FAIL'
        print(f"  {v['test_id']}: {v['description']} — {v['value']} {marker}")

    print(f"\n  Validations passed: {n_validations_passed}/{len(validations)}")
    print(f"  Gate (>=3/6): {'PASS' if gate_passed else 'FAIL'}")

    # ------------------------------------------------------------------
    # 5. Progression table
    # ------------------------------------------------------------------
    print("\n--- Step 5: Progression ---")

    progression = [
        {'phase': '11', 'dict_hit': '11.1%', 'selectivity': '1.92x',
         'notes': 'CV phonotactic model baseline'},
        {'phase': '14', 'dict_hit': '19.4%', 'selectivity': '3.00x',
         'notes': 'Sub-cell feature model breakthrough'},
        {'phase': '15', 'dict_hit': '35.4%', 'selectivity': '2.55x',
         'notes': 'Dict expansion + articulatory consistency'},
        {'phase': '16', 'dict_hit': '43.6%', 'selectivity': '3.38x',
         'notes': 'Modifier detection (full corpus baseline)'},
        {'phase': '47', 'dict_hit': '43.6%', 'selectivity': '3.38x',
         'notes': 'Z-score audit, no change'},
        {'phase': '48', 'dict_hit': '43.6%', 'selectivity': '3.38x',
         'notes': 'CRIB_SUGGESTIVE, no change'},
        {'phase': '49', 'dict_hit': '43.6%', 'selectivity': '3.38x',
         'notes': 'Novel computational approaches'},
        {'phase': '50', 'dict_hit': f'{b_viterbi_rate:.1%}' if b_viterbi_rate > 0 else '43.6%',
         'selectivity': f'{a_full_selectivity:.2f}x',
         'notes': f'{overall_verdict} — Track A: {a_verdict}, '
                  f'Track B: {b_verdict}, Track C: {c_passed}/{c_total}, '
                  f'Track D: {d_top_language}'},
    ]

    for p in progression:
        print(f"  Phase {p['phase']}: {p['dict_hit']} ({p['selectivity']}) — {p['notes']}")

    # ------------------------------------------------------------------
    # 6. Save results
    # ------------------------------------------------------------------
    runtime = time.time() - t0

    result = Phase50IntegrateResult(
        track_a_verdict=a_verdict,
        track_a_full_selectivity=round(a_full_selectivity, 4),
        track_b_verdict=b_verdict,
        track_b_viterbi_rate=round(b_viterbi_rate, 4),
        track_b_improvement=round(b_improvement, 4),
        track_c_verdict=c_verdict,
        track_c_passed=c_passed,
        track_d_verdict=d_verdict,
        track_d_top_language=d_top_language,
        overall_verdict=overall_verdict,
        overall_rationale=overall_rationale,
        validations=validations,
        n_validations_passed=n_validations_passed,
        gate_passed=gate_passed,
        progression_table=progression,
        runtime_seconds=round(runtime, 2),
    )

    out_path = _save_json(rd, 'phase50_integrate.json', asdict(result))
    print(f"\n  Saved: {out_path}")

    print("\n" + "=" * 70)
    print("Phase 50 Integration complete")
    print(f"  Overall verdict:     {overall_verdict}")
    print(f"  Validations passed:  {n_validations_passed}/{len(validations)}")
    print(f"  Gate:                {'PASS' if gate_passed else 'FAIL'}")
    print("=" * 70)

    return asdict(result)


# ---------------------------------------------------------------------------
# Full Phase 50 runner
# ---------------------------------------------------------------------------

def run_phase50() -> None:
    """Run full Phase 50 pipeline: all four tracks + integration."""
    from voynich.phases.permuted_table_null import run_permuted_table_null
    from voynich.phases.word_lm_rescore import run_word_lm_rescore
    from voynich.phases.null_battery_50 import run_null_battery_50
    from voynich.phases.size_matched_langid import run_size_matched_langid

    print("\n" + "█" * 70)
    print("  PHASE 50: WFST Validation + Word-Level LM")
    print("█" * 70)

    run_permuted_table_null()
    run_word_lm_rescore()
    run_null_battery_50()
    run_size_matched_langid()
    run_phase50_integrate()

    print("\n" + "█" * 70)
    print("  PHASE 50 COMPLETE")
    print("█" * 70)
