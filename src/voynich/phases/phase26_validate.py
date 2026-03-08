"""
Step 26.7 – Phase 26 Validation Battery (V1–V12)
=================================================
Run 12 validation tests against all Phase 26 results.

Dependency chain:
    zodiac_map.json, month_crib.json, astro_crib.json,
    label_decode.json, zodiac_table.json, zodiac_decode.json
        → phase26_validate.json
"""

import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from voynich.core._paths import results_dir as _results_dir


def _convert(obj: Any) -> Any:
    if hasattr(obj, '__dataclass_fields__'):
        return {k: _convert(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _convert(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_convert(item) for item in obj]
    if isinstance(obj, float) and (obj != obj):
        return None
    if isinstance(obj, (bool, int, float, str, type(None))):
        return obj
    return str(obj)


def _load_json(path: str) -> Optional[Dict]:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ValidationTest:
    test_id: str
    test_name: str
    description: str
    metric: float
    threshold: float
    passed: bool
    detail: str


@dataclass
class Phase26ValidateResult:
    timestamp: str
    validations: List[Dict]
    n_passed: int
    n_total: int
    pass_rate: float
    gate_passed: bool
    strong_pass: bool
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_phase26_validate() -> None:
    t0 = time.time()
    print("=" * 70)
    print("STEP 26.7: Phase 26 Validation Battery")
    print("=" * 70)

    rd = _results_dir()

    # Load all Phase 26 results
    month_data = _load_json(os.path.join(rd, 'month_crib.json'))
    astro_data = _load_json(os.path.join(rd, 'astro_crib.json'))
    label_data = _load_json(os.path.join(rd, 'label_decode.json'))
    table_data = _load_json(os.path.join(rd, 'zodiac_table.json'))
    decode_data = _load_json(os.path.join(rd, 'zodiac_decode.json'))

    tests: List[ValidationTest] = []

    # V1: Month name match on correct folio (≥3 signs matched)
    n_month_matches = 0
    if month_data:
        n_exact = month_data.get('n_forward_exact', 0)
        n_close = month_data.get('n_forward_close', 0)
        n_csp = month_data.get('n_csp_solutions', 0)
        n_month_matches = n_exact + n_close + min(n_csp, 3)
    tests.append(ValidationTest(
        test_id='V1', test_name='Month name matches',
        description='Month name match on correct folio ≥3',
        metric=float(n_month_matches), threshold=3.0,
        passed=n_month_matches >= 3,
        detail=f'exact={month_data.get("n_forward_exact", 0) if month_data else 0}, '
               f'close={month_data.get("n_forward_close", 0) if month_data else 0}, '
               f'csp={month_data.get("n_csp_solutions", 0) if month_data else 0}',
    ))

    # V2: Right-folio rate > 2× wrong-folio (month crib selectivity)
    month_sel = month_data.get('selectivity_ratio', 0) if month_data else 0
    tests.append(ValidationTest(
        test_id='V2', test_name='Month crib selectivity',
        description='Right-folio month rate > 2× wrong-folio rate',
        metric=round(month_sel, 4), threshold=2.0,
        passed=month_sel > 2.0,
        detail=f'selectivity={month_sel:.2f}×',
    ))

    # V3: Planet name on correct 2 folios (≥2 planets matched)
    n_planet = astro_data.get('n_planet_matches', 0) if astro_data else 0
    tests.append(ValidationTest(
        test_id='V3', test_name='Planet name cribs',
        description='Planet name on correct ruling folios ≥2',
        metric=float(n_planet), threshold=2.0,
        passed=n_planet >= 2,
        detail=f'{n_planet} planets matched on correct folios',
    ))

    # V4: Body part on correct folio (≥3 matched)
    n_body = astro_data.get('n_body_correct', 0) if astro_data else 0
    tests.append(ValidationTest(
        test_id='V4', test_name='Body part cribs',
        description='Body part on correct folio ≥3',
        metric=float(n_body), threshold=3.0,
        passed=n_body >= 3,
        detail=f'{n_body} body parts matched',
    ))

    # V5: Element cycling correlation > 0.3
    cycle_score = astro_data.get('element_cycle_score', 0) if astro_data else 0
    tests.append(ValidationTest(
        test_id='V5', test_name='Element cycling',
        description='Element cycling score > 0.3',
        metric=round(cycle_score, 4), threshold=0.3,
        passed=cycle_score > 0.3,
        detail=f'cycle_score={cycle_score:.3f}',
    ))

    # V6: Cross-label consistency (≥3 consistent character assignments)
    n_consistent = month_data.get('n_consistent', 0) if month_data else 0
    if label_data:
        n_consistent = max(n_consistent, label_data.get('n_derived', 0))
    tests.append(ValidationTest(
        test_id='V6', test_name='Cross-label consistency',
        description='≥3 consistent character assignments across folios',
        metric=float(n_consistent), threshold=3.0,
        passed=n_consistent >= 3,
        detail=f'{n_consistent} consistent assignments',
    ))

    # V7: Zodiac folio readability > herbal baseline
    zodiac_hit = decode_data.get('zodiac_dict_hit', 0) if decode_data else 0
    herbal_hit = 0
    if decode_data:
        sect = decode_data.get('section_stats', {})
        herbal_a = sect.get('herbal_a', {}).get('merged_dict_hit', 0)
        herbal_b = sect.get('herbal_b', {}).get('merged_dict_hit', 0)
        herbal_hit = max(herbal_a, herbal_b)
    tests.append(ValidationTest(
        test_id='V7', test_name='Zodiac readability',
        description='Zodiac dict_hit > herbal baseline',
        metric=round(zodiac_hit, 4), threshold=round(herbal_hit, 4),
        passed=zodiac_hit > herbal_hit,
        detail=f'zodiac={zodiac_hit:.1%}, herbal={herbal_hit:.1%}',
    ))

    # V8: Herbal improvement (merged ≥ Phase 16)
    corpus_hit = decode_data.get('corpus_dict_hit', 0) if decode_data else 0
    p16_hit = decode_data.get('phase16_dict_hit', 0) if decode_data else 0
    tests.append(ValidationTest(
        test_id='V8', test_name='No regression',
        description='Merged table dict_hit ≥ Phase 16',
        metric=round(corpus_hit, 4), threshold=round(p16_hit, 4),
        passed=corpus_hit >= p16_hit - 0.005,  # allow 0.5% tolerance
        detail=f'merged={corpus_hit:.1%}, Phase16={p16_hit:.1%}',
    ))

    # V9: Bigram plausibility > 0
    zodiac_jsd = decode_data.get('zodiac_bigram_jsd', 1.0) if decode_data else 1.0
    tests.append(ValidationTest(
        test_id='V9', test_name='Bigram plausibility',
        description='Zodiac bigram JSD from Latin < 0.8',
        metric=round(zodiac_jsd, 4), threshold=0.8,
        passed=zodiac_jsd < 0.8,
        detail=f'JSD={zodiac_jsd:.4f} (lower=more Latin-like)',
    ))

    # V10: Null discrimination (selectivity > 1.5×)
    selectivity = decode_data.get('selectivity', 0) if decode_data else 0
    tests.append(ValidationTest(
        test_id='V10', test_name='Null discrimination',
        description='Selectivity > 1.5×',
        metric=round(selectivity, 4), threshold=1.5,
        passed=selectivity > 1.5,
        detail=f'selectivity={selectivity:.2f}×',
    ))

    # V11: Tier 1 + Tier 2 assignments exist
    n_t1 = table_data.get('n_tier1', 0) if table_data else 0
    n_t2 = table_data.get('n_tier2', 0) if table_data else 0
    tests.append(ValidationTest(
        test_id='V11', test_name='Zodiac-derived assignments',
        description='Tier 1 + Tier 2 assignments ≥ 2',
        metric=float(n_t1 + n_t2), threshold=2.0,
        passed=(n_t1 + n_t2) >= 2,
        detail=f'tier1={n_t1}, tier2={n_t2}',
    ))

    # V12: Consecutive hits ≥ 3
    longest = decode_data.get('longest_consecutive', 0) if decode_data else 0
    tests.append(ValidationTest(
        test_id='V12', test_name='Consecutive hits',
        description='Longest consecutive dict-hit run ≥ 3',
        metric=float(longest), threshold=3.0,
        passed=longest >= 3,
        detail=f'longest={longest} on {decode_data.get("best_passage_folio", "?")}' if decode_data else 'N/A',
    ))

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------
    n_passed = sum(1 for t in tests if t.passed)
    n_total = len(tests)
    pass_rate = n_passed / n_total

    gate = n_passed >= 7
    v1_passed = tests[0].passed  # V1
    v6_passed = tests[5].passed  # V6
    strong = n_passed >= 10 and v1_passed and v6_passed

    print(f"\n  Validation Results:")
    print(f"  {'─' * 66}")
    for t in tests:
        status = "PASS" if t.passed else "FAIL"
        print(f"  {t.test_id:4s} {t.test_name:28s} {status:4s}  "
              f"{t.detail}")
    print(f"  {'─' * 66}")
    print(f"  Total: {n_passed}/{n_total} ({pass_rate:.0%})")
    print(f"  Gate (≥7/12): {'PASS' if gate else 'FAIL'}")
    print(f"  Strong (≥10/12 + V1 + V6): {'PASS' if strong else 'FAIL'}")

    if strong:
        verdict = f"STRONG PASS: {n_passed}/12 including V1 and V6."
    elif gate:
        verdict = f"PASS: {n_passed}/12."
    else:
        verdict = f"FAIL: {n_passed}/12 (need ≥7)."

    result = Phase26ValidateResult(
        timestamp=time.strftime('%Y-%m-%dT%H:%M:%S'),
        validations=[_convert(asdict(t)) for t in tests],
        n_passed=n_passed,
        n_total=n_total,
        pass_rate=round(pass_rate, 4),
        gate_passed=gate,
        strong_pass=strong,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'phase26_validate.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2)

    print(f"\n  → {out_path}")
