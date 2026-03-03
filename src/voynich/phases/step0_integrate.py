"""
Phase 17.0.6 – Honesty Diagnostics Integration
================================================
Compiles results from all 5 honesty tests into a single verdict matrix
and determines go/no-go for Phase 17 proper.

Dependency chain:
    honesty_dict.json      (Test 1)
    honesty_keywords.json  (Test 2)
    honesty_verbs.json     (Test 3)
    null_corpus.json       (Test 4)
    honesty_words.json     (Test 5)
        → step0_integrate.json (this step)
"""

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

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
    if isinstance(obj, float) and (obj != obj):
        return None
    if isinstance(obj, (bool, int, float, str, type(None))):
        return obj
    return str(obj)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TestVerdict:
    test_name: str
    test_file: str
    gate_passed: bool
    key_metric: float
    gate_threshold: str
    summary: str
    confidence_level: str


@dataclass
class Step0IntegrateResult:
    test_verdicts: List[Dict]
    n_tests: int
    n_passed: int
    n_failed: int
    n_missing: int

    verdict_matrix: Dict[str, bool]

    overall_confidence: str
    confidence_score: float
    go_no_go: str

    strongest_evidence: str
    weakest_evidence: str
    red_flags: List[str]

    phase16_dict_hit: float
    phase16_selectivity: float

    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Test assessment
# ---------------------------------------------------------------------------

def _assess_dict(data: Dict) -> TestVerdict:
    return TestVerdict(
        test_name='dict_control',
        test_file='honesty_dict.json',
        gate_passed=data.get('gate_passed', False),
        key_metric=data.get('original_hit', 0.0),
        gate_threshold='original_hit > 0.25',
        summary=data.get('verdict', ''),
        confidence_level='strong' if data.get('original_hit', 0) > 0.30 else
                         'moderate' if data.get('original_hit', 0) > 0.20 else
                         'weak' if data.get('original_hit', 0) > 0.10 else 'fail',
    )


def _assess_keywords(data: Dict) -> TestVerdict:
    return TestVerdict(
        test_name='keyword_presence',
        test_file='honesty_keywords.json',
        gate_passed=data.get('gate_passed', False),
        key_metric=data.get('n_relaxed_found', 0),
        gate_threshold='n_relaxed >= 20 AND |rho| > 0.3',
        summary=data.get('verdict', ''),
        confidence_level='strong' if data.get('n_relaxed_found', 0) >= 30 else
                         'moderate' if data.get('n_relaxed_found', 0) >= 15 else
                         'weak' if data.get('n_relaxed_found', 0) >= 5 else 'fail',
    )


def _assess_verbs(data: Dict) -> TestVerdict:
    return TestVerdict(
        test_name='verb_decode',
        test_file='honesty_verbs.json',
        gate_passed=data.get('gate_passed', False),
        key_metric=data.get('n_best_ed1', 0),
        gate_threshold='n_ed1_match >= 5 AND |rho| > 0.3',
        summary=data.get('verdict', ''),
        confidence_level='strong' if data.get('n_best_ed1', 0) >= 7 else
                         'moderate' if data.get('n_best_ed1', 0) >= 3 else
                         'weak' if data.get('n_best_ed1', 0) >= 1 else 'fail',
    )


def _assess_null(data: Dict) -> TestVerdict:
    return TestVerdict(
        test_name='null_corpus',
        test_file='null_corpus.json',
        gate_passed=data.get('gate_passed', False),
        key_metric=data.get('null_r3_hit_max', 1.0),
        gate_threshold='null_r3_max < 0.25',
        summary=data.get('verdict', ''),
        confidence_level='strong' if data.get('null_r3_hit_max', 1) < 0.15 else
                         'moderate' if data.get('null_r3_hit_max', 1) < 0.25 else
                         'weak' if data.get('null_r3_hit_max', 1) < 0.40 else 'fail',
    )


def _assess_words(data: Dict) -> TestVerdict:
    return TestVerdict(
        test_name='minimum_words',
        test_file='honesty_words.json',
        gate_passed=data.get('gate_passed', False),
        key_metric=data.get('total_matches', 0),
        gate_threshold='total_matches >= 3',
        summary=data.get('verdict', ''),
        confidence_level='strong' if data.get('total_matches', 0) >= 5 else
                         'moderate' if data.get('total_matches', 0) >= 3 else
                         'weak' if data.get('total_matches', 0) >= 1 else 'fail',
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_step0_integrate() -> None:
    """Step 17.0.6: Compile all honesty tests into go/no-go verdict."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE 17.0.6: Honesty Diagnostics Integration")
    print("=" * 70)

    rd = _results_dir()

    # ─── Load test results ───
    print("\n  1. Loading test results …")

    test_files = {
        'dict_control': ('honesty_dict.json', _assess_dict),
        'keyword_presence': ('honesty_keywords.json', _assess_keywords),
        'verb_decode': ('honesty_verbs.json', _assess_verbs),
        'null_corpus': ('null_corpus.json', _assess_null),
        'minimum_words': ('honesty_words.json', _assess_words),
    }

    verdicts: List[TestVerdict] = []
    verdict_matrix: Dict[str, bool] = {}
    n_missing = 0

    for test_name, (filename, assess_fn) in test_files.items():
        path = os.path.join(rd, filename)
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            tv = assess_fn(data)
            verdicts.append(tv)
            verdict_matrix[test_name] = tv.gate_passed
            status = 'PASS' if tv.gate_passed else 'FAIL'
            print(f"      {test_name:<20} {status:<5} "
                  f"metric={tv.key_metric}, confidence={tv.confidence_level}")
        else:
            verdicts.append(TestVerdict(
                test_name=test_name,
                test_file=filename,
                gate_passed=False,
                key_metric=0.0,
                gate_threshold='N/A',
                summary=f'{filename} not found',
                confidence_level='missing',
            ))
            verdict_matrix[test_name] = False
            n_missing += 1
            print(f"      {test_name:<20} MISSING  ({filename})")

    # ─── Load Phase 16 reference ───
    mod_path = os.path.join(rd, 'modifier_integrate.json')
    p16_hit = 0.5165
    p16_sel = 3.40
    if os.path.exists(mod_path):
        with open(mod_path) as f:
            mod_data = json.load(f)
        p16_hit = mod_data.get('best_dict_hit', p16_hit)
        p16_sel = mod_data.get('best_selectivity', p16_sel)

    # ─── Compute confidence ───
    n_passed = sum(1 for v in verdicts if v.gate_passed)
    n_failed = sum(1 for v in verdicts if not v.gate_passed and v.confidence_level != 'missing')
    n_tests = len(verdicts) - n_missing

    print(f"\n  2. Results: {n_passed}/{n_tests} passed "
          f"({n_missing} missing)")

    # Null corpus has special weight
    null_passed = verdict_matrix.get('null_corpus', False)
    null_failed_hard = False
    for v in verdicts:
        if v.test_name == 'null_corpus' and v.key_metric >= 0.40:
            null_failed_hard = True

    # Confidence scoring
    confidence_weights = {
        'dict_control': 0.20,
        'keyword_presence': 0.15,
        'verb_decode': 0.10,
        'null_corpus': 0.35,
        'minimum_words': 0.20,
    }
    confidence_score = sum(
        confidence_weights.get(v.test_name, 0.0)
        for v in verdicts if v.gate_passed
    )

    # Overall assessment
    if null_failed_hard:
        overall = 'artifact'
        go_no_go = 'NO-GO'
    elif n_passed >= 5:
        overall = 'genuine'
        go_no_go = 'GO'
    elif n_passed >= 4:
        overall = 'probable'
        go_no_go = 'GO'
    elif n_passed >= 3 and null_passed:
        overall = 'suspect'
        go_no_go = 'CONDITIONAL GO'
    elif n_passed >= 2:
        overall = 'suspect'
        go_no_go = 'NO-GO'
    else:
        overall = 'artifact'
        go_no_go = 'NO-GO'

    # ─── Identify strongest/weakest evidence ───
    confidence_order = {'strong': 4, 'moderate': 3, 'weak': 2, 'fail': 1, 'missing': 0}
    sorted_verdicts = sorted(
        verdicts, key=lambda v: confidence_order.get(v.confidence_level, 0), reverse=True,
    )
    strongest = sorted_verdicts[0].test_name if sorted_verdicts else 'none'
    weakest = sorted_verdicts[-1].test_name if sorted_verdicts else 'none'

    # ─── Red flags ───
    red_flags: List[str] = []
    for v in verdicts:
        if v.test_name == 'null_corpus' and not v.gate_passed:
            red_flags.append(
                f"Null corpus achieves comparable dict_hit "
                f"(metric={v.key_metric:.1%}) — pipeline may find "
                f"Latin in structured noise"
            )
        if v.test_name == 'dict_control' and not v.gate_passed:
            red_flags.append(
                f"Original dictionary hit rate only {v.key_metric:.1%} — "
                f"Phase 15-16 gains driven by dictionary expansion"
            )
        if v.test_name == 'minimum_words' and v.key_metric == 0:
            red_flags.append(
                "Zero independently-motivated words decoded correctly"
            )
        if v.test_name == 'keyword_presence' and v.key_metric < 5:
            red_flags.append(
                f"Only {int(v.key_metric)} Latin medical keywords found — "
                f"decoded output lacks basic Latin vocabulary"
            )

    # ─── Print verdict ───
    print(f"\n  3. Verdict Matrix:")
    print(f"      {'Test':<20} {'Gate':>5} {'Confidence':>12}")
    print("      " + "-" * 40)
    for v in verdicts:
        status = 'PASS' if v.gate_passed else 'FAIL'
        print(f"      {v.test_name:<20} {status:>5} {v.confidence_level:>12}")

    print(f"\n  4. Overall Assessment:")
    print(f"      Confidence:  {overall}")
    print(f"      Score:       {confidence_score:.2f}")
    print(f"      Decision:    {go_no_go}")

    if red_flags:
        print(f"\n  5. Red Flags:")
        for flag in red_flags:
            print(f"      ⚠ {flag}")

    # ─── Build verdict string ───
    verdict = (
        f"{go_no_go}: {n_passed}/{n_tests} tests passed. "
        f"Overall confidence: {overall} (score={confidence_score:.2f}). "
        f"Phase 16: {p16_hit:.1%} dict_hit, {p16_sel:.2f}× selectivity. "
        f"{'Red flags: ' + '; '.join(red_flags) if red_flags else 'No red flags.'}"
    )
    print(f"\n  Final: {verdict}")

    # ─── Save ───
    result = Step0IntegrateResult(
        test_verdicts=[_convert(asdict(v)) for v in verdicts],
        n_tests=n_tests,
        n_passed=n_passed,
        n_failed=n_failed,
        n_missing=n_missing,
        verdict_matrix=verdict_matrix,
        overall_confidence=overall,
        confidence_score=round(confidence_score, 4),
        go_no_go=go_no_go,
        strongest_evidence=strongest,
        weakest_evidence=weakest,
        red_flags=red_flags,
        phase16_dict_hit=round(p16_hit, 4),
        phase16_selectivity=round(p16_sel, 2),
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'step0_integrate.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)
    print(f"\n  → {out_path}")
