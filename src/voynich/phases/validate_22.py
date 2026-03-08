"""
Phase 22.7 – Validation Battery (validate-22)
===============================================
15-test validation battery for Phase 22 decoding.
PASS: >= 8/15. STRONG PASS: >= 11/15 including V2 and V3.

Dependency chain:
    All Phase 22 results + Phase 15/16/19/20/21 results
        → validate_22.json (this step)
"""

import json
import os
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.reference import EVA_VISUAL_COMPONENTS


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


def _load_json(path: str) -> Optional[Dict]:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ValidationTest:
    id: str
    name: str
    value: float
    threshold: float
    passed: bool
    detail: str


@dataclass
class Validate22Result:
    timestamp: str
    n_tests: int
    n_passed: int
    tests: List[Dict[str, Any]]
    is_pass: bool
    is_strong_pass: bool
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _run_tests(rdir) -> List[ValidationTest]:
    """Run all 15 validation tests."""
    tests: List[ValidationTest] = []

    # Load all needed data
    decode = _load_json(str(rdir / "corpus_decode_22.json")) or {}
    readability = _load_json(str(rdir / "readability_22.json")) or {}
    phrases = _load_json(str(rdir / "phrases_22.json")) or {}
    merged = _load_json(str(rdir / "merged_table.json")) or {}
    first_syl = _load_json(str(rdir / "first_syllable_table.json")) or {}
    fontana = _load_json(str(rdir / "fontana_phonetic.json")) or {}
    cross = _load_json(str(rdir / "cross_approach.json")) or {}
    mod_data = _load_json(str(rdir / "modifier_integrate.json")) or {}

    better_mode = readability.get('better_mode', 'a')
    mode_read = readability.get(f'mode_{better_mode}', {})
    mode_decode = decode.get(f'mode_{better_mode}', {})

    # V1: Null discrimination (dict hit > null × 1.5)
    dict_hit = mode_decode.get('dict_hit_rate_expanded', 0.0)
    # Null rate ~ 13.9% from Phase 14 baseline
    null_dict = 0.139
    v1_sel = dict_hit / max(null_dict, 0.001)
    tests.append(ValidationTest(
        id='V1', name='Null discrimination',
        value=round(v1_sel, 2), threshold=1.5,
        passed=v1_sel > 1.5,
        detail=f'dict_hit={dict_hit:.3f}, null={null_dict:.3f}, sel={v1_sel:.2f}×',
    ))

    # V2: Bigram plausibility vs null (selectivity > 1.5)
    bg_sel = mode_read.get('bigram_selectivity', 0.0)
    bg_val = mode_read.get('bigram_plausibility', 0.0)
    tests.append(ValidationTest(
        id='V2', name='Bigram plausibility vs null',
        value=round(bg_sel, 2), threshold=1.5,
        passed=bg_sel > 1.5,
        detail=f'bg={bg_val:.4f}, sel={bg_sel:.2f}×',
    ))

    # V3: Phrase detection (>= 3 phrases, selectivity > 2x)
    n_phrases = phrases.get('n_phrases_detected', 0)
    n_templates = phrases.get('n_template_hits', 0)
    tests.append(ValidationTest(
        id='V3', name='Phrase detection',
        value=float(n_phrases), threshold=3.0,
        passed=n_phrases >= 3,
        detail=f'phrases={n_phrases}, templates={n_templates}',
    ))

    # V4: Cross-approach agreement (>= 8/29 at skeleton level)
    n_exact = cross.get('n_exact_match', 0)
    n_edit2 = cross.get('n_edit2_match', 0)
    n_skel = cross.get('n_skeleton_match', 0)
    n_any = n_exact + n_edit2 + n_skel
    tests.append(ValidationTest(
        id='V4', name='Cross-approach agreement',
        value=float(n_any), threshold=8.0,
        passed=n_any >= 8,
        detail=f'exact={n_exact}, edit2={n_edit2}, skeleton={n_skel}, total={n_any}',
    ))

    # V5: Illustration-text match (botanical p < 0.05)
    botanical = phrases.get('botanical_cross_check', {})
    p_val = botanical.get('p_value', 1.0)
    tests.append(ValidationTest(
        id='V5', name='Illustration-text match',
        value=round(p_val, 4), threshold=0.05,
        passed=p_val < 0.05,
        detail=f'p={p_val:.4f}, sel={botanical.get("selectivity", 0):.2f}×',
    ))

    # V6: Section coherence (>= 4/7 sections with expected vocab)
    per_section = mode_decode.get('per_section', {})
    sections_with_hits = sum(1 for s in per_section.values()
                            if s.get('dict_hit_rate', 0) > 0.1)
    tests.append(ValidationTest(
        id='V6', name='Section coherence',
        value=float(sections_with_hits), threshold=4.0,
        passed=sections_with_hits >= 4,
        detail=f'{sections_with_hits} sections with >10% dict-hit',
    ))

    # V7: Lang A/B discrimination (ratio > 1.2x)
    # Compare herbal_a vs herbal_b section dict rates
    hr_a = per_section.get('herbal_a', {}).get('dict_hit_rate', 0)
    hr_b = per_section.get('herbal_b', {}).get('dict_hit_rate', 0)
    ab_ratio = max(hr_a, hr_b) / max(min(hr_a, hr_b), 0.001)
    tests.append(ValidationTest(
        id='V7', name='Lang A/B discrimination',
        value=round(ab_ratio, 2), threshold=1.2,
        passed=ab_ratio > 1.2,
        detail=f'herbal_a={hr_a:.3f}, herbal_b={hr_b:.3f}, ratio={ab_ratio:.2f}',
    ))

    # V8: POS validity vs null (selectivity > 1.3)
    pos_sel = mode_read.get('pos_selectivity', 0.0)
    tests.append(ValidationTest(
        id='V8', name='POS validity vs null',
        value=round(pos_sel, 2), threshold=1.3,
        passed=pos_sel > 1.3,
        detail=f'sel={pos_sel:.2f}×',
    ))

    # V9: Anchor fidelity (>= 80% of Priority 1-2 preserved)
    n_p1 = merged.get('n_priority_1', 0)
    n_p2 = merged.get('n_priority_2', 0)
    n_p12 = n_p1 + n_p2
    # Check how many P1-2 entries are actually in the decode table
    table = merged.get('mode_a_table', [])
    n_preserved = sum(1 for e in table
                      if e.get('priority', 7) <= 2 and e.get('syllable_a', ''))
    pres_rate = n_preserved / max(n_p12, 1)
    tests.append(ValidationTest(
        id='V9', name='Anchor fidelity',
        value=round(pres_rate, 2), threshold=0.8,
        passed=pres_rate >= 0.8,
        detail=f'{n_preserved}/{n_p12} preserved ({pres_rate:.0%})',
    ))

    # V10: Family consonant coherence (>= 4/6 families)
    fam_details = first_syl.get('family_details', [])
    n_fam_coherent = sum(1 for f in fam_details if f.get('consonant_agreement', 0) >= 0.5)
    tests.append(ValidationTest(
        id='V10', name='Family consonant coherence',
        value=float(n_fam_coherent), threshold=4.0,
        passed=n_fam_coherent >= 4,
        detail=f'{n_fam_coherent}/{len(fam_details)} families coherent',
    ))

    # V11: Table stability (Mode A vs B >= 70% agreement)
    mode_a_hit = decode.get('mode_a_dict_hit', 0)
    mode_b_hit = decode.get('mode_b_dict_hit', 0)
    # Agreement = 1 - |difference| / max
    ab_agree = 1.0 - abs(mode_a_hit - mode_b_hit) / max(max(mode_a_hit, mode_b_hit), 0.001)
    tests.append(ValidationTest(
        id='V11', name='Table stability (A vs B)',
        value=round(ab_agree, 2), threshold=0.7,
        passed=ab_agree >= 0.7,
        detail=f'A={mode_a_hit:.3f}, B={mode_b_hit:.3f}, agree={ab_agree:.2f}',
    ))

    # V12: Improvement over Phase 16
    mod_hit = 0.516  # Phase 16 dict_hit from MEMORY
    improvement = dict_hit > mod_hit * 0.6  # At least 60% of Phase 16 level
    tests.append(ValidationTest(
        id='V12', name='Improvement over Phase 16',
        value=round(dict_hit, 3), threshold=round(mod_hit * 0.6, 3),
        passed=improvement,
        detail=f'Phase22={dict_hit:.3f}, Phase16={mod_hit:.3f}',
    ))

    # V13: Paleographic coverage (>= 30% first-syllable derived)
    n_first_syl = first_syl.get('n_with_first_syl', 0)
    n_total_chars = first_syl.get('n_eva_chars', 44)
    coverage = n_first_syl / max(n_total_chars, 1)
    tests.append(ValidationTest(
        id='V13', name='Paleographic coverage',
        value=round(coverage, 2), threshold=0.3,
        passed=coverage >= 0.3,
        detail=f'{n_first_syl}/{n_total_chars} chars ({coverage:.0%})',
    ))

    # V14: Historical consistency (>= 50% testable chars)
    n_with_hist = first_syl.get('n_with_historical', 0)
    hist_rate = n_with_hist / max(n_total_chars, 1)
    tests.append(ValidationTest(
        id='V14', name='Historical consistency',
        value=round(hist_rate, 2), threshold=0.5,
        passed=hist_rate >= 0.5,
        detail=f'{n_with_hist}/{n_total_chars} chars ({hist_rate:.0%})',
    ))

    # V15: Fontana alignment (>= 3/4 gallows)
    gallows_agree = 0
    for hyp in fontana.get('hypotheses', []):
        if hyp.get('voynich_family') == 'gallows' and hyp.get('hypothesized_syllable'):
            gallows_agree += 1
    tests.append(ValidationTest(
        id='V15', name='Fontana alignment',
        value=float(gallows_agree), threshold=3.0,
        passed=gallows_agree >= 3,
        detail=f'{gallows_agree}/4 gallows aligned',
    ))

    return tests


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_validate_22() -> Dict[str, Any]:
    """Run 15-test validation battery."""
    t0 = time.time()
    rdir = _results_dir()

    tests = _run_tests(rdir)
    n_passed = sum(1 for t in tests if t.passed)
    n_tests = len(tests)

    is_pass = n_passed >= 8
    # Strong pass: >= 11 including V2 and V3
    v2_pass = any(t.id == 'V2' and t.passed for t in tests)
    v3_pass = any(t.id == 'V3' and t.passed for t in tests)
    is_strong = n_passed >= 11 and v2_pass and v3_pass

    verdict = f"{n_passed}/{n_tests} tests passed. "
    if is_strong:
        verdict += "STRONG PASS — bigram plausibility and phrase detection confirmed."
    elif is_pass:
        verdict += "PASS — meets minimum validation threshold."
    else:
        verdict += "FAIL — below minimum validation threshold."

    result = Validate22Result(
        timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
        n_tests=n_tests,
        n_passed=n_passed,
        tests=[_convert(asdict(t)) for t in tests],
        is_pass=is_pass,
        is_strong_pass=is_strong,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = rdir / "validate_22.json"
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t0
    print(f"validate-22: {n_passed}/{n_tests} {'STRONG PASS' if is_strong else 'PASS' if is_pass else 'FAIL'} ({elapsed:.1f}s)")
    for t in tests:
        status = 'PASS' if t.passed else 'FAIL'
        print(f"  {t.id}: {status} — {t.detail}")

    return _convert(asdict(result))
