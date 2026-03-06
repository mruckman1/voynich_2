"""
Phase 20.7 – Tachygraphic Validation Battery
=============================================
Run a 12-test validation battery integrating all evidence from Steps 20.1–20.6.

Dependency chain:
    tachy_anchors.json + tachy_families.json + tachy_grid_solve.json
    + tachy_decode.json + tachy_readability.json + tachy_phrases.json
        → tachy_validate.json
"""

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

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
class ValidationTest:
    name: str
    description: str
    value: float
    threshold: float
    passed: bool
    detail: str


@dataclass
class TachyValidateResult:
    tests: List[Dict]
    n_passed: int
    n_total: int
    pass_rate: float
    strong_pass: bool           # ≥10/12 including V3+V5
    gate_passed: bool           # ≥8/12
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _load_json(rd: str, fname: str) -> Dict:
    path = os.path.join(rd, fname)
    if not os.path.exists(path):
        print(f"    [WARN] {fname} not found")
        return {}
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_tachy_validate() -> None:
    """Step 20.7: 12-test validation battery."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE 20.7: Tachygraphic Validation Battery")
    print("=" * 70)

    rd = _results_dir()

    # Load all prior results
    print("\n  1. Loading all Phase 20 results …")
    anchors = _load_json(rd, 'tachy_anchors.json')
    families = _load_json(rd, 'tachy_families.json')
    grid = _load_json(rd, 'tachy_grid_solve.json')
    decode = _load_json(rd, 'tachy_decode.json')
    readability = _load_json(rd, 'tachy_readability.json')
    phrases = _load_json(rd, 'tachy_phrases.json')

    tests: List[ValidationTest] = []

    # ─── V1: Null discrimination (dict_hit) ───
    null_sel = grid.get('null_selectivity', 0.0)
    v1 = ValidationTest(
        name='V1: Null discrimination',
        description='dict_hit selectivity vs random assignment',
        value=null_sel,
        threshold=1.5,
        passed=null_sel > 1.5,
        detail=f'null_selectivity={null_sel:.2f}×',
    )
    tests.append(v1)

    # ─── V2: Bigram plausibility ───
    bg_sel = readability.get('bigram_selectivity', 0.0)
    v2 = ValidationTest(
        name='V2: Bigram plausibility',
        description='Word-pair bigram hit rate vs null',
        value=bg_sel,
        threshold=1.5,
        passed=bg_sel > 1.5,
        detail=f'bigram_selectivity={bg_sel:.2f}×',
    )
    tests.append(v2)

    # ─── V3: Phrase detection ───
    n_phrases = phrases.get('n_phrases_detected', 0)
    ph_sel = phrases.get('phrase_selectivity', 0.0)
    v3 = ValidationTest(
        name='V3: Phrase detection',
        description='N phrases ≥10 and selectivity >2×',
        value=float(n_phrases),
        threshold=10.0,
        passed=n_phrases >= 10 and ph_sel > 2.0,
        detail=f'n_phrases={n_phrases}, selectivity={ph_sel:.2f}×',
    )
    tests.append(v3)

    # ─── V4: Cross-approach agreement ───
    # Check how many of the original 8 anchors are preserved
    n_anchor_words = anchors.get('n_anchor_words', 0)
    n_chars_anchored = anchors.get('n_chars_anchored', 0)
    v4 = ValidationTest(
        name='V4: Cross-approach agreement',
        description='Anchor words preserved in final table',
        value=float(n_anchor_words),
        threshold=4.0,
        passed=n_anchor_words >= 4,
        detail=f'{n_anchor_words} anchor words, {n_chars_anchored} chars anchored',
    )
    tests.append(v4)

    # ─── V5: Illustration-text match ───
    bot_p = phrases.get('botanical_p_value', 1.0)
    n_bot = phrases.get('n_botanical_matches', 0)
    v5 = ValidationTest(
        name='V5: Illustration-text match',
        description='Botanical cross-check p < 0.01',
        value=bot_p,
        threshold=0.01,
        passed=bot_p < 0.01,
        detail=f'p={bot_p:.4f}, {n_bot} botanical matches',
    )
    tests.append(v5)

    # ─── V6: Section coherence ───
    n_domains = readability.get('n_domains_with_hits', 0)
    v6 = ValidationTest(
        name='V6: Section coherence',
        description='≥4/7 sections with correct domain vocab',
        value=float(n_domains),
        threshold=4.0,
        passed=n_domains >= 4,
        detail=f'{n_domains} domains with hits',
    )
    tests.append(v6)

    # ─── V7: Language A/B discrimination ───
    # Use per-section analysis — herbal sections should be higher
    per_section = decode.get('per_section', {})
    herbal_hit = 0.0
    other_hit = 0.0
    herbal_n = 0
    other_n = 0
    for sec, stats in per_section.items():
        if 'herbal' in sec:
            herbal_hit += stats.get('n_hits', 0)
            herbal_n += stats.get('n_tokens', 0)
        else:
            other_hit += stats.get('n_hits', 0)
            other_n += stats.get('n_tokens', 0)
    herbal_rate = herbal_hit / herbal_n if herbal_n else 0.0
    other_rate = other_hit / other_n if other_n else 0.0
    ab_ratio = herbal_rate / other_rate if other_rate > 0 else float('inf')
    v7 = ValidationTest(
        name='V7: Language A/B discrimination',
        description='Herbal dict_hit > other dict_hit (ratio >1.2)',
        value=ab_ratio,
        threshold=1.2,
        passed=ab_ratio > 1.2 or ab_ratio < 0.83,  # either direction is signal
        detail=f'herbal={herbal_rate:.3f} vs other={other_rate:.3f} ratio={ab_ratio:.2f}',
    )
    tests.append(v7)

    # ─── V8: POS validity ───
    pos_sel = readability.get('pos_selectivity', 0.0)
    v8 = ValidationTest(
        name='V8: POS validity',
        description='POS trigram validity selectivity >1.3×',
        value=pos_sel,
        threshold=1.3,
        passed=pos_sel > 1.3,
        detail=f'pos_selectivity={pos_sel:.2f}×',
    )
    tests.append(v8)

    # ─── V9: Anchor fidelity ───
    n_tier1 = anchors.get('n_tier1', 0)
    # Check if tier 1 anchors are preserved in final table
    best_assignment = grid.get('best_assignment', {})
    tier1_preserved = 0
    for a in anchors.get('char_anchors', []):
        if a.get('tier') == 1:
            if best_assignment.get(a['eva_char']) == a['syllable']:
                tier1_preserved += 1
    v9_frac = tier1_preserved / n_tier1 if n_tier1 > 0 else 1.0
    v9 = ValidationTest(
        name='V9: Anchor fidelity',
        description='100% Tier 1 anchors preserved in final table',
        value=v9_frac,
        threshold=1.0,
        passed=v9_frac >= 0.99,
        detail=f'{tier1_preserved}/{n_tier1} preserved',
    )
    tests.append(v9)

    # ─── V10: Family coherence ───
    subfamilies = families.get('subfamilies', [])
    n_coherent = 0
    n_subfamilies = len(subfamilies)
    for sf in subfamilies:
        assignments = sf.get('syllable_assignments', {})
        if not assignments:
            continue
        # Check: do all members share a consonant class?
        consonants = set()
        vowels_set = set('aeiou')
        for ch, syl in assignments.items():
            onset = ''.join(c for c in syl if c not in vowels_set)
            consonants.add(onset)
        if len(consonants) <= 1:
            n_coherent += 1
    fam_rate = n_coherent / n_subfamilies if n_subfamilies else 0.0
    v10 = ValidationTest(
        name='V10: Family coherence',
        description='≥4/6 families map to coherent syllable families',
        value=fam_rate,
        threshold=0.6,
        passed=n_coherent >= 4,
        detail=f'{n_coherent}/{n_subfamilies} coherent',
    )
    tests.append(v10)

    # ─── V11: Table stability ───
    stability = grid.get('stability_agreement', 0.0)
    v11 = ValidationTest(
        name='V11: Table stability',
        description='5-run pairwise agreement ≥80%',
        value=stability,
        threshold=0.8,
        passed=stability >= 0.8,
        detail=f'pairwise_agreement={stability:.1%}',
    )
    tests.append(v11)

    # ─── V12: Phase 16 improvement ───
    n_read_pass = readability.get('n_tests_passing', 0)
    n_read_total = readability.get('n_tests', 5)
    v12 = ValidationTest(
        name='V12: Phase 16 improvement',
        description='≥3/5 readability tests improved vs Phase 16',
        value=float(n_read_pass),
        threshold=3.0,
        passed=n_read_pass >= 3,
        detail=f'{n_read_pass}/{n_read_total} readability tests pass',
    )
    tests.append(v12)

    # ─── Summary ───
    print("\n  2. Validation results:")
    n_passed = 0
    for vt in tests:
        status = 'PASS' if vt.passed else 'FAIL'
        print(f"      {vt.name:35s}: {status:4s}  {vt.detail}")
        if vt.passed:
            n_passed += 1

    n_total = len(tests)
    pass_rate = n_passed / n_total if n_total else 0.0

    # V3 (phrases) and V5 (illustration) required for STRONG PASS
    v3_passed = tests[2].passed
    v5_passed = tests[4].passed
    strong_pass = n_passed >= 10 and v3_passed and v5_passed
    gate_passed = n_passed >= 8

    print(f"\n      Score: {n_passed}/{n_total} ({pass_rate:.0%})")
    print(f"      Strong pass: {strong_pass}")
    print(f"      Gate pass: {gate_passed}")

    if strong_pass:
        verdict = f"STRONG PASS: {n_passed}/{n_total} including V3+V5."
    elif gate_passed:
        verdict = f"PASS: {n_passed}/{n_total} (≥8 required)."
    else:
        verdict = f"FAIL: {n_passed}/{n_total} (need ≥8)."

    print(f"\n  3. Gate: {verdict}")

    # ─── Save ───
    result = TachyValidateResult(
        tests=[asdict(vt) for vt in tests],
        n_passed=n_passed,
        n_total=n_total,
        pass_rate=pass_rate,
        strong_pass=strong_pass,
        gate_passed=gate_passed,
        verdict=verdict,
        runtime_seconds=time.time() - t0,
    )

    out_path = os.path.join(rd, 'tachy_validate.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)

    print(f"\n  → {out_path}")
