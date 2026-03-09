"""
Phase 28.2 – Internal Consistency Test
========================================
Tests whether crib-derived character assignments are internally consistent:
  A) Cross-source: do Phase 14, Phase 19.8, and Phase 26 agree on the
     same syllable for the same triples?
  B) Family typological: do the assigned syllables respect the
     PHONEME_PLACE_MAP / PHONEME_NUCLEUS_MAP constraints?
  C) Null comparison: does the real consistency exceed random permutations?

Dependency chain:
    crib_extraction.json     (Step 28.1)
    feature_csp.json         (Phase 14 assignment)
        → crib_consistency.json  (this step)
"""

import json
import os
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import build_eva_to_triple_lookup
from voynich.core.reference import (
    EVA_VISUAL_COMPONENTS,
    PHONEME_NUCLEUS_MAP,
    PHONEME_PLACE_MAP,
)


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
class CrossSourceTest:
    """Result of testing one triple across multiple sources."""
    triple: str
    syllable_by_source: Dict[str, str]   # source → syllable
    n_sources: int
    all_agree: bool
    agreed_syllable: Optional[str]


@dataclass
class FamilyConsistencyTest:
    """Result of testing one triple against PHONEME maps."""
    triple: str
    syllable: str
    first_stroke: str
    last_stroke: str
    onset_char: str             # first char of syllable (consonant)
    nucleus_char: str           # remaining chars (vowel)
    onset_in_map: bool
    nucleus_in_map: bool
    is_consistent: bool


@dataclass
class ConsistencyResult:
    # Cross-source
    n_cross_source_tests: int
    n_cross_consistent: int
    n_cross_inconsistent: int
    cross_source_rate: float
    cross_source_details: List[Dict]
    cross_source_note: str
    # Sign-family typological
    n_triples_total: int
    n_family_consistent: int
    n_family_inconsistent: int
    family_consistency_rate: float
    inconsistent_triples: List[Dict]
    family_details: List[Dict]
    # Null comparison
    null_n_trials: int
    null_mean_consistency: float
    null_std_consistency: float
    real_consistency: float
    null_z_score: float
    # Summary
    gate_passed: bool
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def _parse_syllable(syllable: str) -> Tuple[str, str]:
    """Split a CV syllable into onset (consonant) and nucleus (vowel).

    Handles: 'de' → ('d', 'e'), 'se' → ('s', 'e'), 'a' → ('', 'a'),
             'sc' + vowel like 'sca' → ('sc', 'a').
    """
    if not syllable:
        return ('', '')
    vowels = set('aeiou')
    # Find first vowel position
    for i, ch in enumerate(syllable):
        if ch in vowels:
            return (syllable[:i], syllable[i:])
    # No vowel found — treat whole thing as onset
    return (syllable, '')


def _compute_cross_source_consistency(
    cribs: List[Dict],
    assignment: Dict[str, str],
) -> Tuple[List[CrossSourceTest], str]:
    """Test cross-source consistency: do independent pipelines agree?

    All Phase 14 hits use the same assignment table, so intra-Phase-14
    consistency is trivially 100%.  The meaningful test is whether Phase 19.8
    and Phase 26 agree with Phase 14.
    """
    # Build triple → {source → syllable} from cribs
    triple_sources: Dict[str, Dict[str, str]] = defaultdict(dict)

    for crib in cribs:
        sources = crib.get('sources', [])
        for alignment in crib.get('alignments', []):
            triple = alignment.get('triple_key', '')
            syllable = alignment.get('syllable', '')
            if not triple or not syllable:
                continue
            for src in sources:
                # Normalize to pipeline level
                if src.startswith('phase19'):
                    pipeline = 'phase19'
                elif src.startswith('phase26'):
                    pipeline = 'phase26'
                else:
                    pipeline = src
                triple_sources[triple][pipeline] = syllable

    # Only test triples with ≥2 distinct pipeline sources
    tests: List[CrossSourceTest] = []
    for triple, src_map in sorted(triple_sources.items()):
        if len(src_map) < 2:
            continue
        syllables = list(src_map.values())
        all_agree = len(set(syllables)) == 1
        tests.append(CrossSourceTest(
            triple=triple,
            syllable_by_source=dict(src_map),
            n_sources=len(src_map),
            all_agree=all_agree,
            agreed_syllable=syllables[0] if all_agree else None,
        ))

    n_consistent = sum(1 for t in tests if t.all_agree)
    note = (
        f"Cross-source consistency: {n_consistent}/{len(tests)} triples agree "
        f"across independent pipelines. "
        f"Note: all 18 Phase 14 hits use the same assignment table and are "
        f"trivially consistent. Only triples tested across Phase 14 + "
        f"Phase 19.8/26 are meaningful."
    )
    return tests, note


def _compute_family_consistency(
    assignment: Dict[str, str],
) -> List[FamilyConsistencyTest]:
    """Test each triple assignment against PHONEME_PLACE/NUCLEUS_MAP."""
    results: List[FamilyConsistencyTest] = []

    for triple, syllable in sorted(assignment.items()):
        parts = triple.split(',')
        if len(parts) != 3:
            continue
        first_stroke, last_stroke, _ = parts

        onset, nucleus = _parse_syllable(syllable)

        # Check onset against PHONEME_PLACE_MAP
        allowed_onsets = PHONEME_PLACE_MAP.get(first_stroke, [])
        onset_ok = (onset in allowed_onsets) or (onset == '' and nucleus != '')

        # Check nucleus against PHONEME_NUCLEUS_MAP
        allowed_nuclei = PHONEME_NUCLEUS_MAP.get(last_stroke, [])
        # Check if any character of nucleus is in allowed list
        nucleus_ok = False
        if nucleus:
            for ch in nucleus:
                if ch in allowed_nuclei:
                    nucleus_ok = True
                    break
        else:
            nucleus_ok = True  # pure consonant syllable — unusual but not wrong

        is_consistent = onset_ok and nucleus_ok

        results.append(FamilyConsistencyTest(
            triple=triple,
            syllable=syllable,
            first_stroke=first_stroke,
            last_stroke=last_stroke,
            onset_char=onset,
            nucleus_char=nucleus,
            onset_in_map=onset_ok,
            nucleus_in_map=nucleus_ok,
            is_consistent=is_consistent,
        ))

    return results


def _null_consistency_test(
    assignment: Dict[str, str],
    n_trials: int = 1000,
    seed: int = 42,
) -> Tuple[float, float, float, float]:
    """Permutation test: randomly reassign syllables to triples and measure
    family consistency rate.

    Returns: (null_mean, null_std, real_rate, z_score)
    """
    rng = random.Random(seed)

    # Real consistency
    real_tests = _compute_family_consistency(assignment)
    real_rate = (sum(1 for t in real_tests if t.is_consistent)
                 / len(real_tests) if real_tests else 0.0)

    # Null distribution: randomly permute syllable values across triples
    triples = list(assignment.keys())
    syllables = list(assignment.values())
    null_rates: List[float] = []

    for _ in range(n_trials):
        shuffled = syllables[:]
        rng.shuffle(shuffled)
        perm_assignment = dict(zip(triples, shuffled))
        perm_tests = _compute_family_consistency(perm_assignment)
        rate = (sum(1 for t in perm_tests if t.is_consistent)
                / len(perm_tests) if perm_tests else 0.0)
        null_rates.append(rate)

    null_mean = sum(null_rates) / len(null_rates) if null_rates else 0.0
    null_var = (sum((r - null_mean) ** 2 for r in null_rates)
                / len(null_rates) if null_rates else 0.0)
    null_std = null_var ** 0.5

    z_score = ((real_rate - null_mean) / null_std) if null_std > 0 else 0.0

    return null_mean, null_std, real_rate, z_score


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_consistency_check() -> None:
    """Step 28.2: Internal consistency test for crib assignments."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE 28.2: Internal Consistency Test")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load crib extraction ──
    print("\n  1. Loading crib extraction …")
    crib_path = os.path.join(rd, 'crib_extraction.json')
    if not os.path.exists(crib_path):
        print("  [SKIP] crib_extraction.json not found — run crib-extract first")
        return
    with open(crib_path) as f:
        crib_data = json.load(f)
    cribs = crib_data.get('cribs', [])
    print(f"     {len(cribs)} cribs loaded")

    # ── 2. Load Phase 14 assignment ──
    print("\n  2. Loading Phase 14 assignment …")
    csp_path = os.path.join(rd, 'feature_csp.json')
    if not os.path.exists(csp_path):
        print("  [SKIP] feature_csp.json not found")
        return
    with open(csp_path) as f:
        csp_data = json.load(f)
    assignment = (csp_data.get('language_results', {})
                  .get('latin', {}).get('best_assignment', {}))
    if not assignment:
        assignment = csp_data.get('best_assignment', {})
    print(f"     {len(assignment)} triple assignments")

    # ── 3. Cross-source consistency ──
    print("\n  3. Cross-source consistency test …")
    cross_tests, cross_note = _compute_cross_source_consistency(cribs, assignment)
    n_cross_tests = len(cross_tests)
    n_cross_consistent = sum(1 for t in cross_tests if t.all_agree)
    cross_rate = n_cross_consistent / n_cross_tests if n_cross_tests > 0 else 1.0

    print(f"     Testable triples (≥2 independent sources): {n_cross_tests}")
    print(f"     Consistent: {n_cross_consistent}/{n_cross_tests} "
          f"({cross_rate:.0%})")
    for t in cross_tests:
        status = "✓" if t.all_agree else "✗"
        print(f"       {status} {t.triple}: {t.syllable_by_source}")

    # ── 4. Family typological consistency ──
    print("\n  4. Sign-family typological consistency …")
    family_tests = _compute_family_consistency(assignment)
    n_family_consistent = sum(1 for t in family_tests if t.is_consistent)
    n_family_total = len(family_tests)
    family_rate = n_family_consistent / n_family_total if n_family_total else 0.0

    print(f"     Consistent: {n_family_consistent}/{n_family_total} "
          f"({family_rate:.0%})")
    inconsistent = [t for t in family_tests if not t.is_consistent]
    for t in inconsistent:
        reasons = []
        if not t.onset_in_map:
            allowed = PHONEME_PLACE_MAP.get(t.first_stroke, [])
            reasons.append(f"onset '{t.onset_char}' not in {allowed}")
        if not t.nucleus_in_map:
            allowed = PHONEME_NUCLEUS_MAP.get(t.last_stroke, [])
            reasons.append(f"nucleus '{t.nucleus_char}' not in {allowed}")
        print(f"       ✗ {t.triple} = '{t.syllable}': {'; '.join(reasons)}")

    # ── 5. Null permutation test ──
    print("\n  5. Null permutation test (1000 trials) …")
    null_mean, null_std, real_rate, z_score = _null_consistency_test(assignment)
    print(f"     Real family consistency: {real_rate:.3f}")
    print(f"     Null mean: {null_mean:.3f} ± {null_std:.3f}")
    print(f"     Z-score: {z_score:.2f}")

    # ── 6. Gate ──
    gate_passed = family_rate >= 0.90 and (n_cross_tests == 0 or cross_rate >= 0.50)
    verdict_parts = [
        f"Family consistency {family_rate:.0%} ({n_family_consistent}/{n_family_total})",
        f"Cross-source {cross_rate:.0%} ({n_cross_consistent}/{n_cross_tests})",
        f"Null z={z_score:.1f}",
    ]
    verdict = (
        f"PASS: {'; '.join(verdict_parts)}"
        if gate_passed
        else f"FAIL: {'; '.join(verdict_parts)}"
    )
    print(f"\n  Gate: {'PASS' if gate_passed else 'FAIL'}")
    print(f"  {verdict}")

    # ── 7. Save ──
    result = ConsistencyResult(
        n_cross_source_tests=n_cross_tests,
        n_cross_consistent=n_cross_consistent,
        n_cross_inconsistent=n_cross_tests - n_cross_consistent,
        cross_source_rate=round(cross_rate, 4),
        cross_source_details=[_convert(asdict(t)) for t in cross_tests],
        cross_source_note=cross_note,
        n_triples_total=n_family_total,
        n_family_consistent=n_family_consistent,
        n_family_inconsistent=n_family_total - n_family_consistent,
        family_consistency_rate=round(family_rate, 4),
        inconsistent_triples=[_convert(asdict(t)) for t in inconsistent],
        family_details=[_convert(asdict(t)) for t in family_tests],
        null_n_trials=1000,
        null_mean_consistency=round(null_mean, 4),
        null_std_consistency=round(null_std, 4),
        real_consistency=round(real_rate, 4),
        null_z_score=round(z_score, 2),
        gate_passed=gate_passed,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'crib_consistency.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)
    print(f"\n  → {out_path}")
