"""
Step 24.16 – Phase 24 Integration
==================================
Combine all Part A (error correction) and Part B (exploratory) results
into a final assessment.

Dependency chain:
    All Phase 24 results (24.1-24.15)
        → phase24_integrate.json (this step)
"""

import json
import os
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

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


def _load_json(path: str):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Result file manifest
# ---------------------------------------------------------------------------

RESULT_FILES = {
    # Part A
    'sensitivity': 'triple_sensitivity.json',
    'error_candidates': 'error_candidates.json',
    'targeted_swap': 'targeted_swap.json',
    'bigram_filter': 'bigram_filter.json',
    'corrected_table': 'corrected_table.json',
    'corrected_decode': 'corrected_decode.json',
    'corrected_readability': 'corrected_readability.json',
    # Part B
    'word_boundary': 'word_boundary.json',
    'ligature_test': 'ligature_test.json',
    'directionality': 'directionality.json',
    'known_text': 'known_text_search.json',
    'folio_isolation': 'folio_isolation.json',
    'cross_section': 'cross_section_transfer.json',
    'reverse_engineer': 'reverse_engineering.json',
    'token_grammar': 'token_grammar.json',
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PartAVerdict:
    n_probably_wrong: int
    n_swaps_accepted: int
    bigram_filter_passed: bool
    final_dict_hit: float
    final_selectivity: float
    final_bigram: float
    improvement_over_phase16: float
    net_improvement: str  # "improved", "no_change", "degraded"


@dataclass
class PartBDiscovery:
    step_name: str
    key_finding: str
    actionable: bool
    detail: str


@dataclass
class Phase24IntegrateResult:
    timestamp: str
    # Part A
    part_a_verdict: Dict
    part_a_available: bool
    # Part B
    part_b_discoveries: List[Dict]
    n_actionable_findings: int
    part_b_available_count: int
    # Progression
    progression: Dict[str, Dict]
    # Best table
    best_table: str  # "corrected" or "phase16"
    best_dict_hit: float
    best_selectivity: float
    # Readiness
    decipherment_readiness: float  # 0-1 scale
    readiness_components: Dict[str, float]
    # Summary
    n_results_loaded: int
    n_results_missing: int
    missing_results: List[str]
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Part A: Error Correction Verdict
# ---------------------------------------------------------------------------

def _build_part_a_verdict(results: Dict[str, Optional[Dict]]) -> Tuple[PartAVerdict, bool]:
    """Summarize Part A (error correction) results."""
    sensitivity = results.get('sensitivity')
    targeted_swap = results.get('targeted_swap')
    bigram_filter = results.get('bigram_filter')
    corrected_readability = results.get('corrected_readability')
    corrected_decode = results.get('corrected_decode')
    corrected_table = results.get('corrected_table')

    # Check if enough Part A data is available
    part_a_available = any(r is not None for r in [
        sensitivity, targeted_swap, corrected_readability,
    ])

    # Probably-wrong triples (from sensitivity analysis 24.1)
    n_probably_wrong = 0
    if sensitivity is not None:
        n_probably_wrong = sensitivity.get('n_probably_wrong', 0)

    # Swaps accepted (from targeted swap 24.3)
    n_swaps_accepted = 0
    if targeted_swap is not None:
        n_swaps_accepted = targeted_swap.get('n_accepted', 0)

    # Bigram filter (from 24.4 or corrected_table 24.5)
    bigram_filter_passed = True  # default if no filter ran
    if bigram_filter is not None:
        bigram_filter_passed = bigram_filter.get('gate_passed', False)
    elif corrected_table is not None:
        bigram_filter_passed = corrected_table.get('bigram_filter_passed', True)

    # Final readability metrics (from corrected_readability 24.7)
    final_dict_hit = 0.516  # Phase 16 baseline
    final_selectivity = 3.38
    final_bigram = 0.0
    if corrected_readability is not None:
        final_bigram = corrected_readability.get('bigram_plausibility', 0.0)
        # dict_hit comes from corrected_decode
        if corrected_decode is not None:
            final_dict_hit = corrected_decode.get('dict_hit_rate', 0.516)
            final_selectivity = corrected_decode.get('selectivity', 3.38)
    elif corrected_decode is not None:
        final_dict_hit = corrected_decode.get('dict_hit_rate', 0.516)
        final_selectivity = corrected_decode.get('selectivity', 3.38)
    elif targeted_swap is not None:
        final_dict_hit = targeted_swap.get('final_dict_hit', 0.516)
        final_bigram = targeted_swap.get('final_bigram', 0.0)

    # Net improvement over Phase 16
    improvement = final_dict_hit - 0.516
    if improvement > 0.005:
        net_improvement = "improved"
    elif improvement < -0.005:
        net_improvement = "degraded"
    else:
        net_improvement = "no_change"

    verdict = PartAVerdict(
        n_probably_wrong=n_probably_wrong,
        n_swaps_accepted=n_swaps_accepted,
        bigram_filter_passed=bigram_filter_passed,
        final_dict_hit=round(final_dict_hit, 4),
        final_selectivity=round(final_selectivity, 4),
        final_bigram=round(final_bigram, 6),
        improvement_over_phase16=round(improvement, 4),
        net_improvement=net_improvement,
    )
    return verdict, part_a_available


# ---------------------------------------------------------------------------
# Part B: Exploratory Discoveries
# ---------------------------------------------------------------------------

def _extract_part_b_discoveries(results: Dict[str, Optional[Dict]]) -> List[PartBDiscovery]:
    """Extract key findings from each Part B analysis."""
    discoveries: List[PartBDiscovery] = []

    # 24.8 Word boundaries
    wb = results.get('word_boundary')
    if wb is not None:
        are_boundaries = wb.get('boundaries_are_word_boundaries', True)
        false_rate = wb.get('estimated_false_boundary_rate', 0.0)
        concat = wb.get('concatenation', {})
        concat_rate = concat.get('concatenation_rate', 0.0) if isinstance(concat, dict) else 0.0
        actionable = not are_boundaries and false_rate > 0.05
        finding = (f"Spaces are {'true' if are_boundaries else 'false'} word boundaries "
                   f"(concat_rate={concat_rate:.3f}, false_boundary={false_rate:.3f})")
        discoveries.append(PartBDiscovery(
            step_name='word_boundary',
            key_finding=finding,
            actionable=actionable,
            detail=wb.get('verdict', ''),
        ))

    # 24.9 Ligature test
    lt = results.get('ligature_test')
    if lt is not None:
        n_strong = lt.get('n_strong', 0)
        supported = lt.get('ligature_hypothesis_supported', False)
        merges = lt.get('recommended_merges', [])
        actionable = supported and n_strong > 0
        finding = (f"{n_strong} strong ligatures detected; "
                   f"hypothesis {'supported' if supported else 'not supported'}; "
                   f"merges={merges}")
        discoveries.append(PartBDiscovery(
            step_name='ligature_test',
            key_finding=finding,
            actionable=actionable,
            detail=lt.get('verdict', ''),
        ))

    # 24.10 Directionality
    dr = results.get('directionality')
    if dr is not None:
        any_non_fwd = dr.get('any_non_forward', False)
        non_fwd_sections = dr.get('non_forward_sections', [])
        best_dir = dr.get('corpus_best_direction', 'forward')
        actionable = any_non_fwd
        finding = (f"Best direction: {best_dir}; "
                   f"{'non-forward sections: ' + str(non_fwd_sections) if any_non_fwd else 'all sections forward'}")
        discoveries.append(PartBDiscovery(
            step_name='directionality',
            key_finding=finding,
            actionable=actionable,
            detail=dr.get('verdict', ''),
        ))

    # 24.11 Known-text crib search
    kt = results.get('known_text')
    if kt is not None:
        n_medical = kt.get('n_medical_matches', 0)
        n_null = kt.get('n_null_matches', 0)
        is_medical = kt.get('is_medical', False)
        ratio = kt.get('medical_vs_null_ratio', 0.0)
        n_corrections = kt.get('n_corrections', 0)
        actionable = is_medical and n_corrections > 0
        finding = (f"{n_medical} medical matches vs {n_null} null; "
                   f"ratio={ratio:.2f}; medical={'YES' if is_medical else 'NO'}; "
                   f"{n_corrections} implied corrections")
        discoveries.append(PartBDiscovery(
            step_name='known_text',
            key_finding=finding,
            actionable=actionable,
            detail=kt.get('verdict', ''),
        ))

    # 24.12 Folio isolation
    fi = results.get('folio_isolation')
    if fi is not None:
        selected = fi.get('selected_folio', '?')
        dict_hit = fi.get('dict_hit_rate', 0.0)
        max_consec = fi.get('max_consecutive_hits', 0)
        fragments = fi.get('coherent_fragments', [])
        n_fragments = len(fragments) if isinstance(fragments, list) else 0
        actionable = n_fragments > 0 and max_consec >= 3
        finding = (f"Folio {selected}: dict_hit={dict_hit:.1%}, "
                   f"max_consecutive={max_consec}, "
                   f"{n_fragments} coherent fragment(s)")
        discoveries.append(PartBDiscovery(
            step_name='folio_isolation',
            key_finding=finding,
            actionable=actionable,
            detail=fi.get('verdict', ''),
        ))

    # 24.13 Cross-section transfer
    cs = results.get('cross_section')
    if cs is not None:
        uniform = cs.get('encoding_is_uniform', True)
        transfer_ratio = cs.get('transfer_ratio', 0.0)
        n_sections = cs.get('n_sections', 0)
        clusters = cs.get('clusters', [])
        actionable = not uniform
        finding = (f"{'Uniform' if uniform else 'Section-specific'} encoding "
                   f"(transfer_ratio={transfer_ratio:.3f}, "
                   f"{n_sections} sections, {len(clusters)} cluster(s))")
        discoveries.append(PartBDiscovery(
            step_name='cross_section',
            key_finding=finding,
            actionable=actionable,
            detail=cs.get('verdict', ''),
        ))

    # 24.14 Reverse engineering
    re_ = results.get('reverse_engineer')
    if re_ is not None:
        n_assigned = re_.get('n_chars_assigned', 0)
        n_consistent = re_.get('n_chars_consistent', 0)
        n_contradictory = re_.get('n_chars_contradictory', 0)
        n_agrees = re_.get('n_agrees_with_phase16', 0)
        n_disagrees = re_.get('n_disagrees_with_phase16', 0)
        n_new_words = re_.get('n_new_words_found', 0)
        actionable = n_assigned > 0 and n_contradictory == 0
        finding = (f"{n_assigned} chars assigned ({n_consistent} consistent, "
                   f"{n_contradictory} contradictory); "
                   f"Phase 16 agreement: {n_agrees}/{n_agrees + n_disagrees}; "
                   f"{n_new_words} new words bootstrapped")
        discoveries.append(PartBDiscovery(
            step_name='reverse_engineer',
            key_finding=finding,
            actionable=actionable,
            detail=re_.get('verdict', ''),
        ))

    # 24.15 Token grammar
    tg = results.get('token_grammar')
    if tg is not None:
        n_violations = tg.get('n_violations', 0)
        violation_rate = tg.get('violation_rate', 0.0)
        n_corrections = tg.get('n_corrections_proposed', 0)
        gallows_initial = tg.get('gallows_initial_rate', 0.0)
        actionable = n_corrections > 0
        finding = (f"{n_violations} Phase 16 violations ({violation_rate:.1%}); "
                   f"{n_corrections} corrections proposed; "
                   f"gallows initial_rate={gallows_initial:.3f}")
        discoveries.append(PartBDiscovery(
            step_name='token_grammar',
            key_finding=finding,
            actionable=actionable,
            detail=tg.get('verdict', ''),
        ))

    return discoveries


# ---------------------------------------------------------------------------
# Progression Table
# ---------------------------------------------------------------------------

def _build_progression(
    final_dict_hit: float,
    final_selectivity: float,
) -> Dict[str, Dict]:
    """Build the full progression table from Phase 11 to Phase 24."""
    return {
        'phase11': {
            'dict_hit': 0.111,
            'selectivity': 1.92,
            'description': 'CV grid CSP',
        },
        'phase13': {
            'dict_hit': 0.1143,
            'selectivity': 1.86,
            'description': 'Context rules',
        },
        'phase14': {
            'dict_hit': 0.194,
            'selectivity': 3.00,
            'description': 'Sub-cell features',
        },
        'phase15': {
            'dict_hit': 0.354,
            'selectivity': 2.55,
            'description': 'Dict expansion + articulatory',
        },
        'phase16': {
            'dict_hit': 0.516,
            'selectivity': 3.38,
            'description': 'Modifier detection (R3 combined)',
        },
        'phase24': {
            'dict_hit': round(final_dict_hit, 4),
            'selectivity': round(final_selectivity, 4),
            'description': 'Error correction + exploratory',
        },
    }


# ---------------------------------------------------------------------------
# Decipherment Readiness Assessment
# ---------------------------------------------------------------------------

def _compute_readiness(
    part_a: PartAVerdict,
    discoveries: List[PartBDiscovery],
    results: Dict[str, Optional[Dict]],
) -> Tuple[float, Dict[str, float]]:
    """
    Score decipherment readiness on a 0-1 scale.

    Base from Phase 19's 0.55 score, add increments:
    - Dict-hit improvement over Phase 16: +0.1 per percentage point
    - Bigram plausibility improvement: +0.2 if improved
    - Confirmed character assignments (reverse engineering): +0.05 per assignment
    - Actionable Part B findings: +0.05 per finding
    - Coherent fragments found: +0.1 per fragment
    """
    components: Dict[str, float] = {}
    base = 0.55
    components['base_phase19'] = base

    # Dict-hit improvement: +0.1 per percentage point above Phase 16
    dict_hit_delta = part_a.improvement_over_phase16
    dict_hit_bonus = max(0.0, dict_hit_delta * 10.0)  # 0.1 per ppt
    dict_hit_bonus = min(dict_hit_bonus, 0.20)  # cap at 0.20
    components['dict_hit_improvement'] = round(dict_hit_bonus, 4)

    # Bigram plausibility: +0.2 if improved (nonzero and above null)
    bigram_bonus = 0.0
    if part_a.final_bigram > 0.0 and part_a.bigram_filter_passed:
        bigram_bonus = 0.2
    components['bigram_plausibility'] = round(bigram_bonus, 4)

    # Confirmed character assignments from reverse engineering
    re_data = results.get('reverse_engineer')
    n_chars_assigned = 0
    if re_data is not None:
        n_chars_assigned = re_data.get('n_chars_assigned', 0)
    char_bonus = min(n_chars_assigned * 0.05, 0.15)  # cap at 0.15
    components['confirmed_char_assignments'] = round(char_bonus, 4)

    # Actionable Part B findings
    n_actionable = sum(1 for d in discoveries if d.actionable)
    actionable_bonus = min(n_actionable * 0.05, 0.10)  # cap at 0.10
    components['actionable_findings'] = round(actionable_bonus, 4)

    # Coherent fragments found (from folio isolation)
    fi = results.get('folio_isolation')
    n_fragments = 0
    if fi is not None:
        fragments = fi.get('coherent_fragments', [])
        n_fragments = len(fragments) if isinstance(fragments, list) else 0
    fragment_bonus = min(n_fragments * 0.1, 0.10)  # cap at 0.10
    components['coherent_fragments'] = round(fragment_bonus, 4)

    total = base + dict_hit_bonus + bigram_bonus + char_bonus + actionable_bonus + fragment_bonus
    total = min(total, 1.0)

    return round(total, 4), components


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_phase24_integrate() -> None:
    """Step 24.16: Phase 24 Integration."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 24.16: Phase 24 Integration")
    print("=" * 70)

    rdir = _results_dir()

    # ------------------------------------------------------------------
    # 1. Load all results
    # ------------------------------------------------------------------
    print("\n  1. Loading Phase 24 results ...")

    results: Dict[str, Optional[Dict]] = {}
    loaded_names: List[str] = []
    missing_names: List[str] = []

    for key, filename in RESULT_FILES.items():
        path = str(rdir / filename)
        data = _load_json(path)
        results[key] = data
        if data is not None:
            loaded_names.append(key)
            print(f"     [OK]   {filename}")
        else:
            missing_names.append(key)
            print(f"     [MISS] {filename}")

    n_loaded = len(loaded_names)
    n_missing = len(missing_names)
    print(f"\n     Loaded {n_loaded}/{len(RESULT_FILES)} result files "
          f"({n_missing} missing)")

    # ------------------------------------------------------------------
    # 2. Part A Verdict
    # ------------------------------------------------------------------
    print("\n  2. Part A: Error Correction Verdict ...")

    part_a, part_a_available = _build_part_a_verdict(results)

    print(f"     Probably-wrong triples:    {part_a.n_probably_wrong}")
    print(f"     Swaps accepted:            {part_a.n_swaps_accepted}")
    print(f"     Bigram filter passed:      {part_a.bigram_filter_passed}")
    print(f"     Final dict-hit:            {part_a.final_dict_hit:.1%}")
    print(f"     Final selectivity:         {part_a.final_selectivity:.2f}x")
    print(f"     Final bigram plausibility: {part_a.final_bigram:.6f}")
    print(f"     Improvement over Phase 16: {part_a.improvement_over_phase16:+.4f}")
    print(f"     Net result:                {part_a.net_improvement.upper()}")

    # ------------------------------------------------------------------
    # 3. Part B Discoveries
    # ------------------------------------------------------------------
    print("\n  3. Part B: Exploratory Discoveries ...")

    discoveries = _extract_part_b_discoveries(results)
    n_actionable = sum(1 for d in discoveries if d.actionable)
    part_b_available = sum(1 for key in [
        'word_boundary', 'ligature_test', 'directionality', 'known_text',
        'folio_isolation', 'cross_section', 'reverse_engineer', 'token_grammar',
    ] if results.get(key) is not None)

    for d in discoveries:
        marker = ">>>" if d.actionable else "   "
        print(f"     {marker} {d.step_name}: {d.key_finding}")

    print(f"\n     {len(discoveries)} analyses completed, "
          f"{n_actionable} actionable findings")

    # ------------------------------------------------------------------
    # 4. Progression Table
    # ------------------------------------------------------------------
    print("\n  4. Progression Table:")

    progression = _build_progression(part_a.final_dict_hit, part_a.final_selectivity)

    print(f"     {'Phase':<10} {'Dict-Hit':>10} {'Selectivity':>13} {'Description'}")
    print(f"     {'-'*10} {'-'*10} {'-'*13} {'-'*35}")
    for phase_name, info in progression.items():
        dh = info['dict_hit']
        sel = info['selectivity']
        desc = info['description']
        print(f"     {phase_name:<10} {dh:>9.1%} {sel:>12.2f}x  {desc}")

    # ------------------------------------------------------------------
    # 5. Decipherment Readiness Assessment
    # ------------------------------------------------------------------
    print("\n  5. Decipherment Readiness Assessment ...")

    readiness, readiness_components = _compute_readiness(
        part_a, discoveries, results,
    )

    print(f"     Score: {readiness:.2f} / 1.00")
    for comp_name, comp_val in readiness_components.items():
        if comp_val > 0:
            print(f"       + {comp_name}: {comp_val:.4f}")
        else:
            print(f"         {comp_name}: {comp_val:.4f}")

    # ------------------------------------------------------------------
    # 6. Best Available Table
    # ------------------------------------------------------------------
    print("\n  6. Best Available Table ...")

    corrected_table = results.get('corrected_table')
    if corrected_table is not None:
        recommended = corrected_table.get('recommended_table', 'phase16')
    elif part_a.bigram_filter_passed and part_a.n_swaps_accepted > 0:
        recommended = 'corrected'
    else:
        recommended = 'phase16'

    best_dict_hit = part_a.final_dict_hit
    best_selectivity = part_a.final_selectivity

    # If Phase 16 is recommended, use Phase 16 metrics
    if recommended == 'phase16':
        best_dict_hit = 0.516
        best_selectivity = 3.38

    print(f"     Recommended: {recommended}")
    print(f"     Dict-hit:    {best_dict_hit:.1%}")
    print(f"     Selectivity: {best_selectivity:.2f}x")

    # ------------------------------------------------------------------
    # 7. Overall Verdict
    # ------------------------------------------------------------------
    if part_a.net_improvement == 'improved' and part_a.bigram_filter_passed:
        verdict = (
            f"IMPROVED: Phase 24 error correction raised dict-hit from "
            f"51.6% to {part_a.final_dict_hit:.1%} "
            f"({part_a.improvement_over_phase16:+.1%}). "
            f"{n_actionable} actionable Part B findings. "
            f"Readiness: {readiness:.2f}."
        )
    elif part_a.net_improvement == 'no_change':
        verdict = (
            f"NO CHANGE: Error correction produced no improvement over "
            f"Phase 16 (51.6%). {part_a.n_swaps_accepted} swaps accepted "
            f"but net effect is zero. "
            f"{n_actionable} actionable Part B findings. "
            f"Readiness: {readiness:.2f}."
        )
    else:
        verdict = (
            f"DEGRADED: Error correction reduced dict-hit from 51.6% to "
            f"{part_a.final_dict_hit:.1%}. Recommending Phase 16 table. "
            f"{n_actionable} actionable Part B findings. "
            f"Readiness: {readiness:.2f}."
        )

    # ------------------------------------------------------------------
    # 8. Build and save result
    # ------------------------------------------------------------------
    result = Phase24IntegrateResult(
        timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
        part_a_verdict=_convert(asdict(part_a)),
        part_a_available=part_a_available,
        part_b_discoveries=[_convert(asdict(d)) for d in discoveries],
        n_actionable_findings=n_actionable,
        part_b_available_count=part_b_available,
        progression=_convert(progression),
        best_table=recommended,
        best_dict_hit=round(best_dict_hit, 4),
        best_selectivity=round(best_selectivity, 4),
        decipherment_readiness=readiness,
        readiness_components=readiness_components,
        n_results_loaded=n_loaded,
        n_results_missing=n_missing,
        missing_results=missing_names,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = rdir / "phase24_integrate.json"
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t0

    print(f"\n{'='*70}")
    print(f"PHASE 24 VERDICT")
    print(f"{'='*70}")
    print(f"  {verdict}")
    print(f"\n  Part A (Error Correction):")
    print(f"    Probably-wrong triples:    {part_a.n_probably_wrong}")
    print(f"    Swaps accepted:            {part_a.n_swaps_accepted}")
    print(f"    Final dict-hit:            {part_a.final_dict_hit:.1%}")
    print(f"    Net improvement:           {part_a.net_improvement}")
    print(f"    Bigram filter:             {'PASS' if part_a.bigram_filter_passed else 'FAIL'}")
    print(f"\n  Part B (Exploratory):")
    print(f"    Analyses completed:        {len(discoveries)}")
    print(f"    Actionable findings:       {n_actionable}")
    for d in discoveries:
        if d.actionable:
            print(f"      -> {d.step_name}: {d.key_finding}")
    print(f"\n  Recommended table:           {recommended}")
    print(f"  Best dict-hit:               {best_dict_hit:.1%}")
    print(f"  Decipherment readiness:      {readiness:.2f}")
    print(f"\n  Progression:")
    for phase_name, info in progression.items():
        dh = info['dict_hit']
        sel = info['selectivity']
        desc = info['description']
        print(f"    {phase_name:<10} {dh:>7.1%} ({sel:.2f}x)  {desc}")
    print(f"\n  Saved -> {out_path}")
    print(f"  ({elapsed:.1f}s)")
