"""
Phase 11.5.4 – Verb constraint integration
============================================
Uses Phase 9 verb-to-imperative assignments as additional CSP constraints.

Step 4a: Build VerbConstraint objects from verb_identification.json.
          Only stems where cell count == syllable count (1-3 expected).
Step 4b: Apply hard constraints to narrow cell domains for length-matched pairs.
Step 4c: Run CSP with verb-aware beam search (VERB_BONUS + soft scoring).
Step 4d: Check consistency between verb constraints and illustration anchors.

Decision gate:
  dict_hit_after >= 0.15   OR   verb_matches >= 5
"""

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_cell_lookup,
    load_corpus,
    token_to_grid_cells,
    tokenize_eva_chars,
)
from voynich.core.reference import (
    LATIN_IMPERATIVE_SYLLABIFICATIONS,
    load_reference_corpus,
)
from voynich.core.stats import build_ngram_lm
from voynich.phases.csp_constraints import (
    AnchorConstraint,
    VerbConstraint,
    build_anchor_constraints,
    build_phoneme_inventory,
    apply_verb_constraints,
    build_verb_constraints,
    score_verb_consistency,
    score_dict_hit_rate,
)
from voynich.phases.csp_solver import (
    _convert,
    ac3_propagate,
    beam_search,
    build_csp_variables,
    decode_token,
    initialise_domains,
    score_assignment_full,
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class VerbConstraintResult:
    """Full Phase 11.5.4 output."""
    n_verb_constraints: int
    n_hard_constraints: int
    n_soft_constraints: int
    constraints_applied: List[Dict]
    constrained_cells: List[str]
    dict_hit_before: float
    dict_hit_after: float
    anchor_matches_before: int
    anchor_matches_after: int
    verb_matches: int
    best_assignment: Dict[str, str]
    illustration_consistency: Dict
    gate_passed: bool
    verdict: str


# ---------------------------------------------------------------------------
# Illustration consistency check
# ---------------------------------------------------------------------------

def check_illustration_consistency(
    assignment: Dict[str, str],
    verb_constraints: List[VerbConstraint],
    rosetta_data: Dict,
    eva_to_cell: Dict[str, str],
    cv_labels: Dict,
) -> Dict:
    """Check for conflicts between verb constraints and illustration anchors.

    For each Rosetta anchor, decodes the dominant stem and checks whether
    the cell assignment contradicts a verb constraint.  A conflict is when
    the same cell is simultaneously required to be X (by a verb constraint)
    and to produce a decoded string close to Y (by an anchor).

    Returns a summary dict with per-folio consistency information.
    """
    from voynich.core.stats import syllabify_latin

    # Build verb cell->target mapping
    verb_cell_targets: Dict[str, str] = {}
    for vc in verb_constraints:
        for cell_key, target_syl in zip(vc.voynich_cells, vc.latin_syllables):
            verb_cell_targets[cell_key] = target_syl

    folio_results: List[Dict] = []
    selected = rosetta_data.get('selected_rosetta_folios', [])
    conflicts: int = 0

    for folio_info in rosetta_data.get('folio_scores', []):
        folio = folio_info.get('folio', '')
        if folio not in selected:
            continue

        stem = folio_info.get('dominant_stem', '')
        plant = folio_info.get('medieval_name', '')
        if not stem:
            continue

        # Decode with current assignment
        decoded = decode_token(stem, assignment, eva_to_cell)

        # Get cells for this stem
        stem_cells = token_to_grid_cells(stem, eva_to_cell)

        # Check for conflicts with verb constraints
        stem_conflicts: List[Dict] = []
        for cell_key in stem_cells:
            if cell_key in verb_cell_targets:
                verb_requires = verb_cell_targets[cell_key]
                actual = assignment.get(cell_key, '?')
                if actual != verb_requires:
                    stem_conflicts.append({
                        'cell': cell_key,
                        'verb_requires': verb_requires,
                        'current_value': actual,
                    })
                    conflicts += 1

        folio_results.append({
            'folio': folio,
            'plant': plant,
            'stem': stem,
            'decoded': decoded,
            'conflicts': stem_conflicts,
            'n_conflicts': len(stem_conflicts),
        })

    return {
        'n_folios_checked': len(folio_results),
        'n_total_conflicts': conflicts,
        'folio_details': folio_results,
        'globally_consistent': conflicts == 0,
    }


# ---------------------------------------------------------------------------
# Verb-constrained CSP run
# ---------------------------------------------------------------------------

def run_verb_constrained_csp(
    corpus_tokens: List[str],
    ref_corpus: Any,
    cv_labels: Dict,
    rosetta_data: Dict,
    eva_to_cell: Dict[str, str],
    verb_data: Dict,
    base_assignment: Dict[str, str],
    base_dict_hit: float,
    base_anchor_matches: int,
    relaxation_level: int = 0,
    best_inherent_vowel: Optional[str] = None,
    beam_width: int = 50,
    max_solutions: int = 20,
) -> Tuple[VerbConstraintResult, List[VerbConstraint]]:
    """Run the CSP with verb constraints integrated.

    1. Build VerbConstraint objects (length-matched only).
    2. Apply hard constraints to narrow cell domains.
    3. Prioritise verb-constrained cells in beam search ordering.
    4. Run beam_search with VERB_BONUS scoring.
    5. Check illustration consistency.
    """
    ref_tokens = ref_corpus.get_combined_tokens('latin')
    if not ref_tokens:
        ref_tokens = ref_corpus.get_combined_tokens(ref_corpus.languages[0])
    ref_word_set = set(ref_tokens[:50000])
    lm = build_ngram_lm(ref_tokens[:10000], order=3, smoothing=0.01)
    anchors = build_anchor_constraints(rosetta_data, cv_labels)

    # 1. Build verb constraints
    verb_constraints = build_verb_constraints(verb_data, cv_labels, eva_to_cell)
    n_hard = sum(1 for vc in verb_constraints if vc.is_hard)
    n_soft = len(verb_constraints) - n_hard

    print(f"\n  Verb constraints built: {len(verb_constraints)} total "
          f"({n_hard} hard, {n_soft} soft)")
    for vc in verb_constraints:
        mode = 'HARD' if vc.is_hard else 'soft'
        print(f"    [{mode}] '{vc.voynich_stem}' → '{vc.latin_verb}' "
              f"({vc.latin_syllables})  cells={len(vc.voynich_cells)}  "
              f"conf={vc.confidence:.3f}")

    constrained_cells = list({cell for vc in verb_constraints for cell in vc.voynich_cells})

    # 2. Build phoneme inventory
    inventory = build_phoneme_inventory(
        'latin', ref_corpus,
        relaxation_level=relaxation_level,
        inherent_vowel=best_inherent_vowel,
    )

    # 3. Build and initialise variables
    variables = build_csp_variables(cv_labels)
    cell_frequencies = {v.cell_key: v.frequency for v in variables}
    variables = initialise_domains(
        variables, inventory, cell_frequencies, anchors, frequency_slack=3,
    )

    # 4. Apply hard verb constraints to narrow domains
    if verb_constraints:
        cell_domains = {v.cell_key: list(v.domain) for v in variables}
        cell_domains = apply_verb_constraints(cell_domains, verb_constraints, inventory)
        for v in variables:
            v.domain = cell_domains.get(v.cell_key, v.domain)

    _, variables = ac3_propagate(variables)

    # 5. Run beam search with verb constraints
    assignments = beam_search(
        variables, lm, corpus_tokens, eva_to_cell, anchors, inventory,
        ref_word_set=ref_word_set,
        verb_constraints=verb_constraints,
        relaxation_level=relaxation_level,
        beam_width=beam_width, max_solutions=max_solutions,
    )

    if assignments:
        best = assignments[0]
        dict_hit_after = best.dict_hit_rate
        anchor_after = best.anchor_match_count
        verb_matches = best.verb_match_count
        best_map = dict(best.mapping)
    else:
        dict_hit_after = 0.0
        anchor_after = 0
        verb_matches = 0
        best_map = {}

    print(f"\n  Result:")
    print(f"    dict_hit before: {base_dict_hit:.4f}")
    print(f"    dict_hit after:  {dict_hit_after:.4f}  (Δ={dict_hit_after - base_dict_hit:+.4f})")
    print(f"    anchor matches:  {anchor_after} (was {base_anchor_matches})")
    print(f"    verb matches:    {verb_matches}")

    # 6. Check illustration consistency
    consistency = check_illustration_consistency(
        best_map, verb_constraints, rosetta_data, eva_to_cell, cv_labels,
    )
    if consistency['globally_consistent']:
        print(f"  Illustration consistency: globally consistent ✓")
    else:
        print(f"  Illustration consistency: {consistency['n_total_conflicts']} conflicts")

    # 7. Constraints applied summary
    constraints_applied = []
    for vc in verb_constraints:
        constraints_applied.append({
            'voynich_stem': vc.voynich_stem,
            'latin_verb': vc.latin_verb,
            'latin_syllables': vc.latin_syllables,
            'voynich_cells': vc.voynich_cells,
            'confidence': vc.confidence,
            'is_hard': vc.is_hard,
        })

    gate_passed = dict_hit_after >= 0.15 or verb_matches >= 5
    if gate_passed:
        verdict = f'verb_constrained_dict_hit_{dict_hit_after:.3f}'
    else:
        verdict = f'verb_constraints_applied_dict_hit_{dict_hit_after:.3f}'

    result = VerbConstraintResult(
        n_verb_constraints=len(verb_constraints),
        n_hard_constraints=n_hard,
        n_soft_constraints=n_soft,
        constraints_applied=constraints_applied,
        constrained_cells=constrained_cells,
        dict_hit_before=base_dict_hit,
        dict_hit_after=dict_hit_after,
        anchor_matches_before=base_anchor_matches,
        anchor_matches_after=anchor_after,
        verb_matches=verb_matches,
        best_assignment=best_map,
        illustration_consistency=consistency,
        gate_passed=gate_passed,
        verdict=verdict,
    )

    return result, verb_constraints


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_verb_constraints() -> Dict:
    """Phase 11.5.4: verb-constrained CSP solving.

    1. Loads Phase 11.5.2-3 refinement results for best_assignment,
       best_relaxation_level, best_inherent_vowel.
    2. Loads Phase 9 verb assignments from verb_identification.json.
    3. Runs verb-constrained CSP.
    4. Gate: dict_hit_after >= 0.15 OR verb_matches >= 5.
    5. Saves to results/verb_constraints.json.
    """
    print("=" * 70)
    print("PHASE 11.5.4: Verb-Constrained CSP Solving")
    print("=" * 70)

    t0_total = time.time()
    rd = _results_dir()

    # ------------------------------------------------------------------
    # 1. Load refinement results
    # ------------------------------------------------------------------
    refinement_path = os.path.join(rd, 'csp_refinement.json')
    if not os.path.exists(refinement_path):
        print("  [WARN] csp_refinement.json not found — using Phase 11 defaults")
        best_relaxation_level = 0
        best_inherent_vowel = 'a'
        base_dict_hit = 0.111  # Phase 11 reported value
        base_anchor_matches = 1
        base_assignment: Dict[str, str] = {}
    else:
        with open(refinement_path) as f:
            refine_data = json.load(f)
        best_relaxation_level = int(refine_data.get('best_relaxation_level', 0))
        best_inherent_vowel = refine_data.get('best_inherent_vowel', 'a')
        base_dict_hit = float(refine_data.get('best_dict_hit_rate', 0.111))
        base_assignment = refine_data.get('final_assignment', {})
        print(f"  Loaded refinement: level={best_relaxation_level}, "
              f"vowel='{best_inherent_vowel}', dict_hit={base_dict_hit:.4f}")

    # Get anchor matches from Phase 11
    decode_path = os.path.join(rd, 'csp_decode.json')
    base_anchor_matches = 1
    if os.path.exists(decode_path):
        with open(decode_path) as f:
            decode_data = json.load(f)
        lang_res = decode_data.get('language_results', {}).get('latin', {})
        base_anchor_matches = int(lang_res.get('anchor_match_count', 1))

    # ------------------------------------------------------------------
    # 2. Load Phase 9 verb data
    # ------------------------------------------------------------------
    verb_path = os.path.join(rd, 'verb_identification.json')
    if not os.path.exists(verb_path):
        print("  [SKIP] verb_identification.json not found")
        return {'verdict': 'skipped', 'reason': 'no_verb_data'}

    with open(verb_path) as f:
        verb_data = json.load(f)

    n_assignments = len(verb_data.get('assignments', []))
    print(f"  Loaded {n_assignments} verb assignments from Phase 9")

    # ------------------------------------------------------------------
    # 3. Load supporting data
    # ------------------------------------------------------------------
    print("\nLoading data...")
    corpus = load_corpus(verbose=False)
    ref_corpus = load_reference_corpus(verbose=False)

    cv_path = os.path.join(rd, 'cv_labels.json')
    with open(cv_path) as f:
        cv_labels = json.load(f)

    rosetta_path = os.path.join(rd, 'rosetta_selection.json')
    with open(rosetta_path) as f:
        rosetta_data = json.load(f)

    eva_to_cell = build_eva_to_cell_lookup(cv_labels)
    corpus_tokens = corpus.get_tokens(language='A', paragraph_only=True)
    sample_tokens = corpus_tokens[:2000]
    print(f"  Corpus sample: {len(sample_tokens)} tokens")

    # ------------------------------------------------------------------
    # 4. Run verb-constrained CSP
    # ------------------------------------------------------------------
    result, verb_constraints = run_verb_constrained_csp(
        sample_tokens, ref_corpus, cv_labels, rosetta_data, eva_to_cell,
        verb_data=verb_data,
        base_assignment=base_assignment,
        base_dict_hit=base_dict_hit,
        base_anchor_matches=base_anchor_matches,
        relaxation_level=best_relaxation_level,
        best_inherent_vowel=best_inherent_vowel,
        beam_width=50, max_solutions=20,
    )

    # ------------------------------------------------------------------
    # 5. Print gate result and save
    # ------------------------------------------------------------------
    print(f"\n  Gate: {'PASS ✓' if result.gate_passed else 'FAIL ✗'}")
    print(f"  Verdict: {result.verdict}")

    out_path = os.path.join(rd, 'verb_constraints.json')
    # Add relaxation_level and inherent_vowel to output for downstream steps
    out_data = _convert(asdict(result))
    out_data['best_relaxation_level'] = best_relaxation_level
    out_data['best_inherent_vowel'] = best_inherent_vowel

    with open(out_path, 'w') as f:
        json.dump(out_data, f, indent=2)

    elapsed = time.time() - t0_total
    print(f"\n  Saved to {out_path} ({elapsed:.1f}s total)")

    return out_data
