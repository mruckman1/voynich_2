"""
Phase 11.5.5 – Iterative CSP refinement loop
=============================================
After each CSP solve, extracts high-confidence dictionary hits and adds
them as new anchor constraints for the next iteration.  The loop runs
until convergence (Δdict_hit < 0.005, no new hits, or max_iterations).

Safety: if selectivity drops below 1.5× at any iteration, reverts to
the previous iteration and stops.

Decision gate:
  final_dict_hit >= 0.15   OR   improvement >= 0.03 (3pp gain)
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
    load_reference_corpus,
)
from voynich.core.stats import build_ngram_lm
from voynich.phases.csp_constraints import (
    AnchorConstraint,
    VerbConstraint,
    build_anchor_constraints,
    build_phoneme_inventory,
    build_verb_constraints,
    extract_confirmed_hits,
    score_cross_entropy,
    score_dict_hit_rate,
)
from voynich.phases.csp_decode import _random_baseline_ce
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
class IterationStep:
    """Result of one iteration of the refinement loop."""
    iteration: int
    n_new_anchors: int
    n_total_anchors: int
    dict_hit_rate: float
    cross_entropy: float
    selectivity: float
    anchor_match_count: int
    converged: bool
    best_assignment: Dict[str, str]


@dataclass
class CSPIterateResult:
    """Full Phase 11.5.5 output."""
    n_iterations: int
    iterations: List[Dict]
    initial_dict_hit: float
    final_dict_hit: float
    improvement: float
    converged: bool
    convergence_reason: str   # 'max_iter' | 'delta_small' | 'no_new_hits' | 'selectivity_dropped'
    final_assignment: Dict[str, str]
    total_anchors_used: int
    gate_passed: bool
    verdict: str


# ---------------------------------------------------------------------------
# Iterative refinement loop
# ---------------------------------------------------------------------------

def run_iterative_csp(
    corpus_tokens: List[str],
    ref_corpus: Any,
    cv_labels: Dict,
    rosetta_data: Dict,
    eva_to_cell: Dict[str, str],
    verb_data: Dict,
    initial_assignment: Dict[str, str],
    initial_dict_hit: float,
    relaxation_level: int = 0,
    best_inherent_vowel: Optional[str] = None,
    max_iterations: int = 5,
    beam_width: int = 40,
    max_solutions: int = 10,
    seed: int = 42,
) -> CSPIterateResult:
    """Run the iterative CSP refinement loop.

    At each iteration:
    1. Extract high-confidence dictionary hits from current assignment.
    2. Add them as AnchorConstraints (folio='confirmed_hit').
    3. Re-run CSP with the expanded anchor set.
    4. Check convergence and selectivity gate.
    """
    ref_tokens = ref_corpus.get_combined_tokens('latin')
    if not ref_tokens:
        ref_tokens = ref_corpus.get_combined_tokens(ref_corpus.languages[0])
    ref_word_set = set(ref_tokens[:50000])
    lm = build_ngram_lm(ref_tokens[:10000], order=3, smoothing=0.01)

    # Illustration anchors (fixed across all iterations)
    illustration_anchors = build_anchor_constraints(rosetta_data, cv_labels)

    # Verb constraints for scoring
    verb_constraints = build_verb_constraints(verb_data, cv_labels, eva_to_cell)

    # Phoneme inventory
    inventory = build_phoneme_inventory(
        'latin', ref_corpus,
        relaxation_level=relaxation_level,
        inherent_vowel=best_inherent_vowel,
    )

    # Random baseline for selectivity
    from voynich.core.reference import build_cv_syllable_table
    cv_base = build_cv_syllable_table('latin')
    cell_keys = list(cv_labels.keys())
    mean_random_ce, _ = _random_baseline_ce(
        cv_base, cell_keys, lm, corpus_tokens[:500], eva_to_cell,
        n_trials=80, max_tokens=300,
    )
    print(f"\n  Random baseline CE: {mean_random_ce:.4f}")

    # State
    current_assignment = dict(initial_assignment)
    all_anchors: List[AnchorConstraint] = list(illustration_anchors)
    confirmed_stems: set = set()  # track stems already added as anchors

    steps: List[IterationStep] = []
    current_dict_hit = initial_dict_hit
    converged = False
    convergence_reason = 'max_iter'

    for iteration in range(max_iterations):
        print(f"\n  --- Iteration {iteration + 1} ---")
        t0 = time.time()

        # Step 1: Extract confirmed hits from current assignment
        new_confirmed = extract_confirmed_hits(
            current_assignment, corpus_tokens, eva_to_cell,
            ref_word_set, min_frequency=3,
        )
        # Filter out stems already added
        new_anchors = [
            a for a in new_confirmed
            if a.voynich_stem not in confirmed_stems
        ]
        for a in new_anchors:
            confirmed_stems.add(a.voynich_stem)
            all_anchors.append(a)

        n_new = len(new_anchors)
        n_total = len(all_anchors)
        print(f"    New confirmed hits: {n_new}  (total anchors: {n_total})")

        if n_new == 0 and iteration > 0:
            converged = True
            convergence_reason = 'no_new_hits'
            print(f"    No new hits — converged.")
            break

        # Step 2: Re-build variables with all anchors
        variables = build_csp_variables(cv_labels)
        cell_frequencies = {v.cell_key: v.frequency for v in variables}
        variables = initialise_domains(
            variables, inventory, cell_frequencies, all_anchors, frequency_slack=3,
        )
        _, variables = ac3_propagate(variables)

        # Step 3: Beam search
        assignments = beam_search(
            variables, lm, corpus_tokens, eva_to_cell, all_anchors, inventory,
            ref_word_set=ref_word_set,
            verb_constraints=verb_constraints,
            relaxation_level=relaxation_level,
            beam_width=beam_width, max_solutions=max_solutions,
            seed=seed + iteration,
        )

        elapsed = time.time() - t0

        if assignments:
            best = assignments[0]
            new_dict_hit = best.dict_hit_rate
            new_ce = best.cross_entropy
            anchor_n = best.anchor_match_count
            new_mapping = dict(best.mapping)
        else:
            new_dict_hit = current_dict_hit
            new_ce = 99.0
            anchor_n = 0
            new_mapping = current_assignment

        selectivity = mean_random_ce / max(new_ce, 0.01)
        delta = new_dict_hit - current_dict_hit
        print(f"    dict_hit={new_dict_hit:.4f}  CE={new_ce:.4f}  "
              f"selectivity={selectivity:.2f}x  anchors={anchor_n}  "
              f"Δdict_hit={delta:+.4f}  ({elapsed:.1f}s)")

        # Safety: check selectivity gate
        if selectivity < 1.5:
            print(f"    [WARN] Selectivity dropped below 1.5× — reverting and stopping")
            converged = True
            convergence_reason = 'selectivity_dropped'
            break

        steps.append(IterationStep(
            iteration=iteration + 1,
            n_new_anchors=n_new,
            n_total_anchors=n_total,
            dict_hit_rate=new_dict_hit,
            cross_entropy=new_ce,
            selectivity=selectivity,
            anchor_match_count=anchor_n,
            converged=False,
            best_assignment=new_mapping,
        ))

        # Check convergence
        if abs(delta) < 0.005:
            converged = True
            convergence_reason = 'delta_small'
            print(f"    Δdict_hit < 0.005 — converged.")
            steps[-1].converged = True
            current_assignment = new_mapping
            current_dict_hit = new_dict_hit
            break

        current_assignment = new_mapping
        current_dict_hit = new_dict_hit

    if not converged:
        convergence_reason = 'max_iter'
        print(f"\n  Reached max_iterations={max_iterations}")

    return CSPIterateResult(
        n_iterations=len(steps),
        iterations=[_convert(asdict(s)) for s in steps],
        initial_dict_hit=initial_dict_hit,
        final_dict_hit=current_dict_hit,
        improvement=current_dict_hit - initial_dict_hit,
        converged=converged,
        convergence_reason=convergence_reason,
        final_assignment=current_assignment,
        total_anchors_used=len(all_anchors),
        gate_passed=current_dict_hit >= 0.15 or (current_dict_hit - initial_dict_hit) >= 0.03,
        verdict=f'csp_iterate_{convergence_reason}_dict_hit_{current_dict_hit:.3f}',
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_csp_iterate() -> Dict:
    """Phase 11.5.5: iterative CSP refinement loop.

    1. Loads Phase 11.5.4 verb_constraints.json for initial assignment
       and configuration (relaxation_level, inherent_vowel).
    2. Falls back to Phase 11.5.2-3 or Phase 11 if not available.
    3. Runs the iterative refinement loop (up to 5 iterations).
    4. Gate: final_dict_hit >= 0.15 OR improvement >= 0.03.
    5. Saves to results/csp_iterate.json.
    """
    print("=" * 70)
    print("PHASE 11.5.5: Iterative CSP Refinement Loop")
    print("=" * 70)

    t0_total = time.time()
    rd = _results_dir()

    # ------------------------------------------------------------------
    # 1. Load initial assignment (prefer verb_constraints, then refinement)
    # ------------------------------------------------------------------
    relaxation_level = 0
    best_inherent_vowel: Optional[str] = 'a'
    initial_dict_hit = 0.111

    verb_path = os.path.join(rd, 'verb_constraints.json')
    if os.path.exists(verb_path):
        with open(verb_path) as f:
            vc_data = json.load(f)
        initial_assignment = vc_data.get('best_assignment', {})
        initial_dict_hit = float(vc_data.get('dict_hit_after', 0.111))
        relaxation_level = int(vc_data.get('best_relaxation_level', 0))
        best_inherent_vowel = vc_data.get('best_inherent_vowel', 'a')
        print(f"  Loaded from verb_constraints.json: "
              f"dict_hit={initial_dict_hit:.4f}, level={relaxation_level}")
    else:
        refinement_path = os.path.join(rd, 'csp_refinement.json')
        if os.path.exists(refinement_path):
            with open(refinement_path) as f:
                refine_data = json.load(f)
            initial_assignment = refine_data.get('final_assignment', {})
            initial_dict_hit = float(refine_data.get('best_dict_hit_rate', 0.111))
            relaxation_level = int(refine_data.get('best_relaxation_level', 0))
            best_inherent_vowel = refine_data.get('best_inherent_vowel', 'a')
            print(f"  Loaded from csp_refinement.json: dict_hit={initial_dict_hit:.4f}")
        else:
            decode_path = os.path.join(rd, 'csp_decode.json')
            if not os.path.exists(decode_path):
                print("  [SKIP] No prior results found — run csp-decode first")
                return {'verdict': 'skipped', 'reason': 'no_prior_results'}
            with open(decode_path) as f:
                decode_data = json.load(f)
            initial_assignment = decode_data.get('best_assignment', {})
            lang_res = decode_data.get('language_results', {}).get('latin', {})
            initial_dict_hit = float(lang_res.get('best_dict_hit', 0.111))
            print(f"  Loaded from csp_decode.json: dict_hit={initial_dict_hit:.4f}")

    if not initial_assignment:
        print("  [SKIP] No valid initial assignment found")
        return {'verdict': 'skipped', 'reason': 'no_assignment'}

    # ------------------------------------------------------------------
    # 2. Load Phase 9 verb data
    # ------------------------------------------------------------------
    verb_id_path = os.path.join(rd, 'verb_identification.json')
    if os.path.exists(verb_id_path):
        with open(verb_id_path) as f:
            verb_data = json.load(f)
    else:
        verb_data = {'assignments': []}
        print("  [WARN] verb_identification.json not found — no verb constraints")

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
    # 4. Run iterative refinement
    # ------------------------------------------------------------------
    result = run_iterative_csp(
        sample_tokens, ref_corpus, cv_labels, rosetta_data, eva_to_cell,
        verb_data=verb_data,
        initial_assignment=initial_assignment,
        initial_dict_hit=initial_dict_hit,
        relaxation_level=relaxation_level,
        best_inherent_vowel=best_inherent_vowel,
        max_iterations=5,
        beam_width=40,
        max_solutions=10,
    )

    # ------------------------------------------------------------------
    # 5. Print and save
    # ------------------------------------------------------------------
    print(f"\n  Final dict_hit: {result.final_dict_hit:.4f} "
          f"(improvement: {result.improvement:+.4f})")
    print(f"  Total anchors used: {result.total_anchors_used}")
    print(f"  Convergence: {result.convergence_reason}")
    print(f"  Gate: {'PASS ✓' if result.gate_passed else 'FAIL ✗'}")

    out_data = _convert(asdict(result))
    # Propagate configuration for downstream steps
    out_data['best_relaxation_level'] = relaxation_level
    out_data['best_inherent_vowel'] = best_inherent_vowel

    out_path = os.path.join(rd, 'csp_iterate.json')
    with open(out_path, 'w') as f:
        json.dump(out_data, f, indent=2)

    elapsed = time.time() - t0_total
    print(f"\n  Saved to {out_path} ({elapsed:.1f}s total)")

    return out_data
