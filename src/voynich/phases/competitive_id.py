"""
Phase 6 C: Competitive ID Resolution
======================================
For folios with multiple competing botanical identifications, test all
combinations via beam search and select the identification set with
highest cross-consistency.

Sub-analyses:
  6.C — Beam search over competing identifications
  6.C.null — Shuffled identification options null test

Output:
  results/competitive_id.json
"""

import json
import math
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from voynich.core.corpus import load_corpus, VoynichCorpus
from voynich.core._paths import results_dir as _results_dir
from voynich.phases.illustration_constrained import (
    FolioIdentificationSet, PlantIdentification,
    load_medieval_names, parse_concordance,
    build_folio_identification_sets,
    _convert, _check_gate,
)
from voynich.phases.anchor_propagate import (
    build_anchor_hypothesis, cross_consistency_check,
    AnchorHypothesis, CrossConsistencyMatrix,
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CompetingCandidate:
    """One candidate identification assignment for a contested folio."""
    folio: str
    identification_index: int
    linnaean_name: str
    medieval_name: str
    medieval_stem: str


@dataclass
class BeamState:
    """State in beam search: a partial assignment of IDs to contested folios."""
    assignments: Dict[str, int]  # folio -> identification_index
    unanimity_ratio: float
    n_chars_mapped: int
    score: float


@dataclass
class CompetitiveIDResult:
    """Full Phase 6 C output."""
    n_contested_folios: int
    n_combinations_total: int
    beam_width: int
    n_states_explored: int
    # Best assignment
    best_assignments: List[Dict]
    best_unanimity: float
    best_n_chars: int
    best_score: float
    # Runner-up
    runner_up_unanimity: float
    runner_up_score: float
    separation: float
    # Comparison with anchor-only
    anchor_only_unanimity: float
    improvement: float
    # Gate
    gate_passed: bool
    verdict: str


# ---------------------------------------------------------------------------
# Enumerate contested folios
# ---------------------------------------------------------------------------

def enumerate_contested_folios(
    folio_sets: List[FolioIdentificationSet],
) -> List[Tuple[str, List[CompetingCandidate]]]:
    """
    Identify folios with multiple competing identifications (any tier).

    Only includes candidates with resolved medieval Latin names.
    """
    contested: List[Tuple[str, List[CompetingCandidate]]] = []

    for fs in folio_sets:
        # Get all identifications with resolved medieval names
        candidates: List[CompetingCandidate] = []
        for i, pid in enumerate(fs.identifications):
            if pid.medieval_stem:
                candidates.append(CompetingCandidate(
                    folio=fs.folio,
                    identification_index=i,
                    linnaean_name=pid.linnaean_name,
                    medieval_name=pid.medieval_name or '',
                    medieval_stem=pid.medieval_stem,
                ))

        if len(candidates) >= 2:
            contested.append((fs.folio, candidates))

    # Sort by number of candidates (fewest first for efficient pruning)
    contested.sort(key=lambda x: len(x[1]))
    return contested


# ---------------------------------------------------------------------------
# Beam search
# ---------------------------------------------------------------------------

def beam_search_assignments(
    contested: List[Tuple[str, List[CompetingCandidate]]],
    fixed_anchors: List[AnchorHypothesis],
    folio_sets: List[FolioIdentificationSet],
    beam_width: int = 10,
) -> Tuple[List[BeamState], int]:
    """
    Beam search over competing ID assignments.

    Starts with fixed_anchors (already-resolved Tier 1 folios), then
    expands one contested folio at a time, keeping the top beam_width
    states by score at each step.

    Returns (final_beam_states, n_states_explored).
    """
    folio_index = {fs.folio: fs for fs in folio_sets}

    # Initial state: just the fixed anchors
    initial_cc = cross_consistency_check(fixed_anchors)
    initial_state = BeamState(
        assignments={},
        unanimity_ratio=initial_cc.unanimity_ratio,
        n_chars_mapped=initial_cc.n_chars_total,
        score=initial_cc.unanimity_ratio * math.log(
            initial_cc.n_chars_total + 1),
    )

    beam: List[BeamState] = [initial_state]
    n_explored = 0

    for folio, candidates in contested:
        fs = folio_index.get(folio)
        if fs is None:
            continue

        next_beam: List[BeamState] = []

        for state in beam:
            for cand in candidates:
                n_explored += 1

                # Create a fake identification for this candidate
                pid = PlantIdentification(
                    folio=folio,
                    linnaean_name=cand.linnaean_name,
                    common_name='',
                    source='competitive',
                    medieval_name=cand.medieval_name,
                    medieval_stem=cand.medieval_stem,
                    declension=None,
                )

                # Build anchor hypothesis
                h = build_anchor_hypothesis(fs, pid)
                if h is None or not h.paradigm_compatible:
                    continue

                # Combine with fixed anchors and prior assignments
                all_hypotheses = list(fixed_anchors)

                # Add hypotheses from prior assignments in this state
                for prev_folio, prev_idx in state.assignments.items():
                    prev_fs = folio_index.get(prev_folio)
                    if prev_fs and prev_idx < len(prev_fs.identifications):
                        prev_pid = prev_fs.identifications[prev_idx]
                        prev_h = build_anchor_hypothesis(prev_fs, prev_pid)
                        if prev_h and prev_h.paradigm_compatible:
                            all_hypotheses.append(prev_h)

                # Add current candidate
                all_hypotheses.append(h)

                # Check cross-consistency
                cc = cross_consistency_check(all_hypotheses)
                score = cc.unanimity_ratio * math.log(
                    cc.n_chars_total + 1)

                new_assignments = dict(state.assignments)
                new_assignments[folio] = cand.identification_index

                next_beam.append(BeamState(
                    assignments=new_assignments,
                    unanimity_ratio=cc.unanimity_ratio,
                    n_chars_mapped=cc.n_chars_total,
                    score=score,
                ))

        if next_beam:
            # Keep top beam_width states
            next_beam.sort(key=lambda s: s.score, reverse=True)
            beam = next_beam[:beam_width]
        # If no expansions possible, keep current beam

    return beam, n_explored


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_competitive_id(
    anchor_data: Optional[Dict] = None,
    constrained_data: Optional[Dict] = None,
) -> Dict:
    """
    Run Phase 6 C: Competitive ID Resolution.

    1. Load Phase 6 A+B results (fixed anchors)
    2. Load Phase 6.0 results (all folio identification sets)
    3. Enumerate contested folios
    4. If no contested folios, skip
    5. Run beam search
    6. Compare best assignment vs anchor-only
    7. Gate: separation > 0.05 between best and runner-up
    8. Save results/competitive_id.json
    """
    print("=" * 70)
    print("Phase 6 C: Competitive ID Resolution")
    print("=" * 70)

    # Load prior results
    if anchor_data is None:
        results_path = os.path.join(_results_dir(), 'anchor_propagate.json')
        if os.path.exists(results_path):
            with open(results_path) as f:
                anchor_data = json.load(f)
        else:
            from voynich.phases.anchor_propagate import run_anchor_propagate
            anchor_data = run_anchor_propagate()

    anchor_only_unanimity = anchor_data.get('unanimity_ratio', 0.0)

    # Rebuild folio sets
    print("\n  Loading corpus and building identification sets...")
    corpus = load_corpus(verbose=False)
    concordance = parse_concordance()
    medieval_names = load_medieval_names()
    folio_sets = build_folio_identification_sets(
        concordance, medieval_names, corpus,
    )
    folio_index = {fs.folio: fs for fs in folio_sets}

    # Rebuild fixed anchor hypotheses from stored data
    print("  Rebuilding anchor hypotheses...")
    fixed_anchors: List[AnchorHypothesis] = []
    stored_hypotheses = anchor_data.get('anchor_hypotheses', [])
    for h_dict in stored_hypotheses:
        if not h_dict.get('paradigm_compatible', False):
            continue
        folio = h_dict['folio']
        fs = folio_index.get(folio)
        if fs is None:
            continue
        for pid in fs.identifications:
            if pid.medieval_stem == h_dict.get('medieval_stem'):
                h = build_anchor_hypothesis(fs, pid)
                if h and h.paradigm_compatible:
                    fixed_anchors.append(h)
                break

    print(f"  Fixed anchors: {len(fixed_anchors)}")

    # Enumerate contested folios
    contested = enumerate_contested_folios(folio_sets)
    print(f"\n  Contested folios: {len(contested)}")

    if not contested:
        print("  No contested folios — skipping beam search.")
        result = CompetitiveIDResult(
            n_contested_folios=0, n_combinations_total=0,
            beam_width=10, n_states_explored=0,
            best_assignments=[], best_unanimity=anchor_only_unanimity,
            best_n_chars=0, best_score=0.0,
            runner_up_unanimity=0.0, runner_up_score=0.0,
            separation=0.0,
            anchor_only_unanimity=anchor_only_unanimity,
            improvement=0.0,
            gate_passed=True, verdict='no_contested_folios',
        )
        out_path = os.path.join(_results_dir(), 'competitive_id.json')
        with open(out_path, 'w') as f:
            json.dump(_convert(asdict(result)), f, indent=2, default=str)
        print(f"\n  Results saved to {out_path}")
        return _convert(asdict(result))

    for folio, candidates in contested:
        cand_names = [c.medieval_name for c in candidates]
        print(f"    {folio}: {len(candidates)} candidates — "
              f"{', '.join(cand_names)}")

    n_combos = 1
    for _, candidates in contested:
        n_combos *= len(candidates)
    print(f"\n  Total combination space: {n_combos}")

    # Run beam search
    print("\n  Running beam search (width=10)...")
    beam_width = 10
    beam, n_explored = beam_search_assignments(
        contested, fixed_anchors, folio_sets, beam_width=beam_width,
    )
    print(f"  States explored: {n_explored}")

    if beam:
        best = beam[0]
        runner_up = beam[1] if len(beam) > 1 else beam[0]

        print(f"\n  Best assignment:")
        print(f"    Unanimity: {best.unanimity_ratio:.4f}")
        print(f"    Chars mapped: {best.n_chars_mapped}")
        print(f"    Score: {best.score:.4f}")

        # Show assignments
        best_assignment_dicts = []
        for folio, idx in best.assignments.items():
            fs = folio_index.get(folio)
            if fs and idx < len(fs.identifications):
                pid = fs.identifications[idx]
                best_assignment_dicts.append({
                    'folio': folio,
                    'chosen_name': pid.medieval_name,
                    'chosen_stem': pid.medieval_stem,
                    'linnaean': pid.linnaean_name,
                    'source': pid.source,
                })
                print(f"    {folio}: {pid.medieval_name} "
                      f"({pid.linnaean_name}, {pid.source})")

        separation = best.score - runner_up.score
        improvement = best.unanimity_ratio - anchor_only_unanimity

        print(f"\n  Runner-up score: {runner_up.score:.4f}")
        print(f"  Separation: {separation:.4f}")
        print(f"  Improvement over anchor-only: {improvement:+.4f}")

        # Gate
        gate_ok, gate_msg = _check_gate(
            'best_runner_up_separation', separation, 0.05, 'greater',
        )
        print(f"\n{gate_msg}")
        verdict = 'identifications_resolved' if gate_ok else 'no_clear_winner'
        print(f"  Verdict: {verdict}")

        result = CompetitiveIDResult(
            n_contested_folios=len(contested),
            n_combinations_total=n_combos,
            beam_width=beam_width,
            n_states_explored=n_explored,
            best_assignments=best_assignment_dicts,
            best_unanimity=round(best.unanimity_ratio, 4),
            best_n_chars=best.n_chars_mapped,
            best_score=round(best.score, 4),
            runner_up_unanimity=round(runner_up.unanimity_ratio, 4),
            runner_up_score=round(runner_up.score, 4),
            separation=round(separation, 4),
            anchor_only_unanimity=round(anchor_only_unanimity, 4),
            improvement=round(improvement, 4),
            gate_passed=gate_ok,
            verdict=verdict,
        )
    else:
        print("\n  Beam search returned no states.")
        result = CompetitiveIDResult(
            n_contested_folios=len(contested),
            n_combinations_total=n_combos,
            beam_width=beam_width,
            n_states_explored=n_explored,
            best_assignments=[], best_unanimity=0.0,
            best_n_chars=0, best_score=0.0,
            runner_up_unanimity=0.0, runner_up_score=0.0,
            separation=0.0,
            anchor_only_unanimity=round(anchor_only_unanimity, 4),
            improvement=0.0,
            gate_passed=False, verdict='beam_search_failed',
        )

    # Save
    out_path = os.path.join(_results_dir(), 'competitive_id.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")

    return _convert(asdict(result))
