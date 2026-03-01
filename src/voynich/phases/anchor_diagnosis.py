"""
Phase 6.1 Fix B: Anchor-Level Inconsistency Diagnosis
=======================================================
Diagnose which specific anchors and character mappings cause cross-consistency
failures. Provides per-anchor consistency profiling, poison anchor identification,
per-character unanimity analysis, and iterative anchor pruning.

Sub-analyses:
  B.1 — Per-anchor consistency profiling
  B.2 — Leave-one-anchor-out unanimity and poison anchor identification
  B.3 — Per-character consistency profiling
  B.4 — Iterative anchor pruning to maximize unanimity

Output:
  results/anchor_diagnosis.json
"""

import json
import os
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
class AnchorConsistencyProfile:
    """Per-anchor consistency metrics."""
    folio: str
    plant_name: str
    voynich_stem: str
    medieval_stem: str
    n_chars_mapped: int
    n_reuse_instances: int
    n_consistent: int
    n_conflicting: int
    consistency_rate: float
    is_poison: bool
    unanimity_without: float
    unanimity_delta: float


@dataclass
class CharConsistencyProfile:
    """Per-character consistency metrics."""
    eva_char: str
    n_assignments: int
    unique_values: int
    majority_value: str
    majority_count: int
    unanimity: float
    classification: str  # 'high', 'medium', 'low'
    all_votes: Dict[str, int]
    source_anchors: Dict[str, str]  # folio -> proposed value


@dataclass
class PruningStep:
    """One step of iterative anchor pruning."""
    step: int
    removed_folio: str
    removed_reason: str
    remaining_anchors: int
    unanimity: float
    n_chars_mapped: int


@dataclass
class AnchorDiagnosisResult:
    """Full anchor diagnosis output."""
    # Baseline
    n_anchors: int
    baseline_unanimity: float
    # Per-anchor profiles
    anchor_profiles: List[Dict]
    n_poison_anchors: int
    poison_anchors: List[str]
    # Per-character profiles
    char_profiles: List[Dict]
    n_high_unanimity_chars: int
    n_medium_unanimity_chars: int
    n_low_unanimity_chars: int
    # Pruning
    pruning_steps: List[Dict]
    pruned_anchors_remaining: int
    pruned_unanimity: float
    pruned_char_coverage: int
    # Verdict
    verdict: str


# ---------------------------------------------------------------------------
# B.1: Per-anchor consistency profiling
# ---------------------------------------------------------------------------

def profile_anchor_consistency(
    hypotheses: List[AnchorHypothesis],
) -> List[AnchorConsistencyProfile]:
    """
    Compute per-anchor consistency rate.

    For each anchor, counts how many of its character-reuse instances
    (characters shared with other anchors) are consistent vs conflicting.
    """
    # Build global vote map: eva_char -> {latin_seg -> [folio, ...]}
    char_sources: Dict[str, Dict[str, List[str]]] = defaultdict(
        lambda: defaultdict(list))
    for h in hypotheses:
        for eva_char, latin_seg in h.char_mappings.items():
            if latin_seg:
                char_sources[eva_char][latin_seg].append(h.folio)

    # For each anchor, count consistent/conflicting reuse instances
    profiles: List[AnchorConsistencyProfile] = []
    for h in hypotheses:
        n_reuse = 0
        n_consistent = 0
        n_conflicting = 0

        for eva_char, latin_seg in h.char_mappings.items():
            if not latin_seg:
                continue
            votes = char_sources.get(eva_char, {})
            total_sources = sum(len(folios) for folios in votes.values())
            if total_sources <= 1:
                # Only this anchor uses this char — no reuse to check
                continue

            n_reuse += 1
            # Check if this anchor's proposal is the majority
            majority_seg = max(votes, key=lambda s: len(votes[s]))
            if latin_seg == majority_seg:
                n_consistent += 1
            else:
                n_conflicting += 1

        rate = n_consistent / max(n_reuse, 1)

        profiles.append(AnchorConsistencyProfile(
            folio=h.folio,
            plant_name=h.medieval_name,
            voynich_stem=h.voynich_stem,
            medieval_stem=h.medieval_stem,
            n_chars_mapped=h.n_chars_mapped,
            n_reuse_instances=n_reuse,
            n_consistent=n_consistent,
            n_conflicting=n_conflicting,
            consistency_rate=round(rate, 4),
            is_poison=False,  # Set in B.2
            unanimity_without=0.0,  # Set in B.2
            unanimity_delta=0.0,  # Set in B.2
        ))

    return profiles


# ---------------------------------------------------------------------------
# B.2: Leave-one-anchor-out and poison identification
# ---------------------------------------------------------------------------

def identify_poison_anchors(
    hypotheses: List[AnchorHypothesis],
    baseline_unanimity: float,
    profiles: List[AnchorConsistencyProfile],
    poison_threshold: float = 0.05,
) -> List[AnchorConsistencyProfile]:
    """
    Compute leave-one-out unanimity for each anchor.

    An anchor is 'poison' if removing it increases unanimity by more
    than poison_threshold (default 0.05).
    """
    folio_to_profile = {p.folio: p for p in profiles}

    for i, h in enumerate(hypotheses):
        remaining = [hyp for j, hyp in enumerate(hypotheses) if j != i]
        if len(remaining) >= 2:
            cc = cross_consistency_check(remaining)
            without_u = cc.unanimity_ratio
        else:
            without_u = 0.0

        delta = without_u - baseline_unanimity
        profile = folio_to_profile[h.folio]
        profile.unanimity_without = round(without_u, 4)
        profile.unanimity_delta = round(delta, 4)
        profile.is_poison = delta > poison_threshold

    return profiles


# ---------------------------------------------------------------------------
# B.3: Per-character consistency profiling
# ---------------------------------------------------------------------------

def profile_character_consistency(
    hypotheses: List[AnchorHypothesis],
) -> List[CharConsistencyProfile]:
    """
    Compute per-character consistency across all anchors.

    For each EVA character that appears in multiple anchors, determines
    the majority vote, unanimity, and classifies as high/medium/low.
    """
    # Collect votes per character
    char_votes: Dict[str, Dict[str, int]] = defaultdict(Counter)
    char_sources: Dict[str, Dict[str, str]] = defaultdict(dict)

    for h in hypotheses:
        for eva_char, latin_seg in h.char_mappings.items():
            if latin_seg:
                char_votes[eva_char][latin_seg] += 1
                char_sources[eva_char][h.folio] = latin_seg

    profiles: List[CharConsistencyProfile] = []
    for eva_char, votes in sorted(char_votes.items()):
        total = sum(votes.values())
        if total < 1:
            continue

        majority_seg, majority_count = votes.most_common(1)[0]
        unanimity = majority_count / total

        if unanimity > 0.80:
            classification = 'high'
        elif unanimity > 0.50:
            classification = 'medium'
        else:
            classification = 'low'

        profiles.append(CharConsistencyProfile(
            eva_char=eva_char,
            n_assignments=total,
            unique_values=len(votes),
            majority_value=majority_seg,
            majority_count=majority_count,
            unanimity=round(unanimity, 4),
            classification=classification,
            all_votes=dict(votes),
            source_anchors=dict(char_sources.get(eva_char, {})),
        ))

    # Sort by unanimity ascending (worst first)
    profiles.sort(key=lambda p: p.unanimity)
    return profiles


# ---------------------------------------------------------------------------
# B.4: Iterative anchor pruning
# ---------------------------------------------------------------------------

def iterative_prune(
    hypotheses: List[AnchorHypothesis],
    target_unanimity: float = 0.50,
    min_anchors: int = 5,
) -> Tuple[List[AnchorHypothesis], List[PruningStep]]:
    """
    Iteratively remove the worst poison anchor until unanimity > target
    or fewer than min_anchors remain.

    Returns (remaining_hypotheses, pruning_steps).
    """
    remaining = list(hypotheses)
    steps: List[PruningStep] = []
    step_num = 0

    while len(remaining) > min_anchors:
        if len(remaining) < 2:
            break

        cc = cross_consistency_check(remaining)
        current_u = cc.unanimity_ratio

        if current_u >= target_unanimity:
            break

        # Find the anchor whose removal gives the biggest unanimity gain
        best_idx = -1
        best_delta = -999.0
        best_without = 0.0

        for i in range(len(remaining)):
            subset = [h for j, h in enumerate(remaining) if j != i]
            if len(subset) >= 2:
                sub_cc = cross_consistency_check(subset)
                delta = sub_cc.unanimity_ratio - current_u
                if delta > best_delta:
                    best_delta = delta
                    best_idx = i
                    best_without = sub_cc.unanimity_ratio

        if best_idx < 0 or best_delta <= 0:
            # No removal improves unanimity — stop
            break

        removed = remaining.pop(best_idx)
        step_num += 1

        # Recompute after removal
        if len(remaining) >= 2:
            new_cc = cross_consistency_check(remaining)
            new_u = new_cc.unanimity_ratio
            new_chars = new_cc.n_chars_total
        else:
            new_u = 0.0
            new_chars = 0

        steps.append(PruningStep(
            step=step_num,
            removed_folio=removed.folio,
            removed_reason=(
                f"Removing {removed.folio} ({removed.medieval_name}) "
                f"improves unanimity by {best_delta:+.4f}"
            ),
            remaining_anchors=len(remaining),
            unanimity=round(new_u, 4),
            n_chars_mapped=new_chars,
        ))

    return remaining, steps


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_anchor_diagnosis(
    anchor_data: Optional[Dict] = None,
    use_tfidf: bool = False,
) -> Dict:
    """
    Run Phase 6.1 Fix B: Anchor-Level Inconsistency Diagnosis.

    1. Load anchor-propagate results and rebuild hypotheses
    2. B.1: Per-anchor consistency profiling
    3. B.2: Leave-one-out and poison anchor identification
    4. B.3: Per-character consistency profiling
    5. B.4: Iterative pruning
    6. Report verdict
    7. Save results/anchor_diagnosis.json
    """
    print("=" * 70)
    print("Phase 6.1 Fix B: Anchor-Level Inconsistency Diagnosis")
    print("=" * 70)

    # Load prior results
    if anchor_data is None:
        results_path = os.path.join(_results_dir(), 'anchor_propagate.json')
        if os.path.exists(results_path):
            with open(results_path) as f:
                anchor_data = json.load(f)
        else:
            print("\n  No anchor_propagate.json found. Run 'voynich anchor' first.")
            return {}

    baseline_unanimity = anchor_data.get('unanimity_ratio', 0.0)
    print(f"\n  Baseline unanimity: {baseline_unanimity:.4f}")

    # Rebuild hypotheses
    print("\n  Loading corpus and rebuilding anchor hypotheses...")
    corpus = load_corpus(verbose=False)
    concordance = parse_concordance()
    medieval_names = load_medieval_names()
    folio_sets = build_folio_identification_sets(
        concordance, medieval_names, corpus,
        use_tfidf=use_tfidf,
    )
    folio_index = {fs.folio: fs for fs in folio_sets}

    hypotheses: List[AnchorHypothesis] = []
    stored = anchor_data.get('anchor_hypotheses', [])
    for h_dict in stored:
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
                    hypotheses.append(h)
                break

    print(f"  Reconstructed anchors: {len(hypotheses)}")

    if len(hypotheses) < 3:
        print("  Too few anchors for diagnosis.")
        return {'verdict': 'insufficient_anchors'}

    # B.1: Per-anchor consistency profiling
    print("\n  B.1: Per-Anchor Consistency Profiling")
    print("  " + "─" * 66)
    profiles = profile_anchor_consistency(hypotheses)

    # B.2: Leave-one-out and poison identification
    print("\n  B.2: Leave-One-Out Unanimity + Poison Detection")
    profiles = identify_poison_anchors(
        hypotheses, baseline_unanimity, profiles,
    )

    print(f"\n  {'Anchor':<8s} {'Plant':<18s} {'Stem':<10s} "
          f"{'Consist.':<10s} {'W/o Unan.':<10s} {'Δ':<8s} {'Poison?'}")
    print(f"  {'─' * 8} {'─' * 18} {'─' * 10} {'─' * 10} {'─' * 10} "
          f"{'─' * 8} {'─' * 7}")
    for p in sorted(profiles, key=lambda x: x.consistency_rate):
        poison_str = "YES" if p.is_poison else ""
        print(f"  {p.folio:<8s} {p.plant_name[:18]:<18s} {p.medieval_stem[:10]:<10s} "
              f"{p.consistency_rate:<10.4f} {p.unanimity_without:<10.4f} "
              f"{p.unanimity_delta:+.4f}  {poison_str}")

    poison_anchors = [p.folio for p in profiles if p.is_poison]
    print(f"\n  Poison anchors: {poison_anchors if poison_anchors else 'None'}")

    # B.3: Per-character consistency profiling
    print("\n  B.3: Per-Character Consistency Profiling")
    print("  " + "─" * 66)
    char_profiles = profile_character_consistency(hypotheses)

    n_high = sum(1 for c in char_profiles if c.classification == 'high')
    n_medium = sum(1 for c in char_profiles if c.classification == 'medium')
    n_low = sum(1 for c in char_profiles if c.classification == 'low')

    print(f"\n  {'EVA char':<10s} {'Assigns':<8s} {'Majority':<10s} "
          f"{'Unanimity':<10s} {'Class':<8s} {'Votes'}")
    print(f"  {'─' * 10} {'─' * 8} {'─' * 10} {'─' * 10} {'─' * 8} {'─' * 20}")
    for cp in char_profiles:
        vote_str = ', '.join(f'{v}:{c}' for v, c in
                             sorted(cp.all_votes.items(), key=lambda x: -x[1]))
        print(f"  {cp.eva_char:<10s} {cp.n_assignments:<8d} "
              f"{cp.majority_value:<10s} {cp.unanimity:<10.4f} "
              f"{cp.classification:<8s} {vote_str}")

    print(f"\n  High unanimity (>0.80): {n_high}")
    print(f"  Medium (0.50-0.80): {n_medium}")
    print(f"  Low (<0.50): {n_low}")

    # B.4: Iterative pruning
    print("\n  B.4: Iterative Anchor Pruning")
    print("  " + "─" * 66)
    pruned, steps = iterative_prune(
        hypotheses, target_unanimity=0.50, min_anchors=5,
    )

    for step in steps:
        print(f"    Step {step.step}: Remove {step.removed_folio} -> "
              f"unanimity={step.unanimity:.4f} "
              f"({step.remaining_anchors} anchors, "
              f"{step.n_chars_mapped} chars)")

    if pruned and len(pruned) >= 2:
        final_cc = cross_consistency_check(pruned)
        pruned_u = final_cc.unanimity_ratio
        pruned_chars = final_cc.n_chars_total
    else:
        pruned_u = 0.0
        pruned_chars = 0

    print(f"\n  After pruning:")
    print(f"    Anchors remaining: {len(pruned)}")
    print(f"    Unanimity: {pruned_u:.4f}")
    print(f"    Character coverage: {pruned_chars}")

    # Verdict
    if pruned_u >= 0.60:
        verdict = 'pruned_set_viable'
    elif pruned_u >= 0.50:
        verdict = 'marginal_improvement'
    elif len(pruned) <= 5 and pruned_u < 0.50:
        verdict = 'systematic_problem'
    else:
        verdict = 'insufficient_improvement'
    print(f"\n  VERDICT: {verdict}")

    # Build result
    result = AnchorDiagnosisResult(
        n_anchors=len(hypotheses),
        baseline_unanimity=round(baseline_unanimity, 4),
        anchor_profiles=[_convert(asdict(p)) for p in profiles],
        n_poison_anchors=len(poison_anchors),
        poison_anchors=poison_anchors,
        char_profiles=[_convert(asdict(cp)) for cp in char_profiles],
        n_high_unanimity_chars=n_high,
        n_medium_unanimity_chars=n_medium,
        n_low_unanimity_chars=n_low,
        pruning_steps=[_convert(asdict(s)) for s in steps],
        pruned_anchors_remaining=len(pruned),
        pruned_unanimity=round(pruned_u, 4),
        pruned_char_coverage=pruned_chars,
        verdict=verdict,
    )

    # Save
    out_path = os.path.join(_results_dir(), 'anchor_diagnosis.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")

    return _convert(asdict(result))
