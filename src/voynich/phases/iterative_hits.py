"""
Phase 15.3 – Iterative Re-Solving with Confirmed Hits
======================================================
Extract high-confidence dictionary hits from the best assignment, use them
as hard constraints (fixing specific triples to specific syllables), and
iteratively re-solve the CSP until convergence.

Dependency chain:
    feature_decode.json (Phase 14)
    dict_expansion.json (Step 15.1 – expanded dictionary)
    articulatory_csp.json (Step 15.2 – best AC approach/delta)
        → iterative_hits.json (this step)
"""

import copy
import json
import os
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    load_corpus,
    token_to_triples,
    tokenize_eva_chars,
)
from voynich.core.reference import (
    build_cv_syllable_table,
    build_expanded_word_set,
    build_triple_phoneme_hypotheses,
    load_reference_corpus,
)
from voynich.core.stats import build_ngram_lm
from voynich.phases.csp_constraints import (
    AnchorConstraint,
    PhonemeInventory,
    build_phoneme_inventory,
)
from voynich.phases.csp_solver import (
    _convert,
    ac3_propagate,
    beam_search,
    decode_corpus,
)
from voynich.phases.feature_csp import (
    FeatureVariable,
    build_feature_variables,
    initialise_feature_domains,
    _build_anchor_constraints_triple,
)
from voynich.phases.articulatory_csp import compute_articulatory_consistency


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class HitConstraint:
    """A confirmed dictionary hit used as a hard CSP constraint."""
    voynich_token: str
    decoded_word: str
    triple_keys: List[str]
    target_syllables: List[str]
    confidence: float


@dataclass
class ConstraintCoverage:
    n_triples_total: int
    n_triples_constrained: int
    mean_constraints_per_triple: float
    n_contradictions: int
    contradiction_details: List[Dict]


@dataclass
class IterationRecord:
    iteration: int
    n_constrained_triples: int
    n_free_triples: int
    dict_hit: float
    selectivity: float
    n_new_hits: int
    new_hit_words: List[str]


@dataclass
class IterativeHitsResult:
    # 3a: Anchor set
    n_confirmed_hits: int
    confirmed_hit_words: List[str]
    expanded_dict_used: bool

    # 3b: Constraint coverage
    coverage: Dict

    # 3c-3d: Iterative loop
    iterations: List[Dict]
    converged_at: int
    convergence_reason: str

    final_dict_hit: float
    final_selectivity: float
    final_assignment: Dict[str, str]
    gate_passed: bool
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Hit extraction
# ---------------------------------------------------------------------------

def extract_high_confidence_hits(
    assignment: Dict[str, str],
    voynich_tokens: List[str],
    eva_to_triple: Dict[str, str],
    ref_word_set: set,
    min_frequency: int = 2,
) -> List[HitConstraint]:
    """Extract confirmed dictionary hits as triple-level constraints.

    A token qualifies when:
    1. It decodes to a string that is in ref_word_set.
    2. The token appears >= min_frequency times.
    3. The decoded string has length >= 3 (avoids noise from short words).
    """
    freq: Counter = Counter(voynich_tokens)
    hits: List[HitConstraint] = []
    seen_tokens: set = set()

    for token in set(voynich_tokens):
        if freq[token] < min_frequency:
            continue

        triples = token_to_triples(token, eva_to_triple)
        if not triples:
            continue

        # Decode by looking up each triple in the assignment
        parts = []
        for tk in triples:
            syl = assignment.get(tk)
            if syl:
                parts.append(syl)
            else:
                break
        else:
            decoded = ''.join(parts)
            if len(decoded) >= 3 and decoded in ref_word_set:
                hits.append(HitConstraint(
                    voynich_token=token,
                    decoded_word=decoded,
                    triple_keys=triples,
                    target_syllables=parts,
                    confidence=1.0,
                ))

    return hits


def compute_constraint_coverage(
    hits: List[HitConstraint],
    n_total_triples: int = 25,
) -> ConstraintCoverage:
    """Compute how many triples are constrained and detect contradictions."""
    triple_constraints: Dict[str, set] = {}  # triple_key -> set of assigned syllables

    for hit in hits:
        for tk, syl in zip(hit.triple_keys, hit.target_syllables):
            if tk not in triple_constraints:
                triple_constraints[tk] = set()
            triple_constraints[tk].add(syl)

    n_constrained = len(triple_constraints)
    constraints_per = [len(v) for v in triple_constraints.values()]
    mean_per = sum(constraints_per) / len(constraints_per) if constraints_per else 0.0

    # Contradictions: a triple constrained to 2+ different syllables
    contradictions: List[Dict] = []
    for tk, syls in triple_constraints.items():
        if len(syls) > 1:
            contradictions.append({
                'triple_key': tk,
                'conflicting_syllables': sorted(syls),
            })

    return ConstraintCoverage(
        n_triples_total=n_total_triples,
        n_triples_constrained=n_constrained,
        mean_constraints_per_triple=round(mean_per, 2),
        n_contradictions=len(contradictions),
        contradiction_details=contradictions,
    )


def apply_hit_constraints(
    variables: List[FeatureVariable],
    hits: List[HitConstraint],
) -> List[FeatureVariable]:
    """Restrict domains for constrained triples to their target syllable.

    If a triple has contradictory constraints, the domain includes all
    conflicting syllables (let beam search resolve).
    """
    # Build triple -> allowed syllables from non-contradictory hits
    triple_target: Dict[str, set] = {}
    for hit in hits:
        for tk, syl in zip(hit.triple_keys, hit.target_syllables):
            if tk not in triple_target:
                triple_target[tk] = set()
            triple_target[tk].add(syl)

    for var in variables:
        if var.cell_key in triple_target:
            targets = triple_target[var.cell_key]
            if len(targets) == 1:
                # Unambiguous: hard constraint
                var.domain = list(targets)
            else:
                # Contradictory: keep all conflicting + any already in domain
                current = set(var.domain)
                current.update(targets)
                var.domain = list(current)

    return variables


# ---------------------------------------------------------------------------
# Random baseline
# ---------------------------------------------------------------------------

def _compute_random_baseline(
    variables_keys: List[str],
    all_syls: List[str],
    voynich_tokens: List[str],
    eva_to_triple: Dict[str, str],
    ref_word_set: set,
    n_trials: int = 50,
    seed: int = 42,
) -> float:
    rng = random.Random(seed)
    hits_list: List[float] = []
    for _ in range(n_trials):
        rand_map = {k: rng.choice(all_syls) for k in variables_keys}
        decoded = decode_corpus(voynich_tokens, rand_map, eva_to_triple, max_tokens=500)
        hits = sum(1 for w in decoded if w in ref_word_set)
        hits_list.append(hits / len(decoded) if decoded else 0.0)
    return sum(hits_list) / len(hits_list) if hits_list else 0.001


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_iterative_hits() -> None:
    """Step 15.3: Iterative re-solving with confirmed hit constraints."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE 15.3: Iterative Re-Solving with Confirmed Hits")
    print("=" * 70)

    rd = _results_dir()

    # Load Phase 14 baseline
    fd_path = os.path.join(rd, 'feature_decode.json')
    if not os.path.exists(fd_path):
        print("  [SKIP] feature_decode.json not found — run feature-decode first")
        return

    with open(fd_path) as f:
        fd_data = json.load(f)

    current_assignment = fd_data.get('best_assignment', {})
    if not current_assignment:
        print("  [SKIP] No best assignment in feature_decode.json")
        return

    baseline_dict_hit = fd_data.get('best_dict_hit', 0.0)

    # Load expanded dictionary if available
    de_path = os.path.join(rd, 'dict_expansion.json')
    expanded_dict_used = False
    if os.path.exists(de_path):
        with open(de_path) as f:
            de_data = json.load(f)
        if de_data.get('gate_passed', False):
            expanded_dict_used = True

    # Load articulatory CSP results for best approach
    ac_path = os.path.join(rd, 'articulatory_csp.json')
    ac_best_assignment = None
    if os.path.exists(ac_path):
        with open(ac_path) as f:
            ac_data = json.load(f)
        ac_best = ac_data.get('best_assignment', {})
        ac_best_hit = ac_data.get('best_dict_hit', 0.0)
        # Use AC assignment if it improved
        if ac_best and ac_best_hit > baseline_dict_hit:
            current_assignment = ac_best
            ac_best_assignment = ac_best
            print(f"  Using articulatory CSP assignment (dict_hit={ac_best_hit:.3f})")

    # Load corpus
    corpus = load_corpus(verbose=False)
    tokens = corpus.get_tokens(language='A', paragraph_only=True)
    if not tokens:
        print("  [SKIP] No Language A tokens found")
        return

    eva_to_triple = build_eva_to_triple_lookup()

    # Build reference word set
    ref_corpus = load_reference_corpus(verbose=False)
    ref_tokens = ref_corpus.get_combined_tokens('latin')
    original_word_set = set(w.lower() for w in ref_tokens if len(w) >= 2)

    if expanded_dict_used:
        ref_word_set, _ = build_expanded_word_set(original_word_set)
        print(f"  Using expanded dictionary: {len(ref_word_set):,} words")
    else:
        ref_word_set = original_word_set
        print(f"  Using original dictionary: {len(ref_word_set):,} words")

    inventory = build_phoneme_inventory('latin', ref_corpus)
    lm = build_ngram_lm(ref_tokens[:10000], order=3, smoothing=0.01)
    all_syls = build_cv_syllable_table('latin')

    # Glyph frequencies
    glyph_freq: Counter = Counter()
    for token in tokens:
        for ch in tokenize_eva_chars(token):
            glyph_freq[ch] += 1

    # Load anchors
    rosetta_path = os.path.join(rd, 'rosetta_selection.json')
    anchors: List[AnchorConstraint] = []
    if os.path.exists(rosetta_path):
        with open(rosetta_path) as f:
            rosetta_data = json.load(f)
        anchors = _build_anchor_constraints_triple(rosetta_data, eva_to_triple)

    # Random baseline
    variables_keys = list(current_assignment.keys())
    random_baseline = _compute_random_baseline(
        variables_keys, all_syls, tokens, eva_to_triple, ref_word_set,
    )

    # ─── 3a: Extract high-confidence hits ───
    print("\n  3a: Extracting high-confidence hits ...")
    all_hits = extract_high_confidence_hits(
        current_assignment, tokens, eva_to_triple, ref_word_set, min_frequency=2,
    )
    confirmed_words = sorted(set(h.decoded_word for h in all_hits))
    print(f"      Confirmed hits: {len(all_hits)} ({len(confirmed_words)} unique words)")
    if confirmed_words[:20]:
        print(f"      Words: {confirmed_words[:20]}")

    # ─── 3b: Constraint coverage ───
    print("\n  3b: Computing constraint coverage ...")
    coverage = compute_constraint_coverage(all_hits, n_total_triples=len(current_assignment))
    print(f"      Triples constrained: {coverage.n_triples_constrained}/{coverage.n_triples_total}")
    print(f"      Contradictions: {coverage.n_contradictions}")

    # Remove contradictory hits (keep the highest-frequency one per triple)
    if coverage.n_contradictions > 0:
        # For contradicted triples, keep only hits from the most frequent token
        contradicted_triples = set()
        for c in coverage.contradiction_details:
            contradicted_triples.add(c['triple_key'])

        token_freq = Counter(tokens)
        filtered_hits: List[HitConstraint] = []
        for hit in all_hits:
            has_contradiction = any(tk in contradicted_triples for tk in hit.triple_keys)
            if not has_contradiction:
                filtered_hits.append(hit)
            else:
                # Keep it but lower confidence
                hit_copy = HitConstraint(
                    voynich_token=hit.voynich_token,
                    decoded_word=hit.decoded_word,
                    triple_keys=hit.triple_keys,
                    target_syllables=hit.target_syllables,
                    confidence=0.5,
                )
                filtered_hits.append(hit_copy)
        all_hits = filtered_hits

    # ─── 3c-3d: Iterative loop ───
    print("\n  3c-3d: Iterative re-solving loop ...")
    iterations: List[IterationRecord] = []
    max_iterations = 10
    converged_at = 0
    convergence_reason = 'max_iter'
    prev_dict_hit = baseline_dict_hit

    for iteration in range(max_iterations):
        print(f"\n      Iteration {iteration} ...")

        # Build the fixed mapping from hit constraints
        fixed_mapping: Dict[str, str] = {}
        for hit in all_hits:
            for tk, syl in zip(hit.triple_keys, hit.target_syllables):
                if tk not in fixed_mapping:
                    fixed_mapping[tk] = syl

        # Build fresh variables — only FREE (unconstrained) triples
        hypothesis_map = build_triple_phoneme_hypotheses('latin', all_syls)
        all_variables = build_feature_variables(eva_to_triple, glyph_freq, inventory, hypothesis_map)
        all_variables = initialise_feature_domains(all_variables, inventory, hypothesis_map, anchors)

        free_variables = [v for v in all_variables if v.cell_key not in fixed_mapping]
        n_free = len(free_variables)
        n_fixed = len(fixed_mapping)
        print(f"        Fixed: {n_fixed}, Free: {n_free}")

        if not free_variables:
            # All triples are constrained — just score the fixed mapping
            decoded = decode_corpus(tokens, fixed_mapping, eva_to_triple, max_tokens=2000)
            hits_count = sum(1 for w in decoded if w in ref_word_set)
            current_dict_hit = hits_count / len(decoded) if decoded else 0.0
            selectivity = current_dict_hit / max(random_baseline, 0.001)
            # Create a mock "best" solution
            current_assignment = dict(fixed_mapping)
        else:
            # AC-3 only on free variables (no all-different conflict with fixed)
            solvable, free_variables = ac3_propagate(free_variables)
            if not solvable:
                print("        Free variables unsolvable after AC-3")
                convergence_reason = 'unsolvable'
                converged_at = iteration
                break

            # Merge fixed mapping into a complete assignment for scoring
            # by pre-seeding the assignment. beam_search will only vary
            # free variables.
            solutions = beam_search(
                variables=free_variables,
                lm=lm,
                voynich_tokens=tokens,
                eva_to_cell=eva_to_triple,
                anchors=anchors,
                inventory=inventory,
                ref_word_set=ref_word_set,
                beam_width=80,
                max_solutions=10,
                seed=42 + iteration,
            )

            if not solutions:
                print("        No solutions from beam search")
                convergence_reason = 'no_solutions'
                converged_at = iteration
                break

            # Merge fixed + free into complete assignment
            best = solutions[0]
            merged = dict(fixed_mapping)
            merged.update(best.mapping)
            current_assignment = merged

            # Re-score with the complete merged assignment
            decoded = decode_corpus(tokens, merged, eva_to_triple, max_tokens=2000)
            hits_count = sum(1 for w in decoded if w in ref_word_set)
            current_dict_hit = hits_count / len(decoded) if decoded else 0.0
            selectivity = current_dict_hit / max(random_baseline, 0.001)

        # Extract new hits from current (merged) assignment
        new_hits = extract_high_confidence_hits(
            current_assignment, tokens, eva_to_triple, ref_word_set, min_frequency=2,
        )
        existing_words = set(h.decoded_word for h in all_hits)
        new_hit_words = sorted(set(h.decoded_word for h in new_hits) - existing_words)

        n_constrained = compute_constraint_coverage(
            all_hits, len(current_assignment)
        ).n_triples_constrained

        record = IterationRecord(
            iteration=iteration,
            n_constrained_triples=n_constrained,
            n_free_triples=len(current_assignment) - n_constrained,
            dict_hit=round(current_dict_hit, 4),
            selectivity=round(selectivity, 2),
            n_new_hits=len(new_hit_words),
            new_hit_words=new_hit_words,
        )
        iterations.append(record)

        print(f"        dict_hit={current_dict_hit:.3f}, selectivity={selectivity:.2f}x, "
              f"new hits: {len(new_hit_words)}")

        # Convergence checks
        delta = current_dict_hit - prev_dict_hit
        if delta < 0.005 and iteration > 0:
            convergence_reason = 'plateau'
            converged_at = iteration
            break

        if selectivity < 1.5:
            convergence_reason = 'selectivity_drop'
            converged_at = iteration
            break

        # Add new hits to constraint set
        for nh in new_hits:
            if nh.decoded_word not in existing_words:
                all_hits.append(nh)

        # Update coverage check for contradictions
        new_coverage = compute_constraint_coverage(all_hits, len(current_assignment))
        if new_coverage.n_contradictions > len(current_assignment) * 0.1:
            convergence_reason = 'contradictions'
            converged_at = iteration
            current_assignment = best.mapping
            break

        prev_dict_hit = current_dict_hit
        current_assignment = best.mapping
        converged_at = iteration

    # Final scoring
    decoded = decode_corpus(tokens, current_assignment, eva_to_triple, max_tokens=2000)
    final_hits = sum(1 for w in decoded if w in ref_word_set)
    final_dict_hit = final_hits / len(decoded) if decoded else 0.0
    final_selectivity = final_dict_hit / max(random_baseline, 0.001)

    gate_passed = final_dict_hit >= baseline_dict_hit

    elapsed = time.time() - t0

    verdict = (
        f"Iterative re-solving: {final_dict_hit:.1%} dict_hit "
        f"({final_selectivity:.2f}x selectivity). "
        f"Converged at iteration {converged_at} ({convergence_reason}). "
        f"Baseline was {baseline_dict_hit:.1%}."
    )

    result = IterativeHitsResult(
        n_confirmed_hits=len(all_hits),
        confirmed_hit_words=sorted(set(h.decoded_word for h in all_hits)),
        expanded_dict_used=expanded_dict_used,
        coverage=asdict(coverage),
        iterations=[asdict(r) for r in iterations],
        converged_at=converged_at,
        convergence_reason=convergence_reason,
        final_dict_hit=round(final_dict_hit, 4),
        final_selectivity=round(final_selectivity, 2),
        final_assignment=current_assignment,
        gate_passed=gate_passed,
        verdict=verdict,
        runtime_seconds=round(elapsed, 2),
    )

    out_path = os.path.join(rd, 'iterative_hits.json')
    with open(out_path, 'w') as f:
        json.dump(asdict(result), f, indent=2, default=_convert)

    print(f"\n  Gate: {'PASS' if gate_passed else 'FAIL'}")
    print(f"  {verdict}")
    print(f"\n  → {out_path}")
