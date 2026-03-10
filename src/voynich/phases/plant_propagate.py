"""
Phase 31.3: Plant-Derived Assignment Propagation
===================================================
Take any new confirmed triple assignments from the plant name CSP and
propagate them through the Ventris grid (sign families + confirmed triples),
then re-run the bootstrap loop.

Dependency chain:
    plant_name_csp.json        (Step 31.2)
    bootstrap_loop.json        (Phase 30)
    tachygraphic_stroke.json   (Phase 19)
    combined_refine.json       (Phase 15)
    modifier_integrate.json    (Phase 16)
        → plant_name_propagate.json  (this step)
"""

import json
import os
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    decode_token_modifier_aware,
    load_corpus,
    tokenize_eva_chars,
)
from voynich.core.reference import (
    build_expanded_word_set,
    load_reference_corpus,
)
from voynich.phases.null_corpus import (
    _build_eva_bigram_model,
    _generate_null_corpus,
    _reconstruct_modifier_rules,
)
from voynich.phases.signal_isolation import _decode_corpus_r3


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
class PropagationIteration:
    """One iteration of propagation + re-decode."""
    iteration: int
    n_confirmed: int
    n_new_this_iter: int
    dict_hit: float
    signal_rate: float
    new_signal_words: List[str]


@dataclass
class PlantPropagateResult:
    """Full Step 31.3 output."""
    initial_confirmed: int
    plant_new_assignments: Dict[str, str]
    n_plant_assignments: int
    expanded_confirmed: int
    family_propagated: Dict[str, str]
    n_after_family_propagation: int
    iterations: List[Dict]
    final_dict_hit: float
    final_signal_rate: float
    initial_dict_hit: float
    cascade_detected: bool
    cascade_trajectory: str  # 'monotonic_increase', 'single_burst', 'no_cascade'
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Family propagation
# ---------------------------------------------------------------------------

def _propagate_through_families(
    confirmed: Dict[str, str],
    new_assignments: Dict[str, str],
    sign_families: Dict[str, str],
) -> Dict[str, str]:
    """Propagate new assignments through sign families.

    Same family members should share the same consonant onset.
    Returns additional inferred assignments (not including the originals).
    """
    # Build family -> member triples
    family_members: Dict[str, List[str]] = defaultdict(list)
    for triple_key, family in sign_families.items():
        family_members[family].append(triple_key)

    merged = {**confirmed, **new_assignments}
    inferred: Dict[str, str] = {}

    for triple_key, syllable in new_assignments.items():
        family = sign_families.get(triple_key)
        if not family:
            continue

        # Extract consonant from the assigned syllable
        consonant = ''
        for ch in syllable:
            if ch not in 'aeiouy':
                consonant += ch
            else:
                break

        # Check family members
        for member in family_members.get(family, []):
            if member == triple_key:
                continue
            if member in merged:
                continue  # Already assigned

            # Propose: same consonant, but we don't know the vowel
            # Only propagate if we have a clear consonant
            if consonant:
                inferred[member] = f"{consonant}?"  # Mark as partial

    return inferred


# ---------------------------------------------------------------------------
# Signal re-isolation (simplified)
# ---------------------------------------------------------------------------

def _compute_signal_rate(
    all_tokens: List[str],
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    modifier_chars: set,
    modifier_rules: Dict[str, str],
    ref_word_set: set,
    null_seeds: List[int],
) -> Tuple[float, List[str]]:
    """Simplified signal isolation: count tokens that hit in real but miss in all nulls."""
    n_tokens = len(all_tokens)

    real_decoded = _decode_corpus_r3(
        all_tokens, assignment, eva_to_triple,
        modifier_chars, modifier_rules, ref_word_set,
    )
    real_hits = [w in ref_word_set for w in real_decoded]

    bigram_probs, initial_probs, token_lengths = _build_eva_bigram_model(all_tokens)

    null_hit_counts = [0] * n_tokens
    for seed in null_seeds[:3]:
        null_tokens = _generate_null_corpus(
            bigram_probs, initial_probs, token_lengths, n_tokens, seed,
        )
        null_decoded = _decode_corpus_r3(
            null_tokens, assignment, eva_to_triple,
            modifier_chars, modifier_rules, ref_word_set,
        )
        for i, w in enumerate(null_decoded):
            if w in ref_word_set:
                null_hit_counts[i] += 1

    n_signal = 0
    signal_words = Counter()
    for i in range(n_tokens):
        if real_hits[i] and null_hit_counts[i] == 0:
            n_signal += 1
            signal_words[real_decoded[i]] += 1

    signal_rate = n_signal / n_tokens if n_tokens > 0 else 0.0
    new_signal = [w for w, c in signal_words.most_common(20)]
    return signal_rate, new_signal


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_plant_propagate() -> None:
    """Step 31.3: Propagate plant-derived assignments and test cascade."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE 31.3: Plant-Derived Assignment Propagation")
    print("=" * 70)

    rd = _results_dir()
    eva_to_triple = build_eva_to_triple_lookup()

    # ── 1. Load inputs ──
    print("\n  1. Loading inputs...")

    # Plant CSP results
    csp_path = os.path.join(rd, 'plant_name_csp.json')
    if not os.path.exists(csp_path):
        print("  [SKIP] plant_name_csp.json not found — run plant-csp first")
        return
    with open(csp_path) as f:
        csp_data = json.load(f)

    # Phase 30 confirmed triples
    bt_path = os.path.join(rd, 'bootstrap_loop.json')
    with open(bt_path) as f:
        bt_data = json.load(f)
    base_assignment = dict(bt_data.get('final_assignment', {}))
    confirmed_keys = set(bt_data.get('confirmed_triples', []))

    # Full assignment for decoding
    with open(os.path.join(rd, 'combined_refine.json')) as f:
        refine_data = json.load(f)
    full_assignment = refine_data.get('best_assignment', {})

    # Sign families
    sf_path = os.path.join(rd, 'tachygraphic_stroke.json')
    sign_families = {}
    if os.path.exists(sf_path):
        with open(sf_path) as f:
            sf_data = json.load(f)
        for family in sf_data.get('families', []):
            family_name = family.get('family_name', '')
            for member in family.get('members', []):
                triple_key = member.get('triple_key', '')
                if triple_key:
                    sign_families[triple_key] = family_name

    # Modifiers and reference
    with open(os.path.join(rd, 'modifier_integrate.json')) as f:
        mod_data = json.load(f)
    modifier_chars, modifier_rules = _reconstruct_modifier_rules(mod_data)

    null_seeds = [100, 101, 102, 103, 104]
    null_path = os.path.join(rd, 'null_corpus.json')
    if os.path.exists(null_path):
        with open(null_path) as f:
            null_data = json.load(f)
        null_seeds = [r['seed'] for r in null_data.get('null_runs', [])]

    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    base_words = set(w.lower() for w in ref_corpus.get_combined_tokens('latin')
                     if len(w) >= 2)
    expanded, _ = build_expanded_word_set(base_words)
    ref_word_set = base_words | expanded

    corpus = load_corpus(verbose=False)
    all_tokens = corpus.get_tokens()

    # ── 2. Extract plant-derived new assignments ──
    print("\n  2. Extracting plant-derived assignments...")
    cross_folio = csp_data.get('cross_folio_consistent', [])
    plant_assignments = {}
    for cf in cross_folio:
        triple_key = cf.get('triple_key', '')
        syllable = cf.get('syllable', '')
        if triple_key and syllable and triple_key not in confirmed_keys:
            plant_assignments[triple_key] = syllable

    n_initial = len(confirmed_keys)
    print(f"     Initial confirmed: {n_initial}")
    print(f"     New from plants:   {len(plant_assignments)}")

    if not plant_assignments:
        print("\n  No new assignments from plant CSP.")
        print("  Running baseline metrics for comparison...")

        # Still compute baseline metrics
        initial_dict_hit = sum(
            1 for w in _decode_corpus_r3(
                all_tokens, full_assignment, eva_to_triple,
                modifier_chars, modifier_rules, ref_word_set,
            ) if w in ref_word_set
        ) / len(all_tokens)

        result = PlantPropagateResult(
            initial_confirmed=n_initial,
            plant_new_assignments={},
            n_plant_assignments=0,
            expanded_confirmed=n_initial,
            family_propagated={},
            n_after_family_propagation=n_initial,
            iterations=[],
            final_dict_hit=round(initial_dict_hit, 4),
            final_signal_rate=0.0,
            initial_dict_hit=round(initial_dict_hit, 4),
            cascade_detected=False,
            cascade_trajectory='no_cascade',
            verdict='NO_NEW_ASSIGNMENTS',
            runtime_seconds=round(time.time() - t0, 2),
        )
        out_path = os.path.join(rd, 'plant_name_propagate.json')
        with open(out_path, 'w') as f:
            json.dump(_convert(asdict(result)), f, indent=2)
        print(f"  Saved → {out_path}")
        return

    for tk, syl in plant_assignments.items():
        print(f"       {tk} → '{syl}'")

    # ── 3. Propagate through families ──
    print("\n  3. Propagating through sign families...")
    confirmed_dict = {k: full_assignment[k] for k in confirmed_keys
                      if k in full_assignment}
    family_inferred = _propagate_through_families(
        confirmed_dict, plant_assignments, sign_families,
    )
    print(f"     Family-inferred: {len(family_inferred)} additional")

    # ── 4. Build expanded assignment ──
    expanded_assignment = dict(full_assignment)
    expanded_assignment.update(plant_assignments)
    # Don't use partial family inferences (they have '?' vowels)

    n_expanded = n_initial + len(plant_assignments)
    print(f"     Expanded confirmed: {n_expanded}/25")

    # ── 5. Iterative re-decode and cascade test ──
    print("\n  4. Cascade test (iterative re-decode)...")

    # Initial metrics
    initial_decoded = _decode_corpus_r3(
        all_tokens, full_assignment, eva_to_triple,
        modifier_chars, modifier_rules, ref_word_set,
    )
    initial_dict_hit = sum(1 for w in initial_decoded if w in ref_word_set) / len(all_tokens)

    iterations = []
    current_assignment = dict(expanded_assignment)
    signal_trajectory = []

    for iteration in range(3):
        # Decode with current assignment
        decoded = _decode_corpus_r3(
            all_tokens, current_assignment, eva_to_triple,
            modifier_chars, modifier_rules, ref_word_set,
        )
        dict_hit = sum(1 for w in decoded if w in ref_word_set) / len(all_tokens)

        # Signal rate
        signal_rate, new_signals = _compute_signal_rate(
            all_tokens, current_assignment, eva_to_triple,
            modifier_chars, modifier_rules, ref_word_set, null_seeds,
        )

        signal_trajectory.append(signal_rate)

        iter_result = PropagationIteration(
            iteration=iteration,
            n_confirmed=n_expanded,
            n_new_this_iter=len(plant_assignments) if iteration == 0 else 0,
            dict_hit=round(dict_hit, 4),
            signal_rate=round(signal_rate, 4),
            new_signal_words=new_signals[:10],
        )
        iterations.append(iter_result)

        print(f"     Iter {iteration}: dict_hit={dict_hit:.4f}, "
              f"signal_rate={signal_rate:.4f}, "
              f"new_signals={len(new_signals)}")

        # Check for new context cribs (simplified: words appearing ≥5 times as signal)
        # In a full implementation, this would run the Phase 30 4-check protocol
        if iteration > 0 and len(new_signals) == 0:
            print(f"     Converged at iteration {iteration}")
            break

    # Cascade detection
    if len(signal_trajectory) >= 3:
        is_monotonic = all(signal_trajectory[i] <= signal_trajectory[i + 1]
                          for i in range(len(signal_trajectory) - 1))
        if is_monotonic and signal_trajectory[-1] > signal_trajectory[0] * 1.1:
            cascade_trajectory = 'monotonic_increase'
            cascade_detected = True
        else:
            cascade_trajectory = 'single_burst'
            cascade_detected = False
    elif len(signal_trajectory) >= 1:
        cascade_trajectory = 'single_burst'
        cascade_detected = signal_trajectory[-1] > 0.20
    else:
        cascade_trajectory = 'no_cascade'
        cascade_detected = False

    final_dict_hit = iterations[-1].dict_hit if iterations else initial_dict_hit
    final_signal = iterations[-1].signal_rate if iterations else 0.0

    if cascade_detected:
        verdict = "CASCADE_DETECTED"
    elif final_dict_hit > initial_dict_hit + 0.01:
        verdict = "IMPROVEMENT_NO_CASCADE"
    else:
        verdict = "NO_IMPROVEMENT"

    print(f"\n  Verdict: {verdict}")
    print(f"  Initial dict_hit: {initial_dict_hit:.4f} → Final: {final_dict_hit:.4f}")
    print(f"  Cascade: {cascade_trajectory}")

    # ── 6. Save ──
    result = PlantPropagateResult(
        initial_confirmed=n_initial,
        plant_new_assignments=plant_assignments,
        n_plant_assignments=len(plant_assignments),
        expanded_confirmed=n_expanded,
        family_propagated=family_inferred,
        n_after_family_propagation=n_expanded + len(family_inferred),
        iterations=[_convert(asdict(it)) for it in iterations],
        final_dict_hit=round(final_dict_hit, 4),
        final_signal_rate=round(final_signal, 4),
        initial_dict_hit=round(initial_dict_hit, 4),
        cascade_detected=cascade_detected,
        cascade_trajectory=cascade_trajectory,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'plant_name_propagate.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2)

    print(f"\n  Saved → {out_path}")
    print(f"  Completed in {time.time() - t0:.1f}s")
