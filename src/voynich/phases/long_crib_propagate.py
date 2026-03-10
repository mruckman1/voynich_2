"""
Phase 33.12 -- Long Crib Propagation
========================================
If Step 33.11 produced new confirmed triples from botanical crib
alignments, propagate them through sign families and re-run the signal
pipeline to test for cascade effects.

Dependency chain:
    long_crib_csp.json         (Step 33.11 -- crib-derived assignments)
    combined_refine.json       (Phase 15 assignment)
    modifier_integrate.json    (Phase 16 modifiers)
    null_corpus.json           (Phase 17 seeds)
    signal_bigrams.json        (Phase 29.1 baseline)
    signal_isolation.json      (Phase 28.4 baseline)
        -> long_crib_propagate.json  (this step)
"""

import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import build_eva_to_triple_lookup, load_corpus
from voynich.core.reference import (
    build_expanded_word_set,
    load_reference_corpus,
    EVA_VISUAL_COMPONENTS,
)
from voynich.phases.csp_solver import decode_token
from voynich.phases.signal_isolation import _decode_corpus_r3
from voynich.phases.null_corpus import (
    _build_eva_bigram_model,
    _generate_null_corpus,
    _reconstruct_modifier_rules,
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
class LongCribPropagateResult:
    # New assignments
    n_new_assignments: int
    new_assignments: Dict[str, str]
    n_family_propagated: int
    family_propagated: Dict[str, str]
    propagated_assignment: Dict[str, str]  # full 25-triple table
    # Signal pipeline
    n_tokens: int
    dict_hit: float
    signal_rate: float
    n_signal: int
    n_anti_signal: int
    bigram_z: float
    n_signal_pairs: int
    n_bigram_hits: int
    # Baseline comparison
    baseline_dict_hit: float
    baseline_signal_rate: float
    baseline_bigram_z: float
    delta_dict_hit: float
    delta_signal_rate: float
    delta_bigram_z: float
    # Cascade test
    cascade_shape: str  # 'accelerating', 'linear', 'decelerating', 'stalled'
    confirmed_before: int
    confirmed_after: int
    # Verdict
    verdict: str  # 'CASCADE_FOUND', 'MARGINAL_IMPROVEMENT', 'NO_IMPROVEMENT'
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Sign family helpers
# ---------------------------------------------------------------------------

def _build_family_index() -> Dict[str, List[str]]:
    """Map first_stroke -> list of triple keys that share it.

    Triples sharing the same first_stroke belong to the same "sign family"
    and should share the same onset consonant.
    """
    family: Dict[str, List[str]] = defaultdict(list)
    seen_triples: Set[str] = set()

    for glyph, components in EVA_VISUAL_COMPONENTS.items():
        tk = (components['first_stroke'] + ',' +
              components['last_stroke'] + ',' +
              components['glyph_class'])
        if tk not in seen_triples:
            seen_triples.add(tk)
            family[components['first_stroke']].append(tk)

    return dict(family)


def _extract_consonant(syllable: str) -> str:
    """Extract onset consonant(s) from a CV syllable.

    Examples: 'ro' -> 'r', 'de' -> 'd', 'a' -> '', 'sca' -> 'sc'.
    """
    vowels = set('aeiou')
    for i, ch in enumerate(syllable):
        if ch in vowels:
            return syllable[:i]
    return syllable


def _propagate_families(
    assignment: Dict[str, str],
    new_assignments: Dict[str, str],
    confirmed_triples: Set[str],
) -> Dict[str, str]:
    """Propagate consonant constraints through sign families.

    If a new assignment assigns triple T1 = 'ro' (consonant 'r'), then
    other triples in the same family (same first_stroke) that are
    unconfirmed may be constrained to consonant 'r'.  We only propagate
    when there is no conflict with existing assignments.

    Returns a dict of triple_key -> syllable for newly propagated triples.
    """
    family_index = _build_family_index()

    # Build reverse: triple -> first_stroke
    triple_to_family: Dict[str, str] = {}
    for fs, triples in family_index.items():
        for tk in triples:
            triple_to_family[tk] = fs

    propagated: Dict[str, str] = {}

    for new_triple, new_syllable in new_assignments.items():
        new_consonant = _extract_consonant(new_syllable)
        if not new_consonant:
            continue  # pure vowel -- no consonant to propagate

        family_key = triple_to_family.get(new_triple)
        if not family_key:
            continue

        siblings = family_index.get(family_key, [])
        for sibling in siblings:
            if sibling == new_triple:
                continue
            if sibling in confirmed_triples:
                continue  # already locked
            if sibling in new_assignments:
                continue  # already being set by the crib itself

            current_syl = assignment.get(sibling, '')
            if not current_syl:
                continue

            current_consonant = _extract_consonant(current_syl)
            if current_consonant == new_consonant:
                continue  # already agrees -- no action needed

            # Build replacement syllable: swap consonant, keep vowel
            vowels = set('aeiou')
            vowel_part = ''
            for i, ch in enumerate(current_syl):
                if ch in vowels:
                    vowel_part = current_syl[i:]
                    break

            if not vowel_part:
                continue

            proposed = new_consonant + vowel_part

            # Check for conflict: if sibling is already propagated with
            # a different consonant, skip
            if sibling in propagated:
                existing_consonant = _extract_consonant(propagated[sibling])
                if existing_consonant != new_consonant:
                    continue  # conflicting propagation -- skip

            propagated[sibling] = proposed

    return propagated


# ---------------------------------------------------------------------------
# Signal classification
# ---------------------------------------------------------------------------

def _classify_tokens(
    real_hits: List[bool],
    null_hits_list: List[List[bool]],
    n_tokens: int,
) -> List[str]:
    """Classify each token as SIGNAL/SHARED_HIT/SHARED_MISS/ANTI_SIGNAL."""
    classifications: List[str] = []
    for idx in range(n_tokens):
        r_hit = real_hits[idx]
        null_hit_count = sum(1 for nh in null_hits_list if nh[idx])
        if r_hit and null_hit_count <= 1:
            classifications.append('SIGNAL')
        elif r_hit and null_hit_count >= 3:
            classifications.append('SHARED_HIT')
        elif not r_hit and null_hit_count >= 3:
            classifications.append('ANTI_SIGNAL')
        else:
            classifications.append('SHARED_MISS')
    return classifications


# ---------------------------------------------------------------------------
# Signal-pair bigram helpers
# ---------------------------------------------------------------------------

def _find_signal_pairs(
    classifications: List[str],
    decoded: List[str],
    folios: List[str],
) -> List[Tuple[str, int, str, str]]:
    """Find consecutive SIGNAL-SIGNAL pairs within the same folio.

    Returns list of (folio, position_i, word_i, word_i+1).
    """
    pairs: List[Tuple[str, int, str, str]] = []
    for i in range(len(classifications) - 1):
        if (classifications[i] == 'SIGNAL'
                and classifications[i + 1] == 'SIGNAL'
                and folios[i] == folios[i + 1]):
            pairs.append((folios[i], i, decoded[i], decoded[i + 1]))
    return pairs


def _build_reference_bigrams(
    ref_tokens: List[str],
) -> Set[Tuple[str, str]]:
    """Build word-level bigram set from reference corpus tokens."""
    bigrams: Set[Tuple[str, str]] = set()
    for i in range(len(ref_tokens) - 1):
        bigrams.add((ref_tokens[i], ref_tokens[i + 1]))
    return bigrams


def _null_bigram_permutation_test(
    n_signal: int,
    n_tokens: int,
    decoded: List[str],
    folios: List[str],
    ref_bigrams: Set[Tuple[str, str]],
    n_perms: int = 1000,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Random relabeling null test for bigram hit rate.

    Returns (observed_rate, null_mean, null_std).
    """
    rng = random.Random(seed)
    indices = list(range(n_tokens))
    null_rates: List[float] = []

    for _ in range(n_perms):
        fake_signal = set(rng.sample(indices, min(n_signal, n_tokens)))
        n_pairs = 0
        n_hits = 0
        for i in range(n_tokens - 1):
            if (i in fake_signal and (i + 1) in fake_signal
                    and folios[i] == folios[i + 1]):
                n_pairs += 1
                if (decoded[i], decoded[i + 1]) in ref_bigrams:
                    n_hits += 1
        rate = n_hits / n_pairs if n_pairs > 0 else 0.0
        null_rates.append(rate)

    null_mean = sum(null_rates) / len(null_rates) if null_rates else 0.0
    null_var = (
        sum((r - null_mean) ** 2 for r in null_rates) / len(null_rates)
        if null_rates else 0.0
    )
    null_std = null_var ** 0.5
    return null_mean, null_std, null_rates


# ---------------------------------------------------------------------------
# Bootstrap iteration helper
# ---------------------------------------------------------------------------

def _bootstrap_one_pass(
    decoded: List[str],
    classifications: List[str],
    folios: List[str],
    assignment: Dict[str, str],
    confirmed_triples: Set[str],
    ref_word_set: set,
    eva_to_triple: Dict[str, str],
    all_tokens: List[str],
) -> Tuple[List[str], Dict[str, str]]:
    """One bootstrap iteration: find new confirmed words from SIGNAL runs.

    Returns (new_words, new_triple_assignments).
    """
    # Find SIGNAL runs of length >= 3
    new_words: List[str] = []
    new_triple_assignments: Dict[str, str] = {}

    n = len(decoded)
    run_start = -1
    in_run = False

    for i in range(n):
        if classifications[i] == 'SIGNAL':
            if not in_run:
                run_start = i
                in_run = True
        else:
            if in_run and (i - run_start) >= 3:
                # Process this run
                for j in range(run_start, i):
                    word = decoded[j]
                    if word in ref_word_set and len(word) >= 3:
                        # Check signal position rate >= 50%
                        total = sum(1 for k in range(n) if decoded[k] == word)
                        sig = sum(1 for k in range(n)
                                  if decoded[k] == word
                                  and classifications[k] == 'SIGNAL')
                        if total > 0 and sig / total >= 0.5:
                            if word not in new_words:
                                new_words.append(word)
            in_run = False

    # Handle run at end of corpus
    if in_run and (n - run_start) >= 3:
        for j in range(run_start, n):
            word = decoded[j]
            if word in ref_word_set and len(word) >= 3:
                total = sum(1 for k in range(n) if decoded[k] == word)
                sig = sum(1 for k in range(n)
                          if decoded[k] == word
                          and classifications[k] == 'SIGNAL')
                if total > 0 and sig / total >= 0.5:
                    if word not in new_words:
                        new_words.append(word)

    # For each new word, extract triple -> syllable proposals
    from voynich.core.corpus import token_to_triples

    for word in new_words:
        # Find EVA tokens that decode to this word
        matching_evas: List[str] = []
        for idx, d in enumerate(decoded):
            if d == word:
                matching_evas.append(all_tokens[idx])
        if not matching_evas:
            continue
        best_eva = Counter(matching_evas).most_common(1)[0][0]
        triples = token_to_triples(best_eva, eva_to_triple)
        for t in triples:
            if t not in confirmed_triples:
                syl = assignment.get(t, '')
                if syl:
                    new_triple_assignments[t] = syl

    return new_words, new_triple_assignments


# ---------------------------------------------------------------------------
# Cascade shape classification
# ---------------------------------------------------------------------------

def _classify_cascade(
    signal_rates: List[float],
) -> str:
    """Classify the trajectory of signal rates across iterations.

    Returns one of 'accelerating', 'linear', 'decelerating', 'stalled'.
    """
    if len(signal_rates) < 2:
        return 'stalled'

    deltas = [signal_rates[i] - signal_rates[i - 1]
              for i in range(1, len(signal_rates))]

    if all(d <= 0.001 for d in deltas):
        return 'stalled'

    if len(deltas) >= 2:
        # Compare early vs late deltas
        mid = len(deltas) // 2
        early_mean = sum(deltas[:mid]) / mid if mid > 0 else 0.0
        late_mean = sum(deltas[mid:]) / (len(deltas) - mid) if (len(deltas) - mid) > 0 else 0.0

        if late_mean > early_mean * 1.5 and late_mean > 0.001:
            return 'accelerating'
        elif early_mean > late_mean * 1.5 and early_mean > 0.001:
            return 'decelerating'

    if any(d > 0.001 for d in deltas):
        return 'linear'

    return 'stalled'


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_long_crib_propagate() -> None:
    """Step 33.12: Propagate crib-derived triples through sign families."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE 33.12: Long Crib Propagation")
    print("=" * 70)

    rd = str(_results_dir())
    eva_to_triple = build_eva_to_triple_lookup()

    # ── 1. Load Step 33.11 results ──
    print("\n  1. Loading long_crib_csp.json (Step 33.11) ...")

    csp_path = os.path.join(rd, 'long_crib_csp.json')
    if not os.path.exists(csp_path):
        print("  [SKIP] long_crib_csp.json not found -- run long-crib-csp first")
        return
    with open(csp_path) as f:
        csp_data = json.load(f)

    n_new = csp_data.get('n_new_confirmed_triples', 0)
    new_assignments_raw = csp_data.get('new_confirmed_assignments', {})

    print(f"     New confirmed triples from crib CSP: {n_new}")

    if n_new == 0:
        # No new triples -- save minimal result and exit
        print("     No new triples to propagate. Recording NO_NEW_TRIPLES.")

        result = LongCribPropagateResult(
            n_new_assignments=0,
            new_assignments={},
            n_family_propagated=0,
            family_propagated={},
            propagated_assignment={},
            n_tokens=0,
            dict_hit=0.0,
            signal_rate=0.0,
            n_signal=0,
            n_anti_signal=0,
            bigram_z=0.0,
            n_signal_pairs=0,
            n_bigram_hits=0,
            baseline_dict_hit=0.0,
            baseline_signal_rate=0.0,
            baseline_bigram_z=0.0,
            delta_dict_hit=0.0,
            delta_signal_rate=0.0,
            delta_bigram_z=0.0,
            cascade_shape='stalled',
            confirmed_before=0,
            confirmed_after=0,
            verdict='NO_NEW_TRIPLES',
            runtime_seconds=round(time.time() - t0, 2),
        )

        out_path = os.path.join(rd, 'long_crib_propagate.json')
        with open(out_path, 'w') as f:
            json.dump(_convert(result), f, indent=2)
        print(f"\n  -> {out_path}")
        return

    for triple, syl in sorted(new_assignments_raw.items()):
        print(f"       {triple} = '{syl}'")

    # ── 2. Load current assignment, modifiers, null seeds ──
    print("\n  2. Loading current assignment and pipeline inputs ...")

    refine_path = os.path.join(rd, 'combined_refine.json')
    if not os.path.exists(refine_path):
        print("  [SKIP] combined_refine.json not found")
        return
    with open(refine_path) as f:
        refine_data = json.load(f)
    assignment = dict(refine_data.get('best_assignment', {}))

    mod_path = os.path.join(rd, 'modifier_integrate.json')
    if not os.path.exists(mod_path):
        print("  [SKIP] modifier_integrate.json not found")
        return
    with open(mod_path) as f:
        mod_data = json.load(f)
    modifier_chars, modifier_rules = _reconstruct_modifier_rules(mod_data)

    null_seeds = [100, 101, 102, 103, 104]
    null_path = os.path.join(rd, 'null_corpus.json')
    if os.path.exists(null_path):
        with open(null_path) as f:
            null_data = json.load(f)
        null_seeds = [r['seed'] for r in null_data.get('null_runs', [])]

    # Count confirmed triples before propagation
    crib_path = os.path.join(rd, 'crib_extraction.json')
    confirmed_triples: Set[str] = set()
    if os.path.exists(crib_path):
        with open(crib_path) as f:
            crib_data = json.load(f)
        confirmed_triples = set(crib_data.get('all_triples_covered', []))

    # Also include bootstrap-confirmed triples
    boot_path = os.path.join(rd, 'bootstrap_loop.json')
    if os.path.exists(boot_path):
        with open(boot_path) as f:
            boot_data = json.load(f)
        confirmed_triples |= set(boot_data.get('confirmed_triples', []))

    confirmed_before = len(confirmed_triples)
    print(f"     Assignment: {len(assignment)} triples")
    print(f"     Modifiers: {len(modifier_chars)} chars")
    print(f"     Null seeds: {null_seeds}")
    print(f"     Confirmed triples before: {confirmed_before}")

    # ── 3. Apply new crib-derived assignments ──
    print("\n  3. Applying new crib-derived triple assignments ...")

    new_assignments: Dict[str, str] = {}
    for triple, syl in new_assignments_raw.items():
        old_syl = assignment.get(triple, '')
        assignment[triple] = syl
        confirmed_triples.add(triple)
        new_assignments[triple] = syl
        if old_syl and old_syl != syl:
            print(f"     CHANGED: {triple}: '{old_syl}' -> '{syl}'")
        else:
            print(f"     SET: {triple} = '{syl}'")

    print(f"     Applied {len(new_assignments)} new assignments")

    # ── 4. Family propagation ──
    print("\n  4. Propagating through sign families ...")

    family_propagated = _propagate_families(
        assignment, new_assignments, confirmed_triples,
    )

    if family_propagated:
        for triple, syl in sorted(family_propagated.items()):
            old = assignment.get(triple, '')
            print(f"     PROPAGATE: {triple}: '{old}' -> '{syl}'")
            assignment[triple] = syl
    else:
        print("     No family propagation candidates found")

    print(f"     Family-propagated: {len(family_propagated)} triples")

    # ── 5. Build expanded assignment and reference word set ──
    print("\n  5. Building reference word set ...")

    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    base_words = set(
        w.lower() for w in ref_corpus.get_combined_tokens('latin')
        if len(w) >= 2
    )
    expanded, _ = build_expanded_word_set(base_words)
    ref_word_set = base_words | expanded
    print(f"     {len(ref_word_set)} reference words")

    # Build reference bigrams for later
    ref_tokens = [w.lower() for w in ref_corpus.get_combined_tokens('latin')
                  if len(w) >= 2]
    ref_bigrams = _build_reference_bigrams(ref_tokens)
    print(f"     {len(ref_bigrams)} reference bigrams")

    # ── 6. Decode full corpus with expanded assignment ──
    print("\n  6. Decoding full corpus with propagated assignment ...")

    corpus = load_corpus(verbose=False)
    all_tokens: List[str] = []
    all_folios: List[str] = []
    for folio, page in corpus.pages.items():
        for token in page.all_tokens:
            all_tokens.append(token)
            all_folios.append(folio)
    n_tokens = len(all_tokens)

    real_decoded = _decode_corpus_r3(
        all_tokens, assignment, eva_to_triple,
        modifier_chars, modifier_rules, ref_word_set,
    )
    real_hits = [w in ref_word_set for w in real_decoded]
    dict_hit = sum(real_hits) / n_tokens if n_tokens > 0 else 0.0
    print(f"     {n_tokens} tokens, dict_hit = {dict_hit:.4f}")

    # ── 7. Regenerate 5 null corpora and decode ──
    print("\n  7. Regenerating and decoding 5 null corpora ...")

    bigram_probs, initial_probs, token_lengths = _build_eva_bigram_model(
        all_tokens,
    )

    null_hits_list: List[List[bool]] = []
    for i, seed in enumerate(null_seeds):
        print(f"     Null corpus {i + 1}/{len(null_seeds)} (seed={seed}) ...")
        null_tokens = _generate_null_corpus(
            bigram_probs, initial_probs, token_lengths, n_tokens, seed,
        )
        null_decoded = _decode_corpus_r3(
            null_tokens, assignment, eva_to_triple,
            modifier_chars, modifier_rules, ref_word_set,
        )
        null_hits = [w in ref_word_set for w in null_decoded]
        null_hits_list.append(null_hits)
        null_rate = sum(null_hits) / len(null_hits) if null_hits else 0.0
        print(f"       dict_hit = {null_rate:.4f}")

    # ── 8. Classify tokens ──
    print("\n  8. Classifying tokens (SIGNAL/SHARED/ANTI) ...")

    classifications = _classify_tokens(real_hits, null_hits_list, n_tokens)
    cls_counts = Counter(classifications)

    n_signal = cls_counts.get('SIGNAL', 0)
    n_shared_hit = cls_counts.get('SHARED_HIT', 0)
    n_shared_miss = cls_counts.get('SHARED_MISS', 0)
    n_anti_signal = cls_counts.get('ANTI_SIGNAL', 0)
    signal_rate = n_signal / n_tokens if n_tokens > 0 else 0.0

    print(f"     SIGNAL:      {n_signal:6d} ({signal_rate:.1%})")
    print(f"     SHARED_HIT:  {n_shared_hit:6d}")
    print(f"     SHARED_MISS: {n_shared_miss:6d}")
    print(f"     ANTI_SIGNAL: {n_anti_signal:6d}")

    # ── 9. Find SIGNAL-SIGNAL pairs and count bigram hits ──
    print("\n  9. Finding SIGNAL-SIGNAL pairs and bigram hits ...")

    signal_pairs = _find_signal_pairs(
        classifications, real_decoded, all_folios,
    )
    n_signal_pairs = len(signal_pairs)

    bigram_hits: List[Tuple[str, str]] = []
    for _, _, w1, w2 in signal_pairs:
        if (w1, w2) in ref_bigrams:
            bigram_hits.append((w1, w2))
    n_bigram_hits = len(bigram_hits)

    bigram_hit_rate = (
        n_bigram_hits / n_signal_pairs if n_signal_pairs > 0 else 0.0
    )
    print(f"     {n_signal_pairs} SIGNAL-SIGNAL pairs")
    print(f"     {n_bigram_hits} bigram hits (rate={bigram_hit_rate:.4f})")

    if bigram_hits:
        for w1, w2 in bigram_hits[:10]:
            print(f"       {w1} {w2}")

    # ── 10. Null permutation test for bigram z-score (1000 perms) ──
    print("\n  10. Null permutation test (1000 permutations) ...")

    null_mean, null_std, null_rates = _null_bigram_permutation_test(
        n_signal, n_tokens, real_decoded, all_folios,
        ref_bigrams, n_perms=1000, seed=42,
    )

    if null_std > 0:
        bigram_z = (bigram_hit_rate - null_mean) / null_std
    else:
        bigram_z = float('inf') if bigram_hit_rate > null_mean else 0.0

    p_value = sum(1 for r in null_rates if r >= bigram_hit_rate) / len(null_rates) if null_rates else 1.0

    print(f"     Null mean: {null_mean:.6f}, std: {null_std:.6f}")
    print(f"     z-score: {bigram_z:.2f}, p-value: {p_value:.4f}")

    # ── 11. Bootstrap iteration: check for further word confirmations ──
    print("\n  11. Running bootstrap iteration for cascade detection ...")

    signal_rate_trajectory: List[float] = [signal_rate]
    total_new_words: List[str] = []
    total_new_triple_assignments: Dict[str, str] = {}
    current_assignment = dict(assignment)
    current_decoded = list(real_decoded)
    current_classifications = list(classifications)
    current_confirmed = set(confirmed_triples)

    max_boot_iters = 3
    for boot_iter in range(1, max_boot_iters + 1):
        print(f"     Bootstrap iteration {boot_iter} ...")

        new_words, new_triple_assns = _bootstrap_one_pass(
            current_decoded, current_classifications, all_folios,
            current_assignment, current_confirmed, ref_word_set,
            eva_to_triple, all_tokens,
        )

        # Filter out already-seen words
        new_words = [w for w in new_words if w not in total_new_words]

        if not new_words:
            print(f"       No new words -- convergence at iteration {boot_iter}")
            break

        print(f"       New words: {', '.join(new_words)}")

        # Apply new triple assignments
        for t, syl in new_triple_assns.items():
            if t not in current_confirmed:
                current_assignment[t] = syl
                current_confirmed.add(t)
                total_new_triple_assignments[t] = syl

        total_new_words.extend(new_words)

        # Re-decode
        current_decoded = _decode_corpus_r3(
            all_tokens, current_assignment, eva_to_triple,
            modifier_chars, modifier_rules, ref_word_set,
        )
        new_real_hits = [w in ref_word_set for w in current_decoded]

        # Re-classify
        boot_null_hits: List[List[bool]] = []
        for seed in null_seeds:
            null_tokens = _generate_null_corpus(
                bigram_probs, initial_probs, token_lengths, n_tokens, seed,
            )
            null_decoded = _decode_corpus_r3(
                null_tokens, current_assignment, eva_to_triple,
                modifier_chars, modifier_rules, ref_word_set,
            )
            boot_null_hits.append([w in ref_word_set for w in null_decoded])

        current_classifications = _classify_tokens(
            new_real_hits, boot_null_hits, n_tokens,
        )
        new_signal_rate = sum(
            1 for c in current_classifications if c == 'SIGNAL'
        ) / n_tokens
        signal_rate_trajectory.append(new_signal_rate)
        print(f"       signal_rate: {new_signal_rate:.4f}")

    confirmed_after = len(current_confirmed)
    print(f"     Bootstrap words confirmed: {len(total_new_words)}")
    print(f"     Confirmed triples: {confirmed_before} -> {confirmed_after}")

    # ── 12. Cascade shape classification ──
    print("\n  12. Classifying cascade shape ...")

    cascade_shape = _classify_cascade(signal_rate_trajectory)
    print(f"     Signal rate trajectory: "
          f"{' -> '.join(f'{r:.4f}' for r in signal_rate_trajectory)}")
    print(f"     Cascade shape: {cascade_shape}")

    # ── 13. Compare to Phase 29 baseline ──
    print("\n  13. Comparing to Phase 29 baseline ...")

    # Load baseline from signal_bigrams.json (Phase 29.1)
    baseline_dict_hit = 0.0
    baseline_signal_rate = 0.0
    baseline_bigram_z = 0.0

    sig_bg_path = os.path.join(rd, 'signal_bigrams.json')
    if os.path.exists(sig_bg_path):
        with open(sig_bg_path) as f:
            bg_data = json.load(f)
        baseline_bigram_z = bg_data.get('bigram_z_score', 0.0)
        baseline_signal_rate = bg_data.get('signal_rate', 0.0)
        # dict_hit from signal_bigrams: reconstruct from token_dict_hits
        token_dh = bg_data.get('token_dict_hits', [])
        if token_dh:
            baseline_dict_hit = sum(1 for h in token_dh if h) / len(token_dh)

    # Fallback to signal_isolation.json
    sig_iso_path = os.path.join(rd, 'signal_isolation.json')
    if baseline_dict_hit == 0.0 and os.path.exists(sig_iso_path):
        with open(sig_iso_path) as f:
            sig_data = json.load(f)
        baseline_signal_rate = sig_data.get('signal_token_rate', baseline_signal_rate)

    # Fallback: compute baseline from Phase 15 assignment
    if baseline_dict_hit == 0.0:
        base_assignment = refine_data.get('best_assignment', {})
        base_decoded = _decode_corpus_r3(
            all_tokens, base_assignment, eva_to_triple,
            modifier_chars, modifier_rules, ref_word_set,
        )
        base_hits = [w in ref_word_set for w in base_decoded]
        baseline_dict_hit = sum(base_hits) / n_tokens if n_tokens > 0 else 0.0

    bigram_z_safe = bigram_z if bigram_z != float('inf') else 999.0
    delta_dict_hit = dict_hit - baseline_dict_hit
    delta_signal_rate = signal_rate - baseline_signal_rate
    delta_bigram_z = bigram_z_safe - baseline_bigram_z

    print(f"     Baseline dict_hit:    {baseline_dict_hit:.4f}")
    print(f"     Current dict_hit:     {dict_hit:.4f}  (delta={delta_dict_hit:+.4f})")
    print(f"     Baseline signal_rate: {baseline_signal_rate:.4f}")
    print(f"     Current signal_rate:  {signal_rate:.4f}  (delta={delta_signal_rate:+.4f})")
    print(f"     Baseline bigram_z:    {baseline_bigram_z:.2f}")
    print(f"     Current bigram_z:     {bigram_z_safe:.2f}  (delta={delta_bigram_z:+.2f})")

    # ── 14. Determine verdict ──
    print("\n  14. Determining verdict ...")

    if (delta_dict_hit > 0.02 and delta_signal_rate > 0.01
            and cascade_shape in ('accelerating', 'linear')
            and len(total_new_words) >= 2):
        verdict = 'CASCADE_FOUND'
    elif (delta_dict_hit > 0.005 or delta_signal_rate > 0.005
          or delta_bigram_z > 0.5):
        verdict = 'MARGINAL_IMPROVEMENT'
    else:
        verdict = 'NO_IMPROVEMENT'

    print(f"     Verdict: {verdict}")

    # ── 15. Save results ──
    print("\n  15. Saving results ...")

    result = LongCribPropagateResult(
        n_new_assignments=len(new_assignments),
        new_assignments=new_assignments,
        n_family_propagated=len(family_propagated),
        family_propagated=family_propagated,
        propagated_assignment=assignment,
        n_tokens=n_tokens,
        dict_hit=round(dict_hit, 6),
        signal_rate=round(signal_rate, 6),
        n_signal=n_signal,
        n_anti_signal=n_anti_signal,
        bigram_z=round(bigram_z_safe, 2),
        n_signal_pairs=n_signal_pairs,
        n_bigram_hits=n_bigram_hits,
        baseline_dict_hit=round(baseline_dict_hit, 6),
        baseline_signal_rate=round(baseline_signal_rate, 6),
        baseline_bigram_z=round(baseline_bigram_z, 2),
        delta_dict_hit=round(delta_dict_hit, 6),
        delta_signal_rate=round(delta_signal_rate, 6),
        delta_bigram_z=round(delta_bigram_z, 2),
        cascade_shape=cascade_shape,
        confirmed_before=confirmed_before,
        confirmed_after=confirmed_after,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'long_crib_propagate.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)
    print(f"\n  -> {out_path}")

    print(f"\n  Result: {verdict}")
    print(f"    New assignments: {len(new_assignments)}, "
          f"family-propagated: {len(family_propagated)}")
    print(f"    dict_hit: {dict_hit:.4f} (delta={delta_dict_hit:+.4f})")
    print(f"    signal_rate: {signal_rate:.4f} (delta={delta_signal_rate:+.4f})")
    print(f"    bigram_z: {bigram_z_safe:.2f} (delta={delta_bigram_z:+.2f})")
    print(f"    cascade: {cascade_shape}, "
          f"confirmed: {confirmed_before} -> {confirmed_after}")
    print(f"    bootstrap words: {total_new_words if total_new_words else '(none)'}")
    print(f"  Completed in {time.time() - t0:.1f}s")
