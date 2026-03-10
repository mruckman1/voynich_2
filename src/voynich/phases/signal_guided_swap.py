"""
Phase 33.3 -- Signal-Guided Triple Swap
=========================================
For each WRONG/SUSPECT triple (those with net_signal < 0.2 and not confirmed),
enumerates alternative syllable assignments from the phoneme maps, then greedily
accepts swaps that maximize SIGNAL tokens while minimizing ANTI_SIGNAL tokens.

Unlike Phase 24 which maximised dict-hit, this step uses the SIGNAL objective:
tokens that distinguish the real Voynich from null corpora.

Dependency chain:
    triple_signal_rates.json   (Step 33.2 — swap candidates, optional)
    combined_refine.json       (Phase 15 best_assignment)
    modifier_integrate.json    (Phase 16 modifiers)
    null_corpus.json           (Phase 17 null seeds)
    bootstrap_loop.json        (Phase 30 confirmed triples, optional)
        → signal_guided_swap.json  (this step)
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
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    decode_token_modifier_aware,
    load_corpus,
    token_to_triples,
    tokenize_eva_chars,
)
from voynich.core.reference import (
    build_expanded_word_set,
    load_reference_corpus,
    PHONEME_PLACE_MAP,
    PHONEME_NUCLEUS_MAP,
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

def _convert(obj):
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
class SwapCandidate:
    triple_key: str
    original_syllable: str
    candidate_syllable: str
    delta_signal: int
    delta_anti_signal: int
    new_signal_count: int
    new_anti_signal_count: int
    new_dict_hit: float
    accepted: bool
    reason: str


@dataclass
class SignalGuidedSwapResult:
    n_target_triples: int
    n_candidates_tested: int
    n_swaps_accepted: int
    accepted_swaps: List[Dict]
    rejected_swaps: List[Dict]  # top 5 rejected
    new_assignment: Dict[str, str]
    original_signal_count: int
    new_signal_count: int
    original_anti_signal_count: int
    new_anti_signal_count: int
    original_dict_hit: float
    new_dict_hit: float
    verdict: str  # 'SWAPS_FOUND' or 'NO_IMPROVEMENT'
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Build triples-per-token index
# ---------------------------------------------------------------------------

def _build_triples_per_token(
    all_tokens: List[str],
    eva_to_triple: Dict[str, str],
) -> List[Set[str]]:
    """For each token, return the set of triple_keys it contains."""
    result = []
    for token in all_tokens:
        triples = token_to_triples(token, eva_to_triple)
        result.append(set(triples))
    return result


# ---------------------------------------------------------------------------
# Generate candidate syllables for a triple
# ---------------------------------------------------------------------------

def _generate_candidate_syllables(
    triple_key: str,
    existing_syllables: Set[str],
) -> List[str]:
    """Enumerate CV syllables from phoneme maps for a given triple_key.

    Filters out any syllable already assigned to another triple
    (all-different constraint).
    """
    parts = triple_key.split(',')
    if len(parts) != 3:
        return []
    first_stroke, last_stroke, _glyph_class = parts

    consonants = PHONEME_PLACE_MAP.get(first_stroke, [])
    vowels = PHONEME_NUCLEUS_MAP.get(last_stroke, [])

    candidates = []
    # CV combinations
    for c in consonants:
        for v in vowels:
            syl = c + v
            if syl not in existing_syllables:
                candidates.append(syl)

    # Pure vowels (some triples map to vowel-only syllables)
    for v in vowels:
        if v not in existing_syllables:
            candidates.append(v)

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for s in candidates:
        if s not in seen:
            seen.add(s)
            unique.append(s)

    return unique


# ---------------------------------------------------------------------------
# Fast-path token re-decode
# ---------------------------------------------------------------------------

def _fast_evaluate_swap(
    all_tokens: List[str],
    triples_per_token: List[Set[str]],
    base_decoded: List[str],
    base_real_hits: List[bool],
    triple_key: str,
    new_syllable: str,
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    modifier_chars: set,
    modifier_rules: Dict[str, str],
    ref_word_set: set,
) -> Tuple[List[str], List[bool], List[int]]:
    """Re-decode only tokens containing the swapped triple."""
    affected = [i for i, tl in enumerate(triples_per_token) if triple_key in tl]

    # Copy and modify assignment
    new_assignment = dict(assignment)
    new_assignment[triple_key] = new_syllable

    # Copy base arrays
    new_decoded = list(base_decoded)
    new_real_hits = list(base_real_hits)

    for idx in affected:
        token = all_tokens[idx]
        # R3 decode: alteration -> strip -> raw
        alt = decode_token_modifier_aware(
            token, new_assignment, eva_to_triple,
            modifier_chars, modifier_rules,
        )
        if alt.lower() in ref_word_set:
            new_decoded[idx] = alt.lower()
            new_real_hits[idx] = True
            continue

        stripped = decode_token_modifier_aware(
            token, new_assignment, eva_to_triple, modifier_chars,
        )
        if stripped.lower() in ref_word_set:
            new_decoded[idx] = stripped.lower()
            new_real_hits[idx] = True
            continue

        raw = decode_token(token, new_assignment, eva_to_triple)
        new_decoded[idx] = raw.lower()
        new_real_hits[idx] = raw.lower() in ref_word_set

    return new_decoded, new_real_hits, affected


def _fast_evaluate_null_swap(
    null_tokens: List[str],
    null_triples_per_token: List[Set[str]],
    base_null_hits: List[bool],
    triple_key: str,
    new_syllable: str,
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    modifier_chars: set,
    modifier_rules: Dict[str, str],
    ref_word_set: set,
) -> List[bool]:
    """Re-decode only null-corpus tokens containing the swapped triple."""
    affected = [i for i, tl in enumerate(null_triples_per_token)
                if triple_key in tl]

    new_assignment = dict(assignment)
    new_assignment[triple_key] = new_syllable

    new_null_hits = list(base_null_hits)
    for idx in affected:
        token = null_tokens[idx]
        # R3 decode
        alt = decode_token_modifier_aware(
            token, new_assignment, eva_to_triple,
            modifier_chars, modifier_rules,
        )
        if alt.lower() in ref_word_set:
            new_null_hits[idx] = True
            continue

        stripped = decode_token_modifier_aware(
            token, new_assignment, eva_to_triple, modifier_chars,
        )
        if stripped.lower() in ref_word_set:
            new_null_hits[idx] = True
            continue

        raw = decode_token(token, new_assignment, eva_to_triple)
        new_null_hits[idx] = raw.lower() in ref_word_set

    return new_null_hits


# ---------------------------------------------------------------------------
# Signal classification
# ---------------------------------------------------------------------------

def _classify_tokens(
    real_hits: List[bool],
    null_hits_lists: List[List[bool]],
    n_tokens: int,
) -> Tuple[int, int, List[str]]:
    """Classify every token position and return (n_signal, n_anti_signal, classifications)."""
    n_signal = 0
    n_anti_signal = 0
    classifications = []

    for idx in range(n_tokens):
        r_hit = real_hits[idx]
        null_hit_count = sum(1 for nh in null_hits_lists if nh[idx])

        if r_hit and null_hit_count <= 1:
            classifications.append('SIGNAL')
            n_signal += 1
        elif r_hit and null_hit_count >= 3:
            classifications.append('SHARED_HIT')
        elif not r_hit and null_hit_count >= 3:
            classifications.append('ANTI_SIGNAL')
            n_anti_signal += 1
        else:
            classifications.append('SHARED_MISS')

    return n_signal, n_anti_signal, classifications


# ---------------------------------------------------------------------------
# Bigram z-score (simplified from signal_bigrams.py)
# ---------------------------------------------------------------------------

def _compute_bigram_z(
    classifications: List[str],
    decoded: List[str],
    folios: List[str],
    ref_bigrams: Set[Tuple[str, str]],
    n_perms: int = 500,
    seed: int = 42,
) -> float:
    """Compute bigram z-score for SIGNAL-SIGNAL pairs.

    Uses a lighter permutation test (500 perms instead of 1000)
    for speed during swap evaluation.
    """
    n_tokens = len(classifications)
    n_signal = sum(1 for c in classifications if c == 'SIGNAL')

    # Count actual SIGNAL-SIGNAL bigram hits
    n_pairs = 0
    n_hits = 0
    for i in range(n_tokens - 1):
        if (classifications[i] == 'SIGNAL'
                and classifications[i + 1] == 'SIGNAL'
                and folios[i] == folios[i + 1]):
            n_pairs += 1
            if (decoded[i], decoded[i + 1]) in ref_bigrams:
                n_hits += 1

    hit_rate = n_hits / n_pairs if n_pairs > 0 else 0.0

    if n_pairs == 0:
        return 0.0

    # Null permutation
    rng = random.Random(seed)
    indices = list(range(n_tokens))
    null_rates = []

    for _ in range(n_perms):
        fake_signal = set(rng.sample(indices, min(n_signal, n_tokens)))
        fp = 0
        fh = 0
        for i in range(n_tokens - 1):
            if (i in fake_signal and (i + 1) in fake_signal
                    and folios[i] == folios[i + 1]):
                fp += 1
                if (decoded[i], decoded[i + 1]) in ref_bigrams:
                    fh += 1
        null_rates.append(fh / fp if fp > 0 else 0.0)

    null_mean = sum(null_rates) / len(null_rates) if null_rates else 0.0
    null_var = (sum((r - null_mean) ** 2 for r in null_rates) / len(null_rates)
                if null_rates else 0.0)
    null_std = null_var ** 0.5

    if null_std > 0:
        return (hit_rate - null_mean) / null_std
    return float('inf') if hit_rate > null_mean else 0.0


# ---------------------------------------------------------------------------
# Load confirmed triples
# ---------------------------------------------------------------------------

def _load_confirmed_triples(rd: str) -> Set[str]:
    """Load set of confirmed triple_keys from bootstrap_loop.json."""
    bt_path = os.path.join(rd, 'bootstrap_loop.json')
    if not os.path.exists(bt_path):
        return set()
    with open(bt_path) as f:
        bt_data = json.load(f)
    confirmed_keys = set(bt_data.get('confirmed_triples', []))
    if not confirmed_keys:
        # Fallback: all triples in final_assignment are considered confirmed
        # if there's no explicit list
        return set()
    return confirmed_keys


# ---------------------------------------------------------------------------
# Load swap candidates from Step 33.2
# ---------------------------------------------------------------------------

def _load_swap_candidates(
    rd: str,
    assignment: Dict[str, str],
    confirmed_triples: Set[str],
) -> List[str]:
    """Load swap candidate triple_keys from triple_signal_rates.json.

    Fallback: all unconfirmed triples (used when 33.2 hasn't run yet).
    """
    tsr_path = os.path.join(rd, 'triple_signal_rates.json')
    if os.path.exists(tsr_path):
        with open(tsr_path) as f:
            tsr_data = json.load(f)

        # Primary: explicit swap_candidates list
        candidates = tsr_data.get('swap_candidates', [])
        if candidates:
            return candidates

        # Secondary: filter triple_profiles by net_signal < 0.2
        profiles = tsr_data.get('triple_profiles', {})
        if profiles:
            candidates = []
            for tk, profile in profiles.items():
                if isinstance(profile, dict):
                    net_sig = profile.get('net_signal', 1.0)
                    if net_sig < 0.2 and tk not in confirmed_triples:
                        candidates.append(tk)
            if candidates:
                return candidates

    # Fallback: all unconfirmed triples
    return [tk for tk in assignment if tk not in confirmed_triples]


# ---------------------------------------------------------------------------
# Build folio tracking arrays
# ---------------------------------------------------------------------------

def _build_folio_arrays(corpus) -> Tuple[List[str], List[str]]:
    """Build parallel (token, folio) arrays from the corpus."""
    all_tokens = []
    token_folios = []
    for folio, page in corpus.pages.items():
        for token in page.all_tokens:
            all_tokens.append(token)
            token_folios.append(folio)
    return all_tokens, token_folios


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_signal_guided_swap() -> None:
    """Step 33.3: Signal-guided triple swap."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE 33.3: Signal-Guided Triple Swap")
    print("=" * 70)

    rd = _results_dir()
    eva_to_triple = build_eva_to_triple_lookup()

    # ── 1. Load inputs ──
    print("\n  1. Loading inputs ...")

    # Phase 15 assignment
    refine_path = os.path.join(rd, 'combined_refine.json')
    if not os.path.exists(refine_path):
        print("  [SKIP] combined_refine.json not found")
        return
    with open(refine_path) as f:
        refine_data = json.load(f)
    assignment = dict(refine_data.get('best_assignment', {}))

    # Phase 16 modifiers
    mod_path = os.path.join(rd, 'modifier_integrate.json')
    if not os.path.exists(mod_path):
        print("  [SKIP] modifier_integrate.json not found")
        return
    with open(mod_path) as f:
        mod_data = json.load(f)
    modifier_chars, modifier_rules = _reconstruct_modifier_rules(mod_data)

    # Null corpus seeds
    null_path = os.path.join(rd, 'null_corpus.json')
    null_seeds = [100, 101, 102, 103, 104]
    if os.path.exists(null_path):
        with open(null_path) as f:
            null_data = json.load(f)
        null_seeds = [r['seed'] for r in null_data.get('null_runs', [])]

    # Confirmed triples
    confirmed_triples = _load_confirmed_triples(rd)

    # Swap candidates
    target_triples = _load_swap_candidates(rd, assignment, confirmed_triples)

    print(f"     Assignment: {len(assignment)} triples")
    print(f"     Modifiers: {len(modifier_chars)} chars")
    print(f"     Confirmed triples: {len(confirmed_triples)}")
    print(f"     Swap target triples: {len(target_triples)}")
    print(f"     Null seeds: {null_seeds}")

    # ── 2. Build reference word set ──
    print("\n  2. Building reference word set ...")
    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    base_words = set(
        w.lower() for w in ref_corpus.get_combined_tokens('latin')
        if len(w) >= 2
    )
    expanded, _ = build_expanded_word_set(base_words)
    ref_word_set = base_words | expanded
    print(f"     {len(ref_word_set)} reference words")

    # Build reference bigrams for z-score validation
    ref_tokens = [w.lower() for w in ref_corpus.get_combined_tokens('latin')
                  if len(w) >= 2]
    ref_bigrams: Set[Tuple[str, str]] = set()
    for i in range(len(ref_tokens) - 1):
        ref_bigrams.add((ref_tokens[i], ref_tokens[i + 1]))
    print(f"     {len(ref_bigrams)} reference bigrams")

    # ── 3. Decode real corpus ──
    print("\n  3. Decoding real corpus ...")
    corpus = load_corpus(verbose=False)
    all_tokens, token_folios = _build_folio_arrays(corpus)
    n_tokens = len(all_tokens)

    real_decoded = _decode_corpus_r3(
        all_tokens, assignment, eva_to_triple,
        modifier_chars, modifier_rules, ref_word_set,
    )
    real_hits = [w in ref_word_set for w in real_decoded]
    original_dict_hit = sum(real_hits) / n_tokens
    print(f"     {n_tokens} tokens, dict_hit = {original_dict_hit:.4f}")

    # Build triples-per-token index for fast evaluation
    triples_per_token = _build_triples_per_token(all_tokens, eva_to_triple)

    # ── 4. Generate and decode null corpora ──
    print("\n  4. Generating and decoding null corpora ...")
    bigram_probs, initial_probs, token_lengths = _build_eva_bigram_model(all_tokens)

    null_tokens_lists: List[List[str]] = []
    null_hits_lists: List[List[bool]] = []
    null_triples_lists: List[List[Set[str]]] = []

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
        null_rate = sum(null_hits) / len(null_hits)
        print(f"       dict_hit = {null_rate:.4f}")

        null_tokens_lists.append(null_tokens)
        null_hits_lists.append(null_hits)
        null_triples_lists.append(
            _build_triples_per_token(null_tokens, eva_to_triple)
        )

    # ── 5. Baseline classification ──
    print("\n  5. Baseline classification ...")
    n_signal_orig, n_anti_orig, classifications_orig = _classify_tokens(
        real_hits, null_hits_lists, n_tokens,
    )
    print(f"     SIGNAL:      {n_signal_orig}")
    print(f"     ANTI_SIGNAL: {n_anti_orig}")
    print(f"     net_signal:  {n_signal_orig - n_anti_orig}")

    # Compute baseline bigram z for the gate
    baseline_z = _compute_bigram_z(
        classifications_orig, real_decoded, token_folios,
        ref_bigrams, n_perms=500, seed=42,
    )
    print(f"     Baseline bigram z-score: {baseline_z:.2f}")

    # ── 6. Greedy swap loop ──
    print(f"\n  6. Greedy swap loop over {len(target_triples)} target triples ...")

    current_assignment = dict(assignment)
    current_decoded = list(real_decoded)
    current_real_hits = list(real_hits)
    current_null_hits_lists = [list(nh) for nh in null_hits_lists]
    current_n_signal = n_signal_orig
    current_n_anti = n_anti_orig

    accepted_swaps: List[SwapCandidate] = []
    all_rejected: List[SwapCandidate] = []
    n_candidates_tested = 0

    # Iterate: find best swap across all target triples, accept it, repeat
    # until no more positive-delta swaps remain
    max_rounds = len(target_triples)  # at most one swap per target triple
    remaining_targets = list(target_triples)

    for round_num in range(max_rounds):
        if not remaining_targets:
            break

        best_candidate: Optional[SwapCandidate] = None
        best_net_delta = 0  # must be strictly positive to accept
        best_new_decoded = None
        best_new_real_hits = None
        best_new_null_hits_lists = None
        best_triple_key = None

        print(f"\n     Round {round_num + 1}: testing {len(remaining_targets)} target triples ...")

        # Collect the set of currently assigned syllables (excluding
        # the triple under test, since we are replacing it)
        for tk in remaining_targets:
            # Syllables used by OTHER triples (all-different constraint)
            other_syllables = set(
                v for k, v in current_assignment.items() if k != tk
            )

            # Generate candidates for this triple
            candidates = _generate_candidate_syllables(tk, other_syllables)
            if not candidates:
                continue

            original_syllable = current_assignment.get(tk, '??')

            for cand_syl in candidates:
                n_candidates_tested += 1

                # Fast re-decode real corpus
                new_decoded, new_real_hits, _ = _fast_evaluate_swap(
                    all_tokens, triples_per_token,
                    current_decoded, current_real_hits,
                    tk, cand_syl,
                    current_assignment, eva_to_triple,
                    modifier_chars, modifier_rules, ref_word_set,
                )

                # Fast re-decode null corpora
                new_null_hits_lists = []
                for ni in range(len(null_seeds)):
                    new_null_hits = _fast_evaluate_null_swap(
                        null_tokens_lists[ni],
                        null_triples_lists[ni],
                        current_null_hits_lists[ni],
                        tk, cand_syl,
                        current_assignment, eva_to_triple,
                        modifier_chars, modifier_rules, ref_word_set,
                    )
                    new_null_hits_lists.append(new_null_hits)

                # Reclassify
                new_n_signal, new_n_anti, _ = _classify_tokens(
                    new_real_hits, new_null_hits_lists, n_tokens,
                )

                delta_signal = new_n_signal - current_n_signal
                delta_anti = new_n_anti - current_n_anti
                # Objective: maximise signal, minimise anti_signal
                # net_delta = gain_in_signal - gain_in_anti
                net_delta = delta_signal - delta_anti

                new_dict_hit = sum(new_real_hits) / n_tokens

                swap = SwapCandidate(
                    triple_key=tk,
                    original_syllable=original_syllable,
                    candidate_syllable=cand_syl,
                    delta_signal=delta_signal,
                    delta_anti_signal=delta_anti,
                    new_signal_count=new_n_signal,
                    new_anti_signal_count=new_n_anti,
                    new_dict_hit=round(new_dict_hit, 6),
                    accepted=False,
                    reason='',
                )

                if net_delta > best_net_delta:
                    best_net_delta = net_delta
                    best_candidate = swap
                    best_new_decoded = new_decoded
                    best_new_real_hits = new_real_hits
                    best_new_null_hits_lists = new_null_hits_lists
                    best_triple_key = tk

        # Accept or stop
        if best_candidate is None or best_net_delta <= 0:
            print(f"     No positive-delta swap found. Stopping.")
            break

        # Gate check: bigram z must not regress below 6.14
        # Recompute classifications with the best swap for z-check
        _, _, new_cls = _classify_tokens(
            best_new_real_hits, best_new_null_hits_lists, n_tokens,
        )
        new_z = _compute_bigram_z(
            new_cls, best_new_decoded, token_folios,
            ref_bigrams, n_perms=500, seed=42,
        )

        if new_z < 6.14:
            best_candidate.accepted = False
            best_candidate.reason = (
                f"bigram z regression: {new_z:.2f} < 6.14"
            )
            all_rejected.append(best_candidate)
            # Remove this triple from remaining targets to avoid infinite loop
            if best_triple_key in remaining_targets:
                remaining_targets.remove(best_triple_key)
            print(f"     REJECTED {best_candidate.triple_key}: "
                  f"{best_candidate.original_syllable} -> "
                  f"{best_candidate.candidate_syllable} "
                  f"(bigram z={new_z:.2f} < 6.14)")
            continue

        # Accept the swap
        best_candidate.accepted = True
        best_candidate.reason = (
            f"net_delta=+{best_net_delta}, bigram z={new_z:.2f}"
        )
        accepted_swaps.append(best_candidate)

        # Update state
        current_assignment[best_triple_key] = best_candidate.candidate_syllable
        current_decoded = best_new_decoded
        current_real_hits = best_new_real_hits
        current_null_hits_lists = best_new_null_hits_lists
        current_n_signal = best_candidate.new_signal_count
        current_n_anti = best_candidate.new_anti_signal_count

        # Remove accepted triple from remaining targets
        if best_triple_key in remaining_targets:
            remaining_targets.remove(best_triple_key)

        print(f"     ACCEPTED {best_candidate.triple_key}: "
              f"{best_candidate.original_syllable} -> "
              f"{best_candidate.candidate_syllable} "
              f"(delta_sig=+{best_candidate.delta_signal}, "
              f"delta_anti={best_candidate.delta_anti_signal:+d}, "
              f"net=+{best_net_delta}, z={new_z:.2f})")

    # ── 7. Final statistics ──
    print(f"\n  7. Final statistics ...")
    new_dict_hit = sum(current_real_hits) / n_tokens

    print(f"     Swaps accepted: {len(accepted_swaps)}")
    print(f"     Candidates tested: {n_candidates_tested}")
    print(f"     SIGNAL:      {n_signal_orig} -> {current_n_signal} "
          f"(delta={current_n_signal - n_signal_orig:+d})")
    print(f"     ANTI_SIGNAL: {n_anti_orig} -> {current_n_anti} "
          f"(delta={current_n_anti - n_anti_orig:+d})")
    print(f"     dict_hit:    {original_dict_hit:.4f} -> {new_dict_hit:.4f} "
          f"(delta={new_dict_hit - original_dict_hit:+.4f})")

    if accepted_swaps:
        print("\n     Accepted swaps:")
        for s in accepted_swaps:
            print(f"       {s.triple_key}: {s.original_syllable} -> "
                  f"{s.candidate_syllable} "
                  f"(sig=+{s.delta_signal}, anti={s.delta_anti_signal:+d})")

    # ── 8. Verdict ──
    if len(accepted_swaps) > 0:
        total_signal_gain = current_n_signal - n_signal_orig
        total_anti_change = current_n_anti - n_anti_orig
        verdict = (
            f"SWAPS_FOUND: {len(accepted_swaps)} swaps accepted. "
            f"SIGNAL {n_signal_orig} -> {current_n_signal} "
            f"(+{total_signal_gain}), "
            f"ANTI_SIGNAL {n_anti_orig} -> {current_n_anti} "
            f"({total_anti_change:+d}), "
            f"dict_hit {original_dict_hit:.4f} -> {new_dict_hit:.4f}."
        )
    else:
        verdict = (
            f"NO_IMPROVEMENT: 0 swaps accepted out of "
            f"{n_candidates_tested} candidates tested across "
            f"{len(target_triples)} target triples. "
            f"Current assignment is locally optimal for SIGNAL objective."
        )

    print(f"\n  Verdict: {verdict}")

    # ── 9. Save ──
    # Sort rejected by net_delta descending, keep top 5
    all_rejected.sort(
        key=lambda s: (s.delta_signal - s.delta_anti_signal), reverse=True,
    )
    top_rejected = all_rejected[:5]

    result = SignalGuidedSwapResult(
        n_target_triples=len(target_triples),
        n_candidates_tested=n_candidates_tested,
        n_swaps_accepted=len(accepted_swaps),
        accepted_swaps=[_convert(asdict(s)) for s in accepted_swaps],
        rejected_swaps=[_convert(asdict(s)) for s in top_rejected],
        new_assignment=current_assignment,
        original_signal_count=n_signal_orig,
        new_signal_count=current_n_signal,
        original_anti_signal_count=n_anti_orig,
        new_anti_signal_count=current_n_anti,
        original_dict_hit=round(original_dict_hit, 6),
        new_dict_hit=round(new_dict_hit, 6),
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'signal_guided_swap.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)
    print(f"\n  -> {out_path}")
