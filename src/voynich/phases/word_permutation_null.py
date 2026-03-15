"""
Word-Level Permutation Null Test
=================================
Test whether the 22 T1 word-level identifications (ratione, coralli,
diasene, etc.) depend on the specific assignment table T_P15, or
whether any random permuted table + the same pipeline produces
comparable T1 counts.

For each of N random tables, the test:
1. Generates a full permutation of the 25 triple → syllable assignments
2. Identifies that table's own signal words via lightweight sigma proxy
3. Determines that table's own confirmed triples
4. Runs the bridge search + scoring + tiering pipeline
5. Counts T1 identifications and computes CI overlap

Dependency chain:
    combined_refine.json        (Phase 15 best table)
    signal_bigrams.json         (Phase 29 parallel arrays)
    modifier_integrate.json     (Phase 16 modifiers)
    triple_tiers.json           (Phase 44 confirmed triples)
    word_catalog.json           (Phase 52 real baseline)
        -> word_permutation_null.json
"""

from __future__ import annotations

import json
import os
import random
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

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
from voynich.phases.concatenation_bridge import (
    BridgeMatch,
    _build_partial_decode,
    _build_pharma_dict,
    _extract_implied_assignments,
    _search_dict,
)
from voynich.phases.csp_solver import decode_token
from voynich.phases.null_corpus import (
    _build_eva_bigram_model,
    _generate_null_corpus,
    _reconstruct_modifier_rules,
)
from voynich.phases.suffix_calibration import SIGNAL_WORDS_SET
from voynich.phases.word_catalog import _detect_ambiguity, _score_and_tier


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_TRIALS = 200
SIGMA_THRESHOLD = 2.0
MIN_SIGNAL_BACKING = 2     # triples must back ≥ N signal words to be confirmed
NULL_SEEDS = [100, 101, 102, 103, 104]
SEED_OFFSET = 7000         # different range from Phase 50's 5000


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
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return _convert(obj.tolist())
    if isinstance(obj, float) and (obj != obj):
        return None
    if isinstance(obj, set):
        return sorted(_convert(item) for item in obj)
    if isinstance(obj, (bool, int, float, str, type(None))):
        return obj
    return str(obj)


def _save_json(rd: str, filename: str, data: Any) -> str:
    path = os.path.join(rd, filename)
    with open(path, 'w') as f:
        json.dump(_convert(data), f, indent=2)
    return path


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TrialResult:
    trial_idx: int
    seed: int
    n_signal_words: int
    n_confirmed_triples: int
    n_bridge_matches: int
    n_T1: int
    n_T2: int
    n_T3: int
    n_distinct_T1_words: int
    T1_words: List[str]
    ci_overlap: float
    mean_folio_spread: float


@dataclass
class WordPermutationNullResult:
    # Real table baseline
    real_n_T1: int
    real_n_distinct_words: int
    real_ci_overlap: float
    real_mean_folio_spread: float
    real_T1_words: List[str]
    # Null distribution
    null_n_trials: int
    null_T1_mean: float
    null_T1_std: float
    null_T1_max: int
    null_distinct_words_mean: float
    null_distinct_words_std: float
    null_ci_overlap_mean: float
    null_ci_overlap_std: float
    null_folio_spread_mean: float
    null_folio_spread_std: float
    # Z-scores and p-values
    T1_z_score: float
    T1_p_value: float
    distinct_words_z: float
    distinct_words_p: float
    ci_overlap_z: float
    folio_spread_z: float
    # Per-trial details
    trial_results: List[Dict]
    # Verdict
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Decode helpers (mirrors signal_isolation._decode_corpus_r3)
# ---------------------------------------------------------------------------

def _decode_corpus_r3(
    tokens: List[str],
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    modifier_chars: set,
    modifier_rules: Dict[str, str],
    ref_word_set: set,
) -> List[str]:
    """Decode tokens using R3 strategy: try alteration, then strip, then raw."""
    decoded = []
    for token in tokens:
        # Alteration
        alt = decode_token_modifier_aware(
            token, assignment, eva_to_triple, modifier_chars, modifier_rules,
        )
        if alt.lower() in ref_word_set:
            decoded.append(alt.lower())
            continue
        # Strip
        stripped = decode_token_modifier_aware(
            token, assignment, eva_to_triple, modifier_chars,
        )
        if stripped.lower() in ref_word_set:
            decoded.append(stripped.lower())
            continue
        # Raw
        raw = decode_token(token, assignment, eva_to_triple)
        decoded.append(raw.lower())
    return decoded


# ---------------------------------------------------------------------------
# Lightweight signal isolation proxy
# ---------------------------------------------------------------------------

def _lightweight_signal_isolation(
    real_tokens: List[str],
    null_token_lists: List[List[str]],
    perm_assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    modifier_chars: set,
    modifier_rules: Dict[str, str],
    ref_word_set: set,
) -> Tuple[Set[str], Set[str], List[str], List[str]]:
    """Lightweight signal isolation for a permuted table.

    Returns:
        signal_words: decoded words with σ > SIGMA_THRESHOLD
        confirmed_triples: triples backing ≥ MIN_SIGNAL_BACKING signal words
        perm_decoded: decoded real corpus
        perm_classifications: per-token classification strings
    """
    # 1. Decode real corpus
    perm_decoded = _decode_corpus_r3(
        real_tokens, perm_assignment, eva_to_triple,
        modifier_chars, modifier_rules, ref_word_set,
    )
    n_tokens = len(perm_decoded)
    real_hits = [w in ref_word_set for w in perm_decoded]

    # 2. Decode null corpora
    null_decoded_list: List[List[str]] = []
    null_hits_list: List[List[bool]] = []
    for null_tokens in null_token_lists:
        null_decoded = _decode_corpus_r3(
            null_tokens, perm_assignment, eva_to_triple,
            modifier_chars, modifier_rules, ref_word_set,
        )
        null_decoded_list.append(null_decoded)
        null_hits_list.append([w in ref_word_set for w in null_decoded])

    # 3. Per-word sigma: test all words that are hits in real corpus
    real_word_counts = Counter(
        w for w, hit in zip(perm_decoded, real_hits) if hit
    )
    signal_words: Set[str] = set()

    for word, real_count in real_word_counts.items():
        null_counts = [
            Counter(nd).get(word, 0) for nd in null_decoded_list
        ]
        null_mean = sum(null_counts) / len(null_counts) if null_counts else 0.0
        null_var = (
            sum((c - null_mean) ** 2 for c in null_counts) / len(null_counts)
            if null_counts else 0.0
        )
        null_std = null_var ** 0.5

        if null_std > 0:
            sigma = (real_count - null_mean) / null_std
        else:
            sigma = float('inf') if real_count > null_mean else 0.0

        if sigma > SIGMA_THRESHOLD:
            signal_words.add(word)

    # 4. Token-level classification
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

    # 5. Confirmed triples: triples backing ≥ MIN_SIGNAL_BACKING signal words
    triple_signal_backing: Dict[str, Set[str]] = defaultdict(set)
    for idx in range(n_tokens):
        word = perm_decoded[idx]
        if word not in signal_words:
            continue
        # Map this token's EVA chars → triples
        chars = tokenize_eva_chars(real_tokens[idx])
        for ch in chars:
            if ch in modifier_chars:
                continue
            triple = eva_to_triple.get(ch)
            if triple and triple in perm_assignment:
                triple_signal_backing[triple].add(word)

    confirmed_triples: Set[str] = {
        triple for triple, words in triple_signal_backing.items()
        if len(words) >= MIN_SIGNAL_BACKING
    }

    return signal_words, confirmed_triples, perm_decoded, classifications


# ---------------------------------------------------------------------------
# Modified bridge search (parameterized signal_words_set)
# ---------------------------------------------------------------------------

def _run_bridge_search_with_signal_set(
    token_evas: List[str],
    token_decoded: List[str],
    token_classifications: List[str],
    token_folios: List[str],
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    modifier_chars: Set[str],
    confirmed_triples: Set[str],
    pharma_dict: Set[str],
    signal_words_set: Set[str],
) -> Tuple[List[BridgeMatch], List[BridgeMatch]]:
    """Bridge search with custom signal words (not hardcoded SIGNAL_WORDS_SET).

    Identical to word_catalog._run_full_bridge_search except signal_words_set
    is passed in rather than imported from suffix_calibration.
    """
    n_tokens = len(token_evas)

    # Find all SIGNAL token positions
    signal_positions = set()
    for i in range(n_tokens):
        if token_decoded[i] in signal_words_set:
            signal_positions.add(i)

    # ── Bridge matches ──
    bridge_matches: List[BridgeMatch] = []
    seen_dark_tokens: Set[int] = set()

    for sig_idx in sorted(signal_positions):
        anchor_word = token_decoded[sig_idx]

        for dist in [1, 2]:
            for offset, position in [(-dist, 'before'), (dist, 'after')]:
                nbr_idx = sig_idx + offset
                if nbr_idx < 0 or nbr_idx >= n_tokens:
                    continue
                if nbr_idx in signal_positions:
                    continue
                if nbr_idx in seen_dark_tokens:
                    continue

                seen_dark_tokens.add(nbr_idx)
                dark_eva = token_evas[nbr_idx]

                pattern, details = _build_partial_decode(
                    dark_eva, assignment, eva_to_triple,
                    modifier_chars, confirmed_triples,
                )

                n_conf = sum(1 for _, _, _, c in details if c)
                n_free = sum(1 for _, _, _, c in details if not c)

                if n_conf < 1 or n_free < 1 or n_free > 3:
                    continue

                matches = _search_dict(pattern, pharma_dict)
                if not matches:
                    continue

                for mword in matches:
                    implied = _extract_implied_assignments(
                        pattern, mword, details, eva_to_triple,
                    )
                    bridge_matches.append(BridgeMatch(
                        token_idx=nbr_idx,
                        token_eva=dark_eva,
                        pattern=pattern,
                        matched_word=mword,
                        n_confirmed_chars=n_conf,
                        n_free_chars=n_free,
                        implied_assignments=implied,
                        anchor_word=anchor_word,
                        anchor_position=position,
                        distance=dist,
                        folio=token_folios[nbr_idx],
                        n_total_matches=len(matches),
                    ))

    # ── Concatenation matches ──
    concat_matches: List[BridgeMatch] = []

    for sig_idx in sorted(signal_positions):
        anchor_word = token_decoded[sig_idx]

        for offset, position in [(-1, 'before'), (1, 'after')]:
            nbr_idx = sig_idx + offset
            if nbr_idx < 0 or nbr_idx >= n_tokens:
                continue
            if nbr_idx in signal_positions:
                continue

            dark_eva = token_evas[nbr_idx]
            pattern, details = _build_partial_decode(
                dark_eva, assignment, eva_to_triple,
                modifier_chars, confirmed_triples,
            )

            if not any(not c for _, _, _, c in details):
                continue

            if position == 'after':
                concat_pattern = anchor_word + pattern
            else:
                concat_pattern = pattern + anchor_word

            concat_hits = _search_dict(concat_pattern, pharma_dict)
            for mword in concat_hits:
                concat_matches.append(BridgeMatch(
                    token_idx=nbr_idx,
                    token_eva=dark_eva,
                    pattern=concat_pattern,
                    matched_word=mword,
                    n_confirmed_chars=sum(1 for _, _, _, c in details if c),
                    n_free_chars=sum(1 for _, _, _, c in details if not c),
                    implied_assignments={},
                    anchor_word=anchor_word,
                    anchor_position=position,
                    distance=1,
                    folio=token_folios[nbr_idx],
                    n_total_matches=len(concat_hits),
                ))

    return bridge_matches, concat_matches


# ---------------------------------------------------------------------------
# Pipeline: group bridge matches → score → tier → metrics
# ---------------------------------------------------------------------------

def _bridge_to_metrics(
    bridge_matches: List[BridgeMatch],
    token_evas: List[str],
    ci_vocab: Set[str],
) -> Tuple[int, int, int, int, int, List[str], float, float]:
    """Group bridge matches, score/tier, and extract metrics.

    Returns:
        n_bridge, n_T1, n_T2, n_T3, n_distinct_T1, T1_words,
        ci_overlap, mean_folio_spread
    """
    eva_corpus_freq = Counter(token_evas)

    word_pairs: Dict[Tuple[str, str], Dict] = defaultdict(lambda: {
        'folios': set(), 'positions': [], 'anchors': set(),
        'count': 0, 'pattern_uniqueness': [], 'implied': {},
        'pattern': '',
    })

    for bm in bridge_matches:
        key = (bm.token_eva, bm.matched_word)
        word_pairs[key]['folios'].add(bm.folio)
        word_pairs[key]['positions'].append(bm.token_idx)
        word_pairs[key]['anchors'].add(bm.anchor_word)
        word_pairs[key]['count'] += 1
        word_pairs[key]['pattern_uniqueness'].append(bm.n_total_matches)
        if bm.implied_assignments:
            word_pairs[key]['implied'].update(bm.implied_assignments)
        if not word_pairs[key]['pattern']:
            word_pairs[key]['pattern'] = bm.pattern

    all_ids = _score_and_tier(dict(word_pairs), eva_corpus_freq)
    all_ids = _detect_ambiguity(all_ids)

    tier_counts = Counter(wid.tier for wid in all_ids)
    n_T1 = tier_counts.get('T1', 0)
    n_T2 = tier_counts.get('T2', 0)
    n_T3 = tier_counts.get('T3', 0)

    T1_words = sorted({wid.latin_word for wid in all_ids if wid.tier == 'T1'})
    n_distinct_T1 = len(T1_words)

    # CI overlap
    if T1_words:
        t1_set = set(T1_words)
        ci_overlap = len(t1_set & ci_vocab) / len(t1_set)
    else:
        ci_overlap = 0.0

    # Mean folio spread
    t1_folio_counts = [
        wid.n_folios for wid in all_ids if wid.tier == 'T1'
    ]
    mean_folio_spread = (
        sum(t1_folio_counts) / len(t1_folio_counts)
        if t1_folio_counts else 0.0
    )

    return (
        len(bridge_matches), n_T1, n_T2, n_T3,
        n_distinct_T1, T1_words, ci_overlap, mean_folio_spread,
    )


# ---------------------------------------------------------------------------
# Single trial
# ---------------------------------------------------------------------------

def _run_single_trial(
    trial_idx: int,
    real_assignment: Dict[str, str],
    real_tokens: List[str],
    token_evas: List[str],
    token_folios: List[str],
    null_token_lists: List[List[str]],
    eva_to_triple: Dict[str, str],
    modifier_chars: set,
    modifier_rules: Dict[str, str],
    ref_word_set: set,
    pharma_dict: Set[str],
    ci_vocab: Set[str],
) -> TrialResult:
    """Run the full word-ID pipeline for one permuted assignment table."""
    seed = SEED_OFFSET + trial_idx

    # 1. Generate permuted assignment
    rng = random.Random(seed)
    all_keys = list(real_assignment.keys())
    all_vals = [real_assignment[k] for k in all_keys]
    rng.shuffle(all_vals)
    perm_assignment = dict(zip(all_keys, all_vals))

    # 2. Lightweight signal isolation
    signal_words, confirmed_triples, perm_decoded, perm_classifications = (
        _lightweight_signal_isolation(
            real_tokens, null_token_lists, perm_assignment,
            eva_to_triple, modifier_chars, modifier_rules, ref_word_set,
        )
    )

    # 3. Bridge search
    bridge_matches, _concat = _run_bridge_search_with_signal_set(
        token_evas, perm_decoded, perm_classifications, token_folios,
        perm_assignment, eva_to_triple, modifier_chars, confirmed_triples,
        pharma_dict, signal_words,
    )

    # 4. Group → score → tier → metrics
    (n_bridge, n_T1, n_T2, n_T3, n_distinct_T1, T1_words,
     ci_overlap, mean_folio_spread) = _bridge_to_metrics(
        bridge_matches, token_evas, ci_vocab,
    )

    return TrialResult(
        trial_idx=trial_idx,
        seed=seed,
        n_signal_words=len(signal_words),
        n_confirmed_triples=len(confirmed_triples),
        n_bridge_matches=n_bridge,
        n_T1=n_T1,
        n_T2=n_T2,
        n_T3=n_T3,
        n_distinct_T1_words=n_distinct_T1,
        T1_words=T1_words,
        ci_overlap=round(ci_overlap, 4),
        mean_folio_spread=round(mean_folio_spread, 2),
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_word_permutation_null() -> None:
    """Word-Level Permutation Null Test."""
    t0 = time.time()

    print("=" * 70)
    print("WORD-LEVEL PERMUTATION NULL TEST")
    print("=" * 70)

    rd = _results_dir()

    # ------------------------------------------------------------------
    # 1. Load shared data
    # ------------------------------------------------------------------
    print("\n--- Step 1: Loading shared data ---")

    with open(os.path.join(rd, 'signal_bigrams.json')) as f:
        sig_data = json.load(f)
    token_evas: List[str] = sig_data['token_evas']
    token_folios: List[str] = sig_data['token_folios']
    n_tokens = len(token_evas)
    print(f"  Tokens: {n_tokens}")

    with open(os.path.join(rd, 'combined_refine.json')) as f:
        refine_data = json.load(f)
    real_assignment: Dict[str, str] = refine_data['best_assignment']
    print(f"  Assignment: {len(real_assignment)} triples")

    with open(os.path.join(rd, 'modifier_integrate.json')) as f:
        mod_data = json.load(f)
    modifier_chars, modifier_rules = _reconstruct_modifier_rules(mod_data)
    print(f"  Modifiers: {len(modifier_chars)} chars")

    with open(os.path.join(rd, 'triple_tiers.json')) as f:
        tiers_data = json.load(f)
    real_confirmed: Set[str] = set()
    for entry in tiers_data['tiers'].get('CONFIRMED', []):
        real_confirmed.add(entry['triple_key'])
    print(f"  Real confirmed triples: {len(real_confirmed)}")

    # ------------------------------------------------------------------
    # 2. Build infrastructure
    # ------------------------------------------------------------------
    print("\n--- Step 2: Building infrastructure ---")

    eva_to_triple = build_eva_to_triple_lookup()
    print(f"  EVA-to-triple: {len(eva_to_triple)} entries")

    pharma_dict = _build_pharma_dict()
    print(f"  Pharma dict: {len(pharma_dict)} words")

    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    base_words = set(
        w.lower() for w in ref_corpus.get_combined_tokens('latin')
        if len(w) >= 2
    )
    expanded, _ = build_expanded_word_set(base_words)
    ref_word_set = base_words | expanded
    print(f"  Reference word set: {len(ref_word_set)} words")

    ci_vocab = set(
        w.lower() for w in ref_corpus.get_combined_tokens('latin')
        if len(w) >= 2
    )
    print(f"  CI vocabulary: {len(ci_vocab)} words")

    # Load real corpus tokens for decode
    corpus = load_corpus(verbose=False)
    real_tokens = corpus.get_tokens()
    print(f"  Real corpus tokens: {len(real_tokens)}")

    # Pre-generate 5 null corpora (EVA tokens)
    print("  Generating null corpora...")
    bigram_probs, initial_probs, token_lengths = _build_eva_bigram_model(
        real_tokens
    )
    null_token_lists: List[List[str]] = []
    for seed in NULL_SEEDS:
        null_tokens = _generate_null_corpus(
            bigram_probs, initial_probs, token_lengths,
            len(real_tokens), seed,
        )
        null_token_lists.append(null_tokens)
    print(f"  Null corpora: {len(null_token_lists)} (seeds {NULL_SEEDS})")

    # ------------------------------------------------------------------
    # 3. Real table baseline
    # ------------------------------------------------------------------
    print("\n--- Step 3: Real table baseline ---")

    # Decode real corpus with real table
    real_decoded = _decode_corpus_r3(
        real_tokens, real_assignment, eva_to_triple,
        modifier_chars, modifier_rules, ref_word_set,
    )

    # Run bridge search with hardcoded SIGNAL_WORDS_SET + real confirmed
    bridge_matches, _concat = _run_bridge_search_with_signal_set(
        token_evas, real_decoded,
        [''] * n_tokens,  # classifications not used by bridge search
        token_folios,
        real_assignment, eva_to_triple, modifier_chars, real_confirmed,
        pharma_dict, SIGNAL_WORDS_SET,
    )

    (real_n_bridge, real_n_T1, real_n_T2, real_n_T3,
     real_n_distinct, real_T1_words, real_ci_overlap,
     real_folio_spread) = _bridge_to_metrics(
        bridge_matches, token_evas, ci_vocab,
    )

    print(f"  Bridge matches: {real_n_bridge}")
    print(f"  T1: {real_n_T1}, T2: {real_n_T2}, T3: {real_n_T3}")
    print(f"  Distinct T1 words: {real_n_distinct}: {real_T1_words}")
    print(f"  CI overlap: {real_ci_overlap:.3f}")
    print(f"  Mean folio spread: {real_folio_spread:.1f}")

    # ------------------------------------------------------------------
    # 4. Permutation loop
    # ------------------------------------------------------------------
    print(f"\n--- Step 4: Running {N_TRIALS} permutation trials ---")

    trial_results: List[TrialResult] = []
    for trial_idx in range(N_TRIALS):
        tr = _run_single_trial(
            trial_idx, real_assignment, real_tokens,
            token_evas, token_folios, null_token_lists,
            eva_to_triple, modifier_chars, modifier_rules,
            ref_word_set, pharma_dict, ci_vocab,
        )
        trial_results.append(tr)

        if (trial_idx + 1) % 10 == 0:
            print(f"  Trial {trial_idx + 1}/{N_TRIALS}: "
                  f"T1={tr.n_T1} signal={tr.n_signal_words} "
                  f"confirmed={tr.n_confirmed_triples}")

    # ------------------------------------------------------------------
    # 5. Statistics
    # ------------------------------------------------------------------
    print("\n--- Step 5: Computing statistics ---")

    t1_counts = [tr.n_T1 for tr in trial_results]
    distinct_counts = [tr.n_distinct_T1_words for tr in trial_results]
    ci_overlaps = [tr.ci_overlap for tr in trial_results]
    folio_spreads = [tr.mean_folio_spread for tr in trial_results]

    def _z_and_p(real_val: float, null_vals: List[float]) -> Tuple[float, float]:
        arr = np.array(null_vals)
        m = float(np.mean(arr))
        s = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        z = (real_val - m) / s if s > 0 else (
            float('inf') if real_val > m else 0.0
        )
        p = float(np.mean(arr >= real_val))
        return z, p

    T1_z, T1_p = _z_and_p(real_n_T1, t1_counts)
    dw_z, dw_p = _z_and_p(real_n_distinct, distinct_counts)
    ci_z, _ = _z_and_p(real_ci_overlap, ci_overlaps)
    fs_z, _ = _z_and_p(real_folio_spread, folio_spreads)

    null_T1_mean = float(np.mean(t1_counts))
    null_T1_std = float(np.std(t1_counts, ddof=1)) if len(t1_counts) > 1 else 0.0
    null_T1_max = int(np.max(t1_counts))

    null_dw_mean = float(np.mean(distinct_counts))
    null_dw_std = float(np.std(distinct_counts, ddof=1)) if len(distinct_counts) > 1 else 0.0

    null_ci_mean = float(np.mean(ci_overlaps))
    null_ci_std = float(np.std(ci_overlaps, ddof=1)) if len(ci_overlaps) > 1 else 0.0

    null_fs_mean = float(np.mean(folio_spreads))
    null_fs_std = float(np.std(folio_spreads, ddof=1)) if len(folio_spreads) > 1 else 0.0

    print(f"  T1 count:       real={real_n_T1}, null={null_T1_mean:.1f}±{null_T1_std:.1f}, "
          f"max={null_T1_max}, z={T1_z:.2f}, p={T1_p:.4f}")
    print(f"  Distinct words: real={real_n_distinct}, null={null_dw_mean:.1f}±{null_dw_std:.1f}, "
          f"z={dw_z:.2f}, p={dw_p:.4f}")
    print(f"  CI overlap:     real={real_ci_overlap:.3f}, null={null_ci_mean:.3f}±{null_ci_std:.3f}, "
          f"z={ci_z:.2f}")
    print(f"  Folio spread:   real={real_folio_spread:.1f}, null={null_fs_mean:.1f}±{null_fs_std:.1f}, "
          f"z={fs_z:.2f}")

    # Signal word and confirmed triple distributions
    sig_counts = [tr.n_signal_words for tr in trial_results]
    conf_counts = [tr.n_confirmed_triples for tr in trial_results]
    print(f"\n  Signal words per trial:    {np.mean(sig_counts):.1f}±{np.std(sig_counts):.1f} "
          f"(range {min(sig_counts)}-{max(sig_counts)})")
    print(f"  Confirmed triples/trial:   {np.mean(conf_counts):.1f}±{np.std(conf_counts):.1f} "
          f"(range {min(conf_counts)}-{max(conf_counts)})")

    # ------------------------------------------------------------------
    # 6. Verdict
    # ------------------------------------------------------------------
    print("\n--- Step 6: Verdict ---")

    if T1_p > 0.05:
        verdict = "ARTIFACT"
    elif T1_p > 0.01:
        verdict = "MARGINAL"
    elif T1_z < 3.0:
        verdict = "MODERATE_SIGNAL"
    else:
        verdict = "GENUINE"

    print(f"  Verdict: {verdict}")

    # ------------------------------------------------------------------
    # 7. Save results
    # ------------------------------------------------------------------
    runtime = time.time() - t0
    print(f"\n  Runtime: {runtime:.1f}s")

    result = WordPermutationNullResult(
        real_n_T1=real_n_T1,
        real_n_distinct_words=real_n_distinct,
        real_ci_overlap=round(real_ci_overlap, 4),
        real_mean_folio_spread=round(real_folio_spread, 2),
        real_T1_words=real_T1_words,
        null_n_trials=N_TRIALS,
        null_T1_mean=round(null_T1_mean, 4),
        null_T1_std=round(null_T1_std, 4),
        null_T1_max=null_T1_max,
        null_distinct_words_mean=round(null_dw_mean, 4),
        null_distinct_words_std=round(null_dw_std, 4),
        null_ci_overlap_mean=round(null_ci_mean, 4),
        null_ci_overlap_std=round(null_ci_std, 4),
        null_folio_spread_mean=round(null_fs_mean, 4),
        null_folio_spread_std=round(null_fs_std, 4),
        T1_z_score=round(T1_z, 4) if T1_z != float('inf') else 999.0,
        T1_p_value=round(T1_p, 4),
        distinct_words_z=round(dw_z, 4) if dw_z != float('inf') else 999.0,
        distinct_words_p=round(dw_p, 4),
        ci_overlap_z=round(ci_z, 4) if ci_z != float('inf') else 999.0,
        folio_spread_z=round(fs_z, 4) if fs_z != float('inf') else 999.0,
        trial_results=[asdict(tr) for tr in trial_results],
        verdict=verdict,
        runtime_seconds=round(runtime, 2),
    )

    out_path = _save_json(rd, 'word_permutation_null.json', asdict(result))
    print(f"  Saved → {out_path}")
