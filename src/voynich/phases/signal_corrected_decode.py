"""
Step 33.4 – Signal-Corrected Full Decode + Validation
========================================================
Applies the SIGNAL-corrected assignment table from Step 33.3 to the full
corpus, then re-runs the complete Phase 28-29 signal pipeline (token
classification, SIGNAL-SIGNAL bigram plausibility, null permutation test,
held-out folio split) to validate the corrections.

Dependency chain:
    signal_guided_swap.json  (Step 33.3 — corrected assignment)
    combined_refine.json     (Phase 15 — fallback if no swaps)
    modifier_integrate.json  (Phase 16 modifiers)
    null_corpus.json         (Phase 17 seeds)
    signal_bigrams.json      (Phase 29.1 — baseline comparison)
        -> signal_corrected_decode.json  (this step)
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
from voynich.core.reference import build_expanded_word_set, load_reference_corpus
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
class SignalCorrectedDecodeResult:
    # Assignment info
    n_swaps_applied: int
    swaps_applied: List[Dict]  # from signal_guided_swap.json
    # Full corpus metrics
    n_tokens: int
    dict_hit: float
    n_signal: int
    signal_rate: float
    n_anti_signal: int
    anti_signal_rate: float
    n_signal_pairs: int
    n_bigram_hits: int
    bigram_hit_rate: float
    bigram_z_score: float
    bigram_p_value: float
    # Baseline comparison
    baseline_dict_hit: float
    baseline_signal_rate: float
    baseline_bigram_z: float
    delta_dict_hit: float
    delta_signal_rate: float
    delta_bigram_z: float
    # Held-out validation
    train_signal_rate: float
    train_bigram_z: float
    test_signal_rate: float
    test_bigram_z: float
    held_out_transfers: bool
    # Anti-signal word analysis
    n_anti_signal_words: int
    top_anti_signal_words: List[Dict]
    # Signal words
    n_signal_words: int
    top_signal_words: List[Dict]
    # Verdict
    improvement: bool
    verdict: str  # 'SIGNAL_IMPROVED', 'SIGNAL_UNCHANGED', 'SIGNAL_DEGRADED'
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Folio number extraction
# ---------------------------------------------------------------------------

def _folio_number(folio: str) -> int:
    """Extract numeric part from folio name (e.g. 'f1r' -> 1, 'f70v2' -> 70)."""
    return int(''.join(c for c in folio if c.isdigit()))


# ---------------------------------------------------------------------------
# Held-out bigram z-score computation
# ---------------------------------------------------------------------------

def _compute_bigram_z(
    classifications: List[str],
    decoded: List[str],
    folios: List[str],
    ref_bigrams: Set[Tuple[str, str]],
    subset_indices: Optional[Set[int]] = None,
) -> Tuple[int, int, int, float, float, float]:
    """Compute SIGNAL bigram stats and z-score for a subset of token indices.

    If subset_indices is None, uses all tokens.

    Returns:
        (n_signal, n_signal_pairs, n_bigram_hits,
         bigram_hit_rate, bigram_z, bigram_p)
    """
    n_total = len(decoded)

    # Build effective classification list for subset
    if subset_indices is not None:
        eff_cls = [
            classifications[i] if i in subset_indices else 'EXCLUDED'
            for i in range(n_total)
        ]
    else:
        eff_cls = classifications

    n_signal = sum(1 for c in eff_cls if c == 'SIGNAL')

    # Find SIGNAL-SIGNAL consecutive pairs within folio boundaries
    n_signal_pairs = 0
    n_bigram_hits = 0
    for i in range(n_total - 1):
        if (eff_cls[i] == 'SIGNAL'
                and eff_cls[i + 1] == 'SIGNAL'
                and folios[i] == folios[i + 1]):
            n_signal_pairs += 1
            if (decoded[i], decoded[i + 1]) in ref_bigrams:
                n_bigram_hits += 1

    bigram_hit_rate = n_bigram_hits / n_signal_pairs if n_signal_pairs > 0 else 0.0

    # Null permutation test
    if n_signal < 2 or n_total < 2:
        return n_signal, n_signal_pairs, n_bigram_hits, bigram_hit_rate, 0.0, 1.0

    rng = random.Random(42)
    # Build valid index pool (only indices in subset)
    if subset_indices is not None:
        valid_indices = sorted(subset_indices)
    else:
        valid_indices = list(range(n_total))

    null_rates: List[float] = []
    n_perms = 1000
    sample_size = min(n_signal, len(valid_indices))

    for _ in range(n_perms):
        fake_signal = set(rng.sample(valid_indices, sample_size))
        n_pairs = 0
        n_hits = 0
        for i in range(n_total - 1):
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

    if null_std > 0:
        bigram_z = (bigram_hit_rate - null_mean) / null_std
    else:
        bigram_z = float('inf') if bigram_hit_rate > null_mean else 0.0

    bigram_p = sum(1 for r in null_rates if r >= bigram_hit_rate) / len(null_rates)

    return n_signal, n_signal_pairs, n_bigram_hits, bigram_hit_rate, bigram_z, bigram_p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_signal_corrected_decode() -> None:
    """Step 33.4: Apply SIGNAL-corrected table and validate via full pipeline."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 33.4: Signal-Corrected Full Decode + Validation")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load corrected assignment ──
    print("\n  1. Loading corrected assignment …")

    swaps_applied: List[Dict] = []
    n_swaps_applied = 0

    swap_path = os.path.join(rd, 'signal_guided_swap.json')
    refine_path = os.path.join(rd, 'combined_refine.json')

    if os.path.exists(swap_path):
        with open(swap_path) as f:
            swap_data = json.load(f)
        swaps_applied = swap_data.get('accepted_swaps', [])
        n_swaps_applied = len(swaps_applied)
        assignment = swap_data.get('new_assignment', {})
        if not assignment or n_swaps_applied == 0:
            # No swaps accepted — fall back
            print("     signal_guided_swap.json has 0 swaps; falling back")
            with open(refine_path) as f:
                refine_data = json.load(f)
            assignment = refine_data.get('best_assignment', {})
            swaps_applied = []
            n_swaps_applied = 0
        else:
            print(f"     Loaded corrected assignment ({n_swaps_applied} swaps)")
    elif os.path.exists(refine_path):
        with open(refine_path) as f:
            refine_data = json.load(f)
        assignment = refine_data.get('best_assignment', {})
        print(f"     No signal_guided_swap.json; using Phase 15 baseline")
    else:
        print("  [SKIP] No assignment file found")
        return

    print(f"     Assignment: {len(assignment)} triples")
    if swaps_applied:
        for sw in swaps_applied[:10]:
            print(f"       {sw.get('triple', '?')}: "
                  f"{sw.get('old_syllable', '?')} -> {sw.get('new_syllable', '?')}")

    # ── 2. Load modifier rules ──
    print("\n  2. Loading modifier rules …")
    mod_path = os.path.join(rd, 'modifier_integrate.json')
    if not os.path.exists(mod_path):
        print("  [SKIP] modifier_integrate.json not found")
        return
    with open(mod_path) as f:
        mod_data = json.load(f)
    modifier_chars, modifier_rules = _reconstruct_modifier_rules(mod_data)
    print(f"     {len(modifier_chars)} modifier chars")

    # ── 3. Load null seeds ──
    print("\n  3. Loading null corpus seeds …")
    null_seeds = [100, 101, 102, 103, 104]
    null_path = os.path.join(rd, 'null_corpus.json')
    if os.path.exists(null_path):
        with open(null_path) as f:
            null_data = json.load(f)
        null_seeds = [r['seed'] for r in null_data.get('null_runs', [])]
    print(f"     Null seeds: {null_seeds}")

    # ── 4. Build reference word set ──
    print("\n  4. Building reference word set …")
    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    base_words = set(
        w.lower() for w in ref_corpus.get_combined_tokens('latin')
        if len(w) >= 2
    )
    expanded, _ = build_expanded_word_set(base_words)
    ref_word_set = base_words | expanded
    print(f"     {len(ref_word_set)} reference words")

    # Build reference bigrams
    ref_tokens = [
        w.lower() for w in ref_corpus.get_combined_tokens('latin')
        if len(w) >= 2
    ]
    ref_bigrams: Set[Tuple[str, str]] = set()
    for i in range(len(ref_tokens) - 1):
        ref_bigrams.add((ref_tokens[i], ref_tokens[i + 1]))
    print(f"     {len(ref_bigrams)} reference bigrams")

    # ── 5. Decode full corpus with corrected assignment (R3 strategy) ──
    print("\n  5. Decoding full corpus with corrected assignment …")
    corpus = load_corpus(verbose=False)
    eva_to_triple = build_eva_to_triple_lookup()

    token_folios: List[str] = []
    token_evas: List[str] = []
    all_tokens: List[str] = []

    for folio, page in corpus.pages.items():
        for token in page.all_tokens:
            token_folios.append(folio)
            token_evas.append(token)
            all_tokens.append(token)

    n_tokens = len(all_tokens)

    real_decoded = _decode_corpus_r3(
        all_tokens, assignment, eva_to_triple,
        modifier_chars, modifier_rules, ref_word_set,
    )
    real_hits = [w in ref_word_set for w in real_decoded]
    dict_hit = sum(real_hits) / n_tokens if n_tokens > 0 else 0.0
    print(f"     {n_tokens} tokens, dict_hit = {dict_hit:.4f}")

    # ── 6. Regenerate 5 null corpora from seeds, decode each ──
    print("\n  6. Regenerating and decoding null corpora …")
    bigram_probs, initial_probs, token_lengths = _build_eva_bigram_model(
        all_tokens,
    )
    null_hits_list: List[List[bool]] = []
    for i, seed in enumerate(null_seeds):
        print(f"     Null corpus {i + 1}/{len(null_seeds)} (seed={seed}) …")
        null_tokens = _generate_null_corpus(
            bigram_probs, initial_probs, token_lengths, n_tokens, seed,
        )
        null_decoded = _decode_corpus_r3(
            null_tokens, assignment, eva_to_triple,
            modifier_chars, modifier_rules, ref_word_set,
        )
        null_hits_list.append([w in ref_word_set for w in null_decoded])

    # ── 7. Classify every token ──
    print("\n  7. Classifying tokens (SIGNAL/SHARED_HIT/SHARED_MISS/ANTI_SIGNAL) …")
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

    cls_counts = Counter(classifications)
    n_signal = cls_counts.get('SIGNAL', 0)
    n_anti_signal = cls_counts.get('ANTI_SIGNAL', 0)
    signal_rate = n_signal / n_tokens if n_tokens > 0 else 0.0
    anti_signal_rate = n_anti_signal / n_tokens if n_tokens > 0 else 0.0

    for cls in ['SIGNAL', 'SHARED_HIT', 'SHARED_MISS', 'ANTI_SIGNAL']:
        print(f"       {cls:14s}: {cls_counts.get(cls, 0):6d}")
    print(f"     SIGNAL rate: {signal_rate:.4f}")

    # ── 8. Find SIGNAL-SIGNAL consecutive pairs (same folio) ──
    print("\n  8. Finding SIGNAL-SIGNAL consecutive pairs …")
    signal_pairs: List[Tuple[int, str, str]] = []  # (position, word_i, word_i+1)
    for i in range(n_tokens - 1):
        if (classifications[i] == 'SIGNAL'
                and classifications[i + 1] == 'SIGNAL'
                and token_folios[i] == token_folios[i + 1]):
            signal_pairs.append((i, real_decoded[i], real_decoded[i + 1]))

    n_signal_pairs = len(signal_pairs)
    print(f"     {n_signal_pairs} consecutive SIGNAL-SIGNAL pairs")

    # ── 9-10. Count bigram hits ──
    print("\n  9. Testing bigram hits …")
    n_bigram_hits = sum(
        1 for _, w1, w2 in signal_pairs
        if (w1, w2) in ref_bigrams
    )
    bigram_hit_rate = n_bigram_hits / n_signal_pairs if n_signal_pairs > 0 else 0.0
    print(f"     {n_bigram_hits} bigram hits out of {n_signal_pairs} pairs")
    print(f"     Bigram hit rate: {bigram_hit_rate:.6f}")

    # ── 11. Null permutation test for bigram z ──
    print("\n  10. Null permutation test (1000 relabelings) …")
    rng = random.Random(42)
    indices = list(range(n_tokens))
    null_rates: List[float] = []

    for _ in range(1000):
        fake_signal = set(rng.sample(indices, n_signal))
        n_pairs = 0
        n_hits = 0
        for i in range(n_tokens - 1):
            if (i in fake_signal and (i + 1) in fake_signal
                    and token_folios[i] == token_folios[i + 1]):
                n_pairs += 1
                if (real_decoded[i], real_decoded[i + 1]) in ref_bigrams:
                    n_hits += 1
        rate = n_hits / n_pairs if n_pairs > 0 else 0.0
        null_rates.append(rate)

    null_mean = sum(null_rates) / len(null_rates) if null_rates else 0.0
    null_var = (
        sum((r - null_mean) ** 2 for r in null_rates) / len(null_rates)
        if null_rates else 0.0
    )
    null_std = null_var ** 0.5

    if null_std > 0:
        bigram_z_score = (bigram_hit_rate - null_mean) / null_std
    else:
        bigram_z_score = float('inf') if bigram_hit_rate > null_mean else 0.0

    bigram_p_value = sum(
        1 for r in null_rates if r >= bigram_hit_rate
    ) / len(null_rates)

    z_display = (
        round(bigram_z_score, 2)
        if bigram_z_score != float('inf') else 999.0
    )

    print(f"     Null mean: {null_mean:.6f}, std: {null_std:.6f}")
    print(f"     z-score: {z_display}")
    print(f"     p-value: {bigram_p_value:.4f}")

    # ── 12. Held-out validation: split corpus 50/50 by folio ──
    print("\n  11. Held-out validation (odd/even folio split) …")

    # Build folio -> token indices mapping
    folio_to_indices: Dict[str, List[int]] = defaultdict(list)
    for idx, folio in enumerate(token_folios):
        folio_to_indices[folio].append(idx)

    train_indices: Set[int] = set()  # odd folio numbers
    test_indices: Set[int] = set()   # even folio numbers

    for folio, idxs in folio_to_indices.items():
        fnum = _folio_number(folio)
        if fnum % 2 == 1:
            train_indices.update(idxs)
        else:
            test_indices.update(idxs)

    print(f"     Train (odd folios): {len(train_indices)} tokens")
    print(f"     Test (even folios): {len(test_indices)} tokens")

    # Compute signal rate and bigram z on each half
    (train_n_signal, train_n_pairs, train_n_hits,
     train_hit_rate, train_bigram_z, train_p) = _compute_bigram_z(
        classifications, real_decoded, token_folios,
        ref_bigrams, subset_indices=train_indices,
    )
    train_signal_rate = train_n_signal / len(train_indices) if train_indices else 0.0
    train_z_display = (
        round(train_bigram_z, 2)
        if train_bigram_z != float('inf') else 999.0
    )

    (test_n_signal, test_n_pairs, test_n_hits,
     test_hit_rate, test_bigram_z, test_p) = _compute_bigram_z(
        classifications, real_decoded, token_folios,
        ref_bigrams, subset_indices=test_indices,
    )
    test_signal_rate = test_n_signal / len(test_indices) if test_indices else 0.0
    test_z_display = (
        round(test_bigram_z, 2)
        if test_bigram_z != float('inf') else 999.0
    )

    # Transfer passes if both halves have z > 2.0
    held_out_transfers = train_z_display > 2.0 and test_z_display > 2.0

    print(f"     Train: signal_rate={train_signal_rate:.4f}, "
          f"bigram_z={train_z_display}")
    print(f"     Test:  signal_rate={test_signal_rate:.4f}, "
          f"bigram_z={test_z_display}")
    print(f"     Held-out transfers: {held_out_transfers}")

    # ── 13. Compare to Phase 29 baseline ──
    print("\n  12. Comparing to Phase 29 baseline …")
    baseline_dict_hit = 0.0
    baseline_signal_rate = 0.1652
    baseline_bigram_z = 6.14

    bg_path = os.path.join(rd, 'signal_bigrams.json')
    if os.path.exists(bg_path):
        with open(bg_path) as f:
            bg_data = json.load(f)
        baseline_bigram_z = bg_data.get('bigram_z_score', 6.14)
        baseline_signal_rate = bg_data.get('signal_rate', 0.1652)
        # Compute baseline dict_hit from classifications
        bg_cls = bg_data.get('token_classifications', [])
        if bg_cls:
            bg_n = len(bg_cls)
            bg_hits = sum(
                1 for c in bg_cls
                if c in ('SIGNAL', 'SHARED_HIT')
            )
            baseline_dict_hit = bg_hits / bg_n if bg_n > 0 else 0.0
        else:
            baseline_dict_hit = dict_hit  # no baseline available

    delta_dict_hit = dict_hit - baseline_dict_hit
    delta_signal_rate = signal_rate - baseline_signal_rate
    delta_bigram_z = z_display - baseline_bigram_z

    print(f"     dict_hit:    {baseline_dict_hit:.4f} -> {dict_hit:.4f} "
          f"(delta={delta_dict_hit:+.4f})")
    print(f"     signal_rate: {baseline_signal_rate:.4f} -> {signal_rate:.4f} "
          f"(delta={delta_signal_rate:+.4f})")
    print(f"     bigram_z:    {baseline_bigram_z:.2f} -> {z_display} "
          f"(delta={delta_bigram_z:+.2f})")

    # ── 14. Anti-signal word analysis ──
    print("\n  13. Anti-signal word analysis …")
    anti_signal_word_counts: Counter = Counter()
    signal_word_counts: Counter = Counter()

    for idx in range(n_tokens):
        w = real_decoded[idx]
        if classifications[idx] == 'ANTI_SIGNAL':
            anti_signal_word_counts[w] += 1
        elif classifications[idx] == 'SIGNAL':
            signal_word_counts[w] += 1

    n_anti_signal_words = len(anti_signal_word_counts)
    top_anti_signal_words = [
        {'word': w, 'count': c}
        for w, c in anti_signal_word_counts.most_common(20)
    ]

    n_signal_words = len(signal_word_counts)
    top_signal_words = [
        {'word': w, 'count': c}
        for w, c in signal_word_counts.most_common(20)
    ]

    print(f"     {n_anti_signal_words} unique anti-signal words")
    if top_anti_signal_words:
        print("     Top anti-signal:")
        for entry in top_anti_signal_words[:10]:
            print(f"       {entry['word']:15s}  count={entry['count']}")

    print(f"     {n_signal_words} unique signal words")
    if top_signal_words:
        print("     Top signal:")
        for entry in top_signal_words[:10]:
            print(f"       {entry['word']:15s}  count={entry['count']}")

    # ── 15. Verdict ──
    print("\n  14. Verdict …")

    if delta_bigram_z > 0.5 and delta_signal_rate > 0.0:
        improvement = True
        verdict = 'SIGNAL_IMPROVED'
    elif delta_bigram_z < -0.5 or delta_signal_rate < -0.01:
        improvement = False
        verdict = 'SIGNAL_DEGRADED'
    else:
        improvement = False
        verdict = 'SIGNAL_UNCHANGED'

    print(f"     Improvement: {improvement}")
    print(f"     Verdict: {verdict}")

    # ── 16. Save ──
    result = SignalCorrectedDecodeResult(
        n_swaps_applied=n_swaps_applied,
        swaps_applied=swaps_applied,
        n_tokens=n_tokens,
        dict_hit=round(dict_hit, 6),
        n_signal=n_signal,
        signal_rate=round(signal_rate, 6),
        n_anti_signal=n_anti_signal,
        anti_signal_rate=round(anti_signal_rate, 6),
        n_signal_pairs=n_signal_pairs,
        n_bigram_hits=n_bigram_hits,
        bigram_hit_rate=round(bigram_hit_rate, 6),
        bigram_z_score=z_display,
        bigram_p_value=round(bigram_p_value, 4),
        baseline_dict_hit=round(baseline_dict_hit, 6),
        baseline_signal_rate=round(baseline_signal_rate, 6),
        baseline_bigram_z=round(baseline_bigram_z, 2),
        delta_dict_hit=round(delta_dict_hit, 6),
        delta_signal_rate=round(delta_signal_rate, 6),
        delta_bigram_z=round(delta_bigram_z, 2),
        train_signal_rate=round(train_signal_rate, 6),
        train_bigram_z=train_z_display,
        test_signal_rate=round(test_signal_rate, 6),
        test_bigram_z=test_z_display,
        held_out_transfers=held_out_transfers,
        n_anti_signal_words=n_anti_signal_words,
        top_anti_signal_words=top_anti_signal_words,
        n_signal_words=n_signal_words,
        top_signal_words=top_signal_words,
        improvement=improvement,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'signal_corrected_decode.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)
    print(f"\n  -> {out_path}")
