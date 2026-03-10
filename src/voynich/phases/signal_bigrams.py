"""
Phase 29.1 – Signal-Filtered Bigram Plausibility
====================================================
Recomputes per-token SIGNAL/SHARED/ANTI classifications, caches them
for all downstream Phase 29 steps, then tests bigram plausibility on
consecutive SIGNAL-SIGNAL token pairs only.

Dependency chain:
    combined_refine.json      (Phase 15 assignment)
    modifier_integrate.json   (Phase 16 modifiers)
    null_corpus.json          (Phase 17 seeds)
        → signal_bigrams.json   (this step)
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
    load_corpus,
)
from voynich.core.reference import (
    build_expanded_word_set,
    load_reference_corpus,
)
from voynich.phases.csp_solver import decode_token
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
class FolioSignalPairStats:
    folio: str
    n_tokens: int
    n_signal: int
    signal_rate: float
    n_signal_pairs: int
    n_bigram_hits: int
    bigram_hit_rate: float


@dataclass
class SignalBigramResult:
    # Per-token cache (parallel arrays)
    token_folios: List[str]
    token_evas: List[str]
    token_decoded: List[str]
    token_classifications: List[str]
    token_dict_hits: List[bool]
    n_tokens: int
    n_signal: int
    signal_rate: float

    # Reference bigram table
    ref_bigram_count: int
    ref_trigram_count: int

    # SIGNAL bigram test
    n_signal_pairs: int
    n_bigram_hits: int
    bigram_hit_rate: float
    bigram_hit_pairs: List[List[str]]       # actual matching pairs
    null_bigram_mean: float
    null_bigram_std: float
    bigram_p_value: float
    bigram_z_score: float

    # SIGNAL trigram test
    n_signal_triples: int
    n_trigram_hits: int
    trigram_hit_rate: float
    trigram_hit_triples: List[List[str]]

    # Relaxed bigram test
    n_relaxed_bigram_hits: int
    relaxed_bigram_hit_rate: float

    # Per-folio ranking
    folio_signal_pair_stats: List[Dict]

    gate_passed: bool
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Token classification
# ---------------------------------------------------------------------------

def _recompute_token_classifications(
    rd: str,
) -> Tuple[
    List[str], List[str], List[str], List[str], List[bool],
    set, List[str],
]:
    """Recompute per-token classifications from scratch.

    Returns:
        (token_folios, token_evas, token_decoded,
         token_classifications, token_dict_hits,
         ref_word_set, base_words_list)
    """
    # Load inputs
    with open(os.path.join(rd, 'combined_refine.json')) as f:
        refine_data = json.load(f)
    assignment = refine_data.get('best_assignment', {})

    with open(os.path.join(rd, 'modifier_integrate.json')) as f:
        mod_data = json.load(f)
    modifier_chars, modifier_rules = _reconstruct_modifier_rules(mod_data)

    null_seeds = [100, 101, 102, 103, 104]
    null_path = os.path.join(rd, 'null_corpus.json')
    if os.path.exists(null_path):
        with open(null_path) as f:
            null_data = json.load(f)
        null_seeds = [r['seed'] for r in null_data.get('null_runs', [])]

    # Build reference word set
    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    base_words = set(
        w.lower() for w in ref_corpus.get_combined_tokens('latin')
        if len(w) >= 2
    )
    expanded, _ = build_expanded_word_set(base_words)
    ref_word_set = base_words | expanded

    # Decode real corpus with folio tracking
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

    # Regenerate and decode null corpora
    bigram_probs, initial_probs, token_lengths = _build_eva_bigram_model(
        all_tokens,
    )
    null_hits_list: List[List[bool]] = []
    for seed in null_seeds:
        null_tokens = _generate_null_corpus(
            bigram_probs, initial_probs, token_lengths, n_tokens, seed,
        )
        null_decoded = _decode_corpus_r3(
            null_tokens, assignment, eva_to_triple,
            modifier_chars, modifier_rules, ref_word_set,
        )
        null_hits_list.append([w in ref_word_set for w in null_decoded])

    # Classify each token
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

    dict_hits = [c in ('SIGNAL', 'SHARED_HIT') for c in classifications]

    # Return ref tokens for bigram building
    ref_tokens = [w.lower() for w in ref_corpus.get_combined_tokens('latin')
                  if len(w) >= 2]

    return (
        token_folios, token_evas, real_decoded,
        classifications, dict_hits,
        ref_word_set, ref_tokens,
    )


# ---------------------------------------------------------------------------
# Bigram / trigram reference table
# ---------------------------------------------------------------------------

def _build_reference_bigrams(
    ref_tokens: List[str],
) -> Tuple[Set[Tuple[str, str]], Set[Tuple[str, str, str]]]:
    """Build word-level bigram and trigram sets from reference corpus."""
    bigrams: Set[Tuple[str, str]] = set()
    trigrams: Set[Tuple[str, str, str]] = set()

    for i in range(len(ref_tokens) - 1):
        bigrams.add((ref_tokens[i], ref_tokens[i + 1]))
    for i in range(len(ref_tokens) - 2):
        trigrams.add((ref_tokens[i], ref_tokens[i + 1], ref_tokens[i + 2]))

    return bigrams, trigrams


# ---------------------------------------------------------------------------
# Signal pair detection
# ---------------------------------------------------------------------------

def _find_signal_pairs(
    classifications: List[str],
    decoded: List[str],
    folios: List[str],
) -> List[Tuple[str, int, str, str]]:
    """Find consecutive SIGNAL-SIGNAL pairs respecting folio boundaries.

    Returns list of (folio, position_i, word_i, word_i+1).
    """
    pairs = []
    for i in range(len(classifications) - 1):
        if (classifications[i] == 'SIGNAL'
                and classifications[i + 1] == 'SIGNAL'
                and folios[i] == folios[i + 1]):
            pairs.append((folios[i], i, decoded[i], decoded[i + 1]))
    return pairs


def _find_signal_triples(
    classifications: List[str],
    decoded: List[str],
    folios: List[str],
) -> List[Tuple[str, int, str, str, str]]:
    """Find consecutive SIGNAL-SIGNAL-SIGNAL triples respecting folio boundaries."""
    triples = []
    for i in range(len(classifications) - 2):
        if (classifications[i] == 'SIGNAL'
                and classifications[i + 1] == 'SIGNAL'
                and classifications[i + 2] == 'SIGNAL'
                and folios[i] == folios[i + 1]
                and folios[i + 1] == folios[i + 2]):
            triples.append((
                folios[i], i,
                decoded[i], decoded[i + 1], decoded[i + 2],
            ))
    return triples


# ---------------------------------------------------------------------------
# Null permutation test
# ---------------------------------------------------------------------------

def _null_permutation_test(
    n_signal: int,
    n_tokens: int,
    decoded: List[str],
    folios: List[str],
    ref_bigrams: Set[Tuple[str, str]],
    n_perms: int = 1000,
    seed: int = 42,
) -> Tuple[List[float], float, float]:
    """Random relabeling null test for bigram hit rate.

    Randomly labels n_signal positions as 'SIGNAL', computes bigram hit
    rate on consecutive pairs.  Returns (null_rates, null_mean, null_std).
    """
    rng = random.Random(seed)
    indices = list(range(n_tokens))
    null_rates: List[float] = []

    for _ in range(n_perms):
        fake_signal = set(rng.sample(indices, n_signal))
        n_pairs = 0
        n_hits = 0
        for i in range(n_tokens - 1):
            if i in fake_signal and (i + 1) in fake_signal and folios[i] == folios[i + 1]:
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
    return null_rates, null_mean, null_std


# ---------------------------------------------------------------------------
# Relaxed bigram matching (edit distance 1)
# ---------------------------------------------------------------------------

def _edit_distance_1(word: str) -> Set[str]:
    """Generate all strings within edit distance 1 of word."""
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    variants: Set[str] = set()
    for i in range(len(word)):
        # deletion
        variants.add(word[:i] + word[i + 1:])
        # substitution
        for c in alphabet:
            if c != word[i]:
                variants.add(word[:i] + c + word[i + 1:])
    # insertion
    for i in range(len(word) + 1):
        for c in alphabet:
            variants.add(word[:i] + c + word[i:])
    return variants


def _relaxed_bigram_test(
    signal_pairs: List[Tuple[str, int, str, str]],
    ref_bigrams: Set[Tuple[str, str]],
    ref_words: Set[str],
) -> int:
    """Count signal pairs within edit distance 1 of a reference bigram.

    For efficiency, check if edit-1 variants of each word in the pair
    form a reference bigram with the other word (or its edit-1 variants).
    """
    n_relaxed = 0
    for _, _, w1, w2 in signal_pairs:
        if (w1, w2) in ref_bigrams:
            continue  # already an exact match
        # Check w1-variants with exact w2
        found = False
        for v1 in _edit_distance_1(w1):
            if (v1, w2) in ref_bigrams:
                found = True
                break
        if not found:
            # Check exact w1 with w2-variants
            for v2 in _edit_distance_1(w2):
                if (w1, v2) in ref_bigrams:
                    found = True
                    break
        if found:
            n_relaxed += 1
    return n_relaxed


# ---------------------------------------------------------------------------
# Per-folio ranking
# ---------------------------------------------------------------------------

def _folio_signal_pair_ranking(
    signal_pairs: List[Tuple[str, int, str, str]],
    ref_bigrams: Set[Tuple[str, str]],
    classifications: List[str],
    folios: List[str],
) -> List[FolioSignalPairStats]:
    """Rank folios by SIGNAL-pair bigram hit rate."""
    # Count tokens and signal per folio
    folio_n: Dict[str, int] = Counter(folios)
    folio_n_signal: Dict[str, int] = Counter()
    for folio, cls in zip(folios, classifications):
        if cls == 'SIGNAL':
            folio_n_signal[folio] += 1

    # Count pairs and hits per folio
    folio_pairs: Dict[str, int] = Counter()
    folio_hits: Dict[str, int] = Counter()
    for folio, _, w1, w2 in signal_pairs:
        folio_pairs[folio] += 1
        if (w1, w2) in ref_bigrams:
            folio_hits[folio] += 1

    stats = []
    for folio in sorted(folio_n.keys()):
        n_tok = folio_n[folio]
        n_sig = folio_n_signal.get(folio, 0)
        n_p = folio_pairs.get(folio, 0)
        n_h = folio_hits.get(folio, 0)
        stats.append(FolioSignalPairStats(
            folio=folio,
            n_tokens=n_tok,
            n_signal=n_sig,
            signal_rate=round(n_sig / n_tok, 4) if n_tok > 0 else 0.0,
            n_signal_pairs=n_p,
            n_bigram_hits=n_h,
            bigram_hit_rate=round(n_h / n_p, 4) if n_p > 0 else 0.0,
        ))

    # Sort by signal_rate descending (bigram_hit_rate as tiebreak)
    stats.sort(key=lambda s: (-s.signal_rate, -s.bigram_hit_rate))
    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_signal_bigrams() -> None:
    """Step 29.1: Signal-filtered bigram plausibility."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE 29.1: Signal-Filtered Bigram Plausibility")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Recompute per-token classifications ──
    print("\n  1. Recomputing per-token classifications …")
    (
        token_folios, token_evas, token_decoded,
        token_classifications, token_dict_hits,
        ref_word_set, ref_tokens,
    ) = _recompute_token_classifications(rd)

    n_tokens = len(token_decoded)
    n_signal = sum(1 for c in token_classifications if c == 'SIGNAL')
    signal_rate = n_signal / n_tokens if n_tokens > 0 else 0.0

    print(f"     {n_tokens} tokens, {n_signal} SIGNAL ({signal_rate:.1%})")
    cls_counts = Counter(token_classifications)
    for cls in ['SIGNAL', 'SHARED_HIT', 'SHARED_MISS', 'ANTI_SIGNAL']:
        print(f"       {cls:14s}: {cls_counts.get(cls, 0):6d}")

    # ── 2. Build reference bigram/trigram table ──
    print("\n  2. Building reference bigram/trigram table …")
    ref_bigrams, ref_trigrams = _build_reference_bigrams(ref_tokens)
    print(f"     {len(ref_bigrams)} unique bigrams, "
          f"{len(ref_trigrams)} unique trigrams")

    # ── 3. Find SIGNAL-SIGNAL pairs ──
    print("\n  3. Finding SIGNAL-SIGNAL pairs …")
    signal_pairs = _find_signal_pairs(
        token_classifications, token_decoded, token_folios,
    )
    print(f"     {len(signal_pairs)} consecutive SIGNAL-SIGNAL pairs")

    # ── 4. Bigram plausibility on SIGNAL pairs ──
    print("\n  4. Testing bigram plausibility on SIGNAL pairs …")
    bigram_hits = []
    for folio, pos, w1, w2 in signal_pairs:
        if (w1, w2) in ref_bigrams:
            bigram_hits.append([w1, w2])

    n_bigram_hits = len(bigram_hits)
    bigram_hit_rate = (
        n_bigram_hits / len(signal_pairs) if signal_pairs else 0.0
    )
    print(f"     {n_bigram_hits} bigram hits out of {len(signal_pairs)} pairs")
    print(f"     Bigram hit rate: {bigram_hit_rate:.4f}")

    if bigram_hits:
        print("     Matching pairs:")
        for pair in bigram_hits[:20]:
            print(f"       {pair[0]} {pair[1]}")

    # ── 5. Null permutation test ──
    print("\n  5. Null permutation test (1000 permutations) …")
    null_rates, null_mean, null_std = _null_permutation_test(
        n_signal, n_tokens, token_decoded, token_folios,
        ref_bigrams, n_perms=1000, seed=42,
    )

    if null_std > 0:
        z_score = (bigram_hit_rate - null_mean) / null_std
    else:
        z_score = float('inf') if bigram_hit_rate > null_mean else 0.0

    p_value = sum(1 for r in null_rates if r >= bigram_hit_rate) / len(null_rates)

    print(f"     Null mean: {null_mean:.6f}, std: {null_std:.6f}")
    print(f"     z-score: {z_score:.2f}, p-value: {p_value:.4f}")

    # ── 6. Per-folio ranking ──
    print("\n  6. Per-folio SIGNAL pair ranking (top 10) …")
    folio_stats = _folio_signal_pair_ranking(
        signal_pairs, ref_bigrams, token_classifications, token_folios,
    )
    for fs in folio_stats[:10]:
        print(f"     {fs.folio:8s}  signal={fs.n_signal:3d}/{fs.n_tokens:3d} "
              f"({fs.signal_rate:.1%})  "
              f"pairs={fs.n_signal_pairs:3d}  "
              f"bigram_hits={fs.n_bigram_hits}")

    # ── 7. SIGNAL trigram test ──
    print("\n  7. SIGNAL trigram test …")
    signal_triples = _find_signal_triples(
        token_classifications, token_decoded, token_folios,
    )
    trigram_hits = []
    for folio, pos, w1, w2, w3 in signal_triples:
        if (w1, w2, w3) in ref_trigrams:
            trigram_hits.append([w1, w2, w3])

    n_trigram_hits = len(trigram_hits)
    trigram_hit_rate = (
        n_trigram_hits / len(signal_triples) if signal_triples else 0.0
    )
    print(f"     {len(signal_triples)} SIGNAL triples, "
          f"{n_trigram_hits} trigram hits")

    if trigram_hits:
        print("     Matching triples:")
        for tri in trigram_hits[:10]:
            print(f"       {' '.join(tri)}")

    # ── 8. Relaxed bigram test ──
    print("\n  8. Relaxed bigram test (edit distance 1) …")
    n_relaxed = _relaxed_bigram_test(signal_pairs, ref_bigrams, ref_word_set)
    relaxed_rate = (
        (n_bigram_hits + n_relaxed) / len(signal_pairs)
        if signal_pairs else 0.0
    )
    print(f"     {n_relaxed} additional relaxed matches")
    print(f"     Combined rate (exact + relaxed): {relaxed_rate:.4f}")

    # ── 9. Gate and verdict ──
    bigram_above_null = z_score > 2.0 and p_value < 0.05
    gate_passed = n_bigram_hits > 0 or n_trigram_hits > 0 or n_relaxed > 0
    verdict = (
        f"SIGNAL bigram hit_rate={bigram_hit_rate:.4f}, "
        f"z={z_score:.2f}, p={p_value:.4f}. "
        f"{'ABOVE NULL' if bigram_above_null else 'NOT above null'}. "
        f"{n_trigram_hits} trigram hits, {n_relaxed} relaxed matches."
    )
    print(f"\n  {'PASS' if gate_passed else 'NO MATCHES'}: {verdict}")

    # ── 10. Save ──
    result = SignalBigramResult(
        token_folios=token_folios,
        token_evas=token_evas,
        token_decoded=token_decoded,
        token_classifications=token_classifications,
        token_dict_hits=token_dict_hits,
        n_tokens=n_tokens,
        n_signal=n_signal,
        signal_rate=round(signal_rate, 4),
        ref_bigram_count=len(ref_bigrams),
        ref_trigram_count=len(ref_trigrams),
        n_signal_pairs=len(signal_pairs),
        n_bigram_hits=n_bigram_hits,
        bigram_hit_rate=round(bigram_hit_rate, 6),
        bigram_hit_pairs=bigram_hits[:50],
        null_bigram_mean=round(null_mean, 6),
        null_bigram_std=round(null_std, 6),
        bigram_p_value=round(p_value, 4),
        bigram_z_score=round(z_score, 2) if z_score != float('inf') else 999.0,
        n_signal_triples=len(signal_triples),
        n_trigram_hits=n_trigram_hits,
        trigram_hit_rate=round(trigram_hit_rate, 6),
        trigram_hit_triples=trigram_hits[:20],
        n_relaxed_bigram_hits=n_relaxed,
        relaxed_bigram_hit_rate=round(relaxed_rate, 6),
        folio_signal_pair_stats=[
            _convert(asdict(fs)) for fs in folio_stats[:30]
        ],
        gate_passed=gate_passed,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'signal_bigrams.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)
    print(f"\n  → {out_path}")
