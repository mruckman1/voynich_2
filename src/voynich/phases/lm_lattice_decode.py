"""
Phase 49 Track A – External LM Lattice Decode
==============================================
Replace internal bigram model with external n-gram LM trained on reference
corpora. Build per-token decode lattice with ED1 expansion, then beam search
for globally optimal decoding under the external LM.

Dependency chain:
    combined_refine.json        (Phase 15 best table)
    disamb_lattice.json         (Phase 47B per-token lattice)
    modifier_integrate.json     (Phase 16 modifiers)
    signal_bigrams.json         (Phase 29 parallel arrays)
    triple_tiers.json           (Phase 45 tier assignments)
    maxsat_landscape.json       (Phase 44 per-triple consensus)
        -> lm_build.json        (Step 49A.1)
        -> lm_lattice.json      (Step 49A.2)
        -> lm_viterbi.json      (Step 49A.3)
        -> lm_calibrate.json    (Step 49A.4)
        -> lm_decode.json       (Step 49A.5)
"""

from __future__ import annotations

import itertools
import json
import math
import os
import pickle
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
from voynich.core.stats import (
    build_ngram_lm,
    cross_entropy_lm,
    build_word_ngram_lm,
    cross_entropy_word_lm,
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
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if v != v else v
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return [_convert(item) for item in obj.tolist()]
    if isinstance(obj, float) and (obj != obj):
        return None
    if isinstance(obj, (bool, int, float, str, type(None))):
        return obj
    return str(obj)


def _safe_load(path: str) -> Dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _save_json(rd: str, filename: str, data: Any) -> str:
    path = os.path.join(rd, filename)
    with open(path, 'w') as f:
        json.dump(_convert(data), f, indent=2)
    return path


# ---------------------------------------------------------------------------
# Step 49A.1 — Build External N-Gram LMs
# ---------------------------------------------------------------------------

@dataclass
class LMBuildResult:
    languages_trained: List[str]
    char_lm_order: int
    word_lm_order: int
    per_lang_stats: Dict[str, Dict]
    combined_char_perplexity: float
    combined_word_perplexity: float
    runtime_seconds: float


def run_lm_build() -> None:
    """Step 49A.1: build external n-gram language models."""
    t0 = time.time()
    print("=" * 70)
    print("STEP 49A.1: Build External N-Gram LMs")
    print("=" * 70)

    rd = _results_dir()
    languages = ['latin', 'italian', 'occitan', 'german']

    ref_corpus = load_reference_corpus(languages=languages, verbose=False)

    per_lang_stats: Dict[str, Dict] = {}
    all_combined_tokens: List[str] = []

    for lang in languages:
        texts = ref_corpus.texts.get(lang, [])
        lang_tokens: List[str] = []
        for rt in texts:
            lang_tokens.extend(rt.tokens)

        n_tokens = len(lang_tokens)
        if n_tokens == 0:
            per_lang_stats[lang] = {
                'n_tokens': 0,
                'char_vocab_size': 0,
                'char_ngrams': 0,
                'word_vocab_size': 0,
                'word_ngrams': 0,
            }
            print(f"\n  {lang}: 0 tokens (skipped)")
            continue

        # Build char LM for stats
        char_lm = build_ngram_lm(lang_tokens, order=5, smoothing=0.001)
        char_vocab_size = char_lm['vocab_size']
        char_ngrams = sum(len(v) for v in char_lm['counts'].values())

        # Build word sequences from individual texts
        word_sequences = [rt.tokens for rt in texts if len(rt.tokens) >= 2]
        word_lm = build_word_ngram_lm(word_sequences, order=3, smoothing=1.0)
        word_vocab_size = word_lm['vocab_size']
        word_ngrams = sum(len(v) for v in word_lm['counts'].values())

        per_lang_stats[lang] = {
            'n_tokens': n_tokens,
            'char_vocab_size': char_vocab_size,
            'char_ngrams': char_ngrams,
            'word_vocab_size': word_vocab_size,
            'word_ngrams': word_ngrams,
        }

        print(f"\n  {lang}: {n_tokens} tokens, char_vocab={char_vocab_size}, "
              f"word_vocab={word_vocab_size}")

        # Accumulate combined tokens for Latin + Italian
        if lang in ('latin', 'italian'):
            all_combined_tokens.extend(lang_tokens)

    # Build combined char 5-gram LM
    print("\n  Building combined char 5-gram LM (Latin + Italian)...")
    if not all_combined_tokens:
        print("  [WARN] No tokens for combined LM")
        combined_char_lm = build_ngram_lm(['placeholder'], order=5, smoothing=0.001)
        combined_word_lm = build_word_ngram_lm([['placeholder']], order=3, smoothing=1.0)
    else:
        combined_char_lm = build_ngram_lm(all_combined_tokens, order=5, smoothing=0.001)

        # Build combined word 3-gram LM
        print("  Building combined word 3-gram LM...")
        combined_word_sequences: List[List[str]] = []
        for lang in ('latin', 'italian'):
            for rt in ref_corpus.texts.get(lang, []):
                if len(rt.tokens) >= 2:
                    combined_word_sequences.append(rt.tokens)
        combined_word_lm = build_word_ngram_lm(
            combined_word_sequences, order=3, smoothing=1.0,
        )

    # Self-test: hold out last 10% of tokens
    n_total = len(all_combined_tokens)
    split_idx = int(n_total * 0.9)
    if split_idx > 0 and split_idx < n_total:
        train_tokens = all_combined_tokens[:split_idx]
        test_tokens = all_combined_tokens[split_idx:]

        train_char_lm = build_ngram_lm(train_tokens, order=5, smoothing=0.001)
        test_text = '_' + '_'.join(test_tokens) + '_'
        char_ce = cross_entropy_lm(test_text, train_char_lm, per_char=True)
        combined_char_perplexity = 2 ** char_ce

        # Word-level test
        train_word_sequences: List[List[str]] = []
        # Group train tokens into chunks of 50 as pseudo-sequences
        chunk_size = 50
        for start in range(0, len(train_tokens), chunk_size):
            chunk = train_tokens[start:start + chunk_size]
            if len(chunk) >= 2:
                train_word_sequences.append(chunk)
        train_word_lm = build_word_ngram_lm(train_word_sequences, order=3, smoothing=1.0)
        word_ce = cross_entropy_word_lm(test_tokens, train_word_lm, per_word=True)
        combined_word_perplexity = 2 ** word_ce
    else:
        combined_char_perplexity = 0.0
        combined_word_perplexity = 0.0

    print(f"\n  Combined char perplexity (held-out): {combined_char_perplexity:.2f}")
    print(f"  Combined word perplexity (held-out): {combined_word_perplexity:.2f}")

    # Pickle LMs to disk
    char_lm_path = os.path.join(rd, 'lm_char5.pkl')
    with open(char_lm_path, 'wb') as f:
        pickle.dump(combined_char_lm, f)
    print(f"  Saved char LM -> {char_lm_path}")

    word_lm_path = os.path.join(rd, 'lm_word3.pkl')
    with open(word_lm_path, 'wb') as f:
        pickle.dump(combined_word_lm, f)
    print(f"  Saved word LM -> {word_lm_path}")

    result = LMBuildResult(
        languages_trained=languages,
        char_lm_order=5,
        word_lm_order=3,
        per_lang_stats=per_lang_stats,
        combined_char_perplexity=round(combined_char_perplexity, 4),
        combined_word_perplexity=round(combined_word_perplexity, 4),
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = _save_json(rd, 'lm_build.json', asdict(result))
    print(f"\n  Saved -> {out_path}")


# ---------------------------------------------------------------------------
# Step 49A.2 — Build Token Decode Lattice with ED1
# ---------------------------------------------------------------------------

@dataclass
class LMLatticeResult:
    n_tokens: int
    n_with_alternatives: int
    n_with_ed1_expansion: int
    mean_lattice_size: float
    max_lattice_size: int
    ed1_vocab_size: int
    runtime_seconds: float


def _generate_ed1(word: str, vocab_chars: str = 'abcdefghijklmnopqrstuvwxyz') -> Set[str]:
    """Generate all edit-distance-1 variants of a word."""
    results = set()
    for i in range(len(word)):
        # Deletion
        results.add(word[:i] + word[i + 1:])
        # Substitution
        for c in vocab_chars:
            if c != word[i]:
                results.add(word[:i] + c + word[i + 1:])
    # Insertion
    for i in range(len(word) + 1):
        for c in vocab_chars:
            results.add(word[:i] + c + word[i:])
    return results


def run_lm_lattice() -> None:
    """Step 49A.2: build token decode lattice with ED1 expansion."""
    t0 = time.time()
    print("=" * 70)
    print("STEP 49A.2: Build Token Decode Lattice with ED1")
    print("=" * 70)

    rd = _results_dir()

    # Load disamb_lattice (Phase 47B)
    disamb_raw = _safe_load(os.path.join(rd, 'disamb_lattice.json'))
    disamb_lattice = disamb_raw.get('lattice', {})

    # Load signal_bigrams for per-token decoded words and EVA strings
    sb = _safe_load(os.path.join(rd, 'signal_bigrams.json'))
    token_decoded = sb.get('token_decoded', [])
    token_evas = sb.get('token_evas', [])
    n_tokens = len(token_decoded)

    if n_tokens == 0:
        print("  [SKIP] No token data")
        return

    # Load char LM for vocabulary
    char_lm_path = os.path.join(rd, 'lm_char5.pkl')
    with open(char_lm_path, 'rb') as f:
        char_lm = pickle.load(f)
    lm_vocab = set(char_lm.get('vocab', []))

    # Build reference word set from LM training tokens (top 10K most frequent)
    ref_corpus = load_reference_corpus(languages=['latin', 'italian'], verbose=False)
    all_ref_tokens: List[str] = []
    for lang in ('latin', 'italian'):
        all_ref_tokens.extend(ref_corpus.get_combined_tokens(lang))

    word_freq = Counter(w.lower() for w in all_ref_tokens if len(w) >= 2)
    ref_word_set = {w for w, _ in word_freq.most_common(10000)}
    ed1_vocab_size = len(ref_word_set)
    print(f"\n  Reference word set for ED1: {ed1_vocab_size} words")

    # Build expanded lattice
    print("  Building ED1-expanded lattice...")
    expanded_lattice: Dict[str, List[List]] = {}
    n_with_alt = 0
    n_with_ed1 = 0
    lattice_sizes: List[int] = []

    for i in range(n_tokens):
        idx_str = str(i)

        # Get base candidates
        if idx_str in disamb_lattice:
            base_candidates = [(w, p) for w, p in disamb_lattice[idx_str]]
        else:
            base_candidates = [(token_decoded[i], 1.0)]

        # Generate ED1 expansion
        seen_words: Set[str] = {w for w, _ in base_candidates}
        ed1_candidates: List[Tuple[str, float]] = []

        for word, prior in base_candidates:
            if len(word) < 2:
                continue
            ed1_variants = _generate_ed1(word)
            for variant in ed1_variants:
                if variant in ref_word_set and variant not in seen_words:
                    ed1_candidates.append((variant, prior * 0.3))
                    seen_words.add(variant)

        has_ed1 = len(ed1_candidates) > 0
        all_candidates = base_candidates + ed1_candidates

        # Deduplicate (should already be unique via seen_words, but be safe)
        deduped: Dict[str, float] = {}
        for w, p in all_candidates:
            if w not in deduped or p > deduped[w]:
                deduped[w] = p
        final_candidates = sorted(deduped.items(), key=lambda x: -x[1])[:50]

        lattice_size = len(final_candidates)
        lattice_sizes.append(lattice_size)

        if lattice_size > 1:
            n_with_alt += 1
        if has_ed1:
            n_with_ed1 += 1

        # Store all tokens in lattice (even single-candidate ones for uniformity)
        if lattice_size > 1:
            expanded_lattice[idx_str] = [[w, round(p, 6)] for w, p in final_candidates]

    mean_size = sum(lattice_sizes) / n_tokens if n_tokens else 0.0
    max_size = max(lattice_sizes) if lattice_sizes else 0

    print(f"\n  Results:")
    print(f"    Tokens: {n_tokens}")
    print(f"    With alternatives: {n_with_alt} ({n_with_alt / n_tokens:.1%})")
    print(f"    With ED1 expansion: {n_with_ed1} ({n_with_ed1 / n_tokens:.1%})")
    print(f"    Mean lattice size: {mean_size:.2f}")
    print(f"    Max lattice size: {max_size}")

    result = LMLatticeResult(
        n_tokens=n_tokens,
        n_with_alternatives=n_with_alt,
        n_with_ed1_expansion=n_with_ed1,
        mean_lattice_size=round(mean_size, 2),
        max_lattice_size=max_size,
        ed1_vocab_size=ed1_vocab_size,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out = _convert(asdict(result))
    out['lattice'] = expanded_lattice

    out_path = os.path.join(rd, 'lm_lattice.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved -> {out_path}")


# ---------------------------------------------------------------------------
# Step 49A.3 — Beam Search Decode with External LM
# ---------------------------------------------------------------------------

@dataclass
class LMViterbiResult:
    n_tokens: int
    n_changed: int
    change_rate: float
    dict_hit_rate_10k: float
    dict_hit_rate_131k: float
    mean_char_ce: float
    mean_word_ce: float
    n_consecutive_hits: int
    top_folios: List[Dict]
    runtime_seconds: float


def _lm_beam_search(
    folio_indices: List[int],
    lattice: Dict[int, List[Tuple[str, float]]],
    token_decoded: List[str],
    char_lm: Dict,
    word_lm: Dict,
    alpha: float = 0.4,
    beta: float = 0.4,
    gamma: float = 0.2,
    beam_width: int = 10,
) -> List[str]:
    """Beam search over token sequence using external LM scoring.

    Each beam state: (cumulative_score, word_sequence, last_word)
    """
    n = len(folio_indices)
    if n == 0:
        return []

    word_lm_counts = word_lm.get('counts', {})
    word_lm_V = word_lm.get('vocab_size', 1)
    word_lm_k = word_lm.get('smoothing', 1.0)

    # Initialize beam
    beam: List[Tuple[float, List[str], str]] = [(0.0, [], '<BOS>')]

    for pos in range(n):
        idx = folio_indices[pos]
        # Get candidates for this position
        if idx in lattice:
            candidates = lattice[idx]
        else:
            candidates = [(token_decoded[idx], 1.0)]

        new_beam: List[Tuple[float, List[str], str]] = []

        for cum_score, word_seq, last_word in beam:
            for candidate_word, prior in candidates:
                # Char LM score: lower cross-entropy = better
                if len(candidate_word) >= 1:
                    char_text = '_' + candidate_word + '_'
                    char_ce = cross_entropy_lm(char_text, char_lm, per_char=True)
                    char_score = -char_ce  # negate: lower CE -> higher score
                else:
                    char_score = 0.0

                # Word LM score: log P(candidate | last_word)
                context_key = last_word
                ctx_counts = word_lm_counts.get(context_key, None)
                if ctx_counts is not None:
                    total_count = sum(ctx_counts.values())
                    word_count = ctx_counts.get(candidate_word, 0)
                    prob = (word_count + word_lm_k) / (total_count + word_lm_k * word_lm_V)
                else:
                    # Backoff to unigram
                    uni_key = '<BOS>'
                    uni_counts = word_lm_counts.get(uni_key, {})
                    if uni_counts:
                        total_count = sum(uni_counts.values())
                        word_count = uni_counts.get(candidate_word, 0)
                        prob = (word_count + word_lm_k) / (total_count + word_lm_k * word_lm_V)
                    else:
                        prob = 1.0 / word_lm_V
                word_score = math.log(prob + 1e-20)

                # Prior score
                prior_score = math.log(prior + 1e-20)

                # Total
                total = alpha * char_score + beta * word_score + gamma * prior_score
                new_score = cum_score + total

                new_beam.append((
                    new_score,
                    word_seq + [candidate_word],
                    candidate_word,
                ))

        # Prune beam
        new_beam.sort(key=lambda x: -x[0])
        beam = new_beam[:beam_width]

    if not beam:
        return [token_decoded[idx] for idx in folio_indices]

    # Return best sequence
    best_score, best_seq, _ = beam[0]
    return best_seq


def _longest_consecutive_hits(decoded: List[str], word_set: Set[str]) -> int:
    """Return length of longest consecutive run of dict hits."""
    best = 0
    current = 0
    for w in decoded:
        if w in word_set:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def run_lm_viterbi() -> None:
    """Step 49A.3: beam search decode with external LM."""
    t0 = time.time()
    print("=" * 70)
    print("STEP 49A.3: Beam Search Decode with External LM")
    print("=" * 70)

    rd = _results_dir()

    # Load expanded lattice
    lattice_raw = _safe_load(os.path.join(rd, 'lm_lattice.json'))
    lattice_entries = lattice_raw.get('lattice', {})
    lattice: Dict[int, List[Tuple[str, float]]] = {}
    for idx_str, entries in lattice_entries.items():
        lattice[int(idx_str)] = [(w, p) for w, p in entries]

    # Load signal_bigrams for per-token data
    sb = _safe_load(os.path.join(rd, 'signal_bigrams.json'))
    token_decoded = sb.get('token_decoded', [])
    token_folios = sb.get('token_folios', [])
    n_tokens = len(token_decoded)

    if n_tokens == 0:
        print("  [SKIP] No data")
        return

    # Load LMs
    char_lm_path = os.path.join(rd, 'lm_char5.pkl')
    with open(char_lm_path, 'rb') as f:
        char_lm = pickle.load(f)

    word_lm_path = os.path.join(rd, 'lm_word3.pkl')
    with open(word_lm_path, 'rb') as f:
        word_lm = pickle.load(f)

    # Group indices by folio
    folio_groups: Dict[str, List[int]] = defaultdict(list)
    for i in range(n_tokens):
        folio_groups[token_folios[i]].append(i)

    print(f"\n  {len(folio_groups)} folios, {n_tokens} tokens, "
          f"{len(lattice)} with alternatives")

    # Run beam search per folio
    print("  Running beam search...")
    lm_decoded: List[str] = list(token_decoded)  # copy

    for folio, indices in folio_groups.items():
        result_words = _lm_beam_search(
            indices, lattice, token_decoded,
            char_lm, word_lm,
            alpha=0.4, beta=0.4, gamma=0.2,
            beam_width=10,
        )
        for idx, word in zip(indices, result_words):
            lm_decoded[idx] = word

    n_changed = sum(
        1 for i in range(n_tokens)
        if lm_decoded[i] != token_decoded[i]
    )
    change_rate = n_changed / n_tokens if n_tokens else 0.0

    # Evaluate dict-hit
    _ref_for_base = load_reference_corpus(languages=['latin'], verbose=False)
    _base_words = set(w.lower() for w in _ref_for_base.get_combined_tokens('latin'))
    expanded_word_set, _ = build_expanded_word_set(_base_words)

    ref_corpus = load_reference_corpus(languages=['latin', 'italian'], verbose=False)
    all_ref_tokens: List[str] = []
    for lang in ('latin', 'italian'):
        all_ref_tokens.extend(ref_corpus.get_combined_tokens(lang))
    word_freq_10k = Counter(w.lower() for w in all_ref_tokens if len(w) >= 2)
    ref_word_set_10k = {w for w, _ in word_freq_10k.most_common(10000)}

    hits_10k = sum(1 for w in lm_decoded if w in ref_word_set_10k)
    hits_131k = sum(1 for w in lm_decoded if w in expanded_word_set)
    rate_10k = hits_10k / n_tokens if n_tokens else 0.0
    rate_131k = hits_131k / n_tokens if n_tokens else 0.0

    # Mean char CE and word CE of decoded tokens
    decoded_text = '_' + '_'.join(lm_decoded) + '_'
    mean_char_ce = cross_entropy_lm(decoded_text, char_lm, per_char=True)
    mean_word_ce = cross_entropy_word_lm(lm_decoded, word_lm, per_word=True)

    # Consecutive hits
    n_consec = _longest_consecutive_hits(lm_decoded, ref_word_set_10k)

    # Per-folio dict-hit (top 10)
    folio_rates: List[Dict] = []
    for folio, indices in folio_groups.items():
        folio_words = [lm_decoded[i] for i in indices]
        folio_hits = sum(1 for w in folio_words if w in ref_word_set_10k)
        folio_rate = folio_hits / len(folio_words) if folio_words else 0.0
        folio_rates.append({
            'folio': folio,
            'n_tokens': len(folio_words),
            'dict_hit_10k': round(folio_rate, 4),
        })
    folio_rates.sort(key=lambda x: -x['dict_hit_10k'])
    top_folios = folio_rates[:10]

    print(f"\n  Results:")
    print(f"    Changed: {n_changed} ({change_rate:.1%})")
    print(f"    Dict-hit (10K): {rate_10k:.1%}")
    print(f"    Dict-hit (131K): {rate_131k:.1%}")
    print(f"    Mean char CE: {mean_char_ce:.4f}")
    print(f"    Mean word CE: {mean_word_ce:.4f}")
    print(f"    Longest consecutive hits: {n_consec}")
    print(f"    Top folio: {top_folios[0]['folio']} ({top_folios[0]['dict_hit_10k']:.1%})"
          if top_folios else "    No folios")

    result = LMViterbiResult(
        n_tokens=n_tokens,
        n_changed=n_changed,
        change_rate=round(change_rate, 4),
        dict_hit_rate_10k=round(rate_10k, 4),
        dict_hit_rate_131k=round(rate_131k, 4),
        mean_char_ce=round(mean_char_ce, 4),
        mean_word_ce=round(mean_word_ce, 4),
        n_consecutive_hits=n_consec,
        top_folios=top_folios,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out = _convert(asdict(result))
    # Store the decoded tokens for downstream use
    out['lm_decoded'] = lm_decoded

    out_path = os.path.join(rd, 'lm_viterbi.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved -> {out_path}")


# ---------------------------------------------------------------------------
# Step 49A.4 — Weight Calibration + Ablation
# ---------------------------------------------------------------------------

@dataclass
class LMCalibrateResult:
    n_configs_tested: int
    best_config: Dict
    best_dict_hit_10k: float
    best_cc_bigrams: int
    ablation_table: List[Dict]
    char_lm_only_dict_hit: float
    word_lm_only_dict_hit: float
    no_ed1_dict_hit: float
    runtime_seconds: float


def _count_cc_bigrams(
    decoded: List[str],
    ref_bigram_set: Set[Tuple[str, str]],
) -> int:
    """Count consecutive token pairs where both are dict hits in ref bigrams."""
    count = 0
    for i in range(len(decoded) - 1):
        if (decoded[i], decoded[i + 1]) in ref_bigram_set:
            count += 1
    return count


def run_lm_calibrate() -> None:
    """Step 49A.4: weight calibration and ablation."""
    t0 = time.time()
    print("=" * 70)
    print("STEP 49A.4: Weight Calibration + Ablation")
    print("=" * 70)

    rd = _results_dir()

    # Load lattice
    lattice_raw = _safe_load(os.path.join(rd, 'lm_lattice.json'))
    lattice_entries = lattice_raw.get('lattice', {})
    lattice: Dict[int, List[Tuple[str, float]]] = {}
    for idx_str, entries in lattice_entries.items():
        lattice[int(idx_str)] = [(w, p) for w, p in entries]

    # Load signal_bigrams for per-token data
    sb = _safe_load(os.path.join(rd, 'signal_bigrams.json'))
    token_decoded = sb.get('token_decoded', [])
    token_folios = sb.get('token_folios', [])
    n_tokens = len(token_decoded)

    if n_tokens == 0:
        print("  [SKIP] No data")
        return

    # Load LMs
    with open(os.path.join(rd, 'lm_char5.pkl'), 'rb') as f:
        char_lm = pickle.load(f)
    with open(os.path.join(rd, 'lm_word3.pkl'), 'rb') as f:
        word_lm = pickle.load(f)

    # Build 10K word set
    ref_corpus = load_reference_corpus(languages=['latin', 'italian'], verbose=False)
    all_ref_tokens: List[str] = []
    for lang in ('latin', 'italian'):
        all_ref_tokens.extend(ref_corpus.get_combined_tokens(lang))
    word_freq = Counter(w.lower() for w in all_ref_tokens if len(w) >= 2)
    ref_word_set_10k = {w for w, _ in word_freq.most_common(10000)}

    # Build reference bigram set from Latin reference corpus
    latin_tokens = [w.lower() for w in ref_corpus.get_combined_tokens('latin') if len(w) >= 2]
    ref_bigram_set: Set[Tuple[str, str]] = set()
    for i in range(len(latin_tokens) - 1):
        ref_bigram_set.add((latin_tokens[i], latin_tokens[i + 1]))

    # Use first 5000 tokens as subsample
    subsample_size = min(5000, n_tokens)
    subsample_indices = list(range(subsample_size))

    # Group subsample indices by folio
    sub_folio_groups: Dict[str, List[int]] = defaultdict(list)
    for i in subsample_indices:
        sub_folio_groups[token_folios[i]].append(i)

    # Generate weight configurations
    alpha_vals = [0.2, 0.4, 0.6]
    beta_vals = [0.2, 0.4, 0.6]
    gamma_vals = [0.1, 0.2, 0.3]

    configs: List[Tuple[float, float, float]] = []
    seen_configs: Set[Tuple[float, float, float]] = set()
    for a in alpha_vals:
        for b in beta_vals:
            for g in gamma_vals:
                total = a + b + g
                norm_a = round(a / total, 4)
                norm_b = round(b / total, 4)
                norm_g = round(1.0 - norm_a - norm_b, 4)
                key = (norm_a, norm_b, norm_g)
                if key not in seen_configs:
                    seen_configs.add(key)
                    configs.append(key)

    print(f"\n  Testing {len(configs)} weight configurations on {subsample_size} tokens...")

    ablation_table: List[Dict] = []
    best_config: Optional[Tuple[float, float, float]] = None
    best_dict_hit = 0.0
    best_cc = 0
    best_decoded: List[str] = []

    for cfg_idx, (a, b, g) in enumerate(configs):
        # Run beam search on subsample
        sub_decoded = list(token_decoded[:subsample_size])
        for folio, indices in sub_folio_groups.items():
            result_words = _lm_beam_search(
                indices, lattice, token_decoded,
                char_lm, word_lm,
                alpha=a, beta=b, gamma=g,
                beam_width=10,
            )
            for idx, word in zip(indices, result_words):
                sub_decoded[idx] = word

        hits = sum(1 for w in sub_decoded if w in ref_word_set_10k)
        rate = hits / subsample_size if subsample_size else 0.0
        cc = _count_cc_bigrams(sub_decoded, ref_bigram_set)

        ablation_table.append({
            'alpha': a,
            'beta': b,
            'gamma': g,
            'dict_hit_10k': round(rate, 4),
            'cc_bigrams': cc,
        })

        if rate > best_dict_hit:
            best_dict_hit = rate
            best_config = (a, b, g)
            best_cc = cc
            best_decoded = sub_decoded

        if (cfg_idx + 1) % 5 == 0 or cfg_idx == len(configs) - 1:
            print(f"    Config {cfg_idx + 1}/{len(configs)}: "
                  f"a={a:.2f} b={b:.2f} g={g:.2f} -> {rate:.1%}")

    # Ablation: char LM only
    print("\n  Ablation: char LM only...")
    char_only_decoded = list(token_decoded[:subsample_size])
    for folio, indices in sub_folio_groups.items():
        result_words = _lm_beam_search(
            indices, lattice, token_decoded,
            char_lm, word_lm,
            alpha=1.0, beta=0.0, gamma=0.0,
            beam_width=10,
        )
        for idx, word in zip(indices, result_words):
            char_only_decoded[idx] = word
    char_lm_only_hit = sum(1 for w in char_only_decoded if w in ref_word_set_10k)
    char_lm_only_rate = char_lm_only_hit / subsample_size if subsample_size else 0.0

    # Ablation: word LM only
    print("  Ablation: word LM only...")
    word_only_decoded = list(token_decoded[:subsample_size])
    for folio, indices in sub_folio_groups.items():
        result_words = _lm_beam_search(
            indices, lattice, token_decoded,
            char_lm, word_lm,
            alpha=0.0, beta=1.0, gamma=0.0,
            beam_width=10,
        )
        for idx, word in zip(indices, result_words):
            word_only_decoded[idx] = word
    word_lm_only_hit = sum(1 for w in word_only_decoded if w in ref_word_set_10k)
    word_lm_only_rate = word_lm_only_hit / subsample_size if subsample_size else 0.0

    # Ablation: no ED1 (filter lattice to only original entries)
    print("  Ablation: no ED1 expansion...")
    disamb_raw = _safe_load(os.path.join(rd, 'disamb_lattice.json'))
    disamb_lattice_entries = disamb_raw.get('lattice', {})
    no_ed1_lattice: Dict[int, List[Tuple[str, float]]] = {}
    for idx_str, entries in disamb_lattice_entries.items():
        no_ed1_lattice[int(idx_str)] = [(w, p) for w, p in entries]

    no_ed1_decoded = list(token_decoded[:subsample_size])
    ba, bb, bg = best_config if best_config else (0.4, 0.4, 0.2)
    for folio, indices in sub_folio_groups.items():
        result_words = _lm_beam_search(
            indices, no_ed1_lattice, token_decoded,
            char_lm, word_lm,
            alpha=ba, beta=bb, gamma=bg,
            beam_width=10,
        )
        for idx, word in zip(indices, result_words):
            no_ed1_decoded[idx] = word
    no_ed1_hit = sum(1 for w in no_ed1_decoded if w in ref_word_set_10k)
    no_ed1_rate = no_ed1_hit / subsample_size if subsample_size else 0.0

    # Sort ablation table
    ablation_table.sort(key=lambda x: -x['dict_hit_10k'])

    best_cfg_dict = {
        'alpha': ba,
        'beta': bb,
        'gamma': bg,
    }

    print(f"\n  Results:")
    print(f"    Configs tested: {len(configs)}")
    print(f"    Best config: alpha={ba:.2f}, beta={bb:.2f}, gamma={bg:.2f}")
    print(f"    Best dict-hit (10K): {best_dict_hit:.1%}")
    print(f"    Best CC bigrams: {best_cc}")
    print(f"    Char LM only: {char_lm_only_rate:.1%}")
    print(f"    Word LM only: {word_lm_only_rate:.1%}")
    print(f"    No ED1: {no_ed1_rate:.1%}")

    result = LMCalibrateResult(
        n_configs_tested=len(configs),
        best_config=best_cfg_dict,
        best_dict_hit_10k=round(best_dict_hit, 4),
        best_cc_bigrams=best_cc,
        ablation_table=ablation_table[:20],  # top 20 only
        char_lm_only_dict_hit=round(char_lm_only_rate, 4),
        word_lm_only_dict_hit=round(word_lm_only_rate, 4),
        no_ed1_dict_hit=round(no_ed1_rate, 4),
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = _save_json(rd, 'lm_calibrate.json', asdict(result))
    print(f"\n  Saved -> {out_path}")


# ---------------------------------------------------------------------------
# Step 49A.5 — Full Corpus LM Decode
# ---------------------------------------------------------------------------

@dataclass
class LMDecodeResult:
    dict_hit_rate_10k: float
    dict_hit_rate_131k: float
    dict_hit_count: int
    n_tokens: int
    selectivity_ratio: float
    cc_bigrams: int
    n_consecutive_hits: int
    best_folio: str
    best_folio_dict_hit: float
    sample_decodings: List[Dict]
    per_section_dict_hit: Dict[str, float]
    delta_vs_phase16: float
    config_used: Dict
    runtime_seconds: float


def run_lm_decode() -> None:
    """Step 49A.5: full corpus LM decode with best config."""
    t0 = time.time()
    print("=" * 70)
    print("STEP 49A.5: Full Corpus LM Decode")
    print("=" * 70)

    rd = _results_dir()

    # Load best config from calibration
    cal = _safe_load(os.path.join(rd, 'lm_calibrate.json'))
    config = cal.get('best_config', {'alpha': 0.4, 'beta': 0.4, 'gamma': 0.2})
    alpha = config.get('alpha', 0.4)
    beta = config.get('beta', 0.4)
    gamma = config.get('gamma', 0.2)

    print(f"\n  Config: alpha={alpha:.2f}, beta={beta:.2f}, gamma={gamma:.2f}")

    # Load lattice
    lattice_raw = _safe_load(os.path.join(rd, 'lm_lattice.json'))
    lattice_entries = lattice_raw.get('lattice', {})
    lattice: Dict[int, List[Tuple[str, float]]] = {}
    for idx_str, entries in lattice_entries.items():
        lattice[int(idx_str)] = [(w, p) for w, p in entries]

    # Load signal_bigrams
    sb = _safe_load(os.path.join(rd, 'signal_bigrams.json'))
    token_decoded = sb.get('token_decoded', [])
    token_folios = sb.get('token_folios', [])
    n_tokens = len(token_decoded)

    if n_tokens == 0:
        print("  [SKIP] No data")
        return

    # Load LMs
    with open(os.path.join(rd, 'lm_char5.pkl'), 'rb') as f:
        char_lm = pickle.load(f)
    with open(os.path.join(rd, 'lm_word3.pkl'), 'rb') as f:
        word_lm = pickle.load(f)

    # Group indices by folio
    folio_groups: Dict[str, List[int]] = defaultdict(list)
    for i in range(n_tokens):
        folio_groups[token_folios[i]].append(i)

    print(f"\n  {len(folio_groups)} folios, {n_tokens} tokens")
    print("  Running beam search with best config...")

    # Run beam search per folio
    lm_decoded: List[str] = list(token_decoded)

    for folio_idx, (folio, indices) in enumerate(folio_groups.items()):
        result_words = _lm_beam_search(
            indices, lattice, token_decoded,
            char_lm, word_lm,
            alpha=alpha, beta=beta, gamma=gamma,
            beam_width=10,
        )
        for idx, word in zip(indices, result_words):
            lm_decoded[idx] = word
        if (folio_idx + 1) % 50 == 0:
            print(f"    Processed {folio_idx + 1}/{len(folio_groups)} folios...")

    n_changed = sum(
        1 for i in range(n_tokens)
        if lm_decoded[i] != token_decoded[i]
    )

    # Build word sets for evaluation
    _ref_for_base = load_reference_corpus(languages=['latin'], verbose=False)
    _base_words = set(w.lower() for w in _ref_for_base.get_combined_tokens('latin'))
    expanded_word_set, _ = build_expanded_word_set(_base_words)

    ref_corpus = load_reference_corpus(languages=['latin', 'italian'], verbose=False)
    all_ref_tokens: List[str] = []
    for lang in ('latin', 'italian'):
        all_ref_tokens.extend(ref_corpus.get_combined_tokens(lang))
    word_freq = Counter(w.lower() for w in all_ref_tokens if len(w) >= 2)
    ref_word_set_10k = {w for w, _ in word_freq.most_common(10000)}

    # Dict-hit rates
    hits_10k = sum(1 for w in lm_decoded if w in ref_word_set_10k)
    hits_131k = sum(1 for w in lm_decoded if w in expanded_word_set)
    rate_10k = hits_10k / n_tokens if n_tokens else 0.0
    rate_131k = hits_131k / n_tokens if n_tokens else 0.0

    # Selectivity ratio: compare against random baseline
    # Random baseline: pick random word from lattice (uniform), measure dict-hit
    rng = np.random.RandomState(42)
    n_random_trials = 5
    random_hits_total = 0
    for trial in range(n_random_trials):
        random_hits = 0
        for i in range(n_tokens):
            if i in lattice:
                cands = lattice[i]
                pick = cands[rng.randint(0, len(cands))][0]
            else:
                pick = token_decoded[i]
            if pick in ref_word_set_10k:
                random_hits += 1
        random_hits_total += random_hits
    random_rate = random_hits_total / (n_tokens * n_random_trials) if n_tokens else 0.0
    selectivity = rate_10k / random_rate if random_rate > 0 else 0.0

    # CC bigrams
    latin_tokens_lower = [w.lower() for w in ref_corpus.get_combined_tokens('latin') if len(w) >= 2]
    ref_bigram_set: Set[Tuple[str, str]] = set()
    for i in range(len(latin_tokens_lower) - 1):
        ref_bigram_set.add((latin_tokens_lower[i], latin_tokens_lower[i + 1]))
    cc_bigrams = _count_cc_bigrams(lm_decoded, ref_bigram_set)

    # Consecutive hits
    n_consec = _longest_consecutive_hits(lm_decoded, ref_word_set_10k)

    # Per-folio dict-hit
    folio_rates: Dict[str, float] = {}
    best_folio = ''
    best_folio_rate = 0.0
    for folio, indices in folio_groups.items():
        folio_words = [lm_decoded[i] for i in indices]
        fhits = sum(1 for w in folio_words if w in ref_word_set_10k)
        frate = fhits / len(folio_words) if folio_words else 0.0
        folio_rates[folio] = round(frate, 4)
        if frate > best_folio_rate:
            best_folio_rate = frate
            best_folio = folio

    # Per-section dict-hit
    corpus = load_corpus(verbose=False)
    folio_to_section: Dict[str, str] = {}
    for folio_id, page in corpus.pages.items():
        folio_to_section[folio_id] = page.section

    section_hits: Dict[str, int] = defaultdict(int)
    section_total: Dict[str, int] = defaultdict(int)
    for i in range(n_tokens):
        sec = folio_to_section.get(token_folios[i], 'unknown')
        section_total[sec] += 1
        if lm_decoded[i] in ref_word_set_10k:
            section_hits[sec] += 1

    per_section_dict_hit: Dict[str, float] = {}
    for sec in section_total:
        per_section_dict_hit[sec] = round(
            section_hits[sec] / section_total[sec], 4,
        ) if section_total[sec] > 0 else 0.0

    # Phase 16 baseline dict-hit
    orig_hits_10k = sum(1 for w in token_decoded if w in ref_word_set_10k)
    orig_rate_10k = orig_hits_10k / n_tokens if n_tokens else 0.0
    delta = rate_10k - orig_rate_10k

    # Sample 50 changed tokens
    sample_decodings: List[Dict] = []
    for i in range(n_tokens):
        if lm_decoded[i] != token_decoded[i]:
            sample_decodings.append({
                'index': i,
                'folio': token_folios[i],
                'original': token_decoded[i],
                'lm_decoded': lm_decoded[i],
                'in_10k': lm_decoded[i] in ref_word_set_10k,
                'orig_in_10k': token_decoded[i] in ref_word_set_10k,
            })
            if len(sample_decodings) >= 50:
                break

    print(f"\n  Results:")
    print(f"    Changed: {n_changed} ({n_changed / n_tokens:.1%})")
    print(f"    Dict-hit (10K): {rate_10k:.1%} (delta={delta:+.1%} vs Phase 16)")
    print(f"    Dict-hit (131K): {rate_131k:.1%}")
    print(f"    Selectivity ratio: {selectivity:.2f}x")
    print(f"    CC bigrams: {cc_bigrams}")
    print(f"    Longest consecutive hits: {n_consec}")
    print(f"    Best folio: {best_folio} ({best_folio_rate:.1%})")
    print(f"\n  Per-section dict-hit:")
    for sec, rate in sorted(per_section_dict_hit.items()):
        print(f"    {sec}: {rate:.1%}")

    result = LMDecodeResult(
        dict_hit_rate_10k=round(rate_10k, 4),
        dict_hit_rate_131k=round(rate_131k, 4),
        dict_hit_count=hits_10k,
        n_tokens=n_tokens,
        selectivity_ratio=round(selectivity, 4),
        cc_bigrams=cc_bigrams,
        n_consecutive_hits=n_consec,
        best_folio=best_folio,
        best_folio_dict_hit=round(best_folio_rate, 4),
        sample_decodings=sample_decodings,
        per_section_dict_hit=per_section_dict_hit,
        delta_vs_phase16=round(delta, 4),
        config_used=config,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out = _convert(asdict(result))
    out['lm_decoded'] = lm_decoded

    out_path = os.path.join(rd, 'lm_decode.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved -> {out_path}")


# ---------------------------------------------------------------------------
# Track A orchestrator
# ---------------------------------------------------------------------------

def run_track_a_49() -> None:
    """Run all Track A steps sequentially."""
    run_lm_build()
    print()
    run_lm_lattice()
    print()
    run_lm_viterbi()
    print()
    run_lm_calibrate()
    print()
    run_lm_decode()
