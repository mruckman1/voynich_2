"""
Step 37.4 – Confirmed Pair Concatenation
==========================================
Test whether adjacent confirmed signal words, when concatenated, form Latin
words at rates above chance.  This tests whether EVA "words" are actually
syllables.

Dependency chain:
    context_10k.json           (Step 36.4)
    signal_10k.json            (Step 36.2)
    combined_refine.json       (Phase 15)
        → pair_concat.json     (this step)
"""

import json
import os
import random
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.reference import load_reference_corpus, build_expanded_word_set


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _convert(obj: Any) -> Any:
    if hasattr(obj, '__dataclass_fields__'):
        from dataclasses import asdict
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


def _safe_load(path: str) -> Dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


# Latin content word categories
_FUNCTION_WORDS = {
    'de', 'in', 'ad', 'et', 'cum', 'per', 'pro', 'ex', 'ab', 'sub',
    'non', 'si', 'ut', 'sed', 'ne', 'ac', 'at', 'an', 'se', 'te',
    'me', 'nos', 'iam', 'est', 'sunt', 'hoc', 'id', 'ea', 'is',
    'di', 'du', 'ce', 'ci', 'co', 'cu', 'bi', 'bo', 'be', 'da',
    'la', 'le', 'li', 'lo', 'ni', 'no', 'nu', 'ra', 're', 'ri',
    'ro', 'sa', 'so', 'su', 'ta', 'ti', 'to',
}


def _is_content_word(word: str) -> bool:
    """Classify a word as content (noun/verb/adj) vs function word."""
    return word.lower() not in _FUNCTION_WORDS and len(word) >= 4


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_pair_concat() -> None:
    """Step 37.4: Confirmed Pair Concatenation."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 37.4: Confirmed Pair Concatenation")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load inputs ──
    print("\n  1. Loading inputs …")
    context_data = _safe_load(os.path.join(rd, 'context_10k.json'))
    signal_data = _safe_load(os.path.join(rd, 'signal_10k.json'))

    confirmed_pairs = context_data.get('confirmed_confirmed_pairs', [])
    word_signals = signal_data.get('word_signals', [])
    genuine_words = set(w['word'] for w in word_signals if w.get('is_genuine_signal'))
    token_decoded = signal_data.get('token_decoded', [])
    token_classifications = signal_data.get('token_classifications', [])
    token_folios = signal_data.get('token_folios', [])

    print(f"     {len(confirmed_pairs)} confirmed-confirmed pairs")
    print(f"     {len(genuine_words)} genuine signal words")

    # ── 2. Build 17K dictionary ──
    print("  2. Building 17K dictionary …")
    ref = load_reference_corpus(languages=['latin'], verbose=False)
    base_words = set(w.lower() for w in ref.get_combined_tokens('latin') if len(w) >= 2)
    expanded, _ = build_expanded_word_set(base_words)
    dict_17k = base_words | expanded
    # Also add a stricter subset for longer matches
    dict_4plus = {w for w in dict_17k if len(w) >= 4}
    print(f"     {len(dict_17k)} words in 17K dictionary")
    print(f"     {len(dict_4plus)} words with length ≥ 4")

    # ── 3. Concatenate pairs and match ──
    print("  3. Concatenating confirmed pairs …")
    pair_matches = []
    pair_content_matches = []
    concat_counts: Dict[str, int] = Counter()

    for pair in confirmed_pairs:
        w1 = pair['word1']
        w2 = pair['word2']
        concat = w1 + w2
        concat_counts[concat] += 1

        if concat in dict_17k:
            is_content = _is_content_word(concat)
            match_info = {
                'word1': w1,
                'word2': w2,
                'concatenated': concat,
                'folio': pair.get('folio', ''),
                'position': pair.get('position', -1),
                'is_content_word': is_content,
            }
            pair_matches.append(match_info)
            if is_content:
                pair_content_matches.append(match_info)

    n_pairs = len(confirmed_pairs)
    n_matches = len(pair_matches)
    n_content = len(pair_content_matches)
    match_rate = n_matches / n_pairs if n_pairs > 0 else 0.0

    print(f"     {n_matches}/{n_pairs} pairs match dictionary ({match_rate:.3%})")
    print(f"     {n_content} are content words")

    # Show unique concatenated words
    unique_concats = Counter(m['concatenated'] for m in pair_matches)
    print("     Top concatenated words:")
    for word, count in unique_concats.most_common(15):
        content_flag = " [CONTENT]" if _is_content_word(word) else ""
        print(f"       {word:<15s} ×{count}{content_flag}")

    # ── 4. Null comparison: random signal word pairs ──
    print("  4. Null comparison (random signal word pairs) …")
    signal_word_list = sorted(genuine_words)
    rng = random.Random(42)
    n_null_trials = 10000
    null_match_counts = []

    for trial in range(5):
        null_matches = 0
        for _ in range(n_pairs):
            w1 = rng.choice(signal_word_list)
            w2 = rng.choice(signal_word_list)
            if (w1 + w2) in dict_17k:
                null_matches += 1
        null_match_counts.append(null_matches)

    null_mean = sum(null_match_counts) / len(null_match_counts)
    null_var = (sum((c - null_mean) ** 2 for c in null_match_counts) /
                len(null_match_counts))
    null_std = null_var ** 0.5
    z_score = ((n_matches - null_mean) / null_std if null_std > 0
               else (10.0 if n_matches > null_mean else 0.0))
    significant = z_score > 1.96

    print(f"     Real matches:  {n_matches}")
    print(f"     Null mean:     {null_mean:.1f} (std={null_std:.1f})")
    print(f"     z-score:       {z_score:.2f}")
    print(f"     Significant:   {'YES (p<0.05)' if significant else 'NO'}")

    # ── 5. Triple concatenation (3 consecutive signal words) ──
    print("  5. Triple concatenation (3 consecutive signal tokens) …")
    triple_matches = []
    n_signal_tokens = len(token_decoded)

    for i in range(n_signal_tokens - 2):
        if (token_classifications[i] == 'SIGNAL' and
                token_classifications[i + 1] == 'SIGNAL' and
                token_classifications[i + 2] == 'SIGNAL'):
            # Check same folio
            if (token_folios[i] == token_folios[i + 1] ==
                    token_folios[i + 2]):
                w1 = token_decoded[i]
                w2 = token_decoded[i + 1]
                w3 = token_decoded[i + 2]
                concat3 = w1 + w2 + w3
                if concat3 in dict_17k and len(concat3) >= 5:
                    triple_matches.append({
                        'words': [w1, w2, w3],
                        'concatenated': concat3,
                        'folio': token_folios[i],
                        'position': i,
                        'is_content_word': _is_content_word(concat3),
                    })

    print(f"     {len(triple_matches)} triple concatenations match dictionary")
    if triple_matches:
        unique_triple_concats = Counter(m['concatenated'] for m in triple_matches)
        print("     Top triple concatenations:")
        for word, count in unique_triple_concats.most_common(10):
            print(f"       {word:<20s} ×{count}")

    # ── 6. Reverse lookup: domain analysis ──
    print("  6. Domain analysis of concatenated matches …")
    # Classify matched words by domain
    medical_terms = {'aqua', 'herba', 'folia', 'radix', 'semen', 'cortex',
                     'flore', 'succo', 'coque', 'cola', 'recipe', 'misce',
                     'adde', 'bene', 'sero', 'codi', 'sene', 'bora',
                     'rosa', 'sale', 'vino', 'oleo', 'mele'}
    botanical_terms = {'viola', 'rosa', 'salvia', 'ruta', 'mentha',
                       'basilico', 'lauro', 'mirra', 'cera', 'piper',
                       'croco', 'aloe'}

    domain_analysis = {'medical': 0, 'botanical': 0, 'other_content': 0, 'function': 0}
    for m in pair_matches:
        word = m['concatenated'].lower()
        if word in medical_terms:
            domain_analysis['medical'] += 1
        elif word in botanical_terms:
            domain_analysis['botanical'] += 1
        elif _is_content_word(word):
            domain_analysis['other_content'] += 1
        else:
            domain_analysis['function'] += 1

    print(f"     Medical:   {domain_analysis['medical']}")
    print(f"     Botanical: {domain_analysis['botanical']}")
    print(f"     Other:     {domain_analysis['other_content']}")
    print(f"     Function:  {domain_analysis['function']}")

    # ── 7. Folio distribution of matches ──
    print("  7. Folio distribution of concatenation matches …")
    folio_match_counts: Dict[str, int] = Counter()
    for m in pair_matches:
        folio_match_counts[m['folio']] += 1
    top_folios = folio_match_counts.most_common(10)
    for fol, cnt in top_folios:
        print(f"       {fol:<8s} {cnt} matches")

    # ── 8. Frequency rank correlation ──
    print("  8. Frequency rank analysis …")
    # Rank concatenated words by corpus frequency
    concat_freq = Counter(m['concatenated'] for m in pair_matches)
    # Rank same words in reference corpus
    ref_tokens = ref.get_combined_tokens('latin')
    ref_word_freq = Counter(w.lower() for w in ref_tokens)

    matched_words = list(concat_freq.keys())
    corpus_ranks = []
    ref_ranks = []
    for i, word in enumerate(sorted(matched_words, key=lambda w: concat_freq[w],
                                    reverse=True)):
        corpus_ranks.append(i + 1)
        ref_count = ref_word_freq.get(word, 0)
        ref_ranks.append(ref_count)

    # Spearman-like rank correlation (simplified)
    if len(corpus_ranks) >= 3:
        n = len(corpus_ranks)
        ref_sorted_idx = sorted(range(n), key=lambda i: ref_ranks[i], reverse=True)
        ref_rank_map = {idx: rank + 1 for rank, idx in enumerate(ref_sorted_idx)}
        d_sq_sum = sum((corpus_ranks[i] - ref_rank_map[i]) ** 2 for i in range(n))
        spearman = 1 - 6 * d_sq_sum / (n * (n ** 2 - 1))
    else:
        spearman = 0.0

    print(f"     {len(matched_words)} unique concatenated words matched")
    print(f"     Spearman rank correlation: {spearman:.4f}")

    # ── 9. Save ──
    elapsed = time.time() - t0
    output = {
        'n_confirmed_pairs': n_pairs,
        'n_pair_matches': n_matches,
        'n_content_matches': n_content,
        'pair_match_rate': round(match_rate, 4),
        'null_mean_matches': round(null_mean, 1),
        'null_std_matches': round(null_std, 1),
        'z_score': round(z_score, 2),
        'significant': significant,
        'unique_concatenated_words': [
            {'word': w, 'count': c, 'is_content': _is_content_word(w)}
            for w, c in unique_concats.most_common(50)
        ],
        'pair_matches': pair_matches[:200],  # Truncate for size
        'n_triple_matches': len(triple_matches),
        'triple_matches': triple_matches[:50],
        'domain_analysis': domain_analysis,
        'top_match_folios': [{'folio': f, 'count': c} for f, c in top_folios],
        'spearman_rank_correlation': round(spearman, 4),
        'verdict': (
            f"Pair concat: {n_matches}/{n_pairs} matches ({match_rate:.1%}), "
            f"z={z_score:.2f} ({'SIGNIFICANT' if significant else 'NOT SIG'}). "
            f"{n_content} content words. "
            f"{len(triple_matches)} triple concats."
        ),
        'runtime_seconds': round(elapsed, 1),
    }

    out_path = os.path.join(rd, 'pair_concat.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(output), f, indent=2)
    print(f"\n  Saved → {out_path} ({elapsed:.1f}s)")
