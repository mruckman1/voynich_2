"""
Step 36.3 – Bigram Plausibility at 10K
========================================
Confirms Track G's z=13.12 and computes the full bigram analysis that
Track G's calibration didn't provide: trigrams, quadruples, type analysis,
per-folio z, and a full bigram catalog.

Dependency chain:
    signal_10k.json           (Step 36.2)
    decode_10k.json           (Step 36.1)
        → bigrams_10k.json   (this step)
"""

import json
import os
import random
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.reference import load_reference_corpus
from voynich.phases.signal_bigrams import (
    _build_reference_bigrams,
    _edit_distance_1,
    _find_signal_pairs,
    _find_signal_triples,
    _null_permutation_test,
    _relaxed_bigram_test,
)


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


# Known function words for bigram type classification
_FUNCTION_WORDS = {
    'de', 'in', 'ad', 'cum', 'per', 'pro', 'sub', 'ex', 'ab',
    'et', 'vel', 'aut', 'sed', 'si', 'ne', 'ut', 'non', 'que',
    'a', 'e', 'se', 'te', 'me',
}


def _classify_bigram_type(w1: str, w2: str) -> str:
    """Classify a bigram as function-function, function-content, or content-content."""
    f1 = w1 in _FUNCTION_WORDS
    f2 = w2 in _FUNCTION_WORDS
    if f1 and f2:
        return 'function-function'
    elif f1 or f2:
        return 'function-content'
    else:
        return 'content-content'


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_bigrams_10k() -> None:
    """Step 36.3: Bigram plausibility at 10K dictionary."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 36.3: Bigram Plausibility at 10K")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load signal_10k.json ──
    print("\n  1. Loading signal_10k.json …")
    with open(os.path.join(rd, 'signal_10k.json')) as f:
        sig_data = json.load(f)

    token_folios = sig_data['token_folios']
    token_decoded = sig_data['token_decoded']
    classifications = sig_data['token_classifications']
    n_tokens = sig_data['n_tokens']
    n_signal = sig_data['n_signal']
    signal_rate = sig_data['signal_rate']

    print(f"     {n_tokens} tokens, {n_signal} SIGNAL ({signal_rate:.3f})")

    # ── 2. Load 10K dictionary for filtering reference bigrams ──
    print("  2. Loading 10K dictionary …")
    with open(os.path.join(rd, 'decode_10k.json')) as f:
        decode_data = json.load(f)

    # Rebuild 10K dict from reference corpus
    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    from voynich.phases.dict_calibration import _build_dict_variants
    from voynich.core.reference import build_expanded_word_set
    base_words = set(
        w.lower() for w in ref_corpus.get_combined_tokens('latin')
        if len(w) >= 2
    )
    dict_variants = _build_dict_variants(base_words, ref_corpus, [10000])
    dict_10k = dict_variants[0][1]
    print(f"     10K dictionary: {len(dict_10k)} words")

    # ── 3. Build reference bigrams ──
    print("  3. Building reference bigrams …")
    ref_tokens = [w.lower() for w in ref_corpus.get_combined_tokens('latin')
                  if len(w) >= 2]
    ref_bigrams, ref_trigrams = _build_reference_bigrams(ref_tokens)
    print(f"     {len(ref_bigrams)} reference bigrams, {len(ref_trigrams)} trigrams")

    # Also build 10K-filtered reference bigrams (both words in 10K)
    ref_bigrams_10k = set(
        (w1, w2) for w1, w2 in ref_bigrams
        if w1 in dict_10k and w2 in dict_10k
    )
    print(f"     {len(ref_bigrams_10k)} 10K-filtered reference bigrams")

    # ── 4. Find SIGNAL-SIGNAL pairs ──
    print("  4. Finding SIGNAL-SIGNAL pairs …")
    signal_pairs = _find_signal_pairs(classifications, token_decoded, token_folios)
    print(f"     {len(signal_pairs)} consecutive SIGNAL-SIGNAL pairs")

    # ── 5. Exact bigram matches (against full ref bigrams) ──
    print("  5. Testing exact bigram matches …")
    exact_hits = []
    for folio, pos, w1, w2 in signal_pairs:
        if (w1, w2) in ref_bigrams:
            exact_hits.append({
                'folio': folio, 'position': pos,
                'word1': w1, 'word2': w2,
                'in_10k_filtered': (w1, w2) in ref_bigrams_10k,
                'type': _classify_bigram_type(w1, w2),
            })
    n_exact = len(exact_hits)
    print(f"     {n_exact} exact bigram hits")

    # Also count against 10K-filtered only
    n_exact_10k = sum(1 for h in exact_hits if h['in_10k_filtered'])
    print(f"     {n_exact_10k} hits in 10K-filtered bigrams")

    # ── 6. Relaxed bigram matches ──
    print("  6. Testing relaxed bigram matches (edit distance ≤ 1) …")
    n_relaxed = _relaxed_bigram_test(signal_pairs, ref_bigrams, dict_10k)
    print(f"     {n_relaxed} relaxed bigram hits (excluding exact)")

    # ── 7. Null permutation test ──
    print("  7. Running null permutation test (1000 permutations) …")
    null_rates, null_mean, null_std = _null_permutation_test(
        n_signal, n_tokens, token_decoded, token_folios,
        ref_bigrams, n_perms=1000, seed=42,
    )
    bigram_hit_rate = n_exact / len(signal_pairs) if signal_pairs else 0.0
    bigram_z = (bigram_hit_rate - null_mean) / null_std if null_std > 0 else 0.0
    p_value = sum(1 for r in null_rates if r >= bigram_hit_rate) / len(null_rates) if null_rates else 1.0

    print(f"     Bigram hit rate: {bigram_hit_rate:.4f}")
    print(f"     Null mean: {null_mean:.6f}, std: {null_std:.6f}")
    print(f"     z = {bigram_z:.2f}, p = {p_value:.6f}")

    # ── 8. Trigram test ──
    print("  8. Testing trigrams …")
    signal_triples = _find_signal_triples(classifications, token_decoded, token_folios)
    trigram_hits = []
    for folio, pos, w1, w2, w3 in signal_triples:
        if (w1, w2, w3) in ref_trigrams:
            trigram_hits.append({
                'folio': folio, 'position': pos,
                'word1': w1, 'word2': w2, 'word3': w3,
            })
    print(f"     {len(signal_triples)} SIGNAL triples, {len(trigram_hits)} trigram hits")

    # ── 9. Quadruple test ──
    print("  9. Testing quadruples …")
    # Build reference quadruples
    ref_quadruples: Set[Tuple[str, str, str, str]] = set()
    for i in range(len(ref_tokens) - 3):
        ref_quadruples.add((ref_tokens[i], ref_tokens[i+1], ref_tokens[i+2], ref_tokens[i+3]))

    quad_hits = []
    for i in range(len(classifications) - 3):
        if (classifications[i] == 'SIGNAL'
                and classifications[i+1] == 'SIGNAL'
                and classifications[i+2] == 'SIGNAL'
                and classifications[i+3] == 'SIGNAL'
                and token_folios[i] == token_folios[i+1]
                and token_folios[i+1] == token_folios[i+2]
                and token_folios[i+2] == token_folios[i+3]):
            words = (token_decoded[i], token_decoded[i+1],
                     token_decoded[i+2], token_decoded[i+3])
            if words in ref_quadruples:
                quad_hits.append({
                    'folio': token_folios[i], 'position': i,
                    'words': list(words),
                })
    print(f"     {len(quad_hits)} quadruple hits")

    # ── 10. Bigram type analysis ──
    print(" 10. Bigram type analysis …")
    type_counts: Dict[str, int] = Counter()
    for h in exact_hits:
        type_counts[h['type']] += 1
    print(f"     function-function: {type_counts.get('function-function', 0)}")
    print(f"     function-content:  {type_counts.get('function-content', 0)}")
    print(f"     content-content:   {type_counts.get('content-content', 0)}")

    # ── 11. Per-folio bigram z ──
    print(" 11. Per-folio bigram z …")
    # Group signal pairs by folio
    folio_pairs: Dict[str, List] = defaultdict(list)
    for folio, pos, w1, w2 in signal_pairs:
        folio_pairs[folio].append((folio, pos, w1, w2))

    folio_bigram_z = []
    for folio, pairs in sorted(folio_pairs.items()):
        if len(pairs) < 3:
            continue
        folio_hits = sum(1 for _, _, w1, w2 in pairs if (w1, w2) in ref_bigrams)
        folio_rate = folio_hits / len(pairs) if pairs else 0.0
        # Simple z using corpus-level null stats
        fz = (folio_rate - null_mean) / null_std if null_std > 0 else 0.0
        folio_bigram_z.append({
            'folio': folio,
            'n_pairs': len(pairs),
            'n_hits': folio_hits,
            'hit_rate': round(folio_rate, 4),
            'z': round(fz, 2),
        })
    folio_bigram_z.sort(key=lambda x: x['z'], reverse=True)

    if folio_bigram_z:
        print("     Top folio bigram z:")
        for fb in folio_bigram_z[:5]:
            print(f"       {fb['folio']:<8s} {fb['n_hits']}/{fb['n_pairs']} pairs"
                  f"  rate={fb['hit_rate']:.3f}  z={fb['z']:.1f}")

    # ── 12. Save ──
    elapsed = time.time() - t0

    output = {
        'n_signal_pairs': len(signal_pairs),
        'n_exact_bigram_hits': n_exact,
        'n_exact_10k_filtered': n_exact_10k,
        'n_relaxed_bigram_hits': n_relaxed,
        'bigram_hit_rate': round(bigram_hit_rate, 6),
        'null_mean': round(null_mean, 6),
        'null_std': round(null_std, 6),
        'bigram_z': round(bigram_z, 2),
        'p_value': round(p_value, 6),
        # Trigrams
        'n_signal_triples': len(signal_triples),
        'n_trigram_hits': len(trigram_hits),
        'trigram_hits': trigram_hits,
        # Quadruples
        'n_quad_hits': len(quad_hits),
        'quad_hits': quad_hits,
        # Type analysis
        'bigram_type_counts': dict(type_counts),
        'n_content_content': type_counts.get('content-content', 0),
        # Per-folio
        'folio_bigram_z': folio_bigram_z[:20],
        # Full catalog
        'bigram_catalog': exact_hits,
        # Reference sizes
        'n_ref_bigrams': len(ref_bigrams),
        'n_ref_bigrams_10k': len(ref_bigrams_10k),
        'n_ref_trigrams': len(ref_trigrams),
        'runtime_seconds': round(elapsed, 1),
    }

    out_path = os.path.join(rd, 'bigrams_10k.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved → {out_path}")

    # ── Summary ──
    print("\n" + "=" * 70)
    print("BIGRAMS 10K SUMMARY")
    print("=" * 70)
    print(f"\n  SIGNAL-SIGNAL pairs: {len(signal_pairs)}")
    print(f"  Exact bigram hits: {n_exact}")
    print(f"  Relaxed hits: {n_relaxed}")
    print(f"  Bigram z = {bigram_z:.2f} (p = {p_value:.6f})")
    print(f"  Trigram hits: {len(trigram_hits)}")
    print(f"  Quadruple hits: {len(quad_hits)}")
    print(f"  Content-content bigrams: {type_counts.get('content-content', 0)}")
    print(f"\n  Runtime: {elapsed:.1f}s")
