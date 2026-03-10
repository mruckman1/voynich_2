"""
Step 37.14 – Italian Signal Pipeline
=======================================
Run the full signal isolation and bigram analysis at the Italian 10K
dictionary. Test merged Latin ∪ Italian dictionary for macaronic text.

Dependency chain:
    italian_10k.json           (Step 37.13)
    signal_10k.json            (Step 36.2)
    decode_10k.json            (Step 36.1)
    bigrams_10k.json           (Step 36.3)
        → italian_signal.json  (this step)
"""

import json
import os
import random
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.reference import load_reference_corpus


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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_italian_signal() -> None:
    """Step 37.14: Italian Signal Pipeline."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 37.14: Italian Signal Pipeline")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load inputs ──
    print("\n  1. Loading inputs …")
    italian_data = _safe_load(os.path.join(rd, 'italian_10k.json'))
    signal_data = _safe_load(os.path.join(rd, 'signal_10k.json'))
    decode_data = _safe_load(os.path.join(rd, 'decode_10k.json'))
    bigram_data = _safe_load(os.path.join(rd, 'bigrams_10k.json'))

    italian_10k_words = italian_data.get('italian_10k_words', [])
    italian_10k = set(italian_10k_words)
    token_decoded = signal_data.get('token_decoded', [])
    token_classifications = signal_data.get('token_classifications', [])
    token_folios = signal_data.get('token_folios', [])
    null_hits_10k = decode_data.get('null_hits_10k', [])
    original_bigram_z = bigram_data.get('bigram_z', 0.0)

    # Build Latin 10K
    ref = load_reference_corpus(languages=['latin'], verbose=False)
    word_freq = Counter(w.lower() for w in ref.get_combined_tokens('latin') if len(w) >= 2)
    latin_10k = set(w for w, _ in word_freq.most_common(10000))

    print(f"     {len(italian_10k)} Italian 10K words")
    print(f"     {len(latin_10k)} Latin 10K words")
    print(f"     {len(token_decoded)} decoded tokens")

    if not italian_10k:
        output = {
            'error': 'No Italian 10K dictionary',
            'verdict': 'FAIL: Italian dictionary not available',
            'runtime_seconds': round(time.time() - t0, 1),
        }
        out_path = os.path.join(rd, 'italian_signal.json')
        with open(out_path, 'w') as f:
            json.dump(output, f, indent=2)
        return

    decoded_lower = [w.lower() for w in token_decoded]

    # ── 2. Signal isolation at Italian 10K ──
    print("  2. Signal isolation at Italian 10K …")
    italian_hits = [w in italian_10k for w in decoded_lower]

    # Null comparison: how often do null-decoded tokens match Italian?
    null_italian_hit_counts = []
    for null_run in null_hits_10k:
        # Approximate: for null positions that hit Latin 10K, check Italian too
        null_count = sum(1 for i, dec in enumerate(decoded_lower)
                        if null_run[i] and dec in italian_10k)
        null_italian_hit_counts.append(null_count)

    n_tokens = len(decoded_lower)
    italian_signal_count = sum(italian_hits)
    null_italian_mean = (sum(null_italian_hit_counts) / len(null_italian_hit_counts)
                         if null_italian_hit_counts else 0.0)
    null_italian_std = 0.0
    if null_italian_hit_counts:
        var = sum((c - null_italian_mean) ** 2 for c in null_italian_hit_counts) / len(null_italian_hit_counts)
        null_italian_std = var ** 0.5

    italian_signal_rate = italian_signal_count / n_tokens if n_tokens > 0 else 0.0

    # Per-word Italian signal analysis
    italian_word_counts = Counter(w for w, h in zip(decoded_lower, italian_hits) if h)
    italian_signal_words = []

    for word, real_count in italian_word_counts.items():
        # Check if also in Latin 10K
        in_latin = word in latin_10k

        # Null counts
        null_counts = []
        for null_run in null_hits_10k:
            nc = sum(1 for i in range(n_tokens)
                    if decoded_lower[i] == word and null_run[i])
            null_counts.append(nc)

        null_mean = sum(null_counts) / len(null_counts) if null_counts else 0.0
        null_var = (sum((c - null_mean) ** 2 for c in null_counts) / len(null_counts)
                    if null_counts else 0.0)
        null_std = null_var ** 0.5
        sigma = ((real_count - null_mean) / null_std if null_std > 0
                 else (10.0 if real_count > null_mean else 0.0))

        if sigma > 2.0 and real_count >= 5:
            italian_signal_words.append({
                'word': word,
                'real_count': real_count,
                'sigma': round(sigma, 2),
                'in_latin_10k': in_latin,
                'italian_only': not in_latin,
            })

    italian_signal_words.sort(key=lambda x: x['sigma'], reverse=True)
    italian_only_signals = [w for w in italian_signal_words if w['italian_only']]

    print(f"     Italian SIGNAL rate: {italian_signal_rate:.3%}")
    print(f"     {len(italian_signal_words)} Italian signal words (σ>2)")
    print(f"     {len(italian_only_signals)} Italian-only signal words")

    if italian_only_signals:
        print("     Italian-only signals:")
        for ws in italian_only_signals[:10]:
            print(f"       {ws['word']:<12s} σ={ws['sigma']:>7.2f} count={ws['real_count']}")

    # ── 3. Italian bigram test ──
    print("  3. Italian bigram test …")
    # Build Italian bigram table from the Italian corpus
    italian_corpus_data = _safe_load(os.path.join(rd, 'italian_corpus.json'))
    italian_top_words = italian_corpus_data.get('top_words', [])
    italian_word_bigrams = italian_corpus_data.get('top_word_bigrams', [])

    # Build bigram set
    italian_bigrams: Set[Tuple[str, str]] = set()
    for bg in italian_word_bigrams:
        pair = bg.get('bigram', [])
        if len(pair) == 2:
            italian_bigrams.add((pair[0], pair[1]))

    # Test SIGNAL-SIGNAL pairs against Italian bigrams
    italian_signal_set = set(w['word'] for w in italian_signal_words)
    italian_exact_hits = 0
    italian_signal_pairs = []

    for i in range(len(decoded_lower) - 1):
        if i >= len(token_folios) - 1:
            break
        if token_folios[i] != token_folios[i + 1]:
            continue
        w1, w2 = decoded_lower[i], decoded_lower[i + 1]
        if w1 in italian_signal_set and w2 in italian_signal_set:
            italian_signal_pairs.append((w1, w2))
            if (w1, w2) in italian_bigrams:
                italian_exact_hits += 1

    # z-score for Italian bigrams
    rng = random.Random(42)
    it_signal_list = list(italian_signal_set)
    null_it_counts = []
    for _ in range(500):
        shuffled = list(it_signal_list) * 5
        rng.shuffle(shuffled)
        null_hits = sum(1 for j in range(len(shuffled) - 1)
                       if (shuffled[j], shuffled[j + 1]) in italian_bigrams)
        null_it_counts.append(null_hits)

    null_it_mean = sum(null_it_counts) / len(null_it_counts) if null_it_counts else 0.0
    null_it_var = (sum((c - null_it_mean) ** 2 for c in null_it_counts) / len(null_it_counts)
                   if null_it_counts else 0.0)
    null_it_std = null_it_var ** 0.5
    italian_bigram_z = ((italian_exact_hits - null_it_mean) / null_it_std
                        if null_it_std > 0
                        else (10.0 if italian_exact_hits > null_it_mean else 0.0))

    print(f"     Italian SIGNAL-SIGNAL pairs: {len(italian_signal_pairs)}")
    print(f"     Italian exact bigram hits: {italian_exact_hits}")
    print(f"     Italian bigram z: {italian_bigram_z:.2f}")

    # ── 4. Merged dictionary ──
    print("  4. Merged Latin ∪ Italian dictionary …")
    merged_dict = latin_10k | italian_10k
    print(f"     Merged dictionary: {len(merged_dict)} words")

    merged_hits = [w in merged_dict for w in decoded_lower]
    merged_hit_rate = sum(merged_hits) / n_tokens if n_tokens > 0 else 0.0

    # Merged signal isolation
    merged_signal_words = []
    merged_word_counts = Counter(w for w, h in zip(decoded_lower, merged_hits) if h)

    for word, real_count in merged_word_counts.items():
        if real_count < 5:
            continue
        null_counts = []
        for null_run in null_hits_10k:
            nc = sum(1 for i in range(n_tokens)
                    if decoded_lower[i] == word and null_run[i])
            null_counts.append(nc)
        null_mean = sum(null_counts) / len(null_counts) if null_counts else 0.0
        null_var = (sum((c - null_mean) ** 2 for c in null_counts) / len(null_counts)
                    if null_counts else 0.0)
        null_std = null_var ** 0.5
        sigma = ((real_count - null_mean) / null_std if null_std > 0
                 else (10.0 if real_count > null_mean else 0.0))
        if sigma > 2.0:
            merged_signal_words.append({
                'word': word,
                'sigma': round(sigma, 2),
                'in_latin': word in latin_10k,
                'in_italian': word in italian_10k,
            })

    merged_signal_words.sort(key=lambda x: x['sigma'], reverse=True)
    merged_signal_rate = len(merged_signal_words) / n_tokens if n_tokens > 0 else 0.0

    print(f"     Merged hit rate: {merged_hit_rate:.3%}")
    print(f"     Merged signal words: {len(merged_signal_words)}")

    # ── 5. Merged bigram test ──
    print("  5. Merged bigram test …")
    # Build Latin bigram table
    ref_tokens_lower = [w.lower() for w in ref.get_combined_tokens('latin') if len(w) >= 2]
    latin_bigrams: Set[Tuple[str, str]] = set()
    for i in range(len(ref_tokens_lower) - 1):
        if ref_tokens_lower[i] in latin_10k and ref_tokens_lower[i + 1] in latin_10k:
            latin_bigrams.add((ref_tokens_lower[i], ref_tokens_lower[i + 1]))

    merged_bigrams = latin_bigrams | italian_bigrams
    merged_signal_set = set(w['word'] for w in merged_signal_words)

    merged_exact = 0
    merged_cc = 0
    for i in range(len(decoded_lower) - 1):
        if i >= len(token_folios) - 1:
            break
        if token_folios[i] != token_folios[i + 1]:
            continue
        w1, w2 = decoded_lower[i], decoded_lower[i + 1]
        if w1 in merged_signal_set and w2 in merged_signal_set:
            if (w1, w2) in merged_bigrams:
                merged_exact += 1
                # Cross-language bigram?
                w1_lat = w1 in latin_10k
                w2_ita = w2 in italian_10k
                if (w1_lat and not w1 in italian_10k) or (w2_ita and not w2 in latin_10k):
                    merged_cc += 1

    # z-score
    null_merged_counts = []
    merged_sl = list(merged_signal_set)
    for _ in range(500):
        shuffled = list(merged_sl) * 5
        rng.shuffle(shuffled)
        null_hits = sum(1 for j in range(len(shuffled) - 1)
                       if (shuffled[j], shuffled[j + 1]) in merged_bigrams)
        null_merged_counts.append(null_hits)

    null_m_mean = sum(null_merged_counts) / len(null_merged_counts) if null_merged_counts else 0.0
    null_m_var = (sum((c - null_m_mean) ** 2 for c in null_merged_counts) / len(null_merged_counts)
                  if null_merged_counts else 0.0)
    null_m_std = null_m_var ** 0.5
    merged_bigram_z = ((merged_exact - null_m_mean) / null_m_std
                       if null_m_std > 0
                       else (10.0 if merged_exact > null_m_mean else 0.0))

    print(f"     Merged exact bigram hits: {merged_exact}")
    print(f"     Cross-language bigrams: {merged_cc}")
    print(f"     Merged bigram z: {merged_bigram_z:.2f}")

    # ── 6. Comparison table ──
    print("  6. Comparison …")
    latin_signal_rate = signal_data.get('signal_rate', 0.0)

    comparison = {
        'latin_10k': {
            'signal_rate': round(latin_signal_rate, 4),
            'bigram_z': round(original_bigram_z, 2),
        },
        'italian_10k': {
            'signal_rate': round(italian_signal_rate, 4),
            'bigram_z': round(italian_bigram_z, 2),
        },
        'merged': {
            'signal_rate': round(merged_signal_rate, 4),
            'bigram_z': round(merged_bigram_z, 2),
            'cross_language_bigrams': merged_cc,
        },
    }

    # Is the text macaronic?
    is_macaronic = merged_bigram_z > original_bigram_z and merged_cc > 0

    print(f"     Latin 10K:   signal={latin_signal_rate:.4f}, z={original_bigram_z:.2f}")
    print(f"     Italian 10K: signal={italian_signal_rate:.4f}, z={italian_bigram_z:.2f}")
    print(f"     Merged:      signal={merged_signal_rate:.4f}, z={merged_bigram_z:.2f}")
    print(f"     Macaronic: {'YES' if is_macaronic else 'NO'}")

    # ── 7. Save ──
    elapsed = time.time() - t0

    output = {
        'italian_signal_rate': round(italian_signal_rate, 4),
        'n_italian_signal_words': len(italian_signal_words),
        'n_italian_only_signals': len(italian_only_signals),
        'italian_signal_words': italian_signal_words[:50],
        'italian_only_signals': italian_only_signals[:30],
        'italian_bigram_z': round(italian_bigram_z, 2),
        'italian_exact_hits': italian_exact_hits,
        'merged_dict_size': len(merged_dict),
        'merged_hit_rate': round(merged_hit_rate, 4),
        'merged_signal_rate': round(merged_signal_rate, 4),
        'merged_bigram_z': round(merged_bigram_z, 2),
        'merged_exact_hits': merged_exact,
        'cross_language_bigrams': merged_cc,
        'comparison': comparison,
        'is_macaronic': is_macaronic,
        'verdict': (
            f"Italian signal: rate={italian_signal_rate:.4f}, z={italian_bigram_z:.2f}. "
            f"Merged: z={merged_bigram_z:.2f}, cross-lang={merged_cc}. "
            f"{'MACARONIC' if is_macaronic else 'NOT MACARONIC'}. "
            f"{len(italian_only_signals)} Italian-only signal words."
        ),
        'runtime_seconds': round(elapsed, 1),
    }

    out_path = os.path.join(rd, 'italian_signal.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(output), f, indent=2)
    print(f"\n  Saved → {out_path} ({elapsed:.1f}s)")
