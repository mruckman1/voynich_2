"""
Step 37.6 – Concatenated Bigram Test
=======================================
Compute bigram plausibility on the merged token stream.  If merged tokens
are real Latin words, their consecutive pairs should match Latin word
bigrams — and these would be content-content bigrams.

Dependency chain:
    concat_signal.json         (Step 37.5)
    pair_concat.json           (Step 37.4)
    signal_10k.json            (Step 36.2)
    bigrams_10k.json           (Step 36.3)
        → concat_bigrams.json  (this step)
"""

import json
import os
import random
import time
from collections import Counter
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


_FUNCTION_WORDS = {
    'de', 'in', 'ad', 'et', 'cum', 'per', 'pro', 'ex', 'ab', 'sub',
    'non', 'si', 'ut', 'sed', 'ne', 'ac', 'at', 'an', 'se', 'te',
    'me', 'nos', 'iam', 'est', 'sunt', 'hoc', 'id', 'ea', 'is',
    'di', 'du', 'ce', 'ci', 'co', 'cu', 'bi', 'bo', 'be', 'da',
    'la', 'le', 'li', 'lo', 'ni', 'no', 'nu', 'ra', 're', 'ri',
    'ro', 'sa', 'so', 'su', 'ta', 'ti', 'to',
}


def _is_content_word(word: str) -> bool:
    return word.lower() not in _FUNCTION_WORDS and len(word) >= 4


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_concat_bigrams() -> None:
    """Step 37.6: Concatenated Bigram Test."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 37.6: Concatenated Bigram Test")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load inputs ──
    print("\n  1. Loading inputs …")
    concat_sig = _safe_load(os.path.join(rd, 'concat_signal.json'))
    signal_data = _safe_load(os.path.join(rd, 'signal_10k.json'))
    bigram_data = _safe_load(os.path.join(rd, 'bigrams_10k.json'))

    merged_signal_words_list = concat_sig.get('merged_signal_words', [])
    merged_signal_set = set(w['word'] for w in merged_signal_words_list)
    original_bigram_z = bigram_data.get('bigram_z', 0.0)

    # Reconstruct merged decoded stream from signal_10k + merge rules
    token_decoded = signal_data.get('token_decoded', [])
    token_classifications = signal_data.get('token_classifications', [])
    token_folios = signal_data.get('token_folios', [])
    merge_pairs_raw = concat_sig.get('merge_pairs', [])
    merge_pairs = set(tuple(p) for p in merge_pairs_raw)

    print(f"     {len(merged_signal_set)} merged signal words")
    print(f"     Original bigram z: {original_bigram_z:.2f}")

    # ── 2. Build 17K reference bigram table ──
    print("  2. Building 17K reference bigram table …")
    ref = load_reference_corpus(languages=['latin'], verbose=False)
    ref_tokens = [w.lower() for w in ref.get_combined_tokens('latin') if len(w) >= 2]
    word_freq = Counter(ref_tokens)
    top_17k = set(w for w, _ in word_freq.most_common(17000))
    top_10k = set(w for w, _ in word_freq.most_common(10000))

    ref_bigrams_17k: Set[Tuple[str, str]] = set()
    ref_bigrams_10k: Set[Tuple[str, str]] = set()
    for i in range(len(ref_tokens) - 1):
        w1, w2 = ref_tokens[i], ref_tokens[i + 1]
        if w1 in top_17k and w2 in top_17k:
            ref_bigrams_17k.add((w1, w2))
        if w1 in top_10k and w2 in top_10k:
            ref_bigrams_10k.add((w1, w2))

    print(f"     {len(ref_bigrams_17k)} reference bigrams (17K)")
    print(f"     {len(ref_bigrams_10k)} reference bigrams (10K)")

    # ── 3. Re-tokenize with merges ──
    print("  3. Re-tokenizing with merges …")
    merged_decoded = []
    merged_folios = []
    merged_cls = []
    i = 0
    while i < len(token_decoded):
        if i < len(token_decoded) - 1:
            w1 = token_decoded[i].lower()
            w2 = token_decoded[i + 1].lower()
            f1 = token_folios[i] if i < len(token_folios) else ''
            f2 = token_folios[i + 1] if i + 1 < len(token_folios) else ''
            c1 = token_classifications[i] if i < len(token_classifications) else ''
            c2 = token_classifications[i + 1] if i + 1 < len(token_classifications) else ''

            if ((w1, w2) in merge_pairs and f1 == f2 and
                    c1 == 'SIGNAL' and c2 == 'SIGNAL'):
                merged_decoded.append(w1 + w2)
                merged_folios.append(f1)
                merged_cls.append('MERGED')
                i += 2
                continue

        merged_decoded.append(token_decoded[i].lower())
        merged_folios.append(token_folios[i] if i < len(token_folios) else '')
        merged_cls.append(token_classifications[i]
                          if i < len(token_classifications) else '')
        i += 1

    # ── 4. Find SIGNAL-SIGNAL pairs in merged stream ──
    print("  4. Finding SIGNAL-SIGNAL pairs in merged stream …")
    # For merged stream, a token is "signal" if it's SIGNAL or MERGED
    signal_signal_pairs = []
    for i in range(len(merged_decoded) - 1):
        if merged_folios[i] != merged_folios[i + 1]:
            continue
        c1 = merged_cls[i]
        c2 = merged_cls[i + 1]
        if c1 in ('SIGNAL', 'MERGED') and c2 in ('SIGNAL', 'MERGED'):
            signal_signal_pairs.append((merged_decoded[i], merged_decoded[i + 1]))

    print(f"     {len(signal_signal_pairs)} SIGNAL/MERGED pairs")

    # ── 5. Bigram test ──
    print("  5. Bigram test on merged pairs …")
    exact_17k = []
    exact_10k = []
    content_content = []

    for w1, w2 in signal_signal_pairs:
        if (w1, w2) in ref_bigrams_17k:
            exact_17k.append((w1, w2))
            if _is_content_word(w1) and _is_content_word(w2):
                content_content.append((w1, w2))
        if (w1, w2) in ref_bigrams_10k:
            exact_10k.append((w1, w2))

    print(f"     Exact hits (17K): {len(exact_17k)}")
    print(f"     Exact hits (10K): {len(exact_10k)}")
    print(f"     Content-content:  {len(content_content)}")

    if exact_17k:
        unique_exact = Counter(exact_17k)
        print("     Top exact bigram matches:")
        for bg, cnt in unique_exact.most_common(10):
            cc = " [CC]" if _is_content_word(bg[0]) and _is_content_word(bg[1]) else ""
            print(f"       \"{bg[0]} {bg[1]}\" ×{cnt}{cc}")

    # ── 6. Null permutation z-score ──
    print("  6. Null permutation z-score …")
    rng = random.Random(42)
    null_counts = []
    merged_signal_tokens = [w for w, c in zip(merged_decoded, merged_cls)
                            if c in ('SIGNAL', 'MERGED')]

    for _ in range(1000):
        shuffled = list(merged_signal_tokens)
        rng.shuffle(shuffled)
        null_hits = 0
        for j in range(len(shuffled) - 1):
            if (shuffled[j], shuffled[j + 1]) in ref_bigrams_17k:
                null_hits += 1
        null_counts.append(null_hits)

    null_mean = sum(null_counts) / len(null_counts)
    null_var = sum((c - null_mean) ** 2 for c in null_counts) / len(null_counts)
    null_std = null_var ** 0.5
    merged_z = ((len(exact_17k) - null_mean) / null_std if null_std > 0
                else (10.0 if len(exact_17k) > null_mean else 0.0))

    print(f"     Real hits: {len(exact_17k)}")
    print(f"     Null mean: {null_mean:.1f} (std={null_std:.1f})")
    print(f"     z-score:   {merged_z:.2f}")

    # ── 7. Comparison to Phase 36 ──
    print("  7. Comparison to Phase 36 …")
    print(f"     Phase 36 bigram z:  {original_bigram_z:.2f}")
    print(f"     Merged bigram z:    {merged_z:.2f}")
    z_improved = merged_z > original_bigram_z

    # ── 8. Save ──
    elapsed = time.time() - t0

    output = {
        'n_merged_tokens': len(merged_decoded),
        'n_signal_merged_pairs': len(signal_signal_pairs),
        'n_exact_hits_17k': len(exact_17k),
        'n_exact_hits_10k': len(exact_10k),
        'n_content_content': len(content_content),
        'content_content_bigrams': [
            {'word1': w1, 'word2': w2} for w1, w2 in content_content[:50]
        ],
        'exact_bigrams_17k': [
            {'word1': w1, 'word2': w2, 'count': c}
            for (w1, w2), c in Counter(exact_17k).most_common(50)
        ],
        'null_mean': round(null_mean, 1),
        'null_std': round(null_std, 1),
        'merged_bigram_z': round(merged_z, 2),
        'original_bigram_z': round(original_bigram_z, 2),
        'z_improved': z_improved,
        'verdict': (
            f"Merged bigrams: z={merged_z:.2f} "
            f"(Phase 36: {original_bigram_z:.2f}). "
            f"{len(exact_17k)} exact 17K hits, "
            f"{len(content_content)} content-content. "
            f"{'Z IMPROVED' if z_improved else 'Z not improved'}."
        ),
        'runtime_seconds': round(elapsed, 1),
    }

    out_path = os.path.join(rd, 'concat_bigrams.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(output), f, indent=2)
    print(f"\n  Saved → {out_path} ({elapsed:.1f}s)")
