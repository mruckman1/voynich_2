"""
Step 37.5 – Concatenated Signal Isolation
===========================================
If Step 37.4 finds significant concatenation hits, re-tokenize the corpus
by merging confirmed signal pairs into single tokens, then re-run signal
isolation.

Dependency chain:
    pair_concat.json           (Step 37.4)
    signal_10k.json            (Step 36.2)
    decode_10k.json            (Step 36.1)
    combined_refine.json       (Phase 15)
    modifier_integrate.json    (Phase 16)
        → concat_signal.json   (this step)
"""

import json
import os
import random
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    decode_token_modifier_aware,
)
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

def run_concat_signal() -> None:
    """Step 37.5: Concatenated Signal Isolation."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 37.5: Concatenated Signal Isolation")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load inputs ──
    print("\n  1. Loading inputs …")
    concat_data = _safe_load(os.path.join(rd, 'pair_concat.json'))
    signal_data = _safe_load(os.path.join(rd, 'signal_10k.json'))
    decode_data = _safe_load(os.path.join(rd, 'decode_10k.json'))
    refine_data = _safe_load(os.path.join(rd, 'combined_refine.json'))
    mod_data = _safe_load(os.path.join(rd, 'modifier_integrate.json'))

    significant = concat_data.get('significant', False)
    unique_concats = concat_data.get('unique_concatenated_words', [])
    token_evas = signal_data.get('token_evas', [])
    token_decoded = signal_data.get('token_decoded', [])
    token_classifications = signal_data.get('token_classifications', [])
    token_folios = signal_data.get('token_folios', [])
    null_hits_10k = decode_data.get('null_hits_10k', [])
    assignment = refine_data.get('best_assignment', {})
    modifier_chars = set(mod_data.get('modifier_chars', []))

    print(f"     Concatenation significant: {significant}")
    print(f"     {len(unique_concats)} unique concatenated words")

    # ── 2. Define merge rules ──
    print("  2. Defining merge rules …")
    # Merge pairs where the concatenation matched the dictionary
    merge_pairs: Set[Tuple[str, str]] = set()
    for uc in unique_concats:
        word = uc.get('word', '')
        # Find the component pair from the concat data
        # The word is word1+word2; we need to find valid splits
        # Check all pair_matches for this word
        pass

    # Actually, merge based on decoded pairs that produced dictionary matches
    pair_matches = concat_data.get('pair_matches', [])
    for pm in pair_matches:
        w1 = pm.get('word1', '')
        w2 = pm.get('word2', '')
        if w1 and w2:
            merge_pairs.add((w1, w2))

    print(f"     {len(merge_pairs)} merge rules defined")

    # ── 3. Re-tokenize corpus with merges ──
    print("  3. Re-tokenizing with merges …")
    merged_decoded = []
    merged_folios = []
    merged_evas = []
    merged_classifications = []
    i = 0
    n_merges = 0

    while i < len(token_decoded):
        if i < len(token_decoded) - 1:
            w1 = token_decoded[i].lower()
            w2 = token_decoded[i + 1].lower()
            f1 = token_folios[i] if i < len(token_folios) else ''
            f2 = token_folios[i + 1] if i + 1 < len(token_folios) else ''

            # Only merge within same folio and if both are SIGNAL
            if ((w1, w2) in merge_pairs and f1 == f2 and
                    token_classifications[i] == 'SIGNAL' and
                    token_classifications[i + 1] == 'SIGNAL'):
                merged_decoded.append(w1 + w2)
                merged_folios.append(f1)
                merged_evas.append(token_evas[i] + '+' + token_evas[i + 1])
                merged_classifications.append('MERGED')
                n_merges += 1
                i += 2
                continue

        merged_decoded.append(token_decoded[i].lower())
        merged_folios.append(token_folios[i] if i < len(token_folios) else '')
        merged_evas.append(token_evas[i] if i < len(token_evas) else '')
        merged_classifications.append(token_classifications[i]
                                      if i < len(token_classifications) else '')
        i += 1

    print(f"     Original tokens: {len(token_decoded)}")
    print(f"     Merged tokens:   {len(merged_decoded)}")
    print(f"     Merges applied:  {n_merges}")

    # ── 4. Build 17K dictionary and match ──
    print("  4. Matching merged corpus against 17K dictionary …")
    ref = load_reference_corpus(languages=['latin'], verbose=False)
    base_words = set(w.lower() for w in ref.get_combined_tokens('latin') if len(w) >= 2)
    word_freq = Counter(w.lower() for w in ref.get_combined_tokens('latin') if len(w) >= 2)
    top_10k = set(w for w, _ in word_freq.most_common(10000))
    top_17k = set(w for w, _ in word_freq.most_common(17000))

    merged_hits_10k = [w in top_10k for w in merged_decoded]
    merged_hits_17k = [w in top_17k for w in merged_decoded]

    hit_rate_10k = sum(merged_hits_10k) / len(merged_hits_10k) if merged_hits_10k else 0.0
    hit_rate_17k = sum(merged_hits_17k) / len(merged_hits_17k) if merged_hits_17k else 0.0

    print(f"     10K hit rate: {hit_rate_10k:.3%}")
    print(f"     17K hit rate: {hit_rate_17k:.3%}")

    # ── 5. Signal isolation on merged corpus ──
    print("  5. Signal isolation on merged corpus …")
    # Count per-word signal scores in merged corpus
    merged_word_counts = Counter(w for w, h in zip(merged_decoded, merged_hits_10k) if h)

    # For null comparison, use original null hits but with merged positions
    # Approximate: count null hits at unmerged positions
    null_word_counts: Dict[str, List[int]] = defaultdict(lambda: [0] * len(null_hits_10k))
    for ni, nh in enumerate(null_hits_10k):
        for i in range(len(nh)):
            if nh[i] and i < len(token_decoded):
                w = token_decoded[i].lower()
                null_word_counts[w][ni] += 1

    merged_signal_words = []
    for word, real_count in merged_word_counts.items():
        null_counts = null_word_counts.get(word, [0] * len(null_hits_10k))
        null_mean = sum(null_counts) / len(null_counts) if null_counts else 0.0
        null_var = (sum((c - null_mean) ** 2 for c in null_counts) / len(null_counts)
                    if null_counts else 0.0)
        null_std = null_var ** 0.5
        sigma = ((real_count - null_mean) / null_std if null_std > 0
                 else (10.0 if real_count > null_mean else 0.0))

        if sigma > 2.0:
            merged_signal_words.append({
                'word': word,
                'real_count': real_count,
                'null_mean': round(null_mean, 2),
                'sigma': round(sigma, 2),
                'is_merged': len(word) > 3 and '+' not in word,
            })

    merged_signal_words.sort(key=lambda x: x['sigma'], reverse=True)
    n_merged_signal = sum(1 for w in merged_signal_words if w.get('is_merged'))

    print(f"     {len(merged_signal_words)} signal words in merged corpus")
    print(f"     {n_merged_signal} are merged (concatenated) words")
    print("     Top merged signal words:")
    for ws in merged_signal_words[:10]:
        m = "[M]" if ws.get('is_merged') else "   "
        print(f"       {m} {ws['word']:<15s} σ={ws['sigma']:>7.2f} count={ws['real_count']}")

    # ── 6. Save ──
    elapsed = time.time() - t0

    output = {
        'concatenation_significant': significant,
        'n_merge_rules': len(merge_pairs),
        'n_original_tokens': len(token_decoded),
        'n_merged_tokens': len(merged_decoded),
        'n_merges_applied': n_merges,
        'merged_hit_rate_10k': round(hit_rate_10k, 4),
        'merged_hit_rate_17k': round(hit_rate_17k, 4),
        'n_signal_words_merged': len(merged_signal_words),
        'n_merged_signal_words': n_merged_signal,
        'merged_signal_words': merged_signal_words[:50],
        'merge_pairs': [list(p) for p in sorted(merge_pairs)],
        'verdict': (
            f"Merged corpus: {len(merged_decoded)} tokens ({n_merges} merges). "
            f"10K hit={hit_rate_10k:.3%}. "
            f"{len(merged_signal_words)} signal words, "
            f"{n_merged_signal} merged. "
            f"Concat {'SIGNIFICANT' if significant else 'not significant'}."
        ),
        'runtime_seconds': round(elapsed, 1),
    }

    out_path = os.path.join(rd, 'concat_signal.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(output), f, indent=2)
    print(f"\n  Saved → {out_path} ({elapsed:.1f}s)")
