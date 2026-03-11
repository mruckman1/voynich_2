"""
Step 40.2 – Venetian Form Matching
====================================
Match the Phase 16 decoded corpus against the Venetian medical form
dictionary and compare to Latin matching.

Dependency chain:
    venetian_forms.json    (Step 40.1)
    merged_signal.json     (Step 38.3)
    null_corpus.json       (Step 17)
        → venetian_match.json  (this step)
"""

import json
import math
import os
import time
from collections import Counter
from typing import Any, Dict, List, Set, Tuple

from voynich.core._paths import results_dir as _results_dir


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
# Core: Signal classification
# ---------------------------------------------------------------------------

def _classify_tokens_4class(
    decoded_tokens: List[str],
    real_word_set: Set[str],
    null_decoded_lists: List[List[str]],
    null_word_set: Set[str],
) -> Tuple[List[str], Dict]:
    """Classify tokens into SIGNAL / SHARED_HIT / SHARED_MISS / ANTI_SIGNAL.

    Uses the same 4-class scheme as merged_signal.py:
    - SIGNAL: real hit, ≤1 null hit
    - SHARED_HIT: real hit, ≥3 null hits
    - ANTI_SIGNAL: no real hit, ≥3 null hits
    - SHARED_MISS: no real hit, ≤1 null hit
    """
    n = len(decoded_tokens)
    classifications = []
    counts = Counter()

    for i in range(n):
        word = decoded_tokens[i]
        real_hit = word in real_word_set if word else False

        # Count null hits: how many null corpora also hit this word?
        null_hit_count = 0
        for null_decoded in null_decoded_lists:
            if i < len(null_decoded):
                null_word = null_decoded[i]
                if null_word in null_word_set:
                    null_hit_count += 1

        if real_hit and null_hit_count <= 1:
            cls = 'SIGNAL'
        elif real_hit and null_hit_count >= 3:
            cls = 'SHARED_HIT'
        elif not real_hit and null_hit_count >= 3:
            cls = 'ANTI_SIGNAL'
        else:
            cls = 'SHARED_MISS'

        classifications.append(cls)
        counts[cls] += 1

    return classifications, dict(counts)


def _compute_per_word_signal(
    decoded_tokens: List[str],
    classifications: List[str],
    null_decoded_lists: List[List[str]],
    venetian_set: Set[str],
) -> List[Dict]:
    """Compute per-word sigma scores for Venetian signal classification."""
    word_real_counts: Counter = Counter()
    word_null_counts: Dict[str, List[int]] = {}

    # Count real occurrences
    for i, word in enumerate(decoded_tokens):
        if not word:
            continue
        if word in venetian_set:
            word_real_counts[word] += 1
        if word not in word_null_counts:
            word_null_counts[word] = [0] * len(null_decoded_lists)

    # Count null occurrences
    for null_idx, null_decoded in enumerate(null_decoded_lists):
        null_hits: Counter = Counter()
        for nw in null_decoded:
            if nw in venetian_set:
                null_hits[nw] += 1
        for word in word_null_counts:
            word_null_counts[word][null_idx] = null_hits.get(word, 0)

    # Compute sigma
    results = []
    for word in sorted(word_real_counts.keys(), key=lambda w: -word_real_counts[w]):
        real_count = word_real_counts[word]
        null_vals = word_null_counts.get(word, [0] * len(null_decoded_lists))
        null_mean = sum(null_vals) / len(null_vals) if null_vals else 0.0
        null_std = (sum((v - null_mean) ** 2 for v in null_vals) / len(null_vals)) ** 0.5 if null_vals else 0.0

        sigma = (real_count - null_mean) / null_std if null_std > 0.01 else (
            float('inf') if real_count > null_mean else 0.0
        )
        selectivity = real_count / null_mean if null_mean > 0.01 else float('inf')

        results.append({
            'word': word,
            'real_count': real_count,
            'null_mean': round(null_mean, 2),
            'null_std': round(null_std, 2),
            'sigma': round(sigma, 2) if not math.isinf(sigma) else 999.0,
            'selectivity': round(selectivity, 2) if not math.isinf(selectivity) else 999.0,
            'is_genuine_signal': sigma > 2.0,
        })

    return results


def _reclassify_signal_words(
    merged_signal_words: List[Dict],
    venetian_set: Set[str],
    venetian_forms_data: Dict,
) -> List[Dict]:
    """Reclassify the 73 merged signal words against Venetian dictionary."""
    provenance = venetian_forms_data.get('provenance_map', {})
    results = []
    for sw in merged_signal_words:
        word = sw.get('word', '')
        entry = {
            'word': word,
            'original_sigma': sw.get('sigma', 0.0),
            'original_source': sw.get('source', ''),
            'in_venetian_extended': word in venetian_set,
            'venetian_origin': provenance.get(word, ''),
        }
        results.append(entry)
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_venetian_match() -> None:
    """Step 40.2: Venetian Form Matching."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 40.2: Venetian Form Matching")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load inputs ──
    print("\n  1. Loading inputs …")
    ven_forms = _safe_load(os.path.join(rd, 'venetian_forms.json'))
    merged_signal = _safe_load(os.path.join(rd, 'merged_signal.json'))
    null_data = _safe_load(os.path.join(rd, 'null_corpus.json'))

    venetian_set = set(ven_forms.get('venetian_extended_set', []))
    decoded_tokens = merged_signal.get('token_decoded', [])
    token_folios = merged_signal.get('token_folios', [])
    merged_classifications = merged_signal.get('token_classifications', [])
    merged_word_signals = merged_signal.get('word_signals', [])

    print(f"    Venetian extended set: {len(venetian_set):,} words")
    print(f"    Decoded tokens: {len(decoded_tokens):,}")

    # ── 2. Compute Venetian dict-hit ──
    print("\n  2. Computing Venetian dict-hit …")
    n_tokens = len(decoded_tokens)
    ven_hits = sum(1 for w in decoded_tokens if w and w in venetian_set)
    ven_dict_hit = ven_hits / n_tokens if n_tokens > 0 else 0.0
    print(f"    Venetian dict-hit: {ven_hits:,}/{n_tokens:,} = {ven_dict_hit:.4f}")

    # Compare to merged dict-hit (from merged_signal)
    merged_dict_hit = merged_signal.get('signal_rate', 0.0)
    # Actually use the stored classification to compute merged hit rate
    merged_hits = sum(1 for c in merged_classifications
                      if c in ('SIGNAL', 'SHARED_HIT'))
    merged_hit_rate = merged_hits / n_tokens if n_tokens > 0 else 0.0
    print(f"    Merged dict-hit (SIGNAL+SHARED): {merged_hits:,}/{n_tokens:,} = {merged_hit_rate:.4f}")
    print(f"    Delta: {ven_dict_hit - merged_hit_rate:+.4f}")

    # ── 3. Load null decoded tokens ──
    print("\n  3. Loading null corpora for signal isolation …")
    null_decoded_lists = []
    for run in null_data.get('null_runs', []):
        null_decoded = run.get('decoded_tokens', [])
        if null_decoded:
            null_decoded_lists.append(null_decoded)
    print(f"    Loaded {len(null_decoded_lists)} null decoded corpora")

    # Compute null hit rates against Venetian set
    null_ven_rates = []
    for nd in null_decoded_lists:
        nh = sum(1 for w in nd if w and w in venetian_set)
        null_ven_rates.append(nh / len(nd) if nd else 0.0)
    null_mean = sum(null_ven_rates) / len(null_ven_rates) if null_ven_rates else 0.0
    print(f"    Null mean Venetian hit rate: {null_mean:.4f}")

    ven_selectivity = ven_dict_hit / null_mean if null_mean > 0.001 else 999.0
    print(f"    Venetian selectivity: {ven_selectivity:.2f}×")

    # ── 4. Classify tokens ──
    print("\n  4. Classifying tokens (4-class) …")
    ven_classifications, class_counts = _classify_tokens_4class(
        decoded_tokens, venetian_set, null_decoded_lists, venetian_set,
    )
    n_signal = class_counts.get('SIGNAL', 0)
    ven_signal_rate = n_signal / n_tokens if n_tokens > 0 else 0.0
    print(f"    SIGNAL: {n_signal:,} ({ven_signal_rate:.4f})")
    print(f"    SHARED_HIT: {class_counts.get('SHARED_HIT', 0):,}")
    print(f"    SHARED_MISS: {class_counts.get('SHARED_MISS', 0):,}")
    print(f"    ANTI_SIGNAL: {class_counts.get('ANTI_SIGNAL', 0):,}")

    # ── 5. Per-word signal analysis ──
    print("\n  5. Computing per-word signal …")
    word_signals = _compute_per_word_signal(
        decoded_tokens, ven_classifications, null_decoded_lists, venetian_set,
    )
    n_genuine = sum(1 for w in word_signals if w['is_genuine_signal'])
    print(f"    Genuine Venetian signal words (σ>2): {n_genuine}")

    # ── 6. Reclassify the 73 merged signal words ──
    print("\n  6. Reclassifying merged signal words …")
    reclassified = _reclassify_signal_words(
        merged_word_signals, venetian_set, ven_forms,
    )
    n_in_ven = sum(1 for r in reclassified if r['in_venetian_extended'])
    print(f"    Merged signal words in Venetian set: {n_in_ven}/{len(reclassified)}")

    # ── 7. Comparison table ──
    print("\n  7. Comparison:")
    # Check Venetian-only matches (in Venetian but not in merged)
    merged_dict = _safe_load(os.path.join(rd, 'merged_dict.json'))
    merged_word_set = set(merged_dict.get('latin_10k_words', []))
    merged_word_set.update(merged_dict.get('italian_10k_words', []))
    for entry in _safe_load(os.path.join(rd, 'venetian_lexicon.json')).get('supplement_words', []):
        if isinstance(entry, str):
            merged_word_set.add(entry)
        elif isinstance(entry, dict):
            merged_word_set.add(entry.get('word', ''))

    ven_only_hits = sum(1 for w in decoded_tokens
                        if w and w in venetian_set and w not in merged_word_set)
    both_hits = sum(1 for w in decoded_tokens
                    if w and w in venetian_set and w in merged_word_set)
    merged_only_hits = sum(1 for w in decoded_tokens
                           if w and w not in venetian_set and w in merged_word_set)
    neither = n_tokens - ven_only_hits - both_hits - merged_only_hits
    print(f"    Venetian-only hits: {ven_only_hits:,}")
    print(f"    Both Venetian + merged: {both_hits:,}")
    print(f"    Merged-only hits: {merged_only_hits:,}")
    print(f"    Neither: {neither:,}")

    # ── 8. Save ──
    elapsed = time.time() - t0

    output = {
        'n_tokens': n_tokens,
        'venetian_dict_hit': round(ven_dict_hit, 6),
        'venetian_dict_hit_count': ven_hits,
        'merged_hit_rate': round(merged_hit_rate, 6),
        'delta_vs_merged': round(ven_dict_hit - merged_hit_rate, 6),
        'null_mean_venetian': round(null_mean, 6),
        'venetian_selectivity': round(ven_selectivity, 4),
        'n_venetian_signal': n_signal,
        'venetian_signal_rate': round(ven_signal_rate, 6),
        'class_counts': class_counts,
        'token_classifications_venetian': ven_classifications,
        'token_folios': token_folios,
        'token_decoded': decoded_tokens,
        'n_genuine_venetian_signal_words': n_genuine,
        'venetian_word_signals': word_signals[:200],  # top 200
        'reclassified_signal_words': reclassified,
        'comparison': {
            'venetian_only': ven_only_hits,
            'both': both_hits,
            'merged_only': merged_only_hits,
            'neither': neither,
        },
        'verdict': ('VENETIAN_IMPROVES' if ven_dict_hit > merged_hit_rate + 0.005
                    else 'VENETIAN_NEUTRAL' if abs(ven_dict_hit - merged_hit_rate) <= 0.005
                    else 'VENETIAN_DEGRADES'),
        'runtime_seconds': round(elapsed, 1),
    }

    out_path = os.path.join(rd, 'venetian_match.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(output), f, indent=2)
    print(f"\n  Saved → {out_path} ({elapsed:.1f}s)")
