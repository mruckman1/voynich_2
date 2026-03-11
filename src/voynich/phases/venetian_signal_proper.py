"""
Step 41.3 – Proper Venetian Signal Isolation
=============================================
With proper null data from Step 41.1, produce the definitive Venetian
signal classification for every token and per-word σ-scores.

Dependency chain:
    null_venetian_decode.json    (Step 41.1 — null decoded tokens)
    venetian_match.json          (Step 40.2 — real decoded tokens)
    venetian_forms.json          (Step 40.1 — Venetian word set)
    merged_signal.json           (Step 38.3 — merged classifications)
        → venetian_signal_proper.json  (this step)
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
# Classification
# ---------------------------------------------------------------------------

def _classify_tokens_4class(
    decoded_tokens: List[str],
    real_word_set: Set[str],
    null_decoded_lists: List[List[str]],
    null_word_set: Set[str],
) -> Tuple[List[str], Dict]:
    """Classify tokens: SIGNAL / SHARED_HIT / SHARED_MISS / ANTI_SIGNAL."""
    n = len(decoded_tokens)
    classifications = []
    counts: Counter = Counter()

    for i in range(n):
        word = decoded_tokens[i]
        real_hit = word in real_word_set if word else False

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
    null_decoded_lists: List[List[str]],
    venetian_set: Set[str],
    min_freq: int = 5,
) -> List[Dict]:
    """Compute per-word σ-scores with proper null baseline."""
    n_tokens = len(decoded_tokens)

    # Count real occurrences of Venetian-matching words
    word_real_counts: Counter = Counter()
    for word in decoded_tokens:
        if word and word in venetian_set:
            word_real_counts[word] += 1

    # Count null occurrences
    word_null_counts: Dict[str, List[int]] = {}
    for word in word_real_counts:
        word_null_counts[word] = []

    for null_decoded in null_decoded_lists:
        null_hits: Counter = Counter()
        for nw in null_decoded:
            if nw and nw in venetian_set:
                null_hits[nw] += 1
        for word in word_null_counts:
            word_null_counts[word].append(null_hits.get(word, 0))

    # Compute σ-scores
    results = []
    for word in sorted(word_real_counts.keys(),
                       key=lambda w: -word_real_counts[w]):
        real_count = word_real_counts[word]
        if real_count < min_freq:
            continue

        null_vals = word_null_counts.get(word, [0] * len(null_decoded_lists))
        null_mean = sum(null_vals) / len(null_vals) if null_vals else 0.0
        null_std = (sum((v - null_mean) ** 2 for v in null_vals)
                    / len(null_vals)) ** 0.5 if null_vals else 0.0

        sigma = ((real_count - null_mean) / null_std if null_std > 0.01 else
                 (999.0 if real_count > null_mean else 0.0))
        selectivity = (real_count / null_mean if null_mean > 0.01
                       else 999.0)

        real_rate = real_count / n_tokens
        null_rate = null_mean / n_tokens

        results.append({
            'word': word,
            'real_count': real_count,
            'null_mean': round(null_mean, 2),
            'null_std': round(null_std, 2),
            'sigma': round(sigma, 2) if sigma < 990 else 999.0,
            'selectivity': round(selectivity, 4) if selectivity < 990 else 999.0,
            'real_rate': round(real_rate, 6),
            'null_rate': round(null_rate, 6),
            'is_genuine_signal': sigma > 2.0,
        })

    return results


def _compare_to_merged(
    ven_word_signals: List[Dict],
    merged_signal_data: Dict,
) -> Dict:
    """Compare Venetian signal words to merged signal words."""
    # Get merged signal words
    merged_words = set()
    for entry in merged_signal_data.get('genuine_signal_words', []):
        if isinstance(entry, dict):
            merged_words.add(entry.get('word', ''))
        elif isinstance(entry, str):
            merged_words.add(entry)

    ven_words = set(w['word'] for w in ven_word_signals if w['is_genuine_signal'])

    shared = ven_words & merged_words
    ven_only = ven_words - merged_words
    merged_only = merged_words - ven_words

    return {
        'n_venetian_signal': len(ven_words),
        'n_merged_signal': len(merged_words),
        'n_shared': len(shared),
        'n_venetian_only': len(ven_only),
        'n_merged_only': len(merged_only),
        'shared_words': sorted(shared),
        'venetian_only_words': sorted(ven_only),
        'merged_only_words': sorted(merged_only),
    }


def _rank_folios(
    decoded_tokens: List[str],
    classifications: List[str],
    token_folios: List[str],
) -> List[Dict]:
    """Rank folios by SIGNAL rate."""
    folio_counts: Dict[str, Dict[str, int]] = {}

    for i in range(len(decoded_tokens)):
        if i >= len(classifications) or i >= len(token_folios):
            break
        folio = token_folios[i]
        if folio not in folio_counts:
            folio_counts[folio] = {'total': 0, 'signal': 0}
        folio_counts[folio]['total'] += 1
        if classifications[i] == 'SIGNAL':
            folio_counts[folio]['signal'] += 1

    ranked = []
    for folio, counts in folio_counts.items():
        if counts['total'] >= 10:
            rate = counts['signal'] / counts['total']
            ranked.append({
                'folio': folio,
                'n_tokens': counts['total'],
                'n_signal': counts['signal'],
                'signal_rate': round(rate, 4),
            })

    ranked.sort(key=lambda x: -x['signal_rate'])
    return ranked


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_venetian_signal_proper() -> None:
    """Step 41.3: Proper Venetian signal isolation with valid null baseline."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 41.3: Proper Venetian Signal Isolation")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load inputs ──
    print("\n  1. Loading inputs …")

    null_ven = _safe_load(os.path.join(rd, 'null_venetian_decode.json'))
    if not null_ven:
        print("  [SKIP] null_venetian_decode.json not found")
        return

    ven_match = _safe_load(os.path.join(rd, 'venetian_match.json'))
    decoded_tokens = ven_match.get('token_decoded', [])
    token_folios = ven_match.get('token_folios', [])

    ven_forms = _safe_load(os.path.join(rd, 'venetian_forms.json'))
    venetian_set = set(ven_forms.get('venetian_extended_set', []))

    merged_signal = _safe_load(os.path.join(rd, 'merged_signal.json'))

    # Load null decoded tokens from Step 41.1
    null_decoded_lists = null_ven.get('null_decoded_tokens', [])

    n_tokens = len(decoded_tokens)
    print(f"    Decoded tokens: {n_tokens:,}")
    print(f"    Venetian set: {len(venetian_set):,}")
    print(f"    Null corpora: {len(null_decoded_lists)}")

    # ── 2. Classify tokens ──
    print("\n  2. Classifying tokens (4-class with proper null) …")
    classifications, class_counts = _classify_tokens_4class(
        decoded_tokens, venetian_set, null_decoded_lists, venetian_set,
    )
    for cls, count in sorted(class_counts.items()):
        rate = count / n_tokens if n_tokens > 0 else 0.0
        print(f"    {cls}: {count:,} ({rate:.4f})")

    signal_rate = class_counts.get('SIGNAL', 0) / n_tokens if n_tokens > 0 else 0.0

    # ── 3. Per-word signal analysis ──
    print("\n  3. Computing per-word σ-scores …")
    word_signals = _compute_per_word_signal(
        decoded_tokens, null_decoded_lists, venetian_set, min_freq=5,
    )
    n_genuine = sum(1 for w in word_signals if w['is_genuine_signal'])
    print(f"    Words tested (freq≥5): {len(word_signals)}")
    print(f"    Genuine signal (σ>2): {n_genuine}")

    # Top 10
    print("\n    Top 10 signal words:")
    for w in word_signals[:10]:
        print(f"      {w['word']:10s} σ={w['sigma']:8.2f}  "
              f"real={w['real_count']:5d}  null={w['null_mean']:6.1f}  "
              f"sel={w['selectivity']:.2f}×")

    # ── 4. Compare to merged signal ──
    print("\n  4. Comparing to merged signal list …")
    comparison = _compare_to_merged(word_signals, merged_signal)
    print(f"    Venetian signal words: {comparison['n_venetian_signal']}")
    print(f"    Merged signal words: {comparison['n_merged_signal']}")
    print(f"    Shared: {comparison['n_shared']}")
    print(f"    Venetian-only: {comparison['n_venetian_only']}")
    print(f"    Merged-only: {comparison['n_merged_only']}")

    # ── 5. Folio ranking ──
    print("\n  5. Ranking folios by SIGNAL rate …")
    folio_ranking = _rank_folios(decoded_tokens, classifications, token_folios)
    print(f"    Top 5 folios:")
    for f in folio_ranking[:5]:
        print(f"      {f['folio']:8s}  {f['n_signal']}/{f['n_tokens']}  "
              f"= {f['signal_rate']:.4f}")

    # ── 6. Comparison table ──
    print("\n  6. Validated comparison table:")

    # Get merged stats
    merged_signal_rate = merged_signal.get('signal_rate', 0.0)
    merged_n_signal = merged_signal.get('n_signal_words', 0)

    print(f"    | Dictionary  | SIGNAL rate | Signal words (σ>2) |")
    print(f"    |-------------|-------------|---------------------|")
    print(f"    | Merged L+I  | {merged_signal_rate:.4f}      | {merged_n_signal:19d} |")
    print(f"    | Venetian    | {signal_rate:.4f}      | {n_genuine:19d} |")

    # ── 7. Save ──
    elapsed = time.time() - t0

    output = {
        'n_tokens': n_tokens,
        'class_counts': class_counts,
        'signal_rate': round(signal_rate, 6),
        'n_genuine_signal_words': n_genuine,
        'word_signals': word_signals,
        'comparison_to_merged': comparison,
        'top_folios': folio_ranking[:20],
        'token_classifications_venetian_proper': classifications,
        'runtime_seconds': round(elapsed, 1),
    }

    out_path = os.path.join(rd, 'venetian_signal_proper.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(output), f, indent=2)
    print(f"\n  Saved → {out_path} ({elapsed:.1f}s)")
