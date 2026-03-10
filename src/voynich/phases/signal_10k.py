"""
Step 36.2 – Signal Isolation at 10K
=====================================
Classifies every token as SIGNAL / SHARED_HIT / SHARED_MISS / ANTI_SIGNAL
using the 10K dictionary.  Different tokens will be SIGNAL at 10K vs 131K
because the stricter dictionary changes which null positions produce hits.

Dependency chain:
    decode_10k.json           (Step 36.1)
    signal_isolation.json     (Phase 28.4 — for comparison)
        → signal_10k.json    (this step)
"""

import json
import math
import os
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.phases.dict_calibration import _classify_tokens


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


# ---------------------------------------------------------------------------
# Per-word signal analysis
# ---------------------------------------------------------------------------

def _per_word_signal(
    decoded: List[str],
    classifications: List[str],
    hits_10k: List[bool],
    null_hits_list: List[List[bool]],
    min_freq: int = 5,
) -> List[Dict]:
    """Compute per-word σ-scores at 10K, same formula as signal_isolation.py."""
    # Count real occurrences of each decoded word
    word_counts: Dict[str, int] = Counter()
    word_signal_counts: Dict[str, int] = Counter()
    for i, w in enumerate(decoded):
        if hits_10k[i]:
            word_counts[w] += 1
            if classifications[i] == 'SIGNAL':
                word_signal_counts[w] += 1

    # Count null occurrences
    n_null = len(null_hits_list)
    null_word_counts: Dict[str, List[int]] = defaultdict(lambda: [0] * n_null)
    for ni, nh in enumerate(null_hits_list):
        for i in range(len(nh)):
            if nh[i]:
                w = decoded[i]  # decoded word at this position
                null_word_counts[w][ni] += 1

    results = []
    for word, real_count in word_counts.items():
        if real_count < min_freq:
            continue
        null_counts = null_word_counts.get(word, [0] * n_null)
        null_mean = sum(null_counts) / n_null if n_null > 0 else 0.0
        null_var = sum((c - null_mean) ** 2 for c in null_counts) / n_null if n_null > 0 else 0.0
        null_std = null_var ** 0.5

        sigma = (real_count - null_mean) / null_std if null_std > 0 else (
            10.0 if real_count > null_mean else 0.0
        )
        selectivity = real_count / null_mean if null_mean > 0 else float('inf')

        results.append({
            'word': word,
            'real_count': real_count,
            'null_mean_count': round(null_mean, 2),
            'null_std_count': round(null_std, 2),
            'signal_sigma': round(sigma, 2),
            'selectivity': round(selectivity, 2),
            'signal_token_count': word_signal_counts.get(word, 0),
            'is_genuine_signal': sigma > 2.0,
        })

    results.sort(key=lambda x: x['signal_sigma'], reverse=True)
    return results


def _rank_folios(
    token_folios: List[str],
    classifications: List[str],
) -> List[Dict]:
    """Rank folios by SIGNAL rate at 10K."""
    folio_total: Dict[str, int] = Counter()
    folio_signal: Dict[str, int] = Counter()
    for i, folio in enumerate(token_folios):
        folio_total[folio] += 1
        if classifications[i] == 'SIGNAL':
            folio_signal[folio] += 1

    results = []
    for folio in folio_total:
        n = folio_total[folio]
        s = folio_signal.get(folio, 0)
        results.append({
            'folio': folio,
            'n_tokens': n,
            'n_signal': s,
            'signal_rate': round(s / n, 4) if n > 0 else 0.0,
        })
    results.sort(key=lambda x: x['signal_rate'], reverse=True)
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_signal_10k() -> None:
    """Step 36.2: Signal isolation at 10K dictionary."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 36.2: Signal Isolation at 10K")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load decode_10k.json ──
    print("\n  1. Loading decode_10k.json …")
    with open(os.path.join(rd, 'decode_10k.json')) as f:
        decode_data = json.load(f)

    token_folios = decode_data['token_folios']
    token_evas = decode_data['token_evas']
    token_decoded = decode_data['token_decoded']
    real_hits_10k = decode_data['token_hits_10k']
    null_hits_10k = decode_data['null_hits_10k']  # List of 5 lists
    n_tokens = decode_data['n_tokens']
    print(f"     {n_tokens} tokens loaded")

    # ── 2. Classify tokens at 10K ──
    print("  2. Classifying tokens at 10K …")
    classifications = _classify_tokens(real_hits_10k, null_hits_10k)

    n_signal = classifications.count('SIGNAL')
    n_shared_hit = classifications.count('SHARED_HIT')
    n_shared_miss = classifications.count('SHARED_MISS')
    n_anti = classifications.count('ANTI_SIGNAL')
    signal_rate = n_signal / n_tokens
    anti_rate = n_anti / n_tokens
    net_signal = signal_rate - anti_rate

    print(f"     SIGNAL={n_signal} ({signal_rate:.3f})")
    print(f"     SHARED_HIT={n_shared_hit} ({n_shared_hit/n_tokens:.3f})")
    print(f"     SHARED_MISS={n_shared_miss} ({n_shared_miss/n_tokens:.3f})")
    print(f"     ANTI_SIGNAL={n_anti} ({anti_rate:.3f})")
    print(f"     Net signal={net_signal:.3f}")

    # ── 3. Per-word signal analysis ──
    print("  3. Per-word signal analysis at 10K …")
    word_signals = _per_word_signal(
        token_decoded, classifications, real_hits_10k, null_hits_10k,
    )
    n_genuine = sum(1 for w in word_signals if w['is_genuine_signal'])
    print(f"     {len(word_signals)} words tested (freq ≥ 5)")
    print(f"     {n_genuine} genuine signal words (σ > 2.0)")

    if word_signals:
        print("     Top 10 signal words:")
        for ws in word_signals[:10]:
            marker = "✓" if ws['is_genuine_signal'] else " "
            print(f"       {marker} {ws['word']:<12s} σ={ws['signal_sigma']:>7.2f}"
                  f"  real={ws['real_count']:>4d}  null_μ={ws['null_mean_count']:>6.1f}"
                  f"  sel={ws['selectivity']:>5.1f}")

    # ── 4. Compare to 131K signal words ──
    print("  4. Comparing to 131K signal words …")
    signal_words_10k = set(w['word'] for w in word_signals if w['is_genuine_signal'])

    # Load Phase 28 131K signal words
    signal_words_131k = set()
    iso_path = os.path.join(rd, 'signal_isolation.json')
    if os.path.exists(iso_path):
        with open(iso_path) as f:
            iso_data = json.load(f)
        for ws in iso_data.get('word_signals', []):
            if ws.get('is_genuine_signal'):
                signal_words_131k.add(ws['word'])

    shared = signal_words_10k & signal_words_131k
    new_10k = signal_words_10k - signal_words_131k
    lost_131k = signal_words_131k - signal_words_10k

    print(f"     Shared (both 10K and 131K): {len(shared)} — {sorted(shared)}")
    print(f"     New at 10K only: {len(new_10k)} — {sorted(new_10k)}")
    print(f"     Lost (131K only): {len(lost_131k)} — {sorted(lost_131k)}")

    # ── 5. Folio ranking ──
    print("  5. Ranking folios by SIGNAL rate at 10K …")
    folio_ranking = _rank_folios(token_folios, classifications)
    print("     Top 10 folios:")
    for fr in folio_ranking[:10]:
        print(f"       {fr['folio']:<8s} {fr['n_signal']:>4d}/{fr['n_tokens']:<4d}"
              f" = {fr['signal_rate']:.3f}")

    # ── 6. Anti-signal analysis ──
    print("  6. Anti-signal analysis at 10K …")
    anti_words: Dict[str, int] = Counter()
    for i in range(n_tokens):
        if classifications[i] == 'ANTI_SIGNAL':
            anti_words[token_decoded[i]] += 1
    anti_list = [
        {'word': w, 'count': c}
        for w, c in anti_words.most_common(20)
    ]
    print(f"     {len(anti_words)} unique anti-signal words")
    if anti_list:
        print("     Top anti-signal:")
        for aw in anti_list[:5]:
            print(f"       {aw['word']:<12s} count={aw['count']}")

    # ── 7. Save ──
    elapsed = time.time() - t0

    output = {
        # Per-token parallel arrays
        'token_folios': token_folios,
        'token_evas': token_evas,
        'token_decoded': token_decoded,
        'token_classifications': classifications,
        'token_hits_10k': real_hits_10k,
        # Aggregate
        'n_tokens': n_tokens,
        'n_signal': n_signal,
        'n_shared_hit': n_shared_hit,
        'n_shared_miss': n_shared_miss,
        'n_anti_signal': n_anti,
        'signal_rate': round(signal_rate, 4),
        'anti_rate': round(anti_rate, 4),
        'net_signal': round(net_signal, 4),
        # Per-word
        'word_signals': word_signals,
        'n_genuine_signal_words': n_genuine,
        # Comparison to 131K
        'shared_with_131k': sorted(shared),
        'new_at_10k': sorted(new_10k),
        'lost_from_131k': sorted(lost_131k),
        # Folio ranking
        'top_signal_folios': folio_ranking[:20],
        # Anti-signal
        'anti_signal_words': anti_list,
        'n_unique_anti_words': len(anti_words),
        # Gate
        'gate_passed': n_genuine >= 5,
        'verdict': (
            f"10K signal: {n_genuine} genuine words, rate={signal_rate:.3f}, "
            f"net={net_signal:.3f}"
        ),
        'runtime_seconds': round(elapsed, 1),
    }

    out_path = os.path.join(rd, 'signal_10k.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved → {out_path}")

    # ── Summary ──
    print("\n" + "=" * 70)
    print("SIGNAL 10K SUMMARY")
    print("=" * 70)
    print(f"\n  SIGNAL rate: {signal_rate:.3f} ({n_signal} tokens)")
    print(f"  ANTI rate:   {anti_rate:.3f} ({n_anti} tokens)")
    print(f"  Net signal:  {net_signal:.3f}")
    print(f"  Genuine signal words: {n_genuine}")
    print(f"  Gate: {'PASS' if n_genuine >= 5 else 'FAIL'}")
    print(f"\n  Runtime: {elapsed:.1f}s")
