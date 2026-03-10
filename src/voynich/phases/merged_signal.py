"""
Step 38.3 – Merged Signal Isolation
====================================
Classify every token as SIGNAL / SHARED_HIT / SHARED_MISS / ANTI_SIGNAL
using the merged Latin ∪ Italian dictionary.

Dependency chain:
    merged_decode.json         (Step 38.2)
    merged_dict.json           (Step 38.1)
    decode_10k.json            (Step 36.1 — token_decoded, token_folios)
    signal_10k.json            (Step 36.2 — for comparison)
        → merged_signal.json   (this step)
"""

import json
import os
import time
from collections import Counter, defaultdict
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
# Core functions
# ---------------------------------------------------------------------------

def _classify_merged(
    real_hits: List[bool],
    null_hits_list: List[List[bool]],
) -> List[str]:
    """Classify tokens based on merged real vs null hit patterns."""
    n_null = len(null_hits_list)
    classifications = []
    for i in range(len(real_hits)):
        n_null_hits = sum(
            null_hits_list[j][i]
            for j in range(n_null)
            if i < len(null_hits_list[j])
        )
        real_hit = real_hits[i]
        if real_hit and n_null_hits <= 1:
            classifications.append('SIGNAL')
        elif real_hit and n_null_hits >= 3:
            classifications.append('SHARED_HIT')
        elif not real_hit and n_null_hits >= 3:
            classifications.append('ANTI_SIGNAL')
        else:
            classifications.append('SHARED_MISS')
    return classifications


def _per_word_signal_merged(
    decoded_lower: List[str],
    classifications: List[str],
    merged_hits: List[bool],
    null_hits_list: List[List[bool]],
    merged_dict: Set[str],
    latin_10k: Set[str],
    italian_10k: Set[str],
    min_freq: int = 5,
) -> List[Dict]:
    """Compute per-word σ-scores at merged dict with language tags."""
    n_null = len(null_hits_list)
    n_tokens = len(decoded_lower)

    # Count real occurrences
    word_counts = Counter(
        w for w, h in zip(decoded_lower, merged_hits) if h
    )

    # Count null occurrences per word
    null_word_counts: Dict[str, List[int]] = defaultdict(lambda: [0] * n_null)
    for ni, nh in enumerate(null_hits_list):
        for i in range(min(len(nh), n_tokens)):
            if nh[i]:
                w = decoded_lower[i]
                null_word_counts[w][ni] += 1

    results = []
    for word, real_count in word_counts.items():
        if real_count < min_freq:
            continue

        null_counts = null_word_counts.get(word, [0] * n_null)
        null_mean = sum(null_counts) / n_null if n_null > 0 else 0.0
        null_var = (sum((c - null_mean) ** 2 for c in null_counts) / n_null
                    if n_null > 0 else 0.0)
        null_std = null_var ** 0.5
        sigma = ((real_count - null_mean) / null_std if null_std > 0
                 else (10.0 if real_count > null_mean else 0.0))
        selectivity = real_count / null_mean if null_mean > 0 else 10.0

        if sigma > 2.0:
            in_lat = word in latin_10k
            in_ita = word in italian_10k
            if in_lat and in_ita:
                source = 'SHARED'
            elif in_lat:
                source = 'LATIN_ONLY'
            else:
                source = 'ITALIAN_ONLY'

            results.append({
                'word': word,
                'real_count': real_count,
                'null_mean': round(null_mean, 2),
                'sigma': round(sigma, 2),
                'selectivity': round(selectivity, 2),
                'source': source,
                'is_genuine_signal': sigma > 2.0 and real_count >= min_freq,
            })

    results.sort(key=lambda x: x['sigma'], reverse=True)
    return results


def _rank_folios_merged(
    classifications: List[str],
    token_folios: List[str],
) -> List[Dict]:
    """Rank folios by merged SIGNAL rate."""
    folio_data: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {'n_tokens': 0, 'n_signal': 0, 'n_anti': 0}
    )
    for i, cls in enumerate(classifications):
        folio = token_folios[i] if i < len(token_folios) else 'unknown'
        folio_data[folio]['n_tokens'] += 1
        if cls == 'SIGNAL':
            folio_data[folio]['n_signal'] += 1
        elif cls == 'ANTI_SIGNAL':
            folio_data[folio]['n_anti'] += 1

    ranking = []
    for folio, data in folio_data.items():
        n = data['n_tokens']
        if n >= 10:
            ranking.append({
                'folio': folio,
                'n_tokens': n,
                'n_signal': data['n_signal'],
                'signal_rate': round(data['n_signal'] / n, 4),
                'anti_rate': round(data['n_anti'] / n, 4),
                'net_signal': round((data['n_signal'] - data['n_anti']) / n, 4),
            })

    ranking.sort(key=lambda x: x['signal_rate'], reverse=True)
    return ranking


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_merged_signal() -> None:
    """Step 38.3: Merged Signal Isolation."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 38.3: Merged Signal Isolation")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load inputs ──
    print("\n  1. Loading inputs …")
    merged_decode = _safe_load(os.path.join(rd, 'merged_decode.json'))
    merged_dict_data = _safe_load(os.path.join(rd, 'merged_dict.json'))
    decode_data = _safe_load(os.path.join(rd, 'decode_10k.json'))
    signal_10k_data = _safe_load(os.path.join(rd, 'signal_10k.json'))

    merged_hits = merged_decode.get('token_merged_hits', [])
    null_merged = merged_decode.get('null_merged_hits', [])

    merged_dict = set(merged_dict_data.get('merged_words', []))
    latin_10k = set(merged_dict_data.get('latin_10k_words', []))
    italian_10k = set(merged_dict_data.get('italian_10k_words', []))

    token_decoded = decode_data.get('token_decoded', [])
    token_folios = decode_data.get('token_folios', [])
    decoded_lower = [w.lower() for w in token_decoded]
    n_tokens = len(decoded_lower)

    print(f"     {n_tokens} tokens, {len(merged_dict)} merged dict words")

    # ── 2. Classify tokens ──
    print("  2. Classifying tokens …")
    classifications = _classify_merged(merged_hits, null_merged)

    n_signal = sum(1 for c in classifications if c == 'SIGNAL')
    n_shared_hit = sum(1 for c in classifications if c == 'SHARED_HIT')
    n_shared_miss = sum(1 for c in classifications if c == 'SHARED_MISS')
    n_anti = sum(1 for c in classifications if c == 'ANTI_SIGNAL')
    signal_rate = n_signal / n_tokens if n_tokens else 0.0
    anti_rate = n_anti / n_tokens if n_tokens else 0.0
    net_signal = signal_rate - anti_rate

    print(f"     SIGNAL: {n_signal} ({signal_rate:.4f})")
    print(f"     SHARED_HIT: {n_shared_hit}")
    print(f"     SHARED_MISS: {n_shared_miss}")
    print(f"     ANTI_SIGNAL: {n_anti} ({anti_rate:.4f})")
    print(f"     Net signal: {net_signal:.4f}")

    # ── 3. Per-word signal analysis ──
    print("  3. Per-word signal analysis …")
    word_signals = _per_word_signal_merged(
        decoded_lower, classifications, merged_hits, null_merged,
        merged_dict, latin_10k, italian_10k,
    )

    n_genuine = len(word_signals)
    shared_signals = [w for w in word_signals if w['source'] == 'SHARED']
    latin_only_signals = [w for w in word_signals if w['source'] == 'LATIN_ONLY']
    italian_only_signals = [w for w in word_signals if w['source'] == 'ITALIAN_ONLY']

    print(f"     Genuine signal words (σ>2, freq≥5): {n_genuine}")
    print(f"       SHARED: {len(shared_signals)}")
    print(f"       LATIN_ONLY: {len(latin_only_signals)}")
    print(f"       ITALIAN_ONLY: {len(italian_only_signals)}")

    if word_signals:
        print("     Top signal words:")
        for ws in word_signals[:15]:
            print(f"       {ws['word']:<12s} σ={ws['sigma']:>8.2f}  "
                  f"count={ws['real_count']:>5d}  {ws['source']}")

    # ── 4. Signal by language source ──
    print("  4. SIGNAL tokens by language source …")
    signal_sources = Counter()
    for i, cls in enumerate(classifications):
        if cls == 'SIGNAL':
            w = decoded_lower[i]
            in_lat = w in latin_10k
            in_ita = w in italian_10k
            if in_lat and in_ita:
                signal_sources['SHARED'] += 1
            elif in_lat:
                signal_sources['LATIN_ONLY'] += 1
            else:
                signal_sources['ITALIAN_ONLY'] += 1

    total_signal = sum(signal_sources.values())
    signal_by_source = {
        'SHARED': signal_sources.get('SHARED', 0),
        'LATIN_ONLY': signal_sources.get('LATIN_ONLY', 0),
        'ITALIAN_ONLY': signal_sources.get('ITALIAN_ONLY', 0),
        'shared_frac': round(signal_sources.get('SHARED', 0) / total_signal, 4) if total_signal else 0.0,
        'latin_only_frac': round(signal_sources.get('LATIN_ONLY', 0) / total_signal, 4) if total_signal else 0.0,
        'italian_only_frac': round(signal_sources.get('ITALIAN_ONLY', 0) / total_signal, 4) if total_signal else 0.0,
    }

    print(f"     SHARED tokens: {signal_by_source['SHARED']} ({signal_by_source['shared_frac']:.4f})")
    print(f"     LATIN_ONLY tokens: {signal_by_source['LATIN_ONLY']} ({signal_by_source['latin_only_frac']:.4f})")
    print(f"     ITALIAN_ONLY tokens: {signal_by_source['ITALIAN_ONLY']} ({signal_by_source['italian_only_frac']:.4f})")

    # ── 5. Folio ranking ──
    print("  5. Folio ranking …")
    folio_ranking = _rank_folios_merged(classifications, token_folios)

    # Compare to Phase 36
    latin_signal_rate = signal_10k_data.get('signal_rate', 0.0)

    print(f"     Top 5 SIGNAL folios:")
    for fr in folio_ranking[:5]:
        print(f"       {fr['folio']:8s}: {fr['signal_rate']:.4f} "
              f"({fr['n_signal']}/{fr['n_tokens']})")

    print(f"\n  Comparison: Latin 10K signal={latin_signal_rate:.4f}, "
          f"Merged signal={signal_rate:.4f}")

    # ── 6. Save ──
    elapsed = time.time() - t0

    output = {
        'n_tokens': n_tokens,
        'n_signal': n_signal,
        'n_shared_hit': n_shared_hit,
        'n_shared_miss': n_shared_miss,
        'n_anti_signal': n_anti,
        'signal_rate': round(signal_rate, 4),
        'anti_rate': round(anti_rate, 4),
        'net_signal': round(net_signal, 4),
        'token_classifications': classifications,
        'token_decoded': decoded_lower,
        'token_folios': token_folios,
        'word_signals': word_signals,
        'n_genuine_signal_words': n_genuine,
        'n_shared_signal_words': len(shared_signals),
        'n_latin_only_signal_words': len(latin_only_signals),
        'n_italian_only_signal_words': len(italian_only_signals),
        'shared_signal_words': shared_signals,
        'latin_only_signal_words': latin_only_signals,
        'italian_only_signal_words': italian_only_signals,
        'signal_by_source': signal_by_source,
        'folio_ranking': folio_ranking[:30],
        'comparison': {
            'latin_10k_signal_rate': round(latin_signal_rate, 4),
            'merged_signal_rate': round(signal_rate, 4),
            'improvement': round(signal_rate - latin_signal_rate, 4),
        },
        'verdict': (
            f"Merged SIGNAL rate: {signal_rate:.4f} "
            f"(vs Latin 10K: {latin_signal_rate:.4f}). "
            f"{n_genuine} signal words: "
            f"{len(shared_signals)} SHARED, "
            f"{len(latin_only_signals)} LATIN_ONLY, "
            f"{len(italian_only_signals)} ITALIAN_ONLY. "
            f"ITALIAN_ONLY fraction of SIGNAL tokens: "
            f"{signal_by_source['italian_only_frac']:.4f}."
        ),
        'runtime_seconds': round(elapsed, 1),
    }

    out_path = os.path.join(rd, 'merged_signal.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(output), f, indent=2)
    print(f"\n  Saved → {out_path} ({elapsed:.1f}s)")
