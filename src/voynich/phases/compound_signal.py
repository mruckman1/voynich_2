"""
Phase 32.2 – Compound-Sign Signal Isolation
=============================================
Re-classify every token as SIGNAL / SHARED_HIT / SHARED_MISS /
ANTI_SIGNAL under the compound-sign decode and measure whether the
SIGNAL rate increases from Phase 28's 16.5%.

Dependency chain:
    compound_decode.json       (Step 32.1)
    signal_bigrams.json        (Phase 29 baseline)
    signal_isolation.json      (Phase 28 baseline)
        → compound_signal.json  (this step)
"""

import json
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from voynich.core._paths import results_dir as _results_dir


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
    if isinstance(obj, float) and (obj != obj):
        return None
    if isinstance(obj, (bool, int, float, str, type(None))):
        return obj
    return str(obj)


def _classify_tokens(
    real_hits: List[bool],
    null_hits_list: List[List[bool]],
    n_tokens: int,
) -> List[str]:
    """Classify each token as SIGNAL/SHARED_HIT/SHARED_MISS/ANTI_SIGNAL."""
    classifications: List[str] = []
    for idx in range(n_tokens):
        r_hit = real_hits[idx]
        null_hit_count = sum(1 for nh in null_hits_list if nh[idx])

        if r_hit and null_hit_count <= 1:
            classifications.append('SIGNAL')
        elif r_hit and null_hit_count >= 3:
            classifications.append('SHARED_HIT')
        elif not r_hit and null_hit_count >= 3:
            classifications.append('ANTI_SIGNAL')
        else:
            classifications.append('SHARED_MISS')
    return classifications


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_compound_signal() -> None:
    """Step 32.2: Compound-sign signal re-classification."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE 32.2: Compound-Sign Signal Isolation")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load compound decode ──
    print("\n  1. Loading compound_decode.json ...")
    with open(os.path.join(rd, 'compound_decode.json')) as f:
        cd = json.load(f)

    real_hits = cd['token_dict_hits']
    null_hits_list = cd['null_token_hits']
    n_tokens = cd['n_tokens']
    decoded = cd['token_decoded']
    folios = cd['token_folios']

    # ── 2. Classify tokens ──
    print("\n  2. Classifying tokens ...")
    classifications = _classify_tokens(real_hits, null_hits_list, n_tokens)

    counts = Counter(classifications)
    n_signal = counts.get('SIGNAL', 0)
    n_shared_hit = counts.get('SHARED_HIT', 0)
    n_shared_miss = counts.get('SHARED_MISS', 0)
    n_anti = counts.get('ANTI_SIGNAL', 0)
    signal_rate = n_signal / n_tokens

    print(f"     SIGNAL:      {n_signal:6d} ({n_signal / n_tokens:.1%})")
    print(f"     SHARED_HIT:  {n_shared_hit:6d} ({n_shared_hit / n_tokens:.1%})")
    print(f"     SHARED_MISS: {n_shared_miss:6d} ({n_shared_miss / n_tokens:.1%})")
    print(f"     ANTI_SIGNAL: {n_anti:6d} ({n_anti / n_tokens:.1%})")

    # ── 3. Load Phase 29 baseline classifications ──
    print("\n  3. Loading Phase 29 baseline ...")
    phase29_path = os.path.join(rd, 'signal_bigrams.json')
    old_classifications: Optional[List[str]] = None
    phase29_signal_rate = 0.0
    phase29_n_signal = 0

    if os.path.exists(phase29_path):
        with open(phase29_path) as f:
            sb = json.load(f)
        old_classifications = sb.get('token_classifications')
        phase29_n_signal = sb.get('n_signal', 0)
        phase29_signal_rate = sb.get('signal_rate', 0.0)
        print(f"     Phase 29 SIGNAL: {phase29_n_signal} ({phase29_signal_rate:.1%})")
    else:
        print("     [SKIP] signal_bigrams.json not found")

    # ── 4. Migration analysis ──
    print("\n  4. Migration analysis ...")
    migration: Dict[str, Dict[str, int]] = {}
    n_gained = 0
    n_lost = 0
    n_shared_to_signal = 0

    if old_classifications and len(old_classifications) == n_tokens:
        for old_cls, new_cls in zip(old_classifications, classifications):
            migration.setdefault(old_cls, Counter())
            migration[old_cls][new_cls] += 1

        n_gained = migration.get('SHARED_MISS', {}).get('SIGNAL', 0)
        n_lost = migration.get('SIGNAL', {}).get('SHARED_MISS', 0)
        n_shared_to_signal = migration.get('SHARED_HIT', {}).get('SIGNAL', 0)

        print(f"     SHARED_MISS → SIGNAL:  {n_gained}")
        print(f"     SHARED_HIT  → SIGNAL:  {n_shared_to_signal}")
        print(f"     SIGNAL → SHARED_MISS:  {n_lost}")
    else:
        print("     [SKIP] Cannot compute migration (no baseline or size mismatch)")

    # ── 5. Per-word signal analysis ──
    print("\n  5. Per-word signal analysis ...")
    # Count each decoded word in SIGNAL vs total
    word_real_counts: Counter = Counter()
    word_null_counts: Dict[str, List[int]] = {}

    for i in range(n_tokens):
        if real_hits[i]:
            word_real_counts[decoded[i]] += 1

    # Count words in null corpora decoded through compound pipeline
    # We don't have null decoded words, only null hits.
    # Approximate: count how many positions decode to each word in real,
    # and how many of those positions are null hits.
    word_null_hit_counts: Dict[str, List[int]] = {}
    for word in word_real_counts:
        word_null_hit_counts[word] = [0] * len(null_hits_list)

    for i in range(n_tokens):
        if real_hits[i]:
            w = decoded[i]
            for s_idx, nh in enumerate(null_hits_list):
                if nh[i]:
                    word_null_hit_counts[w][s_idx] += 1

    word_signals: List[Dict] = []
    for word, real_count in word_real_counts.most_common(50):
        null_counts = word_null_hit_counts.get(word, [0] * len(null_hits_list))
        null_mean = sum(null_counts) / len(null_counts)
        null_var = sum((c - null_mean) ** 2 for c in null_counts) / len(null_counts)
        null_std = null_var ** 0.5

        sigma = ((real_count - null_mean) / null_std) if null_std > 0 else (
            float('inf') if real_count > null_mean else 0.0
        )
        real_rate = real_count / n_tokens
        null_rate = null_mean / n_tokens
        selectivity = real_rate / null_rate if null_rate > 0 else float('inf')

        word_signals.append({
            'word': word,
            'real_count': real_count,
            'null_mean': round(null_mean, 2),
            'null_std': round(null_std, 2),
            'sigma': round(sigma, 2) if sigma != float('inf') else 999.0,
            'selectivity': round(selectivity, 2) if selectivity != float('inf') else 999.0,
            'is_genuine': sigma > 2.0,
        })

    word_signals.sort(key=lambda x: -x['sigma'])
    n_genuine = sum(1 for ws in word_signals if ws['is_genuine'])

    print(f"     Genuine signal words (sigma > 2.0): {n_genuine}")
    for ws in word_signals[:10]:
        flag = "SIGNAL" if ws['is_genuine'] else "shared"
        print(f"       {ws['word']:12s} real={ws['real_count']:5d} "
              f"null={ws['null_mean']:6.1f} sigma={ws['sigma']:6.1f} [{flag}]")

    # ── 6. Folio-level signal rates ──
    print("\n  6. Folio-level signal rates ...")
    folio_total: Counter = Counter()
    folio_signal: Counter = Counter()
    for i in range(n_tokens):
        folio_total[folios[i]] += 1
        if classifications[i] == 'SIGNAL':
            folio_signal[folios[i]] += 1

    top_folios = []
    for folio in sorted(folio_total.keys()):
        n_tok = folio_total[folio]
        n_sig = folio_signal.get(folio, 0)
        rate = n_sig / n_tok if n_tok > 0 else 0.0
        top_folios.append({'folio': folio, 'n_tokens': n_tok,
                           'n_signal': n_sig, 'signal_rate': round(rate, 4)})

    top_folios.sort(key=lambda x: -x['signal_rate'])
    for tf in top_folios[:5]:
        print(f"     {tf['folio']:8s}  SIGNAL={tf['n_signal']:3d}/{tf['n_tokens']:3d} "
              f"({tf['signal_rate']:.1%})")

    # ── 7. Save output ──
    print("\n  7. Saving compound_signal.json ...")
    output = {
        'n_signal': n_signal,
        'n_shared_hit': n_shared_hit,
        'n_shared_miss': n_shared_miss,
        'n_anti_signal': n_anti,
        'signal_rate': round(signal_rate, 6),
        'n_tokens': n_tokens,
        'migration_matrix': {k: dict(v) for k, v in migration.items()},
        'n_gained_signal': n_gained,
        'n_lost_signal': n_lost,
        'n_shared_to_signal': n_shared_to_signal,
        'word_signals': word_signals,
        'n_genuine_signals': n_genuine,
        'phase29_signal_rate': phase29_signal_rate,
        'phase29_n_signal': phase29_n_signal,
        'delta_signal_rate': round(signal_rate - phase29_signal_rate, 6),
        'delta_n_signal': n_signal - phase29_n_signal,
        'top_signal_folios': top_folios[:20],
        'token_classifications': classifications,
        'gate_passed': signal_rate > phase29_signal_rate,
        'verdict': (
            f"SIGNAL rate {signal_rate:.1%} "
            f"({'UP' if signal_rate > phase29_signal_rate else 'DOWN'} "
            f"from {phase29_signal_rate:.1%})"
        ),
        'runtime_seconds': round(time.time() - t0, 1),
    }

    with open(os.path.join(rd, 'compound_signal.json'), 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n  Signal rate: {signal_rate:.1%} (Phase 29: {phase29_signal_rate:.1%})")
    print(f"  Genuine signal words: {n_genuine}")
    print(f"\n  Step 32.2 completed in {time.time() - t0:.1f}s")
