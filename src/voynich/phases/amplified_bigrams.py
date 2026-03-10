"""
Step 39.16 -- Amplified Bigram Test
======================================
Compute bigram plausibility at calibrated dictionary.  Key test: does
the calibrated dict produce EXACT CC bigrams?

Dependency chain:
    amplified_signal.json      (Step 39.15)
    amplified_dict.json        (Step 39.14)
    merged_dict.json           (Step 38.1)
        -> amplified_bigrams.json  (this step)
"""

import json
import os
import random
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
# Edit distance helper
# ---------------------------------------------------------------------------

def _edit_distance_1(w1: str, w2: str) -> bool:
    """Check if edit distance <= 1."""
    if abs(len(w1) - len(w2)) > 1:
        return False
    if w1 == w2:
        return True
    if len(w1) == len(w2):
        return sum(a != b for a, b in zip(w1, w2)) <= 1
    short, long_ = (w1, w2) if len(w1) < len(w2) else (w2, w1)
    i = j = diffs = 0
    while i < len(short) and j < len(long_):
        if short[i] != long_[j]:
            diffs += 1
            if diffs > 1:
                return False
            j += 1
        else:
            i += 1
            j += 1
    return True


# ---------------------------------------------------------------------------
# Bigram helpers
# ---------------------------------------------------------------------------

def _find_signal_pairs(
    classifications: List[str],
    decoded_lower: List[str],
    token_folios: List[str],
) -> List[Tuple[str, int, str, str]]:
    """Find consecutive SIGNAL-SIGNAL pairs within same folio."""
    pairs = []
    n = len(classifications)
    for i in range(n - 1):
        if i >= len(token_folios) - 1:
            break
        if token_folios[i] != token_folios[i + 1]:
            continue
        if classifications[i] == 'SIGNAL' and classifications[i + 1] == 'SIGNAL':
            pairs.append((token_folios[i], i, decoded_lower[i], decoded_lower[i + 1]))
    return pairs


def _is_content_word(word: str) -> bool:
    """Check if word is a content word (not a function word)."""
    function_words = {
        'de', 'in', 'se', 'ne', 'ad', 'la', 'le', 'di',
        'da', 'si', 'no', 'et', 'a', 'e', 'i', 'o', 'u',
        'cum', 'per', 'pro', 'sub', 'que', 'el', 'fa',
    }
    return word not in function_words and len(word) >= 3


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_amplified_bigrams() -> None:
    """Step 39.16: Amplified Bigram Test."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 39.16: Amplified Bigram Test")
    print("=" * 70)

    rd = _results_dir()

    # -- 1. Load inputs --
    print("\n  1. Loading inputs ...")

    amp_signal = _safe_load(os.path.join(rd, 'amplified_signal.json'))
    amp_dict = _safe_load(os.path.join(rd, 'amplified_dict.json'))
    dict_data = _safe_load(os.path.join(rd, 'merged_dict.json'))
    bigram_38 = _safe_load(os.path.join(rd, 'merged_bigrams.json'))

    classifications = amp_signal.get('token_classifications', [])
    decoded_lower = amp_signal.get('token_decoded', [])
    token_folios = amp_signal.get('token_folios', [])

    calibrated_words: Set[str] = set(amp_dict.get('calibrated_words', []))

    # Build bigram set from merged_dict.json bigram_list, filtered to
    # calibrated words
    bigram_list = dict_data.get('bigram_list', [])
    full_bigram_set: Set[Tuple[str, str]] = set()
    calibrated_bigram_set: Set[Tuple[str, str]] = set()
    for pair in bigram_list:
        if len(pair) == 2:
            full_bigram_set.add((pair[0], pair[1]))
            if pair[0] in calibrated_words and pair[1] in calibrated_words:
                calibrated_bigram_set.add((pair[0], pair[1]))

    n_tokens = len(decoded_lower)
    n_signal = sum(1 for c in classifications if c == 'SIGNAL')

    print(f"     {n_tokens} tokens, {n_signal} SIGNAL")
    print(f"     Calibrated dict: {len(calibrated_words)} words")
    print(f"     Full bigram set: {len(full_bigram_set)}")
    print(f"     Calibrated bigram set: {len(calibrated_bigram_set)}")

    # -- 2. Find SIGNAL-SIGNAL pairs --
    print("\n  2. Finding SIGNAL-SIGNAL pairs ...")

    pairs = _find_signal_pairs(classifications, decoded_lower, token_folios)
    signal_words = set(
        w for w, c in zip(decoded_lower, classifications) if c == 'SIGNAL'
    )

    print(f"     {len(pairs)} SIGNAL-SIGNAL pairs")
    print(f"     {len(signal_words)} unique SIGNAL words")

    # -- 3. Count exact and relaxed bigram hits --
    print("\n  3. Testing bigram matches ...")

    exact_hits = 0
    relaxed_hits = 0
    bigram_catalog: List[Dict] = []

    # Track content-content (CC) bigrams specifically
    n_exact_cc = 0
    n_relaxed_cc = 0

    for folio, pos, w1, w2 in pairs:
        is_exact = (w1, w2) in calibrated_bigram_set
        is_relaxed = False

        if is_exact:
            exact_hits += 1
        else:
            # Check relaxed (edit distance 1) against calibrated bigrams
            for bw1, bw2 in calibrated_bigram_set:
                if _edit_distance_1(w1, bw1) and _edit_distance_1(w2, bw2):
                    is_relaxed = True
                    relaxed_hits += 1
                    break

        if is_exact or is_relaxed:
            w1_content = _is_content_word(w1)
            w2_content = _is_content_word(w2)
            if w1_content and w2_content:
                content_type = 'content-content'
                if is_exact:
                    n_exact_cc += 1
                else:
                    n_relaxed_cc += 1
            elif w1_content or w2_content:
                content_type = 'function-content'
            else:
                content_type = 'function-function'

            bigram_catalog.append({
                'folio': folio,
                'position': pos,
                'w1': w1,
                'w2': w2,
                'match_type': 'exact' if is_exact else 'relaxed',
                'content_type': content_type,
            })

    print(f"     Exact bigram hits: {exact_hits}")
    print(f"     Relaxed bigram hits: {relaxed_hits}")
    print(f"     Exact CC: {n_exact_cc}, Relaxed CC: {n_relaxed_cc}")

    # -- 4. Null permutation for z-score --
    print("\n  4. Null permutation test (500 shuffles) ...")

    rng = random.Random(42)
    signal_list = list(signal_words)
    n_perms = 500

    null_counts = []
    for _ in range(n_perms):
        shuffled = list(signal_list) * 5
        rng.shuffle(shuffled)
        null_exact = sum(
            1 for j in range(len(shuffled) - 1)
            if (shuffled[j], shuffled[j + 1]) in calibrated_bigram_set
        )
        null_counts.append(null_exact)

    null_mean = sum(null_counts) / len(null_counts) if null_counts else 0.0
    null_var = (sum((c - null_mean) ** 2 for c in null_counts) / len(null_counts)
                if null_counts else 0.0)
    null_std = null_var ** 0.5
    bigram_z = ((exact_hits - null_mean) / null_std
                if null_std > 0
                else (10.0 if exact_hits > null_mean else 0.0))

    print(f"     Null: mean={null_mean:.2f}, std={null_std:.2f}")
    print(f"     z-score: {bigram_z:.2f}")

    # -- 5. Compare vs Phase 38 baseline --
    print("\n  5. Comparison vs Phase 38 merged bigrams ...")

    p38_z = bigram_38.get('bigram_z', 0.0)
    p38_exact = bigram_38.get('exact_hits', 0)
    p38_relaxed = bigram_38.get('relaxed_hits', 0)
    p38_cc = bigram_38.get('n_content_content', 0)

    delta_z = bigram_z - p38_z
    delta_exact = exact_hits - p38_exact

    print(f"     Phase 38: z={p38_z:.2f}, exact={p38_exact}, CC={p38_cc}")
    print(f"     Amplified: z={bigram_z:.2f}, exact={exact_hits}, "
          f"CC={n_exact_cc}")
    print(f"     Delta z: {delta_z:+.2f}, delta exact: {delta_exact:+d}")

    # -- 6. Bigram catalog (top entries) --
    print("\n  6. Bigram catalog:")
    for entry in bigram_catalog[:15]:
        print(f"     [{entry['match_type']:7s}] {entry['w1']:>8s} "
              f"{entry['w2']:<8s}  {entry['content_type']:18s}  "
              f"{entry['folio']}")

    # -- 7. Verdict --
    if bigram_z >= 5.0 and n_exact_cc >= 3:
        verdict_str = (f"STRONG_BIGRAM_SIGNAL: z={bigram_z:.2f}, "
                       f"{n_exact_cc} exact CC bigrams")
    elif bigram_z >= 3.0:
        verdict_str = (f"MODERATE_BIGRAM_SIGNAL: z={bigram_z:.2f}, "
                       f"{n_exact_cc} exact CC")
    elif bigram_z >= 2.0:
        verdict_str = (f"WEAK_BIGRAM_SIGNAL: z={bigram_z:.2f}, "
                       f"{n_exact_cc} exact CC")
    else:
        verdict_str = (f"NO_BIGRAM_SIGNAL: z={bigram_z:.2f}, "
                       f"{n_exact_cc} exact CC")

    elapsed = time.time() - t0

    output = {
        'n_signal_pairs': len(pairs),
        'exact_hits': exact_hits,
        'relaxed_hits': relaxed_hits,
        'bigram_z': round(bigram_z, 2),
        'n_exact_cc': n_exact_cc,
        'n_relaxed_cc': n_relaxed_cc,
        'null_mean': round(null_mean, 2),
        'null_std': round(null_std, 2),
        'bigram_catalog': bigram_catalog[:50],
        'delta_vs_phase38': {
            'phase38_z': round(p38_z, 2),
            'amplified_z': round(bigram_z, 2),
            'z_delta': round(delta_z, 2),
            'phase38_exact': p38_exact,
            'amplified_exact': exact_hits,
            'exact_delta': delta_exact,
            'phase38_cc': p38_cc,
            'amplified_cc': n_exact_cc,
        },
        'verdict': verdict_str,
        'runtime_seconds': round(elapsed, 1),
    }

    out_path = os.path.join(rd, 'amplified_bigrams.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(output), f, indent=2)
    print(f"\n  Saved -> {out_path} ({elapsed:.1f}s)")
