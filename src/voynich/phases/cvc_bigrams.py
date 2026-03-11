"""
Step 40.8 – CVC Bigram Test
=============================
Bigram plausibility on CVC-decoded SIGNAL pairs.

Dependency chain:
    cvc_signal.json      (Step 40.7)
    merged_bigrams.json  (Step 38.4)
        → cvc_bigrams.json  (this step)
"""

import json
import os
import random
import time
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


def _edit_distance_1(a: str, b: str) -> bool:
    if abs(len(a) - len(b)) > 1:
        return False
    if a == b:
        return True
    if len(a) == len(b):
        return sum(1 for x, y in zip(a, b) if x != y) == 1
    longer, shorter = (a, b) if len(a) > len(b) else (b, a)
    diffs = 0
    i = j = 0
    while i < len(longer) and j < len(shorter):
        if longer[i] != shorter[j]:
            diffs += 1
            i += 1
        else:
            i += 1
            j += 1
    return diffs + (len(longer) - i) <= 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_cvc_bigrams() -> None:
    """Step 40.8: CVC Bigram Test."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 40.8: CVC Bigram Test")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load inputs ──
    print("\n  1. Loading inputs …")
    cvc_signal = _safe_load(os.path.join(rd, 'cvc_signal.json'))
    merged_bigrams = _safe_load(os.path.join(rd, 'merged_bigrams.json'))
    merged_signal = _safe_load(os.path.join(rd, 'merged_signal.json'))

    cvc_classifications = cvc_signal.get('token_classifications_cvc', [])
    decoded_tokens = merged_signal.get('token_decoded', [])
    token_folios = merged_signal.get('token_folios', [])
    print(f"    CVC classifications: {len(cvc_classifications):,}")

    # ── 2. Build reference bigram set ──
    print("\n  2. Building reference bigram set …")
    # Reuse merged bigram catalog
    ref_bigrams: Set[Tuple[str, str]] = set()
    for bg in merged_bigrams.get('bigram_catalog', []):
        w1, w2 = bg.get('w1', ''), bg.get('w2', '')
        if w1 and w2:
            ref_bigrams.add((w1, w2))
    # Also add from Latin/Italian reference texts if present
    import re
    from voynich.core._paths import data_dir
    for lang_dir in ['latin', 'italian']:
        lang_path = os.path.join(data_dir(), 'reference', lang_dir)
        if os.path.isdir(lang_path):
            for fn in os.listdir(lang_path):
                fpath = os.path.join(lang_path, fn)
                if os.path.isfile(fpath) and fn.endswith('.txt'):
                    with open(fpath) as f:
                        words = re.findall(r'[a-z]+', f.read().lower())
                    for i in range(len(words) - 1):
                        ref_bigrams.add((words[i], words[i + 1]))
    print(f"    Reference bigrams: {len(ref_bigrams):,}")

    # ── 3. Find CVC SIGNAL pairs ──
    print("\n  3. Finding CVC SIGNAL-SIGNAL pairs …")
    n = min(len(decoded_tokens), len(cvc_classifications), len(token_folios))
    signal_pairs = []
    for i in range(n - 1):
        if (cvc_classifications[i] == 'SIGNAL' and
                cvc_classifications[i + 1] == 'SIGNAL' and
                token_folios[i] == token_folios[i + 1]):
            signal_pairs.append({
                'folio': token_folios[i],
                'position': i,
                'w1': decoded_tokens[i],
                'w2': decoded_tokens[i + 1],
            })
    print(f"    CVC SIGNAL-SIGNAL pairs: {len(signal_pairs):,}")

    # ── 4. Bigram hits ──
    print("\n  4. Computing bigram hits …")
    exact_hits = 0
    relaxed_hits = 0
    for pair in signal_pairs:
        w1, w2 = pair['w1'], pair['w2']
        if (w1, w2) in ref_bigrams:
            exact_hits += 1
        else:
            found = False
            for rw1, rw2 in list(ref_bigrams)[:2000]:
                if _edit_distance_1(w1, rw1) and _edit_distance_1(w2, rw2):
                    relaxed_hits += 1
                    found = True
                    break

    real_total = exact_hits + relaxed_hits
    print(f"    Exact: {exact_hits}")
    print(f"    Relaxed (edit ≤ 1): {relaxed_hits}")
    print(f"    Total: {real_total}")

    # ── 5. Null permutation z-test ──
    print("\n  5. Null permutation test …")
    all_signal_words = [decoded_tokens[i] for i in range(n)
                        if cvc_classifications[i] == 'SIGNAL' and decoded_tokens[i]]
    rng = random.Random(42)
    null_hits = []
    for _ in range(500):
        shuffled = list(all_signal_words)
        rng.shuffle(shuffled)
        nh = 0
        for k in range(min(len(signal_pairs), len(shuffled) - 1)):
            if (shuffled[k], shuffled[k + 1]) in ref_bigrams:
                nh += 1
        null_hits.append(nh)

    null_mean = sum(null_hits) / len(null_hits) if null_hits else 0.0
    null_std = (sum((h - null_mean) ** 2 for h in null_hits) /
                len(null_hits)) ** 0.5 if null_hits else 0.001
    z = (real_total - null_mean) / null_std if null_std > 0.001 else 0.0
    print(f"    Null mean: {null_mean:.2f}")
    print(f"    z-score: {z:.2f}")

    # ── 6. Compare ──
    merged_z = merged_bigrams.get('bigram_z', 14.37)
    print(f"\n  6. Comparison:")
    print(f"    Merged z: {merged_z:.2f}")
    print(f"    CVC z: {z:.2f}")
    print(f"    Delta: {z - merged_z:+.2f}")

    # ── 7. Save ──
    elapsed = time.time() - t0

    output = {
        'n_cvc_signal_pairs': len(signal_pairs),
        'n_exact_hits': exact_hits,
        'n_relaxed_hits': relaxed_hits,
        'bigram_z': round(z, 4),
        'merged_z': round(merged_z, 4),
        'delta_z_vs_merged': round(z - merged_z, 4),
        'signal_pairs_sample': signal_pairs[:30],
        'verdict': ('CVC_BIGRAM_IMPROVES' if z > merged_z + 0.5
                    else 'COMPARABLE' if abs(z - merged_z) <= 0.5
                    else 'MERGED_BETTER'),
        'runtime_seconds': round(elapsed, 1),
    }

    out_path = os.path.join(rd, 'cvc_bigrams.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(output), f, indent=2)
    print(f"\n  Saved → {out_path} ({elapsed:.1f}s)")
