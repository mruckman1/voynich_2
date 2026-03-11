"""
Step 41.2 – Validated Venetian Z-Score
=======================================
Recompute the Venetian bigram z-score with a PROPER null baseline.

Phase 40's z=319.76 was inflated because _bigram_z_test() compared
(exact + relaxed) real hits against exact-only null permutation hits.
This step fixes that by making the null permutation also count relaxed
hits, producing an apples-to-apples comparison.

Dependency chain:
    null_venetian_decode.json  (Step 41.1 — null corpora decoded)
    venetian_match.json        (Step 40.2 — decoded tokens + classifications)
    venetian_bigrams.json      (Step 40.3 — original z for comparison)
        → venetian_validated.json  (this step)
"""

import json
import os
import random
import re
import time
from collections import Counter
from typing import Any, Dict, List, Set, Tuple

from voynich.core._paths import (
    data_dir as _data_dir,
    results_dir as _results_dir,
)


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
    """Check if two words are within edit distance 1."""
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
# Venetian bigram reference (same as venetian_bigrams.py)
# ---------------------------------------------------------------------------

def _build_venetian_reference_bigrams(
    latin_text: str,
    anonimo_text: str,
) -> Set[Tuple[str, str]]:
    """Build a bigram set from Venetian-transformed Latin text + Anonimo."""
    from voynich.core.reference import apply_venetian_sound_changes

    bigrams: Set[Tuple[str, str]] = set()

    latin_words = re.findall(r'[a-z]+', latin_text.lower())
    venetian_words = []
    for w in latin_words:
        variants = apply_venetian_sound_changes(w)
        if variants:
            venetian_words.append(next(iter(variants)))
        else:
            venetian_words.append(w)
    for i in range(len(venetian_words) - 1):
        bigrams.add((venetian_words[i], venetian_words[i + 1]))

    anonimo_words = re.findall(
        r'[a-zA-ZàèéìòùÀÈÉÌÒÙ]+', anonimo_text.lower(),
    )
    for i in range(len(anonimo_words) - 1):
        bigrams.add((anonimo_words[i], anonimo_words[i + 1]))

    return bigrams


def _build_word_index(
    reference_bigrams: Set[Tuple[str, str]],
) -> Dict[str, Set[str]]:
    """Build word→partner index for fast bigram lookup."""
    index: Dict[str, Set[str]] = {}
    for w1, w2 in reference_bigrams:
        if w1 not in index:
            index[w1] = set()
        index[w1].add(w2)
    return index


# ---------------------------------------------------------------------------
# Fixed bigram z-test
# ---------------------------------------------------------------------------

def _precompute_partner_sets(
    unique_signal_words: Set[str],
    ref_words: Set[str],
) -> Dict[str, Set[str]]:
    """For each signal word, find all reference words within edit distance 1.

    This is the key precomputation that makes the null permutation test
    fast enough to include relaxed hits.
    """
    partners: Dict[str, Set[str]] = {}
    for sw in unique_signal_words:
        if not sw:
            continue
        p = set()
        for rw in ref_words:
            if abs(len(rw) - len(sw)) <= 1 and _edit_distance_1(sw, rw):
                p.add(rw)
        partners[sw] = p
    return partners


def _check_relaxed_pair(
    w1: str,
    w2: str,
    ref_bigrams: Set[Tuple[str, str]],
    word_index: Dict[str, Set[str]],
    partners: Dict[str, Set[str]],
) -> bool:
    """Check if (w1, w2) matches any reference bigram within edit distance 1."""
    # Exact
    if (w1, w2) in ref_bigrams:
        return True

    w1_partners = partners.get(w1, set())
    w2_partners = partners.get(w2, set())

    # (w1, p2) in ref_bigrams?
    if w1 in word_index:
        for rp in word_index[w1]:
            if rp in w2_partners or rp == w2:
                return True

    # (p1, w2) in ref_bigrams?
    for p1 in w1_partners:
        if p1 in word_index:
            for rp in word_index[p1]:
                if rp == w2 or rp in w2_partners:
                    return True

    return False


def _bigram_z_test_fixed(
    signal_pairs: List[Dict],
    reference_bigrams: Set[Tuple[str, str]],
    all_signal_words: List[str],
    n_permutations: int = 500,
) -> Dict:
    """Fixed bigram z-test: null permutation also counts relaxed hits.

    The Phase 40 bug: real counted exact+relaxed, null counted exact-only.
    Fix: both count the same (exact+relaxed via precomputed partner sets).

    Returns dict with z_exact, z_relaxed, z_total, and supporting stats.
    """
    # Build indexes
    word_index = _build_word_index(reference_bigrams)
    ref_words = set()
    for w1, w2 in reference_bigrams:
        ref_words.add(w1)
        ref_words.add(w2)

    # Precompute partner sets for all unique signal words
    unique_signal = set(w for w in all_signal_words if w)
    print(f"      Precomputing partners for {len(unique_signal)} "
          f"signal words against {len(ref_words)} ref words …")
    partners = _precompute_partner_sets(unique_signal, ref_words)
    print(f"      Done. Mean partners per word: "
          f"{sum(len(v) for v in partners.values()) / max(len(partners), 1):.1f}")

    # Count real hits (exact and relaxed separately)
    real_exact = 0
    real_relaxed = 0
    for pair in signal_pairs:
        w1, w2 = pair['w1'], pair['w2']
        if (w1, w2) in reference_bigrams:
            real_exact += 1
        elif _check_relaxed_pair(w1, w2, reference_bigrams,
                                 word_index, partners):
            real_relaxed += 1

    real_total = real_exact + real_relaxed
    print(f"      Real: exact={real_exact}, relaxed={real_relaxed}, "
          f"total={real_total}")

    # Null permutation test — FIXED: counts both exact AND relaxed
    rng = random.Random(42)
    null_exact_list = []
    null_relaxed_list = []
    null_total_list = []

    n_pairs = len(signal_pairs)
    for perm_i in range(n_permutations):
        if (perm_i + 1) % 100 == 0:
            print(f"      Permutation {perm_i + 1}/{n_permutations} …")

        shuffled = list(all_signal_words)
        rng.shuffle(shuffled)

        perm_exact = 0
        perm_relaxed = 0
        for k in range(min(n_pairs, len(shuffled) - 1)):
            w1 = shuffled[k]
            w2 = shuffled[k + 1]
            if not w1 or not w2:
                continue
            if (w1, w2) in reference_bigrams:
                perm_exact += 1
            elif _check_relaxed_pair(w1, w2, reference_bigrams,
                                     word_index, partners):
                perm_relaxed += 1

        null_exact_list.append(perm_exact)
        null_relaxed_list.append(perm_relaxed)
        null_total_list.append(perm_exact + perm_relaxed)

    # Compute z-scores
    def _z(real_val, null_vals):
        n_mean = sum(null_vals) / len(null_vals) if null_vals else 0.0
        n_std = (sum((v - n_mean) ** 2 for v in null_vals)
                 / len(null_vals)) ** 0.5 if null_vals else 0.001
        z = (real_val - n_mean) / n_std if n_std > 0.001 else 0.0
        return z, n_mean, n_std

    z_exact, null_exact_mean, null_exact_std = _z(real_exact, null_exact_list)
    z_relaxed, null_relaxed_mean, null_relaxed_std = _z(
        real_relaxed, null_relaxed_list,
    )
    z_total, null_total_mean, null_total_std = _z(real_total, null_total_list)

    return {
        'real_exact': real_exact,
        'real_relaxed': real_relaxed,
        'real_total': real_total,
        'null_exact_mean': round(null_exact_mean, 2),
        'null_exact_std': round(null_exact_std, 2),
        'null_relaxed_mean': round(null_relaxed_mean, 2),
        'null_relaxed_std': round(null_relaxed_std, 2),
        'null_total_mean': round(null_total_mean, 2),
        'null_total_std': round(null_total_std, 2),
        'z_exact': round(z_exact, 4),
        'z_relaxed': round(z_relaxed, 4),
        'z_total': round(z_total, 4),
        'n_permutations': n_permutations,
    }


def _find_signal_pairs(
    decoded_tokens: List[str],
    classifications: List[str],
    folios: List[str],
) -> List[Dict]:
    """Find consecutive SIGNAL-SIGNAL pairs on same folio."""
    pairs = []
    n = len(decoded_tokens)
    for i in range(n - 1):
        if (i < len(classifications) and i + 1 < len(classifications) and
                classifications[i] == 'SIGNAL' and
                classifications[i + 1] == 'SIGNAL' and
                i < len(folios) and i + 1 < len(folios) and
                folios[i] == folios[i + 1]):
            pairs.append({
                'folio': folios[i],
                'position': i,
                'w1': decoded_tokens[i],
                'w2': decoded_tokens[i + 1],
            })
    return pairs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_venetian_validated() -> None:
    """Step 41.2: Recompute Venetian bigram z with proper null baseline."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 41.2: Validated Venetian Z-Score")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load inputs ──
    print("\n  1. Loading inputs …")

    # Step 41.1 output
    null_ven = _safe_load(os.path.join(rd, 'null_venetian_decode.json'))
    if not null_ven:
        print("  [SKIP] null_venetian_decode.json not found — run null-ven first")
        return

    # Phase 40.2 — decoded tokens and classifications
    ven_match = _safe_load(os.path.join(rd, 'venetian_match.json'))
    decoded_tokens = ven_match.get('token_decoded', [])
    ven_classifications = ven_match.get('token_classifications_venetian', [])
    token_folios = ven_match.get('token_folios', [])
    print(f"    Decoded tokens: {len(decoded_tokens):,}")
    print(f"    Classifications: {len(ven_classifications):,}")

    # Phase 40.3 — original z for comparison
    orig_bigrams = _safe_load(os.path.join(rd, 'venetian_bigrams.json'))
    original_z = orig_bigrams.get('bigram_z', 319.76)
    print(f"    Original (buggy) z: {original_z:.2f}")

    # Selectivity from Step 41.1
    selectivity = null_ven.get('selectivity', 0.0)
    real_hit_rate = null_ven.get('real_venetian_hit_rate', 0.0)
    null_mean_rate = null_ven.get('null_mean_venetian_hit_rate', 0.0)
    print(f"    Venetian selectivity (from 41.1): {selectivity:.2f}×")
    print(f"    Real hit rate: {real_hit_rate:.4f}, Null mean: {null_mean_rate:.4f}")

    # ── 2. Build Venetian reference bigrams ──
    print("\n  2. Building Venetian reference bigram set …")
    latin_dir = os.path.join(_data_dir(), 'reference', 'latin')
    latin_text = ''
    if os.path.isdir(latin_dir):
        for fn in sorted(os.listdir(latin_dir)):
            fpath = os.path.join(latin_dir, fn)
            if os.path.isfile(fpath) and fn.endswith('.txt'):
                with open(fpath) as f:
                    latin_text += f.read() + ' '

    anonimo_path = os.path.join(
        _data_dir(), 'reference', 'italian', 'anonimo_veneziano.txt',
    )
    anonimo_text = ''
    if os.path.exists(anonimo_path):
        with open(anonimo_path) as f:
            anonimo_text = f.read()

    ven_ref_bigrams = _build_venetian_reference_bigrams(latin_text, anonimo_text)
    print(f"    Venetian reference bigrams: {len(ven_ref_bigrams):,}")

    # ── 3. Find SIGNAL-SIGNAL pairs ──
    print("\n  3. Finding Venetian SIGNAL-SIGNAL pairs …")
    signal_pairs = _find_signal_pairs(
        decoded_tokens, ven_classifications, token_folios,
    )
    print(f"    SIGNAL-SIGNAL pairs: {len(signal_pairs):,}")

    all_signal_words = [
        decoded_tokens[i]
        for i in range(len(decoded_tokens))
        if (i < len(ven_classifications) and
            ven_classifications[i] == 'SIGNAL' and
            decoded_tokens[i])
    ]
    print(f"    Total SIGNAL words: {len(all_signal_words):,}")

    # ── 4. Fixed bigram z-test ──
    print("\n  4. Running FIXED bigram z-test (null also counts relaxed) …")
    z_results = _bigram_z_test_fixed(
        signal_pairs, ven_ref_bigrams, all_signal_words,
        n_permutations=500,
    )

    print(f"\n    === RESULTS ===")
    print(f"    Exact:   real={z_results['real_exact']}, "
          f"null={z_results['null_exact_mean']:.1f}±{z_results['null_exact_std']:.1f}, "
          f"z={z_results['z_exact']:.2f}")
    print(f"    Relaxed: real={z_results['real_relaxed']}, "
          f"null={z_results['null_relaxed_mean']:.1f}±{z_results['null_relaxed_std']:.1f}, "
          f"z={z_results['z_relaxed']:.2f}")
    print(f"    Total:   real={z_results['real_total']}, "
          f"null={z_results['null_total_mean']:.1f}±{z_results['null_total_std']:.1f}, "
          f"z={z_results['z_total']:.2f}")

    # ── 5. Comparison table ──
    print("\n  5. Comparison to prior results:")
    merged_bigrams = _safe_load(os.path.join(rd, 'merged_bigrams.json'))
    merged_z = merged_bigrams.get('bigram_z', 14.37)

    print(f"    | Reference       | z (exact) | z (total) | Exact | Relaxed |")
    print(f"    |-----------------|-----------|-----------|-------|---------|")
    print(f"    | Merged (L+I)    |     —     | {merged_z:9.2f} |   —   |    —    |")
    print(f"    | Venetian (P40)  |     —     | {original_z:9.2f} |  157  |  3877   |")
    print(f"    | Venetian (P41)  | {z_results['z_exact']:9.2f} | "
          f"{z_results['z_total']:9.2f} | "
          f"{z_results['real_exact']:5d} | {z_results['real_relaxed']:7d} |")

    delta = z_results['z_total'] - merged_z
    print(f"\n    Δ(Venetian validated − Merged): {delta:+.2f}")

    if z_results['z_total'] > merged_z:
        print(f"    → Venetian reference IMPROVES over merged by {delta:.2f}")
    elif z_results['z_total'] > 3.0:
        print(f"    → Venetian reference is SIGNIFICANT (z>{3.0}) "
              f"but not better than merged")
    else:
        print(f"    → Venetian reference is NOT SIGNIFICANT (z<3.0)")

    # ── 6. Save ──
    elapsed = time.time() - t0

    output = {
        'original_z': round(original_z, 4),
        'merged_z': round(merged_z, 4),
        **z_results,
        'venetian_selectivity': round(selectivity, 4),
        'real_venetian_hit_rate': round(real_hit_rate, 6),
        'null_mean_venetian_hit_rate': round(null_mean_rate, 6),
        'delta_vs_merged': round(delta, 4),
        'venetian_better_than_merged': z_results['z_total'] > merged_z,
        'venetian_significant': z_results['z_total'] > 3.0,
        'verdict': (
            'VENETIAN_BETTER' if z_results['z_total'] > merged_z else
            'VENETIAN_SIGNIFICANT' if z_results['z_total'] > 3.0 else
            'VENETIAN_NOT_SIGNIFICANT'
        ),
        'runtime_seconds': round(elapsed, 1),
    }

    out_path = os.path.join(rd, 'venetian_validated.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(output), f, indent=2)
    print(f"\n  Saved → {out_path} ({elapsed:.1f}s)")
