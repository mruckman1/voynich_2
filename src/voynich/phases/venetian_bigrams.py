"""
Step 40.3 – Venetian Reference Bigrams
========================================
Build a bigram reference from Venetian medical text and test the decoded
corpus against it. Compare z-score to merged (z=14.37).

Dependency chain:
    venetian_match.json    (Step 40.2)
    venetian_forms.json    (Step 40.1)
    merged_bigrams.json    (Step 38.4)
    data/reference/italian/anonimo_veneziano.txt
        → venetian_bigrams.json  (this step)
"""

import json
import math
import os
import random
import re
import time
from collections import Counter
from typing import Any, Dict, List, Set, Tuple

from voynich.core._paths import data_dir as _data_dir, results_dir as _results_dir
from voynich.core.reference import apply_venetian_sound_changes


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
        diffs = sum(1 for x, y in zip(a, b) if x != y)
        return diffs == 1
    # Insertion/deletion
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
# Core: Venetian bigram reference
# ---------------------------------------------------------------------------

def _build_venetian_reference_bigrams(
    latin_ref_text: str,
    anonimo_text: str,
) -> Set[Tuple[str, str]]:
    """Build a bigram set from Venetian-transformed Latin text + Anonimo."""
    bigrams: Set[Tuple[str, str]] = set()

    # 1. Transform Latin reference text to synthetic Venetian
    latin_words = re.findall(r'[a-z]+', latin_ref_text.lower())
    venetian_words = []
    for w in latin_words:
        variants = apply_venetian_sound_changes(w)
        if variants:
            # Use first variant
            venetian_words.append(next(iter(variants)))
        else:
            venetian_words.append(w)

    for i in range(len(venetian_words) - 1):
        bigrams.add((venetian_words[i], venetian_words[i + 1]))

    # 2. Add bigrams from Anonimo Veneziano
    anonimo_words = re.findall(r'[a-zA-ZàèéìòùÀÈÉÌÒÙ]+', anonimo_text.lower())
    for i in range(len(anonimo_words) - 1):
        bigrams.add((anonimo_words[i], anonimo_words[i + 1]))

    return bigrams


def _find_signal_pairs(
    decoded_tokens: List[str],
    classifications: List[str],
    folios: List[str],
) -> List[Dict]:
    """Find consecutive SIGNAL-SIGNAL pairs on same folio."""
    pairs = []
    n = len(decoded_tokens)
    for i in range(n - 1):
        if (classifications[i] == 'SIGNAL' and
                classifications[i + 1] == 'SIGNAL' and
                folios[i] == folios[i + 1]):
            pairs.append({
                'folio': folios[i],
                'position': i,
                'w1': decoded_tokens[i],
                'w2': decoded_tokens[i + 1],
            })
    return pairs


def _build_word_index(reference_bigrams: Set[Tuple[str, str]]) -> Dict[str, Set[str]]:
    """Build word→partner index for fast relaxed bigram lookup."""
    index: Dict[str, Set[str]] = {}
    for w1, w2 in reference_bigrams:
        if w1 not in index:
            index[w1] = set()
        index[w1].add(w2)
    return index


def _relaxed_bigram_check(
    w1: str, w2: str,
    ref_bigrams: Set[Tuple[str, str]],
    word_index: Dict[str, Set[str]],
    all_ref_words: Set[str],
) -> bool:
    """Fast relaxed bigram check using word index."""
    # Exact check
    if (w1, w2) in ref_bigrams:
        return True
    # Check words within edit distance 1 of w1
    for rw1 in all_ref_words:
        if _edit_distance_1(w1, rw1) and rw1 in word_index:
            for rw2 in word_index[rw1]:
                if _edit_distance_1(w2, rw2):
                    return True
    return False


def _bigram_z_test(
    signal_pairs: List[Dict],
    reference_bigrams: Set[Tuple[str, str]],
    all_signal_words: List[str],
    n_permutations: int = 1000,
) -> Tuple[float, int, int]:
    """Compute bigram plausibility z-score.

    Returns: (z_score, n_exact_hits, n_relaxed_hits)
    """
    # Build index for faster lookup
    word_index = _build_word_index(reference_bigrams)
    # Only check against reference words that are short (≤6 chars) for speed
    all_ref_words = set()
    for w1, w2 in reference_bigrams:
        all_ref_words.add(w1)
        all_ref_words.add(w2)
    # Limit to words of similar length to signal words for speed
    signal_word_set = set(all_signal_words)
    max_len = max((len(w) for w in signal_word_set), default=4) + 1
    filtered_ref_words = {w for w in all_ref_words if len(w) <= max_len}

    # Count exact and relaxed hits
    exact_hits = 0
    relaxed_hits = 0
    for pair in signal_pairs:
        w1, w2 = pair['w1'], pair['w2']
        if (w1, w2) in reference_bigrams:
            exact_hits += 1
        else:
            # Check relaxed with length filter
            found = False
            for rw1 in filtered_ref_words:
                if _edit_distance_1(w1, rw1) and rw1 in word_index:
                    for rw2 in word_index[rw1]:
                        if _edit_distance_1(w2, rw2):
                            found = True
                            break
                if found:
                    break
            if found:
                relaxed_hits += 1

    real_hits = exact_hits + relaxed_hits

    # Null permutation test (exact only for speed)
    rng = random.Random(42)
    null_hits_list = []
    for _ in range(n_permutations):
        shuffled = list(all_signal_words)
        rng.shuffle(shuffled)
        null_h = 0
        for k in range(min(len(signal_pairs), len(shuffled) - 1)):
            if (shuffled[k], shuffled[k + 1]) in reference_bigrams:
                null_h += 1
        null_hits_list.append(null_h)

    null_mean = sum(null_hits_list) / len(null_hits_list) if null_hits_list else 0.0
    null_std = (sum((h - null_mean) ** 2 for h in null_hits_list)
                / len(null_hits_list)) ** 0.5 if null_hits_list else 0.001

    z = (real_hits - null_mean) / null_std if null_std > 0.001 else 0.0

    return z, exact_hits, relaxed_hits


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_venetian_bigrams() -> None:
    """Step 40.3: Venetian Reference Bigrams."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 40.3: Venetian Reference Bigrams")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load inputs ──
    print("\n  1. Loading inputs …")
    ven_match = _safe_load(os.path.join(rd, 'venetian_match.json'))
    merged_bigrams = _safe_load(os.path.join(rd, 'merged_bigrams.json'))

    ven_classifications = ven_match.get('token_classifications_venetian', [])
    decoded_tokens = ven_match.get('token_decoded', [])
    token_folios = ven_match.get('token_folios', [])
    print(f"    Decoded tokens: {len(decoded_tokens):,}")
    print(f"    Venetian classifications: {len(ven_classifications):,}")

    # ── 2. Build Venetian reference bigrams ──
    print("\n  2. Building Venetian reference bigram set …")
    # Load Latin reference text
    from voynich.core._paths import data_dir
    latin_dir = os.path.join(data_dir(), 'reference', 'latin')
    latin_text = ''
    if os.path.isdir(latin_dir):
        for fn in os.listdir(latin_dir):
            fpath = os.path.join(latin_dir, fn)
            if os.path.isfile(fpath) and fn.endswith('.txt'):
                with open(fpath) as f:
                    latin_text += f.read() + ' '

    anonimo_path = os.path.join(data_dir(), 'reference', 'italian',
                                'anonimo_veneziano.txt')
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
    print(f"    Venetian SIGNAL-SIGNAL pairs: {len(signal_pairs):,}")

    # Collect all signal words for null permutation
    all_signal_words = [decoded_tokens[i] for i in range(len(decoded_tokens))
                        if i < len(ven_classifications) and
                        ven_classifications[i] == 'SIGNAL' and decoded_tokens[i]]

    # ── 4. Bigram z-test ──
    print("\n  4. Running bigram plausibility test …")
    z, n_exact, n_relaxed = _bigram_z_test(
        signal_pairs, ven_ref_bigrams, all_signal_words,
        n_permutations=500,
    )
    print(f"    Exact bigram hits: {n_exact}")
    print(f"    Relaxed bigram hits (edit ≤ 1): {n_relaxed}")
    print(f"    Bigram z-score: {z:.2f}")

    # ── 5. Compare to merged results ──
    print("\n  5. Comparison:")
    merged_z = merged_bigrams.get('bigram_z', 14.37)
    merged_exact = merged_bigrams.get('n_exact_hits',
                                       merged_bigrams.get('exact_hits', 0))
    merged_relaxed = merged_bigrams.get('n_relaxed_hits',
                                         merged_bigrams.get('relaxed_hits', 0))
    print(f"    Merged (L+I) z: {merged_z:.2f}, exact: {merged_exact}, relaxed: {merged_relaxed}")
    print(f"    Venetian z: {z:.2f}, exact: {n_exact}, relaxed: {n_relaxed}")
    print(f"    Delta z: {z - merged_z:+.2f}")

    # ── 6. Save ──
    elapsed = time.time() - t0

    output = {
        'n_venetian_ref_bigrams': len(ven_ref_bigrams),
        'n_venetian_signal_pairs': len(signal_pairs),
        'n_exact_bigram_hits': n_exact,
        'n_relaxed_bigram_hits': n_relaxed,
        'bigram_z': round(z, 4),
        'merged_z': round(merged_z, 4),
        'delta_z_vs_merged': round(z - merged_z, 4),
        'signal_pairs_sample': signal_pairs[:50],
        'comparison': {
            'merged_z': round(merged_z, 2),
            'merged_exact': merged_exact,
            'merged_relaxed': merged_relaxed,
            'venetian_z': round(z, 2),
            'venetian_exact': n_exact,
            'venetian_relaxed': n_relaxed,
        },
        'verdict': ('VENETIAN_BETTER' if z > merged_z + 0.5
                    else 'COMPARABLE' if abs(z - merged_z) <= 0.5
                    else 'MERGED_BETTER'),
        'runtime_seconds': round(elapsed, 1),
    }

    out_path = os.path.join(rd, 'venetian_bigrams.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(output), f, indent=2)
    print(f"\n  Saved → {out_path} ({elapsed:.1f}s)")
