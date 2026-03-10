"""
Step 38.4 – Merged Bigram Plausibility
=======================================
Confirm z≈17 from Phase 37.14 and compute the full bigram analysis including
language-tagged bigram types. THE critical test for macaronic content.

Dependency chain:
    merged_signal.json         (Step 38.3)
    merged_dict.json           (Step 38.1)
    decode_10k.json            (Step 36.1)
    bigrams_10k.json           (Step 36.3 — for comparison)
        → merged_bigrams.json  (this step)
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
# Core functions
# ---------------------------------------------------------------------------

def _find_signal_pairs_merged(
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


def _edit_distance_1(w1: str, w2: str) -> bool:
    """Check if edit distance ≤ 1."""
    if abs(len(w1) - len(w2)) > 1:
        return False
    if w1 == w2:
        return True
    if len(w1) == len(w2):
        return sum(a != b for a, b in zip(w1, w2)) <= 1
    # Insertion/deletion
    short, long = (w1, w2) if len(w1) < len(w2) else (w2, w1)
    i = j = diffs = 0
    while i < len(short) and j < len(long):
        if short[i] != long[j]:
            diffs += 1
            if diffs > 1:
                return False
            j += 1
        else:
            i += 1
            j += 1
    return True


def _test_merged_bigrams(
    pairs: List[Tuple[str, int, str, str]],
    bigram_set: Set[Tuple[str, str]],
    latin_10k: Set[str],
    italian_10k: Set[str],
) -> Tuple[int, int, List[Dict]]:
    """Test pairs against merged bigram table, tag by language type."""
    exact_hits = 0
    relaxed_hits = 0
    catalog = []

    for folio, pos, w1, w2 in pairs:
        is_exact = (w1, w2) in bigram_set
        is_relaxed = False

        if is_exact:
            exact_hits += 1
        else:
            # Check relaxed (edit distance 1)
            for bw1, bw2 in bigram_set:
                if _edit_distance_1(w1, bw1) and _edit_distance_1(w2, bw2):
                    is_relaxed = True
                    relaxed_hits += 1
                    break

        if is_exact or is_relaxed:
            # Classify language type
            w1_src = ('SHARED' if w1 in latin_10k and w1 in italian_10k
                      else 'LATIN_ONLY' if w1 in latin_10k
                      else 'ITALIAN_ONLY')
            w2_src = ('SHARED' if w2 in latin_10k and w2 in italian_10k
                      else 'LATIN_ONLY' if w2 in latin_10k
                      else 'ITALIAN_ONLY')

            # Bigram language type
            if w1_src == w2_src:
                bigram_type = f"{w1_src}_INTERNAL"
            else:
                bigram_type = 'CROSS_LANGUAGE'

            # Content classification (function words are short common ones)
            function_words = {'de', 'in', 'se', 'ne', 'ad', 'la', 'le', 'di',
                             'da', 'si', 'no', 'et', 'a', 'e', 'i', 'o', 'u',
                             'cum', 'per', 'pro', 'sub', 'que'}
            w1_content = w1 not in function_words and len(w1) >= 3
            w2_content = w2 not in function_words and len(w2) >= 3
            if w1_content and w2_content:
                content_type = 'content-content'
            elif w1_content or w2_content:
                content_type = 'function-content'
            else:
                content_type = 'function-function'

            catalog.append({
                'folio': folio,
                'position': pos,
                'w1': w1,
                'w2': w2,
                'w1_source': w1_src,
                'w2_source': w2_src,
                'bigram_type': bigram_type,
                'content_type': content_type,
                'match_type': 'exact' if is_exact else 'relaxed',
            })

    return exact_hits, relaxed_hits, catalog


def _null_permutation_merged(
    pairs: List[Tuple[str, int, str, str]],
    bigram_set: Set[Tuple[str, str]],
    signal_words: Set[str],
    n_perms: int = 500,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Null permutation test for bigram significance."""
    rng = random.Random(seed)
    signal_list = list(signal_words)

    null_counts = []
    for _ in range(n_perms):
        shuffled = list(signal_list) * 5
        rng.shuffle(shuffled)
        null_hits = sum(
            1 for j in range(len(shuffled) - 1)
            if (shuffled[j], shuffled[j + 1]) in bigram_set
        )
        null_counts.append(null_hits)

    null_mean = sum(null_counts) / len(null_counts) if null_counts else 0.0
    null_var = (sum((c - null_mean) ** 2 for c in null_counts) / len(null_counts)
                if null_counts else 0.0)
    null_std = null_var ** 0.5
    return null_mean, null_std, null_counts


def _per_folio_bigrams(
    catalog: List[Dict],
) -> List[Dict]:
    """Per-folio bigram analysis."""
    folio_data: Dict[str, Dict] = defaultdict(
        lambda: {'n_exact': 0, 'n_relaxed': 0, 'matches': [],
                 'lang_types': Counter(), 'content_types': Counter()}
    )
    for entry in catalog:
        f = entry['folio']
        if entry['match_type'] == 'exact':
            folio_data[f]['n_exact'] += 1
        else:
            folio_data[f]['n_relaxed'] += 1
        folio_data[f]['matches'].append(f"{entry['w1']} {entry['w2']}")
        folio_data[f]['lang_types'][entry['bigram_type']] += 1
        folio_data[f]['content_types'][entry['content_type']] += 1

    result = []
    for folio, data in sorted(folio_data.items(),
                               key=lambda x: x[1]['n_exact'], reverse=True):
        result.append({
            'folio': folio,
            'n_exact': data['n_exact'],
            'n_relaxed': data['n_relaxed'],
            'matches': data['matches'][:10],
            'lang_types': dict(data['lang_types']),
            'content_types': dict(data['content_types']),
        })
    return result


def _macaronic_structure(
    catalog: List[Dict],
) -> Dict[str, Any]:
    """Analyze macaronic patterns in cross-language bigrams."""
    cross_lang = [e for e in catalog if e['bigram_type'] == 'CROSS_LANGUAGE']

    patterns = {
        'shared_function_italian_content': 0,
        'italian_verb_shared_noun': 0,
        'latin_prep_italian_noun': 0,
        'other_cross': 0,
    }

    function_words = {'de', 'in', 'se', 'ne', 'ad', 'la', 'le', 'di',
                     'da', 'si', 'cum', 'per', 'pro', 'sub', 'que', 'et'}

    for e in cross_lang:
        w1_func = e['w1'] in function_words
        w2_func = e['w2'] in function_words
        w1_ita = e['w1_source'] == 'ITALIAN_ONLY'
        w2_ita = e['w2_source'] == 'ITALIAN_ONLY'
        w1_shared = e['w1_source'] == 'SHARED'
        w2_shared = e['w2_source'] == 'SHARED'

        if w1_shared and w1_func and w2_ita and not w2_func:
            patterns['shared_function_italian_content'] += 1
        elif w1_ita and not w1_func and w2_shared:
            patterns['italian_verb_shared_noun'] += 1
        elif e['w1_source'] == 'LATIN_ONLY' and w1_func and w2_ita:
            patterns['latin_prep_italian_noun'] += 1
        else:
            patterns['other_cross'] += 1

    return {
        'n_cross_language': len(cross_lang),
        'patterns': patterns,
        'cross_language_bigrams': [
            {'w1': e['w1'], 'w2': e['w2'],
             'w1_source': e['w1_source'], 'w2_source': e['w2_source']}
            for e in cross_lang
        ],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_merged_bigrams() -> None:
    """Step 38.4: Merged Bigram Plausibility."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 38.4: Merged Bigram Plausibility")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load inputs ──
    print("\n  1. Loading inputs …")
    signal_data = _safe_load(os.path.join(rd, 'merged_signal.json'))
    dict_data = _safe_load(os.path.join(rd, 'merged_dict.json'))
    bigram_10k_data = _safe_load(os.path.join(rd, 'bigrams_10k.json'))

    classifications = signal_data.get('token_classifications', [])
    decoded_lower = signal_data.get('token_decoded', [])
    token_folios = signal_data.get('token_folios', [])

    latin_10k = set(dict_data.get('latin_10k_words', []))
    italian_10k = set(dict_data.get('italian_10k_words', []))

    # Build bigram set from stored bigram list
    bigram_list = dict_data.get('bigram_list', [])
    bigram_set: Set[Tuple[str, str]] = set()
    for pair in bigram_list:
        if len(pair) == 2:
            bigram_set.add((pair[0], pair[1]))

    original_bigram_z = bigram_10k_data.get('bigram_z', 0.0)

    n_tokens = len(decoded_lower)
    n_signal = sum(1 for c in classifications if c == 'SIGNAL')

    print(f"     {n_tokens} tokens, {n_signal} SIGNAL")
    print(f"     {len(bigram_set)} merged bigrams")

    # ── 2. Find SIGNAL-SIGNAL pairs ──
    print("  2. Finding SIGNAL-SIGNAL pairs …")
    pairs = _find_signal_pairs_merged(classifications, decoded_lower, token_folios)
    signal_words = set(
        w for w, c in zip(decoded_lower, classifications) if c == 'SIGNAL'
    )

    print(f"     {len(pairs)} SIGNAL-SIGNAL pairs")
    print(f"     {len(signal_words)} unique SIGNAL words")

    # ── 3. Test bigrams ──
    print("  3. Testing bigram matches …")
    exact_hits, relaxed_hits, catalog = _test_merged_bigrams(
        pairs, bigram_set, latin_10k, italian_10k,
    )

    print(f"     Exact bigram hits: {exact_hits}")
    print(f"     Relaxed bigram hits: {relaxed_hits}")

    # ── 4. Null permutation test ──
    print("  4. Null permutation test (500 shuffles) …")
    null_mean, null_std, null_counts = _null_permutation_merged(
        pairs, bigram_set, signal_words,
    )
    z_score = ((exact_hits - null_mean) / null_std
               if null_std > 0
               else (10.0 if exact_hits > null_mean else 0.0))

    print(f"     Null: mean={null_mean:.2f}, std={null_std:.2f}")
    print(f"     z-score: {z_score:.2f}")
    print(f"     (Phase 36 Latin 10K z: {original_bigram_z:.2f})")

    # ── 5. Content-content analysis ──
    print("  5. Content-content bigram analysis …")
    content_types = Counter(e['content_type'] for e in catalog)
    n_cc = content_types.get('content-content', 0)
    n_fc = content_types.get('function-content', 0)
    n_ff = content_types.get('function-function', 0)

    print(f"     content-content: {n_cc}")
    print(f"     function-content: {n_fc}")
    print(f"     function-function: {n_ff}")

    # ── 6. Language-type analysis ──
    print("  6. Language-type analysis …")
    lang_types = Counter(e['bigram_type'] for e in catalog)
    macaronic = _macaronic_structure(catalog)

    for lt, count in lang_types.most_common():
        print(f"     {lt}: {count}")
    print(f"     Cross-language macaronic patterns: {macaronic['n_cross_language']}")

    # ── 7. Per-folio bigrams ──
    print("  7. Per-folio analysis …")
    folio_bigrams = _per_folio_bigrams(catalog)
    for fb in folio_bigrams[:5]:
        print(f"     {fb['folio']:8s}: {fb['n_exact']} exact, {fb['n_relaxed']} relaxed")

    # ── 8. Bigram catalog (top entries) ──
    print("  8. Bigram catalog:")
    for entry in catalog[:15]:
        print(f"     [{entry['match_type']:7s}] {entry['w1']:>8s} {entry['w2']:<8s}  "
              f"{entry['bigram_type']:20s}  {entry['content_type']:18s}  {entry['folio']}")

    # ── 9. Trigram test ──
    print("  9. Trigram test …")
    trigram_hits = 0
    trigram_catalog = []
    for i in range(len(decoded_lower) - 2):
        if i >= len(token_folios) - 2:
            break
        if (token_folios[i] != token_folios[i + 1] or
            token_folios[i + 1] != token_folios[i + 2]):
            continue
        if (classifications[i] == 'SIGNAL' and
            classifications[i + 1] == 'SIGNAL' and
            classifications[i + 2] == 'SIGNAL'):
            w1, w2, w3 = decoded_lower[i], decoded_lower[i + 1], decoded_lower[i + 2]
            if ((w1, w2) in bigram_set and (w2, w3) in bigram_set):
                trigram_hits += 1
                trigram_catalog.append({
                    'folio': token_folios[i],
                    'position': i,
                    'words': [w1, w2, w3],
                })
    print(f"     Trigram hits (both bigrams match): {trigram_hits}")

    # ── 10. Save ──
    elapsed = time.time() - t0

    output = {
        'n_signal_pairs': len(pairs),
        'n_signal_words': len(signal_words),
        'exact_hits': exact_hits,
        'relaxed_hits': relaxed_hits,
        'null_mean': round(null_mean, 2),
        'null_std': round(null_std, 2),
        'bigram_z': round(z_score, 2),
        'original_latin_z': round(original_bigram_z, 2),
        'n_content_content': n_cc,
        'n_function_content': n_fc,
        'n_function_function': n_ff,
        'content_types': dict(content_types),
        'language_types': dict(lang_types),
        'macaronic_structure': macaronic,
        'bigram_catalog': catalog,
        'per_folio_bigrams': folio_bigrams,
        'trigram_hits': trigram_hits,
        'trigram_catalog': trigram_catalog,
        'comparison': {
            'latin_10k_z': round(original_bigram_z, 2),
            'merged_z': round(z_score, 2),
            'z_improved': z_score > original_bigram_z,
        },
        'verdict': (
            f"Merged bigram z={z_score:.2f} "
            f"(Latin 10K: {original_bigram_z:.2f}). "
            f"Exact: {exact_hits}, Relaxed: {relaxed_hits}. "
            f"CC: {n_cc}, FC: {n_fc}, FF: {n_ff}. "
            f"Cross-language: {macaronic['n_cross_language']}. "
            f"Trigrams: {trigram_hits}."
        ),
        'runtime_seconds': round(elapsed, 1),
    }

    out_path = os.path.join(rd, 'merged_bigrams.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(output), f, indent=2)
    print(f"\n  Saved → {out_path} ({elapsed:.1f}s)")
