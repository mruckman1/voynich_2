"""
Step 41.6 – Venetian Dictionary Search for Unglossed Words
===========================================================
Systematic dictionary lookup for the unglossed signal words: exact match,
edit-distance-1 near-miss, concatenation splitting, and Venetian
morphological stem analysis.

Dependency chain:
    unglossed_analysis.json     (Step 41.5)
    venetian_forms.json         (Step 40.1 — Venetian extended set)
    data/reference/italian/anonimo_veneziano.txt
        → venetian_dictionary_search.json  (this step)
"""

import json
import os
import re
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple

from voynich.core._paths import data_dir as _data_dir, results_dir as _results_dir


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
# Edit distance
# ---------------------------------------------------------------------------

def _edit_distance_1(a: str, b: str) -> bool:
    """Return True if Levenshtein distance between a and b is exactly 1."""
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
# Venetian morphological suffixes
# ---------------------------------------------------------------------------

VENETIAN_SUFFIXES = [
    '-aro', '-ero', '-ato', '-ura',
    '-o', '-a', '-e', '-i',
]


# ---------------------------------------------------------------------------
# Dictionary loaders
# ---------------------------------------------------------------------------

def _load_anonimo_vocab(data_dir: str) -> Set[str]:
    """Extract unique lowercased words from the Anonimo Veneziano text."""
    path = os.path.join(data_dir, 'reference', 'italian', 'anonimo_veneziano.txt')
    if not os.path.exists(path):
        return set()
    with open(path, encoding='utf-8', errors='replace') as f:
        text = f.read()
    # Tokenize: lowercase, alphabetic words only
    words = set(re.findall(r'[a-z]+', text.lower()))
    return words


def _load_latin_word_set(results_dir: str) -> Set[str]:
    """Load Latin word set from merged_dict.json."""
    merged_dict = _safe_load(os.path.join(results_dir, 'merged_dict.json'))
    words: Set[str] = set()
    for key in ('latin_10k_words', 'italian_10k_words'):
        words.update(merged_dict.get(key, []))
    return words


# ---------------------------------------------------------------------------
# Search strategies
# ---------------------------------------------------------------------------

def _exact_match_search(
    word: str,
    venetian_set: Set[str],
    anonimo_vocab: Set[str],
    latin_set: Set[str],
) -> Optional[Dict]:
    """Try exact match across all dictionaries."""
    sources: List[str] = []
    if word in venetian_set:
        sources.append('venetian_extended')
    if word in anonimo_vocab:
        sources.append('anonimo_veneziano')
    if word in latin_set:
        sources.append('latin_italian')

    if sources:
        return {
            'method': 'exact_match',
            'matched_word': word,
            'sources': sources,
            'confidence': 'HIGH' if len(sources) >= 2 else 'MEDIUM',
        }
    return None


def _edit_distance_search(
    word: str,
    venetian_set: Set[str],
    anonimo_vocab: Set[str],
    latin_set: Set[str],
) -> Optional[Dict]:
    """Try edit-distance-1 match across dictionaries."""
    best_candidate = None
    best_source = ''

    # Search Venetian set first (preferred)
    for candidate in venetian_set:
        if abs(len(candidate) - len(word)) > 1:
            continue
        if _edit_distance_1(word, candidate):
            best_candidate = candidate
            best_source = 'venetian_extended'
            break

    # Try Anonimo if no Venetian match
    if not best_candidate:
        for candidate in anonimo_vocab:
            if abs(len(candidate) - len(word)) > 1:
                continue
            if _edit_distance_1(word, candidate):
                best_candidate = candidate
                best_source = 'anonimo_veneziano'
                break

    # Try Latin/Italian
    if not best_candidate:
        for candidate in latin_set:
            if abs(len(candidate) - len(word)) > 1:
                continue
            if _edit_distance_1(word, candidate):
                best_candidate = candidate
                best_source = 'latin_italian'
                break

    if best_candidate:
        return {
            'method': 'edit_distance_1',
            'matched_word': best_candidate,
            'sources': [best_source],
            'confidence': 'MEDIUM',
        }
    return None


def _concatenation_split(
    word: str,
    all_known_words: Set[str],
) -> Optional[Dict]:
    """Try splitting word at every position into two known words."""
    if len(word) < 3:
        return None

    for split_pos in range(1, len(word)):
        left = word[:split_pos]
        right = word[split_pos:]
        if left in all_known_words and right in all_known_words:
            return {
                'method': 'concatenation_split',
                'matched_word': f'{left}+{right}',
                'parts': [left, right],
                'sources': ['concatenation'],
                'confidence': 'LOW',
            }
    return None


def _morphological_stem(
    word: str,
    all_known_words: Set[str],
) -> Optional[Dict]:
    """Strip common Venetian suffixes and check if root is known."""
    # Try suffixes longest first
    suffixes_sorted = sorted(
        [s.lstrip('-') for s in VENETIAN_SUFFIXES],
        key=len, reverse=True,
    )

    for suffix in suffixes_sorted:
        if len(word) > len(suffix) and word.endswith(suffix):
            root = word[:-len(suffix)]
            if len(root) < 2:
                continue
            # Check root in any dictionary
            if root in all_known_words:
                return {
                    'method': 'morphological_stem',
                    'matched_word': root,
                    'suffix_stripped': suffix,
                    'sources': ['morphological'],
                    'confidence': 'LOW',
                }
            # Also try root + common endings
            for alt_ending in ('o', 'a', 'e', 'i'):
                alt = root + alt_ending
                if alt != word and alt in all_known_words:
                    return {
                        'method': 'morphological_stem',
                        'matched_word': alt,
                        'suffix_stripped': suffix,
                        'alt_ending': alt_ending,
                        'sources': ['morphological'],
                        'confidence': 'LOW',
                    }
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_venetian_dict_search() -> None:
    """Step 41.6: Venetian dictionary search for unglossed words."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 41.6: Venetian Dictionary Search")
    print("=" * 70)

    rd = _results_dir()
    dd = _data_dir()

    # ── 1. Load inputs ──
    print("\n  1. Loading inputs …")
    unglossed_data = _safe_load(os.path.join(rd, 'unglossed_analysis.json'))
    ven_forms = _safe_load(os.path.join(rd, 'venetian_forms.json'))

    unglossed_analyses = unglossed_data.get('unglossed_analyses', [])
    venetian_set = set(ven_forms.get('venetian_extended_set', []))

    print(f"    Unglossed words to search: {len(unglossed_analyses)}")
    print(f"    Venetian extended set: {len(venetian_set):,}")

    # ── 2. Load all dictionaries ──
    print("\n  2. Loading dictionaries …")
    anonimo_vocab = _load_anonimo_vocab(dd)
    latin_set = _load_latin_word_set(rd)

    print(f"    Anonimo Veneziano vocab: {len(anonimo_vocab):,}")
    print(f"    Latin/Italian word set: {len(latin_set):,}")

    # Combined set for concatenation / morphological searches
    all_known: Set[str] = set()
    all_known.update(venetian_set)
    all_known.update(anonimo_vocab)
    all_known.update(latin_set)
    print(f"    Combined known words: {len(all_known):,}")

    # ── 3. Search each unglossed word ──
    print("\n  3. Searching each unglossed word …")
    new_glosses: Dict[str, Dict] = {}
    method_counts: Counter = Counter()
    search_results: List[Dict] = []

    for entry in unglossed_analyses:
        word = entry['word']
        result: Dict[str, Any] = {
            'word': word,
            'sigma': entry.get('sigma', 0.0),
            'real_count': entry.get('real_count', 0),
        }

        # Strategy A: exact match
        match = _exact_match_search(word, venetian_set, anonimo_vocab, latin_set)
        if match:
            result['identification'] = match
            method_counts['exact_match'] += 1
            new_glosses[word] = match
            search_results.append(result)
            continue

        # Strategy B: edit-distance-1
        match = _edit_distance_search(word, venetian_set, anonimo_vocab, latin_set)
        if match:
            result['identification'] = match
            method_counts['edit_distance_1'] += 1
            new_glosses[word] = match
            search_results.append(result)
            continue

        # Strategy C: concatenation split
        match = _concatenation_split(word, all_known)
        if match:
            result['identification'] = match
            method_counts['concatenation_split'] += 1
            new_glosses[word] = match
            search_results.append(result)
            continue

        # Strategy D: morphological stem
        match = _morphological_stem(word, all_known)
        if match:
            result['identification'] = match
            method_counts['morphological_stem'] += 1
            new_glosses[word] = match
            search_results.append(result)
            continue

        # No match
        result['identification'] = {
            'method': 'none',
            'matched_word': '',
            'sources': [],
            'confidence': 'NONE',
        }
        method_counts['none'] += 1
        search_results.append(result)

    # Sort by sigma descending
    search_results.sort(key=lambda x: -x.get('sigma', 0.0))

    # ── 4. Print results ──
    n_identified = len(new_glosses)
    n_unidentified = len(unglossed_analyses) - n_identified

    print(f"\n  4. Search results:")
    print(f"    Identified: {n_identified}/{len(unglossed_analyses)}")
    print(f"    Unidentified: {n_unidentified}")
    print(f"\n    By method:")
    for method, count in method_counts.most_common():
        print(f"      {method}: {count}")

    print(f"\n    {'Word':12s} {'σ':>8s} {'Method':20s} {'Match':15s} {'Conf':8s}")
    print(f"    {'—' * 66}")
    for r in search_results[:35]:
        ident = r.get('identification', {})
        method = ident.get('method', '—')
        matched = ident.get('matched_word', '—')
        conf = ident.get('confidence', '—')
        print(f"    {r['word']:12s} {r.get('sigma', 0.0):8.1f} "
              f"{method:20s} {matched:15s} {conf:8s}")
    if len(search_results) > 35:
        print(f"    … and {len(search_results) - 35} more")

    # ── 5. Save ──
    elapsed = time.time() - t0

    output = {
        'n_unglossed_searched': len(unglossed_analyses),
        'n_identified': n_identified,
        'n_unidentified': n_unidentified,
        'method_counts': dict(method_counts),
        'new_glosses': new_glosses,
        'search_results': search_results,
        'runtime_seconds': round(elapsed, 1),
    }

    out_path = os.path.join(rd, 'venetian_dictionary_search.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(output), f, indent=2)
    print(f"\n  Saved → {out_path} ({elapsed:.1f}s)")
