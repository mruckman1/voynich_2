"""
Step 41.5 – Unglossed Signal Word Analysis
============================================
Characterize the 45 unglossed signal words: classify each by whether it
appears in a known dictionary, is a near-miss (edit-distance-1), is a
likely short function word, or remains truly unknown.

Dependency chain:
    syllable_lexicon.json       (Step 40.9 — 73 signal words, 28 glossed)
    venetian_forms.json         (Step 40.1 — Venetian extended set)
    merged_signal.json          (Step 38.3 — word_signals with real_count)
        → unglossed_analysis.json  (this step)
"""

import json
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
# Edit distance
# ---------------------------------------------------------------------------

def _edit_distance_1(a: str, b: str) -> bool:
    """Return True if Levenshtein distance between a and b is exactly 1."""
    if abs(len(a) - len(b)) > 1:
        return False
    if a == b:
        return False
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


def _find_nearest(word: str, word_set: Set[str], max_len_diff: int = 2) -> Tuple[str, int]:
    """Find the nearest word in word_set by edit distance.

    Returns (nearest_word, edit_distance).  Only checks edit-distance-1
    candidates for speed; returns ('', -1) if nothing within distance 1.
    """
    for candidate in word_set:
        if abs(len(candidate) - len(word)) > max_len_diff:
            continue
        if _edit_distance_1(word, candidate):
            return candidate, 1
    return '', -1


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def _classify_unglossed(
    word: str,
    sigma: float,
    real_count: int,
    venetian_set: Set[str],
) -> Tuple[str, str, int]:
    """Classify an unglossed signal word.

    Returns (classification, nearest_dict_word, edit_distance).
    Classifications:
        IDENTIFIABLE  — exact match in a known dictionary
        NEAR_MISS     — edit-distance-1 from a dict word
        SHORT_FUNCTION — length <= 2 and freq > 50 (likely function word)
        UNKNOWN       — none of the above
    """
    # Check exact match
    if word in venetian_set:
        return 'IDENTIFIABLE', word, 0

    # Check near miss
    nearest, dist = _find_nearest(word, venetian_set)
    if dist == 1:
        return 'NEAR_MISS', nearest, 1

    # Short high-frequency function word heuristic
    if len(word) <= 2 and real_count > 50:
        return 'SHORT_FUNCTION', '', -1

    return 'UNKNOWN', '', -1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_unglossed_analysis() -> None:
    """Step 41.5: Analyze unglossed signal words."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 41.5: Unglossed Signal Word Analysis")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load inputs ──
    print("\n  1. Loading inputs …")
    syl_lex = _safe_load(os.path.join(rd, 'syllable_lexicon.json'))
    ven_forms = _safe_load(os.path.join(rd, 'venetian_forms.json'))
    merged_signal = _safe_load(os.path.join(rd, 'merged_signal.json'))

    lexicon = syl_lex.get('syllable_lexicon', {})
    venetian_set = set(ven_forms.get('venetian_extended_set', []))
    word_signals = merged_signal.get('word_signals', [])

    # Build word → signal info lookup
    word_info: Dict[str, Dict] = {}
    for ws in word_signals:
        word_info[ws.get('word', '')] = ws

    print(f"    Lexicon entries: {len(lexicon)}")
    print(f"    Venetian extended set: {len(venetian_set):,}")

    # ── 2. Separate glossed vs unglossed ──
    print("\n  2. Separating glossed / unglossed …")
    glossed_words: Dict[str, Dict] = {}
    unglossed_words: Dict[str, Dict] = {}

    for word, entry in lexicon.items():
        if entry.get('english_gloss', '???') != '???':
            glossed_words[word] = entry
        else:
            unglossed_words[word] = entry

    print(f"    Glossed: {len(glossed_words)}")
    print(f"    Unglossed: {len(unglossed_words)}")

    # ── 3. Classify each unglossed word ──
    print("\n  3. Classifying unglossed words …")
    analyses: List[Dict] = []
    class_counts: Counter = Counter()

    for word in sorted(unglossed_words.keys()):
        entry = unglossed_words[word]
        ws = word_info.get(word, {})
        sigma = ws.get('sigma', entry.get('sigma', 0.0))
        real_count = ws.get('real_count', 0)

        classification, nearest, edit_dist = _classify_unglossed(
            word, sigma, real_count, venetian_set,
        )
        class_counts[classification] += 1

        analyses.append({
            'word': word,
            'length': len(word),
            'sigma': sigma,
            'real_count': real_count,
            'source': ws.get('source', ''),
            'in_venetian_set': word in venetian_set,
            'classification': classification,
            'nearest_dict_word': nearest,
            'edit_distance': edit_dist,
        })

    # Sort by sigma descending
    analyses.sort(key=lambda x: -x['sigma'])

    # ── 4. Print results ──
    print(f"\n  4. Classification summary:")
    for cls, count in class_counts.most_common():
        print(f"    {cls}: {count}")

    print(f"\n    {'Word':12s} {'σ':>8s} {'Count':>6s} {'Class':16s} {'Nearest':12s}")
    print(f"    {'—' * 60}")
    for a in analyses[:30]:
        nearest_str = a['nearest_dict_word'] or '—'
        print(f"    {a['word']:12s} {a['sigma']:8.1f} {a['real_count']:6d} "
              f"{a['classification']:16s} {nearest_str:12s}")
    if len(analyses) > 30:
        print(f"    … and {len(analyses) - 30} more")

    # ── 5. Length distribution ──
    print("\n  5. Length distribution of unglossed words:")
    len_counts: Counter = Counter()
    for a in analyses:
        len_counts[a['length']] += 1
    for length in sorted(len_counts.keys()):
        print(f"    Length {length}: {len_counts[length]}")

    # ── 6. Save ──
    elapsed = time.time() - t0

    output = {
        'n_glossed': len(glossed_words),
        'n_unglossed': len(unglossed_words),
        'classification_counts': dict(class_counts),
        'unglossed_analyses': analyses,
        'glossed_words': list(glossed_words.keys()),
        'length_distribution': {str(k): v for k, v in sorted(len_counts.items())},
        'runtime_seconds': round(elapsed, 1),
    }

    out_path = os.path.join(rd, 'unglossed_analysis.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(output), f, indent=2)
    print(f"\n  Saved → {out_path} ({elapsed:.1f}s)")
