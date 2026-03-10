"""
Step 39.14 -- Signal-Calibrated Dictionary
=============================================
Build a dictionary optimized for Voynich signal: start with 73 signal
words, expand by ED1 neighbors, reference collocates, Italian plant
names, Venetian supplement.  Calibrate null hit rate.

Dependency chain:
    merged_signal.json         (Step 38.3)
    merged_dict.json           (Step 38.1)
    italian_plant_names.json   (Step 39.8)
    venetian_lexicon.json      (Step 39.11)
        -> amplified_dict.json  (this step)
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
# Main
# ---------------------------------------------------------------------------

def run_amplified_dict() -> None:
    """Step 39.14: Signal-Calibrated Dictionary."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 39.14: Signal-Calibrated Dictionary")
    print("=" * 70)

    rd = _results_dir()

    # -- 1. Load inputs --
    print("\n  1. Loading inputs ...")

    signal_data = _safe_load(os.path.join(rd, 'merged_signal.json'))
    dict_data = _safe_load(os.path.join(rd, 'merged_dict.json'))
    plant_data = _safe_load(os.path.join(rd, 'italian_plant_names.json'))
    venetian_data = _safe_load(os.path.join(rd, 'venetian_lexicon.json'))
    null_data = _safe_load(os.path.join(rd, 'null_corpus.json'))

    merged_words: Set[str] = set(dict_data.get('merged_words', []))
    word_signals = signal_data.get('word_signals', [])

    print(f"     Merged dict: {len(merged_words)} words")
    print(f"     Word signals: {len(word_signals)} entries")

    # -- 2. Core: signal words --
    print("\n  2. Building core signal word set ...")

    core_words: Set[str] = set()
    for ws in word_signals:
        if ws.get('is_genuine_signal', False):
            core_words.add(ws['word'])

    n_core = len(core_words)
    print(f"     Core signal words: {n_core}")

    if core_words:
        for w in sorted(core_words)[:20]:
            print(f"       {w}")

    # -- 3. ED1 expansion --
    print("\n  3. ED1 expansion from merged dict ...")

    ed1_expanded: Set[str] = set()
    # For efficiency, only check merged dict words with similar lengths
    core_list = sorted(core_words)

    for signal_word in core_list:
        target_lens = {len(signal_word) - 1, len(signal_word), len(signal_word) + 1}
        for dict_word in merged_words:
            if len(dict_word) in target_lens:
                if _edit_distance_1(signal_word, dict_word):
                    ed1_expanded.add(dict_word)

    # Remove words already in core
    ed1_expanded -= core_words
    n_ed1 = len(ed1_expanded)
    print(f"     ED1 neighbors added: {n_ed1}")

    if ed1_expanded:
        for w in sorted(ed1_expanded)[:10]:
            print(f"       {w}")

    # -- 4. Collocate expansion --
    print("\n  4. Collocate expansion from decoded corpus ...")

    # Use decoded tokens to find words that appear near signal words
    token_decoded = signal_data.get('token_decoded', [])
    classifications = signal_data.get('token_classifications', [])
    decoded_lower = [w.lower() for w in token_decoded]

    collocate_counts: Counter = Counter()
    window = 2
    for i, cls in enumerate(classifications):
        if cls == 'SIGNAL':
            # Look at neighbors in window
            for j in range(max(0, i - window), min(len(decoded_lower), i + window + 1)):
                if j == i:
                    continue
                neighbor = decoded_lower[j]
                if neighbor in merged_words and neighbor not in core_words:
                    collocate_counts[neighbor] += 1

    # Take top collocates (appearing >= 3 times near signal words)
    collocates: Set[str] = set()
    for word, count in collocate_counts.most_common():
        if count >= 3:
            collocates.add(word)
        else:
            break

    collocates -= core_words
    collocates -= ed1_expanded
    n_collocates = len(collocates)
    print(f"     Collocates added: {n_collocates}")

    if collocates:
        top_colloc = sorted(collocates, key=lambda w: collocate_counts[w],
                            reverse=True)
        for w in top_colloc[:10]:
            print(f"       {w} (count={collocate_counts[w]})")

    # -- 5. Italian plant words --
    print("\n  5. Adding Italian plant words ...")

    plant_words: Set[str] = set(plant_data.get('all_plant_words', []))
    plant_words -= core_words
    plant_words -= ed1_expanded
    plant_words -= collocates
    n_plant = len(plant_words)
    print(f"     Plant words added: {n_plant}")

    # -- 6. Venetian supplement --
    print("\n  6. Adding Venetian supplement ...")

    venetian_words: Set[str] = set(venetian_data.get('supplement_words', []))
    venetian_words -= core_words
    venetian_words -= ed1_expanded
    venetian_words -= collocates
    venetian_words -= plant_words
    n_venetian = len(venetian_words)
    print(f"     Venetian words added: {n_venetian}")

    # -- 7. Build calibrated dictionary --
    print("\n  7. Building calibrated dictionary ...")

    calibrated_dict = (core_words | ed1_expanded | collocates |
                       plant_words | venetian_words)
    calibrated_size = len(calibrated_dict)
    print(f"     Calibrated dict size: {calibrated_size}")
    print(f"       Core: {n_core}")
    print(f"       ED1: {n_ed1}")
    print(f"       Collocates: {n_collocates}")
    print(f"       Plant: {n_plant}")
    print(f"       Venetian: {n_venetian}")

    # -- 8. Calibrate null hit rate --
    print("\n  8. Calibrating null hit rate ...")

    null_decoded_lists = null_data.get('null_decoded', [])
    null_hit_rates: List[float] = []

    if isinstance(null_decoded_lists, list):
        for null_decoded in null_decoded_lists[:5]:
            if isinstance(null_decoded, list):
                nd_lower = [w.lower() for w in null_decoded]
                hits = sum(1 for w in nd_lower if w in calibrated_dict)
                null_hit_rates.append(hits / max(len(nd_lower), 1))

    null_hit_rate = (sum(null_hit_rates) / len(null_hit_rates)
                     if null_hit_rates else 0.0)

    # Real hit rate
    real_hits = sum(1 for w in decoded_lower if w in calibrated_dict)
    real_hit_rate = real_hits / max(len(decoded_lower), 1)

    selectivity = real_hit_rate / max(null_hit_rate, 0.001)

    print(f"     Real hit rate: {real_hit_rate:.4f}")
    print(f"     Null hit rate: {null_hit_rate:.4f}")
    print(f"     Selectivity: {selectivity:.2f}x")

    # -- 9. Verdict --
    if selectivity >= 2.0:
        verdict = (f"CALIBRATED_DICT_STRONG: {calibrated_size} words, "
                   f"selectivity={selectivity:.2f}x")
    elif selectivity >= 1.5:
        verdict = (f"CALIBRATED_DICT_MODERATE: {calibrated_size} words, "
                   f"selectivity={selectivity:.2f}x")
    else:
        verdict = (f"CALIBRATED_DICT_WEAK: {calibrated_size} words, "
                   f"selectivity={selectivity:.2f}x")

    elapsed = time.time() - t0

    output = {
        'calibrated_dict_size': calibrated_size,
        'n_core': n_core,
        'n_ed1_expanded': n_ed1,
        'n_collocates': n_collocates,
        'n_plant_words': n_plant,
        'n_venetian': n_venetian,
        'null_hit_rate': round(null_hit_rate, 4),
        'real_hit_rate': round(real_hit_rate, 4),
        'selectivity': round(selectivity, 2),
        'calibrated_words': sorted(calibrated_dict),
        'core_words': sorted(core_words),
        'verdict': verdict,
        'runtime_seconds': round(elapsed, 1),
    }

    out_path = os.path.join(rd, 'amplified_dict.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(output), f, indent=2)
    print(f"\n  Saved -> {out_path} ({elapsed:.1f}s)")
