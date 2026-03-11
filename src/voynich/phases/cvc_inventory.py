"""
Step 40.5 – CVC/CCV Syllable Inventory
========================================
Build CVC and CCV syllable inventories from reference corpora and
profile how the expanded domain affects triple coverage.

Dependency chain:
    consonant_grouping.json     (Step 37.1)
    combined_refine.json        (Step 15)
    data/reference/italian/anonimo_veneziano.txt
        → cvc_inventory.json    (this step)
"""

import json
import os
import re
import time
from collections import Counter
from typing import Any, Dict, List, Set, Tuple

from voynich.core._paths import data_dir as _data_dir, results_dir as _results_dir
from voynich.core.reference import (
    build_cvc_syllable_table,
    build_triple_phoneme_hypotheses,
    PHONEME_INVENTORIES,
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


# ---------------------------------------------------------------------------
# Core: Syllabification of Venetian text
# ---------------------------------------------------------------------------

_VOWELS = set('aeiouàèéìòù')

def _syllabify_word(word: str) -> List[str]:
    """Simple Italian/Venetian syllabification using onset maximization."""
    word = word.lower()
    syllables = []
    current = ''

    i = 0
    while i < len(word):
        ch = word[i]
        current += ch

        if ch in _VOWELS:
            # Look ahead: if next char(s) are consonants followed by vowel,
            # they start a new syllable
            j = i + 1
            # Find the consonant cluster after this vowel
            cluster_start = j
            while j < len(word) and word[j] not in _VOWELS:
                j += 1
            # If there's a following vowel, split consonants:
            # keep at most 1 consonant with current syllable (coda),
            # rest go to next syllable (onset maximization)
            cluster_len = j - cluster_start
            if j < len(word) and cluster_len > 0:
                # Onset maximization: give all but possibly 1 to next syllable
                if cluster_len == 1:
                    # Single consonant → next onset
                    syllables.append(current)
                    current = ''
                elif cluster_len >= 2:
                    # Keep first consonant as coda, rest as onset of next
                    current += word[cluster_start]
                    syllables.append(current)
                    current = ''
                    i = cluster_start + 1
                    continue
            elif j >= len(word) and cluster_len > 0:
                # Consonants at end of word → coda
                current += word[cluster_start:j]
                i = j
                continue
            else:
                # Next char is vowel → close syllable
                syllables.append(current)
                current = ''
        i += 1

    if current:
        syllables.append(current)

    return syllables if syllables else [word]


def _extract_syllable_types(syllables: List[str]) -> Counter:
    """Classify syllables by type: V, CV, CVC, CCV, CCVC, etc."""
    type_counts: Counter = Counter()
    for syl in syllables:
        pattern = ''
        for ch in syl.lower():
            if ch in _VOWELS:
                pattern += 'V'
            else:
                pattern += 'C'
        type_counts[pattern] += 1
    return type_counts


def _syllabify_corpus_text(text: str) -> Tuple[Counter, Counter]:
    """Syllabify all words in a text and count syllable/type frequencies."""
    words = re.findall(r'[a-zA-ZàèéìòùÀÈÉÌÒÙ]+', text.lower())
    syl_freq: Counter = Counter()
    type_freq: Counter = Counter()

    for word in words:
        syls = _syllabify_word(word)
        for syl in syls:
            syl_freq[syl] += 1
        types = _extract_syllable_types(syls)
        type_freq.update(types)

    return syl_freq, type_freq


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_cvc_inventory() -> None:
    """Step 40.5: CVC/CCV Syllable Inventory."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 40.5: CVC/CCV Syllable Inventory")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load inputs ──
    print("\n  1. Loading inputs …")
    cons_group = _safe_load(os.path.join(rd, 'consonant_grouping.json'))
    refine = _safe_load(os.path.join(rd, 'combined_refine.json'))
    best_assignment = refine.get('best_assignment', {})
    print(f"    Phase 15 assignment: {len(best_assignment)} triples")
    print(f"    Consonant groups: {len(cons_group.get('groups', []))}")

    # ── 2. Profile relaxation levels 0–5 ──
    print("\n  2. Profiling CVC relaxation levels …")
    by_level = {}
    for level in range(6):
        try:
            inventory = build_cvc_syllable_table('italian', relaxation_level=level)
            n_syls = len(inventory) if isinstance(inventory, (list, set)) else 0
            if isinstance(inventory, dict):
                # May return a dict; count values
                all_syls = set()
                for v in inventory.values():
                    if isinstance(v, (list, set)):
                        all_syls.update(v)
                    elif isinstance(v, str):
                        all_syls.add(v)
                n_syls = len(all_syls)
                inventory_list = sorted(all_syls)
            else:
                inventory_list = sorted(inventory) if inventory else []

            by_level[level] = {
                'n_syllables': n_syls,
                'sample': inventory_list[:20],
            }
            print(f"    Level {level}: {n_syls} syllables")
        except Exception as e:
            by_level[level] = {
                'n_syllables': 0,
                'error': str(e),
            }
            print(f"    Level {level}: ERROR — {e}")

    # ── 3. Profile triple coverage ──
    print("\n  3. Profiling triple coverage per level …")
    for level in range(6):
        if by_level[level].get('error'):
            continue
        try:
            hyp = build_triple_phoneme_hypotheses('italian')
            n_triples_covered = sum(1 for v in hyp.values() if len(v) > 0)
            by_level[level]['n_triples_covered'] = n_triples_covered
            by_level[level]['triple_coverage'] = round(
                n_triples_covered / 25, 4) if 25 > 0 else 0.0
        except Exception:
            by_level[level]['n_triples_covered'] = 0
            by_level[level]['triple_coverage'] = 0.0

    # ── 4. Syllabify Anonimo Veneziano ──
    print("\n  4. Syllabifying Anonimo Veneziano …")
    anonimo_path = os.path.join(_data_dir(), 'reference', 'italian',
                                'anonimo_veneziano.txt')
    if os.path.exists(anonimo_path):
        with open(anonimo_path) as f:
            anonimo_text = f.read()
        syl_freq, type_freq = _syllabify_corpus_text(anonimo_text)
        print(f"    Total syllable tokens: {sum(syl_freq.values()):,}")
        print(f"    Unique syllables: {len(syl_freq):,}")
        print(f"    Type distribution:")
        for t, c in type_freq.most_common(8):
            print(f"      {t}: {c:,}")
        # Extract top CVC syllables
        cvc_syls = [(s, c) for s, c in syl_freq.most_common()
                    if _get_syl_type(s) == 'CVC']
        ccv_syls = [(s, c) for s, c in syl_freq.most_common()
                    if _get_syl_type(s) == 'CCV']
        print(f"    Top CVC: {cvc_syls[:10]}")
        print(f"    Top CCV: {ccv_syls[:10]}")
    else:
        syl_freq, type_freq = Counter(), Counter()
        cvc_syls, ccv_syls = [], []
        print("    Anonimo Veneziano not found")

    # ── 5. Recommend optimal level ──
    print("\n  5. Recommending optimal level …")
    # Find diminishing returns point
    prev_n = 0
    recommended = 3  # default
    for level in range(6):
        n = by_level[level].get('n_syllables', 0)
        increment = n - prev_n
        if level > 0 and increment < 10:
            recommended = level - 1
            break
        prev_n = n
    print(f"    Recommended relaxation level: {recommended}")

    # ── 6. Save ──
    elapsed = time.time() - t0

    output = {
        'by_level': by_level,
        'recommended_level': recommended,
        'anonimo_syllable_types': dict(type_freq.most_common(20)),
        'top_cvc_syllables': [{'syl': s, 'count': c} for s, c in cvc_syls[:30]],
        'top_ccv_syllables': [{'syl': s, 'count': c} for s, c in ccv_syls[:30]],
        'n_unique_anonimo_syllables': len(syl_freq),
        'runtime_seconds': round(elapsed, 1),
    }

    out_path = os.path.join(rd, 'cvc_inventory.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(output), f, indent=2)
    print(f"\n  Saved → {out_path} ({elapsed:.1f}s)")


def _get_syl_type(syl: str) -> str:
    """Get syllable type pattern (CV, CVC, CCV, etc.)."""
    pattern = ''
    for ch in syl.lower():
        if ch in _VOWELS:
            pattern += 'V'
        else:
            pattern += 'C'
    return pattern
