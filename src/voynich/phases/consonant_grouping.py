"""
Step 37.1 – Consonant Onset Grouping
=====================================
Group the 51 signal words by their decoded consonant onset and test whether
the grouping reveals the tachygraphic consonant classes.

Dependency chain:
    signal_10k.json            (Step 36.2)
    combined_refine.json       (Phase 15)
    tachygraphic_stroke.json   (Phase 19.5)
        → consonant_grouping.json   (this step)
"""

import json
import os
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import build_eva_to_triple_lookup, load_corpus, token_to_triples
from voynich.core.reference import EVA_VISUAL_COMPONENTS


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
# Onset extraction
# ---------------------------------------------------------------------------

# Multi-char Latin onsets to check first (longest match)
_MULTI_ONSETS = ['qu', 'ch', 'ph', 'th', 'sc', 'sp', 'st', 'sh', 'gn', 'gl',
                 'pl', 'pr', 'tr', 'cr', 'cl', 'fl', 'fr', 'br', 'bl', 'gr',
                 'dr', 'str', 'spr']


def _extract_onset(word: str) -> str:
    """Extract the consonant onset from a decoded Latin word.

    Returns the initial consonant(s) before the first vowel.
    For words starting with a vowel, returns '' (empty string).
    """
    if not word:
        return ''
    word = word.lower()
    vowels = set('aeiou')

    # Check multi-char onsets first (longest match)
    for onset in sorted(_MULTI_ONSETS, key=len, reverse=True):
        if word.startswith(onset):
            return onset

    # Single consonant onset
    if word[0] not in vowels:
        return word[0]

    # Vowel-initial
    return ''


def _extract_vowel_after_onset(word: str, onset: str) -> str:
    """Extract the vowel(s) immediately following the onset."""
    rest = word[len(onset):]
    vowels = set('aeiou')
    result = ''
    for ch in rest:
        if ch in vowels:
            result += ch
        else:
            break
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_consonant_grouping() -> None:
    """Step 37.1: Consonant Onset Grouping."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 37.1: Consonant Onset Grouping")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load inputs ──
    print("\n  1. Loading inputs …")
    signal_data = _safe_load(os.path.join(rd, 'signal_10k.json'))
    refine_data = _safe_load(os.path.join(rd, 'combined_refine.json'))
    stroke_data = _safe_load(os.path.join(rd, 'tachygraphic_stroke.json'))
    decode_data = _safe_load(os.path.join(rd, 'decode_10k.json'))

    word_signals = signal_data.get('word_signals', [])
    genuine_signals = [w for w in word_signals if w.get('is_genuine_signal')]
    assignment = refine_data.get('best_assignment', {})
    families = stroke_data.get('sign_families', [])

    print(f"     {len(genuine_signals)} genuine signal words")
    print(f"     {len(assignment)} triple assignments")
    print(f"     {len(families)} sign families")

    # ── 2. Group signal words by consonant onset ──
    print("  2. Grouping by consonant onset …")
    onset_groups: Dict[str, List[Dict]] = defaultdict(list)
    for ws in genuine_signals:
        word = ws['word']
        onset = _extract_onset(word)
        onset_groups[onset].append(ws)

    consonant_groups = []
    for onset in sorted(onset_groups.keys()):
        members = onset_groups[onset]
        words = [m['word'] for m in members]
        selectivities = [m['selectivity'] for m in members]
        vowels = []
        for w in words:
            v = _extract_vowel_after_onset(w, onset)
            if v:
                vowels.append(v)
        unique_vowels = sorted(set(vowels))
        mean_sel = sum(selectivities) / len(selectivities) if selectivities else 0.0
        var_sel = (sum((s - mean_sel) ** 2 for s in selectivities) / len(selectivities)
                   if len(selectivities) > 1 else 0.0)
        std_sel = var_sel ** 0.5

        consonant_groups.append({
            'onset': onset if onset else '<vowel>',
            'words': words,
            'word_count': len(words),
            'mean_selectivity': round(mean_sel, 3),
            'std_selectivity': round(std_sel, 3),
            'member_vowels': vowels,
            'unique_vowels': unique_vowels,
            'vowel_coverage': len(unique_vowels),
            'selectivities': [round(s, 3) for s in selectivities],
        })

    consonant_groups.sort(key=lambda g: g['word_count'], reverse=True)
    n_classes = len(consonant_groups)
    print(f"     {n_classes} distinct consonant classes found")

    for cg in consonant_groups:
        print(f"       {cg['onset']:<10s} {cg['word_count']:>2d} words: "
              f"{', '.join(cg['words'][:8])}{'…' if len(cg['words']) > 8 else ''}")

    # ── 3. Build triple-to-consonant mapping ──
    print("  3. Mapping triples to consonant classes …")
    triple_to_consonant: Dict[str, str] = {}
    for triple_key, syllable in assignment.items():
        onset = _extract_onset(syllable)
        triple_to_consonant[triple_key] = onset if onset else '<vowel>'

    # Group triples by consonant
    consonant_to_triples: Dict[str, List[str]] = defaultdict(list)
    for triple_key, cons in triple_to_consonant.items():
        consonant_to_triples[cons].append(triple_key)

    print(f"     {len(consonant_to_triples)} consonant classes from triple table:")
    for cons in sorted(consonant_to_triples.keys()):
        triples = consonant_to_triples[cons]
        syls = [assignment.get(t, '?') for t in triples]
        print(f"       {cons:<10s} {len(triples)} triples → {', '.join(syls)}")

    # ── 4. Map consonant groups to sign families ──
    print("  4. Mapping consonant groups to sign families …")

    # Build EVA char → family mapping
    eva_to_triple = build_eva_to_triple_lookup()
    char_to_family: Dict[str, str] = {}
    for fam in families:
        glyph_class = fam.get('glyph_class', '')
        for member in fam.get('members', []):
            char_to_family[member] = glyph_class

    # For each consonant group, find which triples produce its words
    # and which families those triples belong to
    group_family_maps = []
    for cg in consonant_groups:
        onset = cg['onset']
        # Find triples that have this onset
        matching_triples = consonant_to_triples.get(
            onset if onset != '<vowel>' else '', [])
        if onset == '<vowel>':
            matching_triples = consonant_to_triples.get('<vowel>', [])

        # Find which families those triples come from
        family_counts: Dict[str, int] = Counter()
        for triple_key in matching_triples:
            # triple_key = "first_stroke,last_stroke,glyph_class"
            parts = triple_key.split(',')
            if len(parts) == 3:
                glyph_class = parts[2]
                family_counts[glyph_class] += 1

        group_family_maps.append({
            'onset': onset,
            'matching_triples': matching_triples,
            'n_triples': len(matching_triples),
            'families': dict(family_counts),
            'primary_family': (family_counts.most_common(1)[0][0]
                              if family_counts else 'none'),
            'maps_to_single_family': len(family_counts) <= 1,
        })

    n_single_family = sum(1 for g in group_family_maps if g['maps_to_single_family'])
    print(f"     {n_single_family}/{len(group_family_maps)} groups map to single family")
    for gfm in group_family_maps:
        multi = "" if gfm['maps_to_single_family'] else " [MULTI]"
        print(f"       {gfm['onset']:<10s} → {gfm['primary_family']}{multi}"
              f"  ({gfm['n_triples']} triples)")

    # ── 5. Check if 5-class model fits ──
    print("  5. Testing C5×V4 model fit …")
    # Count consonant classes that have signal words with onset != vowel
    consonant_only_groups = [g for g in consonant_groups if g['onset'] != '<vowel>']
    n_consonant_classes = len(consonant_only_groups)
    all_selectivities = []
    for g in consonant_groups:
        all_selectivities.extend(g['selectivities'])
    overall_mean_sel = (sum(all_selectivities) / len(all_selectivities)
                        if all_selectivities else 0.0)

    c5v4_prediction = 5.0  # C=5 consonant classes → ~5× selectivity
    c5v4_match = abs(overall_mean_sel - c5v4_prediction) < 1.5

    print(f"     N consonant classes (non-vowel): {n_consonant_classes}")
    print(f"     Overall mean selectivity: {overall_mean_sel:.2f}×")
    print(f"     C5×V4 prediction: {c5v4_prediction:.1f}×")
    print(f"     Match (within 1.5×): {'YES' if c5v4_match else 'NO'}")

    # ── 6. Save ──
    elapsed = time.time() - t0
    output = {
        'n_signal_words': len(genuine_signals),
        'n_consonant_classes': n_classes,
        'n_consonant_only_classes': n_consonant_classes,
        'consonant_groups': consonant_groups,
        'triple_to_consonant': triple_to_consonant,
        'consonant_to_triples': {k: v for k, v in consonant_to_triples.items()},
        'group_family_maps': group_family_maps,
        'overall_mean_selectivity': round(overall_mean_sel, 3),
        'c5v4_prediction': c5v4_prediction,
        'c5v4_match': c5v4_match,
        'n_single_family_maps': n_single_family,
        'verdict': (
            f"{n_consonant_classes} consonant classes, "
            f"mean selectivity={overall_mean_sel:.2f}×, "
            f"C5×V4 match={'YES' if c5v4_match else 'NO'}, "
            f"{n_single_family}/{len(group_family_maps)} map to single family"
        ),
        'runtime_seconds': round(elapsed, 1),
    }

    out_path = os.path.join(rd, 'consonant_grouping.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(output), f, indent=2)
    print(f"\n  Saved → {out_path} ({elapsed:.1f}s)")
