"""
Phase 66, Track 6: Fontana Structural Comparison
==================================================
Purely structural comparison -- count sign families, check
modification/rotation rules. No CVC decode needed.

Dependency chain:
    results/fontana_families.json     (Phase 21.2, may not exist)
    EVA_VISUAL_COMPONENTS             (reference.py)
        -> results/p66_fontana.json
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from voynich.core._paths import data_dir as _data_dir, results_dir as _results_dir
from voynich.core.reference import EVA_VISUAL_COMPONENTS


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def _convert(obj: Any) -> Any:
    if hasattr(obj, '__dataclass_fields__'):
        return {k: _convert(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _convert(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_convert(item) for item in obj]
    if isinstance(obj, float) and (obj != obj):
        return None
    if isinstance(obj, set):
        return sorted(obj)
    if isinstance(obj, (bool, int, float, str, type(None))):
        return obj
    return str(obj)


def _safe_load(path: str) -> Dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _save_json(rd: str, filename: str, data: Any) -> str:
    path = os.path.join(rd, filename)
    with open(path, 'w') as f:
        json.dump(_convert(data), f, indent=2)
    return path


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class FontanaStructResult:
    phase: str = "66"
    step: str = "66.6"
    experiment: str = "fontana_structural"
    n_fontana_families: int = 0
    n_voynich_families: int = 0
    family_count_diff: int = 0
    rotation_principle_present: bool = False
    voynich_has_gallows_rotation: bool = False
    structural_notes: List[str] = field(default_factory=list)
    fn1_family_count: bool = False   # |diff| <= 2
    fn2_rotation: bool = False       # rotation principle present
    gates_passed: int = 0
    gate_passed: bool = False
    verdict: str = ""
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_voynich_families() -> Dict[str, List[str]]:
    """
    Group EVA glyphs by (first_stroke, glyph_class) to form Voynich
    sign families from EVA_VISUAL_COMPONENTS.
    """
    families: Dict[str, List[str]] = defaultdict(list)
    for glyph, comp in EVA_VISUAL_COMPONENTS.items():
        key = (comp['first_stroke'], comp['glyph_class'])
        families[f"{key[0]}_{key[1]}"].append(glyph)
    return dict(families)


def _check_gallows_rotation() -> Tuple[bool, List[str]]:
    """
    Check if the four Voynich gallows (k, t, p, f) share first_stroke
    'ascender' but differ in last_stroke, consistent with a rotation
    or directional modification principle.
    """
    gallows = ['k', 't', 'p', 'f']
    notes = []
    first_strokes = set()
    last_strokes = {}

    for g in gallows:
        comp = EVA_VISUAL_COMPONENTS.get(g, {})
        fs = comp.get('first_stroke', '')
        ls = comp.get('last_stroke', '')
        first_strokes.add(fs)
        last_strokes[g] = ls

    all_ascender = all(
        EVA_VISUAL_COMPONENTS.get(g, {}).get('first_stroke') == 'ascender'
        for g in gallows
    )
    distinct_last = len(set(last_strokes.values()))

    if all_ascender:
        notes.append(
            f"All 4 gallows share first_stroke='ascender'; "
            f"{distinct_last} distinct last_strokes: "
            f"{dict(sorted(last_strokes.items()))}"
        )
    else:
        notes.append(
            f"Gallows first_strokes not uniform: {first_strokes}"
        )

    # Rotation principle: same onset, different last strokes (>= 3 distinct)
    has_rotation = all_ascender and distinct_last >= 3
    if has_rotation:
        notes.append(
            "Rotation principle PRESENT: 4 gallows from 1 ascender base, "
            "differentiated by last_stroke direction."
        )
    else:
        notes.append(
            "Rotation principle NOT confirmed for Voynich gallows."
        )

    return has_rotation, notes


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_fontana_struct() -> FontanaStructResult:
    t0 = time.time()
    rd = str(_results_dir())
    result = FontanaStructResult()

    print("=" * 70)
    print("Phase 66, Track 6: Fontana Structural Comparison")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load Fontana family data
    # ------------------------------------------------------------------
    fontana_path = os.path.join(rd, 'fontana_families.json')
    fontana_data = _safe_load(fontana_path)

    # Also check for data directory as fallback
    fontana_data_dir = str(_data_dir('reference') / 'fontana')
    has_fontana_dir = os.path.isdir(fontana_data_dir)

    if not fontana_data and not has_fontana_dir:
        print("[WARN] fontana_families.json not found and no fontana data directory.")
        result.verdict = "INSUFFICIENT_DATA"
        result.structural_notes.append(
            "Neither fontana_families.json nor data/reference/fontana/ found."
        )
        result.runtime_seconds = round(time.time() - t0, 2)
        _save_json(rd, 'p66_fontana.json', result)
        return result

    # Extract Fontana family count
    fontana_families = fontana_data.get('families', [])
    n_fontana = fontana_data.get('n_families', len(fontana_families))
    result.n_fontana_families = n_fontana

    if n_fontana == 0:
        print("[WARN] Fontana families data is empty.")
        result.verdict = "INSUFFICIENT_DATA"
        result.structural_notes.append("fontana_families.json has 0 families.")
        result.runtime_seconds = round(time.time() - t0, 2)
        _save_json(rd, 'p66_fontana.json', result)
        return result

    print(f"  Fontana families loaded: {n_fontana}")
    for fam in fontana_families:
        print(f"    {fam['base_form']}: {fam['size']} members")

    # ------------------------------------------------------------------
    # 2. Build Voynich sign families from EVA_VISUAL_COMPONENTS
    # ------------------------------------------------------------------
    voynich_families = _build_voynich_families()
    n_voynich = len(voynich_families)
    result.n_voynich_families = n_voynich

    print(f"  Voynich families (first_stroke x glyph_class): {n_voynich}")
    for fam_name, members in sorted(voynich_families.items()):
        print(f"    {fam_name}: {len(members)} glyphs {members}")

    # ------------------------------------------------------------------
    # 3. Family count comparison
    # ------------------------------------------------------------------
    diff = abs(n_voynich - n_fontana)
    result.family_count_diff = diff
    result.structural_notes.append(
        f"Fontana: {n_fontana} families, Voynich: {n_voynich} families, |diff|={diff}"
    )

    # ------------------------------------------------------------------
    # 4. Check rotation/directional modification principle
    # ------------------------------------------------------------------
    # 4a. Fontana: check gallows_rotation_test from the JSON
    fontana_rot = fontana_data.get('gallows_rotation_test', {})
    fontana_has_rotation = fontana_rot.get('rotation_match', False)
    four_member_fams = fontana_rot.get('four_member_families', [])

    if fontana_has_rotation:
        result.structural_notes.append(
            f"Fontana rotation principle: PRESENT "
            f"({len(four_member_fams)} families with 4+ directional variants)"
        )
    else:
        result.structural_notes.append(
            "Fontana rotation principle: NOT confirmed in stored data."
        )

    # 4b. Check Fontana families for directional tick variants >= 4
    fontana_rotation_families = []
    for fam in fontana_families:
        ticks = [f for f in fam.get('added_features', [])
                 if f.startswith('tick_')]
        if len(ticks) >= 4:
            fontana_rotation_families.append({
                'base_form': fam['base_form'],
                'n_ticks': len(ticks),
                'ticks': ticks,
            })

    if fontana_rotation_families:
        result.structural_notes.append(
            f"Fontana families with 4+ directional ticks: "
            f"{len(fontana_rotation_families)} "
            f"({[f['base_form'] for f in fontana_rotation_families]})"
        )
        result.rotation_principle_present = True
    else:
        # Fall back to the stored rotation_match value
        result.rotation_principle_present = fontana_has_rotation

    # 4c. Voynich gallows rotation
    voynich_has_rotation, rotation_notes = _check_gallows_rotation()
    result.voynich_has_gallows_rotation = voynich_has_rotation
    result.structural_notes.extend(rotation_notes)

    # Combined rotation assessment: both systems show the principle
    if result.rotation_principle_present and voynich_has_rotation:
        result.structural_notes.append(
            "BOTH Fontana and Voynich show rotation/directional modification "
            "principle (base form + directional variants)."
        )

    # ------------------------------------------------------------------
    # 5. Additional structural comparison notes
    # ------------------------------------------------------------------
    # Compare modifier toolkit
    modifier_cmp = fontana_data.get('modifier_toolkit_comparison', {})
    if modifier_cmp:
        fontana_props = modifier_cmp.get('fontana_proportions', {})
        if fontana_props:
            dominant = max(fontana_props, key=fontana_props.get)
            result.structural_notes.append(
                f"Fontana dominant modifier type: {dominant} "
                f"({fontana_props[dominant]:.1%})"
            )

    # Compare family size distributions
    fontana_sizes = sorted([fam['size'] for fam in fontana_families], reverse=True)
    voynich_sizes = sorted(
        [len(m) for m in voynich_families.values()], reverse=True
    )
    result.structural_notes.append(
        f"Fontana family sizes (desc): {fontana_sizes}"
    )
    result.structural_notes.append(
        f"Voynich family sizes (desc): {voynich_sizes}"
    )

    # ------------------------------------------------------------------
    # 6. Gate evaluation
    # ------------------------------------------------------------------
    result.fn1_family_count = diff <= 2
    result.fn2_rotation = result.rotation_principle_present
    result.gates_passed = sum([result.fn1_family_count, result.fn2_rotation])
    result.gate_passed = result.gates_passed >= 2

    if result.gate_passed:
        result.verdict = "FONTANA_STRUCTURALLY_SIMILAR"
    elif result.gates_passed == 1:
        result.verdict = "FONTANA_PARTIAL_MATCH"
    else:
        result.verdict = "FONTANA_NO_STRUCTURAL_MATCH"

    # ------------------------------------------------------------------
    # 7. Print summary and save
    # ------------------------------------------------------------------
    result.runtime_seconds = round(time.time() - t0, 2)

    print()
    print("-" * 50)
    print(f"  FN1 |family diff| <= 2:       {result.fn1_family_count}  (diff={diff})")
    print(f"  FN2 rotation principle:        {result.fn2_rotation}")
    print(f"  Gates passed:                  {result.gates_passed}/2")
    print(f"  Verdict:                       {result.verdict}")
    print(f"  Runtime:                       {result.runtime_seconds}s")
    print("-" * 50)
    print()
    print("  Structural notes:")
    for note in result.structural_notes:
        print(f"    - {note}")

    path = _save_json(rd, 'p66_fontana.json', result)
    print(f"  Saved: {path}")
    return result
