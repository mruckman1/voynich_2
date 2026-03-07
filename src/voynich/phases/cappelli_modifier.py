"""
Phase 21.6 – Modifier Identification via Cappelli (cappelli-mod)
================================================================
Matches Phase 16's 15 modifier characters against Cappelli abbreviation
marks using both visual comparison (149 entries) and functional comparison
(all 2,678 entries via bracket notation).

Dependency chain:
    paleo_ingest.json (master_reference.json)
    + modifier_integrate.json (Phase 16)
        → cappelli_modifier.json (this step)
"""

import json
import random
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.reference import EVA_VISUAL_COMPONENTS, normalize_stroke, stroke_category
from voynich.core.stats import stroke_similarity


# ---------------------------------------------------------------------------
# Helpers
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
    if isinstance(obj, (bool, int, float, str, type(None))):
        return obj
    return str(obj)


def _load_json(path: str) -> Optional[Dict]:
    import os
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Cappelli bracket notation → function type mapping
# ---------------------------------------------------------------------------

_BRACKET_FUNCTION: Dict[str, str] = {
    'macron': 'omission_nasal',
    'tilde': 'nasalization',
    'sup': 'superscript',
    '9-sign': 'con_cum_que',
    'stroke-through': 'truncation',
    'stroke through': 'truncation',
    'dot-above': 'abbreviation_marker',
    'dot-below': 'abbreviation_marker',
    'dot above': 'abbreviation_marker',
    'overline': 'omission_nasal',
    'underline': 'emphasis',
    'double-stroke': 'truncation',
}


def _classify_bracket(bracket: str) -> str:
    """Classify a Cappelli bracket notation into a function type."""
    bl = bracket.lower().strip()
    # Direct match
    if bl in _BRACKET_FUNCTION:
        return _BRACKET_FUNCTION[bl]
    # Prefix match (e.g., 'sup:X')
    for key, val in _BRACKET_FUNCTION.items():
        if bl.startswith(key):
            return val
    # Keywords
    if 'macron' in bl or 'overline' in bl:
        return 'omission_nasal'
    if 'tilde' in bl:
        return 'nasalization'
    if 'sup' in bl:
        return 'superscript'
    if 'stroke' in bl:
        return 'truncation'
    if 'dot' in bl:
        return 'abbreviation_marker'
    return 'other'


# ---------------------------------------------------------------------------
# Voynich modifier categories → Cappelli function prediction
# ---------------------------------------------------------------------------

# Phase 16 modifier types → predicted Cappelli function
_MODIFIER_TO_CAPPELLI_PRED: Dict[str, str] = {
    'vowel_changer': 'superscript',      # Changes vowel = superscript letter
    'geminator': 'truncation',            # Gemination = stroke-through
    'nasalizer': 'omission_nasal',        # Nasalization = macron/tilde
    'cluster': 'con_cum_que',             # Cluster = 9-sign (con/cum)
    'silent': 'abbreviation_marker',      # Silent marker = dot
}


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class VisualMatch:
    modifier_char: str
    cappelli_id: str
    similarity_score: float
    cappelli_latin: str
    cappelli_desc: str


@dataclass
class FunctionalAssignment:
    modifier_char: str
    voynich_type: str
    predicted_cappelli_function: str
    cappelli_bracket_matches: int
    distributional_test_passed: bool
    description: str


@dataclass
class CappelliModifierResult:
    timestamp: str
    n_modifiers: int
    visual_matches: List[Dict[str, Any]]
    functional_assignments: List[Dict[str, Any]]
    distributional_test_results: Dict[str, Any]
    null_selectivity: float
    gate_passed: bool


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_cappelli_modifier() -> Dict[str, Any]:
    """Match modifier chars against Cappelli abbreviation marks."""
    t0 = time.time()
    rdir = _results_dir()

    # Load master reference
    master = _load_json("data/reference/paleographic/master_reference.json") or {}
    all_signs = master.get('all_signs', [])
    cappelli_signs = [s for s in all_signs if s.get('source') == 'cappelli']

    # Separate visual (have strokes) vs all (have bracket marks)
    cappelli_visual = [s for s in cappelli_signs if s.get('first_stroke') or s.get('final_stroke')]
    cappelli_all = cappelli_signs  # All have bracket_marks field

    # Load Phase 16 modifiers
    mod_data = _load_json(str(rdir / "modifier_integrate.json")) or {}
    modifier_chars: List[str] = []
    modifier_types: Dict[str, str] = {}

    # Extract modifier list
    for key in ['modifier_chars', 'confirmed_modifiers', 'modifiers']:
        mods = mod_data.get(key, [])
        if mods:
            for m in mods:
                if isinstance(m, str):
                    modifier_chars.append(m)
                elif isinstance(m, dict):
                    mc = m.get('eva_char', m.get('char', ''))
                    if mc:
                        modifier_chars.append(mc)
                        modifier_types[mc] = m.get('modifier_type', m.get('type', 'unknown'))
            break

    if not modifier_chars:
        # Fallback: use known modifiers from Phase 16
        modifier_chars = ['h', 'iin', 'b', 'ckh', 'i', 'iiin', 'u', 'aiin', 'al', 'ar', 'dy', 'ey', 'm', 'n', 'or']

    # --- 6a: Visual comparison (149 entries) ---
    visual_matches: List[VisualMatch] = []
    for mc in modifier_chars:
        if mc not in EVA_VISUAL_COMPONENTS:
            continue
        eva_comps = EVA_VISUAL_COMPONENTS[mc]
        eva_strokes = {
            'first_stroke': eva_comps.get('first_stroke', ''),
            'last_stroke': eva_comps.get('last_stroke', ''),
            'glyph_class': eva_comps.get('glyph_class', ''),
        }

        best_sim = 0.0
        best_match: Optional[Dict] = None
        for cs in cappelli_visual:
            h_strokes = {
                'first_stroke': cs.get('first_stroke', ''),
                'last_stroke': cs.get('final_stroke', '') or cs.get('last_stroke', ''),
                'glyph_class': cs.get('glyph_class', ''),
            }
            sim = stroke_similarity(eva_strokes, h_strokes, include_class=False)
            if sim > best_sim:
                best_sim = sim
                best_match = cs

        if best_match and best_sim > 0.4:
            visual_matches.append(VisualMatch(
                modifier_char=mc,
                cappelli_id=best_match.get('source_id', ''),
                similarity_score=best_sim,
                cappelli_latin=best_match.get('latin_value', '') or '',
                cappelli_desc='',
            ))

    # --- 6b: Functional comparison (all 2678 entries) ---
    # Count bracket function types across all Cappelli
    bracket_func_counts: Counter = Counter()
    for cs in cappelli_all:
        for bm in cs.get('bracket_marks', []):
            func = _classify_bracket(bm)
            bracket_func_counts[func] += 1

    functional_assignments: List[FunctionalAssignment] = []
    distributional_passes = 0
    for mc in modifier_chars:
        vtype = modifier_types.get(mc, 'unknown')
        predicted_func = _MODIFIER_TO_CAPPELLI_PRED.get(vtype, 'other')

        # Count how many Cappelli entries have this function
        n_matches = bracket_func_counts.get(predicted_func, 0)

        # Distributional test: the predicted function should be one of the top functions
        total_brackets = sum(bracket_func_counts.values()) or 1
        predicted_frac = n_matches / total_brackets
        dist_passed = predicted_frac > 0.05  # At least 5% of bracket marks

        if dist_passed:
            distributional_passes += 1

        functional_assignments.append(FunctionalAssignment(
            modifier_char=mc,
            voynich_type=vtype,
            predicted_cappelli_function=predicted_func,
            cappelli_bracket_matches=n_matches,
            distributional_test_passed=dist_passed,
            description=f"{mc}({vtype})→{predicted_func}: {n_matches} Cappelli matches",
        ))

    # --- Null control ---
    # Random modifier-to-mark assignments
    func_types = list(_BRACKET_FUNCTION.values())
    rng = random.Random(42)
    null_passes: List[int] = []
    for _ in range(100):
        null_pass = 0
        for mc in modifier_chars:
            rand_func = rng.choice(func_types)
            n = bracket_func_counts.get(rand_func, 0)
            if n / total_brackets > 0.05:
                null_pass += 1
        null_passes.append(null_pass)

    null_mean = sum(null_passes) / max(len(null_passes), 1)
    selectivity = distributional_passes / max(null_mean, 0.01)

    result = CappelliModifierResult(
        timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
        n_modifiers=len(modifier_chars),
        visual_matches=[_convert(asdict(v)) for v in visual_matches],
        functional_assignments=[_convert(asdict(f)) for f in functional_assignments],
        distributional_test_results={
            'n_tested': len(modifier_chars),
            'n_distributional_passes': distributional_passes,
            'pass_rate': distributional_passes / max(len(modifier_chars), 1),
            'bracket_function_counts': dict(bracket_func_counts),
        },
        null_selectivity=selectivity,
        gate_passed=selectivity > 1.0,
    )

    out_path = rdir / "cappelli_modifier.json"
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t0
    print(f"cappelli-mod: {len(visual_matches)} visual matches, "
          f"{distributional_passes}/{len(modifier_chars)} distributional passes, "
          f"selectivity={selectivity:.2f}x ({elapsed:.1f}s)")

    return _convert(asdict(result))
