"""
Phase 21.2 – Fontana Family Extraction (fontana-families)
=========================================================
Extracts Fontana cipher families from base_form groupings and tests whether
his construction rules match Voynich sign family patterns.

Dependency chain:
    paleo_ingest.json (master_reference.json)
        → fontana_families.json (this step)
"""

import json
import random
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.reference import (
    EVA_VISUAL_COMPONENTS,
    normalize_stroke,
    stroke_category,
)


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


def _load_master_reference() -> Dict:
    import os
    path = "data/reference/paleographic/master_reference.json"
    if not os.path.exists(path):
        raise FileNotFoundError(f"master_reference.json not found — run paleo-ingest first")
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Fontana modifier taxonomy
# ---------------------------------------------------------------------------

# Classify added_feature values into functional categories
_MODIFIER_TAXONOMY: Dict[str, str] = {
    'tick_up': 'directional_tick',
    'tick_down': 'directional_tick',
    'tick_left': 'directional_tick',
    'tick_right': 'directional_tick',
    'tick_northeast': 'directional_tick',
    'tick_northwest': 'directional_tick',
    'tick_southeast': 'directional_tick',
    'tick_southwest': 'directional_tick',
    'stroke_through': 'stroke_through',
    'stroke_through_horizontal': 'stroke_through',
    'stroke_through_vertical': 'stroke_through',
    'stroke_through_diagonal': 'stroke_through',
    'line_through': 'stroke_through',
    'dot_above': 'dot_modification',
    'dot_below': 'dot_modification',
    'dot_inside': 'dot_modification',
    'dot_center': 'dot_modification',
    'double_dot': 'dot_modification',
    'tail': 'structural_addition',
    'tail_down': 'structural_addition',
    'hook': 'structural_addition',
    'loop': 'structural_addition',
    'crossbar': 'structural_addition',
    'double': 'structural_addition',
    'triple': 'structural_addition',
    'none': 'none',
}


def _classify_modifier(added_feature: Optional[str]) -> str:
    """Classify a Fontana added_feature into a modifier category."""
    if not added_feature:
        return 'none'
    af_lower = added_feature.lower().strip()
    if af_lower in _MODIFIER_TAXONOMY:
        return _MODIFIER_TAXONOMY[af_lower]
    # Fuzzy matching by keyword
    if 'tick' in af_lower:
        return 'directional_tick'
    if 'stroke' in af_lower or 'through' in af_lower or 'line' in af_lower:
        return 'stroke_through'
    if 'dot' in af_lower:
        return 'dot_modification'
    if any(k in af_lower for k in ('tail', 'hook', 'loop', 'cross', 'double', 'triple')):
        return 'structural_addition'
    return 'other'


# ---------------------------------------------------------------------------
# Voynich family data (from Phase 19.5)
# ---------------------------------------------------------------------------

def _load_voynich_families() -> Dict[str, Any]:
    """Load Phase 19.5 tachygraphic_stroke.json for Voynich sign families."""
    import os
    path = _results_dir() / "tachygraphic_stroke.json"
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class FontanaFamily:
    base_form: str
    members: List[str]           # sign_ids
    added_features: List[str]    # unique added_features in family
    modifier_categories: List[str]
    size: int
    source: str                  # fontana_bsb or fontana_bnf or combined


@dataclass
class GallowsRotationTest:
    tested: bool
    four_member_families: List[Dict[str, Any]]
    rotation_match: bool
    description: str


@dataclass
class ModifierToolkitComparison:
    fontana_proportions: Dict[str, float]
    voynich_modifier_types: Dict[str, int]
    alignment_score: float
    description: str


@dataclass
class FontanaFamiliesResult:
    timestamp: str
    n_families: int
    families: List[Dict[str, Any]]
    gallows_rotation_test: Dict[str, Any]
    modifier_toolkit_comparison: Dict[str, Any]
    null_selectivity: float
    gate_passed: bool


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_fontana_families() -> Dict[str, Any]:
    """Extract Fontana cipher families and test against Voynich patterns."""
    t0 = time.time()

    master = _load_master_reference()
    all_signs = master.get('all_signs', [])

    # Filter to Fontana signs only (BSB + BNF)
    fontana_signs = [s for s in all_signs if s.get('source', '').startswith('fontana')]

    # --- Group by base_form ---
    by_base: Dict[str, List[Dict]] = defaultdict(list)
    for s in fontana_signs:
        base = s.get('base_form', '') or 'unknown'
        by_base[base].append(s)

    families: List[FontanaFamily] = []
    for base, members in sorted(by_base.items()):
        features = sorted(set(
            m.get('added_feature', '') or 'none' for m in members
        ))
        mod_cats = sorted(set(_classify_modifier(f) for f in features))
        families.append(FontanaFamily(
            base_form=base,
            members=[m.get('source_id', '') for m in members],
            added_features=features,
            modifier_categories=mod_cats,
            size=len(members),
            source='combined',
        ))

    # --- Gallows rotation test ---
    # Check if any base_form group has exactly 4 members with directional ticks
    four_member_families = []
    rotation_match = False
    for fam in families:
        if fam.size == 4:
            dir_ticks = [f for f in fam.added_features
                         if _classify_modifier(f) == 'directional_tick']
            four_member_families.append({
                'base_form': fam.base_form,
                'size': fam.size,
                'features': fam.added_features,
                'directional_ticks': dir_ticks,
                'n_directional': len(dir_ticks),
            })
            if len(dir_ticks) >= 3:  # At least 3 of 4 are directional
                rotation_match = True

    # Also check for families with ≥4 directional tick variants
    for fam in families:
        dir_ticks = [f for f in fam.added_features
                     if _classify_modifier(f) == 'directional_tick']
        if len(dir_ticks) >= 4 and fam.size not in [f['size'] for f in four_member_families]:
            four_member_families.append({
                'base_form': fam.base_form,
                'size': fam.size,
                'features': fam.added_features,
                'directional_ticks': dir_ticks,
                'n_directional': len(dir_ticks),
            })
            rotation_match = True

    gallows_test = GallowsRotationTest(
        tested=len(fontana_signs) > 0,
        four_member_families=four_member_families,
        rotation_match=rotation_match,
        description=(
            f"Found {len(four_member_families)} families with 4 members. "
            f"Rotation match: {rotation_match}. "
            f"Voynich has 4 gallows (k,t,p,f) sharing ascender first_stroke."
        ),
    )

    # --- Modifier toolkit comparison ---
    fontana_mod_counts: Counter = Counter()
    for s in fontana_signs:
        cat = _classify_modifier(s.get('added_feature'))
        fontana_mod_counts[cat] += 1

    total_fontana = sum(fontana_mod_counts.values()) or 1
    fontana_proportions = {k: v / total_fontana for k, v in fontana_mod_counts.items()}

    # Load Voynich modifier categories from Phase 16
    voynich_mod_types: Dict[str, int] = {}
    try:
        mod_path = _results_dir() / "modifier_integrate.json"
        if mod_path.exists():
            with open(mod_path) as f:
                mod_data = json.load(f)
            # Count modifier categories
            for m in mod_data.get('modifier_chars', mod_data.get('confirmed_modifiers', [])):
                if isinstance(m, dict):
                    mtype = m.get('modifier_type', 'unknown')
                else:
                    mtype = 'unknown'
                voynich_mod_types[mtype] = voynich_mod_types.get(mtype, 0) + 1
    except Exception:
        pass

    # Alignment: overlap of category types
    fontana_cats = set(fontana_proportions.keys()) - {'none', 'other'}
    voynich_cats = set(voynich_mod_types.keys()) - {'unknown'}
    # Map Voynich categories to Fontana-like categories
    voynich_mapped = set()
    for vc in voynich_cats:
        vc_lower = vc.lower()
        if 'vowel' in vc_lower or 'alter' in vc_lower:
            voynich_mapped.add('directional_tick')
        elif 'nasaliz' in vc_lower or 'gemina' in vc_lower:
            voynich_mapped.add('dot_modification')
        elif 'cluster' in vc_lower or 'silent' in vc_lower:
            voynich_mapped.add('stroke_through')
        elif 'structur' in vc_lower:
            voynich_mapped.add('structural_addition')

    overlap = fontana_cats & voynich_mapped
    alignment = len(overlap) / max(len(fontana_cats | voynich_mapped), 1)

    modifier_comparison = ModifierToolkitComparison(
        fontana_proportions=fontana_proportions,
        voynich_modifier_types=voynich_mod_types,
        alignment_score=alignment,
        description=(
            f"Fontana has {len(fontana_cats)} modifier categories, "
            f"Voynich mapped to {len(voynich_mapped)}. "
            f"Overlap: {len(overlap)}/{max(len(fontana_cats | voynich_mapped), 1)} = {alignment:.2f}"
        ),
    )

    # --- Null control ---
    # Random groupings of EVA chars into families of same sizes as Fontana families
    eva_chars = list(EVA_VISUAL_COMPONENTS.keys())
    family_sizes = [f.size for f in families if f.size > 1]

    null_scores = []
    rng = random.Random(42)
    for _ in range(100):
        rng.shuffle(eva_chars)
        idx = 0
        null_overlap = 0
        null_total = 0
        for sz in family_sizes:
            group = eva_chars[idx:idx + min(sz, len(eva_chars) - idx)]
            idx += sz
            if len(group) < 2:
                continue
            # Check if group shares first_stroke
            first_strokes = set()
            for ch in group:
                comp = EVA_VISUAL_COMPONENTS.get(ch, {})
                fs = comp.get('first_stroke', '')
                if fs:
                    first_strokes.add(fs)
            null_total += 1
            if len(first_strokes) == 1:
                null_overlap += 1
        null_scores.append(null_overlap / max(null_total, 1))

    # Real score: fraction of Fontana families whose members share first_stroke correspondence
    real_overlap = 0
    real_total = 0
    for fam in families:
        if fam.size < 2:
            continue
        real_total += 1
        # All members share base_form by definition, so check if base_form maps to a single EVA first_stroke category
        canon = normalize_stroke(_FONTANA_BASE_MAP.get(fam.base_form, fam.base_form))
        cat = stroke_category(canon)
        if cat != 'unknown':
            real_overlap += 1

    real_score = real_overlap / max(real_total, 1)
    null_mean = sum(null_scores) / max(len(null_scores), 1)
    selectivity = real_score / max(null_mean, 0.01)

    result = FontanaFamiliesResult(
        timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
        n_families=len(families),
        families=[_convert(asdict(f)) for f in families],
        gallows_rotation_test=_convert(asdict(gallows_test)),
        modifier_toolkit_comparison=_convert(asdict(modifier_comparison)),
        null_selectivity=selectivity,
        gate_passed=selectivity > 1.5,
    )

    out_path = _results_dir() / "fontana_families.json"
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t0
    print(f"fontana-families: {len(families)} families, "
          f"rotation_match={rotation_match}, "
          f"selectivity={selectivity:.2f}x ({elapsed:.1f}s)")

    return _convert(asdict(result))


# Need Fontana base map for real_score computation
from voynich.phases.paleo_ingest import _FONTANA_BASE_MAP
