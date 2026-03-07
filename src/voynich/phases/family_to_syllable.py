"""
Phase 21.5 – Sign Family to Historical Syllable Mapping (family-syllable)
=========================================================================
Maps each Voynich sign family to a historical syllable family, producing
paleographically-grounded phonetic assignments.

Dependency chain:
    tachygraphic_stroke.json (Phase 19.5)
    + chatelain_families.json (21.3)
    + fontana_families.json (21.2)
    + eva_stroke_compare.json (21.4)
    + cross_approach.json (Phase 19.8)
        → family_to_syllable.json (this step)
"""

import json
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.reference import EVA_VISUAL_COMPONENTS


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


_VOWELS = set('aeiouæœ')
_CONSONANTS = set('bcdfghjklmnpqrstvwxyz')


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PaleoAssignment:
    eva_char: str
    consonant: Optional[str] = None
    vowel: Optional[str] = None
    latin_syllable: Optional[str] = None
    evidence: List[str] = field(default_factory=list)
    confidence: str = 'unassigned'  # high|medium|low|unassigned
    priority: int = 6              # 1=highest (anchor+paleo), 6=unassigned


@dataclass
class FamilyToSyllableResult:
    timestamp: str
    n_assignments: int
    n_assigned: int
    n_high_conf: int
    n_medium_conf: int
    n_low_conf: int
    n_unassigned: int
    coverage: float
    assignments: List[Dict[str, Any]]
    anchor_cross_reference: Dict[str, Any]
    family_mapping_summary: Dict[str, Any]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_family_to_syllable() -> Dict[str, Any]:
    """Map Voynich families to historical syllable families."""
    t0 = time.time()
    rdir = _results_dir()

    # --- Load upstream data ---
    # Phase 19.5: Voynich sign families
    tachy = _load_json(str(rdir / "tachygraphic_stroke.json")) or {}
    voynich_families = tachy.get('families', tachy.get('sign_families', []))
    if isinstance(voynich_families, dict):
        voynich_families = list(voynich_families.values())

    # Phase 21.3: Chatelain families
    chat_fam = _load_json(str(rdir / "chatelain_families.json")) or {}
    syllable_table = chat_fam.get('reference_syllable_table', [])

    # Phase 21.2: Fontana families
    font_fam = _load_json(str(rdir / "fontana_families.json")) or {}

    # Phase 21.4: Per-char stroke matches
    eva_compare = _load_json(str(rdir / "eva_stroke_compare.json")) or {}
    per_char = eva_compare.get('per_char_matches', [])
    char_matches: Dict[str, Dict] = {}
    for m in per_char:
        char_matches[m.get('eva_char', '')] = m

    # Phase 19.8: Cross-approach anchors
    anchors_data = _load_json(str(rdir / "cross_approach.json")) or {}
    anchors: Dict[str, str] = {}
    for m in anchors_data.get('mappings', []):
        if isinstance(m, dict):
            eva_tok = m.get('eva_token', '')
            latin = m.get('latin_value', '')
            if eva_tok and latin:
                anchors[eva_tok] = latin

    # Phase 15: Triple assignments (fallback)
    combined = _load_json(str(rdir / "combined_refine.json")) or {}
    triple_assignments: Dict[str, str] = {}
    for ta in combined.get('best_assignments', combined.get('assignments', [])):
        if isinstance(ta, dict):
            tk = ta.get('triple_key', '')
            syl = ta.get('syllable', ta.get('value', ''))
            if tk and syl:
                triple_assignments[tk] = syl

    # Build stroke_pattern → consonant/vowel from Chatelain syllable table
    stroke_to_consonant: Dict[str, str] = {}
    stroke_to_vowel: Dict[str, str] = {}
    for entry in syllable_table:
        sp = entry.get('stroke_pattern', '')
        if entry.get('consonant_class'):
            stroke_to_consonant[sp] = entry['consonant_class']
        if entry.get('vowel_hint'):
            stroke_to_vowel[sp] = entry['vowel_hint']

    # --- Build assignments for each EVA char ---
    assignments: List[PaleoAssignment] = []

    for eva_char, components in EVA_VISUAL_COMPONENTS.items():
        fs = components.get('first_stroke', '')
        ls = components.get('last_stroke', '')
        gc = components.get('glyph_class', '')
        triple_key = f"{fs},{ls},{gc}"

        evidence: List[str] = []
        consonant: Optional[str] = None
        vowel: Optional[str] = None
        latin_syllable: Optional[str] = None
        priority = 6
        confidence = 'unassigned'

        # --- Priority 1: Cross-approach anchor confirmed by paleographic match ---
        # Check if any anchor token starts with or equals this EVA char
        anchor_val = None
        for tok, lat in anchors.items():
            if tok == eva_char:
                anchor_val = lat
                break

        char_match = char_matches.get(eva_char, {})
        top_cands = char_match.get('top_candidates', [])
        paleo_vals = [c.get('latin_value', '') for c in top_cands if c.get('latin_value')]

        if anchor_val and any(anchor_val.lower() == pv.lower() for pv in paleo_vals):
            latin_syllable = anchor_val.lower()
            evidence.append(f"anchor_confirmed_by_paleo:{anchor_val}")
            priority = 1
            confidence = 'high'
        elif anchor_val:
            # Anchor exists but not confirmed by paleo — still useful
            latin_syllable = anchor_val.lower()
            evidence.append(f"anchor_only:{anchor_val}")
            priority = 3
            confidence = 'medium'

        # --- Priority 2: Chatelain Bobbio family match ---
        if priority > 2 and fs in stroke_to_consonant:
            c_val = stroke_to_consonant[fs]
            consonant = c_val
            evidence.append(f"chatelain_consonant:{fs}→{c_val}")
            if priority > 2:
                priority = 2
                confidence = 'high' if consonant else 'medium'

        if fs in stroke_to_vowel:
            vowel = stroke_to_vowel[fs]
            evidence.append(f"chatelain_vowel:{fs}→{vowel}")

        # --- Priority 3: Fontana construction rule match ---
        if priority > 3:
            font_families = font_fam.get('families', [])
            for ff in font_families:
                base = ff.get('base_form', '')
                # Check if Fontana base_form corresponds to this EVA char's first stroke category
                from voynich.core.reference import normalize_stroke, stroke_category
                from voynich.phases.paleo_ingest import _FONTANA_BASE_MAP
                canon_base = _FONTANA_BASE_MAP.get(base, base)
                canon_fs = normalize_stroke(fs)
                if normalize_stroke(canon_base) == canon_fs:
                    evidence.append(f"fontana_family:{base}(size={ff.get('size', 0)})")
                    if priority > 3:
                        priority = 3
                        confidence = 'medium'
                    break

        # --- Priority 4: Individual stroke match to historical sign ---
        if priority > 4 and paleo_vals:
            best_cand = top_cands[0]
            best_level = best_cand.get('match_level', 'none')
            if best_level in ('exact', 'near'):
                if not latin_syllable:
                    latin_syllable = paleo_vals[0]
                evidence.append(f"stroke_match:{paleo_vals[0]}(level={best_level})")
                if priority > 4:
                    priority = 4
                    confidence = 'medium'

        # --- Priority 5: Family propagation ---
        # If this char is in a Voynich family where another member has Priority 1-3,
        # propagate the consonant class
        # (Deferred — would need family membership lookup. Mark as TODO.)

        # --- Priority 6: Statistical fallback ---
        if priority > 5 and triple_key in triple_assignments:
            stat_val = triple_assignments[triple_key]
            if not latin_syllable:
                latin_syllable = stat_val
            evidence.append(f"statistical_fallback:{stat_val}")
            priority = 6
            confidence = 'low'

        # Build syllable from consonant + vowel if no direct assignment
        if not latin_syllable and consonant and vowel:
            latin_syllable = consonant + vowel
        elif not latin_syllable and consonant:
            latin_syllable = consonant + 'a'  # Default vowel

        assignments.append(PaleoAssignment(
            eva_char=eva_char,
            consonant=consonant,
            vowel=vowel,
            latin_syllable=latin_syllable,
            evidence=evidence,
            confidence=confidence,
            priority=priority,
        ))

    # --- Coverage stats ---
    n_assigned = sum(1 for a in assignments if a.confidence != 'unassigned')
    n_high = sum(1 for a in assignments if a.confidence == 'high')
    n_medium = sum(1 for a in assignments if a.confidence == 'medium')
    n_low = sum(1 for a in assignments if a.confidence == 'low')
    n_unassigned = sum(1 for a in assignments if a.confidence == 'unassigned')
    coverage = n_assigned / max(len(assignments), 1)

    # Anchor cross-reference
    anchor_hits = sum(1 for a in assignments if any('anchor' in e for e in a.evidence))
    paleo_confirmed = sum(1 for a in assignments if any('anchor_confirmed_by_paleo' in e for e in a.evidence))

    # Family mapping summary
    priority_counts = Counter(a.priority for a in assignments)

    result = FamilyToSyllableResult(
        timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
        n_assignments=len(assignments),
        n_assigned=n_assigned,
        n_high_conf=n_high,
        n_medium_conf=n_medium,
        n_low_conf=n_low,
        n_unassigned=n_unassigned,
        coverage=coverage,
        assignments=[_convert(asdict(a)) for a in assignments],
        anchor_cross_reference={
            'n_anchors_available': len(anchors),
            'n_anchor_hits': anchor_hits,
            'n_paleo_confirmed': paleo_confirmed,
        },
        family_mapping_summary={
            'priority_distribution': dict(priority_counts),
            'priority_1_2_3_count': sum(priority_counts.get(p, 0) for p in [1, 2, 3]),
        },
    )

    out_path = rdir / "family_to_syllable.json"
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t0
    print(f"family-syllable: {n_assigned}/{len(assignments)} assigned "
          f"(H={n_high} M={n_medium} L={n_low}), "
          f"coverage={coverage:.1%} ({elapsed:.1f}s)")

    return _convert(asdict(result))
