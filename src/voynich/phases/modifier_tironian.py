"""
Phase B.2 -- Modifier-Tironian Cross-Validation
=================================================
Test if Voynich modifier characters match Tironian modification marks.

Loads the 15 modifier chars and 11 syllabic chars from Phase 16's
convergent classification (modifier_integrate.json) and checks each
against the master paleographic reference.  Modifier chars should
preferentially match Tironian modifier marks (sign_type = 'modifier_mark'
or non-empty modifier_marks field), while syllabic chars should match
base signs (sign_type = 'syllable' or 'word').

Gate: >= 10 of 15 modifier candidates match Tironian modifier marks.

Dependency chain:
    results/modifier_integrate.json
    data/reference/paleographic/master_reference.json
        -> modifier_tironian.json (this step)
"""

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from voynich.core._paths import results_dir as _results_dir, data_dir as _data_dir
from voynich.core.reference import (
    EVA_VISUAL_COMPONENTS,
    load_master_reference,
)
from voynich.core.stats import cosine_similarity_triples


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


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ModifierTironianResult:
    """Cross-validation of Voynich modifiers against Tironian modifier marks."""
    n_voynich_modifiers: int
    n_modifier_match: int
    n_base_match: int
    n_no_match: int
    matches: List[Dict]
    syllabic_control_modifier_match: int
    gate_passed: bool
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def _parse_triple_key(triple_key: str) -> Dict[str, str]:
    """Parse 'first_stroke,last_stroke,glyph_class' into a dict."""
    parts = triple_key.split(',')
    if len(parts) == 3:
        return {
            'first_stroke': parts[0],
            'last_stroke': parts[1],
            'glyph_class': parts[2],
        }
    return {'first_stroke': '', 'last_stroke': '', 'glyph_class': ''}


def _is_modifier_sign(sign: Dict) -> bool:
    """Check if a reference sign is classified as a modifier mark."""
    sign_type = sign.get('sign_type', '').lower()
    if sign_type in ('modifier_mark', 'modifier', 'diacritic', 'abbreviation_mark'):
        return True
    modifier_marks = sign.get('modifier_marks', [])
    if modifier_marks:
        return True
    function_val = sign.get('function', '').lower()
    if function_val in ('modifier', 'diacritic', 'mark'):
        return True
    return False


def _is_base_sign(sign: Dict) -> bool:
    """Check if a reference sign is classified as a base syllabic sign."""
    sign_type = sign.get('sign_type', '').lower()
    if sign_type in ('syllable', 'word', 'base', 'syllabic'):
        return True
    latin_value = sign.get('latin_value', '')
    if latin_value and len(latin_value) >= 1:
        return True
    return False


def _match_char_to_reference(
    eva_char: str,
    ref_signs: List[Dict],
    similarity_threshold: float = 0.65,
) -> Tuple[str, List[str]]:
    """Match a single EVA char against reference signs.

    Returns (match_type, list_of_matching_sign_ids) where match_type is
    'MODIFIER_MATCH', 'BASE_MATCH', or 'NO_MATCH'.
    """
    comp = EVA_VISUAL_COMPONENTS.get(eva_char)
    if comp is None:
        return 'NO_MATCH', []

    v_triple = {
        'first_stroke': comp['first_stroke'],
        'last_stroke': comp['last_stroke'],
        'glyph_class': comp['glyph_class'],
    }

    modifier_matches: List[str] = []
    base_matches: List[str] = []

    for ref_sign in ref_signs:
        fs = ref_sign.get('first_stroke', '')
        ls = ref_sign.get('last_stroke', '')
        gc = ref_sign.get('glyph_class', '')
        if not (fs and ls and gc):
            continue

        r_triple = {
            'first_stroke': fs,
            'last_stroke': ls,
            'glyph_class': gc,
        }
        sim = cosine_similarity_triples(v_triple, r_triple)

        if sim >= similarity_threshold:
            sign_id = ref_sign.get('sign_id', ref_sign.get('id', '?'))
            if _is_modifier_sign(ref_sign):
                modifier_matches.append(sign_id)
            elif _is_base_sign(ref_sign):
                base_matches.append(sign_id)

    if modifier_matches:
        return 'MODIFIER_MATCH', modifier_matches
    elif base_matches:
        return 'BASE_MATCH', base_matches
    else:
        return 'NO_MATCH', []


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_modifier_tironian() -> None:
    """Phase B.2: Test if Voynich modifiers match Tironian modification marks."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE B.2: Modifier-Tironian Cross-Validation")
    print("=" * 70)

    rd = _results_dir()

    # ---- Step 1: Load modifier integration results ----
    print("\n  1. Loading modifier integration results ...")
    mi_path = os.path.join(rd, 'modifier_integrate.json')
    if not os.path.exists(mi_path):
        print("      [ERROR] modifier_integrate.json not found. Run mod-integrate first.")
        return

    with open(mi_path) as f:
        mi_data = json.load(f)

    modifier_chars = mi_data.get('modifier_chars', [])
    syllabic_chars = mi_data.get('syllabic_chars', [])

    print(f"      {len(modifier_chars)} modifier chars: {modifier_chars}")
    print(f"      {len(syllabic_chars)} syllabic chars: {syllabic_chars}")

    # ---- Step 2: Load master reference ----
    print("\n  2. Loading master paleographic reference ...")
    master_ref = load_master_reference()

    if master_ref is None:
        print("      [WARN] master_reference.json not found.")
        ref_signs = []
    else:
        ref_signs = master_ref.get('all_signs', [])
        print(f"      {len(ref_signs)} reference signs loaded")

    # ---- Step 3: Match modifier chars ----
    print("\n  3. Matching modifier chars against Tironian reference ...")
    matches: List[Dict] = []
    n_modifier_match = 0
    n_base_match = 0
    n_no_match = 0

    print(f"      {'Char':<10} {'Match Type':<18} {'Sign IDs'}")
    print("      " + "-" * 60)

    for ch in modifier_chars:
        match_type, sign_ids = _match_char_to_reference(ch, ref_signs)
        matches.append({
            'eva_char': ch,
            'match_type': match_type,
            'tironian_sign_ids': sign_ids,
        })

        if match_type == 'MODIFIER_MATCH':
            n_modifier_match += 1
        elif match_type == 'BASE_MATCH':
            n_base_match += 1
        else:
            n_no_match += 1

        id_str = ', '.join(sign_ids[:5])
        if len(sign_ids) > 5:
            id_str += f' ... (+{len(sign_ids) - 5})'
        print(f"      {ch:<10} {match_type:<18} {id_str}")

    print(f"\n      Summary: {n_modifier_match} MODIFIER_MATCH, "
          f"{n_base_match} BASE_MATCH, {n_no_match} NO_MATCH")

    # ---- Step 4: Control check on syllabic chars ----
    print("\n  4. Control: matching syllabic chars (should match base signs) ...")
    syllabic_modifier_count = 0

    print(f"      {'Char':<10} {'Match Type':<18} {'Sign IDs'}")
    print("      " + "-" * 60)

    for ch in syllabic_chars:
        match_type, sign_ids = _match_char_to_reference(ch, ref_signs)
        if match_type == 'MODIFIER_MATCH':
            syllabic_modifier_count += 1

        id_str = ', '.join(sign_ids[:5])
        if len(sign_ids) > 5:
            id_str += f' ... (+{len(sign_ids) - 5})'
        print(f"      {ch:<10} {match_type:<18} {id_str}")

    print(f"\n      Syllabic chars matching as modifiers: {syllabic_modifier_count} "
          f"(should be low)")

    # ---- Step 5: Gate ----
    gate_passed = n_modifier_match >= 10
    print(f"\n  5. Gate B.2 (>= 10 modifier matches): "
          f"{'PASS' if gate_passed else 'FAIL'} "
          f"({n_modifier_match}/15)")

    # ---- Verdict ----
    if gate_passed:
        verdict = (
            f"PASS: {n_modifier_match}/{len(modifier_chars)} modifier chars match "
            f"Tironian modifier marks. Control: {syllabic_modifier_count}/"
            f"{len(syllabic_chars)} syllabic chars match as modifiers (low = good). "
            f"Voynich modifier classification is consistent with Tironian system."
        )
    else:
        verdict = (
            f"FAIL: Only {n_modifier_match}/{len(modifier_chars)} modifier chars match "
            f"Tironian modifier marks (need >= 10). "
            f"{n_base_match} matched base signs instead, {n_no_match} had no match. "
            f"Modifier classification may not align with Tironian modification system."
        )

    print(f"\n  Verdict: {verdict}")

    # ---- Save ----
    result = ModifierTironianResult(
        n_voynich_modifiers=len(modifier_chars),
        n_modifier_match=n_modifier_match,
        n_base_match=n_base_match,
        n_no_match=n_no_match,
        matches=matches,
        syllabic_control_modifier_match=syllabic_modifier_count,
        gate_passed=gate_passed,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'modifier_tironian.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)

    print(f"\n  Results saved -> {out_path}")
