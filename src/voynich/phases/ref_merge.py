"""
Phase A.3b -- Merge Validated Reference Sources
================================================
Merge all validated paleographic reference sources into a master reference
table, detect sign families, and compute overlap with the 25 Voynich
stroke-feature triples from Phase 14.

Dependency chain:
    ref_validate.json  (gate: all_valid must be true)
    data/reference/tironian/*.json
    data/reference/cappelli/*.json
    data/reference/costamagna/*.json
    data/reference/fontana/*.json
    data/reference/milanese/*.json
    data/reference/ligature/*.json
    results/stroke_features.json  (Voynich triples)
        -> data/reference/paleographic/master_reference.json
        -> data/reference/paleographic/sign_families.json
        -> ref_merge.json (this step)
"""

import json
import os
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from voynich.core._paths import data_dir as _data_dir, results_dir as _results_dir
from voynich.core.reference import (
    detect_sign_families,
    load_cappelli_reference,
    load_costamagna_reference,
    load_fontana_reference,
    load_ligature_observations,
    load_milanese_reference,
    load_tironian_reference,
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


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RefMergeResult:
    n_sources_merged: int
    total_signs: int
    n_unique_triple_keys: int
    voynich_triple_overlap: int
    voynich_triple_overlap_pct: float
    voynich_triple_overlap_list: List[str]
    n_sign_families: int
    sign_family_summary: List[Dict]
    gate_passed: bool
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def _normalize_sign(
    entry: Dict[str, Any],
    source_file: str,
    sign_type: str,
) -> Dict[str, Any]:
    """Normalize a sign entry to the common master schema.

    Fields:
        source_file, sign_id, triple_key, latin_value,
        first_stroke, last_stroke, glyph_class, confidence, sign_type
    """
    sign_id = (
        entry.get('sign_id')
        or entry.get('entry_id')
        or entry.get('cipher_id', '?')
    )

    # latin_value: prefer latin_expansion, then syllable_value, then plaintext_value
    latin_value = (
        entry.get('latin_expansion')
        or entry.get('syllable_value')
        or entry.get('plaintext_value')
        or ''
    )

    first_stroke = entry.get('first_stroke', '')
    last_stroke = entry.get('last_stroke', '')
    glyph_class = entry.get('glyph_class', '')

    # Build triple_key from components if not already present
    triple_key = entry.get('triple_key', '')
    if not triple_key and first_stroke and last_stroke and glyph_class:
        triple_key = f"{first_stroke},{last_stroke},{glyph_class}"

    confidence = entry.get('confidence', 'low')

    return {
        'source_file': source_file,
        'sign_id': sign_id,
        'triple_key': triple_key,
        'latin_value': latin_value,
        'first_stroke': first_stroke,
        'last_stroke': last_stroke,
        'glyph_class': glyph_class,
        'confidence': confidence,
        'sign_type': sign_type,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_ref_merge() -> None:
    """Phase A.3b: Merge validated reference sources into master reference."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE A.3b: Merge Validated Reference Sources")
    print("=" * 70)

    rd = _results_dir()
    dd = _data_dir()

    # ─── Step 1: Load ref_validate.json and check gate ───
    print("\n  1. Checking ref_validate.json gate ...")
    validate_path = os.path.join(str(rd), 'ref_validate.json')
    if not os.path.isfile(validate_path):
        print("      ERROR: ref_validate.json not found. Run ref-validate first.")
        return

    with open(validate_path) as f:
        validate_data = json.load(f)

    if not validate_data.get('all_valid', False):
        print("      ERROR: ref_validate.json shows all_valid=false.")
        print(f"      Verdict: {validate_data.get('verdict', '?')}")
        print("      Fix validation errors before merging.")
        return

    n_sources_found = validate_data.get('n_sources_found', 0)
    print(f"      Gate PASSED: {n_sources_found} validated sources ready to merge.")

    # ─── Step 2: Load all available sources ───
    print("\n  2. Loading reference sources ...")
    all_signs: List[Dict[str, Any]] = []
    sources_merged = 0

    # Tironian (schmitz + chatelain)
    tironian_signs = load_tironian_reference(source='all')
    if tironian_signs:
        for s in tironian_signs:
            sf = s.get('source_file', 'tironian')
            all_signs.append(_normalize_sign(s, sf, 'tironian'))
        sources_merged += 1
        print(f"      Tironian: {len(tironian_signs)} signs loaded")

    # Cappelli
    cappelli_entries = load_cappelli_reference()
    if cappelli_entries:
        for e in cappelli_entries:
            all_signs.append(_normalize_sign(e, 'cappelli', 'abbreviation'))
        sources_merged += 1
        print(f"      Cappelli: {len(cappelli_entries)} entries loaded")

    # Costamagna
    costa_family, costa_unaffiliated = load_costamagna_reference()
    n_costa = len(costa_family) + len(costa_unaffiliated)
    if n_costa > 0:
        for s in costa_family:
            all_signs.append(_normalize_sign(s, 'costamagna', 'notarial'))
        for s in costa_unaffiliated:
            all_signs.append(_normalize_sign(s, 'costamagna', 'notarial'))
        sources_merged += 1
        print(f"      Costamagna: {n_costa} signs loaded "
              f"({len(costa_family)} family, {len(costa_unaffiliated)} unaffiliated)")

    # Fontana
    fontana_signs = load_fontana_reference()
    if fontana_signs:
        for s in fontana_signs:
            all_signs.append(_normalize_sign(s, 'fontana', 'cipher'))
        sources_merged += 1
        print(f"      Fontana: {len(fontana_signs)} signs loaded")

    # Milanese
    milanese_ciphers = load_milanese_reference()
    if milanese_ciphers:
        n_mil_signs = 0
        for cipher in milanese_ciphers:
            cipher_id = cipher.get('cipher_id', '?')
            cipher_signs = cipher.get('signs', [])
            for s in cipher_signs:
                s_copy = dict(s)
                s_copy['cipher_id'] = cipher_id
                if 'sign_id' not in s_copy:
                    s_copy['sign_id'] = f"{cipher_id}_{n_mil_signs}"
                all_signs.append(_normalize_sign(s_copy, 'milanese', 'cipher'))
                n_mil_signs += 1
        if n_mil_signs > 0:
            sources_merged += 1
            print(f"      Milanese: {n_mil_signs} signs from "
                  f"{len(milanese_ciphers)} cipher key(s)")

    # Ligature observations (not sign entries, but included for completeness)
    ligature_data = load_ligature_observations()
    if ligature_data:
        sources_merged += 1
        n_pairs = len(ligature_data.get('pair_summaries', []))
        print(f"      Ligature: {n_pairs} pair observations loaded (metadata only)")

    print(f"\n      Total: {len(all_signs)} signs from {sources_merged} sources")

    # ─── Step 3: Normalize entries to common schema ───
    # (Already done during loading above via _normalize_sign)
    print("\n  3. Normalization complete. Building indices ...")

    # ─── Step 4: Build master sign table indexed by triple_key ───
    by_triple_key: Dict[str, List[Dict]] = defaultdict(list)
    for sign in all_signs:
        tk = sign.get('triple_key', '')
        if tk:
            by_triple_key[tk].append(sign)

    n_unique_triples = len(by_triple_key)
    n_with_triple = sum(1 for s in all_signs if s.get('triple_key'))
    print(f"      {n_with_triple}/{len(all_signs)} signs have triple_key")
    print(f"      {n_unique_triples} unique triple_keys in reference")

    # ─── Step 5: Load stroke_features.json and compute overlap ───
    print("\n  4. Computing overlap with Voynich triples ...")
    stroke_features_path = os.path.join(str(rd), 'stroke_features.json')
    voynich_triples: List[str] = []

    if os.path.isfile(stroke_features_path):
        with open(stroke_features_path) as f:
            sf_data = json.load(f)
        # Extract the 25 Voynich triple_keys
        variables = sf_data.get('feature_variables', [])
        if variables:
            voynich_triples = [v.get('triple_key', '') for v in variables
                               if v.get('triple_key')]
        # Fallback: try triple_keys list
        if not voynich_triples:
            voynich_triples = sf_data.get('triple_keys', [])
        print(f"      Loaded {len(voynich_triples)} Voynich triples from "
              f"stroke_features.json")
    else:
        print("      WARNING: stroke_features.json not found. "
              "Overlap computation skipped.")

    overlap_list = sorted(tk for tk in voynich_triples if tk in by_triple_key)
    overlap_count = len(overlap_list)
    overlap_pct = (
        round(100.0 * overlap_count / len(voynich_triples), 1)
        if voynich_triples else 0.0
    )

    print(f"      Overlap: {overlap_count}/{len(voynich_triples)} Voynich "
          f"triples found in reference ({overlap_pct}%)")
    if overlap_list:
        for tk in overlap_list:
            n_ref = len(by_triple_key[tk])
            print(f"        {tk}  ({n_ref} reference signs)")

    # ─── Step 6: Detect sign families ───
    print("\n  5. Detecting sign families ...")
    # Filter to signs that have stroke fields for family detection
    signs_with_strokes = [
        s for s in all_signs
        if s.get('first_stroke') and s.get('first_stroke') != 'unclear'
    ]
    families = detect_sign_families(signs_with_strokes)
    print(f"      {len(families)} sign families detected "
          f"from {len(signs_with_strokes)} signs with stroke fields")

    family_summary: List[Dict] = []
    for fam in families:
        summary = {
            'family_id': fam['family_id'],
            'common_first_stroke': fam['common_first_stroke'],
            'n_members': fam['n_members'],
            'n_distinct_triples': fam.get('n_distinct_triples', 0),
        }
        family_summary.append(summary)
        print(f"        {fam['family_id']}: first_stroke={fam['common_first_stroke']}, "
              f"{fam['n_members']} members, "
              f"{fam.get('n_distinct_triples', 0)} distinct triples")

    # ─── Step 7: Save outputs ───
    print("\n  6. Saving outputs ...")

    # Ensure paleographic directory exists
    paleo_dir = os.path.join(str(dd), 'reference', 'paleographic')
    os.makedirs(paleo_dir, exist_ok=True)

    # Save master_reference.json
    master_ref = {
        'generated_date': datetime.utcnow().isoformat(),
        'n_sources': sources_merged,
        'all_signs': all_signs,
        'by_triple_key': {k: v for k, v in sorted(by_triple_key.items())},
        'voynich_overlapping_triples': overlap_list,
    }
    master_path = os.path.join(paleo_dir, 'master_reference.json')
    with open(master_path, 'w') as f:
        json.dump(_convert(master_ref), f, indent=2)
    print(f"      -> {master_path}")

    # Save sign_families.json
    families_path = os.path.join(paleo_dir, 'sign_families.json')
    with open(families_path, 'w') as f:
        json.dump(_convert(families), f, indent=2)
    print(f"      -> {families_path}")

    # ─── Gate decision ───
    gate_passed = overlap_count > 0

    if gate_passed:
        verdict = (
            f"PASS: {overlap_count}/{len(voynich_triples)} Voynich triples "
            f"have reference matches ({overlap_pct}%). "
            f"{len(all_signs)} signs merged from {sources_merged} sources. "
            f"{len(families)} sign families detected."
        )
    else:
        if not voynich_triples:
            verdict = (
                f"FAIL: stroke_features.json not available -- cannot compute "
                f"Voynich triple overlap. {len(all_signs)} signs merged from "
                f"{sources_merged} sources."
            )
        else:
            verdict = (
                f"FAIL: 0/{len(voynich_triples)} Voynich triples found in "
                f"reference. {len(all_signs)} signs merged from "
                f"{sources_merged} sources, but no overlap with Voynich "
                f"feature decomposition."
            )

    print(f"\n  Gate: {'PASS' if gate_passed else 'FAIL'}")
    print(f"  {verdict}")

    # Save ref_merge.json to results/
    result = RefMergeResult(
        n_sources_merged=sources_merged,
        total_signs=len(all_signs),
        n_unique_triple_keys=n_unique_triples,
        voynich_triple_overlap=overlap_count,
        voynich_triple_overlap_pct=overlap_pct,
        voynich_triple_overlap_list=overlap_list,
        n_sign_families=len(families),
        sign_family_summary=family_summary,
        gate_passed=gate_passed,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(str(rd), 'ref_merge.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)

    print(f"\n  -> {out_path}")
