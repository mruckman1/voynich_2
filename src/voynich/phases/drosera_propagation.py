"""
Step 41.14 -- Drosera Propagation
===================================
Use any Drosera-derived constraints plus the Phase 15 assignment table
to predict EVA forms of other Italian plant names.  For each plant with
a known Italian name, syllabify it, look up which syllables appear in
the assignment table, and record predicted triples.

Dependency chain:
    botanical_data_fix.json    (Step 41.13)
    combined_refine.json       (Phase 15)
    modifier_integrate.json    (Phase 16)
        -> drosera_propagation.json  (this step)
"""

import json
import os
import re
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import build_eva_to_triple_lookup
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
# Italian syllabification
# ---------------------------------------------------------------------------

def _syllabify_italian(name: str) -> List[str]:
    """Simple CV syllabification for Italian words."""
    name = name.lower().strip()
    # Remove non-alpha
    name = re.sub(r'[^a-z]', '', name)
    if not name:
        return []
    # Use regex: consonant cluster followed by a vowel
    syllables = re.findall(r'[bcdfghjklmnpqrstvwxyz]*[aeiou]+', name)
    # Attach any trailing consonants to the last syllable
    consumed = sum(len(s) for s in syllables)
    if consumed < len(name) and syllables:
        syllables[-1] += name[consumed:]
    elif consumed < len(name) and not syllables:
        syllables = [name]
    return syllables


# ---------------------------------------------------------------------------
# Assignment inversion
# ---------------------------------------------------------------------------

def _invert_assignment(assignment: Dict[str, str]) -> Dict[str, List[str]]:
    """Build syllable -> list of triple_keys from the assignment table."""
    syl_to_triples: Dict[str, List[str]] = defaultdict(list)
    for triple_key, syllable in assignment.items():
        syl_to_triples[syllable].append(triple_key)
    return dict(syl_to_triples)


def _build_triple_to_eva(eva_to_triple: Dict[str, str]) -> Dict[str, List[str]]:
    """Build triple_key -> list of EVA glyphs."""
    triple_to_eva: Dict[str, List[str]] = defaultdict(list)
    for glyph, triple_key in eva_to_triple.items():
        triple_to_eva[triple_key].append(glyph)
    return dict(triple_to_eva)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def _predict_eva_form(
    italian_name: str,
    syllables: List[str],
    assignment: Dict[str, str],
    syl_to_triples: Dict[str, List[str]],
    alignment_constraints: List[Dict],
) -> Dict:
    """Predict partial EVA form for a single Italian plant name.

    For each syllable in the syllabified name:
      1. Check alignment constraints (from Drosera) first.
      2. Fall back to the Phase 15 assignment table inversion.
      3. If neither matches, mark the position as unknown ('?').
    """
    # Build a syllable -> triple_key map from alignment constraints
    constraint_map: Dict[str, str] = {}
    for c in alignment_constraints:
        syl = c.get('syllable', '')
        tk = c.get('triple_key', '')
        if syl and tk:
            # Prefer the highest-scoring alignment
            if syl not in constraint_map:
                constraint_map[syl] = tk

    predicted_triples: List[str] = []
    known_positions: List[int] = []
    unknown_positions: List[int] = []
    sources: List[str] = []  # 'constraint', 'assignment', 'unknown'

    for i, syl in enumerate(syllables):
        # Try alignment constraints first
        if syl in constraint_map:
            predicted_triples.append(constraint_map[syl])
            known_positions.append(i)
            sources.append('constraint')
        # Try assignment table inversion
        elif syl in syl_to_triples:
            # Pick the first matching triple
            predicted_triples.append(syl_to_triples[syl][0])
            known_positions.append(i)
            sources.append('assignment')
        else:
            predicted_triples.append('?')
            unknown_positions.append(i)
            sources.append('unknown')

    known_fraction = (
        len(known_positions) / len(syllables) if syllables else 0.0
    )

    return {
        'italian_name': italian_name,
        'syllables': syllables,
        'n_syllables': len(syllables),
        'predicted_triples': predicted_triples,
        'known_positions': known_positions,
        'unknown_positions': unknown_positions,
        'sources': sources,
        'known_fraction': round(known_fraction, 4),
    }


def _predict_eva_glyphs(
    predicted_triples: List[str],
    triple_to_eva: Dict[str, List[str]],
) -> List[List[str]]:
    """For each predicted triple, list the EVA glyphs that map to it."""
    result = []
    for tk in predicted_triples:
        if tk == '?':
            result.append([])
        else:
            result.append(triple_to_eva.get(tk, []))
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_drosera_propagation() -> None:
    """Step 41.14: Propagate Drosera constraints to predict plant EVA forms."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 41.14: Drosera Propagation")
    print("=" * 70)

    rd = _results_dir()

    # -- 1. Load inputs --
    print("\n  1. Loading inputs ...")

    data_fix = _safe_load(os.path.join(rd, 'botanical_data_fix.json'))
    if not data_fix:
        print("     SKIP: botanical_data_fix.json not found.")
        print("     Saving minimal output.")
        output = {
            'skip_reason': 'botanical_data_fix.json not found',
            'n_plants_analyzed': 0,
            'predictions': [],
            'runtime_seconds': round(time.time() - t0, 1),
        }
        out_path = os.path.join(rd, 'drosera_propagation.json')
        with open(out_path, 'w') as f:
            json.dump(_convert(output), f, indent=2)
        print(f"\n  Saved -> {out_path}")
        return

    refine = _safe_load(os.path.join(rd, 'combined_refine.json'))
    assignment = refine.get('best_assignment', {})
    if not assignment:
        print("     WARNING: combined_refine.json missing or no best_assignment")

    mod_data = _safe_load(os.path.join(rd, 'modifier_integrate.json'))
    modifier_chars: Set[str] = set(mod_data.get('modifier_chars', []))

    unified_map = data_fix.get('unified_plant_map', {})
    alignment_constraints = data_fix.get('alignment_constraints', [])

    print(f"     Unified plant map: {len(unified_map)} folios")
    print(f"     Phase 15 assignment: {len(assignment)} triples")
    print(f"     Alignment constraints: {len(alignment_constraints)}")
    print(f"     Modifier chars: {len(modifier_chars)}")

    # -- 2. Build lookup tables --
    print("\n  2. Building lookup tables ...")

    eva_to_triple = build_eva_to_triple_lookup()
    syl_to_triples = _invert_assignment(assignment)
    triple_to_eva = _build_triple_to_eva(eva_to_triple)

    print(f"     EVA-to-triple lookup: {len(eva_to_triple)} glyphs")
    print(f"     Syllable-to-triples: {len(syl_to_triples)} syllables covered")
    print(f"     Triple-to-EVA: {len(triple_to_eva)} triples")

    # Show which syllables are known
    known_syls = sorted(syl_to_triples.keys())
    print(f"     Known syllables: {known_syls[:20]}{'...' if len(known_syls) > 20 else ''}")

    # -- 3. Generate predictions for each plant --
    print("\n  3. Generating predictions ...")

    predictions: List[Dict] = []
    n_skipped = 0

    for folio in sorted(unified_map.keys()):
        entry = unified_map[folio]
        italian_names = entry.get('italian_names', [])
        syllabified = entry.get('syllabified', {})
        tier = entry.get('tier', '?')
        latin_name = entry.get('latin_name', '')

        if not italian_names:
            n_skipped += 1
            continue

        for name in italian_names:
            # Skip multi-word names
            if ' ' in name:
                continue

            # Get syllabification: prefer pre-computed, fall back to regex
            syls = syllabified.get(name, [])
            if not syls:
                syls = _syllabify_italian(name)
            if not syls:
                continue

            pred = _predict_eva_form(
                name, syls, assignment,
                syl_to_triples, alignment_constraints,
            )
            pred['folio'] = folio
            pred['tier'] = tier
            pred['latin_name'] = latin_name

            # Also get possible EVA glyphs for each position
            possible_eva = _predict_eva_glyphs(
                pred['predicted_triples'], triple_to_eva,
            )
            pred['possible_eva_per_position'] = possible_eva

            predictions.append(pred)

    # Sort by known_fraction descending
    predictions.sort(key=lambda p: (-p['known_fraction'], p['folio']))

    n_total = len(predictions)
    n_high = sum(1 for p in predictions if p['known_fraction'] >= 0.75)
    n_medium = sum(
        1 for p in predictions
        if 0.5 <= p['known_fraction'] < 0.75
    )
    n_low = sum(1 for p in predictions if p['known_fraction'] < 0.5)

    print(f"     Plants without Italian names (skipped): {n_skipped}")
    print(f"     Total predictions: {n_total}")
    print(f"     High confidence (>=75% known): {n_high}")
    print(f"     Medium confidence (50-75% known): {n_medium}")
    print(f"     Low confidence (<50% known): {n_low}")

    # -- 4. Show top predictions --
    print("\n  4. Top predictions (by known_fraction) ...")

    for p in predictions[:10]:
        triples_str = ' / '.join(p['predicted_triples'])
        syls_str = '-'.join(p['syllables'])
        src_str = ','.join(p['sources'])
        print(f"     {p['folio']} ({p['tier']}): {p['italian_name']} "
              f"[{syls_str}] -> [{triples_str}] "
              f"({p['known_fraction']:.0%} known, sources={src_str})")

    # -- 5. Compute coverage statistics --
    print("\n  5. Coverage statistics ...")

    all_syllables_seen: Set[str] = set()
    all_syllables_known: Set[str] = set()
    for p in predictions:
        for i, syl in enumerate(p['syllables']):
            all_syllables_seen.add(syl)
            if i in p['known_positions']:
                all_syllables_known.add(syl)

    all_syllables_unknown = all_syllables_seen - all_syllables_known
    print(f"     Unique syllables across all plants: {len(all_syllables_seen)}")
    print(f"     Syllables with known triple mapping: {len(all_syllables_known)}")
    print(f"     Syllables unknown: {len(all_syllables_unknown)}")
    if all_syllables_unknown:
        print(f"     Unknown syllables: "
              f"{sorted(all_syllables_unknown)[:15]}"
              f"{'...' if len(all_syllables_unknown) > 15 else ''}")

    # -- 6. Save --
    elapsed = time.time() - t0

    output = {
        'n_plants_analyzed': n_total,
        'n_skipped_no_italian': n_skipped,
        'n_high_confidence': n_high,
        'n_medium_confidence': n_medium,
        'n_low_confidence': n_low,
        'predictions': predictions,
        'n_unique_syllables_seen': len(all_syllables_seen),
        'n_unique_syllables_known': len(all_syllables_known),
        'unknown_syllables': sorted(all_syllables_unknown),
        'known_syllables': sorted(all_syllables_known),
        'n_alignment_constraints_used': len(alignment_constraints),
        'n_assignment_triples': len(assignment),
        'runtime_seconds': round(elapsed, 1),
    }

    out_path = os.path.join(rd, 'drosera_propagation.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(output), f, indent=2)
    print(f"\n  Saved -> {out_path} ({elapsed:.1f}s)")
