"""
Step 40.14 – Predicted Form Generation
========================================
For each botanical folio with a confident plant identification, generate
the predicted partial EVA form and rank by how much is known.

Dependency chain:
    drosera_constraints.json    (Step 40.13)
    italian_botanical_csp.json  (Step 39.9)
    italian_plant_names.json    (Step 39.8)
    combined_refine.json        (Step 15)
        → botanical_predictions.json  (this step)
"""

import json
import os
import re
import time
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
# Core: Syllabification and prediction
# ---------------------------------------------------------------------------

_VOWELS = set('aeiou')

def _syllabify_italian(word: str) -> List[str]:
    """Simple Italian syllabification."""
    word = word.lower()
    syllables = []
    current = ''
    for i, ch in enumerate(word):
        current += ch
        if ch in _VOWELS:
            # Check if next char starts a new syllable
            remaining = word[i + 1:]
            if remaining:
                # Count consonants before next vowel
                n_cons = 0
                for c in remaining:
                    if c not in _VOWELS:
                        n_cons += 1
                    else:
                        break
                if n_cons >= 2:
                    # Keep one consonant, start new syllable with rest
                    current += remaining[0] if remaining else ''
                    syllables.append(current)
                    current = ''
                    # Skip the consonant we took
                    continue
                elif n_cons == 1 or n_cons == 0:
                    syllables.append(current)
                    current = ''
            else:
                syllables.append(current)
                current = ''
    if current:
        if syllables:
            syllables[-1] += current
        else:
            syllables.append(current)
    return syllables if syllables else [word]


def _invert_assignment(
    syllable: str,
    assignment: Dict[str, str],
) -> List[str]:
    """Find triple keys that map to a given syllable in the assignment."""
    return [k for k, v in assignment.items() if v == syllable]


def _predict_eva_form(
    italian_name: str,
    assignment: Dict[str, str],
    drosera_constraints: List[Dict],
) -> Dict:
    """Predict partial EVA form for an Italian plant name."""
    syllables = _syllabify_italian(italian_name)

    # Build constraint map from Drosera
    drosera_map = {}
    for c in drosera_constraints:
        if c.get('consistent_with_phase15'):
            drosera_map[c['syllable']] = c['triple_key']

    predicted_triples = []
    known_positions = []
    unknown_positions = []

    for i, syl in enumerate(syllables):
        # Check Drosera constraints first
        if syl in drosera_map:
            predicted_triples.append(drosera_map[syl])
            known_positions.append(i)
        else:
            # Check main assignment
            matching_triples = _invert_assignment(syl, assignment)
            if matching_triples:
                predicted_triples.append(matching_triples[0])
                known_positions.append(i)
            else:
                predicted_triples.append('?')
                unknown_positions.append(i)

    known_fraction = len(known_positions) / len(syllables) if syllables else 0.0

    return {
        'italian_name': italian_name,
        'syllables': syllables,
        'predicted_triples': predicted_triples,
        'known_positions': known_positions,
        'unknown_positions': unknown_positions,
        'known_fraction': round(known_fraction, 4),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_botanical_predictions() -> None:
    """Step 40.14: Predicted Form Generation."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 40.14: Predicted Form Generation")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load inputs ──
    print("\n  1. Loading inputs …")
    drosera_data = _safe_load(os.path.join(rd, 'drosera_constraints.json'))
    bot_csp = _safe_load(os.path.join(rd, 'italian_botanical_csp.json'))
    plant_names = _safe_load(os.path.join(rd, 'italian_plant_names.json'))
    refine = _safe_load(os.path.join(rd, 'combined_refine.json'))

    assignment = refine.get('best_assignment', {})
    drosera_constraints = drosera_data.get('confirmed_constraints', [])
    print(f"    Phase 15 assignment: {len(assignment)} triples")
    print(f"    Drosera constraints: {len(drosera_constraints)}")

    # ── 2. Get plant identifications ──
    print("\n  2. Loading plant identifications …")
    # Try multiple sources for plant names
    plant_list = []
    # From Phase 39.8 italian_plant_names.json
    for entry in plant_names.get('folio_plants', plant_names.get('plants', [])):
        if isinstance(entry, dict):
            folio = entry.get('folio', '')
            name = entry.get('italian_name', entry.get('name', ''))
            tier = entry.get('tier', entry.get('confidence', ''))
            if folio and name:
                plant_list.append({'folio': folio, 'italian_name': name, 'tier': tier})

    # From Phase 31 consensus_plants
    if not plant_list:
        cons_plants = _safe_load(os.path.join(rd, 'consensus_plants.json'))
        for entry in cons_plants.get('concordance', []):
            if isinstance(entry, dict):
                folio = entry.get('folio', '')
                names = entry.get('italian_names', entry.get('names', []))
                tier = entry.get('tier', '')
                if folio and names:
                    name = names[0] if isinstance(names, list) else names
                    plant_list.append({'folio': folio, 'italian_name': name, 'tier': tier})

    print(f"    Plant identifications: {len(plant_list)}")

    # ── 3. Generate predictions ──
    print("\n  3. Generating predictions …")
    predictions = []
    for plant in plant_list:
        pred = _predict_eva_form(
            plant['italian_name'], assignment, drosera_constraints,
        )
        pred['folio'] = plant['folio']
        pred['tier'] = plant['tier']
        predictions.append(pred)

    # Sort by known fraction (most constrained first)
    predictions.sort(key=lambda p: p['known_fraction'], reverse=True)

    n_high = sum(1 for p in predictions if p['known_fraction'] >= 0.7)
    n_partial = sum(1 for p in predictions if 0.3 <= p['known_fraction'] < 0.7)
    n_low = sum(1 for p in predictions if p['known_fraction'] < 0.3)
    print(f"    High confidence (≥70% known): {n_high}")
    print(f"    Partial (30-70% known): {n_partial}")
    print(f"    Low (<30% known): {n_low}")

    for p in predictions[:5]:
        print(f"    {p['folio']}: {p['italian_name']} → "
              f"{'/'.join(p['predicted_triples'])} "
              f"({p['known_fraction']:.0%} known)")

    # ── 4. Save ──
    elapsed = time.time() - t0

    output = {
        'n_plants_analyzed': len(plant_list),
        'n_predictions_total': len(predictions),
        'n_high_confidence': n_high,
        'n_partial': n_partial,
        'n_low': n_low,
        'predictions': predictions,
        'by_folio_top_prediction': {
            p['folio']: {
                'italian_name': p['italian_name'],
                'known_fraction': p['known_fraction'],
                'syllables': p['syllables'],
            }
            for p in predictions
        },
        'runtime_seconds': round(elapsed, 1),
    }

    out_path = os.path.join(rd, 'botanical_predictions.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(output), f, indent=2)
    print(f"\n  Saved → {out_path} ({elapsed:.1f}s)")
