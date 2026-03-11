"""
Step 40.13 – Drosera Constraint Extraction
===========================================
Extract triple-level constraints from the confirmed f56r/Drosera alignment
and propagate through sign families.

Dependency chain:
    italian_botanical_csp.json  (Step 39.9)
    combined_refine.json        (Step 15)
        → drosera_constraints.json  (this step)
"""

import json
import os
import time
from typing import Any, Dict, List, Optional, Set

from voynich.core._paths import results_dir as _results_dir
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
# Core: Constraint extraction
# ---------------------------------------------------------------------------

def _extract_alignment_constraints(
    alignments: List[Dict],
    best_assignment: Dict[str, str],
) -> List[Dict]:
    """Extract triple→syllable constraints from botanical alignments.

    For each alignment, check whether the implied triple→syllable mapping
    is consistent with the Phase 15 best_assignment.
    """
    constraints = []
    for aln in alignments:
        folio = aln.get('folio', '')
        plant = aln.get('plant_name', aln.get('italian_name', ''))
        score = aln.get('score', 0.0)
        triple_map = aln.get('triple_assignments', aln.get('mapping', {}))

        for triple_key, syllable in triple_map.items():
            if not triple_key or not syllable:
                continue
            phase15_syl = best_assignment.get(triple_key, '')
            consistent = (phase15_syl == syllable) if phase15_syl else None

            constraints.append({
                'triple_key': triple_key,
                'syllable': syllable,
                'source_folio': folio,
                'source_plant': plant,
                'alignment_score': score,
                'phase15_syllable': phase15_syl,
                'consistent_with_phase15': consistent,
            })

    return constraints


def _build_triple_to_eva(eva_components: Dict) -> Dict[str, List[str]]:
    """Build reverse map: triple_key → list of EVA glyphs sharing it."""
    triple_to_eva: Dict[str, List[str]] = {}
    for glyph, comp in eva_components.items():
        triple_key = f"{comp['first_stroke']},{comp['last_stroke']},{comp['glyph_class']}"
        if triple_key not in triple_to_eva:
            triple_to_eva[triple_key] = []
        triple_to_eva[triple_key].append(glyph)
    return triple_to_eva


def _propagate_family_constraints(
    confirmed: List[Dict],
    triple_to_eva: Dict[str, List[str]],
    eva_components: Dict,
) -> List[Dict]:
    """For confirmed constraints, find all EVA glyphs sharing that triple
    and note the family implications."""
    implications = []
    for c in confirmed:
        triple_key = c['triple_key']
        glyphs = triple_to_eva.get(triple_key, [])

        # Find sign family: glyphs sharing same first_stroke
        parts = triple_key.split(',')
        if len(parts) >= 1:
            first_stroke = parts[0]
            family_members = [
                g for g, comp in eva_components.items()
                if comp.get('first_stroke') == first_stroke
            ]
        else:
            family_members = []

        implications.append({
            'triple_key': triple_key,
            'syllable': c['syllable'],
            'eva_glyphs': glyphs,
            'n_glyphs': len(glyphs),
            'onset_family_members': family_members,
            'n_family_members': len(family_members),
        })

    return implications


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_drosera_constraints() -> None:
    """Step 40.13: Drosera Constraint Extraction."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 40.13: Drosera Constraint Extraction")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load inputs ──
    print("\n  1. Loading inputs …")
    bot_csp = _safe_load(os.path.join(rd, 'italian_botanical_csp.json'))
    refine = _safe_load(os.path.join(rd, 'combined_refine.json'))
    best_assignment = refine.get('best_assignment', {})
    print(f"    Phase 15 assignment: {len(best_assignment)} triples")

    # ── 2. Extract alignments ──
    print("\n  2. Extracting Drosera alignments …")
    # italian_botanical_csp.json stores alignments per folio
    alignments = []
    for folio_result in bot_csp.get('folio_results', []):
        folio = folio_result.get('folio', '')
        for aln in folio_result.get('valid_alignments', []):
            aln['folio'] = folio
            alignments.append(aln)

    # Also check top-level alignments if present
    for aln in bot_csp.get('alignments', []):
        if aln not in alignments:
            alignments.append(aln)

    print(f"    Found {len(alignments)} alignments")

    # ── 3. Extract constraints ──
    print("\n  3. Extracting triple constraints …")
    all_constraints = _extract_alignment_constraints(alignments, best_assignment)
    n_consistent = sum(1 for c in all_constraints
                       if c['consistent_with_phase15'] is True)
    n_conflicting = sum(1 for c in all_constraints
                        if c['consistent_with_phase15'] is False)
    n_novel = sum(1 for c in all_constraints
                  if c['consistent_with_phase15'] is None)
    print(f"    Total constraints: {len(all_constraints)}")
    print(f"    Consistent with Phase 15: {n_consistent}")
    print(f"    Conflicting: {n_conflicting}")
    print(f"    Novel (triple not in Phase 15): {n_novel}")

    # Separate confirmed (consistent) from tension points
    confirmed = [c for c in all_constraints
                 if c['consistent_with_phase15'] is True]
    tension = [c for c in all_constraints
               if c['consistent_with_phase15'] is False]

    # ── 4. Propagate through sign families ──
    print("\n  4. Propagating through sign families …")
    triple_to_eva = _build_triple_to_eva(EVA_VISUAL_COMPONENTS)
    implications = _propagate_family_constraints(
        confirmed, triple_to_eva, EVA_VISUAL_COMPONENTS,
    )
    print(f"    Family implications from {len(implications)} confirmed constraints")

    # ── 5. Compute confidence ──
    total_testable = n_consistent + n_conflicting
    drosera_confidence = (n_consistent / total_testable
                          if total_testable > 0 else 0.0)
    print(f"    Drosera confidence: {drosera_confidence:.3f} "
          f"({n_consistent}/{total_testable} consistent)")

    # ── 6. Save ──
    elapsed = time.time() - t0

    output = {
        'n_alignments_analyzed': len(alignments),
        'n_total_constraints': len(all_constraints),
        'n_confirmed_constraints': n_consistent,
        'n_conflicting': n_conflicting,
        'n_novel': n_novel,
        'confirmed_constraints': confirmed,
        'tension_points': tension,
        'family_implications': implications,
        'drosera_confidence': round(drosera_confidence, 4),
        'verdict': ('CONSISTENT' if n_conflicting == 0 and n_consistent > 0
                    else 'TENSION' if n_conflicting > 0
                    else 'NO_DATA'),
        'runtime_seconds': round(elapsed, 1),
    }

    out_path = os.path.join(rd, 'drosera_constraints.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(output), f, indent=2)
    print(f"\n  Saved → {out_path} ({elapsed:.1f}s)")
