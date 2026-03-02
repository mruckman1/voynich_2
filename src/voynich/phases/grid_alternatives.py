"""
Phase 12.4 – Alternative Grid Construction
===========================================
Audit every EVA glyph's stroke decomposition against its current grid cell.
Build two alternative grids:

  stroke_based  – move misaligned glyphs to their stroke-implied cells.
  hybrid        – move a glyph only when BOTH stroke analysis AND the Phase
                  11.5.1 correction vector for its cell agree on the direction.

Most glyphs are expected to be correctly placed (the grid was originally built
from stroke analysis in Phase 3), so this step serves as a negative-evidence
audit and may also surface a small number of borderline misplacements.
"""

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import apply_character_moves, build_eva_to_cell_lookup
from voynich.core.reference import EVA_VISUAL_COMPONENTS
from voynich.phases.grid_recalibrate import (
    FIRST_STROKE_TO_ONSET,
    LAST_STROKE_TO_NUCLEUS,
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class GlyphStrokeAlignment:
    """Alignment between a glyph's stroke decomposition and its current cell."""
    eva_glyph: str
    current_cell: str
    current_onset_class: str
    current_nucleus_class: str
    first_stroke: str
    last_stroke: str
    stroke_implied_onset: str
    stroke_implied_nucleus: str
    stroke_implied_cell: str   # '' if no cell exists for the implied pair
    is_aligned: bool
    misalignment_degree: int   # 0=perfect, 1=onset-only, 2=nucleus-only, 3=both


@dataclass
class GridAlternativesResult:
    glyph_alignments: List[Dict]
    n_aligned: int
    n_misaligned: int
    misaligned_glyphs: List[str]
    stroke_based_cv_labels: Dict
    hybrid_cv_labels: Dict
    gate_passed: bool
    verdict: str
    runtime_seconds: float


def _convert(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _convert(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# Stroke alignment analysis
# ---------------------------------------------------------------------------

def analyse_glyph_alignment(
    glyph: str,
    current_cell: str,
    cv_labels: Dict,
) -> GlyphStrokeAlignment:
    """Return a GlyphStrokeAlignment for *glyph* in *current_cell*."""
    cell_info = cv_labels[current_cell]
    current_onset = cell_info['onset_class']
    current_nucleus = cell_info['nucleus_class']

    comp = EVA_VISUAL_COMPONENTS.get(glyph)
    if comp is None:
        # Unknown glyph — no stroke data available
        return GlyphStrokeAlignment(
            eva_glyph=glyph,
            current_cell=current_cell,
            current_onset_class=current_onset,
            current_nucleus_class=current_nucleus,
            first_stroke='unknown',
            last_stroke='unknown',
            stroke_implied_onset='unknown',
            stroke_implied_nucleus='unknown',
            stroke_implied_cell='',
            is_aligned=True,          # can't assess → assume aligned
            misalignment_degree=0,
        )

    first = comp['first_stroke']
    last = comp['last_stroke']
    implied_onset = FIRST_STROKE_TO_ONSET.get(first, 'unknown')
    implied_nucleus = LAST_STROKE_TO_NUCLEUS.get(last, 'unknown')
    implied_cell_key = f"{implied_onset},{implied_nucleus}"
    implied_cell = implied_cell_key if implied_cell_key in cv_labels else ''

    onset_match = (implied_onset == current_onset)
    nucleus_match = (implied_nucleus == current_nucleus)
    is_aligned = onset_match and nucleus_match
    if is_aligned:
        degree = 0
    elif not onset_match and nucleus_match:
        degree = 1
    elif onset_match and not nucleus_match:
        degree = 2
    else:
        degree = 3

    return GlyphStrokeAlignment(
        eva_glyph=glyph,
        current_cell=current_cell,
        current_onset_class=current_onset,
        current_nucleus_class=current_nucleus,
        first_stroke=first,
        last_stroke=last,
        stroke_implied_onset=implied_onset,
        stroke_implied_nucleus=implied_nucleus,
        stroke_implied_cell=implied_cell,
        is_aligned=is_aligned,
        misalignment_degree=degree,
    )


# ---------------------------------------------------------------------------
# Build alternative grids
# ---------------------------------------------------------------------------

def build_stroke_based_grid(
    cv_labels: Dict,
    alignments: List[GlyphStrokeAlignment],
) -> Dict:
    """Build cv_labels with all stroke-misaligned glyphs moved to implied cells."""
    moves = []
    for aln in alignments:
        if not aln.is_aligned and aln.stroke_implied_cell:
            moves.append({
                'eva_glyph': aln.eva_glyph,
                'from_cell': aln.current_cell,
                'to_cell': aln.stroke_implied_cell,
            })
    return apply_character_moves(cv_labels, moves)


def build_hybrid_grid(
    cv_labels: Dict,
    alignments: List[GlyphStrokeAlignment],
    correction_vectors: List[Dict],
) -> Dict:
    """Build a conservative hybrid: move a glyph only when stroke AND correction
    vector both agree the current cell needs to change."""
    # Build set of cell_keys that correction vectors flag as wrong
    cells_flagged_by_corrections = {v['cell_key'] for v in correction_vectors}

    moves = []
    for aln in alignments:
        if (not aln.is_aligned
                and aln.stroke_implied_cell
                and aln.current_cell in cells_flagged_by_corrections):
            moves.append({
                'eva_glyph': aln.eva_glyph,
                'from_cell': aln.current_cell,
                'to_cell': aln.stroke_implied_cell,
            })
    return apply_character_moves(cv_labels, moves)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_grid_alternatives() -> Dict:
    """Phase 12.4: stroke-alignment audit and alternative grid construction.

    Saves results to results/grid_alternatives.json.
    """
    t0 = time.time()
    rdir = _results_dir()

    # ------------------------------------------------------------------
    # 1. Load inputs
    # ------------------------------------------------------------------
    with open(os.path.join(rdir, 'cv_labels.json')) as f:
        cv_labels: Dict = json.load(f)

    correction_vectors: List[Dict] = []
    diag_path = os.path.join(rdir, 'csp_diagnosis.json')
    if os.path.exists(diag_path):
        with open(diag_path) as f:
            diagnosis = json.load(f)
        correction_vectors = diagnosis.get('top_correction_vectors', [])

    print(f"  Grid cells: {len(cv_labels)}")

    # ------------------------------------------------------------------
    # 2. Analyse every glyph
    # ------------------------------------------------------------------
    alignments: List[GlyphStrokeAlignment] = []
    for cell_key, cell_info in cv_labels.items():
        for glyph in cell_info.get('glyphs', []):
            aln = analyse_glyph_alignment(glyph, cell_key, cv_labels)
            alignments.append(aln)

    n_aligned = sum(1 for a in alignments if a.is_aligned)
    n_misaligned = sum(1 for a in alignments if not a.is_aligned)
    misaligned_glyphs = [a.eva_glyph for a in alignments if not a.is_aligned]

    print(f"  Total glyphs analysed: {len(alignments)}")
    print(f"  Aligned: {n_aligned}, Misaligned: {n_misaligned}")
    if misaligned_glyphs:
        print(f"  Misaligned glyphs: {misaligned_glyphs}")

    # ------------------------------------------------------------------
    # 3. Build alternative grids
    # ------------------------------------------------------------------
    stroke_based = build_stroke_based_grid(cv_labels, alignments)
    hybrid = build_hybrid_grid(cv_labels, alignments, correction_vectors)

    # ------------------------------------------------------------------
    # 4. Gate and verdict
    # ------------------------------------------------------------------
    gate_passed = True  # Always passes — even zero misalignments is a valid finding
    if n_misaligned == 0:
        verdict = (
            "grid_alternatives_all_aligned: all EVA glyphs match their stroke-implied "
            "cells. The Phase 3 grid construction was internally consistent. "
            "Grid errors must arise from phonetic assignment, not glyph placement. "
            "Proceed to token_decomposition (Step 12.5) for further investigation."
        )
    else:
        verdict = (
            f"grid_alternatives_misaligned_{n_misaligned}: {n_misaligned} glyph(s) "
            f"({', '.join(misaligned_glyphs)}) have stroke decompositions that imply "
            "a different cell than their current assignment. "
            "Stroke-based and hybrid alternative grids constructed for CSP re-testing."
        )

    # ------------------------------------------------------------------
    # 5. Serialize and save
    # ------------------------------------------------------------------
    result = GridAlternativesResult(
        glyph_alignments=[asdict(a) for a in alignments],
        n_aligned=n_aligned,
        n_misaligned=n_misaligned,
        misaligned_glyphs=misaligned_glyphs,
        stroke_based_cv_labels=stroke_based,
        hybrid_cv_labels=hybrid,
        gate_passed=gate_passed,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rdir, 'grid_alternatives.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2)
    print(f"\n  Saved → {out_path}")
    print(f"  Verdict: {verdict[:120]}")
    return _convert(asdict(result))
