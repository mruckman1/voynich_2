"""
Phase 4 Step 2: Section Consistency Diagnosis
===============================================
Diagnose why cross-section grid consistency is low (0.14 Jaccard).
Three sub-analyses:

  2A — Per-section grid building and metrics
  2B — Minimum sample-size calibration for reliable grid construction
  2C — Currier A/B linguistic split test

Output:
  section_diagnosis.json — per-section grids, calibration curve, A/B verdict
"""

import json
import math
import os
import random
from collections import Counter
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from voynich.core.corpus import load_corpus, VoynichCorpus, VOYNICH_SECTIONS
from voynich.core.stats import (
    first_order_entropy, conditional_entropy, bootstrap_ci,
    bigram_transition_matrix, jensen_shannon_divergence,
)
from voynich.core._paths import results_dir as _results_dir
from voynich.phases.grid_validate import build_grid_from_tokens
from voynich.analysis.strokes import SyllabaryGrid, build_ventris_grid
from voynich.analysis.fingerprint import compute_profile


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SectionGridResult:
    """Grid analysis for one section."""
    section: str
    n_tokens: int
    currier_lang: str
    grid_rows: int
    grid_cols: int
    grid_occupancy: float
    grid_n_filled: int
    h1: float
    h2: float
    mean_word_length: float
    reliable: bool


@dataclass
class SampleSizePoint:
    """One data point in the sample-size calibration curve."""
    n_tokens: int
    mean_jaccard: float
    std_jaccard: float
    mean_occupancy: float


@dataclass
class MinSampleCalibration:
    """Result of minimum sample size calibration."""
    curve: List[SampleSizePoint]
    convergence_threshold: float
    minimum_reliable_size: int
    sections_below_minimum: List[str]
    sections_above_minimum: List[str]


@dataclass
class CurrierABResult:
    """Result of Currier A vs B comparison."""
    h1_a: float
    h1_b: float
    h2_a: float
    h2_b: float
    h1_difference: float
    h2_difference: float
    grid_jaccard: float
    bigram_jsd: float
    h2_diff_ci_lower: float
    h2_diff_ci_upper: float
    h2_diff_significant: bool
    n_tokens_a: int
    n_tokens_b: int
    sections_a: List[str]
    sections_b: List[str]
    verdict: str


@dataclass
class SectionDiagnosisResult:
    """Full section diagnosis output."""
    section_grids: List[SectionGridResult]
    calibration: MinSampleCalibration
    currier_ab: CurrierABResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _grid_jaccard(grid_a: SyllabaryGrid, grid_b: SyllabaryGrid) -> float:
    """Compute Jaccard similarity between two grids based on cell keys."""
    cells_a = set(grid_a.cells.keys())
    cells_b = set(grid_b.cells.keys())
    intersection = cells_a & cells_b
    union = cells_a | cells_b
    return len(intersection) / len(union) if union else 0.0


# ---------------------------------------------------------------------------
# Phase 2A: Per-section grid building
# ---------------------------------------------------------------------------

def build_section_grids(corpus: VoynichCorpus) -> List[SectionGridResult]:
    """Build a grid for each section and compute basic metrics."""
    results = []
    sections = sorted(set(p.section for p in corpus.pages.values()))

    for section in sections:
        tokens = corpus.get_tokens(section=section, paragraph_only=True)
        text = ' '.join(tokens)
        n = len(tokens)

        currier = VOYNICH_SECTIONS.get(section, {}).get('currier_lang', '?')

        if n < 50:
            results.append(SectionGridResult(
                section=section, n_tokens=n, currier_lang=currier,
                grid_rows=0, grid_cols=0, grid_occupancy=0, grid_n_filled=0,
                h1=0, h2=0, mean_word_length=0, reliable=False,
            ))
            continue

        # Build grid
        if n >= 500:
            try:
                grid = build_grid_from_tokens(tokens)
            except Exception:
                grid = build_ventris_grid(tokens)
        else:
            grid = build_ventris_grid(tokens)

        h1 = first_order_entropy(text) if text else 0
        h2 = conditional_entropy(text, order=1) if len(text) > 10 else 0
        mean_wl = float(np.mean([len(t) for t in tokens])) if tokens else 0

        results.append(SectionGridResult(
            section=section,
            n_tokens=n,
            currier_lang=currier,
            grid_rows=len(grid.row_labels),
            grid_cols=len(grid.col_labels),
            grid_occupancy=grid.occupancy,
            grid_n_filled=grid.n_filled,
            h1=round(h1, 4),
            h2=round(h2, 4),
            mean_word_length=round(mean_wl, 2),
            reliable=n >= 500,
        ))

    return results


# ---------------------------------------------------------------------------
# Phase 2B: Sample-size calibration
# ---------------------------------------------------------------------------

def calibrate_minimum_sample(
    tokens: List[str],
    full_grid: SyllabaryGrid,
    test_sizes: Optional[List[int]] = None,
    n_bootstrap: int = 50,
    seed: int = 42,
) -> Tuple[List[SampleSizePoint], int]:
    """
    Determine minimum token count for reliable grid construction.

    Returns (curve, minimum_size).
    """
    if test_sizes is None:
        test_sizes = [200, 500, 1000, 2000, 5000, 10000]

    # Filter to sizes that don't exceed corpus
    test_sizes = [s for s in test_sizes if s <= len(tokens)]

    full_cells = set(full_grid.cells.keys())
    rng = random.Random(seed)
    curve = []

    for size in test_sizes:
        jaccards = []
        occupancies = []

        for _ in range(n_bootstrap):
            sample = rng.sample(tokens, size)
            try:
                sample_grid = build_grid_from_tokens(sample)
            except Exception:
                continue

            sample_cells = set(sample_grid.cells.keys())
            union = full_cells | sample_cells
            intersection = full_cells & sample_cells
            j = len(intersection) / len(union) if union else 0.0
            jaccards.append(j)
            occupancies.append(sample_grid.occupancy)

        mean_j = float(np.mean(jaccards)) if jaccards else 0.0
        std_j = float(np.std(jaccards)) if jaccards else 0.0
        mean_occ = float(np.mean(occupancies)) if occupancies else 0.0

        curve.append(SampleSizePoint(
            n_tokens=size,
            mean_jaccard=round(mean_j, 4),
            std_jaccard=round(std_j, 4),
            mean_occupancy=round(mean_occ, 4),
        ))

    # Find minimum size for >= 0.8 Jaccard
    min_size = len(tokens)  # default: need full corpus
    for pt in curve:
        if pt.mean_jaccard >= 0.80:
            min_size = pt.n_tokens
            break

    return curve, min_size


# ---------------------------------------------------------------------------
# Phase 2C: Currier A/B split test
# ---------------------------------------------------------------------------

def test_currier_ab_split(
    corpus: VoynichCorpus,
    seed: int = 42,
) -> CurrierABResult:
    """
    Test whether Currier A and B sections show distinct linguistic profiles.

    A = herbal_a (the only Currier A section, 9449 tokens)
    B = all Currier B sections except herbal_b (181 tokens, too small)
    """
    # Collect tokens by Currier language
    sections_a = []
    sections_b = []
    tokens_a: List[str] = []
    tokens_b: List[str] = []

    for section, info in VOYNICH_SECTIONS.items():
        sec_tokens = corpus.get_tokens(section=section, paragraph_only=True)
        if len(sec_tokens) < 200:
            continue  # Skip herbal_b (181 tokens)
        if info['currier_lang'] == 'A':
            sections_a.append(section)
            tokens_a.extend(sec_tokens)
        else:
            sections_b.append(section)
            tokens_b.extend(sec_tokens)

    text_a = ' '.join(tokens_a)
    text_b = ' '.join(tokens_b)

    # Entropy comparison
    h1_a = first_order_entropy(text_a)
    h1_b = first_order_entropy(text_b)
    h2_a = conditional_entropy(text_a, order=1)
    h2_b = conditional_entropy(text_b, order=1)

    # Grid comparison
    grid_a = build_grid_from_tokens(tokens_a)
    grid_b = build_grid_from_tokens(tokens_b)
    jaccard = _grid_jaccard(grid_a, grid_b)

    # Bigram JSD
    mat_a, alph_a = bigram_transition_matrix(text_a)
    mat_b, alph_b = bigram_transition_matrix(text_b)
    # Align alphabets for JSD
    all_chars = sorted(set(alph_a) | set(alph_b))
    n = len(all_chars)
    char_idx = {c: i for i, c in enumerate(all_chars)}

    aligned_a = np.zeros((n, n))
    for i, ca in enumerate(alph_a):
        for j, cb in enumerate(alph_a):
            aligned_a[char_idx[ca], char_idx[cb]] = mat_a[i, j]

    aligned_b = np.zeros((n, n))
    for i, ca in enumerate(alph_b):
        for j, cb in enumerate(alph_b):
            aligned_b[char_idx[ca], char_idx[cb]] = mat_b[i, j]

    # Flatten and compute JSD
    flat_a = aligned_a.flatten()
    flat_b = aligned_b.flatten()
    sa = flat_a.sum()
    sb = flat_b.sum()
    if sa > 0:
        flat_a = flat_a / sa
    if sb > 0:
        flat_b = flat_b / sb
    bigram_jsd = jensen_shannon_divergence(flat_a, flat_b)

    # Bootstrap CI on H2 difference
    rng = random.Random(seed)
    n_a = len(tokens_a)
    diffs = []
    for _ in range(500):
        # Subsample both to equal size
        sample_size = min(n_a, len(tokens_b), 5000)
        sa_tokens = rng.sample(tokens_a, min(sample_size, n_a))
        sb_tokens = rng.sample(tokens_b, min(sample_size, len(tokens_b)))
        sa_text = ' '.join(sa_tokens)
        sb_text = ' '.join(sb_tokens)
        h2_sa = conditional_entropy(sa_text, order=1) if len(sa_text) > 10 else 0
        h2_sb = conditional_entropy(sb_text, order=1) if len(sb_text) > 10 else 0
        diffs.append(h2_sa - h2_sb)

    diffs_arr = np.array(diffs)
    ci_lower = float(np.percentile(diffs_arr, 2.5))
    ci_upper = float(np.percentile(diffs_arr, 97.5))
    significant = ci_lower > 0 or ci_upper < 0  # CI doesn't include 0

    # Verdict
    h2_diff = h2_a - h2_b
    if jaccard < 0.5 and significant:
        verdict = 'distinct_dialects'
    elif jaccard > 0.8 and not significant:
        verdict = 'same_system'
    else:
        verdict = 'inconclusive'

    return CurrierABResult(
        h1_a=round(h1_a, 4),
        h1_b=round(h1_b, 4),
        h2_a=round(h2_a, 4),
        h2_b=round(h2_b, 4),
        h1_difference=round(h1_a - h1_b, 4),
        h2_difference=round(h2_diff, 4),
        grid_jaccard=round(jaccard, 4),
        bigram_jsd=round(bigram_jsd, 6),
        h2_diff_ci_lower=round(ci_lower, 4),
        h2_diff_ci_upper=round(ci_upper, 4),
        h2_diff_significant=significant,
        n_tokens_a=len(tokens_a),
        n_tokens_b=len(tokens_b),
        sections_a=sections_a,
        sections_b=sections_b,
        verdict=verdict,
    )


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

def _print_results(
    section_grids: List[SectionGridResult],
    calibration: MinSampleCalibration,
    currier: CurrierABResult,
) -> None:
    """Print formatted results."""
    # Section grids
    print("\n--- Phase 2A: Per-Section Grid Analysis ---")
    print(f"{'Section':<20s} {'Tokens':>7s} {'Lang':>5s} {'Grid':>7s} "
          f"{'Occ':>6s} {'H1':>6s} {'H2':>6s} {'Reliable':>8s}")
    print("-" * 70)
    for sg in section_grids:
        grid_str = f"{sg.grid_rows}x{sg.grid_cols}" if sg.grid_rows > 0 else "N/A"
        print(f"{sg.section:<20s} {sg.n_tokens:>7d} {sg.currier_lang:>5s} "
              f"{grid_str:>7s} {sg.grid_occupancy:>6.3f} "
              f"{sg.h1:>6.3f} {sg.h2:>6.3f} "
              f"{'YES' if sg.reliable else 'NO':>8s}")

    # Sample-size calibration
    print("\n--- Phase 2B: Minimum Sample Size Calibration ---")
    print(f"{'N tokens':>10s} {'Mean Jaccard':>13s} {'Std':>7s} {'Mean Occ':>10s}")
    print("-" * 42)
    for pt in calibration.curve:
        marker = " <-- threshold" if pt.n_tokens == calibration.minimum_reliable_size and pt.mean_jaccard >= 0.80 else ""
        print(f"{pt.n_tokens:>10d} {pt.mean_jaccard:>13.4f} {pt.std_jaccard:>7.4f} "
              f"{pt.mean_occupancy:>10.4f}{marker}")

    print(f"\nMinimum for 80% Jaccard: {calibration.minimum_reliable_size} tokens")
    if calibration.sections_below_minimum:
        print(f"Sections BELOW minimum: {', '.join(calibration.sections_below_minimum)}")
    if calibration.sections_above_minimum:
        print(f"Sections ABOVE minimum: {', '.join(calibration.sections_above_minimum)}")

    # Currier A/B
    print("\n--- Phase 2C: Currier A/B Split Test ---")
    print(f"Language A sections: {', '.join(currier.sections_a)} ({currier.n_tokens_a} tokens)")
    print(f"Language B sections: {', '.join(currier.sections_b)} ({currier.n_tokens_b} tokens)")
    print(f"\n  H1(A) = {currier.h1_a:.4f}    H1(B) = {currier.h1_b:.4f}    diff = {currier.h1_difference:+.4f}")
    print(f"  H2(A) = {currier.h2_a:.4f}    H2(B) = {currier.h2_b:.4f}    diff = {currier.h2_difference:+.4f}")
    print(f"  H2 diff 95% CI: [{currier.h2_diff_ci_lower:.4f}, {currier.h2_diff_ci_upper:.4f}]"
          f"  {'SIGNIFICANT' if currier.h2_diff_significant else 'not significant'}")
    print(f"  Grid Jaccard(A, B): {currier.grid_jaccard:.4f}")
    print(f"  Bigram JSD(A, B):   {currier.bigram_jsd:.6f}")

    print(f"\n  DIAGNOSIS: {currier.verdict.upper()}")
    if currier.verdict == 'distinct_dialects':
        print("  -> Currier A and B show distinct linguistic profiles.")
        print("  -> Consider building separate grids for each language variant.")
    elif currier.verdict == 'same_system':
        print("  -> Currier A and B use the same writing system.")
        print("  -> Low cross-section consistency is a sample-size artifact.")
    else:
        print("  -> Evidence is mixed. Grid structure differs but entropy profiles overlap.")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_section_diagnosis() -> Dict:
    """Run the full section consistency diagnosis."""
    print("=" * 70)
    print("PHASE 4 STEP 2: SECTION CONSISTENCY DIAGNOSIS")
    print("=" * 70)

    corpus = load_corpus(verbose=False)
    all_tokens = corpus.get_tokens(paragraph_only=True)

    # Phase 2A: Per-section grids
    print("\nBuilding per-section grids...")
    section_grids = build_section_grids(corpus)

    # Phase 2B: Sample-size calibration
    print("Running sample-size calibration...")
    full_grid = build_grid_from_tokens(all_tokens)
    curve, min_size = calibrate_minimum_sample(all_tokens, full_grid)

    below = [sg.section for sg in section_grids if sg.n_tokens < min_size]
    above = [sg.section for sg in section_grids if sg.n_tokens >= min_size]

    calibration = MinSampleCalibration(
        curve=curve,
        convergence_threshold=0.80,
        minimum_reliable_size=min_size,
        sections_below_minimum=below,
        sections_above_minimum=above,
    )

    # Phase 2C: Currier A/B
    print("Testing Currier A/B split...")
    currier = test_currier_ab_split(corpus)

    # Print
    _print_results(section_grids, calibration, currier)

    # Save
    rd = _results_dir()
    out_data = {
        'section_grids': [asdict(sg) for sg in section_grids],
        'calibration': {
            'curve': [asdict(pt) for pt in calibration.curve],
            'convergence_threshold': calibration.convergence_threshold,
            'minimum_reliable_size': calibration.minimum_reliable_size,
            'sections_below_minimum': calibration.sections_below_minimum,
            'sections_above_minimum': calibration.sections_above_minimum,
        },
        'currier_ab': asdict(currier),
    }
    out_path = os.path.join(rd, 'section_diagnosis.json')
    with open(out_path, 'w') as f:
        json.dump(out_data, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return out_data
