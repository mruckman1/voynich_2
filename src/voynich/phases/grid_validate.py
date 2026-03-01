"""
Workstream E: Grid Validation and Gap Analysis
================================================
Systematic tests of grid structure, frequency distribution, stability,
and cross-section consistency.

Tests:
  E.1 — Systematic gap analysis (empty cell patterns)
  E.2 — Grid cell frequency distribution (Zipf fit)
  E.3 — Grid stability under perturbation (bootstrap reconstruction)
  E.4 — Cross-section grid consistency
"""

import json
import math
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import ks_2samp, chi2_contingency

from voynich.core.corpus import load_corpus, VoynichCorpus, VOYNICH_SECTIONS, tokenize_eva_chars
from voynich.core.stats import cosine_similarity
from voynich.core._paths import results_dir as _results_dir
from voynich.analysis.strokes import (
    SyllabaryGrid, build_ventris_grid, decompose_glyph,
    segment_token_as_syllables, syllable_sequence_stats,
)
from voynich.phases.grid_refine import (
    build_context_vectors, pairwise_similarity_matrix,
    hierarchical_cluster, merge_grid_categories,
    assess_onset_merging,
)


# ---------------------------------------------------------------------------
# E.1: Systematic Gap Analysis
# ---------------------------------------------------------------------------

@dataclass
class GapAnalysisResult:
    """Result of gap pattern analysis in the syllabary grid."""
    total_cells: int
    filled_cells: int
    empty_cells: int
    occupancy: float
    row_fill_rates: Dict[str, float]
    col_fill_rates: Dict[str, float]
    chi2_statistic: float
    chi2_pvalue: float
    chi2_dof: int
    gap_entropy_rows: float
    gap_entropy_cols: float
    systematic_pattern: str
    random_sim_mean_occupancy: float
    random_sim_pvalue: float
    reference_comparison: Dict[str, Dict]


def analyze_gap_patterns(grid: SyllabaryGrid, n_random: int = 10000) -> GapAnalysisResult:
    """
    Test whether empty cells show systematic patterns.

    Method:
    1. Compute per-row and per-column fill rates
    2. Chi-squared test on fill contingency table
    3. Compare gap entropy against random assignment
    4. Compare occupancy pattern to known syllabaries
    """
    n_rows = len(grid.row_labels)
    n_cols = len(grid.col_labels)
    total = n_rows * n_cols
    filled = grid.n_filled
    empty = total - filled

    # Build fill matrix (1 = filled, 0 = empty)
    fill_matrix = np.zeros((n_rows, n_cols), dtype=int)
    for key in grid.cells:
        parts = key.split(',', 1)
        if len(parts) != 2:
            continue
        onset, nucleus = parts
        if onset in grid.row_labels and nucleus in grid.col_labels:
            r = grid.row_labels.index(onset)
            c = grid.col_labels.index(nucleus)
            fill_matrix[r, c] = 1

    # Per-row and per-column fill rates
    row_fills = {}
    for i, label in enumerate(grid.row_labels):
        row_fills[label] = float(fill_matrix[i].sum()) / n_cols

    col_fills = {}
    for j, label in enumerate(grid.col_labels):
        col_fills[label] = float(fill_matrix[:, j].sum()) / n_rows

    # Chi-squared test on 2D contingency table (filled vs empty by row/col)
    # Build contingency: each row has (filled_count, empty_count)
    contingency = np.zeros((n_rows, 2), dtype=int)
    for i in range(n_rows):
        contingency[i, 0] = fill_matrix[i].sum()
        contingency[i, 1] = n_cols - contingency[i, 0]

    # Only test if we have variation
    if contingency[:, 0].min() == contingency[:, 0].max():
        chi2_stat, chi2_p, chi2_dof = 0.0, 1.0, 0
    else:
        try:
            chi2_stat, chi2_p, chi2_dof, _ = chi2_contingency(contingency)
        except ValueError:
            chi2_stat, chi2_p, chi2_dof = 0.0, 1.0, 0

    # Gap entropy: H of fill rates
    def _entropy_of_rates(rates):
        vals = list(rates.values())
        if not vals:
            return 0.0
        arr = np.array(vals)
        arr = arr / arr.sum() if arr.sum() > 0 else arr
        return float(-sum(p * math.log2(p) for p in arr if p > 0))

    gap_h_rows = _entropy_of_rates(row_fills)
    gap_h_cols = _entropy_of_rates(col_fills)

    # Random simulation: distribute n_filled cells randomly into n_rows x n_cols grid
    rng = random.Random(42)
    random_occ = []
    random_row_h = []
    for _ in range(n_random):
        indices = rng.sample(range(total), filled)
        rand_fill = np.zeros((n_rows, n_cols), dtype=int)
        for idx in indices:
            r, c = divmod(idx, n_cols)
            rand_fill[r, c] = 1
        random_occ.append(rand_fill.sum() / total)
        rand_row_rates = {}
        for i, label in enumerate(grid.row_labels):
            rand_row_rates[label] = float(rand_fill[i].sum()) / n_cols
        random_row_h.append(_entropy_of_rates(rand_row_rates))

    random_mean_occ = float(np.mean(random_occ))
    # p-value: fraction of random simulations with row entropy <= observed
    random_p = sum(1 for h in random_row_h if h <= gap_h_rows) / n_random

    # Systematic pattern classification
    if chi2_p < 0.01:
        if gap_h_rows < gap_h_cols:
            pattern = 'row_structured'
        elif gap_h_cols < gap_h_rows:
            pattern = 'col_structured'
        else:
            pattern = 'block'
    else:
        pattern = 'random'

    # Reference comparison
    reference_comparison = _compare_gap_to_syllabaries(grid)

    return GapAnalysisResult(
        total_cells=total,
        filled_cells=filled,
        empty_cells=empty,
        occupancy=grid.occupancy,
        row_fill_rates=row_fills,
        col_fill_rates=col_fills,
        chi2_statistic=float(chi2_stat),
        chi2_pvalue=float(chi2_p),
        chi2_dof=int(chi2_dof),
        gap_entropy_rows=gap_h_rows,
        gap_entropy_cols=gap_h_cols,
        systematic_pattern=pattern,
        random_sim_mean_occupancy=random_mean_occ,
        random_sim_pvalue=random_p,
        reference_comparison=reference_comparison,
    )


def _compare_gap_to_syllabaries(grid: SyllabaryGrid) -> Dict[str, Dict]:
    """Compare the grid's occupancy pattern against known syllabary gaps."""
    refs = {
        'linear_b': {'occupancy': 0.60, 'empty_pct': 0.40,
                      'description': '~35% systematic gaps (no zi, no wu)'},
        'japanese_kana': {'occupancy': 0.92, 'empty_pct': 0.08,
                          'description': '~8% gaps (no yi, no wu in modern)'},
        'cypriot': {'occupancy': 0.55, 'empty_pct': 0.45,
                    'description': '~45% gaps (no wu, no zu)'},
    }
    result = {}
    for name, ref in refs.items():
        result[name] = {
            'ref_occupancy': ref['occupancy'],
            'voynich_occupancy': grid.occupancy,
            'occupancy_diff': abs(grid.occupancy - ref['occupancy']),
            'description': ref['description'],
        }
    return result


# ---------------------------------------------------------------------------
# E.2: Grid Cell Frequency Distribution
# ---------------------------------------------------------------------------

@dataclass
class FrequencyDistResult:
    """Result of grid cell frequency distribution analysis."""
    cell_frequencies: Dict[str, int]
    n_cells: int
    zipf_exponent: float
    zipf_r_squared: float
    ks_statistic: float
    ks_pvalue: float
    frequency_entropy: float
    top_cells: List[Tuple[str, int]]


def grid_cell_frequencies(
    tokens: List[str],
    grid: SyllabaryGrid,
) -> Dict[str, int]:
    """Count how many times each grid cell is used across the corpus."""
    counts: Counter = Counter()
    for token in tokens:
        syls = segment_token_as_syllables(token, grid)
        counts.update(syls)
    return dict(counts)


def frequency_distribution_test(
    tokens: List[str],
    grid: SyllabaryGrid,
) -> FrequencyDistResult:
    """
    Analyze whether grid cell frequencies follow Zipf's law.
    """
    freqs = grid_cell_frequencies(tokens, grid)
    if not freqs:
        return FrequencyDistResult(
            cell_frequencies={}, n_cells=0,
            zipf_exponent=0.0, zipf_r_squared=0.0,
            ks_statistic=0.0, ks_pvalue=1.0,
            frequency_entropy=0.0, top_cells=[],
        )

    ranked_freqs = sorted(freqs.values(), reverse=True)
    n = len(ranked_freqs)

    # Zipf fit
    ranks = np.arange(1, n + 1, dtype=float)
    freq_arr = np.array(ranked_freqs, dtype=float)

    log_ranks = np.log(ranks)
    log_freqs = np.log(freq_arr + 1e-10)

    A = np.vstack([log_ranks, np.ones(n)]).T
    try:
        result = np.linalg.lstsq(A, log_freqs, rcond=None)
        slope, intercept = result[0]
    except np.linalg.LinAlgError:
        slope, intercept = -1.0, 0.0

    predicted = slope * log_ranks + intercept
    ss_res = np.sum((log_freqs - predicted) ** 2)
    ss_tot = np.sum((log_freqs - np.mean(log_freqs)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # KS test: compare observed distribution to ideal Zipf
    # Generate ideal Zipf distribution with same exponent
    ideal_freqs = ranks ** slope * np.exp(intercept)
    ideal_freqs = ideal_freqs / ideal_freqs.sum()
    observed_normed = freq_arr / freq_arr.sum()
    ks_stat, ks_p = ks_2samp(observed_normed, ideal_freqs)

    # Frequency entropy
    total = sum(ranked_freqs)
    h = -sum((f / total) * math.log2(f / total)
             for f in ranked_freqs if f > 0)

    # Top cells
    top = sorted(freqs.items(), key=lambda x: x[1], reverse=True)[:10]

    return FrequencyDistResult(
        cell_frequencies=freqs,
        n_cells=n,
        zipf_exponent=float(-slope),
        zipf_r_squared=float(r_squared),
        ks_statistic=float(ks_stat),
        ks_pvalue=float(ks_p),
        frequency_entropy=float(h),
        top_cells=top,
    )


# ---------------------------------------------------------------------------
# E.3: Grid Stability Under Perturbation
# ---------------------------------------------------------------------------

@dataclass
class StabilityResult:
    """Result of bootstrap grid stability test."""
    n_iterations: int
    subsample_fraction: float
    full_grid_cells: List[str]
    cell_stability: Dict[str, float]
    stable_cells: int
    unstable_cells: int
    grid_jaccard_mean: float
    grid_jaccard_std: float
    occupancy_mean: float
    occupancy_std: float


def build_grid_from_tokens(
    tokens: List[str],
    n_nucleus_clusters: int = 6,
    n_onset_clusters: Optional[int] = 5,
) -> SyllabaryGrid:
    """
    Build a refined grid from an arbitrary token set.

    Applies the same pipeline as Phase 2B:
    1. build_ventris_grid() from strokes.py
    2. Nucleus distributional clustering
    3. Optional onset merging
    """
    if len(tokens) < 10:
        return build_ventris_grid(tokens)

    raw_grid = build_ventris_grid(tokens)

    # Nucleus clustering
    n_nuclei_raw = len(raw_grid.col_labels)
    if n_nucleus_clusters < n_nuclei_raw:
        nuc_vectors, _ = build_context_vectors(tokens, raw_grid, axis='nucleus')
        nuc_sim, nuc_labels = pairwise_similarity_matrix(nuc_vectors)
        nuc_clusters = hierarchical_cluster(nuc_sim, nuc_labels,
                                            n_clusters=n_nucleus_clusters)
        grid = merge_grid_categories(raw_grid, nuc_clusters, axis='nucleus')
    else:
        grid = raw_grid

    # Onset merging
    if n_onset_clusters is not None:
        n_onsets_raw = len(grid.row_labels)
        if n_onset_clusters < n_onsets_raw:
            onset_vectors, _ = build_context_vectors(tokens, grid, axis='onset')
            onset_sim, onset_labels = pairwise_similarity_matrix(onset_vectors)
            onset_clusters = hierarchical_cluster(onset_sim, onset_labels,
                                                  n_clusters=n_onset_clusters)
            grid = merge_grid_categories(grid, onset_clusters, axis='onset')
        else:
            # Check if automatic onset merging is warranted
            onset_merging = assess_onset_merging(tokens, grid)
            if onset_merging is not None:
                grid = merge_grid_categories(grid, onset_merging, axis='onset')

    return grid


def bootstrap_grid_stability(
    tokens: List[str],
    n_iterations: int = 200,
    subsample_fraction: float = 0.5,
    seed: int = 42,
) -> StabilityResult:
    """
    Bootstrap test: subsample tokens, rebuild grid, compare to full-corpus grid.
    """
    full_grid = build_grid_from_tokens(tokens)
    full_cells = set(full_grid.cells.keys())

    rng = random.Random(seed)
    n_sample = int(len(tokens) * subsample_fraction)

    cell_presence: Counter = Counter()
    jaccards = []
    occupancies = []

    for _ in range(n_iterations):
        sample = rng.sample(tokens, n_sample)
        try:
            sample_grid = build_grid_from_tokens(sample)
        except Exception:
            continue

        sample_cells = set(sample_grid.cells.keys())
        cell_presence.update(sample_cells)
        occupancies.append(sample_grid.occupancy)

        # Jaccard similarity
        intersection = full_cells & sample_cells
        union = full_cells | sample_cells
        jaccard = len(intersection) / len(union) if union else 0.0
        jaccards.append(jaccard)

    # Cell stability: fraction of iterations each cell appeared
    cell_stability = {}
    for cell in full_cells:
        cell_stability[cell] = cell_presence[cell] / n_iterations

    stable = sum(1 for v in cell_stability.values() if v >= 0.9)
    unstable = sum(1 for v in cell_stability.values() if v < 0.5)

    return StabilityResult(
        n_iterations=n_iterations,
        subsample_fraction=subsample_fraction,
        full_grid_cells=sorted(full_cells),
        cell_stability=cell_stability,
        stable_cells=stable,
        unstable_cells=unstable,
        grid_jaccard_mean=float(np.mean(jaccards)) if jaccards else 0.0,
        grid_jaccard_std=float(np.std(jaccards)) if jaccards else 0.0,
        occupancy_mean=float(np.mean(occupancies)) if occupancies else 0.0,
        occupancy_std=float(np.std(occupancies)) if occupancies else 0.0,
    )


# ---------------------------------------------------------------------------
# E.4: Cross-Section Grid Consistency
# ---------------------------------------------------------------------------

@dataclass
class SectionConsistencyResult:
    """Result of cross-section grid comparison."""
    section_summaries: Dict[str, Dict]
    vs_full_grid: Dict[str, float]
    mean_jaccard: float
    min_jaccard: float
    min_jaccard_section: str
    consistent: bool


def section_grid_consistency(
    corpus: VoynichCorpus,
    full_grid: Optional[SyllabaryGrid] = None,
) -> SectionConsistencyResult:
    """
    Build grids from each section's tokens, compare to full-corpus grid.
    """
    all_tokens = corpus.get_tokens(paragraph_only=True)

    if full_grid is None:
        full_grid = build_grid_from_tokens(all_tokens)

    full_cells = set(full_grid.cells.keys())

    section_summaries = {}
    vs_full = {}

    for section in sorted(VOYNICH_SECTIONS.keys()):
        sec_tokens = corpus.get_tokens(section=section, paragraph_only=True)
        if len(sec_tokens) < 50:
            section_summaries[section] = {
                'n_tokens': len(sec_tokens),
                'status': 'too_few_tokens',
            }
            continue

        # For small sections, use raw grid only (no clustering)
        if len(sec_tokens) < 500:
            sec_grid = build_ventris_grid(sec_tokens)
        else:
            try:
                sec_grid = build_grid_from_tokens(sec_tokens)
            except Exception:
                sec_grid = build_ventris_grid(sec_tokens)

        sec_cells = set(sec_grid.cells.keys())

        # Jaccard similarity with full grid
        intersection = full_cells & sec_cells
        union = full_cells | sec_cells
        jaccard = len(intersection) / len(union) if union else 0.0

        section_summaries[section] = {
            'n_tokens': len(sec_tokens),
            'n_onsets': len(sec_grid.row_labels),
            'n_nuclei': len(sec_grid.col_labels),
            'n_cells_filled': sec_grid.n_filled,
            'occupancy': round(sec_grid.occupancy, 4),
            'jaccard_vs_full': round(jaccard, 4),
        }
        vs_full[section] = jaccard

    jaccards = list(vs_full.values())
    mean_j = float(np.mean(jaccards)) if jaccards else 0.0
    min_j = min(jaccards) if jaccards else 0.0
    min_sec = min(vs_full, key=vs_full.get) if vs_full else ''

    return SectionConsistencyResult(
        section_summaries=section_summaries,
        vs_full_grid=vs_full,
        mean_jaccard=mean_j,
        min_jaccard=min_j,
        min_jaccard_section=min_sec,
        consistent=mean_j >= 0.60,
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_grid_validation() -> Dict:
    """Run all Workstream E tests and print/save results."""
    rd = _results_dir()

    print("=" * 70)
    print("WORKSTREAM E: GRID VALIDATION AND GAP ANALYSIS")
    print("=" * 70)

    # Load data
    corpus = load_corpus(verbose=False)
    tokens = corpus.get_tokens(paragraph_only=True)
    grid = build_grid_from_tokens(tokens)

    print(f"\n  Working grid: {len(grid.row_labels)}x{len(grid.col_labels)}, "
          f"{grid.n_filled} filled ({grid.occupancy:.1%})")

    # E.1: Gap Analysis
    print("\n--- E.1: Systematic Gap Analysis ---")
    e1 = analyze_gap_patterns(grid)
    print(f"  Total cells: {e1.total_cells}, filled: {e1.filled_cells}, "
          f"empty: {e1.empty_cells}")
    print(f"  Row fill rates:")
    for label, rate in sorted(e1.row_fill_rates.items()):
        print(f"    {label}: {rate:.1%}")
    print(f"  Col fill rates:")
    for label, rate in sorted(e1.col_fill_rates.items()):
        print(f"    {label}: {rate:.1%}")
    print(f"  Chi-squared: {e1.chi2_statistic:.4f} (p={e1.chi2_pvalue:.4f}, "
          f"dof={e1.chi2_dof})")
    print(f"  Gap entropy (rows): {e1.gap_entropy_rows:.4f}")
    print(f"  Gap entropy (cols): {e1.gap_entropy_cols:.4f}")
    print(f"  Random sim p-value: {e1.random_sim_pvalue:.4f}")
    print(f"  >> Pattern: {e1.systematic_pattern}")

    for name, comp in e1.reference_comparison.items():
        print(f"  vs {name}: occupancy diff = "
              f"{comp['occupancy_diff']:.2f}")

    with open(os.path.join(rd, 'grid_gaps.json'), 'w') as f:
        json.dump(asdict(e1), f, indent=2)

    # E.2: Frequency Distribution
    print("\n--- E.2: Grid Cell Frequency Distribution ---")
    e2 = frequency_distribution_test(tokens, grid)
    print(f"  Cells used: {e2.n_cells}")
    print(f"  Zipf exponent: {e2.zipf_exponent:.4f}")
    print(f"  Zipf R^2: {e2.zipf_r_squared:.4f}")
    print(f"  KS statistic: {e2.ks_statistic:.4f} (p={e2.ks_pvalue:.4f})")
    print(f"  Frequency entropy: {e2.frequency_entropy:.4f} bits")
    print(f"  Top 5 cells:")
    for cell, freq in e2.top_cells[:5]:
        print(f"    {cell}: {freq:,}")

    with open(os.path.join(rd, 'grid_frequency.json'), 'w') as f:
        json.dump({
            'n_cells': e2.n_cells,
            'zipf_exponent': e2.zipf_exponent,
            'zipf_r_squared': e2.zipf_r_squared,
            'ks_statistic': e2.ks_statistic,
            'ks_pvalue': e2.ks_pvalue,
            'frequency_entropy': e2.frequency_entropy,
            'top_cells': e2.top_cells,
            'cell_frequencies': e2.cell_frequencies,
        }, f, indent=2)

    # E.3: Grid Stability
    print("\n--- E.3: Grid Stability Under Perturbation ---")
    print("  Running 200 bootstrap iterations (50% subsampling)...")
    e3 = bootstrap_grid_stability(tokens, n_iterations=200)
    print(f"  Full grid cells: {len(e3.full_grid_cells)}")
    print(f"  Stable cells (>90%): {e3.stable_cells}")
    print(f"  Unstable cells (<50%): {e3.unstable_cells}")
    print(f"  Mean Jaccard similarity: {e3.grid_jaccard_mean:.4f} "
          f"(std {e3.grid_jaccard_std:.4f})")
    print(f"  Mean occupancy: {e3.occupancy_mean:.4f} "
          f"(std {e3.occupancy_std:.4f})")
    stable_frac = e3.stable_cells / len(e3.full_grid_cells) if e3.full_grid_cells else 0
    print(f"  >> Stability: {stable_frac:.0%} of cells stable")

    # Per-cell stability
    print(f"  Per-cell stability:")
    for cell, stab in sorted(e3.cell_stability.items(),
                              key=lambda x: x[1], reverse=True):
        marker = "OK" if stab >= 0.9 else ("WEAK" if stab >= 0.5 else "UNSTABLE")
        print(f"    {cell}: {stab:.1%} [{marker}]")

    with open(os.path.join(rd, 'grid_stability.json'), 'w') as f:
        json.dump(asdict(e3), f, indent=2)

    # E.4: Cross-Section Consistency
    print("\n--- E.4: Cross-Section Grid Consistency ---")
    e4 = section_grid_consistency(corpus, grid)
    print(f"  Section grid Jaccard similarities vs full grid:")
    for section, info in sorted(e4.section_summaries.items()):
        if 'jaccard_vs_full' in info:
            print(f"    {section:<20s}: {info['jaccard_vs_full']:.4f} "
                  f"({info['n_tokens']:,} tokens, "
                  f"{info['n_onsets']}x{info['n_nuclei']})")
        else:
            print(f"    {section:<20s}: {info['status']} "
                  f"({info['n_tokens']} tokens)")
    print(f"  Mean Jaccard: {e4.mean_jaccard:.4f}")
    print(f"  Min Jaccard:  {e4.min_jaccard:.4f} ({e4.min_jaccard_section})")
    print(f"  >> Consistent: {e4.consistent}")

    with open(os.path.join(rd, 'grid_sections.json'), 'w') as f:
        json.dump(asdict(e4), f, indent=2)

    # Summary
    print(f"\n{'=' * 70}")
    print("WORKSTREAM E SUMMARY")
    print(f"  E.1 Gap pattern:    {e1.systematic_pattern} (chi2 p={e1.chi2_pvalue:.4f})")
    print(f"  E.2 Zipf fit:       exponent={e2.zipf_exponent:.2f}, "
          f"R^2={e2.zipf_r_squared:.4f}")
    print(f"  E.3 Stability:      {stable_frac:.0%} stable, "
          f"Jaccard={e3.grid_jaccard_mean:.4f}")
    print(f"  E.4 Consistency:    mean Jaccard={e4.mean_jaccard:.4f}, "
          f"consistent={e4.consistent}")
    print(f"{'=' * 70}")

    return {
        'e1': asdict(e1),
        'e2': {
            'n_cells': e2.n_cells,
            'zipf_exponent': e2.zipf_exponent,
            'zipf_r_squared': e2.zipf_r_squared,
            'ks_statistic': e2.ks_statistic,
            'ks_pvalue': e2.ks_pvalue,
        },
        'e3': asdict(e3),
        'e4': asdict(e4),
    }
