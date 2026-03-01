"""
Syllabary Grid Refinement (Phase 2B)
======================================
Merge distributional synonyms in onset/nucleus categories to produce a
denser, more linguistically plausible syllabary grid.

Phases:
  B.1 — Nucleus merging via distributional clustering
  B.2 — Onset merging (if high similarity detected)
  B.3 — Grid validation (occupancy, discriminant, syllable stats)
  B.4 — Language narrowing based on grid dimensions
"""

import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from voynich.core.corpus import load_corpus, tokenize_eva_chars
from voynich.core.stats import cosine_similarity
from voynich.core._paths import results_dir as _results_dir
from voynich.analysis.strokes import (
    SyllabaryGrid, build_ventris_grid, decompose_glyph,
    segment_token_as_syllables, syllable_sequence_stats,
    syllabary_discriminant_test,
)


# ---------------------------------------------------------------------------
# Phase B.1: Distributional Clustering
# ---------------------------------------------------------------------------

def _glyph_onset_nucleus(glyph: str) -> Optional[Tuple[str, str]]:
    """Get (onset, nucleus) for a glyph based on first/last stroke."""
    strokes = decompose_glyph(glyph)
    if not strokes:
        return None
    return strokes[0].value, strokes[-1].value


def build_context_vectors(
    tokens: List[str],
    grid: SyllabaryGrid,
    axis: str = 'nucleus',
) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """
    Build distributional context vectors for nucleus or onset categories.

    For axis='nucleus': each nucleus gets a vector over onset categories,
    counting how often each onset co-occurs with that nucleus.

    For axis='onset': each onset gets a vector over nucleus categories.
    """
    # Count co-occurrences from the corpus
    cooccur: Dict[str, Counter] = defaultdict(Counter)

    for token in tokens:
        for glyph in tokenize_eva_chars(token):
            pair = _glyph_onset_nucleus(glyph)
            if pair is None:
                continue
            onset, nucleus = pair
            if axis == 'nucleus':
                cooccur[nucleus][onset] += 1
            else:
                cooccur[onset][nucleus] += 1

    # Determine feature labels (the cross-axis categories)
    if axis == 'nucleus':
        feature_labels = sorted(grid.row_labels)  # onsets
        category_labels = sorted(cooccur.keys())
    else:
        feature_labels = sorted(grid.col_labels)  # nuclei
        category_labels = sorted(cooccur.keys())

    feat_idx = {f: i for i, f in enumerate(feature_labels)}
    n_features = len(feature_labels)

    vectors = {}
    for cat in category_labels:
        vec = np.zeros(n_features)
        for feat, count in cooccur[cat].items():
            if feat in feat_idx:
                vec[feat_idx[feat]] = count
        vectors[cat] = vec

    return vectors, feature_labels


def pairwise_similarity_matrix(
    vectors: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, List[str]]:
    """Compute pairwise cosine similarity between all category vectors."""
    labels = sorted(vectors.keys())
    n = len(labels)
    matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            a = vectors[labels[i]]
            b = vectors[labels[j]]
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a > 0 and norm_b > 0:
                matrix[i][j] = float(np.dot(a, b) / (norm_a * norm_b))
            else:
                matrix[i][j] = 0.0

    return matrix, labels


def hierarchical_cluster(
    sim_matrix: np.ndarray,
    labels: List[str],
    n_clusters: int,
    linkage: str = 'average',
) -> List[List[str]]:
    """
    Agglomerative hierarchical clustering (numpy only, no scipy).
    Merges the two most similar clusters until n_clusters remain.
    """
    n = len(labels)
    if n_clusters >= n:
        return [[lab] for lab in labels]

    dist = 1.0 - sim_matrix.copy()
    np.fill_diagonal(dist, np.inf)

    # Each cluster is a list of original indices
    clusters: List[List[int]] = [[i] for i in range(n)]

    while len(clusters) > n_clusters:
        best_dist = np.inf
        best_i, best_j = 0, 1

        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                if linkage == 'single':
                    d = min(dist[a][b] for a in clusters[i] for b in clusters[j])
                elif linkage == 'complete':
                    d = max(dist[a][b] for a in clusters[i] for b in clusters[j])
                else:  # average
                    pairs = [(a, b) for a in clusters[i] for b in clusters[j]]
                    d = sum(dist[a][b] for a, b in pairs) / len(pairs)

                if d < best_dist:
                    best_dist = d
                    best_i, best_j = i, j

        # Merge best_j into best_i
        merged = clusters[best_i] + clusters[best_j]
        new_clusters = []
        for k, c in enumerate(clusters):
            if k != best_i and k != best_j:
                new_clusters.append(c)
        new_clusters.append(merged)
        clusters = new_clusters

    return [[labels[i] for i in cluster] for cluster in clusters]


def merge_grid_categories(
    grid: SyllabaryGrid,
    clusters: List[List[str]],
    axis: str = 'nucleus',
) -> SyllabaryGrid:
    """
    Rebuild the grid with merged categories.
    Cluster members are joined with '+' in the label.
    """
    # Build merge mapping: old_label -> new_label
    merge_map = {}
    for cluster in clusters:
        new_label = '+'.join(sorted(cluster))
        for old_label in cluster:
            merge_map[old_label] = new_label

    if axis == 'nucleus':
        new_col_labels = sorted(set(merge_map.values()))
        new_row_labels = list(grid.row_labels)
    else:
        new_row_labels = sorted(set(merge_map.values()))
        new_col_labels = list(grid.col_labels)

    # Rebuild cells with merged keys
    new_cells: Dict[str, List[str]] = {}
    for old_key, glyphs in grid.cells.items():
        parts = old_key.split(',', 1)
        if len(parts) != 2:
            continue
        onset, nucleus = parts

        if axis == 'nucleus':
            new_nucleus = merge_map.get(nucleus, nucleus)
            new_key = f"{onset},{new_nucleus}"
        else:
            new_onset = merge_map.get(onset, onset)
            new_key = f"{new_onset},{nucleus}"

        if new_key not in new_cells:
            new_cells[new_key] = []
        # Avoid duplicates
        for g in glyphs:
            if g not in new_cells[new_key]:
                new_cells[new_key].append(g)

    n_total = len(new_row_labels) * len(new_col_labels)
    n_filled = len(new_cells)
    occupancy = n_filled / n_total if n_total > 0 else 0.0

    return SyllabaryGrid(
        row_labels=new_row_labels,
        col_labels=new_col_labels,
        cells=new_cells,
        occupancy=occupancy,
        n_filled=n_filled,
        n_total=n_total,
    )


# We also need a version of segment_token_as_syllables that works
# with the merged grid's labels.

def segment_token_merged(
    token: str,
    onset_merge: Dict[str, str],
    nucleus_merge: Dict[str, str],
) -> List[str]:
    """Segment a token into syllable keys using merged onset/nucleus labels."""
    glyphs = tokenize_eva_chars(token)
    syllables = []
    for g in glyphs:
        strokes = decompose_glyph(g)
        if strokes:
            raw_onset = strokes[0].value
            raw_nucleus = strokes[-1].value
            onset = onset_merge.get(raw_onset, raw_onset)
            nucleus = nucleus_merge.get(raw_nucleus, raw_nucleus)
            syllables.append(f"{onset},{nucleus}")
    return syllables


# ---------------------------------------------------------------------------
# Phase B.2: Onset Merging Assessment
# ---------------------------------------------------------------------------

def assess_onset_merging(
    tokens: List[str],
    grid: SyllabaryGrid,
    similarity_threshold: float = 0.85,
) -> Optional[List[List[str]]]:
    """
    Check if any onset pairs have very high distributional similarity.
    Returns clustering if merges recommended, None otherwise.
    """
    vectors, _ = build_context_vectors(tokens, grid, axis='onset')
    sim, labels = pairwise_similarity_matrix(vectors)

    # Check if any off-diagonal pair exceeds threshold
    has_merge = False
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if sim[i][j] > similarity_threshold:
                has_merge = True
                break
        if has_merge:
            break

    if not has_merge:
        return None

    # Merge highly similar pairs
    n_target = max(3, len(labels) - 2)  # Don't merge too aggressively
    return hierarchical_cluster(sim, labels, n_clusters=n_target)


# ---------------------------------------------------------------------------
# Phase B.3: Grid Validation
# ---------------------------------------------------------------------------

@dataclass
class GridValidation:
    """Validation metrics for a candidate refined grid."""
    grid_label: str
    n_onsets: int
    n_nuclei: int
    n_cells_total: int
    n_cells_filled: int
    occupancy: float
    z_score_h2: float
    discriminates: bool
    syllable_h1: float
    syllable_h2: float
    mean_syllables_per_token: float
    overall_score: float
    nucleus_clusters: List[List[str]]
    onset_clusters: Optional[List[List[str]]] = None


def validate_grid(
    tokens: List[str],
    grid: SyllabaryGrid,
    label: str = "",
    n_shuffled: int = 50,
    nucleus_clusters: Optional[List[List[str]]] = None,
    onset_clusters: Optional[List[List[str]]] = None,
) -> GridValidation:
    """Run full validation suite on a candidate grid."""
    # Discriminant test
    disc = syllabary_discriminant_test(tokens, grid, n_shuffled=n_shuffled)

    # Syllable stats
    syl_stats = syllable_sequence_stats(tokens, grid)

    z_h2 = disc.get('z_score_h2', 0.0)
    discriminates = disc.get('discriminates', False)
    syl_h1 = syl_stats.get('syllable_h1', 0.0)
    syl_h2 = syl_stats.get('syllable_h2', 0.0)
    mean_spt = syl_stats.get('mean_syllables_per_token', 0.0)

    # Composite score
    occ_score = min(1.0, grid.occupancy / 0.60)  # target 60%
    disc_score = min(1.0, abs(z_h2) / 10.0) if z_h2 != 0 else 0.0
    syl_h2_score = 1.0 if 2.0 <= syl_h2 <= 5.0 else max(0, 1.0 - abs(syl_h2 - 3.5) / 3.5)
    spt_score = 1.0 if 1.5 <= mean_spt <= 4.0 else max(0, 1.0 - abs(mean_spt - 2.75) / 2.75)

    overall = (0.30 * occ_score + 0.25 * disc_score +
               0.25 * syl_h2_score + 0.20 * spt_score)

    return GridValidation(
        grid_label=label,
        n_onsets=len(grid.row_labels),
        n_nuclei=len(grid.col_labels),
        n_cells_total=grid.n_total,
        n_cells_filled=grid.n_filled,
        occupancy=round(grid.occupancy, 4),
        z_score_h2=round(z_h2, 2),
        discriminates=discriminates,
        syllable_h1=round(syl_h1, 4),
        syllable_h2=round(syl_h2, 4),
        mean_syllables_per_token=round(mean_spt, 2),
        overall_score=round(overall, 4),
        nucleus_clusters=nucleus_clusters or [],
        onset_clusters=onset_clusters,
    )


def sweep_cluster_counts(
    tokens: List[str],
    original_grid: SyllabaryGrid,
    nucleus_cuts: List[int] = None,
    n_shuffled: int = 50,
    verbose: bool = True,
) -> List[GridValidation]:
    """Sweep different cluster counts for nucleus merging."""
    if nucleus_cuts is None:
        nucleus_cuts = [4, 5, 6, 7]

    # Build nucleus similarity
    nuc_vectors, _ = build_context_vectors(tokens, original_grid, axis='nucleus')
    nuc_sim, nuc_labels = pairwise_similarity_matrix(nuc_vectors)

    validations: List[GridValidation] = []

    # Validate original grid first
    if verbose:
        print(f"  Validating original grid ({len(original_grid.row_labels)}x"
              f"{len(original_grid.col_labels)})...")
    orig_val = validate_grid(
        tokens, original_grid, label='original',
        n_shuffled=n_shuffled,
        nucleus_clusters=[[l] for l in original_grid.col_labels],
    )
    validations.append(orig_val)

    for n_cut in nucleus_cuts:
        if n_cut >= len(nuc_labels):
            continue
        if verbose:
            print(f"  Trying {len(original_grid.row_labels)} onsets x "
                  f"{n_cut} nuclei...", end='', flush=True)

        clusters = hierarchical_cluster(nuc_sim, nuc_labels, n_clusters=n_cut)
        merged_grid = merge_grid_categories(original_grid, clusters, axis='nucleus')

        # Also check onset merging
        onset_merging = assess_onset_merging(tokens, merged_grid)
        onset_clusters_used = None
        if onset_merging is not None:
            merged_grid = merge_grid_categories(merged_grid, onset_merging, axis='onset')
            onset_clusters_used = onset_merging
            label = f"{len(merged_grid.row_labels)}x{n_cut}_onset_merged"
        else:
            label = f"{len(original_grid.row_labels)}x{n_cut}"

        val = validate_grid(
            tokens, merged_grid, label=label,
            n_shuffled=n_shuffled,
            nucleus_clusters=clusters,
            onset_clusters=onset_clusters_used,
        )
        validations.append(val)

        if verbose:
            print(f" occ={val.occupancy:.1%}, z_h2={val.z_score_h2:.1f}, "
                  f"syl_h2={val.syllable_h2:.3f}, score={val.overall_score:.3f}")

    validations.sort(key=lambda v: v.overall_score, reverse=True)
    return validations


# ---------------------------------------------------------------------------
# Phase B.4: Language Narrowing
# ---------------------------------------------------------------------------

LANGUAGE_GRID_EXPECTATIONS = {
    'japanese_like': {
        'onsets': (8, 12), 'nuclei': (4, 6),
        'description': 'Small CV syllabary (e.g., Japanese kana)',
    },
    'romance_simple': {
        'onsets': (8, 15), 'nuclei': (4, 6),
        'description': 'Medium CV syllabary (Italian, Spanish — simple phonotactics)',
    },
    'latin_classical': {
        'onsets': (12, 18), 'nuclei': (4, 7),
        'description': 'Large CV syllabary (Latin — more consonant clusters)',
    },
    'semitic': {
        'onsets': (15, 25), 'nuclei': (3, 5),
        'description': 'Consonant-heavy (Arabic, Hebrew — many onsets, few vowels)',
    },
    'germanic': {
        'onsets': (12, 20), 'nuclei': (5, 8),
        'description': 'Large mixed syllabary (German — many clusters and vowels)',
    },
}


def narrow_language(grid: SyllabaryGrid) -> List[Dict]:
    """Rank plausible source language families by grid dimension fit."""
    n_onsets = len(grid.row_labels)
    n_nuclei = len(grid.col_labels)
    rankings = []

    for family, params in LANGUAGE_GRID_EXPECTATIONS.items():
        o_lo, o_hi = params['onsets']
        n_lo, n_hi = params['nuclei']

        # Score based on how well dimensions fit the expected range
        o_score = 1.0
        if n_onsets < o_lo:
            o_score = max(0, 1.0 - (o_lo - n_onsets) / o_lo)
        elif n_onsets > o_hi:
            o_score = max(0, 1.0 - (n_onsets - o_hi) / o_hi)

        n_score = 1.0
        if n_nuclei < n_lo:
            n_score = max(0, 1.0 - (n_lo - n_nuclei) / n_lo)
        elif n_nuclei > n_hi:
            n_score = max(0, 1.0 - (n_nuclei - n_hi) / n_hi)

        combined = (o_score + n_score) / 2

        rankings.append({
            'family': family,
            'score': round(combined, 3),
            'description': params['description'],
            'grid_onsets': n_onsets,
            'grid_nuclei': n_nuclei,
            'expected_onsets': f"{o_lo}-{o_hi}",
            'expected_nuclei': f"{n_lo}-{n_hi}",
        })

    rankings.sort(key=lambda r: r['score'], reverse=True)
    return rankings


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_grid_refinement() -> Dict:
    """Run the full syllabary grid refinement pipeline."""
    print("=" * 70)
    print("PHASE 2B: SYLLABARY GRID REFINEMENT")
    print("=" * 70)

    # --- Load corpus and build original grid ---
    corpus = load_corpus(verbose=False)
    tokens = corpus.get_tokens(paragraph_only=True)
    original_grid = build_ventris_grid(tokens)

    print(f"\n  Original grid: {len(original_grid.row_labels)} onsets x "
          f"{len(original_grid.col_labels)} nuclei = "
          f"{original_grid.n_total} cells, "
          f"{original_grid.n_filled} filled ({original_grid.occupancy:.1%})")
    print(f"  Onsets:  {original_grid.row_labels}")
    print(f"  Nuclei:  {original_grid.col_labels}")

    # --- Phase B.1: Nucleus distributional clustering ---
    print("\n--- Phase B.1: Nucleus Distributional Similarity ---")
    nuc_vectors, nuc_features = build_context_vectors(
        tokens, original_grid, axis='nucleus',
    )
    nuc_sim, nuc_labels = pairwise_similarity_matrix(nuc_vectors)

    # Print similarity matrix
    print(f"\n  Nucleus pairwise cosine similarity:")
    header = f"  {'':>12}" + ''.join(f" {l[:8]:>8}" for l in nuc_labels)
    print(header)
    for i, lab in enumerate(nuc_labels):
        row = f"  {lab:>12}"
        for j in range(len(nuc_labels)):
            val = nuc_sim[i][j]
            row += f" {val:>8.3f}"
        print(row)

    # Show clustering at each cut
    print(f"\n  Clustering results:")
    for n_cut in [4, 5, 6, 7]:
        if n_cut >= len(nuc_labels):
            continue
        clusters = hierarchical_cluster(nuc_sim, nuc_labels, n_clusters=n_cut)
        cluster_strs = ['+'.join(sorted(c)) for c in clusters]
        print(f"    {n_cut} clusters: {cluster_strs}")

    # --- Phase B.2: Onset merging assessment ---
    print("\n--- Phase B.2: Onset Merging Assessment ---")
    onset_vectors, onset_features = build_context_vectors(
        tokens, original_grid, axis='onset',
    )
    onset_sim, onset_labels = pairwise_similarity_matrix(onset_vectors)

    # Print onset similarity
    print(f"\n  Onset pairwise cosine similarity:")
    header = f"  {'':>12}" + ''.join(f" {l[:8]:>8}" for l in onset_labels)
    print(header)
    for i, lab in enumerate(onset_labels):
        row = f"  {lab:>12}"
        for j in range(len(onset_labels)):
            val = onset_sim[i][j]
            row += f" {val:>8.3f}"
        print(row)

    # Check for high-similarity pairs
    high_pairs = []
    for i in range(len(onset_labels)):
        for j in range(i + 1, len(onset_labels)):
            if onset_sim[i][j] > 0.85:
                high_pairs.append((onset_labels[i], onset_labels[j],
                                   onset_sim[i][j]))
    if high_pairs:
        print(f"\n  High-similarity onset pairs (>0.85):")
        for a, b, sim in high_pairs:
            print(f"    {a} - {b}: {sim:.3f}")
    else:
        print(f"\n  No onset pairs exceed 0.85 similarity — keeping 7 onsets.")

    # --- Phase B.3: Grid validation sweep ---
    print("\n--- Phase B.3: Grid Validation Sweep ---")
    validations = sweep_cluster_counts(
        tokens, original_grid, n_shuffled=50, verbose=True,
    )

    print(f"\n  {'Label':<25} {'Grid':>8} {'Occ':>6} {'z(H2)':>7} "
          f"{'Syl H2':>7} {'Syl/Tok':>7} {'Score':>6}")
    print(f"  {'-' * 68}")
    for v in validations:
        grid_str = f"{v.n_onsets}x{v.n_nuclei}"
        print(f"  {v.grid_label:<25} {grid_str:>8} {v.occupancy:>5.1%} "
              f"{v.z_score_h2:>7.1f} {v.syllable_h2:>7.3f} "
              f"{v.mean_syllables_per_token:>7.2f} {v.overall_score:>6.3f}")

    best = validations[0]
    print(f"\n  Best grid: {best.grid_label} "
          f"({best.n_onsets}x{best.n_nuclei}, {best.occupancy:.1%} occupancy, "
          f"score={best.overall_score:.3f})")
    if best.nucleus_clusters:
        print(f"  Nucleus clusters:")
        for cluster in best.nucleus_clusters:
            print(f"    {'+'.join(sorted(cluster))}")
    if best.onset_clusters:
        print(f"  Onset clusters:")
        for cluster in best.onset_clusters:
            print(f"    {'+'.join(sorted(cluster))}")

    # --- Phase B.4: Language narrowing ---
    print("\n--- Phase B.4: Language Narrowing ---")

    # Rebuild the best grid for language analysis
    nuc_clusters_best = best.nucleus_clusters
    if nuc_clusters_best and any(len(c) > 1 for c in nuc_clusters_best):
        best_merged_grid = merge_grid_categories(
            original_grid, nuc_clusters_best, axis='nucleus',
        )
        if best.onset_clusters:
            best_merged_grid = merge_grid_categories(
                best_merged_grid, best.onset_clusters, axis='onset',
            )
    else:
        best_merged_grid = original_grid

    rankings = narrow_language(best_merged_grid)

    print(f"\n  Grid dimensions: {len(best_merged_grid.row_labels)} onsets x "
          f"{len(best_merged_grid.col_labels)} nuclei")
    print(f"\n  {'Family':<20} {'Score':>6} {'Expected':>16} {'Description'}")
    print(f"  {'-' * 70}")
    for r in rankings:
        exp = f"{r['expected_onsets']}x{r['expected_nuclei']}"
        print(f"  {r['family']:<20} {r['score']:>6.3f} {exp:>16} {r['description']}")

    # --- Save results ---
    rd = _results_dir()

    with open(os.path.join(rd, 'grid_similarity_matrices.json'), 'w') as f:
        json.dump({
            'nucleus_similarity': {
                'labels': nuc_labels,
                'matrix': nuc_sim.tolist(),
            },
            'onset_similarity': {
                'labels': onset_labels,
                'matrix': onset_sim.tolist(),
            },
        }, f, indent=2)

    with open(os.path.join(rd, 'grid_candidates.json'), 'w') as f:
        json.dump({
            'candidates': [asdict(v) for v in validations],
            'best': best.grid_label,
        }, f, indent=2)

    # Save best grid detail
    with open(os.path.join(rd, 'grid_refined_best.json'), 'w') as f:
        json.dump({
            'row_labels': best_merged_grid.row_labels,
            'col_labels': best_merged_grid.col_labels,
            'cells': best_merged_grid.cells,
            'occupancy': best_merged_grid.occupancy,
            'n_filled': best_merged_grid.n_filled,
            'n_total': best_merged_grid.n_total,
            'nucleus_clusters': nuc_clusters_best,
            'onset_clusters': best.onset_clusters,
        }, f, indent=2)

    with open(os.path.join(rd, 'language_narrowing.json'), 'w') as f:
        json.dump({
            'grid_dimensions': {
                'onsets': len(best_merged_grid.row_labels),
                'nuclei': len(best_merged_grid.col_labels),
            },
            'rankings': rankings,
        }, f, indent=2)

    print(f"\n  Results saved to {rd}/")

    return {
        'validations': validations,
        'best_grid': best_merged_grid,
        'language_rankings': rankings,
    }
