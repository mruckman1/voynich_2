"""
Phase 5.1: Paradigm Discovery
==============================
Group Voynich tokens by shared stems, catalog affix variations, classify
and cluster paradigm shapes.

A paradigm is a set of tokens sharing the same stem but differing in
prefix/suffix — analogous to inflectional paradigms in natural language
(e.g., herba/herbae/herbam in Latin).

Sub-analyses:
  5.1a — Stem equivalence classes (exact + grid-cell grouping)
  5.1b — Paradigm shape classification and clustering

Output:
  results/paradigm_discovery.json
"""

import json
import math
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster

from voynich.core.corpus import load_corpus, tokenize_eva_chars
from voynich.core.stats import (
    bootstrap_ci, selectivity_ratio, paradigm_shape_vector,
)
from voynich.core._paths import results_dir as _results_dir
from voynich.analysis.strokes import (
    decompose_glyph, SyllabaryGrid, segment_token_as_syllables,
    build_ventris_grid,
)
from voynich.phases.morpheme_grid import (
    decompose_token_morphemes, decompose_corpus_morphemes,
    MorphemeDecomposition, MorphemeStats,
    KNOWN_PREFIXES, KNOWN_SUFFIXES,
)
from voynich.phases.grid_validate import build_grid_from_tokens


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class StemParadigm:
    """A stem and all its attested forms."""
    stem: str
    forms: List[str]          # Unique token types with this stem
    token_count: int          # Total occurrences across all forms
    prefixes: List[str]       # Unique prefixes attested
    suffixes: List[str]       # Unique suffixes attested
    n_forms: int              # len(forms)
    paradigm_shape: Tuple[int, int]  # (n_prefix_types, n_suffix_types)


@dataclass
class ParadigmCluster:
    """A cluster of paradigms with similar shapes."""
    cluster_id: int
    n_paradigms: int
    mean_n_forms: float
    mean_n_suffixes: float
    mean_n_prefixes: float
    representative_stems: List[str]  # Top-3 stems by frequency
    member_stems: List[str]          # All stems in cluster


@dataclass
class ParadigmDiscoveryResult:
    """Full Phase 5.1 output."""
    n_stems: int
    n_paradigms_with_affixes: int   # Stems with >= 2 forms
    n_singleton_stems: int          # Stems with exactly 1 form
    paradigm_size_distribution: Dict[int, int]
    mean_paradigm_size: float
    median_paradigm_size: float
    top_20_paradigms: List[Dict]
    # Grid-cell equivalence
    n_stems_grid_merged: int
    mean_paradigm_size_grid: float
    # Clustering
    n_clusters: int
    clusters: List[Dict]
    # Null test
    real_mean_paradigm_size: float
    null_mean: float
    null_std: float
    selectivity_ratio: float
    selectivity_z: float
    # Gate
    gate_passed: bool
    verdict: str


# ---------------------------------------------------------------------------
# 5.1a: Stem equivalence classes
# ---------------------------------------------------------------------------

def group_stems(
    decompositions: List[MorphemeDecomposition],
    token_counts: Dict[str, int],
) -> Dict[str, StemParadigm]:
    """
    Group tokens by exact stem match.

    Args:
        decompositions: Per-token morpheme decompositions
        token_counts: Token type -> corpus frequency

    Returns:
        Dict mapping stem string -> StemParadigm
    """
    stem_groups: Dict[str, List[MorphemeDecomposition]] = defaultdict(list)
    for d in decompositions:
        if d.stem:
            stem_groups[d.stem].append(d)

    paradigms: Dict[str, StemParadigm] = {}
    for stem, group in stem_groups.items():
        if not stem:
            continue
        forms = sorted(set(d.token for d in group))
        prefixes = sorted(set(d.prefix for d in group if d.prefix))
        suffixes = sorted(set(d.suffix for d in group if d.suffix))
        total_count = sum(token_counts.get(d.token, 1) for d in group)
        paradigms[stem] = StemParadigm(
            stem=stem,
            forms=forms,
            token_count=total_count,
            prefixes=prefixes,
            suffixes=suffixes,
            n_forms=len(forms),
            paradigm_shape=(len(prefixes), len(suffixes)),
        )

    return paradigms


def group_stems_by_grid_cell(
    decompositions: List[MorphemeDecomposition],
    token_counts: Dict[str, int],
    grid: SyllabaryGrid,
) -> Dict[str, StemParadigm]:
    """
    Group tokens by grid-cell equivalence of their stem characters.

    Two stems are equivalent if they map to the same sequence of
    grid cells (onset, nucleus). This merges allographic variants.
    """
    # Map each stem string to its grid-cell key
    stem_to_cell_key: Dict[str, str] = {}
    for d in decompositions:
        if not d.stem or d.stem in stem_to_cell_key:
            continue
        cells = segment_token_as_syllables(d.stem, grid)
        cell_key = '|'.join(cells) if cells else d.stem
        stem_to_cell_key[d.stem] = cell_key

    # Group decompositions by cell key
    cell_groups: Dict[str, List[MorphemeDecomposition]] = defaultdict(list)
    for d in decompositions:
        if d.stem:
            key = stem_to_cell_key.get(d.stem, d.stem)
            cell_groups[key].append(d)

    # Build paradigms using the most frequent stem as canonical
    paradigms: Dict[str, StemParadigm] = {}
    for cell_key, group in cell_groups.items():
        # Pick canonical stem = the most frequent stem string
        stem_freq: Counter = Counter()
        for d in group:
            stem_freq[d.stem] += token_counts.get(d.token, 1)
        canonical_stem = stem_freq.most_common(1)[0][0]

        forms = sorted(set(d.token for d in group))
        prefixes = sorted(set(d.prefix for d in group if d.prefix))
        suffixes = sorted(set(d.suffix for d in group if d.suffix))
        total_count = sum(token_counts.get(d.token, 1) for d in group)

        paradigms[canonical_stem] = StemParadigm(
            stem=canonical_stem,
            forms=forms,
            token_count=total_count,
            prefixes=prefixes,
            suffixes=suffixes,
            n_forms=len(forms),
            paradigm_shape=(len(prefixes), len(suffixes)),
        )

    return paradigms


# ---------------------------------------------------------------------------
# 5.1b: Paradigm shape classification and clustering
# ---------------------------------------------------------------------------

def classify_paradigm_shapes(
    paradigms: Dict[str, StemParadigm],
) -> Dict[Tuple[int, int], List[str]]:
    """Group paradigms by their shape (n_prefix_types, n_suffix_types)."""
    shape_groups: Dict[Tuple[int, int], List[str]] = defaultdict(list)
    for stem, p in paradigms.items():
        shape_groups[p.paradigm_shape].append(stem)
    return dict(shape_groups)


def cluster_paradigm_shapes(
    paradigms: Dict[str, StemParadigm],
    n_clusters: int = 5,
) -> List[ParadigmCluster]:
    """
    Hierarchical clustering of paradigm shapes.

    Uses paradigm_shape_vector() from stats.py to build feature vectors,
    then Ward's linkage to find n_clusters natural groupings.
    """
    # Only cluster paradigms with >= 2 forms (singletons are trivial)
    eligible = {s: p for s, p in paradigms.items() if p.n_forms >= 2}

    if len(eligible) < n_clusters:
        # Not enough paradigms to cluster
        return [ParadigmCluster(
            cluster_id=0,
            n_paradigms=len(eligible),
            mean_n_forms=float(np.mean([p.n_forms for p in eligible.values()])) if eligible else 0,
            mean_n_suffixes=float(np.mean([len(p.suffixes) for p in eligible.values()])) if eligible else 0,
            mean_n_prefixes=float(np.mean([len(p.prefixes) for p in eligible.values()])) if eligible else 0,
            representative_stems=sorted(eligible, key=lambda s: eligible[s].token_count, reverse=True)[:3],
            member_stems=list(eligible.keys()),
        )]

    stems_list = sorted(eligible.keys())
    vectors = np.array([
        paradigm_shape_vector(
            eligible[s].n_forms,
            set(eligible[s].suffixes),
            set(eligible[s].prefixes),
        )
        for s in stems_list
    ])

    # Normalize features to [0, 1] for balanced clustering
    col_min = vectors.min(axis=0)
    col_max = vectors.max(axis=0)
    col_range = col_max - col_min
    col_range[col_range == 0] = 1.0
    vectors_norm = (vectors - col_min) / col_range

    # Ward's linkage
    Z = linkage(vectors_norm, method='ward')
    labels = fcluster(Z, t=n_clusters, criterion='maxclust')

    # Build cluster objects
    clusters: List[ParadigmCluster] = []
    for cid in range(1, n_clusters + 1):
        member_idx = [i for i, l in enumerate(labels) if l == cid]
        if not member_idx:
            continue
        members = [stems_list[i] for i in member_idx]
        member_paradigms = [eligible[s] for s in members]

        # Representative = top-3 by frequency
        reps = sorted(members, key=lambda s: eligible[s].token_count, reverse=True)[:3]

        clusters.append(ParadigmCluster(
            cluster_id=cid,
            n_paradigms=len(members),
            mean_n_forms=float(np.mean([p.n_forms for p in member_paradigms])),
            mean_n_suffixes=float(np.mean([len(p.suffixes) for p in member_paradigms])),
            mean_n_prefixes=float(np.mean([len(p.prefixes) for p in member_paradigms])),
            representative_stems=reps,
            member_stems=members,
        ))

    # Sort by size descending
    clusters.sort(key=lambda c: c.n_paradigms, reverse=True)
    return clusters


# ---------------------------------------------------------------------------
# Null test
# ---------------------------------------------------------------------------

def null_test_paradigm_selectivity(
    tokens: List[str],
    real_mean_paradigm_size: float,
    n_trials: int = 100,
    seed: int = 42,
) -> Tuple[float, float, float, float]:
    """
    Null test: shuffle characters within each token, re-decompose, measure
    mean paradigm size. If real > shuffled by > 1.5x, structure is real.

    Returns: (null_mean, null_std, z_score, selectivity)
    """
    rng = random.Random(seed)
    null_means: List[float] = []

    for trial in range(n_trials):
        # Shuffle characters within each token
        shuffled_tokens = []
        for t in tokens:
            chars = list(t)
            rng.shuffle(chars)
            shuffled_tokens.append(''.join(chars))

        # Decompose shuffled tokens
        shuffled_decomps = [decompose_token_morphemes(t) for t in shuffled_tokens]

        # Group by stem
        stem_groups: Dict[str, set] = defaultdict(set)
        for d in shuffled_decomps:
            if d.stem:
                stem_groups[d.stem].add(d.token)

        # Mean paradigm size (forms per stem)
        sizes = [len(forms) for forms in stem_groups.values()]
        null_means.append(float(np.mean(sizes)) if sizes else 1.0)

    null_arr = np.array(null_means)
    null_mean = float(np.mean(null_arr))
    null_std = float(np.std(null_arr))

    z = (real_mean_paradigm_size - null_mean) / null_std if null_std > 0 else 0.0
    sel = selectivity_ratio(real_mean_paradigm_size, null_arr)

    return null_mean, null_std, z, sel


# ---------------------------------------------------------------------------
# Gate check
# ---------------------------------------------------------------------------

def _check_gate(
    name: str, value: float, threshold: float, direction: str = 'greater',
) -> Tuple[bool, str]:
    """Check a single gate condition."""
    if direction == 'greater':
        passed = value > threshold
        op = '>'
    else:
        passed = value < threshold
        op = '<'
    status = 'PASSED' if passed else 'FAILED'
    return passed, f"  Gate [{name}]: {value:.4f} {op} {threshold} -> {status}"


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_paradigm_discovery() -> Dict:
    """
    Run Phase 5.1: Paradigm Discovery.

    1. Load corpus (Language A, paragraph_only)
    2. Decompose tokens into morphemes
    3. Group by exact stems
    4. Group by grid-cell equivalence
    5. Classify and cluster paradigm shapes
    6. Null test: selectivity vs shuffled text
    7. Gate: selectivity > 1.5
    """
    print("=" * 70)
    print("Phase 5.1: Paradigm Discovery")
    print("=" * 70)

    # 1. Load corpus
    corpus = load_corpus(verbose=False)
    tokens = corpus.get_tokens(language='A', paragraph_only=True)
    if not tokens:
        # Fallback: all tokens
        tokens = corpus.get_tokens(paragraph_only=True)
    print(f"\n  Corpus: {len(tokens):,} tokens (Language A)")

    # 2. Decompose
    decompositions, morph_stats = decompose_corpus_morphemes(tokens)
    print(f"  Morpheme decomposition: {morph_stats.n_stem_types} stem types")
    print(f"    {morph_stats.pct_with_prefix:.1f}% with prefix, "
          f"{morph_stats.pct_with_suffix:.1f}% with suffix")

    # Token frequency counts
    token_counts = Counter(tokens)

    # Deduplicate decompositions by token type
    seen_tokens: set = set()
    unique_decomps: List[MorphemeDecomposition] = []
    for d in decompositions:
        if d.token not in seen_tokens:
            seen_tokens.add(d.token)
            unique_decomps.append(d)

    # 3. Group by exact stems
    print("\n  5.1a: Stem equivalence classes")
    paradigms = group_stems(unique_decomps, token_counts)
    n_stems = len(paradigms)

    # Paradigm size statistics
    sizes = [p.n_forms for p in paradigms.values()]
    size_dist: Dict[int, int] = Counter(sizes)
    mean_size = float(np.mean(sizes)) if sizes else 0.0
    median_size = float(np.median(sizes)) if sizes else 0.0
    n_multi = sum(1 for s in sizes if s >= 2)
    n_singleton = sum(1 for s in sizes if s == 1)

    print(f"    Exact stem groups: {n_stems}")
    print(f"    Paradigms (>= 2 forms): {n_multi}")
    print(f"    Singletons: {n_singleton}")
    print(f"    Mean paradigm size: {mean_size:.2f}")
    print(f"    Median paradigm size: {median_size:.1f}")

    # Top-20 paradigms by token count
    top_20 = sorted(paradigms.values(), key=lambda p: p.token_count, reverse=True)[:20]
    print("\n    Top-20 paradigms by frequency:")
    for i, p in enumerate(top_20[:10], 1):
        forms_str = ', '.join(p.forms[:5])
        if len(p.forms) > 5:
            forms_str += f' ... (+{len(p.forms) - 5})'
        print(f"      {i:2d}. stem='{p.stem}' ({p.token_count} tokens, "
              f"{p.n_forms} forms): {forms_str}")

    # 4. Grid-cell equivalence grouping
    print("\n    Grid-cell equivalence grouping:")
    grid = build_grid_from_tokens(tokens)
    grid_paradigms = group_stems_by_grid_cell(unique_decomps, token_counts, grid)
    n_grid = len(grid_paradigms)
    grid_sizes = [p.n_forms for p in grid_paradigms.values()]
    mean_grid_size = float(np.mean(grid_sizes)) if grid_sizes else 0.0
    print(f"    Grid-merged stem groups: {n_grid} (from {n_stems} exact)")
    print(f"    Mean paradigm size (grid): {mean_grid_size:.2f}")

    # 5. Cluster paradigm shapes
    print("\n  5.1b: Paradigm shape clustering")
    clusters = cluster_paradigm_shapes(paradigms, n_clusters=5)
    print(f"    Found {len(clusters)} shape clusters:")
    for c in clusters:
        reps = ', '.join(c.representative_stems[:3])
        print(f"      Cluster {c.cluster_id}: {c.n_paradigms} paradigms, "
              f"mean {c.mean_n_forms:.1f} forms, "
              f"{c.mean_n_suffixes:.1f} suffixes, "
              f"{c.mean_n_prefixes:.1f} prefixes")
        print(f"        Representatives: {reps}")

    # 6. Null test
    print("\n  Null test: paradigm selectivity vs shuffled text")
    null_mean, null_std, z, sel = null_test_paradigm_selectivity(
        tokens, mean_size, n_trials=100, seed=42,
    )
    print(f"    Real mean paradigm size: {mean_size:.4f}")
    print(f"    Null mean: {null_mean:.4f} +/- {null_std:.4f}")
    print(f"    z-score: {z:.2f}")
    print(f"    Selectivity ratio: {sel:.2f}x")

    # 7. Gate
    gate_ok, gate_msg = _check_gate(
        'paradigm_selectivity', sel, 1.5, 'greater',
    )
    print(f"\n{gate_msg}")
    verdict = 'paradigmatic_structure_confirmed' if gate_ok else 'gate_failed'
    print(f"  Verdict: {verdict}")

    # Build result
    result = ParadigmDiscoveryResult(
        n_stems=n_stems,
        n_paradigms_with_affixes=n_multi,
        n_singleton_stems=n_singleton,
        paradigm_size_distribution=dict(size_dist),
        mean_paradigm_size=round(mean_size, 4),
        median_paradigm_size=round(median_size, 4),
        top_20_paradigms=[asdict(p) for p in top_20],
        n_stems_grid_merged=n_grid,
        mean_paradigm_size_grid=round(mean_grid_size, 4),
        n_clusters=len(clusters),
        clusters=[asdict(c) for c in clusters],
        real_mean_paradigm_size=round(mean_size, 4),
        null_mean=round(null_mean, 4),
        null_std=round(null_std, 4),
        selectivity_ratio=round(sel, 4),
        selectivity_z=round(z, 2),
        gate_passed=gate_ok,
        verdict=verdict,
    )

    # Save
    out_path = os.path.join(_results_dir(), 'paradigm_discovery.json')
    with open(out_path, 'w') as f:
        json.dump(asdict(result), f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")

    return asdict(result)
