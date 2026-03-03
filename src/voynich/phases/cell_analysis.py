"""
Phase 14.1 – Within-Cell Character Analysis
=============================================
For each of the 14 occupied grid cells, compute distributional vectors per
EVA glyph and cluster them.  Characters with cosine similarity > 0.8 are
treated as allographic variants of the same phoneme; characters with
similarity < 0.8 are treated as distinct phonemes that the grid collapsed.

Reports the effective phoneme count per cell and the total refined inventory
size.  The key question is whether the total falls in 20–30 (matching the
Romance language phoneme inventory).

Dependency chain:
    cv_labels.json (Phase 3 grid)
        → cell_analysis.json (this step)
"""

import json
import math
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_cell_lookup,
    load_corpus,
    tokenize_eva_chars,
)
from voynich.core.reference import EVA_VISUAL_COMPONENTS


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
class GlyphDistributionVector:
    """Distributional profile for a single EVA glyph."""
    glyph: str
    cell_key: str
    freq: int
    pos_initial: float       # fraction at word-initial position
    pos_medial: float
    pos_final: float
    pos_solo: float          # fraction as single-char word
    right_entropy: float     # entropy over right-neighbour distribution
    left_entropy: float      # entropy over left-neighbour distribution
    top5_right: List[str]    # most common right-neighbour glyphs
    top5_left: List[str]     # most common left-neighbour glyphs


@dataclass
class CellClusterResult:
    """Clustering analysis for one grid cell."""
    cell_key: str
    cv_label: str
    glyphs: List[str]
    n_glyphs: int
    pairwise_cosine: Dict[str, float]   # "g1:g2" -> cosine similarity
    allograph_pairs: List[List[str]]    # cosine > 0.8 (same phoneme)
    distinct_pairs: List[List[str]]     # cosine < 0.7 (distinct phonemes)
    n_distinct_phonemes: int            # number of distributional clusters
    cluster_assignments: Dict[str, int] # glyph -> cluster_id


@dataclass
class CellAnalysisResult:
    """Full analysis across all 14 cells."""
    n_cells: int
    n_glyphs_total: int
    cell_results: List[Dict]
    inventory_size_phase11: int         # 14 (fixed, one phoneme per cell)
    inventory_size_phase14_triples: int # unique (first, last, class) triples
    inventory_size_phase14_clusters: int # from distributional clustering
    gate_passed: bool                   # 20 <= clusters <= 30
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Vector construction
# ---------------------------------------------------------------------------

def _build_distribution_vectors(
    cv_labels: Dict,
    tokens: List[str],
) -> Dict[str, GlyphDistributionVector]:
    """Build a distributional vector for every EVA glyph that appears in cv_labels."""
    eva_to_cell = build_eva_to_cell_lookup(cv_labels)
    all_glyphs = set(eva_to_cell.keys())

    # Accumulate raw counts
    freq: Counter = Counter()
    pos_initial: Counter = Counter()
    pos_medial: Counter = Counter()
    pos_final: Counter = Counter()
    pos_solo: Counter = Counter()
    right_neighbours: Dict[str, Counter] = {g: Counter() for g in all_glyphs}
    left_neighbours: Dict[str, Counter] = {g: Counter() for g in all_glyphs}

    for token in tokens:
        chars = tokenize_eva_chars(token)
        # Only keep known glyphs
        chars = [c for c in chars if c in all_glyphs]
        n = len(chars)
        if n == 0:
            continue
        for i, ch in enumerate(chars):
            freq[ch] += 1
            if n == 1:
                pos_solo[ch] += 1
            elif i == 0:
                pos_initial[ch] += 1
            elif i == n - 1:
                pos_final[ch] += 1
            else:
                pos_medial[ch] += 1
            if i > 0:
                left_neighbours[ch][chars[i - 1]] += 1
            if i < n - 1:
                right_neighbours[ch][chars[i + 1]] += 1

    def _entropy(counter: Counter) -> float:
        total = sum(counter.values())
        if total == 0:
            return 0.0
        h = 0.0
        for v in counter.values():
            p = v / total
            if p > 0:
                h -= p * math.log2(p)
        return h

    vectors: Dict[str, GlyphDistributionVector] = {}
    for glyph in all_glyphs:
        total = freq[glyph]
        if total == 0:
            continue
        vectors[glyph] = GlyphDistributionVector(
            glyph=glyph,
            cell_key=eva_to_cell[glyph],
            freq=total,
            pos_initial=pos_initial[glyph] / total,
            pos_medial=pos_medial[glyph] / total,
            pos_final=pos_final[glyph] / total,
            pos_solo=pos_solo[glyph] / total,
            right_entropy=_entropy(right_neighbours[glyph]),
            left_entropy=_entropy(left_neighbours[glyph]),
            top5_right=[g for g, _ in right_neighbours[glyph].most_common(5)],
            top5_left=[g for g, _ in left_neighbours[glyph].most_common(5)],
        )
    return vectors


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------

def _glyph_feature_vector(v: GlyphDistributionVector) -> List[float]:
    """6-dimensional vector for cosine computation."""
    return [
        v.pos_initial,
        v.pos_medial,
        v.pos_final,
        v.pos_solo,
        v.right_entropy / 5.0,   # normalise to ~[0,1]
        v.left_entropy / 5.0,
    ]


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na < 1e-9 or nb < 1e-9:
        return 1.0  # treat zero-vector as identical (same phoneme)
    return dot / (na * nb)


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def _cluster_glyphs(
    glyphs: List[str],
    vectors: Dict[str, GlyphDistributionVector],
    threshold: float = 0.8,
) -> Tuple[Dict[str, float], List[List[str]], List[List[str]], Dict[str, int], int]:
    """Greedy single-linkage clustering of glyphs within a cell.

    Two glyphs are in the same cluster if their cosine similarity >= threshold.

    Returns
    -------
    pairwise_cosine, allograph_pairs, distinct_pairs, cluster_assignments, n_clusters
    """
    # Only cluster glyphs that appear in the corpus
    active = [g for g in glyphs if g in vectors]

    pairwise: Dict[str, float] = {}
    allograph_pairs: List[List[str]] = []
    distinct_pairs: List[List[str]] = []

    for i in range(len(active)):
        for j in range(i + 1, len(active)):
            g1, g2 = active[i], active[j]
            va = _glyph_feature_vector(vectors[g1])
            vb = _glyph_feature_vector(vectors[g2])
            c = _cosine(va, vb)
            pairwise[f"{g1}:{g2}"] = round(c, 4)
            if c >= 0.8:
                allograph_pairs.append([g1, g2])
            elif c < 0.7:
                distinct_pairs.append([g1, g2])

    # Single-linkage clustering: start with each glyph in its own cluster
    cluster_id: Dict[str, int] = {g: i for i, g in enumerate(active)}

    # Add glyphs that have no corpus data as singletons
    for g in glyphs:
        if g not in cluster_id:
            cluster_id[g] = max(cluster_id.values(), default=-1) + 1

    # Merge any pair with cosine >= threshold
    for pair_key, cos in pairwise.items():
        if cos >= threshold:
            g1, g2 = pair_key.split(':', 1)
            # Union-find merge
            c1 = cluster_id[g1]
            c2 = cluster_id[g2]
            if c1 != c2:
                # Renumber all c2 members to c1
                for g in cluster_id:
                    if cluster_id[g] == c2:
                        cluster_id[g] = c1

    # Compact cluster IDs
    unique_ids = sorted(set(cluster_id.values()))
    remap = {old: new for new, old in enumerate(unique_ids)}
    cluster_id = {g: remap[cid] for g, cid in cluster_id.items()}
    n_clusters = len(unique_ids)

    return pairwise, allograph_pairs, distinct_pairs, cluster_id, n_clusters


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_cell_analysis() -> None:
    """Step 14.1: within-cell character analysis."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE 14.1: Within-Cell Character Analysis")
    print("=" * 70)

    rd = _results_dir()
    cv_path = os.path.join(rd, 'cv_labels.json')
    if not os.path.exists(cv_path):
        print("  [SKIP] cv_labels.json not found")
        return

    with open(cv_path) as f:
        cv_labels = json.load(f)

    # Load Language A tokens
    corpus = load_corpus(verbose=False)
    tokens = corpus.get_tokens(language='A', paragraph_only=True)
    if not tokens:
        print("  [SKIP] No Language A tokens found")
        return

    print(f"\n  Grid cells: {len(cv_labels)}")
    print(f"  Language A tokens: {len(tokens)}")

    # Build distributional vectors for all glyphs
    print("\n  Building distributional vectors...")
    vectors = _build_distribution_vectors(cv_labels, tokens)
    print(f"  Glyph vectors built: {len(vectors)}")

    # Count unique feature triples (for comparison)
    triple_set: set = set()
    for glyph, comp in EVA_VISUAL_COMPONENTS.items():
        triple_key = (
            comp['first_stroke'] + ','
            + comp['last_stroke'] + ','
            + comp['glyph_class']
        )
        triple_set.add(triple_key)
    n_triples = len(triple_set)

    # Analyse each cell
    cell_results: List[Dict] = []
    total_distinct = 0
    total_glyphs = 0

    print("\n  Cell-by-cell cluster analysis:")
    print(f"  {'Cell':<30} {'Glyphs':<30} {'Clusters'}")

    for cell_key, info in sorted(cv_labels.items(), key=lambda x: -x[1].get('frequency', 0)):
        glyphs = info.get('glyphs', [])
        cv_label = info.get('cv_label', cell_key)
        total_glyphs += len(glyphs)

        if len(glyphs) == 1:
            # Single glyph — trivially one phoneme, no clustering needed
            cluster_result = CellClusterResult(
                cell_key=cell_key,
                cv_label=cv_label,
                glyphs=glyphs,
                n_glyphs=1,
                pairwise_cosine={},
                allograph_pairs=[],
                distinct_pairs=[],
                n_distinct_phonemes=1,
                cluster_assignments={glyphs[0]: 0} if glyphs else {},
            )
        else:
            pairwise, allograph_pairs, distinct_pairs, cluster_assignments, n_clusters = (
                _cluster_glyphs(glyphs, vectors)
            )
            cluster_result = CellClusterResult(
                cell_key=cell_key,
                cv_label=cv_label,
                glyphs=glyphs,
                n_glyphs=len(glyphs),
                pairwise_cosine=pairwise,
                allograph_pairs=allograph_pairs,
                distinct_pairs=distinct_pairs,
                n_distinct_phonemes=n_clusters,
                cluster_assignments=cluster_assignments,
            )

        total_distinct += cluster_result.n_distinct_phonemes
        cell_results.append(_convert(cluster_result))

        glyph_str = ', '.join(glyphs[:6]) + ('...' if len(glyphs) > 6 else '')
        print(
            f"  {cv_label:<30} {glyph_str:<30} "
            f"{cluster_result.n_distinct_phonemes} "
            f"{'(all allographs)' if cluster_result.n_distinct_phonemes == 1 and len(glyphs) > 1 else ''}"
        )

    gate_passed = 20 <= total_distinct <= 30

    if total_distinct < 20:
        verdict = (
            f"Under-splitting: {total_distinct} distinct phonemes. "
            "Distributional analysis can't distinguish all phonemes. "
            "Feature decomposition (Step 14.2) may still help."
        )
    elif total_distinct > 30:
        verdict = (
            f"Over-splitting: {total_distinct} distinct phonemes. "
            "Some distinctions are contextual (allographs) rather than phonemic. "
            "Apply context rules from Phase 13 to merge position-dependent variants."
        )
    else:
        verdict = (
            f"PASS: {total_distinct} distinct phonemes is consistent with "
            "Romance language phoneme inventories (20–30). "
            "Cell conflation confirmed as the 11.1%% ceiling cause."
        )

    print(f"\n  ── Summary ──")
    print(f"  Phase 11 inventory:         {len(cv_labels)} cells (one phoneme per cell)")
    print(f"  Phase 14 triples inventory: {n_triples} unique stroke triples")
    print(f"  Phase 14 cluster inventory: {total_distinct} distinct phonemes from clustering")
    print(f"  Gate (20–30): {'PASS' if gate_passed else 'FAIL'}")
    print(f"  Verdict: {verdict}")

    result = CellAnalysisResult(
        n_cells=len(cv_labels),
        n_glyphs_total=total_glyphs,
        cell_results=cell_results,
        inventory_size_phase11=len(cv_labels),
        inventory_size_phase14_triples=n_triples,
        inventory_size_phase14_clusters=total_distinct,
        gate_passed=gate_passed,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'cell_analysis.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)
    print(f"\n  Results saved → {out_path}")
