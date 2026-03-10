"""
Phase 33.14 – Distributional Match (Hungarian Algorithm)
=========================================================
Find the optimal mapping from top-N Voynich tokens to top-N Latin words
that maximises pair frequency correlation.  Uses the Hungarian algorithm
for optimal 1-to-1 assignment, with a greedy fallback when scipy is
unavailable.

Dependency chain:
    token_pair_freq.json  (Step 33.13)
        -> distributional_match.json  (this step)
"""

import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from voynich.core._paths import results_dir as _results_dir

try:
    from scipy.optimize import linear_sum_assignment
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


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


def _cosine_similarity(vec_a: Dict[str, float], vec_b: Dict[str, float]) -> float:
    """Cosine similarity between two dicts {key: count}."""
    keys = set(vec_a.keys()) | set(vec_b.keys())
    dot = sum(vec_a.get(k, 0) * vec_b.get(k, 0) for k in keys)
    norm_a = sum(v ** 2 for v in vec_a.values()) ** 0.5
    norm_b = sum(v ** 2 for v in vec_b.values()) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _spearman_rank_correlation(x_ranks: List[float],
                                y_ranks: List[float]) -> float:
    """Compute Spearman rho from two lists of ranks."""
    n = len(x_ranks)
    if n < 2:
        return 0.0
    d_sq = sum((xi - yi) ** 2 for xi, yi in zip(x_ranks, y_ranks))
    rho = 1 - (6 * d_sq) / (n * (n ** 2 - 1))
    return rho


def _greedy_assignment(cost_matrix: List[List[float]]) -> List[Tuple[int, int]]:
    """
    Greedy 1-to-1 assignment: iteratively pick the (i, j) pair with
    lowest cost that has not yet been assigned.  Fallback when scipy is
    unavailable.
    """
    n = len(cost_matrix)
    # Flatten into (cost, i, j) triples and sort ascending
    candidates = []
    for i in range(n):
        for j in range(n):
            candidates.append((cost_matrix[i][j], i, j))
    candidates.sort()

    used_rows: Set[int] = set()
    used_cols: Set[int] = set()
    assignments: List[Tuple[int, int]] = []

    for cost, i, j in candidates:
        if i in used_rows or j in used_cols:
            continue
        assignments.append((i, j))
        used_rows.add(i)
        used_cols.add(j)
        if len(assignments) == n:
            break

    return assignments


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TokenWordMapping:
    eva_token: str
    eva_rank: int
    latin_word: str
    latin_rank: int
    compatibility: float
    rank_proximity: float
    cooccurrence_similarity: float


@dataclass
class DistributionalMatchResult:
    # Configuration
    n_tokens: int  # N used for matching
    # Hungarian assignment
    optimal_mappings: List[Dict]  # TokenWordMapping as dicts
    optimal_cost: float
    # Pair correlation
    n_mapped_pairs: int
    n_pair_matches: int  # mapped EVA pairs that appear in Latin pairs
    pair_match_rate: float
    spearman_rho: float
    # Null comparison
    null_mean_rho: float
    null_p95_rho: float
    p_value: float
    significant: bool  # p < 0.05
    # Sensitivity
    n20_top10: List[str]  # top-10 EVA tokens at N=20
    n30_top10: List[str]  # top-10 EVA tokens at N=30
    stability: float  # fraction of top-10 that agree between N=20 and N=30
    # Verdict
    verdict: str  # 'SIGNIFICANT_MATCH', 'MARGINAL', 'NO_MATCH'
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Core matching logic
# ---------------------------------------------------------------------------

def _build_cooccurrence_vectors(
    top_tokens: List[str],
    pair_list: List[Dict],
) -> Dict[str, Dict[str, int]]:
    """
    For each token in *top_tokens*, build a co-occurrence vector from the
    pair list.  The vector maps each other top-token to the count of their
    joint pair occurrence.
    """
    top_set = set(top_tokens)
    vectors: Dict[str, Dict[str, int]] = {t: {} for t in top_tokens}

    for entry in pair_list:
        w1, w2 = entry['pair']
        count = entry['count']
        if w1 in top_set and w2 in top_set:
            vectors[w1][w2] = vectors[w1].get(w2, 0) + count
            vectors[w2][w1] = vectors[w2].get(w1, 0) + count

    return vectors


def _build_compatibility_matrix(
    eva_tokens: List[Dict],
    latin_words: List[Dict],
    eva_pairs: List[Dict],
    latin_pairs: List[Dict],
    n: int,
) -> Tuple[List[List[float]], List[Dict], List[Dict]]:
    """
    Build an N x N compatibility matrix.

    Returns:
        matrix:      N x N list-of-lists, entry (i,j) = compatibility score
        eva_subset:  the N EVA token dicts used (ordered by rank)
        latin_subset: the N Latin word dicts used (ordered by rank)
    """
    eva_subset = eva_tokens[:n]
    latin_subset = latin_words[:n]

    eva_names = [e['token'] for e in eva_subset]
    latin_names = [w['token'] for w in latin_subset]

    # Build co-occurrence vectors
    eva_cooc = _build_cooccurrence_vectors(eva_names, eva_pairs)
    latin_cooc = _build_cooccurrence_vectors(latin_names, latin_pairs)

    matrix: List[List[float]] = []
    for i, e_entry in enumerate(eva_subset):
        row: List[float] = []
        e_rank = e_entry['rank']
        e_vec = eva_cooc.get(e_entry['token'], {})
        for j, l_entry in enumerate(latin_subset):
            l_rank = l_entry['rank']
            l_vec = latin_cooc.get(l_entry['token'], {})

            # Rank proximity
            score_rank = 1.0 / (1 + abs(e_rank - l_rank))

            # Co-occurrence profile similarity
            # Re-index vectors by positional index so cosine is meaningful
            e_vec_indexed: Dict[str, float] = {}
            for k, other_e in enumerate(eva_names):
                if other_e in e_vec:
                    e_vec_indexed[str(k)] = float(e_vec[other_e])

            l_vec_indexed: Dict[str, float] = {}
            for k, other_l in enumerate(latin_names):
                if other_l in l_vec:
                    l_vec_indexed[str(k)] = float(l_vec[other_l])

            cosine = _cosine_similarity(e_vec_indexed, l_vec_indexed)

            # Combined
            compatibility = 0.5 * score_rank + 0.5 * cosine
            row.append(compatibility)
        matrix.append(row)

    return matrix, eva_subset, latin_subset


def _solve_assignment(
    compat_matrix: List[List[float]],
) -> Tuple[List[Tuple[int, int]], float]:
    """
    Solve the assignment problem (maximise compatibility = minimise cost).

    Returns:
        assignments: list of (row, col) pairs
        total_cost:  sum of (1 - compatibility) over assigned pairs
    """
    n = len(compat_matrix)
    # Cost = 1 - compatibility
    cost_matrix = [[1.0 - compat_matrix[i][j] for j in range(n)]
                   for i in range(n)]

    if HAS_SCIPY:
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        assignments = list(zip(row_ind.tolist(), col_ind.tolist()))
    else:
        assignments = _greedy_assignment(cost_matrix)

    total_cost = sum(cost_matrix[i][j] for i, j in assignments)
    return assignments, total_cost


def _run_matching_at_n(
    n: int,
    top_eva_tokens: List[Dict],
    top_latin_words: List[Dict],
    eva_top_pairs: List[Dict],
    latin_top_pairs: List[Dict],
) -> Tuple[List[TokenWordMapping], float, List[List[float]]]:
    """
    Run the full matching pipeline at a given N.

    Returns:
        mappings:       list of TokenWordMapping (sorted by EVA rank)
        total_cost:     sum of assignment costs
        compat_matrix:  the N x N compatibility matrix
    """
    compat_matrix, eva_sub, latin_sub = _build_compatibility_matrix(
        top_eva_tokens, top_latin_words,
        eva_top_pairs, latin_top_pairs, n,
    )

    assignments, total_cost = _solve_assignment(compat_matrix)

    # Build co-occurrence vectors for detail fields
    eva_names = [e['token'] for e in eva_sub]
    latin_names = [w['token'] for w in latin_sub]
    eva_cooc = _build_cooccurrence_vectors(eva_names, eva_top_pairs)
    latin_cooc = _build_cooccurrence_vectors(latin_names, latin_top_pairs)

    mappings: List[TokenWordMapping] = []
    for row_i, col_j in assignments:
        e = eva_sub[row_i]
        l = latin_sub[col_j]

        e_rank = e['rank']
        l_rank = l['rank']
        score_rank = 1.0 / (1 + abs(e_rank - l_rank))

        e_vec_idx: Dict[str, float] = {}
        e_vec = eva_cooc.get(e['token'], {})
        for k, other in enumerate(eva_names):
            if other in e_vec:
                e_vec_idx[str(k)] = float(e_vec[other])
        l_vec_idx: Dict[str, float] = {}
        l_vec = latin_cooc.get(l['token'], {})
        for k, other in enumerate(latin_names):
            if other in l_vec:
                l_vec_idx[str(k)] = float(l_vec[other])

        cosine = _cosine_similarity(e_vec_idx, l_vec_idx)

        mappings.append(TokenWordMapping(
            eva_token=e['token'],
            eva_rank=e_rank,
            latin_word=l['token'],
            latin_rank=l_rank,
            compatibility=round(compat_matrix[row_i][col_j], 4),
            rank_proximity=round(score_rank, 4),
            cooccurrence_similarity=round(cosine, 4),
        ))

    # Sort by EVA rank
    mappings.sort(key=lambda m: m.eva_rank)
    return mappings, total_cost, compat_matrix


# ---------------------------------------------------------------------------
# Pair correlation under a mapping
# ---------------------------------------------------------------------------

def _pair_correlation_under_mapping(
    mapping: Dict[str, str],
    eva_top_pairs: List[Dict],
    latin_top_pairs: List[Dict],
    top_n_pairs: int = 100,
) -> Tuple[int, int, float, float]:
    """
    Under a given EVA->Latin mapping, check how many EVA pairs map to
    existing Latin pairs and compute rank correlation.

    Returns:
        n_mapped_pairs:  number of EVA pairs where both tokens are in mapping
        n_pair_matches:  how many of those map to a Latin top-pair
        pair_match_rate: n_pair_matches / n_mapped_pairs
        spearman_rho:    rank correlation over matched pairs
    """
    # Build Latin pair lookup: (w1, w2) -> rank
    latin_pair_rank: Dict[Tuple[str, str], int] = {}
    for entry in latin_top_pairs[:top_n_pairs]:
        w1, w2 = entry['pair']
        latin_pair_rank[(w1, w2)] = entry['rank']

    eva_mapped_ranks: List[float] = []
    latin_matched_ranks: List[float] = []
    n_mapped_pairs = 0
    n_pair_matches = 0

    for entry in eva_top_pairs[:top_n_pairs]:
        w1, w2 = entry['pair']
        mapped_w1 = mapping.get(w1)
        mapped_w2 = mapping.get(w2)
        if mapped_w1 is None or mapped_w2 is None:
            continue
        n_mapped_pairs += 1

        latin_rank = latin_pair_rank.get((mapped_w1, mapped_w2))
        if latin_rank is not None:
            n_pair_matches += 1
            eva_mapped_ranks.append(float(entry['rank']))
            latin_matched_ranks.append(float(latin_rank))

    pair_match_rate = n_pair_matches / n_mapped_pairs if n_mapped_pairs > 0 else 0.0

    if len(eva_mapped_ranks) >= 2:
        rho = _spearman_rank_correlation(eva_mapped_ranks, latin_matched_ranks)
    else:
        rho = 0.0

    return n_mapped_pairs, n_pair_matches, pair_match_rate, rho


# ---------------------------------------------------------------------------
# Null comparison
# ---------------------------------------------------------------------------

def _null_distribution(
    eva_tokens: List[str],
    latin_words: List[str],
    eva_top_pairs: List[Dict],
    latin_top_pairs: List[Dict],
    n_perms: int = 1000,
    top_n_pairs: int = 100,
    seed: int = 42,
) -> List[float]:
    """
    Generate *n_perms* random 1-to-1 assignments and compute pair
    frequency correlation for each.
    """
    rng = random.Random(seed)
    null_rhos: List[float] = []

    for _ in range(n_perms):
        shuffled = list(latin_words)
        rng.shuffle(shuffled)
        rand_mapping = {e: l for e, l in zip(eva_tokens, shuffled)}
        _, _, _, rho = _pair_correlation_under_mapping(
            rand_mapping, eva_top_pairs, latin_top_pairs, top_n_pairs,
        )
        null_rhos.append(rho)

    return null_rhos


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_distributional_match() -> None:
    """Step 33.14: Distributional match via Hungarian algorithm."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 33.14: Distributional Match (Hungarian Algorithm)")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load token_pair_freq.json ──
    print("\n  1. Loading token_pair_freq.json ...")
    tpf_path = os.path.join(rd, 'token_pair_freq.json')
    if not os.path.exists(tpf_path):
        print("  [SKIP] token_pair_freq.json not found")
        return

    with open(tpf_path) as f:
        tpf = json.load(f)

    top_eva_tokens = tpf['top_eva_tokens']
    top_latin_words = tpf['top_latin_words']
    eva_top_pairs = tpf['eva_pair_stats']['top_pairs']
    latin_top_pairs = tpf['latin_pair_stats']['top_pairs']

    print(f"     EVA tokens available: {len(top_eva_tokens)}")
    print(f"     Latin words available: {len(top_latin_words)}")
    print(f"     EVA top pairs: {len(eva_top_pairs)}")
    print(f"     Latin top pairs: {len(latin_top_pairs)}")
    print(f"     scipy available: {HAS_SCIPY}")

    # ── 2. Run matching at N=20 ──
    n20 = min(20, len(top_eva_tokens), len(top_latin_words))
    print(f"\n  2. Running Hungarian matching at N={n20} ...")

    mappings_20, cost_20, _ = _run_matching_at_n(
        n20, top_eva_tokens, top_latin_words,
        eva_top_pairs, latin_top_pairs,
    )

    print(f"     Optimal cost: {cost_20:.4f}")
    print("     Top-10 mappings:")
    for m in mappings_20[:10]:
        print(f"       {m.eva_token:12s} (rank {m.eva_rank:2d}) "
              f"-> {m.latin_word:12s} (rank {m.latin_rank:2d})  "
              f"compat={m.compatibility:.3f}  "
              f"rank_prox={m.rank_proximity:.3f}  "
              f"cooc_sim={m.cooccurrence_similarity:.3f}")

    # ── 3. Pair correlation under optimal mapping (N=20) ──
    print(f"\n  3. Computing pair correlation under optimal mapping ...")

    mapping_dict_20 = {m.eva_token: m.latin_word for m in mappings_20}
    n_mapped, n_matches, match_rate, rho_opt = _pair_correlation_under_mapping(
        mapping_dict_20, eva_top_pairs, latin_top_pairs,
    )

    print(f"     Mapped pairs (both tokens in mapping): {n_mapped}")
    print(f"     Pair matches (mapped pair exists in Latin): {n_matches}")
    print(f"     Match rate: {match_rate:.4f}")
    print(f"     Spearman rho: {rho_opt:.4f}")

    # ── 4. Null comparison ──
    print(f"\n  4. Null comparison (1000 random permutations) ...")

    eva_names_20 = [e['token'] for e in top_eva_tokens[:n20]]
    latin_names_20 = [w['token'] for w in top_latin_words[:n20]]

    null_rhos = _null_distribution(
        eva_names_20, latin_names_20,
        eva_top_pairs, latin_top_pairs,
        n_perms=1000,
    )

    null_mean = sum(null_rhos) / len(null_rhos) if null_rhos else 0.0
    sorted_nulls = sorted(null_rhos)
    null_p95 = sorted_nulls[int(0.95 * len(sorted_nulls))] if sorted_nulls else 0.0
    p_value = sum(1 for r in null_rhos if r >= rho_opt) / len(null_rhos) if null_rhos else 1.0
    significant = p_value < 0.05

    print(f"     Null mean rho: {null_mean:.4f}")
    print(f"     Null 95th percentile: {null_p95:.4f}")
    print(f"     Optimal rho: {rho_opt:.4f}")
    print(f"     p-value: {p_value:.4f}")
    print(f"     Significant (p < 0.05): {significant}")

    # ── 5. Sensitivity: run at N=30 ──
    n30 = min(30, len(top_eva_tokens), len(top_latin_words))
    print(f"\n  5. Sensitivity analysis at N={n30} ...")

    mappings_30, cost_30, _ = _run_matching_at_n(
        n30, top_eva_tokens, top_latin_words,
        eva_top_pairs, latin_top_pairs,
    )

    print(f"     Optimal cost (N={n30}): {cost_30:.4f}")

    # Compare top-10 EVA token assignments between N=20 and N=30
    n20_top10 = [m.eva_token for m in mappings_20[:10]]
    n30_top10 = [m.eva_token for m in mappings_30[:10]]

    # Build assignment dicts for stability comparison
    assign_20 = {m.eva_token: m.latin_word for m in mappings_20}
    assign_30 = {m.eva_token: m.latin_word for m in mappings_30}

    n_agree = 0
    for token in n20_top10:
        if token in assign_30 and assign_20.get(token) == assign_30.get(token):
            n_agree += 1
    stability = n_agree / len(n20_top10) if n20_top10 else 0.0

    print(f"     N=20 top-10: {n20_top10}")
    print(f"     N=30 top-10: {n30_top10}")
    print(f"     Stability (same assignment): {n_agree}/{len(n20_top10)} = {stability:.2f}")

    # ── 6. Verdict ──
    print("\n  6. Verdict ...")

    if significant and stability >= 0.5:
        verdict = 'SIGNIFICANT_MATCH'
    elif significant or stability >= 0.5:
        verdict = 'MARGINAL'
    else:
        verdict = 'NO_MATCH'

    print(f"     Verdict: {verdict}")

    # ── 7. Save results ──
    result = DistributionalMatchResult(
        n_tokens=n20,
        optimal_mappings=[_convert(asdict(m)) for m in mappings_20],
        optimal_cost=round(cost_20, 4),
        n_mapped_pairs=n_mapped,
        n_pair_matches=n_matches,
        pair_match_rate=round(match_rate, 4),
        spearman_rho=round(rho_opt, 4),
        null_mean_rho=round(null_mean, 4),
        null_p95_rho=round(null_p95, 4),
        p_value=round(p_value, 4),
        significant=significant,
        n20_top10=n20_top10,
        n30_top10=n30_top10,
        stability=round(stability, 2),
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'distributional_match.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)
    print(f"\n  -> {out_path}")
