"""Tier 3: Ensemble combination and validation.

Rank-fusion across all 7 methods, T_P15 validation, and per-method diagnostics.
"""

from collections import Counter

import numpy as np
from scipy.stats import norm


def build_ensemble(distance_matrices, method_names, weights=None):
    """Combine multiple distance matrices into one ranking via rank fusion.

    For each method, convert distances to ranks (1 = most similar).
    Compute weighted mean rank across methods.

    Args:
        distance_matrices: list of (n_eva x n_costa) distance matrices
        method_names: list of method name strings
        weights: optional list of floats (default: equal weight)

    Returns: (n_eva x n_costa) mean-rank matrix (lower = better match)
    """
    n_eva, n_costa = distance_matrices[0].shape

    rank_matrices = []
    for mat in distance_matrices:
        ranks = np.zeros_like(mat)
        for i in range(n_eva):
            ranks[i] = np.argsort(np.argsort(mat[i])) + 1
        rank_matrices.append(ranks)

    if weights is None:
        weights = [1.0] * len(rank_matrices)

    total_weight = sum(weights)
    ensemble_ranks = np.zeros((n_eva, n_costa))
    for mat, w in zip(rank_matrices, weights):
        ensemble_ranks += w * mat
    ensemble_ranks /= total_weight

    return ensemble_ranks


def validate_ensemble(ensemble_ranks, eva_names, costa_names, t_p15_table):
    """Validate T_P15 assignments against ensemble rankings.

    For each EVA char with a T_P15 assignment, find where the proposed
    Costamagna syllable ranks in the ensemble ordering.

    Returns dict with per-sign validation and summary statistics.
    """
    costa_name_list = list(costa_names)
    t_p15_ranks = {}

    for eva_name, proposed in t_p15_table.items():
        if eva_name not in eva_names:
            continue
        i = list(eva_names).index(eva_name)

        sorted_indices = np.argsort(ensemble_ranks[i])
        sorted_names = [costa_name_list[j] for j in sorted_indices]

        # Find rank of proposed syllable (handle compound names)
        rank = len(costa_names)
        for r, syl in enumerate(sorted_names):
            if syl == proposed or proposed in syl.split('-'):
                rank = r + 1
                break

        t_p15_ranks[eva_name] = {
            'proposed': proposed,
            'rank': rank,
            'top_5': sorted_names[:5],
            'support': ('STRONG' if rank <= 5 else
                        'MODERATE' if rank <= 15 else
                        'WEAK' if rank <= 50 else 'NONE'),
        }

    levels = Counter(r['support'] for r in t_p15_ranks.values())
    ranks = [r['rank'] for r in t_p15_ranks.values()]

    return {
        'per_sign': t_p15_ranks,
        'support_levels': dict(levels),
        'mean_rank': float(np.mean(ranks)) if ranks else 0,
        'median_rank': float(np.median(ranks)) if ranks else 0,
        'n_tested': len(t_p15_ranks),
        'n_strong': levels.get('STRONG', 0),
        'n_moderate': levels.get('MODERATE', 0),
        'n_weak': levels.get('WEAK', 0),
        'n_none': levels.get('NONE', 0),
    }


def permutation_test_ensemble(ensemble_ranks, eva_names, costa_names,
                              t_p15_table, n_perms=1000, seed=42):
    """Test whether T_P15's mean ensemble rank is better than random.

    Returns dict with z-score and p-value.
    """
    costa_name_list = list(costa_names)
    eva_name_list = list(eva_names)
    rng = np.random.default_rng(seed)

    # Real mean rank
    real_ranks = []
    tested_indices = []
    for eva_name, proposed in t_p15_table.items():
        if eva_name not in eva_name_list:
            continue
        i = eva_name_list.index(eva_name)
        sorted_names = [costa_name_list[j]
                        for j in np.argsort(ensemble_ranks[i])]
        rank = len(costa_names)
        for r, syl in enumerate(sorted_names):
            if syl == proposed or proposed in syl.split('-'):
                rank = r + 1
                break
        real_ranks.append(rank)
        tested_indices.append(i)

    if not real_ranks:
        return {'z': 0.0, 'p': 1.0, 'significant': False}

    real_mean = float(np.mean(real_ranks))

    # Random assignments
    all_syllables = list(set(costa_name_list))
    random_means = []
    for _ in range(n_perms):
        rand_ranks = []
        rand_syls = rng.choice(all_syllables, size=len(tested_indices))
        for idx, rand_syl in zip(tested_indices, rand_syls):
            sorted_names = [costa_name_list[j]
                            for j in np.argsort(ensemble_ranks[idx])]
            rank = len(costa_names)
            for r, syl in enumerate(sorted_names):
                if syl == rand_syl or rand_syl in syl.split('-'):
                    rank = r + 1
                    break
            rand_ranks.append(rank)
        random_means.append(float(np.mean(rand_ranks)))

    null_mean = float(np.mean(random_means))
    null_std = float(np.std(random_means))
    z = (real_mean - null_mean) / null_std if null_std > 0 else 0.0
    p = float(np.mean([r <= real_mean for r in random_means]))

    return {
        'real_mean_rank': real_mean,
        'null_mean_rank': null_mean,
        'null_std': null_std,
        'z': z,
        'p': p,
        'significant': p < 0.05,
        'n_tested': len(real_ranks),
    }


def per_method_diagnostics(distance_matrices, method_names, eva_names,
                           costa_names, t_p15_table):
    """For each method, compute T_P15 validation statistics.

    Returns dict mapping method name -> diagnostic dict.
    """
    costa_name_list = list(costa_names)
    eva_name_list = list(eva_names)

    diagnostics = {}
    for mat, name in zip(distance_matrices, method_names):
        ranks_dict = {}
        for eva_name, proposed in t_p15_table.items():
            if eva_name not in eva_name_list:
                continue
            i = eva_name_list.index(eva_name)
            sorted_names = [costa_name_list[j]
                            for j in np.argsort(mat[i])]
            rank = len(costa_names)
            for r, syl in enumerate(sorted_names):
                if syl == proposed or proposed in syl.split('-'):
                    rank = r + 1
                    break
            ranks_dict[eva_name] = rank

        ranks = list(ranks_dict.values())
        spread = float(mat.max() - mat.min())

        diagnostics[name] = {
            'mean_t_p15_rank': float(np.mean(ranks)) if ranks else 0,
            'n_strong': sum(1 for r in ranks if r <= 5),
            'n_moderate': sum(1 for r in ranks if r <= 15),
            'similarity_spread': spread,
            'top_1_accuracy': (sum(1 for r in ranks if r == 1) / len(ranks)
                               if ranks else 0),
            'n_tested': len(ranks),
        }

    return diagnostics
