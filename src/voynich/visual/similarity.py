"""Cosine similarity matrix computation and match ranking."""

import numpy as np


def compute_similarity_matrix(emb_a, emb_b):
    """Compute cosine similarity between two sets of L2-normalized embeddings.

    Returns: (n_a x n_b) matrix where entry [i,j] is similarity.
    """
    return emb_a @ emb_b.T


def rank_matches(sim_matrix, names_a, names_b, metadata_b=None, top_k=20):
    """For each item in set A, rank all items in set B by similarity.

    Args:
        sim_matrix: (n_a x n_b) cosine similarity matrix
        names_a: Labels for rows (EVA characters)
        names_b: Labels for columns (Costamagna syllables)
        metadata_b: Optional list of dicts with extra info per B item
        top_k: Number of top matches to return

    Returns:
        Dict mapping each A name to its ranked matches.
    """
    rankings = {}

    for i, name_a in enumerate(names_a):
        sorted_indices = np.argsort(sim_matrix[i])[::-1]

        top_matches = []
        for rank, j in enumerate(sorted_indices[:top_k]):
            match = {
                'rank': rank + 1,
                'syllable': names_b[j],
                'similarity': float(sim_matrix[i, j]),
            }
            if metadata_b and j < len(metadata_b):
                meta = metadata_b[j]
                match['tavola'] = meta.get('tavola', '?')
                match['onset'] = meta.get('onset', None)
                match['structure'] = meta.get('structure', '?')
            top_matches.append(match)

        rankings[name_a] = {
            'top_matches': top_matches,
            'best_match': top_matches[0] if top_matches else None,
        }

    return rankings


def find_assignment_ranks(rankings, assignment_table, n_costamagna):
    """For each assigned EVA char, find where the assigned syllable ranks.

    Args:
        rankings: Output of rank_matches()
        assignment_table: Dict mapping EVA char name -> proposed syllable
        n_costamagna: Total number of Costamagna signs (for worst-case rank)

    Returns:
        Dict with per-character validation results.
    """
    validation = {}

    for eva_name, proposed in assignment_table.items():
        if eva_name not in rankings:
            continue

        # Find rank of proposed syllable (handle multi-value like 'ad-at')
        rank = None
        for match in rankings[eva_name]['top_matches']:
            syl = match['syllable']
            # Check exact match or if proposed is part of a compound
            if syl == proposed or proposed in syl.split('-'):
                rank = match['rank']
                break

        if rank is None:
            rank = n_costamagna  # worst possible rank

        best = rankings[eva_name]['best_match']
        validation[eva_name] = {
            'proposed_syllable': proposed,
            'visual_rank': rank,
            'best_visual_match': best['syllable'] if best else '?',
            'best_similarity': best['similarity'] if best else 0.0,
            'support_level': (
                'STRONG' if rank <= 5 else
                'MODERATE' if rank <= 15 else
                'WEAK' if rank <= 50 else
                'NONE'
            ),
        }

    return validation


def family_cohesion(embeddings, names, assignment_table, n_perms=1000,
                     seed=42):
    """Test whether EVA chars sharing an onset consonant cluster together.

    Groups EVA characters by the onset of their T_P15 syllable assignment,
    computes mean within-group pairwise similarity, and compares against
    random groups.

    Returns dict with z-score and p-value.
    """
    # Map EVA chars to onset consonants
    name_to_idx = {n: i for i, n in enumerate(names)}
    onset_groups = {}
    for eva, syllable in assignment_table.items():
        if eva not in name_to_idx or not syllable:
            continue
        onset = syllable[0] if syllable[0] not in 'aeiou' else '_vowel'
        if onset not in onset_groups:
            onset_groups[onset] = []
        onset_groups[onset].append(name_to_idx[eva])

    # Compute real cohesion per group
    group_sims = {}
    for onset, indices in onset_groups.items():
        if len(indices) < 2:
            continue
        group_emb = embeddings[indices]
        pairwise = group_emb @ group_emb.T
        n = len(indices)
        mean_sim = (pairwise.sum() - n) / (n * (n - 1))
        group_sims[onset] = {
            'n_members': n,
            'mean_similarity': float(mean_sim),
            'members': [names[i] for i in indices],
        }

    if not group_sims:
        return {'z': 0.0, 'p': 1.0, 'per_onset': {}, 'families_cluster': False}

    real_mean = float(np.mean([d['mean_similarity'] for d in group_sims.values()]))

    # Random baseline
    rng = np.random.default_rng(seed)
    random_means = []
    for _ in range(n_perms):
        trial_total = 0.0
        trial_count = 0
        for data in group_sims.values():
            n = data['n_members']
            rand_idx = rng.choice(len(embeddings), size=n, replace=False)
            group_emb = embeddings[rand_idx]
            pairwise = group_emb @ group_emb.T
            mean_sim = (pairwise.sum() - n) / (n * (n - 1))
            trial_total += mean_sim
            trial_count += 1
        random_means.append(trial_total / trial_count)

    null_mean = float(np.mean(random_means))
    null_std = float(np.std(random_means))
    z = (real_mean - null_mean) / null_std if null_std > 0 else 0.0

    from scipy.stats import norm
    p = float(1 - norm.cdf(z))

    return {
        'per_onset': group_sims,
        'real_mean_cohesion': real_mean,
        'null_mean_cohesion': null_mean,
        'null_std': null_std,
        'z': z,
        'p': p,
        'families_cluster': z > 1.65,
    }


def permutation_test_ranks(sim_matrix, names_a, names_b, assignment_table,
                            n_perms=1000, seed=42):
    """Test whether T_P15's mean visual rank is better than random assignments.

    For each of n_perms random assignment tables, computes the mean rank of
    assigned syllables in the visual similarity ordering.
    """
    all_syllables = list(set(names_b))
    name_to_idx_a = {n: i for i, n in enumerate(names_a)}

    # Real mean rank
    real_ranks = []
    for eva, proposed in assignment_table.items():
        if eva not in name_to_idx_a:
            continue
        i = name_to_idx_a[eva]
        sorted_names = [names_b[j] for j in np.argsort(sim_matrix[i])[::-1]]
        # Find rank (handle compound syllables)
        found = False
        for r, syl in enumerate(sorted_names):
            if syl == proposed or proposed in syl.split('-'):
                real_ranks.append(r + 1)
                found = True
                break
        if not found:
            real_ranks.append(len(names_b))

    real_mean = float(np.mean(real_ranks))

    # Random assignments
    rng = np.random.default_rng(seed)
    random_means = []
    n_chars = len([e for e in assignment_table if e in name_to_idx_a])

    for _ in range(n_perms):
        rand_ranks = []
        random_syls = rng.choice(all_syllables, size=n_chars)
        for (eva, _), rand_syl in zip(
            [(e, s) for e, s in assignment_table.items() if e in name_to_idx_a],
            random_syls
        ):
            i = name_to_idx_a[eva]
            sorted_names = [names_b[j] for j in np.argsort(sim_matrix[i])[::-1]]
            found = False
            for r, syl in enumerate(sorted_names):
                if syl == rand_syl or rand_syl in syl.split('-'):
                    rand_ranks.append(r + 1)
                    found = True
                    break
            if not found:
                rand_ranks.append(len(names_b))
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
