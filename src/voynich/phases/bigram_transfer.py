"""
Phase 8 / Approach 16: Bigram Transfer Cryptanalysis
=====================================================
Treat the Voynich stem vocabulary as a simple substitution cipher over
Latin stems.  Each Voynich stem is a ciphertext symbol; each Latin stem
is a plaintext symbol.  The "key" is the permutation mapping Voynich
stems to Latin stems.

Classical cryptanalysis recovers this mapping using bigram frequency
statistics: in natural language, bigram frequencies are highly non-uniform,
and the sparsity pattern is language-specific and encoding-preserving.

Sub-analyses:
  16.1 — Build bigram frequency matrices (Voynich, Latin, Occitan, nulls)
  16.2 — SA permutation search (multiple metrics, restarts, seeded init)
  16.3 — Assess best mapping (stability, decoded sample)
  16.4 — Validation battery (null tests, cross-validation)

Output:
  results/bigram_transfer.json
"""

import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from voynich.core.corpus import load_corpus, VoynichCorpus
from voynich.core.stats import (
    selectivity_ratio,
)
from voynich.core._paths import results_dir as _results_dir
from voynich.core.reference import (
    load_reference_corpus, ReferenceCorpus, stem_latin_token,
)
from voynich.phases.morpheme_grid import decompose_token_morphemes


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class BigramMatrixStats:
    """Descriptive statistics for a stem bigram matrix."""
    label: str
    n_vocab: int
    n_nonzero: int
    sparsity: float
    entropy: float
    top_20_bigrams: List[Tuple[str, str, float]]
    effective_rank: int


@dataclass
class SAPermutationResult:
    """Result from SA permutation search for one metric."""
    metric: str
    n_vocab: int
    best_distance: float
    init_distance: float
    random_distance_mean: float
    random_distance_std: float
    improvement_ratio: float
    selectivity: float
    best_permutation: Dict[str, str]
    convergence_history: List[float]
    n_restarts: int


@dataclass
class MappingStability:
    """Stability analysis across SA restarts."""
    mean_pairwise_agreement: float
    high_confidence_stems: List[str]
    low_confidence_stems: List[str]
    n_chains: int
    top_10_consistent: List[Tuple[str, str, float]]


@dataclass
class BigramTransferResult:
    """Full Phase 8 / Approach 16 output."""
    voynich_matrix_stats: Dict
    latin_matrix_stats: Dict
    occitan_matrix_stats: Dict
    sa_results: Dict[str, Dict]
    best_metric: str
    best_n_vocab: int
    best_selectivity: float
    stability: Dict
    decoded_sample: List[str]
    null_tests: Dict
    cross_validation: Dict
    gate_passed: bool
    verdict: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _convert(obj: Any) -> Any:
    """Convert dataclass/numpy types to JSON-serializable form."""
    if hasattr(obj, '__dataclass_fields__'):
        return {k: _convert(v) for k, v in asdict(obj).items()}
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, dict):
        return {str(k): _convert(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_convert(item) for item in obj]
    if isinstance(obj, (bool, int, float, str, type(None))):
        return obj
    return str(obj)


def _prepare_stem_sequence(
    tokens: List[str],
    min_count: int = 3,
) -> Tuple[List[str], List[str], Counter]:
    """
    Decompose Voynich tokens to stems and filter by frequency.

    Returns (stem_sequence, vocabulary, stem_counts).
    """
    stems = []
    for tok in tokens:
        d = decompose_token_morphemes(tok)
        stems.append(d.stem if d.stem else tok)

    counts = Counter(stems)
    vocab = [s for s, c in counts.most_common() if c >= min_count]
    vocab_set = set(vocab)
    filtered = [s for s in stems if s in vocab_set]
    return filtered, vocab, counts


def _prepare_ref_stem_sequence(
    ref_corpus: ReferenceCorpus,
    language: str,
    min_count: int = 3,
) -> Tuple[List[str], List[str], Counter]:
    """
    Prepare stem sequence from a reference language corpus.

    Returns (stem_sequence, vocabulary, stem_counts).
    """
    tokens = ref_corpus.get_combined_tokens(language)
    stems = [stem_latin_token(t) for t in tokens]

    counts = Counter(stems)
    vocab = [s for s, c in counts.most_common() if c >= min_count]
    vocab_set = set(vocab)
    filtered = [s for s in stems if s in vocab_set]
    return filtered, vocab, counts


# ---------------------------------------------------------------------------
# 16.1: Build Bigram Frequency Matrices
# ---------------------------------------------------------------------------

def _build_stem_bigram_matrix(
    stem_sequence: List[str],
    vocab: List[str],
    top_n: int = 200,
) -> Tuple[np.ndarray, List[str]]:
    """
    Build an NxN stem bigram transition probability matrix.

    Uses the top_n most frequent stems from vocab.  Stems outside top_n
    are mapped to a special <RARE> token (excluded from the matrix).

    Returns (probability_matrix, restricted_vocab).
    """
    restricted = vocab[:top_n]
    restricted_set = set(restricted)
    n = len(restricted)
    stem_to_idx = {s: i for i, s in enumerate(restricted)}

    counts = np.zeros((n, n), dtype=np.float64)
    for i in range(len(stem_sequence) - 1):
        s1, s2 = stem_sequence[i], stem_sequence[i + 1]
        if s1 in restricted_set and s2 in restricted_set:
            counts[stem_to_idx[s1]][stem_to_idx[s2]] += 1

    total = counts.sum()
    if total > 0:
        prob = counts / total
    else:
        prob = counts

    return prob, restricted


def _compute_matrix_stats(
    matrix: np.ndarray,
    vocab: List[str],
    label: str,
) -> BigramMatrixStats:
    """Compute descriptive statistics for a bigram matrix."""
    n = matrix.shape[0]
    n_nonzero = int(np.count_nonzero(matrix))
    sparsity = 1.0 - n_nonzero / (n * n) if n > 0 else 1.0

    # Entropy of the bigram distribution
    flat = matrix.flatten()
    flat = flat[flat > 0]
    entropy = -float(np.sum(flat * np.log2(flat))) if len(flat) > 0 else 0.0

    # Top-20 bigrams
    top_indices = np.argsort(matrix.flatten())[-20:][::-1]
    top_20 = []
    for idx in top_indices:
        i, j = divmod(int(idx), n)
        if matrix[i, j] > 0:
            top_20.append((vocab[i], vocab[j], float(matrix[i, j])))

    # Effective rank via SVD
    try:
        s = np.linalg.svd(matrix, compute_uv=False)
        threshold = 0.01 * s[0] if len(s) > 0 else 0
        effective_rank = int(np.sum(s > threshold))
    except np.linalg.LinAlgError:
        effective_rank = n

    return BigramMatrixStats(
        label=label,
        n_vocab=n,
        n_nonzero=n_nonzero,
        sparsity=round(sparsity, 4),
        entropy=round(entropy, 4),
        top_20_bigrams=top_20,
        effective_rank=effective_rank,
    )


# ---------------------------------------------------------------------------
# 16.2: Permutation Search (SA) — Specialized fast implementation
# ---------------------------------------------------------------------------

def _frobenius_cost(v_mat: np.ndarray, t_mat: np.ndarray, perm: np.ndarray) -> float:
    """Frobenius distance between permuted Voynich matrix and target."""
    permuted = v_mat[np.ix_(perm, perm)]
    return float(np.sqrt(np.sum((permuted - t_mat) ** 2)))


def _fast_sa_permutation(
    voynich_matrix: np.ndarray,
    target_matrix: np.ndarray,
    metric: str,
    max_iter: int = 100_000,
    t_start: float = 0.1,
    t_end: float = 0.0001,
    seed: int = 42,
) -> Tuple[np.ndarray, float, List[float]]:
    """
    Fast SA for permutation search using in-place swap evaluation.

    Instead of rebuilding the full permuted matrix each iteration,
    evaluates the cost change from swapping two rows/columns.
    """
    n = voynich_matrix.shape[0]
    rng = np.random.RandomState(seed)

    perm = np.arange(n, dtype=int)
    # Compute initial cost
    permuted = voynich_matrix[np.ix_(perm, perm)]
    diff = permuted - target_matrix
    current_cost = float(np.sqrt(np.sum(diff ** 2)))

    best_cost = current_cost
    best_perm = perm.copy()
    history: List[float] = []

    cooling = (t_end / t_start) ** (1.0 / max(max_iter, 1))
    temp = t_start

    for it in range(max_iter):
        # Pick two random positions to swap
        i, j = rng.randint(0, n, size=2)
        while i == j:
            j = rng.randint(0, n)

        # Swap in permutation
        perm[i], perm[j] = perm[j], perm[i]

        # Recompute cost (full recompute but with numpy vectorization)
        permuted = voynich_matrix[np.ix_(perm, perm)]
        new_diff = permuted - target_matrix
        new_cost = float(np.sqrt(np.sum(new_diff ** 2)))

        delta = new_cost - current_cost

        if delta < 0 or rng.random() < math.exp(-delta / temp):
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_perm = perm.copy()
        else:
            # Undo swap
            perm[i], perm[j] = perm[j], perm[i]

        temp *= cooling

        if it % 10_000 == 0:
            history.append(best_cost)

    return best_perm, best_cost, history


def run_sa_permutation_search(
    voynich_matrix: np.ndarray,
    voynich_vocab: List[str],
    target_matrix: np.ndarray,
    target_vocab: List[str],
    metrics: List[str] = ('frobenius',),
    n_restarts: int = 10,
    max_iter: int = 100_000,
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, SAPermutationResult]:
    """
    Run SA permutation search for each metric.

    Returns dict mapping metric_name -> SAPermutationResult.
    """
    n = voynich_matrix.shape[0]
    results: Dict[str, SAPermutationResult] = {}

    for metric in metrics:
        print(f"\n    SA permutation search: metric={metric}, N={n}")

        # Initial cost (frequency-rank = identity permutation)
        init_perm = np.arange(n, dtype=int)
        init_cost = _frobenius_cost(voynich_matrix, target_matrix, init_perm)

        # Random baseline
        rng_null = np.random.RandomState(seed)
        random_costs = []
        for _ in range(50):
            rp = rng_null.permutation(n)
            random_costs.append(_frobenius_cost(voynich_matrix, target_matrix, rp))
        random_mean = float(np.mean(random_costs))
        random_std = float(np.std(random_costs))

        # Calibrate temperature
        deltas = []
        rng_cal = np.random.RandomState(seed + 1000)
        test_perm = np.arange(n, dtype=int)
        test_cost = _frobenius_cost(voynich_matrix, target_matrix, test_perm)
        for _ in range(100):
            tp = test_perm.copy()
            i, j = rng_cal.randint(0, n, size=2)
            tp[i], tp[j] = tp[j], tp[i]
            nc = _frobenius_cost(voynich_matrix, target_matrix, tp)
            deltas.append(abs(nc - test_cost))
        median_delta = float(np.median(deltas)) if deltas else 0.01
        t_start = max(median_delta * 2.0, 0.01)

        # Run restarts
        global_best_perm = init_perm.copy()
        global_best_cost = init_cost
        all_history: List[float] = []

        for r in range(n_restarts):
            perm, cost, hist = _fast_sa_permutation(
                voynich_matrix, target_matrix, metric,
                max_iter=max_iter,
                t_start=t_start,
                t_end=t_start * 0.001,
                seed=seed + r * 7,
            )
            all_history.extend(hist)
            if cost < global_best_cost:
                global_best_cost = cost
                global_best_perm = perm.copy()
            if verbose:
                print(f"      Restart {r+1}/{n_restarts}: cost={cost:.6f}")

        # Build mapping dict
        mapping = {}
        for i, j in enumerate(global_best_perm):
            if i < len(voynich_vocab) and j < len(target_vocab):
                mapping[voynich_vocab[i]] = target_vocab[int(j)]

        selectivity = random_mean / global_best_cost if global_best_cost > 1e-10 else float('inf')
        improvement = init_cost / global_best_cost if global_best_cost > 1e-10 else float('inf')

        results[metric] = SAPermutationResult(
            metric=metric,
            n_vocab=n,
            best_distance=round(global_best_cost, 6),
            init_distance=round(init_cost, 6),
            random_distance_mean=round(random_mean, 6),
            random_distance_std=round(random_std, 6),
            improvement_ratio=round(improvement, 4),
            selectivity=round(selectivity, 4),
            best_permutation=mapping,
            convergence_history=[round(c, 6) for c in all_history[-20:]],
            n_restarts=n_restarts,
        )

        print(f"      Init distance: {init_cost:.6f}")
        print(f"      Best distance: {global_best_cost:.6f}")
        print(f"      Random mean:   {random_mean:.6f} +/- {random_std:.6f}")
        print(f"      Selectivity:   {selectivity:.4f}x")
        print(f"      Improvement:   {improvement:.4f}x over init")

    return results


# ---------------------------------------------------------------------------
# 16.3: Assess Best Mapping
# ---------------------------------------------------------------------------

def assess_mapping_stability(
    voynich_matrix: np.ndarray,
    target_matrix: np.ndarray,
    voynich_vocab: List[str],
    target_vocab: List[str],
    metric: str,
    n_chains: int = 20,
    max_iter: int = 50_000,
    seed: int = 42,
) -> MappingStability:
    """
    Assess mapping consistency across independent SA chains.

    Runs n_chains independent SA optimizations and measures pairwise
    agreement: fraction of stems mapped to the same target.
    """
    n = voynich_matrix.shape[0]

    # Collect best permutations from each chain
    chain_perms = []
    for c in range(n_chains):
        best_perm, _, _ = _fast_sa_permutation(
            voynich_matrix, target_matrix, metric,
            max_iter=max_iter,
            t_start=0.1,
            t_end=0.0001,
            seed=seed + c * 100,
        )
        chain_perms.append(best_perm)

    # Pairwise agreement
    agreements = []
    for i in range(len(chain_perms)):
        for j in range(i + 1, len(chain_perms)):
            agree = int(np.sum(chain_perms[i] == chain_perms[j]))
            agreements.append(agree / n)
    mean_agreement = float(np.mean(agreements)) if agreements else 0.0

    # Per-stem consistency: for each position, what fraction of chains agree
    # on the same mapping?
    stem_consistency = []
    for pos in range(n):
        mapped_targets = [int(p[pos]) for p in chain_perms]
        most_common_count = Counter(mapped_targets).most_common(1)[0][1]
        stem_consistency.append(most_common_count / n_chains)

    high_conf = [voynich_vocab[i] for i, c in enumerate(stem_consistency)
                 if c >= 0.8 and i < len(voynich_vocab)]
    low_conf = [voynich_vocab[i] for i, c in enumerate(stem_consistency)
                if c < 0.5 and i < len(voynich_vocab)]

    # Top-10 most consistent mappings
    best_chain = chain_perms[0]  # use first chain as representative
    top_10 = []
    indexed = [(i, stem_consistency[i]) for i in range(min(n, len(voynich_vocab)))]
    indexed.sort(key=lambda x: -x[1])
    for i, conf in indexed[:10]:
        j = int(best_chain[i])
        if j < len(target_vocab):
            top_10.append((voynich_vocab[i], target_vocab[j], round(conf, 3)))

    return MappingStability(
        mean_pairwise_agreement=round(mean_agreement, 4),
        high_confidence_stems=high_conf[:20],
        low_confidence_stems=low_conf[:20],
        n_chains=n_chains,
        top_10_consistent=top_10,
    )


def decode_sample(
    tokens: List[str],
    mapping: Dict[str, str],
    n_tokens: int = 50,
) -> List[str]:
    """Apply mapping to decode sample Voynich tokens."""
    decoded = []
    for tok in tokens[:n_tokens]:
        d = decompose_token_morphemes(tok)
        stem = d.stem if d.stem else tok
        mapped = mapping.get(stem, f'?{stem}?')
        decoded.append(f"{tok} -> {mapped}")
    return decoded


# ---------------------------------------------------------------------------
# 16.4: Validation Battery
# ---------------------------------------------------------------------------

def run_null_tests(
    voynich_matrix: np.ndarray,
    voynich_vocab: List[str],
    latin_matrix: np.ndarray,
    latin_vocab: List[str],
    occitan_matrix: np.ndarray,
    occitan_vocab: List[str],
    best_distance: float,
    best_metric: str,
    n_null_trials: int = 10,
    sa_max_iter: int = 50_000,
    seed: int = 42,
) -> Dict:
    """
    Null/validation tests for bigram transfer.

    Tests:
      a) Shuffled Voynich: permute stem labels, rebuild bigram matrix
      b) Random target: replace Latin with random matrix of same density
      c) Latin sanity: encipher Latin bigrams with known permutation, recover
      d) Occitan target: match Voynich to Occitan instead of Latin
    """
    n = voynich_matrix.shape[0]
    results = {}

    # (a) Shuffled Voynich stems
    print("\n    Null test (a): shuffled Voynich stems...")
    shuffled_distances = []
    rng_a = np.random.RandomState(seed)
    for trial in range(n_null_trials):
        perm = rng_a.permutation(n)
        shuffled_v = voynich_matrix[np.ix_(perm, perm)]
        _, best_cost, _ = _fast_sa_permutation(
            shuffled_v, latin_matrix, best_metric,
            max_iter=sa_max_iter, seed=seed + trial,
        )
        shuffled_distances.append(best_cost)

    shuffled_mean = float(np.mean(shuffled_distances))
    shuffled_std = float(np.std(shuffled_distances))
    shuffled_sel = shuffled_mean / best_distance if best_distance > 1e-10 else 0.0
    results['shuffled_voynich'] = {
        'mean_distance': round(shuffled_mean, 6),
        'std_distance': round(shuffled_std, 6),
        'selectivity': round(shuffled_sel, 4),
        'n_trials': n_null_trials,
    }
    print(f"      Shuffled mean: {shuffled_mean:.6f}, selectivity: {shuffled_sel:.4f}x")

    # (b) Random target matrix
    print("\n    Null test (b): random target matrix...")
    random_distances = []
    rng_b = np.random.RandomState(seed + 200)
    for trial in range(n_null_trials):
        rand_matrix = rng_b.random((n, n))
        rand_matrix *= (latin_matrix > 0).astype(float)
        total = rand_matrix.sum()
        if total > 0:
            rand_matrix /= total
        _, best_cost, _ = _fast_sa_permutation(
            voynich_matrix, rand_matrix, best_metric,
            max_iter=sa_max_iter, seed=seed + trial + 100,
        )
        random_distances.append(best_cost)

    random_mean = float(np.mean(random_distances))
    random_std = float(np.std(random_distances))
    random_sel = random_mean / best_distance if best_distance > 1e-10 else 0.0
    results['random_target'] = {
        'mean_distance': round(random_mean, 6),
        'std_distance': round(random_std, 6),
        'selectivity': round(random_sel, 4),
        'n_trials': n_null_trials,
    }
    print(f"      Random target mean: {random_mean:.6f}, selectivity: {random_sel:.4f}x")

    # (c) Latin-to-Latin sanity check
    print("\n    Null test (c): Latin-to-Latin sanity check...")
    true_perm = np.random.RandomState(seed + 500).permutation(n)
    enciphered_latin = latin_matrix[np.ix_(true_perm, true_perm)]
    recovered_perm, recovered_cost, _ = _fast_sa_permutation(
        enciphered_latin, latin_matrix, best_metric,
        max_iter=sa_max_iter * 2, seed=seed + 600,
    )
    inverse_true = np.zeros(n, dtype=int)
    inverse_true[true_perm] = np.arange(n)
    recovery_accuracy = float(np.mean(recovered_perm == inverse_true))
    results['latin_sanity'] = {
        'recovery_accuracy': round(recovery_accuracy, 4),
        'recovered_distance': round(recovered_cost, 6),
        'passed': recovery_accuracy > 0.3,
    }
    print(f"      Recovery accuracy: {recovery_accuracy:.4f}")

    # (d) Occitan target
    print("\n    Null test (d): Occitan target...")
    n_occ = min(n, occitan_matrix.shape[0])
    if n_occ > 0:
        v_sub = voynich_matrix[:n_occ, :n_occ]
        o_sub = occitan_matrix[:n_occ, :n_occ]
        _, occ_cost, _ = _fast_sa_permutation(
            v_sub, o_sub, best_metric,
            max_iter=sa_max_iter, seed=seed + 700,
        )
        occ_selectivity = occ_cost / best_distance if best_distance > 1e-10 else 0.0
        results['occitan_target'] = {
            'best_distance': round(occ_cost, 6),
            'latin_distance': round(best_distance, 6),
            'latin_vs_occitan_ratio': round(occ_selectivity, 4),
            'latin_better': best_distance < occ_cost,
        }
        print(f"      Occitan distance: {occ_cost:.6f} "
              f"(Latin: {best_distance:.6f}, ratio: {occ_selectivity:.4f})")
    else:
        results['occitan_target'] = {'skipped': True}

    return results


def run_cross_validation(
    corpus: VoynichCorpus,
    ref_corpus: ReferenceCorpus,
    best_metric: str,
    n_vocab: int = 200,
    sa_max_iter: int = 100_000,
    seed: int = 42,
) -> Dict:
    """
    Split-half cross-validation: split Voynich by folio ranges, build
    bigram matrices from each half, run SA independently, measure agreement.
    """
    print("\n    Cross-validation: split-half by folios...")

    # Split pages into two halves
    pages_a = corpus.get_pages_by_language('A')
    mid = len(pages_a) // 2
    half1_pages = pages_a[:mid]
    half2_pages = pages_a[mid:]

    half1_tokens = []
    for p in half1_pages:
        half1_tokens.extend(p.all_tokens)
    half2_tokens = []
    for p in half2_pages:
        half2_tokens.extend(p.all_tokens)

    if len(half1_tokens) < 100 or len(half2_tokens) < 100:
        return {'skipped': True, 'reason': 'insufficient_tokens'}

    # Build stem sequences
    h1_stems, h1_vocab, _ = _prepare_stem_sequence(half1_tokens, min_count=2)
    h2_stems, h2_vocab, _ = _prepare_stem_sequence(half2_tokens, min_count=2)

    # Build reference stem sequence
    ref_stems, ref_vocab, _ = _prepare_ref_stem_sequence(ref_corpus, 'latin', min_count=2)

    actual_n = min(n_vocab, len(h1_vocab), len(h2_vocab), len(ref_vocab))
    if actual_n < 20:
        return {'skipped': True, 'reason': 'vocabulary_too_small'}

    # Build matrices
    h1_mat, h1_v = _build_stem_bigram_matrix(h1_stems, h1_vocab, actual_n)
    h2_mat, h2_v = _build_stem_bigram_matrix(h2_stems, h2_vocab, actual_n)
    ref_mat, ref_v = _build_stem_bigram_matrix(ref_stems, ref_vocab, actual_n)

    # Run SA on each half
    perm1, cost1, _ = _fast_sa_permutation(
        h1_mat, ref_mat, best_metric,
        max_iter=sa_max_iter, seed=seed,
    )
    perm2, cost2, _ = _fast_sa_permutation(
        h2_mat, ref_mat, best_metric,
        max_iter=sa_max_iter, seed=seed + 100,
    )

    # Measure agreement: compare mappings via vocab overlap
    # Both halves map to ref_v, so compare perm1 vs perm2
    # Only compare stems that appear in both halves
    common_v = set(h1_v) & set(h2_v)
    if not common_v:
        return {'skipped': True, 'reason': 'no_common_stems'}

    # Build position lookup
    h1_idx = {s: i for i, s in enumerate(h1_v)}
    h2_idx = {s: i for i, s in enumerate(h2_v)}

    agree = 0
    total = 0
    for stem in common_v:
        i1 = h1_idx.get(stem)
        i2 = h2_idx.get(stem)
        if i1 is not None and i2 is not None and i1 < actual_n and i2 < actual_n:
            if perm1[i1] == perm2[i2]:
                agree += 1
            total += 1

    agreement = agree / total if total > 0 else 0.0
    print(f"      Split-half agreement: {agreement:.4f} "
          f"({agree}/{total} common stems)")

    return {
        'agreement': round(agreement, 4),
        'n_common_stems': total,
        'half1_tokens': len(half1_tokens),
        'half2_tokens': len(half2_tokens),
        'half1_distance': round(cost1, 6),
        'half2_distance': round(cost2, 6),
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_bigram_transfer() -> Dict:
    """
    Run Phase 8 / Approach 16: Bigram Transfer Cryptanalysis.

    1. Load corpus and reference corpora
    2. Build stem bigram matrices
    3. SA permutation search
    4. Assess mapping stability
    5. Null tests + cross-validation
    6. Gate check and save
    """
    print("=" * 70)
    print("PHASE 8 / APPROACH 16: BIGRAM TRANSFER CRYPTANALYSIS")
    print("=" * 70)

    # --- Load data ---
    print("\n--- 16.1a: Loading corpora ---")
    corpus = load_corpus(verbose=False)
    ref_corpus = load_reference_corpus(verbose=False)

    voynich_tokens = corpus.get_tokens(language='A')
    print(f"  Language A tokens: {len(voynich_tokens)}")

    # --- Build stem sequences ---
    print("\n--- 16.1b: Building stem sequences ---")
    v_stems, v_vocab, v_counts = _prepare_stem_sequence(voynich_tokens)
    print(f"  Voynich stems: {len(v_stems)} tokens, {len(v_vocab)} unique")

    l_stems, l_vocab, l_counts = _prepare_ref_stem_sequence(ref_corpus, 'latin')
    print(f"  Latin stems:   {len(l_stems)} tokens, {len(l_vocab)} unique")

    o_stems, o_vocab, o_counts = _prepare_ref_stem_sequence(ref_corpus, 'occitan')
    print(f"  Occitan stems: {len(o_stems)} tokens, {len(o_vocab)} unique")

    # --- Build bigram matrices ---
    n_vocab = 100  # Start with manageable size
    actual_n = min(n_vocab, len(v_vocab), len(l_vocab))
    print(f"\n--- 16.1c: Building bigram matrices (N={actual_n}) ---")

    v_mat, v_v = _build_stem_bigram_matrix(v_stems, v_vocab, actual_n)
    l_mat, l_v = _build_stem_bigram_matrix(l_stems, l_vocab, actual_n)
    o_n = min(actual_n, len(o_vocab))
    o_mat, o_v = _build_stem_bigram_matrix(o_stems, o_vocab, o_n)

    v_stats = _compute_matrix_stats(v_mat, v_v, 'voynich')
    l_stats = _compute_matrix_stats(l_mat, l_v, 'latin')
    o_stats = _compute_matrix_stats(o_mat, o_v, 'occitan')

    print(f"  Voynich: {v_stats.n_nonzero} non-zero, "
          f"sparsity={v_stats.sparsity:.3f}, entropy={v_stats.entropy:.2f}, "
          f"rank={v_stats.effective_rank}")
    print(f"  Latin:   {l_stats.n_nonzero} non-zero, "
          f"sparsity={l_stats.sparsity:.3f}, entropy={l_stats.entropy:.2f}, "
          f"rank={l_stats.effective_rank}")
    print(f"  Occitan: {o_stats.n_nonzero} non-zero, "
          f"sparsity={o_stats.sparsity:.3f}, entropy={o_stats.entropy:.2f}, "
          f"rank={o_stats.effective_rank}")

    # --- SA permutation search ---
    print("\n--- 16.2: SA Permutation Search ---")
    sa_results = run_sa_permutation_search(
        voynich_matrix=v_mat,
        voynich_vocab=v_v,
        target_matrix=l_mat,
        target_vocab=l_v,
        metrics=['frobenius'],
        n_restarts=10,
        max_iter=100_000,
        seed=42,
        verbose=True,
    )

    # Select best metric
    best_metric = min(sa_results, key=lambda m: -sa_results[m].selectivity)
    best_result = sa_results[best_metric]
    print(f"\n  Best metric: {best_metric} "
          f"(selectivity={best_result.selectivity:.4f}x)")

    # --- Stability analysis ---
    print("\n--- 16.3a: Mapping Stability Analysis ---")
    stability = assess_mapping_stability(
        voynich_matrix=v_mat,
        target_matrix=l_mat,
        voynich_vocab=v_v,
        target_vocab=l_v,
        metric=best_metric,
        n_chains=10,
        max_iter=50_000,
        seed=42,
    )
    print(f"  Mean pairwise agreement: {stability.mean_pairwise_agreement:.4f}")
    print(f"  High-confidence stems: {len(stability.high_confidence_stems)}")
    print(f"  Low-confidence stems:  {len(stability.low_confidence_stems)}")
    if stability.top_10_consistent:
        print("  Top-10 consistent mappings:")
        for v_stem, l_stem, conf in stability.top_10_consistent:
            print(f"    {v_stem} -> {l_stem}  (conf={conf})")

    # --- Decode sample ---
    print("\n--- 16.3b: Decoded Sample ---")
    decoded = decode_sample(voynich_tokens, best_result.best_permutation, n_tokens=30)
    for line in decoded[:15]:
        print(f"    {line}")
    if len(decoded) > 15:
        print(f"    ... ({len(decoded) - 15} more)")

    # --- Null tests ---
    print("\n--- 16.4: Validation Battery ---")
    null_results = run_null_tests(
        voynich_matrix=v_mat,
        voynich_vocab=v_v,
        latin_matrix=l_mat,
        latin_vocab=l_v,
        occitan_matrix=o_mat,
        occitan_vocab=o_v,
        best_distance=best_result.best_distance,
        best_metric=best_metric,
        n_null_trials=10,
        sa_max_iter=50_000,
        seed=42,
    )

    # --- Cross-validation ---
    cv_results = run_cross_validation(
        corpus=corpus,
        ref_corpus=ref_corpus,
        best_metric=best_metric,
        n_vocab=actual_n,
        sa_max_iter=50_000,
        seed=42,
    )

    # --- Gate check ---
    gate_selectivity = best_result.selectivity >= 1.5
    gate_stability = stability.mean_pairwise_agreement >= 0.3
    gate_passed = gate_selectivity and gate_stability

    if gate_passed:
        verdict = 'bigram_structure_matches_latin'
    elif gate_selectivity:
        verdict = 'selectivity_passed_stability_low'
    elif gate_stability:
        verdict = 'stability_passed_selectivity_low'
    else:
        verdict = 'bigram_structure_insufficient'

    print(f"\n--- Gate Check ---")
    print(f"  Selectivity gate: {best_result.selectivity:.4f}x >= 1.5 -> "
          f"{'PASSED' if gate_selectivity else 'FAILED'}")
    print(f"  Stability gate:   {stability.mean_pairwise_agreement:.4f} >= 0.3 -> "
          f"{'PASSED' if gate_stability else 'FAILED'}")
    print(f"  Overall: {'PASSED' if gate_passed else 'FAILED'}")
    print(f"  Verdict: {verdict}")

    # --- Build result ---
    result = BigramTransferResult(
        voynich_matrix_stats=_convert(asdict(v_stats)),
        latin_matrix_stats=_convert(asdict(l_stats)),
        occitan_matrix_stats=_convert(asdict(o_stats)),
        sa_results={k: _convert(asdict(v)) for k, v in sa_results.items()},
        best_metric=best_metric,
        best_n_vocab=actual_n,
        best_selectivity=round(best_result.selectivity, 4),
        stability=_convert(asdict(stability)),
        decoded_sample=decoded,
        null_tests=_convert(null_results),
        cross_validation=_convert(cv_results),
        gate_passed=gate_passed,
        verdict=verdict,
    )

    out = _convert(asdict(result))
    out_path = _results_dir() / 'bigram_transfer.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)

    print(f"\n  Results saved to {out_path}")
    return out
