"""
Phase 7 / Approach 8: Morpheme-Level Distributional Semantics
==============================================================
If the Voynich encodes a specific language, the geometric structure of
Voynich token embeddings should match the geometric structure of that
language's word embeddings — even though you can't identify which Voynich
token maps to which word. The structures should be alignable via rotation
and scaling (Procrustes analysis) or matched structurally (Gromov-Wasserstein).

Sub-analyses:
  8.1 — Build Voynich stem embedding spaces (Language A + Language B)
  8.2 — Build reference language embedding spaces (Latin, Occitan)
  8.3 — Procrustes + GW alignment and language ranking
  8.4 — Affix embedding space analysis
  8.5 — Cluster-level correspondence
  8.6 — Discriminant validation (null tests)

Output:
  results/distributional.json
"""

import json
import math
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import cdist

from voynich.core.corpus import load_corpus, VoynichCorpus
from voynich.core.stats import (
    build_cooccurrence_matrix, ppmi_matrix, truncated_svd,
    procrustes_alignment, gromov_wasserstein_distance,
    cosine_similarity, jensen_shannon_divergence,
    selectivity_ratio, bootstrap_ci, adjusted_rand_index,
)
from voynich.core._paths import results_dir as _results_dir
from voynich.core.reference import (
    load_reference_corpus, ReferenceCorpus,
    stem_latin_token, LATIN_DECLENSION_SUFFIXES,
)
from voynich.phases.morpheme_grid import (
    decompose_token_morphemes, decompose_corpus_morphemes,
    MorphemeDecomposition, KNOWN_PREFIXES, KNOWN_SUFFIXES,
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class EmbeddingSpace:
    """A stem-level embedding space."""
    label: str
    vocab: List[str]
    vocab_to_idx: Dict[str, int]
    embeddings: np.ndarray       # (n_vocab × n_dim)
    n_vocab: int
    n_dim: int
    total_tokens: int
    total_cooccurrences: int


@dataclass
class AlignmentResult:
    """Result of alignment between two embedding spaces."""
    method: str
    source_label: str
    target_label: str
    score: float                  # residual (Procrustes) or GW distance
    n_seed_pairs: int
    mean_nn_cosine: float         # mean cosine sim of nearest-neighbor pairs


@dataclass
class ClusterCorrespondence:
    """Cluster-level analysis of aligned spaces."""
    n_clusters: int
    ari: float
    cluster_size_rho: float
    voynich_cluster_sizes: List[int]
    reference_cluster_sizes: List[int]


@dataclass
class AffixEmbeddingResult:
    """Affix embedding space analysis."""
    n_affix_types: int
    n_dim: int
    prefix_suffix_separation: float   # Mean distance between groups
    within_prefix_similarity: float
    within_suffix_similarity: float


@dataclass
class LanguageAlignmentSummary:
    """Summary of alignment scores for one reference language."""
    language: str
    procrustes_residual: float
    gw_distance: float
    procrustes_nn_cosine: float
    n_seed_pairs: int
    cluster_ari: float


@dataclass
class PerLanguageAlignment:
    """Alignment results for one Voynich language (A or B) against all references."""
    voynich_language: str
    vocab_size: int
    embedding_dim: int
    total_tokens: int
    section_ari: float
    section_ari_null: float
    alignment_summaries: List[Dict]
    best_procrustes_language: str
    best_gw_language: str
    best_procrustes_residual: float
    best_gw_distance: float
    null_procrustes_mean: float
    null_procrustes_std: float
    procrustes_selectivity: float
    null_gw_mean: float
    null_gw_std: float
    gw_selectivity: float
    procrustes_gate: bool
    gw_gate: bool


@dataclass
class DistributionalResult:
    """Full Approach 8 output."""
    # 8.1: Voynich embeddings
    voynich_a_vocab_size: int
    voynich_a_embedding_dim: int
    voynich_a_total_tokens: int
    voynich_b_vocab_size: int
    voynich_b_embedding_dim: int
    voynich_b_total_tokens: int
    # 8.2: Reference embeddings
    reference_languages: List[str]
    reference_vocab_sizes: Dict[str, int]
    # 8.3: Per-language alignment
    language_a_alignment: Dict
    language_b_alignment: Dict
    # Cross-language convergence
    best_procrustes_language: str
    best_gw_language: str
    a_b_procrustes_agree: bool
    a_b_gw_agree: bool
    # 8.4: Affix space
    affix_n_types: int
    affix_prefix_suffix_separation: float
    # 8.5: Cluster correspondence
    cluster_results: Dict[str, Dict]
    # 8.1 validation: section clustering ARI
    section_ari_a: float
    section_ari_b: float
    section_ari_null: float
    embedding_quality_gate: bool
    # Gates
    procrustes_gate: bool
    gw_gate: bool
    convergence: bool
    gate_passed: bool
    verdict: str


# ---------------------------------------------------------------------------
# 8.1 — Build embedding spaces
# ---------------------------------------------------------------------------

def _prepare_stem_corpus(
    tokens: List[str],
    min_count: int = 3,
) -> Tuple[List[str], List[str]]:
    """
    Decompose tokens to stems and filter by frequency.

    Returns (stem_sequence, vocabulary).
    """
    stems = []
    for tok in tokens:
        d = decompose_token_morphemes(tok)
        stems.append(d.stem if d.stem else tok)

    counts = Counter(stems)
    vocab = [s for s, c in counts.most_common() if c >= min_count]
    vocab_set = set(vocab)
    filtered = [s for s in stems if s in vocab_set]
    return filtered, vocab


def build_embedding_space(
    tokens: List[str],
    label: str,
    window: int = 2,
    n_dim: int = 50,
    min_count: int = 3,
) -> Optional[EmbeddingSpace]:
    """
    Build a stem-level PPMI + SVD embedding space.

    Returns None if vocabulary is too small (< 20 stems).
    """
    stem_seq, vocab = _prepare_stem_corpus(tokens, min_count)
    if len(vocab) < 20:
        print(f"  WARNING: {label} vocabulary too small ({len(vocab)} stems). "
              "Skipping embedding construction.")
        return None

    print(f"  Building {label} embeddings: {len(vocab)} stems, {len(stem_seq)} tokens")
    cooc, word2idx = build_cooccurrence_matrix(stem_seq, vocab, window)
    pmi = ppmi_matrix(cooc, alpha=0.75)
    actual_dim = min(n_dim, len(vocab) - 1)
    embeddings = truncated_svd(pmi, n_components=actual_dim)

    return EmbeddingSpace(
        label=label,
        vocab=vocab,
        vocab_to_idx=word2idx,
        embeddings=embeddings,
        n_vocab=len(vocab),
        n_dim=embeddings.shape[1],
        total_tokens=len(stem_seq),
        total_cooccurrences=int(cooc.sum()),
    )


def _prepare_latin_stem_corpus(
    ref_corpus: ReferenceCorpus,
    language: str,
    min_count: int = 3,
) -> Tuple[List[str], List[str]]:
    """
    Prepare stem corpus from a reference language.

    For Latin: uses heuristic suffix stripping.
    For Occitan: uses same approach (Romance suffixes overlap).
    """
    tokens = ref_corpus.get_combined_tokens(language)
    stems = [stem_latin_token(t) for t in tokens]

    counts = Counter(stems)
    vocab = [s for s, c in counts.most_common() if c >= min_count]
    vocab_set = set(vocab)
    filtered = [s for s in stems if s in vocab_set]
    return filtered, vocab


def validate_embedding_quality(
    space: EmbeddingSpace,
    tokens: List[str],
    corpus: VoynichCorpus,
    language: str,
    n_clusters: int = 5,
    n_null_trials: int = 20,
) -> Tuple[float, float]:
    """
    Validate that embeddings capture meaningful structure.

    Clusters embeddings via k-means and measures ARI against section labels.
    Compares to null (shuffled corpus).

    Returns (real_ari, null_ari).
    """
    # Build section labels for each stem
    stem_sections: Dict[str, Counter] = defaultdict(Counter)
    pages = corpus.get_pages_by_language(language)
    for page in pages:
        section = page.section
        for tok in page.all_tokens:
            d = decompose_token_morphemes(tok)
            stem = d.stem if d.stem else tok
            if stem in space.vocab_to_idx:
                stem_sections[stem][section] += 1

    # Assign majority section to each stem
    stem_labels = {}
    for stem in space.vocab:
        if stem in stem_sections and stem_sections[stem]:
            stem_labels[stem] = stem_sections[stem].most_common(1)[0][0]
        else:
            stem_labels[stem] = 'unknown'

    # K-means clustering on embeddings
    from scipy.cluster.vq import kmeans2
    k = min(n_clusters, space.n_vocab)
    _, cluster_labels = kmeans2(space.embeddings, k, minit='points', seed=42)

    # ARI between cluster labels and section labels
    section_arr = np.array([stem_labels.get(s, 'unknown') for s in space.vocab])
    real_ari = adjusted_rand_index(cluster_labels, section_arr)

    # Null: shuffle section labels
    rng = np.random.RandomState(42)
    null_aris = []
    for _ in range(n_null_trials):
        shuffled_sections = section_arr.copy()
        rng.shuffle(shuffled_sections)
        null_ari = adjusted_rand_index(cluster_labels, shuffled_sections)
        null_aris.append(null_ari)
    null_mean = float(np.mean(null_aris))

    return real_ari, null_mean


# ---------------------------------------------------------------------------
# 8.3 — Alignment
# ---------------------------------------------------------------------------

def _load_seed_pairs() -> List[Dict]:
    """Load Phase 5.3 stem identifications for Procrustes seed pairs."""
    path = _results_dir() / 'stem_identification.json'
    if not path.exists():
        return []
    with open(path) as f:
        data = json.load(f)
    return data.get('identifications', [])


def find_seed_indices(
    voynich_space: EmbeddingSpace,
    ref_space: EmbeddingSpace,
    seed_pairs: List[Dict],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Map voynich_stem → latin_word pairs to embedding indices.

    Only includes pairs where both stem and word are in their
    respective vocabularies.
    """
    v_idx = []
    r_idx = []
    for pair in seed_pairs:
        v_stem = pair.get('voynich_stem', '')
        l_word = pair.get('latin_word', '')
        # Try to find latin_word's stem in reference space
        l_stem = stem_latin_token(l_word)
        if v_stem in voynich_space.vocab_to_idx and l_stem in ref_space.vocab_to_idx:
            v_idx.append(voynich_space.vocab_to_idx[v_stem])
            r_idx.append(ref_space.vocab_to_idx[l_stem])
    return np.array(v_idx, dtype=int), np.array(r_idx, dtype=int)


def align_procrustes(
    voynich_space: EmbeddingSpace,
    ref_space: EmbeddingSpace,
    seed_pairs: List[Dict],
) -> AlignmentResult:
    """
    Procrustes alignment using seed pairs from Phase 5.3.

    Aligns Voynich embedding space to reference via orthogonal rotation,
    then measures residual and nearest-neighbor quality.
    """
    v_idx, r_idx = find_seed_indices(voynich_space, ref_space, seed_pairs)
    n_seeds = len(v_idx)

    if n_seeds < 2:
        return AlignmentResult(
            method='procrustes',
            source_label=voynich_space.label,
            target_label=ref_space.label,
            score=float('inf'),
            n_seed_pairs=n_seeds,
            mean_nn_cosine=0.0,
        )

    # Ensure matching dimensions
    d = min(voynich_space.n_dim, ref_space.n_dim)
    v_emb = voynich_space.embeddings[:, :d]
    r_emb = ref_space.embeddings[:, :d]

    aligned, residual = procrustes_alignment(v_emb, r_emb, v_idx, r_idx)

    # Nearest-neighbor cosine similarity for non-seed items
    nn_cosines = []
    for i in range(voynich_space.n_vocab):
        if i in set(v_idx.tolist()):
            continue
        dists = cdist(aligned[i:i+1], r_emb, metric='cosine')[0]
        nn_cosines.append(1.0 - float(dists.min()))

    mean_nn = float(np.mean(nn_cosines)) if nn_cosines else 0.0

    return AlignmentResult(
        method='procrustes',
        source_label=voynich_space.label,
        target_label=ref_space.label,
        score=residual,
        n_seed_pairs=n_seeds,
        mean_nn_cosine=mean_nn,
    )


def align_gw(
    voynich_space: EmbeddingSpace,
    ref_space: EmbeddingSpace,
    top_n: int = 100,
) -> AlignmentResult:
    """
    Gromov-Wasserstein structural comparison.

    Compares internal distance geometry without requiring seed pairs.
    Uses top-N most frequent stems for tractability.
    """
    n_v = min(top_n, voynich_space.n_vocab)
    n_r = min(top_n, ref_space.n_vocab)

    d = min(voynich_space.n_dim, ref_space.n_dim)
    v_emb = voynich_space.embeddings[:n_v, :d]
    r_emb = ref_space.embeddings[:n_r, :d]

    # Build internal distance matrices
    dist_v = cdist(v_emb, v_emb, metric='cosine')
    dist_r = cdist(r_emb, r_emb, metric='cosine')

    gw = gromov_wasserstein_distance(dist_v, dist_r, n_iter=50)

    return AlignmentResult(
        method='gromov_wasserstein',
        source_label=voynich_space.label,
        target_label=ref_space.label,
        score=gw,
        n_seed_pairs=0,
        mean_nn_cosine=0.0,
    )


# ---------------------------------------------------------------------------
# 8.4 — Affix embedding space
# ---------------------------------------------------------------------------

def build_affix_embeddings(
    tokens: List[str],
    n_dim: int = 20,
    min_count: int = 3,
) -> Optional[AffixEmbeddingResult]:
    """
    Build affix-level embedding space.

    Uses affix-stem co-occurrence: which affixes appear with which stems.
    Tests whether prefixes and suffixes form separate clusters.
    """
    decomps = [decompose_token_morphemes(t) for t in tokens]

    # Build affix vocabulary
    affix_counts: Counter = Counter()
    for d in decomps:
        if d.prefix:
            affix_counts[f'pre:{d.prefix}'] += 1
        if d.suffix:
            affix_counts[f'suf:{d.suffix}'] += 1

    affix_vocab = [a for a, c in affix_counts.most_common() if c >= min_count]
    if len(affix_vocab) < 5:
        return None

    # Build affix-stem co-occurrence
    stem_counts = Counter()
    for d in decomps:
        stem = d.stem if d.stem else d.token
        stem_counts[stem] += 1
    stem_vocab = [s for s, c in stem_counts.most_common() if c >= min_count]

    if len(stem_vocab) < 5:
        return None

    affix2idx = {a: i for i, a in enumerate(affix_vocab)}
    stem2idx = {s: i for i, s in enumerate(stem_vocab)}
    cooc = np.zeros((len(affix_vocab), len(stem_vocab)))

    for d in decomps:
        stem = d.stem if d.stem else d.token
        if stem not in stem2idx:
            continue
        si = stem2idx[stem]
        if d.prefix and f'pre:{d.prefix}' in affix2idx:
            cooc[affix2idx[f'pre:{d.prefix}'], si] += 1
        if d.suffix and f'suf:{d.suffix}' in affix2idx:
            cooc[affix2idx[f'suf:{d.suffix}'], si] += 1

    pmi = ppmi_matrix(cooc, alpha=0.75)
    actual_dim = min(n_dim, min(pmi.shape) - 1)
    if actual_dim < 2:
        return None
    emb = truncated_svd(pmi, n_components=actual_dim)

    # Measure prefix-suffix separation
    pre_idx = [i for i, a in enumerate(affix_vocab) if a.startswith('pre:')]
    suf_idx = [i for i, a in enumerate(affix_vocab) if a.startswith('suf:')]

    if not pre_idx or not suf_idx:
        return AffixEmbeddingResult(
            n_affix_types=len(affix_vocab), n_dim=actual_dim,
            prefix_suffix_separation=0.0,
            within_prefix_similarity=0.0,
            within_suffix_similarity=0.0,
        )

    pre_emb = emb[pre_idx]
    suf_emb = emb[suf_idx]

    # Between-group mean distance
    cross_dists = cdist(pre_emb, suf_emb, metric='cosine')
    between = float(np.mean(cross_dists))

    # Within-group similarity
    if len(pre_idx) > 1:
        pre_dists = cdist(pre_emb, pre_emb, metric='cosine')
        within_pre = 1.0 - float(np.mean(pre_dists[np.triu_indices_from(pre_dists, k=1)]))
    else:
        within_pre = 1.0

    if len(suf_idx) > 1:
        suf_dists = cdist(suf_emb, suf_emb, metric='cosine')
        within_suf = 1.0 - float(np.mean(suf_dists[np.triu_indices_from(suf_dists, k=1)]))
    else:
        within_suf = 1.0

    return AffixEmbeddingResult(
        n_affix_types=len(affix_vocab),
        n_dim=actual_dim,
        prefix_suffix_separation=between,
        within_prefix_similarity=within_pre,
        within_suffix_similarity=within_suf,
    )


# ---------------------------------------------------------------------------
# 8.5 — Cluster correspondence
# ---------------------------------------------------------------------------

def cluster_correspondence(
    voynich_space: EmbeddingSpace,
    ref_space: EmbeddingSpace,
    n_clusters: int = 5,
) -> ClusterCorrespondence:
    """
    Compare cluster structure between two embedding spaces.

    Clusters both, then measures whether cluster size distributions
    correlate (rank correlation).
    """
    from scipy.cluster.vq import kmeans2

    k_v = min(n_clusters, voynich_space.n_vocab)
    k_r = min(n_clusters, ref_space.n_vocab)
    k = min(k_v, k_r)

    d = min(voynich_space.n_dim, ref_space.n_dim)
    _, v_labels = kmeans2(voynich_space.embeddings[:, :d], k, minit='points', seed=42)
    _, r_labels = kmeans2(ref_space.embeddings[:, :d], k, minit='points', seed=42)

    # Cluster sizes
    v_sizes = sorted(Counter(v_labels).values(), reverse=True)
    r_sizes = sorted(Counter(r_labels).values(), reverse=True)

    # Pad to same length
    max_len = max(len(v_sizes), len(r_sizes))
    v_sizes.extend([0] * (max_len - len(v_sizes)))
    r_sizes.extend([0] * (max_len - len(r_sizes)))

    from voynich.core.stats import rank_correlation
    rho, _ = rank_correlation(np.array(v_sizes), np.array(r_sizes))

    # ARI is not directly applicable between different spaces,
    # but we measure if the cluster structure is similar
    return ClusterCorrespondence(
        n_clusters=k,
        ari=0.0,  # Not directly comparable across spaces
        cluster_size_rho=rho,
        voynich_cluster_sizes=v_sizes,
        reference_cluster_sizes=r_sizes,
    )


# ---------------------------------------------------------------------------
# 8.6 — Null tests
# ---------------------------------------------------------------------------

def null_test_procrustes(
    voynich_space: EmbeddingSpace,
    ref_space: EmbeddingSpace,
    seed_pairs: List[Dict],
    real_residual: float,
    n_trials: int = 100,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Null test: shuffle seed pair assignments, re-run Procrustes.

    Returns (null_mean, null_std, selectivity).
    Selectivity = null_mean / real_residual (higher is better for residuals).
    """
    rng = random.Random(seed)
    v_idx, r_idx = find_seed_indices(voynich_space, ref_space, seed_pairs)
    n_seeds = len(v_idx)

    if n_seeds < 2:
        return 0.0, 0.0, 0.0

    d = min(voynich_space.n_dim, ref_space.n_dim)
    v_emb = voynich_space.embeddings[:, :d]
    r_emb = ref_space.embeddings[:, :d]

    null_residuals = []
    for _ in range(n_trials):
        # Shuffle reference indices (break the mapping)
        shuffled_r_idx = r_idx.copy()
        rng.shuffle(shuffled_r_idx)
        _, res = procrustes_alignment(v_emb, r_emb, v_idx, shuffled_r_idx)
        null_residuals.append(res)

    null_arr = np.array(null_residuals)
    null_mean = float(np.mean(null_arr))
    null_std = float(np.std(null_arr))
    # For residuals, lower is better, so selectivity = null_mean / real
    sel = null_mean / real_residual if real_residual > 1e-10 else float('inf')
    return null_mean, null_std, sel


def null_test_gw(
    voynich_space: EmbeddingSpace,
    ref_space: EmbeddingSpace,
    real_gw: float,
    top_n: int = 100,
    n_trials: int = 20,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Null test: randomly rotate reference space, re-compute GW.

    Uses fewer trials (GW is expensive). Returns (null_mean, null_std, selectivity).
    """
    rng = np.random.RandomState(seed)
    n_r = min(top_n, ref_space.n_vocab)
    n_v = min(top_n, voynich_space.n_vocab)
    d = min(voynich_space.n_dim, ref_space.n_dim)

    v_emb = voynich_space.embeddings[:n_v, :d]
    dist_v = cdist(v_emb, v_emb, metric='cosine')

    null_gws = []
    for _ in range(n_trials):
        # Random orthogonal rotation
        Q, _ = np.linalg.qr(rng.randn(d, d))
        r_rotated = ref_space.embeddings[:n_r, :d] @ Q
        dist_r = cdist(r_rotated, r_rotated, metric='cosine')
        gw = gromov_wasserstein_distance(dist_v, dist_r, n_iter=30)
        null_gws.append(gw)

    null_arr = np.array(null_gws)
    null_mean = float(np.mean(null_arr))
    null_std = float(np.std(null_arr))
    sel = null_mean / real_gw if real_gw > 1e-10 else float('inf')
    return null_mean, null_std, sel


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

def _print_alignment_block(label: str, align: Dict):
    """Print alignment results for one Voynich language."""
    summaries = align.get('alignment_summaries', [])
    if not summaries:
        print(f"\n--- 8.3: {label} Alignment ---")
        print("  (no alignment — embedding space not built)")
        return

    print(f"\n--- 8.3: {label} Alignment ---")
    print(f"  Vocab: {align.get('vocab_size', 0)} stems, "
          f"dim={align.get('embedding_dim', 0)}, "
          f"{align.get('total_tokens', 0)} tokens")
    print(f"  Section ARI: {align.get('section_ari', 0):.4f} "
          f"(null: {align.get('section_ari_null', 0):.4f})")
    print(f"  {'Language':<12} {'Procrustes':<14} {'GW dist':<12} {'Seeds':<8} {'NN cosine':<12}")
    for summ in summaries:
        print(f"  {summ['language']:<12} {summ['procrustes_residual']:<14.4f} "
              f"{summ['gw_distance']:<12.4f} {summ['n_seed_pairs']:<8} "
              f"{summ['procrustes_nn_cosine']:<12.4f}")
    print(f"  Best Procrustes: {align.get('best_procrustes_language', 'none')}")
    print(f"  Best GW:         {align.get('best_gw_language', 'none')}")
    proc_sel = align.get('procrustes_selectivity', 0)
    gw_sel = align.get('gw_selectivity', 0)
    print(f"  Procrustes selectivity: {proc_sel:.2f}x  "
          f"(null u={align.get('null_procrustes_mean', 0):.4f} "
          f"s={align.get('null_procrustes_std', 0):.4f})")
    print(f"  GW selectivity:         {gw_sel:.2f}x  "
          f"(null u={align.get('null_gw_mean', 0):.4f} "
          f"s={align.get('null_gw_std', 0):.4f})")
    print(f"  Procrustes gate: {'PASS' if align.get('procrustes_gate') else 'FAIL'}")
    print(f"  GW gate:         {'PASS' if align.get('gw_gate') else 'FAIL'}")


def _print_results(result: DistributionalResult):
    """Print formatted results to console."""
    print("\n" + "=" * 70)
    print("APPROACH 8: MORPHEME-LEVEL DISTRIBUTIONAL SEMANTICS")
    print("=" * 70)

    print("\n--- 8.1: Voynich Embedding Spaces ---")
    print(f"  Language A: {result.voynich_a_vocab_size} stems, "
          f"dim={result.voynich_a_embedding_dim}, {result.voynich_a_total_tokens} tokens")
    print(f"  Language B: {result.voynich_b_vocab_size} stems, "
          f"dim={result.voynich_b_embedding_dim}, {result.voynich_b_total_tokens} tokens")
    print(f"  Section clustering ARI (A): {result.section_ari_a:.4f}")
    print(f"  Section clustering ARI (B): {result.section_ari_b:.4f}")
    print(f"  Null ARI:                   {result.section_ari_null:.4f}")
    print(f"  Embedding quality gate:     {'PASS' if result.embedding_quality_gate else 'FAIL'}")

    print("\n--- 8.2: Reference Embeddings ---")
    for lang in result.reference_languages:
        vs = result.reference_vocab_sizes.get(lang, 0)
        print(f"  {lang}: {vs} stems")

    # Per-language alignment results
    _print_alignment_block('Language A', result.language_a_alignment)
    _print_alignment_block('Language B', result.language_b_alignment)

    print(f"\n--- Cross-Language Convergence ---")
    print(f"  A+B Procrustes agree: {'YES' if result.a_b_procrustes_agree else 'NO'}")
    print(f"  A+B GW agree:         {'YES' if result.a_b_gw_agree else 'NO'}")

    print("\n--- 8.4: Affix Embedding Space ---")
    print(f"  Affix types:               {result.affix_n_types}")
    print(f"  Prefix/suffix separation:  {result.affix_prefix_suffix_separation:.4f}")

    print("\n--- 8.5: Cluster Correspondence ---")
    for lang, cr in result.cluster_results.items():
        print(f"  {lang}: size_rho={cr.get('cluster_size_rho', 0):.3f}")

    print(f"\n--- Overall ---")
    print(f"  Best Procrustes: {result.best_procrustes_language}")
    print(f"  Best GW:         {result.best_gw_language}")
    print(f"  Convergence:     {'YES' if result.convergence else 'NO'}")
    print(f"  Procrustes gate: {'PASS' if result.procrustes_gate else 'FAIL'}")
    print(f"  GW gate:         {'PASS' if result.gw_gate else 'FAIL'}")
    print(f"  Overall gate:    {'PASS' if result.gate_passed else 'FAIL'}")
    print(f"  Verdict:         {result.verdict}")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def _convert(obj):
    """Convert dataclass/numpy types to JSON-serializable form."""
    if hasattr(obj, '__dataclass_fields__'):
        return {k: _convert(v) for k, v in asdict(obj).items()}
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, dict):
        return {str(k): _convert(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_convert(v) for v in obj]
    return obj


def _align_single_space(
    voynich_space: Optional[EmbeddingSpace],
    ref_spaces: Dict[str, EmbeddingSpace],
    seed_pairs: List[Dict],
    language_label: str,
    ari: float,
    ari_null: float,
) -> Dict:
    """
    Align a single Voynich embedding space against all reference languages.

    Runs Procrustes + GW alignment and null tests independently.
    Returns a dict with PerLanguageAlignment fields.
    """
    if voynich_space is None:
        return _convert(asdict(PerLanguageAlignment(
            voynich_language=language_label,
            vocab_size=0, embedding_dim=0, total_tokens=0,
            section_ari=ari, section_ari_null=ari_null,
            alignment_summaries=[],
            best_procrustes_language='none', best_gw_language='none',
            best_procrustes_residual=float('inf'), best_gw_distance=float('inf'),
            null_procrustes_mean=0.0, null_procrustes_std=0.0, procrustes_selectivity=0.0,
            null_gw_mean=0.0, null_gw_std=0.0, gw_selectivity=0.0,
            procrustes_gate=False, gw_gate=False,
        )))

    summaries = []
    best_proc_lang = 'none'
    best_proc_score = float('inf')
    best_gw_lang = 'none'
    best_gw_score = float('inf')

    for lang, ref_space in ref_spaces.items():
        print(f"    Aligning {language_label} to {lang}...")
        proc_result = align_procrustes(voynich_space, ref_space, seed_pairs)
        gw_result = align_gw(voynich_space, ref_space, top_n=100)
        cc = cluster_correspondence(voynich_space, ref_space, n_clusters=5)

        summary = LanguageAlignmentSummary(
            language=lang,
            procrustes_residual=proc_result.score,
            gw_distance=gw_result.score,
            procrustes_nn_cosine=proc_result.mean_nn_cosine,
            n_seed_pairs=proc_result.n_seed_pairs,
            cluster_ari=cc.ari,
        )
        summaries.append(summary)

        if proc_result.score < best_proc_score:
            best_proc_score = proc_result.score
            best_proc_lang = lang
        if gw_result.score < best_gw_score:
            best_gw_score = gw_result.score
            best_gw_lang = lang

    # Null tests against best reference language
    null_proc_mean, null_proc_std, proc_sel = 0.0, 0.0, 0.0
    null_gw_mean, null_gw_std, gw_sel = 0.0, 0.0, 0.0

    if best_proc_lang in ref_spaces:
        best_ref = ref_spaces[best_proc_lang]
        if best_proc_score < float('inf'):
            print(f"    Procrustes null test for {language_label} (100 trials)...")
            null_proc_mean, null_proc_std, proc_sel = null_test_procrustes(
                voynich_space, best_ref, seed_pairs, best_proc_score,
            )
        if best_gw_score < float('inf'):
            print(f"    GW null test for {language_label} (20 trials)...")
            null_gw_mean, null_gw_std, gw_sel = null_test_gw(
                voynich_space, best_ref, best_gw_score,
            )

    return _convert(asdict(PerLanguageAlignment(
        voynich_language=language_label,
        vocab_size=voynich_space.n_vocab,
        embedding_dim=voynich_space.n_dim,
        total_tokens=voynich_space.total_tokens,
        section_ari=ari,
        section_ari_null=ari_null,
        alignment_summaries=[_convert(asdict(s)) for s in summaries],
        best_procrustes_language=best_proc_lang,
        best_gw_language=best_gw_lang,
        best_procrustes_residual=best_proc_score,
        best_gw_distance=best_gw_score,
        null_procrustes_mean=null_proc_mean,
        null_procrustes_std=null_proc_std,
        procrustes_selectivity=proc_sel,
        null_gw_mean=null_gw_mean,
        null_gw_std=null_gw_std,
        gw_selectivity=gw_sel,
        procrustes_gate=proc_sel > 1.5,
        gw_gate=gw_sel > 1.5,
    )))


def run_distributional() -> Dict:
    """
    Run Approach 8: Morpheme-Level Distributional Semantics.

    Builds stem embedding spaces for both Voynich Language A and B,
    reference languages (Latin, Occitan), then tests alignment via
    Procrustes (with Phase 5.3 seed pairs) and Gromov-Wasserstein
    (no seeds needed).
    """
    print("Loading corpus and reference data...")
    corpus = load_corpus(verbose=False)
    ref_corpus = load_reference_corpus()

    # 8.1 — Build Voynich embedding spaces
    print("\n8.1: Building Voynich stem embedding spaces...")
    tokens_a = corpus.get_tokens(language='A', paragraph_only=True)
    tokens_b = corpus.get_tokens(language='B', paragraph_only=True)

    space_a = build_embedding_space(tokens_a, 'voynich_A', window=2, n_dim=50, min_count=3)
    space_b = build_embedding_space(tokens_b, 'voynich_B', window=2, n_dim=50, min_count=3)

    # Validate embedding quality
    ari_a, ari_null_a = (0.0, 0.0)
    ari_b, ari_null_b = (0.0, 0.0)
    if space_a:
        ari_a, ari_null_a = validate_embedding_quality(
            space_a, tokens_a, corpus, 'A', n_clusters=5,
        )
        print(f"  Language A ARI: {ari_a:.4f} (null: {ari_null_a:.4f})")
    if space_b:
        ari_b, ari_null_b = validate_embedding_quality(
            space_b, tokens_b, corpus, 'B', n_clusters=5,
        )
        print(f"  Language B ARI: {ari_b:.4f} (null: {ari_null_b:.4f})")

    best_ari = max(ari_a, ari_b)
    best_null = max(ari_null_a, ari_null_b)
    embedding_quality_gate = best_ari > best_null and best_ari > 0.01

    # Choose primary Voynich space (whichever has better ARI, or B if both fail)
    primary_space = space_b if (space_b and ari_b >= ari_a) else space_a
    if primary_space is None:
        primary_space = space_a or space_b

    # 8.2 — Build reference embedding spaces
    print("\n8.2: Building reference embedding spaces...")
    ref_spaces: Dict[str, EmbeddingSpace] = {}
    ref_vocab_sizes: Dict[str, int] = {}
    for lang in ('latin', 'occitan'):
        try:
            ref_stem_seq, ref_vocab = _prepare_latin_stem_corpus(ref_corpus, lang, min_count=3)
            if len(ref_vocab) < 20:
                print(f"  {lang}: vocabulary too small ({len(ref_vocab)}), skipping")
                continue
            ref_space = build_embedding_space(
                ref_stem_seq, lang, window=2, n_dim=50, min_count=1,
            )
            if ref_space:
                ref_spaces[lang] = ref_space
                ref_vocab_sizes[lang] = ref_space.n_vocab
        except Exception as e:
            print(f"  {lang}: error building embeddings: {e}")

    # 8.3 — Alignment (both A and B independently)
    print("\n8.3: Running alignment...")
    seed_pairs = _load_seed_pairs()
    print(f"  Loaded {len(seed_pairs)} seed pairs from Phase 5.3")

    print("  --- Language A alignment ---")
    lang_a_align = _align_single_space(
        space_a, ref_spaces, seed_pairs, 'A', ari_a, ari_null_a,
    )
    print("  --- Language B alignment ---")
    lang_b_align = _align_single_space(
        space_b, ref_spaces, seed_pairs, 'B', ari_b, ari_null_b,
    )

    # Cross-language convergence
    a_proc_best = lang_a_align.get('best_procrustes_language', 'none')
    b_proc_best = lang_b_align.get('best_procrustes_language', 'none')
    a_gw_best = lang_a_align.get('best_gw_language', 'none')
    b_gw_best = lang_b_align.get('best_gw_language', 'none')

    a_b_proc_agree = (a_proc_best == b_proc_best) and a_proc_best != 'none'
    a_b_gw_agree = (a_gw_best == b_gw_best) and a_gw_best != 'none'

    # Overall best language: prefer agreement; otherwise use best-ARI space
    if a_b_proc_agree:
        best_procrustes_lang = a_proc_best
    elif ari_a >= ari_b:
        best_procrustes_lang = a_proc_best
    else:
        best_procrustes_lang = b_proc_best

    if a_b_gw_agree:
        best_gw_lang = a_gw_best
    elif ari_a >= ari_b:
        best_gw_lang = a_gw_best
    else:
        best_gw_lang = b_gw_best

    # 8.4 — Affix embedding space
    print("\n8.4: Building affix embedding space...")
    all_tokens = tokens_a + tokens_b
    affix_result = build_affix_embeddings(all_tokens, n_dim=20)

    # 8.5 — Cluster correspondence (use primary space = best ARI)
    primary_space = space_a if ari_a >= ari_b else space_b
    if primary_space is None:
        primary_space = space_a or space_b
    cluster_results = {}
    if primary_space:
        for lang, ref_space in ref_spaces.items():
            cc = cluster_correspondence(primary_space, ref_space, n_clusters=5)
            cluster_results[lang] = _convert(asdict(cc))

    # Gates (pass if EITHER language passes)
    a_proc_gate = lang_a_align.get('procrustes_gate', False)
    b_proc_gate = lang_b_align.get('procrustes_gate', False)
    a_gw_gate = lang_a_align.get('gw_gate', False)
    b_gw_gate = lang_b_align.get('gw_gate', False)

    procrustes_gate = a_proc_gate or b_proc_gate
    gw_gate = a_gw_gate or b_gw_gate
    convergence = best_procrustes_lang == best_gw_lang and best_procrustes_lang != 'none'
    gate_passed = (procrustes_gate or gw_gate) and embedding_quality_gate

    if gate_passed and convergence:
        verdict = f'structural_match_{best_procrustes_lang}'
    elif gate_passed:
        verdict = 'partial_structural_match'
    elif embedding_quality_gate:
        verdict = 'embeddings_valid_no_language_match'
    else:
        verdict = 'embedding_quality_insufficient'

    result = DistributionalResult(
        voynich_a_vocab_size=space_a.n_vocab if space_a else 0,
        voynich_a_embedding_dim=space_a.n_dim if space_a else 0,
        voynich_a_total_tokens=space_a.total_tokens if space_a else 0,
        voynich_b_vocab_size=space_b.n_vocab if space_b else 0,
        voynich_b_embedding_dim=space_b.n_dim if space_b else 0,
        voynich_b_total_tokens=space_b.total_tokens if space_b else 0,
        reference_languages=list(ref_spaces.keys()),
        reference_vocab_sizes=ref_vocab_sizes,
        language_a_alignment=lang_a_align,
        language_b_alignment=lang_b_align,
        best_procrustes_language=best_procrustes_lang,
        best_gw_language=best_gw_lang,
        a_b_procrustes_agree=a_b_proc_agree,
        a_b_gw_agree=a_b_gw_agree,
        affix_n_types=affix_result.n_affix_types if affix_result else 0,
        affix_prefix_suffix_separation=affix_result.prefix_suffix_separation if affix_result else 0.0,
        cluster_results=cluster_results,
        section_ari_a=ari_a,
        section_ari_b=ari_b,
        section_ari_null=best_null,
        embedding_quality_gate=embedding_quality_gate,
        procrustes_gate=procrustes_gate,
        gw_gate=gw_gate,
        convergence=convergence,
        gate_passed=gate_passed,
        verdict=verdict,
    )

    _print_results(result)

    out = _convert(asdict(result))
    out_path = _results_dir() / 'distributional.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return out
