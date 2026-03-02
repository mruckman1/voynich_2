"""
Phase 9.1 — Homophonic Substitution Test
=========================================

Rationale
---------
In a homophonic cipher, common plaintext letters/words are assigned multiple
ciphertext symbols to flatten the frequency distribution.  Groups of ciphertext
symbols share distributional properties because they all encode the same
plaintext element.

Sub-analyses
------------
9.1a  Measure vocabulary inflation (V_voynich vs reference languages)
9.1b  Cluster Voynich stems by distributional similarity (cosine > 0.8)
9.1c  Test homophonic decoding (merge groups, rebuild bigrams, compare)
Null  Run same clustering on Latin stems
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import load_corpus
from voynich.core.reference import load_reference_corpus, stem_token
from voynich.core.stats import (
    build_cooccurrence_matrix,
    build_ngram_lm,
    cosine_similarity,
    cross_entropy_lm,
    ppmi_matrix,
    selectivity_ratio,
    truncated_svd,
)
from voynich.phases.morpheme_grid import decompose_token_morphemes


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class VocabularyInflation:
    voynich_stem_types: int
    voynich_token_count: int
    voynich_ttr: float
    reference_comparisons: List[Dict]
    inflation_range: Tuple[float, float]


@dataclass
class DistributionalCluster:
    cluster_id: int
    stems: List[str]
    n_stems: int
    mean_internal_cosine: float
    representative_stem: str


@dataclass
class ClusteringResult:
    n_stems_input: int
    n_pairs_above_threshold: int
    n_clusters: int
    mean_cluster_size: float
    max_cluster_size: int
    effective_vocab_after_merge: int
    reduction_ratio: float
    top_clusters: List[Dict]
    threshold: float


@dataclass
class MergedDecodingResult:
    merged_vocab_size: int
    baseline_sa_selectivity: float
    baseline_mdl_selectivity: float
    note: str


@dataclass
class HomophoneTestResult:
    vocabulary_inflation: Dict
    voynich_clustering: Dict
    merged_decoding: Dict
    latin_clustering: Dict
    latin_n_clusters: int
    latin_reduction_ratio: float
    cluster_selectivity: float
    gate_inflation: bool
    gate_selectivity: bool
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


def _prepare_stems(tokens: List[str], min_count: int = 3) -> Tuple[List[str], List[str], Counter]:
    """Decompose tokens to stems, filter by frequency."""
    stems = []
    for tok in tokens:
        d = decompose_token_morphemes(tok)
        stems.append(d.stem if d.stem else tok)
    counts = Counter(stems)
    vocab = [s for s, c in counts.most_common() if c >= min_count]
    vocab_set = set(vocab)
    filtered = [s for s in stems if s in vocab_set]
    return filtered, vocab, counts


def _prepare_ref_stems(
    ref_corpus, language: str, min_count: int = 3,
) -> Tuple[List[str], List[str], Counter]:
    """Prepare stem sequence from reference language."""
    tokens = ref_corpus.get_combined_tokens(language)
    stems = [stem_token(t, language) for t in tokens]
    counts = Counter(stems)
    vocab = [s for s, c in counts.most_common() if c >= min_count]
    vocab_set = set(vocab)
    filtered = [s for s in stems if s in vocab_set]
    return filtered, vocab, counts


def _build_embeddings(
    stem_seq: List[str], vocab: List[str], n_components: int = 50,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Build PPMI + SVD embeddings for stems. Returns (matrix, stem->idx)."""
    stem_to_idx = {s: i for i, s in enumerate(vocab)}
    cooc, _ = build_cooccurrence_matrix(stem_seq, vocab, window=2)
    weighted = ppmi_matrix(cooc, alpha=0.75)
    n_comp = min(n_components, len(vocab) - 1, weighted.shape[0] - 1)
    if n_comp < 1:
        return np.zeros((len(vocab), 1)), stem_to_idx
    embeddings = truncated_svd(weighted, n_components=n_comp)
    return embeddings, stem_to_idx


def _single_linkage_cluster(
    pairs: List[Tuple[int, int, float]], n_items: int,
) -> List[List[int]]:
    """Union-find based single-linkage clustering from above-threshold pairs."""
    parent = list(range(n_items))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i, j, _ in pairs:
        union(i, j)

    clusters: Dict[int, List[int]] = {}
    for i in range(n_items):
        root = find(i)
        clusters.setdefault(root, []).append(i)

    # Only return clusters with 2+ members
    return [members for members in clusters.values() if len(members) >= 2]


def _cluster_stems(
    embeddings: np.ndarray, vocab: List[str], threshold: float = 0.8,
) -> ClusteringResult:
    """Find distributional homophone clusters via cosine similarity."""
    n = len(vocab)
    # Compute pairwise cosines for above-threshold pairs
    # Normalize for fast cosine via dot product
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    normed = embeddings / norms

    pairs = []
    for i in range(n):
        # Vectorized cosine for row i against all j > i
        if i + 1 >= n:
            break
        sims = normed[i] @ normed[i + 1:].T
        above = np.where(sims >= threshold)[0]
        for offset in above:
            j = i + 1 + offset
            pairs.append((i, j, float(sims[offset])))

    clusters_idx = _single_linkage_cluster(pairs, n)

    # Convert to stem-level clusters
    cluster_objs = []
    for cid, members in enumerate(clusters_idx):
        stems = [vocab[m] for m in members]
        # Mean internal cosine
        if len(members) > 1:
            cos_vals = []
            for a_idx in range(len(members)):
                for b_idx in range(a_idx + 1, len(members)):
                    cos_vals.append(float(
                        normed[members[a_idx]] @ normed[members[b_idx]]
                    ))
            mean_cos = float(np.mean(cos_vals))
        else:
            mean_cos = 1.0
        # Representative = most frequent stem in cluster
        rep = stems[0]  # vocab is already frequency-sorted
        cluster_objs.append(DistributionalCluster(
            cluster_id=cid, stems=stems, n_stems=len(stems),
            mean_internal_cosine=mean_cos, representative_stem=rep,
        ))

    # Sort by size descending
    cluster_objs.sort(key=lambda c: c.n_stems, reverse=True)

    n_merged = sum(c.n_stems for c in cluster_objs)
    effective_vocab = n - n_merged + len(cluster_objs)

    return ClusteringResult(
        n_stems_input=n,
        n_pairs_above_threshold=len(pairs),
        n_clusters=len(cluster_objs),
        mean_cluster_size=float(np.mean([c.n_stems for c in cluster_objs])) if cluster_objs else 0.0,
        max_cluster_size=max((c.n_stems for c in cluster_objs), default=0),
        effective_vocab_after_merge=effective_vocab,
        reduction_ratio=effective_vocab / n if n > 0 else 1.0,
        top_clusters=[_convert(asdict(c)) for c in cluster_objs[:20]],
        threshold=threshold,
    )


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_homophone_test() -> Dict:
    """
    Phase 9.1: Test whether the Voynich vocabulary is inflated by
    homophonic substitution (multiple tokens encoding the same word).
    """
    print("Phase 9.1: Homophonic Substitution Test")
    print("=" * 60)

    # --- Load data ---
    corpus = load_corpus(verbose=False)
    ref_corpus = load_reference_corpus(verbose=False)
    voynich_tokens = corpus.get_tokens(language='A')

    # ===================================================================
    # 9.1a: Vocabulary inflation
    # ===================================================================
    print("\n  9.1a: Vocabulary inflation ...")
    v_stems, v_vocab, v_counts = _prepare_stems(voynich_tokens)
    v_stem_types = len(v_vocab)
    v_ttr = v_stem_types / len(v_stems) if v_stems else 0.0
    print(f"    Voynich: {v_stem_types} stem types, {len(v_stems):,} tokens, TTR={v_ttr:.4f}")

    ref_comparisons = []
    for lang in ('latin', 'occitan', 'italian', 'german'):
        try:
            r_stems, r_vocab, _ = _prepare_ref_stems(ref_corpus, lang)
            if not r_stems:
                continue
            r_types = len(r_vocab)
            r_ttr = r_types / len(r_stems) if r_stems else 0.0
            ratio = v_stem_types / r_types if r_types > 0 else 0.0
            ref_comparisons.append({
                'language': lang,
                'stem_types': r_types,
                'token_count': len(r_stems),
                'ttr': r_ttr,
                'inflation_ratio': ratio,
            })
            print(f"    {lang}: {r_types} types, {len(r_stems):,} tokens, "
                  f"TTR={r_ttr:.4f}, inflation={ratio:.2f}x")
        except Exception as e:
            print(f"    {lang}: failed — {e}")

    ratios = [r['inflation_ratio'] for r in ref_comparisons]
    inflation_range = (min(ratios), max(ratios)) if ratios else (0.0, 0.0)

    vocab_inflation = VocabularyInflation(
        voynich_stem_types=v_stem_types,
        voynich_token_count=len(v_stems),
        voynich_ttr=v_ttr,
        reference_comparisons=ref_comparisons,
        inflation_range=inflation_range,
    )

    # ===================================================================
    # 9.1b: Distributional clustering
    # ===================================================================
    print("\n  9.1b: Distributional clustering (cosine > 0.8) ...")
    v_embeddings, v_idx = _build_embeddings(v_stems, v_vocab)
    v_clustering = _cluster_stems(v_embeddings, v_vocab, threshold=0.8)

    print(f"    Pairs above threshold: {v_clustering.n_pairs_above_threshold}")
    print(f"    Clusters found: {v_clustering.n_clusters}")
    print(f"    Mean cluster size: {v_clustering.mean_cluster_size:.1f}")
    print(f"    Max cluster size: {v_clustering.max_cluster_size}")
    print(f"    Effective vocab after merge: {v_clustering.effective_vocab_after_merge} "
          f"(reduction ratio: {v_clustering.reduction_ratio:.3f})")

    if v_clustering.top_clusters:
        print("    Top 5 clusters:")
        for cl in v_clustering.top_clusters[:5]:
            stems_str = ', '.join(cl['stems'][:6])
            if cl['n_stems'] > 6:
                stems_str += f", ... (+{cl['n_stems'] - 6})"
            print(f"      [{cl['n_stems']}] cos={cl['mean_internal_cosine']:.3f}: "
                  f"{stems_str}")

    # ===================================================================
    # 9.1c: Merged decoding comparison
    # ===================================================================
    print("\n  9.1c: Merged decoding comparison ...")

    # Load Phase 8 baselines
    baseline_sa = 0.0
    baseline_mdl = 0.0
    try:
        with open(_results_dir() / 'bigram_transfer.json', 'r') as f:
            bt_result = json.load(f)
        # Extract best selectivity from SA results
        sa_results = bt_result.get('sa_results', {})
        for key, val in sa_results.items():
            if isinstance(val, dict) and 'selectivity' in val:
                baseline_sa = max(baseline_sa, val['selectivity'])
    except (FileNotFoundError, json.JSONDecodeError):
        print("    bigram_transfer.json not found, using 0.0 baseline")

    try:
        with open(_results_dir() / 'mdl_decode.json', 'r') as f:
            mdl_result = json.load(f)
        mdl_results = mdl_result.get('mcmc_results', {})
        for key, val in mdl_results.items():
            if isinstance(val, dict) and 'compression_ratio' in val:
                baseline_mdl = max(baseline_mdl, val.get('compression_ratio', 0.0))
    except (FileNotFoundError, json.JSONDecodeError):
        print("    mdl_decode.json not found, using 0.0 baseline")

    merged_decoding = MergedDecodingResult(
        merged_vocab_size=v_clustering.effective_vocab_after_merge,
        baseline_sa_selectivity=baseline_sa,
        baseline_mdl_selectivity=baseline_mdl,
        note=("Full merged SA/MDL re-run requires Phase 8 infrastructure. "
              "Vocabulary reduction ratio indicates potential improvement: "
              f"{1.0 - v_clustering.reduction_ratio:.1%} reduction."),
    )
    print(f"    Phase 8 SA baseline selectivity: {baseline_sa:.3f}")
    print(f"    Phase 8 MDL baseline selectivity: {baseline_mdl:.3f}")
    print(f"    Vocabulary reduction: {1.0 - v_clustering.reduction_ratio:.1%}")

    # ===================================================================
    # Null test: Latin stem clustering
    # ===================================================================
    print("\n  Null test: Latin stem clustering ...")
    try:
        l_stems, l_vocab, _ = _prepare_ref_stems(ref_corpus, 'latin')
        # Match vocab size to Voynich for fair comparison
        l_vocab_trimmed = l_vocab[:len(v_vocab)]
        l_stems_trimmed = [s for s in l_stems if s in set(l_vocab_trimmed)]
        l_embeddings, _ = _build_embeddings(l_stems_trimmed, l_vocab_trimmed)
        l_clustering = _cluster_stems(l_embeddings, l_vocab_trimmed, threshold=0.8)

        latin_n_clusters = l_clustering.n_clusters
        latin_reduction = l_clustering.reduction_ratio
        print(f"    Latin clusters: {latin_n_clusters}  "
              f"reduction ratio: {latin_reduction:.3f}")
    except Exception as e:
        print(f"    Latin clustering failed: {e}")
        l_clustering = ClusteringResult(
            n_stems_input=0, n_pairs_above_threshold=0,
            n_clusters=0, mean_cluster_size=0.0, max_cluster_size=0,
            effective_vocab_after_merge=0, reduction_ratio=1.0,
            top_clusters=[], threshold=0.8,
        )
        latin_n_clusters = 0
        latin_reduction = 1.0

    # Selectivity: Voynich should reduce MORE than Latin if homophones exist
    # Lower reduction_ratio = more merging = more homophonic
    v_reduction_amount = 1.0 - v_clustering.reduction_ratio
    l_reduction_amount = 1.0 - latin_reduction
    cluster_selectivity = (v_reduction_amount / l_reduction_amount
                           if l_reduction_amount > 0.001 else 1.0)
    print(f"    Cluster selectivity (Voynich/Latin reduction): "
          f"{cluster_selectivity:.2f}x")

    # ===================================================================
    # Gate
    # ===================================================================
    max_ref_ttr = max((r['ttr'] for r in ref_comparisons), default=0.0)
    gate_inflation = v_ttr > 1.3 * max_ref_ttr if max_ref_ttr > 0 else False
    gate_selectivity = cluster_selectivity >= 1.5
    gate_passed = gate_inflation and gate_selectivity

    if gate_passed:
        verdict = 'homophonic_inflation_confirmed'
    elif gate_selectivity and not gate_inflation:
        verdict = 'distributional_clusters_found_no_inflation'
    elif gate_inflation and not gate_selectivity:
        verdict = 'inflated_vocab_but_no_cluster_signal'
    else:
        verdict = 'no_homophonic_signal'

    print(f"\n  Gate: inflation={gate_inflation}  "
          f"selectivity={gate_selectivity}  passed={gate_passed}")
    print(f"  Verdict: {verdict}")

    # ===================================================================
    # Save
    # ===================================================================
    result = HomophoneTestResult(
        vocabulary_inflation=_convert(asdict(vocab_inflation)),
        voynich_clustering=_convert(asdict(v_clustering)),
        merged_decoding=_convert(asdict(merged_decoding)),
        latin_clustering=_convert(asdict(l_clustering)),
        latin_n_clusters=latin_n_clusters,
        latin_reduction_ratio=latin_reduction,
        cluster_selectivity=cluster_selectivity,
        gate_inflation=gate_inflation,
        gate_selectivity=gate_selectivity,
        gate_passed=gate_passed,
        verdict=verdict,
    )

    out = _convert(asdict(result))
    with open(_results_dir() / 'homophone_test.json', 'w') as f:
        json.dump(out, f, indent=2)

    print(f"\n  Results saved to results/homophone_test.json")
    return out
