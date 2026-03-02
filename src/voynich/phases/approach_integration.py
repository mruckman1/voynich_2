"""
Phase 7: Approach Integration
===============================
Cross-validates Approaches 8 (distributional semantics) and 9 (positional
slot analysis) at multiple convergence points:

1. Do Approach 9 verb candidates cluster together in Approach 8 embedding space?
2. Do noun candidates cluster together?
3. Cohen's kappa between embedding-cluster labels and positional-slot labels
4. Do both approaches identify the same best language?

Output:
  results/approach_integration.json
"""

import json
import math
import random
from collections import Counter
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.distance import cdist

from voynich.core.corpus import load_corpus
from voynich.core.stats import (
    cosine_similarity, cohens_kappa, selectivity_ratio,
    adjusted_rand_index,
)
from voynich.core._paths import results_dir as _results_dir
from voynich.phases.morpheme_grid import decompose_token_morphemes


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ClusterCoherenceTest:
    """Test whether a set of stems clusters together in embedding space."""
    category: str
    n_stems_tested: int
    mean_pairwise_cosine: float
    random_baseline_cosine: float
    ratio: float
    coherent: bool


@dataclass
class IntegrationResult:
    """Cross-validation between Approaches 8 and 9."""
    # Approach 8 summary
    distributional_verdict: str
    best_distributional_language: str
    distributional_gate_passed: bool
    # Approach 9 summary
    positional_verdict: str
    positional_gate_passed: bool
    n_verb_candidates: int
    n_noun_candidates: int
    # Convergence tests
    verb_coherence: Dict
    noun_coherence: Dict
    slot_embedding_kappa: float
    # Joint null
    null_kappa_mean: float
    null_kappa_std: float
    joint_selectivity: float
    # Final
    approaches_converge: bool
    confidence_level: str
    verdict: str


# ---------------------------------------------------------------------------
# Convergence tests
# ---------------------------------------------------------------------------

def _load_embeddings() -> Optional[Dict]:
    """Load Voynich embedding space from distributional.json."""
    # We need the actual embedding vectors, not just the summary.
    # Since we saved only summary stats, we rebuild from the vocabulary.
    # In practice, this loads the full result and checks if embeddings
    # were saved. For now, return the summary.
    path = _results_dir() / 'distributional.json'
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _load_positional() -> Optional[Dict]:
    """Load positional slots results."""
    path = _results_dir() / 'positional_slots.json'
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _rebuild_embedding_space(corpus, language='B', window=2, n_dim=50, min_count=3):
    """Rebuild embedding space for convergence testing."""
    from voynich.phases.distributional import build_embedding_space
    tokens = corpus.get_tokens(language=language, paragraph_only=True)
    return build_embedding_space(tokens, f'voynich_{language}', window, n_dim, min_count)


def test_cluster_coherence(
    candidate_stems: List[str],
    space,
    category: str,
    n_random_trials: int = 100,
    seed: int = 42,
) -> ClusterCoherenceTest:
    """
    Test whether a set of stems clusters more tightly in embedding space
    than random sets of the same size.
    """
    if space is None or not candidate_stems:
        return ClusterCoherenceTest(
            category=category, n_stems_tested=0,
            mean_pairwise_cosine=0.0, random_baseline_cosine=0.0,
            ratio=0.0, coherent=False,
        )

    # Filter to stems in the embedding vocabulary
    in_vocab = [s for s in candidate_stems if s in space.vocab_to_idx]
    if len(in_vocab) < 2:
        return ClusterCoherenceTest(
            category=category, n_stems_tested=len(in_vocab),
            mean_pairwise_cosine=0.0, random_baseline_cosine=0.0,
            ratio=0.0, coherent=False,
        )

    # Mean pairwise cosine similarity
    indices = [space.vocab_to_idx[s] for s in in_vocab]
    emb = space.embeddings[indices]
    dists = cdist(emb, emb, metric='cosine')
    n = len(indices)
    if n > 1:
        triu = dists[np.triu_indices(n, k=1)]
        real_sim = 1.0 - float(np.mean(triu))
    else:
        real_sim = 1.0

    # Random baseline: sample sets of the same size
    rng = random.Random(seed)
    null_sims = []
    all_indices = list(range(space.n_vocab))
    for _ in range(n_random_trials):
        sample = rng.sample(all_indices, min(n, len(all_indices)))
        s_emb = space.embeddings[sample]
        s_dists = cdist(s_emb, s_emb, metric='cosine')
        if len(sample) > 1:
            s_triu = s_dists[np.triu_indices(len(sample), k=1)]
            null_sims.append(1.0 - float(np.mean(s_triu)))
        else:
            null_sims.append(0.0)

    baseline = float(np.mean(null_sims))
    ratio = real_sim / baseline if baseline > 1e-10 else float('inf')

    return ClusterCoherenceTest(
        category=category,
        n_stems_tested=len(in_vocab),
        mean_pairwise_cosine=real_sim,
        random_baseline_cosine=baseline,
        ratio=ratio,
        coherent=ratio > 1.2,
    )


def compute_slot_embedding_kappa(
    verb_stems: List[str],
    noun_stems: List[str],
    space,
    n_clusters: int = 3,
) -> float:
    """
    Cluster embedding space, compute kappa between cluster labels
    and positional-slot labels (verb/noun/other).
    """
    if space is None or space.n_vocab < n_clusters:
        return 0.0

    from scipy.cluster.vq import kmeans2
    k = min(n_clusters, space.n_vocab)
    _, cluster_labels = kmeans2(space.embeddings, k, minit='points', seed=42)

    # Build slot labels for each vocab item
    verb_set = set(verb_stems)
    noun_set = set(noun_stems)
    slot_labels = []
    for stem in space.vocab:
        if stem in verb_set:
            slot_labels.append('verb')
        elif stem in noun_set:
            slot_labels.append('noun')
        else:
            slot_labels.append('other')

    return cohens_kappa(cluster_labels, np.array(slot_labels))


def joint_null_test(
    verb_stems: List[str],
    noun_stems: List[str],
    space,
    real_kappa: float,
    n_trials: int = 100,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Joint null: shuffle slot labels, recompute kappa with embedding clusters.
    """
    if space is None or space.n_vocab < 3:
        return 0.0, 0.0, 0.0

    from scipy.cluster.vq import kmeans2
    k = min(3, space.n_vocab)
    _, cluster_labels = kmeans2(space.embeddings, k, minit='points', seed=42)

    verb_set = set(verb_stems)
    noun_set = set(noun_stems)
    slot_labels = np.array([
        'verb' if s in verb_set else ('noun' if s in noun_set else 'other')
        for s in space.vocab
    ])

    rng = random.Random(seed)
    null_kappas = []
    for _ in range(n_trials):
        shuffled = slot_labels.copy()
        rng.shuffle(shuffled)
        k_val = cohens_kappa(cluster_labels, shuffled)
        null_kappas.append(k_val)

    null_arr = np.array(null_kappas)
    null_mean = float(np.mean(null_arr))
    null_std = float(np.std(null_arr))
    if null_std > 1e-10:
        z = (real_kappa - null_mean) / null_std
        sel = max(0, z / 1.5)
    else:
        sel = float('inf') if real_kappa > null_mean else 0.0
    return null_mean, null_std, sel


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

def _print_results(result: IntegrationResult):
    """Print formatted results to console."""
    print("\n" + "=" * 70)
    print("PHASE 7: APPROACH INTEGRATION")
    print("=" * 70)

    print("\n--- Approach 8 (Distributional) Summary ---")
    print(f"  Verdict:        {result.distributional_verdict}")
    print(f"  Best language:  {result.best_distributional_language}")
    print(f"  Gate passed:    {'YES' if result.distributional_gate_passed else 'NO'}")

    print("\n--- Approach 9 (Positional) Summary ---")
    print(f"  Verdict:        {result.positional_verdict}")
    print(f"  Gate passed:    {'YES' if result.positional_gate_passed else 'NO'}")
    print(f"  Verb candidates:  {result.n_verb_candidates}")
    print(f"  Noun candidates:  {result.n_noun_candidates}")

    print("\n--- Convergence Tests ---")
    vc = result.verb_coherence
    print(f"  Verb cluster coherence:    sim={vc.get('mean_pairwise_cosine', 0):.3f}  "
          f"baseline={vc.get('random_baseline_cosine', 0):.3f}  "
          f"ratio={vc.get('ratio', 0):.2f}  "
          f"{'COHERENT' if vc.get('coherent') else 'not coherent'}")
    nc = result.noun_coherence
    print(f"  Noun cluster coherence:    sim={nc.get('mean_pairwise_cosine', 0):.3f}  "
          f"baseline={nc.get('random_baseline_cosine', 0):.3f}  "
          f"ratio={nc.get('ratio', 0):.2f}  "
          f"{'COHERENT' if nc.get('coherent') else 'not coherent'}")
    print(f"  Slot-embedding kappa:      {result.slot_embedding_kappa:.3f}")

    print("\n--- Joint Null Test ---")
    print(f"  Null kappa:       μ={result.null_kappa_mean:.4f} σ={result.null_kappa_std:.4f}")
    print(f"  Joint selectivity: {result.joint_selectivity:.2f}×")

    print(f"\n  Approaches converge: {'YES' if result.approaches_converge else 'NO'}")
    print(f"  Confidence level:    {result.confidence_level}")
    print(f"  Verdict:             {result.verdict}")


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


def run_approach_integration() -> Dict:
    """
    Run cross-validation between Approaches 8 and 9.

    1. Load results from both approaches
    2. Rebuild embedding space for convergence testing
    3. Test verb/noun cluster coherence
    4. Compute slot-embedding kappa
    5. Joint null test
    6. Determine convergence and confidence
    7. Save results
    """
    print("Loading approach results...")
    dist_data = _load_embeddings()
    pos_data = _load_positional()

    if dist_data is None:
        print("  WARNING: distributional.json not found. Run 'voynich embeddings' first.")
    if pos_data is None:
        print("  WARNING: positional_slots.json not found. Run 'voynich slots' first.")

    # Extract summaries
    dist_verdict = dist_data.get('verdict', 'not_run') if dist_data else 'not_run'
    dist_best_lang = dist_data.get('best_procrustes_language', 'none') if dist_data else 'none'
    dist_gate = dist_data.get('gate_passed', False) if dist_data else False
    pos_verdict = pos_data.get('verdict', 'not_run') if pos_data else 'not_run'
    pos_gate = pos_data.get('gate_passed', False) if pos_data else False

    # Extract verb/noun candidates from Approach 9
    verb_stems = []
    noun_stems = []
    if pos_data:
        verb_stems = [v.get('stem', '') for v in pos_data.get('verb_candidates', [])]
        noun_stems = [ing.get('stem', '') for ing in pos_data.get('ingredient_candidates', [])]

    n_verb = len(verb_stems)
    n_noun = len(noun_stems)

    # Rebuild embedding space for convergence testing
    print("Rebuilding embedding space for convergence tests...")
    corpus = load_corpus(verbose=False)
    space = _rebuild_embedding_space(corpus, language='B')
    if space is None:
        # Try Language A
        space = _rebuild_embedding_space(corpus, language='A')

    # Test verb cluster coherence
    print("Testing verb cluster coherence...")
    verb_coherence = test_cluster_coherence(verb_stems, space, 'verb')

    # Test noun cluster coherence
    print("Testing noun cluster coherence...")
    noun_coherence = test_cluster_coherence(noun_stems, space, 'noun')

    # Slot-embedding kappa
    print("Computing slot-embedding kappa...")
    kappa = compute_slot_embedding_kappa(verb_stems, noun_stems, space)

    # Joint null test
    print("Running joint null test (100 trials)...")
    null_mean, null_std, joint_sel = joint_null_test(
        verb_stems, noun_stems, space, kappa,
    )

    # Determine convergence
    both_gates = dist_gate and pos_gate
    verb_noun_coherent = verb_coherence.coherent or noun_coherence.coherent
    kappa_significant = joint_sel > 1.0

    if both_gates and verb_noun_coherent and kappa_significant:
        approaches_converge = True
        confidence = 'high'
        verdict = 'convergent_evidence_both_approaches'
    elif both_gates and (verb_noun_coherent or kappa_significant):
        approaches_converge = True
        confidence = 'medium'
        verdict = 'partial_convergence'
    elif dist_gate or pos_gate:
        approaches_converge = False
        confidence = 'low'
        verdict = 'single_approach_positive'
    else:
        approaches_converge = False
        confidence = 'none'
        verdict = 'no_significant_signal'

    result = IntegrationResult(
        distributional_verdict=dist_verdict,
        best_distributional_language=dist_best_lang,
        distributional_gate_passed=dist_gate,
        positional_verdict=pos_verdict,
        positional_gate_passed=pos_gate,
        n_verb_candidates=n_verb,
        n_noun_candidates=n_noun,
        verb_coherence=_convert(asdict(verb_coherence)),
        noun_coherence=_convert(asdict(noun_coherence)),
        slot_embedding_kappa=kappa,
        null_kappa_mean=null_mean,
        null_kappa_std=null_std,
        joint_selectivity=joint_sel,
        approaches_converge=approaches_converge,
        confidence_level=confidence,
        verdict=verdict,
    )

    _print_results(result)

    out = _convert(asdict(result))
    out_path = _results_dir() / 'approach_integration.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return out
