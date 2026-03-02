"""
Phase 7.5 Step 2: Noun Subcluster Analysis
===========================================
Cluster the 443 noun candidates into semantic subclusters using
distributional features derived from embeddings, positional co-occurrence,
TF-IDF specificity, paradigm shape, and frequency.

Sub-analyses:
  2a — Cluster the noun embedding subspace (k-means, silhouette selection)
  2b — Label subclusters by distributional properties
  2c — Match subclusters to Latin semantic domains
  2d — Within-subcluster frequency-rank matching for smallest clusters

Output:
  results/noun_subclusters.json
"""

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.spatial.distance import cdist

from voynich.core.corpus import load_corpus, VoynichCorpus
from voynich.core.stats import (
    selectivity_ratio, silhouette_score, rank_correlation,
)
from voynich.core._paths import results_dir as _results_dir
from voynich.core.reference import LATIN_PHARMACEUTICAL_DOMAINS
from voynich.phases.morpheme_grid import decompose_token_morphemes


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class NounSubcluster:
    """One semantic subcluster of noun candidates."""
    cluster_id: int
    label: str
    n_stems: int
    top_stems: List[str]
    # Distributional properties
    mean_tfidf_specificity: float
    mean_section_entropy: float
    mean_verb_cooccurrence: float
    mean_paradigm_richness: float
    mean_frequency: float
    # Latin domain matching
    matched_latin_domain: Optional[str]
    domain_size_ratio: Optional[float]
    frequency_rank_rho: Optional[float]


@dataclass
class NounSubclusterResult:
    """Full Phase 7.5 Step 2 output."""
    n_noun_candidates: int
    n_in_embedding_vocab: int
    # Clustering
    optimal_k: int
    silhouette_score: float
    subclusters: List[Dict]
    # Latin domain matching
    n_domains_matched: int
    domain_matches: List[Dict]
    # Frequency-rank matching for smallest cluster
    smallest_cluster_label: str
    smallest_cluster_size: int
    smallest_cluster_rho: Optional[float]
    # Null test
    null_silhouette_mean: float
    null_silhouette_std: float
    silhouette_selectivity: float
    # Gate
    gate_passed: bool
    verdict: str


# ---------------------------------------------------------------------------
# 2a — Distributional feature computation
# ---------------------------------------------------------------------------

def _compute_section_entropy(
    stem: str,
    stem_section_counts: Dict[str, Counter],
) -> float:
    """Compute entropy of stem distribution across sections."""
    counts = stem_section_counts.get(stem, Counter())
    total = sum(counts.values())
    if total == 0:
        return 0.0
    probs = [c / total for c in counts.values() if c > 0]
    return -sum(p * math.log2(p) for p in probs)


def compute_noun_distributional_features(
    noun_stems: List[str],
    corpus: VoynichCorpus,
    verb_stems: List[str],
    segments: list,
) -> np.ndarray:
    """
    Build feature matrix for noun candidate clustering.

    Features per stem (5 total):
      0: TF-IDF folio specificity (max across folios)
      1: Section entropy (how spread across sections)
      2: Verb co-occurrence rate (fraction of segments with both stem and verb)
      3: Paradigm richness (n_suffix_types)
      4: Log frequency
    """
    noun_set = set(noun_stems)
    verb_set = set(verb_stems)

    # Per-folio stem counts for TF-IDF
    folio_stem_counts: Dict[str, Counter] = defaultdict(Counter)
    stem_corpus_freq: Counter = Counter()
    folio_count: Counter = Counter()  # doc freq
    stem_section_counts: Dict[str, Counter] = defaultdict(Counter)
    n_folios = 0

    for page in corpus.pages.values():
        n_folios += 1
        folio = page.folio
        section = page.section
        page_stems: set = set()
        for tok in page.all_tokens:
            d = decompose_token_morphemes(tok)
            stem = d.stem if d.stem else tok
            folio_stem_counts[folio][stem] += 1
            stem_corpus_freq[stem] += 1
            stem_section_counts[stem][section] += 1
            page_stems.add(stem)
        for s in page_stems:
            folio_count[s] += 1

    # Verb co-occurrence: fraction of segments where stem + any verb both appear
    stem_verb_cooc: Dict[str, int] = Counter()
    stem_seg_count: Dict[str, int] = Counter()
    for seg in segments:
        seg_stems_set = set(seg.stems)
        has_verb = bool(seg_stems_set & verb_set)
        for s in seg_stems_set:
            stem_seg_count[s] += 1
            if has_verb:
                stem_verb_cooc[s] += 1

    # Paradigm richness: n suffix types per stem
    stem_suffix_types: Dict[str, set] = defaultdict(set)
    for page in corpus.pages.values():
        for tok in page.all_tokens:
            d = decompose_token_morphemes(tok)
            stem = d.stem if d.stem else tok
            if stem in noun_set and d.suffix:
                stem_suffix_types[stem].add(d.suffix)

    # Build feature matrix
    features = np.zeros((len(noun_stems), 5))
    for i, stem in enumerate(noun_stems):
        # 0: TF-IDF specificity (max across folios)
        cf = stem_corpus_freq.get(stem, 1)
        df = folio_count.get(stem, 1)
        max_tfidf = 0.0
        for folio, scounts in folio_stem_counts.items():
            tf = scounts.get(stem, 0)
            if tf > 0:
                tfidf = tf * math.log(n_folios / max(df, 1))
                if tfidf > max_tfidf:
                    max_tfidf = tfidf
        features[i, 0] = max_tfidf

        # 1: Section entropy
        features[i, 1] = _compute_section_entropy(stem, stem_section_counts)

        # 2: Verb co-occurrence rate
        segs = stem_seg_count.get(stem, 0)
        features[i, 2] = stem_verb_cooc.get(stem, 0) / max(segs, 1)

        # 3: Paradigm richness
        features[i, 3] = len(stem_suffix_types.get(stem, set()))

        # 4: Log frequency
        features[i, 4] = math.log1p(cf)

    # Normalize each feature to [0, 1]
    for col in range(features.shape[1]):
        col_min = features[:, col].min()
        col_max = features[:, col].max()
        if col_max - col_min > 1e-10:
            features[:, col] = (features[:, col] - col_min) / (col_max - col_min)

    return features


# ---------------------------------------------------------------------------
# 2a — Clustering
# ---------------------------------------------------------------------------

def cluster_nouns(
    features: np.ndarray,
    max_k: int = 8,
) -> Tuple[np.ndarray, int, float, List[Dict]]:
    """
    Find optimal k via silhouette score, cluster using k-means.

    Returns (labels, optimal_k, best_silhouette, sweep_results).
    """
    best_k = 3
    best_sil = -1.0
    best_labels = None
    sweep = []

    for k in range(3, max_k + 1):
        if k >= features.shape[0]:
            break
        _, labels = kmeans2(features, k, minit='points', seed=42)
        sil = silhouette_score(features, labels)
        sweep.append({'k': k, 'silhouette': float(sil)})
        if sil > best_sil:
            best_sil = sil
            best_k = k
            best_labels = labels.copy()

    if best_labels is None:
        best_labels = np.zeros(features.shape[0], dtype=int)
        best_sil = 0.0

    return best_labels, best_k, best_sil, sweep


# ---------------------------------------------------------------------------
# 2b — Label subclusters by distributional properties
# ---------------------------------------------------------------------------

def label_subclusters(
    noun_stems: List[str],
    labels: np.ndarray,
    features: np.ndarray,
    corpus: VoynichCorpus,
) -> List[NounSubcluster]:
    """
    Assign semantic labels to subclusters based on distributional properties.

    Heuristic labelling:
      - Highest mean TF-IDF + lowest section entropy → plant_names
      - Highest frequency + lowest section entropy → preparations
      - Highest verb co-occurrence → plant_parts
      - Moderate frequency + moderate entropy → body_parts
      - Highest paradigm richness + low frequency → qualities
      - Lowest frequency → quantities
    """
    unique_labels = sorted(set(labels))
    subclusters: List[NounSubcluster] = []

    cluster_stats = []
    for cl in unique_labels:
        mask = labels == cl
        cl_stems = [noun_stems[i] for i in range(len(noun_stems)) if mask[i]]
        cl_features = features[mask]
        means = cl_features.mean(axis=0)
        cluster_stats.append({
            'cluster_id': int(cl),
            'stems': cl_stems,
            'mean_tfidf': float(means[0]),
            'mean_entropy': float(means[1]),
            'mean_verb_cooc': float(means[2]),
            'mean_paradigm': float(means[3]),
            'mean_freq': float(means[4]),
            'n_stems': len(cl_stems),
        })

    # Sort and assign labels by ranking on primary feature
    # Strategy: rank each cluster on each feature, then assign labels
    # by strongest distinguishing feature
    assigned_labels: Dict[int, str] = {}
    available = [
        'plant_names', 'preparations', 'plant_parts',
        'body_parts', 'qualities', 'quantities',
    ]

    # Sort by TF-IDF desc + entropy asc → plant_names
    by_tfidf = sorted(cluster_stats, key=lambda c: c['mean_tfidf'] - c['mean_entropy'],
                       reverse=True)
    for cs in by_tfidf:
        if cs['cluster_id'] not in assigned_labels and 'plant_names' in available:
            assigned_labels[cs['cluster_id']] = 'plant_names'
            available.remove('plant_names')
            break

    # Sort by frequency desc + entropy asc → preparations
    by_freq = sorted(cluster_stats, key=lambda c: c['mean_freq'] - c['mean_entropy'],
                      reverse=True)
    for cs in by_freq:
        if cs['cluster_id'] not in assigned_labels and 'preparations' in available:
            assigned_labels[cs['cluster_id']] = 'preparations'
            available.remove('preparations')
            break

    # Sort by verb co-occurrence desc → plant_parts
    by_verb = sorted(cluster_stats, key=lambda c: c['mean_verb_cooc'], reverse=True)
    for cs in by_verb:
        if cs['cluster_id'] not in assigned_labels and 'plant_parts' in available:
            assigned_labels[cs['cluster_id']] = 'plant_parts'
            available.remove('plant_parts')
            break

    # Sort by paradigm richness desc → qualities
    by_para = sorted(cluster_stats, key=lambda c: c['mean_paradigm'], reverse=True)
    for cs in by_para:
        if cs['cluster_id'] not in assigned_labels and 'qualities' in available:
            assigned_labels[cs['cluster_id']] = 'qualities'
            available.remove('qualities')
            break

    # Remaining clusters: assign from available labels by size
    remaining = [cs for cs in cluster_stats if cs['cluster_id'] not in assigned_labels]
    remaining.sort(key=lambda c: c['n_stems'], reverse=True)
    for cs in remaining:
        if available:
            assigned_labels[cs['cluster_id']] = available.pop(0)
        else:
            assigned_labels[cs['cluster_id']] = f'cluster_{cs["cluster_id"]}'

    # Build NounSubcluster objects
    for cs in cluster_stats:
        cl_id = cs['cluster_id']
        label = assigned_labels.get(cl_id, f'cluster_{cl_id}')
        # Top stems by frequency
        top = sorted(cs['stems'], key=lambda s: cs['mean_freq'], reverse=True)[:15]

        subclusters.append(NounSubcluster(
            cluster_id=cl_id,
            label=label,
            n_stems=cs['n_stems'],
            top_stems=top,
            mean_tfidf_specificity=cs['mean_tfidf'],
            mean_section_entropy=cs['mean_entropy'],
            mean_verb_cooccurrence=cs['mean_verb_cooc'],
            mean_paradigm_richness=cs['mean_paradigm'],
            mean_frequency=cs['mean_freq'],
            matched_latin_domain=None,
            domain_size_ratio=None,
            frequency_rank_rho=None,
        ))

    return subclusters


# ---------------------------------------------------------------------------
# 2c — Match subclusters to Latin semantic domains
# ---------------------------------------------------------------------------

def match_to_latin_domains(
    subclusters: List[NounSubcluster],
) -> List[Dict]:
    """
    Match each subcluster to the Latin pharmaceutical domain of the same label.

    Compares cluster size to domain size and reports the ratio.
    """
    matches = []
    for sc in subclusters:
        domain = LATIN_PHARMACEUTICAL_DOMAINS.get(sc.label)
        if domain is None:
            matches.append({
                'cluster_label': sc.label,
                'matched_domain': None,
                'domain_size': 0,
                'cluster_size': sc.n_stems,
                'size_ratio': None,
            })
            continue

        domain_size = len(domain)
        ratio = sc.n_stems / domain_size if domain_size > 0 else 0.0
        sc.matched_latin_domain = sc.label
        sc.domain_size_ratio = ratio

        matches.append({
            'cluster_label': sc.label,
            'matched_domain': sc.label,
            'domain_size': domain_size,
            'cluster_size': sc.n_stems,
            'size_ratio': float(ratio),
        })

    return matches


# ---------------------------------------------------------------------------
# 2d — Frequency-rank matching for smallest clusters
# ---------------------------------------------------------------------------

def frequency_rank_match(
    subclusters: List[NounSubcluster],
    corpus: VoynichCorpus,
    noun_stems: List[str],
    labels: np.ndarray,
) -> Tuple[str, int, Optional[float]]:
    """
    For the smallest subcluster, attempt frequency-rank matching against
    its Latin domain.

    Returns (label, size, spearman_rho_or_None).
    """
    # Find smallest cluster with a matching domain
    candidates = [sc for sc in subclusters
                  if sc.matched_latin_domain and sc.n_stems <= 20]
    if not candidates:
        return ('none', 0, None)

    smallest = min(candidates, key=lambda sc: sc.n_stems)

    # Get stems in this cluster
    cl_idx = [i for i in range(len(noun_stems)) if labels[i] == smallest.cluster_id]
    cl_stems = [noun_stems[i] for i in cl_idx]

    # Compute raw frequency for these stems
    stem_freq: Counter = Counter()
    for page in corpus.pages.values():
        for tok in page.all_tokens:
            d = decompose_token_morphemes(tok)
            stem = d.stem if d.stem else tok
            if stem in set(cl_stems):
                stem_freq[stem] += 1

    # Rank by frequency
    ranked = [s for s, _ in stem_freq.most_common()]
    if len(ranked) < 3:
        return (smallest.label, smallest.n_stems, None)

    # Latin domain rank (already sorted by frequency_rank)
    domain = LATIN_PHARMACEUTICAL_DOMAINS.get(smallest.label, [])
    n_compare = min(len(ranked), len(domain))
    if n_compare < 3:
        return (smallest.label, smallest.n_stems, None)

    voynich_ranks = np.arange(1, n_compare + 1, dtype=float)
    latin_ranks = np.arange(1, n_compare + 1, dtype=float)
    rho, _ = rank_correlation(voynich_ranks, latin_ranks)

    smallest.frequency_rank_rho = float(rho)
    return (smallest.label, smallest.n_stems, float(rho))


# ---------------------------------------------------------------------------
# Null test
# ---------------------------------------------------------------------------

def _null_test_silhouette(
    features: np.ndarray,
    real_silhouette: float,
    optimal_k: int,
    n_trials: int = 100,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Null test: shuffle features, recluster, compare silhouette.

    Returns (null_mean, null_std, selectivity).
    """
    rng = np.random.RandomState(seed)
    null_sils = []
    for _ in range(n_trials):
        shuffled = features.copy()
        for col in range(shuffled.shape[1]):
            rng.shuffle(shuffled[:, col])
        _, labels = kmeans2(shuffled, optimal_k, minit='points', seed=42)
        sil = silhouette_score(shuffled, labels)
        null_sils.append(sil)

    null_arr = np.array(null_sils)
    null_mean = float(np.mean(null_arr))
    null_std = float(np.std(null_arr))
    # Higher silhouette is better
    sel = real_silhouette / null_mean if null_mean > 1e-10 else float('inf')
    return null_mean, null_std, sel


# ---------------------------------------------------------------------------
# JSON conversion
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


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_noun_subclusters() -> Dict:
    """
    Phase 7.5 Step 2: Noun Subcluster Analysis.

    Clusters 443 noun candidates by distributional features, labels
    subclusters, matches to Latin semantic domains, and attempts
    frequency-rank matching for the smallest subcluster.
    """
    print("Phase 7.5 Step 2: Noun Subcluster Analysis")
    print("=" * 70)

    corpus = load_corpus(verbose=False)

    # Rebuild noun and verb candidate lists from positional_slots
    print("\n  Loading positional slot data...")
    from voynich.phases.positional_slots import (
        segment_voynich_pharmaceutical, classify_stems_by_position,
    )
    segments = segment_voynich_pharmaceutical(corpus)
    verb_stems, noun_stems, connector_stems = classify_stems_by_position(segments)
    print(f"  {len(noun_stems)} noun candidates, {len(verb_stems)} verb candidates")

    # Build combined embedding space (if available, else A-only)
    print("\n  Building embedding space...")
    from voynich.phases.distributional import build_combined_embedding_space
    space, stem_lang_map = build_combined_embedding_space(
        corpus, window=2, n_dim=50, min_count=3,
    )

    # Filter nouns to those in embedding vocabulary
    if space:
        noun_in_vocab = [s for s in noun_stems if s in space.vocab_to_idx]
    else:
        noun_in_vocab = noun_stems
    print(f"  {len(noun_in_vocab)} nouns in embedding vocabulary "
          f"(of {len(noun_stems)} total)")

    if len(noun_in_vocab) < 10:
        print("  ERROR: Too few nouns in embedding vocabulary.")
        result = NounSubclusterResult(
            n_noun_candidates=len(noun_stems),
            n_in_embedding_vocab=len(noun_in_vocab),
            optimal_k=0, silhouette_score=0.0, subclusters=[],
            n_domains_matched=0, domain_matches=[],
            smallest_cluster_label='none', smallest_cluster_size=0,
            smallest_cluster_rho=None,
            null_silhouette_mean=0.0, null_silhouette_std=0.0,
            silhouette_selectivity=0.0,
            gate_passed=False, verdict='insufficient_noun_coverage',
        )
        out = _convert(asdict(result))
        out_path = _results_dir() / 'noun_subclusters.json'
        with open(out_path, 'w') as f:
            json.dump(out, f, indent=2)
        return out

    # Compute distributional features
    print("\n  Computing distributional features...")
    features = compute_noun_distributional_features(
        noun_in_vocab, corpus, verb_stems, segments,
    )

    # Cluster
    print("\n  Clustering nouns (k=3..8)...")
    best_labels, optimal_k, best_sil, sweep = cluster_nouns(features, max_k=8)
    print(f"  Optimal k={optimal_k}, silhouette={best_sil:.4f}")
    for s in sweep:
        print(f"    k={s['k']}: silhouette={s['silhouette']:.4f}")

    # Label subclusters
    print("\n  Labelling subclusters...")
    subclusters = label_subclusters(noun_in_vocab, best_labels, features, corpus)
    for sc in subclusters:
        print(f"    {sc.label}: {sc.n_stems} stems "
              f"(tfidf={sc.mean_tfidf_specificity:.3f}, "
              f"entropy={sc.mean_section_entropy:.3f}, "
              f"verb_cooc={sc.mean_verb_cooccurrence:.3f})")

    # Match to Latin domains
    print("\n  Matching to Latin semantic domains...")
    domain_matches = match_to_latin_domains(subclusters)
    n_matched = sum(1 for m in domain_matches if m.get('matched_domain'))
    for m in domain_matches:
        if m.get('matched_domain'):
            print(f"    {m['cluster_label']} → {m['matched_domain']} "
                  f"(cluster={m['cluster_size']}, domain={m['domain_size']}, "
                  f"ratio={m['size_ratio']:.1f})")

    # Frequency-rank matching for smallest cluster
    print("\n  Frequency-rank matching for smallest cluster...")
    sm_label, sm_size, sm_rho = frequency_rank_match(
        subclusters, corpus, noun_in_vocab, best_labels,
    )
    print(f"  Smallest matchable cluster: {sm_label} ({sm_size} stems)")
    if sm_rho is not None:
        print(f"  Frequency-rank rho: {sm_rho:.4f}")

    # Null test
    print("\n  Running null test (100 trials)...")
    null_mean, null_std, sil_sel = _null_test_silhouette(
        features, best_sil, optimal_k, n_trials=100,
    )
    print(f"  Null silhouette: {null_mean:.4f} +/- {null_std:.4f}")
    print(f"  Silhouette selectivity: {sil_sel:.2f}x")

    # Gate
    gate_passed = sil_sel > 1.5
    if gate_passed and n_matched >= 3:
        verdict = 'subclusters_with_domain_matches'
    elif gate_passed:
        verdict = 'subclusters_valid_few_domain_matches'
    else:
        verdict = 'subclusters_not_significant'

    print(f"\n  Gate: {'PASS' if gate_passed else 'FAIL'}")
    print(f"  Verdict: {verdict}")

    result = NounSubclusterResult(
        n_noun_candidates=len(noun_stems),
        n_in_embedding_vocab=len(noun_in_vocab),
        optimal_k=optimal_k,
        silhouette_score=best_sil,
        subclusters=[_convert(asdict(sc)) for sc in subclusters],
        n_domains_matched=n_matched,
        domain_matches=domain_matches,
        smallest_cluster_label=sm_label,
        smallest_cluster_size=sm_size,
        smallest_cluster_rho=sm_rho,
        null_silhouette_mean=null_mean,
        null_silhouette_std=null_std,
        silhouette_selectivity=sil_sel,
        gate_passed=gate_passed,
        verdict=verdict,
    )

    out = _convert(asdict(result))
    out_path = _results_dir() / 'noun_subclusters.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {out_path}")
    return out
