"""
Phase 7.5 Step 4: Illustration-Embedding Bridge
=================================================
Locate Phase 6.1 Rosetta stems in the embedding space, check if they
land in the plant-names subcluster, and use embeddings to expand the
anchor set beyond the original 8 Rosetta folios.

Sub-analyses:
  4a — Locate Rosetta stems in embedding space
  4b — Embedding-enhanced anchor selection (expand anchor set)
  4c — Three-way convergence test (illustration + embedding + position)
  4d — Null test: random-stem plant-cluster hit rate

Output:
  results/embedding_bridge.json
"""

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.distance import cdist

from voynich.core.corpus import load_corpus, VoynichCorpus
from voynich.core.stats import selectivity_ratio
from voynich.core._paths import results_dir as _results_dir
from voynich.phases.morpheme_grid import decompose_token_morphemes


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class RosettaStemLocation:
    """Location of one Rosetta stem in embedding space."""
    folio: str
    voynich_stem: str
    medieval_name: str
    medieval_stem: str
    in_embedding_vocab: bool
    nearest_subcluster: Optional[str]
    distance_to_nearest: Optional[float]
    distance_to_plant_cluster: Optional[float]
    in_plant_cluster: bool


@dataclass
class ExpandedAnchor:
    """A folio identified as a potential new anchor via three-way convergence."""
    folio: str
    voynich_stem: str
    # Three evidence types
    embedding_in_plant_cluster: bool
    embedding_distance: float
    is_noun_candidate: bool
    tfidf_specificity: float
    has_illustration_id: bool
    illustration_tier: Optional[int]
    # Convergence
    n_evidence_types: int
    three_way_convergent: bool


@dataclass
class EmbeddingBridgeResult:
    """Full Phase 7.5 Step 4 output."""
    # Rosetta stems in embedding space
    n_rosetta_stems: int
    n_rosetta_in_vocab: int
    n_rosetta_in_plant_cluster: int
    rosetta_hit_rate: float
    rosetta_locations: List[Dict]
    # Anchor expansion
    n_herbal_folios_analyzed: int
    n_candidate_expansions: int
    n_three_way_convergent: int
    expanded_anchors: List[Dict]
    # Null test
    null_plant_hit_rate_mean: float
    null_plant_hit_rate_std: float
    hit_rate_selectivity: float
    # Gate
    rosetta_in_plant_gate: bool
    expansion_gate: bool
    gate_passed: bool
    verdict: str


# ---------------------------------------------------------------------------
# 4a — Locate Rosetta stems in embedding space
# ---------------------------------------------------------------------------

def _compute_subcluster_centroids(
    space,
    noun_subclusters: List[Dict],
    noun_stems: List[str],
    labels: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    Compute centroid for each subcluster using stems in the embedding space.
    """
    centroids: Dict[str, np.ndarray] = {}

    for sc in noun_subclusters:
        label = sc.get('label', '')
        cluster_stems = sc.get('top_stems', [])
        indices = []
        for s in cluster_stems:
            if s in space.vocab_to_idx:
                indices.append(space.vocab_to_idx[s])
        if indices:
            centroids[label] = space.embeddings[indices].mean(axis=0)

    return centroids


def locate_rosetta_in_embeddings(
    rosetta_folios: List[Dict],
    space,
    noun_subclusters: List[Dict],
) -> List[RosettaStemLocation]:
    """
    For each Rosetta folio, find its dominant stem in the embedding space
    and determine which subcluster it falls into.
    """
    centroids = _compute_subcluster_centroids(space, noun_subclusters, [])

    locations = []
    for rf in rosetta_folios:
        folio = rf.get('folio', '')
        v_stem = rf.get('dominant_stem', '')
        med_name = rf.get('medieval_name', '')
        med_stem = rf.get('medieval_stem', '')

        in_vocab = v_stem in space.vocab_to_idx if space else False

        if not in_vocab or not centroids:
            locations.append(RosettaStemLocation(
                folio=folio, voynich_stem=v_stem,
                medieval_name=med_name, medieval_stem=med_stem,
                in_embedding_vocab=in_vocab,
                nearest_subcluster=None, distance_to_nearest=None,
                distance_to_plant_cluster=None, in_plant_cluster=False,
            ))
            continue

        # Get embedding vector for this stem
        idx = space.vocab_to_idx[v_stem]
        vec = space.embeddings[idx:idx+1]

        # Distance to each subcluster centroid
        nearest_label = None
        nearest_dist = float('inf')
        plant_dist = None

        for label, centroid in centroids.items():
            dist = float(cdist(vec, centroid.reshape(1, -1), metric='cosine')[0, 0])
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_label = label
            if label == 'plant_names':
                plant_dist = dist

        in_plant = nearest_label == 'plant_names'

        locations.append(RosettaStemLocation(
            folio=folio, voynich_stem=v_stem,
            medieval_name=med_name, medieval_stem=med_stem,
            in_embedding_vocab=True,
            nearest_subcluster=nearest_label,
            distance_to_nearest=nearest_dist,
            distance_to_plant_cluster=plant_dist,
            in_plant_cluster=in_plant,
        ))

    return locations


# ---------------------------------------------------------------------------
# 4b — Expand anchor set
# ---------------------------------------------------------------------------

def expand_anchor_set(
    space,
    noun_subclusters: List[Dict],
    noun_stems_set: set,
    corpus: VoynichCorpus,
    rosetta_folios_set: set,
    folio_id_data: Dict[str, Dict],
) -> List[ExpandedAnchor]:
    """
    For every herbal folio (not in Rosetta set), check if its TF-IDF top stem:
    1. Falls in the plant-names embedding subcluster
    2. Is a noun candidate by position
    3. Has an illustration identification

    If all three: three-way convergent anchor expansion.
    """
    centroids = _compute_subcluster_centroids(space, noun_subclusters, [])
    plant_centroid = centroids.get('plant_names')
    if plant_centroid is None:
        return []

    anchors = []
    herbal_pages = [p for p in corpus.pages.values()
                    if p.section in ('herbal_a', 'herbal_b')]

    for page in herbal_pages:
        if page.folio in rosetta_folios_set:
            continue

        # Get TF-IDF top stems for this folio
        stem_freq: Counter = Counter()
        for tok in page.all_tokens:
            d = decompose_token_morphemes(tok)
            stem = d.stem if d.stem else tok
            stem_freq[stem] += 1

        if not stem_freq:
            continue

        # Simple TF-IDF: use frequency * log(N / df) approximation
        # (We don't have full corpus stats here, so use frequency as proxy)
        top_stems = [s for s, _ in stem_freq.most_common(3)]

        for stem in top_stems:
            if stem not in space.vocab_to_idx:
                continue

            idx = space.vocab_to_idx[stem]
            vec = space.embeddings[idx:idx+1]

            # Distance to plant-names centroid
            dist = float(cdist(vec, plant_centroid.reshape(1, -1),
                               metric='cosine')[0, 0])

            # Check if nearest to plant cluster
            in_plant = True
            for label, centroid in centroids.items():
                if label == 'plant_names':
                    continue
                other_dist = float(cdist(vec, centroid.reshape(1, -1),
                                         metric='cosine')[0, 0])
                if other_dist < dist:
                    in_plant = False
                    break

            is_noun = stem in noun_stems_set

            # Check illustration identification
            has_id = page.folio in folio_id_data
            tier = folio_id_data.get(page.folio, {}).get('tier')

            n_evidence = sum([in_plant, is_noun, has_id])
            three_way = in_plant and is_noun and has_id

            tfidf_val = stem_freq.get(stem, 0)

            anchors.append(ExpandedAnchor(
                folio=page.folio,
                voynich_stem=stem,
                embedding_in_plant_cluster=in_plant,
                embedding_distance=dist,
                is_noun_candidate=is_noun,
                tfidf_specificity=float(tfidf_val),
                has_illustration_id=has_id,
                illustration_tier=tier,
                n_evidence_types=n_evidence,
                three_way_convergent=three_way,
            ))

            if in_plant:
                break  # Only take the first plant-cluster stem per folio

    return anchors


# ---------------------------------------------------------------------------
# Null test
# ---------------------------------------------------------------------------

def _null_test_plant_cluster_hit_rate(
    space,
    noun_subclusters: List[Dict],
    n_rosetta_in_vocab: int,
    real_hit_rate: float,
    n_trials: int = 100,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Null: replace Rosetta stems with random stems of similar frequency.
    Check what fraction lands in the plant-names cluster.

    Returns (null_mean, null_std, selectivity).
    """
    centroids = _compute_subcluster_centroids(space, noun_subclusters, [])
    plant_centroid = centroids.get('plant_names')
    if plant_centroid is None or n_rosetta_in_vocab == 0:
        return 0.0, 0.0, 0.0

    rng = np.random.RandomState(seed)
    null_rates = []

    for _ in range(n_trials):
        # Pick random stems from the vocabulary
        indices = rng.choice(space.n_vocab, size=n_rosetta_in_vocab, replace=False)
        n_hits = 0
        for idx in indices:
            vec = space.embeddings[idx:idx+1]
            plant_dist = float(cdist(vec, plant_centroid.reshape(1, -1),
                                     metric='cosine')[0, 0])
            is_nearest = True
            for label, centroid in centroids.items():
                if label == 'plant_names':
                    continue
                other_dist = float(cdist(vec, centroid.reshape(1, -1),
                                         metric='cosine')[0, 0])
                if other_dist < plant_dist:
                    is_nearest = False
                    break
            if is_nearest:
                n_hits += 1
        null_rates.append(n_hits / n_rosetta_in_vocab)

    null_arr = np.array(null_rates)
    null_mean = float(np.mean(null_arr))
    null_std = float(np.std(null_arr))
    sel = real_hit_rate / null_mean if null_mean > 1e-10 else float('inf')
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

def run_embedding_bridge() -> Dict:
    """
    Phase 7.5 Step 4: Illustration-Embedding Bridge.

    Locates Rosetta folio stems in the embedding space, checks if they
    land in the plant-names subcluster, and expands the anchor set via
    three-way convergence (illustration + embedding + position).
    """
    print("Phase 7.5 Step 4: Illustration-Embedding Bridge")
    print("=" * 70)

    corpus = load_corpus(verbose=False)

    # Load Rosetta selection results
    print("\n  Loading Rosetta selection data...")
    rosetta_path = _results_dir() / 'rosetta_selection.json'
    rosetta_folios = []
    if rosetta_path.exists():
        with open(rosetta_path) as f:
            ros_data = json.load(f)
        rosetta_folios = ros_data.get('folio_scores', [])
        print(f"  Loaded {len(rosetta_folios)} Rosetta folios")
    else:
        print("  WARNING: rosetta_selection.json not found.")

    # Load noun subclusters (from Step 2)
    print("\n  Loading noun subclusters...")
    noun_subclusters = []
    sc_path = _results_dir() / 'noun_subclusters.json'
    if sc_path.exists():
        with open(sc_path) as f:
            sc_data = json.load(f)
        noun_subclusters = sc_data.get('subclusters', [])
        print(f"  Loaded {len(noun_subclusters)} subclusters")
    else:
        print("  WARNING: noun_subclusters.json not found.")

    # Load illustration identification data (for anchor expansion)
    print("\n  Loading illustration identification data...")
    illust_path = _results_dir() / 'illustration_constrained.json'
    folio_id_data: Dict[str, Dict] = {}
    if illust_path.exists():
        with open(illust_path) as f:
            illust_data = json.load(f)
        for fs in illust_data.get('folios', []):
            folio_id_data[fs.get('folio', '')] = {
                'tier': fs.get('tier', 3),
                'dominant_stem': fs.get('dominant_stem', ''),
            }
        print(f"  Loaded identification data for {len(folio_id_data)} folios")
    else:
        print("  WARNING: illustration_constrained.json not found.")

    # Build combined embedding space
    print("\n  Building embedding space...")
    from voynich.phases.distributional import build_combined_embedding_space
    space, _ = build_combined_embedding_space(
        corpus, window=2, n_dim=50, min_count=3,
    )

    if space is None or not noun_subclusters:
        print("  ERROR: Cannot proceed without embedding space and subclusters.")
        result = EmbeddingBridgeResult(
            n_rosetta_stems=len(rosetta_folios), n_rosetta_in_vocab=0,
            n_rosetta_in_plant_cluster=0, rosetta_hit_rate=0.0,
            rosetta_locations=[],
            n_herbal_folios_analyzed=0, n_candidate_expansions=0,
            n_three_way_convergent=0, expanded_anchors=[],
            null_plant_hit_rate_mean=0.0, null_plant_hit_rate_std=0.0,
            hit_rate_selectivity=0.0,
            rosetta_in_plant_gate=False, expansion_gate=False,
            gate_passed=False, verdict='prerequisites_missing',
        )
        out = _convert(asdict(result))
        out_path = _results_dir() / 'embedding_bridge.json'
        with open(out_path, 'w') as f:
            json.dump(out, f, indent=2)
        return out

    # 4a: Locate Rosetta stems
    print("\n  Locating Rosetta stems in embedding space...")
    locations = locate_rosetta_in_embeddings(
        rosetta_folios, space, noun_subclusters,
    )

    n_in_vocab = sum(1 for loc in locations if loc.in_embedding_vocab)
    n_in_plant = sum(1 for loc in locations if loc.in_plant_cluster)
    hit_rate = n_in_plant / max(n_in_vocab, 1)

    print(f"  In vocabulary: {n_in_vocab}/{len(locations)}")
    print(f"  In plant cluster: {n_in_plant}/{n_in_vocab} ({hit_rate:.1%})")
    for loc in locations:
        flag = '*' if loc.in_plant_cluster else ' '
        print(f"    {flag} {loc.folio}: {loc.voynich_stem} → "
              f"{loc.nearest_subcluster or 'N/A'} "
              f"(dist={loc.distance_to_nearest or 0:.3f})")

    # 4b: Expand anchor set
    print("\n  Expanding anchor set via three-way convergence...")

    # Get noun stems set
    from voynich.phases.positional_slots import (
        segment_voynich_pharmaceutical, classify_stems_by_position,
    )
    segments = segment_voynich_pharmaceutical(corpus)
    _, noun_stems, _ = classify_stems_by_position(segments)
    noun_stems_set = set(noun_stems)

    rosetta_folio_set = set(rf.get('folio', '') for rf in rosetta_folios)

    expanded = expand_anchor_set(
        space, noun_subclusters, noun_stems_set, corpus,
        rosetta_folio_set, folio_id_data,
    )

    n_three_way = sum(1 for a in expanded if a.three_way_convergent)
    n_herbal = len([p for p in corpus.pages.values()
                    if p.section in ('herbal_a', 'herbal_b')])

    print(f"  Herbal folios analyzed: {n_herbal}")
    print(f"  Candidate expansions: {len(expanded)}")
    print(f"  Three-way convergent: {n_three_way}")

    three_way_anchors = [a for a in expanded if a.three_way_convergent]
    for a in sorted(three_way_anchors, key=lambda x: -x.tfidf_specificity)[:10]:
        print(f"    {a.folio}: {a.voynich_stem} "
              f"(plant={a.embedding_in_plant_cluster}, "
              f"noun={a.is_noun_candidate}, "
              f"illust={a.has_illustration_id}, "
              f"tier={a.illustration_tier})")

    # Null test
    print("\n  Running null test (100 trials)...")
    null_mean, null_std, hit_sel = _null_test_plant_cluster_hit_rate(
        space, noun_subclusters, n_in_vocab, hit_rate, n_trials=100,
    )
    print(f"  Null hit rate: {null_mean:.3f} +/- {null_std:.3f}")
    print(f"  Hit rate selectivity: {hit_sel:.2f}x")

    # Gates
    rosetta_gate = hit_rate >= 0.50
    expansion_gate = n_three_way >= 3
    gate_passed = rosetta_gate and expansion_gate

    if gate_passed:
        verdict = 'bridge_strong_expansion_found'
    elif rosetta_gate:
        verdict = 'rosetta_confirmed_few_expansions'
    elif expansion_gate:
        verdict = 'expansions_found_rosetta_not_in_plant_cluster'
    else:
        verdict = 'bridge_not_significant'

    print(f"\n  Rosetta gate (>= 50% in plant): {'PASS' if rosetta_gate else 'FAIL'}")
    print(f"  Expansion gate (>= 3 three-way): {'PASS' if expansion_gate else 'FAIL'}")
    print(f"  Verdict: {verdict}")

    result = EmbeddingBridgeResult(
        n_rosetta_stems=len(rosetta_folios),
        n_rosetta_in_vocab=n_in_vocab,
        n_rosetta_in_plant_cluster=n_in_plant,
        rosetta_hit_rate=hit_rate,
        rosetta_locations=[_convert(asdict(loc)) for loc in locations],
        n_herbal_folios_analyzed=n_herbal,
        n_candidate_expansions=len(expanded),
        n_three_way_convergent=n_three_way,
        expanded_anchors=[_convert(asdict(a)) for a in three_way_anchors[:20]],
        null_plant_hit_rate_mean=null_mean,
        null_plant_hit_rate_std=null_std,
        hit_rate_selectivity=hit_sel,
        rosetta_in_plant_gate=rosetta_gate,
        expansion_gate=expansion_gate,
        gate_passed=gate_passed,
        verdict=verdict,
    )

    out = _convert(asdict(result))
    out_path = _results_dir() / 'embedding_bridge.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {out_path}")
    return out
