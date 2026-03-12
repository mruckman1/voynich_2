"""
Phase 44 – Track B: Stochastic Block Model on Co-occurrence Graph
==================================================================
Build a multi-layer co-occurrence graph over EVA characters, fit a
degree-corrected SBM (or spectral clustering fallback), and discover
latent sign categories.  Compare communities against stroke-feature
clusters and sign families.

Dependency chain:
    combined_refine.json       (Phase 15 assignment)
    bootstrap_loop.json        (Phase 30 confirmed triples)
        -> sbm_graph.json          (Step 44B.1)
        -> sbm_communities.json    (Step 44B.2)
        -> sbm_comparison.json     (Step 44B.3)
        -> sbm_predictions.json    (Step 44B.4)
        -> sbm_validation.json     (Step 44B.5)
"""

import json
import os
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    load_corpus,
    tokenize_eva_chars,
)
from voynich.core.reference import (
    EVA_VISUAL_COMPONENTS,
    PHONEME_PLACE_MAP,
    PHONEME_NUCLEUS_MAP,
    build_triple_phoneme_hypotheses,
)


# ---------------------------------------------------------------------------
# JSON helpers
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
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if v != v else v
    if isinstance(obj, np.ndarray):
        return _convert(obj.tolist())
    if isinstance(obj, (bool, int, float, str, type(None))):
        return obj
    return str(obj)


def _safe_load(path: str) -> Dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class GraphResult:
    n_nodes: int
    node_labels: List[str]
    n_layers: int
    layer_names: List[str]
    edges_per_layer: List[int]
    mean_degree_per_layer: List[float]
    runtime_seconds: float


@dataclass
class CommunityResult:
    method: str
    n_communities: int
    communities: Dict[str, int]  # EVA char -> community_id
    modularity: float
    community_sizes: List[int]
    best_k: int
    silhouette_score: float
    runtime_seconds: float


@dataclass
class ComparisonResult:
    ari_vs_stroke: float
    nmi_vs_stroke: float
    ari_vs_family: float
    nmi_vs_family: float
    stroke_interpretation: str
    family_interpretation: str
    overall_interpretation: str
    confusion_stroke: List[List[int]]
    confusion_family: List[List[int]]
    gate_passed: bool
    runtime_seconds: float


@dataclass
class PredictionEntry:
    triple_key: str
    community: int
    predicted_onset: str
    confidence: float
    same_community_confirmed: List[str]


@dataclass
class PredictionResult:
    n_unconfirmed: int
    predictions: List[Dict]
    n_high_confidence: int
    runtime_seconds: float


@dataclass
class SBMValidationResult:
    split_half_ari: float
    prediction_stability: float
    community_count_half1: int
    community_count_half2: int
    gate_passed: bool
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Step 44B.1 – Multi-Layer Co-occurrence Graph
# ---------------------------------------------------------------------------

def _build_cooccurrence_layers(
    corpus_tokens: List[str],
    eva_chars: List[str],
    eva_to_idx: Dict[str, int],
    consecutive_pairs: List[Tuple[str, str]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build 4 co-occurrence matrices from the EVA corpus."""
    n = len(eva_chars)

    L1 = np.zeros((n, n), dtype=np.float64)  # adjacent within token
    L2 = np.zeros((n, n), dtype=np.float64)  # same-word co-occurrence
    L3 = np.zeros((n, n), dtype=np.float64)  # positional substitutability
    L4 = np.zeros((n, n), dtype=np.float64)  # cross-word transitions

    # Position counts for L3
    position_counts: Dict[int, Counter] = defaultdict(Counter)

    for token in corpus_tokens:
        chars = tokenize_eva_chars(token)
        idxs = [eva_to_idx[c] for c in chars if c in eva_to_idx]

        # L1: adjacent pairs within token
        for i in range(len(idxs) - 1):
            L1[idxs[i], idxs[i + 1]] += 1

        # L2: all pairs within token (symmetric)
        for i in range(len(idxs)):
            for j in range(i + 1, len(idxs)):
                L2[idxs[i], idxs[j]] += 1
                L2[idxs[j], idxs[i]] += 1

        # Position tracking for L3
        for pos, idx in enumerate(idxs[:5]):
            position_counts[pos][idx] += 1

    # L3: positional substitutability
    for pos in position_counts:
        chars_at_pos = list(position_counts[pos].keys())
        for i in range(len(chars_at_pos)):
            for j in range(i + 1, len(chars_at_pos)):
                ci, cj = chars_at_pos[i], chars_at_pos[j]
                score = min(position_counts[pos][ci], position_counts[pos][cj])
                L3[ci, cj] += score
                L3[cj, ci] += score

    # L4: cross-word transitions
    for last_token, first_token in consecutive_pairs:
        last_chars = tokenize_eva_chars(last_token)
        first_chars = tokenize_eva_chars(first_token)
        if last_chars and first_chars:
            li = eva_to_idx.get(last_chars[-1])
            fi = eva_to_idx.get(first_chars[0])
            if li is not None and fi is not None:
                L4[li, fi] += 1

    return L1, L2, L3, L4


def run_sbm_graph() -> None:
    """Step 44B.1: Build multi-layer co-occurrence graph."""
    t0 = time.time()
    print("=" * 70)
    print("STEP 44B.1: Multi-Layer Co-occurrence Graph")
    print("=" * 70)

    rd = _results_dir()

    # EVA character set (from EVA_VISUAL_COMPONENTS)
    eva_chars = sorted(EVA_VISUAL_COMPONENTS.keys())
    eva_to_idx = {c: i for i, c in enumerate(eva_chars)}
    n = len(eva_chars)
    print(f"\n  {n} EVA characters")

    # Load corpus
    print("  Loading corpus ...")
    corpus = load_corpus(verbose=False)
    all_tokens = []
    consecutive_pairs = []
    for _fol, page in corpus.pages.items():
        tokens = page.all_tokens
        all_tokens.extend(tokens)
        for i in range(len(tokens) - 1):
            consecutive_pairs.append((tokens[i], tokens[i + 1]))

    print(f"  {len(all_tokens)} tokens, {len(consecutive_pairs)} consecutive pairs")

    # Build layers
    print("  Building co-occurrence layers ...")
    L1, L2, L3, L4 = _build_cooccurrence_layers(
        all_tokens, eva_chars, eva_to_idx, consecutive_pairs,
    )

    layer_names = ['adjacent', 'token_cooccurrence', 'positional', 'cross_word']
    layers = [L1, L2, L3, L4]
    edges_per_layer = [int(np.count_nonzero(L)) for L in layers]
    mean_degree = [float(np.sum(L > 0, axis=1).mean()) for L in layers]

    result = GraphResult(
        n_nodes=n,
        node_labels=eva_chars,
        n_layers=4,
        layer_names=layer_names,
        edges_per_layer=edges_per_layer,
        mean_degree_per_layer=[round(d, 2) for d in mean_degree],
        runtime_seconds=round(time.time() - t0, 2),
    )

    # Save graph + matrices
    out_data = {
        'summary': _convert(asdict(result)),
        'matrices': {
            name: _convert(L.tolist())
            for name, L in zip(layer_names, layers)
        },
    }
    out_path = os.path.join(rd, 'sbm_graph.json')
    with open(out_path, 'w') as f:
        json.dump(out_data, f, indent=2)

    for name, edges, deg in zip(layer_names, edges_per_layer, mean_degree):
        print(f"  {name}: {edges} edges, mean degree {deg:.1f}")
    print(f"  Saved -> {out_path}")
    print(f"\n  Step 44B.1 completed in {time.time() - t0:.1f}s")


# ---------------------------------------------------------------------------
# Step 44B.2 – SBM / Spectral Fitting
# ---------------------------------------------------------------------------

def _compute_modularity(adj: np.ndarray, labels: np.ndarray) -> float:
    """Compute Newman modularity Q for undirected graph."""
    m = adj.sum() / 2.0
    if m == 0:
        return 0.0
    n = len(labels)
    Q = 0.0
    for i in range(n):
        for j in range(n):
            if labels[i] == labels[j]:
                ki = adj[i].sum()
                kj = adj[j].sum()
                Q += adj[i, j] - ki * kj / (2 * m)
    return Q / (2 * m)


def run_sbm_fit() -> None:
    """Step 44B.2: Fit SBM / spectral clustering."""
    t0 = time.time()
    print("=" * 70)
    print("STEP 44B.2: SBM / Spectral Clustering")
    print("=" * 70)

    rd = _results_dir()

    # Load graph
    graph_data = _safe_load(os.path.join(rd, 'sbm_graph.json'))
    if not graph_data:
        print("  [SKIP] sbm_graph.json not found")
        return

    matrices = graph_data.get('matrices', {})
    summary = graph_data.get('summary', {})
    node_labels = summary.get('node_labels', [])
    n = len(node_labels)

    # Combine matrices (weighted sum)
    combined = np.zeros((n, n), dtype=np.float64)
    weights = {'adjacent': 2.0, 'token_cooccurrence': 1.0,
               'positional': 1.5, 'cross_word': 1.0}
    for name, w in weights.items():
        if name in matrices:
            mat = np.array(matrices[name], dtype=np.float64)
            combined += w * mat

    # Symmetrize
    combined = (combined + combined.T) / 2.0

    # Normalize to [0, 1]
    max_val = combined.max()
    if max_val > 0:
        combined_norm = combined / max_val
    else:
        combined_norm = combined

    # Try graph-tool first
    method = 'spectral_clustering'
    gt_communities = None
    try:
        import graph_tool.all as gt
        print("  graph-tool available, fitting nested SBM ...")
        g = gt.Graph(directed=False)
        g.add_vertex(n)
        weight_prop = g.new_edge_property("double")
        for i in range(n):
            for j in range(i + 1, n):
                if combined[i, j] > 0:
                    e = g.add_edge(i, j)
                    weight_prop[e] = combined[i, j]
        state = gt.minimize_nested_blockmodel_dl(
            g, state_args=dict(recs=[weight_prop], rec_types=["real-exponential"]),
        )
        levels = state.get_levels()
        gt_communities = np.array(levels[0].get_blocks().a)
        method = 'graph_tool_sbm'
        print(f"  graph-tool SBM: {len(set(gt_communities))} communities")
    except (ImportError, Exception) as e:
        print(f"  graph-tool not available ({e}), using spectral clustering")

    # Spectral clustering
    print("  Running spectral clustering ...")
    from sklearn.cluster import SpectralClustering
    from sklearn.metrics import silhouette_score as sil_score

    best_k = 5
    best_sil = -1.0
    best_labels = None

    for k in range(3, 13):
        try:
            sc = SpectralClustering(
                n_clusters=k, affinity='precomputed',
                random_state=42, n_init=10,
            )
            labels = sc.fit_predict(combined_norm + 1e-8)
            # Silhouette on the distance matrix
            dist = 1.0 - combined_norm
            np.fill_diagonal(dist, 0)
            sil = sil_score(dist, labels, metric='precomputed')
            if sil > best_sil:
                best_sil = sil
                best_k = k
                best_labels = labels
        except Exception:
            continue

    if best_labels is None:
        best_labels = np.zeros(n, dtype=int)
        best_k = 1

    # Use graph-tool result if available, else spectral
    final_labels = gt_communities if gt_communities is not None else best_labels
    final_method = method

    # Communities dict
    communities: Dict[str, int] = {}
    for i, char in enumerate(node_labels):
        communities[char] = int(final_labels[i])

    n_communities = len(set(final_labels))
    community_sizes = [int(np.sum(final_labels == c))
                       for c in sorted(set(final_labels))]
    modularity = _compute_modularity(combined, final_labels)

    result = CommunityResult(
        method=final_method,
        n_communities=n_communities,
        communities=communities,
        modularity=round(modularity, 4),
        community_sizes=community_sizes,
        best_k=best_k,
        silhouette_score=round(best_sil, 4),
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'sbm_communities.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2)

    print(f"\n  Method: {final_method}")
    print(f"  Communities: {n_communities}, sizes: {community_sizes}")
    print(f"  Modularity: {modularity:.4f}, Silhouette: {best_sil:.4f}")
    print(f"  Saved -> {out_path}")
    print(f"\n  Step 44B.2 completed in {time.time() - t0:.1f}s")


# ---------------------------------------------------------------------------
# Step 44B.3 – Community Comparison
# ---------------------------------------------------------------------------

def run_sbm_compare() -> None:
    """Step 44B.3: Compare SBM communities vs stroke features and sign families."""
    t0 = time.time()
    print("=" * 70)
    print("STEP 44B.3: Community Comparison")
    print("=" * 70)

    rd = _results_dir()

    # Load communities
    comm_data = _safe_load(os.path.join(rd, 'sbm_communities.json'))
    if not comm_data:
        print("  [SKIP] sbm_communities.json not found")
        return

    communities = comm_data.get('communities', {})
    node_labels = sorted(communities.keys())

    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    # Build reference labellings
    eva_to_triple = build_eva_to_triple_lookup()

    # Stroke-triple labels: group by triple_key
    stroke_labels = []
    sbm_labels = []
    for char in node_labels:
        triple = eva_to_triple.get(char, 'unknown')
        stroke_labels.append(triple)
        sbm_labels.append(communities.get(char, -1))

    # Family labels: group by glyph_class
    family_labels = []
    for char in node_labels:
        comp = EVA_VISUAL_COMPONENTS.get(char, {})
        family_labels.append(comp.get('glyph_class', 'unknown'))

    # Compute ARI and NMI
    ari_stroke = adjusted_rand_score(stroke_labels, sbm_labels)
    nmi_stroke = normalized_mutual_info_score(stroke_labels, sbm_labels)
    ari_family = adjusted_rand_score(family_labels, sbm_labels)
    nmi_family = normalized_mutual_info_score(family_labels, sbm_labels)

    print(f"\n  ARI(SBM, stroke triples): {ari_stroke:.4f}")
    print(f"  NMI(SBM, stroke triples): {nmi_stroke:.4f}")
    print(f"  ARI(SBM, sign families):  {ari_family:.4f}")
    print(f"  NMI(SBM, sign families):  {nmi_family:.4f}")

    # Interpretation
    if ari_stroke > 0.5 and ari_family > 0.5:
        stroke_interp = "STRONG_CONVERGENCE"
        family_interp = "STRONG_CONVERGENCE"
        overall = "Visual and distributional structure agree — stroke model supported"
    elif ari_stroke > 0.5:
        stroke_interp = "STRONG_CONVERGENCE"
        family_interp = "WEAK" if ari_family > 0.3 else "NO_CONVERGENCE"
        overall = "Distributional agrees with fine-grained strokes, not just families"
    elif ari_family > 0.5:
        stroke_interp = "WEAK" if ari_stroke > 0.3 else "NO_CONVERGENCE"
        family_interp = "STRONG_CONVERGENCE"
        overall = "Distributional follows visual families, not fine-grained strokes"
    elif ari_stroke > 0.3 or ari_family > 0.3:
        stroke_interp = "MODERATE" if ari_stroke > 0.3 else "NO_CONVERGENCE"
        family_interp = "MODERATE" if ari_family > 0.3 else "NO_CONVERGENCE"
        overall = "Moderate convergence — partial agreement"
    else:
        stroke_interp = "NO_CONVERGENCE"
        family_interp = "NO_CONVERGENCE"
        overall = "SBM finds novel structure unrelated to visual features"

    # Confusion matrices
    from collections import Counter as Ctr

    def _confusion(labels_a, labels_b):
        classes_a = sorted(set(labels_a))
        classes_b = sorted(set(labels_b))
        mat = [[0] * len(classes_b) for _ in classes_a]
        a_map = {c: i for i, c in enumerate(classes_a)}
        b_map = {c: i for i, c in enumerate(classes_b)}
        for la, lb in zip(labels_a, labels_b):
            mat[a_map[la]][b_map[lb]] += 1
        return mat

    conf_stroke = _confusion(stroke_labels, sbm_labels)
    conf_family = _confusion(family_labels, sbm_labels)

    gate = ari_stroke > 0.3 or ari_family > 0.3

    result = ComparisonResult(
        ari_vs_stroke=round(ari_stroke, 4),
        nmi_vs_stroke=round(nmi_stroke, 4),
        ari_vs_family=round(ari_family, 4),
        nmi_vs_family=round(nmi_family, 4),
        stroke_interpretation=stroke_interp,
        family_interpretation=family_interp,
        overall_interpretation=overall,
        confusion_stroke=conf_stroke,
        confusion_family=conf_family,
        gate_passed=gate,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'sbm_comparison.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2)

    print(f"\n  Interpretation: {overall}")
    print(f"  Gate: {'PASS' if gate else 'FAIL'}")
    print(f"  Saved -> {out_path}")
    print(f"\n  Step 44B.3 completed in {time.time() - t0:.1f}s")


# ---------------------------------------------------------------------------
# Step 44B.4 – Unconfirmed Triple Prediction
# ---------------------------------------------------------------------------

def run_sbm_predict() -> None:
    """Step 44B.4: Predict consonant class for unconfirmed triples."""
    t0 = time.time()
    print("=" * 70)
    print("STEP 44B.4: SBM Predictions for Unconfirmed Triples")
    print("=" * 70)

    rd = _results_dir()
    eva_to_triple = build_eva_to_triple_lookup()

    # Load communities
    comm_data = _safe_load(os.path.join(rd, 'sbm_communities.json'))
    if not comm_data:
        print("  [SKIP] sbm_communities.json not found")
        return
    communities = comm_data.get('communities', {})

    # Load confirmed triples
    boot_data = _safe_load(os.path.join(rd, 'bootstrap_loop.json'))
    refine_data = _safe_load(os.path.join(rd, 'combined_refine.json'))
    p15_assignment = refine_data.get('best_assignment', {})
    final_assignment = boot_data.get('final_assignment', p15_assignment)
    confirmed_list = set(boot_data.get('confirmed_triples', []))

    # Build triple -> EVA chars mapping
    triple_to_chars: Dict[str, List[str]] = defaultdict(list)
    for char, triple_key in eva_to_triple.items():
        triple_to_chars[triple_key].append(char)

    hypotheses = build_triple_phoneme_hypotheses('latin')
    all_triples = sorted(hypotheses.keys())
    free_triples = [t for t in all_triples if t not in confirmed_list]

    predictions = []
    for t_key in free_triples:
        chars = triple_to_chars.get(t_key, [])
        if not chars:
            continue

        # Get community assignments for this triple's chars
        char_comms = [communities.get(c, -1) for c in chars]
        if not char_comms:
            continue

        # Majority community
        comm_counter = Counter(char_comms)
        majority_comm = comm_counter.most_common(1)[0][0]

        # Find confirmed triples in same community
        same_comm_confirmed = []
        for t_c in confirmed_list:
            c_chars = triple_to_chars.get(t_c, [])
            c_comms = [communities.get(c, -1) for c in c_chars]
            if majority_comm in c_comms:
                syl = final_assignment.get(t_c, '?')
                same_comm_confirmed.append(f"{t_c}={syl}")

        # Predict onset from confirmed triples in same community
        onsets = set()
        for t_c in confirmed_list:
            c_chars = triple_to_chars.get(t_c, [])
            c_comms = [communities.get(c, -1) for c in c_chars]
            if majority_comm in c_comms:
                syl = final_assignment.get(t_c, '')
                if len(syl) >= 2:
                    onsets.add(syl[0])
                elif len(syl) == 1:
                    onsets.add(syl)

        predicted_onset = ','.join(sorted(onsets)) if onsets else '?'
        confidence = comm_counter.most_common(1)[0][1] / len(char_comms) if char_comms else 0.0

        predictions.append({
            'triple_key': t_key,
            'community': majority_comm,
            'predicted_onset': predicted_onset,
            'confidence': round(confidence, 3),
            'same_community_confirmed': same_comm_confirmed,
            'current_assignment': final_assignment.get(t_key, '?'),
        })

    n_high_conf = sum(1 for p in predictions if p['confidence'] > 0.8)

    result = PredictionResult(
        n_unconfirmed=len(free_triples),
        predictions=predictions,
        n_high_confidence=n_high_conf,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'sbm_predictions.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2)

    print(f"\n  {len(predictions)} predictions for {len(free_triples)} unconfirmed triples")
    print(f"  {n_high_conf} high-confidence (>0.8)")
    for p in predictions[:5]:
        print(f"    {p['triple_key']} -> onset={p['predicted_onset']} "
              f"(conf={p['confidence']:.2f}, current={p['current_assignment']})")
    print(f"  Saved -> {out_path}")
    print(f"\n  Step 44B.4 completed in {time.time() - t0:.1f}s")


# ---------------------------------------------------------------------------
# Step 44B.5 – Held-Out Validation
# ---------------------------------------------------------------------------

def run_sbm_validate() -> None:
    """Step 44B.5: Split-half validation of SBM communities."""
    t0 = time.time()
    print("=" * 70)
    print("STEP 44B.5: SBM Split-Half Validation")
    print("=" * 70)

    rd = _results_dir()

    # Load graph data
    graph_data = _safe_load(os.path.join(rd, 'sbm_graph.json'))
    if not graph_data:
        print("  [SKIP] sbm_graph.json not found")
        return

    summary = graph_data.get('summary', {})
    node_labels = summary.get('node_labels', [])
    n = len(node_labels)
    eva_to_idx = {c: i for i, c in enumerate(node_labels)}

    # Load corpus and split by folio
    print("  Splitting corpus 50/50 by folio ...")
    corpus = load_corpus(verbose=False)
    folios = sorted(corpus.pages.keys())
    half1_folios = set(folios[::2])  # even-indexed
    half2_folios = set(folios[1::2])  # odd-indexed

    half1_tokens = []
    half2_tokens = []
    half1_pairs = []
    half2_pairs = []

    for fol, page in corpus.pages.items():
        tokens = page.all_tokens
        if fol in half1_folios:
            half1_tokens.extend(tokens)
            for i in range(len(tokens) - 1):
                half1_pairs.append((tokens[i], tokens[i + 1]))
        else:
            half2_tokens.extend(tokens)
            for i in range(len(tokens) - 1):
                half2_pairs.append((tokens[i], tokens[i + 1]))

    print(f"  Half 1: {len(half1_tokens)} tokens, Half 2: {len(half2_tokens)} tokens")

    # Build co-occurrence for each half
    def _fit_half(tokens, pairs):
        L1, L2, L3, L4 = _build_cooccurrence_layers(
            tokens, node_labels, eva_to_idx, pairs,
        )
        combined = 2.0 * L1 + 1.0 * L2 + 1.5 * L3 + 1.0 * L4
        combined = (combined + combined.T) / 2.0
        max_val = combined.max()
        if max_val > 0:
            combined /= max_val

        from sklearn.cluster import SpectralClustering
        from sklearn.metrics import silhouette_score as sil_score

        best_k, best_labels = 5, np.zeros(n, dtype=int)
        best_sil = -1.0
        for k in range(3, 13):
            try:
                sc = SpectralClustering(
                    n_clusters=k, affinity='precomputed',
                    random_state=42, n_init=10,
                )
                labels = sc.fit_predict(combined + 1e-8)
                dist = 1.0 - combined
                np.fill_diagonal(dist, 0)
                sil = sil_score(dist, labels, metric='precomputed')
                if sil > best_sil:
                    best_sil = sil
                    best_k = k
                    best_labels = labels
            except Exception:
                continue
        return best_labels, best_k

    print("  Fitting half 1 ...")
    labels1, k1 = _fit_half(half1_tokens, half1_pairs)
    print(f"  Half 1: k={k1}")

    print("  Fitting half 2 ...")
    labels2, k2 = _fit_half(half2_tokens, half2_pairs)
    print(f"  Half 2: k={k2}")

    # Compare
    from sklearn.metrics import adjusted_rand_score
    ari = adjusted_rand_score(labels1, labels2)

    # Prediction stability: for each char, check if same community
    # (after label alignment via ARI, which is label-invariant)
    n_stable = sum(1 for i in range(n) if labels1[i] == labels2[i])
    stability = n_stable / n if n > 0 else 0.0

    gate = ari > 0.3
    if ari > 0.5:
        verdict = "STABLE"
    elif ari > 0.3:
        verdict = "MODERATE"
    else:
        verdict = "UNSTABLE"

    result = SBMValidationResult(
        split_half_ari=round(ari, 4),
        prediction_stability=round(stability, 4),
        community_count_half1=k1,
        community_count_half2=k2,
        gate_passed=gate,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'sbm_validation.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2)

    print(f"\n  Split-half ARI: {ari:.4f}")
    print(f"  Prediction stability: {stability:.4f}")
    print(f"  Verdict: {verdict}")
    print(f"  Saved -> {out_path}")
    print(f"\n  Step 44B.5 completed in {time.time() - t0:.1f}s")


# ---------------------------------------------------------------------------
# Track B runner
# ---------------------------------------------------------------------------

def run_track_b() -> None:
    """Run all Track B steps."""
    run_sbm_graph()
    print("\n" + "=" * 70 + "\n")
    run_sbm_fit()
    print("\n" + "=" * 70 + "\n")
    run_sbm_compare()
    print("\n" + "=" * 70 + "\n")
    run_sbm_predict()
    print("\n" + "=" * 70 + "\n")
    run_sbm_validate()
