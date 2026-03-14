"""
Phase 49 Track D – Spectral Graph Matching
============================================
Compare word-transition graph structures via Laplacian eigenvalue spectra
and NetLSD heat kernel signatures. Permutation-invariant language comparison.

Dependency chain:
    signal_bigrams.json         (Phase 29 parallel arrays)
        -> graph_build.json     (Step 49D.1)
        -> graph_laplacian.json (Step 49D.2)
        -> graph_netlsd.json    (Step 49D.3)
        -> graph_verdict.json   (Step 49D.4)
"""

from __future__ import annotations

import json
import math
import os
import pickle
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.distance import cdist

from voynich.core._paths import results_dir as _results_dir
from voynich.core.reference import load_reference_corpus
from voynich.core.stats import netlsd_signature, spectral_distance


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
    if isinstance(obj, float) and (obj != obj):
        return None
    if isinstance(obj, (bool, int, float, str, type(None))):
        return obj
    return str(obj)


def _safe_load(path: str) -> Dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _save_json(rd: str, filename: str, data: Any) -> str:
    path = os.path.join(rd, filename)
    with open(path, 'w') as f:
        json.dump(_convert(data), f, indent=2)
    return path


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class GraphBuildResult:
    voynich_n_nodes: int
    voynich_n_edges: int
    voynich_density: float
    per_lang_n_nodes: Dict[str, int]
    per_lang_n_edges: Dict[str, int]
    per_lang_density: Dict[str, float]
    languages: List[str]
    runtime_seconds: float


@dataclass
class GraphLaplacianResult:
    voynich_n_eigenvalues: int
    voynich_fiedler_value: float
    voynich_spectral_gap: float
    per_lang_fiedler: Dict[str, float]
    per_lang_spectral_gap: Dict[str, float]
    eigenvalue_distance_ranking: List[str]
    eigenvalue_distances: Dict[str, float]
    top_language: str
    runtime_seconds: float


@dataclass
class GraphNetLSDResult:
    per_lang_netlsd_distance: Dict[str, float]
    ranking: List[str]
    top_language: str
    n_timescales: int
    runtime_seconds: float


@dataclass
class GraphVerdictResult:
    laplacian_ranking: List[str]
    netlsd_ranking: List[str]
    combined_ranking: List[str]
    combined_scores: Dict[str, int]
    top_language: str
    fiedler_cluster_a: List[str]
    fiedler_cluster_b: List[str]
    n_cluster_a: int
    n_cluster_b: int
    discriminates_top2: bool
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Graph building helpers
# ---------------------------------------------------------------------------

def _build_transition_graph(
    tokens: List[str],
    top_k: int = 500,
) -> Tuple[List[str], np.ndarray]:
    """Build word bigram adjacency matrix."""
    freq = Counter(tokens)
    vocab = [w for w, _ in freq.most_common(top_k)]
    w2i = {w: i for i, w in enumerate(vocab)}
    n = len(vocab)

    if n == 0:
        return [], np.zeros((0, 0))

    A = np.zeros((n, n))
    for i_pos in range(len(tokens) - 1):
        w1, w2 = tokens[i_pos], tokens[i_pos + 1]
        if w1 in w2i and w2 in w2i:
            A[w2i[w1], w2i[w2]] += 1
            A[w2i[w2], w2i[w1]] += 1  # symmetrize
    return vocab, A


def _graph_stats(A: np.ndarray) -> Tuple[int, int, float]:
    """Return (n_nodes, n_edges, density) for a symmetric adjacency matrix."""
    n = A.shape[0]
    if n == 0:
        return 0, 0, 0.0
    # Count edges (upper triangle with nonzero weight)
    n_edges = int(np.count_nonzero(np.triu(A, k=1)))
    max_edges = n * (n - 1) / 2
    density = n_edges / max_edges if max_edges > 0 else 0.0
    return n, n_edges, density


def _normalized_laplacian_eigenvalues(A: np.ndarray) -> np.ndarray:
    """Compute sorted eigenvalues of the normalized graph Laplacian."""
    n = A.shape[0]
    if n == 0:
        return np.array([])
    d = A.sum(axis=1)
    d_inv_sqrt = np.zeros(n)
    mask = d > 0
    d_inv_sqrt[mask] = 1.0 / np.sqrt(d[mask])
    D_inv_sqrt = np.diag(d_inv_sqrt)
    L = np.eye(n) - D_inv_sqrt @ A @ D_inv_sqrt
    eigenvalues = np.sort(np.linalg.eigvalsh(L))
    return eigenvalues


def _normalized_laplacian_eigen(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute sorted eigenvalues AND eigenvectors of the normalized Laplacian."""
    n = A.shape[0]
    if n == 0:
        return np.array([]), np.zeros((0, 0))
    d = A.sum(axis=1)
    d_inv_sqrt = np.zeros(n)
    mask = d > 0
    d_inv_sqrt[mask] = 1.0 / np.sqrt(d[mask])
    D_inv_sqrt = np.diag(d_inv_sqrt)
    L = np.eye(n) - D_inv_sqrt @ A @ D_inv_sqrt
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    order = np.argsort(eigenvalues)
    return eigenvalues[order], eigenvectors[:, order]


# ---------------------------------------------------------------------------
# Step 49D.1  Build Word Bigram Transition Graphs
# ---------------------------------------------------------------------------

def run_graph_build() -> GraphBuildResult:
    """Build word bigram transition graphs for Voynich and reference corpora."""
    t0 = time.time()
    rd = _results_dir()
    languages = ["latin", "italian", "occitan", "german"]

    # --- Voynich ---
    sb_path = os.path.join(rd, "signal_bigrams.json")
    sb = _safe_load(sb_path)
    voynich_tokens = sb.get("token_decoded", [])
    if not voynich_tokens:
        voynich_tokens = []

    v_vocab, v_A = _build_transition_graph(voynich_tokens, top_k=500)
    v_nodes, v_edges, v_density = _graph_stats(v_A)

    adjacency_store: Dict[str, Tuple[List[str], Any]] = {
        "voynich": (v_vocab, v_A),
    }

    per_lang_n_nodes: Dict[str, int] = {}
    per_lang_n_edges: Dict[str, int] = {}
    per_lang_density: Dict[str, float] = {}

    # --- Reference languages ---
    for lang in languages:
        try:
            ref_corpus = load_reference_corpus(languages=[lang], verbose=False)
            lang_tokens: List[str] = []
            for _name, texts in ref_corpus.texts.items():
                for rt in texts:
                    lang_tokens.extend(rt.tokens)
        except Exception:
            lang_tokens = []

        r_vocab, r_A = _build_transition_graph(lang_tokens, top_k=500)
        r_nodes, r_edges, r_density = _graph_stats(r_A)
        adjacency_store[lang] = (r_vocab, r_A)
        per_lang_n_nodes[lang] = r_nodes
        per_lang_n_edges[lang] = r_edges
        per_lang_density[lang] = round(r_density, 6)

    # Save adjacency matrices in pickle for later steps
    pkl_path = os.path.join(rd, "graph_adjacency.pkl")
    with open(pkl_path, 'wb') as f:
        pickle.dump(adjacency_store, f, protocol=pickle.HIGHEST_PROTOCOL)

    result = GraphBuildResult(
        voynich_n_nodes=v_nodes,
        voynich_n_edges=v_edges,
        voynich_density=round(v_density, 6),
        per_lang_n_nodes=per_lang_n_nodes,
        per_lang_n_edges=per_lang_n_edges,
        per_lang_density=per_lang_density,
        languages=languages,
        runtime_seconds=round(time.time() - t0, 3),
    )
    _save_json(rd, "graph_build.json", asdict(result))
    print(f"[49D.1] Graphs built — Voynich: {v_nodes} nodes, {v_edges} edges, "
          f"density={v_density:.4f}")
    return result


# ---------------------------------------------------------------------------
# Step 49D.2  Laplacian Eigenvalue Spectra
# ---------------------------------------------------------------------------

def run_graph_laplacian() -> GraphLaplacianResult:
    """Compute Laplacian eigenvalue spectra and compare across languages."""
    t0 = time.time()
    rd = _results_dir()

    pkl_path = os.path.join(rd, "graph_adjacency.pkl")
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Missing {pkl_path} — run run_graph_build() first")

    with open(pkl_path, 'rb') as f:
        adjacency_store = pickle.load(f)

    # Voynich eigenvalues
    v_vocab, v_A = adjacency_store["voynich"]
    v_eigs = _normalized_laplacian_eigenvalues(v_A)

    if len(v_eigs) >= 2:
        v_fiedler = float(v_eigs[1])
        v_spectral_gap = float(v_eigs[1] / v_eigs[-1]) if v_eigs[-1] > 0 else 0.0
    else:
        v_fiedler = 0.0
        v_spectral_gap = 0.0

    # Store eigenvalues for NetLSD step
    eig_store: Dict[str, List[float]] = {
        "voynich": v_eigs.tolist(),
    }

    per_lang_fiedler: Dict[str, float] = {}
    per_lang_spectral_gap: Dict[str, float] = {}
    eigenvalue_distances: Dict[str, float] = {}

    languages = [k for k in adjacency_store if k != "voynich"]

    for lang in languages:
        r_vocab, r_A = adjacency_store[lang]
        r_eigs = _normalized_laplacian_eigenvalues(r_A)
        eig_store[lang] = r_eigs.tolist()

        if len(r_eigs) >= 2:
            per_lang_fiedler[lang] = round(float(r_eigs[1]), 8)
            per_lang_spectral_gap[lang] = round(
                float(r_eigs[1] / r_eigs[-1]) if r_eigs[-1] > 0 else 0.0, 8,
            )
        else:
            per_lang_fiedler[lang] = 0.0
            per_lang_spectral_gap[lang] = 0.0

        # L2 distance between eigenvalue spectra (padded to same length with 2.0)
        max_len = max(len(v_eigs), len(r_eigs))
        v_padded = np.full(max_len, 2.0)
        r_padded = np.full(max_len, 2.0)
        v_padded[:len(v_eigs)] = v_eigs
        r_padded[:len(r_eigs)] = r_eigs
        dist = float(np.sqrt(np.sum((v_padded - r_padded) ** 2)))
        eigenvalue_distances[lang] = round(dist, 6)

    # Save eigenvalues pickle for NetLSD step
    eig_pkl_path = os.path.join(rd, "graph_eigenvalues.pkl")
    with open(eig_pkl_path, 'wb') as f:
        pickle.dump(eig_store, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Rank by eigenvalue distance
    ranking = sorted(
        [l for l in languages if math.isfinite(eigenvalue_distances.get(l, float('inf')))],
        key=lambda l: eigenvalue_distances[l],
    )
    ranking += [l for l in languages if l not in ranking]

    top_language = ranking[0] if ranking else "unknown"

    result = GraphLaplacianResult(
        voynich_n_eigenvalues=len(v_eigs),
        voynich_fiedler_value=round(v_fiedler, 8),
        voynich_spectral_gap=round(v_spectral_gap, 8),
        per_lang_fiedler=per_lang_fiedler,
        per_lang_spectral_gap=per_lang_spectral_gap,
        eigenvalue_distance_ranking=ranking,
        eigenvalue_distances=eigenvalue_distances,
        top_language=top_language,
        runtime_seconds=round(time.time() - t0, 3),
    )
    _save_json(rd, "graph_laplacian.json", asdict(result))
    print(f"[49D.2] Laplacian spectra — top={top_language}, "
          f"Fiedler(Voynich)={v_fiedler:.6f}, distances={eigenvalue_distances}")
    return result


# ---------------------------------------------------------------------------
# Step 49D.3  NetLSD Heat Kernel Signatures
# ---------------------------------------------------------------------------

def run_graph_netlsd() -> GraphNetLSDResult:
    """Compare graphs via NetLSD heat kernel signatures."""
    t0 = time.time()
    rd = _results_dir()

    # Try loading eigenvalues from pickle; fall back to recomputing from adjacency
    eig_pkl_path = os.path.join(rd, "graph_eigenvalues.pkl")
    pkl_path = os.path.join(rd, "graph_adjacency.pkl")

    if os.path.exists(eig_pkl_path):
        with open(eig_pkl_path, 'rb') as f:
            eig_store = pickle.load(f)
    elif os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            adjacency_store = pickle.load(f)
        eig_store = {}
        for name, (vocab, A) in adjacency_store.items():
            eigs = _normalized_laplacian_eigenvalues(A)
            eig_store[name] = eigs.tolist()
    else:
        raise FileNotFoundError(
            f"Missing {eig_pkl_path} and {pkl_path} — run earlier steps first"
        )

    # Compute NetLSD signatures
    timescales = np.logspace(-2, 2, 50)
    n_timescales = len(timescales)

    v_eigs = np.array(eig_store.get("voynich", []))
    if v_eigs.size == 0:
        raise ValueError("Voynich eigenvalues are empty")

    v_sig = netlsd_signature(v_eigs, timescales=timescales)

    languages = [k for k in eig_store if k != "voynich"]
    per_lang_netlsd: Dict[str, float] = {}

    for lang in languages:
        r_eigs = np.array(eig_store.get(lang, []))
        if r_eigs.size == 0:
            per_lang_netlsd[lang] = float('inf')
            continue
        r_sig = netlsd_signature(r_eigs, timescales=timescales)
        dist = spectral_distance(v_sig, r_sig)
        per_lang_netlsd[lang] = round(dist, 8)

    # Rank
    ranking = sorted(
        [l for l in languages if math.isfinite(per_lang_netlsd.get(l, float('inf')))],
        key=lambda l: per_lang_netlsd[l],
    )
    ranking += [l for l in languages if l not in ranking]

    top_language = ranking[0] if ranking else "unknown"

    result = GraphNetLSDResult(
        per_lang_netlsd_distance=per_lang_netlsd,
        ranking=ranking,
        top_language=top_language,
        n_timescales=n_timescales,
        runtime_seconds=round(time.time() - t0, 3),
    )
    _save_json(rd, "graph_netlsd.json", asdict(result))
    print(f"[49D.3] NetLSD — top={top_language}, distances={per_lang_netlsd}")
    return result


# ---------------------------------------------------------------------------
# Step 49D.4  Graph Matching Verdict
# ---------------------------------------------------------------------------

def run_graph_verdict() -> GraphVerdictResult:
    """Combine Laplacian and NetLSD rankings; compute Fiedler partition."""
    t0 = time.time()
    rd = _results_dir()

    lap = _safe_load(os.path.join(rd, "graph_laplacian.json"))
    net = _safe_load(os.path.join(rd, "graph_netlsd.json"))

    if not lap or not net:
        raise FileNotFoundError("Missing graph_laplacian.json or graph_netlsd.json")

    l_ranking: List[str] = lap.get("eigenvalue_distance_ranking", [])
    n_ranking: List[str] = net.get("ranking", [])

    # Borda count
    all_langs = list(dict.fromkeys(l_ranking + n_ranking))
    borda: Dict[str, int] = {}
    for lang in all_langs:
        l_rank = (l_ranking.index(lang) + 1) if lang in l_ranking else len(all_langs) + 1
        n_rank = (n_ranking.index(lang) + 1) if lang in n_ranking else len(all_langs) + 1
        borda[lang] = l_rank + n_rank

    combined_ranking = sorted(all_langs, key=lambda l: borda[l])
    top_language = combined_ranking[0] if combined_ranking else "unknown"

    discriminates = False
    if len(combined_ranking) >= 2:
        s1 = borda.get(combined_ranking[0], 0)
        s2 = borda.get(combined_ranking[1], 0)
        discriminates = s2 > s1

    # --- Fiedler vector partition ---
    fiedler_cluster_a: List[str] = []
    fiedler_cluster_b: List[str] = []
    n_cluster_a = 0
    n_cluster_b = 0

    pkl_path = os.path.join(rd, "graph_adjacency.pkl")
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            adjacency_store = pickle.load(f)

        v_vocab, v_A = adjacency_store.get("voynich", ([], np.zeros((0, 0))))
        if len(v_vocab) >= 2 and v_A.shape[0] >= 2:
            eigenvalues, eigenvectors = _normalized_laplacian_eigen(v_A)
            # Fiedler vector = eigenvector for 2nd smallest eigenvalue
            fiedler_vec = eigenvectors[:, 1]

            # Partition by sign
            pos_idx = np.where(fiedler_vec >= 0)[0]
            neg_idx = np.where(fiedler_vec < 0)[0]

            n_cluster_a = len(pos_idx)
            n_cluster_b = len(neg_idx)

            # Sort each cluster by absolute Fiedler value (most extreme first)
            pos_sorted = pos_idx[np.argsort(-np.abs(fiedler_vec[pos_idx]))]
            neg_sorted = neg_idx[np.argsort(-np.abs(fiedler_vec[neg_idx]))]

            fiedler_cluster_a = [v_vocab[i] for i in pos_sorted[:20]]
            fiedler_cluster_b = [v_vocab[i] for i in neg_sorted[:20]]

    result = GraphVerdictResult(
        laplacian_ranking=l_ranking,
        netlsd_ranking=n_ranking,
        combined_ranking=combined_ranking,
        combined_scores=borda,
        top_language=top_language,
        fiedler_cluster_a=fiedler_cluster_a,
        fiedler_cluster_b=fiedler_cluster_b,
        n_cluster_a=n_cluster_a,
        n_cluster_b=n_cluster_b,
        discriminates_top2=discriminates,
        runtime_seconds=round(time.time() - t0, 3),
    )
    _save_json(rd, "graph_verdict.json", asdict(result))
    print(f"[49D.4] Verdict — top={top_language}, borda={borda}, "
          f"clusters=({n_cluster_a}, {n_cluster_b}), discriminates={discriminates}")
    return result


# ---------------------------------------------------------------------------
# Track D runner
# ---------------------------------------------------------------------------

def run_track_d_49() -> Dict[str, Any]:
    """Run all Track D (Spectral Graph Matching) steps sequentially."""
    print("=" * 60)
    print("Phase 49 Track D — Spectral Graph Matching")
    print("=" * 60)

    build = run_graph_build()
    lap = run_graph_laplacian()
    net = run_graph_netlsd()
    verdict = run_graph_verdict()

    summary = {
        "track": "D",
        "steps": ["49D.1", "49D.2", "49D.3", "49D.4"],
        "top_language": verdict.top_language,
        "laplacian_ranking": verdict.laplacian_ranking,
        "netlsd_ranking": verdict.netlsd_ranking,
        "combined_ranking": verdict.combined_ranking,
        "discriminates": verdict.discriminates_top2,
        "fiedler_cluster_a_top5": verdict.fiedler_cluster_a[:5],
        "fiedler_cluster_b_top5": verdict.fiedler_cluster_b[:5],
    }
    print(f"\nTrack D complete — top language: {verdict.top_language}")
    return summary
