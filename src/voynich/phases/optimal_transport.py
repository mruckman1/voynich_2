"""
Phase 49 Track C – Optimal Transport Language Identification
=============================================================
Unsupervised language identification via entropic Sinkhorn and
Gromov-Wasserstein distances on co-occurrence embeddings.

Dependency chain:
    signal_bigrams.json         (Phase 29 parallel arrays)
        -> ot_embeddings.json   (Step 49C.1)
        -> ot_sinkhorn.json     (Step 49C.2)
        -> ot_gromov.json       (Step 49C.3)
        -> ot_langid.json       (Step 49C.4)
"""

from __future__ import annotations

import json
import math
import os
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist
from sklearn.decomposition import TruncatedSVD

from voynich.core._paths import results_dir as _results_dir
from voynich.core.reference import load_reference_corpus
from voynich.core.stats import gromov_wasserstein, sinkhorn_ot


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
class OTEmbeddingsResult:
    voynich_vocab_size: int
    voynich_embedding_dim: int
    per_lang_vocab_size: Dict[str, int]
    per_lang_embedding_dim: Dict[str, int]
    ppmi_window_size: int
    svd_components: int
    languages: List[str]
    runtime_seconds: float


@dataclass
class OTSinkhornResult:
    per_lang_wasserstein: Dict[str, float]
    per_lang_sinkhorn_converged: Dict[str, bool]
    ranking: List[str]
    top_language: str
    margin: float
    runtime_seconds: float


@dataclass
class OTGromovResult:
    per_lang_gw_distance: Dict[str, float]
    per_lang_vocab_used: Dict[str, int]
    ranking: List[str]
    top_language: str
    discriminates_top2: bool
    runtime_seconds: float


@dataclass
class OTLangIDResult:
    wasserstein_ranking: List[str]
    gromov_ranking: List[str]
    combined_ranking: List[str]
    combined_scores: Dict[str, int]
    top_language: str
    discriminates_top2: bool
    agreement_w_gw: bool
    consistency_with_prior: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# PPMI + SVD embedding builder
# ---------------------------------------------------------------------------

def _build_ppmi_svd(
    tokens: List[str],
    top_k: int = 500,
    window: int = 5,
    n_components: int = 50,
) -> Tuple[List[str], np.ndarray]:
    """Build PPMI+SVD embeddings from a token sequence."""
    # Count frequencies
    freq = Counter(tokens)
    vocab = [w for w, _ in freq.most_common(top_k)]
    w2i = {w: i for i, w in enumerate(vocab)}
    n = len(vocab)

    if n == 0:
        return [], np.zeros((0, n_components))

    # Co-occurrence
    cooc = np.zeros((n, n))
    for pos in range(len(tokens)):
        w = tokens[pos]
        if w not in w2i:
            continue
        i = w2i[w]
        for offset in range(1, window + 1):
            if pos + offset < len(tokens):
                w2 = tokens[pos + offset]
                if w2 in w2i:
                    j = w2i[w2]
                    cooc[i, j] += 1
                    cooc[j, i] += 1

    # PPMI
    total = cooc.sum()
    if total == 0:
        return vocab, np.zeros((n, n_components))
    row_sums = cooc.sum(axis=1)
    col_sums = cooc.sum(axis=0)

    with np.errstate(divide='ignore', invalid='ignore'):
        pmi = np.log(cooc * total / (np.outer(row_sums, col_sums) + 1e-20) + 1e-20)
    ppmi = np.maximum(pmi, 0)

    # SVD
    actual_components = min(n_components, n - 1, ppmi.shape[0] - 1)
    if actual_components < 1:
        return vocab, np.zeros((n, n_components))
    svd = TruncatedSVD(n_components=actual_components, random_state=42)
    embeddings = svd.fit_transform(csr_matrix(ppmi))

    # Pad if needed
    if embeddings.shape[1] < n_components:
        pad = np.zeros((n, n_components - embeddings.shape[1]))
        embeddings = np.concatenate([embeddings, pad], axis=1)

    return vocab, embeddings


# ---------------------------------------------------------------------------
# Step 49C.1  PPMI+SVD Embeddings
# ---------------------------------------------------------------------------

def run_ot_embeddings() -> OTEmbeddingsResult:
    """Build PPMI+SVD co-occurrence embeddings for Voynich and reference corpora."""
    t0 = time.time()
    rd = _results_dir()

    top_k = 500
    window = 5
    n_components = 50
    languages = ["latin", "italian", "occitan", "german"]

    # --- Voynich decoded tokens ---
    sb_path = os.path.join(rd, "signal_bigrams.json")
    sb = _safe_load(sb_path)
    voynich_tokens = sb.get("token_decoded", [])
    if not voynich_tokens:
        voynich_tokens = []

    v_vocab, v_emb = _build_ppmi_svd(
        voynich_tokens, top_k=top_k, window=window, n_components=n_components,
    )

    # Store Voynich embeddings
    emb_store: Dict[str, Any] = {
        "voynich_vocab": v_vocab,
        "voynich_embeddings": v_emb.tolist(),
    }

    per_lang_vocab_size: Dict[str, int] = {}
    per_lang_embedding_dim: Dict[str, int] = {}

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

        r_vocab, r_emb = _build_ppmi_svd(
            lang_tokens, top_k=top_k, window=window, n_components=n_components,
        )
        emb_store[f"{lang}_vocab"] = r_vocab
        emb_store[f"{lang}_embeddings"] = r_emb.tolist()
        per_lang_vocab_size[lang] = len(r_vocab)
        per_lang_embedding_dim[lang] = r_emb.shape[1] if len(r_vocab) > 0 else 0

    _save_json(rd, "ot_embeddings.json", emb_store)

    result = OTEmbeddingsResult(
        voynich_vocab_size=len(v_vocab),
        voynich_embedding_dim=v_emb.shape[1] if len(v_vocab) > 0 else 0,
        per_lang_vocab_size=per_lang_vocab_size,
        per_lang_embedding_dim=per_lang_embedding_dim,
        ppmi_window_size=window,
        svd_components=n_components,
        languages=languages,
        runtime_seconds=round(time.time() - t0, 3),
    )
    _save_json(rd, "ot_embeddings_result.json", asdict(result))
    print(f"[49C.1] Embeddings built — Voynich vocab={len(v_vocab)}, "
          f"langs={languages}, dim={n_components}")
    return result


# ---------------------------------------------------------------------------
# Step 49C.2  Sinkhorn OT Distances
# ---------------------------------------------------------------------------

def run_ot_sinkhorn() -> OTSinkhornResult:
    """Compute Sinkhorn OT distances between Voynich and reference embeddings."""
    t0 = time.time()
    rd = _results_dir()

    emb_path = os.path.join(rd, "ot_embeddings.json")
    emb = _safe_load(emb_path)
    if not emb:
        raise FileNotFoundError(f"Missing {emb_path} — run run_ot_embeddings() first")

    v_emb = np.array(emb["voynich_embeddings"])
    if v_emb.size == 0:
        raise ValueError("Voynich embeddings are empty")

    # Determine languages from stored keys
    languages = [
        k.replace("_vocab", "")
        for k in emb
        if k.endswith("_vocab") and k != "voynich_vocab"
    ]

    per_lang_wasserstein: Dict[str, float] = {}
    per_lang_converged: Dict[str, bool] = {}

    for lang in languages:
        r_emb = np.array(emb.get(f"{lang}_embeddings", []))
        if r_emb.size == 0:
            per_lang_wasserstein[lang] = float('inf')
            per_lang_converged[lang] = False
            continue

        # Cost matrix: pairwise Euclidean distances
        C = cdist(v_emb, r_emb, metric='euclidean')

        # Normalize cost to avoid numerical issues
        c_max = C.max()
        if c_max > 0:
            C = C / c_max

        n_v = v_emb.shape[0]
        n_r = r_emb.shape[0]
        a = np.ones(n_v) / n_v
        b = np.ones(n_r) / n_r

        try:
            _gamma, w_dist = sinkhorn_ot(C, a, b, reg=0.1, max_iter=100)
            # Scale distance back
            per_lang_wasserstein[lang] = w_dist * c_max
            per_lang_converged[lang] = True
        except Exception:
            per_lang_wasserstein[lang] = float('inf')
            per_lang_converged[lang] = False

    # Rank by distance (lower = more similar)
    ranking = sorted(
        [l for l in languages if math.isfinite(per_lang_wasserstein.get(l, float('inf')))],
        key=lambda l: per_lang_wasserstein[l],
    )
    # Append any infinite ones at the end
    ranking += [l for l in languages if l not in ranking]

    top_language = ranking[0] if ranking else "unknown"
    if len(ranking) >= 2:
        d1 = per_lang_wasserstein.get(ranking[0], float('inf'))
        d2 = per_lang_wasserstein.get(ranking[1], float('inf'))
        margin = (d2 - d1) / (d1 + 1e-20) if math.isfinite(d1) and math.isfinite(d2) else 0.0
    else:
        margin = 0.0

    result = OTSinkhornResult(
        per_lang_wasserstein=per_lang_wasserstein,
        per_lang_sinkhorn_converged=per_lang_converged,
        ranking=ranking,
        top_language=top_language,
        margin=round(margin, 4),
        runtime_seconds=round(time.time() - t0, 3),
    )
    _save_json(rd, "ot_sinkhorn.json", asdict(result))
    print(f"[49C.2] Sinkhorn OT — top={top_language}, "
          f"W={per_lang_wasserstein.get(top_language, 'N/A'):.4f}, margin={margin:.4f}")
    return result


# ---------------------------------------------------------------------------
# Step 49C.3  Gromov-Wasserstein Distance
# ---------------------------------------------------------------------------

def run_ot_gromov() -> OTGromovResult:
    """Compute Gromov-Wasserstein distances between Voynich and reference embeddings."""
    t0 = time.time()
    rd = _results_dir()

    emb_path = os.path.join(rd, "ot_embeddings.json")
    emb = _safe_load(emb_path)
    if not emb:
        raise FileNotFoundError(f"Missing {emb_path} — run run_ot_embeddings() first")

    v_emb = np.array(emb["voynich_embeddings"])
    if v_emb.size == 0:
        raise ValueError("Voynich embeddings are empty")

    languages = [
        k.replace("_vocab", "")
        for k in emb
        if k.endswith("_vocab") and k != "voynich_vocab"
    ]

    # Subsample Voynich if needed
    max_vocab = 200
    if v_emb.shape[0] > max_vocab:
        rng = np.random.RandomState(42)
        idx_v = rng.choice(v_emb.shape[0], max_vocab, replace=False)
        v_sub = v_emb[idx_v]
    else:
        v_sub = v_emb

    D_voynich = cdist(v_sub, v_sub, metric='euclidean')
    # Normalize
    d_max_v = D_voynich.max()
    if d_max_v > 0:
        D_voynich = D_voynich / d_max_v

    per_lang_gw: Dict[str, float] = {}
    per_lang_vocab_used: Dict[str, int] = {}

    for lang in languages:
        r_emb = np.array(emb.get(f"{lang}_embeddings", []))
        if r_emb.size == 0:
            per_lang_gw[lang] = float('inf')
            per_lang_vocab_used[lang] = 0
            continue

        # Subsample reference if needed
        if r_emb.shape[0] > max_vocab:
            rng = np.random.RandomState(42)
            idx_r = rng.choice(r_emb.shape[0], max_vocab, replace=False)
            r_sub = r_emb[idx_r]
        else:
            r_sub = r_emb

        per_lang_vocab_used[lang] = r_sub.shape[0]

        D_ref = cdist(r_sub, r_sub, metric='euclidean')
        d_max_r = D_ref.max()
        if d_max_r > 0:
            D_ref = D_ref / d_max_r

        p = np.ones(v_sub.shape[0]) / v_sub.shape[0]
        q = np.ones(r_sub.shape[0]) / r_sub.shape[0]

        try:
            _coupling, gw_dist = gromov_wasserstein(
                D_voynich, D_ref, p, q, reg=0.1, max_iter=50,
            )
            per_lang_gw[lang] = gw_dist
        except Exception:
            per_lang_gw[lang] = float('inf')

    # Rank
    ranking = sorted(
        [l for l in languages if math.isfinite(per_lang_gw.get(l, float('inf')))],
        key=lambda l: per_lang_gw[l],
    )
    ranking += [l for l in languages if l not in ranking]

    top_language = ranking[0] if ranking else "unknown"
    discriminates = False
    if len(ranking) >= 2:
        d1 = per_lang_gw.get(ranking[0], float('inf'))
        d2 = per_lang_gw.get(ranking[1], float('inf'))
        if math.isfinite(d1) and d1 > 0:
            discriminates = (d2 / d1) > 1.3

    result = OTGromovResult(
        per_lang_gw_distance=per_lang_gw,
        per_lang_vocab_used=per_lang_vocab_used,
        ranking=ranking,
        top_language=top_language,
        discriminates_top2=discriminates,
        runtime_seconds=round(time.time() - t0, 3),
    )
    _save_json(rd, "ot_gromov.json", asdict(result))
    print(f"[49C.3] Gromov-Wasserstein — top={top_language}, "
          f"GW={per_lang_gw.get(top_language, 'N/A'):.6f}, "
          f"discriminates={discriminates}")
    return result


# ---------------------------------------------------------------------------
# Step 49C.4  Language ID Verdict
# ---------------------------------------------------------------------------

def run_ot_langid() -> OTLangIDResult:
    """Combine Sinkhorn and GW rankings via Borda count for language ID verdict."""
    t0 = time.time()
    rd = _results_dir()

    sink_path = os.path.join(rd, "ot_sinkhorn.json")
    gw_path = os.path.join(rd, "ot_gromov.json")
    sink = _safe_load(sink_path)
    gw = _safe_load(gw_path)

    if not sink or not gw:
        raise FileNotFoundError("Missing ot_sinkhorn.json or ot_gromov.json")

    w_ranking: List[str] = sink.get("ranking", [])
    g_ranking: List[str] = gw.get("ranking", [])

    # Collect all languages
    all_langs = list(dict.fromkeys(w_ranking + g_ranking))

    # Borda count: rank 1 gets score 1, rank 2 gets 2, etc. Lowest total wins.
    borda: Dict[str, int] = {}
    for lang in all_langs:
        w_rank = (w_ranking.index(lang) + 1) if lang in w_ranking else len(all_langs) + 1
        g_rank = (g_ranking.index(lang) + 1) if lang in g_ranking else len(all_langs) + 1
        borda[lang] = w_rank + g_rank

    combined_ranking = sorted(all_langs, key=lambda l: borda[l])
    top_language = combined_ranking[0] if combined_ranking else "unknown"

    # Check discrimination
    discriminates = False
    if len(combined_ranking) >= 2:
        s1 = borda.get(combined_ranking[0], 0)
        s2 = borda.get(combined_ranking[1], 0)
        discriminates = s2 > s1  # strict advantage

    # Agreement between the two methods
    w_top = w_ranking[0] if w_ranking else None
    g_top = g_ranking[0] if g_ranking else None
    agreement = w_top == g_top

    # Consistency with prior Latin hypothesis
    consistency = "agrees" if top_language == "latin" else "disagrees"

    result = OTLangIDResult(
        wasserstein_ranking=w_ranking,
        gromov_ranking=g_ranking,
        combined_ranking=combined_ranking,
        combined_scores=borda,
        top_language=top_language,
        discriminates_top2=discriminates,
        agreement_w_gw=agreement,
        consistency_with_prior=consistency,
        runtime_seconds=round(time.time() - t0, 3),
    )
    _save_json(rd, "ot_langid.json", asdict(result))
    print(f"[49C.4] Language ID — top={top_language}, "
          f"borda={borda}, agreement={agreement}, prior={consistency}")
    return result


# ---------------------------------------------------------------------------
# Track C runner
# ---------------------------------------------------------------------------

def run_track_c_49() -> Dict[str, Any]:
    """Run all Track C (Optimal Transport) steps sequentially."""
    print("=" * 60)
    print("Phase 49 Track C — Optimal Transport Language Identification")
    print("=" * 60)

    emb = run_ot_embeddings()
    sink = run_ot_sinkhorn()
    gw = run_ot_gromov()
    langid = run_ot_langid()

    summary = {
        "track": "C",
        "steps": ["49C.1", "49C.2", "49C.3", "49C.4"],
        "top_language": langid.top_language,
        "wasserstein_ranking": langid.wasserstein_ranking,
        "gromov_ranking": langid.gromov_ranking,
        "combined_ranking": langid.combined_ranking,
        "discriminates": langid.discriminates_top2,
        "agreement": langid.agreement_w_gw,
        "consistency_with_prior": langid.consistency_with_prior,
    }
    print(f"\nTrack C complete — top language: {langid.top_language}")
    return summary
