"""
Phase 50 Track D – Size-Matched Language Identification
========================================================
Re-run language identification with all corpora subsampled to 11K tokens,
eliminating the corpus-size bias that put German #1 in Phase 49.

Dependency chain:
    signal_bigrams.json     (Phase 29)
        -> size_matched_langid.json
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist
from sklearn.decomposition import TruncatedSVD

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import build_eva_to_triple_lookup, tokenize_eva_chars
from voynich.core.reference import load_reference_corpus
from voynich.core.stats import sinkhorn_ot, gromov_wasserstein, netlsd_signature, spectral_distance


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _convert(obj: Any) -> Any:
    if hasattr(obj, '__dataclass_fields__'):
        return {k: _convert(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _convert(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_convert(item) for item in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return _convert(obj.tolist())
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
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SizeMatchedLangIDResult:
    d1_gw: Dict[str, Dict]       # {lang: {mean, std, rank}}
    d2_spectral: Dict[str, Dict]
    d3_ngram: Dict[str, Dict]
    consensus_ranking: List[Dict]  # [{lang, mean_rank}, ...]
    top_language: str
    margin: float
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Helper: contiguous subsample
# ---------------------------------------------------------------------------

def _subsample_contiguous(
    tokens: List[str],
    target_size: int,
    seed: int,
) -> List[str]:
    """Pick a random contiguous chunk of target_size tokens from the token list."""
    if len(tokens) <= target_size:
        return list(tokens)
    rng = random.Random(seed)
    max_start = len(tokens) - target_size
    start = rng.randint(0, max_start)
    return tokens[start:start + target_size]


# ---------------------------------------------------------------------------
# Helper: PPMI + SVD
# ---------------------------------------------------------------------------

def _build_ppmi_svd(
    tokens: List[str],
    top_k: int = 200,
    window: int = 2,
    n_components: int = 50,
) -> Tuple[List[str], np.ndarray]:
    """Build PPMI+SVD embeddings from a token sequence."""
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
# Helper: transition graph
# ---------------------------------------------------------------------------

def _build_transition_graph(
    tokens: List[str],
    top_k: int = 200,
) -> Tuple[List[str], np.ndarray]:
    """Build symmetric word bigram adjacency matrix."""
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


# ---------------------------------------------------------------------------
# Helper: normalized Laplacian eigenvalues
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Helper: character n-gram profile
# ---------------------------------------------------------------------------

def _char_ngram_profile(
    tokens: List[str],
    orders: Tuple[int, ...] = (2, 3, 4),
) -> Dict[int, Dict[str, float]]:
    """Build character n-gram frequency profiles."""
    text = '_'.join(w.lower() for w in tokens if w.isalpha())
    profiles: Dict[int, Dict[str, float]] = {}
    for n in orders:
        counts: Counter = Counter()
        for i in range(len(text) - n + 1):
            counts[text[i:i + n]] += 1
        total = sum(counts.values())
        if total > 0:
            profiles[n] = {k: v / total for k, v in counts.items()}
        else:
            profiles[n] = {}
    return profiles


# ---------------------------------------------------------------------------
# Helper: profile cosine distance
# ---------------------------------------------------------------------------

def _profile_cosine_distance(p1: Dict[str, float], p2: Dict[str, float]) -> float:
    """Cosine distance between two profiles (dicts of n-gram -> frequency)."""
    all_keys = set(p1.keys()) | set(p2.keys())
    if not all_keys:
        return 1.0

    dot = 0.0
    norm1_sq = 0.0
    norm2_sq = 0.0
    for k in all_keys:
        v1 = p1.get(k, 0.0)
        v2 = p2.get(k, 0.0)
        dot += v1 * v2
        norm1_sq += v1 * v1
        norm2_sq += v2 * v2

    norm1 = math.sqrt(norm1_sq)
    norm2 = math.sqrt(norm2_sq)
    if norm1 < 1e-12 or norm2 < 1e-12:
        return 1.0

    cosine_sim = dot / (norm1 * norm2)
    return 1.0 - cosine_sim


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_size_matched_langid() -> Dict[str, Any]:
    """Phase 50 Track D: Size-Matched Language Identification."""
    t0 = time.time()
    rd = _results_dir()

    TARGET_SIZE = 11000
    N_BOOTSTRAP = 10
    languages = ['latin', 'italian', 'occitan', 'german']

    print("=" * 70)
    print("Phase 50 Track D: Size-Matched Language Identification")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load decoded Voynich tokens
    # ------------------------------------------------------------------
    print("\n--- Step 1: Loading Voynich decoded tokens ---")

    sig_path = os.path.join(rd, 'signal_bigrams.json')
    sig_data = _safe_load(sig_path)
    token_decoded: List[str] = sig_data.get('token_decoded', [])
    print(f"  Decoded tokens: {len(token_decoded)}")

    # Filter to non-empty alpha tokens
    voynich_tokens = [w.lower() for w in token_decoded if w and len(w) >= 2]
    print(f"  Filtered Voynich tokens: {len(voynich_tokens)}")

    # ------------------------------------------------------------------
    # 2. Load reference corpora
    # ------------------------------------------------------------------
    print("\n--- Step 2: Loading reference corpora ---")

    ref_tokens_all: Dict[str, List[str]] = {}
    for lang in languages:
        try:
            ref = load_reference_corpus(languages=[lang], verbose=False)
            raw = ref.get_combined_tokens(lang)
            ref_tokens_all[lang] = [w.lower() for w in raw if len(w) >= 2 and w.isalpha()]
            print(f"  {lang}: {len(ref_tokens_all[lang])} tokens")
        except Exception as e:
            print(f"  {lang}: SKIP ({e})")
            ref_tokens_all[lang] = []

    # ------------------------------------------------------------------
    # 3. Bootstrap loop
    # ------------------------------------------------------------------
    print(f"\n--- Step 3: {N_BOOTSTRAP} bootstrap iterations (target={TARGET_SIZE}) ---")

    # Accumulators: method -> lang -> list of distances
    gw_distances: Dict[str, List[float]] = {lang: [] for lang in languages}
    spectral_distances: Dict[str, List[float]] = {lang: [] for lang in languages}
    ngram_distances: Dict[str, List[float]] = {lang: [] for lang in languages}

    for boot_i in range(N_BOOTSTRAP):
        seed = 100 + boot_i
        print(f"\n  Bootstrap {boot_i + 1}/{N_BOOTSTRAP} (seed={seed})")

        # Subsample Voynich
        v_sub = _subsample_contiguous(voynich_tokens, TARGET_SIZE, seed)

        for lang in languages:
            if len(ref_tokens_all[lang]) < 100:
                gw_distances[lang].append(float('inf'))
                spectral_distances[lang].append(float('inf'))
                ngram_distances[lang].append(float('inf'))
                continue

            r_sub = _subsample_contiguous(ref_tokens_all[lang], TARGET_SIZE, seed)

            # --- D.1 Gromov-Wasserstein ---
            try:
                v_vocab, v_emb = _build_ppmi_svd(v_sub, top_k=200, window=2, n_components=50)
                r_vocab, r_emb = _build_ppmi_svd(r_sub, top_k=200, window=2, n_components=50)

                if v_emb.shape[0] > 0 and r_emb.shape[0] > 0:
                    # Compute intra-distance matrices (cosine)
                    D_v = cdist(v_emb, v_emb, metric='cosine')
                    D_r = cdist(r_emb, r_emb, metric='cosine')

                    # Replace NaN with 1.0
                    D_v = np.nan_to_num(D_v, nan=1.0)
                    D_r = np.nan_to_num(D_r, nan=1.0)

                    # Match sizes
                    n_min = min(D_v.shape[0], D_r.shape[0])
                    D_v_matched = D_v[:n_min, :n_min]
                    D_r_matched = D_r[:n_min, :n_min]

                    p = np.ones(n_min) / n_min
                    q = np.ones(n_min) / n_min

                    gw_dist = gromov_wasserstein(
                        D_v_matched, D_r_matched, p, q,
                        reg=0.05, max_iter=100,
                    )
                    gw_distances[lang].append(float(gw_dist))
                else:
                    gw_distances[lang].append(float('inf'))
            except Exception:
                gw_distances[lang].append(float('inf'))

            # --- D.2 NetLSD Spectral ---
            try:
                _, v_A = _build_transition_graph(v_sub, top_k=200)
                _, r_A = _build_transition_graph(r_sub, top_k=200)

                v_eigs = _normalized_laplacian_eigenvalues(v_A)
                r_eigs = _normalized_laplacian_eigenvalues(r_A)

                timescales = np.logspace(-2, 2, 100)
                v_sig = netlsd_signature(v_eigs, timescales=timescales)
                r_sig = netlsd_signature(r_eigs, timescales=timescales)

                sd = spectral_distance(v_sig, r_sig)
                spectral_distances[lang].append(float(sd))
            except Exception:
                spectral_distances[lang].append(float('inf'))

            # --- D.3 Char N-Gram Profile ---
            try:
                v_profile = _char_ngram_profile(v_sub, orders=(2, 3, 4))
                r_profile = _char_ngram_profile(r_sub, orders=(2, 3, 4))

                dists_per_order: List[float] = []
                for order in (2, 3, 4):
                    vp = v_profile.get(order, {})
                    rp = r_profile.get(order, {})
                    dists_per_order.append(_profile_cosine_distance(vp, rp))
                avg_dist = float(np.mean(dists_per_order))
                ngram_distances[lang].append(avg_dist)
            except Exception:
                ngram_distances[lang].append(float('inf'))

        if (boot_i + 1) % 5 == 0:
            print(f"    Completed {boot_i + 1}/{N_BOOTSTRAP}")

    # ------------------------------------------------------------------
    # 4. Compute per-method rankings
    # ------------------------------------------------------------------
    print("\n--- Step 4: Computing rankings ---")

    def _compute_stats_and_rank(
        dist_dict: Dict[str, List[float]],
    ) -> Tuple[Dict[str, Dict], List[str]]:
        """Compute mean/std per language and rank by mean distance."""
        stats: Dict[str, Dict] = {}
        for lang in languages:
            vals = [v for v in dist_dict[lang] if v != float('inf')]
            if vals:
                mean = float(np.mean(vals))
                std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            else:
                mean = float('inf')
                std = 0.0
            stats[lang] = {'mean': round(mean, 6), 'std': round(std, 6)}

        ranking = sorted(languages, key=lambda l: stats[l]['mean'])
        for rank, lang in enumerate(ranking):
            stats[lang]['rank'] = rank + 1

        return stats, ranking

    gw_stats, gw_ranking = _compute_stats_and_rank(gw_distances)
    spec_stats, spec_ranking = _compute_stats_and_rank(spectral_distances)
    ngram_stats, ngram_ranking = _compute_stats_and_rank(ngram_distances)

    print("\n  D.1 Gromov-Wasserstein ranking:")
    for lang in gw_ranking:
        s = gw_stats[lang]
        print(f"    {lang}: mean={s['mean']:.6f}, std={s['std']:.6f}, rank={s['rank']}")

    print("\n  D.2 Spectral ranking:")
    for lang in spec_ranking:
        s = spec_stats[lang]
        print(f"    {lang}: mean={s['mean']:.6f}, std={s['std']:.6f}, rank={s['rank']}")

    print("\n  D.3 Char N-Gram ranking:")
    for lang in ngram_ranking:
        s = ngram_stats[lang]
        print(f"    {lang}: mean={s['mean']:.6f}, std={s['std']:.6f}, rank={s['rank']}")

    # ------------------------------------------------------------------
    # 5. Consensus (Borda count)
    # ------------------------------------------------------------------
    print("\n--- Step 5: Consensus ranking (Borda count) ---")

    borda: Dict[str, float] = {}
    for lang in languages:
        total_rank = (
            gw_stats[lang]['rank']
            + spec_stats[lang]['rank']
            + ngram_stats[lang]['rank']
        )
        borda[lang] = total_rank / 3.0

    consensus_ranking = sorted(languages, key=lambda l: borda[l])
    consensus_list: List[Dict] = []
    for lang in consensus_ranking:
        consensus_list.append({
            'language': lang,
            'mean_rank': round(borda[lang], 2),
        })
        print(f"  {lang}: mean_rank={borda[lang]:.2f}")

    top_language = consensus_ranking[0]
    if len(consensus_ranking) >= 2:
        margin = borda[consensus_ranking[1]] - borda[consensus_ranking[0]]
    else:
        margin = 0.0

    print(f"\n  Top language: {top_language}")
    print(f"  Margin to 2nd: {margin:.2f}")

    # ------------------------------------------------------------------
    # 6. Verdict
    # ------------------------------------------------------------------
    if top_language in ('latin', 'italian') and margin > 0.5:
        verdict = "ROMANCE_CONFIRMED"
    elif top_language in ('latin', 'italian') and margin <= 0.5:
        verdict = "ROMANCE_MARGINAL"
    elif top_language == 'german':
        verdict = "GERMAN_PERSISTS"
    else:
        verdict = "UNEXPECTED"

    print(f"  Verdict: {verdict}")

    runtime = time.time() - t0
    print(f"  Runtime: {runtime:.1f}s")

    # ------------------------------------------------------------------
    # 7. Save results
    # ------------------------------------------------------------------
    result = SizeMatchedLangIDResult(
        d1_gw=gw_stats,
        d2_spectral=spec_stats,
        d3_ngram=ngram_stats,
        consensus_ranking=consensus_list,
        top_language=top_language,
        margin=round(margin, 4),
        verdict=verdict,
        runtime_seconds=round(runtime, 2),
    )

    out_path = _save_json(rd, 'size_matched_langid.json', asdict(result))
    print(f"\n  Saved: {out_path}")

    print("\n" + "=" * 70)
    print("Phase 50 Track D complete")
    print(f"  Top language: {top_language}")
    print(f"  Margin:       {margin:.2f}")
    print(f"  Verdict:      {verdict}")
    print("=" * 70)

    return asdict(result)
