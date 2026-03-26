"""
Phase 67, Track 5: Direct EVA-to-Latin Distributional Mapping
===============================================================
Build distributional (PPMI + SVD) vectors for EVA token types and
Latin word types.  Align the two spaces via Procrustes using signal
words as anchors.  For each EVA token containing unresolved triples,
find the nearest Latin words and extract syllable constraints.

Dependency chain:
    results/combined_refine.json      (Phase 15)
    results/triple_tiers.json         (Phase 28/53)
    data/reference/latin/             (Latin reference corpora)
        -> results/p67_distributional.json
"""

import json
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    load_corpus,
    tokenize_eva_chars,
)
from voynich.core.reference import build_expanded_word_set, load_reference_corpus
from voynich.phases.corrected_coda import (
    build_coda_table_v2,
    decode_token_cvc_v2,
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
    if isinstance(obj, set):
        return sorted(obj)
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
# Confirmed / unresolved triple separation
# ---------------------------------------------------------------------------

def _get_confirmed_and_unresolved(
    rd: str,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Return (confirmed_12, unresolved_13)."""
    refine_data = _safe_load(os.path.join(rd, 'combined_refine.json'))
    assignment = refine_data.get('best_assignment', {})

    tier_data = _safe_load(os.path.join(rd, 'triple_tiers.json'))
    confirmed_keys: Set[str] = set()

    if tier_data and 'tiers' in tier_data:
        tiers = tier_data['tiers']
        if isinstance(tiers, dict):
            for entry in tiers.get('CONFIRMED', []):
                confirmed_keys.add(entry.get('triple_key', ''))
        elif isinstance(tiers, list):
            for entry in tiers:
                if entry.get('tier', '') == 'CONFIRMED':
                    confirmed_keys.add(entry.get('triple_key', ''))

    if not confirmed_keys:
        return dict(assignment), {}

    confirmed = {k: v for k, v in assignment.items() if k in confirmed_keys}
    unresolved = {k: v for k, v in assignment.items() if k not in confirmed_keys}
    return confirmed, unresolved


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DistributionalResult:
    phase: str = "67"
    step: str = "67.5"
    experiment: str = "distributional_mapping"
    # Corpus stats
    eva_vocab_size: int = 0
    latin_vocab_size: int = 0
    svd_dim: int = 0
    # Anchor pairs
    n_anchor_pairs: int = 0
    anchor_pairs: List[Dict[str, str]] = field(default_factory=list)
    procrustes_residual: float = 0.0
    mean_cosine_anchors: float = 0.0
    # Validation
    n_tested: int = 0
    n_exact_hit: int = 0
    n_related_hit: int = 0
    exact_hit_rate: float = 0.0
    related_hit_rate: float = 0.0
    # Per-token matches (sample)
    sample_matches: List[Dict[str, Any]] = field(default_factory=list)
    # Triple constraints derived
    triple_candidates: Dict[str, List[str]] = field(default_factory=dict)
    # Gates
    g1_enough_anchors: bool = False   # D1: >= 10 anchors
    g2_exact_hit: bool = False        # D2: exact hit rate > 10%
    g3_related_hit: bool = False      # D3: related hit rate > 25%
    g4_convergence: bool = False      # D4: >= 5 tokens converge
    gates_passed: int = 0
    gate_passed: bool = False
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# PPMI + SVD
# ---------------------------------------------------------------------------

def _build_ppmi_svd(
    tokens: List[str],
    min_freq: int = 10,
    window: int = 5,
    svd_k: int = 30,
) -> Tuple[np.ndarray, List[str], Dict[str, int]]:
    """Build PPMI co-occurrence matrix and reduce via SVD.

    Returns (vectors, vocab, word_to_idx).
    """
    # Count frequencies
    type_counts = Counter(tokens)
    frequent = sorted(t for t, c in type_counts.items() if c >= min_freq)
    word_to_idx = {w: i for i, w in enumerate(frequent)}
    n = len(frequent)

    if n < svd_k + 2:
        svd_k = max(2, n - 2)

    # Build co-occurrence matrix
    cooc = np.zeros((n, n), dtype=np.float64)
    for i, token in enumerate(tokens):
        if token not in word_to_idx:
            continue
        t_idx = word_to_idx[token]
        for j in range(max(0, i - window), min(len(tokens), i + window + 1)):
            if i == j:
                continue
            ctx = tokens[j]
            if ctx not in word_to_idx:
                continue
            c_idx = word_to_idx[ctx]
            cooc[t_idx, c_idx] += 1

    # PPMI
    row_sums = cooc.sum(axis=1, keepdims=True) + 1e-10
    col_sums = cooc.sum(axis=0, keepdims=True) + 1e-10
    total = cooc.sum() + 1e-10

    expected = (row_sums * col_sums) / total
    # Only compute PMI where cooc > 0 to avoid log(0)
    with np.errstate(divide='ignore', invalid='ignore'):
        pmi = np.where(cooc > 0, np.log2(cooc / expected + 1e-10), 0.0)
    ppmi = np.maximum(pmi, 0.0)

    # SVD
    from scipy.sparse.linalg import svds
    from scipy.sparse import csr_matrix

    sparse_ppmi = csr_matrix(ppmi)
    try:
        U, S, Vt = svds(sparse_ppmi, k=svd_k)
        vectors = U * np.sqrt(S)
    except Exception:
        # Fallback: dense SVD
        U, S, Vt = np.linalg.svd(ppmi, full_matrices=False)
        vectors = U[:, :svd_k] * np.sqrt(S[:svd_k])

    return vectors, frequent, word_to_idx


# ---------------------------------------------------------------------------
# Anchor construction
# ---------------------------------------------------------------------------

def _build_anchor_pairs(
    eva_tokens: List[str],
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    coda_table,
    eva_word_to_idx: Dict[str, int],
    latin_word_to_idx: Dict[str, int],
    ref_word_set: Set[str],
) -> List[Tuple[int, int, str, str]]:
    """Build anchor pairs from EVA tokens whose decoded form is in the dictionary.

    For each EVA token type:
    1. Decode it to a Latin string
    2. If the decoded string is in ref_word_set AND in the Latin vocab
    3. Create an anchor pair (eva_idx, latin_idx)

    Returns list of (eva_idx, latin_idx, eva_token, latin_word).
    """
    anchors = []
    seen_pairs: Set[Tuple[int, int]] = set()

    # Build decode cache: eva_token_type -> decoded_cvc string
    eva_types = set(eva_tokens)
    decode_cache: Dict[str, str] = {}
    for token in eva_types:
        result = decode_token_cvc_v2(token, assignment, eva_to_triple, coda_table)
        decoded = result.decoded_cvc
        if decoded and '?' not in decoded:
            decode_cache[token] = decoded

    for eva_token, decoded in decode_cache.items():
        if eva_token not in eva_word_to_idx:
            continue
        if decoded not in latin_word_to_idx:
            continue
        if decoded not in ref_word_set:
            continue

        eva_idx = eva_word_to_idx[eva_token]
        latin_idx = latin_word_to_idx[decoded]
        pair = (eva_idx, latin_idx)
        if pair not in seen_pairs:
            seen_pairs.add(pair)
            anchors.append((eva_idx, latin_idx, eva_token, decoded))

    return anchors


# ---------------------------------------------------------------------------
# Procrustes alignment
# ---------------------------------------------------------------------------

def _procrustes_align(
    eva_vectors: np.ndarray,
    latin_vectors: np.ndarray,
    anchor_pairs: List[Tuple[int, int, str, str]],
) -> Tuple[Optional[np.ndarray], float]:
    """Align EVA vector space to Latin space via Procrustes.

    Returns (aligned_eva_vectors, residual).
    Returns (None, -1) if alignment fails.
    """
    if len(anchor_pairs) < 3:
        return None, -1.0

    from scipy.linalg import orthogonal_procrustes

    eva_indices = [p[0] for p in anchor_pairs]
    latin_indices = [p[1] for p in anchor_pairs]

    # Ensure vectors have same dimensionality
    dim_eva = eva_vectors.shape[1]
    dim_latin = latin_vectors.shape[1]
    target_dim = min(dim_eva, dim_latin)

    X = eva_vectors[eva_indices, :target_dim].copy()
    Y = latin_vectors[latin_indices, :target_dim].copy()

    # Center
    X -= X.mean(axis=0)
    Y -= Y.mean(axis=0)

    # Normalize
    X_norm = np.linalg.norm(X)
    Y_norm = np.linalg.norm(Y)
    if X_norm > 1e-10:
        X /= X_norm
    if Y_norm > 1e-10:
        Y /= Y_norm

    try:
        R, scale = orthogonal_procrustes(X, Y)
        residual = float(np.sum((X @ R - Y) ** 2))
    except Exception:
        return None, -1.0

    # Transform all EVA vectors
    aligned = eva_vectors[:, :target_dim] @ R

    return aligned, residual


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    dot = np.dot(a, b)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(dot / (na * nb))


# ---------------------------------------------------------------------------
# Nearest neighbor matching
# ---------------------------------------------------------------------------

def _find_nearest_matches(
    aligned_eva: np.ndarray,
    latin_vectors: np.ndarray,
    eva_vocab: List[str],
    latin_vocab: List[str],
    eva_word_to_idx: Dict[str, int],
    top_n: int = 10,
) -> Dict[str, List[Tuple[str, float]]]:
    """For each EVA token, find closest Latin words in aligned space.

    Returns {eva_token: [(latin_word, cosine_sim), ...]}.
    """
    target_dim = min(aligned_eva.shape[1], latin_vectors.shape[1])
    aligned = aligned_eva[:, :target_dim]
    latin = latin_vectors[:, :target_dim]

    # Normalize for cosine
    aligned_norm = aligned / (np.linalg.norm(aligned, axis=1, keepdims=True) + 1e-10)
    latin_norm = latin / (np.linalg.norm(latin, axis=1, keepdims=True) + 1e-10)

    # Compute similarity in batches to avoid memory issues
    matches: Dict[str, List[Tuple[str, float]]] = {}
    batch_size = 500
    n_eva = len(eva_vocab)

    for start in range(0, n_eva, batch_size):
        end = min(start + batch_size, n_eva)
        batch = aligned_norm[start:end]
        sims = batch @ latin_norm.T  # (batch_size, n_latin)

        for i in range(end - start):
            eva_token = eva_vocab[start + i]
            sorted_indices = np.argsort(sims[i])[::-1][:top_n]
            top_matches = [(latin_vocab[j], float(sims[i, j]))
                           for j in sorted_indices]
            matches[eva_token] = top_matches

    return matches


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_distrib_map():
    """Track 5: Direct EVA-to-Latin distributional mapping."""
    t0 = time.time()
    rd = str(_results_dir())
    svd_k = 30

    print("Phase 67.5 — Distributional EVA-to-Latin Mapping")
    print("=" * 52)

    # --- Load data ---
    confirmed, unresolved = _get_confirmed_and_unresolved(rd)
    full_assignment = {**confirmed, **unresolved}
    print(f"  Confirmed: {len(confirmed)}, Unresolved: {len(unresolved)}")

    eva_to_triple = build_eva_to_triple_lookup()
    coda_table = build_coda_table_v2()

    corpus = load_corpus(verbose=False)
    eva_tokens = corpus.get_tokens()
    print(f"  EVA tokens: {len(eva_tokens)}")

    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    latin_tokens = [w.lower() for w in ref_corpus.get_combined_tokens('latin')
                    if len(w) >= 2]
    print(f"  Latin tokens: {len(latin_tokens)}")

    base_words = set(latin_tokens)
    expanded, _ = build_expanded_word_set(base_words)
    ref_word_set = base_words | expanded

    # --- Build EVA vectors ---
    print("\n  Building EVA distributional vectors...")
    eva_vectors, eva_vocab, eva_w2i = _build_ppmi_svd(
        eva_tokens, min_freq=10, window=5, svd_k=svd_k)
    print(f"  EVA vocab: {len(eva_vocab)}, dim: {eva_vectors.shape[1]}")

    # --- Build Latin vectors ---
    print("  Building Latin distributional vectors...")
    latin_vectors, latin_vocab, latin_w2i = _build_ppmi_svd(
        latin_tokens, min_freq=5, window=5, svd_k=svd_k)
    print(f"  Latin vocab: {len(latin_vocab)}, dim: {latin_vectors.shape[1]}")

    # --- Build anchor pairs ---
    print("  Building anchor pairs...")
    anchors = _build_anchor_pairs(
        eva_tokens, full_assignment, eva_to_triple, coda_table,
        eva_w2i, latin_w2i, ref_word_set)
    print(f"  Anchor pairs: {len(anchors)}")

    anchor_details = [{'eva': a[2], 'latin': a[3]} for a in anchors[:30]]

    if len(anchors) < 3:
        print("  ERROR: Too few anchor pairs for Procrustes alignment.")
        result = DistributionalResult(
            eva_vocab_size=len(eva_vocab),
            latin_vocab_size=len(latin_vocab),
            svd_dim=svd_k,
            n_anchor_pairs=len(anchors),
            anchor_pairs=anchor_details,
            runtime_seconds=round(time.time() - t0, 1),
        )
        _save_json(rd, 'p67_distributional.json', result)
        return

    # --- Procrustes alignment ---
    print("  Aligning via Procrustes...")
    aligned_eva, residual = _procrustes_align(
        eva_vectors, latin_vectors, anchors)

    if aligned_eva is None:
        print("  ERROR: Procrustes alignment failed.")
        result = DistributionalResult(
            eva_vocab_size=len(eva_vocab),
            latin_vocab_size=len(latin_vocab),
            svd_dim=svd_k,
            n_anchor_pairs=len(anchors),
            anchor_pairs=anchor_details,
            procrustes_residual=-1.0,
            runtime_seconds=round(time.time() - t0, 1),
        )
        _save_json(rd, 'p67_distributional.json', result)
        return

    print(f"  Procrustes residual: {residual:.4f}")

    # Measure anchor quality
    target_dim = min(aligned_eva.shape[1], latin_vectors.shape[1])
    cosines = []
    for eva_idx, latin_idx, _, _ in anchors:
        cs = _cosine_sim(aligned_eva[eva_idx, :target_dim],
                         latin_vectors[latin_idx, :target_dim])
        cosines.append(cs)
    mean_cosine = float(np.mean(cosines)) if cosines else 0.0
    print(f"  Mean anchor cosine: {mean_cosine:.4f}")

    # --- Find nearest matches ---
    print("  Finding nearest Latin matches for EVA tokens...")
    all_matches = _find_nearest_matches(
        aligned_eva, latin_vectors, eva_vocab, latin_vocab, eva_w2i, top_n=10)

    # --- Validate against signal words ---
    from voynich.phases.suffix_calibration import SIGNAL_WORDS_51
    signal_set = set(SIGNAL_WORDS_51.keys())

    # Decode each EVA token type to see which ones produce signal words
    eva_type_to_decoded: Dict[str, str] = {}
    for token_type in eva_vocab:
        result_obj = decode_token_cvc_v2(
            token_type, full_assignment, eva_to_triple, coda_table)
        decoded = result_obj.decoded_cvc
        if decoded and '?' not in decoded:
            eva_type_to_decoded[token_type] = decoded

    n_tested = 0
    n_exact = 0
    n_related = 0
    sample_matches_out = []

    for eva_token, decoded in eva_type_to_decoded.items():
        if decoded not in signal_set:
            continue
        if eva_token not in all_matches:
            continue

        n_tested += 1
        top10 = [m[0] for m in all_matches[eva_token][:10]]

        exact = decoded in top10
        related = any(_edit_distance(decoded, w) <= 2 for w in top10)

        if exact:
            n_exact += 1
        if related:
            n_related += 1

        if n_tested <= 20:
            sample_matches_out.append({
                'eva_token': eva_token,
                'known_decoded': decoded,
                'in_top_10': exact,
                'related_in_top_10': related,
                'top_5': all_matches[eva_token][:5],
            })

    exact_rate = n_exact / n_tested if n_tested > 0 else 0.0
    related_rate = n_related / n_tested if n_tested > 0 else 0.0
    print(f"  Signal word validation: {n_tested} tested, "
          f"{n_exact} exact ({exact_rate:.1%}), "
          f"{n_related} related ({related_rate:.1%})")

    # --- Extract triple constraints from distributional matches ---
    # For EVA tokens with unresolved triples, check if distributional
    # matches suggest specific syllable values
    triple_candidates: Dict[str, Counter] = {}
    confirmed_keys = set(confirmed.keys())

    for eva_token in eva_vocab:
        if eva_token not in all_matches or eva_token not in eva_type_to_decoded:
            continue

        # Check if this token contains unresolved triples
        eva_chars = tokenize_eva_chars(eva_token)
        has_unresolved = False
        for ch in eva_chars:
            triple = eva_to_triple.get(ch, '')
            if triple and triple not in confirmed_keys:
                has_unresolved = True
                break

        if not has_unresolved:
            continue

        # Top distributional match
        top_match = all_matches[eva_token][0][0] if all_matches[eva_token] else None
        if not top_match:
            continue

        # We can't easily decompose which part of the match corresponds to
        # which triple, but we record the top match for the integration step
        for ch in eva_chars:
            triple = eva_to_triple.get(ch, '')
            if triple and triple not in confirmed_keys:
                if triple not in triple_candidates:
                    triple_candidates[triple] = Counter()
                triple_candidates[triple][top_match] += 1

    triple_cand_out = {}
    for tk, counter in triple_candidates.items():
        triple_cand_out[tk] = [w for w, _ in counter.most_common(10)]

    # --- Gates ---
    g1 = len(anchors) >= 10
    g2 = exact_rate > 0.10
    g3 = related_rate > 0.25
    g4 = n_related >= 5
    gates_passed = sum([g1, g2, g3, g4])

    result = DistributionalResult(
        eva_vocab_size=len(eva_vocab),
        latin_vocab_size=len(latin_vocab),
        svd_dim=svd_k,
        n_anchor_pairs=len(anchors),
        anchor_pairs=anchor_details,
        procrustes_residual=round(residual, 4),
        mean_cosine_anchors=round(mean_cosine, 4),
        n_tested=n_tested,
        n_exact_hit=n_exact,
        n_related_hit=n_related,
        exact_hit_rate=round(exact_rate, 4),
        related_hit_rate=round(related_rate, 4),
        sample_matches=sample_matches_out,
        triple_candidates=triple_cand_out,
        g1_enough_anchors=g1,
        g2_exact_hit=g2,
        g3_related_hit=g3,
        g4_convergence=g4,
        gates_passed=gates_passed,
        gate_passed=gates_passed >= 2,
        runtime_seconds=round(time.time() - t0, 1),
    )

    path = _save_json(rd, 'p67_distributional.json', result)

    # --- Summary ---
    print(f"\n  Summary")
    print(f"  -------")
    print(f"  Anchors:         {len(anchors)} ({'PASS' if g1 else 'FAIL'} >= 10)")
    print(f"  Procrustes res:  {residual:.4f}")
    print(f"  Mean anchor cos: {mean_cosine:.4f}")
    print(f"  Exact hit rate:  {exact_rate:.1%} ({'PASS' if g2 else 'FAIL'} > 10%)")
    print(f"  Related hit rate: {related_rate:.1%} ({'PASS' if g3 else 'FAIL'} > 25%)")
    print(f"  Related hits:    {n_related} ({'PASS' if g4 else 'FAIL'} >= 5)")
    print(f"  Gates: {gates_passed}/4")
    print(f"  Saved: {path}")
    print(f"  Time: {result.runtime_seconds:.1f}s")


def _edit_distance(a: str, b: str) -> int:
    """Levenshtein edit distance."""
    if len(a) < len(b):
        return _edit_distance(b, a)
    if len(b) == 0:
        return len(a)

    prev = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        curr = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr

    return prev[len(b)]
