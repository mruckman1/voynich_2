"""
Phase 69, Track 3: Enhanced Distributional Mapping (200+ Anchors)
===================================================================
Phase 67 had 39 anchor pairs for Procrustes alignment.  The 223 T1
identifications provide up to 200+ anchors.  With more anchors, the
alignment is dramatically more reliable.

Requires Track 0 >= PARTIAL.

Dependency chain:
    results/p69_clean_corpus.json        (Step 0)
    results/p69_clean_validation.json    (Track 0, must be >= PARTIAL)
    results/combined_refine.json         (Phase 15)
    results/triple_tiers.json            (Phase 28/53)
        -> results/p69_clean_distrib.json
"""

import json
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Set, Tuple

import numpy as np
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    load_corpus,
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


def _get_confirmed_and_unresolved(rd: str) -> Tuple[Dict[str, str], Dict[str, str]]:
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


def _edit_distance(a: str, b: str) -> int:
    if len(a) < len(b):
        return _edit_distance(b, a)
    if len(b) == 0:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            cost = 0 if ca == cb else 1
            curr.append(min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost))
        prev = curr
    return prev[len(b)]


# ---------------------------------------------------------------------------
# PPMI + SVD distributional vectors
# ---------------------------------------------------------------------------

def _build_ppmi_vectors(tokens: List[str], window: int = 5,
                         min_count: int = 3, svd_dim: int = 50,
                         ) -> Tuple[np.ndarray, List[str], Dict[str, int]]:
    """Build PPMI co-occurrence matrix and reduce with SVD.

    Returns (vectors [V x dim], vocab list, vocab_to_idx).
    """
    # Build vocabulary
    counts = Counter(tokens)
    vocab = sorted([w for w, c in counts.items() if c >= min_count])
    vocab_to_idx = {w: i for i, w in enumerate(vocab)}
    V = len(vocab)

    if V < 10:
        return np.zeros((0, svd_dim)), [], {}

    # Build co-occurrence matrix
    cooc = Counter()
    for i, token in enumerate(tokens):
        if token not in vocab_to_idx:
            continue
        ti = vocab_to_idx[token]
        for j in range(max(0, i - window), min(len(tokens), i + window + 1)):
            if j == i:
                continue
            other = tokens[j]
            if other in vocab_to_idx:
                cooc[(ti, vocab_to_idx[other])] += 1

    # Build sparse PPMI matrix
    total = sum(cooc.values())
    if total == 0:
        return np.zeros((V, svd_dim)), vocab, vocab_to_idx

    row_sums = np.zeros(V)
    col_sums = np.zeros(V)
    for (i, j), c in cooc.items():
        row_sums[i] += c
        col_sums[j] += c

    rows, cols, data = [], [], []
    for (i, j), c in cooc.items():
        pmi = np.log2((c * total) / (row_sums[i] * col_sums[j] + 1e-10))
        ppmi = max(0.0, pmi)
        if ppmi > 0:
            rows.append(i)
            cols.append(j)
            data.append(ppmi)

    if not data:
        return np.zeros((V, svd_dim)), vocab, vocab_to_idx

    sparse_mat = csr_matrix((data, (rows, cols)), shape=(V, V))

    # SVD
    k = min(svd_dim, V - 1)
    if k < 1:
        return np.zeros((V, svd_dim)), vocab, vocab_to_idx

    U, S, _ = svds(sparse_mat.astype(float), k=k)
    vectors = U * np.sqrt(S)

    return vectors, vocab, vocab_to_idx


# ---------------------------------------------------------------------------
# Procrustes alignment
# ---------------------------------------------------------------------------

def _weighted_procrustes(source: np.ndarray, target: np.ndarray,
                          weights: np.ndarray) -> np.ndarray:
    """Weighted orthogonal Procrustes: find R minimizing ||W(source@R - target)||.

    Returns rotation matrix R.
    """
    W_sqrt = np.diag(np.sqrt(weights))
    M = (W_sqrt @ source).T @ (W_sqrt @ target)
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    return R


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class CleanDistribResult:
    phase: str = "69"
    step: str = "69.4"
    experiment: str = "clean_distributional"
    validation_status: str = ""
    # Vocabulary sizes
    n_eva_types: int = 0
    n_latin_types: int = 0
    svd_dim: int = 50
    # Anchors
    n_anchors: int = 0
    anchor_details: List[Dict[str, Any]] = field(default_factory=list)
    # Procrustes
    procrustes_error: float = 0.0
    # Matches
    n_clean_types_matched: int = 0
    convergent_count: int = 0
    total_tested: int = 0
    convergence_rate: float = 0.0
    top_matches: List[Dict[str, Any]] = field(default_factory=list)
    # Gates
    gate_cd1: bool = False    # >= 100 anchor pairs
    gate_cd2: bool = False    # convergence > 20%
    gate_cd3: bool = False    # >= 30 clean types matched
    gates_passed: int = 0
    gate_passed: bool = False
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_clean_distrib():
    """Track 3: Enhanced Procrustes with 200+ T1 anchors."""
    t0 = time.time()
    rd = str(_results_dir())

    print("Phase 69.4 — Enhanced Distributional Mapping")
    print("=" * 47)

    # --- Check validation gate ---
    val_data = _safe_load(os.path.join(rd, 'p69_clean_validation.json'))
    val_verdict = val_data.get('verdict', 'FAILED')
    if val_verdict == 'FAILED':
        print("  SKIPPED: Validation gate = FAILED")
        result = CleanDistribResult(
            validation_status='SKIPPED_FAILED_VALIDATION',
            runtime_seconds=round(time.time() - t0, 1),
        )
        _save_json(rd, 'p69_clean_distrib.json', result)
        return
    print(f"  Validation: {val_verdict}")

    # --- Load T1 catalogue ---
    clean_data = _safe_load(os.path.join(rd, 'p69_clean_corpus.json'))
    if not clean_data:
        print("  ERROR: p69_clean_corpus.json not found.")
        return

    t1_catalogue = clean_data.get('t1_catalogue', [])
    clean_indices = set(clean_data.get('clean_indices', []))
    clean_decoded = clean_data.get('clean_decoded', [])
    clean_idx_list = clean_data.get('clean_indices', [])
    print(f"  T1 words: {len(t1_catalogue)}")

    # --- Load corpus ---
    confirmed, unresolved = _get_confirmed_and_unresolved(rd)
    full_assignment = {**confirmed, **unresolved}
    eva_to_triple = build_eva_to_triple_lookup()
    coda_table = build_coda_table_v2()

    corpus = load_corpus(verbose=False)
    all_tokens = corpus.get_tokens()

    # --- Build EVA distributional vectors ---
    print("\n  Building EVA distributional vectors...")
    eva_vectors, eva_vocab, eva_vocab_idx = _build_ppmi_vectors(
        all_tokens, window=5, min_count=3, svd_dim=50)
    print(f"  EVA vocab: {len(eva_vocab)}, vectors: {eva_vectors.shape}")

    # --- Build Latin distributional vectors ---
    print("  Building Latin distributional vectors...")
    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    ref_tokens = [w.lower() for w in ref_corpus.get_combined_tokens('latin')
                  if len(w) >= 2]
    base_words = set(ref_tokens)
    expanded, _ = build_expanded_word_set(base_words)
    ref_word_set = base_words | expanded

    latin_vectors, latin_vocab, latin_vocab_idx = _build_ppmi_vectors(
        ref_tokens, window=5, min_count=5, svd_dim=50)
    print(f"  Latin vocab: {len(latin_vocab)}, vectors: {latin_vectors.shape}")

    # --- Build anchor pairs from T1 catalogue ---
    print("\n  Building anchor pairs...")
    anchors = []
    tier_weights = {'TIER_1': 1.0, 'TIER_2': 0.8, 'TIER_3': 0.6}

    for entry in t1_catalogue:
        eva_type = entry['eva_type']
        latin_word = entry['matched_word']
        tier = entry.get('tier', 'TIER_3')

        if eva_type in eva_vocab_idx and latin_word in latin_vocab_idx:
            anchors.append({
                'eva_type': eva_type,
                'latin_word': latin_word,
                'tier': tier,
                'weight': tier_weights.get(tier, 0.5),
                'eva_idx': eva_vocab_idx[eva_type],
                'latin_idx': latin_vocab_idx[latin_word],
            })

    n_anchors = len(anchors)
    print(f"  Anchor pairs: {n_anchors}")

    if n_anchors < 5:
        print("  ERROR: Too few anchors for Procrustes alignment.")
        result = CleanDistribResult(
            validation_status=val_verdict,
            n_eva_types=len(eva_vocab),
            n_latin_types=len(latin_vocab),
            n_anchors=n_anchors,
            runtime_seconds=round(time.time() - t0, 1),
        )
        _save_json(rd, 'p69_clean_distrib.json', result)
        return

    # --- Weighted Procrustes alignment ---
    print("  Running weighted Procrustes alignment...")
    eva_anchor_vecs = np.array([eva_vectors[a['eva_idx']] for a in anchors])
    latin_anchor_vecs = np.array([latin_vectors[a['latin_idx']] for a in anchors])
    weights = np.array([a['weight'] for a in anchors])

    R = _weighted_procrustes(eva_anchor_vecs, latin_anchor_vecs, weights)

    # Compute alignment error
    aligned_anchors = eva_anchor_vecs @ R
    errors = np.linalg.norm(aligned_anchors - latin_anchor_vecs, axis=1)
    mean_error = float(np.mean(errors))
    print(f"  Mean Procrustes error: {mean_error:.4f}")

    # --- Align all EVA vectors ---
    aligned_eva = eva_vectors @ R

    # Normalize for cosine similarity
    eva_norms = np.linalg.norm(aligned_eva, axis=1, keepdims=True) + 1e-10
    aligned_eva_normed = aligned_eva / eva_norms

    latin_norms = np.linalg.norm(latin_vectors, axis=1, keepdims=True) + 1e-10
    latin_normed = latin_vectors / latin_norms

    # --- Map clean EVA types to Latin ---
    print("\n  Mapping clean EVA types to Latin...")

    # Build CVC decode for each EVA type (for convergence check)
    idx_to_decoded_map: Dict[int, str] = {}
    for i, ci in enumerate(clean_idx_list):
        if i < len(clean_decoded):
            idx_to_decoded_map[ci] = clean_decoded[i]

    # Get most common decoded form for each EVA type in clean subset
    type_decoded_counts: Dict[str, Counter] = {}
    for i, ci in enumerate(clean_idx_list):
        token = all_tokens[ci]
        d = clean_decoded[i] if i < len(clean_decoded) else ''
        if d and '?' not in d:
            if token not in type_decoded_counts:
                type_decoded_counts[token] = Counter()
            type_decoded_counts[token][d] += 1

    type_to_decoded: Dict[str, str] = {}
    for t, counts in type_decoded_counts.items():
        type_to_decoded[t] = counts.most_common(1)[0][0]

    # Find nearest Latin neighbor for each clean EVA type
    clean_types = set(all_tokens[i] for i in clean_indices if all_tokens[i] in eva_vocab_idx)

    matches = []
    convergent = 0
    total_tested = 0

    for eva_type in sorted(clean_types):
        if eva_type not in eva_vocab_idx:
            continue

        ei = eva_vocab_idx[eva_type]
        eva_vec = aligned_eva_normed[ei]

        sims = latin_normed @ eva_vec
        top_indices = np.argsort(sims)[::-1][:5]
        top_matches_list = [(latin_vocab[j], float(sims[j])) for j in top_indices]

        cvc_decoded = type_to_decoded.get(eva_type, '')
        distrib_word = top_matches_list[0][0] if top_matches_list else ''

        is_convergent = False
        if cvc_decoded and distrib_word:
            total_tested += 1
            ed = _edit_distance(cvc_decoded, distrib_word)
            if ed <= 2:
                convergent += 1
                is_convergent = True

        matches.append({
            'eva_type': eva_type,
            'cvc_decoded': cvc_decoded,
            'top_latin': top_matches_list[:3],
            'convergent': is_convergent,
        })

    convergence_rate = convergent / total_tested if total_tested > 0 else 0.0
    n_matched = sum(1 for m in matches if m['top_latin'])

    print(f"  Clean types mapped: {n_matched}")
    print(f"  Convergent (CVC ≈ distributional): {convergent}/{total_tested} "
          f"({convergence_rate:.1%})")

    # --- Gates ---
    gate_cd1 = n_anchors >= 100
    gate_cd2 = convergence_rate > 0.20
    gate_cd3 = n_matched >= 30
    gates_passed = sum([gate_cd1, gate_cd2, gate_cd3])

    # Top convergent matches for display
    convergent_matches = [m for m in matches if m.get('convergent')]
    top_display = sorted(convergent_matches,
                        key=lambda m: m['top_latin'][0][1] if m['top_latin'] else 0,
                        reverse=True)[:20]

    result = CleanDistribResult(
        validation_status=val_verdict,
        n_eva_types=len(eva_vocab),
        n_latin_types=len(latin_vocab),
        svd_dim=50,
        n_anchors=n_anchors,
        anchor_details=[{
            'eva_type': a['eva_type'],
            'latin_word': a['latin_word'],
            'tier': a['tier'],
        } for a in anchors[:50]],
        procrustes_error=round(mean_error, 4),
        n_clean_types_matched=n_matched,
        convergent_count=convergent,
        total_tested=total_tested,
        convergence_rate=round(convergence_rate, 4),
        top_matches=top_display,
        gate_cd1=gate_cd1,
        gate_cd2=gate_cd2,
        gate_cd3=gate_cd3,
        gates_passed=gates_passed,
        gate_passed=gates_passed >= 2,
        runtime_seconds=round(time.time() - t0, 1),
    )

    path = _save_json(rd, 'p69_clean_distrib.json', result)

    print(f"\n  Summary")
    print(f"  -------")
    print(f"  Anchors:       {n_anchors} ({'PASS' if gate_cd1 else 'FAIL'} >= 100)")
    print(f"  Convergence:   {convergence_rate:.1%} ({'PASS' if gate_cd2 else 'FAIL'} > 20%)")
    print(f"  Types matched: {n_matched} ({'PASS' if gate_cd3 else 'FAIL'} >= 30)")
    print(f"  Gates: {gates_passed}/3")
    print(f"  Saved: {path}")
    print(f"  Time: {result.runtime_seconds:.1f}s")
