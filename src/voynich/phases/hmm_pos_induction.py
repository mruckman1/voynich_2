"""
Phase 18.4 – Unsupervised HMM POS Induction
=============================================

Trains a K=8-state categorical Hidden Markov Model on the Voynich
token sequence using Baum-Welch EM.  The 8×8 transition matrix
reveals whether the text has grammar-like sequential structure.

  H1 (Hoax)       → rigid, low-entropy transitions (table → table)
  H2 (Cipher)     → complex, moderately sparse transitions
                     (comparable to Latin)
  H3 (Taxonomic)  → structured but not uniform; specific patterns

Pure numpy implementation — no hmmlearn required.
Forward-backward uses per-timestep scaling (Rabiner 1989).

Dependency chain:
    (none — reads corpus directly)
        -> hmm_pos_induction.json
"""

import json
import math
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import load_corpus
from voynich.core.reference import load_reference_corpus
from voynich.core.stats import jensen_shannon_divergence


# ---------------------------------------------------------------------------
# JSON serialiser
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
    if isinstance(obj, (bool, int, float, str, type(None))):
        return obj
    return str(obj)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class HMMPosResult:
    vocab_size: int
    n_states: int
    n_training_tokens: int
    final_log_likelihood: float
    n_em_iterations: int
    transition_matrix: List[List[float]]
    transition_entropy_mean: float
    transition_entropy_std: float
    transition_sparsity: float         # fraction of A entries < 0.05
    dominant_transition_fraction: float  # fraction of argmax transitions
    emission_entropy_mean: float
    latin_transition_entropy_mean: Optional[float]
    latin_transition_sparsity: Optional[float]
    voynich_latin_transition_jsd: Optional[float]
    hypothesis_support: Dict[str, float]
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


def _encode_tokens(tokens: List[str], vocab: List[str]) -> np.ndarray:
    """Map tokens to integer ids.  Unknown tokens → len(vocab) (UNK)."""
    tok_to_id = {w: i for i, w in enumerate(vocab)}
    unk = len(vocab)
    return np.array([tok_to_id.get(t, unk) for t in tokens], dtype=np.int32)


def _initialize_hmm(
    K: int, V: int, seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random initialisation of HMM parameters.

    Returns (pi, A, B) where:
      pi: (K,) initial distribution
      A:  (K, K) transition matrix
      B:  (K, V) emission matrix
    """
    rng = np.random.default_rng(seed)

    pi = rng.dirichlet(np.ones(K))

    A = np.empty((K, K))
    for i in range(K):
        A[i] = rng.dirichlet(np.ones(K))

    B = np.empty((K, V))
    for i in range(K):
        B[i] = rng.dirichlet(np.ones(V))

    return pi, A, B


def _forward(
    obs: np.ndarray, pi: np.ndarray, A: np.ndarray, B: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Scaled forward pass.

    Returns (alpha, scales) where:
      alpha: (T, K)  scaled forward probabilities
      scales: (T,)   normalisation constants
    """
    T = len(obs)
    K = len(pi)
    alpha = np.zeros((T, K))
    scales = np.zeros(T)

    alpha[0] = pi * B[:, obs[0]]
    scales[0] = alpha[0].sum()
    if scales[0] > 0:
        alpha[0] /= scales[0]

    for t in range(1, T):
        alpha[t] = B[:, obs[t]] * (alpha[t - 1] @ A)
        scales[t] = alpha[t].sum()
        if scales[t] > 0:
            alpha[t] /= scales[t]

    return alpha, scales


def _backward(
    obs: np.ndarray, A: np.ndarray, B: np.ndarray, scales: np.ndarray,
) -> np.ndarray:
    """Scaled backward pass.  Returns beta: (T, K)."""
    T = len(obs)
    K = A.shape[0]
    beta = np.zeros((T, K))
    beta[T - 1] = 1.0

    for t in range(T - 2, -1, -1):
        beta[t] = A @ (B[:, obs[t + 1]] * beta[t + 1])
        if scales[t + 1] > 0:
            beta[t] /= scales[t + 1]

    return beta


def _baum_welch(
    obs: np.ndarray,
    K: int,
    V: int,
    max_iter: int = 100,
    tol: float = 1e-4,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
    """Baum-Welch EM.

    Returns (pi, A, B, log_likelihood, n_iterations).
    """
    pi, A, B = _initialize_hmm(K, V, seed)
    T = len(obs)
    prev_ll = -np.inf

    for iteration in range(1, max_iter + 1):
        # E step
        alpha, scales = _forward(obs, pi, A, B)
        beta = _backward(obs, A, B, scales)

        # Log-likelihood
        log_ll = float(np.sum(np.log(scales + 1e-300)))

        # Convergence check
        if abs(log_ll - prev_ll) < tol:
            return pi, A, B, log_ll, iteration
        prev_ll = log_ll

        # Gamma: (T, K) — posterior P(z_t = k | obs)
        gamma = alpha * beta
        gamma_sum = gamma.sum(axis=1, keepdims=True)
        gamma_sum = np.where(gamma_sum > 0, gamma_sum, 1.0)
        gamma = gamma / gamma_sum

        # Xi: (T-1, K, K) — posterior P(z_t=i, z_{t+1}=j | obs)
        # Computed in vectorised form using (K, K) accumulation
        xi_sum = np.zeros((K, K))
        for t in range(T - 1):
            numer = np.outer(alpha[t], B[:, obs[t + 1]] * beta[t + 1]) * A
            denom = numer.sum()
            if denom > 0:
                xi_sum += numer / denom

        # M step
        pi = gamma[0] / gamma[0].sum()

        # Transition
        gamma_t_sum = gamma[:-1].sum(axis=0)  # (K,)
        gamma_t_sum = np.where(gamma_t_sum > 0, gamma_t_sum, 1.0)
        A = xi_sum / gamma_t_sum[:, None]
        # Normalise rows
        row_sums = A.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > 0, row_sums, 1.0)
        A = A / row_sums

        # Emission
        B_new = np.zeros((K, V))
        for t in range(T):
            B_new[:, obs[t]] += gamma[t]
        b_row = B_new.sum(axis=1, keepdims=True)
        b_row = np.where(b_row > 0, b_row, 1.0)
        B = B_new / b_row

    return pi, A, B, prev_ll, max_iter


def _row_entropy(matrix: np.ndarray) -> np.ndarray:
    """Entropy of each row of a probability matrix (bits)."""
    eps = 1e-12
    return -np.sum(matrix * np.log2(matrix + eps), axis=1)


def _matrix_sparsity(matrix: np.ndarray, threshold: float = 0.05) -> float:
    """Fraction of entries below *threshold*."""
    return float((matrix < threshold).sum()) / matrix.size


def _dominant_fraction(matrix: np.ndarray) -> float:
    """Fraction of probability mass in the argmax entry per row, averaged."""
    return float(np.mean(np.max(matrix, axis=1)))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_hmm_pos_induction() -> None:
    """Phase 18.4: unsupervised HMM POS induction."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE 18.4: Unsupervised HMM POS Induction")
    print("=" * 70)

    rd = _results_dir()
    K = 8           # number of hidden states
    TOP_N = 500     # vocabulary size

    # ── 1. Voynich data ───────────────────────────────────────────────
    print("\n  1. Preparing Voynich token sequence …")
    corpus = load_corpus(verbose=False)
    tokens_a = corpus.get_tokens(language='A', paragraph_only=True)
    counts = Counter(tokens_a)
    vocab = [w for w, _ in counts.most_common(TOP_N)]
    V = len(vocab) + 1  # +1 for UNK
    obs = _encode_tokens(tokens_a, vocab)
    print(f"     {len(obs):,} tokens  |  V={V} (top {TOP_N} + UNK)")

    # ── 2. Baum-Welch on Voynich (5 random inits) ────────────────────
    print("\n  2. Training HMM (K=8, 5 initialisations) …")
    best_ll = -np.inf
    best_result = None
    for s in range(5):
        seed = 42 + s
        pi, A, B, ll, n_iter = _baum_welch(obs, K, V, max_iter=100, seed=seed)
        print(f"     seed={seed}  LL={ll:.1f}  iters={n_iter}")
        if ll > best_ll:
            best_ll = ll
            best_result = (pi, A, B, ll, n_iter)

    pi_best, A_best, B_best, ll_best, iter_best = best_result

    # ── 3. Transition matrix analysis ─────────────────────────────────
    print("\n  3. Analysing transition matrix …")
    t_entropy = _row_entropy(A_best)
    t_entropy_mean = float(np.mean(t_entropy))
    t_entropy_std = float(np.std(t_entropy))
    t_sparsity = _matrix_sparsity(A_best)
    t_dominant = _dominant_fraction(A_best)

    e_entropy = _row_entropy(B_best)
    e_entropy_mean = float(np.mean(e_entropy))

    print(f"     Transition entropy: mean={t_entropy_mean:.3f} ± {t_entropy_std:.3f}")
    print(f"     Transition sparsity: {t_sparsity:.3f}")
    print(f"     Dominant fraction: {t_dominant:.3f}")
    print(f"     Emission entropy: mean={e_entropy_mean:.3f}")

    # ── 4. Latin comparison ───────────────────────────────────────────
    print("\n  4. Latin HMM comparison …")
    latin_t_ent: Optional[float] = None
    latin_t_spar: Optional[float] = None
    v_l_jsd: Optional[float] = None
    try:
        ref = load_reference_corpus(languages=['latin'], verbose=False)
        lat_tokens = ref.get_combined_tokens('latin')
        if lat_tokens and len(lat_tokens) > 500:
            lat_counts = Counter(lat_tokens)
            lat_vocab = [w for w, _ in lat_counts.most_common(TOP_N)]
            lat_V = len(lat_vocab) + 1
            lat_obs = _encode_tokens(lat_tokens, lat_vocab)

            # Single run (Latin is control, not optimised)
            _, A_lat, _, _, _ = _baum_welch(lat_obs, K, lat_V, max_iter=80, seed=42)
            lat_ent = _row_entropy(A_lat)
            latin_t_ent = float(np.mean(lat_ent))
            latin_t_spar = _matrix_sparsity(A_lat)
            print(f"     Latin transition entropy: {latin_t_ent:.3f}")
            print(f"     Latin transition sparsity: {latin_t_spar:.3f}")

            # JSD between flattened transition matrices
            v_l_jsd = float(jensen_shannon_divergence(A_best.flatten(), A_lat.flatten()))
            print(f"     Voynich-Latin transition JSD: {v_l_jsd:.4f}")
    except Exception as e:
        print(f"     WARNING: Latin HMM unavailable ({e})")

    # ── 5. Hypothesis scoring ─────────────────────────────────────────
    print("\n  5. Scoring hypotheses …")
    # H1: low transition entropy → rigid table
    h1 = _sigmoid(-(t_entropy_mean - 1.0) / 0.5)
    # H2: entropy comparable to Latin
    ref_ent = latin_t_ent if latin_t_ent is not None else 2.0
    h2 = _sigmoid(-abs(t_entropy_mean - ref_ent) / 0.4)
    # H3: moderate entropy + high sparsity
    h3 = _sigmoid((t_sparsity - 0.3) / 0.2) * _sigmoid((t_entropy_mean - 1.5) / 0.5)

    total = h1 + h2 + h3
    if total > 0:
        h1, h2, h3 = h1 / total, h2 / total, h3 / total

    hypothesis_support = {'H1': round(h1, 4), 'H2': round(h2, 4), 'H3': round(h3, 4)}
    print(f"     H1={h1:.3f}  H2={h2:.3f}  H3={h3:.3f}")

    # ── Verdict ───────────────────────────────────────────────────────
    if t_entropy_mean < 1.2:
        verdict = (f"RIGID TRANSITIONS: mean entropy = {t_entropy_mean:.3f} bits — "
                   "low, consistent with deterministic table generator (H1).")
    elif latin_t_ent is not None and abs(t_entropy_mean - latin_t_ent) < 0.4:
        verdict = (f"GRAMMAR-LIKE: transition entropy = {t_entropy_mean:.3f} matches "
                   f"Latin ({latin_t_ent:.3f}). Consistent with natural-language "
                   "syntax (H2/H3).")
    elif t_sparsity > 0.5:
        verdict = (f"SPARSE GRAMMAR: sparsity = {t_sparsity:.3f} with moderate entropy "
                   f"({t_entropy_mean:.3f}). May indicate a structured taxonomic "
                   "grammar (H3).")
    else:
        verdict = (f"MIXED: transition entropy = {t_entropy_mean:.3f}, sparsity = "
                   f"{t_sparsity:.3f}. No clear single-hypothesis match.")

    print(f"\n  Verdict: {verdict}")

    # ── Save ──────────────────────────────────────────────────────────
    result = HMMPosResult(
        vocab_size=V,
        n_states=K,
        n_training_tokens=len(obs),
        final_log_likelihood=round(ll_best, 2),
        n_em_iterations=iter_best,
        transition_matrix=[[round(float(x), 4) for x in row] for row in A_best],
        transition_entropy_mean=round(t_entropy_mean, 4),
        transition_entropy_std=round(t_entropy_std, 4),
        transition_sparsity=round(t_sparsity, 4),
        dominant_transition_fraction=round(t_dominant, 4),
        emission_entropy_mean=round(e_entropy_mean, 4),
        latin_transition_entropy_mean=round(latin_t_ent, 4) if latin_t_ent is not None else None,
        latin_transition_sparsity=round(latin_t_spar, 4) if latin_t_spar is not None else None,
        voynich_latin_transition_jsd=round(v_l_jsd, 4) if v_l_jsd is not None else None,
        hypothesis_support=hypothesis_support,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'hmm_pos_induction.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)

    print(f"\n  -> {out_path}")
