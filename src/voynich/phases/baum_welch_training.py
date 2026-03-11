"""
Step 43.12 – Baum-Welch Training
===================================
Train the HMM parameters (transition, emission, initial) from the corpus
using anchor-constrained Baum-Welch EM.

Pure numpy implementation with per-timestep scaling (Rabiner 1989).
Anchored positions are hard-clamped during the E-step: gamma[t,k]=1.0
for the anchored state, 0 for all others.

Dependency chain:
    results/hmm_architecture.json     (Step 43.10: initialized params)
    results/anchor_initialization.json (Step 43.11: anchor mask)
    data/corpus/                       (EVA transcription)
        → baum_welch_training.json     (this step)
"""

import json
import os
import time
from collections import Counter
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    decode_token_modifier_aware,
    load_corpus,
    tokenize_eva_chars,
)
from voynich.core.stats import jensen_shannon_divergence


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
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if v != v else v
    if isinstance(obj, np.integer):
        return int(obj)
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
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class BaumWelchResult:
    n_tokens: int
    n_char_positions: int
    n_hidden_states: int
    n_observation_types: int
    n_anchored_positions: int
    anchor_rate: float
    # Training
    n_em_iterations: int
    n_restarts: int
    best_restart_seed: int
    final_log_likelihood: float
    convergence_curve: List[float]
    converged: bool
    # Learned parameters (stored as lists for JSON)
    pi: List[float]
    A: List[List[float]]
    B: List[List[float]]
    # Analysis
    A_entropy_mean: float
    A_sparsity: float
    B_entropy_mean: float
    B_sparsity: float
    # Top emissions per state (for inspection)
    top_emissions: Dict[str, List[Dict]]
    # Comparison
    init_log_likelihood: float
    ll_improvement: float
    runtime_seconds: float


# ---------------------------------------------------------------------------
# HMM training functions
# ---------------------------------------------------------------------------

def _encode_corpus_chars(
    corpus,
    obs_vocab: List[str],
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """Encode the entire corpus as a sequence of observation indices.

    Returns:
        obs_seq: (N,) array of observation indices
        token_boundaries: list of (start, end) indices for each token
    """
    char_to_idx = {ch: i for i, ch in enumerate(obs_vocab)}
    unk_idx = len(obs_vocab) - 1  # use last index for unknown

    obs_list: List[int] = []
    boundaries: List[Tuple[int, int]] = []

    for folio_id, page in corpus.pages.items():
        for token in page.all_tokens:
            chars = tokenize_eva_chars(token)
            start = len(obs_list)
            for ch in chars:
                obs_list.append(char_to_idx.get(ch, unk_idx))
            end = len(obs_list)
            if end > start:
                boundaries.append((start, end))

    return np.array(obs_list, dtype=np.int32), boundaries


def _build_anchor_mask(
    corpus,
    obs_vocab: List[str],
    state_labels: List[str],
    assignment: Dict[str, str],
    modifier_chars: set,
    bedrock_words: Dict[str, List[str]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Build anchor arrays aligned with the observation sequence.

    Returns:
        is_anchored: (N,) bool array — True at anchored positions
        anchor_state: (N,) int array — state index at anchored positions (0 elsewhere)
    """
    char_to_idx = {ch: i for i, ch in enumerate(obs_vocab)}
    state_index = {label: i for i, label in enumerate(state_labels)}
    eva_to_triple = build_eva_to_triple_lookup()

    is_anchored_list: List[bool] = []
    anchor_state_list: List[int] = []

    for folio_id, page in corpus.pages.items():
        for token in page.all_tokens:
            chars = tokenize_eva_chars(token)

            # Decode this token
            decoded = decode_token_modifier_aware(
                token, assignment, eva_to_triple, modifier_chars
            )

            if decoded in bedrock_words:
                syllables = bedrock_words[decoded]
                syl_idx = 0
                for ch in chars:
                    if ch in modifier_chars:
                        is_anchored_list.append(False)
                        anchor_state_list.append(0)
                    elif syl_idx < len(syllables):
                        syl = syllables[syl_idx]
                        si = state_index.get(syl)
                        if si is not None:
                            is_anchored_list.append(True)
                            anchor_state_list.append(si)
                        else:
                            is_anchored_list.append(False)
                            anchor_state_list.append(0)
                        syl_idx += 1
                    else:
                        is_anchored_list.append(False)
                        anchor_state_list.append(0)
            else:
                for ch in chars:
                    is_anchored_list.append(False)
                    anchor_state_list.append(0)

    return (
        np.array(is_anchored_list, dtype=bool),
        np.array(anchor_state_list, dtype=np.int32),
    )


def _forward(
    obs: np.ndarray, pi: np.ndarray, A: np.ndarray, B: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Scaled forward pass. Returns (alpha, scales)."""
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
    """Scaled backward pass. Returns beta: (T, K)."""
    T = len(obs)
    K = A.shape[0]
    beta = np.zeros((T, K))
    beta[T - 1] = 1.0

    for t in range(T - 2, -1, -1):
        beta[t] = A @ (B[:, obs[t + 1]] * beta[t + 1])
        if scales[t + 1] > 0:
            beta[t] /= scales[t + 1]

    return beta


def _baum_welch_anchored(
    obs: np.ndarray,
    pi: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    is_anchored: np.ndarray,
    anchor_state: np.ndarray,
    max_iter: int = 50,
    tol: float = 1e-3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, int, List[float]]:
    """Anchor-constrained Baum-Welch EM.

    At anchored positions, gamma is hard-clamped to the known state.
    """
    T = len(obs)
    K = len(pi)
    V = B.shape[1]
    prev_ll = -np.inf
    ll_curve: List[float] = []

    for iteration in range(1, max_iter + 1):
        # E step: forward-backward
        alpha, scales = _forward(obs, pi, A, B)
        beta = _backward(obs, A, B, scales)

        # Log-likelihood
        log_ll = float(np.sum(np.log(scales + 1e-300)))
        ll_curve.append(log_ll)

        # Convergence
        if abs(log_ll - prev_ll) < tol:
            return pi, A, B, log_ll, iteration, ll_curve
        prev_ll = log_ll

        # Gamma: posterior P(z_t = k | obs)
        gamma = alpha * beta
        gamma_sum = gamma.sum(axis=1, keepdims=True)
        gamma_sum = np.where(gamma_sum > 0, gamma_sum, 1.0)
        gamma = gamma / gamma_sum

        # ANCHOR CLAMPING
        anchor_positions = np.where(is_anchored)[0]
        for t in anchor_positions:
            gamma[t, :] = 0.0
            gamma[t, anchor_state[t]] = 1.0

        # Xi accumulation (vectorized over time)
        xi_sum = np.zeros((K, K))
        for t in range(T - 1):
            numer = np.outer(alpha[t], B[:, obs[t + 1]] * beta[t + 1]) * A
            denom = numer.sum()
            if denom > 0:
                xi_sum += numer / denom

        # M step
        pi = gamma[0] / gamma[0].sum()

        # Transition
        gamma_t_sum = gamma[:-1].sum(axis=0)
        gamma_t_sum = np.where(gamma_t_sum > 0, gamma_t_sum, 1.0)
        A = xi_sum / gamma_t_sum[:, None]
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

        if iteration % 10 == 0:
            print(f"       Iter {iteration}: LL = {log_ll:.1f}")

    return pi, A, B, prev_ll, max_iter, ll_curve


def _row_entropy(matrix: np.ndarray) -> np.ndarray:
    """Entropy of each row of a probability matrix (bits)."""
    eps = 1e-12
    return -np.sum(matrix * np.log2(matrix + eps), axis=1)


def _matrix_sparsity(matrix: np.ndarray, threshold: float = 0.01) -> float:
    """Fraction of entries below threshold."""
    return float((matrix < threshold).sum()) / matrix.size


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_baum_welch_training() -> None:
    """Step 43.12: train HMM via anchor-constrained Baum-Welch."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 43.12: Baum-Welch Training")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load architecture ──
    print("\n  1. Loading HMM architecture …")
    hmm_arch = _safe_load(os.path.join(rd, 'hmm_architecture.json'))
    state_labels = hmm_arch.get('state_labels', [])
    obs_vocab = hmm_arch.get('observation_vocab', [])
    K = len(state_labels)
    V = len(obs_vocab)

    pi_init = np.array(hmm_arch.get('pi', [1.0 / K] * K))
    A_init = np.array(hmm_arch.get('A', [[1.0 / K] * K] * K))
    B_init = np.array(hmm_arch.get('B', [[1.0 / V] * V] * K))

    print(f"     K={K} states, V={V} observations")

    # ── 2. Load assignment and modifiers ──
    print("\n  2. Loading assignment and modifiers …")
    combined = _safe_load(os.path.join(rd, 'combined_refine.json'))
    assignment = combined.get('best_assignment', {})
    mod_data = _safe_load(os.path.join(rd, 'modifier_integrate.json'))
    modifier_chars = set(mod_data.get('modifier_chars', []))

    # ── 3. Encode corpus ──
    print("\n  3. Encoding corpus …")
    corpus = load_corpus(verbose=False)
    obs_seq, token_boundaries = _encode_corpus_chars(corpus, obs_vocab)
    T = len(obs_seq)
    print(f"     {T:,} character positions, {len(token_boundaries):,} tokens")

    # ── 4. Build anchor mask ──
    print("\n  4. Building anchor mask …")
    from voynich.phases.anchor_initialization import BEDROCK_WORDS
    is_anchored, anchor_state = _build_anchor_mask(
        corpus, obs_vocab, state_labels, assignment, modifier_chars, BEDROCK_WORDS
    )
    n_anchored = int(is_anchored.sum())
    print(f"     Anchored positions: {n_anchored:,} / {T:,} ({n_anchored / T:.1%})")

    # ── 5. Train with multiple restarts ──
    print("\n  5. Training HMM (max 50 iters, 3 restarts) …")
    MAX_ITER = 50
    N_RESTARTS = 3

    # Compute initial LL for comparison
    _, init_scales = _forward(obs_seq, pi_init, A_init, B_init)
    init_ll = float(np.sum(np.log(init_scales + 1e-300)))
    print(f"     Initial LL: {init_ll:.1f}")

    best_ll = -np.inf
    best_result = None

    for restart in range(N_RESTARTS):
        print(f"\n     Restart {restart + 1}/{N_RESTARTS}:")

        # Perturb initialization
        rng = np.random.default_rng(42 + restart)
        if restart == 0:
            pi_r = pi_init.copy()
            A_r = A_init.copy()
            B_r = B_init.copy()
        else:
            # Add Dirichlet noise
            noise_scale = 0.1
            pi_r = pi_init + rng.dirichlet(np.ones(K) * 10) * noise_scale
            pi_r /= pi_r.sum()

            A_r = A_init.copy()
            for i in range(K):
                A_r[i] += rng.dirichlet(np.ones(K) * 10) * noise_scale
                A_r[i] /= A_r[i].sum()

            B_r = B_init.copy()
            for i in range(K):
                B_r[i] += rng.dirichlet(np.ones(V) * 10) * noise_scale
                B_r[i] /= B_r[i].sum()

        pi_t, A_t, B_t, ll, n_iter, ll_curve = _baum_welch_anchored(
            obs_seq, pi_r, A_r, B_r,
            is_anchored, anchor_state,
            max_iter=MAX_ITER, tol=1e-3,
        )

        print(f"       Final LL={ll:.1f}, iters={n_iter}")

        if ll > best_ll:
            best_ll = ll
            best_result = (pi_t, A_t, B_t, ll, n_iter, ll_curve, restart)

    pi_best, A_best, B_best, ll_best, iter_best, ll_curve_best, best_seed = best_result
    print(f"\n     Best restart: seed={42 + best_seed}, LL={ll_best:.1f}")

    # ── 6. Analyze learned parameters ──
    print("\n  6. Analyzing learned parameters …")

    a_entropy = _row_entropy(A_best)
    a_entropy_mean = float(np.mean(a_entropy))
    a_sparsity = _matrix_sparsity(A_best)
    print(f"     A entropy mean: {a_entropy_mean:.3f}")
    print(f"     A sparsity: {a_sparsity:.3f}")

    b_entropy = _row_entropy(B_best)
    b_entropy_mean = float(np.mean(b_entropy))
    b_sparsity = _matrix_sparsity(B_best)
    print(f"     B entropy mean: {b_entropy_mean:.3f}")
    print(f"     B sparsity: {b_sparsity:.3f}")

    # Top emissions per state
    top_emissions: Dict[str, List[Dict]] = {}
    for k in range(K):
        top_indices = np.argsort(B_best[k])[::-1][:5]
        entries = []
        for idx in top_indices:
            if idx < len(obs_vocab):
                entries.append({
                    'eva_char': obs_vocab[idx],
                    'probability': round(float(B_best[k, idx]), 4),
                })
        top_emissions[state_labels[k]] = entries

    # Print a few examples
    print("\n     Top emissions (first 5 states):")
    for k, label in enumerate(state_labels[:5]):
        top = top_emissions[label]
        top_str = ', '.join(f"{e['eva_char']}={e['probability']:.3f}" for e in top[:3])
        print(f"       {label}: {top_str}")

    # ── 7. Save ──
    elapsed = time.time() - t0

    result = BaumWelchResult(
        n_tokens=len(token_boundaries),
        n_char_positions=T,
        n_hidden_states=K,
        n_observation_types=V,
        n_anchored_positions=n_anchored,
        anchor_rate=round(n_anchored / T, 6) if T > 0 else 0.0,
        n_em_iterations=iter_best,
        n_restarts=N_RESTARTS,
        best_restart_seed=42 + best_seed,
        final_log_likelihood=round(ll_best, 2),
        convergence_curve=[round(x, 2) for x in ll_curve_best],
        converged=iter_best < MAX_ITER,
        pi=pi_best.tolist(),
        A=A_best.tolist(),
        B=B_best.tolist(),
        A_entropy_mean=round(a_entropy_mean, 4),
        A_sparsity=round(a_sparsity, 4),
        B_entropy_mean=round(b_entropy_mean, 4),
        B_sparsity=round(b_sparsity, 4),
        top_emissions=top_emissions,
        init_log_likelihood=round(init_ll, 2),
        ll_improvement=round(ll_best - init_ll, 2),
        runtime_seconds=round(elapsed, 1),
    )

    out_path = os.path.join(rd, 'baum_welch_training.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)
    print(f"\n  → {out_path} ({elapsed:.1f}s)")
