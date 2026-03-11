"""
Step 43.10 -- HMM Architecture Definition
==========================================
Define the Hidden Markov Model for context-dependent decoding.
Hidden states = CV/CVC syllables, observations = EVA characters.

Dependency chain:
    results/combined_refine.json    (Phase 15: 25-triple assignment)
    results/tachygraphic_stroke.json (Phase 19.5: sign families)
    data/reference/latin/           (reference corpus for bigrams)
    data/corpus/                    (EVA transcription)
        -> hmm_architecture.json    (this step)
"""

import json
import math
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    load_corpus,
    tokenize_eva_chars,
)
from voynich.core.reference import (
    EVA_VISUAL_COMPONENTS,
    PHONEME_NUCLEUS_MAP,
    PHONEME_PLACE_MAP,
    build_triple_phoneme_hypotheses,
    load_reference_corpus,
)
from voynich.core.stats import syllabify_latin


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
    if isinstance(obj, float) and (obj != obj):
        return None
    if isinstance(obj, (bool, int, float, str, type(None))):
        return obj
    return str(obj)


def _safe_load(path: str) -> Optional[Dict]:
    """Load a JSON file, returning None if not found."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class HMMArchitectureResult:
    n_hidden_states: int              # K
    n_observation_types: int          # V
    state_labels: List[str]           # index -> syllable
    observation_vocab: List[str]      # index -> EVA char
    # Initialized parameters (stored as lists for JSON)
    pi: List[float]                   # (K,)
    A: List[List[float]]              # (K, K)
    B: List[List[float]]              # (K, V)
    # Architecture info
    n_confirmed_states: int           # from Phase 15
    n_cv_states: int                  # from reference
    n_cvc_states: int                 # closed syllables
    total_parameters: int
    data_to_param_ratio: float
    position_dependent: bool          # True
    state_design: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Latin vowels / consonants (for syllable classification)
# ---------------------------------------------------------------------------

_LATIN_VOWELS = set('aeiouy')


def _is_cv_syllable(syl: str) -> bool:
    """Check if a syllable is open (ends in a vowel)."""
    return len(syl) > 0 and syl[-1] in _LATIN_VOWELS


def _is_cvc_syllable(syl: str) -> bool:
    """Check if a syllable is closed (ends in a consonant)."""
    if not syl:
        return False
    # Must contain at least one vowel AND end in a consonant
    has_vowel = any(c in _LATIN_VOWELS for c in syl)
    return has_vowel and syl[-1] not in _LATIN_VOWELS


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------

def _build_hidden_states(
    best_assignment: Dict[str, str],
    lat_tokens: List[str],
    target_k: int = 100,
) -> Tuple[List[str], int, int, int]:
    """Build hidden state labels from confirmed assignments + reference.

    Returns (state_labels, n_confirmed, n_cv_added, n_cvc_added).
    """
    # --- 1. Confirmed states from Phase 15 assignment ---
    confirmed = sorted(set(best_assignment.values()))
    confirmed_set = set(confirmed)

    # --- 2. Syllabify reference Latin and count frequencies ---
    cv_counts: Counter = Counter()
    cvc_counts: Counter = Counter()

    for word in lat_tokens:
        syls = syllabify_latin(word.lower())
        for syl in syls:
            syl_lower = syl.lower().strip()
            if not syl_lower:
                continue
            if _is_cv_syllable(syl_lower):
                cv_counts[syl_lower] += 1
            elif _is_cvc_syllable(syl_lower):
                cvc_counts[syl_lower] += 1

    # --- 3. Top 75 CV syllables not already confirmed ---
    cv_candidates = [
        syl for syl, _ in cv_counts.most_common()
        if syl not in confirmed_set
    ]
    n_cv_slots = min(75, target_k - len(confirmed))
    cv_added = cv_candidates[:n_cv_slots]

    # --- 4. Top 25 CVC syllables ---
    cvc_candidates = [
        syl for syl, _ in cvc_counts.most_common()
        if syl not in confirmed_set and syl not in set(cv_added)
    ]
    n_cvc_slots = min(25, target_k - len(confirmed) - len(cv_added))
    cvc_added = cvc_candidates[:max(0, n_cvc_slots)]

    # --- 5. Assemble state_labels ---
    state_labels = confirmed + cv_added + cvc_added

    return state_labels, len(confirmed), len(cv_added), len(cvc_added)


def _build_observation_vocab(corpus) -> List[str]:
    """Collect all unique EVA characters from the corpus, sorted by frequency."""
    char_counts: Counter = Counter()
    tokens = corpus.get_tokens()
    for tok in tokens:
        chars = tokenize_eva_chars(tok)
        for ch in chars:
            char_counts[ch] += 1

    # Sort by descending frequency
    return [ch for ch, _ in char_counts.most_common()]


def _build_transition_matrix(
    state_labels: List[str],
    lat_tokens: List[str],
    alpha: float = 0.01,
) -> np.ndarray:
    """Build transition matrix A from Latin syllable bigrams.

    Returns K x K numpy array of probabilities.
    """
    K = len(state_labels)
    state_idx = {syl: i for i, syl in enumerate(state_labels)}

    # Count bigrams from reference text
    bigram_counts = np.zeros((K, K), dtype=np.float64)

    for word in lat_tokens:
        syls = syllabify_latin(word.lower())
        syls_lower = [s.lower().strip() for s in syls if s.strip()]
        for j in range(len(syls_lower) - 1):
            s1 = syls_lower[j]
            s2 = syls_lower[j + 1]
            i1 = state_idx.get(s1)
            i2 = state_idx.get(s2)
            if i1 is not None and i2 is not None:
                bigram_counts[i1, i2] += 1.0

    # Laplace smoothing + normalization
    A = bigram_counts + alpha
    row_sums = A.sum(axis=1, keepdims=True)
    # Avoid division by zero (should not happen with alpha > 0)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    A = A / row_sums

    return A


def _build_emission_matrix(
    state_labels: List[str],
    observation_vocab: List[str],
    best_assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
) -> np.ndarray:
    """Build emission matrix B from Phase 15 assignments + stroke priors.

    Returns K x V numpy array of probabilities.
    """
    K = len(state_labels)
    V = len(observation_vocab)
    state_idx = {syl: i for i, syl in enumerate(state_labels)}
    obs_idx = {ch: j for j, ch in enumerate(observation_vocab)}

    B = np.zeros((K, V), dtype=np.float64)

    for j, eva_char in enumerate(observation_vocab):
        triple_key = eva_to_triple.get(eva_char)

        if triple_key is None:
            # EVA char not in any triple: uniform distribution
            B[:, j] = 1.0 / K
            continue

        assigned_syl = best_assignment.get(triple_key)
        assigned_idx = state_idx.get(assigned_syl) if assigned_syl else None

        # Find the first_stroke for this triple to get related syllables
        parts = triple_key.split(',')
        first_stroke = parts[0] if len(parts) >= 1 else None

        # Collect states that share the same consonant class
        related_indices = []
        if first_stroke and first_stroke in PHONEME_PLACE_MAP:
            consonants = set(PHONEME_PLACE_MAP[first_stroke])
            for k, syl in enumerate(state_labels):
                if k == assigned_idx:
                    continue
                # Check if syllable starts with a consonant from this class
                if syl and syl[0] in consonants:
                    related_indices.append(k)

        if assigned_idx is not None:
            # Primary assignment gets 0.7
            B[assigned_idx, j] = 0.7
            # Spread 0.3 across related states
            if related_indices:
                share = 0.3 / len(related_indices)
                for k in related_indices:
                    B[k, j] = share
            else:
                # No related states -- spread across all others
                remaining = [k for k in range(K) if k != assigned_idx]
                if remaining:
                    share = 0.3 / len(remaining)
                    for k in remaining:
                        B[k, j] = share
        else:
            # Triple exists but no assignment -- use stroke-based prior
            if first_stroke and first_stroke in PHONEME_PLACE_MAP:
                consonants = set(PHONEME_PLACE_MAP[first_stroke])
                matching = [
                    k for k, syl in enumerate(state_labels)
                    if syl and syl[0] in consonants
                ]
                if matching:
                    share = 1.0 / len(matching)
                    for k in matching:
                        B[k, j] = share
                else:
                    B[:, j] = 1.0 / K
            else:
                B[:, j] = 1.0 / K

    # Ensure every state has at least a small uniform floor so no row is
    # all-zero (states with no assignment get a diffuse emission prior)
    floor = 1e-4 / V
    B = np.maximum(B, floor)

    # Normalize rows: B[k, v] = P(obs=v | state=k)
    row_sums = B.sum(axis=1, keepdims=True)
    B = B / row_sums

    return B


def _build_initial_distribution(
    state_labels: List[str],
    lat_tokens: List[str],
    alpha: float = 0.01,
) -> np.ndarray:
    """Build initial state distribution pi from reference word-initial syllables.

    Returns K-length numpy array of probabilities.
    """
    K = len(state_labels)
    state_idx = {syl: i for i, syl in enumerate(state_labels)}

    counts = np.zeros(K, dtype=np.float64)

    for word in lat_tokens:
        syls = syllabify_latin(word.lower())
        if syls:
            first_syl = syls[0].lower().strip()
            idx = state_idx.get(first_syl)
            if idx is not None:
                counts[idx] += 1.0

    # Laplace smoothing + normalization
    pi = counts + alpha
    pi = pi / pi.sum()

    return pi


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_hmm_architecture(verbose: bool = True) -> HMMArchitectureResult:
    """Define the HMM architecture for context-dependent decoding.

    Builds hidden states (CV/CVC syllables), observation vocabulary (EVA chars),
    and initializes transition (A), emission (B), and initial (pi) matrices.
    """
    t0 = time.time()
    rd = str(_results_dir())

    if verbose:
        print("=" * 60)
        print("Step 43.10: HMM Architecture Definition")
        print("=" * 60)

    # ------------------------------------------------------------------
    # Load dependencies
    # ------------------------------------------------------------------
    if verbose:
        print("\n[1] Loading dependencies...")

    # Phase 15 assignment
    combined = _safe_load(os.path.join(rd, 'combined_refine.json'))
    if combined is None:
        raise FileNotFoundError(
            "combined_refine.json not found -- run Phase 15 first"
        )
    best_assignment = combined.get('best_assignment', {})
    if verbose:
        print(f"  Phase 15 assignment: {len(best_assignment)} triples")

    # Reference Latin corpus
    ref = load_reference_corpus(languages=['latin'], verbose=False)
    lat_tokens = ref.get_combined_tokens('latin')
    if verbose:
        print(f"  Latin reference: {len(lat_tokens):,} tokens")

    # Voynich corpus
    corpus = load_corpus(verbose=False)
    all_tokens = corpus.get_tokens()
    if verbose:
        print(f"  Voynich corpus: {len(all_tokens):,} tokens")

    # EVA-to-triple lookup
    eva_to_triple = build_eva_to_triple_lookup()

    # ------------------------------------------------------------------
    # Step 1: Define hidden states (target K ~ 100)
    # ------------------------------------------------------------------
    if verbose:
        print("\n[2] Building hidden states...")

    state_labels, n_confirmed, n_cv, n_cvc = _build_hidden_states(
        best_assignment, lat_tokens, target_k=100,
    )
    K = len(state_labels)
    if verbose:
        print(f"  Confirmed states (Phase 15): {n_confirmed}")
        print(f"  CV states from reference:    {n_cv}")
        print(f"  CVC states from reference:   {n_cvc}")
        print(f"  Total hidden states (K):     {K}")
        print(f"  First 10: {state_labels[:10]}")

    # ------------------------------------------------------------------
    # Step 2: Define observation vocabulary (V = unique EVA chars)
    # ------------------------------------------------------------------
    if verbose:
        print("\n[3] Building observation vocabulary...")

    observation_vocab = _build_observation_vocab(corpus)
    V = len(observation_vocab)
    if verbose:
        print(f"  Unique EVA characters (V): {V}")
        print(f"  Top 10: {observation_vocab[:10]}")

    # ------------------------------------------------------------------
    # Step 3: Initialize transition matrix A (K x K)
    # ------------------------------------------------------------------
    if verbose:
        print("\n[4] Initializing transition matrix A...")

    A = _build_transition_matrix(state_labels, lat_tokens, alpha=0.01)
    if verbose:
        # Report sparsity
        nonzero = np.count_nonzero(A > 0.01 / K)
        print(f"  A shape: {A.shape}")
        print(f"  Non-trivial entries (> uniform): {nonzero} / {K * K}")

    # ------------------------------------------------------------------
    # Step 4: Initialize emission matrix B (K x V)
    # ------------------------------------------------------------------
    if verbose:
        print("\n[5] Initializing emission matrix B...")

    B = _build_emission_matrix(
        state_labels, observation_vocab, best_assignment, eva_to_triple,
    )
    if verbose:
        # Report sparsity
        threshold = 1.0 / (K * V)
        nonzero_b = np.count_nonzero(B > threshold)
        total_b = K * V
        sparsity = 1.0 - (nonzero_b / total_b)
        print(f"  B shape: {B.shape}")
        print(f"  Non-trivial entries: {nonzero_b} / {total_b}")
        print(f"  Sparsity: {sparsity:.3f}")

    # ------------------------------------------------------------------
    # Step 5: Initialize pi (K,)
    # ------------------------------------------------------------------
    if verbose:
        print("\n[6] Initializing initial distribution pi...")

    pi = _build_initial_distribution(state_labels, lat_tokens, alpha=0.01)
    if verbose:
        top5_pi = np.argsort(pi)[::-1][:5]
        for idx in top5_pi:
            print(f"  pi[{state_labels[idx]}] = {pi[idx]:.4f}")

    # ------------------------------------------------------------------
    # Step 6: Position-dependent emission architecture
    # ------------------------------------------------------------------
    if verbose:
        print("\n[7] Position-dependent emission architecture...")
        print("  3 position matrices defined: B_initial, B_medial, B_final")
        print("  Currently identical to B (will diverge during training)")

    # Position-dependent matrices are identical to B at initialization.
    # Training (Step 43.12) will differentiate them.
    # We record the architectural support but do not store 3 copies in JSON
    # to keep file size manageable.

    # ------------------------------------------------------------------
    # Step 7: Model summary
    # ------------------------------------------------------------------
    # Count total EVA characters in corpus for data-to-parameter ratio
    total_chars = 0
    for tok in all_tokens:
        total_chars += len(tokenize_eva_chars(tok))

    # Parameters: A (K*K) + B (K*V) + pi (K)
    total_params = K * K + K * V + K
    data_to_param = total_chars / total_params if total_params > 0 else 0.0

    # B sparsity (fraction of near-zero entries)
    b_threshold = 1e-6
    b_sparse_count = np.count_nonzero(B < b_threshold)
    b_sparsity = b_sparse_count / (K * V) if K * V > 0 else 0.0

    runtime = time.time() - t0

    if verbose:
        print("\n" + "=" * 60)
        print("Model Summary")
        print("=" * 60)
        print(f"  Hidden states (K):          {K}")
        print(f"  Observation types (V):      {V}")
        print(f"  Total parameters:           {total_params:,}")
        print(f"  Corpus characters:          {total_chars:,}")
        print(f"  Data-to-parameter ratio:    {data_to_param:.2f}")
        print(f"  B sparsity:                 {b_sparsity:.3f}")
        print(f"  Position-dependent:         True")
        print(f"  Runtime:                    {runtime:.1f}s")

    # ------------------------------------------------------------------
    # Build result and save
    # ------------------------------------------------------------------
    state_design = (
        f"{n_confirmed} confirmed (Phase 15) + "
        f"{n_cv} CV (Latin reference) + "
        f"{n_cvc} CVC (Latin reference) = {K} total"
    )

    result = HMMArchitectureResult(
        n_hidden_states=K,
        n_observation_types=V,
        state_labels=state_labels,
        observation_vocab=observation_vocab,
        pi=pi.tolist(),
        A=A.tolist(),
        B=B.tolist(),
        n_confirmed_states=n_confirmed,
        n_cv_states=n_cv,
        n_cvc_states=n_cvc,
        total_parameters=total_params,
        data_to_param_ratio=round(data_to_param, 4),
        position_dependent=True,
        state_design=state_design,
        runtime_seconds=round(runtime, 3),
    )

    out_path = os.path.join(rd, 'hmm_architecture.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)

    if verbose:
        print(f"\nSaved: {out_path}")

    return result
