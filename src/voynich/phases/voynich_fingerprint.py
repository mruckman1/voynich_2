"""
Step 43.1 – Voynich Statistical Target Fingerprint
===================================================
Build the comprehensive statistical fingerprint of the Voynich manuscript
that any correct encoding must reproduce.

Dependency chain:
    data/corpus/                  (EVA transcription)
    results/modifier_integrate.json  (Phase 16 modifier handling)
        → voynich_fingerprint.json  (this step)
"""

import json
import math
import os
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import load_corpus, tokenize_eva_chars
from voynich.core.stats import (
    first_order_entropy,
    conditional_entropy,
    compute_all_entropy,
    word_conditional_entropy,
    char_frequencies,
)


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


def _safe_load(path: str) -> Dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SECTIONS = [
    'herbal_a', 'herbal_b', 'astronomical', 'biological',
    'cosmological', 'pharmaceutical', 'recipes',
]

# Weights for composite fingerprint distance
WEIGHT_CHAR_FREQ = 3.0
WEIGHT_ENTROPY = 2.0
WEIGHT_TOKEN_LENGTH = 2.0
WEIGHT_ZIPF = 1.0
WEIGHT_BIGRAM = 1.0
WEIGHT_SECTION = 0.5


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CharLevelStats:
    """Character-level statistics."""
    n_unique_chars: int
    total_chars: int
    char_freqs: Dict[str, float]            # char -> frequency
    char_vocabulary: List[str]              # sorted list of all EVA chars
    bigram_matrix: List[List[float]]        # 44×44 (or n_chars×n_chars) count matrix
    bigram_alphabet: List[str]              # ordering for bigram matrix rows/cols
    entropy_curve: Dict[str, float]         # H0..H6


@dataclass
class TokenLevelStats:
    """Token-level statistics."""
    n_tokens: int
    n_types: int
    type_token_ratio: float
    hapax_count: int
    hapax_rate: float
    mean_token_length: float
    std_token_length: float
    token_length_dist: Dict[int, int]       # length -> count
    zipf_exponent: float
    zipf_r_squared: float
    top_20_tokens: List[Tuple[str, int]]


@dataclass
class TokenBigramStats:
    """Token bigram statistics."""
    bigram_entropy: float
    n_unique_bigrams: int
    top_30_bigrams: List[Tuple[str, int]]   # "w1 w2" -> count


@dataclass
class SectionStats:
    """Per-section statistics."""
    section: str
    n_tokens: int
    n_chars: int
    char_freqs: Dict[str, float]
    entropy_curve: Dict[str, float]


@dataclass
class FingerprintVector:
    """The assembled fingerprint vector with labels and weights."""
    values: List[float]
    labels: List[str]
    weights: List[float]
    n_dimensions: int


@dataclass
class VoynichFingerprintResult:
    """Complete result of Step 43.1."""
    char_level: CharLevelStats
    token_level: TokenLevelStats
    token_bigram: TokenBigramStats
    section_stats: List[SectionStats]
    cross_section_correlation: Dict[str, float]  # "secA_vs_secB" -> corr
    fingerprint: FingerprintVector
    elapsed_seconds: float


# ---------------------------------------------------------------------------
# Character-level analysis
# ---------------------------------------------------------------------------

def _build_eva_char_text(tokens: List[str]) -> Tuple[List[str], List[str]]:
    """Decompose all tokens into EVA characters.

    Returns:
        (flat_chars, vocabulary) where flat_chars is the sequence of EVA chars
        across the entire corpus and vocabulary is the sorted unique set.
    """
    flat: List[str] = []
    for tok in tokens:
        chars = tokenize_eva_chars(tok)
        flat.extend(chars)
    vocab = sorted(set(flat))
    return flat, vocab


def _compute_entropy_curve(text: str, max_order: int = 6) -> Dict[str, float]:
    """Compute entropy at orders 0 through max_order.

    H0 = log2(alphabet_size)
    H1 = first_order_entropy (unigram)
    H2..H6 = conditional entropy at order 1..5
    """
    curve: Dict[str, float] = {}

    # H0: log2 of alphabet size
    chars_no_space = [c for c in text if c != ' ']
    alphabet_size = len(set(chars_no_space)) if chars_no_space else 1
    curve['H0'] = math.log2(alphabet_size) if alphabet_size > 0 else 0.0

    # H1: unigram entropy
    curve['H1'] = first_order_entropy(text)

    # H2..H6: conditional entropy at increasing orders
    for order in range(1, max_order):
        key = f'H{order + 1}'
        curve[key] = conditional_entropy(text, order=order)

    return curve


def _build_char_bigram_matrix(
    flat_chars: List[str], vocab: List[str]
) -> Tuple[List[List[float]], List[str]]:
    """Build an N x N character bigram count matrix.

    flat_chars: sequence of EVA characters (not raw letters).
    vocab: sorted unique EVA chars for row/col ordering.

    Returns (matrix_as_lists, vocab).
    """
    char_to_idx = {c: i for i, c in enumerate(vocab)}
    n = len(vocab)
    matrix = np.zeros((n, n), dtype=float)

    for i in range(len(flat_chars) - 1):
        c1, c2 = flat_chars[i], flat_chars[i + 1]
        idx1 = char_to_idx.get(c1)
        idx2 = char_to_idx.get(c2)
        if idx1 is not None and idx2 is not None:
            matrix[idx1, idx2] += 1.0

    return matrix.tolist(), vocab


def _compute_char_level(
    tokens: List[str], text: str
) -> Tuple[CharLevelStats, List[str], List[str]]:
    """Compute all character-level statistics.

    Returns (CharLevelStats, flat_chars, vocabulary).
    """
    flat_chars, vocab = _build_eva_char_text(tokens)

    # Frequencies over EVA chars
    total = len(flat_chars)
    counts = Counter(flat_chars)
    freqs = {c: counts.get(c, 0) / total for c in vocab} if total > 0 else {}

    # Bigram matrix (over EVA chars)
    bigram_mat, bigram_alpha = _build_char_bigram_matrix(flat_chars, vocab)

    # Entropy curve (over raw EVA text — single-letter level)
    entropy_curve = _compute_entropy_curve(text, max_order=6)

    stats = CharLevelStats(
        n_unique_chars=len(vocab),
        total_chars=total,
        char_freqs=freqs,
        char_vocabulary=vocab,
        bigram_matrix=bigram_mat,
        bigram_alphabet=bigram_alpha,
        entropy_curve=entropy_curve,
    )
    return stats, flat_chars, vocab


# ---------------------------------------------------------------------------
# Token-level analysis
# ---------------------------------------------------------------------------

def _compute_token_level(tokens: List[str]) -> TokenLevelStats:
    """Compute token-level statistics including Zipf."""
    counts = Counter(tokens)
    n_tokens = len(tokens)
    n_types = len(counts)

    # Type-token ratio
    ttr = n_types / n_tokens if n_tokens > 0 else 0.0

    # Hapax legomena
    hapax_count = sum(1 for c in counts.values() if c == 1)
    hapax_rate = hapax_count / n_types if n_types > 0 else 0.0

    # Token length distribution (in EVA chars)
    lengths = [len(tokenize_eva_chars(t)) for t in tokens]
    mean_len = float(np.mean(lengths)) if lengths else 0.0
    std_len = float(np.std(lengths)) if lengths else 0.0
    length_dist = dict(Counter(lengths))

    # Zipf exponent via log-log linear regression
    ranked_freqs = sorted(counts.values(), reverse=True)
    n = len(ranked_freqs)
    if n >= 2:
        ranks = np.arange(1, n + 1, dtype=float)
        freqs = np.array(ranked_freqs, dtype=float)
        log_ranks = np.log(ranks)
        log_freqs = np.log(freqs + 1e-10)
        A = np.vstack([log_ranks, np.ones(n)]).T
        try:
            result = np.linalg.lstsq(A, log_freqs, rcond=None)
            slope, _ = result[0]
        except np.linalg.LinAlgError:
            slope = -1.0
        zipf_exp = -slope

        predicted = slope * log_ranks + result[0][1]
        ss_res = float(np.sum((log_freqs - predicted) ** 2))
        ss_tot = float(np.sum((log_freqs - np.mean(log_freqs)) ** 2))
        r_sq = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    else:
        zipf_exp = 0.0
        r_sq = 0.0

    top_20 = counts.most_common(20)

    return TokenLevelStats(
        n_tokens=n_tokens,
        n_types=n_types,
        type_token_ratio=ttr,
        hapax_count=hapax_count,
        hapax_rate=hapax_rate,
        mean_token_length=mean_len,
        std_token_length=std_len,
        token_length_dist=length_dist,
        zipf_exponent=zipf_exp,
        zipf_r_squared=r_sq,
        top_20_tokens=top_20,
    )


# ---------------------------------------------------------------------------
# Token bigram analysis
# ---------------------------------------------------------------------------

def _compute_token_bigrams(tokens: List[str]) -> TokenBigramStats:
    """Compute token-level bigram statistics."""
    if len(tokens) < 2:
        return TokenBigramStats(
            bigram_entropy=0.0,
            n_unique_bigrams=0,
            top_30_bigrams=[],
        )

    # Count bigrams
    bigram_counts: Counter = Counter()
    for i in range(len(tokens) - 1):
        bigram_counts[(tokens[i], tokens[i + 1])] += 1

    n_unique = len(bigram_counts)

    # Bigram entropy: H(W_n | W_{n-1})
    bg_entropy = word_conditional_entropy(tokens, order=1)

    # Top 30 bigrams formatted as "w1 w2"
    top_30 = [
        (f"{w1} {w2}", c)
        for (w1, w2), c in bigram_counts.most_common(30)
    ]

    return TokenBigramStats(
        bigram_entropy=bg_entropy,
        n_unique_bigrams=n_unique,
        top_30_bigrams=top_30,
    )


# ---------------------------------------------------------------------------
# Section-level analysis
# ---------------------------------------------------------------------------

def _compute_section_stats(corpus, vocab: List[str]) -> List[SectionStats]:
    """Compute per-section character frequencies and entropy curves."""
    results = []
    for section in SECTIONS:
        text = corpus.get_text(section=section, paragraph_only=True)
        tokens = text.split() if text else []
        if not tokens:
            results.append(SectionStats(
                section=section,
                n_tokens=0,
                n_chars=0,
                char_freqs={},
                entropy_curve={},
            ))
            continue

        # EVA char frequencies
        flat_chars, _ = _build_eva_char_text(tokens)
        total = len(flat_chars)
        counts = Counter(flat_chars)
        freqs = {c: counts.get(c, 0) / total for c in vocab} if total > 0 else {}

        # Entropy curve
        ec = _compute_entropy_curve(text, max_order=6)

        results.append(SectionStats(
            section=section,
            n_tokens=len(tokens),
            n_chars=total,
            char_freqs=freqs,
            entropy_curve=ec,
        ))

    return results


def _cross_section_correlation(
    section_stats: List[SectionStats], vocab: List[str]
) -> Dict[str, float]:
    """Compute pairwise Pearson correlation of char freq vectors across sections."""
    corrs: Dict[str, float] = {}
    # Build vectors
    vectors: Dict[str, np.ndarray] = {}
    for ss in section_stats:
        if ss.n_chars == 0:
            continue
        vec = np.array([ss.char_freqs.get(c, 0.0) for c in vocab])
        vectors[ss.section] = vec

    secs = sorted(vectors.keys())
    for i in range(len(secs)):
        for j in range(i + 1, len(secs)):
            v1 = vectors[secs[i]]
            v2 = vectors[secs[j]]
            # Pearson correlation
            if np.std(v1) > 0 and np.std(v2) > 0:
                r = float(np.corrcoef(v1, v2)[0, 1])
            else:
                r = 0.0
            key = f"{secs[i]}_vs_{secs[j]}"
            corrs[key] = round(r, 6)

    return corrs


# ---------------------------------------------------------------------------
# Fingerprint vector assembly
# ---------------------------------------------------------------------------

def _assemble_fingerprint(
    char_stats: CharLevelStats,
    token_stats: TokenLevelStats,
    bigram_stats: TokenBigramStats,
    section_stats: List[SectionStats],
    vocab: List[str],
) -> FingerprintVector:
    """Assemble a flat fingerprint vector from all computed statistics.

    Components (approximate dimensionality):
        1. Char frequencies              (~44 dims)
        2. Entropy curve H0-H6           (7 dims)
        3. Bigram matrix flattened        (top eigenvalues, ~20 dims)
        4. Token length distribution      (~15 dims, lengths 1-15)
        5. Scalar token stats             (6 dims)
        6. Token bigram entropy           (1 dim)
        7. Section char freq vectors      (7 sections × 10 top chars = 70 dims)
        8. Section entropy curves         (7 sections × 7 orders = 49 dims)
    Total: ~210-220 dimensions
    """
    values: List[float] = []
    labels: List[str] = []
    weights: List[float] = []

    # --- 1. Character frequencies (sorted by vocab) ---
    for c in vocab:
        values.append(char_stats.char_freqs.get(c, 0.0))
        labels.append(f"char_freq_{c}")
        weights.append(WEIGHT_CHAR_FREQ)

    # --- 2. Entropy curve H0-H6 ---
    for order in range(7):
        key = f"H{order}"
        val = char_stats.entropy_curve.get(key, 0.0)
        values.append(val)
        labels.append(f"entropy_{key}")
        weights.append(WEIGHT_ENTROPY)

    # --- 3. Bigram matrix: top 20 singular values ---
    mat = np.array(char_stats.bigram_matrix)
    if mat.size > 0:
        try:
            sv = np.linalg.svd(mat, compute_uv=False)
            # Normalize by largest singular value
            if sv[0] > 0:
                sv = sv / sv[0]
            top_sv = sv[:20].tolist()
        except np.linalg.LinAlgError:
            top_sv = [0.0] * 20
    else:
        top_sv = [0.0] * 20
    # Pad if fewer than 20
    while len(top_sv) < 20:
        top_sv.append(0.0)
    for i, s in enumerate(top_sv):
        values.append(float(s))
        labels.append(f"bigram_sv_{i}")
        weights.append(WEIGHT_BIGRAM)

    # --- 4. Token length distribution (lengths 1 through 15) ---
    total_tokens = token_stats.n_tokens if token_stats.n_tokens > 0 else 1
    for length in range(1, 16):
        count = token_stats.token_length_dist.get(length, 0)
        values.append(count / total_tokens)
        labels.append(f"tok_len_{length}")
        weights.append(WEIGHT_TOKEN_LENGTH)

    # --- 5. Scalar token stats ---
    scalar_pairs = [
        ("ttr", token_stats.type_token_ratio, WEIGHT_TOKEN_LENGTH),
        ("hapax_rate", token_stats.hapax_rate, WEIGHT_TOKEN_LENGTH),
        ("mean_tok_len", token_stats.mean_token_length, WEIGHT_TOKEN_LENGTH),
        ("std_tok_len", token_stats.std_token_length, WEIGHT_TOKEN_LENGTH),
        ("zipf_exponent", token_stats.zipf_exponent, WEIGHT_ZIPF),
        ("zipf_r2", token_stats.zipf_r_squared, WEIGHT_ZIPF),
    ]
    for label, val, w in scalar_pairs:
        values.append(val)
        labels.append(label)
        weights.append(w)

    # --- 6. Token bigram entropy ---
    values.append(bigram_stats.bigram_entropy)
    labels.append("tok_bigram_entropy")
    weights.append(WEIGHT_BIGRAM)

    # --- 7. Section char freq vectors (top 10 chars per section) ---
    # Use the 10 most frequent chars corpus-wide
    sorted_chars = sorted(
        char_stats.char_freqs.items(), key=lambda x: -x[1]
    )[:10]
    top_char_keys = [c for c, _ in sorted_chars]

    for ss in section_stats:
        for c in top_char_keys:
            val = ss.char_freqs.get(c, 0.0)
            values.append(val)
            labels.append(f"sec_{ss.section}_{c}")
            weights.append(WEIGHT_SECTION)

    # --- 8. Section entropy curves ---
    for ss in section_stats:
        for order in range(7):
            key = f"H{order}"
            val = ss.entropy_curve.get(key, 0.0)
            values.append(val)
            labels.append(f"sec_{ss.section}_{key}")
            weights.append(WEIGHT_SECTION)

    # --- Min-max normalize values to [0, 1] ---
    raw = np.array(values, dtype=float)
    vmin = np.min(raw) if len(raw) > 0 else 0.0
    vmax = np.max(raw) if len(raw) > 0 else 1.0
    rng = vmax - vmin
    if rng > 0:
        normalized = ((raw - vmin) / rng).tolist()
    else:
        normalized = [0.0] * len(raw)

    return FingerprintVector(
        values=normalized,
        labels=labels,
        weights=weights,
        n_dimensions=len(normalized),
    )


# ---------------------------------------------------------------------------
# Distance metric
# ---------------------------------------------------------------------------

def fingerprint_distance(
    fp_a: FingerprintVector, fp_b: FingerprintVector
) -> float:
    """Weighted Euclidean distance between two fingerprint vectors.

    Both vectors must have the same dimensionality and label ordering.
    Uses the weights from fp_a.
    """
    if fp_a.n_dimensions != fp_b.n_dimensions:
        raise ValueError(
            f"Dimension mismatch: {fp_a.n_dimensions} vs {fp_b.n_dimensions}"
        )
    va = np.array(fp_a.values)
    vb = np.array(fp_b.values)
    w = np.array(fp_a.weights)
    diff = va - vb
    return float(np.sqrt(np.sum(w * diff ** 2)))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_voynich_fingerprint() -> Dict[str, Any]:
    """Step 43.1 — Build the Voynich statistical target fingerprint."""
    t0 = time.time()
    print("=" * 70)
    print("STEP 43.1: Voynich Statistical Target Fingerprint")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load corpus
    # ------------------------------------------------------------------
    print("\n[1] Loading corpus ...")
    corpus = load_corpus(verbose=False)
    tokens = corpus.get_tokens(paragraph_only=True)
    text = corpus.get_text(paragraph_only=True)
    print(f"    Tokens: {len(tokens):,}   Characters: {len(text.replace(' ', '')):,}")

    # ------------------------------------------------------------------
    # 2. Character-level statistics
    # ------------------------------------------------------------------
    print("\n[2] Computing character-level statistics ...")
    char_stats, flat_chars, vocab = _compute_char_level(tokens, text)
    print(f"    EVA vocabulary: {char_stats.n_unique_chars} unique chars")
    print(f"    Entropy curve: " + ", ".join(
        f"{k}={v:.3f}" for k, v in sorted(char_stats.entropy_curve.items())
    ))

    # ------------------------------------------------------------------
    # 3. Token-level statistics
    # ------------------------------------------------------------------
    print("\n[3] Computing token-level statistics ...")
    token_stats = _compute_token_level(tokens)
    print(f"    Types: {token_stats.n_types:,}  TTR: {token_stats.type_token_ratio:.4f}")
    print(f"    Hapax: {token_stats.hapax_count:,} ({token_stats.hapax_rate:.3f})")
    print(f"    Mean token length: {token_stats.mean_token_length:.2f} +/- "
          f"{token_stats.std_token_length:.2f} EVA chars")
    print(f"    Zipf exponent: {token_stats.zipf_exponent:.3f} "
          f"(R^2={token_stats.zipf_r_squared:.4f})")

    # ------------------------------------------------------------------
    # 4. Token bigram statistics
    # ------------------------------------------------------------------
    print("\n[4] Computing token bigram statistics ...")
    bigram_stats = _compute_token_bigrams(tokens)
    print(f"    Unique bigrams: {bigram_stats.n_unique_bigrams:,}")
    print(f"    Bigram entropy H(W|W-1): {bigram_stats.bigram_entropy:.3f}")
    if bigram_stats.top_30_bigrams:
        top3 = bigram_stats.top_30_bigrams[:3]
        print(f"    Top 3: " + ", ".join(
            f"'{b}' ({c})" for b, c in top3
        ))

    # ------------------------------------------------------------------
    # 5. Section-level statistics
    # ------------------------------------------------------------------
    print("\n[5] Computing section-level statistics ...")
    sec_stats = _compute_section_stats(corpus, vocab)
    for ss in sec_stats:
        if ss.n_tokens > 0:
            h1 = ss.entropy_curve.get('H1', 0.0)
            print(f"    {ss.section:20s}  tokens={ss.n_tokens:6,}  "
                  f"chars={ss.n_chars:7,}  H1={h1:.3f}")
        else:
            print(f"    {ss.section:20s}  (no data)")

    # Cross-section correlation
    print("\n[6] Computing cross-section correlations ...")
    cross_corr = _cross_section_correlation(sec_stats, vocab)
    if cross_corr:
        mean_corr = sum(cross_corr.values()) / len(cross_corr)
        min_corr = min(cross_corr.values())
        max_corr = max(cross_corr.values())
        print(f"    Mean pairwise char-freq correlation: {mean_corr:.4f}")
        print(f"    Range: [{min_corr:.4f}, {max_corr:.4f}]")

    # ------------------------------------------------------------------
    # 6. Assemble fingerprint vector
    # ------------------------------------------------------------------
    print("\n[7] Assembling fingerprint vector ...")
    fingerprint = _assemble_fingerprint(
        char_stats, token_stats, bigram_stats, sec_stats, vocab
    )
    print(f"    Dimensions: {fingerprint.n_dimensions}")
    print(f"    Value range: [{min(fingerprint.values):.4f}, "
          f"{max(fingerprint.values):.4f}]")

    # Sanity: self-distance should be 0
    self_dist = fingerprint_distance(fingerprint, fingerprint)
    print(f"    Self-distance (sanity): {self_dist:.6f}")

    elapsed = time.time() - t0

    # ------------------------------------------------------------------
    # 7. Build result and save
    # ------------------------------------------------------------------
    result = VoynichFingerprintResult(
        char_level=char_stats,
        token_level=token_stats,
        token_bigram=bigram_stats,
        section_stats=sec_stats,
        cross_section_correlation=cross_corr,
        fingerprint=fingerprint,
        elapsed_seconds=round(elapsed, 2),
    )

    out_path = _results_dir() / "voynich_fingerprint.json"
    with open(out_path, "w") as f:
        json.dump(_convert(result), f, indent=2)

    print(f"\n    Saved → {out_path}")
    print(f"    Elapsed: {elapsed:.1f}s")
    print("=" * 70)

    return _convert(result)
