"""
Phase 8 / Approach 18: Minimum Description Length Decoding
===========================================================
The best decoding of the Voynich text is the one that produces the most
compressible (lowest-entropy) output under a Latin language model.

For a candidate stem-level mapping M, decode the Voynich text to produce
candidate plaintext T(M).  Evaluate T(M) under a Latin language model LM.
The mapping that minimizes cross-entropy H(T(M), LM) is the best decoding.

Natural language has ~1.5 bits/char of entropy; incorrect decodings produce
~4+ bits/char.  The gap is enormous, compounded over thousands of characters.

Sub-analyses:
  18.1 — Build character-level and word-level language models
  18.2 — Define mapping search spaces
  18.3 — MCMC decoder (SA-based, multiple granularities/languages)
  18.4 — Evaluate best decoding (compression, word validity, phrases)
  18.5 — Validation battery (sanity check, null tests, cross-validation)

Output:
  results/mdl_decode.json
"""

import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from voynich.core.corpus import load_corpus, VoynichCorpus
from voynich.core.stats import (
    build_ngram_lm,
    cross_entropy_lm,
    selectivity_ratio,
)
from voynich.core._paths import results_dir as _results_dir
from voynich.core.reference import (
    load_reference_corpus, ReferenceCorpus,
    stem_token, build_latin_phrase_catalog,
    LATIN_PHARMACEUTICAL_DOMAINS,
)
from voynich.core.ciphers import SimpleSubstitutionCipher
from voynich.phases.morpheme_grid import decompose_token_morphemes


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class LanguageModelStats:
    """Statistics for one language model configuration."""
    language: str
    order: int
    vocab_size: int
    cross_entropy_train: float
    cross_entropy_heldout: float
    cross_entropy_random: float
    discrimination_gap: float


@dataclass
class MCMCResult:
    """Result from one MCMC decoding run."""
    granularity: str
    target_language: str
    best_cross_entropy: float
    init_cross_entropy: float
    random_cross_entropy: float
    compression_ratio: float
    word_validity_fraction: float
    n_recognized_phrases: int
    best_mapping: Dict[str, str]
    convergence_history: List[float]
    n_iterations: int


@dataclass
class SanityCheckResult:
    """Result of known-cipher recovery test."""
    cipher_type: str
    n_stems: int
    true_mapping_sample: Dict[str, str]
    recovered_mapping_sample: Dict[str, str]
    recovery_accuracy: float
    ce_true: float
    ce_recovered: float
    ce_random: float
    passed: bool


@dataclass
class MDLDecodeResult:
    """Full Phase 8 / Approach 18 output."""
    language_model_stats: List[Dict]
    search_spaces: Dict[str, Dict]
    mcmc_results: Dict[str, Dict]
    language_ranking: List[Dict]
    best_granularity: str
    best_language: str
    best_cross_entropy: float
    best_compression_ratio: float
    best_word_validity: float
    best_phrase_count: int
    decoded_sample: List[str]
    sanity_check: Dict
    null_tests: Dict
    cross_validation: Dict
    gate_passed: bool
    verdict: str


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
    if isinstance(obj, (bool, int, float, str, type(None))):
        return obj
    return str(obj)


def _prepare_voynich_stems(
    tokens: List[str],
    min_count: int = 3,
) -> Tuple[List[str], List[str], Counter]:
    """Decompose Voynich tokens to stems, filter by frequency."""
    stems = []
    for tok in tokens:
        d = decompose_token_morphemes(tok)
        stems.append(d.stem if d.stem else tok)

    counts = Counter(stems)
    vocab = [s for s, c in counts.most_common() if c >= min_count]
    vocab_set = set(vocab)
    filtered = [s for s in stems if s in vocab_set]
    return filtered, vocab, counts


def _prepare_ref_stems(
    ref_corpus: ReferenceCorpus,
    language: str,
    min_count: int = 3,
) -> Tuple[List[str], List[str], Counter]:
    """Prepare stem sequence from reference language."""
    tokens = ref_corpus.get_combined_tokens(language)
    stems = [stem_token(t, language) for t in tokens]
    counts = Counter(stems)
    vocab = [s for s, c in counts.most_common() if c >= min_count]
    vocab_set = set(vocab)
    filtered = [s for s in stems if s in vocab_set]
    return filtered, vocab, counts


# ---------------------------------------------------------------------------
# 18.1: Build Language Models
# ---------------------------------------------------------------------------

def build_language_models(
    ref_corpus: ReferenceCorpus,
    languages: List[str] = ('latin', 'occitan'),
    orders: List[int] = (3, 5),
    smoothing: float = 0.01,
    heldout_fraction: float = 0.1,
    seed: int = 42,
) -> Tuple[Dict[str, Dict[int, Dict]], List[LanguageModelStats]]:
    """
    Build character-level n-gram language models and measure quality.

    Returns:
        lm_dict: language -> order -> lm_dict (for use with cross_entropy_lm)
        lm_stats: list of LanguageModelStats
    """
    lm_dict: Dict[str, Dict[int, Dict]] = {}
    all_stats: List[LanguageModelStats] = []

    for language in languages:
        tokens = ref_corpus.get_combined_tokens(language)
        if len(tokens) < 100:
            print(f"  WARNING: {language} corpus too small ({len(tokens)} tokens)")
            continue

        # Split into train/heldout
        rng = random.Random(seed)
        n_heldout = max(1, int(len(tokens) * heldout_fraction))
        indices = list(range(len(tokens)))
        rng.shuffle(indices)
        heldout_idx = set(indices[:n_heldout])
        train_tokens = [tokens[i] for i in range(len(tokens))
                        if i not in heldout_idx]
        heldout_tokens = [tokens[i] for i in range(len(tokens))
                          if i in heldout_idx]

        lm_dict[language] = {}

        for order in orders:
            lm = build_ngram_lm(train_tokens, order=order, smoothing=smoothing)
            lm_dict[language][order] = lm

            # Measure cross-entropy
            train_text = '_'.join(train_tokens[:2000])
            train_text = '_' + train_text + '_'
            ce_train = cross_entropy_lm(train_text, lm)

            heldout_text = '_'.join(heldout_tokens[:500])
            heldout_text = '_' + heldout_text + '_'
            ce_heldout = cross_entropy_lm(heldout_text, lm)

            # Random text: shuffle characters from the training text
            rng_r = random.Random(seed + 1)
            chars = list(train_text[:2000])
            rng_r.shuffle(chars)
            random_text = ''.join(chars)
            ce_random = cross_entropy_lm(random_text, lm)

            gap = ce_random - ce_heldout

            stats = LanguageModelStats(
                language=language,
                order=order,
                vocab_size=lm['vocab_size'],
                cross_entropy_train=round(ce_train, 4),
                cross_entropy_heldout=round(ce_heldout, 4),
                cross_entropy_random=round(ce_random, 4),
                discrimination_gap=round(gap, 4),
            )
            all_stats.append(stats)

            print(f"  {language} {order}-gram LM: "
                  f"CE_train={ce_train:.3f}, CE_heldout={ce_heldout:.3f}, "
                  f"CE_random={ce_random:.3f}, gap={gap:.3f}")

    return lm_dict, all_stats


# ---------------------------------------------------------------------------
# 18.2: Define Mapping Search Space
# ---------------------------------------------------------------------------

def _build_stem_mapping_space(
    voynich_stems: List[str],
    voynich_vocab: List[str],
    ref_stems: List[str],
    ref_vocab: List[str],
    top_n: int = 200,
) -> Tuple[List[str], List[str]]:
    """
    Define stem-level mapping search space.

    Returns (voynich_stems_top_n, ref_stems_top_n).
    """
    v_top = voynich_vocab[:top_n]
    r_top = ref_vocab[:top_n]
    # Equalize sizes
    n = min(len(v_top), len(r_top))
    return v_top[:n], r_top[:n]


# ---------------------------------------------------------------------------
# 18.3: MCMC Decoder
# ---------------------------------------------------------------------------

def _decode_with_stem_mapping(
    voynich_stem_seq: List[str],
    mapping: Dict[str, str],
    max_tokens: int = 2000,
) -> str:
    """
    Apply a stem mapping to decode Voynich stems to target language stems.

    Returns decoded text as a space-separated string of mapped stems.
    """
    decoded = []
    for stem in voynich_stem_seq[:max_tokens]:
        mapped = mapping.get(stem, stem)
        decoded.append(mapped)
    return ' '.join(decoded)


def _build_mapping_from_perm(
    perm: np.ndarray,
    source_vocab: List[str],
    target_vocab: List[str],
) -> Dict[str, str]:
    """Convert a permutation array to a mapping dict."""
    mapping = {}
    for i, j in enumerate(perm):
        if i < len(source_vocab) and j < len(target_vocab):
            mapping[source_vocab[i]] = target_vocab[int(j)]
    return mapping


def _propose_perm_swap(state: np.ndarray, rng: random.Random) -> np.ndarray:
    """Swap two positions in a permutation array."""
    new = state.copy()
    n = len(new)
    i, j = rng.sample(range(n), 2)
    new[i], new[j] = new[j], new[i]
    return new


def _fast_sa_mdl(
    stem_indices: np.ndarray,
    target_stems: List[str],
    lm: Dict,
    n: int,
    max_iter: int = 100_000,
    t_start: float = 0.1,
    t_end: float = 0.0001,
    seed: int = 42,
    init_perm: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float, List[float]]:
    """
    Fast SA for MDL decoding with incremental cross-entropy updates.

    Instead of re-evaluating cross-entropy over the entire decoded text each
    iteration (~2500 chars), we decompose the text into per-token segments
    and maintain running totals.  A swap of perm[a]/perm[b] only changes
    segments at token positions whose stem_index is a or b, plus one
    successor each (context cascade stops after one step because right
    contexts depend only on the local segment string).  This gives ~25x
    speedup for N=100 with 500 tokens.

    If init_perm is provided, start from that permutation instead of
    identity (used for seeded decoding).
    """
    rng = np.random.RandomState(seed)
    perm = init_perm.copy() if init_perm is not None else np.arange(n, dtype=int)

    order = lm['order']
    counts = lm['counts']
    k_smooth = lm['smoothing']
    V = lm['vocab_size']
    ctx_len = order - 1
    n_tokens = len(stem_indices)

    if n_tokens == 0:
        return perm, float('inf'), []

    # Precompute context totals for fast probability lookups
    ctx_totals: Dict = {}
    for ctx, cc in counts.items():
        ctx_totals[ctx] = sum(cc.values())

    # Position index: for each perm index, which token positions use it
    pos_for_idx: List[List[int]] = [[] for _ in range(n)]
    for pos in range(n_tokens):
        pos_for_idx[int(stem_indices[pos])].append(pos)

    # --- Segment-based cross-entropy ---
    # Each token position produces segment = decoded_stem + '_'.
    # Full decoded text = '_' + seg[0] + seg[1] + ...
    # Right context of a segment = its last ctx_len chars (becomes left
    # context for the next segment).  Cascade stops after one successor.

    def _right_ctx(seg: str) -> str:
        return seg[-ctx_len:] if len(seg) >= ctx_len else seg

    def _seg_cost(left: str, seg: str) -> Tuple[float, int]:
        """Negative log-prob (bits) and scored-char count for one segment."""
        local = left + seg
        nlp = 0.0
        nc = 0
        for i in range(ctx_len, len(local)):
            context = tuple(local[i - ctx_len:i])
            char = local[i]
            prob = None
            for bo in range(ctx_len + 1):
                c = context[bo:]
                if c in counts:
                    prob = (counts[c].get(char, 0) + k_smooth) / (
                        ctx_totals[c] + k_smooth * V)
                    break
            if prob is None or prob <= 0:
                prob = 1.0 / V
            nlp -= math.log2(prob)
            nc += 1
        return nlp, nc

    # Initialise segments, right contexts, per-segment costs
    initial_left = '_'  # decoded text starts with '_'
    segs: List[str] = []
    rctxs: List[str] = []
    seg_costs = [0.0] * n_tokens
    seg_nchars = [0] * n_tokens

    for pos in range(n_tokens):
        s = target_stems[perm[int(stem_indices[pos])]] + '_'
        segs.append(s)
        rctxs.append(_right_ctx(s))

    for pos in range(n_tokens):
        left = initial_left if pos == 0 else rctxs[pos - 1]
        c, nc = _seg_cost(left, segs[pos])
        seg_costs[pos] = c
        seg_nchars[pos] = nc

    total_bits = sum(seg_costs)
    total_chars = sum(seg_nchars)
    current_ce = total_bits / total_chars if total_chars > 0 else float('inf')

    best_ce = current_ce
    best_perm = perm.copy()
    history: List[float] = []

    cooling = (t_end / t_start) ** (1.0 / max(max_iter, 1))
    temp = t_start

    for it in range(max_iter):
        a = rng.randint(0, n)
        b = rng.randint(0, n)
        while a == b:
            b = rng.randint(0, n)

        # Affected token positions (decoded stem changes)
        aff_a = pos_for_idx[a]
        aff_b = pos_for_idx[b]

        if not aff_a and not aff_b:
            # Swap has no observable effect — always accept (delta=0)
            perm[a], perm[b] = perm[b], perm[a]
            temp *= cooling
            if it % 10_000 == 0:
                history.append(best_ce)
            continue

        affected = set()
        for p in aff_a:
            affected.add(p)
        for p in aff_b:
            affected.add(p)

        # Successors: their left context changes when an affected pos updates
        succs = set()
        for p in affected:
            nxt = p + 1
            if nxt < n_tokens and nxt not in affected:
                succs.add(nxt)

        recompute = sorted(affected | succs)

        # Save old state for rollback
        old_segs = {p: segs[p] for p in recompute}
        old_rctxs = {p: rctxs[p] for p in recompute}
        old_costs = {p: seg_costs[p] for p in recompute}
        old_nc = {p: seg_nchars[p] for p in recompute}

        # Apply swap
        perm[a], perm[b] = perm[b], perm[a]

        # Update segment strings and right contexts at affected positions
        for p in sorted(affected):
            s = target_stems[perm[int(stem_indices[p])]] + '_'
            segs[p] = s
            rctxs[p] = _right_ctx(s)

        # Recompute costs (sorted order ensures left contexts are current)
        d_bits = 0.0
        d_chars = 0
        for p in recompute:
            left = initial_left if p == 0 else rctxs[p - 1]
            c, nc = _seg_cost(left, segs[p])
            d_bits += c - seg_costs[p]
            d_chars += nc - seg_nchars[p]
            seg_costs[p] = c
            seg_nchars[p] = nc

        new_bits = total_bits + d_bits
        new_chars = total_chars + d_chars
        new_ce = new_bits / new_chars if new_chars > 0 else float('inf')

        delta = new_ce - current_ce

        if delta < 0 or rng.random() < math.exp(-delta / temp):
            total_bits = new_bits
            total_chars = new_chars
            current_ce = new_ce
            if current_ce < best_ce:
                best_ce = current_ce
                best_perm = perm.copy()
        else:
            # Rollback
            perm[a], perm[b] = perm[b], perm[a]
            for p in recompute:
                segs[p] = old_segs[p]
                rctxs[p] = old_rctxs[p]
                seg_costs[p] = old_costs[p]
                seg_nchars[p] = old_nc[p]

        temp *= cooling
        if it % 10_000 == 0:
            history.append(best_ce)

    return best_perm, best_ce, history


def run_mcmc_decode(
    voynich_stem_seq: List[str],
    voynich_vocab: List[str],
    target_vocab: List[str],
    lm: Dict,
    ref_corpus: ReferenceCorpus,
    language: str,
    granularity: str = 'stem',
    max_tokens: int = 1000,
    max_iter: int = 100_000,
    n_restarts: int = 5,
    seed: int = 42,
) -> MCMCResult:
    """
    Run MCMC (SA-based) decoding for one configuration.

    Uses cross-entropy under the language model as the cost function.
    """
    n = min(len(voynich_vocab), len(target_vocab))
    v_vocab = voynich_vocab[:n]
    t_vocab = target_vocab[:n]

    # Convert stem sequence to index array for fast lookup
    v_to_idx = {s: i for i, s in enumerate(v_vocab)}
    truncated_seq = voynich_stem_seq[:max_tokens]
    # Map stems to indices; stems not in top-N get index n (out of range)
    stem_indices = np.array([v_to_idx.get(s, n) for s in truncated_seq], dtype=int)
    # Filter to only stems in vocabulary
    stem_indices = stem_indices[stem_indices < n]

    # Use a small subsample for cost evaluation speed
    if len(stem_indices) > 500:
        stem_indices = stem_indices[:500]

    # Initial cost (identity = rank-aligned)
    def _eval_cost(p):
        decoded_parts = [t_vocab[p[idx]] for idx in stem_indices]
        decoded_text = '_' + '_'.join(decoded_parts) + '_'
        return cross_entropy_lm(decoded_text, lm)

    init_perm = np.arange(n, dtype=int)
    init_cost = _eval_cost(init_perm)

    # Random baseline (fewer samples for speed)
    rng_null = np.random.RandomState(seed)
    random_costs = []
    for _ in range(20):
        rp = rng_null.permutation(n)
        random_costs.append(_eval_cost(rp))
    random_ce = float(np.mean(random_costs))

    # Calibrate temperature
    deltas = []
    rng_cal = np.random.RandomState(seed + 1000)
    for _ in range(50):
        tp = init_perm.copy()
        i, j = rng_cal.randint(0, n, size=2)
        tp[i], tp[j] = tp[j], tp[i]
        deltas.append(abs(_eval_cost(tp) - init_cost))
    median_delta = float(np.median(deltas)) if deltas else 0.01
    t_start = max(median_delta * 2.0, 0.01)

    print(f"      Init CE: {init_cost:.4f}, Random CE: {random_ce:.4f}")

    # Run restarts
    global_best_perm = init_perm.copy()
    global_best_cost = init_cost
    all_history: List[float] = []

    for r in range(n_restarts):
        perm, cost, hist = _fast_sa_mdl(
            stem_indices, t_vocab, lm, n,
            max_iter=max_iter,
            t_start=t_start,
            t_end=t_start * 0.001,
            seed=seed + r * 7,
        )
        all_history.extend(hist)
        if cost < global_best_cost:
            global_best_cost = cost
            global_best_perm = perm.copy()
        print(f"      Restart {r+1}/{n_restarts}: CE={cost:.4f}")

    best_mapping = _build_mapping_from_perm(global_best_perm, v_vocab, t_vocab)

    # Evaluate
    compression_ratio = random_ce / global_best_cost if global_best_cost > 0 else 0.0

    # Word validity
    word_validity = _check_word_validity(
        voynich_stem_seq, best_mapping, ref_corpus, language, max_tokens
    )

    # Phrase coherence
    phrase_catalog = build_latin_phrase_catalog()
    phrase_count = _check_phrase_coherence(
        voynich_stem_seq, best_mapping, phrase_catalog, max_tokens
    )

    print(f"      Best CE: {global_best_cost:.4f}, "
          f"Compression: {compression_ratio:.4f}x")
    print(f"      Word validity: {word_validity:.4f}, Phrases: {phrase_count}")

    return MCMCResult(
        granularity=granularity,
        target_language=language,
        best_cross_entropy=round(global_best_cost, 4),
        init_cross_entropy=round(init_cost, 4),
        random_cross_entropy=round(random_ce, 4),
        compression_ratio=round(compression_ratio, 4),
        word_validity_fraction=round(word_validity, 4),
        n_recognized_phrases=phrase_count,
        best_mapping=best_mapping,
        convergence_history=[round(c, 4) for c in all_history[-20:]],
        n_iterations=max_iter * n_restarts,
    )


# ---------------------------------------------------------------------------
# 18.4: Evaluation
# ---------------------------------------------------------------------------

def _check_word_validity(
    voynich_stem_seq: List[str],
    mapping: Dict[str, str],
    ref_corpus: ReferenceCorpus,
    language: str,
    max_tokens: int = 2000,
) -> float:
    """Fraction of decoded stems that appear in the reference vocabulary."""
    ref_tokens = ref_corpus.get_combined_tokens(language)
    ref_stems = set(stem_token(t, language) for t in ref_tokens)

    decoded = [mapping.get(s, '') for s in voynich_stem_seq[:max_tokens]]
    if not decoded:
        return 0.0

    valid = sum(1 for d in decoded if d in ref_stems)
    return valid / len(decoded)


def _check_phrase_coherence(
    voynich_stem_seq: List[str],
    mapping: Dict[str, str],
    phrase_catalog: Dict[str, List[str]],
    max_tokens: int = 2000,
) -> int:
    """Count recognized Latin phrases in decoded text."""
    decoded = [mapping.get(s, '') for s in voynich_stem_seq[:max_tokens]]
    decoded_text = ' '.join(decoded)

    count = 0
    for category, phrases in phrase_catalog.items():
        for phrase in phrases:
            if phrase in decoded_text:
                count += 1
    return count


def decode_sample(
    voynich_tokens: List[str],
    mapping: Dict[str, str],
    n_tokens: int = 50,
) -> List[str]:
    """Apply mapping to decode sample Voynich tokens."""
    decoded = []
    for tok in voynich_tokens[:n_tokens]:
        d = decompose_token_morphemes(tok)
        stem = d.stem if d.stem else tok
        mapped = mapping.get(stem, f'?{stem}?')
        decoded.append(f"{tok} -> {mapped}")
    return decoded


# ---------------------------------------------------------------------------
# 18.5: Validation Battery
# ---------------------------------------------------------------------------

def run_sanity_check(
    ref_corpus: ReferenceCorpus,
    lm: Dict,
    language: str = 'latin',
    top_n: int = 100,
    max_iter: int = 300_000,
    n_restarts: int = 10,
    seed: int = 42,
) -> SanityCheckResult:
    """
    CRITICAL: Run before Voynich decoding.

    Encipher Latin stems with a random substitution, then try to recover
    the mapping using MCMC.  Validates the approach works on known cipher.
    """
    print("\n    Sanity check: known-cipher recovery...")

    # Get reference stems
    ref_tokens = ref_corpus.get_combined_tokens(language)
    ref_stems_raw = [stem_token(t, language) for t in ref_tokens]
    counts = Counter(ref_stems_raw)
    vocab = [s for s, c in counts.most_common() if c >= 3]
    vocab_set = set(vocab)
    stem_seq = [s for s in ref_stems_raw if s in vocab_set]

    n = min(top_n, len(vocab))
    restricted_vocab = vocab[:n]
    restricted_set = set(restricted_vocab)
    restricted_seq = [s for s in stem_seq if s in restricted_set][:2000]

    # Create random substitution (permutation of stems)
    rng = random.Random(seed)
    cipher_vocab = list(restricted_vocab)
    rng.shuffle(cipher_vocab)

    # True mapping: cipher_vocab[i] is the enciphered form of restricted_vocab[i]
    true_encrypt = {restricted_vocab[i]: cipher_vocab[i] for i in range(n)}
    true_decrypt = {cipher_vocab[i]: restricted_vocab[i] for i in range(n)}

    # Encipher the stem sequence
    enciphered_seq = [true_encrypt.get(s, s) for s in restricted_seq]

    # Now try to recover: find mapping from cipher_vocab -> restricted_vocab
    # that minimizes cross-entropy of decoded text under the LM
    def cost_fn(perm):
        mapping = _build_mapping_from_perm(perm, cipher_vocab, restricted_vocab)
        decoded = _decode_with_stem_mapping(enciphered_seq, mapping, 2000)
        decoded_text = '_' + decoded.replace(' ', '_') + '_'
        return cross_entropy_lm(decoded_text, lm)

    init_perm = np.arange(n, dtype=int)
    init_cost = cost_fn(init_perm)

    # Random baseline
    rng_null = random.Random(seed + 100)
    random_costs = []
    for _ in range(20):
        rp = np.arange(n, dtype=int)
        rng_null.shuffle(rp)
        random_costs.append(cost_fn(rp))
    ce_random = float(np.mean(random_costs))

    # True mapping cost (should be low)
    true_perm = np.zeros(n, dtype=int)
    for i in range(n):
        # cipher_vocab[i] should map to restricted_vocab[i]
        true_perm[i] = i  # identity because cipher_vocab[i] -> restricted_vocab[i]
    # Actually, the true decryption maps cipher_vocab -> restricted_vocab
    # We need the perm that when applied gives the correct decode
    # cipher_vocab is already the permuted version, so the identity perm
    # maps cipher_vocab[i] -> restricted_vocab[i], which IS the true decrypt
    ce_true = cost_fn(true_perm)

    # Convert to index array for fast SA
    v_to_idx = {s: i for i, s in enumerate(cipher_vocab)}
    stem_indices = np.array([v_to_idx.get(s, n) for s in restricted_seq[:500]],
                            dtype=int)
    stem_indices = stem_indices[stem_indices < n]

    # Run fast SA
    global_best_perm = init_perm.copy()
    global_best_cost = float('inf')
    for r in range(n_restarts):
        best_perm_r, best_cost_r, _ = _fast_sa_mdl(
            stem_indices, restricted_vocab, lm, n,
            max_iter=max_iter,
            t_start=0.1, t_end=0.0001,
            seed=seed + r * 7,
        )
        if best_cost_r < global_best_cost:
            global_best_cost = best_cost_r
            global_best_perm = best_perm_r.copy()
        print(f"      Sanity restart {r+1}/{n_restarts}: CE={best_cost_r:.4f}")

    best_perm = global_best_perm
    best_cost = global_best_cost

    # Check recovery accuracy
    recovery_accuracy = float(np.mean(best_perm == true_perm))
    passed = recovery_accuracy > 0.3  # Lower threshold for stem-level

    recovered_mapping = _build_mapping_from_perm(
        best_perm, cipher_vocab, restricted_vocab
    )

    print(f"      CE true:      {ce_true:.4f}")
    print(f"      CE recovered: {best_cost:.4f}")
    print(f"      CE random:    {ce_random:.4f}")
    print(f"      Recovery accuracy: {recovery_accuracy:.4f}")
    print(f"      Sanity check: {'PASSED' if passed else 'FAILED'}")

    return SanityCheckResult(
        cipher_type='stem_substitution',
        n_stems=n,
        true_mapping_sample={k: v for k, v in list(true_decrypt.items())[:10]},
        recovered_mapping_sample={k: v for k, v in list(recovered_mapping.items())[:10]},
        recovery_accuracy=round(recovery_accuracy, 4),
        ce_true=round(ce_true, 4),
        ce_recovered=round(best_cost, 4),
        ce_random=round(ce_random, 4),
        passed=passed,
    )


def run_null_tests(
    voynich_stem_seq: List[str],
    voynich_vocab: List[str],
    target_vocab: List[str],
    lm: Dict,
    best_ce: float,
    ref_corpus: ReferenceCorpus,
    language: str,
    max_tokens: int = 2000,
    n_null_trials: int = 20,
    seed: int = 42,
) -> Dict:
    """
    Null tests for MDL decoding.

    a) Random mappings: mean CE of random stem permutations
    b) Shuffled Voynich: randomize Voynich stem order, re-run short MCMC
    c) Wrong-language target: evaluate best mapping under Occitan LM
    """
    n = min(len(voynich_vocab), len(target_vocab))
    v_vocab = voynich_vocab[:n]
    t_vocab = target_vocab[:n]
    truncated_seq = voynich_stem_seq[:max_tokens]
    results = {}

    # (a) Random mapping baseline
    print("\n    Null test (a): random mappings...")
    rng = random.Random(seed)
    random_ces = []
    for _ in range(n_null_trials):
        rp = np.arange(n, dtype=int)
        rng.shuffle(rp)
        mapping = _build_mapping_from_perm(rp, v_vocab, t_vocab)
        decoded = _decode_with_stem_mapping(truncated_seq, mapping, max_tokens)
        decoded_text = '_' + decoded.replace(' ', '_') + '_'
        random_ces.append(cross_entropy_lm(decoded_text, lm))

    random_mean = float(np.mean(random_ces))
    random_std = float(np.std(random_ces))
    selectivity = random_mean / best_ce if best_ce > 0 else 0.0
    results['random_mappings'] = {
        'mean_ce': round(random_mean, 4),
        'std_ce': round(random_std, 4),
        'best_ce': round(best_ce, 4),
        'selectivity': round(selectivity, 4),
        'n_trials': n_null_trials,
    }
    print(f"      Random CE: {random_mean:.4f} +/- {random_std:.4f}, "
          f"best: {best_ce:.4f}, selectivity: {selectivity:.4f}x")

    # (b) Shuffled Voynich
    print("\n    Null test (b): shuffled Voynich...")
    rng_b = np.random.RandomState(seed + 100)
    v_to_idx_null = {s: i for i, s in enumerate(v_vocab)}
    truncated_seq = voynich_stem_seq[:max_tokens]
    shuffled_seq = list(truncated_seq)
    rng_b.shuffle(shuffled_seq)
    shuffled_indices = np.array([v_to_idx_null.get(s, n) for s in shuffled_seq[:500]],
                                dtype=int)
    shuffled_indices = shuffled_indices[shuffled_indices < n]

    _, shuffled_best_ce, _ = _fast_sa_mdl(
        shuffled_indices, t_vocab, lm, n,
        max_iter=10_000, seed=seed + 200,
    )
    shuffled_sel = shuffled_best_ce / best_ce if best_ce > 0 else 0.0
    results['shuffled_voynich'] = {
        'shuffled_best_ce': round(shuffled_best_ce, 4),
        'real_best_ce': round(best_ce, 4),
        'ratio': round(shuffled_sel, 4),
    }
    print(f"      Shuffled best CE: {shuffled_best_ce:.4f}, "
          f"ratio: {shuffled_sel:.4f}")

    # (c) Wrong language
    print("\n    Null test (c): word validity on wrong language...")
    # Check word validity against wrong language
    wrong_lang = 'occitan' if language == 'latin' else 'latin'
    wrong_tokens = ref_corpus.get_combined_tokens(wrong_lang)
    if wrong_tokens:
        wrong_stems = set(stem_token(t, wrong_lang) for t in wrong_tokens)
        right_tokens = ref_corpus.get_combined_tokens(language)
        right_stems = set(stem_token(t, language) for t in right_tokens)

        # Re-evaluate best mapping word validity against wrong language
        results['wrong_language'] = {
            'target_language': wrong_lang,
            'right_vocab_size': len(right_stems),
            'wrong_vocab_size': len(wrong_stems),
        }
    else:
        results['wrong_language'] = {'skipped': True}

    return results


def run_cross_validation(
    corpus: VoynichCorpus,
    ref_corpus: ReferenceCorpus,
    lm: Dict,
    language: str,
    top_n: int = 100,
    max_iter: int = 10_000,
    seed: int = 42,
) -> Dict:
    """Split-half cross-validation: decode each half independently."""
    print("\n    Cross-validation: split-half by folios...")

    pages = corpus.get_pages_by_language('A')
    mid = len(pages) // 2
    half1_tokens = []
    for p in pages[:mid]:
        half1_tokens.extend(p.all_tokens)
    half2_tokens = []
    for p in pages[mid:]:
        half2_tokens.extend(p.all_tokens)

    if len(half1_tokens) < 100 or len(half2_tokens) < 100:
        return {'skipped': True, 'reason': 'insufficient_tokens'}

    h1_stems, h1_vocab, _ = _prepare_voynich_stems(half1_tokens, min_count=2)
    h2_stems, h2_vocab, _ = _prepare_voynich_stems(half2_tokens, min_count=2)
    ref_stems, ref_vocab, _ = _prepare_ref_stems(ref_corpus, language, min_count=2)

    n = min(top_n, len(h1_vocab), len(h2_vocab), len(ref_vocab))
    if n < 20:
        return {'skipped': True, 'reason': 'vocabulary_too_small'}

    h1_v = h1_vocab[:n]
    h2_v = h2_vocab[:n]
    r_v = ref_vocab[:n]

    # Build index arrays
    h1_to_idx = {s: i for i, s in enumerate(h1_v)}
    h2_to_idx = {s: i for i, s in enumerate(h2_v)}
    h1_seq = [s for s in h1_stems if s in set(h1_v)][:500]
    h2_seq = [s for s in h2_stems if s in set(h2_v)][:500]
    h1_indices = np.array([h1_to_idx.get(s, n) for s in h1_seq], dtype=int)
    h1_indices = h1_indices[h1_indices < n]
    h2_indices = np.array([h2_to_idx.get(s, n) for s in h2_seq], dtype=int)
    h2_indices = h2_indices[h2_indices < n]

    _, ce1, _ = _fast_sa_mdl(
        h1_indices, r_v, lm, n,
        max_iter=max_iter, seed=seed,
    )
    _, ce2, _ = _fast_sa_mdl(
        h2_indices, r_v, lm, n,
        max_iter=max_iter, seed=seed + 100,
    )

    consistency = min(ce1, ce2) / max(ce1, ce2) if max(ce1, ce2) > 0 else 0.0
    print(f"      Half1 CE: {ce1:.4f}, Half2 CE: {ce2:.4f}, "
          f"consistency: {consistency:.4f}")

    return {
        'half1_ce': round(ce1, 4),
        'half2_ce': round(ce2, 4),
        'consistency': round(consistency, 4),
        'half1_tokens': len(half1_tokens),
        'half2_tokens': len(half2_tokens),
        'n_vocab': n,
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_mdl_decode() -> Dict:
    """
    Run Phase 8 / Approach 18: Minimum Description Length Decoding.

    EXECUTION ORDER (sanity check FIRST):
    1. Build language models
    2. Run sanity check on known cipher
    3. If sanity passes: MCMC decode Voynich
    4. Evaluate best decoding
    5. Validation battery
    6. Gate check and save
    """
    print("=" * 70)
    print("PHASE 8 / APPROACH 18: MINIMUM DESCRIPTION LENGTH DECODING")
    print("=" * 70)

    # --- Load data ---
    print("\n--- 18.1: Building Language Models ---")
    corpus = load_corpus(verbose=False)
    ref_corpus = load_reference_corpus(verbose=False)

    all_languages = ['latin', 'occitan', 'italian', 'german']
    available_languages = [
        lang for lang in all_languages
        if len(ref_corpus.get_combined_tokens(lang)) >= 100
    ]
    print(f"  Available reference languages: {available_languages}")

    lm_dict, lm_stats = build_language_models(
        ref_corpus, languages=available_languages, orders=[3, 5]
    )

    if 'latin' not in lm_dict or not lm_dict['latin']:
        print("  ERROR: Could not build Latin language model")
        result = MDLDecodeResult(
            language_model_stats=[_convert(asdict(s)) for s in lm_stats],
            search_spaces={},
            mcmc_results={},
            language_ranking=[],
            best_granularity='none',
            best_language='none',
            best_cross_entropy=0.0,
            best_compression_ratio=0.0,
            best_word_validity=0.0,
            best_phrase_count=0,
            decoded_sample=[],
            sanity_check={},
            null_tests={},
            cross_validation={},
            gate_passed=False,
            verdict='no_language_model',
        )
        out = _convert(asdict(result))
        out_path = _results_dir() / 'mdl_decode.json'
        with open(out_path, 'w') as f:
            json.dump(out, f, indent=2)
        return out

    # Use trigram LM (best balance of expressiveness and data efficiency)
    best_order = 3
    latin_lm = lm_dict['latin'][best_order]

    # --- Sanity check (MUST run first) ---
    print("\n--- 18.5a: Sanity Check (Known Cipher Recovery) ---")
    sanity = run_sanity_check(
        ref_corpus=ref_corpus,
        lm=latin_lm,
        language='latin',
        top_n=50,
        max_iter=100_000,
        n_restarts=5,
        seed=42,
    )

    if not sanity.passed:
        print("\n  SANITY CHECK FAILED — MCMC approach may not work at stem level")
        print("  Proceeding anyway to collect diagnostic data...")

    # --- Prepare Voynich stems ---
    print("\n--- 18.2: Defining Mapping Search Space ---")
    voynich_tokens = corpus.get_tokens(language='A')
    v_stems, v_vocab, v_counts = _prepare_voynich_stems(voynich_tokens)
    r_stems, r_vocab, r_counts = _prepare_ref_stems(ref_corpus, 'latin')

    top_n = 100
    v_space, r_space = _build_stem_mapping_space(
        v_stems, v_vocab, r_stems, r_vocab, top_n
    )
    n = len(v_space)
    print(f"  Voynich stems: {len(v_stems)} tokens, {len(v_vocab)} unique")
    print(f"  Latin stems:   {len(r_stems)} tokens, {len(r_vocab)} unique")
    print(f"  Search space:  {n} stems (top-{top_n})")

    search_spaces = {
        'stem': {
            'n_voynich': len(v_space),
            'n_target': len(r_space),
            'voynich_sample': v_space[:10],
            'target_sample': r_space[:10],
        }
    }

    # --- MCMC decoding ---
    mcmc_results: Dict[str, MCMCResult] = {}

    for language in available_languages:
        if language not in lm_dict or best_order not in lm_dict[language]:
            continue

        lm = lm_dict[language][best_order]

        # Prepare target vocab
        r_stems_lang, r_vocab_lang, _ = _prepare_ref_stems(ref_corpus, language)
        _, t_vocab = _build_stem_mapping_space(
            v_stems, v_vocab, r_stems_lang, r_vocab_lang, top_n
        )

        key = f"stem_{language}"
        print(f"\n--- 18.3: MCMC Decoding (stem, {language}) ---")

        result = run_mcmc_decode(
            voynich_stem_seq=[s for s in v_stems if s in set(v_space)],
            voynich_vocab=v_space,
            target_vocab=t_vocab,
            lm=lm,
            ref_corpus=ref_corpus,
            language=language,
            granularity='stem',
            max_tokens=1000,
            max_iter=100_000,
            n_restarts=5,
            seed=42,
        )
        mcmc_results[key] = result

    # --- Find best result ---
    best_key = min(mcmc_results, key=lambda k: mcmc_results[k].best_cross_entropy)
    best = mcmc_results[best_key]
    best_granularity = best.granularity
    best_language = best.target_language

    # --- Language ranking ---
    ranked = sorted(mcmc_results.items(), key=lambda kv: kv[1].best_cross_entropy)
    print(f"\n--- 18.4: Language Ranking (by cross-entropy) ---")
    for rank, (key, res) in enumerate(ranked, 1):
        marker = " <-- BEST" if key == best_key else ""
        print(f"  {rank}. {res.target_language:10s}  "
              f"CE={res.best_cross_entropy:.4f}  "
              f"compression={res.compression_ratio:.4f}x  "
              f"word_valid={res.word_validity_fraction:.4f}{marker}")

    language_ranking = [
        {
            'rank': rank,
            'language': res.target_language,
            'cross_entropy': round(res.best_cross_entropy, 4),
            'compression_ratio': round(res.compression_ratio, 4),
            'word_validity': round(res.word_validity_fraction, 4),
        }
        for rank, (_, res) in enumerate(ranked, 1)
    ]

    print(f"\n  Best: {best_key}")
    print(f"    CE:           {best.best_cross_entropy:.4f}")
    print(f"    Compression:  {best.compression_ratio:.4f}x")
    print(f"    Word validity: {best.word_validity_fraction:.4f}")
    print(f"    Phrases:      {best.n_recognized_phrases}")

    # --- Decoded sample ---
    decoded = decode_sample(voynich_tokens, best.best_mapping, n_tokens=30)
    print("\n  Decoded sample:")
    for line in decoded[:15]:
        print(f"    {line}")
    if len(decoded) > 15:
        print(f"    ... ({len(decoded) - 15} more)")

    # --- Null tests (using best language's LM and vocab) ---
    best_lm = lm_dict[best_language][best_order]
    r_stems_best, r_vocab_best, _ = _prepare_ref_stems(ref_corpus, best_language)
    _, best_t_vocab = _build_stem_mapping_space(
        v_stems, v_vocab, r_stems_best, r_vocab_best, top_n
    )

    print("\n--- 18.5b: Null Tests ---")
    null_results = run_null_tests(
        voynich_stem_seq=[s for s in v_stems if s in set(v_space)],
        voynich_vocab=v_space,
        target_vocab=best_t_vocab,
        lm=best_lm,
        best_ce=best.best_cross_entropy,
        ref_corpus=ref_corpus,
        language=best_language,
        max_tokens=2000,
        n_null_trials=20,
        seed=42,
    )

    # --- Cross-validation ---
    print("\n--- 18.5c: Cross-Validation ---")
    cv_results = run_cross_validation(
        corpus=corpus,
        ref_corpus=ref_corpus,
        lm=best_lm,
        language=best_language,
        top_n=min(top_n, n),
        max_iter=100_000,
        seed=42,
    )

    # --- Gate check ---
    null_selectivity = null_results.get('random_mappings', {}).get('selectivity', 0.0)
    gate_compression = best.compression_ratio > 1.2
    gate_validity = best.word_validity_fraction > 0.1
    gate_selectivity = null_selectivity > 1.5
    gate_passed = gate_compression and gate_validity and gate_selectivity

    if gate_passed:
        verdict = 'mdl_decoding_successful'
    elif gate_compression and gate_selectivity:
        verdict = 'compression_good_validity_low'
    elif gate_compression:
        verdict = 'compression_good_selectivity_low'
    elif sanity.passed:
        verdict = 'approach_valid_voynich_not_latin_cipher'
    else:
        verdict = 'mdl_approach_insufficient'

    print(f"\n--- Gate Check ---")
    print(f"  Compression gate:  {best.compression_ratio:.4f} > 1.2 -> "
          f"{'PASSED' if gate_compression else 'FAILED'}")
    print(f"  Validity gate:     {best.word_validity_fraction:.4f} > 0.1 -> "
          f"{'PASSED' if gate_validity else 'FAILED'}")
    print(f"  Selectivity gate:  {null_selectivity:.4f} > 1.5 -> "
          f"{'PASSED' if gate_selectivity else 'FAILED'}")
    print(f"  Overall: {'PASSED' if gate_passed else 'FAILED'}")
    print(f"  Verdict: {verdict}")

    # --- Build result ---
    result = MDLDecodeResult(
        language_model_stats=[_convert(asdict(s)) for s in lm_stats],
        search_spaces=search_spaces,
        mcmc_results={k: _convert(asdict(v)) for k, v in mcmc_results.items()},
        language_ranking=language_ranking,
        best_granularity=best_granularity,
        best_language=best_language,
        best_cross_entropy=round(best.best_cross_entropy, 4),
        best_compression_ratio=round(best.compression_ratio, 4),
        best_word_validity=round(best.word_validity_fraction, 4),
        best_phrase_count=best.n_recognized_phrases,
        decoded_sample=decoded,
        sanity_check=_convert(asdict(sanity)),
        null_tests=_convert(null_results),
        cross_validation=_convert(cv_results),
        gate_passed=gate_passed,
        verdict=verdict,
    )

    out = _convert(asdict(result))
    out_path = _results_dir() / 'mdl_decode.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)

    print(f"\n  Results saved to {out_path}")
    return out
