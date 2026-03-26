"""
Phase 66, Track 2: Reverse Simulation (Viterbi Word-Level Decode)
==================================================================
Builds a forward encoding model P(decoded | Latin_word), trains a
word-level Latin bigram language model, then runs Viterbi decoding
on real + shuffled + null passages. Compares log-probabilities.

Dependency chain:
    results/combined_refine.json      (Phase 15)
    data/reference/latin/             (Latin reference corpus)
    p66_validation.py                 (shared controls)
        -> results/p66_reverse_sim.json
"""
from __future__ import annotations

import json
import math
import os
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import build_eva_to_triple_lookup, load_corpus
from voynich.core.reference import load_reference_corpus
from voynich.core.stats import syllabify_latin
from voynich.phases.corrected_coda import (
    build_coda_table_v2,
    decode_corpus_cvc_v2,
    decode_token_cvc_v2,
)
from voynich.phases.p66_validation import (
    _edit_distance,
    forward_encode_word,
    generate_controls,
    score_against_controls,
)
from voynich.phases.suffix_calibration import SIGNAL_WORDS_51


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
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ReverseSimResult:
    phase: str = "66"
    step: str = "66.2"
    experiment: str = "reverse_simulation"
    vocab_size: int = 0
    n_forward_encoded: int = 0
    n_real: int = 0
    n_shuffled: int = 0
    n_null: int = 0
    # Aggregate
    real_mean_logprob: float = 0.0
    shuffled_mean_logprob: float = 0.0
    null_mean_logprob: float = 0.0
    real_mean_coverage: float = 0.0
    shuffled_mean_coverage: float = 0.0
    null_mean_coverage: float = 0.0
    real_mean_n_words: float = 0.0
    # Per-passage results (top 5 only for JSON size)
    top_real_readings: List[Dict] = field(default_factory=list)
    # Gates
    r1_logprob_vs_shuf: bool = False
    r2_logprob_vs_null: bool = False
    r3_recognizable_words: bool = False
    r4_coverage_gap: bool = False
    gates_passed: int = 0
    gate_passed: bool = False
    verdict: str = ""
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Vocabulary + forward model
# ---------------------------------------------------------------------------

def _build_vocab(
    ref_corpus: Any,
    n: int = 1000,
) -> List[str]:
    """Top n most frequent Latin words + signal words."""
    all_tokens = ref_corpus.get_combined_tokens('latin')
    freq = Counter(w.lower().strip('.,;:!?()[]') for w in all_tokens)
    # Remove very short words and non-alpha
    freq = {w: c for w, c in freq.items() if len(w) >= 2 and w.isalpha()}
    top = [w for w, _ in sorted(freq.items(), key=lambda x: -x[1])[:n]]
    # Add signal words
    for sw in SIGNAL_WORDS_51:
        if sw not in top:
            top.append(sw)
    return top


def _build_forward_model(
    vocab: List[str],
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    coda_table: Any,
) -> Dict[str, str]:
    """For each vocab word, compute expected decoded form.

    Returns {latin_word: decoded_form} for words that encode successfully.
    """
    model = {}
    for word in vocab:
        encoded = forward_encode_word(word, assignment, eva_to_triple,
                                       coda_table)
        if encoded and len(encoded) >= 2:
            model[word] = encoded
    return model


# ---------------------------------------------------------------------------
# Word-level bigram LM
# ---------------------------------------------------------------------------

def _build_word_bigram_lm(
    ref_corpus: Any,
    vocab_set: Set[str],
    alpha: float = 0.1,
) -> Tuple[Dict[Tuple[str, str], float], Dict[str, float]]:
    """Smoothed word bigram model.

    Returns (bigram_probs, unigram_probs) where probs are log-probabilities.
    """
    all_tokens = ref_corpus.get_combined_tokens('latin')
    words = [w.lower().strip('.,;:!?()[]') for w in all_tokens
             if w.lower().strip('.,;:!?()[]') in vocab_set]

    unigram_counts: Counter = Counter(words)
    bigram_counts: Counter = Counter()
    for i in range(len(words) - 1):
        bigram_counts[(words[i], words[i + 1])] += 1

    total = sum(unigram_counts.values())
    V = len(vocab_set)

    unigram_probs = {}
    for w in vocab_set:
        unigram_probs[w] = math.log(
            (unigram_counts.get(w, 0) + alpha) / (total + alpha * V))

    bigram_probs = {}
    for (w1, w2), count in bigram_counts.items():
        denom = unigram_counts.get(w1, 0) + alpha * V
        bigram_probs[(w1, w2)] = math.log((count + alpha) / denom)

    return bigram_probs, unigram_probs


# ---------------------------------------------------------------------------
# Viterbi decoder
# ---------------------------------------------------------------------------

def _viterbi_segment(
    passage: str,
    vocab_decoded: Dict[str, str],
    bigram_probs: Dict[Tuple[str, str], float],
    unigram_probs: Dict[str, float],
    default_logprob: float = -20.0,
    max_word_len: int = 12,
    ed_penalty: float = 2.0,
) -> Tuple[List[str], float, float]:
    """Viterbi DP to find best word segmentation of decoded stream.

    Returns (best_words, total_log_prob, coverage).
    """
    n = len(passage)
    if n == 0:
        return [], 0.0, 0.0

    # Precompute: for each position and length, find best-matching word
    # Cache: (start, length) -> (best_word, log_p_encoding)
    match_cache: Dict[Tuple[int, int], Tuple[Optional[str], float]] = {}

    for start in range(n):
        for length in range(2, min(max_word_len, n - start) + 1):
            observed = passage[start:start + length]
            best_word = None
            best_score = -float('inf')

            for word, expected in vocab_decoded.items():
                # Quick length filter
                if abs(len(expected) - length) > 3:
                    continue

                ed = _edit_distance(observed, expected)
                max_len = max(len(observed), len(expected), 1)
                log_p = -ed_penalty * ed / max_len

                if log_p > best_score:
                    best_score = log_p
                    best_word = word

            match_cache[(start, length)] = (best_word, best_score)

    # DP: dp[i] = (best_total_log_prob, backpointer, last_word)
    INF = float('inf')
    dp: List[Tuple[float, int, Optional[str]]] = [
        (-INF, -1, None)] * (n + 1)
    dp[0] = (0.0, -1, None)

    for i in range(n):
        if dp[i][0] == -INF:
            continue

        for length in range(2, min(max_word_len, n - i) + 1):
            best_word, enc_score = match_cache.get(
                (i, length), (None, -INF))
            if best_word is None:
                continue

            # Language model score
            prev_word = dp[i][2]
            if prev_word and (prev_word, best_word) in bigram_probs:
                lm_score = bigram_probs[(prev_word, best_word)]
            else:
                lm_score = unigram_probs.get(best_word, default_logprob)

            total = dp[i][0] + enc_score + lm_score
            j = i + length

            if total > dp[j][0]:
                dp[j] = (total, i, best_word)

        # Also allow skipping 1 character (UNK)
        unk_penalty = -5.0
        j = i + 1
        if dp[i][0] + unk_penalty > dp[j][0]:
            dp[j] = (dp[i][0] + unk_penalty, i, '<UNK>')

    # Backtrack
    words = []
    pos = n
    covered = 0
    while pos > 0 and dp[pos][1] >= 0:
        word = dp[pos][2]
        start = dp[pos][1]
        if word != '<UNK>':
            covered += pos - start
        words.append(word)
        pos = start
    words.reverse()

    total_logprob = dp[n][0] if dp[n][0] > -INF else -INF
    coverage = covered / n if n > 0 else 0.0

    # Filter out UNK
    real_words = [w for w in words if w != '<UNK>']

    return real_words, total_logprob, coverage


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_reverse_sim() -> None:
    """Phase 66, Track 2: Reverse Simulation (Viterbi)."""
    t0 = time.time()
    rd = str(_results_dir())

    print("=" * 70)
    print("Phase 66, Track 2: Reverse Simulation (Viterbi)")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Load dependencies
    # ------------------------------------------------------------------
    print("\n[1] Loading dependencies...")
    corpus = load_corpus(verbose=False)
    cr = _safe_load(os.path.join(rd, 'combined_refine.json'))
    assignment = cr.get('best_assignment', {})
    eva_to_triple = build_eva_to_triple_lookup()
    coda_table = build_coda_table_v2()

    if not assignment:
        print("  ERROR: No assignment table found")
        result = ReverseSimResult(
            verdict="ERROR — no assignment table",
            runtime_seconds=round(time.time() - t0, 2),
        )
        _save_json(rd, 'p66_reverse_sim.json', asdict(result))
        return

    ref = load_reference_corpus(languages=['latin'], verbose=False)
    print(f"  Reference corpus: {len(ref.get_combined_tokens('latin'))} tokens")

    # ------------------------------------------------------------------
    # Build vocabulary and models
    # ------------------------------------------------------------------
    print("\n[2] Building vocabulary and models...")
    vocab = _build_vocab(ref, n=1000)
    vocab_set = set(vocab)
    print(f"  Vocabulary: {len(vocab)} words")

    vocab_decoded = _build_forward_model(
        vocab, assignment, eva_to_triple, coda_table)
    print(f"  Forward-encoded: {len(vocab_decoded)}/{len(vocab)} words")

    bigram_probs, unigram_probs = _build_word_bigram_lm(ref, vocab_set)
    print(f"  Bigram LM: {len(bigram_probs)} bigrams")

    # ------------------------------------------------------------------
    # Select passages
    # ------------------------------------------------------------------
    print("\n[3] Selecting passages...")
    all_corpus_tokens = []
    for page in corpus.pages.values():
        all_corpus_tokens.extend(page.all_tokens)

    # Select 20 real passages
    real_passages = []
    for folio_id, page in sorted(corpus.pages.items()):
        if len(real_passages) >= 20:
            break
        tokens = page.all_tokens
        if len(tokens) < 5:
            continue
        decoded = decode_corpus_cvc_v2(
            tokens, assignment, eva_to_triple, coda_table)
        stream = ''.join(d for d in decoded if d and '?' not in d)
        if len(stream) < 40:
            continue
        stream = stream[:200]
        real_passages.append({
            'stream': stream,
            'folio': folio_id,
            'section': page.section or 'unknown',
            'control_type': None,
        })

    print(f"  Selected {len(real_passages)} real passages")

    # Generate controls
    shuffled, nulls = generate_controls(
        real_passages, all_corpus_tokens, assignment,
        eva_to_triple, coda_table, base_seed=42)
    print(f"  Generated {len(shuffled)} shuffled + {len(nulls)} null controls")

    # ------------------------------------------------------------------
    # Run Viterbi on all passages
    # ------------------------------------------------------------------
    print("\n[4] Running Viterbi decode...")
    all_passages = (
        [(p, 'real') for p in real_passages]
        + [(p, 'shuffled') for p in shuffled]
        + [(p, 'null') for p in nulls]
    )

    real_logprobs = []
    shuf_logprobs = []
    null_logprobs = []
    real_coverages = []
    shuf_coverages = []
    null_coverages = []
    real_word_counts = []
    top_readings = []

    for i, (passage, ptype) in enumerate(all_passages):
        words, logprob, coverage = _viterbi_segment(
            passage['stream'], vocab_decoded, bigram_probs, unigram_probs)

        if ptype == 'real':
            real_logprobs.append(logprob)
            real_coverages.append(coverage)
            real_word_counts.append(len(words))
            if len(top_readings) < 5:
                top_readings.append({
                    'folio': passage.get('folio', '?'),
                    'stream_len': len(passage['stream']),
                    'words': words[:20],
                    'n_words': len(words),
                    'logprob': round(logprob, 2),
                    'coverage': round(coverage, 4),
                    'reading': ' '.join(words[:20]),
                })
        elif ptype == 'shuffled':
            shuf_logprobs.append(logprob)
            shuf_coverages.append(coverage)
        elif ptype == 'null':
            null_logprobs.append(logprob)
            null_coverages.append(coverage)

        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{len(all_passages)}...")

    # ------------------------------------------------------------------
    # Compare
    # ------------------------------------------------------------------
    print("\n[5] Comparing real vs controls...")
    real_mean_lp = float(np.mean(real_logprobs)) if real_logprobs else -999.0
    shuf_mean_lp = float(np.mean(shuf_logprobs)) if shuf_logprobs else -999.0
    null_mean_lp = float(np.mean(null_logprobs)) if null_logprobs else -999.0

    real_mean_cov = float(np.mean(real_coverages)) if real_coverages else 0.0
    shuf_mean_cov = float(np.mean(shuf_coverages)) if shuf_coverages else 0.0
    null_mean_cov = float(np.mean(null_coverages)) if null_coverages else 0.0

    real_mean_nw = float(np.mean(real_word_counts)) if real_word_counts else 0.0

    print(f"  Real mean logprob:    {real_mean_lp:.2f}")
    print(f"  Shuffled mean logprob: {shuf_mean_lp:.2f}")
    print(f"  Null mean logprob:     {null_mean_lp:.2f}")
    print(f"  Real mean coverage:    {real_mean_cov:.3f}")
    print(f"  Shuffled mean cov:     {shuf_mean_cov:.3f}")

    # ------------------------------------------------------------------
    # Gates
    # ------------------------------------------------------------------
    r1 = real_mean_lp > shuf_mean_lp
    r2 = real_mean_lp > null_mean_lp
    r3 = sum(1 for wc in real_word_counts if wc >= 3) >= 5
    r4 = real_mean_cov > shuf_mean_cov + 0.05

    gates = [r1, r2, r3, r4]
    gates_passed = sum(gates)

    if gates_passed >= 3:
        verdict = "REVERSE_SIGNAL"
    elif gates_passed >= 2:
        verdict = "WEAK_SIGNAL"
    else:
        verdict = "NO_SIGNAL"

    print(f"\n  R1 logprob>shuffled: {'PASS' if r1 else 'FAIL'}")
    print(f"  R2 logprob>null: {'PASS' if r2 else 'FAIL'}")
    print(f"  R3 ≥5 with ≥3 words: {'PASS' if r3 else 'FAIL'}")
    print(f"  R4 coverage gap: {'PASS' if r4 else 'FAIL'}")
    print(f"  Gates: {gates_passed}/4 → {verdict}")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    result = ReverseSimResult(
        vocab_size=len(vocab),
        n_forward_encoded=len(vocab_decoded),
        n_real=len(real_passages),
        n_shuffled=len(shuffled),
        n_null=len(nulls),
        real_mean_logprob=round(real_mean_lp, 2),
        shuffled_mean_logprob=round(shuf_mean_lp, 2),
        null_mean_logprob=round(null_mean_lp, 2),
        real_mean_coverage=round(real_mean_cov, 4),
        shuffled_mean_coverage=round(shuf_mean_cov, 4),
        null_mean_coverage=round(null_mean_cov, 4),
        real_mean_n_words=round(real_mean_nw, 1),
        top_real_readings=top_readings,
        r1_logprob_vs_shuf=r1,
        r2_logprob_vs_null=r2,
        r3_recognizable_words=r3,
        r4_coverage_gap=r4,
        gates_passed=gates_passed,
        gate_passed=gates_passed >= 2,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    _save_json(rd, 'p66_reverse_sim.json', asdict(result))
    print(f"\n  Saved to results/p66_reverse_sim.json")
    print(f"  Runtime: {result.runtime_seconds}s")
