"""
Step 33.5 – Latin Character-Level Language Model
==================================================
Builds a character-level n-gram language model on the Latin reference
corpus and uses it to score decoded Voynich text by perplexity (bits
per character).  The trained LM is serialized so that Steps 33.6 and
33.7 can reload it for perplexity-based triple optimization.

Algorithm:
  1. Load Latin reference text → lowercase, a-z + space only.
  2. Train 3-gram and 5-gram LMs with add-1 (Laplace) smoothing.
  3. Calibrate: held-out Latin ≈ 3-8 bpc, shuffled text ≈ 15-20 bpc.
  4. Score full decoded Voynich corpus (Phase 16, R3 strategy).
  5. Score SIGNAL-only tokens (should be lower = more Latin-like).

Dependency chain:
    combined_refine.json      (Phase 15 assignment)
    modifier_integrate.json   (Phase 16 modifiers)
    signal_bigrams.json       (Phase 29 token classifications)
        → latin_lm.json       (this step)
"""

import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import build_eva_to_triple_lookup, load_corpus
from voynich.core.reference import build_expanded_word_set, load_reference_corpus
from voynich.phases.signal_isolation import _decode_corpus_r3
from voynich.phases.null_corpus import _reconstruct_modifier_rules


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _convert(obj):
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


def _preprocess_text(raw: str) -> str:
    """Lowercase, keep only a-z and space, strip other characters."""
    out = []
    for ch in raw.lower():
        if 'a' <= ch <= 'z' or ch == ' ':
            out.append(ch)
    # Collapse multiple spaces
    text = ''.join(out)
    while '  ' in text:
        text = text.replace('  ', ' ')
    return text.strip()


# ---------------------------------------------------------------------------
# Character-level n-gram LM
# ---------------------------------------------------------------------------

class CharNgramLM:
    """Character-level n-gram language model with add-1 smoothing."""

    def __init__(self, order: int = 5, alpha: float = 1.0):
        self.order = order
        self.alpha = alpha  # Laplace smoothing parameter
        self.vocab: Set[str] = set()
        self.counts: Dict[str, Counter] = defaultdict(Counter)  # context -> {next_char: count}
        self.context_totals: Counter = Counter()  # context -> total count

    def train(self, text: str) -> None:
        """Train on a character string."""
        self.vocab = set(text)
        # Pad with boundary markers
        padded = '^' * (self.order - 1) + text + '$'
        for i in range(len(padded) - self.order + 1):
            context = padded[i:i + self.order - 1]
            next_char = padded[i + self.order - 1]
            self.counts[context][next_char] += 1
            self.context_totals[context] += 1

    def log_prob(self, text: str) -> float:
        """Compute total log2-probability of text."""
        padded = '^' * (self.order - 1) + text + '$'
        total = 0.0
        vocab_size = len(self.vocab) + 2  # +2 for ^ and $
        for i in range(len(padded) - self.order + 1):
            context = padded[i:i + self.order - 1]
            next_char = padded[i + self.order - 1]
            count = self.counts[context][next_char]
            total_count = self.context_totals[context]
            prob = (count + self.alpha) / (total_count + self.alpha * vocab_size)
            total += math.log2(prob)
        return total

    def perplexity(self, text: str) -> float:
        """Compute perplexity (2^bits_per_char)."""
        n = len(text) + 1  # +1 for the $ boundary
        log_p = self.log_prob(text)
        return 2.0 ** (-log_p / n) if n > 0 else float('inf')

    def bits_per_char(self, text: str) -> float:
        """Compute cross-entropy in bits per character."""
        n = len(text) + 1  # +1 for the $ boundary
        log_p = self.log_prob(text)
        return -log_p / n if n > 0 else float('inf')

    def serialize_counts(self, min_count: int = 1) -> Dict[str, Dict[str, int]]:
        """Convert counts to plain dicts, optionally pruning low-count contexts."""
        result: Dict[str, Dict[str, int]] = {}
        for ctx, char_counts in self.counts.items():
            total = self.context_totals[ctx]
            if total >= min_count:
                result[ctx] = dict(char_counts)
        return result


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class LatinLMResult:
    # Training info
    train_chars: int
    test_chars: int
    vocab_size: int
    # 3-gram calibration
    trigram_latin_bpc: float      # bits per char on held-out Latin
    trigram_shuffled_bpc: float   # bits per char on shuffled text
    trigram_gap: float
    # 5-gram calibration
    fivegram_latin_bpc: float
    fivegram_shuffled_bpc: float
    fivegram_gap: float
    # Decoded corpus scoring (using 5-gram)
    corpus_bpc: float             # decoded Voynich corpus bits/char
    signal_bpc: float             # SIGNAL tokens only bits/char
    non_signal_bpc: float         # non-SIGNAL tokens bits/char
    signal_vs_corpus_delta: float # signal_bpc - corpus_bpc (negative = more Latin-like)
    # LM parameters (stored for downstream use)
    lm_order: int
    lm_alpha: float
    lm_counts_trigram: Dict       # context -> {char: count} for 3-gram (serialized)
    lm_counts_fivegram: Dict      # context -> {char: count} for 5-gram (serialized)
    # Verdict
    calibration_valid: bool       # gap > 3 bits/char
    signal_more_latin: bool       # signal_bpc < corpus_bpc
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_latin_lm() -> None:
    """Step 33.5: Build and calibrate a Latin character-level language model."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 33.5: Latin Character-Level Language Model")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load Latin reference corpus ──
    print("\n  1. Loading Latin reference corpus …")
    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    raw_text = ref_corpus.get_combined_text('latin')
    full_text = _preprocess_text(raw_text)
    print(f"     Raw text: {len(raw_text)} chars → preprocessed: {len(full_text)} chars")

    # ── 2. Train/test split (80/20) ──
    print("\n  2. Splitting train/test (80/20) …")
    split_idx = int(len(full_text) * 0.8)
    train_text = full_text[:split_idx]
    test_text = full_text[split_idx:]
    print(f"     Train: {len(train_text)} chars")
    print(f"     Test:  {len(test_text)} chars")

    # ── 3. Train LMs ──
    print("\n  3. Training language models …")

    # 3-gram
    lm3 = CharNgramLM(order=3, alpha=1.0)
    lm3.train(train_text)
    vocab_size = len(lm3.vocab)
    print(f"     3-gram: {len(lm3.counts)} contexts, vocab={vocab_size}")

    # 5-gram
    lm5 = CharNgramLM(order=5, alpha=1.0)
    lm5.train(train_text)
    print(f"     5-gram: {len(lm5.counts)} contexts, vocab={vocab_size}")

    # ── 4. Calibration ──
    print("\n  4. Perplexity calibration …")

    # Held-out Latin
    trigram_latin_bpc = lm3.bits_per_char(test_text)
    fivegram_latin_bpc = lm5.bits_per_char(test_text)
    print(f"     Held-out Latin:")
    print(f"       3-gram bpc = {trigram_latin_bpc:.4f}")
    print(f"       5-gram bpc = {fivegram_latin_bpc:.4f}")

    # Shuffled text
    rng = random.Random(42)
    shuffled_chars = list(test_text)
    rng.shuffle(shuffled_chars)
    shuffled_text = ''.join(shuffled_chars)

    trigram_shuffled_bpc = lm3.bits_per_char(shuffled_text)
    fivegram_shuffled_bpc = lm5.bits_per_char(shuffled_text)
    print(f"     Shuffled text:")
    print(f"       3-gram bpc = {trigram_shuffled_bpc:.4f}")
    print(f"       5-gram bpc = {fivegram_shuffled_bpc:.4f}")

    trigram_gap = trigram_shuffled_bpc - trigram_latin_bpc
    fivegram_gap = fivegram_shuffled_bpc - fivegram_latin_bpc
    print(f"     Discrimination gap:")
    print(f"       3-gram: {trigram_gap:.4f} bits/char")
    print(f"       5-gram: {fivegram_gap:.4f} bits/char")

    # ── 5. Decode Voynich corpus ──
    print("\n  5. Decoding Voynich corpus (R3 strategy) …")

    # Load assignment
    refine_path = os.path.join(rd, 'combined_refine.json')
    if not os.path.exists(refine_path):
        print("  [SKIP] combined_refine.json not found")
        return
    with open(refine_path) as f:
        refine_data = json.load(f)
    assignment = refine_data.get('best_assignment', {})

    # Load modifiers
    mod_path = os.path.join(rd, 'modifier_integrate.json')
    if not os.path.exists(mod_path):
        print("  [SKIP] modifier_integrate.json not found")
        return
    with open(mod_path) as f:
        mod_data = json.load(f)
    modifier_chars, modifier_rules = _reconstruct_modifier_rules(mod_data)

    # Build reference word set (needed by R3 decode)
    base_words = set(w.lower() for w in ref_corpus.get_combined_tokens('latin')
                     if len(w) >= 2)
    expanded, _ = build_expanded_word_set(base_words)
    ref_word_set = base_words | expanded

    # Load corpus and decode
    eva_to_triple = build_eva_to_triple_lookup()
    corpus = load_corpus(verbose=False)
    all_tokens = corpus.get_tokens()
    n_tokens = len(all_tokens)

    decoded = _decode_corpus_r3(
        all_tokens, assignment, eva_to_triple,
        modifier_chars, modifier_rules, ref_word_set,
    )
    print(f"     {n_tokens} tokens decoded")

    # ── 6. Score decoded corpus with 5-gram LM ──
    print("\n  6. Scoring decoded corpus …")

    corpus_text = _preprocess_text(' '.join(decoded))
    corpus_bpc = lm5.bits_per_char(corpus_text)
    print(f"     Full corpus bpc = {corpus_bpc:.4f}")

    # ── 7. Score SIGNAL-only and non-SIGNAL tokens ──
    print("\n  7. Scoring SIGNAL vs non-SIGNAL tokens …")

    signal_bpc = float('inf')
    non_signal_bpc = float('inf')
    signal_vs_corpus_delta = 0.0

    sig_path = os.path.join(rd, 'signal_bigrams.json')
    if os.path.exists(sig_path):
        with open(sig_path) as f:
            sig_data = json.load(f)
        classifications = sig_data.get('token_classifications', [])

        if len(classifications) == len(decoded):
            signal_words = [
                decoded[i] for i in range(len(decoded))
                if classifications[i] == 'SIGNAL'
            ]
            non_signal_words = [
                decoded[i] for i in range(len(decoded))
                if classifications[i] != 'SIGNAL'
            ]

            if signal_words:
                signal_text = _preprocess_text(' '.join(signal_words))
                signal_bpc = lm5.bits_per_char(signal_text)
            if non_signal_words:
                non_signal_text = _preprocess_text(' '.join(non_signal_words))
                non_signal_bpc = lm5.bits_per_char(non_signal_text)

            signal_vs_corpus_delta = signal_bpc - corpus_bpc

            print(f"     SIGNAL tokens: {len(signal_words)}, bpc = {signal_bpc:.4f}")
            print(f"     Non-SIGNAL tokens: {len(non_signal_words)}, bpc = {non_signal_bpc:.4f}")
            print(f"     SIGNAL vs corpus delta = {signal_vs_corpus_delta:+.4f}")
        else:
            print(f"     [WARN] Classification length mismatch: "
                  f"{len(classifications)} vs {len(decoded)} decoded tokens")
    else:
        print("     [WARN] signal_bigrams.json not found — skipping SIGNAL split")

    # ── 8. Serialize LM counts ──
    print("\n  8. Serializing LM counts …")
    lm_counts_trigram = lm3.serialize_counts(min_count=2)
    lm_counts_fivegram = lm5.serialize_counts(min_count=2)
    print(f"     3-gram contexts (count>=2): {len(lm_counts_trigram)}")
    print(f"     5-gram contexts (count>=2): {len(lm_counts_fivegram)}")

    # ── 9. Verdict ──
    calibration_valid = fivegram_gap > 3.0
    signal_more_latin = signal_bpc < corpus_bpc

    verdict_parts = []
    if calibration_valid:
        verdict_parts.append(
            f"CALIBRATION_VALID: 5-gram gap = {fivegram_gap:.2f} bits/char (>3.0)"
        )
    else:
        verdict_parts.append(
            f"CALIBRATION_WEAK: 5-gram gap = {fivegram_gap:.2f} bits/char (<=3.0)"
        )

    verdict_parts.append(
        f"Corpus bpc = {corpus_bpc:.2f}, "
        f"Latin bpc = {fivegram_latin_bpc:.2f}, "
        f"shuffled bpc = {fivegram_shuffled_bpc:.2f}"
    )

    if signal_bpc != float('inf'):
        if signal_more_latin:
            verdict_parts.append(
                f"SIGNAL_MORE_LATIN: signal bpc={signal_bpc:.2f} < "
                f"corpus bpc={corpus_bpc:.2f} (delta={signal_vs_corpus_delta:+.2f})"
            )
        else:
            verdict_parts.append(
                f"SIGNAL_NOT_MORE_LATIN: signal bpc={signal_bpc:.2f} >= "
                f"corpus bpc={corpus_bpc:.2f} (delta={signal_vs_corpus_delta:+.2f})"
            )

    verdict = '; '.join(verdict_parts)

    print(f"\n  Calibration: {'VALID' if calibration_valid else 'WEAK'}")
    print(f"  Signal more Latin: {signal_more_latin}")
    print(f"  {verdict}")

    # ── 10. Save ──
    elapsed = round(time.time() - t0, 2)

    result = LatinLMResult(
        train_chars=len(train_text),
        test_chars=len(test_text),
        vocab_size=vocab_size,
        trigram_latin_bpc=round(trigram_latin_bpc, 6),
        trigram_shuffled_bpc=round(trigram_shuffled_bpc, 6),
        trigram_gap=round(trigram_gap, 6),
        fivegram_latin_bpc=round(fivegram_latin_bpc, 6),
        fivegram_shuffled_bpc=round(fivegram_shuffled_bpc, 6),
        fivegram_gap=round(fivegram_gap, 6),
        corpus_bpc=round(corpus_bpc, 6),
        signal_bpc=round(signal_bpc, 6) if signal_bpc != float('inf') else None,
        non_signal_bpc=round(non_signal_bpc, 6) if non_signal_bpc != float('inf') else None,
        signal_vs_corpus_delta=round(signal_vs_corpus_delta, 6),
        lm_order=5,
        lm_alpha=1.0,
        lm_counts_trigram=lm_counts_trigram,
        lm_counts_fivegram=lm_counts_fivegram,
        calibration_valid=calibration_valid,
        signal_more_latin=signal_more_latin,
        verdict=verdict,
        runtime_seconds=elapsed,
    )

    out_path = os.path.join(rd, 'latin_lm.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)
    print(f"\n  → {out_path}  ({elapsed:.1f}s)")
