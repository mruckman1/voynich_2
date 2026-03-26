"""
Phase 65, Step 4: Character LM Perplexity Minimization
=======================================================
Train a character-level n-gram LM on Latin pharmaceutical text (with
word boundary markers), then find the segmentation of the decoded
Voynich stream that minimizes perplexity via Viterbi DP.

Dependency chain:
    results/p65_decoded_stream.json  (Step 65.1)
    data/reference/latin/            (training data)
        -> results/p65_lm_segment.json
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
from voynich.core.reference import load_reference_corpus
from voynich.phases.suffix_calibration import SIGNAL_WORDS_SET


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
class LMSegmentResult:
    phase: str = "65"
    step: str = "65.4"
    experiment: str = "lm_segment"
    # LM stats
    lm_order: int = 5
    lm_vocab_size: int = 0
    lm_n_ngrams: int = 0
    training_chars: int = 0
    # Latin calibration
    latin_precision: float = 0.0
    latin_recall: float = 0.0
    latin_f1: float = 0.0
    latin_dict_hit: float = 0.0
    latin_perplexity_unseg: float = 0.0
    latin_perplexity_seg: float = 0.0
    # Voynich results
    n_boundaries: int = 0
    n_words: int = 0
    mean_word_length: float = 0.0
    dict_hit_rate: float = 0.0
    null_dict_hit_rate: float = 0.0
    selectivity: float = 0.0
    perplexity_unseg: float = 0.0
    perplexity_seg: float = 0.0
    top_words: List[Dict] = field(default_factory=list)
    word_length_dist: Dict[int, int] = field(default_factory=dict)
    sample_segmentation: str = ""
    signal_words_found: List[str] = field(default_factory=list)
    per_section: Dict[str, Dict] = field(default_factory=dict)
    # Gates
    g1_latin_f1: bool = False
    g2_dict_hit: bool = False
    g3_selectivity: bool = False
    g4_perplexity: bool = False
    g5_signal_words: bool = False
    gates_passed: int = 0
    gate_passed: bool = False
    verdict: str = ""
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Character Language Model
# ---------------------------------------------------------------------------

BOUNDARY = '#'


class CharLM:
    """Character-level n-gram language model with add-k smoothing."""

    def __init__(self, order: int = 5, smoothing_k: float = 0.01):
        self.order = order
        self.k = smoothing_k
        self.ngram_counts: Dict[str, Counter] = defaultdict(Counter)
        self.alphabet: set = set()
        self.total_chars = 0

    def train(self, texts_with_boundaries: List[str]):
        """Train on Latin texts where words are separated by BOUNDARY char."""
        for text in texts_with_boundaries:
            # Pad start with boundary chars
            padded = BOUNDARY * self.order + text
            for i in range(self.order, len(padded)):
                context = padded[i - self.order:i]
                char = padded[i]
                self.ngram_counts[context][char] += 1
                self.alphabet.add(char)
                self.total_chars += 1

        self.alphabet.add(BOUNDARY)
        self._vocab_size = len(self.alphabet)

    def log_prob(self, char: str, context: str) -> float:
        """Log2 probability of char given context."""
        # Pad context if too short
        if len(context) < self.order:
            context = (BOUNDARY * self.order + context)[-self.order:]

        counts = self.ngram_counts.get(context, Counter())
        total = sum(counts.values()) + self.k * self._vocab_size
        count = counts.get(char, 0) + self.k
        return math.log2(count / total) if total > 0 else -10.0

    def score_text(self, text: str) -> float:
        """Total log2 probability of text."""
        padded = BOUNDARY * self.order + text
        total = 0.0
        for i in range(self.order, len(padded)):
            context = padded[i - self.order:i]
            char = padded[i]
            total += self.log_prob(char, context)
        return total

    def perplexity(self, text: str) -> float:
        """Perplexity of text under the model."""
        n = len(text)
        if n == 0:
            return float('inf')
        log_prob = self.score_text(text)
        avg_log_prob = log_prob / n
        return 2.0 ** (-avg_log_prob)

    @property
    def n_ngrams(self) -> int:
        return sum(len(v) for v in self.ngram_counts.values())


# ---------------------------------------------------------------------------
# Viterbi segmentation
# ---------------------------------------------------------------------------

def viterbi_segment(
    stream: str, lm: CharLM,
    min_word_len: int = 2, max_word_len: int = 12,
) -> Tuple[List[str], List[int], float]:
    """Find optimal word boundary placement via Viterbi DP.

    Returns (words, boundaries, total_cost).
    """
    n = len(stream)
    INF = float('inf')

    # dp[i] = (min_cost, backpointer)
    dp_cost = [INF] * (n + 1)
    dp_back = [-1] * (n + 1)
    dp_cost[0] = 0.0

    for i in range(n):
        if dp_cost[i] == INF:
            continue

        for L in range(min_word_len, min(max_word_len, n - i) + 1):
            j = i + L
            word = stream[i:j]

            # Cost = negative log prob of word with boundary markers
            word_with_bounds = BOUNDARY + word + BOUNDARY
            cost = 0.0
            for k in range(len(word_with_bounds)):
                if k < lm.order:
                    ctx = (BOUNDARY * lm.order + word_with_bounds[:k])[-lm.order:]
                else:
                    ctx = word_with_bounds[k - lm.order:k]
                cost -= lm.log_prob(word_with_bounds[k], ctx)

            # Length prior: mild penalty for extreme lengths
            length_penalty = abs(L - 5.5) * 0.05

            total = dp_cost[i] + cost + length_penalty
            if total < dp_cost[j]:
                dp_cost[j] = total
                dp_back[j] = i

    # Backtrack
    boundaries = []
    pos = n
    while pos > 0:
        prev = dp_back[pos]
        if prev < 0:
            break
        if prev > 0:
            boundaries.append(prev)
        pos = prev

    boundaries = sorted(set(boundaries))
    words = _segment(stream, boundaries)
    return words, boundaries, dp_cost[n]


def _segment(stream: str, boundaries: List[int]) -> List[str]:
    if not boundaries:
        return [stream] if stream else []
    words = []
    prev = 0
    for b in sorted(set(boundaries)):
        if b > prev and b <= len(stream):
            words.append(stream[prev:b])
            prev = b
    if prev < len(stream):
        words.append(stream[prev:])
    return words


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _boundary_prf(predicted: List[int], gold: List[int], tolerance: int = 2):
    if not predicted or not gold:
        return 0.0, 0.0, 0.0
    gold_set = set(gold)
    tp = 0
    matched = set()
    for p in predicted:
        for off in range(-tolerance, tolerance + 1):
            if (p + off) in gold_set and (p + off) not in matched:
                tp += 1
                matched.add(p + off)
                break
    prec = tp / len(predicted) if predicted else 0.0
    rec = tp / len(gold) if gold else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def _build_10k_dict() -> set:
    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    freq = Counter(w.lower() for w in ref_corpus.get_combined_tokens('latin')
                   if len(w) >= 2 and w.isalpha())
    return set(w for w, _ in freq.most_common(10000))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_lm_segment():
    """Phase 65.4: Character LM perplexity minimization."""
    t0 = time.time()
    rd = str(_results_dir())
    print("=" * 70)
    print("Phase 65, Step 4: Character LM Perplexity Minimization")
    print("=" * 70)

    stream_data = _safe_load(os.path.join(rd, 'p65_decoded_stream.json'))
    if not stream_data:
        print("  ERROR: p65_decoded_stream.json not found.")
        return None

    full_text = stream_data['full_stream']['text']
    section_texts = stream_data.get('section_stream_texts', {})
    latin_streams = stream_data.get('latin_streams', [])
    dictionary = _build_10k_dict()
    print(f"  Dictionary: {len(dictionary)} words")

    # Train character LM on Latin reference
    print("\n  Training character LM...")
    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    latin_words = [w.lower() for w in ref_corpus.get_combined_tokens('latin')
                   if len(w) >= 2 and w.isalpha()]

    # Split 90/10 for train/calibrate
    n_train = int(len(latin_words) * 0.9)
    train_words = latin_words[:n_train]
    calib_words = latin_words[n_train:]

    # Build training text with boundary markers
    train_text = BOUNDARY.join(train_words)
    lm = CharLM(order=5, smoothing_k=0.01)
    lm.train([train_text])
    print(f"  LM trained: {lm.n_ngrams} n-grams, "
          f"{lm.total_chars} chars, vocab {lm._vocab_size}")

    # Latin calibration: segment held-out text
    print("\n  Latin calibration...")
    calib_stream = ''.join(calib_words)  # no boundaries
    calib_boundaries_gold = []
    pos = 0
    for w in calib_words:
        pos += len(w)
        calib_boundaries_gold.append(pos)

    # Perplexity of unsegmented vs segmented Latin
    latin_ppx_unseg = lm.perplexity(calib_stream)
    latin_ppx_seg = lm.perplexity(BOUNDARY.join(calib_words))
    print(f"  Latin perplexity: unseg={latin_ppx_unseg:.1f}, seg={latin_ppx_seg:.1f}")

    # Viterbi segment the calibration stream
    calib_words_found, calib_boundaries_found, _ = viterbi_segment(
        calib_stream, lm)
    prec, rec, f1 = _boundary_prf(calib_boundaries_found, calib_boundaries_gold)
    calib_dict_hits = sum(1 for w in calib_words_found if w in dictionary)
    calib_dict_rate = calib_dict_hits / len(calib_words_found) if calib_words_found else 0.0
    print(f"  Latin Viterbi: F1={f1:.3f} (P={prec:.3f}, R={rec:.3f}), "
          f"dict={calib_dict_rate:.3f}")

    # Apply to Voynich sections
    print("\n  Segmenting Voynich sections...")
    all_words: List[str] = []
    all_boundaries: List[int] = []
    per_section: Dict[str, Dict] = {}
    section_boundary_map: Dict[str, List[int]] = {}

    for section_key in sorted(section_texts.keys()):
        section_text = section_texts[section_key]
        if len(section_text) < 20:
            continue
        print(f"    {section_key} ({len(section_text)} chars)...", end=" ", flush=True)

        words, boundaries, cost = viterbi_segment(section_text, lm)
        section_boundary_map[section_key] = boundaries

        dict_hits = sum(1 for w in words if w in dictionary)
        dict_rate = dict_hits / len(words) if words else 0.0

        # Null comparison
        rng = np.random.default_rng(42)
        null_rates = []
        for _ in range(50):
            rb = sorted(rng.choice(len(section_text), size=max(1, len(boundaries)),
                                   replace=False).tolist())
            rw = _segment(section_text, rb)
            null_rates.append(sum(1 for w in rw if w in dictionary) / len(rw) if rw else 0.0)
        null_mean = float(np.mean(null_rates))
        sel = dict_rate / null_mean if null_mean > 0 else float('inf')

        ppx_unseg = lm.perplexity(section_text)
        ppx_seg = lm.perplexity(BOUNDARY.join(words))

        per_section[section_key] = {
            'n_words': len(words),
            'mean_length': round(float(np.mean([len(w) for w in words])), 2) if words else 0.0,
            'dict_hit_rate': round(dict_rate, 4),
            'null_dict_rate': round(null_mean, 4),
            'selectivity': round(sel, 3),
            'perplexity_unseg': round(ppx_unseg, 1),
            'perplexity_seg': round(ppx_seg, 1),
        }
        all_words.extend(words)
        print(f"{len(words)} words, dict {dict_rate:.3f}, sel {sel:.2f}x")

    # Aggregate
    if all_words:
        agg_dict_hits = sum(1 for w in all_words if w in dictionary)
        agg_dict_rate = agg_dict_hits / len(all_words)
    else:
        agg_dict_rate = 0.0

    # Full stream null
    total_boundaries = sum(len(b) for b in section_boundary_map.values())
    rng = np.random.default_rng(99)
    null_rates = []
    for _ in range(50):
        rb = sorted(rng.choice(len(full_text), size=max(1, total_boundaries),
                               replace=False).tolist())
        rw = _segment(full_text, rb)
        null_rates.append(sum(1 for w in rw if w in dictionary) / len(rw) if rw else 0.0)
    null_mean = float(np.mean(null_rates))
    selectivity = agg_dict_rate / null_mean if null_mean > 0 else float('inf')

    # Perplexity on full stream
    ppx_unseg = lm.perplexity(full_text)
    ppx_seg = lm.perplexity(BOUNDARY.join(all_words))

    # Signal words found
    signal_found = [w for w in set(all_words) if w in SIGNAL_WORDS_SET]
    top20 = Counter(all_words).most_common(20)
    top20_signal = [w for w, _ in top20 if w in SIGNAL_WORDS_SET]

    word_lengths = [len(w) for w in all_words]
    mean_wl = float(np.mean(word_lengths)) if word_lengths else 0.0

    # Gates
    g1 = f1 > 0.4
    g2 = agg_dict_rate > 0.15
    g3 = selectivity > 2.0
    g4 = ppx_seg < ppx_unseg * 0.8
    g5 = len(top20_signal) >= 3
    gates_passed = sum([g1, g2, g3, g4, g5])

    verdict = "LM_PASS" if gates_passed >= 3 else (
        "LM_PARTIAL" if gates_passed >= 2 else "LM_FAIL")

    print(f"\n  Aggregate: {len(all_words)} words, mean length {mean_wl:.1f}")
    print(f"  Dict hit: {agg_dict_rate:.3f}, null: {null_mean:.3f}, sel: {selectivity:.2f}x")
    print(f"  Perplexity: unseg={ppx_unseg:.1f}, seg={ppx_seg:.1f}")
    print(f"  Signal words in top-20: {top20_signal}")
    print(f"  Gates: L1(F1)={'PASS' if g1 else 'FAIL'} "
          f"L2(dict)={'PASS' if g2 else 'FAIL'} "
          f"L3(sel)={'PASS' if g3 else 'FAIL'} "
          f"L4(ppx)={'PASS' if g4 else 'FAIL'} "
          f"L5(signal)={'PASS' if g5 else 'FAIL'}")
    print(f"  Verdict: {verdict} ({gates_passed}/5)")

    result = LMSegmentResult(
        lm_order=5,
        lm_vocab_size=lm._vocab_size,
        lm_n_ngrams=lm.n_ngrams,
        training_chars=lm.total_chars,
        latin_precision=round(prec, 4),
        latin_recall=round(rec, 4),
        latin_f1=round(f1, 4),
        latin_dict_hit=round(calib_dict_rate, 4),
        latin_perplexity_unseg=round(latin_ppx_unseg, 1),
        latin_perplexity_seg=round(latin_ppx_seg, 1),
        n_boundaries=total_boundaries,
        n_words=len(all_words),
        mean_word_length=round(mean_wl, 2),
        dict_hit_rate=round(agg_dict_rate, 4),
        null_dict_hit_rate=round(null_mean, 4),
        selectivity=round(selectivity, 3),
        perplexity_unseg=round(ppx_unseg, 1),
        perplexity_seg=round(ppx_seg, 1),
        top_words=[{'word': w, 'count': c, 'in_dict': w in dictionary}
                   for w, c in Counter(all_words).most_common(30)],
        word_length_dist={str(k): v for k, v in Counter(word_lengths).items()},
        sample_segmentation=' '.join(all_words[:100])[:500],
        signal_words_found=sorted(signal_found),
        per_section=per_section,
        g1_latin_f1=g1,
        g2_dict_hit=g2,
        g3_selectivity=g3,
        g4_perplexity=g4,
        g5_signal_words=g5,
        gates_passed=gates_passed,
        gate_passed=gates_passed >= 3,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    save_data = asdict(result)
    save_data['section_boundaries'] = section_boundary_map
    _save_json(rd, 'p65_lm_segment.json', save_data)

    print(f"\n  Sample: {result.sample_segmentation[:200]}...")
    print(f"  Runtime: {result.runtime_seconds:.1f}s")
    return result
