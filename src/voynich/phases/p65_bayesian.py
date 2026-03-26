"""
Phase 65, Step 3: Bayesian Word Segmentation (MDL / Brent 1999)
================================================================
Find the word segmentation that minimizes description length:
  DL = codebook_cost + data_cost
     = |V|*avg_word_len*log(|alphabet|) + sum(-count(w)*log(count(w)/N))

Solved via iterative DP: segment, re-estimate word frequencies, repeat.
This is a Bayesian MAP estimate under a unigram Dirichlet-multinomial model.

Dependency chain:
    results/p65_decoded_stream.json  (Step 65.1)
        -> results/p65_bayesian.json
"""

import json
import math
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np

from voynich.core._paths import results_dir as _results_dir
from voynich.core.reference import load_reference_corpus


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
class BayesianResult:
    phase: str = "65"
    step: str = "65.3"
    experiment: str = "bayesian_segment"
    # Configuration
    n_iterations: int = 0
    alpha: float = 0.0
    # Latin calibration
    latin_best_alpha: float = 0.0
    latin_precision: float = 0.0
    latin_recall: float = 0.0
    latin_f1: float = 0.0
    latin_calibration: List[Dict] = field(default_factory=list)
    # Voynich results
    n_boundaries: int = 0
    n_words: int = 0
    n_types: int = 0
    mean_word_length: float = 0.0
    dict_hit_rate: float = 0.0
    null_dict_hit_rate: float = 0.0
    selectivity: float = 0.0
    top_words: List[Dict] = field(default_factory=list)
    word_length_dist: Dict[int, int] = field(default_factory=dict)
    sample_segmentation: str = ""
    convergence_log: List[Dict] = field(default_factory=list)
    per_section: Dict[str, Dict] = field(default_factory=dict)
    # Gates
    g1_word_length: bool = False
    g2_dict_hit: bool = False
    g3_selectivity: bool = False
    g4_top20_in_dict: bool = False
    gates_passed: int = 0
    gate_passed: bool = False
    verdict: str = ""
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# MDL word segmentation via iterative Viterbi
# ---------------------------------------------------------------------------

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


def viterbi_mdl(
    stream: str, word_freq: Counter, total_words: int,
    alpha: float = 1.0, min_len: int = 2, max_len: int = 12,
    alphabet_size: int = 26,
) -> Tuple[List[str], List[int], float]:
    """Viterbi segmentation minimizing MDL cost.

    Cost of a word w = -log P(w) where:
    P(w) = (count(w) + alpha) / (total + alpha * V_est)
    V_est = estimated vocabulary size

    For unseen words: P(w) = alpha / (total + alpha * V_est)
    with word-length penalty: cost += len(w) * log(alphabet_size) * penalty_weight
    """
    n = len(stream)
    if n == 0:
        return [], [], 0.0

    INF = float('inf')
    v_est = max(len(word_freq), 100)
    denom = total_words + alpha * v_est

    dp_cost = [INF] * (n + 1)
    dp_back = [-1] * (n + 1)
    dp_cost[0] = 0.0

    for i in range(n):
        if dp_cost[i] == INF:
            continue
        for L in range(min_len, min(max_len, n - i) + 1):
            j = i + L
            word = stream[i:j]
            count = word_freq.get(word, 0)

            if count > 0:
                # Known word: cheap
                cost = -math.log((count + alpha) / denom)
            else:
                # Unknown word: expensive (MDL codebook cost)
                cost = -math.log(alpha / denom) + L * math.log(alphabet_size) * 0.1

            # Mild length prior: prefer words near mean length 5.5
            cost += abs(L - 5.5) * 0.02

            total = dp_cost[i] + cost
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


def iterative_segment(
    stream: str, n_iterations: int = 10, alpha: float = 1.0,
    min_len: int = 2, max_len: int = 12,
) -> Tuple[List[str], List[int], List[Dict]]:
    """Iterative MDL segmentation: segment, count, re-segment.

    Start with uniform word frequencies, then iterate:
    1. Viterbi segment using current word frequencies
    2. Update word frequencies from the new segmentation
    3. Repeat until convergence or max iterations
    """
    alphabet_size = len(set(stream))

    # Initialize: uniform word freq (all substrings equally likely)
    word_freq: Counter = Counter()
    total_words = 0
    convergence_log: List[Dict] = []
    prev_boundaries = None

    for iteration in range(n_iterations):
        words, boundaries, cost = viterbi_mdl(
            stream, word_freq, total_words, alpha=alpha,
            min_len=min_len, max_len=max_len,
            alphabet_size=alphabet_size,
        )

        # Update word frequencies
        word_freq = Counter(words)
        total_words = len(words)

        lengths = [len(w) for w in words]
        convergence_log.append({
            'iteration': iteration,
            'n_words': len(words),
            'n_types': len(word_freq),
            'mean_length': round(float(np.mean(lengths)), 2) if lengths else 0.0,
            'cost': round(cost, 1),
        })

        # Check convergence
        if boundaries == prev_boundaries:
            break
        prev_boundaries = boundaries

    return words, boundaries, convergence_log


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


def _eval_seg(stream: str, boundaries: List[int], dictionary: set, n_null: int = 50):
    words = _segment(stream, boundaries)
    if not words:
        return {'n_words': 0, 'dict_hit_rate': 0.0, 'selectivity': 0.0}
    dict_hits = sum(1 for w in words if w in dictionary)
    dict_rate = dict_hits / len(words)

    rng = np.random.default_rng(123)
    null_rates = []
    for _ in range(n_null):
        rb = sorted(rng.choice(max(1, len(stream) - 1),
                               size=min(len(boundaries), len(stream) - 1),
                               replace=False).tolist())
        rw = _segment(stream, rb)
        null_rates.append(sum(1 for w in rw if w in dictionary) / len(rw) if rw else 0.0)
    null_mean = float(np.mean(null_rates))
    sel = dict_rate / null_mean if null_mean > 0 else float('inf')

    return {
        'n_words': len(words),
        'n_types': len(set(words)),
        'mean_length': round(float(np.mean([len(w) for w in words])), 2),
        'dict_hit_rate': round(dict_rate, 4),
        'null_dict_rate': round(null_mean, 4),
        'selectivity': round(sel, 3),
        'top_words': [{'word': w, 'count': c, 'in_dict': w in dictionary}
                      for w, c in Counter(words).most_common(30)],
        'word_length_dist': dict(Counter(len(w) for w in words)),
    }


def _build_10k_dict() -> set:
    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    freq = Counter(w.lower() for w in ref_corpus.get_combined_tokens('latin')
                   if len(w) >= 2 and w.isalpha())
    return set(w for w, _ in freq.most_common(10000))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_bayesian_segment():
    """Phase 65.3: Bayesian (MDL) word segmentation."""
    t0 = time.time()
    rd = str(_results_dir())
    print("=" * 70)
    print("Phase 65, Step 3: Bayesian (MDL) Word Segmentation")
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

    # Latin calibration
    print("\n  Latin calibration...")
    latin_calib = []
    best_alpha = 1.0
    best_latin_f1 = 0.0

    if latin_streams:
        latin_text = latin_streams[0]['text']
        latin_boundaries = latin_streams[0]['word_boundaries']
        # Subsample for calibration
        calib_len = min(5000, len(latin_text))
        calib_text = latin_text[:calib_len]
        calib_bounds = [b for b in latin_boundaries if b <= calib_len]

        for alpha in [0.1, 0.5, 1.0, 5.0, 10.0]:
            print(f"    alpha={alpha}...", end=" ", flush=True)
            words, boundaries, _ = iterative_segment(
                calib_text, n_iterations=10, alpha=alpha)
            prec, rec, f1 = _boundary_prf(boundaries, calib_bounds)
            latin_calib.append({
                'alpha': alpha, 'precision': round(prec, 4),
                'recall': round(rec, 4), 'f1': round(f1, 4),
                'n_boundaries': len(boundaries),
                'n_words': len(words),
            })
            print(f"F1={f1:.3f} ({len(boundaries)} boundaries, {len(words)} words)")
            if f1 > best_latin_f1:
                best_latin_f1 = f1
                best_alpha = alpha

    print(f"  Best alpha: {best_alpha}, Latin F1: {best_latin_f1:.3f}")

    # Apply to Voynich sections
    print(f"\n  Segmenting Voynich (alpha={best_alpha})...")
    all_words: List[str] = []
    per_section: Dict[str, Dict] = {}
    section_boundary_map: Dict[str, List[int]] = {}
    all_convergence: List[Dict] = []

    for section_key in sorted(section_texts.keys()):
        section_text = section_texts[section_key]
        if len(section_text) < 20:
            continue
        print(f"    {section_key} ({len(section_text)} chars)...", end=" ", flush=True)

        words, boundaries, conv_log = iterative_segment(
            section_text, n_iterations=10, alpha=best_alpha)
        section_boundary_map[section_key] = boundaries
        all_convergence.extend(conv_log)

        sec_eval = _eval_seg(section_text, boundaries, dictionary)
        per_section[section_key] = sec_eval
        all_words.extend(words)

        print(f"{sec_eval['n_words']} words, dict {sec_eval['dict_hit_rate']:.3f}, "
              f"sel {sec_eval['selectivity']:.2f}x")

    # Aggregate
    if all_words:
        agg_dict_hits = sum(1 for w in all_words if w in dictionary)
        agg_dict_rate = agg_dict_hits / len(all_words)
    else:
        agg_dict_rate = 0.0

    # Null comparison on full stream
    total_boundaries = sum(len(b) for b in section_boundary_map.values())
    rng = np.random.default_rng(99)
    null_rates = []
    for _ in range(50):
        rb = sorted(rng.choice(max(1, len(full_text) - 1),
                               size=min(total_boundaries, len(full_text) - 1),
                               replace=False).tolist())
        rw = _segment(full_text, rb)
        null_rates.append(sum(1 for w in rw if w in dictionary) / len(rw) if rw else 0.0)
    null_mean = float(np.mean(null_rates))
    selectivity = agg_dict_rate / null_mean if null_mean > 0 else float('inf')

    word_lengths = [len(w) for w in all_words]
    mean_wl = float(np.mean(word_lengths)) if word_lengths else 0.0

    top20 = Counter(all_words).most_common(20)
    top20_in_dict = sum(1 for w, _ in top20 if w in dictionary)

    # Gates
    g1 = 4.0 <= mean_wl <= 8.0
    g2 = agg_dict_rate > 0.10
    g3 = selectivity > 1.5
    g4 = top20_in_dict >= 5
    gates_passed = sum([g1, g2, g3, g4])

    verdict = "BAYESIAN_PASS" if gates_passed >= 3 else (
        "BAYESIAN_PARTIAL" if gates_passed >= 2 else "BAYESIAN_FAIL")

    print(f"\n  Aggregate: {len(all_words)} words, {len(set(all_words))} types")
    print(f"  Dict hit: {agg_dict_rate:.3f}, null: {null_mean:.3f}, sel: {selectivity:.2f}x")
    print(f"  Mean word length: {mean_wl:.1f}")
    print(f"  Top-20 in dict: {top20_in_dict}")
    print(f"  Gates: B1(length)={'PASS' if g1 else 'FAIL'} "
          f"B2(dict)={'PASS' if g2 else 'FAIL'} "
          f"B3(sel)={'PASS' if g3 else 'FAIL'} "
          f"B4(top20)={'PASS' if g4 else 'FAIL'}")
    print(f"  Verdict: {verdict} ({gates_passed}/4)")

    result = BayesianResult(
        n_iterations=10,
        alpha=best_alpha,
        latin_best_alpha=best_alpha,
        latin_precision=round(latin_calib[-1]['precision'], 4) if latin_calib else 0.0,
        latin_recall=round(latin_calib[-1]['recall'], 4) if latin_calib else 0.0,
        latin_f1=best_latin_f1,
        latin_calibration=latin_calib,
        n_boundaries=total_boundaries,
        n_words=len(all_words),
        n_types=len(set(all_words)),
        mean_word_length=round(mean_wl, 2),
        dict_hit_rate=round(agg_dict_rate, 4),
        null_dict_hit_rate=round(null_mean, 4),
        selectivity=round(selectivity, 3),
        top_words=[{'word': w, 'count': c, 'in_dict': w in dictionary}
                   for w, c in top20],
        word_length_dist={str(k): v for k, v in Counter(word_lengths).items()},
        sample_segmentation=' '.join(all_words[:100])[:500],
        convergence_log=all_convergence[:20],
        per_section=per_section,
        g1_word_length=g1,
        g2_dict_hit=g2,
        g3_selectivity=g3,
        g4_top20_in_dict=g4,
        gates_passed=gates_passed,
        gate_passed=gates_passed >= 3,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    save_data = asdict(result)
    save_data['section_boundaries'] = section_boundary_map
    _save_json(rd, 'p65_bayesian.json', save_data)

    print(f"\n  Sample: {result.sample_segmentation[:200]}...")
    print(f"  Runtime: {result.runtime_seconds:.1f}s")
    return result
