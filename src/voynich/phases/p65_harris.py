"""
Phase 65, Step 2: Harris MI Boundary Detection
===============================================
Use mutual information between character contexts and next characters
to find word boundaries.  At word boundaries, MI drops because the
next character is less predictable.

Dependency chain:
    results/p65_decoded_stream.json  (Step 65.1)
        -> results/p65_harris.json
"""

import json
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelmin

from voynich.core._paths import results_dir as _results_dir
from voynich.core.reference import load_reference_corpus, build_expanded_word_set


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
class HarrisResult:
    phase: str = "65"
    step: str = "65.2"
    experiment: str = "harris_boundaries"
    # Latin calibration
    latin_best_context: int = 0
    latin_best_k: float = 0.0
    latin_best_method: str = ""
    latin_precision: float = 0.0
    latin_recall: float = 0.0
    latin_f1: float = 0.0
    latin_calibration_grid: List[Dict] = field(default_factory=list)
    # Voynich results
    n_boundaries: int = 0
    n_words: int = 0
    mean_word_length: float = 0.0
    std_word_length: float = 0.0
    dict_hit_rate: float = 0.0
    null_dict_hit_rate: float = 0.0
    selectivity: float = 0.0
    top_words: List[Dict] = field(default_factory=list)
    word_length_dist: Dict[int, int] = field(default_factory=dict)
    sample_segmentation: str = ""
    per_section: Dict[str, Dict] = field(default_factory=dict)
    # Gates
    g1_latin_f1: bool = False
    g2_dict_hit: bool = False
    g3_selectivity: bool = False
    g4_word_length: bool = False
    gates_passed: int = 0
    gate_passed: bool = False
    verdict: str = ""
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# MI computation
# ---------------------------------------------------------------------------

def compute_mi_profile(stream: str, context_length: int = 3) -> np.ndarray:
    """Compute pointwise MI between context and next char at each position."""
    n = len(stream)
    if n <= context_length:
        return np.zeros(n)

    context_counts: Counter = Counter()
    joint_counts: Counter = Counter()
    char_counts: Counter = Counter()

    for i in range(context_length, n):
        context = stream[i - context_length:i]
        next_char = stream[i]
        context_counts[context] += 1
        joint_counts[(context, next_char)] += 1
        char_counts[next_char] += 1

    total = sum(char_counts.values())
    if total == 0:
        return np.zeros(n)

    mi_profile = np.zeros(n)
    for i in range(context_length, n):
        context = stream[i - context_length:i]
        next_char = stream[i]

        p_next = char_counts[next_char] / total
        p_next_given_ctx = joint_counts[(context, next_char)] / context_counts[context]

        if p_next > 0 and p_next_given_ctx > 0:
            mi_profile[i] = np.log2(p_next_given_ctx / p_next)

    return mi_profile


def compute_multi_scale_mi(
    stream: str, context_lengths: List[int] = None,
) -> np.ndarray:
    """Average MI across multiple context lengths, weighted by length."""
    if context_lengths is None:
        context_lengths = [1, 2, 3, 4, 5]
    profiles = [compute_mi_profile(stream, cl) for cl in context_lengths]
    weights = np.array(context_lengths, dtype=float)
    weights /= weights.sum()
    combined = np.zeros(len(stream))
    for profile, weight in zip(profiles, weights):
        combined += weight * profile
    return combined


# ---------------------------------------------------------------------------
# Boundary detection
# ---------------------------------------------------------------------------

def find_boundaries_threshold(mi_profile: np.ndarray, k: float = 1.0) -> List[int]:
    """Boundary where MI < mean - k*std."""
    threshold = np.mean(mi_profile) - k * np.std(mi_profile)
    return [i for i in range(len(mi_profile)) if mi_profile[i] < threshold]


def find_boundaries_local_minima(
    mi_profile: np.ndarray, sigma: float = 3.0,
    order: int = 5, min_depth: float = 0.3,
) -> List[int]:
    """Boundary at local minima of smoothed MI profile."""
    smoothed = gaussian_filter1d(mi_profile, sigma=sigma)
    minima = argrelmin(smoothed, order=order)[0]

    boundaries = []
    for m in minima:
        left_start = max(0, m - 10)
        right_end = min(len(smoothed), m + 10)
        left_max = smoothed[left_start:m].max() if m > left_start else smoothed[0]
        right_max = smoothed[m:right_end].max() if right_end > m else smoothed[-1]
        depth = max(left_max, right_max) - smoothed[m]
        if depth >= min_depth:
            boundaries.append(int(m))
    return boundaries


def find_boundaries_derivative(
    mi_profile: np.ndarray, n_boundaries: int = None,
) -> List[int]:
    """Boundary at positions with steepest MI drop."""
    if n_boundaries is None:
        n_boundaries = len(mi_profile) // 6
    gradient = np.gradient(mi_profile)
    sorted_positions = np.argsort(gradient)
    return sorted(int(p) for p in sorted_positions[:n_boundaries])


# ---------------------------------------------------------------------------
# Segmentation and evaluation
# ---------------------------------------------------------------------------

def segment_at_boundaries(stream: str, boundaries: List[int]) -> List[str]:
    """Split stream at boundary positions into words."""
    if not boundaries:
        return [stream] if stream else []
    words = []
    prev = 0
    for b in sorted(set(boundaries)):
        if b > prev and b <= len(stream):
            word = stream[prev:b]
            if word:
                words.append(word)
            prev = b
    if prev < len(stream):
        words.append(stream[prev:])
    return words


def boundary_precision_recall(
    predicted: List[int], gold: List[int], tolerance: int = 2,
) -> Tuple[float, float, float]:
    """Precision/recall/F1 of predicted boundaries vs gold, with tolerance."""
    if not predicted or not gold:
        return 0.0, 0.0, 0.0
    gold_set = set(gold)
    # True positives: predicted boundaries within tolerance of a gold boundary
    tp = 0
    matched_gold = set()
    for p in predicted:
        for offset in range(-tolerance, tolerance + 1):
            if (p + offset) in gold_set and (p + offset) not in matched_gold:
                tp += 1
                matched_gold.add(p + offset)
                break
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def evaluate_segmentation(
    stream: str, boundaries: List[int], dictionary: set,
    n_null: int = 100,
) -> Dict[str, Any]:
    """Evaluate a segmentation: dict hit rate, selectivity, word stats."""
    words = segment_at_boundaries(stream, boundaries)
    if not words:
        return {'n_words': 0, 'dict_hit_rate': 0.0, 'selectivity': 0.0}

    lengths = [len(w) for w in words]
    dict_hits = sum(1 for w in words if w in dictionary)
    dict_rate = dict_hits / len(words)

    # Null: random boundaries at same density
    rng = np.random.default_rng(42)
    null_rates = []
    for _ in range(n_null):
        rand_boundaries = sorted(rng.choice(
            len(stream), size=len(boundaries), replace=False).tolist())
        rand_words = segment_at_boundaries(stream, rand_boundaries)
        null_rate = sum(1 for w in rand_words if w in dictionary) / len(rand_words) if rand_words else 0.0
        null_rates.append(null_rate)

    null_mean = float(np.mean(null_rates))
    selectivity = dict_rate / null_mean if null_mean > 0 else float('inf')

    word_counter = Counter(words)

    return {
        'n_words': len(words),
        'mean_length': float(np.mean(lengths)),
        'std_length': float(np.std(lengths)),
        'dict_hit_rate': round(dict_rate, 4),
        'dict_hits': dict_hits,
        'null_dict_rate': round(null_mean, 4),
        'selectivity': round(selectivity, 3),
        'word_length_dist': dict(Counter(lengths)),
        'top_words': [{'word': w, 'count': c, 'in_dict': w in dictionary}
                      for w, c in word_counter.most_common(50)],
    }


# ---------------------------------------------------------------------------
# Latin calibration
# ---------------------------------------------------------------------------

def calibrate_on_latin(
    latin_text: str, latin_boundaries: List[int], dictionary: set,
) -> Tuple[Dict, List[Dict]]:
    """Grid search Harris parameters on Latin calibration stream.

    Returns (best_config, all_configs).
    """
    best_config = None
    best_f1 = -1.0
    all_configs: List[Dict] = []

    for context_length in [1, 2, 3, 4, 5]:
        mi_profile = compute_mi_profile(latin_text, context_length)

        for k in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
            boundaries = find_boundaries_threshold(mi_profile, k=k)
            precision, recall, f1 = boundary_precision_recall(
                boundaries, latin_boundaries, tolerance=2)

            config = {
                'method': 'threshold',
                'context_length': context_length,
                'k': k,
                'n_boundaries': len(boundaries),
                'precision': round(precision, 4),
                'recall': round(recall, 4),
                'f1': round(f1, 4),
            }
            all_configs.append(config)

            if f1 > best_f1:
                best_f1 = f1
                best_config = config

    # Also try local minima with best context length
    best_cl = best_config['context_length'] if best_config else 3
    mi_profile = compute_mi_profile(latin_text, best_cl)
    for sigma in [2.0, 3.0, 5.0]:
        for min_depth in [0.2, 0.3, 0.5]:
            boundaries = find_boundaries_local_minima(
                mi_profile, sigma=sigma, min_depth=min_depth)
            precision, recall, f1 = boundary_precision_recall(
                boundaries, latin_boundaries, tolerance=2)
            config = {
                'method': 'local_minima',
                'context_length': best_cl,
                'sigma': sigma,
                'min_depth': min_depth,
                'n_boundaries': len(boundaries),
                'precision': round(precision, 4),
                'recall': round(recall, 4),
                'f1': round(f1, 4),
            }
            all_configs.append(config)
            if f1 > best_f1:
                best_f1 = f1
                best_config = config

    # Try derivative
    for divisor in [5, 6, 7, 8]:
        n_b = len(latin_text) // divisor
        boundaries = find_boundaries_derivative(mi_profile, n_boundaries=n_b)
        precision, recall, f1 = boundary_precision_recall(
            boundaries, latin_boundaries, tolerance=2)
        config = {
            'method': 'derivative',
            'context_length': best_cl,
            'divisor': divisor,
            'n_boundaries': len(boundaries),
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4),
        }
        all_configs.append(config)
        if f1 > best_f1:
            best_f1 = f1
            best_config = config

    return best_config, all_configs


# ---------------------------------------------------------------------------
# Dictionary building
# ---------------------------------------------------------------------------

def _build_10k_dict() -> set:
    """Build 10K evaluation dictionary from Latin reference."""
    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    base_words = set(w.lower() for w in ref_corpus.get_combined_tokens('latin')
                     if len(w) >= 2 and w.isalpha())
    # Take most frequent 10K
    freq = Counter(w.lower() for w in ref_corpus.get_combined_tokens('latin')
                   if len(w) >= 2 and w.isalpha())
    top_10k = set(w for w, _ in freq.most_common(10000))
    return top_10k


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_harris_segment():
    """Phase 65.2: Harris MI boundary detection."""
    t0 = time.time()
    rd = str(_results_dir())
    print("=" * 70)
    print("Phase 65, Step 2: Harris MI Boundary Detection")
    print("=" * 70)

    # Load stream data
    stream_data = _safe_load(os.path.join(rd, 'p65_decoded_stream.json'))
    if not stream_data:
        print("  ERROR: p65_decoded_stream.json not found. Run build-stream first.")
        return None

    full_text = stream_data['full_stream']['text']
    section_texts = stream_data.get('section_stream_texts', {})
    latin_streams = stream_data.get('latin_streams', [])

    print(f"  Full stream: {len(full_text)} chars")
    print(f"  Sections: {len(section_texts)}")

    # Build dictionary
    print("\n  Building 10K evaluation dictionary...")
    dictionary = _build_10k_dict()
    print(f"  Dictionary size: {len(dictionary)}")

    # Latin calibration
    print("\n  Latin calibration...")
    if latin_streams:
        latin_text = latin_streams[0]['text']
        latin_boundaries = latin_streams[0]['word_boundaries']
        print(f"  Latin stream: {len(latin_text)} chars, {len(latin_boundaries)} words")

        best_config, all_configs = calibrate_on_latin(
            latin_text, latin_boundaries, dictionary)
        print(f"  Best config: {best_config['method']}, "
              f"context={best_config.get('context_length')}, "
              f"F1={best_config['f1']:.3f} "
              f"(P={best_config['precision']:.3f}, R={best_config['recall']:.3f})")
    else:
        best_config = {'method': 'threshold', 'context_length': 3, 'k': 1.0, 'f1': 0.0}
        all_configs = []

    # Apply best method to Voynich section streams
    print("\n  Applying to Voynich sections...")
    all_voynich_boundaries: List[int] = []
    all_voynich_words: List[str] = []
    per_section_results: Dict[str, Dict] = {}

    # Also collect for full stream
    if best_config['method'] == 'threshold':
        mi_full = compute_mi_profile(full_text, best_config.get('context_length', 3))
        full_boundaries = find_boundaries_threshold(mi_full, k=best_config.get('k', 1.0))
    elif best_config['method'] == 'local_minima':
        mi_full = compute_mi_profile(full_text, best_config.get('context_length', 3))
        full_boundaries = find_boundaries_local_minima(
            mi_full, sigma=best_config.get('sigma', 3.0),
            min_depth=best_config.get('min_depth', 0.3))
    else:
        mi_full = compute_mi_profile(full_text, best_config.get('context_length', 3))
        divisor = best_config.get('divisor', 6)
        full_boundaries = find_boundaries_derivative(
            mi_full, n_boundaries=len(full_text) // divisor)

    eval_result = evaluate_segmentation(full_text, full_boundaries, dictionary)
    all_voynich_words = segment_at_boundaries(full_text, full_boundaries)

    print(f"  Full stream: {eval_result['n_words']} words, "
          f"dict hit {eval_result['dict_hit_rate']:.3f}, "
          f"selectivity {eval_result['selectivity']:.2f}x, "
          f"mean length {eval_result['mean_length']:.1f}")

    # Per-section analysis
    for section_key, section_text in section_texts.items():
        if len(section_text) < 50:
            continue
        if best_config['method'] == 'threshold':
            mi = compute_mi_profile(section_text, best_config.get('context_length', 3))
            boundaries = find_boundaries_threshold(mi, k=best_config.get('k', 1.0))
        elif best_config['method'] == 'local_minima':
            mi = compute_mi_profile(section_text, best_config.get('context_length', 3))
            boundaries = find_boundaries_local_minima(
                mi, sigma=best_config.get('sigma', 3.0),
                min_depth=best_config.get('min_depth', 0.3))
        else:
            mi = compute_mi_profile(section_text, best_config.get('context_length', 3))
            divisor = best_config.get('divisor', 6)
            boundaries = find_boundaries_derivative(
                mi, n_boundaries=len(section_text) // divisor)
        sec_eval = evaluate_segmentation(section_text, boundaries, dictionary, n_null=50)
        per_section_results[section_key] = {
            'n_chars': len(section_text),
            'n_boundaries': len(boundaries),
            **sec_eval,
        }
        print(f"    {section_key}: {sec_eval['n_words']} words, "
              f"dict {sec_eval['dict_hit_rate']:.3f}, sel {sec_eval['selectivity']:.2f}x")

    # Gates
    latin_f1 = best_config.get('f1', 0.0)
    g1 = latin_f1 > 0.3
    g2 = eval_result['dict_hit_rate'] > 0.10
    g3 = eval_result['selectivity'] > 1.5
    g4 = 4.0 <= eval_result.get('mean_length', 0) <= 8.0
    gates_passed = sum([g1, g2, g3, g4])

    if gates_passed >= 3:
        verdict = "HARRIS_PASS"
    elif gates_passed >= 2:
        verdict = "HARRIS_PARTIAL"
    else:
        verdict = "HARRIS_FAIL"

    sample_words = all_voynich_words[:100]
    sample_seg = ' '.join(sample_words)

    print(f"\n  Gates: H1(Latin F1>{0.3})={'PASS' if g1 else 'FAIL'} ({latin_f1:.3f})")
    print(f"         H2(dict>{0.10})={'PASS' if g2 else 'FAIL'} ({eval_result['dict_hit_rate']:.3f})")
    print(f"         H3(sel>{1.5})={'PASS' if g3 else 'FAIL'} ({eval_result['selectivity']:.2f}x)")
    print(f"         H4(length 4-8)={'PASS' if g4 else 'FAIL'} ({eval_result.get('mean_length', 0):.1f})")
    print(f"  Verdict: {verdict} ({gates_passed}/4)")

    result = HarrisResult(
        latin_best_context=best_config.get('context_length', 0),
        latin_best_k=best_config.get('k', 0.0),
        latin_best_method=best_config.get('method', ''),
        latin_precision=best_config.get('precision', 0.0),
        latin_recall=best_config.get('recall', 0.0),
        latin_f1=latin_f1,
        latin_calibration_grid=all_configs[:20],  # top 20 only
        n_boundaries=len(full_boundaries),
        n_words=eval_result['n_words'],
        mean_word_length=round(eval_result.get('mean_length', 0), 2),
        std_word_length=round(eval_result.get('std_length', 0), 2),
        dict_hit_rate=eval_result['dict_hit_rate'],
        null_dict_hit_rate=eval_result['null_dict_rate'],
        selectivity=eval_result['selectivity'],
        top_words=eval_result.get('top_words', [])[:30],
        word_length_dist={str(k): v for k, v in eval_result.get('word_length_dist', {}).items()},
        sample_segmentation=sample_seg[:500],
        per_section=per_section_results,
        g1_latin_f1=g1,
        g2_dict_hit=g2,
        g3_selectivity=g3,
        g4_word_length=g4,
        gates_passed=gates_passed,
        gate_passed=gates_passed >= 3,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    # Save boundaries for integration step
    save_data = asdict(result)
    save_data['boundaries'] = full_boundaries
    save_data['section_boundaries'] = {
        k: find_boundaries_threshold(
            compute_mi_profile(section_texts[k], best_config.get('context_length', 3)),
            k=best_config.get('k', 1.0))
        if best_config['method'] == 'threshold' else []
        for k in section_texts
    }
    _save_json(rd, 'p65_harris.json', save_data)

    print(f"\n  Sample: {sample_seg[:200]}...")
    print(f"  Runtime: {result.runtime_seconds:.1f}s")
    return result
