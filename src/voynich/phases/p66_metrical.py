"""
Phase 66, Track 11: Metrical Analysis
======================================
Test for verse structure via stress-pattern periodicity and line-length
regularity.  Gates are designed to CONFIRM prose (NOT verse).

Dependency chain:
    results/combined_refine.json      (Phase 15)
        -> results/p66_metrical.json
"""

from __future__ import annotations

import json
import math
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import build_eva_to_triple_lookup, load_corpus, tokenize_eva_chars
from voynich.core.reference import build_expanded_word_set, load_reference_corpus
from voynich.phases.corrected_coda import build_coda_table_v2, decode_corpus_cvc_v2, decode_token_cvc_v2
from voynich.phases.suffix_calibration import SIGNAL_WORDS_51


# ---------------------------------------------------------------------------
# JSON helpers (standard pattern)
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
class MetricalResult:
    phase: str = "66"
    step: str = "66.11"
    experiment: str = "metrical_analysis"
    n_lines: int = 0
    n_pages: int = 0
    mean_line_length: float = 0.0
    std_line_length: float = 0.0
    cv_line_length: float = 0.0
    median_line_length: float = 0.0
    max_autocorr: float = 0.0
    max_autocorr_lag: int = 0
    autocorr_profile: List[float] = field(default_factory=list)
    line_length_histogram: Dict[str, int] = field(default_factory=dict)
    # Gates (prose confirmation)
    m1_autocorr: bool = False      # max autocorr < 0.10 (NOT verse)
    m2_line_var: bool = False      # std > 3.0 (variable = prose)
    gates_passed: int = 0
    gate_passed: bool = False
    verdict: str = ""
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Syllable estimation
# ---------------------------------------------------------------------------

def _estimate_syllable_count(decoded_word: str) -> int:
    """Estimate number of syllables in a decoded Latin-like word.

    Heuristic: count vowel groups (a, e, i, o, u).  Each contiguous
    run of vowels counts as one syllable.  Minimum 1 syllable for any
    non-empty word.
    """
    if not decoded_word or decoded_word == '?':
        return 0
    vowels = set('aeiouAEIOU')
    count = 0
    in_vowel = False
    for ch in decoded_word:
        if ch in vowels:
            if not in_vowel:
                count += 1
                in_vowel = True
        else:
            in_vowel = False
    return max(count, 1) if decoded_word else 0


def _build_stress_vector(decoded_words: List[str]) -> List[int]:
    """Build a binary stress vector from decoded words.

    Applies Latin penultimate stress rule:
    - 1-syllable words: stress on that syllable.
    - 2+ syllable words: stress on the penultimate syllable.

    Returns a list of 0/1 values, one per syllable across the stream.
    """
    stress_vec: List[int] = []
    for word in decoded_words:
        n_syl = _estimate_syllable_count(word)
        if n_syl <= 0:
            continue
        if n_syl == 1:
            stress_vec.append(1)
        else:
            # penultimate stress
            for i in range(n_syl):
                if i == n_syl - 2:
                    stress_vec.append(1)
                else:
                    stress_vec.append(0)
    return stress_vec


def _autocorrelation(signal: List[int], max_lag: int = 20) -> List[float]:
    """Compute normalised autocorrelation at lags 1..max_lag.

    Returns a list of length max_lag with autocorrelation values.
    """
    if len(signal) < max_lag + 1:
        return [0.0] * max_lag

    arr = np.array(signal, dtype=np.float64)
    mean = arr.mean()
    arr_centered = arr - mean
    var = np.dot(arr_centered, arr_centered)
    if var == 0:
        return [0.0] * max_lag

    result: List[float] = []
    for lag in range(1, max_lag + 1):
        if lag >= len(arr_centered):
            result.append(0.0)
            continue
        c = np.dot(arr_centered[:len(arr_centered) - lag], arr_centered[lag:])
        result.append(float(c / var))
    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_metrical() -> MetricalResult:
    """Phase 66, Track 11: Metrical Analysis."""
    t0 = time.time()
    rd = _results_dir()
    result = MetricalResult()

    print("=" * 70)
    print("Phase 66, Track 11: Metrical Analysis")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load corpus and dependencies
    # ------------------------------------------------------------------
    print("\n[1] Loading corpus and CVC decode resources ...")
    corpus = load_corpus(verbose=False)

    refine_path = os.path.join(rd, "combined_refine.json")
    refine = _safe_load(refine_path)
    if not refine:
        print("  WARNING: combined_refine.json not found; using empty assignment")
    assignment = refine.get("best_assignment", {})

    eva_to_triple = build_eva_to_triple_lookup()
    coda_table = build_coda_table_v2()

    # ------------------------------------------------------------------
    # 2. Extract lines from all pages
    # ------------------------------------------------------------------
    print("[2] Extracting paragraph lines ...")
    all_line_lengths: List[int] = []
    all_decoded_words: List[str] = []
    pages_with_lines = 0

    for folio, page in corpus.pages.items():
        page_has_lines = False
        for locus in page.loci:
            # Only paragraph lines
            if not locus.locus_type.startswith('P'):
                continue
            page_has_lines = True

            # Tokenize the line and decode each token
            line_tokens = locus.clean_text.split()
            if not line_tokens:
                continue

            decoded_line_words: List[str] = []
            line_syllable_count = 0
            for tok in line_tokens:
                dr = decode_token_cvc_v2(tok, assignment, eva_to_triple, coda_table)
                decoded = dr.decoded_cvc if dr.decoded_cvc else dr.decoded_cv
                if decoded and decoded != '?':
                    decoded_line_words.append(decoded)
                    line_syllable_count += _estimate_syllable_count(decoded)

            all_decoded_words.extend(decoded_line_words)
            all_line_lengths.append(line_syllable_count)

        if page_has_lines:
            pages_with_lines += 1

    result.n_lines = len(all_line_lengths)
    result.n_pages = pages_with_lines
    print(f"  {result.n_lines} lines from {result.n_pages} pages")

    # ------------------------------------------------------------------
    # 3. Line-length statistics
    # ------------------------------------------------------------------
    print("[3] Computing line-length statistics ...")
    if all_line_lengths:
        arr = np.array(all_line_lengths, dtype=np.float64)
        result.mean_line_length = float(np.mean(arr))
        result.std_line_length = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        result.cv_line_length = (
            result.std_line_length / result.mean_line_length
            if result.mean_line_length > 0 else 0.0
        )
        result.median_line_length = float(np.median(arr))

        # Histogram
        counts = Counter(all_line_lengths)
        result.line_length_histogram = {
            str(k): v for k, v in sorted(counts.items())
        }
    else:
        print("  WARNING: No lines found.")

    print(f"  Mean={result.mean_line_length:.1f}  Std={result.std_line_length:.2f}  "
          f"CV={result.cv_line_length:.3f}  Median={result.median_line_length:.1f}")

    # ------------------------------------------------------------------
    # 4. Stress pattern and autocorrelation
    # ------------------------------------------------------------------
    print("[4] Building stress vector and computing autocorrelation ...")
    stress_vec = _build_stress_vector(all_decoded_words)
    print(f"  Stress vector length: {len(stress_vec)} syllables")

    max_lag = 20
    autocorr = _autocorrelation(stress_vec, max_lag=max_lag)
    result.autocorr_profile = [round(v, 6) for v in autocorr]

    if autocorr:
        abs_autocorr = [abs(v) for v in autocorr]
        result.max_autocorr = max(abs_autocorr)
        result.max_autocorr_lag = abs_autocorr.index(result.max_autocorr) + 1  # lags 1-based
    else:
        result.max_autocorr = 0.0
        result.max_autocorr_lag = 0

    print(f"  Max |autocorrelation| = {result.max_autocorr:.4f} at lag {result.max_autocorr_lag}")
    print(f"  Autocorrelation profile (lags 1-{max_lag}):")
    for i, v in enumerate(result.autocorr_profile[:10], 1):
        print(f"    lag {i:2d}: {v:+.4f}")
    if len(result.autocorr_profile) > 10:
        print(f"    ... (lags 11-{max_lag} omitted for brevity)")

    # ------------------------------------------------------------------
    # 5. Evaluate gates
    # ------------------------------------------------------------------
    print("\n[5] Evaluating gates (prose confirmation) ...")

    # M1: max autocorrelation < 0.10 (NOT verse)
    result.m1_autocorr = result.max_autocorr < 0.10
    print(f"  M1 autocorr < 0.10: {result.m1_autocorr}  "
          f"(max_autocorr={result.max_autocorr:.4f})")

    # M2: line length std > 3.0 (variable line lengths = prose)
    result.m2_line_var = result.std_line_length > 3.0
    print(f"  M2 line_std > 3.0:  {result.m2_line_var}  "
          f"(std={result.std_line_length:.2f})")

    result.gates_passed = sum([result.m1_autocorr, result.m2_line_var])
    result.gate_passed = result.gates_passed >= 2

    if result.gate_passed:
        result.verdict = "PROSE_CONFIRMED"
    elif result.gates_passed == 1:
        result.verdict = "PROSE_LIKELY"
    else:
        result.verdict = "VERSE_POSSIBLE"

    # ------------------------------------------------------------------
    # 6. Summary and save
    # ------------------------------------------------------------------
    result.runtime_seconds = round(time.time() - t0, 2)

    print("\n" + "=" * 70)
    print(f"VERDICT: {result.verdict}")
    print(f"  Gates passed: {result.gates_passed}/2")
    print(f"  Lines: {result.n_lines}  Pages: {result.n_pages}")
    print(f"  Line length: mean={result.mean_line_length:.1f} "
          f"std={result.std_line_length:.2f} CV={result.cv_line_length:.3f}")
    print(f"  Max autocorrelation: {result.max_autocorr:.4f} at lag {result.max_autocorr_lag}")
    print(f"  Runtime: {result.runtime_seconds:.1f}s")
    print("=" * 70)

    saved = _save_json(rd, "p66_metrical.json", asdict(result))
    print(f"  Saved -> {saved}")

    return result
