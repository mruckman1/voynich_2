"""
Phase 18.2 – Stride-Entropy (Decimation) Analysis
===================================================

Tests whether the Voynich EVA character stream conceals a compressed
plaintext at a specific stride (expansion ratio).  If the manuscript is
a verbose cipher (H2), extracting every K-th character should collapse
the entropy floor to match natural-language Latin.

Dependency chain:
    (none — reads corpus directly)
        -> stride_entropy.json
"""

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import load_corpus, tokenize_eva_chars
from voynich.core.reference import load_reference_corpus
from voynich.core.stats import conditional_entropy, first_order_entropy


# ---------------------------------------------------------------------------
# JSON serialiser
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


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class StrideEntropyResult:
    eva_stream_length: int
    n_unique_eva_chars: int
    baseline_entropy_curve: Dict[str, float]   # order -> H  (stride=1)
    baseline_h6: float
    latin_h6_reference: float
    stride_results: List[Dict[str, Any]]       # per stride: {stride, h_curve, h6, collapse}
    best_stride: int
    best_stride_h6: float
    min_h6_across_strides: float
    floor_collapse_found: bool
    floor_collapse_strides: List[int]
    latin_any_collapse: bool                   # control — should be False
    hypothesis_support: Dict[str, float]
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


def _build_eva_char_stream(tokens: List[str]) -> Tuple[str, Dict[str, str], List[str]]:
    """Convert a token list into a single Unicode string where each
    character = one EVA unit (ligatures like 'sh' become a single
    private-use codepoint).

    Returns (encoded_string, eva_to_unicode_map, ordered_unique_chars).
    """
    # Flatten tokens into EVA char sequence
    eva_chars: List[str] = []
    for tok in tokens:
        eva_chars.extend(tokenize_eva_chars(tok))

    # Build mapping: unique EVA char -> U+E000 + index
    unique_sorted = sorted(set(eva_chars))
    eva_to_uni = {ch: chr(0xE000 + i) for i, ch in enumerate(unique_sorted)}

    encoded = ''.join(eva_to_uni[ch] for ch in eva_chars)
    return encoded, eva_to_uni, unique_sorted


def _entropy_curve_custom(text: str, max_order: int = 6) -> Dict[int, float]:
    """Compute character-level entropy curve {0: H1, 1: H2, …, max_order: H(max_order+1)}.

    Mirrors stats.entropy_curve() but operates on the custom-encoded
    string (private-use Unicode codepoints).
    """
    if len(text) < 2:
        return {o: 0.0 for o in range(max_order + 1)}

    curve: Dict[int, float] = {}
    curve[0] = first_order_entropy(text)
    for order in range(1, max_order + 1):
        try:
            curve[order] = conditional_entropy(text, order=order)
        except Exception:
            curve[order] = curve.get(order - 1, 0.0)
    return curve


def _compute_stride_profiles(
    char_stream: str,
    max_stride: int = 8,
    max_order: int = 6,
    latin_h6: float = 1.2,
    collapse_threshold: float = 0.3,
) -> List[Dict[str, Any]]:
    """For strides 1..max_stride, compute the entropy curve of the
    decimated sub-string and check for floor collapse."""
    results: List[Dict[str, Any]] = []
    for stride in range(1, max_stride + 1):
        decimated = char_stream[::stride]
        if len(decimated) < 50:
            # Too short to compute meaningful entropy
            results.append({
                'stride': stride,
                'length': len(decimated),
                'entropy_curve': {},
                'h6': None,
                'floor_collapse': False,
            })
            continue

        curve = _entropy_curve_custom(decimated, max_order)
        h6 = curve.get(max_order, curve.get(max_order - 1, 0.0))
        collapse = abs(h6 - latin_h6) < collapse_threshold
        results.append({
            'stride': stride,
            'length': len(decimated),
            'entropy_curve': {str(k): round(v, 4) for k, v in curve.items()},
            'h6': round(h6, 4),
            'floor_collapse': collapse,
        })
    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_stride_entropy() -> None:
    """Phase 18.2: stride-entropy decimation test."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE 18.2: Stride-Entropy (Decimation) Analysis")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Build EVA character stream ──────────────────────────────────
    print("\n  1. Building EVA character stream …")
    corpus = load_corpus(verbose=False)
    tokens_a = corpus.get_tokens(language='A', paragraph_only=True)
    char_stream, eva_map, unique_chars = _build_eva_char_stream(tokens_a)
    print(f"     {len(char_stream):,} EVA chars  |  {len(unique_chars)} unique EVA units")

    # ── 2. Baseline entropy curve (stride = 1) ────────────────────────
    print("\n  2. Baseline entropy curve (stride = 1) …")
    baseline_curve = _entropy_curve_custom(char_stream, max_order=6)
    baseline_h6 = baseline_curve.get(6, baseline_curve.get(5, 0.0))
    print(f"     H1={baseline_curve.get(0,0):.3f}  H3={baseline_curve.get(2,0):.3f}  "
          f"H6={baseline_h6:.3f}")

    # ── 3. Latin H6 reference ─────────────────────────────────────────
    print("\n  3. Computing Latin H6 reference …")
    latin_h6 = 1.2  # default fallback
    try:
        ref = load_reference_corpus(languages=['latin'], verbose=False)
        latin_text = ref.get_combined_text('latin')
        if latin_text and len(latin_text) > 200:
            lat_curve = _entropy_curve_custom(latin_text, max_order=6)
            latin_h6 = lat_curve.get(6, lat_curve.get(5, 1.2))
            print(f"     Latin H6 = {latin_h6:.3f}")
        else:
            print(f"     Latin corpus too short — using default H6 = {latin_h6}")
    except Exception:
        print(f"     WARNING: Latin corpus unavailable — using default H6 = {latin_h6}")

    # ── 4. Stride profiles ────────────────────────────────────────────
    print("\n  4. Computing stride profiles (K = 1..8) …")
    stride_profiles = _compute_stride_profiles(
        char_stream, max_stride=8, max_order=6,
        latin_h6=latin_h6, collapse_threshold=0.3,
    )

    for sp in stride_profiles:
        tag = " *** COLLAPSE ***" if sp['floor_collapse'] else ""
        h6_str = f"{sp['h6']:.3f}" if sp['h6'] is not None else "N/A"
        print(f"     stride={sp['stride']}  len={sp['length']:,}  H6={h6_str}{tag}")

    collapse_strides = [sp['stride'] for sp in stride_profiles if sp['floor_collapse']]
    collapse_found = len(collapse_strides) > 0

    h6_values = [sp['h6'] for sp in stride_profiles if sp['h6'] is not None]
    min_h6 = min(h6_values) if h6_values else baseline_h6
    best_stride_idx = int(np.argmin(h6_values)) if h6_values else 0
    best_stride = stride_profiles[best_stride_idx]['stride'] if stride_profiles else 1
    best_stride_h6 = h6_values[best_stride_idx] if h6_values else baseline_h6

    # ── 5. Latin control ──────────────────────────────────────────────
    print("\n  5. Latin control (no stride should collapse) …")
    latin_any_collapse = False
    try:
        ref = load_reference_corpus(languages=['latin'], verbose=False)
        latin_text = ref.get_combined_text('latin')
        if latin_text and len(latin_text) > 200:
            # Encode Latin chars as themselves (they are single characters already)
            for ks in range(2, 9):
                dec = latin_text[::ks]
                if len(dec) < 50:
                    continue
                lat_curve_dec = _entropy_curve_custom(dec, max_order=6)
                lat_h6_dec = lat_curve_dec.get(6, lat_curve_dec.get(5, 0.0))
                if abs(lat_h6_dec - latin_h6) < 0.3:
                    latin_any_collapse = True
                    break
    except Exception:
        pass
    print(f"     Latin collapse at any stride: {latin_any_collapse}")

    # ── 6. Hypothesis scoring ─────────────────────────────────────────
    print("\n  6. Scoring hypotheses …")
    h2_score = 1.0 if collapse_found else _sigmoid(-(baseline_h6 - min_h6) / 0.5)
    h1_score = _sigmoid(-(baseline_h6 - 1.0) / 0.5)
    h3_score = 0.3  # stride analysis is largely neutral for H3

    total = h1_score + h2_score + h3_score
    if total > 0:
        h1_score, h2_score, h3_score = h1_score / total, h2_score / total, h3_score / total

    hypothesis_support = {'H1': round(h1_score, 4), 'H2': round(h2_score, 4), 'H3': round(h3_score, 4)}
    print(f"     H1={h1_score:.3f}  H2={h2_score:.3f}  H3={h3_score:.3f}")

    # ── Verdict ───────────────────────────────────────────────────────
    if collapse_found:
        verdict = (f"FLOOR COLLAPSE at stride(s) {collapse_strides}: decimated H6 matches "
                   f"Latin H6 ({latin_h6:.3f}). Strong evidence for verbose cipher (H2) "
                   f"with expansion ratio ≈ {collapse_strides[0]}:1.")
    elif min_h6 < baseline_h6 - 0.5:
        verdict = (f"PARTIAL REDUCTION: best stride K={best_stride} reduces H6 from "
                   f"{baseline_h6:.3f} to {best_stride_h6:.3f}, but does not reach Latin "
                   f"floor ({latin_h6:.3f}). Weak H2 signal.")
    else:
        verdict = (f"NO COLLAPSE: no stride reduces H6 significantly (baseline={baseline_h6:.3f}, "
                   f"min={min_h6:.3f}, Latin={latin_h6:.3f}). H2 (verbose cipher) not supported "
                   "by decimation analysis.")
    print(f"\n  Verdict: {verdict}")

    # ── Save ──────────────────────────────────────────────────────────
    result = StrideEntropyResult(
        eva_stream_length=len(char_stream),
        n_unique_eva_chars=len(unique_chars),
        baseline_entropy_curve={str(k): round(v, 4) for k, v in baseline_curve.items()},
        baseline_h6=round(baseline_h6, 4),
        latin_h6_reference=round(latin_h6, 4),
        stride_results=stride_profiles,
        best_stride=best_stride,
        best_stride_h6=round(best_stride_h6, 4),
        min_h6_across_strides=round(min_h6, 4),
        floor_collapse_found=collapse_found,
        floor_collapse_strides=collapse_strides,
        latin_any_collapse=latin_any_collapse,
        hypothesis_support=hypothesis_support,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'stride_entropy.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)

    print(f"\n  -> {out_path}")
