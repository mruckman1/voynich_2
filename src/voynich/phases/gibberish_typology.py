"""
Step 27.1 -- Gibberish and Self-Citation Typology Classification
================================================================
Run the Gaskell-Bowern gibberish corpus and a Timm-Schinner
self-citation simulation through the Phase 9.5 text typology
classifier.  If the classifier labels gibberish as
``encoded_natural`` (the same label it gives Voynich), the
classifier's verdict is undermined.

Dependency chain:
    data/gibberish_transcriptions/*.txt
    results/text_typology.json
        -> gibberish_typology.json
"""

from __future__ import annotations

import glob
import json
import math
import os
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from voynich.core._paths import data_dir, results_dir as _results_dir
from voynich.core.corpus import load_corpus
from voynich.core.reference import load_reference_corpus
from voynich.core.stats import entropy_curve, first_order_entropy
from voynich.phases.text_typology import (
    _classify_text,
    _compute_text_stats,
    _entropy_decay_rate,
)


# ---------------------------------------------------------------------------
# JSON serialiser
# ---------------------------------------------------------------------------

def _convert(obj: Any) -> Any:
    """Recursively convert dataclasses/numpy/NaN to JSON-safe types."""
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
    if isinstance(obj, float) and (obj != obj):  # NaN
        return None
    if isinstance(obj, (bool, int, float, str, type(None))):
        return obj
    return str(obj)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SampleClassification:
    sample_id: str
    source: str          # 'gibberish' | 'timm_schinner'
    n_tokens: int
    most_likely_type: str
    confidence: float
    h2_h1_ratio: float
    ttr: float
    zipf_r_squared: float
    entropy_floor: float
    entropy_curve: Dict[str, float]


@dataclass
class GibberishTypologyResult:
    timestamp: str
    # Gibberish corpus
    n_gibberish_samples: int
    gibberish_results: List[Dict]
    gibberish_type_distribution: Dict[str, int]
    gibberish_encoded_natural_count: int
    gibberish_h2_h1_mean: float
    gibberish_h2_h1_std: float
    gibberish_entropy_floor_mean: float
    # Timm-Schinner corpus
    n_timm_samples: int
    timm_results: List[Dict]
    timm_type_distribution: Dict[str, int]
    timm_encoded_natural_count: int
    timm_parameter_sensitivity: List[Dict]
    # Comparison table
    comparison_table: List[Dict]
    # Entropy curve comparison
    entropy_curve_comparison: Dict[str, Any]
    # Voynich reference
    voynich_type: str
    voynich_h2_h1: float
    voynich_entropy_floor: float
    # Methodological note
    methodological_note: str
    # Gate
    discriminant_power: float
    gate_passed: bool
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Gibberish corpus loader
# ---------------------------------------------------------------------------

def _load_gibberish_samples(
    gibberish_dir: str,
) -> List[Tuple[str, str, List[str]]]:
    """
    Load all Gaskell-Bowern gibberish transcription files.

    Returns list of (filename, text, tokens).
    Files have two sections separated by a line of underscores.
    Both sections are concatenated.
    """
    pattern = os.path.join(gibberish_dir, 'Gibberish*.txt')
    files = sorted(glob.glob(pattern))

    samples: List[Tuple[str, str, List[str]]] = []
    for fpath in files:
        fname = os.path.basename(fpath)
        with open(fpath, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Concatenate all lines, skipping separator lines (all underscores)
        text_parts: List[str] = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            # Skip separator lines
            if all(ch == '_' for ch in stripped):
                continue
            text_parts.append(stripped)

        text = ' '.join(text_parts)
        tokens = text.split()
        if tokens:
            samples.append((fname, text, tokens))

    return samples


# ---------------------------------------------------------------------------
# Timm-Schinner self-citation generator
# ---------------------------------------------------------------------------

def _build_voynich_char_freq(voynich_text: str) -> Dict[str, float]:
    """Build character frequency distribution from Voynich text."""
    counts: Counter = Counter()
    for ch in voynich_text:
        if ch != ' ':
            counts[ch] += 1
    total = sum(counts.values())
    return {ch: n / total for ch, n in counts.items()}


def _build_voynich_word_lengths(voynich_tokens: List[str]) -> List[int]:
    """Get word length distribution from Voynich tokens."""
    return [len(t) for t in voynich_tokens if t]


def _generate_fresh_word(
    char_freq: Dict[str, float],
    word_lengths: List[int],
    rng: random.Random,
) -> str:
    """Generate a fresh word from character frequency and word length distributions."""
    chars = list(char_freq.keys())
    weights = list(char_freq.values())
    length = rng.choice(word_lengths)
    return ''.join(rng.choices(chars, weights=weights, k=length))


def _mutate_word(word: str, char_freq: Dict[str, float], p_mutate: float,
                 rng: random.Random) -> str:
    """Apply small random mutation to a word."""
    if rng.random() >= p_mutate or not word:
        return word

    chars = list(word)
    mutation_type = rng.choice(['substitute', 'insert', 'delete'])

    freq_chars = list(char_freq.keys())
    freq_weights = list(char_freq.values())

    if mutation_type == 'substitute' and chars:
        pos = rng.randrange(len(chars))
        chars[pos] = rng.choices(freq_chars, weights=freq_weights, k=1)[0]
    elif mutation_type == 'insert':
        pos = rng.randrange(len(chars) + 1)
        chars.insert(pos, rng.choices(freq_chars, weights=freq_weights, k=1)[0])
    elif mutation_type == 'delete' and len(chars) > 1:
        pos = rng.randrange(len(chars))
        del chars[pos]

    return ''.join(chars)


def _generate_timm_schinner(
    char_freq: Dict[str, float],
    word_lengths: List[int],
    p_copy: float = 0.7,
    p_mutate: float = 0.10,
    buffer_size: int = 100,
    n_tokens: int = 10000,
    seed: int = 42,
) -> List[str]:
    """
    Generate self-citation text following Timm & Schinner (2020).

    Maintains a memory buffer of previously generated words.
    Each new word is either:
      - Copied from a random buffer position (with optional mutation)
      - Generated fresh from Voynich char/length distributions
    """
    rng = random.Random(seed)
    tokens: List[str] = []

    for i in range(n_tokens):
        if tokens and rng.random() < p_copy:
            # Copy from buffer (look back up to buffer_size positions)
            lookback = min(len(tokens), buffer_size)
            src_idx = len(tokens) - 1 - rng.randrange(lookback)
            word = _mutate_word(tokens[src_idx], char_freq, p_mutate, rng)
        else:
            word = _generate_fresh_word(char_freq, word_lengths, rng)
        tokens.append(word)

    return tokens


# ---------------------------------------------------------------------------
# Classifier wrapper
# ---------------------------------------------------------------------------

def _run_typology_classifier(
    text: str,
    tokens: List[str],
    max_order: int = 6,
) -> Dict[str, Any]:
    """
    Run the Phase 9.5 classifier on arbitrary text.

    Returns dict with classification, features, and entropy curve.
    """
    if len(tokens) < 20:
        return {
            'most_likely_type': 'insufficient_data',
            'confidence': 0.0,
            'h2_h1_ratio': 0.0,
            'ttr': 0.0,
            'zipf_r_squared': 0.0,
            'entropy_floor': 0.0,
            'entropy_curve': {},
        }

    features = _compute_text_stats(text, tokens)
    classification = _classify_text(features)
    curve = entropy_curve(text, max_order=max_order)
    floor = curve.get(max_order, 0.0)

    return {
        'most_likely_type': classification.most_likely_type,
        'confidence': classification.confidence,
        'h2_h1_ratio': features['h2_h1_ratio'],
        'ttr': features['ttr'],
        'zipf_r_squared': features['zipf_r_squared'],
        'entropy_floor': floor,
        'entropy_curve': {str(k): round(v, 4) for k, v in curve.items()},
    }


# ---------------------------------------------------------------------------
# Entropy curve comparison
# ---------------------------------------------------------------------------

def _entropy_curve_comparison(
    voynich_curve: Dict[int, float],
    latin_curve: Dict[int, float],
    gibberish_curves: List[Dict[int, float]],
    timm_curves: List[Dict[int, float]],
) -> Dict[str, Any]:
    """Compare entropy curves across text types."""
    max_order = max(voynich_curve.keys()) if voynich_curve else 6

    # Convert curves to arrays for statistics
    def curve_to_array(c: Dict[int, float]) -> np.ndarray:
        return np.array([c.get(k, 0.0) for k in range(max_order + 1)])

    v_arr = curve_to_array(voynich_curve)
    l_arr = curve_to_array(latin_curve) if latin_curve else np.zeros(max_order + 1)

    gib_arrs = [curve_to_array(c) for c in gibberish_curves if c]
    timm_arrs = [curve_to_array(c) for c in timm_curves if c]

    result: Dict[str, Any] = {
        'voynich_floor': float(v_arr[-1]) if len(v_arr) > 0 else 0.0,
        'latin_floor': float(l_arr[-1]) if len(l_arr) > 0 else 0.0,
    }

    if gib_arrs:
        gib_mat = np.array(gib_arrs)
        gib_mean = gib_mat.mean(axis=0)
        gib_std = gib_mat.std(axis=0)
        result['gibberish_mean_curve'] = [round(float(v), 4) for v in gib_mean]
        result['gibberish_std_curve'] = [round(float(v), 4) for v in gib_std]
        result['gibberish_floor_mean'] = round(float(gib_mat[:, -1].mean()), 4)
        result['gibberish_floor_std'] = round(float(gib_mat[:, -1].std()), 4)

    if timm_arrs:
        timm_mat = np.array(timm_arrs)
        timm_mean = timm_mat.mean(axis=0)
        result['timm_mean_curve'] = [round(float(v), 4) for v in timm_mean]
        result['timm_floor_mean'] = round(float(timm_mat[:, -1].mean()), 4)

    # Does gibberish also have elevated entropy floor?
    if gib_arrs:
        elevated = sum(1 for a in gib_arrs if a[-1] > 0.6)
        result['gibberish_elevated_floor_count'] = int(elevated)
        result['gibberish_elevated_floor_fraction'] = round(
            elevated / len(gib_arrs), 3
        )

    return result


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_gibberish_typology() -> None:
    """Step 27.1: Gibberish and self-citation typology classification."""
    t0 = time.time()
    rd = str(_results_dir())

    print("=" * 60)
    print("Step 27.1: Gibberish & Self-Citation Typology Control")
    print("=" * 60)

    # ── 1. Load Voynich reference ─────────────────────────────────────
    print("\n  1. Loading Voynich reference data ...")

    corpus = load_corpus(verbose=False)
    voynich_tokens = corpus.get_tokens(language='A')
    voynich_text = corpus.get_text(language='A')

    # Load existing Phase 9.5 result for comparison
    typology_path = os.path.join(rd, 'text_typology.json')
    if os.path.exists(typology_path):
        with open(typology_path) as f:
            phase95 = json.load(f)
        voynich_type = phase95.get('classification', {}).get('most_likely_type', 'unknown')
        voynich_h2_h1 = phase95.get('classification', {}).get('voynich_features', {}).get('h2_h1_ratio', 0.622)
        voynich_floor = phase95.get('entropy_curves', {}).get('voynich_floor', 0.978)
    else:
        # Compute live
        v_features = _compute_text_stats(voynich_text, voynich_tokens)
        v_class = _classify_text(v_features)
        voynich_type = v_class.most_likely_type
        voynich_h2_h1 = v_features['h2_h1_ratio']
        v_curve = entropy_curve(voynich_text, max_order=6)
        voynich_floor = v_curve.get(6, 0.0)

    voynich_curve = entropy_curve(voynich_text, max_order=6)

    print(f"    Voynich: type={voynich_type}, H2/H1={voynich_h2_h1:.3f}, "
          f"floor={voynich_floor:.3f}")

    # Latin reference for entropy comparison
    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    latin_text = ref_corpus.get_combined_text('latin')
    latin_text = latin_text[:len(voynich_text)]  # subsample to Voynich size
    latin_curve = entropy_curve(latin_text, max_order=6) if latin_text else {}

    # ── 2. Load and classify gibberish corpus ─────────────────────────
    print("\n  2. Loading and classifying gibberish corpus ...")

    gib_dir = str(data_dir('gibberish_transcriptions'))
    gib_samples = _load_gibberish_samples(gib_dir)
    n_gib = len(gib_samples)
    print(f"    Found {n_gib} gibberish samples")

    gib_results: List[SampleClassification] = []
    gib_curves: List[Dict[int, float]] = []

    for i, (fname, text, tokens) in enumerate(gib_samples):
        result = _run_typology_classifier(text, tokens)
        sc = SampleClassification(
            sample_id=fname,
            source='gibberish',
            n_tokens=len(tokens),
            most_likely_type=result['most_likely_type'],
            confidence=result['confidence'],
            h2_h1_ratio=result['h2_h1_ratio'],
            ttr=result['ttr'],
            zipf_r_squared=result['zipf_r_squared'],
            entropy_floor=result['entropy_floor'],
            entropy_curve=result['entropy_curve'],
        )
        gib_results.append(sc)

        # Parse curve back to int keys for comparison
        curve_int = {int(k): v for k, v in result['entropy_curve'].items()}
        gib_curves.append(curve_int)

        if (i + 1) % 10 == 0 or i == n_gib - 1:
            print(f"    Classified {i + 1}/{n_gib}: {fname} -> {result['most_likely_type']}")

    # Aggregate gibberish
    gib_type_dist: Dict[str, int] = Counter(r.most_likely_type for r in gib_results)
    gib_enc_nat = gib_type_dist.get('encoded_natural', 0)
    gib_h2_h1_vals = [r.h2_h1_ratio for r in gib_results]
    gib_h2_h1_mean = float(np.mean(gib_h2_h1_vals)) if gib_h2_h1_vals else 0.0
    gib_h2_h1_std = float(np.std(gib_h2_h1_vals)) if gib_h2_h1_vals else 0.0
    gib_floor_vals = [r.entropy_floor for r in gib_results]
    gib_floor_mean = float(np.mean(gib_floor_vals)) if gib_floor_vals else 0.0

    print(f"\n    Gibberish type distribution: {dict(gib_type_dist)}")
    print(f"    Classified as encoded_natural: {gib_enc_nat}/{n_gib}")
    print(f"    H2/H1 mean={gib_h2_h1_mean:.3f} +/- {gib_h2_h1_std:.3f} "
          f"(Voynich={voynich_h2_h1:.3f})")
    print(f"    Entropy floor mean={gib_floor_mean:.3f} "
          f"(Voynich={voynich_floor:.3f})")

    # ── 3. Generate and classify Timm-Schinner text ───────────────────
    print("\n  3. Generating and classifying Timm-Schinner self-citation text ...")

    char_freq = _build_voynich_char_freq(voynich_text)
    word_lengths = _build_voynich_word_lengths(voynich_tokens)

    # Default parameters: 10 samples
    timm_results: List[SampleClassification] = []
    timm_curves: List[Dict[int, float]] = []
    timm_sensitivity: List[Dict] = []

    default_params = {'p_copy': 0.7, 'p_mutate': 0.10, 'buffer_size': 100}

    print("    Default parameters (p_copy=0.7, p_mutate=0.10, buffer=100):")
    for trial in range(10):
        tokens = _generate_timm_schinner(
            char_freq, word_lengths,
            seed=42 + trial,
            **default_params,
        )
        text = ' '.join(tokens)
        result = _run_typology_classifier(text, tokens)
        sc = SampleClassification(
            sample_id=f'timm_default_{trial}',
            source='timm_schinner',
            n_tokens=len(tokens),
            most_likely_type=result['most_likely_type'],
            confidence=result['confidence'],
            h2_h1_ratio=result['h2_h1_ratio'],
            ttr=result['ttr'],
            zipf_r_squared=result['zipf_r_squared'],
            entropy_floor=result['entropy_floor'],
            entropy_curve=result['entropy_curve'],
        )
        timm_results.append(sc)
        curve_int = {int(k): v for k, v in result['entropy_curve'].items()}
        timm_curves.append(curve_int)

    default_types = Counter(r.most_likely_type for r in timm_results[:10])
    print(f"      Distribution: {dict(default_types)}")

    # Sensitivity grid
    sensitivity_configs = [
        {'p_copy': 0.6, 'p_mutate': 0.10, 'buffer_size': 100},
        {'p_copy': 0.8, 'p_mutate': 0.10, 'buffer_size': 100},
        {'p_copy': 0.7, 'p_mutate': 0.05, 'buffer_size': 100},
        {'p_copy': 0.7, 'p_mutate': 0.15, 'buffer_size': 100},
        {'p_copy': 0.7, 'p_mutate': 0.10, 'buffer_size': 50},
        {'p_copy': 0.7, 'p_mutate': 0.10, 'buffer_size': 200},
    ]

    print("    Sensitivity analysis:")
    for cfg in sensitivity_configs:
        cfg_types: List[str] = []
        for trial in range(3):
            tokens = _generate_timm_schinner(
                char_freq, word_lengths,
                seed=200 + trial,
                **cfg,
            )
            text = ' '.join(tokens)
            result = _run_typology_classifier(text, tokens)
            sc = SampleClassification(
                sample_id=f"timm_p{cfg['p_copy']}_m{cfg['p_mutate']}_b{cfg['buffer_size']}_{trial}",
                source='timm_schinner',
                n_tokens=len(tokens),
                most_likely_type=result['most_likely_type'],
                confidence=result['confidence'],
                h2_h1_ratio=result['h2_h1_ratio'],
                ttr=result['ttr'],
                zipf_r_squared=result['zipf_r_squared'],
                entropy_floor=result['entropy_floor'],
                entropy_curve=result['entropy_curve'],
            )
            timm_results.append(sc)
            cfg_types.append(result['most_likely_type'])

        cfg_dist = dict(Counter(cfg_types))
        timm_sensitivity.append({
            'params': cfg,
            'type_distribution': cfg_dist,
            'n_encoded_natural': cfg_types.count('encoded_natural'),
        })
        print(f"      p_copy={cfg['p_copy']}, p_mutate={cfg['p_mutate']}, "
              f"buffer={cfg['buffer_size']}: {cfg_dist}")

    # Aggregate Timm-Schinner
    timm_type_dist: Dict[str, int] = Counter(r.most_likely_type for r in timm_results)
    timm_enc_nat = timm_type_dist.get('encoded_natural', 0)
    n_timm = len(timm_results)

    print(f"\n    Timm-Schinner total: {n_timm} samples")
    print(f"    Type distribution: {dict(timm_type_dist)}")
    print(f"    Classified as encoded_natural: {timm_enc_nat}/{n_timm}")

    # ── 4. Build comparison table ─────────────────────────────────────
    print("\n  4. Building comparison table ...")

    # Latin reference features
    latin_tokens = latin_text.split() if latin_text else []
    if latin_tokens and len(latin_tokens) >= 20:
        latin_result = _run_typology_classifier(latin_text, latin_tokens)
    else:
        latin_result = {
            'most_likely_type': 'natural',
            'confidence': 0.0,
            'h2_h1_ratio': 0.0,
            'ttr': 0.0,
            'zipf_r_squared': 0.0,
            'entropy_floor': 0.0,
        }

    # Find best-matching gibberish sample (closest H2/H1 to Voynich)
    best_gib_idx = 0
    best_gib_dist = float('inf')
    for i, r in enumerate(gib_results):
        dist = abs(r.h2_h1_ratio - voynich_h2_h1)
        if dist < best_gib_dist:
            best_gib_dist = dist
            best_gib_idx = i

    comparison_table = [
        {
            'text': 'Voynich Language A',
            'classification': voynich_type,
            'h2_h1_ratio': round(voynich_h2_h1, 3),
            'zipf_r2': round(_compute_text_stats(voynich_text, voynich_tokens)['zipf_r_squared'], 3),
            'ttr': round(_compute_text_stats(voynich_text, voynich_tokens)['ttr'], 3),
            'entropy_floor': round(voynich_floor, 3),
        },
        {
            'text': 'Latin (reference)',
            'classification': latin_result['most_likely_type'],
            'h2_h1_ratio': round(latin_result['h2_h1_ratio'], 3),
            'zipf_r2': round(latin_result['zipf_r_squared'], 3),
            'ttr': round(latin_result['ttr'], 3),
            'entropy_floor': round(latin_result['entropy_floor'], 3),
        },
        {
            'text': 'Gibberish mean',
            'classification': max(gib_type_dist, key=gib_type_dist.get) if gib_type_dist else 'none',
            'h2_h1_ratio': round(gib_h2_h1_mean, 3),
            'zipf_r2': round(float(np.mean([r.zipf_r_squared for r in gib_results])), 3) if gib_results else 0.0,
            'ttr': round(float(np.mean([r.ttr for r in gib_results])), 3) if gib_results else 0.0,
            'entropy_floor': round(gib_floor_mean, 3),
        },
        {
            'text': f'Gibberish best-match ({gib_results[best_gib_idx].sample_id})',
            'classification': gib_results[best_gib_idx].most_likely_type if gib_results else 'none',
            'h2_h1_ratio': round(gib_results[best_gib_idx].h2_h1_ratio, 3) if gib_results else 0.0,
            'zipf_r2': round(gib_results[best_gib_idx].zipf_r_squared, 3) if gib_results else 0.0,
            'ttr': round(gib_results[best_gib_idx].ttr, 3) if gib_results else 0.0,
            'entropy_floor': round(gib_results[best_gib_idx].entropy_floor, 3) if gib_results else 0.0,
        },
        {
            'text': 'Timm-Schinner default',
            'classification': max(default_types, key=default_types.get) if default_types else 'none',
            'h2_h1_ratio': round(float(np.mean([r.h2_h1_ratio for r in timm_results[:10]])), 3),
            'zipf_r2': round(float(np.mean([r.zipf_r_squared for r in timm_results[:10]])), 3),
            'ttr': round(float(np.mean([r.ttr for r in timm_results[:10]])), 3),
            'entropy_floor': round(float(np.mean([r.entropy_floor for r in timm_results[:10]])), 3),
        },
    ]

    print("\n    Comparison Table:")
    print(f"    {'Text':<35s} {'Type':<18s} {'H2/H1':>6s} {'Zipf R2':>7s} "
          f"{'TTR':>6s} {'Floor':>6s}")
    print("    " + "-" * 82)
    for row in comparison_table:
        print(f"    {row['text']:<35s} {row['classification']:<18s} "
              f"{row['h2_h1_ratio']:>6.3f} {row['zipf_r2']:>7.3f} "
              f"{row['ttr']:>6.3f} {row['entropy_floor']:>6.3f}")

    # ── 5. Entropy curve comparison ───────────────────────────────────
    print("\n  5. Entropy curve comparison ...")

    ecurve_comparison = _entropy_curve_comparison(
        voynich_curve, latin_curve, gib_curves, timm_curves,
    )
    print(f"    Voynich floor: {ecurve_comparison['voynich_floor']:.3f}")
    print(f"    Latin floor: {ecurve_comparison['latin_floor']:.3f}")
    if 'gibberish_floor_mean' in ecurve_comparison:
        print(f"    Gibberish floor mean: {ecurve_comparison['gibberish_floor_mean']:.3f}")
        print(f"    Gibberish elevated floor: "
              f"{ecurve_comparison.get('gibberish_elevated_floor_count', 0)}/{n_gib}")

    # ── 6. Gate and verdict ───────────────────────────────────────────
    print("\n  6. Gate assessment ...")

    # Discriminant power: fraction of all non-Voynich samples NOT classified
    # as encoded_natural
    total_controls = n_gib + n_timm
    total_enc_nat = gib_enc_nat + timm_enc_nat
    discriminant_power = 1.0 - (total_enc_nat / total_controls) if total_controls > 0 else 0.0

    if gib_enc_nat == 0 and timm_enc_nat == 0:
        verdict = "CLASSIFIER_ROBUST"
        gate_passed = True
    elif gib_enc_nat <= 5 and timm_enc_nat <= 2:
        verdict = "PARTIALLY_ROBUST"
        gate_passed = True
    else:
        verdict = "CLASSIFIER_COMPROMISED"
        gate_passed = False

    print(f"    Gibberish encoded_natural: {gib_enc_nat}/{n_gib}")
    print(f"    Timm-Schinner encoded_natural: {timm_enc_nat}/{n_timm}")
    print(f"    Discriminant power: {discriminant_power:.3f}")
    print(f"    Gate: {'PASS' if gate_passed else 'FAIL'}")
    print(f"    Verdict: {verdict}")

    # Methodological note
    methodological_note = (
        "The Gaskell-Bowern (2022) ML classifier used word-length autocorrelation, "
        "triple-repeat rates, and character placement biases. The Phase 9.5 classifier "
        "uses entropy ratios (H2/H1), Zipf R-squared, and type-token ratio. These are "
        "largely non-overlapping feature sets, so the two classifiers may legitimately "
        "disagree on the same texts."
    )

    # ── 7. Save ───────────────────────────────────────────────────────
    result = GibberishTypologyResult(
        timestamp=time.strftime('%Y-%m-%dT%H:%M:%S'),
        n_gibberish_samples=n_gib,
        gibberish_results=[_convert(asdict(r)) for r in gib_results],
        gibberish_type_distribution=dict(gib_type_dist),
        gibberish_encoded_natural_count=gib_enc_nat,
        gibberish_h2_h1_mean=round(gib_h2_h1_mean, 4),
        gibberish_h2_h1_std=round(gib_h2_h1_std, 4),
        gibberish_entropy_floor_mean=round(gib_floor_mean, 4),
        n_timm_samples=n_timm,
        timm_results=[_convert(asdict(r)) for r in timm_results],
        timm_type_distribution=dict(timm_type_dist),
        timm_encoded_natural_count=timm_enc_nat,
        timm_parameter_sensitivity=timm_sensitivity,
        comparison_table=comparison_table,
        entropy_curve_comparison=_convert(ecurve_comparison),
        voynich_type=voynich_type,
        voynich_h2_h1=round(voynich_h2_h1, 4),
        voynich_entropy_floor=round(voynich_floor, 4),
        methodological_note=methodological_note,
        discriminant_power=round(discriminant_power, 4),
        gate_passed=gate_passed,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'gibberish_typology.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2)
    print(f"\n  -> {out_path}")
