"""
Phase 55B – Currier Cross-Boundary Self-Correlation
====================================================
Measures the mutual information between the last character of word N and
the first character of word N+1 in the Voynich manuscript (Currier's ~4×
self-correlation anomaly) and tests whether the tachygraphic simulation
(with syllable-as-token encoding) reproduces this statistic.

Four entry points:
    run_currier_voynich()   → phase55_currier_voynich.json
    run_currier_tachy()     → phase55_currier_tachy.json
    run_currier_controls()  → phase55_currier_controls.json
    run_currier_verdict()   → phase55_currier_verdict.json

Dependency chain for verdict:
    phase55_currier_voynich.json
    phase55_currier_tachy.json
    phase55_currier_controls.json
        → phase55_currier_verdict.json
"""

import json
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import load_corpus, tokenize_eva_chars
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
# Core MI measurement
# ---------------------------------------------------------------------------

def measure_cross_boundary_mi(tokens: List[str]) -> Dict:
    """
    Compute mutual information between last char of token N and
    first char of token N+1.

    Returns:
        mi:         bits (mutual information)
        ratio:      weighted mean P(first|last) / P(first) across all pairs
        n_pairs:    number of consecutive pairs measured
        top_pairs:  top 10 pairs by count
    """
    pairs: List[Tuple[str, str]] = []
    for i in range(len(tokens) - 1):
        chars_this = tokenize_eva_chars(tokens[i])
        chars_next = tokenize_eva_chars(tokens[i + 1])
        if chars_this and chars_next:
            pairs.append((chars_this[-1], chars_next[0]))

    n_pairs = len(pairs)
    if n_pairs == 0:
        return {'mi': 0.0, 'ratio': 1.0, 'n_pairs': 0, 'top_pairs': []}

    joint = Counter(pairs)
    last_counts = Counter(p[0] for p in pairs)
    first_counts = Counter(p[1] for p in pairs)

    # Mutual information
    mi = 0.0
    for (last, first), count in joint.items():
        p_joint = count / n_pairs
        p_last = last_counts[last] / n_pairs
        p_first = first_counts[first] / n_pairs
        if p_joint > 0 and p_last > 0 and p_first > 0:
            mi += p_joint * np.log2(p_joint / (p_last * p_first))

    # Weighted predictability ratio: E[P(first|last) / P(first)]
    weighted_ratio = 0.0
    for (last, first), count in joint.items():
        p_first_given_last = count / last_counts[last]
        p_first = first_counts[first] / n_pairs
        if p_first > 0:
            weighted_ratio += count * (p_first_given_last / p_first)
    if n_pairs > 0:
        weighted_ratio /= n_pairs

    # Top 10 pairs
    top_pairs = [
        {
            'last': last,
            'first': first,
            'count': count,
            'ratio': round((count / last_counts[last]) / (first_counts[first] / n_pairs), 4),
        }
        for (last, first), count in joint.most_common(10)
    ]

    return {
        'mi': round(float(mi), 6),
        'ratio': round(float(weighted_ratio), 6),
        'n_pairs': n_pairs,
        'n_unique_pairs': len(joint),
        'n_unique_last': len(last_counts),
        'n_unique_first': len(first_counts),
        'top_pairs': top_pairs,
    }


def measure_shuffled_baseline(
    all_tokens_by_page: List[List[str]],
    n_shuffles: int = 1000,
    rng: Optional[np.random.Generator] = None,
) -> Dict:
    """
    Shuffle token order within each page and remeasure MI.
    Destroys cross-word dependency while keeping within-page structure.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    null_mis = []
    null_ratios = []

    # Flatten to one list for faster shuffling
    flat = [tok for page in all_tokens_by_page for tok in page]

    for _ in range(n_shuffles):
        shuffled = list(flat)
        rng.shuffle(shuffled)
        r = measure_cross_boundary_mi(shuffled)
        null_mis.append(r['mi'])
        null_ratios.append(r['ratio'])

    return {
        'mean_mi': round(float(np.mean(null_mis)), 6),
        'std_mi': round(float(np.std(null_mis)), 6),
        'mean_ratio': round(float(np.mean(null_ratios)), 6),
        'std_ratio': round(float(np.std(null_ratios)), 6),
        'n_shuffles': n_shuffles,
    }


# ---------------------------------------------------------------------------
# Tachygraphic encoding helpers
# ---------------------------------------------------------------------------

def _syllabify_simple(word: str) -> List[str]:
    """Split a word into CV-ish syllables (from entropy_shift_cipher.py)."""
    _VOWELS = set('aeiou')
    _CONSONANTS = set('bcdfghjklmnpqrstvwxyz')
    word = word.lower()
    syllables = []
    current = ''
    for ch in word:
        if ch not in _VOWELS and ch not in _CONSONANTS:
            continue
        current += ch
        if ch in _VOWELS:
            syllables.append(current)
            current = ''
    if current:
        if syllables:
            syllables[-1] += current
        else:
            syllables.append(current)
    return syllables if syllables else [word]


def _build_tachy_table(n_bases: int, n_mods: int, seed: int) -> Dict[str, str]:
    """
    Build tachygraphic encoding table: syllable → 2-char encoded form.
    Matches TachygraphicEncoder logic from entropy_shift_cipher.py.
    """
    import random as _random
    rng = _random.Random(seed)

    consonant_classes = [
        ['b', 'p'], ['d', 't'], ['g', 'k', 'c'],
        ['f', 'v'], ['l', 'r'], ['m', 'n'],
        ['s', 'z'],
    ][:n_bases]

    vowel_mods = ['a', 'e', 'i', 'o', 'u'][:n_mods]

    table: Dict[str, str] = {}
    alpha = 'abcdefghijklmnopqrstuvwxyz'

    for bi, consonants in enumerate(consonant_classes):
        base_char = alpha[bi * 2]
        for vi, vowel in enumerate(vowel_mods):
            mod_char = (alpha[bi * 2 + 1] if vi % 2 == 0
                        else alpha[20 + vi % 6] if 20 + vi % 6 < 26
                        else alpha[vi % 20])
            for consonant in consonants:
                syl = consonant + vowel
                table[syl] = base_char + mod_char

    for vi, vowel in enumerate(vowel_mods):
        table[vowel] = alpha[14 + vi] if 14 + vi < 26 else alpha[vi]

    return table


def _encode_syllable(syl: str, table: Dict[str, str]) -> str:
    """Encode one syllable using the tachygraphic table."""
    if syl in table:
        return table[syl]
    # Fallback: encode char by char
    result = []
    for ch in syl:
        if ch + 'a' in table:
            result.append(table[ch + 'a'][:1])
        elif ch in table:
            result.append(table[ch])
        else:
            result.append(ch)
    return ''.join(result) if result else syl


def build_tachy_syllable_tokens(latin_tokens: List[str], table: Dict[str, str]) -> List[str]:
    """
    Variant A (syllable-as-token): each Latin syllable → one output token.
    Syllable boundaries become word boundaries in the output.
    """
    output = []
    for word in latin_tokens:
        syls = _syllabify_simple(word)
        for syl in syls:
            encoded = _encode_syllable(syl, table)
            if encoded:
                output.append(encoded)
    return output


def build_tachy_word_tokens(latin_tokens: List[str], table: Dict[str, str]) -> List[str]:
    """
    Variant B (word-as-token): each Latin word's syllables concatenated → one token.
    Only Latin word boundaries become word boundaries.
    """
    output = []
    for word in latin_tokens:
        syls = _syllabify_simple(word)
        parts = [_encode_syllable(syl, table) for syl in syls]
        token = ''.join(parts)
        if token:
            output.append(token)
    return output


# ---------------------------------------------------------------------------
# B.1: Voynich measurement
# ---------------------------------------------------------------------------

def run_currier_voynich() -> None:
    """Phase 55B.1: Measure Currier self-correlation on real Voynich."""
    t0 = time.time()
    rd = str(_results_dir())

    print("=" * 70)
    print("PHASE 55B.1: Currier Self-Correlation — Real Voynich")
    print("=" * 70)

    corpus = load_corpus(verbose=False)

    # Collect tokens page by page (avoid crossing folio boundaries)
    print("\n  1. Collecting token pairs by page …")
    all_tokens_by_page = []
    all_pairs: List[Tuple[str, str]] = []

    for folio, page in corpus.pages.items():
        page_tokens = [t for t in page.all_tokens if t]
        if len(page_tokens) < 2:
            continue
        all_tokens_by_page.append(page_tokens)

    # Flatten and measure
    flat_tokens = [t for page in all_tokens_by_page for t in page]
    print(f"    {len(corpus.pages)} pages, {len(flat_tokens):,} tokens")

    # Within-page measurement: collect pairs only within each page
    joint_counter: Counter = Counter()
    last_counts_c: Counter = Counter()
    first_counts_c: Counter = Counter()
    n_pairs = 0

    for page_tokens in all_tokens_by_page:
        for i in range(len(page_tokens) - 1):
            chars_this = tokenize_eva_chars(page_tokens[i])
            chars_next = tokenize_eva_chars(page_tokens[i + 1])
            if chars_this and chars_next:
                pair = (chars_this[-1], chars_next[0])
                joint_counter[pair] += 1
                last_counts_c[pair[0]] += 1
                first_counts_c[pair[1]] += 1
                n_pairs += 1

    print(f"    Within-page pairs: {n_pairs:,}")

    # MI and ratio
    mi = 0.0
    for (last, first), count in joint_counter.items():
        p_joint = count / n_pairs
        p_last = last_counts_c[last] / n_pairs
        p_first = first_counts_c[first] / n_pairs
        if p_joint > 0 and p_last > 0 and p_first > 0:
            mi += p_joint * np.log2(p_joint / (p_last * p_first))

    weighted_ratio = 0.0
    for (last, first), count in joint_counter.items():
        p_first_given_last = count / last_counts_c[last]
        p_first = first_counts_c[first] / n_pairs
        if p_first > 0:
            weighted_ratio += count * (p_first_given_last / p_first)
    weighted_ratio /= n_pairs if n_pairs > 0 else 1

    top_pairs = [
        {
            'last': last,
            'first': first,
            'count': count,
            'ratio': round((count / last_counts_c[last]) / (first_counts_c[first] / n_pairs), 3),
        }
        for (last, first), count in joint_counter.most_common(10)
    ]

    print(f"    MI = {mi:.4f} bits")
    print(f"    Weighted ratio = {weighted_ratio:.4f}×")
    print(f"    Top pair: {joint_counter.most_common(1)}")

    # ── Null baseline (1000 shuffles) ────────────────────────────────────
    print("\n  2. Computing null baseline (1000 shuffles) …")

    null = measure_shuffled_baseline(all_tokens_by_page, n_shuffles=1000)
    print(f"    Null MI = {null['mean_mi']:.4f} ± {null['std_mi']:.4f}")
    print(f"    Null ratio = {null['mean_ratio']:.4f} ± {null['std_ratio']:.4f}")

    mi_z = (mi - null['mean_mi']) / null['std_mi'] if null['std_mi'] > 0 else 0.0
    ratio_z = (weighted_ratio - null['mean_ratio']) / null['std_ratio'] if null['std_ratio'] > 0 else 0.0
    print(f"    MI z-score = {mi_z:.2f}")
    print(f"    Ratio z-score = {ratio_z:.2f}")

    runtime = round(time.time() - t0, 2)
    output = {
        'phase': '55B.1',
        'experiment': 'currier_voynich',
        'n_pages': len(all_tokens_by_page),
        'n_tokens': len(flat_tokens),
        'n_pairs': n_pairs,
        'mi': round(float(mi), 6),
        'ratio': round(float(weighted_ratio), 6),
        'mi_z_vs_null': round(float(mi_z), 3),
        'ratio_z_vs_null': round(float(ratio_z), 3),
        'top_pairs': top_pairs,
        'null_baseline': null,
        'runtime_seconds': runtime,
    }

    out_path = _save_json(rd, 'phase55_currier_voynich.json', output)
    print(f"\n  Saved → {out_path}")
    print(f"  Completed in {runtime:.1f}s")


# ---------------------------------------------------------------------------
# B.2: Tachygraphic simulation measurement
# ---------------------------------------------------------------------------

def run_currier_tachy() -> None:
    """Phase 55B.2: Measure Currier self-correlation on tachygraphic simulation."""
    t0 = time.time()
    rd = str(_results_dir())

    print("=" * 70)
    print("PHASE 55B.2: Currier Self-Correlation — Tachygraphic Simulation")
    print("=" * 70)

    # Load Latin reference tokens
    print("\n  1. Loading Latin reference corpus …")
    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    latin_tokens = ref_corpus.get_combined_tokens('latin')
    if not latin_tokens:
        raise RuntimeError("No Latin tokens found in reference corpus")
    print(f"    Latin tokens: {len(latin_tokens):,}")

    n_seeds = 20
    # Use C5_V4 configuration (best config from Phase 19.6)
    n_bases = 5
    n_mods = 4

    syllable_results = []
    word_results = []

    print(f"\n  2. Tachygraphic encoding ({n_seeds} seeds, C{n_bases}_V{n_mods}) …")

    for seed in range(n_seeds):
        table = _build_tachy_table(n_bases, n_mods, seed)

        # Variant A: syllable-as-token
        syl_tokens = build_tachy_syllable_tokens(latin_tokens, table)
        r_syl = measure_cross_boundary_mi(syl_tokens)
        syllable_results.append(r_syl)

        # Variant B: word-as-token
        word_tokens = build_tachy_word_tokens(latin_tokens, table)
        r_word = measure_cross_boundary_mi(word_tokens)
        word_results.append(r_word)

        if seed % 5 == 0:
            print(f"    seed {seed:2d}: "
                  f"syl_ratio={r_syl['ratio']:.4f}  "
                  f"word_ratio={r_word['ratio']:.4f}  "
                  f"syl_tokens={r_syl['n_pairs']:,}")

    syl_ratios = [r['ratio'] for r in syllable_results]
    word_ratios = [r['ratio'] for r in word_results]
    syl_mis = [r['mi'] for r in syllable_results]
    word_mis = [r['mi'] for r in word_results]

    syl_summary = {
        'mean_ratio': round(float(np.mean(syl_ratios)), 6),
        'std_ratio': round(float(np.std(syl_ratios)), 6),
        'mean_mi': round(float(np.mean(syl_mis)), 6),
        'std_mi': round(float(np.std(syl_mis)), 6),
        'n_seeds': n_seeds,
    }
    word_summary = {
        'mean_ratio': round(float(np.mean(word_ratios)), 6),
        'std_ratio': round(float(np.std(word_ratios)), 6),
        'mean_mi': round(float(np.mean(word_mis)), 6),
        'std_mi': round(float(np.std(word_mis)), 6),
        'n_seeds': n_seeds,
    }

    print(f"\n  Syllable-as-token: ratio={syl_summary['mean_ratio']:.4f} "
          f"± {syl_summary['std_ratio']:.4f}")
    print(f"  Word-as-token:    ratio={word_summary['mean_ratio']:.4f} "
          f"± {word_summary['std_ratio']:.4f}")

    runtime = round(time.time() - t0, 2)
    output = {
        'phase': '55B.2',
        'experiment': 'currier_tachy',
        'config': f'C{n_bases}_V{n_mods}',
        'n_seeds': n_seeds,
        'n_latin_tokens': len(latin_tokens),
        'tachygraphic_syllable': syl_summary,
        'tachygraphic_word': word_summary,
        'per_seed': [
            {
                'seed': i,
                'syllable': syllable_results[i],
                'word': word_results[i],
            }
            for i in range(n_seeds)
        ],
        'runtime_seconds': runtime,
    }

    out_path = _save_json(rd, 'phase55_currier_tachy.json', output)
    print(f"\n  Saved → {out_path}")
    print(f"  Completed in {runtime:.1f}s")


# ---------------------------------------------------------------------------
# B.3: Control measurements (Latin, Schinner, Cardan)
# ---------------------------------------------------------------------------

def run_currier_controls() -> None:
    """Phase 55B.3: Measure self-correlation on Latin, Schinner, Cardan controls."""
    t0 = time.time()
    rd = str(_results_dir())

    print("=" * 70)
    print("PHASE 55B.3: Currier Self-Correlation — Control Corpora")
    print("=" * 70)

    # ── Latin plaintext ───────────────────────────────────────────────────
    print("\n  1. Latin plaintext …")
    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    latin_tokens = ref_corpus.get_combined_tokens('latin')
    latin_result = measure_cross_boundary_mi(latin_tokens)
    print(f"    MI={latin_result['mi']:.4f}  ratio={latin_result['ratio']:.4f}  "
          f"n_pairs={latin_result['n_pairs']:,}")

    # ── Schinner (re-generate from scratch, seed=0) ──────────────────────
    print("\n  2. Schinner model (regenerating seed=0) …")

    from voynich.phases.schinner_generator import (
        build_schinner_model, generate_schinner_text
    )

    corpus = load_corpus(verbose=False)
    voynich_text = corpus.get_text()
    corpus_tokens_voynich = [t for t in voynich_text.split() if t]
    schinner_model = build_schinner_model(corpus_tokens_voynich)

    n_tokens = 36_000
    rng0 = np.random.default_rng(0)

    schinner_simple_tokens = generate_schinner_text(
        schinner_model, n_tokens, positional=False, rng=np.random.default_rng(0)
    )
    schinner_pos_tokens = generate_schinner_text(
        schinner_model, n_tokens, positional=True, rng=np.random.default_rng(1)
    )

    schinner_simple_result = measure_cross_boundary_mi(schinner_simple_tokens)
    schinner_pos_result = measure_cross_boundary_mi(schinner_pos_tokens)

    print(f"    schinner_simple:    MI={schinner_simple_result['mi']:.4f}  "
          f"ratio={schinner_simple_result['ratio']:.4f}")
    print(f"    schinner_positional: MI={schinner_pos_result['mi']:.4f}  "
          f"ratio={schinner_pos_result['ratio']:.4f}")

    # ── Cardan (re-generate from scratch, seed=0) ────────────────────────
    print("\n  3. Cardan grille (regenerating seed=0) …")

    from voynich.phases.cardan_generator import (
        build_syllable_table, generate_cardan_text
    )

    table = build_syllable_table(corpus_tokens_voynich)

    cardan_3_tokens = generate_cardan_text(
        table, n_tokens, n_holes=3, n_grilles=10, rng=np.random.default_rng(100)
    )
    cardan_4_tokens = generate_cardan_text(
        table, n_tokens, n_holes=4, n_grilles=10, rng=np.random.default_rng(101)
    )
    cardan_3_tokens = [t for t in cardan_3_tokens if t]
    cardan_4_tokens = [t for t in cardan_4_tokens if t]

    cardan_3_result = measure_cross_boundary_mi(cardan_3_tokens)
    cardan_4_result = measure_cross_boundary_mi(cardan_4_tokens)

    print(f"    cardan_3hole: MI={cardan_3_result['mi']:.4f}  "
          f"ratio={cardan_3_result['ratio']:.4f}")
    print(f"    cardan_4hole: MI={cardan_4_result['mi']:.4f}  "
          f"ratio={cardan_4_result['ratio']:.4f}")

    runtime = round(time.time() - t0, 2)
    output = {
        'phase': '55B.3',
        'experiment': 'currier_controls',
        'latin_plaintext': latin_result,
        'schinner_simple': schinner_simple_result,
        'schinner_positional': schinner_pos_result,
        'cardan_3hole': cardan_3_result,
        'cardan_4hole': cardan_4_result,
        'n_tokens_generated': n_tokens,
        'runtime_seconds': runtime,
    }

    out_path = _save_json(rd, 'phase55_currier_controls.json', output)
    print(f"\n  Saved → {out_path}")
    print(f"  Completed in {runtime:.1f}s")


# ---------------------------------------------------------------------------
# B.4: Verdict
# ---------------------------------------------------------------------------

def run_currier_verdict() -> None:
    """Phase 55B.4: Integrate Currier self-correlation results into verdict."""
    t0 = time.time()
    rd = str(_results_dir())

    print("=" * 70)
    print("PHASE 55B.4: Currier Self-Correlation — Verdict")
    print("=" * 70)

    voynich_data = _safe_load(os.path.join(rd, 'phase55_currier_voynich.json'))
    tachy_data = _safe_load(os.path.join(rd, 'phase55_currier_tachy.json'))
    controls_data = _safe_load(os.path.join(rd, 'phase55_currier_controls.json'))

    for name, data in [('voynich', voynich_data), ('tachy', tachy_data),
                        ('controls', controls_data)]:
        if not data:
            raise FileNotFoundError(
                f"phase55_currier_{name}.json not found — run currier-{name} first"
            )

    # Extract key metrics
    voynich_ratio = voynich_data.get('ratio', 0.0)
    null_ratio = voynich_data.get('null_baseline', {}).get('mean_ratio', 1.0)
    null_std = voynich_data.get('null_baseline', {}).get('std_ratio', 0.1)

    tachy_syl_ratio = tachy_data.get('tachygraphic_syllable', {}).get('mean_ratio', 0.0)
    tachy_word_ratio = tachy_data.get('tachygraphic_word', {}).get('mean_ratio', 0.0)

    latin_ratio = controls_data.get('latin_plaintext', {}).get('ratio', 0.0)
    schinner_simple_ratio = controls_data.get('schinner_simple', {}).get('ratio', 0.0)
    schinner_pos_ratio = controls_data.get('schinner_positional', {}).get('ratio', 0.0)
    cardan_3_ratio = controls_data.get('cardan_3hole', {}).get('ratio', 0.0)
    cardan_4_ratio = controls_data.get('cardan_4hole', {}).get('ratio', 0.0)

    print(f"\n  Summary of ratios:")
    print(f"    Real Voynich:           {voynich_ratio:.4f}×")
    print(f"    Shuffled null:          {null_ratio:.4f}×")
    print(f"    Tachy (syllable-token): {tachy_syl_ratio:.4f}×")
    print(f"    Tachy (word-token):     {tachy_word_ratio:.4f}×")
    print(f"    Latin plaintext:        {latin_ratio:.4f}×")
    print(f"    Schinner simple:        {schinner_simple_ratio:.4f}×")
    print(f"    Schinner positional:    {schinner_pos_ratio:.4f}×")
    print(f"    Cardan 3-hole:          {cardan_3_ratio:.4f}×")
    print(f"    Cardan 4-hole:          {cardan_4_ratio:.4f}×")

    # Gates
    print("\n  Gates:")

    g1 = voynich_ratio > 2.5
    print(f"    G1 Voynich ratio > 2.5×: {'PASS' if g1 else 'FAIL'} ({voynich_ratio:.4f})")

    g2 = null_ratio < 1.5
    print(f"    G2 Null ratio < 1.5×:    {'PASS' if g2 else 'FAIL'} ({null_ratio:.4f})")

    prediction_match = (voynich_ratio > 0 and
                        abs(tachy_syl_ratio - voynich_ratio) / voynich_ratio < 0.30)
    g3 = prediction_match
    print(f"    G3 Tachy syl within 30% of Voynich: {'PASS' if g3 else 'FAIL'} "
          f"({tachy_syl_ratio:.4f} vs {voynich_ratio:.4f}, "
          f"diff={abs(tachy_syl_ratio - voynich_ratio) / voynich_ratio:.1%})")

    g4 = tachy_syl_ratio > tachy_word_ratio
    print(f"    G4 Syl ratio > word ratio: {'PASS' if g4 else 'FAIL'} "
          f"({tachy_syl_ratio:.4f} vs {tachy_word_ratio:.4f})")

    g5 = latin_ratio < 2.0
    print(f"    G5 Latin ratio < 2.0×: {'PASS' if g5 else 'FAIL'} ({latin_ratio:.4f})")

    gates = {'G1': g1, 'G2': g2, 'G3': g3, 'G4': g4, 'G5': g5}
    n_passed = sum(gates.values())

    # Does Schinner reproduce Currier's anomaly?
    schinner_reproduces = max(schinner_simple_ratio, schinner_pos_ratio) > 3.0
    print(f"\n  Schinner reproduces anomaly (>3×): {schinner_reproduces}")

    # Verdict
    if g3 and n_passed >= 4:
        if schinner_reproduces:
            verdict = 'PREDICTION_CONFIRMED_NOT_UNIQUE'
        else:
            verdict = 'PREDICTION_CONFIRMED_UNIQUE'
    elif g3 and n_passed >= 2:
        verdict = 'PREDICTION_PARTIAL'
    else:
        verdict = 'PREDICTION_FAILED'

    print(f"\n  VERDICT: {verdict}  ({n_passed}/5 gates passed)")

    runtime = round(time.time() - t0, 2)
    output = {
        'phase': '55B.4',
        'experiment': 'currier_verdict',
        'ratios': {
            'voynich': voynich_ratio,
            'null': null_ratio,
            'tachy_syllable': tachy_syl_ratio,
            'tachy_word': tachy_word_ratio,
            'latin': latin_ratio,
            'schinner_simple': schinner_simple_ratio,
            'schinner_positional': schinner_pos_ratio,
            'cardan_3hole': cardan_3_ratio,
            'cardan_4hole': cardan_4_ratio,
        },
        'prediction_match': prediction_match,
        'schinner_reproduces': schinner_reproduces,
        'gates': gates,
        'n_gates_passed': n_passed,
        'verdict': verdict,
        'runtime_seconds': runtime,
    }

    out_path = _save_json(rd, 'phase55_currier_verdict.json', output)
    print(f"\n  Saved → {out_path}")
    print(f"  Completed in {runtime:.1f}s")
