"""
Phase 55A.1 – Schinner Stochastic Generator
=============================================
Reconstructs Schinner (2007)'s position-conditioned character-level Markov
process from the published algorithm description and measures its entropy
shift cosine vs the Voynich target shift.

Dependency chain:
    results/entropy_shift_cipher.json   (latin baseline + observed shift)
    corpus (Voynich EVA tokens)
        → results/phase55_schinner_gen.json
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
from voynich.core.stats import cosine_similarity, entropy_curve


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
# Model building
# ---------------------------------------------------------------------------

def build_schinner_model(
    corpus_tokens: List[str],
    rng: Optional[np.random.Generator] = None,
) -> Dict:
    """
    Build position-conditioned bigram transition matrices from Voynich tokens.

    Each token is decomposed into EVA characters. Three transition matrices
    are built (initial / medial / final) based on bigram position within word.

    Returns a dict with:
        - 'chars': sorted list of EVA character types
        - 'char_to_idx': dict
        - 'idx_to_char': list
        - 'trans_simple': np.ndarray (n_chars × n_chars), unconditional
        - 'trans_initial': np.ndarray, position 0→1
        - 'trans_medial': np.ndarray, positions 1..(n-2) → 2..(n-1)
        - 'trans_final': np.ndarray, position (n-2)→(n-1)
        - 'initial_dist': np.ndarray (first character of words)
        - 'length_dist': np.ndarray (word length PMF, index = length)
        - 'max_len': int
    """
    # Build character inventory
    char_set = set()
    eva_tokens = []
    for tok in corpus_tokens:
        chars = tokenize_eva_chars(tok)
        if chars:
            eva_tokens.append(chars)
            char_set.update(chars)

    chars_sorted = sorted(char_set)
    n_chars = len(chars_sorted)
    char_to_idx = {c: i for i, c in enumerate(chars_sorted)}
    idx_to_char = chars_sorted

    # Transition matrices (add-1 smoothed counts)
    trans_simple = np.ones((n_chars, n_chars), dtype=float)
    trans_initial = np.ones((n_chars, n_chars), dtype=float)
    trans_medial = np.ones((n_chars, n_chars), dtype=float)
    trans_final = np.ones((n_chars, n_chars), dtype=float)

    # Initial character distribution (first char of each word)
    initial_dist = np.ones(n_chars, dtype=float)

    # Word length distribution
    lengths = []

    for chars in eva_tokens:
        n = len(chars)
        lengths.append(n)

        # Initial char
        c0 = char_to_idx.get(chars[0])
        if c0 is not None:
            initial_dist[c0] += 1

        # Bigram transitions
        for i in range(n - 1):
            c1 = char_to_idx.get(chars[i])
            c2 = char_to_idx.get(chars[i + 1])
            if c1 is None or c2 is None:
                continue

            trans_simple[c1, c2] += 1

            if i == 0:
                pos_mat = trans_initial
            elif i == n - 2:
                pos_mat = trans_final
            else:
                pos_mat = trans_medial
            pos_mat[c1, c2] += 1

    # Row-normalize
    for mat in [trans_simple, trans_initial, trans_medial, trans_final]:
        row_sums = mat.sum(axis=1, keepdims=True)
        mat /= row_sums

    initial_dist /= initial_dist.sum()

    # Word length PMF
    max_len = max(lengths) if lengths else 10
    length_counts = np.bincount(lengths, minlength=max_len + 1).astype(float)
    length_counts[0] = 0  # zero-length words impossible
    length_dist = length_counts / length_counts.sum()

    return {
        'chars': idx_to_char,
        'char_to_idx': char_to_idx,
        'idx_to_char': idx_to_char,
        'trans_simple': trans_simple,
        'trans_initial': trans_initial,
        'trans_medial': trans_medial,
        'trans_final': trans_final,
        'initial_dist': initial_dist,
        'length_dist': length_dist,
        'max_len': max_len,
        'n_chars': n_chars,
    }


# ---------------------------------------------------------------------------
# Text generation
# ---------------------------------------------------------------------------

def generate_schinner_text(
    model: Dict,
    n_tokens: int,
    positional: bool,
    rng: np.random.Generator,
) -> List[str]:
    """
    Generate n_tokens Schinner-style EVA tokens.

    positional=False: single unconditional bigram matrix (schinner_simple)
    positional=True:  position-conditioned matrices (schinner_positional)
    """
    n_chars = model['n_chars']
    idx_to_char = model['idx_to_char']
    length_dist = model['length_dist']
    initial_dist = model['initial_dist']
    trans_simple = model['trans_simple']
    trans_initial = model['trans_initial']
    trans_medial = model['trans_medial']
    trans_final = model['trans_final']

    length_idx = np.arange(len(length_dist))
    char_idx = np.arange(n_chars)

    tokens = []
    for _ in range(n_tokens):
        # Sample word length (at least 1)
        wlen = int(rng.choice(length_idx, p=length_dist))
        if wlen < 1:
            wlen = 1

        # First character
        c_idx = int(rng.choice(char_idx, p=initial_dist))
        char_indices = [c_idx]

        # Subsequent characters
        for j in range(1, wlen):
            prev = char_indices[-1]

            if not positional:
                row = trans_simple[prev]
            elif j == 1:
                # Transition from position 0 to position 1
                row = trans_initial[prev]
            elif j == wlen - 1:
                # Generating the last character
                row = trans_final[prev]
            else:
                row = trans_medial[prev]

            c_idx = int(rng.choice(char_idx, p=row))
            char_indices.append(c_idx)

        token = ''.join(idx_to_char[i] for i in char_indices)
        tokens.append(token)

    return tokens


# ---------------------------------------------------------------------------
# Entropy computation helpers
# ---------------------------------------------------------------------------

def _compute_entropy_curve_for_tokens(
    tokens: List[str], max_order: int = 6
) -> Dict[int, float]:
    """Join tokens with spaces and compute entropy curve."""
    text = ' '.join(tokens)
    return entropy_curve(text, max_order=max_order)


def _compute_cosine_vs_shift(
    curve: Dict[int, float],
    latin_curve: Dict[int, float],
    voynich_shift: List[float],
    orders: List[int],
) -> Tuple[List[float], float]:
    """Compute shift vector and cosine similarity vs Voynich target."""
    shift = [curve.get(k, 0.0) - latin_curve.get(k, 0.0) for k in orders]
    shift_arr = np.array(shift)
    voynich_arr = np.array(voynich_shift)
    cos = float(cosine_similarity(shift_arr, voynich_arr))
    return shift, cos


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_schinner_gen() -> None:
    """Phase 55A.1: Schinner stochastic generator entropy shift."""
    t0 = time.time()
    rd = str(_results_dir())

    print("=" * 70)
    print("PHASE 55A.1: Schinner Stochastic Generator")
    print("=" * 70)

    # ── 1. Load Phase 19.2 baseline ──────────────────────────────────────
    print("\n  1. Loading Phase 19.2 Latin baseline + Voynich shift …")

    phase19_path = os.path.join(rd, 'entropy_shift_cipher.json')
    phase19 = _safe_load(phase19_path)

    if not phase19:
        raise FileNotFoundError(
            "entropy_shift_cipher.json not found — run phase19 first"
        )

    voynich_shift = phase19.get('observed_shift_vector', [])
    latin_curve_raw = phase19.get('latin_entropy_curve', {})
    # Keys are strings in JSON; convert to int
    latin_curve = {int(k): float(v) for k, v in latin_curve_raw.items()}

    orders = list(range(7))  # H0–H6
    print(f"    Voynich shift: {[f'{s:.3f}' for s in voynich_shift]}")
    print(f"    Latin H0={latin_curve.get(0, 0):.3f}, H2={latin_curve.get(2, 0):.3f}")

    # ── 2. Load Voynich corpus and build Schinner model ──────────────────
    print("\n  2. Building Schinner model from Voynich EVA tokens …")

    corpus = load_corpus(verbose=False)
    voynich_text = corpus.get_text()
    corpus_tokens = [t for t in voynich_text.split() if t]

    model = build_schinner_model(corpus_tokens)
    n_chars = model['n_chars']
    print(f"    Corpus tokens: {len(corpus_tokens):,}")
    print(f"    EVA char inventory: {n_chars} characters")

    # ── 3. Generate 2 variants × 20 seeds ───────────────────────────────
    n_seeds = 20
    n_tokens = 36_000
    max_order = 6

    variants = [
        ('schinner_simple', False),
        ('schinner_positional', True),
    ]

    results_by_variant = {}

    for variant_name, positional in variants:
        print(f"\n  3. Generating {variant_name} ({n_seeds} seeds × {n_tokens:,} tokens) …")

        per_seed = []
        for seed in range(n_seeds):
            rng = np.random.default_rng(seed)
            tokens = generate_schinner_text(model, n_tokens, positional=positional, rng=rng)
            curve = _compute_entropy_curve_for_tokens(tokens, max_order=max_order)
            shift, cos = _compute_cosine_vs_shift(curve, latin_curve, voynich_shift, orders)

            per_seed.append({
                'seed': seed,
                'curve': {k: round(v, 6) for k, v in curve.items()},
                'shift': [round(s, 6) for s in shift],
                'cosine': round(cos, 6),
            })

            if seed % 5 == 0:
                print(f"    seed {seed:2d}: cosine={cos:.4f}")

        cosines = [r['cosine'] for r in per_seed]
        mean_cos = float(np.mean(cosines))
        std_cos = float(np.std(cosines))
        ci_lower = float(np.percentile(cosines, 2.5))
        ci_upper = float(np.percentile(cosines, 97.5))

        print(f"    {variant_name}: mean_cosine={mean_cos:.4f} "
              f"± {std_cos:.4f}  CI=[{ci_lower:.4f}, {ci_upper:.4f}]")

        results_by_variant[variant_name] = {
            'mean_cosine': mean_cos,
            'std_cosine': std_cos,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'n_seeds': n_seeds,
            'n_tokens_per_seed': n_tokens,
            'per_seed': per_seed,
        }

    # ── 4. Save ──────────────────────────────────────────────────────────
    runtime = round(time.time() - t0, 2)
    output = {
        'phase': '55A.1',
        'experiment': 'schinner_generator',
        'n_chars': n_chars,
        'n_corpus_tokens': len(corpus_tokens),
        'variants': results_by_variant,
        'voynich_shift_reference': voynich_shift,
        'runtime_seconds': runtime,
    }

    out_path = _save_json(rd, 'phase55_schinner_gen.json', output)
    print(f"\n  Saved → {out_path}")
    print(f"  Completed in {runtime:.1f}s")
