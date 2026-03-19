"""
Phase 55A.2 – Rugg-Taylor Cardan Grille Generator
==================================================
Reconstructs Rugg (2004) / Rugg-Taylor (2017)'s Cardan grille method and
measures its entropy shift cosine vs the Voynich target shift.

Dependency chain:
    results/entropy_shift_cipher.json   (latin baseline + observed shift)
    corpus (Voynich EVA tokens)
        → results/phase55_cardan_gen.json
"""

import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import load_corpus
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
# Syllable table construction
# ---------------------------------------------------------------------------

def build_syllable_table(
    corpus_tokens: List[str],
    n_rows: int = 6,
    n_cols: int = 6,
) -> List[List[str]]:
    """
    Build a Rugg-style syllable table from Voynich token character group frequencies.

    Column 0: common word-initial character groups (1-2 chars)
    Columns 1-4: common word-medial groups (1-3 chars)
    Column 5: common word-final character groups (1-2 chars)
    """
    initial_groups: Dict[str, int] = {}
    medial_groups: Dict[str, int] = {}
    final_groups: Dict[str, int] = {}

    for token in corpus_tokens:
        if len(token) < 2:
            continue

        # Initial: first 1-2 chars
        for k in [1, 2]:
            if k <= len(token):
                g = token[:k]
                initial_groups[g] = initial_groups.get(g, 0) + 1

        # Final: last 1-2 chars
        for k in [1, 2]:
            if k <= len(token):
                g = token[-k:]
                final_groups[g] = final_groups.get(g, 0) + 1

        # Medial: 1-3 char substrings from middle region
        if len(token) > 2:
            mid = token[1:-1]
            for k in [1, 2, 3]:
                for i in range(len(mid) - k + 1):
                    g = mid[i:i + k]
                    medial_groups[g] = medial_groups.get(g, 0) + 1

    def top_n(groups: Dict[str, int], n: int) -> List[str]:
        return [g for g, _ in sorted(groups.items(), key=lambda x: -x[1])[:n]]

    n_medial_cells = n_rows * (n_cols - 2)
    top_initial = top_n(initial_groups, n_rows)
    top_medial = top_n(medial_groups, n_medial_cells)
    top_final = top_n(final_groups, n_rows)

    # Populate table
    table: List[List[str]] = [['' for _ in range(n_cols)] for _ in range(n_rows)]
    for r in range(n_rows):
        table[r][0] = top_initial[r] if r < len(top_initial) else ''
        table[r][n_cols - 1] = top_final[r] if r < len(top_final) else ''
        for c in range(1, n_cols - 1):
            idx = r * (n_cols - 2) + (c - 1)
            table[r][c] = top_medial[idx] if idx < len(top_medial) else ''

    return table


# ---------------------------------------------------------------------------
# Grille generation and text generation
# ---------------------------------------------------------------------------

def generate_grille(
    n_rows: int, n_cols: int, n_holes: int, rng: np.random.Generator
) -> List[Tuple[int, int]]:
    """Generate a random Cardan grille with n_holes visible positions."""
    all_positions = [(r, c) for r in range(n_rows) for c in range(n_cols)]
    n_total = len(all_positions)
    n_select = min(n_holes, n_total)
    selected_idx = rng.choice(n_total, size=n_select, replace=False)
    selected = sorted([all_positions[i] for i in selected_idx])
    return selected


def generate_cardan_text(
    table: List[List[str]],
    n_tokens: int,
    n_holes: int,
    n_grilles: int,
    rng: np.random.Generator,
) -> List[str]:
    """
    Generate tokens using Cardan grille method.

    For each token: pick a random grille from the pre-generated set,
    read visible cells left-to-right top-to-bottom, concatenate.
    """
    n_rows = len(table)
    n_cols = len(table[0]) if table else 0

    # Pre-generate grille set
    grilles = [generate_grille(n_rows, n_cols, n_holes, rng) for _ in range(n_grilles)]

    tokens = []
    grille_indices = rng.integers(len(grilles), size=n_tokens)

    for g_idx in grille_indices:
        grille = grilles[int(g_idx)]
        parts = [table[r][c] for r, c in grille if table[r][c]]
        if parts:
            tokens.append(''.join(parts))

    return tokens


# ---------------------------------------------------------------------------
# Entropy computation helpers
# ---------------------------------------------------------------------------

def _compute_entropy_curve_for_tokens(
    tokens: List[str], max_order: int = 6
) -> Dict[int, float]:
    text = ' '.join(tokens)
    return entropy_curve(text, max_order=max_order)


def _compute_cosine_vs_shift(
    curve: Dict[int, float],
    latin_curve: Dict[int, float],
    voynich_shift: List[float],
    orders: List[int],
) -> Tuple[List[float], float]:
    shift = [curve.get(k, 0.0) - latin_curve.get(k, 0.0) for k in orders]
    shift_arr = np.array(shift)
    voynich_arr = np.array(voynich_shift)
    cos = float(cosine_similarity(shift_arr, voynich_arr))
    return shift, cos


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_cardan_gen() -> None:
    """Phase 55A.2: Rugg-Taylor Cardan grille generator entropy shift."""
    t0 = time.time()
    rd = str(_results_dir())

    print("=" * 70)
    print("PHASE 55A.2: Rugg-Taylor Cardan Grille Generator")
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
    latin_curve = {int(k): float(v) for k, v in latin_curve_raw.items()}

    orders = list(range(7))  # H0–H6

    # ── 2. Load Voynich corpus and build syllable table ──────────────────
    print("\n  2. Building Rugg syllable table from Voynich tokens …")

    corpus = load_corpus(verbose=False)
    voynich_text = corpus.get_text()
    corpus_tokens = [t for t in voynich_text.split() if t]

    table = build_syllable_table(corpus_tokens)

    n_rows = len(table)
    n_cols = len(table[0]) if table else 0
    print(f"    Corpus tokens: {len(corpus_tokens):,}")
    print(f"    Table: {n_rows}×{n_cols}")
    print(f"    Sample cells: {table[0][:3]} … {table[-1][-3:]}")

    # ── 3. Generate 2 variants × 20 seeds ───────────────────────────────
    n_seeds = 20
    n_tokens = 36_000
    n_grilles = 10
    max_order = 6

    variants = [
        ('cardan_3hole', 3),
        ('cardan_4hole', 4),
    ]

    results_by_variant = {}

    for variant_name, n_holes in variants:
        print(f"\n  3. Generating {variant_name} ({n_seeds} seeds × {n_tokens:,} tokens) …")

        per_seed = []
        for seed in range(n_seeds):
            rng = np.random.default_rng(seed + 100)  # offset to differ from Schinner seeds
            tokens = generate_cardan_text(
                table, n_tokens, n_holes=n_holes, n_grilles=n_grilles, rng=rng
            )

            # Filter any empty tokens
            tokens = [t for t in tokens if t]
            if not tokens:
                tokens = ['a']  # fallback

            curve = _compute_entropy_curve_for_tokens(tokens, max_order=max_order)
            shift, cos = _compute_cosine_vs_shift(curve, latin_curve, voynich_shift, orders)

            per_seed.append({
                'seed': seed,
                'n_tokens_generated': len(tokens),
                'curve': {k: round(v, 6) for k, v in curve.items()},
                'shift': [round(s, 6) for s in shift],
                'cosine': round(cos, 6),
            })

            if seed % 5 == 0:
                print(f"    seed {seed:2d}: cosine={cos:.4f}, tokens={len(tokens):,}")

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
            'n_holes': n_holes,
            'n_grilles': n_grilles,
            'n_tokens_per_seed': n_tokens,
            'per_seed': per_seed,
        }

    # ── 4. Save ──────────────────────────────────────────────────────────
    runtime = round(time.time() - t0, 2)
    output = {
        'phase': '55A.2',
        'experiment': 'cardan_generator',
        'table': table,
        'table_shape': [n_rows, n_cols],
        'n_corpus_tokens': len(corpus_tokens),
        'variants': results_by_variant,
        'voynich_shift_reference': voynich_shift,
        'runtime_seconds': runtime,
    }

    out_path = _save_json(rd, 'phase55_cardan_gen.json', output)
    print(f"\n  Saved → {out_path}")
    print(f"  Completed in {runtime:.1f}s")
