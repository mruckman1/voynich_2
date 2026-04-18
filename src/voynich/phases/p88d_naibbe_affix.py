"""
Phase 88d — Naibbe at affix (sub-token) granularity.

Motivation: Phase 88c measured the tachygraphic at both syllable and word
granularities. For symmetry and to close the ``what about finer
granularity'' loophole, measure Naibbe at its sub-token unit (the
prefix/suffix affix produced by one half of a bigram encoding).
Cross-boundary MI is computed between the suffix of Naibbe token N (the
last affix) and the prefix of Naibbe token N+1 (the first affix) --
i.e., only across original Naibbe token boundaries, mirroring how
tachygraphic cross-boundary MI measures across word boundaries at
syllable granularity.

Dependency chain:
    data/reference/greshko/nathist_book16.txt
    src/voynich/phases/p88_naibbe_generalized.py  (slot grammar, tables, helpers)
        -> results/p88d_naibbe_affix.json
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

from voynich.core._paths import results_dir as _results_dir
from voynich.phases.p88_naibbe_generalized import (
    _build_cumulative_weights,
    _convert,
    _freq_conn_rho_plain,
    _load_latin_file,
    make_grammar_targeting_length,
    make_tables,
    N_TABLES,
    OUTPUT_ALPHA_SIZE,
    TABLE_WEIGHTS,
)


def _encode_with_affix_split(
    plaintext: str,
    tables: List[Tuple[Dict, Dict]],
    rng: random.Random,
    cum_weights: List[float],
) -> Tuple[List[str], List[str], List[str]]:
    """Encode and return (prefixes, suffixes, joined_tokens) in parallel."""
    chars = [c for c in plaintext.lower() if c.isalpha()]
    if len(chars) % 2 == 1:
        chars.append('a')

    n_tables = len(tables)
    prefixes, suffixes, tokens = [], [], []
    for i in range(0, len(chars), 2):
        c1, c2 = chars[i], chars[i + 1]
        r = rng.random()
        t_idx = min(
            next(j for j, cw in enumerate(cum_weights) if r <= cw),
            n_tables - 1,
        )
        p = tables[t_idx][0][c1]
        s = tables[t_idx][1][c2]
        prefixes.append(p)
        suffixes.append(s)
        tokens.append(p + s)
    return prefixes, suffixes, tokens


def _mi_across_boundaries(prefixes: List[str],
                          suffixes: List[str]) -> Dict[str, float]:
    """Cross-boundary MI: last char of suffix[i] vs first char of prefix[i+1]."""
    pairs = []
    for i in range(len(suffixes) - 1):
        s, p = suffixes[i], prefixes[i + 1]
        if s and p:
            pairs.append((s[-1], p[0]))

    n = len(pairs)
    if n == 0:
        return {'ratio': 1.0, 'mi': 0.0, 'n_pairs': 0}

    joint = Counter(pairs)
    last_counts = Counter(pa[0] for pa in pairs)
    first_counts = Counter(pa[1] for pa in pairs)

    mi = 0.0
    for (ll, ff), c in joint.items():
        p_joint = c / n
        p_l = last_counts[ll] / n
        p_f = first_counts[ff] / n
        if p_joint > 0 and p_l > 0 and p_f > 0:
            mi += p_joint * math.log2(p_joint / (p_l * p_f))

    weighted_ratio = 0.0
    for (ll, ff), c in joint.items():
        p_fl = c / last_counts[ll]
        p_f = first_counts[ff] / n
        if p_f > 0:
            weighted_ratio += c * (p_fl / p_f)
    weighted_ratio /= n

    return {
        'ratio': round(float(weighted_ratio), 6),
        'mi': round(float(mi), 6),
        'n_pairs': n,
    }


@dataclass
class AffixResult:
    n_seeds: int
    mean_mi_ratio: float
    std_mi_ratio: float
    mean_rho: float
    std_rho: float
    mean_n_pairs: float
    mean_n_affix_types: float


@dataclass
class P88dResult:
    timestamp: str
    runtime_seconds: float
    n_seeds: int
    config: str
    latin_source: str
    affix_result: AffixResult
    token_result: AffixResult
    voynich_reference: Dict[str, float]
    tachy_syllable_reference: Dict[str, float]
    tachy_word_reference: Dict[str, float]
    comparison_table: List[Dict[str, Any]]
    narrative: str


def run_p88d() -> None:
    t0 = time.time()
    rd = str(_results_dir())

    print("=" * 70)
    print("Phase 88d: Naibbe at Affix (Sub-Token) Granularity")
    print("=" * 70)

    latin_path = os.path.join('data', 'reference', 'greshko', 'nathist_book16.txt')
    latin_text = _load_latin_file(latin_path)

    cum_weights = _build_cumulative_weights(TABLE_WEIGHTS)

    n_seeds = 20
    affix_mi, affix_rho, affix_npairs, affix_ntypes = [], [], [], []
    token_mi, token_rho = [], []

    for i in range(n_seeds):
        seed = i * 997 + 13
        rng = random.Random(seed)
        out_alpha = sorted(rng.sample(
            list('abcdefghijklmnopqrstuvwxyz'), OUTPUT_ALPHA_SIZE))
        half = OUTPUT_ALPHA_SIZE // 2
        pre_slots, _, _, _ = make_grammar_targeting_length(out_alpha[:half], rng)
        suf_slots, _, _, _ = make_grammar_targeting_length(out_alpha[half:], rng)
        tables = make_tables(pre_slots, suf_slots, N_TABLES, rng)

        prefixes, suffixes, tokens = _encode_with_affix_split(
            latin_text, tables, rng, cum_weights)

        # Across-boundary MI at affix granularity
        mi = _mi_across_boundaries(prefixes, suffixes)
        affix_mi.append(mi['ratio'])
        affix_npairs.append(mi['n_pairs'])

        # Freq-conn on the affix vocabulary
        fc = _freq_conn_rho_plain(prefixes + suffixes, max_types=2000)
        affix_rho.append(fc['rho'])
        affix_ntypes.append(fc['n_types'])

        # Token-level for sanity (should reproduce Phase 88)
        from voynich.phases.p88_naibbe_generalized import _cross_boundary_ratio_plain
        mi_t = _cross_boundary_ratio_plain(tokens)
        fc_t = _freq_conn_rho_plain(tokens, max_types=2000)
        token_mi.append(mi_t['ratio'])
        token_rho.append(fc_t['rho'])

        if i % 5 == 0:
            print(f"  seed {i:2d}: affix MI={mi['ratio']:.4f} rho={fc['rho']:+.4f}  "
                  f"token MI={mi_t['ratio']:.4f} rho={fc_t['rho']:+.4f}")

    affix = AffixResult(
        n_seeds=n_seeds,
        mean_mi_ratio=round(float(np.mean(affix_mi)), 4),
        std_mi_ratio=round(float(np.std(affix_mi)), 4),
        mean_rho=round(float(np.mean(affix_rho)), 4),
        std_rho=round(float(np.std(affix_rho)), 4),
        mean_n_pairs=round(float(np.mean(affix_npairs)), 1),
        mean_n_affix_types=round(float(np.mean(affix_ntypes)), 1),
    )
    token = AffixResult(
        n_seeds=n_seeds,
        mean_mi_ratio=round(float(np.mean(token_mi)), 4),
        std_mi_ratio=round(float(np.std(token_mi)), 4),
        mean_rho=round(float(np.mean(token_rho)), 4),
        std_rho=round(float(np.std(token_rho)), 4),
        mean_n_pairs=0.0,
        mean_n_affix_types=0.0,
    )

    voy_ref = {'mi': 1.448, 'rho': 0.615, 'granularity': 'token (= syllable by hypothesis)'}
    tachy_syl = {'mi': 1.285, 'rho': 0.585, 'granularity': 'syllable'}
    tachy_word = {'mi': 1.061, 'rho': 0.235, 'granularity': 'word'}

    comparison = [
        {'system': 'Voynich', 'granularity': voy_ref['granularity'], 'mi': voy_ref['mi'], 'rho': voy_ref['rho']},
        {'system': 'Tachygraphic', 'granularity': tachy_syl['granularity'], 'mi': tachy_syl['mi'], 'rho': tachy_syl['rho']},
        {'system': 'Tachygraphic', 'granularity': tachy_word['granularity'], 'mi': tachy_word['mi'], 'rho': tachy_word['rho']},
        {'system': 'Naibbe', 'granularity': 'affix (across original token boundary)', 'mi': affix.mean_mi_ratio, 'rho': affix.mean_rho},
        {'system': 'Naibbe', 'granularity': 'token', 'mi': token.mean_mi_ratio, 'rho': token.mean_rho},
    ]

    narrative = (
        f"Naibbe fails both token-adjacency diagnostics at every granularity "
        f"at which the measurement is well-defined. At token granularity "
        f"(each token = one plaintext bigram): MI = {token.mean_mi_ratio:.4f}, "
        f"rho = {token.mean_rho:+.4f}. At affix sub-token granularity "
        f"(each affix = one plaintext letter; MI measured between the suffix "
        f"of token N and the prefix of token N+1, i.e. across original "
        f"Naibbe token boundaries): MI = {affix.mean_mi_ratio:.4f}, "
        f"rho = {affix.mean_rho:+.4f}. The affix-level rho is slightly higher "
        f"than token-level because the smaller affix vocabulary produces more "
        f"accidental edit-distance-1 collisions, but remains well below the "
        f"tachygraphic simulation's rho = +0.585 at syllable granularity and "
        f"the Voynich's observed +0.615. The cross-boundary MI stays at the "
        f"1.0 null by construction: Naibbe selects a table independently for "
        f"each bigram, so the suffix of token N carries no information about "
        f"the prefix of token N+1."
    )

    result = P88dResult(
        timestamp=time.strftime('%Y-%m-%dT%H:%M:%S'),
        runtime_seconds=round(time.time() - t0, 2),
        n_seeds=n_seeds,
        config='greshko_defaults (N_TABLES=6, weights=5:2:2:2:1:1, alpha=20, affix=2-3)',
        latin_source=latin_path,
        affix_result=affix,
        token_result=token,
        voynich_reference=voy_ref,
        tachy_syllable_reference=tachy_syl,
        tachy_word_reference=tachy_word,
        comparison_table=comparison,
        narrative=narrative,
    )

    out_path = os.path.join(rd, 'p88d_naibbe_affix.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2)

    print("\n" + "=" * 70)
    print("Final four-cell comparison")
    print("=" * 70)
    print(f"{'System':<14s} {'Granularity':<42s} {'MI':>8s} {'ρ':>8s}")
    print("-" * 74)
    for row in comparison:
        print(f"{row['system']:<14s} {row['granularity']:<42s} "
              f"{row['mi']:>8.4f} {row['rho']:>+8.4f}")

    print(f"\n  -> {out_path}")
    print(f"  Runtime: {result.runtime_seconds:.1f}s")
