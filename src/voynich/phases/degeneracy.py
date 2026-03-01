"""
Workstream D: Break Substitution vs Syllabary Degeneracy
=========================================================
Three independent tests to determine whether Voynich glyphs map to
individual Latin letters (substitution) or to CV syllables (syllabary).

Tests:
  D.1 — Token length correlation (char vs syllable)
  D.2 — Bigram transition structure comparison
  D.3 — Position-within-token entropy profile comparison
"""

import json
import math
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import wasserstein_distance as emd

from voynich.core.corpus import load_corpus, VoynichCorpus, tokenize_eva_chars
from voynich.core.stats import (
    syllabify_latin, syllabify_latin_text,
    bigram_transition_matrix, dtw_distance, frobenius_distance,
    bootstrap_ci, pearson_correlation, first_order_entropy,
)
from voynich.core.reference import (
    load_reference_corpus, get_reference_text,
    get_reference_syllable_stats, ReferenceCorpus,
)
from voynich.core._paths import results_dir as _results_dir


# ---------------------------------------------------------------------------
# D.1: Token Length Correlation Test
# ---------------------------------------------------------------------------

@dataclass
class LengthCorrelationResult:
    """Result of comparing Voynich token lengths vs Latin word lengths."""
    voynich_mean_length: float
    voynich_std_length: float
    latin_char_mean_length: float
    latin_char_std_length: float
    latin_syl_mean_length: float
    latin_syl_std_length: float
    r_voynich_vs_char: float
    r_voynich_vs_char_p: float
    r_voynich_vs_syl: float
    r_voynich_vs_syl_p: float
    r_char_ci: Tuple[float, float]
    r_syl_ci: Tuple[float, float]
    emd_voynich_vs_char: float
    emd_voynich_vs_syl: float
    null_r_char_mean: float
    null_r_syl_mean: float
    verdict: str


def _length_histogram(lengths: List[int], max_len: int = 15) -> np.ndarray:
    """Build normalized frequency histogram of lengths 1..max_len."""
    hist = np.zeros(max_len)
    for l in lengths:
        idx = min(l, max_len) - 1
        hist[idx] += 1
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


def voynich_length_distribution(corpus: VoynichCorpus) -> List[int]:
    """Get list of Voynich token lengths (in EVA glyph count)."""
    tokens = corpus.get_tokens(paragraph_only=True)
    lengths = []
    for t in tokens:
        glyphs = tokenize_eva_chars(t)
        lengths.append(len(glyphs))
    return lengths


def latin_length_distributions(
    text: str,
) -> Tuple[List[int], List[int]]:
    """Get lists of Latin word lengths in chars and syllables."""
    words = text.split()
    char_lengths = [len(w) for w in words]
    syl_lengths = [len(syllabify_latin(w)) for w in words]
    return char_lengths, syl_lengths


def length_correlation_test(
    corpus: VoynichCorpus,
    reference_corpus: Optional[ReferenceCorpus] = None,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> LengthCorrelationResult:
    """
    Compare distribution shapes using histogram correlation and EMD.

    Method: build frequency-weighted length histograms, compare via
    Pearson correlation and Earth Mover's Distance.
    """
    # Load reference if needed
    if reference_corpus is None:
        reference_corpus = load_reference_corpus(verbose=False)

    latin_text = get_reference_text('latin', n_words=5000, seed=seed,
                                    corpus=reference_corpus)

    # Get length distributions
    voynich_lengths = voynich_length_distribution(corpus)
    char_lengths, syl_lengths = latin_length_distributions(latin_text)

    # Build histograms
    max_len = 15
    v_hist = _length_histogram(voynich_lengths, max_len)
    c_hist = _length_histogram(char_lengths, max_len)
    s_hist = _length_histogram(syl_lengths, max_len)

    # Pearson correlations on histograms
    r_char, p_char = pearson_correlation(v_hist, c_hist)
    r_syl, p_syl = pearson_correlation(v_hist, s_hist)

    # Bootstrap CIs on correlations
    rng = np.random.RandomState(seed)

    def _boot_corr_char(data):
        idx = rng.randint(0, len(voynich_lengths), size=len(voynich_lengths))
        boot_lens = [voynich_lengths[i] for i in idx]
        boot_hist = _length_histogram(boot_lens, max_len)
        r, _ = pearson_correlation(boot_hist, c_hist)
        return r

    def _boot_corr_syl(data):
        idx = rng.randint(0, len(voynich_lengths), size=len(voynich_lengths))
        boot_lens = [voynich_lengths[i] for i in idx]
        boot_hist = _length_histogram(boot_lens, max_len)
        r, _ = pearson_correlation(boot_hist, s_hist)
        return r

    boot_chars = [_boot_corr_char(None) for _ in range(n_bootstrap)]
    boot_syls = [_boot_corr_syl(None) for _ in range(n_bootstrap)]
    r_char_ci = (float(np.percentile(boot_chars, 2.5)),
                 float(np.percentile(boot_chars, 97.5)))
    r_syl_ci = (float(np.percentile(boot_syls, 2.5)),
                float(np.percentile(boot_syls, 97.5)))

    # Earth Mover's Distance
    emd_char = float(emd(v_hist, c_hist))
    emd_syl = float(emd(v_hist, s_hist))

    # Null baseline: shuffled Voynich lengths
    null_r_chars = []
    null_r_syls = []
    for i in range(100):
        shuffled = list(voynich_lengths)
        rng.shuffle(shuffled)
        sh_hist = _length_histogram(shuffled, max_len)
        r_c, _ = pearson_correlation(sh_hist, c_hist)
        r_s, _ = pearson_correlation(sh_hist, s_hist)
        null_r_chars.append(r_c)
        null_r_syls.append(r_s)

    # Verdict
    delta = abs(r_syl - r_char)
    if r_syl > r_char and delta > 0.15 and r_syl_ci[0] > r_char_ci[1]:
        verdict = 'syllabary'
    elif r_char > r_syl and delta > 0.15 and r_char_ci[0] > r_syl_ci[1]:
        verdict = 'substitution'
    else:
        verdict = 'inconclusive'

    return LengthCorrelationResult(
        voynich_mean_length=float(np.mean(voynich_lengths)),
        voynich_std_length=float(np.std(voynich_lengths)),
        latin_char_mean_length=float(np.mean(char_lengths)),
        latin_char_std_length=float(np.std(char_lengths)),
        latin_syl_mean_length=float(np.mean(syl_lengths)),
        latin_syl_std_length=float(np.std(syl_lengths)),
        r_voynich_vs_char=r_char,
        r_voynich_vs_char_p=p_char,
        r_voynich_vs_syl=r_syl,
        r_voynich_vs_syl_p=p_syl,
        r_char_ci=r_char_ci,
        r_syl_ci=r_syl_ci,
        emd_voynich_vs_char=emd_char,
        emd_voynich_vs_syl=emd_syl,
        null_r_char_mean=float(np.mean(null_r_chars)),
        null_r_syl_mean=float(np.mean(null_r_syls)),
        verdict=verdict,
    )


# ---------------------------------------------------------------------------
# D.2: Bigram Transition Structure Test
# ---------------------------------------------------------------------------

@dataclass
class BigramStructureResult:
    """Result of bigram matrix comparison under substitution vs syllabary."""
    frobenius_substitution: float
    frobenius_syllabary: float
    jsd_substitution: float
    jsd_syllabary: float
    null_frobenius_sub_mean: float
    null_frobenius_syl_mean: float
    selectivity_sub: float
    selectivity_syl: float
    voynich_alphabet_size: int
    latin_char_alphabet_size: int
    latin_syl_alphabet_size: int
    verdict: str


def optimal_permutation_distance(
    mat_source: np.ndarray,
    mat_target: np.ndarray,
) -> float:
    """
    Find the permutation of mat_target rows/cols that minimizes
    Frobenius distance to mat_source.

    Uses Hungarian algorithm on a row-similarity cost matrix.
    Handles rectangular matrices by padding to square.
    """
    ns = mat_source.shape[0]
    nt = mat_target.shape[0]
    n = max(ns, nt)

    # Pad to square
    padded_s = np.zeros((n, n))
    padded_t = np.zeros((n, n))
    padded_s[:ns, :ns] = mat_source
    padded_t[:nt, :nt] = mat_target

    # Cost matrix: cost[i][j] = ||row_i_source - row_j_target||^2
    cost = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cost[i, j] = np.sum((padded_s[i] - padded_t[j]) ** 2)

    row_ind, col_ind = linear_sum_assignment(cost)

    # Apply permutation and compute Frobenius distance
    perm_target = padded_t[col_ind][:, col_ind]
    dist = frobenius_distance(padded_s, perm_target)

    return dist


def bigram_structure_test(
    corpus: VoynichCorpus,
    reference_corpus: Optional[ReferenceCorpus] = None,
    seed: int = 42,
) -> BigramStructureResult:
    """
    Build and compare bigram matrices under substitution and syllabary models.
    """
    if reference_corpus is None:
        reference_corpus = load_reference_corpus(verbose=False)

    # Voynich character bigram matrix
    tokens = corpus.get_tokens(paragraph_only=True)
    voynich_text = ' '.join(tokens)
    v_mat, v_alph = bigram_transition_matrix(voynich_text)

    # Latin reference stats
    ref_stats = get_reference_syllable_stats('latin', corpus=reference_corpus,
                                             n_words=5000, seed=seed)
    c_mat, c_alph = ref_stats['char_bigrams']
    s_mat, s_alph = ref_stats['syllable_bigrams']

    # Substitution model: Voynich chars <-> Latin chars
    frob_sub = optimal_permutation_distance(v_mat, c_mat)

    # Syllabary model: Voynich chars <-> Latin syllables
    # Use top-N syllables matching Voynich alphabet size for fair comparison
    n_v = len(v_alph)
    if len(s_alph) > n_v:
        # Take top-n_v syllables by row sum (frequency proxy)
        syl_freqs = s_mat.sum(axis=1)
        top_idx = np.argsort(syl_freqs)[-n_v:]
        s_mat_trimmed = s_mat[np.ix_(top_idx, top_idx)]
    else:
        s_mat_trimmed = s_mat

    frob_syl = optimal_permutation_distance(v_mat, s_mat_trimmed)

    # JSD comparison
    from voynich.core.stats import jensen_shannon_divergence
    v_flat = v_mat.flatten() + 1e-10
    v_flat /= v_flat.sum()

    c_flat = c_mat.flatten() + 1e-10
    c_flat /= c_flat.sum()

    # Pad/trim for JSD
    max_flat = max(len(v_flat), len(c_flat))
    v_p = np.zeros(max_flat)
    c_p = np.zeros(max_flat)
    v_p[:len(v_flat)] = v_flat
    c_p[:len(c_flat)] = c_flat
    v_p /= v_p.sum()
    c_p /= c_p.sum()
    jsd_sub = jensen_shannon_divergence(v_p, c_p)

    s_flat = s_mat_trimmed.flatten() + 1e-10
    s_flat /= s_flat.sum()
    s_p = np.zeros(max(len(v_flat), len(s_flat)))
    v_p2 = np.zeros(max(len(v_flat), len(s_flat)))
    s_p[:len(s_flat)] = s_flat
    v_p2[:len(v_flat)] = v_flat
    s_p /= s_p.sum()
    v_p2 /= v_p2.sum()
    jsd_syl = jensen_shannon_divergence(v_p2, s_p)

    # Null baseline: shuffled Voynich text
    rng = random.Random(seed)
    null_frob_subs = []
    null_frob_syls = []
    for _ in range(50):
        shuffled_tokens = list(tokens)
        rng.shuffle(shuffled_tokens)
        sh_text = ' '.join(shuffled_tokens)
        sh_mat, _ = bigram_transition_matrix(sh_text)
        null_frob_subs.append(optimal_permutation_distance(sh_mat, c_mat))
        null_frob_syls.append(optimal_permutation_distance(sh_mat, s_mat_trimmed))

    null_sub_mean = float(np.mean(null_frob_subs))
    null_syl_mean = float(np.mean(null_frob_syls))

    sel_sub = null_sub_mean / frob_sub if frob_sub > 0 else 0
    sel_syl = null_syl_mean / frob_syl if frob_syl > 0 else 0

    # Verdict
    if frob_syl < frob_sub and (frob_sub - frob_syl) / frob_sub > 0.10:
        verdict = 'syllabary'
    elif frob_sub < frob_syl and (frob_syl - frob_sub) / frob_syl > 0.10:
        verdict = 'substitution'
    else:
        verdict = 'inconclusive'

    return BigramStructureResult(
        frobenius_substitution=frob_sub,
        frobenius_syllabary=frob_syl,
        jsd_substitution=jsd_sub,
        jsd_syllabary=jsd_syl,
        null_frobenius_sub_mean=null_sub_mean,
        null_frobenius_syl_mean=null_syl_mean,
        selectivity_sub=sel_sub,
        selectivity_syl=sel_syl,
        voynich_alphabet_size=len(v_alph),
        latin_char_alphabet_size=len(c_alph),
        latin_syl_alphabet_size=len(s_alph),
        verdict=verdict,
    )


# ---------------------------------------------------------------------------
# D.3: Position-Within-Token Entropy Profile Test
# ---------------------------------------------------------------------------

@dataclass
class PositionalEntropyResult:
    """Result of positional entropy curve comparison."""
    voynich_curve: List[float]
    latin_char_curve: List[float]
    latin_syl_curve: List[float]
    dtw_voynich_vs_char: float
    dtw_voynich_vs_syl: float
    null_dtw_char_mean: float
    null_dtw_syl_mean: float
    selectivity_char: float
    selectivity_syl: float
    verdict: str


def positional_entropy_curve(
    items_per_token: List[List[str]],
    max_pos: int = 10,
) -> List[float]:
    """
    Compute H(unit | position=k) for k=0..max_pos-1.

    items_per_token: list of lists, where each inner list is
    the sequence of units (chars or syllables) in one token.
    """
    pos_units: Dict[int, List[str]] = defaultdict(list)
    for units in items_per_token:
        for k, u in enumerate(units):
            if k < max_pos:
                pos_units[k].append(u)

    curve = []
    for pos in range(max_pos):
        if pos not in pos_units or not pos_units[pos]:
            break
        counts = Counter(pos_units[pos])
        total = len(pos_units[pos])
        h = -sum((c / total) * math.log2(c / total)
                 for c in counts.values() if c > 0)
        curve.append(round(h, 4))

    return curve


def positional_entropy_test(
    corpus: VoynichCorpus,
    reference_corpus: Optional[ReferenceCorpus] = None,
    seed: int = 42,
) -> PositionalEntropyResult:
    """
    Compare positional entropy curves using DTW distance.
    """
    if reference_corpus is None:
        reference_corpus = load_reference_corpus(verbose=False)

    max_pos = 10

    # Voynich: glyphs per token
    tokens = corpus.get_tokens(paragraph_only=True)
    voynich_items = [list(tokenize_eva_chars(t)) for t in tokens]
    v_curve = positional_entropy_curve(voynich_items, max_pos)

    # Latin: chars per word and syllables per word
    latin_text = get_reference_text('latin', n_words=5000, seed=seed,
                                    corpus=reference_corpus)
    latin_words = latin_text.split()

    char_items = [list(w) for w in latin_words]
    c_curve = positional_entropy_curve(char_items, max_pos)

    syl_items = syllabify_latin_text(latin_text)
    s_curve = positional_entropy_curve(syl_items, max_pos)

    # DTW distances
    dtw_char = dtw_distance(np.array(v_curve), np.array(c_curve))
    dtw_syl = dtw_distance(np.array(v_curve), np.array(s_curve))

    # Null baseline: shuffled Voynich (shuffle chars within tokens)
    rng = random.Random(seed)
    null_dtw_chars = []
    null_dtw_syls = []
    for _ in range(100):
        shuffled_items = []
        for items in voynich_items:
            shuffled = list(items)
            rng.shuffle(shuffled)
            shuffled_items.append(shuffled)
        sh_curve = positional_entropy_curve(shuffled_items, max_pos)
        sh_arr = np.array(sh_curve)
        null_dtw_chars.append(dtw_distance(sh_arr, np.array(c_curve)))
        null_dtw_syls.append(dtw_distance(sh_arr, np.array(s_curve)))

    null_char_mean = float(np.mean(null_dtw_chars))
    null_syl_mean = float(np.mean(null_dtw_syls))

    sel_char = null_char_mean / dtw_char if dtw_char > 0 else 0
    sel_syl = null_syl_mean / dtw_syl if dtw_syl > 0 else 0

    # Verdict
    delta_pct = abs(dtw_syl - dtw_char) / max(dtw_syl, dtw_char, 1e-10)
    if dtw_syl < dtw_char and delta_pct > 0.15:
        verdict = 'syllabary'
    elif dtw_char < dtw_syl and delta_pct > 0.15:
        verdict = 'substitution'
    else:
        verdict = 'inconclusive'

    return PositionalEntropyResult(
        voynich_curve=v_curve,
        latin_char_curve=c_curve,
        latin_syl_curve=s_curve,
        dtw_voynich_vs_char=dtw_char,
        dtw_voynich_vs_syl=dtw_syl,
        null_dtw_char_mean=null_char_mean,
        null_dtw_syl_mean=null_syl_mean,
        selectivity_char=sel_char,
        selectivity_syl=sel_syl,
        verdict=verdict,
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_degeneracy_analysis() -> Dict:
    """Run all Workstream D tests and print/save results."""
    rd = _results_dir()

    print("=" * 70)
    print("WORKSTREAM D: SUBSTITUTION vs SYLLABARY DEGENERACY")
    print("=" * 70)

    # Load data once
    corpus = load_corpus(verbose=False)
    ref_corpus = load_reference_corpus(verbose=False)

    # D.1: Length Correlation
    print("\n--- D.1: Token Length Correlation Test ---")
    d1 = length_correlation_test(corpus, ref_corpus)
    print(f"  Voynich mean glyph length:  {d1.voynich_mean_length:.2f} "
          f"(std {d1.voynich_std_length:.2f})")
    print(f"  Latin mean char length:     {d1.latin_char_mean_length:.2f} "
          f"(std {d1.latin_char_std_length:.2f})")
    print(f"  Latin mean syllable length: {d1.latin_syl_mean_length:.2f} "
          f"(std {d1.latin_syl_std_length:.2f})")
    print(f"  r(voynich, char):  {d1.r_voynich_vs_char:.4f}  "
          f"CI [{d1.r_char_ci[0]:.4f}, {d1.r_char_ci[1]:.4f}]")
    print(f"  r(voynich, syl):   {d1.r_voynich_vs_syl:.4f}  "
          f"CI [{d1.r_syl_ci[0]:.4f}, {d1.r_syl_ci[1]:.4f}]")
    print(f"  EMD(voynich, char): {d1.emd_voynich_vs_char:.4f}")
    print(f"  EMD(voynich, syl):  {d1.emd_voynich_vs_syl:.4f}")
    print(f"  Null r(char) mean:  {d1.null_r_char_mean:.4f}")
    print(f"  Null r(syl) mean:   {d1.null_r_syl_mean:.4f}")
    print(f"  >> D.1 VERDICT: {d1.verdict}")

    with open(os.path.join(rd, 'degeneracy_length.json'), 'w') as f:
        json.dump(asdict(d1), f, indent=2)

    # D.2: Bigram Structure
    print("\n--- D.2: Bigram Transition Structure Test ---")
    d2 = bigram_structure_test(corpus, ref_corpus)
    print(f"  Voynich alphabet size:     {d2.voynich_alphabet_size}")
    print(f"  Latin char alphabet size:  {d2.latin_char_alphabet_size}")
    print(f"  Latin syllable types:      {d2.latin_syl_alphabet_size}")
    print(f"  Frobenius (substitution):  {d2.frobenius_substitution:.4f}")
    print(f"  Frobenius (syllabary):     {d2.frobenius_syllabary:.4f}")
    print(f"  JSD (substitution):        {d2.jsd_substitution:.4f}")
    print(f"  JSD (syllabary):           {d2.jsd_syllabary:.4f}")
    print(f"  Null Frobenius sub mean:   {d2.null_frobenius_sub_mean:.4f}")
    print(f"  Null Frobenius syl mean:   {d2.null_frobenius_syl_mean:.4f}")
    print(f"  Selectivity sub:           {d2.selectivity_sub:.2f}x")
    print(f"  Selectivity syl:           {d2.selectivity_syl:.2f}x")
    print(f"  >> D.2 VERDICT: {d2.verdict}")

    with open(os.path.join(rd, 'degeneracy_bigram.json'), 'w') as f:
        json.dump(asdict(d2), f, indent=2)

    # D.3: Positional Entropy
    print("\n--- D.3: Position-Within-Token Entropy Profile ---")
    d3 = positional_entropy_test(corpus, ref_corpus)
    print(f"  Voynich curve ({len(d3.voynich_curve)} positions):")
    print(f"    {[f'{v:.2f}' for v in d3.voynich_curve]}")
    print(f"  Latin char curve ({len(d3.latin_char_curve)} positions):")
    print(f"    {[f'{v:.2f}' for v in d3.latin_char_curve]}")
    print(f"  Latin syl curve ({len(d3.latin_syl_curve)} positions):")
    print(f"    {[f'{v:.2f}' for v in d3.latin_syl_curve]}")
    print(f"  DTW(voynich, char): {d3.dtw_voynich_vs_char:.4f}")
    print(f"  DTW(voynich, syl):  {d3.dtw_voynich_vs_syl:.4f}")
    print(f"  Null DTW char mean: {d3.null_dtw_char_mean:.4f}")
    print(f"  Null DTW syl mean:  {d3.null_dtw_syl_mean:.4f}")
    print(f"  Selectivity char:   {d3.selectivity_char:.2f}x")
    print(f"  Selectivity syl:    {d3.selectivity_syl:.2f}x")
    print(f"  >> D.3 VERDICT: {d3.verdict}")

    with open(os.path.join(rd, 'degeneracy_positional.json'), 'w') as f:
        json.dump(asdict(d3), f, indent=2)

    # Overall verdict
    verdicts = [d1.verdict, d2.verdict, d3.verdict]
    syl_count = verdicts.count('syllabary')
    sub_count = verdicts.count('substitution')
    if syl_count >= 2:
        overall = 'syllabary'
    elif sub_count >= 2:
        overall = 'substitution'
    else:
        overall = 'inconclusive'

    print(f"\n{'=' * 70}")
    print(f"OVERALL DEGENERACY VERDICT: {overall}")
    print(f"  D.1={d1.verdict}, D.2={d2.verdict}, D.3={d3.verdict}")
    print(f"{'=' * 70}")

    verdict_data = {
        'd1_verdict': d1.verdict,
        'd2_verdict': d2.verdict,
        'd3_verdict': d3.verdict,
        'overall_verdict': overall,
    }
    with open(os.path.join(rd, 'degeneracy_verdict.json'), 'w') as f:
        json.dump(verdict_data, f, indent=2)

    return {
        'd1': asdict(d1),
        'd2': asdict(d2),
        'd3': asdict(d3),
        'verdict': verdict_data,
    }
