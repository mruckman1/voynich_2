"""
Phase 4.5 Priorities A + C: Language A Isolation and qo- Token Removal
=======================================================================
Isolate Currier Language A (herbal sections) from Language B (biological,
astronomical, recipe sections) and rerun core analyses on each independently.
Also profile and remove qo- prefix tokens to test their effect on metrics.

Rationale:
  Finding #2: Language B has H₂ = 0.74 driven by ~13 core tokens, behaving
  like a notation system rather than natural language cipher. Including it
  contaminates every metric.

  Finding #5: qo- tokens cluster at paragraph ends (48.9% in Q4), are
  asymmetric between A and B, and may be functional markers.

Sub-analyses:
  A.1-A.4 — Language A/B split, per-language profiling, metric comparison
  A.5     — Language B standalone characterization
  C.1-C.3 — qo- token profiling, removal impact, grid cell diagnostics

Output:
  results/language_a_isolation.json
"""

import json
import math
import os
import random
from collections import Counter
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from voynich.core.corpus import (
    load_corpus, VoynichCorpus, VOYNICH_SECTIONS,
    tokenize_eva_chars,
)
from voynich.core.stats import (
    first_order_entropy, conditional_entropy,
    word_unigram_entropy,
    bigram_transition_matrix,
    jensen_shannon_divergence,
)
from voynich.core._paths import results_dir as _results_dir
from voynich.analysis.fingerprint import compute_profile, EntropyProfile
from voynich.analysis.strokes import (
    build_ventris_grid, SyllabaryGrid,
    syllable_sequence_stats,
)
from voynich.phases.grid_validate import build_grid_from_tokens, bootstrap_grid_stability
from voynich.phases.abugida_test import (
    decompose_tokens_onset_nucleus, compute_onset_nucleus_entropy,
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class LanguageProfile:
    """Complete analysis profile for one Currier language variant."""
    language: str
    n_folios: int
    n_tokens: int
    n_types: int
    sections: List[str]
    h1: float
    h2: float
    h3: float
    word_h1: float
    type_token_ratio: float
    grid_n_rows: int
    grid_n_cols: int
    grid_occupancy: float
    abugida_r: float
    abugida_reverse_r: float
    abugida_mi: float
    top_20_tokens: List[Tuple[str, int]]


@dataclass
class LanguageComparisonResult:
    """Comparison between Language A and B."""
    profile_a: LanguageProfile
    profile_b: LanguageProfile
    bigram_jsd: float
    grid_jaccard: float
    h2_difference: float
    h2_diff_ci: Tuple[float, float]
    h2_diff_significant: bool
    vocabulary_overlap: float
    null_jsd_mean: float
    null_jsd_std: float
    jsd_z_score: float
    verdict: str


@dataclass
class QoAnalysisResult:
    """Profile and removal impact of qo- tokens."""
    n_qo_tokens: int
    n_qo_types: int
    pct_corpus_qo: float
    pct_lang_a_qo: float
    pct_lang_b_qo: float
    qo_type_examples: List[str]
    # Grid cell clustering
    qo_grid_cells: Dict[str, int]
    qo_clustered: bool
    qo_top_cell_concentration: float
    # Removal effect (on full corpus)
    h1_with: float
    h1_without: float
    h2_with: float
    h2_without: float
    word_h1_with: float
    word_h1_without: float
    grid_occupancy_with: float
    grid_occupancy_without: float
    grid_jaccard_with_vs_without: float
    verdict: str


# ---------------------------------------------------------------------------
# A.1-A.2: Language profiling
# ---------------------------------------------------------------------------

def _build_language_profile(
    corpus: VoynichCorpus,
    language: str,
) -> Tuple[LanguageProfile, List[str], str]:
    """
    Build a complete analysis profile for one Currier language.

    Returns:
        (profile, tokens, text) — profile plus raw data for further analysis.
    """
    text = corpus.get_text(language=language, paragraph_only=True)
    tokens = text.split()
    pages = corpus.get_pages_by_language(language)
    sections = sorted(set(p.section for p in pages))

    if not tokens:
        raise ValueError(f"No tokens found for language '{language}'")

    # Entropy
    h1 = first_order_entropy(text)
    h2 = conditional_entropy(text, order=2)
    h3 = conditional_entropy(text, order=3)
    word_h1 = word_unigram_entropy(tokens)

    # Type-token ratio
    n_types = len(set(tokens))
    ttr = n_types / len(tokens)

    # Grid
    grid = build_grid_from_tokens(tokens)

    # Abugida decomposition
    pairs = decompose_tokens_onset_nucleus(tokens)
    ent = compute_onset_nucleus_entropy(pairs)

    # Top tokens
    counter = Counter(tokens)
    top_20 = counter.most_common(20)

    profile = LanguageProfile(
        language=language,
        n_folios=len(pages),
        n_tokens=len(tokens),
        n_types=n_types,
        sections=sections,
        h1=round(h1, 4),
        h2=round(h2, 4),
        h3=round(h3, 4),
        word_h1=round(word_h1, 4),
        type_token_ratio=round(ttr, 4),
        grid_n_rows=len(grid.row_labels),
        grid_n_cols=len(grid.col_labels),
        grid_occupancy=round(grid.occupancy, 4),
        abugida_r=ent.reduction_r,
        abugida_reverse_r=ent.reverse_r,
        abugida_mi=ent.mi_onset_nucleus,
        top_20_tokens=top_20,
    )
    return profile, tokens, text


# ---------------------------------------------------------------------------
# A.3: Language comparison
# ---------------------------------------------------------------------------

def _grid_jaccard(tokens_a: List[str], tokens_b: List[str]) -> float:
    """Jaccard similarity between grid cells built from two token sets."""
    grid_a = build_grid_from_tokens(tokens_a)
    grid_b = build_grid_from_tokens(tokens_b)
    cells_a = set(grid_a.cells.keys())
    cells_b = set(grid_b.cells.keys())
    if not cells_a and not cells_b:
        return 0.0
    return len(cells_a & cells_b) / len(cells_a | cells_b)


def _bigram_jsd(text_a: str, text_b: str) -> float:
    """JSD between character bigram transition matrices."""
    mat_a, labels_a = bigram_transition_matrix(text_a)
    mat_b, labels_b = bigram_transition_matrix(text_b)

    # Align to common label set
    all_labels = sorted(set(labels_a) | set(labels_b))
    n = len(all_labels)
    label_idx_a = {l: i for i, l in enumerate(labels_a)}
    label_idx_b = {l: i for i, l in enumerate(labels_b)}

    aligned_a = np.zeros((n, n))
    aligned_b = np.zeros((n, n))
    for i, la in enumerate(all_labels):
        for j, lb in enumerate(all_labels):
            if la in label_idx_a and lb in label_idx_a:
                aligned_a[i, j] = mat_a[label_idx_a[la], label_idx_a[lb]]
            if la in label_idx_b and lb in label_idx_b:
                aligned_b[i, j] = mat_b[label_idx_b[la], label_idx_b[lb]]

    # Flatten and normalize to probability distributions
    flat_a = aligned_a.flatten()
    flat_b = aligned_b.flatten()
    sum_a = flat_a.sum()
    sum_b = flat_b.sum()
    if sum_a > 0:
        flat_a = flat_a / sum_a
    if sum_b > 0:
        flat_b = flat_b / sum_b

    return jensen_shannon_divergence(flat_a, flat_b)


def _bootstrap_h2_diff(
    tokens_a: List[str],
    tokens_b: List[str],
    n_bootstrap: int = 500,
    seed: int = 42,
) -> Tuple[float, float]:
    """Bootstrap 95% CI on H₂(A) - H₂(B)."""
    rng = random.Random(seed)
    diffs = []
    for _ in range(n_bootstrap):
        sample_a = rng.choices(tokens_a, k=len(tokens_a))
        sample_b = rng.choices(tokens_b, k=len(tokens_b))
        text_a = ' '.join(sample_a)
        text_b = ' '.join(sample_b)
        h2_a = conditional_entropy(text_a, order=2)
        h2_b = conditional_entropy(text_b, order=2)
        diffs.append(h2_a - h2_b)
    arr = np.array(diffs)
    return (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))


def compare_languages(
    profile_a: LanguageProfile,
    profile_b: LanguageProfile,
    tokens_a: List[str],
    tokens_b: List[str],
    text_a: str,
    text_b: str,
) -> LanguageComparisonResult:
    """Compare Language A and B across multiple metrics."""
    # Bigram JSD
    jsd = _bigram_jsd(text_a, text_b)

    # Grid Jaccard
    jaccard = _grid_jaccard(tokens_a, tokens_b)

    # H2 difference with bootstrap CI
    h2_diff = profile_a.h2 - profile_b.h2
    ci = _bootstrap_h2_diff(tokens_a, tokens_b, n_bootstrap=500)
    h2_significant = not (ci[0] <= 0 <= ci[1])

    # Vocabulary overlap
    vocab_a = set(tokens_a)
    vocab_b = set(tokens_b)
    overlap = len(vocab_a & vocab_b) / len(vocab_a | vocab_b) if (vocab_a | vocab_b) else 0.0

    # Null test: random split
    null_mean, null_std, z = _null_test_language_split(tokens_a, tokens_b, jsd)

    # Verdict
    if jaccard < 0.5 and h2_significant:
        verdict = 'distinct_systems'
    elif jaccard > 0.8 and not h2_significant:
        verdict = 'same_system'
    else:
        verdict = 'inconclusive'

    return LanguageComparisonResult(
        profile_a=profile_a,
        profile_b=profile_b,
        bigram_jsd=round(jsd, 6),
        grid_jaccard=round(jaccard, 4),
        h2_difference=round(h2_diff, 4),
        h2_diff_ci=(round(ci[0], 4), round(ci[1], 4)),
        h2_diff_significant=h2_significant,
        vocabulary_overlap=round(overlap, 4),
        null_jsd_mean=round(null_mean, 6),
        null_jsd_std=round(null_std, 6),
        jsd_z_score=round(z, 2),
        verdict=verdict,
    )


def _null_test_language_split(
    tokens_a: List[str],
    tokens_b: List[str],
    real_jsd: float,
    n_trials: int = 100,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Null test: shuffle all tokens into two random pools, compute bigram JSD.
    If the A/B split is meaningful, real JSD should exceed null.
    """
    rng = random.Random(seed)
    all_tokens = tokens_a + tokens_b
    n_a = len(tokens_a)
    null_jsds = []

    for _ in range(n_trials):
        shuffled = list(all_tokens)
        rng.shuffle(shuffled)
        pool_a = shuffled[:n_a]
        pool_b = shuffled[n_a:]
        text_a = ' '.join(pool_a)
        text_b = ' '.join(pool_b)
        null_jsds.append(_bigram_jsd(text_a, text_b))

    arr = np.array(null_jsds)
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    z = (real_jsd - mean) / std if std > 0 else 0.0
    return mean, std, z


# ---------------------------------------------------------------------------
# C.1-C.3: qo- token analysis
# ---------------------------------------------------------------------------

def _identify_qo_tokens(tokens: List[str]) -> Tuple[List[str], List[str]]:
    """
    Split tokens into qo-prefixed and non-qo tokens.
    A token is qo-prefixed if its EVA decomposition starts with 'qo', 'qok', or 'qot'.
    """
    qo_tokens = []
    non_qo_tokens = []
    qo_prefixes = {'qo', 'qok', 'qot'}

    for token in tokens:
        eva_chars = tokenize_eva_chars(token)
        if eva_chars and eva_chars[0] in qo_prefixes:
            qo_tokens.append(token)
        else:
            non_qo_tokens.append(token)

    return qo_tokens, non_qo_tokens


def _map_tokens_to_grid_cells(tokens: List[str], grid: SyllabaryGrid) -> Dict[str, int]:
    """Count how many tokens map to each grid cell."""
    from voynich.analysis.strokes import segment_token_as_syllables
    cell_counts: Counter = Counter()
    for token in tokens:
        syls = segment_token_as_syllables(token, grid)
        for syl in syls:
            cell_counts[syl] += 1
    return dict(cell_counts)


def analyze_qo_removal(
    corpus: VoynichCorpus,
    language: Optional[str] = None,
) -> QoAnalysisResult:
    """
    Profile qo- tokens and measure the effect of removing them.
    """
    # Get tokens (optionally filtered by language)
    all_tokens = corpus.get_tokens(language=language, paragraph_only=True)
    all_text = ' '.join(all_tokens)
    qo_tokens, non_qo_tokens = _identify_qo_tokens(all_tokens)

    # Per-language qo fractions
    lang_a_tokens = corpus.get_tokens(language='A', paragraph_only=True)
    lang_b_tokens = corpus.get_tokens(language='B', paragraph_only=True)
    qo_a, _ = _identify_qo_tokens(lang_a_tokens)
    qo_b, _ = _identify_qo_tokens(lang_b_tokens)
    pct_a = len(qo_a) / len(lang_a_tokens) * 100 if lang_a_tokens else 0
    pct_b = len(qo_b) / len(lang_b_tokens) * 100 if lang_b_tokens else 0

    # qo- types
    qo_types = sorted(set(qo_tokens))

    # Grid cell mapping for qo- tokens
    grid_with = build_grid_from_tokens(all_tokens)
    qo_cell_counts = _map_tokens_to_grid_cells(qo_tokens, grid_with)

    # Check clustering: do top 3 cells account for > 60% of qo- tokens?
    total_qo_syls = sum(qo_cell_counts.values())
    top_3_count = sum(c for _, c in Counter(qo_cell_counts).most_common(3))
    top_3_pct = top_3_count / total_qo_syls if total_qo_syls > 0 else 0
    clustered = top_3_pct > 0.6

    # Entropy with qo
    h1_with = first_order_entropy(all_text)
    h2_with = conditional_entropy(all_text, order=2)
    word_h1_with = word_unigram_entropy(all_tokens)

    # Entropy without qo
    non_qo_text = ' '.join(non_qo_tokens)
    h1_without = first_order_entropy(non_qo_text)
    h2_without = conditional_entropy(non_qo_text, order=2)
    word_h1_without = word_unigram_entropy(non_qo_tokens)

    # Grid without qo
    grid_without = build_grid_from_tokens(non_qo_tokens) if len(non_qo_tokens) > 50 else grid_with
    cells_with = set(grid_with.cells.keys())
    cells_without = set(grid_without.cells.keys())
    jaccard = len(cells_with & cells_without) / len(cells_with | cells_without) if (cells_with | cells_without) else 1.0

    # Verdict
    h2_improved = h2_without > h2_with + 0.01
    if h2_improved and clustered:
        verdict = 'removal_helps'
    elif not h2_improved and not clustered:
        verdict = 'removal_hurts'
    else:
        verdict = 'removal_neutral'

    return QoAnalysisResult(
        n_qo_tokens=len(qo_tokens),
        n_qo_types=len(qo_types),
        pct_corpus_qo=round(len(qo_tokens) / len(all_tokens) * 100, 2) if all_tokens else 0,
        pct_lang_a_qo=round(pct_a, 2),
        pct_lang_b_qo=round(pct_b, 2),
        qo_type_examples=qo_types[:20],
        qo_grid_cells=dict(Counter(qo_cell_counts).most_common(10)),
        qo_clustered=clustered,
        qo_top_cell_concentration=round(top_3_pct, 4),
        h1_with=round(h1_with, 4),
        h1_without=round(h1_without, 4),
        h2_with=round(h2_with, 4),
        h2_without=round(h2_without, 4),
        word_h1_with=round(word_h1_with, 4),
        word_h1_without=round(word_h1_without, 4),
        grid_occupancy_with=round(grid_with.occupancy, 4),
        grid_occupancy_without=round(grid_without.occupancy, 4),
        grid_jaccard_with_vs_without=round(jaccard, 4),
        verdict=verdict,
    )


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

def _print_results(
    comparison: LanguageComparisonResult,
    qo_full: QoAnalysisResult,
    qo_lang_a: QoAnalysisResult,
) -> None:
    """Print formatted Phase 4.5A+C results."""
    pa = comparison.profile_a
    pb = comparison.profile_b

    print("\nLanguage A Isolation Results")
    print("=" * 65)

    print("\nCorpus Split:")
    print(f"  Language A: {pa.n_tokens:,} tokens ({pa.n_types:,} types, "
          f"{pa.n_folios} folios)")
    print(f"    Sections: {', '.join(pa.sections)}")
    print(f"  Language B: {pb.n_tokens:,} tokens ({pb.n_types:,} types, "
          f"{pb.n_folios} folios)")
    print(f"    Sections: {', '.join(pb.sections)}")

    print(f"\n  H₁(A) = {pa.h1:.4f}   H₁(B) = {pb.h1:.4f}")
    print(f"  H₂(A) = {pa.h2:.4f}   H₂(B) = {pb.h2:.4f}   "
          f"Δ = {comparison.h2_difference:+.4f}")
    ci = comparison.h2_diff_ci
    print(f"    95% CI: [{ci[0]:+.4f}, {ci[1]:+.4f}]  "
          f"{'SIGNIFICANT' if comparison.h2_diff_significant else 'not significant'}")
    print(f"  Word H₁(A) = {pa.word_h1:.4f}   Word H₁(B) = {pb.word_h1:.4f}")

    print(f"\nGrid Comparison:")
    print(f"  Grid A: {pa.grid_n_rows}×{pa.grid_n_cols} "
          f"(occupancy {pa.grid_occupancy:.1%})")
    print(f"  Grid B: {pb.grid_n_rows}×{pb.grid_n_cols} "
          f"(occupancy {pb.grid_occupancy:.1%})")
    print(f"  Jaccard(A, B): {comparison.grid_jaccard:.4f}")

    print(f"\nAbugida Decomposition:")
    print(f"  R(A) = {pa.abugida_r:.4f}   Reverse R(A) = {pa.abugida_reverse_r:.4f}")
    print(f"  R(B) = {pb.abugida_r:.4f}   Reverse R(B) = {pb.abugida_reverse_r:.4f}")

    print(f"\nBigram JSD: {comparison.bigram_jsd:.6f}")
    print(f"  Null JSD: {comparison.null_jsd_mean:.6f} ± {comparison.null_jsd_std:.6f}")
    print(f"  z-score: {comparison.jsd_z_score:.2f}")
    print(f"\nVocabulary overlap: {comparison.vocabulary_overlap:.4f}")
    print(f"\nVERDICT: {comparison.verdict.upper()}")

    # Language B characterization
    print("\n" + "-" * 65)
    print("Language B Profile:")
    print(f"  Top 20 tokens: {', '.join(t for t, _ in pb.top_20_tokens)}")
    top_20_count = sum(c for _, c in pb.top_20_tokens)
    print(f"  Top 20 account for {top_20_count / pb.n_tokens:.1%} of Language B")
    print(f"  TTR: {pb.type_token_ratio:.4f}")

    # qo- analysis
    print("\n" + "-" * 65)
    print("qo- Token Analysis:")
    print(f"  Total qo- tokens: {qo_full.n_qo_tokens:,} "
          f"({qo_full.pct_corpus_qo:.1f}% of corpus)")
    print(f"  Unique qo- types: {qo_full.n_qo_types}")
    print(f"  In Language A: {qo_full.pct_lang_a_qo:.1f}%")
    print(f"  In Language B: {qo_full.pct_lang_b_qo:.1f}%")
    print(f"  Grid cell clustering: "
          f"{'CONCENTRATED' if qo_full.qo_clustered else 'DISTRIBUTED'} "
          f"(top-3 = {qo_full.qo_top_cell_concentration:.1%})")

    print(f"\n  Effect of Removal (full corpus):")
    print(f"  {'Metric':<25s} {'With qo-':>10s} {'Without qo-':>12s} {'Change':>10s}")
    print(f"  {'-'*57}")
    for label, w, wo in [
        ('H₁', qo_full.h1_with, qo_full.h1_without),
        ('H₂', qo_full.h2_with, qo_full.h2_without),
        ('Word H₁', qo_full.word_h1_with, qo_full.word_h1_without),
        ('Grid occupancy', qo_full.grid_occupancy_with, qo_full.grid_occupancy_without),
    ]:
        delta = wo - w
        print(f"  {label:<25s} {w:>10.4f} {wo:>12.4f} {delta:>+10.4f}")
    print(f"  Grid Jaccard (with vs without): {qo_full.grid_jaccard_with_vs_without:.4f}")
    print(f"\n  VERDICT: {qo_full.verdict.upper()}")

    if qo_lang_a.n_qo_tokens > 0:
        print(f"\n  qo- removal on Language A only:")
        print(f"    H₂ change: {qo_lang_a.h2_with:.4f} → {qo_lang_a.h2_without:.4f} "
              f"({qo_lang_a.h2_without - qo_lang_a.h2_with:+.4f})")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_language_a_isolation() -> Dict:
    """Run Priority A (language isolation) and Priority C (qo- removal)."""
    print("=" * 70)
    print("PHASE 4.5 PRIORITIES A+C: LANGUAGE A ISOLATION + qo- REMOVAL")
    print("=" * 70)

    corpus = load_corpus(verbose=False)

    # A.1-A.2: Build per-language profiles
    print("\nBuilding Language A profile...")
    profile_a, tokens_a, text_a = _build_language_profile(corpus, 'A')

    print("Building Language B profile...")
    profile_b, tokens_b, text_b = _build_language_profile(corpus, 'B')

    # A.3-A.4: Compare languages
    print("Comparing languages...")
    comparison = compare_languages(
        profile_a, profile_b, tokens_a, tokens_b, text_a, text_b,
    )

    # C.1-C.3: qo- analysis
    print("Analyzing qo- tokens (full corpus)...")
    qo_full = analyze_qo_removal(corpus)

    print("Analyzing qo- tokens (Language A only)...")
    qo_lang_a = analyze_qo_removal(corpus, language='A')

    # Print results
    _print_results(comparison, qo_full, qo_lang_a)

    # Save
    rd = _results_dir()
    out_data = {
        'language_a': asdict(profile_a),
        'language_b': asdict(profile_b),
        'comparison': {
            'bigram_jsd': comparison.bigram_jsd,
            'grid_jaccard': comparison.grid_jaccard,
            'h2_difference': comparison.h2_difference,
            'h2_diff_ci': list(comparison.h2_diff_ci),
            'h2_diff_significant': comparison.h2_diff_significant,
            'vocabulary_overlap': comparison.vocabulary_overlap,
            'null_jsd_mean': comparison.null_jsd_mean,
            'null_jsd_std': comparison.null_jsd_std,
            'jsd_z_score': comparison.jsd_z_score,
            'verdict': comparison.verdict,
        },
        'qo_full_corpus': asdict(qo_full),
        'qo_language_a': asdict(qo_lang_a),
    }

    out_path = os.path.join(rd, 'language_a_isolation.json')
    with open(out_path, 'w') as f:
        json.dump(out_data, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return out_data
