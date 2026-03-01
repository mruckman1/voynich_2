"""
Phase 4 Step 4: Multi-Language Comparison
==========================================
Rank all available languages across three independent metrics:
  1. Fingerprint similarity (entropy profile cosine distance)
  2. Bigram structure distance (JSD of character bigram matrices)
  3. PMI distribution correlation (syllable-level mutual information)

Each metric gets bootstrap confidence intervals. The module determines
whether the best language can be statistically separated from alternatives.

Output:
  multi_language.json — per-language rankings with CIs, combined ranking
"""

import json
import math
import os
import random
from collections import Counter
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from voynich.core.corpus import load_corpus, VoynichCorpus
from voynich.core.stats import (
    bigram_transition_matrix, jensen_shannon_divergence,
    cosine_similarity, pearson_correlation, syllabify_latin_text,
    first_order_entropy,
)
from voynich.core.reference import (
    load_reference_corpus, get_reference_text,
    ReferenceCorpus,
)
from voynich.core.ciphers import REFERENCE_LANGUAGES
from voynich.core._paths import results_dir as _results_dir
from voynich.analysis.fingerprint import (
    compute_profile, EntropyProfile,
)
from voynich.phases.syllable_match import (
    assign_cv_labels, retranscribe_token, compute_syllable_pmi,
    _build_merge_maps, CVLabel,
)
from voynich.phases.grid_validate import build_grid_from_tokens
from voynich.analysis.strokes import SyllabaryGrid


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class LanguageRanking:
    """One language's scores across all metrics."""
    language: str
    corpus_type: str             # 'real' or 'synthetic'
    fingerprint_similarity: float
    fingerprint_ci_lower: float
    fingerprint_ci_upper: float
    bigram_jsd: float
    bigram_jsd_ci_lower: float
    bigram_jsd_ci_upper: float
    pmi_correlation: float
    pmi_ci_lower: float
    pmi_ci_upper: float
    mean_rank: float
    overall_rank: int


@dataclass
class MultiLanguageResult:
    """Full multi-language comparison output."""
    languages_tested: List[str]
    rankings: List[LanguageRanking]
    best_language: str
    second_language: str
    separation_significant: bool
    separation_details: Dict[str, bool]  # per-metric CI overlap check


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _has_real_corpus(language: str, ref_corpus: ReferenceCorpus) -> bool:
    """Check whether a language has real corpus files (vs synthetic only)."""
    return language in ref_corpus.languages and len(ref_corpus.get_texts(language)) > 0


def _compute_bigram_jsd(text_a: str, text_b: str) -> float:
    """Compute JSD between character bigram distributions of two texts."""
    mat_a, alph_a = bigram_transition_matrix(text_a)
    mat_b, alph_b = bigram_transition_matrix(text_b)

    all_chars = sorted(set(alph_a) | set(alph_b))
    n = len(all_chars)
    char_idx = {c: i for i, c in enumerate(all_chars)}

    aligned_a = np.zeros((n, n))
    for i, ca in enumerate(alph_a):
        for j, cb in enumerate(alph_a):
            aligned_a[char_idx[ca], char_idx[cb]] = mat_a[i, j]

    aligned_b = np.zeros((n, n))
    for i, ca in enumerate(alph_b):
        for j, cb in enumerate(alph_b):
            aligned_b[char_idx[ca], char_idx[cb]] = mat_b[i, j]

    flat_a = aligned_a.flatten()
    flat_b = aligned_b.flatten()
    sa = flat_a.sum()
    sb = flat_b.sum()
    if sa > 0:
        flat_a = flat_a / sa
    if sb > 0:
        flat_b = flat_b / sb
    return jensen_shannon_divergence(flat_a, flat_b)


# ---------------------------------------------------------------------------
# Metric 1: Fingerprint similarity
# ---------------------------------------------------------------------------

def compute_fingerprint_rankings(
    voynich_profile: EntropyProfile,
    ref_corpus: ReferenceCorpus,
    languages: List[str],
    n_bootstrap: int = 100,
    n_words: int = 2000,
    seed: int = 42,
) -> List[Tuple[str, float, float, float]]:
    """
    Rank languages by entropy profile similarity.

    Returns list of (language, similarity, ci_lower, ci_upper).
    """
    q_vec = voynich_profile.to_vector()
    results = []

    for lang in languages:
        sims = []
        for trial in range(n_bootstrap):
            trial_seed = seed + trial
            try:
                ref_text = get_reference_text(lang, n_words=n_words,
                                              seed=trial_seed, corpus=ref_corpus)
            except Exception:
                continue
            ref_tokens = ref_text.split()
            if len(ref_tokens) < 50:
                continue
            ref_profile = compute_profile(ref_text, ref_tokens, label=lang)
            ref_vec = ref_profile.to_vector()
            sim = cosine_similarity(q_vec, ref_vec)
            sims.append(sim)

        if sims:
            arr = np.array(sims)
            mean_sim = float(np.mean(arr))
            ci_low = float(np.percentile(arr, 2.5))
            ci_high = float(np.percentile(arr, 97.5))
        else:
            mean_sim = 0.0
            ci_low = 0.0
            ci_high = 0.0

        results.append((lang, mean_sim, ci_low, ci_high))

    results.sort(key=lambda x: x[1], reverse=True)
    return results


# ---------------------------------------------------------------------------
# Metric 2: Bigram JSD
# ---------------------------------------------------------------------------

def compute_bigram_rankings(
    voynich_text: str,
    ref_corpus: ReferenceCorpus,
    languages: List[str],
    n_bootstrap: int = 100,
    n_words: int = 2000,
    seed: int = 42,
) -> List[Tuple[str, float, float, float]]:
    """
    Rank languages by character bigram JSD (lower = more similar).

    Returns list of (language, jsd, ci_lower, ci_upper).
    """
    results = []

    for lang in languages:
        jsds = []
        for trial in range(n_bootstrap):
            trial_seed = seed + trial
            try:
                ref_text = get_reference_text(lang, n_words=n_words,
                                              seed=trial_seed, corpus=ref_corpus)
            except Exception:
                continue
            if len(ref_text) < 100:
                continue
            jsd = _compute_bigram_jsd(voynich_text, ref_text)
            jsds.append(jsd)

        if jsds:
            arr = np.array(jsds)
            mean_jsd = float(np.mean(arr))
            ci_low = float(np.percentile(arr, 2.5))
            ci_high = float(np.percentile(arr, 97.5))
        else:
            mean_jsd = 1.0
            ci_low = 1.0
            ci_high = 1.0

        results.append((lang, mean_jsd, ci_low, ci_high))

    results.sort(key=lambda x: x[1])  # lower JSD = better
    return results


# ---------------------------------------------------------------------------
# Metric 3: PMI correlation
# ---------------------------------------------------------------------------

def compute_pmi_rankings(
    tokens: List[str],
    cv_labels: Dict[str, CVLabel],
    grid: SyllabaryGrid,
    ref_corpus: ReferenceCorpus,
    languages: List[str],
    n_bootstrap: int = 100,
    n_words: int = 3000,
    seed: int = 42,
) -> List[Tuple[str, float, float, float]]:
    """
    Rank languages by PMI distribution correlation.

    Returns list of (language, correlation, ci_lower, ci_upper).
    """
    onset_merge, nucleus_merge = _build_merge_maps(grid)

    # Voynich PMI (fixed across bootstrap)
    voynich_seqs = []
    for token in tokens:
        cv_seq = retranscribe_token(token, cv_labels, onset_merge, nucleus_merge)
        voynich_seqs.append(cv_seq)
    v_pmi = compute_syllable_pmi(voynich_seqs, top_n=50)

    if not v_pmi:
        return [(lang, 0.0, 0.0, 0.0) for lang in languages]

    v_values = np.array(sorted(v_pmi.values(), reverse=True))

    results = []
    for lang in languages:
        corrs = []
        for trial in range(n_bootstrap):
            trial_seed = seed + trial
            try:
                ref_text = get_reference_text(lang, n_words=n_words,
                                              seed=trial_seed, corpus=ref_corpus)
            except Exception:
                continue
            ref_syl_seqs = syllabify_latin_text(ref_text)
            r_pmi = compute_syllable_pmi(ref_syl_seqs, top_n=50)
            if not r_pmi:
                continue

            r_values = np.array(sorted(r_pmi.values(), reverse=True))
            n = min(len(v_values), len(r_values))
            if n < 3:
                continue

            r_corr, _ = pearson_correlation(v_values[:n], r_values[:n])
            corrs.append(r_corr)

        if corrs:
            arr = np.array(corrs)
            mean_corr = float(np.mean(arr))
            ci_low = float(np.percentile(arr, 2.5))
            ci_high = float(np.percentile(arr, 97.5))
        else:
            mean_corr = 0.0
            ci_low = 0.0
            ci_high = 0.0

        results.append((lang, mean_corr, ci_low, ci_high))

    results.sort(key=lambda x: x[1], reverse=True)
    return results


# ---------------------------------------------------------------------------
# Combine rankings
# ---------------------------------------------------------------------------

def _ci_overlaps(ci_a: Tuple[float, float], ci_b: Tuple[float, float]) -> bool:
    """Check whether two confidence intervals overlap."""
    return ci_a[0] <= ci_b[1] and ci_b[0] <= ci_a[1]


def combine_rankings(
    fingerprint: List[Tuple[str, float, float, float]],
    bigram: List[Tuple[str, float, float, float]],
    pmi: List[Tuple[str, float, float, float]],
    ref_corpus: ReferenceCorpus,
) -> MultiLanguageResult:
    """Combine three ranking lists into a unified ranking."""
    # Build per-language rank maps
    fp_rank = {lang: rank + 1 for rank, (lang, *_) in enumerate(fingerprint)}
    bg_rank = {lang: rank + 1 for rank, (lang, *_) in enumerate(bigram)}
    pmi_rank = {lang: rank + 1 for rank, (lang, *_) in enumerate(pmi)}

    # Build lookup dicts
    fp_dict = {lang: (sim, lo, hi) for lang, sim, lo, hi in fingerprint}
    bg_dict = {lang: (jsd, lo, hi) for lang, jsd, lo, hi in bigram}
    pmi_dict = {lang: (corr, lo, hi) for lang, corr, lo, hi in pmi}

    all_langs = list(fp_rank.keys())
    rows = []

    for lang in all_langs:
        fp_sim, fp_lo, fp_hi = fp_dict.get(lang, (0, 0, 0))
        bg_jsd, bg_lo, bg_hi = bg_dict.get(lang, (1, 1, 1))
        pm_corr, pm_lo, pm_hi = pmi_dict.get(lang, (0, 0, 0))

        mean_r = (fp_rank.get(lang, len(all_langs)) +
                  bg_rank.get(lang, len(all_langs)) +
                  pmi_rank.get(lang, len(all_langs))) / 3.0

        rows.append(LanguageRanking(
            language=lang,
            corpus_type='real' if _has_real_corpus(lang, ref_corpus) else 'synthetic',
            fingerprint_similarity=round(fp_sim, 4),
            fingerprint_ci_lower=round(fp_lo, 4),
            fingerprint_ci_upper=round(fp_hi, 4),
            bigram_jsd=round(bg_jsd, 6),
            bigram_jsd_ci_lower=round(bg_lo, 6),
            bigram_jsd_ci_upper=round(bg_hi, 6),
            pmi_correlation=round(pm_corr, 4),
            pmi_ci_lower=round(pm_lo, 4),
            pmi_ci_upper=round(pm_hi, 4),
            mean_rank=round(mean_r, 2),
            overall_rank=0,
        ))

    rows.sort(key=lambda r: r.mean_rank)
    for i, row in enumerate(rows):
        row.overall_rank = i + 1

    # Check separation between #1 and #2
    best = rows[0].language if rows else ''
    second = rows[1].language if len(rows) > 1 else ''

    sep_details = {}
    if len(rows) >= 2:
        fp1 = fp_dict.get(best, (0, 0, 0))
        fp2 = fp_dict.get(second, (0, 0, 0))
        sep_details['fingerprint'] = not _ci_overlaps((fp1[1], fp1[2]), (fp2[1], fp2[2]))

        bg1 = bg_dict.get(best, (1, 1, 1))
        bg2 = bg_dict.get(second, (1, 1, 1))
        sep_details['bigram'] = not _ci_overlaps((bg1[1], bg1[2]), (bg2[1], bg2[2]))

        pm1 = pmi_dict.get(best, (0, 0, 0))
        pm2 = pmi_dict.get(second, (0, 0, 0))
        sep_details['pmi'] = not _ci_overlaps((pm1[1], pm1[2]), (pm2[1], pm2[2]))

    any_separated = any(sep_details.values()) if sep_details else False

    return MultiLanguageResult(
        languages_tested=all_langs,
        rankings=rows,
        best_language=best,
        second_language=second,
        separation_significant=any_separated,
        separation_details=sep_details,
    )


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

def _print_results(result: MultiLanguageResult) -> None:
    """Print formatted multi-language comparison."""
    print(f"\nLanguages tested: {len(result.languages_tested)}")
    for r in result.rankings:
        tag = f" ({r.corpus_type})" if r.corpus_type == 'synthetic' else ""
        print(f"  {r.language}{tag}")

    # Fingerprint ranking
    print("\n--- Fingerprint Similarity (higher = better) ---")
    print(f"  {'Rank':>4s} {'Language':<12s} {'Similarity':>11s} {'95% CI':>20s}")
    print("  " + "-" * 49)
    for r in sorted(result.rankings, key=lambda x: x.fingerprint_similarity, reverse=True):
        print(f"  {0:>4d} {r.language:<12s} {r.fingerprint_similarity:>11.4f} "
              f"[{r.fingerprint_ci_lower:.4f}, {r.fingerprint_ci_upper:.4f}]")

    # Bigram JSD ranking
    print("\n--- Bigram JSD (lower = better) ---")
    print(f"  {'Rank':>4s} {'Language':<12s} {'JSD':>11s} {'95% CI':>24s}")
    print("  " + "-" * 53)
    for r in sorted(result.rankings, key=lambda x: x.bigram_jsd):
        print(f"  {0:>4d} {r.language:<12s} {r.bigram_jsd:>11.6f} "
              f"[{r.bigram_jsd_ci_lower:.6f}, {r.bigram_jsd_ci_upper:.6f}]")

    # PMI ranking
    print("\n--- PMI Correlation (higher = better) ---")
    print(f"  {'Rank':>4s} {'Language':<12s} {'Correlation':>12s} {'95% CI':>20s}")
    print("  " + "-" * 50)
    for r in sorted(result.rankings, key=lambda x: x.pmi_correlation, reverse=True):
        print(f"  {0:>4d} {r.language:<12s} {r.pmi_correlation:>12.4f} "
              f"[{r.pmi_ci_lower:.4f}, {r.pmi_ci_upper:.4f}]")

    # Combined
    print("\n--- Combined Ranking ---")
    print(f"  {'Rank':>4s} {'Language':<12s} {'Type':<10s} {'Mean Rank':>10s}")
    print("  " + "-" * 38)
    for r in result.rankings:
        print(f"  {r.overall_rank:>4d} {r.language:<12s} {r.corpus_type:<10s} "
              f"{r.mean_rank:>10.2f}")

    # Separation
    print(f"\n  Best language:   {result.best_language}")
    print(f"  Second language: {result.second_language}")
    print(f"  Separation details:")
    for metric, separated in result.separation_details.items():
        print(f"    {metric}: {'SEPARATED (non-overlapping CIs)' if separated else 'overlapping CIs'}")
    print(f"  Overall separation: {'YES' if result.separation_significant else 'NO'}")

    if not result.separation_significant:
        print(f"  -> Cannot statistically distinguish {result.best_language} from "
              f"{result.second_language}")
    else:
        metrics = [m for m, s in result.separation_details.items() if s]
        print(f"  -> {result.best_language} separates from {result.second_language} "
              f"on: {', '.join(metrics)}")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_multi_language() -> Dict:
    """Run the multi-language comparison and save results."""
    print("=" * 70)
    print("PHASE 4 STEP 4: MULTI-LANGUAGE COMPARISON")
    print("=" * 70)

    # Load data
    corpus = load_corpus(verbose=False)
    tokens = corpus.get_tokens(paragraph_only=True)
    voynich_text = ' '.join(tokens)

    print("\nLoading reference corpora...")
    ref_corpus = load_reference_corpus(verbose=False)
    languages = REFERENCE_LANGUAGES

    print(f"Languages: {', '.join(languages)}")
    for lang in languages:
        if _has_real_corpus(lang, ref_corpus):
            texts = ref_corpus.get_texts(lang)
            total = sum(t.token_count for t in texts)
            print(f"  {lang}: {len(texts)} real texts, {total:,} tokens")
        else:
            print(f"  {lang}: synthetic vocabulary only")

    # Compute Voynich profile
    print("\nComputing Voynich fingerprint profile...")
    voynich_profile = compute_profile(voynich_text, tokens, label='voynich')

    # Build grid and CV labels for PMI
    print("Building grid and CV labels...")
    grid = build_grid_from_tokens(tokens)
    cv_labels = assign_cv_labels(grid, tokens)

    # Metric 1: Fingerprint
    print("\nComputing fingerprint rankings (100 bootstrap samples per language)...")
    fp_rankings = compute_fingerprint_rankings(
        voynich_profile, ref_corpus, languages, n_bootstrap=100)

    # Metric 2: Bigram JSD
    print("Computing bigram JSD rankings (100 bootstrap samples per language)...")
    bg_rankings = compute_bigram_rankings(
        voynich_text, ref_corpus, languages, n_bootstrap=100)

    # Metric 3: PMI correlation
    print("Computing PMI correlation rankings (100 bootstrap samples per language)...")
    pmi_rankings = compute_pmi_rankings(
        tokens, cv_labels, grid, ref_corpus, languages, n_bootstrap=100)

    # Combine
    result = combine_rankings(fp_rankings, bg_rankings, pmi_rankings, ref_corpus)
    _print_results(result)

    # Save
    rd = _results_dir()
    out_data = {
        'languages_tested': result.languages_tested,
        'rankings': [asdict(r) for r in result.rankings],
        'best_language': result.best_language,
        'second_language': result.second_language,
        'separation_significant': result.separation_significant,
        'separation_details': result.separation_details,
        'fingerprint_rankings': [
            {'language': l, 'similarity': s, 'ci_lower': lo, 'ci_upper': hi}
            for l, s, lo, hi in fp_rankings
        ],
        'bigram_rankings': [
            {'language': l, 'jsd': j, 'ci_lower': lo, 'ci_upper': hi}
            for l, j, lo, hi in bg_rankings
        ],
        'pmi_rankings': [
            {'language': l, 'correlation': c, 'ci_lower': lo, 'ci_upper': hi}
            for l, c, lo, hi in pmi_rankings
        ],
    }
    out_path = os.path.join(rd, 'multi_language.json')
    with open(out_path, 'w') as f:
        json.dump(out_data, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return out_data
