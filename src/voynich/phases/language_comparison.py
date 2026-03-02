"""
Phase 9.4 — Expanded Language Comparison
=========================================

Rationale
---------
The 4-language MDL results from Phase 8 were confounded by corpus size.
This phase runs every validated metric against all four reference corpora
with matched sample sizes (11 K tokens) and bootstrap confidence intervals
to resolve the source language question.

Sub-analyses
------------
9.4a  Corpus size normalization (subsample all to 11 K tokens)
9.4b  Matched metric matrix (6 metrics x 4 languages)
9.4c  Language ranking with bootstrap CIs
9.4d  Occitan vs Italian head-to-head
Null  Bootstrap subsample variance
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import load_corpus
from voynich.core.reference import load_reference_corpus
from voynich.core.stats import (
    bigram_transition_matrix,
    bootstrap_ci,
    compare_bigram_matrices,
    conditional_entropy,
    type_token_ratio_at_n,
    word_length_distribution,
    zipf_analysis,
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class NormalizedCorpusStats:
    source: str
    original_token_count: int
    normalized_token_count: int
    vocab_size: int
    ttr: float
    subsampled: bool


@dataclass
class MetricRow:
    metric: str
    voynich: float
    latin: float
    occitan: float
    italian: float
    german: float
    closest_language: str
    closest_distance: float


@dataclass
class LanguageRankingWithCI:
    language: str
    mean_distance_to_voynich: float
    ci_lower: float
    ci_upper: float
    rank: int
    n_metrics_closest: int


@dataclass
class HeadToHeadResult:
    metric: str
    occitan_distance: float
    italian_distance: float
    closer_to_voynich: str
    margin: float


@dataclass
class LanguageComparisonResult:
    corpus_stats: List[Dict]
    normalized_token_count: int
    metric_matrix: List[Dict]
    language_ranking: List[Dict]
    best_language: str
    second_language: str
    separation_significant: bool
    head_to_head: List[Dict]
    head_to_head_winner: str
    head_to_head_margin: float
    subsample_variance: Dict
    gate_passed: bool
    verdict: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _convert(obj: Any) -> Any:
    """Convert dataclass/numpy types to JSON-serializable form."""
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
    if isinstance(obj, (bool, int, float, str, type(None))):
        return obj
    return str(obj)


def _subsample_tokens(
    tokens: List[str], target_n: int, seed: int = 42,
) -> List[str]:
    """
    Subsample tokens to *target_n* by selecting a contiguous chunk
    (preserves bigram structure).
    """
    if len(tokens) <= target_n:
        return tokens
    rng = random.Random(seed)
    start = rng.randint(0, len(tokens) - target_n)
    return tokens[start:start + target_n]


def _compute_metrics(
    text: str, tokens: List[str],
) -> Dict[str, float]:
    """Compute the 6 standardized metrics for a token sequence."""
    h2 = conditional_entropy(text, order=1)
    h3 = conditional_entropy(text, order=2)
    za = zipf_analysis(tokens)
    wld = word_length_distribution(tokens)
    ttr_vals = type_token_ratio_at_n(tokens, n_values=[len(tokens)])
    ttr = ttr_vals.get(len(tokens), len(set(tokens)) / len(tokens) if tokens else 0.0)

    return {
        'h2': h2,
        'h3': h3,
        'zipf_exponent': za['zipf_exponent'],
        'word_length_mean': wld.get('mean', 0.0),
        'ttr': ttr,
        'zipf_r_squared': za['r_squared'],
    }


def _bigram_jsd(text_a: str, text_b: str) -> float:
    """Compute JSD between character bigram matrices of two texts."""
    mat_a, alph_a = bigram_transition_matrix(text_a)
    mat_b, alph_b = bigram_transition_matrix(text_b)
    return compare_bigram_matrices(mat_a, mat_b, alph_a, alph_b)


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_language_comparison() -> Dict:
    """
    Phase 9.4: Four-language matched comparison with bootstrap CIs.
    """
    print("Phase 9.4: Expanded Language Comparison")
    print("=" * 60)

    # --- Load data ---
    corpus = load_corpus(verbose=False)
    ref_corpus = load_reference_corpus(verbose=False)
    voynich_tokens = corpus.get_tokens(language='A')
    voynich_text = corpus.get_text(language='A')

    # ===================================================================
    # 9.4a: Corpus size normalization
    # ===================================================================
    print("\n  9.4a: Corpus normalization ...")

    languages = ['latin', 'occitan', 'italian', 'german']
    raw_tokens: Dict[str, List[str]] = {}
    for lang in languages:
        try:
            toks = ref_corpus.get_combined_tokens(lang)
            raw_tokens[lang] = toks
            print(f"    {lang}: {len(toks):,} tokens")
        except Exception as e:
            print(f"    {lang}: unavailable — {e}")

    available_langs = [l for l in languages if l in raw_tokens and len(raw_tokens[l]) >= 100]
    min_size = min(len(raw_tokens[l]) for l in available_langs)
    target_n = min(min_size, 11_000)
    print(f"    Normalization target: {target_n:,} tokens")

    corpus_stats: List[NormalizedCorpusStats] = []
    # Voynich stats
    v_sub = _subsample_tokens(voynich_tokens, target_n)
    v_text_sub = ' '.join(v_sub)
    corpus_stats.append(NormalizedCorpusStats(
        source='voynich', original_token_count=len(voynich_tokens),
        normalized_token_count=len(v_sub), vocab_size=len(set(v_sub)),
        ttr=len(set(v_sub)) / len(v_sub), subsampled=len(voynich_tokens) > target_n,
    ))

    ref_sub: Dict[str, Tuple[List[str], str]] = {}
    for lang in available_langs:
        toks = _subsample_tokens(raw_tokens[lang], target_n)
        text = ' '.join(toks)
        ref_sub[lang] = (toks, text)
        corpus_stats.append(NormalizedCorpusStats(
            source=lang, original_token_count=len(raw_tokens[lang]),
            normalized_token_count=len(toks), vocab_size=len(set(toks)),
            ttr=len(set(toks)) / len(toks), subsampled=len(raw_tokens[lang]) > target_n,
        ))
        print(f"    {lang} subsampled: {len(toks):,} tokens, "
              f"{len(set(toks))} types")

    # ===================================================================
    # 9.4b: Metric matrix
    # ===================================================================
    print("\n  9.4b: Computing metric matrix ...")
    v_metrics = _compute_metrics(v_text_sub, v_sub)

    lang_metrics: Dict[str, Dict[str, float]] = {}
    for lang in available_langs:
        toks, text = ref_sub[lang]
        lang_metrics[lang] = _compute_metrics(text, toks)

    # Bigram JSD (computed separately since it's a pairwise metric)
    for lang in available_langs:
        _, text = ref_sub[lang]
        jsd = _bigram_jsd(v_text_sub, text)
        lang_metrics[lang]['bigram_jsd'] = jsd

    metric_names = ['h2', 'h3', 'zipf_exponent', 'word_length_mean', 'ttr', 'bigram_jsd']
    metric_rows: List[MetricRow] = []

    for m in metric_names:
        v_val = v_metrics.get(m, 0.0)
        lang_vals = {}
        for lang in available_langs:
            if m == 'bigram_jsd':
                lang_vals[lang] = lang_metrics[lang].get('bigram_jsd', 0.0)
            else:
                lang_vals[lang] = abs(v_val - lang_metrics[lang].get(m, 0.0))

        if m == 'bigram_jsd':
            # For JSD, smaller = closer
            closest = min(lang_vals, key=lang_vals.get) if lang_vals else 'none'
            closest_dist = lang_vals.get(closest, 0.0)
        else:
            closest = min(lang_vals, key=lang_vals.get) if lang_vals else 'none'
            closest_dist = lang_vals.get(closest, 0.0)

        row_vals = {lang: lang_vals.get(lang, 0.0) for lang in languages}
        row = MetricRow(
            metric=m, voynich=v_val,
            latin=row_vals.get('latin', 0.0),
            occitan=row_vals.get('occitan', 0.0),
            italian=row_vals.get('italian', 0.0),
            german=row_vals.get('german', 0.0),
            closest_language=closest,
            closest_distance=closest_dist,
        )
        metric_rows.append(row)
        print(f"    {m}: Voynich={v_val:.4f}  closest={closest} "
              f"(dist={closest_dist:.4f})")

    # ===================================================================
    # 9.4c: Language ranking with bootstrap CIs
    # ===================================================================
    print("\n  9.4c: Language ranking with bootstrap CIs ...")

    n_bootstrap = 100
    lang_distances_samples: Dict[str, List[float]] = {l: [] for l in available_langs}

    for b in range(n_bootstrap):
        seed = 42 + b
        v_b = _subsample_tokens(voynich_tokens, target_n, seed=seed)
        v_b_text = ' '.join(v_b)
        v_b_metrics = _compute_metrics(v_b_text, v_b)

        for lang in available_langs:
            r_b = _subsample_tokens(raw_tokens[lang], target_n, seed=seed)
            r_b_text = ' '.join(r_b)
            r_b_metrics = _compute_metrics(r_b_text, r_b)

            # Composite distance (sum of absolute metric differences)
            dist = 0.0
            for m in ['h2', 'h3', 'zipf_exponent', 'word_length_mean', 'ttr']:
                dist += abs(v_b_metrics[m] - r_b_metrics[m])
            # Add bigram JSD
            jsd = _bigram_jsd(v_b_text, r_b_text)
            dist += jsd
            lang_distances_samples[lang].append(dist)

    rankings: List[LanguageRankingWithCI] = []
    for lang in available_langs:
        samples = lang_distances_samples[lang]
        mean_d = float(np.mean(samples))
        ci_lo = float(np.percentile(samples, 2.5))
        ci_hi = float(np.percentile(samples, 97.5))
        n_closest = sum(1 for r in metric_rows if r.closest_language == lang)
        rankings.append(LanguageRankingWithCI(
            language=lang, mean_distance_to_voynich=mean_d,
            ci_lower=ci_lo, ci_upper=ci_hi, rank=0,
            n_metrics_closest=n_closest,
        ))

    rankings.sort(key=lambda r: r.mean_distance_to_voynich)
    for i, r in enumerate(rankings):
        r.rank = i + 1
        print(f"    #{r.rank} {r.language}: dist={r.mean_distance_to_voynich:.4f} "
              f"CI=[{r.ci_lower:.4f}, {r.ci_upper:.4f}]  "
              f"closest on {r.n_metrics_closest} metrics")

    best_lang = rankings[0].language if rankings else 'unknown'
    second_lang = rankings[1].language if len(rankings) > 1 else 'unknown'

    # Check CI separation
    separation = False
    if len(rankings) >= 2:
        separation = rankings[0].ci_upper < rankings[1].ci_lower
    print(f"    Separation significant: {separation}")

    # ===================================================================
    # 9.4d: Occitan vs Italian head-to-head
    # ===================================================================
    print("\n  9.4d: Occitan vs Italian head-to-head ...")
    h2h_results: List[HeadToHeadResult] = []
    occ_wins = 0
    ita_wins = 0

    if 'occitan' in available_langs and 'italian' in available_langs:
        for r in metric_rows:
            occ_dist = r.occitan
            ita_dist = r.italian
            closer = 'occitan' if occ_dist < ita_dist else 'italian'
            margin = abs(occ_dist - ita_dist)
            if closer == 'occitan':
                occ_wins += 1
            else:
                ita_wins += 1
            h2h_results.append(HeadToHeadResult(
                metric=r.metric, occitan_distance=occ_dist,
                italian_distance=ita_dist, closer_to_voynich=closer,
                margin=margin,
            ))
            print(f"    {r.metric}: closer={closer}  margin={margin:.4f}")

    h2h_winner = 'occitan' if occ_wins > ita_wins else 'italian' if ita_wins > occ_wins else 'tie'
    h2h_margin_total = occ_wins - ita_wins
    print(f"    Winner: {h2h_winner} ({occ_wins} vs {ita_wins})")

    # ===================================================================
    # Subsample variance (null stability check)
    # ===================================================================
    variance_by_metric: Dict[str, float] = {}
    for lang in available_langs:
        for m in ['h2', 'h3', 'zipf_exponent', 'word_length_mean', 'ttr']:
            key = f"{lang}_{m}"
            vals = []
            for b in range(min(30, n_bootstrap)):
                r_b = _subsample_tokens(raw_tokens[lang], target_n, seed=100 + b)
                r_b_text = ' '.join(r_b)
                r_b_metrics = _compute_metrics(r_b_text, r_b)
                vals.append(r_b_metrics[m])
            variance_by_metric[key] = float(np.std(vals))

    # ===================================================================
    # Gate
    # ===================================================================
    gate_passed = separation

    if gate_passed:
        verdict = f'definitive_language_identification_{best_lang}'
    elif len(rankings) >= 2 and rankings[0].mean_distance_to_voynich < rankings[1].mean_distance_to_voynich * 0.9:
        verdict = f'probable_{best_lang}_but_cis_overlap'
    else:
        verdict = 'languages_indistinguishable_at_this_sample_size'

    print(f"\n  Gate: separation={gate_passed}")
    print(f"  Verdict: {verdict}")

    # ===================================================================
    # Save
    # ===================================================================
    result = LanguageComparisonResult(
        corpus_stats=[_convert(asdict(cs)) for cs in corpus_stats],
        normalized_token_count=target_n,
        metric_matrix=[_convert(asdict(mr)) for mr in metric_rows],
        language_ranking=[_convert(asdict(lr)) for lr in rankings],
        best_language=best_lang,
        second_language=second_lang,
        separation_significant=separation,
        head_to_head=[_convert(asdict(h)) for h in h2h_results],
        head_to_head_winner=h2h_winner,
        head_to_head_margin=float(h2h_margin_total),
        subsample_variance=variance_by_metric,
        gate_passed=gate_passed,
        verdict=verdict,
    )

    out = _convert(asdict(result))
    with open(_results_dir() / 'language_comparison.json', 'w') as f:
        json.dump(out, f, indent=2)

    print(f"\n  Results saved to results/language_comparison.json")
    return out
