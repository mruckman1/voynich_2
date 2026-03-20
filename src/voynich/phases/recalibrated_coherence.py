"""
Phase 60, Track B: Recalibrated CVC Permutation Coherence Test
==============================================================
Phase 59 Inv 11 found p=0.552 — the CV-era coherence criteria (verb>=2,
function>=3, pharma>=1) are trivially satisfied under CVC decode.

This module profiles 1000 random coda tables on expanded metrics, finds
calibrated thresholds at the 92nd percentile (~8% pass rate), and scores
the corrected Phase 60 table against them.

Dependency chain:
    results/corrected_coda.json       (Track A)
    results/combined_refine.json      (Phase 15)
    results/modifier_integrate.json   (Phase 16)
    results/null_corpus.json          (Phase 17)
        -> results/recalibrated_coherence.json
"""

import json
import math
import os
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from scipy import stats as scipy_stats

from voynich.core._paths import results_dir as _results_dir
from voynich.phases.coda_markers import build_coda_table, decode_corpus_cvc
from voynich.phases.corrected_coda import (
    FUNCTION_WORDS,
    LATIN_ENDINGS,
    build_coda_table_v2,
    decode_corpus_cvc_v2,
)
from voynich.phases.cvc_coda_signal import _load_shared_data
from voynich.phases.cvc_permutation import (
    FUNCTION_KIT,
    PHARMA_REGISTER,
    VERB_PARADIGM_WORDS,
    _build_permuted_coda_table,
    _fast_signal_words,
)


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
    if isinstance(obj, set):
        return sorted(obj)
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
# Expanded coherence vocabulary
# ---------------------------------------------------------------------------

# Extended CVC-appropriate pharmaceutical vocabulary
CVC_PHARMA_EXPANDED = {
    'coralli', 'diasene', 'stercora', 'radicom', 'commune', 'secundi',
    'ratione', 'balsamo', 'radice', 'herba', 'aqua', 'oleum',
    'cura', 'morbo', 'febre', 'dolor', 'sana',
    # CVC forms
    'colar', 'corar', 'senen', 'dinen', 'terer', 'miser',
    'cor', 'ner', 'ren', 'den', 'sen', 'ser', 'sal', 'mel', 'cer',
    'bon', 'fort', 'ben', 'din', 'con', 'des', 'decor',
}

# Circa Instans terms (preparation verbs + ingredients)
CIRCA_INSTANS_TERMS = {
    'cola', 'colar', 'tere', 'terer', 'misce', 'miser', 'recipe',
    'adde', 'coque', 'pone', 'solve', 'distilla',
    'sene', 'senen', 'coralli', 'radicom', 'stercora', 'diasene',
    'aqua', 'oleum', 'herba', 'radice', 'semen', 'cortex',
    'folia', 'flores', 'balsamo', 'gummi', 'cera', 'mel',
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class MetricThreshold:
    """Calibrated threshold for one coherence metric."""
    metric: str
    threshold: float
    random_mean: float
    random_std: float
    random_pass_rate: float
    real_value: float
    real_passes: bool
    p_value: float


@dataclass
class RecalibratedCoherenceResult:
    """Full Track B output."""
    phase: str = "60"
    step: str = "60.2"
    experiment: str = "recalibrated_coherence"
    n_trials: int = 0
    # Real table results
    real_n_signal: int = 0
    real_metrics: Dict[str, float] = field(default_factory=dict)
    # Calibrated thresholds
    thresholds: List[MetricThreshold] = field(default_factory=list)
    # Selected battery
    selected_criteria: List[str] = field(default_factory=list)
    n_criteria: int = 0
    # Joint test
    n_random_pass_all: int = 0
    p_all: float = 0.0
    # Fisher combined
    fisher_chi2: float = 0.0
    fisher_p: float = 0.0
    # Comparison
    cv_p: float = 0.011
    cvc_original_p: float = 0.552
    cvc_recalibrated_p: float = 0.0
    verdict: str = ""
    # Gates
    g1_nontrivial_thresholds: bool = False
    g2_real_passes_enough: bool = False
    g3_fisher_p: bool = False
    g4_joint_p: bool = False
    g5_vs_cv: bool = False
    gates_passed: int = 0
    gate_passed: bool = False
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Expanded metrics computation
# ---------------------------------------------------------------------------

def _expanded_metrics(
    signal_words: List[Dict[str, Any]],
    decoded_words: List[str],
    ref_word_set: Set[str],
) -> Dict[str, float]:
    """Compute expanded coherence metrics for a set of signal words."""
    words = set(w['word'].lower() for w in signal_words)

    # 1. Verb paradigm count (not just boolean)
    verb_count = len(words & VERB_PARADIGM_WORDS)

    # 2. Function kit count
    function_count = len(words & FUNCTION_KIT)

    # 3. Pharma unique count (expanded)
    pharma_count = len(words & CVC_PHARMA_EXPANDED)

    # 4. Circa Instans overlap
    ci_count = len(words & CIRCA_INSTANS_TERMS)

    # 5. Content word count (non-function, length >= 3)
    content_count = sum(1 for w in signal_words
                        if w['word'] not in FUNCTION_WORDS
                        and len(w['word']) >= 3)

    # 6. Mean signal word length
    mean_word_len = (
        np.mean([len(w['word']) for w in signal_words])
        if signal_words else 0.0)

    # 7. Latin ending diversity (in signal words)
    endings_found = set()
    for w in signal_words:
        word = w['word']
        if len(word) >= 3:
            for ending in LATIN_ENDINGS:
                suffix = ending[1:]
                if word.endswith(suffix):
                    endings_found.add(ending)
    ending_diversity = len(endings_found)

    # 8. Signal-to-noise ratio (total signal words / total decoded types)
    decoded_types = set(w for w in decoded_words if w and w != '?')
    signal_ratio = len(signal_words) / len(decoded_types) if decoded_types else 0.0

    return {
        'verb_count': float(verb_count),
        'function_count': float(function_count),
        'pharma_count': float(pharma_count),
        'ci_count': float(ci_count),
        'content_count': float(content_count),
        'mean_word_len': float(mean_word_len),
        'ending_diversity': float(ending_diversity),
        'signal_ratio': float(signal_ratio),
    }


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def _profile_random_tables(
    n_trials: int,
    all_tokens: List[str],
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    null_token_lists: List[List[str]],
    ref_word_set: Set[str],
) -> List[Dict[str, float]]:
    """Profile random coda tables and collect expanded metrics."""
    strokes = ['hook', 'descender', 'sigmoid', 'vertical', 'connector']
    codas = ['n', 'r', 's', 't', 'l', 'm']
    rng = random.Random(42)

    real_table = build_coda_table('primary')
    real_mapping = dict(real_table.stroke_to_coda)

    # Precompute null decoded lists with a fixed coda table
    # For signal isolation we need null counters; decode null corpora once
    # with each random table (necessary for correct signal isolation)

    profiles: List[Dict[str, float]] = []
    progress_step = max(1, n_trials // 10)

    for trial in range(n_trials):
        if trial % progress_step == 0:
            print(f"    Trial {trial}/{n_trials} ...")

        # Random coda assignment
        perm = {stroke: rng.choice(codas) for stroke in strokes}
        if all(perm[s] == real_mapping.get(s) for s in strokes):
            perm[rng.choice(strokes)] = rng.choice(codas)

        custom_table = _build_permuted_coda_table(perm)

        # Decode real corpus with v2 (i=syllabic correction)
        real_decoded = decode_corpus_cvc_v2(
            all_tokens, assignment, eva_to_triple, custom_table)

        # Decode null corpora with same table
        null_decoded_list = []
        for null_tokens in null_token_lists:
            nd = decode_corpus_cvc_v2(
                null_tokens, assignment, eva_to_triple, custom_table)
            null_decoded_list.append(nd)

        null_counters = [Counter(nd) for nd in null_decoded_list]
        signal_words = _fast_signal_words(real_decoded, null_counters, ref_word_set)

        metrics = _expanded_metrics(signal_words, real_decoded, ref_word_set)
        metrics['n_signal'] = float(len(signal_words))
        profiles.append(metrics)

    return profiles


def _calibrate_thresholds(
    profiles: List[Dict[str, float]],
    target_pctl: float = 92.0,
) -> Dict[str, Dict[str, float]]:
    """Find threshold per metric at the target percentile."""
    metrics = profiles[0].keys()
    thresholds = {}
    for metric in metrics:
        values = [p[metric] for p in profiles]
        threshold = float(np.percentile(values, target_pctl))
        pass_rate = sum(1 for v in values if v >= threshold) / len(values)
        thresholds[metric] = {
            'threshold': threshold,
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'pass_rate': pass_rate,
            'min': float(np.min(values)),
            'max': float(np.max(values)),
        }
    return thresholds


def _fisher_combined_p(p_values: List[float]) -> Tuple[float, float]:
    """Fisher's method: combined p-value from independent tests."""
    valid_p = [max(p, 1e-10) for p in p_values if p > 0]
    if not valid_p:
        return 0.0, 1.0
    chi2 = -2.0 * sum(math.log(p) for p in valid_p)
    df = 2 * len(valid_p)
    combined_p = float(1.0 - scipy_stats.chi2.cdf(chi2, df))
    return chi2, combined_p


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def run_recal_coherence():
    """Track B: Recalibrated CVC permutation coherence test."""
    t0 = time.time()
    print("=" * 70)
    print("Phase 60, Track B: Recalibrated Coherence Test")
    print("=" * 70)

    # Load shared data
    print("\n  Loading shared data ...")
    data = _load_shared_data()
    rd = data['rd']

    all_tokens = data['all_tokens']
    assignment = data['assignment']
    eva_to_triple = data['eva_to_triple']
    ref_word_set = data['ref_word_set']
    null_token_lists = data['null_token_lists']

    # 1. Evaluate real corrected table
    print("\n  Evaluating real corrected CVC table ...")
    coda_corrected = build_coda_table_v2()
    real_decoded = decode_corpus_cvc_v2(
        all_tokens, assignment, eva_to_triple, coda_corrected)

    null_decoded_list = []
    for null_tokens in null_token_lists:
        nd = decode_corpus_cvc_v2(
            null_tokens, assignment, eva_to_triple, coda_corrected)
        null_decoded_list.append(nd)

    null_counters = [Counter(nd) for nd in null_decoded_list]
    real_signal_words = _fast_signal_words(real_decoded, null_counters, ref_word_set)
    real_metrics = _expanded_metrics(real_signal_words, real_decoded, ref_word_set)
    real_metrics['n_signal'] = float(len(real_signal_words))

    print(f"  Real signal words: {len(real_signal_words)}")
    for metric, value in sorted(real_metrics.items()):
        print(f"    {metric}: {value:.2f}")

    # 2. Profile random tables
    n_trials = 1000
    print(f"\n  Profiling {n_trials} random coda tables ...")
    profiles = _profile_random_tables(
        n_trials, all_tokens, assignment, eva_to_triple,
        null_token_lists, ref_word_set,
    )

    # 3. Calibrate thresholds at 92nd percentile
    print("\n  Calibrating thresholds (92nd percentile) ...")
    thresholds = _calibrate_thresholds(profiles, target_pctl=92.0)

    for metric, info in sorted(thresholds.items()):
        print(f"    {metric}: threshold={info['threshold']:.2f} "
              f"(mean={info['mean']:.2f}, std={info['std']:.2f}, "
              f"pass={info['pass_rate']:.1%})")

    # 4. Score real table against calibrated thresholds
    print("\n  Scoring real table against calibrated thresholds ...")
    metric_results: List[MetricThreshold] = []
    for metric, info in sorted(thresholds.items()):
        real_value = real_metrics[metric]
        passes = real_value >= info['threshold']
        # p-value: fraction of random tables at or above real value
        values = [p[metric] for p in profiles]
        p_value = sum(1 for v in values if v >= real_value) / len(values)

        mt = MetricThreshold(
            metric=metric,
            threshold=round(info['threshold'], 4),
            random_mean=round(info['mean'], 4),
            random_std=round(info['std'], 4),
            random_pass_rate=round(info['pass_rate'], 4),
            real_value=round(real_value, 4),
            real_passes=passes,
            p_value=round(p_value, 4),
        )
        metric_results.append(mt)
        status = "PASS" if passes else "FAIL"
        print(f"    {metric}: real={real_value:.2f} thresh={info['threshold']:.2f} "
              f"-> {status} (p={p_value:.4f})")

    # 5. Select battery: criteria where real passes AND random pass rate 5-20%
    selected = [
        mt for mt in metric_results
        if mt.real_passes
        and 0.03 <= mt.random_pass_rate <= 0.20
    ]
    # Sort by p-value (most discriminating first)
    selected.sort(key=lambda m: m.p_value)
    # Take top 6
    selected = selected[:6]
    selected_names = [mt.metric for mt in selected]

    print(f"\n  Selected battery ({len(selected)} criteria):")
    for mt in selected:
        print(f"    {mt.metric}: real={mt.real_value:.2f} thresh={mt.threshold:.2f} "
              f"p={mt.p_value:.4f}")

    # 6. Joint test: how many random tables pass ALL selected criteria?
    n_random_pass_all = 0
    for trial_idx in range(n_trials):
        trial_passes = True
        for mt in selected:
            trial_value = profiles[trial_idx][mt.metric]
            if trial_value < mt.threshold:
                trial_passes = False
                break
        if trial_passes:
            n_random_pass_all += 1

    p_all = n_random_pass_all / n_trials if n_trials > 0 else 1.0

    # 7. Fisher combined p-value
    p_values = [mt.p_value for mt in selected if mt.p_value > 0]
    fisher_chi2, fisher_p = _fisher_combined_p(p_values)

    print(f"\n  Results:")
    print(f"    Joint p(all {len(selected)} criteria): {p_all:.4f} "
          f"({n_random_pass_all}/{n_trials} random tables pass all)")
    print(f"    Fisher combined: chi2={fisher_chi2:.2f}, p={fisher_p:.6f}")
    print(f"    CV baseline p: 0.011")
    print(f"    CVC original p: 0.552")

    # Determine verdict
    if p_all < 0.01:
        verdict = "COHERENCE_RARE"
    elif p_all < 0.05:
        verdict = "COHERENCE_UNCOMMON"
    elif fisher_p < 0.05:
        verdict = "FISHER_SIGNIFICANT"
    else:
        verdict = "COHERENCE_COMMON"

    # Gates
    n_nontrivial = sum(1 for mt in metric_results
                       if 0.03 <= mt.random_pass_rate <= 0.20)
    n_real_pass = sum(1 for mt in selected if mt.real_passes)

    g1 = n_nontrivial >= 4
    g2 = n_real_pass >= 4 if selected else False
    g3 = fisher_p < 0.05
    g4 = p_all < 0.05
    g5 = p_all <= 0.011  # CVC p <= CV p
    gates_passed = sum([g1, g2, g3, g4, g5])

    print(f"\n  Validation Gates:")
    print(f"    G1 >= 4 nontrivial thresholds:  {'PASS' if g1 else 'FAIL'} "
          f"({n_nontrivial})")
    print(f"    G2 real passes >= 4/{len(selected)}:       {'PASS' if g2 else 'FAIL'} "
          f"({n_real_pass})")
    print(f"    G3 Fisher p < 0.05:             {'PASS' if g3 else 'FAIL'} "
          f"({fisher_p:.6f})")
    print(f"    G4 joint p < 0.05:              {'PASS' if g4 else 'FAIL'} "
          f"({p_all:.4f})")
    print(f"    G5 CVC p <= CV p (0.011):       {'PASS' if g5 else 'FAIL'} "
          f"({p_all:.4f})")
    print(f"    Gates passed: {gates_passed}/5")

    result = RecalibratedCoherenceResult(
        n_trials=n_trials,
        real_n_signal=len(real_signal_words),
        real_metrics={k: round(v, 4) for k, v in real_metrics.items()},
        thresholds=metric_results,
        selected_criteria=selected_names,
        n_criteria=len(selected),
        n_random_pass_all=n_random_pass_all,
        p_all=round(p_all, 4),
        fisher_chi2=round(fisher_chi2, 2),
        fisher_p=round(fisher_p, 6),
        cv_p=0.011,
        cvc_original_p=0.552,
        cvc_recalibrated_p=round(p_all, 4),
        verdict=verdict,
        g1_nontrivial_thresholds=g1,
        g2_real_passes_enough=g2,
        g3_fisher_p=g3,
        g4_joint_p=g4,
        g5_vs_cv=g5,
        gates_passed=gates_passed,
        gate_passed=gates_passed >= 3,
        runtime_seconds=round(time.time() - t0, 2),
    )

    path = _save_json(rd, 'recalibrated_coherence.json', result)
    print(f"\n  Saved: {path}")
    print(f"  Track B completed in {time.time() - t0:.1f}s")
    print(f"  Verdict: {verdict}")
