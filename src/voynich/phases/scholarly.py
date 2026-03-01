"""
Workstream G: Scholarly Validation Framework
==============================================
Pre-registration, comprehensive null testing, effect size reporting,
reproducibility packaging, and sensitivity analysis.

Components:
  G.1 — Pre-registration of hypotheses and thresholds
  G.2 — Null testing harness (wraps any metric)
  G.3 — Effect size and power reporting
  G.4 — Reproducibility package generation
  G.5 — Sensitivity analysis
"""

import hashlib
import json
import math
import os
import platform
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from voynich.core.corpus import load_corpus, VoynichCorpus, tokenize_eva_chars
from voynich.core.stats import cohens_d, log_bayes_factor, bootstrap_ci, first_order_entropy
from voynich.core._paths import results_dir as _results_dir
from voynich.analysis.fingerprint import generate_null_text


# ---------------------------------------------------------------------------
# G.1: Pre-Registration
# ---------------------------------------------------------------------------

@dataclass
class Hypothesis:
    """A pre-registered hypothesis with success criteria."""
    id: str
    description: str
    metric: str
    direction: str       # 'lower', 'higher', or 'different'
    threshold: float
    alpha: float = 0.05
    result: Optional[float] = None
    passed: Optional[bool] = None
    notes: str = ''


PHASE3_HYPOTHESES: List[Hypothesis] = [
    Hypothesis(
        id='D1',
        description='Voynich token length distribution is closer to Latin syllable-count '
                    'distribution than to Latin character-count distribution',
        metric='emd_voynich_vs_syl < emd_voynich_vs_char',
        direction='lower',
        threshold=0.0,
    ),
    Hypothesis(
        id='D2',
        description='Optimal permutation mapping under syllabary model yields lower '
                    'Frobenius distance than under substitution model',
        metric='frobenius_syllabary < frobenius_substitution',
        direction='lower',
        threshold=0.0,
    ),
    Hypothesis(
        id='D3',
        description='DTW distance of Voynich entropy curve to Latin syllable curve '
                    'is lower than to Latin character curve',
        metric='dtw_voynich_vs_syl < dtw_voynich_vs_char',
        direction='lower',
        threshold=0.0,
    ),
    Hypothesis(
        id='E1',
        description='Grid gap pattern is non-random (chi-squared p < 0.05)',
        metric='chi2_pvalue',
        direction='lower',
        threshold=0.05,
    ),
    Hypothesis(
        id='E3',
        description='Grid is stable under 50% subsampling (>80% of cells stable in >90% of iterations)',
        metric='stable_cells_fraction',
        direction='higher',
        threshold=0.80,
    ),
    Hypothesis(
        id='F3',
        description='Latin yields lowest distance among candidate languages '
                    'in syllable bigram matching',
        metric='best_language_is_latin',
        direction='higher',
        threshold=0.5,
    ),
    Hypothesis(
        id='F4',
        description='PMI correlation between Voynich and Latin syllable sequences '
                    'is positive and significant',
        metric='pmi_correlation',
        direction='higher',
        threshold=0.0,
    ),
]


def pre_register_hypotheses() -> List[Hypothesis]:
    """Return the list of pre-registered hypotheses (frozen before analysis)."""
    return [Hypothesis(**asdict(h)) for h in PHASE3_HYPOTHESES]


def evaluate_hypotheses(results: Dict) -> List[Hypothesis]:
    """Evaluate pre-registered hypotheses against actual results."""
    hypotheses = pre_register_hypotheses()

    for h in hypotheses:
        if h.id == 'D1' and 'degeneracy_length.json' in _list_results():
            d1 = _load_result('degeneracy_length.json')
            h.result = d1.get('emd_voynich_vs_syl', 0) - d1.get('emd_voynich_vs_char', 0)
            h.passed = d1.get('emd_voynich_vs_syl', 1) < d1.get('emd_voynich_vs_char', 0)
            h.notes = f"EMD syl={d1.get('emd_voynich_vs_syl', 'N/A')}, " \
                      f"EMD char={d1.get('emd_voynich_vs_char', 'N/A')}"

        elif h.id == 'D2' and 'degeneracy_bigram.json' in _list_results():
            d2 = _load_result('degeneracy_bigram.json')
            h.result = d2.get('frobenius_syllabary', 0) - d2.get('frobenius_substitution', 0)
            h.passed = d2.get('frobenius_syllabary', 1) < d2.get('frobenius_substitution', 0)

        elif h.id == 'D3' and 'degeneracy_positional.json' in _list_results():
            d3 = _load_result('degeneracy_positional.json')
            h.result = d3.get('dtw_voynich_vs_syl', 0) - d3.get('dtw_voynich_vs_char', 0)
            h.passed = d3.get('dtw_voynich_vs_syl', 1) < d3.get('dtw_voynich_vs_char', 0)

        elif h.id == 'E1' and 'grid_gaps.json' in _list_results():
            e1 = _load_result('grid_gaps.json')
            h.result = e1.get('chi2_pvalue', 1.0)
            h.passed = e1.get('chi2_pvalue', 1.0) < h.threshold

        elif h.id == 'E3' and 'grid_stability.json' in _list_results():
            e3 = _load_result('grid_stability.json')
            n_cells = len(e3.get('full_grid_cells', []))
            n_stable = e3.get('stable_cells', 0)
            frac = n_stable / n_cells if n_cells > 0 else 0
            h.result = frac
            h.passed = frac >= h.threshold

        elif h.id == 'F3' and 'syllable_language_ranking.json' in _list_results():
            f3 = _load_result('syllable_language_ranking.json')
            if f3 and len(f3) > 0:
                best = f3[0]
                h.result = 1.0 if best.get('language') == 'latin' else 0.0
                h.passed = best.get('language') == 'latin'
                h.notes = f"Best: {best.get('language')}"

        elif h.id == 'F4' and 'syllable_pmi.json' in _list_results():
            f4 = _load_result('syllable_pmi.json')
            h.result = f4.get('pmi_correlation', 0)
            h.passed = f4.get('significant', False) and f4.get('pmi_correlation', 0) > 0

    return hypotheses


def _list_results() -> List[str]:
    """List available result files."""
    rd = _results_dir()
    if not os.path.isdir(rd):
        return []
    return os.listdir(rd)


def _load_result(filename: str) -> Any:
    """Load a result JSON file."""
    path = os.path.join(_results_dir(), filename)
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# G.2: Null Testing Harness
# ---------------------------------------------------------------------------

@dataclass
class NullTestResult:
    """Result of testing a metric against null text variants."""
    metric_name: str
    real_value: float
    null_type: str
    null_mean: float
    null_std: float
    z_score: float
    cohens_d_value: float
    p_value_empirical: float
    selectivity: float
    discriminates: bool


def null_test_harness(
    metric_fn: Callable[[List[str], str], float],
    metric_name: str,
    real_tokens: List[str],
    real_text: str,
    null_types: Optional[List[str]] = None,
    n_trials: int = 20,
    seed: int = 42,
) -> Dict[str, NullTestResult]:
    """
    Generic null testing harness.

    Takes any metric function f(tokens, text) -> float, generates null text
    using each null type, computes the metric on each, and reports statistics.
    """
    if null_types is None:
        null_types = ['shuffle', 'random', 'markov', 'token_shuffle']

    real_value = metric_fn(real_tokens, real_text)
    results = {}

    for null_type in null_types:
        null_values = []
        rng = random.Random(seed)

        for trial in range(n_trials):
            trial_seed = seed + trial

            if null_type == 'token_shuffle':
                shuffled = list(real_tokens)
                rng.shuffle(shuffled)
                null_text = ' '.join(shuffled)
                null_tokens = shuffled
            else:
                null_text = generate_null_text(real_tokens, method=null_type,
                                               seed=trial_seed)
                null_tokens = null_text.split()

            null_values.append(metric_fn(null_tokens, null_text))

        null_arr = np.array(null_values)
        null_mean = float(np.mean(null_arr))
        null_std = float(np.std(null_arr))

        z = (real_value - null_mean) / null_std if null_std > 0 else 0.0
        d = cohens_d(np.array([real_value]), null_arr)
        p_emp = sum(1 for v in null_values
                    if abs(v - null_mean) >= abs(real_value - null_mean)) / n_trials
        selectivity = abs(real_value) / abs(null_mean) if null_mean != 0 else 0

        results[null_type] = NullTestResult(
            metric_name=metric_name,
            real_value=float(real_value),
            null_type=null_type,
            null_mean=float(null_mean),
            null_std=float(null_std),
            z_score=float(z),
            cohens_d_value=float(d),
            p_value_empirical=float(p_emp),
            selectivity=float(selectivity),
            discriminates=bool(abs(z) > 2.0),
        )

    return results


def comprehensive_null_test(
    tokens: List[str],
    text: str,
    n_trials: int = 20,
) -> Dict[str, Dict[str, NullTestResult]]:
    """Run null testing on key metrics from Phase 3."""
    # Define metrics
    metrics = {
        'H1': lambda t, txt: first_order_entropy(txt),
        'H2': lambda t, txt: __import__('voynich.core.stats', fromlist=['conditional_entropy']).conditional_entropy(txt, order=1),
        'word_H1': lambda t, txt: __import__('voynich.core.stats', fromlist=['word_unigram_entropy']).word_unigram_entropy(t),
        'mean_word_length': lambda t, txt: float(np.mean([len(w) for w in t])) if t else 0,
        'zipf_exponent': lambda t, txt: __import__('voynich.core.stats', fromlist=['zipf_analysis']).zipf_analysis(t).get('zipf_exponent', 0),
    }

    results = {}
    for name, fn in metrics.items():
        results[name] = null_test_harness(fn, name, tokens, text,
                                          n_trials=n_trials)

    return results


# ---------------------------------------------------------------------------
# G.3: Effect Size and Power Reporting
# ---------------------------------------------------------------------------

@dataclass
class EffectReport:
    """Comprehensive effect size report for one metric."""
    metric_name: str
    point_estimate: float
    ci_lower: float
    ci_upper: float
    cohens_d_value: float
    log_bf: float
    interpretation: str


def compute_effect_report(
    metric_name: str,
    real_value: float,
    null_values: List[float],
) -> EffectReport:
    """Compute Cohen's d, bootstrap CI, and Bayes factor for one metric."""
    null_arr = np.array(null_values)
    null_mean = float(np.mean(null_arr))
    null_std = float(np.std(null_arr))

    d = cohens_d(np.array([real_value]), null_arr)
    bf = log_bayes_factor(null_arr, null_mean, real_value, null_std)

    # Bootstrap CI on the null distribution
    _, lo, hi = bootstrap_ci(null_arr, np.mean, n_bootstrap=500)

    # Interpretation
    abs_d = abs(d)
    if abs_d < 0.2:
        interp = 'negligible'
    elif abs_d < 0.5:
        interp = 'small'
    elif abs_d < 0.8:
        interp = 'medium'
    else:
        interp = 'large'

    return EffectReport(
        metric_name=metric_name,
        point_estimate=real_value,
        ci_lower=lo,
        ci_upper=hi,
        cohens_d_value=d,
        log_bf=bf,
        interpretation=interp,
    )


# ---------------------------------------------------------------------------
# G.4: Reproducibility Package
# ---------------------------------------------------------------------------

def _file_sha256(filepath: str) -> str:
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    try:
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                h.update(chunk)
        return h.hexdigest()
    except FileNotFoundError:
        return 'file_not_found'


def generate_reproducibility_manifest(
    results_dir: str = None,
    data_dir: str = None,
) -> Dict:
    """Generate a reproducibility manifest."""
    if results_dir is None:
        results_dir = str(_results_dir())
    if data_dir is None:
        from voynich.core._paths import data_dir as _data_dir
        data_dir = str(_data_dir())
    manifest = {
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'python_version': sys.version,
        'platform': platform.platform(),
        'numpy_version': np.__version__,
    }

    try:
        import scipy
        manifest['scipy_version'] = scipy.__version__
    except ImportError:
        manifest['scipy_version'] = 'not installed'

    # Random seeds used
    manifest['random_seeds'] = {
        'primary': 42,
        'bootstrap': 42,
        'null_testing': 42,
    }

    # Data file hashes
    data_hashes = {}
    for root, dirs, files in os.walk(data_dir):
        for f in sorted(files):
            path = os.path.join(root, f)
            rel_path = os.path.relpath(path, '.')
            data_hashes[rel_path] = _file_sha256(path)
    manifest['data_hashes'] = data_hashes

    # Result file hashes
    result_hashes = {}
    if os.path.isdir(results_dir):
        for f in sorted(os.listdir(results_dir)):
            path = os.path.join(results_dir, f)
            result_hashes[f] = _file_sha256(path)
    manifest['result_hashes'] = result_hashes

    return manifest


# ---------------------------------------------------------------------------
# G.5: Sensitivity Analysis
# ---------------------------------------------------------------------------

@dataclass
class SensitivityResult:
    """Result of varying a parameter and observing metric change."""
    parameter_name: str
    parameter_values: List[Any]
    metric_values: List[float]
    metric_name: str
    baseline_value: float
    sensitivity: float       # max deviation / baseline
    conclusion_robust: bool  # qualitative conclusion unchanged?


def sensitivity_sweep(
    parameter_name: str,
    parameter_values: List[Any],
    experiment_fn: Callable[[Any], float],
    metric_name: str,
) -> SensitivityResult:
    """
    Run an experiment at multiple parameter values, track metric variation.
    """
    metric_values = []
    for val in parameter_values:
        try:
            metric_values.append(experiment_fn(val))
        except Exception:
            metric_values.append(float('nan'))

    baseline = metric_values[len(metric_values) // 2]
    max_dev = max(abs(v - baseline) for v in metric_values
                  if not math.isnan(v))
    sensitivity = max_dev / abs(baseline) if baseline != 0 else 0

    return SensitivityResult(
        parameter_name=parameter_name,
        parameter_values=parameter_values,
        metric_values=metric_values,
        metric_name=metric_name,
        baseline_value=baseline,
        sensitivity=sensitivity,
        conclusion_robust=sensitivity < 0.20,
    )


def run_sensitivity_analyses(corpus: VoynichCorpus) -> List[SensitivityResult]:
    """Run standard sensitivity analyses."""
    tokens = corpus.get_tokens(paragraph_only=True)

    results = []

    # Sensitivity to grid nucleus cluster count
    from voynich.phases.grid_validate import build_grid_from_tokens
    from voynich.analysis.strokes import syllable_sequence_stats

    def _grid_occupancy(n_nuclei):
        grid = build_grid_from_tokens(tokens, n_nucleus_clusters=n_nuclei)
        return grid.occupancy

    results.append(sensitivity_sweep(
        'n_nucleus_clusters', [4, 5, 6, 7, 8],
        _grid_occupancy, 'grid_occupancy',
    ))

    # Sensitivity to subsample size for H1
    from voynich.core.stats import first_order_entropy

    def _h1_at_n(n_tokens):
        subset = tokens[:n_tokens]
        text = ' '.join(subset)
        return first_order_entropy(text)

    n_values = [1000, 5000, 10000, 20000, len(tokens)]
    n_values = [n for n in n_values if n <= len(tokens)]
    results.append(sensitivity_sweep(
        'corpus_size', n_values,
        _h1_at_n, 'H1',
    ))

    return results


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_scholarly_validation() -> Dict:
    """Run all Workstream G validation and print/save results."""
    rd = _results_dir()

    print("=" * 70)
    print("WORKSTREAM G: SCHOLARLY VALIDATION FRAMEWORK")
    print("=" * 70)

    corpus = load_corpus(verbose=False)
    tokens = corpus.get_tokens(paragraph_only=True)
    text = ' '.join(tokens)

    # G.1: Pre-Registration
    print("\n--- G.1: Pre-Registered Hypotheses ---")
    hypotheses = evaluate_hypotheses({})
    print(f"  {'ID':<5} {'Passed':>6} {'Result':>10} {'Description'}")
    print(f"  {'-' * 65}")
    for h in hypotheses:
        passed_str = 'YES' if h.passed else ('NO' if h.passed is not None else 'N/A')
        result_str = f'{h.result:.4f}' if h.result is not None else 'N/A'
        print(f"  {h.id:<5} {passed_str:>6} {result_str:>10} {h.description[:50]}")

    with open(os.path.join(rd, 'hypotheses_preregistered.json'), 'w') as f:
        json.dump([asdict(h) for h in hypotheses], f, indent=2)

    # G.2: Comprehensive Null Testing
    print("\n--- G.2: Comprehensive Null Testing ---")
    print("  Running null tests on key metrics (20 trials each)...")
    null_results = comprehensive_null_test(tokens, text, n_trials=20)

    print(f"\n  {'Metric':<20} {'Real':>8} {'Null Type':<15} "
          f"{'Null Mean':>10} {'z-score':>8} {'Disc':>5}")
    print(f"  {'-' * 68}")
    for metric_name, type_results in null_results.items():
        for null_type, result in type_results.items():
            disc = 'YES' if result.discriminates else 'no'
            print(f"  {metric_name:<20} {result.real_value:>8.4f} "
                  f"{null_type:<15} {result.null_mean:>10.4f} "
                  f"{result.z_score:>8.2f} {disc:>5}")

    # Serialize
    null_data = {}
    for metric_name, type_results in null_results.items():
        null_data[metric_name] = {
            nt: asdict(nr) for nt, nr in type_results.items()
        }
    with open(os.path.join(rd, 'null_test_results.json'), 'w') as f:
        json.dump(null_data, f, indent=2)

    # G.3: Effect Sizes
    print("\n--- G.3: Effect Size Reports ---")
    effect_reports = []
    for metric_name, type_results in null_results.items():
        for null_type, result in type_results.items():
            if null_type == 'shuffle':  # Report against shuffle baseline
                # Reconstruct null values from z and std
                null_mean = result.null_mean
                null_std = result.null_std
                null_values = list(np.random.normal(null_mean, null_std, 20))
                report = compute_effect_report(metric_name, result.real_value,
                                               null_values)
                effect_reports.append(report)

    print(f"  {'Metric':<20} {'Estimate':>10} {'CI':>20} "
          f"{'d':>6} {'Interp':>10}")
    print(f"  {'-' * 68}")
    for r in effect_reports:
        ci_str = f"[{r.ci_lower:.4f}, {r.ci_upper:.4f}]"
        print(f"  {r.metric_name:<20} {r.point_estimate:>10.4f} {ci_str:>20} "
              f"{r.cohens_d_value:>6.2f} {r.interpretation:>10}")

    with open(os.path.join(rd, 'effect_sizes.json'), 'w') as f:
        json.dump([asdict(r) for r in effect_reports], f, indent=2)

    # G.4: Reproducibility Manifest
    print("\n--- G.4: Reproducibility Manifest ---")
    manifest = generate_reproducibility_manifest()
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  NumPy: {manifest['numpy_version']}")
    print(f"  SciPy: {manifest['scipy_version']}")
    print(f"  Data files: {len(manifest['data_hashes'])}")
    print(f"  Result files: {len(manifest['result_hashes'])}")

    with open(os.path.join(rd, 'reproducibility_manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)

    # G.5: Sensitivity Analysis
    print("\n--- G.5: Sensitivity Analysis ---")
    sensitivities = run_sensitivity_analyses(corpus)
    for s in sensitivities:
        print(f"  {s.parameter_name}: sensitivity={s.sensitivity:.4f}, "
              f"robust={s.conclusion_robust}")
        for val, metric in zip(s.parameter_values, s.metric_values):
            print(f"    {val}: {metric:.4f}")

    with open(os.path.join(rd, 'sensitivity.json'), 'w') as f:
        json.dump([asdict(s) for s in sensitivities], f, indent=2)

    # Summary
    n_passed = sum(1 for h in hypotheses if h.passed)
    n_tested = sum(1 for h in hypotheses if h.passed is not None)
    n_disc = sum(1 for results in null_results.values()
                 for r in results.values() if r.discriminates)
    n_total_null = sum(len(results) for results in null_results.values())

    print(f"\n{'=' * 70}")
    print("WORKSTREAM G SUMMARY")
    print(f"  Hypotheses: {n_passed}/{n_tested} passed")
    print(f"  Null discrimination: {n_disc}/{n_total_null} metrics discriminate")
    print(f"  Sensitivity: {sum(1 for s in sensitivities if s.conclusion_robust)}"
          f"/{len(sensitivities)} robust")
    print(f"{'=' * 70}")

    return {
        'hypotheses': [asdict(h) for h in hypotheses],
        'null_tests': null_data,
        'effect_sizes': [asdict(r) for r in effect_reports],
        'manifest': manifest,
        'sensitivity': [asdict(s) for s in sensitivities],
    }
