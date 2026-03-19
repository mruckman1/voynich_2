"""
Phase 54 Integration: Gallo-Italic Dialect Identification Verdict
=================================================================
Combine results from all 8 Phase 54 experiments (54.1–54.8) to produce
a composite dialect identification verdict with bootstrap confidence
intervals and Fisher's combined significance test.

Dependency chain:
    phase54_degemination.json    (54.1)
    phase54_lenition.json        (54.2)
    phase54_articles.json        (54.3)
    phase54_pharma_region.json   (54.4)
    phase54_co_syntax.json       (54.5)
    phase54_verb_morph.json      (54.6)
    phase54_dialect_sim.json     (54.7)
    phase54_zodiac.json          (54.8)
        -> phase54_integrate.json
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from voynich.core._paths import results_dir as _results_dir


# ---------------------------------------------------------------------------
# Helpers
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


def _save_json(rd: str, filename: str, data: Any) -> str:
    path = os.path.join(rd, filename)
    with open(path, 'w') as f:
        json.dump(_convert(data), f, indent=2)
    return path


def _load_exp(filename: str) -> Optional[Dict]:
    path = os.path.join(_results_dir(), filename)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXPERIMENT_FILES = [
    ('54.1', 'degemination', 'phase54_degemination.json'),
    ('54.2', 'lenition_pattern', 'phase54_lenition.json'),
    ('54.3', 'article_pronoun_system', 'phase54_articles.json'),
    ('54.4', 'pharma_regionalization', 'phase54_pharma_region.json'),
    ('54.5', 'co_syntactic_validation', 'phase54_co_syntax.json'),
    ('54.6', 'verb_morphology', 'phase54_verb_morph.json'),
    ('54.7', 'simulated_macaronic', 'phase54_dialect_sim.json'),
    ('54.8', 'zodiac_dialect_decode', 'phase54_zodiac.json'),
]

DIALECTS = ['venetian', 'lombard', 'ligurian', 'emilian', 'tuscan']


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DialectIntegrateResult:
    phase: str
    experiment: str
    n_experiments_run: int
    n_experiments_gated: int
    per_experiment_results: List[Dict]
    composite_dialect_scores: Dict[str, float]
    ranking: List[Dict]
    fisher_chi2: float
    fisher_p: float
    agreement_rate: float
    consistency_check: bool
    validations: Dict[str, bool]
    n_validations_passed: int
    verdict: str
    summary: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_dialect_verdict() -> None:
    """Phase 54 Integration: dialect identification verdict."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE 54: Gallo-Italic Dialect Identification \u2014 Integration")
    print("=" * 70)

    rd = _results_dir()

    # ── Step 1: Load all experiment results ────────────────────────────
    print("\n  Loading experiment results...")

    experiments: List[Dict] = []
    for phase_id, exp_name, filename in EXPERIMENT_FILES:
        path = os.path.join(rd, filename)
        if not os.path.exists(path):
            experiments.append({
                'phase': phase_id,
                'experiment': exp_name,
                'status': 'NOT_RUN',
                'dialect_scores': None,
                'gates': {},
                'z_score': None,
                'selectivity': None,
                'verdict': 'NOT_RUN',
            })
            continue
        with open(path) as f:
            data = json.load(f)

        # Extract dialect scores
        scores = data.get('dialect_scores', {})

        # Extract gates
        gates = data.get('gates', {})

        # Count gates passed
        gates_passed = sum(1 for v in gates.values() if v) if gates else 0
        total_gates = len(gates) if gates else 0

        # Extract z_score (may be z_score or z_a)
        z = data.get('z_score', data.get('z_a', None))

        # Extract selectivity
        sel = data.get('selectivity', None)

        # Extract p-value: derive from z if needed
        # p ~ 1 - Phi(z) for one-sided test
        if z is not None:
            p_value = 0.5 * (1.0 + math.erf(-z / math.sqrt(2)))
        else:
            p_value = 1.0

        experiments.append({
            'phase': phase_id,
            'experiment': exp_name,
            'status': 'COMPLETED',
            'dialect_scores': scores,
            'gates': gates,
            'gates_passed': gates_passed,
            'total_gates': total_gates,
            'z_score': z,
            'selectivity': sel,
            'p_value': p_value,
            'verdict': data.get('verdict', 'UNKNOWN'),
        })

    n_run = sum(1 for e in experiments if e['status'] == 'COMPLETED')
    print(f"       {n_run} / {len(EXPERIMENT_FILES)} experiments loaded")

    # ── Step 2: Normalize dialect scores ───────────────────────────────
    for exp in experiments:
        if exp['status'] != 'COMPLETED' or not exp['dialect_scores']:
            continue
        scores = exp['dialect_scores']
        total = sum(scores.get(d, 0) for d in DIALECTS)
        if total > 0:
            exp['normalized_scores'] = {d: scores.get(d, 0) / total for d in DIALECTS}
        else:
            exp['normalized_scores'] = {d: 0.2 for d in DIALECTS}

    # ── Step 3: Weight by gate passage ─────────────────────────────────
    for exp in experiments:
        if exp['status'] != 'COMPLETED':
            exp['weight'] = 0.0
            continue
        total_gates = exp.get('total_gates', 0)
        gates_passed = exp.get('gates_passed', 0)
        if total_gates == 0:
            exp['weight'] = 0.5  # no gates defined
        elif gates_passed == total_gates:
            exp['weight'] = 1.0
        elif gates_passed >= total_gates / 2:
            exp['weight'] = 0.5
        else:
            exp['weight'] = 0.0

    # ── Step 4: Aggregate via weighted average ─────────────────────────
    composite_scores = {d: 0.0 for d in DIALECTS}
    total_weight = 0.0

    for exp in experiments:
        if exp['status'] != 'COMPLETED' or exp['weight'] == 0.0:
            continue
        if 'normalized_scores' not in exp:
            continue
        w = exp['weight']
        for d in DIALECTS:
            composite_scores[d] += w * exp['normalized_scores'][d]
        total_weight += w

    if total_weight > 0:
        composite_scores = {d: s / total_weight for d, s in composite_scores.items()}

    n_gated = sum(1 for e in experiments if e['status'] == 'COMPLETED' and e['weight'] > 0)

    # ── Step 5: Bootstrap CIs (1000 iterations, seed 42) ──────────────
    rng = random.Random(42)
    valid_experiments = [
        e for e in experiments
        if e['status'] == 'COMPLETED' and e['weight'] > 0 and 'normalized_scores' in e
    ]

    bootstrap_scores: Dict[str, List[float]] = {d: [] for d in DIALECTS}
    for _ in range(1000):
        sample = rng.choices(valid_experiments, k=len(valid_experiments))
        boot_composite = {d: 0.0 for d in DIALECTS}
        boot_weight = 0.0
        for exp in sample:
            w = exp['weight']
            for d in DIALECTS:
                boot_composite[d] += w * exp['normalized_scores'][d]
            boot_weight += w
        if boot_weight > 0:
            for d in DIALECTS:
                bootstrap_scores[d].append(boot_composite[d] / boot_weight)

    ci_lower: Dict[str, float] = {}
    ci_upper: Dict[str, float] = {}
    for d in DIALECTS:
        vals = sorted(bootstrap_scores[d])
        if vals:
            ci_lower[d] = vals[int(len(vals) * 0.025)]
            ci_upper[d] = vals[int(len(vals) * 0.975)]
        else:
            ci_lower[d] = 0.0
            ci_upper[d] = 0.0

    # ── Step 6: Fisher's combined test ─────────────────────────────────
    p_values = [
        e['p_value'] for e in experiments
        if e['status'] == 'COMPLETED'
        and e.get('p_value') is not None
        and e['p_value'] < 1.0
    ]
    if p_values:
        fisher_chi2 = -2.0 * sum(math.log(max(p, 1e-300)) for p in p_values)
        fisher_df = 2 * len(p_values)
        # Approximate p-value using normal approximation of chi2
        fisher_z = (
            (fisher_chi2 / fisher_df) ** (1 / 3)
            - (1 - 2 / (9 * fisher_df))
        ) / math.sqrt(2 / (9 * fisher_df))
        fisher_p = 0.5 * (1.0 + math.erf(-fisher_z / math.sqrt(2)))
    else:
        fisher_chi2 = 0.0
        fisher_p = 1.0

    # ── Step 7: Cross-experiment agreement ─────────────────────────────
    experiment_winners: List[str] = []
    for exp in experiments:
        if exp['status'] == 'COMPLETED' and exp['weight'] > 0 and 'normalized_scores' in exp:
            winner = max(exp['normalized_scores'], key=exp['normalized_scores'].get)
            experiment_winners.append(winner)

    if experiment_winners:
        most_common = Counter(experiment_winners).most_common(1)[0]
        agreement_rate = most_common[1] / len(experiment_winners)
    else:
        agreement_rate = 0.0

    # ── Step 8: Decision matrix ────────────────────────────────────────
    ranking = sorted(DIALECTS, key=lambda d: composite_scores[d], reverse=True)
    top_score = composite_scores[ranking[0]]
    second_score = composite_scores[ranking[1]] if len(ranking) > 1 else 0
    gap = top_score - second_score

    # Override check
    consistency_ok = agreement_rate >= 0.5

    if not consistency_ok:
        verdict = 'DIALECT_INDETERMINATE'
    elif top_score >= 0.40 and gap >= 0.10 and fisher_p < 0.01:
        verdict = 'DIALECT_IDENTIFIED'
    elif top_score >= 0.30 and gap >= 0.05 and fisher_p < 0.05:
        verdict = 'DIALECT_PROBABLE'
    elif top_score >= 0.25 and fisher_p < 0.05:
        verdict = 'DIALECT_SUGGESTIVE'
    else:
        verdict = 'DIALECT_INDETERMINATE'

    # ── Step 9: Validation battery V1-V10 ──────────────────────────────
    print("\n  Validation battery...")

    validations: Dict[str, bool] = {}

    # V1: Degemination rate != 0.5 +/- 0.1
    v1_data = _load_exp('phase54_degemination.json')
    validations['V1_degemination_direction'] = (
        v1_data is not None
        and abs(v1_data.get('degemination_rate', 0.5) - 0.5) > 0.1
    ) if v1_data else False

    # V2: Article system top dialect separated >= 0.10
    v2_data = _load_exp('phase54_articles.json')
    if v2_data and 'ranking' in v2_data and len(v2_data['ranking']) >= 2:
        validations['V2_article_separation'] = (
            v2_data['ranking'][0].get('composite', 0)
            - v2_data['ranking'][1].get('composite', 0) >= 0.10
        )
    else:
        validations['V2_article_separation'] = False

    # V3: Verb paradigm coherent for <= 2 dialects
    v3_data = _load_exp('phase54_verb_morph.json')
    if v3_data and 'paradigm_coherence' in v3_data:
        n_coherent = sum(1 for v in v3_data['paradigm_coherence'].values() if v)
        validations['V3_verb_narrow'] = n_coherent <= 2
    else:
        validations['V3_verb_narrow'] = False

    # V4: co precedes nouns above chance
    v4_data = _load_exp('phase54_co_syntax.json')
    validations['V4_co_syntactic'] = (
        v4_data is not None and v4_data.get('z_a', 0) > 2.0
    ) if v4_data else False

    # V5: Lenition rate consistent with 1 dialect family
    v5_data = _load_exp('phase54_lenition.json')
    validations['V5_lenition_consistent'] = (
        v5_data is not None and v5_data.get('gates', {}).get('G2', False)
    ) if v5_data else False

    # V6: Pharma tradition 1 >= 2x others
    v6_data = _load_exp('phase54_pharma_region.json')
    validations['V6_pharma_regional'] = (
        v6_data is not None and v6_data.get('gates', {}).get('G2', False)
    ) if v6_data else False

    # V7: Simulated text closest to decoded
    v7_data = _load_exp('phase54_dialect_sim.json')
    validations['V7_simulation_match'] = (
        v7_data is not None and v7_data.get('z_score', 0) >= 2.0
    ) if v7_data else False

    # V8: Zodiac >= 2 correct-folio dialect matches
    v8_data = _load_exp('phase54_zodiac.json')
    validations['V8_zodiac_matches'] = (
        v8_data is not None and v8_data.get('n_correct_folio', 0) >= 2
    ) if v8_data else False

    # V9: Cross-experiment agreement >= 0.60
    validations['V9_cross_agreement'] = agreement_rate >= 0.60

    # V10: Fisher combined p < 0.05
    validations['V10_fisher_significance'] = fisher_p < 0.05

    n_validations_passed = sum(1 for v in validations.values() if v)

    for vname, vpassed in validations.items():
        status = 'PASS' if vpassed else 'FAIL'
        print(f"       {vname}: {status}")

    print(f"\n       Passed: {n_validations_passed} / {len(validations)}")

    # ── Step 10: Summary ───────────────────────────────────────────────
    top_dialect = ranking[0]
    summary = (
        f"Phase 54 Dialect Identification: {verdict}. "
        f"Top dialect: {top_dialect} (score={composite_scores[top_dialect]:.3f}, "
        f"CI=[{ci_lower[top_dialect]:.3f}, {ci_upper[top_dialect]:.3f}]). "
        f"Gap to #2 ({ranking[1]}): {gap:.3f}. "
        f"Agreement: {agreement_rate:.1%} ({len(experiment_winners)} experiments). "
        f"Fisher p={fisher_p:.4f}. "
        f"Validations: {n_validations_passed}/10."
    )

    # ── Console output ─────────────────────────────────────────────────
    print("\n  Per-experiment summary:")
    print(f"  {'Phase':<8} {'Experiment':<28} {'Status':<12} {'Weight':<8} {'z':<8} {'Verdict'}")
    print(f"  {'-'*8} {'-'*28} {'-'*12} {'-'*8} {'-'*8} {'-'*20}")
    for exp in experiments:
        z_str = f"{exp['z_score']:.2f}" if exp.get('z_score') is not None else "n/a"
        w_str = f"{exp.get('weight', 0.0):.1f}" if exp['status'] == 'COMPLETED' else "n/a"
        print(f"  {exp['phase']:<8} {exp['experiment']:<28} {exp['status']:<12} "
              f"{w_str:<8} {z_str:<8} {exp['verdict']}")

    print("\n  Composite dialect scores:")
    for d in ranking:
        bar = '#' * int(composite_scores[d] * 40)
        print(f"       {d:<12} {composite_scores[d]:.3f}  "
              f"[{ci_lower[d]:.3f}, {ci_upper[d]:.3f}]  {bar}")

    print(f"\n  Ranking:")
    ranking_list = []
    for i, d in enumerate(ranking):
        ranking_list.append({
            'dialect': d,
            'score': round(composite_scores[d], 4),
            'ci_lower': round(ci_lower[d], 4),
            'ci_upper': round(ci_upper[d], 4),
        })
        print(f"       #{i+1}  {d:<12}  score={composite_scores[d]:.4f}")

    print(f"\n  Fisher chi2={fisher_chi2:.2f}, p={fisher_p:.4f}")
    print(f"  Cross-experiment agreement: {agreement_rate:.1%}")
    print(f"  Consistency check: {'PASS' if consistency_ok else 'FAIL'}")

    print(f"\n  VERDICT: {verdict}")
    print(f"  {summary}")

    # ── Save ───────────────────────────────────────────────────────────
    runtime = round(time.time() - t0, 2)

    result = DialectIntegrateResult(
        phase="54",
        experiment="dialect_integration",
        n_experiments_run=n_run,
        n_experiments_gated=n_gated,
        per_experiment_results=experiments,
        composite_dialect_scores={d: round(composite_scores[d], 4) for d in DIALECTS},
        ranking=ranking_list,
        fisher_chi2=round(fisher_chi2, 4),
        fisher_p=round(fisher_p, 6),
        agreement_rate=round(agreement_rate, 4),
        consistency_check=consistency_ok,
        validations=validations,
        n_validations_passed=n_validations_passed,
        verdict=verdict,
        summary=summary,
        runtime_seconds=runtime,
    )

    out_path = _save_json(rd, 'phase54_integrate.json', asdict(result))
    print(f"\n  Saved -> {out_path}")
    print(f"  Completed in {runtime:.1f}s")
