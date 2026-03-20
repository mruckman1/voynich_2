"""
Phase 59 Integration: CVC Refinement Verdict
==============================================
Combines all 11 investigations into an overall assessment of the CVC
coda decode quality.

Three questions:
  1. Is the coda interpretation correct?  (Inv 1, 6, 5, 8)
  2. Is the specific mapping right?       (Inv 3, 7)
  3. Does CVC produce better content?     (Inv 2, 4, 9, 10, 11)

Dependency chain:
    results/cvc_segmentation.json     (Inv 1)
    results/cvc_positional.json       (Inv 6)
    results/cvc_tm_ambiguity.json     (Inv 3)
    results/cvc_connectors.json       (Inv 7)
    results/cvc_dictionary.json       (Inv 2)
    results/cvc_glossing.json         (Inv 4)
    results/cvc_cross_mi.json         (Inv 5)
    results/cvc_combination.json      (Inv 8)
    results/cvc_recipes.json          (Inv 9)
    results/cvc_aiin_family.json      (Inv 10)
    results/cvc_permutation.json      (Inv 11)
        -> results/phase59_integrate.json
"""

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from voynich.core._paths import results_dir as _results_dir


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
# Investigation result files
# ---------------------------------------------------------------------------

INVESTIGATIONS = [
    (1, 'cvc_segmentation.json', 'Syllable Segmentation'),
    (2, 'cvc_dictionary.json', 'CVC-Aware Dictionary'),
    (3, 'cvc_tm_ambiguity.json', 't/m Coda Ambiguity'),
    (4, 'cvc_glossing.json', 'Signal Word Glossing'),
    (5, 'cvc_cross_mi.json', 'Cross-Boundary MI'),
    (6, 'cvc_positional.json', 'Positional Distribution'),
    (7, 'cvc_connectors.json', 'Connector Group'),
    (8, 'cvc_combination.json', 'Combination Rules'),
    (9, 'cvc_recipes.json', 'Recipe Reading'),
    (10, 'cvc_aiin_family.json', 'aiin Family'),
    (11, 'cvc_permutation.json', 'Permutation Coherence'),
]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class InvestigationSummary:
    """Summary of one investigation."""
    number: int
    title: str
    gate_passed: bool
    gates_passed: int
    total_gates: int
    key_metric: str
    key_value: str


@dataclass
class Phase59Result:
    """Full Phase 59 integration output."""
    phase: str = "59"
    experiment: str = "phase59_integrate"
    # Per-investigation summaries
    investigations: List[InvestigationSummary] = field(default_factory=list)
    n_available: int = 0
    n_passed: int = 0
    n_failed: int = 0
    # Three questions
    q1_coda_correct: bool = False        # Inv 1, 6, 5, 8
    q1_score: str = ''
    q2_mapping_refined: bool = False     # Inv 3, 7
    q2_score: str = ''
    q3_content_improved: bool = False    # Inv 2, 4, 9, 10, 11
    q3_score: str = ''
    # Overall verdict
    verdict: str = ''  # CVC_VALIDATED / CVC_PARTIAL / CVC_WEAK / CVC_REJECTED
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def _extract_key_metric(inv_num: int, data: Dict) -> tuple:
    """Extract the most important metric from each investigation."""
    if inv_num == 1:
        rate = data.get('attestation_rate_token', 0)
        return f"attestation={rate:.1%}", str(rate)
    elif inv_num == 2:
        sel = data.get('cvc_selectivity', 0)
        return f"selectivity={sel:.2f}×", str(sel)
    elif inv_num == 3:
        return f"verdict={data.get('verdict', '?')}", data.get('verdict', '?')
    elif inv_num == 4:
        frac = data.get('content_fraction', 0)
        return f"content={frac:.1%}", str(frac)
    elif inv_num == 5:
        dec = data.get('cvc_decreased', False)
        return f"decreased={'YES' if dec else 'NO'}", str(dec)
    elif inv_num == 6:
        gs = data.get('overall_stats', {})
        ifrac = gs.get('initial_frac', 1.0) if isinstance(gs, dict) else 1.0
        return f"initial={ifrac:.1%}", str(ifrac)
    elif inv_num == 7:
        best = data.get('best_coda', '?')
        return f"best_coda={best}", str(best)
    elif inv_num == 8:
        rate = data.get('overall_respect_rate', 0)
        return f"respect={rate:.1%}", str(rate)
    elif inv_num == 9:
        frac = data.get('mean_glossed_fraction', 0)
        return f"glossed={frac:.1%}", str(frac)
    elif inv_num == 10:
        frac = data.get('latin_ending_fraction', 0)
        return f"latin_endings={frac:.1%}", str(frac)
    elif inv_num == 11:
        p = data.get('p_all_three', 1.0)
        return f"p={p:.4f}", str(p)
    return "?", "?"


def load_all_investigations(rd: str) -> List[tuple]:
    """Load all available investigation results."""
    results = []
    for inv_num, filename, title in INVESTIGATIONS:
        data = _safe_load(os.path.join(rd, filename))
        if data:
            results.append((inv_num, title, filename, data))
    return results


# ---------------------------------------------------------------------------
# CLI entry points
# ---------------------------------------------------------------------------

def run_phase59_verdict():
    """Phase 59 verdict: integrate all 11 investigations."""
    t0 = time.time()
    print("=" * 70)
    print("Phase 59: CVC Refinement Integration & Verdict")
    print("=" * 70)

    rd = str(_results_dir())

    # Load all investigations
    all_inv = load_all_investigations(rd)
    print(f"\n  Investigations available: {len(all_inv)}/11")

    summaries: List[InvestigationSummary] = []
    for inv_num, title, filename, data in all_inv:
        gate_passed = data.get('gate_passed', False)
        gates_passed = data.get('gates_passed', 0)

        # Infer total gates from the data
        gate_keys = [k for k in data if k.startswith('g') and k[1:2].isdigit()]
        total_gates = len(gate_keys) if gate_keys else 0

        key_metric, key_value = _extract_key_metric(inv_num, data)

        summaries.append(InvestigationSummary(
            number=inv_num,
            title=title,
            gate_passed=gate_passed,
            gates_passed=gates_passed,
            total_gates=total_gates,
            key_metric=key_metric,
            key_value=key_value,
        ))

    # Print summary table
    print(f"\n  {'#':>3} {'Investigation':<28} {'Gates':>8} {'Pass':>6} {'Key Metric'}")
    print(f"  {'-'*3} {'-'*28} {'-'*8} {'-'*6} {'-'*30}")
    for s in summaries:
        status = 'PASS' if s.gate_passed else 'FAIL'
        print(f"  {s.number:>3} {s.title:<28} {s.gates_passed}/{s.total_gates:>2} "
              f"  {status:<6} {s.key_metric}")

    # Score three questions
    inv_map = {s.number: s for s in summaries}

    # Q1: Is the coda interpretation correct? (Inv 1, 6, 5, 8)
    q1_invs = [inv_map.get(n) for n in [1, 6, 5, 8]]
    q1_invs = [i for i in q1_invs if i is not None]
    q1_passed = sum(1 for i in q1_invs if i.gate_passed)
    q1_total = len(q1_invs)
    q1_correct = q1_passed >= (q1_total * 0.5) if q1_total > 0 else False
    q1_score = f"{q1_passed}/{q1_total}"

    # Q2: Is the specific mapping right? (Inv 3, 7)
    q2_invs = [inv_map.get(n) for n in [3, 7]]
    q2_invs = [i for i in q2_invs if i is not None]
    q2_passed = sum(1 for i in q2_invs if i.gate_passed)
    q2_total = len(q2_invs)
    q2_refined = q2_passed >= 1 if q2_total > 0 else False
    q2_score = f"{q2_passed}/{q2_total}"

    # Q3: Does CVC produce better content? (Inv 2, 4, 9, 10, 11)
    q3_invs = [inv_map.get(n) for n in [2, 4, 9, 10, 11]]
    q3_invs = [i for i in q3_invs if i is not None]
    q3_passed = sum(1 for i in q3_invs if i.gate_passed)
    q3_total = len(q3_invs)
    q3_improved = q3_passed >= (q3_total * 0.5) if q3_total > 0 else False
    q3_score = f"{q3_passed}/{q3_total}"

    print(f"\n  Three Questions:")
    print(f"    Q1 Coda interpretation correct? {q1_score} → "
          f"{'YES' if q1_correct else 'NO'}")
    print(f"    Q2 Mapping refined?             {q2_score} → "
          f"{'YES' if q2_refined else 'NO'}")
    print(f"    Q3 Content improved?            {q3_score} → "
          f"{'YES' if q3_improved else 'NO'}")

    # Overall verdict
    n_passed = sum(1 for s in summaries if s.gate_passed)
    n_total = len(summaries)

    if n_passed >= 8:
        verdict = 'CVC_VALIDATED'
    elif n_passed >= 5:
        verdict = 'CVC_PARTIAL'
    elif n_passed >= 3:
        verdict = 'CVC_WEAK'
    else:
        verdict = 'CVC_REJECTED'

    print(f"\n  Overall: {n_passed}/{n_total} investigations passed")
    print(f"  VERDICT: {verdict}")

    result = Phase59Result(
        investigations=summaries,
        n_available=n_total,
        n_passed=n_passed,
        n_failed=n_total - n_passed,
        q1_coda_correct=q1_correct,
        q1_score=q1_score,
        q2_mapping_refined=q2_refined,
        q2_score=q2_score,
        q3_content_improved=q3_improved,
        q3_score=q3_score,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    path = _save_json(rd, 'phase59_integrate.json', result)
    print(f"\n  Saved: {path}")
    print(f"  Phase 59 verdict completed in {time.time() - t0:.1f}s")


def run_phase59():
    """Run full Phase 59 pipeline: all 11 investigations + verdict."""
    t0 = time.time()
    print("=" * 70)
    print("Phase 59: CVC Refinement and Deep Investigation")
    print("=" * 70)
    print("  Running all 11 investigations in dependency order ...\n")

    # Tier 1: Foundational
    from voynich.phases.cvc_segmentation import run_cvc_segmentation
    from voynich.phases.cvc_positional import run_cvc_positional
    run_cvc_segmentation()
    print()
    run_cvc_positional()
    print()

    # Tier 2: Mapping Refinement
    from voynich.phases.cvc_tm_ambiguity import run_cvc_tm
    from voynich.phases.cvc_connectors import run_cvc_connector
    run_cvc_tm()
    print()
    run_cvc_connector()
    print()

    # Tier 3: Content and Evaluation
    from voynich.phases.cvc_dictionary import run_cvc_dict
    from voynich.phases.cvc_glossing import run_cvc_gloss
    from voynich.phases.cvc_recipes import run_cvc_recipe
    from voynich.phases.cvc_aiin_family import run_cvc_aiin
    run_cvc_dict()
    print()
    run_cvc_gloss()
    print()
    run_cvc_recipe()
    print()
    run_cvc_aiin()
    print()

    # Tier 4: Validation and Prediction
    from voynich.phases.cvc_cross_mi import run_cvc_mi
    from voynich.phases.cvc_combination import run_cvc_combo
    from voynich.phases.cvc_permutation import run_cvc_perm
    run_cvc_mi()
    print()
    run_cvc_combo()
    print()
    run_cvc_perm()
    print()

    # Integration
    run_phase59_verdict()

    print(f"\n  Full Phase 59 pipeline completed in {time.time() - t0:.1f}s")
