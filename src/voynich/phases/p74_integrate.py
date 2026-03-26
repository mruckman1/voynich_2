"""
Phase 74: Integration — Descender Investigation + T1 Vocabulary Push
=====================================================================
Collect results from Tracks A1, A2, B1, B2, B3 and determine verdict.

Dependency chain:
    results/p74_descender.json           (Track A1)
    results/p74_descender_context.json   (Track A2)
    results/p74_patterns.json            (Track B1)
    results/p74_llm_gapfill.json         (Track B2)
    results/p74_complete_readings.json   (Track B3)
        -> results/p74_integrate.json
"""

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


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
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class Phase74IntegrateResult:
    phase: str = "74"
    step: str = "74.6"
    experiment: str = "phase74_integrate"
    tracks_run: List[str] = field(default_factory=list)
    track_summaries: List[Dict[str, Any]] = field(default_factory=list)
    # Gate totals
    total_gates_available: int = 0
    total_gates_passed: int = 0
    a1_gates: int = 0
    a2_gates: int = 0
    b1_gates: int = 0
    b2_gates: int = 0
    b3_gates: int = 0
    # Key metrics
    descender_best_value: str = ""
    descender_improvement: float = 0.0
    context_dependent: bool = False
    n_new_identifications: int = 0
    n_accepted_gap_fills: int = 0
    n_complete_readings: int = 0
    key_findings: List[str] = field(default_factory=list)
    verdict: str = "UNKNOWN"
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Verdict logic
# ---------------------------------------------------------------------------

def _determine_verdict(a1, a2, b1, b2, b3) -> str:
    """
    COMPLETE_READING:
      B3 produces ≥1 fully filled passage with pharmaceutical content
      AND B2 known-answer calibration ≥ 30%
      AND B2 confidence selectivity > 1.5×

    DESCENDER_RESOLVED_AND_VOCAB_EXPANDED:
      Path A finds a better descender value or context-dependent model
      AND Path B expands vocabulary by ≥ 50 types

    VOCABULARY_EXPANDED:
      Path B expands beyond 300 identified types

    DESCENDER_RESOLVED:
      Path A identifies the descender problem and proposes a fix

    INCREMENTAL:
      Some improvement but no breakthrough
    """
    b3_fully_filled = b3.get('n_fully_filled', 0) >= 1
    b3_interpretable = b3.get('gate_b3_3', False)
    b2_calibrated = b2.get('ka_accuracy', 0) >= 0.30
    b2_selective = b2.get('confidence_selectivity', 0) > 1.5

    a1_revised = a1.get('best_value', 'r') != 'r'
    a1_improved = a1.get('improvement_over_r', 0) > 0.005
    a2_context = a2.get('context_dependent_position', False) or \
        a2.get('context_dependent_triple', False)
    descender_resolved = (a1_revised and a1_improved) or a2_context

    b1_new = b1.get('n_total_new', 0)
    b2_accepted = b2.get('n_accepted', 0)
    total_new = b1_new + b2_accepted
    vocab_expanded = total_new >= 50

    total_identified = b1.get('total_identified_types', 0) + b2_accepted

    if b3_fully_filled and b3_interpretable and b2_calibrated and b2_selective:
        return 'COMPLETE_READING'

    if b3_fully_filled and b2_calibrated:
        return 'COMPLETE_READING_PARTIAL'

    if descender_resolved and vocab_expanded:
        return 'DESCENDER_RESOLVED_AND_VOCAB_EXPANDED'

    if total_identified >= 300:
        return 'VOCABULARY_EXPANDED'

    if descender_resolved:
        return 'DESCENDER_RESOLVED'

    if total_new > 0 or a1_improved:
        return 'INCREMENTAL'

    return 'NO_IMPROVEMENT'


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def run_phase74_verdict() -> Phase74IntegrateResult:
    """Integration: collect all track results and produce verdict."""
    t0 = time.time()
    rd = str(_results_dir())

    print("Phase 74 — Integration Verdict")
    print("=" * 35)

    # --- Load all results ---
    a1 = _safe_load(os.path.join(rd, 'p74_descender.json'))
    a2 = _safe_load(os.path.join(rd, 'p74_descender_context.json'))
    b1 = _safe_load(os.path.join(rd, 'p74_patterns.json'))
    b2 = _safe_load(os.path.join(rd, 'p74_llm_gapfill.json'))
    b3 = _safe_load(os.path.join(rd, 'p74_complete_readings.json'))

    tracks_run = []
    track_summaries = []

    # Track A1
    a1_gates = a1.get('gates_passed', 0) if a1 else 0
    if a1:
        tracks_run.append('descender_test')
        track_summaries.append({
            'track': 'Track A1: Descender Test',
            'verdict': a1.get('verdict', '?'),
            'gates': f"{a1_gates}/4",
            'best_value': a1.get('best_value', '?'),
            'improvement': f"{a1.get('improvement_over_r', 0):+.4f}",
            'verbal_fraction': f"{100*a1.get('best_verbal_fraction', 0):.1f}%",
        })
        print(f"  A1: {a1.get('verdict', '?')} ({a1_gates}/4) — "
              f"best={a1.get('best_value', '?')}, "
              f"Δ={a1.get('improvement_over_r', 0):+.4f}")

    # Track A2
    a2_gates = a2.get('gates_passed', 0) if a2 else 0
    if a2:
        tracks_run.append('descender_context')
        track_summaries.append({
            'track': 'Track A2: Descender Context',
            'verdict': a2.get('verdict', '?'),
            'gates': f"{a2_gates}/3",
            'position_dependent': a2.get('context_dependent_position', False),
            'triple_dependent': a2.get('context_dependent_triple', False),
            'n_prefer_other': a2.get('n_triples_prefer_other', 0),
        })
        print(f"  A2: {a2.get('verdict', '?')} ({a2_gates}/3) — "
              f"pos_dep={a2.get('context_dependent_position', '?')}, "
              f"triple_dep={a2.get('context_dependent_triple', '?')}")

    # Track B1
    b1_gates = b1.get('gates_passed', 0) if b1 else 0
    if b1:
        tracks_run.append('eva_patterns')
        track_summaries.append({
            'track': 'Track B1: EVA Patterns',
            'verdict': b1.get('verdict', '?'),
            'gates': f"{b1_gates}/2",
            'n_distributional': b1.get('n_distributional', 0),
            'n_positional': b1.get('n_positional', 0),
            'total_new': b1.get('n_total_new', 0),
        })
        print(f"  B1: {b1.get('verdict', '?')} ({b1_gates}/2) — "
              f"{b1.get('n_distributional', 0)} distrib, "
              f"{b1.get('n_positional', 0)} positional")

    # Track B2
    b2_gates = b2.get('gates_passed', 0) if b2 else 0
    if b2:
        tracks_run.append('llm_gap_fill')
        track_summaries.append({
            'track': 'Track B2: LLM Gap-Fill',
            'verdict': b2.get('verdict', '?'),
            'gates': f"{b2_gates}/6",
            'ka_accuracy': f"{100*b2.get('ka_accuracy', 0):.1f}%",
            'selectivity': f"{b2.get('confidence_selectivity', 0):.2f}×",
            'consistency': f"{100*b2.get('consistency', 0):.1f}%",
            'n_accepted': b2.get('n_accepted', 0),
        })
        print(f"  B2: {b2.get('verdict', '?')} ({b2_gates}/6) — "
              f"KA={100*b2.get('ka_accuracy', 0):.1f}%, "
              f"accepted={b2.get('n_accepted', 0)}")

    # Track B3
    b3_gates = b3.get('gates_passed', 0) if b3 else 0
    if b3:
        tracks_run.append('complete_read')
        track_summaries.append({
            'track': 'Track B3: Complete Readings',
            'verdict': b3.get('verdict', '?'),
            'gates': f"{b3_gates}/3",
            'n_fully_filled': b3.get('n_fully_filled', 0),
            'n_near_complete': b3.get('n_near_complete', 0),
            'best_fraction': f"{100*b3.get('best_complete_fraction', 0):.1f}%",
        })
        print(f"  B3: {b3.get('verdict', '?')} ({b3_gates}/3) — "
              f"fully={b3.get('n_fully_filled', 0)}, "
              f"near={b3.get('n_near_complete', 0)}")

    # --- Totals ---
    total_avail = 4 + 3 + 2 + 6 + 3  # = 18
    total_passed = a1_gates + a2_gates + b1_gates + b2_gates + b3_gates

    print(f"\n  Total gates: {total_passed}/{total_avail}")

    # --- Key findings ---
    findings = []

    if a1:
        best = a1.get('best_value', 'r')
        if best != 'r':
            findings.append(
                f"Descender best value is '{best}' (not 'r'), "
                f"improvement {a1.get('improvement_over_r', 0):+.4f}")
        else:
            findings.append("Descender→r confirmed as best global value")

        pos = a1.get('position_analysis', {})
        final_frac = pos.get('final_fraction', 0)
        findings.append(
            f"Descender position: {final_frac:.0%} final "
            f"(cf. connector 1.9% final)")

    if a2:
        if a2.get('context_dependent_position'):
            findings.append(
                f"Context-dependent by position: "
                f"final→{a2.get('final_best_value', '?')}, "
                f"medial→{a2.get('medial_best_value', '?')}")
        if a2.get('context_dependent_triple'):
            findings.append(
                f"Context-dependent by preceding triple: "
                f"{a2.get('n_triples_prefer_other', 0)} triples prefer non-r")

    if b1:
        findings.append(
            f"EVA patterns: {b1.get('n_total_new', 0)} new types identified "
            f"({b1.get('n_distributional', 0)} distributional, "
            f"{b1.get('n_positional', 0)} positional)")

    if b2:
        findings.append(
            f"LLM gap-filling: {b2.get('n_accepted', 0)} accepted proposals, "
            f"KA accuracy {100*b2.get('ka_accuracy', 0):.1f}%")

    if b3 and b3.get('n_fully_filled', 0) > 0:
        findings.append(
            f"COMPLETE READINGS: {b3['n_fully_filled']} passages fully filled")

    print(f"\n  Key findings:")
    for f in findings:
        print(f"    • {f}")

    # --- Verdict ---
    verdict = _determine_verdict(a1, a2, b1, b2, b3)
    print(f"\n  VERDICT: {verdict}")

    # --- Build result ---
    result = Phase74IntegrateResult(
        tracks_run=tracks_run,
        track_summaries=track_summaries,
        total_gates_available=total_avail,
        total_gates_passed=total_passed,
        a1_gates=a1_gates,
        a2_gates=a2_gates,
        b1_gates=b1_gates,
        b2_gates=b2_gates,
        b3_gates=b3_gates,
        descender_best_value=a1.get('best_value', '') if a1 else '',
        descender_improvement=a1.get('improvement_over_r', 0.0) if a1 else 0.0,
        context_dependent=bool(
            (a2 and a2.get('context_dependent_position')) or
            (a2 and a2.get('context_dependent_triple'))),
        n_new_identifications=b1.get('n_total_new', 0) if b1 else 0,
        n_accepted_gap_fills=b2.get('n_accepted', 0) if b2 else 0,
        n_complete_readings=b3.get('n_fully_filled', 0) if b3 else 0,
        key_findings=findings,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 1),
    )

    path = _save_json(rd, 'p74_integrate.json', asdict(result))
    print(f"\n  Saved: {path}")

    return result


def run_phase74() -> Phase74IntegrateResult:
    """Run full Phase 74 pipeline."""
    t0 = time.time()

    print("=" * 60)
    print("Phase 74: Descender Investigation + T1 Vocabulary Push")
    print("=" * 60)

    # Path A: Descender investigation
    print("\n" + "=" * 60)
    print("PATH A: Descender Investigation")
    print("=" * 60)

    from voynich.phases.p74_descender import run_descender_test
    run_descender_test()

    from voynich.phases.p74_descender_context import run_descender_context
    run_descender_context()

    # Path B: T1 vocabulary push
    print("\n" + "=" * 60)
    print("PATH B: T1 Vocabulary Push")
    print("=" * 60)

    from voynich.phases.p74_eva_patterns import run_eva_patterns
    run_eva_patterns()

    from voynich.phases.p74_llm_gapfill import run_llm_gap_fill
    run_llm_gap_fill()

    from voynich.phases.p74_complete_read import run_complete_read
    run_complete_read()

    # Integration
    print("\n" + "=" * 60)
    print("INTEGRATION")
    print("=" * 60)

    result = run_phase74_verdict()
    result.runtime_seconds = round(time.time() - t0, 1)

    print(f"\n  Total pipeline runtime: {result.runtime_seconds:.1f}s")

    return result
