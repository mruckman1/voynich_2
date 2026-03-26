"""
Phase 72: Integration — Decode Model Diagnosis and Revision
============================================================
Aggregates results from 5 tracks:
  Track 1: Connector investigation (13 values tested)
  Track 2: Cross-validation failure diagnosis
  Track 3: Alternative combination models (6 models tested)
  Track 4: Expanded T1 identification (5 tiers)
  Track 5: Variable-length encoding hypothesis

Applies all recommended changes together and evaluates combined effect.

Dependency chain:
    results/phase72_connector.json      (Track 1)
    results/phase72_xval.json           (Track 2)
    results/phase72_combination.json    (Track 3)
    results/phase72_t1_expand.json      (Track 4)
    results/phase72_var_length.json     (Track 5)
        -> results/phase72_integrate.json
"""

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

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
# Verdict logic
# ---------------------------------------------------------------------------

def _determine_verdict(t1: Dict, t2: Dict, t3: Dict, t4: Dict, t5: Dict) -> str:
    """Determine phase-level verdict from track results.

    MODEL_REVISED:
        Tracks 1-3 identify a specific fix that improves dict-hit > 2pp
        AND bigram_z preserved AND >= 8 total gates.

    CONNECTOR_REVISED:
        Track 1 shows improvement but Tracks 3/5 marginal.

    T1_BYPASS:
        Track 4 achieves > 20% token coverage with FPR < 30%.

    DIAGNOSIS_COMPLETE:
        Tracks 1-3 identify failure points but no single fix improves
        significantly. Multiple smaller fixes together may help.

    NO_IMPROVEMENT:
        Nothing helps.
    """
    total_gates = (
        t1.get('gates_passed', 0) +
        t2.get('gates_passed', 0) +
        t3.get('gates_passed', 0) +
        t4.get('gates_passed', 0) +
        t5.get('gates_passed', 0)
    )

    t1_improved = t1.get('gate_cn2', False)
    t3_improved = t3.get('gate_cm2', False)
    t4_bypass = t4.get('gate_t2', False) and t4.get('gate_t3', False)
    t5_improved = t5.get('gate_vl3', False)

    if (t1_improved or t3_improved or t5_improved) and total_gates >= 8:
        return 'MODEL_REVISED'

    if t1_improved and not t3_improved and not t5_improved:
        return 'CONNECTOR_REVISED'

    if t4_bypass:
        return 'T1_BYPASS'

    if total_gates >= 5:
        return 'DIAGNOSIS_COMPLETE'

    return 'NO_IMPROVEMENT'


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class Phase72IntegrateResult:
    phase: str = "72"
    step: str = "72.6"
    experiment: str = "phase72_integrate"
    # Track summaries
    tracks_run: List[str] = field(default_factory=list)
    track_summaries: List[Dict[str, Any]] = field(default_factory=list)
    # Gate counts
    total_gates_available: int = 0
    total_gates_passed: int = 0
    track1_gates: int = 0
    track2_gates: int = 0
    track3_gates: int = 0
    track4_gates: int = 0
    track5_gates: int = 0
    # Recommendations
    recommended_connector: str = "r"
    recommended_model: str = "append"
    recommended_t1_tier: str = "B"
    recommended_length_changes: int = 0
    # Key findings
    key_findings: List[str] = field(default_factory=list)
    verdict: str = "UNKNOWN"
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_phase72_verdict():
    """Collect all track results and produce integration verdict."""
    t0 = time.time()
    rd = str(_results_dir())

    print("Phase 72.6 — Integration Verdict")
    print("=" * 33)

    # --- Load track results ---
    t1 = _safe_load(os.path.join(rd, 'phase72_connector.json'))
    t2 = _safe_load(os.path.join(rd, 'phase72_xval.json'))
    t3 = _safe_load(os.path.join(rd, 'phase72_combination.json'))
    t4 = _safe_load(os.path.join(rd, 'phase72_t1_expand.json'))
    t5 = _safe_load(os.path.join(rd, 'phase72_var_length.json'))

    tracks_run = []
    track_summaries = []

    # --- Track 1: Connector ---
    if t1:
        tracks_run.append('connector')
        track_summaries.append({
            'track': 'Track 1: Connector Investigation',
            'verdict': t1.get('verdict', '?'),
            'gates': f"{t1.get('gates_passed', 0)}/4",
            'best_value': t1.get('best_value', '?'),
            'improvement': t1.get('improvement_over_r', 0),
        })
        print(f"\n  Track 1 (Connector): {t1.get('verdict', '?')} "
              f"({t1.get('gates_passed', 0)}/4)")
        print(f"    Best connector: '{t1.get('best_value', '?')}' "
              f"(improvement: {t1.get('improvement_over_r', 0):+.4f})")

    # --- Track 2: Cross-Validation ---
    if t2:
        tracks_run.append('xval')
        track_summaries.append({
            'track': 'Track 2: Cross-Validation Diagnosis',
            'verdict': t2.get('verdict', '?'),
            'gates': f"{t2.get('gates_passed', 0)}/4",
            'dominant_error': t2.get('dominant_error_source', '?'),
            'overall_xval': t2.get('overall_xval', 0),
        })
        print(f"\n  Track 2 (XVal): {t2.get('verdict', '?')} "
              f"({t2.get('gates_passed', 0)}/4)")
        print(f"    Dominant error: {t2.get('dominant_error_source', '?')}")
        print(f"    Overall xval: {t2.get('overall_xval', 0):.1%}")
        print(f"    Worst coda: -{t2.get('worst_coda', '?')}")
        print(f"    Confirmed vs unresolved gap: {t2.get('triple_gap', 0):.1%}")

    # --- Track 3: Combination Models ---
    if t3:
        tracks_run.append('combination')
        track_summaries.append({
            'track': 'Track 3: Combination Models',
            'verdict': t3.get('verdict', '?'),
            'gates': f"{t3.get('gates_passed', 0)}/5",
            'best_model': t3.get('best_model', '?'),
            'improvement': t3.get('improvement', 0),
        })
        print(f"\n  Track 3 (Combination): {t3.get('verdict', '?')} "
              f"({t3.get('gates_passed', 0)}/5)")
        print(f"    Best model: {t3.get('best_model', '?')} "
              f"(improvement: {t3.get('improvement', 0):+.4f})")

    # --- Track 4: T1 Expansion ---
    if t4:
        tracks_run.append('t1_expand')
        track_summaries.append({
            'track': 'Track 4: T1 Expansion',
            'verdict': t4.get('verdict', '?'),
            'gates': f"{t4.get('gates_passed', 0)}/5",
            'cumulative_ids': t4.get('cumulative_identifications', 0),
            'recommended_tier': t4.get('recommended_tier', '?'),
        })
        print(f"\n  Track 4 (T1 Expand): {t4.get('verdict', '?')} "
              f"({t4.get('gates_passed', 0)}/5)")
        print(f"    Cumulative IDs: {t4.get('cumulative_identifications', 0)}")
        print(f"    Recommended tier: {t4.get('recommended_tier', '?')}")

    # --- Track 5: Variable Length ---
    if t5:
        tracks_run.append('var_length')
        track_summaries.append({
            'track': 'Track 5: Variable-Length Encoding',
            'verdict': t5.get('verdict', '?'),
            'gates': f"{t5.get('gates_passed', 0)}/5",
            'n_changed': t5.get('n_changed', 0),
            'improvement': t5.get('improvement', 0),
        })
        print(f"\n  Track 5 (Variable Length): {t5.get('verdict', '?')} "
              f"({t5.get('gates_passed', 0)}/5)")
        print(f"    Changed: {t5.get('n_changed', 0)} triples")
        print(f"    Dict-hit improvement: {t5.get('improvement', 0):+.3f}")

    # --- Gate totals ---
    t1_gates = t1.get('gates_passed', 0) if t1 else 0
    t2_gates = t2.get('gates_passed', 0) if t2 else 0
    t3_gates = t3.get('gates_passed', 0) if t3 else 0
    t4_gates = t4.get('gates_passed', 0) if t4 else 0
    t5_gates = t5.get('gates_passed', 0) if t5 else 0
    total_gates = t1_gates + t2_gates + t3_gates + t4_gates + t5_gates
    total_available = 4 + 4 + 5 + 5 + 5  # 23

    print(f"\n  Total gates: {total_gates}/{total_available}")

    # --- Key findings ---
    findings = []

    if t1 and t1.get('best_value', 'r') != 'r':
        findings.append(f"Connector '{t1['best_value']}' outperforms 'r' "
                        f"(improvement: {t1.get('improvement_over_r', 0):+.4f})")
    elif t1:
        findings.append("Connector 'r' remains the best assignment")

    if t2:
        findings.append(f"Cross-validation failure dominated by: "
                        f"{t2.get('dominant_error_source', '?')}")
        findings.append(f"Overall xval: {t2.get('overall_xval', 0):.1%}")

    if t3 and t3.get('best_model', 'append') != 'append':
        findings.append(f"Combination model '{t3['best_model']}' outperforms 'append'")
    elif t3:
        findings.append("Append model remains the best combination rule")

    if t4:
        findings.append(f"T1 expansion: {t4.get('cumulative_identifications', 0)} "
                        f"identifications across 5 tiers")

    if t5 and t5.get('n_changed', 0) > 0:
        findings.append(f"Variable-length: {t5['n_changed']} triples benefit from "
                        f"non-standard lengths (improvement: {t5.get('improvement', 0):+.3f})")
    elif t5:
        findings.append("Fixed 2-char CV encoding is optimal")

    for f in findings:
        print(f"  * {f}")

    # --- Recommendations ---
    rec_connector = t1.get('best_value', 'r') if t1 else 'r'
    rec_model = t3.get('best_model', 'append') if t3 else 'append'
    rec_tier = t4.get('recommended_tier', 'B') if t4 else 'B'
    rec_length_changes = t5.get('n_changed', 0) if t5 else 0

    # --- Verdict ---
    verdict = _determine_verdict(t1 or {}, t2 or {}, t3 or {}, t4 or {}, t5 or {})
    print(f"\n  Verdict: {verdict}")

    # --- Build result ---
    result = Phase72IntegrateResult(
        tracks_run=tracks_run,
        track_summaries=track_summaries,
        total_gates_available=total_available,
        total_gates_passed=total_gates,
        track1_gates=t1_gates,
        track2_gates=t2_gates,
        track3_gates=t3_gates,
        track4_gates=t4_gates,
        track5_gates=t5_gates,
        recommended_connector=rec_connector,
        recommended_model=rec_model,
        recommended_t1_tier=rec_tier,
        recommended_length_changes=rec_length_changes,
        key_findings=findings,
        verdict=verdict,
        runtime_seconds=time.time() - t0,
    )

    path = _save_json(rd, 'phase72_integrate.json', asdict(result))
    print(f"\n  Saved: {path}")
    print(f"  Runtime: {result.runtime_seconds:.1f}s")

    return result


def run_phase72():
    """Full Phase 72 pipeline: all tracks + integration."""
    # Track 2 (simplest, no nulls)
    from voynich.phases.p72_xval import run_xval_diagnosis
    run_xval_diagnosis()
    print("\n" + "=" * 70 + "\n")

    # Track 1 (connector investigation)
    from voynich.phases.p72_connector import run_connector_test
    run_connector_test()
    print("\n" + "=" * 70 + "\n")

    # Track 5 (variable length — independent)
    from voynich.phases.p72_variable_length import run_variable_length
    run_variable_length()
    print("\n" + "=" * 70 + "\n")

    # Track 4 (T1 expansion — independent)
    from voynich.phases.p72_t1_expand import run_t1_expand
    run_t1_expand()
    print("\n" + "=" * 70 + "\n")

    # Track 3 (combination models — depends on Track 1)
    from voynich.phases.p72_combination import run_combination_models
    run_combination_models()
    print("\n" + "=" * 70 + "\n")

    # Integration
    result = run_phase72_verdict()
    return result
