"""
Phase 75: Integration — 3-Coda Model Pipeline
===============================================
Collect results from Step 0 and Tracks 1-5, determine overall verdict.

The 3-coda model applies connector→null AND descender→null, leaving
only 3 genuine coda consonants (n, s, t).

Dependency chain:
    results/p75_redecode.json          (Step 0)
    results/p75_revalidate.json        (Track 1)
    results/p75_grammar.json           (Track 2)
    results/p75_t1.json                (Track 3)
    results/p75_paradigms.json         (Track 4)
    results/p75_readings.json          (Track 5)
        -> results/p75_integrate.json
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
class Phase75IntegrateResult:
    phase: str = "75"
    step: str = "75.6"
    experiment: str = "phase75_integrate"
    tracks_run: List[str] = field(default_factory=list)
    track_summaries: List[Dict[str, Any]] = field(default_factory=list)
    # Step 0 summary
    n_changed_tokens: int = 0
    new_dict_hit: float = 0.0
    delta_dict_hit: float = 0.0
    new_xval: float = 0.0
    # Gate totals
    total_gates_available: int = 0
    total_gates_passed: int = 0
    step0_pass: bool = False
    track1_gates: int = 0
    track2_gates: int = 0
    track3_gates: int = 0
    track4_gates: int = 0
    track5_gates: int = 0
    # Key comparison metrics
    old_verbal_fraction: float = 0.0
    new_verbal_fraction: float = 0.0
    t1_stability: float = 0.0
    template_selectivity: float = 0.0
    lexical_selectivity: float = 0.0
    distributional_coverage: float = 0.0
    # Key findings
    key_findings: List[str] = field(default_factory=list)
    verdict: str = "UNKNOWN"
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Verdict logic
# ---------------------------------------------------------------------------

def _determine_verdict(s0, t1, t2, t3, t4, t5) -> str:
    """
    THREE_CODA_VALIDATED:
      Track 2 grammar >= 4/5 gates AND Track 1 >= 2/3 gates.

    THREE_CODA_READING:
      Track 5 template selectivity > 1.3x AND Track 2 grammar >= 4/5 gates.

    THREE_CODA_IMPROVED:
      Step 0 dict-hit improves (delta >= 0) AND Track 3 T1 stable
      (stability >= 0.80).

    THREE_CODA_NEUTRAL:
      Otherwise.
    """
    t1_gates = t1.get('gates_passed', 0)
    t2_gates = t2.get('gates_passed', 0)
    t3_stability = t3.get('stability_fraction', 0.0)
    t5_template_sel = t5.get('template_selectivity', 0.0)
    s0_delta = s0.get('delta_dict_hit', 0.0)

    if t2_gates >= 4 and t1_gates >= 2:
        return 'THREE_CODA_VALIDATED'

    if t5_template_sel > 1.3 and t2_gates >= 4:
        return 'THREE_CODA_READING'

    if s0_delta >= 0.0 and t3_stability >= 0.80:
        return 'THREE_CODA_IMPROVED'

    return 'THREE_CODA_NEUTRAL'


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def run_phase75_verdict() -> Phase75IntegrateResult:
    """Integration: collect all track results and produce verdict."""
    t0 = time.time()
    rd = str(_results_dir())

    print("Phase 75 — Integration Verdict (3-Coda Model)")
    print("=" * 48)

    # --- Load all results ---
    s0 = _safe_load(os.path.join(rd, 'p75_redecode.json'))
    t1 = _safe_load(os.path.join(rd, 'p75_revalidate.json'))
    t2 = _safe_load(os.path.join(rd, 'p75_grammar.json'))
    t3 = _safe_load(os.path.join(rd, 'p75_t1.json'))
    t4 = _safe_load(os.path.join(rd, 'p75_paradigms.json'))
    t5 = _safe_load(os.path.join(rd, 'p75_readings.json'))

    tracks_run = []
    track_summaries = []

    # Step 0
    if s0:
        tracks_run.append('redecode')
        track_summaries.append({
            'track': 'Step 0: Redecode (3-coda)',
            'verdict': s0.get('verdict', '?'),
            'n_changed': s0.get('n_changed', 0),
            'dict_hit': f"{100*s0.get('new_dict_hit', 0):.1f}%",
            'delta': f"{100*s0.get('delta_dict_hit', 0):+.1f}%",
        })

    # Track 1
    t1_gates = t1.get('gates_passed', 0) if t1 else 0
    if t1:
        tracks_run.append('revalidate')
        track_summaries.append({
            'track': 'Track 1: Revalidation',
            'verdict': t1.get('verdict', '?'),
            'gates': f"{t1_gates}/3",
            'p_0a': t1.get('test_0a_p', '?'),
            'p_0b': t1.get('test_0b_p', '?'),
            'p_0c': t1.get('test_0c_p', '?'),
        })

    # Track 2
    t2_gates = t2.get('gates_passed', 0) if t2 else 0
    if t2:
        tracks_run.append('grammar')
        track_summaries.append({
            'track': 'Track 2: Grammar',
            'verdict': t2.get('verdict', '?'),
            'gates': f"{t2_gates}/5",
            'verbal': f"{100*t2.get('new_verbal_fraction', 0):.1f}%",
        })

    # Track 3
    t3_gates = t3.get('gates_passed', 0) if t3 else 0
    if t3:
        tracks_run.append('t1')
        track_summaries.append({
            'track': 'Track 3: T1',
            'verdict': t3.get('verdict', '?'),
            'gates': f"{t3_gates}/3",
            'stability': f"{100*t3.get('stability_fraction', 0):.1f}%",
            'total_ids': t3.get('n_identifications', 0),
        })

    # Track 4
    t4_gates = t4.get('gates_passed', 0) if t4 else 0
    if t4:
        tracks_run.append('paradigms')
        track_summaries.append({
            'track': 'Track 4: Paradigms',
            'verdict': t4.get('verdict', '?'),
            'gates': f"{t4_gates}/5",
            'n_paradigms': t4.get('n_paradigms', 0),
            'mean_size': t4.get('mean_paradigm_size', 0),
        })

    # Track 5
    t5_gates = t5.get('gates_passed', 0) if t5 else 0
    if t5:
        tracks_run.append('readings')
        track_summaries.append({
            'track': 'Track 5: Readings',
            'verdict': t5.get('verdict', '?'),
            'gates': f"{t5_gates}/6",
            'mean_id': f"{100*t5.get('mean_identified_fraction', 0):.1f}%",
            'template_sel': f"{t5.get('template_selectivity', 0):.2f}x",
            'distrib_coverage': f"{100*t5.get('distributional_coverage', 0):.1f}%",
        })

    total_gates = 1 + 3 + 5 + 3 + 5 + 6  # s0 + t1-t5
    total_passed = ((1 if s0.get('gate_r0', False) else 0) +
                    t1_gates + t2_gates + t3_gates + t4_gates + t5_gates)

    # --- Verdict ---
    verdict = _determine_verdict(s0, t1, t2, t3, t4, t5)

    # --- Key findings ---
    findings = []
    if s0:
        findings.append(
            f"3-coda model changes {s0.get('n_changed', 0)} tokens "
            f"({100*s0.get('changed_fraction', 0):.1f}%)")
        findings.append(
            f"Dict-hit: {100*s0.get('old_dict_hit', 0):.1f}% -> "
            f"{100*s0.get('new_dict_hit', 0):.1f}%")
        findings.append(
            f"Cross-validation: {100*s0.get('old_xval', 0):.1f}% -> "
            f"{100*s0.get('new_xval', 0):.1f}%")
    if t2:
        findings.append(
            f"Verbal fraction: {100*t2.get('old_verbal_fraction', 0):.1f}% -> "
            f"{100*t2.get('new_verbal_fraction', 0):.1f}%")
    if t3:
        findings.append(
            f"T1 stability: {100*t3.get('stability_fraction', 0):.1f}% "
            f"({t3.get('n_stable', 0)}/{t3.get('old_n_identifications', 0)})")
    if t4:
        findings.append(
            f"Paradigms: {t4.get('n_paradigms', 0)} "
            f"(was {t4.get('old_n_paradigms', 0)} in P73), "
            f"mean size {t4.get('mean_paradigm_size', 0):.1f}")
        findings.append(
            f"Consistent codas: {t4.get('n_consistent_codas', 0)} (of 3: n, s, t)")
    if t5:
        findings.append(
            f"Template selectivity: {t5.get('template_selectivity', 0):.2f}x")
        findings.append(
            f"Distributional coverage: {100*t5.get('distributional_coverage', 0):.1f}%")

    result = Phase75IntegrateResult(
        tracks_run=tracks_run,
        track_summaries=track_summaries,
        n_changed_tokens=s0.get('n_changed', 0),
        new_dict_hit=s0.get('new_dict_hit', 0.0),
        delta_dict_hit=s0.get('delta_dict_hit', 0.0),
        new_xval=s0.get('new_xval', 0.0),
        total_gates_available=total_gates,
        total_gates_passed=total_passed,
        step0_pass=s0.get('gate_r0', False),
        track1_gates=t1_gates,
        track2_gates=t2_gates,
        track3_gates=t3_gates,
        track4_gates=t4_gates,
        track5_gates=t5_gates,
        old_verbal_fraction=t2.get('old_verbal_fraction', 0.0),
        new_verbal_fraction=t2.get('new_verbal_fraction', 0.0),
        t1_stability=t3.get('stability_fraction', 0.0),
        template_selectivity=t5.get('template_selectivity', 0.0),
        lexical_selectivity=t5.get('lexical_selectivity', 0.0),
        distributional_coverage=t5.get('distributional_coverage', 0.0),
        key_findings=findings,
        verdict=verdict,
        runtime_seconds=time.time() - t0,
    )

    path = _save_json(rd, 'p75_integrate.json', asdict(result))

    # --- Print summary ---
    print(f"\n  Tracks run: {len(tracks_run)}")
    for s in track_summaries:
        print(f"    {s['track']}: {s['verdict']} ({s.get('gates', '?')})")

    print(f"\n  Gates: {total_passed}/{total_gates}")
    print(f"\n  Key findings:")
    for f in findings:
        print(f"    - {f}")

    print(f"\n  VERDICT: {verdict}")
    print(f"  Saved: {path}")

    return result


def run_phase75() -> Phase75IntegrateResult:
    """Run full Phase 75 pipeline (3-coda model)."""
    t0 = time.time()
    print("=" * 60)
    print("Phase 75: 3-Coda Model Pipeline (Connector+Descender -> Null)")
    print("=" * 60)

    # Step 0: Re-decode (required by all tracks)
    from voynich.phases.p75_redecode import run_redecode_3coda
    print("\n" + "-" * 60)
    run_redecode_3coda()

    # Track 1: Revalidation (mandatory gate)
    from voynich.phases.p75_revalidate import run_revalidate_3coda
    print("\n" + "-" * 60)
    run_revalidate_3coda()

    # Track 2: Grammar
    from voynich.phases.p75_grammar import run_grammar_3coda
    print("\n" + "-" * 60)
    run_grammar_3coda()

    # Track 3: T1
    from voynich.phases.p75_t1 import run_t1_3coda
    print("\n" + "-" * 60)
    run_t1_3coda()

    # Track 4: Paradigms (depends on Track 3)
    from voynich.phases.p75_paradigms import run_paradigms_3coda
    print("\n" + "-" * 60)
    run_paradigms_3coda()

    # Track 5: Readings (depends on Tracks 2-4 + Phase 74)
    from voynich.phases.p75_readings import run_readings_3coda
    print("\n" + "-" * 60)
    run_readings_3coda()

    # Integration
    print("\n" + "-" * 60)
    result = run_phase75_verdict()

    total_time = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"Phase 75 complete in {total_time:.0f}s")
    print(f"{'=' * 60}")

    return result
