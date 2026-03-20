"""
Phase 60 Integration: Verdict + Full Pipeline
==============================================
Loads all 4 track results, summarizes, and produces an overall verdict.

Tracks:
  A: Corrected coda mapping (corrected_coda.json)
  B: Recalibrated coherence (recalibrated_coherence.json)
  C: CVC evaluation framework (cvc_evaluator.json)
  D: Recipe annotation (recipe_annotation.json)

Dependency chain:
    results/corrected_coda.json           (Track A)
    results/recalibrated_coherence.json   (Track B)
    results/cvc_evaluator.json            (Track C)
    results/recipe_annotation.json        (Track D)
        -> results/phase60_integrate.json
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
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TrackSummary:
    """Summary of one track's results."""
    letter: str
    name: str
    key_metric: str
    key_value: str
    gates_passed: int
    gates_total: int
    passed: bool


@dataclass
class Phase60Result:
    """Full Phase 60 integration output."""
    phase: str = "60"
    experiment: str = "phase60_integrate"
    tracks: List[TrackSummary] = field(default_factory=list)
    n_tracks_passed: int = 0
    n_tracks_total: int = 4
    best_strategy: str = ""
    corrected_bigram_z: float = 0.0
    corrected_net_signal: int = 0
    coherence_p: float = 0.0
    composite_score: float = 0.0
    mean_recipe_glossed: float = 0.0
    verdict: str = ""
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# CLI entry points
# ---------------------------------------------------------------------------

def run_phase60_verdict():
    """Load all 4 track results and produce verdict."""
    t0 = time.time()
    print("=" * 70)
    print("Phase 60 Integration: Verdict")
    print("=" * 70)

    rd = str(_results_dir())
    tracks: List[TrackSummary] = []

    # Track A: Corrected Coda
    a_data = _safe_load(os.path.join(rd, 'corrected_coda.json'))
    a_passed = a_data.get('gate_passed', False)
    a_gates = a_data.get('gates_passed', 0)
    a_best = a_data.get('best_strategy', '?')

    # Extract corrected strategy metrics
    corrected_bigram_z = 0.0
    corrected_net_signal = 0
    for s in a_data.get('strategies', []):
        if s.get('name') == 'cvc_corrected':
            corrected_bigram_z = s.get('bigram_z', 0.0)
            corrected_net_signal = s.get('net_signal', 0)

    tracks.append(TrackSummary(
        letter='A', name='Corrected Coda Mapping',
        key_metric='best_strategy',
        key_value=f"{a_best} (bigram_z={corrected_bigram_z:.2f})",
        gates_passed=a_gates, gates_total=6,
        passed=a_passed,
    ))

    # Track B: Recalibrated Coherence
    b_data = _safe_load(os.path.join(rd, 'recalibrated_coherence.json'))
    b_passed = b_data.get('gate_passed', False)
    b_gates = b_data.get('gates_passed', 0)
    b_p = b_data.get('cvc_recalibrated_p', 1.0)
    b_verdict = b_data.get('verdict', '?')

    tracks.append(TrackSummary(
        letter='B', name='Recalibrated Coherence',
        key_metric='p_all',
        key_value=f"p={b_p:.4f} ({b_verdict})",
        gates_passed=b_gates, gates_total=5,
        passed=b_passed,
    ))

    # Track C: CVC Evaluator
    c_data = _safe_load(os.path.join(rd, 'cvc_evaluator.json'))
    c_passed = c_data.get('gate_passed', False)
    c_gates = c_data.get('gates_passed', 0)
    c_best = c_data.get('best_strategy', '?')

    composite_score = 0.0
    for ev in c_data.get('evaluations', []):
        if ev.get('name') == 'cvc_corrected':
            composite_score = ev.get('composite', 0.0)

    tracks.append(TrackSummary(
        letter='C', name='CVC Evaluation Framework',
        key_metric='composite',
        key_value=f"{c_best} (composite={composite_score:.4f})",
        gates_passed=c_gates, gates_total=3,
        passed=c_passed,
    ))

    # Track D: Recipe Annotation
    d_data = _safe_load(os.path.join(rd, 'recipe_annotation.json'))
    d_passed = d_data.get('gate_passed', False)
    d_gates = d_data.get('gates_passed', 0)
    d_glossed = d_data.get('mean_glossed_fraction', 0.0)
    d_consec = d_data.get('max_consecutive_glossed', 0)

    tracks.append(TrackSummary(
        letter='D', name='Recipe Annotation',
        key_metric='mean_glossed',
        key_value=f"{d_glossed:.1%} glossed, {d_consec} max consecutive",
        gates_passed=d_gates, gates_total=5,
        passed=d_passed,
    ))

    # Overall verdict
    n_passed = sum(1 for t in tracks if t.passed)
    if n_passed >= 4:
        verdict = "CVC_CORRECTED_VALIDATED"
    elif n_passed >= 3:
        verdict = "CVC_CORRECTED_PARTIAL"
    elif n_passed >= 2:
        verdict = "CVC_CORRECTED_MARGINAL"
    elif n_passed >= 1:
        verdict = "CVC_CORRECTED_WEAK"
    else:
        verdict = "CVC_NO_IMPROVEMENT"

    # Print summary
    print(f"\n  Track Summary:")
    print(f"  {'Track':<8} {'Name':<30} {'Gates':>10} {'Status':>10}")
    print(f"  {'-'*8} {'-'*30} {'-'*10} {'-'*10}")
    for t in tracks:
        status = "PASS" if t.passed else "FAIL"
        print(f"  {t.letter:<8} {t.name:<30} {t.gates_passed}/{t.gates_total:>7} "
              f"{status:>10}")

    print(f"\n  Key Metrics:")
    print(f"    Corrected bigram z: {corrected_bigram_z:.2f}")
    print(f"    Corrected net signal: {corrected_net_signal}")
    print(f"    Coherence p: {b_p:.4f}")
    print(f"    Composite score: {composite_score:.4f}")
    print(f"    Recipe glossed: {d_glossed:.1%}")

    print(f"\n  Tracks passed: {n_passed}/4")
    print(f"  VERDICT: {verdict}")

    result = Phase60Result(
        tracks=tracks,
        n_tracks_passed=n_passed,
        best_strategy=a_best,
        corrected_bigram_z=round(corrected_bigram_z, 2),
        corrected_net_signal=corrected_net_signal,
        coherence_p=round(b_p, 4),
        composite_score=round(composite_score, 4),
        mean_recipe_glossed=round(d_glossed, 3),
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    path = _save_json(rd, 'phase60_integrate.json', result)
    print(f"\n  Saved: {path}")


def run_phase60():
    """Run full Phase 60 pipeline: all 4 tracks + verdict."""
    t0 = time.time()
    print("=" * 70)
    print("Phase 60: Full Pipeline")
    print("=" * 70)

    # Track A: Corrected Coda Mapping
    print("\n" + "=" * 70)
    print("TRACK A: Corrected Coda Mapping")
    print("=" * 70)
    from voynich.phases.corrected_coda import run_corrected_coda
    run_corrected_coda()

    # Track B: Recalibrated Coherence
    print("\n" + "=" * 70)
    print("TRACK B: Recalibrated Coherence")
    print("=" * 70)
    from voynich.phases.recalibrated_coherence import run_recal_coherence
    run_recal_coherence()

    # Track C: CVC Evaluation Framework
    print("\n" + "=" * 70)
    print("TRACK C: CVC Evaluation Framework")
    print("=" * 70)
    from voynich.phases.cvc_evaluator import run_cvc_eval
    run_cvc_eval()

    # Track D: Recipe Annotation
    print("\n" + "=" * 70)
    print("TRACK D: Recipe Annotation")
    print("=" * 70)
    from voynich.phases.recipe_annotation import run_recipe_annotate
    run_recipe_annotate()

    # Integration
    print("\n" + "=" * 70)
    print("INTEGRATION")
    print("=" * 70)
    run_phase60_verdict()

    total = time.time() - t0
    print(f"\n  Phase 60 total runtime: {total:.1f}s ({total/60:.1f}m)")
