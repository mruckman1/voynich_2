"""
Phase 61 Integration: Verdict + Full Pipeline
===============================================
Loads all 4 track results, summarizes, and produces an overall verdict.

Tracks:
  A: Deep recipe reading (phase61_deep_recipes.json)
  B: Full CV permutation under CVC (phase61_cvc_full_permutation.json)
  C: Costamagna sequence rules (phase61_costamagna_sequences.json)
  D: Zodiac CVC re-decode (phase61_zodiac_cvc.json)

Dependency chain:
    results/phase61_deep_recipes.json           (Track A)
    results/phase61_cvc_full_permutation.json   (Track B)
    results/phase61_costamagna_sequences.json   (Track C)
    results/phase61_zodiac_cvc.json             (Track D)
        -> results/phase61_integrate.json
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
    letter: str
    name: str
    key_metric: str
    key_value: str
    gates_passed: int
    gates_total: int
    passed: bool


@dataclass
class Phase61Result:
    phase: str = "61"
    experiment: str = "phase61_integrate"
    tracks: List[Dict[str, Any]] = field(default_factory=list)
    n_tracks_passed: int = 0
    n_tracks_total: int = 4
    # Key metrics from each track
    mean_reading_confidence: float = 0.0
    p_count: float = 0.0
    p_cvc_coherence: float = 0.0
    best_selectivity: float = 0.0
    zodiac_selectivity: float = 0.0
    verdict: str = ""
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

def run_phase61_verdict():
    """Load all 4 track results and produce verdict."""
    t0 = time.time()
    print("=" * 70)
    print("Phase 61 Integration: Verdict")
    print("=" * 70)

    rd = str(_results_dir())
    tracks: List[TrackSummary] = []

    # Track A: Deep Recipe Reading
    a_data = _safe_load(os.path.join(rd, 'phase61_deep_recipes.json'))
    a_passed = a_data.get('gate_passed', False)
    a_gates = a_data.get('gates_passed', 0)
    a_conf = a_data.get('mean_reading_confidence', 0.0)
    a_verb = a_data.get('n_with_verb', 0)
    a_ingr = a_data.get('n_with_ingredient', 0)

    tracks.append(TrackSummary(
        letter='A', name='Deep Recipe Reading',
        key_metric='reading_confidence',
        key_value=f"conf={a_conf:.3f}, verbs={a_verb}/5, ingr={a_ingr}/5",
        gates_passed=a_gates, gates_total=5,
        passed=a_passed,
    ))

    # Track B: Full CV Permutation
    b_data = _safe_load(os.path.join(rd, 'phase61_cvc_full_permutation.json'))
    b_passed = b_data.get('gate_passed', False)
    b_gates = b_data.get('gates_passed', 0)
    b_p_count = b_data.get('p_count', 1.0)
    b_p_cvc = b_data.get('p_cvc_coherence', 1.0)

    tracks.append(TrackSummary(
        letter='B', name='Full CV Permutation (CVC)',
        key_metric='p_values',
        key_value=f"p_count={b_p_count:.4f}, p_cvc_coh={b_p_cvc:.4f}",
        gates_passed=b_gates, gates_total=4,
        passed=b_passed,
    ))

    # Track C: Costamagna Sequences
    c_data = _safe_load(os.path.join(rd, 'phase61_costamagna_sequences.json'))
    c_passed = c_data.get('gate_passed', False)
    c_gates = c_data.get('gates_passed', 0)
    c_sel = c_data.get('best_selectivity', 0.0)
    c_lower = c_data.get('n_real_lower', 0)

    tracks.append(TrackSummary(
        letter='C', name='Costamagna Sequence Rules',
        key_metric='selectivity',
        key_value=f"best_sel={c_sel:.2f}×, real<null={c_lower}",
        gates_passed=c_gates, gates_total=4,
        passed=c_passed,
    ))

    # Track D: Zodiac CVC
    d_data = _safe_load(os.path.join(rd, 'phase61_zodiac_cvc.json'))
    d_passed = d_data.get('gate_passed', False)
    d_gates = d_data.get('gates_passed', 0)
    d_sel = d_data.get('selectivity', 0.0)
    d_matches = d_data.get('n_matches_ed2', 0)
    d_correct = d_data.get('n_correct_folio', 0)

    tracks.append(TrackSummary(
        letter='D', name='Zodiac CVC Re-Decode',
        key_metric='folio_selectivity',
        key_value=f"sel={d_sel:.2f}×, matches={d_matches}, correct={d_correct}",
        gates_passed=d_gates, gates_total=5,
        passed=d_passed,
    ))

    # Overall verdict
    n_passed = sum(1 for t in tracks if t.passed)
    if n_passed >= 4:
        verdict = "PHASE61_COMPREHENSIVE"
    elif n_passed >= 3:
        verdict = "PHASE61_STRONG"
    elif n_passed >= 2:
        verdict = "PHASE61_PARTIAL"
    elif n_passed >= 1:
        verdict = "PHASE61_MARGINAL"
    else:
        verdict = "PHASE61_NEGATIVE"

    # Print summary
    print(f"\n  Track Summary:")
    print(f"  {'Track':<8} {'Name':<30} {'Gates':>10} {'Status':>10}")
    print(f"  {'-'*8} {'-'*30} {'-'*10} {'-'*10}")
    for t in tracks:
        status = "PASS" if t.passed else "FAIL"
        print(f"  {t.letter:<8} {t.name:<30} "
              f"{t.gates_passed}/{t.gates_total:>7} {status:>10}")
        print(f"           {t.key_value}")

    print(f"\n  Tracks passed: {n_passed}/4")
    print(f"  VERDICT: {verdict}")

    result = Phase61Result(
        tracks=[_convert(asdict(t)) for t in tracks],
        n_tracks_passed=n_passed,
        mean_reading_confidence=round(a_conf, 3),
        p_count=round(b_p_count, 4),
        p_cvc_coherence=round(b_p_cvc, 4),
        best_selectivity=round(c_sel, 2),
        zodiac_selectivity=round(d_sel, 2),
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    path = _save_json(rd, 'phase61_integrate.json', result)
    print(f"\n  Saved: {path}")


def run_phase61():
    """Run full Phase 61 pipeline: all 4 tracks + verdict."""
    t0 = time.time()
    print("=" * 70)
    print("Phase 61: Full Pipeline")
    print("=" * 70)

    # Track A: Deep Recipe Reading
    print("\n" + "=" * 70)
    print("TRACK A: Deep Pharmaceutical Recipe Reading")
    print("=" * 70)
    from voynich.phases.deep_recipe_reading import run_deep_recipes
    run_deep_recipes()

    # Track B: Full CV Permutation (longest ~45-60 min)
    print("\n" + "=" * 70)
    print("TRACK B: Full CV Permutation Under CVC Decode")
    print("=" * 70)
    from voynich.phases.cvc_full_permutation import run_cvc_full_perm
    run_cvc_full_perm()

    # Track C: Costamagna Sequence Rules
    print("\n" + "=" * 70)
    print("TRACK C: Costamagna Combination Rules")
    print("=" * 70)
    from voynich.phases.costamagna_sequence_rules import run_cost_sequences
    run_cost_sequences()

    # Track D: Zodiac CVC
    print("\n" + "=" * 70)
    print("TRACK D: Zodiac Labels Under CVC Decode")
    print("=" * 70)
    from voynich.phases.zodiac_cvc import run_zodiac_cvc
    run_zodiac_cvc()

    # Integration
    print("\n" + "=" * 70)
    print("INTEGRATION")
    print("=" * 70)
    run_phase61_verdict()

    total = time.time() - t0
    print(f"\n  Phase 61 total runtime: {total:.1f}s ({total/60:.1f}m)")
