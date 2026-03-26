"""Phase 63 Integration: Visual sign comparison verdict."""

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

from voynich.core._paths import results_dir


@dataclass
class GateResult:
    name: str = ''
    threshold: str = ''
    value: str = ''
    passed: bool = False


@dataclass
class Phase63Result:
    gates: List[GateResult] = field(default_factory=list)
    gates_passed: int = 0
    gates_total: int = 5
    overall_verdict: str = ''
    summary: str = ''
    elapsed: float = 0.0


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
    return obj


def _save_json(rd: str, filename: str, data: Any) -> str:
    path = os.path.join(rd, filename)
    with open(path, 'w') as f:
        json.dump(_convert(data), f, indent=2)
    return path


def _safe_load(path: str) -> Any:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def run_phase63_verdict():
    """Phase 63 verdict: integrate all validation results."""
    t0 = time.time()
    rd = str(results_dir())

    validate = _safe_load(os.path.join(rd, 'p63_validate.json'))
    if not validate:
        print("ERROR: Validation data not found. Run vis-validate first.")
        return

    print("=" * 60)
    print("PHASE 63: VISUAL SIGN COMPARISON — VERDICT")
    print("=" * 60)

    gates = []

    # A-G1: >= 8/25 T_P15 assignments rank in top-5
    strong = validate.get('strong', 0)
    g1 = GateResult(
        name='A-G1: Top-5 visual support',
        threshold='>= 8/25 assignments in top-5',
        value=f'{strong}/25',
        passed=strong >= 8,
    )
    gates.append(g1)

    # A-G2: >= 15/25 assignments rank in top-15
    moderate_plus = validate.get('strong', 0) + validate.get('moderate', 0)
    g2 = GateResult(
        name='A-G2: Top-15 visual support',
        threshold='>= 15/25 assignments in top-15',
        value=f'{moderate_plus}/25',
        passed=moderate_plus >= 15,
    )
    gates.append(g2)

    # A-G3: Permutation test p < 0.05
    perm_p = validate.get('perm_p', 1.0)
    g3 = GateResult(
        name='A-G3: Permutation test',
        threshold='p < 0.05',
        value=f'p = {perm_p:.4f}',
        passed=perm_p < 0.05,
    )
    gates.append(g3)

    # A-G4: Family clustering z > 1.65
    family_z = validate.get('family_z', 0.0)
    g4 = GateResult(
        name='A-G4: Family clustering',
        threshold='z > 1.65',
        value=f'z = {family_z:.2f}',
        passed=family_z > 1.65,
    )
    gates.append(g4)

    # A-G5: >= 3 confirmed syllables in top-3
    n_conf_top3 = validate.get('n_confirmed_in_top3', 0)
    g5 = GateResult(
        name='A-G5: Confirmed syllables in top-3',
        threshold='>= 3 confirmed in top-3',
        value=str(n_conf_top3),
        passed=n_conf_top3 >= 3,
    )
    gates.append(g5)

    n_passed = sum(1 for g in gates if g.passed)

    # Verdict
    if n_passed >= 4:
        verdict = 'VISUAL_CONFIRMED'
    elif n_passed >= 2:
        verdict = 'VISUAL_PARTIAL'
    else:
        verdict = 'VISUAL_NO_SIGNAL'

    # Print gates
    print()
    for g in gates:
        status = 'PASS' if g.passed else 'FAIL'
        print(f"  [{status}] {g.name}")
        print(f"         Threshold: {g.threshold}")
        print(f"         Value:     {g.value}")
        print()

    print(f"  Gates passed: {n_passed}/{len(gates)}")
    print(f"\n  VERDICT: {verdict}")

    summary = (
        f"{verdict} ({n_passed}/5 gates). "
        f"Top-5: {strong}, top-15: {moderate_plus}, "
        f"perm p={perm_p:.4f}, family z={family_z:.2f}, "
        f"confirmed top-3: {n_conf_top3}."
    )

    result = Phase63Result(
        gates=gates,
        gates_passed=n_passed,
        overall_verdict=verdict,
        summary=summary,
        elapsed=time.time() - t0,
    )

    _save_json(rd, 'p63_integrate.json', asdict(result))

    print(f"\n  {summary}")
    print(f"  Elapsed: {result.elapsed:.1f}s")


def run_phase63():
    """Run full Phase 63 pipeline: A1 through A6 + integration."""
    print("=" * 60)
    print("PHASE 63: VISUAL SIGN COMPARISON (FULL PIPELINE)")
    print("=" * 60)

    from voynich.phases.p63_render import run_p63_render
    from voynich.phases.p63_normalize import run_p63_normalize
    from voynich.phases.p63_embed import run_p63_embed
    from voynich.phases.p63_similarity import run_p63_similarity
    from voynich.phases.p63_validate import run_p63_validate
    from voynich.phases.p63_report import run_p63_report

    print("\n--- Step A1: Render EVA Glyphs ---")
    run_p63_render()

    print("\n--- Step A2: Normalize Images ---")
    run_p63_normalize()

    print("\n--- Step A3: Embed via Gemini ---")
    run_p63_embed()

    print("\n--- Step A4: Compute Similarity ---")
    run_p63_similarity()

    print("\n--- Step A5: Validate T_P15 ---")
    run_p63_validate()

    print("\n--- Step A6: Generate Report ---")
    run_p63_report()

    print("\n--- Integration ---")
    run_phase63_verdict()
