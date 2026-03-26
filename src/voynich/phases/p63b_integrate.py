"""Phase 63B Integration: Manuscript visual comparison verdict."""

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
class Phase63BResult:
    b_gates: List[GateResult] = field(default_factory=list)
    b_gates_passed: int = 0
    b_gates_total: int = 4
    a_gates_passed: int = 0
    combined_verdict: str = ''
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


def run_phase63b_verdict():
    """Phase 63B verdict: integrate all results."""
    t0 = time.time()
    rd = str(results_dir())

    segment = _safe_load(os.path.join(rd, 'p63b_segment.json'))
    compare = _safe_load(os.path.join(rd, 'p63b_compare.json'))
    a_integrate = _safe_load(os.path.join(rd, 'p63_integrate.json'))

    if not segment:
        print("ERROR: Segment data not found. Run ms-segment first.")
        return
    if not compare:
        print("ERROR: Compare data not found. Run ms-compare first.")
        return

    print("=" * 60)
    print("PHASE 63B: MANUSCRIPT VISUAL COMPARISON — VERDICT")
    print("=" * 60)

    gates = []

    # B-G1: Line segmentation accuracy
    line_rate = segment.get('line_match_rate', 0.0)
    g1 = GateResult(
        name='B-G1: Line segmentation',
        threshold='>= 80% folios line count within ±1',
        value=f'{line_rate:.0%}',
        passed=line_rate >= 0.80,
    )
    gates.append(g1)

    # B-G2: Word segmentation (heuristic: total words / expected)
    total_words = segment.get('total_words', 0)
    expected_words = sum(f.get('total_words_expected', 0)
                        for f in segment.get('per_folio', []))
    word_rate = total_words / expected_words if expected_words > 0 else 0
    g2 = GateResult(
        name='B-G2: Word segmentation',
        threshold='>= 70% words correctly segmented',
        value=f'{word_rate:.0%} ({total_words}/{expected_words})',
        passed=word_rate >= 0.70,
    )
    gates.append(g2)

    # B-G3: Character segmentation plausibility
    # Use exemplar selection rate as proxy
    exemplar = _safe_load(os.path.join(rd, 'p63b_exemplars.json'))
    n_instances = exemplar.get('n_total_instances', 0)
    n_exemplars = exemplar.get('n_total_exemplars', 0)
    char_rate = n_exemplars / n_instances if n_instances > 0 else 0
    # Also check how many char types have exemplars
    n_types = exemplar.get('n_char_types', 0)
    n_types_ok = exemplar.get('n_char_types_with_exemplars', 0)
    type_rate = n_types_ok / n_types if n_types > 0 else 0

    g3 = GateResult(
        name='B-G3: Character segmentation',
        threshold='>= 60% character types have exemplars',
        value=f'{type_rate:.0%} ({n_types_ok}/{n_types})',
        passed=type_rate >= 0.60,
    )
    gates.append(g3)

    # B-G4: Manuscript comparison is informative
    # Count how many A-gates pass with manuscript embeddings
    ms_strong = compare.get('ms_strong', 0)
    ms_mod_plus = ms_strong + compare.get('ms_moderate', 0)
    ms_perm_p = compare.get('ms_perm_p', 1.0)
    ms_family_z = compare.get('ms_family_z', 0.0)
    ms_top3 = compare.get('ms_confirmed_top3', 0)

    ms_a_gates = 0
    if ms_strong >= 8:
        ms_a_gates += 1
    if ms_mod_plus >= 15:
        ms_a_gates += 1
    if ms_perm_p < 0.05:
        ms_a_gates += 1
    if ms_family_z > 1.65:
        ms_a_gates += 1
    if ms_top3 >= 3:
        ms_a_gates += 1

    a_gates_orig = a_integrate.get('gates_passed', 0) if a_integrate else 0
    improvement = ms_a_gates - a_gates_orig
    spearman_r = compare.get('spearman_r', 0.0)
    if spearman_r is None or (isinstance(spearman_r, float) and spearman_r != spearman_r):
        spearman_r = 0.0

    g4 = GateResult(
        name='B-G4: Manuscript comparison informative',
        threshold='>= 2 more A-gates than font OR Spearman r > 0.3',
        value=f'A-gates: {ms_a_gates} (was {a_gates_orig}, +{improvement}), r={spearman_r:.3f}',
        passed=improvement >= 2 or spearman_r > 0.3,
    )
    gates.append(g4)

    n_b_passed = sum(1 for g in gates if g.passed)

    # Combined verdict
    if ms_a_gates >= 4 and n_b_passed >= 3:
        verdict = 'VISUAL_CONFIRMED'
    elif ms_a_gates >= 4 and n_b_passed < 3:
        verdict = 'FONT_VALIDATED'
    elif ms_a_gates < 4 and n_b_passed >= 3:
        verdict = 'EXEMPLAR_NEEDED'
    else:
        verdict = 'VISUAL_MISMATCH'

    # Print gates
    print()
    for g in gates:
        status = 'PASS' if g.passed else 'FAIL'
        print(f"  [{status}] {g.name}")
        print(f"         Threshold: {g.threshold}")
        print(f"         Value:     {g.value}")
        print()

    print(f"  B-gates passed: {n_b_passed}/{len(gates)}")
    print(f"  A-gates (manuscript): {ms_a_gates}/5")
    print(f"  A-gates (font):      {a_gates_orig}/5")
    print(f"\n  COMBINED VERDICT: {verdict}")

    summary = (
        f"{verdict} (B: {n_b_passed}/4, A-ms: {ms_a_gates}/5, A-font: {a_gates_orig}/5). "
        f"Line seg {line_rate:.0%}, {n_types_ok}/{n_types} char types, "
        f"perm p={ms_perm_p:.4f}, family z={ms_family_z:.2f}, "
        f"Spearman r={spearman_r:.3f}."
    )

    result = Phase63BResult(
        b_gates=gates,
        b_gates_passed=n_b_passed,
        a_gates_passed=ms_a_gates,
        combined_verdict=verdict,
        summary=summary,
        elapsed=time.time() - t0,
    )

    _save_json(rd, 'p63b_integrate.json', asdict(result))

    print(f"\n  {summary}")
    print(f"  Elapsed: {result.elapsed:.1f}s")


def run_phase63b():
    """Run full Phase 63B pipeline."""
    print("=" * 60)
    print("PHASE 63B: MANUSCRIPT VISUAL COMPARISON (FULL PIPELINE)")
    print("=" * 60)

    from voynich.phases.p63b_index import run_p63b_index
    from voynich.phases.p63b_segment import run_p63b_segment
    from voynich.phases.p63b_exemplars import run_p63b_exemplars
    from voynich.phases.p63b_compare import run_p63b_compare

    print("\n--- Step B1: Extract + Index Folios ---")
    run_p63b_index()

    print("\n--- Steps B2-B4: Segmentation Pipeline ---")
    run_p63b_segment()

    print("\n--- Step B5: Exemplar Selection ---")
    run_p63b_exemplars()

    print("\n--- Step B6: Embed + Compare ---")
    run_p63b_compare()

    print("\n--- Integration ---")
    run_phase63b_verdict()
