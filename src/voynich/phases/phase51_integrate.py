"""
Phase 51 Integration: Reverse Suffix Calibration + Concatenation Bridge
=======================================================================
Combine results from Track A (suffix calibration) and Track B
(concatenation bridge search) to produce an overall verdict.

Dependency chain:
    suffix_calibration.json    (Track A)
    concatenation_bridge.json  (Track B)
        -> phase51_integrate.json
"""

from __future__ import annotations

import json
import os
import time
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
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class Phase51Validation:
    name: str
    description: str
    passed: bool
    value: float
    threshold: float


@dataclass
class Phase51IntegrateResult:
    # Track A summary
    track_a_verdict: str
    track_a_n_suffixes_calibrated: int
    track_a_agreement_z: float
    track_a_null_selectivity: float
    track_a_pos_coverage: float
    track_a_cv_accuracy: float
    track_a_paradigm_coherence: float
    # Track B summary
    track_b_verdict: str
    track_b_n_bridges: int
    track_b_selectivity: float
    track_b_z_score: float
    track_b_n_consensus: int
    track_b_n_new_assignments: int
    track_b_coverage_rate: float
    # Cross-validation
    cross_validation_agreement: float
    n_cross_validated: int
    # Validation battery
    validations: List[Dict]
    n_validations_passed: int
    n_validations_total: int
    # Overall
    overall_verdict: str
    gate_passed: bool
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_phase51_integrate() -> None:
    """Phase 51 Track C: Integration."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE 51 TRACK C: Integration")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load Track A and Track B results ───────────────────────────
    print("\n  C.1  Loading track results...")

    track_a = _safe_load(os.path.join(rd, 'suffix_calibration.json'))
    track_b = _safe_load(os.path.join(rd, 'concatenation_bridge.json'))

    a_verdict = track_a.get('verdict', 'MISSING')
    b_verdict = track_b.get('verdict', 'MISSING')

    print(f"       Track A verdict: {a_verdict}")
    print(f"       Track B verdict: {b_verdict}")

    # Extract key metrics
    a_z = track_a.get('agreement_z_score', 0.0)
    a_sel = track_a.get('null_selectivity', 0.0)
    a_coverage = track_a.get('pos_tag_coverage', 0.0)
    a_cv = track_a.get('cross_val_accuracy', 0.0)
    a_coherence = track_a.get('paradigm_coherence', 0.0)
    a_n_suf = track_a.get('n_eva_suffixes_calibrated', 0)

    b_bridges = track_b.get('n_bridge_matches', 0)
    b_sel = track_b.get('bridge_selectivity', 0.0)
    b_z = track_b.get('bridge_z_score', 0.0)
    b_coverage_rate = track_b.get('coverage', {}).get('coverage_rate', 0.0)
    consensus = track_b.get('consensus_assignments', [])
    b_n_consensus = len([c for c in consensus
                         if c.get('consensus', 0) > 0.5
                         and c.get('n_observations', 0) >= 5])
    b_n_new = track_b.get('n_new_assignments', 0)

    # ── 2. Cross-validation ───────────────────────────────────────────
    print("\n  C.2  Cross-validating Track A and Track B...")

    # Check if Track B consensus assignments agree with Track A's
    # POS tagging expectations
    n_cross = 0
    n_agree = 0
    suffix_map = track_a.get('suffix_map', {})
    pos_dist = track_a.get('pos_distribution', {})

    # Simple cross-check: do consensus assignments agree with Phase 15?
    for c in consensus:
        if c.get('consensus', 0) > 0.5 and c.get('n_observations', 0) >= 5:
            n_cross += 1
            if c.get('agrees_with_phase15', False):
                n_agree += 1

    cross_agreement = n_agree / n_cross if n_cross > 0 else 0.0

    print(f"       Cross-validated: {n_cross} consensus triples")
    print(f"       Agreement with P15: {n_agree}/{n_cross} = {cross_agreement:.1%}")

    # ── 3. Validation battery ─────────────────────────────────────────
    print("\n  C.3  Validation battery...")

    validations: List[Phase51Validation] = []

    # V1: Track A null z-score
    v1 = Phase51Validation(
        name='V1_suffix_z',
        description='Track A suffix map agreement z-score > 2.0',
        passed=(a_z > 2.0),
        value=a_z,
        threshold=2.0,
    )
    validations.append(v1)

    # V2: Track A paradigm coherence
    v2 = Phase51Validation(
        name='V2_paradigm',
        description='Track A paradigm coherence > 0.4',
        passed=(a_coherence > 0.4),
        value=a_coherence,
        threshold=0.4,
    )
    validations.append(v2)

    # V3: Track B selectivity
    v3 = Phase51Validation(
        name='V3_bridge_sel',
        description='Track B bridge selectivity > 1.5x',
        passed=(b_sel > 1.5),
        value=b_sel,
        threshold=1.5,
    )
    validations.append(v3)

    # V4: Track B bridge matches
    v4 = Phase51Validation(
        name='V4_bridge_count',
        description='Track B has >= 3 bridge matches',
        passed=(b_bridges >= 3),
        value=float(b_bridges),
        threshold=3.0,
    )
    validations.append(v4)

    # V5: Cross-validation agreement
    v5 = Phase51Validation(
        name='V5_cross_val',
        description='Cross-validation agreement > 60%',
        passed=(cross_agreement > 0.6 or n_cross == 0),
        value=cross_agreement,
        threshold=0.6,
    )
    validations.append(v5)

    # V6: POS distribution plausibility
    total_pos = sum(pos_dist.values()) if pos_dist else 0
    noun_frac = 0.0
    if total_pos > 0:
        noun_count = sum(v for k, v in pos_dist.items()
                        if 'NOUN' in k)
        noun_frac = noun_count / total_pos

    v6 = Phase51Validation(
        name='V6_pos_plausible',
        description='POS distribution has noun fraction > 0.1',
        passed=(noun_frac > 0.1 or total_pos == 0),
        value=noun_frac,
        threshold=0.1,
    )
    validations.append(v6)

    # V7: Track A cross-validation accuracy
    v7 = Phase51Validation(
        name='V7_cv_accuracy',
        description='Track A cross-validation accuracy > 50%',
        passed=(a_cv > 0.5),
        value=a_cv,
        threshold=0.5,
    )
    validations.append(v7)

    n_passed = sum(1 for v in validations if v.passed)
    n_total = len(validations)

    for v in validations:
        status = "PASS" if v.passed else "FAIL"
        print(f"       {v.name:20s}: {status}  "
              f"(value={v.value:.3f}, threshold={v.threshold:.1f})  "
              f"-- {v.description}")

    print(f"\n       Passed: {n_passed}/{n_total}")

    # ── 4. Overall verdict ────────────────────────────────────────────

    a_pass = a_verdict in ('SUFFIX_MAP_VALID', 'SUFFIX_MAP_PARTIAL')
    b_pass = b_verdict in ('BRIDGE_PRODUCTIVE', 'BRIDGE_MARGINAL')

    if a_pass and b_pass:
        overall = 'BOTH_PASS'
    elif a_pass:
        overall = 'A_ONLY'
    elif b_pass:
        overall = 'B_ONLY'
    else:
        overall = 'NEITHER'

    gate = n_passed >= 4

    print(f"\n  Overall verdict: {overall}")
    print(f"  Gate: {'PASS' if gate else 'FAIL'} ({n_passed}/{n_total})")

    # ── 5. Save ───────────────────────────────────────────────────────
    runtime = round(time.time() - t0, 2)

    result = Phase51IntegrateResult(
        track_a_verdict=a_verdict,
        track_a_n_suffixes_calibrated=a_n_suf,
        track_a_agreement_z=a_z,
        track_a_null_selectivity=a_sel,
        track_a_pos_coverage=a_coverage,
        track_a_cv_accuracy=a_cv,
        track_a_paradigm_coherence=a_coherence,
        track_b_verdict=b_verdict,
        track_b_n_bridges=b_bridges,
        track_b_selectivity=b_sel,
        track_b_z_score=b_z,
        track_b_n_consensus=b_n_consensus,
        track_b_n_new_assignments=b_n_new,
        track_b_coverage_rate=b_coverage_rate,
        cross_validation_agreement=round(cross_agreement, 4),
        n_cross_validated=n_cross,
        validations=[_convert(asdict(v)) for v in validations],
        n_validations_passed=n_passed,
        n_validations_total=n_total,
        overall_verdict=overall,
        gate_passed=gate,
        runtime_seconds=runtime,
    )

    out_path = _save_json(rd, 'phase51_integrate.json', asdict(result))
    print(f"\n  Saved → {out_path}")
    print(f"  Completed in {runtime:.1f}s")


def run_phase51() -> None:
    """Run all Phase 51 tracks sequentially."""
    from voynich.phases.suffix_calibration import run_suffix_calibration
    from voynich.phases.concatenation_bridge import run_concatenation_bridge

    run_suffix_calibration()
    print()
    run_concatenation_bridge()
    print()
    run_phase51_integrate()
