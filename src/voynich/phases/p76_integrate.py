"""
Phase 76: Integration -- Multi-Track Resolution Pipeline
==========================================================
Collect results from Tracks 1-4, determine overall verdict.

Dependency chain:
    results/p76_wildcard_prop.json       (Track 1)
    results/p76_skeleton.json            (Track 2)
    results/p76_freq_gap.json            (Track 3)
    results/p76_gapfill.json             (Track 4)
        -> results/p76_integrate.json
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
class Phase76IntegrateResult:
    phase: str = "76"
    step: str = "76.5"
    experiment: str = "phase76_integrate"
    tracks_run: List[str] = field(default_factory=list)
    track_summaries: List[Dict[str, Any]] = field(default_factory=list)
    # Track 1 summary
    n_resolved: int = 0
    n_likely: int = 0
    loo_accuracy: float = 0.0
    # Track 2 summary
    n_parallel_pairs: int = 0
    n_diagnostic_diffs: int = 0
    template_selectivity: float = 0.0
    # Track 3 summary
    gap_mapped: bool = False
    freq_gap_coverage: float = 0.0
    # Track 4 summary
    gapfill_skipped: bool = True
    n_accepted_proposals: int = 0
    # Gate totals
    total_gates_available: int = 0
    total_gates_passed: int = 0
    track1_gates: int = 0
    track2_gates: int = 0
    track3_gates: int = 0
    track4_gates: int = 0
    # Key findings
    key_findings: List[str] = field(default_factory=list)
    verdict: str = "UNKNOWN"
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Verdict logic
# ---------------------------------------------------------------------------

def _determine_verdict(t1, t2, t3, t4) -> str:
    """
    TRIPLES_RESOLVED:
      Track 1 >=3 RESOLVED + LOO >50% + Track 4 >=1 accepted.

    TRIPLES_CONSTRAINED:
      Track 1 (>=3 RESOLVED or >=5 LIKELY+) + LOO >50%.

    STRUCTURE_MAPPED:
      Track 2 section-selective + Track 3 gap mapped.

    NO_PROGRESS:
      Otherwise.
    """
    t1_n_resolved = t1.get('n_resolved', 0)
    t1_n_likely = t1.get('n_likely', 0)
    t1_loo = t1.get('loo_accuracy', 0.0)

    t2_selective = t2.get('recipe_sections_higher', False)
    t2_gates = t2.get('gates_passed', 0)

    t3_gap_mapped = t3.get('gap_mapped', False)
    t3_gates = t3.get('gates_passed', 0)

    t4_skipped = t4.get('skipped', True)
    t4_n_accepted = t4.get('n_accepted', 0)

    # TRIPLES_RESOLVED: strongest outcome
    if t1_n_resolved >= 3 and t1_loo > 0.50 and not t4_skipped and t4_n_accepted >= 1:
        return 'TRIPLES_RESOLVED'

    # TRIPLES_CONSTRAINED
    enough_triples = (t1_n_resolved >= 3) or (t1_n_resolved + t1_n_likely >= 5)
    if enough_triples and t1_loo > 0.50:
        return 'TRIPLES_CONSTRAINED'

    # STRUCTURE_MAPPED
    if t2_selective and t2_gates >= 2 and (t3_gap_mapped or t3_gates >= 2):
        return 'STRUCTURE_MAPPED'

    return 'NO_PROGRESS'


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def run_phase76_verdict() -> Phase76IntegrateResult:
    """Integration: collect all track results and produce verdict."""
    t0 = time.time()
    rd = str(_results_dir())

    print("Phase 76 -- Integration Verdict")
    print("=" * 35)

    # --- Load all results ---
    t1 = _safe_load(os.path.join(rd, 'p76_wildcard_prop.json'))
    t2 = _safe_load(os.path.join(rd, 'p76_skeleton.json'))
    t3 = _safe_load(os.path.join(rd, 'p76_freq_gap.json'))
    t4 = _safe_load(os.path.join(rd, 'p76_gapfill.json'))

    tracks_run = []
    track_summaries = []

    # Track 1: Wildcard Propagation
    t1_gates = t1.get('gates_passed', 0) if t1 else 0
    if t1:
        tracks_run.append('wildcard_prop')
        track_summaries.append({
            'track': 'Track 1: Wildcard Propagation',
            'verdict': t1.get('verdict', '?'),
            'gates': f"{t1_gates}/{t1.get('total_gates', '?')}",
            'n_resolved': t1.get('n_resolved', 0),
            'n_likely': t1.get('n_likely', 0),
            'loo_accuracy': f"{100*t1.get('loo_accuracy', 0):.1f}%",
        })

    # Track 2: Skeleton Parse
    t2_gates = t2.get('gates_passed', 0) if t2 else 0
    if t2:
        tracks_run.append('skeleton')
        track_summaries.append({
            'track': 'Track 2: Skeleton Parse',
            'verdict': t2.get('verdict', '?'),
            'gates': f"{t2_gates}/4",
            'n_parallel': t2.get('n_parallel_pairs', 0),
            'n_diagnostic': t2.get('n_diagnostic_diffs', 0),
            'template_sel': f"{t2.get('template_selectivity', 0):.2f}x",
        })

    # Track 3: Frequency Gap
    t3_gates = t3.get('gates_passed', 0) if t3 else 0
    if t3:
        tracks_run.append('freq_gap')
        track_summaries.append({
            'track': 'Track 3: Frequency Gap',
            'verdict': t3.get('verdict', '?'),
            'gates': f"{t3_gates}/{t3.get('total_gates', '?')}",
            'gap_mapped': t3.get('gap_mapped', False),
            'coverage': f"{100*t3.get('freq_gap_coverage', 0):.1f}%",
        })

    # Track 4: Conditional Gap-Fill
    t4_gates = t4.get('gates_passed', 0) if t4 else 0
    if t4:
        tracks_run.append('gapfill')
        skipped = t4.get('skipped', True)
        track_summaries.append({
            'track': 'Track 4: Conditional Gap-Fill',
            'verdict': t4.get('verdict', '?'),
            'gates': f"{t4_gates}/3",
            'skipped': skipped,
            'n_accepted': t4.get('n_accepted', 0) if not skipped else 'N/A',
        })

    # --- Count total gates ---
    # Track 1: variable (use reported total), Track 2: 4, Track 3: variable, Track 4: 3
    t1_total = t1.get('total_gates', 0) if t1 else 0
    t3_total = t3.get('total_gates', 0) if t3 else 0
    total_gates = t1_total + 4 + t3_total + 3
    total_passed = t1_gates + t2_gates + t3_gates + t4_gates

    # --- Verdict ---
    verdict = _determine_verdict(t1, t2, t3, t4)

    # --- Key findings ---
    findings = []
    if t1:
        findings.append(
            f"Track 1: {t1.get('n_resolved', 0)} resolved, "
            f"{t1.get('n_likely', 0)} likely triples, "
            f"LOO accuracy {100*t1.get('loo_accuracy', 0):.1f}%")
    if t2:
        findings.append(
            f"Track 2: {t2.get('n_parallel_pairs', 0)} parallel pairs, "
            f"{t2.get('n_diagnostic_diffs', 0)} diagnostic diffs, "
            f"template selectivity {t2.get('template_selectivity', 0):.2f}x")
    if t3:
        findings.append(
            f"Track 3: gap mapped={t3.get('gap_mapped', False)}, "
            f"coverage {100*t3.get('freq_gap_coverage', 0):.1f}%")
    if t4:
        if t4.get('skipped', True):
            findings.append(
                f"Track 4: SKIPPED ({t4.get('skip_reason', 'preconditions not met')})")
        else:
            findings.append(
                f"Track 4: {t4.get('n_accepted', 0)} accepted proposals, "
                f"KA accuracy {100*t4.get('ka_accuracy', 0):.1f}%")

    result = Phase76IntegrateResult(
        tracks_run=tracks_run,
        track_summaries=track_summaries,
        n_resolved=t1.get('n_resolved', 0),
        n_likely=t1.get('n_likely', 0),
        loo_accuracy=t1.get('loo_accuracy', 0.0),
        n_parallel_pairs=t2.get('n_parallel_pairs', 0),
        n_diagnostic_diffs=t2.get('n_diagnostic_diffs', 0),
        template_selectivity=t2.get('template_selectivity', 0.0),
        gap_mapped=t3.get('gap_mapped', False),
        freq_gap_coverage=t3.get('freq_gap_coverage', 0.0),
        gapfill_skipped=t4.get('skipped', True),
        n_accepted_proposals=t4.get('n_accepted', 0),
        total_gates_available=total_gates,
        total_gates_passed=total_passed,
        track1_gates=t1_gates,
        track2_gates=t2_gates,
        track3_gates=t3_gates,
        track4_gates=t4_gates,
        key_findings=findings,
        verdict=verdict,
        runtime_seconds=time.time() - t0,
    )

    path = _save_json(rd, 'p76_integrate.json', asdict(result))

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


def run_phase76() -> Phase76IntegrateResult:
    """Run full Phase 76 pipeline."""
    t0 = time.time()
    print("=" * 60)
    print("Phase 76: Multi-Track Resolution Pipeline")
    print("=" * 60)

    # Track 1: Wildcard Propagation
    from voynich.phases.p76_wildcard_prop import run_wildcard_prop
    print("\n" + "-" * 60)
    run_wildcard_prop()

    # Track 2: Skeleton Parse
    from voynich.phases.p76_skeleton import run_skeleton_parse
    print("\n" + "-" * 60)
    run_skeleton_parse()

    # Track 3: Frequency Gap
    from voynich.phases.p76_freq_gap import run_freq_gap
    print("\n" + "-" * 60)
    run_freq_gap()

    # Track 4: Conditional Gap-Fill (depends on Track 1 results)
    from voynich.phases.p76_cond_gapfill import run_cond_gapfill
    print("\n" + "-" * 60)
    run_cond_gapfill()

    # Integration
    print("\n" + "-" * 60)
    result = run_phase76_verdict()

    total_time = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"Phase 76 complete in {total_time:.0f}s")
    print(f"{'=' * 60}")

    return result
