"""
Phase 71: Inflectional Reverse Engineering — Integration
========================================================
Collects results from all 3 tracks and determines the overall verdict.

Dependency chain:
    results/phase71_inflectional_catalog.json   (Track 1)
    results/phase71_root_identification.json    (Track 2)
    results/phase71_grammatical_reading.json    (Track 3)
        -> results/phase71_integrate.json
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
class Phase71IntegrateResult:
    phase: str = "71"
    step: str = "71.4"
    experiment: str = "phase71_integrate"
    # Track summaries
    tracks_run: List[str] = field(default_factory=list)
    track_summaries: List[Dict[str, Any]] = field(default_factory=list)
    # Gate totals
    total_gates_available: int = 0
    total_gates_passed: int = 0
    # Per-track gates
    track1_gates: int = 0
    track2_gates: int = 0
    track3_gates: int = 0
    # Key findings
    key_findings: List[str] = field(default_factory=list)
    # Verdict
    verdict: str = "UNKNOWN"
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Verdict logic
# ---------------------------------------------------------------------------

def _determine_verdict(t1: Dict, t2: Dict, t3: Dict) -> str:
    """Determine overall Phase 71 verdict.

    GRAMMATICAL_READING:
        Track 3 has interpretable passage AND Track 1 null test passes

    GRAMMATICAL_STRUCTURE:
        Track 1 >= 4 gates AND Track 2 >= 3 gates
        AND Track 3 template selectivity > 1.3x

    INFLECTIONAL_CONFIRMED:
        Track 1 >= 3 gates (coda-grammar mapping validated)

    MARGINAL:
        otherwise
    """
    t1_gates = t1.get('gates_passed', 0)
    t2_gates = t2.get('gates_passed', 0)
    t3_gates = t3.get('gates_passed', 0)

    t1_null_sig = t1.get('null_test', {}).get('significant', False)
    t3_has_interp = t3.get('n_interpretable', 0) >= 1
    t3_template_sel = t3.get('null_controls', {}).get('template_selectivity', 0.0)

    if t3_has_interp and t1_null_sig and t3_gates >= 4:
        return 'GRAMMATICAL_READING'

    if t1_gates >= 4 and t2_gates >= 3 and t3_template_sel > 1.3:
        return 'GRAMMATICAL_STRUCTURE'

    if t1_gates >= 3:
        return 'INFLECTIONAL_CONFIRMED'

    return 'MARGINAL'


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def run_phase71_verdict():
    """Collect all track results and determine the Phase 71 verdict."""
    t0 = time.time()
    rd = str(_results_dir())

    print("Phase 71.4 — Integration & Verdict")
    print("=" * 36)

    # Load track results
    t1 = _safe_load(os.path.join(rd, 'phase71_inflectional_catalog.json'))
    t2 = _safe_load(os.path.join(rd, 'phase71_root_identification.json'))
    t3 = _safe_load(os.path.join(rd, 'phase71_grammatical_reading.json'))

    tracks_run = []
    track_summaries = []
    key_findings = []

    # Track 1: Inflectional catalog
    t1_gates = t1.get('gates_passed', 0)
    t1_verdict = t1.get('verdict', 'NOT_RUN')
    if t1:
        tracks_run.append('inflectional_catalog')
        track_summaries.append({
            'track': 'Track 1: Inflectional Catalog',
            'verdict': t1_verdict,
            'gates': f"{t1_gates}/5",
            'n_verbal': t1.get('broad_distribution', {}).get('VERBAL', 0),
            'n_nominal': t1.get('broad_distribution', {}).get('NOMINAL', 0),
            'null_p': t1.get('null_test', {}).get('p_value', 1.0),
            'section_p': t1.get('section_chi2_p', 1.0),
        })
        broad = t1.get('broad_distribution', {})
        verbal_pct = broad.get('VERBAL', 0)
        nominal_pct = broad.get('NOMINAL', 0)
        key_findings.append(
            f"Inflectional: VERBAL={verbal_pct:.1%}, NOMINAL={nominal_pct:.1%}, "
            f"null p={t1.get('null_test', {}).get('p_value', 1.0):.4f}, "
            f"xval={t1.get('cross_validation_agreement', 0):.1%}")
        print(f"  Track 1: {t1_verdict} ({t1_gates}/5 gates)")
    else:
        print("  Track 1: NOT RUN")

    # Track 2: Root identification
    t2_gates = t2.get('gates_passed', 0)
    t2_verdict = t2.get('verdict', 'NOT_RUN')
    if t2:
        tracks_run.append('root_identification')
        track_summaries.append({
            'track': 'Track 2: Root Identification',
            'verdict': t2_verdict,
            'gates': f"{t2_gates}/5",
            'n_paradigms': t2.get('n_paradigms', 0),
            'n_3plus': t2.get('n_paradigms_3plus', 0),
            'identified_fraction': t2.get('identified_fraction', 0.0),
            'paradigm_coverage': t2.get('paradigm_coverage', 0.0),
        })
        key_findings.append(
            f"Roots: {t2.get('n_paradigms_3plus', 0)} paradigms (3+), "
            f"{t2.get('identified_fraction', 0):.1%} identified, "
            f"{t2.get('paradigm_coverage', 0):.1%} coverage")
        print(f"  Track 2: {t2_verdict} ({t2_gates}/5 gates)")
    else:
        print("  Track 2: NOT RUN")

    # Track 3: Grammatical reading
    t3_gates = t3.get('gates_passed', 0)
    t3_verdict = t3.get('verdict', 'NOT_RUN')
    if t3:
        tracks_run.append('grammatical_reading')
        track_summaries.append({
            'track': 'Track 3: Grammatical Reading',
            'verdict': t3_verdict,
            'gates': f"{t3_gates}/6",
            'mean_gram': t3.get('mean_grammatical_fraction', 0.0),
            'mean_lex': t3.get('mean_identified_fraction', 0.0),
            'n_template': t3.get('n_template_matches', 0),
            'n_interpretable': t3.get('n_interpretable', 0),
            'template_selectivity': t3.get('null_controls', {}).get(
                'template_selectivity', 0.0),
        })
        key_findings.append(
            f"Reading: gram={t3.get('mean_grammatical_fraction', 0):.1%}, "
            f"lex={t3.get('mean_identified_fraction', 0):.1%}, "
            f"{t3.get('n_interpretable', 0)} interpretable")
        print(f"  Track 3: {t3_verdict} ({t3_gates}/6 gates)")
    else:
        print("  Track 3: NOT RUN")

    # Total gates
    total_available = 5 + 5 + 6  # 16
    total_passed = t1_gates + t2_gates + t3_gates

    # Verdict
    verdict = _determine_verdict(t1, t2, t3)

    print(f"\n  Total gates: {total_passed}/{total_available}")
    print(f"\n  Key findings:")
    for finding in key_findings:
        print(f"    • {finding}")

    print(f"\n  ═══ VERDICT: {verdict} ═══")

    # Build result
    result = Phase71IntegrateResult(
        tracks_run=tracks_run,
        track_summaries=track_summaries,
        total_gates_available=total_available,
        total_gates_passed=total_passed,
        track1_gates=t1_gates,
        track2_gates=t2_gates,
        track3_gates=t3_gates,
        key_findings=key_findings,
        verdict=verdict,
        runtime_seconds=time.time() - t0,
    )

    out = _save_json(rd, 'phase71_integrate.json', asdict(result))
    print(f"\n  Saved: {out}")
    print(f"  Runtime: {result.runtime_seconds:.1f}s")

    return result


def run_phase71():
    """Full Phase 71 pipeline: all 3 tracks + integration."""
    t0 = time.time()
    print("╔══════════════════════════════════════════════════════╗")
    print("║  Phase 71: Inflectional Reverse Engineering         ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()

    # Track 1: Inflectional catalog
    from voynich.phases.inflectional_catalog import run_inflect_catalog
    print("─" * 50)
    run_inflect_catalog()
    print()

    # Track 2: Root identification
    from voynich.phases.root_identification import run_root_id
    print("─" * 50)
    run_root_id()
    print()

    # Track 3: Grammatical reading
    from voynich.phases.grammatical_reading import run_gram_read
    print("─" * 50)
    run_gram_read()
    print()

    # Integration
    print("─" * 50)
    result = run_phase71_verdict()

    total_time = time.time() - t0
    print(f"\n  Total Phase 71 runtime: {total_time:.1f}s")

    return result
