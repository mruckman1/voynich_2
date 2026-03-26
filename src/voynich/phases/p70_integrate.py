"""
Phase 70: Token-as-Word Exploitation — Integration
====================================================
Collects results from all 4 tracks and determines the overall verdict.

Dependency chain:
    results/phase70_pharma_dict.json     (Track 1)
    results/phase70_paradigms.json       (Track 2)
    results/phase70_phrases.json         (Track 3)
    results/phase70_readings.json        (Track 4)
        -> results/phase70_integrate.json
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
class Phase70IntegrateResult:
    phase: str = "70"
    step: str = "70.5"
    experiment: str = "phase70_integrate"
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
    track4_gates: int = 0
    # Key findings
    key_findings: List[str] = field(default_factory=list)
    # Verdict
    verdict: str = "UNKNOWN"
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Verdict logic
# ---------------------------------------------------------------------------

def _determine_verdict(t1: Dict, t2: Dict, t3: Dict, t4: Dict) -> str:
    """Determine overall Phase 70 verdict.

    PHARMACEUTICAL_READING:
        Track 4 gates >= 4 AND has coherent interpretation
        AND selectivity vs random > 1.5×

    STRUCTURED_VOCABULARY:
        Track 1 gates >= 3 AND Track 2 gates >= 3
        AND Track 3 gates >= 2

    VOCABULARY_EXPANDED:
        Track 1 gates >= 3

    MARGINAL:
        otherwise
    """
    t4_gates = t4.get('gates_passed', 0)
    t4_has_interp = t4.get('n_interpretations', 0) >= 1
    t4_selectivity = t4.get('selectivity_vs_random', 0.0)

    t1_gates = t1.get('gates_passed', 0)
    t2_gates = t2.get('gates_passed', 0)
    t3_gates = t3.get('gates_passed', 0)

    if t4_gates >= 4 and t4_has_interp and t4_selectivity > 1.5:
        return 'PHARMACEUTICAL_READING'

    if t1_gates >= 3 and t2_gates >= 3 and t3_gates >= 2:
        return 'STRUCTURED_VOCABULARY'

    if t1_gates >= 3:
        return 'VOCABULARY_EXPANDED'

    return 'MARGINAL'


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def run_phase70_verdict():
    """Collect all track results and determine the Phase 70 verdict."""
    t0 = time.time()
    rd = str(_results_dir())

    print("Phase 70.5 — Integration & Verdict")
    print("=" * 36)

    # Load track results
    t1 = _safe_load(os.path.join(rd, 'phase70_pharma_dict.json'))
    t2 = _safe_load(os.path.join(rd, 'phase70_paradigms.json'))
    t3 = _safe_load(os.path.join(rd, 'phase70_phrases.json'))
    t4 = _safe_load(os.path.join(rd, 'phase70_readings.json'))

    tracks_run = []
    track_summaries = []
    key_findings = []

    # Track 1: Dictionary
    t1_gates = t1.get('gates_passed', 0)
    t1_verdict = t1.get('verdict', 'NOT_RUN')
    if t1:
        tracks_run.append('pharma_dictionary')
        track_summaries.append({
            'track': 'Track 1: Pharmaceutical Dictionary',
            'verdict': t1_verdict,
            'gates': f"{t1_gates}/5",
            'dict_size': t1.get('n_combined', 0),
            'clean_dict_hit': t1.get('clean_dict_hit_new', 0.0),
            'selectivity': t1.get('selectivity', 0.0),
            'delta': t1.get('delta', 0.0),
        })
        if t1.get('delta', 0) > 0:
            key_findings.append(
                f"Dictionary expansion: {t1.get('clean_dict_hit_old', 0):.1%} → "
                f"{t1.get('clean_dict_hit_new', 0):.1%} "
                f"(+{t1.get('delta', 0):.1%}, {t1.get('selectivity', 0):.2f}× sel)")
        print(f"  Track 1: {t1_verdict} ({t1_gates}/5 gates)")
    else:
        print("  Track 1: NOT RUN")

    # Track 2: Paradigms
    t2_gates = t2.get('gates_passed', 0)
    t2_verdict = t2.get('verdict', 'NOT_RUN')
    if t2:
        tracks_run.append('paradigm_mapping')
        track_summaries.append({
            'track': 'Track 2: Morphological Paradigms',
            'verdict': t2_verdict,
            'gates': f"{t2_gates}/4",
            'n_paradigms': t2.get('n_paradigms', 0),
            'n_3plus': t2.get('n_paradigms_with_3plus', 0),
            'n_consistent_codas': t2.get('n_consistent_codas', 0),
        })
        if t2.get('n_paradigms', 0) > 0:
            key_findings.append(
                f"Paradigms: {t2.get('n_paradigms', 0)} found, "
                f"{t2.get('n_paradigms_with_3plus', 0)} with 3+ forms, "
                f"{t2.get('n_consistent_codas', 0)} consistent codas")
        print(f"  Track 2: {t2_verdict} ({t2_gates}/4 gates)")
    else:
        print("  Track 2: NOT RUN")

    # Track 3: Phrases
    t3_gates = t3.get('gates_passed', 0)
    t3_verdict = t3.get('verdict', 'NOT_RUN')
    if t3:
        tracks_run.append('phrase_assembly')
        track_summaries.append({
            'track': 'Track 3: Phrase Assembly',
            'verdict': t3_verdict,
            'gates': f"{t3_gates}/4",
            'n_pairs': t3.get('n_ordered_pairs', 0),
            'glossed_fraction': t3.get('glossed_fraction', 0.0),
            'n_verb_object': t3.get('n_verb_object', 0),
            'n_trigrams': t3.get('n_unique_trigrams', 0),
        })
        if t3.get('glossed_fraction', 0) > 0:
            key_findings.append(
                f"Phrases: {t3.get('glossed_fraction', 0):.0%} glossed, "
                f"{t3.get('n_verb_object', 0)} VERB+OBJ, "
                f"{t3.get('n_fully_glossed_trigrams', 0)} trigrams")
        print(f"  Track 3: {t3_verdict} ({t3_gates}/4 gates)")
    else:
        print("  Track 3: NOT RUN")

    # Track 4: Readings
    t4_gates = t4.get('gates_passed', 0)
    t4_verdict = t4.get('verdict', 'NOT_RUN')
    if t4:
        tracks_run.append('annotated_readings')
        track_summaries.append({
            'track': 'Track 4: Annotated Readings',
            'verdict': t4_verdict,
            'gates': f"{t4_gates}/6",
            'mean_identified': t4.get('mean_identified_fraction', 0.0),
            'n_ci_matches': t4.get('n_ci_matches', 0),
            'n_interpretations': t4.get('n_interpretations', 0),
            'selectivity': t4.get('selectivity_vs_random', 0.0),
        })
        if t4.get('mean_identified_fraction', 0) > 0:
            key_findings.append(
                f"Readings: {t4.get('mean_identified_fraction', 0):.0%} mean identified, "
                f"{t4.get('n_ci_matches', 0)} CI matches, "
                f"{t4.get('selectivity_vs_random', 0):.2f}× selectivity")
        print(f"  Track 4: {t4_verdict} ({t4_gates}/6 gates)")
    else:
        print("  Track 4: NOT RUN")

    # Total gates
    total_available = 5 + 4 + 4 + 6  # 19
    total_passed = t1_gates + t2_gates + t3_gates + t4_gates

    # Verdict
    verdict = _determine_verdict(t1, t2, t3, t4)

    print(f"\n  Total gates: {total_passed}/{total_available}")
    print(f"\n  Key findings:")
    for finding in key_findings:
        print(f"    • {finding}")

    print(f"\n  ═══ VERDICT: {verdict} ═══")

    # Build result
    result = Phase70IntegrateResult(
        tracks_run=tracks_run,
        track_summaries=track_summaries,
        total_gates_available=total_available,
        total_gates_passed=total_passed,
        track1_gates=t1_gates,
        track2_gates=t2_gates,
        track3_gates=t3_gates,
        track4_gates=t4_gates,
        key_findings=key_findings,
        verdict=verdict,
        runtime_seconds=time.time() - t0,
    )

    out = _save_json(rd, 'phase70_integrate.json', asdict(result))
    print(f"\n  Saved: {out}")
    print(f"  Runtime: {result.runtime_seconds:.1f}s")

    return result


def run_phase70():
    """Full Phase 70 pipeline: all 4 tracks + integration."""
    t0 = time.time()
    print("╔══════════════════════════════════════════════════╗")
    print("║  Phase 70: Token-as-Word Exploitation           ║")
    print("╚══════════════════════════════════════════════════╝")
    print()

    # Track 1: Dictionary
    from voynich.phases.p70_pharma_dict import run_pharma_dict
    print("─" * 50)
    run_pharma_dict()
    print()

    # Track 2: Paradigms
    from voynich.phases.p70_paradigm_map import run_paradigm_map
    print("─" * 50)
    run_paradigm_map()
    print()

    # Track 3: Phrases
    from voynich.phases.p70_phrase_assembly import run_phrase_assemble
    print("─" * 50)
    run_phrase_assemble()
    print()

    # Track 4: Readings
    from voynich.phases.p70_annotated_read import run_annotate_read
    print("─" * 50)
    run_annotate_read()
    print()

    # Integration
    print("─" * 50)
    result = run_phase70_verdict()

    total_time = time.time() - t0
    print(f"\n  Total Phase 70 runtime: {total_time:.1f}s")

    return result
