"""
Phase 66: Integration and Verdict
===================================
Loads all 12 track results, determines overall verdict.

Dependency chain:
    results/p66_llm_reading.json      (Track 1)
    results/p66_reverse_sim.json      (Track 2)
    results/p66_f116v_crib.json       (Track 3)
    results/p66_illus_align.json      (Track 4)
    results/p66_parallel_align.json   (Track 5)
    results/p66_fontana.json          (Track 6)
    results/p66_lang_a.json           (Track 7)
    results/p66_hand4.json            (Track 8)
    results/p66_collocations.json     (Track 9)
    results/p66_ngram_freq.json       (Track 10)
    results/p66_metrical.json         (Track 11)
    results/p66_astro_deep.json       (Track 12)
        -> results/p66_integrate.json
"""
from __future__ import annotations

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
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Phase66IntegrateResult:
    phase: str = "66"
    experiment: str = "phase66_integrate"
    n_tracks_run: int = 0
    n_tracks_passed: int = 0
    tier1_summary: Dict = field(default_factory=dict)
    tier2_summary: Dict = field(default_factory=dict)
    tier3_summary: Dict = field(default_factory=dict)
    tier4_summary: Dict = field(default_factory=dict)
    track_results: List[Dict] = field(default_factory=list)
    reading_level: str = ""
    content_level: str = ""
    overall_verdict: str = ""
    summary: str = ""
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Track metadata
# ---------------------------------------------------------------------------

TRACKS = [
    (1, 'p66_llm_reading.json', 'LLM Pharmaceutical Reading', 1),
    (2, 'p66_reverse_sim.json', 'Reverse Simulation (Viterbi)', 1),
    (3, 'p66_f116v_crib.json', 'f116v Crib Exploitation', 1),
    (4, 'p66_illus_align.json', 'Illustration-Text Alignment', 2),
    (5, 'p66_parallel_align.json', 'CI Parallel Alignment', 2),
    (6, 'p66_fontana.json', 'Fontana Structural Comparison', 2),
    (7, 'p66_lang_a.json', 'Language A Focus', 3),
    (8, 'p66_hand4.json', 'Hand 4 Focus', 3),
    (9, 'p66_collocations.json', 'Collocational Analysis', 3),
    (10, 'p66_ngram_freq.json', 'N-gram Frequency Ranking', 3),
    (11, 'p66_metrical.json', 'Metrical Analysis', 4),
    (12, 'p66_astro_deep.json', 'Astronomical Deep Dive', 4),
]


# ---------------------------------------------------------------------------
# Verdict logic
# ---------------------------------------------------------------------------

def _determine_verdict(track_data: Dict[int, Dict]) -> Dict[str, str]:
    """Determine Phase 66 verdict from track results.

    READING_ACHIEVED:
      Track 1 calibration passes AND ≥ 3 valid readings AND
      mean shuffled ratio ≥ 2× AND ≥ 1 valid translation

    CONTENT_CONFIRMED:
      (Track 4 or Track 5 passes) AND (Track 9 or Track 10 passes)

    STRUCTURAL_INSIGHT:
      Tier 2-4 tracks provide structural findings

    NEGATIVE:
      Calibration fails AND no content/structural findings
    """
    # Track 1 assessment
    t1 = track_data.get(1, {})
    cal_passed = t1.get('calibration_passed', False)
    t1_gates = t1.get('gates_passed', 0)

    if cal_passed and t1_gates >= 6:
        reading_level = 'READING_ACHIEVED'
    elif cal_passed and t1_gates >= 4:
        reading_level = 'PARTIAL_READING'
    elif cal_passed:
        reading_level = 'CONTROLS_DOMINATE'
    else:
        reading_level = 'NO_READING'

    # Content assessment (Tier 2 + Tier 3)
    t4_passed = track_data.get(4, {}).get('gate_passed', False)
    t5_passed = track_data.get(5, {}).get('gate_passed', False)
    t9_passed = track_data.get(9, {}).get('gate_passed', False)
    t10_passed = track_data.get(10, {}).get('gate_passed', False)

    context_evidence = t4_passed or t5_passed
    corpus_evidence = t9_passed or t10_passed

    if context_evidence and corpus_evidence:
        content_level = 'CONTENT_CONFIRMED'
    elif context_evidence or corpus_evidence:
        content_level = 'PARTIAL_CONTENT'
    else:
        content_level = 'CONTENT_UNCONFIRMED'

    # Overall
    if reading_level == 'READING_ACHIEVED':
        overall = 'READING_ACHIEVED'
    elif content_level == 'CONTENT_CONFIRMED':
        overall = 'CONTENT_CONFIRMED'
    elif reading_level in ('PARTIAL_READING', 'CONTROLS_DOMINATE'):
        overall = 'STRUCTURAL_INSIGHT'
    else:
        # Count total passed gates across all tracks
        total_passed = sum(
            1 for td in track_data.values()
            if td.get('gate_passed', False)
        )
        if total_passed >= 4:
            overall = 'STRUCTURAL_INSIGHT'
        else:
            overall = 'NEGATIVE'

    return {
        'reading_level': reading_level,
        'content_level': content_level,
        'overall': overall,
    }


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------

def run_phase66_verdict() -> None:
    """Phase 66: Integration and Verdict."""
    t0 = time.time()
    rd = str(_results_dir())

    print("=" * 70)
    print("Phase 66: Integration and Verdict")
    print("=" * 70)

    # Load all track results
    track_data: Dict[int, Dict] = {}
    track_summaries = []
    n_run = 0
    n_passed = 0

    for track_num, filename, title, tier in TRACKS:
        data = _safe_load(os.path.join(rd, filename))
        exists = bool(data)
        passed = data.get('gate_passed', False) if data else False
        verdict = data.get('verdict', 'NOT_RUN') if data else 'NOT_RUN'
        gates = data.get('gates_passed', 0) if data else 0

        if exists:
            track_data[track_num] = data
            n_run += 1
            if passed:
                n_passed += 1

        track_summaries.append({
            'track': track_num,
            'title': title,
            'tier': tier,
            'run': exists,
            'gate_passed': passed,
            'gates_passed': gates,
            'verdict': verdict,
        })

        status = ('PASS' if passed else 'FAIL') if exists else 'NOT_RUN'
        print(f"  Track {track_num:2d} (T{tier}): {title:40s} "
              f"[{status}] {verdict}")

    # Tier summaries
    tier_summaries = {}
    for tier_num in [1, 2, 3, 4]:
        tier_tracks = [s for s in track_summaries if s['tier'] == tier_num]
        tier_run = sum(1 for s in tier_tracks if s['run'])
        tier_passed = sum(1 for s in tier_tracks if s['gate_passed'])
        tier_summaries[tier_num] = {
            'n_tracks': len(tier_tracks),
            'n_run': tier_run,
            'n_passed': tier_passed,
            'pass_rate': (tier_passed / tier_run
                         if tier_run > 0 else 0.0),
        }

    # Verdict
    verdict_info = _determine_verdict(track_data)

    # Summary string
    summary_parts = [
        f"Tracks run: {n_run}/12",
        f"Tracks passed: {n_passed}/{n_run}",
        f"Reading: {verdict_info['reading_level']}",
        f"Content: {verdict_info['content_level']}",
    ]
    summary = ' | '.join(summary_parts)

    print(f"\n{'=' * 70}")
    print("VERDICT")
    print(f"{'=' * 70}")
    for tier_num in [1, 2, 3, 4]:
        ts = tier_summaries[tier_num]
        print(f"  Tier {tier_num}: {ts['n_passed']}/{ts['n_run']} passed")
    print(f"\n  Reading level:  {verdict_info['reading_level']}")
    print(f"  Content level:  {verdict_info['content_level']}")
    print(f"  Overall:        {verdict_info['overall']}")

    # Save
    result = Phase66IntegrateResult(
        n_tracks_run=n_run,
        n_tracks_passed=n_passed,
        tier1_summary=tier_summaries.get(1, {}),
        tier2_summary=tier_summaries.get(2, {}),
        tier3_summary=tier_summaries.get(3, {}),
        tier4_summary=tier_summaries.get(4, {}),
        track_results=track_summaries,
        reading_level=verdict_info['reading_level'],
        content_level=verdict_info['content_level'],
        overall_verdict=verdict_info['overall'],
        summary=summary,
        runtime_seconds=round(time.time() - t0, 2),
    )

    _save_json(rd, 'p66_integrate.json', asdict(result))
    print(f"\n  Saved to results/p66_integrate.json")
    print(f"  Runtime: {result.runtime_seconds}s")


def run_phase66() -> None:
    """Run full Phase 66 pipeline."""
    print("=" * 70)
    print("Phase 66: Multi-Vector Attack with Hallucination Controls")
    print("=" * 70)
    print("\nRunning all 12 tracks sequentially...\n")

    runners = []

    # Tier 1
    try:
        from voynich.phases.p66_llm_reading import run_llm_reading
        runners.append(('Track 1: LLM Reading', run_llm_reading))
    except ImportError:
        pass

    try:
        from voynich.phases.p66_reverse_sim import run_reverse_sim
        runners.append(('Track 2: Reverse Sim', run_reverse_sim))
    except ImportError:
        pass

    try:
        from voynich.phases.p66_f116v_crib import run_f116v_crib
        runners.append(('Track 3: f116v Crib', run_f116v_crib))
    except ImportError:
        pass

    # Tier 2
    try:
        from voynich.phases.p66_illus_align import run_illus_align
        runners.append(('Track 4: Illus Align', run_illus_align))
    except ImportError:
        pass

    try:
        from voynich.phases.p66_parallel_align import run_parallel_align
        runners.append(('Track 5: CI Align', run_parallel_align))
    except ImportError:
        pass

    try:
        from voynich.phases.p66_fontana import run_fontana_struct
        runners.append(('Track 6: Fontana', run_fontana_struct))
    except ImportError:
        pass

    # Tier 3
    try:
        from voynich.phases.p66_lang_a import run_lang_a
        runners.append(('Track 7: Lang A', run_lang_a))
    except ImportError:
        pass

    try:
        from voynich.phases.p66_hand4 import run_hand4
        runners.append(('Track 8: Hand 4', run_hand4))
    except ImportError:
        pass

    try:
        from voynich.phases.p66_collocations import run_collocations
        runners.append(('Track 9: Collocations', run_collocations))
    except ImportError:
        pass

    try:
        from voynich.phases.p66_ngram_freq import run_ngram_freq
        runners.append(('Track 10: N-gram Freq', run_ngram_freq))
    except ImportError:
        pass

    # Tier 4
    try:
        from voynich.phases.p66_metrical import run_metrical
        runners.append(('Track 11: Metrical', run_metrical))
    except ImportError:
        pass

    try:
        from voynich.phases.p66_astro_deep import run_astro_deep
        runners.append(('Track 12: Astro Deep', run_astro_deep))
    except ImportError:
        pass

    for label, runner in runners:
        print(f"\n{'─' * 70}")
        print(f"Starting {label}")
        print(f"{'─' * 70}")
        try:
            runner()
        except Exception as e:
            print(f"  ERROR in {label}: {e}")

    # Integration
    print(f"\n{'─' * 70}")
    print("Running Integration...")
    print(f"{'─' * 70}")
    run_phase66_verdict()
