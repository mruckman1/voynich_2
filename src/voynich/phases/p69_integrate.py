"""
Phase 69: The Clean Core — Integration
=========================================
Collects results from all tracks, evaluates the mandatory validation gate,
and produces a final verdict.

Dependency chain:
    results/p69_clean_corpus.json          (Step 0)
    results/p69_clean_validation.json      (Track 0)
    results/p69_clean_segmentation.json    (Track 1)
    results/p69_clean_llm.json             (Track 2)
    results/p69_clean_distrib.json         (Track 3)
    results/p69_t1_network.json            (Track 4)
    results/p69_t1_reading.json            (Track 5)
    results/p69_t1_ci.json                 (Track 6)
        -> results/p69_integrate.json
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
class Phase69IntegrateResult:
    phase: str = "69"
    step: str = "69.8"
    experiment: str = "phase69_integrate"
    # Validation gate
    validation_verdict: str = "UNKNOWN"
    validation_gates: int = 0
    # Clean corpus stats
    clean_fraction: float = 0.0
    clean_dict_hit: float = 0.0
    n_t1_identifications: int = 0
    # Track summaries
    tracks_run: List[str] = field(default_factory=list)
    track_summaries: List[Dict[str, Any]] = field(default_factory=list)
    # Gate totals
    total_gates_available: int = 0
    total_gates_passed: int = 0
    # Key findings
    key_findings: List[str] = field(default_factory=list)
    # Verdict
    verdict: str = "UNKNOWN"
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Entry point: verdict
# ---------------------------------------------------------------------------

def run_phase69_verdict():
    """Integration: combine all track results and evaluate."""
    t0 = time.time()
    rd = str(_results_dir())

    print("Phase 69.8 — Integration Verdict")
    print("=" * 35)

    # --- Load all results ---
    corpus_data = _safe_load(os.path.join(rd, 'p69_clean_corpus.json'))
    val_data = _safe_load(os.path.join(rd, 'p69_clean_validation.json'))
    seg_data = _safe_load(os.path.join(rd, 'p69_clean_segmentation.json'))
    llm_data = _safe_load(os.path.join(rd, 'p69_clean_llm.json'))
    dist_data = _safe_load(os.path.join(rd, 'p69_clean_distrib.json'))
    net_data = _safe_load(os.path.join(rd, 'p69_t1_network.json'))
    read_data = _safe_load(os.path.join(rd, 'p69_t1_reading.json'))
    ci_data = _safe_load(os.path.join(rd, 'p69_t1_ci.json'))

    # --- Validation gate ---
    val_verdict = val_data.get('verdict', 'UNKNOWN')
    val_gates = val_data.get('gates_passed', 0)
    print(f"  Validation: {val_verdict} ({val_gates}/3)")

    # --- Corpus stats ---
    clean_frac = corpus_data.get('clean_fraction', 0.0)
    clean_dict_hit = corpus_data.get('clean_dict_hit', 0.0)
    n_t1 = corpus_data.get('n_t1_identifications', 0)
    print(f"  Clean fraction: {clean_frac:.1%}")
    print(f"  Clean dict hit: {clean_dict_hit:.1%}")
    print(f"  T1 words: {n_t1}")

    # --- Collect track summaries ---
    tracks: List[Dict[str, Any]] = []
    total_available = 0
    total_passed = 0

    track_configs = [
        ('Track 0: Validation', val_data, 3),
        ('Track 1: Segmentation', seg_data, 4),
        ('Track 2: LLM Reading', llm_data, 5),
        ('Track 3: Distributional', dist_data, 3),
        ('Track 4: T1 Network', net_data, 3),
        ('Track 5: T1 Reading', read_data, 4),
        ('Track 6: T1 CI Cross-Ref', ci_data, 3),
    ]

    tracks_run = []
    for name, data, n_gates in track_configs:
        if not data:
            tracks.append({'name': name, 'status': 'NOT_RUN', 'gates': '—'})
            continue

        gates_passed = data.get('gates_passed', 0)
        gate_passed = data.get('gate_passed', False)
        verdict = data.get('verdict', data.get('validation_status', ''))

        total_available += n_gates
        total_passed += gates_passed
        tracks_run.append(name)

        tracks.append({
            'name': name,
            'status': 'PASS' if gate_passed else ('SKIP' if 'SKIPPED' in str(verdict) else 'FAIL'),
            'gates': f"{gates_passed}/{n_gates}",
            'verdict': verdict,
        })

    print(f"\n  Track Results:")
    for t in tracks:
        print(f"    {t['name']}: {t['status']} ({t['gates']})")

    # --- Key findings ---
    findings: List[str] = []

    if val_verdict == 'VALIDATED':
        findings.append("Clean subset VALIDATED: all 3 permutation tests significant")
    elif val_verdict == 'PARTIAL':
        findings.append(f"Clean subset PARTIAL: {val_gates}/3 permutation tests significant")

    if seg_data and seg_data.get('gate_passed'):
        lm_hit = seg_data.get('lm_mean_dict_hit', 0)
        findings.append(f"Segmentation IMPROVED: LM dict-hit {lm_hit:.1%} on clean subset")

    if dist_data and dist_data.get('n_anchors', 0) >= 100:
        conv = dist_data.get('convergence_rate', 0)
        findings.append(f"Distributional mapping: {dist_data.get('n_anchors')} anchors, "
                       f"{conv:.1%} convergence")

    if net_data and net_data.get('n_paradigms', 0) >= 10:
        findings.append(f"Morphological paradigms: {net_data.get('n_paradigms')} families found")

    if read_data and read_data.get('n_pharma_readings', 0) >= 1:
        findings.append(f"Pharmaceutical reading: {read_data.get('n_pharma_readings')} "
                       f"passages with pharma content")

    if ci_data and ci_data.get('gate_ci3', False):
        findings.append(f"CI cross-reference: significant (p={ci_data.get('perm_p', 1):.4f})")

    # --- Overall verdict ---
    tracks_passing = sum(1 for t in tracks if t['status'] == 'PASS')

    if val_verdict == 'VALIDATED' and tracks_passing >= 4:
        verdict = 'CLEAN_CORE_VALIDATED'
    elif val_verdict in ('VALIDATED', 'PARTIAL') and tracks_passing >= 2:
        verdict = 'CLEAN_CORE_PARTIAL'
    else:
        verdict = 'CLEAN_CORE_FAILED'

    print(f"\n  Tracks passing: {tracks_passing}/7")
    print(f"  Total gates: {total_passed}/{total_available}")
    print(f"  Verdict: {verdict}")

    if findings:
        print(f"\n  Key Findings:")
        for f in findings:
            print(f"    • {f}")

    # --- Build result ---
    result = Phase69IntegrateResult(
        validation_verdict=val_verdict,
        validation_gates=val_gates,
        clean_fraction=clean_frac,
        clean_dict_hit=clean_dict_hit,
        n_t1_identifications=n_t1,
        tracks_run=tracks_run,
        track_summaries=tracks,
        total_gates_available=total_available,
        total_gates_passed=total_passed,
        key_findings=findings,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 1),
    )

    path = _save_json(rd, 'p69_integrate.json', result)
    print(f"\n  Saved: {path}")
    print(f"  Time: {result.runtime_seconds:.1f}s")


# ---------------------------------------------------------------------------
# Entry point: full pipeline
# ---------------------------------------------------------------------------

def run_phase69():
    """Run full Phase 69 pipeline: data prep + validation gate + all tracks."""
    t0 = time.time()

    print("Phase 69 — The Clean Core (Full Pipeline)")
    print("=" * 44)
    print()

    # Step 0: Build clean corpus
    from voynich.phases.p69_clean_corpus import run_build_clean
    run_build_clean()
    print()

    # Track 0: Mandatory validation gate
    from voynich.phases.p69_clean_validation import run_validate_clean
    run_validate_clean()
    print()

    # Check validation gate
    rd = str(_results_dir())
    val_data = _safe_load(os.path.join(rd, 'p69_clean_validation.json'))
    val_verdict = val_data.get('verdict', 'FAILED')

    if val_verdict == 'FAILED':
        print("  VALIDATION FAILED — skipping Tracks 1-3 (clean-dependent)")
        print("  Running T1 tracks only...")
        print()

    # Tracks 1-3 (require validation >= PARTIAL)
    if val_verdict != 'FAILED':
        from voynich.phases.p69_clean_segmentation import run_clean_segment
        run_clean_segment()
        print()

        from voynich.phases.p69_clean_llm import run_clean_llm_read
        run_clean_llm_read()
        print()

        from voynich.phases.p69_clean_distrib import run_clean_distrib
        run_clean_distrib()
        print()

    # Tracks 4-6 (always run)
    from voynich.phases.p69_t1_network import run_t1_network
    run_t1_network()
    print()

    from voynich.phases.p69_t1_reading import run_t1_read
    run_t1_read()
    print()

    from voynich.phases.p69_t1_ci import run_t1_ci_crossref
    run_t1_ci_crossref()
    print()

    # Integration verdict
    run_phase69_verdict()

    elapsed = time.time() - t0
    print(f"\nPhase 69 total time: {elapsed:.1f}s")
