"""
Phase 53 Integration: Paradigm-Constrained Free Triple Resolution
==================================================================
Combine results from Track A (paradigm constraints), Track B (triple
resolution), and Track C (resolved decode) to produce overall verdict.

Dependency chain:
    paradigm_constraints.json  (Track A)
    triple_resolution.json     (Track B)
    resolved_decode.json       (Track C)
        -> phase53_integrate.json
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List

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
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Phase53Validation:
    name: str
    description: str
    passed: bool
    value: float
    threshold: float


@dataclass
class Phase53IntegrateResult:
    # Track A
    paradigms_analyzed: int
    total_constraints: int
    unique_constraints: int
    triples_with_constraints: int
    accepted_corrections: int
    # Track B
    corrections_applied: int
    dict_hit_pre: float
    dict_hit_post: float
    delta_dict_hit: float
    signal_words_preserved: int
    null_z_score: float
    null_selectivity: float
    track_b_verdict: str
    # Track C
    newly_decoded_tokens: int
    longest_run: int
    n_content_runs: int
    best_content_run_length: int
    best_content_run_text: str
    # Validation
    validations: List[Dict]
    n_passed: int
    n_total: int
    # Verdict
    verdict: str
    gate_passed: bool
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_phase53_integrate() -> None:
    """Phase 53 Integration: verdict from all three tracks."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE 53 INTEGRATION: Paradigm-Constrained Free Triple Resolution")
    print("=" * 70)

    rd = _results_dir()

    # ── Load track results ────────────────────────────────────────────
    print("\n  Loading track results...")

    constraints = _safe_load(os.path.join(rd, 'paradigm_constraints.json'))
    resolution = _safe_load(os.path.join(rd, 'triple_resolution.json'))
    decode = _safe_load(os.path.join(rd, 'resolved_decode.json'))

    # Track A metrics
    paradigms_analyzed = constraints.get('paradigms_analyzed', 0)
    total_constraints = constraints.get('total_constraints', 0)
    unique_constraints = constraints.get('unique_constraints', 0)
    n_triples_with = constraints.get('n_triples_with_constraints', 0)
    n_accepted_a = len(constraints.get('accepted_corrections', []))

    # Find max consensus across triples
    max_consensus = 0.0
    n_consensus_triples = 0
    for summary in constraints.get('per_triple_summary', {}).values():
        cons = summary.get('consensus', 0.0)
        n_obs = summary.get('n_unique_constraints', 0)
        if n_obs >= 3 and cons > 0.5:
            n_consensus_triples += 1
        if cons > max_consensus:
            max_consensus = cons

    # Track B metrics
    corrections_applied = resolution.get('corrections_applied', 0)
    baseline = resolution.get('baseline', {})
    post = resolution.get('post_correction', {})
    null_test = resolution.get('null_test', {})

    dict_hit_pre = baseline.get('dict_hit_10k', 0.0)
    dict_hit_post = post.get('dict_hit_10k', 0.0)
    delta = post.get('delta_dict_hit_10k', 0.0)
    signal_preserved = post.get('signal_word_count',
                               baseline.get('signal_word_count', 0))
    null_z = null_test.get('z_score', 0.0)
    null_sel = null_test.get('selectivity', 0.0)
    track_b_verdict = resolution.get('verdict', 'NO_CORRECTIONS')

    # Track C metrics
    newly_decoded = decode.get('newly_decoded_tokens', 0)
    longest_run = decode.get('longest_run', 0)
    n_content_runs = decode.get('n_content_runs', 0)
    bcr = decode.get('best_content_run', {})
    bcr_length = bcr.get('length', 0)
    bcr_text = bcr.get('text', '')
    bcr_n_content = bcr.get('n_content_words', 0)

    print(f"       Track A: {unique_constraints} constraints, "
          f"{n_triples_with} triples, {n_accepted_a} accepted")
    print(f"       Track B: {corrections_applied} corrections, "
          f"delta={delta:+.4f}, null z={null_z:.2f}")
    print(f"       Track C: {newly_decoded} newly decoded, "
          f"longest run={longest_run}, content runs={n_content_runs}")

    # ── Validation battery ────────────────────────────────────────────
    print("\n  Validation battery...")

    validations = [
        Phase53Validation(
            'V1_constraints_extracted',
            'Unique constraints from paradigms >= 20',
            unique_constraints >= 20,
            float(unique_constraints), 20.0,
        ),
        Phase53Validation(
            'V2_triple_consensus',
            'At least 1 free triple with consensus > 0.5',
            n_consensus_triples >= 1,
            float(n_consensus_triples), 1.0,
        ),
        Phase53Validation(
            'V3_null_selectivity',
            'Real consensus > null consensus (z > 2.0)',
            null_z > 2.0,
            null_z, 2.0,
        ),
        Phase53Validation(
            'V4_signal_words',
            'No signal word regression (preserved == baseline)',
            signal_preserved >= baseline.get('signal_word_count', 0),
            float(signal_preserved),
            float(baseline.get('signal_word_count', 0)),
        ),
        Phase53Validation(
            'V5_dict_hit_improvement',
            'Dict-hit improvement > 0%',
            delta > 0.0,
            delta, 0.0,
        ),
        Phase53Validation(
            'V6_newly_decoded',
            'Dark-to-decoded tokens >= 50',
            newly_decoded >= 50,
            float(newly_decoded), 50.0,
        ),
        Phase53Validation(
            'V7_content_run',
            'Content run with >= 2 content words and >= 5 tokens',
            bcr_length >= 5 and bcr_n_content >= 2,
            float(bcr_length), 5.0,
        ),
    ]

    n_passed = sum(1 for v in validations if v.passed)
    for v in validations:
        status = 'PASS' if v.passed else 'FAIL'
        print(f"       {v.name}: {v.value:.2f} vs {v.threshold:.2f} -> {status}")

    print(f"\n       Passed: {n_passed} / {len(validations)}")

    # ── Verdict ───────────────────────────────────────────────────────
    v_map = {v.name: v for v in validations}

    # V4 is non-negotiable
    if not v_map['V4_signal_words'].passed:
        verdict = 'SIGNAL_WORD_REGRESSION'
    elif (v_map['V2_triple_consensus'].passed and
          v_map['V3_null_selectivity'].passed):
        if (v_map['V5_dict_hit_improvement'].passed and
            delta > 0.02 and v_map['V7_content_run'].passed):
            verdict = 'FREE_TRIPLES_RESOLVED'
        elif v_map['V5_dict_hit_improvement'].passed:
            verdict = 'FREE_TRIPLES_PARTIAL'
        else:
            verdict = 'CONSTRAINTS_VALID_NO_IMPROVEMENT'
    elif v_map['V1_constraints_extracted'].passed and not v_map['V2_triple_consensus'].passed:
        verdict = 'CONSTRAINTS_FOUND_NO_CONSENSUS'
    elif v_map['V1_constraints_extracted'].passed:
        verdict = 'CONSTRAINTS_FOUND_NO_CONSENSUS'
    else:
        verdict = 'INSUFFICIENT_PARADIGM_DATA'

    gate_passed = verdict in ('FREE_TRIPLES_RESOLVED', 'FREE_TRIPLES_PARTIAL')

    print(f"\n  VERDICT: {verdict}")
    print(f"  Gate: {'PASS' if gate_passed else 'FAIL'}")

    # ── Save ──────────────────────────────────────────────────────────
    runtime = round(time.time() - t0, 2)

    result = Phase53IntegrateResult(
        paradigms_analyzed=paradigms_analyzed,
        total_constraints=total_constraints,
        unique_constraints=unique_constraints,
        triples_with_constraints=n_triples_with,
        accepted_corrections=n_accepted_a,
        corrections_applied=corrections_applied,
        dict_hit_pre=dict_hit_pre,
        dict_hit_post=dict_hit_post,
        delta_dict_hit=delta,
        signal_words_preserved=signal_preserved,
        null_z_score=null_z,
        null_selectivity=null_sel,
        track_b_verdict=track_b_verdict,
        newly_decoded_tokens=newly_decoded,
        longest_run=longest_run,
        n_content_runs=n_content_runs,
        best_content_run_length=bcr_length,
        best_content_run_text=bcr_text[:300],
        validations=[asdict(v) for v in validations],
        n_passed=n_passed,
        n_total=len(validations),
        verdict=verdict,
        gate_passed=gate_passed,
        runtime_seconds=runtime,
    )

    out_path = _save_json(rd, 'phase53_integrate.json', asdict(result))
    print(f"\n  Saved -> {out_path}")
    print(f"  Completed in {runtime:.1f}s")


def run_phase53() -> None:
    """Run all Phase 53 tracks + integration."""
    from voynich.phases.paradigm_constraints import run_paradigm_constraints
    from voynich.phases.triple_resolution import run_triple_resolution
    from voynich.phases.resolved_decode import run_resolved_decode

    run_paradigm_constraints()
    print()
    run_triple_resolution()
    print()
    run_resolved_decode()
    print()
    run_phase53_integrate()
