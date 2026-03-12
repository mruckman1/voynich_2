"""
Phase 47 Integration
====================
Combines Track A (Z-Score Audit), Track B (Word Disambiguation),
Track C (Structural Reading), and Track D (Sequence Analysis)
into a final Phase 47 verdict.

Dependency chain:
    z_reproduce_42.json     (Track A, 47A.1)
    z_reproduce_46.json     (Track A, 47A.2)
    z_diff.json             (Track A, 47A.3)
    z_canonical.json        (Track A, 47A.4)
    z_sensitivity.json      (Track A, 47A.5)
    disamb_lattice.json     (Track B, 47B.1)
    disamb_bigram.json      (Track B, 47B.2)
    disamb_viterbi.json     (Track B, 47B.3)
    disamb_eval.json        (Track B, 47B.4)
    disamb_compare.json     (Track B, 47B.5)
    read_ngrams.json        (Track C, 47C.1)
    read_recipes.json       (Track C, 47C.2)
    read_topics.json        (Track C, 47C.3)
    read_star_folios.json   (Track C, 47C.4)
    read_sections.json      (Track C, 47C.5)
    seq_overlap.json        (Track D, 47D.1)
    seq_continuity.json     (Track D, 47D.2)
    seq_boundaries.json     (Track D, 47D.3)
    seq_reorder.json        (Track D, 47D.4)
    phase46_integrate.json  (Phase 46 baseline)
        -> phase47_integrate.json
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


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------

@dataclass
class Phase47IntegrateResult:
    # Track A summary
    track_a_p29_reproduced: bool
    track_a_p46_reproduced: bool
    track_a_canonical_z: float
    track_a_canonical_table: str
    track_a_z_robust: bool
    # Track B summary
    track_b_best_variant: str
    track_b_delta_dict_hit: float
    track_b_beneficial: bool
    # Track C summary
    track_c_n_ngrams: int
    track_c_n_recipes: int
    track_c_optimal_k: int
    track_c_silhouette: float
    track_c_n_glosses: int
    track_c_mean_jsd: float
    # Track D summary
    track_d_consecutive_jaccard: float
    track_d_plausibility_rate: float
    track_d_n_anomalies: int
    track_d_reorder_verdict: str
    # Cross-track findings
    cross_track_findings: List[str]
    # Validation battery
    validations: Dict[str, bool]
    n_validations_passed: int
    gate_passed: bool
    # Verdict
    phase47_verdict: str
    phase47_rationale: str
    # Progression
    progression: List[Dict]
    runtime_seconds: float


def run_phase47_integrate() -> None:
    """Compile all Phase 47 tracks into a verdict."""
    t0 = time.time()
    print("=" * 70)
    print("PHASE 47 INTEGRATION")
    print("=" * 70)

    rd = _results_dir()

    # ------ Track A ------
    z42 = _safe_load(os.path.join(rd, 'z_reproduce_42.json'))
    z46 = _safe_load(os.path.join(rd, 'z_reproduce_46.json'))
    z_diff = _safe_load(os.path.join(rd, 'z_diff.json'))
    z_canon = _safe_load(os.path.join(rd, 'z_canonical.json'))
    z_sens = _safe_load(os.path.join(rd, 'z_sensitivity.json'))

    p29_reproduced = z42.get('within_tolerance', False)
    p46_reproduced = z46.get('within_tolerance', False)
    canonical_z = z_canon.get('best_z_total', 0.0)
    canonical_table = z_canon.get('best_table', '')
    z_robust = len(z_sens.get('sensitive_parameters', [])) <= 1

    # ------ Track B ------
    disamb = _safe_load(os.path.join(rd, 'disamb_compare.json'))
    b_best = disamb.get('best_variant', 'none')
    b_delta = disamb.get('best_delta_dict_hit', 0.0)
    b_beneficial = disamb.get('disambiguation_beneficial', False)

    # ------ Track C ------
    ngrams = _safe_load(os.path.join(rd, 'read_ngrams.json'))
    recipes = _safe_load(os.path.join(rd, 'read_recipes.json'))
    topics = _safe_load(os.path.join(rd, 'read_topics.json'))
    star = _safe_load(os.path.join(rd, 'read_star_folios.json'))
    sections = _safe_load(os.path.join(rd, 'read_sections.json'))

    c_n_ngrams = ngrams.get('n_filtered_ngrams', 0)
    c_n_recipes = recipes.get('n_recipes', 0)
    c_k = topics.get('optimal_k', 0)
    c_sil = topics.get('silhouette_score', 0.0)
    c_glosses = star.get('total_gloss_attempts', 0)
    c_jsd = sections.get('mean_jsd', 0.0)

    # ------ Track D ------
    overlap = _safe_load(os.path.join(rd, 'seq_overlap.json'))
    continuity = _safe_load(os.path.join(rd, 'seq_continuity.json'))
    boundaries = _safe_load(os.path.join(rd, 'seq_boundaries.json'))
    reorder = _safe_load(os.path.join(rd, 'seq_reorder.json'))

    d_consec_jac = overlap.get('mean_consecutive_jaccard', 0.0)
    d_plaus = continuity.get('plausibility_rate', 0.0)
    d_n_anom = boundaries.get('n_anomalous', 0)
    d_reorder = reorder.get('verdict', 'UNKNOWN')

    # ------ Validation Battery V1-V8 ------
    consec_vs_random = overlap.get('consecutive_vs_random_ratio', 0.0)

    validations = {
        'V1_p29_reproduced': p29_reproduced,
        'V2_p46_reproduced': p46_reproduced,
        'V3_canonical_z_computed': len(z_canon.get('per_table', [])) >= 4,
        'V4_disamb_change_rate': any(
            0.05 <= v.get('change_rate', 0) <= 0.30
            for v in disamb.get('summary', [])
        ),
        'V5_bedrock_survived': all(
            v for v in (disamb.get('bedrock_survival') or {}).values()
        ) if disamb.get('bedrock_survival') else True,
        'V6_ngrams_found': c_n_ngrams >= 10,
        'V7_topics_meaningful': c_k >= 3 and c_sil >= 0.1,
        'V8_overlap_computed': overlap.get('n_folios', 0) >= 200,
    }
    n_passed = sum(1 for v in validations.values() if v)
    gate_passed = n_passed >= 6

    # ------ Cross-Track Findings ------
    findings = []
    if p29_reproduced and p46_reproduced:
        findings.append(
            f"Z-score discrepancy RESOLVED: methodological differences explain "
            f"Phase 29 z~{z42.get('reproduced_z',0):.1f} vs Phase 46 z~{z46.get('reproduced_z',0):.1f}. "
            f"Canonical z={canonical_z:.1f} on {canonical_table}."
        )
    if b_beneficial:
        findings.append(
            f"Word-level disambiguation BENEFICIAL: {b_best} improves dict-hit by {b_delta:+.1%}."
        )
    else:
        findings.append(
            "Word-level disambiguation NOT BENEFICIAL: internal bigram model too noisy."
        )
    if c_n_ngrams >= 10:
        findings.append(
            f"Structural reading: {c_n_ngrams} recurring n-grams, {c_n_recipes} recipes, "
            f"k={c_k} topic clusters (sil={c_sil:.2f})."
        )
    if consec_vs_random > 1.1:
        findings.append(
            f"Sequence analysis: consecutive folios {consec_vs_random:.2f}x more similar "
            f"than random. {d_n_anom} anomalous boundaries. Verdict: {d_reorder}."
        )

    # ------ Verdict ------
    # Decision table from README
    if not p29_reproduced and not p46_reproduced:
        verdict = 'Z_METHODOLOGY_ERROR'
        rationale = 'Neither z-score could be reproduced — publication risk.'
    elif b_beneficial and c_n_ngrams >= 10:
        verdict = 'WORD_LEVEL_IMPROVEMENT'
        rationale = (
            f'Disambiguation improves dict-hit by {b_delta:+.1%}. '
            f'{c_n_ngrams} structural patterns found.'
        )
    elif c_n_ngrams >= 10:
        verdict = 'READING_ONLY'
        rationale = (
            f'Content extraction successful ({c_n_ngrams} patterns, '
            f'{c_glosses} glosses) but no decode improvement.'
        )
    elif n_passed >= 6:
        verdict = 'COMPUTATIONAL_CEILING'
        rationale = 'No further internal progress possible. External evidence needed.'
    else:
        verdict = 'PARTIAL_ANALYSIS'
        rationale = f'Only {n_passed}/8 validations passed.'

    # ------ Progression ------
    progression = [
        {'phase': 'Phase 11', 'dict_hit': 0.111, 'selectivity': 1.92, 'note': 'CSP phonetic decoder'},
        {'phase': 'Phase 14', 'dict_hit': 0.194, 'selectivity': 3.00, 'note': '25 stroke-feature triples'},
        {'phase': 'Phase 15', 'dict_hit': 0.354, 'selectivity': 2.55, 'note': 'Medieval dict expansion'},
        {'phase': 'Phase 16', 'dict_hit': 0.436, 'selectivity': 3.38, 'note': 'Modifier detection (full corpus)'},
        {'phase': 'Phase 46', 'dict_hit': 0.436, 'note': 'TABLE_SELECTED T_P15'},
        {'phase': 'Phase 47', 'dict_hit': 0.436 + (b_delta if b_beneficial else 0),
         'note': verdict},
    ]

    print(f"\n  Track A: P29={'PASS' if p29_reproduced else 'FAIL'}, "
          f"P46={'PASS' if p46_reproduced else 'FAIL'}, "
          f"canonical z={canonical_z:.1f}")
    print(f"  Track B: {b_best}, delta={b_delta:+.4f}, "
          f"beneficial={b_beneficial}")
    print(f"  Track C: {c_n_ngrams} n-grams, {c_n_recipes} recipes, "
          f"k={c_k}, sil={c_sil:.2f}")
    print(f"  Track D: consec_jac={d_consec_jac:.4f}, "
          f"plaus={d_plaus:.1%}, anomalies={d_n_anom}, {d_reorder}")
    print(f"\n  Validations: {n_passed}/8  {'PASS' if gate_passed else 'FAIL'}")
    for k, v in validations.items():
        print(f"    {k:30s}: {'PASS' if v else 'FAIL'}")
    print(f"\n  Phase 47 verdict: {verdict}")
    print(f"  Rationale: {rationale}")

    result = Phase47IntegrateResult(
        track_a_p29_reproduced=p29_reproduced,
        track_a_p46_reproduced=p46_reproduced,
        track_a_canonical_z=round(canonical_z, 4),
        track_a_canonical_table=canonical_table,
        track_a_z_robust=z_robust,
        track_b_best_variant=b_best,
        track_b_delta_dict_hit=round(b_delta, 4),
        track_b_beneficial=b_beneficial,
        track_c_n_ngrams=c_n_ngrams,
        track_c_n_recipes=c_n_recipes,
        track_c_optimal_k=c_k,
        track_c_silhouette=round(c_sil, 4),
        track_c_n_glosses=c_glosses,
        track_c_mean_jsd=round(c_jsd, 6),
        track_d_consecutive_jaccard=round(d_consec_jac, 4),
        track_d_plausibility_rate=round(d_plaus, 4),
        track_d_n_anomalies=d_n_anom,
        track_d_reorder_verdict=d_reorder,
        cross_track_findings=findings,
        validations=validations,
        n_validations_passed=n_passed,
        gate_passed=gate_passed,
        phase47_verdict=verdict,
        phase47_rationale=rationale,
        progression=progression,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'phase47_integrate.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2)
    print(f"\n  Saved -> {out_path}")


# ---------------------------------------------------------------------------
# Full Phase 47 pipeline
# ---------------------------------------------------------------------------

def run_phase47() -> None:
    """Run full Phase 47 pipeline: A -> B -> C -> D -> Integration."""
    from voynich.phases.zscore_audit import run_track_a_47
    from voynich.phases.word_disambiguation import run_track_b_47
    from voynich.phases.structural_reading_47 import run_track_c_47
    from voynich.phases.sequence_analysis import run_track_d_47

    print("\n" + "=" * 70)
    print("PHASE 47: Z-Score Audit, Disambiguation, Reading, Sequence")
    print("=" * 70)

    print("\n\n>>> TRACK A: Z-Score Methodology Audit <<<\n")
    run_track_a_47()

    print("\n\n>>> TRACK B: Word-Level Disambiguation <<<\n")
    run_track_b_47()

    print("\n\n>>> TRACK C: Structural Reading <<<\n")
    run_track_c_47()

    print("\n\n>>> TRACK D: Manuscript Sequence Analysis <<<\n")
    run_track_d_47()

    print("\n\n>>> INTEGRATION <<<\n")
    run_phase47_integrate()
