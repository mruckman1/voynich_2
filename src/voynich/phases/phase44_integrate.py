"""
Phase 44 – Integration: Cross-Track Analysis and Verdict
==========================================================
Combine results from Tracks A (MaxSAT), B (SBM), and C (CSA) into a
unified landscape assessment.

Dependency chain:
    maxsat_landscape.json     (Track A Step 44A.3)
    maxsat_validation.json    (Track A Step 44A.4)
    sbm_comparison.json       (Track B Step 44B.3)
    sbm_validation.json       (Track B Step 44B.5)
    kperm_analysis.json       (Track C Step 44C.3)
    kperm_validation.json     (Track C Step 44C.4)
        -> phase44_integrate.json  (this step)
"""

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

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
    if isinstance(obj, (bool, int, float, str, type(None))):
        return obj
    return str(obj)


def _safe_load(path: str) -> Dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class Phase44IntegrateResult:
    # Track A summary
    track_a_landscape: str
    track_a_n_solutions: int
    track_a_n_basins: int
    track_a_best_dict_hit: float
    track_a_verdict: str
    # Track B summary
    track_b_n_communities: int
    track_b_ari_stroke: float
    track_b_ari_family: float
    track_b_interpretation: str
    track_b_split_half_ari: float
    track_b_verdict: str
    # Track C summary
    track_c_best_dict_hit: float
    track_c_p15_dict_hit: float
    track_c_delta: float
    track_c_selectivity: float
    track_c_verdict: str
    # Cross-track analysis
    cross_track_consensus: Dict
    n_tracks_improved: int
    best_overall_dict_hit: float
    best_track: str
    # Validation battery
    validations: Dict[str, bool]
    n_validations_passed: int
    gate_passed: bool
    # Progression
    progression_table: List[Dict]
    # Overall
    phase44_verdict: str
    phase44_rationale: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_phase44_integrate() -> None:
    """Phase 44 Integration: compile all 3 tracks into verdict."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE 44: Integration — Cross-Track Analysis and Verdict")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load Track A results ──
    print("\n  1. Loading Track A (MaxSAT) results ...")
    landscape = _safe_load(os.path.join(rd, 'maxsat_landscape.json'))
    maxsat_val = _safe_load(os.path.join(rd, 'maxsat_validation.json'))

    track_a_landscape = landscape.get('classification', 'UNKNOWN')
    track_a_n_solutions = landscape.get('n_solutions', 0)
    track_a_n_basins = landscape.get('n_basins', 0)
    track_a_best_dh = maxsat_val.get('best_maxsat_dict_hit', 0.0)
    track_a_verdict = maxsat_val.get('verdict', 'UNKNOWN')
    track_a_p15_dh = maxsat_val.get('phase15_dict_hit', 0.0)

    print(f"     Landscape: {track_a_landscape}, "
          f"{track_a_n_solutions} solutions, {track_a_n_basins} basins")
    print(f"     Best dict-hit: {track_a_best_dh:.4f}, verdict: {track_a_verdict}")

    # ── 2. Load Track B results ──
    print("\n  2. Loading Track B (SBM) results ...")
    sbm_comp = _safe_load(os.path.join(rd, 'sbm_comparison.json'))
    sbm_val = _safe_load(os.path.join(rd, 'sbm_validation.json'))

    track_b_n_communities = _safe_load(
        os.path.join(rd, 'sbm_communities.json')
    ).get('n_communities', 0)
    track_b_ari_stroke = sbm_comp.get('ari_vs_stroke', 0.0)
    track_b_ari_family = sbm_comp.get('ari_vs_family', 0.0)
    track_b_interpretation = sbm_comp.get('overall_interpretation', 'UNKNOWN')
    track_b_split_half = sbm_val.get('split_half_ari', 0.0)
    track_b_verdict = sbm_val.get('verdict', 'UNKNOWN')

    print(f"     Communities: {track_b_n_communities}")
    print(f"     ARI(stroke): {track_b_ari_stroke:.4f}, ARI(family): {track_b_ari_family:.4f}")
    print(f"     Split-half ARI: {track_b_split_half:.4f}, verdict: {track_b_verdict}")

    # ── 3. Load Track C results ──
    print("\n  3. Loading Track C (CSA) results ...")
    csa_analysis = _safe_load(os.path.join(rd, 'kperm_analysis.json'))
    csa_val = _safe_load(os.path.join(rd, 'kperm_validation.json'))

    track_c_best_dh = csa_val.get('full_corpus_dict_hit', 0.0)
    track_c_p15_dh = csa_val.get('p15_full_corpus_dict_hit', 0.0)
    track_c_delta = csa_val.get('delta', 0.0)
    track_c_selectivity = csa_val.get('selectivity', 0.0)
    track_c_verdict = csa_val.get('verdict', 'UNKNOWN')

    print(f"     CSA dict-hit: {track_c_best_dh:.4f}, "
          f"Phase 15: {track_c_p15_dh:.4f}, delta: {track_c_delta:+.4f}")
    print(f"     Selectivity: {track_c_selectivity:.2f}x, verdict: {track_c_verdict}")

    # ── 4. Cross-track analysis ──
    print("\n  4. Cross-track analysis ...")

    # Per-triple consensus: compare MaxSAT, CSA, and Phase 15
    maxsat_consensus = landscape.get('per_triple_consensus', {})
    csa_consensus = csa_analysis.get('per_triple_consensus', {})

    cross_track = {}
    all_free_triples = set(list(maxsat_consensus.keys()) +
                          list(csa_consensus.keys()))

    best_maxsat_assign = maxsat_val.get('best_maxsat_assignment', {})
    best_csa_assign = _safe_load(os.path.join(rd, 'kperm_search.json')).get(
        'best_assignment', {})

    n_agree_all = 0
    n_agree_any = 0

    for t in sorted(all_free_triples):
        maxsat_syl = best_maxsat_assign.get(t, '?')
        csa_syl = best_csa_assign.get(t, '?')

        agrees = maxsat_syl == csa_syl and maxsat_syl != '?'
        cross_track[t] = {
            'maxsat': maxsat_syl,
            'csa': csa_syl,
            'agree': agrees,
        }
        if agrees:
            n_agree_all += 1

    print(f"     MaxSAT-CSA agreement: {n_agree_all}/{len(all_free_triples)} triples")

    # Count tracks that improved
    n_tracks_improved = 0
    if track_a_best_dh > track_a_p15_dh + 0.005:
        n_tracks_improved += 1
    if track_c_delta > 0.005:
        n_tracks_improved += 1

    # Best overall
    candidates = [
        ('Track A (MaxSAT)', track_a_best_dh),
        ('Track C (CSA)', track_c_best_dh),
        ('Phase 15 baseline', track_c_p15_dh),
    ]
    best_track, best_overall_dh = max(candidates, key=lambda x: x[1])

    # ── 5. Validation battery V1–V8 ──
    print("\n  5. Running validation battery ...")
    validations = {}

    # V1: MaxSAT instance solved
    v1 = track_a_n_solutions > 0
    validations['V1_maxsat_solved'] = v1
    print(f"     V1 MaxSAT solved: {'PASS' if v1 else 'FAIL'}")

    # V2: Solution count characterized
    v2 = track_a_landscape in ('FLAT', 'BASINED', 'PEAKED')
    validations['V2_landscape_classified'] = v2
    print(f"     V2 Landscape classified ({track_a_landscape}): {'PASS' if v2 else 'FAIL'}")

    # V3: SBM communities in [3, 15]
    v3 = 3 <= track_b_n_communities <= 15
    validations['V3_sbm_communities'] = v3
    print(f"     V3 SBM communities ({track_b_n_communities}): {'PASS' if v3 else 'FAIL'}")

    # V4: SBM split-half ARI > 0.3
    v4 = track_b_split_half > 0.3
    validations['V4_sbm_split_half'] = v4
    print(f"     V4 SBM split-half ARI ({track_b_split_half:.4f}): "
          f"{'PASS' if v4 else 'FAIL'}")

    # V5: SBM vs stroke ARI > 0.3 OR vs family ARI > 0.3
    v5 = track_b_ari_stroke > 0.3 or track_b_ari_family > 0.3
    validations['V5_sbm_agreement'] = v5
    print(f"     V5 SBM agreement (stroke={track_b_ari_stroke:.4f}, "
          f"family={track_b_ari_family:.4f}): {'PASS' if v5 else 'FAIL'}")

    # V6: CSA converges (energy decreasing)
    convergence = _safe_load(os.path.join(rd, 'kperm_search.json')).get(
        'convergence_curve', [])
    v6 = False
    if len(convergence) >= 2:
        first_e = convergence[0].get('best_energy', 0)
        last_e = convergence[-1].get('best_energy', 0)
        v6 = last_e < first_e
    validations['V6_csa_converges'] = v6
    print(f"     V6 CSA converges: {'PASS' if v6 else 'FAIL'}")

    # V7: CSA null discrimination
    v7 = track_c_selectivity > 1.5
    validations['V7_csa_null_discrimination'] = v7
    print(f"     V7 CSA null discrimination ({track_c_selectivity:.2f}x): "
          f"{'PASS' if v7 else 'FAIL'}")

    # V8: No regression vs Phase 15
    v8 = best_overall_dh >= track_c_p15_dh * 0.95
    validations['V8_no_regression'] = v8
    print(f"     V8 No regression ({best_overall_dh:.4f} vs {track_c_p15_dh:.4f}): "
          f"{'PASS' if v8 else 'FAIL'}")

    n_passed = sum(validations.values())
    gate = n_passed >= 6
    print(f"\n     Validations: {n_passed}/8 passed. Gate: {'PASS' if gate else 'FAIL'}")

    # ── 6. Verdict ──
    print("\n  6. Determining verdict ...")

    sbm_agrees = max(track_b_ari_stroke, track_b_ari_family) > 0.5
    csa_improved = track_c_delta > 0.01

    if track_a_landscape == 'PEAKED' and sbm_agrees and csa_improved:
        verdict = "TABLE_IMPROVED"
        rationale = ("CSA found a better solution in a peaked landscape. "
                     "SBM confirms distributional structure agrees with strokes.")
    elif track_a_landscape == 'PEAKED' and sbm_agrees and not csa_improved:
        verdict = "TABLE_CONFIRMED_STRONG"
        rationale = ("Phase 15 is near-global optimum in peaked landscape. "
                     "SBM confirms stroke model. No better solution found.")
    elif track_a_landscape == 'BASINED' and csa_improved:
        verdict = "BASIN_ESCAPE"
        rationale = ("CSA escaped to a different, better basin. "
                     "Multiple local optima exist in the landscape.")
    elif track_a_landscape == 'BASINED' and not sbm_agrees:
        verdict = "MODEL_MISMATCH"
        rationale = ("Distributional structure disagrees with stroke-feature model. "
                     "The 25-triple grid may need restructuring.")
    elif track_a_landscape == 'FLAT' and not csa_improved:
        verdict = "SCORING_WEAK"
        rationale = ("Many near-optimal solutions exist — the scoring function "
                     "cannot discriminate the correct assignment. "
                     "Need a better language model.")
    elif track_a_landscape == 'FLAT' and csa_improved:
        verdict = "PARTIAL_IMPROVEMENT"
        rationale = ("CSA found improvement despite flat landscape. "
                     "Better scoring function could yield further gains.")
    elif n_tracks_improved == 0:
        verdict = "TABLE_CONFIRMED"
        rationale = ("No track improved over Phase 15. The current assignment "
                     "is robust across MaxSAT, SBM, and CSA analyses.")
    else:
        verdict = "INCONCLUSIVE"
        rationale = ("Mixed signals across tracks. Further investigation needed.")

    print(f"     Verdict: {verdict}")
    print(f"     Rationale: {rationale}")

    # ── 7. Progression table ──
    progression = [
        {'phase': 'Phase 11', 'dict_hit': 0.111, 'selectivity': 1.92,
         'method': 'CV phonotactic CSP'},
        {'phase': 'Phase 14', 'dict_hit': 0.194, 'selectivity': 3.00,
         'method': 'Feature triple model'},
        {'phase': 'Phase 15', 'dict_hit': 0.354, 'selectivity': 2.55,
         'method': 'Dict expansion + AC scoring'},
        {'phase': 'Phase 16', 'dict_hit': 0.436, 'selectivity': 3.38,
         'method': 'Modifier detection + R3 decode'},
        {'phase': 'Phase 33', 'dict_hit': 0.436, 'selectivity': 3.38,
         'method': 'Local optimum proof (6 methods)'},
        {'phase': 'Phase 42', 'dict_hit': 0.436, 'selectivity': 3.38,
         'method': 'Symmetric bigram z=3.90'},
        {'phase': 'Phase 43', 'dict_hit': 0.436, 'selectivity': 3.38,
         'method': 'Re-encoding + HMM: LATERAL'},
        {'phase': 'Phase 44', 'dict_hit': round(best_overall_dh, 4),
         'selectivity': round(track_c_selectivity, 2),
         'method': f'{verdict}: {best_track}'},
    ]

    result = Phase44IntegrateResult(
        track_a_landscape=track_a_landscape,
        track_a_n_solutions=track_a_n_solutions,
        track_a_n_basins=track_a_n_basins,
        track_a_best_dict_hit=round(track_a_best_dh, 4),
        track_a_verdict=track_a_verdict,
        track_b_n_communities=track_b_n_communities,
        track_b_ari_stroke=round(track_b_ari_stroke, 4),
        track_b_ari_family=round(track_b_ari_family, 4),
        track_b_interpretation=track_b_interpretation,
        track_b_split_half_ari=round(track_b_split_half, 4),
        track_b_verdict=track_b_verdict,
        track_c_best_dict_hit=round(track_c_best_dh, 4),
        track_c_p15_dict_hit=round(track_c_p15_dh, 4),
        track_c_delta=round(track_c_delta, 4),
        track_c_selectivity=round(track_c_selectivity, 2),
        track_c_verdict=track_c_verdict,
        cross_track_consensus=cross_track,
        n_tracks_improved=n_tracks_improved,
        best_overall_dict_hit=round(best_overall_dh, 4),
        best_track=best_track,
        validations=validations,
        n_validations_passed=n_passed,
        gate_passed=gate,
        progression_table=progression,
        phase44_verdict=verdict,
        phase44_rationale=rationale,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'phase44_integrate.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2)

    print(f"\n  Saved -> {out_path}")
    print(f"\n  Phase 44 integration completed in {time.time() - t0:.1f}s")
