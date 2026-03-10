"""
Phase 31.9: Phase 31 Integration
===================================
Combine Path 2 (Botanical Anchors) and Path 4 (Structural Reframing) results
into a unified assessment with interaction effects and combined best table.

Dependency chain:
    consensus_plants.json      (31.1)
    plant_name_csp.json        (31.2)
    plant_name_propagate.json  (31.3)
    botanical_signal.json      (31.4)
    determinative_test.json    (31.5)
    compound_sign_test.json    (31.6)
    interleaved_test.json      (31.7)
    resegmentation_test.json   (31.8)
        → phase31_integrate.json  (this step)
"""

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


def _load_json(rd: str, filename: str) -> Optional[Dict]:
    """Load a JSON result file, returning None if not found."""
    path = os.path.join(rd, filename)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Path2Summary:
    """Summary of Path 2: Botanical Anchor Attack."""
    n_tier_a_folios: int
    n_tier_b_folios: int
    n_cross_folio_consistent: int
    plant_csp_verdict: str
    propagation_verdict: str
    cascade_detected: bool
    botanical_signal_verdict: str
    new_confirmed_triples: int
    final_dict_hit: float


@dataclass
class Path4Summary:
    """Summary of Path 4: Structural Reframing."""
    determinative_verdict: str
    gallows_strip_delta: float
    compound_sign_verdict: str
    root_only_delta: float
    interleaved_verdict: str
    separation_improvement: float
    resegmentation_verdict: str
    best_reseg_delta: float


@dataclass
class ProgressionEntry:
    """One row of the progression table."""
    phase: str
    dict_hit: float
    signal: str
    bigram_z: str
    confirmed: str
    triples: str


@dataclass
class Phase31IntegrateResult:
    """Full Step 31.9 output."""
    path2_summary: Dict
    path4_summary: Dict
    interaction_effects: List[str]
    recommended_changes: List[str]
    combined_best_dict_hit: float
    progression_table: List[Dict]
    overall_verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_phase31_integrate() -> None:
    """Step 31.9: Integrate all Phase 31 results."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE 31.9: Phase 31 Integration")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load all results ──
    print("\n  1. Loading all Phase 31 results...")

    cp = _load_json(rd, 'consensus_plants.json')
    csp = _load_json(rd, 'plant_name_csp.json')
    prop = _load_json(rd, 'plant_name_propagate.json')
    bot = _load_json(rd, 'botanical_signal.json')
    det = _load_json(rd, 'determinative_test.json')
    comp = _load_json(rd, 'compound_sign_test.json')
    inter = _load_json(rd, 'interleaved_test.json')
    reseg = _load_json(rd, 'resegmentation_test.json')

    loaded = sum(1 for x in [cp, csp, prop, bot, det, comp, inter, reseg] if x)
    print(f"     Loaded {loaded}/8 result files")

    # ── 2. Path 2 Assessment ──
    print("\n  2. Path 2: Botanical Anchor Attack")

    n_tier_a = cp.get('n_tier_a', 0) if cp else 0
    n_tier_b = cp.get('n_tier_b', 0) if cp else 0
    n_cross = csp.get('n_new_triple_assignments', 0) if csp else 0
    csp_verdict = csp.get('verdict', 'NOT_RUN') if csp else 'NOT_RUN'
    prop_verdict = prop.get('verdict', 'NOT_RUN') if prop else 'NOT_RUN'
    cascade = prop.get('cascade_detected', False) if prop else False
    bot_verdict = bot.get('verdict', 'NOT_RUN') if bot else 'NOT_RUN'
    new_triples = n_cross
    final_dict_hit = prop.get('final_dict_hit', 0.0) if prop else 0.0

    path2 = Path2Summary(
        n_tier_a_folios=n_tier_a,
        n_tier_b_folios=n_tier_b,
        n_cross_folio_consistent=n_cross,
        plant_csp_verdict=csp_verdict,
        propagation_verdict=prop_verdict,
        cascade_detected=cascade,
        botanical_signal_verdict=bot_verdict,
        new_confirmed_triples=new_triples,
        final_dict_hit=final_dict_hit,
    )

    print(f"     Tier A: {n_tier_a}, Tier B: {n_tier_b}")
    print(f"     Cross-folio consistent: {n_cross}")
    print(f"     CSP: {csp_verdict}")
    print(f"     Propagation: {prop_verdict}")
    print(f"     Cascade: {cascade}")
    print(f"     Botanical signal: {bot_verdict}")

    # ── 3. Path 4 Assessment ──
    print("\n  3. Path 4: Structural Reframing")

    det_verdict = det.get('verdict', 'NOT_RUN') if det else 'NOT_RUN'
    det_strip = det.get('stripping', {}).get('delta_dict_hit', 0.0) if det else 0.0
    comp_verdict = comp.get('verdict', 'NOT_RUN') if comp else 'NOT_RUN'
    comp_delta = comp.get('root_delta', 0.0) if comp else 0.0
    inter_verdict = inter.get('verdict', 'NOT_RUN') if inter else 'NOT_RUN'
    inter_imp = inter.get('real_improvement', 0.0) if inter else 0.0
    reseg_verdict = reseg.get('verdict', 'NOT_RUN') if reseg else 'NOT_RUN'
    reseg_delta = reseg.get('best_delta', 0.0) if reseg else 0.0

    path4 = Path4Summary(
        determinative_verdict=det_verdict,
        gallows_strip_delta=det_strip,
        compound_sign_verdict=comp_verdict,
        root_only_delta=comp_delta,
        interleaved_verdict=inter_verdict,
        separation_improvement=inter_imp,
        resegmentation_verdict=reseg_verdict,
        best_reseg_delta=reseg_delta,
    )

    print(f"     Determinative: {det_verdict} (Δ={det_strip:+.4f})")
    print(f"     Compound sign: {comp_verdict} (root Δ={comp_delta:+.4f})")
    print(f"     Interleaved: {inter_verdict} (Δ={inter_imp:+.4f})")
    print(f"     Re-segmentation: {reseg_verdict} (Δ={reseg_delta:+.4f})")

    # ── 4. Interaction effects ──
    print("\n  4. Interaction effects...")
    interactions = []

    if det_verdict.startswith('DETERMINATIVE') and 'LIKELY' in det_verdict:
        if n_cross > 0:
            interactions.append(
                "Gallows as determinatives + botanical CSP: if gallows are "
                "non-phonetic, botanical CSP alignments should improve when "
                "gallows chars are excluded from syllabic char count."
            )

    if reseg_verdict == 'RESEGMENTATION_IMPROVES' and n_cross > 0:
        interactions.append(
            "Re-segmentation + botanical CSP: re-segmented chars may produce "
            "better plant name alignments. Re-run plant CSP with new scheme."
        )

    if inter_verdict == 'SEPARATION_BENEFICIAL':
        interactions.append(
            "Language B separation should be applied BEFORE all other analyses "
            "for cleaner signal. Re-run Path 2 on Stream A only."
        )

    if not interactions:
        interactions.append("No significant interaction effects detected.")

    for ie in interactions:
        print(f"     • {ie}")

    # ── 5. Recommended changes ──
    print("\n  5. Recommended changes...")
    recommendations = []

    if 'LIKELY' in det_verdict:
        recommendations.append("Strip gallows before decoding (treat as determinatives)")
    if comp_verdict == 'COMPOUND_SIGN_SUPPORTED':
        recommendations.append("Decode roots only (strip prefixes/suffixes)")
    if inter_verdict == 'SEPARATION_BENEFICIAL':
        recommendations.append("Separate Language B tokens before analysis")
    if reseg_verdict == 'RESEGMENTATION_IMPROVES':
        best = reseg.get('best_scheme', '?') if reseg else '?'
        recommendations.append(f"Adopt re-segmentation scheme {best}")
    if n_cross > 0:
        recommendations.append(f"Add {n_cross} new triple assignments from botanical CSP")

    if not recommendations:
        recommendations.append("No changes recommended — current framework is best available")

    for rec in recommendations:
        print(f"     → {rec}")

    # ── 6. Combined best dict_hit ──
    # This would require re-running decoding with all recommended changes
    # For now, estimate by summing positive deltas
    baseline_dict_hit = 0.436  # Phase 16 full-corpus baseline
    combined_delta = 0.0
    if 'LIKELY' in det_verdict and det_strip > 0:
        combined_delta += det_strip
    if comp_delta > 0.01:
        combined_delta += comp_delta * 0.5  # Partial credit
    if inter_imp > 0 and inter_verdict == 'SEPARATION_BENEFICIAL':
        combined_delta += inter_imp
    if reseg_delta > 0:
        combined_delta += reseg_delta

    combined_best = baseline_dict_hit + combined_delta

    print(f"\n  6. Combined best dict_hit estimate: {combined_best:.4f}")

    # ── 7. Progression table ──
    progression = [
        ProgressionEntry('16', 0.436, '—', '—', '—', '—'),
        ProgressionEntry('28', 0.436, '16.5%', '—', '8', '12/25'),
        ProgressionEntry('29', 0.436, '16.5%', '6.14', '8', '12/25'),
        ProgressionEntry('30', 0.436, '16.5%', '6.14', '10', '12/25'),
        ProgressionEntry('31', round(combined_best, 3),
                         f"{(final_dict_hit or 0.165):.1%}",
                         '—',
                         str(10 + n_cross),
                         f"{12 + n_cross}/25"),
    ]

    print("\n  7. Progression table:")
    print(f"     {'Phase':>6s} | {'Dict-hit':>8s} | {'Signal':>8s} | "
          f"{'Bigram z':>8s} | {'Confirmed':>9s} | {'Triples':>7s}")
    print(f"     {'-' * 6} | {'-' * 8} | {'-' * 8} | {'-' * 8} | {'-' * 9} | {'-' * 7}")
    for p in progression:
        print(f"     {p.phase:>6s} | {p.dict_hit:>8.3f} | {p.signal:>8s} | "
              f"{p.bigram_z:>8s} | {p.confirmed:>9s} | {p.triples:>7s}")

    # ── 8. Overall verdict ──
    n_positive = sum([
        n_cross > 0,
        'LIKELY' in det_verdict,
        comp_verdict in ('COMPOUND_SIGN_SUPPORTED', 'COMPOUND_SIGN_POSSIBLE'),
        inter_verdict == 'SEPARATION_BENEFICIAL',
        reseg_verdict == 'RESEGMENTATION_IMPROVES',
    ])

    if n_positive >= 3:
        overall = "MAJOR_FRAMEWORK_REVISION"
    elif n_positive >= 1:
        overall = "INCREMENTAL_IMPROVEMENT"
    elif cascade:
        overall = "CASCADE_BREAKTHROUGH"
    else:
        overall = "FRAMEWORK_CONFIRMED"

    print(f"\n  Overall verdict: {overall}")
    print(f"  ({n_positive}/5 structural tests produced positive results)")

    # ── 9. Save ──
    result = Phase31IntegrateResult(
        path2_summary=_convert(asdict(path2)),
        path4_summary=_convert(asdict(path4)),
        interaction_effects=interactions,
        recommended_changes=recommendations,
        combined_best_dict_hit=round(combined_best, 4),
        progression_table=[_convert(asdict(p)) for p in progression],
        overall_verdict=overall,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'phase31_integrate.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2)

    print(f"\n  Saved → {out_path}")
    print(f"  Completed in {time.time() - t0:.1f}s")
