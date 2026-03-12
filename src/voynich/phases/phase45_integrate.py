"""
Phase 45 – Integration: Cross-Track Analysis and Verdict
==========================================================
Compile results from Track A (SBM Forensics), Track B (SBM Decode),
and Track C (Triple Consolidation) into a final Phase 45 verdict.

Dependency chain:
    sbm_profiles.json          (Track A.1)
    sbm_positions.json         (Track A.2)
    sbm_morphemes.json         (Track A.3)
    sbm_modifiers.json         (Track A.4)
    sbm_transitions.json       (Track A.5)
    sbm_factorization.json     (Track A.6)
    sbm_signal_words.json      (Track A.7)
    sbm_encoding.json          (Track B.1)
    sbm_csp.json               (Track B.2)
    sbm_signal.json            (Track B.3)
    sbm_hybrid.json            (Track B.4)
    sbm_maxsat.json            (Track B.5)
    triple_tiers.json          (Track C.1)
    triple_ambiguous.json      (Track C.2)
    canonical_table.json       (Track C.3)
    triple_impact.json         (Track C.4)
        -> phase45_integrate.json
"""

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

from voynich.core._paths import results_dir as _results_dir


# ── Helpers ──────────────────────────────────────────────────────────

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


# ── Dataclasses ──────────────────────────────────────────────────────

@dataclass
class Phase45IntegrateResult:
    # Track A summary
    track_a_community_encoding: str
    track_a_positional_significant: bool
    track_a_morphological_signal: bool
    track_a_modifier_concentrated: bool
    track_a_best_labeling: str
    track_a_best_ari: float
    track_a_signal_word_pattern: bool
    # Track B summary
    track_b_community_csp_dict_hit: float
    track_b_community_csp_selectivity: float
    track_b_hybrid_best_variant: str
    track_b_hybrid_dict_hit: float
    track_b_hybrid_selectivity: float
    track_b_landscape_shape: str
    track_b_landscape_differs: bool
    # Track C summary
    track_c_n_confirmed: int
    track_c_n_landscape_confirmed: int
    track_c_n_ambiguous: int
    track_c_n_changes: int
    track_c_canonical_dict_hit: float
    track_c_ambiguity_budget: float
    track_c_budget_gate: str
    # Validation battery
    validations: Dict[str, bool]
    n_validations_passed: int
    gate_passed: bool
    # Verdict
    phase45_verdict: str
    phase45_rationale: str
    # Progression
    progression: List[Dict]
    runtime_seconds: float


# ══════════════════════════════════════════════════════════════════════
#  Phase 45 Integration
# ══════════════════════════════════════════════════════════════════════

def run_phase45_integrate() -> None:
    """Compile all 3 tracks into Phase 45 verdict."""
    t0 = time.time()
    print("=" * 70)
    print("PHASE 45 INTEGRATION: Cross-Track Analysis and Verdict")
    print("=" * 70)

    rd = _results_dir()

    # ── Load Track A results ──
    profiles = _safe_load(os.path.join(rd, 'sbm_profiles.json'))
    positions = _safe_load(os.path.join(rd, 'sbm_positions.json'))
    morphemes = _safe_load(os.path.join(rd, 'sbm_morphemes.json'))
    modifiers = _safe_load(os.path.join(rd, 'sbm_modifiers.json'))
    transitions = _safe_load(os.path.join(rd, 'sbm_transitions.json'))
    factorization = _safe_load(os.path.join(rd, 'sbm_factorization.json'))
    signal_words = _safe_load(os.path.join(rd, 'sbm_signal_words.json'))

    # ── Load Track B results ──
    encoding = _safe_load(os.path.join(rd, 'sbm_encoding.json'))
    csp = _safe_load(os.path.join(rd, 'sbm_csp.json'))
    signal = _safe_load(os.path.join(rd, 'sbm_signal.json'))
    hybrid = _safe_load(os.path.join(rd, 'sbm_hybrid.json'))
    landscape = _safe_load(os.path.join(rd, 'sbm_maxsat.json'))

    # ── Load Track C results ──
    tiers = _safe_load(os.path.join(rd, 'triple_tiers.json'))
    ambiguous = _safe_load(os.path.join(rd, 'triple_ambiguous.json'))
    canonical = _safe_load(os.path.join(rd, 'canonical_table.json'))
    impact = _safe_load(os.path.join(rd, 'triple_impact.json'))

    # ── Track A summary ──
    positional_sig = positions.get('gate_passed', False)
    morph_signal = morphemes.get('gate_passed', False)
    modifier_conc = modifiers.get('gate_passed', False)
    best_labeling = factorization.get('best_labeling', 'unknown')
    best_ari = factorization.get('best_ari', 0.0)
    signal_pattern = signal_words.get('gate_passed', False)

    # Determine community identity
    if positional_sig and best_labeling == 'positional':
        community_encoding = 'POSITIONAL'
    elif morph_signal and best_labeling in ('morphological', 'modifier_class'):
        community_encoding = 'MORPHOLOGICAL'
    elif best_labeling == 'onset_consonant' and best_ari > 0.3:
        community_encoding = 'CONSONANT_CLASS'
    elif best_labeling == 'vowel' and best_ari > 0.3:
        community_encoding = 'VOWEL_CLASS'
    elif best_labeling == 'frequency_tier' and best_ari > 0.3:
        community_encoding = 'FREQUENCY_TIER'
    elif best_ari > 0.3:
        community_encoding = best_labeling.upper()
    else:
        community_encoding = 'UNDETERMINED'

    # ── Track B summary ──
    comm_csp_dh = csp.get('best_dict_hit', 0.0)
    comm_csp_sel = csp.get('selectivity', 0.0)
    hybrid_best = hybrid.get('best_variant', 'HYBRID_NONE')
    hybrid_dh = hybrid.get('best_dict_hit', 0.0)
    hybrid_sel = hybrid.get('best_selectivity', 0.0)
    land_shape = landscape.get('landscape_shape', 'UNKNOWN')
    land_differs = landscape.get('differs_from_stroke', False)

    # ── Track C summary ──
    n_confirmed = tiers.get('n_confirmed', 0)
    n_landscape = tiers.get('n_landscape_confirmed', 0)
    n_ambig = tiers.get('n_ambiguous', 0)
    n_changes = canonical.get('n_changes_from_p15', 0)
    can_dh = canonical.get('dict_hit', 0.0)
    budget = impact.get('ambiguity_budget', 0.0)
    budget_gate = impact.get('gate_label', 'UNKNOWN')

    # ── Validation Battery (V1–V8) ──
    v1 = len(profiles.get('profiles', [])) >= 6
    v2 = positional_sig
    v3 = best_ari > 0.3
    v4 = signal_pattern
    v5 = hybrid_sel > 1.5
    v6 = land_differs
    v7 = len(canonical.get('table', {})) == 25
    v8 = budget > 0  # ambiguity budget computed

    validations = {
        'V1_profiles_complete': v1,
        'V2_positional_significant': v2,
        'V3_best_labeling_ari': v3,
        'V4_signal_word_pattern': v4,
        'V5_hybrid_selectivity': v5,
        'V6_landscape_differs': v6,
        'V7_canonical_table': v7,
        'V8_ambiguity_budget': v8,
    }
    n_passed = sum(validations.values())
    gate = n_passed >= 6

    print(f"\n  Track A: community_encoding={community_encoding}, "
          f"best_labeling={best_labeling} (ARI={best_ari:.4f})")
    print(f"  Track B: hybrid_best={hybrid_best}, dict_hit={hybrid_dh:.4f}, "
          f"selectivity={hybrid_sel:.2f}×")
    print(f"  Track C: {n_confirmed}+{n_landscape}+{n_ambig} tiers, "
          f"{n_changes} changes, dict_hit={can_dh:.4f}, "
          f"budget={budget:.4f}")

    # ── Verdict ──
    hybrid_improves = (hybrid_best != 'HYBRID_NONE'
                       and hybrid_dh > hybrid.get('baseline_dict_hit', 0) * 1.01)
    hybrid_c_wins = hybrid_best == 'HYBRID_C' and hybrid_improves
    hybrid_v_wins = hybrid_best == 'HYBRID_V' and hybrid_improves

    if community_encoding == 'POSITIONAL' and not hybrid_improves:
        verdict = 'POSITIONAL_STRUCTURE'
        rationale = ("Communities encode positional roles (initial/medial/final), "
                     "not phonological categories. SBM structure is syntactic.")
    elif community_encoding == 'MORPHOLOGICAL' and not hybrid_improves:
        verdict = 'MORPHOLOGICAL_STRUCTURE'
        rationale = ("Communities correspond to morphological roles, "
                     "confirming Phase 31 compound model from independent angle.")
    elif hybrid_c_wins and budget > 0.02:
        verdict = 'CV_REDISCOVERED'
        rationale = ("Communities capture CONSONANT classes. "
                     "HYBRID_C improves decoding — re-do CSP with community constraints.")
    elif hybrid_v_wins and budget > 0.02:
        verdict = 'CV_REDISCOVERED'
        rationale = ("Communities capture VOWEL classes. "
                     "HYBRID_V improves decoding — re-do CSP with community constraints.")
    elif (community_encoding == 'FREQUENCY_TIER'
          or (best_labeling == 'frequency_tier' and not hybrid_improves)):
        verdict = 'FREQUENCY_ARTIFACT'
        rationale = (f"Communities are frequency tiers (ARI={best_ari:.3f}) — "
                     "epiphenomenal. High-frequency chars cluster together.")
    elif community_encoding == 'UNDETERMINED' and not hybrid_improves:
        verdict = 'NOVEL_STRUCTURE'
        rationale = ("Communities capture an unknown encoding dimension. "
                     "Not positional, morphological, or phonological.")
    else:
        verdict = 'CONFIRMED_CEILING'
        rationale = (f"Community structure is {community_encoding} but does not "
                     f"improve decoding. Table stable at {can_dh:.1%}.")

    # ── Progression table ──
    progression = [
        {'phase': 'Phase 11', 'dict_hit': 0.111, 'selectivity': 1.92},
        {'phase': 'Phase 14', 'dict_hit': 0.194, 'selectivity': 3.00},
        {'phase': 'Phase 15', 'dict_hit': 0.354, 'selectivity': 2.55},
        {'phase': 'Phase 16', 'dict_hit': 0.436, 'selectivity': 3.38,
         'note': 'full corpus baseline'},
        {'phase': 'Phase 44', 'dict_hit': 0.436, 'selectivity': 0.0,
         'note': 'FLAT landscape'},
        {'phase': 'Phase 45', 'dict_hit': round(can_dh, 4),
         'selectivity': round(hybrid_sel, 2),
         'note': verdict},
    ]

    print(f"\n  Validations: {n_passed}/8 ({'PASS' if gate else 'FAIL'})")
    for k, v in validations.items():
        print(f"    {k}: {'PASS' if v else 'FAIL'}")
    print(f"\n  VERDICT: {verdict}")
    print(f"  Rationale: {rationale}")

    result = Phase45IntegrateResult(
        track_a_community_encoding=community_encoding,
        track_a_positional_significant=positional_sig,
        track_a_morphological_signal=morph_signal,
        track_a_modifier_concentrated=modifier_conc,
        track_a_best_labeling=best_labeling,
        track_a_best_ari=round(best_ari, 4),
        track_a_signal_word_pattern=signal_pattern,
        track_b_community_csp_dict_hit=round(comm_csp_dh, 4),
        track_b_community_csp_selectivity=round(comm_csp_sel, 2),
        track_b_hybrid_best_variant=hybrid_best,
        track_b_hybrid_dict_hit=round(hybrid_dh, 4),
        track_b_hybrid_selectivity=round(hybrid_sel, 2),
        track_b_landscape_shape=land_shape,
        track_b_landscape_differs=land_differs,
        track_c_n_confirmed=n_confirmed,
        track_c_n_landscape_confirmed=n_landscape,
        track_c_n_ambiguous=n_ambig,
        track_c_n_changes=n_changes,
        track_c_canonical_dict_hit=round(can_dh, 4),
        track_c_ambiguity_budget=round(budget, 4),
        track_c_budget_gate=budget_gate,
        validations=validations,
        n_validations_passed=n_passed,
        gate_passed=gate,
        phase45_verdict=verdict,
        phase45_rationale=rationale,
        progression=progression,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'phase45_integrate.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2)
    print(f"\n  Saved -> {out_path}")
