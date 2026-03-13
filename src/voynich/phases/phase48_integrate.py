"""
Phase 48 Integration: Marginal Bilingual Crib Exploitation
============================================================
Compile results from all four tracks (A-D) and produce final verdict.

Dependency chain:
    f116v_transcription.json   (48A.1)
    f116v_decode.json          (48A.2)
    f116v_context.json         (48A.3)
    f116v_match.json           (48A.4)
    f116v_reverse.json         (48A.5)
    f17r_extract.json          (48B.1)
    f66r_extract.json          (48B.2)
    margin_decode.json         (48B.3)
    margin_hands.json          (48B.4)
    marci_source.json          (48C.1)
    marci_correspondences.json (48C.2)
    marci_comparison.json      (48C.3)
    marci_test.json            (48C.4)
    crib_collection.json       (48D.1)
    crib_consistency.json      (48D.2)
    crib_propagation.json      (48D.3)
    crib_decode.json           (48D.4)
    crib_validation.json       (48D.5)
        → phase48_integrate.json
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
    path = os.path.join(rd, filename)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _save_json(rd: str, filename: str, data: Any) -> str:
    path = os.path.join(rd, filename)
    with open(path, 'w') as f:
        json.dump(_convert(data), f, indent=2)
    return path


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    """One V-test result."""
    test_id: str
    description: str
    metric: str
    threshold: str
    value: str
    passed: bool


@dataclass
class Phase48IntegrateResult:
    """Full Phase 48 integration output."""
    track_a_summary: Dict
    track_b_summary: Dict
    track_c_summary: Dict
    track_d_summary: Dict
    validations: List[Dict]
    n_validations_passed: int
    n_validations_total: int
    gate_passed: bool
    phase48_verdict: str
    progression_table: List[Dict]
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_phase48_integrate() -> None:
    """Phase 48 Integration: compile all tracks and produce verdict."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE 48 INTEGRATION: Marginal Bilingual Crib Exploitation")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load all results ──
    print("\n  1. Loading all Phase 48 results...")

    # Track A
    a1 = _load_json(rd, 'f116v_transcription.json')
    a2 = _load_json(rd, 'f116v_decode.json')
    a3 = _load_json(rd, 'f116v_context.json')
    a4 = _load_json(rd, 'f116v_match.json')
    a5 = _load_json(rd, 'f116v_reverse.json')

    # Track B
    b1 = _load_json(rd, 'f17r_extract.json')
    b2 = _load_json(rd, 'f66r_extract.json')
    b3 = _load_json(rd, 'margin_decode.json')
    b4 = _load_json(rd, 'margin_hands.json')

    # Track C
    c1 = _load_json(rd, 'marci_source.json')
    c2 = _load_json(rd, 'marci_correspondences.json')
    c3 = _load_json(rd, 'marci_comparison.json')
    c4 = _load_json(rd, 'marci_test.json')

    # Track D
    d1 = _load_json(rd, 'crib_collection.json')
    d2 = _load_json(rd, 'crib_consistency.json')
    d3 = _load_json(rd, 'crib_propagation.json')
    d4 = _load_json(rd, 'crib_decode.json')
    d5 = _load_json(rd, 'crib_validation.json')

    loaded = sum(1 for x in [a1, a2, a3, a4, a5, b1, b2, b3, b4,
                              c1, c2, c3, c4, d1, d2, d3, d4, d5] if x)
    print(f"     Loaded {loaded}/18 result files")

    # ── 2. Track A Summary ──
    print("\n  2. Track A: f116v Voynichese Decode")

    f116v_match_level = a4.get('best_match_level', 'NOT_RUN') if a4 else 'NOT_RUN'
    f116v_gate = a4.get('gate_result', 'NOT_RUN') if a4 else 'NOT_RUN'
    f116v_best_word = a4.get('best_match_word', '') if a4 else ''
    f116v_n_testable = a5.get('n_testable', 0) if a5 else 0
    n_words_decoded = len(a2.get('primary_decodes', [])) if a2 else 0

    track_a = {
        'n_words_decoded': n_words_decoded,
        'best_match_level': f116v_match_level,
        'best_match_word': f116v_best_word,
        'gate': f116v_gate,
        'n_reverse_testable': f116v_n_testable,
        'n_dict_hits_10k': a2.get('n_dict_hits_10k', 0) if a2 else 0,
        'n_dict_hits_131k': a2.get('n_dict_hits_131k', 0) if a2 else 0,
        'n_context_readings': a3.get('n_readings', 0) if a3 else 0,
    }

    print(f"     Words decoded: {n_words_decoded}")
    print(f"     Best match: {f116v_match_level} ('{f116v_best_word}')")
    print(f"     Gate: {f116v_gate}")
    print(f"     Reverse-testable candidates: {f116v_n_testable}")

    # ── 3. Track B Summary ──
    print("\n  3. Track B: f17r/f66r Marginal Notes")

    f17r_has_voynich = b1.get('has_voynichese_marginal', False) if b1 else False
    f66r_has_voynich = b2.get('has_voynichese', False) if b2 else False
    margin_hits = b3.get('n_dict_hits_131k', 0) if b3 else 0

    track_b = {
        'f17r_has_voynichese_marginal': f17r_has_voynich,
        'f17r_marginal_text': b1.get('marginal_text', '') if b1 else '',
        'f66r_has_voynichese': f66r_has_voynich,
        'margin_dict_hits_131k': margin_hits,
        'same_hand_assessment': b4.get('same_hand_assessment', 'NOT_RUN') if b4 else 'NOT_RUN',
        'n_dialect_folios': len(b4.get('dialect_evidence', [])) if b4 else 0,
    }

    print(f"     f17r Voynichese marginal: {f17r_has_voynich}")
    print(f"     f66r Voynichese: {f66r_has_voynich}")
    print(f"     Marginal dict hits (131K): {margin_hits}")

    # ── 4. Track C Summary ──
    print("\n  4. Track C: Marci Annotations")

    marci_available = c1.get('data_available', False) if c1 else False
    marci_quality = c2.get('data_quality', 'NOT_RUN') if c2 else 'NOT_RUN'
    marci_ari = c3.get('consonant_ari', 0.0) if c3 else 0.0
    marci_perf = c4.get('performance_class', 'NOT_RUN') if c4 else 'NOT_RUN'

    track_c = {
        'data_available': marci_available,
        'data_quality': marci_quality,
        'consonant_ari': marci_ari,
        'performance_class': marci_perf,
        'note': 'No machine-readable Marci transcription available' if not marci_available
                else f'ARI={marci_ari:.4f}, Performance={marci_perf}',
    }

    print(f"     Data available: {marci_available}")
    print(f"     Data quality: {marci_quality}")
    print(f"     Consonant ARI: {marci_ari}")
    print(f"     Performance: {marci_perf}")

    # ── 5. Track D Summary ──
    print("\n  5. Track D: Crib Propagation")

    n_cribs = d1.get('n_total', 0) if d1 else 0
    n_accepted = d3.get('n_accepted', 0) if d3 else 0
    crib_delta = d4.get('delta_dict_hit', 0.0) if d4 else 0.0
    crib_z = d5.get('bigram_z', 0.0) if d5 else 0.0
    crib_gate = d5.get('gate_result', 'NOT_RUN') if d5 else 'NOT_RUN'

    track_d = {
        'n_cribs_collected': n_cribs,
        'n_propagations_accepted': n_accepted,
        'dict_hit_delta': crib_delta,
        'bigram_z': crib_z,
        'validation_gate': crib_gate,
    }

    print(f"     Cribs collected: {n_cribs}")
    print(f"     Propagations accepted: {n_accepted}")
    print(f"     Dict-hit delta: {crib_delta:+.4f}")
    print(f"     Bigram z: {crib_z:.2f}")
    print(f"     Validation gate: {crib_gate}")

    # ── 6. Validation Battery (V1-V8) ──
    print("\n  6. Validation Battery:")

    validations = []

    # V1: f116v words extracted and decoded (≥2)
    v1 = n_words_decoded >= 2
    validations.append(asdict(ValidationResult(
        test_id='V1', description='f116v Voynichese words extracted and decoded',
        metric='n_words_decoded', threshold='≥2',
        value=str(n_words_decoded), passed=v1,
    )))

    # V2: All competing readings compiled (≥3)
    n_readings = a3.get('n_readings', 0) if a3 else 0
    v2 = n_readings >= 3
    validations.append(asdict(ValidationResult(
        test_id='V2', description='All competing readings compiled',
        metric='n_readings', threshold='≥3',
        value=str(n_readings), passed=v2,
    )))

    # V3: At least 1 STRONG or PARTIAL match
    v3 = f116v_match_level in ('STRONG_MATCH', 'PARTIAL_MATCH')
    validations.append(asdict(ValidationResult(
        test_id='V3', description='At least 1 STRONG or PARTIAL match',
        metric='best_match_level', threshold='STRONG or PARTIAL',
        value=f116v_match_level, passed=v3,
    )))

    # V4: f17r/f66r marginal content extracted
    v4 = b1 is not None and b2 is not None
    validations.append(asdict(ValidationResult(
        test_id='V4', description='f17r/f66r marginal content extracted',
        metric='content_extracted', threshold='both extracted',
        value=f'f17r={b1 is not None}, f66r={b2 is not None}', passed=v4,
    )))

    # V5: Marci source data located
    v5 = c1 is not None
    validations.append(asdict(ValidationResult(
        test_id='V5', description='Marci source data located',
        metric='data_found', threshold='found or documented as unavailable',
        value=f'available={marci_available}, documented={c1 is not None}', passed=v5,
    )))

    # V6: Crib consistency checked
    v6 = d2 is not None
    validations.append(asdict(ValidationResult(
        test_id='V6', description='Crib consistency checked',
        metric='matrix_computed', threshold='computed',
        value=str(d2 is not None), passed=v6,
    )))

    # V7: Signal words survive any table changes
    signal_surviving = d5.get('signal_words_surviving', 0) if d5 else 0
    signal_total = d5.get('signal_words_total', 8) if d5 else 8
    v7 = signal_surviving >= signal_total or n_accepted == 0  # Pass if no changes made
    validations.append(asdict(ValidationResult(
        test_id='V7', description='Signal words survive any table changes',
        metric='signal_survival', threshold='8/8',
        value=f'{signal_surviving}/{signal_total}', passed=v7,
    )))

    # V8: Bigram z >= Phase 47 canonical z
    phase47 = _load_json(rd, 'phase47_integrate.json')
    canonical_z = phase47.get('track_a_canonical_z', 14.78) if phase47 else 14.78
    v8 = crib_z >= canonical_z or n_accepted == 0  # Pass if no changes made
    validations.append(asdict(ValidationResult(
        test_id='V8', description='Bigram z >= Phase 47 canonical z',
        metric='bigram_z', threshold=f'≥{canonical_z:.2f}',
        value=f'{crib_z:.2f}', passed=v8,
    )))

    n_pass = sum(1 for v in validations if v['passed'])
    n_total = len(validations)

    for v in validations:
        marker = '✓' if v['passed'] else '✗'
        print(f"     {marker} {v['test_id']}: {v['description']} — {v['value']} {'PASS' if v['passed'] else 'FAIL'}")

    print(f"\n     Result: {n_pass}/{n_total}")

    # ── 7. Gate ──
    gate_passed = n_pass >= 6
    print(f"\n  7. Gate (≥6/8): {'PASS' if gate_passed else 'FAIL'}")

    # ── 8. Verdict ──
    print("\n  8. Phase 48 Verdict:")

    # Decision table from README
    if (f116v_match_level == 'STRONG_MATCH' and marci_ari > 0.3
            and crib_gate == 'ACCEPTED' and crib_z > canonical_z):
        verdict = 'BILINGUAL_BREAKTHROUGH'
    elif f116v_match_level == 'STRONG_MATCH' and crib_gate == 'ACCEPTED':
        verdict = 'CRIB_CONFIRMED'
    elif f116v_match_level == 'PARTIAL_MATCH' and marci_ari > 0.3 and crib_gate == 'ACCEPTED':
        verdict = 'CONVERGENT_EVIDENCE'
    elif f116v_match_level == 'PARTIAL_MATCH':
        verdict = 'CRIB_SUGGESTIVE'
    elif marci_ari > 0.5 and crib_gate == 'ACCEPTED':
        verdict = 'MARCI_VALIDATED'
    elif f116v_match_level == 'NO_MATCH' and marci_quality == 'UNAVAILABLE':
        verdict = 'MARGINAL_UNINFORMATIVE'
    elif f116v_match_level in ('WEAK_MATCH', 'NO_MATCH'):
        verdict = 'MARGINAL_UNINFORMATIVE'
    else:
        verdict = 'CRIB_SUGGESTIVE'

    print(f"     Verdict: {verdict}")

    # ── 9. Progression table ──
    print("\n  9. Progression table:")

    progression = [
        {'phase': '16', 'dict_hit': '43.6%', 'signal': '16.5%',
         'bigram_z': '—', 'confirmed': '12/25', 'notes': 'Full corpus baseline'},
        {'phase': '28', 'dict_hit': '43.6%', 'signal': '16.5%',
         'bigram_z': '6.14', 'confirmed': '12/25', 'notes': 'Crib extraction'},
        {'phase': '47', 'dict_hit': '43.6%', 'signal': '—',
         'bigram_z': f'{canonical_z:.1f}', 'confirmed': '22/25',
         'notes': 'Z-score audit canonical'},
        {'phase': '48', 'dict_hit': f'{(0.436 + crib_delta):.1%}',
         'signal': '—', 'bigram_z': f'{crib_z:.1f}' if crib_z > 0 else '—',
         'confirmed': f'{22 + n_accepted}/25',
         'notes': f'{verdict} ({n_pass}/{n_total} validations)'},
    ]

    print(f"     {'Phase':>6s} | {'Dict-hit':>9s} | {'Signal':>8s} | "
          f"{'Bigram z':>8s} | {'Confirmed':>9s} | Notes")
    print(f"     {'-' * 6} | {'-' * 9} | {'-' * 8} | {'-' * 8} | {'-' * 9} | {'-' * 30}")
    for p in progression:
        print(f"     {p['phase']:>6s} | {p['dict_hit']:>9s} | {p['signal']:>8s} | "
              f"{p['bigram_z']:>8s} | {p['confirmed']:>9s} | {p['notes']}")

    # ── 10. Save ──
    result = Phase48IntegrateResult(
        track_a_summary=track_a,
        track_b_summary=track_b,
        track_c_summary=track_c,
        track_d_summary=track_d,
        validations=validations,
        n_validations_passed=n_pass,
        n_validations_total=n_total,
        gate_passed=gate_passed,
        phase48_verdict=verdict,
        progression_table=progression,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out = _save_json(rd, 'phase48_integrate.json', asdict(result))
    print(f"\n  Saved → {out}")
    print(f"  Completed in {time.time() - t0:.1f}s")


# ---------------------------------------------------------------------------
# Full Phase 48 runner
# ---------------------------------------------------------------------------

def run_phase48() -> None:
    """Run all Phase 48 tracks and integration."""
    from voynich.phases.marginal_cribs import run_track_a_48
    from voynich.phases.marginal_secondary import run_track_b_48
    from voynich.phases.marci_annotations import run_track_c_48
    from voynich.phases.bilingual_propagation import run_track_d_48

    print("\n" + "█" * 70)
    print("  PHASE 48: Marginal Bilingual Crib Exploitation")
    print("█" * 70)

    run_track_a_48()
    run_track_b_48()
    run_track_c_48()
    run_track_d_48()
    run_phase48_integrate()

    print("\n" + "█" * 70)
    print("  PHASE 48 COMPLETE")
    print("█" * 70)
