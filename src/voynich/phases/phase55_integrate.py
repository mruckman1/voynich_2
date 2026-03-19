"""
Phase 55 Integration – Entropy Shift Generalization + Currier Prediction
=========================================================================
Combines Track A (extended entropy shift ranking) and Track B (Currier
self-correlation prediction) into a single verdict.

Dependency chain:
    results/phase55_entropy_extended.json   (Track A)
    results/phase55_currier_verdict.json    (Track B)
        → results/phase55_integrate.json

Full pipeline runner: run_phase55()
"""

import json
import os
import time
from dataclasses import asdict, dataclass
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


def _save_json(rd: str, filename: str, data: Any) -> str:
    path = os.path.join(rd, filename)
    with open(path, 'w') as f:
        json.dump(_convert(data), f, indent=2)
    return path


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------

def run_phase55_integrate() -> None:
    """Phase 55 Integration: combine Track A + Track B verdicts."""
    t0 = time.time()
    rd = str(_results_dir())

    print("=" * 70)
    print("PHASE 55 INTEGRATION: Entropy Shift + Currier Prediction")
    print("=" * 70)

    # ── Load tracks ──────────────────────────────────────────────────────
    print("\n  Loading track results …")

    entropy_data = _safe_load(os.path.join(rd, 'phase55_entropy_extended.json'))
    currier_data = _safe_load(os.path.join(rd, 'phase55_currier_verdict.json'))

    if not entropy_data:
        raise FileNotFoundError(
            "phase55_entropy_extended.json not found — run entropy-extended first"
        )
    if not currier_data:
        raise FileNotFoundError(
            "phase55_currier_verdict.json not found — run currier-verdict first"
        )

    track_a_verdict = entropy_data.get('verdict', 'UNKNOWN')
    track_b_verdict = currier_data.get('verdict', 'UNKNOWN')

    print(f"    Track A verdict: {track_a_verdict}")
    print(f"    Track B verdict: {track_b_verdict}")

    # ── Extract key values ────────────────────────────────────────────────
    tachy_cosine = entropy_data.get('tachygraphy_cosine', 0.0)
    tachy_unique = entropy_data.get('tachygraphy_uniquely_positive', False)
    schinner_above = entropy_data.get('schinner_above_tachygraphy', False)
    new_mechs = entropy_data.get('new_mechanisms', {})

    voynich_ratio = currier_data.get('ratios', {}).get('voynich', 0.0)
    tachy_syl_ratio = currier_data.get('ratios', {}).get('tachy_syllable', 0.0)
    prediction_match = currier_data.get('prediction_match', False)
    schinner_reproduces = currier_data.get('schinner_reproduces', False)

    # ── Validation battery V1–V6 ─────────────────────────────────────────
    print("\n  Validation battery V1–V6 …")

    track_a_gates = entropy_data.get('gates', {})
    track_b_gates = currier_data.get('gates', {})

    schinner_simple = new_mechs.get('schinner_simple', {})
    schinner_pos = new_mechs.get('schinner_positional', {})
    cardan_3 = new_mechs.get('cardan_3hole', {})
    cardan_4 = new_mechs.get('cardan_4hole', {})

    # V1: Schinner CI below tachygraphy CI (strictly below, not above)
    # G1/G2 now check ci_upper(schinner) < ci_lower(tachy)
    v1 = track_a_gates.get('G1', False) and track_a_gates.get('G2', False)
    print(f"    V1 Schinner CI strictly below tachygraphy: {'PASS' if v1 else 'FAIL'}")

    # V2: Cardan CI below tachygraphy CI (non-overlapping)
    v2 = track_a_gates.get('G3', False) and track_a_gates.get('G4', False)
    print(f"    V2 Cardan CI below tachygraphy:   {'PASS' if v2 else 'FAIL'}")

    # V3: Tachygraphy remains rank 1 (highest cosine)
    v3 = track_a_gates.get('G5', False)
    print(f"    V3 Tachygraphy rank 1:             {'PASS' if v3 else 'FAIL'}")

    # V4: Voynich self-correlation > 2.5× (Currier confirmed)
    v4 = track_b_gates.get('G1', False)
    print(f"    V4 Voynich ratio > 2.5×:           {'PASS' if v4 else 'FAIL'} "
          f"({voynich_ratio:.4f}×)")

    # V5: Tachygraphic prediction within 30% of Voynich
    v5 = track_b_gates.get('G3', False)
    print(f"    V5 Tachy prediction matches:       {'PASS' if v5 else 'FAIL'} "
          f"({tachy_syl_ratio:.4f}× vs {voynich_ratio:.4f}×)")

    # V6: Syllable-as-token > word-as-token (mechanism confirmed)
    v6 = track_b_gates.get('G4', False)
    print(f"    V6 Syl-token > word-token:         {'PASS' if v6 else 'FAIL'}")

    validations = {
        'V1_schinner_below_tachy': v1,
        'V2_cardan_below_tachy': v2,
        'V3_tachy_rank1': v3,
        'V4_voynich_currier_confirmed': v4,
        'V5_tachy_prediction_matches': v5,
        'V6_syllable_drives_anomaly': v6,
    }
    n_passed = sum(validations.values())

    print(f"\n  Validations passed: {n_passed}/6")

    # ── Overall verdict ───────────────────────────────────────────────────
    print("\n  Overall verdict …")

    if n_passed >= 4:
        if track_b_verdict in ('PREDICTION_CONFIRMED_UNIQUE', 'PREDICTION_CONFIRMED_NOT_UNIQUE'):
            if track_a_verdict == 'SCHINNER_ABOVE_TACHYGRAPHY':
                verdict = 'TRACK_B_CONFIRMED_SCHINNER_LIMITATION'
            else:
                verdict = 'BOTH_CONFIRMED'
        else:
            verdict = 'TRACK_B_CONFIRMED_TRACK_A_PARTIAL'
    elif n_passed >= 2:
        verdict = 'PARTIAL'
    else:
        verdict = 'INSUFFICIENT'

    gate_passed = n_passed >= 4

    print(f"  VERDICT: {verdict}")
    print(f"  Gate:    {'PASS' if gate_passed else 'FAIL'}")

    # ── Paper impact summary ──────────────────────────────────────────────
    tachy_rank = entropy_data.get('tachygraphy_rank', None)
    rank_str = f"rank {tachy_rank}" if tachy_rank else "rank displaced"
    track_a_summary = (
        f"Tachygraphy {rank_str} ({tachy_cosine:+.3f}). "
        f"Schinner: {schinner_simple.get('cosine', 0):.3f}/{schinner_pos.get('cosine', 0):.3f} "
        f"({'ABOVE' if schinner_above else 'below'} tachy). "
        f"Cardan: {cardan_3.get('cosine', 0):.3f}/{cardan_4.get('cosine', 0):.3f}. "
        f"Tachy uniquely positive: {tachy_unique}."
    )
    track_b_summary = (
        f"Voynich ratio={voynich_ratio:.3f}×. "
        f"Tachy (syl)={tachy_syl_ratio:.3f}×. "
        f"Prediction match: {prediction_match}. "
        f"Schinner reproduces: {schinner_reproduces}."
    )

    print(f"\n  Track A: {track_a_summary}")
    print(f"  Track B: {track_b_summary}")

    runtime = round(time.time() - t0, 2)
    output = {
        'phase': '55',
        'experiment': 'phase55_integrate',
        'track_a': {
            'verdict': track_a_verdict,
            'tachygraphy_cosine': tachy_cosine,
            'tachygraphy_rank': entropy_data.get('tachygraphy_rank', None),
            'tachygraphy_uniquely_positive': tachy_unique,
            'schinner_above_tachygraphy': schinner_above,
            'new_mechanisms': new_mechs,
            'gates': track_a_gates,
            'summary': track_a_summary,
        },
        'track_b': {
            'verdict': track_b_verdict,
            'voynich_ratio': voynich_ratio,
            'tachy_syllable_ratio': tachy_syl_ratio,
            'prediction_match': prediction_match,
            'schinner_reproduces': schinner_reproduces,
            'gates': track_b_gates,
            'summary': track_b_summary,
        },
        'validations': validations,
        'n_validations_passed': n_passed,
        'n_validations_total': 6,
        'verdict': verdict,
        'gate_passed': gate_passed,
        'runtime_seconds': runtime,
    }

    out_path = _save_json(rd, 'phase55_integrate.json', output)
    print(f"\n  Saved → {out_path}")
    print(f"  Completed in {runtime:.1f}s")


# ---------------------------------------------------------------------------
# Full pipeline runner
# ---------------------------------------------------------------------------

def run_phase55() -> None:
    """Run full Phase 55 pipeline (all tracks + integration)."""

    from voynich.phases.schinner_generator import run_schinner_gen
    from voynich.phases.cardan_generator import run_cardan_gen
    from voynich.phases.entropy_shift_extended import run_entropy_extended
    from voynich.phases.currier_selfcorr import (
        run_currier_voynich,
        run_currier_tachy,
        run_currier_controls,
        run_currier_verdict,
    )

    run_schinner_gen()
    print()
    run_cardan_gen()
    print()
    run_entropy_extended()
    print()
    run_currier_voynich()
    print()
    run_currier_tachy()
    print()
    run_currier_controls()
    print()
    run_currier_verdict()
    print()
    run_phase55_integrate()
