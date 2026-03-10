"""
Step 35.8 – Combined Readability Battery
==========================================
Full 12-test readability battery with cross-phase progression table
for the combined spatial+10K model.

Dependency chain:
    combined_decode.json       (Step 35.2)
    combined_signal.json       (Step 35.3)
    combined_bigrams.json      (Step 35.4)
    combined_context.json      (Step 35.5)
    combined_bootstrap.json    (Step 35.6)
    combined_folio.json        (Step 35.7)
    spatial_preprocess.json    (Step 35.1)
    signal_bigrams.json        (Phase 29 baseline)
    dict_calibration.json      (Track G baseline)
        → combined_readability.json (this step)
"""

import json
import math
import os
import time
from collections import Counter
from typing import Any, Dict, List, Optional

from voynich.core._paths import results_dir as _results_dir


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(rd: str, filename: str) -> Optional[Dict]:
    path = os.path.join(rd, filename)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_combined_readability() -> None:
    """Step 35.8: Combined spatial+10K readability battery."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 35.8: Combined Readability Battery")
    print("=" * 70)

    rd = _results_dir()

    # ── Load all inputs ──
    print("\n  Loading all inputs ...")
    cd = _load_json(rd, 'combined_decode.json')
    cs = _load_json(rd, 'combined_signal.json')
    cb = _load_json(rd, 'combined_bigrams.json')
    cc = _load_json(rd, 'combined_context.json')
    cboot = _load_json(rd, 'combined_bootstrap.json')
    cfolio = _load_json(rd, 'combined_folio.json')
    sp = _load_json(rd, 'spatial_preprocess.json')
    p29 = _load_json(rd, 'signal_bigrams.json')
    dcal = _load_json(rd, 'dict_calibration.json')

    validations = []

    # V1: dict_hit_rate_10k >= 0.20
    dict_hit = cd.get('dict_hit_rate_10k', 0) if cd else 0
    v1 = dict_hit >= 0.20
    validations.append({
        'id': 'V1', 'name': 'dict_hit_10k >= 0.20',
        'value': round(dict_hit, 4), 'threshold': 0.20,
        'passed': v1,
    })

    # V2: Bigram JSD < 0.80 (approximate from z-score)
    bigram_z = cb.get('bigram_z_score', 0) if cb else 0
    jsd_approx = max(0, 1.0 - bigram_z / 20.0)
    v2 = jsd_approx < 0.80
    validations.append({
        'id': 'V2', 'name': 'bigram_jsd_approx < 0.80',
        'value': round(jsd_approx, 4), 'threshold': 0.80,
        'passed': v2,
    })

    # V3: Section chi-squared > 3.84
    if cd:
        folios = cd['token_folios']
        dict_hits = cd['token_dict_hits_10k']
        section_hits: Dict[str, List[bool]] = {}
        for f, h in zip(folios, dict_hits):
            sec = f[:2]
            section_hits.setdefault(sec, []).append(h)
        section_rates = {s: sum(h) / len(h) for s, h in section_hits.items() if h}
        overall = sum(dict_hits) / len(dict_hits) if dict_hits else 0
        denom = max(overall * (1 - overall), 0.001)
        chi_sq = sum(
            len(section_hits[s]) * (r - overall) ** 2 / denom
            for s, r in section_rates.items()
        )
    else:
        chi_sq = 0.0
    v3 = chi_sq > 3.84
    validations.append({
        'id': 'V3', 'name': 'section_chi_sq > 3.84',
        'value': round(chi_sq, 2), 'threshold': 3.84,
        'passed': v3,
    })

    # V4: Signal sigma mean >= 2.0
    word_signals = cs.get('word_signals', []) if cs else []
    genuine_sigmas = [ws['sigma'] for ws in word_signals if ws.get('is_genuine')]
    sigma_mean = sum(genuine_sigmas) / len(genuine_sigmas) if genuine_sigmas else 0
    v4 = sigma_mean >= 2.0
    validations.append({
        'id': 'V4', 'name': 'signal_sigma_mean >= 2.0',
        'value': round(sigma_mean, 2), 'threshold': 2.0,
        'passed': v4,
    })

    # V5: n_genuine_signals >= 8
    n_genuine = cs.get('n_genuine_signals', 0) if cs else 0
    v5 = n_genuine >= 8
    validations.append({
        'id': 'V5', 'name': 'n_genuine >= 8',
        'value': n_genuine, 'threshold': 8,
        'passed': v5,
    })

    # V6: Longest SIGNAL run > 4
    longest_run = 0
    if cfolio:
        for af in cfolio.get('annotated_folios', []):
            for sr in af.get('signal_runs', []):
                longest_run = max(longest_run, sr.get('length', 0))
    v6 = longest_run > 4
    validations.append({
        'id': 'V6', 'name': 'longest_run > 4',
        'value': longest_run, 'threshold': 4,
        'passed': v6,
    })

    # V7: Conditioning rate 0.30–0.60
    cond_rate = sp.get('conditioning_rate', 0) if sp else 0
    v7 = 0.30 <= cond_rate <= 0.60
    validations.append({
        'id': 'V7', 'name': 'conditioning_rate 0.30-0.60',
        'value': round(cond_rate, 3), 'threshold': '0.30-0.60',
        'passed': v7,
    })

    # V8: Bigram z >= 6.14 (Phase 29 baseline)
    v8 = bigram_z >= 6.14
    validations.append({
        'id': 'V8', 'name': 'bigram_z >= 6.14',
        'value': round(bigram_z, 2), 'threshold': 6.14,
        'passed': v8,
    })

    # V9: No regression vs Phase 29 z
    p29_z = p29.get('bigram_z_score', 0) if p29 else 0
    delta_z = bigram_z - p29_z
    v9 = delta_z >= -0.5
    validations.append({
        'id': 'V9', 'name': 'no_regression (delta_z >= -0.5)',
        'value': round(delta_z, 2), 'threshold': -0.5,
        'passed': v9,
    })

    # V10: Selectivity_10k > 1.3
    selectivity = cd.get('selectivity_10k', 0) if cd else 0
    v10 = selectivity > 1.3
    validations.append({
        'id': 'V10', 'name': 'selectivity_10k > 1.3',
        'value': round(selectivity, 2) if selectivity != 999.0 else 999.0,
        'threshold': 1.3,
        'passed': v10,
    })

    # V11: SIGNAL rate > Track E (0.274)
    signal_rate = cs.get('signal_rate', 0) if cs else 0
    v11 = signal_rate > 0.274
    validations.append({
        'id': 'V11', 'name': 'signal_rate > 0.274 (Track E)',
        'value': round(signal_rate, 4), 'threshold': 0.274,
        'passed': v11,
    })

    # V12: Bootstrap cascade >= 1
    boot_accepted = cboot.get('n_total_accepted', 0) if cboot else 0
    v12 = boot_accepted >= 1
    validations.append({
        'id': 'V12', 'name': 'bootstrap_cascade >= 1',
        'value': boot_accepted, 'threshold': 1,
        'passed': v12,
    })

    # ── Summary ──
    n_passed = sum(1 for v in validations if v['passed'])
    n_total = len(validations)

    print(f"\n  Readability battery: {n_passed}/{n_total} passed\n")
    for v in validations:
        flag = "PASS" if v['passed'] else "FAIL"
        print(f"     [{flag}] {v['id']:3s} {v['name']:35s} "
              f"value={v['value']}, threshold={v['threshold']}")

    # ── Cross-phase progression ──
    n_confirmed = cboot.get('confirmed_vocabulary_size', 0) if cboot else 0
    track_g_z = dcal.get('optimal_bigram_z', 13.12) if dcal else 13.12
    track_g_signal = dcal.get('optimal_signal_rate', 0.187) if dcal else 0.187

    progression = [
        {'phase': 14, 'dict_hit': 0.194, 'dict_size': '17K',
         'signal': None, 'bigram_z': None, 'confirmed': None,
         'model': 'Stroke-triple'},
        {'phase': 16, 'dict_hit': 0.436, 'dict_size': '131K',
         'signal': None, 'bigram_z': None, 'confirmed': None,
         'model': 'Modifier handling'},
        {'phase': 28, 'dict_hit': 0.436, 'dict_size': '131K',
         'signal': 0.165, 'bigram_z': None, 'confirmed': 8,
         'model': 'Signal isolation'},
        {'phase': 29, 'dict_hit': 0.436, 'dict_size': '131K',
         'signal': 0.165, 'bigram_z': 6.14, 'confirmed': 8,
         'model': 'Bigram discovery'},
        {'phase': 30, 'dict_hit': 0.436, 'dict_size': '131K',
         'signal': 0.165, 'bigram_z': 6.14, 'confirmed': 10,
         'model': 'Bootstrap (+2)'},
        {'phase': '34G', 'dict_hit': 0.227, 'dict_size': '10K',
         'signal': round(track_g_signal, 3), 'bigram_z': round(track_g_z, 2),
         'confirmed': 10, 'model': 'Dict right-sizing'},
        {'phase': '34E', 'dict_hit': 0.436, 'dict_size': '131K',
         'signal': 0.274, 'bigram_z': None, 'confirmed': 10,
         'model': 'Spatial conditioning'},
        {'phase': 35, 'dict_hit': round(dict_hit, 3), 'dict_size': '10K',
         'signal': round(signal_rate, 3),
         'bigram_z': round(bigram_z, 2),
         'confirmed': n_confirmed,
         'model': 'Combined E+G'},
    ]

    print("\n  Cross-phase progression:")
    print(f"  {'Phase':>5s} | {'dict_hit':>8s} | {'dict':>5s} | {'SIGNAL':>7s} | "
          f"{'bigram_z':>8s} | {'conf':>5s} | Model")
    print("  " + "-" * 72)
    for p in progression:
        dh = f"{p['dict_hit']:.3f}" if p['dict_hit'] is not None else "  —"
        sr = f"{p['signal']:.3f}" if p['signal'] is not None else "  —"
        bz = f"{p['bigram_z']:.2f}" if p['bigram_z'] is not None else "  —"
        cf = f"{p['confirmed']}" if p['confirmed'] is not None else "  —"
        print(f"  {str(p['phase']):>5s} | {dh:>8s} | {p['dict_size']:>5s} | {sr:>7s} | "
              f"{bz:>8s} | {cf:>5s} | {p['model']}")

    # ── Save ──
    output = {
        'validations': validations,
        'n_passed': n_passed,
        'n_total': n_total,
        'progression': progression,
        'gate_passed': n_passed >= 8,
        'verdict': f"{n_passed}/{n_total} validations passed",
        'runtime_seconds': round(time.time() - t0, 1),
    }

    with open(os.path.join(rd, 'combined_readability.json'), 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n  Step 35.8 completed in {time.time() - t0:.1f}s")
