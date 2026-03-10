"""
Step 35.9 – Phase 35 Verdict
==============================
Integrate all Phase 35 results and produce the final verdict on whether
combining spatial conditioning with the 10K dictionary produces
multiplicative improvement.

Dependency chain:
    spatial_preprocess.json    (Step 35.1)
    combined_decode.json       (Step 35.2)
    combined_signal.json       (Step 35.3)
    combined_bigrams.json      (Step 35.4)
    combined_context.json      (Step 35.5)
    combined_bootstrap.json    (Step 35.6)
    combined_folio.json        (Step 35.7)
    combined_readability.json  (Step 35.8)
        → phase35_verdict.json (this step)
"""

import json
import os
import time
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

def run_phase35_verdict() -> None:
    """Step 35.9: Phase 35 verdict."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 35.9: Phase 35 Verdict")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load all results ──
    print("\n  1. Loading all Phase 35 results ...")
    sp = _load_json(rd, 'spatial_preprocess.json')
    cd = _load_json(rd, 'combined_decode.json')
    cs = _load_json(rd, 'combined_signal.json')
    cb = _load_json(rd, 'combined_bigrams.json')
    cc = _load_json(rd, 'combined_context.json')
    cboot = _load_json(rd, 'combined_bootstrap.json')
    cfolio = _load_json(rd, 'combined_folio.json')
    cread = _load_json(rd, 'combined_readability.json')
    dcal = _load_json(rd, 'dict_calibration.json')

    # ── 2. Extract key metrics ──
    print("\n  2. Extracting key metrics ...")

    # Dict hit
    dict_hit_10k = cd.get('dict_hit_rate_10k', 0) if cd else 0
    dict_hit_131k = cd.get('dict_hit_rate_131k', 0) if cd else 0

    # Signal
    signal_rate = cs.get('signal_rate', 0) if cs else 0
    n_anti = cs.get('n_anti_signal', 0) if cs else 0
    n_tokens = cs.get('n_tokens', 1) if cs else 1
    net_signal = signal_rate - (n_anti / n_tokens)

    # Bigram z
    bigram_z = cb.get('bigram_z_score', 0) if cb else 0
    phase29_z = cb.get('phase29_bigram_z', 6.14) if cb else 6.14
    track_g_z = cb.get('track_g_bigram_z', 13.12) if cb else 13.12

    # Bootstrap
    n_boot = cboot.get('n_total_accepted', 0) if cboot else 0
    boot_words = cboot.get('accepted_words', []) if cboot else []
    confirmed_vocab_size = cboot.get('confirmed_vocabulary_size', 0) if cboot else 0

    # Readability
    n_validations_passed = cread.get('n_passed', 0) if cread else 0
    n_validations_total = cread.get('n_total', 12) if cread else 12

    # Spatial stats
    conditioning_rate = sp.get('conditioning_rate', 0) if sp else 0
    n_stripped = (sp.get('spatial_stats', {}).get('n_strip_preceding', 0) +
                  sp.get('spatial_stats', {}).get('n_strip_following', 0) +
                  sp.get('spatial_stats', {}).get('n_strip_mixed', 0)) if sp else 0
    n_silent = sp.get('spatial_stats', {}).get('n_standalone_silent', 0) if sp else 0
    n_intersecting = sp.get('spatial_stats', {}).get('n_intersecting_kept', 0) if sp else 0

    # Folio
    best_fragment = cfolio.get('best_fragment') if cfolio else None
    n_runs = cfolio.get('n_total_runs', 0) if cfolio else 0
    longest_run = 0
    if cfolio:
        for af in cfolio.get('annotated_folios', []):
            for sr in af.get('signal_runs', []):
                longest_run = max(longest_run, sr.get('length', 0))

    # Bigram type breakdown
    n_content_content = 0
    if cb:
        type_counts = cb.get('bigram_type_counts', {})
        n_content_content = type_counts.get('content_content', 0)

    # Context
    n_cribs = cc.get('n_new_crib_candidates', 0) if cc else 0
    n_chains = cc.get('n_chains_found', 0) if cc else 0

    print(f"     dict_hit_10k:     {dict_hit_10k:.4f}")
    print(f"     signal_rate:      {signal_rate:.4f}")
    print(f"     net_signal:       {net_signal:.4f}")
    print(f"     bigram_z:         {bigram_z:.2f}")
    print(f"     Phase 29 z:       {phase29_z:.2f}")
    print(f"     Track G z:        {track_g_z:.2f}")
    print(f"     bootstrap:        {n_boot} words accepted")
    print(f"     validations:      {n_validations_passed}/{n_validations_total}")
    print(f"     conditioning:     {conditioning_rate:.1%}")

    # ── 3. Component contributions ──
    print("\n  3. Assessing component contributions ...")

    # Track E baseline: signal_rate = 0.274 (from gallows_geometry.json)
    track_e_signal = 0.274
    # Track G baseline: signal_rate = 0.187 (from dict_calibration.json)
    track_g_signal = dcal.get('optimal_signal_rate', 0.187) if dcal else 0.187

    spatial_contribution = 'NEUTRAL'
    if signal_rate > track_g_signal + 0.05:
        spatial_contribution = 'POSITIVE'
    elif signal_rate < track_g_signal - 0.02:
        spatial_contribution = 'NEGATIVE'

    dict10k_contribution = 'NEUTRAL'
    if bigram_z > phase29_z + 2.0:
        dict10k_contribution = 'POSITIVE'
    elif bigram_z < phase29_z - 1.0:
        dict10k_contribution = 'NEGATIVE'

    print(f"     Spatial contribution: {spatial_contribution}")
    print(f"     Dict 10K contribution: {dict10k_contribution}")

    # ── 4. Verdict ──
    print("\n  4. Determining verdict ...")

    # Check predictions
    signal_exceeds_e = signal_rate > track_e_signal
    z_exceeds_g = bigram_z > track_g_z

    has_latin_phrase = False
    if best_fragment:
        ps = best_fragment.get('parse', {}).get('parse_score', 0)
        fl = best_fragment.get('length', 0)
        has_latin_phrase = ps >= 0.5 and fl >= 4

    if (signal_rate > 0.35 and bigram_z > 15 and has_latin_phrase
            and n_boot >= 5 and n_content_content > 0):
        verdict = "COMBINED_BREAKTHROUGH"
    elif (signal_rate > 0.30 and z_exceeds_g and n_boot >= 2
          and longest_run > 4):
        verdict = "COMBINED_AMPLIFICATION"
    elif signal_exceeds_e or z_exceeds_g:
        verdict = "ADDITIVE_ONLY"
    else:
        verdict = "NO_INTERACTION"

    print(f"\n  VERDICT: {verdict}")
    print(f"  Signal prediction (>{track_e_signal:.1%}): "
          f"{'PASS' if signal_exceeds_e else 'FAIL'} ({signal_rate:.1%})")
    print(f"  Bigram z prediction (>{track_g_z:.2f}): "
          f"{'PASS' if z_exceeds_g else 'FAIL'} ({bigram_z:.2f})")

    # ── 5. Gap analysis ──
    print("\n  5. Gap analysis ...")
    next_steps = []

    if verdict == "COMBINED_BREAKTHROUGH":
        next_steps.append("Publish annotated transliterations from Step 35.7")
        next_steps.append("Map determinatives to semantic domains")
        next_steps.append("Extend to full manuscript reading")
    elif verdict == "COMBINED_AMPLIFICATION":
        next_steps.append("Focus on improving triple assignments for the remaining "
                          f"{25 - confirmed_vocab_size} unconfirmed triples")
        next_steps.append("Test alternative spatial conditioning thresholds")
        next_steps.append("Try 5K dictionary (even stricter)")
    elif verdict == "ADDITIVE_ONLY":
        next_steps.append("The two tracks contribute independently but don't multiply")
        next_steps.append("Consider CVC/CCV expansion to break the CV ceiling")
        next_steps.append("Test boustrophedon + spatial conditioning")
    else:
        next_steps.append("Spatial conditioning and 10K dictionary on different axes")
        next_steps.append("Focus on the stronger individual track")
        next_steps.append("Consider entirely different encoding model")

    for step in next_steps:
        print(f"     - {step}")

    # ── 6. Save ──
    print("\n  6. Saving phase35_verdict.json ...")
    output = {
        # Key metrics
        'dict_hit_10k': round(dict_hit_10k, 4),
        'dict_hit_131k': round(dict_hit_131k, 4),
        'signal_rate': round(signal_rate, 4),
        'net_signal': round(net_signal, 4),
        'bigram_z': round(bigram_z, 2) if bigram_z != float('inf') else 999.0,
        'phase29_bigram_z': phase29_z,
        'track_g_bigram_z': track_g_z,
        'delta_vs_phase29': round(bigram_z - phase29_z, 2),
        'delta_vs_track_g': round(bigram_z - track_g_z, 2),
        # Predictions
        'signal_exceeds_track_e': signal_exceeds_e,
        'z_exceeds_track_g': z_exceeds_g,
        # Validation
        'n_validations_passed': n_validations_passed,
        'n_validations_total': n_validations_total,
        # Bootstrap
        'n_bootstrap_confirmed': n_boot,
        'bootstrap_words': boot_words,
        'confirmed_vocabulary_size': confirmed_vocab_size,
        # Spatial analysis
        'conditioning_rate': round(conditioning_rate, 4),
        'n_stripped': n_stripped,
        'n_silent': n_silent,
        'n_intersecting_kept': n_intersecting,
        # Component contributions
        'spatial_contribution': spatial_contribution,
        'dict10k_contribution': dict10k_contribution,
        # Folio analysis
        'n_signal_runs': n_runs,
        'longest_signal_run': longest_run,
        'best_fragment': best_fragment,
        'n_content_content_bigrams': n_content_content,
        # Context
        'n_crib_candidates': n_cribs,
        'n_chains': n_chains,
        # Verdict
        'verdict': verdict,
        'next_steps': next_steps,
        'runtime_seconds': round(time.time() - t0, 1),
    }

    out_path = os.path.join(rd, 'phase35_verdict.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n  Saved → {out_path}")
    print(f"  Phase 35 completed in {time.time() - t0:.1f}s")
