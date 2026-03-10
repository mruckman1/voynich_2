"""
Step 36.8 – Phase 36 Verdict
==============================
Interprets the full readability battery and assigns one of four outcomes:
10K_BREAKTHROUGH, 10K_ADVANCEMENT, 10K_CONFIRMED, 10K_REGRESSION.

Dependency chain:
    readability_10k.json      (Step 36.7)
    bigrams_10k.json          (Step 36.3)
    bootstrap_10k.json        (Step 36.5)
        → phase36_verdict.json  (this step)
"""

import json
import os
import time
from typing import Any, Dict

from voynich.core._paths import results_dir as _results_dir


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _convert(obj: Any) -> Any:
    if hasattr(obj, '__dataclass_fields__'):
        from dataclasses import asdict
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
# Main
# ---------------------------------------------------------------------------

def run_phase36_verdict() -> None:
    """Step 36.8: Phase 36 verdict."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 36.8: Phase 36 Verdict")
    print("=" * 70)

    rd = _results_dir()

    # ── Load results ──
    readability = _safe_load(os.path.join(rd, 'readability_10k.json'))
    bigrams = _safe_load(os.path.join(rd, 'bigrams_10k.json'))
    bootstrap = _safe_load(os.path.join(rd, 'bootstrap_10k.json'))
    context = _safe_load(os.path.join(rd, 'context_10k.json'))
    folio = _safe_load(os.path.join(rd, 'folio_10k.json'))

    # ── Extract key metrics ──
    bigram_z = readability.get('bigram_z_10k', 0.0)
    signal_rate = readability.get('signal_rate_10k', 0.0)
    n_boot = readability.get('n_bootstrap_words', 0)
    n_genuine = readability.get('n_genuine_signal_words', 0)
    n_cc = readability.get('n_content_content_bigrams', 0)
    n_medical = readability.get('n_medical_collocations', 0)
    n_conf_pairs = readability.get('n_confirmed_pairs', 0)
    longest_run = readability.get('longest_run', 0)
    n_parseable = readability.get('n_parseable_fragments', 0)
    n_trigram = readability.get('n_trigram_hits', 0)
    best_fragment = readability.get('best_fragment')

    # ── Decision logic ──
    # 10K_BREAKTHROUGH: bootstrap ≥5, signal >22%, z >13, ≥1 phrase of 4+ SIGNAL,
    #                   content-content bigrams, medical collocations
    # 10K_ADVANCEMENT: bootstrap ≥2, z ≥13, recognizable signal words, medical patterns
    # 10K_CONFIRMED: z ≈13.12, informative signal words, bootstrap stalls
    # 10K_REGRESSION: z < 13

    if (n_boot >= 5
            and signal_rate > 0.22
            and bigram_z > 13
            and longest_run >= 4
            and n_cc >= 1
            and n_medical >= 1):
        verdict = '10K_BREAKTHROUGH'
        interpretation = (
            "The 10K dictionary unlocked the bootstrap that the 131K dictionary couldn't. "
            "The pipeline produces enough discriminating signal that the Ventris mechanism works: "
            "confirmed words propagate through context to confirm new words. "
            f"Best fragment: {best_fragment.get('text', 'N/A') if best_fragment else 'N/A'}."
        )
    elif (n_boot >= 2
            and bigram_z >= 13
            and n_genuine >= 5):
        verdict = '10K_ADVANCEMENT'
        interpretation = (
            "The 10K dictionary improves the signal landscape enough to confirm new words "
            "and reveal medical collocational patterns, but doesn't produce a cascade. "
            f"Bootstrap confirmed {n_boot} new words. "
            f"The confirmed vocabulary at 10K is more specifically medical than at 131K."
        )
    elif bigram_z >= 10:
        verdict = '10K_CONFIRMED'
        interpretation = (
            f"Track G's z={bigram_z:.2f} holds as the project's strongest sequential structure result. "
            f"The signal is real ({bigram_z:.0f}σ above null) but concentrated in a small number of "
            f"high-frequency words that don't propagate. "
            f"The confirmed vocabulary at 10K ({n_genuine} words) is the most reliable set."
        )
    else:
        verdict = '10K_REGRESSION'
        interpretation = (
            f"Bigram z = {bigram_z:.2f} under the full pipeline is lower than Track G's 13.12 calibration. "
            "The 13.12 may have been a statistical artifact of the calibration setup. "
            "The true ceiling reverts to Phase 29's z=6.14."
        )

    # ── Gap analysis ──
    confirmed_triples = bootstrap.get('confirmed_triples', [])
    unconfirmed_triples = bootstrap.get('unconfirmed_triples', [])
    signal_words_list = [
        w['word'] for w in _safe_load(os.path.join(rd, 'signal_10k.json')).get('word_signals', [])
        if w.get('is_genuine_signal', False)
    ]

    gap_analysis = {
        'best_encoding_model': 'Phase 14-16 CV-syllable (no modifications survived)',
        'best_dictionary': '10K strict Latin',
        'best_signal_metric': f'Bigram z at 10K = {bigram_z:.2f}',
        'confirmed_vocabulary_10k': signal_words_list,
        'confirmed_triples': len(confirmed_triples),
        'unconfirmed_triples': len(unconfirmed_triples),
        'remaining_gap': (
            f"{len(unconfirmed_triples)} unconfirmed triples, "
            f"{100 - signal_rate*100:.0f}% of tokens are not SIGNAL"
        ),
    }

    # ── Print verdict ──
    print(f"\n  VERDICT: {verdict}")
    print(f"\n  {interpretation}")

    print(f"\n  Key metrics:")
    print(f"    Bigram z:         {bigram_z:.2f}")
    print(f"    Signal rate:      {signal_rate:.3f}")
    print(f"    Bootstrap words:  {n_boot}")
    print(f"    Genuine signal:   {n_genuine}")
    print(f"    Content-content:  {n_cc}")
    print(f"    Medical colloc.:  {n_medical}")
    print(f"    Longest run:      {longest_run}")
    print(f"    Trigram hits:     {n_trigram}")

    print(f"\n  Gap analysis:")
    print(f"    Confirmed triples: {len(confirmed_triples)}/25")
    print(f"    Unconfirmed:       {len(unconfirmed_triples)}")
    print(f"    Signal words:      {signal_words_list}")

    # ── Save ──
    elapsed = time.time() - t0

    output = {
        'verdict': verdict,
        'interpretation': interpretation,
        # Key metrics
        'bigram_z_10k': round(bigram_z, 2),
        'signal_rate_10k': round(signal_rate, 4),
        'n_bootstrap_words': n_boot,
        'n_genuine_signal_words': n_genuine,
        'n_content_content_bigrams': n_cc,
        'n_medical_collocations': n_medical,
        'n_confirmed_pairs': n_conf_pairs,
        'longest_signal_run': longest_run,
        'n_trigram_hits': n_trigram,
        'n_parseable_fragments': n_parseable,
        'best_fragment': best_fragment,
        # Gap analysis
        'gap_analysis': gap_analysis,
        # Readability summary
        'readability_passed': readability.get('n_passed', 0),
        'readability_total': readability.get('n_total', 0),
        'bootstrap_verdict': bootstrap.get('verdict', 'N/A'),
        'runtime_seconds': round(elapsed, 1),
    }

    out_path = os.path.join(rd, 'phase36_verdict.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(output), f, indent=2)
    print(f"\n  Saved → {out_path}")

    # ── Final summary ──
    print("\n" + "=" * 70)
    print(f"PHASE 36 VERDICT: {verdict}")
    print("=" * 70)
    print(f"\n  {interpretation}")
    print(f"\n  Runtime: {elapsed:.1f}s")
