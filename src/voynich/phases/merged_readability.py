"""
Step 38.9 – Full Readability Battery
======================================
The definitive cross-phase comparison using the macaronic pipeline.
14 metrics covering dict-hit, signal, bigrams, content structure, and
language composition.

Dependency chain:
    merged_dict.json           (Step 38.1)
    merged_decode.json         (Step 38.2)
    merged_signal.json         (Step 38.3)
    merged_bigrams.json        (Step 38.4)
    merged_context.json        (Step 38.5)
    merged_bootstrap.json      (Step 38.6)
    merged_concat.json         (Step 38.7)
    merged_folio.json          (Step 38.8)
    bigrams_10k.json           (Step 36.3 — for comparison)
    signal_10k.json            (Step 36.2 — for comparison)
        → merged_readability.json (this step)
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

def run_merged_readability() -> None:
    """Step 38.9: Full Readability Battery."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 38.9: Full Readability Battery")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load all upstream results ──
    print("\n  1. Loading all upstream results …")
    dict_data = _safe_load(os.path.join(rd, 'merged_dict.json'))
    decode_data = _safe_load(os.path.join(rd, 'merged_decode.json'))
    signal_data = _safe_load(os.path.join(rd, 'merged_signal.json'))
    bigram_data = _safe_load(os.path.join(rd, 'merged_bigrams.json'))
    context_data = _safe_load(os.path.join(rd, 'merged_context.json'))
    bootstrap_data = _safe_load(os.path.join(rd, 'merged_bootstrap.json'))
    concat_data = _safe_load(os.path.join(rd, 'merged_concat.json'))
    folio_data = _safe_load(os.path.join(rd, 'merged_folio.json'))

    # Comparison baselines
    bigram_10k = _safe_load(os.path.join(rd, 'bigrams_10k.json'))
    signal_10k = _safe_load(os.path.join(rd, 'signal_10k.json'))

    # ── 2. Collect 14 metrics ──
    print("  2. Collecting metrics …")

    metrics = {}

    # M1: Merged dict-hit rate
    metrics['M01_dict_hit_rate'] = decode_data.get('merged_hit_rate', 0.0)

    # M2: Merged SIGNAL rate
    metrics['M02_signal_rate'] = signal_data.get('signal_rate', 0.0)

    # M3: Merged bigram z
    metrics['M03_bigram_z'] = bigram_data.get('bigram_z', 0.0)

    # M4: Trigram hits
    metrics['M04_trigram_hits'] = bigram_data.get('trigram_hits', 0)

    # M5: Confirmed vocabulary size (total)
    metrics['M05_confirmed_vocab'] = signal_data.get('n_genuine_signal_words', 0)

    # M6: Longest SIGNAL run
    longest_run = 0
    for fr in folio_data.get('folio_results', []):
        lr = fr.get('longest_run', 0)
        if lr > longest_run:
            longest_run = lr
    metrics['M06_longest_signal_run'] = longest_run

    # M7: Parseable macaronic fragments
    n_macaronic = sum(
        fr.get('n_macaronic_runs', 0)
        for fr in folio_data.get('folio_results', [])
    )
    metrics['M07_macaronic_fragments'] = n_macaronic

    # M8: Net signal
    metrics['M08_net_signal'] = signal_data.get('net_signal', 0.0)

    # M9: Selectivity
    metrics['M09_selectivity'] = decode_data.get('merged_selectivity', 0.0)

    # M10: Content-content bigram count
    metrics['M10_cc_bigrams'] = bigram_data.get('n_content_content', 0)

    # M11: Confirmed-confirmed pairs with language tags
    confirmed_pairs = context_data.get('confirmed_pair_types', {})
    metrics['M11_confirmed_pairs'] = sum(confirmed_pairs.values())

    # M12: Medical phrase matches
    n_medical = sum(
        fr.get('n_medical', 0) for fr in folio_data.get('folio_results', [])
    )
    metrics['M12_medical_phrases'] = n_medical

    # M13: Cross-language bigram count
    macaronic_struct = bigram_data.get('macaronic_structure', {})
    metrics['M13_cross_language_bigrams'] = macaronic_struct.get('n_cross_language', 0)

    # M14: f57v Italian vocabulary match rate
    f57v = folio_data.get('f57v_venetian', {})
    n_f57v = f57v.get('n_f57v_unique', 1)
    metrics['M14_f57v_italian_rate'] = round(
        f57v.get('n_italian_matches', 0) / n_f57v, 4
    ) if n_f57v > 0 else 0.0

    # ── 3. Print metrics ──
    print("\n  Readability Metrics:")
    print("  " + "-" * 55)
    for key, val in sorted(metrics.items()):
        label = key[4:]  # Strip M0X_ prefix
        if isinstance(val, float):
            print(f"    {label:35s}: {val:.4f}")
        else:
            print(f"    {label:35s}: {val}")
    print("  " + "-" * 55)

    # ── 4. Progression table ──
    print("\n  4. Cross-phase progression …")
    progression = {
        'Phase_29_131K': {
            'dict': '131K Latin',
            'signal_rate': 0.165,
            'bigram_z': 6.14,
            'cc_bigrams': 0,
            'cross_lang': 0,
        },
        'Phase_36_10K': {
            'dict': '10K Latin',
            'signal_rate': signal_10k.get('signal_rate', 0.1853),
            'bigram_z': bigram_10k.get('bigram_z', 12.66),
            'cc_bigrams': 0,
            'cross_lang': 0,
        },
        'Phase_37_merged': {
            'dict': 'merged (19K)',
            'signal_rate': 0.002,  # Phase 37 merged signal rate
            'bigram_z': 16.97,
            'cc_bigrams': 0,
            'cross_lang': 1,
        },
        'Phase_38_merged': {
            'dict': 'merged (full pipeline)',
            'signal_rate': metrics['M02_signal_rate'],
            'bigram_z': metrics['M03_bigram_z'],
            'cc_bigrams': metrics['M10_cc_bigrams'],
            'cross_lang': metrics['M13_cross_language_bigrams'],
        },
    }

    print(f"    {'Phase':20s} {'Dict':15s} {'SIGNAL':>8s} {'Bigram z':>10s} "
          f"{'CC':>4s} {'X-lang':>7s}")
    print("    " + "-" * 70)
    for phase, data in progression.items():
        print(f"    {phase:20s} {data['dict']:15s} "
              f"{data['signal_rate']:>8.4f} {data['bigram_z']:>10.2f} "
              f"{data['cc_bigrams']:>4d} {data['cross_lang']:>7d}")

    # ── 5. Language partition of vocabulary ──
    print("\n  5. Vocabulary partition …")
    vocab_partition = {
        'shared': signal_data.get('n_shared_signal_words', 0),
        'latin_only': signal_data.get('n_latin_only_signal_words', 0),
        'italian_only': signal_data.get('n_italian_only_signal_words', 0),
    }
    for k, v in vocab_partition.items():
        print(f"    {k:15s}: {v}")

    # ── 6. Bootstrap summary ──
    print("\n  6. Bootstrap summary …")
    boot_confirmed = bootstrap_data.get('n_confirmed_total', 0)
    boot_italian = bootstrap_data.get('n_confirmed_italian_only', 0)
    boot_shape = bootstrap_data.get('shape', 'unknown')
    print(f"    Confirmed: {boot_confirmed} (Italian: {boot_italian})")
    print(f"    Shape: {boot_shape}")

    # ── 7. Save ──
    elapsed = time.time() - t0

    output = {
        'metrics': metrics,
        'progression': progression,
        'vocab_partition': vocab_partition,
        'bootstrap_summary': {
            'n_confirmed': boot_confirmed,
            'n_italian': boot_italian,
            'shape': boot_shape,
        },
        'confirmed_pair_types': confirmed_pairs,
        'verdict': (
            f"Bigram z={metrics['M03_bigram_z']:.2f}, "
            f"SIGNAL={metrics['M02_signal_rate']:.4f}, "
            f"CC={metrics['M10_cc_bigrams']}, "
            f"cross-lang={metrics['M13_cross_language_bigrams']}, "
            f"medical={metrics['M12_medical_phrases']}, "
            f"vocab={metrics['M05_confirmed_vocab']} "
            f"({vocab_partition['italian_only']} Italian-only)."
        ),
        'runtime_seconds': round(elapsed, 1),
    }

    out_path = os.path.join(rd, 'merged_readability.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(output), f, indent=2)
    print(f"\n  Saved → {out_path} ({elapsed:.1f}s)")
