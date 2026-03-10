"""
Step 35.4 – Combined Bigram Plausibility (THE PREDICTION TEST)
===============================================================
Measure bigram plausibility on consecutive SIGNAL pairs under the
combined spatial+10K model.

THE PREDICTION: bigram z should exceed 13.12 (Track G alone) because
spatial conditioning provides more SIGNAL pairs for bigram testing.

Dependency chain:
    combined_decode.json       (Step 35.2)
    combined_signal.json       (Step 35.3)
    signal_bigrams.json        (Phase 29 baseline)
    dict_calibration.json      (Track G baseline z)
        → combined_bigrams.json (this step)
"""

import json
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.reference import (
    build_expanded_word_set,
    load_reference_corpus,
)
from voynich.phases.signal_bigrams import (
    _build_reference_bigrams,
    _find_signal_pairs,
    _find_signal_triples,
    _null_permutation_test,
    _relaxed_bigram_test,
    _folio_signal_pair_ranking,
    FolioSignalPairStats,
)


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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_combined_bigrams() -> None:
    """Step 35.4: Combined bigram plausibility — THE PREDICTION TEST."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 35.4: Combined Bigram Plausibility (THE PREDICTION TEST)")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load inputs ──
    print("\n  1. Loading inputs ...")
    with open(os.path.join(rd, 'combined_decode.json')) as f:
        cd = json.load(f)
    with open(os.path.join(rd, 'combined_signal.json')) as f:
        cs = json.load(f)

    token_folios = cd['token_folios']
    token_evas = cd['token_evas']
    token_decoded = cd['token_decoded']
    token_dict_hits = cd['token_dict_hits_10k']
    classifications = cs['token_classifications']
    n_tokens = cd['n_tokens']
    n_signal = cs['n_signal']
    signal_rate = cs['signal_rate']

    print(f"     {n_tokens} tokens, {n_signal} SIGNAL ({signal_rate:.1%})")

    # Load Phase 29 baseline
    phase29_path = os.path.join(rd, 'signal_bigrams.json')
    phase29_z = 0.0
    phase29_n_hits = 0
    phase29_n_relaxed = 0
    if os.path.exists(phase29_path):
        with open(phase29_path) as f:
            p29 = json.load(f)
        phase29_z = p29.get('bigram_z_score', 0.0)
        phase29_n_hits = p29.get('n_bigram_hits', 0)
        phase29_n_relaxed = p29.get('n_relaxed_bigram_hits', 0)
        print(f"     Phase 29 baseline: z={phase29_z:.2f}")

    # Load Track G baseline
    track_g_z = 13.12
    dcal_path = os.path.join(rd, 'dict_calibration.json')
    if os.path.exists(dcal_path):
        with open(dcal_path) as f:
            dcal = json.load(f)
        track_g_z = dcal.get('optimal_bigram_z', 13.12)
    print(f"     Track G baseline: z={track_g_z:.2f}")

    # ── 2. Build reference bigram/trigram table ──
    print("\n  2. Building reference bigram/trigram table ...")
    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    ref_tokens = [w.lower() for w in ref_corpus.get_combined_tokens('latin')
                  if len(w) >= 2]
    base_words = set(ref_tokens)
    expanded, _ = build_expanded_word_set(base_words)
    ref_word_set = base_words | expanded

    ref_bigrams, ref_trigrams = _build_reference_bigrams(ref_tokens)
    print(f"     {len(ref_bigrams)} bigrams, {len(ref_trigrams)} trigrams")

    # ── 3. Find SIGNAL-SIGNAL pairs ──
    print("\n  3. Finding SIGNAL-SIGNAL pairs ...")
    signal_pairs = _find_signal_pairs(
        classifications, token_decoded, token_folios,
    )
    print(f"     {len(signal_pairs)} consecutive SIGNAL-SIGNAL pairs")

    # ── 4. Bigram plausibility test ──
    print("\n  4. Bigram plausibility on SIGNAL pairs ...")
    bigram_hits = []
    for folio, pos, w1, w2 in signal_pairs:
        if (w1, w2) in ref_bigrams:
            bigram_hits.append([w1, w2])

    n_bigram_hits = len(bigram_hits)
    bigram_hit_rate = n_bigram_hits / len(signal_pairs) if signal_pairs else 0.0
    print(f"     {n_bigram_hits} exact bigram hits (rate={bigram_hit_rate:.6f})")

    if bigram_hits:
        print("     Matching pairs:")
        for pair in bigram_hits[:20]:
            print(f"       {pair[0]} {pair[1]}")

    # ── 5. Null permutation test ──
    print("\n  5. Null permutation test (1000 permutations) ...")
    null_rates, null_mean, null_std = _null_permutation_test(
        n_signal, n_tokens, token_decoded, token_folios,
        ref_bigrams, n_perms=1000, seed=42,
    )

    if null_std > 0:
        z_score = (bigram_hit_rate - null_mean) / null_std
    else:
        z_score = float('inf') if bigram_hit_rate > null_mean else 0.0

    p_value = sum(1 for r in null_rates if r >= bigram_hit_rate) / len(null_rates)

    print(f"     Null mean: {null_mean:.6f}, std: {null_std:.6f}")
    print(f"     z-score: {z_score:.2f}, p-value: {p_value:.4f}")
    print(f"     Phase 29 z-score: {phase29_z:.2f}")
    print(f"     Track G z-score:  {track_g_z:.2f}")
    print(f"     Delta vs Phase 29: {z_score - phase29_z:+.2f}")
    print(f"     Delta vs Track G:  {z_score - track_g_z:+.2f}")

    # ── 6. Trigram test ──
    print("\n  6. Trigram plausibility test ...")
    signal_triples = _find_signal_triples(
        classifications, token_decoded, token_folios,
    )
    trigram_hits = []
    for folio, pos, w1, w2, w3 in signal_triples:
        if (w1, w2, w3) in ref_trigrams:
            trigram_hits.append([w1, w2, w3])

    n_trigram_hits = len(trigram_hits)
    trigram_hit_rate = n_trigram_hits / len(signal_triples) if signal_triples else 0.0
    print(f"     {len(signal_triples)} SIGNAL triples, {n_trigram_hits} trigram hits")

    if trigram_hits:
        for tri in trigram_hits[:10]:
            print(f"       {' '.join(tri)}")

    # ── 7. Relaxed bigram test ──
    print("\n  7. Relaxed bigram test (edit distance 1) ...")
    n_relaxed = _relaxed_bigram_test(signal_pairs, ref_bigrams, ref_word_set)
    relaxed_rate = (n_bigram_hits + n_relaxed) / len(signal_pairs) if signal_pairs else 0.0
    print(f"     {n_relaxed} additional relaxed matches")
    print(f"     Combined rate: {relaxed_rate:.6f}")

    # ── 8. Per-folio ranking ──
    print("\n  8. Per-folio SIGNAL pair ranking (top 10) ...")
    folio_stats = _folio_signal_pair_ranking(
        signal_pairs, ref_bigrams, classifications, token_folios,
    )
    for fs in folio_stats[:10]:
        print(f"     {fs.folio:8s}  signal={fs.n_signal:3d}/{fs.n_tokens:3d} "
              f"({fs.signal_rate:.1%})  "
              f"pairs={fs.n_signal_pairs:3d}  "
              f"bigram_hits={fs.n_bigram_hits}")

    # ── 9. Bigram type analysis ──
    print("\n  9. Bigram type analysis ...")
    _FUNCTION_WORDS = {'de', 'in', 'ad', 'cum', 'per', 'pro', 'ex', 'ab',
                       'et', 'sed', 'si', 'ut', 'non', 'nec', 'aut', 'vel'}
    type_counts: Counter = Counter()
    for w1, w2 in bigram_hits:
        is_func1 = w1 in _FUNCTION_WORDS
        is_func2 = w2 in _FUNCTION_WORDS
        if is_func1 and is_func2:
            type_counts['function_function'] += 1
        elif is_func1 or is_func2:
            type_counts['function_content'] += 1
        else:
            type_counts['content_content'] += 1

    for btype, cnt in type_counts.most_common():
        print(f"     {btype:25s}: {cnt}")

    # ── 10. Verdict ──
    delta_p29 = z_score - phase29_z
    delta_g = z_score - track_g_z

    if z_score > 15:
        verdict_label = "COMBINED_BREAKTHROUGH"
    elif z_score > track_g_z:
        verdict_label = "COMBINED_AMPLIFICATION"
    elif z_score > 8:
        verdict_label = "COMBINED_IMPROVEMENT"
    elif z_score > phase29_z:
        verdict_label = "COMBINED_CONFIRMED"
    else:
        verdict_label = "NO_INTERACTION"

    verdict = (
        f"{verdict_label}: z={z_score:.2f} "
        f"(Phase 29: {phase29_z:.2f}, Track G: {track_g_z:.2f}, "
        f"delta_P29={delta_p29:+.2f}, delta_G={delta_g:+.2f}), "
        f"{n_bigram_hits} hits, {n_relaxed} relaxed, {n_trigram_hits} trigrams"
    )
    print(f"\n  VERDICT: {verdict}")

    # ── 11. Save ──
    print("\n  11. Saving combined_bigrams.json ...")
    output = {
        # Per-token cache (propagated downstream)
        'token_folios': token_folios,
        'token_evas': token_evas,
        'token_decoded': token_decoded,
        'token_classifications': classifications,
        'token_dict_hits': token_dict_hits,
        # Metrics
        'n_tokens': n_tokens,
        'n_signal': n_signal,
        'signal_rate': round(signal_rate, 6),
        'ref_bigram_count': len(ref_bigrams),
        'ref_trigram_count': len(ref_trigrams),
        'n_signal_pairs': len(signal_pairs),
        'n_bigram_hits': n_bigram_hits,
        'bigram_hit_rate': round(bigram_hit_rate, 6),
        'bigram_hit_pairs': bigram_hits[:50],
        'null_bigram_mean': round(null_mean, 6),
        'null_bigram_std': round(null_std, 6),
        'bigram_p_value': round(p_value, 4),
        'bigram_z_score': round(z_score, 2) if z_score != float('inf') else 999.0,
        'n_signal_triples': len(signal_triples),
        'n_trigram_hits': n_trigram_hits,
        'trigram_hit_rate': round(trigram_hit_rate, 6),
        'trigram_hit_triples': trigram_hits[:20],
        'n_relaxed_bigram_hits': n_relaxed,
        'relaxed_bigram_hit_rate': round(relaxed_rate, 6),
        'bigram_type_counts': dict(type_counts),
        # Baselines
        'phase29_bigram_z': phase29_z,
        'phase29_n_bigram_hits': phase29_n_hits,
        'phase29_n_relaxed': phase29_n_relaxed,
        'track_g_bigram_z': track_g_z,
        'delta_bigram_z_vs_p29': round(delta_p29, 2),
        'delta_bigram_z_vs_g': round(delta_g, 2),
        'delta_n_bigram_hits': n_bigram_hits - phase29_n_hits,
        'delta_n_relaxed': n_relaxed - phase29_n_relaxed,
        # Folio stats
        'folio_signal_pair_stats': [
            _convert(asdict(fs)) for fs in folio_stats[:30]
        ],
        'gate_passed': z_score > phase29_z,
        'verdict': verdict,
        'runtime_seconds': round(time.time() - t0, 1),
    }

    with open(os.path.join(rd, 'combined_bigrams.json'), 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n  Step 35.4 completed in {time.time() - t0:.1f}s")
