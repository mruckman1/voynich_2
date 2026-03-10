"""
Step 34.18 – Dictionary Right-Sizing (Track G)
===============================================
Re-runs signal isolation using dictionaries of different sizes to find
the optimal dictionary that maximises SIGNAL rate while minimising
ANTI_SIGNAL (false positives).

The optimal dictionary size balances coverage (large enough to catch real
decoded words) against specificity (small enough that random sequences
rarely match).

Dependency chain:
    combined_refine.json      (Phase 15 assignment)
    modifier_integrate.json   (Phase 16 modifiers)
    null_corpus.json          (Phase 17 seeds)
    signal_bigrams.json       (Phase 29 baseline)
        → dict_calibration.json   (this step)
"""

import json
import math
import os
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    load_corpus,
)
from voynich.core.reference import (
    build_expanded_word_set,
    load_reference_corpus,
)
from voynich.phases.null_corpus import (
    _build_eva_bigram_model,
    _generate_null_corpus,
    _reconstruct_modifier_rules,
)
from voynich.phases.signal_bigrams import (
    _build_reference_bigrams,
    _find_signal_pairs,
    _null_permutation_test,
    _relaxed_bigram_test,
)
from voynich.phases.signal_isolation import _decode_corpus_r3


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
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DictVariantMetrics:
    label: str
    dict_size: int
    dict_hit_rate: float
    signal_rate: float
    anti_rate: float
    shared_hit_rate: float
    shared_miss_rate: float
    net_signal: float          # signal_rate - anti_rate
    selectivity: float         # dict_hit / null_dict_hit
    bigram_z: float
    n_signal: int
    n_bigram_hits: int
    n_relaxed_bigram_hits: int
    null_dict_hit: float


@dataclass
class DictCalibrationResult:
    variants: List[Dict]
    optimal_label: str
    optimal_size: int
    optimal_signal_rate: float
    optimal_bigram_z: float
    optimal_dict_hit: float
    optimal_net_signal: float
    baseline_label: str
    baseline_bigram_z: float
    baseline_signal_rate: float
    baseline_dict_hit: float
    improvement_over_baseline: Dict[str, float]
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Dictionary variant building
# ---------------------------------------------------------------------------

def _build_dict_variants(
    base_words: set,
    ref_corpus,
    target_sizes: List[int],
) -> List[Tuple[str, set]]:
    """Build dictionary subsets of increasing sizes.

    Strategy:
    - Small sizes (≤ len(base_words)): take most frequent words
    - Large sizes (> len(base_words)): progressively expand via
      medieval spelling variants
    """
    # Rank base words by frequency in reference corpus
    ref_tokens = [w.lower() for w in ref_corpus.get_combined_tokens('latin')
                  if len(w) >= 2]
    freq = Counter(ref_tokens)
    ranked = sorted(base_words, key=lambda w: freq.get(w, 0), reverse=True)

    # Full expanded dictionary
    expanded_full, _ = build_expanded_word_set(base_words)
    full_dict = base_words | expanded_full
    expanded_ranked = sorted(expanded_full - base_words,
                             key=lambda w: freq.get(w, 0), reverse=True)
    all_ranked = ranked + expanded_ranked

    variants: List[Tuple[str, set]] = []
    for size in target_sizes:
        if size >= len(full_dict):
            label = f"D{len(variants)+1}_{len(full_dict)}"
            variants.append((label, full_dict.copy()))
        else:
            subset = set(all_ranked[:size])
            actual_size = len(subset)
            label = f"D{len(variants)+1}_{actual_size}"
            variants.append((label, subset))

    return variants


def _classify_tokens(
    real_hits: List[bool],
    null_hits_list: List[List[bool]],
) -> List[str]:
    """Classify each token as SIGNAL/SHARED_HIT/SHARED_MISS/ANTI_SIGNAL."""
    classifications: List[str] = []
    for idx in range(len(real_hits)):
        r_hit = real_hits[idx]
        null_hit_count = sum(1 for nh in null_hits_list if nh[idx])
        if r_hit and null_hit_count <= 1:
            classifications.append('SIGNAL')
        elif r_hit and null_hit_count >= 3:
            classifications.append('SHARED_HIT')
        elif not r_hit and null_hit_count >= 3:
            classifications.append('ANTI_SIGNAL')
        else:
            classifications.append('SHARED_MISS')
    return classifications


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_dict_calibration() -> None:
    """Step 34.18: Signal isolation across dictionary sizes."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 34.18: Dictionary Right-Sizing (Track G)")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load assignment + modifiers ──
    print("\n  1. Loading assignment and modifiers …")
    with open(os.path.join(rd, 'combined_refine.json')) as f:
        refine_data = json.load(f)
    assignment = refine_data.get('best_assignment', {})

    with open(os.path.join(rd, 'modifier_integrate.json')) as f:
        mod_data = json.load(f)
    modifier_chars, modifier_rules = _reconstruct_modifier_rules(mod_data)

    null_seeds = [100, 101, 102, 103, 104]
    null_path = os.path.join(rd, 'null_corpus.json')
    if os.path.exists(null_path):
        with open(null_path) as f:
            null_data = json.load(f)
        null_seeds = [r['seed'] for r in null_data.get('null_runs', [])]

    # ── 2. Load corpus + build lookup ──
    print("  2. Loading corpus …")
    corpus = load_corpus(verbose=False)
    eva_to_triple = build_eva_to_triple_lookup()

    token_folios: List[str] = []
    token_evas: List[str] = []
    all_tokens: List[str] = []
    for folio, page in corpus.pages.items():
        for token in page.all_tokens:
            token_folios.append(folio)
            token_evas.append(token)
            all_tokens.append(token)
    n_tokens = len(all_tokens)
    print(f"     {n_tokens} tokens across {len(corpus.pages)} folios")

    # ── 3. Build base word set ──
    print("  3. Building base word set …")
    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    base_words = set(
        w.lower() for w in ref_corpus.get_combined_tokens('latin')
        if len(w) >= 2
    )
    print(f"     Base dictionary: {len(base_words)} words")

    # ── 4. Build null corpora once (reused across all variants) ──
    print("  4. Building null corpora …")
    bigram_probs, initial_probs, token_lengths = _build_eva_bigram_model(
        all_tokens,
    )
    null_corpora: List[List[str]] = []
    for seed in null_seeds:
        null_tokens = _generate_null_corpus(
            bigram_probs, initial_probs, token_lengths, n_tokens, seed,
        )
        null_corpora.append(null_tokens)
    print(f"     Generated {len(null_corpora)} null corpora")

    # ── 5. Decode real corpus once (decoded text is the same for all
    #        dictionary variants — only the matching changes) ──
    print("  5. Decoding real corpus (Phase 16 R3 strategy) …")
    # We need to decode with a large enough ref_word_set for R3 to pick
    # the best variant.  Use the full expanded dict for decoding, then
    # match against each variant's dict separately.
    expanded_full, _ = build_expanded_word_set(base_words)
    full_ref_set = base_words | expanded_full

    real_decoded = _decode_corpus_r3(
        all_tokens, assignment, eva_to_triple,
        modifier_chars, modifier_rules, full_ref_set,
    )

    # Decode null corpora
    null_decoded_list: List[List[str]] = []
    for null_tokens in null_corpora:
        nd = _decode_corpus_r3(
            null_tokens, assignment, eva_to_triple,
            modifier_chars, modifier_rules, full_ref_set,
        )
        null_decoded_list.append(nd)

    # ── 6. Build dictionary variants ──
    print("  6. Building dictionary variants …")
    target_sizes = [5000, 10000, 17000, 30000, 50000, 80000, 131000]
    dict_variants = _build_dict_variants(base_words, ref_corpus, target_sizes)

    for label, dset in dict_variants:
        print(f"     {label}: {len(dset)} words")

    # ── 7. Build reference bigrams ──
    print("  7. Building reference bigrams …")
    ref_tokens = [w.lower() for w in ref_corpus.get_combined_tokens('latin')
                  if len(w) >= 2]
    ref_bigrams, _ = _build_reference_bigrams(ref_tokens)
    print(f"     {len(ref_bigrams)} reference bigrams")

    # ── 8. Evaluate each dictionary variant ──
    print("\n  8. Evaluating dictionary variants …")
    variant_results: List[DictVariantMetrics] = []

    for label, dict_set in dict_variants:
        print(f"\n     --- {label} ({len(dict_set)} words) ---")

        # Match real decoded against this dict
        real_hits = [w.lower() in dict_set for w in real_decoded]

        # Match null decoded against this dict
        null_hits_list: List[List[bool]] = []
        for nd in null_decoded_list:
            null_hits_list.append([w.lower() in dict_set for w in nd])

        # Classify tokens
        classifications = _classify_tokens(real_hits, null_hits_list)

        # Compute rates
        n_signal = classifications.count('SIGNAL')
        n_shared_hit = classifications.count('SHARED_HIT')
        n_anti = classifications.count('ANTI_SIGNAL')
        n_shared_miss = classifications.count('SHARED_MISS')

        signal_rate = n_signal / n_tokens if n_tokens > 0 else 0.0
        anti_rate = n_anti / n_tokens if n_tokens > 0 else 0.0
        shared_hit_rate = n_shared_hit / n_tokens if n_tokens > 0 else 0.0
        shared_miss_rate = n_shared_miss / n_tokens if n_tokens > 0 else 0.0
        dict_hit_rate = sum(real_hits) / n_tokens if n_tokens > 0 else 0.0
        net_signal = signal_rate - anti_rate

        # Null dict hit rate (average across null corpora)
        null_dict_hits = []
        for nh in null_hits_list:
            null_dict_hits.append(sum(nh) / n_tokens if n_tokens > 0 else 0.0)
        null_dict_hit = sum(null_dict_hits) / len(null_dict_hits) if null_dict_hits else 0.0
        selectivity = dict_hit_rate / null_dict_hit if null_dict_hit > 0 else float('inf')

        # Bigram z-score
        signal_pairs = _find_signal_pairs(
            classifications, real_decoded, token_folios,
        )
        n_bigram_hits = sum(
            1 for _, _, w1, w2 in signal_pairs
            if (w1, w2) in ref_bigrams
        )

        _, null_mean, null_std = _null_permutation_test(
            n_signal, n_tokens, real_decoded, token_folios,
            ref_bigrams, n_perms=500, seed=42,
        )
        bigram_hit_rate = n_bigram_hits / len(signal_pairs) if signal_pairs else 0.0
        bigram_z = (bigram_hit_rate - null_mean) / null_std if null_std > 0 else 0.0

        # Relaxed bigram hits
        n_relaxed = _relaxed_bigram_test(
            signal_pairs, ref_bigrams, dict_set,
        )

        vm = DictVariantMetrics(
            label=label,
            dict_size=len(dict_set),
            dict_hit_rate=round(dict_hit_rate, 4),
            signal_rate=round(signal_rate, 4),
            anti_rate=round(anti_rate, 4),
            shared_hit_rate=round(shared_hit_rate, 4),
            shared_miss_rate=round(shared_miss_rate, 4),
            net_signal=round(net_signal, 4),
            selectivity=round(selectivity, 4),
            bigram_z=round(bigram_z, 2),
            n_signal=n_signal,
            n_bigram_hits=n_bigram_hits,
            n_relaxed_bigram_hits=n_relaxed,
            null_dict_hit=round(null_dict_hit, 4),
        )
        variant_results.append(vm)

        print(f"       dict_hit={vm.dict_hit_rate:.3f}  SIGNAL={vm.signal_rate:.3f}"
              f"  ANTI={vm.anti_rate:.3f}  net={vm.net_signal:.3f}"
              f"  z={vm.bigram_z:.2f}  sel={vm.selectivity:.2f}")

    # ── 9. Find optimal ──
    print("\n  9. Finding optimal dictionary size …")
    # Optimise composite: SIGNAL_rate × bigram_z
    best = max(variant_results,
               key=lambda v: v.signal_rate * max(v.bigram_z, 0.01))

    print(f"     Optimal: {best.label} (size={best.dict_size})")
    print(f"       SIGNAL={best.signal_rate:.3f}  bigram_z={best.bigram_z:.2f}"
          f"  dict_hit={best.dict_hit_rate:.3f}  net_signal={best.net_signal:.3f}")

    # ── 10. Compare to 131K baseline ──
    baseline = next(
        (v for v in variant_results if v.dict_size >= 100000), variant_results[-1]
    )

    # Load Phase 29 baseline z
    phase29_z = 0.0
    bg_path = os.path.join(rd, 'signal_bigrams.json')
    if os.path.exists(bg_path):
        with open(bg_path) as f:
            bg = json.load(f)
        phase29_z = bg.get('bigram_z_score', 0.0)

    improvement = {
        'signal_rate_delta': round(best.signal_rate - baseline.signal_rate, 4),
        'bigram_z_delta': round(best.bigram_z - baseline.bigram_z, 2),
        'dict_hit_delta': round(best.dict_hit_rate - baseline.dict_hit_rate, 4),
        'net_signal_delta': round(best.net_signal - baseline.net_signal, 4),
        'vs_phase29_z_delta': round(best.bigram_z - phase29_z, 2),
    }

    elapsed = time.time() - t0

    result = DictCalibrationResult(
        variants=[asdict(v) for v in variant_results],
        optimal_label=best.label,
        optimal_size=best.dict_size,
        optimal_signal_rate=best.signal_rate,
        optimal_bigram_z=best.bigram_z,
        optimal_dict_hit=best.dict_hit_rate,
        optimal_net_signal=best.net_signal,
        baseline_label=baseline.label,
        baseline_bigram_z=baseline.bigram_z,
        baseline_signal_rate=baseline.signal_rate,
        baseline_dict_hit=baseline.dict_hit_rate,
        improvement_over_baseline=improvement,
        runtime_seconds=round(elapsed, 1),
    )

    # ── 11. Save ──
    out_path = os.path.join(rd, 'dict_calibration.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)
    print(f"\n  Saved → {out_path}")

    # ── Summary ──
    print("\n" + "=" * 70)
    print("DICT CALIBRATION SUMMARY")
    print("=" * 70)
    print(f"\n  {'Label':<15s} {'Size':>7s} {'DictHit':>8s} {'SIGNAL':>8s}"
          f" {'ANTI':>8s} {'Net':>8s} {'z':>7s} {'Sel':>7s}")
    print("  " + "-" * 62)
    for v in variant_results:
        marker = " ◀ OPTIMAL" if v.label == best.label else ""
        print(f"  {v.label:<15s} {v.dict_size:>7d} {v.dict_hit_rate:>8.3f}"
              f" {v.signal_rate:>8.3f} {v.anti_rate:>8.3f}"
              f" {v.net_signal:>8.3f} {v.bigram_z:>7.2f}"
              f" {v.selectivity:>7.2f}{marker}")

    print(f"\n  Phase 29 baseline z = {phase29_z:.2f}")
    print(f"  Optimal z = {best.bigram_z:.2f}"
          f" (Δ = {improvement['vs_phase29_z_delta']:+.2f})")
    print(f"\n  Completed in {elapsed:.1f}s")
