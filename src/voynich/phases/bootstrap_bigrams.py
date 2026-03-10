"""
Phase 30.3 – Post-Bootstrap Bigram Plausibility Test
=======================================================
Re-runs Phase 29.1's SIGNAL bigram plausibility test with the evolved
signal set from the bootstrap.  This file also becomes the new per-token
cache (parallel arrays) for downstream Phase 30 steps.

Dependency chain:
    bootstrap_loop.json        (Step 30.1 — evolved assignment)
    combined_refine.json       (Phase 15 — fallback)
    modifier_integrate.json    (Phase 16 modifiers)
    null_corpus.json           (Phase 17 seeds)
    signal_bigrams.json        (Phase 29.1 — baseline for comparison)
        → bootstrap_bigrams.json  (this step — new per-token cache)
"""

import json
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
    _find_signal_triples,
    _folio_signal_pair_ranking,
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
class BootstrapBigramResult:
    # Per-token cache (parallel arrays — downstream steps read these)
    token_folios: List[str]
    token_evas: List[str]
    token_decoded: List[str]
    token_classifications: List[str]
    token_dict_hits: List[bool]
    n_tokens: int
    n_signal: int
    signal_rate: float

    # Reference bigram table
    ref_bigram_count: int
    ref_trigram_count: int

    # SIGNAL bigram test
    n_signal_pairs: int
    n_bigram_hits: int
    bigram_hit_rate: float
    bigram_hit_pairs: List[List[str]]
    null_bigram_mean: float
    null_bigram_std: float
    bigram_p_value: float
    bigram_z_score: float

    # SIGNAL trigram test
    n_signal_triples: int
    n_trigram_hits: int
    trigram_hit_rate: float
    trigram_hit_triples: List[List[str]]

    # Relaxed bigram test
    n_relaxed_bigram_hits: int
    relaxed_bigram_hit_rate: float

    # Per-folio ranking
    folio_signal_pair_stats: List[Dict]

    # Bootstrap comparison
    baseline_bigram_z: float
    baseline_n_bigram_hits: int
    baseline_n_relaxed: int
    delta_bigram_z: float
    delta_n_bigram_hits: int
    delta_n_relaxed: int

    gate_passed: bool
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Token classification with custom assignment
# ---------------------------------------------------------------------------

def _recompute_with_assignment(
    rd: str,
    assignment: Dict[str, str],
) -> Tuple[
    List[str], List[str], List[str], List[str], List[bool],
    set, List[str],
]:
    """Recompute per-token classifications using a custom assignment.

    Same logic as signal_bigrams._recompute_token_classifications() but
    accepts an assignment parameter instead of reading combined_refine.json.

    Returns:
        (token_folios, token_evas, token_decoded,
         token_classifications, token_dict_hits,
         ref_word_set, base_words_list)
    """
    with open(os.path.join(rd, 'modifier_integrate.json')) as f:
        mod_data = json.load(f)
    modifier_chars, modifier_rules = _reconstruct_modifier_rules(mod_data)

    null_seeds = [100, 101, 102, 103, 104]
    null_path = os.path.join(rd, 'null_corpus.json')
    if os.path.exists(null_path):
        with open(null_path) as f:
            null_data = json.load(f)
        null_seeds = [r['seed'] for r in null_data.get('null_runs', [])]

    # Build reference word set
    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    base_words = set(
        w.lower() for w in ref_corpus.get_combined_tokens('latin')
        if len(w) >= 2
    )
    expanded, _ = build_expanded_word_set(base_words)
    ref_word_set = base_words | expanded

    # Decode real corpus with folio tracking
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

    real_decoded = _decode_corpus_r3(
        all_tokens, assignment, eva_to_triple,
        modifier_chars, modifier_rules, ref_word_set,
    )
    real_hits = [w in ref_word_set for w in real_decoded]

    # Regenerate and decode null corpora
    bigram_probs, initial_probs, token_lengths = _build_eva_bigram_model(
        all_tokens,
    )
    null_hits_list: List[List[bool]] = []
    for seed in null_seeds:
        null_tokens = _generate_null_corpus(
            bigram_probs, initial_probs, token_lengths, n_tokens, seed,
        )
        null_decoded = _decode_corpus_r3(
            null_tokens, assignment, eva_to_triple,
            modifier_chars, modifier_rules, ref_word_set,
        )
        null_hits_list.append([w in ref_word_set for w in null_decoded])

    # Classify each token
    classifications: List[str] = []
    for idx in range(n_tokens):
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

    dict_hits = [c in ('SIGNAL', 'SHARED_HIT') for c in classifications]

    ref_tokens = [w.lower() for w in ref_corpus.get_combined_tokens('latin')
                  if len(w) >= 2]

    return (
        token_folios, token_evas, real_decoded,
        classifications, dict_hits,
        ref_word_set, ref_tokens,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_bootstrap_bigrams() -> None:
    """Step 30.3: Post-bootstrap bigram plausibility test."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE 30.3: Post-Bootstrap Bigram Plausibility")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load assignment ──
    print("\n  1. Loading assignment …")
    boot_path = os.path.join(rd, 'bootstrap_loop.json')
    refine_path = os.path.join(rd, 'combined_refine.json')

    if os.path.exists(boot_path):
        with open(boot_path) as f:
            boot_data = json.load(f)
        assignment = boot_data.get('final_assignment', {})
        print(f"     Using bootstrap assignment")
    elif os.path.exists(refine_path):
        with open(refine_path) as f:
            refine_data = json.load(f)
        assignment = refine_data.get('best_assignment', {})
        print(f"     Using Phase 15 assignment (fallback)")
    else:
        print("  [SKIP] No assignment found")
        return

    # Load baseline for comparison
    baseline_z = 0.0
    baseline_hits = 0
    baseline_relaxed = 0
    bg_path = os.path.join(rd, 'signal_bigrams.json')
    if os.path.exists(bg_path):
        with open(bg_path) as f:
            bg_baseline = json.load(f)
        baseline_z = bg_baseline.get('bigram_z_score', 0.0)
        baseline_hits = bg_baseline.get('n_bigram_hits', 0)
        baseline_relaxed = bg_baseline.get('n_relaxed_bigram_hits', 0)

    # ── 2. Recompute per-token classifications with evolved assignment ──
    print("\n  2. Recomputing per-token classifications …")
    (
        token_folios, token_evas, token_decoded,
        token_classifications, token_dict_hits,
        ref_word_set, ref_tokens,
    ) = _recompute_with_assignment(rd, assignment)

    n_tokens = len(token_decoded)
    n_signal = sum(1 for c in token_classifications if c == 'SIGNAL')
    signal_rate = n_signal / n_tokens if n_tokens > 0 else 0.0

    print(f"     {n_tokens} tokens, {n_signal} SIGNAL ({signal_rate:.1%})")
    cls_counts = Counter(token_classifications)
    for cls in ['SIGNAL', 'SHARED_HIT', 'SHARED_MISS', 'ANTI_SIGNAL']:
        print(f"       {cls:14s}: {cls_counts.get(cls, 0):6d}")

    # ── 3. Build reference bigram/trigram table ──
    print("\n  3. Building reference bigram/trigram table …")
    ref_bigrams, ref_trigrams = _build_reference_bigrams(ref_tokens)
    print(f"     {len(ref_bigrams)} bigrams, {len(ref_trigrams)} trigrams")

    # ── 4. Find SIGNAL-SIGNAL pairs ──
    print("\n  4. Finding SIGNAL-SIGNAL pairs …")
    signal_pairs = _find_signal_pairs(
        token_classifications, token_decoded, token_folios,
    )
    print(f"     {len(signal_pairs)} consecutive SIGNAL-SIGNAL pairs")

    # ── 5. Bigram plausibility on SIGNAL pairs ──
    print("\n  5. Testing bigram plausibility on SIGNAL pairs …")
    n_bigram_hits = sum(
        1 for _, _, w1, w2 in signal_pairs
        if (w1, w2) in ref_bigrams
    )
    bigram_hit_rate = n_bigram_hits / len(signal_pairs) if signal_pairs else 0.0
    bigram_hit_pairs = [
        [w1, w2] for _, _, w1, w2 in signal_pairs
        if (w1, w2) in ref_bigrams
    ][:50]

    print(f"     Exact bigram hits: {n_bigram_hits}/{len(signal_pairs)} "
          f"({bigram_hit_rate:.4f})")

    # ── 6. Null permutation test ──
    print("\n  6. Null permutation test (1000 relabelings) …")
    null_rates, null_mean, null_std = _null_permutation_test(
        n_signal, n_tokens, token_decoded, token_folios,
        ref_bigrams, n_perms=1000, seed=42,
    )
    if null_std > 0:
        z_score = (bigram_hit_rate - null_mean) / null_std
    else:
        z_score = float('inf') if bigram_hit_rate > null_mean else 0.0

    p_value = sum(1 for r in null_rates if r >= bigram_hit_rate) / len(null_rates)
    z_display = round(z_score, 2) if z_score != float('inf') else 999.0

    print(f"     Null mean={null_mean:.6f}, std={null_std:.6f}")
    print(f"     z-score = {z_display}")
    print(f"     p-value = {p_value:.4f}")

    # ── 7. SIGNAL trigram test ──
    print("\n  7. Finding SIGNAL-SIGNAL-SIGNAL triples …")
    signal_triple_tuples = _find_signal_triples(
        token_classifications, token_decoded, token_folios,
    )
    n_trigram_hits = sum(
        1 for _, _, w1, w2, w3 in signal_triple_tuples
        if (w1, w2, w3) in ref_trigrams
    )
    trigram_hit_rate = (
        n_trigram_hits / len(signal_triple_tuples)
        if signal_triple_tuples else 0.0
    )
    trigram_hit_triples = [
        [w1, w2, w3] for _, _, w1, w2, w3 in signal_triple_tuples
        if (w1, w2, w3) in ref_trigrams
    ][:50]

    print(f"     {len(signal_triple_tuples)} SIGNAL triples, "
          f"{n_trigram_hits} trigram hits ({trigram_hit_rate:.4f})")

    # ── 8. Relaxed bigram test ──
    print("\n  8. Relaxed bigram test (edit distance 1) …")
    n_relaxed = _relaxed_bigram_test(signal_pairs, ref_bigrams, ref_word_set)
    relaxed_rate = (
        n_relaxed / len(signal_pairs) if signal_pairs else 0.0
    )
    print(f"     {n_relaxed} relaxed hits ({relaxed_rate:.4f})")

    # ── 9. Per-folio ranking ──
    print("\n  9. Per-folio signal-pair ranking …")
    folio_stats = _folio_signal_pair_ranking(
        signal_pairs, ref_bigrams, token_classifications, token_folios,
    )
    for fs in folio_stats[:5]:
        print(f"     {fs.folio:8s}  pairs={fs.n_signal_pairs:3d}  "
              f"hits={fs.n_bigram_hits:2d}  signal_rate={fs.signal_rate:.3f}")

    # ── 10. Comparison ──
    delta_z = z_display - baseline_z
    delta_hits = n_bigram_hits - baseline_hits
    delta_relaxed = n_relaxed - baseline_relaxed

    print(f"\n  10. Comparison to Phase 29 baseline …")
    print(f"     Bigram z: {baseline_z:.2f} → {z_display:.2f} (Δ={delta_z:+.2f})")
    print(f"     Exact hits: {baseline_hits} → {n_bigram_hits} (Δ={delta_hits:+d})")
    print(f"     Relaxed hits: {baseline_relaxed} → {n_relaxed} (Δ={delta_relaxed:+d})")

    # Gate
    gate = (z_display >= 4.0) or (z_display > 2.0 and delta_z >= -1.0)
    if z_display >= 6.0:
        verdict = f"BIGRAM_STRONG (z={z_display:.2f})"
    elif z_display >= 4.0:
        verdict = f"BIGRAM_MAINTAINED (z={z_display:.2f})"
    elif z_display >= 2.0:
        verdict = f"BIGRAM_WEAK (z={z_display:.2f})"
    else:
        verdict = f"BIGRAM_NONE (z={z_display:.2f})"

    print(f"\n     Verdict: {verdict}")
    print(f"     Gate: {'PASS' if gate else 'FAIL'}")

    result = BootstrapBigramResult(
        token_folios=token_folios,
        token_evas=token_evas,
        token_decoded=token_decoded,
        token_classifications=token_classifications,
        token_dict_hits=token_dict_hits,
        n_tokens=n_tokens,
        n_signal=n_signal,
        signal_rate=round(signal_rate, 6),
        ref_bigram_count=len(ref_bigrams),
        ref_trigram_count=len(ref_trigrams),
        n_signal_pairs=len(signal_pairs),
        n_bigram_hits=n_bigram_hits,
        bigram_hit_rate=round(bigram_hit_rate, 6),
        bigram_hit_pairs=bigram_hit_pairs,
        null_bigram_mean=round(null_mean, 6),
        null_bigram_std=round(null_std, 6),
        bigram_p_value=round(p_value, 4),
        bigram_z_score=z_display,
        n_signal_triples=len(signal_triple_tuples),
        n_trigram_hits=n_trigram_hits,
        trigram_hit_rate=round(trigram_hit_rate, 6),
        trigram_hit_triples=trigram_hit_triples,
        n_relaxed_bigram_hits=n_relaxed,
        relaxed_bigram_hit_rate=round(relaxed_rate, 6),
        folio_signal_pair_stats=[_convert(asdict(fs)) for fs in folio_stats[:30]],
        baseline_bigram_z=round(baseline_z, 2),
        baseline_n_bigram_hits=baseline_hits,
        baseline_n_relaxed=baseline_relaxed,
        delta_bigram_z=round(delta_z, 2),
        delta_n_bigram_hits=delta_hits,
        delta_n_relaxed=delta_relaxed,
        gate_passed=gate,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 1),
    )

    out_path = os.path.join(rd, 'bootstrap_bigrams.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)
    print(f"\n  → {out_path}")
