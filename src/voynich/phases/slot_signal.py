"""
Phase 34.7 – Slot-Conditioned Signal Pipeline
===============================================
Standard signal analysis pipeline applied to the slot-conditioned decode
from Step 34.6.  Decodes the real corpus and 5 null corpora through the
slot table, classifies every token as SIGNAL / SHARED_HIT / SHARED_MISS /
ANTI_SIGNAL, and computes the bigram z-score via null permutation test.

Compares results to Phase 29 baselines (SIGNAL=16.5%, z=6.14).

Dependency chain:
    slot_csp.json              (Step 34.6 — slot assignment)
    combined_refine.json       (Phase 15 — for null decode fallback)
    modifier_integrate.json    (Phase 16 modifiers)
    null_corpus.json           (Phase 17 seeds)
        -> slot_signal.json  (this step)
"""

import json
import os
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    load_corpus,
    tokenize_eva_chars,
)
from voynich.core.reference import (
    build_expanded_word_set,
    load_reference_corpus,
)
from voynich.phases.morpheme_grid import decompose_token_morphemes
from voynich.phases.null_corpus import (
    _build_eva_bigram_model,
    _generate_null_corpus,
    _reconstruct_modifier_rules,
)
from voynich.phases.signal_bigrams import (
    _build_reference_bigrams,
    _find_signal_pairs,
    _null_permutation_test,
)
from voynich.phases.slot_csp import (
    _decode_corpus_slotted,
    _decode_token_slotted,
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
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TokenClassificationStats:
    """Aggregate counts for each token classification."""
    n_signal: int
    n_shared_hit: int
    n_shared_miss: int
    n_anti_signal: int


@dataclass
class FolioSignalStats:
    """Per-folio signal statistics."""
    folio: str
    n_tokens: int
    n_signal: int
    signal_rate: float


@dataclass
class SlotSignalResult:
    """Full Step 34.7 output."""
    # Token classifications
    n_tokens: int
    n_signal: int
    signal_rate: float
    n_shared_hit: int
    n_shared_miss: int
    n_anti_signal: int
    # Dict hit
    dict_hit_rate: float
    n_dict_hits: int
    # Bigram test
    n_signal_pairs: int
    n_bigram_hits: int
    bigram_hit_rate: float
    bigram_hit_pairs: List[List[str]]
    null_bigram_mean: float
    null_bigram_std: float
    bigram_z_score: float
    bigram_p_value: float
    # Phase 29 baselines
    phase29_signal_rate: float
    phase29_bigram_z: float
    delta_signal_rate: float
    delta_bigram_z: float
    # Per-folio ranking
    top_signal_folios: List[Dict]
    # Null corpora info
    null_n_corpora: int
    null_seeds: List[int]
    null_dict_hits: List[float]
    # Per-token cache (parallel arrays) — truncated to save space
    token_classifications: List[str]
    token_dict_hits: List[bool]
    token_folios: List[str]
    token_decoded: List[str]
    gate_passed: bool
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Slot-aware null decode
# ---------------------------------------------------------------------------

def _decode_null_corpus_slotted(
    null_tokens: List[str],
    prefix_map: Dict[str, str],
    root_map: Dict[str, str],
    suffix_map: Dict[str, str],
    eva_to_triple: Dict[str, str],
) -> List[str]:
    """Decode a null corpus through the slot-conditioned table.

    Null tokens are random EVA sequences, so morpheme decomposition may
    not produce meaningful prefix/root/suffix splits.  We still apply the
    same decomposition rules for consistency.
    """
    return _decode_corpus_slotted(
        null_tokens, prefix_map, root_map, suffix_map, eva_to_triple,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_slot_signal() -> None:
    """Step 34.7: Signal pipeline on slot-conditioned decode."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE 34.7: Slot-Conditioned Signal Pipeline")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load slot CSP assignment ──
    print("\n  1. Loading slot CSP assignment ...")
    csp_path = os.path.join(rd, 'slot_csp.json')
    if not os.path.exists(csp_path):
        print("  [SKIP] slot_csp.json not found -- run slot-csp first")
        return
    with open(csp_path) as f:
        csp_data = json.load(f)

    prefix_map = csp_data.get('prefix_assignment', {})
    root_map = csp_data.get('root_assignment', {})
    suffix_map = csp_data.get('suffix_assignment', {})
    print(f"     PREFIX: {len(prefix_map)}, ROOT: {len(root_map)}, "
          f"SUFFIX: {len(suffix_map)} assignments")

    # ── 2. Load modifiers ──
    print("\n  2. Loading modifier rules ...")
    mod_path = os.path.join(rd, 'modifier_integrate.json')
    if not os.path.exists(mod_path):
        print("  [SKIP] modifier_integrate.json not found")
        return
    with open(mod_path) as f:
        mod_data = json.load(f)
    modifier_chars, modifier_rules = _reconstruct_modifier_rules(mod_data)

    # ── 3. Load null seeds ──
    null_seeds = [100, 101, 102, 103, 104]
    null_path = os.path.join(rd, 'null_corpus.json')
    if os.path.exists(null_path):
        with open(null_path) as f:
            null_data = json.load(f)
        null_seeds = [r['seed'] for r in null_data.get('null_runs', [])]
    print(f"     Null seeds: {null_seeds}")

    # ── 4. Build reference word set ──
    print("\n  4. Building reference word set ...")
    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    base_words = set(
        w.lower() for w in ref_corpus.get_combined_tokens('latin')
        if len(w) >= 2
    )
    expanded, _ = build_expanded_word_set(base_words)
    ref_word_set = base_words | expanded
    ref_tokens = [
        w.lower() for w in ref_corpus.get_combined_tokens('latin')
        if len(w) >= 2
    ]
    print(f"     {len(ref_word_set)} reference words")

    # ── 5. Decode real corpus ──
    print("\n  5. Decoding real corpus ...")
    corpus = load_corpus(verbose=False)
    eva_to_triple = build_eva_to_triple_lookup()

    # Build token list with folio tracking
    token_folios: List[str] = []
    token_evas: List[str] = []
    all_tokens: List[str] = []

    for folio, page in corpus.pages.items():
        for token in page.all_tokens:
            token_folios.append(folio)
            token_evas.append(token)
            all_tokens.append(token)

    n_tokens = len(all_tokens)

    real_decoded = _decode_corpus_slotted(
        all_tokens, prefix_map, root_map, suffix_map, eva_to_triple,
    )
    real_hits = [w in ref_word_set for w in real_decoded]
    real_hit_rate = sum(real_hits) / n_tokens if n_tokens > 0 else 0.0
    n_dict_hits = sum(real_hits)
    print(f"     {n_tokens} tokens, dict_hit = {real_hit_rate:.4f} "
          f"({n_dict_hits} hits)")

    # ── 6. Regenerate and decode null corpora ──
    print("\n  6. Regenerating and decoding null corpora ...")
    bigram_probs, initial_probs, token_lengths = _build_eva_bigram_model(all_tokens)

    null_hits_list: List[List[bool]] = []
    null_dict_hits: List[float] = []

    for i, seed in enumerate(null_seeds):
        print(f"     Null corpus {i + 1}/{len(null_seeds)} (seed={seed}) ...")
        null_tokens = _generate_null_corpus(
            bigram_probs, initial_probs, token_lengths, n_tokens, seed,
        )
        null_decoded = _decode_null_corpus_slotted(
            null_tokens, prefix_map, root_map, suffix_map, eva_to_triple,
        )
        null_h = [w in ref_word_set for w in null_decoded]
        null_hits_list.append(null_h)
        null_rate = sum(null_h) / len(null_h) if null_h else 0.0
        null_dict_hits.append(round(null_rate, 6))
        print(f"       dict_hit = {null_rate:.4f}")

    # ── 7. Token-level classification ──
    print("\n  7. Token-level classification ...")
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

    n_signal = sum(1 for c in classifications if c == 'SIGNAL')
    n_shared_hit = sum(1 for c in classifications if c == 'SHARED_HIT')
    n_shared_miss = sum(1 for c in classifications if c == 'SHARED_MISS')
    n_anti_signal = sum(1 for c in classifications if c == 'ANTI_SIGNAL')
    signal_rate = n_signal / n_tokens if n_tokens > 0 else 0.0

    print(f"     SIGNAL:      {n_signal:6d} ({signal_rate:.1%})")
    print(f"     SHARED_HIT:  {n_shared_hit:6d}")
    print(f"     SHARED_MISS: {n_shared_miss:6d}")
    print(f"     ANTI_SIGNAL: {n_anti_signal:6d}")

    dict_hits_bool = [c in ('SIGNAL', 'SHARED_HIT') for c in classifications]

    # ── 8. Build reference bigrams ──
    print("\n  8. Building reference bigram table ...")
    ref_bigrams, ref_trigrams = _build_reference_bigrams(ref_tokens)
    print(f"     {len(ref_bigrams)} unique bigrams")

    # ── 9. Find SIGNAL-SIGNAL pairs ──
    print("\n  9. Finding SIGNAL-SIGNAL pairs ...")
    signal_pairs = _find_signal_pairs(
        classifications, real_decoded, token_folios,
    )
    print(f"     {len(signal_pairs)} consecutive SIGNAL-SIGNAL pairs")

    # ── 10. Bigram plausibility on SIGNAL pairs ──
    print("\n  10. Testing bigram plausibility ...")
    bigram_hits: List[List[str]] = []
    for folio, pos, w1, w2 in signal_pairs:
        if (w1, w2) in ref_bigrams:
            bigram_hits.append([w1, w2])

    n_bigram_hits = len(bigram_hits)
    bigram_hit_rate = n_bigram_hits / len(signal_pairs) if signal_pairs else 0.0
    print(f"     {n_bigram_hits} bigram hits out of {len(signal_pairs)} pairs")
    print(f"     Bigram hit rate: {bigram_hit_rate:.4f}")

    if bigram_hits:
        print("     Matching pairs:")
        for pair in bigram_hits[:15]:
            print(f"       {pair[0]} {pair[1]}")

    # ── 11. Null permutation test ──
    print("\n  11. Null permutation test (1000 permutations) ...")
    null_rates, null_mean, null_std = _null_permutation_test(
        n_signal, n_tokens, real_decoded, token_folios,
        ref_bigrams, n_perms=1000, seed=42,
    )

    if null_std > 0:
        z_score = (bigram_hit_rate - null_mean) / null_std
    else:
        z_score = float('inf') if bigram_hit_rate > null_mean else 0.0

    p_value = sum(1 for r in null_rates if r >= bigram_hit_rate) / len(null_rates)
    print(f"     Null mean: {null_mean:.6f}, std: {null_std:.6f}")
    print(f"     z-score: {z_score:.2f}, p-value: {p_value:.4f}")

    # ── 12. Per-folio signal ranking ──
    print("\n  12. Per-folio signal ranking (top 10) ...")
    folio_n: Dict[str, int] = Counter(token_folios)
    folio_n_signal: Dict[str, int] = Counter()
    for folio, cls in zip(token_folios, classifications):
        if cls == 'SIGNAL':
            folio_n_signal[folio] += 1

    folio_stats: List[FolioSignalStats] = []
    for folio in sorted(folio_n.keys()):
        n_f = folio_n[folio]
        n_s = folio_n_signal.get(folio, 0)
        folio_stats.append(FolioSignalStats(
            folio=folio,
            n_tokens=n_f,
            n_signal=n_s,
            signal_rate=round(n_s / n_f, 4) if n_f > 0 else 0.0,
        ))
    folio_stats.sort(key=lambda f: -f.signal_rate)

    for fs in folio_stats[:10]:
        print(f"     {fs.folio:8s}  {fs.n_signal:3d}/{fs.n_tokens:3d}  "
              f"({fs.signal_rate:.1%})")

    # ── 13. Phase 29 comparison ──
    print("\n  13. Phase 29 comparison ...")
    phase29_signal_rate = 0.165
    phase29_bigram_z = 6.14
    delta_signal = signal_rate - phase29_signal_rate
    delta_z = (z_score - phase29_bigram_z) if z_score != float('inf') else 999.0
    print(f"     Phase 29 SIGNAL rate: {phase29_signal_rate:.1%}")
    print(f"     Slot SIGNAL rate:     {signal_rate:.1%} ({delta_signal:+.1%})")
    print(f"     Phase 29 bigram z:    {phase29_bigram_z:.2f}")
    print(f"     Slot bigram z:        {z_score:.2f} ({delta_z:+.2f})")

    # ── 14. Gate and verdict ──
    z_finite = z_score if z_score != float('inf') else 999.0
    gate_passed = z_finite > 2.0 and signal_rate > 0.10

    if z_finite > phase29_bigram_z and signal_rate > phase29_signal_rate:
        verdict = (
            f"IMPROVEMENT: SIGNAL={signal_rate:.1%} (vs {phase29_signal_rate:.1%}), "
            f"bigram z={z_finite:.2f} (vs {phase29_bigram_z:.2f}). "
            f"Slot-conditioning improves signal structure."
        )
    elif z_finite > 2.0:
        verdict = (
            f"SIGNAL_PRESERVED: SIGNAL={signal_rate:.1%}, bigram z={z_finite:.2f}. "
            f"Signal structure maintained under slot-conditioning "
            f"(Phase 29: {phase29_signal_rate:.1%}, z={phase29_bigram_z:.2f})."
        )
    else:
        verdict = (
            f"SIGNAL_LOST: SIGNAL={signal_rate:.1%}, bigram z={z_finite:.2f}. "
            f"Slot-conditioning disrupts signal structure "
            f"(Phase 29: {phase29_signal_rate:.1%}, z={phase29_bigram_z:.2f})."
        )

    print(f"\n  Gate: {'PASS' if gate_passed else 'FAIL'}")
    print(f"  {verdict}")

    # ── 15. Save ──
    result = SlotSignalResult(
        n_tokens=n_tokens,
        n_signal=n_signal,
        signal_rate=round(signal_rate, 4),
        n_shared_hit=n_shared_hit,
        n_shared_miss=n_shared_miss,
        n_anti_signal=n_anti_signal,
        dict_hit_rate=round(real_hit_rate, 6),
        n_dict_hits=n_dict_hits,
        n_signal_pairs=len(signal_pairs),
        n_bigram_hits=n_bigram_hits,
        bigram_hit_rate=round(bigram_hit_rate, 6),
        bigram_hit_pairs=bigram_hits[:50],
        null_bigram_mean=round(null_mean, 6),
        null_bigram_std=round(null_std, 6),
        bigram_z_score=round(z_finite, 2),
        bigram_p_value=round(p_value, 4),
        phase29_signal_rate=phase29_signal_rate,
        phase29_bigram_z=phase29_bigram_z,
        delta_signal_rate=round(delta_signal, 4),
        delta_bigram_z=round(delta_z, 2),
        top_signal_folios=[_convert(asdict(fs)) for fs in folio_stats[:20]],
        null_n_corpora=len(null_seeds),
        null_seeds=null_seeds,
        null_dict_hits=null_dict_hits,
        token_classifications=classifications,
        token_dict_hits=dict_hits_bool,
        token_folios=token_folios,
        token_decoded=real_decoded,
        gate_passed=gate_passed,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'slot_signal.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)
    print(f"\n  -> {out_path}")
