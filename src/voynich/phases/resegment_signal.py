"""
Phase 34.13 – Signal Pipeline on Re-Segmented Decode
=======================================================
Runs the standard signal isolation and bigram plausibility pipeline on
the Viterbi re-segmented text from Step 34.12.  Generates 5 null corpora,
decodes them through the same pipeline (decode → strip spaces → Viterbi
re-segment), classifies tokens as SIGNAL/SHARED_HIT/SHARED_MISS/
ANTI_SIGNAL, and computes bigram z-score.  Compares to Phase 29 baselines.

Dependency chain:
    combined_refine.json       (Phase 15 assignment)
    modifier_integrate.json    (Phase 16 modifiers)
    null_corpus.json           (Phase 17 seeds)
    signal_bigrams.json        (Phase 29 — baseline for comparison)
        → resegment_signal.json   (this step)
"""

import json
import math
import os
import random
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
from voynich.phases.null_corpus import (
    _build_eva_bigram_model,
    _generate_null_corpus,
    _reconstruct_modifier_rules,
)
from voynich.phases.signal_isolation import _decode_corpus_r3
from voynich.phases.signal_bigrams import (
    _build_reference_bigrams,
    _find_signal_pairs,
    _null_permutation_test,
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
# Viterbi segmentation (duplicated from resegment_decode to avoid
# circular dependency — this module must be self-contained)
# ---------------------------------------------------------------------------

def _build_unigram_model(
    base_words: set,
    ref_tokens: List[str],
) -> Tuple[Dict[str, float], float]:
    """Build unigram log-probability model from base Latin dictionary."""
    word_counts: Counter = Counter()
    for token in ref_tokens:
        w = token.lower()
        if w in base_words:
            word_counts[w] += 1

    floor_count = 1
    for w in base_words:
        if w not in word_counts:
            word_counts[w] = floor_count

    total = sum(word_counts.values())
    word_log_probs: Dict[str, float] = {}
    for w, count in word_counts.items():
        word_log_probs[w] = math.log10(count / total)

    unknown_log_prob = math.log10(1e-8)
    return word_log_probs, unknown_log_prob


def _viterbi_segment(
    stream: str,
    word_log_probs: Dict[str, float],
    unknown_log_prob: float,
    max_word_len: int = 15,
) -> List[str]:
    """Viterbi DP to find optimal word segmentation."""
    n = len(stream)
    if n == 0:
        return []

    INF = float('-inf')
    dp: List[Tuple[float, int]] = [(INF, -1)] * (n + 1)
    dp[0] = (0.0, 0)

    for i in range(1, n + 1):
        best_prob = INF
        best_j = 0
        lo = max(0, i - max_word_len)
        for j in range(lo, i):
            if dp[j][0] == INF:
                continue
            substr = stream[j:i]
            lp = word_log_probs.get(substr, unknown_log_prob)
            total = dp[j][0] + lp
            if total > best_prob:
                best_prob = total
                best_j = j
        dp[i] = (best_prob, best_j)

    words: List[str] = []
    pos = n
    while pos > 0:
        j = dp[pos][1]
        words.append(stream[j:pos])
        pos = j
    words.reverse()
    return words


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TokenClassCounts:
    n_signal: int
    n_shared_hit: int
    n_shared_miss: int
    n_anti_signal: int


@dataclass
class ResegmentSignalResult:
    # Token counts
    n_real_tokens: int
    n_real_signal: int
    real_signal_rate: float
    real_dict_hit: float
    token_classification: Dict

    # Bigram plausibility
    n_signal_pairs: int
    n_bigram_hits: int
    bigram_hit_rate: float
    bigram_z_score: float
    bigram_p_value: float

    # Null corpora summary
    null_n_corpora: int
    null_seeds: List[int]
    null_dict_hit_mean: float
    null_dict_hit_std: float
    null_signal_rate_mean: float

    # Phase 29 comparison
    baseline_signal_rate: float
    baseline_bigram_z: float
    delta_signal_rate: float
    delta_bigram_z: float

    # Top signal folios
    top_signal_folios: List[Dict]

    # Sample bigram hits
    bigram_hit_pairs: List[List[str]]

    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_resegment_signal() -> None:
    """Step 34.13: Signal pipeline on re-segmented decoded text."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE 34.13: Re-Segmented Signal Analysis")
    print("=" * 70)

    rd = _results_dir()
    eva_to_triple = build_eva_to_triple_lookup()

    # ── 1. Load inputs ──
    print("\n  1. Loading inputs …")

    refine_path = os.path.join(rd, 'combined_refine.json')
    if not os.path.exists(refine_path):
        print("  [SKIP] combined_refine.json not found")
        return
    with open(refine_path) as f:
        refine_data = json.load(f)
    assignment = refine_data.get('best_assignment', {})

    mod_path = os.path.join(rd, 'modifier_integrate.json')
    if not os.path.exists(mod_path):
        print("  [SKIP] modifier_integrate.json not found")
        return
    with open(mod_path) as f:
        mod_data = json.load(f)
    modifier_chars, modifier_rules = _reconstruct_modifier_rules(mod_data)

    null_seeds = [100, 101, 102, 103, 104]
    null_path = os.path.join(rd, 'null_corpus.json')
    if os.path.exists(null_path):
        with open(null_path) as f:
            null_data = json.load(f)
        null_seeds = [r['seed'] for r in null_data.get('null_runs', [])]

    # Phase 29 baselines
    baseline_signal_rate = 0.165   # 16.5%
    baseline_bigram_z = 6.14
    sig_bigram_path = os.path.join(rd, 'signal_bigrams.json')
    if os.path.exists(sig_bigram_path):
        with open(sig_bigram_path) as f:
            sb_data = json.load(f)
        baseline_signal_rate = sb_data.get('signal_rate', 0.165)
        baseline_bigram_z = sb_data.get('bigram_z_score', 6.14)

    print(f"     Assignment: {len(assignment)} triples")
    print(f"     Modifiers: {len(modifier_chars)} chars")
    print(f"     Null seeds: {null_seeds}")
    print(f"     Phase 29 baseline: signal_rate={baseline_signal_rate:.4f}, "
          f"bigram_z={baseline_bigram_z:.2f}")

    # ── 2. Build reference word sets ──
    print("\n  2. Building reference word sets …")
    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    ref_tokens = [
        w.lower() for w in ref_corpus.get_combined_tokens('latin')
        if len(w) >= 2
    ]
    base_words = set(ref_tokens)
    expanded, _ = build_expanded_word_set(base_words)
    ref_word_set = base_words | expanded

    # Build unigram model for Viterbi (17K base dict)
    word_log_probs, unknown_log_prob = _build_unigram_model(base_words, ref_tokens)
    print(f"     Base dict: {len(base_words)} words")
    print(f"     Expanded dict: {len(ref_word_set)} words")

    # Build reference bigrams for plausibility test
    ref_bigrams, _ = _build_reference_bigrams(ref_tokens)
    print(f"     Reference bigrams: {len(ref_bigrams)}")

    # ── 3. Decode real corpus → continuous → Viterbi re-segment ──
    print("\n  3. Decoding and re-segmenting real corpus …")
    corpus = load_corpus(verbose=False)

    # Collect all tokens with folio tracking
    all_eva_tokens: List[str] = []
    all_folios_flat: List[str] = []
    folio_token_ranges: Dict[str, Tuple[int, int]] = {}

    idx = 0
    for folio, page in corpus.pages.items():
        tokens = page.all_tokens
        start = idx
        for token in tokens:
            all_eva_tokens.append(token)
            all_folios_flat.append(folio)
            idx += 1
        folio_token_ranges[folio] = (start, idx)

    n_eva_tokens = len(all_eva_tokens)

    # Decode all tokens
    real_decoded = _decode_corpus_r3(
        all_eva_tokens, assignment, eva_to_triple,
        modifier_chars, modifier_rules, ref_word_set,
    )

    # Re-segment per folio: decode → strip spaces → Viterbi
    real_viterbi_words: List[str] = []
    real_viterbi_folios: List[str] = []

    for folio, page in corpus.pages.items():
        if folio not in folio_token_ranges:
            continue
        start, end = folio_token_ranges[folio]
        folio_decoded = real_decoded[start:end]

        # Continuous stream
        continuous = ''.join(folio_decoded)

        # Viterbi re-segment
        viterbi_words = _viterbi_segment(
            continuous, word_log_probs, unknown_log_prob,
        )
        real_viterbi_words.extend(viterbi_words)
        for _ in viterbi_words:
            real_viterbi_folios.append(folio)

    n_real_tokens = len(real_viterbi_words)
    real_hits = [w in ref_word_set for w in real_viterbi_words]
    real_dict_hit = sum(real_hits) / n_real_tokens if n_real_tokens > 0 else 0.0

    print(f"     {n_eva_tokens} EVA tokens → {n_real_tokens} Viterbi tokens")
    print(f"     Re-segmented dict_hit: {real_dict_hit:.4f}")

    # ── 4. Generate and process null corpora through same pipeline ──
    print("\n  4. Generating null corpora through decode → Viterbi pipeline …")
    bigram_probs, initial_probs, token_lengths = _build_eva_bigram_model(
        all_eva_tokens,
    )

    null_viterbi_hits_list: List[List[bool]] = []
    null_dict_hits: List[float] = []

    for i, seed in enumerate(null_seeds):
        print(f"     Null corpus {i + 1}/{len(null_seeds)} (seed={seed}) …")

        # Generate null EVA tokens
        null_tokens = _generate_null_corpus(
            bigram_probs, initial_probs, token_lengths, n_eva_tokens, seed,
        )

        # Decode null tokens
        null_decoded = _decode_corpus_r3(
            null_tokens, assignment, eva_to_triple,
            modifier_chars, modifier_rules, ref_word_set,
        )

        # Build continuous stream per folio and Viterbi re-segment
        # For null corpora, we don't have folio structure — treat as single stream
        # but split into chunks matching real folio sizes for fair comparison
        null_viterbi_words: List[str] = []
        null_idx = 0
        for folio, page in corpus.pages.items():
            if folio not in folio_token_ranges:
                continue
            start, end = folio_token_ranges[folio]
            folio_size = end - start
            chunk_end = min(null_idx + folio_size, len(null_decoded))
            folio_null_decoded = null_decoded[null_idx:chunk_end]
            null_idx = chunk_end

            continuous = ''.join(folio_null_decoded)
            vit_words = _viterbi_segment(
                continuous, word_log_probs, unknown_log_prob,
            )
            null_viterbi_words.extend(vit_words)

        # Pad or truncate to match real token count for per-position comparison
        # Since Viterbi produces different numbers of tokens, we align by
        # computing hit rate rather than per-position comparison
        null_hits = [w in ref_word_set for w in null_viterbi_words]
        null_dict_hit = (
            sum(null_hits) / len(null_hits) if null_hits else 0.0
        )
        null_dict_hits.append(null_dict_hit)
        print(f"       {len(null_viterbi_words)} tokens, "
              f"dict_hit = {null_dict_hit:.4f}")

        # For token-level classification, we need position-aligned hits.
        # Pad/truncate null hits to match real token count.
        if len(null_hits) < n_real_tokens:
            null_hits.extend([False] * (n_real_tokens - len(null_hits)))
        else:
            null_hits = null_hits[:n_real_tokens]
        null_viterbi_hits_list.append(null_hits)

    # ── 5. Token-level classification ──
    print("\n  5. Token-level classification …")
    n_signal = 0
    n_shared_hit = 0
    n_shared_miss = 0
    n_anti_signal = 0

    classifications: List[str] = []
    for idx in range(n_real_tokens):
        r_hit = real_hits[idx]
        null_hit_count = sum(
            1 for nh in null_viterbi_hits_list if nh[idx]
        )

        if r_hit and null_hit_count <= 1:
            n_signal += 1
            classifications.append('SIGNAL')
        elif r_hit and null_hit_count >= 3:
            n_shared_hit += 1
            classifications.append('SHARED_HIT')
        elif not r_hit and null_hit_count >= 3:
            n_anti_signal += 1
            classifications.append('ANTI_SIGNAL')
        else:
            n_shared_miss += 1
            classifications.append('SHARED_MISS')

    signal_rate = n_signal / n_real_tokens if n_real_tokens > 0 else 0.0

    print(f"     SIGNAL:      {n_signal:6d} ({signal_rate:.1%})")
    print(f"     SHARED_HIT:  {n_shared_hit:6d}")
    print(f"     SHARED_MISS: {n_shared_miss:6d}")
    print(f"     ANTI_SIGNAL: {n_anti_signal:6d}")

    # ── 6. Bigram plausibility on SIGNAL pairs ──
    print("\n  6. Bigram plausibility on SIGNAL-SIGNAL pairs …")
    signal_pairs = _find_signal_pairs(
        classifications, real_viterbi_words, real_viterbi_folios,
    )
    print(f"     {len(signal_pairs)} consecutive SIGNAL-SIGNAL pairs")

    bigram_hits: List[List[str]] = []
    for folio, pos, w1, w2 in signal_pairs:
        if (w1, w2) in ref_bigrams:
            bigram_hits.append([w1, w2])

    n_bigram_hits = len(bigram_hits)
    bigram_hit_rate = (
        n_bigram_hits / len(signal_pairs) if signal_pairs else 0.0
    )
    print(f"     {n_bigram_hits} bigram hits (rate={bigram_hit_rate:.4f})")

    if bigram_hits:
        print("     Matching pairs:")
        for pair in bigram_hits[:10]:
            print(f"       {pair[0]} {pair[1]}")

    # ── 7. Null permutation test ──
    print("\n  7. Null permutation test (1000 permutations) …")
    null_rates, null_mean, null_std = _null_permutation_test(
        n_signal, n_real_tokens, real_viterbi_words, real_viterbi_folios,
        ref_bigrams, n_perms=1000, seed=42,
    )

    if null_std > 0:
        z_score = (bigram_hit_rate - null_mean) / null_std
    else:
        z_score = float('inf') if bigram_hit_rate > null_mean else 0.0

    p_value = (
        sum(1 for r in null_rates if r >= bigram_hit_rate) / len(null_rates)
        if null_rates else 1.0
    )

    print(f"     Null mean: {null_mean:.6f}, std: {null_std:.6f}")
    print(f"     z-score: {z_score:.2f}, p-value: {p_value:.4f}")

    # ── 8. Per-folio signal distribution ──
    print("\n  8. Per-folio signal distribution (top 10) …")
    folio_n: Dict[str, int] = Counter(real_viterbi_folios)
    folio_n_signal: Dict[str, int] = Counter()
    for folio, cls in zip(real_viterbi_folios, classifications):
        if cls == 'SIGNAL':
            folio_n_signal[folio] += 1

    folio_stats: List[Dict] = []
    for folio in sorted(folio_n.keys()):
        n_tok = folio_n[folio]
        n_sig = folio_n_signal.get(folio, 0)
        sr = n_sig / n_tok if n_tok > 0 else 0.0
        folio_stats.append({
            'folio': folio,
            'n_tokens': n_tok,
            'n_signal': n_sig,
            'signal_rate': round(sr, 4),
        })
    folio_stats.sort(key=lambda f: -f['signal_rate'])

    for fs in folio_stats[:10]:
        print(f"     {fs['folio']:8s}  {fs['n_signal']:3d}/{fs['n_tokens']:3d}  "
              f"({fs['signal_rate']:.1%})")

    # ── 9. Comparison to Phase 29 baselines ──
    print("\n  9. Comparison to Phase 29 baselines …")
    delta_signal_rate = signal_rate - baseline_signal_rate
    delta_bigram_z = z_score - baseline_bigram_z

    print(f"     Signal rate: {baseline_signal_rate:.4f} → {signal_rate:.4f} "
          f"(Δ={delta_signal_rate:+.4f})")
    z_display = z_score if z_score != float('inf') else 999.0
    print(f"     Bigram z:    {baseline_bigram_z:.2f} → {z_display:.2f} "
          f"(Δ={delta_bigram_z:+.2f})")

    # ── 10. Null dict-hit summary ──
    null_mean_dh = (
        sum(null_dict_hits) / len(null_dict_hits) if null_dict_hits else 0.0
    )
    null_std_dh = (
        (sum((x - null_mean_dh) ** 2 for x in null_dict_hits)
         / len(null_dict_hits)) ** 0.5
        if null_dict_hits else 0.0
    )
    null_signal_rate_mean = (
        sum(null_dict_hits) / len(null_dict_hits) if null_dict_hits else 0.0
    )

    print(f"\n     Null dict_hit: {null_mean_dh:.4f} ± {null_std_dh:.4f}")

    # ── 11. Verdict ──
    z_safe = z_score if z_score != float('inf') else 999.0

    if z_safe > 3.0 and signal_rate > baseline_signal_rate:
        verdict = (
            f"RESEG_SIGNAL_STRONG: z={z_safe:.2f} (above null), "
            f"signal_rate={signal_rate:.3f} exceeds Phase 29 baseline "
            f"({baseline_signal_rate:.3f}). Re-segmentation preserves or "
            f"improves signal structure."
        )
    elif z_safe > 2.0:
        verdict = (
            f"RESEG_SIGNAL_PRESENT: z={z_safe:.2f} (above null), "
            f"signal_rate={signal_rate:.3f} vs baseline {baseline_signal_rate:.3f}. "
            f"Re-segmented text retains genuine sequential signal."
        )
    elif z_safe > 0:
        verdict = (
            f"RESEG_SIGNAL_WEAK: z={z_safe:.2f} (marginal), "
            f"signal_rate={signal_rate:.3f}. Re-segmentation weakens "
            f"but does not eliminate signal."
        )
    else:
        verdict = (
            f"RESEG_NO_SIGNAL: z={z_safe:.2f}, "
            f"signal_rate={signal_rate:.3f}. Re-segmentation destroys "
            f"sequential signal — EVA spaces are structurally important."
        )

    print(f"\n  Verdict: {verdict}")

    # ── 12. Save ──
    result = ResegmentSignalResult(
        n_real_tokens=n_real_tokens,
        n_real_signal=n_signal,
        real_signal_rate=round(signal_rate, 4),
        real_dict_hit=round(real_dict_hit, 4),
        token_classification={
            'n_signal': n_signal,
            'n_shared_hit': n_shared_hit,
            'n_shared_miss': n_shared_miss,
            'n_anti_signal': n_anti_signal,
        },
        n_signal_pairs=len(signal_pairs),
        n_bigram_hits=n_bigram_hits,
        bigram_hit_rate=round(bigram_hit_rate, 6),
        bigram_z_score=round(z_safe, 2),
        bigram_p_value=round(p_value, 4),
        null_n_corpora=len(null_seeds),
        null_seeds=null_seeds,
        null_dict_hit_mean=round(null_mean_dh, 4),
        null_dict_hit_std=round(null_std_dh, 4),
        null_signal_rate_mean=round(null_signal_rate_mean, 4),
        baseline_signal_rate=round(baseline_signal_rate, 4),
        baseline_bigram_z=round(baseline_bigram_z, 2),
        delta_signal_rate=round(delta_signal_rate, 4),
        delta_bigram_z=round(delta_bigram_z if delta_bigram_z != float('inf') else 999.0, 2),
        top_signal_folios=folio_stats[:20],
        bigram_hit_pairs=bigram_hits[:50],
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'resegment_signal.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)
    print(f"\n  → {out_path}")
