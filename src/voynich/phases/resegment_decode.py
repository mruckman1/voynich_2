"""
Phase 34.12 – Viterbi Re-Segmentation of Continuous Decoded Stream
====================================================================
Applies a Viterbi word-boundary model to the continuous decoded stream
produced by Step 34.11.  Uses unigram word frequencies from the 17K base
Latin dictionary (NOT the 131K expanded set) to find the optimal
segmentation, then compares dict-hit rate and word-length distribution
to the original EVA-space segmentation.

Dependency chain:
    combined_refine.json       (Phase 15 assignment)
    modifier_integrate.json    (Phase 16 modifiers)
    continua_stream.json       (Step 34.11 — optional, for reference)
        → resegment_decode.json   (this step)
"""

import json
import math
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

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
from voynich.phases.null_corpus import _reconstruct_modifier_rules
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
# Unigram word model
# ---------------------------------------------------------------------------

def _build_unigram_model(
    base_words: set,
    ref_tokens: List[str],
) -> Tuple[Dict[str, float], float]:
    """Build unigram log-probability model from the 17K base Latin dictionary.

    Uses word frequencies from the reference corpus to estimate P(word).
    Words in the dictionary but not in the corpus get a floor probability.

    Returns:
        word_log_probs: {word: log10(P(word))}
        unknown_log_prob: log10(penalty) for unknown substrings
    """
    # Count word frequencies in reference corpus
    word_counts: Counter = Counter()
    for token in ref_tokens:
        w = token.lower()
        if w in base_words:
            word_counts[w] += 1

    # Add floor count for dictionary words that never appeared
    floor_count = 1
    for w in base_words:
        if w not in word_counts:
            word_counts[w] = floor_count

    total = sum(word_counts.values())

    word_log_probs: Dict[str, float] = {}
    for w, count in word_counts.items():
        word_log_probs[w] = math.log10(count / total)

    # Unknown word penalty
    unknown_log_prob = math.log10(1e-8)

    return word_log_probs, unknown_log_prob


# ---------------------------------------------------------------------------
# Viterbi word segmentation
# ---------------------------------------------------------------------------

def _viterbi_segment(
    stream: str,
    word_log_probs: Dict[str, float],
    unknown_log_prob: float,
    max_word_len: int = 15,
) -> List[str]:
    """Viterbi dynamic programming to find optimal word segmentation.

    dp[i] = (best_log_prob for stream[0:i], backpointer)
    At each position i, try all substrings stream[j:i] for j in [max(0, i-max_word_len) .. i-1].

    Returns the list of words in the best segmentation.
    """
    n = len(stream)
    if n == 0:
        return []

    # dp[i] = (best cumulative log prob for stream[:i], backpointer j)
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

    # Backtrack
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
class FolioResegStats:
    folio: str
    section: str
    n_original_tokens: int
    n_viterbi_tokens: int
    original_dict_hit: float
    viterbi_dict_hit: float
    dict_hit_delta: float
    original_mean_word_len: float
    viterbi_mean_word_len: float


@dataclass
class ResegmentDecodeResult:
    # Corpus-wide statistics
    n_folios: int
    n_total_original_tokens: int
    n_total_viterbi_tokens: int

    # Phase 16 (EVA spaces) baseline
    original_dict_hit: float
    original_mean_word_len: float

    # Viterbi re-segmentation
    viterbi_dict_hit: float
    viterbi_mean_word_len: float
    dict_hit_delta: float
    mean_word_len_delta: float

    # Dictionary info
    n_base_dict_words: int
    n_expanded_dict_words: int
    unigram_vocab_size: int
    unknown_log_prob: float

    # Per-folio stats
    per_folio_stats: List[Dict]
    top_improved_folios: List[Dict]
    top_degraded_folios: List[Dict]

    # Sample segmentations
    sample_segmentations: List[Dict]

    # Herbal re-test (botanical label check)
    herbal_viterbi_dict_hit: float
    herbal_original_dict_hit: float
    herbal_mean_viterbi_word_len: float

    # Viterbi token list (for downstream use, first 500)
    viterbi_tokens_sample: List[str]
    viterbi_folios_sample: List[str]

    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_resegment_decode() -> None:
    """Step 34.12: Viterbi re-segmentation of continuous decoded stream."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE 34.12: Viterbi Re-Segmentation Decode")
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

    print(f"     Assignment: {len(assignment)} triples")
    print(f"     Modifiers: {len(modifier_chars)} chars")

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

    # Build unigram model from 17K base dict only (NOT expanded)
    word_log_probs, unknown_log_prob = _build_unigram_model(base_words, ref_tokens)
    print(f"     Base dict: {len(base_words)} words")
    print(f"     Expanded dict: {len(ref_word_set)} words")
    print(f"     Unigram model: {len(word_log_probs)} entries")

    # ── 3. Decode and re-segment per folio ──
    print("\n  3. Decoding and re-segmenting per folio …")
    corpus = load_corpus(verbose=False)

    per_folio_stats: List[FolioResegStats] = []
    sample_segmentations: List[Dict] = []

    all_original_decoded: List[str] = []
    all_viterbi_words: List[str] = []
    all_viterbi_folios: List[str] = []

    herbal_original_decoded: List[str] = []
    herbal_viterbi_words: List[str] = []

    n_folios = 0
    for folio, page in corpus.pages.items():
        tokens = page.all_tokens
        if not tokens:
            continue
        n_folios += 1

        # Phase 16 decode (EVA space segmentation)
        decoded_words = _decode_corpus_r3(
            tokens, assignment, eva_to_triple,
            modifier_chars, modifier_rules, ref_word_set,
        )
        all_original_decoded.extend(decoded_words)

        # Build continuous decoded stream for this folio
        continuous = ''.join(decoded_words)

        # Viterbi re-segment
        viterbi_words = _viterbi_segment(
            continuous, word_log_probs, unknown_log_prob,
        )
        all_viterbi_words.extend(viterbi_words)
        for _ in viterbi_words:
            all_viterbi_folios.append(folio)

        # Dict-hit rates (using expanded dict for fair comparison)
        orig_hits = sum(1 for w in decoded_words if w in ref_word_set)
        orig_dict_hit = orig_hits / len(decoded_words) if decoded_words else 0.0

        vit_hits = sum(1 for w in viterbi_words if w in ref_word_set)
        vit_dict_hit = vit_hits / len(viterbi_words) if viterbi_words else 0.0

        orig_mean_wl = (
            sum(len(w) for w in decoded_words) / len(decoded_words)
            if decoded_words else 0.0
        )
        vit_mean_wl = (
            sum(len(w) for w in viterbi_words) / len(viterbi_words)
            if viterbi_words else 0.0
        )

        delta = vit_dict_hit - orig_dict_hit

        fs = FolioResegStats(
            folio=folio,
            section=page.section,
            n_original_tokens=len(decoded_words),
            n_viterbi_tokens=len(viterbi_words),
            original_dict_hit=round(orig_dict_hit, 4),
            viterbi_dict_hit=round(vit_dict_hit, 4),
            dict_hit_delta=round(delta, 4),
            original_mean_word_len=round(orig_mean_wl, 2),
            viterbi_mean_word_len=round(vit_mean_wl, 2),
        )
        per_folio_stats.append(fs)

        # Herbal folios
        if page.section in ('herbal_a', 'herbal_b'):
            herbal_original_decoded.extend(decoded_words)
            herbal_viterbi_words.extend(viterbi_words)

        # Sample segmentations (first 5 folios)
        if len(sample_segmentations) < 5:
            sample_segmentations.append({
                'folio': folio,
                'original': ' '.join(decoded_words[:30]),
                'viterbi': ' '.join(viterbi_words[:30]),
                'continuous': continuous[:150],
                'n_original': len(decoded_words),
                'n_viterbi': len(viterbi_words),
            })

    # ── 4. Global statistics ──
    print("\n  4. Computing global statistics …")

    n_orig = len(all_original_decoded)
    n_vit = len(all_viterbi_words)

    orig_dict_hit_global = (
        sum(1 for w in all_original_decoded if w in ref_word_set) / n_orig
        if n_orig > 0 else 0.0
    )
    vit_dict_hit_global = (
        sum(1 for w in all_viterbi_words if w in ref_word_set) / n_vit
        if n_vit > 0 else 0.0
    )

    orig_mean_wl_global = (
        sum(len(w) for w in all_original_decoded) / n_orig
        if n_orig > 0 else 0.0
    )
    vit_mean_wl_global = (
        sum(len(w) for w in all_viterbi_words) / n_vit
        if n_vit > 0 else 0.0
    )

    dict_hit_delta = vit_dict_hit_global - orig_dict_hit_global
    mean_wl_delta = vit_mean_wl_global - orig_mean_wl_global

    print(f"     Original tokens: {n_orig}")
    print(f"     Viterbi tokens:  {n_vit}")
    print(f"     Original dict_hit: {orig_dict_hit_global:.4f}")
    print(f"     Viterbi dict_hit:  {vit_dict_hit_global:.4f} "
          f"(Δ={dict_hit_delta:+.4f})")
    print(f"     Original mean word len: {orig_mean_wl_global:.2f}")
    print(f"     Viterbi mean word len:  {vit_mean_wl_global:.2f} "
          f"(Δ={mean_wl_delta:+.2f})")

    # ── 5. Top improved / degraded folios ──
    print("\n  5. Top improved folios (by Δdict_hit) …")
    by_delta = sorted(per_folio_stats, key=lambda f: -f.dict_hit_delta)
    for fs in by_delta[:5]:
        print(f"     {fs.folio:8s}  {fs.section:12s}  "
              f"orig={fs.original_dict_hit:.3f}  "
              f"vit={fs.viterbi_dict_hit:.3f}  "
              f"Δ={fs.dict_hit_delta:+.3f}")

    print("\n     Top degraded folios …")
    for fs in by_delta[-5:]:
        print(f"     {fs.folio:8s}  {fs.section:12s}  "
              f"orig={fs.original_dict_hit:.3f}  "
              f"vit={fs.viterbi_dict_hit:.3f}  "
              f"Δ={fs.dict_hit_delta:+.3f}")

    # ── 6. Herbal re-test ──
    print("\n  6. Herbal folio re-test …")
    herbal_orig_hit = (
        sum(1 for w in herbal_original_decoded if w in ref_word_set)
        / len(herbal_original_decoded)
        if herbal_original_decoded else 0.0
    )
    herbal_vit_hit = (
        sum(1 for w in herbal_viterbi_words if w in ref_word_set)
        / len(herbal_viterbi_words)
        if herbal_viterbi_words else 0.0
    )
    herbal_vit_mean_wl = (
        sum(len(w) for w in herbal_viterbi_words) / len(herbal_viterbi_words)
        if herbal_viterbi_words else 0.0
    )
    print(f"     Herbal original dict_hit: {herbal_orig_hit:.4f}")
    print(f"     Herbal Viterbi dict_hit:  {herbal_vit_hit:.4f}")
    print(f"     Herbal Viterbi mean word len: {herbal_vit_mean_wl:.2f}")

    # ── 7. Verdict ──
    if dict_hit_delta > 0.05:
        verdict = (
            f"VITERBI_IMPROVES: dict_hit {orig_dict_hit_global:.3f} → "
            f"{vit_dict_hit_global:.3f} (Δ={dict_hit_delta:+.4f}). "
            f"Re-segmentation substantially improves decoding — EVA spaces "
            f"may not mark true word boundaries."
        )
    elif dict_hit_delta > 0.01:
        verdict = (
            f"VITERBI_MARGINAL: dict_hit {orig_dict_hit_global:.3f} → "
            f"{vit_dict_hit_global:.3f} (Δ={dict_hit_delta:+.4f}). "
            f"Modest improvement from re-segmentation."
        )
    elif dict_hit_delta > -0.01:
        verdict = (
            f"VITERBI_NEUTRAL: dict_hit {orig_dict_hit_global:.3f} → "
            f"{vit_dict_hit_global:.3f} (Δ={dict_hit_delta:+.4f}). "
            f"Re-segmentation neither helps nor hurts — EVA spaces may "
            f"approximate true word boundaries."
        )
    else:
        verdict = (
            f"VITERBI_DEGRADES: dict_hit {orig_dict_hit_global:.3f} → "
            f"{vit_dict_hit_global:.3f} (Δ={dict_hit_delta:+.4f}). "
            f"EVA spaces are better segmentation markers than Viterbi. "
            f"The original spaces carry real word-boundary information."
        )

    print(f"\n  Verdict: {verdict}")

    # ── 8. Save ──
    # Sort per-folio by folio name
    per_folio_stats.sort(key=lambda f: f.folio)

    result = ResegmentDecodeResult(
        n_folios=n_folios,
        n_total_original_tokens=n_orig,
        n_total_viterbi_tokens=n_vit,
        original_dict_hit=round(orig_dict_hit_global, 4),
        original_mean_word_len=round(orig_mean_wl_global, 2),
        viterbi_dict_hit=round(vit_dict_hit_global, 4),
        viterbi_mean_word_len=round(vit_mean_wl_global, 2),
        dict_hit_delta=round(dict_hit_delta, 4),
        mean_word_len_delta=round(mean_wl_delta, 2),
        n_base_dict_words=len(base_words),
        n_expanded_dict_words=len(ref_word_set),
        unigram_vocab_size=len(word_log_probs),
        unknown_log_prob=round(unknown_log_prob, 2),
        per_folio_stats=[_convert(asdict(fs)) for fs in per_folio_stats],
        top_improved_folios=[
            _convert(asdict(fs)) for fs in
            sorted(per_folio_stats, key=lambda f: -f.dict_hit_delta)[:10]
        ],
        top_degraded_folios=[
            _convert(asdict(fs)) for fs in
            sorted(per_folio_stats, key=lambda f: f.dict_hit_delta)[:10]
        ],
        sample_segmentations=sample_segmentations,
        herbal_viterbi_dict_hit=round(herbal_vit_hit, 4),
        herbal_original_dict_hit=round(herbal_orig_hit, 4),
        herbal_mean_viterbi_word_len=round(herbal_vit_mean_wl, 2),
        viterbi_tokens_sample=all_viterbi_words[:500],
        viterbi_folios_sample=all_viterbi_folios[:500],
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'resegment_decode.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)
    print(f"\n  → {out_path}")
