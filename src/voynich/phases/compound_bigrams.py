"""
Phase 32.3 – Compound-Sign Bigram Plausibility (THE DECISIVE TEST)
====================================================================
Measure bigram plausibility on consecutive SIGNAL pairs under the
compound-sign decode and compare to Phase 29's z = 6.14.

If z > 8: compound decode produces better Latin sequences.
If z > 12: approaching readable text.
If z ~ 6: improvement is collisions from shorter words, not better decoding.

Dependency chain:
    compound_decode.json       (Step 32.1)
    compound_signal.json       (Step 32.2)
    signal_bigrams.json        (Phase 29 baseline)
        → compound_bigrams.json (this step)
"""

import json
import os
import random
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
from voynich.phases.compound_decode import SUFFIX_ENDING_MAP


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
# POS tagging heuristic based on Latin suffix ending
# ---------------------------------------------------------------------------

_POS_ENDING_MAP = {
    # Noun/adjective endings
    'a': 'NOUN', 'i': 'NOUN', 'e': 'NOUN', 'um': 'NOUN',
    'is': 'NOUN', 'ae': 'NOUN', 'em': 'NOUN', 'am': 'NOUN',
    'en': 'NOUN', 'ium': 'NOUN', 'an': 'NOUN',
    # No ending
    '': 'UNKNOWN', 'n': 'UNKNOWN',
}

# Common function words
_FUNCTION_WORDS = {'de', 'in', 'ad', 'cum', 'per', 'pro', 'ex', 'ab', 'et', 'sed',
                   'si', 'ut', 'non', 'nec', 'aut', 'vel', 'nam', 'iam', 'tum'}

_VERB_PATTERNS = {'re', 'ere', 'ire', 'are', 'et', 'it', 'at', 'nt', 'ur'}


def _tag_pos(word: str, latin_ending: str) -> str:
    """Rough POS tag based on Latin ending and word form."""
    if word in _FUNCTION_WORDS:
        return 'FUNC'
    if latin_ending in _POS_ENDING_MAP:
        return _POS_ENDING_MAP[latin_ending]
    return 'UNKNOWN'


def _pos_bigram_analysis(
    signal_pairs: List[Tuple[str, int, str, str]],
    latin_endings: List[str],
    classifications: List[str],
) -> Tuple[Dict[str, int], float, float]:
    """Analyse POS bigrams among SIGNAL pairs.

    Returns (pos_counts, valid_fraction, chi_sq).
    """
    pos_counts: Counter = Counter()
    n_valid = 0
    n_total = 0

    # Valid POS bigrams in Latin
    valid_bigrams = {
        ('FUNC', 'NOUN'), ('FUNC', 'FUNC'), ('NOUN', 'FUNC'),
        ('NOUN', 'NOUN'), ('VERB', 'NOUN'), ('NOUN', 'VERB'),
        ('FUNC', 'VERB'), ('VERB', 'FUNC'),
    }

    for _, pos_i, w1, w2 in signal_pairs:
        if pos_i + 1 >= len(latin_endings):
            continue
        p1 = _tag_pos(w1, latin_endings[pos_i])
        p2 = _tag_pos(w2, latin_endings[pos_i + 1])
        key = f"{p1}_{p2}"
        pos_counts[key] += 1
        n_total += 1
        if (p1, p2) in valid_bigrams:
            n_valid += 1

    valid_frac = n_valid / n_total if n_total > 0 else 0.0

    # Simple chi-squared: observed valid vs expected (50% baseline)
    if n_total > 0:
        expected = n_total * 0.5
        chi_sq = ((n_valid - expected) ** 2 / expected
                  + ((n_total - n_valid) - expected) ** 2 / expected)
    else:
        chi_sq = 0.0

    return dict(pos_counts), valid_frac, chi_sq


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_compound_bigrams() -> None:
    """Step 32.3: Compound bigram plausibility — THE DECISIVE TEST."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE 32.3: Compound-Sign Bigram Plausibility (DECISIVE TEST)")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load inputs ──
    print("\n  1. Loading inputs ...")
    with open(os.path.join(rd, 'compound_decode.json')) as f:
        cd = json.load(f)
    with open(os.path.join(rd, 'compound_signal.json')) as f:
        cs = json.load(f)

    token_folios = cd['token_folios']
    token_evas = cd['token_evas']
    token_decoded = cd['token_decoded']
    token_dict_hits = cd['token_dict_hits']
    token_latin_endings = cd['token_latin_endings']
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
        print(f"     Phase 29 baseline: z={phase29_z:.2f}, "
              f"{phase29_n_hits} hits, {phase29_n_relaxed} relaxed")

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

    # ── 3. Build inflected-form bigram table ──
    print("\n  3. Building inflected-form bigram table ...")
    inflected_tokens = list(ref_word_set)
    inflected_bigrams: Set[Tuple[str, str]] = set()
    # Build bigrams from expanded word set paired with reference words
    for i in range(len(ref_tokens) - 1):
        w1, w2 = ref_tokens[i], ref_tokens[i + 1]
        inflected_bigrams.add((w1, w2))
        # Also add expanded variants of each word
        for ending in SUFFIX_ENDING_MAP.values():
            if ending:
                v1 = w1 + ending
                v2 = w2 + ending
                if v1 in ref_word_set:
                    inflected_bigrams.add((v1, w2))
                    inflected_bigrams.add((v1, v2 if v2 in ref_word_set else w2))
                if v2 in ref_word_set:
                    inflected_bigrams.add((w1, v2))
    print(f"     {len(inflected_bigrams)} inflected bigrams")

    # ── 4. Find SIGNAL-SIGNAL pairs ──
    print("\n  4. Finding SIGNAL-SIGNAL pairs ...")
    signal_pairs = _find_signal_pairs(
        classifications, token_decoded, token_folios,
    )
    print(f"     {len(signal_pairs)} consecutive SIGNAL-SIGNAL pairs")

    # ── 5. Bigram plausibility test ──
    print("\n  5. Bigram plausibility on SIGNAL pairs ...")
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

    # ── 6. Null permutation test ──
    print("\n  6. Null permutation test (1000 permutations) ...")
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
    print(f"     Delta z: {z_score - phase29_z:+.2f}")

    # ── 7. Trigram test ──
    print("\n  7. Trigram plausibility test ...")
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

    # ── 8. Relaxed bigram test ──
    print("\n  8. Relaxed bigram test (edit distance 1) ...")
    n_relaxed = _relaxed_bigram_test(signal_pairs, ref_bigrams, ref_word_set)
    relaxed_rate = (n_bigram_hits + n_relaxed) / len(signal_pairs) if signal_pairs else 0.0
    print(f"     {n_relaxed} additional relaxed matches")
    print(f"     Combined rate: {relaxed_rate:.6f}")

    # ── 9. Inflected-form bigram test ──
    print("\n  9. Inflected-form bigram test ...")
    n_inflected_hits = 0
    for folio, pos, w1, w2 in signal_pairs:
        if (w1, w2) in inflected_bigrams and (w1, w2) not in ref_bigrams:
            n_inflected_hits += 1
    inflected_rate = n_inflected_hits / len(signal_pairs) if signal_pairs else 0.0
    print(f"     {n_inflected_hits} inflected bigram hits (beyond exact)")

    # ── 10. POS bigram test ──
    print("\n  10. POS bigram test ...")
    pos_counts, pos_valid_frac, pos_chi_sq = _pos_bigram_analysis(
        signal_pairs, token_latin_endings, classifications,
    )
    print(f"     Valid POS bigrams: {pos_valid_frac:.1%}")
    print(f"     POS chi-squared: {pos_chi_sq:.2f}")
    for k, v in sorted(pos_counts.items(), key=lambda x: -x[1])[:8]:
        print(f"       {k}: {v}")

    # ── 11. Per-folio ranking ──
    print("\n  11. Per-folio SIGNAL pair ranking (top 10) ...")
    folio_stats = _folio_signal_pair_ranking(
        signal_pairs, ref_bigrams, classifications, token_folios,
    )
    for fs in folio_stats[:10]:
        print(f"     {fs.folio:8s}  signal={fs.n_signal:3d}/{fs.n_tokens:3d} "
              f"({fs.signal_rate:.1%})  "
              f"pairs={fs.n_signal_pairs:3d}  "
              f"bigram_hits={fs.n_bigram_hits}")

    # ── 12. Verdict ──
    delta_z = z_score - phase29_z

    if z_score > 10:
        verdict_label = "COMPOUND_BREAKTHROUGH"
    elif z_score > 8 or (z_score > 6.5 and delta_z > 0.5):
        verdict_label = "COMPOUND_IMPROVEMENT"
    elif z_score > 4 and delta_z > -1.0:
        verdict_label = "COMPOUND_CONFIRMED"
    else:
        verdict_label = "COMPOUND_COLLISIONS"

    verdict = (
        f"{verdict_label}: z={z_score:.2f} (Phase 29: {phase29_z:.2f}, "
        f"delta={delta_z:+.2f}), {n_bigram_hits} hits, "
        f"{n_relaxed} relaxed, {n_trigram_hits} trigrams"
    )
    print(f"\n  VERDICT: {verdict}")

    # ── 13. Save ──
    print("\n  13. Saving compound_bigrams.json ...")
    output = {
        'token_folios': token_folios,
        'token_evas': token_evas,
        'token_decoded': token_decoded,
        'token_classifications': classifications,
        'token_dict_hits': token_dict_hits,
        'token_latin_endings': token_latin_endings,
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
        'inflected_ref_bigram_count': len(inflected_bigrams),
        'n_inflected_bigram_hits': n_inflected_hits,
        'inflected_bigram_hit_rate': round(inflected_rate, 6),
        'pos_bigram_counts': pos_counts,
        'pos_valid_fraction': round(pos_valid_frac, 4),
        'pos_chi_sq': round(pos_chi_sq, 2),
        'phase29_bigram_z': phase29_z,
        'phase29_n_bigram_hits': phase29_n_hits,
        'phase29_n_relaxed': phase29_n_relaxed,
        'delta_bigram_z': round(delta_z, 2),
        'delta_n_bigram_hits': n_bigram_hits - phase29_n_hits,
        'delta_n_relaxed': n_relaxed - phase29_n_relaxed,
        'folio_signal_pair_stats': [
            _convert(asdict(fs)) for fs in folio_stats[:30]
        ],
        'gate_passed': z_score > 4.0,
        'verdict': verdict,
        'runtime_seconds': round(time.time() - t0, 1),
    }

    with open(os.path.join(rd, 'compound_bigrams.json'), 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n  Step 32.3 completed in {time.time() - t0:.1f}s")
