"""
Phase 66, Track 8: Hand 4 Focus
================================
Latin ending distribution, common multi-token sequences, and
pharmaceutical vocabulary density on Hand 4 (scribe 4, biological
section) only.

Dependency chain:
    results/combined_refine.json      (Phase 15)
        -> results/p66_hand4.json
"""
from __future__ import annotations

import json
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

import numpy as np

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    load_corpus,
    _infer_scribe,
)
from voynich.core.reference import build_expanded_word_set, load_reference_corpus
from voynich.phases.corrected_coda import (
    build_coda_table_v2,
    decode_corpus_cvc_v2,
)
from voynich.phases.suffix_calibration import SIGNAL_WORDS_SET


# ---------------------------------------------------------------------------
# JSON helpers
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
    if isinstance(obj, set):
        return sorted(obj)
    if isinstance(obj, (bool, int, float, str, type(None))):
        return obj
    return str(obj)


def _safe_load(path: str) -> Dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _save_json(rd: str, filename: str, data: Any) -> str:
    path = os.path.join(rd, filename)
    with open(path, 'w') as f:
        json.dump(_convert(data), f, indent=2)
    return path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LATIN_ENDINGS_2CHAR = {
    'us', 'um', 'is', 'es', 'em', 'en', 'er', 'or', 'on',
    'as', 'os', 'am', 'in', 'an',
}

LATIN_ENDINGS_3CHAR = {
    'ius', 'ium', 'ens', 'ans', 'ter', 'tor', 'unt', 'ent',
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _latin_ending_frac(decoded_tokens: List[str]) -> float:
    """Fraction of decoded tokens ending in common Latin suffixes."""
    n_matched = 0
    n_total = 0
    for w in decoded_tokens:
        if not w or w == '?' or len(w) < 3:
            continue
        n_total += 1
        suf2 = w[-2:]
        suf3 = w[-3:] if len(w) >= 4 else ''
        if suf2 in LATIN_ENDINGS_2CHAR or suf3 in LATIN_ENDINGS_3CHAR:
            n_matched += 1
    return n_matched / n_total if n_total > 0 else 0.0


def _ending_distribution(decoded_tokens: List[str]) -> Dict[str, int]:
    """Histogram of final 2 characters across decoded tokens."""
    endings = Counter()
    for w in decoded_tokens:
        if w and w != '?' and len(w) >= 2:
            endings[w[-2:]] += 1
    return dict(endings.most_common(50))


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class Hand4Result:
    phase: str = "66"
    step: str = "66.8"
    experiment: str = "hand4_focus"
    n_hand4_pages: int = 0
    n_hand4_tokens: int = 0
    hand4_dict_hit: float = 0.0
    full_corpus_dict_hit: float = 0.0
    hand4_latin_ending_frac: float = 0.0
    full_corpus_latin_ending_frac: float = 0.0
    hand4_signal_rate: float = 0.0
    n_recurring_bigrams: int = 0
    n_recurring_trigrams: int = 0
    top_bigrams: List[Dict] = field(default_factory=list)
    top_trigrams: List[Dict] = field(default_factory=list)
    ending_distribution: Dict[str, int] = field(default_factory=dict)
    h41_latin_ending: bool = False      # latin ending fraction > 15%
    h42_pharma_vocab: bool = False      # signal rate > 5% (as proxy for pharma)
    h43_multi_token: bool = False       # >= 10 recurring 2-token sequences
    gates_passed: int = 0
    gate_passed: bool = False
    verdict: str = ""
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def run_hand4():
    """Phase 66.8: Hand 4 focused analysis."""
    t0 = time.time()
    rd = str(_results_dir())
    print("=" * 70)
    print("Phase 66, Track 8: Hand 4 Focus")
    print("=" * 70)

    # Load dependencies
    eva_to_triple = build_eva_to_triple_lookup()
    refine = _safe_load(os.path.join(rd, 'combined_refine.json'))
    assignment = refine.get('best_assignment', {})
    if not assignment:
        print("  WARNING: combined_refine.json not found or empty; using empty assignment")
    coda_table = build_coda_table_v2()

    # Build dictionary
    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    base_words = set(w.lower() for w in ref_corpus.get_combined_tokens('latin') if len(w) >= 2)
    expanded, _ = build_expanded_word_set(base_words)
    ref_word_set = base_words | expanded

    corpus = load_corpus(verbose=False)

    # Split corpus: Hand 4 vs full
    hand4_tokens = []
    full_tokens = []
    n_hand4_pages = 0

    for folio_id, page in corpus.pages.items():
        tokens = page.all_tokens
        full_tokens.extend(tokens)
        hand = _infer_scribe(folio_id)
        if hand == 4:
            hand4_tokens.extend(tokens)
            n_hand4_pages += 1

    print(f"  Hand 4 pages: {n_hand4_pages}")
    print(f"  Hand 4 tokens: {len(hand4_tokens)}")
    print(f"  Full corpus tokens: {len(full_tokens)}")

    if not hand4_tokens:
        print("  WARNING: No Hand 4 tokens found")

    # CVC decode
    hand4_decoded = decode_corpus_cvc_v2(hand4_tokens, assignment, eva_to_triple, coda_table)
    full_decoded = decode_corpus_cvc_v2(full_tokens, assignment, eva_to_triple, coda_table)

    h4_valid = [d for d in hand4_decoded if d and d != '?']
    full_valid = [d for d in full_decoded if d and d != '?']

    # Dict hit
    h4_dict_hit = sum(1 for d in h4_valid if d in ref_word_set) / len(h4_valid) if h4_valid else 0.0
    full_dict_hit = sum(1 for d in full_valid if d in ref_word_set) / len(full_valid) if full_valid else 0.0

    print(f"  Hand 4 dict_hit: {h4_dict_hit:.1%}")
    print(f"  Full corpus dict_hit: {full_dict_hit:.1%}")

    # Latin ending fraction
    h4_latin_end = _latin_ending_frac(h4_valid)
    full_latin_end = _latin_ending_frac(full_valid)
    print(f"  Hand 4 Latin ending frac: {h4_latin_end:.1%}")
    print(f"  Full corpus Latin ending frac: {full_latin_end:.1%}")

    # Signal rate (proxy for pharma vocab)
    h4_signal_count = sum(1 for d in h4_valid if d in SIGNAL_WORDS_SET)
    h4_signal_rate = h4_signal_count / len(h4_valid) if h4_valid else 0.0
    print(f"  Hand 4 signal rate: {h4_signal_rate:.1%}")

    # Multi-token sequences: bigrams and trigrams of consecutive decoded tokens
    bigram_counter = Counter()
    trigram_counter = Counter()
    for i in range(len(h4_valid) - 1):
        bg = (h4_valid[i], h4_valid[i + 1])
        bigram_counter[bg] += 1
    for i in range(len(h4_valid) - 2):
        tg = (h4_valid[i], h4_valid[i + 1], h4_valid[i + 2])
        trigram_counter[tg] += 1

    # Recurring = count >= 3
    recurring_bigrams = [(bg, c) for bg, c in bigram_counter.most_common() if c >= 3]
    recurring_trigrams = [(tg, c) for tg, c in trigram_counter.most_common() if c >= 3]

    top_bigrams = [
        {'bigram': list(bg), 'count': c}
        for bg, c in recurring_bigrams[:30]
    ]
    top_trigrams = [
        {'trigram': list(tg), 'count': c}
        for tg, c in recurring_trigrams[:20]
    ]

    print(f"  Recurring bigrams (count>=3): {len(recurring_bigrams)}")
    print(f"  Recurring trigrams (count>=3): {len(recurring_trigrams)}")
    if recurring_bigrams:
        print(f"  Top bigrams: {recurring_bigrams[:5]}")

    # Ending distribution
    end_dist = _ending_distribution(h4_valid)

    # Gates
    h41 = h4_latin_end > 0.15
    h42 = h4_signal_rate > 0.05
    h43 = len(recurring_bigrams) >= 10
    gates_passed = sum([h41, h42, h43])

    if gates_passed >= 2:
        verdict = "HAND4_STRUCTURED"
    elif gates_passed == 1:
        verdict = "HAND4_MARGINAL"
    else:
        verdict = "HAND4_NO_STRUCTURE"

    result = Hand4Result(
        n_hand4_pages=n_hand4_pages,
        n_hand4_tokens=len(h4_valid),
        hand4_dict_hit=round(h4_dict_hit, 4),
        full_corpus_dict_hit=round(full_dict_hit, 4),
        hand4_latin_ending_frac=round(h4_latin_end, 4),
        full_corpus_latin_ending_frac=round(full_latin_end, 4),
        hand4_signal_rate=round(h4_signal_rate, 4),
        n_recurring_bigrams=len(recurring_bigrams),
        n_recurring_trigrams=len(recurring_trigrams),
        top_bigrams=top_bigrams,
        top_trigrams=top_trigrams,
        ending_distribution=end_dist,
        h41_latin_ending=h41,
        h42_pharma_vocab=h42,
        h43_multi_token=h43,
        gates_passed=gates_passed,
        gate_passed=gates_passed >= 2,
        verdict=verdict,
        runtime_seconds=time.time() - t0,
    )

    # Print summary
    print(f"\n  {'Metric':<25s} {'Hand 4':>10s} {'Full':>10s}")
    print(f"  {'Tokens':<25s} {len(h4_valid):10d} {len(full_valid):10d}")
    print(f"  {'Dict hit':<25s} {h4_dict_hit:10.1%} {full_dict_hit:10.1%}")
    print(f"  {'Latin ending frac':<25s} {h4_latin_end:10.1%} {full_latin_end:10.1%}")
    print(f"  {'Signal rate':<25s} {h4_signal_rate:10.1%}       {'--':>5s}")
    print(f"  {'Recurring bigrams':<25s} {len(recurring_bigrams):10d}       {'--':>5s}")
    print(f"  {'Recurring trigrams':<25s} {len(recurring_trigrams):10d}       {'--':>5s}")
    print(f"\n  Gates: H41={'PASS' if h41 else 'FAIL'} H42={'PASS' if h42 else 'FAIL'} "
          f"H43={'PASS' if h43 else 'FAIL'} ({gates_passed}/3)")
    print(f"  Verdict: {verdict}")

    path = _save_json(rd, 'p66_hand4.json', asdict(result))
    print(f"\n  Saved: {path}")
    print(f"  Runtime: {result.runtime_seconds:.1f}s")
    return result
