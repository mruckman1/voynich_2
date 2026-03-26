"""
Phase 66, Track 7: Language A Focus
====================================
Separate signal isolation, vocabulary analysis, and herbal content
detection on Language A (Currier A / herbal_a section) only.

Dependency chain:
    results/combined_refine.json      (Phase 15)
        -> results/p66_lang_a.json
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
)
from voynich.core.reference import build_expanded_word_set, load_reference_corpus
from voynich.phases.corrected_coda import (
    build_coda_table_v2,
    decode_corpus_cvc_v2,
)
from voynich.phases.suffix_calibration import SIGNAL_WORDS_51, SIGNAL_WORDS_SET


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
# Botanical / pharmaceutical vocabulary
# ---------------------------------------------------------------------------

BOTANICAL_SIGNAL_WORDS = {
    w for w, info in SIGNAL_WORDS_51.items()
    if info.get('type') in ('pharm', 'botanical', 'plant', 'herb')
}

# Fallback pharma vocabulary from known signal words with botanical/medical
# glosses, plus common herbal-text Latin words.
PHARMA_VOCAB = BOTANICAL_SIGNAL_WORDS | {
    'sene', 'cola', 'tere', 'codi', 'raro', 'sero',
    'radix', 'herba', 'folia', 'semen', 'oleum',
    'aqua', 'pulvis', 'succus', 'cortex', 'flores',
}


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class LangAResult:
    phase: str = "66"
    step: str = "66.7"
    experiment: str = "language_a_focus"
    n_lang_a_pages: int = 0
    n_lang_a_tokens: int = 0
    n_full_corpus_tokens: int = 0
    lang_a_dict_hit: float = 0.0
    full_corpus_dict_hit: float = 0.0
    dict_hit_ratio: float = 0.0
    lang_a_signal_rate: float = 0.0
    full_corpus_signal_rate: float = 0.0
    lang_a_pharma_density: float = 0.0
    full_corpus_pharma_density: float = 0.0
    top_signal_words: List[Dict] = field(default_factory=list)
    top_decoded_types: List[Dict] = field(default_factory=list)
    la1_signal_rate: bool = False       # signal rate > 10%
    la2_pharma_density: bool = False    # pharma density > corpus mean
    la3_dict_hit: bool = False          # dict hit > 40%
    gates_passed: int = 0
    gate_passed: bool = False
    verdict: str = ""
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def run_lang_a():
    """Phase 66.7: Language A focused analysis."""
    t0 = time.time()
    rd = str(_results_dir())
    print("=" * 70)
    print("Phase 66, Track 7: Language A Focus")
    print("=" * 70)

    # Load dependencies
    eva_to_triple = build_eva_to_triple_lookup()
    refine = _safe_load(os.path.join(rd, 'combined_refine.json'))
    assignment = refine.get('best_assignment', {})
    if not assignment:
        print("  WARNING: combined_refine.json not found or empty; using empty assignment")
    coda_table = build_coda_table_v2()

    # Build dictionary (10K most frequent + signal words)
    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    all_ref_tokens = [w.lower() for w in ref_corpus.get_combined_tokens('latin') if len(w) >= 2]
    freq_counter = Counter(all_ref_tokens)
    top_10k = set(w for w, _ in freq_counter.most_common(10000))
    ref_word_set = top_10k | SIGNAL_WORDS_SET

    # Also build expanded set for full dict_hit comparison
    base_words = set(all_ref_tokens)
    expanded, _ = build_expanded_word_set(base_words)
    full_ref_set = base_words | expanded

    corpus = load_corpus(verbose=False)

    # Split corpus: Language A vs full
    lang_a_tokens = []
    full_tokens = []

    n_lang_a_pages = 0
    for folio_id, page in corpus.pages.items():
        tokens = page.all_tokens
        full_tokens.extend(tokens)
        # Language A: Currier language 'A' OR section 'herbal_a'
        if page.language == 'A' or page.section == 'herbal_a':
            lang_a_tokens.extend(tokens)
            n_lang_a_pages += 1

    print(f"  Language A pages: {n_lang_a_pages}")
    print(f"  Language A tokens: {len(lang_a_tokens)}")
    print(f"  Full corpus tokens: {len(full_tokens)}")

    # CVC decode both subsets
    lang_a_decoded = decode_corpus_cvc_v2(lang_a_tokens, assignment, eva_to_triple, coda_table)
    full_decoded = decode_corpus_cvc_v2(full_tokens, assignment, eva_to_triple, coda_table)

    la_valid = [d for d in lang_a_decoded if d and d != '?']
    full_valid = [d for d in full_decoded if d and d != '?']

    # Dict hit rates (using full expanded dictionary for fair comparison)
    la_dict_hit = sum(1 for d in la_valid if d in full_ref_set) / len(la_valid) if la_valid else 0.0
    full_dict_hit = sum(1 for d in full_valid if d in full_ref_set) / len(full_valid) if full_valid else 0.0
    dict_hit_ratio = la_dict_hit / full_dict_hit if full_dict_hit > 0 else 0.0

    print(f"  Lang A dict_hit: {la_dict_hit:.1%}")
    print(f"  Full corpus dict_hit: {full_dict_hit:.1%}")

    # Signal rate
    la_signal_count = sum(1 for d in la_valid if d in SIGNAL_WORDS_SET)
    full_signal_count = sum(1 for d in full_valid if d in SIGNAL_WORDS_SET)
    la_signal_rate = la_signal_count / len(la_valid) if la_valid else 0.0
    full_signal_rate = full_signal_count / len(full_valid) if full_valid else 0.0

    # Pharma density
    la_pharma_count = sum(1 for d in la_valid if d in PHARMA_VOCAB)
    full_pharma_count = sum(1 for d in full_valid if d in PHARMA_VOCAB)
    la_pharma_density = la_pharma_count / len(la_valid) if la_valid else 0.0
    full_pharma_density = full_pharma_count / len(full_valid) if full_valid else 0.0

    # Top signal words in Language A
    la_signal_counter = Counter(d for d in la_valid if d in SIGNAL_WORDS_SET)
    top_signal = [
        {'word': w, 'count': c,
         'sigma': SIGNAL_WORDS_51[w].get('sigma', 0.0) if w in SIGNAL_WORDS_51 else 0.0,
         'type': SIGNAL_WORDS_51[w].get('type', '') if w in SIGNAL_WORDS_51 else ''}
        for w, c in la_signal_counter.most_common(20)
    ]

    # Top decoded types in Language A
    la_type_counter = Counter(la_valid)
    top_types = [
        {'word': w, 'count': c, 'in_dict': w in full_ref_set, 'is_signal': w in SIGNAL_WORDS_SET}
        for w, c in la_type_counter.most_common(30)
    ]

    # Herbal content scoring: count botanical signal words
    botanical_hits = Counter(d for d in la_valid if d in PHARMA_VOCAB)
    n_botanical = sum(botanical_hits.values())
    print(f"  Botanical/pharma hits in Lang A: {n_botanical} ({la_pharma_density:.1%})")
    if botanical_hits:
        print(f"  Top botanical: {botanical_hits.most_common(10)}")

    # Gates
    la1 = la_signal_rate > 0.10
    la2 = la_pharma_density > full_pharma_density
    la3 = la_dict_hit > 0.40
    gates_passed = sum([la1, la2, la3])

    if gates_passed >= 2:
        verdict = "LANG_A_ENRICHED"
    elif gates_passed == 1:
        verdict = "LANG_A_MARGINAL"
    else:
        verdict = "LANG_A_NO_ENRICHMENT"

    result = LangAResult(
        n_lang_a_pages=n_lang_a_pages,
        n_lang_a_tokens=len(la_valid),
        n_full_corpus_tokens=len(full_valid),
        lang_a_dict_hit=round(la_dict_hit, 4),
        full_corpus_dict_hit=round(full_dict_hit, 4),
        dict_hit_ratio=round(dict_hit_ratio, 4),
        lang_a_signal_rate=round(la_signal_rate, 4),
        full_corpus_signal_rate=round(full_signal_rate, 4),
        lang_a_pharma_density=round(la_pharma_density, 4),
        full_corpus_pharma_density=round(full_pharma_density, 4),
        top_signal_words=top_signal,
        top_decoded_types=top_types,
        la1_signal_rate=la1,
        la2_pharma_density=la2,
        la3_dict_hit=la3,
        gates_passed=gates_passed,
        gate_passed=gates_passed >= 2,
        verdict=verdict,
        runtime_seconds=time.time() - t0,
    )

    # Print summary
    print(f"\n  {'Metric':<25s} {'Lang A':>10s} {'Full':>10s}")
    print(f"  {'Tokens':<25s} {len(la_valid):10d} {len(full_valid):10d}")
    print(f"  {'Dict hit':<25s} {la_dict_hit:10.1%} {full_dict_hit:10.1%}")
    print(f"  {'Signal rate':<25s} {la_signal_rate:10.1%} {full_signal_rate:10.1%}")
    print(f"  {'Pharma density':<25s} {la_pharma_density:10.1%} {full_pharma_density:10.1%}")
    print(f"\n  Gates: LA1={'PASS' if la1 else 'FAIL'} LA2={'PASS' if la2 else 'FAIL'} "
          f"LA3={'PASS' if la3 else 'FAIL'} ({gates_passed}/3)")
    print(f"  Verdict: {verdict}")

    path = _save_json(rd, 'p66_lang_a.json', asdict(result))
    print(f"\n  Saved: {path}")
    print(f"  Runtime: {result.runtime_seconds:.1f}s")
    return result
