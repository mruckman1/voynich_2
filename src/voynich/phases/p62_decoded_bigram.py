"""
Phase 62, Investigation 4: Decoded Bigram Frequency vs Latin
=============================================================
Compare consecutive decoded token pairs against syllabified Latin.
Two types of Latin bigrams: within-word (consecutive syllables of same
word) and cross-word (last syllable + first syllable of next).
Classify Voynich bigrams accordingly.

Dependency chain:
    results/combined_refine.json      (Phase 15)
    results/modifier_integrate.json   (Phase 16)
    data/reference/latin/             (Latin reference corpus)
        -> results/phase62_decoded_bigram.json
"""

import json
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy import stats as sp_stats

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import build_eva_to_triple_lookup, load_corpus
from voynich.core.reference import load_reference_corpus
from voynich.core.stats import syllabify_latin
from voynich.phases.corrected_coda import (
    build_coda_table_v2,
    decode_corpus_cvc_v2,
)


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
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DecodedBigramResult:
    phase: str = "62"
    step: str = "62.4"
    experiment: str = "decoded_bigram"
    n_voynich_bigrams: int = 0
    n_latin_within: int = 0
    n_latin_cross: int = 0
    # Classification of top-200 Voynich bigrams
    n_classified: int = 0
    n_within_word: int = 0
    n_cross_word: int = 0
    n_ambiguous: int = 0
    n_not_in_latin: int = 0
    within_word_fraction: float = 0.0
    cross_word_fraction: float = 0.0
    top_classified: List[Dict] = field(default_factory=list)
    # Rank correlation
    spearman_within: float = 0.0
    spearman_cross: float = 0.0
    # Gates
    g1_enough_data: bool = False       # >= 50 classifiable bigrams
    g2_within_fraction: bool = False   # within-word fraction > 40%
    g3_rank_corr: bool = False         # rank correlation with within-word > 0.3
    gates_passed: int = 0
    gate_passed: bool = False
    verdict: str = ""
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def run_decoded_bigram():
    """Phase 62.4: Decoded bigram frequency vs Latin."""
    t0 = time.time()
    rd = str(_results_dir())
    print("=" * 70)
    print("Phase 62, Investigation 4: Decoded Bigram Frequency vs Latin")
    print("=" * 70)

    # Load Voynich decoded
    eva_to_triple = build_eva_to_triple_lookup()
    refine = _safe_load(os.path.join(rd, 'combined_refine.json'))
    assignment = refine.get('best_assignment', {})
    coda_table = build_coda_table_v2()

    corpus = load_corpus(verbose=False)
    all_tokens = corpus.get_tokens()
    decoded = decode_corpus_cvc_v2(all_tokens, assignment, eva_to_triple, coda_table)
    v_tokens = [d for d in decoded if d and d != '?']
    print(f"  Voynich decoded tokens: {len(v_tokens)}")

    # Voynich bigrams (consecutive decoded token pairs)
    voynich_bigrams = Counter()
    for i in range(len(v_tokens) - 1):
        voynich_bigrams[(v_tokens[i], v_tokens[i + 1])] += 1

    # Load and syllabify Latin
    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    latin_words = [w.lower() for w in ref_corpus.get_combined_tokens('latin') if len(w) >= 2]
    print(f"  Latin reference words: {len(latin_words)}")

    latin_within = Counter()   # consecutive syllables within same word
    latin_cross = Counter()    # last syl of word N, first syl of word N+1

    prev_syls = None
    for i, word in enumerate(latin_words):
        syls = syllabify_latin(word)
        if not syls:
            prev_syls = None
            continue

        # Within-word bigrams
        for j in range(len(syls) - 1):
            latin_within[(syls[j], syls[j + 1])] += 1

        # Cross-word bigrams
        if prev_syls:
            latin_cross[(prev_syls[-1], syls[0])] += 1

        prev_syls = syls

    print(f"  Latin within-word bigrams: {len(latin_within)}")
    print(f"  Latin cross-word bigrams: {len(latin_cross)}")

    # Classify top-200 Voynich bigrams
    classified = []
    for bigram, count in voynich_bigrams.most_common(200):
        within_count = latin_within.get(bigram, 0)
        cross_count = latin_cross.get(bigram, 0)

        if within_count > cross_count * 2:
            bg_type = 'WITHIN_WORD'
        elif cross_count > within_count * 2:
            bg_type = 'CROSS_WORD'
        elif within_count > 0 or cross_count > 0:
            bg_type = 'AMBIGUOUS'
        else:
            bg_type = 'NOT_IN_LATIN'

        classified.append({
            'bigram': list(bigram),
            'voynich_count': count,
            'latin_within': within_count,
            'latin_cross': cross_count,
            'type': bg_type,
        })

    n_within = sum(1 for c in classified if c['type'] == 'WITHIN_WORD')
    n_cross = sum(1 for c in classified if c['type'] == 'CROSS_WORD')
    n_ambig = sum(1 for c in classified if c['type'] == 'AMBIGUOUS')
    n_not = sum(1 for c in classified if c['type'] == 'NOT_IN_LATIN')
    n_classifiable = n_within + n_cross + n_ambig
    within_frac = n_within / n_classifiable if n_classifiable > 0 else 0.0
    cross_frac = n_cross / n_classifiable if n_classifiable > 0 else 0.0

    # Rank correlation: for bigrams that appear in both Voynich and Latin within-word
    common_within = [c for c in classified if c['latin_within'] > 0]
    if len(common_within) >= 5:
        v_counts = [c['voynich_count'] for c in common_within]
        l_counts = [c['latin_within'] for c in common_within]
        spearman_within = float(sp_stats.spearmanr(v_counts, l_counts).statistic)
        if np.isnan(spearman_within):
            spearman_within = 0.0
    else:
        spearman_within = 0.0

    common_cross = [c for c in classified if c['latin_cross'] > 0]
    if len(common_cross) >= 5:
        v_counts = [c['voynich_count'] for c in common_cross]
        l_counts = [c['latin_cross'] for c in common_cross]
        spearman_cross = float(sp_stats.spearmanr(v_counts, l_counts).statistic)
        if np.isnan(spearman_cross):
            spearman_cross = 0.0
    else:
        spearman_cross = 0.0

    # Gates
    g1 = n_classifiable >= 50
    g2 = within_frac > 0.40
    g3 = spearman_within > 0.3
    gates_passed = sum([g1, g2, g3])

    if n_within > n_cross:
        verdict = "TOKENS_ARE_SYLLABLES"
    elif n_cross > n_within:
        verdict = "TOKENS_ARE_WORDS"
    else:
        verdict = "INCONCLUSIVE"

    result = DecodedBigramResult(
        n_voynich_bigrams=len(voynich_bigrams),
        n_latin_within=len(latin_within),
        n_latin_cross=len(latin_cross),
        n_classified=len(classified),
        n_within_word=n_within,
        n_cross_word=n_cross,
        n_ambiguous=n_ambig,
        n_not_in_latin=n_not,
        within_word_fraction=round(within_frac, 4),
        cross_word_fraction=round(cross_frac, 4),
        top_classified=classified[:30],
        spearman_within=round(spearman_within, 4),
        spearman_cross=round(spearman_cross, 4),
        g1_enough_data=g1,
        g2_within_fraction=g2,
        g3_rank_corr=g3,
        gates_passed=gates_passed,
        gate_passed=gates_passed >= 2,
        verdict=verdict,
        runtime_seconds=time.time() - t0,
    )

    # Print summary
    print(f"\n  Top-200 Voynich bigrams classified:")
    print(f"    WITHIN_WORD: {n_within}  CROSS_WORD: {n_cross}  "
          f"AMBIGUOUS: {n_ambig}  NOT_IN_LATIN: {n_not}")
    print(f"    Within-word fraction: {within_frac:.1%}")
    print(f"    Cross-word fraction: {cross_frac:.1%}")
    print(f"  Rank correlation (within-word): {spearman_within:.3f}")
    print(f"  Rank correlation (cross-word): {spearman_cross:.3f}")
    print(f"\n  Top classified bigrams:")
    for c in classified[:10]:
        print(f"    {c['bigram'][0]:6s} {c['bigram'][1]:6s}  v={c['voynich_count']:4d}  "
              f"within={c['latin_within']:4d}  cross={c['latin_cross']:4d}  {c['type']}")
    print(f"\n  Gates: G1={'PASS' if g1 else 'FAIL'} G2={'PASS' if g2 else 'FAIL'} "
          f"G3={'PASS' if g3 else 'FAIL'} ({gates_passed}/3)")
    print(f"  Verdict: {verdict}")

    path = _save_json(rd, 'phase62_decoded_bigram.json', asdict(result))
    print(f"\n  Saved: {path}")
    print(f"  Runtime: {result.runtime_seconds:.1f}s")
    return result
