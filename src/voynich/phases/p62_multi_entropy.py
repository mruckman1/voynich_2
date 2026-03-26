"""
Phase 62, Investigation 11: Multi-Level Entropy Comparison
==========================================================
Compare Voynich and Latin at 4 levels: character, syllable/token,
bigram of tokens, trigram of tokens.  Compute H1 and H2 at each level.

Dependency chain:
    results/combined_refine.json      (Phase 15)
    results/modifier_integrate.json   (Phase 16)
    data/reference/latin/             (Latin reference corpus)
        -> results/phase62_multi_entropy.json
"""

import json
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np

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
# Entropy helpers
# ---------------------------------------------------------------------------

def _h1(tokens):
    """Shannon entropy of unigram distribution."""
    counts = Counter(tokens)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((c / total) * np.log2(c / total) for c in counts.values() if c > 0)


def _h2(tokens):
    """Conditional entropy H(X_n | X_{n-1})."""
    bigrams = Counter(zip(tokens[:-1], tokens[1:]))
    unigrams = Counter(tokens[:-1])
    total_bg = sum(bigrams.values())
    if total_bg == 0:
        return 0.0
    h = 0.0
    for (a, b), cnt in bigrams.items():
        p_ab = cnt / total_bg
        p_b_a = cnt / unigrams[a]
        if p_b_a > 0:
            h -= p_ab * np.log2(p_b_a)
    return h


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class EntropyLevel:
    level: str              # character / syllable / bigram / trigram
    voynich_h1: float = 0.0
    voynich_h2: float = 0.0
    latin_h1: float = 0.0
    latin_h2: float = 0.0
    h1_ratio: float = 0.0
    h2_ratio: float = 0.0


@dataclass
class MultiEntropyResult:
    phase: str = "62"
    step: str = "62.11"
    experiment: str = "multi_entropy"
    levels: List[Dict] = field(default_factory=list)
    overall_similarity: float = 0.0
    # Gates
    g1_char_h1: bool = False           # character H1 ratio 0.7-1.3
    g2_syl_h1: bool = False            # syllable H1 ratio 0.5-1.5
    g3_char_tighter: bool = False      # char divergence < syllable divergence
    gates_passed: int = 0
    gate_passed: bool = False
    verdict: str = ""
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def run_multi_entropy():
    """Phase 62.11: Multi-level entropy comparison."""
    t0 = time.time()
    rd = str(_results_dir())
    print("=" * 70)
    print("Phase 62, Investigation 11: Multi-Level Entropy")
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

    # Load Latin
    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    latin_words = [w.lower() for w in ref_corpus.get_combined_tokens('latin') if len(w) >= 2]
    print(f"  Latin reference words: {len(latin_words)}")

    # Syllabify Latin
    l_syls = []
    for w in latin_words:
        syls = syllabify_latin(w)
        if syls:
            l_syls.extend(syls)
    print(f"  Latin syllables: {len(l_syls)}")

    levels = []

    # Level 1: Character
    v_chars = list(''.join(v_tokens))
    l_chars = list(''.join(l_syls))
    char_level = EntropyLevel(
        level='character',
        voynich_h1=round(_h1(v_chars), 4),
        voynich_h2=round(_h2(v_chars), 4),
        latin_h1=round(_h1(l_chars), 4),
        latin_h2=round(_h2(l_chars), 4),
    )
    char_level.h1_ratio = round(char_level.voynich_h1 / char_level.latin_h1, 4) if char_level.latin_h1 else 0.0
    char_level.h2_ratio = round(char_level.voynich_h2 / char_level.latin_h2, 4) if char_level.latin_h2 else 0.0
    levels.append(char_level)

    # Level 2: Syllable / token
    syl_level = EntropyLevel(
        level='syllable',
        voynich_h1=round(_h1(v_tokens), 4),
        voynich_h2=round(_h2(v_tokens), 4),
        latin_h1=round(_h1(l_syls), 4),
        latin_h2=round(_h2(l_syls), 4),
    )
    syl_level.h1_ratio = round(syl_level.voynich_h1 / syl_level.latin_h1, 4) if syl_level.latin_h1 else 0.0
    syl_level.h2_ratio = round(syl_level.voynich_h2 / syl_level.latin_h2, 4) if syl_level.latin_h2 else 0.0
    levels.append(syl_level)

    # Level 3: Bigram of tokens
    v_bigrams = list(zip(v_tokens[:-1], v_tokens[1:]))
    l_bigrams = list(zip(l_syls[:-1], l_syls[1:]))
    bg_level = EntropyLevel(
        level='bigram',
        voynich_h1=round(_h1(v_bigrams), 4),
        latin_h1=round(_h1(l_bigrams), 4),
    )
    bg_level.h1_ratio = round(bg_level.voynich_h1 / bg_level.latin_h1, 4) if bg_level.latin_h1 else 0.0
    levels.append(bg_level)

    # Level 4: Trigram of tokens
    v_trigrams = list(zip(v_tokens[:-2], v_tokens[1:-1], v_tokens[2:]))
    l_trigrams = list(zip(l_syls[:-2], l_syls[1:-1], l_syls[2:]))
    tg_level = EntropyLevel(
        level='trigram',
        voynich_h1=round(_h1(v_trigrams), 4),
        latin_h1=round(_h1(l_trigrams), 4),
    )
    tg_level.h1_ratio = round(tg_level.voynich_h1 / tg_level.latin_h1, 4) if tg_level.latin_h1 else 0.0
    levels.append(tg_level)

    # Overall similarity: mean of H1 ratios (closer to 1.0 = more similar)
    h1_ratios = [lv.h1_ratio for lv in levels if lv.h1_ratio > 0]
    overall = 1.0 - np.mean([abs(r - 1.0) for r in h1_ratios]) if h1_ratios else 0.0

    # Gates
    g1 = 0.7 <= char_level.h1_ratio <= 1.3
    g2 = 0.5 <= syl_level.h1_ratio <= 1.5
    char_div = abs(char_level.h1_ratio - 1.0)
    syl_div = abs(syl_level.h1_ratio - 1.0)
    g3 = char_div < syl_div
    gates_passed = sum([g1, g2, g3])

    if gates_passed == 3:
        verdict = "ALL_LEVELS_COMPATIBLE"
    elif gates_passed >= 2:
        verdict = "MOSTLY_COMPATIBLE"
    else:
        verdict = "DIVERGENT"

    result = MultiEntropyResult(
        levels=[_convert(asdict(lv)) for lv in levels],
        overall_similarity=round(float(overall), 4),
        g1_char_h1=g1,
        g2_syl_h1=g2,
        g3_char_tighter=g3,
        gates_passed=gates_passed,
        gate_passed=gates_passed >= 2,
        verdict=verdict,
        runtime_seconds=time.time() - t0,
    )

    # Print summary
    print(f"\n  {'Level':<12s} {'V_H1':>8s} {'L_H1':>8s} {'Ratio':>8s} {'V_H2':>8s} {'L_H2':>8s}")
    for lv in levels:
        print(f"  {lv.level:<12s} {lv.voynich_h1:8.3f} {lv.latin_h1:8.3f} "
              f"{lv.h1_ratio:8.3f} {lv.voynich_h2:8.3f} {lv.latin_h2:8.3f}")
    print(f"\n  Overall similarity: {overall:.3f}")
    print(f"  Char divergence: {char_div:.3f}  Syllable divergence: {syl_div:.3f}")
    print(f"\n  Gates: G1={'PASS' if g1 else 'FAIL'} G2={'PASS' if g2 else 'FAIL'} "
          f"G3={'PASS' if g3 else 'FAIL'} ({gates_passed}/3)")
    print(f"  Verdict: {verdict}")

    path = _save_json(rd, 'phase62_multi_entropy.json', asdict(result))
    print(f"\n  Saved: {path}")
    print(f"  Runtime: {result.runtime_seconds:.1f}s")
    return result
