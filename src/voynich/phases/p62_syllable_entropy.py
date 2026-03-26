"""
Phase 62, Investigation 8: Syllable-Level Entropy Comparison
=============================================================
Treat each CVC-decoded token as one syllable unit. Compute H1 (unigram
entropy), H2 (conditional bigram entropy), and TTR.  Compare to Latin
syllabified from Circa Instans / reference corpus.

Dependency chain:
    results/combined_refine.json      (Phase 15)
    results/modifier_integrate.json   (Phase 16)
    data/reference/latin/             (Latin reference corpus)
        -> results/phase62_syllable_entropy.json
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

def _compute_h1(tokens: List[str]) -> float:
    """Shannon entropy of unigram distribution."""
    counts = Counter(tokens)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((c / total) * np.log2(c / total)
                for c in counts.values() if c > 0)


def _compute_h2(tokens: List[str]) -> float:
    """Conditional entropy H(X_n | X_{n-1})."""
    bigrams = Counter(zip(tokens[:-1], tokens[1:]))
    unigrams = Counter(tokens[:-1])
    total_bigrams = sum(bigrams.values())
    if total_bigrams == 0:
        return 0.0

    h2 = 0.0
    for (a, b), count in bigrams.items():
        p_ab = count / total_bigrams
        p_b_given_a = count / unigrams[a]
        if p_b_given_a > 0:
            h2 -= p_ab * np.log2(p_b_given_a)
    return h2


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SyllableEntropyResult:
    phase: str = "62"
    step: str = "62.8"
    experiment: str = "syllable_entropy"
    # Voynich
    voynich_h1: float = 0.0
    voynich_h2: float = 0.0
    voynich_ttr: float = 0.0
    voynich_n_types: int = 0
    voynich_n_tokens: int = 0
    voynich_top_syllables: List[List] = field(default_factory=list)
    # Latin
    latin_h1: float = 0.0
    latin_h2: float = 0.0
    latin_ttr: float = 0.0
    latin_n_types: int = 0
    latin_n_tokens: int = 0
    latin_top_syllables: List[List] = field(default_factory=list)
    # Ratios
    h1_ratio: float = 0.0
    h2_ratio: float = 0.0
    ttr_ratio: float = 0.0
    # Gates
    g1_h1_ratio: bool = False          # 0.5 <= h1_ratio <= 1.5
    g2_h2_ratio: bool = False          # 0.5 <= h2_ratio <= 1.5
    g3_ttr_ratio: bool = False         # TTR within 2x
    gates_passed: int = 0
    gate_passed: bool = False
    verdict: str = ""
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def run_syllable_entropy():
    """Phase 62.8: Syllable-level entropy comparison."""
    t0 = time.time()
    rd = str(_results_dir())
    print("=" * 70)
    print("Phase 62, Investigation 8: Syllable-Level Entropy")
    print("=" * 70)

    # Load Voynich decoded
    eva_to_triple = build_eva_to_triple_lookup()
    refine = _safe_load(os.path.join(rd, 'combined_refine.json'))
    assignment = refine.get('best_assignment', {})
    coda_table = build_coda_table_v2()

    corpus = load_corpus(verbose=False)
    all_tokens = corpus.get_tokens()
    decoded = decode_corpus_cvc_v2(all_tokens, assignment, eva_to_triple, coda_table)

    # Filter valid decoded tokens (treat each as one syllable)
    v_syls = [d for d in decoded if d and d != '?']
    print(f"  Voynich syllable tokens: {len(v_syls)}")

    # Load Latin reference
    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    latin_words = ref_corpus.get_combined_tokens('latin')
    print(f"  Latin reference words: {len(latin_words)}")

    # Syllabify Latin
    l_syls = []
    for word in latin_words:
        syls = syllabify_latin(word.lower())
        if syls:
            l_syls.extend(syls)
    print(f"  Latin syllables: {len(l_syls)}")

    # Compute entropies
    v_h1 = _compute_h1(v_syls)
    v_h2 = _compute_h2(v_syls)
    v_counter = Counter(v_syls)
    v_ttr = len(v_counter) / len(v_syls) if v_syls else 0.0

    l_h1 = _compute_h1(l_syls)
    l_h2 = _compute_h2(l_syls)
    l_counter = Counter(l_syls)
    l_ttr = len(l_counter) / len(l_syls) if l_syls else 0.0

    h1_ratio = v_h1 / l_h1 if l_h1 > 0 else 0.0
    h2_ratio = v_h2 / l_h2 if l_h2 > 0 else 0.0
    ttr_ratio = v_ttr / l_ttr if l_ttr > 0 else 0.0

    # Gates
    g1 = 0.5 <= h1_ratio <= 1.5
    g2 = 0.5 <= h2_ratio <= 1.5
    g3 = 0.5 <= ttr_ratio <= 2.0
    gates_passed = sum([g1, g2, g3])

    if gates_passed == 3:
        verdict = "ENTROPY_COMPATIBLE"
    elif gates_passed >= 2:
        verdict = "PARTIALLY_COMPATIBLE"
    else:
        verdict = "ENTROPY_DIVERGENT"

    result = SyllableEntropyResult(
        voynich_h1=round(v_h1, 4),
        voynich_h2=round(v_h2, 4),
        voynich_ttr=round(v_ttr, 6),
        voynich_n_types=len(v_counter),
        voynich_n_tokens=len(v_syls),
        voynich_top_syllables=v_counter.most_common(20),
        latin_h1=round(l_h1, 4),
        latin_h2=round(l_h2, 4),
        latin_ttr=round(l_ttr, 6),
        latin_n_types=len(l_counter),
        latin_n_tokens=len(l_syls),
        latin_top_syllables=l_counter.most_common(20),
        h1_ratio=round(h1_ratio, 4),
        h2_ratio=round(h2_ratio, 4),
        ttr_ratio=round(ttr_ratio, 4),
        g1_h1_ratio=g1,
        g2_h2_ratio=g2,
        g3_ttr_ratio=g3,
        gates_passed=gates_passed,
        gate_passed=gates_passed >= 2,
        verdict=verdict,
        runtime_seconds=time.time() - t0,
    )

    # Print summary
    print(f"\n  {'Metric':<20s} {'Voynich':>10s} {'Latin':>10s} {'Ratio':>8s}")
    print(f"  {'H1 (unigram)':<20s} {v_h1:10.3f} {l_h1:10.3f} {h1_ratio:8.3f}")
    print(f"  {'H2 (conditional)':<20s} {v_h2:10.3f} {l_h2:10.3f} {h2_ratio:8.3f}")
    print(f"  {'TTR':<20s} {v_ttr:10.5f} {l_ttr:10.5f} {ttr_ratio:8.3f}")
    print(f"  {'Types':<20s} {len(v_counter):10d} {len(l_counter):10d}")
    print(f"  {'Tokens':<20s} {len(v_syls):10d} {len(l_syls):10d}")
    print(f"\n  Gates: G1={'PASS' if g1 else 'FAIL'} G2={'PASS' if g2 else 'FAIL'} "
          f"G3={'PASS' if g3 else 'FAIL'} ({gates_passed}/3)")
    print(f"  Verdict: {verdict}")

    path = _save_json(rd, 'phase62_syllable_entropy.json', asdict(result))
    print(f"\n  Saved: {path}")
    print(f"  Runtime: {result.runtime_seconds:.1f}s")
    return result
