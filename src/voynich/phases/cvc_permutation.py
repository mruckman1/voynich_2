"""
Phase 59, Investigation 11: CVC Permutation Coherence Test
============================================================
The p=0.011 coherence test from the paper was on CV signal words.
This module tests whether the CVC vocabulary shows equal or better
coherence by generating 1000 random coda tables (permuting coda
assignments across stroke groups) and comparing.

Dependency chain:
    results/coda_table.json           (Phase 57.1)
    results/combined_refine.json      (Phase 15)
    results/modifier_integrate.json   (Phase 16)
    results/null_corpus.json          (Phase 17)
        -> results/cvc_permutation.json
"""

import itertools
import json
import os
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import build_eva_to_triple_lookup, load_corpus
from voynich.core.reference import build_expanded_word_set, load_reference_corpus
from voynich.phases.coda_markers import (
    CodaTable,
    STROKE_TO_CODA_PRIMARY,
    build_coda_table,
    decode_corpus_cvc,
)
from voynich.phases.cvc_coda_signal import (
    _build_folio_list,
    _load_shared_data,
)
from voynich.phases.null_corpus import (
    _build_eva_bigram_model,
    _generate_null_corpus,
    _reconstruct_modifier_rules,
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
# Known coherence vocabulary sets (from reviewer_coherence.py)
# ---------------------------------------------------------------------------

VERB_PARADIGM_WORDS = {
    'dice', 'dico', 'dica', 'dise', 'dine',  # dire paradigm
    'cola', 'codi', 'cora',                    # medical imperatives
    'sene', 'sera', 'sero',                    # sentire/servare
}

FUNCTION_KIT = {
    'de', 'di', 'da', 'du',
    'in', 'ad', 'et', 'se', 'si', 'cu', 'ce',
    'la', 'le', 'lo', 'li', 'ne', 'no', 'ni',
    'con', 'per', 'non', 'bene',
}

PHARMA_REGISTER = {
    'coralli', 'diasene', 'stercora', 'radicom', 'commune', 'secundi',
    'ratione', 'balsamo', 'radice', 'herba', 'aqua', 'oleum',
    'cura', 'morbo', 'febre', 'dolor', 'sana',
    'colar', 'corar', 'senen', 'dinen',  # CVC variants
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TrialResult:
    """Result of one random coda table trial."""
    trial: int
    coda_perm: Dict[str, str]
    n_signal: int
    mean_selectivity: float
    has_verb: bool
    has_function: bool
    has_pharma: bool
    all_three: bool


@dataclass
class CvcPermutationResult:
    """Full Investigation 11 output."""
    phase: str = "59"
    investigation: str = "11"
    experiment: str = "cvc_permutation"
    n_trials: int = 0
    # Real table results
    real_n_signal: int = 0
    real_mean_selectivity: float = 0.0
    real_all_three: bool = False
    # Random distribution
    random_mean_signal: float = 0.0
    random_std_signal: float = 0.0
    n_random_all_three: int = 0
    p_all_three: float = 0.0
    cv_p_all_three: float = 0.011   # Phase permutation test baseline
    cvc_vs_cv: str = ''             # 'improved', 'similar', 'degraded'
    # Per-criterion
    p_verb: float = 0.0
    p_function: float = 0.0
    p_pharma: float = 0.0
    # Percentile of real vs random
    signal_percentile: float = 0.0
    # Gates
    g1_coherent: bool = False       # CVC p(all_three) < 0.05
    g2_vs_cv: bool = False          # CVC p ≤ CV p
    g3_signal_count: bool = False   # CVC signal count ≥ CV
    gates_passed: int = 0
    gate_passed: bool = False
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def _build_permuted_coda_table(perm: Dict[str, str]) -> CodaTable:
    """Build a CodaTable with a specific stroke→coda permutation."""
    table = build_coda_table('primary')
    table.stroke_to_coda = dict(perm)
    return table


def _fast_signal_words(
    real_decoded: List[str],
    null_counters: List[Counter],
    ref_word_set: Set[str],
) -> List[Dict[str, Any]]:
    """Fast signal word extraction — word-level only, no per-token classification.

    Equivalent to _run_signal_isolation().top_signal_words but ~10× faster
    because it:
    1. Accepts precomputed null_counters instead of rebuilding Counter(nd)
       for every test word
    2. Skips per-token SIGNAL/SHARED_HIT/SHARED_MISS/ANTI_SIGNAL classification
       (not needed for coherence scoring)
    """
    real_word_counts = Counter(w for w in real_decoded if w in ref_word_set)

    word_signals: List[Dict[str, Any]] = []
    for word in sorted(real_word_counts.keys()):
        real_count = real_word_counts[word]
        null_counts = [nc.get(word, 0) for nc in null_counters]
        null_mean = sum(null_counts) / len(null_counts) if null_counts else 0.0
        null_var = (sum((c - null_mean) ** 2 for c in null_counts)
                    / len(null_counts) if null_counts else 0.0)
        null_std = null_var ** 0.5

        sigma = ((real_count - null_mean) / null_std) if null_std > 0 else (
            999.0 if real_count > null_mean else 0.0)
        selectivity = (real_count / null_mean) if null_mean > 0 else 999.0

        if sigma > 2.0:
            word_signals.append({
                'word': word,
                'sigma': round(sigma, 2),
                'real_count': real_count,
                'null_mean': round(null_mean, 2),
                'selectivity': round(selectivity, 2),
            })

    word_signals.sort(key=lambda w: -w['sigma'])
    return word_signals


def _score_coherence(signal_words: List[Dict[str, Any]]) -> Dict[str, bool]:
    """Score whether signal words show verb paradigm, function kit, pharma register."""
    words = set(w['word'].lower() for w in signal_words)

    has_verb = len(words & VERB_PARADIGM_WORDS) >= 2
    has_function = len(words & FUNCTION_KIT) >= 3
    has_pharma = len(words & PHARMA_REGISTER) >= 1

    return {
        'verb_paradigm': has_verb,
        'function_kit': has_function,
        'pharma_register': has_pharma,
        'all_three': has_verb and has_function and has_pharma,
    }


def run_cvc_perm():
    """Investigation 11: CVC permutation coherence test."""
    t0 = time.time()
    print("=" * 70)
    print("Phase 59, Investigation 11: CVC Permutation Coherence Test")
    print("=" * 70)

    rd = str(_results_dir())

    # Load shared data
    print("\n  Loading shared data ...")
    data = _load_shared_data()

    all_tokens = data['all_tokens']
    assignment = data['assignment']
    eva_to_triple = data['eva_to_triple']
    ref_word_set = data['ref_word_set']
    null_token_lists = data['null_token_lists']
    n_tokens = len(all_tokens)

    # The 5 stroke types and 6 possible coda consonants
    strokes = ['hook', 'descender', 'sigmoid', 'vertical', 'connector']
    codas = ['n', 'r', 's', 't', 'l', 'm']

    # 1. Evaluate real table
    print("\n  Evaluating real CVC table (primary) ...")
    real_table = build_coda_table('primary')
    real_decoded = decode_corpus_cvc(all_tokens, assignment, eva_to_triple, real_table)

    null_decoded_list = []
    for null_tokens in null_token_lists:
        nd = decode_corpus_cvc(null_tokens, assignment, eva_to_triple, real_table)
        null_decoded_list.append(nd)

    # Precompute null word counters (avoids rebuilding Counter per test word)
    null_counters = [Counter(nd) for nd in null_decoded_list]

    real_signal_words = _fast_signal_words(real_decoded, null_counters, ref_word_set)
    real_coherence = _score_coherence(real_signal_words)

    finite_sels = [w['selectivity'] for w in real_signal_words if w['selectivity'] < 900]
    real_mean_sel = sum(finite_sels) / len(finite_sels) if finite_sels else 0.0

    print(f"  Real signal words: {len(real_signal_words)}")
    print(f"  Real mean selectivity: {real_mean_sel:.2f}×")
    print(f"  Real coherence: verb={real_coherence['verb_paradigm']}, "
          f"function={real_coherence['function_kit']}, "
          f"pharma={real_coherence['pharma_register']}")
    print(f"  All three: {real_coherence['all_three']}")

    # 2. Generate permutations
    # All permutations of 5 codas chosen from 6 (with replacement not needed,
    # but we use random sampling for 1000 trials from 6^5=7776 possible)
    n_trials = 1000
    print(f"\n  Running {n_trials} random coda permutations ...")

    rng = random.Random(42)
    trials: List[TrialResult] = []
    progress_step = n_trials // 10

    for trial in range(n_trials):
        if trial % progress_step == 0:
            elapsed = time.time() - t0
            rate = trial / elapsed if elapsed > 0 and trial > 0 else 0
            eta = (n_trials - trial) / rate if rate > 0 else 0
            print(f"    Trial {trial}/{n_trials} "
                  f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining) ...")

        # Random coda assignment: each stroke gets a random coda
        perm = {stroke: rng.choice(codas) for stroke in strokes}

        # Skip if identical to real
        if all(perm[s] == real_table.stroke_to_coda.get(s) for s in strokes):
            perm[rng.choice(strokes)] = rng.choice(codas)

        custom_table = _build_permuted_coda_table(perm)
        decoded = decode_corpus_cvc(all_tokens, assignment, eva_to_triple, custom_table)

        # Fast signal word extraction (precomputed null counters, no per-token)
        signal_words = _fast_signal_words(decoded, null_counters, ref_word_set)
        coherence = _score_coherence(signal_words)

        finite_s = [w['selectivity'] for w in signal_words if w['selectivity'] < 900]
        mean_sel = sum(finite_s) / len(finite_s) if finite_s else 0.0

        trials.append(TrialResult(
            trial=trial,
            coda_perm=perm,
            n_signal=len(signal_words),
            mean_selectivity=mean_sel,
            has_verb=coherence['verb_paradigm'],
            has_function=coherence['function_kit'],
            has_pharma=coherence['pharma_register'],
            all_three=coherence['all_three'],
        ))

    # 3. Analyze results
    signal_counts = [t.n_signal for t in trials]
    n_all_three = sum(1 for t in trials if t.all_three)
    n_verb = sum(1 for t in trials if t.has_verb)
    n_function = sum(1 for t in trials if t.has_function)
    n_pharma = sum(1 for t in trials if t.has_pharma)

    p_all_three = n_all_three / n_trials
    p_verb = n_verb / n_trials
    p_function = n_function / n_trials
    p_pharma = n_pharma / n_trials

    random_mean = float(np.mean(signal_counts))
    random_std = float(np.std(signal_counts))
    real_n_signal = len(real_signal_words)
    signal_percentile = (sum(1 for s in signal_counts if s <= real_n_signal)
                         / n_trials * 100) if signal_counts else 0

    cv_p = 0.011  # Phase permutation test
    cvc_vs_cv = ('improved' if p_all_three < cv_p
                 else 'degraded' if p_all_three > 0.05
                 else 'similar')

    print(f"\n  Results:")
    print(f"    Real signal words:   {real_n_signal}")
    print(f"    Random mean:         {random_mean:.1f} ± {random_std:.1f}")
    print(f"    Signal percentile:   {signal_percentile:.0f}th")
    print(f"\n  Coherence:")
    print(f"    p(all_three):   {p_all_three:.4f} (CV baseline: {cv_p:.3f})")
    print(f"    p(verb):        {p_verb:.4f}")
    print(f"    p(function):    {p_function:.4f}")
    print(f"    p(pharma):      {p_pharma:.4f}")
    print(f"    CVC vs CV:      {cvc_vs_cv}")

    # Gates
    g1 = p_all_three < 0.05
    g2 = p_all_three <= cv_p
    g3 = real_n_signal >= 56  # CV baseline count
    gates_passed = sum([g1, g2, g3])

    print(f"\n  Validation Gates:")
    print(f"    G1 p(all_three) < 0.05:   {'PASS' if g1 else 'FAIL'} ({p_all_three:.4f})")
    print(f"    G2 CVC p ≤ CV p (0.011):  {'PASS' if g2 else 'FAIL'}")
    print(f"    G3 signal ≥ 56:           {'PASS' if g3 else 'FAIL'} "
          f"({real_n_signal})")
    print(f"    Gates passed: {gates_passed}/3")

    result = CvcPermutationResult(
        n_trials=n_trials,
        real_n_signal=real_n_signal,
        real_mean_selectivity=round(real_mean_sel, 2),
        real_all_three=real_coherence['all_three'],
        random_mean_signal=round(random_mean, 1),
        random_std_signal=round(random_std, 1),
        n_random_all_three=n_all_three,
        p_all_three=round(p_all_three, 4),
        cv_p_all_three=cv_p,
        cvc_vs_cv=cvc_vs_cv,
        p_verb=round(p_verb, 4),
        p_function=round(p_function, 4),
        p_pharma=round(p_pharma, 4),
        signal_percentile=round(signal_percentile, 1),
        g1_coherent=g1,
        g2_vs_cv=g2,
        g3_signal_count=g3,
        gates_passed=gates_passed,
        gate_passed=gates_passed >= 2,
        runtime_seconds=round(time.time() - t0, 2),
    )

    path = _save_json(rd, 'cvc_permutation.json', result)
    print(f"\n  Saved: {path}")
    print(f"  Investigation 11 completed in {time.time() - t0:.1f}s")
