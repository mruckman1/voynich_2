"""
Phase 73, Track 2: Corrected Grammatical Analysis
===================================================
Re-run Phase 71 Track 1's inflectional catalog with the corrected
coda table (connector→null). With connector strokes no longer producing
'r', the 47% passive-voice fraction should drop dramatically.

Descender→r is unchanged (genuine coda). Connector→'' (null).

Also runs an exhaustive 3! = 6 permutation test for the 3 non-descender
codas (hook, sigmoid, vertical → which of n,s,t?).

Dependency chain:
    results/p73_redecode.json                 (Step 0)
    results/combined_refine.json              (Phase 15)
    results/p69_clean_corpus.json             (Phase 69)
    results/phase71_inflectional_catalog.json (Phase 71, for comparison)
        -> results/p73_grammar.json
"""

import json
import os
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from itertools import permutations
from typing import Any, Dict, List, Set, Tuple

import numpy as np

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import build_eva_to_triple_lookup, load_corpus, tokenize_eva_chars
from voynich.phases.coda_markers import get_coda
from voynich.phases.corrected_coda import (
    classify_token_chars_v2,
    decode_token_cvc_v2,
)
from voynich.phases.inflectional_catalog import (
    CODA_GRAMMAR,
    DOUBLE_CODA_GRAMMAR,
    _VERBAL_FUNCS,
    _NOMINAL_FUNCS,
    _CI_EXPECTED,
    _classify_all_tokens,
    _compute_broad_distribution,
    _chi2_distance,
    _section_and_hand_profiles,
    _chi2_contingency_p,
    _cross_validation_agreement,
    _build_section_list,
    _build_hand_list,
    _determine_gram_function,
)
from voynich.phases.p72_connector import _build_coda_table_with_connector
from voynich.phases.suffix_grammar import _classify_latin_ending


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
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class CorrectedGrammarResult:
    phase: str = "73"
    step: str = "73.2"
    experiment: str = "corrected_grammar"
    # Token counts
    n_tokens: int = 0
    n_with_coda: int = 0
    n_single_coda: int = 0
    n_double_coda: int = 0
    n_unmarked: int = 0
    n_function_stem: int = 0
    # Grammatical distribution
    grammatical_counts: Dict[str, int] = field(default_factory=dict)
    broad_distribution: Dict[str, float] = field(default_factory=dict)
    # Per-coda breakdown
    coda_function_counts: Dict[str, Dict[str, int]] = field(default_factory=dict)
    # Profiles
    section_profiles: Dict[str, Dict[str, int]] = field(default_factory=dict)
    hand_profiles: Dict[str, Dict[str, int]] = field(default_factory=dict)
    section_chi2_p: float = 1.0
    hand_chi2_p: float = 1.0
    # Null validation (exhaustive 6 permutations + bootstrap)
    null_exhaustive: Dict[str, Any] = field(default_factory=dict)
    null_bootstrap: Dict[str, Any] = field(default_factory=dict)
    # Cross-validation
    cross_validation_agreement: float = 0.0
    # Comparison with Phase 71
    old_verbal_fraction: float = 0.0
    new_verbal_fraction: float = 0.0
    old_r_coda_count: int = 0
    new_r_coda_count: int = 0
    old_xval: float = 0.0
    # Gates
    gate_g1: bool = False  # Verbal 10-25%
    gate_g2: bool = False  # Nominal 15-40%
    gate_g3: bool = False  # Best permutation (rank 1/6)
    gate_g4: bool = False  # Bootstrap p < 0.10
    gate_g5: bool = False  # Section chi² p < 0.05
    gates_passed: int = 0
    verdict: str = "UNKNOWN"
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Exhaustive 3-coda permutation test
# ---------------------------------------------------------------------------

def _exhaustive_coda_permutation(
    all_tokens: List[str],
    coda_table,
    real_distribution: Dict[str, float],
) -> Dict[str, Any]:
    """Test all 3! = 6 permutations of {hook, sigmoid, vertical} → {n, s, t}.

    Descender→r is FIXED (genuine coda). Connector→'' is FIXED (null).
    Only the three non-descender modifier strokes are permuted.
    """
    # Pre-compute: for each token, get coda consonants under corrected model
    token_coda_lists: List[List[str]] = []
    for token in all_tokens:
        eva_chars = tokenize_eva_chars(token)
        classified = classify_token_chars_v2(eva_chars, coda_table)
        codas = []
        for role, char in classified:
            if role == 'CODA_MARKER':
                coda_val = get_coda(char, coda_table)
                if coda_val:  # non-empty (excludes connector→'')
                    codas.append(coda_val)
        token_coda_lists.append(codas)

    real_distance = _chi2_distance(real_distribution, _CI_EXPECTED)

    # The real mapping: hook→n, sigmoid→s, vertical→t, descender→r
    # Permuting only {n, s, t} assignments (not r)
    consonants = ['n', 's', 't']
    all_perms = list(permutations(consonants))

    results = []
    for perm in all_perms:
        # Map real coda consonants to permuted values
        remap = {'n': perm[0], 's': perm[1], 't': perm[2], 'r': 'r'}

        counts: Dict[str, int] = {
            'VERBAL': 0, 'NOMINAL': 0, 'FUNCTION_STEM': 0, 'UNMARKED': 0,
        }
        for codas, token in zip(token_coda_lists, all_tokens):
            if not codas:
                if len(token) <= 3:
                    counts['FUNCTION_STEM'] += 1
                else:
                    counts['UNMARKED'] += 1
                continue

            last_coda = remap.get(codas[-1], codas[-1])
            grammar = CODA_GRAMMAR.get(last_coda, {})
            cat = grammar.get('category', 'UNMARKED')
            if cat == 'VERBAL':
                counts['VERBAL'] += 1
            elif cat == 'NOMINAL':
                counts['NOMINAL'] += 1
            else:
                counts['UNMARKED'] += 1

        total = sum(counts.values())
        dist = {k: v / total for k, v in counts.items()} if total > 0 else {}
        distance = _chi2_distance(dist, _CI_EXPECTED)

        results.append({
            'mapping': dict(zip(['hook', 'sigmoid', 'vertical'], perm)),
            'distribution': dist,
            'ci_distance': distance,
        })

    results.sort(key=lambda x: x['ci_distance'])
    real_rank = next(
        (i + 1 for i, r in enumerate(results)
         if r['ci_distance'] == real_distance),
        len(results)
    )

    return {
        'real_ci_distance': real_distance,
        'real_rank': real_rank,
        'is_best': real_rank == 1,
        'all_permutations': results,
    }


def _bootstrap_coda_null(
    all_tokens: List[str],
    coda_table,
    real_distribution: Dict[str, float],
    n_trials: int = 500,
) -> Dict[str, Any]:
    """Bootstrap: random draws from {n, r, s, t} for all 4 strokes."""
    token_coda_lists: List[List[str]] = []
    for token in all_tokens:
        eva_chars = tokenize_eva_chars(token)
        classified = classify_token_chars_v2(eva_chars, coda_table)
        codas = []
        for role, char in classified:
            if role == 'CODA_MARKER':
                coda_val = get_coda(char, coda_table)
                if coda_val:
                    codas.append(coda_val)
        token_coda_lists.append(codas)

    real_distance = _chi2_distance(real_distribution, _CI_EXPECTED)
    coda_letters = ['n', 'r', 's', 't']

    null_distances = []
    rng = np.random.default_rng(seed=42)

    for _ in range(n_trials):
        shuffled = list(coda_letters)
        rng.shuffle(shuffled)
        remap = dict(zip(coda_letters, shuffled))

        counts: Dict[str, int] = {
            'VERBAL': 0, 'NOMINAL': 0, 'FUNCTION_STEM': 0, 'UNMARKED': 0,
        }
        for codas, token in zip(token_coda_lists, all_tokens):
            if not codas:
                if len(token) <= 3:
                    counts['FUNCTION_STEM'] += 1
                else:
                    counts['UNMARKED'] += 1
                continue

            last_coda = remap.get(codas[-1], codas[-1])
            grammar = CODA_GRAMMAR.get(last_coda, {})
            cat = grammar.get('category', 'UNMARKED')
            if cat == 'VERBAL':
                counts['VERBAL'] += 1
            elif cat == 'NOMINAL':
                counts['NOMINAL'] += 1
            else:
                counts['UNMARKED'] += 1

        total = sum(counts.values())
        dist = {k: v / total for k, v in counts.items()} if total > 0 else {}
        null_distances.append(_chi2_distance(dist, _CI_EXPECTED))

    null_distances_arr = np.array(null_distances)
    p = float(np.mean(null_distances_arr <= real_distance))

    return {
        'real_ci_distance': real_distance,
        'null_mean': float(np.mean(null_distances_arr)),
        'null_std': float(np.std(null_distances_arr)),
        'p_value': p,
        'significant': p < 0.10,
        'n_trials': n_trials,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_corrected_grammar() -> CorrectedGrammarResult:
    """Track 2: Re-run inflectional catalog with connector→null."""
    t0 = time.time()
    rd = str(_results_dir())

    print("Phase 73.2 — Corrected Grammatical Analysis")
    print("=" * 50)

    # --- Load data ---
    eva_to_triple = build_eva_to_triple_lookup()
    corrected_coda = _build_coda_table_with_connector('')

    refine_data = _safe_load(os.path.join(rd, 'combined_refine.json'))
    assignment = refine_data.get('best_assignment', {})

    corpus = load_corpus(verbose=False)
    all_tokens = corpus.get_tokens()

    from voynich.phases.inflectional_catalog import _build_folio_list
    folios = _build_folio_list(corpus)
    sections = _build_section_list(corpus)
    hands = _build_hand_list(corpus)

    clean_data = _safe_load(os.path.join(rd, 'p69_clean_corpus.json'))
    clean_indices = set(clean_data.get('clean_indices', []))

    # --- Load old Phase 71 for comparison ---
    old_data = _safe_load(os.path.join(rd, 'phase71_inflectional_catalog.json'))
    old_verbal_frac = old_data.get('broad_distribution', {}).get('VERBAL', 0.572)
    old_xval = old_data.get('cross_validation_agreement', 0.24)

    # Count old r-codas from Phase 71
    old_r_count = 0
    old_coda_counts = old_data.get('coda_function_counts', {})
    if 'r' in old_coda_counts:
        old_r_count = sum(old_coda_counts['r'].values()) if isinstance(
            old_coda_counts['r'], dict) else 0

    print(f"  Tokens: {len(all_tokens)}")
    print(f"  Old verbal fraction: {100*old_verbal_frac:.1f}%")

    # --- Classify all tokens with corrected coda ---
    print("  Classifying tokens with corrected coda table...")
    catalog = _classify_all_tokens(
        all_tokens, assignment, eva_to_triple, corrected_coda,
        folios, sections, hands, clean_indices)

    # --- Compute statistics ---
    gram_counts = Counter(entry['gram_function'] for entry in catalog)
    broad_dist = _compute_broad_distribution(catalog)

    n_with_coda = sum(1 for e in catalog if e['n_codas'] > 0)
    n_single = sum(1 for e in catalog if e['n_codas'] == 1)
    n_double = sum(1 for e in catalog if e['n_codas'] == 2)
    n_unmarked = sum(1 for e in catalog if e['gram_category'] == 'UNMARKED')
    n_func = sum(1 for e in catalog if e['gram_category'] == 'FUNCTION_STEM')

    # Per-coda breakdown
    coda_func_counts: Dict[str, Counter] = defaultdict(Counter)
    new_r_count = 0
    for entry in catalog:
        for coda in entry['coda_consonants']:
            if coda:
                coda_func_counts[coda][entry['gram_function']] += 1
                if coda == 'r':
                    new_r_count += 1

    new_verbal = broad_dist.get('VERBAL', 0.0)
    new_nominal = broad_dist.get('NOMINAL', 0.0)

    print(f"  Verbal fraction: {100*new_verbal:.1f}% (was {100*old_verbal_frac:.1f}%)")
    print(f"  Nominal fraction: {100*new_nominal:.1f}%")
    print(f"  r-coda tokens: {new_r_count} (was {old_r_count})")

    # --- Section and hand profiles ---
    section_profiles, hand_profiles = _section_and_hand_profiles(catalog)
    section_chi2_p = _chi2_contingency_p(section_profiles)
    hand_chi2_p = _chi2_contingency_p(hand_profiles)
    print(f"  Section chi² p: {section_chi2_p:.2e}")

    # --- Cross-validation ---
    xval = _cross_validation_agreement(catalog)
    print(f"  Cross-validation: {100*xval:.1f}% (was {100*old_xval:.1f}%)")

    # --- Exhaustive 6-permutation test ---
    print("  Running exhaustive 3! permutation test...")
    exhaustive = _exhaustive_coda_permutation(all_tokens, corrected_coda, broad_dist)
    print(f"    Real mapping rank: {exhaustive['real_rank']}/6 "
          f"({'BEST' if exhaustive['is_best'] else 'not best'})")

    # --- Bootstrap null test ---
    print("  Running bootstrap null test (500 trials)...")
    bootstrap = _bootstrap_coda_null(all_tokens, corrected_coda, broad_dist, n_trials=500)
    print(f"    Bootstrap p: {bootstrap['p_value']:.4f}")

    # --- Gates ---
    gate_g1 = 0.10 <= new_verbal <= 0.25
    gate_g2 = 0.15 <= new_nominal <= 0.40
    gate_g3 = exhaustive['is_best']
    gate_g4 = bootstrap['p_value'] < 0.10
    gate_g5 = section_chi2_p < 0.05
    gates_passed = sum([gate_g1, gate_g2, gate_g3, gate_g4, gate_g5])

    if gates_passed >= 4:
        verdict = 'GRAMMAR_VALIDATED'
    elif gates_passed >= 2:
        verdict = 'GRAMMAR_PARTIAL'
    else:
        verdict = 'GRAMMAR_FAILED'

    result = CorrectedGrammarResult(
        n_tokens=len(all_tokens),
        n_with_coda=n_with_coda,
        n_single_coda=n_single,
        n_double_coda=n_double,
        n_unmarked=n_unmarked,
        n_function_stem=n_func,
        grammatical_counts=dict(gram_counts.most_common()),
        broad_distribution=broad_dist,
        coda_function_counts={k: dict(v) for k, v in coda_func_counts.items()},
        section_profiles=section_profiles,
        hand_profiles=hand_profiles,
        section_chi2_p=section_chi2_p,
        hand_chi2_p=hand_chi2_p,
        null_exhaustive=exhaustive,
        null_bootstrap=bootstrap,
        cross_validation_agreement=xval,
        old_verbal_fraction=old_verbal_frac,
        new_verbal_fraction=new_verbal,
        old_r_coda_count=old_r_count,
        new_r_coda_count=new_r_count,
        old_xval=old_xval,
        gate_g1=gate_g1,
        gate_g2=gate_g2,
        gate_g3=gate_g3,
        gate_g4=gate_g4,
        gate_g5=gate_g5,
        gates_passed=gates_passed,
        verdict=verdict,
        runtime_seconds=time.time() - t0,
    )

    path = _save_json(rd, 'p73_grammar.json', asdict(result))
    print(f"\n  Verdict: {verdict} ({gates_passed}/5)")
    print(f"  Saved: {path}")
    print(f"  Runtime: {result.runtime_seconds:.1f}s")
    return result
