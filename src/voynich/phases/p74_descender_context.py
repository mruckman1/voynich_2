"""
Phase 74, Track A2: Context-Dependent Descender Analysis
========================================================
Track A1 tests which single value is best for descender across the
entire corpus. Track A2 asks: does the descender encode DIFFERENT
values depending on context?

Two sub-analyses:
  A. Position-split: do token-final and token-medial descenders
     prefer different values?
  B. Preceding-triple: does the descender encode different consonants
     after different syllables?

If descenders are context-dependent, a mixed model (e.g., final→r,
medial→null) would outperform any single global value.

Dependency chain:
    results/combined_refine.json         (Phase 15)
    results/modifier_integrate.json      (Phase 16)
    results/p69_clean_corpus.json        (Phase 69)
        -> results/p74_descender_context.json
"""

import json
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import build_eva_to_triple_lookup, load_corpus, tokenize_eva_chars
from voynich.core.reference import build_expanded_word_set, load_reference_corpus
from voynich.phases.coda_markers import (
    CodaTable,
    get_coda,
)
from voynich.phases.corrected_coda import (
    build_coda_table_v2,
    classify_token_chars_v2,
    decode_token_cvc_v2,
)
from voynich.phases.cvc_coda_signal import (
    _build_folio_list,
    _load_shared_data,
)
from voynich.phases.p74_descender import (
    _build_coda_table_with_descender,
    _convert,
    _safe_load,
    _save_json,
)


# ---------------------------------------------------------------------------
# Position-split analysis
# ---------------------------------------------------------------------------

def _classify_descender_positions(
    all_tokens: List[str],
    coda_table: CodaTable,
) -> Dict[str, List[int]]:
    """Partition token indices by descender position.

    Returns dict with keys:
    - 'final_only': tokens where all descenders are token-final
    - 'medial_only': tokens where all descenders are token-medial
    - 'both': tokens with both final and medial descenders
    - 'no_descender': tokens with no descender markers
    """
    final_only = []
    medial_only = []
    both = []
    no_descender = []

    for idx, token in enumerate(all_tokens):
        eva_chars = tokenize_eva_chars(token)
        classified = classify_token_chars_v2(eva_chars, coda_table)

        has_final = False
        has_medial = False

        for pos, (role, char) in enumerate(classified):
            if role != 'CODA_MARKER':
                continue
            last_stroke = coda_table.eva_modifiers.get(char)
            if last_stroke != 'descender':
                continue

            if pos == len(classified) - 1:
                has_final = True
            else:
                has_medial = True

        if has_final and has_medial:
            both.append(idx)
        elif has_final:
            final_only.append(idx)
        elif has_medial:
            medial_only.append(idx)
        else:
            no_descender.append(idx)

    return {
        'final_only': final_only,
        'medial_only': medial_only,
        'both': both,
        'no_descender': no_descender,
    }


def _test_values_on_subset(
    token_indices: List[int],
    all_tokens: List[str],
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    ref_word_set: Set[str],
) -> Dict[str, float]:
    """Test reduced value set on a subset of tokens.

    Returns dict of value_name -> dict_hit rate.
    """
    TEST_VALUES = {
        'r': 'r', 'null': '', 'n': 'n', 's': 's', 't': 't',
        'l': 'l', 'e': 'e', 'a': 'a',
    }

    results = {}
    for value_name, value in TEST_VALUES.items():
        coda_table = _build_coda_table_with_descender(value)

        hits = 0
        for idx in token_indices:
            token = all_tokens[idx]
            result = decode_token_cvc_v2(token, assignment, eva_to_triple,
                                         coda_table)
            if result.decoded_cvc and result.decoded_cvc.lower() in ref_word_set:
                hits += 1

        results[value_name] = hits / len(token_indices) if token_indices else 0.0

    return results


# ---------------------------------------------------------------------------
# Preceding-triple analysis
# ---------------------------------------------------------------------------

def _group_by_preceding_triple(
    all_tokens: List[str],
    coda_table: CodaTable,
    eva_to_triple: Dict[str, str],
) -> Dict[str, List[int]]:
    """Group token indices by which triple precedes the descender.

    For each descender occurrence, identify the immediately preceding
    SYLLABIC character's triple key. Return dict of triple_key -> [token_indices].
    """
    by_triple = {}

    for idx, token in enumerate(all_tokens):
        eva_chars = tokenize_eva_chars(token)
        classified = classify_token_chars_v2(eva_chars, coda_table)

        for pos, (role, char) in enumerate(classified):
            if role != 'CODA_MARKER':
                continue
            last_stroke = coda_table.eva_modifiers.get(char)
            if last_stroke != 'descender':
                continue

            # Find preceding SYLLABIC character
            if pos > 0:
                prev_role, prev_char = classified[pos - 1]
                if prev_role == 'SYLLABIC':
                    triple_key = eva_to_triple.get(prev_char, 'unknown')
                    if triple_key not in by_triple:
                        by_triple[triple_key] = []
                    by_triple[triple_key].append(idx)

    return by_triple


# ---------------------------------------------------------------------------
# Mixed model evaluation
# ---------------------------------------------------------------------------

def _evaluate_mixed_model(
    final_best: str,
    medial_best: str,
    position_groups: Dict[str, List[int]],
    all_tokens: List[str],
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    ref_word_set: Set[str],
) -> Dict[str, Any]:
    """Evaluate a mixed model where final and medial descenders use
    different values. Compare to uniform-r baseline."""
    # Build two coda tables
    final_table = _build_coda_table_with_descender(
        '' if final_best == 'null' else final_best)
    medial_table = _build_coda_table_with_descender(
        '' if medial_best == 'null' else medial_best)
    r_table = _build_coda_table_with_descender('r')

    # Score each token with the appropriate model
    mixed_hits = 0
    uniform_hits = 0
    total = 0

    # Tokens with only final descenders → use final_table
    for idx in position_groups.get('final_only', []):
        token = all_tokens[idx]
        mixed_dec = decode_token_cvc_v2(token, assignment, eva_to_triple,
                                         final_table).decoded_cvc
        r_dec = decode_token_cvc_v2(token, assignment, eva_to_triple,
                                     r_table).decoded_cvc
        if mixed_dec and mixed_dec.lower() in ref_word_set:
            mixed_hits += 1
        if r_dec and r_dec.lower() in ref_word_set:
            uniform_hits += 1
        total += 1

    # Tokens with only medial descenders → use medial_table
    for idx in position_groups.get('medial_only', []):
        token = all_tokens[idx]
        mixed_dec = decode_token_cvc_v2(token, assignment, eva_to_triple,
                                         medial_table).decoded_cvc
        r_dec = decode_token_cvc_v2(token, assignment, eva_to_triple,
                                     r_table).decoded_cvc
        if mixed_dec and mixed_dec.lower() in ref_word_set:
            mixed_hits += 1
        if r_dec and r_dec.lower() in ref_word_set:
            uniform_hits += 1
        total += 1

    # Tokens with both → use final_table (conservative)
    for idx in position_groups.get('both', []):
        token = all_tokens[idx]
        mixed_dec = decode_token_cvc_v2(token, assignment, eva_to_triple,
                                         final_table).decoded_cvc
        r_dec = decode_token_cvc_v2(token, assignment, eva_to_triple,
                                     r_table).decoded_cvc
        if mixed_dec and mixed_dec.lower() in ref_word_set:
            mixed_hits += 1
        if r_dec and r_dec.lower() in ref_word_set:
            uniform_hits += 1
        total += 1

    mixed_rate = mixed_hits / total if total > 0 else 0.0
    uniform_rate = uniform_hits / total if total > 0 else 0.0

    return {
        'mixed_dict_hit': mixed_rate,
        'uniform_r_dict_hit': uniform_rate,
        'delta': mixed_rate - uniform_rate,
        'n_tokens': total,
        'final_value': final_best,
        'medial_value': medial_best,
    }


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class DescenderContextResult:
    phase: str = "74"
    step: str = "74.A2"
    experiment: str = "descender_context_analysis"
    # Position split
    n_final_only: int = 0
    n_medial_only: int = 0
    n_both: int = 0
    n_no_descender: int = 0
    final_best_value: str = ""
    medial_best_value: str = ""
    final_results: Dict[str, float] = field(default_factory=dict)
    medial_results: Dict[str, float] = field(default_factory=dict)
    context_dependent_position: bool = False
    # Mixed model
    mixed_model: Dict[str, Any] = field(default_factory=dict)
    # Preceding triple
    per_triple_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    n_triples_prefer_r: int = 0
    n_triples_prefer_other: int = 0
    context_dependent_triple: bool = False
    # Gates
    gate_da5: bool = False   # Context-dependent by position
    gate_da6: bool = False   # ≥3 triples prefer non-r
    gate_da7: bool = False   # Mixed model > uniform-r
    gates_passed: int = 0
    gate_passed: bool = False
    verdict: str = "UNKNOWN"
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_descender_context():
    """Track A2: Context-dependent descender analysis."""
    t0 = time.time()
    rd = str(_results_dir())

    print("Phase 74.A2 — Context-Dependent Descender Analysis")
    print("=" * 52)

    # --- Load shared data ---
    print("  Loading shared data...")
    shared = _load_shared_data()
    all_tokens = shared['all_tokens']
    assignment = shared['assignment']
    eva_to_triple = shared['eva_to_triple']
    ref_word_set = shared['ref_word_set']

    # Build base coda table (connector→null)
    base_coda_table = build_coda_table_v2()
    base_coda_table.stroke_to_coda['connector'] = ''

    # =====================================================================
    # Sub-analysis A: Position-split testing
    # =====================================================================
    print("\n  A. Position-split analysis...")

    position_groups = _classify_descender_positions(all_tokens, base_coda_table)
    n_final = len(position_groups['final_only'])
    n_medial = len(position_groups['medial_only'])
    n_both = len(position_groups['both'])
    n_none = len(position_groups['no_descender'])

    print(f"    Final-only tokens: {n_final}")
    print(f"    Medial-only tokens: {n_medial}")
    print(f"    Both: {n_both}")
    print(f"    No descender: {n_none}")

    # Test values on each partition
    final_results = {}
    medial_results = {}

    if n_final >= 20:
        print(f"    Testing values on {n_final} final-descender tokens...")
        final_results = _test_values_on_subset(
            position_groups['final_only'], all_tokens, assignment,
            eva_to_triple, ref_word_set)
        final_best = max(final_results, key=final_results.get)
        print(f"    Final best: {final_best} ({final_results[final_best]:.3f})")
        for name in sorted(final_results, key=final_results.get, reverse=True)[:5]:
            marker = " <-- CURRENT" if name == 'r' else ""
            print(f"      {name}: {final_results[name]:.3f}{marker}")
    else:
        final_best = 'r'
        print(f"    Too few final-descender tokens ({n_final}), defaulting to r")

    if n_medial >= 20:
        print(f"    Testing values on {n_medial} medial-descender tokens...")
        medial_results = _test_values_on_subset(
            position_groups['medial_only'], all_tokens, assignment,
            eva_to_triple, ref_word_set)
        medial_best = max(medial_results, key=medial_results.get)
        print(f"    Medial best: {medial_best} ({medial_results[medial_best]:.3f})")
        for name in sorted(medial_results, key=medial_results.get, reverse=True)[:5]:
            marker = " <-- CURRENT" if name == 'r' else ""
            print(f"      {name}: {medial_results[name]:.3f}{marker}")
    else:
        medial_best = 'r'
        print(f"    Too few medial-descender tokens ({n_medial}), defaulting to r")

    ctx_dep_position = final_best != medial_best
    print(f"\n    Context-dependent by position: {ctx_dep_position}")
    if ctx_dep_position:
        print(f"    → Final prefers '{final_best}', medial prefers '{medial_best}'")

    # --- Mixed model evaluation ---
    print("\n  Evaluating mixed model...")
    if ctx_dep_position:
        mixed = _evaluate_mixed_model(
            final_best, medial_best, position_groups,
            all_tokens, assignment, eva_to_triple, ref_word_set)
        print(f"    Mixed model dict-hit: {mixed['mixed_dict_hit']:.3f}")
        print(f"    Uniform-r dict-hit:   {mixed['uniform_r_dict_hit']:.3f}")
        print(f"    Delta:                {mixed['delta']:+.3f}")
    else:
        mixed = {
            'mixed_dict_hit': 0.0, 'uniform_r_dict_hit': 0.0,
            'delta': 0.0, 'n_tokens': 0,
            'final_value': final_best, 'medial_value': medial_best,
        }
        print("    Skipped (not context-dependent)")

    # =====================================================================
    # Sub-analysis B: Preceding-triple testing
    # =====================================================================
    print("\n  B. Preceding-triple analysis...")

    triple_groups = _group_by_preceding_triple(
        all_tokens, base_coda_table, eva_to_triple)

    per_triple_results = {}
    n_prefer_r = 0
    n_prefer_other = 0

    for triple_key in sorted(triple_groups, key=lambda k: -len(triple_groups[k])):
        indices = triple_groups[triple_key]
        if len(indices) < 20:
            continue

        triple_vals = _test_values_on_subset(
            indices, all_tokens, assignment, eva_to_triple, ref_word_set)
        best = max(triple_vals, key=triple_vals.get)
        prefers_r = (best == 'r')

        per_triple_results[triple_key] = {
            'n_tokens': len(indices),
            'best_value': best,
            'best_dict_hit': triple_vals[best],
            'r_dict_hit': triple_vals.get('r', 0.0),
            'prefers_r': prefers_r,
            'all_results': triple_vals,
        }

        if prefers_r:
            n_prefer_r += 1
        else:
            n_prefer_other += 1

        marker = " (same as current)" if prefers_r else " (DIFFERENT)"
        print(f"    {triple_key}: best={best} ({triple_vals[best]:.3f}), "
              f"r={triple_vals.get('r', 0):.3f}, n={len(indices)}{marker}")

    ctx_dep_triple = n_prefer_r > 0 and n_prefer_other > 0

    print(f"\n    Triples preferring r: {n_prefer_r}")
    print(f"    Triples preferring other: {n_prefer_other}")
    print(f"    Context-dependent by triple: {ctx_dep_triple}")

    # =====================================================================
    # Gates
    # =====================================================================
    g5 = ctx_dep_position
    g6 = n_prefer_other >= 3
    g7 = mixed.get('delta', 0.0) > 0.005

    gates_passed = sum([g5, g6, g7])

    print(f"\n  Gates:")
    print(f"    DA5 (context-dep by position): {'PASS' if g5 else 'FAIL'}")
    print(f"    DA6 (≥3 triples prefer non-r): {'PASS' if g6 else 'FAIL'} "
          f"({n_prefer_other} triples)")
    print(f"    DA7 (mixed > uniform-r): {'PASS' if g7 else 'FAIL'} "
          f"(delta={mixed.get('delta', 0):.4f})")
    print(f"    Total: {gates_passed}/3")

    # --- Verdict ---
    if g5 and g6 and g7:
        verdict = 'CONTEXT_DEPENDENT_CONFIRMED'
    elif g5 or g6:
        verdict = 'CONTEXT_DEPENDENT_PARTIAL'
    elif n_prefer_other > 0:
        verdict = 'WEAK_CONTEXT_SIGNAL'
    else:
        verdict = 'UNIFORM_R_CONFIRMED'

    print(f"\n  Verdict: {verdict}")

    # --- Build result ---
    result = DescenderContextResult(
        n_final_only=n_final,
        n_medial_only=n_medial,
        n_both=n_both,
        n_no_descender=n_none,
        final_best_value=final_best,
        medial_best_value=medial_best,
        final_results=final_results,
        medial_results=medial_results,
        context_dependent_position=ctx_dep_position,
        mixed_model=mixed,
        per_triple_results=per_triple_results,
        n_triples_prefer_r=n_prefer_r,
        n_triples_prefer_other=n_prefer_other,
        context_dependent_triple=ctx_dep_triple,
        gate_da5=g5,
        gate_da6=g6,
        gate_da7=g7,
        gates_passed=gates_passed,
        gate_passed=gates_passed >= 1,
        verdict=verdict,
        runtime_seconds=time.time() - t0,
    )

    path = _save_json(rd, 'p74_descender_context.json', asdict(result))
    print(f"\n  Saved: {path}")
    print(f"  Runtime: {result.runtime_seconds:.1f}s")

    return result
