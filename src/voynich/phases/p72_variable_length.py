"""
Phase 72, Track 5: Variable-Length Encoding Hypothesis
======================================================
The T_P15 table assigns every syllabic character a fixed 2-character CV
value ("ra", "di", "co").  But what if some characters encode 1 character
(just a vowel "a" or consonant "r"), and some encode 3 characters
(a CVC syllable "ran" or consonant cluster "str")?

This track tests per-triple length preferences via:
  1. Individual sweep: 12 confirmed triples × 7 length variants on subsample
  2. Greedy assembly: hill-climbing from best individual picks
  3. Final evaluation: full corpus decode + signal + bigram_z

Dependency chain:
    results/combined_refine.json         (Phase 15)
    results/triple_tiers.json            (Phase 28/53)
    results/p69_clean_corpus.json        (Phase 69)
    results/modifier_integrate.json      (Phase 16)
    results/null_corpus.json             (Phase 17)
        -> results/phase72_var_length.json
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
from voynich.phases.corrected_coda import (
    build_coda_table_v2,
    decode_corpus_cvc_v2,
)
from voynich.phases.cvc_coda_signal import (
    _build_folio_list,
    _compute_bigram_z,
    _load_shared_data,
    _run_signal_isolation,
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
# Triple tier loading
# ---------------------------------------------------------------------------

def _get_confirmed_and_unresolved(rd: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    refine_data = _safe_load(os.path.join(rd, 'combined_refine.json'))
    assignment = refine_data.get('best_assignment', {})

    tier_data = _safe_load(os.path.join(rd, 'triple_tiers.json'))
    confirmed_keys: Set[str] = set()

    if tier_data and 'tiers' in tier_data:
        tiers = tier_data['tiers']
        if isinstance(tiers, dict):
            for entry in tiers.get('CONFIRMED', []):
                confirmed_keys.add(entry.get('triple_key', ''))
        elif isinstance(tiers, list):
            for entry in tiers:
                if entry.get('tier', '') == 'CONFIRMED':
                    confirmed_keys.add(entry.get('triple_key', ''))

    confirmed = {k: v for k, v in assignment.items() if k in confirmed_keys}
    unresolved = {k: v for k, v in assignment.items() if k not in confirmed_keys}
    return confirmed, unresolved


# ---------------------------------------------------------------------------
# Generate alternative values
# ---------------------------------------------------------------------------

def _generate_alt_values(current_cv: str) -> Dict[str, str]:
    """Generate 1-char and 3-char alternatives for a 2-char CV value.

    For a CV like "di":
      1-char: "d" (onset only), "i" (vowel only)
      3-char: "din", "dir", "dis", "dit" (CVC with common codas)
    """
    if len(current_cv) < 2:
        return {'current': current_cv}

    onset = current_cv[0]
    vowel = current_cv[-1]
    vowels = set('aeiou')

    variants = {'current': current_cv}

    # 1-char alternatives
    if onset not in vowels:
        variants['onset_only'] = onset
    variants['vowel_only'] = vowel

    # 3-char alternatives: append common Latin codas
    for coda in ['n', 'r', 's', 't']:
        variants[f'cvc_{coda}'] = current_cv + coda

    return variants


# ---------------------------------------------------------------------------
# Fast dict-hit evaluation
# ---------------------------------------------------------------------------

def _evaluate_assignment(
    assignment: Dict[str, str],
    tokens: List[str],
    eva_to_triple: Dict[str, str],
    coda_table,
    ref_word_set: Set[str],
) -> float:
    """Decode tokens with given assignment, return dict_hit."""
    decoded = decode_corpus_cvc_v2(tokens, assignment, eva_to_triple, coda_table)
    hits = sum(1 for d in decoded if d and d.lower() in ref_word_set)
    return hits / len(decoded) if decoded else 0.0


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class TripleLengthResult:
    triple_key: str = ""
    current_value: str = ""
    best_variant_name: str = ""
    best_value: str = ""
    best_length: int = 2
    best_dict_hit: float = 0.0
    current_dict_hit: float = 0.0
    improvement: float = 0.0
    prefers_shorter: bool = False
    prefers_longer: bool = False
    all_variants: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class VarLengthResult:
    phase: str = "72"
    step: str = "72.5"
    experiment: str = "variable_length"
    # Per-triple results
    per_triple: List[TripleLengthResult] = field(default_factory=list)
    n_prefer_shorter: int = 0
    n_prefer_longer: int = 0
    n_prefer_standard: int = 0
    # Greedy optimization
    greedy_best_assignment: Dict[str, str] = field(default_factory=dict)
    greedy_dict_hit: float = 0.0
    greedy_signal_count: int = 0
    greedy_bigram_z: float = 0.0
    greedy_mean_length: float = 0.0
    n_changed: int = 0
    # Baseline
    baseline_dict_hit: float = 0.0
    baseline_signal_count: int = 0
    baseline_bigram_z: float = 0.0
    baseline_mean_length: float = 0.0
    improvement: float = 0.0
    # Gates
    gate_vl1: bool = False   # >= 2 triples prefer shorter (1-char)
    gate_vl2: bool = False   # >= 2 triples prefer longer (3-char)
    gate_vl3: bool = False   # Joint dict-hit > current + 2pp
    gate_vl4: bool = False   # Mean decoded length closer to 5.8
    gate_vl5: bool = False   # Greedy bigram_z >= 90% of baseline
    gates_passed: int = 0
    gate_passed: bool = False
    verdict: str = "UNKNOWN"
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_variable_length():
    """Track 5: Test variable-length encoding hypothesis."""
    t0 = time.time()
    rd = str(_results_dir())

    print("Phase 72.5 — Variable-Length Encoding Hypothesis")
    print("=" * 50)

    # --- Load data ---
    print("  Loading shared data...")
    shared = _load_shared_data()
    all_tokens = shared['all_tokens']
    folios = shared['folios']
    eva_to_triple = shared['eva_to_triple']
    ref_word_set = shared['ref_word_set']
    null_token_lists = shared['null_token_lists']

    confirmed, unresolved = _get_confirmed_and_unresolved(rd)
    full_assignment = {**confirmed, **unresolved}
    confirmed_keys = list(confirmed.keys())

    coda_table = build_coda_table_v2()

    print(f"  Tokens: {len(all_tokens)}")
    print(f"  Confirmed triples: {len(confirmed)}")

    # --- Subsample for individual sweep ---
    rng = np.random.default_rng(seed=42)
    subsample_size = min(5000, len(all_tokens))
    subsample_indices = sorted(rng.choice(len(all_tokens), size=subsample_size, replace=False))
    subsample_tokens = [all_tokens[i] for i in subsample_indices]

    print(f"  Subsample: {subsample_size} tokens")

    # --- Step 1: Individual sweep ---
    print("\n  Step 1: Individual triple-length sweep...")
    per_triple_results = []

    for triple_key in confirmed_keys:
        current_value = confirmed[triple_key]
        variants = _generate_alt_values(current_value)

        variant_scores = {}

        for var_name, var_value in variants.items():
            # Build test assignment
            test_assignment = dict(full_assignment)
            test_assignment[triple_key] = var_value

            # Evaluate on subsample
            dict_hit = _evaluate_assignment(
                test_assignment, subsample_tokens, eva_to_triple,
                coda_table, ref_word_set)

            # Mean decoded length
            decoded = decode_corpus_cvc_v2(
                subsample_tokens, test_assignment, eva_to_triple, coda_table)
            mean_len = float(np.mean([len(d) for d in decoded if d]))

            variant_scores[var_name] = {
                'value': var_value,
                'length': len(var_value),
                'dict_hit': dict_hit,
                'mean_decoded_length': mean_len,
            }

        # Find best
        best_name = max(variant_scores, key=lambda k: variant_scores[k]['dict_hit'])
        best = variant_scores[best_name]

        result = TripleLengthResult(
            triple_key=triple_key,
            current_value=current_value,
            best_variant_name=best_name,
            best_value=best['value'],
            best_length=best['length'],
            best_dict_hit=best['dict_hit'],
            current_dict_hit=variant_scores['current']['dict_hit'],
            improvement=best['dict_hit'] - variant_scores['current']['dict_hit'],
            prefers_shorter=best['length'] < len(current_value),
            prefers_longer=best['length'] > len(current_value),
            all_variants=variant_scores,
        )
        per_triple_results.append(result)

        status = ""
        if result.prefers_shorter:
            status = " [SHORTER]"
        elif result.prefers_longer:
            status = " [LONGER]"
        print(f"    {triple_key} ({current_value}): best={best['value']} "
              f"(dict_hit={best['dict_hit']:.3f} vs {variant_scores['current']['dict_hit']:.3f})"
              f"{status}")

    n_shorter = sum(1 for r in per_triple_results if r.prefers_shorter)
    n_longer = sum(1 for r in per_triple_results if r.prefers_longer)
    n_standard = len(per_triple_results) - n_shorter - n_longer

    print(f"\n  Prefer shorter: {n_shorter}")
    print(f"  Prefer longer: {n_longer}")
    print(f"  Prefer standard: {n_standard}")

    # --- Step 2: Greedy hill-climbing ---
    print("\n  Step 2: Greedy hill-climbing assembly...")

    # Start from baseline
    greedy_assignment = dict(full_assignment)

    # Apply best individual picks for confirmed triples
    for result in per_triple_results:
        if result.best_dict_hit > result.current_dict_hit:
            greedy_assignment[result.triple_key] = result.best_value

    # Greedy refinement: try flipping each confirmed triple
    improved = True
    iteration = 0
    while improved and iteration < 3:
        improved = False
        iteration += 1
        for triple_key in confirmed_keys:
            current_value = greedy_assignment[triple_key]
            variants = _generate_alt_values(confirmed[triple_key])

            best_value = current_value
            best_score = _evaluate_assignment(
                greedy_assignment, subsample_tokens, eva_to_triple,
                coda_table, ref_word_set)

            for var_name, var_value in variants.items():
                if var_value == current_value:
                    continue
                test = dict(greedy_assignment)
                test[triple_key] = var_value
                score = _evaluate_assignment(
                    test, subsample_tokens, eva_to_triple,
                    coda_table, ref_word_set)
                if score > best_score + 0.001:
                    best_score = score
                    best_value = var_value
                    improved = True

            greedy_assignment[triple_key] = best_value

        print(f"    Iteration {iteration}: dict_hit={best_score:.4f}")

    n_changed = sum(1 for k in confirmed_keys
                    if greedy_assignment.get(k) != full_assignment.get(k))

    # --- Step 3: Full corpus evaluation ---
    print("\n  Step 3: Full corpus evaluation...")

    # Baseline
    baseline_decoded = decode_corpus_cvc_v2(
        all_tokens, full_assignment, eva_to_triple, coda_table)
    baseline_dict_hit = sum(1 for d in baseline_decoded
                           if d and d.lower() in ref_word_set) / len(baseline_decoded)
    baseline_mean_len = float(np.mean([len(d) for d in baseline_decoded if d]))

    # Null decoded for signal
    null_decoded_list = []
    for null_tokens in null_token_lists:
        null_dec = decode_corpus_cvc_v2(
            null_tokens, full_assignment, eva_to_triple, coda_table)
        null_decoded_list.append(null_dec)

    baseline_signal = _run_signal_isolation(
        baseline_decoded, null_decoded_list, ref_word_set, len(baseline_decoded))
    baseline_bigram_z = _compute_bigram_z(
        baseline_decoded, null_decoded_list, ref_word_set, folios, n_perms=200)

    print(f"  Baseline: dict_hit={baseline_dict_hit:.3f}, "
          f"signal={len(baseline_signal.top_signal_words)}, "
          f"bigram_z={baseline_bigram_z:.2f}, mean_len={baseline_mean_len:.1f}")

    # Greedy
    greedy_decoded = decode_corpus_cvc_v2(
        all_tokens, greedy_assignment, eva_to_triple, coda_table)
    greedy_dict_hit = sum(1 for d in greedy_decoded
                         if d and d.lower() in ref_word_set) / len(greedy_decoded)
    greedy_mean_len = float(np.mean([len(d) for d in greedy_decoded if d]))

    null_decoded_greedy = []
    for null_tokens in null_token_lists:
        null_dec = decode_corpus_cvc_v2(
            null_tokens, greedy_assignment, eva_to_triple, coda_table)
        null_decoded_greedy.append(null_dec)

    greedy_signal = _run_signal_isolation(
        greedy_decoded, null_decoded_greedy, ref_word_set, len(greedy_decoded))
    greedy_bigram_z = _compute_bigram_z(
        greedy_decoded, null_decoded_greedy, ref_word_set, folios, n_perms=200)

    print(f"  Greedy:   dict_hit={greedy_dict_hit:.3f}, "
          f"signal={len(greedy_signal.top_signal_words)}, "
          f"bigram_z={greedy_bigram_z:.2f}, mean_len={greedy_mean_len:.1f}")
    print(f"  Changed:  {n_changed}/{len(confirmed_keys)} triples")
    print(f"  Improvement: {greedy_dict_hit - baseline_dict_hit:+.3f}")

    # Show changes
    if n_changed > 0:
        print("\n  Changes:")
        for k in confirmed_keys:
            if greedy_assignment.get(k) != full_assignment.get(k):
                print(f"    {k}: '{full_assignment[k]}' -> '{greedy_assignment[k]}'")

    # --- Gates ---
    # Latin mean word length ~5.8 chars
    baseline_len_diff = abs(baseline_mean_len - 5.8)
    greedy_len_diff = abs(greedy_mean_len - 5.8)

    g1 = n_shorter >= 2
    g2 = n_longer >= 2
    g3 = greedy_dict_hit > baseline_dict_hit + 0.02
    g4 = greedy_len_diff < baseline_len_diff
    g5 = greedy_bigram_z >= baseline_bigram_z * 0.90

    gates_passed = sum([g1, g2, g3, g4, g5])

    print(f"\n  Gates:")
    print(f"    VL1 (>= 2 prefer shorter): {'PASS' if g1 else 'FAIL'} ({n_shorter})")
    print(f"    VL2 (>= 2 prefer longer): {'PASS' if g2 else 'FAIL'} ({n_longer})")
    print(f"    VL3 (dict-hit > baseline + 2pp): {'PASS' if g3 else 'FAIL'}")
    print(f"    VL4 (length closer to 5.8): {'PASS' if g4 else 'FAIL'} "
          f"({greedy_mean_len:.1f} vs {baseline_mean_len:.1f})")
    print(f"    VL5 (bigram_z >= 90% baseline): {'PASS' if g5 else 'FAIL'}")
    print(f"    Total: {gates_passed}/5")

    # --- Verdict ---
    if g3 and g5 and gates_passed >= 3:
        verdict = 'VARIABLE_LENGTH_CONFIRMED'
    elif g1 or g2:
        verdict = 'LENGTH_PREFERENCES_FOUND'
    elif gates_passed >= 2:
        verdict = 'MARGINAL_IMPROVEMENT'
    else:
        verdict = 'FIXED_LENGTH_CONFIRMED'

    print(f"\n  Verdict: {verdict}")

    # --- Build result ---
    result = VarLengthResult(
        per_triple=per_triple_results,
        n_prefer_shorter=n_shorter,
        n_prefer_longer=n_longer,
        n_prefer_standard=n_standard,
        greedy_best_assignment={k: greedy_assignment[k] for k in confirmed_keys},
        greedy_dict_hit=greedy_dict_hit,
        greedy_signal_count=len(greedy_signal.top_signal_words),
        greedy_bigram_z=greedy_bigram_z,
        greedy_mean_length=greedy_mean_len,
        n_changed=n_changed,
        baseline_dict_hit=baseline_dict_hit,
        baseline_signal_count=len(baseline_signal.top_signal_words),
        baseline_bigram_z=baseline_bigram_z,
        baseline_mean_length=baseline_mean_len,
        improvement=greedy_dict_hit - baseline_dict_hit,
        gate_vl1=g1,
        gate_vl2=g2,
        gate_vl3=g3,
        gate_vl4=g4,
        gate_vl5=g5,
        gates_passed=gates_passed,
        gate_passed=gates_passed >= 2,
        verdict=verdict,
        runtime_seconds=time.time() - t0,
    )

    path = _save_json(rd, 'phase72_var_length.json', asdict(result))
    print(f"\n  Saved: {path}")
    print(f"  Runtime: {result.runtime_seconds:.1f}s")

    return result
