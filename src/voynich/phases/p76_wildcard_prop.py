"""
Phase 76, Track 1: T1 Wildcard Constraint Extraction + LOO Validation + Application
====================================================================================
Extract character-level constraints on unresolved triples from T1 identifications.
Each T1 identification maps an EVA token to a Latin word -- if the token contains
unresolved triples (wildcards in the pattern), the matched word fills those
wildcards, constraining the triple.  LOO cross-validate on confirmed triples,
then apply.

Dependency chain:
    results/p75_t1.json               (Phase 75 Track 3)
    results/combined_refine.json      (Phase 15)
    results/triple_tiers.json         (Phase 28/53)
        -> results/p76_wildcard_prop.json
"""

import json
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    load_corpus,
    tokenize_eva_chars,
)
from voynich.core.reference import build_expanded_word_set, load_reference_corpus
from voynich.phases.coda_markers import get_coda
from voynich.phases.corrected_coda import (
    classify_token_chars_v2,
    decode_token_cvc_v2,
)
from voynich.phases.p68_expanded_t1 import (
    _aggregate_constraints,
    _build_dict_by_length,
    _build_patterns,
    _extract_constraints,
    _match_patterns,
)
from voynich.phases.p69_clean_corpus import _classify_token_confidence
from voynich.phases.p69_clean_validation import _get_confirmed_and_unresolved
from voynich.phases.p75_redecode import _build_3coda_table


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
class WildcardPropResult:
    phase: str = "76"
    step: str = "76.1"
    experiment: str = "wildcard_propagation"
    # Extraction
    n_identifications: int = 0
    n_constrained_triples: int = 0
    mean_consistency: float = 0.0
    per_triple_resolution: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    n_resolved: int = 0
    n_likely: int = 0
    n_tentative: int = 0
    n_insufficient: int = 0
    # LOO cross-validation
    loo_results: List[Dict[str, Any]] = field(default_factory=list)
    loo_recovery_rate: float = 0.0
    loo_validated: bool = False
    # Application
    updated_dict_hit: float = 0.0
    baseline_dict_hit: float = 0.0
    dict_improvement: float = 0.0
    old_clean_fraction: float = 0.0
    new_clean_fraction: float = 0.0
    signal_count: int = 0
    baseline_signal_count: int = 0
    # Gates
    w1_loo_recovery: bool = False
    w2_n_resolved: bool = False
    w3_resolved_plus_likely: bool = False
    w4_clean_fraction: bool = False
    w5_dict_improvement: bool = False
    w6_signal_maintained: bool = False
    gates_passed: int = 0
    gate_passed: bool = False
    verdict: str = "UNKNOWN"
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Confidence classification for triple constraints
# ---------------------------------------------------------------------------

def _classify_confidence(n_obs: int, consistency: float) -> str:
    """Assign confidence tier based on observation count and consistency."""
    if n_obs >= 10 and consistency > 0.80:
        return 'RESOLVED'
    if n_obs >= 5 and consistency > 0.60:
        return 'LIKELY'
    if n_obs >= 3 and consistency > 0.50:
        return 'TENTATIVE'
    return 'INSUFFICIENT'


# ---------------------------------------------------------------------------
# Parse triple_details into per-triple resolution dict
# ---------------------------------------------------------------------------

def _build_per_triple_resolution(
    triple_candidates: Dict[str, str],
    triple_details: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Build per-triple resolution dictionary with confidence tiers."""
    resolution: Dict[str, Dict[str, Any]] = {}

    for detail in triple_details:
        triple_key = detail['triple_key']
        best_syllable = detail['best_syllable']
        total_obs = detail['total_obs']
        positions = detail.get('positions', {})

        # Compute overall consistency across positions
        consistencies = []
        onset_dist: Dict[str, int] = {}
        vowel_dist: Dict[str, int] = {}

        for pos_str, pos_info in positions.items():
            consistencies.append(pos_info.get('consistency', 0.0))
            all_chars = pos_info.get('all_chars', {})
            pos_idx = int(pos_str)
            if pos_idx == 0:
                onset_dist = dict(all_chars)
            elif pos_idx == 1:
                vowel_dist = dict(all_chars)

        consistency = (sum(consistencies) / len(consistencies)
                       if consistencies else 0.0)
        confidence = _classify_confidence(total_obs, consistency)

        resolution[triple_key] = {
            'n_obs': total_obs,
            'best_syllable': best_syllable,
            'consistency': round(consistency, 4),
            'confidence': confidence,
            'onset_dist': onset_dist,
            'vowel_dist': vowel_dist,
        }

    return resolution


# ---------------------------------------------------------------------------
# LOO cross-validation
# ---------------------------------------------------------------------------

def _run_loo_validation(
    confirmed: Dict[str, str],
    unresolved: Dict[str, str],
    eva_to_triple: Dict[str, str],
    coda_table,
    corpus,
    all_tokens: List[str],
    ref_word_set: Set[str],
    dict_by_length: Dict[int, List[str]],
) -> Tuple[List[Dict[str, Any]], float]:
    """Leave-one-out cross-validation on confirmed triples.

    For each confirmed triple, move it to unresolved, re-run the
    pipeline, and check if the constraint extraction recovers its
    true value.
    """
    confirmed_keys_full = set(confirmed.keys())
    loo_results: List[Dict[str, Any]] = []
    n_correct = 0
    n_tested = 0

    token_types = sorted(set(all_tokens))

    for target_key in sorted(confirmed.keys()):
        true_value = confirmed[target_key]

        # Move target to unresolved
        loo_confirmed = {k: v for k, v in confirmed.items() if k != target_key}
        loo_unresolved = dict(unresolved)
        loo_unresolved[target_key] = true_value  # keep value in full assignment
        loo_confirmed_keys = set(loo_confirmed.keys())
        loo_full_assignment = {**loo_confirmed, **loo_unresolved}

        # Build patterns with target triple now unresolved
        patterns = _build_patterns(
            token_types, loo_full_assignment, eva_to_triple, coda_table,
            loo_confirmed_keys, min_known_frac=0.50)

        # Match
        all_matches = _match_patterns(patterns, dict_by_length, max_matches=20)

        # Extract constraints
        identifications, triple_constraints = _extract_constraints(
            patterns, all_matches, corpus, all_tokens, min_folios=3)

        # Aggregate
        triple_candidates, triple_details, mean_cons = _aggregate_constraints(
            triple_constraints, patterns, loo_full_assignment, eva_to_triple,
            coda_table, loo_confirmed_keys)

        # Check recovery
        recovered_value = triple_candidates.get(target_key, '')
        correct = recovered_value == true_value
        n_tested += 1
        if correct:
            n_correct += 1

        loo_results.append({
            'triple_key': target_key,
            'true_value': true_value,
            'recovered_value': recovered_value,
            'correct': correct,
            'n_constraints': len(triple_constraints.get(target_key, [])),
            'n_identifications': len(identifications),
        })

    recovery_rate = n_correct / n_tested if n_tested > 0 else 0.0
    return loo_results, recovery_rate


# ---------------------------------------------------------------------------
# Compute clean fraction with updated assignment
# ---------------------------------------------------------------------------

def _compute_clean_fraction(
    all_tokens: List[str],
    eva_to_triple: Dict[str, str],
    confirmed_keys: Set[str],
    coda_table,
) -> float:
    """Compute fraction of tokens where all syllabic chars map to confirmed triples."""
    n_clean = 0
    for token in all_tokens:
        n_confirmed, n_coda, n_unresolved = _classify_token_confidence(
            token, eva_to_triple, confirmed_keys, coda_table)
        if n_unresolved == 0 and (n_confirmed + n_coda) > 0:
            n_clean += 1
    return n_clean / len(all_tokens) if all_tokens else 0.0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_wildcard_prop() -> WildcardPropResult:
    """Track 1: T1 wildcard constraint extraction, LOO validation, and application."""
    t0 = time.time()
    rd = str(_results_dir())

    print("Phase 76.1 -- T1 Wildcard Constraint Extraction + LOO Validation")
    print("=" * 66)

    # --- Load shared data ---
    confirmed, unresolved = _get_confirmed_and_unresolved(rd)
    confirmed_keys = set(confirmed.keys())
    full_assignment = {**confirmed, **unresolved}

    eva_to_triple = build_eva_to_triple_lookup()
    coda_table = _build_3coda_table()

    corpus = load_corpus(verbose=False)
    all_tokens = corpus.get_tokens()

    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    base_words = set(w.lower() for w in ref_corpus.get_combined_tokens('latin')
                     if len(w) >= 2)
    expanded, _ = build_expanded_word_set(base_words)
    ref_word_set = base_words | expanded
    dict_by_length = _build_dict_by_length(ref_word_set)

    print(f"  Confirmed triples: {len(confirmed)}")
    print(f"  Unresolved triples: {len(unresolved)}")
    print(f"  Corpus tokens: {len(all_tokens)}")
    print(f"  Dictionary size: {len(ref_word_set)}")

    # Load T1 identifications from Phase 75
    t1_data = _safe_load(os.path.join(rd, 'p75_t1.json'))
    t1_identifications = t1_data.get('identifications', [])
    print(f"  T1 identifications (Phase 75): {len(t1_identifications)}")

    # ===================================================================
    # Step 1a: Extract constraints
    # ===================================================================
    print("\n  Step 1a: Extract constraints from T1 identifications")
    print("  " + "-" * 55)

    token_types = sorted(set(all_tokens))

    print("  Building wildcard patterns...")
    patterns = _build_patterns(
        token_types, full_assignment, eva_to_triple, coda_table,
        confirmed_keys, min_known_frac=0.50)
    print(f"  Patterns built: {len(patterns)}")

    print("  Matching against dictionary...")
    all_matches = _match_patterns(patterns, dict_by_length, max_matches=20)
    n_unique = sum(1 for m in all_matches if len(m) == 1)
    print(f"  Unique matches: {n_unique}")

    print("  Extracting identifications and constraints...")
    identifications, triple_constraints = _extract_constraints(
        patterns, all_matches, corpus, all_tokens, min_folios=3)
    n_identifications = len(identifications)
    print(f"  Identifications: {n_identifications}")

    print("  Aggregating constraints...")
    triple_candidates, triple_details, mean_consistency = _aggregate_constraints(
        triple_constraints, patterns, full_assignment, eva_to_triple,
        coda_table, confirmed_keys)

    n_constrained = len(triple_candidates)
    print(f"  Triples constrained: {n_constrained}")
    print(f"  Mean consistency: {mean_consistency:.1%}")

    # Build per-triple resolution with confidence tiers
    per_triple_resolution = _build_per_triple_resolution(
        triple_candidates, triple_details)

    n_resolved = sum(1 for v in per_triple_resolution.values()
                     if v['confidence'] == 'RESOLVED')
    n_likely = sum(1 for v in per_triple_resolution.values()
                   if v['confidence'] == 'LIKELY')
    n_tentative = sum(1 for v in per_triple_resolution.values()
                      if v['confidence'] == 'TENTATIVE')
    n_insufficient = sum(1 for v in per_triple_resolution.values()
                         if v['confidence'] == 'INSUFFICIENT')

    print(f"\n  Confidence tiers:")
    print(f"    RESOLVED:     {n_resolved}")
    print(f"    LIKELY:       {n_likely}")
    print(f"    TENTATIVE:    {n_tentative}")
    print(f"    INSUFFICIENT: {n_insufficient}")

    for tk, info in sorted(per_triple_resolution.items(),
                           key=lambda x: -x[1]['n_obs']):
        print(f"    {tk}: '{info['best_syllable']}' "
              f"({info['n_obs']} obs, {info['consistency']:.1%}, "
              f"{info['confidence']})")

    # ===================================================================
    # Step 1b: LOO cross-validation
    # ===================================================================
    print(f"\n  Step 1b: LOO cross-validation on {len(confirmed)} confirmed triples")
    print("  " + "-" * 55)

    loo_results, loo_recovery_rate = _run_loo_validation(
        confirmed, unresolved, eva_to_triple, coda_table,
        corpus, all_tokens, ref_word_set, dict_by_length)

    n_loo_correct = sum(1 for r in loo_results if r['correct'])
    n_loo_tested = len(loo_results)
    print(f"  LOO results: {n_loo_correct}/{n_loo_tested} correct "
          f"({loo_recovery_rate:.1%})")

    for r in loo_results:
        status = 'OK' if r['correct'] else 'MISS'
        print(f"    {r['triple_key']}: true='{r['true_value']}' "
              f"recovered='{r['recovered_value']}' "
              f"[{status}, {r['n_constraints']} constraints]")

    loo_validated = loo_recovery_rate > 0.50

    # ===================================================================
    # Step 1c: Apply and evaluate
    # ===================================================================
    print(f"\n  Step 1c: Apply RESOLVED/LIKELY constraints and evaluate")
    print("  " + "-" * 55)

    # Compute baseline dict-hit and clean fraction
    print("  Computing baseline metrics...")
    baseline_decoded = []
    for token in all_tokens:
        result = decode_token_cvc_v2(token, full_assignment, eva_to_triple, coda_table)
        baseline_decoded.append(result.decoded_cvc)

    baseline_dict_count = sum(1 for d in baseline_decoded
                              if d and d.lower() in ref_word_set)
    baseline_dict_hit = baseline_dict_count / len(all_tokens) if all_tokens else 0.0

    old_clean_fraction = _compute_clean_fraction(
        all_tokens, eva_to_triple, confirmed_keys, coda_table)

    print(f"  Baseline dict-hit: {baseline_dict_hit:.1%}")
    print(f"  Baseline clean fraction: {old_clean_fraction:.1%}")

    # Build updated assignment: replace RESOLVED/LIKELY triples
    updated_assignment = dict(full_assignment)
    promoted_keys: Set[str] = set()

    for triple_key, info in per_triple_resolution.items():
        if info['confidence'] in ('RESOLVED', 'LIKELY'):
            updated_assignment[triple_key] = info['best_syllable']
            promoted_keys.add(triple_key)

    print(f"  Promoted triples: {len(promoted_keys)} "
          f"(RESOLVED={n_resolved}, LIKELY={n_likely})")

    # Re-decode with updated assignment
    print("  Re-decoding corpus with updated assignment...")
    updated_decoded = []
    for token in all_tokens:
        result = decode_token_cvc_v2(
            token, updated_assignment, eva_to_triple, coda_table)
        updated_decoded.append(result.decoded_cvc)

    updated_dict_count = sum(1 for d in updated_decoded
                             if d and d.lower() in ref_word_set)
    updated_dict_hit = updated_dict_count / len(all_tokens) if all_tokens else 0.0
    dict_improvement = updated_dict_hit - baseline_dict_hit

    print(f"  Updated dict-hit: {updated_dict_hit:.1%} "
          f"(delta={dict_improvement:+.1%})")

    # Compute new clean fraction: confirmed + RESOLVED/LIKELY treated as confirmed
    expanded_confirmed_keys = confirmed_keys | promoted_keys
    new_clean_fraction = _compute_clean_fraction(
        all_tokens, eva_to_triple, expanded_confirmed_keys, coda_table)
    print(f"  New clean fraction: {new_clean_fraction:.1%} "
          f"(was {old_clean_fraction:.1%})")

    # Signal count: count words that appear in ref_word_set for baseline and updated
    baseline_signal_words = set(d for d in baseline_decoded
                                if d and d.lower() in ref_word_set)
    updated_signal_words = set(d for d in updated_decoded
                               if d and d.lower() in ref_word_set)
    baseline_signal_count = len(baseline_signal_words)
    signal_count = len(updated_signal_words)
    print(f"  Signal word types: {baseline_signal_count} -> {signal_count}")

    # ===================================================================
    # Gates
    # ===================================================================
    w1 = loo_recovery_rate > 0.50
    w2 = n_resolved >= 3
    w3 = (n_resolved + n_likely) >= 5
    w4 = new_clean_fraction > 0.70
    w5 = dict_improvement > 0.01
    w6 = signal_count >= baseline_signal_count

    gates_passed = sum([w1, w2, w3, w4, w5, w6])

    if w1 and w2 and w4:
        verdict = 'CONSTRAINTS_RESOLVED'
    elif w1 and (w2 or w3):
        verdict = 'CONSTRAINTS_FOUND'
    elif any([w1, w2, w3, w4, w5, w6]):
        verdict = 'CONSTRAINTS_WEAK'
    else:
        verdict = 'NO_CONSTRAINTS'

    print(f"\n  Gates:")
    print(f"    W1 LOO recovery > 50%:            {w1} ({loo_recovery_rate:.1%})")
    print(f"    W2 n_resolved >= 3:               {w2} ({n_resolved})")
    print(f"    W3 resolved+likely >= 5:          {w3} ({n_resolved + n_likely})")
    print(f"    W4 clean_fraction > 70%:          {w4} ({new_clean_fraction:.1%})")
    print(f"    W5 dict improvement > 1pp:        {w5} ({dict_improvement:+.1%})")
    print(f"    W6 signal maintained/increased:   {w6} ({baseline_signal_count}->{signal_count})")
    print(f"  Gates passed: {gates_passed}/6")
    print(f"  Verdict: {verdict}")

    # ===================================================================
    # Build and save result
    # ===================================================================
    result = WildcardPropResult(
        n_identifications=n_identifications,
        n_constrained_triples=n_constrained,
        mean_consistency=round(mean_consistency, 4),
        per_triple_resolution=per_triple_resolution,
        n_resolved=n_resolved,
        n_likely=n_likely,
        n_tentative=n_tentative,
        n_insufficient=n_insufficient,
        loo_results=loo_results,
        loo_recovery_rate=round(loo_recovery_rate, 4),
        loo_validated=loo_validated,
        updated_dict_hit=round(updated_dict_hit, 4),
        baseline_dict_hit=round(baseline_dict_hit, 4),
        dict_improvement=round(dict_improvement, 4),
        old_clean_fraction=round(old_clean_fraction, 4),
        new_clean_fraction=round(new_clean_fraction, 4),
        signal_count=signal_count,
        baseline_signal_count=baseline_signal_count,
        w1_loo_recovery=w1,
        w2_n_resolved=w2,
        w3_resolved_plus_likely=w3,
        w4_clean_fraction=w4,
        w5_dict_improvement=w5,
        w6_signal_maintained=w6,
        gates_passed=gates_passed,
        gate_passed=gates_passed >= 3,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 1),
    )

    path = _save_json(rd, 'p76_wildcard_prop.json', result)
    print(f"\n  Saved: {path}")
    print(f"  Runtime: {result.runtime_seconds:.1f}s")
    return result
