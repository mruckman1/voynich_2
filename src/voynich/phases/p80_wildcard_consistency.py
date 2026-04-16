"""
Phase 80: Wildcard Consistency Check (Reviewer 3.7)
====================================================
For each word-level identification that contains wildcards (unresolved
triples), extract the implied phonetic values from the matched Latin word.
Check whether these implied values are mutually consistent across different
identifications.

The reviewer's concern: if character 't' gets one value in 'otol' -> ratione
and a different value in 'oty' -> rabidi, the substitution premise is violated.

Dependency chain:
    results/p75_t1.json               (Phase 75 Track 3 identifications)
    results/combined_refine.json      (Phase 15 assignment table)
    results/triple_tiers.json         (Phase 28/53 tier classifications)
        -> results/p80_wildcard_consistency.json
"""

import json
import os
import random
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    load_corpus,
    tokenize_eva_chars,
)
from voynich.phases.coda_markers import get_coda
from voynich.phases.corrected_coda import (
    classify_token_chars_v2,
)
from voynich.phases.p68_expanded_t1 import (
    _decode_with_provenance,
    _get_confirmed_and_unresolved,
    ProvenanceChar,
)
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
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TripleConsistencyEntry:
    """Consistency report for one triple key."""
    triple_key: str
    current_assignment: str        # what T_P15 assigns
    tier: str                      # CONFIRMED / LANDSCAPE_CONFIRMED / AMBIGUOUS
    n_observations: int            # total implied-value observations
    n_source_identifications: int  # number of distinct identifications contributing
    implied_values: Dict[str, int] # {implied_syllable: count}
    top_value: str                 # most common implied syllable
    consistency: float             # fraction agreeing with top_value
    agrees_with_assignment: bool   # does top_value == current_assignment?
    source_words: List[str]        # which matched words contributed


@dataclass
class ConflictEntry:
    """A specific cross-identification conflict."""
    triple_key: str
    token_a: str
    matched_word_a: str
    implied_value_a: str
    token_b: str
    matched_word_b: str
    implied_value_b: str


@dataclass
class WildcardConsistencyResult:
    phase: str = "80"
    experiment: str = "wildcard_consistency"
    # Input stats
    n_total_identifications: int = 0
    n_with_wildcards: int = 0
    n_without_wildcards: int = 0
    n_triples_observed: int = 0
    # Per-triple consistency
    per_triple: List[TripleConsistencyEntry] = field(default_factory=list)
    mean_consistency: float = 0.0
    n_consistent_80pct: int = 0
    n_agrees_with_assignment: int = 0
    # Conflicts
    conflicts: List[ConflictEntry] = field(default_factory=list)
    n_conflicts: int = 0
    # Null test
    null_mean_consistency: float = 0.0
    null_std_consistency: float = 0.0
    null_p_value: float = 1.0
    n_null_permutations: int = 0
    # Verdict
    verdict: str = ""
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def _extract_implied_values(
    identifications: List[Dict],
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    coda_table,
    confirmed_keys: Set[str],
) -> Dict[str, List[Tuple[str, str, str]]]:
    """Extract implied syllable values for each unresolved triple.

    Returns: {triple_key: [(implied_syllable, token, matched_word), ...]}

    For each identification with wildcards, we decode the token with
    provenance tracking, then align the LOW-confidence positions against
    the matched word to read off what the unresolved triple should encode.
    """
    triple_implied: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)

    for ident in identifications:
        token = ident['token']
        matched_word = ident['matched_word']
        n_unresolved = ident.get('n_unresolved', 0)

        if n_unresolved == 0:
            continue

        # Decode with provenance to find LOW positions
        provenance = _decode_with_provenance(
            token, assignment, eva_to_triple, coda_table, confirmed_keys)

        if not provenance or len(provenance) != len(matched_word):
            continue

        # Group provenance chars by triple_key to build implied syllable
        # A triple produces 2 chars (CV). We need to reconstruct the
        # complete implied syllable for each unresolved triple occurrence.

        # First pass: identify runs of LOW chars from the same triple
        runs: List[Tuple[str, int, int]] = []  # (triple_key, start, end)
        i = 0
        while i < len(provenance):
            pc = provenance[i]
            if pc.confidence == 'LOW':
                triple_key = pc.triple_key
                start = i
                end = i + 1
                while end < len(provenance) and provenance[end].confidence == 'LOW' and provenance[end].triple_key == triple_key:
                    end += 1
                runs.append((triple_key, start, end))
                i = end
            else:
                i += 1

        # For each run, extract the implied syllable from matched_word
        for triple_key, start, end in runs:
            implied_syllable = matched_word[start:end]
            triple_implied[triple_key].append(
                (implied_syllable, token, matched_word))

    return dict(triple_implied)


def _compute_consistency(
    triple_implied: Dict[str, List[Tuple[str, str, str]]],
    assignment: Dict[str, str],
    tier_map: Dict[str, str],
) -> List[TripleConsistencyEntry]:
    """Compute consistency metrics for each triple."""
    entries = []
    for triple_key in sorted(triple_implied.keys()):
        observations = triple_implied[triple_key]
        syllable_counts: Counter = Counter()
        source_words = []
        for implied_syl, token, matched_word in observations:
            syllable_counts[implied_syl] += 1
            source_words.append(matched_word)

        n_obs = sum(syllable_counts.values())
        top_value = syllable_counts.most_common(1)[0][0] if syllable_counts else ''
        top_count = syllable_counts[top_value] if top_value else 0
        consistency = top_count / n_obs if n_obs > 0 else 0.0

        current = assignment.get(triple_key, '?')
        tier = tier_map.get(triple_key, 'UNKNOWN')

        entries.append(TripleConsistencyEntry(
            triple_key=triple_key,
            current_assignment=current,
            tier=tier,
            n_observations=n_obs,
            n_source_identifications=len(set(
                (t, w) for _, t, w in observations)),
            implied_values=dict(syllable_counts),
            top_value=top_value,
            consistency=round(consistency, 4),
            agrees_with_assignment=(top_value == current),
            source_words=sorted(set(source_words)),
        ))
    return entries


def _find_conflicts(
    triple_implied: Dict[str, List[Tuple[str, str, str]]],
) -> List[ConflictEntry]:
    """Find specific cross-identification conflicts."""
    conflicts = []
    for triple_key, observations in triple_implied.items():
        # Find pairs with different implied values
        seen: Dict[str, Tuple[str, str]] = {}  # implied_syl -> (token, word)
        for implied_syl, token, matched_word in observations:
            if implied_syl in seen:
                continue
            for other_syl, other_tok, other_word in observations:
                if other_syl != implied_syl and other_tok != token:
                    conflicts.append(ConflictEntry(
                        triple_key=triple_key,
                        token_a=token,
                        matched_word_a=matched_word,
                        implied_value_a=implied_syl,
                        token_b=other_tok,
                        matched_word_b=other_word,
                        implied_value_b=other_syl,
                    ))
                    seen[implied_syl] = (token, matched_word)
                    seen[other_syl] = (other_tok, other_word)
                    break
    return conflicts


def _null_test(
    identifications: List[Dict],
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    coda_table,
    confirmed_keys: Set[str],
    tier_map: Dict[str, str],
    n_perms: int = 500,
) -> Tuple[float, float, float]:
    """Null test: shuffle matched_word column and recompute consistency."""
    # Only use identifications with wildcards
    wildcard_ids = [i for i in identifications if i.get('n_unresolved', 0) > 0]
    if len(wildcard_ids) < 3:
        return 0.0, 0.0, 1.0

    null_consistencies = []
    rng = random.Random(42)

    for _ in range(n_perms):
        # Shuffle the matched_word column
        shuffled = []
        words = [i['matched_word'] for i in wildcard_ids]
        rng.shuffle(words)
        for orig, new_word in zip(wildcard_ids, words):
            shuffled.append({
                'token': orig['token'],
                'matched_word': new_word,
                'n_unresolved': orig['n_unresolved'],
            })

        implied = _extract_implied_values(
            shuffled, assignment, eva_to_triple, coda_table, confirmed_keys)
        if not implied:
            null_consistencies.append(0.0)
            continue

        entries = _compute_consistency(implied, assignment, tier_map)
        if entries:
            mean_c = sum(e.consistency for e in entries) / len(entries)
            null_consistencies.append(mean_c)
        else:
            null_consistencies.append(0.0)

    if not null_consistencies:
        return 0.0, 0.0, 1.0

    null_mean = sum(null_consistencies) / len(null_consistencies)
    null_std = (sum((x - null_mean)**2 for x in null_consistencies) / len(null_consistencies)) ** 0.5

    return null_mean, null_std, 0.0  # p-value computed in caller


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_wildcard_consistency():
    """Phase 80: Check consistency of implied values across word-level identifications."""
    t0 = time.time()
    rd = _results_dir()
    print("Phase 80: Wildcard Consistency Check")
    print("=" * 60)

    # Load resources
    assignment_data = _safe_load(os.path.join(rd, 'combined_refine.json'))
    assignment = assignment_data.get('best_assignment', {})
    eva_to_triple = build_eva_to_triple_lookup()
    coda_table = _build_3coda_table()
    confirmed, unresolved = _get_confirmed_and_unresolved(rd)
    confirmed_keys = set(confirmed.keys())

    # Load tier classifications
    tier_data = _safe_load(os.path.join(rd, 'triple_tiers.json'))
    tier_map: Dict[str, str] = {}
    if tier_data and 'tiers' in tier_data:
        tiers = tier_data['tiers']
        if isinstance(tiers, dict):
            for tier_name, entries in tiers.items():
                for entry in entries:
                    tier_map[entry.get('triple_key', '')] = tier_name

    # Load identifications
    t1_data = _safe_load(os.path.join(rd, 'p75_t1.json'))
    identifications = t1_data.get('identifications', [])

    n_with = sum(1 for i in identifications if i.get('n_unresolved', 0) > 0)
    n_without = len(identifications) - n_with
    print(f"  Total identifications: {len(identifications)}")
    print(f"  With wildcards: {n_with}")
    print(f"  Without wildcards (fully decoded): {n_without}")

    # Extract implied values
    triple_implied = _extract_implied_values(
        identifications, assignment, eva_to_triple, coda_table, confirmed_keys)

    print(f"  Triples with implied values: {len(triple_implied)}")

    # Compute consistency
    per_triple = _compute_consistency(triple_implied, assignment, tier_map)

    # Print per-triple results
    print("\n--- Per-triple consistency ---")
    for entry in per_triple:
        status = "AGREES" if entry.agrees_with_assignment else "DIFFERS"
        print(f"  {entry.triple_key}")
        print(f"    Current: {entry.current_assignment} | Top implied: {entry.top_value} | "
              f"Consistency: {entry.consistency:.1%} | {status}")
        print(f"    Values: {entry.implied_values}")
        print(f"    Sources: {entry.source_words}")

    # Find conflicts
    conflicts = _find_conflicts(triple_implied)
    print(f"\n  Cross-identification conflicts: {len(conflicts)}")
    for c in conflicts[:10]:
        print(f"    {c.triple_key}: {c.token_a}->{c.matched_word_a} implies '{c.implied_value_a}' "
              f"but {c.token_b}->{c.matched_word_b} implies '{c.implied_value_b}'")

    # Global metrics
    if per_triple:
        mean_consistency = sum(e.consistency for e in per_triple) / len(per_triple)
        n_consistent_80 = sum(1 for e in per_triple if e.consistency >= 0.80)
        n_agrees = sum(1 for e in per_triple if e.agrees_with_assignment)
    else:
        mean_consistency = 0.0
        n_consistent_80 = 0
        n_agrees = 0

    print(f"\n  Mean consistency: {mean_consistency:.1%}")
    print(f"  Triples >=80% consistent: {n_consistent_80}/{len(per_triple)}")
    print(f"  Agree with T_P15 assignment: {n_agrees}/{len(per_triple)}")

    # Null test
    print("\n--- Null test (500 permutations) ---")
    null_mean, null_std, _ = _null_test(
        identifications, assignment, eva_to_triple, coda_table,
        confirmed_keys, tier_map, n_perms=500)

    if null_std > 0:
        z = (mean_consistency - null_mean) / null_std
        p_value = max(0.001, 1.0 - _normal_cdf(z))
    else:
        z = 0.0
        p_value = 1.0

    print(f"  Real mean consistency: {mean_consistency:.3f}")
    print(f"  Null mean: {null_mean:.3f} +/- {null_std:.3f}")
    print(f"  z = {z:.2f}, p = {p_value:.4f}")

    # Also check: are the 301 fully-decoded identifications consistent?
    # (Every char is from a confirmed triple, so the decode IS the assignment)
    print(f"\n--- Fully-decoded identifications (no wildcards) ---")
    print(f"  {n_without} identifications are fully decoded from confirmed triples.")
    print(f"  These are internally consistent by construction (each triple has")
    print(f"  exactly one assigned value used across all identifications).")

    # Verdict
    if n_with == 0:
        verdict = "ALL_CONSISTENT_NO_WILDCARDS"
    elif mean_consistency >= 0.80 and p_value < 0.05:
        verdict = "CONSISTENT_SIGNIFICANT"
    elif mean_consistency >= 0.80:
        verdict = "CONSISTENT_NOT_SIGNIFICANT"
    elif len(conflicts) == 0:
        verdict = "NO_CONFLICTS"
    else:
        verdict = "INCONSISTENCIES_FOUND"

    print(f"\n  Verdict: {verdict}")

    # Build result
    result = WildcardConsistencyResult(
        n_total_identifications=len(identifications),
        n_with_wildcards=n_with,
        n_without_wildcards=n_without,
        n_triples_observed=len(per_triple),
        per_triple=per_triple,
        mean_consistency=round(mean_consistency, 4),
        n_consistent_80pct=n_consistent_80,
        n_agrees_with_assignment=n_agrees,
        conflicts=conflicts,
        n_conflicts=len(conflicts),
        null_mean_consistency=round(null_mean, 4),
        null_std_consistency=round(null_std, 4),
        null_p_value=round(p_value, 4),
        n_null_permutations=500,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    path = _save_json(rd, 'p80_wildcard_consistency.json', result)
    print(f"\n  Saved -> {path}")
    print(f"  Runtime: {result.runtime_seconds:.1f}s")
    return result


def _normal_cdf(z: float) -> float:
    """Approximate standard normal CDF."""
    import math
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
