"""
Phase 68, Track 3: Paradigmatic Analysis (Minimal Pairs)
=========================================================
Find EVA token types that differ by exactly one EVA character.
When one differing character maps to a confirmed triple and the other
to an unresolved triple, dictionary lookup constrains the unresolved
triple's value.

Dependency chain:
    results/combined_refine.json      (Phase 15)
    results/triple_tiers.json         (Phase 28/53)
        -> results/p68_paradigmatic.json
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
from voynich.phases.corrected_coda import (
    build_coda_table_v2,
    decode_token_cvc_v2,
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
# Confirmed / unresolved triple separation
# ---------------------------------------------------------------------------

def _get_confirmed_and_unresolved(
    rd: str,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Return (confirmed_12, unresolved_13)."""
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

    if not confirmed_keys:
        return dict(assignment), {}

    confirmed = {k: v for k, v in assignment.items() if k in confirmed_keys}
    unresolved = {k: v for k, v in assignment.items() if k not in confirmed_keys}
    return confirmed, unresolved


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class ParadigmaticResult:
    phase: str = "68"
    step: str = "68.3"
    experiment: str = "paradigmatic_analysis"
    n_token_types: int = 0
    n_minimal_pairs: int = 0
    n_diagnostic_pairs: int = 0  # one confirmed, one unresolved
    n_triples_constrained: int = 0
    # Per-triple results
    triple_candidates: Dict[str, str] = field(default_factory=dict)
    triple_details: List[Dict[str, Any]] = field(default_factory=list)
    # Sample pairs
    sample_pairs: List[Dict[str, Any]] = field(default_factory=list)
    # Gates
    g1_pairs: bool = False          # MP1: >= 200 minimal pairs
    g2_diagnostic: bool = False     # MP2: >= 50 diagnostic pairs
    g3_triples: bool = False        # MP3: >= 5 triples constrained
    g4_narrow: bool = False         # MP4: >= 2 triples with < 10 candidates
    gates_passed: int = 0
    gate_passed: bool = False
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def _find_minimal_pairs(
    token_types: Dict[str, int],
    min_freq: int = 5,
) -> List[Dict[str, Any]]:
    """Find token type pairs differing at exactly one EVA character position.

    Only considers types appearing >= min_freq times.
    Pre-filters by length for O(n^2)-per-length-bucket performance.
    """
    # Filter by frequency
    frequent = {t: f for t, f in token_types.items() if f >= min_freq}

    # Group by EVA-char length
    by_length: Dict[int, List[Tuple[str, List[str], int]]] = {}
    for token, freq in frequent.items():
        chars = tokenize_eva_chars(token)
        if not chars:
            continue
        length = len(chars)
        if length not in by_length:
            by_length[length] = []
        by_length[length].append((token, chars, freq))

    pairs: List[Dict[str, Any]] = []

    for length, group in sorted(by_length.items()):
        n = len(group)
        for i in range(n):
            for j in range(i + 1, n):
                tok_a, chars_a, freq_a = group[i]
                tok_b, chars_b, freq_b = group[j]

                # Count differing positions
                diff_pos = -1
                n_diffs = 0
                for k in range(length):
                    if chars_a[k] != chars_b[k]:
                        n_diffs += 1
                        diff_pos = k
                        if n_diffs > 1:
                            break

                if n_diffs == 1:
                    pairs.append({
                        'type_a': tok_a,
                        'type_b': tok_b,
                        'diff_position': diff_pos,
                        'char_a': chars_a[diff_pos],
                        'char_b': chars_b[diff_pos],
                        'freq_a': freq_a,
                        'freq_b': freq_b,
                    })

    return pairs


def _classify_pairs(
    pairs: List[Dict[str, Any]],
    eva_to_triple: Dict[str, str],
    confirmed_keys: Set[str],
) -> List[Dict[str, Any]]:
    """Keep pairs where exactly one differing char is confirmed, one unresolved."""
    diagnostic: List[Dict[str, Any]] = []

    for pair in pairs:
        triple_a = eva_to_triple.get(pair['char_a'], '')
        triple_b = eva_to_triple.get(pair['char_b'], '')

        if not triple_a or not triple_b:
            continue

        a_confirmed = triple_a in confirmed_keys
        b_confirmed = triple_b in confirmed_keys

        # Exactly one confirmed, one unresolved
        if a_confirmed == b_confirmed:
            continue

        if a_confirmed:
            confirmed_side = 'a'
            confirmed_char = pair['char_a']
            confirmed_triple = triple_a
            unresolved_char = pair['char_b']
            unresolved_triple = triple_b
        else:
            confirmed_side = 'b'
            confirmed_char = pair['char_b']
            confirmed_triple = triple_b
            unresolved_char = pair['char_a']
            unresolved_triple = triple_a

        diagnostic.append({
            **pair,
            'triple_a': triple_a,
            'triple_b': triple_b,
            'confirmed_side': confirmed_side,
            'confirmed_char': confirmed_char,
            'confirmed_triple': confirmed_triple,
            'unresolved_char': unresolved_char,
            'unresolved_triple': unresolved_triple,
        })

    return diagnostic


def _constrain_from_dictionary(
    diagnostic_pairs: List[Dict[str, Any]],
    full_assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    coda_table: Any,
    ref_word_set: Set[str],
    candidate_syllables: List[str],
) -> Dict[str, List[Dict[str, Any]]]:
    """For each diagnostic pair, decode both sides with candidate syllables.

    Returns per-unresolved-triple list of constraint observations.
    """
    constraints: Dict[str, List[Dict[str, Any]]] = {}

    for pair in diagnostic_pairs:
        # Decode the confirmed-side token
        confirmed_token = pair['type_a'] if pair['confirmed_side'] == 'a' else pair['type_b']
        confirmed_result = decode_token_cvc_v2(
            confirmed_token, full_assignment, eva_to_triple, coda_table)
        confirmed_decoded = confirmed_result.decoded_cvc if confirmed_result.decoded_cvc else ''

        # Skip if confirmed side is not a dict hit (unreliable anchor)
        if not confirmed_decoded or '?' in confirmed_decoded or confirmed_decoded not in ref_word_set:
            continue

        # Decode the unresolved-side token with each candidate syllable
        unresolved_token = pair['type_b'] if pair['confirmed_side'] == 'a' else pair['type_a']
        unresolved_triple = pair['unresolved_triple']

        if unresolved_triple not in constraints:
            constraints[unresolved_triple] = []

        hits: List[str] = []
        for candidate in candidate_syllables:
            test_assignment = dict(full_assignment)
            test_assignment[unresolved_triple] = candidate

            result = decode_token_cvc_v2(
                unresolved_token, test_assignment, eva_to_triple, coda_table)
            decoded = result.decoded_cvc if result.decoded_cvc else ''

            if decoded and '?' not in decoded and decoded in ref_word_set:
                hits.append(candidate)

        constraints[unresolved_triple].append({
            'confirmed_token': confirmed_token,
            'confirmed_decoded': confirmed_decoded,
            'unresolved_token': unresolved_token,
            'unresolved_char': pair['unresolved_char'],
            'candidate_hits': hits,
            'n_candidate_hits': len(hits),
        })

    return constraints


def _aggregate_votes(
    constraints: Dict[str, List[Dict[str, Any]]],
) -> Tuple[Dict[str, str], List[Dict[str, Any]]]:
    """Per unresolved triple, count how often each candidate syllable was implied.

    Returns (best_candidates, detail_list).
    """
    best_candidates: Dict[str, str] = {}
    details: List[Dict[str, Any]] = []

    for triple_key, observations in sorted(constraints.items()):
        vote_counter: Counter = Counter()
        for obs in observations:
            for syl in obs['candidate_hits']:
                vote_counter[syl] += 1

        if not vote_counter:
            details.append({
                'triple_key': triple_key,
                'n_observations': len(observations),
                'n_candidates': 0,
                'top_candidate': None,
                'top_votes': 0,
                'all_votes': {},
            })
            continue

        ranked = vote_counter.most_common()
        top_syl, top_votes = ranked[0]
        best_candidates[triple_key] = top_syl

        details.append({
            'triple_key': triple_key,
            'n_observations': len(observations),
            'n_candidates': len(vote_counter),
            'top_candidate': top_syl,
            'top_votes': top_votes,
            'all_votes': dict(vote_counter.most_common(20)),
        })

    return best_candidates, details


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_paradigmatic():
    """Track 3: Paradigmatic analysis (minimal pairs)."""
    t0 = time.time()
    rd = str(_results_dir())

    print("Phase 68.3 — Paradigmatic Analysis (Minimal Pairs)")
    print("=" * 50)

    # --- Load data ---
    confirmed, unresolved = _get_confirmed_and_unresolved(rd)
    confirmed_keys = set(confirmed.keys())
    full_assignment = {**confirmed, **unresolved}
    print(f"  Confirmed triples: {len(confirmed)}")
    print(f"  Unresolved triples: {len(unresolved)}")

    eva_to_triple = build_eva_to_triple_lookup()
    coda_table = build_coda_table_v2()

    corpus = load_corpus(verbose=False)
    all_tokens = corpus.get_tokens()
    print(f"  Corpus tokens: {len(all_tokens)}")

    # Build dictionary
    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    base_words = set(w.lower() for w in ref_corpus.get_combined_tokens('latin')
                     if len(w) >= 2)
    expanded, _ = build_expanded_word_set(base_words)
    ref_word_set = base_words | expanded
    print(f"  Dictionary size: {len(ref_word_set)}")

    # --- Step 1: Find minimal pairs ---
    print("\n  Finding minimal pairs...")
    type_freq: Counter = Counter(all_tokens)
    n_token_types = len(type_freq)
    print(f"  Token types: {n_token_types}")

    pairs = _find_minimal_pairs(type_freq, min_freq=5)
    n_minimal_pairs = len(pairs)
    print(f"  Minimal pairs found: {n_minimal_pairs}")

    # --- Step 2: Classify pairs ---
    print("\n  Classifying pairs (confirmed vs unresolved)...")
    diagnostic_pairs = _classify_pairs(pairs, eva_to_triple, confirmed_keys)
    n_diagnostic = len(diagnostic_pairs)
    print(f"  Diagnostic pairs: {n_diagnostic}")

    # Sample pairs for output
    sample_pairs = diagnostic_pairs[:50]

    # --- Step 3: Constrain from dictionary ---
    print("\n  Constraining unresolved triples from dictionary...")
    # Candidate syllables: confirmed values + standard Italian CV
    candidate_syllables = sorted(set(confirmed.values()) | set(unresolved.values()) |
                                  {'ba', 'be', 'bi', 'bo', 'bu',
                                   'ca', 'ce', 'ci', 'co', 'cu',
                                   'da', 'de', 'di', 'do', 'du',
                                   'fa', 'fe', 'fi', 'fo', 'fu',
                                   'la', 'le', 'li', 'lo', 'lu',
                                   'ma', 'me', 'mi', 'mo', 'mu',
                                   'na', 'ne', 'ni', 'no', 'nu',
                                   'pa', 'pe', 'pi', 'po', 'pu',
                                   'ra', 're', 'ri', 'ro', 'ru',
                                   'sa', 'se', 'si', 'so', 'su',
                                   'ta', 'te', 'ti', 'to', 'tu',
                                   'va', 've', 'vi', 'vo', 'vu'})

    constraints = _constrain_from_dictionary(
        diagnostic_pairs, full_assignment, eva_to_triple,
        coda_table, ref_word_set, candidate_syllables)

    # --- Step 4: Aggregate votes ---
    print("\n  Aggregating votes...")
    triple_candidates, triple_details = _aggregate_votes(constraints)
    n_triples_constrained = len(triple_candidates)
    print(f"  Triples constrained: {n_triples_constrained}")

    # Count narrowly constrained (< 10 candidates)
    narrow_triples = [d for d in triple_details if 0 < d['n_candidates'] < 10]
    n_narrow = len(narrow_triples)
    print(f"  Narrowly constrained (< 10 candidates): {n_narrow}")

    for detail in triple_details:
        if detail['top_candidate']:
            print(f"    {detail['triple_key']}: {detail['top_candidate']} "
                  f"({detail['top_votes']} votes, {detail['n_candidates']} candidates)")

    # --- Gates ---
    g1 = n_minimal_pairs >= 200
    g2 = n_diagnostic >= 50
    g3 = n_triples_constrained >= 5
    g4 = n_narrow >= 2
    gates_passed = sum([g1, g2, g3, g4])

    result = ParadigmaticResult(
        n_token_types=n_token_types,
        n_minimal_pairs=n_minimal_pairs,
        n_diagnostic_pairs=n_diagnostic,
        n_triples_constrained=n_triples_constrained,
        triple_candidates=triple_candidates,
        triple_details=triple_details,
        sample_pairs=sample_pairs,
        g1_pairs=g1,
        g2_diagnostic=g2,
        g3_triples=g3,
        g4_narrow=g4,
        gates_passed=gates_passed,
        gate_passed=gates_passed >= 2,
        runtime_seconds=round(time.time() - t0, 1),
    )

    path = _save_json(rd, 'p68_paradigmatic.json', result)

    # --- Summary ---
    print(f"\n  Summary")
    print(f"  -------")
    print(f"  Token types:      {n_token_types}")
    print(f"  Minimal pairs:    {n_minimal_pairs} ({'PASS' if g1 else 'FAIL'} >= 200)")
    print(f"  Diagnostic pairs: {n_diagnostic} ({'PASS' if g2 else 'FAIL'} >= 50)")
    print(f"  Triples constr.:  {n_triples_constrained} ({'PASS' if g3 else 'FAIL'} >= 5)")
    print(f"  Narrow (< 10):    {n_narrow} ({'PASS' if g4 else 'FAIL'} >= 2)")
    print(f"  Candidates:       {len(triple_candidates)}")
    print(f"  Gates: {gates_passed}/4")
    print(f"  Saved: {path}")
    print(f"  Time: {result.runtime_seconds:.1f}s")
