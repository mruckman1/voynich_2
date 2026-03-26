"""
Phase 68, Track 6: Distributional Constraint Propagation
==========================================================
Reuse Phase 67's distributional anchors (PPMI+SVD+Procrustes).
For EVA tokens with strong Latin distributional matches, compare
current decode to the matched Latin word.  Differences at positions
from unresolved triples constrain those triples.

Dependency chain:
    results/p67_distributional.json   (Phase 67, Track 5)
    results/combined_refine.json      (Phase 15)
    results/triple_tiers.json         (Phase 28/53)
        -> results/p68_distributional.json
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
from voynich.phases.coda_markers import CodaTable, get_coda
from voynich.phases.corrected_coda import (
    build_coda_table_v2,
    classify_token_chars_v2,
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
class DistribConstraintResult:
    phase: str = "68"
    step: str = "68.6"
    experiment: str = "distributional_constraints"
    n_p67_anchors: int = 0
    n_p67_matches_used: int = 0
    n_constraints_derived: int = 0
    n_triples_with_data: int = 0
    n_triples_with_clear_top: int = 0
    # Triple candidates
    triple_candidates: Dict[str, str] = field(default_factory=dict)
    triple_details: List[Dict[str, Any]] = field(default_factory=list)
    # Sample constraints
    sample_constraints: List[Dict[str, Any]] = field(default_factory=list)
    # Gates
    g1_coverage: bool = False       # DC1: >= 8 triples receive constraints
    g2_clear: bool = False          # DC2: >= 3 triples with clear top (2x ratio)
    gates_passed: int = 0
    gate_passed: bool = False
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Provenance-aware decode
# ---------------------------------------------------------------------------

@dataclass
class _ProvenanceChar:
    """One decoded character with provenance information."""
    char: str
    confidence: str    # HIGH, LOW, CODA
    triple_key: str    # which triple produced this (empty for CODA)
    position: int      # position in decoded string


def _decode_with_provenance(
    token: str,
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    coda_table: CodaTable,
    confirmed_keys: Set[str],
) -> List[_ProvenanceChar]:
    """Decode a token and track which triple produced each character.

    Similar to decode_token_cvc_v2 but returns per-character provenance.
    """
    eva_chars = tokenize_eva_chars(token)
    if not eva_chars:
        return []

    classified = classify_token_chars_v2(eva_chars, coda_table)

    result: List[_ProvenanceChar] = []
    pos = 0

    for role, char in classified:
        if role == 'SYLLABIC':
            triple_key = eva_to_triple.get(char, '')
            syllable = assignment.get(triple_key, '?') if triple_key else '?'
            confidence = 'HIGH' if triple_key in confirmed_keys else 'LOW'

            for c in syllable:
                result.append(_ProvenanceChar(
                    char=c,
                    confidence=confidence,
                    triple_key=triple_key,
                    position=pos,
                ))
                pos += 1

        elif role == 'CODA_MARKER':
            coda = get_coda(char, coda_table)
            if coda:
                result.append(_ProvenanceChar(
                    char=coda,
                    confidence='CODA',
                    triple_key='',
                    position=pos,
                ))
                pos += 1

    return result


# ---------------------------------------------------------------------------
# Load Phase 67 distributional data
# ---------------------------------------------------------------------------

def _load_p67_distributional(rd: str) -> Dict:
    """Load p67_distributional.json.

    Extract triple_candidates, anchor_pairs, nearest_matches, and
    n_anchor_pairs from the Phase 67 distributional output.
    Return the loaded dict (empty dict if file missing).
    """
    data = _safe_load(os.path.join(rd, 'p67_distributional.json'))
    return data


# ---------------------------------------------------------------------------
# Build constraints from distributional matches
# ---------------------------------------------------------------------------

def _build_constraints(
    p67_data: Dict,
    all_tokens: List[str],
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    coda_table: CodaTable,
    confirmed_keys: Set[str],
) -> Tuple[Dict[str, Counter], int]:
    """Derive triple constraints from Phase 67 distributional matches.

    If p67 has per-EVA-token matches (sample_matches with top_5):
        - For each EVA token with a match, decode with provenance
        - Compare decoded string to the nearest Latin word char by char
        - Where they differ AND the position is from an unresolved triple
          (LOW confidence), record (triple_key, implied char from Latin word)

    If p67 only has triple_candidates (no per-token matches):
        - Use those directly as pass-through candidates

    Returns (constraints, n_matches_used) where constraints maps
    triple_key -> Counter(implied_char -> count).
    """
    constraints: Dict[str, Counter] = {}
    n_matches_used = 0

    # Strategy 1: Use sample_matches (per-EVA-token distributional matches)
    sample_matches = p67_data.get('sample_matches', [])
    if sample_matches:
        # Build a lookup: eva_token -> list of (latin_word, similarity)
        token_matches: Dict[str, List[Tuple[str, float]]] = {}
        for entry in sample_matches:
            eva_token = entry.get('eva_token', '')
            top_5 = entry.get('top_5', [])
            if not eva_token or not top_5:
                continue
            matches = []
            for match in top_5:
                if isinstance(match, (list, tuple)) and len(match) >= 2:
                    matches.append((str(match[0]), float(match[1])))
                elif isinstance(match, dict):
                    w = match.get('word', match.get('latin', ''))
                    s = match.get('similarity', match.get('score', 0.0))
                    if w:
                        matches.append((str(w), float(s)))
            if matches:
                token_matches[eva_token] = matches

        # Also try to reconstruct from all EVA token types in the corpus
        # that decode to something, using the sample_matches as anchors
        for eva_token, matches in token_matches.items():
            if not matches:
                continue

            # Decode the EVA token with provenance
            prov_chars = _decode_with_provenance(
                eva_token, assignment, eva_to_triple, coda_table, confirmed_keys)
            if not prov_chars:
                continue

            # Check if any positions are from unresolved triples
            has_low = any(pc.confidence == 'LOW' for pc in prov_chars)
            if not has_low:
                continue

            # Compare against top Latin match
            latin_word = matches[0][0]
            similarity = matches[0][1]

            # Only use matches with reasonable similarity
            if similarity < 0.1:
                continue

            n_matches_used += 1
            decoded_str = ''.join(pc.char for pc in prov_chars)

            # Align decoded string with Latin word character by character
            min_len = min(len(decoded_str), len(latin_word))
            for i in range(min_len):
                if i >= len(prov_chars):
                    break
                pc = prov_chars[i]
                if pc.confidence == 'LOW' and pc.triple_key:
                    # This position is from an unresolved triple
                    latin_char = latin_word[i]
                    if latin_char != pc.char:
                        # The distributional match suggests a different char
                        if pc.triple_key not in constraints:
                            constraints[pc.triple_key] = Counter()
                        constraints[pc.triple_key][latin_char] += 1

    # Strategy 2: If no per-token matches were found but p67 has
    # triple_candidates, use them as pass-through
    if not constraints and not sample_matches:
        p67_candidates = p67_data.get('triple_candidates', {})
        if p67_candidates:
            for triple_key, values in p67_candidates.items():
                if isinstance(values, list):
                    # triple_candidates is Dict[str, List[str]]
                    ctr = Counter()
                    for val in values:
                        ctr[val] += 1
                    constraints[triple_key] = ctr
                elif isinstance(values, str):
                    constraints[triple_key] = Counter({values: 1})

    return constraints, n_matches_used


# ---------------------------------------------------------------------------
# Aggregate votes into syllable candidates
# ---------------------------------------------------------------------------

def _aggregate_votes(
    constraints: Dict[str, Counter],
) -> Tuple[Dict[str, str], List[Dict[str, Any]], int]:
    """Per triple, pick top implied character if count ratio > 2.0.

    Since each constraint is a single character (not a syllable), we group
    by position within the triple's syllable to reconstruct a 2-char
    syllable when possible.

    Returns (triple_candidates, triple_details, n_clear_top).
    """
    triple_candidates: Dict[str, str] = {}
    triple_details: List[Dict[str, Any]] = []
    n_clear_top = 0

    for triple_key in sorted(constraints.keys()):
        ctr = constraints[triple_key]
        if not ctr:
            continue

        ranked = ctr.most_common()
        top_char, top_count = ranked[0]
        second_count = ranked[1][1] if len(ranked) > 1 else 0

        # Clear top: top count >= 2x second count, and top_count >= 2
        clear = (top_count >= 2 and
                 (second_count == 0 or top_count >= 2.0 * second_count))
        if clear:
            n_clear_top += 1

        # Build syllable: if top chars suggest a CV pair, combine them
        # Otherwise just use top character as the candidate
        # For single-char constraints, use the character directly
        best_syllable = top_char
        if len(top_char) == 1 and len(ranked) >= 2:
            # Check if the top two chars could form a consonant+vowel pair
            vowels = set('aeiou')
            consonants = set('bcdfglmnprstvxz')
            second_char = ranked[1][0]
            if (len(second_char) == 1 and
                    top_char in consonants and second_char in vowels):
                best_syllable = top_char + second_char
            elif (len(second_char) == 1 and
                  top_char in vowels and second_char in consonants):
                best_syllable = second_char + top_char

        triple_candidates[triple_key] = best_syllable

        triple_details.append({
            'triple_key': triple_key,
            'proposed_value': best_syllable,
            'n_observations': sum(ctr.values()),
            'top_candidates': [
                {'char': c, 'count': n} for c, n in ranked[:5]
            ],
            'clear_top': clear,
        })

    return triple_candidates, triple_details, n_clear_top


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_distrib_constrain():
    """Track 6: Distributional constraint propagation."""
    t0 = time.time()
    rd = str(_results_dir())

    print("Phase 68.6 — Distributional Constraint Propagation")
    print("=" * 52)

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

    # --- Load Phase 67 distributional results ---
    print("\n  [68.6] Loading Phase 67 distributional results...")
    p67_data = _load_p67_distributional(rd)

    if not p67_data:
        print("  WARNING: p67_distributional.json not found — producing empty result.")
        result = DistribConstraintResult(
            runtime_seconds=round(time.time() - t0, 1),
        )
        path = _save_json(rd, 'p68_distributional.json', result)
        print(f"  Saved: {path}")
        print(f"  Time: {result.runtime_seconds:.1f}s")
        return

    n_p67_anchors = p67_data.get('n_anchor_pairs', 0)
    print(f"  [68.6] Phase 67 anchors: {n_p67_anchors}")

    p67_sample = p67_data.get('sample_matches', [])
    p67_candidates = p67_data.get('triple_candidates', {})
    print(f"  [68.6] Phase 67 sample matches: {len(p67_sample)}")
    print(f"  [68.6] Phase 67 triple candidates: {len(p67_candidates)}")

    # --- Build constraints from distributional matches ---
    print("\n  [68.6] Building constraints from distributional matches...")
    constraints, n_matches_used = _build_constraints(
        p67_data, all_tokens, full_assignment, eva_to_triple,
        coda_table, confirmed_keys)

    n_constraints_derived = sum(sum(c.values()) for c in constraints.values())
    n_triples_with_data = len(constraints)
    print(f"  [68.6] Triples with constraint data: {n_triples_with_data}")
    print(f"  [68.6] Total constraint observations: {n_constraints_derived}")
    print(f"  [68.6] Matches used: {n_matches_used}")

    # --- Aggregate votes ---
    print("\n  [68.6] Aggregating votes per triple...")
    triple_candidates, triple_details, n_clear_top = _aggregate_votes(constraints)

    # Print per-triple results
    for detail in triple_details:
        tk = detail['triple_key']
        proposed = detail['proposed_value']
        current = unresolved.get(tk, confirmed.get(tk, '?'))
        n_obs = detail['n_observations']
        clear_tag = ' [CLEAR]' if detail['clear_top'] else ''
        changed = '*' if proposed != current else ' '
        print(f"    {changed} {tk}: {current} -> {proposed} "
              f"(obs={n_obs}){clear_tag}")

    # --- Build sample constraints for JSON ---
    sample_constraints: List[Dict[str, Any]] = []
    for tk, ctr in sorted(constraints.items())[:10]:
        sample_constraints.append({
            'triple_key': tk,
            'votes': {c: n for c, n in ctr.most_common(5)},
            'total_obs': sum(ctr.values()),
        })

    # --- Gates ---
    g1 = n_triples_with_data >= 8
    g2 = n_clear_top >= 3
    gates_passed = sum([g1, g2])

    result = DistribConstraintResult(
        n_p67_anchors=n_p67_anchors,
        n_p67_matches_used=n_matches_used,
        n_constraints_derived=n_constraints_derived,
        n_triples_with_data=n_triples_with_data,
        n_triples_with_clear_top=n_clear_top,
        triple_candidates=triple_candidates,
        triple_details=triple_details,
        sample_constraints=sample_constraints,
        g1_coverage=g1,
        g2_clear=g2,
        gates_passed=gates_passed,
        gate_passed=gates_passed >= 2,
        runtime_seconds=round(time.time() - t0, 1),
    )

    path = _save_json(rd, 'p68_distributional.json', result)

    # --- Summary ---
    print(f"\n  Summary")
    print(f"  -------")
    print(f"  Phase 67 anchors:     {n_p67_anchors}")
    print(f"  Matches used:         {n_matches_used}")
    print(f"  Constraints derived:  {n_constraints_derived}")
    print(f"  Triples w/ data:      {n_triples_with_data} "
          f"({'PASS' if g1 else 'FAIL'} >= 8)")
    print(f"  Clear top (2x ratio): {n_clear_top} "
          f"({'PASS' if g2 else 'FAIL'} >= 3)")
    print(f"  Candidates proposed:  {len(triple_candidates)}")
    print(f"  Gates: {gates_passed}/2")
    print(f"  Saved: {path}")
    print(f"  Time: {result.runtime_seconds:.1f}s")
