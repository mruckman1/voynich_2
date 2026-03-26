"""
Phase 68, Track 4: Expanded T1 Identification Pipeline
========================================================
Re-run the T1 word identification pipeline with CVC-enhanced decode.
More characters are known (coda markers resolved), so wildcard patterns
have fewer unknowns, producing more unique dictionary matches that
constrain unresolved triples.

Dependency chain:
    results/combined_refine.json      (Phase 15)
    results/triple_tiers.json         (Phase 28/53)
        -> results/p68_expanded_t1.json
"""

import json
import os
import re
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
from voynich.phases.coda_markers import CodaTable, get_coda
from voynich.phases.corrected_coda import (
    build_coda_table_v2,
    classify_token_chars_v2,
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
    """Return (confirmed_12, unresolved_13).  Only truly CONFIRMED triples."""
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
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ProvenanceChar:
    """One decoded character with provenance information."""
    char: str
    confidence: str    # HIGH, LOW, CODA
    triple_key: str    # which triple produced this (empty for CODA)
    position: int      # position in decoded string


@dataclass
class ExpandedT1Result:
    phase: str = "68"
    step: str = "68.4"
    experiment: str = "expanded_t1_pipeline"
    n_token_types: int = 0
    n_patterns_built: int = 0
    n_skipped_low_known: int = 0
    n_unique_matches: int = 0
    n_few_matches: int = 0
    n_no_matches: int = 0
    # Identifications
    identifications: List[Dict[str, Any]] = field(default_factory=list)
    n_identifications: int = 0
    # Triple constraints
    triple_candidates: Dict[str, str] = field(default_factory=dict)
    triple_details: List[Dict[str, Any]] = field(default_factory=list)
    n_triples_constrained: int = 0
    # Consistency
    mean_consistency: float = 0.0
    # Gates
    g1_ids: bool = False            # T1_1: >= 30 identifications
    g2_triples: bool = False        # T1_2: >= 5 triples receive evidence
    g3_consistency: bool = False    # T1_3: consistency > 60%
    gates_passed: int = 0
    gate_passed: bool = False
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Provenance-tracking decode
# ---------------------------------------------------------------------------

def _decode_with_provenance(
    token: str,
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    coda_table: CodaTable,
    confirmed_keys: Set[str],
) -> List[ProvenanceChar]:
    """Decode a token and track which triple produced each character.

    Similar to decode_token_cvc_v2 but returns per-character provenance.
    """
    eva_chars = tokenize_eva_chars(token)
    if not eva_chars:
        return []

    classified = classify_token_chars_v2(eva_chars, coda_table)

    result: List[ProvenanceChar] = []
    pos = 0

    for role, char in classified:
        if role == 'SYLLABIC':
            triple_key = eva_to_triple.get(char, '')
            syllable = assignment.get(triple_key, '?') if triple_key else '?'
            confidence = 'HIGH' if triple_key in confirmed_keys else 'LOW'

            for c in syllable:
                result.append(ProvenanceChar(
                    char=c,
                    confidence=confidence,
                    triple_key=triple_key,
                    position=pos,
                ))
                pos += 1

        elif role == 'CODA_MARKER':
            coda = get_coda(char, coda_table)
            if coda:
                result.append(ProvenanceChar(
                    char=coda,
                    confidence='CODA',
                    triple_key='',
                    position=pos,
                ))
                pos += 1

    return result


# ---------------------------------------------------------------------------
# Dict indexing by length
# ---------------------------------------------------------------------------

def _build_dict_by_length(ref_word_set: Set[str]) -> Dict[int, List[str]]:
    """Pre-index dictionary by word length for fast lookup."""
    by_len: Dict[int, List[str]] = {}
    for word in ref_word_set:
        wl = len(word)
        if wl not in by_len:
            by_len[wl] = []
        by_len[wl].append(word)
    return by_len


# ---------------------------------------------------------------------------
# Pattern building
# ---------------------------------------------------------------------------

def _build_patterns(
    token_types: List[str],
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    coda_table: CodaTable,
    confirmed_keys: Set[str],
    min_known_frac: float = 0.50,
) -> List[Tuple[str, str, int, List[Tuple[int, str]]]]:
    """Build regex patterns for each unique token type.

    Returns list of (token_type, regex_pattern, target_len, unresolved_positions)
    where unresolved_positions = [(decoded_position, triple_key), ...] for LOW chars.
    Skips tokens where known_fraction < min_known_frac.
    """
    patterns: List[Tuple[str, str, int, List[Tuple[int, str]]]] = []

    for token in token_types:
        provenance = _decode_with_provenance(
            token, assignment, eva_to_triple, coda_table, confirmed_keys)
        if not provenance:
            continue

        # Build pattern and track unresolved positions
        pattern_chars = []
        n_known = 0
        unresolved_positions: List[Tuple[int, str]] = []

        for pc in provenance:
            if pc.confidence in ('HIGH', 'CODA'):
                pattern_chars.append(re.escape(pc.char))
                n_known += 1
            else:
                pattern_chars.append('[a-z]')
                unresolved_positions.append((pc.position, pc.triple_key))

        known_frac = n_known / len(provenance) if provenance else 0.0
        if known_frac < min_known_frac:
            continue

        regex = '^' + ''.join(pattern_chars) + '$'
        target_len = len(provenance)
        patterns.append((token, regex, target_len, unresolved_positions))

    return patterns


# ---------------------------------------------------------------------------
# Pattern matching
# ---------------------------------------------------------------------------

def _match_patterns(
    patterns: List[Tuple[str, str, int, List[Tuple[int, str]]]],
    dict_by_length: Dict[int, List[str]],
    max_matches: int = 20,
) -> List[List[str]]:
    """Match each pattern against dictionary words of matching length.

    Returns list of match lists, parallel to patterns.
    """
    all_matches: List[List[str]] = []

    for token, regex, target_len, unresolved_positions in patterns:
        words = dict_by_length.get(target_len, [])
        if not words:
            all_matches.append([])
            continue

        compiled = re.compile(regex)
        matches = []
        for word in words:
            if compiled.match(word):
                matches.append(word)
                if len(matches) >= max_matches:
                    break
        all_matches.append(matches)

    return all_matches


# ---------------------------------------------------------------------------
# Constraint extraction
# ---------------------------------------------------------------------------

def _extract_constraints(
    patterns: List[Tuple[str, str, int, List[Tuple[int, str]]]],
    all_matches: List[List[str]],
    corpus,
    all_tokens: List[str],
    min_folios: int = 3,
) -> Tuple[List[Dict[str, Any]], Dict[str, List[Tuple[str, int]]]]:
    """Extract identifications and triple constraints from unique matches.

    For patterns with exactly 1 match:
      - Check folio distribution (require >= min_folios folios)
      - Record as identification
      - For each LOW position, read the matched character -> constrains the triple

    Returns (identifications, triple_constraints)
    where triple_constraints = {triple_key: [(implied_char, position_within_syllable), ...]}
    """
    # Build token-to-folio map for folio distribution check
    token_folios: Dict[str, Set[str]] = {}
    for page_id, page in corpus.pages.items():
        for token in page.all_tokens:
            tok_str = token if isinstance(token, str) else str(token)
            if tok_str not in token_folios:
                token_folios[tok_str] = set()
            token_folios[tok_str].add(page_id)

    identifications: List[Dict[str, Any]] = []
    triple_constraints: Dict[str, List[Tuple[str, int]]] = {}

    for (token, regex, target_len, unresolved_positions), matches in zip(
            patterns, all_matches):
        if len(matches) != 1:
            continue

        matched_word = matches[0]
        if len(matched_word) != target_len:
            continue

        # Check folio distribution
        folios = token_folios.get(token, set())
        n_folios = len(folios)
        if n_folios < min_folios:
            continue

        # Record identification
        identifications.append({
            'token': token,
            'matched_word': matched_word,
            'n_folios': n_folios,
            'n_unresolved': len(unresolved_positions),
        })

        # Extract constraints from unresolved positions
        for decoded_pos, triple_key in unresolved_positions:
            if decoded_pos < len(matched_word):
                implied_char = matched_word[decoded_pos]
                # Determine position within syllable (0 or 1 for 2-char CV)
                # Group by triple_key: collect (implied_char, position_within_syllable)
                # We need to figure out which position within the syllable this is.
                # Find all positions in the provenance that share this triple_key
                # and compute relative position.
                # For now, record (implied_char, decoded_pos) — we group later.
                if triple_key not in triple_constraints:
                    triple_constraints[triple_key] = []
                triple_constraints[triple_key].append((implied_char, decoded_pos))

    return identifications, triple_constraints


# ---------------------------------------------------------------------------
# Constraint aggregation
# ---------------------------------------------------------------------------

def _aggregate_constraints(
    triple_constraints: Dict[str, List[Tuple[str, int]]],
    patterns: List[Tuple[str, str, int, List[Tuple[int, str]]]],
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    coda_table: CodaTable,
    confirmed_keys: Set[str],
) -> Tuple[Dict[str, str], List[Dict[str, Any]], float]:
    """Aggregate per-triple constraints into best syllable candidates.

    Each triple produces a 2-char CV syllable.  For each observation, we
    determine the position within the syllable (0 or 1) by examining
    the provenance of the token that produced it.

    Returns (triple_candidates, triple_details, mean_consistency).
    """
    # For each triple_key, collect chars by position-within-syllable
    # position_within_syllable: {triple_key: {0: Counter, 1: Counter}}
    per_triple: Dict[str, Dict[int, Counter]] = {}

    # Re-decode the tokens that contributed constraints to get syllable positions
    for (token, regex, target_len, unresolved_positions), matches in zip(
            patterns, [[] for _ in patterns]):
        # We need to re-examine provenance for position-within-syllable mapping
        pass

    # Simpler approach: for each triple_key, group the implied chars by
    # relative position within that triple's output.
    # A triple produces N chars (typically 2 for CV). When the same triple_key
    # appears at multiple decoded positions in a token, the first is pos 0,
    # second is pos 1, etc.
    # We'll rebuild provenance for each identified token to get this mapping.

    # Build a map: for each identified token, get the provenance
    # and figure out which syllable-internal position each LOW char occupies
    token_set = set()
    for (token, regex, target_len, unresolved_positions) in patterns:
        if unresolved_positions:
            token_set.add(token)

    # Build provenance for these tokens
    token_provenance: Dict[str, List[ProvenanceChar]] = {}
    for token in token_set:
        prov = _decode_with_provenance(
            token, assignment, eva_to_triple, coda_table, confirmed_keys)
        token_provenance[token] = prov

    # Now re-examine the constraints with syllable-internal positions
    for (token, regex, target_len, unresolved_positions), matches_placeholder in zip(
            patterns, [None] * len(patterns)):
        pass  # handled below

    # Direct approach: for each triple_key and its observations,
    # determine syllable-internal position from provenance
    for triple_key, observations in triple_constraints.items():
        if triple_key not in per_triple:
            per_triple[triple_key] = {}

        for implied_char, decoded_pos in observations:
            # Find a token that has this triple_key at this decoded_pos
            # and determine the syllable-internal position
            syl_pos = _get_syllable_internal_position(
                triple_key, decoded_pos, token_provenance, patterns)
            if syl_pos not in per_triple[triple_key]:
                per_triple[triple_key][syl_pos] = Counter()
            per_triple[triple_key][syl_pos][implied_char] += 1

    # Build syllable candidates
    triple_candidates: Dict[str, str] = {}
    triple_details: List[Dict[str, Any]] = []
    consistencies: List[float] = []

    for triple_key in sorted(per_triple.keys()):
        pos_counters = per_triple[triple_key]
        total_obs = sum(sum(c.values()) for c in pos_counters.values())

        # Get best char at each position
        syllable_chars = []
        pos_details = {}
        for pos_idx in sorted(pos_counters.keys()):
            counter = pos_counters[pos_idx]
            top_char, top_count = counter.most_common(1)[0]
            pos_total = sum(counter.values())
            consistency = top_count / pos_total if pos_total > 0 else 0.0
            syllable_chars.append(top_char)
            pos_details[str(pos_idx)] = {
                'top_char': top_char,
                'top_count': top_count,
                'total': pos_total,
                'consistency': round(consistency, 4),
                'all_chars': dict(counter.most_common(10)),
            }
            consistencies.append(consistency)

        best_syllable = ''.join(syllable_chars)
        triple_candidates[triple_key] = best_syllable

        triple_details.append({
            'triple_key': triple_key,
            'best_syllable': best_syllable,
            'total_obs': total_obs,
            'positions': pos_details,
        })

    mean_consistency = (sum(consistencies) / len(consistencies)
                        if consistencies else 0.0)

    return triple_candidates, triple_details, mean_consistency


def _get_syllable_internal_position(
    triple_key: str,
    decoded_pos: int,
    token_provenance: Dict[str, List[ProvenanceChar]],
    patterns: List[Tuple[str, str, int, List[Tuple[int, str]]]],
) -> int:
    """Determine the position within the syllable for a decoded position.

    A triple typically produces 2 chars (CV syllable).  If a triple_key
    appears at decoded positions [3, 4], then pos 3 is syllable-internal
    position 0 and pos 4 is syllable-internal position 1.
    """
    # Search through token provenances to find one that has this triple_key
    # at this decoded_pos
    for token, prov in token_provenance.items():
        # Find all positions in this provenance for the given triple_key
        positions_for_triple = [pc.position for pc in prov
                                if pc.triple_key == triple_key]
        if decoded_pos in positions_for_triple:
            # syllable-internal position = index within the triple's chars
            return positions_for_triple.index(decoded_pos)

    # Fallback: use decoded_pos modulo 2 (typical CV syllable length)
    return decoded_pos % 2


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_expanded_t1():
    """Track 4: CVC-enhanced T1 identification pipeline."""
    t0 = time.time()
    rd = str(_results_dir())

    print("Phase 68.4 — Expanded T1 Identification Pipeline")
    print("=" * 52)

    # --- Load data ---
    confirmed, unresolved = _get_confirmed_and_unresolved(rd)
    confirmed_keys = set(confirmed.keys())
    full_assignment = {**confirmed, **unresolved}
    print(f"  Confirmed triples: {len(confirmed)}")
    print(f"  Unresolved triples: {len(unresolved)}")

    eva_to_triple = build_eva_to_triple_lookup()
    coda_table = build_coda_table_v2()

    # Load corpus
    corpus = load_corpus(verbose=False)
    all_tokens = corpus.get_tokens()
    print(f"  Corpus tokens: {len(all_tokens)}")

    # Build dictionary
    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    base_words = set(w.lower() for w in ref_corpus.get_combined_tokens('latin')
                     if len(w) >= 2)
    expanded, _ = build_expanded_word_set(base_words)
    ref_word_set = base_words | expanded
    dict_by_length = _build_dict_by_length(ref_word_set)
    print(f"  Dictionary size: {len(ref_word_set)}")

    # --- Step 1: Build unique token types ---
    token_types = sorted(set(all_tokens))
    n_token_types = len(token_types)
    print(f"\n  Unique token types: {n_token_types}")

    # --- Step 2: Build patterns ---
    print("  Building wildcard patterns...")
    patterns = _build_patterns(
        token_types, full_assignment, eva_to_triple, coda_table,
        confirmed_keys, min_known_frac=0.50)
    n_patterns_built = len(patterns)
    n_skipped_low_known = n_token_types - n_patterns_built
    print(f"  Patterns built: {n_patterns_built} (skipped {n_skipped_low_known} with < 50% known)")

    # --- Step 3: Match patterns ---
    print("  Matching patterns against dictionary...")
    all_matches = _match_patterns(patterns, dict_by_length, max_matches=20)

    n_unique = 0
    n_few = 0
    n_no = 0
    for matches in all_matches:
        nm = len(matches)
        if nm == 1:
            n_unique += 1
        elif 2 <= nm <= 20:
            n_few += 1
        else:
            n_no += 1

    print(f"  Unique matches (1):   {n_unique}")
    print(f"  Few matches (2-20):   {n_few}")
    print(f"  No matches (0):       {n_no}")

    # --- Step 4: Extract constraints ---
    print("\n  Extracting constraints from unique matches...")
    identifications, triple_constraints = _extract_constraints(
        patterns, all_matches, corpus, all_tokens, min_folios=3)
    n_identifications = len(identifications)
    print(f"  Identifications (unique + >= 3 folios): {n_identifications}")

    if identifications:
        for ident in identifications[:20]:
            print(f"    {ident['token']} -> {ident['matched_word']} "
                  f"({ident['n_folios']} folios, {ident['n_unresolved']} unresolved)")
        if len(identifications) > 20:
            print(f"    ... and {len(identifications) - 20} more")

    # --- Step 5: Aggregate constraints ---
    print("\n  Aggregating triple constraints...")
    triple_candidates, triple_details, mean_consistency = _aggregate_constraints(
        triple_constraints, patterns, full_assignment, eva_to_triple,
        coda_table, confirmed_keys)
    n_triples_constrained = len(triple_candidates)
    print(f"  Triples constrained: {n_triples_constrained}")
    print(f"  Mean consistency:    {mean_consistency:.1%}")

    for detail in triple_details:
        print(f"    {detail['triple_key']}: '{detail['best_syllable']}' "
              f"({detail['total_obs']} observations)")

    # --- Gates ---
    g1 = n_identifications >= 30
    g2 = n_triples_constrained >= 5
    g3 = mean_consistency > 0.60
    gates_passed = sum([g1, g2, g3])

    result = ExpandedT1Result(
        n_token_types=n_token_types,
        n_patterns_built=n_patterns_built,
        n_skipped_low_known=n_skipped_low_known,
        n_unique_matches=n_unique,
        n_few_matches=n_few,
        n_no_matches=n_no,
        identifications=identifications,
        n_identifications=n_identifications,
        triple_candidates=triple_candidates,
        triple_details=triple_details,
        n_triples_constrained=n_triples_constrained,
        mean_consistency=round(mean_consistency, 4),
        g1_ids=g1,
        g2_triples=g2,
        g3_consistency=g3,
        gates_passed=gates_passed,
        gate_passed=gates_passed >= 2,
        runtime_seconds=round(time.time() - t0, 1),
    )

    path = _save_json(rd, 'p68_expanded_t1.json', result)

    # --- Summary ---
    print(f"\n  Summary")
    print(f"  -------")
    print(f"  Identifications: {n_identifications} ({'PASS' if g1 else 'FAIL'} >= 30)")
    print(f"  Triples constrained: {n_triples_constrained} ({'PASS' if g2 else 'FAIL'} >= 5)")
    print(f"  Mean consistency: {mean_consistency:.1%} ({'PASS' if g3 else 'FAIL'} > 60%)")
    print(f"  Gates: {gates_passed}/3")
    print(f"  Saved: {path}")
    print(f"  Time: {result.runtime_seconds:.1f}s")
