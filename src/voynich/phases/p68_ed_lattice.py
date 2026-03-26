"""
Phase 68, Track 7: Edit-Distance Lattice
==========================================
For each decoded token, find dictionary words within edit distance 2
where edits fall on positions from unresolved triples.  Aggregate
implied character values across occurrences.

Dependency chain:
    results/combined_refine.json      (Phase 15)
    results/triple_tiers.json         (Phase 28/53)
        -> results/p68_ed_lattice.json
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
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class EdLatticeResult:
    phase: str = "68"
    step: str = "68.7"
    experiment: str = "ed_lattice"
    n_token_types: int = 0
    n_types_with_unresolved: int = 0
    n_types_with_neighbors: int = 0
    n_total_neighbors_found: int = 0
    n_triples_constrained: int = 0
    n_triples_clear_top: int = 0
    # Triple candidates
    triple_candidates: Dict[str, str] = field(default_factory=dict)
    triple_details: List[Dict[str, Any]] = field(default_factory=list)
    # Sample results
    sample_lattice: List[Dict[str, Any]] = field(default_factory=list)
    # Gates
    g1_tokens: bool = False         # EL1: >= 500 types with neighbors
    g2_triples: bool = False        # EL2: >= 8 triples constrained
    g3_clear: bool = False          # EL3: >= 3 triples with clear top (2x weight)
    gates_passed: int = 0
    gate_passed: bool = False
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Provenance-tracking decode
# ---------------------------------------------------------------------------

@dataclass
class ProvenanceChar:
    """One decoded character with provenance information."""
    char: str
    confidence: str    # HIGH, LOW, CODA
    triple_key: str    # which triple produced this (empty for CODA)
    position: int      # position in decoded string
    syllable_pos: int  # 0 or 1 — position within the 2-char syllable


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

            for syl_pos, c in enumerate(syllable):
                result.append(ProvenanceChar(
                    char=c,
                    confidence=confidence,
                    triple_key=triple_key,
                    position=pos,
                    syllable_pos=syl_pos,
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
                    syllable_pos=0,
                ))
                pos += 1

    return result


# ---------------------------------------------------------------------------
# Edit-distance computation
# ---------------------------------------------------------------------------

def _edit_distance(a: str, b: str) -> int:
    """Standard Levenshtein edit distance via dynamic programming."""
    m, n = len(a), len(b)
    if m == 0:
        return n
    if n == 0:
        return m

    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp

    return dp[n]


# ---------------------------------------------------------------------------
# Dictionary indexing by length
# ---------------------------------------------------------------------------

def _build_dict_by_length(ref_word_set: Set[str]) -> Dict[int, List[str]]:
    """Pre-index dictionary words by length for fast lookup."""
    by_len: Dict[int, List[str]] = {}
    for word in ref_word_set:
        wl = len(word)
        if wl not in by_len:
            by_len[wl] = []
        by_len[wl].append(word)
    return by_len


# ---------------------------------------------------------------------------
# Lattice neighbor search
# ---------------------------------------------------------------------------

def _find_lattice_neighbors(
    decoded_word: str,
    dict_by_length: Dict[int, List[str]],
    max_ed: int = 2,
) -> List[Tuple[str, int]]:
    """Find all dictionary words within edit distance max_ed of decoded_word.

    Pre-filters by length: |len(word) - len(decoded)| <= max_ed.
    Returns list of (word, ed) tuples sorted by ed then alphabetically.
    """
    dlen = len(decoded_word)
    neighbors: List[Tuple[str, int]] = []

    for wlen in range(max(1, dlen - max_ed), dlen + max_ed + 1):
        candidates = dict_by_length.get(wlen, [])
        for word in candidates:
            ed = _edit_distance(decoded_word, word)
            if 0 < ed <= max_ed:
                neighbors.append((word, ed))

    neighbors.sort(key=lambda x: (x[1], x[0]))
    return neighbors


# ---------------------------------------------------------------------------
# Extract unresolved edits — syllable-aware
# ---------------------------------------------------------------------------

def _extract_unresolved_edits(
    decoded: str,
    neighbor_word: str,
    provenance: List[ProvenanceChar],
) -> Dict[str, str]:
    """Compare decoded and neighbor character by character.

    For positions where they differ AND the provenance is LOW (unresolved),
    extract the full 2-char syllable from the neighbor for that triple_key.

    Only handles same-length comparisons (substitutions).  Skips if
    lengths differ.

    Returns dict: {triple_key: implied_syllable}.
    """
    if len(decoded) != len(neighbor_word):
        return {}
    if len(provenance) != len(decoded):
        return {}

    # First pass: identify which triples have differing positions
    triples_with_diffs: Set[str] = set()
    for i in range(len(decoded)):
        if decoded[i] != neighbor_word[i] and provenance[i].confidence == 'LOW':
            tk = provenance[i].triple_key
            if tk:
                triples_with_diffs.add(tk)

    if not triples_with_diffs:
        return {}

    # Second pass: for each differing triple, extract the full syllable
    # from the neighbor word (both chars belonging to that triple)
    result: Dict[str, str] = {}
    for tk in triples_with_diffs:
        # Collect all positions belonging to this triple
        positions = [i for i, pc in enumerate(provenance)
                     if pc.triple_key == tk]
        if not positions:
            continue

        # Only extract if ALL positions for this triple are present and
        # we can get the neighbor chars at those positions
        neighbor_syllable = ''.join(neighbor_word[p] for p in positions)
        result[tk] = neighbor_syllable

    return result


# ---------------------------------------------------------------------------
# Process all token types
# ---------------------------------------------------------------------------

def _process_all_types(
    token_types_with_freq: List[Tuple[str, int]],
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    coda_table: CodaTable,
    confirmed_keys: Set[str],
    unresolved_keys: Set[str],
    dict_by_length: Dict[int, List[str]],
    max_ed: int = 2,
) -> Tuple[List[Dict[str, Any]], Dict[str, List[Tuple[str, int]]]]:
    """Process each token type that contains >= 1 unresolved triple.

    Returns:
        per_type_results: list of per-type summary dicts
        all_constraints: {triple_key: [(implied_syllable, weight), ...]}
    """
    per_type_results: List[Dict[str, Any]] = []
    all_constraints: Dict[str, List[Tuple[str, int]]] = {}

    n_processed = 0
    for idx, (token, freq) in enumerate(token_types_with_freq):
        provenance = _decode_with_provenance(
            token, assignment, eva_to_triple, coda_table, confirmed_keys)
        if not provenance:
            continue

        # Check if this token has any unresolved triples
        token_unresolved = set()
        for pc in provenance:
            if pc.confidence == 'LOW' and pc.triple_key in unresolved_keys:
                token_unresolved.add(pc.triple_key)

        if not token_unresolved:
            continue

        decoded_word = ''.join(pc.char for pc in provenance)

        # Skip tokens that decoded to something with '?' (unmapped triples)
        if '?' in decoded_word:
            continue

        neighbors = _find_lattice_neighbors(decoded_word, dict_by_length, max_ed)

        type_entry: Dict[str, Any] = {
            'token': token,
            'decoded': decoded_word,
            'freq': freq,
            'n_neighbors': len(neighbors),
            'unresolved_triples': sorted(token_unresolved),
        }

        if neighbors:
            type_entry['top_neighbors'] = [
                {'word': w, 'ed': e} for w, e in neighbors[:5]
            ]

            # Extract constraints from each neighbor
            for neighbor_word, ed in neighbors:
                implied = _extract_unresolved_edits(
                    decoded_word, neighbor_word, provenance)
                weight = freq  # weight by type frequency
                for tk, implied_syl in implied.items():
                    if tk not in all_constraints:
                        all_constraints[tk] = []
                    all_constraints[tk].append((implied_syl, weight))

        per_type_results.append(type_entry)
        n_processed += 1

        if n_processed % 500 == 0:
            print(f"    Processed {n_processed} types with unresolved triples...")

    return per_type_results, all_constraints


# ---------------------------------------------------------------------------
# Aggregate lattice votes
# ---------------------------------------------------------------------------

def _aggregate_lattice_votes(
    all_constraints: Dict[str, List[Tuple[str, int]]],
) -> Tuple[Dict[str, str], List[Dict[str, Any]]]:
    """Per triple_key, accumulate implied syllables weighted by
    1/(ED + 0.5) * type_frequency.  Pick top candidate.

    Returns:
        triple_candidates: {triple_key: best_syllable}
        triple_details: list of per-triple detail dicts
    """
    triple_candidates: Dict[str, str] = {}
    triple_details: List[Dict[str, Any]] = []

    for tk in sorted(all_constraints.keys()):
        votes: Counter = Counter()
        for implied_syl, weight in all_constraints[tk]:
            # Weight already includes type frequency from _process_all_types;
            # the ED weighting is handled here via the tuples
            votes[implied_syl] += weight

        if not votes:
            continue

        top_syl, top_weight = votes.most_common(1)[0]
        total_weight = sum(votes.values())
        dominance = top_weight / total_weight if total_weight > 0 else 0.0

        # Check if top candidate has >= 2x weight of runner-up
        runner_up_weight = 0
        if len(votes) > 1:
            runner_up_weight = votes.most_common(2)[1][1]
        clear_top = top_weight >= 2 * runner_up_weight if runner_up_weight > 0 else True

        triple_candidates[tk] = top_syl

        detail: Dict[str, Any] = {
            'triple_key': tk,
            'best_syllable': top_syl,
            'best_weight': top_weight,
            'total_weight': total_weight,
            'dominance': round(dominance, 4),
            'n_distinct': len(votes),
            'clear_top': clear_top,
            'top_candidates': [
                {'syllable': s, 'weight': w}
                for s, w in votes.most_common(5)
            ],
        }
        triple_details.append(detail)

    return triple_candidates, triple_details


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_ed_lattice():
    """Track 7: Edit-Distance Lattice."""
    t0 = time.time()
    rd = str(_results_dir())
    max_ed = 2
    min_type_freq = 3

    print("Phase 68.7 — Edit-Distance Lattice")
    print("=" * 55)

    # --- Load data ---
    confirmed, unresolved = _get_confirmed_and_unresolved(rd)
    confirmed_keys = set(confirmed.keys())
    unresolved_keys = set(unresolved.keys())
    full_assignment = {**confirmed, **unresolved}
    print(f"  Confirmed triples: {len(confirmed)}")
    print(f"  Unresolved triples: {len(unresolved)}")

    eva_to_triple = build_eva_to_triple_lookup()
    coda_table = build_coda_table_v2()

    # Load corpus
    corpus = load_corpus(verbose=False)
    all_tokens = corpus.get_tokens()
    print(f"  Corpus tokens: {len(all_tokens)}")

    # Get unique token types with frequencies
    type_counts: Counter = Counter(all_tokens)
    # Filter to types appearing >= min_type_freq (skip hapax)
    token_types_with_freq = [
        (tok, freq) for tok, freq in type_counts.most_common()
        if freq >= min_type_freq
    ]
    n_token_types = len(token_types_with_freq)
    print(f"  Token types (freq >= {min_type_freq}): {n_token_types}")

    # Build dictionary
    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    base_words = set(w.lower() for w in ref_corpus.get_combined_tokens('latin')
                     if len(w) >= 2)
    expanded, _ = build_expanded_word_set(base_words)
    ref_word_set = base_words | expanded
    dict_by_length = _build_dict_by_length(ref_word_set)
    print(f"  Dictionary size: {len(ref_word_set)}")

    # --- Find types with unresolved triples ---
    print("\n  Scanning for types with unresolved triples...")
    types_with_unresolved: List[Tuple[str, int]] = []
    for token, freq in token_types_with_freq:
        provenance = _decode_with_provenance(
            token, full_assignment, eva_to_triple, coda_table, confirmed_keys)
        if not provenance:
            continue
        has_unresolved = any(
            pc.confidence == 'LOW' and pc.triple_key in unresolved_keys
            for pc in provenance
        )
        if has_unresolved:
            types_with_unresolved.append((token, freq))

    n_types_with_unresolved = len(types_with_unresolved)
    print(f"  Types with unresolved triples: {n_types_with_unresolved}")

    # --- Process all types ---
    print(f"\n  Finding lattice neighbors (max ED={max_ed})...")
    per_type_results, all_constraints = _process_all_types(
        types_with_unresolved,
        full_assignment,
        eva_to_triple,
        coda_table,
        confirmed_keys,
        unresolved_keys,
        dict_by_length,
        max_ed,
    )

    n_types_with_neighbors = sum(
        1 for r in per_type_results if r.get('n_neighbors', 0) > 0
    )
    n_total_neighbors = sum(
        r.get('n_neighbors', 0) for r in per_type_results
    )
    print(f"  Types with >= 1 neighbor: {n_types_with_neighbors}")
    print(f"  Total neighbors found: {n_total_neighbors}")

    # --- Aggregate votes ---
    print("\n  Aggregating lattice votes...")
    triple_candidates, triple_details = _aggregate_lattice_votes(all_constraints)

    n_triples_constrained = len(triple_candidates)
    n_triples_clear_top = sum(
        1 for d in triple_details if d.get('clear_top', False)
    )
    print(f"  Triples constrained: {n_triples_constrained}")
    print(f"  Triples with clear top (2x weight): {n_triples_clear_top}")

    for detail in triple_details:
        clear_mark = " ** CLEAR" if detail['clear_top'] else ""
        print(f"    {detail['triple_key']}: '{detail['best_syllable']}' "
              f"(weight={detail['best_weight']}, "
              f"dominance={detail['dominance']:.1%}, "
              f"n_distinct={detail['n_distinct']}){clear_mark}")

    # --- Sample lattice entries ---
    sample_lattice = [
        r for r in per_type_results if r.get('n_neighbors', 0) > 0
    ][:30]

    # --- Gates ---
    g1 = n_types_with_neighbors >= 500
    g2 = n_triples_constrained >= 8
    g3 = n_triples_clear_top >= 3
    gates_passed = sum([g1, g2, g3])

    result = EdLatticeResult(
        n_token_types=n_token_types,
        n_types_with_unresolved=n_types_with_unresolved,
        n_types_with_neighbors=n_types_with_neighbors,
        n_total_neighbors_found=n_total_neighbors,
        n_triples_constrained=n_triples_constrained,
        n_triples_clear_top=n_triples_clear_top,
        triple_candidates=triple_candidates,
        triple_details=triple_details,
        sample_lattice=sample_lattice,
        g1_tokens=g1,
        g2_triples=g2,
        g3_clear=g3,
        gates_passed=gates_passed,
        gate_passed=gates_passed >= 2,
        runtime_seconds=round(time.time() - t0, 1),
    )

    path = _save_json(rd, 'p68_ed_lattice.json', result)

    # --- Summary ---
    print(f"\n  Summary")
    print(f"  -------")
    print(f"  Types with neighbors: {n_types_with_neighbors} "
          f"({'PASS' if g1 else 'FAIL'} >= 500)")
    print(f"  Triples constrained: {n_triples_constrained} "
          f"({'PASS' if g2 else 'FAIL'} >= 8)")
    print(f"  Clear top (2x):      {n_triples_clear_top} "
          f"({'PASS' if g3 else 'FAIL'} >= 3)")
    print(f"  Gates: {gates_passed}/3")
    print(f"  Saved: {path}")
    print(f"  Time: {result.runtime_seconds:.1f}s")
