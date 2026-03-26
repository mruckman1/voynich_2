"""
Phase 67, Track 1: Confidence-Weighted Wildcard Matching
=========================================================
Mark characters from confirmed triples as HIGH-confidence literals and
characters from unresolved triples as LOW-confidence wildcards.  Match
wildcard patterns against the dictionary.  Unique matches constrain what
the unresolved triples must decode to.

Dependency chain:
    results/combined_refine.json      (Phase 15)
    results/triple_tiers.json         (Phase 28/53)
    results/modifier_integrate.json   (Phase 16)
    results/null_corpus.json          (Phase 17)
        -> results/p67_wildcard.json
"""

import json
import os
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

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
class TripleConstraint:
    triple_key: str
    implied_chars: Dict[str, int]   # {char: count}
    top_char: str
    top_count: int
    total_obs: int
    consistency: float
    confident: bool


@dataclass
class WildcardResult:
    phase: str = "67"
    step: str = "67.1"
    experiment: str = "wildcard_matching"
    # Corpus stats
    n_tokens: int = 0
    n_with_wildcards: int = 0
    n_skipped: int = 0          # tokens with < 50% literal
    # Match stats
    n_unique: int = 0
    n_few: int = 0
    n_many: int = 0
    n_none: int = 0
    unique_rate: float = 0.0
    # Confidence quartiles
    by_quartile: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # Constraints propagated to triples
    constraints: List[TripleConstraint] = field(default_factory=list)
    n_consistent_triples: int = 0
    # Null comparison
    null_unique_rate: float = 0.0
    selectivity: float = 0.0
    # Signal word recovery
    signal_words_recovered: List[str] = field(default_factory=list)
    n_signal_recovered: int = 0
    # Top unique matches
    top_unique_words: List[Dict[str, Any]] = field(default_factory=list)
    # Gates
    g1_unique_rate: bool = False       # W1: > 5%
    g2_selectivity: bool = False       # W2: > 1.5×
    g3_consistent: bool = False        # W3: >= 3 triples consistent
    g4_signal_words: bool = False      # W4: >= 5 signal words
    g5_quartile_gradient: bool = False # W5: Q4 > 2× Q1 unique rate
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
# Wildcard pattern building and matching
# ---------------------------------------------------------------------------

def _to_wildcard_pattern(provenance: List[ProvenanceChar]) -> Tuple[str, float]:
    """Convert provenance list to a wildcard pattern.

    HIGH and CODA chars are kept as literals.
    LOW chars are replaced with '.'.

    Returns (regex_pattern, literal_fraction).
    """
    if not provenance:
        return '', 0.0

    pattern_chars = []
    n_literal = 0
    for pc in provenance:
        if pc.confidence in ('HIGH', 'CODA'):
            pattern_chars.append(re.escape(pc.char))
            n_literal += 1
        else:
            pattern_chars.append('[a-z]')

    pattern = ''.join(pattern_chars)
    literal_frac = n_literal / len(provenance) if provenance else 0.0

    return pattern, literal_frac


def _wildcard_match(
    pattern: str,
    target_len: int,
    dict_by_length: Dict[int, List[str]],
    max_matches: int = 20,
) -> List[str]:
    """Find dictionary words matching the wildcard pattern.

    Only tests words of the same length as the pattern.
    """
    words = dict_by_length.get(target_len, [])
    if not words:
        return []

    regex = re.compile('^' + pattern + '$')
    matches = []
    for word in words:
        if regex.match(word):
            matches.append(word)
            if len(matches) >= max_matches:
                break

    return matches


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
# Constraint propagation
# ---------------------------------------------------------------------------

def _propagate_constraints(
    all_provenance: List[List[ProvenanceChar]],
    all_matches: List[List[str]],
    min_literal_frac: float = 0.50,
) -> Dict[str, Counter]:
    """For tokens with UNIQUE matches, extract what each LOW position must be.

    Returns {triple_key: Counter(implied_char -> count)}.
    """
    triple_constraints: Dict[str, Counter] = {}

    for provenance, matches in zip(all_provenance, all_matches):
        if len(matches) != 1:
            continue

        matched_word = matches[0]
        if len(matched_word) != len(provenance):
            continue

        for i, pc in enumerate(provenance):
            if pc.confidence == 'LOW' and pc.triple_key:
                implied_char = matched_word[i]
                if pc.triple_key not in triple_constraints:
                    triple_constraints[pc.triple_key] = Counter()
                triple_constraints[pc.triple_key][implied_char] += 1

    return triple_constraints


# ---------------------------------------------------------------------------
# Null corpus wildcard matching
# ---------------------------------------------------------------------------

def _run_null_wildcard(
    null_tokens: List[str],
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    coda_table: CodaTable,
    confirmed_keys: Set[str],
    dict_by_length: Dict[int, List[str]],
    min_literal_frac: float = 0.50,
) -> float:
    """Run wildcard matching on a null corpus, return unique-match rate."""
    n_attempted = 0
    n_unique = 0

    for token in null_tokens:
        provenance = _decode_with_provenance(
            token, assignment, eva_to_triple, coda_table, confirmed_keys)
        if not provenance:
            continue

        pattern, literal_frac = _to_wildcard_pattern(provenance)
        if literal_frac < min_literal_frac:
            continue

        n_attempted += 1
        target_len = len(provenance)
        matches = _wildcard_match(pattern, target_len, dict_by_length)
        if len(matches) == 1:
            n_unique += 1

    return n_unique / n_attempted if n_attempted > 0 else 0.0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_wildcard_match():
    """Track 1: Confidence-weighted wildcard matching."""
    t0 = time.time()
    rd = str(_results_dir())
    min_literal_frac = 0.50

    print("Phase 67.1 — Confidence-Weighted Wildcard Matching")
    print("=" * 55)

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

    # --- Decode with provenance and match ---
    print("\n  Decoding with provenance and matching...")
    all_provenance: List[List[ProvenanceChar]] = []
    all_matches: List[List[str]] = []
    all_literal_fracs: List[float] = []

    n_skipped = 0
    n_unique = 0
    n_few = 0
    n_many = 0
    n_none = 0
    unique_words_counter: Counter = Counter()

    for token in all_tokens:
        provenance = _decode_with_provenance(
            token, full_assignment, eva_to_triple, coda_table, confirmed_keys)
        all_provenance.append(provenance)

        if not provenance:
            all_matches.append([])
            all_literal_fracs.append(0.0)
            n_skipped += 1
            continue

        pattern, literal_frac = _to_wildcard_pattern(provenance)
        all_literal_fracs.append(literal_frac)

        if literal_frac < min_literal_frac:
            all_matches.append([])
            n_skipped += 1
            continue

        target_len = len(provenance)
        matches = _wildcard_match(pattern, target_len, dict_by_length)
        all_matches.append(matches)

        n_matches = len(matches)
        if n_matches == 1:
            n_unique += 1
            unique_words_counter[matches[0]] += 1
        elif 1 < n_matches <= 5:
            n_few += 1
        elif n_matches > 5:
            n_many += 1
        else:
            n_none += 1

    n_attempted = len(all_tokens) - n_skipped
    unique_rate = n_unique / n_attempted if n_attempted > 0 else 0.0

    print(f"  Attempted: {n_attempted} (skipped {n_skipped} with < {min_literal_frac:.0%} literal)")
    print(f"  Unique:    {n_unique} ({unique_rate:.1%})")
    print(f"  Few (2-5): {n_few}")
    print(f"  Many (6+): {n_many}")
    print(f"  None:      {n_none}")

    # --- Confidence quartile analysis ---
    quartiles: Dict[str, Dict[str, Any]] = {}
    for q_name, lo, hi in [('Q1_low', 0, 0.25), ('Q2', 0.25, 0.5),
                             ('Q3', 0.5, 0.75), ('Q4_high', 0.75, 1.01)]:
        q_indices = [i for i, f in enumerate(all_literal_fracs)
                     if lo <= f < hi and f >= min_literal_frac]
        q_unique = sum(1 for i in q_indices
                       if all_matches[i] and len(all_matches[i]) == 1)
        quartiles[q_name] = {
            'n_tokens': len(q_indices),
            'n_unique': q_unique,
            'unique_rate': q_unique / len(q_indices) if q_indices else 0.0,
        }

    # --- Propagate constraints ---
    print("\n  Propagating constraints to unresolved triples...")
    raw_constraints = _propagate_constraints(all_provenance, all_matches, min_literal_frac)

    constraints = []
    n_consistent = 0
    for triple_key in sorted(unresolved.keys()):
        if triple_key not in raw_constraints:
            continue

        char_counts = raw_constraints[triple_key]
        total = sum(char_counts.values())
        top_char, top_count = char_counts.most_common(1)[0]
        consistency = top_count / total if total > 0 else 0.0
        confident = consistency > 0.70 and total >= 10

        if confident:
            n_consistent += 1

        constraints.append(TripleConstraint(
            triple_key=triple_key,
            implied_chars=dict(char_counts.most_common(10)),
            top_char=top_char,
            top_count=top_count,
            total_obs=total,
            consistency=round(consistency, 4),
            confident=confident,
        ))

        conf_mark = " ** CONFIDENT" if confident else ""
        print(f"    {triple_key}: '{top_char}' ({top_count}/{total}, "
              f"consistency={consistency:.1%}){conf_mark}")

    # --- Signal word recovery ---
    from voynich.phases.suffix_calibration import SIGNAL_WORDS_51
    signal_set = set(SIGNAL_WORDS_51.keys())
    recovered = signal_set & set(unique_words_counter.keys())
    n_signal_recovered = len(recovered)
    print(f"\n  Signal words recovered via unique match: {n_signal_recovered}")

    # --- Null comparison ---
    print("\n  Running null wildcard comparison...")
    mod_data = _safe_load(os.path.join(rd, 'modifier_integrate.json'))
    modifier_chars, modifier_rules = _reconstruct_modifier_rules(mod_data)

    null_data = _safe_load(os.path.join(rd, 'null_corpus.json'))
    null_seeds = ([r['seed'] for r in null_data.get('null_runs', [])]
                  if null_data else [100, 101, 102, 103, 104])

    bigram_probs, initial_probs, token_lengths = _build_eva_bigram_model(all_tokens)

    null_rates = []
    for seed in null_seeds[:3]:  # Use 3 null corpora for speed
        null_tokens = _generate_null_corpus(
            bigram_probs, initial_probs, token_lengths, len(all_tokens), seed)
        rate = _run_null_wildcard(
            null_tokens, full_assignment, eva_to_triple, coda_table,
            confirmed_keys, dict_by_length, min_literal_frac)
        null_rates.append(rate)
        print(f"    Null seed {seed}: {rate:.1%} unique")

    null_mean_rate = sum(null_rates) / len(null_rates) if null_rates else 0.0
    selectivity = unique_rate / null_mean_rate if null_mean_rate > 0 else float('inf')
    print(f"  Null mean:     {null_mean_rate:.1%}")
    print(f"  Selectivity:   {selectivity:.2f}×")

    # --- Gates ---
    q4_rate = quartiles.get('Q4_high', {}).get('unique_rate', 0)
    q1_rate = quartiles.get('Q1_low', {}).get('unique_rate', 0)
    g1 = unique_rate > 0.05
    g2 = selectivity > 1.5
    g3 = n_consistent >= 3
    g4 = n_signal_recovered >= 5
    g5 = q4_rate > 2 * q1_rate if q1_rate > 0 else q4_rate > 0
    gates_passed = sum([g1, g2, g3, g4, g5])

    top_words = [{'word': w, 'count': c}
                 for w, c in unique_words_counter.most_common(50)]

    result = WildcardResult(
        n_tokens=len(all_tokens),
        n_with_wildcards=n_attempted,
        n_skipped=n_skipped,
        n_unique=n_unique,
        n_few=n_few,
        n_many=n_many,
        n_none=n_none,
        unique_rate=round(unique_rate, 4),
        by_quartile=quartiles,
        constraints=constraints,
        n_consistent_triples=n_consistent,
        null_unique_rate=round(null_mean_rate, 4),
        selectivity=round(selectivity, 4),
        signal_words_recovered=sorted(recovered),
        n_signal_recovered=n_signal_recovered,
        top_unique_words=top_words,
        g1_unique_rate=g1,
        g2_selectivity=g2,
        g3_consistent=g3,
        g4_signal_words=g4,
        g5_quartile_gradient=g5,
        gates_passed=gates_passed,
        gate_passed=gates_passed >= 3,
        runtime_seconds=round(time.time() - t0, 1),
    )

    path = _save_json(rd, 'p67_wildcard.json', result)

    # --- Summary ---
    print(f"\n  Summary")
    print(f"  -------")
    print(f"  Unique rate:       {unique_rate:.1%} ({'PASS' if g1 else 'FAIL'} > 5%)")
    print(f"  Selectivity:       {selectivity:.2f}× ({'PASS' if g2 else 'FAIL'} > 1.5×)")
    print(f"  Consistent triples: {n_consistent} ({'PASS' if g3 else 'FAIL'} >= 3)")
    print(f"  Signal recovered:  {n_signal_recovered} ({'PASS' if g4 else 'FAIL'} >= 5)")
    print(f"  Quartile gradient: {'PASS' if g5 else 'FAIL'} (Q4 > 2× Q1)")
    print(f"  Gates: {gates_passed}/5")
    print(f"  Saved: {path}")
    print(f"  Time: {result.runtime_seconds:.1f}s")
