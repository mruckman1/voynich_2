"""
Step 33.9 -- Suffix-Constrained Search
========================================
Use suffix-to-grammar mapping from Step 33.8 to constrain the search for
unconfirmed root triples.  If a token has a mapped suffix -> Latin ending,
the decoded root + that ending must form a valid Latin word.

The insight: suffix grammar tells us what Latin ending a given EVA suffix
encodes. So if we know a token ends with EVA suffix "-dy" which maps to
Latin "-us", then (decoded root) + "us" must be a real Latin word.  This
constrains which syllable can be assigned to unconfirmed root triples.

Dependency chain:
    suffix_grammar.json        (Step 33.8 -- suffix-to-ending mapping)
    combined_refine.json       (Phase 15 best_assignment)
    modifier_integrate.json    (Phase 16 modifiers)
    bootstrap_loop.json        (Phase 30 confirmed/unconfirmed triples)
    signal_guided_swap.json    (Step 33.3 -- signal-optimal swaps, optional)
    perplexity_search.json     (Step 33.7 -- perplexity-optimal swaps, optional)
        -> suffix_constrained_search.json  (this step)
"""

import json
import os
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    decode_token_modifier_aware,
    load_corpus,
    token_to_triples,
    tokenize_eva_chars,
)
from voynich.core.reference import (
    build_expanded_word_set,
    load_reference_corpus,
    PHONEME_PLACE_MAP,
    PHONEME_NUCLEUS_MAP,
)
from voynich.phases.csp_solver import decode_token
from voynich.phases.null_corpus import _reconstruct_modifier_rules


# ---------------------------------------------------------------------------
# Helpers
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
    if isinstance(obj, (bool, int, float, str, type(None))):
        return obj
    return str(obj)


# ---------------------------------------------------------------------------
# Suffix / root detection helpers
# ---------------------------------------------------------------------------

def _detect_suffix(
    eva_chars: List[str],
    known_suffixes: Set[str],
) -> Tuple[Optional[str], List[str]]:
    """Detect EVA suffix on a token.

    Returns (suffix, root_chars).  If no suffix found, returns (None, original).
    """
    if not eva_chars:
        return None, eva_chars
    last = eva_chars[-1]
    if last in known_suffixes:
        return last, eva_chars[:-1]
    return None, eva_chars


def _decode_root_chars(
    root_chars: List[str],
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
) -> str:
    """Decode just the root EVA characters (no suffix)."""
    parts = []
    for ch in root_chars:
        triple = eva_to_triple.get(ch)
        if triple and triple in assignment:
            parts.append(assignment[triple])
    return ''.join(parts)


# ---------------------------------------------------------------------------
# Candidate syllable generation (same as signal_guided_swap.py)
# ---------------------------------------------------------------------------

def _generate_candidate_syllables(
    triple_key: str,
    existing_syllables: Set[str],
) -> List[str]:
    """Enumerate CV syllables from phoneme maps for a given triple_key.

    Filters out any syllable already assigned to another triple
    (all-different constraint).
    """
    parts = triple_key.split(',')
    if len(parts) != 3:
        return []
    first_stroke, last_stroke, _glyph_class = parts

    consonants = PHONEME_PLACE_MAP.get(first_stroke, [])
    vowels = PHONEME_NUCLEUS_MAP.get(last_stroke, [])

    candidates = []
    # CV combinations
    for c in consonants:
        for v in vowels:
            syl = c + v
            if syl not in existing_syllables:
                candidates.append(syl)

    # Pure vowels (some triples map to vowel-only syllables)
    for v in vowels:
        if v not in existing_syllables:
            candidates.append(v)

    # Deduplicate while preserving order
    seen: Set[str] = set()
    unique: List[str] = []
    for s in candidates:
        if s not in seen:
            seen.add(s)
            unique.append(s)

    return unique


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SuffixConstrainedTriple:
    triple_key: str
    current_syllable: str
    best_candidate: str
    n_valid_formations: int
    n_total_tokens: int
    valid_fraction: float
    cross_suffix_count: int   # how many different suffixes produce valid words
    signal_agrees: bool       # agrees with signal_guided_swap
    ppl_agrees: bool          # agrees with perplexity_search
    example_words: List[str]  # example valid Latin words formed


@dataclass
class SuffixConstrainedSearchResult:
    n_unconfirmed_triples: int
    n_with_suffix_evidence: int
    n_improvements_found: int
    triple_results: List[Dict]
    best_assignment: Dict[str, str]
    # Three-way agreement
    n_three_way_agree: int
    n_two_way_agree: int
    three_way_triples: List[str]
    # Validation
    dict_hit: float
    baseline_dict_hit: float
    delta_dict_hit: float
    verdict: str  # 'SUFFIX_CONSTRAINTS_FOUND', 'WEAK_CONSTRAINTS', 'NO_CONSTRAINTS'
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------

def _load_unconfirmed_triples(
    rd: str,
    assignment: Dict[str, str],
) -> Tuple[Set[str], Set[str]]:
    """Load confirmed and unconfirmed triple sets.

    Tries bootstrap_loop.json first, then anti_signal_diagnosis.json.
    Falls back to treating all triples as unconfirmed.
    """
    confirmed: Set[str] = set()

    # Try bootstrap_loop.json
    bt_path = os.path.join(rd, 'bootstrap_loop.json')
    if os.path.exists(bt_path):
        with open(bt_path) as f:
            bt_data = json.load(f)
        confirmed = set(bt_data.get('confirmed_triples', []))
        explicit_unconf = bt_data.get('unconfirmed_triples', [])
        if explicit_unconf:
            return confirmed, set(explicit_unconf)

    # Try anti_signal_diagnosis.json
    asd_path = os.path.join(rd, 'anti_signal_diagnosis.json')
    if os.path.exists(asd_path):
        with open(asd_path) as f:
            asd_data = json.load(f)
        for diag in asd_data.get('triple_diagnoses', []):
            if diag.get('confirmed', False):
                confirmed.add(diag.get('triple_key', ''))

    # Derive unconfirmed from assignment - confirmed
    unconfirmed = set(assignment.keys()) - confirmed
    return confirmed, unconfirmed


def _load_signal_swap_assignments(rd: str) -> Dict[str, str]:
    """Load per-triple assignments from signal_guided_swap.json, if available."""
    path = os.path.join(rd, 'signal_guided_swap.json')
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        data = json.load(f)
    return dict(data.get('new_assignment', {}))


def _load_ppl_swap_assignments(rd: str) -> Dict[str, str]:
    """Load per-triple assignments from perplexity_search.json, if available."""
    path = os.path.join(rd, 'perplexity_search.json')
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        data = json.load(f)
    return dict(data.get('best_assignment', data.get('new_assignment', {})))


def _load_suffix_grammars(rd: str) -> List[Dict]:
    """Load suffix grammar list from suffix_grammar.json."""
    path = os.path.join(rd, 'suffix_grammar.json')
    if not os.path.exists(path):
        return []
    with open(path) as f:
        data = json.load(f)
    return data.get('suffix_grammars', [])


def _build_suffix_ending_map(suffix_grammars: List[Dict]) -> Dict[str, str]:
    """Build mapping from EVA suffix -> Latin ending (e.g. 'dy' -> 'us').

    Only includes suffixes with confidence >= 0.3 and a non-empty dominant
    ending.  Strips the leading dash from the ending.
    """
    mapping: Dict[str, str] = {}
    for sg in suffix_grammars:
        suffix = sg.get('suffix', '')
        ending = sg.get('dominant_ending', '')
        confidence = sg.get('confidence', 0.0)
        if suffix and ending and confidence >= 0.3:
            # Strip leading dash (e.g. '-us' -> 'us')
            bare_ending = ending.lstrip('-')
            mapping[suffix] = bare_ending
    return mapping


# ---------------------------------------------------------------------------
# R3 decode for dict_hit computation
# ---------------------------------------------------------------------------

def _decode_corpus_r3(
    tokens: List[str],
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    modifier_chars: set,
    modifier_rules: Dict[str, str],
    ref_word_set: set,
) -> List[str]:
    """Decode tokens using R3 strategy: try alteration, then strip, then raw."""
    decoded = []
    for token in tokens:
        # Alteration
        alt = decode_token_modifier_aware(
            token, assignment, eva_to_triple, modifier_chars, modifier_rules,
        )
        if alt.lower() in ref_word_set:
            decoded.append(alt.lower())
            continue
        # Strip
        stripped = decode_token_modifier_aware(
            token, assignment, eva_to_triple, modifier_chars,
        )
        if stripped.lower() in ref_word_set:
            decoded.append(stripped.lower())
            continue
        # Raw
        raw = decode_token(token, assignment, eva_to_triple)
        decoded.append(raw.lower())
    return decoded


# ---------------------------------------------------------------------------
# Core: find suffix-constrained candidates for each unconfirmed triple
# ---------------------------------------------------------------------------

def _find_suffix_constrained_tokens(
    all_tokens: List[str],
    eva_to_triple: Dict[str, str],
    known_suffixes: Set[str],
    suffix_ending_map: Dict[str, str],
    unconfirmed_triples: Set[str],
) -> Dict[str, List[Dict]]:
    """For each unconfirmed triple, find tokens where:
      - The triple appears in root position (not as the suffix character)
      - The token has a known EVA suffix with a mapped Latin ending

    Returns { triple_key: [ {token, root_chars, suffix, latin_ending}, ... ] }
    """
    triple_tokens: Dict[str, List[Dict]] = defaultdict(list)

    for token in all_tokens:
        eva_chars = tokenize_eva_chars(token)
        if len(eva_chars) < 2:
            # Need at least root + suffix
            continue

        suffix, root_chars = _detect_suffix(eva_chars, known_suffixes)
        if suffix is None:
            continue
        latin_ending = suffix_ending_map.get(suffix)
        if not latin_ending:
            continue
        if not root_chars:
            continue

        # Find which unconfirmed triples appear in root position
        for ch in root_chars:
            triple = eva_to_triple.get(ch)
            if triple and triple in unconfirmed_triples:
                triple_tokens[triple].append({
                    'token': token,
                    'root_chars': root_chars,
                    'suffix': suffix,
                    'latin_ending': latin_ending,
                })

    return dict(triple_tokens)


def _evaluate_candidate_for_triple(
    triple_key: str,
    candidate_syllable: str,
    token_entries: List[Dict],
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    ref_word_set: set,
) -> Tuple[int, int, Dict[str, int], List[str]]:
    """Evaluate a candidate syllable for a triple using suffix constraints.

    For each token where this triple appears in root position and the token
    has a known suffix -> Latin ending:
      - Swap the triple's assignment to candidate_syllable
      - Decode the root portion
      - Append the Latin ending
      - Check if the result is a valid Latin word

    Returns (n_valid, n_total, per_suffix_counts, example_words).
    """
    test_assignment = dict(assignment)
    test_assignment[triple_key] = candidate_syllable

    n_valid = 0
    n_total = len(token_entries)
    per_suffix_valid: Dict[str, int] = Counter()
    examples: List[str] = []

    for entry in token_entries:
        root_chars = entry['root_chars']
        latin_ending = entry['latin_ending']
        suffix = entry['suffix']

        # Decode root with modified assignment
        decoded_root = _decode_root_chars(root_chars, test_assignment, eva_to_triple)
        # Form complete word: root + Latin ending
        candidate_word = decoded_root + latin_ending

        if candidate_word.lower() in ref_word_set:
            n_valid += 1
            per_suffix_valid[suffix] += 1
            if len(examples) < 5:
                examples.append(candidate_word.lower())

    return n_valid, n_total, dict(per_suffix_valid), examples


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_suffix_constrained_search() -> None:
    """Step 33.9: Suffix-Constrained Search."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE 33.9: Suffix-Constrained Search")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load suffix_grammar.json ────────────────────────────────────
    print("\n  1. Loading suffix grammar...")

    suffix_grammars = _load_suffix_grammars(rd)
    if not suffix_grammars:
        print("  [SKIP] suffix_grammar.json not found or empty")
        return

    suffix_ending_map = _build_suffix_ending_map(suffix_grammars)
    known_suffixes = set(suffix_ending_map.keys())

    print(f"     {len(suffix_grammars)} suffix grammars loaded")
    print(f"     {len(suffix_ending_map)} suffixes with confident endings:")
    for sfx, ending in sorted(suffix_ending_map.items()):
        print(f"       {sfx:6s} -> -{ending}")

    # ── 2. Load assignment and modifiers ───────────────────────────────
    print("\n  2. Loading assignment and modifiers...")

    refine_path = os.path.join(rd, 'combined_refine.json')
    if not os.path.exists(refine_path):
        print("  [SKIP] combined_refine.json not found")
        return
    with open(refine_path) as f:
        refine_data = json.load(f)
    assignment = dict(refine_data.get('best_assignment', {}))

    mod_path = os.path.join(rd, 'modifier_integrate.json')
    if not os.path.exists(mod_path):
        print("  [SKIP] modifier_integrate.json not found")
        return
    with open(mod_path) as f:
        mod_data = json.load(f)
    modifier_chars, modifier_rules = _reconstruct_modifier_rules(mod_data)

    print(f"     Assignment: {len(assignment)} triples")
    print(f"     Modifiers: {len(modifier_chars)} chars")

    # ── 3. Build reference word set ────────────────────────────────────
    print("\n  3. Building reference word set...")

    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    base_words = set(
        w.lower() for w in ref_corpus.get_combined_tokens('latin')
        if len(w) >= 2
    )
    expanded, _ = build_expanded_word_set(base_words)
    ref_word_set = base_words | expanded
    print(f"     {len(ref_word_set)} reference words")

    # ── 4. Load corpus and build lookup ────────────────────────────────
    print("\n  4. Loading corpus...")

    corpus = load_corpus(verbose=False)
    eva_to_triple = build_eva_to_triple_lookup()

    all_tokens: List[str] = []
    for folio, page in corpus.pages.items():
        for token in page.all_tokens:
            all_tokens.append(token)
    n_tokens = len(all_tokens)
    print(f"     {n_tokens} tokens")

    # ── 5. Identify unconfirmed triples ────────────────────────────────
    print("\n  5. Identifying unconfirmed triples...")

    confirmed, unconfirmed = _load_unconfirmed_triples(rd, assignment)
    print(f"     Confirmed: {len(confirmed)}")
    print(f"     Unconfirmed: {len(unconfirmed)}")
    for tk in sorted(unconfirmed):
        print(f"       {tk}: {assignment.get(tk, '??')}")

    # ── 6. Find suffix-constrained tokens per triple ───────────────────
    print("\n  6. Finding suffix-constrained tokens for each unconfirmed triple...")

    triple_token_map = _find_suffix_constrained_tokens(
        all_tokens, eva_to_triple, known_suffixes,
        suffix_ending_map, unconfirmed,
    )

    n_with_evidence = len(triple_token_map)
    print(f"     {n_with_evidence}/{len(unconfirmed)} unconfirmed triples "
          f"have suffix-constrained tokens")
    for tk in sorted(triple_token_map.keys()):
        entries = triple_token_map[tk]
        suffixes_seen = set(e['suffix'] for e in entries)
        print(f"       {tk}: {len(entries)} tokens, "
              f"suffixes: {', '.join(sorted(suffixes_seen))}")

    # ── 7. Evaluate candidates for each triple ─────────────────────────
    print("\n  7. Evaluating candidates for each unconfirmed triple...")

    # Load comparison assignments for three-way agreement
    signal_assign = _load_signal_swap_assignments(rd)
    ppl_assign = _load_ppl_swap_assignments(rd)
    has_signal = bool(signal_assign)
    has_ppl = bool(ppl_assign)
    print(f"     Signal-guided assignment loaded: {has_signal}")
    print(f"     Perplexity assignment loaded: {has_ppl}")

    triple_results: List[SuffixConstrainedTriple] = []
    n_improvements = 0

    for triple_key in sorted(triple_token_map.keys()):
        token_entries = triple_token_map[triple_key]
        current_syl = assignment.get(triple_key, '??')

        # Generate candidate syllables (all-different constraint)
        other_syllables = set(
            v for k, v in assignment.items() if k != triple_key
        )
        candidates = _generate_candidate_syllables(triple_key, other_syllables)

        # Always evaluate current syllable for baseline
        curr_valid, curr_total, _, _ = _evaluate_candidate_for_triple(
            triple_key, current_syl, token_entries,
            assignment, eva_to_triple, ref_word_set,
        )

        best_candidate = current_syl
        best_valid = curr_valid
        best_total = curr_total
        best_cross_suffix: Dict[str, int] = {}
        best_examples: List[str] = []

        # Test each candidate
        for cand in candidates:
            n_valid, n_total, per_suffix, examples = _evaluate_candidate_for_triple(
                triple_key, cand, token_entries,
                assignment, eva_to_triple, ref_word_set,
            )

            if n_valid > best_valid:
                best_candidate = cand
                best_valid = n_valid
                best_total = n_total
                best_cross_suffix = per_suffix
                best_examples = examples

        # Cross-suffix count: how many different suffixes produce valid words
        cross_suffix_count = len(best_cross_suffix)

        # Three-way agreement checks
        signal_agrees = (
            signal_assign.get(triple_key) == best_candidate
            if has_signal else False
        )
        ppl_agrees = (
            ppl_assign.get(triple_key) == best_candidate
            if has_ppl else False
        )

        valid_fraction = best_valid / best_total if best_total > 0 else 0.0
        is_improvement = (
            best_candidate != current_syl
            and best_valid > curr_valid
            and best_valid >= 2
        )

        if is_improvement:
            n_improvements += 1

        result = SuffixConstrainedTriple(
            triple_key=triple_key,
            current_syllable=current_syl,
            best_candidate=best_candidate,
            n_valid_formations=best_valid,
            n_total_tokens=best_total,
            valid_fraction=round(valid_fraction, 4),
            cross_suffix_count=cross_suffix_count,
            signal_agrees=signal_agrees,
            ppl_agrees=ppl_agrees,
            example_words=best_examples,
        )
        triple_results.append(result)

        tag = "IMPROVED" if is_improvement else "unchanged"
        agree_str = ""
        if has_signal or has_ppl:
            agree_parts = []
            if signal_agrees:
                agree_parts.append("signal")
            if ppl_agrees:
                agree_parts.append("ppl")
            agree_str = f" agrees=[{','.join(agree_parts)}]" if agree_parts else ""

        print(f"     {triple_key}: {current_syl} -> {best_candidate} "
              f"({best_valid}/{best_total} valid, "
              f"{cross_suffix_count} suffix types) "
              f"[{tag}]{agree_str}")
        if best_examples:
            print(f"       examples: {', '.join(best_examples[:5])}")

    # ── 8. Cross-suffix validation ─────────────────────────────────────
    print("\n  8. Cross-suffix validation...")

    multi_suffix_triples = [
        r for r in triple_results
        if r.cross_suffix_count >= 2 and r.n_valid_formations >= 2
    ]
    print(f"     {len(multi_suffix_triples)} triples validated across "
          f"multiple suffix types")
    for r in multi_suffix_triples:
        print(f"       {r.triple_key}: {r.best_candidate} "
              f"({r.cross_suffix_count} suffix types, "
              f"{r.n_valid_formations} valid)")

    # ── 9. Three-way agreement ─────────────────────────────────────────
    print("\n  9. Three-way agreement analysis...")

    n_three_way = 0
    n_two_way = 0
    three_way_triples: List[str] = []

    for r in triple_results:
        agrees = [True]  # suffix always agrees with itself
        if has_signal:
            agrees.append(r.signal_agrees)
        if has_ppl:
            agrees.append(r.ppl_agrees)

        n_agree = sum(agrees)
        if n_agree >= 3:
            n_three_way += 1
            three_way_triples.append(r.triple_key)
        elif n_agree >= 2:
            n_two_way += 1

    print(f"     Three-way agreement: {n_three_way} triples")
    print(f"     Two-way agreement:   {n_two_way} triples")
    if three_way_triples:
        for tk in three_way_triples:
            r = next(x for x in triple_results if x.triple_key == tk)
            print(f"       {tk}: {r.best_candidate} "
                  f"(all three methods agree)")

    # ── 10. Build best assignment and compute dict_hit ──────────────────
    print("\n  10. Computing dict_hit with suffix-constrained assignment...")

    # Build best assignment: apply improvements from suffix constraints
    best_assignment = dict(assignment)
    for r in triple_results:
        if (r.best_candidate != r.current_syllable
                and r.n_valid_formations >= 2
                and r.valid_fraction > 0.0):
            best_assignment[r.triple_key] = r.best_candidate

    # Baseline dict_hit
    baseline_decoded = _decode_corpus_r3(
        all_tokens, assignment, eva_to_triple,
        modifier_chars, modifier_rules, ref_word_set,
    )
    baseline_hits = sum(1 for w in baseline_decoded if w in ref_word_set)
    baseline_dict_hit = baseline_hits / n_tokens

    # New dict_hit
    new_decoded = _decode_corpus_r3(
        all_tokens, best_assignment, eva_to_triple,
        modifier_chars, modifier_rules, ref_word_set,
    )
    new_hits = sum(1 for w in new_decoded if w in ref_word_set)
    new_dict_hit = new_hits / n_tokens

    delta = new_dict_hit - baseline_dict_hit

    print(f"     Baseline dict_hit: {baseline_dict_hit:.4f}")
    print(f"     New dict_hit:      {new_dict_hit:.4f}")
    print(f"     Delta:             {delta:+.4f}")

    # ── 11. Verdict ────────────────────────────────────────────────────
    # SUFFIX_CONSTRAINTS_FOUND: >= 3 improvements with cross-suffix validation
    # WEAK_CONSTRAINTS: >= 1 improvement
    # NO_CONSTRAINTS: 0 improvements

    if n_improvements >= 3 and len(multi_suffix_triples) >= 2:
        verdict = 'SUFFIX_CONSTRAINTS_FOUND'
    elif n_improvements >= 1:
        verdict = 'WEAK_CONSTRAINTS'
    else:
        verdict = 'NO_CONSTRAINTS'

    print(f"\n  Verdict: {verdict}")
    print(f"     Improvements: {n_improvements}/{n_with_evidence}")
    print(f"     Cross-suffix validated: {len(multi_suffix_triples)}")
    print(f"     Three-way agreement: {n_three_way}")

    # ── 12. Save ───────────────────────────────────────────────────────
    runtime = round(time.time() - t0, 2)

    result = SuffixConstrainedSearchResult(
        n_unconfirmed_triples=len(unconfirmed),
        n_with_suffix_evidence=n_with_evidence,
        n_improvements_found=n_improvements,
        triple_results=[_convert(asdict(r)) for r in triple_results],
        best_assignment=best_assignment,
        n_three_way_agree=n_three_way,
        n_two_way_agree=n_two_way,
        three_way_triples=three_way_triples,
        dict_hit=round(new_dict_hit, 6),
        baseline_dict_hit=round(baseline_dict_hit, 6),
        delta_dict_hit=round(delta, 6),
        verdict=verdict,
        runtime_seconds=runtime,
    )

    out_path = os.path.join(rd, 'suffix_constrained_search.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)

    print(f"\n  Saved -> {out_path}")
    print(f"  Completed in {runtime:.1f}s")
