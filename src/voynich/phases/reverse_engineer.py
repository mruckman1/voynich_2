"""
Step 24.14 -- Reverse Engineering from Confirmed Words
=====================================================
Work backward from confirmed decoded words to extract definite
character-level assignments. Bootstrap from these anchors.

Dependency chain:
    cross_approach.json (Phase 19.8)
    modifier_integrate.json (Phase 16)
    combined_refine.json (Phase 15)
        -> reverse_engineering.json (this step)
"""

import json
import os
import random
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    decode_token_modifier_aware,
    load_corpus,
    tokenize_eva_chars,
)
from voynich.core.reference import (
    EVA_VISUAL_COMPONENTS,
    build_cv_syllable_table,
    build_expanded_word_set,
    load_reference_corpus,
)
from voynich.phases.csp_solver import decode_token


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
# Confirmed word list
# ---------------------------------------------------------------------------

CONFIRMED_WORDS = [
    # Phase 19.8: cross-approach bidirectional matches
    {'latin': 'de', 'source': 'phase19.8', 'match_type': 'exact'},
    {'latin': 'bene', 'source': 'phase19.8', 'match_type': 'exact'},
    {'latin': 'et', 'source': 'phase19.8', 'match_type': 'edit_le_2'},
    {'latin': 'in', 'source': 'phase19.8', 'match_type': 'edit_le_2'},
    {'latin': 'terra', 'source': 'phase19.8', 'match_type': 'edit_le_2'},
    {'latin': 'rosa', 'source': 'phase19.8', 'match_type': 'edit_le_2'},
    {'latin': 'sal', 'source': 'phase19.8', 'match_type': 'edit_le_2'},
    {'latin': 'adde', 'source': 'phase19.8', 'match_type': 'edit_le_2'},
    # Phase 12/13: illustration-correlated matches
    {'latin': 'aqua', 'source': 'phase12', 'match_type': 'illustration'},
    {'latin': 'bibe', 'source': 'phase12', 'match_type': 'illustration'},
    {'latin': 'coque', 'source': 'phase12', 'match_type': 'illustration'},
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ConfirmedAlignment:
    latin_word: str
    source: str
    match_type: str
    eva_token: str
    eva_chars: List[str]
    syllables: List[str]
    char_to_syllable: Dict[str, str]  # eva_char -> syllable
    n_aligned: int


@dataclass
class CharAssignment:
    eva_char: str
    triple_key: str
    assigned_syllable: str
    n_sources: int
    sources: List[str]  # which confirmed words support this
    agrees_with_phase16: bool
    phase16_syllable: str


@dataclass
class ReverseEngineerResult:
    timestamp: str
    # Confirmed words
    n_confirmed_words: int
    n_found_in_corpus: int
    confirmed_alignments: List[Dict]
    # Character-level table
    n_chars_assigned: int
    n_chars_consistent: int
    n_chars_contradictory: int
    char_assignments: List[Dict]
    contradictions: List[Dict]  # {eva_char, syllable_1, source_1, syllable_2, source_2}
    # Bootstrap
    n_bootstrap_iterations: int
    n_new_words_found: int
    bootstrap_words: List[str]
    # Phase 16 comparison
    n_agrees_with_phase16: int
    n_disagrees_with_phase16: int
    disagreements: List[Dict]  # {eva_char, reverse_eng_syllable, phase16_syllable}
    # Verdict
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Latin syllabification
# ---------------------------------------------------------------------------

def _syllabify_latin(word: str) -> List[str]:
    """Split Latin word into CV-ish syllables."""
    vowels = set('aeiou')
    syllables: List[str] = []
    current = ''
    for ch in word:
        current += ch
        if ch in vowels:
            syllables.append(current)
            current = ''
    if current:
        if syllables:
            syllables[-1] += current
        else:
            syllables.append(current)
    return syllables


# ---------------------------------------------------------------------------
# Helper: safe decode
# ---------------------------------------------------------------------------

def _decode_token_safe(
    token: str,
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    modifier_chars: Set[str],
) -> str:
    """Decode a token via Phase 15/16 pipeline, returning the decoded string."""
    try:
        return decode_token_modifier_aware(
            token, assignment, eva_to_triple, modifier_chars,
        )
    except Exception:
        # Fallback: direct triple mapping
        from voynich.core.corpus import token_to_triples
        triples = token_to_triples(token, eva_to_triple)
        return ''.join(assignment.get(t, '?') for t in triples)


def _load_json(path: str) -> Optional[Dict]:
    """Load a JSON file if it exists, else return None."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Core algorithms
# ---------------------------------------------------------------------------

def _find_tokens_for_word(
    target_word: str,
    all_tokens: List[str],
    decoded_cache: Dict[str, str],
) -> List[str]:
    """Find all EVA tokens whose decoded form matches the target Latin word."""
    matches = []
    target_lower = target_word.lower()
    for tok in set(all_tokens):
        decoded = decoded_cache.get(tok, '')
        if decoded.lower() == target_lower:
            matches.append(tok)
    return sorted(matches)


def _align_chars_to_syllables(
    eva_chars: List[str],
    syllables: List[str],
    modifier_chars: Set[str],
) -> Dict[str, str]:
    """
    Align non-modifier EVA characters to Latin syllables 1:1.

    Each non-modifier EVA char maps to one triple which maps to one syllable.
    Returns a dict from eva_char -> syllable for all aligned chars.
    Returns empty dict if lengths do not match.
    """
    # Filter out modifier chars
    syllabic_chars = [ch for ch in eva_chars if ch not in modifier_chars]

    if len(syllabic_chars) != len(syllables):
        return {}

    mapping: Dict[str, str] = {}
    for ch, syl in zip(syllabic_chars, syllables):
        mapping[ch] = syl

    return mapping


def _build_char_table(
    alignments: List[ConfirmedAlignment],
    eva_to_triple: Dict[str, str],
    assignment: Dict[str, str],
) -> Tuple[List[CharAssignment], List[Dict]]:
    """
    Build a character-level partial assignment table from all confirmed alignments.

    Returns (char_assignments, contradictions).
    """
    # Collect all (eva_char -> syllable) assignments with their sources
    char_evidence: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))

    for alignment in alignments:
        for eva_char, syllable in alignment.char_to_syllable.items():
            char_evidence[eva_char][syllable].append(
                f"{alignment.latin_word}({alignment.source})"
            )

    char_assignments: List[CharAssignment] = []
    contradictions: List[Dict] = []

    for eva_char in sorted(char_evidence.keys()):
        syl_map = char_evidence[eva_char]
        triple_key = eva_to_triple.get(eva_char, '?')
        phase16_syl = assignment.get(triple_key, '?')

        if len(syl_map) == 1:
            # Consistent: only one syllable assignment
            syllable = list(syl_map.keys())[0]
            all_sources = list(syl_map.values())[0]
            agrees = (syllable == phase16_syl)

            char_assignments.append(CharAssignment(
                eva_char=eva_char,
                triple_key=triple_key,
                assigned_syllable=syllable,
                n_sources=len(all_sources),
                sources=all_sources,
                agrees_with_phase16=agrees,
                phase16_syllable=phase16_syl,
            ))
        else:
            # Contradictory: multiple syllable assignments
            syls = list(syl_map.keys())
            for i in range(len(syls)):
                for j in range(i + 1, len(syls)):
                    contradictions.append({
                        'eva_char': eva_char,
                        'syllable_1': syls[i],
                        'source_1': syl_map[syls[i]],
                        'syllable_2': syls[j],
                        'source_2': syl_map[syls[j]],
                    })

            # Use the most-attested syllable as the "best" assignment
            best_syl = max(syl_map, key=lambda s: len(syl_map[s]))
            all_sources_flat = []
            for sources in syl_map.values():
                all_sources_flat.extend(sources)
            agrees = (best_syl == phase16_syl)

            char_assignments.append(CharAssignment(
                eva_char=eva_char,
                triple_key=triple_key,
                assigned_syllable=best_syl,
                n_sources=len(all_sources_flat),
                sources=all_sources_flat,
                agrees_with_phase16=agrees,
                phase16_syllable=phase16_syl,
            ))

    return char_assignments, contradictions


def _bootstrap_decode(
    partial_table: Dict[str, str],
    all_tokens: List[str],
    modifier_chars: Set[str],
    eva_to_triple: Dict[str, str],
    ref_word_set: set,
    max_iterations: int = 10,
) -> Tuple[int, int, List[str], Dict[str, str]]:
    """
    Use the partial character-level table to bootstrap new word discoveries.

    For each corpus token:
    - If all non-modifier chars have assignments in the partial table -> fully decoded
    - If the decoded token matches a dictionary word -> new confirmed word
    - Iterate: add new confirmed words, re-align, expand table

    Returns (n_iterations, n_new_words, new_word_list, final_table).
    """
    current_table = dict(partial_table)
    all_new_words: List[str] = []
    unique_tokens = sorted(set(all_tokens))

    for iteration in range(max_iterations):
        new_words_this_round: List[str] = []

        for token in unique_tokens:
            chars = tokenize_eva_chars(token)
            syllabic_chars = [ch for ch in chars if ch not in modifier_chars]

            if not syllabic_chars:
                continue

            # Check if all syllabic chars have assignments
            all_assigned = all(ch in current_table for ch in syllabic_chars)
            if not all_assigned:
                continue

            # Decode using partial table
            decoded = ''.join(current_table[ch] for ch in syllabic_chars)

            if decoded.lower() in ref_word_set and decoded.lower() not in all_new_words:
                new_words_this_round.append(decoded.lower())

                # Reverse-align to potentially expand the table
                syllables = _syllabify_latin(decoded.lower())
                if len(syllabic_chars) == len(syllables):
                    for ch, syl in zip(syllabic_chars, syllables):
                        if ch not in current_table:
                            current_table[ch] = syl

        if not new_words_this_round:
            print(f"      Iteration {iteration + 1}: no new words, stopping.")
            return iteration + 1, len(all_new_words), all_new_words, current_table

        all_new_words.extend(new_words_this_round)
        print(f"      Iteration {iteration + 1}: {len(new_words_this_round)} new words "
              f"(total: {len(all_new_words)})")

    return max_iterations, len(all_new_words), all_new_words, current_table


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_reverse_engineer() -> None:
    """Step 24.14: Reverse-engineer character assignments from confirmed words."""
    t0 = time.time()
    rd = str(_results_dir())

    print("=" * 60)
    print("Step 24.14: Reverse Engineering from Confirmed Words")
    print("=" * 60)

    # ── 1. Load dependencies ──────────────────────────────────────────
    print("\n  1. Loading dependencies ...")

    refine_data = _load_json(os.path.join(rd, 'combined_refine.json'))
    mod_data = _load_json(os.path.join(rd, 'modifier_integrate.json'))
    cross_data = _load_json(os.path.join(rd, 'cross_approach.json'))

    # Extract assignment from combined_refine.json
    assignment: Dict[str, str] = {}
    if refine_data:
        for key in ['best_assignment', 'assignment', 'latin_assignment', 'best_latin_assignment']:
            if key in refine_data:
                assignment = refine_data[key]
                break

    # Extract modifier chars from modifier_integrate.json
    modifier_chars: Set[str] = set()
    if mod_data and 'modifier_chars' in mod_data:
        modifier_chars = set(mod_data['modifier_chars'])

    # Build EVA-to-triple lookup
    eva_to_triple = build_eva_to_triple_lookup()

    # Load corpus
    corpus = load_corpus(verbose=False)
    all_tokens = corpus.get_tokens()

    # Load reference word set (expanded)
    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    base_words = set(
        w.lower() for w in ref_corpus.get_combined_tokens('latin')
        if len(w) >= 2
    )
    expanded_set, _ = build_expanded_word_set(base_words)
    ref_word_set = base_words | expanded_set

    print(f"    Assignment: {len(assignment)} triple mappings")
    print(f"    Modifiers: {len(modifier_chars)} chars: {sorted(modifier_chars)}")
    print(f"    Corpus: {len(all_tokens)} tokens ({len(set(all_tokens))} unique)")
    print(f"    Dictionary: {len(ref_word_set)} words (base: {len(base_words)})")

    # ── 2. Decode all tokens ─────────────────────────────────────────
    print("\n  2. Decoding all tokens via Phase 15/16 pipeline ...")

    decoded_cache: Dict[str, str] = {}
    for tok in set(all_tokens):
        decoded_cache[tok] = _decode_token_safe(tok, assignment, eva_to_triple, modifier_chars)

    n_decoded = sum(1 for d in decoded_cache.values() if d and '?' not in d)
    print(f"    {n_decoded}/{len(decoded_cache)} tokens successfully decoded")

    # ── 3. Find confirmed words in corpus ────────────────────────────
    print("\n  3. Finding confirmed words in decoded corpus ...")

    alignments: List[ConfirmedAlignment] = []
    n_found = 0

    for cw in CONFIRMED_WORDS:
        latin = cw['latin']
        source = cw['source']
        match_type = cw['match_type']

        matching_tokens = _find_tokens_for_word(latin, all_tokens, decoded_cache)

        if matching_tokens:
            n_found += 1
            syllables = _syllabify_latin(latin)

            for tok in matching_tokens[:5]:  # limit to 5 tokens per word
                eva_chars = tokenize_eva_chars(tok)
                char_mapping = _align_chars_to_syllables(
                    eva_chars, syllables, modifier_chars,
                )

                alignment = ConfirmedAlignment(
                    latin_word=latin,
                    source=source,
                    match_type=match_type,
                    eva_token=tok,
                    eva_chars=eva_chars,
                    syllables=syllables,
                    char_to_syllable=char_mapping,
                    n_aligned=len(char_mapping),
                )
                alignments.append(alignment)
                status = "ALIGNED" if char_mapping else "MISMATCH"
                print(f"    {status}: '{latin}' <- '{tok}' "
                      f"(chars={eva_chars}, syls={syllables})")
        else:
            print(f"    NOT FOUND: '{latin}' ({source})")

    print(f"\n    Found {n_found}/{len(CONFIRMED_WORDS)} confirmed words in corpus")
    print(f"    {len(alignments)} total alignments")

    # ── 4. Build character-level table ───────────────────────────────
    print("\n  4. Building character-level partial assignment table ...")

    # Filter to only aligned entries
    aligned = [a for a in alignments if a.n_aligned > 0]
    print(f"    {len(aligned)} alignments with successful char->syllable mapping")

    char_assignments, contradictions = _build_char_table(
        aligned, eva_to_triple, assignment,
    )

    n_consistent = sum(1 for c in char_assignments
                       if len([a for a in aligned
                               if c.eva_char in a.char_to_syllable]) == c.n_sources)
    n_contradictory = len(contradictions)

    print(f"    {len(char_assignments)} characters assigned")
    print(f"    {n_consistent} consistent (same syllable in all occurrences)")
    print(f"    {n_contradictory} contradictions found")

    if char_assignments:
        print("\n    Character-level table:")
        for ca in char_assignments:
            agree_marker = "==" if ca.agrees_with_phase16 else "!="
            print(f"      '{ca.eva_char}' -> '{ca.assigned_syllable}' "
                  f"(x{ca.n_sources}) "
                  f"[Phase16: '{ca.phase16_syllable}' {agree_marker}]")

    if contradictions:
        print("\n    Contradictions:")
        for ctr in contradictions:
            print(f"      '{ctr['eva_char']}': "
                  f"'{ctr['syllable_1']}' ({ctr['source_1']}) vs "
                  f"'{ctr['syllable_2']}' ({ctr['source_2']})")

    # ── 5. Bootstrap decode ──────────────────────────────────────────
    print("\n  5. Bootstrap decoding from partial table ...")

    # Build initial partial table from confirmed alignments
    partial_table: Dict[str, str] = {}
    for ca in char_assignments:
        partial_table[ca.eva_char] = ca.assigned_syllable

    print(f"    Starting with {len(partial_table)} character assignments")

    if partial_table:
        n_iters, n_new, new_words, final_table = _bootstrap_decode(
            partial_table, all_tokens, modifier_chars, eva_to_triple,
            ref_word_set, max_iterations=10,
        )
        print(f"\n    Bootstrap complete: {n_iters} iterations, "
              f"{n_new} new words discovered")
        if new_words:
            print(f"    New words: {new_words[:20]}")
            if len(new_words) > 20:
                print(f"    ... and {len(new_words) - 20} more")
    else:
        n_iters = 0
        n_new = 0
        new_words = []
        final_table = {}
        print("    No partial table available -- skipping bootstrap.")

    # ── 6. Compare with Phase 16 ─────────────────────────────────────
    print("\n  6. Comparing reverse-engineered assignments with Phase 16 ...")

    n_agrees = sum(1 for ca in char_assignments if ca.agrees_with_phase16)
    n_disagrees = sum(1 for ca in char_assignments if not ca.agrees_with_phase16)

    disagreements: List[Dict] = []
    for ca in char_assignments:
        if not ca.agrees_with_phase16:
            disagreements.append({
                'eva_char': ca.eva_char,
                'reverse_eng_syllable': ca.assigned_syllable,
                'phase16_syllable': ca.phase16_syllable,
                'n_sources': ca.n_sources,
                'sources': ca.sources,
            })

    print(f"    Agreements: {n_agrees}/{len(char_assignments)}")
    print(f"    Disagreements: {n_disagrees}/{len(char_assignments)}")

    if disagreements:
        print("\n    Specific disagreements:")
        for d in disagreements:
            print(f"      '{d['eva_char']}': "
                  f"reverse_eng='{d['reverse_eng_syllable']}' vs "
                  f"phase16='{d['phase16_syllable']}' "
                  f"(supported by {d['n_sources']} confirmed words)")

    # ── 7. Verdict ───────────────────────────────────────────────────
    print("\n  7. Verdict ...")

    verdict_parts: List[str] = []
    verdict_parts.append(f"{n_found}/{len(CONFIRMED_WORDS)} confirmed words found in corpus")
    verdict_parts.append(f"{len(char_assignments)} character-level assignments extracted")
    verdict_parts.append(f"{n_contradictory} contradictions")
    verdict_parts.append(f"{n_new} bootstrap words discovered")
    verdict_parts.append(f"Phase 16 agreement: {n_agrees}/{len(char_assignments)}")

    if len(char_assignments) == 0:
        verdict = "INSUFFICIENT DATA: no character-level assignments could be extracted"
    elif n_contradictory > len(char_assignments) // 2:
        verdict = ("INCONSISTENT: more than half of character assignments are contradictory; "
                   "confirmed words may not share a consistent cipher")
    elif n_agrees > n_disagrees:
        verdict = (f"PARTIAL VALIDATION: {n_agrees}/{len(char_assignments)} assignments "
                   f"agree with Phase 16. {n_new} bootstrap words found. "
                   f"Reverse engineering supports Phase 16 table partially.")
    elif n_disagrees > n_agrees and len(char_assignments) >= 3:
        verdict = (f"DIVERGENT: {n_disagrees}/{len(char_assignments)} assignments "
                   f"disagree with Phase 16. Confirmed words suggest different mapping. "
                   f"{n_new} bootstrap words found.")
    else:
        verdict = (f"INCONCLUSIVE: {len(char_assignments)} assignments, "
                   f"{n_agrees} agree, {n_disagrees} disagree with Phase 16. "
                   f"{n_new} bootstrap words found.")

    for part in verdict_parts:
        print(f"    {part}")
    print(f"\n    VERDICT: {verdict}")

    # ── 8. Save results ──────────────────────────────────────────────
    runtime = time.time() - t0

    result = ReverseEngineerResult(
        timestamp=time.strftime('%Y-%m-%dT%H:%M:%S'),
        n_confirmed_words=len(CONFIRMED_WORDS),
        n_found_in_corpus=n_found,
        confirmed_alignments=[_convert(asdict(a)) for a in alignments],
        n_chars_assigned=len(char_assignments),
        n_chars_consistent=n_consistent,
        n_chars_contradictory=n_contradictory,
        char_assignments=[_convert(asdict(ca)) for ca in char_assignments],
        contradictions=contradictions,
        n_bootstrap_iterations=n_iters,
        n_new_words_found=n_new,
        bootstrap_words=new_words,
        n_agrees_with_phase16=n_agrees,
        n_disagrees_with_phase16=n_disagrees,
        disagreements=disagreements,
        verdict=verdict,
        runtime_seconds=round(runtime, 2),
    )

    out_path = os.path.join(rd, 'reverse_engineering.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)

    print(f"\n  Saved: {out_path}")
    print(f"  Runtime: {runtime:.1f}s")
    print("=" * 60)
