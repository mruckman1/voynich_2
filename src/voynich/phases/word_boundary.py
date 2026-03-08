"""
Step 24.8 – Word Boundary Re-Analysis
======================================
Test whether EVA "word" boundaries (spaces in the manuscript) are
actually word boundaries, or whether some represent syllable boundaries
within longer words.

Dependency chain:
    combined_refine.json (Phase 15)
    modifier_integrate.json (Phase 16)
        → word_boundary.json (this step)
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
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ConcatenationTest:
    n_adjacent_pairs: int
    n_concatenation_matches: int
    concatenation_rate: float
    n_neither_individual_match: int   # pairs where neither word alone matches dict
    n_both_miss_but_concat_hits: int  # the key signal
    example_concatenations: List[Dict]  # [{word1, word2, concatenated, latin_word}]


@dataclass
class NullBaseline:
    n_random_pairs: int
    n_null_matches: int
    null_rate: float
    selectivity: float  # concatenation_rate / null_rate


@dataclass
class SplitTest:
    n_long_tokens: int   # tokens > 4 chars that miss dict
    n_splittable: int
    split_rate: float
    example_splits: List[Dict]  # [{token, split1, split2}]


@dataclass
class LineBreakAnalysis:
    n_line_breaks: int
    n_continuation_matches: int
    continuation_rate: float
    within_line_rate: float
    cross_line_enrichment: float  # continuation_rate / within_line_rate


@dataclass
class WordBoundaryResult:
    timestamp: str
    n_tokens_decoded: int
    n_lines: int
    concatenation: Dict
    null_baseline: Dict
    split_test: Dict
    line_break: Dict
    # Verdict
    boundaries_are_word_boundaries: bool  # True if concat rate < 2x null
    estimated_false_boundary_rate: float  # (concat - null) / concat if significant
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _decode_token_r3(
    token: str,
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    modifier_chars: Set[str],
    modifier_rules: Dict[str, str],
    ref_word_set: set,
) -> str:
    """R3 combined decode: alteration -> stripping -> original."""
    # Try alteration
    alt = decode_token_modifier_aware(
        token, assignment, eva_to_triple, modifier_chars,
        modifier_rules=modifier_rules,
    )
    if alt.lower() in ref_word_set:
        return alt.lower()

    # Try stripping
    stripped = decode_token_modifier_aware(
        token, assignment, eva_to_triple, modifier_chars,
    )
    if stripped.lower() in ref_word_set:
        return stripped.lower()

    # Fall back to original decoding
    original = decode_token(token, assignment, eva_to_triple)
    return original.lower()


def _extract_lines_from_corpus(corpus) -> List[List[str]]:
    """Extract token lists per line from the corpus.

    The corpus uses VoynichPage objects with .loci (one per line/label).
    If that structure is unavailable, falls back to chunking tokens into
    groups of ~10.
    """
    lines: List[List[str]] = []
    try:
        for page in corpus.pages.values():
            for locus in page.loci:
                text = locus.clean_text
                if text and text.strip():
                    tokens = text.strip().split()
                    if tokens:
                        lines.append(tokens)
    except (AttributeError, TypeError):
        # Fallback: chunk get_tokens() into groups of 10
        all_tokens = corpus.get_tokens()
        chunk_size = 10
        for i in range(0, len(all_tokens), chunk_size):
            chunk = all_tokens[i:i + chunk_size]
            if chunk:
                lines.append(chunk)
    return lines


# ---------------------------------------------------------------------------
# Analysis steps
# ---------------------------------------------------------------------------

def _run_concatenation_test(
    decoded_lines: List[List[str]],
    ref_word_set: set,
    max_examples: int = 20,
) -> ConcatenationTest:
    """Test adjacent decoded words: does concatenation hit dictionary?"""
    n_pairs = 0
    n_matches = 0
    n_neither = 0
    n_both_miss_concat_hit = 0
    examples: List[Dict] = []

    for line in decoded_lines:
        for i in range(len(line) - 1):
            w1 = line[i]
            w2 = line[i + 1]
            concat = w1 + w2
            n_pairs += 1

            w1_hit = w1 in ref_word_set
            w2_hit = w2 in ref_word_set
            concat_hit = concat in ref_word_set

            if concat_hit:
                n_matches += 1
                if len(examples) < max_examples:
                    examples.append({
                        'word1': w1,
                        'word2': w2,
                        'concatenated': concat,
                        'latin_word': concat,
                    })

            if not w1_hit and not w2_hit:
                n_neither += 1
                if concat_hit:
                    n_both_miss_concat_hit += 1

    rate = n_matches / n_pairs if n_pairs > 0 else 0.0

    return ConcatenationTest(
        n_adjacent_pairs=n_pairs,
        n_concatenation_matches=n_matches,
        concatenation_rate=round(rate, 6),
        n_neither_individual_match=n_neither,
        n_both_miss_but_concat_hits=n_both_miss_concat_hit,
        example_concatenations=examples,
    )


def _run_null_baseline(
    decoded_lines: List[List[str]],
    ref_word_set: set,
    concat_rate: float,
    n_random: int = 10_000,
) -> NullBaseline:
    """Null test: randomly pair decoded words from different lines."""
    rng = random.Random(42)

    # Flatten all decoded words with their line index
    words_by_line: List[Tuple[int, str]] = []
    for li, line in enumerate(decoded_lines):
        for w in line:
            words_by_line.append((li, w))

    if len(words_by_line) < 2:
        return NullBaseline(
            n_random_pairs=0,
            n_null_matches=0,
            null_rate=0.0,
            selectivity=0.0,
        )

    n_null_matches = 0
    n_tested = 0

    for _ in range(n_random):
        idx1 = rng.randrange(len(words_by_line))
        idx2 = rng.randrange(len(words_by_line))
        line1, w1 = words_by_line[idx1]
        line2, w2 = words_by_line[idx2]

        # Ensure different lines
        if line1 == line2:
            continue

        concat = w1 + w2
        n_tested += 1
        if concat in ref_word_set:
            n_null_matches += 1

    null_rate = n_null_matches / n_tested if n_tested > 0 else 0.0
    selectivity = concat_rate / null_rate if null_rate > 0 else float('inf')

    return NullBaseline(
        n_random_pairs=n_tested,
        n_null_matches=n_null_matches,
        null_rate=round(null_rate, 6),
        selectivity=round(selectivity, 4),
    )


def _run_split_test(
    decoded_lines: List[List[str]],
    ref_word_set: set,
    min_token_len: int = 5,
    max_examples: int = 20,
) -> SplitTest:
    """For decoded tokens that miss the dictionary and are > 4 chars,
    try splitting at every position and check if both halves hit."""
    # Collect all decoded tokens that miss dict and are long enough
    long_misses: List[str] = []
    seen: Set[str] = set()
    for line in decoded_lines:
        for w in line:
            if w not in ref_word_set and len(w) >= min_token_len and w not in seen:
                long_misses.append(w)
                seen.add(w)

    n_splittable = 0
    examples: List[Dict] = []

    for token in long_misses:
        found_split = False
        for pos in range(2, len(token) - 1):
            left = token[:pos]
            right = token[pos:]
            if left in ref_word_set and right in ref_word_set:
                if not found_split:
                    n_splittable += 1
                    found_split = True
                if len(examples) < max_examples:
                    examples.append({
                        'token': token,
                        'split1': left,
                        'split2': right,
                    })
                break  # one valid split is enough per token

    split_rate = n_splittable / len(long_misses) if long_misses else 0.0

    return SplitTest(
        n_long_tokens=len(long_misses),
        n_splittable=n_splittable,
        split_rate=round(split_rate, 6),
        example_splits=examples,
    )


def _run_line_break_analysis(
    decoded_lines: List[List[str]],
    ref_word_set: set,
    within_line_rate: float,
) -> LineBreakAnalysis:
    """At line boundaries, test if last-word + first-word concatenation
    hits the dictionary more often than within-line pairs."""
    n_breaks = 0
    n_continuation = 0

    for i in range(len(decoded_lines) - 1):
        line_curr = decoded_lines[i]
        line_next = decoded_lines[i + 1]
        if not line_curr or not line_next:
            continue

        last_word = line_curr[-1]
        first_word = line_next[0]
        concat = last_word + first_word
        n_breaks += 1
        if concat in ref_word_set:
            n_continuation += 1

    continuation_rate = n_continuation / n_breaks if n_breaks > 0 else 0.0
    enrichment = (continuation_rate / within_line_rate
                  if within_line_rate > 0 else float('inf'))

    return LineBreakAnalysis(
        n_line_breaks=n_breaks,
        n_continuation_matches=n_continuation,
        continuation_rate=round(continuation_rate, 6),
        within_line_rate=round(within_line_rate, 6),
        cross_line_enrichment=round(enrichment, 4),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_word_boundary() -> None:
    """Step 24.8: Word Boundary Re-Analysis."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 24.8: Word Boundary Re-Analysis")
    print("=" * 70)

    rd = _results_dir()

    # ─── Load Phase 15 best assignment ───
    print("\n  1. Loading Phase 15 assignment (combined_refine.json) ...")
    refine_path = os.path.join(rd, 'combined_refine.json')
    if not os.path.exists(refine_path):
        print("  [SKIP] combined_refine.json not found — run combined-refine first")
        return
    with open(refine_path) as f:
        refine_data = json.load(f)
    assignment = refine_data.get('best_assignment', {})
    print(f"      Loaded assignment with {len(assignment)} mappings.")

    # ─── Load Phase 16 modifier info ───
    print("\n  2. Loading Phase 16 modifier data (modifier_integrate.json) ...")
    mod_path = os.path.join(rd, 'modifier_integrate.json')
    if not os.path.exists(mod_path):
        print("  [SKIP] modifier_integrate.json not found — run mod-integrate first")
        return
    with open(mod_path) as f:
        mod_data = json.load(f)
    modifier_chars = set(mod_data.get('modifier_chars', []))
    modifier_rules: Dict[str, str] = {}
    for c in mod_data.get('classifications', []):
        if c.get('final_classification') == 'modifier':
            modifier_rules[c['eva_char']] = c.get('modifier_type', 'silent')
    print(f"      {len(modifier_chars)} modifier chars, "
          f"{len(modifier_rules)} modifier rules.")

    # ─── Load corpus ───
    print("\n  3. Loading corpus ...")
    corpus = load_corpus(verbose=False)
    eva_to_triple = build_eva_to_triple_lookup()
    raw_lines = _extract_lines_from_corpus(corpus)
    n_lines = len(raw_lines)
    n_tokens_total = sum(len(line) for line in raw_lines)
    print(f"      {n_lines} lines, {n_tokens_total} tokens.")

    # ─── Build reference word set ───
    print("\n  4. Building expanded reference word set ...")
    try:
        ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
        base_words = set(
            w.lower() for w in ref_corpus.get_combined_tokens('latin')
            if len(w) >= 2
        )
    except (FileNotFoundError, KeyError):
        base_words = set()
    expanded_words, _ = build_expanded_word_set(base_words)
    ref_word_set = base_words | expanded_words
    print(f"      {len(ref_word_set)} words in reference set.")

    # ─── Decode corpus line by line using R3 combined strategy ───
    print("\n  5. Decoding corpus line-by-line (R3 combined) ...")
    decoded_lines: List[List[str]] = []
    n_decoded = 0
    for li, raw_line in enumerate(raw_lines):
        decoded_line: List[str] = []
        for token in raw_line:
            decoded = _decode_token_r3(
                token, assignment, eva_to_triple,
                modifier_chars, modifier_rules, ref_word_set,
            )
            decoded_line.append(decoded)
            n_decoded += 1
        if decoded_line:
            decoded_lines.append(decoded_line)
        if (li + 1) % 500 == 0:
            print(f"      ... decoded {li + 1}/{n_lines} lines "
                  f"({n_decoded} tokens)")
    print(f"      Done: {n_decoded} tokens across {len(decoded_lines)} lines.")

    # ─── Step A: Concatenation test ───
    print("\n  6. Concatenation test (adjacent within-line pairs) ...")
    concat_result = _run_concatenation_test(decoded_lines, ref_word_set)
    print(f"      {concat_result.n_adjacent_pairs} pairs, "
          f"{concat_result.n_concatenation_matches} concat hits "
          f"({concat_result.concatenation_rate:.4%})")
    print(f"      {concat_result.n_neither_individual_match} pairs "
          f"where neither individual word hits dict")
    print(f"      {concat_result.n_both_miss_but_concat_hits} pairs "
          f"where both miss but concat hits (key signal)")
    if concat_result.example_concatenations:
        print(f"      Examples:")
        for ex in concat_result.example_concatenations[:5]:
            print(f"        {ex['word1']} + {ex['word2']} = {ex['concatenated']}")

    # ─── Step B: Null baseline ───
    print("\n  7. Null baseline (random cross-line pairs) ...")
    null_result = _run_null_baseline(
        decoded_lines, ref_word_set, concat_result.concatenation_rate,
    )
    print(f"      {null_result.n_random_pairs} random pairs tested, "
          f"{null_result.n_null_matches} null matches "
          f"({null_result.null_rate:.4%})")
    print(f"      Selectivity: {null_result.selectivity:.2f}x "
          f"(concat / null)")

    # ─── Step C: Split test ───
    print("\n  8. Split test (long dict-miss tokens) ...")
    split_result = _run_split_test(decoded_lines, ref_word_set)
    print(f"      {split_result.n_long_tokens} long tokens miss dict, "
          f"{split_result.n_splittable} have valid splits "
          f"({split_result.split_rate:.4%})")
    if split_result.example_splits:
        print(f"      Examples:")
        for ex in split_result.example_splits[:5]:
            print(f"        {ex['token']} -> {ex['split1']} + {ex['split2']}")

    # ─── Step D: Line-break analysis ───
    print("\n  9. Line-break analysis (cross-line concatenation) ...")
    lb_result = _run_line_break_analysis(
        decoded_lines, ref_word_set, concat_result.concatenation_rate,
    )
    print(f"      {lb_result.n_line_breaks} line breaks, "
          f"{lb_result.n_continuation_matches} continuation matches "
          f"({lb_result.continuation_rate:.4%})")
    print(f"      Within-line rate: {lb_result.within_line_rate:.4%}")
    print(f"      Cross-line enrichment: {lb_result.cross_line_enrichment:.2f}x")

    # ─── Verdict ───
    print("\n  10. Verdict ...")
    concat_rate = concat_result.concatenation_rate
    null_rate = null_result.null_rate

    # Boundaries are true word boundaries if concatenation rate is < 2x null
    boundaries_are_words = (
        null_result.selectivity < 2.0
        if null_rate > 0 else concat_rate < 0.01
    )

    # Estimated false boundary rate
    if concat_rate > 0 and concat_rate > null_rate:
        false_boundary_rate = (concat_rate - null_rate) / concat_rate
    else:
        false_boundary_rate = 0.0

    parts: List[str] = []
    if boundaries_are_words:
        parts.append(
            f"WORD BOUNDARIES CONFIRMED: concat selectivity "
            f"{null_result.selectivity:.2f}x (< 2.0x threshold). "
            f"EVA spaces are genuine word boundaries."
        )
    else:
        parts.append(
            f"POSSIBLE SYLLABLE BOUNDARIES: concat selectivity "
            f"{null_result.selectivity:.2f}x (>= 2.0x threshold). "
            f"Some EVA spaces may mark syllable rather than word boundaries."
        )

    parts.append(
        f"Split test: {split_result.n_splittable}/{split_result.n_long_tokens} "
        f"long tokens ({split_result.split_rate:.1%}) have valid splits."
    )

    if lb_result.cross_line_enrichment > 1.5:
        parts.append(
            f"Line-break enrichment {lb_result.cross_line_enrichment:.2f}x "
            f"suggests some words continue across line breaks."
        )
    else:
        parts.append(
            f"Line-break enrichment {lb_result.cross_line_enrichment:.2f}x — "
            f"no strong evidence of cross-line word continuation."
        )

    parts.append(
        f"Both-miss-but-concat-hits: {concat_result.n_both_miss_but_concat_hits} "
        f"(strongest signal of false boundaries)."
    )

    verdict = " ".join(parts)
    print(f"\n      {verdict}")

    # ─── Build result ───
    result = WordBoundaryResult(
        timestamp=time.strftime('%Y-%m-%dT%H:%M:%S'),
        n_tokens_decoded=n_decoded,
        n_lines=len(decoded_lines),
        concatenation=_convert(concat_result),
        null_baseline=_convert(null_result),
        split_test=_convert(split_result),
        line_break=_convert(lb_result),
        boundaries_are_word_boundaries=boundaries_are_words,
        estimated_false_boundary_rate=round(false_boundary_rate, 6),
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    # ─── Save ───
    out_path = os.path.join(rd, 'word_boundary.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)
    print(f"\n  Saved → {out_path}")
    print(f"  Runtime: {result.runtime_seconds:.1f}s")
