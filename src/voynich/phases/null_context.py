"""
Phase 13.6 – Null Hypothesis: No Context Rules Exist
=====================================================
Tests three alternative explanations for the 11.1% dict_hit ceiling,
providing a rigorous baseline against which to evaluate context rules:

  1. Cell conflation — the 14-cell grid forces multiple Latin phonemes into
     single cells, making disambiguation impossible regardless of context.

  2. Dictionary gaps — near-miss tokens are actually correct decodings of
     medieval Latin words not in the reference word set (variant spellings,
     abbreviations, inflected forms).

  3. Random error distribution — errors are distributed uniformly across
     positions and context types (no MI above shuffled baseline).

This module should be run alongside Step 13.1.  If any of these alternatives
explains the near-misses better than context rules, the 11.1% ceiling is
structural and cannot be improved by reading rules alone.
"""

import json
import os
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import build_eva_to_cell_lookup, load_corpus
from voynich.core.reference import (
    MEDIEVAL_LATIN_VARIANTS,
    expand_latin_word_set,
    load_reference_corpus,
)
from voynich.phases.csp_constraints import build_phoneme_inventory
from voynich.phases.csp_solver import _convert, decode_token
from voynich.phases.csp_diagnosis import (
    _edit_distance,
    _bucket_by_length,
    _nearest_word,
    categorize_token,
    _get_cells_used,
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CellConflationResult:
    """Cell conflation analysis for one grid cell."""
    cell_key: str
    cv_label: str
    current_assignment: str
    n_tokens: int
    n_near_miss: int
    target_phonemes: List[str]          # Needed phonemes from near-miss alignments
    n_distinct_needed: int              # How many distinct phonemes this cell "needs"
    disambiguation_possible: bool       # n_distinct_needed <= 2
    conflation_ratio: float             # n_distinct_needed / len(inventory)


@dataclass
class DictionaryExpansionResult:
    """Effect of expanding the Latin word set."""
    original_word_set_size: int
    expanded_word_set_size: int
    n_near_miss_original: int
    n_near_miss_after_expansion: int
    n_converted_to_hit: int
    conversion_rate: float
    new_hits_sample: List[str]
    gate_passed: bool                   # >20% of near-misses become hits


@dataclass
class NullContextResult:
    """Full Phase 13.6 output."""
    cell_conflation: List[Dict]
    n_cells_conflated: int              # cells where n_distinct_needed > 2
    conflation_verdict: str
    dictionary_expansion: Dict
    dict_expansion_verdict: str
    combined_verdict: str
    recommendation: str


# ---------------------------------------------------------------------------
# Test 1: Cell conflation
# ---------------------------------------------------------------------------

def test_cell_conflation(
    best_assignment: Dict[str, str],
    corpus_tokens: List[str],
    ref_word_set: set,
    ref_words_by_len: Dict,
    inventory: Any,
    eva_to_cell: Dict[str, str],
    cv_labels: Dict,
) -> List[CellConflationResult]:
    """For each cell, determine how many distinct target phonemes it needs.

    For each near-miss token, we ask: if this cell had a DIFFERENT value,
    what value would make the decoded string match the target word?
    That 'needed' phoneme is what the cell 'should' produce for this token.
    If a cell needs to produce many different phonemes across different tokens,
    it is conflated (too many phonemes assigned to one cell).
    """
    # Decode and diagnose tokens
    cell_needed: Dict[str, Counter] = defaultdict(Counter)  # cell → {needed_phoneme: count}
    cell_tokens: Dict[str, int] = defaultdict(int)
    cell_near_miss: Dict[str, int] = defaultdict(int)

    for token in corpus_tokens:
        decoded = decode_token(token, best_assignment, eva_to_cell)
        cells_used = _get_cells_used(token, eva_to_cell)
        cat, best_match, best_dist = categorize_token(
            decoded, ref_word_set, ref_words_by_len, inventory,
        )

        for ck in cells_used:
            cell_tokens[ck] += 1
            if cat == 'NEAR_MISS' and best_match:
                cell_near_miss[ck] += 1

        if cat != 'NEAR_MISS' or not best_match:
            continue

        # Simple single-character comparison: try to find needed phoneme
        # by testing each cell with each possible value
        current_syl = best_assignment.get(cells_used[0] if cells_used else '', '')
        # Build character-level span for each cell
        pos = 0
        spans: List[Tuple[str, int, int]] = []
        for ck in cells_used:
            syl = best_assignment.get(ck, '')
            spans.append((ck, pos, pos + len(syl)))
            pos += len(syl)

        # Find which cells' syllables overlap with mismatch positions
        for ci, (ck, start, end) in enumerate(spans):
            current_syl = best_assignment.get(ck, '')
            # Check if this cell's characters align with a mismatch
            dec_seg = decoded[start:end] if end <= len(decoded) else ''
            target_start = start
            target_end = min(end, len(best_match))
            if target_end <= target_start:
                continue
            tgt_seg = best_match[target_start:target_end] if target_end <= len(best_match) else ''
            if dec_seg != tgt_seg and tgt_seg:
                # Record each character that was needed
                for ch in tgt_seg:
                    if ch.isalpha():
                        cell_needed[ck][ch] += 1

    # Build results
    results: List[CellConflationResult] = []
    all_cells = set(cell_tokens.keys()) | set(cell_needed.keys())
    for ck in all_cells:
        if cell_tokens[ck] < 5:
            continue
        needed_phonemes = list(cell_needed[ck].most_common(10))
        n_distinct = len([p for p, c in needed_phonemes if c >= 2])
        cv_label = cv_labels.get(ck, {}).get('cv_label', '?') if isinstance(cv_labels.get(ck), dict) else '?'
        results.append(CellConflationResult(
            cell_key=ck,
            cv_label=cv_label,
            current_assignment=best_assignment.get(ck, '?'),
            n_tokens=cell_tokens[ck],
            n_near_miss=cell_near_miss[ck],
            target_phonemes=[p for p, _ in needed_phonemes],
            n_distinct_needed=n_distinct,
            disambiguation_possible=(n_distinct <= 2),
            conflation_ratio=round(n_distinct / max(len(best_assignment), 1), 3),
        ))

    results.sort(key=lambda r: r.n_distinct_needed, reverse=True)
    return results


# ---------------------------------------------------------------------------
# Test 2: Dictionary expansion
# ---------------------------------------------------------------------------

def test_dictionary_expansion(
    corpus_tokens: List[str],
    best_assignment: Dict[str, str],
    eva_to_cell: Dict[str, str],
    original_ref_word_set: set,
    ref_corpus: Any,
    inventory: Any,
    ref_words_by_len: Dict,
) -> DictionaryExpansionResult:
    """Expand the Latin word set with medieval variants and remeasure dict_hit.

    Tests whether near-misses are actually correct medieval Latin that just
    wasn't in the original reference word set.
    """
    expanded_word_set = expand_latin_word_set(original_ref_word_set)
    print(f"    Original word set: {len(original_ref_word_set)}")
    print(f"    Expanded word set: {len(expanded_word_set)}")

    # Build new ref_words_by_len for expanded set
    expanded_list = sorted(expanded_word_set)
    expanded_by_len = _bucket_by_length(expanded_list, max_per_bucket=100)

    n_near_miss_original = 0
    n_near_miss_after = 0
    n_hit_original = 0
    n_hit_after = 0
    new_hits: List[str] = []

    for token in corpus_tokens:
        decoded = decode_token(token, best_assignment, eva_to_cell)
        if not decoded or '?' in decoded:
            continue

        # Original categorization
        cat_orig, _, _ = categorize_token(decoded, original_ref_word_set, ref_words_by_len, inventory)
        # Expanded categorization
        cat_exp, _, _ = categorize_token(decoded, expanded_word_set, expanded_by_len, inventory)

        if cat_orig == 'NEAR_MISS':
            n_near_miss_original += 1
            if cat_exp == 'HIT':
                n_hit_after += 1
                if len(new_hits) < 20:
                    new_hits.append(decoded)

        if cat_orig == 'HIT':
            n_hit_original += 1
        if cat_exp in ('HIT',):
            n_near_miss_after += sum(1 for _ in [cat_orig] if cat_orig == 'NEAR_MISS')

    conversion_rate = n_hit_after / max(n_near_miss_original, 1)
    gate_passed = conversion_rate > 0.20  # >20% of near-misses become hits = significant gap

    return DictionaryExpansionResult(
        original_word_set_size=len(original_ref_word_set),
        expanded_word_set_size=len(expanded_word_set),
        n_near_miss_original=n_near_miss_original,
        n_near_miss_after_expansion=n_near_miss_original - n_hit_after,
        n_converted_to_hit=n_hit_after,
        conversion_rate=round(conversion_rate, 3),
        new_hits_sample=new_hits[:10],
        gate_passed=gate_passed,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_null_context() -> Dict:
    """Phase 13.6: Test null hypotheses for the 11.1% dict_hit ceiling.

    Tests (1) cell conflation, (2) dictionary gaps, and interprets results.
    Also loads MI selectivity from error_patterns.json if available.
    """
    print("=" * 70)
    print("PHASE 13.6: Null Hypothesis — No Context Rules Exist")
    print("=" * 70)

    t0 = time.time()
    rd = _results_dir()

    # ------------------------------------------------------------------
    # 1. Load CSP assignment
    # ------------------------------------------------------------------
    for fname in ('recalibrated_csp.json', 'csp_final.json', 'csp_decode.json'):
        candidate = os.path.join(rd, fname)
        if os.path.exists(candidate):
            with open(candidate) as f:
                decode_data = json.load(f)
            print(f"  Loaded from {fname}")
            break
    else:
        print("  [SKIP] No CSP result found")
        return {'verdict': 'skipped'}

    if 'best_assignment' in decode_data:
        best_assignment: Dict[str, str] = decode_data['best_assignment']
        eva_to_cell_map: Dict[str, str] = decode_data.get('eva_to_cell_mapping', {})
    elif 'language_results' in decode_data:
        lat = decode_data['language_results'].get('latin', {})
        best_assignment = lat.get('best_assignment', {})
        eva_to_cell_map = decode_data.get('eva_to_cell_mapping', {})
    else:
        return {'verdict': 'skipped', 'reason': 'no_assignment'}

    # ------------------------------------------------------------------
    # 2. Load corpus and reference
    # ------------------------------------------------------------------
    cv_path = os.path.join(rd, 'cv_labels.json')
    with open(cv_path) as f:
        cv_labels = json.load(f)

    corpus = load_corpus(verbose=False)
    ref_corpus = load_reference_corpus(verbose=False)

    corpus_tokens = corpus.get_tokens(language='A', paragraph_only=True)[:1500]

    ref_tokens = ref_corpus.get_combined_tokens('latin')
    ref_word_set: set = set(ref_tokens[:50000])
    ref_words_by_len = _bucket_by_length(ref_tokens[:10000], max_per_bucket=60)

    inventory = build_phoneme_inventory('latin', ref_corpus)

    if eva_to_cell_map:
        eva_to_cell = eva_to_cell_map
    else:
        eva_to_cell = build_eva_to_cell_lookup(cv_labels)

    # ------------------------------------------------------------------
    # 3. Cell conflation test
    # ------------------------------------------------------------------
    print("\n  Test 1: Cell conflation...")
    conflation_results = test_cell_conflation(
        best_assignment, corpus_tokens, ref_word_set,
        ref_words_by_len, inventory, eva_to_cell, cv_labels,
    )
    n_conflated = sum(1 for r in conflation_results if not r.disambiguation_possible)
    print(f"  Cells analyzed: {len(conflation_results)}")
    print(f"  Cells with > 2 distinct needed phonemes (conflated): {n_conflated}")
    for r in conflation_results[:5]:
        print(f"    {r.cv_label} ({r.current_assignment}): needs {r.n_distinct_needed} phonemes "
              f"[{', '.join(r.target_phonemes[:5])}]  "
              f"{'CONFLATED' if not r.disambiguation_possible else 'ok'}")

    if n_conflated > 7:
        conflation_verdict = (
            f"HIGH_CONFLATION: {n_conflated}/14 cells have >2 needed phonemes. "
            "The 14-cell grid may be too coarse to disambiguate via context rules alone."
        )
    elif n_conflated > 3:
        conflation_verdict = (
            f"MODERATE_CONFLATION: {n_conflated}/14 cells conflated. "
            "Context rules may partially disambiguate but grid refinement may also be needed."
        )
    else:
        conflation_verdict = (
            f"LOW_CONFLATION: Only {n_conflated}/14 cells have high ambiguity. "
            "Context rules should be sufficient to resolve most near-misses."
        )
    print(f"\n  Conflation verdict: {conflation_verdict}")

    # ------------------------------------------------------------------
    # 4. Dictionary expansion test
    # ------------------------------------------------------------------
    print("\n  Test 2: Dictionary expansion...")
    dict_expansion = test_dictionary_expansion(
        corpus_tokens, best_assignment, eva_to_cell,
        ref_word_set, ref_corpus, inventory, ref_words_by_len,
    )
    print(f"  Near-miss tokens: {dict_expansion.n_near_miss_original}")
    print(f"  Converted to HIT with expanded dict: {dict_expansion.n_converted_to_hit}")
    print(f"  Conversion rate: {dict_expansion.conversion_rate:.1%}")
    if dict_expansion.new_hits_sample:
        print(f"  Sample new hits: {dict_expansion.new_hits_sample[:5]}")

    if dict_expansion.gate_passed:
        dict_expansion_verdict = (
            f"DICTIONARY_GAPS_SIGNIFICANT: {dict_expansion.conversion_rate:.0%} of near-misses "
            "are medieval Latin variants. Expanding the dictionary should improve dict_hit directly."
        )
    elif dict_expansion.conversion_rate > 0.10:
        dict_expansion_verdict = (
            f"DICTIONARY_GAPS_MODERATE: {dict_expansion.conversion_rate:.0%} of near-misses "
            "are medieval variants. Dictionary expansion is worth pursuing alongside context rules."
        )
    else:
        dict_expansion_verdict = (
            f"DICTIONARY_GAPS_MINOR: Only {dict_expansion.conversion_rate:.0%} of near-misses "
            "are medieval variants. Dictionary gaps are not the main explanation."
        )
    print(f"\n  Dictionary verdict: {dict_expansion_verdict}")

    # ------------------------------------------------------------------
    # 5. Load MI selectivity from error_patterns.json if available
    # ------------------------------------------------------------------
    mi_selectivity = None
    ep_path = os.path.join(rd, 'error_patterns.json')
    if os.path.exists(ep_path):
        with open(ep_path) as f:
            ep_data = json.load(f)
        mi_selectivity = ep_data.get('mi_selectivity')
        print(f"\n  MI selectivity from error_patterns.json: {mi_selectivity}")

    # ------------------------------------------------------------------
    # 6. Combined verdict
    # ------------------------------------------------------------------
    if mi_selectivity is not None and mi_selectivity < 1.0:
        combined_verdict = "CEILING_IS_STRUCTURAL"
        recommendation = (
            "MI < 1.0x: errors are random (no context effects). "
            f"Dictionary expansion covers {dict_expansion.conversion_rate:.0%} of near-misses. "
            f"{n_conflated}/14 cells are conflated. "
            "The 11.1% ceiling is inherent to the 14-cell CV phonotactic model. "
            "Future progress requires finer grid decomposition or a different encoding model."
        )
    elif dict_expansion.gate_passed:
        combined_verdict = "DICTIONARY_EXPANSION_RECOMMENDED"
        recommendation = (
            f"Dictionary expansion converts {dict_expansion.conversion_rate:.0%} of near-misses to hits. "
            "Apply medieval Latin variants to the word set as a first step before context rules."
        )
    elif n_conflated > 7:
        combined_verdict = "GRID_TOO_COARSE"
        recommendation = (
            f"{n_conflated}/14 cells are conflated. "
            "Context rules can partially help but the grid needs finer decomposition "
            "to represent the full Latin phoneme inventory."
        )
    else:
        combined_verdict = "CONTEXT_RULES_VIABLE"
        recommendation = (
            f"Cell conflation is low ({n_conflated}/14), dictionary gaps are minor "
            f"({dict_expansion.conversion_rate:.0%}). "
            "Context-dependent reading rules are the most promising remaining approach."
        )

    print(f"\n  Combined verdict: {combined_verdict}")
    print(f"  Recommendation: {recommendation}")

    # ------------------------------------------------------------------
    # 7. Save results
    # ------------------------------------------------------------------
    result = NullContextResult(
        cell_conflation=[_convert(asdict(r)) for r in conflation_results],
        n_cells_conflated=n_conflated,
        conflation_verdict=conflation_verdict,
        dictionary_expansion=_convert(asdict(dict_expansion)),
        dict_expansion_verdict=dict_expansion_verdict,
        combined_verdict=combined_verdict,
        recommendation=recommendation,
    )

    out_path = os.path.join(rd, 'null_context.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2)

    elapsed = time.time() - t0
    print(f"\n  Saved to {out_path} ({elapsed:.1f}s)")
    return _convert(asdict(result))
