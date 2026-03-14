"""
Phase 53 Track A: Extract Per-Triple Constraints from Paradigms
===============================================================
For each of the 56 morphological paradigms (Phase 52), cross-reference
word catalog entries to extract what each free triple MUST produce in
each identified word. Aggregate constraints per triple and check for
convergent consensus.

Dependency chain:
    word_catalog.json          (Phase 52 Track A)
    word_validation.json       (Phase 52 Track B)
    combined_refine.json       (Phase 15)
    bootstrap_loop.json        (Phase 30)
    modifier_integrate.json    (Phase 16)
    signal_bigrams.json        (Phase 29)
        -> paradigm_constraints.json (this step)
"""

from __future__ import annotations

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
    tokenize_eva_chars,
)
from voynich.phases.suffix_calibration import SIGNAL_WORDS_51, SIGNAL_WORDS_SET


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
    if isinstance(obj, set):
        return sorted(_convert(item) for item in obj)
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
# Step A.1: Build paradigm-to-catalog cross-reference
# ---------------------------------------------------------------------------

def _build_paradigm_catalog_index(
    paradigms: List[Dict],
    catalog: List[Dict],
) -> Dict[str, List[Dict]]:
    """Index catalog entries by EVA type for fast lookup."""
    eva_index: Dict[str, List[Dict]] = defaultdict(list)
    for entry in catalog:
        eva_index[entry['eva_type']].append(entry)
    return eva_index


def _match_paradigm_to_catalog(
    paradigm: Dict,
    eva_index: Dict[str, List[Dict]],
    tier_filter: Optional[Set[str]] = None,
) -> List[Dict]:
    """Find catalog entries matching a paradigm's (eva_type, stem) pairs.

    Returns list of catalog entries where:
    - eva_type is in the paradigm's eva_types
    - latin_word starts with the paradigm's stem
    """
    stem = paradigm['stem']
    matched = []
    seen = set()

    for eva_type in paradigm['eva_types']:
        for entry in eva_index.get(eva_type, []):
            if tier_filter and entry.get('tier') not in tier_filter:
                continue
            if entry['latin_word'].startswith(stem):
                key = (entry['eva_type'], entry['latin_word'])
                if key not in seen:
                    seen.add(key)
                    matched.append(entry)

    return matched


# ---------------------------------------------------------------------------
# Step A.2: Extract implied assignments from catalog entries
# ---------------------------------------------------------------------------

def _extract_catalog_constraints(
    matched_entries: List[Dict],
    paradigm_stem: str,
) -> List[Dict]:
    """Extract constraints from catalog entries that have implied_assignments."""
    constraints = []
    for entry in matched_entries:
        implied = entry.get('implied_assignments', {})
        if not implied:
            continue
        for triple_key, implied_value in implied.items():
            constraints.append({
                'triple': triple_key,
                'implied_value': implied_value,
                'from_word': entry['latin_word'],
                'from_eva_type': entry['eva_type'],
                'paradigm_stem': paradigm_stem,
                'tier': entry.get('tier', '?'),
                'confidence': entry.get('confidence', 0.0),
                'source': 'catalog',
                'n_shared': 1,  # catalog only stores unambiguous cases
            })
    return constraints


# ---------------------------------------------------------------------------
# Step A.3: Fallback alignment for entries without implied_assignments
# ---------------------------------------------------------------------------

def _alignment_constraints(
    matched_entries: List[Dict],
    paradigm_stem: str,
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    modifier_chars: Set[str],
    confirmed_triples: Set[str],
) -> Tuple[List[Dict], int]:
    """Try to extract constraints via greedy alignment for entries
    that don't have implied_assignments in the catalog.

    Returns (constraints, n_failures).
    """
    constraints = []
    n_failures = 0

    for entry in matched_entries:
        # Skip if catalog already has implied_assignments
        if entry.get('implied_assignments'):
            continue

        eva_type = entry['eva_type']
        latin_word = entry['latin_word']

        # Tokenize EVA into chars, strip modifiers, map to triples
        chars = tokenize_eva_chars(eva_type)
        char_info = []  # (char, triple_key, is_confirmed, syllable)
        for ch in chars:
            if ch in modifier_chars:
                continue
            triple = eva_to_triple.get(ch)
            if triple is None:
                char_info.append((ch, None, False, None))
            elif triple in confirmed_triples:
                syl = assignment.get(triple, '')
                char_info.append((ch, triple, True, syl))
            else:
                char_info.append((ch, triple, False, None))

        if not char_info:
            continue

        # Build lists of confirmed and free positions
        confirmed_positions = [(i, ci[3]) for i, ci in enumerate(char_info)
                               if ci[2] and ci[3]]
        free_positions = [(i, ci[1]) for i, ci in enumerate(char_info)
                          if not ci[2] and ci[1] is not None]

        if not confirmed_positions or not free_positions:
            continue

        # Greedy left-to-right alignment of confirmed syllables in latin_word
        alignments = []  # (char_idx, word_start, word_end)
        search_from = 0
        failed = False

        for char_idx, syl in confirmed_positions:
            pos = latin_word.find(syl, search_from)
            if pos == -1:
                failed = True
                break
            alignments.append((char_idx, pos, pos + len(syl)))
            search_from = pos + len(syl)

        if failed:
            n_failures += 1
            continue

        # Extract gaps for free triples
        # Only keep constraints where exactly ONE free triple fills a gap
        for free_idx, free_triple in free_positions:
            # Find the preceding and following confirmed alignment
            prev_end = 0
            next_start = len(latin_word)

            for ci, ws, we in alignments:
                if ci < free_idx:
                    prev_end = max(prev_end, we)
                elif ci > free_idx:
                    next_start = min(next_start, ws)
                    break

            # Count how many free triples share this gap
            n_free_in_gap = 0
            for fi, _ in free_positions:
                # A free triple is in this gap if it's between same anchors
                fi_prev = 0
                fi_next = len(latin_word)
                for ci, ws, we in alignments:
                    if ci < fi:
                        fi_prev = max(fi_prev, we)
                    elif ci > fi:
                        fi_next = min(fi_next, ws)
                        break
                if fi_prev == prev_end and fi_next == next_start:
                    n_free_in_gap += 1

            if n_free_in_gap != 1:
                continue  # Ambiguous — multiple free triples share this gap

            gap = latin_word[prev_end:next_start]
            if not gap or len(gap) > 4:
                continue

            constraints.append({
                'triple': free_triple,
                'implied_value': gap,
                'from_word': latin_word,
                'from_eva_type': entry['eva_type'],
                'paradigm_stem': paradigm_stem,
                'tier': entry.get('tier', '?'),
                'confidence': entry.get('confidence', 0.0),
                'source': 'alignment',
                'n_shared': 1,
            })

    return constraints, n_failures


# ---------------------------------------------------------------------------
# Step A.4: Aggregate constraints per triple
# ---------------------------------------------------------------------------

def _aggregate_constraints(
    all_constraints: List[Dict],
    assignment: Dict[str, str],
) -> Dict[str, Dict]:
    """Aggregate constraints per free triple and compute consensus."""
    triple_groups: Dict[str, List[Dict]] = defaultdict(list)
    for c in all_constraints:
        triple_groups[c['triple']].append(c)

    summaries = {}
    for triple_key, constraints in sorted(triple_groups.items()):
        values = [c['implied_value'] for c in constraints]
        value_counts = Counter(values)
        total = len(values)
        top_value, top_count = value_counts.most_common(1)[0]
        consensus = top_count / total if total > 0 else 0.0

        paradigm_stems = set(c['paradigm_stem'] for c in constraints)
        tiers = Counter(c['tier'] for c in constraints)

        # Separate by source
        catalog_constraints = [c for c in constraints if c['source'] == 'catalog']
        alignment_constraints = [c for c in constraints if c['source'] == 'alignment']

        summaries[triple_key] = {
            'current_assignment': assignment.get(triple_key, '?'),
            'n_unique_constraints': total,
            'n_catalog_constraints': len(catalog_constraints),
            'n_alignment_constraints': len(alignment_constraints),
            'n_paradigms': len(paradigm_stems),
            'paradigm_stems': sorted(paradigm_stems),
            'top_implied_value': top_value,
            'consensus': round(consensus, 4),
            'all_values': dict(value_counts.most_common()),
            'tier_distribution': dict(tiers),
        }

    return summaries


# ---------------------------------------------------------------------------
# Step A.5: Signal word safety check
# ---------------------------------------------------------------------------

def _check_signal_word_safety(
    triple_key: str,
    new_value: str,
    assignment: Dict[str, str],
    token_evas: List[str],
    token_decoded: List[str],
    eva_to_triple: Dict[str, str],
    modifier_chars: Set[str],
) -> Tuple[bool, int]:
    """Check if changing a triple's assignment breaks any signal words.

    Compares decode output of original vs modified assignment for all
    tokens that currently decode to signal words. If ANY token changes
    its decoded output, the correction is unsafe.

    Returns (is_safe, n_signal_words_verified).
    """
    modified = dict(assignment)
    modified[triple_key] = new_value

    # Find unique EVA types that produce signal words
    signal_eva_types: Set[str] = set()
    for eva, decoded in zip(token_evas, token_decoded):
        if decoded in SIGNAL_WORDS_SET:
            signal_eva_types.add(eva)

    n_verified = 0
    for eva_type in signal_eva_types:
        # Compare original vs modified decode
        original_decode = decode_token_modifier_aware(
            eva_type, assignment, eva_to_triple, modifier_chars,
        )
        modified_decode = decode_token_modifier_aware(
            eva_type, modified, eva_to_triple, modifier_chars,
        )

        if original_decode != modified_decode:
            return False, n_verified

        n_verified += 1

    return True, n_verified


# ---------------------------------------------------------------------------
# Step A.6: Apply acceptance criteria
# ---------------------------------------------------------------------------

def _evaluate_recommendations(
    summaries: Dict[str, Dict],
    assignment: Dict[str, str],
    token_evas: List[str],
    token_decoded: List[str],
    eva_to_triple: Dict[str, str],
    modifier_chars: Set[str],
) -> Tuple[Dict[str, Dict], List[Dict]]:
    """Evaluate each triple summary and produce recommendations.

    Returns (updated_summaries, accepted_corrections).
    """
    accepted = []

    for triple_key, summary in summaries.items():
        n_obs = summary['n_unique_constraints']
        n_paradigms = summary['n_paradigms']
        consensus = summary['consensus']
        top_value = summary['top_implied_value']

        # Gate 1: sufficient evidence
        if n_obs < 3 or n_paradigms < 2:
            summary['recommendation'] = 'INSUFFICIENT'
            summary['signal_word_safe'] = None
            continue

        # Gate 2: consensus
        if consensus <= 0.5:
            summary['recommendation'] = 'NO_CONSENSUS'
            summary['signal_word_safe'] = None
            continue

        # Gate 3: plausible value length
        if len(top_value) < 1 or len(top_value) > 4:
            summary['recommendation'] = 'IMPLAUSIBLE_LENGTH'
            summary['signal_word_safe'] = None
            continue

        # Gate 4: actually different from current
        if top_value == summary['current_assignment']:
            summary['recommendation'] = 'ALREADY_ASSIGNED'
            summary['signal_word_safe'] = True
            continue

        # Gate 5: signal word safety
        is_safe, n_checked = _check_signal_word_safety(
            triple_key, top_value, assignment,
            token_evas, token_decoded, eva_to_triple, modifier_chars,
        )
        summary['signal_word_safe'] = is_safe
        summary['n_signal_words_checked'] = n_checked

        if not is_safe:
            summary['recommendation'] = 'BREAKS_SIGNAL_WORDS'
            continue

        # All gates passed
        summary['recommendation'] = 'ACCEPT'
        accepted.append({
            'triple': triple_key,
            'old_value': summary['current_assignment'],
            'new_value': top_value,
            'consensus': consensus,
            'n_paradigms': n_paradigms,
            'n_observations': n_obs,
        })

    return summaries, accepted


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_paradigm_constraints() -> None:
    """Phase 53 Track A: Extract per-triple constraints from paradigms."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE 53 TRACK A: Paradigm-Constrained Triple Extraction")
    print("=" * 70)

    rd = _results_dir()

    # ── Load data ─────────────────────────────────────────────────────
    print("\n  A.1  Loading data...")

    catalog_data = _safe_load(os.path.join(rd, 'word_catalog.json'))
    if not catalog_data:
        print("  *** word_catalog.json not found — run Phase 52 Track A first ***")
        return
    catalog = catalog_data.get('single_token_ids', [])

    validation_data = _safe_load(os.path.join(rd, 'word_validation.json'))
    if not validation_data:
        print("  *** word_validation.json not found — run Phase 52 Track B first ***")
        return
    paradigms = validation_data.get('paradigms', [])

    with open(os.path.join(rd, 'combined_refine.json')) as f:
        assignment = json.load(f)['best_assignment']

    with open(os.path.join(rd, 'bootstrap_loop.json')) as f:
        boot_data = json.load(f)
    confirmed_triples = set(boot_data.get('confirmed_triples', []))

    with open(os.path.join(rd, 'modifier_integrate.json')) as f:
        mod_data = json.load(f)
    modifier_chars = set(mod_data.get('modifier_chars', []))

    with open(os.path.join(rd, 'signal_bigrams.json')) as f:
        bigram_data = json.load(f)
    token_evas = bigram_data['token_evas']
    token_decoded = bigram_data['token_decoded']

    eva_to_triple = build_eva_to_triple_lookup()

    # Identify free triples
    free_triples = set(assignment.keys()) - confirmed_triples
    print(f"       {len(catalog)} catalog entries, {len(paradigms)} paradigms")
    print(f"       {len(confirmed_triples)} confirmed triples, "
          f"{len(free_triples)} free triples")
    print(f"       {len(SIGNAL_WORDS_SET)} signal words to protect")

    # ── Filter paradigms ──────────────────────────────────────────────
    print("\n  A.2  Filtering paradigms...")

    filtered = [p for p in paradigms
                if len(p['stem']) >= 4 and len(set(p['eva_types'])) >= 2]
    print(f"       {len(filtered)} paradigms pass filters "
          f"(stem >= 4, >= 2 EVA types)")

    # ── Build paradigm-to-catalog index ───────────────────────────────
    print("\n  A.3  Cross-referencing paradigms with catalog...")

    eva_index = _build_paradigm_catalog_index(paradigms, catalog)

    all_constraints: List[Dict] = []
    total_alignment_failures = 0
    paradigms_with_constraints = 0

    # T2-primary analysis (since no T1 members exist in paradigms)
    for paradigm in filtered:
        # T2 analysis
        matched_t2 = _match_paradigm_to_catalog(
            paradigm, eva_index, tier_filter={'T1', 'T2'},
        )
        # T2+T3 analysis
        matched_all = _match_paradigm_to_catalog(
            paradigm, eva_index, tier_filter={'T1', 'T2', 'T3'},
        )

        # Extract constraints from catalog implied_assignments
        cat_constraints = _extract_catalog_constraints(matched_t2, paradigm['stem'])

        # Fallback alignment for entries without implied_assignments
        align_constraints, n_fail = _alignment_constraints(
            matched_all, paradigm['stem'],
            assignment, eva_to_triple, modifier_chars, confirmed_triples,
        )
        total_alignment_failures += n_fail

        paradigm_constraints = cat_constraints + align_constraints
        if paradigm_constraints:
            paradigms_with_constraints += 1
        all_constraints.extend(paradigm_constraints)

    print(f"       Total constraints: {len(all_constraints)}")
    print(f"       Paradigms with constraints: {paradigms_with_constraints}")
    print(f"       Alignment failures: {total_alignment_failures}")

    # ── Aggregate per triple ──────────────────────────────────────────
    print("\n  A.4  Aggregating constraints per free triple...")

    summaries = _aggregate_constraints(all_constraints, assignment)
    print(f"       Triples with constraints: {len(summaries)}")

    for triple_key, summary in sorted(summaries.items()):
        top = summary['top_implied_value']
        cons = summary['consensus']
        n = summary['n_unique_constraints']
        np_ = summary['n_paradigms']
        cur = summary['current_assignment']
        print(f"         {triple_key}: {cur} -> {top} "
              f"(consensus={cons:.2f}, n={n}, paradigms={np_})")

    # ── Signal word safety + acceptance ────────────────────────────────
    print("\n  A.5  Evaluating recommendations...")

    summaries, accepted_corrections = _evaluate_recommendations(
        summaries, assignment,
        token_evas, token_decoded, eva_to_triple, modifier_chars,
    )

    for triple_key, summary in sorted(summaries.items()):
        rec = summary['recommendation']
        print(f"         {triple_key}: {rec}")

    print(f"\n       Accepted corrections: {len(accepted_corrections)}")
    for corr in accepted_corrections:
        print(f"         {corr['triple']}: {corr['old_value']} -> "
              f"{corr['new_value']} (consensus={corr['consensus']:.2f}, "
              f"paradigms={corr['n_paradigms']})")

    # ── Value length distribution ─────────────────────────────────────
    length_dist = Counter(len(c['implied_value']) for c in all_constraints)

    # ── Save ──────────────────────────────────────────────────────────
    runtime = round(time.time() - t0, 2)

    result = {
        'paradigms_analyzed': len(filtered),
        'paradigms_with_constraints': paradigms_with_constraints,
        'total_constraints': len(all_constraints),
        'unique_constraints': len([c for c in all_constraints
                                   if c['n_shared'] == 1]),
        'catalog_constraints': len([c for c in all_constraints
                                    if c['source'] == 'catalog']),
        'alignment_constraints': len([c for c in all_constraints
                                      if c['source'] == 'alignment']),
        'alignment_failures': total_alignment_failures,
        'value_length_distribution': dict(length_dist),
        'per_triple_summary': summaries,
        'accepted_corrections': accepted_corrections,
        'n_free_triples': len(free_triples),
        'n_triples_with_constraints': len(summaries),
        'runtime_seconds': runtime,
    }

    out_path = _save_json(rd, 'paradigm_constraints.json', result)
    print(f"\n  Saved -> {out_path}")
    print(f"  Completed in {runtime:.1f}s")
