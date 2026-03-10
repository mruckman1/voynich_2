"""
Step 39.2 – Vowel Error Map
============================
Consolidate triple-level corrections from the CC bigram decomposition,
cross-referenced with consonant class structure from Phase 37.1.
Trace each decoded character error back to its EVA character and triple.

Dependency chain:
    ed1_decomposition.json     (Step 39.1)
    combined_refine.json       (Step 15)
    consonant_grouping.json    (Step 37.1)
    decode_10k.json            (Step 36.1)
    modifier_integrate.json    (Step 16)
    vowel_confusion.json       (Step 37.3)
        → vowel_error_map.json (this step)
"""

import json
import os
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    tokenize_eva_chars,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _convert(obj: Any) -> Any:
    if hasattr(obj, '__dataclass_fields__'):
        from dataclasses import asdict
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


def _safe_load(path: str) -> Dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


# ---------------------------------------------------------------------------
# Syllable alignment
# ---------------------------------------------------------------------------

def _extract_vowel(syllable: str) -> str:
    """Extract the vowel from a CV syllable like 'co' → 'o'."""
    vowels = set('aeiou')
    for ch in syllable:
        if ch in vowels:
            return ch
    return ''


def _extract_onset(syllable: str) -> str:
    """Extract the onset consonant from a CV syllable like 'co' → 'c'."""
    vowels = set('aeiou')
    onset = []
    for ch in syllable:
        if ch in vowels:
            break
        onset.append(ch)
    return ''.join(onset)


def _syllable_with_vowel(syllable: str, new_vowel: str) -> str:
    """Replace the vowel in a syllable: ('co', 'u') → 'cu'."""
    vowels = set('aeiou')
    result = []
    replaced = False
    for ch in syllable:
        if ch in vowels and not replaced:
            result.append(new_vowel)
            replaced = True
        else:
            result.append(ch)
    return ''.join(result)


def _trace_error_to_triple(
    decoded_word: str,
    reference_word: str,
    edit_op: Dict,
    eva_token: str,
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    modifier_chars: Set[str],
) -> Optional[Dict]:
    """Trace a vowel substitution error back to a specific triple.

    The decoded word is produced by concatenating syllables from each
    non-modifier EVA char's triple assignment. Identify which syllable
    position contains the error, and which triple produced it.
    """
    if edit_op['error_type'] != 'SUBSTITUTION':
        return None

    # Decompose EVA token into syllabic chars (skip modifiers)
    chars = tokenize_eva_chars(eva_token)
    syllabic_chars = [ch for ch in chars if ch not in modifier_chars]

    # Build per-char syllable assignments
    syllables = []
    triple_keys = []
    for ch in syllabic_chars:
        triple = eva_to_triple.get(ch, '')
        syl = assignment.get(triple, '?') if triple else '?'
        syllables.append(syl)
        triple_keys.append(triple)

    # Reconstruct the decoded word from syllables
    reconstructed = ''.join(syllables)

    # Find which syllable contains the error position
    error_pos = edit_op['error_position']
    cumulative = 0
    for syl_idx, syl in enumerate(syllables):
        syl_start = cumulative
        syl_end = cumulative + len(syl)
        if syl_start <= error_pos < syl_end:
            # This syllable contains the error
            local_pos = error_pos - syl_start
            needed_vowel = edit_op['reference_char']
            current_vowel = edit_op['decoded_char']
            needed_syllable = _syllable_with_vowel(syl, needed_vowel)

            return {
                'triple_key': triple_keys[syl_idx],
                'eva_char': syllabic_chars[syl_idx],
                'syllable_index': syl_idx,
                'current_syllable': syl,
                'needed_syllable': needed_syllable,
                'current_vowel': current_vowel,
                'needed_vowel': needed_vowel,
                'onset': _extract_onset(syl),
            }
        cumulative = syl_end

    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_vowel_error_map() -> None:
    """Step 39.2: Vowel Error Map."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 39.2: Vowel Error Map")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load inputs ──
    print("\n  1. Loading inputs …")
    ed1_data = _safe_load(os.path.join(rd, 'ed1_decomposition.json'))
    refine_data = _safe_load(os.path.join(rd, 'combined_refine.json'))
    consonant_data = _safe_load(os.path.join(rd, 'consonant_grouping.json'))
    vowel_conf_data = _safe_load(os.path.join(rd, 'vowel_confusion.json'))
    mod_data = _safe_load(os.path.join(rd, 'modifier_integrate.json'))

    assignment = refine_data.get('best_assignment', {})
    triple_to_consonant = consonant_data.get('triple_to_consonant', {})
    consonant_to_triples = consonant_data.get('consonant_to_triples', {})
    cc_entries = ed1_data.get('cc_entries', [])

    # Reconstruct modifier chars
    modifier_set = set()
    for entry in mod_data.get('classifications', []):
        if isinstance(entry, dict) and entry.get('final_class') == 'modifier':
            modifier_set.add(entry.get('eva_char', ''))
        elif isinstance(entry, str):
            modifier_set.add(entry)
    if not modifier_set:
        modifier_set = {'h', 'iin', 'b', 'ckh', 'i', 'iiin', 'u', 'aiin',
                        'al', 'ar', 'dy', 'ey', 'm', 'n', 'or'}

    eva_to_triple = build_eva_to_triple_lookup()

    print(f"     CC entries: {len(cc_entries)}")
    print(f"     Assignment triples: {len(assignment)}")
    print(f"     Consonant classes: {len(consonant_to_triples)}")

    # ── 2. Trace each vowel error to its triple ──
    print("\n  2. Tracing vowel errors to triples …")

    # Collect all triple-level corrections
    # triple_key → list of (needed_syllable, source_entry)
    triple_corrections: Dict[str, List[Dict]] = defaultdict(list)
    all_tracings = []

    for entry in cc_entries:
        if not entry.get('has_vowel_error'):
            continue

        for ref_match in entry.get('reference_matches', []):
            # Check w1 vowel error
            if ref_match.get('w1_vowel_error') and ref_match.get('w1_edit'):
                trace = _trace_error_to_triple(
                    entry['w1'], ref_match['ref_w1'], ref_match['w1_edit'],
                    entry['eva_token_w1'], assignment, eva_to_triple, modifier_set,
                )
                if trace:
                    trace['decoded_word'] = entry['w1']
                    trace['reference_word'] = ref_match['ref_w1']
                    trace['folio'] = entry['folio']
                    trace['position'] = entry['position']
                    trace['word_position'] = 'w1'
                    all_tracings.append(trace)
                    triple_corrections[trace['triple_key']].append({
                        'needed_syllable': trace['needed_syllable'],
                        'current_syllable': trace['current_syllable'],
                        'decoded_word': entry['w1'],
                        'reference_word': ref_match['ref_w1'],
                        'folio': entry['folio'],
                    })

            # Check w2 vowel error
            if ref_match.get('w2_vowel_error') and ref_match.get('w2_edit'):
                trace = _trace_error_to_triple(
                    entry['w2'], ref_match['ref_w2'], ref_match['w2_edit'],
                    entry['eva_token_w2'], assignment, eva_to_triple, modifier_set,
                )
                if trace:
                    trace['decoded_word'] = entry['w2']
                    trace['reference_word'] = ref_match['ref_w2']
                    trace['folio'] = entry['folio']
                    trace['position'] = entry['position']
                    trace['word_position'] = 'w2'
                    all_tracings.append(trace)
                    triple_corrections[trace['triple_key']].append({
                        'needed_syllable': trace['needed_syllable'],
                        'current_syllable': trace['current_syllable'],
                        'decoded_word': entry['w2'],
                        'reference_word': ref_match['ref_w2'],
                        'folio': entry['folio'],
                    })

    print(f"     Total tracings: {len(all_tracings)}")
    print(f"     Triples with corrections: {len(triple_corrections)}")

    # ── 3. Aggregate and assign confidence tiers ──
    print("\n  3. Aggregating corrections by triple …")

    corrections_by_triple = []
    for triple_key, corrs in triple_corrections.items():
        # Group by needed syllable
        needed_counts = Counter(c['needed_syllable'] for c in corrs)
        current = corrs[0]['current_syllable']
        consonant_class = triple_to_consonant.get(triple_key, 'unknown')

        # Most common needed syllable
        most_common_needed, top_count = needed_counts.most_common(1)[0]

        # Check for conflicts
        is_conflicted = len(needed_counts) > 1

        # Assign tier
        if is_conflicted:
            tier = 'CONFLICTED'
        elif top_count >= 3:
            tier = 'TIER1'
        elif top_count >= 2:
            tier = 'TIER2'
        else:
            tier = 'TIER3'

        corrections_by_triple.append({
            'triple_key': triple_key,
            'consonant_class': consonant_class,
            'current_syllable': current,
            'most_common_needed': most_common_needed,
            'n_supporting': top_count,
            'tier': tier,
            'is_conflicted': is_conflicted,
            'all_needed_counts': dict(needed_counts),
            'supporting_evidence': [
                {'decoded': c['decoded_word'],
                 'reference': c['reference_word'],
                 'folio': c['folio']}
                for c in corrs[:10]  # limit stored evidence
            ],
        })

    corrections_by_triple.sort(key=lambda x: x['n_supporting'], reverse=True)

    # Count by tier
    tier_counts = Counter(c['tier'] for c in corrections_by_triple)
    for tier_name in ['TIER1', 'TIER2', 'TIER3', 'CONFLICTED']:
        print(f"     {tier_name}: {tier_counts.get(tier_name, 0)}")

    # ── 4. Within-class vowel rotation analysis ──
    print("\n  4. Within-class vowel rotation analysis …")

    swap_proposals = []
    for corr in corrections_by_triple:
        if corr['tier'] == 'CONFLICTED':
            continue
        triple_key = corr['triple_key']
        cons_class = corr['consonant_class']
        needed = corr['most_common_needed']

        # Find if another triple in the same class currently has the needed syllable
        class_triples = consonant_to_triples.get(cons_class, [])
        swap_partner = None
        for other_triple in class_triples:
            if other_triple != triple_key and assignment.get(other_triple) == needed:
                swap_partner = other_triple
                break

        if swap_partner:
            swap_proposals.append({
                'triple_key': triple_key,
                'swap_partner': swap_partner,
                'consonant_class': cons_class,
                'current_to_needed': f"{corr['current_syllable']} → {needed}",
                'partner_current': assignment.get(swap_partner, '?'),
                'tier': corr['tier'],
            })
            print(f"     Swap: {triple_key} ({corr['current_syllable']}→{needed}) "
                  f"↔ {swap_partner} ({assignment.get(swap_partner, '?')})")

    # ── 5. Cross-reference with Phase 37.3 ──
    print("\n  5. Cross-referencing with Phase 37.3 vowel confusion …")

    phase37_corrections = []
    class_results = vowel_conf_data.get('class_results', [])
    for cr in class_results:
        if cr.get('changed'):
            phase37_corrections.append({
                'consonant': cr['consonant'],
                'current_vowels': cr.get('current_vowels', []),
                'best_vowel_ordering': cr.get('best_vowel_ordering', []),
            })

    # Compare CC-bigram corrections with Phase 37.3
    agreements = 0
    disagreements = 0
    no_overlap = 0
    cross_reference = []
    for corr in corrections_by_triple:
        triple_key = corr['triple_key']
        cons_class = corr['consonant_class']
        # Find if Phase 37.3 has a correction for this class
        p37_match = None
        for p37 in phase37_corrections:
            if p37['consonant'] == cons_class:
                p37_match = p37
                break
        if p37_match:
            # Check agreement
            if corr['most_common_needed'] in [
                cons_class + v for v in 'aeiou'
                if cons_class + v in set(
                    assignment.get(t, '')
                    for t in consonant_to_triples.get(cons_class, [])
                )
            ]:
                status = 'OVERLAP'
            else:
                status = 'DIFFERENT'
                disagreements += 1
            cross_reference.append({
                'triple': triple_key,
                'cc_correction': corr['most_common_needed'],
                'phase37_class': cons_class,
                'status': status,
            })
        else:
            no_overlap += 1

    print(f"     Phase 37.3 corrections: {len(phase37_corrections)}")
    print(f"     Cross-reference overlaps: {len(cross_reference)}")

    # ── 6. Save ──
    elapsed = time.time() - t0

    output = {
        'n_tracings': len(all_tracings),
        'n_triples_with_corrections': len(corrections_by_triple),
        'n_tier1': tier_counts.get('TIER1', 0),
        'n_tier2': tier_counts.get('TIER2', 0),
        'n_tier3': tier_counts.get('TIER3', 0),
        'n_conflicted': tier_counts.get('CONFLICTED', 0),
        'corrections_by_triple': corrections_by_triple,
        'swap_proposals': swap_proposals,
        'all_tracings': all_tracings[:100],  # limit for file size
        'cross_reference_phase37': cross_reference,
        'phase37_corrections': phase37_corrections,
        'verdict': (
            f"{len(all_tracings)} tracings → "
            f"{len(corrections_by_triple)} triples "
            f"(T1={tier_counts.get('TIER1', 0)}, "
            f"T2={tier_counts.get('TIER2', 0)}, "
            f"T3={tier_counts.get('TIER3', 0)}, "
            f"CONFL={tier_counts.get('CONFLICTED', 0)}). "
            f"{len(swap_proposals)} swap proposals."
        ),
        'runtime_seconds': round(elapsed, 1),
    }

    out_path = os.path.join(rd, 'vowel_error_map.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(output), f, indent=2)
    print(f"\n  Saved → {out_path} ({elapsed:.1f}s)")
