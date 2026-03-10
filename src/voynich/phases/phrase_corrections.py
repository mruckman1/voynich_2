"""
Step 39.7 – Phrase-Derived Corrections
=======================================
Consolidate corrections from phrase cribs (39.5) and template alignment
(39.6).  Cross-reference with Track A's CC-bigram corrections.  Categorize
as CONVERGENT / CONFLICTED / PHRASE_ONLY / TRACK_A_ONLY.  Validate on
held-out folios.  Build combined correction table.

Dependency chain:
    phrase_cribs.json          (Step 39.5)
    phrase_alignment.json      (Step 39.6)
    targeted_vowel_fix.json    (Step 39.3)
        → phrase_corrections.json (this step)
"""

import json
import os
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from voynich.core._paths import results_dir as _results_dir


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
# Correction extraction helpers
# ---------------------------------------------------------------------------

def _extract_phrase_corrections(
    crib_data: Dict,
    alignment_data: Dict,
) -> Dict[str, Dict]:
    """Extract all phrase-derived corrections keyed by triple_key.

    Returns dict of triple_key → {proposed_syllable, source, support_count, …}
    """
    corrections: Dict[str, List[Dict]] = defaultdict(list)

    # From phrase_cribs.json: proposed_corrections
    for corr in crib_data.get('proposed_corrections', []):
        triple_key = corr.get('triple_key', '')
        if not triple_key:
            continue
        corrections[triple_key].append({
            'proposed_syllable': corr.get('proposed_syllable', ''),
            'source': 'phrase_crib',
            'n_supporting': corr.get('n_supporting_phrases', 1),
            'folio': corr.get('folio', ''),
        })

    # From phrase_alignment.json: template_predicted_corrections
    # These predict words, not syllables directly — but we can match
    # predicted words to triple corrections if alignment data provides them
    for pred in alignment_data.get('all_predictions', []):
        miss_word = pred.get('miss_word', '')
        slot_type = pred.get('slot_type', '')
        for pw in pred.get('predicted_words', [])[:3]:
            # Store as a word-level prediction (no triple_key)
            corrections[f'WORD:{miss_word}'].append({
                'predicted_word': pw,
                'source': 'template_alignment',
                'slot_type': slot_type,
                'template_name': pred.get('template_name', ''),
                'folio': pred.get('folio', ''),
            })

    # Consolidate: pick most-supported syllable per triple
    consolidated: Dict[str, Dict] = {}
    for key, entries in corrections.items():
        if key.startswith('WORD:'):
            # Word-level prediction — keep as-is
            consolidated[key] = {
                'type': 'word_prediction',
                'predictions': entries,
                'n_sources': len(entries),
            }
        else:
            # Triple-level correction — pick consensus syllable
            syl_counts: Counter = Counter()
            for e in entries:
                syl = e.get('proposed_syllable', '')
                if syl:
                    syl_counts[syl] += 1

            if syl_counts:
                best_syl, best_count = syl_counts.most_common(1)[0]
                consolidated[key] = {
                    'type': 'triple_correction',
                    'proposed_syllable': best_syl,
                    'n_supporting': best_count,
                    'total_entries': len(entries),
                    'all_syllables': dict(syl_counts),
                }

    return consolidated


def _extract_track_a_corrections(
    vowel_data: Dict,
) -> Dict[str, Dict]:
    """Extract Track A corrections keyed by triple_key."""
    corrections: Dict[str, Dict] = {}

    for corr in vowel_data.get('corrections_applied', []):
        triple_key = corr.get('triple_key', '')
        if not triple_key:
            continue
        corrections[triple_key] = {
            'old_syllable': corr.get('old_syllable', ''),
            'new_syllable': corr.get('new_syllable', ''),
            'tier': corr.get('tier', ''),
            'n_supporting': corr.get('n_supporting', 0),
        }

    return corrections


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def _classify_corrections(
    phrase_corrections: Dict[str, Dict],
    track_a_corrections: Dict[str, Dict],
    current_assignment: Dict[str, str],
) -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict]]:
    """Classify each correction as CONVERGENT/CONFLICTED/PHRASE_ONLY/TRACK_A_ONLY.

    Returns (convergent, conflicted, phrase_only, track_a_only) lists.
    """
    convergent: List[Dict] = []
    conflicted: List[Dict] = []
    phrase_only: List[Dict] = []
    track_a_only: List[Dict] = []

    # Triple-level phrase corrections
    phrase_triple_keys = {k for k, v in phrase_corrections.items()
                          if v.get('type') == 'triple_correction'}
    track_a_keys = set(track_a_corrections.keys())

    # Intersection: both tracks propose a correction for same triple
    shared_keys = phrase_triple_keys & track_a_keys

    for triple_key in shared_keys:
        phrase_syl = phrase_corrections[triple_key].get('proposed_syllable', '')
        track_a_syl = track_a_corrections[triple_key].get('new_syllable', '')
        current_syl = current_assignment.get(triple_key, '')

        entry = {
            'triple_key': triple_key,
            'current_syllable': current_syl,
            'phrase_syllable': phrase_syl,
            'track_a_syllable': track_a_syl,
            'phrase_support': phrase_corrections[triple_key].get('n_supporting', 0),
            'track_a_support': track_a_corrections[triple_key].get('n_supporting', 0),
        }

        if phrase_syl == track_a_syl:
            entry['category'] = 'CONVERGENT'
            convergent.append(entry)
        else:
            entry['category'] = 'CONFLICTED'
            conflicted.append(entry)

    # Phrase-only corrections
    for triple_key in phrase_triple_keys - track_a_keys:
        phrase_syl = phrase_corrections[triple_key].get('proposed_syllable', '')
        current_syl = current_assignment.get(triple_key, '')
        phrase_only.append({
            'triple_key': triple_key,
            'current_syllable': current_syl,
            'proposed_syllable': phrase_syl,
            'phrase_support': phrase_corrections[triple_key].get('n_supporting', 0),
            'category': 'PHRASE_ONLY',
        })

    # Track A-only corrections
    for triple_key in track_a_keys - phrase_triple_keys:
        track_a_syl = track_a_corrections[triple_key].get('new_syllable', '')
        old_syl = track_a_corrections[triple_key].get('old_syllable', '')
        track_a_only.append({
            'triple_key': triple_key,
            'old_syllable': old_syl,
            'new_syllable': track_a_syl,
            'track_a_support': track_a_corrections[triple_key].get('n_supporting', 0),
            'category': 'TRACK_A_ONLY',
        })

    return convergent, conflicted, phrase_only, track_a_only


# ---------------------------------------------------------------------------
# Build combined assignment
# ---------------------------------------------------------------------------

def _build_combined_assignment(
    base_assignment: Dict[str, str],
    convergent: List[Dict],
    phrase_only: List[Dict],
    track_a_only: List[Dict],
    apply_phrase_only: bool = False,
    apply_track_a_only: bool = True,
) -> Dict[str, str]:
    """Build a combined corrected assignment.

    Always applies CONVERGENT corrections.  Optionally applies
    PHRASE_ONLY and TRACK_A_ONLY based on flags.
    """
    combined = dict(base_assignment)

    # Apply convergent corrections (always)
    for entry in convergent:
        triple_key = entry['triple_key']
        # Use the agreed-upon syllable (phrase and track_a agree)
        combined[triple_key] = entry['phrase_syllable']

    # Apply track_a_only if requested (default: yes — already validated)
    if apply_track_a_only:
        for entry in track_a_only:
            triple_key = entry['triple_key']
            combined[triple_key] = entry['new_syllable']

    # Apply phrase_only if requested (default: no — needs validation)
    if apply_phrase_only:
        for entry in phrase_only:
            triple_key = entry['triple_key']
            combined[triple_key] = entry['proposed_syllable']

    return combined


# ---------------------------------------------------------------------------
# Held-out validation
# ---------------------------------------------------------------------------

def _validate_on_held_out(
    base_assignment: Dict[str, str],
    corrected_assignment: Dict[str, str],
    token_evas: List[str],
    token_folios: List[str],
    merged_words: Set[str],
    eva_to_triple_lookup: Dict[str, str],
    modifier_chars: Set[str],
    modifier_rules: Dict[str, str],
) -> Dict:
    """Validate corrections on held-out (even-index) folios."""
    from voynich.core.corpus import decode_token_modifier_aware

    unique_folios = sorted(set(token_folios))
    even_folios = set(f for i, f in enumerate(unique_folios) if i % 2 == 0)

    even_indices = [i for i in range(len(token_folios))
                    if token_folios[i] in even_folios]

    if not even_indices:
        return {
            'n_held_out_tokens': 0,
            'baseline_hit': 0.0,
            'corrected_hit': 0.0,
            'delta': 0.0,
            'generalizes': False,
        }

    # Decode with base assignment
    base_hits = 0
    for idx in even_indices:
        d = decode_token_modifier_aware(
            token_evas[idx], base_assignment, eva_to_triple_lookup,
            modifier_chars, modifier_rules,
        )
        if d.lower() in merged_words:
            base_hits += 1

    # Decode with corrected assignment
    corr_hits = 0
    for idx in even_indices:
        d = decode_token_modifier_aware(
            token_evas[idx], corrected_assignment, eva_to_triple_lookup,
            modifier_chars, modifier_rules,
        )
        if d.lower() in merged_words:
            corr_hits += 1

    n = len(even_indices)
    base_rate = base_hits / n if n else 0.0
    corr_rate = corr_hits / n if n else 0.0

    return {
        'n_held_out_tokens': n,
        'baseline_hit': round(base_rate, 4),
        'corrected_hit': round(corr_rate, 4),
        'delta': round(corr_rate - base_rate, 4),
        'generalizes': corr_rate >= base_rate,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_phrase_corrections() -> None:
    """Step 39.7: Phrase-Derived Corrections."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 39.7: Phrase-Derived Corrections")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load inputs ──
    print("\n  1. Loading inputs …")
    crib_data = _safe_load(os.path.join(rd, 'phrase_cribs.json'))
    alignment_data = _safe_load(os.path.join(rd, 'phrase_alignment.json'))
    vowel_data = _safe_load(os.path.join(rd, 'targeted_vowel_fix.json'))
    refine_data = _safe_load(os.path.join(rd, 'combined_refine.json'))
    decode_data = _safe_load(os.path.join(rd, 'decode_10k.json'))
    dict_data = _safe_load(os.path.join(rd, 'merged_dict.json'))
    mod_data = _safe_load(os.path.join(rd, 'modifier_integrate.json'))

    # Base assignment (pre-vowel-fix)
    base_assignment = dict(refine_data.get('best_assignment', {}))

    # Corrected assignment from Track A
    track_a_assignment = dict(vowel_data.get('corrected_assignment', {}))
    if not track_a_assignment:
        track_a_assignment = dict(base_assignment)

    merged_words = set(dict_data.get('merged_words', []))
    token_evas = decode_data.get('token_evas', [])
    token_folios = decode_data.get('token_folios', [])

    # Reconstruct modifier rules
    from voynich.phases.null_corpus import _reconstruct_modifier_rules
    modifier_chars, modifier_rules = _reconstruct_modifier_rules(mod_data)

    from voynich.core.corpus import build_eva_to_triple_lookup
    eva_to_triple = build_eva_to_triple_lookup()

    print(f"     Base assignment entries: {len(base_assignment)}")
    print(f"     Track A assignment entries: {len(track_a_assignment)}")
    print(f"     Phrase crib corrections: "
          f"{len(crib_data.get('proposed_corrections', []))}")
    print(f"     Template predictions: "
          f"{len(alignment_data.get('all_predictions', []))}")
    print(f"     Track A corrections applied: "
          f"{len(vowel_data.get('corrections_applied', []))}")

    # ── 2. Extract corrections from both tracks ──
    print("\n  2. Extracting corrections from both tracks …")
    phrase_corrections = _extract_phrase_corrections(crib_data, alignment_data)
    track_a_corrections = _extract_track_a_corrections(vowel_data)

    n_phrase_triple = sum(1 for v in phrase_corrections.values()
                         if v.get('type') == 'triple_correction')
    n_phrase_word = sum(1 for v in phrase_corrections.values()
                       if v.get('type') == 'word_prediction')

    print(f"     Phrase triple corrections: {n_phrase_triple}")
    print(f"     Phrase word predictions: {n_phrase_word}")
    print(f"     Track A corrections: {len(track_a_corrections)}")

    # ── 3. Classify corrections ──
    print("\n  3. Classifying corrections …")
    convergent, conflicted, phrase_only, track_a_only = _classify_corrections(
        phrase_corrections, track_a_corrections, track_a_assignment,
    )

    print(f"     CONVERGENT (both agree): {len(convergent)}")
    print(f"     CONFLICTED (disagree):   {len(conflicted)}")
    print(f"     PHRASE_ONLY:             {len(phrase_only)}")
    print(f"     TRACK_A_ONLY:            {len(track_a_only)}")

    for entry in convergent:
        print(f"       CONVERGENT: {entry['triple_key']} → "
              f"'{entry['phrase_syllable']}' "
              f"(phrase={entry['phrase_support']}, "
              f"track_a={entry['track_a_support']})")

    for entry in conflicted[:5]:
        print(f"       CONFLICTED: {entry['triple_key']} "
              f"phrase='{entry['phrase_syllable']}' vs "
              f"track_a='{entry['track_a_syllable']}'")

    # ── 4. Build combined assignment ──
    print("\n  4. Building combined corrected assignment …")
    # Start from Track A's corrected assignment (which already has
    # Track A corrections applied).  Layer convergent on top.
    combined_assignment = _build_combined_assignment(
        track_a_assignment,
        convergent,
        phrase_only,
        track_a_only,
        apply_phrase_only=False,  # conservative
        apply_track_a_only=False,  # already in track_a_assignment
    )

    n_changed = sum(1 for k in combined_assignment
                    if combined_assignment.get(k) != track_a_assignment.get(k))
    print(f"     Changes from Track A baseline: {n_changed}")

    # ── 5. Held-out validation ──
    print("\n  5. Held-out validation …")
    validation = _validate_on_held_out(
        track_a_assignment,
        combined_assignment,
        token_evas,
        token_folios,
        merged_words,
        eva_to_triple,
        modifier_chars,
        modifier_rules,
    )

    print(f"     Held-out tokens: {validation['n_held_out_tokens']}")
    print(f"     Baseline dict_hit: {validation['baseline_hit']:.4f}")
    print(f"     Corrected dict_hit: {validation['corrected_hit']:.4f}")
    print(f"     Delta: {validation['delta']:+.4f}")
    print(f"     Generalizes: {validation['generalizes']}")

    # ── 6. Summary of combined corrections ──
    print("\n  6. Combined corrections summary …")
    all_combined: List[Dict] = []
    for entry in convergent:
        all_combined.append({
            'triple_key': entry['triple_key'],
            'category': 'CONVERGENT',
            'syllable': entry['phrase_syllable'],
            'applied': True,
        })
    for entry in conflicted:
        all_combined.append({
            'triple_key': entry['triple_key'],
            'category': 'CONFLICTED',
            'phrase_syllable': entry['phrase_syllable'],
            'track_a_syllable': entry['track_a_syllable'],
            'applied': False,
        })
    for entry in phrase_only:
        all_combined.append({
            'triple_key': entry['triple_key'],
            'category': 'PHRASE_ONLY',
            'syllable': entry['proposed_syllable'],
            'applied': False,
        })
    for entry in track_a_only:
        all_combined.append({
            'triple_key': entry['triple_key'],
            'category': 'TRACK_A_ONLY',
            'syllable': entry['new_syllable'],
            'applied': True,  # already in Track A assignment
        })

    n_applied = sum(1 for c in all_combined if c.get('applied'))
    print(f"     Total corrections cataloged: {len(all_combined)}")
    print(f"     Corrections applied: {n_applied}")

    # ── 7. Save ──
    elapsed = time.time() - t0

    verdict_parts = [
        f"{len(convergent)} CONVERGENT",
        f"{len(conflicted)} CONFLICTED",
        f"{len(phrase_only)} PHRASE_ONLY",
        f"{len(track_a_only)} TRACK_A_ONLY",
        f"held-out delta={validation['delta']:+.4f}",
        f"generalizes={validation['generalizes']}",
    ]

    output = {
        'n_convergent': len(convergent),
        'n_conflicted': len(conflicted),
        'n_phrase_only': len(phrase_only),
        'n_track_a_only': len(track_a_only),
        'convergent': convergent,
        'conflicted': conflicted,
        'phrase_only': phrase_only[:50],
        'track_a_only': track_a_only[:50],
        'combined_corrections': all_combined,
        'final_corrected_assignment': combined_assignment,
        'held_out_validation': validation,
        'n_changes_from_track_a': n_changed,
        'verdict': '. '.join(verdict_parts) + '.',
        'runtime_seconds': round(elapsed, 1),
    }

    out_path = os.path.join(rd, 'phrase_corrections.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(output), f, indent=2)
    print(f"\n  Saved → {out_path} ({elapsed:.1f}s)")
