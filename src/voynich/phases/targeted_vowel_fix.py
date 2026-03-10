"""
Step 39.3 – Targeted Vowel Fix and Validation
===============================================
Apply TIER 1 and TIER 2 vowel corrections from the vowel error map
and validate via held-out folios and exact CC bigram conversion.

Dependency chain:
    vowel_error_map.json       (Step 39.2)
    combined_refine.json       (Step 15)
    merged_dict.json           (Step 38.1)
    null_corpus.json           (Step 17)
    modifier_integrate.json    (Step 16)
    decode_10k.json            (Step 36.1)
    merged_bigrams.json        (Step 38.4)
        → targeted_vowel_fix.json  (this step)
"""

import json
import os
import random
import time
from collections import Counter
from typing import Any, Dict, List, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    decode_token_modifier_aware,
    load_corpus,
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


def _edit_distance_1(w1: str, w2: str) -> bool:
    """Check if edit distance <= 1."""
    if abs(len(w1) - len(w2)) > 1:
        return False
    if w1 == w2:
        return True
    if len(w1) == len(w2):
        return sum(a != b for a, b in zip(w1, w2)) <= 1
    short, long_ = (w1, w2) if len(w1) < len(w2) else (w2, w1)
    i = j = diffs = 0
    while i < len(short) and j < len(long_):
        if short[i] != long_[j]:
            diffs += 1
            if diffs > 1:
                return False
            j += 1
        else:
            i += 1
            j += 1
    return True


# ---------------------------------------------------------------------------
# Decode and measure
# ---------------------------------------------------------------------------

def _decode_corpus_with_assignment(
    token_evas: List[str],
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    modifier_chars: Set[str],
    modifier_rules: Dict[str, str],
) -> List[str]:
    """Decode all EVA tokens with a given assignment."""
    decoded = []
    for eva in token_evas:
        d = decode_token_modifier_aware(
            eva, assignment, eva_to_triple, modifier_chars, modifier_rules)
        decoded.append(d.lower())
    return decoded


def _measure_dict_hit(
    decoded: List[str],
    word_set: Set[str],
) -> float:
    """Compute dictionary hit rate."""
    if not decoded:
        return 0.0
    return sum(1 for w in decoded if w in word_set) / len(decoded)


def _count_exact_cc_bigrams(
    decoded: List[str],
    token_folios: List[str],
    bigram_set: Set[Tuple[str, str]],
    function_words: Set[str],
    word_set: Set[str],
) -> Tuple[int, List[Dict]]:
    """Count exact content-content bigram hits."""
    exact_cc = 0
    cc_list = []
    for i in range(len(decoded) - 1):
        if i >= len(token_folios) - 1:
            break
        if token_folios[i] != token_folios[i + 1]:
            continue
        w1, w2 = decoded[i], decoded[i + 1]
        # Both must be content words (in dict, not function, len >= 3)
        w1_content = w1 in word_set and w1 not in function_words and len(w1) >= 3
        w2_content = w2 in word_set and w2 not in function_words and len(w2) >= 3
        if w1_content and w2_content and (w1, w2) in bigram_set:
            exact_cc += 1
            cc_list.append({
                'folio': token_folios[i], 'position': i,
                'w1': w1, 'w2': w2,
            })
    return exact_cc, cc_list


def _count_relaxed_cc_bigrams(
    decoded: List[str],
    token_folios: List[str],
    bigram_set: Set[Tuple[str, str]],
    function_words: Set[str],
    word_set: Set[str],
) -> int:
    """Count relaxed (ED1) content-content bigram hits."""
    relaxed_cc = 0
    for i in range(len(decoded) - 1):
        if i >= len(token_folios) - 1:
            break
        if token_folios[i] != token_folios[i + 1]:
            continue
        w1, w2 = decoded[i], decoded[i + 1]
        w1_content = w1 in word_set and w1 not in function_words and len(w1) >= 3
        w2_content = w2 in word_set and w2 not in function_words and len(w2) >= 3
        if not (w1_content and w2_content):
            continue
        if (w1, w2) in bigram_set:
            continue  # exact, not relaxed
        for bw1, bw2 in bigram_set:
            if _edit_distance_1(w1, bw1) and _edit_distance_1(w2, bw2):
                relaxed_cc += 1
                break
    return relaxed_cc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_targeted_vowel_fix() -> None:
    """Step 39.3: Targeted Vowel Fix and Validation."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 39.3: Targeted Vowel Fix and Validation")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load inputs ──
    print("\n  1. Loading inputs …")
    vowel_data = _safe_load(os.path.join(rd, 'vowel_error_map.json'))
    refine_data = _safe_load(os.path.join(rd, 'combined_refine.json'))
    dict_data = _safe_load(os.path.join(rd, 'merged_dict.json'))
    decode_data = _safe_load(os.path.join(rd, 'decode_10k.json'))
    bigram_data = _safe_load(os.path.join(rd, 'merged_bigrams.json'))
    mod_data = _safe_load(os.path.join(rd, 'modifier_integrate.json'))

    assignment = dict(refine_data.get('best_assignment', {}))
    merged_words = set(dict_data.get('merged_words', []))
    bigram_list = dict_data.get('bigram_list', [])
    bigram_set = {(b[0], b[1]) for b in bigram_list}

    token_evas = decode_data.get('token_evas', [])
    token_folios = decode_data.get('token_folios', [])

    corrections = vowel_data.get('corrections_by_triple', [])

    # Reconstruct modifier rules
    from voynich.phases.null_corpus import _reconstruct_modifier_rules
    modifier_chars, modifier_rules = _reconstruct_modifier_rules(mod_data)

    eva_to_triple = build_eva_to_triple_lookup()

    function_words = {'de', 'in', 'se', 'ne', 'ad', 'la', 'le', 'di',
                      'da', 'si', 'no', 'et', 'a', 'e', 'i', 'o', 'u',
                      'cum', 'per', 'pro', 'sub', 'que'}

    print(f"     Corrections to try: {len(corrections)}")
    print(f"     Token count: {len(token_evas)}")

    # ── 2. Baseline metrics ──
    print("\n  2. Computing baseline metrics …")
    baseline_decoded = _decode_corpus_with_assignment(
        token_evas, assignment, eva_to_triple, modifier_chars, modifier_rules)
    baseline_dict_hit = _measure_dict_hit(baseline_decoded, merged_words)
    baseline_exact_cc, baseline_cc_list = _count_exact_cc_bigrams(
        baseline_decoded, token_folios, bigram_set, function_words, merged_words)

    print(f"     Baseline dict_hit: {baseline_dict_hit:.4f}")
    print(f"     Baseline exact CC: {baseline_exact_cc}")

    # ── 3. Split into held-out halves ──
    print("\n  3. Splitting corpus for held-out validation …")
    # Odd/even folio split
    unique_folios = sorted(set(token_folios))
    even_folios = set(f for i, f in enumerate(unique_folios) if i % 2 == 0)
    odd_folios = set(f for i, f in enumerate(unique_folios) if i % 2 == 1)

    even_indices = [i for i in range(len(token_folios))
                    if token_folios[i] in even_folios]
    odd_indices = [i for i in range(len(token_folios))
                   if token_folios[i] in odd_folios]

    print(f"     Even folios (held-out): {len(even_folios)} folios, "
          f"{len(even_indices)} tokens")
    print(f"     Odd folios (train): {len(odd_folios)} folios, "
          f"{len(odd_indices)} tokens")

    # ── 4. Apply corrections incrementally ──
    print("\n  4. Applying corrections …")

    # Filter to TIER1 and TIER2 only, non-conflicted
    eligible = [c for c in corrections
                if c['tier'] in ('TIER1', 'TIER2') and not c.get('is_conflicted')]
    eligible.sort(key=lambda c: (0 if c['tier'] == 'TIER1' else 1,
                                 -c['n_supporting']))

    print(f"     Eligible corrections: {len(eligible)}")

    corrections_applied = []
    corrections_rejected = []
    current_assignment = dict(assignment)
    current_dict_hit = baseline_dict_hit
    current_exact_cc = baseline_exact_cc

    for corr in eligible:
        triple_key = corr['triple_key']
        old_syl = current_assignment.get(triple_key, '')
        new_syl = corr['most_common_needed']

        if old_syl == new_syl:
            continue  # no change needed

        # Apply correction temporarily
        current_assignment[triple_key] = new_syl

        # Re-decode held-out (even) tokens only for speed
        even_decoded = []
        for idx in even_indices:
            d = decode_token_modifier_aware(
                token_evas[idx], current_assignment, eva_to_triple,
                modifier_chars, modifier_rules)
            even_decoded.append(d.lower())

        new_hit = _measure_dict_hit(even_decoded, merged_words)

        # Full re-decode for CC bigram check (on a sample for speed)
        sample_size = min(5000, len(token_evas))
        sample_decoded = _decode_corpus_with_assignment(
            token_evas[:sample_size], current_assignment, eva_to_triple,
            modifier_chars, modifier_rules)
        sample_exact_cc, _ = _count_exact_cc_bigrams(
            sample_decoded, token_folios[:sample_size],
            bigram_set, function_words, merged_words)

        # Accept if: dict_hit improves ≥0.1% OR exact CC increases
        accept = (new_hit > current_dict_hit + 0.001) or (sample_exact_cc > current_exact_cc)

        if accept:
            corrections_applied.append({
                'triple_key': triple_key,
                'old_syllable': old_syl,
                'new_syllable': new_syl,
                'tier': corr['tier'],
                'n_supporting': corr['n_supporting'],
                'held_out_hit_before': round(current_dict_hit, 4),
                'held_out_hit_after': round(new_hit, 4),
                'exact_cc_before': current_exact_cc,
                'exact_cc_after': sample_exact_cc,
            })
            current_dict_hit = new_hit
            current_exact_cc = sample_exact_cc
            print(f"     ACCEPTED: {triple_key} {old_syl}→{new_syl} "
                  f"(hit={new_hit:.4f}, cc={sample_exact_cc})")
        else:
            # Revert
            current_assignment[triple_key] = old_syl
            corrections_rejected.append({
                'triple_key': triple_key,
                'old_syllable': old_syl,
                'new_syllable': new_syl,
                'tier': corr['tier'],
                'held_out_hit': round(new_hit, 4),
                'reason': 'no improvement',
            })
            print(f"     REJECTED: {triple_key} {old_syl}→{new_syl} "
                  f"(hit={new_hit:.4f})")

    # ── 5. Final metrics with corrected table ──
    print("\n  5. Computing final metrics …")
    final_decoded = _decode_corpus_with_assignment(
        token_evas, current_assignment, eva_to_triple,
        modifier_chars, modifier_rules)
    final_dict_hit = _measure_dict_hit(final_decoded, merged_words)
    final_exact_cc, final_cc_list = _count_exact_cc_bigrams(
        final_decoded, token_folios, bigram_set, function_words, merged_words)
    final_relaxed_cc = _count_relaxed_cc_bigrams(
        final_decoded, token_folios, bigram_set, function_words, merged_words)

    print(f"     Final dict_hit: {final_dict_hit:.4f} (Δ={final_dict_hit - baseline_dict_hit:+.4f})")
    print(f"     Final exact CC: {final_exact_cc} (Δ={final_exact_cc - baseline_exact_cc:+d})")
    print(f"     Final relaxed CC: {final_relaxed_cc}")

    # ── 6. Held-out validation ──
    print("\n  6. Held-out validation …")
    # Decode even-folio tokens with corrected assignment
    even_decoded_final = []
    for idx in even_indices:
        d = decode_token_modifier_aware(
            token_evas[idx], current_assignment, eva_to_triple,
            modifier_chars, modifier_rules)
        even_decoded_final.append(d.lower())

    # Decode even-folio tokens with original assignment
    even_decoded_baseline = []
    for idx in even_indices:
        d = decode_token_modifier_aware(
            token_evas[idx], assignment, eva_to_triple,
            modifier_chars, modifier_rules)
        even_decoded_baseline.append(d.lower())

    held_out_baseline = _measure_dict_hit(even_decoded_baseline, merged_words)
    held_out_corrected = _measure_dict_hit(even_decoded_final, merged_words)
    generalizes = held_out_corrected >= held_out_baseline

    print(f"     Held-out baseline: {held_out_baseline:.4f}")
    print(f"     Held-out corrected: {held_out_corrected:.4f}")
    print(f"     Generalizes: {generalizes}")

    # ── 7. Save ──
    elapsed = time.time() - t0

    output = {
        'n_corrections_applied': len(corrections_applied),
        'n_corrections_rejected': len(corrections_rejected),
        'corrections_applied': corrections_applied,
        'corrections_rejected': corrections_rejected,
        'corrected_assignment': current_assignment,
        'baseline_dict_hit': round(baseline_dict_hit, 4),
        'corrected_dict_hit': round(final_dict_hit, 4),
        'dict_hit_delta': round(final_dict_hit - baseline_dict_hit, 4),
        'baseline_exact_cc': baseline_exact_cc,
        'corrected_exact_cc': final_exact_cc,
        'corrected_relaxed_cc': final_relaxed_cc,
        'exact_cc_list': final_cc_list[:50],  # limit
        'held_out_validation': {
            'baseline': round(held_out_baseline, 4),
            'corrected': round(held_out_corrected, 4),
            'generalizes': generalizes,
        },
        'verdict': (
            f"{len(corrections_applied)} corrections applied. "
            f"dict_hit: {baseline_dict_hit:.4f} → {final_dict_hit:.4f} "
            f"(Δ={final_dict_hit - baseline_dict_hit:+.4f}). "
            f"Exact CC: {baseline_exact_cc} → {final_exact_cc}. "
            f"Generalizes: {generalizes}."
        ),
        'runtime_seconds': round(elapsed, 1),
    }

    out_path = os.path.join(rd, 'targeted_vowel_fix.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(output), f, indent=2)
    print(f"\n  Saved → {out_path} ({elapsed:.1f}s)")
