"""
Step 37.3 – Vowel Confusion Matrix
=====================================
If Steps 37.1–37.2 confirm the consonant-correct hypothesis, map which
vowels are confused with which within each consonant class.  Enumerate
vowel permutations within each class and find the combination that
maximizes content-word recovery.

Dependency chain:
    consonant_grouping.json    (Step 37.1)
    cv_correlation.json        (Step 37.2)
    combined_refine.json       (Phase 15)
    modifier_integrate.json    (Phase 16)
    decode_10k.json            (Step 36.1)
    signal_10k.json            (Step 36.2)
        → vowel_confusion.json (this step)
"""

import itertools
import json
import os
import random
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    decode_token_modifier_aware,
    load_corpus,
    token_to_triples,
)
from voynich.core.reference import load_reference_corpus, build_expanded_word_set


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


_FUNCTION_WORDS = {
    'de', 'in', 'ad', 'et', 'cum', 'per', 'pro', 'ex', 'ab', 'sub',
    'non', 'si', 'ut', 'sed', 'ne', 'ac', 'at', 'an', 'se', 'te',
    'me', 'nos', 'iam', 'est', 'sunt', 'hoc', 'id', 'ea', 'is',
    'di', 'du', 'ce', 'ci', 'co', 'cu', 'bi', 'bo', 'be', 'da',
    'la', 'le', 'li', 'lo', 'ni', 'no', 'nu', 'ra', 're', 'ri',
    'ro', 'sa', 'so', 'su', 'ta', 'ti', 'to',
}


def _is_content_word(word: str) -> bool:
    return word.lower() not in _FUNCTION_WORDS and len(word) >= 4


def _extract_onset(word: str) -> str:
    """Extract consonant onset from a syllable."""
    vowels = set('aeiou')
    onset = ''
    for ch in word:
        if ch in vowels:
            break
        onset += ch
    return onset


def _extract_vowel(syllable: str) -> str:
    """Extract vowel part from a CV syllable."""
    onset = _extract_onset(syllable)
    return syllable[len(onset):]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_vowel_confusion() -> None:
    """Step 37.3: Vowel Confusion Matrix."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 37.3: Vowel Confusion Matrix")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load inputs ──
    print("\n  1. Loading inputs …")
    cg_data = _safe_load(os.path.join(rd, 'consonant_grouping.json'))
    cv_data = _safe_load(os.path.join(rd, 'cv_correlation.json'))
    refine_data = _safe_load(os.path.join(rd, 'combined_refine.json'))
    mod_data = _safe_load(os.path.join(rd, 'modifier_integrate.json'))
    signal_data = _safe_load(os.path.join(rd, 'signal_10k.json'))

    assignment = refine_data.get('best_assignment', {})
    consonant_to_triples = cg_data.get('consonant_to_triples', {})
    hypothesis_confirmed = cv_data.get('hypothesis_confirmed', False)
    modifier_chars = set(mod_data.get('modifier_chars', []))

    token_evas = signal_data.get('token_evas', [])
    token_decoded = signal_data.get('token_decoded', [])
    token_folios = signal_data.get('token_folios', [])

    print(f"     Hypothesis confirmed: {hypothesis_confirmed}")
    print(f"     {len(assignment)} triple assignments")
    print(f"     {len(consonant_to_triples)} consonant classes")

    # ── 2. Build dictionary and decode tools ──
    print("  2. Building dictionary and decode tools …")
    ref = load_reference_corpus(languages=['latin'], verbose=False)
    base_words = set(w.lower() for w in ref.get_combined_tokens('latin') if len(w) >= 2)
    # Use 10K subset for strict matching
    word_freq = Counter(w.lower() for w in ref.get_combined_tokens('latin') if len(w) >= 2)
    top_10k = set(w for w, _ in word_freq.most_common(10000))

    eva_to_triple = build_eva_to_triple_lookup()

    # Build modifier rules from modifier_integrate.json
    modifier_rules: Dict[str, str] = {}
    for c in mod_data.get('classifications', []):
        if c.get('eva_char') in modifier_chars:
            modifier_rules[c['eva_char']] = 'silent'

    # ── 3. Identify consonant classes with their triples ──
    print("  3. Identifying consonant classes …")
    classes = []
    for cons, triples in consonant_to_triples.items():
        if len(triples) < 2:
            continue
        # Get current syllable assignments for these triples
        members = []
        for t in triples:
            syl = assignment.get(t, '')
            if syl:
                vowel = _extract_vowel(syl)
                members.append({'triple': t, 'syllable': syl, 'vowel': vowel})
        if len(members) >= 2:
            classes.append({
                'consonant': cons,
                'members': members,
                'n_members': len(members),
                'vowels': [m['vowel'] for m in members],
            })
    classes.sort(key=lambda c: c['n_members'], reverse=True)

    print(f"     {len(classes)} classes with ≥2 members:")
    for cls in classes:
        syls = [m['syllable'] for m in cls['members']]
        print(f"       {cls['consonant']:<10s} {cls['n_members']} members: {', '.join(syls)}")

    # ── 4. Per-class vowel permutation search ──
    print("  4. Per-class vowel permutation search …")

    # Use a sample of tokens for speed
    sample_size = min(5000, len(token_evas))
    rng = random.Random(42)
    sample_indices = sorted(rng.sample(range(len(token_evas)), sample_size))
    sample_evas = [token_evas[i] for i in sample_indices]

    def _decode_sample(table: Dict[str, str]) -> List[str]:
        decoded = []
        for eva in sample_evas:
            d = decode_token_modifier_aware(
                eva, table, eva_to_triple, modifier_chars, modifier_rules)
            decoded.append(d.lower())
        return decoded

    def _score(decoded: List[str]) -> Tuple[float, int]:
        hits = sum(1 for w in decoded if w in top_10k)
        content_hits = sum(1 for w in decoded if w in top_10k and _is_content_word(w))
        return hits / len(decoded), content_hits

    # Baseline
    baseline_decoded = _decode_sample(assignment)
    baseline_hit, baseline_content = _score(baseline_decoded)
    print(f"     Baseline: dict_hit={baseline_hit:.3%}, content_words={baseline_content}")

    class_results = []
    best_perms: Dict[str, List[str]] = {}  # consonant → best vowel ordering

    for cls in classes:
        members = cls['members']
        consonant = cls['consonant']
        n = len(members)
        current_vowels = [m['vowel'] for m in members]
        unique_vowels = list(set(current_vowels))

        # Generate permutations of vowel assignments
        # If members share vowels, enumerate permutations of available vowels
        perms = list(itertools.permutations(current_vowels))
        # Remove duplicates
        perms = list(set(perms))

        # Limit to manageable number
        if len(perms) > 120:
            rng.shuffle(perms)
            perms = perms[:120]

        best_perm = tuple(current_vowels)
        best_hit = baseline_hit
        best_content = baseline_content
        perm_scores = []

        for perm in perms:
            # Create modified assignment
            new_table = dict(assignment)
            for i, member in enumerate(members):
                new_syl = consonant + perm[i] if consonant != '<vowel>' else perm[i]
                new_table[member['triple']] = new_syl

            decoded = _decode_sample(new_table)
            hit_rate, content_count = _score(decoded)
            perm_scores.append({
                'vowel_ordering': list(perm),
                'dict_hit': round(hit_rate, 4),
                'content_words': content_count,
            })

            if content_count > best_content or (
                    content_count == best_content and hit_rate > best_hit):
                best_hit = hit_rate
                best_content = content_count
                best_perm = perm

        delta_hit = best_hit - baseline_hit
        delta_content = best_content - baseline_content

        class_results.append({
            'consonant': consonant,
            'n_members': n,
            'current_vowels': current_vowels,
            'best_vowel_ordering': list(best_perm),
            'n_permutations_tested': len(perms),
            'baseline_hit': round(baseline_hit, 4),
            'best_hit': round(best_hit, 4),
            'delta_hit': round(delta_hit, 4),
            'baseline_content': baseline_content,
            'best_content': best_content,
            'delta_content': delta_content,
            'changed': list(best_perm) != current_vowels,
        })

        best_perms[consonant] = list(best_perm)

        status = "CHANGED" if list(best_perm) != current_vowels else "unchanged"
        print(f"       {consonant:<10s} {len(perms):>3d} perms tested → "
              f"hit={best_hit:.3%} content={best_content} [{status}]")

    # ── 5. Joint optimization ──
    print("  5. Joint optimization (all classes simultaneously) …")
    corrected_table = dict(assignment)
    n_changes = 0
    for cls in classes:
        consonant = cls['consonant']
        if consonant not in best_perms:
            continue
        best_vowels = best_perms[consonant]
        for i, member in enumerate(cls['members']):
            new_syl = consonant + best_vowels[i] if consonant != '<vowel>' else best_vowels[i]
            if new_syl != assignment.get(member['triple'], ''):
                corrected_table[member['triple']] = new_syl
                n_changes += 1

    joint_decoded = _decode_sample(corrected_table)
    joint_hit, joint_content = _score(joint_decoded)
    print(f"     {n_changes} triple assignments changed")
    print(f"     Joint hit rate: {joint_hit:.3%} (baseline: {baseline_hit:.3%})")
    print(f"     Joint content:  {joint_content} (baseline: {baseline_content})")

    # ── 6. Held-out validation ──
    print("  6. Held-out validation …")
    # Split by folio: odd-numbered positions = train, even = test
    even_indices = [i for i in range(len(token_evas)) if i % 2 == 0]
    even_evas = [token_evas[i] for i in even_indices[:2500]]

    def _decode_held_out(table: Dict[str, str]) -> List[str]:
        decoded = []
        for eva in even_evas:
            d = decode_token_modifier_aware(
                eva, table, eva_to_triple, modifier_chars, modifier_rules)
            decoded.append(d.lower())
        return decoded

    baseline_ho = _decode_held_out(assignment)
    corrected_ho = _decode_held_out(corrected_table)
    baseline_ho_hit = sum(1 for w in baseline_ho if w in top_10k) / len(baseline_ho)
    corrected_ho_hit = sum(1 for w in corrected_ho if w in top_10k) / len(corrected_ho)
    baseline_ho_content = sum(1 for w in baseline_ho if w in top_10k and _is_content_word(w))
    corrected_ho_content = sum(1 for w in corrected_ho if w in top_10k and _is_content_word(w))

    print(f"     Held-out baseline: hit={baseline_ho_hit:.3%}, content={baseline_ho_content}")
    print(f"     Held-out corrected: hit={corrected_ho_hit:.3%}, content={corrected_ho_content}")
    generalizes = corrected_ho_hit >= baseline_ho_hit

    # ── 7. Check for content-content bigrams ──
    print("  7. Checking for content-content bigrams …")
    # Decode full corpus with corrected table
    full_decoded = []
    for eva in token_evas:
        d = decode_token_modifier_aware(
            eva, corrected_table, eva_to_triple, modifier_chars, modifier_rules)
        full_decoded.append(d.lower())

    # Build reference bigrams
    ref_tokens_lower = [w.lower() for w in ref.get_combined_tokens('latin')]
    ref_bigrams = set()
    for i in range(len(ref_tokens_lower) - 1):
        if ref_tokens_lower[i] in top_10k and ref_tokens_lower[i + 1] in top_10k:
            ref_bigrams.add((ref_tokens_lower[i], ref_tokens_lower[i + 1]))

    content_content_bigrams = []
    for i in range(len(full_decoded) - 1):
        w1, w2 = full_decoded[i], full_decoded[i + 1]
        if (w1 in top_10k and w2 in top_10k and
                _is_content_word(w1) and _is_content_word(w2)):
            if (w1, w2) in ref_bigrams:
                content_content_bigrams.append({
                    'word1': w1, 'word2': w2,
                    'folio': token_folios[i] if i < len(token_folios) else '',
                    'position': i,
                })

    n_cc = len(content_content_bigrams)
    print(f"     Content-content bigrams: {n_cc}")
    if content_content_bigrams:
        for cc in content_content_bigrams[:10]:
            print(f"       \"{cc['word1']} {cc['word2']}\" on {cc['folio']}")

    # ── 8. Save ──
    elapsed = time.time() - t0

    output = {
        'hypothesis_confirmed': hypothesis_confirmed,
        'n_consonant_classes': len(classes),
        'class_results': class_results,
        'n_changes': n_changes,
        'corrected_table': corrected_table,
        'baseline_dict_hit': round(baseline_hit, 4),
        'baseline_content_words': baseline_content,
        'joint_dict_hit': round(joint_hit, 4),
        'joint_content_words': joint_content,
        'delta_hit': round(joint_hit - baseline_hit, 4),
        'delta_content': joint_content - baseline_content,
        'held_out_baseline_hit': round(baseline_ho_hit, 4),
        'held_out_corrected_hit': round(corrected_ho_hit, 4),
        'generalizes': generalizes,
        'n_content_content_bigrams': n_cc,
        'content_content_bigrams': content_content_bigrams[:50],
        'verdict': (
            f"Vowel confusion: {n_changes} changes, "
            f"hit={joint_hit:.3%} (Δ={joint_hit - baseline_hit:+.3%}), "
            f"content={joint_content} (Δ={joint_content - baseline_content:+d}), "
            f"CC bigrams={n_cc}. "
            f"Held-out: {'GENERALIZES' if generalizes else 'OVERFITS'}."
        ),
        'runtime_seconds': round(elapsed, 1),
    }

    out_path = os.path.join(rd, 'vowel_confusion.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(output), f, indent=2)
    print(f"\n  Saved → {out_path} ({elapsed:.1f}s)")
