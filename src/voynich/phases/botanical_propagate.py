"""
Step 39.10 -- Botanical Propagation
=====================================
If Step 39.9 found cross-folio consistent assignments, propagate through
sign families and re-run signal pipeline on botanical sections.  Measure
dict_hit improvement.

Dependency chain:
    italian_botanical_csp.json  (Step 39.9)
    combined_refine.json        (Phase 15)
    tachygraphic_stroke.json    (Phase 19.5)
        -> botanical_propagate.json  (this step)
"""

import json
import os
import time
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    decode_token_modifier_aware,
    load_corpus,
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
# Family propagation
# ---------------------------------------------------------------------------

def _propagate_through_families(
    base_assignment: Dict[str, str],
    new_assignments: Dict[str, str],
    sign_families: Dict[str, str],
) -> Dict[str, str]:
    """Propagate new assignments through sign families.

    Same family members should share the same consonant onset.
    Returns additional inferred assignments.
    """
    family_members: Dict[str, List[str]] = defaultdict(list)
    for triple_key, family in sign_families.items():
        family_members[family].append(triple_key)

    merged = {**base_assignment, **new_assignments}
    inferred: Dict[str, str] = {}

    for triple_key, syllable in new_assignments.items():
        family = sign_families.get(triple_key)
        if not family:
            continue

        # Extract consonant from the assigned syllable
        consonant = ''
        for ch in syllable:
            if ch not in 'aeiouy':
                consonant += ch
            else:
                break

        for member in family_members.get(family, []):
            if member == triple_key:
                continue
            if member in merged:
                continue
            if consonant:
                inferred[member] = f"{consonant}?"

    return inferred


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------

def _decode_tokens_simple(
    tokens: List[str],
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    modifier_chars: Set[str],
    word_set: Set[str],
) -> List[str]:
    """Decode tokens using assignment, return lowercase decoded forms."""
    decoded = []
    for token in tokens:
        d = decode_token_modifier_aware(
            token, assignment, eva_to_triple, modifier_chars)
        decoded.append(d.lower())
    return decoded


def _compute_dict_hit(decoded: List[str], word_set: Set[str]) -> float:
    """Compute dict hit rate."""
    if not decoded:
        return 0.0
    hits = sum(1 for w in decoded if w in word_set)
    return hits / len(decoded)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_botanical_propagate() -> None:
    """Step 39.10: Botanical Propagation."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 39.10: Botanical Propagation")
    print("=" * 70)

    rd = _results_dir()
    eva_to_triple = build_eva_to_triple_lookup()

    # -- 1. Load inputs --
    print("\n  1. Loading inputs ...")

    bot_csp = _safe_load(os.path.join(rd, 'italian_botanical_csp.json'))
    cross_folio = bot_csp.get('cross_folio_assignments', [])

    refine_data = _safe_load(os.path.join(rd, 'combined_refine.json'))
    base_assignment = refine_data.get('best_assignment', {})

    # Sign families from Phase 19
    sf_data = _safe_load(os.path.join(rd, 'tachygraphic_stroke.json'))
    sign_families: Dict[str, str] = {}
    for family in sf_data.get('families', []):
        family_name = family.get('family_name', '')
        for member in family.get('members', []):
            triple_key = member.get('triple_key', '')
            if triple_key:
                sign_families[triple_key] = family_name

    # Modifier chars
    mod_data = _safe_load(os.path.join(rd, 'modifier_integrate.json'))
    modifier_chars: Set[str] = set(mod_data.get('modifier_chars', []))

    # Reference word set: load from merged_dict for Italian+Latin coverage
    dict_data = _safe_load(os.path.join(rd, 'merged_dict.json'))
    word_set: Set[str] = set(dict_data.get('merged_words', []))
    # Add plant words if available
    plant_data = _safe_load(os.path.join(rd, 'italian_plant_names.json'))
    plant_words = set(plant_data.get('all_plant_words', []))
    word_set |= plant_words

    print(f"     Cross-folio assignments: {len(cross_folio)}")
    print(f"     Base assignment: {len(base_assignment)} triples")
    print(f"     Sign families: {len(sign_families)} entries")
    print(f"     Word set: {len(word_set)} words")

    # -- 2. Check for propagation opportunity --
    if not cross_folio:
        print("\n  2. No cross-folio assignments -- nothing to propagate.")

        # Compute baseline anyway for botanical sections
        corpus = load_corpus(verbose=False)
        botanical_folios = []
        for f in corpus.pages:
            if not f.startswith('f'):
                continue
            # Extract numeric part: f1r, f56v, f67r1 → 1, 56, 67
            digits = ''
            for ch in f[1:]:
                if ch.isdigit():
                    digits += ch
                else:
                    break
            if digits and int(digits) <= 56:
                botanical_folios.append(f)
        bot_tokens: List[str] = []
        for fid in botanical_folios:
            page = corpus.pages.get(fid)
            if page:
                bot_tokens.extend(page.all_tokens)

        if bot_tokens:
            decoded = _decode_tokens_simple(
                bot_tokens, base_assignment, eva_to_triple,
                modifier_chars, word_set)
            baseline_hit = _compute_dict_hit(decoded, word_set)
        else:
            baseline_hit = 0.0

        elapsed = time.time() - t0
        output = {
            'n_propagated': 0,
            'n_cross_folio': 0,
            'botanical_dict_hit_baseline': round(baseline_hit, 4),
            'botanical_dict_hit_corrected': round(baseline_hit, 4),
            'confirmed_triples': [],
            'verdict': 'NO_BOTANICAL_PROPAGATION',
            'runtime_seconds': round(elapsed, 1),
        }
        out_path = os.path.join(rd, 'botanical_propagate.json')
        with open(out_path, 'w') as f:
            json.dump(_convert(output), f, indent=2)
        print(f"\n  Saved -> {out_path} ({elapsed:.1f}s)")
        return

    # -- 3. Build corrected assignment --
    print("\n  3. Building corrected assignment ...")

    new_assignments: Dict[str, str] = {}
    for cfa in cross_folio:
        tk = cfa.get('triple_key', '')
        syl = cfa.get('syllable', '')
        if tk and syl:
            new_assignments[tk] = syl

    corrected_assignment = dict(base_assignment)
    corrected_assignment.update(new_assignments)

    print(f"     New assignments from Italian botanical: {len(new_assignments)}")
    for tk, syl in new_assignments.items():
        print(f"       {tk} -> '{syl}'")

    # -- 4. Family propagation --
    print("\n  4. Propagating through sign families ...")

    family_inferred = _propagate_through_families(
        base_assignment, new_assignments, sign_families)
    n_propagated = len(new_assignments) + len(family_inferred)

    print(f"     Family-inferred: {len(family_inferred)} additional")
    print(f"     Total propagated: {n_propagated}")

    # -- 5. Re-decode botanical sections --
    print("\n  5. Re-decoding botanical sections ...")

    corpus = load_corpus(verbose=False)

    # Identify botanical folios (herbal_a/herbal_b sections, roughly f1-f56)
    botanical_folios = []
    for f in corpus.pages:
        if not f.startswith('f'):
            continue
        digits = ''
        for ch in f[1:]:
            if ch.isdigit():
                digits += ch
            else:
                break
        if digits and int(digits) <= 56:
            botanical_folios.append(f)

    bot_tokens: List[str] = []
    for fid in sorted(botanical_folios):
        page = corpus.pages.get(fid)
        if page:
            bot_tokens.extend(page.all_tokens)

    if not bot_tokens:
        bot_tokens = corpus.get_tokens()[:5000]

    print(f"     Botanical tokens: {len(bot_tokens)}")

    # Baseline
    decoded_baseline = _decode_tokens_simple(
        bot_tokens, base_assignment, eva_to_triple,
        modifier_chars, word_set)
    baseline_hit = _compute_dict_hit(decoded_baseline, word_set)

    # Corrected
    decoded_corrected = _decode_tokens_simple(
        bot_tokens, corrected_assignment, eva_to_triple,
        modifier_chars, word_set)
    corrected_hit = _compute_dict_hit(decoded_corrected, word_set)

    delta = corrected_hit - baseline_hit

    print(f"     Baseline dict_hit: {baseline_hit:.4f}")
    print(f"     Corrected dict_hit: {corrected_hit:.4f}")
    print(f"     Delta: {delta:+.4f}")

    # -- 6. Verdict --
    confirmed_triples = list(new_assignments.keys())

    if delta > 0.02:
        verdict = f"BOTANICAL_IMPROVEMENT (+{delta:.4f})"
    elif delta > 0.0:
        verdict = f"MARGINAL_IMPROVEMENT (+{delta:.4f})"
    elif n_propagated > 0:
        verdict = f"NO_IMPROVEMENT ({n_propagated} propagated, delta={delta:+.4f})"
    else:
        verdict = "NO_BOTANICAL_PROPAGATION"

    elapsed = time.time() - t0

    output = {
        'n_propagated': n_propagated,
        'n_cross_folio': len(cross_folio),
        'botanical_dict_hit_baseline': round(baseline_hit, 4),
        'botanical_dict_hit_corrected': round(corrected_hit, 4),
        'delta': round(delta, 4),
        'confirmed_triples': confirmed_triples,
        'new_assignments': new_assignments,
        'family_inferred': family_inferred,
        'verdict': verdict,
        'runtime_seconds': round(elapsed, 1),
    }

    out_path = os.path.join(rd, 'botanical_propagate.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(output), f, indent=2)
    print(f"\n  Saved -> {out_path} ({elapsed:.1f}s)")
