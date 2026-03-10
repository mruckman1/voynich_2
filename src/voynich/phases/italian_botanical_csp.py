"""
Step 39.9 -- Italian Botanical CSP
====================================
Re-run botanical alignment using Italian plant names instead of Latin.
For each Tier A/B folio from consensus_plants.json, get label candidates.
For each (label_token, italian_plant_name) pair, decompose EVA into triples,
syllabify Italian name, check length compatibility, enumerate alignments,
check confirmed triple consistency.  Cross-folio validation.

Dependency chain:
    italian_plant_names.json   (Step 39.8)
    combined_refine.json       (Phase 15)
    targeted_vowel_fix.json    (Step 39.3)
    modifier_integrate.json    (Phase 16)
    consensus_plants.json      (Phase 31.1)
        -> italian_botanical_csp.json  (this step)
"""

import json
import os
import time
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import build_eva_to_triple_lookup, tokenize_eva_chars


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
# Syllabification
# ---------------------------------------------------------------------------

def _syllabify_italian(word: str) -> List[str]:
    """Simple CV syllabification for Italian words."""
    vowels = set('aeiou')
    word = word.lower()
    syllables: List[str] = []
    current: List[str] = []

    i = 0
    while i < len(word):
        ch = word[i]
        if ch in vowels:
            current.append(ch)
            syllables.append(''.join(current))
            current = []
        else:
            if current and any(c in vowels for c in current):
                syllables.append(''.join(current))
                current = [ch]
            else:
                current.append(ch)
        i += 1

    if current:
        if syllables:
            syllables[-1] += ''.join(current)
        else:
            syllables.append(''.join(current))

    return syllables


# ---------------------------------------------------------------------------
# Alignment enumeration
# ---------------------------------------------------------------------------

def _check_length_compatibility(
    n_triples: int,
    n_syllables: int,
) -> Tuple[bool, str]:
    """Check if triple count is compatible with syllable count."""
    diff = abs(n_triples - n_syllables)
    if diff == 0:
        return True, 'exact'
    elif diff == 1:
        return True, 'off_by_1'
    return False, 'incompatible'


def _enumerate_alignments(
    triple_keys: List[str],
    syllables: List[str],
    max_alignments: int = 200,
) -> List[Dict[str, str]]:
    """Enumerate possible triple_key -> syllable alignments."""
    n_t = len(triple_keys)
    n_s = len(syllables)
    if n_t == 0 or n_s == 0:
        return []

    alignments: List[Dict[str, str]] = []

    if n_t == n_s:
        # Direct 1:1
        mapping = {}
        for tk, syl in zip(triple_keys, syllables):
            mapping[tk] = syl
        alignments.append(mapping)

    elif n_t == n_s + 1:
        # One extra triple -- try skipping each position
        for skip in range(n_t):
            mapping = {}
            syl_idx = 0
            valid = True
            for pos, tk in enumerate(triple_keys):
                if pos == skip:
                    continue
                if syl_idx >= n_s:
                    valid = False
                    break
                mapping[tk] = syllables[syl_idx]
                syl_idx += 1
            if valid and syl_idx == n_s:
                alignments.append(mapping)
            if len(alignments) >= max_alignments:
                break

    elif n_t == n_s - 1:
        # One fewer triple -- assign first n_t syllables
        mapping = {}
        for tk, syl in zip(triple_keys, syllables[:n_t]):
            mapping[tk] = syl
        alignments.append(mapping)

    return alignments[:max_alignments]


# ---------------------------------------------------------------------------
# Consistency checking
# ---------------------------------------------------------------------------

def _check_confirmed_consistency(
    alignment: Dict[str, str],
    confirmed_assignment: Dict[str, str],
) -> Tuple[int, int]:
    """Check alignment against confirmed assignment.

    Returns (n_consistent, n_conflicting).
    """
    n_consistent = 0
    n_conflicting = 0
    for triple_key, proposed_syl in alignment.items():
        if triple_key in confirmed_assignment:
            if confirmed_assignment[triple_key] == proposed_syl:
                n_consistent += 1
            else:
                n_conflicting += 1
    return n_consistent, n_conflicting


def _score_alignment(
    n_confirmed_consistent: int,
    n_confirmed_conflicting: int,
    n_unconfirmed: int,
    n_total: int,
) -> float:
    """Score an alignment based on consistency and coverage."""
    if n_confirmed_conflicting > 0:
        return 0.0
    confirmed_score = 1.0 if n_confirmed_consistent > 0 else 0.5
    coverage_score = n_total / max(n_total, 1)
    return (confirmed_score * 0.5 +
            (n_unconfirmed / 25.0) * 0.3 +
            coverage_score * 0.2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_italian_botanical_csp() -> None:
    """Step 39.9: Italian Botanical CSP."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 39.9: Italian Botanical CSP")
    print("=" * 70)

    rd = _results_dir()
    eva_to_triple = build_eva_to_triple_lookup()

    # -- 1. Load inputs --
    print("\n  1. Loading inputs ...")

    plant_data = _safe_load(os.path.join(rd, 'italian_plant_names.json'))
    plant_name_table = plant_data.get('plant_name_table', [])

    # Best assignment: prefer targeted_vowel_fix, fall back to combined_refine
    vowel_fix = _safe_load(os.path.join(rd, 'targeted_vowel_fix.json'))
    refine_data = _safe_load(os.path.join(rd, 'combined_refine.json'))

    if vowel_fix.get('corrected_assignment'):
        assignment = vowel_fix['corrected_assignment']
        assignment_source = 'targeted_vowel_fix'
    else:
        assignment = refine_data.get('best_assignment', {})
        assignment_source = 'combined_refine'

    # Consensus plants for tier folios and label candidates
    cp_data = _safe_load(os.path.join(rd, 'consensus_plants.json'))

    # Modifier chars
    mod_data = _safe_load(os.path.join(rd, 'modifier_integrate.json'))
    modifier_chars: Set[str] = set(mod_data.get('modifier_chars', []))

    tier_a = cp_data.get('tier_a_folios', [])
    tier_b = cp_data.get('tier_b_folios', [])
    all_tier_folios = tier_a + tier_b

    print(f"     Plant name table: {len(plant_name_table)} entries")
    print(f"     Assignment source: {assignment_source} ({len(assignment)} triples)")
    print(f"     Tier A folios: {len(tier_a)}, Tier B folios: {len(tier_b)}")

    # -- 2. Build folio -> Italian names lookup --
    print("\n  2. Building folio -> Italian names lookup ...")

    folio_to_italian: Dict[str, List[Dict]] = defaultdict(list)
    for entry in plant_name_table:
        folio = entry.get('folio', '')
        italian_names = entry.get('italian_names', [])
        venetian_names = entry.get('venetian_names', [])
        syllabified = entry.get('syllabified', {})
        if folio and (italian_names or venetian_names):
            folio_to_italian[folio].append({
                'italian_names': italian_names,
                'venetian_names': venetian_names,
                'syllabified': syllabified,
            })

    print(f"     Folios with Italian names: {len(folio_to_italian)}")

    # -- 3. Test each tier folio --
    print("\n  3. Testing alignments per folio ...")

    n_folios_tested = 0
    n_pairs_tested = 0
    n_valid_alignments = 0
    all_alignments: List[Dict] = []
    folio_proposals: Dict[str, List[Dict]] = defaultdict(list)

    for tier_entry in all_tier_folios:
        folio = tier_entry.get('folio', '')
        label_cands = tier_entry.get('label_candidates', [])
        if not folio or not label_cands:
            continue

        # Get Italian names for this folio
        italian_entries = folio_to_italian.get(folio, [])
        if not italian_entries:
            continue

        n_folios_tested += 1

        # Collect all Italian/Venetian names with syllabifications
        name_syllables: List[Tuple[str, List[str]]] = []
        for ie in italian_entries:
            for name in ie.get('italian_names', []) + ie.get('venetian_names', []):
                if ' ' in name:
                    continue  # Skip multi-word names
                syls = ie.get('syllabified', {}).get(name, [])
                if not syls:
                    syls = _syllabify_italian(name)
                if syls:
                    name_syllables.append((name, syls))

        for lc in label_cands[:5]:
            token = lc.get('token', '')
            eva_chars_raw = lc.get('eva_chars', [])
            if not token or not eva_chars_raw:
                continue

            # Filter modifier chars to get syllabic chars
            syllabic_chars = [ch for ch in eva_chars_raw if ch not in modifier_chars]
            # Get triple keys
            triple_keys = []
            for ch in syllabic_chars:
                tk = eva_to_triple.get(ch, ch)
                triple_keys.append(tk)

            for plant_name, syllables in name_syllables:
                n_pairs_tested += 1

                compatible, mode = _check_length_compatibility(
                    len(triple_keys), len(syllables))
                if not compatible:
                    continue

                alignments = _enumerate_alignments(triple_keys, syllables)

                for alignment in alignments:
                    n_cons, n_conf = _check_confirmed_consistency(
                        alignment, assignment)

                    if n_conf > 0:
                        continue

                    n_unconfirmed = sum(
                        1 for k in alignment if k not in assignment
                        or assignment[k] != alignment[k])
                    n_total = len(alignment)

                    score = _score_alignment(
                        n_cons, n_conf, n_unconfirmed, n_total)

                    if score > 0:
                        n_valid_alignments += 1
                        entry = {
                            'folio': folio,
                            'token': token,
                            'plant_name': plant_name,
                            'triple_keys': triple_keys,
                            'syllables': syllables,
                            'assignments': alignment,
                            'confirmed_consistent': n_cons,
                            'confirmed_conflicting': n_conf,
                            'unconfirmed': n_unconfirmed,
                            'score': round(score, 4),
                            'mode': mode,
                        }
                        all_alignments.append(entry)

                        # Record proposals for cross-folio check
                        for tk, syl in alignment.items():
                            folio_proposals[folio].append({
                                'triple_key': tk,
                                'syllable': syl,
                                'plant_name': plant_name,
                                'score': score,
                            })

        print(f"     {folio}: {len([a for a in all_alignments if a['folio'] == folio])} "
              f"valid alignments")

    print(f"\n     Folios tested: {n_folios_tested}")
    print(f"     Pairs tested: {n_pairs_tested}")
    print(f"     Valid alignments: {n_valid_alignments}")

    # -- 4. Cross-folio validation --
    print("\n  4. Cross-folio validation ...")

    triple_proposals: Dict[str, Dict[str, List[str]]] = defaultdict(
        lambda: defaultdict(list))
    for folio, proposals in folio_proposals.items():
        for p in proposals:
            tk = p['triple_key']
            syl = p['syllable']
            if folio not in triple_proposals[tk][syl]:
                triple_proposals[tk][syl].append(folio)

    cross_folio_assignments: List[Dict] = []
    for triple_key, syl_folios in triple_proposals.items():
        for syllable, folios in syl_folios.items():
            if len(folios) >= 2:
                cross_folio_assignments.append({
                    'triple_key': triple_key,
                    'syllable': syllable,
                    'supporting_folios': sorted(folios),
                    'n_folios': len(folios),
                })

    cross_folio_assignments.sort(key=lambda c: -c['n_folios'])
    n_cross = len(cross_folio_assignments)

    if cross_folio_assignments:
        print(f"     {n_cross} cross-folio consistent assignments:")
        for cfa in cross_folio_assignments[:10]:
            print(f"       {cfa['triple_key']} -> '{cfa['syllable']}' "
                  f"(from {cfa['n_folios']} folios: "
                  f"{', '.join(cfa['supporting_folios'])})")
    else:
        print("     No cross-folio consistent assignments found")

    # -- 5. Null control: wrong plant names from other folios --
    print("\n  5. Null control (wrong plant names) ...")

    null_valid = 0
    null_pairs = 0
    for tier_entry in all_tier_folios[:5]:
        folio = tier_entry.get('folio', '')
        label_cands = tier_entry.get('label_candidates', [])
        if not folio or not label_cands:
            continue

        # Get names from OTHER folios
        for other_folio, other_entries in folio_to_italian.items():
            if other_folio == folio:
                continue
            for ie in other_entries:
                for name in ie.get('italian_names', [])[:2]:
                    if ' ' in name:
                        continue
                    syls = ie.get('syllabified', {}).get(name, [])
                    if not syls:
                        syls = _syllabify_italian(name)
                    if not syls:
                        continue
                    for lc in label_cands[:2]:
                        token = lc.get('token', '')
                        eva_chars_raw = lc.get('eva_chars', [])
                        if not token or not eva_chars_raw:
                            continue
                        syllabic_chars = [ch for ch in eva_chars_raw
                                          if ch not in modifier_chars]
                        triple_keys = [eva_to_triple.get(ch, ch)
                                       for ch in syllabic_chars]
                        null_pairs += 1
                        compatible, _ = _check_length_compatibility(
                            len(triple_keys), len(syls))
                        if not compatible:
                            continue
                        aligns = _enumerate_alignments(triple_keys, syls)
                        for alignment in aligns:
                            nc, nf = _check_confirmed_consistency(
                                alignment, assignment)
                            if nf > 0:
                                continue
                            nu = sum(1 for k in alignment
                                     if k not in assignment
                                     or assignment[k] != alignment[k])
                            s = _score_alignment(nc, nf, nu, len(alignment))
                            if s > 0:
                                null_valid += 1
                                break

    null_rate = null_valid / max(null_pairs, 1)
    real_rate = n_valid_alignments / max(n_pairs_tested, 1)
    null_selectivity = real_rate / max(null_rate, 0.001)

    print(f"     Real alignment rate: {real_rate:.4f}")
    print(f"     Null alignment rate: {null_rate:.4f}")
    print(f"     Null selectivity: {null_selectivity:.2f}x")

    # -- 6. Verdict and save --
    if n_cross >= 3:
        verdict = f"STRONG_ITALIAN_BOTANICAL ({n_cross} cross-folio)"
    elif n_cross >= 1:
        verdict = f"PARTIAL_ITALIAN_BOTANICAL ({n_cross} cross-folio)"
    elif n_valid_alignments > 0:
        verdict = f"WEAK_ITALIAN_BOTANICAL ({n_valid_alignments} valid, 0 cross-folio)"
    else:
        verdict = "NO_ITALIAN_BOTANICAL_SIGNAL"

    elapsed = time.time() - t0

    output = {
        'n_folios_tested': n_folios_tested,
        'n_pairs_tested': n_pairs_tested,
        'n_valid_alignments': n_valid_alignments,
        'alignments': all_alignments[:100],
        'n_cross_folio_consistent': n_cross,
        'cross_folio_assignments': cross_folio_assignments,
        'null_selectivity': round(null_selectivity, 2),
        'null_rate': round(null_rate, 4),
        'real_rate': round(real_rate, 4),
        'assignment_source': assignment_source,
        'verdict': verdict,
        'runtime_seconds': round(elapsed, 1),
    }

    out_path = os.path.join(rd, 'italian_botanical_csp.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(output), f, indent=2)
    print(f"\n  Saved -> {out_path} ({elapsed:.1f}s)")
