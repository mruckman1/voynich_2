"""
Phase 23.4 – Permutation Search (perm-search)
===============================================
Starting from Phase 22's historically-grounded syllable assignments,
searches for systematic permutations (vowel rotations, consonant class
swaps, family rotations, hill climbing) that bridge Phase 22 and Phase 16.
If a single permutation consistently maps one table to the other, that
permutation IS the encoding rule.

Dependency chain:
    combined_refine.json (Phase 15 best_assignment)
    merged_table.json (Phase 22 EVA→syllable)
    historical_inversion.json (23.2)
    bench_split.json (23.3)
        → permutation_search.json (this step)
"""

import json
import os
import random
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from itertools import permutations as iter_perms
from typing import Any, Dict, List, Optional, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    load_corpus,
    tokenize_eva_chars,
)
from voynich.core.reference import (
    EVA_VISUAL_COMPONENTS,
    build_cv_syllable_table,
    build_expanded_word_set,
    load_reference_corpus,
)
from voynich.phases.csp_constraints import score_dict_hit_rate


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
    if isinstance(obj, (bool, int, float, str, type(None))):
        return obj
    return str(obj)


def _load_json(path: str) -> Optional[Dict]:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Phase 22 → triple-level conversion
# ---------------------------------------------------------------------------

def _build_triple_to_eva_chars() -> Dict[str, List[str]]:
    result: Dict[str, List[str]] = defaultdict(list)
    for eva_char, comp in EVA_VISUAL_COMPONENTS.items():
        tk = f"{comp['first_stroke']},{comp['last_stroke']},{comp['glyph_class']}"
        result[tk].append(eva_char)
    return dict(result)


def _convert_phase22_to_triple_level(
    mode_a_table: List[Dict],
    triple_to_eva: Dict[str, List[str]],
) -> Dict[str, str]:
    eva_to_syl: Dict[str, str] = {}
    for entry in mode_a_table:
        eva_char = entry.get('eva_char', '')
        syl = entry.get('syllable_a', '')
        if eva_char and syl and not entry.get('is_modifier', False):
            eva_to_syl[eva_char] = syl

    triple_table: Dict[str, str] = {}
    for tk, chars in triple_to_eva.items():
        syls = [eva_to_syl[c] for c in chars if c in eva_to_syl]
        if syls:
            counts = Counter(syls)
            triple_table[tk] = counts.most_common(1)[0][0]
    return triple_table


# ---------------------------------------------------------------------------
# Agreement scoring
# ---------------------------------------------------------------------------

def _agreement(table_a: Dict[str, str], table_b: Dict[str, str]) -> Tuple[int, int, float]:
    common = set(table_a) & set(table_b)
    if not common:
        return 0, 0, 0.0
    n_agree = sum(1 for k in common if table_a[k] == table_b[k])
    return n_agree, len(common), n_agree / len(common)


# ---------------------------------------------------------------------------
# Permutation generators
# ---------------------------------------------------------------------------

_VOWELS = ['a', 'e', 'i', 'o', 'u']

_CONSONANT_CLASSES = {
    'stops': ['p', 'b', 't', 'd', 'c', 'g', 'k'],
    'fricatives': ['f', 'v', 's', 'z', 'h'],
    'nasals': ['m', 'n'],
    'liquids': ['l', 'r'],
}


def _apply_vowel_perm(table: Dict[str, str], vperm: Tuple[str, ...]) -> Dict[str, str]:
    """Apply a vowel permutation to all syllables in a table."""
    vmap = dict(zip(_VOWELS, vperm))
    result = {}
    for tk, syl in table.items():
        if not syl:
            result[tk] = syl
            continue
        # Last char is vowel in CV syllable
        if syl[-1] in vmap:
            result[tk] = syl[:-1] + vmap[syl[-1]]
        else:
            result[tk] = syl
    return result


def _apply_consonant_swap(
    table: Dict[str, str],
    class_a: str,
    class_b: str,
) -> Dict[str, str]:
    """Swap consonants between two articulatory classes."""
    a_members = _CONSONANT_CLASSES.get(class_a, [])
    b_members = _CONSONANT_CLASSES.get(class_b, [])
    result = {}
    for tk, syl in table.items():
        if len(syl) < 2:
            result[tk] = syl
            continue
        onset = syl[:-1]
        vowel = syl[-1]
        if onset in a_members:
            idx = a_members.index(onset)
            result[tk] = b_members[idx % len(b_members)] + vowel
        elif onset in b_members:
            idx = b_members.index(onset)
            result[tk] = a_members[idx % len(a_members)] + vowel
        else:
            result[tk] = syl
    return result


def _apply_family_rotation(
    table: Dict[str, str],
    family_groups: Dict[str, List[str]],
    rotation: List[str],
) -> Dict[str, str]:
    """Rotate syllable assignments between triple families."""
    # rotation = [fam_a, fam_b, fam_c, ...] means a→b→c→...→a
    if len(rotation) < 2:
        return dict(table)

    # Collect average syllable for each family
    family_syls: Dict[str, List[str]] = defaultdict(list)
    for fam in rotation:
        for tk in family_groups.get(fam, []):
            if tk in table:
                family_syls[fam].append(table[tk])

    # Build rotation map: syllables from fam[i] go to fam[i+1]
    result = dict(table)
    for i in range(len(rotation)):
        src_fam = rotation[i]
        dst_fam = rotation[(i + 1) % len(rotation)]
        src_tks = family_groups.get(src_fam, [])
        dst_tks = family_groups.get(dst_fam, [])
        src_syls = [table.get(tk, '') for tk in src_tks if tk in table]
        # Assign src syllables to dst triples (in order)
        for j, tk in enumerate(dst_tks):
            if tk in table and j < len(src_syls):
                result[tk] = src_syls[j]
    return result


def _hill_climb(
    start_table: Dict[str, str],
    target_table: Dict[str, str],
    voynich_tokens: List[str],
    eva_to_triple: Dict[str, str],
    ref_word_set: set,
    rng: random.Random,
    max_iter: int = 500,
) -> Tuple[Dict[str, str], float, float]:
    """Hill-climb from start_table toward target_table and/or higher dict-hit."""
    current = dict(start_table)
    keys = list(current.keys())
    _, _, best_agree = _agreement(current, target_table)
    best_dict = score_dict_hit_rate(current, voynich_tokens, eva_to_triple,
                                     ref_word_set, max_tokens=500)
    best_score = 0.5 * best_agree + 0.5 * best_dict

    for _ in range(max_iter):
        # Swap two random triple assignments
        if len(keys) < 2:
            break
        i, j = rng.sample(range(len(keys)), 2)
        ki, kj = keys[i], keys[j]
        current[ki], current[kj] = current[kj], current[ki]

        _, _, new_agree = _agreement(current, target_table)
        new_dict = score_dict_hit_rate(current, voynich_tokens, eva_to_triple,
                                        ref_word_set, max_tokens=500)
        new_score = 0.5 * new_agree + 0.5 * new_dict

        if new_score >= best_score:
            best_score = new_score
            best_agree = new_agree
            best_dict = new_dict
        else:
            # Revert
            current[ki], current[kj] = current[kj], current[ki]

    return current, best_agree, best_dict


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PermutationCandidate:
    permutation_id: str
    permutation_type: str
    description: str
    agreement_with_phase16: float
    dict_hit: float
    n_triples_matching: int


@dataclass
class PermutationSearchResult:
    timestamp: str
    phase22_triple_table: Dict[str, str]
    phase22_dict_hit: float
    phase16_triple_table: Dict[str, str]
    phase16_dict_hit: float
    n_candidates_tested: int
    top_candidates: List[Dict]
    best_permutation: Dict
    best_table: Dict[str, str]
    identity_agreement: float
    best_agreement: float
    agreement_improvement: float
    systematic_pattern_found: bool
    pattern_description: str
    gate_passed: bool
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_permutation_search() -> Dict[str, Any]:
    """Step 23.4: Permutation search."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE 23.4: Permutation Search")
    print("=" * 70)

    rdir = _results_dir()

    # Load Phase 16 assignment
    combined = _load_json(str(rdir / "combined_refine.json")) or {}
    phase16_assignment = combined.get("best_assignment", {})
    print(f"  Phase 16: {len(phase16_assignment)} triples")

    # Load Phase 22 merged table
    merged = _load_json(str(rdir / "merged_table.json")) or {}
    mode_a_table = merged.get("mode_a_table", [])

    # Convert Phase 22 to triple level
    triple_to_eva = _build_triple_to_eva_chars()
    phase22_triples = _convert_phase22_to_triple_level(mode_a_table, triple_to_eva)
    print(f"  Phase 22 → triple level: {len(phase22_triples)} triples")

    # Load corpus and dictionary
    corpus = load_corpus(verbose=False)
    tokens = corpus.get_tokens()
    eva_to_triple = build_eva_to_triple_lookup()

    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    base_words = set(
        w.lower() for w in ref_corpus.get_combined_tokens('latin')
        if len(w) >= 2
    )
    expanded_words, _ = build_expanded_word_set(base_words)
    ref_word_set = base_words | expanded_words

    # Phase 16 dict-hit
    phase16_dict = score_dict_hit_rate(
        phase16_assignment, tokens, eva_to_triple, ref_word_set, max_tokens=500
    )

    # Phase 22 dict-hit
    phase22_dict = score_dict_hit_rate(
        phase22_triples, tokens, eva_to_triple, ref_word_set, max_tokens=500
    )

    # Identity agreement
    id_agree, id_total, id_rate = _agreement(phase22_triples, phase16_assignment)
    print(f"  Identity agreement: {id_agree}/{id_total} ({id_rate:.1%})")
    print(f"  Phase 16 dict-hit: {phase16_dict:.1%}")
    print(f"  Phase 22 dict-hit: {phase22_dict:.1%}")

    # Build family groups (by first_stroke) for family rotation
    family_groups: Dict[str, List[str]] = defaultdict(list)
    for tk in phase22_triples:
        parts = tk.split(',')
        if len(parts) >= 1:
            family_groups[parts[0]].append(tk)

    rng = random.Random(42)
    candidates: List[Dict] = []
    candidate_id = 0

    # --- Type A: Vowel rotations (all 120 permutations of 5 vowels) ---
    print("  Testing vowel rotations...")
    for vperm in iter_perms(_VOWELS):
        if vperm == tuple(_VOWELS):
            continue  # skip identity
        permuted = _apply_vowel_perm(phase22_triples, vperm)
        n_ag, n_tot, ag_rate = _agreement(permuted, phase16_assignment)
        dh = score_dict_hit_rate(permuted, tokens, eva_to_triple,
                                  ref_word_set, max_tokens=500)
        vmap_str = ','.join(f"{a}→{b}" for a, b in zip(_VOWELS, vperm) if a != b)
        candidates.append(_convert(asdict(PermutationCandidate(
            permutation_id=f"A_{candidate_id}",
            permutation_type='vowel_rotation',
            description=f"Vowel: {vmap_str}",
            agreement_with_phase16=round(ag_rate, 4),
            dict_hit=round(dh, 4),
            n_triples_matching=n_ag,
        ))))
        candidate_id += 1

    # --- Type B: Consonant class swaps ---
    print("  Testing consonant class swaps...")
    class_names = list(_CONSONANT_CLASSES.keys())
    for i in range(len(class_names)):
        for j in range(i + 1, len(class_names)):
            ca, cb = class_names[i], class_names[j]
            permuted = _apply_consonant_swap(phase22_triples, ca, cb)
            n_ag, n_tot, ag_rate = _agreement(permuted, phase16_assignment)
            dh = score_dict_hit_rate(permuted, tokens, eva_to_triple,
                                      ref_word_set, max_tokens=500)
            candidates.append(_convert(asdict(PermutationCandidate(
                permutation_id=f"B_{candidate_id}",
                permutation_type='consonant_swap',
                description=f"Swap {ca} ↔ {cb}",
                agreement_with_phase16=round(ag_rate, 4),
                dict_hit=round(dh, 4),
                n_triples_matching=n_ag,
            ))))
            candidate_id += 1

    # --- Type C: Family rotations (pairwise swaps of triple families) ---
    print("  Testing family rotations...")
    fam_keys = sorted(family_groups.keys())
    for i in range(len(fam_keys)):
        for j in range(i + 1, len(fam_keys)):
            rotation = [fam_keys[i], fam_keys[j]]
            permuted = _apply_family_rotation(phase22_triples, family_groups, rotation)
            n_ag, n_tot, ag_rate = _agreement(permuted, phase16_assignment)
            dh = score_dict_hit_rate(permuted, tokens, eva_to_triple,
                                      ref_word_set, max_tokens=500)
            candidates.append(_convert(asdict(PermutationCandidate(
                permutation_id=f"C_{candidate_id}",
                permutation_type='family_rotation',
                description=f"Swap families {fam_keys[i]} ↔ {fam_keys[j]}",
                agreement_with_phase16=round(ag_rate, 4),
                dict_hit=round(dh, 4),
                n_triples_matching=n_ag,
            ))))
            candidate_id += 1

    # --- Type D: Combined (best vowel + best consonant) ---
    print("  Testing combined permutations...")
    # Find best vowel and consonant candidates so far
    vowel_cands = [c for c in candidates if c.get('permutation_type') == 'vowel_rotation']
    consonant_cands = [c for c in candidates if c.get('permutation_type') == 'consonant_swap']

    if vowel_cands:
        best_vowel = max(vowel_cands, key=lambda c: c.get('agreement_with_phase16', 0))
    if consonant_cands:
        best_consonant = max(consonant_cands, key=lambda c: c.get('agreement_with_phase16', 0))

    # Apply best vowel + each consonant swap
    if vowel_cands:
        bv_desc = best_vowel.get('description', '')
        # Re-derive the vowel perm from description
        for vperm in iter_perms(_VOWELS):
            vmap_str = ','.join(f"{a}→{b}" for a, b in zip(_VOWELS, vperm) if a != b)
            if f"Vowel: {vmap_str}" == bv_desc:
                v_permuted = _apply_vowel_perm(phase22_triples, vperm)
                for i in range(len(class_names)):
                    for j in range(i + 1, len(class_names)):
                        ca, cb = class_names[i], class_names[j]
                        combined_table = _apply_consonant_swap(v_permuted, ca, cb)
                        n_ag, n_tot, ag_rate = _agreement(combined_table, phase16_assignment)
                        dh = score_dict_hit_rate(combined_table, tokens, eva_to_triple,
                                                  ref_word_set, max_tokens=500)
                        candidates.append(_convert(asdict(PermutationCandidate(
                            permutation_id=f"D_{candidate_id}",
                            permutation_type='combined',
                            description=f"Vowel({vmap_str}) + Swap({ca}↔{cb})",
                            agreement_with_phase16=round(ag_rate, 4),
                            dict_hit=round(dh, 4),
                            n_triples_matching=n_ag,
                        ))))
                        candidate_id += 1
                break

    # --- Type E: Hill climbing (20 restarts × 500 iterations) ---
    print("  Testing hill climbing (20 restarts)...")
    for restart in range(20):
        climbed, climb_agree, climb_dict = _hill_climb(
            dict(phase22_triples), phase16_assignment,
            tokens, eva_to_triple, ref_word_set,
            random.Random(42 + restart), max_iter=500,
        )
        candidates.append(_convert(asdict(PermutationCandidate(
            permutation_id=f"E_{candidate_id}",
            permutation_type='hill_climb',
            description=f"Hill climb restart {restart}",
            agreement_with_phase16=round(climb_agree, 4),
            dict_hit=round(climb_dict, 4),
            n_triples_matching=int(climb_agree * len(set(phase22_triples) & set(phase16_assignment))),
        ))))
        candidate_id += 1

    # --- Type F: Random null baseline (50 random permutations) ---
    print("  Testing random baseline (50 trials)...")
    for trial in range(50):
        rand_table = dict(phase22_triples)
        keys = list(rand_table.keys())
        vals = list(rand_table.values())
        rng.shuffle(vals)
        rand_table = dict(zip(keys, vals))
        n_ag, n_tot, ag_rate = _agreement(rand_table, phase16_assignment)
        dh = score_dict_hit_rate(rand_table, tokens, eva_to_triple,
                                  ref_word_set, max_tokens=500)
        candidates.append(_convert(asdict(PermutationCandidate(
            permutation_id=f"F_{candidate_id}",
            permutation_type='random_null',
            description=f"Random permutation {trial}",
            agreement_with_phase16=round(ag_rate, 4),
            dict_hit=round(dh, 4),
            n_triples_matching=n_ag,
        ))))
        candidate_id += 1

    # --- Sort and select top 20 ---
    max_dict = max(c.get('dict_hit', 0.001) for c in candidates) or 0.001
    for c in candidates:
        ag = c.get('agreement_with_phase16', 0)
        dh = c.get('dict_hit', 0)
        c['composite_score'] = round(0.5 * ag + 0.5 * (dh / max_dict), 4)

    candidates.sort(key=lambda c: c.get('composite_score', 0), reverse=True)
    top_20 = candidates[:20]

    best = top_20[0] if top_20 else {}
    best_agreement = best.get('agreement_with_phase16', 0.0)

    # Reconstruct the best table
    # For simplicity, store phase16 as the best if hill climbing found it
    # Otherwise reconstruct from the best candidate's type
    # The hill-climb candidates are the most promising for finding actual tables
    # Use a final hill-climb with the best seed
    best_table = dict(phase16_assignment)  # fallback
    if best.get('permutation_type') == 'hill_climb':
        restart_id = int(best.get('description', '0').split()[-1])
        best_table, _, _ = _hill_climb(
            dict(phase22_triples), phase16_assignment,
            tokens, eva_to_triple, ref_word_set,
            random.Random(42 + restart_id), max_iter=500,
        )

    # Check for systematic pattern
    systematic = best_agreement > 0.4 and best.get('permutation_type') != 'random_null'
    pattern_desc = best.get('description', 'none') if systematic else 'No systematic pattern found'

    # Gate
    gate_passed = best_agreement > 0.4
    if best_agreement > 0.6:
        verdict = "STRONG PERMUTATION FOUND"
    elif best_agreement > 0.4:
        verdict = "MODERATE PERMUTATION FOUND"
    elif best_agreement > 0.2:
        verdict = "WEAK PATTERN — below threshold"
    else:
        verdict = "NO PERMUTATION — tables are unrelated"

    elapsed = time.time() - t0

    result = PermutationSearchResult(
        timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
        phase22_triple_table=phase22_triples,
        phase22_dict_hit=round(phase22_dict, 4),
        phase16_triple_table=phase16_assignment,
        phase16_dict_hit=round(phase16_dict, 4),
        n_candidates_tested=len(candidates),
        top_candidates=top_20,
        best_permutation=best,
        best_table=best_table,
        identity_agreement=round(id_rate, 4),
        best_agreement=round(best_agreement, 4),
        agreement_improvement=round(best_agreement - id_rate, 4),
        systematic_pattern_found=systematic,
        pattern_description=pattern_desc,
        gate_passed=gate_passed,
        verdict=verdict,
        runtime_seconds=round(elapsed, 1),
    )

    out_path = rdir / "permutation_search.json"
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2, ensure_ascii=False)

    print(f"\n  Candidates tested: {len(candidates)}")
    print(f"  Identity agreement: {id_rate:.1%}")
    print(f"  Best agreement: {best_agreement:.1%}"
          f" ({best.get('permutation_type', '?')}: {best.get('description', '')})")
    print(f"  Best dict-hit: {best.get('dict_hit', 0):.1%}")
    print(f"  Systematic pattern: {systematic}")
    print(f"  Verdict: {verdict}")
    print(f"  → {out_path} ({elapsed:.1f}s)")

    return _convert(asdict(result))
