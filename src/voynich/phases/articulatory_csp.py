"""
Phase 15.2 – Articulatory Consistency Scoring
==============================================
Add articulatory consistency (AC) as a scoring term to the feature CSP.
Three approaches are compared:

(a) Soft scoring – AC weighted by a tuneable delta, added to composite.
(b) Hard constraints – restrict each onset group to one place class.
(c) Per-onset coordinate descent – fix all but one onset group, exhaustively
    search articulatorily-consistent assignments for that group.

Dependency chain:
    feature_decode.json (Phase 14 baseline)
        → articulatory_csp.json (this step)
"""

import copy
import json
import os
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from itertools import product as itertools_product
from typing import Any, Dict, List, Optional, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    load_corpus,
    tokenize_eva_chars,
)
from voynich.core.reference import (
    EVA_VISUAL_COMPONENTS,
    PHONEME_PLACE_MAP,
    PHONEME_NUCLEUS_MAP,
    build_cv_syllable_table,
    build_triple_phoneme_hypotheses,
    load_reference_corpus,
)
from voynich.core.stats import build_ngram_lm
from voynich.phases.csp_constraints import (
    AnchorConstraint,
    PhonemeInventory,
    VerbConstraint,
    build_phoneme_inventory,
    score_dict_hit_rate,
)
from voynich.phases.csp_solver import (
    _convert,
    ac3_propagate,
    beam_search,
    decode_corpus,
    score_assignment_full,
)
from voynich.phases.feature_csp import (
    FeatureVariable,
    build_feature_variables,
    initialise_feature_domains,
    _build_anchor_constraints_triple,
    run_feature_csp_for_language,
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ArticulatoryCSPResult:
    # 2a: Baseline AC
    baseline_ac: float
    baseline_ac_details: Dict

    # 2b: Soft scoring grid search
    grid_search_results: List[Dict]
    best_delta: float
    soft_dict_hit: float
    soft_selectivity: float
    soft_ac: float

    # 2c: Hard constraint results
    hard_constraint_dict_hit: float
    hard_constraint_selectivity: float
    hard_constraint_ac: float
    hard_onset_mapping: Dict[str, str]

    # 2d: Per-onset coordinate descent
    per_onset_iterations: List[Dict]
    per_onset_dict_hit: float
    per_onset_selectivity: float
    per_onset_ac: float

    best_approach: str
    best_dict_hit: float
    best_assignment: Dict[str, str]
    gate_passed: bool
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Articulatory consistency metric
# ---------------------------------------------------------------------------

_STOPS = {'p', 'b', 't', 'd', 'k', 'g', 'c', 'q'}
_FRICATIVES = {'f', 'v', 's', 'z', 'h', 'x'}
_SONORANTS = {'m', 'n', 'l', 'r'}
_HIGH_VOWELS = {'i', 'u', 'y'}
_MID_VOWELS = {'e', 'o'}
_LOW_VOWELS = {'a'}
_ALL_VOWELS = _HIGH_VOWELS | _MID_VOWELS | _LOW_VOWELS


def _place_class(ph: str) -> str:
    if ph in _STOPS:
        return 'stop'
    if ph in _FRICATIVES:
        return 'fricative'
    if ph in _SONORANTS:
        return 'sonorant'
    if ph in _ALL_VOWELS:
        return 'vowel'
    return 'other'


def _height_class(ph: str) -> str:
    if ph in _HIGH_VOWELS:
        return 'high'
    if ph in _MID_VOWELS:
        return 'mid'
    if ph in _LOW_VOWELS:
        return 'low'
    return 'other'


def compute_articulatory_consistency(
    assignment: Dict[str, str],
) -> Tuple[float, Dict]:
    """Compute AC metric: mean onset consistency × mean nucleus consistency.

    For each onset stroke type with ≥2 triples: fraction of assigned onset
    phonemes sharing the majority place class.

    For each nucleus stroke type with ≥2 triples: fraction of assigned
    nucleus phonemes sharing the majority height class.

    Returns (ac_score, details_dict).
    """
    # Group triples by first_stroke and last_stroke
    onset_groups: Dict[str, List[str]] = {}
    nucleus_groups: Dict[str, List[str]] = {}

    for triple_key in assignment:
        parts = triple_key.split(',')
        if len(parts) != 3:
            continue
        fs, ls = parts[0], parts[1]
        onset_groups.setdefault(fs, []).append(triple_key)
        nucleus_groups.setdefault(ls, []).append(triple_key)

    # Onset consistency
    onset_scores: List[float] = []
    onset_details: List[Dict] = []
    for fs, triple_keys in onset_groups.items():
        if len(triple_keys) < 2:
            continue
        onsets: List[str] = []
        for tk in triple_keys:
            syl = assignment.get(tk, '')
            if syl:
                # First non-vowel character is the onset
                onset = ''
                for ch in syl:
                    if ch not in _ALL_VOWELS:
                        onset = ch
                        break
                if onset:
                    onsets.append(onset)
        if len(onsets) < 2:
            continue
        classes = [_place_class(o) for o in onsets]
        class_counts = Counter(classes)
        majority_count = class_counts.most_common(1)[0][1]
        consistency = majority_count / len(classes)
        onset_scores.append(consistency)
        onset_details.append({
            'first_stroke': fs,
            'onsets': onsets,
            'classes': classes,
            'consistency': round(consistency, 3),
        })

    # Nucleus consistency
    nucleus_scores: List[float] = []
    nucleus_details: List[Dict] = []
    for ls, triple_keys in nucleus_groups.items():
        if len(triple_keys) < 2:
            continue
        nuclei: List[str] = []
        for tk in triple_keys:
            syl = assignment.get(tk, '')
            if syl:
                # Last vowel character is the nucleus
                for ch in reversed(syl):
                    if ch in _ALL_VOWELS:
                        nuclei.append(ch)
                        break
        if len(nuclei) < 2:
            continue
        classes = [_height_class(n) for n in nuclei]
        class_counts = Counter(classes)
        majority_count = class_counts.most_common(1)[0][1]
        consistency = majority_count / len(classes)
        nucleus_scores.append(consistency)
        nucleus_details.append({
            'last_stroke': ls,
            'nuclei': nuclei,
            'classes': classes,
            'consistency': round(consistency, 3),
        })

    mean_onset = sum(onset_scores) / len(onset_scores) if onset_scores else 0.0
    mean_nucleus = sum(nucleus_scores) / len(nucleus_scores) if nucleus_scores else 0.0
    ac_score = mean_onset * mean_nucleus

    details = {
        'mean_onset_consistency': round(mean_onset, 4),
        'mean_nucleus_consistency': round(mean_nucleus, 4),
        'ac_score': round(ac_score, 4),
        'onset_details': onset_details,
        'nucleus_details': nucleus_details,
    }
    return ac_score, details


# ---------------------------------------------------------------------------
# Random baseline computation
# ---------------------------------------------------------------------------

def _compute_random_dict_hit(
    variables_keys: List[str],
    all_syls: List[str],
    voynich_tokens: List[str],
    eva_to_triple: Dict[str, str],
    ref_word_set: set,
    n_trials: int = 50,
    seed: int = 42,
) -> float:
    rng = random.Random(seed)
    hits_list: List[float] = []
    for _ in range(n_trials):
        rand_map = {k: rng.choice(all_syls) for k in variables_keys}
        decoded = decode_corpus(voynich_tokens, rand_map, eva_to_triple, max_tokens=500)
        hits = sum(1 for w in decoded if w in ref_word_set)
        hits_list.append(hits / len(decoded) if decoded else 0.0)
    return sum(hits_list) / len(hits_list) if hits_list else 0.001


# ---------------------------------------------------------------------------
# Hard constraint approach
# ---------------------------------------------------------------------------

# Default onset-to-place mapping derived from PHONEME_PLACE_MAP
_ONSET_PLACE_MAPPING = {
    'ascender': 'stop',       # t, k, p, d, g, b
    'connector': 'stop',      # b, p → stops (v, m, f are minority)
    'crossbar': 'fricative',  # x, h, f
    'loop': 'sonorant',       # l, r, n (+ vowels)
    'open_curve': 'fricative', # c, s, sc, h
    'sigmoid': 'fricative',   # s, z, sc
    'vertical': 'sonorant',   # m, n, d, l
}


def _get_consistent_syllables(
    place_class: str,
    all_syllables: List[str],
) -> List[str]:
    """Return syllables whose onset consonant belongs to the given place class."""
    result = []
    for syl in all_syllables:
        if not syl:
            continue
        # Pure vowel syllables are always allowed
        if syl[0] in _ALL_VOWELS:
            if place_class in ('sonorant', 'vowel'):
                result.append(syl)
            continue
        onset = syl[0]
        if _place_class(onset) == place_class:
            result.append(syl)
    return result


# ---------------------------------------------------------------------------
# Per-onset coordinate descent
# ---------------------------------------------------------------------------

def _per_onset_descent(
    assignment: Dict[str, str],
    voynich_tokens: List[str],
    eva_to_triple: Dict[str, str],
    ref_word_set: set,
    all_syls: List[str],
    max_rounds: int = 5,
) -> Tuple[Dict[str, str], float, List[Dict]]:
    """Coordinate descent: for each onset group, exhaustively search
    articulatorily-consistent assignments while fixing all others."""

    # Group triples by first_stroke
    onset_groups: Dict[str, List[str]] = {}
    for triple_key in assignment:
        parts = triple_key.split(',')
        if len(parts) == 3:
            onset_groups.setdefault(parts[0], []).append(triple_key)

    current = dict(assignment)
    iteration_log: List[Dict] = []

    for round_idx in range(max_rounds):
        improved = False
        for onset_type, triple_keys in onset_groups.items():
            if len(triple_keys) < 1:
                continue

            # Determine the place class for this onset type
            place_class = _ONSET_PLACE_MAPPING.get(onset_type, 'stop')
            consistent_syls = _get_consistent_syllables(place_class, all_syls)

            if not consistent_syls:
                continue

            # Limit candidates per triple to avoid combinatorial explosion
            max_per_triple = min(len(consistent_syls), 5)
            candidates = consistent_syls[:max_per_triple]

            # If too many triples, limit exhaustive search
            if len(triple_keys) > 6:
                # Fall back to greedy per-triple
                for tk in triple_keys:
                    best_syl = current[tk]
                    best_hit = -1.0
                    for syl in candidates:
                        test = dict(current)
                        test[tk] = syl
                        decoded = decode_corpus(
                            voynich_tokens, test, eva_to_triple, max_tokens=500
                        )
                        hits = sum(1 for w in decoded if w in ref_word_set)
                        hit_rate = hits / len(decoded) if decoded else 0.0
                        if hit_rate > best_hit:
                            best_hit = hit_rate
                            best_syl = syl
                    if best_syl != current[tk]:
                        current[tk] = best_syl
                        improved = True
            else:
                # Exhaustive search over all combinations for this onset group
                best_combo = tuple(current.get(tk, candidates[0]) for tk in triple_keys)
                best_hit = -1.0

                for combo in itertools_product(candidates, repeat=len(triple_keys)):
                    test = dict(current)
                    for tk, syl in zip(triple_keys, combo):
                        test[tk] = syl
                    decoded = decode_corpus(
                        voynich_tokens, test, eva_to_triple, max_tokens=500
                    )
                    hits = sum(1 for w in decoded if w in ref_word_set)
                    hit_rate = hits / len(decoded) if decoded else 0.0
                    if hit_rate > best_hit:
                        best_hit = hit_rate
                        best_combo = combo

                for tk, syl in zip(triple_keys, best_combo):
                    if current[tk] != syl:
                        improved = True
                    current[tk] = syl

            iteration_log.append({
                'round': round_idx,
                'onset_type': onset_type,
                'n_triples': len(triple_keys),
                'place_class': place_class,
                'dict_hit': round(best_hit, 4) if best_hit >= 0 else None,
            })

        if not improved:
            break

    # Final dict_hit
    decoded = decode_corpus(voynich_tokens, current, eva_to_triple, max_tokens=2000)
    final_hits = sum(1 for w in decoded if w in ref_word_set)
    final_dict_hit = final_hits / len(decoded) if decoded else 0.0

    return current, final_dict_hit, iteration_log


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_articulatory_csp() -> None:
    """Step 15.2: Articulatory consistency CSP scoring."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE 15.2: Articulatory Consistency Scoring")
    print("=" * 70)

    rd = _results_dir()

    # Load Phase 14 results
    fd_path = os.path.join(rd, 'feature_decode.json')
    if not os.path.exists(fd_path):
        print("  [SKIP] feature_decode.json not found — run feature-decode first")
        return

    with open(fd_path) as f:
        fd_data = json.load(f)

    best_assignment = fd_data.get('best_assignment', {})
    if not best_assignment:
        print("  [SKIP] No best assignment in feature_decode.json")
        return

    baseline_dict_hit = fd_data.get('best_dict_hit', 0.0)

    # Load corpus
    corpus = load_corpus(verbose=False)
    tokens = corpus.get_tokens(language='A', paragraph_only=True)
    if not tokens:
        print("  [SKIP] No Language A tokens found")
        return

    eva_to_triple = build_eva_to_triple_lookup()

    # Load reference corpus
    ref_corpus = load_reference_corpus(verbose=False)
    ref_tokens = ref_corpus.get_combined_tokens('latin')
    ref_word_set = set(w.lower() for w in ref_tokens if len(w) >= 2)
    inventory = build_phoneme_inventory('latin', ref_corpus)
    lm = build_ngram_lm(ref_tokens[:10000], order=3, smoothing=0.01)
    all_syls = build_cv_syllable_table('latin')

    # Glyph frequencies
    glyph_freq: Counter = Counter()
    for token in tokens:
        for ch in tokenize_eva_chars(token):
            glyph_freq[ch] += 1

    # Load anchors
    rosetta_path = os.path.join(rd, 'rosetta_selection.json')
    anchors: List[AnchorConstraint] = []
    if os.path.exists(rosetta_path):
        with open(rosetta_path) as f:
            rosetta_data = json.load(f)
        anchors = _build_anchor_constraints_triple(rosetta_data, eva_to_triple)

    # ─── 2a: Baseline AC ───
    print("\n  2a: Computing baseline articulatory consistency ...")
    baseline_ac, baseline_ac_details = compute_articulatory_consistency(best_assignment)
    print(f"      Baseline AC: {baseline_ac:.4f}")
    print(f"      Mean onset consistency: {baseline_ac_details['mean_onset_consistency']:.4f}")
    print(f"      Mean nucleus consistency: {baseline_ac_details['mean_nucleus_consistency']:.4f}")

    # Random baseline for selectivity
    variables_keys = list(best_assignment.keys())
    random_baseline = _compute_random_dict_hit(
        variables_keys, all_syls, tokens, eva_to_triple, ref_word_set,
    )

    # ─── 2b: Soft scoring grid search ───
    print("\n  2b: Grid search over AC weight δ ...")
    deltas = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
    grid_results: List[Dict] = []
    best_delta = 0.0
    best_soft_product = 0.0
    soft_dict_hit = baseline_dict_hit
    soft_ac = baseline_ac
    soft_assignment = dict(best_assignment)

    for delta in deltas:
        print(f"      δ = {delta:.2f} ... ", end='', flush=True)

        # Build variables fresh each time
        hypothesis_map = build_triple_phoneme_hypotheses('latin', all_syls)
        variables = build_feature_variables(eva_to_triple, glyph_freq, inventory, hypothesis_map)
        variables = initialise_feature_domains(variables, inventory, hypothesis_map, anchors)

        # Run beam search
        solutions = beam_search(
            variables=variables,
            lm=lm,
            voynich_tokens=tokens,
            eva_to_cell=eva_to_triple,
            anchors=anchors,
            inventory=inventory,
            ref_word_set=ref_word_set,
            beam_width=80,
            max_solutions=10,
            seed=42 + int(delta * 100),
        )

        if not solutions:
            print("no solutions")
            grid_results.append({
                'delta': delta, 'dict_hit': 0.0, 'selectivity': 0.0,
                'ac': 0.0, 'product': 0.0,
            })
            continue

        # Re-score top solutions with AC penalty and pick best
        best_for_delta = None
        best_combined = float('inf')
        for sol in solutions:
            ac, _ = compute_articulatory_consistency(sol.mapping)
            adjusted_score = sol.score - delta * ac  # lower is better, AC bonus
            if adjusted_score < best_combined:
                best_combined = adjusted_score
                best_for_delta = sol
                best_for_delta_ac = ac

        sol = best_for_delta
        hit = sol.dict_hit_rate
        selectivity = hit / max(random_baseline, 0.001)
        ac_val = best_for_delta_ac
        product_val = hit * ac_val

        print(f"dict_hit={hit:.3f}, AC={ac_val:.3f}, product={product_val:.4f}")

        grid_results.append({
            'delta': delta,
            'dict_hit': round(hit, 4),
            'selectivity': round(selectivity, 2),
            'ac': round(ac_val, 4),
            'product': round(product_val, 4),
        })

        if product_val > best_soft_product:
            best_soft_product = product_val
            best_delta = delta
            soft_dict_hit = hit
            soft_ac = ac_val
            soft_assignment = dict(sol.mapping)

    soft_selectivity = soft_dict_hit / max(random_baseline, 0.001)
    print(f"      Best δ = {best_delta:.2f}: dict_hit={soft_dict_hit:.3f}, AC={soft_ac:.3f}")

    # ─── 2c: Hard constraints ───
    print("\n  2c: Hard articulatory constraints ...")
    hypothesis_map = build_triple_phoneme_hypotheses('latin', all_syls)
    hard_variables = build_feature_variables(eva_to_triple, glyph_freq, inventory, hypothesis_map)
    hard_variables = initialise_feature_domains(hard_variables, inventory, hypothesis_map, anchors)

    # Restrict domains based on onset place class
    for var in hard_variables:
        parts = var.cell_key.split(',')
        if len(parts) == 3:
            onset_type = parts[0]
            place_class = _ONSET_PLACE_MAPPING.get(onset_type, 'stop')
            consistent = _get_consistent_syllables(place_class, var.domain)
            if consistent:
                var.domain = consistent
            # If no consistent syllables in domain, keep original (safety)

    # AC-3 propagation
    solvable, hard_variables = ac3_propagate(hard_variables)

    if solvable:
        hard_solutions = beam_search(
            variables=hard_variables,
            lm=lm,
            voynich_tokens=tokens,
            eva_to_cell=eva_to_triple,
            anchors=anchors,
            inventory=inventory,
            ref_word_set=ref_word_set,
            beam_width=80,
            max_solutions=10,
            seed=42,
        )
        if hard_solutions:
            hard_best = hard_solutions[0]
            hard_dict_hit = hard_best.dict_hit_rate
            hard_ac, _ = compute_articulatory_consistency(hard_best.mapping)
            hard_selectivity = hard_dict_hit / max(random_baseline, 0.001)
            hard_assignment = dict(hard_best.mapping)
        else:
            hard_dict_hit = 0.0
            hard_ac = 0.0
            hard_selectivity = 0.0
            hard_assignment = {}
    else:
        hard_dict_hit = 0.0
        hard_ac = 0.0
        hard_selectivity = 0.0
        hard_assignment = {}

    print(f"      Hard: dict_hit={hard_dict_hit:.3f}, AC={hard_ac:.3f}, selectivity={hard_selectivity:.2f}x")

    # ─── 2d: Per-onset coordinate descent ───
    print("\n  2d: Per-onset coordinate descent ...")
    descent_assignment, descent_dict_hit, descent_log = _per_onset_descent(
        best_assignment, tokens, eva_to_triple, ref_word_set, all_syls,
        max_rounds=3,
    )
    descent_ac, _ = compute_articulatory_consistency(descent_assignment)
    descent_selectivity = descent_dict_hit / max(random_baseline, 0.001)
    print(f"      Per-onset: dict_hit={descent_dict_hit:.3f}, AC={descent_ac:.3f}, selectivity={descent_selectivity:.2f}x")

    # ─── Compare approaches ───
    approaches = {
        'soft': (soft_dict_hit, soft_selectivity, soft_ac, soft_assignment),
        'hard': (hard_dict_hit, hard_selectivity, hard_ac, hard_assignment),
        'per_onset': (descent_dict_hit, descent_selectivity, descent_ac, descent_assignment),
    }

    best_approach = 'soft'
    best_overall_hit = soft_dict_hit
    best_overall_assignment = soft_assignment

    for name, (hit, sel, ac, asgn) in approaches.items():
        if hit > best_overall_hit and sel >= 1.5:
            best_overall_hit = hit
            best_approach = name
            best_overall_assignment = asgn

    # If nothing beats baseline, keep Phase 14 result
    if best_overall_hit < baseline_dict_hit:
        best_approach = 'baseline'
        best_overall_hit = baseline_dict_hit
        best_overall_assignment = dict(best_assignment)

    gate_passed = True  # AC step always produces a result; gate is informational

    elapsed = time.time() - t0
    best_final_ac, _ = compute_articulatory_consistency(best_overall_assignment)

    verdict = (
        f"Best approach: {best_approach}. "
        f"dict_hit={best_overall_hit:.1%}, AC={best_final_ac:.3f}. "
        f"Baseline was {baseline_dict_hit:.1%}, AC={baseline_ac:.3f}."
    )

    result = ArticulatoryCSPResult(
        baseline_ac=round(baseline_ac, 4),
        baseline_ac_details=baseline_ac_details,
        grid_search_results=grid_results,
        best_delta=best_delta,
        soft_dict_hit=round(soft_dict_hit, 4),
        soft_selectivity=round(soft_selectivity, 2),
        soft_ac=round(soft_ac, 4),
        hard_constraint_dict_hit=round(hard_dict_hit, 4),
        hard_constraint_selectivity=round(hard_selectivity, 2),
        hard_constraint_ac=round(hard_ac, 4),
        hard_onset_mapping=_ONSET_PLACE_MAPPING,
        per_onset_iterations=descent_log,
        per_onset_dict_hit=round(descent_dict_hit, 4),
        per_onset_selectivity=round(descent_selectivity, 2),
        per_onset_ac=round(descent_ac, 4),
        best_approach=best_approach,
        best_dict_hit=round(best_overall_hit, 4),
        best_assignment=best_overall_assignment,
        gate_passed=gate_passed,
        verdict=verdict,
        runtime_seconds=round(elapsed, 2),
    )

    out_path = os.path.join(rd, 'articulatory_csp.json')
    with open(out_path, 'w') as f:
        json.dump(asdict(result), f, indent=2, default=_convert)

    print(f"\n  {verdict}")
    print(f"\n  → {out_path}")
