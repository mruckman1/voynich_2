"""
Phase 11.5.2-3 – Inherent vowel + graduated relaxation sweep
=============================================================
Step 11.5.2: Tests each of {a, e, i} as an inherent vowel for
             cells that appear to carry no explicit nucleus marker.
Step 11.5.3: Graduates the syllable inventory from strict CV (Level 0)
             through CVC and CCV extensions (Levels 1-5), expanding
             only the high-error cells identified in Phase 11.5.1.

Decision gates:
  best_dict_hit_rate >= 0.15   OR   improvement_factor >= 1.35
"""

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_cell_lookup,
    load_corpus,
)
from voynich.core.reference import (
    INHERENT_VOWEL_CANDIDATES,
    build_cv_syllable_table,
    build_cvc_syllable_table,
    load_reference_corpus,
)
from voynich.core.stats import build_ngram_lm
from voynich.phases.csp_constraints import (
    build_anchor_constraints,
    build_phoneme_inventory,
    score_cross_entropy,
    score_dict_hit_rate,
)
from voynich.phases.csp_decode import run_csp_for_language
from voynich.phases.csp_solver import (
    CSPAssignment,
    _convert,
    ac3_propagate,
    beam_search,
    build_csp_variables,
    initialise_domains,
    score_assignment_full,
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class InherentVowelResult:
    """Result of testing one inherent vowel candidate."""
    vowel: str
    dict_hit_rate: float
    cross_entropy: float
    best_assignment: Dict[str, str]


@dataclass
class RelaxationLevelResult:
    """Result at one relaxation level."""
    level: int
    description: str
    n_syllables: int
    dict_hit_rate: float
    cross_entropy: float
    selectivity: float
    anchor_match_count: int
    best_assignment: Dict[str, str]
    runtime_seconds: float


@dataclass
class CSPRefinementResult:
    """Full Phase 11.5.2-3 output."""
    inherent_vowel_results: List[Dict]
    best_inherent_vowel: str
    inherent_vowel_improvement: float    # delta dict_hit over baseline
    relaxation_results: List[Dict]
    best_relaxation_level: int
    best_dict_hit_rate: float
    baseline_dict_hit_rate: float        # Level 0 (Phase 11)
    improvement_factor: float
    high_error_cells: List[str]
    final_assignment: Dict[str, str]
    gate_passed: bool
    verdict: str


# ---------------------------------------------------------------------------
# Level descriptions
# ---------------------------------------------------------------------------

LEVEL_DESCRIPTIONS = {
    0: 'Strict CV only (~75 syllables for Latin)',
    1: 'CV + inherent vowel singletons',
    2: 'CV + top-25 CVC syllables',
    3: 'CV + full CVC extensions',
    4: 'CV + CVC + top-25 CCV syllables',
    5: 'CV + full CVC + full CCV',
}


# ---------------------------------------------------------------------------
# Inherent vowel sweep (Step 11.5.2)
# ---------------------------------------------------------------------------

def run_inherent_vowel_sweep(
    corpus_tokens: List[str],
    ref_corpus: Any,
    cv_labels: Dict,
    rosetta_data: Dict,
    eva_to_cell: Dict[str, str],
    base_assignment: Dict[str, str],
    base_dict_hit: float,
    beam_width: int = 20,
    max_solutions: int = 5,
) -> List[InherentVowelResult]:
    """Test each of {a, e, i} as the inherent vowel (Level 1 relaxation).

    Runs a quick beam search for each candidate and returns results sorted
    by dict_hit_rate descending.
    """
    print("\n  --- Inherent Vowel Sweep ---")
    results: List[InherentVowelResult] = []

    ref_tokens = ref_corpus.get_combined_tokens('latin')
    if not ref_tokens:
        ref_tokens = ref_corpus.get_combined_tokens(ref_corpus.languages[0])
    ref_word_set = set(ref_tokens[:50000])
    lm = build_ngram_lm(ref_tokens[:10000], order=3, smoothing=0.01)
    anchors = build_anchor_constraints(rosetta_data, cv_labels)

    for vowel in INHERENT_VOWEL_CANDIDATES:
        print(f"\n  Testing inherent vowel '{vowel}'...")
        t0 = time.time()

        inventory = build_phoneme_inventory(
            'latin', ref_corpus,
            relaxation_level=1, inherent_vowel=vowel,
        )
        print(f"    Inventory size: {len(inventory.cv_syllables)} syllables")

        variables = build_csp_variables(cv_labels)
        cell_frequencies = {v.cell_key: v.frequency for v in variables}
        variables = initialise_domains(
            variables, inventory, cell_frequencies, anchors, frequency_slack=4,
        )
        _, variables = ac3_propagate(variables)

        assignments = beam_search(
            variables, lm, corpus_tokens, eva_to_cell, anchors, inventory,
            ref_word_set=ref_word_set,
            beam_width=beam_width, max_solutions=max_solutions,
        )

        if assignments:
            best = assignments[0]
            dict_hit = best.dict_hit_rate
            ce = best.cross_entropy
            best_map = dict(best.mapping)
        else:
            dict_hit = 0.0
            ce = 99.0
            best_map = {}

        elapsed = time.time() - t0
        delta = dict_hit - base_dict_hit
        print(f"    dict_hit={dict_hit:.4f}  CE={ce:.4f}  Δ={delta:+.4f}  ({elapsed:.1f}s)")

        results.append(InherentVowelResult(
            vowel=vowel,
            dict_hit_rate=dict_hit,
            cross_entropy=ce,
            best_assignment=best_map,
        ))

    results.sort(key=lambda r: r.dict_hit_rate, reverse=True)
    print(f"\n  Best inherent vowel: '{results[0].vowel}' (dict_hit={results[0].dict_hit_rate:.4f})")
    return results


# ---------------------------------------------------------------------------
# Graduated relaxation sweep (Step 11.5.3)
# ---------------------------------------------------------------------------

def run_relaxation_sweep(
    corpus_tokens: List[str],
    ref_corpus: Any,
    cv_labels: Dict,
    rosetta_data: Dict,
    eva_to_cell: Dict[str, str],
    best_inherent_vowel: str,
    high_error_cells: Optional[List[str]] = None,
    baseline_dict_hit: float = 0.0,
    beam_width: int = 40,
    max_solutions: int = 10,
) -> List[RelaxationLevelResult]:
    """Sweep relaxation levels 0-5 and record dict_hit, CE, selectivity.

    At levels >= 2, expanded domains are applied only to *high_error_cells*
    (from csp_diagnosis); all other cells remain at Level 0 domain.
    """
    print("\n  --- Graduated Relaxation Sweep ---")

    ref_tokens = ref_corpus.get_combined_tokens('latin')
    if not ref_tokens:
        ref_tokens = ref_corpus.get_combined_tokens(ref_corpus.languages[0])
    ref_word_set = set(ref_tokens[:50000])
    lm = build_ngram_lm(ref_tokens[:10000], order=3, smoothing=0.01)
    anchors = build_anchor_constraints(rosetta_data, cv_labels)
    cell_keys = list(cv_labels.keys())

    # Random baseline CE for selectivity
    from voynich.phases.csp_decode import _random_baseline_ce as _rbc
    cv_base = build_cv_syllable_table('latin')
    mean_random_ce, _ = _rbc(
        cv_base, cell_keys, lm, corpus_tokens[:500], eva_to_cell,
        n_trials=100, max_tokens=300,
    )
    print(f"  Random baseline CE: {mean_random_ce:.4f}")

    results: List[RelaxationLevelResult] = []

    for level in range(6):
        desc = LEVEL_DESCRIPTIONS.get(level, f'Level {level}')
        print(f"\n  Level {level}: {desc}")
        t0 = time.time()

        # Build inventory for this level
        if level == 0:
            inventory = build_phoneme_inventory('latin', ref_corpus, relaxation_level=0)
        else:
            inventory = build_phoneme_inventory(
                'latin', ref_corpus,
                relaxation_level=level, inherent_vowel=best_inherent_vowel,
            )

        n_syls = len(inventory.cv_syllables)
        print(f"    Inventory size: {n_syls} syllables")

        # Build variables and initialise domains
        variables = build_csp_variables(cv_labels)
        cell_frequencies = {v.cell_key: v.frequency for v in variables}

        # For levels >= 2, use targeted expansion: only expand high-error cells.
        # Other cells get Level 0 domain.
        if level >= 2 and high_error_cells:
            base_inventory = build_phoneme_inventory('latin', ref_corpus, relaxation_level=0)
            # Initialise all cells with level-0 inventory
            variables_base = build_csp_variables(cv_labels)
            variables_base = initialise_domains(
                variables_base, base_inventory, cell_frequencies, anchors, frequency_slack=3,
            )
            base_domains = {v.cell_key: list(v.domain) for v in variables_base}

            # For high-error cells, use expanded inventory
            expanded_domains = {v.cell_key: list(inventory.cv_syllables) for v in variables}
            cell_domains: Dict[str, List[str]] = {}
            for v in variables:
                if v.cell_key in high_error_cells:
                    cell_domains[v.cell_key] = expanded_domains[v.cell_key]
                else:
                    cell_domains[v.cell_key] = base_domains.get(v.cell_key, base_domains.get(
                        list(base_domains.keys())[0], list(base_inventory.cv_syllables),
                    ))
            for v in variables:
                v.domain = cell_domains.get(v.cell_key, list(base_inventory.cv_syllables))
        else:
            variables = initialise_domains(
                variables, inventory, cell_frequencies, anchors, frequency_slack=3,
            )

        _, variables = ac3_propagate(variables)

        assignments = beam_search(
            variables, lm, corpus_tokens, eva_to_cell, anchors, inventory,
            ref_word_set=ref_word_set,
            relaxation_level=level,
            beam_width=beam_width, max_solutions=max_solutions,
        )

        elapsed = time.time() - t0

        if assignments:
            best = assignments[0]
            dict_hit = best.dict_hit_rate
            ce = best.cross_entropy
            anchor_n = best.anchor_match_count
            best_map = dict(best.mapping)
        else:
            dict_hit = 0.0
            ce = 99.0
            anchor_n = 0
            best_map = {}

        selectivity = mean_random_ce / max(ce, 0.01)
        delta = dict_hit - baseline_dict_hit
        print(f"    dict_hit={dict_hit:.4f}  CE={ce:.4f}  "
              f"selectivity={selectivity:.2f}x  anchors={anchor_n}  "
              f"Δdict_hit={delta:+.4f}  ({elapsed:.1f}s)")

        if selectivity < 1.5:
            print(f"    [WARN] Selectivity dropped below 1.5× gate!")

        results.append(RelaxationLevelResult(
            level=level,
            description=desc,
            n_syllables=n_syls,
            dict_hit_rate=dict_hit,
            cross_entropy=ce,
            selectivity=selectivity,
            anchor_match_count=anchor_n,
            best_assignment=best_map,
            runtime_seconds=elapsed,
        ))

    return results


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_csp_refinement() -> Dict:
    """Phase 11.5.2-3: inherent vowel + graduated relaxation sweep.

    1. Loads Phase 11.5.1 diagnosis results for high_error_cells.
    2. Loads Phase 11 results for baseline dict_hit and best_assignment.
    3. Runs inherent vowel sweep → best_inherent_vowel.
    4. Runs graduated relaxation sweep (0-5) → best_relaxation_level.
    5. Gate: best_dict_hit >= 0.15 OR improvement_factor >= 1.35.
    6. Saves to results/csp_refinement.json.
    """
    print("=" * 70)
    print("PHASE 11.5.2-3: Inherent Vowel + Graduated Relaxation Sweep")
    print("=" * 70)

    t0_total = time.time()
    rd = _results_dir()

    # ------------------------------------------------------------------
    # 1. Load diagnosis results
    # ------------------------------------------------------------------
    diagnosis_path = os.path.join(rd, 'csp_diagnosis.json')
    if not os.path.exists(diagnosis_path):
        print("  [WARN] csp_diagnosis.json not found — proceeding without high_error_cells")
        high_error_cells: List[str] = []
        diagnosis_data: Dict = {}
    else:
        with open(diagnosis_path) as f:
            diagnosis_data = json.load(f)
        high_error_cells = diagnosis_data.get('high_error_cells', [])
        print(f"  High-error cells from diagnosis: {len(high_error_cells)}")

    # ------------------------------------------------------------------
    # 2. Load Phase 11 results (baseline)
    # ------------------------------------------------------------------
    decode_path = os.path.join(rd, 'csp_decode.json')
    if not os.path.exists(decode_path):
        print("  [SKIP] csp_decode.json not found — run csp-decode first")
        return {'verdict': 'skipped', 'reason': 'no_csp_decode'}

    with open(decode_path) as f:
        decode_data = json.load(f)

    baseline_dict_hit = float(decode_data.get('best_dict_hit', 0.0))
    # Look in language_results for Latin dict hit
    lang_results = decode_data.get('language_results', {})
    if 'latin' in lang_results:
        baseline_dict_hit = float(lang_results['latin'].get('best_dict_hit', baseline_dict_hit))

    print(f"  Baseline dict_hit (Phase 11): {baseline_dict_hit:.4f}")

    # ------------------------------------------------------------------
    # 3. Load corpus and reference data
    # ------------------------------------------------------------------
    print("\nLoading data...")
    corpus = load_corpus(verbose=False)
    ref_corpus = load_reference_corpus(verbose=False)

    cv_path = os.path.join(rd, 'cv_labels.json')
    with open(cv_path) as f:
        cv_labels = json.load(f)

    rosetta_path = os.path.join(rd, 'rosetta_selection.json')
    with open(rosetta_path) as f:
        rosetta_data = json.load(f)

    eva_to_cell = build_eva_to_cell_lookup(cv_labels)
    corpus_tokens = corpus.get_tokens(language='A', paragraph_only=True)
    print(f"  Corpus tokens: {len(corpus_tokens)}")

    # Use a moderate sample for speed
    sample_tokens = corpus_tokens[:1500]

    # ------------------------------------------------------------------
    # 4. Inherent vowel sweep
    # ------------------------------------------------------------------
    iv_results = run_inherent_vowel_sweep(
        sample_tokens, ref_corpus, cv_labels, rosetta_data, eva_to_cell,
        base_assignment=decode_data.get('best_assignment', {}),
        base_dict_hit=baseline_dict_hit,
        beam_width=20, max_solutions=5,
    )
    best_inherent_vowel = iv_results[0].vowel if iv_results else 'a'
    iv_improvement = (iv_results[0].dict_hit_rate - baseline_dict_hit) if iv_results else 0.0

    # ------------------------------------------------------------------
    # 5. Graduated relaxation sweep
    # ------------------------------------------------------------------
    relax_results = run_relaxation_sweep(
        sample_tokens, ref_corpus, cv_labels, rosetta_data, eva_to_cell,
        best_inherent_vowel=best_inherent_vowel,
        high_error_cells=high_error_cells,
        baseline_dict_hit=baseline_dict_hit,
        beam_width=40, max_solutions=10,
    )

    # Find best level (by dict_hit, enforcing selectivity >= 1.5)
    valid_results = [r for r in relax_results if r.selectivity >= 1.5]
    if valid_results:
        best_result = max(valid_results, key=lambda r: r.dict_hit_rate)
    else:
        # Fall back to highest selectivity if nothing passes the gate
        best_result = max(relax_results, key=lambda r: r.selectivity)

    best_relaxation_level = best_result.level
    best_dict_hit = best_result.dict_hit_rate
    improvement_factor = best_dict_hit / max(baseline_dict_hit, 0.001)
    final_assignment = best_result.best_assignment

    print(f"\n  Best relaxation level: {best_relaxation_level} ({best_result.description})")
    print(f"  Best dict_hit: {best_dict_hit:.4f} (baseline: {baseline_dict_hit:.4f})")
    print(f"  Improvement factor: {improvement_factor:.2f}x")

    # ------------------------------------------------------------------
    # 6. Gate check
    # ------------------------------------------------------------------
    gate_passed = best_dict_hit >= 0.15 or improvement_factor >= 1.35

    if gate_passed:
        verdict = f'refinement_improved_dict_hit_{best_dict_hit:.3f}'
    elif best_dict_hit < 0.10:
        verdict = 'refinement_minimal_improvement_check_grid_decomposition'
    else:
        verdict = f'refinement_partial_improvement_{best_dict_hit:.3f}'

    print(f"\n  Gate: {'PASS ✓' if gate_passed else 'FAIL ✗'}")
    print(f"  Verdict: {verdict}")

    # ------------------------------------------------------------------
    # 7. Save results
    # ------------------------------------------------------------------
    result = CSPRefinementResult(
        inherent_vowel_results=[_convert(asdict(r)) for r in iv_results],
        best_inherent_vowel=best_inherent_vowel,
        inherent_vowel_improvement=round(iv_improvement, 4),
        relaxation_results=[_convert(asdict(r)) for r in relax_results],
        best_relaxation_level=best_relaxation_level,
        best_dict_hit_rate=round(best_dict_hit, 4),
        baseline_dict_hit_rate=round(baseline_dict_hit, 4),
        improvement_factor=round(improvement_factor, 3),
        high_error_cells=high_error_cells,
        final_assignment=final_assignment,
        gate_passed=gate_passed,
        verdict=verdict,
    )

    out_path = os.path.join(rd, 'csp_refinement.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2)

    elapsed = time.time() - t0_total
    print(f"\n  Saved to {out_path} ({elapsed:.1f}s total)")

    return _convert(asdict(result))
