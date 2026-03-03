"""
Phase 15.4 – Combined Optimization + Ablation Study
====================================================
Run all three Phase 15 improvements (dictionary expansion, articulatory
consistency, iterative hit constraints) in combination and perform a
2^3 ablation study to understand which improvements contribute most.

Dependency chain:
    feature_decode.json (Phase 14)
    dict_expansion.json (Step 15.1)
    articulatory_csp.json (Step 15.2)
    iterative_hits.json (Step 15.3)
        → combined_refine.json (this step)
"""

import copy
import json
import os
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    load_corpus,
    tokenize_eva_chars,
)
from voynich.core.reference import (
    build_cv_syllable_table,
    build_expanded_word_set,
    build_triple_phoneme_hypotheses,
    load_reference_corpus,
)
from voynich.core.stats import build_ngram_lm
from voynich.phases.csp_constraints import (
    AnchorConstraint,
    PhonemeInventory,
    build_phoneme_inventory,
)
from voynich.phases.csp_solver import (
    _convert,
    ac3_propagate,
    beam_search,
    decode_corpus,
)
from voynich.phases.feature_csp import (
    FeatureVariable,
    build_feature_variables,
    initialise_feature_domains,
    _build_anchor_constraints_triple,
)
from voynich.phases.articulatory_csp import (
    _ONSET_PLACE_MAPPING,
    _get_consistent_syllables,
    compute_articulatory_consistency,
)
from voynich.phases.iterative_hits import (
    extract_high_confidence_hits,
    apply_hit_constraints,
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CombinedRefineResult:
    # Ablation table
    ablation_table: List[Dict]

    # Best combined result
    best_config: str
    best_dict_hit: float
    best_selectivity: float
    best_ac: float
    best_assignment: Dict[str, str]

    # Convergence
    convergence_curve: List[Dict]
    n_iterations: int

    # Synergy analysis
    baseline_dict_hit: float
    dict_only_delta: float
    ac_only_delta: float
    hits_only_delta: float
    combined_delta: float
    synergy: float

    gate_passed: bool
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Random baseline
# ---------------------------------------------------------------------------

def _compute_random_baseline(
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
# Single ablation run
# ---------------------------------------------------------------------------

def _run_config(
    config_name: str,
    use_expanded_dict: bool,
    use_ac: bool,
    use_hits: bool,
    # Shared resources
    tokens: List[str],
    eva_to_triple: Dict[str, str],
    original_word_set: set,
    expanded_word_set: set,
    inventory: PhonemeInventory,
    lm: Dict,
    all_syls: List[str],
    glyph_freq: Counter,
    anchors: List[AnchorConstraint],
    baseline_assignment: Dict[str, str],
) -> Dict:
    """Run one ablation configuration and return results dict."""
    ref_word_set = expanded_word_set if use_expanded_dict else original_word_set

    # Build variables
    hypothesis_map = build_triple_phoneme_hypotheses('latin', all_syls)
    variables = build_feature_variables(eva_to_triple, glyph_freq, inventory, hypothesis_map)
    variables = initialise_feature_domains(variables, inventory, hypothesis_map, anchors)

    # Apply articulatory hard constraints if enabled
    if use_ac:
        for var in variables:
            parts = var.cell_key.split(',')
            if len(parts) == 3:
                onset_type = parts[0]
                place_class = _ONSET_PLACE_MAPPING.get(onset_type, 'stop')
                consistent = _get_consistent_syllables(place_class, var.domain)
                if consistent:
                    var.domain = consistent

    # Apply hit constraints if enabled — use split-variable approach
    fixed_mapping: Dict[str, str] = {}
    if use_hits:
        hit_constraints = extract_high_confidence_hits(
            baseline_assignment, tokens, eva_to_triple, ref_word_set, min_frequency=2,
        )
        if hit_constraints:
            for hc in hit_constraints:
                for tk, syl in zip(hc.triple_keys, hc.target_syllables):
                    if tk not in fixed_mapping:
                        fixed_mapping[tk] = syl
            # Only search over free variables
            variables = [v for v in variables if v.cell_key not in fixed_mapping]

    # AC-3 (safe on free variables only)
    if variables:
        solvable, variables = ac3_propagate(variables)
        if not solvable:
            return {
                'config': config_name,
                'dict_expansion': use_expanded_dict,
                'ac_scoring': use_ac,
                'hit_constraints': use_hits,
                'dict_hit': 0.0,
                'selectivity': 0.0,
                'ac': 0.0,
                'note': 'unsolvable',
            }

    # Beam search (on free variables only if hits enabled)
    if not variables:
        # All triples constrained — just score the fixed mapping
        decoded = decode_corpus(tokens, fixed_mapping, eva_to_triple, max_tokens=2000)
        hits = sum(1 for w in decoded if w in ref_word_set)
        dict_hit = hits / len(decoded) if decoded else 0.0
        ac_score, _ = compute_articulatory_consistency(fixed_mapping)
        variables_keys = list(fixed_mapping.keys())
        random_hit = _compute_random_baseline(
            variables_keys, all_syls, tokens, eva_to_triple, ref_word_set,
            n_trials=30, seed=42,
        )
        return {
            'config': config_name,
            'dict_expansion': use_expanded_dict,
            'ac_scoring': use_ac,
            'hit_constraints': use_hits,
            'dict_hit': round(dict_hit, 4),
            'selectivity': round(dict_hit / max(random_hit, 0.001), 2),
            'ac': round(ac_score, 4),
            'assignment': fixed_mapping,
        }

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
        seed=42,
    )

    if not solutions:
        return {
            'config': config_name,
            'dict_expansion': use_expanded_dict,
            'ac_scoring': use_ac,
            'hit_constraints': use_hits,
            'dict_hit': 0.0,
            'selectivity': 0.0,
            'ac': 0.0,
            'note': 'no solutions',
        }

    best = solutions[0]

    # Merge fixed + free assignments
    merged = dict(fixed_mapping)
    merged.update(best.mapping)

    # Re-score with complete merged assignment
    decoded = decode_corpus(tokens, merged, eva_to_triple, max_tokens=2000)
    hit_count = sum(1 for w in decoded if w in ref_word_set)
    dict_hit = hit_count / len(decoded) if decoded else 0.0

    ac_score, _ = compute_articulatory_consistency(merged)

    # Compute selectivity vs random
    variables_keys = list(merged.keys())
    random_hit = _compute_random_baseline(
        variables_keys, all_syls, tokens, eva_to_triple, ref_word_set,
        n_trials=30, seed=42,
    )
    selectivity = dict_hit / max(random_hit, 0.001)

    return {
        'config': config_name,
        'dict_expansion': use_expanded_dict,
        'ac_scoring': use_ac,
        'hit_constraints': use_hits,
        'dict_hit': round(dict_hit, 4),
        'selectivity': round(selectivity, 2),
        'ac': round(ac_score, 4),
        'assignment': merged,
    }


# ---------------------------------------------------------------------------
# Iterative combined pipeline
# ---------------------------------------------------------------------------

def _run_combined_iterative(
    tokens: List[str],
    eva_to_triple: Dict[str, str],
    ref_word_set: set,
    inventory: PhonemeInventory,
    lm: Dict,
    all_syls: List[str],
    glyph_freq: Counter,
    anchors: List[AnchorConstraint],
    initial_assignment: Dict[str, str],
    max_iterations: int = 5,
) -> Tuple[Dict[str, str], float, List[Dict]]:
    """Full combined pipeline with iterative hit propagation."""
    current_assignment = dict(initial_assignment)
    convergence_curve: List[Dict] = []

    for iteration in range(max_iterations):
        # Extract hits from current assignment
        hit_constraints = extract_high_confidence_hits(
            current_assignment, tokens, eva_to_triple, ref_word_set, min_frequency=2,
        )

        # Build fixed mapping from hits
        fixed_mapping: Dict[str, str] = {}
        if hit_constraints:
            for hc in hit_constraints:
                for tk, syl in zip(hc.triple_keys, hc.target_syllables):
                    if tk not in fixed_mapping:
                        fixed_mapping[tk] = syl

        # Build variables — only free (unconstrained) triples
        hypothesis_map = build_triple_phoneme_hypotheses('latin', all_syls)
        all_variables = build_feature_variables(eva_to_triple, glyph_freq, inventory, hypothesis_map)
        all_variables = initialise_feature_domains(all_variables, inventory, hypothesis_map, anchors)

        # AC constraints on all variables first
        for var in all_variables:
            parts = var.cell_key.split(',')
            if len(parts) == 3:
                onset_type = parts[0]
                place_class = _ONSET_PLACE_MAPPING.get(onset_type, 'stop')
                consistent = _get_consistent_syllables(place_class, var.domain)
                if consistent:
                    var.domain = consistent

        # Split into free variables only
        free_variables = [v for v in all_variables if v.cell_key not in fixed_mapping]

        if not free_variables:
            # All constrained — just score
            decoded = decode_corpus(tokens, fixed_mapping, eva_to_triple, max_tokens=2000)
            hits = sum(1 for w in decoded if w in ref_word_set)
            dict_hit = hits / len(decoded) if decoded else 0.0
            convergence_curve.append({
                'iteration': iteration,
                'dict_hit': round(dict_hit, 4),
                'n_hit_constraints': len(hit_constraints),
            })
            current_assignment = dict(fixed_mapping)
            break

        solvable, free_variables = ac3_propagate(free_variables)
        if not solvable:
            break

        solutions = beam_search(
            variables=free_variables,
            lm=lm,
            voynich_tokens=tokens,
            eva_to_cell=eva_to_triple,
            anchors=anchors,
            inventory=inventory,
            ref_word_set=ref_word_set,
            beam_width=80,
            max_solutions=10,
            seed=42 + iteration,
        )

        if not solutions:
            break

        # Merge fixed + free
        best = solutions[0]
        merged = dict(fixed_mapping)
        merged.update(best.mapping)

        decoded = decode_corpus(tokens, merged, eva_to_triple, max_tokens=2000)
        hits = sum(1 for w in decoded if w in ref_word_set)
        dict_hit = hits / len(decoded) if decoded else 0.0

        convergence_curve.append({
            'iteration': iteration,
            'dict_hit': round(dict_hit, 4),
            'n_hit_constraints': len(hit_constraints),
        })

        # Check convergence
        if iteration > 0:
            prev = convergence_curve[-2]['dict_hit']
            if dict_hit - prev < 0.005:
                current_assignment = merged
                break

        current_assignment = merged

    # Final dict_hit
    decoded = decode_corpus(tokens, current_assignment, eva_to_triple, max_tokens=2000)
    final_hits = sum(1 for w in decoded if w in ref_word_set)
    final_dict_hit = final_hits / len(decoded) if decoded else 0.0

    return current_assignment, final_dict_hit, convergence_curve


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_combined_refine() -> None:
    """Step 15.4: Combined optimization + ablation study."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE 15.4: Combined Optimization + Ablation Study")
    print("=" * 70)

    rd = _results_dir()

    # Load Phase 14 baseline
    fd_path = os.path.join(rd, 'feature_decode.json')
    if not os.path.exists(fd_path):
        print("  [SKIP] feature_decode.json not found — run feature-decode first")
        return

    with open(fd_path) as f:
        fd_data = json.load(f)

    baseline_assignment = fd_data.get('best_assignment', {})
    baseline_dict_hit = fd_data.get('best_dict_hit', 0.0)

    if not baseline_assignment:
        print("  [SKIP] No best assignment")
        return

    # Load corpus
    corpus = load_corpus(verbose=False)
    tokens = corpus.get_tokens(language='A', paragraph_only=True)
    if not tokens:
        print("  [SKIP] No Language A tokens")
        return

    eva_to_triple = build_eva_to_triple_lookup()

    # Reference data
    ref_corpus = load_reference_corpus(verbose=False)
    ref_tokens = ref_corpus.get_combined_tokens('latin')
    original_word_set = set(w.lower() for w in ref_tokens if len(w) >= 2)
    expanded_word_set, _ = build_expanded_word_set(original_word_set)
    inventory = build_phoneme_inventory('latin', ref_corpus)
    lm = build_ngram_lm(ref_tokens[:10000], order=3, smoothing=0.01)
    all_syls = build_cv_syllable_table('latin')

    glyph_freq: Counter = Counter()
    for token in tokens:
        for ch in tokenize_eva_chars(token):
            glyph_freq[ch] += 1

    # Anchors
    rosetta_path = os.path.join(rd, 'rosetta_selection.json')
    anchors: List[AnchorConstraint] = []
    if os.path.exists(rosetta_path):
        with open(rosetta_path) as f:
            rosetta_data = json.load(f)
        anchors = _build_anchor_constraints_triple(rosetta_data, eva_to_triple)

    # ─── Ablation study: 2^3 configurations ───
    print("\n  Running 2^3 ablation study ...")
    configs = [
        ('baseline',        False, False, False),
        ('dict',            True,  False, False),
        ('ac',              False, True,  False),
        ('hits',            False, False, True),
        ('dict+ac',         True,  True,  False),
        ('dict+hits',       True,  False, True),
        ('ac+hits',         False, True,  True),
        ('dict+ac+hits',    True,  True,  True),
    ]

    ablation_table: List[Dict] = []
    config_assignments: Dict[str, Dict[str, str]] = {}

    for name, use_dict, use_ac, use_hits in configs:
        print(f"\n    Config: {name} ... ", end='', flush=True)
        result = _run_config(
            config_name=name,
            use_expanded_dict=use_dict,
            use_ac=use_ac,
            use_hits=use_hits,
            tokens=tokens,
            eva_to_triple=eva_to_triple,
            original_word_set=original_word_set,
            expanded_word_set=expanded_word_set,
            inventory=inventory,
            lm=lm,
            all_syls=all_syls,
            glyph_freq=glyph_freq,
            anchors=anchors,
            baseline_assignment=baseline_assignment,
        )
        print(f"dict_hit={result['dict_hit']:.3f}, selectivity={result['selectivity']:.2f}x")

        # Store assignment separately (don't bloat ablation table)
        if 'assignment' in result:
            config_assignments[name] = result.pop('assignment')
        ablation_table.append(result)

    # Print ablation summary
    print("\n  ┌─────────────────┬──────────┬─────────────┬────────┐")
    print("  │ Config          │ dict_hit │ selectivity │   AC   │")
    print("  ├─────────────────┼──────────┼─────────────┼────────┤")
    for row in ablation_table:
        print(f"  │ {row['config']:<15s} │ {row['dict_hit']:>7.3f}  │ {row['selectivity']:>10.2f}x │ {row['ac']:>6.3f} │")
    print("  └─────────────────┴──────────┴─────────────┴────────┘")

    # Select best config (highest dict_hit with selectivity >= 1.5)
    valid = [r for r in ablation_table if r['selectivity'] >= 1.5]
    if not valid:
        valid = ablation_table  # fallback
    best_config_row = max(valid, key=lambda r: r['dict_hit'])
    best_config = best_config_row['config']
    best_ablation_assignment = config_assignments.get(best_config, baseline_assignment)

    # ─── Full combined iterative pipeline with best config ───
    print(f"\n  Running full combined iterative pipeline ({best_config}) ...")
    use_expanded = best_config_row.get('dict_expansion', False)
    ref_word_set_final = expanded_word_set if use_expanded else original_word_set

    combined_assignment, combined_dict_hit, convergence_curve = _run_combined_iterative(
        tokens=tokens,
        eva_to_triple=eva_to_triple,
        ref_word_set=ref_word_set_final,
        inventory=inventory,
        lm=lm,
        all_syls=all_syls,
        glyph_freq=glyph_freq,
        anchors=anchors,
        initial_assignment=best_ablation_assignment,
        max_iterations=5,
    )

    combined_ac, _ = compute_articulatory_consistency(combined_assignment)
    variables_keys = list(combined_assignment.keys())
    random_hit = _compute_random_baseline(
        variables_keys, all_syls, tokens, eva_to_triple, ref_word_set_final,
    )
    combined_selectivity = combined_dict_hit / max(random_hit, 0.001)

    # Synergy analysis
    baseline_row = next((r for r in ablation_table if r['config'] == 'baseline'), None)
    baseline_hit = baseline_row['dict_hit'] if baseline_row else baseline_dict_hit

    dict_only_row = next((r for r in ablation_table if r['config'] == 'dict'), None)
    ac_only_row = next((r for r in ablation_table if r['config'] == 'ac'), None)
    hits_only_row = next((r for r in ablation_table if r['config'] == 'hits'), None)

    dict_only_delta = (dict_only_row['dict_hit'] - baseline_hit) if dict_only_row else 0.0
    ac_only_delta = (ac_only_row['dict_hit'] - baseline_hit) if ac_only_row else 0.0
    hits_only_delta = (hits_only_row['dict_hit'] - baseline_hit) if hits_only_row else 0.0
    combined_delta = combined_dict_hit - baseline_hit
    synergy = combined_delta - (dict_only_delta + ac_only_delta + hits_only_delta)

    gate_passed = combined_dict_hit > baseline_dict_hit and combined_selectivity >= 1.5

    elapsed = time.time() - t0

    verdict = (
        f"Best config: {best_config}. "
        f"Combined dict_hit={combined_dict_hit:.1%} ({combined_selectivity:.2f}x), "
        f"AC={combined_ac:.3f}. "
        f"Baseline: {baseline_dict_hit:.1%}. "
        f"Synergy: {synergy:+.3f}."
    )

    result = CombinedRefineResult(
        ablation_table=ablation_table,
        best_config=best_config,
        best_dict_hit=round(combined_dict_hit, 4),
        best_selectivity=round(combined_selectivity, 2),
        best_ac=round(combined_ac, 4),
        best_assignment=combined_assignment,
        convergence_curve=convergence_curve,
        n_iterations=len(convergence_curve),
        baseline_dict_hit=round(baseline_dict_hit, 4),
        dict_only_delta=round(dict_only_delta, 4),
        ac_only_delta=round(ac_only_delta, 4),
        hits_only_delta=round(hits_only_delta, 4),
        combined_delta=round(combined_delta, 4),
        synergy=round(synergy, 4),
        gate_passed=gate_passed,
        verdict=verdict,
        runtime_seconds=round(elapsed, 2),
    )

    out_path = os.path.join(rd, 'combined_refine.json')
    with open(out_path, 'w') as f:
        json.dump(asdict(result), f, indent=2, default=_convert)

    print(f"\n  Gate: {'PASS' if gate_passed else 'FAIL'}")
    print(f"  {verdict}")
    print(f"\n  → {out_path}")
