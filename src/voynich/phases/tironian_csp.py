"""
Phase C.1-C.2 -- Tironian-Prior CSP Re-Solve
=============================================
Inject Tironian note domain priors into the CSP feature variables
and re-solve.  For each attested Voynich stroke triple, Tironian
candidates (from the master paleographic reference) are placed at
the front of the domain so the beam search explores them first.

The output is a full phonetic assignment that can be compared
directly to the Phase 16 baseline (modifier-integrated decode).

Dependency chain:
    data/reference/paleographic/master_reference.json
    results/stroke_features.json    (Phase 14.2)
    results/modifier_integrate.json (Phase 16.6)
    results/combined_refine.json    (Phase 15.4 fallback)
        -> tironian_csp.json (this step)
"""

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
    token_to_triples,
    tokenize_eva_chars,
)
from voynich.core.reference import (
    EVA_VISUAL_COMPONENTS,
    build_cv_syllable_table,
    build_expanded_word_set,
    build_tironian_domain_priors,
    build_triple_phoneme_hypotheses,
    load_master_reference,
    load_reference_corpus,
)
from voynich.core.stats import build_ngram_lm
from voynich.phases.csp_constraints import (
    AnchorConstraint,
    build_phoneme_inventory,
    prune_by_frequency,
    prune_by_inventory,
    prune_by_phonotactics,
)
from voynich.phases.csp_solver import (
    beam_search,
    decode_corpus,
    decode_token,
)
from voynich.phases.feature_csp import (
    FeatureVariable,
    build_feature_variables,
    initialise_feature_domains,
    _build_anchor_constraints_triple,
)


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


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TironianCSPResult:
    """Phase C.1-C.2: CSP re-solve with Tironian priors."""
    n_triples_with_priors: int
    n_triples_without_priors: int
    injected_candidates: Dict[str, List[str]]
    best_assignment: Dict[str, str]
    best_dict_hit: float
    best_selectivity: float
    phase16_dict_hit: float
    improvement: float
    decoded_sample: List[List[str]]
    gate_passed: bool
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_tironian_csp() -> None:
    """Phase C.1-C.2: Inject Tironian priors into CSP domains and re-solve."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE C.1-C.2: Tironian-Prior CSP Re-Solve")
    print("=" * 70)

    rd = _results_dir()

    # ------------------------------------------------------------------ 1
    print("\n  1. Loading master reference and building Tironian priors ...")
    master_ref = load_master_reference()
    if master_ref is None:
        print("    [WARN] master_reference.json not found -- running without Tironian priors")
        master_ref = {'all_signs': []}

    # ------------------------------------------------------------------ 2
    print("\n  2. Loading stroke_features.json for 25 attested triples ...")
    sf_path = os.path.join(rd, 'stroke_features.json')
    if not os.path.exists(sf_path):
        print("    [SKIP] stroke_features.json not found -- run stroke-features first")
        return
    with open(sf_path) as f:
        stroke_data = json.load(f)
    attested_triples = [t['triple_key'] for t in stroke_data.get('triples', [])]
    if not attested_triples:
        # Fallback: build from EVA_VISUAL_COMPONENTS
        attested_triples = sorted(set(
            f"{v['first_stroke']},{v['last_stroke']},{v['glyph_class']}"
            for v in EVA_VISUAL_COMPONENTS.values()
        ))
    print(f"    {len(attested_triples)} attested triples")

    # Build Tironian priors
    tironian_priors = build_tironian_domain_priors(master_ref, attested_triples)
    n_with_priors = sum(1 for p in tironian_priors.values() if p['tironian_candidates'])
    n_without_priors = len(attested_triples) - n_with_priors
    injected_candidates: Dict[str, List[str]] = {
        tk: info['tironian_candidates']
        for tk, info in tironian_priors.items()
        if info['tironian_candidates']
    }
    print(f"    Triples with Tironian priors: {n_with_priors}")
    print(f"    Triples without priors:       {n_without_priors}")

    # ------------------------------------------------------------------ 3
    print("\n  3. Loading modifier_integrate.json for Phase 16 baseline ...")
    mod_path = os.path.join(rd, 'modifier_integrate.json')
    phase16_dict_hit = 0.0
    modifier_chars: set = set()
    if os.path.exists(mod_path):
        with open(mod_path) as f:
            mod_data = json.load(f)
        phase16_dict_hit = mod_data.get('best_dict_hit', 0.0)
        modifier_chars = set(mod_data.get('modifier_chars', []))
        print(f"    Phase 16 dict_hit: {phase16_dict_hit:.4f}")
    else:
        print("    [WARN] modifier_integrate.json not found")

    # ------------------------------------------------------------------ 4
    print("\n  4. Loading combined_refine.json for Phase 15 fallback assignment ...")
    refine_path = os.path.join(rd, 'combined_refine.json')
    fallback_assignment: Dict[str, str] = {}
    if os.path.exists(refine_path):
        with open(refine_path) as f:
            refine_data = json.load(f)
        fallback_assignment = refine_data.get('best_assignment', {})
        print(f"    Fallback assignment loaded ({len(fallback_assignment)} mappings)")
    else:
        print("    [WARN] combined_refine.json not found")

    # ------------------------------------------------------------------ 5
    print("\n  5. Building feature variables with Tironian-enriched domains ...")
    corpus = load_corpus(verbose=False)
    tokens = corpus.get_tokens(language='A', paragraph_only=True)
    if not tokens:
        print("    [SKIP] No Language A tokens found")
        return
    print(f"    {len(tokens)} Language A tokens")

    eva_to_triple = build_eva_to_triple_lookup()

    # Glyph frequencies
    glyph_freq: Counter = Counter()
    for token in tokens:
        for ch in tokenize_eva_chars(token):
            glyph_freq[ch] += 1

    # Reference data for Latin
    ref_corpus = load_reference_corpus(verbose=False)
    ref_tokens = ref_corpus.get_combined_tokens('latin')
    inventory = build_phoneme_inventory('latin', ref_corpus)
    lm = build_ngram_lm(ref_tokens[:10000], order=3, smoothing=0.01)
    # Original (non-expanded) word set for scoring
    ref_word_set = set(w.lower() for w in ref_tokens if len(w) >= 2)

    hypothesis_map = build_triple_phoneme_hypotheses(
        'latin', build_cv_syllable_table('latin')
    )

    # Build anchor constraints
    rosetta_path = os.path.join(rd, 'rosetta_selection.json')
    anchors: List[AnchorConstraint] = []
    if os.path.exists(rosetta_path):
        with open(rosetta_path) as f:
            rosetta_data = json.load(f)
        anchors = _build_anchor_constraints_triple(rosetta_data, eva_to_triple)

    # Build base feature variables
    variables = build_feature_variables(
        eva_to_triple, glyph_freq, inventory, hypothesis_map
    )

    # Initialise domains via standard pipeline
    variables = initialise_feature_domains(
        variables, inventory, hypothesis_map, anchors
    )

    # Inject Tironian candidates at the front of each domain
    legal_cv = set(inventory.cv_syllables)
    for var in variables:
        tir_cands = injected_candidates.get(var.cell_key, [])
        if tir_cands:
            # Only add candidates that are legal CV syllables
            valid_tir = [c for c in tir_cands if c in legal_cv]
            if valid_tir:
                # Tironian candidates first, then existing domain (deduplicated)
                existing = [s for s in var.domain if s not in set(valid_tir)]
                var.domain = valid_tir + existing

    # ------------------------------------------------------------------ 6
    print("\n  6. Domain injection summary:")
    n_injected = 0
    for var in variables:
        tir = injected_candidates.get(var.cell_key, [])
        valid_tir = [c for c in tir if c in legal_cv]
        if valid_tir:
            n_injected += 1
            print(f"    {var.cell_key}: +{len(valid_tir)} Tironian candidates -> domain size {len(var.domain)}")
    print(f"    Total: {n_injected} triples enriched, {len(variables) - n_injected} unchanged")

    # ------------------------------------------------------------------ 7
    print("\n  7. Running beam search (width=200, depth=25) ...")
    solutions = beam_search(
        variables=variables,  # type: ignore[arg-type]
        lm=lm,
        voynich_tokens=tokens,
        eva_to_cell=eva_to_triple,
        anchors=anchors,
        inventory=inventory,
        ref_word_set=ref_word_set,
        beam_width=200,
        max_solutions=25,
    )

    if not solutions:
        print("    [WARN] No solutions found -- using fallback assignment")
        best_assignment = fallback_assignment
        best_dict_hit = 0.0
    else:
        best_assignment = solutions[0].mapping
        best_dict_hit = solutions[0].dict_hit_rate
        print(f"    Found {len(solutions)} solutions")
        print(f"    Best dict_hit: {best_dict_hit:.4f}")

    # ------------------------------------------------------------------ 8
    print("\n  8. Decoding full corpus with best assignment ...")
    decoded_all = decode_corpus(tokens, best_assignment, eva_to_triple, max_tokens=5000)

    # ------------------------------------------------------------------ 9
    print("\n  9. Scoring against original dictionary (~17K words) ...")
    hits = sum(1 for w in decoded_all if w in ref_word_set)
    dict_hit = hits / len(decoded_all) if decoded_all else 0.0
    # Use dict_hit from beam search if available, else recalculate
    if solutions:
        best_dict_hit = solutions[0].dict_hit_rate
    else:
        best_dict_hit = dict_hit
    print(f"    dict_hit = {best_dict_hit:.4f} ({hits} hits / {len(decoded_all)} tokens)")

    # Selectivity vs random baseline
    rng = random.Random(42)
    all_syls = list(inventory.cv_syllables)
    random_hits: List[float] = []
    for _ in range(50):
        rand_map = {v.cell_key: rng.choice(all_syls) for v in variables}
        rand_decoded = decode_corpus(tokens, rand_map, eva_to_triple, max_tokens=500)
        rh = sum(1 for w in rand_decoded if w in ref_word_set)
        random_hits.append(rh / len(rand_decoded) if rand_decoded else 0.0)
    random_baseline = sum(random_hits) / len(random_hits) if random_hits else 0.001
    selectivity = best_dict_hit / max(random_baseline, 0.001)
    print(f"    Random baseline: {random_baseline:.4f}")
    print(f"    Selectivity: {selectivity:.2f}x")

    # ------------------------------------------------------------------ 10
    print("\n  10. Comparison with Phase 16 baseline ...")
    improvement = best_dict_hit - phase16_dict_hit
    print(f"    Phase 16 dict_hit:   {phase16_dict_hit:.4f}")
    print(f"    Tironian CSP:        {best_dict_hit:.4f}")
    print(f"    Improvement:         {improvement:+.4f}")

    # ------------------------------------------------------------------ 11
    print("\n  11. Comparison table:")
    print(f"    {'Metric':<25} {'Phase 16':>10} {'Tironian CSP':>14}")
    print(f"    {'-' * 25} {'-' * 10} {'-' * 14}")
    print(f"    {'dict_hit':<25} {phase16_dict_hit:>10.4f} {best_dict_hit:>14.4f}")
    print(f"    {'selectivity':<25} {'---':>10} {selectivity:>14.2f}x")
    print(f"    {'improvement':<25} {'':>10} {improvement:>+14.4f}")

    # Decoded sample (first 20 tokens)
    decoded_sample: List[List[str]] = []
    for i in range(min(20, len(tokens), len(decoded_all))):
        decoded_sample.append([tokens[i], decoded_all[i]])

    print(f"\n    First 20 decoded tokens:")
    for tok, dec in decoded_sample:
        marker = '*' if dec in ref_word_set else ' '
        print(f"    {marker} {tok:>15} -> {dec}")

    # Gate: improvement > 0
    gate_passed = improvement > 0

    if gate_passed:
        verdict = (
            f"PASS: Tironian priors improve dict_hit by {improvement:+.4f} "
            f"({best_dict_hit:.1%} vs Phase 16 {phase16_dict_hit:.1%}). "
            f"Selectivity {selectivity:.2f}x. "
            f"{n_with_priors}/{len(attested_triples)} triples received priors."
        )
    else:
        verdict = (
            f"FAIL: Tironian priors did not improve dict_hit "
            f"({best_dict_hit:.1%} vs Phase 16 {phase16_dict_hit:.1%}). "
            f"Delta: {improvement:+.4f}. "
            f"Proceeding with Phase 16 assignment."
        )

    print(f"\n  Gate: {'PASS' if gate_passed else 'FAIL'}")
    print(f"  Verdict: {verdict}")

    # Save
    result = TironianCSPResult(
        n_triples_with_priors=n_with_priors,
        n_triples_without_priors=n_without_priors,
        injected_candidates=injected_candidates,
        best_assignment=best_assignment,
        best_dict_hit=round(best_dict_hit, 4),
        best_selectivity=round(selectivity, 2),
        phase16_dict_hit=round(phase16_dict_hit, 4),
        improvement=round(improvement, 4),
        decoded_sample=decoded_sample,
        gate_passed=gate_passed,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'tironian_csp.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)

    print(f"\n  Results saved -> {out_path}")
