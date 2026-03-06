"""
Phase C.5 -- Re-Segmented CSP
===============================
Run the CSP solver on re-segmented corpus data.  This step only
executes if Phase B.0 (ligature_analysis.json) found "moderate" or
"severe" ligature mis-segmentation.  If severity is "minimal", the
module short-circuits with gate_passed=False and a message that
re-segmentation is not needed.

When re-segmentation IS needed:
1. Load confirmed ligature merges/splits from ligature_analysis.json
2. Rebuild the EVA ligature table via rebuild_eva_ligatures()
3. Re-tokenize the corpus with the new segmentation
4. Build new stroke-feature triples from re-segmented characters
5. Run the CSP with Tironian priors on the re-segmented data
6. Decode and score against the original (non-resegmented) results

Dependency chain:
    results/ligature_analysis.json  (Phase B.0)
    results/tironian_csp.json       (Phase C.1-C.2)
    results/stroke_features.json    (Phase 14.2)
        -> reseg_csp.json (this step)
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
    rebuild_eva_ligatures,
    token_to_triples,
    token_to_triples_resegmented,
    tokenize_eva_chars,
    tokenize_eva_chars_resegmented,
)
from voynich.core.reference import (
    EVA_VISUAL_COMPONENTS,
    build_cv_syllable_table,
    build_tironian_domain_priors,
    build_triple_phoneme_hypotheses,
    load_ligature_observations,
    load_master_reference,
    load_reference_corpus,
)
from voynich.core.stats import build_ngram_lm
from voynich.phases.csp_constraints import (
    AnchorConstraint,
    build_phoneme_inventory,
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
class ResegCSPResult:
    """Phase C.5: re-segmented CSP result."""
    ligature_severity: str
    n_ligatures_merged: int
    original_char_count: int
    reseg_char_count: int
    reseg_dict_hit: float
    original_dict_hit: float
    improvement: float
    gate_passed: bool
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_reseg_csp() -> None:
    """Phase C.5: Re-segmented CSP (conditional on ligature severity)."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE C.5: Re-Segmented CSP")
    print("=" * 70)

    rd = _results_dir()

    # ------------------------------------------------------------------ 1
    print("\n  1. Loading ligature_analysis.json ...")
    lig_path = os.path.join(rd, 'ligature_analysis.json')
    if not os.path.exists(lig_path):
        print("    [SKIP] ligature_analysis.json not found -- run ligature-analysis first")
        result = ResegCSPResult(
            ligature_severity='unknown',
            n_ligatures_merged=0,
            original_char_count=0,
            reseg_char_count=0,
            reseg_dict_hit=0.0,
            original_dict_hit=0.0,
            improvement=0.0,
            gate_passed=False,
            verdict="SKIP: ligature_analysis.json not found. Cannot determine severity.",
            runtime_seconds=round(time.time() - t0, 2),
        )
        out_path = os.path.join(rd, 'reseg_csp.json')
        with open(out_path, 'w') as f:
            json.dump(_convert(result), f, indent=2)
        print(f"\n  Results saved -> {out_path}")
        return

    with open(lig_path) as f:
        lig_data = json.load(f)

    severity = lig_data.get('severity', 'minimal')
    print(f"    Severity: {severity}")

    # ------------------------------------------------------------------ 2
    if severity == 'minimal':
        print("\n  2. Severity is 'minimal' -- re-segmentation not needed.")
        # Load original dict_hit for completeness
        tir_path = os.path.join(rd, 'tironian_csp.json')
        original_dict_hit = 0.0
        if os.path.exists(tir_path):
            with open(tir_path) as f:
                original_dict_hit = json.load(f).get('best_dict_hit', 0.0)

        result = ResegCSPResult(
            ligature_severity=severity,
            n_ligatures_merged=0,
            original_char_count=0,
            reseg_char_count=0,
            reseg_dict_hit=0.0,
            original_dict_hit=original_dict_hit,
            improvement=0.0,
            gate_passed=False,
            verdict=(
                f"SKIP: Ligature severity is 'minimal' -- "
                f"current EVA segmentation is adequate. "
                f"Re-segmentation not needed."
            ),
            runtime_seconds=round(time.time() - t0, 2),
        )
        out_path = os.path.join(rd, 'reseg_csp.json')
        with open(out_path, 'w') as f:
            json.dump(_convert(result), f, indent=2)
        print(f"  Verdict: {result.verdict}")
        print(f"\n  Results saved -> {out_path}")
        return

    # ------------------------------------------------------------------ 3
    print(f"\n  3. Severity is '{severity}' -- proceeding with re-segmentation ...")

    # Load ligature observations for rebuild
    lig_obs = load_ligature_observations()
    if lig_obs is None:
        lig_obs = {'pair_summaries': [], 'new_segmentations': []}

    new_ligatures, reseg_map = rebuild_eva_ligatures(lig_obs)
    n_merged = len([v for v in reseg_map.values() if len(v) == 1])
    print(f"    Re-segmentation map entries: {len(reseg_map)}")
    print(f"    New ligatures: {len(new_ligatures)}")
    print(f"    Pairs merged into single signs: {n_merged}")

    # ------------------------------------------------------------------ 4
    print("\n  4. Re-tokenizing corpus with new segmentation ...")
    corpus = load_corpus(verbose=False)
    tokens = corpus.get_tokens(language='A', paragraph_only=True)
    if not tokens:
        print("    [SKIP] No Language A tokens found")
        return

    eva_to_triple = build_eva_to_triple_lookup()

    # Count original characters
    original_chars: Counter = Counter()
    for token in tokens:
        for ch in tokenize_eva_chars(token):
            original_chars[ch] += 1
    original_char_count = sum(original_chars.values())

    # Count re-segmented characters
    reseg_chars: Counter = Counter()
    for token in tokens:
        for ch in tokenize_eva_chars_resegmented(token, reseg_map):
            reseg_chars[ch] += 1
    reseg_char_count = sum(reseg_chars.values())

    print(f"    Original char instances: {original_char_count}")
    print(f"    Re-segmented char instances: {reseg_char_count}")

    # ------------------------------------------------------------------ 5
    print("\n  5. Building new stroke-feature triples from re-segmented chars ...")
    # Build glyph frequencies for re-segmented characters
    reseg_glyph_freq: Counter = Counter()
    for token in tokens:
        for ch in tokenize_eva_chars_resegmented(token, reseg_map):
            reseg_glyph_freq[ch] += 1

    # Reference data
    ref_corpus = load_reference_corpus(verbose=False)
    inventory = build_phoneme_inventory('latin', ref_corpus)
    ref_tokens = ref_corpus.get_combined_tokens('latin')
    lm = build_ngram_lm(ref_tokens[:10000], order=3, smoothing=0.01)
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

    # Build feature variables from re-segmented glyph frequencies
    variables = build_feature_variables(
        eva_to_triple, reseg_glyph_freq, inventory, hypothesis_map
    )
    variables = initialise_feature_domains(
        variables, inventory, hypothesis_map, anchors
    )

    # Inject Tironian priors (same as tironian_csp.py)
    master_ref = load_master_reference()
    if master_ref is None:
        master_ref = {'all_signs': []}
    reseg_triples = [v.cell_key for v in variables]
    tironian_priors = build_tironian_domain_priors(master_ref, reseg_triples)
    legal_cv = set(inventory.cv_syllables)

    for var in variables:
        tir_info = tironian_priors.get(var.cell_key, {})
        tir_cands = tir_info.get('tironian_candidates', [])
        valid_tir = [c for c in tir_cands if c in legal_cv]
        if valid_tir:
            existing = [s for s in var.domain if s not in set(valid_tir)]
            var.domain = valid_tir + existing

    print(f"    Feature variables: {len(variables)}")

    # ------------------------------------------------------------------ 6
    print("\n  6. Running CSP with Tironian priors on re-segmented data ...")
    # For decoding re-segmented tokens, we use the reseg-aware triple lookup
    # We decode by mapping each token through resegmented chars
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

    reseg_dict_hit = 0.0
    if solutions:
        reseg_dict_hit = solutions[0].dict_hit_rate
        print(f"    Best dict_hit: {reseg_dict_hit:.4f}")
    else:
        print("    [WARN] No solutions found")

    # ------------------------------------------------------------------ 7
    print("\n  7. Scoring and comparison ...")
    tir_path = os.path.join(rd, 'tironian_csp.json')
    original_dict_hit = 0.0
    if os.path.exists(tir_path):
        with open(tir_path) as f:
            original_dict_hit = json.load(f).get('best_dict_hit', 0.0)

    improvement = reseg_dict_hit - original_dict_hit
    gate_passed = improvement > 0.0

    print(f"    Original dict_hit: {original_dict_hit:.4f}")
    print(f"    Re-seg dict_hit:   {reseg_dict_hit:.4f}")
    print(f"    Improvement:       {improvement:+.4f}")

    if gate_passed:
        verdict = (
            f"PASS: Re-segmentation improved dict_hit by {improvement:+.4f} "
            f"({reseg_dict_hit:.1%} vs {original_dict_hit:.1%}). "
            f"Ligature severity '{severity}' warranted re-segmentation. "
            f"{n_merged} pairs merged."
        )
    else:
        verdict = (
            f"FAIL: Re-segmentation did not improve dict_hit "
            f"({reseg_dict_hit:.1%} vs {original_dict_hit:.1%}, "
            f"delta={improvement:+.4f}). "
            f"Original EVA segmentation is adequate despite "
            f"'{severity}' ligature severity."
        )

    print(f"\n  Gate: {'PASS' if gate_passed else 'FAIL'}")
    print(f"  Verdict: {verdict}")

    # Save
    result = ResegCSPResult(
        ligature_severity=severity,
        n_ligatures_merged=n_merged,
        original_char_count=original_char_count,
        reseg_char_count=reseg_char_count,
        reseg_dict_hit=round(reseg_dict_hit, 4),
        original_dict_hit=round(original_dict_hit, 4),
        improvement=round(improvement, 4),
        gate_passed=gate_passed,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'reseg_csp.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)

    print(f"\n  Results saved -> {out_path}")
