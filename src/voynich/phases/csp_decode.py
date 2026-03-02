"""
Phase 11 – Multi-language CSP phonetic decoding pipeline
=========================================================
Orchestrates the CSP solver across Latin, Occitan, Italian, and German,
ranks results, and compares with Phase 8 MDL findings.
"""

import json
import os
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_cell_lookup,
    load_corpus,
    tokenize_eva_chars,
)
from voynich.core.reference import (
    build_cv_syllable_table,
    load_reference_corpus,
)
from voynich.core.stats import build_ngram_lm, cross_entropy_lm

from voynich.phases.csp_constraints import (
    build_anchor_constraints,
    build_phoneme_inventory,
    score_cross_entropy,
    score_dict_hit_rate,
)
from voynich.phases.csp_solver import (
    CSPAssignment,
    _convert,
    ac3_propagate,
    beam_search,
    build_csp_variables,
    decode_corpus,
    decode_token,
    initialise_domains,
    score_assignment_full,
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class LanguageDecoding:
    """CSP decoding result for one language."""
    language: str
    best_cross_entropy: float
    best_word_validity: float
    best_dict_hit: float
    best_assignment: Dict[str, str]
    domain_sizes_initial: Dict[str, int]
    domain_sizes_pruned: Dict[str, int]
    anchor_match_count: int
    anchor_penalty: float
    decoded_sample: List[Any]
    top_5: List[Dict]
    runtime_seconds: float


@dataclass
class CSPDecodeResult:
    """Full Phase 11 output."""
    grid_cells_used: int
    eva_to_cell_mapping: Dict[str, str]
    language_results: Dict[str, Dict]
    language_ranking: List[Dict]
    best_language: str
    best_cross_entropy: float
    best_assignment: Dict[str, str]
    best_decoded_sample: List[Any]
    anchor_details: List[Dict]
    phase8_agreement: bool
    random_baseline_mean_ce: float
    selectivity: float
    gate_passed: bool
    verdict: str


# ---------------------------------------------------------------------------
# Random baseline
# ---------------------------------------------------------------------------

def _random_baseline_ce(
    cv_syllables: List[str],
    cell_keys: List[str],
    lm: Dict,
    voynich_tokens: List[str],
    eva_to_cell: Dict[str, str],
    n_trials: int = 200,
    max_tokens: int = 500,
    seed: int = 42,
) -> Tuple[float, float]:
    """Compute mean and best CE from random cell→syllable assignments.

    Returns (mean_ce, best_ce).
    """
    rng = random.Random(seed)
    ces: List[float] = []

    for _ in range(n_trials):
        mapping = {}
        shuffled = list(cv_syllables)
        rng.shuffle(shuffled)
        for i, cell_key in enumerate(cell_keys):
            mapping[cell_key] = shuffled[i % len(shuffled)]

        ce = score_cross_entropy(
            mapping, lm, voynich_tokens, eva_to_cell, max_tokens=max_tokens,
        )
        ces.append(ce)

    mean_ce = sum(ces) / len(ces) if ces else 99.0
    best_ce = min(ces) if ces else 99.0
    return mean_ce, best_ce


# ---------------------------------------------------------------------------
# Single-language CSP
# ---------------------------------------------------------------------------

def run_csp_for_language(
    language: str,
    corpus_tokens: List[str],
    ref_corpus: Any,
    cv_labels: Dict,
    rosetta_data: Dict,
    eva_to_cell: Dict[str, str],
    beam_width: int = 50,
    max_solutions: int = 20,
) -> LanguageDecoding:
    """Run the full CSP pipeline for one target language."""
    t0 = time.time()

    print(f"\n  --- {language.upper()} ---")

    # 1. Build phoneme inventory
    inventory = build_phoneme_inventory(language, ref_corpus)
    print(f"  CV syllable inventory: {len(inventory.cv_syllables)} syllables")

    # 2. Build language model (pass token list, not joined string)
    ref_tokens = ref_corpus.get_combined_tokens(language)
    if not ref_tokens:
        ref_tokens = ref_corpus.get_combined_tokens('latin')
    lm_tokens = ref_tokens[:10000] if ref_tokens else ['a', 'e', 'i', 'o', 'u']
    lm = build_ngram_lm(lm_tokens, order=3, smoothing=0.01)

    # Reference word set for dictionary hit rate
    ref_word_set = set(ref_tokens[:50000])

    # 3. Build anchor constraints
    anchors = build_anchor_constraints(rosetta_data, cv_labels)
    print(f"  Anchor constraints: {len(anchors)}")

    # 4. Build CSP variables
    variables = build_csp_variables(cv_labels)
    cell_frequencies = {v.cell_key: v.frequency for v in variables}

    # Record initial domain sizes
    domain_sizes_initial = {v.cv_label: len(inventory.cv_syllables) for v in variables}

    # 5. Initialise domains (Layers 1-3 + 5)
    variables = initialise_domains(
        variables, inventory, cell_frequencies, anchors, frequency_slack=3,
    )

    domain_sizes_after_init = {v.cv_label: len(v.domain) for v in variables}
    total_init = sum(len(v.domain) for v in variables)
    print(f"  Domains after Layer 1-3+5: {domain_sizes_after_init}")
    print(f"  Total domain values: {total_init}")

    # 6. AC-3 propagation
    solvable, variables = ac3_propagate(variables)
    domain_sizes_pruned = {v.cv_label: len(v.domain) for v in variables}
    total_pruned = sum(len(v.domain) for v in variables)
    print(f"  AC-3 solvable: {solvable}, total after: {total_pruned}")

    if not solvable:
        print(f"  [WARN] AC-3 found no solution — relaxing constraints")
        # Re-initialise with wider slack
        variables = build_csp_variables(cv_labels)
        variables = initialise_domains(
            variables, inventory, cell_frequencies, anchors,
            frequency_slack=6,
        )
        solvable, variables = ac3_propagate(variables)
        domain_sizes_pruned = {v.cv_label: len(v.domain) for v in variables}

    # 7. Beam search
    print(f"  Running beam search (width={beam_width})...")
    assignments = beam_search(
        variables, lm, corpus_tokens, eva_to_cell,
        anchors, inventory,
        ref_word_set=ref_word_set,
        beam_width=beam_width, max_solutions=max_solutions,
    )

    elapsed = time.time() - t0
    print(f"  Found {len(assignments)} assignments in {elapsed:.1f}s")

    if not assignments:
        return LanguageDecoding(
            language=language,
            best_cross_entropy=99.0,
            best_word_validity=0.0,
            best_dict_hit=0.0,
            best_assignment={},
            domain_sizes_initial=domain_sizes_initial,
            domain_sizes_pruned=domain_sizes_pruned,
            anchor_match_count=0,
            anchor_penalty=99.0,
            decoded_sample=[],
            top_5=[],
            runtime_seconds=elapsed,
        )

    best = assignments[0]

    # Print best result
    print(f"\n  Best assignment for {language}:")
    print(f"    Cross-entropy:  {best.cross_entropy:.4f}")
    print(f"    Word validity:  {best.word_validity:.4f}")
    print(f"    Dict hit rate:  {best.dict_hit_rate:.4f}")
    print(f"    Anchor matches: {best.anchor_match_count}/{len(anchors)}")
    print(f"    Composite score: {best.score:.4f}")
    print(f"\n  Phonetic table:")
    for cell_key, syl in sorted(best.mapping.items()):
        cv_label = cv_labels.get(cell_key, {}).get('cv_label', '?')
        glyphs = cv_labels.get(cell_key, {}).get('glyphs', [])
        print(f"    {cv_label} ({','.join(glyphs[:3])}) → {syl}")

    print(f"\n  Decoded sample (first 20 tokens):")
    for voyn, decoded in best.decoded_sample[:20]:
        print(f"    {voyn:20s} → {decoded}")

    top_5 = [_convert(asdict(a)) for a in assignments[:5]]

    return LanguageDecoding(
        language=language,
        best_cross_entropy=best.cross_entropy,
        best_word_validity=best.word_validity,
        best_dict_hit=best.dict_hit_rate,
        best_assignment=dict(best.mapping),
        domain_sizes_initial=domain_sizes_initial,
        domain_sizes_pruned=domain_sizes_pruned,
        anchor_match_count=best.anchor_match_count,
        anchor_penalty=best.anchor_penalty,
        decoded_sample=best.decoded_sample,
        top_5=top_5,
        runtime_seconds=elapsed,
    )


# ---------------------------------------------------------------------------
# Multi-language orchestrator
# ---------------------------------------------------------------------------

def run_csp_decode() -> Dict:
    """Phase 11 orchestrator: run CSP decoding for all candidate languages.

    1. Load corpus, reference corpus, grid data, Rosetta data
    2. Build EVA-to-cell mapping
    3. For each language: run CSP pipeline
    4. Rank by cross-entropy
    5. Compare with Phase 8 MDL ranking
    6. Gate check
    7. Save results
    """
    print("=" * 70)
    print("PHASE 11.2: Multi-Language CSP Phonetic Decoding")
    print("=" * 70)

    t0_total = time.time()

    # Load data
    print("\nLoading data...")
    corpus = load_corpus(verbose=False)
    ref_corpus = load_reference_corpus(verbose=False)

    rd = _results_dir()

    # Load cv_labels
    cv_labels_path = os.path.join(rd, 'cv_labels.json')
    with open(cv_labels_path) as f:
        cv_labels = json.load(f)
    print(f"  Grid cells: {len(cv_labels)}")

    # Load rosetta data
    rosetta_path = os.path.join(rd, 'rosetta_selection.json')
    with open(rosetta_path) as f:
        rosetta_data = json.load(f)
    print(f"  Rosetta folios: {rosetta_data.get('n_selected', 0)}")

    # Build EVA-to-cell mapping
    eva_to_cell = build_eva_to_cell_lookup(cv_labels)
    print(f"  EVA-to-cell mappings: {len(eva_to_cell)}")

    # Get Language A tokens
    corpus_tokens = corpus.get_tokens(language='A', paragraph_only=True)
    print(f"  Language A tokens: {len(corpus_tokens)}")

    # Run CSP for each language
    languages = ['latin', 'occitan', 'italian', 'german']
    language_results: Dict[str, LanguageDecoding] = {}

    for lang in languages:
        try:
            result = run_csp_for_language(
                lang, corpus_tokens, ref_corpus, cv_labels,
                rosetta_data, eva_to_cell,
                beam_width=50, max_solutions=20,
            )
            language_results[lang] = result
        except Exception as e:
            print(f"\n  [ERROR] {lang}: {e}")
            import traceback
            traceback.print_exc()

    # Rank by cross-entropy
    ranking = sorted(
        language_results.items(),
        key=lambda x: x[1].best_cross_entropy,
    )
    print("\n" + "=" * 70)
    print("LANGUAGE RANKING (by cross-entropy, lower = better)")
    print("=" * 70)
    for i, (lang, res) in enumerate(ranking):
        print(
            f"  {i+1}. {lang:10s}  CE={res.best_cross_entropy:.4f}  "
            f"validity={res.best_word_validity:.4f}  "
            f"anchors={res.anchor_match_count}  "
            f"time={res.runtime_seconds:.1f}s"
        )

    best_lang = ranking[0][0] if ranking else 'unknown'
    best_result = ranking[0][1] if ranking else None

    # Random baseline (use correctly-built LM)
    print("\nComputing random baseline...")
    cv_syllables = build_cv_syllable_table(best_lang)
    cell_keys = list(cv_labels.keys())
    ref_tokens = ref_corpus.get_combined_tokens(best_lang)
    if not ref_tokens:
        ref_tokens = ref_corpus.get_combined_tokens('latin')
    lm_tokens_baseline = ref_tokens[:10000] if ref_tokens else ['a', 'e', 'i', 'o', 'u']
    lm = build_ngram_lm(lm_tokens_baseline, order=3, smoothing=0.01)

    rand_mean_ce, rand_best_ce = _random_baseline_ce(
        cv_syllables, cell_keys, lm, corpus_tokens, eva_to_cell,
        n_trials=200,
    )
    selectivity = rand_mean_ce / best_result.best_cross_entropy if (
        best_result and best_result.best_cross_entropy > 0
    ) else 0.0

    print(f"  Random baseline mean CE: {rand_mean_ce:.4f}")
    print(f"  Random baseline best CE: {rand_best_ce:.4f}")
    print(f"  Best CSP CE: {best_result.best_cross_entropy:.4f}" if best_result else "  No result")
    print(f"  Selectivity (mean_random / best_CSP): {selectivity:.4f}")

    # Compare with Phase 8
    phase8_agreement = False
    try:
        mdl_path = os.path.join(rd, 'mdl_decode.json')
        if os.path.exists(mdl_path):
            with open(mdl_path) as f:
                mdl_data = json.load(f)
            mdl_ranking = mdl_data.get('language_ranking', [])
            if mdl_ranking:
                mdl_best = mdl_ranking[0].get('language', '')
                phase8_agreement = (mdl_best == best_lang)
                print(f"\n  Phase 8 MDL best: {mdl_best}")
                print(f"  Phase 11 CSP best: {best_lang}")
                print(f"  Agreement: {phase8_agreement}")
    except Exception:
        pass

    # Gate check
    gate_passed = selectivity >= 1.5
    if gate_passed:
        verdict = f"csp_decode_significant_selectivity_{selectivity:.2f}x"
    else:
        verdict = f"csp_decode_low_selectivity_{selectivity:.2f}x"

    print(f"\n  Gate: selectivity {'≥' if gate_passed else '<'} 1.5× → "
          f"{'PASSED' if gate_passed else 'FAILED'}")

    # Build anchor details
    anchor_details: List[Dict] = []
    if best_result:
        anchors = build_anchor_constraints(rosetta_data, cv_labels)
        for anchor in anchors:
            decoded_parts = []
            for cell in anchor.voynich_cells:
                syl = best_result.best_assignment.get(cell, '?')
                decoded_parts.append(syl)
            decoded_stem = ''.join(decoded_parts)
            target = ''.join(anchor.target_syllables)
            anchor_details.append({
                'folio': anchor.folio,
                'voynich_stem': anchor.voynich_stem,
                'target_word': anchor.target_word,
                'target_syllables': anchor.target_syllables,
                'decoded_stem': decoded_stem,
                'match': decoded_stem.lower() == target.lower(),
            })

    # Compile result
    decode_result = CSPDecodeResult(
        grid_cells_used=len(cv_labels),
        eva_to_cell_mapping=eva_to_cell,
        language_results={
            lang: _convert(asdict(res))
            for lang, res in language_results.items()
        },
        language_ranking=[
            {'rank': i + 1, 'language': lang,
             'cross_entropy': res.best_cross_entropy,
             'word_validity': res.best_word_validity,
             'anchor_matches': res.anchor_match_count}
            for i, (lang, res) in enumerate(ranking)
        ],
        best_language=best_lang,
        best_cross_entropy=best_result.best_cross_entropy if best_result else 99.0,
        best_assignment=best_result.best_assignment if best_result else {},
        best_decoded_sample=best_result.decoded_sample if best_result else [],
        anchor_details=anchor_details,
        phase8_agreement=phase8_agreement,
        random_baseline_mean_ce=rand_mean_ce,
        selectivity=selectivity,
        gate_passed=gate_passed,
        verdict=verdict,
    )

    # Save
    out_path = os.path.join(rd, 'csp_decode.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(decode_result)), f, indent=2)

    total_elapsed = time.time() - t0_total
    print(f"\n  Total time: {total_elapsed:.1f}s")
    print(f"  Results saved to results/csp_decode.json")

    return _convert(asdict(decode_result))
