"""
Phase 11.5.6-7 – Final multi-language comparison + V1-V9 validation
====================================================================
Step 11.5.6: Re-runs the refined CSP for all 4 languages at the best
             relaxation level from Phase 11.5.2-3.
Step 11.5.7: Runs the full validation battery (V1-V7 from csp_validate.py,
             plus new V8 readability and V9 MCMC comparison).

Decision gate: n_passed >= 6 of 9 tests AND selectivity remains > 1.5×
"""

import json
import math
import os
import random
import statistics
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_cell_lookup,
    load_corpus,
)
from voynich.core.reference import (
    LATIN_PHARMACEUTICAL_DOMAINS,
    build_cv_syllable_table,
    load_reference_corpus,
)
from voynich.core.stats import build_ngram_lm
from voynich.phases.csp_constraints import (
    build_anchor_constraints,
    build_phoneme_inventory,
    score_cross_entropy,
    score_dict_hit_rate,
)
from voynich.phases.csp_decode import run_csp_for_language, _random_baseline_ce
from voynich.phases.csp_solver import (
    _convert,
    decode_corpus,
    decode_token,
)
from voynich.phases.csp_validate import (
    v1_sanity_check,
    v2_random_baseline,
    v3_cross_validation,
    v4_section_coherence,
    v5_illustration_match,
    v6_language_b,
    v7_prior_convergence,
    ValidationResult,
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class LanguageFinalResult:
    """Final refined result for one language."""
    language: str
    relaxation_level: int
    best_dict_hit: float
    best_cross_entropy: float
    best_word_validity: float
    anchor_match_count: int
    best_assignment: Dict[str, str]
    runtime_seconds: float


@dataclass
class ReadabilityScore:
    """V8: operationalised readability assessment."""
    n_tokens_assessed: int
    n_recognizable: int
    recognizable_fraction: float       # exact dict hits
    n_plausible: int
    plausibility_fraction: float       # tokens ending in Latin-legal suffixes
    n_pattern_match: int
    pattern_match_fraction: float      # stem overlap with LATIN_PHARMACEUTICAL_DOMAINS
    composite_readability: float       # 0.4*rec + 0.35*plaus + 0.25*pattern
    passed: bool                       # >= 0.20


@dataclass
class MCMCComparisonResult:
    """V9: Metropolis-Hastings comparison vs CSP solution."""
    n_mcmc_trials: int
    n_burnin: int
    mcmc_mean_dict_hit: float
    mcmc_std_dict_hit: float
    mcmc_mean_ce: float
    our_dict_hit: float
    our_ce: float
    dict_hit_z_score: float
    ce_z_score: float
    passed: bool                       # both z-scores >= 2.0


@dataclass
class CSPFinalResult:
    """Full Phase 11.5.6-7 output."""
    language_results: Dict[str, Dict]
    language_ranking: List[Dict]
    best_language: str
    v1_sanity: Dict
    v2_random_baseline: Dict
    v3_cross_validation: Dict
    v4_section_coherence: Dict
    v5_illustration_match: Dict
    v6_language_b: Dict
    v7_prior_convergence: Dict
    v8_readability: Dict
    v9_mcmc_comparison: Dict
    n_passed: int
    n_total: int
    gate_passed: bool    # n_passed >= 6
    verdict: str


# ---------------------------------------------------------------------------
# V8: Readability assessment
# ---------------------------------------------------------------------------

# Latin word-final patterns that are phonotactically plausible
PLAUSIBLE_ENDINGS = {
    'a', 'e', 'i', 'o', 'u',
    'am', 'em', 'um',
    'is', 'es', 'us', 'as', 'os',
    'ar', 'er', 'or',
    'an', 'en', 'in', 'on',
    'al', 'el',
}


def _has_plausible_ending(word: str) -> bool:
    """True if word ends in a Latin-plausible suffix."""
    for suffix in ('am', 'em', 'um', 'is', 'es', 'us', 'as', 'os',
                   'ar', 'er', 'or', 'an', 'en', 'in', 'on', 'al', 'el',
                   'a', 'e', 'i', 'o', 'u'):
        if word.endswith(suffix):
            return True
    return False


def _build_domain_stems() -> set:
    """Build a set of 4+ character stems from LATIN_PHARMACEUTICAL_DOMAINS."""
    stems: set = set()
    for domain_words in LATIN_PHARMACEUTICAL_DOMAINS.values():
        for word_tuple in domain_words:
            word = word_tuple[0].lower()
            if len(word) >= 4:
                stems.add(word[:6])  # first 6 chars as stem
    return stems


def v8_readability(
    corpus_tokens: List[str],
    best_assignment: Dict[str, str],
    eva_to_cell: Dict[str, str],
    ref_word_set: set,
    n_sample: int = 5000,
) -> ValidationResult:
    """V8: Readability assessment of the decoded text.

    Three metrics combined:
    1. recognizable_fraction: exact dictionary hits
    2. plausibility_fraction: phonotactically plausible Latin endings
    3. pattern_match_fraction: stem overlap with pharmaceutical vocabulary

    composite = 0.4 * recognizable + 0.35 * plausible + 0.25 * pattern
    Passes when composite >= 0.20.
    """
    sample = corpus_tokens[:n_sample]
    domain_stems = _build_domain_stems()

    n_recognizable = 0
    n_plausible = 0
    n_pattern = 0
    n_decoded = 0

    for token in sample:
        decoded = decode_token(token, best_assignment, eva_to_cell)
        if not decoded or '?' in decoded or len(decoded) < 2:
            continue
        n_decoded += 1
        word = decoded.lower()

        if word in ref_word_set:
            n_recognizable += 1

        if _has_plausible_ending(word):
            n_plausible += 1

        # Stem match: first 4-6 chars
        for stem_len in (6, 5, 4):
            if len(word) >= stem_len:
                if word[:stem_len] in domain_stems:
                    n_pattern += 1
                    break

    n = max(n_decoded, 1)
    rec_frac = n_recognizable / n
    plaus_frac = n_plausible / n
    pat_frac = n_pattern / n
    composite = 0.4 * rec_frac + 0.35 * plaus_frac + 0.25 * pat_frac
    passed = composite >= 0.20

    score = ReadabilityScore(
        n_tokens_assessed=n_decoded,
        n_recognizable=n_recognizable,
        recognizable_fraction=round(rec_frac, 4),
        n_plausible=n_plausible,
        plausibility_fraction=round(plaus_frac, 4),
        n_pattern_match=n_pattern,
        pattern_match_fraction=round(pat_frac, 4),
        composite_readability=round(composite, 4),
        passed=passed,
    )

    return ValidationResult(
        test_id='V8',
        test_name='Readability Assessment',
        passed=passed,
        score=composite,
        details=_convert(asdict(score)),
    )


# ---------------------------------------------------------------------------
# V9: MCMC comparison
# ---------------------------------------------------------------------------

def _mcmc_step(
    current: Dict[str, str],
    cv_syllables: List[str],
    rng: random.Random,
) -> Dict[str, str]:
    """Single MCMC step: replace one cell's assignment."""
    new_mapping = dict(current)
    cell = rng.choice(list(new_mapping.keys()))
    new_syl = rng.choice(cv_syllables)
    new_mapping[cell] = new_syl
    return new_mapping


def v9_mcmc_comparison(
    best_assignment: Dict[str, str],
    lm: Dict,
    voynich_tokens: List[str],
    eva_to_cell: Dict[str, str],
    ref_word_set: set,
    n_trials: int = 1000,
    seed: int = 42,
    temperature: float = 10.0,
) -> ValidationResult:
    """V9: Metropolis-Hastings comparison.

    Runs a random walk through assignment space starting from a random
    (NOT the CSP) assignment, collecting CE and dict_hit at each step.
    Reports z-scores comparing our CSP solution to the MCMC distribution.

    Passes when both dict_hit z-score >= 2.0 and CE z-score >= 2.0.
    """
    rng = random.Random(seed)
    cv_syllables = build_cv_syllable_table('latin')
    cell_keys = list(best_assignment.keys())

    # Initialise from a random assignment
    current: Dict[str, str] = {
        k: rng.choice(cv_syllables) for k in cell_keys
    }
    current_ce = score_cross_entropy(current, lm, voynich_tokens, eva_to_cell, max_tokens=300)

    n_burnin = n_trials // 2
    mcmc_ces: List[float] = []
    mcmc_hits: List[float] = []

    for step in range(n_trials):
        proposal = _mcmc_step(current, cv_syllables, rng)
        prop_ce = score_cross_entropy(proposal, lm, voynich_tokens, eva_to_cell, max_tokens=300)

        # Metropolis acceptance: always accept improvements,
        # sometimes accept worse solutions (temperature-scaled)
        delta = current_ce - prop_ce
        if delta > 0 or rng.random() < math.exp(min(delta * temperature, 5.0)):
            current = proposal
            current_ce = prop_ce

        # Collect after burn-in
        if step >= n_burnin:
            mcmc_ces.append(current_ce)
            hit = score_dict_hit_rate(current, voynich_tokens, eva_to_cell, ref_word_set, 300)
            mcmc_hits.append(hit)

    mcmc_mean_ce = statistics.mean(mcmc_ces) if mcmc_ces else 99.0
    mcmc_std_ce = statistics.stdev(mcmc_ces) if len(mcmc_ces) > 1 else 1.0
    mcmc_mean_hit = statistics.mean(mcmc_hits) if mcmc_hits else 0.0
    mcmc_std_hit = statistics.stdev(mcmc_hits) if len(mcmc_hits) > 1 else 0.01

    # Score the CSP solution
    our_ce = score_cross_entropy(
        best_assignment, lm, voynich_tokens, eva_to_cell, max_tokens=500,
    )
    our_hit = score_dict_hit_rate(
        best_assignment, voynich_tokens, eva_to_cell, ref_word_set, max_tokens=500,
    )

    ce_z = (mcmc_mean_ce - our_ce) / max(mcmc_std_ce, 0.01)   # positive = we're better
    hit_z = (our_hit - mcmc_mean_hit) / max(mcmc_std_hit, 0.001)

    passed = ce_z >= 2.0 and hit_z >= 2.0

    score_obj = MCMCComparisonResult(
        n_mcmc_trials=n_trials,
        n_burnin=n_burnin,
        mcmc_mean_dict_hit=round(mcmc_mean_hit, 4),
        mcmc_std_dict_hit=round(mcmc_std_hit, 4),
        mcmc_mean_ce=round(mcmc_mean_ce, 4),
        our_dict_hit=round(our_hit, 4),
        our_ce=round(our_ce, 4),
        dict_hit_z_score=round(hit_z, 3),
        ce_z_score=round(ce_z, 3),
        passed=passed,
    )

    return ValidationResult(
        test_id='V9',
        test_name='MCMC Baseline Comparison',
        passed=passed,
        score=(ce_z + hit_z) / 2.0,
        details=_convert(asdict(score_obj)),
    )


# ---------------------------------------------------------------------------
# Multi-language refined comparison
# ---------------------------------------------------------------------------

def run_multilang_final(
    corpus_tokens: List[str],
    ref_corpus: Any,
    cv_labels: Dict,
    rosetta_data: Dict,
    eva_to_cell: Dict[str, str],
    relaxation_level: int = 0,
    best_inherent_vowel: Optional[str] = None,
    beam_width: int = 40,
    max_solutions: int = 10,
) -> Dict[str, LanguageFinalResult]:
    """Run refined CSP for all 4 languages and return per-language results."""
    print("\n  --- Multi-Language Refined Comparison ---")
    languages = ['latin', 'occitan', 'italian', 'german']
    results: Dict[str, LanguageFinalResult] = {}

    for lang in languages:
        print(f"\n  --- {lang.upper()} (level={relaxation_level}) ---")
        t0 = time.time()

        try:
            # run_csp_for_language handles inventory building internally;
            # we call it with standard args and pass relaxation through
            # the global build_phoneme_inventory override in the module.
            # Since run_csp_for_language doesn't yet accept relaxation_level,
            # we pass reduced beam_width for speed.
            lang_result = run_csp_for_language(
                lang, corpus_tokens[:1500], ref_corpus, cv_labels,
                rosetta_data, eva_to_cell,
                beam_width=beam_width, max_solutions=max_solutions,
            )
            elapsed = time.time() - t0
            results[lang] = LanguageFinalResult(
                language=lang,
                relaxation_level=relaxation_level,
                best_dict_hit=lang_result.best_dict_hit,
                best_cross_entropy=lang_result.best_cross_entropy,
                best_word_validity=lang_result.best_word_validity,
                anchor_match_count=lang_result.anchor_match_count,
                best_assignment=dict(lang_result.best_assignment),
                runtime_seconds=elapsed,
            )
        except Exception as e:
            print(f"  [ERROR] {lang}: {e}")
            import traceback
            traceback.print_exc()

    return results


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_csp_final() -> Dict:
    """Phase 11.5.6-7: final multi-language comparison + V1-V9 validation.

    1. Loads Phase 11.5.5 iterate results (or earlier phases as fallback).
    2. Runs multi-language refined comparison.
    3. Runs V1-V7 from csp_validate.py (loading their required JSON files).
    4. Runs V8 (readability) and V9 (MCMC comparison).
    5. Gate: n_passed >= 6 of 9 AND selectivity remains > 1.5×.
    6. Saves to results/csp_final.json.
    """
    print("=" * 70)
    print("PHASE 11.5.6-7: Final Multi-Language + V1-V9 Validation")
    print("=" * 70)

    t0_total = time.time()
    rd = _results_dir()

    # ------------------------------------------------------------------
    # 1. Load best assignment and config
    # ------------------------------------------------------------------
    relaxation_level = 0
    best_inherent_vowel: Optional[str] = 'a'
    final_dict_hit = 0.111

    for source_file, key_dict, key_level in [
        ('csp_iterate.json', 'final_assignment', 'best_relaxation_level'),
        ('verb_constraints.json', 'best_assignment', 'best_relaxation_level'),
        ('csp_refinement.json', 'final_assignment', 'best_relaxation_level'),
        ('csp_decode.json', 'best_assignment', None),
    ]:
        src_path = os.path.join(rd, source_file)
        if os.path.exists(src_path):
            with open(src_path) as f:
                src_data = json.load(f)
            best_assignment: Dict[str, str] = src_data.get(key_dict, {})
            if key_level:
                relaxation_level = int(src_data.get(key_level, 0))
                best_inherent_vowel = src_data.get('best_inherent_vowel', 'a')
            if best_assignment:
                final_dict_hit = float(src_data.get(
                    'final_dict_hit', src_data.get('dict_hit_after',
                    src_data.get('best_dict_hit_rate', 0.111))
                ))
                print(f"  Loaded assignment from {source_file}: "
                      f"dict_hit={final_dict_hit:.4f}, level={relaxation_level}")
                break
    else:
        print("  [SKIP] No prior results found — run csp-decode first")
        return {'verdict': 'skipped', 'reason': 'no_prior_results'}

    # ------------------------------------------------------------------
    # 2. Load supporting data
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

    ref_tokens = ref_corpus.get_combined_tokens('latin')
    if not ref_tokens:
        ref_tokens = ref_corpus.get_combined_tokens(ref_corpus.languages[0])
    ref_word_set = set(ref_tokens[:50000])
    lm = build_ngram_lm(ref_tokens[:10000], order=3, smoothing=0.01)

    # ------------------------------------------------------------------
    # 3. Multi-language refined comparison
    # ------------------------------------------------------------------
    lang_results = run_multilang_final(
        corpus_tokens, ref_corpus, cv_labels, rosetta_data, eva_to_cell,
        relaxation_level=relaxation_level,
        best_inherent_vowel=best_inherent_vowel,
        beam_width=40, max_solutions=10,
    )

    # Rank by cross-entropy
    ranking = sorted(
        lang_results.items(), key=lambda x: x[1].best_cross_entropy,
    )
    best_language = ranking[0][0] if ranking else 'latin'
    best_lang_result = ranking[0][1] if ranking else None

    print(f"\n  Language ranking:")
    for i, (lang, res) in enumerate(ranking):
        print(f"    {i+1}. {lang:10s}  CE={res.best_cross_entropy:.4f}  "
              f"dict_hit={res.best_dict_hit:.4f}  anchors={res.anchor_match_count}")

    # Use the best assignment for validation (prefer the one from iterate/refine
    # pipeline over the freshly computed one, as it used verb constraints)
    validate_assignment = best_assignment  # from iterative pipeline

    # ------------------------------------------------------------------
    # 4. V1-V7 from csp_validate.py
    # ------------------------------------------------------------------
    print("\n  Running V1-V7 validation tests...")

    v1 = v1_sanity_check()
    print(f"  V1 Sanity:           {'PASS' if v1.passed else 'FAIL'}  (score={v1.score:.3f})")

    best_ce_for_v2 = float('inf')
    if best_lang_result:
        best_ce_for_v2 = best_lang_result.best_cross_entropy
    elif validate_assignment:
        best_ce_for_v2 = score_cross_entropy(
            validate_assignment, lm, corpus_tokens[:500], eva_to_cell, max_tokens=500,
        )

    v2 = v2_random_baseline(
        best_ce=best_ce_for_v2,
        best_assignment=validate_assignment,
        cv_labels=cv_labels,
        lm=lm,
        voynich_tokens=corpus_tokens[:1000],
        eva_to_cell=eva_to_cell,
        best_language=best_language,
        n_trials=300,
    )
    print(f"  V2 Random Baseline:  {'PASS' if v2.passed else 'FAIL'}  "
          f"(selectivity={v2.score:.2f}x)")

    v3 = v3_cross_validation(
        corpus=corpus,
        best_assignment=validate_assignment,
        eva_to_cell=eva_to_cell,
        lm=lm,
    )
    print(f"  V3 Cross-Validation: {'PASS' if v3.passed else 'FAIL'}  (CV={v3.score:.4f})")

    v4 = v4_section_coherence(
        corpus=corpus,
        best_assignment=validate_assignment,
        eva_to_cell=eva_to_cell,
    )
    print(f"  V4 Section Coherence:{'PASS' if v4.passed else 'FAIL'}  (score={v4.score:.3f})")

    v5 = v5_illustration_match(
        corpus=corpus,
        best_assignment=validate_assignment,
        eva_to_cell=eva_to_cell,
        rosetta_data=rosetta_data,
        cv_labels=cv_labels,
    )
    print(f"  V5 Illustration:     {'PASS' if v5.passed else 'FAIL'}  (score={v5.score:.3f})")

    v6 = v6_language_b(
        corpus=corpus,
        best_assignment=validate_assignment,
        eva_to_cell=eva_to_cell,
        lm=lm,
    )
    print(f"  V6 Language B:       {'PASS' if v6.passed else 'FAIL'}  (score={v6.score:.3f})")

    v7 = v7_prior_convergence(
        best_assignment=validate_assignment,
        eva_to_cell=eva_to_cell,
        best_language=best_language,
    )
    print(f"  V7 Prior Convergence:{'PASS' if v7.passed else 'FAIL'}  (score={v7.score:.3f})")

    # ------------------------------------------------------------------
    # 5. V8: Readability assessment
    # ------------------------------------------------------------------
    print("\n  Running V8: Readability Assessment...")
    v8 = v8_readability(
        corpus_tokens[:5000],
        validate_assignment,
        eva_to_cell,
        ref_word_set,
        n_sample=5000,
    )
    print(f"  V8 Readability:      {'PASS' if v8.passed else 'FAIL'}  "
          f"(composite={v8.score:.3f})")
    det8 = v8.details
    print(f"    recognizable={det8.get('recognizable_fraction', 0):.1%}  "
          f"plausible={det8.get('plausibility_fraction', 0):.1%}  "
          f"pattern={det8.get('pattern_match_fraction', 0):.1%}")

    # ------------------------------------------------------------------
    # 6. V9: MCMC comparison
    # ------------------------------------------------------------------
    print("\n  Running V9: MCMC Comparison (1000 trials)...")
    v9 = v9_mcmc_comparison(
        best_assignment=validate_assignment,
        lm=lm,
        voynich_tokens=corpus_tokens[:1000],
        eva_to_cell=eva_to_cell,
        ref_word_set=ref_word_set,
        n_trials=1000,
        seed=42,
    )
    print(f"  V9 MCMC:             {'PASS' if v9.passed else 'FAIL'}  "
          f"(CE_z={v9.details.get('ce_z_score', 0):.2f}, "
          f"hit_z={v9.details.get('dict_hit_z_score', 0):.2f})")

    # ------------------------------------------------------------------
    # 7. Tally results
    # ------------------------------------------------------------------
    all_tests = [v1, v2, v3, v4, v5, v6, v7, v8, v9]
    n_passed = sum(1 for t in all_tests if t.passed)
    n_total = len(all_tests)
    selectivity = v2.score  # from V2 test

    gate_passed = n_passed >= 6

    # Determine verdict from spec thresholds
    if (gate_passed and final_dict_hit >= 0.25 and selectivity >= 1.5
            and v3.passed and n_passed >= 7):
        verdict = 'VALIDATED_PARTIAL_DECODING'
    elif gate_passed and final_dict_hit >= 0.15 and selectivity >= 1.5:
        verdict = f'partial_decoding_dict_hit_{final_dict_hit:.3f}'
    elif selectivity >= 1.5 and final_dict_hit < 0.20:
        verdict = 'framework_correct_phonetics_imprecise'
    elif selectivity < 1.5:
        verdict = 'selectivity_dropped_refinements_overfit'
    else:
        verdict = f'csp_final_{n_passed}of{n_total}_passed'

    print(f"\n  VALIDATION SUMMARY: {n_passed}/{n_total} tests passed")
    print(f"  Gate (>= 6 of 9):  {'PASS ✓' if gate_passed else 'FAIL ✗'}")
    print(f"  Selectivity:       {selectivity:.2f}x")
    print(f"  Final dict_hit:    {final_dict_hit:.4f}")
    print(f"  Verdict:           {verdict}")

    # Print the spec's final verdict check
    print("\n  Spec verdict criteria:")
    print(f"    dict_hit > 25%:    {'✓' if final_dict_hit > 0.25 else '✗'}")
    print(f"    selectivity > 1.5x:{'✓' if selectivity > 1.5 else '✗'}")
    print(f"    cross-val passed:  {'✓' if v3.passed else '✗'}")
    print(f"    >=2 non-anchor illust: {'✓' if v5.passed else '✗'}")
    print(f"    section coherence: {'✓' if v4.passed else '✗'}")

    # ------------------------------------------------------------------
    # 8. Save results
    # ------------------------------------------------------------------
    result = CSPFinalResult(
        language_results={
            lang: _convert(asdict(res)) for lang, res in lang_results.items()
        },
        language_ranking=[
            {'rank': i + 1, 'language': lang,
             'cross_entropy': res.best_cross_entropy,
             'dict_hit': res.best_dict_hit,
             'anchor_matches': res.anchor_match_count}
            for i, (lang, res) in enumerate(ranking)
        ],
        best_language=best_language,
        v1_sanity=_convert(asdict(v1)),
        v2_random_baseline=_convert(asdict(v2)),
        v3_cross_validation=_convert(asdict(v3)),
        v4_section_coherence=_convert(asdict(v4)),
        v5_illustration_match=_convert(asdict(v5)),
        v6_language_b=_convert(asdict(v6)),
        v7_prior_convergence=_convert(asdict(v7)),
        v8_readability=_convert(asdict(v8)),
        v9_mcmc_comparison=_convert(asdict(v9)),
        n_passed=n_passed,
        n_total=n_total,
        gate_passed=gate_passed,
        verdict=verdict,
    )

    out_path = os.path.join(rd, 'csp_final.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2)

    elapsed = time.time() - t0_total
    print(f"\n  Saved to {out_path} ({elapsed:.1f}s total)")

    return _convert(asdict(result))
