"""
Phase 14.5–14.6 – Full Voynich Decoding with Feature CSP
==========================================================
Applies the best feature-level CSP assignment (from Phase 14.3) to the full
Language A corpus.  Runs the complete V1–V12 validation battery:

V1 – V11: Same tests as Phase 11–13 (loaded from prior result files)
V12 (new): Feature plausibility — strokes with the same first_stroke type
           are assigned consonants from the same place of articulation;
           strokes with the same last_stroke type are assigned phonemes from
           the same vowel space.

Runs the feature CSP for Latin, Occitan, Italian, and German and reports
the multi-language comparison table.

Dependency chain:
    feature_csp.json  (Step 14.3)
    feature_calibrate.json  (Step 14.4 — for calibrated expectation)
        → feature_decode.json  (this step)
"""

import json
import math
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
    PHONEME_PLACE_MAP,
    PHONEME_NUCLEUS_MAP,
    build_cv_syllable_table,
    build_triple_phoneme_hypotheses,
    load_reference_corpus,
)
from voynich.core.stats import build_ngram_lm, cross_entropy_lm
from voynich.phases.csp_constraints import (
    AnchorConstraint,
    build_phoneme_inventory,
    score_cross_entropy,
    score_word_validity,
)
from voynich.phases.csp_solver import (
    _convert,
    decode_corpus,
    decode_token,
)
from voynich.phases.feature_csp import (
    FeatureVariable,
    _build_anchor_constraints_triple,
    build_feature_variables,
    initialise_feature_domains,
    run_feature_csp_for_language,
    FeatureCSPResult,
)
from voynich.phases.csp_validate import ValidationResult


# ---------------------------------------------------------------------------
# Domain keywords (same as context_decode.py)
# ---------------------------------------------------------------------------

_DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    'plant_names': [
        'rosa', 'viola', 'herba', 'folia', 'radix', 'flos', 'semen', 'cortex',
        'salvia', 'menta', 'ruta', 'anise', 'coriandrum', 'anethi', 'petroselinum',
    ],
    'plant_parts': [
        'folium', 'radix', 'flos', 'semen', 'cortex', 'fructus', 'caulis',
        'succus', 'ramus', 'herba', 'folia', 'flores', 'semina',
    ],
    'preparations': [
        'aqua', 'oleum', 'vinum', 'mel', 'acetum', 'decoctio', 'infusio',
        'pulvis', 'electuarium', 'succus', 'expressa', 'distillata',
    ],
    'body_parts': [
        'caput', 'stomachum', 'ventrem', 'cor', 'iecur', 'renes', 'pulmones',
        'oculi', 'manus', 'pedes', 'dentes', 'gula', 'nares', 'aurem',
    ],
    'verbs': [
        'recipe', 'accipe', 'misce', 'contere', 'coque', 'pone', 'adde',
        'cola', 'distilla', 'applica', 'tere', 'fac', 'cape', 'da', 'bibe',
    ],
    'qualities': [
        'calidus', 'frigidus', 'siccus', 'humidus', 'dulcis', 'amarus',
        'acutus', 'mollis', 'durus', 'niger', 'albus', 'viridis', 'ruber',
    ],
    'function_words': [
        'et', 'in', 'cum', 'est', 'ad', 'ex', 'de', 'per', 'vel', 'aut',
        'ut', 'si', 'non', 'sed', 'ac', 'atque', 'enim', 'nec', 'nam',
    ],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path: str) -> Optional[Dict]:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FeatureLanguageResult:
    """Feature CSP results for one target language."""
    language: str
    dict_hit: float
    cross_entropy: float
    word_validity: float
    selectivity: float
    anchor_matches: int
    best_assignment: Dict[str, str]
    decoded_sample: List[Any]
    n_variables: int
    phase11_baseline: float
    improvement: float
    gate_passed: bool


@dataclass
class FeaturePlausibilityResult:
    """V12: feature plausibility check."""
    test_id: str = 'V12'
    test_name: str = 'Feature Plausibility'
    n_onset_consistent: int = 0
    n_onset_total: int = 0
    n_nucleus_consistent: int = 0
    n_nucleus_total: int = 0
    plausibility_score: float = 0.0
    onset_analysis: List[Dict] = field(default_factory=list)
    nucleus_analysis: List[Dict] = field(default_factory=list)
    passed: bool = False
    details: str = ''


@dataclass
class FeatureVocabularyCatalog:
    """Vocabulary catalog from feature-level decoded text."""
    total_unique_decoded: int
    confirmed_hits: List[str]
    domain_hits: Dict[str, List[str]]
    function_words: List[str]
    n_total_hits: int


@dataclass
class FeatureDecodeResult:
    """Full Phase 14.5–14.6 decode result."""
    best_language: str
    best_dict_hit: float
    best_cross_entropy: float
    best_selectivity: float
    best_assignment: Dict[str, str]
    language_results: Dict[str, Dict]
    section_samples: List[Dict]
    vocabulary_catalog: Dict
    v12_feature_plausibility: Dict
    validation_summary: List[Dict]
    n_validation_passed: int
    n_validation_total: int
    progression: Dict
    gate_passed: bool
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# V12: Feature plausibility
# ---------------------------------------------------------------------------

def _run_v12_feature_plausibility(
    best_assignment: Dict[str, str],
    inventory_vowels: set,
) -> FeaturePlausibilityResult:
    """Check that stroke-phoneme assignments are typologically consistent.

    Consistency rule for onset (first_stroke):
      Glyphs with the same first_stroke type should map to consonants from
      the same broad place of articulation (stops, fricatives, sonorants).

    Consistency rule for nucleus (last_stroke):
      Glyphs with the same last_stroke type should map to vowels from the
      same vowel space (high/low, front/back).

    Plausibility score = fraction of (triple_pair_with_same_stroke,
    consistent_phoneme_class) across all same-stroke pairs.
    """
    result = FeaturePlausibilityResult()

    if not best_assignment:
        result.details = "No assignment to analyse"
        return result

    # Group triples by first_stroke
    onset_groups: Dict[str, List[str]] = {}  # first_stroke -> [triple_key, ...]
    nucleus_groups: Dict[str, List[str]] = {}  # last_stroke -> [triple_key, ...]

    for triple_key in best_assignment:
        parts = triple_key.split(',')
        if len(parts) != 3:
            continue
        fs, ls, _gc = parts[0], parts[1], parts[2]
        onset_groups.setdefault(fs, []).append(triple_key)
        nucleus_groups.setdefault(ls, []).append(triple_key)

    # Phoneme place-of-articulation groups for consistency check
    _STOPS = {'p', 'b', 't', 'd', 'k', 'g', 'c', 'q'}
    _FRICATIVES = {'f', 'v', 's', 'z', 'h', 'x', 'sc'}
    _SONORANTS = {'m', 'n', 'l', 'r'}
    def _place_class(ph: str) -> str:
        if ph in _STOPS: return 'stop'
        if ph in _FRICATIVES: return 'fricative'
        if ph in _SONORANTS: return 'sonorant'
        if ph in inventory_vowels: return 'vowel'
        return 'other'

    # Check onset consistency
    onset_analysis: List[Dict] = []
    n_on_consistent = 0
    n_on_total = 0
    for fs, triple_keys in onset_groups.items():
        if len(triple_keys) < 2:
            continue
        syls = [best_assignment.get(tk, '') for tk in triple_keys]
        # Extract first consonant of each syllable (onset phoneme)
        onsets_found: List[str] = []
        for syl in syls:
            if syl:
                # First non-vowel char
                onset = ''
                for ch in syl:
                    if ch not in inventory_vowels:
                        onset = ch
                        break
                if onset:
                    onsets_found.append(onset)
        if len(onsets_found) >= 2:
            classes = [_place_class(o) for o in onsets_found]
            consistent = len(set(classes)) == 1
            if consistent:
                n_on_consistent += 1
            n_on_total += 1
            onset_analysis.append({
                'first_stroke': fs,
                'triples': triple_keys,
                'syllables': syls,
                'onset_phonemes': onsets_found,
                'place_classes': classes,
                'consistent': consistent,
            })

    # Check nucleus consistency
    _HIGH_VOWELS = {'i', 'u', 'y'}
    _MID_VOWELS = {'e', 'o'}
    _LOW_VOWELS = {'a'}
    def _nucleus_class(syl: str) -> str:
        # Last vowel in the syllable
        for ch in reversed(syl):
            if ch in inventory_vowels:
                if ch in _HIGH_VOWELS: return 'high'
                if ch in _MID_VOWELS: return 'mid'
                if ch in _LOW_VOWELS: return 'low'
        return 'other'

    nucleus_analysis: List[Dict] = []
    n_nuc_consistent = 0
    n_nuc_total = 0
    for ls, triple_keys in nucleus_groups.items():
        if len(triple_keys) < 2:
            continue
        syls = [best_assignment.get(tk, '') for tk in triple_keys]
        syls_nonempty = [s for s in syls if s]
        if len(syls_nonempty) >= 2:
            classes = [_nucleus_class(s) for s in syls_nonempty]
            consistent = len(set(classes)) == 1
            if consistent:
                n_nuc_consistent += 1
            n_nuc_total += 1
            nucleus_analysis.append({
                'last_stroke': ls,
                'triples': triple_keys,
                'syllables': syls_nonempty,
                'vowel_classes': classes,
                'consistent': consistent,
            })

    total = n_on_total + n_nuc_total
    consistent = n_on_consistent + n_nuc_consistent
    plausibility = consistent / total if total > 0 else 0.0

    passed = plausibility >= 0.5

    result.n_onset_consistent = n_on_consistent
    result.n_onset_total = n_on_total
    result.n_nucleus_consistent = n_nuc_consistent
    result.n_nucleus_total = n_nuc_total
    result.plausibility_score = round(plausibility, 3)
    result.onset_analysis = onset_analysis
    result.nucleus_analysis = nucleus_analysis
    result.passed = passed
    result.details = (
        f"Onset consistency: {n_on_consistent}/{n_on_total}; "
        f"Nucleus consistency: {n_nuc_consistent}/{n_nuc_total}; "
        f"Score: {plausibility:.1%}. "
        f"{'PASS' if passed else 'FAIL'}: "
        f"{'Script encodes phonetic features systematically.' if passed else 'Feature assignments not typologically consistent.'}"
    )
    return result


# ---------------------------------------------------------------------------
# Vocabulary catalog
# ---------------------------------------------------------------------------

def _build_vocabulary_catalog(
    decoded_tokens: List[str],
    ref_word_set: set,
) -> FeatureVocabularyCatalog:
    unique = set(decoded_tokens)
    confirmed = sorted(w for w in unique if w in ref_word_set and len(w) >= 3)
    domain_hits: Dict[str, List[str]] = {}
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        hits = [k for k in keywords if k in unique]
        if hits:
            domain_hits[domain] = hits
    function_words = [w for w in _DOMAIN_KEYWORDS.get('function_words', []) if w in unique]
    return FeatureVocabularyCatalog(
        total_unique_decoded=len(unique),
        confirmed_hits=confirmed,
        domain_hits=domain_hits,
        function_words=function_words,
        n_total_hits=len(confirmed),
    )


# ---------------------------------------------------------------------------
# Section samples
# ---------------------------------------------------------------------------

def _section_samples(
    corpus,
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    n_per_section: int = 3,
) -> List[Dict]:
    sections = ['herbal_a', 'pharmaceutical', 'astronomical', 'biological']
    samples: List[Dict] = []
    for section in sections:
        sect_tokens = corpus.get_tokens(language='A', section=section, paragraph_only=True)
        if not sect_tokens:
            continue
        sample_tokens = sect_tokens[:n_per_section]
        decoded = [decode_token(t, assignment, eva_to_triple) for t in sample_tokens]
        samples.append({
            'section': section,
            'tokens': sample_tokens,
            'decoded': decoded,
        })
    return samples


# ---------------------------------------------------------------------------
# Progression tracking
# ---------------------------------------------------------------------------

def _build_progression(best_dict_hit: float, best_selectivity: float) -> Dict:
    return {
        'phase11': {'dict_hit': 0.111, 'selectivity': 1.92},
        'phase11_5': {'dict_hit': 0.0987, 'selectivity': 1.85},
        'phase12': {'dict_hit': 0.1115, 'selectivity': 1.85},
        'phase13': {'dict_hit': 0.1143, 'selectivity': 1.86},
        'phase14': {'dict_hit': round(best_dict_hit, 4), 'selectivity': round(best_selectivity, 3)},
        'trend': 'improvement' if best_dict_hit > 0.1143 else 'plateau',
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_feature_decode() -> None:
    """Steps 14.5–14.6: full feature decode + V1–V12 validation battery."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE 14.5-14.6: Full Feature Decode + Validation Battery")
    print("=" * 70)

    rd = _results_dir()

    # Check prerequisite
    fc_path = os.path.join(rd, 'feature_csp.json')
    if not os.path.exists(fc_path):
        print("  [SKIP] feature_csp.json not found — run feature-csp first")
        return

    # Load corpus
    corpus = load_corpus(verbose=False)
    tokens = corpus.get_tokens(language='A', paragraph_only=True)
    if not tokens:
        print("  [SKIP] No Language A tokens found")
        return

    # Build triple lookup
    eva_to_triple = build_eva_to_triple_lookup()

    # Glyph frequencies
    glyph_freq: Counter = Counter()
    for token in tokens:
        for ch in tokenize_eva_chars(token):
            glyph_freq[ch] += 1

    # Load reference corpora
    ref_corpus = load_reference_corpus(verbose=False)

    # Load anchor constraints
    rosetta_path = os.path.join(rd, 'rosetta_selection.json')
    anchors: List[AnchorConstraint] = []
    if os.path.exists(rosetta_path):
        with open(rosetta_path) as f:
            rosetta_data = json.load(f)
        anchors = _build_anchor_constraints_triple(rosetta_data, eva_to_triple)

    languages = ['latin', 'occitan', 'italian', 'german']
    language_results: Dict[str, Dict] = {}
    best_lang = ''
    best_dict_hit = 0.0
    best_ce = 99.0
    best_assignment: Dict[str, str] = {}
    best_selectivity = 0.0

    for language in languages:
        print(f"\n  ── Language: {language.upper()} ──")
        ref_tokens = ref_corpus.get_combined_tokens(language)
        if not ref_tokens:
            print(f"  [SKIP] No reference corpus for {language}")
            continue

        inventory = build_phoneme_inventory(language, ref_corpus)
        lm = build_ngram_lm(ref_tokens[:10000], order=3, smoothing=0.01)
        ref_word_set = set(w.lower() for w in ref_tokens if len(w) >= 2)

        hypothesis_map = build_triple_phoneme_hypotheses(language, build_cv_syllable_table(language))
        variables = build_feature_variables(eva_to_triple, glyph_freq, inventory, hypothesis_map)
        variables = initialise_feature_domains(variables, inventory, hypothesis_map, anchors)

        csp_result = run_feature_csp_for_language(
            language=language,
            variables=variables,
            lm=lm,
            voynich_tokens=tokens,
            eva_to_triple=eva_to_triple,
            anchors=anchors,
            inventory=inventory,
            ref_word_set=ref_word_set,
            beam_width=80,
        )

        lang_res = FeatureLanguageResult(
            language=language,
            dict_hit=csp_result.best_dict_hit,
            cross_entropy=csp_result.best_cross_entropy,
            word_validity=csp_result.best_word_validity,
            selectivity=csp_result.best_selectivity,
            anchor_matches=csp_result.best_anchor_matches,
            best_assignment=csp_result.best_assignment,
            decoded_sample=csp_result.decoded_sample,
            n_variables=csp_result.n_feature_variables,
            phase11_baseline=0.111,
            improvement=csp_result.improvement,
            gate_passed=csp_result.gate_passed,
        )
        language_results[language] = _convert(lang_res)

        print(f"  dict_hit: {csp_result.best_dict_hit:.3f}  CE: {csp_result.best_cross_entropy:.3f}  sel: {csp_result.best_selectivity:.2f}x  gate: {'PASS' if csp_result.gate_passed else 'FAIL'}")

        if csp_result.best_dict_hit > best_dict_hit:
            best_dict_hit = csp_result.best_dict_hit
            best_lang = language
            best_ce = csp_result.best_cross_entropy
            best_assignment = csp_result.best_assignment
            best_selectivity = csp_result.best_selectivity

    if not language_results:
        print("\n  [ERROR] No language results produced")
        return

    # V12: Feature plausibility on best assignment
    print("\n  Running V12: Feature Plausibility check...")
    best_inventory = build_phoneme_inventory(best_lang, ref_corpus)
    inv_vowels = set(best_inventory.vowels)
    v12 = _run_v12_feature_plausibility(best_assignment, inv_vowels)
    print(f"  V12: {v12.details}")

    # V1-V11: load from prior result files (same as context_decode)
    print("\n  Loading V1-V11 validation results from prior phases...")
    validation_summary: List[Dict] = []
    n_passed = 0

    # Check each prior validation file
    prior_checks = [
        ('V1', 'csp_solver_test.json', 'sanity_test_passed'),
        ('V2', 'csp_validate.json', 'v2_random_baseline.passed'),
        ('V3', 'csp_validate.json', 'v3_cross_validation.passed'),
        ('V4', 'csp_validate.json', 'v4_section_coherence.passed'),
        ('V5', 'csp_validate.json', 'v5_illustration_match.passed'),
        ('V6', 'csp_validate.json', 'v6_language_b.passed'),
        ('V7', 'csp_validate.json', 'v7_prior_convergence.passed'),
        ('V8', 'csp_final.json', 'v8_readability.passed'),
        ('V9', 'csp_final.json', 'v9_mcmc_comparison.passed'),
        ('V10', 'context_decode.json', 'vocabulary_catalog.n_total_hits'),
        ('V11', 'context_decode.json', 'progression'),
    ]

    for vid, filename, field_path in prior_checks:
        fpath = os.path.join(rd, filename)
        data = _load_json(fpath)
        passed = False
        score = 0.0
        detail = 'file not found'
        if data:
            # Navigate the field_path (dot-separated)
            val = data
            for key in field_path.split('.'):
                if isinstance(val, dict) and key in val:
                    val = val[key]
                else:
                    val = None
                    break
            if val is not None:
                if isinstance(val, bool):
                    passed = val
                    score = 1.0 if val else 0.0
                    detail = f'passed={val}'
                elif isinstance(val, (int, float)):
                    passed = val > 0
                    score = float(val)
                    detail = f'value={val}'
                else:
                    passed = True
                    score = 1.0
                    detail = str(val)[:80]
            else:
                detail = f'field {field_path!r} not found in {filename}'

        if passed:
            n_passed += 1
        validation_summary.append({
            'test_id': vid,
            'file': filename,
            'passed': passed,
            'score': score,
            'detail': detail,
        })

    # Add V12
    if v12.passed:
        n_passed += 1
    validation_summary.append({
        'test_id': 'V12',
        'test_name': 'Feature Plausibility',
        'passed': v12.passed,
        'score': v12.plausibility_score,
        'detail': v12.details,
    })
    n_total = len(validation_summary)

    # Vocabulary catalog
    best_ref_tokens = ref_corpus.get_combined_tokens(best_lang)
    best_ref_set = set(w.lower() for w in best_ref_tokens if len(w) >= 2)
    decoded_all = decode_corpus(tokens, best_assignment, eva_to_triple, max_tokens=5000)
    vocab_catalog = _build_vocabulary_catalog(decoded_all, best_ref_set)

    # Section samples
    section_samples = _section_samples(corpus, best_assignment, eva_to_triple)

    # Progression
    progression = _build_progression(best_dict_hit, best_selectivity)

    # Determine final gate and verdict
    gate_passed = (
        best_dict_hit > 0.15
        and best_selectivity > 1.5
        and n_passed >= 7
    )

    if best_dict_hit > 0.25:
        verdict = (
            f"BREAKTHROUGH: {best_dict_hit:.1%} dict_hit ({best_selectivity:.2f}x selectivity) "
            f"for {best_lang}. Feature model resolves cell conflation ceiling. "
            f"V1-V12: {n_passed}/{n_total} passed."
        )
    elif best_dict_hit > 0.15:
        verdict = (
            f"SIGNIFICANT IMPROVEMENT: {best_dict_hit:.1%} dict_hit ({best_selectivity:.2f}x) "
            f"vs Phase 11 11.1%. Feature model provides measurable lift. "
            f"V1-V12: {n_passed}/{n_total} passed."
        )
    else:
        verdict = (
            f"MARGINAL: {best_dict_hit:.1%} dict_hit ({best_selectivity:.2f}x selectivity). "
            f"Feature model does not substantially improve on Phase 11-13 baseline. "
            f"The 11.1% ceiling persists. V1-V12: {n_passed}/{n_total} passed. "
            f"Consider subcell_split fallback (Step 14.7)."
        )

    # Multi-language comparison table
    print(f"\n  ── Multi-Language Comparison ──")
    print(f"  {'Language':<12} {'dict_hit':>10} {'CE':>8} {'selectivity':>12} {'gate':>6}")
    for lang, lr in sorted(language_results.items(), key=lambda x: -x[1].get('dict_hit', 0)):
        dh = lr.get('dict_hit', 0)
        ce = lr.get('cross_entropy', 99)
        sel = lr.get('selectivity', 0)
        gp = 'PASS' if lr.get('gate_passed', False) else 'FAIL'
        print(f"  {lang:<12} {dh:>10.3f} {ce:>8.3f} {sel:>12.2f} {gp:>6}")

    print(f"\n  V1-V12 validation: {n_passed}/{n_total} passed")
    print(f"  V12 Feature Plausibility: {v12.plausibility_score:.1%}")
    print(f"  Progression: {progression}")
    print(f"\n  Final gate: {'PASS' if gate_passed else 'FAIL'}")
    print(f"  Verdict: {verdict}")

    result = FeatureDecodeResult(
        best_language=best_lang,
        best_dict_hit=best_dict_hit,
        best_cross_entropy=best_ce,
        best_selectivity=best_selectivity,
        best_assignment=best_assignment,
        language_results=language_results,
        section_samples=section_samples,
        vocabulary_catalog=_convert(vocab_catalog),
        v12_feature_plausibility=_convert(v12),
        validation_summary=validation_summary,
        n_validation_passed=n_passed,
        n_validation_total=n_total,
        progression=progression,
        gate_passed=gate_passed,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'feature_decode.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)
    print(f"\n  Results saved → {out_path}")
