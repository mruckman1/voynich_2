"""
Phase C.6 -- Full Validation Battery (17 Tests)
=================================================
Run the comprehensive V1-V17 validation suite on all Phase C results.
This is the final gate for Phase C: the paleographic-prior-informed
CSP decode must pass >= 12/17 tests (excluding SKIPs from the
denominator) AND must pass V13 (phrase selectivity) and V15 (null
corpus control).

Tests:
    V1  Sanity check (dict_hit > 0)
    V2  Random baseline selectivity > 1.5
    V3  Cross-validation CV < 0.10
    V4  Section coherence (herbal/pharma/cosmo vocabulary differences)
    V5  Illustration match (botanical folios -> plant words)
    V6  Language B consistency
    V7  Prior-phase convergence
    V8  Readability assessment
    V9  Comparison to all prior phases
    V10 Vocabulary catalog by semantic domain
    V11 Progression tracking
    V12 Feature plausibility
    V13 Phrase selectivity > 2x  (from phrase_detect.json)
    V14 Domain coverage >= 3/6 semantic domains with hits
    V15 Null corpus control
    V16 Keyword presence (top-100 Latin words)
    V17 Verb decode (>= 5/15 at edit distance <= 1)

Dependency chain:
    All Phase C result files
        -> phaseC_validate.json (this step)
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
    tokenize_eva_chars,
)
from voynich.core.reference import (
    build_cv_syllable_table,
    build_expanded_word_set,
    load_reference_corpus,
)
from voynich.phases.csp_constraints import build_phoneme_inventory
from voynich.phases.csp_solver import decode_corpus, decode_token


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
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def _get_nested(data: Dict, field_path: str) -> Any:
    """Navigate a dot-separated field path in nested dicts."""
    val = data
    for key in field_path.split('.'):
        if isinstance(val, dict) and key in val:
            val = val[key]
        else:
            return None
    return val


def _edit_distance(s1: str, s2: str) -> int:
    """Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return _edit_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    return prev_row[-1]


# ---------------------------------------------------------------------------
# Word lists for validation
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
}

# Top-100 Latin words (function words + common content words)
_TOP_100_LATIN = {
    'et', 'in', 'est', 'non', 'ut', 'cum', 'ad', 'de', 'si', 'sed',
    'ex', 'per', 'aut', 'vel', 'nec', 'ab', 'ac', 'quod', 'enim',
    'nam', 'hoc', 'tam', 'pro', 'sic', 'iam', 'sub', 'ante', 'post',
    'super', 'inter', 'contra', 'aqua', 'terra', 'ignis', 'aer',
    'herba', 'radix', 'flos', 'folia', 'semen', 'cortex', 'vinum',
    'oleum', 'mel', 'sal', 'acetum', 'pulvis', 'succus', 'opus',
    'dies', 'nox', 'mane', 'hora', 'pars', 'corpus', 'caput', 'cor',
    'sanguis', 'febris', 'morbus', 'dolor', 'cura', 'virtus', 'vis',
    'rex', 'deus', 'homo', 'vir', 'res', 'locus', 'tempus', 'annus',
    'modus', 'genus', 'causa', 'ratio', 'vita', 'mors', 'finis',
    'ordo', 'ars', 'lex', 'pax', 'lux', 'vox', 'nomen', 'verbum',
    'liber', 'manus', 'oculus', 'pes', 'dens', 'os', 'auris',
    'calidus', 'frigidus', 'siccus', 'humidus', 'bonus', 'malus',
    'magnus', 'parvus', 'novus', 'vetus', 'albus', 'niger',
}

# 15 target verbs from verb_identification.json
_TARGET_VERBS = [
    'recipe', 'accipe', 'misce', 'contere', 'coque', 'pone', 'adde',
    'cola', 'distilla', 'applica', 'tere', 'fac', 'cape', 'da', 'bibe',
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PhaseCValidateResult:
    """Phase C.6: full validation battery."""
    test_results: List[Dict]
    n_passed: int
    n_failed: int
    n_skipped: int
    n_total: int
    pass_rate: float
    v13_passed: bool
    v15_passed: bool
    gate_passed: bool
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_phaseC_validate() -> None:
    """Phase C.6: Full validation battery (17 tests)."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE C.6: Full Validation Battery (V1-V17)")
    print("=" * 70)

    rd = _results_dir()

    # ------------------------------------------------------------------ 1
    print("\n  1. Loading all Phase C results ...")
    tir_data = _load_json(os.path.join(rd, 'tironian_csp.json'))
    phrase_data = _load_json(os.path.join(rd, 'phrase_detect.json'))
    modclean_data = _load_json(os.path.join(rd, 'modifier_clean.json'))
    reseg_data = _load_json(os.path.join(rd, 'reseg_csp.json'))
    mod_int_data = _load_json(os.path.join(rd, 'modifier_integrate.json'))
    feature_decode_data = _load_json(os.path.join(rd, 'feature_decode.json'))
    combined_refine_data = _load_json(os.path.join(rd, 'combined_refine.json'))

    # Determine best assignment from Phase C
    best_assignment: Dict[str, str] = {}
    best_dict_hit = 0.0
    best_selectivity = 0.0
    if tir_data and tir_data.get('best_assignment'):
        best_assignment = tir_data['best_assignment']
        best_dict_hit = tir_data.get('best_dict_hit', 0.0)
        best_selectivity = tir_data.get('best_selectivity', 0.0)
    elif combined_refine_data and combined_refine_data.get('best_assignment'):
        best_assignment = combined_refine_data['best_assignment']
        best_dict_hit = combined_refine_data.get('best_dict_hit', 0.0)
        best_selectivity = combined_refine_data.get('best_selectivity', 0.0)

    if not best_assignment:
        print("    [SKIP] No assignment found in any Phase C result file")
        return

    print(f"    dict_hit={best_dict_hit:.1%}, selectivity={best_selectivity:.2f}x")

    # Load corpus and reference data for active tests
    corpus = load_corpus(verbose=False)
    tokens = corpus.get_tokens(language='A', paragraph_only=True)
    eva_to_triple = build_eva_to_triple_lookup()

    ref_corpus = load_reference_corpus(verbose=False)
    ref_tokens = ref_corpus.get_combined_tokens('latin')
    ref_word_set = set(w.lower() for w in ref_tokens if len(w) >= 2)
    inventory = build_phoneme_inventory('latin', ref_corpus)

    # Decode full corpus once for reuse
    decoded_all = decode_corpus(tokens, best_assignment, eva_to_triple, max_tokens=len(tokens))
    decoded_set = set(w.lower() for w in decoded_all)

    # ------------------------------------------------------------------ 2
    print("\n  2. Running V1-V17 validation tests ...\n")

    test_results: List[Dict] = []
    n_passed = 0
    n_failed = 0
    n_skipped = 0

    def _record(test_id: str, test_name: str, passed: bool,
                skipped: bool = False, details: str = '') -> None:
        nonlocal n_passed, n_failed, n_skipped
        if skipped:
            n_skipped += 1
            status = 'SKIP'
        elif passed:
            n_passed += 1
            status = 'PASS'
        else:
            n_failed += 1
            status = 'FAIL'
        print(f"    {test_id:>4} ({test_name}): {status} -- {details}")
        test_results.append({
            'test_id': test_id,
            'test_name': test_name,
            'passed': passed,
            'skipped': skipped,
            'details': details,
        })

    # --- V1: Sanity check ---
    v1_pass = best_dict_hit > 0.0
    _record('V1', 'Sanity Check', v1_pass,
            details=f'dict_hit={best_dict_hit:.4f} (threshold: > 0)')

    # --- V2: Random baseline selectivity ---
    rng = random.Random(42)
    all_syls = list(inventory.cv_syllables)
    random_hits: List[float] = []
    for _ in range(50):
        rand_map = {k: rng.choice(all_syls) for k in best_assignment}
        rand_decoded = decode_corpus(tokens, rand_map, eva_to_triple, max_tokens=500)
        rh = sum(1 for w in rand_decoded if w in ref_word_set)
        random_hits.append(rh / len(rand_decoded) if rand_decoded else 0.0)
    rand_baseline = sum(random_hits) / len(random_hits) if random_hits else 0.001
    v2_selectivity = best_dict_hit / max(rand_baseline, 0.001)
    v2_pass = v2_selectivity > 1.5
    _record('V2', 'Random Baseline', v2_pass,
            details=f'selectivity={v2_selectivity:.2f}x (threshold: > 1.5)')

    # --- V3: Cross-validation CV < 0.10 ---
    # Split corpus into 5 folds, check dict_hit consistency
    fold_size = len(tokens) // 5
    fold_hits: List[float] = []
    for fold in range(5):
        start = fold * fold_size
        end = start + fold_size
        fold_tokens = tokens[start:end]
        fold_decoded = decode_corpus(fold_tokens, best_assignment, eva_to_triple, max_tokens=len(fold_tokens))
        fh = sum(1 for w in fold_decoded if w in ref_word_set)
        fold_hits.append(fh / len(fold_decoded) if fold_decoded else 0.0)
    mean_fold = sum(fold_hits) / len(fold_hits) if fold_hits else 0.0
    std_fold = (sum((x - mean_fold) ** 2 for x in fold_hits) / len(fold_hits)) ** 0.5 if fold_hits else 0.0
    cv = std_fold / mean_fold if mean_fold > 0 else 999.0
    v3_pass = cv < 0.10
    _record('V3', 'Cross-Validation', v3_pass,
            details=f'CV={cv:.4f} (threshold: < 0.10), fold_hits={[round(h, 4) for h in fold_hits]}')

    # --- V4: Section coherence ---
    sections = ['herbal_a', 'pharmaceutical', 'astronomical', 'biological']
    section_vocab: Dict[str, set] = {}
    for section in sections:
        sect_tokens = corpus.get_tokens(language='A', section=section, paragraph_only=True)
        if sect_tokens:
            sect_decoded = decode_corpus(sect_tokens, best_assignment, eva_to_triple, max_tokens=500)
            section_vocab[section] = set(w.lower() for w in sect_decoded)
    # Check: herbal should differ from astronomical
    n_sections_with_data = len(section_vocab)
    if n_sections_with_data >= 2:
        pairs = list(section_vocab.keys())
        jaccard_scores = []
        for i in range(len(pairs)):
            for j in range(i + 1, len(pairs)):
                s1 = section_vocab[pairs[i]]
                s2 = section_vocab[pairs[j]]
                inter = len(s1 & s2)
                union = len(s1 | s2)
                jaccard_scores.append(inter / union if union > 0 else 0.0)
        avg_jaccard = sum(jaccard_scores) / len(jaccard_scores) if jaccard_scores else 1.0
        # Sections should differ (avg Jaccard < 0.8)
        v4_pass = avg_jaccard < 0.80
        _record('V4', 'Section Coherence', v4_pass,
                details=f'avg_jaccard={avg_jaccard:.3f} across {n_sections_with_data} sections (threshold: < 0.80)')
    else:
        _record('V4', 'Section Coherence', False, skipped=True,
                details=f'Only {n_sections_with_data} section(s) with data')

    # --- V5: Illustration match ---
    # Botanical folios should decode to plant-related words
    plant_words = set(_DOMAIN_KEYWORDS['plant_names'] + _DOMAIN_KEYWORDS['plant_parts'])
    herbal_tokens = corpus.get_tokens(language='A', section='herbal_a', paragraph_only=True)
    if herbal_tokens:
        herbal_decoded = decode_corpus(herbal_tokens, best_assignment, eva_to_triple, max_tokens=500)
        plant_hits = sum(1 for w in herbal_decoded if w.lower() in plant_words)
        plant_rate = plant_hits / len(herbal_decoded) if herbal_decoded else 0.0
        # Also check non-herbal for comparison
        nonherb_tokens = corpus.get_tokens(language='A', section='astronomical', paragraph_only=True)
        if nonherb_tokens:
            nonherb_decoded = decode_corpus(nonherb_tokens, best_assignment, eva_to_triple, max_tokens=500)
            nonherb_plant = sum(1 for w in nonherb_decoded if w.lower() in plant_words)
            nonherb_rate = nonherb_plant / len(nonherb_decoded) if nonherb_decoded else 0.0
            v5_pass = plant_rate > nonherb_rate
            _record('V5', 'Illustration Match', v5_pass,
                    details=f'herbal plant_rate={plant_rate:.4f} vs astro={nonherb_rate:.4f}')
        else:
            v5_pass = plant_rate > 0.0
            _record('V5', 'Illustration Match', v5_pass,
                    details=f'herbal plant_rate={plant_rate:.4f} (no astro for comparison)')
    else:
        _record('V5', 'Illustration Match', False, skipped=True,
                details='No herbal_a tokens found')

    # --- V6: Language B consistency ---
    lang_b_tokens = corpus.get_tokens(language='B', paragraph_only=True)
    if lang_b_tokens:
        lang_b_decoded = decode_corpus(lang_b_tokens, best_assignment, eva_to_triple, max_tokens=500)
        lang_b_hits = sum(1 for w in lang_b_decoded if w in ref_word_set)
        lang_b_rate = lang_b_hits / len(lang_b_decoded) if lang_b_decoded else 0.0
        # Language B should have lower dict_hit than Language A
        v6_pass = lang_b_rate < best_dict_hit
        _record('V6', 'Language B Consistency', v6_pass,
                details=f'lang_B dict_hit={lang_b_rate:.4f} < lang_A={best_dict_hit:.4f}')
    else:
        _record('V6', 'Language B Consistency', False, skipped=True,
                details='No Language B tokens found')

    # --- V7: Prior-phase convergence ---
    # Current results should not contradict validated Phase 1-17 findings
    # Check: dict_hit >= Phase 16 baseline
    phase16_hit = 0.0
    if mod_int_data:
        phase16_hit = mod_int_data.get('best_dict_hit', 0.0)
    elif tir_data:
        phase16_hit = tir_data.get('phase16_dict_hit', 0.0)
    # Phase C should not regress below 80% of Phase 16
    v7_pass = best_dict_hit >= phase16_hit * 0.80
    _record('V7', 'Prior-Phase Convergence', v7_pass,
            details=f'Phase C dict_hit={best_dict_hit:.4f} vs 80% of Phase 16={phase16_hit * 0.80:.4f}')

    # --- V8: Readability assessment ---
    # Check average decoded word length (should be 3-10 chars like real Latin)
    word_lens = [len(w) for w in decoded_all if w and w != '?']
    avg_len = sum(word_lens) / len(word_lens) if word_lens else 0.0
    v8_pass = 3.0 <= avg_len <= 10.0
    _record('V8', 'Readability', v8_pass,
            details=f'avg_word_length={avg_len:.2f} chars (target: 3-10)')

    # --- V9: Comparison to all prior phases ---
    phase11_hit = 0.111
    phase14_hit = 0.194
    phase15_hit = 0.354
    if feature_decode_data:
        phase14_hit = feature_decode_data.get('best_dict_hit', 0.194)
    if combined_refine_data:
        phase15_hit = combined_refine_data.get('best_dict_hit', 0.354)
    # Phase C should be at least as good as Phase 14
    v9_pass = best_dict_hit >= phase14_hit
    _record('V9', 'Phase Comparison', v9_pass,
            details=f'Phase C={best_dict_hit:.1%} vs P11={phase11_hit:.1%}, P14={phase14_hit:.1%}, P15={phase15_hit:.1%}, P16={phase16_hit:.1%}')

    # --- V10: Vocabulary catalog by semantic domain ---
    domain_hits: Dict[str, List[str]] = {}
    for domain_name, keywords in _DOMAIN_KEYWORDS.items():
        hits = [k for k in keywords if k in decoded_set]
        if hits:
            domain_hits[domain_name] = hits
    n_domains = len(domain_hits)
    v10_pass = n_domains >= 2
    _record('V10', 'Vocabulary Catalog', v10_pass,
            details=f'{n_domains}/6 domains with hits: {list(domain_hits.keys())}')

    # --- V11: Progression tracking ---
    v11_pass = best_dict_hit >= phase15_hit * 0.90  # Allow 10% regression tolerance
    _record('V11', 'Progression', v11_pass,
            details=f'P11={phase11_hit:.1%}->P14={phase14_hit:.1%}->P15={phase15_hit:.1%}->P16={phase16_hit:.1%}->PC={best_dict_hit:.1%}')

    # --- V12: Feature plausibility ---
    # Load from feature_decode.json if available
    v12_data = _load_json(os.path.join(rd, 'feature_decode.json'))
    if v12_data and 'v12_feature_plausibility' in v12_data:
        v12_info = v12_data['v12_feature_plausibility']
        v12_score = v12_info.get('plausibility_score', 0.0)
        v12_pass = v12_info.get('passed', False)
        _record('V12', 'Feature Plausibility', v12_pass,
                details=f'plausibility={v12_score:.1%}')
    else:
        _record('V12', 'Feature Plausibility', False, skipped=True,
                details='feature_decode.json v12 data not available')

    # --- V13: Phrase selectivity > 2x ---
    v13_pass = False
    if phrase_data:
        phrase_sel = phrase_data.get('phrase_selectivity', 0.0)
        n_phrases = phrase_data.get('n_phrases_detected', 0)
        v13_pass = phrase_data.get('gate_passed', False)
        _record('V13', 'Phrase Selectivity', v13_pass,
                details=f'{n_phrases} phrases, selectivity={phrase_sel:.2f}x (threshold: >= 3 phrases AND > 2.0x)')
    else:
        _record('V13', 'Phrase Selectivity', False, skipped=True,
                details='phrase_detect.json not found')

    # --- V14: Domain coverage >= 3/6 ---
    v14_pass = n_domains >= 3
    _record('V14', 'Domain Coverage', v14_pass,
            details=f'{n_domains}/6 domains (threshold: >= 3)')

    # --- V15: Null corpus control ---
    # Generate a random "corpus" of same size and decode it -- should get fewer hits
    rng2 = random.Random(123)
    all_chars = sorted(eva_to_triple.keys())
    if all_chars:
        null_tokens: List[str] = []
        for tok in tokens[:2000]:
            n_chars = len(tokenize_eva_chars(tok))
            null_tok = ''.join(rng2.choice(all_chars) for _ in range(n_chars))
            null_tokens.append(null_tok)
        null_decoded = decode_corpus(null_tokens, best_assignment, eva_to_triple, max_tokens=len(null_tokens))
        null_hits = sum(1 for w in null_decoded if w in ref_word_set)
        null_rate = null_hits / len(null_decoded) if null_decoded else 0.0
        # Real corpus should be significantly better than null corpus
        v15_pass = best_dict_hit > null_rate * 1.5
        _record('V15', 'Null Corpus Control', v15_pass,
                details=f'real={best_dict_hit:.4f} vs null={null_rate:.4f} (threshold: real > 1.5 * null)')
    else:
        v15_pass = False
        _record('V15', 'Null Corpus Control', False, skipped=True,
                details='No EVA chars available for null corpus')

    # --- V16: Keyword presence (top-100 Latin words) ---
    top100_hits = _TOP_100_LATIN & decoded_set
    n_top100 = len(top100_hits)
    v16_pass = n_top100 >= 5
    _record('V16', 'Keyword Presence', v16_pass,
            details=f'{n_top100}/100 top Latin words found (threshold: >= 5). Hits: {sorted(top100_hits)[:20]}')

    # --- V17: Verb decode (>= 5/15 at edit distance <= 1) ---
    verb_matches = 0
    verb_details: List[str] = []
    for verb in _TARGET_VERBS:
        # Check if any decoded token is within edit distance 1 of this verb
        matched = False
        for decoded_w in decoded_set:
            if _edit_distance(verb, decoded_w) <= 1:
                verb_details.append(f'{verb}~{decoded_w}')
                matched = True
                break
        if matched:
            verb_matches += 1
    v17_pass = verb_matches >= 5
    _record('V17', 'Verb Decode', v17_pass,
            details=f'{verb_matches}/15 verbs at edit_dist<=1 (threshold: >= 5). Matches: {verb_details[:10]}')

    # ------------------------------------------------------------------ 3
    # Handle SKIPs: exclude from denominator
    n_effective = n_passed + n_failed  # excludes skipped
    pass_rate = n_passed / n_effective if n_effective > 0 else 0.0

    print(f"\n  Summary:")
    print(f"    Passed:  {n_passed}/{n_effective} (excluding {n_skipped} skipped)")
    print(f"    Failed:  {n_failed}/{n_effective}")
    print(f"    Skipped: {n_skipped}")
    print(f"    Rate:    {pass_rate:.1%}")

    # ------------------------------------------------------------------ 4
    # Gate: >= 12/17 pass (excl skips), V13 and V15 must pass
    gate_passed = (
        n_passed >= 12
        and v13_pass
        and v15_pass
    )

    # Allow relaxed gate if skips reduce effective total
    if not gate_passed and n_effective < 17:
        # Require >= 70% of effective tests to pass
        if pass_rate >= 0.70 and v13_pass and v15_pass:
            gate_passed = True

    if gate_passed:
        verdict = (
            f"PASS: {n_passed}/{n_effective} tests passed ({pass_rate:.1%}). "
            f"V13 (phrase selectivity): {'PASS' if v13_pass else 'FAIL'}. "
            f"V15 (null control): {'PASS' if v15_pass else 'FAIL'}. "
            f"Phase C decode is validated."
        )
    else:
        failing_tests = [
            tr['test_id'] for tr in test_results
            if not tr['passed'] and not tr.get('skipped', False)
        ]
        verdict = (
            f"FAIL: {n_passed}/{n_effective} tests passed ({pass_rate:.1%}), "
            f"need >= 12. "
            f"V13: {'PASS' if v13_pass else 'FAIL (required)'}. "
            f"V15: {'PASS' if v15_pass else 'FAIL (required)'}. "
            f"Failing: {failing_tests}."
        )

    print(f"\n  Gate: {'PASS' if gate_passed else 'FAIL'}")
    print(f"  Verdict: {verdict}")

    # Save
    result = PhaseCValidateResult(
        test_results=test_results,
        n_passed=n_passed,
        n_failed=n_failed,
        n_skipped=n_skipped,
        n_total=17,
        pass_rate=round(pass_rate, 4),
        v13_passed=v13_pass,
        v15_passed=v15_pass,
        gate_passed=gate_passed,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'phaseC_validate.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)

    print(f"\n  Results saved -> {out_path}")
