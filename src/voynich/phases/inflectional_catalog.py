"""
Phase 71, Track 1: Complete Inflectional Catalog
=================================================
Map every coda marker to a grammatical function, classify every token
(36,238) by grammatical role, compute corpus-wide statistics, per-section
and per-hand profiles, and run null validation (500 trials).

Phase 70 Track 2 discovered:
  -s  -> 2nd person singular  (100%, 20 obs)
  -t  -> 3rd person singular  (82%, 11 obs)
  -n  -> accusative case      (Phase 59: -en 27.5%, -in 24.3%, -an 8.2%)
  -r  -> passive voice        (1 obs: coratur)

Dependency chain:
    results/combined_refine.json         (Phase 15: best_assignment)
    results/p69_clean_corpus.json        (T1 catalogue, clean_indices)
    results/modifier_integrate.json      (Phase 16: modifier chars)
        -> results/phase71_inflectional_catalog.json
"""

import json
import os
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import build_eva_to_triple_lookup, load_corpus, tokenize_eva_chars
from voynich.phases.coda_markers import get_coda
from voynich.phases.corrected_coda import (
    build_coda_table_v2,
    classify_token_chars_v2,
    decode_token_cvc_v2,
)
from voynich.phases.suffix_grammar import _classify_latin_ending


# ---------------------------------------------------------------------------
# JSON helpers
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
    if isinstance(obj, set):
        return sorted(obj)
    if isinstance(obj, (bool, int, float, str, type(None))):
        return obj
    return str(obj)


def _safe_load(path: str) -> Dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _save_json(rd: str, filename: str, data: Any) -> str:
    path = os.path.join(rd, filename)
    with open(path, 'w') as f:
        json.dump(_convert(data), f, indent=2)
    return path


# ---------------------------------------------------------------------------
# Coda-to-grammar mapping (from Phase 70 Track 2)
# ---------------------------------------------------------------------------

CODA_GRAMMAR: Dict[str, Dict[str, Any]] = {
    's': {
        'primary': 'VERB_2SG',
        'description': '2nd person singular present/imperative',
        'confidence': 1.00,
        'n_obs': 20,
        'category': 'VERBAL',
        'note': 'Pharmaceutical imperative: "colas" = you strain',
    },
    't': {
        'primary': 'VERB_3SG',
        'description': '3rd person singular present',
        'confidence': 0.82,
        'n_obs': 11,
        'category': 'VERBAL',
        'note': 'Property description: "valet" = it is worth',
    },
    'n': {
        'primary': 'NOUN_ACC',
        'description': 'accusative case (-am/-em/-um -> -an/-en/-on)',
        'confidence': 0.62,
        'n_obs': 3837,
        'category': 'NOMINAL',
        'note': 'Direct objects and prepositional complements',
    },
    'r': {
        'primary': 'VERB_PASSIVE',
        'description': 'passive voice (-tur, -atur, -etur)',
        'confidence': 0.50,
        'n_obs': 1,
        'category': 'VERBAL',
        'note': 'Passive: "colatur" = let it be strained',
    },
}

DOUBLE_CODA_GRAMMAR: Dict[str, Dict[str, str]] = {
    'nt': {
        'function': 'VERB_3PL',
        'description': '3rd person plural',
        'category': 'VERBAL',
    },
    'ns': {
        'function': 'PARTICIPLE',
        'description': 'present participle accusative',
        'category': 'VERBAL',
    },
    'rs': {
        'function': 'NOUN_NOM_PL',
        'description': '3rd declension nominative plural',
        'category': 'NOMINAL',
    },
    'st': {
        'function': 'VERB_EST',
        'description': '"est" (is) in property descriptions',
        'category': 'VERBAL',
    },
}

# Broad category grouping
_VERBAL_FUNCS = {'VERB_2SG', 'VERB_3SG', 'VERB_PASSIVE', 'VERB_3PL',
                 'PARTICIPLE', 'VERB_EST'}
_NOMINAL_FUNCS = {'NOUN_ACC', 'NOUN_NOM_PL'}

# CI expected distribution (approximate from pharmaceutical Latin)
_CI_EXPECTED = {
    'VERBAL': 0.15,
    'NOMINAL': 0.35,
    'FUNCTION_STEM': 0.30,
    'UNMARKED': 0.20,
}


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def _build_folio_list(corpus) -> List[str]:
    """Build flat list of folio IDs, one per token."""
    folios: List[str] = []
    for folio, page in corpus.pages.items():
        for _ in page.all_tokens:
            folios.append(folio)
    return folios


def _build_section_list(corpus) -> List[str]:
    """Build flat list of section labels, one per token."""
    sections: List[str] = []
    for _folio, page in corpus.pages.items():
        for _ in page.all_tokens:
            sections.append(getattr(page, 'section', 'unknown'))
    return sections


def _build_hand_list(corpus) -> List[str]:
    """Build flat list of hand labels, one per token."""
    hands: List[str] = []
    for _folio, page in corpus.pages.items():
        hand = getattr(page, 'hand', 'unknown')
        for _ in page.all_tokens:
            hands.append(str(hand))
    return hands


def _determine_gram_function(
    codas: List[Dict[str, str]],
    decoded: str,
) -> Dict[str, Any]:
    """Determine grammatical function from coda marker(s)."""
    if not codas:
        if decoded and len(decoded) <= 3:
            return {
                'category': 'FUNCTION_STEM',
                'function': 'FUNCTION_OR_SHORT_STEM',
                'detail': 'No coda — likely function word or bare stem',
                'confidence': 0.30,
            }
        return {
            'category': 'UNMARKED',
            'function': 'UNMARKED',
            'detail': 'Multi-syllable token without coda marker',
            'confidence': 0.20,
        }

    if len(codas) == 1:
        coda = codas[0]['coda_consonant']
        grammar = CODA_GRAMMAR.get(coda, {})
        func = grammar.get('primary', f'CODA_{coda.upper()}')
        cat = grammar.get('category', 'UNKNOWN')
        return {
            'category': cat,
            'function': func,
            'detail': grammar.get('description', f'Coda {coda}'),
            'confidence': grammar.get('confidence', 0.50),
        }

    if len(codas) == 2:
        cluster = codas[0]['coda_consonant'] + codas[1]['coda_consonant']
        double = DOUBLE_CODA_GRAMMAR.get(cluster, {})
        if double:
            return {
                'category': double['category'],
                'function': double['function'],
                'detail': double['description'],
                'confidence': 0.70,
            }
        return {
            'category': 'DOUBLE_CODA',
            'function': f'DOUBLE_{cluster.upper()}',
            'detail': f'Unrecognized double coda: {cluster}',
            'confidence': 0.30,
        }

    return {
        'category': 'MULTI_CODA',
        'function': 'MULTI_CODA',
        'detail': f'{len(codas)} coda markers — likely segmentation issue',
        'confidence': 0.10,
    }


def _classify_all_tokens(
    all_tokens: List[str],
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    coda_table: Any,
    folio_list: List[str],
    section_list: List[str],
    hand_list: List[str],
    clean_indices: Set[int],
) -> List[Dict[str, Any]]:
    """Classify every token in the corpus by grammatical role."""
    catalog = []

    for idx, token in enumerate(all_tokens):
        eva_chars = tokenize_eva_chars(token)
        classified = classify_token_chars_v2(eva_chars, coda_table)

        # Extract coda markers
        codas = []
        for role, char in classified:
            if role == 'CODA_MARKER':
                coda_val = get_coda(char, coda_table)
                if coda_val:
                    codas.append({
                        'eva_char': char,
                        'coda_consonant': coda_val,
                    })

        # Decode
        try:
            result = decode_token_cvc_v2(token, assignment, eva_to_triple, coda_table)
            decoded = result.decoded_cvc
        except Exception:
            decoded = ''

        # Grammatical classification
        gram = _determine_gram_function(codas, decoded)

        # Cross-validation with Latin ending analysis
        pos_ending, case_ending = _classify_latin_ending(decoded) if decoded else ('', '')

        catalog.append({
            'token_idx': idx,
            'folio': folio_list[idx] if idx < len(folio_list) else '?',
            'section': section_list[idx] if idx < len(section_list) else '?',
            'hand': hand_list[idx] if idx < len(hand_list) else '?',
            'is_clean': idx in clean_indices,
            'n_codas': len(codas),
            'coda_consonants': [c['coda_consonant'] for c in codas],
            'gram_function': gram['function'],
            'gram_category': gram['category'],
            'gram_detail': gram['detail'],
            'gram_confidence': gram['confidence'],
            'decoded': decoded,
            'latin_pos': pos_ending,
            'latin_case': case_ending,
        })

    return catalog


def _compute_broad_distribution(catalog: List[Dict]) -> Dict[str, float]:
    """Aggregate grammatical categories into broad groups."""
    total = len(catalog)
    if total == 0:
        return {}

    counts: Dict[str, int] = {
        'VERBAL': 0, 'NOMINAL': 0, 'FUNCTION_STEM': 0, 'UNMARKED': 0,
    }

    for entry in catalog:
        func = entry['gram_function']
        cat = entry['gram_category']
        if func in _VERBAL_FUNCS or cat == 'VERBAL':
            counts['VERBAL'] += 1
        elif func in _NOMINAL_FUNCS or cat == 'NOMINAL':
            counts['NOMINAL'] += 1
        elif cat == 'FUNCTION_STEM':
            counts['FUNCTION_STEM'] += 1
        else:
            counts['UNMARKED'] += 1

    return {k: v / total for k, v in counts.items()}


def _chi2_distance(dist_a: Dict[str, float], dist_b: Dict[str, float]) -> float:
    """Chi-squared distance between two distributions."""
    keys = set(dist_a.keys()) | set(dist_b.keys())
    total = 0.0
    for k in keys:
        a = dist_a.get(k, 0.0)
        b = dist_b.get(k, 0.0)
        denom = a + b
        if denom > 0:
            total += (a - b) ** 2 / denom
    return total


def _run_null_validation(
    all_tokens: List[str],
    coda_table: Any,
    real_distribution: Dict[str, float],
    n_trials: int = 500,
) -> Dict[str, Any]:
    """Null test: shuffle coda-to-grammar mapping and compare distributions.

    For each trial, randomly reassign which coda consonant each stroke
    group maps to, then reclassify all tokens and compute chi-squared
    distance to CI expected distribution.
    """
    real_distance = _chi2_distance(real_distribution, _CI_EXPECTED)

    # Pre-compute: for each token, what coda consonants does it have?
    coda_letters = ['n', 'r', 's', 't']
    token_coda_lists: List[List[str]] = []
    for token in all_tokens:
        eva_chars = tokenize_eva_chars(token)
        classified = classify_token_chars_v2(eva_chars, coda_table)
        codas = []
        for role, char in classified:
            if role == 'CODA_MARKER':
                coda_val = get_coda(char, coda_table)
                if coda_val:
                    codas.append(coda_val)
        token_coda_lists.append(codas)

    null_distances = []
    rng = np.random.default_rng(seed=42)

    for _ in range(n_trials):
        # Shuffle: create random permutation of grammar functions
        shuffled = list(coda_letters)
        rng.shuffle(shuffled)
        remap = dict(zip(coda_letters, shuffled))

        # Reclassify all tokens under shuffled mapping
        counts: Dict[str, int] = {
            'VERBAL': 0, 'NOMINAL': 0, 'FUNCTION_STEM': 0, 'UNMARKED': 0,
        }
        for codas, token in zip(token_coda_lists, all_tokens):
            if not codas:
                decoded_len = len(token)
                if decoded_len <= 3:
                    counts['FUNCTION_STEM'] += 1
                else:
                    counts['UNMARKED'] += 1
                continue

            # Remap the last coda consonant
            last_coda = remap.get(codas[-1], codas[-1])
            grammar = CODA_GRAMMAR.get(last_coda, {})
            cat = grammar.get('category', 'UNMARKED')
            if cat == 'VERBAL':
                counts['VERBAL'] += 1
            elif cat == 'NOMINAL':
                counts['NOMINAL'] += 1
            else:
                counts['UNMARKED'] += 1

        total = sum(counts.values())
        dist = {k: v / total for k, v in counts.items()} if total > 0 else {}
        null_distances.append(_chi2_distance(dist, _CI_EXPECTED))

    null_distances = np.array(null_distances)
    null_mean = float(np.mean(null_distances))
    null_std = float(np.std(null_distances))
    z = (null_mean - real_distance) / null_std if null_std > 0 else 0.0
    p = float(np.mean(null_distances <= real_distance))

    return {
        'real_ci_distance': real_distance,
        'null_mean_distance': null_mean,
        'null_std': null_std,
        'z_score': z,
        'p_value': p,
        'n_trials': n_trials,
        'significant': p < 0.05,
    }


def _section_and_hand_profiles(
    catalog: List[Dict],
) -> Tuple[Dict[str, Dict[str, int]], Dict[str, Dict[str, int]]]:
    """Per-section and per-hand grammatical distributions."""
    section_counts: Dict[str, Counter] = defaultdict(Counter)
    hand_counts: Dict[str, Counter] = defaultdict(Counter)

    for entry in catalog:
        section_counts[entry['section']][entry['gram_category']] += 1
        hand_counts[entry['hand']][entry['gram_category']] += 1

    return (
        {s: dict(c) for s, c in section_counts.items()},
        {h: dict(c) for h, c in hand_counts.items()},
    )


def _chi2_contingency_p(profiles: Dict[str, Dict[str, int]]) -> float:
    """Compute chi-squared contingency test p-value for profiles."""
    try:
        from scipy.stats import chi2_contingency
    except ImportError:
        return 1.0

    categories = set()
    for counts in profiles.values():
        categories.update(counts.keys())
    categories = sorted(categories)

    groups = sorted(profiles.keys())
    if len(groups) < 2 or len(categories) < 2:
        return 1.0

    table = []
    for g in groups:
        row = [profiles[g].get(c, 0) for c in categories]
        if sum(row) > 0:
            table.append(row)

    if len(table) < 2:
        return 1.0

    try:
        _, p, _, _ = chi2_contingency(table)
        return float(p)
    except Exception:
        return 1.0


def _cross_validation_agreement(catalog: List[Dict]) -> float:
    """Fraction of clean tokens where coda-based and ending-based POS agree."""
    n_comparable = 0
    n_agree = 0

    for entry in catalog:
        if not entry['is_clean']:
            continue
        if entry['gram_category'] in ('FUNCTION_STEM', 'UNMARKED'):
            continue
        if not entry['latin_pos']:
            continue

        n_comparable += 1
        coda_is_verbal = entry['gram_category'] == 'VERBAL'
        ending_is_verbal = entry['latin_pos'] == 'VERB'

        coda_is_nominal = entry['gram_category'] == 'NOMINAL'
        ending_is_nominal = entry['latin_pos'] == 'NOUN'

        if (coda_is_verbal and ending_is_verbal) or \
           (coda_is_nominal and ending_is_nominal):
            n_agree += 1

    return n_agree / n_comparable if n_comparable > 0 else 0.0


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class InflectionalCatalogResult:
    phase: str = "71"
    step: str = "71.1"
    experiment: str = "inflectional_catalog"
    # Token counts
    n_tokens: int = 0
    n_with_coda: int = 0
    n_single_coda: int = 0
    n_double_coda: int = 0
    n_unmarked: int = 0
    n_function_stem: int = 0
    # Grammatical distribution
    grammatical_counts: Dict[str, int] = field(default_factory=dict)
    grammatical_fractions: Dict[str, float] = field(default_factory=dict)
    broad_distribution: Dict[str, float] = field(default_factory=dict)
    # Per-coda breakdown
    coda_function_counts: Dict[str, Dict[str, int]] = field(default_factory=dict)
    # Section and hand profiles
    section_profiles: Dict[str, Dict[str, int]] = field(default_factory=dict)
    hand_profiles: Dict[str, Dict[str, int]] = field(default_factory=dict)
    section_chi2_p: float = 1.0
    hand_chi2_p: float = 1.0
    # Null validation
    null_test: Dict[str, Any] = field(default_factory=dict)
    # Cross-validation
    cross_validation_agreement: float = 0.0
    # Gates
    gate_i1: bool = False  # >= 5000 tokens classified as VERBAL
    gate_i2: bool = False  # >= 10000 tokens classified as NOMINAL
    gate_i3: bool = False  # Null test p < 0.05
    gate_i4: bool = False  # Section profiles differ (chi2 p < 0.05)
    gate_i5: bool = False  # Verbal fraction between 10-25%
    gates_passed: int = 0
    gate_passed: bool = False
    verdict: str = "UNKNOWN"
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_inflect_catalog():
    """Track 1: Complete inflectional catalog of the corpus."""
    t0 = time.time()
    rd = str(_results_dir())

    print("Phase 71.1 — Complete Inflectional Catalog")
    print("=" * 43)

    # --- Load dependencies ---
    refine_data = _safe_load(os.path.join(rd, 'combined_refine.json'))
    assignment = refine_data.get('best_assignment', {})

    clean_data = _safe_load(os.path.join(rd, 'p69_clean_corpus.json'))
    clean_indices = set(clean_data.get('clean_indices', []))
    print(f"  Clean tokens: {len(clean_indices)}")

    eva_to_triple = build_eva_to_triple_lookup()
    coda_table = build_coda_table_v2()

    corpus = load_corpus(verbose=False)
    all_tokens = corpus.get_tokens()
    folio_list = _build_folio_list(corpus)
    section_list = _build_section_list(corpus)
    hand_list = _build_hand_list(corpus)
    print(f"  Total tokens: {len(all_tokens)}")

    # --- Classify all tokens ---
    print("\n  Classifying all tokens by grammatical role...")
    catalog = _classify_all_tokens(
        all_tokens, assignment, eva_to_triple, coda_table,
        folio_list, section_list, hand_list, clean_indices)

    # --- Compute statistics ---
    func_counts = Counter(e['gram_function'] for e in catalog)
    cat_counts = Counter(e['gram_category'] for e in catalog)
    total = len(catalog)

    n_with_coda = sum(1 for e in catalog if e['n_codas'] > 0)
    n_single = sum(1 for e in catalog if e['n_codas'] == 1)
    n_double = sum(1 for e in catalog if e['n_codas'] >= 2)
    n_unmarked = cat_counts.get('UNMARKED', 0)
    n_func_stem = cat_counts.get('FUNCTION_STEM', 0)

    print(f"  Tokens with coda: {n_with_coda} ({n_with_coda/total:.1%})")
    print(f"    Single coda: {n_single}")
    print(f"    Double+ coda: {n_double}")
    print(f"  Unmarked: {n_unmarked} ({n_unmarked/total:.1%})")
    print(f"  Function/stem: {n_func_stem} ({n_func_stem/total:.1%})")

    print("\n  Grammatical function distribution:")
    for func, count in func_counts.most_common():
        print(f"    {func}: {count} ({count/total:.1%})")

    # Broad distribution
    broad = _compute_broad_distribution(catalog)
    print("\n  Broad categories:")
    for cat, frac in sorted(broad.items()):
        ci_exp = _CI_EXPECTED.get(cat, 0.0)
        print(f"    {cat}: {frac:.1%} (CI expected: {ci_exp:.0%})")

    # Per-coda breakdown
    coda_func_counts: Dict[str, Dict[str, int]] = defaultdict(Counter)
    for entry in catalog:
        for coda in entry['coda_consonants']:
            coda_func_counts[coda][entry['gram_function']] += 1
    coda_func_dict = {c: dict(cnt) for c, cnt in coda_func_counts.items()}

    print("\n  Per-coda function counts:")
    for coda in sorted(coda_func_dict.keys()):
        counts = coda_func_dict[coda]
        top = max(counts, key=counts.get) if counts else '?'
        total_c = sum(counts.values())
        print(f"    -{coda}: {total_c} tokens, primary={top} "
              f"({counts.get(top, 0)/total_c:.0%})")

    # --- Section and hand profiles ---
    print("\n  Computing section and hand profiles...")
    section_profiles, hand_profiles = _section_and_hand_profiles(catalog)
    section_p = _chi2_contingency_p(section_profiles)
    hand_p = _chi2_contingency_p(hand_profiles)
    print(f"  Section chi² p: {section_p:.4f}")
    print(f"  Hand chi² p: {hand_p:.4f}")

    for section in sorted(section_profiles.keys()):
        prof = section_profiles[section]
        total_s = sum(prof.values())
        verbal = prof.get('VERBAL', 0)
        nominal = prof.get('NOMINAL', 0)
        print(f"    {section}: V={verbal/total_s:.1%} N={nominal/total_s:.1%} "
              f"(n={total_s})")

    # --- Cross-validation ---
    cv_agreement = _cross_validation_agreement(catalog)
    print(f"\n  Cross-validation agreement (coda vs ending): {cv_agreement:.1%}")

    # --- Null validation ---
    print(f"\n  Running null validation (500 trials)...")
    null_results = _run_null_validation(
        all_tokens, coda_table, broad, n_trials=500)
    print(f"    Real CI distance: {null_results['real_ci_distance']:.4f}")
    print(f"    Null mean distance: {null_results['null_mean_distance']:.4f}")
    print(f"    Z-score: {null_results['z_score']:.2f}")
    print(f"    P-value: {null_results['p_value']:.4f}")
    print(f"    Significant: {null_results['significant']}")

    # --- Gates ---
    n_verbal = sum(1 for e in catalog if e['gram_category'] == 'VERBAL')
    n_nominal = sum(1 for e in catalog if e['gram_category'] == 'NOMINAL')
    verbal_frac = broad.get('VERBAL', 0.0)

    g1 = n_verbal >= 5000
    g2 = n_nominal >= 10000
    g3 = null_results['significant']
    g4 = section_p < 0.05
    g5 = 0.10 <= verbal_frac <= 0.25

    gates_passed = sum([g1, g2, g3, g4, g5])

    print(f"\n  Gates: {gates_passed}/5")
    print(f"    I1 (≥5000 VERBAL): {'PASS' if g1 else 'FAIL'} ({n_verbal})")
    print(f"    I2 (≥10000 NOMINAL): {'PASS' if g2 else 'FAIL'} ({n_nominal})")
    print(f"    I3 (null p < 0.05): {'PASS' if g3 else 'FAIL'} "
          f"(p={null_results['p_value']:.4f})")
    print(f"    I4 (section chi² p < 0.05): {'PASS' if g4 else 'FAIL'} "
          f"(p={section_p:.4f})")
    print(f"    I5 (verbal 10-25%): {'PASS' if g5 else 'FAIL'} "
          f"({verbal_frac:.1%})")

    if gates_passed >= 4:
        verdict = 'INFLECTIONAL_CONFIRMED'
    elif gates_passed >= 2:
        verdict = 'PARTIAL_INFLECTIONAL'
    else:
        verdict = 'NOT_CONFIRMED'

    print(f"\n  Verdict: {verdict}")

    # --- Build result ---
    result = InflectionalCatalogResult(
        n_tokens=total,
        n_with_coda=n_with_coda,
        n_single_coda=n_single,
        n_double_coda=n_double,
        n_unmarked=n_unmarked,
        n_function_stem=n_func_stem,
        grammatical_counts=dict(func_counts.most_common()),
        grammatical_fractions={k: v / total for k, v in func_counts.most_common()},
        broad_distribution=broad,
        coda_function_counts=coda_func_dict,
        section_profiles=section_profiles,
        hand_profiles=hand_profiles,
        section_chi2_p=section_p,
        hand_chi2_p=hand_p,
        null_test=null_results,
        cross_validation_agreement=cv_agreement,
        gate_i1=g1,
        gate_i2=g2,
        gate_i3=g3,
        gate_i4=g4,
        gate_i5=g5,
        gates_passed=gates_passed,
        gate_passed=gates_passed >= 4,
        verdict=verdict,
        runtime_seconds=time.time() - t0,
    )

    out = _save_json(rd, 'phase71_inflectional_catalog.json', asdict(result))
    print(f"\n  Saved: {out}")
    print(f"  Runtime: {result.runtime_seconds:.1f}s")

    return result
