"""
Phase 72, Track 2: Cross-Validation Failure Diagnosis
======================================================
The Phase 71 CVC model's internal consistency is only 24% — coda-based
grammatical role (VERBAL/NOMINAL) agrees with Latin-ending-based POS only
24% of the time.  This track diagnoses WHERE and WHY by breaking down
cross-validation by coda type, by triple status, by section, by decoded
string length, and by number of coda markers.

Also introduces a more direct diagnostic: does the coda consonant actually
appear at the expected position in the decoded string?  If not, the
combination model itself (append) may be wrong.

Dependency chain:
    results/combined_refine.json         (Phase 15: best_assignment)
    results/triple_tiers.json            (Phase 28/53: tiered triples)
    results/p69_clean_corpus.json        (Phase 69: clean indices)
    results/modifier_integrate.json      (Phase 16: modifier chars)
        -> results/phase72_xval.json
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
# Triple tier loading (same pattern as p69_clean_corpus.py)
# ---------------------------------------------------------------------------

def _get_confirmed_and_unresolved(rd: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Return (confirmed, unresolved) assignment dicts."""
    refine_data = _safe_load(os.path.join(rd, 'combined_refine.json'))
    assignment = refine_data.get('best_assignment', {})

    tier_data = _safe_load(os.path.join(rd, 'triple_tiers.json'))
    confirmed_keys: Set[str] = set()

    if tier_data and 'tiers' in tier_data:
        tiers = tier_data['tiers']
        if isinstance(tiers, dict):
            for entry in tiers.get('CONFIRMED', []):
                confirmed_keys.add(entry.get('triple_key', ''))
        elif isinstance(tiers, list):
            for entry in tiers:
                if entry.get('tier', '') == 'CONFIRMED':
                    confirmed_keys.add(entry.get('triple_key', ''))

    confirmed = {k: v for k, v in assignment.items() if k in confirmed_keys}
    unresolved = {k: v for k, v in assignment.items() if k not in confirmed_keys}
    return confirmed, unresolved


# ---------------------------------------------------------------------------
# Coda-to-grammar mapping (from Phase 71 inflectional_catalog.py)
# ---------------------------------------------------------------------------

CODA_GRAMMAR: Dict[str, Dict[str, Any]] = {
    's': {'primary': 'VERB_2SG', 'category': 'VERBAL'},
    't': {'primary': 'VERB_3SG', 'category': 'VERBAL'},
    'n': {'primary': 'NOUN_ACC', 'category': 'NOMINAL'},
    'r': {'primary': 'VERB_PASSIVE', 'category': 'VERBAL'},
}


# ---------------------------------------------------------------------------
# Flat list builders
# ---------------------------------------------------------------------------

def _build_folio_list(corpus) -> List[str]:
    folios: List[str] = []
    for folio, page in corpus.pages.items():
        for _ in page.all_tokens:
            folios.append(folio)
    return folios


def _build_section_list(corpus) -> List[str]:
    sections: List[str] = []
    for _folio, page in corpus.pages.items():
        for _ in page.all_tokens:
            sections.append(getattr(page, 'section', 'unknown'))
    return sections


# ---------------------------------------------------------------------------
# Per-coda cross-validation
# ---------------------------------------------------------------------------

def _per_coda_xval(
    all_tokens: List[str],
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    coda_table: Any,
    clean_indices: Set[int],
) -> Dict[str, Dict[str, Any]]:
    """For each coda consonant (n, r, s, t), compute:
    - How often coda-based category agrees with ending-based POS
    - Mismatch distribution (what does the ending say when the coda disagrees?)
    - Whether the coda consonant is literally present in the decoded string
    """
    per_coda: Dict[str, Dict[str, Any]] = {}

    for coda_letter in ['n', 'r', 's', 't']:
        matches = 0
        mismatches = 0
        mismatch_details: Dict[str, int] = Counter()
        coda_present_count = 0
        coda_last_char_count = 0
        total_tokens_with_this_coda = 0

        for idx, token in enumerate(all_tokens):
            if idx not in clean_indices:
                continue

            eva_chars = tokenize_eva_chars(token)
            classified = classify_token_chars_v2(eva_chars, coda_table)

            # Check if this token has this coda type
            token_codas = []
            for role, char in classified:
                if role == 'CODA_MARKER':
                    coda_val = get_coda(char, coda_table)
                    if coda_val:
                        token_codas.append(coda_val)

            if coda_letter not in token_codas:
                continue

            total_tokens_with_this_coda += 1

            # Decode
            try:
                result = decode_token_cvc_v2(token, assignment, eva_to_triple, coda_table)
                decoded = result.decoded_cvc
            except Exception:
                continue

            if not decoded:
                continue

            # Check: is the coda consonant literally present in the decoded string?
            if coda_letter in decoded:
                coda_present_count += 1
            if decoded and decoded[-1] == coda_letter:
                coda_last_char_count += 1

            # Cross-validation: coda-based category vs ending-based POS
            coda_cat = CODA_GRAMMAR.get(coda_letter, {}).get('category', 'UNKNOWN')
            pos_ending, _case = _classify_latin_ending(decoded)

            if not pos_ending or pos_ending == 'UNCLEAR':
                continue
            if coda_cat in ('UNKNOWN',):
                continue

            coda_is_verbal = coda_cat == 'VERBAL'
            ending_is_verbal = pos_ending == 'VERB'
            coda_is_nominal = coda_cat == 'NOMINAL'
            ending_is_nominal = pos_ending == 'NOUN'

            if (coda_is_verbal and ending_is_verbal) or \
               (coda_is_nominal and ending_is_nominal):
                matches += 1
            else:
                mismatches += 1
                mismatch_details[f'coda_{coda_cat}_end_{pos_ending}'] += 1

        total_compared = matches + mismatches
        per_coda[coda_letter] = {
            'total_tokens': total_tokens_with_this_coda,
            'n_compared': total_compared,
            'matches': matches,
            'mismatches': mismatches,
            'agreement_rate': matches / total_compared if total_compared > 0 else 0.0,
            'mismatch_distribution': dict(mismatch_details.most_common(10)),
            'coda_present_in_decoded': coda_present_count / total_tokens_with_this_coda
            if total_tokens_with_this_coda > 0 else 0.0,
            'coda_is_last_char': coda_last_char_count / total_tokens_with_this_coda
            if total_tokens_with_this_coda > 0 else 0.0,
        }

    return per_coda


# ---------------------------------------------------------------------------
# Per-triple cross-validation
# ---------------------------------------------------------------------------

def _per_triple_xval(
    all_tokens: List[str],
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    coda_table: Any,
    clean_indices: Set[int],
    confirmed_keys: Set[str],
) -> Dict[str, Dict[str, Any]]:
    """For each triple (12 confirmed + 13 unresolved), compute xval rate
    when that triple appears in a token that also has a coda marker."""
    per_triple: Dict[str, Dict[str, Any]] = {}

    for idx, token in enumerate(all_tokens):
        if idx not in clean_indices:
            continue

        eva_chars = tokenize_eva_chars(token)
        classified = classify_token_chars_v2(eva_chars, coda_table)

        # Get codas
        codas = []
        for role, char in classified:
            if role == 'CODA_MARKER':
                coda_val = get_coda(char, coda_table)
                if coda_val:
                    codas.append(coda_val)

        if not codas:
            continue

        # Decode
        try:
            result = decode_token_cvc_v2(token, assignment, eva_to_triple, coda_table)
            decoded = result.decoded_cvc
        except Exception:
            continue

        if not decoded:
            continue

        pos_ending, _ = _classify_latin_ending(decoded)
        if not pos_ending or pos_ending == 'UNCLEAR':
            continue

        # Get last coda category for comparison
        last_coda = codas[-1]
        coda_cat = CODA_GRAMMAR.get(last_coda, {}).get('category', 'UNKNOWN')
        if coda_cat == 'UNKNOWN':
            continue

        coda_is_verbal = coda_cat == 'VERBAL'
        ending_is_verbal = pos_ending == 'VERB'
        coda_is_nominal = coda_cat == 'NOMINAL'
        ending_is_nominal = pos_ending == 'NOUN'
        agrees = (coda_is_verbal and ending_is_verbal) or \
                 (coda_is_nominal and ending_is_nominal)

        # Attribute to all triples in this token
        for role, char in classified:
            if role == 'SYLLABIC':
                triple_key = eva_to_triple.get(char, '')
                if not triple_key:
                    continue
                if triple_key not in per_triple:
                    per_triple[triple_key] = {
                        'is_confirmed': triple_key in confirmed_keys,
                        'matches': 0,
                        'mismatches': 0,
                    }
                if agrees:
                    per_triple[triple_key]['matches'] += 1
                else:
                    per_triple[triple_key]['mismatches'] += 1

    # Compute rates
    for triple_key, data in per_triple.items():
        total = data['matches'] + data['mismatches']
        data['total'] = total
        data['xval_rate'] = data['matches'] / total if total > 0 else 0.0

    # Aggregate: confirmed vs unresolved
    confirmed_rates = [v['xval_rate'] for v in per_triple.values()
                       if v['is_confirmed'] and v.get('total', 0) > 0]
    unresolved_rates = [v['xval_rate'] for v in per_triple.values()
                        if not v['is_confirmed'] and v.get('total', 0) > 0]

    confirmed_mean = float(np.mean(confirmed_rates)) if confirmed_rates else 0.0
    unresolved_mean = float(np.mean(unresolved_rates)) if unresolved_rates else 0.0

    return {
        'per_triple': per_triple,
        'confirmed_mean_xval': confirmed_mean,
        'unresolved_mean_xval': unresolved_mean,
        'gap': confirmed_mean - unresolved_mean,
        'n_confirmed_triples': len(confirmed_rates),
        'n_unresolved_triples': len(unresolved_rates),
    }


# ---------------------------------------------------------------------------
# Per-section cross-validation
# ---------------------------------------------------------------------------

def _per_section_xval(
    all_tokens: List[str],
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    coda_table: Any,
    clean_indices: Set[int],
    section_list: List[str],
) -> Dict[str, Dict[str, Any]]:
    """Cross-validation rate by manuscript section."""
    section_data: Dict[str, Dict[str, int]] = defaultdict(lambda: {'match': 0, 'total': 0})

    for idx, token in enumerate(all_tokens):
        if idx not in clean_indices:
            continue

        eva_chars = tokenize_eva_chars(token)
        classified = classify_token_chars_v2(eva_chars, coda_table)

        codas = []
        for role, char in classified:
            if role == 'CODA_MARKER':
                coda_val = get_coda(char, coda_table)
                if coda_val:
                    codas.append(coda_val)

        if not codas:
            continue

        try:
            result = decode_token_cvc_v2(token, assignment, eva_to_triple, coda_table)
            decoded = result.decoded_cvc
        except Exception:
            continue

        if not decoded:
            continue

        pos_ending, _ = _classify_latin_ending(decoded)
        if not pos_ending or pos_ending == 'UNCLEAR':
            continue

        last_coda = codas[-1]
        coda_cat = CODA_GRAMMAR.get(last_coda, {}).get('category', 'UNKNOWN')
        if coda_cat == 'UNKNOWN':
            continue

        agrees = ((coda_cat == 'VERBAL' and pos_ending == 'VERB') or
                  (coda_cat == 'NOMINAL' and pos_ending == 'NOUN'))

        section = section_list[idx] if idx < len(section_list) else 'unknown'
        section_data[section]['total'] += 1
        if agrees:
            section_data[section]['match'] += 1

    result_dict = {}
    for section, counts in sorted(section_data.items()):
        total = counts['total']
        result_dict[section] = {
            'match': counts['match'],
            'total': total,
            'xval_rate': counts['match'] / total if total > 0 else 0.0,
        }

    return result_dict


# ---------------------------------------------------------------------------
# By decoded string length and number of codas
# ---------------------------------------------------------------------------

def _by_length_and_codas(
    all_tokens: List[str],
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    coda_table: Any,
    clean_indices: Set[int],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Cross-validation rate by decoded string length and by number of codas."""
    length_data: Dict[int, Dict[str, int]] = defaultdict(lambda: {'match': 0, 'total': 0})
    coda_count_data: Dict[int, Dict[str, int]] = defaultdict(lambda: {'match': 0, 'total': 0})

    for idx, token in enumerate(all_tokens):
        if idx not in clean_indices:
            continue

        eva_chars = tokenize_eva_chars(token)
        classified = classify_token_chars_v2(eva_chars, coda_table)

        codas = []
        for role, char in classified:
            if role == 'CODA_MARKER':
                coda_val = get_coda(char, coda_table)
                if coda_val:
                    codas.append(coda_val)

        if not codas:
            continue

        try:
            result = decode_token_cvc_v2(token, assignment, eva_to_triple, coda_table)
            decoded = result.decoded_cvc
        except Exception:
            continue

        if not decoded:
            continue

        pos_ending, _ = _classify_latin_ending(decoded)
        if not pos_ending or pos_ending == 'UNCLEAR':
            continue

        last_coda = codas[-1]
        coda_cat = CODA_GRAMMAR.get(last_coda, {}).get('category', 'UNKNOWN')
        if coda_cat == 'UNKNOWN':
            continue

        agrees = ((coda_cat == 'VERBAL' and pos_ending == 'VERB') or
                  (coda_cat == 'NOMINAL' and pos_ending == 'NOUN'))

        # Bin by decoded length (2-char bins)
        length_bin = (len(decoded) // 2) * 2
        length_data[length_bin]['total'] += 1
        if agrees:
            length_data[length_bin]['match'] += 1

        # By number of codas
        n_codas = len(codas)
        coda_count_data[n_codas]['total'] += 1
        if agrees:
            coda_count_data[n_codas]['match'] += 1

    by_length = {
        str(k): {
            'match': v['match'],
            'total': v['total'],
            'xval_rate': v['match'] / v['total'] if v['total'] > 0 else 0.0,
        }
        for k, v in sorted(length_data.items())
    }

    by_n_codas = {
        str(k): {
            'match': v['match'],
            'total': v['total'],
            'xval_rate': v['match'] / v['total'] if v['total'] > 0 else 0.0,
        }
        for k, v in sorted(coda_count_data.items())
    }

    return by_length, by_n_codas


# ---------------------------------------------------------------------------
# Error taxonomy
# ---------------------------------------------------------------------------

def _error_taxonomy(
    all_tokens: List[str],
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    coda_table: Any,
    clean_indices: Set[int],
) -> Dict[str, Any]:
    """Classify each cross-validation disagreement into error categories."""
    error_types: Dict[str, int] = Counter()
    n_total = 0
    n_coda_but_no_ending = 0
    n_ending_but_no_coda = 0

    for idx, token in enumerate(all_tokens):
        if idx not in clean_indices:
            continue

        eva_chars = tokenize_eva_chars(token)
        classified = classify_token_chars_v2(eva_chars, coda_table)

        codas = []
        for role, char in classified:
            if role == 'CODA_MARKER':
                coda_val = get_coda(char, coda_table)
                if coda_val:
                    codas.append(coda_val)

        if not codas:
            continue

        try:
            result = decode_token_cvc_v2(token, assignment, eva_to_triple, coda_table)
            decoded = result.decoded_cvc
        except Exception:
            continue

        if not decoded:
            continue

        pos_ending, case_ending = _classify_latin_ending(decoded)
        last_coda = codas[-1]
        coda_cat = CODA_GRAMMAR.get(last_coda, {}).get('category', 'UNKNOWN')

        if coda_cat == 'UNKNOWN':
            continue

        n_total += 1

        if not pos_ending or pos_ending == 'UNCLEAR':
            n_coda_but_no_ending += 1
            continue

        coda_verbal = coda_cat == 'VERBAL'
        coda_nominal = coda_cat == 'NOMINAL'
        end_verbal = pos_ending == 'VERB'
        end_nominal = pos_ending == 'NOUN'
        end_particle = pos_ending == 'PARTICLE'

        if (coda_verbal and end_verbal) or (coda_nominal and end_nominal):
            error_types['AGREE'] += 1
        elif coda_verbal and end_nominal:
            error_types['CODA_VERBAL_END_NOMINAL'] += 1
        elif coda_nominal and end_verbal:
            error_types['CODA_NOMINAL_END_VERBAL'] += 1
        elif end_particle:
            error_types['ENDING_PARTICLE'] += 1
        else:
            error_types['OTHER'] += 1

    return {
        'n_total_with_coda': n_total,
        'n_coda_but_no_ending': n_coda_but_no_ending,
        'error_types': dict(error_types),
        'error_fractions': {
            k: v / n_total if n_total > 0 else 0.0
            for k, v in error_types.items()
        },
    }


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class XvalDiagnosisResult:
    phase: str = "72"
    step: str = "72.2"
    experiment: str = "xval_diagnosis"
    # Per-coda analysis
    per_coda: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    worst_coda: str = ""
    best_coda: str = ""
    # Per-triple analysis
    per_triple_summary: Dict[str, Any] = field(default_factory=dict)
    confirmed_mean_xval: float = 0.0
    unresolved_mean_xval: float = 0.0
    triple_gap: float = 0.0
    # Per-section analysis
    per_section: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    section_variance: float = 0.0
    # By decoded length
    by_decoded_length: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # By number of codas
    by_n_codas: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # Error taxonomy
    error_taxonomy: Dict[str, Any] = field(default_factory=dict)
    # Coda presence diagnostics
    coda_r_present_in_decoded: float = 0.0
    coda_r_is_last_char: float = 0.0
    # Overall
    overall_xval: float = 0.0
    dominant_error_source: str = ""
    # Gates
    gate_xv1: bool = False   # >= 1 coda type has agreement rate > 50%
    gate_xv2: bool = False   # Confirmed triples xval > unresolved + 10pp
    gate_xv3: bool = False   # Connector-r xval < other codas by > 15pp
    gate_xv4: bool = False   # Section variation significant (range > 10pp)
    gates_passed: int = 0
    gate_passed: bool = False
    verdict: str = "UNKNOWN"
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_xval_diagnosis():
    """Track 2: Diagnose cross-validation failure in CVC model."""
    t0 = time.time()
    rd = str(_results_dir())

    print("Phase 72.2 — Cross-Validation Failure Diagnosis")
    print("=" * 49)

    # --- Load dependencies ---
    refine_data = _safe_load(os.path.join(rd, 'combined_refine.json'))
    assignment = refine_data.get('best_assignment', {})

    clean_data = _safe_load(os.path.join(rd, 'p69_clean_corpus.json'))
    clean_indices = set(clean_data.get('clean_indices', []))

    confirmed, unresolved = _get_confirmed_and_unresolved(rd)
    confirmed_keys = set(confirmed.keys())

    eva_to_triple = build_eva_to_triple_lookup()
    coda_table = build_coda_table_v2()

    corpus = load_corpus(verbose=False)
    all_tokens = corpus.get_tokens()
    section_list = _build_section_list(corpus)

    print(f"  Total tokens: {len(all_tokens)}")
    print(f"  Clean tokens: {len(clean_indices)}")
    print(f"  Confirmed triples: {len(confirmed_keys)}")
    print(f"  Unresolved triples: {len(unresolved)}")

    # --- 1. Per-coda cross-validation ---
    print("\n  1. Per-coda cross-validation...")
    per_coda = _per_coda_xval(all_tokens, assignment, eva_to_triple,
                              coda_table, clean_indices)

    for coda in ['n', 'r', 's', 't']:
        data = per_coda[coda]
        print(f"    -{coda}: {data['agreement_rate']:.1%} agreement "
              f"({data['matches']}/{data['n_compared']}), "
              f"present in decoded: {data['coda_present_in_decoded']:.1%}, "
              f"is last char: {data['coda_is_last_char']:.1%}")

    # Find worst and best
    coda_rates = {c: d['agreement_rate'] for c, d in per_coda.items()
                  if d['n_compared'] > 0}
    worst_coda = min(coda_rates, key=coda_rates.get) if coda_rates else ''
    best_coda = max(coda_rates, key=coda_rates.get) if coda_rates else ''
    print(f"    Worst: -{worst_coda} ({coda_rates.get(worst_coda, 0):.1%})")
    print(f"    Best:  -{best_coda} ({coda_rates.get(best_coda, 0):.1%})")

    # --- 2. Per-triple cross-validation ---
    print("\n  2. Per-triple cross-validation...")
    triple_results = _per_triple_xval(all_tokens, assignment, eva_to_triple,
                                      coda_table, clean_indices, confirmed_keys)

    print(f"    Confirmed mean xval: {triple_results['confirmed_mean_xval']:.1%} "
          f"({triple_results['n_confirmed_triples']} triples)")
    print(f"    Unresolved mean xval: {triple_results['unresolved_mean_xval']:.1%} "
          f"({triple_results['n_unresolved_triples']} triples)")
    print(f"    Gap: {triple_results['gap']:.1%}")

    # Show per-triple details for top/bottom 5
    per_triple = triple_results.get('per_triple', {})
    sorted_triples = sorted(per_triple.items(),
                            key=lambda x: x[1].get('xval_rate', 0),
                            reverse=True)

    if sorted_triples:
        print("    Best triples:")
        for tk, td in sorted_triples[:5]:
            status = 'C' if td['is_confirmed'] else 'U'
            syl = assignment.get(tk, '??')
            print(f"      [{status}] {tk} -> {syl}: {td['xval_rate']:.1%} "
                  f"({td.get('total', 0)} obs)")
        print("    Worst triples:")
        for tk, td in sorted_triples[-5:]:
            status = 'C' if td['is_confirmed'] else 'U'
            syl = assignment.get(tk, '??')
            print(f"      [{status}] {tk} -> {syl}: {td['xval_rate']:.1%} "
                  f"({td.get('total', 0)} obs)")

    # --- 3. Per-section cross-validation ---
    print("\n  3. Per-section cross-validation...")
    per_section = _per_section_xval(all_tokens, assignment, eva_to_triple,
                                    coda_table, clean_indices, section_list)

    section_rates = []
    for section in sorted(per_section.keys()):
        data = per_section[section]
        print(f"    {section}: {data['xval_rate']:.1%} ({data['total']} tokens)")
        if data['total'] >= 50:
            section_rates.append(data['xval_rate'])

    section_variance = max(section_rates) - min(section_rates) if len(section_rates) >= 2 else 0.0
    print(f"    Range: {section_variance:.1%}")

    # --- 4. By decoded length and number of codas ---
    print("\n  4. By decoded string length and coda count...")
    by_length, by_n_codas = _by_length_and_codas(
        all_tokens, assignment, eva_to_triple, coda_table, clean_indices)

    for length_bin, data in by_length.items():
        if data['total'] >= 20:
            print(f"    Length {length_bin}-{int(length_bin)+1}: "
                  f"{data['xval_rate']:.1%} ({data['total']} tokens)")

    for n_codas, data in by_n_codas.items():
        print(f"    {n_codas} coda(s): {data['xval_rate']:.1%} ({data['total']} tokens)")

    # --- 5. Error taxonomy ---
    print("\n  5. Error taxonomy...")
    taxonomy = _error_taxonomy(all_tokens, assignment, eva_to_triple,
                               coda_table, clean_indices)

    for err_type, count in sorted(taxonomy['error_types'].items(),
                                  key=lambda x: -x[1]):
        frac = taxonomy['error_fractions'].get(err_type, 0)
        print(f"    {err_type}: {count} ({frac:.1%})")

    print(f"    Tokens with coda but no classifiable ending: "
          f"{taxonomy['n_coda_but_no_ending']}")

    # --- Determine dominant error source ---
    r_rate = per_coda.get('r', {}).get('agreement_rate', 0)
    other_rates = [per_coda[c]['agreement_rate'] for c in ['n', 's', 't']
                   if per_coda[c].get('n_compared', 0) > 0]
    other_mean = float(np.mean(other_rates)) if other_rates else 0.0

    if r_rate < other_mean - 0.15:
        dominant = 'CONNECTOR_R_DOMINANT'
    elif triple_results['gap'] > 0.15:
        dominant = 'UNRESOLVED_TRIPLES_DOMINANT'
    elif section_variance > 0.15:
        dominant = 'SECTION_DEPENDENT'
    elif max(coda_rates.values()) < 0.35 if coda_rates else True:
        dominant = 'COMBINATION_MODEL_WRONG'
    else:
        dominant = 'DISTRIBUTED_ERROR'

    print(f"\n  Dominant error source: {dominant}")

    # --- Overall xval ---
    total_agree = sum(per_coda[c]['matches'] for c in per_coda)
    total_compared = sum(per_coda[c]['n_compared'] for c in per_coda)
    overall_xval = total_agree / total_compared if total_compared > 0 else 0.0
    print(f"  Overall cross-validation: {overall_xval:.1%}")

    # --- Gates ---
    g1 = any(d['agreement_rate'] > 0.50 for d in per_coda.values()
             if d['n_compared'] > 0)
    g2 = triple_results['gap'] > 0.10
    g3 = r_rate < other_mean - 0.15 if other_rates else False
    g4 = section_variance > 0.10

    gates_passed = sum([g1, g2, g3, g4])

    print(f"\n  Gates:")
    print(f"    XV1 (any coda > 50% agreement): {'PASS' if g1 else 'FAIL'}")
    print(f"    XV2 (confirmed > unresolved + 10pp): {'PASS' if g2 else 'FAIL'}")
    print(f"    XV3 (connector-r worst by > 15pp): {'PASS' if g3 else 'FAIL'}")
    print(f"    XV4 (section variance > 10pp): {'PASS' if g4 else 'FAIL'}")
    print(f"    Total: {gates_passed}/4")

    # --- Verdict ---
    if g3 and gates_passed >= 3:
        verdict = 'CONNECTOR_R_IS_PROBLEM'
    elif g2 and gates_passed >= 2:
        verdict = 'TRIPLE_RESOLUTION_NEEDED'
    elif not g1 and gates_passed <= 1:
        verdict = 'COMBINATION_MODEL_FLAWED'
    else:
        verdict = 'MIXED_DIAGNOSIS'

    print(f"\n  Verdict: {verdict}")

    # --- Build result ---
    result = XvalDiagnosisResult(
        per_coda=per_coda,
        worst_coda=worst_coda,
        best_coda=best_coda,
        per_triple_summary={
            'confirmed_mean_xval': triple_results['confirmed_mean_xval'],
            'unresolved_mean_xval': triple_results['unresolved_mean_xval'],
            'gap': triple_results['gap'],
            'n_confirmed': triple_results['n_confirmed_triples'],
            'n_unresolved': triple_results['n_unresolved_triples'],
        },
        confirmed_mean_xval=triple_results['confirmed_mean_xval'],
        unresolved_mean_xval=triple_results['unresolved_mean_xval'],
        triple_gap=triple_results['gap'],
        per_section=per_section,
        section_variance=section_variance,
        by_decoded_length=by_length,
        by_n_codas=by_n_codas,
        error_taxonomy=taxonomy,
        coda_r_present_in_decoded=per_coda.get('r', {}).get('coda_present_in_decoded', 0.0),
        coda_r_is_last_char=per_coda.get('r', {}).get('coda_is_last_char', 0.0),
        overall_xval=overall_xval,
        dominant_error_source=dominant,
        gate_xv1=g1,
        gate_xv2=g2,
        gate_xv3=g3,
        gate_xv4=g4,
        gates_passed=gates_passed,
        gate_passed=gates_passed >= 2,
        verdict=verdict,
        runtime_seconds=time.time() - t0,
    )

    path = _save_json(rd, 'phase72_xval.json', asdict(result))
    print(f"\n  Saved: {path}")
    print(f"  Runtime: {result.runtime_seconds:.1f}s")

    return result
