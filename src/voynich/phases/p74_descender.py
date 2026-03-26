"""
Phase 74, Track A1: Exhaustive Descender Value Testing
======================================================
Phase 72 found that connector→null is correct (xval 77.9%→90.5%), but
descender→r remains with only 16.9% cross-validation (Phase 72 Track 2).
The descender produces 14,164 r-codas (39% of all coda tokens), creating
a 57% verbal fraction incompatible with natural Latin.

This track applies the same methodology as Phase 72 Track 1: exhaustively
test 13 possible descender values while holding connector→null fixed.

Test values: 6 consonants (l, m, n, r, s, t), 5 vowels (a, e, i, o, u),
null (descender produces nothing), word boundary (descender splits token).

Also analyzes WHERE descenders appear within tokens (medial vs final).

Dependency chain:
    results/combined_refine.json         (Phase 15)
    results/modifier_integrate.json      (Phase 16)
    results/p69_clean_corpus.json        (Phase 69)
    results/null_corpus.json             (Phase 17)
        -> results/p74_descender.json
"""

import json
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import build_eva_to_triple_lookup, load_corpus, tokenize_eva_chars
from voynich.core.reference import build_expanded_word_set, load_reference_corpus
from voynich.phases.coda_markers import (
    CodaTable,
    build_coda_table,
    classify_token_chars,
    get_coda,
)
from voynich.phases.corrected_coda import (
    build_coda_table_v2,
    classify_token_chars_v2,
    decode_token_cvc_v2,
)
from voynich.phases.cvc_coda_signal import (
    _build_folio_list,
    _compute_bigram_z,
    _load_shared_data,
    _run_signal_isolation,
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
# Build coda table with custom descender value (connector always null)
# ---------------------------------------------------------------------------

def _build_coda_table_with_descender(descender_value: str) -> CodaTable:
    """Build corrected coda table with connector→null and custom descender.

    descender_value can be:
    - A consonant/vowel character: descender maps to that character
    - '' (empty): descender produces nothing (null)
    - ' ' (space): descender marks a word boundary within the token
    """
    table = build_coda_table_v2()
    # Apply Phase 73 correction: connector → null
    table.stroke_to_coda['connector'] = ''
    # Set descender to test value
    if descender_value == '':
        table.stroke_to_coda['descender'] = ''
    elif descender_value == ' ':
        table.stroke_to_coda['descender'] = ' '
    else:
        table.stroke_to_coda['descender'] = descender_value
    return table


# ---------------------------------------------------------------------------
# Decode with custom descender
# ---------------------------------------------------------------------------

def _decode_corpus_with_descender(
    tokens: List[str],
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    coda_table: CodaTable,
    is_boundary: bool = False,
) -> List[str]:
    """Decode corpus using a custom descender value."""
    decoded = []
    for token in tokens:
        result = decode_token_cvc_v2(token, assignment, eva_to_triple, coda_table)
        decoded.append(result.decoded_cvc)
    return decoded


def _compute_metrics(
    decoded: List[str],
    null_decoded_list: List[List[str]],
    ref_word_set: Set[str],
    folios: List[str],
    is_boundary: bool = False,
) -> Dict[str, Any]:
    """Compute dict-hit, signal words, bigram_z for a decoded corpus."""
    if is_boundary:
        all_words = []
        word_folios = []
        for i, d in enumerate(decoded):
            if d:
                parts = d.split()
                all_words.extend(parts)
                word_folios.extend([folios[i]] * len(parts))
            else:
                all_words.append('')
                word_folios.append(folios[i])

        dict_hits = sum(1 for w in all_words if w and w.lower() in ref_word_set)
        dict_rate = dict_hits / len(all_words) if all_words else 0.0
        mean_len = float(np.mean([len(w) for w in all_words if w])) if all_words else 0.0

        signal_stats = _run_signal_isolation(
            all_words, [['' for _ in all_words]] * 5,
            ref_word_set, len(all_words))
        signal_count = signal_stats.n_signal_words

        return {
            'dict_hit': dict_rate,
            'n_dict_hits': dict_hits,
            'n_words': len(all_words),
            'signal_count': signal_count,
            'bigram_z': 0.0,
            'mean_word_length': mean_len,
        }
    else:
        dict_hits = sum(1 for d in decoded if d and d.lower() in ref_word_set)
        dict_rate = dict_hits / len(decoded) if decoded else 0.0
        mean_len = float(np.mean([len(d) for d in decoded if d])) if decoded else 0.0

        signal_stats = _run_signal_isolation(
            decoded, null_decoded_list, ref_word_set, len(decoded))
        signal_count = signal_stats.n_signal_words

        bigram_z = _compute_bigram_z(decoded, null_decoded_list,
                                     ref_word_set, folios, n_perms=200)

        return {
            'dict_hit': dict_rate,
            'n_dict_hits': dict_hits,
            'n_words': len(decoded),
            'signal_count': signal_count,
            'bigram_z': bigram_z,
            'mean_word_length': mean_len,
        }


# ---------------------------------------------------------------------------
# Cross-validation for a specific descender value
# ---------------------------------------------------------------------------

def _compute_xval_for_descender(
    decoded: List[str],
    all_tokens: List[str],
    coda_table: CodaTable,
    clean_indices: Set[int],
) -> float:
    """Compute coda-vs-ending cross-validation rate for given decoded output."""
    from voynich.phases.inflectional_catalog import CODA_GRAMMAR

    n_agree = 0
    n_comparable = 0

    for idx, (token, dec) in enumerate(zip(all_tokens, decoded)):
        if idx not in clean_indices:
            continue
        if not dec:
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

        last_coda = codas[-1]
        coda_cat = CODA_GRAMMAR.get(last_coda, {}).get('category', 'UNKNOWN')
        if coda_cat == 'UNKNOWN':
            continue

        pos_ending, _ = _classify_latin_ending(dec)
        if not pos_ending or pos_ending == 'UNCLEAR':
            continue

        n_comparable += 1
        if ((coda_cat == 'VERBAL' and pos_ending == 'VERB') or
                (coda_cat == 'NOMINAL' and pos_ending == 'NOUN')):
            n_agree += 1

    return n_agree / n_comparable if n_comparable > 0 else 0.0


# ---------------------------------------------------------------------------
# Grammatical distribution computation
# ---------------------------------------------------------------------------

def _compute_verbal_fraction(
    decoded: List[str],
    all_tokens: List[str],
    coda_table: CodaTable,
    clean_indices: Set[int],
) -> Dict[str, Any]:
    """Compute the fraction of tokens classified as VERBAL by coda markers."""
    from voynich.phases.inflectional_catalog import CODA_GRAMMAR

    counts = Counter()

    for idx, (token, dec) in enumerate(zip(all_tokens, decoded)):
        if idx not in clean_indices:
            continue
        if not dec:
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
            counts['UNMARKED'] += 1
            continue

        last_coda = codas[-1]
        grammar = CODA_GRAMMAR.get(last_coda, {})
        cat = grammar.get('category', 'UNMARKED')
        counts[cat] += 1

    total = sum(counts.values())
    verbal_fraction = counts.get('VERBAL', 0) / total if total > 0 else 0.0

    return {
        'verbal': counts.get('VERBAL', 0),
        'nominal': counts.get('NOMINAL', 0),
        'unmarked': counts.get('UNMARKED', 0),
        'total': total,
        'verbal_fraction': verbal_fraction,
        'nominal_fraction': counts.get('NOMINAL', 0) / total if total > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Descender position analysis
# ---------------------------------------------------------------------------

def _descender_position_analysis(
    all_tokens: List[str],
    coda_table: CodaTable,
) -> Dict[str, Any]:
    """Analyze where descender-type modifiers appear within tokens.

    Compare to connector analysis from Phase 72:
    - Connectors were 98.1% token-medial → scribal ligature
    - If descenders are mostly token-final → genuine coda consonant
    - If mixed → context-dependent behavior
    """
    positions = {
        'total': 0,
        'n_final': 0,
        'n_medial': 0,
        'n_initial_adjacent': 0,
        'relative_positions': [],
        'preceding_roles': Counter(),
        'following_roles': Counter(),
    }

    for token in all_tokens:
        eva_chars = tokenize_eva_chars(token)
        classified = classify_token_chars_v2(eva_chars, coda_table)

        for pos, (role, char) in enumerate(classified):
            if role != 'CODA_MARKER':
                continue

            # Check if this is a descender-type coda
            last_stroke = coda_table.eva_modifiers.get(char)
            if last_stroke != 'descender':
                continue

            positions['total'] += 1
            is_final = pos == len(classified) - 1
            is_medial = 0 < pos < len(classified) - 1

            if is_final:
                positions['n_final'] += 1
            if is_medial:
                positions['n_medial'] += 1
            if pos <= 1:
                positions['n_initial_adjacent'] += 1

            rel_pos = pos / len(classified) if classified else 0
            positions['relative_positions'].append(rel_pos)

            if pos > 0:
                positions['preceding_roles'][classified[pos - 1][0]] += 1
            if pos < len(classified) - 1:
                positions['following_roles'][classified[pos + 1][0]] += 1

    mean_rel = float(np.mean(positions['relative_positions'])) \
        if positions['relative_positions'] else 0.0

    return {
        'total_descenders': positions['total'],
        'n_final': positions['n_final'],
        'n_medial': positions['n_medial'],
        'n_initial_adjacent': positions['n_initial_adjacent'],
        'final_fraction': positions['n_final'] / positions['total']
        if positions['total'] > 0 else 0.0,
        'medial_fraction': positions['n_medial'] / positions['total']
        if positions['total'] > 0 else 0.0,
        'mean_relative_position': mean_rel,
        'preceding_roles': dict(positions['preceding_roles']),
        'following_roles': dict(positions['following_roles']),
    }


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class DescenderResult:
    phase: str = "74"
    step: str = "74.A1"
    experiment: str = "descender_investigation"
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    ranking: List[Tuple[str, float]] = field(default_factory=list)
    best_value: str = ""
    best_composite: float = 0.0
    current_r_composite: float = 0.0
    improvement_over_r: float = 0.0
    best_verbal_fraction: float = 0.0
    current_r_verbal_fraction: float = 0.0
    # Position analysis
    position_analysis: Dict[str, Any] = field(default_factory=dict)
    # Comparison to connector
    connector_final_fraction: float = 0.019  # From Phase 72
    # Gates
    gate_da1: bool = False   # Best value != 'r'
    gate_da2: bool = False   # Best composite > r composite + 0.005
    gate_da3: bool = False   # Best value verbal fraction < 40%
    gate_da4: bool = False   # Descender final_fraction > 60%
    gate_da5: bool = False   # Reserved for context-dependent (Track A2)
    gate_da6: bool = False   # Reserved for per-triple (Track A2)
    gates_passed: int = 0
    gate_passed: bool = False
    verdict: str = "UNKNOWN"
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_descender_test():
    """Track A1: Test all possible descender values."""
    t0 = time.time()
    rd = str(_results_dir())

    print("Phase 74.A1 — Exhaustive Descender Value Testing")
    print("=" * 50)

    # --- Load shared data ---
    print("  Loading shared data...")
    shared = _load_shared_data()
    all_tokens = shared['all_tokens']
    folios = shared['folios']
    assignment = shared['assignment']
    eva_to_triple = shared['eva_to_triple']
    ref_word_set = shared['ref_word_set']
    null_token_lists = shared['null_token_lists']

    clean_data = _safe_load(os.path.join(rd, 'p69_clean_corpus.json'))
    clean_indices = set(clean_data.get('clean_indices', []))

    print(f"  Tokens: {len(all_tokens)}, Clean: {len(clean_indices)}")

    # --- Test values ---
    TEST_VALUES = {
        # Consonants
        'l': 'l', 'm': 'm', 'n': 'n', 'r': 'r', 's': 's', 't': 't',
        # Vowels
        'a': 'a', 'e': 'e', 'i': 'i', 'o': 'o', 'u': 'u',
        # Special
        'null': '',
        'boundary': ' ',
    }

    candidates = {}

    for value_name, value in TEST_VALUES.items():
        print(f"\n  Testing descender -> '{value_name}' ({value!r})...")

        coda_table = _build_coda_table_with_descender(value)
        is_boundary = (value == ' ')

        # Decode real corpus
        decoded = _decode_corpus_with_descender(
            all_tokens, assignment, eva_to_triple, coda_table, is_boundary)

        # Decode null corpora (same descender value)
        null_decoded_list = []
        for null_tokens in null_token_lists:
            null_dec = _decode_corpus_with_descender(
                null_tokens, assignment, eva_to_triple, coda_table, is_boundary)
            null_decoded_list.append(null_dec)

        # Compute metrics
        metrics = _compute_metrics(decoded, null_decoded_list, ref_word_set,
                                   folios, is_boundary)

        # Cross-validation (skip for boundary)
        if not is_boundary:
            xval = _compute_xval_for_descender(decoded, all_tokens, coda_table,
                                                clean_indices)
        else:
            xval = 0.0

        # Grammatical distribution
        if not is_boundary:
            gram = _compute_verbal_fraction(decoded, all_tokens, coda_table,
                                            clean_indices)
        else:
            gram = {'verbal_fraction': 0.0, 'nominal_fraction': 0.0}

        candidates[value_name] = {
            'value_name': value_name,
            'value': value,
            'dict_hit': metrics['dict_hit'],
            'signal_count': metrics['signal_count'],
            'bigram_z': metrics['bigram_z'],
            'cross_validation': xval,
            'mean_word_length': metrics['mean_word_length'],
            'verbal_fraction': gram['verbal_fraction'],
            'nominal_fraction': gram.get('nominal_fraction', 0.0),
        }

        print(f"    dict_hit: {metrics['dict_hit']:.3f}, signal: {metrics['signal_count']}, "
              f"bigram_z: {metrics['bigram_z']:.2f}, xval: {xval:.3f}, "
              f"mean_len: {metrics['mean_word_length']:.1f}, "
              f"verbal: {gram['verbal_fraction']:.1%}")

    # --- Composite scoring ---
    # Same weights as Phase 72 (90%) + verbal fraction penalty (10%)
    # Verbal fraction target: ~15% (CI expected)
    print("\n  Computing composite scores...")
    for name, c in candidates.items():
        verbal_penalty = max(0, 1 - abs(c['verbal_fraction'] - 0.15) / 0.50)
        c['composite'] = (
            0.25 * c['dict_hit'] +
            0.20 * min(c['signal_count'] / 100.0, 1.0) +
            0.15 * min(max(c['bigram_z'], 0) / 200.0, 1.0) +
            0.20 * c['cross_validation'] +
            0.10 * max(0, 1 - abs(c['mean_word_length'] - 5.8) / 5.8) +
            0.10 * verbal_penalty
        )

    ranked = sorted(candidates.items(), key=lambda x: -x[1]['composite'])

    print("\n  Ranking:")
    for rank, (name, c) in enumerate(ranked, 1):
        marker = " <-- CURRENT" if name == 'r' else ""
        print(f"    {rank}. {name}: composite={c['composite']:.4f} "
              f"(dict={c['dict_hit']:.3f}, sig={c['signal_count']}, "
              f"z={c['bigram_z']:.1f}, xval={c['cross_validation']:.3f}, "
              f"verbal={c['verbal_fraction']:.1%}){marker}")

    best_name = ranked[0][0]
    best_composite = ranked[0][1]['composite']
    r_composite = candidates['r']['composite']
    r_xval = candidates['r']['cross_validation']
    best_xval = ranked[0][1]['cross_validation']
    best_verbal = ranked[0][1]['verbal_fraction']
    r_verbal = candidates['r']['verbal_fraction']

    print(f"\n  Best: {best_name} (composite={best_composite:.4f}, "
          f"verbal={best_verbal:.1%})")
    print(f"  Current (r): composite={r_composite:.4f}, "
          f"verbal={r_verbal:.1%}")
    print(f"  Improvement: {best_composite - r_composite:+.4f}")

    # --- Position analysis ---
    print("\n  Descender position analysis...")
    base_coda_table = build_coda_table_v2()
    base_coda_table.stroke_to_coda['connector'] = ''  # Apply Phase 73 correction
    pos_analysis = _descender_position_analysis(all_tokens, base_coda_table)
    print(f"    Total descenders: {pos_analysis['total_descenders']}")
    print(f"    Final: {pos_analysis['n_final']} ({pos_analysis['final_fraction']:.1%})")
    print(f"    Medial: {pos_analysis['n_medial']} ({pos_analysis['medial_fraction']:.1%})")
    print(f"    Mean relative position: {pos_analysis['mean_relative_position']:.2f}")
    print(f"    (cf. connector: 1.9% final, 98.1% medial)")

    # --- Gates ---
    g1 = best_name != 'r'
    g2 = best_composite > r_composite + 0.005
    g3 = best_verbal < 0.40
    g4 = pos_analysis['final_fraction'] > 0.60

    gates_passed = sum([g1, g2, g3, g4])

    print(f"\n  Gates:")
    print(f"    DA1 (best != r): {'PASS' if g1 else 'FAIL'}")
    print(f"    DA2 (composite > r + 0.005): {'PASS' if g2 else 'FAIL'}")
    print(f"    DA3 (best verbal < 40%): {'PASS' if g3 else 'FAIL'} "
          f"({best_verbal:.1%})")
    print(f"    DA4 (final_fraction > 60%): {'PASS' if g4 else 'FAIL'} "
          f"({pos_analysis['final_fraction']:.1%})")
    print(f"    Total: {gates_passed}/4")

    # --- Verdict ---
    if g1 and g2 and g3:
        verdict = 'DESCENDER_REVISED'
    elif g1 and g2:
        verdict = 'DESCENDER_IMPROVED'
    elif g1:
        verdict = 'DESCENDER_SUBOPTIMAL'
    else:
        verdict = 'DESCENDER_R_CONFIRMED'

    print(f"\n  Verdict: {verdict}")

    # --- Build result ---
    result = DescenderResult(
        candidates=[c for c in candidates.values()],
        ranking=[(name, c['composite']) for name, c in ranked],
        best_value=best_name,
        best_composite=best_composite,
        current_r_composite=r_composite,
        improvement_over_r=best_composite - r_composite,
        best_verbal_fraction=best_verbal,
        current_r_verbal_fraction=r_verbal,
        position_analysis=pos_analysis,
        gate_da1=g1,
        gate_da2=g2,
        gate_da3=g3,
        gate_da4=g4,
        gates_passed=gates_passed,
        gate_passed=gates_passed >= 2,
        verdict=verdict,
        runtime_seconds=time.time() - t0,
    )

    path = _save_json(rd, 'p74_descender.json', asdict(result))
    print(f"\n  Saved: {path}")
    print(f"  Runtime: {result.runtime_seconds:.1f}s")

    return result
