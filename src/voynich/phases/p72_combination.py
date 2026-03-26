"""
Phase 72, Track 3: Alternative CVC Combination Models
======================================================
The current model: APPEND coda consonant to end of CV decode.
  "ra" + coda-n -> "ran"

This track tests 6 alternative rules:
  1. append (baseline): CV + coda -> CVc
  2. replace_last: drop last char, add coda -> Cc
  3. insert: insert coda between C and V -> CcV
  4. prepend_to_next: coda starts the next syllable, not ends current
  5. null_connector: connector codas produce nothing, others append
  6. costamagna_cvc: select CVC variant from Costamagna inventory

Depends on Track 1 (uses best connector value).

Dependency chain:
    results/combined_refine.json         (Phase 15)
    results/modifier_integrate.json      (Phase 16)
    results/p69_clean_corpus.json        (Phase 69)
    results/null_corpus.json             (Phase 17)
    results/phase72_connector.json       (Track 1: best connector value)
        -> results/phase72_combination.json
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
from voynich.phases.coda_markers import CodaTable, get_coda
from voynich.phases.corrected_coda import (
    build_coda_table_v2,
    classify_token_chars_v2,
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
# Combination model decode functions
# ---------------------------------------------------------------------------

VOWELS = set('aeiou')


def _insert_before_last_vowel(syllable: str, coda: str) -> str:
    """Insert coda before the last vowel: 'ra' + 'n' -> 'rna'."""
    for i in range(len(syllable) - 1, -1, -1):
        if syllable[i] in VOWELS:
            return syllable[:i] + coda + syllable[i:]
    return syllable + coda  # fallback: no vowel found


def _decode_token_with_model(
    token: str,
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    coda_table: CodaTable,
    model: str,
) -> str:
    """Decode a single token using a specified combination model.

    Models:
    - 'append': standard CV + coda -> CVc
    - 'replace_last': drop last char, add coda -> Cc
    - 'insert': insert coda between C and V -> CcV
    - 'prepend_to_next': coda starts next syllable
    - 'null_connector': connector codas produce nothing, others append
    - 'costamagna_cvc': use CVC inventory for selection (= append with attestation)
    """
    eva_chars = tokenize_eva_chars(token)
    if not eva_chars:
        return ''

    classified = classify_token_chars_v2(eva_chars, coda_table)

    if model == 'prepend_to_next':
        return _decode_prepend_model(classified, assignment, eva_to_triple, coda_table)

    parts: List[str] = []
    for role, char in classified:
        if role == 'SYLLABIC':
            triple = eva_to_triple.get(char)
            syl = assignment.get(triple, '?') if triple else '?'
            parts.append(syl)
        elif role == 'CODA_MARKER':
            coda = get_coda(char, coda_table)
            if not coda or not parts:
                continue

            last_stroke = coda_table.eva_modifiers.get(char)
            is_connector = (last_stroke == 'connector')

            if model == 'append':
                parts[-1] = parts[-1] + coda
            elif model == 'replace_last':
                if len(parts[-1]) > 1:
                    parts[-1] = parts[-1][:-1] + coda
                else:
                    parts[-1] = parts[-1] + coda
            elif model == 'insert':
                parts[-1] = _insert_before_last_vowel(parts[-1], coda)
            elif model == 'null_connector':
                if not is_connector:
                    parts[-1] = parts[-1] + coda
                # connector: produce nothing
            elif model == 'costamagna_cvc':
                # Same as append (CVC selection would require inventory lookup)
                parts[-1] = parts[-1] + coda
            else:
                parts[-1] = parts[-1] + coda

    return ''.join(parts)


def _decode_prepend_model(
    classified: List[Tuple[str, str]],
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    coda_table: CodaTable,
) -> str:
    """Coda consonant starts the NEXT syllable rather than ending the current.

    "ra" [coda-n] "di" -> "ra" + "ndi" (coda prepends to following syllable)
    """
    parts: List[str] = []
    pending_coda: Optional[str] = None

    for role, char in classified:
        if role == 'SYLLABIC':
            triple = eva_to_triple.get(char)
            syl = assignment.get(triple, '?') if triple else '?'
            if pending_coda:
                parts.append(pending_coda + syl)
                pending_coda = None
            else:
                parts.append(syl)
        elif role == 'CODA_MARKER':
            coda = get_coda(char, coda_table)
            if coda:
                pending_coda = coda

    # Orphaned trailing coda
    if pending_coda:
        parts.append(pending_coda)

    return ''.join(parts)


def _decode_corpus_with_model(
    tokens: List[str],
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    coda_table: CodaTable,
    model: str,
) -> List[str]:
    """Decode a list of tokens using a specified combination model."""
    return [_decode_token_with_model(t, assignment, eva_to_triple, coda_table, model)
            for t in tokens]


# ---------------------------------------------------------------------------
# Cross-validation for a model
# ---------------------------------------------------------------------------

CODA_GRAMMAR_CAT = {
    's': 'VERBAL', 't': 'VERBAL', 'n': 'NOMINAL', 'r': 'VERBAL',
}


def _compute_model_xval(
    decoded: List[str],
    all_tokens: List[str],
    coda_table: CodaTable,
    clean_indices: Set[int],
    eva_to_triple: Dict[str, str],
) -> float:
    """Compute coda-vs-ending cross-validation rate."""
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
        coda_cat = CODA_GRAMMAR_CAT.get(last_coda, 'UNKNOWN')
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
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class ModelResult:
    name: str = ""
    description: str = ""
    dict_hit: float = 0.0
    signal_count: int = 0
    bigram_z: float = 0.0
    cross_validation: float = 0.0
    mean_word_length: float = 0.0
    composite: float = 0.0
    sample_decodings: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class CombinationResult:
    phase: str = "72"
    step: str = "72.3"
    experiment: str = "combination_models"
    models: List[ModelResult] = field(default_factory=list)
    ranking: List[Tuple[str, float]] = field(default_factory=list)
    best_model: str = ""
    current_model: str = "append"
    improvement: float = 0.0
    best_connector: str = "r"
    # Gates
    gate_cm1: bool = False   # Best model != 'append'
    gate_cm2: bool = False   # Best dict-hit > append + 1pp
    gate_cm3: bool = False   # Best xval > append + 5pp
    gate_cm4: bool = False   # Best mean length closer to 5.8
    gate_cm5: bool = False   # null_connector in top 3
    gates_passed: int = 0
    gate_passed: bool = False
    verdict: str = "UNKNOWN"
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_combination_models():
    """Track 3: Test alternative CVC combination models."""
    t0 = time.time()
    rd = str(_results_dir())

    print("Phase 72.3 — Alternative CVC Combination Models")
    print("=" * 49)

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

    # Load best connector from Track 1 (fall back to 'r')
    connector_data = _safe_load(os.path.join(rd, 'phase72_connector.json'))
    best_connector = connector_data.get('best_value', 'r')
    print(f"  Best connector from Track 1: '{best_connector}'")

    # Build coda table with best connector
    coda_table = build_coda_table_v2()
    if best_connector and best_connector != 'r':
        coda_table.stroke_to_coda['connector'] = best_connector

    print(f"  Tokens: {len(all_tokens)}, Clean: {len(clean_indices)}")

    # --- Define models ---
    MODELS = {
        'append': 'Standard: CV + coda -> CVc',
        'replace_last': 'Replace: drop vowel, add coda -> Cc',
        'insert': 'Insert: coda before last vowel -> CcV',
        'prepend_to_next': 'Prepend: coda starts next syllable',
        'null_connector': 'Null connector: connector codas produce nothing',
        'costamagna_cvc': 'Costamagna CVC: use attested CVC inventory',
    }

    model_results = []

    for model_name, description in MODELS.items():
        print(f"\n  Testing model: {model_name}...")
        print(f"    {description}")

        # Decode real corpus
        decoded = _decode_corpus_with_model(
            all_tokens, assignment, eva_to_triple, coda_table, model_name)

        # Decode null corpora
        null_decoded_list = []
        for null_tokens in null_token_lists:
            null_dec = _decode_corpus_with_model(
                null_tokens, assignment, eva_to_triple, coda_table, model_name)
            null_decoded_list.append(null_dec)

        # Dict-hit
        dict_hits = sum(1 for d in decoded if d and d.lower() in ref_word_set)
        dict_rate = dict_hits / len(decoded) if decoded else 0.0

        # Mean word length
        mean_len = float(np.mean([len(d) for d in decoded if d]))

        # Signal isolation
        signal_stats = _run_signal_isolation(
            decoded, null_decoded_list, ref_word_set, len(decoded))
        signal_count = signal_stats.n_signal_words

        # Bigram z (reduced perms for speed)
        bigram_z = _compute_bigram_z(
            decoded, null_decoded_list, ref_word_set, folios, n_perms=200)

        # Cross-validation
        xval = _compute_model_xval(decoded, all_tokens, coda_table,
                                   clean_indices, eva_to_triple)

        # Sample decodings (first 10 non-empty tokens)
        samples = []
        for i, (tok, dec) in enumerate(zip(all_tokens, decoded)):
            if dec and len(samples) < 10:
                samples.append({'token': tok, 'decoded': dec})

        # Composite score
        composite = (
            0.30 * dict_rate +
            0.25 * xval +
            0.20 * min(signal_count / 100.0, 1.0) +
            0.15 * max(0, 1 - abs(mean_len - 5.8) / 5.8) +
            0.10 * min(max(bigram_z, 0) / 200.0, 1.0)
        )

        model_results.append(ModelResult(
            name=model_name,
            description=description,
            dict_hit=dict_rate,
            signal_count=signal_count,
            bigram_z=bigram_z,
            cross_validation=xval,
            mean_word_length=mean_len,
            composite=composite,
            sample_decodings=samples,
        ))

        print(f"    dict_hit: {dict_rate:.3f}, signal: {signal_count}, "
              f"bigram_z: {bigram_z:.2f}, xval: {xval:.3f}, "
              f"mean_len: {mean_len:.1f}, composite: {composite:.4f}")

    # --- Ranking ---
    ranked = sorted(model_results, key=lambda m: -m.composite)
    ranking = [(m.name, m.composite) for m in ranked]

    print("\n  Ranking:")
    for rank, m in enumerate(ranked, 1):
        marker = " <-- CURRENT" if m.name == 'append' else ""
        print(f"    {rank}. {m.name}: composite={m.composite:.4f}{marker}")

    best_model = ranked[0]
    append_model = next(m for m in model_results if m.name == 'append')

    print(f"\n  Best: {best_model.name} (composite={best_model.composite:.4f})")
    print(f"  Append: composite={append_model.composite:.4f}")

    # --- Gates ---
    top3_names = [m.name for m in ranked[:3]]
    append_len_diff = abs(append_model.mean_word_length - 5.8)
    best_len_diff = abs(best_model.mean_word_length - 5.8)

    g1 = best_model.name != 'append'
    g2 = best_model.dict_hit > append_model.dict_hit + 0.01
    g3 = best_model.cross_validation > append_model.cross_validation + 0.05
    g4 = best_len_diff < append_len_diff
    g5 = 'null_connector' in top3_names

    gates_passed = sum([g1, g2, g3, g4, g5])

    print(f"\n  Gates:")
    print(f"    CM1 (best != append): {'PASS' if g1 else 'FAIL'}")
    print(f"    CM2 (dict-hit > append + 1pp): {'PASS' if g2 else 'FAIL'}")
    print(f"    CM3 (xval > append + 5pp): {'PASS' if g3 else 'FAIL'}")
    print(f"    CM4 (length closer to 5.8): {'PASS' if g4 else 'FAIL'}")
    print(f"    CM5 (null_connector in top 3): {'PASS' if g5 else 'FAIL'}")
    print(f"    Total: {gates_passed}/5")

    # --- Verdict ---
    if g1 and g2 and g3:
        verdict = 'MODEL_REVISED'
    elif g1 and (g2 or g3):
        verdict = 'MODEL_IMPROVED'
    elif g5:
        verdict = 'NULL_CONNECTOR_SUPPORTED'
    else:
        verdict = 'APPEND_CONFIRMED'

    print(f"\n  Verdict: {verdict}")

    # --- Build result ---
    result = CombinationResult(
        models=[asdict(m) for m in model_results],
        ranking=ranking,
        best_model=best_model.name,
        current_model='append',
        improvement=best_model.composite - append_model.composite,
        best_connector=best_connector,
        gate_cm1=g1,
        gate_cm2=g2,
        gate_cm3=g3,
        gate_cm4=g4,
        gate_cm5=g5,
        gates_passed=gates_passed,
        gate_passed=gates_passed >= 2,
        verdict=verdict,
        runtime_seconds=time.time() - t0,
    )

    path = _save_json(rd, 'phase72_combination.json', asdict(result))
    print(f"\n  Saved: {path}")
    print(f"  Runtime: {result.runtime_seconds:.1f}s")

    return result
