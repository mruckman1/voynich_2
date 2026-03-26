"""
Phase 75, Step 0: Re-Decode Corpus (3-Coda Model)
====================================================
Phase 73 established that connector strokes have no phonetic value
(connector→null). This step tests the "3-coda model" which additionally
sets descender→null, leaving only hook, sigmoid, and vertical as
phonetically active coda strokes.

OLD model = Phase 73 baseline: connector→null, descender→'r'
NEW model = 3-coda:            connector→null, descender→null

This step re-decodes all 36,238 tokens under the 3-coda model
and compares against the Phase 73 baseline.

Dependency chain:
    results/combined_refine.json         (Phase 15)
    results/modifier_integrate.json      (Phase 16)
    results/null_corpus.json             (Phase 17)
        -> results/p75_redecode.json
"""

import json
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Set, Tuple

import numpy as np

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import build_eva_to_triple_lookup, load_corpus, tokenize_eva_chars
from voynich.core.reference import build_expanded_word_set, load_reference_corpus
from voynich.phases.corrected_coda import (
    build_coda_table_v2,
    classify_token_chars_v2,
    decode_token_cvc_v2,
)
from voynich.phases.cvc_coda_signal import (
    _build_folio_list,
    _compute_bigram_z,
    _run_signal_isolation,
)
from voynich.phases.null_corpus import (
    _build_eva_bigram_model,
    _generate_null_corpus,
    _reconstruct_modifier_rules,
)
from voynich.phases.p72_connector import _build_coda_table_with_connector


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
# 3-coda table builder
# ---------------------------------------------------------------------------

def _build_3coda_table():
    """Build 3-coda table: connector→null AND descender→null."""
    table = _build_coda_table_with_connector('')  # connector→null
    table.stroke_to_coda['descender'] = ''  # descender→null
    return table


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class RedecodeResult:
    phase: str = "75"
    step: str = "75.0"
    experiment: str = "redecode_3coda"
    # Corpus stats
    n_tokens: int = 0
    n_changed: int = 0
    changed_fraction: float = 0.0
    # Dict-hit comparison
    old_dict_hit: float = 0.0
    new_dict_hit: float = 0.0
    delta_dict_hit: float = 0.0
    new_dict_hits: int = 0
    lost_dict_hits: int = 0
    # Signal comparison
    old_signal_count: int = 0
    new_signal_count: int = 0
    # Bigram z
    old_bigram_z: float = 0.0
    new_bigram_z: float = 0.0
    # Mean word length
    old_mean_length: float = 0.0
    new_mean_length: float = 0.0
    # Cross-validation
    old_xval: float = 0.0
    new_xval: float = 0.0
    # Per-section dict-hit
    section_dict_hits: Dict[str, float] = field(default_factory=dict)
    # Sample changes (first 50)
    sample_changes: List[Dict[str, str]] = field(default_factory=list)
    # Decoded corpus (stored for downstream tracks)
    decoded_tokens: List[str] = field(default_factory=list)
    folio_list: List[str] = field(default_factory=list)
    section_list: List[str] = field(default_factory=list)
    # Gate
    gate_r0: bool = False
    verdict: str = "UNKNOWN"
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Section/hand list builders
# ---------------------------------------------------------------------------

def _build_section_list(corpus) -> List[str]:
    """Build flat list of section labels, one per token."""
    sections: List[str] = []
    for _folio, page in corpus.pages.items():
        for _ in page.all_tokens:
            sections.append(getattr(page, 'section', 'unknown'))
    return sections


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def run_redecode_3coda() -> RedecodeResult:
    """Re-decode the full corpus with 3-coda model and compare against Phase 73 baseline."""
    t0 = time.time()
    rd = str(_results_dir())

    print("Phase 75, Step 0: Re-Decode Corpus (3-Coda Model)")
    print("=" * 60)

    # --- Load shared data ---
    eva_to_triple = build_eva_to_triple_lookup()

    refine_data = _safe_load(os.path.join(rd, 'combined_refine.json'))
    assignment = refine_data.get('best_assignment', {})

    mod_data = _safe_load(os.path.join(rd, 'modifier_integrate.json'))
    modifier_chars, modifier_rules = _reconstruct_modifier_rules(mod_data)

    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    base_words = set(w.lower() for w in ref_corpus.get_combined_tokens('latin')
                     if len(w) >= 2)
    expanded, _ = build_expanded_word_set(base_words)
    ref_word_set = base_words | expanded

    corpus = load_corpus(verbose=False)
    all_tokens = corpus.get_tokens()
    folios = _build_folio_list(corpus)
    sections = _build_section_list(corpus)

    # --- Build null corpora for signal/bigram ---
    null_data = _safe_load(os.path.join(rd, 'null_corpus.json'))
    null_seeds = ([r['seed'] for r in null_data.get('null_runs', [])]
                  if null_data else [100, 101, 102, 103, 104])

    bigram_probs, initial_probs, token_lengths = _build_eva_bigram_model(all_tokens)

    # --- Build two coda tables ---
    old_coda = _build_coda_table_with_connector('')  # Phase 73 baseline: connector→null
    new_coda = _build_3coda_table()  # 3-coda: connector→null AND descender→null

    print(f"  Corpus: {len(all_tokens)} tokens")
    print(f"  Old coda (Phase 73): connector→'{old_coda.stroke_to_coda['connector']}' (null), "
          f"descender→'{old_coda.stroke_to_coda['descender']}'")
    print(f"  New coda (3-coda):   connector→'{new_coda.stroke_to_coda['connector']}' (null), "
          f"descender→'{new_coda.stroke_to_coda['descender']}' (null)")

    # --- Decode both ways ---
    print("  Decoding with old model (Phase 73 baseline)...")
    old_decoded = []
    for token in all_tokens:
        result = decode_token_cvc_v2(token, assignment, eva_to_triple, old_coda)
        old_decoded.append(result.decoded_cvc)

    print("  Decoding with new model (3-coda: connector+descender→null)...")
    new_decoded = []
    for token in all_tokens:
        result = decode_token_cvc_v2(token, assignment, eva_to_triple, new_coda)
        new_decoded.append(result.decoded_cvc)

    # --- Compare ---
    n_changed = 0
    new_hits = 0
    lost_hits = 0
    sample_changes = []

    for i, (old_d, new_d) in enumerate(zip(old_decoded, new_decoded)):
        if old_d != new_d:
            n_changed += 1
            old_in = old_d.lower() in ref_word_set if old_d else False
            new_in = new_d.lower() in ref_word_set if new_d else False
            if new_in and not old_in:
                new_hits += 1
            elif old_in and not new_in:
                lost_hits += 1

            if len(sample_changes) < 50:
                sample_changes.append({
                    'idx': i,
                    'folio': folios[i],
                    'section': sections[i],
                    'eva': all_tokens[i],
                    'old': old_d,
                    'new': new_d,
                    'old_dict': old_in,
                    'new_dict': new_in,
                })

    # --- Dict-hit ---
    old_dict_count = sum(1 for d in old_decoded if d and d.lower() in ref_word_set)
    new_dict_count = sum(1 for d in new_decoded if d and d.lower() in ref_word_set)
    old_dict_hit = old_dict_count / len(old_decoded) if old_decoded else 0.0
    new_dict_hit = new_dict_count / len(new_decoded) if new_decoded else 0.0

    # --- Mean length ---
    old_lengths = [len(d) for d in old_decoded if d]
    new_lengths = [len(d) for d in new_decoded if d]
    old_mean_len = float(np.mean(old_lengths)) if old_lengths else 0.0
    new_mean_len = float(np.mean(new_lengths)) if new_lengths else 0.0

    print(f"  Changed tokens: {n_changed} ({100*n_changed/len(all_tokens):.1f}%)")
    print(f"  Dict-hit: {100*old_dict_hit:.1f}% → {100*new_dict_hit:.1f}% "
          f"(Δ={100*(new_dict_hit-old_dict_hit):+.1f}%)")
    print(f"  Mean length: {old_mean_len:.2f} → {new_mean_len:.2f}")

    # --- Generate null decoded corpora under NEW model ---
    print("  Generating null corpora under 3-coda model...")
    null_decoded_list = []
    for seed in null_seeds:
        null_tokens = _generate_null_corpus(
            bigram_probs, initial_probs, token_lengths, len(all_tokens), seed)
        null_dec = []
        for token in null_tokens:
            result = decode_token_cvc_v2(token, assignment, eva_to_triple, new_coda)
            null_dec.append(result.decoded_cvc)
        null_decoded_list.append(null_dec)

    # --- Also generate null decoded for OLD model ---
    print("  Generating null corpora under Phase 73 baseline...")
    old_null_decoded_list = []
    for seed in null_seeds:
        null_tokens = _generate_null_corpus(
            bigram_probs, initial_probs, token_lengths, len(all_tokens), seed)
        null_dec = []
        for token in null_tokens:
            result = decode_token_cvc_v2(token, assignment, eva_to_triple, old_coda)
            null_dec.append(result.decoded_cvc)
        old_null_decoded_list.append(null_dec)

    # --- Signal isolation ---
    print("  Running signal isolation (old)...")
    old_signal = _run_signal_isolation(old_decoded, old_null_decoded_list,
                                       ref_word_set, len(all_tokens))
    print("  Running signal isolation (new)...")
    new_signal = _run_signal_isolation(new_decoded, null_decoded_list,
                                       ref_word_set, len(all_tokens))

    print(f"  Signal words: {old_signal.n_signal_words} → {new_signal.n_signal_words}")

    # --- Bigram z ---
    print("  Computing bigram z (old)...")
    old_bigram_z = _compute_bigram_z(old_decoded, old_null_decoded_list,
                                      ref_word_set, folios, n_perms=200)
    print("  Computing bigram z (new)...")
    new_bigram_z = _compute_bigram_z(new_decoded, null_decoded_list,
                                      ref_word_set, folios, n_perms=200)

    print(f"  Bigram z: {old_bigram_z:.2f} → {new_bigram_z:.2f}")

    # --- Cross-validation ---
    from voynich.phases.p72_connector import _compute_xval_for_connector
    clean_data = _safe_load(os.path.join(rd, 'p69_clean_corpus.json'))
    clean_indices = set(clean_data.get('clean_indices', []))

    old_xval = _compute_xval_for_connector(old_decoded, all_tokens, old_coda,
                                            clean_indices)
    new_xval = _compute_xval_for_connector(new_decoded, all_tokens, new_coda,
                                            clean_indices)
    print(f"  Cross-validation: {100*old_xval:.1f}% → {100*new_xval:.1f}%")

    # --- Per-section dict-hit ---
    section_hits: Dict[str, List[bool]] = {}
    for i, (dec, sec) in enumerate(zip(new_decoded, sections)):
        if sec not in section_hits:
            section_hits[sec] = []
        section_hits[sec].append(bool(dec and dec.lower() in ref_word_set))

    section_dict_hits = {
        sec: sum(hits) / len(hits) if hits else 0.0
        for sec, hits in sorted(section_hits.items())
    }
    for sec, rate in section_dict_hits.items():
        print(f"    {sec}: {100*rate:.1f}%")

    # --- Gate ---
    gate_r0 = new_dict_hit >= old_dict_hit - 0.005  # allow 0.5% tolerance
    verdict = "IMPROVED" if new_dict_hit > old_dict_hit else (
        "NEUTRAL" if gate_r0 else "DEGRADED")

    result = RedecodeResult(
        n_tokens=len(all_tokens),
        n_changed=n_changed,
        changed_fraction=n_changed / len(all_tokens) if all_tokens else 0.0,
        old_dict_hit=old_dict_hit,
        new_dict_hit=new_dict_hit,
        delta_dict_hit=new_dict_hit - old_dict_hit,
        new_dict_hits=new_hits,
        lost_dict_hits=lost_hits,
        old_signal_count=old_signal.n_signal_words,
        new_signal_count=new_signal.n_signal_words,
        old_bigram_z=old_bigram_z,
        new_bigram_z=new_bigram_z,
        old_mean_length=old_mean_len,
        new_mean_length=new_mean_len,
        old_xval=old_xval,
        new_xval=new_xval,
        section_dict_hits=section_dict_hits,
        sample_changes=sample_changes,
        decoded_tokens=new_decoded,
        folio_list=folios,
        section_list=sections,
        gate_r0=gate_r0,
        verdict=verdict,
        runtime_seconds=time.time() - t0,
    )

    path = _save_json(rd, 'p75_redecode.json', asdict(result))
    print(f"\n  Verdict: {verdict}")
    print(f"  Saved: {path}")
    print(f"  Runtime: {result.runtime_seconds:.1f}s")
    return result
