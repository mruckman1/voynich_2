"""
Step 35.2 – Combined Spatial + 10K Dictionary Decode
=====================================================
Decode the spatial-conditioned corpus through the Phase 15 triple table
and evaluate against 10K / 17K / 131K dictionaries.

Tokens conditioned to '' (STANDALONE silent) decode to '' with no dict hit.
R3 decode uses the 131K dictionary for best alteration/stripping choices,
then final matching checks all three dictionary sizes.

Also decodes 5 null corpora through the same pipeline.

Dependency chain:
    spatial_preprocess.json    (Step 35.1)
    combined_refine.json       (Phase 15 assignment)
    modifier_integrate.json    (Phase 16 modifiers)
        → combined_decode.json (this step)
"""

import json
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    load_corpus,
    tokenize_eva_chars,
)
from voynich.core.reference import (
    build_expanded_word_set,
    load_reference_corpus,
)
from voynich.phases.dict_calibration import _build_dict_variants
from voynich.phases.null_corpus import _reconstruct_modifier_rules
from voynich.phases.signal_isolation import _decode_corpus_r3


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
# Decode pipeline
# ---------------------------------------------------------------------------

def _decode_conditioned_tokens(
    conditioned_tokens: List[str],
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    modifier_chars: set,
    modifier_rules: Dict[str, str],
    ref_word_set_131k: set,
) -> List[str]:
    """Decode spatial-conditioned tokens via R3 strategy.

    Tokens that are '' (standalone silent) decode to ''.
    All others: standard R3 decode.
    """
    # Separate empty vs non-empty tokens
    non_empty_indices = []
    non_empty_tokens = []
    for i, t in enumerate(conditioned_tokens):
        if t:
            non_empty_indices.append(i)
            non_empty_tokens.append(t)

    # R3 decode on non-empty tokens (uses 131K dict for best alteration choices)
    if non_empty_tokens:
        decoded_non_empty = _decode_corpus_r3(
            non_empty_tokens, assignment, eva_to_triple,
            modifier_chars, modifier_rules, ref_word_set_131k,
        )
    else:
        decoded_non_empty = []

    # Reassemble full decoded array
    decoded = [''] * len(conditioned_tokens)
    for idx, orig_idx in enumerate(non_empty_indices):
        decoded[orig_idx] = decoded_non_empty[idx]

    return decoded


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_combined_decode() -> None:
    """Step 35.2: Combined spatial+10K corpus decode."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 35.2: Combined Spatial + 10K Dictionary Decode")
    print("=" * 70)

    rd = _results_dir()
    eva_to_triple = build_eva_to_triple_lookup()

    # ── 1. Load inputs ──
    print("\n  1. Loading inputs ...")

    sp_path = os.path.join(rd, 'spatial_preprocess.json')
    if not os.path.exists(sp_path):
        print("  [SKIP] spatial_preprocess.json not found")
        return
    with open(sp_path) as f:
        sp = json.load(f)

    with open(os.path.join(rd, 'combined_refine.json')) as f:
        refine_data = json.load(f)
    assignment = refine_data.get('best_assignment', {})

    with open(os.path.join(rd, 'modifier_integrate.json')) as f:
        mod_data = json.load(f)
    modifier_chars, modifier_rules = _reconstruct_modifier_rules(mod_data)

    print(f"     Assignment: {len(assignment)} triples")
    print(f"     Modifiers: {len(modifier_chars)} chars")

    # ── 2. Build dictionaries ──
    print("\n  2. Building 10K / 17K / 131K dictionaries ...")
    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    base_words = set(
        w.lower() for w in ref_corpus.get_combined_tokens('latin')
        if len(w) >= 2
    )

    variants = _build_dict_variants(base_words, ref_corpus, [10000, 17000])
    dict_10k = variants[0][1]
    dict_17k = variants[1][1]

    expanded, _ = build_expanded_word_set(base_words)
    dict_131k = base_words | expanded

    print(f"     10K: {len(dict_10k)} words")
    print(f"     17K: {len(dict_17k)} words")
    print(f"     131K: {len(dict_131k)} words")

    # ── 3. Decode real corpus ──
    print("\n  3. Decoding real corpus (spatial-conditioned) ...")
    token_conditioned = sp['token_conditioned']
    token_folios = sp['token_folios']
    token_evas = sp['token_evas']
    n_tokens = sp['n_tokens']

    decoded = _decode_conditioned_tokens(
        token_conditioned, assignment, eva_to_triple,
        modifier_chars, modifier_rules, dict_131k,
    )

    hits_10k = [w in dict_10k for w in decoded]
    hits_17k = [w in dict_17k for w in decoded]
    hits_131k = [w in dict_131k for w in decoded]

    rate_10k = sum(hits_10k) / n_tokens
    rate_17k = sum(hits_17k) / n_tokens
    rate_131k = sum(hits_131k) / n_tokens

    print(f"     {n_tokens} tokens")
    print(f"     dict_hit 10K:  {rate_10k:.4f}")
    print(f"     dict_hit 17K:  {rate_17k:.4f}")
    print(f"     dict_hit 131K: {rate_131k:.4f}")

    # ── 4. Decode null corpora ──
    print("\n  4. Decoding 5 null corpora ...")
    null_conditioned = sp['null_conditioned']
    null_token_hits_10k: List[List[bool]] = []
    null_hit_rates: List[float] = []

    for i, null_cond in enumerate(null_conditioned):
        print(f"     Null corpus {i + 1}/{len(null_conditioned)} ...")
        null_decoded = _decode_conditioned_tokens(
            null_cond, assignment, eva_to_triple,
            modifier_chars, modifier_rules, dict_131k,
        )
        null_hits = [w in dict_10k for w in null_decoded]
        null_token_hits_10k.append(null_hits)
        null_rate = sum(null_hits) / len(null_hits) if null_hits else 0
        null_hit_rates.append(null_rate)
        print(f"       dict_hit 10K: {null_rate:.4f}")

    null_mean = sum(null_hit_rates) / len(null_hit_rates) if null_hit_rates else 0
    selectivity = rate_10k / null_mean if null_mean > 0 else float('inf')

    # ── 5. Compute baseline for comparison ──
    print("\n  5. Computing baseline (unspatial R3 decode) ...")
    baseline_decoded = _decode_corpus_r3(
        token_evas, assignment, eva_to_triple,
        modifier_chars, modifier_rules, dict_131k,
    )
    baseline_hits_131k = sum(1 for w in baseline_decoded if w in dict_131k)
    baseline_rate_131k = baseline_hits_131k / n_tokens
    print(f"     Baseline 131K dict_hit: {baseline_rate_131k:.4f}")

    # ── 6. Save ──
    print("\n  6. Saving combined_decode.json ...")
    output = {
        'token_folios': token_folios,
        'token_evas': token_evas,
        'token_conditioned': token_conditioned,
        'token_decoded': decoded,
        'token_dict_hits_10k': hits_10k,
        'token_dict_hits_17k': hits_17k,
        'token_dict_hits_131k': hits_131k,
        'n_tokens': n_tokens,
        'dict_hit_rate_10k': round(rate_10k, 6),
        'dict_hit_rate_17k': round(rate_17k, 6),
        'dict_hit_rate_131k': round(rate_131k, 6),
        'baseline_dict_hit_131k': round(baseline_rate_131k, 6),
        'delta_dict_hit_131k': round(rate_131k - baseline_rate_131k, 6),
        'null_dict_hit_rates_10k': [round(r, 6) for r in null_hit_rates],
        'null_mean_dict_hit_10k': round(null_mean, 6),
        'selectivity_10k': round(selectivity, 4) if selectivity != float('inf') else 999.0,
        'null_token_hits_10k': null_token_hits_10k,
        'runtime_seconds': round(time.time() - t0, 1),
    }

    out_path = os.path.join(rd, 'combined_decode.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n  Combined 10K dict_hit: {rate_10k:.4f}")
    print(f"  Null mean 10K:        {null_mean:.4f}")
    print(f"  Selectivity:          {selectivity:.2f}x")
    print(f"\n  Step 35.2 completed in {time.time() - t0:.1f}s")
