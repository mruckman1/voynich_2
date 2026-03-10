"""
Step 36.1 – Phase 16 Decode with 10K Matching
===============================================
Takes the existing Phase 16 decoded corpus (R3 strategy using 131K for
decode selection) and matches against the 10K dictionary.  No re-decoding
is needed — only the evaluation dictionary changes.

Also decodes 5 null corpora and matches against 10K for downstream
signal classification.

Dependency chain:
    combined_refine.json      (Phase 15 assignment)
    modifier_integrate.json   (Phase 16 modifiers)
    null_corpus.json          (Phase 17 seeds)
        → decode_10k.json     (this step)
"""

import json
import os
import time
from typing import Any, Dict, List, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    load_corpus,
)
from voynich.core.reference import (
    build_expanded_word_set,
    load_reference_corpus,
)
from voynich.phases.dict_calibration import _build_dict_variants
from voynich.phases.null_corpus import (
    _build_eva_bigram_model,
    _generate_null_corpus,
    _reconstruct_modifier_rules,
)
from voynich.phases.signal_isolation import _decode_corpus_r3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _convert(obj: Any) -> Any:
    if hasattr(obj, '__dataclass_fields__'):
        from dataclasses import asdict
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
# Main
# ---------------------------------------------------------------------------

def run_decode_10k() -> None:
    """Step 36.1: Phase 16 decode matched against 10K dictionary."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 36.1: Phase 16 Decode with 10K Matching")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load assignment + modifiers ──
    print("\n  1. Loading assignment and modifiers …")
    with open(os.path.join(rd, 'combined_refine.json')) as f:
        refine_data = json.load(f)
    assignment = refine_data.get('best_assignment', {})

    with open(os.path.join(rd, 'modifier_integrate.json')) as f:
        mod_data = json.load(f)
    modifier_chars, modifier_rules = _reconstruct_modifier_rules(mod_data)

    null_seeds = [100, 101, 102, 103, 104]
    null_path = os.path.join(rd, 'null_corpus.json')
    if os.path.exists(null_path):
        with open(null_path) as f:
            null_data = json.load(f)
        null_seeds = [r['seed'] for r in null_data.get('null_runs', [])]

    # ── 2. Load corpus + build lookup ──
    print("  2. Loading corpus …")
    corpus = load_corpus(verbose=False)
    eva_to_triple = build_eva_to_triple_lookup()

    token_folios: List[str] = []
    token_evas: List[str] = []
    all_tokens: List[str] = []
    for folio, page in corpus.pages.items():
        for token in page.all_tokens:
            token_folios.append(folio)
            token_evas.append(token)
            all_tokens.append(token)
    n_tokens = len(all_tokens)
    print(f"     {n_tokens} tokens across {len(corpus.pages)} folios")

    # ── 3. Build dictionaries ──
    print("  3. Building dictionaries …")
    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    base_words = set(
        w.lower() for w in ref_corpus.get_combined_tokens('latin')
        if len(w) >= 2
    )
    expanded_full, _ = build_expanded_word_set(base_words)
    dict_131k = base_words | expanded_full
    print(f"     131K dictionary: {len(dict_131k)} words")

    # Build 10K and 17K variants
    dict_variants = _build_dict_variants(base_words, ref_corpus, [10000, 17000])
    dict_10k = dict_variants[0][1]
    dict_17k = dict_variants[1][1]
    print(f"     10K dictionary: {len(dict_10k)} words")
    print(f"     17K dictionary: {len(dict_17k)} words")

    # ── 4. Decode real corpus (R3 using 131K for strategy selection) ──
    print("  4. Decoding real corpus (R3 strategy, 131K ref) …")
    real_decoded = _decode_corpus_r3(
        all_tokens, assignment, eva_to_triple,
        modifier_chars, modifier_rules, dict_131k,
    )

    # Match against all 3 dictionaries
    real_hits_10k = [w.lower() in dict_10k for w in real_decoded]
    real_hits_17k = [w.lower() in dict_17k for w in real_decoded]
    real_hits_131k = [w.lower() in dict_131k for w in real_decoded]

    dict_hit_10k = sum(real_hits_10k) / n_tokens
    dict_hit_17k = sum(real_hits_17k) / n_tokens
    dict_hit_131k = sum(real_hits_131k) / n_tokens
    print(f"     Dict-hit: 10K={dict_hit_10k:.3f}  17K={dict_hit_17k:.3f}  131K={dict_hit_131k:.3f}")

    # ── 5. Generate and decode null corpora ──
    print("  5. Generating and decoding null corpora …")
    bigram_probs, initial_probs, token_lengths = _build_eva_bigram_model(all_tokens)

    null_hits_10k_list: List[List[bool]] = []
    null_hits_17k_list: List[List[bool]] = []
    null_hits_131k_list: List[List[bool]] = []

    for i, seed in enumerate(null_seeds):
        null_tokens = _generate_null_corpus(
            bigram_probs, initial_probs, token_lengths, n_tokens, seed,
        )
        null_decoded = _decode_corpus_r3(
            null_tokens, assignment, eva_to_triple,
            modifier_chars, modifier_rules, dict_131k,
        )
        null_hits_10k_list.append([w.lower() in dict_10k for w in null_decoded])
        null_hits_17k_list.append([w.lower() in dict_17k for w in null_decoded])
        null_hits_131k_list.append([w.lower() in dict_131k for w in null_decoded])
        null_rate_10k = sum(null_hits_10k_list[-1]) / n_tokens
        print(f"     Null {i+1} (seed={seed}): 10K hit={null_rate_10k:.3f}")

    # ── 6. Compute aggregate null rates ──
    null_mean_10k = sum(sum(nh) / n_tokens for nh in null_hits_10k_list) / len(null_hits_10k_list)
    null_mean_17k = sum(sum(nh) / n_tokens for nh in null_hits_17k_list) / len(null_hits_17k_list)
    null_mean_131k = sum(sum(nh) / n_tokens for nh in null_hits_131k_list) / len(null_hits_131k_list)

    sel_10k = dict_hit_10k / null_mean_10k if null_mean_10k > 0 else float('inf')
    sel_17k = dict_hit_17k / null_mean_17k if null_mean_17k > 0 else float('inf')
    sel_131k = dict_hit_131k / null_mean_131k if null_mean_131k > 0 else float('inf')

    print(f"\n     Selectivity: 10K={sel_10k:.2f}  17K={sel_17k:.2f}  131K={sel_131k:.2f}")
    print(f"     Null mean:   10K={null_mean_10k:.3f}  17K={null_mean_17k:.3f}  131K={null_mean_131k:.3f}")

    # ── 7. Save ──
    elapsed = time.time() - t0

    output = {
        'n_tokens': n_tokens,
        'n_folios': len(corpus.pages),
        'dict_sizes': {
            '10k': len(dict_10k),
            '17k': len(dict_17k),
            '131k': len(dict_131k),
        },
        # Per-token parallel arrays
        'token_folios': token_folios,
        'token_evas': token_evas,
        'token_decoded': real_decoded,
        'token_hits_10k': real_hits_10k,
        'token_hits_17k': real_hits_17k,
        'token_hits_131k': real_hits_131k,
        # Null corpus per-token hits (for signal classification)
        'null_hits_10k': [list(nh) for nh in null_hits_10k_list],
        'null_hits_17k': [list(nh) for nh in null_hits_17k_list],
        'null_hits_131k': [list(nh) for nh in null_hits_131k_list],
        'null_seeds': null_seeds,
        # Aggregate statistics
        'dict_hit_10k': round(dict_hit_10k, 4),
        'dict_hit_17k': round(dict_hit_17k, 4),
        'dict_hit_131k': round(dict_hit_131k, 4),
        'null_mean_10k': round(null_mean_10k, 4),
        'null_mean_17k': round(null_mean_17k, 4),
        'null_mean_131k': round(null_mean_131k, 4),
        'selectivity_10k': round(sel_10k, 4),
        'selectivity_17k': round(sel_17k, 4),
        'selectivity_131k': round(sel_131k, 4),
        'runtime_seconds': round(elapsed, 1),
    }

    out_path = os.path.join(rd, 'decode_10k.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved → {out_path}")

    # ── Summary ──
    print("\n" + "=" * 70)
    print("DECODE 10K SUMMARY")
    print("=" * 70)
    print(f"\n  {'Dict':<8s} {'Size':>7s} {'Hit':>7s} {'Null':>7s} {'Sel':>7s}")
    print("  " + "-" * 36)
    print(f"  {'10K':<8s} {len(dict_10k):>7d} {dict_hit_10k:>7.3f} {null_mean_10k:>7.3f} {sel_10k:>7.2f}")
    print(f"  {'17K':<8s} {len(dict_17k):>7d} {dict_hit_17k:>7.3f} {null_mean_17k:>7.3f} {sel_17k:>7.2f}")
    print(f"  {'131K':<8s} {len(dict_131k):>7d} {dict_hit_131k:>7.3f} {null_mean_131k:>7.3f} {sel_131k:>7.2f}")
    print(f"\n  Runtime: {elapsed:.1f}s")
