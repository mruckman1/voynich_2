"""
Phase 32.1 – Compound-Sign Corpus Decode
==========================================
Decode every token through the compound-sign pipeline: decompose into
prefix + stem + suffix, strip gallows from stem, decode stem via R3,
map suffix to Latin ending, assemble the final word.

Also decode 5 null corpora through the same pipeline for fair signal
classification in Step 32.2.

Dependency chain:
    combined_refine.json       (Phase 15 assignment)
    modifier_integrate.json    (Phase 16 modifiers)
    null_corpus.json           (Phase 17 seeds)
    compound_sign_test.json    (Phase 31.6 — suffix map verification)
        → compound_decode.json  (this step)
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
from voynich.phases.morpheme_grid import decompose_token_morphemes
from voynich.phases.null_corpus import (
    _build_eva_bigram_model,
    _generate_null_corpus,
    _reconstruct_modifier_rules,
)
from voynich.phases.signal_isolation import _decode_corpus_r3


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUFFIX_ENDING_MAP = {
    'dy': 'a', 'y': 'i', 'ey': 'e', 'aiin': 'um',
    'ol': 'is', 'al': 'ae', 'in': 'em', 'am': 'am',
    'iin': 'en', 'm': 'um', 'aiiin': 'ium', 'iiin': 'ium',
    'an': 'an', 'n': 'n',
}

GALLOWS_CHARS = {'k', 't', 'p', 'f'}


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


def _strip_gallows(eva_token: str) -> str:
    """Remove gallows characters (k, t, p, f) from an EVA token."""
    chars = tokenize_eva_chars(eva_token)
    stripped = [ch for ch in chars if ch not in GALLOWS_CHARS]
    return ''.join(stripped) if stripped else eva_token


# ---------------------------------------------------------------------------
# Compound decode pipeline
# ---------------------------------------------------------------------------

def _compound_decode_tokens(
    tokens: List[str],
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    modifier_chars: set,
    modifier_rules: Dict[str, str],
    ref_word_set: set,
) -> Tuple[List[str], List[str], List[str], List[str], List[str], List[str]]:
    """Compound-sign decode a list of tokens.

    For each token:
      1. Decompose into prefix / stem / suffix
      2. Strip gallows from stem
      3. Decode cleaned stem via R3
      4. Map suffix → Latin ending
      5. Try root alone → root+ending → root[:-1]+ending → pick first dict hit

    Returns six parallel arrays:
      (decoded_words, decoded_roots, prefixes, suffixes, latin_endings, strategies)
    """
    # Step 1: Decompose all tokens
    decomps = [decompose_token_morphemes(t) for t in tokens]

    # Step 2: Strip gallows from stems
    cleaned_stems = [_strip_gallows(d.stem) for d in decomps]

    # Step 3: Decode cleaned stems via R3
    root_decoded = _decode_corpus_r3(
        cleaned_stems, assignment, eva_to_triple,
        modifier_chars, modifier_rules, ref_word_set,
    )

    # Step 4–5: Assemble final decoded words
    decoded_words: List[str] = []
    strategies: List[str] = []
    prefixes: List[str] = []
    suffixes: List[str] = []
    latin_endings: List[str] = []

    for i, d in enumerate(decomps):
        root = root_decoded[i]
        suffix = d.suffix
        ending = SUFFIX_ENDING_MAP.get(suffix, '')
        prefixes.append(d.prefix)
        suffixes.append(suffix)
        latin_endings.append(ending)

        # Try root alone
        if root in ref_word_set:
            decoded_words.append(root)
            strategies.append('root_only')
            continue

        # Try root + Latin ending
        if ending:
            combined = root + ending
            if combined in ref_word_set:
                decoded_words.append(combined)
                strategies.append('root_plus_ending')
                continue

            # Try trimming last char of root and adding ending
            if len(root) > 2:
                trimmed = root[:-1] + ending
                if trimmed in ref_word_set:
                    decoded_words.append(trimmed)
                    strategies.append('trimmed_plus_ending')
                    continue

        # No dict hit — use root + ending as best guess
        final = (root + ending) if ending else root
        decoded_words.append(final)
        strategies.append('no_hit')

    return (
        decoded_words, root_decoded,
        prefixes, suffixes, latin_endings, strategies,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_compound_decode() -> None:
    """Step 32.1: Full compound-sign corpus decode."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE 32.1: Compound-Sign Corpus Decode")
    print("=" * 70)

    rd = _results_dir()
    eva_to_triple = build_eva_to_triple_lookup()

    # ── 1. Load inputs ──
    print("\n  1. Loading inputs ...")

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

    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    base_words = set(
        w.lower() for w in ref_corpus.get_combined_tokens('latin')
        if len(w) >= 2
    )
    expanded, _ = build_expanded_word_set(base_words)
    ref_word_set = base_words | expanded

    print(f"     Assignment: {len(assignment)} triples")
    print(f"     Modifiers: {len(modifier_chars)} chars")
    print(f"     Reference words: {len(ref_word_set)}")

    # ── 2. Decode real corpus ──
    print("\n  2. Decoding real corpus (compound-sign) ...")
    corpus = load_corpus(verbose=False)

    token_folios: List[str] = []
    token_evas: List[str] = []
    all_tokens: List[str] = []

    for folio, page in corpus.pages.items():
        for token in page.all_tokens:
            token_folios.append(folio)
            token_evas.append(token)
            all_tokens.append(token)

    n_tokens = len(all_tokens)

    (
        decoded_words, decoded_roots,
        prefixes, suffixes, latin_endings, strategies,
    ) = _compound_decode_tokens(
        all_tokens, assignment, eva_to_triple,
        modifier_chars, modifier_rules, ref_word_set,
    )

    real_hits = [w in ref_word_set for w in decoded_words]
    dict_hit_rate = sum(real_hits) / n_tokens
    print(f"     {n_tokens} tokens, compound dict_hit = {dict_hit_rate:.4f}")

    # Strategy breakdown
    strat_counts = Counter(strategies)
    for s in ['root_only', 'root_plus_ending', 'trimmed_plus_ending', 'no_hit']:
        n = strat_counts.get(s, 0)
        print(f"       {s}: {n} ({n / n_tokens:.1%})")

    # ── 3. Decode null corpora ──
    print("\n  3. Decoding 5 null corpora (compound-sign) ...")
    bigram_probs, initial_probs, token_lengths = _build_eva_bigram_model(
        all_tokens,
    )

    null_hit_rates: List[float] = []
    null_token_hits: List[List[bool]] = []

    for i, seed in enumerate(null_seeds):
        print(f"     Null corpus {i + 1}/{len(null_seeds)} (seed={seed}) ...")
        null_tokens = _generate_null_corpus(
            bigram_probs, initial_probs, token_lengths, n_tokens, seed,
        )
        null_decoded, _, _, _, _, _ = _compound_decode_tokens(
            null_tokens, assignment, eva_to_triple,
            modifier_chars, modifier_rules, ref_word_set,
        )
        hits = [w in ref_word_set for w in null_decoded]
        null_token_hits.append(hits)
        rate = sum(hits) / len(hits)
        null_hit_rates.append(rate)
        print(f"       compound dict_hit = {rate:.4f}")

    null_mean = sum(null_hit_rates) / len(null_hit_rates)
    selectivity = dict_hit_rate / null_mean if null_mean > 0 else float('inf')

    # ── 4. Compute baseline for comparison ──
    print("\n  4. Computing baseline (full-token R3) ...")
    baseline_decoded = _decode_corpus_r3(
        all_tokens, assignment, eva_to_triple,
        modifier_chars, modifier_rules, ref_word_set,
    )
    baseline_hits = sum(1 for w in baseline_decoded if w in ref_word_set)
    baseline_dict_hit = baseline_hits / n_tokens
    print(f"     Baseline dict_hit = {baseline_dict_hit:.4f}")
    print(f"     Delta = {dict_hit_rate - baseline_dict_hit:+.4f}")

    # ── 5. Save output ──
    print("\n  5. Saving compound_decode.json ...")
    output = {
        'token_folios': token_folios,
        'token_evas': token_evas,
        'token_decoded': decoded_words,
        'token_dict_hits': real_hits,
        'token_roots': decoded_roots,
        'token_prefixes': prefixes,
        'token_suffixes': suffixes,
        'token_latin_endings': latin_endings,
        'token_strategies': strategies,
        'n_tokens': n_tokens,
        'dict_hit_rate': round(dict_hit_rate, 6),
        'baseline_dict_hit': round(baseline_dict_hit, 6),
        'delta_dict_hit': round(dict_hit_rate - baseline_dict_hit, 6),
        'n_root_only': strat_counts.get('root_only', 0),
        'n_root_plus_ending': strat_counts.get('root_plus_ending', 0),
        'n_trimmed_plus_ending': strat_counts.get('trimmed_plus_ending', 0),
        'n_no_hit': strat_counts.get('no_hit', 0),
        'null_dict_hit_rates': [round(r, 6) for r in null_hit_rates],
        'null_mean_dict_hit': round(null_mean, 6),
        'compound_selectivity': round(selectivity, 4),
        'null_token_hits': null_token_hits,
        'runtime_seconds': round(time.time() - t0, 1),
    }

    with open(os.path.join(rd, 'compound_decode.json'), 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n  Compound dict_hit: {dict_hit_rate:.4f}")
    print(f"  Null mean:         {null_mean:.4f}")
    print(f"  Selectivity:       {selectivity:.2f}x")
    print(f"\n  Step 32.1 completed in {time.time() - t0:.1f}s")
