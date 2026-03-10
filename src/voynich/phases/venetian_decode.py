"""
Step 39.12 -- Venetian Decode Test
====================================
Match decoded corpus against FULL dictionary (Latin 10K + Italian 10K +
Venetian supplement).  Compare Venetian-only selectivity vs Latin-only
vs Italian-only.

Dependency chain:
    venetian_lexicon.json      (Step 39.11)
    merged_dict.json           (Step 38.1)
    combined_refine.json       (Phase 15)  or  targeted_vowel_fix.json (39.3)
    null_corpus.json           (Phase 17)
    modifier_integrate.json    (Phase 16)
    decode_10k.json            (Step 36.1)
        -> venetian_decode.json  (this step)
"""

import json
import os
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import build_eva_to_triple_lookup, decode_token_modifier_aware


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


def _safe_load(path: str) -> Dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


# ---------------------------------------------------------------------------
# Signal classification
# ---------------------------------------------------------------------------

def _classify_4class(
    real_hits: List[bool],
    null_hits_list: List[List[bool]],
) -> List[str]:
    """4-class token classification: SIGNAL / SHARED_HIT / SHARED_MISS / ANTI_SIGNAL."""
    n_null = len(null_hits_list)
    classifications = []
    for i in range(len(real_hits)):
        n_null_hits = sum(
            null_hits_list[j][i]
            for j in range(n_null)
            if i < len(null_hits_list[j])
        )
        real_hit = real_hits[i]
        if real_hit and n_null_hits <= 1:
            classifications.append('SIGNAL')
        elif real_hit and n_null_hits >= 3:
            classifications.append('SHARED_HIT')
        elif not real_hit and n_null_hits >= 3:
            classifications.append('ANTI_SIGNAL')
        else:
            classifications.append('SHARED_MISS')
    return classifications


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_venetian_decode() -> None:
    """Step 39.12: Venetian Decode Test."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 39.12: Venetian Decode Test")
    print("=" * 70)

    rd = _results_dir()
    eva_to_triple = build_eva_to_triple_lookup()

    # -- 1. Load inputs --
    print("\n  1. Loading inputs ...")

    venetian_data = _safe_load(os.path.join(rd, 'venetian_lexicon.json'))
    dict_data = _safe_load(os.path.join(rd, 'merged_dict.json'))
    decode_data = _safe_load(os.path.join(rd, 'decode_10k.json'))
    null_data = _safe_load(os.path.join(rd, 'null_corpus.json'))
    mod_data = _safe_load(os.path.join(rd, 'modifier_integrate.json'))

    # Best assignment
    vowel_fix = _safe_load(os.path.join(rd, 'targeted_vowel_fix.json'))
    refine_data = _safe_load(os.path.join(rd, 'combined_refine.json'))

    if vowel_fix.get('corrected_assignment'):
        assignment = vowel_fix['corrected_assignment']
        assignment_source = 'targeted_vowel_fix'
    else:
        assignment = refine_data.get('best_assignment', {})
        assignment_source = 'combined_refine'

    modifier_chars: Set[str] = set(mod_data.get('modifier_chars', []))

    # Build dictionaries
    latin_10k: Set[str] = set(dict_data.get('latin_10k_words', []))
    italian_10k: Set[str] = set(dict_data.get('italian_10k_words', []))
    venetian_supplement: Set[str] = set(venetian_data.get('supplement_words', []))
    full_dict = latin_10k | italian_10k | venetian_supplement

    print(f"     Latin 10K: {len(latin_10k)}")
    print(f"     Italian 10K: {len(italian_10k)}")
    print(f"     Venetian supplement: {len(venetian_supplement)}")
    print(f"     Full dict: {len(full_dict)}")
    print(f"     Assignment source: {assignment_source}")

    # -- 2. Load or re-decode tokens --
    print("\n  2. Loading decoded tokens ...")

    token_decoded = decode_data.get('token_decoded', [])
    token_folios = decode_data.get('token_folios', [])

    if not token_decoded:
        print("     [WARN] No pre-decoded tokens found in decode_10k.json")
        token_decoded = []
        token_folios = []

    decoded_lower = [w.lower() for w in token_decoded]
    n_tokens = len(decoded_lower)

    print(f"     {n_tokens} tokens loaded")

    # -- 3. Match against each dictionary --
    print("\n  3. Matching against dictionaries ...")

    full_hits = [w in full_dict for w in decoded_lower]
    latin_only_hits = [w in latin_10k for w in decoded_lower]
    italian_only_hits = [w in italian_10k for w in decoded_lower]
    venetian_only_hits = [w in venetian_supplement and w not in latin_10k
                          and w not in italian_10k for w in decoded_lower]

    dict_hit_full = sum(full_hits) / max(n_tokens, 1)
    dict_hit_latin = sum(latin_only_hits) / max(n_tokens, 1)
    dict_hit_italian = sum(italian_only_hits) / max(n_tokens, 1)
    dict_hit_venetian_only = sum(venetian_only_hits) / max(n_tokens, 1)

    n_venetian_only_hits = sum(venetian_only_hits)

    print(f"     Full dict_hit: {dict_hit_full:.4f}")
    print(f"     Latin 10K hit: {dict_hit_latin:.4f}")
    print(f"     Italian 10K hit: {dict_hit_italian:.4f}")
    print(f"     Venetian-only hit: {dict_hit_venetian_only:.4f} "
          f"({n_venetian_only_hits} tokens)")

    # -- 4. Null comparison for selectivity --
    print("\n  4. Computing selectivity against null corpora ...")

    null_decoded_lists = null_data.get('null_decoded', [])
    null_full_hits: List[List[bool]] = []
    null_full_rates: List[float] = []

    if isinstance(null_decoded_lists, list) and null_decoded_lists:
        for null_decoded in null_decoded_lists[:5]:
            if isinstance(null_decoded, list):
                nd_lower = [w.lower() for w in null_decoded]
                nh = [w in full_dict for w in nd_lower]
                null_full_hits.append(nh)
                null_full_rates.append(sum(nh) / max(len(nh), 1))

    if null_full_rates:
        null_hit_rate = sum(null_full_rates) / len(null_full_rates)
    else:
        null_hit_rate = 0.0

    selectivity_full = dict_hit_full / max(null_hit_rate, 0.001)

    # Per-language null selectivity (approximate: use same null corpora)
    null_venetian_rates: List[float] = []
    if isinstance(null_decoded_lists, list) and null_decoded_lists:
        for null_decoded in null_decoded_lists[:5]:
            if isinstance(null_decoded, list):
                nd_lower = [w.lower() for w in null_decoded]
                nh_ven = sum(1 for w in nd_lower
                             if w in venetian_supplement
                             and w not in latin_10k
                             and w not in italian_10k)
                null_venetian_rates.append(nh_ven / max(len(nd_lower), 1))

    null_ven_mean = (sum(null_venetian_rates) / len(null_venetian_rates)
                     if null_venetian_rates else 0.0)
    venetian_selectivity = dict_hit_venetian_only / max(null_ven_mean, 0.001)

    print(f"     Null hit rate (full): {null_hit_rate:.4f}")
    print(f"     Selectivity (full): {selectivity_full:.2f}x")
    print(f"     Venetian selectivity: {venetian_selectivity:.2f}x")

    # -- 5. Signal isolation at full dict --
    print("\n  5. Signal isolation at full dict ...")

    if null_full_hits:
        classifications = _classify_4class(full_hits, null_full_hits)
    else:
        classifications = ['SHARED_MISS'] * n_tokens

    n_signal = sum(1 for c in classifications if c == 'SIGNAL')
    signal_rate = n_signal / max(n_tokens, 1)

    print(f"     SIGNAL tokens: {n_signal} ({signal_rate:.4f})")

    # -- 6. Venetian-specific signal words --
    print("\n  6. Venetian-specific signal words ...")

    venetian_signal_words: List[Dict] = []
    word_counts: Dict[str, int] = Counter()
    for i, cls in enumerate(classifications):
        if cls == 'SIGNAL' and venetian_only_hits[i]:
            word_counts[decoded_lower[i]] += 1

    for word, count in word_counts.most_common(30):
        venetian_signal_words.append({
            'word': word,
            'count': count,
            'in_prep_verbs': word in set(
                v['word'] for v in venetian_data.get('preparation_verbs', [])),
            'in_ingredients': word in set(
                v['word'] for v in venetian_data.get('preparation_ingredients', [])),
        })

    if venetian_signal_words:
        print(f"     {len(venetian_signal_words)} Venetian signal words:")
        for ws in venetian_signal_words[:10]:
            flags = []
            if ws['in_prep_verbs']:
                flags.append('VERB')
            if ws['in_ingredients']:
                flags.append('INGR')
            flag_str = f" [{','.join(flags)}]" if flags else ''
            print(f"       {ws['word']}: {ws['count']}{flag_str}")
    else:
        print("     No Venetian-specific signal words found")

    # -- 7. Language comparison summary --
    print("\n  7. Language comparison ...")

    comparison = {
        'latin': {
            'dict_size': len(latin_10k),
            'hit_rate': round(dict_hit_latin, 4),
        },
        'italian': {
            'dict_size': len(italian_10k),
            'hit_rate': round(dict_hit_italian, 4),
        },
        'venetian_supplement': {
            'dict_size': len(venetian_supplement),
            'hit_rate': round(dict_hit_venetian_only, 4),
            'n_hits': n_venetian_only_hits,
        },
        'full': {
            'dict_size': len(full_dict),
            'hit_rate': round(dict_hit_full, 4),
        },
    }

    for lang, data in comparison.items():
        print(f"     {lang}: {data['hit_rate']:.4f} "
              f"({data['dict_size']} words)")

    # -- 8. Save --
    elapsed = time.time() - t0

    output = {
        'dict_hit_full': round(dict_hit_full, 4),
        'n_venetian_only_hits': n_venetian_only_hits,
        'venetian_only_hit_rate': round(dict_hit_venetian_only, 4),
        'selectivity_full': round(selectivity_full, 2),
        'venetian_selectivity': round(venetian_selectivity, 2),
        'signal_rate_full': round(signal_rate, 4),
        'venetian_signal_words': venetian_signal_words,
        'comparison_by_language': comparison,
        'n_tokens': n_tokens,
        'null_hit_rate': round(null_hit_rate, 4),
        'assignment_source': assignment_source,
        'verdict': (
            f"Full dict_hit={dict_hit_full:.4f} "
            f"(selectivity {selectivity_full:.2f}x). "
            f"Venetian-only: {n_venetian_only_hits} hits "
            f"({dict_hit_venetian_only:.4f}, "
            f"selectivity {venetian_selectivity:.2f}x). "
            f"Signal rate: {signal_rate:.4f}."
        ),
        'runtime_seconds': round(elapsed, 1),
    }

    out_path = os.path.join(rd, 'venetian_decode.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(output), f, indent=2)
    print(f"\n  Saved -> {out_path} ({elapsed:.1f}s)")
