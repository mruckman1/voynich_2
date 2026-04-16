"""
Phase 86 – Self-Similar Word Analysis
=======================================
Quantitative analysis of self-similar (reduplicated) Voynich words.

The reviewer notes that words like dydydy encode the same syllable
repeated three times and that "no Latin or Italian word has this
structure."  The 10.25% rate is elevated.  Current treatment cites
reduplication but doesn't decode specific examples.

This phase:
  1. Decodes every self-similar token under TP15, including reviewer-cited
     examples (dydydy, olol)
  2. Classifies each into: (a) dictionary match, (b) consecutive-char
     artifact (ee, dd → 2-syllable repeat), (c) genuine 3+ repetition
  3. Computes the same self-similarity detection rate on Latin, Italian,
     German reference corpora for comparison
  4. Computes the rate on tachygraphic-encoded Latin for model prediction

Dependency chain:
    results/combined_refine.json  (TP15 assignment)
    corpus (IVTFF)
    data/reference/<language>/
        -> p86_self_similar_decode.json
"""

import json
import math
import os
import random
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


# ---------------------------------------------------------------------------
# JSON serialiser
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


def _safe_load(path: str) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


# ---------------------------------------------------------------------------
# Self-similarity detection (mirrors p79_known_properties logic)
# ---------------------------------------------------------------------------

def _detect_self_similar(token_types: Counter) -> List[dict]:
    """
    Detect self-similar tokens.  Returns list of dicts with:
      token, chars, count, pattern, pattern_type
    where pattern_type is 'consecutive' (XX) or 'full_repeat' (ABAB...).
    """
    results = []
    seen = set()

    for tok, count in token_types.items():
        if tok in seen:
            continue

        chars = list(tok)  # character-level for reference corpora
        if len(chars) < 2:
            continue

        # Check for XX pattern (consecutive identical chars)
        for i in range(len(chars) - 1):
            if chars[i] == chars[i + 1]:
                results.append({
                    'token': tok,
                    'count': count,
                    'pattern': f"{chars[i]}{chars[i]}",
                    'pattern_type': 'consecutive',
                })
                seen.add(tok)
                break

        if tok in seen:
            continue

        # Check for full ABAB+ pattern
        if len(chars) >= 4:
            for seg_len in range(1, len(chars) // 2 + 1):
                seg = chars[:seg_len]
                n_reps = len(chars) // seg_len
                if n_reps >= 2 and seg_len * n_reps == len(chars):
                    if all(chars[j] == seg[j % seg_len]
                           for j in range(len(chars))):
                        results.append({
                            'token': tok,
                            'count': count,
                            'pattern': f"({''.join(seg)}) × {n_reps}",
                            'pattern_type': 'full_repeat',
                        })
                        seen.add(tok)
                        break

    return results


def _detect_self_similar_eva(token_types: Counter) -> List[dict]:
    """
    Detect self-similar Voynich tokens using EVA-char tokenization.
    """
    results = []
    seen = set()

    for tok, count in token_types.items():
        if tok in seen:
            continue

        chars = tokenize_eva_chars(tok)
        if len(chars) < 2:
            continue

        # Consecutive identical EVA chars
        for i in range(len(chars) - 1):
            if chars[i] == chars[i + 1]:
                results.append({
                    'token': tok,
                    'chars': chars,
                    'count': count,
                    'pattern': f"{chars[i]}{chars[i]}",
                    'pattern_type': 'consecutive',
                })
                seen.add(tok)
                break

        if tok in seen:
            continue

        # Full ABAB+ pattern at EVA-char level
        if len(chars) >= 4:
            char_str = '|'.join(chars)
            for seg_len in range(1, len(chars) // 2 + 1):
                seg = '|'.join(chars[:seg_len])
                n_reps = len(chars) // seg_len
                if (n_reps >= 2 and seg_len * n_reps == len(chars)
                        and char_str == '|'.join([seg] * n_reps)):
                    results.append({
                        'token': tok,
                        'chars': chars,
                        'count': count,
                        'pattern': f"({'|'.join(chars[:seg_len])}) × {n_reps}",
                        'pattern_type': 'full_repeat',
                    })
                    seen.add(tok)
                    break

    return results


# ---------------------------------------------------------------------------
# Decode and classify
# ---------------------------------------------------------------------------

def _decode_token(token: str, eva_to_triple: dict, assignment: dict) -> dict:
    """Decode a single EVA token under TP15."""
    chars = tokenize_eva_chars(token)
    syllables = []
    triples = []
    for ch in chars:
        triple = eva_to_triple.get(ch)
        if triple:
            syl = assignment.get(triple, '?')
            syllables.append(syl)
            triples.append(triple)
        else:
            syllables.append('MOD')
            triples.append('modifier')

    decoded = ''.join(s for s in syllables if s != 'MOD')
    return {
        'eva': token,
        'chars': chars,
        'triples': triples,
        'syllables': syllables,
        'decoded': decoded,
    }


def _classify_self_similar(
    redupl_list: List[dict],
    eva_to_triple: dict,
    assignment: dict,
    word_set: set,
) -> List[dict]:
    """
    Classify each self-similar token:
      category_a: decoded form is in the Latin/Italian dictionary
      category_b: consecutive-char artifact (simple pair repeat)
      category_c: genuine 3+ repetition of multi-char segment
    """
    classified = []

    for r in redupl_list:
        tok = r['token']
        chars = r.get('chars', tokenize_eva_chars(tok))
        count = r['count']
        pattern_type = r['pattern_type']

        # Decode
        dec = _decode_token(tok, eva_to_triple, assignment)
        decoded = dec['decoded']

        # Classify
        if decoded.lower() in word_set:
            category = 'dictionary_match'
        elif pattern_type == 'consecutive' and len(chars) <= 4:
            category = 'consecutive_artifact'
        elif pattern_type == 'full_repeat':
            # Check repetition count
            seg_len = None
            for sl in range(1, len(chars) // 2 + 1):
                seg = chars[:sl]
                n_reps = len(chars) // sl
                if sl * n_reps == len(chars):
                    if all(chars[j] == seg[j % sl] for j in range(len(chars))):
                        seg_len = sl
                        break
            if seg_len and len(chars) // seg_len >= 3:
                category = 'triple_plus_repeat'
            else:
                category = 'double_repeat'
        else:
            # Consecutive with longer token — check if decoded is a word
            if len(decoded) <= 4 and decoded.lower() in word_set:
                category = 'dictionary_match'
            else:
                category = 'consecutive_artifact'

        classified.append({
            **r,
            'decoded': decoded,
            'syllables': dec['syllables'],
            'category': category,
        })

    return classified


# ---------------------------------------------------------------------------
# Reference corpus self-similarity rate
# ---------------------------------------------------------------------------

def _compute_ref_rate(tokens: List[str]) -> dict:
    """Compute self-similarity rate for a reference corpus token list."""
    tc = Counter(tokens)
    redupl = _detect_self_similar(tc)
    total_tokens = sum(tc.values())
    redupl_tokens = sum(r['count'] for r in redupl)
    n_consecutive = sum(1 for r in redupl if r['pattern_type'] == 'consecutive')
    n_full = sum(1 for r in redupl if r['pattern_type'] == 'full_repeat')
    return {
        'n_types': len(redupl),
        'n_tokens': redupl_tokens,
        'total_tokens': total_tokens,
        'rate': round(redupl_tokens / total_tokens, 4) if total_tokens else 0.0,
        'n_consecutive': n_consecutive,
        'n_full_repeat': n_full,
    }


# ---------------------------------------------------------------------------
# Tachygraphic simulation self-similarity
# ---------------------------------------------------------------------------

VOWELS_SET = set('aeiouàáâãäåæèéêëìíîïòóôõöùúûüyœ')

def _syllabify_simple(word: str) -> List[str]:
    syllables: List[str] = []
    current = ''
    for ch in word.lower():
        if ch not in VOWELS_SET and not ch.isalpha():
            continue
        current += ch
        if ch in VOWELS_SET:
            syllables.append(current)
            current = ''
    if current:
        if syllables:
            syllables[-1] += current
        else:
            syllables.append(current)
    return syllables if syllables else [word]


def _encode_tachygraphic(latin_tokens: List[str], seed: int = 42) -> List[str]:
    """Simple tachygraphic encoding of Latin tokens for self-similarity test."""
    rng = random.Random(seed)
    consonant_classes = [
        ['b', 'p'], ['d', 't'], ['g', 'k', 'c'],
        ['f', 'v'], ['l', 'r'], ['m', 'n'],
        ['s', 'z'],
    ]
    vowel_mods = list('aeiou')
    table: Dict[str, str] = {}
    alpha = 'abcdefghijklmnopqrstuvwxyz'
    for bi, consonants in enumerate(consonant_classes):
        base_char = alpha[bi * 2]
        for vi, vowel in enumerate(vowel_mods):
            mod_char = (alpha[bi * 2 + 1] if vi % 2 == 0
                        else alpha[20 + vi % 6] if 20 + vi % 6 < 26
                        else alpha[vi % 20])
            for consonant in consonants:
                table[consonant + vowel] = base_char + mod_char
    for vi, vowel in enumerate(vowel_mods):
        table[vowel] = alpha[14 + vi] if 14 + vi < 26 else alpha[vi]

    encoded_tokens = []
    for word in latin_tokens:
        syls = _syllabify_simple(word)
        parts = []
        for syl in syls:
            if syl in table:
                parts.append(table[syl])
            else:
                for ch in syl:
                    if ch + 'a' in table:
                        parts.append(table[ch + 'a'][:1])
                    elif ch in table:
                        parts.append(table[ch])
                    else:
                        parts.append(ch)
        encoded_tokens.append(''.join(parts))
    return encoded_tokens


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SelfSimilarResult:
    # Voynich self-similarity
    voynich_total_types: int
    voynich_total_tokens: int
    voynich_total_corpus: int
    voynich_rate: float
    # Breakdown by category
    n_dictionary_match: int
    tok_dictionary_match: int
    n_consecutive_artifact: int
    tok_consecutive_artifact: int
    n_double_repeat: int
    tok_double_repeat: int
    n_triple_plus: int
    tok_triple_plus: int
    # Reference corpus rates
    reference_rates: Dict[str, Any]
    # Tachygraphic simulation rate
    tachy_rate: Dict[str, Any]
    # Specific examples decoded
    reviewer_examples: List[Dict[str, Any]]
    top_examples: List[Dict[str, Any]]
    # Verdicts
    anomaly_fraction: float
    verdict: str
    gate_passed: bool
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_self_similar_decode() -> None:
    """Phase 86: Self-similar word quantitative analysis."""
    t0 = time.time()
    rd = str(_results_dir())

    print("=" * 60)
    print("Phase 86: Self-Similar Word Analysis")
    print("=" * 60)

    # ── 1. Load data ────────────────────────────────────────────────
    print("\n  1. Loading corpus and assignment table ...")
    corpus = load_corpus(verbose=False)
    eva_to_triple = build_eva_to_triple_lookup()
    combined = _safe_load(os.path.join(rd, 'combined_refine.json'))
    assignment = combined.get('best_assignment', {})

    # Build expanded dictionary for matching
    ref_corpus = load_reference_corpus(
        languages=['latin', 'italian', 'german'],
        verbose=False,
    )
    latin_tokens = ref_corpus.get_combined_tokens('latin')
    base_set = set(w.lower() for w in latin_tokens[:50000])
    expanded_set, _ = build_expanded_word_set(base_set)
    # Add Italian tokens
    try:
        it_tokens = ref_corpus.get_combined_tokens('italian')
        expanded_set.update(w.lower() for w in it_tokens[:30000])
    except Exception:
        pass

    print(f"    Dictionary size: {len(expanded_set):,}")

    # ── 2. Detect self-similar Voynich tokens ───────────────────────
    print("\n  2. Detecting self-similar tokens ...")
    all_tokens = []
    for _, page in corpus.pages.items():
        all_tokens.extend(page.all_tokens)

    token_types = Counter(all_tokens)
    redupl = _detect_self_similar_eva(token_types)
    total_redupl_tokens = sum(r['count'] for r in redupl)
    rate = total_redupl_tokens / len(all_tokens) if all_tokens else 0.0

    print(f"    Types: {len(redupl)}, Tokens: {total_redupl_tokens}, "
          f"Rate: {rate:.2%}")

    # ── 3. Decode and classify ──────────────────────────────────────
    print("\n  3. Decoding and classifying ...")
    classified = _classify_self_similar(
        redupl, eva_to_triple, assignment, expanded_set,
    )

    # Count by category
    cats = Counter(r['category'] for r in classified)
    tok_cats: Dict[str, int] = {}
    for cat in ['dictionary_match', 'consecutive_artifact',
                'double_repeat', 'triple_plus_repeat']:
        tok_cats[cat] = sum(r['count'] for r in classified
                            if r['category'] == cat)

    print(f"    Dictionary matches: {cats.get('dictionary_match', 0)} types / "
          f"{tok_cats.get('dictionary_match', 0)} tokens")
    print(f"    Consecutive artifacts: {cats.get('consecutive_artifact', 0)} types / "
          f"{tok_cats.get('consecutive_artifact', 0)} tokens")
    print(f"    Double repeats: {cats.get('double_repeat', 0)} types / "
          f"{tok_cats.get('double_repeat', 0)} tokens")
    print(f"    Triple+ repeats: {cats.get('triple_plus_repeat', 0)} types / "
          f"{tok_cats.get('triple_plus_repeat', 0)} tokens")

    # ── 4. Decode reviewer-cited examples ───────────────────────────
    print("\n  4. Decoding reviewer examples ...")
    reviewer_tokens = ['dydydy', 'olol', 'oror', 'ololol', 'dydydy']
    reviewer_examples = []
    for tok in reviewer_tokens:
        if tok in token_types:
            dec = _decode_token(tok, eva_to_triple, assignment)
            in_dict = dec['decoded'].lower() in expanded_set
            reviewer_examples.append({
                **dec,
                'count': token_types[tok],
                'in_dictionary': in_dict,
            })
            print(f"    {tok} ({token_types[tok]}×) → {dec['decoded']} "
                  f"({'DICT' if in_dict else 'no match'})")
        else:
            # Token might not exist; still decode
            dec = _decode_token(tok, eva_to_triple, assignment)
            reviewer_examples.append({
                **dec,
                'count': 0,
                'in_dictionary': dec['decoded'].lower() in expanded_set,
                'note': 'not found in corpus',
            })
            print(f"    {tok} (not in corpus) → {dec['decoded']}")

    # ── 5. Top examples ─────────────────────────────────────────────
    print("\n  5. Top 20 self-similar tokens (by frequency) ...")
    top20 = sorted(classified, key=lambda r: -r['count'])[:20]
    top_examples = []
    for r in top20:
        dec = _decode_token(r['token'], eva_to_triple, assignment)
        top_examples.append({
            'eva': r['token'],
            'count': r['count'],
            'decoded': dec['decoded'],
            'pattern': r['pattern'],
            'category': r['category'],
        })
        print(f"    {r['token']:>12s} ({r['count']:>4d}×) → "
              f"{dec['decoded']:<12s}  [{r['category']}]")

    # ── 6. Reference corpus rates ───────────────────────────────────
    print("\n  6. Reference corpus self-similarity rates ...")
    reference_rates = {}
    for lang in ('latin', 'italian', 'german'):
        try:
            tokens = ref_corpus.get_combined_tokens(lang)
            rr = _compute_ref_rate(tokens[:len(all_tokens)])
            reference_rates[lang] = rr
            print(f"    {lang:>8s}: {rr['rate']:.2%} "
                  f"({rr['n_types']} types, {rr['n_tokens']} tokens)")
        except Exception as e:
            print(f"    {lang}: failed — {e}")

    # ── 7. Tachygraphic simulation rate ─────────────────────────────
    print("\n  7. Tachygraphic simulation self-similarity rate ...")
    tachy_tokens = _encode_tachygraphic(latin_tokens[:len(all_tokens)])
    tachy_rate = _compute_ref_rate(tachy_tokens)
    print(f"    Tachy(Latin): {tachy_rate['rate']:.2%} "
          f"({tachy_rate['n_types']} types, {tachy_rate['n_tokens']} tokens)")

    # ── 8. Verdict ──────────────────────────────────────────────────
    anomaly_tokens = tok_cats.get('triple_plus_repeat', 0)
    anomaly_fraction = (anomaly_tokens / len(all_tokens)
                        if all_tokens else 0.0)

    # The model explains the self-similar words if:
    # - Most are consecutive-char artifacts or dictionary matches
    # - The genuine anomaly rate (3+ repetitions) is < 2%
    # - The tachygraphic simulation produces a comparable overall rate
    gate = anomaly_fraction < 0.02

    if anomaly_fraction < 0.01:
        verdict = (
            f"MOSTLY_EXPLAINED: {rate:.1%} self-similar rate decomposes into "
            f"{tok_cats.get('consecutive_artifact', 0)} consecutive-artifact tokens, "
            f"{tok_cats.get('dictionary_match', 0)} dictionary matches, "
            f"and only {anomaly_tokens} genuine anomaly tokens ({anomaly_fraction:.2%})"
        )
    elif anomaly_fraction < 0.02:
        verdict = (
            f"PARTIALLY_EXPLAINED: genuine anomaly rate {anomaly_fraction:.2%} "
            f"is low but nonzero; most self-similar tokens are "
            f"consecutive-char artifacts"
        )
    else:
        verdict = (
            f"ELEVATED_ANOMALY: {anomaly_fraction:.2%} of corpus tokens are "
            f"genuine 3+ repetitions not explained by the model"
        )

    print(f"\n  Anomaly fraction: {anomaly_fraction:.3%}")
    print(f"  Verdict: {verdict}")
    print(f"  Gate: {'PASS' if gate else 'FAIL'}")

    # ── Save ────────────────────────────────────────────────────────
    result = SelfSimilarResult(
        voynich_total_types=len(redupl),
        voynich_total_tokens=total_redupl_tokens,
        voynich_total_corpus=len(all_tokens),
        voynich_rate=round(rate, 4),
        n_dictionary_match=cats.get('dictionary_match', 0),
        tok_dictionary_match=tok_cats.get('dictionary_match', 0),
        n_consecutive_artifact=cats.get('consecutive_artifact', 0),
        tok_consecutive_artifact=tok_cats.get('consecutive_artifact', 0),
        n_double_repeat=cats.get('double_repeat', 0),
        tok_double_repeat=tok_cats.get('double_repeat', 0),
        n_triple_plus=cats.get('triple_plus_repeat', 0),
        tok_triple_plus=tok_cats.get('triple_plus_repeat', 0),
        reference_rates=reference_rates,
        tachy_rate=tachy_rate,
        reviewer_examples=reviewer_examples,
        top_examples=top_examples,
        anomaly_fraction=round(anomaly_fraction, 4),
        verdict=verdict,
        gate_passed=gate,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'p86_self_similar_decode.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)
    print(f"\n  Saved → {out_path}")
    print(f"  Runtime: {result.runtime_seconds:.1f}s")
