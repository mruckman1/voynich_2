"""
Phase C.4 -- Modifier-Clean Subset Test
=========================================
Decode only tokens where ALL EVA characters are classified as syllabic
(no modifiers, no ambiguous characters).  This gives a cleaner signal
because modifier-contaminated tokens introduce noise from uncertain
stripping/alteration heuristics.

Comparing the clean-subset dict_hit to the full-corpus dict_hit
reveals how much of the decoding quality comes from syllabic core
characters versus modifier handling.

Dependency chain:
    results/tironian_csp.json       (Phase C.1-C.2)
    results/modifier_integrate.json (Phase 16.6 -- modifier classification)
        -> modifier_clean.json (this step)
"""

import json
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    load_corpus,
    token_to_triples,
    tokenize_eva_chars,
)
from voynich.core.reference import (
    build_expanded_word_set,
    load_reference_corpus,
)
from voynich.phases.csp_solver import decode_corpus, decode_token


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
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ModifierCleanResult:
    """Phase C.4: modifier-clean subset test."""
    n_total_tokens: int
    n_clean_tokens: int
    clean_fraction: float
    clean_dict_hit: float
    full_dict_hit: float
    n_phrases_clean: int
    clean_decoded_sample: List[List[str]]
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Phrase scanning (lightweight copy for clean subset)
# ---------------------------------------------------------------------------

_IMPERATIVES = {
    'recipe', 'accipe', 'misce', 'cola', 'tera', 'solve',
    'distilla', 'lava', 'coque', 'adde', 'tere', 'cape',
    'pone', 'da', 'fac', 'applica', 'bibe',
}
_PREPOSITIONS = {
    'cum', 'in', 'ad', 'de', 'per', 'pro', 'sub',
    'super', 'ante', 'ex', 'ab', 'sine',
}
_ADJECTIVES = {
    'calidus', 'calida', 'calidum', 'frigidus', 'frigida',
    'siccus', 'sicca', 'humidus', 'humida',
    'niger', 'nigra', 'albus', 'alba', 'dulcis', 'dulce',
}
_NOUNS = {
    'aqua', 'radix', 'folia', 'flos', 'semen', 'cortex',
    'vinum', 'oleum', 'herba', 'pulvis', 'mel', 'acetum',
    'succus', 'rosa', 'viola', 'salvia', 'menta', 'ruta',
}
_GENITIVE_NOUNS = {
    'viole', 'rose', 'salvie', 'vini', 'olei', 'mellis',
    'radicis', 'floris', 'seminis', 'corticis', 'aque', 'herbe',
}
_CONJUNCTIONS = {'et', 'ac', 'atque', 'vel', 'aut', 'nec', 'sed'}


def _count_phrases(decoded_tokens: List[str]) -> int:
    """Count multi-word Latin phrase patterns in decoded output."""
    n = 0
    for i in range(len(decoded_tokens) - 1):
        w1 = decoded_tokens[i].lower()
        w2 = decoded_tokens[i + 1].lower()

        if w1 in _IMPERATIVES and w2 in _NOUNS:
            n += 1
        elif w1 in _PREPOSITIONS and w2 in _NOUNS:
            n += 1
        elif (w1 in _ADJECTIVES and w2 in _NOUNS) or \
             (w1 in _NOUNS and w2 in _ADJECTIVES):
            n += 1
        elif w1 in _NOUNS and w2 in _GENITIVE_NOUNS:
            n += 1
        elif w1 in _CONJUNCTIONS and (
            w2 in _NOUNS or w2 in _ADJECTIVES or w2 in _IMPERATIVES
            or w2 in _PREPOSITIONS
        ):
            n += 1
    return n


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_modifier_clean() -> None:
    """Phase C.4: Modifier-clean subset test."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE C.4: Modifier-Clean Subset Test")
    print("=" * 70)

    rd = _results_dir()

    # ------------------------------------------------------------------ 1
    print("\n  1. Loading tironian_csp.json for best assignment ...")
    tir_path = os.path.join(rd, 'tironian_csp.json')
    if not os.path.exists(tir_path):
        print("    [SKIP] tironian_csp.json not found -- run tironian-csp first")
        return

    with open(tir_path) as f:
        tir_data = json.load(f)
    best_assignment = tir_data.get('best_assignment', {})
    full_dict_hit = tir_data.get('best_dict_hit', 0.0)
    if not best_assignment:
        print("    [SKIP] No assignment found in tironian_csp.json")
        return
    print(f"    Loaded assignment ({len(best_assignment)} mappings)")
    print(f"    Full-corpus dict_hit: {full_dict_hit:.4f}")

    # ------------------------------------------------------------------ 2
    print("\n  2. Loading modifier classification from modifier_integrate.json ...")
    mod_path = os.path.join(rd, 'modifier_integrate.json')
    if not os.path.exists(mod_path):
        print("    [SKIP] modifier_integrate.json not found -- run modifier-integrate first")
        return

    with open(mod_path) as f:
        mod_data = json.load(f)

    modifier_chars = set(mod_data.get('modifier_chars', []))
    syllabic_chars = set(mod_data.get('syllabic_chars', []))
    ambiguous_chars = set(mod_data.get('ambiguous_chars', []))

    print(f"    Modifier chars  ({len(modifier_chars)}): {sorted(modifier_chars)}")
    print(f"    Syllabic chars  ({len(syllabic_chars)}): {sorted(syllabic_chars)}")
    print(f"    Ambiguous chars ({len(ambiguous_chars)}): {sorted(ambiguous_chars)}")

    # ------------------------------------------------------------------ 3
    print("\n  3. Loading corpus and filtering to modifier-clean tokens ...")
    corpus = load_corpus(verbose=False)
    tokens = corpus.get_tokens(language='A', paragraph_only=True)
    if not tokens:
        print("    [SKIP] No Language A tokens found")
        return

    eva_to_triple = build_eva_to_triple_lookup()

    # Filter: keep tokens where ALL EVA chars are syllabic
    clean_tokens: List[str] = []
    for token in tokens:
        chars = tokenize_eva_chars(token)
        if all(ch in syllabic_chars for ch in chars):
            clean_tokens.append(token)

    n_total = len(tokens)
    n_clean = len(clean_tokens)
    clean_fraction = n_clean / n_total if n_total > 0 else 0.0

    # ------------------------------------------------------------------ 4
    print(f"\n  4. Filter results:")
    print(f"    Total tokens:   {n_total}")
    print(f"    Clean tokens:   {n_clean}")
    print(f"    Clean fraction: {clean_fraction:.1%}")

    # ------------------------------------------------------------------ 5
    print("\n  5. Decoding modifier-clean subset ...")
    clean_decoded = decode_corpus(
        clean_tokens, best_assignment, eva_to_triple, max_tokens=len(clean_tokens)
    )

    # Build reference word set (original, non-expanded)
    ref_corpus = load_reference_corpus(verbose=False)
    ref_tokens = ref_corpus.get_combined_tokens('latin')
    ref_word_set = set(w.lower() for w in ref_tokens if len(w) >= 2)

    # ------------------------------------------------------------------ 6
    print("\n  6. Computing dict_hit on clean subset ...")
    clean_hits = sum(1 for w in clean_decoded if w in ref_word_set)
    clean_dict_hit = clean_hits / len(clean_decoded) if clean_decoded else 0.0
    print(f"    Clean dict_hit: {clean_dict_hit:.4f} ({clean_hits}/{len(clean_decoded)})")
    print(f"    Full  dict_hit: {full_dict_hit:.4f}")
    delta = clean_dict_hit - full_dict_hit
    print(f"    Delta:          {delta:+.4f}")

    # ------------------------------------------------------------------ 7
    print("\n  7. Phrase detection on clean subset ...")
    n_phrases_clean = _count_phrases(clean_decoded)
    print(f"    Phrases in clean subset: {n_phrases_clean}")

    # ------------------------------------------------------------------ 8
    print("\n  8. Comparison with full-corpus results:")
    print(f"    {'Metric':<25} {'Full corpus':>12} {'Clean subset':>14}")
    print(f"    {'-' * 25} {'-' * 12} {'-' * 14}")
    print(f"    {'dict_hit':<25} {full_dict_hit:>12.4f} {clean_dict_hit:>14.4f}")
    print(f"    {'n_tokens':<25} {n_total:>12d} {n_clean:>14d}")
    print(f"    {'n_phrases':<25} {'---':>12} {n_phrases_clean:>14d}")

    # Decoded sample (first 20 clean tokens)
    clean_decoded_sample: List[List[str]] = []
    for i in range(min(20, len(clean_tokens), len(clean_decoded))):
        clean_decoded_sample.append([clean_tokens[i], clean_decoded[i]])

    print(f"\n    First 20 clean decoded tokens:")
    for tok, dec in clean_decoded_sample:
        marker = '*' if dec in ref_word_set else ' '
        print(f"    {marker} {tok:>15} -> {dec}")

    # Verdict
    if clean_dict_hit >= full_dict_hit:
        verdict = (
            f"Clean subset ({clean_fraction:.1%} of tokens) achieves "
            f"{clean_dict_hit:.1%} dict_hit >= full corpus {full_dict_hit:.1%}. "
            f"Modifier handling is not degrading decode quality. "
            f"{n_phrases_clean} phrases detected in clean subset."
        )
    elif clean_dict_hit >= full_dict_hit * 0.8:
        verdict = (
            f"Clean subset ({clean_fraction:.1%} of tokens) achieves "
            f"{clean_dict_hit:.1%} dict_hit, close to full corpus {full_dict_hit:.1%}. "
            f"Modifier removal causes minor degradation ({delta:+.4f}). "
            f"{n_phrases_clean} phrases in clean subset."
        )
    else:
        verdict = (
            f"Clean subset ({clean_fraction:.1%} of tokens) achieves only "
            f"{clean_dict_hit:.1%} dict_hit vs full corpus {full_dict_hit:.1%}. "
            f"Syllabic chars alone carry less signal than expected. "
            f"Modifier-handled tokens may be contributing genuine decodes."
        )

    print(f"\n  Verdict: {verdict}")

    # Save
    result = ModifierCleanResult(
        n_total_tokens=n_total,
        n_clean_tokens=n_clean,
        clean_fraction=round(clean_fraction, 4),
        clean_dict_hit=round(clean_dict_hit, 4),
        full_dict_hit=round(full_dict_hit, 4),
        n_phrases_clean=n_phrases_clean,
        clean_decoded_sample=clean_decoded_sample,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'modifier_clean.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)

    print(f"\n  Results saved -> {out_path}")
