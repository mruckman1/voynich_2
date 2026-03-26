"""
Phase 62, Investigation 3: Gallows as Word-Initial Markers
===========================================================
Test whether gallows characters (k, t, p, f) mark the beginning of
Latin words.  Concatenate gallows-initial tokens with the next 1-3
tokens and check dictionary hit rates.

Dependency chain:
    results/combined_refine.json      (Phase 15)
    results/modifier_integrate.json   (Phase 16)
        -> results/phase62_gallows_initial.json
"""

import json
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import build_eva_to_triple_lookup, load_corpus, tokenize_eva_chars
from voynich.core.reference import build_expanded_word_set, load_reference_corpus
from voynich.phases.corrected_coda import (
    build_coda_table_v2,
    decode_token_cvc_v2,
    decode_corpus_cvc_v2,
    LATIN_ENDINGS,
)
from voynich.phases.coda_markers import SIMPLE_GALLOWS


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
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class GallowsInitialResult:
    phase: str = "62"
    step: str = "62.3"
    experiment: str = "gallows_initial"
    n_gallows_tokens: int = 0
    n_non_gallows: int = 0
    # Concat hit rates by lookahead
    gallows_single_hit_rate: float = 0.0
    gallows_concat_1_hit_rate: float = 0.0
    gallows_concat_2_hit_rate: float = 0.0
    gallows_concat_3_hit_rate: float = 0.0
    baseline_single_hit_rate: float = 0.0
    baseline_concat_1_hit_rate: float = 0.0
    baseline_concat_2_hit_rate: float = 0.0
    baseline_concat_3_hit_rate: float = 0.0
    best_concat_ratio: float = 0.0
    # Word-final predecessor analysis
    n_with_wordfinal_pred: int = 0
    wordfinal_pred_rate: float = 0.0
    # Per-gallows breakdown
    per_gallows: Dict[str, Dict] = field(default_factory=dict)
    # Example hits
    example_concat_hits: List[Dict] = field(default_factory=list)
    # Gates
    g1_concat_rate: bool = False       # concat rate > 1.5x baseline
    g2_wordfinal: bool = False         # > 30% have word-final predecessor
    g3_sigla: bool = False             # >= 1 gallows where sigla helps
    gates_passed: int = 0
    gate_passed: bool = False
    verdict: str = ""
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

WORD_FINAL_SUFFIXES = ('us', 'um', 'em', 'is', 'en', 'on', 'er', 'or', 'es', 'am', 'ar')


def run_gallows_initial():
    """Phase 62.3: Gallows as word-initial markers."""
    t0 = time.time()
    rd = str(_results_dir())
    print("=" * 70)
    print("Phase 62, Investigation 3: Gallows as Word-Initial Markers")
    print("=" * 70)

    # Load
    eva_to_triple = build_eva_to_triple_lookup()
    refine = _safe_load(os.path.join(rd, 'combined_refine.json'))
    assignment = refine.get('best_assignment', {})
    coda_table = build_coda_table_v2()

    # Load dictionary
    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    base_words = set(w.lower() for w in ref_corpus.get_combined_tokens('latin') if len(w) >= 2)
    expanded, _ = build_expanded_word_set(base_words)
    ref_word_set = base_words | expanded

    corpus = load_corpus(verbose=False)
    all_tokens = corpus.get_tokens()
    decoded = decode_corpus_cvc_v2(all_tokens, assignment, eva_to_triple, coda_table)

    print(f"  Tokens: {len(all_tokens)}  Dictionary: {len(ref_word_set)}")

    # Identify gallows-initial tokens
    gallows_indices = []
    non_gallows_indices = []
    for idx, token in enumerate(all_tokens):
        chars = tokenize_eva_chars(token)
        if chars and chars[0] in SIMPLE_GALLOWS:
            gallows_indices.append(idx)
        else:
            non_gallows_indices.append(idx)

    print(f"  Gallows-initial: {len(gallows_indices)}  Non-gallows: {len(non_gallows_indices)}")

    # Test concatenation hit rates
    def _concat_hit_rates(indices, max_ahead=3):
        """For each token at idx, concatenate decoded[idx:idx+ahead+1] and check dict."""
        rates = {0: 0, 1: 0, 2: 0, 3: 0}
        counts = {0: 0, 1: 0, 2: 0, 3: 0}
        examples = []

        for idx in indices:
            d = decoded[idx]
            if not d or d == '?':
                continue

            # Single token
            counts[0] += 1
            if d in ref_word_set:
                rates[0] += 1

            # Concat with next 1, 2, 3
            for ahead in range(1, max_ahead + 1):
                if idx + ahead >= len(decoded):
                    break
                concat = d
                all_valid = True
                for j in range(1, ahead + 1):
                    nxt = decoded[idx + j]
                    if not nxt or nxt == '?':
                        all_valid = False
                        break
                    concat += nxt

                if not all_valid:
                    continue

                counts[ahead] += 1
                if concat in ref_word_set:
                    rates[ahead] += 1
                    if len(examples) < 20 and ahead >= 1:
                        examples.append({
                            'concat': concat,
                            'tokens': [all_tokens[idx + j] for j in range(ahead + 1)],
                            'decoded_parts': [decoded[idx + j] for j in range(ahead + 1)],
                            'n_tokens': ahead + 1,
                        })

        hit_rates = {}
        for k in range(max_ahead + 1):
            hit_rates[k] = rates[k] / counts[k] if counts[k] > 0 else 0.0
        return hit_rates, examples

    gallows_rates, gallows_examples = _concat_hit_rates(gallows_indices)
    baseline_rates, _ = _concat_hit_rates(non_gallows_indices)

    # Best concat ratio
    concat_ratios = []
    for ahead in range(1, 4):
        if baseline_rates[ahead] > 0:
            concat_ratios.append(gallows_rates[ahead] / baseline_rates[ahead])
    best_ratio = max(concat_ratios) if concat_ratios else 0.0

    # Word-final predecessor analysis
    n_wordfinal = 0
    n_checked = 0
    for idx in gallows_indices:
        if idx == 0:
            continue
        prev = decoded[idx - 1]
        if not prev or prev == '?':
            continue
        n_checked += 1
        if any(prev.endswith(suf) for suf in WORD_FINAL_SUFFIXES):
            n_wordfinal += 1
    wordfinal_rate = n_wordfinal / n_checked if n_checked > 0 else 0.0

    # Per-gallows char breakdown
    per_gallows = {}
    for g_char in sorted(SIMPLE_GALLOWS):
        g_indices = [idx for idx in gallows_indices
                     if tokenize_eva_chars(all_tokens[idx])[0] == g_char]
        if not g_indices:
            continue
        g_rates, _ = _concat_hit_rates(g_indices)
        per_gallows[g_char] = {
            'count': len(g_indices),
            'single_hit_rate': round(g_rates[0], 4),
            'concat_1_rate': round(g_rates[1], 4),
            'concat_2_rate': round(g_rates[2], 4),
        }

    # Gates
    g1 = best_ratio > 1.5
    g2 = wordfinal_rate > 0.30
    # G3: sigla test (simplified) — check if any single gallows char decoding
    # is a common function word
    sigla_found = False
    for g_char in SIMPLE_GALLOWS:
        res = decode_token_cvc_v2(g_char, assignment, eva_to_triple, coda_table)
        d = res.decoded_cvc
        if d in {'de', 'et', 'te', 'be', 'per', 'cum', 'con'}:
            sigla_found = True
            break
    g3 = sigla_found
    gates_passed = sum([g1, g2, g3])

    if gates_passed >= 2:
        verdict = "GALLOWS_MARK_WORDS"
    elif gates_passed == 1:
        verdict = "WEAK_EVIDENCE"
    else:
        verdict = "NO_WORD_MARKING"

    result = GallowsInitialResult(
        n_gallows_tokens=len(gallows_indices),
        n_non_gallows=len(non_gallows_indices),
        gallows_single_hit_rate=round(gallows_rates[0], 4),
        gallows_concat_1_hit_rate=round(gallows_rates[1], 4),
        gallows_concat_2_hit_rate=round(gallows_rates[2], 4),
        gallows_concat_3_hit_rate=round(gallows_rates[3], 4),
        baseline_single_hit_rate=round(baseline_rates[0], 4),
        baseline_concat_1_hit_rate=round(baseline_rates[1], 4),
        baseline_concat_2_hit_rate=round(baseline_rates[2], 4),
        baseline_concat_3_hit_rate=round(baseline_rates[3], 4),
        best_concat_ratio=round(best_ratio, 3),
        n_with_wordfinal_pred=n_wordfinal,
        wordfinal_pred_rate=round(wordfinal_rate, 4),
        per_gallows=per_gallows,
        example_concat_hits=gallows_examples[:15],
        g1_concat_rate=g1,
        g2_wordfinal=g2,
        g3_sigla=g3,
        gates_passed=gates_passed,
        gate_passed=gates_passed >= 2,
        verdict=verdict,
        runtime_seconds=time.time() - t0,
    )

    # Print summary
    print(f"\n  Hit rates (single / +1 / +2 / +3 tokens):")
    print(f"    Gallows:  {gallows_rates[0]:.1%} / {gallows_rates[1]:.1%} / "
          f"{gallows_rates[2]:.1%} / {gallows_rates[3]:.1%}")
    print(f"    Baseline: {baseline_rates[0]:.1%} / {baseline_rates[1]:.1%} / "
          f"{baseline_rates[2]:.1%} / {baseline_rates[3]:.1%}")
    print(f"  Best concat ratio (gallows/baseline): {best_ratio:.2f}")
    print(f"  Word-final predecessor rate: {wordfinal_rate:.1%} ({n_wordfinal}/{n_checked})")
    print(f"  Sigla found: {sigla_found}")
    if gallows_examples:
        print(f"  Example concat hits:")
        for ex in gallows_examples[:5]:
            print(f"    {ex['concat']}  ({' + '.join(ex['decoded_parts'])})")
    print(f"\n  Gates: G1={'PASS' if g1 else 'FAIL'} G2={'PASS' if g2 else 'FAIL'} "
          f"G3={'PASS' if g3 else 'FAIL'} ({gates_passed}/3)")
    print(f"  Verdict: {verdict}")

    path = _save_json(rd, 'phase62_gallows_initial.json', asdict(result))
    print(f"\n  Saved: {path}")
    print(f"  Runtime: {result.runtime_seconds:.1f}s")
    return result
