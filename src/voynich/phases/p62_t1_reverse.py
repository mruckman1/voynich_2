"""
Phase 62, Investigation 1: T1 Reverse Engineering Under CVC
============================================================
The 22 T1 identifications have known Latin words and known EVA tokens.
Work backward: CVC-decode each EVA token and compare to the Latin target.
Measures edit distance, character-length ratios, and syllabic-char-to-
Latin-syllable ratios to determine whether EVA tokens encode whole words
or individual syllables.

Dependency chain:
    results/word_catalog.json         (Phase 52)
    results/combined_refine.json      (Phase 15)
    results/modifier_integrate.json   (Phase 16)
        -> results/phase62_t1_reverse.json
"""

import json
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import build_eva_to_triple_lookup, tokenize_eva_chars
from voynich.core.stats import syllabify_latin
from voynich.phases.corrected_coda import (
    build_coda_table_v2,
    classify_token_chars_v2,
    decode_token_cvc_v2,
)


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
# Edit distance
# ---------------------------------------------------------------------------

def _edit_distance(a: str, b: str) -> int:
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class T1WordComparison:
    latin_word: str
    eva_type: str
    decoded_cvc: str
    decoded_cv: str
    edit_distance: int
    ed_ratio: float             # ED / max(len_decoded, len_latin)
    latin_length: int
    decoded_length: int
    n_latin_syllables: int
    n_eva_chars: int
    n_syllabic_chars: int
    n_coda_chars: int
    syllabic_to_latin_syl_ratio: float


@dataclass
class T1ReverseResult:
    phase: str = "62"
    step: str = "62.1"
    experiment: str = "t1_reverse_engineer"
    n_words: int = 0
    per_word: List[Dict] = field(default_factory=list)
    mean_ed: float = 0.0
    mean_ed_ratio: float = 0.0
    n_exact: int = 0
    n_within_1: int = 0
    n_within_2: int = 0
    length_correlation: float = 0.0
    mean_syllabic_ratio: float = 0.0
    std_syllabic_ratio: float = 0.0
    # Inferences
    tokens_are_words: bool = False
    tokens_are_syllables: bool = False
    tokens_are_mixed: bool = False
    # Gates
    g1_mean_ed: bool = False          # mean ED <= 3.0
    g2_length_corr: bool = False      # length correlation > 0.5
    g3_near_matches: bool = False     # >= 50% with ED <= 2
    gates_passed: int = 0
    gate_passed: bool = False
    verdict: str = ""
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def _load_t1_identifications(rd: str) -> List[Dict]:
    """Load T1 words from word_catalog.json."""
    catalog = _safe_load(os.path.join(rd, 'word_catalog.json'))
    t1_words = []
    for entry in catalog.get('single_token_ids', []):
        if entry.get('tier') == 'T1':
            t1_words.append({
                'eva_type': entry['eva_type'],
                'latin_word': entry['latin_word'],
                'n_folios': entry.get('n_folios', 0),
            })
    return t1_words


def run_t1_reverse():
    """Phase 62.1: Reverse-engineer T1 identifications under CVC."""
    t0 = time.time()
    rd = str(_results_dir())
    print("=" * 70)
    print("Phase 62, Investigation 1: T1 Reverse Engineering Under CVC")
    print("=" * 70)

    # Load dependencies
    eva_to_triple = build_eva_to_triple_lookup()
    refine = _safe_load(os.path.join(rd, 'combined_refine.json'))
    assignment = refine.get('best_assignment', {})
    coda_table = build_coda_table_v2()

    t1_words = _load_t1_identifications(rd)
    if not t1_words:
        print("  WARNING: No T1 identifications found in word_catalog.json")
        print("  Falling back to hardcoded set")
        t1_words = [
            {'eva_type': 'otol', 'latin_word': 'ratione', 'n_folios': 5},
            {'eva_type': 'qopchedy', 'latin_word': 'coralli', 'n_folios': 3},
            {'eva_type': 'qokain', 'latin_word': 'diasene', 'n_folios': 2},
            {'eva_type': 'opchedy', 'latin_word': 'stercora', 'n_folios': 3},
            {'eva_type': 'otol', 'latin_word': 'radicom', 'n_folios': 2},
            {'eva_type': 'chedy', 'latin_word': 'commune', 'n_folios': 4},
            {'eva_type': 'shedy', 'latin_word': 'secundi', 'n_folios': 2},
            {'eva_type': 'qokeedy', 'latin_word': 'balsamo', 'n_folios': 2},
            {'eva_type': 'dchedy', 'latin_word': 'decoctum', 'n_folios': 3},
        ]

    print(f"\n  T1 identifications loaded: {len(t1_words)}")

    # Decode each T1 word
    comparisons = []
    for ident in t1_words:
        eva_type = ident['eva_type']
        latin_word = ident['latin_word']

        # CVC decode
        result = decode_token_cvc_v2(eva_type, assignment, eva_to_triple, coda_table)
        decoded_cvc = result.decoded_cvc
        decoded_cv = result.decoded_cv

        # Edit distance
        ed = _edit_distance(decoded_cvc.lower(), latin_word.lower())
        max_len = max(len(decoded_cvc), len(latin_word), 1)
        ed_ratio = ed / max_len

        # Latin syllable count
        latin_syls = syllabify_latin(latin_word)
        n_latin_syls = len(latin_syls) if latin_syls else 1

        # EVA character analysis
        eva_chars = tokenize_eva_chars(eva_type)
        n_eva = len(eva_chars)
        classified = classify_token_chars_v2(eva_chars, coda_table)
        n_syllabic = sum(1 for role, _ in classified if role == 'SYLLABIC')
        n_coda = sum(1 for role, _ in classified if role == 'CODA_MARKER')

        syl_ratio = n_syllabic / n_latin_syls if n_latin_syls > 0 else 0.0

        comp = T1WordComparison(
            latin_word=latin_word,
            eva_type=eva_type,
            decoded_cvc=decoded_cvc,
            decoded_cv=decoded_cv,
            edit_distance=ed,
            ed_ratio=ed_ratio,
            latin_length=len(latin_word),
            decoded_length=len(decoded_cvc),
            n_latin_syllables=n_latin_syls,
            n_eva_chars=n_eva,
            n_syllabic_chars=n_syllabic,
            n_coda_chars=n_coda,
            syllabic_to_latin_syl_ratio=syl_ratio,
        )
        comparisons.append(comp)

        print(f"    {eva_type:15s} -> CVC: {decoded_cvc:12s}  "
              f"Latin: {latin_word:12s}  ED={ed}  ratio={ed_ratio:.2f}  "
              f"syls={n_syllabic}/{n_latin_syls}")

    # Aggregate metrics
    eds = [c.edit_distance for c in comparisons]
    ed_ratios = [c.ed_ratio for c in comparisons]
    syl_ratios = [c.syllabic_to_latin_syl_ratio for c in comparisons]
    latin_lens = [c.latin_length for c in comparisons]
    decoded_lens = [c.decoded_length for c in comparisons]

    mean_ed = float(np.mean(eds))
    mean_ed_ratio = float(np.mean(ed_ratios))
    n_exact = sum(1 for ed in eds if ed == 0)
    n_within_1 = sum(1 for ed in eds if ed <= 1)
    n_within_2 = sum(1 for ed in eds if ed <= 2)

    # Length correlation
    if len(latin_lens) >= 3:
        length_corr = float(np.corrcoef(latin_lens, decoded_lens)[0, 1])
        if np.isnan(length_corr):
            length_corr = 0.0
    else:
        length_corr = 0.0

    mean_syl_ratio = float(np.mean(syl_ratios))
    std_syl_ratio = float(np.std(syl_ratios))

    # Inferences
    tokens_are_words = mean_ed <= 2.0
    tokens_are_syllables = mean_syl_ratio < 0.5
    tokens_are_mixed = (0.5 <= mean_syl_ratio <= 1.5) and mean_ed > 2.0

    # Gates
    g1 = mean_ed <= 3.0
    g2 = length_corr > 0.5
    n_total = len(comparisons)
    g3 = (n_within_2 / n_total >= 0.5) if n_total > 0 else False
    gates_passed = sum([g1, g2, g3])

    # Verdict
    if tokens_are_words:
        verdict = "TOKENS_APPROXIMATE_WORDS"
    elif tokens_are_syllables:
        verdict = "TOKENS_ARE_SYLLABLES"
    elif tokens_are_mixed:
        verdict = "MIXED_ENCODING"
    else:
        verdict = "INCONCLUSIVE"

    result = T1ReverseResult(
        n_words=n_total,
        per_word=[_convert(asdict(c)) for c in comparisons],
        mean_ed=mean_ed,
        mean_ed_ratio=mean_ed_ratio,
        n_exact=n_exact,
        n_within_1=n_within_1,
        n_within_2=n_within_2,
        length_correlation=length_corr,
        mean_syllabic_ratio=mean_syl_ratio,
        std_syllabic_ratio=std_syl_ratio,
        tokens_are_words=tokens_are_words,
        tokens_are_syllables=tokens_are_syllables,
        tokens_are_mixed=tokens_are_mixed,
        g1_mean_ed=g1,
        g2_length_corr=g2,
        g3_near_matches=g3,
        gates_passed=gates_passed,
        gate_passed=gates_passed >= 2,
        verdict=verdict,
        runtime_seconds=time.time() - t0,
    )

    # Summary
    print(f"\n  Mean ED: {mean_ed:.2f}  Mean ED ratio: {mean_ed_ratio:.2f}")
    print(f"  Exact: {n_exact}  Within-1: {n_within_1}  Within-2: {n_within_2}")
    print(f"  Length correlation: {length_corr:.3f}")
    print(f"  Mean syllabic ratio: {mean_syl_ratio:.2f} ± {std_syl_ratio:.2f}")
    print(f"  Gates: G1={'PASS' if g1 else 'FAIL'} G2={'PASS' if g2 else 'FAIL'} "
          f"G3={'PASS' if g3 else 'FAIL'} ({gates_passed}/3)")
    print(f"  Verdict: {verdict}")

    path = _save_json(rd, 'phase62_t1_reverse.json', asdict(result))
    print(f"\n  Saved: {path}")
    print(f"  Runtime: {result.runtime_seconds:.1f}s")
    return result
