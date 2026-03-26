"""
Phase 62, Investigation 9: Language A / Language B Under CVC
=============================================================
Split the CVC-decoded corpus by Currier language (A vs B).  Compare
vocabulary overlap, coda distribution, and signal word distribution.

Dependency chain:
    results/combined_refine.json      (Phase 15)
    results/modifier_integrate.json   (Phase 16)
        -> results/phase62_lang_ab_cvc.json
"""

import json
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Set, Tuple

import numpy as np
from scipy import stats as sp_stats

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    load_corpus,
    _infer_language,
)
from voynich.core.reference import build_expanded_word_set, load_reference_corpus
from voynich.phases.corrected_coda import (
    build_coda_table_v2,
    classify_token_chars_v2,
    decode_corpus_cvc_v2,
)
from voynich.phases.coda_markers import get_coda
from voynich.phases.suffix_calibration import SIGNAL_WORDS_SET
from voynich.core.corpus import tokenize_eva_chars


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
class LangSubsetStats:
    lang: str
    n_tokens: int = 0
    n_types: int = 0
    dict_hit_rate: float = 0.0
    signal_rate: float = 0.0
    n_signal_words: int = 0
    signal_words_exclusive: List[str] = field(default_factory=list)
    coda_distribution: Dict[str, int] = field(default_factory=dict)
    top_words: List[List] = field(default_factory=list)
    mean_decoded_length: float = 0.0


@dataclass
class LangABResult:
    phase: str = "62"
    step: str = "62.9"
    experiment: str = "lang_ab_cvc"
    lang_a: Dict = field(default_factory=dict)
    lang_b: Dict = field(default_factory=dict)
    eva_vocab_overlap: float = 0.138    # Phase 4 finding
    cvc_vocab_overlap: float = 0.0
    overlap_change: float = 0.0
    coda_chi2: float = 0.0
    coda_p: float = 1.0
    signal_chi2: float = 0.0
    signal_p: float = 1.0
    n_exclusive_a: int = 0
    n_exclusive_b: int = 0
    # Gates
    g1_overlap_differs: bool = False    # CVC overlap differs from EVA by >5pp
    g2_coda_differs: bool = False       # coda chi² p < 0.05
    g3_signal_exclusive: bool = False   # >= 5 signal words exclusive to one subsystem
    gates_passed: int = 0
    gate_passed: bool = False
    verdict: str = ""
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def _count_codas(tokens: List[str], eva_tokens: List[str], coda_table) -> Dict[str, int]:
    """Count coda consonant distribution for a subset."""
    coda_counts = Counter()
    for eva_tok in eva_tokens:
        chars = tokenize_eva_chars(eva_tok)
        classified = classify_token_chars_v2(chars, coda_table)
        for role, char in classified:
            if role == 'CODA_MARKER':
                coda = get_coda(char, coda_table)
                if coda:
                    coda_counts[coda] += 1
    return dict(coda_counts.most_common())


def run_lang_ab_cvc():
    """Phase 62.9: Language A/B under CVC."""
    t0 = time.time()
    rd = str(_results_dir())
    print("=" * 70)
    print("Phase 62, Investigation 9: Language A/B Under CVC")
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

    # Split by language
    a_eva_tokens = []
    a_decoded = []
    b_eva_tokens = []
    b_decoded = []

    for folio_id, page in corpus.pages.items():
        lang = _infer_language(folio_id)
        tokens = page.all_tokens
        dec = decode_corpus_cvc_v2(tokens, assignment, eva_to_triple, coda_table)
        if lang == 'A':
            a_eva_tokens.extend(tokens)
            a_decoded.extend(dec)
        else:
            b_eva_tokens.extend(tokens)
            b_decoded.extend(dec)

    print(f"  Language A: {len(a_decoded)} tokens")
    print(f"  Language B: {len(b_decoded)} tokens")

    # Vocabulary overlap (Jaccard)
    a_vocab = set(d for d in a_decoded if d and d != '?')
    b_vocab = set(d for d in b_decoded if d and d != '?')
    union = a_vocab | b_vocab
    intersection = a_vocab & b_vocab
    cvc_overlap = len(intersection) / len(union) if union else 0.0
    overlap_change = cvc_overlap - 0.138

    # Dict hit
    a_valid = [d for d in a_decoded if d and d != '?']
    b_valid = [d for d in b_decoded if d and d != '?']
    a_dict_hit = sum(1 for d in a_valid if d in ref_word_set) / len(a_valid) if a_valid else 0.0
    b_dict_hit = sum(1 for d in b_valid if d in ref_word_set) / len(b_valid) if b_valid else 0.0

    # Signal words
    a_signals = set(d for d in a_valid if d in SIGNAL_WORDS_SET)
    b_signals = set(d for d in b_valid if d in SIGNAL_WORDS_SET)
    a_exclusive = a_signals - b_signals
    b_exclusive = b_signals - a_signals

    # Coda distributions
    a_codas = _count_codas(a_decoded, a_eva_tokens, coda_table)
    b_codas = _count_codas(b_decoded, b_eva_tokens, coda_table)

    # Chi-squared on codas
    all_codas_keys = sorted(set(list(a_codas.keys()) + list(b_codas.keys())))
    if len(all_codas_keys) >= 2:
        a_counts = [a_codas.get(k, 0) for k in all_codas_keys]
        b_counts = [b_codas.get(k, 0) for k in all_codas_keys]
        contingency = np.array([a_counts, b_counts])
        chi2, coda_p = sp_stats.chi2_contingency(contingency)[:2]
    else:
        chi2, coda_p = 0.0, 1.0

    # Signal distribution chi² (per-signal-word contingency)
    all_signal_keys = sorted(a_signals | b_signals)
    if len(all_signal_keys) >= 2:
        a_sig_counts = [Counter(d for d in a_valid if d in SIGNAL_WORDS_SET).get(k, 0)
                        for k in all_signal_keys]
        b_sig_counts = [Counter(d for d in b_valid if d in SIGNAL_WORDS_SET).get(k, 0)
                        for k in all_signal_keys]
        sig_contingency = np.array([a_sig_counts, b_sig_counts])
        # Remove zero columns
        col_sums = sig_contingency.sum(axis=0)
        sig_contingency = sig_contingency[:, col_sums > 0]
        if sig_contingency.shape[1] >= 2:
            signal_chi2, signal_p = sp_stats.chi2_contingency(sig_contingency)[:2]
        else:
            signal_chi2, signal_p = 0.0, 1.0
    else:
        signal_chi2, signal_p = 0.0, 1.0

    # Build subset stats
    a_stats = LangSubsetStats(
        lang='A',
        n_tokens=len(a_valid),
        n_types=len(a_vocab),
        dict_hit_rate=round(a_dict_hit, 4),
        signal_rate=round(len([d for d in a_valid if d in SIGNAL_WORDS_SET]) / len(a_valid), 4) if a_valid else 0.0,
        n_signal_words=len(a_signals),
        signal_words_exclusive=sorted(a_exclusive),
        coda_distribution=a_codas,
        top_words=Counter(a_valid).most_common(15),
        mean_decoded_length=round(float(np.mean([len(d) for d in a_valid])), 2) if a_valid else 0.0,
    )
    b_stats = LangSubsetStats(
        lang='B',
        n_tokens=len(b_valid),
        n_types=len(b_vocab),
        dict_hit_rate=round(b_dict_hit, 4),
        signal_rate=round(len([d for d in b_valid if d in SIGNAL_WORDS_SET]) / len(b_valid), 4) if b_valid else 0.0,
        n_signal_words=len(b_signals),
        signal_words_exclusive=sorted(b_exclusive),
        coda_distribution=b_codas,
        top_words=Counter(b_valid).most_common(15),
        mean_decoded_length=round(float(np.mean([len(d) for d in b_valid])), 2) if b_valid else 0.0,
    )

    # Gates
    g1 = abs(overlap_change) > 0.05
    g2 = coda_p < 0.05
    g3 = len(a_exclusive) + len(b_exclusive) >= 5
    gates_passed = sum([g1, g2, g3])

    if gates_passed >= 2:
        verdict = "AB_DIVERGENT_UNDER_CVC"
    elif gates_passed == 1:
        verdict = "AB_MINOR_DIFFERENCES"
    else:
        verdict = "AB_SIMILAR_UNDER_CVC"

    result = LangABResult(
        lang_a=_convert(asdict(a_stats)),
        lang_b=_convert(asdict(b_stats)),
        cvc_vocab_overlap=round(cvc_overlap, 4),
        overlap_change=round(overlap_change, 4),
        coda_chi2=round(float(chi2), 2),
        coda_p=round(float(coda_p), 6),
        signal_chi2=round(float(signal_chi2), 2),
        signal_p=round(float(signal_p), 6),
        n_exclusive_a=len(a_exclusive),
        n_exclusive_b=len(b_exclusive),
        g1_overlap_differs=g1,
        g2_coda_differs=g2,
        g3_signal_exclusive=g3,
        gates_passed=gates_passed,
        gate_passed=gates_passed >= 2,
        verdict=verdict,
        runtime_seconds=time.time() - t0,
    )

    # Print summary
    print(f"\n  {'Metric':<25s} {'Lang A':>10s} {'Lang B':>10s}")
    print(f"  {'Tokens':<25s} {len(a_valid):10d} {len(b_valid):10d}")
    print(f"  {'Types':<25s} {len(a_vocab):10d} {len(b_vocab):10d}")
    print(f"  {'Dict hit':<25s} {a_dict_hit:10.1%} {b_dict_hit:10.1%}")
    print(f"  {'Signal words':<25s} {len(a_signals):10d} {len(b_signals):10d}")
    print(f"\n  EVA vocab overlap: 13.8%  CVC vocab overlap: {cvc_overlap:.1%} (Δ={overlap_change:+.1%})")
    print(f"  Coda chi²={chi2:.1f}, p={coda_p:.4f}")
    print(f"  A-exclusive signals: {sorted(a_exclusive)}")
    print(f"  B-exclusive signals: {sorted(b_exclusive)}")
    print(f"\n  Gates: G1={'PASS' if g1 else 'FAIL'} G2={'PASS' if g2 else 'FAIL'} "
          f"G3={'PASS' if g3 else 'FAIL'} ({gates_passed}/3)")
    print(f"  Verdict: {verdict}")

    path = _save_json(rd, 'phase62_lang_ab_cvc.json', asdict(result))
    print(f"\n  Saved: {path}")
    print(f"  Runtime: {result.runtime_seconds:.1f}s")
    return result
