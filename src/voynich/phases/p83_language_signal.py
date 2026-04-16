"""
Phase 83: Cross-Language Signal Comparison (Reviewer 3.10)
===========================================================
Runs the full signal isolation pipeline against German and Hebrew
dictionaries using the SAME T_P15 assignment table, to test whether
the signal is language-specific or an artifact of CV-to-dictionary geometry.

Uses raw CV decode (no R3 optimization) for fairness: all languages are
tested against the same decoded strings.

Output: results/p83_language_signal.json
"""

import json
import math
import os
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    load_corpus,
    tokenize_eva_chars,
)
from voynich.core.reference import load_reference_corpus
from voynich.phases.coda_markers import get_coda
from voynich.phases.corrected_coda import decode_token_cvc_v2
from voynich.phases.dict_calibration import _classify_tokens
from voynich.phases.null_corpus import (
    _build_eva_bigram_model,
    _generate_null_corpus,
)
from voynich.phases.p75_redecode import _build_3coda_table


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
class LanguageSignalEntry:
    """Signal results for one language dictionary."""
    language: str
    dict_size: int
    # Raw dict-hit
    raw_dict_hit: float
    raw_dict_hit_count: int
    # Signal classification
    n_signal: int
    n_shared_hit: int
    n_anti_signal: int
    n_shared_miss: int
    signal_rate: float
    anti_signal_rate: float
    net_signal: float           # signal_rate - anti_signal_rate
    # Per-word signal
    n_signal_words: int         # words with σ > 2.0
    top_signal_words: List[Dict[str, Any]]  # top 20 signal words
    mean_selectivity: float     # of signal words only
    # Null baseline
    null_mean_hit_rate: float
    selectivity: float          # dict_hit / null_mean_hit


@dataclass
class LanguageSignalResult:
    phase: str = "83"
    experiment: str = "language_signal_comparison"
    n_tokens: int = 0
    n_null_corpora: int = 5
    languages: List[LanguageSignalEntry] = field(default_factory=list)
    comparison_table: List[Dict[str, Any]] = field(default_factory=list)
    verdict: str = ""
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Dictionary building
# ---------------------------------------------------------------------------

def _build_german_dict(max_size: int = 10000) -> Set[str]:
    """Build German dictionary from reference corpus."""
    ref = load_reference_corpus(languages=['german'], verbose=False)
    tokens = ref.get_combined_tokens('german')
    # Lowercase and filter
    freq = Counter(w.lower() for w in tokens if len(w) >= 2 and w.isalpha())
    # Take top N by frequency
    top_words = [w for w, _ in freq.most_common(max_size)]
    return set(top_words)


def _build_hebrew_dict(data_dir: str, max_size: int = 10000) -> Set[str]:
    """Build Hebrew dictionary from frequency file."""
    he_path = os.path.join(data_dir, 'hebrew', 'he_50k.txt')
    if not os.path.exists(he_path):
        print(f"  WARNING: Hebrew data not found at {he_path}")
        return set()

    words = []
    with open(he_path, encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                word = parts[0]
                if len(word) >= 2:
                    words.append(word)
                if len(words) >= max_size:
                    break
    return set(words)


def _build_latin_10k_dict() -> Set[str]:
    """Rebuild the Latin 10K dictionary (same as Phase 36)."""
    from voynich.core.reference import build_expanded_word_set
    ref = load_reference_corpus(languages=['latin'], verbose=False)
    base_words = set(w.lower() for w in ref.get_combined_tokens('latin')
                     if len(w) >= 2)
    expanded, _ = build_expanded_word_set(base_words)
    all_words = base_words | expanded

    # Rank by frequency and take top 10K
    freq = Counter(w.lower() for w in ref.get_combined_tokens('latin'))
    ranked = sorted(all_words, key=lambda w: freq.get(w, 0), reverse=True)
    return set(ranked[:10000])


def _build_italian_10k_dict(data_dir: str) -> Set[str]:
    """Build Italian 10K dictionary from reference corpus."""
    try:
        ref = load_reference_corpus(languages=['italian'], verbose=False)
        tokens = ref.get_combined_tokens('italian')
    except Exception:
        # Try loading from file directly
        italian_dir = os.path.join(data_dir, 'italian')
        if not os.path.isdir(italian_dir):
            return set()
        tokens = []
        for f in os.listdir(italian_dir):
            if f.endswith('.txt'):
                with open(os.path.join(italian_dir, f), encoding='utf-8',
                          errors='ignore') as fh:
                    tokens.extend(fh.read().split())

    freq = Counter(w.lower() for w in tokens if len(w) >= 2 and w.isalpha())
    top_words = [w for w, _ in freq.most_common(10000)]
    return set(top_words)


# ---------------------------------------------------------------------------
# Signal pipeline
# ---------------------------------------------------------------------------

def _decode_raw_cv(
    tokens: List[str],
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    coda_table,
) -> List[str]:
    """Decode tokens using raw CV decode (no R3, no dictionary optimization).

    Uses CVC decode for all tokens, producing the same decoded string
    regardless of which dictionary is being matched against.
    """
    decoded = []
    for tok in tokens:
        result = decode_token_cvc_v2(tok, assignment, eva_to_triple, coda_table)
        decoded.append(result.decoded_cvc.lower())
    return decoded


def _run_signal_for_language(
    language: str,
    word_set: Set[str],
    real_decoded: List[str],
    null_decoded_list: List[List[str]],
) -> LanguageSignalEntry:
    """Run the full signal classification for one language dictionary."""
    n_tokens = len(real_decoded)

    # Check real hits
    real_hits = [w in word_set for w in real_decoded]
    raw_hit_count = sum(real_hits)
    raw_hit_rate = raw_hit_count / n_tokens if n_tokens > 0 else 0.0

    # Check null hits
    null_hits_list = []
    null_hit_rates = []
    for null_decoded in null_decoded_list:
        nh = [w in word_set for w in null_decoded]
        null_hits_list.append(nh)
        null_hit_rates.append(sum(nh) / len(nh) if nh else 0.0)

    null_mean_hit = sum(null_hit_rates) / len(null_hit_rates) if null_hit_rates else 0.0
    selectivity = raw_hit_rate / null_mean_hit if null_mean_hit > 0 else float('inf')

    # Classify tokens
    classifications = _classify_tokens(real_hits, null_hits_list)
    class_counts = Counter(classifications)

    n_signal = class_counts.get('SIGNAL', 0)
    n_shared_hit = class_counts.get('SHARED_HIT', 0)
    n_anti = class_counts.get('ANTI_SIGNAL', 0)
    n_shared_miss = class_counts.get('SHARED_MISS', 0)
    signal_rate = n_signal / n_tokens if n_tokens > 0 else 0.0
    anti_rate = n_anti / n_tokens if n_tokens > 0 else 0.0

    # Per-word signal analysis
    word_real_counts: Dict[str, int] = Counter()
    word_signal_counts: Dict[str, int] = Counter()
    for i, w in enumerate(real_decoded):
        if real_hits[i]:
            word_real_counts[w] += 1
            if classifications[i] == 'SIGNAL':
                word_signal_counts[w] += 1

    n_null = len(null_decoded_list)
    null_word_counts: Dict[str, List[int]] = defaultdict(lambda: [0] * n_null)
    for ni, null_decoded in enumerate(null_decoded_list):
        for i, w in enumerate(null_decoded):
            if w in word_set:
                null_word_counts[w][ni] += 1

    signal_words = []
    for word, real_count in word_real_counts.items():
        if real_count < 3:
            continue
        null_counts = null_word_counts.get(word, [0] * n_null)
        null_mean = sum(null_counts) / n_null if n_null > 0 else 0.0
        null_var = sum((c - null_mean)**2 for c in null_counts) / n_null if n_null > 0 else 0.0
        null_std = null_var ** 0.5

        sigma = (real_count - null_mean) / null_std if null_std > 0 else (
            10.0 if real_count > null_mean else 0.0)
        sel = real_count / null_mean if null_mean > 0 else float('inf')

        if sigma > 2.0:
            signal_words.append({
                'word': word,
                'real_count': real_count,
                'null_mean': round(null_mean, 2),
                'sigma': round(sigma, 2),
                'selectivity': round(sel, 2),
            })

    signal_words.sort(key=lambda x: x['sigma'], reverse=True)

    mean_sel = (sum(w['selectivity'] for w in signal_words) / len(signal_words)
                if signal_words else 0.0)

    return LanguageSignalEntry(
        language=language,
        dict_size=len(word_set),
        raw_dict_hit=round(raw_hit_rate, 4),
        raw_dict_hit_count=raw_hit_count,
        n_signal=n_signal,
        n_shared_hit=n_shared_hit,
        n_anti_signal=n_anti,
        n_shared_miss=n_shared_miss,
        signal_rate=round(signal_rate, 4),
        anti_signal_rate=round(anti_rate, 4),
        net_signal=round(signal_rate - anti_rate, 4),
        n_signal_words=len(signal_words),
        top_signal_words=signal_words,
        mean_selectivity=round(mean_sel, 2),
        null_mean_hit_rate=round(null_mean_hit, 4),
        selectivity=round(selectivity, 2),
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_language_signal():
    """Phase 83: Compare signal isolation across Latin, Italian, German, Hebrew."""
    t0 = time.time()
    rd = _results_dir()
    print("Phase 83: Cross-Language Signal Comparison")
    print("=" * 60)

    # Load resources
    combined = _safe_load(os.path.join(rd, 'combined_refine.json'))
    assignment = combined.get('best_assignment', {})
    eva_to_triple = build_eva_to_triple_lookup()
    coda_table = _build_3coda_table()
    corpus = load_corpus(verbose=False)

    # Collect all tokens
    all_tokens = []
    for _, page in corpus.pages.items():
        all_tokens.extend(page.all_tokens)
    n_tokens = len(all_tokens)
    print(f"  Corpus: {n_tokens} tokens")

    # ------------------------------------------------------------------
    # 1. Decode real corpus (raw CVC, no R3)
    # ------------------------------------------------------------------
    print("\n  1. Decoding real corpus (raw CVC) ...")
    real_decoded = _decode_raw_cv(all_tokens, assignment, eva_to_triple, coda_table)
    print(f"     {len(real_decoded)} tokens decoded")

    # ------------------------------------------------------------------
    # 2. Generate and decode null corpora
    # ------------------------------------------------------------------
    print("  2. Generating and decoding 5 null corpora ...")
    bigram_probs, initial_probs, token_lengths = _build_eva_bigram_model(all_tokens)
    null_seeds = [100, 101, 102, 103, 104]
    null_decoded_list = []
    for i, seed in enumerate(null_seeds):
        null_tokens = _generate_null_corpus(
            bigram_probs, initial_probs, token_lengths, n_tokens, seed)
        null_decoded = _decode_raw_cv(
            null_tokens, assignment, eva_to_triple, coda_table)
        null_decoded_list.append(null_decoded)
        print(f"     Null {i+1} (seed={seed}): {len(null_decoded)} tokens")

    # ------------------------------------------------------------------
    # 3. Build dictionaries
    # ------------------------------------------------------------------
    print("\n  3. Building dictionaries ...")
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(__file__)))), 'data', 'reference')

    latin_dict = _build_latin_10k_dict()
    print(f"     Latin 10K: {len(latin_dict)} words")

    italian_dict = _build_italian_10k_dict(data_dir)
    print(f"     Italian 10K: {len(italian_dict)} words")

    german_dict = _build_german_dict(max_size=10000)
    print(f"     German 10K: {len(german_dict)} words")

    hebrew_dict = _build_hebrew_dict(data_dir, max_size=10000)
    print(f"     Hebrew 10K: {len(hebrew_dict)} words")

    # ------------------------------------------------------------------
    # 4. Run signal isolation for each language
    # ------------------------------------------------------------------
    print("\n  4. Running signal isolation per language ...")
    languages = [
        ('Latin_10K', latin_dict),
        ('Italian_10K', italian_dict),
        ('German_10K', german_dict),
        ('Hebrew_10K', hebrew_dict),
    ]

    entries = []
    for lang_name, word_set in languages:
        if not word_set:
            print(f"     {lang_name}: SKIPPED (empty dictionary)")
            continue
        print(f"\n     --- {lang_name} ({len(word_set)} words) ---")
        entry = _run_signal_for_language(
            lang_name, word_set, real_decoded, null_decoded_list)
        entries.append(entry)
        print(f"     Dict hit: {entry.raw_dict_hit:.1%}")
        print(f"     Null hit: {entry.null_mean_hit_rate:.1%}")
        print(f"     Selectivity: {entry.selectivity:.2f}×")
        print(f"     SIGNAL: {entry.n_signal} ({entry.signal_rate:.1%})")
        print(f"     ANTI_SIGNAL: {entry.n_anti_signal} ({entry.anti_signal_rate:.1%})")
        print(f"     Net signal: {entry.net_signal:.1%}")
        print(f"     Signal words (σ>2): {entry.n_signal_words}")
        if entry.top_signal_words:
            top3 = entry.top_signal_words[:3]
            top3_strs = [w['word'] + '(s=' + str(w['sigma']) + ')' for w in top3]
            print(f"     Top 3: {', '.join(top3_strs)}")

    # ------------------------------------------------------------------
    # 5. Build comparison table
    # ------------------------------------------------------------------
    comparison = []
    for e in entries:
        comparison.append({
            'language': e.language,
            'dict_size': e.dict_size,
            'dict_hit': f"{e.raw_dict_hit:.1%}",
            'null_hit': f"{e.null_mean_hit_rate:.1%}",
            'selectivity': f"{e.selectivity:.2f}×",
            'signal_rate': f"{e.signal_rate:.1%}",
            'anti_signal_rate': f"{e.anti_signal_rate:.1%}",
            'net_signal': f"{e.net_signal:.1%}",
            'signal_words': e.n_signal_words,
            'mean_selectivity': f"{e.mean_selectivity:.2f}×",
        })

    print("\n\n  === COMPARISON TABLE ===")
    header = f"{'Language':<15} {'Dict':>6} {'Hit%':>7} {'Null%':>7} {'Sel':>6} {'Sig%':>7} {'Anti%':>7} {'Net%':>7} {'Words':>6}"
    print(f"  {header}")
    print(f"  {'-'*len(header)}")
    for c in comparison:
        print(f"  {c['language']:<15} {c['dict_size']:>6} {c['dict_hit']:>7} "
              f"{c['null_hit']:>7} {c['selectivity']:>6} {c['signal_rate']:>7} "
              f"{c['anti_signal_rate']:>7} {c['net_signal']:>7} {c['signal_words']:>6}")

    # Verdict
    latin_entry = next((e for e in entries if 'Latin' in e.language), None)
    non_romance = [e for e in entries if 'German' in e.language or 'Hebrew' in e.language]

    if latin_entry and non_romance:
        max_non_romance_signal = max(e.n_signal_words for e in non_romance)
        if latin_entry.n_signal_words > max_non_romance_signal * 2:
            verdict = "LANGUAGE_DISCRIMINATING"
        elif latin_entry.n_signal_words > max_non_romance_signal:
            verdict = "WEAKLY_DISCRIMINATING"
        else:
            verdict = "NOT_DISCRIMINATING"
    else:
        verdict = "INCOMPLETE"

    print(f"\n  Verdict: {verdict}")

    result = LanguageSignalResult(
        n_tokens=n_tokens,
        languages=entries,
        comparison_table=comparison,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    path = _save_json(rd, 'p83_language_signal.json', result)
    print(f"\n  Saved -> {path}")
    print(f"  Runtime: {result.runtime_seconds:.1f}s")
    return result
