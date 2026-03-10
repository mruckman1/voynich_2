"""
Phase 31.7: Interleaved Text Separation
==========================================
Test whether Language A text and Language B notation are interleaved within
folios, and whether separating them produces cleaner decoded output.

Dependency chain:
    lang_b_combinatorial.json  (Phase 3)
    combined_refine.json       (Phase 15 assignment)
    modifier_integrate.json    (Phase 16 modifiers)
    null_corpus.json           (Phase 17 seeds)
        → interleaved_test.json  (this step)
"""

import json
import os
import random
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

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
class LangBDistribution:
    """Distribution analysis of Language B tokens."""
    n_lang_b_tokens: int
    n_total_tokens: int
    b_fraction: float
    per_section_rates: Dict[str, float]
    per_section_counts: Dict[str, Tuple[int, int]]  # section -> (b_count, total)
    periodicity_peaks: List[Tuple[int, float]]  # (period, power)
    line_boundary_clustering: float  # fraction of B tokens at line start/end


@dataclass
class StreamReadability:
    """Readability metrics for a separated stream."""
    stream_name: str
    n_tokens: int
    dict_hit: float
    signal_rate: float


@dataclass
class StreamBAnalysis:
    """Analysis of Language B tokens as separate stream."""
    n_types: int
    top_types: List[Tuple[str, int]]
    per_section_type_overlap: Dict[str, float]
    functional_hypothesis: str


@dataclass
class InterleavedResult:
    """Full Step 31.7 output."""
    # Distribution
    distribution: Dict
    # Stream A readability
    stream_a: Dict
    # Combined (baseline) readability
    combined: Dict
    # Stream B analysis
    stream_b_analysis: Dict
    # Null control
    null_mean_dict_hit_improvement: float
    null_std_dict_hit_improvement: float
    real_improvement: float
    improvement_z_score: float
    # Verdict
    separation_improves: bool
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Language B identification
# ---------------------------------------------------------------------------

def _load_lang_b_vocabulary(rd: str) -> Set[str]:
    """Load Language B type list from Phase 3 results."""
    lb_path = os.path.join(rd, 'lang_b_combinatorial.json')
    if not os.path.exists(lb_path):
        return set()

    with open(lb_path) as f:
        data = json.load(f)

    types = set(data.get('lang_b_type_list', []))
    # Also add edy and aiin family members
    for fam in ['edy_family', 'aiin_family']:
        types.update(data.get(fam, []))

    return types


def _tag_tokens(
    all_tokens: List[str],
    lang_b_vocab: Set[str],
) -> List[bool]:
    """Tag each token as Language B (True) or Language A (False)."""
    return [token in lang_b_vocab for token in all_tokens]


# ---------------------------------------------------------------------------
# Distribution analysis
# ---------------------------------------------------------------------------

def _distribution_analysis(
    corpus,
    all_tokens: List[str],
    is_b: List[bool],
) -> LangBDistribution:
    """Analyse the distribution of Language B tokens."""
    n_total = len(all_tokens)
    n_b = sum(is_b)

    # Per-section rates
    section_counts: Dict[str, Tuple[int, int]] = defaultdict(lambda: (0, 0))
    idx = 0
    for folio, page in corpus.pages.items():
        section = page.section
        n_page = len(page.all_tokens)
        b_count = sum(1 for i in range(idx, idx + n_page) if i < len(is_b) and is_b[i])
        old_b, old_t = section_counts[section]
        section_counts[section] = (old_b + b_count, old_t + n_page)
        idx += n_page

    per_section_rates = {}
    for section, (bc, tc) in section_counts.items():
        per_section_rates[section] = round(bc / max(tc, 1), 4)

    # Periodicity: FFT on binary is_b signal
    signal = np.array([1.0 if b else 0.0 for b in is_b])
    if len(signal) > 100:
        # Remove mean
        signal = signal - signal.mean()
        fft_result = np.abs(np.fft.rfft(signal))
        freqs = np.fft.rfftfreq(len(signal))
        # Find top peaks (skip DC component)
        peak_indices = np.argsort(fft_result[1:])[-5:] + 1
        periodicity_peaks = []
        for pi in peak_indices:
            if freqs[pi] > 0:
                period = int(1.0 / freqs[pi])
                power = float(fft_result[pi])
                periodicity_peaks.append((period, round(power, 2)))
        periodicity_peaks.sort(key=lambda x: -x[1])
    else:
        periodicity_peaks = []

    # Line boundary clustering: check if B tokens cluster at token-position 0
    # (first token on a line) or last token on a line
    # Approximate by checking position within page
    n_at_boundary = 0
    idx = 0
    for folio, page in corpus.pages.items():
        n_page = len(page.all_tokens)
        for i in range(n_page):
            global_idx = idx + i
            if global_idx < len(is_b) and is_b[global_idx]:
                # First or last token on this page counts as boundary
                if i <= 1 or i >= n_page - 2:
                    n_at_boundary += 1
        idx += n_page

    boundary_rate = n_at_boundary / max(n_b, 1)

    return LangBDistribution(
        n_lang_b_tokens=n_b,
        n_total_tokens=n_total,
        b_fraction=round(n_b / max(n_total, 1), 4),
        per_section_rates=per_section_rates,
        per_section_counts={s: list(v) for s, v in section_counts.items()},
        periodicity_peaks=periodicity_peaks[:5],
        line_boundary_clustering=round(boundary_rate, 4),
    )


# ---------------------------------------------------------------------------
# Stream separation and readability
# ---------------------------------------------------------------------------

def _separate_and_decode(
    all_tokens: List[str],
    is_b: List[bool],
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    modifier_chars: set,
    modifier_rules: Dict[str, str],
    ref_word_set: set,
) -> Tuple[StreamReadability, StreamReadability]:
    """Separate into Stream A and combined, decode both, measure readability."""
    n_total = len(all_tokens)

    # Combined baseline
    combined_decoded = _decode_corpus_r3(
        all_tokens, assignment, eva_to_triple,
        modifier_chars, modifier_rules, ref_word_set,
    )
    combined_hits = sum(1 for w in combined_decoded if w in ref_word_set)
    combined_dict_hit = combined_hits / n_total

    # Stream A: only Language A tokens
    stream_a_tokens = [t for t, b in zip(all_tokens, is_b) if not b]
    n_a = len(stream_a_tokens)

    if n_a > 0:
        stream_a_decoded = _decode_corpus_r3(
            stream_a_tokens, assignment, eva_to_triple,
            modifier_chars, modifier_rules, ref_word_set,
        )
        stream_a_hits = sum(1 for w in stream_a_decoded if w in ref_word_set)
        stream_a_dict_hit = stream_a_hits / n_a
    else:
        stream_a_dict_hit = 0.0

    # Signal rates (simplified — use dict_hit as proxy)
    combined_result = StreamReadability(
        stream_name='combined',
        n_tokens=n_total,
        dict_hit=round(combined_dict_hit, 4),
        signal_rate=0.0,
    )

    stream_a_result = StreamReadability(
        stream_name='stream_a',
        n_tokens=n_a,
        dict_hit=round(stream_a_dict_hit, 4),
        signal_rate=0.0,
    )

    return stream_a_result, combined_result


def _analyse_stream_b(
    all_tokens: List[str],
    is_b: List[bool],
    corpus,
) -> StreamBAnalysis:
    """Analyse Language B tokens as a separate notation stream."""
    stream_b_tokens = [t for t, b in zip(all_tokens, is_b) if b]
    type_counts = Counter(stream_b_tokens)

    # Per-section type overlap
    section_types: Dict[str, Set[str]] = defaultdict(set)
    idx = 0
    for folio, page in corpus.pages.items():
        section = page.section
        n_page = len(page.all_tokens)
        for i in range(n_page):
            global_idx = idx + i
            if global_idx < len(is_b) and is_b[global_idx]:
                section_types[section].add(all_tokens[global_idx])
        idx += n_page

    # Compute pairwise Jaccard overlap between sections
    sections = sorted(section_types.keys())
    overlap_scores = {}
    for i, s1 in enumerate(sections):
        for s2 in sections[i + 1:]:
            t1 = section_types[s1]
            t2 = section_types[s2]
            if t1 | t2:
                jaccard = len(t1 & t2) / len(t1 | t2)
                overlap_scores[f"{s1}_{s2}"] = round(jaccard, 4)

    # Functional hypothesis
    n_types = len(type_counts)
    if n_types < 20:
        hypothesis = "fixed_labels"
    elif n_types < 100:
        hypothesis = "structured_notation"
    else:
        hypothesis = "mixed_vocabulary"

    return StreamBAnalysis(
        n_types=n_types,
        top_types=type_counts.most_common(20),
        per_section_type_overlap=overlap_scores,
        functional_hypothesis=hypothesis,
    )


# ---------------------------------------------------------------------------
# Null control
# ---------------------------------------------------------------------------

def _null_random_removal(
    all_tokens: List[str],
    removal_fraction: float,
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    modifier_chars: set,
    modifier_rules: Dict[str, str],
    ref_word_set: set,
    n_trials: int = 100,
) -> Tuple[float, float]:
    """Randomly remove the same fraction of tokens and compare improvement."""
    n_total = len(all_tokens)
    n_remove = int(n_total * removal_fraction)

    # Combined baseline
    combined_decoded = _decode_corpus_r3(
        all_tokens, assignment, eva_to_triple,
        modifier_chars, modifier_rules, ref_word_set,
    )
    combined_dict_hit = sum(1 for w in combined_decoded if w in ref_word_set) / n_total

    rng = random.Random(42)
    improvements = []

    for trial in range(n_trials):
        # Remove random tokens
        remove_indices = set(rng.sample(range(n_total), min(n_remove, n_total)))
        remaining = [t for i, t in enumerate(all_tokens) if i not in remove_indices]

        if not remaining:
            continue

        remaining_decoded = _decode_corpus_r3(
            remaining, assignment, eva_to_triple,
            modifier_chars, modifier_rules, ref_word_set,
        )
        remaining_dict_hit = (sum(1 for w in remaining_decoded if w in ref_word_set)
                              / len(remaining))
        improvements.append(remaining_dict_hit - combined_dict_hit)

    if not improvements:
        return 0.0, 0.0

    mean_imp = sum(improvements) / len(improvements)
    var_imp = sum((x - mean_imp) ** 2 for x in improvements) / len(improvements)
    std_imp = var_imp ** 0.5

    return mean_imp, std_imp


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_interleaved_test() -> None:
    """Step 31.7: Test Language A / Language B stream separation."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE 31.7: Interleaved Text Separation")
    print("=" * 70)

    rd = _results_dir()
    eva_to_triple = build_eva_to_triple_lookup()

    # ── 1. Load inputs ──
    print("\n  1. Loading inputs...")

    with open(os.path.join(rd, 'combined_refine.json')) as f:
        refine_data = json.load(f)
    assignment = refine_data.get('best_assignment', {})

    with open(os.path.join(rd, 'modifier_integrate.json')) as f:
        mod_data = json.load(f)
    modifier_chars, modifier_rules = _reconstruct_modifier_rules(mod_data)

    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    base_words = set(w.lower() for w in ref_corpus.get_combined_tokens('latin')
                     if len(w) >= 2)
    expanded, _ = build_expanded_word_set(base_words)
    ref_word_set = base_words | expanded

    corpus = load_corpus(verbose=False)
    all_tokens = corpus.get_tokens()
    n_tokens = len(all_tokens)
    print(f"     {len(assignment)} triples, {n_tokens} tokens, "
          f"{len(ref_word_set)} reference words")

    # ── 2. Identify Language B tokens ──
    print("\n  2. Identifying Language B tokens...")
    lang_b_vocab = _load_lang_b_vocabulary(rd)
    is_b = _tag_tokens(all_tokens, lang_b_vocab)
    n_b = sum(is_b)
    b_frac = n_b / n_tokens
    print(f"     Language B vocabulary: {len(lang_b_vocab)} types")
    print(f"     Tagged {n_b} Language B tokens ({b_frac:.1%} of corpus)")

    # ── 3. Distribution analysis ──
    print("\n  3. Language B distribution analysis...")
    dist = _distribution_analysis(corpus, all_tokens, is_b)
    for section, rate in sorted(dist.per_section_rates.items()):
        print(f"     {section:20s}: {rate:.1%} Language B")
    if dist.periodicity_peaks:
        top_peak = dist.periodicity_peaks[0]
        print(f"     Strongest periodicity: every ~{top_peak[0]} tokens "
              f"(power={top_peak[1]:.1f})")
    print(f"     Boundary clustering: {dist.line_boundary_clustering:.1%}")

    # ── 4. Stream separation and readability ──
    print("\n  4. Stream separation and readability comparison...")
    stream_a, combined = _separate_and_decode(
        all_tokens, is_b, assignment, eva_to_triple,
        modifier_chars, modifier_rules, ref_word_set,
    )
    print(f"     Combined:  {combined.n_tokens} tokens, "
          f"dict_hit={combined.dict_hit:.4f}")
    print(f"     Stream A:  {stream_a.n_tokens} tokens, "
          f"dict_hit={stream_a.dict_hit:.4f}")
    real_improvement = stream_a.dict_hit - combined.dict_hit
    print(f"     Improvement: {real_improvement:+.4f}")

    # ── 5. Stream B analysis ──
    print("\n  5. Analysing Language B stream...")
    stream_b_analysis = _analyse_stream_b(all_tokens, is_b, corpus)
    print(f"     {stream_b_analysis.n_types} unique types in Stream B")
    print(f"     Top types: {', '.join(f'{w}({c})' for w, c in stream_b_analysis.top_types[:5])}")
    print(f"     Functional hypothesis: {stream_b_analysis.functional_hypothesis}")

    # ── 6. Null control ──
    print("\n  6. Null control (random removal, 100 trials)...")
    null_mean, null_std = _null_random_removal(
        all_tokens, b_frac, assignment, eva_to_triple,
        modifier_chars, modifier_rules, ref_word_set,
        n_trials=100,
    )
    z_score = ((real_improvement - null_mean) / null_std
               if null_std > 0 else 0.0)
    print(f"     Null mean improvement: {null_mean:+.4f} ± {null_std:.4f}")
    print(f"     Real improvement: {real_improvement:+.4f}")
    print(f"     Z-score: {z_score:.2f}")

    # ── 7. Verdict ──
    separation_improves = real_improvement > 0 and z_score > 2.0

    if separation_improves:
        verdict = "SEPARATION_BENEFICIAL"
    elif real_improvement > 0:
        verdict = "SEPARATION_MARGINAL"
    else:
        verdict = "SEPARATION_NOT_BENEFICIAL"

    print(f"\n  Verdict: {verdict}")

    # ── 8. Save ──
    result = InterleavedResult(
        distribution=_convert(asdict(dist)),
        stream_a=_convert(asdict(stream_a)),
        combined=_convert(asdict(combined)),
        stream_b_analysis=_convert(asdict(stream_b_analysis)),
        null_mean_dict_hit_improvement=round(null_mean, 4),
        null_std_dict_hit_improvement=round(null_std, 4),
        real_improvement=round(real_improvement, 4),
        improvement_z_score=round(z_score, 2),
        separation_improves=separation_improves,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'interleaved_test.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2)

    print(f"\n  Saved → {out_path}")
    print(f"  Completed in {time.time() - t0:.1f}s")
