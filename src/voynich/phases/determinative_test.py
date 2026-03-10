"""
Phase 31.5: Gallows as Determinatives Test
============================================
Test whether the four gallows characters (k, t, p, f) function as silent
semantic classifiers rather than phonetic syllables.

Dependency chain:
    combined_refine.json     (Phase 15 assignment)
    modifier_integrate.json  (Phase 16 modifiers)
    null_corpus.json         (Phase 17 seeds)
        → determinative_test.json  (this step)
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
    decode_token_modifier_aware,
    load_corpus,
    tokenize_eva_chars,
)
from voynich.core.reference import (
    build_expanded_word_set,
    load_reference_corpus,
)
from voynich.phases.csp_solver import decode_token
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


GALLOWS_CHARS = {'k', 't', 'p', 'f'}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class GallowsDistribution:
    """Distribution analysis for gallows characters."""
    char: str
    frequency: int
    fraction: float
    position_profile: Dict[str, float]  # 'initial', 'medial', 'final'
    top_following_chars: List[Tuple[str, int]]


@dataclass
class StrippingResult:
    """Result of removing gallows from all tokens and re-decoding."""
    baseline_dict_hit: float
    stripped_dict_hit: float
    delta_dict_hit: float
    baseline_signal_rate: float
    stripped_signal_rate: float
    delta_signal_rate: float
    n_tokens_affected: int
    pct_tokens_affected: float


@dataclass
class SemanticClassification:
    """Result of grouping tokens by initial gallows and comparing semantics."""
    group: str  # gallows char or 'none'
    n_tokens: int
    top_decoded_words: List[Tuple[str, int]]
    semantic_profile: Dict[str, int]  # word category -> count


@dataclass
class DeterminativeResult:
    """Full Step 31.5 output."""
    # Distribution
    gallows_total_frequency: int
    gallows_total_fraction: float
    distributions: List[Dict]
    # Stripping test
    stripping: Dict
    # Semantic classification
    semantic_groups: List[Dict]
    semantic_chi_sq: float
    semantic_p_value: float
    semantic_df: int
    # Section distribution
    section_gallows_rates: Dict[str, float]
    section_chi_sq: float
    section_p_value: float
    # Null control
    null_mean_delta_dict_hit: float
    null_std_delta_dict_hit: float
    gallows_z_score: float
    null_mean_semantic_chi_sq: float
    # Verdict
    strip_improves: bool
    semantic_differentiation: bool
    section_nonuniform: bool
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def _gallows_distribution(
    corpus,
) -> Tuple[List[GallowsDistribution], int, float]:
    """Compute per-gallows frequency, position profile, co-occurrence."""
    all_tokens = corpus.get_tokens()
    total_chars = 0
    gallows_freq = Counter()
    position_counts = defaultdict(lambda: Counter())  # char -> position -> count
    following_chars = defaultdict(lambda: Counter())   # char -> next_char -> count

    for token in all_tokens:
        chars = tokenize_eva_chars(token)
        total_chars += len(chars)
        n = len(chars)
        for i, ch in enumerate(chars):
            if ch in GALLOWS_CHARS:
                gallows_freq[ch] += 1
                if i == 0:
                    position_counts[ch]['initial'] += 1
                elif i == n - 1:
                    position_counts[ch]['final'] += 1
                else:
                    position_counts[ch]['medial'] += 1
                # Following char
                if i + 1 < n:
                    following_chars[ch][chars[i + 1]] += 1

    distributions = []
    for ch in sorted(GALLOWS_CHARS):
        freq = gallows_freq[ch]
        pos = position_counts[ch]
        total_pos = sum(pos.values()) or 1
        profile = {
            'initial': round(pos['initial'] / total_pos, 3),
            'medial': round(pos['medial'] / total_pos, 3),
            'final': round(pos['final'] / total_pos, 3),
        }
        top_follow = following_chars[ch].most_common(5)
        distributions.append(GallowsDistribution(
            char=ch,
            frequency=freq,
            fraction=round(freq / max(total_chars, 1), 4),
            position_profile=profile,
            top_following_chars=top_follow,
        ))

    total_gallows = sum(gallows_freq.values())
    total_frac = total_gallows / max(total_chars, 1)
    return distributions, total_gallows, total_frac


def _strip_gallows_from_token(token: str) -> str:
    """Remove all gallows characters from a token."""
    chars = tokenize_eva_chars(token)
    stripped = [ch for ch in chars if ch not in GALLOWS_CHARS]
    return ''.join(stripped) if stripped else ''


def _strip_and_decode(
    corpus,
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    modifier_chars: set,
    modifier_rules: Dict[str, str],
    ref_word_set: set,
    null_seeds: List[int],
) -> StrippingResult:
    """Strip gallows from all tokens and compare dict-hit."""
    all_tokens = corpus.get_tokens()
    n_tokens = len(all_tokens)

    # Baseline
    baseline_decoded = _decode_corpus_r3(
        all_tokens, assignment, eva_to_triple,
        modifier_chars, modifier_rules, ref_word_set,
    )
    baseline_hits = sum(1 for w in baseline_decoded if w in ref_word_set)
    baseline_dict_hit = baseline_hits / n_tokens

    # Stripped
    stripped_tokens = []
    n_affected = 0
    for token in all_tokens:
        st = _strip_gallows_from_token(token)
        if st != token:
            n_affected += 1
        stripped_tokens.append(st if st else token)  # keep original if fully stripped

    stripped_decoded = _decode_corpus_r3(
        stripped_tokens, assignment, eva_to_triple,
        modifier_chars, modifier_rules, ref_word_set,
    )
    stripped_hits = sum(1 for w in stripped_decoded if w in ref_word_set)
    stripped_dict_hit = stripped_hits / n_tokens

    # Signal isolation on stripped (simplified — count SIGNAL-like tokens)
    # Use null corpora for signal comparison
    bigram_probs, initial_probs, token_lengths = _build_eva_bigram_model(all_tokens)

    baseline_signal = 0
    stripped_signal = 0

    for seed in null_seeds[:3]:  # Use 3 null corpora for speed
        null_tokens = _generate_null_corpus(
            bigram_probs, initial_probs, token_lengths, n_tokens, seed,
        )
        null_decoded = _decode_corpus_r3(
            null_tokens, assignment, eva_to_triple,
            modifier_chars, modifier_rules, ref_word_set,
        )
        null_hits = [w in ref_word_set for w in null_decoded]

        # Stripped null
        null_stripped = [_strip_gallows_from_token(t) if t else t for t in null_tokens]
        null_stripped = [t if t else nt for t, nt in zip(null_stripped, null_tokens)]
        null_stripped_decoded = _decode_corpus_r3(
            null_stripped, assignment, eva_to_triple,
            modifier_chars, modifier_rules, ref_word_set,
        )
        null_stripped_hits = [w in ref_word_set for w in null_stripped_decoded]

        # Count signal tokens (real hit, null miss)
        for i in range(n_tokens):
            if i < len(baseline_decoded) and i < len(null_decoded):
                if (baseline_decoded[i] in ref_word_set and
                        null_decoded[i] not in ref_word_set):
                    baseline_signal += 1
                if (i < len(stripped_decoded) and i < len(null_stripped_decoded) and
                        stripped_decoded[i] in ref_word_set and
                        null_stripped_decoded[i] not in ref_word_set):
                    stripped_signal += 1

    n_null = len(null_seeds[:3])
    baseline_signal_rate = baseline_signal / (n_tokens * n_null) if n_tokens * n_null > 0 else 0
    stripped_signal_rate = stripped_signal / (n_tokens * n_null) if n_tokens * n_null > 0 else 0

    return StrippingResult(
        baseline_dict_hit=round(baseline_dict_hit, 4),
        stripped_dict_hit=round(stripped_dict_hit, 4),
        delta_dict_hit=round(stripped_dict_hit - baseline_dict_hit, 4),
        baseline_signal_rate=round(baseline_signal_rate, 4),
        stripped_signal_rate=round(stripped_signal_rate, 4),
        delta_signal_rate=round(stripped_signal_rate - baseline_signal_rate, 4),
        n_tokens_affected=n_affected,
        pct_tokens_affected=round(n_affected / n_tokens, 4),
    )


def _semantic_classification(
    corpus,
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    modifier_chars: set,
    modifier_rules: Dict[str, str],
    ref_word_set: set,
) -> Tuple[List[SemanticClassification], float, float, int]:
    """Group tokens by initial gallows and compare decoded-word distributions."""
    all_tokens = corpus.get_tokens()

    # Group tokens
    groups: Dict[str, List[str]] = defaultdict(list)  # group -> [tokens]
    for token in all_tokens:
        chars = tokenize_eva_chars(token)
        if chars and chars[0] in GALLOWS_CHARS:
            groups[chars[0]].append(token)
        else:
            groups['none'].append(token)

    # Decode each group
    results = []
    word_distributions: Dict[str, Counter] = {}

    for group_name in sorted(groups.keys()):
        tokens = groups[group_name]
        # Decode only the non-gallows portion for gallows-initial tokens
        if group_name in GALLOWS_CHARS:
            decode_tokens = [_strip_gallows_from_token(t) for t in tokens]
            decode_tokens = [t if t else tokens[i] for i, t in enumerate(decode_tokens)]
        else:
            decode_tokens = tokens

        decoded = _decode_corpus_r3(
            decode_tokens, assignment, eva_to_triple,
            modifier_chars, modifier_rules, ref_word_set,
        )

        word_counts = Counter(w for w in decoded if w in ref_word_set)
        word_distributions[group_name] = word_counts

        results.append(SemanticClassification(
            group=group_name,
            n_tokens=len(tokens),
            top_decoded_words=word_counts.most_common(10),
            semantic_profile=dict(word_counts.most_common(20)),
        ))

    # Chi-squared test: are word distributions different across groups?
    # Build contingency table with top-N most common words
    all_words = Counter()
    for wc in word_distributions.values():
        all_words.update(wc)
    top_words = [w for w, _ in all_words.most_common(30)]

    if len(top_words) < 2 or len(word_distributions) < 2:
        return results, 0.0, 1.0, 0

    # Contingency table
    group_names = sorted(word_distributions.keys())
    observed = np.zeros((len(group_names), len(top_words)))
    for i, gn in enumerate(group_names):
        for j, w in enumerate(top_words):
            observed[i, j] = word_distributions[gn].get(w, 0)

    # Chi-squared (manual to avoid scipy dependency)
    row_sums = observed.sum(axis=1, keepdims=True)
    col_sums = observed.sum(axis=0, keepdims=True)
    total = observed.sum()
    if total == 0:
        return results, 0.0, 1.0, 0

    expected = row_sums * col_sums / total
    # Avoid division by zero
    mask = expected > 0
    chi_sq = np.sum(((observed[mask] - expected[mask]) ** 2) / expected[mask])
    df = (len(group_names) - 1) * (len(top_words) - 1)

    # Approximate p-value using normal approximation for large df
    if df > 0:
        z = (chi_sq - df) / (2 * df) ** 0.5
        # Simple p-value approximation
        p_value = max(1e-10, 0.5 * (1.0 - min(1.0, abs(z) / 5.0)))
        if z > 3:
            p_value = 0.001
        elif z > 2:
            p_value = 0.01
        elif z > 1.5:
            p_value = 0.05
    else:
        p_value = 1.0

    return results, float(chi_sq), p_value, df


def _section_distribution(
    corpus,
) -> Tuple[Dict[str, float], float, float]:
    """Compute per-section gallows frequency ratios and test uniformity."""
    section_gallows = defaultdict(int)
    section_total = defaultdict(int)

    for folio, page in corpus.pages.items():
        section = page.section
        for token in page.all_tokens:
            chars = tokenize_eva_chars(token)
            section_total[section] += len(chars)
            for ch in chars:
                if ch in GALLOWS_CHARS:
                    section_gallows[section] += 1

    rates = {}
    for section in sorted(section_total.keys()):
        total = section_total[section]
        gallows = section_gallows.get(section, 0)
        rates[section] = round(gallows / max(total, 1), 4)

    # Chi-squared uniformity test
    sections = sorted(rates.keys())
    if len(sections) < 2:
        return rates, 0.0, 1.0

    observed = np.array([section_gallows.get(s, 0) for s in sections], dtype=float)
    totals = np.array([section_total.get(s, 1) for s in sections], dtype=float)
    overall_rate = observed.sum() / totals.sum()
    expected = totals * overall_rate

    mask = expected > 0
    chi_sq = np.sum(((observed[mask] - expected[mask]) ** 2) / expected[mask])
    df = len(sections) - 1

    # Approximate p-value
    if df > 0:
        z = (chi_sq - df) / max((2 * df) ** 0.5, 1)
        p_value = max(1e-10, 0.5 * (1.0 - min(1.0, abs(z) / 5.0)))
        if z > 3:
            p_value = 0.001
    else:
        p_value = 1.0

    return rates, float(chi_sq), p_value


def _null_random_chars_test(
    corpus,
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    modifier_chars: set,
    modifier_rules: Dict[str, str],
    ref_word_set: set,
    n_trials: int = 50,
) -> Tuple[float, float, float]:
    """Null control: strip random sets of 4 chars and compare to gallows stripping."""
    all_tokens = corpus.get_tokens()
    n_tokens = len(all_tokens)

    # Get all EVA chars in corpus
    all_chars_counter = Counter()
    for token in all_tokens:
        for ch in tokenize_eva_chars(token):
            all_chars_counter[ch] += 1

    # Exclude gallows and very rare chars
    candidate_chars = [ch for ch, cnt in all_chars_counter.items()
                       if ch not in GALLOWS_CHARS and cnt >= 100]

    # Baseline dict_hit
    baseline_decoded = _decode_corpus_r3(
        all_tokens, assignment, eva_to_triple,
        modifier_chars, modifier_rules, ref_word_set,
    )
    baseline_dict_hit = sum(1 for w in baseline_decoded if w in ref_word_set) / n_tokens

    rng = random.Random(42)
    deltas = []

    for trial in range(n_trials):
        # Pick 4 random chars
        if len(candidate_chars) < 4:
            break
        random_chars = set(rng.sample(candidate_chars, 4))

        # Strip these chars from all tokens
        stripped_tokens = []
        for token in all_tokens:
            chars = tokenize_eva_chars(token)
            remaining = [ch for ch in chars if ch not in random_chars]
            st = ''.join(remaining) if remaining else token
            stripped_tokens.append(st)

        stripped_decoded = _decode_corpus_r3(
            stripped_tokens, assignment, eva_to_triple,
            modifier_chars, modifier_rules, ref_word_set,
        )
        stripped_dict_hit = sum(1 for w in stripped_decoded if w in ref_word_set) / n_tokens
        deltas.append(stripped_dict_hit - baseline_dict_hit)

    if not deltas:
        return 0.0, 0.0, 0.0

    null_mean = sum(deltas) / len(deltas)
    null_var = sum((d - null_mean) ** 2 for d in deltas) / len(deltas)
    null_std = null_var ** 0.5

    return null_mean, null_std, null_std


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_determinative_test() -> None:
    """Step 31.5: Test whether gallows characters are determinatives."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE 31.5: Gallows as Determinatives")
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

    null_seeds = [100, 101, 102, 103, 104]
    null_path = os.path.join(rd, 'null_corpus.json')
    if os.path.exists(null_path):
        with open(null_path) as f:
            null_data = json.load(f)
        null_seeds = [r['seed'] for r in null_data.get('null_runs', [])]

    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    base_words = set(w.lower() for w in ref_corpus.get_combined_tokens('latin')
                     if len(w) >= 2)
    expanded, _ = build_expanded_word_set(base_words)
    ref_word_set = base_words | expanded
    print(f"     {len(assignment)} triples, {len(ref_word_set)} reference words")

    corpus = load_corpus(verbose=False)

    # ── 2. Gallows distribution ──
    print("\n  2. Gallows distribution...")
    distributions, total_gallows, total_frac = _gallows_distribution(corpus)
    for d in distributions:
        print(f"     {d.char}: freq={d.frequency}, frac={d.fraction:.4f}, "
              f"init={d.position_profile['initial']:.1%}, "
              f"med={d.position_profile['medial']:.1%}, "
              f"final={d.position_profile['final']:.1%}")

    # ── 3. Stripping test ──
    print("\n  3. Stripping test (removing all gallows)...")
    strip_result = _strip_and_decode(
        corpus, assignment, eva_to_triple,
        modifier_chars, modifier_rules, ref_word_set, null_seeds,
    )
    print(f"     Baseline dict_hit: {strip_result.baseline_dict_hit:.4f}")
    print(f"     Stripped dict_hit: {strip_result.stripped_dict_hit:.4f}")
    print(f"     Delta: {strip_result.delta_dict_hit:+.4f}")
    print(f"     Tokens affected: {strip_result.n_tokens_affected} "
          f"({strip_result.pct_tokens_affected:.1%})")
    print(f"     Baseline signal rate: {strip_result.baseline_signal_rate:.4f}")
    print(f"     Stripped signal rate: {strip_result.stripped_signal_rate:.4f}")

    # ── 4. Semantic classification ──
    print("\n  4. Semantic classification by initial gallows...")
    sem_groups, sem_chi_sq, sem_p, sem_df = _semantic_classification(
        corpus, assignment, eva_to_triple,
        modifier_chars, modifier_rules, ref_word_set,
    )
    for sg in sem_groups:
        top_3 = sg.top_decoded_words[:3]
        top_str = ', '.join(f'{w}({c})' for w, c in top_3)
        print(f"     {sg.group:5s}: {sg.n_tokens:6d} tokens, top: {top_str}")
    print(f"     Chi-squared: {sem_chi_sq:.1f}, df={sem_df}, p≈{sem_p:.4f}")

    # ── 5. Section distribution ──
    print("\n  5. Per-section gallows rates...")
    section_rates, section_chi_sq, section_p = _section_distribution(corpus)
    for section, rate in section_rates.items():
        print(f"     {section:20s}: {rate:.4f}")
    print(f"     Chi-squared: {section_chi_sq:.1f}, p≈{section_p:.4f}")

    # ── 6. Null control ──
    print("\n  6. Null control (stripping random chars, 50 trials)...")
    null_mean_delta, null_std_delta, _ = _null_random_chars_test(
        corpus, assignment, eva_to_triple,
        modifier_chars, modifier_rules, ref_word_set,
        n_trials=50,
    )

    gallows_delta = strip_result.delta_dict_hit
    gallows_z = ((gallows_delta - null_mean_delta) / null_std_delta
                 if null_std_delta > 0 else 0.0)
    print(f"     Null mean Δdict_hit: {null_mean_delta:+.4f} ± {null_std_delta:.4f}")
    print(f"     Gallows Δdict_hit: {gallows_delta:+.4f}")
    print(f"     Gallows z-score: {gallows_z:.2f}")

    # ── 7. Verdict ──
    strip_improves = strip_result.delta_dict_hit > 0.005
    semantic_diff = sem_p < 0.05
    section_nonuniform = section_p < 0.05

    if strip_improves and semantic_diff:
        verdict = "DETERMINATIVE_LIKELY"
    elif strip_improves or semantic_diff:
        verdict = "DETERMINATIVE_POSSIBLE"
    else:
        verdict = "DETERMINATIVE_UNLIKELY"

    print(f"\n  Verdict: {verdict}")
    print(f"     Strip improves dict_hit: {strip_improves}")
    print(f"     Semantic differentiation: {semantic_diff} (p={sem_p:.4f})")
    print(f"     Section non-uniform: {section_nonuniform} (p={section_p:.4f})")

    # ── 8. Save ──
    result = DeterminativeResult(
        gallows_total_frequency=total_gallows,
        gallows_total_fraction=round(total_frac, 4),
        distributions=[_convert(asdict(d)) for d in distributions],
        stripping=_convert(asdict(strip_result)),
        semantic_groups=[_convert(asdict(sg)) for sg in sem_groups],
        semantic_chi_sq=round(sem_chi_sq, 2),
        semantic_p_value=round(sem_p, 6),
        semantic_df=sem_df,
        section_gallows_rates=section_rates,
        section_chi_sq=round(section_chi_sq, 2),
        section_p_value=round(section_p, 6),
        null_mean_delta_dict_hit=round(null_mean_delta, 4),
        null_std_delta_dict_hit=round(null_std_delta, 4),
        gallows_z_score=round(gallows_z, 2),
        null_mean_semantic_chi_sq=0.0,  # not computed for speed
        strip_improves=strip_improves,
        semantic_differentiation=semantic_diff,
        section_nonuniform=section_nonuniform,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'determinative_test.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2)

    print(f"\n  Saved → {out_path}")
    print(f"  Completed in {time.time() - t0:.1f}s")
