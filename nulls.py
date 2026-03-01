"""
Null Character Identification (Phase 2A)
==========================================
Determine whether certain EVA characters are meaningless padding ("nulls")
inserted to regularize token appearance without carrying linguistic information.

Phases:
  A.1 — Per-character information content analysis
  A.2 — Systematic stripping experiment (profile shift tracking)
  A.3 — Cross-validation with stroke positional analysis
  A.4 — Discriminant validation on best-stripped text
"""

import itertools
import json
import math
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from corpus import VoynichCorpus, load_corpus, tokenize_eva_chars
from stats import first_order_entropy, conditional_entropy, compute_all_entropy
from fingerprint import (
    compute_profile, EntropyProfile, ReferenceLibrary,
    compute_voynich_profile, generate_null_text,
)
from strokes import decompose_glyph, stroke_positional_analysis, Stroke


# ---------------------------------------------------------------------------
# Phase A.1: Per-Character Information Content
# ---------------------------------------------------------------------------

@dataclass
class CharacterInfoProfile:
    """Information content profile for a single EVA character."""
    char: str
    frequency: int
    frequency_rank: int
    relative_frequency: float
    h_next_given_c: float
    h_prev_given_c: float
    mi_position: float
    h1_after_removal: float
    h2_after_removal: float
    h1_delta: float
    h2_delta: float
    null_score: float


def _char_stream(tokens: List[str]) -> List[str]:
    """Flatten tokens into EVA character stream (no spaces)."""
    chars = []
    for t in tokens:
        chars.extend(tokenize_eva_chars(t))
    return chars


def char_context_entropy(
    char_stream: List[str],
    target_char: str,
) -> Tuple[float, float]:
    """
    Compute H(next_char | target_char) and H(prev_char | target_char).
    Low values mean the character's neighbors are highly predictable.
    """
    next_counts: Counter = Counter()
    prev_counts: Counter = Counter()

    for i, c in enumerate(char_stream):
        if c != target_char:
            continue
        if i + 1 < len(char_stream):
            next_counts[char_stream[i + 1]] += 1
        if i - 1 >= 0:
            prev_counts[char_stream[i - 1]] += 1

    def _entropy(counts: Counter) -> float:
        total = sum(counts.values())
        if total == 0:
            return 0.0
        h = 0.0
        for n in counts.values():
            p = n / total
            if p > 0:
                h -= p * math.log2(p)
        return h

    return _entropy(next_counts), _entropy(prev_counts)


def char_position_mi(
    tokens: List[str],
    target_char: str,
) -> float:
    """
    Compute MI(char=target_char, position_in_token).
    Positions: initial, medial, final, singleton.
    High MI = strong positional preference (NOT null-like).
    """
    positions = ['initial', 'medial', 'final', 'singleton']
    pos_counts = Counter()
    total_occurrences = 0

    for token in tokens:
        chars = tokenize_eva_chars(token)
        n = len(chars)
        for i, c in enumerate(chars):
            if c != target_char:
                continue
            total_occurrences += 1
            if n == 1:
                pos_counts['singleton'] += 1
            elif i == 0:
                pos_counts['initial'] += 1
            elif i == n - 1:
                pos_counts['final'] += 1
            else:
                pos_counts['medial'] += 1

    if total_occurrences == 0:
        return 0.0

    # Count all characters by position for marginal P(position)
    all_pos_counts = Counter()
    all_total = 0
    for token in tokens:
        chars = tokenize_eva_chars(token)
        n = len(chars)
        for i in range(n):
            all_total += 1
            if n == 1:
                all_pos_counts['singleton'] += 1
            elif i == 0:
                all_pos_counts['initial'] += 1
            elif i == n - 1:
                all_pos_counts['final'] += 1
            else:
                all_pos_counts['medial'] += 1

    if all_total == 0:
        return 0.0

    # P(char=target) across corpus
    p_char = total_occurrences / all_total

    mi = 0.0
    for pos in positions:
        joint = pos_counts.get(pos, 0)
        if joint == 0:
            continue
        p_joint = joint / all_total
        p_pos = all_pos_counts.get(pos, 0) / all_total
        if p_pos > 0 and p_char > 0:
            mi += p_joint * math.log2(p_joint / (p_char * p_pos))

    return mi


def strip_characters(
    tokens: List[str],
    chars_to_strip: List[str],
) -> Tuple[List[str], str]:
    """
    Remove all instances of specified characters from every token.
    Works at the raw character level, then drops empty tokens.
    Returns (stripped_tokens, stripped_text).
    """
    strip_set = set(chars_to_strip)
    stripped_tokens = []
    for token in tokens:
        # Strip at raw character level
        new_token = ''.join(c for c in token if c not in strip_set)
        if new_token:
            stripped_tokens.append(new_token)
    stripped_text = ' '.join(stripped_tokens)
    return stripped_tokens, stripped_text


def compute_all_char_info(
    tokens: List[str],
    text: str,
) -> List[CharacterInfoProfile]:
    """
    Compute CharacterInfoProfile for every EVA character in the corpus.
    Returns list sorted by null_score descending (most likely nulls first).
    """
    stream = _char_stream(tokens)
    freq = Counter(stream)

    # Baseline H1 and H2
    ent = compute_all_entropy(text)
    h1_base = ent['H1']
    h2_base = ent['H2']

    # Rank characters by frequency
    ranked = freq.most_common()
    rank_map = {char: i + 1 for i, (char, _) in enumerate(ranked)}
    total_chars = sum(freq.values())

    profiles = []

    for char, count in ranked:
        # Context entropy
        h_next, h_prev = char_context_entropy(stream, char)

        # Positional MI
        mi_pos = char_position_mi(tokens, char)

        # Information gain of removal
        stripped_tokens, stripped_text = strip_characters(tokens, [char])
        if stripped_text and len(stripped_tokens) >= 10:
            stripped_ent = compute_all_entropy(stripped_text)
            h1_stripped = stripped_ent['H1']
            h2_stripped = stripped_ent['H2']
        else:
            h1_stripped = h1_base
            h2_stripped = h2_base

        profiles.append(CharacterInfoProfile(
            char=char,
            frequency=count,
            frequency_rank=rank_map[char],
            relative_frequency=count / total_chars,
            h_next_given_c=round(h_next, 4),
            h_prev_given_c=round(h_prev, 4),
            mi_position=round(mi_pos, 6),
            h1_after_removal=round(h1_stripped, 4),
            h2_after_removal=round(h2_stripped, 4),
            h1_delta=round(h1_stripped - h1_base, 4),
            h2_delta=round(h2_stripped - h2_base, 4),
            null_score=0.0,  # computed below
        ))

    # Compute composite null_score
    # Null signature: high frequency, low context entropy, low positional MI,
    # negative h2_delta (text more predictable after removal)
    if not profiles:
        return profiles

    # Normalize each metric to [0, 1]
    max_freq = max(p.relative_frequency for p in profiles)
    max_mi = max(abs(p.mi_position) for p in profiles) or 1.0
    max_ctx = max(
        max(p.h_next_given_c for p in profiles),
        max(p.h_prev_given_c for p in profiles),
    ) or 1.0
    max_h2_delta = max(abs(p.h2_delta) for p in profiles) or 1.0

    for p in profiles:
        freq_score = p.relative_frequency / max_freq  # high freq = more null-like
        mi_score = 1.0 - abs(p.mi_position) / max_mi  # low MI = more null-like
        ctx_score = 1.0 - ((p.h_next_given_c + p.h_prev_given_c) / 2) / max_ctx
        h2_score = (-p.h2_delta) / max_h2_delta  # negative delta = more null-like
        h2_score = max(0.0, min(1.0, (h2_score + 1) / 2))  # normalize to [0, 1]

        p.null_score = round(
            0.25 * freq_score + 0.25 * mi_score + 0.25 * ctx_score + 0.25 * h2_score,
            4,
        )

    profiles.sort(key=lambda p: p.null_score, reverse=True)
    return profiles


# ---------------------------------------------------------------------------
# Phase A.2: Systematic Stripping Experiment
# ---------------------------------------------------------------------------

@dataclass
class StrippingResult:
    """Result of stripping one or more characters and re-profiling."""
    stripped_chars: List[str]
    best_language: str
    best_encoding: str
    best_similarity: float
    top_5: List[Dict]
    null_insertion_best_rank: int
    tokens_remaining: int
    tokens_lost: int


def run_stripping_experiment(
    tokens: List[str],
    chars_to_strip: List[str],
    library: ReferenceLibrary,
    label: str = "",
) -> Optional[StrippingResult]:
    """Strip specified characters, compute new profile, match against library."""
    stripped_tokens, stripped_text = strip_characters(tokens, chars_to_strip)

    if len(stripped_tokens) < 50:
        return None

    profile = compute_profile(stripped_text, stripped_tokens,
                              label=label or f"stripped_{'_'.join(chars_to_strip)}")
    matches = library.match(profile, metric='cosine')

    if not matches:
        return None

    # Find best null_insertion rank
    null_rank = -1
    for i, m in enumerate(matches):
        if m['encoding'] == 'null_insertion':
            null_rank = i + 1
            break

    return StrippingResult(
        stripped_chars=chars_to_strip,
        best_language=matches[0]['language'],
        best_encoding=matches[0]['encoding'],
        best_similarity=matches[0]['similarity'],
        top_5=[{k: v for k, v in m.items()} for m in matches[:5]],
        null_insertion_best_rank=null_rank,
        tokens_remaining=len(stripped_tokens),
        tokens_lost=len(tokens) - len(stripped_tokens),
    )


def systematic_stripping(
    tokens: List[str],
    library: ReferenceLibrary,
    char_rankings: List[CharacterInfoProfile],
    verbose: bool = True,
) -> List[StrippingResult]:
    """
    Run stripping experiments: singles, top pairs, top triple.
    """
    results: List[StrippingResult] = []
    all_chars = [p.char for p in char_rankings]
    total = len(all_chars)

    # Singles
    if verbose:
        print(f"\n  Stripping singles ({total} characters)...")
    for i, char in enumerate(all_chars):
        if verbose:
            print(f"    [{i + 1}/{total}] strip '{char}'...", end='', flush=True)
        result = run_stripping_experiment(tokens, [char], library)
        if result:
            results.append(result)
            if verbose:
                print(f" -> {result.best_language}+{result.best_encoding} "
                      f"({result.best_similarity:.4f}), "
                      f"{result.tokens_remaining} tokens")
        elif verbose:
            print(" -> too few tokens remaining")

    # Pairs from top 5 candidates
    top5 = [p.char for p in char_rankings[:5]]
    pairs = list(itertools.combinations(top5, 2))
    if verbose:
        print(f"\n  Stripping pairs ({len(pairs)} combinations from top 5)...")
    for pair in pairs:
        pair_list = list(pair)
        if verbose:
            print(f"    strip {pair_list}...", end='', flush=True)
        result = run_stripping_experiment(tokens, pair_list, library)
        if result:
            results.append(result)
            if verbose:
                print(f" -> {result.best_language}+{result.best_encoding} "
                      f"({result.best_similarity:.4f})")
        elif verbose:
            print(" -> too few tokens")

    # Triple from top 3
    top3 = [p.char for p in char_rankings[:3]]
    if verbose:
        print(f"\n  Stripping triple {top3}...", end='', flush=True)
    result = run_stripping_experiment(tokens, top3, library)
    if result:
        results.append(result)
        if verbose:
            print(f" -> {result.best_language}+{result.best_encoding} "
                  f"({result.best_similarity:.4f})")
    elif verbose:
        print(" -> too few tokens")

    results.sort(key=lambda r: r.best_similarity, reverse=True)
    return results


# ---------------------------------------------------------------------------
# Phase A.3: Stroke Cross-Validation
# ---------------------------------------------------------------------------

def stroke_cross_validation(
    null_candidates: List[str],
    tokens: List[str],
) -> Dict[str, Dict]:
    """
    For each null candidate, check whether its stroke's positional MI
    supports or contradicts null status.
    """
    pos_analysis = stroke_positional_analysis(tokens)
    stroke_mi_per_stroke = pos_analysis.get('position_entropy_per_stroke', {})
    stroke_probs = pos_analysis.get('stroke_position_probs', {})

    results = {}
    for char in null_candidates:
        strokes = decompose_glyph(char)
        stroke_names = [s.value for s in strokes]

        # Get positional entropy for each stroke in this character
        stroke_entropies = {}
        stroke_pos = {}
        for s_name in set(stroke_names):
            stroke_entropies[s_name] = stroke_mi_per_stroke.get(s_name, 0.0)
            stroke_pos[s_name] = stroke_probs.get(s_name, {})

        # A null character's strokes should have HIGH positional entropy
        # (appear in many positions) = low positional specificity
        mean_entropy = (sum(stroke_entropies.values()) / len(stroke_entropies)
                        if stroke_entropies else 0.0)

        # Positional entropy > 1.0 bit means fairly spread across positions
        supports_null = mean_entropy > 0.8

        results[char] = {
            'strokes': stroke_names,
            'stroke_positional_entropy': stroke_entropies,
            'stroke_position_probs': stroke_pos,
            'mean_positional_entropy': round(mean_entropy, 4),
            'supports_null': supports_null,
            'explanation': (
                f"Mean positional entropy {mean_entropy:.2f} bits — "
                f"{'consistent with null' if supports_null else 'contradicts null'} "
                f"(strokes are {'spread across' if supports_null else 'concentrated in'} positions)"
            ),
        }

    return results


# ---------------------------------------------------------------------------
# Phase A.4: Discriminant Validation on Stripped Text
# ---------------------------------------------------------------------------

def stripped_discriminant_test(
    original_tokens: List[str],
    chars_to_strip: List[str],
    library: ReferenceLibrary,
    n_trials: int = 50,
    seed: int = 42,
) -> Dict:
    """
    Test whether stripped real text still discriminates from stripped shuffled.
    """
    # Strip the real text
    stripped_tokens, stripped_text = strip_characters(original_tokens, chars_to_strip)
    if len(stripped_tokens) < 50:
        return {'error': 'too few tokens after stripping'}

    real_profile = compute_profile(stripped_text, stripped_tokens, label='stripped_real')
    real_matches = library.match(real_profile)
    real_best = real_matches[0] if real_matches else None

    # Generate shuffled versions of the stripped text
    rng = random.Random(seed)
    null_distances = []

    for trial in range(n_trials):
        # Shuffle characters within each stripped token
        shuffled_tokens = []
        for t in stripped_tokens:
            chars = list(t)
            rng.shuffle(chars)
            shuffled_tokens.append(''.join(chars))
        shuffled_text = ' '.join(shuffled_tokens)
        shuffled_profile = compute_profile(shuffled_text, shuffled_tokens,
                                           label=f'stripped_shuffle_{trial}')
        shuffled_matches = library.match(shuffled_profile)
        if shuffled_matches:
            null_distances.append(shuffled_matches[0]['distance'])

    result = {
        'stripped_chars': chars_to_strip,
        'tokens_after_strip': len(stripped_tokens),
        'real_best_match': real_best,
        'real_best_distance': real_best['distance'] if real_best else None,
    }

    if null_distances:
        mean_d = float(np.mean(null_distances))
        std_d = float(np.std(null_distances))
        result['shuffled_mean_distance'] = round(mean_d, 6)
        result['shuffled_std_distance'] = round(std_d, 6)
        if std_d > 0 and real_best:
            z = (real_best['distance'] - mean_d) / std_d
            result['z_score'] = round(z, 2)
            result['discriminates'] = abs(z) > 2.0
        else:
            result['z_score'] = 0.0
            result['discriminates'] = False
    else:
        result['shuffled_mean_distance'] = 0.0
        result['shuffled_std_distance'] = 0.0
        result['z_score'] = 0.0
        result['discriminates'] = False

    return result


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_null_analysis() -> Dict:
    """Run the full null character identification pipeline."""
    print("=" * 70)
    print("PHASE 2A: NULL CHARACTER IDENTIFICATION")
    print("=" * 70)

    # --- Load corpus ---
    corpus = load_corpus(verbose=False)
    tokens = corpus.get_tokens(paragraph_only=True)
    text = corpus.get_text(paragraph_only=True)

    # --- Phase A.1: Per-character information content ---
    print("\n--- Phase A.1: Per-Character Information Content ---")
    char_profiles = compute_all_char_info(tokens, text)

    print(f"\n  {'Char':<6} {'Freq':>6} {'Rank':>4} {'Rel%':>6} "
          f"{'H(nxt)':>7} {'H(prv)':>7} {'MI(pos)':>8} "
          f"{'ΔH1':>7} {'ΔH2':>7} {'Score':>6}")
    print(f"  {'-' * 74}")
    for p in char_profiles:
        print(f"  {p.char:<6} {p.frequency:>6} {p.frequency_rank:>4} "
              f"{p.relative_frequency * 100:>5.1f}% "
              f"{p.h_next_given_c:>7.3f} {p.h_prev_given_c:>7.3f} "
              f"{p.mi_position:>8.4f} "
              f"{p.h1_delta:>+7.3f} {p.h2_delta:>+7.3f} "
              f"{p.null_score:>6.3f}")

    top5 = char_profiles[:5]
    print(f"\n  Top 5 null candidates: {', '.join(p.char for p in top5)}")

    # --- Build Reference Library (reused for A.2 + A.4) ---
    print("\n--- Building Reference Library (reused for all experiments) ---")
    try:
        from reference import load_reference_corpus
        ref_corpus = load_reference_corpus(verbose=False)
    except (FileNotFoundError, ImportError):
        ref_corpus = None

    library = ReferenceLibrary(
        n_samples=30, n_words=500, verbose=True,
        reference_corpus=ref_corpus,
    )
    library.build()

    # Baseline match
    baseline_profile = compute_voynich_profile(corpus)
    baseline_matches = library.match(baseline_profile)
    if baseline_matches:
        b = baseline_matches[0]
        print(f"\n  Baseline (unstripped): {b['language']}+{b['encoding']} "
              f"(sim={b['similarity']:.4f})")

    # --- Phase A.2: Systematic stripping ---
    print("\n--- Phase A.2: Systematic Stripping Experiment ---")
    all_results = systematic_stripping(tokens, library, char_profiles, verbose=True)

    if all_results:
        print(f"\n  Top 10 stripping results:")
        print(f"  {'Stripped':<20} {'Best Match':<30} {'Sim':>8} "
              f"{'Null Rank':>9} {'Tokens':>7}")
        print(f"  {'-' * 76}")
        for r in all_results[:10]:
            chars_str = '+'.join(r.stripped_chars)
            match_str = f"{r.best_language}+{r.best_encoding}"
            print(f"  {chars_str:<20} {match_str:<30} {r.best_similarity:>8.4f} "
                  f"{r.null_insertion_best_rank:>9} {r.tokens_remaining:>7}")

    # --- Phase A.3: Stroke cross-validation ---
    print("\n--- Phase A.3: Stroke Cross-Validation ---")
    top_candidates = [p.char for p in top5]
    stroke_val = stroke_cross_validation(top_candidates, tokens)

    for char, info in stroke_val.items():
        print(f"  {char}: strokes={info['strokes']}, "
              f"mean_pos_entropy={info['mean_positional_entropy']:.3f} — "
              f"{info['explanation']}")

    # --- Phase A.4: Discriminant test ---
    print("\n--- Phase A.4: Discriminant Validation on Best Stripped Config ---")
    if all_results:
        best = all_results[0]
        disc = stripped_discriminant_test(
            tokens, best.stripped_chars, library, n_trials=50,
        )
        print(f"  Stripped chars: {best.stripped_chars}")
        print(f"  Real best-match distance: {disc.get('real_best_distance', 'N/A')}")
        print(f"  Shuffled mean distance: {disc.get('shuffled_mean_distance', 'N/A')} "
              f"(+/- {disc.get('shuffled_std_distance', 'N/A')})")
        print(f"  Z-score: {disc.get('z_score', 'N/A')}")
        print(f"  Discriminates: {disc.get('discriminates', False)}")
    else:
        disc = {'error': 'no stripping results'}

    # --- Save results ---
    os.makedirs('results', exist_ok=True)

    with open(os.path.join('results', 'null_char_profiles.json'), 'w') as f:
        json.dump({
            'baseline': {'h1': round(compute_all_entropy(text)['H1'], 4),
                         'h2': round(compute_all_entropy(text)['H2'], 4),
                         'total_tokens': len(tokens)},
            'characters': [asdict(p) for p in char_profiles],
        }, f, indent=2)

    with open(os.path.join('results', 'stripping_experiment.json'), 'w') as f:
        baseline_info = baseline_matches[0] if baseline_matches else {}
        json.dump({
            'baseline_match': baseline_info,
            'results': [asdict(r) for r in all_results],
        }, f, indent=2)

    with open(os.path.join('results', 'stroke_null_validation.json'), 'w') as f:
        json.dump(stroke_val, f, indent=2)

    with open(os.path.join('results', 'stripped_discriminant.json'), 'w') as f:
        json.dump(disc, f, indent=2, default=str)

    print(f"\n  Results saved to results/")

    return {
        'char_profiles': char_profiles,
        'stripping_results': all_results,
        'stroke_validation': stroke_val,
        'discriminant': disc,
    }
