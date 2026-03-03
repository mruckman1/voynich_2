"""
Phase 17.0.4 – Null Corpus Control Test
========================================
Generates synthetic text that matches the Voynich's EVA-character-level
statistical properties, then applies the same Phase 15/16 decode pipeline.
If the null corpus achieves comparable dict_hit, the real result is an artifact.

Design: The null corpus is generated at the EVA-token level using a character
bigram model trained on the real Voynich.  The *same* Phase 15 assignment and
Phase 16 modifier classification are applied — this tests whether the
assignment produces dictionary hits on ANY text with Voynich-like statistics.

Dependency chain:
    modifier_integrate.json  (Phase 16 modifiers)
    combined_refine.json     (Phase 15 best_assignment)
        → null_corpus.json   (this step)
"""

import json
import math
import os
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Set, Tuple

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


def _reconstruct_modifier_rules(data: Dict) -> Tuple[Set[str], Dict[str, str]]:
    modifier_chars = set(data.get('modifier_chars', []))
    modifier_rules: Dict[str, str] = {}
    for c in data.get('classifications', []):
        if c.get('final_classification') == 'modifier':
            modifier_rules[c['eva_char']] = c.get('modifier_type', 'silent')
    return modifier_chars, modifier_rules


def _compute_dict_hit(decoded: List[str], ref_word_set: set) -> float:
    if not decoded:
        return 0.0
    hits = sum(1 for w in decoded if w.lower() in ref_word_set)
    return hits / len(decoded)


# ---------------------------------------------------------------------------
# Null corpus generation
# ---------------------------------------------------------------------------

def _build_eva_bigram_model(
    tokens: List[str],
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float], List[int]]:
    """
    Build an EVA-character-level bigram model from real Voynich tokens.

    Returns:
        bigram_probs: {prev_char: {next_char: probability}}
        initial_probs: {char: probability at token start}
        token_lengths: list of EVA-char-lengths for each token
    """
    initial_counts: Counter = Counter()
    bigram_counts: Dict[str, Counter] = {}
    token_lengths: List[int] = []

    for token in tokens:
        chars = tokenize_eva_chars(token)
        if not chars:
            continue
        token_lengths.append(len(chars))
        initial_counts[chars[0]] += 1

        for i in range(len(chars) - 1):
            c1, c2 = chars[i], chars[i + 1]
            if c1 not in bigram_counts:
                bigram_counts[c1] = Counter()
            bigram_counts[c1][c2] += 1

    # Normalize initial probabilities
    total_initial = sum(initial_counts.values())
    initial_probs = {c: n / total_initial for c, n in initial_counts.items()}

    # Normalize bigram probabilities
    bigram_probs: Dict[str, Dict[str, float]] = {}
    for c1, counts in bigram_counts.items():
        total = sum(counts.values())
        bigram_probs[c1] = {c2: n / total for c2, n in counts.items()}

    return bigram_probs, initial_probs, token_lengths


def _sample_from_probs(probs: Dict[str, float], rng: random.Random) -> str:
    """Sample a character from a probability distribution."""
    items = list(probs.items())
    r = rng.random()
    cumulative = 0.0
    for char, p in items:
        cumulative += p
        if r <= cumulative:
            return char
    return items[-1][0]  # fallback


def _generate_null_corpus(
    bigram_probs: Dict[str, Dict[str, float]],
    initial_probs: Dict[str, float],
    token_lengths: List[int],
    n_tokens: int,
    seed: int,
) -> List[str]:
    """Generate a null corpus using the bigram character model."""
    rng = random.Random(seed)

    # Build length distribution
    length_counts = Counter(token_lengths)
    total_lengths = sum(length_counts.values())
    length_probs = {k: v / total_lengths for k, v in length_counts.items()}

    null_tokens: List[str] = []
    for _ in range(n_tokens):
        # Sample token length
        target_len = _sample_from_probs(
            {str(k): v for k, v in length_probs.items()}, rng,
        )
        target_len = int(target_len)
        target_len = max(1, target_len)

        # Generate token character by character
        chars = []
        current = _sample_from_probs(initial_probs, rng)
        chars.append(current)

        for _ in range(target_len - 1):
            next_probs = bigram_probs.get(current, initial_probs)
            current = _sample_from_probs(next_probs, rng)
            chars.append(current)

        null_tokens.append(''.join(chars))

    return null_tokens


def _compute_corpus_stats(tokens: List[str]) -> Dict[str, float]:
    """Compute basic statistical properties of a corpus."""
    char_counts: Counter = Counter()
    lengths = []
    for token in tokens:
        chars = tokenize_eva_chars(token)
        lengths.append(len(chars))
        for c in chars:
            char_counts[c] += 1

    total_chars = sum(char_counts.values())
    n_types = len(set(tokens))

    # First-order entropy
    h1 = 0.0
    for count in char_counts.values():
        p = count / total_chars if total_chars > 0 else 0
        if p > 0:
            h1 -= p * math.log2(p)

    return {
        'n_tokens': len(tokens),
        'n_types': n_types,
        'type_token_ratio': n_types / len(tokens) if tokens else 0,
        'mean_token_length': sum(lengths) / len(lengths) if lengths else 0,
        'h1': round(h1, 4),
        'n_unique_chars': len(char_counts),
    }


def _compute_jsd(real_tokens: List[str], null_tokens: List[str]) -> float:
    """Jensen-Shannon divergence between EVA character distributions."""
    real_counts: Counter = Counter()
    null_counts: Counter = Counter()
    for t in real_tokens:
        for c in tokenize_eva_chars(t):
            real_counts[c] += 1
    for t in null_tokens:
        for c in tokenize_eva_chars(t):
            null_counts[c] += 1

    all_chars = set(real_counts.keys()) | set(null_counts.keys())
    real_total = sum(real_counts.values())
    null_total = sum(null_counts.values())

    if real_total == 0 or null_total == 0:
        return 1.0

    jsd = 0.0
    for c in all_chars:
        p = real_counts[c] / real_total
        q = null_counts[c] / null_total
        m = (p + q) / 2
        if p > 0 and m > 0:
            jsd += 0.5 * p * math.log2(p / m)
        if q > 0 and m > 0:
            jsd += 0.5 * q * math.log2(q / m)

    return round(jsd, 6)


# ---------------------------------------------------------------------------
# Decode pipeline for null corpus
# ---------------------------------------------------------------------------

def _decode_null_corpus(
    null_tokens: List[str],
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    modifier_chars: Set[str],
    modifier_rules: Dict[str, str],
    base_words: set,
    expanded_set: set,
    max_tokens: int = 2000,
) -> Dict[str, float]:
    """Run three decode strategies on a null corpus."""
    limited = null_tokens[:max_tokens]

    # Naive decode (no modifiers)
    naive = [decode_token(t, assignment, eva_to_triple) for t in limited]
    naive_original_hit = _compute_dict_hit(naive, base_words)
    naive_expanded_hit = _compute_dict_hit(naive, expanded_set)

    # R1 strip
    r1 = [
        decode_token_modifier_aware(t, assignment, eva_to_triple, modifier_chars)
        for t in limited
    ]
    r1_expanded_hit = _compute_dict_hit(r1, expanded_set)

    # R3 combined
    r3 = []
    for token in limited:
        alt = decode_token_modifier_aware(
            token, assignment, eva_to_triple, modifier_chars,
            modifier_rules=modifier_rules,
        )
        if alt.lower() in expanded_set:
            r3.append(alt)
            continue
        stripped = decode_token_modifier_aware(
            token, assignment, eva_to_triple, modifier_chars,
        )
        if stripped.lower() in expanded_set:
            r3.append(stripped)
            continue
        r3.append(decode_token(token, assignment, eva_to_triple))
    r3_expanded_hit = _compute_dict_hit(r3, expanded_set)

    # Selectivity: random assignment baseline
    rng = random.Random(42)
    syllables = list(set(assignment.values()))
    rand_hits = []
    for _ in range(3):
        rand_assign = {k: rng.choice(syllables) for k in assignment}
        rand_decoded = [decode_token(t, rand_assign, eva_to_triple) for t in limited]
        rand_hits.append(_compute_dict_hit(rand_decoded, expanded_set))
    rand_baseline = sum(rand_hits) / len(rand_hits) if rand_hits else 0.01
    r3_selectivity = r3_expanded_hit / max(rand_baseline, 0.001)

    return {
        'naive_original_hit': round(naive_original_hit, 4),
        'naive_expanded_hit': round(naive_expanded_hit, 4),
        'r1_expanded_hit': round(r1_expanded_hit, 4),
        'r3_expanded_hit': round(r3_expanded_hit, 4),
        'r3_selectivity': round(r3_selectivity, 2),
        'random_baseline': round(rand_baseline, 4),
    }


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class NullCorpusRun:
    run_id: int
    seed: int
    corpus_stats: Dict
    char_jsd: float
    naive_original_hit: float
    naive_expanded_hit: float
    r1_expanded_hit: float
    r3_expanded_hit: float
    r3_selectivity: float


@dataclass
class HonestyNullResult:
    n_null_corpora: int
    generation_method: str

    # Real Voynich reference
    real_stats: Dict
    real_modifier_dict_hit: float
    real_expanded_dict_hit: float

    # Per-null-corpus results
    null_runs: List[Dict]

    # Summary
    null_r3_hit_mean: float
    null_r3_hit_std: float
    null_r3_hit_min: float
    null_r3_hit_max: float

    null_naive_original_mean: float

    separation_sigma: float
    selectivity_vs_null: float

    gate_passed: bool
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_null_corpus() -> None:
    """Step 17.0.4: Null corpus end-to-end control test."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE 17.0.4: Null Corpus Control Test")
    print("=" * 70)

    rd = _results_dir()
    N_CORPORA = 5

    # ─── Load Phase 16 results ───
    print("\n  1. Loading Phase 16 results …")
    mod_path = os.path.join(rd, 'modifier_integrate.json')
    if not os.path.exists(mod_path):
        print("  [SKIP] modifier_integrate.json not found")
        return
    with open(mod_path) as f:
        mod_data = json.load(f)
    modifier_chars, modifier_rules = _reconstruct_modifier_rules(mod_data)
    real_modifier_hit = mod_data.get('best_dict_hit', 0.5165)

    # ─── Load Phase 15 assignment ───
    refine_path = os.path.join(rd, 'combined_refine.json')
    if not os.path.exists(refine_path):
        print("  [SKIP] combined_refine.json not found")
        return
    with open(refine_path) as f:
        refine_data = json.load(f)
    assignment = refine_data.get('best_assignment', {})

    # ─── Load corpus ───
    print("\n  2. Loading corpus …")
    corpus = load_corpus(verbose=False)
    tokens = corpus.get_tokens()
    eva_to_triple = build_eva_to_triple_lookup()
    print(f"      {len(tokens)} tokens")

    # ─── Build dictionaries ───
    print("\n  3. Building dictionaries …")
    try:
        ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
        base_words = set(
            w.lower() for w in ref_corpus.get_combined_tokens('latin')
            if len(w) >= 2
        )
    except (FileNotFoundError, KeyError):
        base_words = set()
    expanded_words, _ = build_expanded_word_set(base_words)
    expanded_set = base_words | expanded_words
    print(f"      Original: {len(base_words):,} | Expanded: {len(expanded_set):,}")

    # ─── Compute real Voynich dict_hit for reference ───
    print("\n  4. Computing real Voynich dict_hit (expanded, no modifiers) …")
    real_naive = [decode_token(t, assignment, eva_to_triple) for t in tokens[:2000]]
    real_expanded_hit = _compute_dict_hit(real_naive, expanded_set)
    print(f"      Real naive expanded: {real_expanded_hit:.1%}")
    print(f"      Real R3 modifier:    {real_modifier_hit:.1%}")

    real_stats = _compute_corpus_stats(tokens)
    print(f"      Real stats: {real_stats}")

    # ─── Build bigram model ───
    print("\n  5. Building EVA character bigram model …")
    bigram_probs, initial_probs, token_lengths = _build_eva_bigram_model(tokens)
    print(f"      {len(initial_probs)} initial chars, "
          f"{sum(len(v) for v in bigram_probs.values())} bigram transitions")

    # ─── Generate and test null corpora ───
    print(f"\n  6. Generating and testing {N_CORPORA} null corpora …")
    null_runs: List[NullCorpusRun] = []

    for i in range(N_CORPORA):
        seed = 100 + i
        print(f"\n      ── Null corpus {i + 1} (seed={seed}) ──")

        # Generate
        null_tokens = _generate_null_corpus(
            bigram_probs, initial_probs, token_lengths,
            n_tokens=len(tokens), seed=seed,
        )
        null_stats = _compute_corpus_stats(null_tokens)
        char_jsd = _compute_jsd(tokens, null_tokens)
        print(f"      Stats: {null_stats}")
        print(f"      Char JSD from real: {char_jsd:.6f}")

        # Decode
        decode_results = _decode_null_corpus(
            null_tokens, assignment, eva_to_triple,
            modifier_chars, modifier_rules,
            base_words, expanded_set,
        )
        print(f"      Naive original:  {decode_results['naive_original_hit']:.1%}")
        print(f"      Naive expanded:  {decode_results['naive_expanded_hit']:.1%}")
        print(f"      R1 expanded:     {decode_results['r1_expanded_hit']:.1%}")
        print(f"      R3 expanded:     {decode_results['r3_expanded_hit']:.1%}")
        print(f"      R3 selectivity:  {decode_results['r3_selectivity']:.2f}×")

        null_runs.append(NullCorpusRun(
            run_id=i + 1,
            seed=seed,
            corpus_stats=null_stats,
            char_jsd=char_jsd,
            naive_original_hit=decode_results['naive_original_hit'],
            naive_expanded_hit=decode_results['naive_expanded_hit'],
            r1_expanded_hit=decode_results['r1_expanded_hit'],
            r3_expanded_hit=decode_results['r3_expanded_hit'],
            r3_selectivity=decode_results['r3_selectivity'],
        ))

    # ─── Summary statistics ───
    r3_hits = [r.r3_expanded_hit for r in null_runs]
    r3_mean = sum(r3_hits) / len(r3_hits)
    r3_std = (sum((x - r3_mean) ** 2 for x in r3_hits) / len(r3_hits)) ** 0.5
    r3_min = min(r3_hits)
    r3_max = max(r3_hits)

    naive_orig_hits = [r.naive_original_hit for r in null_runs]
    naive_orig_mean = sum(naive_orig_hits) / len(naive_orig_hits)

    separation = (real_modifier_hit - r3_mean) / max(r3_std, 0.001)
    selectivity_vs_null = real_modifier_hit / max(r3_mean, 0.001)

    print(f"\n  7. Summary:")
    print(f"      Real R3 modifier:     {real_modifier_hit:.1%}")
    print(f"      Null R3 mean ± std:   {r3_mean:.1%} ± {r3_std:.1%}")
    print(f"      Null R3 range:        [{r3_min:.1%}, {r3_max:.1%}]")
    print(f"      Null naive orig mean: {naive_orig_mean:.1%}")
    print(f"      Separation:           {separation:.1f}σ")
    print(f"      Selectivity vs null:  {selectivity_vs_null:.2f}×")

    # ─── Gate ───
    gate_passed = r3_max < 0.25
    print(f"\n  8. Gate: null_r3_max < 0.25")
    print(f"      null_r3_max = {r3_max:.1%}")
    print(f"      {'PASS' if gate_passed else 'FAIL'}")

    # ─── Verdict ───
    if gate_passed:
        verdict = (
            f"PASS: Null corpora achieve {r3_mean:.1%} R3 dict_hit "
            f"(max {r3_max:.1%}), vs real {real_modifier_hit:.1%}. "
            f"Separation {separation:.1f}σ. Pipeline does not replicate "
            f"results on null text."
        )
    elif r3_max < 0.40:
        genuine_signal = real_modifier_hit - r3_mean
        verdict = (
            f"MARGINAL: Null corpora achieve {r3_mean:.1%} "
            f"(max {r3_max:.1%}), vs real {real_modifier_hit:.1%}. "
            f"Pipeline finds some Latin in noise. "
            f"Genuine signal ≈ {genuine_signal:.1%} "
            f"({separation:.1f}σ separation)."
        )
    else:
        verdict = (
            f"FAIL: Null corpora achieve {r3_mean:.1%} "
            f"(max {r3_max:.1%}), comparable to real {real_modifier_hit:.1%}. "
            f"Pipeline produces similar results on null text — "
            f"Voynich results may be artifacts."
        )

    print(f"\n  Verdict: {verdict}")

    # ─── Save ───
    result = HonestyNullResult(
        n_null_corpora=N_CORPORA,
        generation_method='eva_char_bigram',
        real_stats=real_stats,
        real_modifier_dict_hit=round(real_modifier_hit, 4),
        real_expanded_dict_hit=round(real_expanded_hit, 4),
        null_runs=[_convert(asdict(r)) for r in null_runs],
        null_r3_hit_mean=round(r3_mean, 4),
        null_r3_hit_std=round(r3_std, 4),
        null_r3_hit_min=round(r3_min, 4),
        null_r3_hit_max=round(r3_max, 4),
        null_naive_original_mean=round(naive_orig_mean, 4),
        separation_sigma=round(separation, 2),
        selectivity_vs_null=round(selectivity_vs_null, 2),
        gate_passed=gate_passed,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'null_corpus.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)
    print(f"\n  → {out_path}")
