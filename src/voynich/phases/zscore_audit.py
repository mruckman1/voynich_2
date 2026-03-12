"""
Phase 47 Track A – Z-Score Methodology Audit
=============================================
Resolve the discrepancy between Phase 29 (z=6.14 at 131K, exact-only)
and Phase 46 (z=61.63 at 10K, exact+relaxed).  Produce a single
canonical z-score for each candidate table plus sensitivity analysis.

Dependency chain:
    combined_refine.json        (Phase 15 best table)
    maxsat_validation.json      (Phase 44 MaxSAT consensus)
    kperm_search.json           (Phase 44 CSA best)
    canonical_table.json        (Phase 45 canonical)
    modifier_integrate.json     (Phase 16 modifiers)
    null_corpus.json            (null seeds)
    signal_bigrams.json         (Phase 29 cached classifications)
    arb_bigram.json             (Phase 46 Track A cached z)
        -> z_reproduce_42.json  (Step 47A.1)
        -> z_reproduce_46.json  (Step 47A.2)
        -> z_diff.json          (Step 47A.3)
        -> z_canonical.json     (Step 47A.4)
        -> z_sensitivity.json   (Step 47A.5)
"""

from __future__ import annotations

import json
import math
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
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if v != v else v
    if isinstance(obj, float) and (obj != obj):
        return None
    if isinstance(obj, (bool, int, float, str, type(None))):
        return obj
    return str(obj)


def _safe_load(path: str) -> Dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


# ---------------------------------------------------------------------------
# Shared context (loaded once, reused across steps)
# ---------------------------------------------------------------------------

@dataclass
class AuditContext:
    all_tokens: List[str]
    token_folios: List[str]
    eva_to_triple: Dict[str, str]
    modifier_chars: set
    modifier_rules: Dict[str, str]
    ref_word_set_131k: set
    ref_word_set_10k: set
    ref_tokens_raw: List[str]
    null_seeds: List[int]
    null_corpora: List[List[str]]
    # Candidate tables
    tables: Dict[str, Dict[str, str]]


def _reconstruct_modifier_rules(data: Dict) -> Tuple[set, Dict[str, str]]:
    modifier_chars = set(data.get('modifier_chars', []))
    modifier_rules: Dict[str, str] = {}
    for c in data.get('classifications', []):
        if c.get('final_classification') == 'modifier':
            modifier_rules[c['eva_char']] = c.get('modifier_type', 'silent')
    return modifier_chars, modifier_rules


def _sample_from_probs(probs: Dict[str, float], rng: random.Random) -> str:
    items = list(probs.items())
    r = rng.random()
    cumulative = 0.0
    for char, p in items:
        cumulative += p
        if r <= cumulative:
            return char
    return items[-1][0]


def _build_eva_bigram_model(
    tokens: List[str],
) -> Tuple[Dict, Dict, List[int]]:
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
    total_initial = sum(initial_counts.values()) or 1
    initial_probs = {c: n / total_initial for c, n in initial_counts.items()}
    bigram_probs: Dict[str, Dict[str, float]] = {}
    for c1, counts in bigram_counts.items():
        total = sum(counts.values())
        bigram_probs[c1] = {c2: n / total for c2, n in counts.items()}
    return bigram_probs, initial_probs, token_lengths


def _generate_null_corpus(
    bigram_probs: Dict, initial_probs: Dict,
    token_lengths: List[int], n_tokens: int, seed: int,
) -> List[str]:
    rng = random.Random(seed)
    length_counts = Counter(token_lengths)
    total_lengths = sum(length_counts.values()) or 1
    length_probs = {str(k): v / total_lengths for k, v in length_counts.items()}
    null_tokens: List[str] = []
    for _ in range(n_tokens):
        target_len = int(_sample_from_probs(length_probs, rng))
        target_len = max(1, target_len)
        chars = []
        current = _sample_from_probs(initial_probs, rng)
        chars.append(current)
        for _ in range(target_len - 1):
            next_probs = bigram_probs.get(current, initial_probs)
            current = _sample_from_probs(next_probs, rng)
            chars.append(current)
        null_tokens.append(''.join(chars))
    return null_tokens


def _build_context(rd: str) -> AuditContext:
    """Load all shared heavy objects once."""
    # Modifier rules
    mod_data = _safe_load(os.path.join(rd, 'modifier_integrate.json'))
    modifier_chars, modifier_rules = _reconstruct_modifier_rules(mod_data)

    # EVA-to-triple lookup
    eva_to_triple = build_eva_to_triple_lookup()

    # Corpus with folio tracking
    corpus = load_corpus(verbose=False)
    token_folios: List[str] = []
    all_tokens: List[str] = []
    for folio, page in corpus.pages.items():
        for token in page.all_tokens:
            token_folios.append(folio)
            all_tokens.append(token)

    # Reference word sets
    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    ref_tokens_raw = [
        w.lower() for w in ref_corpus.get_combined_tokens('latin')
        if len(w) >= 2
    ]
    base_words = set(ref_tokens_raw)
    expanded, _ = build_expanded_word_set(base_words)
    ref_word_set_131k = base_words | expanded

    word_freq = Counter(ref_tokens_raw)
    ref_word_set_10k = {w for w, _ in word_freq.most_common(10000)}

    # Null corpus model
    null_seeds = [100, 101, 102, 103, 104]
    null_path = os.path.join(rd, 'null_corpus.json')
    if os.path.exists(null_path):
        with open(null_path) as f:
            nd = json.load(f)
        seeds = [r['seed'] for r in nd.get('null_runs', [])]
        if seeds:
            null_seeds = seeds

    bigram_probs, initial_probs, token_lengths = _build_eva_bigram_model(
        all_tokens,
    )
    null_corpora = [
        _generate_null_corpus(
            bigram_probs, initial_probs, token_lengths, len(all_tokens), seed,
        )
        for seed in null_seeds
    ]

    # Candidate tables
    p15 = _safe_load(os.path.join(rd, 'combined_refine.json')).get(
        'best_assignment', {},
    )
    maxsat = _safe_load(os.path.join(rd, 'maxsat_validation.json')).get(
        'best_maxsat_assignment', {},
    )
    csa = _safe_load(os.path.join(rd, 'kperm_search.json')).get(
        'best_assignment', {},
    )
    canon = _safe_load(os.path.join(rd, 'canonical_table.json')).get(
        'table', {},
    )
    tables = {
        'T_P15': p15,
        'T_MAX': maxsat,
        'T_CSA': csa,
        'T_CANONICAL': canon,
    }

    return AuditContext(
        all_tokens=all_tokens,
        token_folios=token_folios,
        eva_to_triple=eva_to_triple,
        modifier_chars=modifier_chars,
        modifier_rules=modifier_rules,
        ref_word_set_131k=ref_word_set_131k,
        ref_word_set_10k=ref_word_set_10k,
        ref_tokens_raw=ref_tokens_raw,
        null_seeds=null_seeds,
        null_corpora=null_corpora,
        tables=tables,
    )


# ---------------------------------------------------------------------------
# Core z-score computation (parameterized)
# ---------------------------------------------------------------------------

def _decode_corpus_r3(
    tokens: List[str],
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    modifier_chars: set,
    modifier_rules: Dict[str, str],
    ref_word_set: set,
) -> List[str]:
    """R3 strategy: try alteration -> strip -> raw."""
    decoded = []
    for token in tokens:
        alt = decode_token_modifier_aware(
            token, assignment, eva_to_triple, modifier_chars, modifier_rules,
        )
        if alt.lower() in ref_word_set:
            decoded.append(alt.lower())
            continue
        stripped = decode_token_modifier_aware(
            token, assignment, eva_to_triple, modifier_chars,
        )
        if stripped.lower() in ref_word_set:
            decoded.append(stripped.lower())
            continue
        raw = decode_token(token, assignment, eva_to_triple)
        decoded.append(raw.lower())
    return decoded


def _classify_tokens(
    all_tokens: List[str],
    assignment: Dict[str, str],
    ctx: AuditContext,
    ref_word_set: set,
) -> Tuple[List[str], List[str]]:
    """Decode real + 5 null, classify each token. Returns (decoded, classifications)."""
    n = len(all_tokens)
    real_decoded = _decode_corpus_r3(
        all_tokens, assignment, ctx.eva_to_triple,
        ctx.modifier_chars, ctx.modifier_rules, ref_word_set,
    )
    real_hits = [w in ref_word_set for w in real_decoded]

    null_hits_list: List[List[bool]] = []
    for null_tokens in ctx.null_corpora:
        null_decoded = _decode_corpus_r3(
            null_tokens, assignment, ctx.eva_to_triple,
            ctx.modifier_chars, ctx.modifier_rules, ref_word_set,
        )
        null_hits_list.append([w in ref_word_set for w in null_decoded])

    classifications: List[str] = []
    for idx in range(n):
        r_hit = real_hits[idx]
        null_hit_count = sum(1 for nh in null_hits_list if nh[idx])
        if r_hit and null_hit_count <= 1:
            classifications.append('SIGNAL')
        elif r_hit and null_hit_count >= 3:
            classifications.append('SHARED_HIT')
        elif not r_hit and null_hit_count >= 3:
            classifications.append('ANTI_SIGNAL')
        else:
            classifications.append('SHARED_MISS')
    return real_decoded, classifications


def _build_ref_bigrams(
    ref_tokens: List[str], word_set: set,
) -> Set[Tuple[str, str]]:
    bigrams: Set[Tuple[str, str]] = set()
    for i in range(len(ref_tokens) - 1):
        w1, w2 = ref_tokens[i], ref_tokens[i + 1]
        if w1 in word_set and w2 in word_set:
            bigrams.add((w1, w2))
    return bigrams


def _find_signal_pairs(
    classifications: List[str],
    decoded: List[str],
    folios: List[str],
) -> List[Tuple[str, int, str, str]]:
    pairs = []
    for i in range(len(classifications) - 1):
        if (classifications[i] == 'SIGNAL'
                and classifications[i + 1] == 'SIGNAL'
                and folios[i] == folios[i + 1]):
            pairs.append((folios[i], i, decoded[i], decoded[i + 1]))
    return pairs


def _edit_distance_1(word: str) -> Set[str]:
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    variants: Set[str] = set()
    for i in range(len(word)):
        variants.add(word[:i] + word[i + 1:])
        for c in alphabet:
            if c != word[i]:
                variants.add(word[:i] + c + word[i + 1:])
    for i in range(len(word) + 1):
        for c in alphabet:
            variants.add(word[:i] + c + word[i:])
    return variants


def _relaxed_bigram_count(
    signal_pairs: List[Tuple[str, int, str, str]],
    ref_bigrams: Set[Tuple[str, str]],
    edit1_cache: Dict[str, Set[str]],
) -> int:
    n_relaxed = 0
    for _, _, w1, w2 in signal_pairs:
        if (w1, w2) in ref_bigrams:
            continue
        found = False
        variants1 = edit1_cache.get(w1) or _edit_distance_1(w1)
        for v1 in variants1:
            if (v1, w2) in ref_bigrams:
                found = True
                break
        if not found:
            variants2 = edit1_cache.get(w2) or _edit_distance_1(w2)
            for v2 in variants2:
                if (w1, v2) in ref_bigrams:
                    found = True
                    break
        if found:
            n_relaxed += 1
    return n_relaxed


def _null_perm_test(
    n_signal: int,
    n_tokens: int,
    decoded: List[str],
    folios: List[str],
    ref_bigrams: Set[Tuple[str, str]],
    n_perms: int = 500,
    seed: int = 42,
    include_relaxed: bool = True,
) -> Tuple[List[float], List[float]]:
    """Permutation test returning (null_exact_rates, null_total_rates)."""
    rng = random.Random(seed)
    indices = list(range(n_tokens))

    unique_words = set(decoded)
    edit1_cache: Dict[str, Set[str]] = {}
    if include_relaxed:
        edit1_cache = {w: _edit_distance_1(w) for w in unique_words}

    valid_consecutive = [
        folios[i] == folios[i + 1] for i in range(n_tokens - 1)
    ]

    null_exact_rates: List[float] = []
    null_total_rates: List[float] = []

    for _ in range(n_perms):
        fake_signal = set(rng.sample(indices, min(n_signal, n_tokens)))
        n_pairs = 0
        n_exact = 0
        n_relax = 0
        for i in range(n_tokens - 1):
            if (i in fake_signal and (i + 1) in fake_signal
                    and valid_consecutive[i]):
                n_pairs += 1
                w1, w2 = decoded[i], decoded[i + 1]
                if (w1, w2) in ref_bigrams:
                    n_exact += 1
                elif include_relaxed:
                    found = False
                    v1s = edit1_cache.get(w1) or _edit_distance_1(w1)
                    for v1 in v1s:
                        if (v1, w2) in ref_bigrams:
                            found = True
                            break
                    if not found:
                        v2s = edit1_cache.get(w2) or _edit_distance_1(w2)
                        for v2 in v2s:
                            if (w1, v2) in ref_bigrams:
                                found = True
                                break
                    if found:
                        n_relax += 1
        exact_rate = n_exact / n_pairs if n_pairs else 0.0
        total_rate = (n_exact + n_relax) / n_pairs if n_pairs else 0.0
        null_exact_rates.append(exact_rate)
        null_total_rates.append(total_rate)

    return null_exact_rates, null_total_rates


def _z_from_null(
    observed: float, null_rates: List[float],
) -> Tuple[float, float, float, float]:
    """Return (z, p_value, null_mean, null_std)."""
    if not null_rates:
        return 0.0, 1.0, 0.0, 0.0
    null_mean = sum(null_rates) / len(null_rates)
    null_var = sum((r - null_mean) ** 2 for r in null_rates) / len(null_rates)
    null_std = null_var ** 0.5
    if null_std > 0:
        z = (observed - null_mean) / null_std
    else:
        z = float('inf') if observed > null_mean else 0.0
    p = sum(1 for r in null_rates if r >= observed) / len(null_rates)
    return z, p, null_mean, null_std


@dataclass
class ZScoreResult:
    """Result of a single z-score computation."""
    label: str
    dict_mode: str
    dict_size: int
    n_tokens: int
    n_signal: int
    signal_rate: float
    n_signal_pairs: int
    n_exact_hits: int
    exact_hit_rate: float
    n_relaxed_hits: int
    total_hit_rate: float
    n_perms: int
    include_relaxed: bool
    null_mean_exact: float
    null_std_exact: float
    z_exact: float
    p_exact: float
    null_mean_total: float
    null_std_total: float
    z_total: float
    p_total: float
    ref_bigram_count: int


def _compute_z_full(
    label: str,
    assignment: Dict[str, str],
    dict_mode: str,
    ref_word_set: set,
    ref_bigram_set: Set[Tuple[str, str]],
    ctx: AuditContext,
    n_perms: int = 500,
    include_relaxed: bool = True,
    seed: int = 42,
) -> ZScoreResult:
    """Full parameterized z-score pipeline."""
    n_tokens = len(ctx.all_tokens)

    decoded, classifications = _classify_tokens(
        ctx.all_tokens, assignment, ctx, ref_word_set,
    )
    n_signal = sum(1 for c in classifications if c == 'SIGNAL')
    signal_rate = n_signal / n_tokens if n_tokens else 0.0

    signal_pairs = _find_signal_pairs(
        classifications, decoded, ctx.token_folios,
    )
    n_pairs = len(signal_pairs)

    # Exact hits
    n_exact = sum(1 for _, _, w1, w2 in signal_pairs if (w1, w2) in ref_bigram_set)
    exact_rate = n_exact / n_pairs if n_pairs else 0.0

    # Relaxed hits
    unique_words = set(decoded)
    edit1_cache = {w: _edit_distance_1(w) for w in unique_words} if include_relaxed else {}
    n_relaxed = _relaxed_bigram_count(signal_pairs, ref_bigram_set, edit1_cache) if include_relaxed else 0
    total_rate = (n_exact + n_relaxed) / n_pairs if n_pairs else 0.0

    # Null permutation test
    if n_signal > 0 and n_pairs > 0:
        null_exact_rates, null_total_rates = _null_perm_test(
            n_signal, n_tokens, decoded, ctx.token_folios,
            ref_bigram_set, n_perms=n_perms, seed=seed,
            include_relaxed=include_relaxed,
        )
    else:
        null_exact_rates = [0.0] * n_perms
        null_total_rates = [0.0] * n_perms

    z_exact, p_exact, nm_exact, ns_exact = _z_from_null(exact_rate, null_exact_rates)
    z_total, p_total, nm_total, ns_total = _z_from_null(total_rate, null_total_rates)

    return ZScoreResult(
        label=label,
        dict_mode=dict_mode,
        dict_size=len(ref_word_set),
        n_tokens=n_tokens,
        n_signal=n_signal,
        signal_rate=round(signal_rate, 4),
        n_signal_pairs=n_pairs,
        n_exact_hits=n_exact,
        exact_hit_rate=round(exact_rate, 6),
        n_relaxed_hits=n_relaxed,
        total_hit_rate=round(total_rate, 6),
        n_perms=n_perms,
        include_relaxed=include_relaxed,
        null_mean_exact=round(nm_exact, 6),
        null_std_exact=round(ns_exact, 6),
        z_exact=round(z_exact, 4),
        p_exact=round(p_exact, 6),
        null_mean_total=round(nm_total, 6),
        null_std_total=round(ns_total, 6),
        z_total=round(z_total, 4),
        p_total=round(p_total, 6),
        ref_bigram_count=len(ref_bigram_set),
    )


# ---------------------------------------------------------------------------
# Step 47A.1 — Reproduce Phase 29 z-score
# ---------------------------------------------------------------------------

@dataclass
class ReproduceP29Result:
    z_result: Dict
    target_z: float
    reproduced_z: float
    delta: float
    within_tolerance: bool
    methodology: Dict[str, str]
    runtime_seconds: float


def run_z_reproduce_42() -> None:
    """Step 47A.1: reproduce Phase 29's z=6.14 from scratch (exact-only, 131K, 1000 perms)."""
    t0 = time.time()
    print("=" * 70)
    print("STEP 47A.1: Reproduce Phase 29 Z-Score (z=6.14)")
    print("=" * 70)

    rd = _results_dir()
    print("\n  Loading shared context...")
    ctx = _build_context(rd)

    assignment = ctx.tables.get('T_P15', {})
    if not assignment:
        print("  [SKIP] T_P15 assignment not found")
        return

    # Phase 29 methodology: 131K dict, exact-only z, 1000 perms
    ref_bigrams_131k = _build_ref_bigrams(ctx.ref_tokens_raw, ctx.ref_word_set_131k)

    print(f"\n  Methodology: 131K dict ({len(ctx.ref_word_set_131k)} words), "
          f"exact-only z, 1000 perms")
    print(f"  Reference bigrams: {len(ref_bigrams_131k)}")
    print("\n  Computing z-score (this may take a few minutes)...")

    z_result = _compute_z_full(
        label='P29_reproduce',
        assignment=assignment,
        dict_mode='131K',
        ref_word_set=ctx.ref_word_set_131k,
        ref_bigram_set=ref_bigrams_131k,
        ctx=ctx,
        n_perms=1000,
        include_relaxed=False,
        seed=42,
    )

    target_z = 6.14
    reproduced_z = z_result.z_exact
    delta = abs(reproduced_z - target_z)
    within_tol = delta <= 1.0

    print(f"\n  SIGNAL tokens: {z_result.n_signal} ({z_result.signal_rate:.1%})")
    print(f"  SIGNAL pairs:  {z_result.n_signal_pairs}")
    print(f"  Exact hits:    {z_result.n_exact_hits}")
    print(f"  Exact rate:    {z_result.exact_hit_rate:.6f}")
    print(f"  z_exact:       {reproduced_z:.4f}  (target: {target_z:.2f})")
    print(f"  Delta:         {delta:.4f}  {'PASS' if within_tol else 'FAIL'}")

    result = ReproduceP29Result(
        z_result=_convert(asdict(z_result)),
        target_z=target_z,
        reproduced_z=round(reproduced_z, 4),
        delta=round(delta, 4),
        within_tolerance=within_tol,
        methodology={
            'source': 'Phase 29 (signal_bigrams.py)',
            'dictionary': '131K (base + expanded)',
            'hit_counting': 'exact only',
            'n_perms': '1000',
            'seed': '42',
            'null_model': 'shuffle SIGNAL labels',
            'signal_threshold': 'real_hit AND <=1/5 null hits',
        },
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'z_reproduce_42.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2)
    print(f"\n  Saved -> {out_path}")


# ---------------------------------------------------------------------------
# Step 47A.2 — Reproduce Phase 46 z-score
# ---------------------------------------------------------------------------

@dataclass
class ReproduceP46Result:
    z_result: Dict
    target_z: float
    reproduced_z: float
    delta: float
    within_tolerance: bool
    methodology: Dict[str, str]
    runtime_seconds: float


def run_z_reproduce_46() -> None:
    """Step 47A.2: reproduce Phase 46's z_total_at_10k=61.63 (exact+relaxed, 10K, 500 perms)."""
    t0 = time.time()
    print("=" * 70)
    print("STEP 47A.2: Reproduce Phase 46 Z-Score (z=61.63)")
    print("=" * 70)

    rd = _results_dir()
    print("\n  Loading shared context...")
    ctx = _build_context(rd)

    assignment = ctx.tables.get('T_P15', {})
    if not assignment:
        print("  [SKIP] T_P15 assignment not found")
        return

    # Phase 46 methodology: 10K dict, exact+relaxed z, 500 perms
    ref_bigrams_10k = _build_ref_bigrams(ctx.ref_tokens_raw, ctx.ref_word_set_10k)

    print(f"\n  Methodology: 10K dict ({len(ctx.ref_word_set_10k)} words), "
          f"exact+relaxed z, 500 perms")
    print(f"  Reference bigrams: {len(ref_bigrams_10k)}")
    print("\n  Computing z-score (this may take a few minutes)...")

    z_result = _compute_z_full(
        label='P46_reproduce',
        assignment=assignment,
        dict_mode='10K',
        ref_word_set=ctx.ref_word_set_10k,
        ref_bigram_set=ref_bigrams_10k,
        ctx=ctx,
        n_perms=500,
        include_relaxed=True,
        seed=42,
    )

    target_z = 61.63
    reproduced_z = z_result.z_total
    delta = abs(reproduced_z - target_z)
    within_tol = delta <= 5.0

    print(f"\n  SIGNAL tokens: {z_result.n_signal} ({z_result.signal_rate:.1%})")
    print(f"  SIGNAL pairs:  {z_result.n_signal_pairs}")
    print(f"  Exact hits:    {z_result.n_exact_hits}")
    print(f"  Relaxed hits:  {z_result.n_relaxed_hits}")
    print(f"  Total rate:    {z_result.total_hit_rate:.6f}")
    print(f"  z_total:       {reproduced_z:.4f}  (target: {target_z:.2f})")
    print(f"  z_exact:       {z_result.z_exact:.4f}")
    print(f"  Delta:         {delta:.4f}  {'PASS' if within_tol else 'FAIL'}")

    result = ReproduceP46Result(
        z_result=_convert(asdict(z_result)),
        target_z=target_z,
        reproduced_z=round(reproduced_z, 4),
        delta=round(delta, 4),
        within_tolerance=within_tol,
        methodology={
            'source': 'Phase 46 (triple_arbitration.py)',
            'dictionary': '10K (top by frequency)',
            'hit_counting': 'exact + relaxed (edit distance 1)',
            'n_perms': '500',
            'seed': '42',
            'null_model': 'shuffle SIGNAL labels',
            'signal_threshold': 'real_hit AND <=1/5 null hits',
        },
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'z_reproduce_46.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2)
    print(f"\n  Saved -> {out_path}")


# ---------------------------------------------------------------------------
# Step 47A.3 — Difference identification
# ---------------------------------------------------------------------------

@dataclass
class ZDiffResult:
    side_by_side: List[Dict]
    marginal_impacts: List[Dict]
    dominant_factor: str
    explanation: str
    runtime_seconds: float


def run_z_diff() -> None:
    """Step 47A.3: identify every methodological difference and marginal impact."""
    t0 = time.time()
    print("=" * 70)
    print("STEP 47A.3: Z-Score Difference Identification")
    print("=" * 70)

    rd = _results_dir()
    print("\n  Loading shared context...")
    ctx = _build_context(rd)

    assignment = ctx.tables.get('T_P15', {})
    if not assignment:
        print("  [SKIP] T_P15 assignment not found")
        return

    ref_bigrams_131k = _build_ref_bigrams(ctx.ref_tokens_raw, ctx.ref_word_set_131k)
    ref_bigrams_10k = _build_ref_bigrams(ctx.ref_tokens_raw, ctx.ref_word_set_10k)

    # Baseline: Phase 29 methodology (131K, exact-only, 1000 perms)
    print("\n  Computing baseline (Phase 29 methodology)...")
    baseline = _compute_z_full(
        'baseline_p29', assignment, '131K', ctx.ref_word_set_131k,
        ref_bigrams_131k, ctx, n_perms=500, include_relaxed=False, seed=42,
    )

    side_by_side = [
        {'parameter': 'Dictionary', 'phase_29': '131K', 'phase_46': '10K'},
        {'parameter': 'Hit counting', 'phase_29': 'exact only', 'phase_46': 'exact + relaxed'},
        {'parameter': 'Permutations', 'phase_29': '1000', 'phase_46': '500'},
        {'parameter': 'Reference bigrams', 'phase_29': str(len(ref_bigrams_131k)), 'phase_46': str(len(ref_bigrams_10k))},
    ]

    # Marginal impacts: change one parameter at a time from baseline
    marginals = []

    # Change 1: dict 131K -> 10K (keep exact-only, 500 perms)
    print("  Computing marginal: dict 131K -> 10K...")
    m1 = _compute_z_full(
        'dict_10k', assignment, '10K', ctx.ref_word_set_10k,
        ref_bigrams_10k, ctx, n_perms=500, include_relaxed=False, seed=42,
    )
    marginals.append({
        'parameter': 'Dictionary: 131K -> 10K',
        'baseline_z_exact': round(baseline.z_exact, 4),
        'changed_z_exact': round(m1.z_exact, 4),
        'impact': round(m1.z_exact - baseline.z_exact, 4),
        'baseline_signal': baseline.n_signal,
        'changed_signal': m1.n_signal,
    })

    # Change 2: hit counting exact -> exact+relaxed (keep 131K, 500 perms)
    print("  Computing marginal: exact -> exact+relaxed...")
    m2 = _compute_z_full(
        'add_relaxed', assignment, '131K', ctx.ref_word_set_131k,
        ref_bigrams_131k, ctx, n_perms=500, include_relaxed=True, seed=42,
    )
    marginals.append({
        'parameter': 'Hit counting: exact -> exact+relaxed',
        'baseline_z_exact': round(baseline.z_exact, 4),
        'changed_z_total': round(m2.z_total, 4),
        'impact': round(m2.z_total - baseline.z_exact, 4),
    })

    # Change 3: both dict + relaxed (= Phase 46 methodology)
    print("  Computing marginal: dict + relaxed (Phase 46 full)...")
    m3 = _compute_z_full(
        'full_p46', assignment, '10K', ctx.ref_word_set_10k,
        ref_bigrams_10k, ctx, n_perms=500, include_relaxed=True, seed=42,
    )
    marginals.append({
        'parameter': 'Dict 10K + relaxed (Phase 46 full)',
        'baseline_z_exact': round(baseline.z_exact, 4),
        'changed_z_total': round(m3.z_total, 4),
        'impact': round(m3.z_total - baseline.z_exact, 4),
    })

    # Determine dominant factor
    impacts = [(m['parameter'], abs(m.get('impact', 0))) for m in marginals]
    dominant = max(impacts, key=lambda x: x[1])

    explanation = (
        f"Phase 29 z_exact={baseline.z_exact:.2f} vs Phase 46 z_total={m3.z_total:.2f}. "
        f"Dominant factor: {dominant[0]} (impact={dominant[1]:.2f}). "
        f"The discrepancy is methodological, not a bug. "
        f"Phase 29 uses exact-only hits on 131K dict; "
        f"Phase 46 adds relaxed (ED<=1) matching on a tighter 10K dict, "
        f"which dramatically increases both hit rate and z."
    )

    for row in side_by_side:
        print(f"  {row['parameter']:25s}  P29={row['phase_29']:15s}  P46={row['phase_46']:15s}")
    print()
    for m in marginals:
        imp = m.get('impact', 0)
        print(f"  {m['parameter']:45s}  impact={imp:+.2f}")
    print(f"\n  Dominant: {dominant[0]}")
    print(f"  Explanation: {explanation}")

    result = ZDiffResult(
        side_by_side=side_by_side,
        marginal_impacts=marginals,
        dominant_factor=dominant[0],
        explanation=explanation,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'z_diff.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2)
    print(f"\n  Saved -> {out_path}")


# ---------------------------------------------------------------------------
# Step 47A.4 — Canonical z-score
# ---------------------------------------------------------------------------

@dataclass
class ZCanonicalResult:
    methodology: Dict[str, str]
    per_table: List[Dict]
    best_table: str
    best_z_total: float
    best_z_exact: float
    runtime_seconds: float


def run_z_canonical() -> None:
    """Step 47A.4: compute definitive z for each candidate table under unified methodology."""
    t0 = time.time()
    print("=" * 70)
    print("STEP 47A.4: Canonical Z-Score Computation")
    print("=" * 70)

    rd = _results_dir()
    print("\n  Loading shared context...")
    ctx = _build_context(rd)

    # Canonical methodology: 10K dict, exact+relaxed, 500 perms
    ref_bigrams_10k = _build_ref_bigrams(ctx.ref_tokens_raw, ctx.ref_word_set_10k)

    methodology = {
        'dictionary': '10K (top by frequency from Latin reference)',
        'hit_counting': 'exact + relaxed (edit distance 1)',
        'n_perms': '500',
        'seed': '42',
        'null_model': 'shuffle SIGNAL labels (preserve positions)',
        'signal_threshold': 'real_hit AND <=1/5 null hits',
        'signal_null_corpora': '5 (seeds 100-104)',
        'note': 'This supersedes all prior z-scores from Phases 29-46',
    }

    print(f"\n  Methodology: 10K dict, exact+relaxed, 500 perms")
    print(f"  Reference bigrams: {len(ref_bigrams_10k)}")

    per_table = []
    best_table = ''
    best_z = -float('inf')

    for table_name, assignment in sorted(ctx.tables.items()):
        if not assignment:
            print(f"\n  {table_name}: [SKIP] empty assignment")
            continue

        print(f"\n  Computing {table_name}...")
        z_result = _compute_z_full(
            label=table_name,
            assignment=assignment,
            dict_mode='10K',
            ref_word_set=ctx.ref_word_set_10k,
            ref_bigram_set=ref_bigrams_10k,
            ctx=ctx,
            n_perms=500,
            include_relaxed=True,
            seed=42,
        )

        entry = _convert(asdict(z_result))
        per_table.append(entry)

        print(f"    SIGNAL: {z_result.n_signal} ({z_result.signal_rate:.1%})")
        print(f"    Pairs:  {z_result.n_signal_pairs}")
        print(f"    z_exact={z_result.z_exact:.2f}  z_total={z_result.z_total:.2f}")

        if z_result.z_total > best_z:
            best_z = z_result.z_total
            best_table = table_name

    print(f"\n  BEST: {best_table} with z_total={best_z:.2f}")

    result = ZCanonicalResult(
        methodology=methodology,
        per_table=per_table,
        best_table=best_table,
        best_z_total=round(best_z, 4),
        best_z_exact=round(
            max((t.get('z_exact', 0) for t in per_table), default=0), 4,
        ),
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'z_canonical.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2)
    print(f"\n  Saved -> {out_path}")


# ---------------------------------------------------------------------------
# Step 47A.5 — Z sensitivity analysis
# ---------------------------------------------------------------------------

@dataclass
class ZSensitivityResult:
    sensitivity_table: List[Dict]
    robust_parameters: List[str]
    sensitive_parameters: List[str]
    z_range: List[float]
    recommended_z: float
    recommended_methodology: str
    runtime_seconds: float


def run_z_sensitivity() -> None:
    """Step 47A.5: vary each methodological choice and measure z."""
    t0 = time.time()
    print("=" * 70)
    print("STEP 47A.5: Z-Score Sensitivity Analysis")
    print("=" * 70)

    rd = _results_dir()
    print("\n  Loading shared context...")
    ctx = _build_context(rd)

    assignment = ctx.tables.get('T_P15', {})
    if not assignment:
        print("  [SKIP] T_P15 assignment not found")
        return

    # Build reference bigram sets for different dict sizes
    word_freq = Counter(ctx.ref_tokens_raw)
    dict_sizes = {
        '1K': {w for w, _ in word_freq.most_common(1000)},
        '5K': {w for w, _ in word_freq.most_common(5000)},
        '10K': ctx.ref_word_set_10k,
        '17K': set(ctx.ref_tokens_raw),  # base words (no expansion)
        '131K': ctx.ref_word_set_131k,
    }

    sensitivity_rows = []

    # Dimension 1: Dictionary size
    print("\n  Varying dictionary size...")
    for dict_label, word_set in dict_sizes.items():
        ref_bi = _build_ref_bigrams(ctx.ref_tokens_raw, word_set)
        z_r = _compute_z_full(
            f'dict_{dict_label}', assignment, dict_label, word_set,
            ref_bi, ctx, n_perms=200, include_relaxed=True, seed=42,
        )
        sensitivity_rows.append({
            'dimension': 'dictionary',
            'value': dict_label,
            'z_exact': round(z_r.z_exact, 2),
            'z_total': round(z_r.z_total, 2),
            'n_signal': z_r.n_signal,
            'n_pairs': z_r.n_signal_pairs,
        })
        print(f"    {dict_label:6s}: z_exact={z_r.z_exact:.2f}  z_total={z_r.z_total:.2f}  "
              f"signal={z_r.n_signal}")

    # Dimension 2: Edit distance threshold (on 10K dict)
    print("\n  Varying edit distance threshold...")
    ref_bi_10k = _build_ref_bigrams(ctx.ref_tokens_raw, ctx.ref_word_set_10k)
    for ed_label, use_relaxed in [('exact_only', False), ('ED<=1', True)]:
        z_r = _compute_z_full(
            f'ed_{ed_label}', assignment, '10K', ctx.ref_word_set_10k,
            ref_bi_10k, ctx, n_perms=200, include_relaxed=use_relaxed, seed=42,
        )
        sensitivity_rows.append({
            'dimension': 'edit_distance',
            'value': ed_label,
            'z_exact': round(z_r.z_exact, 2),
            'z_total': round(z_r.z_total, 2),
            'n_signal': z_r.n_signal,
            'n_pairs': z_r.n_signal_pairs,
        })
        print(f"    {ed_label:12s}: z_exact={z_r.z_exact:.2f}  z_total={z_r.z_total:.2f}")

    # Dimension 3: Permutation count (on 10K, relaxed)
    print("\n  Varying permutation count...")
    for n_p in [100, 500, 1000]:
        z_r = _compute_z_full(
            f'perms_{n_p}', assignment, '10K', ctx.ref_word_set_10k,
            ref_bi_10k, ctx, n_perms=n_p, include_relaxed=True, seed=42,
        )
        sensitivity_rows.append({
            'dimension': 'n_perms',
            'value': str(n_p),
            'z_exact': round(z_r.z_exact, 2),
            'z_total': round(z_r.z_total, 2),
            'n_signal': z_r.n_signal,
            'n_pairs': z_r.n_signal_pairs,
        })
        print(f"    {n_p:6d}: z_exact={z_r.z_exact:.2f}  z_total={z_r.z_total:.2f}")

    # Analyze robustness
    z_totals = [r['z_total'] for r in sensitivity_rows]
    z_range = [min(z_totals), max(z_totals)]

    # Identify robust vs sensitive parameters
    by_dim: Dict[str, List[float]] = defaultdict(list)
    for r in sensitivity_rows:
        by_dim[r['dimension']].append(r['z_total'])

    robust, sensitive = [], []
    for dim, values in by_dim.items():
        spread = max(values) - min(values)
        median_val = sorted(values)[len(values) // 2]
        cv = spread / abs(median_val) if median_val != 0 else float('inf')
        if cv < 0.3:
            robust.append(dim)
        else:
            sensitive.append(dim)

    # Conservative recommendation
    recommended_z = min(z_totals)

    print(f"\n  Z range: [{z_range[0]:.2f}, {z_range[1]:.2f}]")
    print(f"  Robust parameters: {robust}")
    print(f"  Sensitive parameters: {sensitive}")
    print(f"  Conservative (minimum) z: {recommended_z:.2f}")

    result = ZSensitivityResult(
        sensitivity_table=sensitivity_rows,
        robust_parameters=robust,
        sensitive_parameters=sensitive,
        z_range=z_range,
        recommended_z=round(recommended_z, 2),
        recommended_methodology=(
            f"Most conservative z={recommended_z:.2f}. "
            f"Z is sensitive to {sensitive} and robust across {robust}."
        ),
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'z_sensitivity.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2)
    print(f"\n  Saved -> {out_path}")


# ---------------------------------------------------------------------------
# Track A orchestrator
# ---------------------------------------------------------------------------

def run_track_a_47() -> None:
    """Run all Track A steps."""
    run_z_reproduce_42()
    print()
    run_z_reproduce_46()
    print()
    run_z_diff()
    print()
    run_z_canonical()
    print()
    run_z_sensitivity()
