"""
Phase 46 Track A – Triple Arbitration
======================================
Arbitrate between Phase 15 and MaxSAT on 6 disputed triples using
the validated bigram z metric (Phase 42).

Dependency chain:
    combined_refine.json        (Phase 15 best table)
    maxsat_validation.json      (Phase 44 MaxSAT consensus)
    kperm_search.json           (Phase 44 CSA best)
    canonical_table.json        (Phase 45 canonical)
    modifier_integrate.json     (Phase 16 modifiers)
    null_corpus.json            (null seeds)
    signal_10k.json             (Phase 36 signal words)
    merged_dict.json            (Phase 38 merged 19K dict)
        -> arb_tables.json      (Step 46A.1)
        -> arb_bigram.json      (Step 46A.2)
        -> arb_signal.json      (Step 46A.3)
        -> arb_10k.json         (Step 46A.4)
        -> arb_selection.json   (Step 46A.5)
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
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
# Constants
# ---------------------------------------------------------------------------

BEDROCK_SIGNAL_WORDS = [
    'bene', 'codi', 'sero', 'sene', 'de', 'raro', 'dine', 'cola',
]
BOOTSTRAP_WORDS = ['ci', 'dico']

COMPOSITE_WEIGHTS = {
    'z_total_10k': 0.4,
    'selectivity_10k': 0.3,
    'signal_survival': 0.2,
    'dict_hit_10k': 0.1,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _convert(obj: Any) -> Any:
    """Recursively convert dataclass / numpy / NaN objects for JSON."""
    if hasattr(obj, '__dataclass_fields__'):
        return {k: _convert(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _convert(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_convert(item) for item in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
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
# Shared context (heavy objects loaded once)
# ---------------------------------------------------------------------------


@dataclass
class ArbitrationContext:
    """Heavy objects shared across all Track A steps."""
    all_tokens: List[str]
    token_folios: List[str]
    eva_to_triple: Dict[str, str]
    modifier_chars: set
    modifier_rules: Dict[str, str]
    ref_word_set_131k: set
    ref_word_set_10k: set
    ref_tokens: List[str]          # raw Latin reference token list
    null_seeds: List[int]
    bigram_probs: Dict
    initial_probs: Dict
    token_lengths: List[int]
    # Optional: merged 19K word set from Phase 38
    ref_word_set_19k: Optional[set] = None
    # Cached null corpus EVA tokens (generated once, decoded per-call)
    null_corpora: Optional[List[List[str]]] = None


def _reconstruct_modifier_rules(
    data: Dict,
) -> Tuple[set, Dict[str, str]]:
    """Extract modifier chars and rules from modifier_integrate.json."""
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
    bigram_probs: Dict,
    initial_probs: Dict,
    token_lengths: List[int],
    n_tokens: int,
    seed: int,
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


def _build_context(rd: str) -> ArbitrationContext:
    """Load all shared objects once."""
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

    # Reference word sets (10K and 131K)
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

    # Merged 19K dict (Phase 38)
    ref_word_set_19k = None
    merged_path = os.path.join(rd, 'merged_dict.json')
    if os.path.exists(merged_path):
        with open(merged_path) as f:
            md = json.load(f)
        merged_words = md.get('merged_words', [])
        if merged_words:
            ref_word_set_19k = set(w.lower() for w in merged_words)

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

    # Pre-generate null corpora (EVA token strings). These only depend
    # on the bigram model + seeds, not on any assignment or dictionary,
    # so we generate them once here and reuse across all _classify_tokens calls.
    n_corpus = len(all_tokens)
    null_corpora = [
        _generate_null_corpus(
            bigram_probs, initial_probs, token_lengths, n_corpus, seed,
        )
        for seed in null_seeds
    ]

    return ArbitrationContext(
        all_tokens=all_tokens,
        token_folios=token_folios,
        eva_to_triple=eva_to_triple,
        modifier_chars=modifier_chars,
        modifier_rules=modifier_rules,
        ref_word_set_131k=ref_word_set_131k,
        ref_word_set_10k=ref_word_set_10k,
        ref_tokens=ref_tokens_raw,
        null_seeds=null_seeds,
        bigram_probs=bigram_probs,
        initial_probs=initial_probs,
        token_lengths=token_lengths,
        ref_word_set_19k=ref_word_set_19k,
        null_corpora=null_corpora,
    )


# ---------------------------------------------------------------------------
# Decode helpers
# ---------------------------------------------------------------------------


def _decode_corpus_r3(
    tokens: List[str],
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    modifier_chars: set,
    modifier_rules: Dict[str, str],
    ref_word_set: set,
) -> List[str]:
    """R3 strategy: try alteration → strip → raw."""
    decoded = []
    for token in tokens:
        # Alteration (with modifier rules)
        alt = decode_token_modifier_aware(
            token, assignment, eva_to_triple, modifier_chars, modifier_rules,
        )
        if alt.lower() in ref_word_set:
            decoded.append(alt.lower())
            continue
        # Strip (modifiers stripped, no rules)
        stripped = decode_token_modifier_aware(
            token, assignment, eva_to_triple, modifier_chars,
        )
        if stripped.lower() in ref_word_set:
            decoded.append(stripped.lower())
            continue
        # Raw (no modifier processing)
        raw = decode_token(token, assignment, eva_to_triple)
        decoded.append(raw.lower())
    return decoded


# ---------------------------------------------------------------------------
# Bigram z helpers
# ---------------------------------------------------------------------------


def _classify_tokens(
    all_tokens: List[str],
    assignment: Dict[str, str],
    ctx: ArbitrationContext,
    ref_word_set: set,
) -> Tuple[List[str], List[str]]:
    """Decode real + 5 null, classify each token.

    Returns (decoded, classifications).
    """
    n = len(all_tokens)

    # Decode real corpus
    real_decoded = _decode_corpus_r3(
        all_tokens, assignment, ctx.eva_to_triple,
        ctx.modifier_chars, ctx.modifier_rules, ref_word_set,
    )
    real_hits = [w in ref_word_set for w in real_decoded]

    # Decode 5 null corpora (use cached EVA tokens from context)
    null_hits_list: List[List[bool]] = []
    null_token_lists = ctx.null_corpora if ctx.null_corpora else [
        _generate_null_corpus(
            ctx.bigram_probs, ctx.initial_probs,
            ctx.token_lengths, n, seed,
        )
        for seed in ctx.null_seeds
    ]
    for null_tokens in null_token_lists:
        null_decoded = _decode_corpus_r3(
            null_tokens, assignment, ctx.eva_to_triple,
            ctx.modifier_chars, ctx.modifier_rules, ref_word_set,
        )
        null_hits_list.append([w in ref_word_set for w in null_decoded])

    # Classify
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
    ref_tokens: List[str],
    word_set: set,
) -> Set[Tuple[str, str]]:
    """Build reference bigram set, filtered to word_set."""
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
    """Consecutive SIGNAL-SIGNAL pairs respecting folio boundaries."""
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
    edit1_cache: Optional[Dict[str, Set[str]]] = None,
) -> int:
    """Count non-exact SIGNAL pairs within edit distance 1 of a ref bigram."""
    n_relaxed = 0
    for _, _, w1, w2 in signal_pairs:
        if (w1, w2) in ref_bigrams:
            continue
        found = False
        variants1 = (edit1_cache[w1] if edit1_cache and w1 in edit1_cache
                      else _edit_distance_1(w1))
        for v1 in variants1:
            if (v1, w2) in ref_bigrams:
                found = True
                break
        if not found:
            variants2 = (edit1_cache[w2] if edit1_cache and w2 in edit1_cache
                          else _edit_distance_1(w2))
            for v2 in variants2:
                if (w1, v2) in ref_bigrams:
                    found = True
                    break
        if found:
            n_relaxed += 1
    return n_relaxed


def _is_relaxed_match_cached(
    w1: str, w2: str,
    ref_bigrams: Set[Tuple[str, str]],
    edit1_cache: Dict[str, Set[str]],
) -> bool:
    variants1 = edit1_cache.get(w1) or _edit_distance_1(w1)
    for v1 in variants1:
        if (v1, w2) in ref_bigrams:
            return True
    variants2 = edit1_cache.get(w2) or _edit_distance_1(w2)
    for v2 in variants2:
        if (w1, v2) in ref_bigrams:
            return True
    return False


def _null_perm_test(
    n_signal: int,
    n_tokens: int,
    decoded: List[str],
    folios: List[str],
    ref_bigrams: Set[Tuple[str, str]],
    n_perms: int = 500,
    seed: int = 42,
) -> Tuple[List[float], List[float]]:
    """Permutation test returning (null_exact_rates, null_total_rates)."""
    rng = random.Random(seed)
    indices = list(range(n_tokens))
    null_exact_rates: List[float] = []
    null_total_rates: List[float] = []

    # Precompute edit-distance-1 neighborhoods for all unique decoded words.
    # This is the critical optimization: avoids regenerating ~286 string
    # variants per word for every pair in every permutation.
    unique_words = set(decoded)
    edit1_cache: Dict[str, Set[str]] = {
        w: _edit_distance_1(w) for w in unique_words
    }

    # Precompute folio-boundary-valid consecutive pairs
    valid_consecutive = [
        folios[i] == folios[i + 1] for i in range(n_tokens - 1)
    ]

    for _ in range(n_perms):
        fake_signal = set(rng.sample(indices, min(n_signal, n_tokens)))
        n_pairs = 0
        n_exact = 0
        n_relaxed = 0
        for i in range(n_tokens - 1):
            if (i in fake_signal and (i + 1) in fake_signal
                    and valid_consecutive[i]):
                n_pairs += 1
                if (decoded[i], decoded[i + 1]) in ref_bigrams:
                    n_exact += 1
                elif _is_relaxed_match_cached(
                    decoded[i], decoded[i + 1],
                    ref_bigrams, edit1_cache,
                ):
                    n_relaxed += 1
        exact_rate = n_exact / n_pairs if n_pairs else 0.0
        total_rate = (n_exact + n_relaxed) / n_pairs if n_pairs else 0.0
        null_exact_rates.append(exact_rate)
        null_total_rates.append(total_rate)

    return null_exact_rates, null_total_rates


def _z_from_null(
    observed: float,
    null_rates: List[float],
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


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class TableBigramZ:
    table_name: str
    dict_mode: str
    n_tokens: int
    n_signal: int
    signal_rate: float
    n_signal_pairs: int
    n_bigram_hits_exact: int
    bigram_hit_rate_exact: float
    n_bigram_hits_relaxed: int
    bigram_hit_rate_total: float
    null_mean_exact: float
    null_std_exact: float
    z_exact: float
    p_value_exact: float
    null_mean_total: float
    null_std_total: float
    z_total: float
    p_value_total: float


@dataclass
class ArbTablesResult:
    tables: List[Dict]
    n_disputed: int
    disputed_triples: List[Dict]
    runtime_seconds: float


@dataclass
class ArbBigramResult:
    per_table: List[Dict]
    best6_decisions: List[Dict]
    best6_final_assignment: Dict[str, str]
    best_z_exact: str
    best_z_total: str
    runtime_seconds: float


@dataclass
class ArbSignalResult:
    per_table: List[Dict]
    best_survival: str
    runtime_seconds: float


@dataclass
class ArbDict10KResult:
    per_table: List[Dict]
    best_dict_hit: str
    best_selectivity: str
    runtime_seconds: float


@dataclass
class ArbSelectionResult:
    rankings: List[Dict]
    definitive_table_name: str
    definitive_assignment: Dict[str, str]
    composite_weights: Dict[str, float]
    vs_p15: Dict[str, float]
    vs_canonical: Dict[str, float]
    gate_passed: bool
    verdict: str
    rationale: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Candidate table assembly
# ---------------------------------------------------------------------------


def _build_candidate_tables(rd: str) -> List[Dict]:
    """Build 8 candidate tables as dicts with name, assignment, dict_mode."""
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

    # Identify disputed triples
    disputed = [k for k in sorted(p15.keys()) if p15.get(k) != maxsat.get(k)]

    # T_VOTE: majority vote across 4 sources
    vote = dict(p15)
    for tk in disputed:
        candidates = [
            p15.get(tk), maxsat.get(tk), csa.get(tk), canon.get(tk),
        ]
        counts = Counter(c for c in candidates if c)
        if counts:
            vote[tk] = counts.most_common(1)[0][0]

    # T_BEST6: placeholder (= P15), updated in Step 46A.2
    best6 = dict(p15)

    tables = [
        {'name': 'T_P15', 'assignment': p15, 'dict_mode': '131K',
         'provenance': 'Phase 15 best_assignment'},
        {'name': 'T_MAX', 'assignment': maxsat, 'dict_mode': '131K',
         'provenance': 'MaxSAT best assignment'},
        {'name': 'T_P15_10K', 'assignment': dict(p15), 'dict_mode': '10K',
         'provenance': 'Phase 15 scored against 10K dict'},
        {'name': 'T_MAX_10K', 'assignment': dict(maxsat), 'dict_mode': '10K',
         'provenance': 'MaxSAT scored against 10K dict'},
        {'name': 'T_BEST6', 'assignment': best6, 'dict_mode': '10K',
         'provenance': 'Per-triple best of P15/MaxSAT by z (placeholder)'},
        {'name': 'T_VOTE', 'assignment': vote, 'dict_mode': '131K',
         'provenance': 'Majority vote across P15/MaxSAT/CSA/Canonical'},
        {'name': 'T_CSA', 'assignment': csa, 'dict_mode': '131K',
         'provenance': 'CSA (kperm_search) best assignment'},
        {'name': 'T_CANONICAL', 'assignment': canon, 'dict_mode': '131K',
         'provenance': 'Phase 45 canonical table'},
    ]
    return tables


# ---------------------------------------------------------------------------
# Core bigram z computation for one table
# ---------------------------------------------------------------------------


def _compute_bigram_z(
    table_name: str,
    assignment: Dict[str, str],
    dict_mode: str,
    ctx: ArbitrationContext,
    ref_bigrams: Set[Tuple[str, str]],
    ref_word_set: set,
    n_perms: int = 500,
) -> TableBigramZ:
    """Full bigram z pipeline for one table + dictionary combination."""
    n_tokens = len(ctx.all_tokens)

    # Decode and classify
    decoded, classifications = _classify_tokens(
        ctx.all_tokens, assignment, ctx, ref_word_set,
    )
    n_signal = sum(1 for c in classifications if c == 'SIGNAL')
    signal_rate = n_signal / n_tokens if n_tokens else 0.0

    # Find SIGNAL-SIGNAL pairs
    signal_pairs = _find_signal_pairs(
        classifications, decoded, ctx.token_folios,
    )
    n_pairs = len(signal_pairs)

    # Precompute edit-distance-1 cache for all unique decoded words
    unique_words = set(decoded)
    edit1_cache: Dict[str, Set[str]] = {
        w: _edit_distance_1(w) for w in unique_words
    }

    # Exact bigram hits
    n_exact = sum(
        1 for _, _, w1, w2 in signal_pairs if (w1, w2) in ref_bigrams
    )
    exact_rate = n_exact / n_pairs if n_pairs else 0.0

    # Relaxed (edit-distance-1) bigram hits
    n_relaxed = _relaxed_bigram_count(signal_pairs, ref_bigrams, edit1_cache)
    total_rate = (n_exact + n_relaxed) / n_pairs if n_pairs else 0.0

    # Null permutation test
    if n_signal > 0 and n_pairs > 0:
        null_exact_rates, null_total_rates = _null_perm_test(
            n_signal, n_tokens, decoded, ctx.token_folios,
            ref_bigrams, n_perms=n_perms,
        )
    else:
        null_exact_rates = [0.0] * n_perms
        null_total_rates = [0.0] * n_perms

    z_exact, p_exact, nm_exact, ns_exact = _z_from_null(
        exact_rate, null_exact_rates,
    )
    z_total, p_total, nm_total, ns_total = _z_from_null(
        total_rate, null_total_rates,
    )

    return TableBigramZ(
        table_name=table_name,
        dict_mode=dict_mode,
        n_tokens=n_tokens,
        n_signal=n_signal,
        signal_rate=round(signal_rate, 4),
        n_signal_pairs=n_pairs,
        n_bigram_hits_exact=n_exact,
        bigram_hit_rate_exact=round(exact_rate, 6),
        n_bigram_hits_relaxed=n_relaxed,
        bigram_hit_rate_total=round(total_rate, 6),
        null_mean_exact=round(nm_exact, 6),
        null_std_exact=round(ns_exact, 6),
        z_exact=round(z_exact, 4),
        p_value_exact=round(p_exact, 6),
        null_mean_total=round(nm_total, 6),
        null_std_total=round(ns_total, 6),
        z_total=round(z_total, 4),
        p_value_total=round(p_total, 6),
    )


# ---------------------------------------------------------------------------
# Step 46A.1 — Candidate Table Assembly
# ---------------------------------------------------------------------------


def run_arb_tables(ctx: Optional[ArbitrationContext] = None) -> None:
    """Step 46A.1: Assemble 8 candidate tables."""
    t0 = time.time()
    print("=" * 70)
    print("STEP 46A.1: Candidate Table Assembly")
    print("=" * 70)

    rd = _results_dir()
    tables = _build_candidate_tables(rd)

    # Build disputed triples detail
    p15 = tables[0]['assignment']
    maxsat = tables[1]['assignment']
    csa = tables[6]['assignment']
    canon = tables[7]['assignment']

    disputed_detail = []
    for k in sorted(p15.keys()):
        if p15.get(k) != maxsat.get(k):
            disputed_detail.append({
                'triple_key': k,
                'P15': p15.get(k, '?'),
                'MaxSAT': maxsat.get(k, '?'),
                'CSA': csa.get(k, '?'),
                'Canonical': canon.get(k, '?'),
            })

    # Print summary
    for ct in tables:
        n_diff = sum(
            1 for k in p15
            if ct['assignment'].get(k) != p15.get(k)
        )
        print(f"  {ct['name']:15s}  dict_mode={ct['dict_mode']}  "
              f"diff_from_P15={n_diff}  {ct['provenance']}")

    print(f"\n  Disputed triples: {len(disputed_detail)}")
    for d in disputed_detail:
        print(f"    {d['triple_key']}: P15={d['P15']} MaxSAT={d['MaxSAT']} "
              f"CSA={d['CSA']} Canon={d['Canonical']}")

    result = ArbTablesResult(
        tables=[{
            'name': ct['name'],
            'assignment': ct['assignment'],
            'dict_mode': ct['dict_mode'],
            'provenance': ct['provenance'],
        } for ct in tables],
        n_disputed=len(disputed_detail),
        disputed_triples=disputed_detail,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'arb_tables.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2)
    print(f"\n  Saved -> {out_path}")


# ---------------------------------------------------------------------------
# Step 46A.2 — Validated Bigram z for Each Table
# ---------------------------------------------------------------------------


def run_arb_bigram(ctx: Optional[ArbitrationContext] = None) -> None:
    """Step 46A.2: Compute validated bigram z for all candidate tables."""
    t0 = time.time()
    print("=" * 70)
    print("STEP 46A.2: Validated Bigram z for Each Table")
    print("=" * 70)

    rd = _results_dir()
    if ctx is None:
        print("  Building context...")
        ctx = _build_context(rd)

    # Load candidate tables
    arb_data = _safe_load(os.path.join(rd, 'arb_tables.json'))
    tables = arb_data.get('tables', [])
    if not tables:
        tables = _build_candidate_tables(rd)
    disputed = arb_data.get('disputed_triples', [])
    disputed_keys = [d['triple_key'] for d in disputed]

    # Build reference bigrams for 10K and 131K
    print("  Building reference bigrams...")
    ref_bigrams_10k = _build_ref_bigrams(ctx.ref_tokens, ctx.ref_word_set_10k)
    ref_bigrams_131k = _build_ref_bigrams(
        ctx.ref_tokens, ctx.ref_word_set_131k,
    )
    ref_bigrams_19k = None
    if ctx.ref_word_set_19k:
        ref_bigrams_19k = _build_ref_bigrams(
            ctx.ref_tokens, ctx.ref_word_set_19k,
        )
    print(f"    10K bigrams: {len(ref_bigrams_10k)}")
    print(f"    131K bigrams: {len(ref_bigrams_131k)}")

    # Compute z for each table
    per_table: List[Dict] = []
    for i, ct in enumerate(tables):
        name = ct['name']
        assignment = ct['assignment']
        dm = ct['dict_mode']

        # Choose ref set and bigrams based on dict_mode
        if dm == '10K':
            ref_ws = ctx.ref_word_set_10k
            ref_bg = ref_bigrams_10k
        else:
            ref_ws = ctx.ref_word_set_131k
            ref_bg = ref_bigrams_131k

        print(f"\n  [{i+1}/{len(tables)}] {name} (dict_mode={dm})...")
        sys.stdout.flush()
        bz = _compute_bigram_z(
            name, assignment, dm, ctx, ref_bg, ref_ws, n_perms=500,
        )
        entry = _convert(asdict(bz))

        # Also compute at 10K if this table used 131K
        if dm == '131K':
            print(f"    Also computing at 10K...")
            bz_10k = _compute_bigram_z(
                name + '_at10K', assignment, '10K', ctx,
                ref_bigrams_10k, ctx.ref_word_set_10k, n_perms=100,
            )
            entry['z_total_at_10k'] = bz_10k.z_total
            entry['z_exact_at_10k'] = bz_10k.z_exact
        else:
            entry['z_total_at_10k'] = bz.z_total
            entry['z_exact_at_10k'] = bz.z_exact

        # Also compute at 19K if available (fewer perms — secondary metric)
        if ref_bigrams_19k is not None and ctx.ref_word_set_19k is not None:
            bz_19k = _compute_bigram_z(
                name + '_at19K', assignment, '19K', ctx,
                ref_bigrams_19k, ctx.ref_word_set_19k, n_perms=100,
            )
            entry['z_total_at_19k'] = bz_19k.z_total
        else:
            entry['z_total_at_19k'] = None

        print(f"    z_exact={bz.z_exact:.2f}  z_total={bz.z_total:.2f}  "
              f"z_total_at_10k={entry['z_total_at_10k']:.2f}  "
              f"n_signal={bz.n_signal}  n_pairs={bz.n_signal_pairs}")
        sys.stdout.flush()
        per_table.append(entry)

    # Construct T_BEST6: for each disputed triple, try swapping P15→MaxSAT
    print("\n  Constructing T_BEST6 via per-triple z comparison...")
    p15_assignment = tables[0]['assignment']
    maxsat_assignment = tables[1]['assignment']
    best6 = dict(p15_assignment)
    best6_decisions = []

    for dk in disputed_keys:
        # Table with just this one triple swapped to MaxSAT
        swapped = dict(p15_assignment)
        swapped[dk] = maxsat_assignment.get(dk, p15_assignment.get(dk))

        # Compute z at 10K for the swapped table
        bz_swapped = _compute_bigram_z(
            f'swap_{dk}', swapped, '10K', ctx,
            ref_bigrams_10k, ctx.ref_word_set_10k, n_perms=200,
        )

        # Compare with P15's z at 10K
        p15_z = next(
            (e['z_total_at_10k'] for e in per_table if e['table_name'] == 'T_P15'),
            0.0,
        )
        winner = 'MaxSAT' if bz_swapped.z_total > p15_z else 'P15'
        if winner == 'MaxSAT':
            best6[dk] = maxsat_assignment[dk]

        best6_decisions.append({
            'triple_key': dk,
            'P15_value': p15_assignment.get(dk),
            'MaxSAT_value': maxsat_assignment.get(dk),
            'z_with_swap': round(bz_swapped.z_total, 4),
            'z_p15_baseline': round(p15_z, 4),
            'winner': winner,
        })
        print(f"    {dk}: P15={p15_assignment.get(dk)} MaxSAT={maxsat_assignment.get(dk)} "
              f"z_swap={bz_swapped.z_total:.2f} -> {winner}")

    # Now compute z for the assembled T_BEST6
    print("\n  Computing z for assembled T_BEST6...")
    bz_best6 = _compute_bigram_z(
        'T_BEST6', best6, '10K', ctx,
        ref_bigrams_10k, ctx.ref_word_set_10k, n_perms=500,
    )
    best6_entry = _convert(asdict(bz_best6))
    best6_entry['z_total_at_10k'] = bz_best6.z_total
    best6_entry['z_exact_at_10k'] = bz_best6.z_exact
    if ref_bigrams_19k is not None and ctx.ref_word_set_19k is not None:
        bz_19k = _compute_bigram_z(
            'T_BEST6_at19K', best6, '19K', ctx,
            ref_bigrams_19k, ctx.ref_word_set_19k, n_perms=100,
        )
        best6_entry['z_total_at_19k'] = bz_19k.z_total
    else:
        best6_entry['z_total_at_19k'] = None

    # Update T_BEST6 in per_table
    for entry in per_table:
        if entry['table_name'] == 'T_BEST6':
            entry.update(best6_entry)
            entry['table_name'] = 'T_BEST6'
            break

    # Find best tables
    best_z_exact = max(per_table, key=lambda e: e.get('z_exact', 0))
    best_z_total = max(per_table, key=lambda e: e.get('z_total_at_10k', 0))

    print(f"\n  Best z_exact: {best_z_exact['table_name']} ({best_z_exact.get('z_exact', 0):.2f})")
    print(f"  Best z_total_10K: {best_z_total['table_name']} ({best_z_total.get('z_total_at_10k', 0):.2f})")

    result = ArbBigramResult(
        per_table=per_table,
        best6_decisions=best6_decisions,
        best6_final_assignment=best6,
        best_z_exact=best_z_exact['table_name'],
        best_z_total=best_z_total['table_name'],
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'arb_bigram.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2)
    print(f"\n  Saved -> {out_path}")


# ---------------------------------------------------------------------------
# Step 46A.3 — Signal Word Survival
# ---------------------------------------------------------------------------


def run_arb_signal(ctx: Optional[ArbitrationContext] = None) -> None:
    """Step 46A.3: Check signal word survival under each table."""
    t0 = time.time()
    print("=" * 70)
    print("STEP 46A.3: Signal Word Survival")
    print("=" * 70)

    rd = _results_dir()
    if ctx is None:
        print("  Building context...")
        ctx = _build_context(rd)

    # Load tables from arb_tables.json (or rebuild)
    arb_data = _safe_load(os.path.join(rd, 'arb_tables.json'))
    tables = arb_data.get('tables', [])
    if not tables:
        tables = _build_candidate_tables(rd)

    # Update T_BEST6 assignment if arb_bigram.json exists
    bigram_data = _safe_load(os.path.join(rd, 'arb_bigram.json'))
    best6_assignment = bigram_data.get('best6_final_assignment')
    if best6_assignment:
        for ct in tables:
            if ct['name'] == 'T_BEST6':
                ct['assignment'] = best6_assignment

    # Load Phase 36 signal words
    s10_data = _safe_load(os.path.join(rd, 'signal_10k.json'))
    phase36_words = []
    if 'word_signals' in s10_data:
        phase36_words = [
            w['word'] for w in s10_data['word_signals']
            if w.get('is_genuine_signal')
        ]

    per_table: List[Dict] = []
    for i, ct in enumerate(tables):
        name = ct['name']
        assignment = ct['assignment']
        print(f"\n  [{i+1}/{len(tables)}] {name}...")

        # Decode full corpus
        decoded = _decode_corpus_r3(
            ctx.all_tokens, assignment, ctx.eva_to_triple,
            ctx.modifier_chars, ctx.modifier_rules,
            ctx.ref_word_set_131k,
        )
        decoded_set = set(decoded)

        # Check bedrock signal words
        bedrock_surviving = [w for w in BEDROCK_SIGNAL_WORDS if w in decoded_set]
        bedrock_lost = [w for w in BEDROCK_SIGNAL_WORDS if w not in decoded_set]

        # Check bootstrap words
        boot_surviving = [w for w in BOOTSTRAP_WORDS if w in decoded_set]
        boot_lost = [w for w in BOOTSTRAP_WORDS if w not in decoded_set]

        # Check Phase 36 words
        p36_surviving = [w for w in phase36_words if w in decoded_set]
        p36_lost = [w for w in phase36_words if w not in decoded_set]

        total = len(bedrock_surviving) + len(boot_surviving) + len(p36_surviving)

        print(f"    Bedrock: {len(bedrock_surviving)}/{len(BEDROCK_SIGNAL_WORDS)} "
              f"Bootstrap: {len(boot_surviving)}/{len(BOOTSTRAP_WORDS)} "
              f"Phase36: {len(p36_surviving)}/{len(phase36_words)}")

        per_table.append({
            'table_name': name,
            'bedrock_surviving': bedrock_surviving,
            'bedrock_lost': bedrock_lost,
            'bedrock_survival_rate': round(
                len(bedrock_surviving) / len(BEDROCK_SIGNAL_WORDS), 4,
            ),
            'bootstrap_surviving': boot_surviving,
            'bootstrap_lost': boot_lost,
            'phase36_surviving': p36_surviving,
            'phase36_lost': p36_lost,
            'total_survival_count': total,
        })

    best = max(per_table, key=lambda e: e['total_survival_count'])

    result = ArbSignalResult(
        per_table=per_table,
        best_survival=best['table_name'],
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'arb_signal.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2)
    print(f"\n  Saved -> {out_path}")


# ---------------------------------------------------------------------------
# Step 46A.4 — Dict-Hit at 10K
# ---------------------------------------------------------------------------


def run_arb_10k(ctx: Optional[ArbitrationContext] = None) -> None:
    """Step 46A.4: Dict-hit at 10K for all 8 tables."""
    t0 = time.time()
    print("=" * 70)
    print("STEP 46A.4: Dict-Hit at 10K")
    print("=" * 70)

    rd = _results_dir()
    if ctx is None:
        print("  Building context...")
        ctx = _build_context(rd)

    arb_data = _safe_load(os.path.join(rd, 'arb_tables.json'))
    tables = arb_data.get('tables', [])
    if not tables:
        tables = _build_candidate_tables(rd)

    # Update T_BEST6
    bigram_data = _safe_load(os.path.join(rd, 'arb_bigram.json'))
    best6_assignment = bigram_data.get('best6_final_assignment')
    if best6_assignment:
        for ct in tables:
            if ct['name'] == 'T_BEST6':
                ct['assignment'] = best6_assignment

    n_tokens = len(ctx.all_tokens)
    per_table: List[Dict] = []

    for i, ct in enumerate(tables):
        name = ct['name']
        assignment = ct['assignment']
        print(f"\n  [{i+1}/{len(tables)}] {name}...")

        # Decode + classify at 10K
        decoded, classifications = _classify_tokens(
            ctx.all_tokens, assignment, ctx, ctx.ref_word_set_10k,
        )

        n_signal = sum(1 for c in classifications if c == 'SIGNAL')
        n_anti = sum(1 for c in classifications if c == 'ANTI_SIGNAL')
        signal_rate = n_signal / n_tokens if n_tokens else 0.0

        dict_hits = sum(
            1 for c in classifications if c in ('SIGNAL', 'SHARED_HIT')
        )
        dict_hit_rate = dict_hits / n_tokens if n_tokens else 0.0

        # Null signal rate for selectivity
        null_signal_rates = []
        for seed in ctx.null_seeds:
            null_tokens = _generate_null_corpus(
                ctx.bigram_probs, ctx.initial_probs,
                ctx.token_lengths, n_tokens, seed,
            )
            null_decoded = _decode_corpus_r3(
                null_tokens, assignment, ctx.eva_to_triple,
                ctx.modifier_chars, ctx.modifier_rules,
                ctx.ref_word_set_10k,
            )
            null_hits = sum(1 for w in null_decoded if w in ctx.ref_word_set_10k)
            null_signal_rates.append(null_hits / n_tokens if n_tokens else 0.0)

        null_mean = sum(null_signal_rates) / len(null_signal_rates) if null_signal_rates else 0.0
        selectivity = dict_hit_rate / null_mean if null_mean > 0 else float('inf')

        print(f"    dict_hit_10k={dict_hit_rate:.4f}  signal={n_signal}  "
              f"selectivity={selectivity:.2f}")

        per_table.append({
            'table_name': name,
            'dict_hit_10k': round(dict_hit_rate, 4),
            'n_signal_10k': n_signal,
            'n_anti_signal_10k': n_anti,
            'signal_rate_10k': round(signal_rate, 4),
            'net_signal': n_signal - n_anti,
            'selectivity_10k': round(selectivity, 4),
            'null_mean_10k': round(null_mean, 4),
        })

    best_dh = max(per_table, key=lambda e: e['dict_hit_10k'])
    best_sel = max(per_table, key=lambda e: e['selectivity_10k'])

    result = ArbDict10KResult(
        per_table=per_table,
        best_dict_hit=best_dh['table_name'],
        best_selectivity=best_sel['table_name'],
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'arb_10k.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2)
    print(f"\n  Saved -> {out_path}")


# ---------------------------------------------------------------------------
# Step 46A.5 — Definitive Table Selection
# ---------------------------------------------------------------------------


def run_arb_select(ctx: Optional[ArbitrationContext] = None) -> None:
    """Step 46A.5: Select definitive table via composite scoring."""
    t0 = time.time()
    print("=" * 70)
    print("STEP 46A.5: Definitive Table Selection")
    print("=" * 70)

    rd = _results_dir()

    # Load all previous step results
    bigram_data = _safe_load(os.path.join(rd, 'arb_bigram.json'))
    signal_data = _safe_load(os.path.join(rd, 'arb_signal.json'))
    dict_data = _safe_load(os.path.join(rd, 'arb_10k.json'))
    arb_data = _safe_load(os.path.join(rd, 'arb_tables.json'))

    bigram_per = bigram_data.get('per_table', [])
    signal_per = signal_data.get('per_table', [])
    dict_per = dict_data.get('per_table', [])

    # Index by table name
    bigram_by_name = {e['table_name']: e for e in bigram_per}
    signal_by_name = {e['table_name']: e for e in signal_per}
    dict_by_name = {e['table_name']: e for e in dict_per}

    # Get all table names
    all_names = [e['table_name'] for e in bigram_per]

    # Extract raw metrics
    raw: List[Dict] = []
    for name in all_names:
        z_total_10k = bigram_by_name.get(name, {}).get('z_total_at_10k', 0.0) or 0.0
        selectivity_10k = dict_by_name.get(name, {}).get('selectivity_10k', 0.0) or 0.0
        bedrock_surv = signal_by_name.get(name, {}).get('bedrock_survival_rate', 0.0) or 0.0
        dict_hit_10k = dict_by_name.get(name, {}).get('dict_hit_10k', 0.0) or 0.0

        raw.append({
            'table_name': name,
            'z_total_10k': z_total_10k,
            'selectivity_10k': selectivity_10k,
            'signal_survival': bedrock_surv,
            'dict_hit_10k': dict_hit_10k,
        })

    # Normalize each metric to [0, 1] by dividing by max
    def _normalize(values: List[float]) -> List[float]:
        mx = max(values) if values and max(values) > 0 else 1.0
        return [v / mx for v in values]

    z_vals = [r['z_total_10k'] for r in raw]
    sel_vals = [r['selectivity_10k'] for r in raw]
    surv_vals = [r['signal_survival'] for r in raw]
    dh_vals = [r['dict_hit_10k'] for r in raw]

    z_norm = _normalize(z_vals)
    sel_norm = _normalize(sel_vals)
    # signal_survival is already 0-1, but normalize anyway for fairness
    surv_norm = _normalize(surv_vals)
    dh_norm = _normalize(dh_vals)

    # Compute composite scores
    rankings: List[Dict] = []
    for i, r in enumerate(raw):
        composite = (
            COMPOSITE_WEIGHTS['z_total_10k'] * z_norm[i]
            + COMPOSITE_WEIGHTS['selectivity_10k'] * sel_norm[i]
            + COMPOSITE_WEIGHTS['signal_survival'] * surv_norm[i]
            + COMPOSITE_WEIGHTS['dict_hit_10k'] * dh_norm[i]
        )
        rankings.append({
            'table_name': r['table_name'],
            'z_total_10k': round(r['z_total_10k'], 4),
            'selectivity_10k': round(r['selectivity_10k'], 4),
            'signal_survival': round(r['signal_survival'], 4),
            'dict_hit_10k': round(r['dict_hit_10k'], 4),
            'composite': round(composite, 4),
        })

    # Sort by composite descending
    rankings.sort(key=lambda r: -r['composite'])
    for rank_i, r in enumerate(rankings):
        r['rank'] = rank_i + 1

    winner = rankings[0]['table_name']
    print(f"\n  Rankings:")
    for r in rankings:
        print(f"    #{r['rank']}  {r['table_name']:15s}  composite={r['composite']:.4f}  "
              f"z_10k={r['z_total_10k']:.2f}  sel={r['selectivity_10k']:.2f}  "
              f"surv={r['signal_survival']:.2f}  dh={r['dict_hit_10k']:.4f}")

    # Get the definitive assignment
    all_tables = arb_data.get('tables', [])
    definitive_assignment = {}
    for ct in all_tables:
        if ct['name'] == winner:
            definitive_assignment = ct['assignment']
            break

    # If winner is T_BEST6, use the updated assignment from bigram step
    if winner == 'T_BEST6':
        best6_asgn = bigram_data.get('best6_final_assignment')
        if best6_asgn:
            definitive_assignment = best6_asgn

    # Compare vs P15 and Canonical
    p15_entry = next((r for r in rankings if r['table_name'] == 'T_P15'), None)
    canon_entry = next(
        (r for r in rankings if r['table_name'] == 'T_CANONICAL'), None,
    )
    winner_entry = rankings[0]

    vs_p15 = {}
    if p15_entry:
        vs_p15 = {
            'z_delta': round(winner_entry['z_total_10k'] - p15_entry['z_total_10k'], 4),
            'dict_hit_delta': round(winner_entry['dict_hit_10k'] - p15_entry['dict_hit_10k'], 4),
            'composite_delta': round(winner_entry['composite'] - p15_entry['composite'], 4),
        }
    vs_canonical = {}
    if canon_entry:
        vs_canonical = {
            'z_delta': round(winner_entry['z_total_10k'] - canon_entry['z_total_10k'], 4),
            'dict_hit_delta': round(winner_entry['dict_hit_10k'] - canon_entry['dict_hit_10k'], 4),
            'composite_delta': round(winner_entry['composite'] - canon_entry['composite'], 4),
        }

    # Determine verdict
    if winner == 'T_P15' or winner == 'T_P15_10K':
        verdict = 'TABLE_CONFIRMED'
        rationale = (
            f"Phase 15 table ({winner}) wins composite scoring. "
            "MaxSAT disagreements are artifacts of constraint formulation."
        )
    elif winner in ('T_MAX', 'T_MAX_10K'):
        verdict = 'TABLE_UPDATED'
        rationale = (
            f"MaxSAT consensus ({winner}) improves linguistic quality "
            "despite lower dict-hit. The 6 MaxSAT corrections are genuine."
        )
    elif winner == 'T_BEST6':
        verdict = 'TABLE_HYBRID'
        rationale = (
            "Per-triple cherry-picking found the best combination. "
            "Some triples follow MaxSAT, others Phase 15."
        )
    elif winner == 'T_CANONICAL':
        verdict = 'TABLE_CANONICAL_WINS'
        rationale = (
            "Phase 45 canonical table (MaxSAT consensus + P15 defaults for "
            "ambiguous) is the overall best."
        )
    else:
        verdict = f'TABLE_SELECTED_{winner}'
        rationale = f"{winner} wins composite scoring."

    gate_passed = bool(definitive_assignment and len(definitive_assignment) == 25)

    print(f"\n  VERDICT: {verdict}")
    print(f"  Winner: {winner} (composite={winner_entry['composite']:.4f})")
    print(f"  Rationale: {rationale}")

    result = ArbSelectionResult(
        rankings=rankings,
        definitive_table_name=winner,
        definitive_assignment=definitive_assignment,
        composite_weights=COMPOSITE_WEIGHTS,
        vs_p15=vs_p15,
        vs_canonical=vs_canonical,
        gate_passed=gate_passed,
        verdict=verdict,
        rationale=rationale,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'arb_selection.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2)
    print(f"\n  Saved -> {out_path}")


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------


def run_track_a_46() -> None:
    """Run all Track A steps with shared context."""
    rd = _results_dir()
    print("  Building shared context for Track A...")
    ctx = _build_context(rd)

    run_arb_tables(ctx)
    print("\n" + "=" * 70 + "\n")
    run_arb_bigram(ctx)
    print("\n" + "=" * 70 + "\n")
    run_arb_signal(ctx)
    print("\n" + "=" * 70 + "\n")
    run_arb_10k(ctx)
    print("\n" + "=" * 70 + "\n")
    run_arb_select(ctx)
