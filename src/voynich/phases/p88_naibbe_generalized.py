"""
Phase 88 — Generalized Naibbe Cipher (Greshko 2026 correspondence)
===================================================================
Faithful port of Michael Greshko's generalized-Naibbe Jupyter notebook
(private repo greshko/naibbe-proxy-analysis, supplied by email 2026-04-15),
into the voynich_2 phase/dataclass/JSON pipeline.

The generalized Naibbe differs from the simplified word-level proxy in
naibbe_entropy.py (Phase 27.2) in four ways:
    (1) plaintext is respaced into 100%% orthographic bigrams
    (2) 6 tables applied with 5:2:2:2:1:1 weighting
    (3) each table = prefix slot grammar + suffix slot grammar
    (4) output alphabet restricted to ~20 chars

Dependency chain:
    data/reference/greshko/{nathist_book16,divcom_output_ciphertext,
                            nathist_output_ciphertext}.txt
    data/corpus/ZL3b-n.txt  (via load_corpus)
    results/entropy_shift_cipher.json  (tachygraphic cos 0.820 reference)
    results/naibbe_entropy.json  (simplified Naibbe cos -0.843 reference)
        -> results/p88_naibbe_generalized.json
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import load_corpus, tokenize_eva_chars
from voynich.core.reference import load_reference_corpus
from voynich.core.stats import cosine_similarity, entropy_curve


# ---------------------------------------------------------------------------
# JSON serialiser (matches naibbe_entropy.py pattern)
# ---------------------------------------------------------------------------

def _convert(obj: Any) -> Any:
    if hasattr(obj, '__dataclass_fields__'):
        return {k: _convert(v) for k, v in asdict(obj).items()}
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
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
# Parameters (match Greshko's notebook Cell 2)
# ---------------------------------------------------------------------------

OUTPUT_ALPHA_SIZE = 20
N_TABLES = 6
TABLE_WEIGHTS = [5, 2, 2, 2, 1, 1]
N_BROAD = 200
MAX_ORDER = 6
H1_TOL_LOW = 0.30
H1_TOL_HIGH = 0.50

_PLAINTEXT_ALPHABET = list("abcdefghijklmnopqrstuvwxyz")


def _build_cumulative_weights(weights: List[int]) -> List[float]:
    total = sum(weights)
    cum, running = [], 0.0
    for w in weights:
        running += w / total
        cum.append(running)
    return cum


_CUM_WEIGHTS = _build_cumulative_weights(TABLE_WEIGHTS)


# ---------------------------------------------------------------------------
# Slot grammar + table generation (port of notebook Cell 4)
# ---------------------------------------------------------------------------

def make_slot_grammar(chars: List[str], rng: random.Random,
                      n_slots: int = 4, p_optional: float = 0.5) -> List:
    slots = []
    for i in range(n_slots):
        k = rng.randint(2, max(2, len(chars)))
        slot_chars = rng.sample(chars, k)
        required = (i == 0) or (rng.random() > p_optional)
        slots.append((slot_chars, required))
    return slots


def generate_string(slots: List, rng: random.Random) -> str:
    result = []
    for char_set, required in slots:
        if required or rng.random() > 0.5:
            result.append(rng.choice(char_set))
    return "".join(result) if result else rng.choice(slots[0][0])


def mean_string_length(slots: List, n_samples: int = 2000,
                       rng: Optional[random.Random] = None) -> float:
    if rng is None:
        rng = random.Random(0)
    return float(np.mean([len(generate_string(slots, rng))
                          for _ in range(n_samples)]))


def make_grammar_targeting_length(chars: List[str], rng: random.Random,
                                  lo: float = 2.0, hi: float = 3.0,
                                  max_attempts: int = 300
                                  ) -> Tuple[List, float, int, float]:
    for _ in range(max_attempts):
        n_slots = rng.randint(3, 5)
        p_opt = rng.uniform(0.3, 0.7)
        slots = make_slot_grammar(chars, rng, n_slots=n_slots, p_optional=p_opt)
        ml = mean_string_length(slots, rng=rng)
        if lo <= ml <= hi:
            return slots, ml, n_slots, p_opt
    slots = make_slot_grammar(chars, rng, n_slots=3, p_optional=0.5)
    return slots, mean_string_length(slots, rng=rng), 3, 0.5


def make_tables(prefix_slots: List, suffix_slots: List,
                n_tables: int, rng: random.Random,
                max_attempts: int = 500) -> List[Tuple[Dict, Dict]]:
    tables = []
    for _ in range(n_tables):
        prefixes, seen_p = {}, set()
        for letter in _PLAINTEXT_ALPHABET:
            for _ in range(max_attempts):
                s = generate_string(prefix_slots, rng)
                if s not in seen_p:
                    prefixes[letter] = s
                    seen_p.add(s)
                    break
            else:
                s = generate_string(prefix_slots, rng)
                while s in seen_p:
                    s += rng.choice(prefix_slots[0][0])
                prefixes[letter] = s
                seen_p.add(s)

        suffixes, seen_s = {}, set()
        for letter in _PLAINTEXT_ALPHABET:
            for _ in range(max_attempts):
                s = generate_string(suffix_slots, rng)
                if s not in seen_s:
                    suffixes[letter] = s
                    seen_s.add(s)
                    break
            else:
                s = generate_string(suffix_slots, rng)
                while s in seen_s:
                    s += rng.choice(suffix_slots[0][0])
                suffixes[letter] = s
                seen_s.add(s)

        tables.append((prefixes, suffixes))
    return tables


# ---------------------------------------------------------------------------
# Encoder (port of notebook Cell 5)
# ---------------------------------------------------------------------------

def encode_bigrams(plaintext: str,
                   tables: List[Tuple[Dict, Dict]],
                   rng: random.Random,
                   cum_weights: Optional[List[float]] = None) -> List[str]:
    chars = [c for c in plaintext.lower() if c.isalpha()]
    if len(chars) % 2 == 1:
        chars.append("a")

    if cum_weights is None:
        cum_weights = _CUM_WEIGHTS

    n_tables = len(tables)
    tokens = []
    for i in range(0, len(chars), 2):
        c1, c2 = chars[i], chars[i + 1]
        r = rng.random()
        t_idx = min(
            next(j for j, cw in enumerate(cum_weights) if r <= cw),
            n_tables - 1,
        )
        token = tables[t_idx][0][c1] + tables[t_idx][1][c2]
        tokens.append(token)
    return tokens


# ---------------------------------------------------------------------------
# Utility: corpus size matching (notebook Cell 3)
# ---------------------------------------------------------------------------

def _match_corpus_size(text: str, target_len: int) -> str:
    if len(text) >= target_len:
        return text[:target_len]
    if len(text) == 0:
        return ""
    repeats = target_len // len(text) + 2
    return (text * repeats)[:target_len]


# ---------------------------------------------------------------------------
# Diagnostic helpers
# ---------------------------------------------------------------------------

def _cross_boundary_ratio_plain(tokens: List[str]) -> Dict[str, float]:
    """
    Cross-boundary mutual information on plain character tokenisation
    (no EVA ligature handling). Same formula as
    currier_selfcorr.measure_cross_boundary_mi but each token is split
    into individual characters via list(token).
    """
    pairs: List[Tuple[str, str]] = []
    for i in range(len(tokens) - 1):
        a, b = tokens[i], tokens[i + 1]
        if a and b:
            pairs.append((a[-1], b[0]))

    n_pairs = len(pairs)
    if n_pairs == 0:
        return {'mi': 0.0, 'ratio': 1.0, 'n_pairs': 0}

    joint = Counter(pairs)
    last_counts = Counter(p[0] for p in pairs)
    first_counts = Counter(p[1] for p in pairs)

    mi = 0.0
    for (last, first), count in joint.items():
        p_joint = count / n_pairs
        p_last = last_counts[last] / n_pairs
        p_first = first_counts[first] / n_pairs
        if p_joint > 0 and p_last > 0 and p_first > 0:
            mi += p_joint * math.log2(p_joint / (p_last * p_first))

    weighted_ratio = 0.0
    for (last, first), count in joint.items():
        p_first_given_last = count / last_counts[last]
        p_first = first_counts[first] / n_pairs
        if p_first > 0:
            weighted_ratio += count * (p_first_given_last / p_first)
    weighted_ratio /= n_pairs

    return {
        'mi': round(float(mi), 6),
        'ratio': round(float(weighted_ratio), 6),
        'n_pairs': n_pairs,
    }


def _levenshtein1_check(s1: str, s2: str) -> bool:
    """Fast check for Levenshtein distance == 1 (used only for ±1-length
    pairs; same-length is handled separately)."""
    if abs(len(s1) - len(s2)) != 1:
        return False
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    i = j = 0
    diffs = 0
    while i < len(s1) and j < len(s2):
        if s1[i] == s2[j]:
            i += 1
            j += 1
        else:
            diffs += 1
            if diffs > 1:
                return False
            j += 1
    diffs += len(s2) - j
    return diffs == 1


def _freq_conn_rho_plain(tokens: List[str], min_freq: int = 3,
                         max_types: int = 3000) -> Dict[str, float]:
    """
    Spearman correlation between log(type frequency) and ED-1 neighbor
    count. Plain-character tokenisation (no EVA handling). Capped at
    max_types frequent types to keep runtime bounded.
    """
    from scipy.stats import spearmanr

    counts = Counter(tokens)
    frequent = [(t, c) for t, c in counts.items() if c >= min_freq]
    if not frequent:
        return {'rho': 0.0, 'p': 1.0, 'n_types': 0}

    frequent.sort(key=lambda x: -x[1])
    frequent = frequent[:max_types]
    type_list = [t for t, _ in frequent]
    freq_map = dict(frequent)

    by_len: Dict[int, List[str]] = {}
    for t in type_list:
        by_len.setdefault(len(t), []).append(t)

    connectivity: Dict[str, int] = {}
    for t in type_list:
        L = len(t)
        neighbors = 0
        for other in by_len.get(L, []):
            if other == t:
                continue
            diffs = sum(1 for a, b in zip(t, other) if a != b)
            if diffs == 1:
                neighbors += 1
        for dl in (L - 1, L + 1):
            if dl < 1:
                continue
            for other in by_len.get(dl, []):
                if _levenshtein1_check(t, other):
                    neighbors += 1
        connectivity[t] = neighbors

    if len(type_list) < 10:
        return {'rho': 0.0, 'p': 1.0, 'n_types': len(type_list)}

    log_freqs = [math.log(freq_map[t]) for t in type_list]
    conns = [connectivity[t] for t in type_list]
    rho, p_val = spearmanr(log_freqs, conns)
    if rho is None or (isinstance(rho, float) and math.isnan(rho)):
        return {'rho': 0.0, 'p': 1.0, 'n_types': len(type_list)}
    return {
        'rho': round(float(rho), 4),
        'p': float(p_val),
        'n_types': len(type_list),
    }


def _freq_conn_rho_eva(tokens: List[str], min_freq: int = 3,
                       max_types: int = 3000) -> Dict[str, float]:
    """Same as _freq_conn_rho_plain but uses EVA ligature tokenisation
    (for Voynich reference only)."""
    from scipy.stats import spearmanr

    counts = Counter(tokens)
    frequent = [(t, c) for t, c in counts.items() if c >= min_freq]
    if not frequent:
        return {'rho': 0.0, 'p': 1.0, 'n_types': 0}
    frequent.sort(key=lambda x: -x[1])
    frequent = frequent[:max_types]
    type_list = [t for t, _ in frequent]
    freq_map = dict(frequent)

    char_lists = {t: tokenize_eva_chars(t) for t in type_list}
    by_len: Dict[int, List[str]] = {}
    for t in type_list:
        by_len.setdefault(len(char_lists[t]), []).append(t)

    connectivity: Dict[str, int] = {}
    for t in type_list:
        chars = char_lists[t]
        L = len(chars)
        neighbors = 0
        for other in by_len.get(L, []):
            if other == t:
                continue
            diffs = sum(1 for a, b in zip(chars, char_lists[other]) if a != b)
            if diffs == 1:
                neighbors += 1
        t_joined = ''.join(chars)
        for dl in (L - 1, L + 1):
            if dl < 1:
                continue
            for other in by_len.get(dl, []):
                if _levenshtein1_check(t_joined, ''.join(char_lists[other])):
                    neighbors += 1
        connectivity[t] = neighbors

    if len(type_list) < 10:
        return {'rho': 0.0, 'p': 1.0, 'n_types': len(type_list)}
    log_freqs = [math.log(freq_map[t]) for t in type_list]
    conns = [connectivity[t] for t in type_list]
    rho, p_val = spearmanr(log_freqs, conns)
    if rho is None or (isinstance(rho, float) and math.isnan(rho)):
        return {'rho': 0.0, 'p': 1.0, 'n_types': len(type_list)}
    return {
        'rho': round(float(rho), 4),
        'p': float(p_val),
        'n_types': len(type_list),
    }


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ProxyRunResult:
    seed: int
    curve: Dict[int, float]
    cos_shift_full: float
    cos_shift_b: float
    pre_mean_len: float
    suf_mean_len: float
    mean_tok_len: float
    unique_types: int
    most_common_pct: float
    pre_n_slots: int
    suf_n_slots: int
    pre_n_req: int
    suf_n_req: int
    pre_mean_slot_size: float
    suf_mean_slot_size: float
    pre_min_slot_size: int
    suf_min_slot_size: int
    pre_adj_overlap: float
    suf_adj_overlap: float
    # Populated only for low-H1 subset and real Naibbe
    cross_boundary_ratio: Optional[float] = None
    freq_conn_rho: Optional[float] = None


@dataclass
class ExternalRefCheck:
    name: str
    n_tokens: int
    n_chars_sampled: int
    curve: Dict[int, float]
    cos_shift_full: float
    cos_shift_b: float
    cross_boundary_ratio: float
    freq_conn_rho: float


@dataclass
class NaibbeGeneralizedResult:
    timestamp: str
    runtime_seconds: float
    parameters: Dict[str, Any]
    voynich_full_curve: Dict[str, float]
    voynich_b_curve: Dict[str, float]
    latin_curve: Dict[str, float]
    voynich_full_shift: List[float]
    voynich_b_shift: List[float]
    voynich_full_cross_boundary_ratio: float
    voynich_b_cross_boundary_ratio: float
    voynich_full_freq_conn_rho: float
    voynich_b_freq_conn_rho: float
    # External reference ciphers
    real_naibbe_divcom: ExternalRefCheck
    real_naibbe_nathist: ExternalRefCheck
    # Aggregates
    n_broad: int
    broad_cos_shift_full_mean: float
    broad_cos_shift_full_std: float
    broad_cos_shift_b_mean: float
    broad_cos_shift_b_std: float
    broad_cos_shift_full_range: List[float]
    broad_cos_shift_b_range: List[float]
    # Low-H1 subset
    h1_window_b: List[float]
    n_low_h1: int
    low_h1_cos_shift_full_mean: float
    low_h1_cos_shift_full_std: float
    low_h1_cos_shift_b_mean: float
    low_h1_cos_shift_b_std: float
    low_h1_cross_boundary_ratio_mean: float
    low_h1_cross_boundary_ratio_std: float
    low_h1_freq_conn_rho_mean: float
    low_h1_freq_conn_rho_std: float
    low_h1_mean_curve: Dict[str, float]
    # References from earlier phases
    phase19_tachy_cosine_full: float
    phase27_simplified_cosine_full: float
    # Per-run
    broad_runs: List[ProxyRunResult]
    low_h1_run_seeds: List[int]


# ---------------------------------------------------------------------------
# Per-run encoder + measurement
# ---------------------------------------------------------------------------

def run_one_proxy(
    seed: int,
    latin_text: str,
    voy_chars_full: str,
    voy_chars_b: str,
    voy_shift_full: np.ndarray,
    voy_shift_b: np.ndarray,
    latin_curve: Dict[int, float],
    output_alpha_size: int = OUTPUT_ALPHA_SIZE,
    n_tables: int = N_TABLES,
    max_order: int = MAX_ORDER,
) -> Tuple[ProxyRunResult, List[str]]:
    """One random proxy cipher instantiation. Returns (result, tokens)."""
    rng = random.Random(seed)

    out_alpha = sorted(rng.sample(list("abcdefghijklmnopqrstuvwxyz"),
                                   output_alpha_size))
    half = output_alpha_size // 2
    prefix_chars = out_alpha[:half]
    suffix_chars = out_alpha[half:]

    pre_slots, pre_ml, pre_ns, pre_po = make_grammar_targeting_length(
        prefix_chars, rng)
    suf_slots, suf_ml, suf_ns, suf_po = make_grammar_targeting_length(
        suffix_chars, rng)

    tables = make_tables(pre_slots, suf_slots, n_tables, rng)
    tokens = encode_bigrams(latin_text, tables, rng)
    cipher_chars = "".join(tokens)

    # Size-match to Voynich-full for entropy comparison (notebook convention)
    sample = _match_corpus_size(cipher_chars, len(voy_chars_full))
    curve = entropy_curve(sample, max_order=max_order)

    orders = list(range(max_order + 1))
    shift = np.array([curve.get(k, 0.0) - latin_curve.get(k, 0.0)
                      for k in orders])
    cos_full = float(cosine_similarity(shift, voy_shift_full))
    cos_b = float(cosine_similarity(shift, voy_shift_b))

    tok_counts = Counter(tokens)
    unique_types = len(tok_counts)
    most_common_pct = 100.0 * tok_counts.most_common(1)[0][1] / len(tokens) if tokens else 0.0
    mean_tok_len = float(np.mean([len(t) for t in tokens])) if tokens else 0.0

    def _adj_overlap(slots):
        if len(slots) < 2:
            return 0.0
        vals = []
        for i in range(len(slots) - 1):
            a, b = set(slots[i][0]), set(slots[i + 1][0])
            u = a | b
            vals.append(len(a & b) / len(u) if u else 0.0)
        return float(np.mean(vals))

    result = ProxyRunResult(
        seed=seed,
        curve={k: round(v, 4) for k, v in curve.items()},
        cos_shift_full=round(cos_full, 4),
        cos_shift_b=round(cos_b, 4),
        pre_mean_len=round(pre_ml, 3),
        suf_mean_len=round(suf_ml, 3),
        mean_tok_len=round(mean_tok_len, 3),
        unique_types=unique_types,
        most_common_pct=round(most_common_pct, 3),
        pre_n_slots=pre_ns,
        suf_n_slots=suf_ns,
        pre_n_req=sum(1 for _, r in pre_slots if r),
        suf_n_req=sum(1 for _, r in suf_slots if r),
        pre_mean_slot_size=round(float(np.mean([len(sc) for sc, _ in pre_slots])), 2),
        suf_mean_slot_size=round(float(np.mean([len(sc) for sc, _ in suf_slots])), 2),
        pre_min_slot_size=min(len(sc) for sc, _ in pre_slots),
        suf_min_slot_size=min(len(sc) for sc, _ in suf_slots),
        pre_adj_overlap=round(_adj_overlap(pre_slots), 3),
        suf_adj_overlap=round(_adj_overlap(suf_slots), 3),
    )
    return result, tokens


# ---------------------------------------------------------------------------
# Plaintext / corpus loading
# ---------------------------------------------------------------------------

def _load_latin_file(path: str) -> str:
    """Load a Latin text file; keep only lowercase ASCII a-z and single spaces."""
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    cleaned = "".join(c if "a" <= c <= "z" else " " for c in raw.lower())
    return " ".join(cleaned.split())


def _load_ciphertext_file(path: str) -> List[str]:
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    return [t for t in raw.split() if t.strip()]


# ---------------------------------------------------------------------------
# External reference cipher evaluation (real Naibbe ciphertexts)
# ---------------------------------------------------------------------------

def _eval_external_cipher(
    name: str,
    tokens: List[str],
    voy_chars_full: str,
    voy_chars_b: str,
    voy_shift_full: np.ndarray,
    voy_shift_b: np.ndarray,
    latin_curve: Dict[int, float],
    max_order: int = MAX_ORDER,
) -> ExternalRefCheck:
    chars_joined = "".join(tokens)
    sample = _match_corpus_size(chars_joined, len(voy_chars_full))
    curve = entropy_curve(sample, max_order=max_order)
    orders = list(range(max_order + 1))
    shift = np.array([curve.get(k, 0.0) - latin_curve.get(k, 0.0) for k in orders])
    cos_full = float(cosine_similarity(shift, voy_shift_full))
    cos_b = float(cosine_similarity(shift, voy_shift_b))

    mi_res = _cross_boundary_ratio_plain(tokens)
    fc_res = _freq_conn_rho_plain(tokens)

    return ExternalRefCheck(
        name=name,
        n_tokens=len(tokens),
        n_chars_sampled=len(sample),
        curve={str(k): round(v, 4) for k, v in curve.items()},
        cos_shift_full=round(cos_full, 4),
        cos_shift_b=round(cos_b, 4),
        cross_boundary_ratio=round(mi_res['ratio'], 4),
        freq_conn_rho=round(fc_res['rho'], 4),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_naibbe_generalized() -> None:
    t0 = time.time()
    rd = str(_results_dir())

    print("=" * 60)
    print("Phase 88: Greshko Generalized Naibbe Cipher")
    print("=" * 60)

    # ── 1. Voynich (full + B) ───────────────────────────────────────────
    print("\n  1. Loading Voynich corpus (full + Language B) ...")
    corpus = load_corpus(verbose=False)
    voy_text_full = corpus.get_text()
    voy_text_b = corpus.get_text(language='B')
    voy_tokens_full = voy_text_full.split()
    voy_tokens_b = voy_text_b.split()
    voy_chars_full = "".join(voy_tokens_full)
    voy_chars_b = "".join(voy_tokens_b)

    voy_curve_full = entropy_curve(voy_chars_full, max_order=MAX_ORDER)
    voy_curve_b = entropy_curve(voy_chars_b, max_order=MAX_ORDER)
    print(f"    Voynich full : {len(voy_tokens_full):,} tokens, {len(voy_chars_full):,} chars")
    print(f"      H0={voy_curve_full[0]:.3f} H1={voy_curve_full[1]:.3f} "
          f"H6={voy_curve_full[MAX_ORDER]:.3f}")
    print(f"    Voynich B    : {len(voy_tokens_b):,} tokens, {len(voy_chars_b):,} chars")
    print(f"      H0={voy_curve_b[0]:.3f} H1={voy_curve_b[1]:.3f} "
          f"H6={voy_curve_b[MAX_ORDER]:.3f}")

    # ── 2. Latin plaintext + baseline ───────────────────────────────────
    print("\n  2. Loading Latin plaintext (Greshko nathist_book16.txt) ...")
    latin_path = os.path.join('data', 'reference', 'greshko', 'nathist_book16.txt')
    if not os.path.exists(latin_path):
        raise FileNotFoundError(f"Greshko Latin file missing: {latin_path}")
    latin_text = _load_latin_file(latin_path)
    latin_chars = latin_text.replace(" ", "")
    latin_curve = entropy_curve(latin_chars, max_order=MAX_ORDER)
    print(f"    Latin source : {len(latin_chars):,} chars")
    print(f"      H0={latin_curve[0]:.3f} H1={latin_curve[1]:.3f} "
          f"H6={latin_curve[MAX_ORDER]:.3f}")

    orders = list(range(MAX_ORDER + 1))
    voy_shift_full = np.array([voy_curve_full.get(k, 0.0) - latin_curve.get(k, 0.0)
                               for k in orders])
    voy_shift_b = np.array([voy_curve_b.get(k, 0.0) - latin_curve.get(k, 0.0)
                            for k in orders])
    print(f"    Shift (full) : {[round(float(v), 3) for v in voy_shift_full]}")
    print(f"    Shift (B)    : {[round(float(v), 3) for v in voy_shift_b]}")

    # ── 3. Voynich diagnostic baselines ─────────────────────────────────
    print("\n  3. Computing Voynich cross-boundary MI + freq-connectivity ...")
    from voynich.phases.currier_selfcorr import measure_cross_boundary_mi
    voy_mi_full = measure_cross_boundary_mi(voy_tokens_full)
    voy_mi_b = measure_cross_boundary_mi(voy_tokens_b)
    voy_fc_full = _freq_conn_rho_eva(voy_tokens_full)
    voy_fc_b = _freq_conn_rho_eva(voy_tokens_b)
    print(f"    Voynich full: MI ratio={voy_mi_full['ratio']:.4f} "
          f"freq-conn rho={voy_fc_full['rho']:.4f}")
    print(f"    Voynich B   : MI ratio={voy_mi_b['ratio']:.4f} "
          f"freq-conn rho={voy_fc_b['rho']:.4f}")

    # ── 4. Real Naibbe ciphertexts ──────────────────────────────────────
    print("\n  4. Evaluating real Naibbe ciphertexts ...")
    divcom_path = os.path.join('data', 'reference', 'greshko',
                                'divcom_output_ciphertext.txt')
    nathist_path = os.path.join('data', 'reference', 'greshko',
                                 'nathist_output_ciphertext.txt')
    divcom_tokens = _load_ciphertext_file(divcom_path)
    nathist_tokens = _load_ciphertext_file(nathist_path)
    divcom_res = _eval_external_cipher(
        'real_naibbe_divcom', divcom_tokens,
        voy_chars_full, voy_chars_b, voy_shift_full, voy_shift_b, latin_curve,
    )
    nathist_res = _eval_external_cipher(
        'real_naibbe_nathist', nathist_tokens,
        voy_chars_full, voy_chars_b, voy_shift_full, voy_shift_b, latin_curve,
    )
    print(f"    Divina Commedia : {divcom_res.n_tokens:,} tokens, "
          f"cos_full={divcom_res.cos_shift_full:+.4f} "
          f"cos_B={divcom_res.cos_shift_b:+.4f} "
          f"MI={divcom_res.cross_boundary_ratio:.4f} "
          f"rho={divcom_res.freq_conn_rho:.4f}")
    print(f"    Nat.Hist. Bk16  : {nathist_res.n_tokens:,} tokens, "
          f"cos_full={nathist_res.cos_shift_full:+.4f} "
          f"cos_B={nathist_res.cos_shift_b:+.4f} "
          f"MI={nathist_res.cross_boundary_ratio:.4f} "
          f"rho={nathist_res.freq_conn_rho:.4f}")

    # ── 5. Broad proxy run (200 random grammars) ────────────────────────
    print(f"\n  5. Broad proxy run (N_BROAD={N_BROAD}) ...")
    broad_runs: List[ProxyRunResult] = []
    # Keep token lists only for low-H1 subset (saves memory)
    run_tokens: Dict[int, List[str]] = {}

    # Dynamic H1 window based on Voynich B (matching notebook)
    h1_lo = voy_curve_b[1] - H1_TOL_LOW
    h1_hi = voy_curve_b[1] + H1_TOL_HIGH
    print(f"    Low-H1 window (Voynich B): [{h1_lo:.4f}, {h1_hi:.4f}]")

    for i in range(N_BROAD):
        seed = i * 997 + 13
        res, tokens = run_one_proxy(
            seed=seed,
            latin_text=latin_text,
            voy_chars_full=voy_chars_full,
            voy_chars_b=voy_chars_b,
            voy_shift_full=voy_shift_full,
            voy_shift_b=voy_shift_b,
            latin_curve=latin_curve,
        )
        broad_runs.append(res)
        if h1_lo <= res.curve.get(1, 0.0) <= h1_hi:
            run_tokens[seed] = tokens
        if (i + 1) % 25 == 0 or i == 0:
            print(f"    run {i + 1:4d}/{N_BROAD}: H1={res.curve[1]:.3f} "
                  f"cos_full={res.cos_shift_full:+.3f} "
                  f"cos_B={res.cos_shift_b:+.3f}")

    # Extend with more seeds if low-H1 count < 10 (notebook behavior)
    extra_seed_idx = N_BROAD
    while len(run_tokens) < 10 and extra_seed_idx < N_BROAD * 3:
        seed = extra_seed_idx * 997 + 13
        res, tokens = run_one_proxy(
            seed=seed,
            latin_text=latin_text,
            voy_chars_full=voy_chars_full,
            voy_chars_b=voy_chars_b,
            voy_shift_full=voy_shift_full,
            voy_shift_b=voy_shift_b,
            latin_curve=latin_curve,
        )
        broad_runs.append(res)
        if h1_lo <= res.curve.get(1, 0.0) <= h1_hi:
            run_tokens[seed] = tokens
        extra_seed_idx += 1

    # ── 6. Low-H1 subset: compute MI + freq-conn ────────────────────────
    low_h1_seeds = sorted(run_tokens.keys())
    print(f"\n  6. Low-H1 subset: {len(low_h1_seeds)} runs")
    print(f"     Computing cross-boundary MI + freq-conn for each ...")
    for seed in low_h1_seeds:
        toks = run_tokens[seed]
        mi = _cross_boundary_ratio_plain(toks)
        fc = _freq_conn_rho_plain(toks)
        for r in broad_runs:
            if r.seed == seed:
                r.cross_boundary_ratio = round(mi['ratio'], 4)
                r.freq_conn_rho = round(fc['rho'], 4)
                break

    # Aggregates
    broad_full = [r.cos_shift_full for r in broad_runs]
    broad_b = [r.cos_shift_b for r in broad_runs]
    low_h1_runs = [r for r in broad_runs if r.seed in run_tokens]
    low_h1_full = [r.cos_shift_full for r in low_h1_runs]
    low_h1_b = [r.cos_shift_b for r in low_h1_runs]
    low_h1_mi = [r.cross_boundary_ratio for r in low_h1_runs if r.cross_boundary_ratio is not None]
    low_h1_fc = [r.freq_conn_rho for r in low_h1_runs if r.freq_conn_rho is not None]

    # Low-H1 mean curve
    low_h1_mean_curve: Dict[int, float] = {}
    for k in orders:
        vals = [r.curve.get(k, 0.0) for r in low_h1_runs]
        low_h1_mean_curve[k] = round(float(np.mean(vals)) if vals else 0.0, 4)

    print(f"\n     Broad  cos_full: mean={np.mean(broad_full):+.4f} "
          f"std={np.std(broad_full):.4f}  "
          f"range=[{np.min(broad_full):+.4f}, {np.max(broad_full):+.4f}]")
    print(f"     Broad  cos_B   : mean={np.mean(broad_b):+.4f} "
          f"std={np.std(broad_b):.4f}  "
          f"range=[{np.min(broad_b):+.4f}, {np.max(broad_b):+.4f}]")
    if low_h1_runs:
        print(f"     Low-H1 cos_full: mean={np.mean(low_h1_full):+.4f} "
              f"std={np.std(low_h1_full):.4f}")
        print(f"     Low-H1 cos_B   : mean={np.mean(low_h1_b):+.4f} "
              f"std={np.std(low_h1_b):.4f}")
        if low_h1_mi:
            print(f"     Low-H1 MI ratio: mean={np.mean(low_h1_mi):.4f} "
                  f"std={np.std(low_h1_mi):.4f}")
        if low_h1_fc:
            print(f"     Low-H1 freq-conn rho: mean={np.mean(low_h1_fc):+.4f} "
                  f"std={np.std(low_h1_fc):.4f}")

    # ── 7. Load Phase 19/27 references ──────────────────────────────────
    phase19_cos = 0.820  # default
    shift_path = os.path.join(rd, 'entropy_shift_cipher.json')
    if os.path.exists(shift_path):
        with open(shift_path) as f:
            d = json.load(f)
        phase19_cos = float(d.get('best_match_cosine', 0.820))

    phase27_cos = -0.843  # default
    naibbe_path = os.path.join(rd, 'naibbe_entropy.json')
    if os.path.exists(naibbe_path):
        with open(naibbe_path) as f:
            d = json.load(f)
        phase27_cos = float(d.get('greshko_cosine', -0.843))

    # ── 8. Save ──────────────────────────────────────────────────────────
    result = NaibbeGeneralizedResult(
        timestamp=time.strftime('%Y-%m-%dT%H:%M:%S'),
        runtime_seconds=round(time.time() - t0, 2),
        parameters={
            'OUTPUT_ALPHA_SIZE': OUTPUT_ALPHA_SIZE,
            'N_TABLES': N_TABLES,
            'TABLE_WEIGHTS': TABLE_WEIGHTS,
            'N_BROAD': N_BROAD,
            'MAX_ORDER': MAX_ORDER,
            'H1_TOL_LOW': H1_TOL_LOW,
            'H1_TOL_HIGH': H1_TOL_HIGH,
            'seed_formula': 'i*997+13',
            'latin_source': 'data/reference/greshko/nathist_book16.txt',
        },
        voynich_full_curve={str(k): round(v, 4) for k, v in voy_curve_full.items()},
        voynich_b_curve={str(k): round(v, 4) for k, v in voy_curve_b.items()},
        latin_curve={str(k): round(v, 4) for k, v in latin_curve.items()},
        voynich_full_shift=[round(float(v), 4) for v in voy_shift_full],
        voynich_b_shift=[round(float(v), 4) for v in voy_shift_b],
        voynich_full_cross_boundary_ratio=round(voy_mi_full['ratio'], 4),
        voynich_b_cross_boundary_ratio=round(voy_mi_b['ratio'], 4),
        voynich_full_freq_conn_rho=round(voy_fc_full['rho'], 4),
        voynich_b_freq_conn_rho=round(voy_fc_b['rho'], 4),
        real_naibbe_divcom=divcom_res,
        real_naibbe_nathist=nathist_res,
        n_broad=len(broad_runs),
        broad_cos_shift_full_mean=round(float(np.mean(broad_full)), 4),
        broad_cos_shift_full_std=round(float(np.std(broad_full)), 4),
        broad_cos_shift_b_mean=round(float(np.mean(broad_b)), 4),
        broad_cos_shift_b_std=round(float(np.std(broad_b)), 4),
        broad_cos_shift_full_range=[round(float(np.min(broad_full)), 4),
                                    round(float(np.max(broad_full)), 4)],
        broad_cos_shift_b_range=[round(float(np.min(broad_b)), 4),
                                 round(float(np.max(broad_b)), 4)],
        h1_window_b=[round(h1_lo, 4), round(h1_hi, 4)],
        n_low_h1=len(low_h1_runs),
        low_h1_cos_shift_full_mean=round(float(np.mean(low_h1_full)), 4) if low_h1_full else 0.0,
        low_h1_cos_shift_full_std=round(float(np.std(low_h1_full)), 4) if low_h1_full else 0.0,
        low_h1_cos_shift_b_mean=round(float(np.mean(low_h1_b)), 4) if low_h1_b else 0.0,
        low_h1_cos_shift_b_std=round(float(np.std(low_h1_b)), 4) if low_h1_b else 0.0,
        low_h1_cross_boundary_ratio_mean=round(float(np.mean(low_h1_mi)), 4) if low_h1_mi else 0.0,
        low_h1_cross_boundary_ratio_std=round(float(np.std(low_h1_mi)), 4) if low_h1_mi else 0.0,
        low_h1_freq_conn_rho_mean=round(float(np.mean(low_h1_fc)), 4) if low_h1_fc else 0.0,
        low_h1_freq_conn_rho_std=round(float(np.std(low_h1_fc)), 4) if low_h1_fc else 0.0,
        low_h1_mean_curve={str(k): v for k, v in low_h1_mean_curve.items()},
        phase19_tachy_cosine_full=round(phase19_cos, 4),
        phase27_simplified_cosine_full=round(phase27_cos, 4),
        broad_runs=broad_runs,
        low_h1_run_seeds=low_h1_seeds,
    )

    out_path = os.path.join(rd, 'p88_naibbe_generalized.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2)
    print(f"\n  -> {out_path}")
    print(f"  Runtime: {result.runtime_seconds:.1f}s")
