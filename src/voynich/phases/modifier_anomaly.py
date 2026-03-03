"""
Phase 16.2 – Frequency Anomaly Detection (Approach D)
=====================================================
Detects modifier status through frequency anomalies and co-occurrence
patterns: Zipf residuals, obligatory co-occurrence, positional entropy,
and token-length correlation.

In any writing system, modifiers (diacritics, vowel killers) have
anomalous frequency patterns: they're overrepresented for their type
rank, they always co-occur with specific stem characters, and their
presence inflates token length without adding semantic content.

Dependency chain:
    corpus (IVTFF)
        → modifier_anomaly.json (this step)
"""

import json
import math
import os
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
from voynich.core.reference import EVA_VISUAL_COMPONENTS


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
class AnomalyProfile:
    """Anomaly signals for a single EVA character."""
    eva_char: str
    triple_key: str
    frequency: int
    zipf_rank: int
    zipf_expected_freq: float
    zipf_residual: float          # positive = overrepresented
    top_cooccurrence_partner: str
    top_cooccurrence_strength: float  # P(partner | this_char)
    obligatory_partner: Optional[str]  # partner with P > 0.90
    length_correlation: float     # Pearson r with token length
    positional_concentration: float  # max(initial%, medial%, final%)
    anomaly_score: float


@dataclass
class AnomalyResult:
    n_chars_analyzed: int
    anomaly_profiles: List[Dict]
    modifier_candidates: List[str]
    modifier_threshold: float
    zipf_alpha: float
    zipf_c: float
    obligatory_pairs: List[Dict]  # char -> partner with P > 0.90
    gate_passed: bool
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def compute_zipf_residuals(
    char_freqs: Counter,
) -> Tuple[Dict[str, float], float, float]:
    """Fit Zipf's law and compute per-character residuals.

    Returns (residuals_dict, alpha, C) where
    expected_freq(rank) = C / rank^alpha
    residual = (observed - expected) / expected
    Positive residual = overrepresented for rank.
    """
    if not char_freqs:
        return {}, 1.0, 0.0

    # Sort by frequency descending
    ranked = char_freqs.most_common()
    ranks = np.arange(1, len(ranked) + 1, dtype=float)
    freqs = np.array([f for _, f in ranked], dtype=float)

    # Fit log-log linear regression: log(freq) = log(C) - alpha * log(rank)
    log_ranks = np.log(ranks)
    log_freqs = np.log(np.maximum(freqs, 1))

    # Least squares fit
    n = len(ranks)
    mean_lr = log_ranks.mean()
    mean_lf = log_freqs.mean()
    cov = ((log_ranks - mean_lr) * (log_freqs - mean_lf)).sum()
    var = ((log_ranks - mean_lr) ** 2).sum()
    alpha = -cov / var if var > 0 else 1.0
    log_c = mean_lf + alpha * mean_lr
    C = math.exp(log_c)

    # Compute residuals
    residuals = {}
    for i, (ch, freq) in enumerate(ranked):
        expected = C / (ranks[i] ** alpha)
        residuals[ch] = (freq - expected) / max(expected, 1)

    return residuals, alpha, C


def compute_obligatory_cooccurrence(
    tokens: List[str],
) -> Dict[str, Dict[str, float]]:
    """For each EVA char, compute P(other_char adjacent | this_char present).

    Returns dict: char -> {neighbour -> P(neighbour_adjacent | char)}
    """
    # Count tokens containing each char
    char_token_count: Counter = Counter()
    # Count tokens where char is adjacent to specific neighbour
    adjacency_count: Dict[str, Counter] = defaultdict(Counter)

    for token in tokens:
        chars = tokenize_eva_chars(token)
        seen = set()
        for ci, ch in enumerate(chars):
            if ch not in seen:
                char_token_count[ch] += 1
                seen.add(ch)
            # Record adjacency
            if ci > 0:
                adjacency_count[ch][chars[ci - 1]] += 1
            if ci < len(chars) - 1:
                adjacency_count[ch][chars[ci + 1]] += 1

    # Compute conditional probabilities
    result: Dict[str, Dict[str, float]] = {}
    for ch in adjacency_count:
        total_adj = sum(adjacency_count[ch].values())
        if total_adj > 0:
            result[ch] = {
                nbr: count / total_adj
                for nbr, count in adjacency_count[ch].most_common()
            }
        else:
            result[ch] = {}

    return result


def compute_length_correlation(
    tokens: List[str],
) -> Dict[str, float]:
    """Pearson correlation between each char's presence and token length.

    Modifiers should positively correlate with longer tokens (they inflate
    length without adding semantic content).
    """
    # Parse all tokens
    parsed = [tokenize_eva_chars(t) for t in tokens]
    lengths = np.array([len(p) for p in parsed], dtype=float)

    # Get all chars
    all_chars: Set[str] = set()
    for p in parsed:
        all_chars.update(p)

    # For each char, compute correlation
    mean_len = lengths.mean()
    std_len = lengths.std()
    correlations: Dict[str, float] = {}

    for ch in all_chars:
        presence = np.array([1.0 if ch in p else 0.0 for p in parsed])
        mean_p = presence.mean()
        std_p = presence.std()

        if std_p > 0 and std_len > 0:
            cov = ((presence - mean_p) * (lengths - mean_len)).mean()
            correlations[ch] = cov / (std_p * std_len)
        else:
            correlations[ch] = 0.0

    return correlations


def compute_positional_concentration(
    tokens: List[str],
) -> Dict[str, float]:
    """Max positional fraction for each char. High = positionally locked."""
    pos_counts: Dict[str, Counter] = defaultdict(Counter)

    for token in tokens:
        chars = tokenize_eva_chars(token)
        n = len(chars)
        for ci, ch in enumerate(chars):
            if n == 1:
                pos_counts[ch]['solo'] += 1
            elif ci == 0:
                pos_counts[ch]['initial'] += 1
            elif ci == n - 1:
                pos_counts[ch]['final'] += 1
            else:
                pos_counts[ch]['medial'] += 1

    result: Dict[str, float] = {}
    for ch, counts in pos_counts.items():
        total = sum(counts.values())
        if total > 0:
            result[ch] = max(counts.values()) / total
        else:
            result[ch] = 0.0
    return result


def score_anomaly(
    zipf_residual: float,
    top_cooccurrence_strength: float,
    length_corr: float,
    positional_conc: float,
) -> float:
    """Combine anomaly signals into 0-1 score.

    Components:
      0.25 * high Zipf residual (overrepresented)
      0.25 * high obligatory co-occurrence (restricted context)
      0.25 * high length correlation (inflates token length)
      0.25 * high positional concentration (positionally locked)
    """
    # Zipf: sigmoid transform centred at 0
    zipf_sig = 1 / (1 + math.exp(-2 * zipf_residual))

    # Co-occurrence: already in [0, 1]
    cooc = min(top_cooccurrence_strength, 1.0)

    # Length correlation: shift from [-1, 1] to [0, 1]
    len_sig = (length_corr + 1) / 2

    # Positional concentration: already in [0, 1]
    pos = min(positional_conc, 1.0)

    return 0.25 * zipf_sig + 0.25 * cooc + 0.25 * len_sig + 0.25 * pos


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_modifier_anomaly() -> None:
    """Step 16.2: Frequency anomaly modifier detection (Approach D)."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE 16.2: Frequency Anomaly Detection (Approach D)")
    print("=" * 70)

    rd = _results_dir()

    # ─── Load corpus ───
    print("\n  1. Loading corpus …")
    corpus = load_corpus(verbose=False)
    tokens = corpus.get_tokens()
    print(f"      {len(tokens)} tokens loaded")

    eva_to_triple = build_eva_to_triple_lookup()

    # ─── Character frequencies ───
    print("\n  2. Computing character frequencies …")
    char_freqs: Counter = Counter()
    for token in tokens:
        for ch in tokenize_eva_chars(token):
            char_freqs[ch] += 1
    print(f"      {len(char_freqs)} distinct chars/ligatures")

    # ─── Zipf residuals ───
    print("\n  3. Fitting Zipf's law …")
    residuals, alpha, C = compute_zipf_residuals(char_freqs)
    print(f"      Zipf α = {alpha:.3f}, C = {C:.1f}")

    # ─── Obligatory co-occurrence ───
    print("\n  4. Computing co-occurrence patterns …")
    cooccurrence = compute_obligatory_cooccurrence(tokens)

    # ─── Length correlation ───
    print("\n  5. Computing token-length correlations …")
    length_corr = compute_length_correlation(tokens)

    # ─── Positional concentration ───
    print("\n  6. Computing positional concentration …")
    pos_conc = compute_positional_concentration(tokens)

    # ─── Build profiles ───
    print("\n  7. Building anomaly profiles …")
    profiles: List[AnomalyProfile] = []

    for ch in sorted(EVA_VISUAL_COMPONENTS.keys()):
        freq = char_freqs.get(ch, 0)
        if freq == 0:
            continue

        triple_key = eva_to_triple.get(ch, '?')
        zipf_res = residuals.get(ch, 0.0)

        # Zipf rank
        ranked = char_freqs.most_common()
        zipf_rank = next(
            (i + 1 for i, (c, _) in enumerate(ranked) if c == ch), 0
        )

        # Top co-occurrence
        cooc_dict = cooccurrence.get(ch, {})
        if cooc_dict:
            top_partner = max(cooc_dict, key=cooc_dict.get)
            top_strength = cooc_dict[top_partner]
        else:
            top_partner = ''
            top_strength = 0.0

        # Obligatory partner (P > 0.90)
        obligatory = None
        for partner, p in cooc_dict.items():
            if p > 0.90:
                obligatory = partner
                break

        len_c = length_corr.get(ch, 0.0)
        pc = pos_conc.get(ch, 0.0)

        anom_score = score_anomaly(zipf_res, top_strength, len_c, pc)

        # Expected frequency under Zipf
        zipf_expected = C / (zipf_rank ** alpha) if zipf_rank > 0 else 0.0

        profiles.append(AnomalyProfile(
            eva_char=ch,
            triple_key=triple_key,
            frequency=freq,
            zipf_rank=zipf_rank,
            zipf_expected_freq=round(zipf_expected, 1),
            zipf_residual=round(zipf_res, 4),
            top_cooccurrence_partner=top_partner,
            top_cooccurrence_strength=round(top_strength, 4),
            obligatory_partner=obligatory,
            length_correlation=round(len_c, 4),
            positional_concentration=round(pc, 4),
            anomaly_score=round(anom_score, 4),
        ))

    profiles.sort(key=lambda p: p.anomaly_score, reverse=True)

    # ─── Identify modifier candidates ───
    threshold = 0.5
    modifier_candidates = [p.eva_char for p in profiles if p.anomaly_score >= threshold]

    # Obligatory pairs
    obligatory_pairs = [
        {'char': p.eva_char, 'partner': p.obligatory_partner,
         'strength': p.top_cooccurrence_strength}
        for p in profiles if p.obligatory_partner is not None
    ]

    # ─── Print top profiles ───
    print(f"\n  8. Top 15 profiles by anomaly score:")
    print(f"      {'Char':<8} {'Freq':>6} {'Rank':>5} {'Zipf.Res':>9} "
          f"{'TopCooc':>8} {'Str':>6} {'Len.r':>6} {'Pos.C':>6} {'Score':>6}")
    print("      " + "-" * 80)
    for p in profiles[:15]:
        print(f"      {p.eva_char:<8} {p.frequency:>6} {p.zipf_rank:>5} "
              f"{p.zipf_residual:>9.3f} {p.top_cooccurrence_partner:>8} "
              f"{p.top_cooccurrence_strength:>6.3f} "
              f"{p.length_correlation:>6.3f} {p.positional_concentration:>6.3f} "
              f"{p.anomaly_score:>6.3f}")

    if obligatory_pairs:
        print(f"\n  9. Obligatory co-occurrence pairs (P > 0.90):")
        for pair in obligatory_pairs:
            print(f"      {pair['char']} → {pair['partner']} "
                  f"(P = {pair['strength']:.3f})")

    # ─── Gate ───
    gate_passed = len(modifier_candidates) >= 3
    verdict = (
        f"PASS: {len(modifier_candidates)} chars with anomaly score >= {threshold}. "
        f"{len(obligatory_pairs)} obligatory co-occurrence pairs found."
        if gate_passed
        else f"FAIL: Only {len(modifier_candidates)} anomalous chars "
        f"(need >= 3 with score >= {threshold})."
    )

    print(f"\n  Gate: {'PASS' if gate_passed else 'FAIL'}")
    print(f"  {verdict}")

    # ─── Save ───
    result = AnomalyResult(
        n_chars_analyzed=len(profiles),
        anomaly_profiles=[_convert(asdict(p)) for p in profiles],
        modifier_candidates=modifier_candidates,
        modifier_threshold=threshold,
        zipf_alpha=round(alpha, 4),
        zipf_c=round(C, 2),
        obligatory_pairs=obligatory_pairs,
        gate_passed=gate_passed,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'modifier_anomaly.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)

    print(f"\n  → {out_path}")
