"""
Phase 10.1 — Token-Level Entropy Curves
=========================================

Rationale
---------
The three surviving hypotheses (H1: constructed script, H2: information
dispersion, H3: keyed cipher) predict different shapes for the token-level
conditional entropy curve H(token_t | context of length n).

Section/language strategy:
  - Language A combined, Language A herbal-only, Language A pharmaceutical-only
  - Language B as negative control (mechanical, should be flat)
  - If H1: all three A curves have the same shape (same script)
  - If H3 + key varies by section: curves differ between sections

Sub-analyses
------------
10.1a  Build token-level conditional entropy series at orders 0,1,2,3,5,10
10.1b  Build reference entropy curves (Latin, Occitan, Italian, German,
       shuffled Voynich, Markov-order-2 Voynich)
10.1c  Section consistency analysis (herbal vs pharma vs combined)
10.1d  Quantitative hypothesis discrimination
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import load_corpus
from voynich.core.reference import load_reference_corpus
from voynich.core.stats import (
    pearson_correlation,
    token_entropy_curve,
    word_unigram_entropy,
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TokenEntropyCurve:
    label: str
    curve: Dict[int, float]           # {order: H}
    reduction_rates: Dict[str, float]  # {"R_1": val, ...}
    decay_rate: float                  # linear slope of H vs order
    floor: float                       # H at max order


@dataclass
class SectionConsistency:
    combined_curve: Dict               # serialized TokenEntropyCurve
    herbal_curve: Dict
    pharma_curve: Dict
    lang_b_curve: Dict                 # negative control
    herbal_pharma_correlation: float   # Pearson r between section R(n)
    combined_vs_herbal_correlation: float
    sections_consistent: bool          # r > 0.95 → H1 consistent


@dataclass
class HypothesisScore:
    h1_correlation: float              # Pearson r of R(n) vs best ref
    h1_best_match: str
    h1_section_consistent: bool
    h2_backload_ratio: float           # R(5→10) / R(1→2)
    h2_ref_ratios: Dict[str, float]
    h3_floor_ratio: float              # voynich H10 / mean ref H10
    h3_context_decay: float            # R(10) / R(1)
    h3_section_divergent: bool
    best_hypothesis: str


@dataclass
class EntropyCurveResult:
    section_analysis: Dict
    reference_curves: Dict[str, Dict]
    shuffled_curve: Dict
    markov2_curve: Dict
    hypothesis_scores: Dict
    gate_passed: bool
    verdict: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _convert(obj: Any) -> Any:
    """Convert dataclass/numpy types to JSON-serializable form."""
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
    if isinstance(obj, (bool, int, float, str, type(None))):
        return obj
    return str(obj)


ORDERS = (0, 1, 2, 3, 5, 10)


def _build_token_entropy_curve(tokens: List[str], label: str) -> TokenEntropyCurve:
    """Compute token-level entropy at each context order."""
    curve = token_entropy_curve(tokens, orders=ORDERS)

    # Reduction rates: R(n) = H(0) - H(n)
    h0 = curve.get(0, 0.0)
    reduction_rates = {}
    for o in ORDERS:
        if o == 0:
            continue
        reduction_rates[f"R_{o}"] = h0 - curve.get(o, h0)

    # Decay rate: linear slope of H vs order
    orders_list = sorted(curve.keys())
    if len(orders_list) >= 2:
        x = np.array(orders_list, dtype=float)
        y = np.array([curve[o] for o in orders_list], dtype=float)
        A = np.vstack([x, np.ones(len(x))]).T
        slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
        decay_rate = float(slope)
    else:
        decay_rate = 0.0

    floor = curve.get(max(ORDERS), curve.get(0, 0.0))

    return TokenEntropyCurve(
        label=label,
        curve=curve,
        reduction_rates=reduction_rates,
        decay_rate=decay_rate,
        floor=floor,
    )


def _build_shuffled_baseline(tokens: List[str], seed: int = 42) -> List[str]:
    """Return a copy of tokens with order randomized."""
    shuffled = list(tokens)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    return shuffled


def _generate_markov_tokens(
    source_text: str, n_chars: int, order: int = 2, seed: int = 42,
) -> List[str]:
    """Generate text from a character-level Markov model and tokenize."""
    chars = list(source_text)
    if len(chars) <= order:
        return source_text.split()

    transitions: Dict[tuple, List[str]] = {}
    for i in range(len(chars) - order):
        ctx = tuple(chars[i:i + order])
        nxt = chars[i + order]
        transitions.setdefault(ctx, []).append(nxt)

    rng = random.Random(seed)
    start = rng.randint(0, len(chars) - order - 1)
    ctx = tuple(chars[start:start + order])
    result = list(ctx)

    for _ in range(n_chars - order):
        candidates = transitions.get(ctx)
        if not candidates:
            start = rng.randint(0, len(chars) - order - 1)
            ctx = tuple(chars[start:start + order])
            result.append(ctx[-1])
            continue
        ch = rng.choice(candidates)
        result.append(ch)
        ctx = (*ctx[1:], ch)

    return ''.join(result).split()


def _reduction_rate_vector(curve: TokenEntropyCurve) -> np.ndarray:
    """Extract ordered R(n) vector for correlation comparison."""
    keys = sorted(curve.reduction_rates.keys(),
                  key=lambda k: int(k.split('_')[1]))
    return np.array([curve.reduction_rates[k] for k in keys])


def _section_consistency(
    combined: TokenEntropyCurve,
    herbal: TokenEntropyCurve,
    pharma: TokenEntropyCurve,
    lang_b: TokenEntropyCurve,
) -> SectionConsistency:
    """Compare entropy curves across sections."""
    r_herbal = _reduction_rate_vector(herbal)
    r_pharma = _reduction_rate_vector(pharma)
    r_combined = _reduction_rate_vector(combined)

    if len(r_herbal) >= 2 and len(r_pharma) >= 2:
        hp_corr, _ = pearson_correlation(r_herbal, r_pharma)
    else:
        hp_corr = 0.0

    if len(r_combined) >= 2 and len(r_herbal) >= 2:
        ch_corr, _ = pearson_correlation(r_combined, r_herbal)
    else:
        ch_corr = 0.0

    return SectionConsistency(
        combined_curve=_convert(asdict(combined)),
        herbal_curve=_convert(asdict(herbal)),
        pharma_curve=_convert(asdict(pharma)),
        lang_b_curve=_convert(asdict(lang_b)),
        herbal_pharma_correlation=hp_corr,
        combined_vs_herbal_correlation=ch_corr,
        sections_consistent=hp_corr > 0.95,
    )


def _score_hypotheses(
    section: SectionConsistency,
    combined: TokenEntropyCurve,
    ref_curves: Dict[str, TokenEntropyCurve],
) -> HypothesisScore:
    """Score each hypothesis from entropy curve evidence."""
    r_voynich = _reduction_rate_vector(combined)

    # H1: correlation of R(n) with each reference language
    best_corr = -1.0
    best_lang = 'none'
    for lang, rc in ref_curves.items():
        r_ref = _reduction_rate_vector(rc)
        if len(r_ref) >= 2 and len(r_voynich) >= 2:
            corr, _ = pearson_correlation(r_voynich, r_ref)
            if corr > best_corr:
                best_corr = corr
                best_lang = lang

    # H2: back-loaded ratio = R(5→10) / R(1→2)
    rr = combined.reduction_rates
    r_1 = rr.get('R_1', 0.0)
    r_2 = rr.get('R_2', 0.0)
    r_5 = rr.get('R_5', 0.0)
    r_10 = rr.get('R_10', 0.0)
    r_early = r_2 - r_1 if r_2 > r_1 else max(r_2, 0.001)
    r_late = r_10 - r_5
    backload_ratio = r_late / r_early if r_early > 0.001 else 0.0

    # H2 reference ratios
    h2_ref_ratios = {}
    for lang, rc in ref_curves.items():
        rr_ref = rc.reduction_rates
        r1r = rr_ref.get('R_1', 0.0)
        r2r = rr_ref.get('R_2', 0.0)
        r5r = rr_ref.get('R_5', 0.0)
        r10r = rr_ref.get('R_10', 0.0)
        early_r = r2r - r1r if r2r > r1r else max(r2r, 0.001)
        late_r = r10r - r5r
        h2_ref_ratios[lang] = late_r / early_r if early_r > 0.001 else 0.0

    # H3: floor ratio and context decay
    ref_floors = [rc.floor for rc in ref_curves.values() if rc.floor > 0]
    mean_ref_floor = np.mean(ref_floors) if ref_floors else 1.0
    h3_floor_ratio = combined.floor / mean_ref_floor if mean_ref_floor > 0 else 0.0

    h3_context_decay = r_10 / r_1 if r_1 > 0.001 else 0.0

    # Section consistency for H1 vs H3
    h1_section_consistent = section.sections_consistent
    h3_section_divergent = section.herbal_pharma_correlation < 0.80

    # Determine best hypothesis
    h1_score = best_corr + (0.3 if h1_section_consistent else 0.0)
    h2_score = backload_ratio / max(
        max(h2_ref_ratios.values()) if h2_ref_ratios else 1.0, 0.001)
    h3_score = h3_floor_ratio * (1.5 if h3_section_divergent else 0.5)

    scores = {'H1': h1_score, 'H2': h2_score, 'H3': h3_score}
    best = max(scores, key=scores.get)

    return HypothesisScore(
        h1_correlation=best_corr,
        h1_best_match=best_lang,
        h1_section_consistent=h1_section_consistent,
        h2_backload_ratio=backload_ratio,
        h2_ref_ratios=h2_ref_ratios,
        h3_floor_ratio=h3_floor_ratio,
        h3_context_decay=h3_context_decay,
        h3_section_divergent=h3_section_divergent,
        best_hypothesis=best,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_entropy_curves() -> Dict[str, Any]:
    """Run Phase 10.1: token-level entropy curves."""
    print("=" * 60)
    print("Phase 10.1 — Token-Level Entropy Curves")
    print("=" * 60)

    # --- Load data ---
    corpus = load_corpus(verbose=False)
    ref_corpus = load_reference_corpus(verbose=False)

    # --- Voynich section curves ---
    print("\n  Building Voynich section curves...")
    tokens_a = corpus.get_tokens(language='A')
    tokens_herbal = corpus.get_tokens(section='herbal_a', paragraph_only=True)
    tokens_pharma = corpus.get_tokens(section='pharmaceutical', paragraph_only=True)
    tokens_b = corpus.get_tokens(language='B')

    print(f"    Language A combined: {len(tokens_a)} tokens")
    print(f"    Language A herbal:   {len(tokens_herbal)} tokens")
    print(f"    Language A pharma:   {len(tokens_pharma)} tokens")
    print(f"    Language B:          {len(tokens_b)} tokens")

    curve_combined = _build_token_entropy_curve(tokens_a, 'voynich_A_combined')
    curve_herbal = _build_token_entropy_curve(tokens_herbal, 'voynich_A_herbal')
    curve_pharma = _build_token_entropy_curve(tokens_pharma, 'voynich_A_pharma')
    curve_lang_b = _build_token_entropy_curve(tokens_b, 'voynich_B')

    section = _section_consistency(curve_combined, curve_herbal, curve_pharma, curve_lang_b)

    print(f"    Herbal-Pharma correlation: {section.herbal_pharma_correlation:.3f}")
    print(f"    Sections consistent: {section.sections_consistent}")

    # --- Reference language curves ---
    print("\n  Building reference language curves...")
    ref_curves: Dict[str, TokenEntropyCurve] = {}
    for lang in ref_corpus.languages:
        ref_tokens = ref_corpus.get_combined_tokens(lang)
        if len(ref_tokens) < 100:
            continue
        ref_curves[lang] = _build_token_entropy_curve(ref_tokens, lang)
        print(f"    {lang}: {len(ref_tokens)} tokens, "
              f"H0={ref_curves[lang].curve.get(0, 0):.3f}, "
              f"floor={ref_curves[lang].floor:.3f}")

    # --- Baselines ---
    print("\n  Building baselines...")
    shuffled_tokens = _build_shuffled_baseline(tokens_a)
    curve_shuffled = _build_token_entropy_curve(shuffled_tokens, 'shuffled')
    print(f"    Shuffled: H0={curve_shuffled.curve.get(0, 0):.3f}, "
          f"floor={curve_shuffled.floor:.3f}")

    voynich_text = corpus.get_text(language='A')
    markov_tokens = _generate_markov_tokens(voynich_text, len(voynich_text), order=2)
    curve_markov = _build_token_entropy_curve(markov_tokens, 'markov_order2')
    print(f"    Markov-2: {len(markov_tokens)} tokens, "
          f"H0={curve_markov.curve.get(0, 0):.3f}, "
          f"floor={curve_markov.floor:.3f}")

    # --- Hypothesis scoring ---
    print("\n  Scoring hypotheses...")
    scores = _score_hypotheses(section, curve_combined, ref_curves)

    print(f"    H1 correlation:    {scores.h1_correlation:.3f} (best: {scores.h1_best_match})")
    print(f"    H1 section consis: {scores.h1_section_consistent}")
    print(f"    H2 backload ratio: {scores.h2_backload_ratio:.3f}")
    print(f"    H3 floor ratio:    {scores.h3_floor_ratio:.3f}")
    print(f"    H3 section diverg: {scores.h3_section_divergent}")
    print(f"\n    Best hypothesis:   {scores.best_hypothesis}")

    # --- Gate ---
    gate_passed = scores.h1_correlation > 0.5 or scores.h2_backload_ratio > 1.5 or scores.h3_floor_ratio > 1.3

    if scores.best_hypothesis == 'H1':
        verdict = (f"entropy_curve_supports_H1_constructed_script: "
                   f"parallel shift (r={scores.h1_correlation:.3f}) with "
                   f"{scores.h1_best_match}, sections consistent={scores.h1_section_consistent}")
    elif scores.best_hypothesis == 'H2':
        verdict = (f"entropy_curve_supports_H2_information_dispersion: "
                   f"back-loaded reduction ratio={scores.h2_backload_ratio:.3f}")
    else:
        verdict = (f"entropy_curve_supports_H3_keyed_cipher: "
                   f"floor ratio={scores.h3_floor_ratio:.3f}, "
                   f"section divergent={scores.h3_section_divergent}")

    print(f"\n  Gate passed: {gate_passed}")
    print(f"  Verdict: {verdict}")

    # --- Voynich combined curve summary ---
    print("\n  Voynich A combined entropy curve:")
    for o in ORDERS:
        h = curve_combined.curve.get(o, 0.0)
        r = curve_combined.reduction_rates.get(f'R_{o}', 0.0) if o > 0 else 0.0
        tag = f"  R={r:.3f}" if o > 0 else ""
        print(f"    H(token | {o:2d} ctx) = {h:.4f}{tag}")

    # --- Save ---
    result = EntropyCurveResult(
        section_analysis=_convert(asdict(section)),
        reference_curves={k: _convert(asdict(v)) for k, v in ref_curves.items()},
        shuffled_curve=_convert(asdict(curve_shuffled)),
        markov2_curve=_convert(asdict(curve_markov)),
        hypothesis_scores=_convert(asdict(scores)),
        gate_passed=gate_passed,
        verdict=verdict,
    )

    out = _convert(asdict(result))
    out_path = _results_dir() / 'entropy_curves.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n  Results saved to {out_path}")

    return out
