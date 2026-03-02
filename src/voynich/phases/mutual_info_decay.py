"""
Phase 10.2 — Multi-Token Mutual Information Decay
===================================================

Rationale
---------
If information is dispersed across multiple tokens (H2), distant tokens carry
more mutual information about each other than in natural language.  The MI
decay curve's time constant τ distinguishes H2 from H1 and H3.

Section strategy:
  - Language A only, but compute per-section τ values.
  - If H2 is correct, τ should be similar across sections (consistent mechanism).
  - If τ differs dramatically between herbal and pharmaceutical, the "dispersion"
    might be section-specific rather than a property of the encoding.

Sub-analyses
------------
10.2a  Token-gap MI at distances d = 1,2,3,5,10,20
10.2b  Exponential decay fit → τ comparison
10.2c  Per-section τ consistency
10.2d  Phrase-level Procrustes alignment (if H2 supported)
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
    build_cooccurrence_matrix,
    fit_exponential_decay,
    mutual_information_lag,
    ppmi_matrix,
    procrustes_alignment,
    selectivity_ratio,
    truncated_svd,
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class MIDecayCurve:
    label: str
    mi_values: Dict[int, float]    # {lag: MI}
    tau: float                      # exponential decay constant
    amplitude: float                # A from fit
    fit_r_squared: float


@dataclass
class SectionTauComparison:
    combined_tau: float
    herbal_tau: float
    pharma_tau: float
    tau_consistent: bool            # herbal/pharma within 30% of each other


@dataclass
class PhraseAlignmentResult:
    phrase_length: int
    procrustes_residual: float
    selectivity_vs_token: float
    improvement_over_token: bool


@dataclass
class MIDecayResult:
    voynich_mi: Dict
    section_tau: Dict
    reference_mi: Dict[str, Dict]
    shuffled_mi: Dict
    tau_ratio_vs_best_ref: float
    h2_supported: bool
    phrase_alignment: Optional[List[Dict]]
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


MAX_LAG = 20


def _compute_mi_decay(tokens: List[str], label: str) -> MIDecayCurve:
    """Compute MI at increasing lags and fit exponential decay."""
    mi_vals = mutual_information_lag(tokens, max_lag=MAX_LAG)

    # Fit exponential decay
    lags = sorted(mi_vals.keys())
    x_vals = [float(k) for k in lags]
    y_vals = [mi_vals[k] for k in lags]

    A, tau, r_sq = fit_exponential_decay(x_vals, y_vals)

    return MIDecayCurve(
        label=label,
        mi_values=mi_vals,
        tau=tau,
        amplitude=A,
        fit_r_squared=r_sq,
    )


def _compute_section_taus(corpus) -> SectionTauComparison:
    """Compute MI decay τ for combined, herbal, and pharma sections."""
    tokens_a = corpus.get_tokens(language='A')
    tokens_herbal = corpus.get_tokens(section='herbal_a', paragraph_only=True)
    tokens_pharma = corpus.get_tokens(section='pharmaceutical', paragraph_only=True)

    mi_combined = _compute_mi_decay(tokens_a, 'combined')
    mi_herbal = _compute_mi_decay(tokens_herbal, 'herbal')
    mi_pharma = _compute_mi_decay(tokens_pharma, 'pharma')

    # Tau consistency: within 30% of each other
    tau_h = mi_herbal.tau
    tau_p = mi_pharma.tau
    if tau_h > 0 and tau_p > 0:
        ratio = max(tau_h, tau_p) / min(tau_h, tau_p)
        consistent = ratio < 1.3
    else:
        consistent = False

    return SectionTauComparison(
        combined_tau=mi_combined.tau,
        herbal_tau=tau_h,
        pharma_tau=tau_p,
        tau_consistent=consistent,
    )


def _build_phrase_embeddings(
    tokens: List[str], phrase_length: int, n_components: int = 50,
) -> Tuple[np.ndarray, List[str]]:
    """Build phrase-level embeddings by sliding a window over tokens."""
    phrases = []
    for i in range(len(tokens) - phrase_length + 1):
        phrase = '_'.join(tokens[i:i + phrase_length])
        phrases.append(phrase)

    phrase_counts = Counter(phrases)
    vocab = [p for p, _ in phrase_counts.most_common(200)]

    if len(vocab) < n_components + 5:
        return np.array([]), []

    cooc, idx = build_cooccurrence_matrix(phrases, vocab, window=5)
    if cooc.shape[0] < n_components + 1:
        return np.array([]), []

    ppmi = ppmi_matrix(cooc)
    nc = min(n_components, ppmi.shape[0] - 1)
    embeddings = truncated_svd(ppmi, n_components=nc)

    return embeddings, vocab


def _phrase_level_alignment(
    voynich_tokens: List[str],
    ref_tokens: List[str],
    phrase_lengths: List[int] = (3, 5, 7),
) -> List[PhraseAlignmentResult]:
    """Attempt phrase-level Procrustes alignment for each phrase length."""
    results = []

    # Token-level baseline
    n_components = 30
    v_counts = Counter(voynich_tokens)
    r_counts = Counter(ref_tokens)
    v_vocab = [w for w, _ in v_counts.most_common(200)]
    r_vocab = [w for w, _ in r_counts.most_common(200)]

    v_cooc, v_idx = build_cooccurrence_matrix(voynich_tokens, v_vocab, window=5)
    r_cooc, r_idx = build_cooccurrence_matrix(ref_tokens, r_vocab, window=5)

    v_ppmi = ppmi_matrix(v_cooc)
    r_ppmi = ppmi_matrix(r_cooc)

    nc = min(n_components, v_ppmi.shape[0] - 1, r_ppmi.shape[0] - 1)
    if nc < 5:
        return results

    v_emb_tok = truncated_svd(v_ppmi, n_components=nc)
    r_emb_tok = truncated_svd(r_ppmi, n_components=nc)

    # Match by frequency rank
    n_anchors = min(20, v_emb_tok.shape[0], r_emb_tok.shape[0])
    src_idx = np.arange(n_anchors)
    tgt_idx = np.arange(n_anchors)

    if n_anchors < 5:
        return results

    _, token_residual = procrustes_alignment(v_emb_tok, r_emb_tok, src_idx, tgt_idx)

    for pl in phrase_lengths:
        v_emb_phrase, v_labels = _build_phrase_embeddings(voynich_tokens, pl, n_components=nc)
        r_emb_phrase, r_labels = _build_phrase_embeddings(ref_tokens, pl, n_components=nc)

        if v_emb_phrase.size == 0 or r_emb_phrase.size == 0:
            results.append(PhraseAlignmentResult(
                phrase_length=pl,
                procrustes_residual=float('inf'),
                selectivity_vs_token=0.0,
                improvement_over_token=False,
            ))
            continue

        n_anc = min(15, v_emb_phrase.shape[0], r_emb_phrase.shape[0])
        if n_anc < 5:
            results.append(PhraseAlignmentResult(
                phrase_length=pl,
                procrustes_residual=float('inf'),
                selectivity_vs_token=0.0,
                improvement_over_token=False,
            ))
            continue

        s_idx = np.arange(n_anc)
        t_idx = np.arange(n_anc)
        _, phrase_residual = procrustes_alignment(v_emb_phrase, r_emb_phrase, s_idx, t_idx)

        sel = token_residual / phrase_residual if phrase_residual > 0 else 0.0

        results.append(PhraseAlignmentResult(
            phrase_length=pl,
            procrustes_residual=phrase_residual,
            selectivity_vs_token=sel,
            improvement_over_token=phrase_residual < token_residual,
        ))

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_mutual_info_decay() -> Dict[str, Any]:
    """Run Phase 10.2: mutual information decay analysis."""
    print("=" * 60)
    print("Phase 10.2 — Multi-Token Mutual Information Decay")
    print("=" * 60)

    # --- Load data ---
    corpus = load_corpus(verbose=False)
    ref_corpus = load_reference_corpus(verbose=False)

    # --- Voynich MI decay ---
    print("\n  Computing Voynich MI decay...")
    tokens_a = corpus.get_tokens(language='A')
    voynich_mi = _compute_mi_decay(tokens_a, 'voynich_A')
    print(f"    τ = {voynich_mi.tau:.3f}, A = {voynich_mi.amplitude:.4f}, "
          f"R² = {voynich_mi.fit_r_squared:.3f}")

    # --- Per-section τ ---
    print("\n  Computing per-section τ values...")
    section_tau = _compute_section_taus(corpus)
    print(f"    Combined τ = {section_tau.combined_tau:.3f}")
    print(f"    Herbal τ   = {section_tau.herbal_tau:.3f}")
    print(f"    Pharma τ   = {section_tau.pharma_tau:.3f}")
    print(f"    Consistent = {section_tau.tau_consistent}")

    # --- Reference MI decay ---
    print("\n  Computing reference MI decay...")
    ref_mi: Dict[str, MIDecayCurve] = {}
    for lang in ref_corpus.languages:
        ref_tokens = ref_corpus.get_combined_tokens(lang)
        if len(ref_tokens) < 100:
            continue
        ref_mi[lang] = _compute_mi_decay(ref_tokens, lang)
        print(f"    {lang}: τ = {ref_mi[lang].tau:.3f}, "
              f"A = {ref_mi[lang].amplitude:.4f}")

    # --- Shuffled baseline ---
    print("\n  Computing shuffled baseline...")
    shuffled = list(tokens_a)
    rng = random.Random(42)
    rng.shuffle(shuffled)
    shuffled_mi = _compute_mi_decay(shuffled, 'shuffled')
    print(f"    Shuffled τ = {shuffled_mi.tau:.3f}")

    # --- τ ratio ---
    ref_taus = [rc.tau for rc in ref_mi.values() if rc.tau > 0]
    best_ref_tau = max(ref_taus) if ref_taus else 1.0
    tau_ratio = voynich_mi.tau / best_ref_tau if best_ref_tau > 0 else 0.0
    print(f"\n  τ_voynich / τ_best_ref = {tau_ratio:.3f}")

    # H2 supported if τ_voynich >> τ_reference
    h2_supported = tau_ratio > 1.5

    # --- Phrase-level alignment (if H2 supported) ---
    phrase_alignment = None
    if h2_supported:
        print("\n  H2 supported — running phrase-level alignment...")
        best_ref_lang = max(ref_mi, key=lambda k: ref_mi[k].tau) if ref_mi else None
        if best_ref_lang:
            ref_tokens = ref_corpus.get_combined_tokens(best_ref_lang)
            phrase_alignment = _phrase_level_alignment(tokens_a, ref_tokens)
            for pa in phrase_alignment:
                print(f"    Phrase len {pa.phrase_length}: "
                      f"residual={pa.procrustes_residual:.4f}, "
                      f"selectivity={pa.selectivity_vs_token:.3f}, "
                      f"improvement={pa.improvement_over_token}")
    else:
        print("\n  H2 not supported by τ ratio — skipping phrase alignment")

    # --- Gate ---
    gate_passed = h2_supported or tau_ratio < 0.5  # H2 or clear H3 signal

    if h2_supported:
        verdict = (f"mi_decay_supports_H2: τ_voynich={voynich_mi.tau:.3f} >> "
                   f"τ_best_ref={best_ref_tau:.3f} (ratio={tau_ratio:.2f}), "
                   f"section_consistent={section_tau.tau_consistent}")
    elif tau_ratio < 0.5:
        verdict = (f"mi_decay_supports_H3: τ_voynich={voynich_mi.tau:.3f} << "
                   f"τ_best_ref={best_ref_tau:.3f} (rapid decorrelation)")
    else:
        verdict = (f"mi_decay_supports_H1: τ_voynich={voynich_mi.tau:.3f} ≈ "
                   f"τ_best_ref={best_ref_tau:.3f} (normal decay)")

    print(f"\n  Gate passed: {gate_passed}")
    print(f"  Verdict: {verdict}")

    # --- MI values summary ---
    print("\n  Voynich MI(d) values:")
    for lag in sorted(voynich_mi.mi_values.keys()):
        print(f"    MI(d={lag:2d}) = {voynich_mi.mi_values[lag]:.5f}")

    # --- Save ---
    result = MIDecayResult(
        voynich_mi=_convert(asdict(voynich_mi)),
        section_tau=_convert(asdict(section_tau)),
        reference_mi={k: _convert(asdict(v)) for k, v in ref_mi.items()},
        shuffled_mi=_convert(asdict(shuffled_mi)),
        tau_ratio_vs_best_ref=tau_ratio,
        h2_supported=h2_supported,
        phrase_alignment=[_convert(asdict(pa)) for pa in phrase_alignment] if phrase_alignment else None,
        gate_passed=gate_passed,
        verdict=verdict,
    )

    out = _convert(asdict(result))
    out_path = _results_dir() / 'mi_decay.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n  Results saved to {out_path}")

    return out
