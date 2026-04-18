"""
Phase 88 integration — three-diagnostic verdict on Greshko's generalized
Naibbe cipher.

Consumes results/p88_naibbe_generalized.json and emits
results/p88_integrate.json with per-diagnostic verdicts (entropy, MI,
freq-connectivity) and an overall NAIBBE_MATCHES_VOYNICH_ON_N_OF_3 score.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

from voynich.core._paths import results_dir as _results_dir


# Thresholds (derived from paper Sections 4.2, 4.4, 5.1)
ENTROPY_CONFIRM_THRESHOLD = 0.820      # Tachygraphic (paper Section 4.2)
ENTROPY_COMPETITIVE_THRESHOLD = 0.566  # Homophonic (paper Table 19.2)
MI_TACHY = 1.284                        # Tachygraphic (paper Section 4.4)
MI_NULL = 1.1                           # Weak positive anomaly
FREQ_CONN_CONFIRM = 0.5
FREQ_CONN_PARTIAL = 0.3
VOYNICH_MI_FULL = 1.450                 # Paper Section 4.4 observed


@dataclass
class DiagnosticVerdict:
    diagnostic: str
    voynich_full: float
    voynich_b: float
    tachygraphic_reference: float
    simplified_naibbe_reference: float
    generalized_naibbe_broad_mean: float
    generalized_naibbe_low_h1_mean: float
    real_naibbe_divcom: float
    real_naibbe_nathist: float
    verdict: str
    rationale: str


@dataclass
class IntegrateResult:
    timestamp: str
    source: str
    entropy: DiagnosticVerdict
    cross_boundary_mi: DiagnosticVerdict
    freq_connectivity: DiagnosticVerdict
    n_of_3_confirms: int
    overall_verdict: str
    paper_recommendations: List[str]


def _classify_entropy(low_h1_mean: float) -> str:
    if low_h1_mean > ENTROPY_CONFIRM_THRESHOLD:
        return 'ENTROPY_CONFIRMS_GRESHKO'
    if low_h1_mean > ENTROPY_COMPETITIVE_THRESHOLD:
        return 'ENTROPY_COMPETITIVE'
    return 'ENTROPY_INSUFFICIENT'


def _classify_mi(low_h1_mean: float) -> str:
    if low_h1_mean >= MI_TACHY:
        return 'MI_CONFIRMS_GRESHKO'
    if low_h1_mean >= MI_NULL:
        return 'MI_PARTIAL'
    return 'MI_INSUFFICIENT'


def _classify_freq_conn(low_h1_mean: float) -> str:
    if low_h1_mean >= FREQ_CONN_CONFIRM:
        return 'FREQ_CONN_CONFIRMS_GRESHKO'
    if low_h1_mean >= FREQ_CONN_PARTIAL:
        return 'FREQ_CONN_PARTIAL'
    return 'FREQ_CONN_INSUFFICIENT'


def _paper_recommendations(
    entropy_v: str, mi_v: str, fc_v: str,
    low_h1_cos_full: float, low_h1_cos_b: float,
    low_h1_mi: float, low_h1_fc: float,
    real_naibbe_mi: float, real_naibbe_fc: float,
) -> List[str]:
    recs: List[str] = []

    if entropy_v == 'ENTROPY_CONFIRMS_GRESHKO':
        recs.append(
            "Section 4.2 ('No configuration produces a positive cosine'): "
            "MUST BE REVISED. Greshko's generalized Naibbe produces low-H1 "
            f"mean cosine = {low_h1_cos_full:+.4f} (full) / "
            f"{low_h1_cos_b:+.4f} (Voynich B) — exceeds tachygraphic 0.820."
        )
    elif entropy_v == 'ENTROPY_COMPETITIVE':
        recs.append(
            "Section 4.2 ('No configuration produces a positive cosine'): "
            f"MUST BE SOFTENED. Low-H1 Naibbe mean cosine = {low_h1_cos_full:+.4f} "
            f"(full) / {low_h1_cos_b:+.4f} (B) is positive and exceeds the "
            "homophonic alternative, though below tachygraphic 0.820."
        )
    else:
        recs.append(
            "Section 4.2 paper claim holds: generalized Naibbe low-H1 cosine "
            f"= {low_h1_cos_full:+.4f} (full) / {low_h1_cos_b:+.4f} (B) remains "
            "below the next-best cipher (homophonic 0.566)."
        )

    if mi_v == 'MI_CONFIRMS_GRESHKO':
        recs.append(
            f"Section 4.4 / Table 2 ('Freq-connectivity: Explained'): "
            f"Low-H1 Naibbe cross-boundary ratio = {low_h1_mi:.4f} "
            f"meets or exceeds tachygraphic 1.284×. The MI anomaly is NOT "
            "unique to the tachygraphic model."
        )
    elif mi_v == 'MI_PARTIAL':
        recs.append(
            f"Section 4.4: Low-H1 Naibbe MI ratio = {low_h1_mi:.4f} is "
            f"directionally anomalous (> 1.1×) but below tachygraphic 1.284×. "
            "Tachygraphic still best-fit, margin narrower than paper implies."
        )
    else:
        recs.append(
            f"Section 4.4 paper claim holds: Naibbe MI ratio {low_h1_mi:.4f} "
            "is at or near the 1.0 null, confirming tachygraphic remains the "
            "unique explanation for the 1.450× anomaly."
        )
    recs.append(
        f"  (Real Naibbe ciphertext MI: {real_naibbe_mi:.4f} — "
        "independent, zero-parameter-search data point.)"
    )

    if fc_v == 'FREQ_CONN_CONFIRMS_GRESHKO':
        recs.append(
            f"Section 5.1 ('Freq-connectivity correlation ρ = 0.618'): "
            f"Low-H1 Naibbe ρ = {low_h1_fc:.4f} also generates the correlation. "
            "Table 2 'Freq-connectivity: Explained' no longer uniquely "
            "supports the tachygraphic model."
        )
    elif fc_v == 'FREQ_CONN_PARTIAL':
        recs.append(
            f"Section 5.1: Low-H1 Naibbe ρ = {low_h1_fc:.4f} is positive but "
            "weaker than Voynich's 0.618. Tachygraphic still best-fit here."
        )
    else:
        recs.append(
            f"Section 5.1 paper claim holds: Naibbe ρ = {low_h1_fc:.4f} is "
            "near zero, confirming the correlation is specific to the "
            "tachygraphic-style encoding."
        )
    recs.append(
        f"  (Real Naibbe ciphertext freq-conn ρ: {real_naibbe_fc:.4f} — "
        "independent data point.)"
    )

    return recs


def _overall_verdict(n_confirms: int) -> str:
    if n_confirms == 0:
        return ("NAIBBE_MATCHES_VOYNICH_ON_0_OF_3: paper's argument against "
                "Naibbe stands. Tachygraphic remains the unique best fit.")
    if n_confirms == 1:
        return ("NAIBBE_MATCHES_VOYNICH_ON_1_OF_3: one diagnostic weakened; "
                "the other two still distinguish tachygraphic from Naibbe.")
    if n_confirms == 2:
        return ("NAIBBE_MATCHES_VOYNICH_ON_2_OF_3: two diagnostics weakened. "
                "The paper's uniqueness claim needs substantial revision; "
                "only one of the three known-property checks remains "
                "tachygraphic-specific.")
    return ("NAIBBE_MATCHES_VOYNICH_ON_3_OF_3: all three diagnostics match. "
            "Paper's tachygraphic-is-unique-best-fit claim fails — a "
            "verbose-substitution cipher family is indistinguishable from "
            "tachygraphy under the current discriminators. A new "
            "discriminator (e.g. adjacent-token / boundary analysis as "
            "Greshko suggests) is required.")


def run_integrate() -> None:
    t0 = time.time()
    rd = str(_results_dir())

    print("=" * 60)
    print("Phase 88 Integration: Generalized Naibbe Verdict")
    print("=" * 60)

    source_path = os.path.join(rd, 'p88_naibbe_generalized.json')
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Missing {source_path} — run phase 88 first")
    with open(source_path) as f:
        d = json.load(f)

    # Entropy
    ent_v = _classify_entropy(d['low_h1_cos_shift_full_mean'])
    entropy = DiagnosticVerdict(
        diagnostic='entropy_shift_cosine',
        voynich_full=1.0,
        voynich_b=1.0,
        tachygraphic_reference=d['phase19_tachy_cosine_full'],
        simplified_naibbe_reference=d['phase27_simplified_cosine_full'],
        generalized_naibbe_broad_mean=d['broad_cos_shift_full_mean'],
        generalized_naibbe_low_h1_mean=d['low_h1_cos_shift_full_mean'],
        real_naibbe_divcom=d['real_naibbe_divcom']['cos_shift_full'],
        real_naibbe_nathist=d['real_naibbe_nathist']['cos_shift_full'],
        verdict=ent_v,
        rationale=(
            f"Low-H1 generalized-Naibbe mean cosine (full corpus ref) = "
            f"{d['low_h1_cos_shift_full_mean']:+.4f}; "
            f"tachygraphic = {d['phase19_tachy_cosine_full']:+.4f}; "
            f"simplified Naibbe = {d['phase27_simplified_cosine_full']:+.4f}."
        ),
    )

    # Cross-boundary MI
    mi_v = _classify_mi(d['low_h1_cross_boundary_ratio_mean'])
    mi = DiagnosticVerdict(
        diagnostic='cross_boundary_mi_ratio',
        voynich_full=d['voynich_full_cross_boundary_ratio'],
        voynich_b=d['voynich_b_cross_boundary_ratio'],
        tachygraphic_reference=MI_TACHY,
        simplified_naibbe_reference=1.0,
        generalized_naibbe_broad_mean=d['low_h1_cross_boundary_ratio_mean'],
        generalized_naibbe_low_h1_mean=d['low_h1_cross_boundary_ratio_mean'],
        real_naibbe_divcom=d['real_naibbe_divcom']['cross_boundary_ratio'],
        real_naibbe_nathist=d['real_naibbe_nathist']['cross_boundary_ratio'],
        verdict=mi_v,
        rationale=(
            f"Low-H1 Naibbe MI ratio = "
            f"{d['low_h1_cross_boundary_ratio_mean']:.4f}; "
            f"Voynich full = {d['voynich_full_cross_boundary_ratio']:.4f} "
            f"(paper reports 1.450×); tachygraphic ≈ {MI_TACHY}×."
        ),
    )

    # Frequency-connectivity
    fc_v = _classify_freq_conn(d['low_h1_freq_conn_rho_mean'])
    fc = DiagnosticVerdict(
        diagnostic='freq_connectivity_rho',
        voynich_full=d['voynich_full_freq_conn_rho'],
        voynich_b=d['voynich_b_freq_conn_rho'],
        tachygraphic_reference=0.618,
        simplified_naibbe_reference=0.0,
        generalized_naibbe_broad_mean=d['low_h1_freq_conn_rho_mean'],
        generalized_naibbe_low_h1_mean=d['low_h1_freq_conn_rho_mean'],
        real_naibbe_divcom=d['real_naibbe_divcom']['freq_conn_rho'],
        real_naibbe_nathist=d['real_naibbe_nathist']['freq_conn_rho'],
        verdict=fc_v,
        rationale=(
            f"Low-H1 Naibbe Spearman ρ = {d['low_h1_freq_conn_rho_mean']:+.4f}; "
            f"Voynich full = {d['voynich_full_freq_conn_rho']:+.4f}; "
            "Timm & Schinner 2020 report 0.618."
        ),
    )

    n_confirms = sum(
        'CONFIRMS' in v for v in (ent_v, mi_v, fc_v)
    )
    overall = _overall_verdict(n_confirms)

    recs = _paper_recommendations(
        ent_v, mi_v, fc_v,
        d['low_h1_cos_shift_full_mean'], d['low_h1_cos_shift_b_mean'],
        d['low_h1_cross_boundary_ratio_mean'], d['low_h1_freq_conn_rho_mean'],
        d['real_naibbe_divcom']['cross_boundary_ratio'],
        d['real_naibbe_divcom']['freq_conn_rho'],
    )

    result = IntegrateResult(
        timestamp=time.strftime('%Y-%m-%dT%H:%M:%S'),
        source='results/p88_naibbe_generalized.json',
        entropy=entropy,
        cross_boundary_mi=mi,
        freq_connectivity=fc,
        n_of_3_confirms=n_confirms,
        overall_verdict=overall,
        paper_recommendations=recs,
    )

    out_path = os.path.join(rd, 'p88_integrate.json')
    with open(out_path, 'w') as f:
        json.dump(asdict(result), f, indent=2)

    # Print summary
    print(f"\nEntropy shift cosine (Section 4.2):")
    print(f"  Voynich vs. self   : 1.000")
    print(f"  Tachygraphic       : {entropy.tachygraphic_reference:+.4f}")
    print(f"  Simplified Naibbe  : {entropy.simplified_naibbe_reference:+.4f}")
    print(f"  Gen. Naibbe broad  : {entropy.generalized_naibbe_broad_mean:+.4f}")
    print(f"  Gen. Naibbe low-H1 : {entropy.generalized_naibbe_low_h1_mean:+.4f}")
    print(f"  Real Naibbe (DivC) : {entropy.real_naibbe_divcom:+.4f}")
    print(f"  Real Naibbe (NatH) : {entropy.real_naibbe_nathist:+.4f}")
    print(f"  VERDICT: {ent_v}")

    print(f"\nCross-boundary MI ratio (Section 4.4):")
    print(f"  Voynich (full)     : {mi.voynich_full:.4f} (paper: 1.450×)")
    print(f"  Voynich (B)        : {mi.voynich_b:.4f}")
    print(f"  Tachygraphic ref   : {mi.tachygraphic_reference:.4f}")
    print(f"  Gen. Naibbe low-H1 : {mi.generalized_naibbe_low_h1_mean:.4f}")
    print(f"  Real Naibbe (DivC) : {mi.real_naibbe_divcom:.4f}")
    print(f"  Real Naibbe (NatH) : {mi.real_naibbe_nathist:.4f}")
    print(f"  VERDICT: {mi_v}")

    print(f"\nFrequency-connectivity ρ (Section 5.1):")
    print(f"  Voynich (full)     : {fc.voynich_full:+.4f} (paper: 0.618)")
    print(f"  Voynich (B)        : {fc.voynich_b:+.4f}")
    print(f"  Gen. Naibbe low-H1 : {fc.generalized_naibbe_low_h1_mean:+.4f}")
    print(f"  Real Naibbe (DivC) : {fc.real_naibbe_divcom:+.4f}")
    print(f"  Real Naibbe (NatH) : {fc.real_naibbe_nathist:+.4f}")
    print(f"  VERDICT: {fc_v}")

    print(f"\nOVERALL: {n_confirms}/3 diagnostics CONFIRM Greshko")
    print(f"  {overall}")

    print(f"\nPaper recommendations:")
    for r in recs:
        print(f"  - {r}")

    print(f"\n  -> {out_path}")
    print(f"  Runtime: {time.time() - t0:.1f}s")
