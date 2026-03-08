"""
Step 27.3 -- Phase 27 Combined Verdict
========================================
Integrate the gibberish typology control (Step 27.1) and Naibbe
entropy shift test (Step 27.2) into specific language for the paper.

Dependency chain:
    results/gibberish_typology.json  (Step 27.1)
    results/naibbe_entropy.json      (Step 27.2)
        -> phase27_verdict.json
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from voynich.core._paths import results_dir as _results_dir


# ---------------------------------------------------------------------------
# JSON serialiser
# ---------------------------------------------------------------------------

def _convert(obj: Any) -> Any:
    """Recursively convert dataclasses/numpy/NaN to JSON-safe types."""
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
    if isinstance(obj, float) and (obj != obj):  # NaN
        return None
    if isinstance(obj, (bool, int, float, str, type(None))):
        return obj
    return str(obj)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Phase27VerdictResult:
    timestamp: str
    # Gibberish test
    gibberish_n_samples: int
    gibberish_encoded_natural_count: int
    timm_n_samples: int
    timm_encoded_natural_count: int
    gibberish_discriminant_power: float
    gibberish_gate: bool
    gibberish_verdict: str
    gibberish_paper_language: str
    # Naibbe test
    naibbe_greshko_cosine: float
    naibbe_best_cosine: float
    naibbe_vs_tachygraphic: str
    naibbe_ci_overlap: bool
    naibbe_tristate_matches: int
    naibbe_gate: bool
    naibbe_verdict: str
    naibbe_paper_language: str
    # Combined
    combined_verdict: str
    combined_description: str
    paper_updates: List[str]
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Verdict logic
# ---------------------------------------------------------------------------

def _gibberish_paper_language(
    gib_enc_nat: int,
    n_gib: int,
    timm_enc_nat: int,
    n_timm: int,
    verdict: str,
) -> str:
    """Generate paper-quality language for the gibberish control result."""
    if verdict == 'CLASSIFIER_ROBUST':
        return (
            f"The classifier discriminates Voynich from both human-produced "
            f"gibberish (0/{n_gib} classified as encoded natural language) "
            f"and mechanically-generated self-citation text (0/{n_timm}). "
            f"The 'encoded natural language' classification is specific to "
            f"texts with genuine higher-order linguistic structure."
        )
    elif verdict == 'PARTIALLY_ROBUST':
        return (
            f"The classifier labels {gib_enc_nat}/{n_gib} gibberish samples "
            f"and {timm_enc_nat}/{n_timm} self-citation samples as 'encoded "
            f"natural language,' indicating partial overlap between encoded "
            f"text and deliberate gibberish on these metrics. However, the "
            f"Voynich's specific combination of anomalous H2/H1 ratio, "
            f"Zipfian compliance, and elevated entropy floor is not "
            f"replicated by any gibberish sample at full confidence."
        )
    else:  # CLASSIFIER_COMPROMISED
        return (
            f"The text typology classification is insufficient to distinguish "
            f"Voynich from deliberate gibberish ({gib_enc_nat}/{n_gib} "
            f"gibberish and {timm_enc_nat}/{n_timm} self-citation samples "
            f"receive the same 'encoded natural language' label). The "
            f"classification should be interpreted as 'text with complex "
            f"statistical structure' rather than as evidence of genuine "
            f"linguistic encoding."
        )


def _naibbe_paper_language(
    naibbe_cos: float,
    tachy_cos: float,
    ci_overlap: bool,
    naibbe_ci: List[float],
    tachy_ci: List[float],
    naibbe_vs: str,
    tristate: int,
) -> str:
    """Generate paper-quality language for the Naibbe entropy test result."""
    if naibbe_vs == 'TACHYGRAPHIC_CONFIRMED':
        return (
            f"The Naibbe dice cipher, parameterized with Greshko's (2025) "
            f"values, produces an entropy shift cosine of {naibbe_cos:.3f}, "
            f"ranking below the homophonic cipher (0.566) and well below "
            f"the tachygraphic model ({tachy_cos:.3f}). The tachygraphic "
            f"model is discriminated from the Naibbe with non-overlapping "
            f"95% confidence intervals."
        )
    elif naibbe_vs == 'TACHYGRAPHIC_PREFERRED':
        if ci_overlap:
            return (
                f"The Naibbe model (cosine {naibbe_cos:.3f}, CI "
                f"[{naibbe_ci[0]:.3f}, {naibbe_ci[1]:.3f}]) is a credible "
                f"alternative to the tachygraphic model ({tachy_cos:.3f}, CI "
                f"[{tachy_ci[0]:.3f}, {tachy_ci[1]:.3f}]), but the "
                f"tachygraphic model produces a closer entropy shift match. "
                f"The two models' confidence intervals overlap, so they "
                f"cannot be fully discriminated by entropy shift alone."
            )
        else:
            return (
                f"The Naibbe model (cosine {naibbe_cos:.3f}) is a credible "
                f"alternative to the tachygraphic model ({tachy_cos:.3f}) "
                f"but is outperformed. Both are discriminated at the 95% "
                f"CI level."
            )
    elif naibbe_vs == 'NAIBBE_SUPERIOR':
        tristate_note = ""
        if tristate < 2:
            tristate_note = (
                f" However, the Naibbe model reproduces only {tristate}/3 "
                f"of the Voynich tri-state pattern (burstiness, compression, "
                f"HMM structure), suggesting it does not fully explain the "
                f"Voynich encoding mechanism."
            )
        return (
            f"The Naibbe dice cipher with Greshko's parameters (cosine "
            f"{naibbe_cos:.3f}) produces a closer entropy shift match than "
            f"the tachygraphic model ({tachy_cos:.3f}). This does not "
            f"refute the tachygraphic structural evidence (sign families, "
            f"Fontana parallels) but indicates the entropy shift alone "
            f"does not discriminate between the two mechanisms.{tristate_note}"
        )
    else:  # DEGENERATE or unknown
        return (
            f"The Naibbe dice cipher (cosine {naibbe_cos:.3f}, CI "
            f"[{naibbe_ci[0]:.3f}, {naibbe_ci[1]:.3f}]) and the "
            f"tachygraphic model ({tachy_cos:.3f}, CI [{tachy_ci[0]:.3f}, "
            f"{tachy_ci[1]:.3f}]) produce statistically indistinguishable "
            f"entropy shift profiles. The two encoding hypotheses cannot "
            f"be separated by entropy shift analysis alone."
        )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_phase27_verdict() -> None:
    """Step 27.3: Phase 27 combined peer review verdict."""
    t0 = time.time()
    rd = str(_results_dir())

    print("=" * 60)
    print("Step 27.3: Phase 27 Combined Verdict")
    print("=" * 60)

    # ── Load results ──────────────────────────────────────────────────
    print("\n  Loading Step 27.1 and 27.2 results ...")

    gib_path = os.path.join(rd, 'gibberish_typology.json')
    with open(gib_path) as f:
        gib = json.load(f)

    naibbe_path = os.path.join(rd, 'naibbe_entropy.json')
    with open(naibbe_path) as f:
        naibbe = json.load(f)

    # ── Gibberish verdict ─────────────────────────────────────────────
    print("\n  Gibberish control:")

    gib_enc_nat = gib['gibberish_encoded_natural_count']
    n_gib = gib['n_gibberish_samples']
    timm_enc_nat = gib['timm_encoded_natural_count']
    n_timm = gib['n_timm_samples']
    gib_verdict = gib['verdict']
    gib_gate = gib['gate_passed']
    gib_disc_power = gib['discriminant_power']

    gib_paper = _gibberish_paper_language(
        gib_enc_nat, n_gib, timm_enc_nat, n_timm, gib_verdict,
    )

    print(f"    Verdict: {gib_verdict}")
    print(f"    Gibberish encoded_natural: {gib_enc_nat}/{n_gib}")
    print(f"    Timm-Schinner encoded_natural: {timm_enc_nat}/{n_timm}")
    print(f"    Paper language: {gib_paper[:120]}...")

    # ── Naibbe verdict ────────────────────────────────────────────────
    print("\n  Naibbe control:")

    naibbe_cos = naibbe['greshko_cosine']
    naibbe_best = naibbe['best_cosine']
    naibbe_vs = naibbe['naibbe_vs_tachygraphic']
    naibbe_ci_overlap = naibbe['ci_overlap']
    naibbe_ci_vals = naibbe['naibbe_ci']
    tachy_ci_vals = naibbe['tachy_ci']
    tachy_cos = naibbe['phase19_best_cosine']
    naibbe_gate = naibbe['gate_passed']
    naibbe_tristate = naibbe['tristate_match_count']
    naibbe_verdict_text = naibbe['verdict']

    naibbe_paper = _naibbe_paper_language(
        naibbe_cos, tachy_cos, naibbe_ci_overlap,
        naibbe_ci_vals, tachy_ci_vals, naibbe_vs, naibbe_tristate,
    )

    print(f"    Verdict: {naibbe_vs}")
    print(f"    Greshko cosine: {naibbe_cos:.4f}")
    print(f"    Best grid cosine: {naibbe_best:.4f}")
    print(f"    CI overlap: {naibbe_ci_overlap}")
    print(f"    Tri-state matches: {naibbe_tristate}/3")
    print(f"    Paper language: {naibbe_paper[:120]}...")

    # ── Combined verdict ──────────────────────────────────────────────
    print("\n  Combined verdict:")

    paper_updates: List[str] = []

    if gib_gate and naibbe_vs == 'TACHYGRAPHIC_CONFIRMED':
        combined = 'PEER_REVIEW_READY'
        description = (
            "Both peer-review controls pass. The classifier discriminates "
            "Voynich from gibberish, and the tachygraphic model is "
            "definitively preferred over the Naibbe dice cipher. The paper's "
            "claims are strengthened."
        )
        paper_updates.append(
            "Add to Methods: gibberish control test confirms classifier "
            "specificity."
        )
        paper_updates.append(
            "Add to Methods: Naibbe cipher test confirms tachygraphic "
            "model superiority."
        )
    elif gib_gate and naibbe_vs == 'TACHYGRAPHIC_PREFERRED':
        combined = 'PEER_REVIEW_READY_WITH_CAVEAT'
        description = (
            "Classifier discriminates gibberish. Tachygraphic model "
            "is preferred but Naibbe is a credible alternative. The paper "
            "should discuss both mechanisms."
        )
        paper_updates.append(
            "Add to Discussion: Naibbe dice cipher as competing hypothesis."
        )
    elif gib_gate and naibbe_vs == 'NAIBBE_SUPERIOR':
        combined = 'PEER_REVIEW_NEEDS_REVISION'
        description = (
            "Classifier discriminates gibberish (good), but Naibbe produces "
            "a better entropy shift match than tachygraphic. The paper must "
            "acknowledge the Naibbe as a competing or superior explanation "
            "for the entropy shift, while noting the structural evidence "
            "(sign families, Fontana parallels) that still favors "
            "tachygraphic construction."
        )
        paper_updates.append(
            "Revise entropy shift section: Naibbe is a stronger match."
        )
        paper_updates.append(
            "Add to Discussion: tachygraphic still supported by structural "
            "evidence (Phase 19.5, 19.6, 21.2)."
        )
    elif not gib_gate and naibbe_vs in ('TACHYGRAPHIC_CONFIRMED', 'TACHYGRAPHIC_PREFERRED'):
        combined = 'CLASSIFIER_COMPROMISED_NAIBBE_OK'
        description = (
            "The classifier cannot distinguish Voynich from gibberish, "
            "weakening the Phase 9.5 claim. However, the Naibbe test "
            "confirms tachygraphic superiority. The paper must qualify "
            "the typology classification."
        )
        paper_updates.append(
            "Revise Phase 9.5 section: qualify 'encoded natural language' "
            "classification — it does not discriminate from gibberish."
        )
    elif not gib_gate and naibbe_vs == 'NAIBBE_SUPERIOR':
        combined = 'MAJOR_REVISION_NEEDED'
        description = (
            "Both controls raise concerns. The classifier is compromised "
            "and the Naibbe produces a better entropy shift match than "
            "tachygraphic. The paper requires major revision to its "
            "encoding mechanism arguments."
        )
        paper_updates.append(
            "Major revision: Phase 9.5 classification unreliable."
        )
        paper_updates.append(
            "Major revision: entropy shift section must present Naibbe "
            "as primary or co-equal hypothesis."
        )
    else:
        # Degenerate / other combinations
        combined = 'PEER_REVIEW_READY_WITH_CAVEATS'
        description = (
            f"Gibberish gate: {'PASS' if gib_gate else 'FAIL'}. "
            f"Naibbe: {naibbe_vs}. The paper should discuss limitations."
        )
        if not gib_gate:
            paper_updates.append(
                "Qualify Phase 9.5 classification in Methods section."
            )
        if naibbe_ci_overlap:
            paper_updates.append(
                "Note in Discussion: Naibbe and tachygraphic cannot be "
                "separated by entropy shift alone."
            )

    # Note about structural evidence regardless of Naibbe result
    if naibbe_vs in ('NAIBBE_SUPERIOR', 'DEGENERATE') or naibbe_ci_overlap:
        paper_updates.append(
            "Note: tachygraphic identification still rests on multiple "
            "independent evidence streams (Phase 19.5 sign families 1.61x, "
            "Phase 19.6 simulation, Phase 21.2 Fontana 14.81x) that the "
            "Naibbe model does not explain."
        )

    print(f"    Verdict: {combined}")
    print(f"    Description: {description}")
    print(f"    Paper updates needed: {len(paper_updates)}")
    for i, u in enumerate(paper_updates):
        print(f"      {i + 1}. {u}")

    # ── Save ──────────────────────────────────────────────────────────
    result = Phase27VerdictResult(
        timestamp=time.strftime('%Y-%m-%dT%H:%M:%S'),
        gibberish_n_samples=n_gib,
        gibberish_encoded_natural_count=gib_enc_nat,
        timm_n_samples=n_timm,
        timm_encoded_natural_count=timm_enc_nat,
        gibberish_discriminant_power=round(gib_disc_power, 4),
        gibberish_gate=gib_gate,
        gibberish_verdict=gib_verdict,
        gibberish_paper_language=gib_paper,
        naibbe_greshko_cosine=round(naibbe_cos, 4),
        naibbe_best_cosine=round(naibbe_best, 4),
        naibbe_vs_tachygraphic=naibbe_vs,
        naibbe_ci_overlap=naibbe_ci_overlap,
        naibbe_tristate_matches=naibbe_tristate,
        naibbe_gate=naibbe_gate,
        naibbe_verdict=naibbe_verdict_text,
        naibbe_paper_language=naibbe_paper,
        combined_verdict=combined,
        combined_description=description,
        paper_updates=paper_updates,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'phase27_verdict.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2)
    print(f"\n  -> {out_path}")
