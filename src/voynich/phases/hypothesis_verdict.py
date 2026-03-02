"""
Phase 10.5 — Hypothesis Integration and Verdict
==================================================

Rationale
---------
Compile evidence from all Phase 10 sub-analyses (10.1–10.4) into a unified
scoring of the three surviving hypotheses.  Determine which hypothesis best
explains the data, and identify the actionable next step.

Sub-analyses
------------
10.5a  Score each hypothesis from all evidence
10.5b  Decision matrix
10.5c  Actionable outcomes
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from voynich.core._paths import results_dir as _results_dir


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class EvidenceItem:
    test: str           # which phase (10.1, 10.2, 10.3, 10.4)
    hypothesis: str     # H1, H2, or H3
    metric: str         # short name of the metric
    value: float
    supports: bool
    weight: float       # 1.0 for direct test, 0.5 for indirect


@dataclass
class HypothesisVerdict:
    h1_evidence: List[Dict]
    h2_evidence: List[Dict]
    h3_evidence: List[Dict]
    h1_score: float
    h2_score: float
    h3_score: float
    winning_hypothesis: str
    actionable_next_step: str
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


def _load_results(name: str) -> Optional[Dict]:
    """Load a Phase 10 results JSON."""
    path = _results_dir() / f'{name}.json'
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def _compile_evidence() -> List[EvidenceItem]:
    """Extract evidence items from all Phase 10 results."""
    evidence = []

    # --- 10.1: Entropy curves ---
    ec = _load_results('entropy_curves')
    if ec:
        hs = ec.get('hypothesis_scores', {})

        # H1: parallel shift correlation
        h1_corr = float(hs.get('h1_correlation', 0))
        evidence.append(EvidenceItem(
            test='10.1', hypothesis='H1',
            metric='entropy_curve_correlation',
            value=h1_corr, supports=h1_corr > 0.8, weight=1.0,
        ))

        # H1: section consistency
        h1_sec = bool(hs.get('h1_section_consistent', False))
        evidence.append(EvidenceItem(
            test='10.1', hypothesis='H1',
            metric='section_consistency',
            value=1.0 if h1_sec else 0.0, supports=h1_sec, weight=0.5,
        ))

        # H2: back-loaded reduction
        h2_bl = float(hs.get('h2_backload_ratio', 0))
        evidence.append(EvidenceItem(
            test='10.1', hypothesis='H2',
            metric='backload_ratio',
            value=h2_bl, supports=h2_bl > 1.5, weight=1.0,
        ))

        # H3: floor ratio
        h3_fr = float(hs.get('h3_floor_ratio', 0))
        evidence.append(EvidenceItem(
            test='10.1', hypothesis='H3',
            metric='entropy_floor_ratio',
            value=h3_fr, supports=h3_fr > 1.3, weight=1.0,
        ))

        # H3: section divergence
        h3_div = bool(hs.get('h3_section_divergent', False))
        evidence.append(EvidenceItem(
            test='10.1', hypothesis='H3',
            metric='section_divergence',
            value=1.0 if h3_div else 0.0, supports=h3_div, weight=0.5,
        ))

    # --- 10.2: MI decay ---
    mid = _load_results('mi_decay')
    if mid:
        tau_ratio = float(mid.get('tau_ratio_vs_best_ref', 0))
        h2_supp = bool(mid.get('h2_supported', False))

        evidence.append(EvidenceItem(
            test='10.2', hypothesis='H2',
            metric='tau_ratio',
            value=tau_ratio, supports=h2_supp, weight=1.0,
        ))

        # H3: if tau is very short
        evidence.append(EvidenceItem(
            test='10.2', hypothesis='H3',
            metric='tau_ratio_inverse',
            value=tau_ratio, supports=tau_ratio < 0.5, weight=0.5,
        ))

        # H2: section tau consistency
        st = mid.get('section_tau', {})
        tau_con = bool(st.get('tau_consistent', False))
        evidence.append(EvidenceItem(
            test='10.2', hypothesis='H2',
            metric='section_tau_consistency',
            value=1.0 if tau_con else 0.0, supports=tau_con and h2_supp, weight=0.5,
        ))

        # Phrase alignment
        pa = mid.get('phrase_alignment')
        if pa:
            any_improvement = any(p.get('improvement_over_token', False) for p in pa)
            evidence.append(EvidenceItem(
                test='10.2', hypothesis='H2',
                metric='phrase_alignment_improvement',
                value=1.0 if any_improvement else 0.0,
                supports=any_improvement, weight=1.0,
            ))

    # --- 10.3: Folio shifts ---
    fs = _load_results('folio_shift')
    if fs:
        h3_supp = bool(fs.get('h3_supported', False))

        jsd = fs.get('jsd_analysis', {})
        residual = bool(jsd.get('residual_significant', False))
        evidence.append(EvidenceItem(
            test='10.3', hypothesis='H3',
            metric='residual_jsd',
            value=1.0 if residual else 0.0, supports=residual, weight=1.0,
        ))

        cv = fs.get('function_word_cv', {})
        cv_inflated = bool(cv.get('cv_inflated', False))
        evidence.append(EvidenceItem(
            test='10.3', hypothesis='H3',
            metric='function_word_cv_inflated',
            value=1.0 if cv_inflated else 0.0, supports=cv_inflated, weight=1.0,
        ))

        qb = fs.get('quire_boundary', {})
        quire_eff = bool(qb.get('quire_effect', False))
        evidence.append(EvidenceItem(
            test='10.3', hypothesis='H3',
            metric='quire_boundary_effect',
            value=1.0 if quire_eff else 0.0, supports=quire_eff, weight=0.5,
        ))

        # H1/H2: if no folio shifts, supports non-H3 hypotheses
        if not h3_supp:
            evidence.append(EvidenceItem(
                test='10.3', hypothesis='H1',
                metric='no_folio_shifts',
                value=1.0, supports=True, weight=0.5,
            ))
            evidence.append(EvidenceItem(
                test='10.3', hypothesis='H2',
                metric='no_folio_shifts',
                value=1.0, supports=True, weight=0.5,
            ))

    # --- 10.4: Glyph grammar ---
    gg = _load_results('glyph_grammar')
    if gg:
        h1_supp = bool(gg.get('h1_supported', False))

        # Script grid similarity
        comps = gg.get('grid_comparisons', [])
        if comps:
            best_sim = float(comps[0].get('similarity_to_voynich', 0))
            evidence.append(EvidenceItem(
                test='10.4', hypothesis='H1',
                metric='script_grid_similarity',
                value=best_sim, supports=best_sim > 0.3, weight=1.0,
            ))

        # Construction vs morphology
        ct = gg.get('construction_test', {})
        diagnosis = ct.get('diagnosis', '')
        evidence.append(EvidenceItem(
            test='10.4', hypothesis='H1',
            metric='construction_diagnosis',
            value=1.0 if diagnosis == 'construction' else 0.0,
            supports=diagnosis == 'construction', weight=1.0,
        ))

        # CSP viability
        csp = gg.get('csp_result', {})
        if csp:
            csp_viable = bool(csp.get('decoding_viable', False))
            evidence.append(EvidenceItem(
                test='10.4', hypothesis='H1',
                metric='csp_decoding_viable',
                value=1.0 if csp_viable else 0.0,
                supports=csp_viable, weight=1.0,
            ))

            # Language B consistency
            lb = csp.get('lang_b_consistency', {})
            if lb:
                b_subset = bool(lb.get('lang_b_uses_subset', False))
                evidence.append(EvidenceItem(
                    test='10.4', hypothesis='H1',
                    metric='lang_b_subset',
                    value=1.0 if b_subset else 0.0,
                    supports=b_subset, weight=0.5,
                ))

    return evidence


def _score_hypotheses(evidence: List[EvidenceItem]) -> Dict[str, float]:
    """Weighted score for each hypothesis."""
    scores = {'H1': 0.0, 'H2': 0.0, 'H3': 0.0}
    for e in evidence:
        if e.supports:
            scores[e.hypothesis] += e.weight
    return scores


def _determine_verdict(scores: Dict[str, float]) -> Tuple[str, str]:
    """Determine winning hypothesis and actionable next step."""
    winner = max(scores, key=scores.get)
    gap = scores[winner] - sorted(scores.values())[-2]

    if winner == 'H1':
        next_step = (
            "Constructed script confirmed. The 14-variable CSP is the decoding "
            "path. Each grid cell = one phoneme or syllable. Phonotactic "
            "constraints of Romance languages prune the search space. "
            "Illustration constraints provide anchor values. Estimate: "
            "constraint propagation reduces to ~10^3-10^6 candidates."
        )
    elif winner == 'H2':
        next_step = (
            "Information dispersion confirmed. Phrase-level analysis is the "
            "correct granularity. Rebuild Phase 5-8 approaches at phrase level: "
            "phrase embeddings, phrase-level frequency matching, phrase-level "
            "MDL decoding. The encoding unit is a multi-token phrase."
        )
    elif winner == 'H3':
        next_step = (
            "Keyed cipher confirmed. Focus shifts to key recovery. Search the "
            "manuscript's physical structure (quire marks, page numbers, "
            "decorative elements) for key material. If key period was detected, "
            "treat each period as a separate simple cipher."
        )
    else:
        next_step = "Ambiguous result. No single hypothesis clearly wins."

    return winner, next_step


from typing import Tuple


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_hypothesis_verdict() -> Dict[str, Any]:
    """Run Phase 10.5: hypothesis integration and verdict."""
    print("=" * 60)
    print("Phase 10.5 — Hypothesis Integration and Verdict")
    print("=" * 60)

    # --- Compile evidence ---
    print("\n  Compiling evidence from Phase 10.1–10.4...")
    evidence = _compile_evidence()

    h1_evidence = [e for e in evidence if e.hypothesis == 'H1']
    h2_evidence = [e for e in evidence if e.hypothesis == 'H2']
    h3_evidence = [e for e in evidence if e.hypothesis == 'H3']

    print(f"\n  H1 evidence items: {len(h1_evidence)}")
    for e in h1_evidence:
        mark = "+" if e.supports else "-"
        print(f"    [{mark}] {e.test} {e.metric}: {e.value:.3f} (w={e.weight})")

    print(f"\n  H2 evidence items: {len(h2_evidence)}")
    for e in h2_evidence:
        mark = "+" if e.supports else "-"
        print(f"    [{mark}] {e.test} {e.metric}: {e.value:.3f} (w={e.weight})")

    print(f"\n  H3 evidence items: {len(h3_evidence)}")
    for e in h3_evidence:
        mark = "+" if e.supports else "-"
        print(f"    [{mark}] {e.test} {e.metric}: {e.value:.3f} (w={e.weight})")

    # --- Score ---
    scores = _score_hypotheses(evidence)
    print(f"\n  Hypothesis scores:")
    print(f"    H1 (Constructed script):     {scores['H1']:.1f}")
    print(f"    H2 (Information dispersion): {scores['H2']:.1f}")
    print(f"    H3 (Keyed cipher):           {scores['H3']:.1f}")

    winner, next_step = _determine_verdict(scores)

    # Gate: clear winner (margin > 1.0)
    sorted_scores = sorted(scores.values(), reverse=True)
    margin = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else 0
    gate_passed = margin > 1.0

    if gate_passed:
        verdict = f"hypothesis_{winner}_wins: score={scores[winner]:.1f}, margin={margin:.1f}"
    else:
        verdict = (f"hypothesis_ambiguous: H1={scores['H1']:.1f}, "
                   f"H2={scores['H2']:.1f}, H3={scores['H3']:.1f}, margin={margin:.1f}")

    print(f"\n  Winner: {winner}")
    print(f"  Margin: {margin:.1f}")
    print(f"  Gate passed: {gate_passed}")
    print(f"  Verdict: {verdict}")
    print(f"\n  Actionable next step:")
    print(f"    {next_step}")

    # --- Save ---
    result = HypothesisVerdict(
        h1_evidence=[_convert(asdict(e)) for e in h1_evidence],
        h2_evidence=[_convert(asdict(e)) for e in h2_evidence],
        h3_evidence=[_convert(asdict(e)) for e in h3_evidence],
        h1_score=scores['H1'],
        h2_score=scores['H2'],
        h3_score=scores['H3'],
        winning_hypothesis=winner,
        actionable_next_step=next_step,
        gate_passed=gate_passed,
        verdict=verdict,
    )

    out = _convert(asdict(result))
    out_path = _results_dir() / 'hypothesis_verdict.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n  Results saved to {out_path}")

    return out
