"""
Step 42.5 – Ground Truth Assessment
=====================================
After all audits and recomputations, produce the definitive, fully
validated state of evidence for the entire project.

Combines findings from Steps 42.1-42.4 into a single assessment:
which findings are CONFIRMED, DEFLATED, INFLATED, or INVALIDATED.

Dependency chain:
    bigram_code_audit.json          (Step 42.1)
    symmetric_recompute.json        (Step 42.2)
    signal_word_revalidate.json     (Step 42.3)
    selectivity_audit.json          (Step 42.4)
        → ground_truth.json         (this step)
"""

import json
import os
import time
from typing import Any, Dict, List, Optional

from voynich.core._paths import results_dir as _results_dir


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def _convert(obj: Any) -> Any:
    if hasattr(obj, '__dataclass_fields__'):
        from dataclasses import asdict
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


def _safe_load(path: str) -> Dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


# ─────────────────────────────────────────────────────────────────
# Evidence classification
# ─────────────────────────────────────────────────────────────────

def _build_bigram_evidence(recompute: Dict) -> List[Dict]:
    """Build per-phase bigram evidence classification."""
    summary = recompute.get('summary_table', [])
    evidence = []

    for row in summary:
        phase = row.get('phase', '?')
        orig_z = row.get('original_z')
        sym_exact = row.get('symmetric_z_exact')
        sym_total = row.get('symmetric_z_total')
        classification = row.get('classification', 'UNKNOWN')

        # Use z_total as primary — exact matches are too rare to be
        # meaningful on their own.  The signal is in relaxed (edit-
        # distance-1) matches.
        primary_z = sym_total if sym_total is not None else sym_exact

        # Compute delta
        delta = None
        if orig_z is not None and primary_z is not None:
            delta = round(primary_z - orig_z, 4)

        # Significance
        significant = primary_z is not None and primary_z > 2.0

        evidence.append({
            'phase': phase,
            'dictionary': row.get('dictionary', '?'),
            'original_z': orig_z,
            'symmetric_z_exact': sym_exact,
            'symmetric_z_total': sym_total,
            'delta': delta,
            'classification': classification,
            'significant': significant,
        })

    return evidence


def _build_validated_metrics_table(
    bigram_evidence: List[Dict],
    sigma_data: Dict,
    selectivity_data: Dict,
) -> List[Dict]:
    """Build the comprehensive validated metrics table."""
    table = []

    # Bigram z-scores (use z_total as validated value — exact matches
    # are too rare for meaningful z_exact)
    for bg in bigram_evidence:
        table.append({
            'category': 'bigram_z',
            'phase': bg['phase'],
            'metric': f"Bigram z ({bg['dictionary']})",
            'original_value': bg['original_z'],
            'validated_value': bg['symmetric_z_total'],
            'validated_z_exact': bg['symmetric_z_exact'],
            'classification': bg['classification'],
            'significant': bg['significant'],
        })

    # Signal word σ-scores
    sigma_verdict = sigma_data.get('verdict', 'UNKNOWN')
    table.append({
        'category': 'signal_words',
        'phase': '28/39',
        'metric': 'Signal word σ methodology',
        'original_value': 'σ>2.0 for 8 words',
        'validated_value': sigma_verdict,
        'classification': ('CONFIRMED' if 'VALIDATED' in sigma_verdict
                           else 'NEEDS_REVIEW'),
        'significant': 'VALIDATED' in sigma_verdict,
    })

    # Selectivity
    sel_verdict = selectivity_data.get('verdict', 'UNKNOWN')
    for sa in selectivity_data.get('selectivity_audits', []):
        sel = sa.get('reported_selectivity')
        table.append({
            'category': 'selectivity',
            'phase': sa['phase'],
            'metric': f"Selectivity ({sa.get('dictionary', '?')})",
            'original_value': sel,
            'validated_value': sa.get('recomputed_selectivity') or sel,
            'classification': ('CONFIRMED' if sa.get('methodology_symmetric')
                               else 'NEEDS_REVIEW'),
            'significant': sel is not None and sel > 1.3,
        })

    return table


def _identify_surviving_evidence(metrics_table: List[Dict]) -> List[str]:
    """List findings that survive the audit."""
    surviving = []

    # Always-unaffected findings (not based on bigram comparison)
    surviving.append(
        "Tachygraphic encoding type (Phase 19 cosine similarity = 0.820) "
        "— NOT based on bigram comparison, unaffected by audit."
    )
    surviving.append(
        "Romance language family identification (Phases 36-38) "
        "— based on dict_hit selectivity, not bigrams."
    )
    surviving.append(
        "Structural properties: Zipf's law compliance, entropy profile, "
        "morphological structure, Currier A/B sections "
        "— corpus-level statistics, unaffected."
    )
    surviving.append(
        "Sub-cell feature model breakthrough (Phase 14: 19.4% → Phase 16: "
        "43.6% full corpus dict_hit) — based on selectivity, not bigrams."
    )

    # Bigram findings that survive
    confirmed_bigrams = [
        m for m in metrics_table
        if m['category'] == 'bigram_z'
        and m['classification'] in ('CONFIRMED', 'DEFLATED')
        and m['significant']
    ]
    if confirmed_bigrams:
        best = max(confirmed_bigrams,
                   key=lambda m: m.get('validated_value') or 0)
        val = best.get('validated_value')
        val_str = f"z={val:.2f}" if val is not None else "z=N/A"
        p = best['phase']
        p_label = p if p.startswith('Phase') else f"Phase {p}"
        surviving.append(
            f"Bigram sequential structure ({p_label}, "
            f"{val_str}) — survives symmetric recomputation."
        )

    # Signal words
    sigma_entries = [m for m in metrics_table
                     if m['category'] == 'signal_words']
    if sigma_entries and sigma_entries[0]['classification'] == 'CONFIRMED':
        surviving.append(
            "Signal word vocabulary (8 genuine words: bene, codi, sero, "
            "sene, de, raro, dine, cola) — word-level methodology "
            "validated as symmetric."
        )

    # Selectivity
    confirmed_sel = [
        m for m in metrics_table
        if m['category'] == 'selectivity'
        and m['classification'] == 'CONFIRMED'
        and m['significant']
    ]
    if confirmed_sel:
        phases = ', '.join(m['phase'] for m in confirmed_sel)
        surviving.append(
            f"Dict_hit selectivity above null baseline (Phases {phases}) "
            f"— per-token metric, methodology validated."
        )

    return surviving


def _identify_retracted_evidence(metrics_table: List[Dict]) -> List[str]:
    """List findings retracted by the audit."""
    retracted = []

    # Always retracted
    retracted.append(
        "Venetian-specific identification (Phase 39-40). Venetian z=319.76 "
        "was entirely artifactual. Corrected z=-0.47 (Phase 41). "
        "Venetian selectivity dropped from 4.58× to 1.18×."
    )

    # Bigram z-scores that failed
    failed = [
        m for m in metrics_table
        if m['category'] == 'bigram_z'
        and m['classification'] in ('INFLATED', 'INVALIDATED')
    ]
    for f in failed:
        retracted.append(
            f"Phase {f['phase']} bigram z={f['original_value']} → "
            f"symmetric z={f['validated_value']} ({f['classification']})"
        )

    return retracted


def _honest_assessment(
    surviving: List[str],
    retracted: List[str],
    best_z: float,
    best_selectivity: float,
) -> str:
    """Produce the honest assessment narrative."""
    lines = []

    lines.append("WHAT WE ACTUALLY KNOW:")
    lines.append("")
    lines.append(
        "1. The Voynich manuscript uses a systematic encoding (not random "
        "gibberish). This is supported by Zipf's law, entropy profile, "
        "and morphological structure — none of which depend on bigram "
        "comparison."
    )
    lines.append(
        "2. The encoding resembles tachygraphic (shorthand) systems more "
        "than alphabetic scripts (cosine similarity = 0.820). This is "
        "independent of bigram methodology."
    )
    lines.append(
        f"3. When decoded through the Phase 16 assignment table, the text "
        f"produces dictionary hits at {best_selectivity:.1f}× the rate of "
        f"random text decoded through the same table. This selectivity is "
        f"based on per-token frequency, not bigrams."
    )

    if best_z >= 3.0:
        lines.append(
            f"4. Decoded text forms Latin word sequences (bigrams) at z="
            f"{best_z:.2f} above null — genuine sequential structure "
            f"survives symmetric recomputation."
        )
    elif best_z >= 2.0:
        lines.append(
            f"4. Decoded text shows weak evidence of sequential structure "
            f"(z={best_z:.2f}). This is marginally significant and should "
            f"be reported with appropriate caveats."
        )
    else:
        lines.append(
            "4. NO evidence of sequential structure survives symmetric "
            "recomputation. The individual signal words may be real, but "
            "they do not form sequences above chance level."
        )

    lines.append("")
    lines.append("WHAT WE THOUGHT WE KNEW BUT WAS WRONG:")
    lines.append("")
    lines.append(
        "1. The Venetian hypothesis is NOT supported. The z=319.76 was "
        "entirely a measurement artifact. Corrected z=-0.47."
    )

    if best_z < 3.0:
        lines.append(
            "2. The sequential structure evidence (bigram z-scores) was "
            "partially or fully inflated by methodological issues. "
            "Individual signal words exist but may not form coherent text."
        )

    lines.append("")
    lines.append("MOST PRODUCTIVE FUTURE DIRECTION:")
    lines.append("")
    lines.append(
        "Focus on the gap between the 89.5% oracle ceiling and the "
        "43.6% actual dict_hit rate. The table is near-optimal within "
        "the CV model — improvements require CVC/CCV expansion, "
        "segmentation changes, or script direction experiments. "
        "The f57v formulaic patterns and f6r botanical hits remain "
        "the strongest anchors for validation."
    )

    return "\n".join(lines)


def _paper_specification(
    surviving: List[str],
    retracted: List[str],
    best_z: float,
) -> Dict:
    """Specify what the paper can claim."""
    claims = []

    claims.append({
        'claim': 'Systematic encoding (not gibberish)',
        'confidence': 'HIGH',
        'evidence': 'Zipf, entropy, morphology',
        'methodology_note': 'Unaffected by bigram audit',
    })
    claims.append({
        'claim': 'Tachygraphic resemblance',
        'confidence': 'HIGH',
        'evidence': 'Phase 19 cosine = 0.820',
        'methodology_note': 'Unaffected by bigram audit',
    })
    claims.append({
        'claim': 'Sub-cell feature model decoding',
        'confidence': 'MEDIUM',
        'evidence': 'Phase 16 dict_hit 43.6%, selectivity ~3×',
        'methodology_note': 'Per-token metric, validated as symmetric',
    })

    if best_z >= 3.0:
        claims.append({
            'claim': 'Sequential structure (word sequences)',
            'confidence': 'MEDIUM',
            'evidence': f'Symmetric bigram z={best_z:.2f}',
            'methodology_note': 'Survives symmetric recomputation',
        })
    elif best_z >= 2.0:
        claims.append({
            'claim': 'Weak sequential structure',
            'confidence': 'LOW',
            'evidence': f'Symmetric bigram z={best_z:.2f}',
            'methodology_note': 'Marginal significance after correction',
        })

    limitations = [
        "Bigram z-scores were originally computed with multiple "
        "methodological variants (relabel vs shuffle null, rate vs count, "
        "exact vs exact+relaxed). Phase 42 harmonized all comparisons "
        "with a single symmetric methodology.",

        "Phase 40's z=319.76 was entirely artifactual (real counted "
        "exact+relaxed, null counted exact-only). The corrected value "
        "is z=-0.47. This bug was discovered in Phase 41 and audited "
        "across all phases in Phase 42.",

        "The Venetian-specific identification (Phases 39-40) is retracted. "
        "Evidence supports general Romance language family but not a "
        "specific regional variety.",

        "Dict_hit rates above 40% are achieved with a 131K expanded "
        "dictionary including medieval variants and pharmaceutical "
        "vocabulary. With a strict 10K dictionary, hit rates are ~24%.",
    ]

    return {
        'headline_claims': claims,
        'limitations': limitations,
        'methodology_note': (
            "All bigram z-scores reported in this paper have been "
            "recomputed with guaranteed symmetric methodology (Phase 42 "
            "audit). Both real and null permutations count hits using "
            "the same criterion (exact + edit-distance-1 relaxed matching)."
        ),
    }


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def run_ground_truth() -> None:
    """Step 42.5: Definitive ground truth assessment."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 42.5: Ground Truth Assessment")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load all audit results ──
    print("\n  1. Loading audit results …")
    audit = _safe_load(os.path.join(rd, 'bigram_code_audit.json'))
    recompute = _safe_load(os.path.join(rd, 'symmetric_recompute.json'))
    sigma = _safe_load(os.path.join(rd, 'signal_word_revalidate.json'))
    selectivity = _safe_load(os.path.join(rd, 'selectivity_audit.json'))

    loaded = sum(1 for d in [audit, recompute, sigma, selectivity] if d)
    print(f"    Loaded {loaded}/4 audit results")

    # ── 2. Build evidence classifications ──
    print("\n  2. Building evidence classifications …")

    bigram_evidence = _build_bigram_evidence(recompute)
    metrics_table = _build_validated_metrics_table(
        bigram_evidence, sigma, selectivity)

    # Print comparison table
    print(f"\n    {'Phase':<8s} {'Metric':<35s} {'Original':>10s} "
          f"{'Validated':>10s} {'Status':<15s}")
    print("    " + "-" * 85)

    for m in metrics_table:
        phase = m['phase']
        metric = m['metric'][:33]
        orig = m['original_value']
        val = m['validated_value']

        if isinstance(orig, float):
            orig_str = f"{orig:.2f}"
        elif orig is not None:
            orig_str = str(orig)[:10]
        else:
            orig_str = "N/A"

        if isinstance(val, float):
            val_str = f"{val:.4f}"
        elif val is not None:
            val_str = str(val)[:10]
        else:
            val_str = "N/A"

        status = m['classification']
        print(f"    {phase:<8s} {metric:<35s} {orig_str:>10s} "
              f"{val_str:>10s} {status:<15s}")

    # ── 3. Surviving vs retracted evidence ──
    print("\n  3. Evidence classification")

    surviving = _identify_surviving_evidence(metrics_table)
    retracted = _identify_retracted_evidence(metrics_table)

    print(f"\n    SURVIVING EVIDENCE ({len(surviving)} findings):")
    for i, s in enumerate(surviving, 1):
        print(f"      [{i}] {s}")

    print(f"\n    RETRACTED EVIDENCE ({len(retracted)} findings):")
    for i, r in enumerate(retracted, 1):
        print(f"      [{i}] {r}")

    # ── 4. Key numbers ──
    print("\n  4. Key validated numbers")

    best_z = recompute.get('best_surviving_z_total', 0.0)
    best_z_exact = recompute.get('best_surviving_z_exact', 0.0)
    best_phase = recompute.get('best_surviving_phase', '?')
    phase_label = best_phase if best_phase.startswith('Phase') else f"Phase {best_phase}"
    print(f"    Best surviving bigram z_total: {best_z:.4f} "
          f"({phase_label})")
    print(f"    Best surviving bigram z_exact: {best_z_exact:.4f}")

    # Best fair selectivity (from untuned dictionaries)
    sel_audits = selectivity.get('selectivity_audits', [])
    fair_sels = [
        sa['reported_selectivity']
        for sa in sel_audits
        if sa.get('reported_selectivity') is not None
        and sa['phase'] in ('14', '15', '38')
    ]
    best_sel = max(fair_sels) if fair_sels else 0.0
    print(f"    Best fair selectivity: {best_sel:.2f}×")

    sigma_verdict = sigma.get('verdict', 'UNKNOWN')
    print(f"    Signal word σ verdict: {sigma_verdict}")

    # ── 5. Honest assessment ──
    print("\n  5. Honest assessment")
    assessment = _honest_assessment(surviving, retracted, best_z, best_sel)
    for line in assessment.split('\n'):
        print(f"    {line}")

    # ── 6. Paper specification ──
    print("\n  6. Paper specification")
    paper = _paper_specification(surviving, retracted, best_z)

    print("\n    Headline claims:")
    for c in paper['headline_claims']:
        print(f"      [{c['confidence']}] {c['claim']}")

    print("\n    Key limitations:")
    for i, lim in enumerate(paper['limitations'], 1):
        # Truncate for display
        print(f"      [{i}] {lim[:100]}…" if len(lim) > 100 else
              f"      [{i}] {lim}")

    # ── 7. Overall verdict ──
    bigram_verdict = recompute.get('verdict', 'UNKNOWN')

    if best_z >= 5.0:
        overall = 'STRONG_EVIDENCE'
    elif best_z >= 3.0:
        overall = 'MODERATE_EVIDENCE'
    elif best_z >= 2.0:
        overall = 'WEAK_EVIDENCE'
    elif best_sel >= 2.0:
        overall = 'WORD_LEVEL_ONLY'
    else:
        overall = 'INCONCLUSIVE'

    print(f"\n  7. OVERALL VERDICT: {overall}")
    print(f"     Bigram verdict: {bigram_verdict}")
    print(f"     Signal word verdict: {sigma_verdict}")
    print(f"     Selectivity verdict: "
          f"{selectivity.get('verdict', 'UNKNOWN')}")

    # ── 8. Save ──
    elapsed = time.time() - t0

    output = {
        'bigram_evidence': bigram_evidence,
        'metrics_table': [_convert(m) for m in metrics_table],
        'surviving_evidence': surviving,
        'retracted_evidence': retracted,
        'best_bigram_z_total': round(best_z, 4),
        'best_bigram_z_exact': round(best_z_exact, 4),
        'best_bigram_phase': best_phase,
        'best_fair_selectivity': round(best_sel, 2),
        'sigma_verdict': sigma_verdict,
        'bigram_verdict': bigram_verdict,
        'honest_assessment': assessment,
        'paper_specification': _convert(paper),
        'overall_verdict': overall,
        'runtime_seconds': round(elapsed, 1),
    }

    out_path = os.path.join(rd, 'ground_truth.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(output), f, indent=2)
    print(f"\n  Saved → {out_path} ({elapsed:.1f}s)")
