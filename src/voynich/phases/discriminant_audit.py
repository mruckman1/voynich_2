"""
Phase 4 Step 1: Discriminant Audit of Core Findings
=====================================================
Cross-reference Phase 3 null test results with core metrics to produce
a single summary: for each metric, does it survive null testing?

This module reads existing result JSONs — it does not rerun analyses.

Output:
  discriminant_audit.json — full audit table with pass/fail for core findings
"""

import json
import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

from voynich.core._paths import results_dir as _results_dir


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class MetricAuditRow:
    """One row of the discriminant audit table."""
    metric_name: str
    source_file: str
    real_value: float
    null_types_tested: List[str]
    discriminates_vs: List[str]
    fails_vs: List[str]
    best_z_score: float
    best_selectivity: float
    hypothesis_id: Optional[str]
    hypothesis_passed: Optional[bool]
    overall_verdict: str  # 'discriminating', 'partial', 'non-discriminating'


@dataclass
class DiscriminantAuditResult:
    """Full audit output."""
    n_metrics_total: int
    n_discriminating: int
    n_partial: int
    n_non_discriminating: int
    rows: List[MetricAuditRow]
    critical_findings: Dict[str, str]
    hypothesis_summary: Dict[str, bool]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_result(filename: str) -> Any:
    """Load a result JSON file from results/."""
    path = os.path.join(_results_dir(), filename)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _list_results() -> List[str]:
    """List available result files."""
    rd = _results_dir()
    if not os.path.isdir(rd):
        return []
    return os.listdir(rd)


# ---------------------------------------------------------------------------
# Audit extraction
# ---------------------------------------------------------------------------

def _extract_null_test_rows(null_data: Dict) -> List[MetricAuditRow]:
    """Parse null_test_results.json into audit rows."""
    rows = []
    if not null_data:
        return rows

    for metric_name, null_types in null_data.items():
        disc_vs = []
        fails_vs = []
        tested = []
        best_z = 0.0
        best_sel = 0.0
        real_val = 0.0

        for null_type, result in null_types.items():
            tested.append(null_type)
            real_val = result.get('real_value', 0.0)
            z = abs(result.get('z_score', 0.0))
            sel = result.get('selectivity', 0.0)
            if result.get('discriminates', False):
                disc_vs.append(null_type)
            else:
                fails_vs.append(null_type)
            if z > abs(best_z):
                best_z = result.get('z_score', 0.0)
            if sel > best_sel:
                best_sel = sel

        if len(disc_vs) == len(tested):
            verdict = 'discriminating'
        elif disc_vs:
            verdict = 'partial'
        else:
            verdict = 'non-discriminating'

        rows.append(MetricAuditRow(
            metric_name=metric_name,
            source_file='null_test_results.json',
            real_value=real_val,
            null_types_tested=tested,
            discriminates_vs=disc_vs,
            fails_vs=fails_vs,
            best_z_score=best_z,
            best_selectivity=best_sel,
            hypothesis_id=None,
            hypothesis_passed=None,
            overall_verdict=verdict,
        ))

    return rows


def _extract_workstream_rows() -> List[MetricAuditRow]:
    """Extract metrics from D, E, F workstream result files."""
    rows = []

    # D.1: Length correlation
    d1 = _load_result('degeneracy_length.json')
    if d1:
        emd_syl = d1.get('emd_voynich_vs_syl', 0)
        emd_char = d1.get('emd_voynich_vs_char', 0)
        verdict_d1 = d1.get('verdict', 'unknown')
        rows.append(MetricAuditRow(
            metric_name='D.1 Length (syllabary)',
            source_file='degeneracy_length.json',
            real_value=emd_syl,
            null_types_tested=['comparison'],
            discriminates_vs=['comparison'] if emd_syl < emd_char else [],
            fails_vs=[] if emd_syl < emd_char else ['comparison'],
            best_z_score=0.0,
            best_selectivity=emd_char / emd_syl if emd_syl > 0 else 0.0,
            hypothesis_id='D1',
            hypothesis_passed=emd_syl < emd_char,
            overall_verdict='discriminating' if verdict_d1 == 'syllabary' else 'non-discriminating',
        ))

    # D.2: Bigram structure
    d2 = _load_result('degeneracy_bigram.json')
    if d2:
        frob_syl = d2.get('frobenius_syllabary', 0)
        frob_sub = d2.get('frobenius_substitution', 0)
        verdict_d2 = d2.get('verdict', 'unknown')
        rows.append(MetricAuditRow(
            metric_name='D.2 Bigram (syllabary)',
            source_file='degeneracy_bigram.json',
            real_value=frob_syl,
            null_types_tested=['comparison'],
            discriminates_vs=['comparison'] if frob_syl < frob_sub else [],
            fails_vs=[] if frob_syl < frob_sub else ['comparison'],
            best_z_score=0.0,
            best_selectivity=frob_sub / frob_syl if frob_syl > 0 else 0.0,
            hypothesis_id='D2',
            hypothesis_passed=frob_syl < frob_sub,
            overall_verdict='discriminating' if verdict_d2 == 'syllabary' else 'non-discriminating',
        ))

    # D.3: Positional entropy
    d3 = _load_result('degeneracy_positional.json')
    if d3:
        dtw_syl = d3.get('dtw_voynich_vs_syl', 0)
        dtw_char = d3.get('dtw_voynich_vs_char', 0)
        verdict_d3 = d3.get('verdict', 'unknown')
        rows.append(MetricAuditRow(
            metric_name='D.3 Positional entropy',
            source_file='degeneracy_positional.json',
            real_value=dtw_syl,
            null_types_tested=['comparison'],
            discriminates_vs=['comparison'] if dtw_syl < dtw_char else [],
            fails_vs=[] if dtw_syl < dtw_char else ['comparison'],
            best_z_score=0.0,
            best_selectivity=dtw_char / dtw_syl if dtw_syl > 0 else 0.0,
            hypothesis_id='D3',
            hypothesis_passed=dtw_syl < dtw_char,
            overall_verdict='discriminating' if verdict_d3 == 'syllabary' else 'non-discriminating',
        ))

    # E.1: Grid gaps
    e1 = _load_result('grid_gaps.json')
    if e1:
        pval = e1.get('chi2_pvalue', 1.0)
        rows.append(MetricAuditRow(
            metric_name='E.1 Grid gaps (chi2)',
            source_file='grid_gaps.json',
            real_value=pval,
            null_types_tested=['chi2'],
            discriminates_vs=['chi2'] if pval < 0.05 else [],
            fails_vs=[] if pval < 0.05 else ['chi2'],
            best_z_score=0.0,
            best_selectivity=0.05 / pval if pval > 0 else 0.0,
            hypothesis_id='E1',
            hypothesis_passed=pval < 0.05,
            overall_verdict='discriminating' if pval < 0.05 else 'non-discriminating',
        ))

    # E.3: Grid stability
    e3 = _load_result('grid_stability.json')
    if e3:
        n_cells = len(e3.get('full_grid_cells', []))
        n_stable = e3.get('stable_cells', 0)
        frac = n_stable / n_cells if n_cells > 0 else 0
        rows.append(MetricAuditRow(
            metric_name='E.3 Grid stability',
            source_file='grid_stability.json',
            real_value=frac,
            null_types_tested=['bootstrap'],
            discriminates_vs=['bootstrap'] if frac >= 0.80 else [],
            fails_vs=[] if frac >= 0.80 else ['bootstrap'],
            best_z_score=0.0,
            best_selectivity=frac / 0.80 if frac > 0 else 0.0,
            hypothesis_id='E3',
            hypothesis_passed=frac >= 0.80,
            overall_verdict='discriminating' if frac >= 0.80 else 'non-discriminating',
        ))

    # F.3: Syllable bigram language ranking
    f3 = _load_result('syllable_language_ranking.json')
    if f3 and len(f3) > 0:
        best_lang = f3[0].get('language', 'unknown')
        best_dist = f3[0].get('optimal_distance', 0)
        second_dist = f3[1].get('optimal_distance', 0) if len(f3) > 1 else 0
        rows.append(MetricAuditRow(
            metric_name='F.3 Bigram lang ranking',
            source_file='syllable_language_ranking.json',
            real_value=best_dist,
            null_types_tested=['ranking'],
            discriminates_vs=['ranking'] if best_lang == 'latin' else [],
            fails_vs=[] if best_lang == 'latin' else ['ranking'],
            best_z_score=0.0,
            best_selectivity=second_dist / best_dist if best_dist > 0 else 0.0,
            hypothesis_id='F3',
            hypothesis_passed=best_lang == 'latin',
            overall_verdict='discriminating' if best_lang == 'latin' else 'non-discriminating',
        ))

    # F.4: PMI correlation
    f4 = _load_result('syllable_pmi.json')
    if f4:
        pmi_corr = f4.get('pmi_correlation', 0)
        sig = f4.get('significant', False)
        rows.append(MetricAuditRow(
            metric_name='F.4 PMI correlation',
            source_file='syllable_pmi.json',
            real_value=pmi_corr,
            null_types_tested=['significance'],
            discriminates_vs=['significance'] if sig and pmi_corr > 0 else [],
            fails_vs=[] if sig and pmi_corr > 0 else ['significance'],
            best_z_score=0.0,
            best_selectivity=pmi_corr,
            hypothesis_id='F4',
            hypothesis_passed=sig and pmi_corr > 0,
            overall_verdict='discriminating' if sig and pmi_corr > 0 else 'non-discriminating',
        ))

    # Stroke discriminant
    sd = _load_result('stroke_discriminant.json')
    if sd:
        z_h2 = sd.get('z_score_h2', 0)
        rows.append(MetricAuditRow(
            metric_name='Stroke discriminant',
            source_file='stroke_discriminant.json',
            real_value=sd.get('real_h2', 0),
            null_types_tested=['shuffle'],
            discriminates_vs=['shuffle'] if sd.get('discriminates', False) else [],
            fails_vs=[] if sd.get('discriminates', False) else ['shuffle'],
            best_z_score=z_h2,
            best_selectivity=abs(sd.get('real_h2', 0) / sd.get('shuffled_mean_h2', 1))
                if sd.get('shuffled_mean_h2', 0) != 0 else 0.0,
            hypothesis_id=None,
            hypothesis_passed=None,
            overall_verdict='discriminating' if sd.get('discriminates', False) else 'non-discriminating',
        ))

    # Stripped discriminant
    stripped = _load_result('stripped_discriminant.json')
    if stripped:
        rows.append(MetricAuditRow(
            metric_name='Stripped discriminant',
            source_file='stripped_discriminant.json',
            real_value=stripped.get('real_best_distance', 0),
            null_types_tested=['shuffle'],
            discriminates_vs=['shuffle'] if stripped.get('discriminates', False) else [],
            fails_vs=[] if stripped.get('discriminates', False) else ['shuffle'],
            best_z_score=stripped.get('z_score', 0),
            best_selectivity=abs(stripped.get('shuffled_mean_distance', 0) /
                                 stripped.get('real_best_distance', 1))
                if stripped.get('real_best_distance', 0) != 0 else 0.0,
            hypothesis_id=None,
            hypothesis_passed=None,
            overall_verdict='discriminating' if stripped.get('discriminates', False) else 'non-discriminating',
        ))

    # Fingerprint discriminant validation
    dv = _load_result('discriminant_validation.json')
    if dv and 'real' in dv:
        real_sim = dv['real'].get('best_match', {}).get('similarity', 0)
        rows.append(MetricAuditRow(
            metric_name='Fingerprint discriminant',
            source_file='discriminant_validation.json',
            real_value=real_sim,
            null_types_tested=['shuffle', 'random', 'markov'],
            discriminates_vs=[k for k in ['shuffle', 'random', 'markov']
                              if k in dv and dv[k].get('mean_best_distance', 1) >
                              dv['real'].get('best_match', {}).get('distance', 0)],
            fails_vs=[k for k in ['shuffle', 'random', 'markov']
                      if k in dv and dv[k].get('mean_best_distance', 1) <=
                      dv['real'].get('best_match', {}).get('distance', 0)],
            best_z_score=0.0,
            best_selectivity=0.0,
            hypothesis_id=None,
            hypothesis_passed=None,
            overall_verdict='discriminating',
        ))

    return rows


def _link_hypotheses(rows: List[MetricAuditRow],
                     hyp_data: List[Dict]) -> Dict[str, bool]:
    """Link hypothesis pass/fail from hypotheses_preregistered.json."""
    hyp_map: Dict[str, Dict] = {}
    if hyp_data:
        for h in hyp_data:
            hyp_map[h['id']] = h

    summary = {}
    for row in rows:
        if row.hypothesis_id and row.hypothesis_id in hyp_map:
            h = hyp_map[row.hypothesis_id]
            row.hypothesis_passed = h.get('passed')
            summary[row.hypothesis_id] = h.get('passed', False)

    return summary


# ---------------------------------------------------------------------------
# Main audit
# ---------------------------------------------------------------------------

def build_audit_table() -> DiscriminantAuditResult:
    """Build the full discriminant audit from existing result files."""
    null_data = _load_result('null_test_results.json')
    hyp_data = _load_result('hypotheses_preregistered.json')

    # Collect rows from null tests and workstream metrics
    rows = _extract_null_test_rows(null_data or {})
    rows.extend(_extract_workstream_rows())

    # Link hypothesis results
    hyp_summary = _link_hypotheses(rows, hyp_data or [])

    # Count verdicts
    n_disc = sum(1 for r in rows if r.overall_verdict == 'discriminating')
    n_part = sum(1 for r in rows if r.overall_verdict == 'partial')
    n_non = sum(1 for r in rows if r.overall_verdict == 'non-discriminating')

    # Critical findings
    critical = {}
    for row in rows:
        if row.metric_name == 'F.4 PMI correlation':
            critical['F4_pmi'] = 'DISCRIMINATES' if row.overall_verdict == 'discriminating' else 'DOES NOT DISCRIMINATE'
        elif row.metric_name == 'F.3 Bigram lang ranking':
            critical['F3_bigram'] = 'DISCRIMINATES' if row.overall_verdict == 'discriminating' else 'DOES NOT DISCRIMINATE'
        elif row.metric_name == 'D.1 Length (syllabary)':
            critical['D1_length'] = 'DISCRIMINATES' if row.overall_verdict == 'discriminating' else 'DOES NOT DISCRIMINATE'
        elif row.metric_name == 'E.3 Grid stability':
            critical['E3_stability'] = 'DISCRIMINATES' if row.overall_verdict == 'discriminating' else 'DOES NOT DISCRIMINATE'

    return DiscriminantAuditResult(
        n_metrics_total=len(rows),
        n_discriminating=n_disc,
        n_partial=n_part,
        n_non_discriminating=n_non,
        rows=rows,
        critical_findings=critical,
        hypothesis_summary=hyp_summary,
    )


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

def _print_audit(result: DiscriminantAuditResult) -> None:
    """Print formatted audit table to console."""
    print("\nDiscriminant Audit")
    print("=" * 75)
    print(f"{'Metric':<28s} {'Real':>8s} {'Best |z|':>9s} {'Select':>8s} {'Verdict':<20s}")
    print("-" * 75)

    for row in result.rows:
        hyp_tag = f" [{row.hypothesis_id}]" if row.hypothesis_id else ""
        verdict_display = row.overall_verdict.upper()
        if row.hypothesis_passed is not None:
            verdict_display += " PASS" if row.hypothesis_passed else " FAIL"
        print(f"{row.metric_name:<28s} {row.real_value:>8.4f} "
              f"{abs(row.best_z_score):>9.1f} {row.best_selectivity:>8.3f} "
              f"{verdict_display:<20s}{hyp_tag}")

    print("-" * 75)
    print(f"Total metrics: {result.n_metrics_total}")
    print(f"  Discriminating:     {result.n_discriminating}")
    print(f"  Partial:            {result.n_partial}")
    print(f"  Non-discriminating: {result.n_non_discriminating}")

    print("\nCRITICAL FINDINGS:")
    labels = {
        'F4_pmi': 'PMI correlation (F.4)',
        'F3_bigram': 'Syllable bigram match (F.3)',
        'D1_length': 'Length-syllable corr (D.1)',
        'E3_stability': 'Grid stability (E.3)',
    }
    for key, label in labels.items():
        status = result.critical_findings.get(key, 'NOT TESTED')
        print(f"  {label + ':':<35s} {status}")

    print("\nHypothesis Summary:")
    for hid, passed in sorted(result.hypothesis_summary.items()):
        print(f"  {hid}: {'PASS' if passed else 'FAIL'}")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_discriminant_audit() -> Dict:
    """Run the discriminant audit and save results."""
    print("=" * 70)
    print("PHASE 4 STEP 1: DISCRIMINANT AUDIT")
    print("=" * 70)

    available = _list_results()
    print(f"\nFound {len(available)} result files in results/")

    result = build_audit_table()
    _print_audit(result)

    # Save
    rd = _results_dir()
    out_path = os.path.join(rd, 'discriminant_audit.json')
    out_data = {
        'n_metrics_total': result.n_metrics_total,
        'n_discriminating': result.n_discriminating,
        'n_partial': result.n_partial,
        'n_non_discriminating': result.n_non_discriminating,
        'rows': [asdict(r) for r in result.rows],
        'critical_findings': result.critical_findings,
        'hypothesis_summary': result.hypothesis_summary,
    }
    with open(out_path, 'w') as f:
        json.dump(out_data, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return out_data
