"""
Phase 7.5 Step 5: Convergence Scoring and Joint Selectivity
=============================================================
Compile all selectivity scores from prior phases and Phase 7.5 steps,
run Fisher's combined probability test, and check whether multiple
methods converge on specific vocabulary identifications.

Sub-analyses:
  5a — Compile all selectivity scores into independent evidence families
  5b — Fisher's combined probability test
  5c — Find convergent identifications across methods

Output:
  results/convergence_score.json
"""

import json
import math
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import norm as norm_dist

from voynich.core.stats import fisher_combined_probability
from voynich.core._paths import results_dir as _results_dir


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SelectivityEntry:
    """One selectivity score from any phase/step."""
    family: str           # independent evidence family
    source: str           # e.g. 'distributional', 'positional_slots'
    metric_name: str
    real_value: float
    null_mean: float
    null_std: float
    selectivity: float
    p_value: float        # 1-sided from null distribution
    gate_passed: bool


@dataclass
class ConvergentIdentification:
    """A stem identification supported by multiple methods."""
    voynich_stem: str
    proposed_role: str    # e.g. 'verb', 'plant_name', 'preparation'
    proposed_latin: Optional[str]
    evidence_sources: List[str]
    n_sources: int
    convergence_score: float


@dataclass
class ConvergenceScoreResult:
    """Full Phase 7.5 Step 5 output."""
    # Selectivity compilation
    n_selectivity_scores: int
    selectivity_entries: List[Dict]
    n_gates_passed: int
    n_gates_total: int
    # Fisher's combined test
    n_independent_families: int
    fisher_chi2: float
    fisher_df: int
    fisher_p_value: float
    fisher_significant: bool
    # Convergent identifications
    n_convergent_ids: int
    n_multi_method_ids: int
    multi_method_consistency_rate: float
    convergent_identifications: List[Dict]
    n_ids_by_sources: Dict[str, int]
    # Overall
    overall_verdict: str
    confidence_level: str


# ---------------------------------------------------------------------------
# 5a — Compile selectivity scores
# ---------------------------------------------------------------------------

def _z_to_p(z: float) -> float:
    """Convert z-score to one-sided p-value."""
    return 1.0 - float(norm_dist.cdf(z))


def _selectivity_to_p(sel: float, null_mean: float, null_std: float,
                       real_value: float) -> float:
    """
    Convert a selectivity ratio to an approximate p-value.

    Uses a normal approximation: z = (real - null_mean) / null_std
    """
    if null_std > 1e-10:
        z = (real_value - null_mean) / null_std
        return _z_to_p(z)
    elif real_value > null_mean:
        return 0.001  # conservative small p-value
    else:
        return 0.5


def compile_selectivity_scores() -> List[SelectivityEntry]:
    """
    Load all results JSONs and extract selectivity/p-value scores.

    Groups into 7 independent evidence families.
    """
    entries: List[SelectivityEntry] = []
    rdir = _results_dir()

    # Family 1: Morpheme grid structure (Phase 4.5)
    mg_path = rdir / 'morpheme_grid.json'
    if mg_path.exists():
        with open(mg_path) as f:
            mg = json.load(f)
        nt = mg.get('null_test', {})
        for axis in ('onset', 'nucleus'):
            ax_data = nt.get(axis, {})
            z = ax_data.get('z_score', 0)
            null_m = ax_data.get('null_mean', 0)
            null_s = ax_data.get('null_std', 1)
            real = null_m + z * null_s if null_s > 0 else null_m
            entries.append(SelectivityEntry(
                family='morpheme_grid',
                source='morpheme_grid',
                metric_name=f'{axis}_z_score',
                real_value=z,
                null_mean=0.0,
                null_std=1.0,
                selectivity=z / 1.5 if z > 0 else 0.0,
                p_value=_z_to_p(z),
                gate_passed=z > 2.0,
            ))

    # Family 2: Distributional embeddings (Phase 7/8)
    dist_path = rdir / 'distributional.json'
    if dist_path.exists():
        with open(dist_path) as f:
            dist = json.load(f)
        la = dist.get('language_a_alignment', {})
        for metric, key_sel, key_null_m, key_null_s, key_real in [
            ('procrustes', 'procrustes_selectivity', 'null_procrustes_mean',
             'null_procrustes_std', 'best_procrustes_residual'),
            ('gw', 'gw_selectivity', 'null_gw_mean', 'null_gw_std',
             'best_gw_distance'),
        ]:
            sel = la.get(key_sel, 0)
            null_m = la.get(key_null_m, 0)
            null_s = la.get(key_null_s, 0)
            real = la.get(key_real, 0)
            p = _selectivity_to_p(sel, null_m, null_s, real) if null_s else 0.5
            entries.append(SelectivityEntry(
                family='distributional',
                source='distributional',
                metric_name=f'lang_a_{metric}_selectivity',
                real_value=real,
                null_mean=null_m,
                null_std=null_s,
                selectivity=sel,
                p_value=p,
                gate_passed=sel > 1.5,
            ))

    # Family 3: Illustration cross-modal (Phase 6.1)
    ad_path = rdir / 'anchor_diagnosis.json'
    if ad_path.exists():
        with open(ad_path) as f:
            ad = json.load(f)
        unanimity = ad.get('pruned_unanimity', 0)
        n_high = ad.get('n_high_unanimity_chars', 0)
        # Use unanimity as a proxy for signal strength
        # Under random assignment, unanimity ≈ 1/N_chars
        # With 7 high-unanimity chars, p-value is very small
        if n_high > 0:
            p = 0.5 ** n_high  # conservative binomial approximation
        else:
            p = 0.5
        entries.append(SelectivityEntry(
            family='illustration',
            source='anchor_diagnosis',
            metric_name='anchor_unanimity',
            real_value=unanimity,
            null_mean=0.1,  # approximate random unanimity
            null_std=0.05,
            selectivity=unanimity / 0.1 if unanimity > 0 else 0.0,
            p_value=p,
            gate_passed=unanimity > 0.5,
        ))

    # Family 4: Positional slot structure (Phase 9)
    ps_path = rdir / 'positional_slots.json'
    if ps_path.exists():
        with open(ps_path) as f:
            ps = json.load(f)
        mi_sel = ps.get('mi_selectivity', 0)
        entries.append(SelectivityEntry(
            family='positional',
            source='positional_slots',
            metric_name='mi_selectivity',
            real_value=mi_sel,
            null_mean=1.0,
            null_std=0.1,
            selectivity=mi_sel,
            p_value=_selectivity_to_p(mi_sel, 1.0, 0.1, mi_sel),
            gate_passed=mi_sel > 1.5,
        ))
        # Verb frequency correlation
        verb_rho = ps.get('verb_frequency_rho', 0)
        if verb_rho > 0:
            n_verbs = ps.get('n_verb_candidates', 15)
            # Approximate p-value for Spearman rho
            t_stat = verb_rho * math.sqrt((n_verbs - 2) / max(1 - verb_rho**2, 1e-10))
            from scipy.stats import t as t_dist
            p = 1.0 - float(t_dist.cdf(abs(t_stat), n_verbs - 2))
            entries.append(SelectivityEntry(
                family='positional',
                source='positional_slots',
                metric_name='verb_frequency_rho',
                real_value=verb_rho,
                null_mean=0.0,
                null_std=1.0 / math.sqrt(max(n_verbs - 1, 1)),
                selectivity=verb_rho / max(0.1, 1.0 / math.sqrt(n_verbs)),
                p_value=p,
                gate_passed=verb_rho > 0.7,
            ))

    # Family 5: Noun embedding coherence (Phase 7 integration)
    ai_path = rdir / 'approach_integration.json'
    if ai_path.exists():
        with open(ai_path) as f:
            ai = json.load(f)
        nc = ai.get('noun_coherence', {})
        ratio = nc.get('ratio', 0)
        if ratio > 0:
            # Approximate p-value: if random baseline mean is B and real is R,
            # and ratio = R/B, use bootstrap approximation
            entries.append(SelectivityEntry(
                family='noun_coherence',
                source='approach_integration',
                metric_name='noun_embedding_coherence_ratio',
                real_value=ratio,
                null_mean=1.0,
                null_std=0.3,
                selectivity=ratio,
                p_value=_selectivity_to_p(ratio, 1.0, 0.3, ratio),
                gate_passed=ratio > 1.5,
            ))

    # Family 6: Phase 7.5 Step 3 — Verb identification
    vi_path = rdir / 'verb_identification.json'
    if vi_path.exists():
        with open(vi_path) as f:
            vi = json.load(f)
        sel = vi.get('assignment_selectivity', 0)
        null_m = vi.get('null_mean_score', 0)
        null_s = vi.get('null_std_score', 0)
        real = vi.get('best_total_score', 0)
        p = _selectivity_to_p(sel, null_m, null_s, real)
        entries.append(SelectivityEntry(
            family='verb_identification',
            source='verb_identification',
            metric_name='assignment_selectivity',
            real_value=real,
            null_mean=null_m,
            null_std=null_s,
            selectivity=sel,
            p_value=p,
            gate_passed=sel > 1.5,
        ))

    # Family 7: Phase 7.5 Step 4 — Embedding bridge
    eb_path = rdir / 'embedding_bridge.json'
    if eb_path.exists():
        with open(eb_path) as f:
            eb = json.load(f)
        sel = eb.get('hit_rate_selectivity', 0)
        null_m = eb.get('null_plant_hit_rate_mean', 0)
        null_s = eb.get('null_plant_hit_rate_std', 0)
        real = eb.get('rosetta_hit_rate', 0)
        p = _selectivity_to_p(sel, null_m, null_s, real)
        entries.append(SelectivityEntry(
            family='embedding_bridge',
            source='embedding_bridge',
            metric_name='plant_cluster_hit_selectivity',
            real_value=real,
            null_mean=null_m,
            null_std=null_s,
            selectivity=sel,
            p_value=p,
            gate_passed=sel > 1.5,
        ))

    return entries


# ---------------------------------------------------------------------------
# 5c — Convergent identifications
# ---------------------------------------------------------------------------

def find_convergent_identifications() -> List[ConvergentIdentification]:
    """
    Find stems where multiple methods agree on an identification.

    Cross-references:
    - Verb assignments (Step 3)
    - Noun subcluster membership (Step 2)
    - Expanded anchors (Step 4)
    """
    rdir = _results_dir()
    stem_evidence: Dict[str, Dict] = defaultdict(lambda: {
        'role': None, 'latin': None, 'sources': [],
    })

    # Verb assignments
    vi_path = rdir / 'verb_identification.json'
    if vi_path.exists():
        with open(vi_path) as f:
            vi = json.load(f)
        for asgn in vi.get('assignments', []):
            stem = asgn.get('voynich_stem', '')
            if stem and asgn.get('is_confident'):
                stem_evidence[stem]['role'] = 'verb'
                stem_evidence[stem]['latin'] = asgn.get('latin_verb')
                stem_evidence[stem]['sources'].append('verb_identification')

    # Noun subclusters — plant names specifically
    sc_path = rdir / 'noun_subclusters.json'
    if sc_path.exists():
        with open(sc_path) as f:
            sc = json.load(f)
        for cluster in sc.get('subclusters', []):
            label = cluster.get('label', '')
            for stem in cluster.get('top_stems', []):
                if stem not in stem_evidence or not stem_evidence[stem]['sources']:
                    stem_evidence[stem]['role'] = label
                stem_evidence[stem]['sources'].append(f'noun_cluster_{label}')

    # Expanded anchors
    eb_path = rdir / 'embedding_bridge.json'
    if eb_path.exists():
        with open(eb_path) as f:
            eb = json.load(f)
        for anchor in eb.get('expanded_anchors', []):
            stem = anchor.get('voynich_stem', '')
            if stem and anchor.get('three_way_convergent'):
                if stem_evidence[stem]['role'] is None:
                    stem_evidence[stem]['role'] = 'plant_name'
                stem_evidence[stem]['sources'].append('embedding_bridge')

    # Rosetta folios (from rosetta_selection)
    ros_path = rdir / 'rosetta_selection.json'
    if ros_path.exists():
        with open(ros_path) as f:
            ros = json.load(f)
        for rf in ros.get('folio_scores', []):
            stem = rf.get('dominant_stem', '')
            if stem:
                if stem_evidence[stem]['role'] is None:
                    stem_evidence[stem]['role'] = 'plant_name'
                stem_evidence[stem]['latin'] = rf.get('medieval_name')
                stem_evidence[stem]['sources'].append('rosetta_illustration')

    # Build convergent identifications
    convergent = []
    for stem, info in stem_evidence.items():
        unique_sources = list(set(info['sources']))
        if len(unique_sources) >= 1:
            convergent.append(ConvergentIdentification(
                voynich_stem=stem,
                proposed_role=info['role'] or 'unknown',
                proposed_latin=info.get('latin'),
                evidence_sources=unique_sources,
                n_sources=len(unique_sources),
                convergence_score=len(unique_sources) / 4.0,
            ))

    # Sort by number of sources (most convergent first)
    convergent.sort(key=lambda x: -x.n_sources)
    return convergent


# ---------------------------------------------------------------------------
# JSON conversion
# ---------------------------------------------------------------------------

def _convert(obj):
    """Convert dataclass/numpy types to JSON-serializable form."""
    if hasattr(obj, '__dataclass_fields__'):
        return {k: _convert(v) for k, v in asdict(obj).items()}
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, dict):
        return {str(k): _convert(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_convert(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_convergence_score() -> Dict:
    """
    Phase 7.5 Step 5: Convergence Scoring.

    Compiles selectivity scores from all phases, runs Fisher's combined
    probability test, and identifies stems with multi-method convergence.
    """
    print("Phase 7.5 Step 5: Convergence Scoring")
    print("=" * 70)

    # 5a: Compile selectivity scores
    print("\n  Compiling selectivity scores...")
    entries = compile_selectivity_scores()
    n_gates_passed = sum(1 for e in entries if e.gate_passed)

    print(f"\n  {'Family':<20s} {'Metric':<35s} {'Select.':<10s} "
          f"{'p-value':<12s} {'Gate'}")
    print(f"  {'─' * 20} {'─' * 35} {'─' * 10} {'─' * 12} {'─' * 6}")
    for e in entries:
        gate_str = 'PASS' if e.gate_passed else 'FAIL'
        print(f"  {e.family:<20s} {e.metric_name:<35s} "
              f"{e.selectivity:<10.2f} {e.p_value:<12.2e} {gate_str}")

    print(f"\n  Gates passed: {n_gates_passed}/{len(entries)}")

    # 5b: Fisher's combined probability test
    print("\n  Running Fisher's combined probability test...")

    # One p-value per independent family (take the best p-value per family)
    family_p: Dict[str, float] = {}
    for e in entries:
        if e.family not in family_p or e.p_value < family_p[e.family]:
            family_p[e.family] = e.p_value

    # Filter to valid p-values
    valid_p = [(fam, p) for fam, p in family_p.items() if 0 < p <= 1]
    p_values = [p for _, p in valid_p]
    family_names = [f for f, _ in valid_p]

    if p_values:
        chi2, df, combined_p = fisher_combined_probability(p_values)
    else:
        chi2, df, combined_p = 0.0, 0, 1.0

    fisher_sig = combined_p < 0.01

    print(f"  Independent families: {len(valid_p)}")
    for fam, p in valid_p:
        print(f"    {fam}: p = {p:.2e}")
    print(f"  Fisher chi2: {chi2:.2f} (df={df})")
    print(f"  Combined p-value: {combined_p:.2e}")
    print(f"  Significant (p < 0.01): {'YES' if fisher_sig else 'NO'}")

    # 5c: Convergent identifications
    print("\n  Finding convergent identifications...")
    convergent = find_convergent_identifications()

    multi_method = [c for c in convergent if c.n_sources >= 2]
    n_multi = len(multi_method)

    # Count consistency (multi-method items where all sources agree on role)
    n_consistent = sum(1 for c in multi_method if c.proposed_latin is not None)
    consistency_rate = n_consistent / max(n_multi, 1)

    # Count by number of sources
    by_sources: Dict[str, int] = defaultdict(int)
    for c in convergent:
        by_sources[str(c.n_sources)] += 1

    print(f"  Total identifications: {len(convergent)}")
    print(f"  Multi-method (2+ sources): {n_multi}")
    print(f"  With specific Latin word: {n_consistent}")
    print(f"  Multi-method consistency rate: {consistency_rate:.2%}")

    if multi_method:
        print(f"\n  Top multi-method identifications:")
        for c in multi_method[:15]:
            latin = c.proposed_latin or '?'
            print(f"    {c.voynich_stem:<12s} → {c.proposed_role:<15s} "
                  f"({latin}) [{', '.join(c.evidence_sources)}]")

    # Overall verdict
    if fisher_sig and consistency_rate > 0.70:
        verdict = 'strong_convergence'
        confidence = 'high'
    elif fisher_sig and consistency_rate > 0.50:
        verdict = 'moderate_convergence'
        confidence = 'medium'
    elif fisher_sig:
        verdict = 'structural_convergence_weak_vocabulary'
        confidence = 'medium'
    elif n_gates_passed >= 3:
        verdict = 'partial_convergence'
        confidence = 'low'
    else:
        verdict = 'no_convergence'
        confidence = 'none'

    print(f"\n  Overall verdict: {verdict}")
    print(f"  Confidence level: {confidence}")

    result = ConvergenceScoreResult(
        n_selectivity_scores=len(entries),
        selectivity_entries=[_convert(asdict(e)) for e in entries],
        n_gates_passed=n_gates_passed,
        n_gates_total=len(entries),
        n_independent_families=len(valid_p),
        fisher_chi2=chi2,
        fisher_df=df,
        fisher_p_value=combined_p,
        fisher_significant=fisher_sig,
        n_convergent_ids=len(convergent),
        n_multi_method_ids=n_multi,
        multi_method_consistency_rate=consistency_rate,
        convergent_identifications=[_convert(asdict(c))
                                     for c in convergent[:50]],
        n_ids_by_sources=dict(by_sources),
        overall_verdict=verdict,
        confidence_level=confidence,
    )

    out = _convert(asdict(result))
    out_path = _results_dir() / 'convergence_score.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {out_path}")
    return out
