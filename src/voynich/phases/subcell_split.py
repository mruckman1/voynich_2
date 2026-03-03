"""
Phase 14.7 – Data-Driven Sub-Cell Splitting (Fallback)
========================================================
Takes the distributional cluster assignments from Phase 14.1
(cell_analysis.json) and splits multi-glyph cells into sub-cells based on
those clusters.  Runs the Phase 11 beam_search on the expanded grid
(20–30 cells) and compares dict_hit against the feature CSP (Step 14.3).

If the feature-driven and data-driven splitting produce the same result, they
validate each other.  If one outperforms, it's the better model.

This is the final fallback: if neither approach improves on 11.1%, the
ceiling is confirmed to be caused by something other than cell conflation.

Dependency chain:
    cell_analysis.json (Step 14.1)
    feature_csp.json   (Step 14.3)
        → subcell_split.json (this step)
"""

import json
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_cell_lookup,
    load_corpus,
    tokenize_eva_chars,
)
from voynich.core.reference import (
    build_cv_syllable_table,
    load_reference_corpus,
)
from voynich.core.stats import build_ngram_lm
from voynich.phases.csp_constraints import (
    AnchorConstraint,
    build_anchor_constraints,
    build_phoneme_inventory,
)
from voynich.phases.csp_solver import (
    CSPVariable,
    _convert,
    ac3_propagate,
    beam_search,
    build_csp_variables,
    decode_corpus,
    initialise_domains,
    score_assignment_full,
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SplitRecord:
    """How one original cell was split."""
    original_cell_key: str
    original_cv_label: str
    original_glyphs: List[str]
    n_clusters: int
    subcell_keys: List[str]          # new keys like "C2V3__0", "C2V3__1"
    subcell_glyph_groups: List[List[str]]  # glyphs per sub-cell


@dataclass
class SubcellSplitResult:
    """Full data-driven expanded-grid CSP result."""
    n_original_cells: int
    n_subcells: int
    splits_applied: List[Dict]
    feature_csp_dict_hit: float     # from feature_csp.json
    subcell_csp_dict_hit: float     # from this run
    subcell_csp_cross_entropy: float
    subcell_csp_selectivity: float
    best_assignment: Dict[str, str]
    better_approach: str            # "feature", "subcell", or "tie"
    decoded_sample: List[Any]
    gate_passed: bool
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Expand grid from cluster assignments
# ---------------------------------------------------------------------------

def _expand_cv_labels(
    cv_labels: Dict,
    cell_clusters: Dict[str, Dict],
) -> Tuple[Dict, List[SplitRecord]]:
    """Build an expanded cv_labels dict with split cells.

    For each cell where n_distinct_phonemes > 1, create one sub-cell per
    cluster.  Sub-cell keys are ``"<original_key>__<cluster_id>"``.
    Cells with a single cluster are left unchanged.

    Returns (expanded_cv_labels, list_of_splits).
    """
    expanded: Dict = {}
    splits: List[SplitRecord] = []

    for cell_key, info in cv_labels.items():
        cv_label = info.get('cv_label', cell_key)
        original_glyphs = info.get('glyphs', [])
        freq = info.get('frequency', 0)

        # Get cluster info for this cell from cell_analysis
        cluster_info = cell_clusters.get(cell_key)
        if cluster_info is None or cluster_info.get('n_distinct_phonemes', 1) <= 1:
            # No split — keep as-is
            expanded[cell_key] = dict(info)
            continue

        cluster_assignments: Dict[str, int] = cluster_info.get('cluster_assignments', {})
        n_clusters = cluster_info.get('n_distinct_phonemes', 1)

        # Group glyphs by cluster
        cluster_to_glyphs: Dict[int, List[str]] = {}
        for glyph in original_glyphs:
            cid = cluster_assignments.get(glyph, 0)
            cluster_to_glyphs.setdefault(cid, []).append(glyph)

        subcell_keys: List[str] = []
        subcell_glyph_groups: List[List[str]] = []
        for cid in sorted(cluster_to_glyphs.keys()):
            glyphs_in_cluster = cluster_to_glyphs[cid]
            subcell_key = f"{cell_key}__{cid}"
            subcell_keys.append(subcell_key)
            subcell_glyph_groups.append(glyphs_in_cluster)
            expanded[subcell_key] = {
                'cv_label': f"{cv_label}_{cid}",
                'glyphs': glyphs_in_cluster,
                'frequency': freq // max(n_clusters, 1),
            }

        splits.append(SplitRecord(
            original_cell_key=cell_key,
            original_cv_label=cv_label,
            original_glyphs=original_glyphs,
            n_clusters=n_clusters,
            subcell_keys=subcell_keys,
            subcell_glyph_groups=subcell_glyph_groups,
        ))

    return expanded, splits


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_subcell_split() -> None:
    """Step 14.7: data-driven expanded-grid CSP (fallback)."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE 14.7: Data-Driven Sub-Cell Splitting (Fallback)")
    print("=" * 70)

    rd = _results_dir()

    # Load prerequisites
    ca_path = os.path.join(rd, 'cell_analysis.json')
    cv_path = os.path.join(rd, 'cv_labels.json')
    fc_path = os.path.join(rd, 'feature_csp.json')

    for path, name in [(ca_path, 'cell_analysis.json'), (cv_path, 'cv_labels.json')]:
        if not os.path.exists(path):
            print(f"  [SKIP] {name} not found — run required predecessor steps first")
            return

    with open(ca_path) as f:
        cell_analysis = json.load(f)
    with open(cv_path) as f:
        cv_labels = json.load(f)

    # Load feature CSP dict_hit for comparison
    feature_dict_hit = 0.0
    if os.path.exists(fc_path):
        with open(fc_path) as f:
            fc_data = json.load(f)
        feature_dict_hit = fc_data.get('best_dict_hit', 0.0)
    print(f"\n  Feature CSP baseline (Step 14.3): {feature_dict_hit:.3f}")

    # Build cluster lookup: cell_key -> cluster_info
    cell_clusters: Dict[str, Dict] = {}
    for cr in cell_analysis.get('cell_results', []):
        cell_key = cr.get('cell_key', '')
        if cell_key:
            cell_clusters[cell_key] = cr

    # Expand cv_labels
    expanded_cv, splits = _expand_cv_labels(cv_labels, cell_clusters)
    n_original = len(cv_labels)
    n_expanded = len(expanded_cv)

    print(f"\n  Original cells: {n_original}")
    print(f"  Expanded cells (after splitting): {n_expanded}")
    print(f"  Cells split: {len(splits)}")
    for sp in splits:
        print(f"    {sp.original_cv_label}: {sp.original_glyphs} -> {sp.n_clusters} sub-cells")

    # Load corpus
    corpus = load_corpus(verbose=False)
    tokens = corpus.get_tokens(language='A', paragraph_only=True)
    if not tokens:
        print("  [SKIP] No Language A tokens found")
        return

    # Load Latin reference corpus (best language from Phase 11)
    ref_corpus = load_reference_corpus(verbose=False)
    latin_tokens = ref_corpus.get_combined_tokens('latin')
    if not latin_tokens:
        print("  [SKIP] No Latin reference corpus")
        return

    # Build lookups and inventory
    eva_to_subcell = build_eva_to_cell_lookup(expanded_cv)
    inventory = build_phoneme_inventory('latin', ref_corpus)
    lm = build_ngram_lm(latin_tokens[:10000], order=3, smoothing=0.01)
    ref_word_set = set(w.lower() for w in latin_tokens if len(w) >= 2)

    # Load anchor constraints (using original cell lookup for anchors, then remap)
    rosetta_path = os.path.join(rd, 'rosetta_selection.json')
    anchors: List[AnchorConstraint] = []
    if os.path.exists(rosetta_path):
        with open(rosetta_path) as f:
            rosetta_data = json.load(f)
        # Build anchors with original cell keys first, then remap to subcell keys
        original_anchors = build_anchor_constraints(rosetta_data, cv_labels)
        # Remap voynich_cells from original cell_key to subcell key
        # Use cluster 0 (first subcell) for any split cell
        split_remap: Dict[str, str] = {}
        for sp in splits:
            if sp.subcell_keys:
                split_remap[sp.original_cell_key] = sp.subcell_keys[0]
        for anchor in original_anchors:
            new_cells = [split_remap.get(ck, ck) for ck in anchor.voynich_cells]
            anchors.append(AnchorConstraint(
                folio=anchor.folio,
                voynich_stem=anchor.voynich_stem,
                voynich_cells=new_cells,
                target_word=anchor.target_word,
                target_syllables=anchor.target_syllables,
                weight=anchor.weight,
            ))
        print(f"\n  Anchor constraints: {len(anchors)}")

    # Build CSP variables from expanded grid
    variables = build_csp_variables(expanded_cv)
    cell_frequencies = {cell_key: info.get('frequency', 0) for cell_key, info in expanded_cv.items()}

    # Initialise domains
    variables = initialise_domains(
        variables, inventory, cell_frequencies, anchors, frequency_slack=4,
    )

    domain_avg = sum(len(v.domain) for v in variables) / len(variables) if variables else 0
    print(f"\n  Variables: {len(variables)}, avg domain: {domain_avg:.1f}")

    # AC-3
    solvable, variables = ac3_propagate(variables)
    if not solvable:
        print("  [WARN] AC-3 found unsolvable state — proceeding anyway")

    # Run beam search
    print("\n  Running beam search on expanded grid...")
    t_beam = time.time()
    solutions = beam_search(
        variables=variables,
        lm=lm,
        voynich_tokens=tokens,
        eva_to_cell=eva_to_subcell,
        anchors=anchors,
        inventory=inventory,
        ref_word_set=ref_word_set,
        beam_width=80,
        max_solutions=20,
    )
    elapsed = time.time() - t_beam
    print(f"  Beam search: {elapsed:.1f}s")

    if not solutions:
        print("  [ERROR] No solutions found")
        return

    best = solutions[0]

    # Selectivity vs random
    import random as _random
    rng = _random.Random(42)
    all_syls = list(inventory.cv_syllables)
    random_hits: List[float] = []
    for _ in range(50):
        rand_map = {v.cell_key: rng.choice(all_syls) for v in variables}
        decoded_r = decode_corpus(tokens, rand_map, eva_to_subcell, max_tokens=500)
        hits = sum(1 for w in decoded_r if w in ref_word_set)
        random_hits.append(hits / len(decoded_r) if decoded_r else 0.0)
    random_baseline = sum(random_hits) / len(random_hits) if random_hits else 0.001
    selectivity = best.dict_hit_rate / max(random_baseline, 0.001)

    # Comparison
    phase11_baseline = 0.111
    if best.dict_hit_rate > feature_dict_hit and best.dict_hit_rate > phase11_baseline:
        better = 'subcell'
    elif feature_dict_hit > best.dict_hit_rate and feature_dict_hit > phase11_baseline:
        better = 'feature'
    elif abs(best.dict_hit_rate - feature_dict_hit) < 0.005:
        better = 'tie'
    else:
        better = 'neither'

    gate_passed = best.dict_hit_rate > phase11_baseline and selectivity >= 1.5

    if better == 'subcell':
        verdict = (
            f"SUBCELL WINS: {best.dict_hit_rate:.1%} vs feature {feature_dict_hit:.1%}. "
            f"Data-driven splitting outperforms stroke-feature model. "
            f"Distributional clusters capture the phonemic distinctions better."
        )
    elif better == 'feature':
        verdict = (
            f"FEATURE WINS: {feature_dict_hit:.1%} vs subcell {best.dict_hit_rate:.1%}. "
            f"Stroke-feature model outperforms data-driven splitting. "
            f"The articulatory hypothesis for stroke types is confirmed."
        )
    elif better == 'tie':
        verdict = (
            f"TIE: Both approaches achieve ~{best.dict_hit_rate:.1%}. "
            f"Feature and data-driven models converge — consistent evidence "
            f"that cell conflation was the bottleneck."
        )
    else:
        verdict = (
            f"Neither approach improves on Phase 11 baseline ({phase11_baseline:.1%}). "
            f"The 11.1%% ceiling is NOT caused by cell-level conflation. "
            f"A deeper structural mismatch (wrong language, encoding model, etc.) limits further progress."
        )

    print(f"\n  ── Sub-Cell Split Summary ──")
    print(f"  Expanded cells: {n_expanded} (from {n_original})")
    print(f"  Best dict_hit:  {best.dict_hit_rate:.3f}  (selectivity: {selectivity:.2f}x)")
    print(f"  Feature CSP:    {feature_dict_hit:.3f}")
    print(f"  Phase 11:       {phase11_baseline:.3f}")
    print(f"  Better approach: {better.upper()}")
    print(f"  Gate: {'PASS' if gate_passed else 'FAIL'}")
    print(f"  Verdict: {verdict}")

    result = SubcellSplitResult(
        n_original_cells=n_original,
        n_subcells=n_expanded,
        splits_applied=[_convert(s) for s in splits],
        feature_csp_dict_hit=feature_dict_hit,
        subcell_csp_dict_hit=best.dict_hit_rate,
        subcell_csp_cross_entropy=best.cross_entropy,
        subcell_csp_selectivity=selectivity,
        best_assignment=best.mapping,
        better_approach=better,
        decoded_sample=best.decoded_sample,
        gate_passed=gate_passed,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'subcell_split.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)
    print(f"\n  Results saved → {out_path}")
