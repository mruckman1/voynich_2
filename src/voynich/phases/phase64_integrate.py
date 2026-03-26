"""Phase 64: Multi-Method Visual Sign Comparison.

Seven independent methods (2 LLM, 5 CV) compare EVA characters against
Costamagna tachygraphic signs. A rank-fusion ensemble combines them.
Replaces Phase 63's single-embedding approach which collapsed all signs
into a narrow similarity band.
"""

import asyncio
import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

import numpy as np

from voynich.core._paths import results_dir


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _convert(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _convert(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_convert(item) for item in obj]
    if isinstance(obj, float) and (obj != obj):
        return None
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, set):
        return sorted(obj)
    return obj


def _save_json(rd: str, filename: str, data: Any) -> str:
    path = os.path.join(rd, filename)
    with open(path, 'w') as f:
        json.dump(_convert(data), f, indent=2)
    return path


def _safe_load(path: str) -> Any:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _load_image_items(metadata_path, image_dir, name_key='eva_name',
                      name_fallback='syllable'):
    """Load metadata and build list of {'name': ..., 'image_path': ...}."""
    if not os.path.exists(metadata_path):
        return []
    with open(metadata_path) as f:
        metadata = json.load(f)

    items = []
    for entry in metadata:
        name = entry.get(name_key) or entry.get(name_fallback, '')
        img_path = entry.get('image_path', '')
        if not os.path.isabs(img_path):
            img_path = os.path.join(image_dir, os.path.basename(img_path))
        if os.path.exists(img_path):
            items.append({'name': name, 'image_path': img_path,
                          **{k: v for k, v in entry.items()
                             if k not in ('image_path',)}})
    return items


def _load_assets():
    """Load EVA and Costamagna image items.

    Tries Phase 63 normalized output first, falls back to raw crops.
    """
    rd = str(results_dir())

    # Try Phase 63 normalized metadata
    p63_norm = os.path.join(rd, 'p63_normalize.json')
    if os.path.exists(p63_norm):
        norm_data = _safe_load(p63_norm)
        norm_dir = os.path.join(rd, 'p63_normalized')

        eva_items = []
        for entry in norm_data.get('eva_metadata', []):
            path = entry.get('normalized_path', entry.get('image_path', ''))
            name = entry.get('eva_name', '')
            if os.path.exists(path):
                eva_items.append({'name': name, 'image_path': path, **entry})

        costa_items = []
        for entry in norm_data.get('costamagna_metadata', []):
            path = entry.get('normalized_path', entry.get('image_path', ''))
            name = entry.get('syllable', '')
            if os.path.exists(path):
                costa_items.append({'name': name, 'image_path': path, **entry})

        if eva_items and costa_items:
            return eva_items, costa_items

    # Fallback: generate EVA renders on the fly + load raw Costamagna crops
    from voynich.core._paths import data_dir
    dd = str(data_dir())

    # EVA renders — generate if not present
    eva_render_dir = os.path.join(dd, 'eva_renders')
    eva_meta_path = os.path.join(eva_render_dir, 'metadata.json')
    if not os.path.exists(eva_meta_path):
        # Try to render from font
        font_path = os.path.join(dd, 'Voynich EVA Hand A.ttf')
        if os.path.exists(font_path):
            from voynich.visual.render_eva import render_all_eva
            print("  Rendering EVA characters from font...")
            eva_meta = render_all_eva(font_path, eva_render_dir)
            with open(eva_meta_path, 'w') as f:
                json.dump(eva_meta, f, indent=2)

    eva_items = _load_image_items(eva_meta_path, eva_render_dir,
                                  name_key='eva_name')

    # Costamagna crops
    costa_crops_dir = os.path.join(dd, 'GL.S.III.MISC.12',
                                   'costamagna_crops')
    costa_meta_path = os.path.join(costa_crops_dir, 'metadata.json')
    if os.path.exists(costa_meta_path):
        with open(costa_meta_path) as f:
            costa_meta = json.load(f)
        costa_items = []
        for entry in costa_meta:
            name = entry.get('syllable', '')
            # metadata uses 'image_symbol' as relative path
            img_rel = entry.get('image_symbol',
                                entry.get('image_path', ''))
            img_path = os.path.join(costa_crops_dir, img_rel)
            if not os.path.exists(img_path):
                img_path = os.path.join(costa_crops_dir, 'symbols',
                                        f'{name}.png')
            if os.path.exists(img_path):
                costa_items.append({'name': name, 'image_path': img_path,
                                    **entry})
    else:
        costa_items = []

    return eva_items, costa_items


# ---------------------------------------------------------------------------
# Individual step runners
# ---------------------------------------------------------------------------

def run_p64_stroke_extract():
    """Method 2: Extract skeleton graph features and build distance matrix."""
    t0 = time.time()
    rd = str(results_dir())
    eva_items, costa_items = _load_assets()
    print(f"Phase 64 Method 2: Stroke Extraction")
    print(f"  EVA: {len(eva_items)}, Costamagna: {len(costa_items)}")

    from voynich.visual.stroke_extraction import (
        build_graph_distance_matrix,
        extract_graph_features,
        extract_skeleton,
    )

    def extract(item):
        skeleton, _ = extract_skeleton(item['image_path'])
        return extract_graph_features(skeleton)

    print("  Extracting EVA graph features...")
    eva_feats = [extract(item) for item in eva_items]
    print("  Extracting Costamagna graph features...")
    costa_feats = [extract(item) for item in costa_items]

    matrix = build_graph_distance_matrix(eva_feats, costa_feats)

    # Save
    eva_names = [item['name'] for item in eva_items]
    costa_names = [item['name'] for item in costa_items]
    np.savez(os.path.join(rd, 'p64_graph_matrix.npz'),
             matrix=matrix, eva_names=eva_names, costa_names=costa_names)

    feat_data = {
        'eva': [{'name': eva_items[i]['name'],
                 'features': _convert(eva_feats[i])}
                for i in range(len(eva_items))],
        'costa': [{'name': costa_items[i]['name'],
                   'features': _convert(costa_feats[i])}
                  for i in range(len(costa_items))],
    }
    _save_json(rd, 'p64_graph_features.json', feat_data)

    elapsed = time.time() - t0
    print(f"  Done ({elapsed:.1f}s). Matrix shape: {matrix.shape}")
    print(f"  Distance range: [{matrix.min():.3f}, {matrix.max():.3f}]")


def run_p64_shape_desc():
    """Method 3: Classical shape descriptors."""
    t0 = time.time()
    rd = str(results_dir())
    eva_items, costa_items = _load_assets()
    print(f"Phase 64 Method 3: Shape Descriptors")
    print(f"  EVA: {len(eva_items)}, Costamagna: {len(costa_items)}")

    from voynich.visual.shape_descriptors import (
        build_shape_distance_matrix,
        compute_shape_feature_vector,
    )

    print("  Computing EVA shape features...")
    eva_feats = [compute_shape_feature_vector(item['image_path'])
                 for item in eva_items]
    print("  Computing Costamagna shape features...")
    costa_feats = [compute_shape_feature_vector(item['image_path'])
                   for item in costa_items]

    matrix = build_shape_distance_matrix(eva_feats, costa_feats)

    eva_names = [item['name'] for item in eva_items]
    costa_names = [item['name'] for item in costa_items]
    np.savez(os.path.join(rd, 'p64_shape_matrix.npz'),
             matrix=matrix, eva_names=eva_names, costa_names=costa_names)

    elapsed = time.time() - t0
    print(f"  Done ({elapsed:.1f}s). Matrix shape: {matrix.shape}")
    print(f"  Distance range: [{matrix.min():.3f}, {matrix.max():.3f}]")


def run_p64_topo_features():
    """Method 4: Topological features."""
    t0 = time.time()
    rd = str(results_dir())
    eva_items, costa_items = _load_assets()
    print(f"Phase 64 Method 4: Topological Features")
    print(f"  EVA: {len(eva_items)}, Costamagna: {len(costa_items)}")

    from voynich.visual.topological_features import (
        build_topo_distance_matrix,
        compute_topological_features,
    )

    print("  Computing EVA topological features...")
    eva_topos = [compute_topological_features(item['image_path'])
                 for item in eva_items]
    print("  Computing Costamagna topological features...")
    costa_topos = [compute_topological_features(item['image_path'])
                   for item in costa_items]

    matrix = build_topo_distance_matrix(eva_topos, costa_topos)

    eva_names = [item['name'] for item in eva_items]
    costa_names = [item['name'] for item in costa_items]
    np.savez(os.path.join(rd, 'p64_topo_matrix.npz'),
             matrix=matrix, eva_names=eva_names, costa_names=costa_names)

    feat_data = {
        'eva': [{'name': eva_items[i]['name'], 'features': eva_topos[i]}
                for i in range(len(eva_items))],
        'costa': [{'name': costa_items[i]['name'], 'features': costa_topos[i]}
                  for i in range(len(costa_items))],
    }
    _save_json(rd, 'p64_topo_features.json', feat_data)

    elapsed = time.time() - t0
    print(f"  Done ({elapsed:.1f}s). Matrix shape: {matrix.shape}")
    print(f"  Distance range: [{matrix.min():.3f}, {matrix.max():.3f}]")


def run_p64_hog_compare():
    """Method 5: HOG features."""
    t0 = time.time()
    rd = str(results_dir())
    eva_items, costa_items = _load_assets()
    print(f"Phase 64 Method 5: HOG Features")
    print(f"  EVA: {len(eva_items)}, Costamagna: {len(costa_items)}")

    from voynich.visual.hog_features import (
        build_hog_distance_matrix,
        compute_hog_features,
    )

    print("  Computing EVA HOG features...")
    eva_hog = [compute_hog_features(item['image_path'])
               for item in eva_items]
    print("  Computing Costamagna HOG features...")
    costa_hog = [compute_hog_features(item['image_path'])
                 for item in costa_items]

    matrix = build_hog_distance_matrix(eva_hog, costa_hog)

    eva_names = [item['name'] for item in eva_items]
    costa_names = [item['name'] for item in costa_items]
    np.savez(os.path.join(rd, 'p64_hog_matrix.npz'),
             matrix=matrix, eva_names=eva_names, costa_names=costa_names)

    elapsed = time.time() - t0
    print(f"  Done ({elapsed:.1f}s). Matrix shape: {matrix.shape}")
    print(f"  Distance range: [{matrix.min():.3f}, {matrix.max():.3f}]")


def run_p64_hybrid():
    """Method 6: Hybrid combined features."""
    t0 = time.time()
    rd = str(results_dir())
    eva_items, costa_items = _load_assets()
    print(f"Phase 64 Method 6: Hybrid Features")
    print(f"  EVA: {len(eva_items)}, Costamagna: {len(costa_items)}")

    from voynich.visual.hybrid_features import (
        build_hybrid_distance_matrix,
        compute_hybrid_vector,
    )

    print("  Computing EVA hybrid features...")
    eva_hybrids = [compute_hybrid_vector(item['image_path'])
                   for item in eva_items]
    print("  Computing Costamagna hybrid features...")
    costa_hybrids = [compute_hybrid_vector(item['image_path'])
                     for item in costa_items]

    matrix = build_hybrid_distance_matrix(eva_hybrids, costa_hybrids)

    eva_names = [item['name'] for item in eva_items]
    costa_names = [item['name'] for item in costa_items]
    np.savez(os.path.join(rd, 'p64_hybrid_matrix.npz'),
             matrix=matrix, eva_names=eva_names, costa_names=costa_names)

    elapsed = time.time() - t0
    print(f"  Done ({elapsed:.1f}s). Matrix shape: {matrix.shape}")
    print(f"  Distance range: [{matrix.min():.3f}, {matrix.max():.3f}]")


def run_p64_morph_describe():
    """Method 1: LLM morphology descriptions."""
    t0 = time.time()
    rd = str(results_dir())
    eva_items, costa_items = _load_assets()
    print(f"Phase 64 Method 1: Morphology Description (LLM)")
    print(f"  EVA: {len(eva_items)}, Costamagna: {len(costa_items)}")
    print(f"  API calls: ~{len(eva_items) + len(costa_items)}")

    from voynich.visual.morphology_description import (
        _get_openrouter_client,
        build_morphology_matrix,
        describe_all_signs,
    )

    client = _get_openrouter_client()
    eva_morphs, costa_morphs = asyncio.run(
        describe_all_signs(client, eva_items, costa_items)
    )

    _save_json(rd, 'p64_morphology_eva.json', eva_morphs)
    _save_json(rd, 'p64_morphology_costa.json', costa_morphs)

    matrix = build_morphology_matrix(eva_morphs, costa_morphs)

    eva_names = [m.get('sign_id', '').replace('eva_', '')
                 for m in eva_morphs]
    costa_names = [m.get('sign_id', '').replace('costa_', '')
                   for m in costa_morphs]
    np.savez(os.path.join(rd, 'p64_morphology_matrix.npz'),
             matrix=matrix, eva_names=eva_names, costa_names=costa_names)

    elapsed = time.time() - t0
    print(f"  Done ({elapsed:.1f}s). EVA described: {len(eva_morphs)}, "
          f"Costa described: {len(costa_morphs)}")
    print(f"  Matrix shape: {matrix.shape}")
    print(f"  Distance range: [{matrix.min():.3f}, {matrix.max():.3f}]")


def run_p64_llm_pairwise():
    """Method 7: LLM pairwise comparison."""
    t0 = time.time()
    rd = str(results_dir())
    eva_items, costa_items = _load_assets()

    from voynich.visual.llm_pairwise import (
        _get_openrouter_client,
        run_t_p15_pairwise,
        score_pairwise_results,
    )
    from voynich.visual.render_eva import T_P15

    print(f"Phase 64 Method 7: LLM Pairwise Comparison")
    print(f"  T_P15 assignments: {len(T_P15)}")

    client = _get_openrouter_client()
    comparisons = asyncio.run(
        run_t_p15_pairwise(client, eva_items, costa_items, T_P15)
    )

    scores = score_pairwise_results(comparisons)

    _save_json(rd, 'p64_pairwise.json', {
        'comparisons': comparisons,
        'scores': scores,
    })

    elapsed = time.time() - t0
    print(f"  Done ({elapsed:.1f}s). Comparisons: {len(comparisons)}")
    print(f"  Win rate: {scores['win_rate']:.1%}")
    print(f"  Mean proposed sim: {scores['mean_proposed_sim']:.3f}")
    print(f"  Mean control sim: {scores['mean_control_sim']:.3f}")


def run_p64_ensemble():
    """Tier 3: Ensemble combination + validation."""
    t0 = time.time()
    rd = str(results_dir())

    from voynich.visual.ensemble import (
        build_ensemble,
        per_method_diagnostics,
        permutation_test_ensemble,
        validate_ensemble,
    )
    from voynich.visual.render_eva import T_P15

    # Load all available distance matrices
    method_names = []
    distance_matrices = []
    eva_names = None
    costa_names = None

    matrix_files = [
        ('p64_graph_matrix.npz', 'M2_Graph'),
        ('p64_shape_matrix.npz', 'M3_Shape'),
        ('p64_topo_matrix.npz', 'M4_Topology'),
        ('p64_hog_matrix.npz', 'M5_HOG'),
        ('p64_hybrid_matrix.npz', 'M6_Hybrid'),
        ('p64_morphology_matrix.npz', 'M1_Morphology'),
    ]

    for fname, mname in matrix_files:
        path = os.path.join(rd, fname)
        if os.path.exists(path):
            data = np.load(path, allow_pickle=True)
            distance_matrices.append(data['matrix'])
            method_names.append(mname)
            if eva_names is None:
                eva_names = list(data['eva_names'])
                costa_names = list(data['costa_names'])

    print(f"Phase 64 Ensemble: {len(distance_matrices)} methods loaded")
    if not distance_matrices:
        print("  ERROR: No distance matrices found. Run individual methods first.")
        return

    # Build ensemble
    ensemble_ranks = build_ensemble(distance_matrices, method_names)

    # Validate
    validation = validate_ensemble(ensemble_ranks, eva_names, costa_names,
                                   T_P15)
    print(f"  T_P15 validation: {validation['n_strong']} STRONG, "
          f"{validation['n_moderate']} MODERATE, "
          f"{validation['n_weak']} WEAK, {validation['n_none']} NONE")
    print(f"  Mean rank: {validation['mean_rank']:.1f}")

    # Permutation test
    perm = permutation_test_ensemble(ensemble_ranks, eva_names, costa_names,
                                     T_P15)
    print(f"  Permutation test: z={perm['z']:.2f}, p={perm['p']:.4f}")

    # Per-method diagnostics
    diagnostics = per_method_diagnostics(distance_matrices, method_names,
                                         eva_names, costa_names, T_P15)
    for mname, diag in diagnostics.items():
        print(f"  {mname}: mean_rank={diag['mean_t_p15_rank']:.1f}, "
              f"strong={diag['n_strong']}, spread={diag['similarity_spread']:.3f}")

    # Save
    np.savez(os.path.join(rd, 'p64_ensemble_matrix.npz'),
             matrix=ensemble_ranks, eva_names=eva_names,
             costa_names=costa_names)
    _save_json(rd, 'p64_diagnostics.json', diagnostics)
    _save_json(rd, 'p64_validation.json', {
        'validation': validation,
        'permutation': perm,
    })

    elapsed = time.time() - t0
    print(f"  Done ({elapsed:.1f}s)")


# ---------------------------------------------------------------------------
# Gates and verdict
# ---------------------------------------------------------------------------

@dataclass
class Phase64Result:
    n_methods: int = 0
    method_names: List[str] = field(default_factory=list)
    # Ensemble validation
    n_strong: int = 0
    n_moderate: int = 0
    n_weak: int = 0
    n_none: int = 0
    mean_rank: float = 0.0
    perm_z: float = 0.0
    perm_p: float = 1.0
    # LLM pairwise
    pairwise_win_rate: float = 0.0
    pairwise_n_structure: int = 0
    # Per-method
    n_methods_with_spread: int = 0
    # Topology filter
    n_topo_compatible: int = 0
    # Gates
    gates: List[Dict] = field(default_factory=list)
    gates_passed: int = 0
    gates_total: int = 7
    verdict: str = ''
    elapsed: float = 0.0


def run_phase64_verdict():
    """Evaluate gates and produce Phase 64 verdict."""
    t0 = time.time()
    rd = str(results_dir())

    from voynich.visual.render_eva import T_P15

    result = Phase64Result()

    # Load validation results
    val_data = _safe_load(os.path.join(rd, 'p64_validation.json'))
    validation = val_data.get('validation', {})
    perm = val_data.get('permutation', {})

    result.n_strong = validation.get('n_strong', 0)
    result.n_moderate = validation.get('n_moderate', 0)
    result.n_weak = validation.get('n_weak', 0)
    result.n_none = validation.get('n_none', 0)
    result.mean_rank = validation.get('mean_rank', 0)
    result.perm_z = perm.get('z', 0)
    result.perm_p = perm.get('p', 1)

    # Load pairwise scores
    pw_data = _safe_load(os.path.join(rd, 'p64_pairwise.json'))
    pw_scores = pw_data.get('scores', {})
    result.pairwise_win_rate = pw_scores.get('win_rate', 0)
    result.pairwise_n_structure = pw_scores.get('n_same_structure', 0)

    # Load diagnostics
    diag = _safe_load(os.path.join(rd, 'p64_diagnostics.json'))
    result.n_methods = len(diag)
    result.method_names = list(diag.keys())
    result.n_methods_with_spread = sum(
        1 for d in diag.values() if d.get('similarity_spread', 0) > 0.3
    )

    # Topology compatibility
    topo_data = _safe_load(os.path.join(rd, 'p64_topo_features.json'))
    if topo_data:
        eva_topos = {e['name']: e['features']
                     for e in topo_data.get('eva', [])}
        costa_topos = {c['name']: c['features']
                       for c in topo_data.get('costa', [])}
        n_compat = 0
        for eva_name, proposed in T_P15.items():
            if eva_name not in eva_topos:
                continue
            et = eva_topos[eva_name]
            # Find Costamagna match
            for cname, ct in costa_topos.items():
                if cname == proposed or proposed in cname.split('-'):
                    # Compatible if holes and endpoints within 1
                    if (abs(et.get('n_holes', 0) - ct.get('n_holes', 0)) <= 1
                            and abs(et.get('n_endpoints', 0)
                                    - ct.get('n_endpoints', 0)) <= 2):
                        n_compat += 1
                    break
        result.n_topo_compatible = n_compat

    # Evaluate gates
    n_tested = validation.get('n_tested', 25)
    gates = [
        {'id': 'G1', 'name': 'Ensemble top-5',
         'threshold': '≥5/25 in top-5',
         'value': result.n_strong,
         'passed': result.n_strong >= 5},
        {'id': 'G2', 'name': 'Ensemble top-15',
         'threshold': '≥12/25 in top-15',
         'value': result.n_strong + result.n_moderate,
         'passed': (result.n_strong + result.n_moderate) >= 12},
        {'id': 'G3', 'name': 'LLM pairwise win rate',
         'threshold': '>50%',
         'value': result.pairwise_win_rate,
         'passed': result.pairwise_win_rate > 0.5},
        {'id': 'G4', 'name': 'Method discrimination',
         'threshold': '≥2 methods with spread >0.3',
         'value': result.n_methods_with_spread,
         'passed': result.n_methods_with_spread >= 2},
        {'id': 'G5', 'name': 'Permutation test',
         'threshold': 'p<0.05',
         'value': result.perm_p,
         'passed': result.perm_p < 0.05},
        {'id': 'G6', 'name': 'LLM structural match',
         'threshold': '≥10/25 same_basic_structure',
         'value': result.pairwise_n_structure,
         'passed': result.pairwise_n_structure >= 10},
        {'id': 'G7', 'name': 'Topology compatible',
         'threshold': '≥15/25 pass topology',
         'value': result.n_topo_compatible,
         'passed': result.n_topo_compatible >= 15},
    ]

    result.gates = gates
    result.gates_passed = sum(1 for g in gates if g['passed'])

    if result.gates_passed >= 5:
        result.verdict = 'VISUAL_SUPPORT'
    elif result.gates_passed >= 3:
        result.verdict = 'PARTIAL_SUPPORT'
    elif result.gates_passed >= 1:
        result.verdict = 'WEAK_SUPPORT'
    else:
        result.verdict = 'NO_SUPPORT'

    result.elapsed = time.time() - t0

    _save_json(rd, 'phase64.json', asdict(result))

    print(f"\nPhase 64 Verdict: {result.verdict} ({result.gates_passed}/{result.gates_total} gates)")
    for g in gates:
        status = 'PASS' if g['passed'] else 'FAIL'
        print(f"  {g['id']} {g['name']}: {g['value']} [{g['threshold']}] → {status}")

    return result


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_phase64():
    """Run the complete Phase 64 pipeline."""
    print("=" * 60)
    print("Phase 64: Multi-Method Visual Sign Comparison")
    print("=" * 60)
    t0 = time.time()

    eva_items, costa_items = _load_assets()
    if not eva_items or not costa_items:
        print("ERROR: Cannot load image assets.")
        print("  Run 'voynich vis-render' and 'voynich vis-normalize' first,")
        print("  or ensure EVA renders and Costamagna crops are in place.")
        return

    print(f"\nAssets: {len(eva_items)} EVA, {len(costa_items)} Costamagna")

    # Step 1: CV methods
    print("\n--- CV Methods (no API) ---")
    run_p64_stroke_extract()
    run_p64_shape_desc()
    run_p64_topo_features()
    run_p64_hog_compare()
    run_p64_hybrid()

    # Step 2: LLM methods
    print("\n--- LLM Methods (API required) ---")
    try:
        run_p64_morph_describe()
    except Exception as e:
        print(f"  Method 1 (Morphology) SKIPPED: {e}")

    try:
        run_p64_llm_pairwise()
    except Exception as e:
        print(f"  Method 7 (Pairwise) SKIPPED: {e}")

    # Step 3: Ensemble
    print("\n--- Ensemble ---")
    run_p64_ensemble()

    # Step 4: Verdict
    print("\n--- Verdict ---")
    result = run_phase64_verdict()

    elapsed = time.time() - t0
    print(f"\nTotal elapsed: {elapsed:.1f}s")
    return result
