"""Phase 63 Step A5: Validate T_P15 assignments against visual rankings."""

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

import numpy as np

from voynich.core._paths import results_dir


@dataclass
class ValidateResult:
    # Individual sign validation
    n_tested: int = 0
    strong: int = 0
    moderate: int = 0
    weak: int = 0
    none: int = 0
    strong_fraction: float = 0.0
    moderate_plus_fraction: float = 0.0
    # Family validation
    family_z: float = 0.0
    family_p: float = 0.0
    families_cluster: bool = False
    # Permutation test
    perm_real_mean_rank: float = 0.0
    perm_null_mean_rank: float = 0.0
    perm_z: float = 0.0
    perm_p: float = 0.0
    perm_significant: bool = False
    # Confirmed syllables in top-3
    n_confirmed_in_top3: int = 0
    confirmed_in_top3: List[str] = field(default_factory=list)
    # Per-sign details
    per_sign: Dict = field(default_factory=dict)
    elapsed: float = 0.0


def _convert(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _convert(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_convert(item) for item in obj]
    if isinstance(obj, float) and (obj != obj):
        return None
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


def run_p63_validate():
    """Validate T_P15 assignments against visual similarity rankings."""
    t0 = time.time()
    rd = str(results_dir())

    from voynich.visual.render_eva import T_P15
    from voynich.visual.similarity import (
        family_cohesion,
        find_assignment_ranks,
        permutation_test_ranks,
    )

    # Load similarity data
    sim_data = _safe_load(os.path.join(rd, 'p63_similarity.json'))
    if not sim_data or 'rankings_visual' not in sim_data:
        print("ERROR: Similarity data not found. Run vis-similarity first.")
        return

    rankings = sim_data['rankings_visual']

    # Load similarity matrix for permutation test
    mat_path = os.path.join(rd, 'p63_similarity_matrix.npz')
    if not os.path.exists(mat_path):
        print("ERROR: Similarity matrix not found. Run vis-similarity first.")
        return

    mat_data = np.load(mat_path, allow_pickle=True)
    sim_matrix = mat_data['sim_visual']
    eva_names = list(mat_data['eva_names'])
    costa_names = list(mat_data['costa_names'])

    # Load embeddings for family cohesion test
    emb_path = os.path.join(rd, 'p63_embeddings.npz')
    emb_data = np.load(emb_path, allow_pickle=True)
    eva_emb = emb_data['eva_embeddings']
    eva_emb_names = list(emb_data['eva_names'])

    # Load Costamagna metadata
    norm_dir = os.path.join(rd, 'p63_normalized')
    costa_meta = _safe_load(os.path.join(norm_dir, 'costamagna_metadata.json'))

    n_costa = len(costa_names)

    print("Phase 63 A5: Validating T_P15 against visual rankings...")

    # 1. Individual sign validation
    print("\n  1. Individual sign validation...")
    per_sign = find_assignment_ranks(rankings, T_P15, n_costa)

    counts = {'STRONG': 0, 'MODERATE': 0, 'WEAK': 0, 'NONE': 0}
    for data in per_sign.values():
        counts[data['support_level']] += 1

    total = sum(counts.values())
    print(f"    STRONG (top-5):  {counts['STRONG']}")
    print(f"    MODERATE (top-15): {counts['MODERATE']}")
    print(f"    WEAK (top-50):   {counts['WEAK']}")
    print(f"    NONE:            {counts['NONE']}")

    # 2. Family-level validation
    print("\n  2. Family-level validation...")
    family_result = family_cohesion(eva_emb, eva_emb_names, T_P15)
    print(f"    Real mean cohesion: {family_result['real_mean_cohesion']:.4f}")
    print(f"    Null mean cohesion: {family_result['null_mean_cohesion']:.4f}")
    print(f"    z = {family_result['z']:.2f}, p = {family_result['p']:.4f}")
    print(f"    Families cluster: {family_result['families_cluster']}")

    # 3. Permutation test
    print("\n  3. Permutation test (1000 trials)...")
    perm_result = permutation_test_ranks(
        sim_matrix, eva_names, costa_names, T_P15)
    print(f"    Real mean rank: {perm_result['real_mean_rank']:.1f}")
    print(f"    Null mean rank: {perm_result['null_mean_rank']:.1f}")
    print(f"    z = {perm_result['z']:.2f}, p = {perm_result['p']:.4f}")
    print(f"    Significant: {perm_result['significant']}")

    # 4. Confirmed syllables in top-3
    # The 12 confirmed triples from Phase 14+19
    confirmed_syls = {'di', 'ne', 'se', 'be', 'ra', 'de', 'mi', 'ro', 'ni',
                      'ca', 'sa', 'la'}
    in_top3 = []
    for eva, data in per_sign.items():
        if data['proposed_syllable'] in confirmed_syls and data['visual_rank'] <= 3:
            in_top3.append(eva)

    print(f"\n  4. Confirmed syllables in top-3: {len(in_top3)}")
    if in_top3:
        for eva in in_top3:
            syl = per_sign[eva]['proposed_syllable']
            rank = per_sign[eva]['visual_rank']
            print(f"    EVA '{eva}' -> '{syl}' (rank #{rank})")

    # Build result
    result = ValidateResult(
        n_tested=total,
        strong=counts['STRONG'],
        moderate=counts['MODERATE'],
        weak=counts['WEAK'],
        none=counts['NONE'],
        strong_fraction=counts['STRONG'] / total if total else 0.0,
        moderate_plus_fraction=(counts['STRONG'] + counts['MODERATE']) / total if total else 0.0,
        family_z=family_result['z'],
        family_p=family_result['p'],
        families_cluster=family_result['families_cluster'],
        perm_real_mean_rank=perm_result['real_mean_rank'],
        perm_null_mean_rank=perm_result['null_mean_rank'],
        perm_z=perm_result['z'],
        perm_p=perm_result['p'],
        perm_significant=perm_result['significant'],
        n_confirmed_in_top3=len(in_top3),
        confirmed_in_top3=in_top3,
        per_sign=per_sign,
        elapsed=time.time() - t0,
    )

    _save_json(rd, 'p63_validate.json', {
        **asdict(result),
        'family_detail': family_result,
        'perm_detail': perm_result,
    })

    print(f"\n  Elapsed: {result.elapsed:.1f}s")
