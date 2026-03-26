"""Phase 63 Step A4: Compute similarity matrix and rank matches."""

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

import numpy as np

from voynich.core._paths import results_dir


@dataclass
class SimilarityResult:
    n_eva: int = 0
    n_costamagna: int = 0
    mean_max_sim_visual: float = 0.0
    mean_max_sim_multimodal: float = 0.0
    std_max_sim_visual: float = 0.0
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


def run_p63_similarity():
    """Compute similarity matrices and rank matches."""
    t0 = time.time()
    rd = str(results_dir())

    from voynich.visual.similarity import (
        compute_similarity_matrix,
        rank_matches,
    )

    # Load embeddings
    npz_path = os.path.join(rd, 'p63_embeddings.npz')
    if not os.path.exists(npz_path):
        print("ERROR: Embeddings not found. Run vis-embed first.")
        return

    data = np.load(npz_path, allow_pickle=True)
    eva_names = list(data['eva_names'])
    eva_emb = data['eva_embeddings']
    costa_names = list(data['costa_names'])
    costa_emb = data['costa_embeddings']
    costa_mm_names = list(data.get('costa_mm_names', []))
    costa_mm_emb = data.get('costa_mm_embeddings', np.array([]))

    # Load Costamagna metadata for enrichment
    norm_dir = os.path.join(rd, 'p63_normalized')
    costa_meta = _safe_load(os.path.join(norm_dir, 'costamagna_metadata.json'))
    if not costa_meta:
        costa_meta = [{'syllable': n} for n in costa_names]

    print(f"Phase 63 A4: Computing similarity matrices...")
    print(f"  EVA chars: {len(eva_names)}")
    print(f"  Costamagna signs: {len(costa_names)}")

    # Visual similarity matrix
    sim_visual = compute_similarity_matrix(eva_emb, costa_emb)
    rankings_visual = rank_matches(sim_visual, eva_names, costa_names,
                                    metadata_b=costa_meta, top_k=20)

    # Multimodal similarity matrix (if available)
    rankings_mm = {}
    sim_mm = np.array([])
    if len(costa_mm_emb) > 0 and costa_mm_emb.ndim == 2:
        sim_mm = compute_similarity_matrix(eva_emb, costa_mm_emb)
        rankings_mm = rank_matches(sim_mm, eva_names, costa_mm_names,
                                    metadata_b=costa_meta, top_k=20)

    # Save similarity matrices
    save_dict = {
        'eva_names': np.array(eva_names),
        'costa_names': np.array(costa_names),
        'sim_visual': sim_visual,
    }
    if len(sim_mm) > 0:
        save_dict['sim_multimodal'] = sim_mm
    np.savez(os.path.join(rd, 'p63_similarity_matrix.npz'), **save_dict)

    # Compute stats
    max_sims_visual = sim_visual.max(axis=1)
    max_sims_mm = sim_mm.max(axis=1) if len(sim_mm) > 0 else np.array([])

    result = SimilarityResult(
        n_eva=len(eva_names),
        n_costamagna=len(costa_names),
        mean_max_sim_visual=float(max_sims_visual.mean()),
        mean_max_sim_multimodal=float(max_sims_mm.mean()) if len(max_sims_mm) > 0 else 0.0,
        std_max_sim_visual=float(max_sims_visual.std()),
        elapsed=time.time() - t0,
    )

    # Save rankings + result
    _save_json(rd, 'p63_similarity.json', {
        'result': asdict(result),
        'rankings_visual': rankings_visual,
        'rankings_multimodal': rankings_mm,
    })

    _save_json(rd, 'p63_similarity_result.json', asdict(result))

    print(f"\n  Mean max similarity (visual): {result.mean_max_sim_visual:.4f} +/- {result.std_max_sim_visual:.4f}")
    if result.mean_max_sim_multimodal > 0:
        print(f"  Mean max similarity (multimodal): {result.mean_max_sim_multimodal:.4f}")

    # Show top matches for a few key chars
    print(f"\n  Sample top matches (visual):")
    for char in ['d', 'o', 'ch', 'a', 'e']:
        if char in rankings_visual:
            best = rankings_visual[char]['best_match']
            if best:
                print(f"    EVA '{char}' -> '{best['syllable']}' (sim={best['similarity']:.3f})")

    print(f"\n  Elapsed: {result.elapsed:.1f}s")
