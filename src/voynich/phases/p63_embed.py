"""Phase 63 Step A3: Embed all normalized images via Gemini Embedding 2.

Idempotent: saves progress after each batch (EVA, Costamagna visual,
Costamagna multimodal) so interrupted runs resume from the last
completed batch.
"""

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

import numpy as np

from voynich.core._paths import results_dir

MODEL = "gemini-embedding-2-preview"
OUTPUT_DIM = 768


@dataclass
class EmbedResult:
    n_eva_embedded: int = 0
    n_costa_visual_embedded: int = 0
    n_costa_multimodal_embedded: int = 0
    embedding_dim: int = 0
    model_name: str = ''
    total_api_calls: int = 0
    failed: List[str] = field(default_factory=list)
    resumed: bool = False
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


def _load_partial(npz_path):
    """Load partial progress if it exists. Returns dict of arrays or empty."""
    if os.path.exists(npz_path):
        return dict(np.load(npz_path, allow_pickle=True))
    return {}


def _save_progress(npz_path, **arrays):
    """Save current progress to .npz."""
    np.savez(npz_path, **arrays)


def run_p63_embed(force=False):
    """Embed all normalized images using Gemini Embedding 2.

    Resumes from partial progress: if the .npz already contains EVA
    embeddings, skips directly to Costamagna, etc.
    """
    t0 = time.time()
    rd = str(results_dir())
    npz_path = os.path.join(rd, 'p63_embeddings.npz')

    from voynich.visual.embed import _get_client, embed_batch

    # Load normalized metadata
    norm_dir = os.path.join(rd, 'p63_normalized')
    eva_meta = _safe_load(os.path.join(norm_dir, 'eva_metadata.json'))
    costa_meta = _safe_load(os.path.join(norm_dir, 'costamagna_metadata.json'))

    if not eva_meta:
        print("ERROR: EVA metadata not found. Run vis-normalize first.")
        return
    if not costa_meta:
        print("ERROR: Costamagna metadata not found. Run vis-normalize first.")
        return

    # Check existing progress
    partial = _load_partial(npz_path)
    resumed = bool(partial)

    has_eva = 'eva_embeddings' in partial and partial['eva_embeddings'].ndim == 2 and partial['eva_embeddings'].shape[0] > 0
    has_costa = 'costa_embeddings' in partial and partial['costa_embeddings'].ndim == 2 and partial['costa_embeddings'].shape[0] > 0
    has_costa_mm = 'costa_mm_embeddings' in partial and partial['costa_mm_embeddings'].ndim == 2 and partial['costa_mm_embeddings'].shape[0] > 0

    # If all three complete and not forcing, just report
    if has_eva and has_costa and has_costa_mm and not force:
        n_eva = len(partial['eva_names'])
        n_costa = len(partial['costa_names'])
        n_mm = len(partial['costa_mm_names'])
        print(f"Phase 63 A3: All embeddings cached ({n_eva} EVA, {n_costa} Costamagna visual, {n_mm} multimodal).")
        print(f"  Use force=True to re-embed.")
        dim = int(partial['eva_embeddings'].shape[1])
        result = EmbedResult(
            n_eva_embedded=n_eva, n_costa_visual_embedded=n_costa,
            n_costa_multimodal_embedded=n_mm, embedding_dim=dim,
            model_name=MODEL, resumed=True, elapsed=time.time() - t0,
        )
        _save_json(rd, 'p63_embed.json', asdict(result))
        return

    client = _get_client()
    all_failed = []
    api_calls = 0

    print(f"Phase 63 A3: Embedding images via {MODEL}...")
    if resumed:
        print(f"  Resuming from partial progress (EVA={'done' if has_eva else 'pending'}, "
              f"costa={'done' if has_costa else 'pending'}, mm={'done' if has_costa_mm else 'pending'})")

    # --- Batch 1: EVA image-only ---
    if has_eva and not force:
        eva_names = list(partial['eva_names'])
        eva_emb = partial['eva_embeddings']
        print(f"\n  EVA: using cached {len(eva_names)} embeddings")
    else:
        print(f"\n  Embedding {len(eva_meta)} EVA characters (image-only)...")
        eva_items = [{'name': m['eva_name'], 'image_path': m['normalized_path']}
                     for m in eva_meta]
        eva_names, eva_emb, eva_failed = embed_batch(
            client, eva_items, mode='image_only', model=MODEL, output_dim=OUTPUT_DIM)
        all_failed.extend(eva_failed)
        api_calls += len(eva_items)

        # Save progress immediately
        _save_progress(npz_path,
                       eva_names=np.array(eva_names),
                       eva_embeddings=eva_emb)
        print(f"  EVA done: {len(eva_names)} embedded, {len(eva_failed)} failed. Progress saved.")

    # --- Batch 2: Costamagna image-only ---
    if has_costa and not force:
        costa_names = list(partial['costa_names'])
        costa_emb = partial['costa_embeddings']
        print(f"  Costamagna visual: using cached {len(costa_names)} embeddings")
    else:
        print(f"  Embedding {len(costa_meta)} Costamagna signs (image-only)...")
        costa_items = [{'name': m['syllable'], 'image_path': m['image_path']}
                       for m in costa_meta]
        costa_names, costa_emb, costa_failed = embed_batch(
            client, costa_items, mode='image_only', model=MODEL, output_dim=OUTPUT_DIM)
        all_failed.extend(costa_failed)
        api_calls += len(costa_items)

        # Save progress
        _save_progress(npz_path,
                       eva_names=np.array(eva_names),
                       eva_embeddings=eva_emb,
                       costa_names=np.array(costa_names),
                       costa_embeddings=costa_emb)
        print(f"  Costamagna visual done: {len(costa_names)} embedded, {len(costa_failed)} failed. Progress saved.")

    # --- Batch 3: Costamagna image+text ---
    if has_costa_mm and not force:
        costa_mm_names = list(partial['costa_mm_names'])
        costa_mm_emb = partial['costa_mm_embeddings']
        print(f"  Costamagna multimodal: using cached {len(costa_mm_names)} embeddings")
    else:
        print(f"  Embedding {len(costa_meta)} Costamagna signs (image+text)...")
        costa_mm_items = [{'name': m['syllable'], 'image_path': m['image_path'],
                           'text_label': m['syllable']}
                          for m in costa_meta]
        costa_mm_names, costa_mm_emb, costa_mm_failed = embed_batch(
            client, costa_mm_items, mode='image_text', model=MODEL, output_dim=OUTPUT_DIM)
        all_failed.extend(costa_mm_failed)
        api_calls += len(costa_mm_items)

        # Save final
        _save_progress(npz_path,
                       eva_names=np.array(eva_names),
                       eva_embeddings=eva_emb,
                       costa_names=np.array(costa_names),
                       costa_embeddings=costa_emb,
                       costa_mm_names=np.array(costa_mm_names),
                       costa_mm_embeddings=costa_mm_emb)
        print(f"  Costamagna multimodal done: {len(costa_mm_names)} embedded, {len(costa_mm_failed)} failed. Progress saved.")

    result = EmbedResult(
        n_eva_embedded=len(eva_names),
        n_costa_visual_embedded=len(costa_names),
        n_costa_multimodal_embedded=len(costa_mm_names),
        embedding_dim=OUTPUT_DIM,
        model_name=MODEL,
        total_api_calls=api_calls,
        failed=all_failed,
        resumed=resumed,
        elapsed=time.time() - t0,
    )

    _save_json(rd, 'p63_embed.json', asdict(result))

    print(f"\n  Summary:")
    print(f"    EVA: {result.n_eva_embedded}")
    print(f"    Costamagna visual: {result.n_costa_visual_embedded}")
    print(f"    Costamagna multimodal: {result.n_costa_multimodal_embedded}")
    print(f"    API calls this run: {api_calls}")
    print(f"    Failed: {len(all_failed)}")
    print(f"    Elapsed: {result.elapsed:.1f}s")
