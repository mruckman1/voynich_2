"""Phase 63B Step B6: Embed exemplars and compare against Costamagna."""

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

import numpy as np

from voynich.core._paths import results_dir


@dataclass
class CompareResult:
    n_char_types_embedded: int = 0
    n_exemplars_embedded: int = 0
    n_failed: int = 0
    # A-gate results with manuscript embeddings
    ms_strong: int = 0
    ms_moderate: int = 0
    ms_weak: int = 0
    ms_none: int = 0
    ms_perm_p: float = 1.0
    ms_family_z: float = 0.0
    ms_confirmed_top3: int = 0
    # Comparison with font-based results
    spearman_r: float = 0.0
    spearman_p: float = 1.0
    api_calls: int = 0
    elapsed: float = 0.0


def _convert(obj: Any) -> Any:
    if hasattr(obj, '__dataclass_fields__'):
        return {k: _convert(v) for k, v in asdict(obj).items()}
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


def run_p63b_compare():
    """Embed manuscript exemplars and compare against Costamagna signs."""
    t0 = time.time()
    rd = str(results_dir())

    from voynich.visual.embed import _get_client, embed_batch
    from voynich.visual.render_eva import T_P15
    from voynich.visual.similarity import (
        compute_similarity_matrix,
        family_cohesion,
        find_assignment_ranks,
        permutation_test_ranks,
        rank_matches,
    )

    # Load exemplar info
    exemplar_info = _safe_load(os.path.join(rd, 'p63b_exemplars.json'))
    if not exemplar_info:
        print("ERROR: Exemplar data not found. Run ms-exemplars first.")
        return

    output_dir = exemplar_info.get('output_dir', os.path.join(rd, 'p63b_exemplars'))

    # Load Costamagna embeddings from Workstream A
    costa_path = os.path.join(rd, 'p63_embeddings.npz')
    if not os.path.exists(costa_path):
        print("ERROR: Costamagna embeddings not found. Run vis-embed first.")
        return

    costa_data = np.load(costa_path, allow_pickle=True)
    costa_names = list(costa_data['costa_names'])
    costa_emb = costa_data['costa_embeddings']

    # Load Costamagna metadata
    norm_dir = os.path.join(rd, 'p63_normalized')
    costa_meta = _safe_load(os.path.join(norm_dir, 'costamagna_metadata.json'))

    print("Phase 63B B6: Embedding manuscript exemplars...")

    # Collect all exemplar images
    client = _get_client()
    all_items = []
    char_to_indices = {}  # char_name -> list of indices in all_items

    per_char = exemplar_info.get('per_char', [])
    for char_info in per_char:
        eva_char = char_info['eva_char']
        if char_info['n_exemplars'] == 0:
            continue

        safe_name = eva_char.replace('/', '_')
        char_dir = os.path.join(output_dir, safe_name)
        if not os.path.exists(char_dir):
            continue

        start_idx = len(all_items)
        for fname in sorted(os.listdir(char_dir)):
            if fname.endswith('.png'):
                all_items.append({
                    'name': f"{eva_char}_{fname}",
                    'image_path': os.path.join(char_dir, fname),
                })
        char_to_indices[eva_char] = list(range(start_idx, len(all_items)))

    print(f"  Embedding {len(all_items)} exemplar images...")
    names, embeddings, failed = embed_batch(
        client, all_items, mode='image_only',
        model="gemini-embedding-2-preview", output_dim=768)

    if len(embeddings) == 0:
        print("ERROR: No embeddings computed.")
        return

    # Compute mean embedding per character type
    eva_char_names = []
    eva_mean_embs = []

    for eva_char, indices in sorted(char_to_indices.items()):
        # Find which embeddings correspond to this char
        char_embs = []
        for idx in indices:
            item_name = all_items[idx]['name']
            if item_name in names:
                emb_idx = names.index(item_name)
                char_embs.append(embeddings[emb_idx])

        if char_embs:
            mean_emb = np.mean(char_embs, axis=0)
            norm = np.linalg.norm(mean_emb)
            if norm > 0:
                mean_emb = mean_emb / norm
            eva_char_names.append(eva_char)
            eva_mean_embs.append(mean_emb)

    eva_mean_embs = np.array(eva_mean_embs)

    print(f"  Mean embeddings computed for {len(eva_char_names)} character types")

    # Compute similarity matrix against Costamagna
    sim_matrix = compute_similarity_matrix(eva_mean_embs, costa_emb)

    # Save similarity matrix
    np.savez(os.path.join(rd, 'p63b_similarity_matrix.npz'),
             eva_names=np.array(eva_char_names),
             costa_names=np.array(costa_names),
             sim_visual=sim_matrix)

    # Rank matches
    rankings = rank_matches(sim_matrix, eva_char_names, costa_names,
                            metadata_b=costa_meta, top_k=20)

    # Validate T_P15
    n_costa = len(costa_names)
    per_sign = find_assignment_ranks(rankings, T_P15, n_costa)

    counts = {'STRONG': 0, 'MODERATE': 0, 'WEAK': 0, 'NONE': 0}
    for data in per_sign.values():
        counts[data['support_level']] += 1

    # Family cohesion
    family_result = family_cohesion(eva_mean_embs, eva_char_names, T_P15)

    # Permutation test
    perm_result = permutation_test_ranks(
        sim_matrix, eva_char_names, costa_names, T_P15)

    # Confirmed syllables in top-3
    confirmed_syls = {'di', 'ne', 'se', 'be', 'ra', 'de', 'mi', 'ro', 'ni',
                      'ca', 'sa', 'la'}
    in_top3 = []
    for eva, data in per_sign.items():
        if data['proposed_syllable'] in confirmed_syls and data['visual_rank'] <= 3:
            in_top3.append(eva)

    # Spearman correlation with font-based rankings
    font_validate = _safe_load(os.path.join(rd, 'p63_validate.json'))
    spearman_r, spearman_p = 0.0, 1.0
    if font_validate and 'per_sign' in font_validate:
        from scipy.stats import spearmanr
        font_ranks = []
        ms_ranks = []
        for eva in per_sign:
            if eva in font_validate['per_sign']:
                fr = font_validate['per_sign'][eva].get('visual_rank', n_costa)
                mr = per_sign[eva].get('visual_rank', n_costa)
                font_ranks.append(fr)
                ms_ranks.append(mr)
        if len(font_ranks) >= 3:
            corr = spearmanr(font_ranks, ms_ranks)
            spearman_r = float(corr.statistic) if hasattr(corr, 'statistic') else float(corr[0])
            spearman_p = float(corr.pvalue) if hasattr(corr, 'pvalue') else float(corr[1])

    result = CompareResult(
        n_char_types_embedded=len(eva_char_names),
        n_exemplars_embedded=len(names),
        n_failed=len(failed),
        ms_strong=counts['STRONG'],
        ms_moderate=counts['MODERATE'],
        ms_weak=counts['WEAK'],
        ms_none=counts['NONE'],
        ms_perm_p=perm_result['p'],
        ms_family_z=family_result['z'],
        ms_confirmed_top3=len(in_top3),
        spearman_r=spearman_r,
        spearman_p=spearman_p,
        api_calls=len(all_items),
        elapsed=time.time() - t0,
    )

    _save_json(rd, 'p63b_compare.json', {
        **asdict(result),
        'rankings': rankings,
        'per_sign': per_sign,
        'family_detail': family_result,
        'perm_detail': perm_result,
        'confirmed_in_top3': in_top3,
    })

    # Generate HTML report (reuse pattern from A6)
    _generate_report(rd, rankings, per_sign, result, eva_char_names, output_dir)

    print(f"\n  Results:")
    print(f"    STRONG (top-5):  {counts['STRONG']}")
    print(f"    MODERATE (top-15): {counts['MODERATE']}")
    print(f"    WEAK (top-50):   {counts['WEAK']}")
    print(f"    NONE:            {counts['NONE']}")
    print(f"    Perm p = {perm_result['p']:.4f}")
    print(f"    Family z = {family_result['z']:.2f}")
    print(f"    Confirmed in top-3: {len(in_top3)}")
    print(f"    Spearman r (vs font) = {spearman_r:.3f} (p={spearman_p:.4f})")
    print(f"  Elapsed: {result.elapsed:.1f}s")


def _generate_report(rd, rankings, per_sign, result, eva_names, exemplar_dir):
    """Generate HTML report for manuscript comparison."""
    norm_dir = os.path.join(rd, 'p63_normalized')

    lines = [
        '<!DOCTYPE html>',
        '<html><head><meta charset="utf-8"/>',
        '<title>Phase 63B: Manuscript Visual Comparison</title>',
        '<style>',
        'body { font-family: -apple-system, sans-serif; margin: 20px; background: #fafafa; }',
        '.char-row { display: flex; align-items: center; margin: 8px 0; padding: 12px; background: white; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }',
        '.eva-block { min-width: 140px; text-align: center; }',
        '.eva-img { width: 80px; height: 80px; border: 2px solid #333; border-radius: 4px; }',
        '.matches { display: flex; gap: 8px; margin-left: 20px; }',
        '.match-img { width: 60px; height: 60px; border: 1px solid #ccc; border-radius: 4px; }',
        '.info { margin-left: auto; min-width: 180px; text-align: right; font-size: 0.9em; }',
        '.STRONG { border-left: 4px solid #2ecc71; }',
        '.MODERATE { border-left: 4px solid #f39c12; }',
        '.WEAK { border-left: 4px solid #e74c3c; }',
        '.NONE { border-left: 4px solid #95a5a6; }',
        '</style></head><body>',
        '<h1>Phase 63B: Manuscript Visual Comparison</h1>',
        f'<p>Strong: {result.ms_strong}, Moderate: {result.ms_moderate}, '
        f'Weak: {result.ms_weak}, None: {result.ms_none}</p>',
        f'<p>Perm p={result.ms_perm_p:.4f}, Family z={result.ms_family_z:.2f}, '
        f'Spearman r={result.spearman_r:.3f}</p>',
    ]

    for eva_name in eva_names:
        if eva_name not in rankings:
            continue
        sign_info = per_sign.get(eva_name, {})
        support = sign_info.get('support_level', 'NONE')
        proposed = sign_info.get('proposed_syllable', '?')
        rank = sign_info.get('visual_rank', '?')

        safe_name = eva_name.replace('/', '_')
        exemplar_path = os.path.join(exemplar_dir, safe_name, 'exemplar_000.png')
        exemplar_rel = os.path.relpath(exemplar_path, rd) if os.path.exists(exemplar_path) else ''

        lines.append(f'<div class="char-row {support}">')
        lines.append(f'<div class="eva-block">')
        if exemplar_rel:
            lines.append(f'<img class="eva-img" src="{exemplar_rel}" />')
        lines.append(f'<div><b>EVA {eva_name}</b></div></div>')

        lines.append('<div class="matches">')
        for match in rankings[eva_name].get('top_matches', [])[:5]:
            syl = match['syllable']
            sim = match['similarity']
            lines.append(f'<div style="text-align:center;margin:0 5px;">')
            lines.append(f'<div><b>{syl}</b></div>')
            lines.append(f'<div style="font-size:0.8em;">{sim:.3f}</div></div>')
        lines.append('</div>')

        lines.append(f'<div class="info">T_P15: <b>{proposed}</b> (rank #{rank})<br/>Support: <b>{support}</b></div>')
        lines.append('</div>')

    lines.append('</body></html>')

    report_path = os.path.join(rd, 'p63b_visual_report.html')
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  Report: {report_path}")
