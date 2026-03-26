"""Phase 63 Step A6: Generate HTML visual comparison report."""

import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Any

from voynich.core._paths import results_dir


@dataclass
class ReportResult:
    report_path: str = ''
    n_comparisons: int = 0
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


def _build_html(rankings, validate, eva_meta, costa_meta, renders_dir, norm_dir):
    """Build the HTML report content."""
    lines = [
        '<!DOCTYPE html>',
        '<html><head><meta charset="utf-8"/>',
        '<title>Phase 63: Visual Sign Comparison</title>',
        '<style>',
        'body { font-family: -apple-system, sans-serif; margin: 20px; background: #fafafa; }',
        'h1 { color: #333; }',
        'h2 { color: #555; margin-top: 30px; }',
        '.summary { background: white; padding: 20px; border-radius: 8px; margin: 20px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }',
        '.summary td { padding: 4px 12px; }',
        '.char-row { display: flex; align-items: center; margin: 8px 0; padding: 12px; background: white; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }',
        '.eva-block { min-width: 140px; text-align: center; }',
        '.eva-img { width: 80px; height: 80px; border: 2px solid #333; border-radius: 4px; }',
        '.matches { display: flex; gap: 8px; margin-left: 20px; }',
        '.match-card { text-align: center; padding: 4px; border-radius: 4px; }',
        '.match-img { width: 60px; height: 60px; border: 1px solid #ccc; border-radius: 4px; }',
        '.info { margin-left: auto; min-width: 180px; text-align: right; font-size: 0.9em; }',
        '.STRONG { border-left: 4px solid #2ecc71; }',
        '.MODERATE { border-left: 4px solid #f39c12; }',
        '.WEAK { border-left: 4px solid #e74c3c; }',
        '.NONE { border-left: 4px solid #95a5a6; }',
        '.highlight { background: #e8f5e9; border: 2px solid #2ecc71; border-radius: 4px; }',
        '</style></head><body>',
        '<h1>Phase 63: Visual Sign Comparison Report</h1>',
    ]

    # Summary section
    if validate:
        lines.append('<div class="summary"><h2>Summary</h2><table>')
        lines.append(f'<tr><td>Signs tested:</td><td><b>{validate.get("n_tested", 0)}</b></td></tr>')
        lines.append(f'<tr><td>STRONG (top-5):</td><td><b>{validate.get("strong", 0)}</b></td></tr>')
        lines.append(f'<tr><td>MODERATE (top-15):</td><td><b>{validate.get("moderate", 0)}</b></td></tr>')
        lines.append(f'<tr><td>WEAK (top-50):</td><td><b>{validate.get("weak", 0)}</b></td></tr>')
        lines.append(f'<tr><td>NONE:</td><td><b>{validate.get("none", 0)}</b></td></tr>')
        lines.append(f'<tr><td>Permutation p:</td><td><b>{validate.get("perm_p", 1.0):.4f}</b></td></tr>')
        lines.append(f'<tr><td>Family z:</td><td><b>{validate.get("family_z", 0.0):.2f}</b></td></tr>')
        lines.append(f'<tr><td>Confirmed in top-3:</td><td><b>{validate.get("n_confirmed_in_top3", 0)}</b></td></tr>')
        lines.append('</table></div>')

    # Per-character comparison
    lines.append('<h2>Per-Character Visual Matches</h2>')

    per_sign = validate.get('per_sign', {}) if validate else {}

    # Build eva name → metadata lookup
    eva_lookup = {m.get('eva_name', ''): m for m in (eva_meta or [])}

    for eva_name, ranking_data in sorted(rankings.items()):
        sign_info = per_sign.get(eva_name, {})
        support = sign_info.get('support_level', 'NONE')
        proposed = sign_info.get('proposed_syllable', '?')
        rank = sign_info.get('visual_rank', '?')

        lines.append(f'<div class="char-row {support}">')

        # EVA image
        eva_img = os.path.join(renders_dir, f'{eva_name}.png')
        eva_rel = os.path.relpath(eva_img, os.path.dirname(norm_dir))
        lines.append(f'<div class="eva-block">')
        lines.append(f'<img class="eva-img" src="{eva_rel}" alt="EVA {eva_name}" />')
        lines.append(f'<div><b>EVA {eva_name}</b></div>')

        meta = eva_lookup.get(eva_name, {})
        role = meta.get('role', '')
        if role:
            lines.append(f'<div style="font-size:0.8em;color:#888;">{role}</div>')
        lines.append('</div>')

        # Top-5 matches
        lines.append('<div class="matches">')
        top_matches = ranking_data.get('top_matches', [])[:5]
        for match in top_matches:
            syl = match['syllable']
            sim = match['similarity']
            is_proposed = (syl == proposed or proposed in syl.split('-'))
            cls = ' highlight' if is_proposed else ''

            costa_img = os.path.join(norm_dir, 'costamagna', f'costa_{syl}.png')
            costa_rel = os.path.relpath(costa_img, os.path.dirname(norm_dir))

            lines.append(f'<div class="match-card{cls}">')
            lines.append(f'<img class="match-img" src="{costa_rel}" alt="{syl}" />')
            lines.append(f'<div><b>{syl}</b></div>')
            lines.append(f'<div style="font-size:0.8em;">{sim:.3f}</div>')
            lines.append('</div>')
        lines.append('</div>')

        # Info
        lines.append('<div class="info">')
        if proposed != '?':
            lines.append(f'T_P15: <b>{proposed}</b> (rank #{rank})<br/>')
        lines.append(f'Support: <b>{support}</b>')
        lines.append('</div>')

        lines.append('</div>')

    lines.extend(['</body></html>'])
    return '\n'.join(lines)


def run_p63_report():
    """Generate HTML visual comparison report."""
    t0 = time.time()
    rd = str(results_dir())

    # Load data
    sim_data = _safe_load(os.path.join(rd, 'p63_similarity.json'))
    validate = _safe_load(os.path.join(rd, 'p63_validate.json'))

    if not sim_data or 'rankings_visual' not in sim_data:
        print("ERROR: Similarity data not found. Run vis-similarity first.")
        return

    rankings = sim_data['rankings_visual']

    norm_dir = os.path.join(rd, 'p63_normalized')
    eva_meta = _safe_load(os.path.join(norm_dir, 'eva_metadata.json'))
    costa_meta = _safe_load(os.path.join(norm_dir, 'costamagna_metadata.json'))
    renders_dir = os.path.join(rd, 'p63_renders')

    print("Phase 63 A6: Generating visual comparison report...")

    html = _build_html(rankings, validate, eva_meta, costa_meta,
                       renders_dir, norm_dir)

    report_path = os.path.join(rd, 'p63_visual_report.html')
    with open(report_path, 'w') as f:
        f.write(html)

    result = ReportResult(
        report_path=report_path,
        n_comparisons=len(rankings),
        elapsed=time.time() - t0,
    )

    _save_json(rd, 'p63_report.json', asdict(result))

    print(f"  Report: {report_path}")
    print(f"  Comparisons: {result.n_comparisons}")
    print(f"  Elapsed: {result.elapsed:.1f}s")
