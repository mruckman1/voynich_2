"""
Phase 21.4 – EVA-to-Historical Stroke Comparison (eva-compare)
==============================================================
Compares each of 44 EVA characters against all historical signs at the
stroke level using two-tier similarity (canonical + category).

Dependency chain:
    paleo_ingest.json (master_reference.json) + EVA_VISUAL_COMPONENTS
        → eva_stroke_compare.json (this step)
"""

import json
import random
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.reference import (
    EVA_VISUAL_COMPONENTS,
    normalize_stroke,
    stroke_category,
)
from voynich.core.stats import stroke_similarity


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _convert(obj: Any) -> Any:
    if hasattr(obj, '__dataclass_fields__'):
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


def _load_master_reference() -> Dict:
    import os
    path = "data/reference/paleographic/master_reference.json"
    if not os.path.exists(path):
        raise FileNotFoundError("master_reference.json not found — run paleo-ingest first")
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Match levels
# ---------------------------------------------------------------------------

_EXACT_THRESHOLD = 0.85
_NEAR_THRESHOLD = 0.65
_PARTIAL_THRESHOLD = 0.45


def _match_level(score: float) -> str:
    if score >= _EXACT_THRESHOLD:
        return 'exact'
    if score >= _NEAR_THRESHOLD:
        return 'near'
    if score >= _PARTIAL_THRESHOLD:
        return 'partial'
    return 'none'


# Source priority: Bobbio > other Italian > general Tironian > Cappelli
_SOURCE_PRIORITY: Dict[str, int] = {
    'chatelain': 3,  # Includes Bobbio plates
    'schmitz': 2,
    'cappelli': 1,
    'fontana_bsb': 1,
    'fontana_bnf': 1,
}


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class EVAMatchEntry:
    eva_char: str
    eva_first_stroke: str
    eva_last_stroke: str
    eva_glyph_class: str
    n_matches: int
    top_candidates: List[Dict[str, Any]]  # Top 3 Latin values with evidence
    best_match_source: str
    best_match_score: float
    best_match_level: str


@dataclass
class EVAStrokeCompareResult:
    timestamp: str
    n_eva_chars: int
    n_historical_with_strokes: int
    per_char_matches: List[Dict[str, Any]]
    match_level_counts: Dict[str, int]
    null_selectivity: float
    null_mean_score: float
    real_mean_score: float
    gate_passed: bool


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_eva_stroke_compare() -> Dict[str, Any]:
    """Compare 44 EVA chars against all historical signs."""
    t0 = time.time()

    master = _load_master_reference()
    all_signs = master.get('all_signs', [])

    # Filter to signs with stroke data (Chatelain + Schmitz + 149 Cappelli visual)
    hist_with_strokes = [
        s for s in all_signs
        if s.get('first_stroke') or s.get('final_stroke')
    ]

    # Also load prior anchors for cross-referencing
    anchors: Dict[str, str] = {}
    try:
        anchor_path = _results_dir() / "cross_approach.json"
        if anchor_path.exists():
            with open(anchor_path) as f:
                anchor_data = json.load(f)
            for m in anchor_data.get('mappings', []):
                if isinstance(m, dict):
                    eva_tok = m.get('eva_token', '')
                    latin = m.get('latin_value', '')
                    if eva_tok and latin:
                        anchors[eva_tok] = latin
    except Exception:
        pass

    # --- Compare each EVA char ---
    per_char_matches: List[EVAMatchEntry] = []
    real_scores: List[float] = []

    for eva_char, components in EVA_VISUAL_COMPONENTS.items():
        eva_strokes = {
            'first_stroke': components.get('first_stroke', ''),
            'last_stroke': components.get('last_stroke', ''),
            'glyph_class': components.get('glyph_class', ''),
        }

        # Score against every historical sign
        scored_matches: List[Tuple[float, Dict]] = []
        for h_sign in hist_with_strokes:
            h_strokes = {
                'first_stroke': h_sign.get('first_stroke', ''),
                'last_stroke': h_sign.get('final_stroke', '') or h_sign.get('last_stroke', ''),
                'glyph_class': h_sign.get('glyph_class', ''),
            }
            sim = stroke_similarity(eva_strokes, h_strokes, include_class=True)
            if sim >= _PARTIAL_THRESHOLD:
                scored_matches.append((sim, h_sign))

        # Sort by score (desc), then source priority (desc)
        scored_matches.sort(
            key=lambda x: (x[0], _SOURCE_PRIORITY.get(x[1].get('source', ''), 0)),
            reverse=True,
        )

        # Aggregate top candidates: group by Latin value
        latin_to_evidence: Dict[str, Dict[str, Any]] = {}
        for score, h_sign in scored_matches[:20]:  # Top 20 matches
            lv = h_sign.get('latin_value', '') or ''
            if not lv:
                continue
            lv_lower = lv.lower().strip()
            if lv_lower not in latin_to_evidence:
                latin_to_evidence[lv_lower] = {
                    'latin_value': lv_lower,
                    'best_score': score,
                    'match_level': _match_level(score),
                    'sources': [],
                    'evidence_count': 0,
                }
            entry = latin_to_evidence[lv_lower]
            entry['sources'].append(h_sign.get('source', ''))
            entry['evidence_count'] += 1

        # Sort candidates by evidence count then best score
        top_candidates = sorted(
            latin_to_evidence.values(),
            key=lambda x: (x['evidence_count'], x['best_score']),
            reverse=True,
        )[:3]

        best_score = scored_matches[0][0] if scored_matches else 0.0
        best_source = scored_matches[0][1].get('source', '') if scored_matches else ''

        real_scores.append(best_score)

        per_char_matches.append(EVAMatchEntry(
            eva_char=eva_char,
            eva_first_stroke=components.get('first_stroke', ''),
            eva_last_stroke=components.get('last_stroke', ''),
            eva_glyph_class=components.get('glyph_class', ''),
            n_matches=len(scored_matches),
            top_candidates=top_candidates,
            best_match_source=best_source,
            best_match_score=best_score,
            best_match_level=_match_level(best_score),
        ))

    # --- Match level counts ---
    level_counts = Counter(m.best_match_level for m in per_char_matches)

    # --- Null control ---
    # Permute EVA stroke assignments 50 times, re-compare
    rng = random.Random(42)
    null_mean_scores: List[float] = []

    all_first = [c.get('first_stroke', '') for c in EVA_VISUAL_COMPONENTS.values()]
    all_last = [c.get('last_stroke', '') for c in EVA_VISUAL_COMPONENTS.values()]

    for _ in range(50):
        shuffled_first = list(all_first)
        shuffled_last = list(all_last)
        rng.shuffle(shuffled_first)
        rng.shuffle(shuffled_last)

        null_scores_iter: List[float] = []
        for j, (eva_char, components) in enumerate(EVA_VISUAL_COMPONENTS.items()):
            null_strokes = {
                'first_stroke': shuffled_first[j],
                'last_stroke': shuffled_last[j],
                'glyph_class': components.get('glyph_class', ''),
            }
            best_null = 0.0
            for h_sign in hist_with_strokes:
                h_strokes = {
                    'first_stroke': h_sign.get('first_stroke', ''),
                    'last_stroke': h_sign.get('final_stroke', '') or h_sign.get('last_stroke', ''),
                    'glyph_class': h_sign.get('glyph_class', ''),
                }
                sim = stroke_similarity(null_strokes, h_strokes, include_class=True)
                if sim > best_null:
                    best_null = sim
            null_scores_iter.append(best_null)
        null_mean_scores.append(sum(null_scores_iter) / max(len(null_scores_iter), 1))

    real_mean = sum(real_scores) / max(len(real_scores), 1)
    null_mean = sum(null_mean_scores) / max(len(null_mean_scores), 1)
    selectivity = real_mean / max(null_mean, 0.01)

    result = EVAStrokeCompareResult(
        timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
        n_eva_chars=len(per_char_matches),
        n_historical_with_strokes=len(hist_with_strokes),
        per_char_matches=[_convert(asdict(m)) for m in per_char_matches],
        match_level_counts=dict(level_counts),
        null_selectivity=selectivity,
        null_mean_score=null_mean,
        real_mean_score=real_mean,
        gate_passed=selectivity > 1.5,
    )

    out_path = _results_dir() / "eva_stroke_compare.json"
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t0
    print(f"eva-compare: {len(per_char_matches)} EVA chars vs {len(hist_with_strokes)} historical, "
          f"levels={dict(level_counts)}, selectivity={selectivity:.2f}x ({elapsed:.1f}s)")

    return _convert(asdict(result))
