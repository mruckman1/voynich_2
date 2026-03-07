"""
Phase 21.1 – Paleographic Source Normalization (paleo-ingest)
=============================================================
Loads all 5 historical sources from data/2Translate/, normalizes stroke
vocabularies, and writes a unified sign database to
data/reference/paleographic/master_reference.json.

Dependency chain:
    data/2Translate/*.json + EVA_VISUAL_COMPONENTS
        → paleo_ingest.json (this step)
        → data/reference/paleographic/master_reference.json
"""

import json
import os
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.reference import (
    normalize_stroke,
    stroke_category,
    infer_glyph_class,
    STROKE_CANONICAL_MAP,
    STROKE_CATEGORY_MAP,
)


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


# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------

_TRANSLATE_DIR = Path("data/2Translate")
_CHATELAIN_JSON = (
    _TRANSLATE_DIR
    / "Chatelain_Introduction à la lecture des notes tironiennes_DONE"
    / "Chatelain_Introduction à la lecture des notes tironiennes_extracted.json"
)
_SCHMITZ_JSON = (
    _TRANSLATE_DIR
    / "Schmitz_Commentarii_Notarum_Tironianarum_DONE"
    / "Schmitz_Commentarii_Notarum_Tironianarum_extracted.json"
)
_CAPPELLI_JSON = (
    _TRANSLATE_DIR
    / "Cappelli_Lexicon Abbreviaturarum_DONE"
    / "Cappelli_Lexicon Abbreviaturarum_extracted.json"
)
_FONTANA_BSB_JSON = (
    _TRANSLATE_DIR
    / "Fontana_Bellicorum_Instrumentorum_Liber_bsb_DONE"
    / "Fontana_Bellicorum_Instrumentorum_Liber_extracted.json"
)


def _fontana_bnf_path() -> Path:
    """Locate the Fontana BNF extracted JSON (filename is long)."""
    matches = list(
        _TRANSLATE_DIR.glob("Fontana_Secretum*/*_extracted.json")
    )
    if not matches:
        raise FileNotFoundError("Fontana BNF extracted JSON not found")
    return matches[0]


# ---------------------------------------------------------------------------
# Dataclass for unified sign records
# ---------------------------------------------------------------------------

@dataclass
class NormalizedSign:
    source: str                          # chatelain|schmitz|cappelli|fontana_bsb|fontana_bnf
    source_id: str                       # unique within source
    latin_value: Optional[str] = None    # syllable/word this sign represents
    first_stroke: Optional[str] = None   # canonical form
    middle_strokes: List[str] = field(default_factory=list)
    final_stroke: Optional[str] = None   # canonical form
    first_category: Optional[str] = None
    final_category: Optional[str] = None
    modifier_marks: List[str] = field(default_factory=list)
    glyph_class: Optional[str] = None
    base_form: Optional[str] = None      # Fontana only
    added_feature: Optional[str] = None  # Fontana only
    bracket_marks: List[str] = field(default_factory=list)
    geographic_source: Optional[str] = None
    variant_of: Optional[str] = None
    sign_complexity: str = 'simple'
    confidence: str = 'low'


# ---------------------------------------------------------------------------
# Per-source normalization
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> Dict:
    with open(path) as f:
        return json.load(f)


def _normalize_chatelain(data: Dict) -> List[NormalizedSign]:
    """Normalize Chatelain entries — stroke triples + label_text."""
    entries = data.get('entries', [])
    signs: List[NormalizedSign] = []

    for i, e in enumerate(entries):
        fs_raw = e.get('first_stroke', '') or ''
        ms_raw = e.get('middle_strokes', []) or []
        ls_raw = e.get('final_stroke', '') or ''

        fs = normalize_stroke(fs_raw)
        ms = [normalize_stroke(m) for m in ms_raw if m]
        ls = normalize_stroke(ls_raw)

        middle_count = len(ms)
        complexity = 'compound' if middle_count > 0 else 'simple'
        gc = infer_glyph_class(fs_raw, middle_count, complexity)

        label = e.get('label_text', '') or ''
        # Extract Latin value: label_text is typically the Latin word
        latin_val = label.strip() if label else None

        signs.append(NormalizedSign(
            source='chatelain',
            source_id=f"ch_{e.get('plate', '?')}_{e.get('position', i)}",
            latin_value=latin_val,
            first_stroke=fs if fs else None,
            middle_strokes=ms,
            final_stroke=ls if ls else None,
            first_category=stroke_category(fs) if fs else None,
            final_category=stroke_category(ls) if ls else None,
            modifier_marks=e.get('modifier_marks', []) or [],
            glyph_class=gc,
            geographic_source=e.get('geographic_source'),
            variant_of=e.get('variant_of_position'),
            sign_complexity=complexity,
            confidence=e.get('confidence', 'low') or 'low',
        ))

    return signs


def _normalize_schmitz(data: Dict) -> List[NormalizedSign]:
    """Normalize Schmitz entries — stroke triples + latin_expansion."""
    entries = data.get('entries', [])
    signs: List[NormalizedSign] = []

    for i, e in enumerate(entries):
        fs_raw = e.get('first_stroke', '') or ''
        ms_raw = e.get('middle_strokes', []) or []
        ls_raw = e.get('final_stroke', '') or ''

        fs = normalize_stroke(fs_raw)
        ms = [normalize_stroke(m) for m in ms_raw if m]
        ls = normalize_stroke(ls_raw)

        middle_count = len(ms)
        sc = e.get('sign_complexity', 'simple') or 'simple'
        complexity = sc if sc in ('simple', 'compound') else 'simple'
        gc = infer_glyph_class(fs_raw, middle_count, complexity)

        latin_val = (e.get('latin_expansion', '') or '').strip() or None

        signs.append(NormalizedSign(
            source='schmitz',
            source_id=f"sm_{e.get('plate', '?')}_{e.get('position', i)}",
            latin_value=latin_val,
            first_stroke=fs if fs else None,
            middle_strokes=ms,
            final_stroke=ls if ls else None,
            first_category=stroke_category(fs) if fs else None,
            final_category=stroke_category(ls) if ls else None,
            modifier_marks=e.get('modifier_marks', []) or [],
            glyph_class=gc,
            sign_complexity=complexity,
            confidence=e.get('confidence', 'low') or 'low',
        ))

    return signs


# Bracket marks to extract from Cappelli abbreviated_form
_BRACKET_PATTERN = re.compile(r'\[([^\]]+)\]')


def _extract_bracket_marks(abbreviated_form: str) -> List[str]:
    """Extract bracket notation types from Cappelli abbreviated_form."""
    if not abbreviated_form:
        return []
    return _BRACKET_PATTERN.findall(abbreviated_form)


def _parse_visual_description(desc: str) -> Dict[str, Optional[str]]:
    """Parse Cappelli visual_description into stroke-like fields.

    Returns dict with first_stroke, final_stroke as canonical forms.
    Uses keyword matching against known stroke vocabulary.
    """
    if not desc:
        return {'first_stroke': None, 'final_stroke': None}

    desc_lower = desc.lower()
    first = None
    final = None

    # Keyword → canonical stroke mapping for visual descriptions
    stroke_keywords = [
        ('vertical line', 'vertical_stroke'),
        ('vertical stroke', 'vertical_stroke'),
        ('horizontal line', 'horizontal_stroke'),
        ('horizontal stroke', 'horizontal_stroke'),
        ('diagonal', 'diagonal_right'),
        ('curve', 'open_curve'),
        ('curved', 'open_curve'),
        ('loop', 'closed_loop'),
        ('circle', 'closed_loop'),
        ('hook', 'hook'),
        ('ascend', 'ascender'),
        ('descend', 'descender'),
        ('dot', 'dot'),
        ('cross', 'crossbar'),
        ('tick', 'tick'),
        ('wavy', 'sigmoid'),
        ('s-shaped', 'sigmoid'),
        ('zigzag', 'sigmoid'),
    ]

    found = []
    for keyword, canonical in stroke_keywords:
        if keyword in desc_lower:
            found.append(canonical)

    if found:
        first = found[0]
        final = found[-1] if len(found) > 1 else found[0]

    return {'first_stroke': first, 'final_stroke': final}


def _normalize_cappelli(data: Dict) -> List[NormalizedSign]:
    """Normalize Cappelli entries — bracket marks + optional visual description."""
    entries = data.get('entries', [])
    signs: List[NormalizedSign] = []

    for i, e in enumerate(entries):
        bracket_marks = _extract_bracket_marks(e.get('abbreviated_form', '') or '')
        visual = _parse_visual_description(e.get('visual_description', '') or '')

        fs = visual.get('first_stroke')
        ls = visual.get('final_stroke')

        latin_val = (e.get('latin_expansion', '') or '').strip() or None
        priority = e.get('priority', '') or ''

        signs.append(NormalizedSign(
            source='cappelli',
            source_id=f"cap_{e.get('page', '?')}_{i}",
            latin_value=latin_val,
            first_stroke=fs,
            final_stroke=ls,
            first_category=stroke_category(fs) if fs else None,
            final_category=stroke_category(ls) if ls else None,
            bracket_marks=bracket_marks,
            glyph_class=None,  # Cappelli entries don't have enough stroke info
            sign_complexity='simple',
            confidence=e.get('confidence', 'low') or 'low',
        ))

    return signs


# Fontana base_form → canonical stroke mapping
_FONTANA_BASE_MAP: Dict[str, str] = {
    'circle': 'closed_loop',
    'oval': 'closed_loop',
    'vertical_stroke': 'vertical_stroke',
    'vertical_line': 'vertical_stroke',
    'horizontal_stroke': 'horizontal_stroke',
    'horizontal_line': 'horizontal_stroke',
    'diagonal_line': 'diagonal_right',
    'open_curve_right': 'open_curve_right',
    'open_curve_left': 'open_curve_left',
    'zigzag': 'sigmoid',
    'cross': 'crossbar',
    'dot': 'dot',
    'hook': 'hook',
    'triangle': 'diagonal_right',
    'square': 'horizontal_stroke',
    'star': 'crossbar',
}


def _normalize_fontana(data: Dict, source_name: str) -> List[NormalizedSign]:
    """Normalize Fontana entries — flatten cipher_signs from all pages."""
    entries = data.get('entries', [])
    signs: List[NormalizedSign] = []
    seen_ids: set = set()

    for page_entry in entries:
        folio = page_entry.get('folio', '?')
        for s in page_entry.get('cipher_signs', []):
            sign_id = s.get('sign_id', '')
            if sign_id in seen_ids:
                continue
            seen_ids.add(sign_id)

            base = (s.get('base_form', '') or '').lower()
            added = s.get('added_feature', '') or ''
            canon_base = _FONTANA_BASE_MAP.get(base, base)

            # For Fontana, base_form maps to first_stroke; no real final_stroke
            # unless the added_feature provides directional info
            fs = normalize_stroke(canon_base) if canon_base else None

            signs.append(NormalizedSign(
                source=source_name,
                source_id=f"{source_name}_{sign_id}",
                latin_value=s.get('letter_value'),  # Almost always None
                first_stroke=fs,
                final_stroke=None,
                first_category=stroke_category(fs) if fs else None,
                final_category=None,
                base_form=base,
                added_feature=added if added else None,
                glyph_class=None,
                sign_complexity='simple',
                confidence=s.get('confidence', 'low') or 'low',
            ))

    return signs


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class PaleoIngestResult:
    timestamp: str
    source_counts: Dict[str, int]
    total_signs: int
    signs_with_latin: int
    signs_with_strokes: int
    simple_signs: int
    compound_signs: int
    stroke_normalization_stats: Dict[str, Any]
    per_source_summary: Dict[str, Dict[str, Any]]
    master_reference_path: str


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_paleo_ingest() -> Dict[str, Any]:
    """Load all 5 sources, normalize, write unified master_reference.json."""
    t0 = time.time()

    all_signs: List[NormalizedSign] = []

    # --- Chatelain ---
    ch_data = _load_json(_CHATELAIN_JSON)
    ch_signs = _normalize_chatelain(ch_data)
    all_signs.extend(ch_signs)

    # --- Schmitz ---
    sm_data = _load_json(_SCHMITZ_JSON)
    sm_signs = _normalize_schmitz(sm_data)
    all_signs.extend(sm_signs)

    # --- Cappelli ---
    cap_data = _load_json(_CAPPELLI_JSON)
    cap_signs = _normalize_cappelli(cap_data)
    all_signs.extend(cap_signs)

    # --- Fontana BSB ---
    fb_data = _load_json(_FONTANA_BSB_JSON)
    fb_signs = _normalize_fontana(fb_data, 'fontana_bsb')
    all_signs.extend(fb_signs)

    # --- Fontana BNF ---
    fn_data = _load_json(_fontana_bnf_path())
    fn_signs = _normalize_fontana(fn_data, 'fontana_bnf')
    all_signs.extend(fn_signs)

    # --- Statistics ---
    source_counts: Dict[str, int] = Counter()
    signs_with_latin = 0
    signs_with_strokes = 0
    simple_count = 0
    compound_count = 0
    raw_strokes_seen: Counter = Counter()
    canonical_strokes_seen: Counter = Counter()

    for s in all_signs:
        source_counts[s.source] += 1
        if s.latin_value:
            signs_with_latin += 1
        if s.first_stroke or s.final_stroke:
            signs_with_strokes += 1
        if s.sign_complexity == 'simple':
            simple_count += 1
        else:
            compound_count += 1
        if s.first_stroke:
            canonical_strokes_seen[s.first_stroke] += 1
        if s.final_stroke:
            canonical_strokes_seen[s.final_stroke] += 1

    # Per-source summaries
    per_source: Dict[str, Dict[str, Any]] = {}
    for src_name in ['chatelain', 'schmitz', 'cappelli', 'fontana_bsb', 'fontana_bnf']:
        src_signs = [s for s in all_signs if s.source == src_name]
        n_latin = sum(1 for s in src_signs if s.latin_value)
        n_strokes = sum(1 for s in src_signs if s.first_stroke or s.final_stroke)
        n_simple = sum(1 for s in src_signs if s.sign_complexity == 'simple')
        per_source[src_name] = {
            'count': len(src_signs),
            'with_latin_value': n_latin,
            'with_stroke_data': n_strokes,
            'simple': n_simple,
            'compound': len(src_signs) - n_simple,
        }

    # --- Write master_reference.json ---
    paleo_dir = Path("data/reference/paleographic")
    paleo_dir.mkdir(parents=True, exist_ok=True)
    master_path = paleo_dir / "master_reference.json"

    master_data = {
        'metadata': {
            'generated_by': 'paleo_ingest',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_signs': len(all_signs),
            'source_counts': dict(source_counts),
        },
        'all_signs': [_convert(asdict(s)) for s in all_signs],
    }
    with open(master_path, 'w') as f:
        json.dump(master_data, f, indent=2, ensure_ascii=False)

    # --- Build result ---
    result = PaleoIngestResult(
        timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
        source_counts=dict(source_counts),
        total_signs=len(all_signs),
        signs_with_latin=signs_with_latin,
        signs_with_strokes=signs_with_strokes,
        simple_signs=simple_count,
        compound_signs=compound_count,
        stroke_normalization_stats={
            'canonical_map_size': len(STROKE_CANONICAL_MAP),
            'category_map_size': len(STROKE_CATEGORY_MAP),
            'unique_canonical_strokes': len(canonical_strokes_seen),
            'top_canonical_strokes': dict(canonical_strokes_seen.most_common(15)),
        },
        per_source_summary=per_source,
        master_reference_path=str(master_path),
    )

    # Save results
    out_path = _results_dir() / "paleo_ingest.json"
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t0
    print(f"paleo-ingest: {len(all_signs)} signs from 5 sources, "
          f"{signs_with_latin} with Latin values, "
          f"{signs_with_strokes} with stroke data ({elapsed:.1f}s)")

    return _convert(asdict(result))
