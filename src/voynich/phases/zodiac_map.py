"""
Step 26.1 – Zodiac Folio Mapping and Label Catalog
===================================================
Build a folio-by-folio map of the zodiac section: which sign is depicted,
what labels and circular text are present, positions, and missing folios.

Dependency chain:
    (none — foundational step)
        → zodiac_map.json
"""

import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import load_corpus, tokenize_eva_chars


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
# Hardcoded folio-to-zodiac mapping from ZL3b-n.txt scholarly annotations
# ---------------------------------------------------------------------------

FOLIO_ZODIAC_MAP = {
    'f70v2':  {'sign': 'pisces',      'month_idx': 3,  'std_word': 'mars',  'std_lang': 'french',  'note': 'Two fish; word "Mars" in Roman alphabet'},
    'f70v1':  {'sign': 'aries',       'month_idx': 4,  'std_word': 'abril', 'std_lang': 'spanish', 'note': 'Aries (dark); word "Abril" in Roman alphabet'},
    'f71r':   {'sign': 'aries',       'month_idx': 4,  'std_word': '',      'std_lang': '',        'note': 'Aries (light)'},
    'f71v':   {'sign': 'taurus',      'month_idx': 5,  'std_word': 'may',   'std_lang': 'english', 'note': 'Taurus (light); word "May" in Roman alphabet'},
    'f72r1':  {'sign': 'taurus',      'month_idx': 5,  'std_word': '',      'std_lang': '',        'note': 'Taurus (dark)'},
    'f72r2':  {'sign': 'gemini',      'month_idx': 6,  'std_word': '',      'std_lang': '',        'note': 'Two persons holding hands'},
    'f72r3':  {'sign': 'cancer',      'month_idx': 7,  'std_word': '',      'std_lang': '',        'note': 'Crab/crayfish'},
    'f72v3':  {'sign': 'leo',         'month_idx': 8,  'std_word': '',      'std_lang': '',        'note': 'Lion'},
    'f72v2':  {'sign': 'virgo',       'month_idx': 9,  'std_word': '',      'std_lang': '',        'note': 'Female figure'},
    'f72v1':  {'sign': 'libra',       'month_idx': 10, 'std_word': '',      'std_lang': '',        'note': 'Scales'},
    'f73r':   {'sign': 'scorpio',     'month_idx': 11, 'std_word': '',      'std_lang': '',        'note': 'Scorpion'},
    'f73v':   {'sign': 'sagittarius', 'month_idx': 12, 'std_word': '',      'std_lang': '',        'note': 'Archer/centaur'},
}

MISSING_SIGNS = {
    'capricornus': {'month_idx': 1, 'note': 'f74 missing from manuscript'},
    'aquarius':    {'month_idx': 2, 'note': 'f74 missing from manuscript'},
}

ALL_ZODIAC_SIGNS = [
    'aries', 'taurus', 'gemini', 'cancer', 'leo', 'virgo',
    'libra', 'scorpio', 'sagittarius', 'capricornus', 'aquarius', 'pisces',
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ZodiacLabel:
    locus_id: str
    locus_type: str       # L, C, R, P
    eva_text: str
    clock_position: str
    n_tokens: int
    n_eva_chars: int
    is_continuation: bool


@dataclass
class ZodiacFolio:
    folio: str
    zodiac_sign: str
    standard_script_word: str
    std_lang: str
    month_index: int
    note: str
    missing: bool
    n_labels: int
    n_circular: int
    n_radial: int
    n_paragraph: int
    n_tokens_total: int
    labels: List[Dict]
    circular_texts: List[Dict]
    radial_labels: List[Dict]


@dataclass
class ZodiacMapResult:
    timestamp: str
    n_folios: int
    n_present: int
    n_missing: int
    n_labels_total: int
    n_circular_total: int
    n_radial_total: int
    n_tokens_total: int
    folio_map: List[Dict]
    zodiac_coverage: List[str]
    missing_signs: List[str]
    standard_script_labels: List[Dict]
    folio_order: List[str]
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_clock_position(raw_text: str) -> str:
    """Extract clock position from IVTFF raw text like <!HH:MM>."""
    match = re.search(r'<!(\d{1,2}:\d{2})', raw_text)
    if match:
        return match.group(1)
    return ''


def _classify_locus(locus) -> str:
    """Classify locus type: L=label, C=circular, R=radial, P=paragraph."""
    lid = locus.locus_id
    # The locus_id format: f70v2.3,&Lz or f70r1.2,@Cc
    # Type character appears after the comma and @/& prefix
    if ',@' in lid or ',&' in lid or ',+' in lid:
        parts = lid.split(',')
        if len(parts) >= 2:
            type_str = parts[-1].lstrip('@&+')
            if type_str.startswith('L'):
                return 'L'
            elif type_str.startswith('C'):
                return 'C'
            elif type_str.startswith('R'):
                return 'R'
            elif type_str.startswith('P'):
                return 'P'
    # Fallback: use locus_type property
    lt = getattr(locus, 'locus_type', '')
    if lt in ('L', 'C', 'R', 'P'):
        return lt
    return 'U'  # unknown


def _is_continuation(locus) -> bool:
    """Check if locus is a continuation (&) rather than start (@)."""
    lid = locus.locus_id
    return ',&' in lid or ',+' in lid


def _count_tokens(text: str) -> int:
    """Count tokens in cleaned EVA text."""
    if not text or not text.strip():
        return 0
    return len([t for t in text.split() if t.strip()])


def _count_eva_chars(text: str) -> int:
    """Count EVA characters in text."""
    total = 0
    for token in text.split():
        token = token.strip()
        if token:
            total += len(tokenize_eva_chars(token))
    return total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_zodiac_map() -> None:
    t0 = time.time()
    print("=" * 70)
    print("STEP 26.1: Zodiac Folio Mapping and Label Catalog")
    print("=" * 70)

    rd = _results_dir()
    corpus = load_corpus(verbose=False)

    folio_results: List[ZodiacFolio] = []
    all_std_labels: List[Dict] = []
    folio_order: List[str] = []

    print(f"\n  1. Mapping {len(FOLIO_ZODIAC_MAP)} zodiac folios ...")

    for folio_id, info in FOLIO_ZODIAC_MAP.items():
        page = corpus.get_page(folio_id)

        labels: List[ZodiacLabel] = []
        circular: List[ZodiacLabel] = []
        radial: List[ZodiacLabel] = []
        paragraphs: List[ZodiacLabel] = []
        n_tokens = 0

        if page is not None:
            for locus in page.loci:
                ltype = _classify_locus(locus)
                clock = _extract_clock_position(locus.raw_text)
                text = locus.clean_text
                ntok = _count_tokens(text)
                nchars = _count_eva_chars(text)
                n_tokens += ntok
                cont = _is_continuation(locus)

                zl = ZodiacLabel(
                    locus_id=locus.locus_id,
                    locus_type=ltype,
                    eva_text=text,
                    clock_position=clock,
                    n_tokens=ntok,
                    n_eva_chars=nchars,
                    is_continuation=cont,
                )

                if ltype == 'L':
                    labels.append(zl)
                elif ltype == 'C':
                    circular.append(zl)
                elif ltype == 'R':
                    radial.append(zl)
                elif ltype == 'P':
                    paragraphs.append(zl)
                else:
                    labels.append(zl)  # fallback

        zf = ZodiacFolio(
            folio=folio_id,
            zodiac_sign=info['sign'],
            standard_script_word=info['std_word'],
            std_lang=info['std_lang'],
            month_index=info['month_idx'],
            note=info['note'],
            missing=False,
            n_labels=len(labels),
            n_circular=len(circular),
            n_radial=len(radial),
            n_paragraph=len(paragraphs),
            n_tokens_total=n_tokens,
            labels=[_convert(asdict(l)) for l in labels],
            circular_texts=[_convert(asdict(c)) for c in circular],
            radial_labels=[_convert(asdict(r)) for r in radial],
        )
        folio_results.append(zf)
        folio_order.append(folio_id)

        if info['std_word']:
            all_std_labels.append({
                'folio': folio_id,
                'word': info['std_word'],
                'language': info['std_lang'],
                'zodiac_sign': info['sign'],
                'month_index': info['month_idx'],
            })

        status = "FOUND" if page is not None else "MISSING"
        n_lab = len(labels)
        n_circ = len(circular)
        n_rad = len(radial)
        std = f" (std-script: '{info['std_word']}')" if info['std_word'] else ""
        print(f"      {folio_id:8s} → {info['sign']:14s} | {status} | "
              f"labels={n_lab}, circular={n_circ}, radial={n_rad}, "
              f"tokens={n_tokens}{std}")

    # Add missing signs
    print(f"\n  2. Missing signs (f74 absent from manuscript):")
    for sign, minfo in MISSING_SIGNS.items():
        print(f"      {sign:14s} — month {minfo['month_idx']:2d} — {minfo['note']}")

    # Coverage summary
    present_signs = sorted(set(info['sign'] for info in FOLIO_ZODIAC_MAP.values()))
    missing_signs = sorted(set(ALL_ZODIAC_SIGNS) - set(present_signs))

    total_labels = sum(zf.n_labels for zf in folio_results)
    total_circular = sum(zf.n_circular for zf in folio_results)
    total_radial = sum(zf.n_radial for zf in folio_results)
    total_tokens = sum(zf.n_tokens_total for zf in folio_results)

    print(f"\n  3. Summary:")
    print(f"      Folios present:  {len(folio_results)}")
    print(f"      Signs present:   {len(present_signs)}/12 ({', '.join(present_signs)})")
    print(f"      Signs missing:   {len(missing_signs)} ({', '.join(missing_signs)})")
    print(f"      Total labels:    {total_labels}")
    print(f"      Total circular:  {total_circular}")
    print(f"      Total radial:    {total_radial}")
    print(f"      Total tokens:    {total_tokens}")
    print(f"      Std-script labels: {len(all_std_labels)}")
    for sl in all_std_labels:
        print(f"        {sl['folio']} → '{sl['word']}' ({sl['language']})")

    # Note on duplicate signs (Aries and Taurus have 2 pages each)
    sign_counts = {}
    for info in FOLIO_ZODIAC_MAP.values():
        sign_counts[info['sign']] = sign_counts.get(info['sign'], 0) + 1
    duplicates = {s: c for s, c in sign_counts.items() if c > 1}
    if duplicates:
        print(f"\n  4. Duplicate-page signs (useful for cross-folio consistency):")
        for sign, count in duplicates.items():
            folios = [f for f, i in FOLIO_ZODIAC_MAP.items() if i['sign'] == sign]
            print(f"      {sign}: {count} pages ({', '.join(folios)})")

    result = ZodiacMapResult(
        timestamp=time.strftime('%Y-%m-%dT%H:%M:%S'),
        n_folios=len(folio_results),
        n_present=len(folio_results),
        n_missing=len(missing_signs),
        n_labels_total=total_labels,
        n_circular_total=total_circular,
        n_radial_total=total_radial,
        n_tokens_total=total_tokens,
        folio_map=[_convert(asdict(zf)) for zf in folio_results],
        zodiac_coverage=present_signs,
        missing_signs=missing_signs,
        standard_script_labels=all_std_labels,
        folio_order=folio_order,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'zodiac_map.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2)

    print(f"\n  → {out_path}")
