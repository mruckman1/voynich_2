"""
Step 41.12 – Complete f57v Reading Assembly
=============================================
Assemble the most complete possible reading of folio f57v by combining
all upstream analyses: formula segmentation, content-zone token analysis,
ingredient search, and the syllable/complete lexicon.  Produces a layered
reading with confidence-coded token annotations and recipe-structured
interpretation.

Dependency chain:
    f57v_reading.json            (Step 40.11)
    formula_segmentation.json    (Step 41.9)
    inter_formula_tokens.json    (Step 41.10)
    ingredient_search.json       (Step 41.11)
    complete_lexicon.json | syllable_lexicon.json  (Step 40.9+)
        → f57v_complete_reading.json  (this step)
"""

import json
import os
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from voynich.core._paths import results_dir as _results_dir


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _convert(obj: Any) -> Any:
    if hasattr(obj, '__dataclass_fields__'):
        from dataclasses import asdict
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


def _safe_load(path: str) -> Dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


# ---------------------------------------------------------------------------
# Core: build layered reading
# ---------------------------------------------------------------------------

def _determine_zone_type(
    position: int,
    zones: List[Dict],
) -> str:
    """Determine which zone type a position belongs to."""
    for z in zones:
        if z['start'] <= position < z['end']:
            return z['zone_type']
    return 'UNKNOWN'


def _determine_zone_id(
    position: int,
    zones: List[Dict],
) -> Optional[int]:
    """Determine which zone id a position belongs to."""
    for z in zones:
        if z['start'] <= position < z['end']:
            return z['zone_id']
    return None


def _find_ingredient_match(
    decoded: str,
    zone_id: Optional[int],
    ingredient_data: Dict,
) -> Optional[Dict]:
    """Look up ingredient match for a decoded token in a specific zone."""
    zone_matches = ingredient_data.get('zone_match_results', [])
    for zm in zone_matches:
        if zm['zone_id'] != zone_id:
            continue
        for m in zm.get('matches', []):
            if m.get('decoded') == decoded and m.get('best_match'):
                return m
        for m in zm.get('head_matches', []):
            if m.get('decoded') == decoded and m.get('best_match'):
                return m
    return None


def _assign_confidence(
    classification: str,
    english_gloss: str,
    lexicon_entry: Dict,
    ingredient_match: Optional[Dict],
) -> str:
    """Assign a confidence level to a token.

    GREEN:  confirmed SIGNAL + glossed in lexicon
    YELLOW: SIGNAL but ambiguous or unglossed, or SHARED_HIT + glossed
    ORANGE: ingredient match at edit-distance-1, or non-SIGNAL with
            partial match
    RED:    no match at all
    """
    is_signal = classification == 'SIGNAL'
    is_glossed = english_gloss not in ('___', '???', '')
    has_lexicon = bool(lexicon_entry)
    has_ingredient = ingredient_match is not None

    if is_signal and is_glossed and has_lexicon:
        return 'GREEN'
    if is_signal and (is_glossed or has_lexicon or has_ingredient):
        return 'YELLOW'
    if classification == 'SHARED_HIT' and is_glossed:
        return 'YELLOW'
    if has_ingredient:
        # Check if exact or ed1
        if ingredient_match:
            src = ingredient_match.get('best_match_source', '')
            if 'ed1' in src:
                return 'ORANGE'
            else:
                return 'YELLOW'
        return 'ORANGE'
    return 'RED'


def _build_layered_reading(
    line_by_line: List[Dict],
    zones: List[Dict],
    lexicon: Dict[str, Dict],
    ingredient_data: Dict,
) -> List[Dict]:
    """Build the layered reading for each token position."""
    reading = []

    for t in line_by_line:
        pos = t['position']
        decoded = t['decoded']
        classification = t['classification']
        english_gloss = t['english_gloss']

        zone_type = _determine_zone_type(pos, zones)
        zone_id = _determine_zone_id(pos, zones)

        # Look up in lexicon
        lex_entry = lexicon.get(decoded, {})
        lex_gloss = lex_entry.get('english_gloss', '')
        lex_pos = lex_entry.get('part_of_speech', '')
        venetian_form = lex_entry.get('venetian_form', '')

        # Look up ingredient match
        ing_match = _find_ingredient_match(decoded, zone_id, ingredient_data)
        ing_word = ing_match.get('best_match', '') if ing_match else ''
        ing_english = ing_match.get('best_english', '') if ing_match else ''
        ing_source = ing_match.get('best_match_source', '') if ing_match else ''

        # Determine best available gloss
        if english_gloss not in ('___', '???', ''):
            best_gloss = english_gloss
        elif lex_gloss and lex_gloss not in ('???', ''):
            best_gloss = lex_gloss
        elif ing_english:
            best_gloss = ing_english
        elif ing_word:
            best_gloss = f'[{ing_word}?]'
        else:
            best_gloss = ''

        # Assign confidence
        confidence = _assign_confidence(
            classification, english_gloss, lex_entry, ing_match,
        )

        reading.append({
            'position': pos,
            'eva_token': decoded,  # decoded form from upstream
            'decoded': decoded,
            'venetian_word': venetian_form or ing_word or '',
            'english_gloss': best_gloss,
            'confidence': confidence,
            'zone_type': zone_type,
            'zone_id': zone_id,
            'classification': classification,
            'pos': lex_pos or t.get('pos', ''),
            'ingredient_match': ing_word,
            'ingredient_english': ing_english,
            'ingredient_source': ing_source,
        })

    return reading


def _compute_coverage(reading: List[Dict]) -> Dict:
    """Count tokens at each confidence level."""
    counts = Counter(r['confidence'] for r in reading)
    n = len(reading)
    return {
        'GREEN': counts.get('GREEN', 0),
        'YELLOW': counts.get('YELLOW', 0),
        'ORANGE': counts.get('ORANGE', 0),
        'RED': counts.get('RED', 0),
        'total': n,
        'green_pct': round(counts.get('GREEN', 0) / max(n, 1), 4),
        'yellow_pct': round(counts.get('YELLOW', 0) / max(n, 1), 4),
        'orange_pct': round(counts.get('ORANGE', 0) / max(n, 1), 4),
        'red_pct': round(counts.get('RED', 0) / max(n, 1), 4),
        'glossed_pct': round(
            (counts.get('GREEN', 0) + counts.get('YELLOW', 0)
             + counts.get('ORANGE', 0)) / max(n, 1),
            4,
        ),
    }


def _find_best_passage(reading: List[Dict]) -> Dict:
    """Find the longest consecutive run of GREEN + YELLOW tokens."""
    best_start = 0
    best_len = 0
    cur_start = 0
    cur_len = 0

    for i, r in enumerate(reading):
        if r['confidence'] in ('GREEN', 'YELLOW'):
            if cur_len == 0:
                cur_start = i
            cur_len += 1
            if cur_len > best_len:
                best_len = cur_len
                best_start = cur_start
        else:
            cur_len = 0

    passage_tokens = reading[best_start:best_start + best_len]
    passage_text = ' '.join(
        r['english_gloss'] if r['english_gloss'] else f"[{r['decoded']}]"
        for r in passage_tokens
    )

    return {
        'start': best_start,
        'length': best_len,
        'tokens': [
            {
                'position': r['position'],
                'decoded': r['decoded'],
                'english_gloss': r['english_gloss'],
                'confidence': r['confidence'],
            }
            for r in passage_tokens
        ],
        'passage_text': passage_text,
    }


def _assemble_recipe_segments(
    reading: List[Dict],
    zones: List[Dict],
    formula_glosses: List[str],
) -> List[Dict]:
    """For each FORMULA + CONTENT pair, assemble a recipe interpretation."""
    recipes = []
    i = 0

    while i < len(zones):
        z = zones[i]

        if z['zone_type'] == 'FORMULA':
            formula_zone = z
            content_zone = None
            # Look for the next CONTENT zone
            if i + 1 < len(zones) and zones[i + 1]['zone_type'] == 'CONTENT':
                content_zone = zones[i + 1]
                i += 2
            else:
                i += 1

            # Extract formula reading
            formula_tokens = [
                r for r in reading
                if formula_zone['start'] <= r['position'] < formula_zone['end']
            ]
            formula_text = ' '.join(
                r['english_gloss'] if r['english_gloss'] else f"[{r['decoded']}]"
                for r in formula_tokens
            )

            # Extract content reading
            content_text = ''
            content_ingredients = []
            content_tokens_list = []
            if content_zone:
                content_tokens_list = [
                    r for r in reading
                    if content_zone['start'] <= r['position'] < content_zone['end']
                ]
                content_text = ' '.join(
                    r['english_gloss'] if r['english_gloss']
                    else f"[{r['decoded']}]"
                    for r in content_tokens_list
                )
                # Collect ingredient matches
                for r in content_tokens_list:
                    if r.get('ingredient_match'):
                        content_ingredients.append({
                            'decoded': r['decoded'],
                            'ingredient': r['ingredient_match'],
                            'english': r['ingredient_english'],
                            'source': r['ingredient_source'],
                        })

            recipes.append({
                'formula_zone_id': formula_zone['zone_id'],
                'content_zone_id': content_zone['zone_id'] if content_zone else None,
                'formula_text': formula_text,
                'formula_glossed': ' | '.join(formula_glosses) if formula_glosses else '',
                'content_text': content_text,
                'n_content_tokens': len(content_tokens_list),
                'content_ingredients': content_ingredients,
                'n_ingredients': len(content_ingredients),
            })
        elif z['zone_type'] == 'HEADER':
            # Include header as a preamble
            header_tokens = [
                r for r in reading
                if z['start'] <= r['position'] < z['end']
            ]
            header_text = ' '.join(
                r['english_gloss'] if r['english_gloss']
                else f"[{r['decoded']}]"
                for r in header_tokens
            )
            recipes.append({
                'formula_zone_id': None,
                'content_zone_id': z['zone_id'],
                'formula_text': '(HEADER)',
                'formula_glossed': '',
                'content_text': header_text,
                'n_content_tokens': len(header_tokens),
                'content_ingredients': [],
                'n_ingredients': 0,
            })
            i += 1
        else:
            # Standalone CONTENT zone (after last formula, or orphan)
            content_tokens_list = [
                r for r in reading
                if z['start'] <= r['position'] < z['end']
            ]
            content_text = ' '.join(
                r['english_gloss'] if r['english_gloss']
                else f"[{r['decoded']}]"
                for r in content_tokens_list
            )
            content_ingredients = []
            for r in content_tokens_list:
                if r.get('ingredient_match'):
                    content_ingredients.append({
                        'decoded': r['decoded'],
                        'ingredient': r['ingredient_match'],
                        'english': r['ingredient_english'],
                        'source': r['ingredient_source'],
                    })

            recipes.append({
                'formula_zone_id': None,
                'content_zone_id': z['zone_id'],
                'formula_text': '(CONTENT only)',
                'formula_glossed': '',
                'content_text': content_text,
                'n_content_tokens': len(content_tokens_list),
                'content_ingredients': content_ingredients,
                'n_ingredients': len(content_ingredients),
            })
            i += 1

    return recipes


def _build_definitive_summary(
    reading: List[Dict],
    coverage: Dict,
    best_passage: Dict,
    recipes: List[Dict],
    pattern_info: Dict,
) -> str:
    """Produce a human-readable summary of the f57v reading."""
    lines = []
    lines.append("DEFINITIVE f57v READING SUMMARY")
    lines.append("=" * 40)
    lines.append("")

    n = coverage['total']
    lines.append(f"Total tokens: {n}")
    lines.append(f"  GREEN  (confirmed): {coverage['GREEN']:3d} ({coverage['green_pct']:.1%})")
    lines.append(f"  YELLOW (probable):  {coverage['YELLOW']:3d} ({coverage['yellow_pct']:.1%})")
    lines.append(f"  ORANGE (tentative): {coverage['ORANGE']:3d} ({coverage['orange_pct']:.1%})")
    lines.append(f"  RED    (unknown):   {coverage['RED']:3d} ({coverage['red_pct']:.1%})")
    lines.append(f"  Overall glossed:    {coverage['glossed_pct']:.1%}")
    lines.append("")

    pat = pattern_info.get('pattern_str', '???')
    pat_count = pattern_info.get('count', 0)
    lines.append(f"Repeating formula: \"{pat}\" (x{pat_count})")
    lines.append(f"Best passage: {best_passage['length']} consecutive GREEN/YELLOW tokens")
    lines.append(f"  \"{best_passage['passage_text'][:200]}\"")
    lines.append("")

    lines.append("RECIPE SEGMENTS:")
    for i, rec in enumerate(recipes):
        lines.append(f"  [{i}] Formula: {rec['formula_text'][:80]}")
        lines.append(f"      Content:  {rec['content_text'][:80]}")
        if rec['content_ingredients']:
            ings = ', '.join(
                f"{ci['ingredient']}({ci['english']})"
                if ci['english'] else ci['ingredient']
                for ci in rec['content_ingredients']
            )
            lines.append(f"      Ingredients: {ings}")
        lines.append("")

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_f57v_complete_reading() -> None:
    """Step 41.12: Assemble complete f57v reading."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 41.12: Complete f57v Reading Assembly")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load all upstream ──
    print("\n  1. Loading upstream data ...")
    f57v_data = _safe_load(os.path.join(rd, 'f57v_reading.json'))
    seg_data = _safe_load(os.path.join(rd, 'formula_segmentation.json'))
    inter_data = _safe_load(os.path.join(rd, 'inter_formula_tokens.json'))
    ing_data = _safe_load(os.path.join(rd, 'ingredient_search.json'))

    # Try complete_lexicon first, fall back to syllable_lexicon
    lex_data = _safe_load(os.path.join(rd, 'complete_lexicon.json'))
    if not lex_data:
        lex_data = _safe_load(os.path.join(rd, 'syllable_lexicon.json'))

    if not f57v_data:
        print("    ERROR: f57v_reading.json not found. Cannot proceed.")
        output = {'error': 'f57v_reading.json not found', 'runtime_seconds': 0.0}
        out_path = os.path.join(rd, 'f57v_complete_reading.json')
        with open(out_path, 'w') as f:
            json.dump(output, f, indent=2)
        return

    line_by_line = f57v_data.get('line_by_line', [])
    zones = seg_data.get('zones', []) if seg_data else []
    lexicon = lex_data.get('syllable_lexicon', lex_data.get('lexicon', {}))
    formula_glosses = seg_data.get('formula_glosses', []) if seg_data else []
    pattern_info = seg_data.get('pattern', {}) if seg_data else {}

    print(f"    f57v tokens: {len(line_by_line)}")
    print(f"    Zones: {len(zones)}")
    print(f"    Lexicon entries: {len(lexicon)}")
    print(f"    Ingredient data: {'loaded' if ing_data else 'not found'}")

    # ── 2. Build layered reading ──
    print("\n  2. Building layered reading ...")
    reading = _build_layered_reading(
        line_by_line, zones, lexicon, ing_data,
    )
    print(f"    Reading entries: {len(reading)}")

    # ── 3. Compute coverage ──
    print("\n  3. Computing coverage ...")
    coverage = _compute_coverage(reading)
    print(f"    GREEN:  {coverage['GREEN']:3d} ({coverage['green_pct']:.1%})")
    print(f"    YELLOW: {coverage['YELLOW']:3d} ({coverage['yellow_pct']:.1%})")
    print(f"    ORANGE: {coverage['ORANGE']:3d} ({coverage['orange_pct']:.1%})")
    print(f"    RED:    {coverage['RED']:3d} ({coverage['red_pct']:.1%})")
    print(f"    Glossed: {coverage['glossed_pct']:.1%}")

    # ── 4. Find best passage ──
    print("\n  4. Finding best passage ...")
    best_passage = _find_best_passage(reading)
    print(f"    Best passage: {best_passage['length']} tokens "
          f"starting at position {best_passage['start']}")
    preview = best_passage['passage_text'][:120]
    print(f"    Text: \"{preview}{'...' if len(best_passage['passage_text']) > 120 else ''}\"")

    # ── 5. Assemble recipe segments ──
    print("\n  5. Assembling recipe segments ...")
    if zones:
        recipes = _assemble_recipe_segments(reading, zones, formula_glosses)
    else:
        # No segmentation available: treat entire folio as one segment
        full_text = ' '.join(
            r['english_gloss'] if r['english_gloss'] else f"[{r['decoded']}]"
            for r in reading
        )
        recipes = [{
            'formula_zone_id': None,
            'content_zone_id': 0,
            'formula_text': '(no segmentation)',
            'formula_glossed': '',
            'content_text': full_text[:500],
            'n_content_tokens': len(reading),
            'content_ingredients': [],
            'n_ingredients': 0,
        }]

    print(f"    Recipe segments: {len(recipes)}")
    for i, rec in enumerate(recipes):
        n_ing = rec['n_ingredients']
        ftext = rec['formula_text'][:50]
        ctext = rec['content_text'][:50]
        print(f"    [{i}] F: {ftext}{'...' if len(rec['formula_text']) > 50 else ''}")
        print(f"        C: {ctext}{'...' if len(rec['content_text']) > 50 else ''}")
        if n_ing > 0:
            ings = ', '.join(ci['ingredient'] for ci in rec['content_ingredients'])
            print(f"        Ingredients: {ings}")

    # ── 6. Build definitive summary ──
    print("\n  6. Building definitive summary ...")
    summary_text = _build_definitive_summary(
        reading, coverage, best_passage, recipes, pattern_info,
    )
    # Print the summary
    for line in summary_text.split('\n'):
        print(f"    {line}")

    # ── 7. Per-zone coverage breakdown ──
    print("\n  7. Per-zone coverage breakdown ...")
    zone_coverage = {}
    for z in zones:
        zone_tokens = [
            r for r in reading
            if z['start'] <= r['position'] < z['end']
        ]
        if not zone_tokens:
            continue
        zcov = Counter(r['confidence'] for r in zone_tokens)
        n_z = len(zone_tokens)
        zone_coverage[str(z['zone_id'])] = {
            'zone_type': z['zone_type'],
            'n_tokens': n_z,
            'GREEN': zcov.get('GREEN', 0),
            'YELLOW': zcov.get('YELLOW', 0),
            'ORANGE': zcov.get('ORANGE', 0),
            'RED': zcov.get('RED', 0),
            'glossed_pct': round(
                (zcov.get('GREEN', 0) + zcov.get('YELLOW', 0)
                 + zcov.get('ORANGE', 0)) / max(n_z, 1),
                4,
            ),
        }
        print(f"    Zone {z['zone_id']:2d} [{z['zone_type']:8s}]: "
              f"G={zcov.get('GREEN', 0)} Y={zcov.get('YELLOW', 0)} "
              f"O={zcov.get('ORANGE', 0)} R={zcov.get('RED', 0)} "
              f"({zone_coverage[str(z['zone_id'])]['glossed_pct']:.0%} glossed)")

    # ── 8. Full reading text ──
    full_reading_parts = []
    for r in reading:
        conf = r['confidence']
        if conf == 'GREEN':
            full_reading_parts.append(r['english_gloss'])
        elif conf == 'YELLOW':
            full_reading_parts.append(f"({r['english_gloss']})")
        elif conf == 'ORANGE':
            full_reading_parts.append(f"[{r['english_gloss']}?]")
        else:
            full_reading_parts.append('[...]')
    full_reading_text = ' '.join(full_reading_parts)

    # ── 9. Save ──
    elapsed = time.time() - t0

    output = {
        'n_tokens': len(reading),
        'coverage': coverage,
        'best_passage': best_passage,
        'n_recipe_segments': len(recipes),
        'recipe_segments': recipes,
        'zone_coverage': zone_coverage,
        'reading': reading,
        'full_reading_text': full_reading_text[:3000],
        'summary_text': summary_text,
        'pattern_info': pattern_info,
        'formula_glosses': formula_glosses,
        'interpretation': (
            f"f57v complete reading: {len(reading)} tokens. "
            f"Coverage: {coverage['green_pct']:.0%} GREEN, "
            f"{coverage['yellow_pct']:.0%} YELLOW, "
            f"{coverage['orange_pct']:.0%} ORANGE, "
            f"{coverage['red_pct']:.0%} RED. "
            f"Best passage: {best_passage['length']} consecutive tokens. "
            f"{len(recipes)} recipe segments assembled."
        ),
        'runtime_seconds': round(elapsed, 1),
    }

    out_path = os.path.join(rd, 'f57v_complete_reading.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(output), f, indent=2)
    print(f"\n  Saved -> {out_path} ({elapsed:.1f}s)")
