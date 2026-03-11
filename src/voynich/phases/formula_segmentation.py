"""
Step 41.9 – Formula Segmentation of f57v
=========================================
Segment f57v into FORMULA and CONTENT zones based on the 7-token repeating
pattern discovered in Phase 40.11.  The pattern "ra ne di ne hi fa de"
appears 4 times at 14-token intervals (positions 48, 62, 76, 90).

Dependency chain:
    f57v_reading.json          (Step 40.11)
    syllable_lexicon.json      (Step 40.9)
        → formula_segmentation.json  (this step)
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
# Core: pattern detection
# ---------------------------------------------------------------------------

def _find_best_repeating_pattern(
    decoded: List[str],
    min_len: int = 5,
    max_len: int = 7,
    min_repeats: int = 3,
) -> Optional[Dict]:
    """Find the most common subsequence of length min_len..max_len that
    appears at least min_repeats times.  Prefer longer patterns, then
    higher counts."""
    n = len(decoded)
    best = None

    for plen in range(max_len, min_len - 1, -1):
        pattern_counts: Dict[Tuple[str, ...], List[int]] = {}
        for i in range(n - plen + 1):
            key = tuple(decoded[i:i + plen])
            pattern_counts.setdefault(key, []).append(i)

        for pat, positions in pattern_counts.items():
            if len(positions) < min_repeats:
                continue
            # Check for regular spacing (all intervals equal)
            intervals = [positions[j + 1] - positions[j]
                         for j in range(len(positions) - 1)]
            regular = len(set(intervals)) == 1
            score = len(positions) * plen + (10 if regular else 0)
            if best is None or score > best['score']:
                best = {
                    'pattern': list(pat),
                    'pattern_str': ' '.join(pat),
                    'length': plen,
                    'count': len(positions),
                    'positions': positions,
                    'intervals': intervals,
                    'regular_spacing': regular,
                    'score': score,
                }

    return best


def _build_formula_mask(
    n_tokens: int,
    pattern_positions: List[int],
    pattern_length: int,
) -> List[bool]:
    """Return a boolean list: True where a token is part of a formula
    occurrence."""
    mask = [False] * n_tokens
    for start in pattern_positions:
        for offset in range(pattern_length):
            idx = start + offset
            if idx < n_tokens:
                mask[idx] = True
    return mask


def _segment_into_zones(
    decoded: List[str],
    classifications: List[str],
    glosses: List[str],
    formula_mask: List[bool],
    lexicon: Dict[str, Dict],
) -> List[Dict]:
    """Segment the token sequence into HEADER, FORMULA, and CONTENT zones."""
    n = len(decoded)
    zones: List[Dict] = []

    # Walk through tokens, grouping consecutive tokens of the same type
    i = 0
    zone_id = 0
    # Determine where the first formula starts
    first_formula = None
    for j in range(n):
        if formula_mask[j]:
            first_formula = j
            break

    while i < n:
        if formula_mask[i]:
            # FORMULA zone
            start = i
            while i < n and formula_mask[i]:
                i += 1
            end = i  # exclusive
            zone_tokens = decoded[start:end]
            zone_cls = classifications[start:end]
            zone_glosses = glosses[start:end]
            n_signal = sum(1 for c in zone_cls if c == 'SIGNAL')
            n_glossable = sum(1 for g in zone_glosses if g not in ('___', '???'))
            zones.append({
                'zone_id': zone_id,
                'zone_type': 'FORMULA',
                'start': start,
                'end': end,
                'n_tokens': end - start,
                'decoded_tokens': zone_tokens,
                'signal_rate': round(n_signal / max(end - start, 1), 4),
                'n_glossable': n_glossable,
                'glosses': zone_glosses,
            })
            zone_id += 1
        else:
            # Non-formula zone
            start = i
            while i < n and not formula_mask[i]:
                i += 1
            end = i
            zone_tokens = decoded[start:end]
            zone_cls = classifications[start:end]
            zone_glosses = glosses[start:end]
            n_signal = sum(1 for c in zone_cls if c == 'SIGNAL')
            n_glossable = sum(1 for g in zone_glosses if g not in ('___', '???'))

            # Decide HEADER vs CONTENT
            if first_formula is not None and start < first_formula:
                ztype = 'HEADER'
            else:
                ztype = 'CONTENT'

            zones.append({
                'zone_id': zone_id,
                'zone_type': ztype,
                'start': start,
                'end': end,
                'n_tokens': end - start,
                'decoded_tokens': zone_tokens,
                'signal_rate': round(n_signal / max(end - start, 1), 4),
                'n_glossable': n_glossable,
                'glosses': zone_glosses,
            })
            zone_id += 1

    return zones


def _compute_content_zone_similarity(zones: List[Dict]) -> Dict:
    """Measure vocabulary overlap between CONTENT zones."""
    content_zones = [z for z in zones if z['zone_type'] == 'CONTENT']
    if len(content_zones) < 2:
        return {
            'n_content_zones': len(content_zones),
            'pairwise_jaccard': [],
            'mean_jaccard': 0.0,
            'shared_vocabulary': [],
            'zone_unique_words': {},
        }

    zone_vocabs = []
    for z in content_zones:
        vocab = set(z['decoded_tokens'])
        zone_vocabs.append((z['zone_id'], vocab))

    # Pairwise Jaccard
    pairwise = []
    for i in range(len(zone_vocabs)):
        for j in range(i + 1, len(zone_vocabs)):
            zid_a, va = zone_vocabs[i]
            zid_b, vb = zone_vocabs[j]
            inter = len(va & vb)
            union = len(va | vb)
            jac = inter / max(union, 1)
            pairwise.append({
                'zone_a': zid_a,
                'zone_b': zid_b,
                'jaccard': round(jac, 4),
                'shared_words': sorted(va & vb),
            })

    mean_jac = sum(p['jaccard'] for p in pairwise) / max(len(pairwise), 1)

    # Global shared vocabulary (appears in 2+ content zones)
    word_zone_count: Dict[str, int] = Counter()
    for _, vocab in zone_vocabs:
        for w in vocab:
            word_zone_count[w] += 1
    shared = sorted(w for w, c in word_zone_count.items() if c >= 2)

    # Zone-unique words
    zone_unique = {}
    all_words = set()
    for _, vocab in zone_vocabs:
        all_words |= vocab
    for zid, vocab in zone_vocabs:
        others = set()
        for zid2, vocab2 in zone_vocabs:
            if zid2 != zid:
                others |= vocab2
        unique = sorted(vocab - others)
        zone_unique[str(zid)] = unique

    return {
        'n_content_zones': len(content_zones),
        'pairwise_jaccard': pairwise,
        'mean_jaccard': round(mean_jac, 4),
        'shared_vocabulary': shared,
        'zone_unique_words': zone_unique,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_formula_segmentation() -> None:
    """Step 41.9: Segment f57v by repeating formula pattern."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 41.9: Formula Segmentation of f57v")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load inputs ──
    print("\n  1. Loading inputs ...")
    f57v_data = _safe_load(os.path.join(rd, 'f57v_reading.json'))
    lex_data = _safe_load(os.path.join(rd, 'syllable_lexicon.json'))

    if not f57v_data:
        print("    ERROR: f57v_reading.json not found. Cannot proceed.")
        output = {'error': 'f57v_reading.json not found', 'runtime_seconds': 0.0}
        out_path = os.path.join(rd, 'formula_segmentation.json')
        with open(out_path, 'w') as f:
            json.dump(output, f, indent=2)
        return

    line_by_line = f57v_data.get('line_by_line', [])
    n_tokens = len(line_by_line)
    lexicon = lex_data.get('syllable_lexicon', {})

    decoded = [t['decoded'] for t in line_by_line]
    classifications = [t['classification'] for t in line_by_line]
    glosses = [t['english_gloss'] for t in line_by_line]

    print(f"    f57v tokens: {n_tokens}")
    print(f"    Lexicon entries: {len(lexicon)}")

    # ── 2. Find the repeating pattern ──
    print("\n  2. Detecting repeating formula pattern ...")
    pattern_info = _find_best_repeating_pattern(decoded)

    if pattern_info is None:
        print("    WARNING: No repeating pattern found (min 5 tokens, 3 repeats).")
        # Fallback: use the stored repetitions from f57v_reading.json
        stored_reps = f57v_data.get('repetitions', [])
        if stored_reps:
            top = stored_reps[0]
            pattern_info = {
                'pattern': top['pattern'].split(),
                'pattern_str': top['pattern'],
                'length': top['length'],
                'count': top['count'],
                'positions': top['positions'],
                'intervals': [top['positions'][j + 1] - top['positions'][j]
                              for j in range(len(top['positions']) - 1)],
                'regular_spacing': len(set(
                    top['positions'][j + 1] - top['positions'][j]
                    for j in range(len(top['positions']) - 1)
                )) == 1,
                'score': top['count'] * top['length'],
            }
            print(f"    Fallback: used stored repetition data.")

    if pattern_info:
        print(f"    Pattern: '{pattern_info['pattern_str']}'")
        print(f"    Length: {pattern_info['length']}, Count: {pattern_info['count']}")
        print(f"    Positions: {pattern_info['positions']}")
        print(f"    Intervals: {pattern_info['intervals']}")
        print(f"    Regular spacing: {pattern_info['regular_spacing']}")
    else:
        print("    No pattern found even in stored data. Using empty segmentation.")
        pattern_info = {
            'pattern': [], 'pattern_str': '', 'length': 0, 'count': 0,
            'positions': [], 'intervals': [], 'regular_spacing': False,
            'score': 0,
        }

    # ── 3. Build formula mask and segment ──
    print("\n  3. Building formula mask and segmenting ...")
    formula_mask = _build_formula_mask(
        n_tokens,
        pattern_info['positions'],
        pattern_info['length'],
    )
    n_formula_tokens = sum(formula_mask)
    n_content_tokens = n_tokens - n_formula_tokens
    print(f"    FORMULA tokens: {n_formula_tokens}")
    print(f"    Non-FORMULA tokens: {n_content_tokens}")

    # ── 4. Segment into zones ──
    print("\n  4. Segmenting into zones ...")
    zones = _segment_into_zones(decoded, classifications, glosses,
                                formula_mask, lexicon)

    zone_type_counts = Counter(z['zone_type'] for z in zones)
    for ztype, cnt in sorted(zone_type_counts.items()):
        print(f"    {ztype}: {cnt} zones")

    for z in zones:
        ztype = z['zone_type']
        tokens_preview = ' '.join(z['decoded_tokens'][:8])
        if len(z['decoded_tokens']) > 8:
            tokens_preview += ' ...'
        print(f"    Zone {z['zone_id']:2d} [{ztype:8s}] pos {z['start']:3d}-{z['end']:3d} "
              f"({z['n_tokens']:2d} tok, sig={z['signal_rate']:.2f}, "
              f"gloss={z['n_glossable']:2d}) | {tokens_preview}")

    # ── 5. Gloss the formula pattern ──
    print("\n  5. Glossing formula pattern ...")
    formula_glosses = []
    for word in pattern_info.get('pattern', []):
        entry = lexicon.get(word, {})
        gloss = entry.get('english_gloss', '???')
        formula_glosses.append(gloss)
    formula_gloss_str = ' | '.join(formula_glosses) if formula_glosses else '(none)'
    print(f"    Formula: {pattern_info.get('pattern_str', '')}")
    print(f"    Glossed: {formula_gloss_str}")

    # ── 6. Content zone similarity ──
    print("\n  6. Computing content zone similarity ...")
    similarity = _compute_content_zone_similarity(zones)
    print(f"    Content zones: {similarity['n_content_zones']}")
    print(f"    Mean Jaccard: {similarity['mean_jaccard']:.4f}")
    print(f"    Shared vocabulary ({len(similarity['shared_vocabulary'])} words): "
          f"{', '.join(similarity['shared_vocabulary'][:15])}")

    for pair in similarity['pairwise_jaccard'][:5]:
        print(f"      Zone {pair['zone_a']} vs {pair['zone_b']}: "
              f"J={pair['jaccard']:.3f} ({len(pair['shared_words'])} shared)")

    # ── 7. Summary statistics ──
    print("\n  7. Summary statistics ...")
    content_zones = [z for z in zones if z['zone_type'] == 'CONTENT']
    header_zones = [z for z in zones if z['zone_type'] == 'HEADER']
    formula_zones = [z for z in zones if z['zone_type'] == 'FORMULA']

    content_signal_rates = [z['signal_rate'] for z in content_zones]
    formula_signal_rates = [z['signal_rate'] for z in formula_zones]

    mean_content_signal = (sum(content_signal_rates) / max(len(content_signal_rates), 1))
    mean_formula_signal = (sum(formula_signal_rates) / max(len(formula_signal_rates), 1))

    content_sizes = [z['n_tokens'] for z in content_zones]
    mean_content_size = sum(content_sizes) / max(len(content_sizes), 1)

    print(f"    Mean content zone signal rate: {mean_content_signal:.4f}")
    print(f"    Mean formula zone signal rate: {mean_formula_signal:.4f}")
    print(f"    Mean content zone size: {mean_content_size:.1f} tokens")

    # ── 8. Save ──
    elapsed = time.time() - t0

    output = {
        'n_tokens': n_tokens,
        'pattern': pattern_info,
        'formula_gloss': formula_gloss_str,
        'formula_glosses': formula_glosses,
        'n_formula_tokens': n_formula_tokens,
        'n_content_tokens': n_content_tokens,
        'n_zones': len(zones),
        'zone_type_counts': dict(zone_type_counts),
        'zones': zones,
        'content_zone_similarity': similarity,
        'mean_content_signal_rate': round(mean_content_signal, 4),
        'mean_formula_signal_rate': round(mean_formula_signal, 4),
        'mean_content_zone_size': round(mean_content_size, 2),
        'n_header_zones': len(header_zones),
        'n_formula_zones': len(formula_zones),
        'n_content_zones': len(content_zones),
        'interpretation': (
            f"f57v segmented into {len(zones)} zones: "
            f"{len(header_zones)} HEADER, {len(formula_zones)} FORMULA, "
            f"{len(content_zones)} CONTENT. "
            f"Formula pattern '{pattern_info.get('pattern_str', '')}' repeats "
            f"{pattern_info.get('count', 0)}x at {pattern_info.get('intervals', [])} intervals. "
            f"Content zone similarity (Jaccard): {similarity['mean_jaccard']:.3f}."
        ),
        'runtime_seconds': round(elapsed, 1),
    }

    out_path = os.path.join(rd, 'formula_segmentation.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(output), f, indent=2)
    print(f"\n  Saved -> {out_path} ({elapsed:.1f}s)")
