"""
Step 40.11 – f57v Dedicated Venetian Reading
==============================================
Produce the most complete possible reading of f57v — the folio with 54.9%
SIGNAL rate, 9 Venetian verb forms, and a 58-token continuous chain.

Dependency chain:
    syllable_lexicon.json      (Step 40.9)
    folio_reconstruction.json  (Step 40.10)
    f57v_eva_analysis.json     (Step 37.10)
    merged_signal.json         (Step 38.3)
    merged_folio.json          (Step 38.8)
        → f57v_reading.json    (this step)
"""

import json
import os
import re
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
# Core: f57v reading
# ---------------------------------------------------------------------------

def _extract_f57v_tokens(
    decoded_tokens: List[str],
    token_folios: List[str],
    classifications: List[str],
) -> Tuple[List[str], List[str]]:
    """Extract f57v tokens and their classifications."""
    f57v_decoded = []
    f57v_cls = []
    for i in range(len(decoded_tokens)):
        if i < len(token_folios) and token_folios[i] == 'f57v':
            f57v_decoded.append(decoded_tokens[i])
            f57v_cls.append(classifications[i] if i < len(classifications) else 'SHARED_MISS')
    return f57v_decoded, f57v_cls


def _find_longest_chain(
    decoded: List[str],
    classifications: List[str],
) -> Tuple[int, int, List[str]]:
    """Find the longest consecutive chain of non-MISS tokens."""
    best_start = 0
    best_len = 0
    current_start = 0
    current_len = 0

    for i in range(len(classifications)):
        if classifications[i] in ('SIGNAL', 'SHARED_HIT'):
            if current_len == 0:
                current_start = i
            current_len += 1
            if current_len > best_len:
                best_len = current_len
                best_start = current_start
        else:
            current_len = 0

    chain = decoded[best_start:best_start + best_len]
    return best_start, best_len, chain


def _try_concatenation(
    chain: List[str],
    word_set: set,
) -> List[Dict]:
    """Try concatenating adjacent pairs/triples to form longer words."""
    results = []
    for i in range(len(chain) - 1):
        pair = chain[i] + chain[i + 1]
        if pair in word_set:
            results.append({
                'type': 'pair',
                'position': i,
                'parts': [chain[i], chain[i + 1]],
                'concatenated': pair,
            })

    for i in range(len(chain) - 2):
        triple = chain[i] + chain[i + 1] + chain[i + 2]
        if triple in word_set:
            results.append({
                'type': 'triple',
                'position': i,
                'parts': [chain[i], chain[i + 1], chain[i + 2]],
                'concatenated': triple,
            })

    return results


def _detect_repetitions(
    decoded: List[str],
    min_pattern: int = 3,
    min_repeats: int = 2,
) -> List[Dict]:
    """Detect repeating patterns in the decoded sequence."""
    patterns = []
    n = len(decoded)

    for pattern_len in range(min_pattern, min(8, n // 2 + 1)):
        for start in range(n - pattern_len):
            pattern = tuple(decoded[start:start + pattern_len])
            # Count occurrences
            count = 0
            positions = []
            for j in range(n - pattern_len + 1):
                if tuple(decoded[j:j + pattern_len]) == pattern:
                    count += 1
                    positions.append(j)
            if count >= min_repeats:
                # Check if we already found a superset pattern
                pattern_str = ' '.join(pattern)
                already_found = any(p['pattern'] == pattern_str for p in patterns)
                if not already_found:
                    patterns.append({
                        'pattern': pattern_str,
                        'length': pattern_len,
                        'count': count,
                        'positions': positions,
                    })

    # Sort by count × length (favor longer, more frequent patterns)
    patterns.sort(key=lambda p: p['count'] * p['length'], reverse=True)
    return patterns[:20]


def _build_line_by_line(
    decoded: List[str],
    classifications: List[str],
    lexicon: Dict[str, Dict],
) -> List[Dict]:
    """Build line-by-line annotated reading."""
    lines = []
    for i, word in enumerate(decoded):
        cls = classifications[i] if i < len(classifications) else 'SHARED_MISS'
        entry = lexicon.get(word, {})
        gloss = entry.get('english_gloss', '???') if cls == 'SIGNAL' else '___'
        pos = entry.get('part_of_speech', '') if cls == 'SIGNAL' else ''
        domain = entry.get('medical_domain', '') if cls == 'SIGNAL' else ''

        lines.append({
            'position': i,
            'decoded': word,
            'classification': cls,
            'english_gloss': gloss,
            'pos': pos,
            'domain': domain,
        })

    return lines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_f57v_reading() -> None:
    """Step 40.11: f57v Dedicated Venetian Reading."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 40.11: f57v Dedicated Venetian Reading")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load inputs ──
    print("\n  1. Loading inputs …")
    lex_data = _safe_load(os.path.join(rd, 'syllable_lexicon.json'))
    f57v_eva = _safe_load(os.path.join(rd, 'f57v_eva_analysis.json'))
    merged_signal = _safe_load(os.path.join(rd, 'merged_signal.json'))
    merged_folio = _safe_load(os.path.join(rd, 'merged_folio.json'))
    merged_dict = _safe_load(os.path.join(rd, 'merged_dict.json'))

    lexicon = lex_data.get('syllable_lexicon', {})
    decoded_tokens = merged_signal.get('token_decoded', [])
    token_folios = merged_signal.get('token_folios', [])
    classifications = merged_signal.get('token_classifications', [])

    # Build word set for concatenation testing
    word_set = set(merged_dict.get('latin_10k_words', []))
    word_set.update(merged_dict.get('italian_10k_words', []))

    # ── 2. Extract f57v tokens ──
    print("\n  2. Extracting f57v tokens …")
    f57v_decoded, f57v_cls = _extract_f57v_tokens(
        decoded_tokens, token_folios, classifications,
    )
    n_tokens = len(f57v_decoded)
    n_signal = sum(1 for c in f57v_cls if c == 'SIGNAL')
    signal_rate = n_signal / n_tokens if n_tokens > 0 else 0.0
    print(f"    f57v tokens: {n_tokens}")
    print(f"    SIGNAL: {n_signal} ({signal_rate:.2%})")

    # ── 3. Find longest chain ──
    print("\n  3. Finding longest chain …")
    chain_start, chain_len, chain = _find_longest_chain(f57v_decoded, f57v_cls)
    print(f"    Longest chain: {chain_len} tokens starting at position {chain_start}")
    print(f"    Chain: {' '.join(chain[:20])}{'...' if chain_len > 20 else ''}")

    # ── 4. Try alternative segmentations ──
    print("\n  4. Testing concatenations …")
    concat_results = _try_concatenation(chain, word_set)
    print(f"    Concatenation hits: {len(concat_results)}")
    for cr in concat_results[:5]:
        print(f"      {'+'.join(cr['parts'])} = {cr['concatenated']}")

    # ── 5. Detect repetitions ──
    print("\n  5. Detecting repetitions …")
    repetitions = _detect_repetitions(f57v_decoded)
    print(f"    Repeating patterns found: {len(repetitions)}")
    for rep in repetitions[:5]:
        print(f"      '{rep['pattern']}' × {rep['count']} (len {rep['length']})")

    # ── 6. Build line-by-line reading ──
    print("\n  6. Building line-by-line reading …")
    line_by_line = _build_line_by_line(f57v_decoded, f57v_cls, lexicon)

    # Compute coverage
    n_glossed = sum(1 for l in line_by_line if l['english_gloss'] != '___')
    coverage = n_glossed / n_tokens if n_tokens > 0 else 0.0
    print(f"    Glossed: {n_glossed}/{n_tokens} ({coverage:.2%})")

    # Build best reading text
    reading_parts = []
    for l in line_by_line:
        if l['english_gloss'] == '___':
            reading_parts.append('[...]')
        elif l['english_gloss'] == '???':
            reading_parts.append(f'[{l["decoded"]}?]')
        else:
            reading_parts.append(l['english_gloss'])
    best_reading = ' '.join(reading_parts)

    # ── 7. Repetition interpretation ──
    print("\n  7. Repetition interpretation:")
    if repetitions:
        top_rep = repetitions[0]
        rep_words = top_rep['pattern'].split()
        rep_glosses = []
        for w in rep_words:
            entry = lexicon.get(w, {})
            rep_glosses.append(entry.get('english_gloss', '???'))
        print(f"    Most frequent pattern: '{top_rep['pattern']}'")
        print(f"    Glossed: {' | '.join(rep_glosses)}")
        print(f"    Interpretation: Possible formulaic recipe entry "
              f"(repeats {top_rep['count']}×)")
        rep_interpretation = {
            'pattern': top_rep['pattern'],
            'glossed': ' | '.join(rep_glosses),
            'count': top_rep['count'],
        }
    else:
        rep_interpretation = {}

    # ── 8. Coherence score ──
    coherent_pairs = 0
    for i in range(len(line_by_line) - 1):
        if (line_by_line[i]['english_gloss'] != '___' and
                line_by_line[i + 1]['english_gloss'] != '___'):
            coherent_pairs += 1
    coherence = coherent_pairs / max(n_tokens - 1, 1)

    # ── 9. Save ──
    elapsed = time.time() - t0

    output = {
        'n_tokens': n_tokens,
        'n_signal': n_signal,
        'signal_rate': round(signal_rate, 4),
        'n_glossed': n_glossed,
        'coverage_pct': round(coverage, 4),
        'coherence_score': round(coherence, 4),
        'longest_chain_length': chain_len,
        'longest_chain_start': chain_start,
        'chain_words': chain[:30],
        'concatenation_hits': concat_results,
        'repetitions': repetitions[:10],
        'repetition_interpretation': rep_interpretation,
        'line_by_line': line_by_line,
        'best_reading_text': best_reading[:1000],
        'interpretation': (
            f"f57v contains {n_tokens} tokens with {signal_rate:.0%} SIGNAL rate. "
            f"Longest chain: {chain_len} consecutive glossable tokens. "
            f"Coherence: {coherence:.2%}. "
            f"{'Formulaic structure detected.' if repetitions else 'No clear repetition.'}"
        ),
        'runtime_seconds': round(elapsed, 1),
    }

    out_path = os.path.join(rd, 'f57v_reading.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(output), f, indent=2)
    print(f"\n  Saved → {out_path} ({elapsed:.1f}s)")
