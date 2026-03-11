"""
Step 40.10 – Folio Text Reconstruction
========================================
For the top SIGNAL folios, produce a partial Venetian reading by inserting
lexicon translations at SIGNAL positions and marking gaps.

Dependency chain:
    syllable_lexicon.json  (Step 40.9)
    merged_signal.json     (Step 38.3)
    merged_folio.json      (Step 38.8)
        → folio_reconstruction.json  (this step)
"""

import json
import os
import time
from typing import Any, Dict, List, Optional

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
# Core: Folio reconstruction
# ---------------------------------------------------------------------------

def _build_folio_reading(
    folio: str,
    folio_indices: List[int],
    decoded_tokens: List[str],
    classifications: List[str],
    lexicon: Dict[str, Dict],
) -> Dict:
    """Build a dual-layer annotated reading for one folio."""
    decoded_line = []
    gloss_line = []
    confidence_line = []
    n_glossed = 0
    n_consecutive = 0
    max_consecutive = 0
    current_run = 0

    for idx in folio_indices:
        word = decoded_tokens[idx] if idx < len(decoded_tokens) else ''
        cls = classifications[idx] if idx < len(classifications) else 'SHARED_MISS'

        decoded_line.append(word)

        if cls == 'SIGNAL' and word in lexicon:
            entry = lexicon[word]
            gloss = entry.get('english_gloss', '???')
            conf = entry.get('confidence', 'LOW')
            gloss_line.append(gloss)
            confidence_line.append(conf)
            n_glossed += 1
            current_run += 1
            if current_run > max_consecutive:
                max_consecutive = current_run
        elif cls in ('SIGNAL', 'SHARED_HIT') and word:
            # Hit but not in lexicon — mark as known but unglossed
            gloss_line.append(f'[{word}]')
            confidence_line.append('MINIMAL')
            n_glossed += 1
            current_run += 1
            if current_run > max_consecutive:
                max_consecutive = current_run
        else:
            gloss_line.append('___')
            confidence_line.append('UNKNOWN')
            current_run = 0

    n_tokens = len(folio_indices)
    coverage = n_glossed / n_tokens if n_tokens > 0 else 0.0

    # Coherence: fraction of consecutive pairs where both are glossed
    n_pairs = max(n_tokens - 1, 1)
    coherent_pairs = 0
    for i in range(len(gloss_line) - 1):
        if gloss_line[i] != '___' and gloss_line[i + 1] != '___':
            coherent_pairs += 1
    coherence = coherent_pairs / n_pairs

    # Detect recipe patterns: sequences of VERB-NOUN or PREP-NOUN
    recipe_patterns = 0
    for i in range(len(gloss_line) - 1):
        g1 = gloss_line[i]
        g2 = gloss_line[i + 1]
        if g1 != '___' and g2 != '___':
            # Simple heuristic: "strain" + word, "of" + word, etc.
            if any(v in g1.lower() for v in ['strain', 'make', 'take', 'boil']):
                recipe_patterns += 1
            if any(p in g1.lower() for p in ['of', 'from', 'with']):
                recipe_patterns += 1

    # Build best reading string
    reading_parts = []
    for g in gloss_line:
        if g == '___':
            reading_parts.append('[...]')
        else:
            reading_parts.append(g)
    best_reading = ' '.join(reading_parts)

    return {
        'folio': folio,
        'n_tokens': n_tokens,
        'n_glossed': n_glossed,
        'coverage': round(coverage, 4),
        'coherence': round(coherence, 4),
        'max_consecutive_glossed': max_consecutive,
        'recipe_patterns': recipe_patterns,
        'decoded_line': decoded_line,
        'gloss_line': gloss_line,
        'confidence_line': confidence_line,
        'best_reading': best_reading,
    }


def _cross_folio_consistency(readings: List[Dict], lexicon: Dict) -> Dict:
    """Check if the same signal words receive the same glosses across folios."""
    word_glosses: Dict[str, set] = {}
    for reading in readings:
        for decoded, gloss in zip(reading['decoded_line'], reading['gloss_line']):
            if gloss != '___' and not gloss.startswith('['):
                if decoded not in word_glosses:
                    word_glosses[decoded] = set()
                word_glosses[decoded].add(gloss)

    consistent = sum(1 for glosses in word_glosses.values() if len(glosses) == 1)
    total = len(word_glosses)
    return {
        'n_words_tested': total,
        'n_consistent': consistent,
        'consistency_rate': round(consistent / total, 4) if total > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_folio_reconstruction() -> None:
    """Step 40.10: Folio Text Reconstruction."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 40.10: Folio Text Reconstruction")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load inputs ──
    print("\n  1. Loading inputs …")
    lex_data = _safe_load(os.path.join(rd, 'syllable_lexicon.json'))
    merged_signal = _safe_load(os.path.join(rd, 'merged_signal.json'))
    merged_folio = _safe_load(os.path.join(rd, 'merged_folio.json'))

    lexicon = lex_data.get('syllable_lexicon', {})
    decoded_tokens = merged_signal.get('token_decoded', [])
    classifications = merged_signal.get('token_classifications', [])
    token_folios = merged_signal.get('token_folios', [])
    print(f"    Lexicon entries: {len(lexicon)}")
    print(f"    Tokens: {len(decoded_tokens):,}")

    # ── 2. Identify top SIGNAL folios ──
    print("\n  2. Identifying top SIGNAL folios …")
    folio_ranking = merged_folio.get('folio_ranking',
                                      merged_signal.get('folio_ranking', []))
    top_folios = [f['folio'] for f in folio_ranking[:6]]
    if not top_folios:
        # Fallback: use most common folios
        from collections import Counter
        fc = Counter(token_folios)
        top_folios = [f for f, _ in fc.most_common(6)]
    print(f"    Top folios: {top_folios}")

    # Build folio → token indices
    folio_indices: Dict[str, List[int]] = {}
    for i, f in enumerate(token_folios):
        if f not in folio_indices:
            folio_indices[f] = []
        folio_indices[f].append(i)

    # ── 3. Build readings ──
    print("\n  3. Building folio readings …")
    readings = []
    for folio in top_folios:
        indices = folio_indices.get(folio, [])
        if not indices:
            continue
        reading = _build_folio_reading(
            folio, indices, decoded_tokens, classifications, lexicon,
        )
        readings.append(reading)
        print(f"    {folio}: {reading['n_tokens']} tokens, "
              f"coverage {reading['coverage']:.2%}, "
              f"coherence {reading['coherence']:.2%}, "
              f"max run {reading['max_consecutive_glossed']}")

    # ── 4. Cross-folio consistency ──
    print("\n  4. Cross-folio consistency …")
    consistency = _cross_folio_consistency(readings, lexicon)
    print(f"    {consistency['n_consistent']}/{consistency['n_words_tested']} "
          f"consistent ({consistency['consistency_rate']:.2%})")

    # ── 5. Medical coherence summary ──
    print("\n  5. Medical coherence:")
    best_reading = max(readings, key=lambda r: r['coverage'] * r['coherence']) if readings else None
    if best_reading:
        print(f"    Best folio: {best_reading['folio']}")
        print(f"    Coverage: {best_reading['coverage']:.2%}")
        print(f"    Coherence: {best_reading['coherence']:.2%}")
        print(f"    Recipe patterns: {best_reading['recipe_patterns']}")
        # Print first 100 chars of reading
        reading_preview = best_reading['best_reading'][:200]
        print(f"    Reading preview: {reading_preview}")

    # ── 6. Save ──
    elapsed = time.time() - t0

    output = {
        'n_folios_with_readings': len(readings),
        'top_folios': top_folios,
        'folio_readings': readings,
        'cross_folio_consistency': consistency,
        'best_coverage_folio': best_reading['folio'] if best_reading else '',
        'best_coverage_pct': best_reading['coverage'] if best_reading else 0.0,
        'best_coherence_pct': best_reading['coherence'] if best_reading else 0.0,
        'sample_reading': best_reading['best_reading'][:500] if best_reading else '',
        'runtime_seconds': round(elapsed, 1),
    }

    out_path = os.path.join(rd, 'folio_reconstruction.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(output), f, indent=2)
    print(f"\n  Saved → {out_path} ({elapsed:.1f}s)")
