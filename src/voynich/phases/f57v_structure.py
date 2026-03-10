"""
Step 37.11 – f57v Structural Analysis
========================================
Analyze f57v's text structure to determine what kind of content it encodes,
using the signal word distribution and the decoded token patterns.

Dependency chain:
    f57v_eva_analysis.json     (Step 37.10)
    signal_10k.json            (Step 36.2)
    decode_10k.json            (Step 36.1)
        → f57v_structure.json  (this step)
"""

import json
import os
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import load_corpus
from voynich.core.reference import load_reference_corpus


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
# Main
# ---------------------------------------------------------------------------

def run_f57v_structure() -> None:
    """Step 37.11: f57v Structural Analysis."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 37.11: f57v Structural Analysis")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load inputs ──
    print("\n  1. Loading inputs …")
    f57v_data = _safe_load(os.path.join(rd, 'f57v_eva_analysis.json'))
    signal_data = _safe_load(os.path.join(rd, 'signal_10k.json'))

    token_folios = signal_data.get('token_folios', [])
    token_decoded = signal_data.get('token_decoded', [])
    token_evas = signal_data.get('token_evas', [])
    token_classifications = signal_data.get('token_classifications', [])

    f57v_decoded_seq = f57v_data.get('f57v_decoded_sequence', [])
    diversity_verdict = f57v_data.get('diversity_verdict', '')

    # Get f57v positions
    f57v_positions = [i for i, f in enumerate(token_folios) if f == 'f57v']
    n_f57v = len(f57v_positions)
    print(f"     {n_f57v} tokens on f57v")
    print(f"     Diversity verdict: {diversity_verdict}")

    # ── 2. Signal density by line ──
    print("  2. Signal density map …")
    # Group f57v tokens into approximate "lines" (chunks of ~15-20 tokens)
    line_size = 15
    n_lines = (n_f57v + line_size - 1) // line_size
    line_densities = []

    for line_idx in range(n_lines):
        start = line_idx * line_size
        end = min(start + line_size, n_f57v)
        line_positions = f57v_positions[start:end]
        n_line = len(line_positions)
        n_signal = sum(1 for p in line_positions
                      if p < len(token_classifications) and
                      token_classifications[p] == 'SIGNAL')
        density = n_signal / n_line if n_line > 0 else 0.0
        line_densities.append({
            'line': line_idx + 1,
            'n_tokens': n_line,
            'n_signal': n_signal,
            'signal_density': round(density, 3),
        })

    # Find signal clusters vs gaps
    high_signal_lines = [ld for ld in line_densities if ld['signal_density'] > 0.5]
    low_signal_lines = [ld for ld in line_densities if ld['signal_density'] < 0.2]

    print(f"     {n_lines} lines analyzed")
    print(f"     {len(high_signal_lines)} high-signal lines (>50%)")
    print(f"     {len(low_signal_lines)} low-signal lines (<20%)")

    # ── 3. Gallows/determinative analysis ──
    print("  3. Gallows character analysis …")
    gallows = {'k', 't', 'p', 'f'}
    gallows_positions = []
    for idx, pos in enumerate(f57v_positions):
        if pos < len(token_evas):
            eva = token_evas[pos]
            # Check if token starts with or contains a gallows character
            for g in gallows:
                if g in eva:
                    gallows_positions.append({
                        'line_position': idx,
                        'token': eva,
                        'gallows_char': g,
                        'decoded': token_decoded[pos] if pos < len(token_decoded) else '',
                        'position_in_line': idx % line_size,
                    })

    n_gallows = len(gallows_positions)
    gallows_initial = sum(1 for gp in gallows_positions
                         if gp['position_in_line'] == 0)

    print(f"     {n_gallows} tokens contain gallows characters")
    print(f"     {gallows_initial} at line-initial position")

    # Check if gallows mark sections
    gallows_line_numbers = set(gp['line_position'] // line_size
                              for gp in gallows_positions)
    print(f"     Gallows appear on {len(gallows_line_numbers)}/{n_lines} lines")

    # ── 4. Cross-folio word frequency comparison ──
    print("  4. Cross-folio frequency comparison …")
    # f57v's most frequent decoded words
    f57v_word_freq = Counter()
    for pos in f57v_positions:
        if pos < len(token_decoded):
            f57v_word_freq[token_decoded[pos].lower()] += 1

    # Corpus-wide frequency for same words
    corpus_word_freq = Counter(w.lower() for w in token_decoded)
    n_corpus = len(token_decoded)

    cross_folio = []
    for word, f57v_count in f57v_word_freq.most_common(20):
        corpus_count = corpus_word_freq.get(word, 0)
        f57v_rate = f57v_count / n_f57v if n_f57v > 0 else 0.0
        corpus_rate = corpus_count / n_corpus if n_corpus > 0 else 0.0
        ratio = f57v_rate / corpus_rate if corpus_rate > 0 else float('inf')

        cross_folio.append({
            'word': word,
            'f57v_count': f57v_count,
            'f57v_rate': round(f57v_rate, 4),
            'corpus_count': corpus_count,
            'corpus_rate': round(corpus_rate, 4),
            'enrichment_ratio': round(ratio, 2),
        })

    # Words significantly enriched on f57v
    enriched = [cf for cf in cross_folio if cf['enrichment_ratio'] > 2.0]
    depleted = [cf for cf in cross_folio if cf['enrichment_ratio'] < 0.5]

    print(f"     {len(enriched)} words enriched (>2×) on f57v:")
    for cf in enriched[:5]:
        print(f"       {cf['word']:<12s} f57v={cf['f57v_rate']:.3f} "
              f"corpus={cf['corpus_rate']:.3f} ratio={cf['enrichment_ratio']:.1f}×")

    # ── 5. Recipe template matching ──
    print("  5. Recipe template matching …")
    ref = load_reference_corpus(languages=['latin'], verbose=False)
    ref_tokens = [w.lower() for w in ref.get_combined_tokens('latin') if len(w) >= 2]

    # Recipe keywords from pharmaceutical texts
    recipe_keywords = {'recipe', 'coque', 'cola', 'adde', 'misce', 'fiat',
                       'bibat', 'sume', 'tere', 'solve', 'distilla',
                       'aqua', 'herba', 'radix', 'folia', 'semen'}

    # Count recipe keywords in f57v vs rest of corpus
    f57v_recipe_count = sum(1 for pos in f57v_positions
                            if pos < len(token_decoded) and
                            token_decoded[pos].lower() in recipe_keywords)
    corpus_recipe_count = sum(1 for w in token_decoded if w.lower() in recipe_keywords)

    f57v_recipe_rate = f57v_recipe_count / n_f57v if n_f57v > 0 else 0.0
    corpus_recipe_rate = corpus_recipe_count / n_corpus if n_corpus > 0 else 0.0

    # Reference corpus recipe frequency
    ref_recipe_count = sum(1 for w in ref_tokens if w in recipe_keywords)
    ref_recipe_rate = ref_recipe_count / len(ref_tokens) if ref_tokens else 0.0

    # Correlation between f57v word frequencies and recipe-section frequencies
    # Build recipe-section word frequency profile from reference
    recipe_words_in_ref = Counter(w for w in ref_tokens if w in recipe_keywords)

    # Simple matching: does f57v's word profile correlate more with recipe sections?
    matches_recipe = f57v_recipe_rate > corpus_recipe_rate

    print(f"     Recipe keywords on f57v: {f57v_recipe_count} ({f57v_recipe_rate:.3%})")
    print(f"     Recipe keywords corpus:  {corpus_recipe_count} ({corpus_recipe_rate:.3%})")
    print(f"     Recipe keywords ref:     {ref_recipe_count} ({ref_recipe_rate:.3%})")
    print(f"     f57v matches recipe template: "
          f"{'YES' if matches_recipe else 'NO'}")

    # ── 6. Content interpretation ──
    print("  6. Content interpretation …")
    if diversity_verdict == 'TABLE_COLLAPSE':
        content_type = 'DIVERSE_TEXT_COLLAPSED'
        explanation = ("f57v contains diverse text that the triple table "
                       "reduces to repetitive syllables. The folio may contain "
                       "names, ingredients, or varied content.")
    elif diversity_verdict == 'GENUINE_REPETITION':
        if matches_recipe:
            content_type = 'PHARMACEUTICAL_RECIPES'
            explanation = ("f57v contains genuinely repetitive text matching "
                           "pharmaceutical recipe patterns (repeated imperatives "
                           "and ingredient lists).")
        else:
            content_type = 'REPETITIVE_LIST'
            explanation = ("f57v contains genuinely repetitive text — "
                           "possibly a list, table, or formulaic text.")
    else:
        content_type = 'MODERATE_REPETITION'
        explanation = ("f57v shows moderate repetition — "
                       "partially genuine, partially from table collapse.")

    print(f"     Content type: {content_type}")

    # ── 7. Save ──
    elapsed = time.time() - t0

    output = {
        'n_f57v_tokens': n_f57v,
        'diversity_verdict': diversity_verdict,
        'line_densities': line_densities,
        'n_high_signal_lines': len(high_signal_lines),
        'n_low_signal_lines': len(low_signal_lines),
        'n_gallows_tokens': n_gallows,
        'gallows_initial': gallows_initial,
        'gallows_line_coverage': f"{len(gallows_line_numbers)}/{n_lines}",
        'gallows_positions': gallows_positions[:20],
        'cross_folio_comparison': cross_folio,
        'n_enriched_words': len(enriched),
        'n_depleted_words': len(depleted),
        'f57v_recipe_rate': round(f57v_recipe_rate, 4),
        'corpus_recipe_rate': round(corpus_recipe_rate, 4),
        'matches_recipe_template': matches_recipe,
        'content_type': content_type,
        'content_explanation': explanation,
        'verdict': (
            f"f57v: {content_type}. "
            f"{len(enriched)} enriched words, "
            f"recipe match={'YES' if matches_recipe else 'NO'}. "
            f"{explanation}"
        ),
        'runtime_seconds': round(elapsed, 1),
    }

    out_path = os.path.join(rd, 'f57v_structure.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(output), f, indent=2)
    print(f"\n  Saved → {out_path} ({elapsed:.1f}s)")
