"""
Step 37.12 – Italian Reference Corpus
========================================
Build a Northern Italian reference corpus from the Anonimo Veneziano
cookbook manuscript, the only available period Venetian text in the data.

Dependency chain:
    data/reference/italian/anonimo_veneziano.txt  (raw text)
    combined_refine.json                           (Phase 15)
        → italian_corpus.json                      (this step)
"""

import json
import os
import re
import time
from collections import Counter
from typing import Any, Dict, List, Set, Tuple

from voynich.core._paths import data_dir, results_dir as _results_dir
from voynich.core.reference import load_reference_corpus
from voynich.core.stats import first_order_entropy


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
# Italian text processing
# ---------------------------------------------------------------------------

def _normalize_medieval_italian(text: str) -> str:
    """Normalize medieval Italian orthographic conventions."""
    # u/v normalization: 'v' as vowel → 'u', 'u' as consonant → 'v'
    # In medieval texts these are interchangeable; keep as-is for now
    # and handle during dictionary building
    text = text.lower()
    # Remove digits and Roman numerals used for recipe numbers
    text = re.sub(r'\b[IVXLCDM]+\b', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\b\d+\b', ' ', text)
    # Remove punctuation but keep apostrophes (d', l', etc.)
    text = re.sub(r"[^\w\s']", ' ', text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _tokenize_italian(text: str) -> List[str]:
    """Tokenize medieval Italian text."""
    normalized = _normalize_medieval_italian(text)
    tokens = normalized.split()
    # Filter: minimum length 2, alphabetic
    tokens = [t for t in tokens if len(t) >= 2 and t.isalpha()]
    return tokens


# Northern Italian sound changes (Latin → N. Italian)
_SOUND_CHANGES = [
    # Intervocalic voicing
    (r'(?<=[aeiou])p(?=[aeiou])', 'b'),
    (r'(?<=[aeiou])t(?=[aeiou])', 'd'),
    (r'(?<=[aeiou])c(?=[aeiou])', 'g'),
    # Geminate simplification
    (r'pp', 'p'),
    (r'tt', 't'),
    (r'cc', 'c'),
    (r'll', 'l'),
    (r'rr', 'r'),
    (r'ss', 's'),
    (r'nn', 'n'),
    (r'mm', 'm'),
    # Final consonant loss
    (r'[mst]$', ''),
    # Latin -tio → -zione
    (r'tio$', 'zione'),
    (r'tione$', 'zione'),
    # Latin -alis → -ale
    (r'alis$', 'ale'),
    # Latin -us → -o
    (r'us$', 'o'),
    # Latin -um → -o
    (r'um$', 'o'),
    # Latin -ae → -e
    (r'ae$', 'e'),
]


def _apply_sound_changes(word: str) -> Set[str]:
    """Apply Northern Italian sound changes to a Latin word."""
    variants = {word}
    current = word.lower()
    for pattern, replacement in _SOUND_CHANGES:
        new = re.sub(pattern, replacement, current)
        if new != current and len(new) >= 2:
            variants.add(new)
    # Also try cumulative changes
    cumulative = word.lower()
    for pattern, replacement in _SOUND_CHANGES:
        cumulative = re.sub(pattern, replacement, cumulative)
    if len(cumulative) >= 2:
        variants.add(cumulative)
    variants.discard(word)  # Remove original
    return variants


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_italian_corpus() -> None:
    """Step 37.12: Italian Reference Corpus."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 37.12: Italian Reference Corpus")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load Anonimo Veneziano ──
    print("\n  1. Loading Anonimo Veneziano …")
    italian_dir = data_dir('reference/italian')
    av_path = os.path.join(italian_dir, 'anonimo_veneziano.txt')

    if not os.path.exists(av_path):
        print(f"     ERROR: {av_path} not found")
        output = {
            'error': f'File not found: {av_path}',
            'verdict': 'FAIL: Italian reference corpus not available',
            'runtime_seconds': round(time.time() - t0, 1),
        }
        out_path = os.path.join(rd, 'italian_corpus.json')
        with open(out_path, 'w') as f:
            json.dump(output, f, indent=2)
        return

    with open(av_path, encoding='utf-8', errors='replace') as f:
        raw_text = f.read()

    n_lines = raw_text.count('\n') + 1
    print(f"     {n_lines} lines loaded ({len(raw_text)} chars)")

    # ── 2. Tokenize ──
    print("  2. Tokenizing …")
    tokens = _tokenize_italian(raw_text)
    print(f"     {len(tokens)} tokens extracted")

    # ── 3. Character statistics ──
    print("  3. Character statistics …")
    all_chars = ''.join(tokens)
    char_freq = Counter(all_chars)
    char_total = sum(char_freq.values())

    # Character bigrams
    char_bigrams: Counter = Counter()
    for token in tokens:
        for i in range(len(token) - 1):
            char_bigrams[token[i:i + 2]] += 1

    # Entropy
    h1 = first_order_entropy(' '.join(tokens))

    print(f"     {len(char_freq)} unique characters")
    print(f"     H1 = {h1:.4f} bits")
    print(f"     Top chars: {', '.join(f'{c}:{n}' for c, n in char_freq.most_common(10))}")

    # ── 4. Word statistics ──
    print("  4. Word statistics …")
    word_freq = Counter(tokens)
    n_types = len(word_freq)
    ttr = n_types / len(tokens) if tokens else 0.0

    # Zipf exponent (simplified log-log regression)
    import math
    ranks = list(range(1, n_types + 1))
    freqs = [c for _, c in word_freq.most_common()]
    if len(ranks) >= 10:
        log_ranks = [math.log(r) for r in ranks[:100]]
        log_freqs = [math.log(f) for f in freqs[:100]]
        n = len(log_ranks)
        mean_x = sum(log_ranks) / n
        mean_y = sum(log_freqs) / n
        num = sum((x - mean_x) * (y - mean_y) for x, y in zip(log_ranks, log_freqs))
        den = sum((x - mean_x) ** 2 for x in log_ranks)
        zipf_exp = -num / den if den > 0 else 0.0
    else:
        zipf_exp = 0.0

    # Word bigrams
    word_bigrams: Counter = Counter()
    for i in range(len(tokens) - 1):
        word_bigrams[(tokens[i], tokens[i + 1])] += 1

    print(f"     {n_types} types, TTR={ttr:.4f}")
    print(f"     Zipf exponent: {zipf_exp:.3f}")
    print(f"     Top words: {', '.join(f'{w}:{c}' for w, c in word_freq.most_common(15))}")

    # ── 5. Vocabulary overlap with Latin ──
    print("  5. Vocabulary overlap with Latin …")
    ref = load_reference_corpus(languages=['latin'], verbose=False)
    latin_tokens = [w.lower() for w in ref.get_combined_tokens('latin') if len(w) >= 2]
    latin_types = set(latin_tokens)
    italian_types = set(tokens)

    shared_vocab = latin_types & italian_types
    italian_only = italian_types - latin_types
    latin_only = latin_types - italian_types

    print(f"     Shared vocabulary: {len(shared_vocab)} words")
    print(f"     Italian only:     {len(italian_only)} words")
    print(f"     Latin only:       {len(latin_only)} words")
    print(f"     Shared examples:  {', '.join(sorted(shared_vocab)[:15])}")

    # ── 6. Generate synthetic Italian from Latin ──
    print("  6. Generating synthetic Italian from Latin …")
    synthetic_words: Set[str] = set()
    for word in latin_types:
        variants = _apply_sound_changes(word)
        synthetic_words |= variants

    # Combine: Anonimo Veneziano + synthetic
    combined_italian = italian_types | synthetic_words
    print(f"     {len(synthetic_words)} synthetic Italian words generated")
    print(f"     Combined Italian vocabulary: {len(combined_italian)} types")

    # ── 7. Corpus statistics comparison ──
    print("  7. Corpus statistics comparison …")
    latin_h1 = first_order_entropy(' '.join(latin_tokens[:len(tokens)]))

    comparison = {
        'latin': {
            'source': 'Circa Instans + De Viribus Herbarum',
            'n_tokens': len(latin_tokens),
            'n_types': len(latin_types),
            'h1': round(latin_h1, 4),
        },
        'italian_natural': {
            'source': 'Anonimo Veneziano',
            'n_tokens': len(tokens),
            'n_types': n_types,
            'h1': round(h1, 4),
            'zipf_exponent': round(zipf_exp, 3),
        },
        'italian_synthetic': {
            'source': 'Latin + sound changes',
            'n_types': len(synthetic_words),
        },
        'italian_combined': {
            'source': 'Natural + synthetic',
            'n_types': len(combined_italian),
        },
    }

    print(f"     Latin:            {len(latin_tokens)} tokens, "
          f"{len(latin_types)} types, H1={latin_h1:.4f}")
    print(f"     Italian natural:  {len(tokens)} tokens, "
          f"{n_types} types, H1={h1:.4f}")
    print(f"     Italian combined: {len(combined_italian)} types")

    # ── 8. Save ──
    elapsed = time.time() - t0

    output = {
        'source_file': av_path,
        'n_lines': n_lines,
        'n_raw_chars': len(raw_text),
        'n_tokens': len(tokens),
        'n_types': n_types,
        'type_token_ratio': round(ttr, 4),
        'h1': round(h1, 4),
        'zipf_exponent': round(zipf_exp, 3),
        'top_words': [{'word': w, 'count': c} for w, c in word_freq.most_common(100)],
        'top_char_bigrams': [{'bigram': b, 'count': c}
                             for b, c in char_bigrams.most_common(50)],
        'top_word_bigrams': [{'bigram': list(b), 'count': c}
                             for b, c in word_bigrams.most_common(50)],
        'shared_with_latin': sorted(list(shared_vocab)[:200]),
        'n_shared_vocab': len(shared_vocab),
        'n_italian_only': len(italian_only),
        'n_latin_only': len(latin_only),
        'n_synthetic_italian': len(synthetic_words),
        'n_combined_italian': len(combined_italian),
        'combined_italian_words': sorted(list(combined_italian)),
        'comparison': comparison,
        'verdict': (
            f"Italian corpus: {len(tokens)} tokens, {n_types} types, "
            f"H1={h1:.4f}. Shared with Latin: {len(shared_vocab)}. "
            f"Combined vocabulary: {len(combined_italian)} types."
        ),
        'runtime_seconds': round(elapsed, 1),
    }

    out_path = os.path.join(rd, 'italian_corpus.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(output), f, indent=2)
    print(f"\n  Saved → {out_path} ({elapsed:.1f}s)")
