"""
Step 43.2 – Parameterized Tachygraphic Encoder
================================================
Build a flexible tachygraphic encoder that takes a plaintext string and
an encoding table parameter, producing an encoded character stream in a
synthetic "EVA-like" alphabet.

Dependency chain:
    results/combined_refine.json      (Phase 15: 25-triple table)
    results/tachygraphic_stroke.json  (Phase 19.5: sign families)
    results/modifier_integrate.json   (Phase 16: modifier chars)
    data/reference/latin/             (reference texts)
    data/reference/italian/           (Anonimo Veneziano)
        → tachygraphic_encoder.json   (this step)
"""

import json
import math
import os
import random
import time
from collections import Counter
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import load_corpus, build_eva_to_triple_lookup, tokenize_eva_chars
from voynich.core.reference import (
    EVA_VISUAL_COMPONENTS,
    load_reference_corpus,
)
from voynich.core.stats import (
    syllabify_latin,
    first_order_entropy,
    conditional_entropy,
)


# ---------------------------------------------------------------------------
# JSON helpers
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


def _safe_load(path: str) -> Dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


# ---------------------------------------------------------------------------
# Encoding table dataclass
# ---------------------------------------------------------------------------

@dataclass
class EncodingTable:
    """Parameterized encoding table for the tachygraphic encoder."""
    syllable_to_triple: Dict[str, str]     # syllable → triple_key
    triple_to_eva_chars: Dict[str, List[str]]  # triple_key → list of EVA chars
    modifier_chars: List[str]              # EVA chars that act as modifiers
    modifier_insertion_rate: float         # probability of inserting a modifier
    n_syllable_mappings: int
    n_triples: int


@dataclass
class TachygraphicEncoderResult:
    # Encoding table summary
    n_syllable_mappings: int
    n_triples: int
    n_modifier_chars: int
    n_sign_families: int
    syllable_to_triple: Dict[str, str]
    # Demonstration on sample text
    sample_plaintext: str
    sample_syllabified: List[List[str]]
    sample_encoded_tokens: List[str]
    sample_n_tokens: int
    sample_mean_token_length: float
    # Self-test: encode → decode round-trip
    round_trip_n_words: int
    round_trip_recovered: int
    round_trip_loss: float
    # Encoded text statistics (for comparison with Voynich fingerprint)
    encoded_h1: float
    encoded_h2: float
    encoded_mean_token_length: float
    encoded_n_unique_chars: int
    # Language comparison
    latin_sample_stats: Dict[str, float]
    italian_sample_stats: Dict[str, float]
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Core encoder functions
# ---------------------------------------------------------------------------

def _build_triple_to_eva(
) -> Dict[str, List[str]]:
    """Build mapping from triple_key to list of EVA chars with that triple."""
    triple_to_chars: Dict[str, List[str]] = {}
    for eva_char, components in EVA_VISUAL_COMPONENTS.items():
        triple_key = f"{components['first_stroke']},{components['last_stroke']},{components['glyph_class']}"
        triple_to_chars.setdefault(triple_key, []).append(eva_char)
    return triple_to_chars


def _invert_assignment(
    assignment: Dict[str, str],
) -> Dict[str, str]:
    """Invert Phase 15 assignment (triple→syllable) to (syllable→triple).

    When multiple triples map to the same syllable, keep the one with the
    most EVA characters (broadest encoding).
    """
    triple_to_eva = _build_triple_to_eva()
    syl_to_triple: Dict[str, str] = {}
    syl_to_count: Dict[str, int] = {}

    for triple_key, syllable in assignment.items():
        n_chars = len(triple_to_eva.get(triple_key, []))
        if syllable not in syl_to_triple or n_chars > syl_to_count.get(syllable, 0):
            syl_to_triple[syllable] = triple_key
            syl_to_count[syllable] = n_chars

    return syl_to_triple


def build_encoding_table(
    assignment: Dict[str, str],
    modifier_chars: List[str],
    modifier_insertion_rate: float = 0.15,
) -> EncodingTable:
    """Build an EncodingTable from Phase 15 assignment (inverted)."""
    syl_to_triple = _invert_assignment(assignment)
    triple_to_eva = _build_triple_to_eva()

    return EncodingTable(
        syllable_to_triple=syl_to_triple,
        triple_to_eva_chars=triple_to_eva,
        modifier_chars=modifier_chars,
        modifier_insertion_rate=modifier_insertion_rate,
        n_syllable_mappings=len(syl_to_triple),
        n_triples=len(triple_to_eva),
    )


def _syllable_to_nearest(
    syllable: str,
    known_syllables: Set[str],
) -> Optional[str]:
    """Find the nearest known syllable by edit distance."""
    if syllable in known_syllables:
        return syllable

    # Try CV truncation: take first consonant + first vowel
    vowels = set('aeiou')
    if len(syllable) >= 2:
        cv = syllable[:2]
        if cv in known_syllables:
            return cv

    # Try just the onset + 'a' (most common vowel)
    if len(syllable) >= 1 and syllable[0] not in vowels:
        for v in 'aeio':
            cv = syllable[0] + v
            if cv in known_syllables:
                return cv

    # Fallback: best edit distance
    best = None
    best_dist = float('inf')
    for ks in known_syllables:
        d = sum(1 for a, b in zip(syllable, ks) if a != b) + abs(len(syllable) - len(ks))
        if d < best_dist:
            best_dist = d
            best = ks
    return best


def encode_text(
    plaintext: str,
    table: EncodingTable,
    rng: Optional[random.Random] = None,
) -> Tuple[List[str], List[List[str]]]:
    """Encode plaintext through the tachygraphic table.

    Returns (encoded_tokens, syllabified_words) where encoded_tokens is a
    list of EVA-like token strings.
    """
    if rng is None:
        rng = random.Random(42)

    known_syllables = set(table.syllable_to_triple.keys())
    words = plaintext.lower().split()
    encoded_tokens: List[str] = []
    syllabified_words: List[List[str]] = []

    for word in words:
        # Strip non-alpha
        clean = ''.join(c for c in word if c.isalpha())
        if not clean:
            continue

        syllables = syllabify_latin(clean)
        syllabified_words.append(syllables)

        token_chars: List[str] = []
        for syl in syllables:
            # Map syllable to triple
            mapped_syl = _syllable_to_nearest(syl, known_syllables)
            if mapped_syl is None:
                continue

            triple_key = table.syllable_to_triple[mapped_syl]
            # Pick a random EVA char from this triple
            eva_chars = table.triple_to_eva_chars.get(triple_key, [])
            if eva_chars:
                chosen = rng.choice(eva_chars)
                token_chars.append(chosen)

            # Maybe insert a modifier char
            if table.modifier_chars and rng.random() < table.modifier_insertion_rate:
                mod = rng.choice(table.modifier_chars)
                token_chars.append(mod)

        if token_chars:
            encoded_tokens.append(''.join(token_chars))

    return encoded_tokens, syllabified_words


def decode_encoded_token(
    token: str,
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    modifier_set: Set[str],
) -> str:
    """Decode an encoded token back through the assignment table."""
    chars = tokenize_eva_chars(token)
    syllables = []
    for ch in chars:
        if ch in modifier_set:
            continue
        triple = eva_to_triple.get(ch)
        if triple:
            syl = assignment.get(triple, '?')
            syllables.append(syl)
    return ''.join(syllables)


def compute_encoded_stats(
    encoded_tokens: List[str],
) -> Dict[str, float]:
    """Compute basic statistics on encoded token stream."""
    if not encoded_tokens:
        return {'h1': 0.0, 'h2': 0.0, 'mean_length': 0.0, 'n_unique_chars': 0}

    text = ' '.join(encoded_tokens)
    lengths = [len(t) for t in encoded_tokens]
    unique_chars = set()
    for t in encoded_tokens:
        for ch in tokenize_eva_chars(t):
            unique_chars.add(ch)

    return {
        'h1': first_order_entropy(text),
        'h2': conditional_entropy(text, order=1),
        'mean_length': float(np.mean(lengths)) if lengths else 0.0,
        'n_unique_chars': len(unique_chars),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_tachygraphic_encoder() -> None:
    """Step 43.2: build parameterized tachygraphic encoder."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 43.2: Parameterized Tachygraphic Encoder")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load inputs ──
    print("\n  1. Loading inputs …")

    combined = _safe_load(os.path.join(rd, 'combined_refine.json'))
    assignment = combined.get('best_assignment', {})
    print(f"     Phase 15 assignment: {len(assignment)} triples")

    tachy = _safe_load(os.path.join(rd, 'tachygraphic_stroke.json'))
    sign_families = tachy.get('sign_families', [])
    n_families = tachy.get('n_families', 0)
    print(f"     Sign families: {n_families}")

    mod_data = _safe_load(os.path.join(rd, 'modifier_integrate.json'))
    modifier_chars = mod_data.get('modifier_chars', [])
    print(f"     Modifier chars: {len(modifier_chars)}")

    # ── 2. Build encoding table ──
    print("\n  2. Building encoding table …")
    table = build_encoding_table(assignment, modifier_chars)
    print(f"     Syllable mappings: {table.n_syllable_mappings}")
    print(f"     Triples with EVA chars: {table.n_triples}")
    print(f"     Syllable→triple: {dict(list(table.syllable_to_triple.items())[:5])} …")

    # ── 3. Load reference texts ──
    print("\n  3. Loading reference texts …")
    try:
        ref = load_reference_corpus(languages=['latin'], verbose=False)
        latin_tokens = ref.get_combined_tokens('latin')
        latin_text = ' '.join(latin_tokens[:5000])
    except Exception:
        latin_text = ''
        latin_tokens = []
    print(f"     Latin tokens: {len(latin_tokens)}")

    try:
        ref_it = load_reference_corpus(languages=['italian'], verbose=False)
        italian_tokens = ref_it.get_combined_tokens('italian')
        italian_text = ' '.join(italian_tokens[:5000])
    except Exception:
        italian_text = ''
        italian_tokens = []
    print(f"     Italian tokens: {len(italian_tokens)}")

    # ── 4. Encode Latin sample ──
    print("\n  4. Encoding Latin sample (first 500 words) …")
    sample_text = ' '.join(latin_tokens[:500]) if latin_tokens else "circa instans de simplicibus medicinis"
    encoded_latin, syl_latin = encode_text(sample_text, table)
    latin_stats = compute_encoded_stats(encoded_latin)
    print(f"     Encoded {len(encoded_latin)} tokens")
    print(f"     Mean token length: {latin_stats['mean_length']:.2f}")
    print(f"     H1: {latin_stats['h1']:.3f}, H2: {latin_stats['h2']:.3f}")
    print(f"     Unique chars: {latin_stats['n_unique_chars']}")
    if encoded_latin:
        print(f"     Sample: {' '.join(encoded_latin[:10])} …")

    # ── 5. Encode Italian sample ──
    print("\n  5. Encoding Italian sample (first 500 words) …")
    sample_it = ' '.join(italian_tokens[:500]) if italian_tokens else "anonimo veneziano"
    encoded_italian, syl_italian = encode_text(sample_it, table)
    italian_stats = compute_encoded_stats(encoded_italian)
    print(f"     Encoded {len(encoded_italian)} tokens")
    print(f"     Mean token length: {italian_stats['mean_length']:.2f}")
    print(f"     H1: {italian_stats['h1']:.3f}")

    # ── 6. Round-trip test ──
    print("\n  6. Round-trip test (encode → decode) …")
    eva_to_triple = build_eva_to_triple_lookup()
    modifier_set = set(modifier_chars)

    rt_words = sample_text.lower().split()[:100]
    rt_encoded, rt_syls = encode_text(' '.join(rt_words), table, rng=random.Random(0))
    n_recovered = 0
    for i, (enc_tok, orig_syls) in enumerate(zip(rt_encoded, rt_syls)):
        decoded = decode_encoded_token(enc_tok, assignment, eva_to_triple, modifier_set)
        original = ''.join(orig_syls)
        if decoded == original:
            n_recovered += 1

    rt_n = min(len(rt_encoded), len(rt_syls))
    rt_loss = 1.0 - (n_recovered / rt_n) if rt_n > 0 else 1.0
    print(f"     {n_recovered}/{rt_n} words recovered ({(1 - rt_loss) * 100:.1f}%)")
    print(f"     Round-trip loss: {rt_loss:.3f}")

    # ── 7. Voynich comparison ──
    print("\n  7. Voynich comparison …")
    corpus = load_corpus(verbose=False)
    voynich_tokens = corpus.get_tokens(paragraph_only=True)
    voynich_text = ' '.join(voynich_tokens[:5000])
    v_h1 = first_order_entropy(voynich_text)
    v_h2 = conditional_entropy(voynich_text, order=1)
    v_lengths = [len(t) for t in voynich_tokens]
    v_mean_len = float(np.mean(v_lengths)) if v_lengths else 0.0
    print(f"     Voynich H1={v_h1:.3f}, H2={v_h2:.3f}, mean_len={v_mean_len:.2f}")
    print(f"     Encoded Latin H1={latin_stats['h1']:.3f}, H2={latin_stats['h2']:.3f}, "
          f"mean_len={latin_stats['mean_length']:.2f}")

    # ── 8. Save ──
    elapsed = time.time() - t0

    result = TachygraphicEncoderResult(
        n_syllable_mappings=table.n_syllable_mappings,
        n_triples=table.n_triples,
        n_modifier_chars=len(modifier_chars),
        n_sign_families=n_families,
        syllable_to_triple=table.syllable_to_triple,
        sample_plaintext=sample_text[:500],
        sample_syllabified=syl_latin[:20],
        sample_encoded_tokens=encoded_latin[:50],
        sample_n_tokens=len(encoded_latin),
        sample_mean_token_length=latin_stats['mean_length'],
        round_trip_n_words=rt_n,
        round_trip_recovered=n_recovered,
        round_trip_loss=round(rt_loss, 4),
        encoded_h1=round(latin_stats['h1'], 4),
        encoded_h2=round(latin_stats['h2'], 4),
        encoded_mean_token_length=round(latin_stats['mean_length'], 3),
        encoded_n_unique_chars=latin_stats['n_unique_chars'],
        latin_sample_stats={k: round(v, 4) if isinstance(v, float) else v
                           for k, v in latin_stats.items()},
        italian_sample_stats={k: round(v, 4) if isinstance(v, float) else v
                             for k, v in italian_stats.items()},
        runtime_seconds=round(elapsed, 1),
    )

    out_path = os.path.join(rd, 'tachygraphic_encoder.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)
    print(f"\n  → {out_path} ({elapsed:.1f}s)")
