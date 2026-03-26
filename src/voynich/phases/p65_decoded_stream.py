"""
Phase 65, Step 1: Build Decoded Character Streams
==================================================
Concatenate CVC-decoded tokens into continuous character streams
for word boundary discovery.  Also build Latin calibration streams
with known word boundaries.

Dependency chain:
    results/combined_refine.json   (Phase 15)
    corrected_coda (Phase 60)
    data/reference/latin/          (Latin reference corpus)
    results/cvc_recipes.json       (Phase 59)
        -> results/p65_decoded_stream.json
"""

import json
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import build_eva_to_triple_lookup, load_corpus
from voynich.core.reference import load_reference_corpus
from voynich.core.stats import syllabify_latin
from voynich.phases.corrected_coda import (
    build_coda_table_v2,
    decode_corpus_cvc_v2,
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
    if isinstance(obj, set):
        return sorted(obj)
    if isinstance(obj, (bool, int, float, str, type(None))):
        return obj
    return str(obj)


def _safe_load(path: str) -> Dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _save_json(rd: str, filename: str, data: Any) -> str:
    path = os.path.join(rd, filename)
    with open(path, 'w') as f:
        json.dump(_convert(data), f, indent=2)
    return path


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class StreamInfo:
    """A single character stream with metadata."""
    key: str = ""
    text: str = ""
    n_chars: int = 0
    n_tokens: int = 0
    token_boundaries: List[int] = field(default_factory=list)


@dataclass
class LatinCalibStream:
    """Latin calibration stream with known word boundaries."""
    name: str = ""
    text: str = ""
    n_chars: int = 0
    n_words: int = 0
    word_boundaries: List[int] = field(default_factory=list)
    mean_word_length: float = 0.0


@dataclass
class DecodedStreamResult:
    phase: str = "65"
    step: str = "65.1"
    experiment: str = "decoded_stream"
    # Full corpus stream
    full_stream_chars: int = 0
    full_stream_tokens: int = 0
    # Section-level streams
    n_sections: int = 0
    section_keys: List[str] = field(default_factory=list)
    section_chars: Dict[str, int] = field(default_factory=dict)
    # Page-level streams
    n_pages: int = 0
    mean_chars_per_page: float = 0.0
    # Recipe streams
    n_recipes: int = 0
    mean_recipe_chars: float = 0.0
    # Latin calibration
    latin_stream_chars: int = 0
    latin_n_words: int = 0
    latin_mean_word_length: float = 0.0
    # Char distribution
    voynich_alphabet: List[str] = field(default_factory=list)
    latin_alphabet: List[str] = field(default_factory=list)
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Stream building
# ---------------------------------------------------------------------------

def _build_stream_from_tokens(
    decoded_tokens: List[str],
) -> Tuple[str, List[int]]:
    """Concatenate decoded tokens into a char stream.

    Returns (stream_text, token_boundaries) where token_boundaries[i]
    is the char offset where token i+1 begins.
    """
    chars: List[str] = []
    boundaries: List[int] = []
    for decoded in decoded_tokens:
        if not decoded or decoded == '?':
            continue
        chars.extend(decoded.lower())
        boundaries.append(len(chars))
    return ''.join(chars), boundaries


def build_voynich_streams(
    corpus, assignment, eva_to_triple, coda_table,
) -> Tuple[Dict[str, StreamInfo], StreamInfo]:
    """Build all Voynich character streams.

    Returns (streams_dict, full_stream).
    streams_dict keys: 'section/<name>', 'page/<folio>', 'lang_a', 'hand_4'
    """
    streams: Dict[str, StreamInfo] = {}

    # Collect tokens per page with metadata
    page_data: List[Tuple[str, str, str, int, List[str]]] = []  # folio, section, language, hand, tokens
    for folio, page in corpus.pages.items():
        tokens = page.all_tokens
        if not tokens:
            continue
        decoded = decode_corpus_cvc_v2(tokens, assignment, eva_to_triple, coda_table)
        page_data.append((folio, page.section, page.language or '', page.hand or 0, decoded))

    # Full stream
    all_decoded: List[str] = []
    for _, _, _, _, decoded in page_data:
        all_decoded.extend(decoded)
    full_text, full_boundaries = _build_stream_from_tokens(all_decoded)
    full_stream = StreamInfo(
        key='full', text=full_text, n_chars=len(full_text),
        n_tokens=len(all_decoded), token_boundaries=full_boundaries,
    )

    # Page streams
    for folio, section, language, hand, decoded in page_data:
        text, boundaries = _build_stream_from_tokens(decoded)
        if text:
            streams[f'page/{folio}'] = StreamInfo(
                key=f'page/{folio}', text=text, n_chars=len(text),
                n_tokens=len(decoded), token_boundaries=boundaries,
            )

    # Section streams
    section_tokens: Dict[str, List[str]] = {}
    for _, section, _, _, decoded in page_data:
        section_tokens.setdefault(section, []).extend(decoded)
    for section, tokens in section_tokens.items():
        text, boundaries = _build_stream_from_tokens(tokens)
        if text:
            streams[f'section/{section}'] = StreamInfo(
                key=f'section/{section}', text=text, n_chars=len(text),
                n_tokens=len(tokens), token_boundaries=boundaries,
            )

    # Language A stream
    lang_a_tokens: List[str] = []
    for _, _, language, _, decoded in page_data:
        if language == 'A':
            lang_a_tokens.extend(decoded)
    if lang_a_tokens:
        text, boundaries = _build_stream_from_tokens(lang_a_tokens)
        streams['lang_a'] = StreamInfo(
            key='lang_a', text=text, n_chars=len(text),
            n_tokens=len(lang_a_tokens), token_boundaries=boundaries,
        )

    # Hand 4 stream
    hand4_tokens: List[str] = []
    for _, _, _, hand, decoded in page_data:
        if hand == 4:
            hand4_tokens.extend(decoded)
    if hand4_tokens:
        text, boundaries = _build_stream_from_tokens(hand4_tokens)
        streams['hand_4'] = StreamInfo(
            key='hand_4', text=text, n_chars=len(text),
            n_tokens=len(hand4_tokens), token_boundaries=boundaries,
        )

    return streams, full_stream


def build_recipe_streams(
    cvc_recipes_path: str,
) -> List[StreamInfo]:
    """Build one character stream per recipe from cvc_recipes.json."""
    data = _safe_load(cvc_recipes_path)
    if not data:
        return []

    recipes_raw = data.get('top_recipes', [])
    if not recipes_raw:
        # Try the full list
        recipes_raw = data.get('recipes', [])

    recipe_streams: List[StreamInfo] = []
    for i, recipe in enumerate(recipes_raw):
        tokens = recipe.get('tokens', [])
        if not tokens:
            continue
        text, boundaries = _build_stream_from_tokens(tokens)
        if text:
            recipe_streams.append(StreamInfo(
                key=f'recipe/{i}',
                text=text,
                n_chars=len(text),
                n_tokens=len(tokens),
                token_boundaries=boundaries,
            ))
    return recipe_streams


def build_latin_calibration_streams() -> List[LatinCalibStream]:
    """Build Latin character streams with known word boundaries.

    Mimics CVC decode output: syllabify each word, concatenate syllables
    into a character stream, record where word boundaries fall.
    """
    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    latin_words = [w.lower() for w in ref_corpus.get_combined_tokens('latin')
                   if len(w) >= 2 and w.isalpha()]

    # Build full Latin calibration stream
    chars: List[str] = []
    word_boundaries: List[int] = []

    for word in latin_words:
        syls = syllabify_latin(word)
        if not syls:
            continue
        for syl in syls:
            chars.extend(syl.lower())
        word_boundaries.append(len(chars))

    text = ''.join(chars)
    word_lengths = []
    prev = 0
    for b in word_boundaries:
        word_lengths.append(b - prev)
        prev = b

    import numpy as np
    mean_wl = float(np.mean(word_lengths)) if word_lengths else 0.0

    return [LatinCalibStream(
        name='latin_combined',
        text=text,
        n_chars=len(text),
        n_words=len(word_boundaries),
        word_boundaries=word_boundaries,
        mean_word_length=round(mean_wl, 2),
    )]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_build_stream():
    """Phase 65.1: Build decoded character streams."""
    t0 = time.time()
    rd = str(_results_dir())
    print("=" * 70)
    print("Phase 65, Step 1: Build Decoded Character Streams")
    print("=" * 70)

    # Load decode infrastructure
    eva_to_triple = build_eva_to_triple_lookup()
    refine = _safe_load(os.path.join(rd, 'combined_refine.json'))
    assignment = refine.get('best_assignment', {})
    coda_table = build_coda_table_v2()
    corpus = load_corpus(verbose=False)
    print(f"  Corpus: {len(corpus.pages)} pages")
    print(f"  Assignment triples: {len(assignment)}")

    # Build Voynich streams
    print("\n  Building Voynich streams...")
    streams, full_stream = build_voynich_streams(
        corpus, assignment, eva_to_triple, coda_table)
    print(f"  Full stream: {full_stream.n_chars} chars from {full_stream.n_tokens} tokens")

    section_streams = {k: v for k, v in streams.items() if k.startswith('section/')}
    page_streams = {k: v for k, v in streams.items() if k.startswith('page/')}
    print(f"  Sections: {len(section_streams)}")
    for k, v in sorted(section_streams.items()):
        print(f"    {k}: {v.n_chars} chars, {v.n_tokens} tokens")
    print(f"  Pages: {len(page_streams)}")

    import numpy as np
    page_chars = [v.n_chars for v in page_streams.values()]
    mean_page = float(np.mean(page_chars)) if page_chars else 0.0

    # Build recipe streams
    print("\n  Building recipe streams...")
    recipe_path = os.path.join(rd, 'cvc_recipes.json')
    recipe_streams = build_recipe_streams(recipe_path)
    recipe_chars = [r.n_chars for r in recipe_streams]
    mean_recipe = float(np.mean(recipe_chars)) if recipe_chars else 0.0
    print(f"  Recipes: {len(recipe_streams)}, mean {mean_recipe:.1f} chars")

    # Build Latin calibration streams
    print("\n  Building Latin calibration streams...")
    latin_streams = build_latin_calibration_streams()
    for ls in latin_streams:
        print(f"  {ls.name}: {ls.n_chars} chars, {ls.n_words} words, "
              f"mean word length {ls.mean_word_length}")

    # Character distributions
    voynich_chars = sorted(set(full_stream.text))
    latin_chars = sorted(set(latin_streams[0].text)) if latin_streams else []
    print(f"\n  Voynich alphabet: {''.join(voynich_chars)} ({len(voynich_chars)} chars)")
    print(f"  Latin alphabet:   {''.join(latin_chars)} ({len(latin_chars)} chars)")

    # Save everything
    # For JSON, we store stream texts and boundaries but NOT the full
    # position maps (too large). Methods will reload from this file.
    save_data = {
        'full_stream': asdict(full_stream),
        'section_streams': {k: asdict(v) for k, v in section_streams.items()},
        'page_streams': {k: {'n_chars': v.n_chars, 'n_tokens': v.n_tokens}
                         for k, v in page_streams.items()},
        'recipe_streams': [asdict(r) for r in recipe_streams],
        'latin_streams': [asdict(ls) for ls in latin_streams],
    }

    # Also save the section stream texts separately for methods to load
    # (they're the primary input for Harris/Bayesian/LM)
    save_data['section_stream_texts'] = {
        k: v.text for k, v in section_streams.items()
    }
    save_data['section_stream_boundaries'] = {
        k: v.token_boundaries for k, v in section_streams.items()
    }

    result = DecodedStreamResult(
        full_stream_chars=full_stream.n_chars,
        full_stream_tokens=full_stream.n_tokens,
        n_sections=len(section_streams),
        section_keys=sorted(section_streams.keys()),
        section_chars={k: v.n_chars for k, v in section_streams.items()},
        n_pages=len(page_streams),
        mean_chars_per_page=round(mean_page, 1),
        n_recipes=len(recipe_streams),
        mean_recipe_chars=round(mean_recipe, 1),
        latin_stream_chars=latin_streams[0].n_chars if latin_streams else 0,
        latin_n_words=latin_streams[0].n_words if latin_streams else 0,
        latin_mean_word_length=latin_streams[0].mean_word_length if latin_streams else 0.0,
        voynich_alphabet=voynich_chars,
        latin_alphabet=latin_chars,
        runtime_seconds=round(time.time() - t0, 2),
    )

    save_data['result'] = asdict(result)
    _save_json(rd, 'p65_decoded_stream.json', save_data)
    print(f"\n  Saved to results/p65_decoded_stream.json")
    print(f"  Runtime: {result.runtime_seconds:.1f}s")
    return result
