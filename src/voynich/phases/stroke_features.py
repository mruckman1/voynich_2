"""
Phase 14.2 – Stroke Feature Decomposition
==========================================
Enumerates the unique (first_stroke, last_stroke, glyph_class) triples from
EVA_VISUAL_COMPONENTS.  For each triple, records which EVA glyphs share it
and its total corpus frequency.  Builds articulatory hypotheses mapping each
stroke type to candidate phonemes.

The key output is the list of ~23 attested triples — these become the Phase 14
CSP variables, replacing the 14 grid cells from Phase 11.

Dependency chain:
    cv_labels.json  (for frequency data)
    cell_analysis.json  (for cluster context)
        → stroke_features.json (this step)
"""

import json
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    load_corpus,
    tokenize_eva_chars,
)
from voynich.core.reference import (
    EVA_VISUAL_COMPONENTS,
    PHONEME_PLACE_MAP,
    PHONEME_NUCLEUS_MAP,
    build_cv_syllable_table,
    build_triple_phoneme_hypotheses,
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
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class AttestedTriple:
    """One unique (first_stroke, last_stroke, glyph_class) triple."""
    triple_key: str             # "first_stroke,last_stroke,glyph_class"
    first_stroke: str
    last_stroke: str
    glyph_class: str
    eva_glyphs: List[str]       # EVA characters with this exact triple
    corpus_freq: int            # total frequency in Language A
    is_singleton: bool          # only one glyph -> unique phoneme slot
    onset_candidates: List[str] # from PHONEME_PLACE_MAP[first_stroke]
    nucleus_candidates: List[str]  # from PHONEME_NUCLEUS_MAP[last_stroke]
    hypothesis_syllables: List[str]  # cross-product filtered to Latin


@dataclass
class StrokeTypeEntry:
    """Summary of one stroke type and its phoneme candidates."""
    stroke_type: str
    role: str                   # 'onset', 'nucleus', or 'modifier'
    n_glyphs: int               # how many EVA glyphs have this stroke type
    candidate_phonemes: List[str]
    rationale: str


@dataclass
class StrokeFeaturesResult:
    """Full stroke decomposition analysis."""
    first_stroke_types: List[str]   # unique types found in EVA_VISUAL_COMPONENTS
    last_stroke_types: List[str]
    glyph_class_types: List[str]
    n_attested_triples: int
    n_singleton_triples: int        # triples with exactly one glyph
    n_collision_triples: int        # triples with multiple glyphs
    attested_triples: List[Dict]
    onset_entries: List[Dict]       # StrokeTypeEntry for each first_stroke type
    nucleus_entries: List[Dict]     # StrokeTypeEntry for each last_stroke type
    search_space_variables: int     # = n_attested_triples
    search_space_candidates: int    # average domain size
    search_space_estimate: str      # descriptive note
    recommended_model: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_stroke_features() -> None:
    """Step 14.2: enumerate stroke triples and build phoneme hypotheses."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE 14.2: Stroke Feature Decomposition")
    print("=" * 70)

    rd = _results_dir()

    # Load Language A corpus frequencies per glyph
    corpus = load_corpus(verbose=False)
    tokens = corpus.get_tokens(language='A', paragraph_only=True)

    glyph_freq: Counter = Counter()
    for token in tokens:
        for ch in tokenize_eva_chars(token):
            glyph_freq[ch] += 1

    print(f"\n  Total Language A tokens: {len(tokens)}")
    print(f"  Unique glyphs with corpus frequency: {len(glyph_freq)}")

    # Build the list of attested triples
    triple_to_glyphs: Dict[str, List[str]] = {}
    triple_to_components: Dict[str, Tuple[str, str, str]] = {}

    for glyph, comp in EVA_VISUAL_COMPONENTS.items():
        fs = comp['first_stroke']
        ls = comp['last_stroke']
        gc = comp['glyph_class']
        triple_key = f"{fs},{ls},{gc}"
        if triple_key not in triple_to_glyphs:
            triple_to_glyphs[triple_key] = []
            triple_to_components[triple_key] = (fs, ls, gc)
        triple_to_glyphs[triple_key].append(glyph)

    # Build hypothesis syllables for Latin
    latin_inventory = build_cv_syllable_table('latin')
    hypothesis_map = build_triple_phoneme_hypotheses('latin', latin_inventory)

    attested: List[AttestedTriple] = []
    for triple_key in sorted(triple_to_glyphs.keys()):
        glyphs = triple_to_glyphs[triple_key]
        fs, ls, gc = triple_to_components[triple_key]
        corpus_freq = sum(glyph_freq.get(g, 0) for g in glyphs)

        triple = AttestedTriple(
            triple_key=triple_key,
            first_stroke=fs,
            last_stroke=ls,
            glyph_class=gc,
            eva_glyphs=glyphs,
            corpus_freq=corpus_freq,
            is_singleton=(len(glyphs) == 1),
            onset_candidates=PHONEME_PLACE_MAP.get(fs, []),
            nucleus_candidates=PHONEME_NUCLEUS_MAP.get(ls, []),
            hypothesis_syllables=hypothesis_map.get(triple_key, []),
        )
        attested.append(triple)

    # Sort by corpus frequency descending
    attested.sort(key=lambda t: t.corpus_freq, reverse=True)

    n_singleton = sum(1 for t in attested if t.is_singleton)
    n_collision = len(attested) - n_singleton

    # Enumerate stroke type summaries
    first_types = sorted(set(comp['first_stroke'] for comp in EVA_VISUAL_COMPONENTS.values()))
    last_types = sorted(set(comp['last_stroke'] for comp in EVA_VISUAL_COMPONENTS.values()))
    class_types = sorted(set(comp['glyph_class'] for comp in EVA_VISUAL_COMPONENTS.values()))

    # Count how many EVA glyphs have each first_stroke type
    first_glyph_count: Counter = Counter(
        comp['first_stroke'] for comp in EVA_VISUAL_COMPONENTS.values()
    )
    last_glyph_count: Counter = Counter(
        comp['last_stroke'] for comp in EVA_VISUAL_COMPONENTS.values()
    )

    onset_entries = [
        _convert(StrokeTypeEntry(
            stroke_type=st,
            role='onset',
            n_glyphs=first_glyph_count[st],
            candidate_phonemes=PHONEME_PLACE_MAP.get(st, []),
            rationale=(
                "tall strokes = stops (most common consonant class in Latin)" if st == 'ascender' else
                "connected strokes = labials and bilabials" if st == 'connector' else
                "crossbar = rare/fricative category" if st == 'crossbar' else
                "loop onset = liquids, sonorants, open vowels" if st == 'loop' else
                "open curve = sibilants / palatals" if st == 'open_curve' else
                "sigmoid = sibilant category" if st == 'sigmoid' else
                "vertical strokes = nasals and dentals" if st == 'vertical' else
                "unknown"
            ),
        ))
        for st in sorted(first_types)
    ]

    nucleus_entries = [
        _convert(StrokeTypeEntry(
            stroke_type=st,
            role='nucleus',
            n_glyphs=last_glyph_count[st],
            candidate_phonemes=PHONEME_NUCLEUS_MAP.get(st, []),
            rationale=(
                "tall ascender end = open/low vowels" if st == 'ascender' else
                "connector end = mid vowels" if st == 'connector' else
                "crossbar end = coronal coda / dental closure" if st == 'crossbar' else
                "descender = high back vowels" if st == 'descender' else
                "hook end = nasal coda" if st == 'hook' else
                "loop end = round / back vowels" if st == 'loop' else
                "open curve end = open vowels" if st == 'open_curve' else
                "plume end = labial coda" if st == 'plume' else
                "sigmoid end = rhotic / sibilant coda" if st == 'sigmoid' else
                "tail end = unrounded front vowels" if st == 'tail' else
                "vertical end = high front / lateral" if st == 'vertical' else
                "unknown"
            ),
        ))
        for st in sorted(last_types)
    ]

    # Search space estimate
    avg_domain = (
        sum(len(t.hypothesis_syllables) for t in attested) / len(attested)
        if attested else 0
    )
    search_space_estimate = (
        f"{len(attested)} variables × ~{avg_domain:.0f} candidates each "
        f"= ~{len(attested) * avg_domain:.0f} effective states "
        f"(vs 14 × 30 = 420 in Phase 11). "
        f"Beam search with width 80 is tractable."
    )

    # Print summary table
    print(f"\n  Attested triples: {len(attested)} ({n_singleton} singletons, {n_collision} collisions)")
    print(f"\n  {'Triple':<40} {'Glyphs':<25} {'Freq':>8}  {'Hypotheses'}")
    for t in attested:
        glyph_str = ', '.join(t.eva_glyphs[:4]) + ('...' if len(t.eva_glyphs) > 4 else '')
        hyp_str = ', '.join(t.hypothesis_syllables[:4]) + ('...' if len(t.hypothesis_syllables) > 4 else '')
        print(f"  {t.triple_key:<40} {glyph_str:<25} {t.corpus_freq:>8}  {hyp_str}")

    print(f"\n  First-stroke types ({len(first_types)}): {', '.join(first_types)}")
    print(f"  Last-stroke types  ({len(last_types)}): {', '.join(last_types)}")
    print(f"  Glyph-class types  ({len(class_types)}): {', '.join(class_types)}")
    print(f"\n  Search space: {search_space_estimate}")

    result = StrokeFeaturesResult(
        first_stroke_types=first_types,
        last_stroke_types=last_types,
        glyph_class_types=class_types,
        n_attested_triples=len(attested),
        n_singleton_triples=n_singleton,
        n_collision_triples=n_collision,
        attested_triples=[_convert(t) for t in attested],
        onset_entries=onset_entries,
        nucleus_entries=nucleus_entries,
        search_space_variables=len(attested),
        search_space_candidates=int(avg_domain),
        search_space_estimate=search_space_estimate,
        recommended_model='attested_triples',
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'stroke_features.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)
    print(f"\n  Results saved → {out_path}")
