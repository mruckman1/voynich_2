"""
Step 34.14 – Gallows 2D Spatial Geometry (Track E)
===================================================
Extracts 2D spatial relationships between gallows and bench characters.
Gallows characters (k, t, p, f) have distinctive spatial arrangements:
intersecting (straddle ligatures like cth/ckh/cph/cfh), preceding,
following, or standalone.  This step tags every gallows occurrence with
its spatial relationship and analyses section distributions and
vocabulary profiles per spatial type.

Dependency chain:
    combined_refine.json       (Phase 15 assignment)
    modifier_integrate.json    (Phase 16 modifiers)
    signal_bigrams.json        (Phase 29 token classifications)
        → gallows_geometry.json   (this step)
"""

import json
import os
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    load_corpus,
    tokenize_eva_chars,
)
from voynich.core.reference import (
    EVA_VISUAL_COMPONENTS,
    build_expanded_word_set,
    load_reference_corpus,
)
from voynich.phases.csp_solver import decode_token
from voynich.phases.null_corpus import _reconstruct_modifier_rules
from voynich.phases.signal_isolation import _decode_corpus_r3


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
# Constants
# ---------------------------------------------------------------------------

# Pure gallows characters (tall, ascender-class)
GALLOWS_CHARS = {'k', 't', 'p', 'f'}

# Gallows-bench ligatures: gallows physically straddles a bench character
GALLOWS_BENCH_LIGATURES = {'cth', 'ckh', 'cph', 'cfh'}

# All bench-class characters (from EVA_VISUAL_COMPONENTS)
BENCH_CHARS = {
    ch for ch, comp in EVA_VISUAL_COMPONENTS.items()
    if comp.get('glyph_class') == 'bench'
}

# Spatial relationship types
SPATIAL_INTERSECTING = 'INTERSECTING'   # gallows-bench ligature (cth/ckh/cph/cfh)
SPATIAL_PRECEDING = 'PRECEDING'         # gallows immediately before non-gallows
SPATIAL_FOLLOWING = 'FOLLOWING'         # gallows immediately after non-gallows
SPATIAL_STANDALONE = 'STANDALONE'       # gallows not adjacent to bench chars


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class GallowsOccurrence:
    """A single gallows occurrence within a token."""
    folio: str
    token: str
    eva_char: str
    position: int           # position within token's EVA char list
    spatial_type: str       # one of the SPATIAL_* constants
    adjacent_chars: List[str]   # neighbouring EVA chars


@dataclass
class SpatialVocabProfile:
    """Vocabulary profile for tokens containing a given spatial type."""
    spatial_type: str
    n_tokens: int
    n_dict_hits: int
    dict_hit_rate: float
    top_decoded_words: List[Dict]   # word → count, top 20
    n_signal: int
    signal_rate: float


@dataclass
class SectionSpatialDistribution:
    """Distribution of spatial types across a manuscript section."""
    section: str
    n_tokens: int
    n_gallows_tokens: int
    gallows_rate: float
    spatial_counts: Dict[str, int]
    spatial_rates: Dict[str, float]


@dataclass
class DeterminativeAnalysis:
    """Test whether gallows spatial type acts as a determinative marker."""
    spatial_type: str
    n_tokens: int
    unique_decoded_words: int
    top_word_concentration: float  # fraction of tokens decoded as top-5 words
    semantic_coherence: float      # ratio of top-domain words to total


@dataclass
class GallowsGeometryResult:
    # Overall counts
    n_tokens_total: int
    n_gallows_total: int
    n_gallows_tokens: int
    gallows_token_rate: float
    spatial_counts: Dict[str, int]
    spatial_rates: Dict[str, float]

    # Per EVA char breakdown
    per_char_spatial: Dict[str, Dict[str, int]]  # eva_char → {type: count}

    # Per-section distribution
    per_section_distribution: List[Dict]

    # Spatial vocabulary profiles
    spatial_vocab_profiles: List[Dict]

    # Determinative analysis
    determinative_analysis: List[Dict]

    # Sample occurrences
    sample_occurrences: List[Dict]

    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Gallows spatial classification
# ---------------------------------------------------------------------------

def _classify_gallows_spatial(
    token: str,
    eva_chars: List[str],
) -> List[Tuple[str, int, str, List[str]]]:
    """Classify spatial relationships for every gallows char in a token.

    Returns list of (eva_char, position, spatial_type, adjacent_chars).
    """
    results = []
    n = len(eva_chars)

    for i, ch in enumerate(eva_chars):
        # Check for gallows-bench ligature (intersecting)
        if ch in GALLOWS_BENCH_LIGATURES:
            adj = []
            if i > 0:
                adj.append(eva_chars[i - 1])
            if i < n - 1:
                adj.append(eva_chars[i + 1])
            results.append((ch, i, SPATIAL_INTERSECTING, adj))
            continue

        # Check for pure gallows character
        if ch not in GALLOWS_CHARS:
            continue

        prev_ch = eva_chars[i - 1] if i > 0 else None
        next_ch = eva_chars[i + 1] if i < n - 1 else None

        adj = []
        if prev_ch:
            adj.append(prev_ch)
        if next_ch:
            adj.append(next_ch)

        # Determine spatial relationship
        has_bench_before = prev_ch is not None and (
            prev_ch in BENCH_CHARS or prev_ch in GALLOWS_BENCH_LIGATURES
        )
        has_bench_after = next_ch is not None and (
            next_ch in BENCH_CHARS or next_ch in GALLOWS_BENCH_LIGATURES
        )
        has_nonG_before = prev_ch is not None and prev_ch not in GALLOWS_CHARS
        has_nonG_after = next_ch is not None and next_ch not in GALLOWS_CHARS

        if n == 1:
            spatial = SPATIAL_STANDALONE
        elif has_nonG_after and not has_nonG_before:
            spatial = SPATIAL_PRECEDING
        elif has_nonG_before and not has_nonG_after:
            spatial = SPATIAL_FOLLOWING
        elif has_nonG_before and has_nonG_after:
            # Gallows between other chars — treat as preceding (more common)
            spatial = SPATIAL_PRECEDING
        else:
            spatial = SPATIAL_STANDALONE

        results.append((ch, i, spatial, adj))

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_gallows_geometry() -> None:
    """Step 34.14: Gallows 2D spatial geometry extraction."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 34.14: Gallows 2D Spatial Geometry (Track E)")
    print("=" * 70)

    rd = _results_dir()
    eva_to_triple = build_eva_to_triple_lookup()

    # ── 1. Load inputs ──
    print("\n  1. Loading inputs ...")

    # Phase 15 assignment
    refine_path = os.path.join(rd, 'combined_refine.json')
    if not os.path.exists(refine_path):
        print("  [SKIP] combined_refine.json not found")
        return
    with open(refine_path) as f:
        refine_data = json.load(f)
    assignment = refine_data.get('best_assignment', {})

    # Phase 16 modifiers
    mod_path = os.path.join(rd, 'modifier_integrate.json')
    if not os.path.exists(mod_path):
        print("  [SKIP] modifier_integrate.json not found")
        return
    with open(mod_path) as f:
        mod_data = json.load(f)
    modifier_chars, modifier_rules = _reconstruct_modifier_rules(mod_data)

    # Signal classifications (optional)
    signal_classifications: List[str] = []
    sig_path = os.path.join(rd, 'signal_bigrams.json')
    if os.path.exists(sig_path):
        with open(sig_path) as f:
            sig_data = json.load(f)
        signal_classifications = sig_data.get('token_classifications', [])

    print(f"     Assignment: {len(assignment)} triples")
    print(f"     Modifiers: {len(modifier_chars)} chars")
    print(f"     Signal classifications: {len(signal_classifications)} tokens")

    # ── 2. Build reference word set ──
    print("\n  2. Building reference word set ...")
    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    base_words = set(
        w.lower() for w in ref_corpus.get_combined_tokens('latin')
        if len(w) >= 2
    )
    expanded, _ = build_expanded_word_set(base_words)
    ref_word_set = base_words | expanded
    print(f"     {len(ref_word_set)} reference words")

    # ── 3. Load and process corpus ──
    print("\n  3. Loading corpus and classifying gallows ...")
    corpus = load_corpus(verbose=False)

    all_tokens: List[str] = []
    token_folios: List[str] = []
    token_sections: List[str] = []

    for folio, page in corpus.pages.items():
        for token in page.all_tokens:
            all_tokens.append(token)
            token_folios.append(folio)
            token_sections.append(page.section)

    n_tokens = len(all_tokens)
    print(f"     {n_tokens} tokens")

    # ── 4. Decode corpus ──
    print("\n  4. Decoding corpus ...")
    decoded = _decode_corpus_r3(
        all_tokens, assignment, eva_to_triple,
        modifier_chars, modifier_rules, ref_word_set,
    )
    dict_hits = [w in ref_word_set for w in decoded]

    # ── 5. Classify gallows spatial relationships ──
    print("\n  5. Classifying gallows spatial relationships ...")

    all_occurrences: List[Dict] = []
    token_spatial_tags: List[List[str]] = []  # per-token: list of spatial types
    gallows_token_indices: List[int] = []

    spatial_counter: Counter = Counter()
    per_char_spatial: Dict[str, Counter] = defaultdict(Counter)

    for idx, token in enumerate(all_tokens):
        eva_chars = tokenize_eva_chars(token)
        classifications = _classify_gallows_spatial(token, eva_chars)

        spatial_tags = []
        for eva_char, pos, spatial_type, adj in classifications:
            spatial_counter[spatial_type] += 1
            per_char_spatial[eva_char][spatial_type] += 1
            spatial_tags.append(spatial_type)

            all_occurrences.append({
                'folio': token_folios[idx],
                'token': token,
                'eva_char': eva_char,
                'position': pos,
                'spatial_type': spatial_type,
                'adjacent_chars': adj,
                'decoded': decoded[idx],
                'dict_hit': dict_hits[idx],
            })

        token_spatial_tags.append(spatial_tags)
        if spatial_tags:
            gallows_token_indices.append(idx)

    n_gallows_total = sum(spatial_counter.values())
    n_gallows_tokens = len(gallows_token_indices)
    gallows_token_rate = n_gallows_tokens / n_tokens if n_tokens > 0 else 0.0

    spatial_rates = {
        k: round(v / n_gallows_total, 4) if n_gallows_total > 0 else 0.0
        for k, v in spatial_counter.items()
    }

    print(f"     Total gallows occurrences: {n_gallows_total}")
    print(f"     Gallows-bearing tokens: {n_gallows_tokens} "
          f"({gallows_token_rate:.1%})")
    for st in [SPATIAL_INTERSECTING, SPATIAL_PRECEDING,
               SPATIAL_FOLLOWING, SPATIAL_STANDALONE]:
        cnt = spatial_counter.get(st, 0)
        print(f"       {st:14s}: {cnt:5d} ({spatial_rates.get(st, 0):.1%})")

    # ── 6. Per-section distribution ──
    print("\n  6. Per-section gallows distribution ...")
    section_tokens: Dict[str, int] = Counter(token_sections)
    section_gallows: Dict[str, int] = Counter()
    section_spatial: Dict[str, Counter] = defaultdict(Counter)

    for idx in gallows_token_indices:
        sec = token_sections[idx]
        section_gallows[sec] += 1
        for st in token_spatial_tags[idx]:
            section_spatial[sec][st] += 1

    section_distributions: List[SectionSpatialDistribution] = []
    for sec in sorted(section_tokens.keys()):
        n_tok = section_tokens[sec]
        n_gal = section_gallows.get(sec, 0)
        sc = dict(section_spatial.get(sec, {}))
        total_spatial = sum(sc.values()) if sc else 1
        sr = {k: round(v / total_spatial, 4) for k, v in sc.items()}

        dist = SectionSpatialDistribution(
            section=sec,
            n_tokens=n_tok,
            n_gallows_tokens=n_gal,
            gallows_rate=round(n_gal / n_tok, 4) if n_tok > 0 else 0.0,
            spatial_counts=sc,
            spatial_rates=sr,
        )
        section_distributions.append(dist)
        print(f"     {sec:15s}  tokens={n_tok:5d}  "
              f"gallows={n_gal:4d} ({dist.gallows_rate:.1%})  "
              f"I={sc.get(SPATIAL_INTERSECTING, 0)} "
              f"P={sc.get(SPATIAL_PRECEDING, 0)} "
              f"F={sc.get(SPATIAL_FOLLOWING, 0)} "
              f"S={sc.get(SPATIAL_STANDALONE, 0)}")

    # ── 7. Spatial vocabulary profiles ──
    print("\n  7. Spatial vocabulary profiles ...")

    # Group tokens by dominant spatial type
    spatial_type_tokens: Dict[str, List[int]] = defaultdict(list)
    for idx in gallows_token_indices:
        tags = token_spatial_tags[idx]
        if tags:
            # Use the first (dominant) spatial type
            spatial_type_tokens[tags[0]].append(idx)

    vocab_profiles: List[SpatialVocabProfile] = []
    for st in [SPATIAL_INTERSECTING, SPATIAL_PRECEDING,
               SPATIAL_FOLLOWING, SPATIAL_STANDALONE]:
        indices = spatial_type_tokens.get(st, [])
        n_st = len(indices)
        if n_st == 0:
            vocab_profiles.append(SpatialVocabProfile(
                spatial_type=st, n_tokens=0, n_dict_hits=0,
                dict_hit_rate=0.0, top_decoded_words=[], n_signal=0,
                signal_rate=0.0,
            ))
            continue

        st_hits = sum(1 for i in indices if dict_hits[i])
        st_hit_rate = st_hits / n_st if n_st > 0 else 0.0

        word_counts: Counter = Counter()
        for i in indices:
            if dict_hits[i]:
                word_counts[decoded[i]] += 1

        top_words = [{'word': w, 'count': c}
                     for w, c in word_counts.most_common(20)]

        # Signal count if available
        n_sig = 0
        if signal_classifications:
            for i in indices:
                if i < len(signal_classifications):
                    if signal_classifications[i] == 'SIGNAL':
                        n_sig += 1
        sig_rate = n_sig / n_st if n_st > 0 else 0.0

        profile = SpatialVocabProfile(
            spatial_type=st,
            n_tokens=n_st,
            n_dict_hits=st_hits,
            dict_hit_rate=round(st_hit_rate, 4),
            top_decoded_words=top_words,
            n_signal=n_sig,
            signal_rate=round(sig_rate, 4),
        )
        vocab_profiles.append(profile)
        print(f"     {st:14s}  n={n_st:4d}  dict_hit={st_hit_rate:.1%}  "
              f"signal={sig_rate:.1%}  top: "
              + ", ".join(f"{w['word']}({w['count']})"
                          for w in top_words[:5]))

    # ── 8. Determinative analysis ──
    print("\n  8. Determinative analysis ...")

    det_analyses: List[DeterminativeAnalysis] = []
    for st in [SPATIAL_INTERSECTING, SPATIAL_PRECEDING,
               SPATIAL_FOLLOWING, SPATIAL_STANDALONE]:
        indices = spatial_type_tokens.get(st, [])
        n_st = len(indices)
        if n_st == 0:
            det_analyses.append(DeterminativeAnalysis(
                spatial_type=st, n_tokens=0, unique_decoded_words=0,
                top_word_concentration=0.0, semantic_coherence=0.0,
            ))
            continue

        word_counts: Counter = Counter()
        for i in indices:
            word_counts[decoded[i]] += 1

        unique_words = len(word_counts)
        top5_count = sum(c for _, c in word_counts.most_common(5))
        top_concentration = top5_count / n_st if n_st > 0 else 0.0

        # Semantic coherence: what fraction of dict-hit words are in the
        # top decoded domain?  Crude proxy: top-10 words / total unique
        top10_count = sum(c for _, c in word_counts.most_common(10))
        semantic_coh = top10_count / n_st if n_st > 0 else 0.0

        det = DeterminativeAnalysis(
            spatial_type=st,
            n_tokens=n_st,
            unique_decoded_words=unique_words,
            top_word_concentration=round(top_concentration, 4),
            semantic_coherence=round(semantic_coh, 4),
        )
        det_analyses.append(det)
        print(f"     {st:14s}  n={n_st:4d}  unique={unique_words:4d}  "
              f"top5_conc={top_concentration:.1%}  "
              f"sem_coh={semantic_coh:.1%}")

    # ── 9. Verdict ──
    # A spatial type is determinative-like if top5 concentration > 30%
    det_types = [d.spatial_type for d in det_analyses
                 if d.top_word_concentration > 0.30 and d.n_tokens >= 10]
    has_differential = False
    if len(vocab_profiles) >= 2:
        rates = [p.dict_hit_rate for p in vocab_profiles if p.n_tokens > 0]
        if rates:
            has_differential = max(rates) - min(rates) > 0.05

    verdict = (
        f"{n_gallows_total} gallows occurrences in {n_gallows_tokens} tokens. "
        f"Spatial types: I={spatial_counter.get(SPATIAL_INTERSECTING, 0)}, "
        f"P={spatial_counter.get(SPATIAL_PRECEDING, 0)}, "
        f"F={spatial_counter.get(SPATIAL_FOLLOWING, 0)}, "
        f"S={spatial_counter.get(SPATIAL_STANDALONE, 0)}. "
        f"{'DIFFERENTIAL vocab across types' if has_differential else 'NO differential vocab'}. "
        f"Determinative-like types: {det_types if det_types else 'none'}."
    )
    print(f"\n  VERDICT: {verdict}")

    # ── 10. Save ──
    elapsed = round(time.time() - t0, 2)

    result = GallowsGeometryResult(
        n_tokens_total=n_tokens,
        n_gallows_total=n_gallows_total,
        n_gallows_tokens=n_gallows_tokens,
        gallows_token_rate=round(gallows_token_rate, 4),
        spatial_counts=dict(spatial_counter),
        spatial_rates=spatial_rates,
        per_char_spatial={
            ch: dict(counts)
            for ch, counts in sorted(per_char_spatial.items())
        },
        per_section_distribution=[
            _convert(asdict(d)) for d in section_distributions
        ],
        spatial_vocab_profiles=[
            _convert(asdict(p)) for p in vocab_profiles
        ],
        determinative_analysis=[
            _convert(asdict(d)) for d in det_analyses
        ],
        sample_occurrences=all_occurrences[:100],
        verdict=verdict,
        runtime_seconds=elapsed,
    )

    out_path = os.path.join(rd, 'gallows_geometry.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)
    print(f"\n  Saved -> {out_path}")
    print(f"  Completed in {elapsed:.1f}s")
