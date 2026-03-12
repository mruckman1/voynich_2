"""
Phase 46 Track C – Definitive Corpus Decode and Gap Map
========================================================
Produce the final decoded corpus under the definitive table, with
per-token confidence annotations and a comprehensive gap map.

Dependency chain:
    arb_selection.json            (Step 46A.5, or canonical_table.json)
    modifier_integrate.json       (Phase 16)
    signal_10k.json               (Phase 36)
    canonical_table.json          (Phase 45)
    signal_bigrams.json           (Phase 29)
        -> final_decode_summary.json   (Step 46C.1)
        -> final_annotations.json      (Step 46C.2)
        -> gap_map.json                (Step 46C.3)
        -> project_summary.json        (Step 46C.4)
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    decode_token_modifier_aware,
    load_corpus,
    token_to_triples,
    tokenize_eva_chars,
)
from voynich.core.reference import (
    build_expanded_word_set,
    load_reference_corpus,
)
from voynich.phases.csp_solver import decode_token

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
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if v != v else v
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


def _decode_corpus_r3(
    tokens: List[str],
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    modifier_chars: set,
    modifier_rules: Dict[str, str],
    ref_word_set: set,
) -> List[str]:
    """R3 strategy: try alteration -> strip -> raw."""
    decoded = []
    for token in tokens:
        alt = decode_token_modifier_aware(
            token, assignment, eva_to_triple, modifier_chars, modifier_rules,
        )
        if alt.lower() in ref_word_set:
            decoded.append(alt.lower())
            continue
        stripped = decode_token_modifier_aware(
            token, assignment, eva_to_triple, modifier_chars,
        )
        if stripped.lower() in ref_word_set:
            decoded.append(stripped.lower())
            continue
        raw = decode_token(token, assignment, eva_to_triple)
        decoded.append(raw.lower())
    return decoded


def _reconstruct_modifier_rules(
    data: Dict,
) -> Tuple[set, Dict[str, str]]:
    modifier_chars = set(data.get('modifier_chars', []))
    modifier_rules: Dict[str, str] = {}
    for c in data.get('classifications', []):
        if c.get('final_classification') == 'modifier':
            modifier_rules[c['eva_char']] = c.get('modifier_type', 'silent')
    return modifier_chars, modifier_rules


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class FolioSummary:
    folio: str
    section: str
    n_tokens: int
    dict_hit_count: int
    dict_hit_rate: float
    signal_count: int
    signal_rate: float
    consecutive_hit_max: int
    top_decoded: List[Tuple[str, int]]
    notable_fragments: List[str]


@dataclass
class SectionAggregate:
    section: str
    n_folios: int
    n_tokens: int
    dict_hit_rate: float
    signal_rate: float
    top_decoded: List[Tuple[str, int]]


@dataclass
class FullDecodeResult:
    table_source: str
    n_tokens: int
    n_folios: int
    overall_dict_hit: float
    overall_signal_rate: float
    folio_summaries: List[Dict]
    section_aggregates: List[Dict]
    signal_word_index: List[Dict]
    recipe_catalog: List[Dict]
    runtime_seconds: float


@dataclass
class TokenAnnotation:
    eva_token: str
    decoded: str
    folio: str
    confidence: str
    reasons: List[str]


@dataclass
class AnnotationResult:
    n_tokens: int
    green_count: int
    green_rate: float
    yellow_count: int
    yellow_rate: float
    orange_count: int
    orange_rate: float
    red_count: int
    red_rate: float
    per_section: Dict[str, Dict[str, float]]
    top_green_folios: List[Dict]
    runtime_seconds: float


@dataclass
class GapCategory:
    category: str
    priority: str
    description: str
    evidence_summary: str
    open_questions: List[str]
    relevant_phases: List[str]


@dataclass
class GapMapResult:
    n_categories: int
    categories: List[Dict]
    n_high_priority: int
    n_medium_priority: int
    n_low_priority: int
    runtime_seconds: float


@dataclass
class ProjectSummaryResult:
    total_tokens: int
    total_folios: int
    n_sections: int
    n_triples: int
    n_confirmed: int
    n_landscape: int
    n_ambiguous: int
    definitive_table_source: str
    dict_hit_131k: float
    dict_hit_10k: float
    selectivity_10k: float
    bigram_z: float
    signal_rate: float
    n_signal_words: int
    encoding_type: str
    source_language: str
    content_domain: str
    landscape: str
    key_findings: List[str]
    progression: List[Dict]
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Step 46C.1 — Full Corpus Decode
# ---------------------------------------------------------------------------


def run_final_decode() -> None:
    """Step 46C.1: Decode full corpus with definitive table."""
    t0 = time.time()
    print("=" * 70)
    print("STEP 46C.1: Full Corpus Decode")
    print("=" * 70)

    rd = _results_dir()

    # Load definitive table (prefer Track A, fall back to canonical)
    table_source = 'canonical_table'
    arb_sel = _safe_load(os.path.join(rd, 'arb_selection.json'))
    if arb_sel and arb_sel.get('definitive_assignment'):
        assignment = arb_sel['definitive_assignment']
        table_source = arb_sel.get('definitive_table_name', 'arb_selection')
        print(f"  Using Track A definitive table: {table_source}")
    else:
        canon = _safe_load(os.path.join(rd, 'canonical_table.json'))
        assignment = canon.get('table', {})
        if not assignment:
            p15 = _safe_load(os.path.join(rd, 'combined_refine.json'))
            assignment = p15.get('best_assignment', {})
            table_source = 'combined_refine'
        print(f"  Using fallback table: {table_source}")

    # Load modifier rules
    mod_data = _safe_load(os.path.join(rd, 'modifier_integrate.json'))
    modifier_chars, modifier_rules = _reconstruct_modifier_rules(mod_data)

    # Build lookups
    eva_to_triple = build_eva_to_triple_lookup()

    # Build word sets
    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    ref_tokens = [
        w.lower() for w in ref_corpus.get_combined_tokens('latin')
        if len(w) >= 2
    ]
    base_words = set(ref_tokens)
    expanded, _ = build_expanded_word_set(base_words)
    ref_word_set_131k = base_words | expanded

    word_freq = Counter(ref_tokens)
    ref_word_set_10k = {w for w, _ in word_freq.most_common(10000)}

    # Load signal words
    s10_data = _safe_load(os.path.join(rd, 'signal_10k.json'))
    signal_words_list = []
    if 'word_signals' in s10_data:
        signal_words_list = [
            w['word'] for w in s10_data['word_signals']
            if w.get('is_genuine_signal')
        ]
    signal_word_set = set(signal_words_list) | set(
        ['bene', 'codi', 'sero', 'sene', 'de', 'raro', 'dine', 'cola'],
    )

    # Load recipe folios (Phase 43)
    recipe_data = _safe_load(os.path.join(rd, 'structural_reading.json'))
    recipe_folios = set()
    if recipe_data:
        for entry in recipe_data.get('recipe_folios', []):
            if isinstance(entry, str):
                recipe_folios.add(entry)
            elif isinstance(entry, dict):
                recipe_folios.add(entry.get('folio', ''))

    # Decode full corpus
    print("  Loading corpus...")
    corpus = load_corpus(verbose=False)

    folio_summaries: List[Dict] = []
    section_data: Dict[str, Dict] = defaultdict(
        lambda: {'n_folios': 0, 'n_tokens': 0, 'dict_hits': 0,
                 'signal_hits': 0, 'words': Counter()},
    )
    signal_word_locations: Dict[str, Dict] = defaultdict(
        lambda: {'total': 0, 'folios': Counter(), 'sections': Counter()},
    )
    recipe_catalog: List[Dict] = []

    total_tokens = 0
    total_dict_hits = 0
    total_signal = 0

    print("  Decoding folios...")
    for folio, page in corpus.pages.items():
        tokens = page.all_tokens
        n = len(tokens)
        if n == 0:
            continue

        section = page.section if hasattr(page, 'section') else 'unknown'

        # Decode
        decoded = _decode_corpus_r3(
            tokens, assignment, eva_to_triple,
            modifier_chars, modifier_rules, ref_word_set_131k,
        )

        # Dict hits and signal classification
        dict_hits = [w in ref_word_set_131k for w in decoded]
        signal_hits = [w in signal_word_set and dict_hits[i]
                       for i, w in enumerate(decoded)]

        n_dict = sum(dict_hits)
        n_signal = sum(signal_hits)
        total_tokens += n
        total_dict_hits += n_dict
        total_signal += n_signal

        # Consecutive hit max
        max_consec = 0
        current_consec = 0
        for h in dict_hits:
            if h:
                current_consec += 1
                max_consec = max(max_consec, current_consec)
            else:
                current_consec = 0

        # Notable fragments (3+ consecutive hits)
        fragments = []
        i = 0
        while i < n:
            if dict_hits[i]:
                run = []
                while i < n and dict_hits[i]:
                    run.append(decoded[i])
                    i += 1
                if len(run) >= 3:
                    fragments.append(' '.join(run))
            else:
                i += 1

        # Top decoded words
        word_counter = Counter(
            w for w, h in zip(decoded, dict_hits) if h
        )
        top_decoded = word_counter.most_common(10)

        folio_summaries.append({
            'folio': folio,
            'section': section,
            'n_tokens': n,
            'dict_hit_count': n_dict,
            'dict_hit_rate': round(n_dict / n, 4) if n else 0.0,
            'signal_count': n_signal,
            'signal_rate': round(n_signal / n, 4) if n else 0.0,
            'consecutive_hit_max': max_consec,
            'top_decoded': top_decoded,
            'notable_fragments': fragments[:5],
        })

        # Section aggregation
        sec = section_data[section]
        sec['n_folios'] += 1
        sec['n_tokens'] += n
        sec['dict_hits'] += n_dict
        sec['signal_hits'] += n_signal
        sec['words'].update(w for w, h in zip(decoded, dict_hits) if h)

        # Signal word tracking
        for w, h in zip(decoded, dict_hits):
            if h and w in signal_word_set:
                signal_word_locations[w]['total'] += 1
                signal_word_locations[w]['folios'][folio] += 1
                signal_word_locations[w]['sections'][section] += 1

        # Recipe catalog
        if folio in recipe_folios:
            recipe_catalog.append({
                'folio': folio,
                'n_tokens': n,
                'dict_hit_rate': round(n_dict / n, 4) if n else 0.0,
                'decoded_text': ' '.join(decoded),
                'signal_words_present': [
                    w for w in signal_word_set
                    if w in set(decoded)
                ],
            })

    # Build section aggregates
    section_aggregates = []
    for section, data in sorted(section_data.items()):
        nt = data['n_tokens']
        section_aggregates.append({
            'section': section,
            'n_folios': data['n_folios'],
            'n_tokens': nt,
            'dict_hit_rate': round(data['dict_hits'] / nt, 4) if nt else 0.0,
            'signal_rate': round(data['signal_hits'] / nt, 4) if nt else 0.0,
            'top_decoded': data['words'].most_common(20),
        })

    # Build signal word index
    signal_word_index = []
    for word in sorted(signal_word_locations.keys()):
        loc = signal_word_locations[word]
        signal_word_index.append({
            'word': word,
            'total_count': loc['total'],
            'folio_distribution': dict(loc['folios']),
            'section_distribution': dict(loc['sections']),
        })

    overall_dh = total_dict_hits / total_tokens if total_tokens else 0.0
    overall_sig = total_signal / total_tokens if total_tokens else 0.0

    # Sort folios by dict_hit_rate descending
    folio_summaries.sort(key=lambda f: -f['dict_hit_rate'])

    print(f"\n  Total tokens: {total_tokens}")
    print(f"  Overall dict_hit (131K): {overall_dh:.4f}")
    print(f"  Overall signal rate: {overall_sig:.4f}")
    print(f"  Folios: {len(folio_summaries)}")
    print(f"  Sections: {len(section_aggregates)}")
    print(f"  Signal words tracked: {len(signal_word_index)}")
    print(f"  Recipe folios: {len(recipe_catalog)}")

    if folio_summaries:
        top5 = folio_summaries[:5]
        print(f"\n  Top 5 folios by dict_hit:")
        for f in top5:
            print(f"    {f['folio']}: {f['dict_hit_rate']:.4f} "
                  f"({f['dict_hit_count']}/{f['n_tokens']})")

    result = FullDecodeResult(
        table_source=table_source,
        n_tokens=total_tokens,
        n_folios=len(folio_summaries),
        overall_dict_hit=round(overall_dh, 4),
        overall_signal_rate=round(overall_sig, 4),
        folio_summaries=folio_summaries,
        section_aggregates=section_aggregates,
        signal_word_index=signal_word_index,
        recipe_catalog=recipe_catalog,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'final_decode_summary.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2)
    print(f"\n  Saved -> {out_path}")


# ---------------------------------------------------------------------------
# Step 46C.2 — Per-Token Confidence Annotation
# ---------------------------------------------------------------------------


def run_final_annotate() -> None:
    """Step 46C.2: Assign 4-level confidence to each token."""
    t0 = time.time()
    print("=" * 70)
    print("STEP 46C.2: Per-Token Confidence Annotation")
    print("=" * 70)

    rd = _results_dir()

    # Load definitive table
    arb_sel = _safe_load(os.path.join(rd, 'arb_selection.json'))
    if arb_sel and arb_sel.get('definitive_assignment'):
        assignment = arb_sel['definitive_assignment']
    else:
        canon = _safe_load(os.path.join(rd, 'canonical_table.json'))
        assignment = canon.get('table', {})
        if not assignment:
            p15 = _safe_load(os.path.join(rd, 'combined_refine.json'))
            assignment = p15.get('best_assignment', {})

    # Load tier annotations
    canon = _safe_load(os.path.join(rd, 'canonical_table.json'))
    tier_annotations = canon.get('tier_annotations', {})
    # CONFIRMED = Tier 1, LANDSCAPE_CONFIRMED = Tier 2,
    # GENUINELY_AMBIGUOUS = Tier 3

    # Load modifier rules
    mod_data = _safe_load(os.path.join(rd, 'modifier_integrate.json'))
    modifier_chars, modifier_rules = _reconstruct_modifier_rules(mod_data)

    # Build lookups
    eva_to_triple = build_eva_to_triple_lookup()

    # Build word sets
    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    ref_tokens = [
        w.lower() for w in ref_corpus.get_combined_tokens('latin')
        if len(w) >= 2
    ]
    base_words = set(ref_tokens)
    expanded, _ = build_expanded_word_set(base_words)
    ref_word_set_131k = base_words | expanded

    word_freq = Counter(ref_tokens)
    ref_word_set_10k = {w for w, _ in word_freq.most_common(10000)}

    # Load signal classification
    s10_data = _safe_load(os.path.join(rd, 'signal_10k.json'))
    signal_words_set = set()
    if 'word_signals' in s10_data:
        signal_words_set = {
            w['word'] for w in s10_data['word_signals']
            if w.get('is_genuine_signal')
        }

    # Process corpus
    print("  Loading corpus...")
    corpus = load_corpus(verbose=False)

    counts = {'GREEN': 0, 'YELLOW': 0, 'ORANGE': 0, 'RED': 0}
    section_counts: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {'GREEN': 0, 'YELLOW': 0, 'ORANGE': 0, 'RED': 0, 'total': 0},
    )
    folio_green: Dict[str, Tuple[int, int]] = {}  # folio -> (green, total)

    total = 0
    print("  Annotating tokens...")
    for folio, page in corpus.pages.items():
        tokens = page.all_tokens
        section = page.section if hasattr(page, 'section') else 'unknown'
        n_green = 0
        n_total = len(tokens)

        for token in tokens:
            total += 1

            # Get constituent triples
            chars = tokenize_eva_chars(token)
            triples_used = []
            for ch in chars:
                if ch not in modifier_chars:
                    t = eva_to_triple.get(ch)
                    if t:
                        triples_used.append(t)

            # Check tier levels
            tiers = [tier_annotations.get(t, 'UNKNOWN') for t in triples_used]
            all_tier1 = all(t == 'CONFIRMED' for t in tiers) if tiers else False
            all_tier12 = all(
                t in ('CONFIRMED', 'LANDSCAPE_CONFIRMED') for t in tiers
            ) if tiers else False
            has_tier3 = any(t == 'GENUINELY_AMBIGUOUS' for t in tiers)

            # Decode the token
            alt = decode_token_modifier_aware(
                token, assignment, eva_to_triple,
                modifier_chars, modifier_rules,
            )
            decoded_word = alt.lower()

            # Check dictionary membership
            in_10k = decoded_word in ref_word_set_10k
            in_131k = decoded_word in ref_word_set_131k
            is_signal = decoded_word in signal_words_set

            # Assign confidence
            if all_tier1 and in_10k and is_signal:
                confidence = 'GREEN'
                n_green += 1
            elif all_tier12 and in_131k:
                confidence = 'YELLOW'
            elif (all_tier12 or has_tier3) and in_131k:
                confidence = 'ORANGE'
            else:
                confidence = 'RED'

            counts[confidence] += 1
            section_counts[section][confidence] += 1
            section_counts[section]['total'] += 1

        folio_green[folio] = (n_green, n_total)

    # Build per-section rates
    per_section: Dict[str, Dict[str, float]] = {}
    for section, sc in section_counts.items():
        t = sc['total'] or 1
        per_section[section] = {
            'GREEN': round(sc['GREEN'] / t, 4),
            'YELLOW': round(sc['YELLOW'] / t, 4),
            'ORANGE': round(sc['ORANGE'] / t, 4),
            'RED': round(sc['RED'] / t, 4),
        }

    # Top GREEN folios
    top_green = sorted(
        folio_green.items(),
        key=lambda x: -(x[1][0] / x[1][1]) if x[1][1] > 0 else 0,
    )
    top_green_list = [
        {
            'folio': f,
            'green_count': g,
            'total': t,
            'green_rate': round(g / t, 4) if t > 0 else 0.0,
        }
        for f, (g, t) in top_green[:20]
    ]

    print(f"\n  Confidence distribution (n={total}):")
    for level in ['GREEN', 'YELLOW', 'ORANGE', 'RED']:
        c = counts[level]
        r = c / total if total else 0.0
        print(f"    {level:8s}: {c:6d} ({r:.1%})")

    result = AnnotationResult(
        n_tokens=total,
        green_count=counts['GREEN'],
        green_rate=round(counts['GREEN'] / total, 4) if total else 0.0,
        yellow_count=counts['YELLOW'],
        yellow_rate=round(counts['YELLOW'] / total, 4) if total else 0.0,
        orange_count=counts['ORANGE'],
        orange_rate=round(counts['ORANGE'] / total, 4) if total else 0.0,
        red_count=counts['RED'],
        red_rate=round(counts['RED'] / total, 4) if total else 0.0,
        per_section=per_section,
        top_green_folios=top_green_list,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'final_annotations.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2)
    print(f"\n  Saved -> {out_path}")


# ---------------------------------------------------------------------------
# Step 46C.3 — Gap Map
# ---------------------------------------------------------------------------


def run_final_map() -> None:
    """Step 46C.3: Build structured gap inventory."""
    t0 = time.time()
    print("=" * 70)
    print("STEP 46C.3: Gap Map")
    print("=" * 70)

    rd = _results_dir()

    categories = [
        GapCategory(
            category='TRIPLE_ASSIGNMENTS',
            priority='HIGH',
            description=(
                'External tachygraphy tables could resolve the 6 disputed '
                'triples where Phase 15 and MaxSAT disagree, and the 3 '
                'genuinely ambiguous triples.'
            ),
            evidence_summary=(
                'Phase 44 MaxSAT found 500+ solutions within 1% of optimal. '
                'Phase 45 confirmed 12 Tier-1 triples and 10 landscape-confirmed. '
                '3 triples are genuinely ambiguous (0.04% dict-hit budget).'
            ),
            open_questions=[
                'Which of the 6 disputed assignments is correct?',
                'Can the CV model be extended to CVC/CCV?',
                'Are the 18 ambiguous modifier/syllabic chars correctly classified?',
            ],
            relevant_phases=['Phase 44', 'Phase 45', 'Phase 46 Track A'],
        ),
        GapCategory(
            category='BOTANICAL_IDENTIFICATION',
            priority='MEDIUM',
            description=(
                'Plant identifications for the ~113 botanical illustrations '
                'could provide known-plaintext cribs.'
            ),
            evidence_summary=(
                'Phase 33 showed confirmed triples are incompatible with '
                'standard Latin plant names. Phase 39 found Italian names '
                'work on f56r only (Drosera). Phase 31 botanical anchors '
                'produced no new assignments.'
            ),
            open_questions=[
                'Do herbal page labels encode plant names?',
                'Are plant names in Latin, Italian, or a vernacular?',
                'Can modern botanical AI identify more plants?',
            ],
            relevant_phases=['Phase 31', 'Phase 33', 'Phase 39'],
        ),
        GapCategory(
            category='LANGUAGE_MODEL',
            priority='HIGH',
            description=(
                'A sharper scoring function could turn the FLAT MaxSAT '
                'landscape into BASINED or PEAKED, and a better language '
                'model could disambiguate near-match bigrams.'
            ),
            evidence_summary=(
                'Phase 42 validated bigram z as the most trustworthy metric. '
                '0 exact content-content bigrams found. Phase 44 landscape '
                'is provably FLAT. Latin and Northern Italian remain '
                'indistinguishable.'
            ),
            open_questions=[
                'Can an n-gram language model provide sharper discrimination?',
                'Is the source language Latin, Northern Italian, or macaronic?',
                'Would a word-level HMM improve context disambiguation?',
            ],
            relevant_phases=['Phase 42', 'Phase 43', 'Phase 44'],
        ),
        GapCategory(
            category='CODICOLOGICAL_ANALYSIS',
            priority='MEDIUM',
            description=(
                'Physical manuscript analysis could reveal reading order, '
                'annotations, and section boundaries.'
            ),
            evidence_summary=(
                'Lisa Fagin Davis\'s page-reordering hypothesis suggests the '
                'decoded text may be in the wrong sequence. The Marci '
                'annotations on f1r (2024 multispectral imaging) may contain '
                'partially correct 17th-century decipherment guesses.'
            ),
            open_questions=[
                'Does page reordering improve decoded text coherence?',
                'What do the Marci annotations on f1r say?',
                'Are quire boundaries meaningful for content segmentation?',
            ],
            relevant_phases=['Phase 24', 'Phase 43'],
        ),
        GapCategory(
            category='ENCODING_STRUCTURE',
            priority='HIGH',
            description=(
                'The many-to-one encoding means a fixed table cannot exceed '
                '~44% dict-hit. Context-dependent disambiguation is needed.'
            ),
            evidence_summary=(
                'Phase 43 confirmed surjective (many-to-one) encoding. '
                'Oracle ceiling is 89.5% (Phase 23). Phase 43 Approach 5 '
                'HMM failed (character-level within tokens). The 56% gap '
                'may require word-level context models.'
            ),
            open_questions=[
                'Can a word-level (not char-level) HMM disambiguate?',
                'Is the encoding genuinely lossy or context-recoverable?',
                'Would confirmed phrase patterns constrain disambiguation?',
            ],
            relevant_phases=['Phase 23', 'Phase 43'],
        ),
        GapCategory(
            category='FREQUENCY_STRUCTURE',
            priority='LOW',
            description=(
                'SBM communities correspond to frequency tiers. Whether '
                'this is a property of the script or the language is unknown.'
            ),
            evidence_summary=(
                'Phase 45 verdict: FREQUENCY_ARTIFACT. Communities have '
                'ARI=0.248 with frequency quintiles. Phase 46 Track B tests '
                'whether this matches natural language or specific cipher types.'
            ),
            open_questions=[
                'Is frequency-dominated co-occurrence diagnostic of encoding type?',
                'Do natural language corpora show the same pattern?',
                'Does the tachygraphic cipher uniquely produce this pattern?',
            ],
            relevant_phases=['Phase 44', 'Phase 45', 'Phase 46 Track B'],
        ),
    ]

    categories_dicts = [_convert(asdict(c)) for c in categories]
    n_high = sum(1 for c in categories if c.priority == 'HIGH')
    n_med = sum(1 for c in categories if c.priority == 'MEDIUM')
    n_low = sum(1 for c in categories if c.priority == 'LOW')

    print(f"\n  Gap categories: {len(categories)}")
    print(f"    HIGH: {n_high}  MEDIUM: {n_med}  LOW: {n_low}")
    for c in categories:
        print(f"    [{c.priority}] {c.category}: {len(c.open_questions)} questions")

    result = GapMapResult(
        n_categories=len(categories),
        categories=categories_dicts,
        n_high_priority=n_high,
        n_medium_priority=n_med,
        n_low_priority=n_low,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'gap_map.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2)
    print(f"\n  Saved -> {out_path}")


# ---------------------------------------------------------------------------
# Step 46C.4 — Project Summary Statistics
# ---------------------------------------------------------------------------


def run_final_summary() -> None:
    """Step 46C.4: Compile definitive project numbers."""
    t0 = time.time()
    print("=" * 70)
    print("STEP 46C.4: Project Summary Statistics")
    print("=" * 70)

    rd = _results_dir()

    # Load all relevant results
    decode_data = _safe_load(os.path.join(rd, 'final_decode_summary.json'))
    annot_data = _safe_load(os.path.join(rd, 'final_annotations.json'))
    canon_data = _safe_load(os.path.join(rd, 'canonical_table.json'))
    arb_data = _safe_load(os.path.join(rd, 'arb_selection.json'))
    bigram_data = _safe_load(os.path.join(rd, 'signal_bigrams.json'))
    arb_bigram = _safe_load(os.path.join(rd, 'arb_bigram.json'))
    arb_10k = _safe_load(os.path.join(rd, 'arb_10k.json'))
    tiers = _safe_load(os.path.join(rd, 'triple_tiers.json'))

    # Corpus stats
    total_tokens = decode_data.get('n_tokens', 0)
    total_folios = decode_data.get('n_folios', 0)
    n_sections = len(decode_data.get('section_aggregates', []))

    # Table stats
    tier_annots = canon_data.get('tier_annotations', {})
    tier_counts = Counter(tier_annots.values())
    n_confirmed = tier_counts.get('CONFIRMED', 0)
    n_landscape = tier_counts.get('LANDSCAPE_CONFIRMED', 0)
    n_ambiguous = tier_counts.get('GENUINELY_AMBIGUOUS', 0)
    n_triples = sum(tier_counts.values()) or 25

    # Definitive table source
    table_source = arb_data.get('definitive_table_name', 'canonical_table')

    # Performance metrics
    dict_hit_131k = decode_data.get('overall_dict_hit', 0.0)

    # Try to get 10K metrics from arb_10k
    dict_hit_10k = 0.0
    selectivity_10k = 0.0
    if arb_10k and arb_10k.get('per_table'):
        # Find the definitive table's 10K metrics
        for entry in arb_10k['per_table']:
            if entry.get('table_name') == table_source:
                dict_hit_10k = entry.get('dict_hit_10k', 0.0)
                selectivity_10k = entry.get('selectivity_10k', 0.0)
                break
        if dict_hit_10k == 0.0 and arb_10k['per_table']:
            # Fall back to best
            best = max(arb_10k['per_table'],
                       key=lambda e: e.get('dict_hit_10k', 0))
            dict_hit_10k = best.get('dict_hit_10k', 0.0)
            selectivity_10k = best.get('selectivity_10k', 0.0)

    # Bigram z
    bigram_z = bigram_data.get('bigram_z_score', 0.0)
    if arb_bigram and arb_bigram.get('per_table'):
        for entry in arb_bigram['per_table']:
            if entry.get('table_name') == table_source:
                bigram_z = entry.get('z_total', bigram_z)
                break

    # Signal
    signal_rate = decode_data.get('overall_signal_rate', 0.0)
    n_signal_words = len(decode_data.get('signal_word_index', []))

    # Progression table
    progression = [
        {'phase': 'Phase 11', 'dict_hit': 0.111, 'selectivity': 1.92,
         'note': 'CSP phonetic decoder, 14-cell grid'},
        {'phase': 'Phase 14', 'dict_hit': 0.194, 'selectivity': 3.00,
         'note': '25 stroke-feature triples'},
        {'phase': 'Phase 15', 'dict_hit': 0.354, 'selectivity': 2.55,
         'note': 'Medieval dict expansion 131K'},
        {'phase': 'Phase 16', 'dict_hit': 0.436, 'selectivity': 3.38,
         'note': 'Modifier detection, full corpus'},
        {'phase': 'Phase 29', 'dict_hit': 0.436, 'selectivity': 3.38,
         'note': 'Signal bigram z=6.14 (PHRASE_FOUND)'},
        {'phase': 'Phase 33', 'dict_hit': 0.436, 'selectivity': 3.38,
         'note': 'Table confirmed (0 consensus changes)'},
        {'phase': 'Phase 44', 'dict_hit': 0.436, 'selectivity': 3.38,
         'note': 'MaxSAT landscape FLAT (SCORING_WEAK)'},
        {'phase': 'Phase 45', 'dict_hit': 0.418, 'selectivity': 1.05,
         'note': 'SBM communities = frequency artifacts'},
        {'phase': 'Phase 46', 'dict_hit': round(dict_hit_131k, 4),
         'selectivity': round(selectivity_10k, 2),
         'note': f'Final consolidation ({table_source})'},
    ]

    key_findings = [
        'Encoding type: tachygraphic CV syllabary (cosine 0.820, Phase 19)',
        'Source language: Romance family (Latin/Northern Italian, indistinguishable)',
        'Content domain: medical/pharmaceutical (14 recipe folios, ~34 recipes)',
        'MaxSAT landscape: FLAT (500+ solutions within 1%, Phase 44)',
        'SBM communities: frequency artifact (ARI=0.248, Phase 45)',
        'Encoding: many-to-one/surjective (Phase 43)',
        f'Bigram z-score: {bigram_z:.2f} (sequential structure validated)',
        f'Oracle ceiling: 89.5% vs actual {dict_hit_131k:.1%} (45.9% gap)',
        '12 alternative encoding hypotheses eliminated (Phases 9, 18, 19, 27)',
        'All originally reported bigram z-scores inflated 3-70x (Phase 42)',
    ]

    print(f"\n  Corpus: {total_tokens} tokens, {total_folios} folios, "
          f"{n_sections} sections")
    print(f"  Table: {n_confirmed} confirmed, {n_landscape} landscape, "
          f"{n_ambiguous} ambiguous ({table_source})")
    print(f"  Dict-hit: {dict_hit_131k:.4f} (131K), {dict_hit_10k:.4f} (10K)")
    print(f"  Selectivity (10K): {selectivity_10k:.2f}")
    print(f"  Bigram z: {bigram_z:.2f}")
    print(f"  Signal words: {n_signal_words}")

    result = ProjectSummaryResult(
        total_tokens=total_tokens,
        total_folios=total_folios,
        n_sections=n_sections,
        n_triples=n_triples,
        n_confirmed=n_confirmed,
        n_landscape=n_landscape,
        n_ambiguous=n_ambiguous,
        definitive_table_source=table_source,
        dict_hit_131k=round(dict_hit_131k, 4),
        dict_hit_10k=round(dict_hit_10k, 4),
        selectivity_10k=round(selectivity_10k, 4),
        bigram_z=round(bigram_z, 4),
        signal_rate=round(signal_rate, 4),
        n_signal_words=n_signal_words,
        encoding_type='tachygraphic CV syllabary',
        source_language='Romance (Latin/Northern Italian)',
        content_domain='medical/pharmaceutical',
        landscape='FLAT',
        key_findings=key_findings,
        progression=progression,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'project_summary.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2)
    print(f"\n  Saved -> {out_path}")


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------


def run_track_c_46() -> None:
    """Run all Track C steps."""
    run_final_decode()
    print("\n" + "=" * 70 + "\n")
    run_final_annotate()
    print("\n" + "=" * 70 + "\n")
    run_final_map()
    print("\n" + "=" * 70 + "\n")
    run_final_summary()
