"""
Phase 52 Track A: Word-Level Identification Catalog
====================================================
Re-run the bridge search from Phase 51B to recover ALL match records,
group by (EVA_type, matched_word) pair, score/tier each pair, and
compile a unified vocabulary catalog.

Dependency chain:
    signal_bigrams.json        (Step 29.1)
    combined_refine.json       (Step 15)
    modifier_integrate.json    (Step 16)
    triple_tiers.json          (Step 44)
        -> word_catalog.json   (this step)
"""

from __future__ import annotations

import json
import os
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import build_eva_to_triple_lookup, tokenize_eva_chars
from voynich.phases.concatenation_bridge import (
    BridgeMatch,
    _build_partial_decode,
    _build_pharma_dict,
    _extract_implied_assignments,
    _search_dict,
)
from voynich.phases.suffix_calibration import SIGNAL_WORDS_51, SIGNAL_WORDS_SET


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
    if isinstance(obj, set):
        return sorted(_convert(item) for item in obj)
    if isinstance(obj, (bool, int, float, str, type(None))):
        return obj
    return str(obj)


def _save_json(rd: str, filename: str, data: Any) -> str:
    path = os.path.join(rd, filename)
    with open(path, 'w') as f:
        json.dump(_convert(data), f, indent=2)
    return path


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class WordIdentification:
    eva_type: str
    latin_word: str
    pattern: str
    tier: str                        # T1, T2, T3, REJECT
    confidence: float
    n_folios: int
    folios: List[str]
    n_anchors: int
    anchors: List[str]
    n_bridge_instances: int
    total_corpus_count: int
    mean_alternatives: float
    implied_assignments: Dict[str, str]
    is_ambiguous: bool
    competing_words: List[str]
    # Sub-scores
    folio_score: float
    uniqueness_score: float
    frequency_score: float
    anchor_diversity: float
    length_score: float


@dataclass
class WordCatalogResult:
    n_bridge_matches_total: int
    n_concat_matches_total: int
    n_unique_pairs: int
    single_token_ids: List[Dict]
    multi_token_ids: List[Dict]
    signal_words: List[Dict]
    n_tier1: int
    n_tier2: int
    n_tier3: int
    n_rejected: int
    n_ambiguous: int
    corpus_coverage: float
    n_corpus_tokens: int
    pharma_dict_size: int
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Bridge search (replicated from concatenation_bridge.py to store ALL matches)
# ---------------------------------------------------------------------------

def _run_full_bridge_search(
    token_evas: List[str],
    token_decoded: List[str],
    token_classifications: List[str],
    token_folios: List[str],
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    modifier_chars: Set[str],
    confirmed_triples: Set[str],
    pharma_dict: Set[str],
) -> Tuple[List[BridgeMatch], List[BridgeMatch]]:
    """Run bridge search storing ALL matches (not just top 30)."""
    n_tokens = len(token_evas)

    # Find all SIGNAL token positions
    signal_positions = set()
    for i in range(n_tokens):
        if token_decoded[i] in SIGNAL_WORDS_SET:
            signal_positions.add(i)

    # ── Bridge matches ──
    bridge_matches: List[BridgeMatch] = []
    seen_dark_tokens: Set[int] = set()

    for sig_idx in sorted(signal_positions):
        anchor_word = token_decoded[sig_idx]

        for dist in [1, 2]:
            for offset, position in [(-dist, 'before'), (dist, 'after')]:
                nbr_idx = sig_idx + offset
                if nbr_idx < 0 or nbr_idx >= n_tokens:
                    continue
                if nbr_idx in signal_positions:
                    continue
                if nbr_idx in seen_dark_tokens:
                    continue

                seen_dark_tokens.add(nbr_idx)
                dark_eva = token_evas[nbr_idx]

                pattern, details = _build_partial_decode(
                    dark_eva, assignment, eva_to_triple,
                    modifier_chars, confirmed_triples,
                )

                n_conf = sum(1 for _, _, _, c in details if c)
                n_free = sum(1 for _, _, _, c in details if not c)

                if n_conf < 1 or n_free < 1 or n_free > 3:
                    continue

                matches = _search_dict(pattern, pharma_dict)
                if not matches:
                    continue

                for mword in matches:
                    implied = _extract_implied_assignments(
                        pattern, mword, details, eva_to_triple,
                    )
                    bridge_matches.append(BridgeMatch(
                        token_idx=nbr_idx,
                        token_eva=dark_eva,
                        pattern=pattern,
                        matched_word=mword,
                        n_confirmed_chars=n_conf,
                        n_free_chars=n_free,
                        implied_assignments=implied,
                        anchor_word=anchor_word,
                        anchor_position=position,
                        distance=dist,
                        folio=token_folios[nbr_idx],
                        n_total_matches=len(matches),
                    ))

    # ── Concatenation matches ──
    concat_matches: List[BridgeMatch] = []

    for sig_idx in sorted(signal_positions):
        anchor_word = token_decoded[sig_idx]

        for offset, position in [(-1, 'before'), (1, 'after')]:
            nbr_idx = sig_idx + offset
            if nbr_idx < 0 or nbr_idx >= n_tokens:
                continue
            if nbr_idx in signal_positions:
                continue

            dark_eva = token_evas[nbr_idx]
            pattern, details = _build_partial_decode(
                dark_eva, assignment, eva_to_triple,
                modifier_chars, confirmed_triples,
            )

            if not any(not c for _, _, _, c in details):
                continue

            if position == 'after':
                concat_pattern = anchor_word + pattern
            else:
                concat_pattern = pattern + anchor_word

            concat_hits = _search_dict(concat_pattern, pharma_dict)
            for mword in concat_hits:
                concat_matches.append(BridgeMatch(
                    token_idx=nbr_idx,
                    token_eva=dark_eva,
                    pattern=concat_pattern,
                    matched_word=mword,
                    n_confirmed_chars=sum(1 for _, _, _, c in details if c),
                    n_free_chars=sum(1 for _, _, _, c in details if not c),
                    implied_assignments={},
                    anchor_word=anchor_word,
                    anchor_position=position,
                    distance=1,
                    folio=token_folios[nbr_idx],
                    n_total_matches=len(concat_hits),
                ))

    return bridge_matches, concat_matches


# ---------------------------------------------------------------------------
# Scoring and tiering
# ---------------------------------------------------------------------------

def _score_and_tier(
    word_pairs: Dict[Tuple[str, str], Dict],
    eva_corpus_freq: Counter,
) -> List[WordIdentification]:
    """Score each (EVA_type, matched_word) pair and assign tiers."""
    ids: List[WordIdentification] = []

    for (eva_type, latin_word), data in word_pairs.items():
        n_folios = len(data['folios'])
        n_anchors = len(data['anchors'])
        total_corpus_count = eva_corpus_freq.get(eva_type, 0)
        pattern_uniq = data['pattern_uniqueness']
        mean_alt = sum(pattern_uniq) / len(pattern_uniq) if pattern_uniq else 1.0

        folio_score = min(n_folios / 5.0, 1.0)
        uniqueness_score = 1.0 / max(mean_alt, 1.0)
        frequency_score = min(total_corpus_count / 20.0, 1.0)
        anchor_diversity = min(n_anchors / 3.0, 1.0)
        length_score = min(max(len(latin_word) - 3, 0) / 5.0, 1.0)

        confidence = (
            0.30 * folio_score
            + 0.25 * uniqueness_score
            + 0.20 * frequency_score
            + 0.15 * anchor_diversity
            + 0.10 * length_score
        )

        # Tier classification
        if (confidence >= 0.7 and n_folios >= 3
                and all(p == 1 for p in pattern_uniq)):
            tier = 'T1'
        elif confidence >= 0.5 and n_folios >= 2:
            tier = 'T2'
        elif confidence >= 0.3:
            tier = 'T3'
        else:
            tier = 'REJECT'

        ids.append(WordIdentification(
            eva_type=eva_type,
            latin_word=latin_word,
            pattern=data.get('pattern', ''),
            tier=tier,
            confidence=round(confidence, 4),
            n_folios=n_folios,
            folios=sorted(data['folios']),
            n_anchors=n_anchors,
            anchors=sorted(data['anchors']),
            n_bridge_instances=data['count'],
            total_corpus_count=total_corpus_count,
            mean_alternatives=round(mean_alt, 3),
            implied_assignments=data.get('implied', {}),
            is_ambiguous=False,
            competing_words=[],
            folio_score=round(folio_score, 4),
            uniqueness_score=round(uniqueness_score, 4),
            frequency_score=round(frequency_score, 4),
            anchor_diversity=round(anchor_diversity, 4),
            length_score=round(length_score, 4),
        ))

    return ids


def _detect_ambiguity(ids: List[WordIdentification]) -> List[WordIdentification]:
    """Flag EVA types that map to multiple words with similar confidence."""
    by_eva: Dict[str, List[WordIdentification]] = defaultdict(list)
    for wid in ids:
        if wid.tier != 'REJECT':
            by_eva[wid.eva_type].append(wid)

    for eva_type, candidates in by_eva.items():
        if len(candidates) < 2:
            continue
        candidates.sort(key=lambda w: w.confidence, reverse=True)
        top = candidates[0].confidence
        second = candidates[1].confidence
        if second > 0 and top / second < 2.0:
            competing = [c.latin_word for c in candidates]
            for c in candidates:
                c.is_ambiguous = True
                c.competing_words = [w for w in competing if w != c.latin_word]

    return ids


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_word_catalog() -> None:
    """Phase 52 Track A: Word-Level Identification Catalog."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE 52 TRACK A: Word-Level Identification Catalog")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load inputs ──────────────────────────────────────────────
    print("\n  A.1  Loading inputs...")

    with open(os.path.join(rd, 'signal_bigrams.json')) as f:
        bigram_data = json.load(f)
    token_evas = bigram_data['token_evas']
    token_decoded = bigram_data['token_decoded']
    token_classifications = bigram_data['token_classifications']
    token_folios = bigram_data['token_folios']
    n_tokens = len(token_evas)

    with open(os.path.join(rd, 'combined_refine.json')) as f:
        refine_data = json.load(f)
    assignment = refine_data['best_assignment']

    with open(os.path.join(rd, 'modifier_integrate.json')) as f:
        mod_data = json.load(f)
    modifier_chars = set(mod_data['modifier_chars'])

    with open(os.path.join(rd, 'triple_tiers.json')) as f:
        tiers_data = json.load(f)

    confirmed_triples: Set[str] = set()
    for entry in tiers_data['tiers'].get('CONFIRMED', []):
        confirmed_triples.add(entry['triple_key'])
    # Note: LANDSCAPE_CONFIRMED excluded — original Phase 51B only used
    # CONFIRMED triples for partial decode (more wildcards = more matches)

    eva_to_triple = build_eva_to_triple_lookup()

    print(f"       {n_tokens} tokens")
    print(f"       {len(assignment)} triple assignments")
    print(f"       {len(modifier_chars)} modifier chars")
    print(f"       {len(confirmed_triples)} confirmed triples")

    # ── 2. Build pharmaceutical dictionary ───────────────────────────
    print("\n  A.2  Building pharmaceutical dictionary...")
    pharma_dict = _build_pharma_dict()
    print(f"       Dictionary size: {len(pharma_dict)} words")

    # ── 3. Re-run full bridge search ─────────────────────────────────
    print("\n  A.3  Running full bridge search (all matches)...")

    bridge_matches, concat_matches = _run_full_bridge_search(
        token_evas, token_decoded, token_classifications, token_folios,
        assignment, eva_to_triple, modifier_chars, confirmed_triples,
        pharma_dict,
    )

    print(f"       Bridge matches: {len(bridge_matches)}")
    print(f"       Concat matches: {len(concat_matches)}")

    # ── 4. Group by (EVA_type, matched_word) ─────────────────────────
    print("\n  A.4  Grouping by (EVA_type, matched_word)...")

    word_pairs: Dict[Tuple[str, str], Dict] = defaultdict(lambda: {
        'folios': set(), 'positions': [], 'anchors': set(),
        'count': 0, 'pattern_uniqueness': [], 'implied': {},
        'pattern': '',
    })

    for bm in bridge_matches:
        key = (bm.token_eva, bm.matched_word)
        word_pairs[key]['folios'].add(bm.folio)
        word_pairs[key]['positions'].append(bm.token_idx)
        word_pairs[key]['anchors'].add(bm.anchor_word)
        word_pairs[key]['count'] += 1
        word_pairs[key]['pattern_uniqueness'].append(bm.n_total_matches)
        if bm.implied_assignments:
            word_pairs[key]['implied'].update(bm.implied_assignments)
        if not word_pairs[key]['pattern']:
            word_pairs[key]['pattern'] = bm.pattern

    print(f"       Unique (EVA, word) pairs: {len(word_pairs)}")

    # ── 5. Count total corpus frequency per EVA type ─────────────────
    print("\n  A.5  Counting corpus frequencies...")
    eva_corpus_freq = Counter(token_evas)

    # ── 6. Score and tier ────────────────────────────────────────────
    print("\n  A.6  Scoring and tiering...")
    all_ids = _score_and_tier(dict(word_pairs), eva_corpus_freq)
    all_ids = _detect_ambiguity(all_ids)

    tier_counts = Counter(wid.tier for wid in all_ids)
    print(f"       T1: {tier_counts.get('T1', 0)}")
    print(f"       T2: {tier_counts.get('T2', 0)}")
    print(f"       T3: {tier_counts.get('T3', 0)}")
    print(f"       REJECT: {tier_counts.get('REJECT', 0)}")
    print(f"       Ambiguous: {sum(1 for w in all_ids if w.is_ambiguous)}")

    # Show top T1/T2 identifications
    top_ids = sorted(
        [w for w in all_ids if w.tier in ('T1', 'T2')],
        key=lambda w: w.confidence, reverse=True,
    )
    if top_ids:
        print("\n       Top identifications:")
        for wid in top_ids[:20]:
            print(f"         {wid.eva_type:15s} → {wid.latin_word:15s} "
                  f"tier={wid.tier} conf={wid.confidence:.3f} "
                  f"folios={wid.n_folios} alt={wid.mean_alternatives:.1f}")

    # ── 7. Concatenation IDs (multi-token) ───────────────────────────
    print("\n  A.7  Grouping concatenation matches...")

    concat_pairs: Dict[Tuple[str, str], Dict] = defaultdict(lambda: {
        'folios': set(), 'positions': [], 'anchors': set(),
        'count': 0, 'pattern_uniqueness': [], 'pattern': '',
        'anchor_word': '', 'dark_eva': '',
    })

    for cm in concat_matches:
        key = (cm.anchor_word + '+' + cm.token_eva, cm.matched_word)
        concat_pairs[key]['folios'].add(cm.folio)
        concat_pairs[key]['positions'].append(cm.token_idx)
        concat_pairs[key]['anchors'].add(cm.anchor_word)
        concat_pairs[key]['count'] += 1
        concat_pairs[key]['pattern_uniqueness'].append(cm.n_total_matches)
        if not concat_pairs[key]['pattern']:
            concat_pairs[key]['pattern'] = cm.pattern
        concat_pairs[key]['anchor_word'] = cm.anchor_word
        concat_pairs[key]['dark_eva'] = cm.token_eva

    # Score concat pairs using same logic
    concat_ids = _score_and_tier(dict(concat_pairs), eva_corpus_freq)
    concat_tier_counts = Counter(wid.tier for wid in concat_ids)
    print(f"       Concat pairs: {len(concat_pairs)}")
    print(f"       Concat T1: {concat_tier_counts.get('T1', 0)}")
    print(f"       Concat T2: {concat_tier_counts.get('T2', 0)}")
    print(f"       Concat T3: {concat_tier_counts.get('T3', 0)}")

    # ── 8. Compute corpus coverage ───────────────────────────────────
    print("\n  A.8  Computing corpus coverage...")

    # All EVA types in catalog (T1+T2+T3, single-token only)
    catalog_eva_types = set()
    for wid in all_ids:
        if wid.tier != 'REJECT':
            catalog_eva_types.add(wid.eva_type)

    # Signal word decoded forms
    signal_decoded = set(SIGNAL_WORDS_51.keys())

    n_glossed = 0
    for i in range(n_tokens):
        if token_decoded[i] in signal_decoded:
            n_glossed += 1
        elif token_evas[i] in catalog_eva_types:
            n_glossed += 1

    coverage = n_glossed / n_tokens if n_tokens > 0 else 0.0
    print(f"       Signal words cover: "
          f"{sum(1 for i in range(n_tokens) if token_decoded[i] in signal_decoded)} tokens")
    print(f"       Catalog adds: "
          f"{sum(1 for i in range(n_tokens) if token_decoded[i] not in signal_decoded and token_evas[i] in catalog_eva_types)} tokens")
    print(f"       Total glossed: {n_glossed} / {n_tokens} ({coverage:.1%})")

    # ── 9. Build signal words list for catalog ───────────────────────
    signal_list = []
    for word, info in SIGNAL_WORDS_51.items():
        signal_list.append({
            'decoded': word,
            'gloss': info['gloss'],
            'type': info['type'],
            'lang': info['lang'],
            'sigma': info['sigma'],
            'real_count': info['real_count'],
            'tier': 'TIER_0',
        })

    # ── 10. Save ─────────────────────────────────────────────────────
    runtime = round(time.time() - t0, 2)

    result = WordCatalogResult(
        n_bridge_matches_total=len(bridge_matches),
        n_concat_matches_total=len(concat_matches),
        n_unique_pairs=len(word_pairs),
        single_token_ids=[asdict(w) for w in sorted(all_ids,
                          key=lambda w: w.confidence, reverse=True)
                          if w.tier != 'REJECT'],
        multi_token_ids=[asdict(w) for w in sorted(concat_ids,
                         key=lambda w: w.confidence, reverse=True)
                         if w.tier != 'REJECT'],
        signal_words=signal_list,
        n_tier1=tier_counts.get('T1', 0),
        n_tier2=tier_counts.get('T2', 0),
        n_tier3=tier_counts.get('T3', 0),
        n_rejected=tier_counts.get('REJECT', 0),
        n_ambiguous=sum(1 for w in all_ids if w.is_ambiguous),
        corpus_coverage=round(coverage, 4),
        n_corpus_tokens=n_tokens,
        pharma_dict_size=len(pharma_dict),
        runtime_seconds=runtime,
    )

    out_path = _save_json(rd, 'word_catalog.json', asdict(result))
    print(f"\n  Saved → {out_path}")
    print(f"  Completed in {runtime:.1f}s")
