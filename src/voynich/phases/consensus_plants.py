"""
Phase 31.1: Consensus Plant Identification
============================================
From the multi-source concordance, identify folios where multiple independent
researchers agree on the same plant genus.  These high-confidence identifications
become the "city names" for the Ventris-style botanical anchor attack.

Dependency chain:
    data/reference/voynich_plant/Voynich_Herbal_Multi-Source_Identification_Concordance.csv
    data/reference/voynich_plant/medieval_latin_names.json
    results/illustration_constrained.json  (Phase 6.0 stem stats)
        → results/consensus_plants.json  (this step)
"""

import json
import math
import os
import re
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import load_corpus, tokenize_eva_chars
from voynich.phases.illustration_constrained import (
    parse_concordance,
    load_medieval_names,
    _extract_genus,
    build_corpus_stem_stats,
    compute_stem_specificity,
)
from voynich.phases.morpheme_grid import decompose_token_morphemes


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


# Researcher source normalisation — map CSV strings to canonical groups
_SOURCE_GROUP = {
    'General Botanical List': 'General',
    'General Botanical / Sherwood': 'General+Sherwood',  # ambiguous, treat as 1.5
    'Stephen Bax': 'Bax',
    'Tucker & Janick': 'Tucker',
    'Janick & Tucker': 'Tucker',
    'Tucker & Talbert': 'Tucker',
    'Edith Sherwood': 'Sherwood',
    'Anagram / Alternative Translation': 'Anagram',
    'Anagram Analysis': 'Anagram',
    'European Hypothesis': 'European',
    'Finnish Biologist / Bax': 'Bax',
    'Hugh O\'Neill / Tucker & Janick': 'ONeill+Tucker',
    'Gianinazzi / Bax': 'Gianinazzi+Bax',
    'General Botanical List (Missing Folio Theory)': 'General',
}

# New World plants — anachronistic for a 15th-century European manuscript
NEW_WORLD_GENERA: Set[str] = {
    'Musa',          # banana
    'Passiflora',    # passionflower
    'Psacalium',     # ragwort (New World)
    'Helianthus',    # sunflower
    'Lithophragma',  # woodland star (New World)
    'Duranta',       # golden dewdrop (New World)
    'Agave',         # maguey
    'Ipomoea',       # morning glory tree (New World sp.)
    'Telfairia',     # fluted pumpkin (African/tropical)
    'Pippenalia',    # Tucker & Janick hybrid
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ResearcherConsensus:
    """Genus-level consensus for a single folio."""
    folio: str
    genus: Optional[str]
    n_independent_sources: int
    sources: List[str]
    all_genera: Dict[str, int]
    tier: str  # 'A', 'B', 'C', 'X'
    is_anachronistic: bool


@dataclass
class MedievalNameEntry:
    """Medieval Latin name(s) for a plant genus."""
    genus: str
    linnaean_name: str
    medieval_name: str
    medieval_stem: str
    alternate_names: List[str]
    syllabified: List[str]  # CV syllables of medieval_name


@dataclass
class LabelCandidate:
    """A candidate EVA token that might encode the plant name."""
    token: str
    eva_chars: List[str]
    n_syllabic_chars: int
    tfidf_score: float
    is_first_line: bool
    is_unique_to_folio: bool
    stem: str
    label_likelihood: float


@dataclass
class FolioPlantEntry:
    """Complete entry for one consensus folio."""
    folio: str
    consensus: Dict
    medieval_names: List[Dict]
    label_candidates: List[Dict]
    n_tokens: int
    section: str


@dataclass
class ConsensusPlantResult:
    """Full Step 31.1 output."""
    n_concordance_entries: int
    n_folios_with_ids: int
    n_tier_a: int
    n_tier_b: int
    n_tier_c: int
    n_tier_x: int
    n_anachronistic_filtered: int
    tier_a_folios: List[Dict]
    tier_b_folios: List[Dict]
    all_folios: List[Dict]
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Source parsing and consensus scoring
# ---------------------------------------------------------------------------

def _normalise_source(raw_source: str) -> List[str]:
    """Map a raw CSV source string to independent researcher group(s).

    Some entries list combined sources (e.g. 'General Botanical / Sherwood')
    which count as multiple independent groups.
    """
    raw = raw_source.strip()

    # Direct lookup
    if raw in _SOURCE_GROUP:
        grp = _SOURCE_GROUP[raw]
        # Combined entries count as two independent sources
        if '+' in grp:
            return grp.split('+')
        return [grp]

    # Fallback — try partial matches
    groups = []
    if 'Bax' in raw:
        groups.append('Bax')
    if 'Tucker' in raw or 'Janick' in raw:
        groups.append('Tucker')
    if 'Sherwood' in raw:
        groups.append('Sherwood')
    if 'General' in raw:
        groups.append('General')
    if 'Anagram' in raw:
        groups.append('Anagram')
    if 'European' in raw:
        groups.append('European')
    if "O'Neill" in raw or 'O\'Neill' in raw:
        groups.append('ONeill')
    if 'Gianinazzi' in raw:
        groups.append('Gianinazzi')
    if 'Finnish' in raw:
        groups.append('Finnish')

    return groups if groups else ['Unknown']


def _compute_genus_consensus(
    identifications: List[Dict],
) -> ResearcherConsensus:
    """Score genus-level consensus for one folio.

    Returns ResearcherConsensus with tier classification.
    """
    folio = identifications[0]['folio']

    # For each genus, collect the independent source groups that identify it
    genus_sources: Dict[str, Set[str]] = defaultdict(set)

    for ident in identifications:
        genus = _extract_genus(ident['linnaean_name'])
        src_groups = _normalise_source(ident['source'])
        for g in src_groups:
            genus_sources[genus].add(g)

    # Find the genus with the most independent sources
    best_genus = None
    best_count = 0
    for genus, sources in genus_sources.items():
        if len(sources) > best_count:
            best_genus = genus
            best_count = len(sources)

    all_genera = {g: len(s) for g, s in genus_sources.items()}

    # Check anachronism
    is_anachronistic = best_genus in NEW_WORLD_GENERA if best_genus else False

    # Tier classification
    if best_count >= 3:
        tier = 'A'
    elif best_count == 2:
        tier = 'B' if not is_anachronistic else 'C'
    elif len(genus_sources) == 1:
        # Single identification — check if credible researcher
        sources = list(genus_sources[best_genus]) if best_genus else []
        credible = {'Bax', 'Tucker', 'Sherwood'}
        if any(s in credible for s in sources) and not is_anachronistic:
            tier = 'B'
        else:
            tier = 'C'
    elif len(genus_sources) > 1 and best_count == 1:
        # Multiple identifications, all disagree
        tier = 'X'
    else:
        tier = 'C'

    return ResearcherConsensus(
        folio=folio,
        genus=best_genus,
        n_independent_sources=best_count,
        sources=sorted(genus_sources.get(best_genus, set())) if best_genus else [],
        all_genera=all_genera,
        tier=tier,
        is_anachronistic=is_anachronistic,
    )


# ---------------------------------------------------------------------------
# Medieval name resolution and syllabification
# ---------------------------------------------------------------------------

def _syllabify_latin(word: str) -> List[str]:
    """Syllabify a Latin word into CV-like syllables.

    Uses a simple greedy approach: each consonant-vowel pair forms a syllable.
    Remaining consonant clusters attach to the following vowel.
    """
    word = word.lower().strip()
    vowels = set('aeiouy')
    result = []
    current = ''

    for ch in word:
        current += ch
        if ch in vowels:
            result.append(current)
            current = ''

    # Trailing consonants attach to last syllable
    if current and result:
        result[-1] += current
    elif current:
        result.append(current)

    return result


def _resolve_medieval_names(
    genus: str,
    linnaean_names: List[str],
    medieval_db: Dict[str, Dict],
) -> List[MedievalNameEntry]:
    """Resolve medieval Latin names for a genus from the pre-compiled database."""
    entries = []
    seen_names: Set[str] = set()

    for lname in linnaean_names:
        if lname not in medieval_db:
            continue

        info = medieval_db[lname]
        med_name = info.get('medieval_name')
        if not med_name or med_name in seen_names:
            continue

        seen_names.add(med_name)
        syllables = _syllabify_latin(med_name)

        entry = MedievalNameEntry(
            genus=genus,
            linnaean_name=lname,
            medieval_name=med_name,
            medieval_stem=info.get('medieval_stem', ''),
            alternate_names=info.get('alternate_names', []),
            syllabified=syllables,
        )
        entries.append(entry)

        # Also add alternate names
        for alt in info.get('alternate_names', []):
            if alt and alt not in seen_names:
                seen_names.add(alt)
                alt_syls = _syllabify_latin(alt)
                entries.append(MedievalNameEntry(
                    genus=genus,
                    linnaean_name=lname,
                    medieval_name=alt,
                    medieval_stem=alt[:max(len(alt) - 2, 3)],
                    alternate_names=[],
                    syllabified=alt_syls,
                ))

    return entries


# ---------------------------------------------------------------------------
# Label candidate extraction
# ---------------------------------------------------------------------------

def _extract_label_candidates(
    folio: str,
    corpus,
    stem_stats,
    n_top: int = 10,
) -> List[LabelCandidate]:
    """Rank tokens on a folio by likelihood of being the plant name label.

    Combines TF-IDF, first-line position, folio uniqueness, and morpheme stem.
    """
    page = corpus.pages.get(folio)
    if not page:
        return []

    tokens = page.all_tokens
    if not tokens:
        return []

    # Identify first-line tokens (first 8 tokens or until a clear break)
    first_line_tokens = set(tokens[:min(8, len(tokens))])

    # Get corpus-wide token frequency
    all_tokens = []
    for p in corpus.pages.values():
        all_tokens.extend(p.all_tokens)
    corpus_freq = Counter(all_tokens)

    # Get per-folio token frequency
    folio_freq = Counter(tokens)

    # Score each unique token
    candidates = []
    seen_tokens: Set[str] = set()

    for token in tokens:
        if token in seen_tokens:
            continue
        seen_tokens.add(token)

        eva_chars = tokenize_eva_chars(token)
        # Count syllabic chars (non-modifier)
        n_syl = len(eva_chars)  # Rough; modifiers will be filtered in plant_csp

        # TF-IDF via stem specificity
        decomp = decompose_token_morphemes(token)
        stem = decomp.stem if decomp.stem else token

        tfidf = 0.0
        if stem_stats and stem in stem_stats.corpus_freq:
            spec = compute_stem_specificity(stem, folio, stem_stats)
            tfidf = spec.tfidf

        is_first_line = token in first_line_tokens

        # Folio uniqueness: appears only on this folio or very rarely elsewhere
        folio_count = folio_freq[token]
        other_count = corpus_freq[token] - folio_count
        is_unique = other_count <= 2

        # Label likelihood: weighted combination
        likelihood = (
            tfidf * 0.4
            + (3.0 if is_first_line else 0.0) * 0.2
            + (3.0 if is_unique else 0.0) * 0.2
            + (1.0 / max(corpus_freq[token], 1)) * 0.2 * 100  # rarity bonus
        )

        candidates.append(LabelCandidate(
            token=token,
            eva_chars=eva_chars,
            n_syllabic_chars=n_syl,
            tfidf_score=round(tfidf, 4),
            is_first_line=is_first_line,
            is_unique_to_folio=is_unique,
            stem=stem,
            label_likelihood=round(likelihood, 4),
        ))

    # Sort by label likelihood
    candidates.sort(key=lambda c: c.label_likelihood, reverse=True)
    return candidates[:n_top]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_consensus_plants() -> None:
    """Step 31.1: Identify high-confidence plant identifications via multi-source consensus."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE 31.1: Consensus Plant Identification")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load concordance and medieval names ──
    print("\n  Loading concordance and medieval Latin names...")
    concordance = parse_concordance()
    medieval_db = load_medieval_names()

    n_entries = sum(len(ids) for ids in concordance.values())
    n_folios = len(concordance)
    print(f"     {n_entries} identifications across {n_folios} folios")
    print(f"     {len(medieval_db)} medieval Latin names resolved")

    # ── 2. Compute genus consensus per folio ──
    print("\n  Computing genus-level consensus...")
    consensus_map: Dict[str, ResearcherConsensus] = {}
    for folio, ids in sorted(concordance.items()):
        consensus_map[folio] = _compute_genus_consensus(ids)

    # Count tiers
    tier_counts = Counter(c.tier for c in consensus_map.values())
    n_anachron = sum(1 for c in consensus_map.values() if c.is_anachronistic)

    print(f"     Tier A (≥3 sources agree): {tier_counts.get('A', 0)} folios")
    print(f"     Tier B (2 sources or credible single): {tier_counts.get('B', 0)} folios")
    print(f"     Tier C (single low-confidence): {tier_counts.get('C', 0)} folios")
    print(f"     Tier X (contested genus): {tier_counts.get('X', 0)} folios")
    print(f"     Anachronistic (filtered): {n_anachron}")

    # ── 3. List Tier A and B folios ──
    tier_ab = [c for c in consensus_map.values()
               if c.tier in ('A', 'B') and not c.is_anachronistic]
    tier_ab.sort(key=lambda c: (c.tier, c.folio))

    print(f"\n  Tier A+B usable folios: {len(tier_ab)}")
    for c in tier_ab:
        print(f"     {c.folio}: {c.genus} (Tier {c.tier}, "
              f"{c.n_independent_sources} sources: {', '.join(c.sources)})")

    # ── 4. Resolve medieval names for Tier A+B ──
    print("\n  Resolving medieval Latin names for anchor folios...")
    corpus = load_corpus()

    # Build corpus stem stats for label candidate ranking
    stem_stats = build_corpus_stem_stats(corpus)

    folio_entries: List[FolioPlantEntry] = []
    tier_a_entries: List[Dict] = []
    tier_b_entries: List[Dict] = []

    for cons in tier_ab:
        # Collect all Linnaean names for this genus on this folio
        folio_ids = concordance.get(cons.folio, [])
        linnaean_for_genus = [
            ident['linnaean_name'] for ident in folio_ids
            if _extract_genus(ident['linnaean_name']) == cons.genus
        ]

        # Resolve medieval names
        med_names = _resolve_medieval_names(
            cons.genus, linnaean_for_genus, medieval_db,
        )

        if not med_names:
            # Try genus name itself as fallback (many Linnaean genera = medieval names)
            genus_lower = cons.genus.lower()
            med_names = [MedievalNameEntry(
                genus=cons.genus,
                linnaean_name=cons.genus,
                medieval_name=genus_lower,
                medieval_stem=genus_lower[:max(len(genus_lower) - 2, 3)],
                alternate_names=[],
                syllabified=_syllabify_latin(genus_lower),
            )]

        # Extract label candidates
        label_cands = _extract_label_candidates(
            cons.folio, corpus, stem_stats, n_top=10,
        )

        # Get section
        page = corpus.pages.get(cons.folio)
        section = page.section if page else 'unknown'
        n_tokens = len(page.all_tokens) if page else 0

        entry = FolioPlantEntry(
            folio=cons.folio,
            consensus=asdict(cons),
            medieval_names=[asdict(m) for m in med_names],
            label_candidates=[asdict(lc) for lc in label_cands],
            n_tokens=n_tokens,
            section=section,
        )
        folio_entries.append(entry)

        entry_dict = _convert(asdict(entry))
        if cons.tier == 'A':
            tier_a_entries.append(entry_dict)
        else:
            tier_b_entries.append(entry_dict)

        # Print summary
        name_str = ', '.join(m.medieval_name for m in med_names[:3])
        top_label = label_cands[0].token if label_cands else '?'
        print(f"     {cons.folio} ({cons.genus}): medieval={name_str}, "
              f"top_label={top_label}, {n_tokens} tokens")

    # ── 5. Build all-folios summary ──
    all_folio_entries = []
    for folio, cons in sorted(consensus_map.items()):
        all_folio_entries.append(_convert(asdict(cons)))

    # ── 6. Save results ──
    result = ConsensusPlantResult(
        n_concordance_entries=n_entries,
        n_folios_with_ids=n_folios,
        n_tier_a=tier_counts.get('A', 0),
        n_tier_b=len([c for c in consensus_map.values()
                      if c.tier == 'B' and not c.is_anachronistic]),
        n_tier_c=tier_counts.get('C', 0),
        n_tier_x=tier_counts.get('X', 0),
        n_anachronistic_filtered=n_anachron,
        tier_a_folios=tier_a_entries,
        tier_b_folios=tier_b_entries,
        all_folios=all_folio_entries,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'consensus_plants.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2)

    elapsed = time.time() - t0
    print(f"\n  Result: {result.n_tier_a} Tier A + {result.n_tier_b} Tier B "
          f"anchor folios identified")
    print(f"  Saved → {out_path}")
    print(f"  Completed in {elapsed:.1f}s")
