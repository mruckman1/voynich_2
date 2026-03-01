"""
Phase 6.0: Illustration-Constrained Setup
==========================================
Parse the botanical concordance, map Linnaean names to medieval Latin,
classify identification confidence tiers, and extract per-folio dominant
stems from the Voynich corpus.

This module is the foundation for all Phase 6 analyses. Every other Phase 6
module depends on its output.

Sub-analyses:
  6.0a — Concordance parsing and medieval name resolution
  6.0b — Folio identification tier classification
  6.0c — Per-folio dominant stem extraction

Output:
  results/illustration_constrained.json
"""

import csv
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from voynich.core.corpus import load_corpus, VoynichCorpus, VoynichPage
from voynich.core._paths import data_dir as _data_dir, results_dir as _results_dir
from voynich.core.reference import (
    infer_declension, expected_paradigm_shape, extract_latin_stem,
    LATIN_DECLENSION_SUFFIXES,
)
from voynich.phases.morpheme_grid import (
    decompose_token_morphemes, MorphemeDecomposition,
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PlantIdentification:
    """One researcher's identification of a folio's illustration."""
    folio: str
    linnaean_name: str
    common_name: str
    source: str
    medieval_name: Optional[str] = None
    medieval_stem: Optional[str] = None
    declension: Optional[str] = None
    alternate_stems: List[str] = field(default_factory=list)


@dataclass
class StemSpecificityScore:
    """Specificity scores for a stem on a particular folio."""
    stem: str
    folio: str
    tf: int                 # term frequency on this folio
    cf: int                 # corpus frequency (total across all folios)
    df: int                 # document frequency (number of folios stem appears on)
    tfidf: float            # TF-IDF score
    specificity_ratio: float  # tf / cf — fraction of corpus occurrences on this folio
    exclusivity: float      # tf / (cf - tf + 1) — this folio vs all others
    pmi: float              # log(P(S,F) / (P(S) * P(F)))


@dataclass
class FolioIdentificationSet:
    """All identifications for a single folio."""
    folio: str
    identifications: List[PlantIdentification]
    n_sources: int
    consensus_genus: Optional[str] = None
    tier: int = 3
    dominant_stem: Optional[str] = None
    dominant_stem_forms: List[str] = field(default_factory=list)
    dominant_stem_paradigm_shape: Optional[Tuple[int, int]] = None
    dominant_stem_token_count: int = 0
    all_stems: Dict[str, int] = field(default_factory=dict)
    token_count: int = 0
    # Phase 6.1: specificity-based stem extraction
    specificity_stem: Optional[str] = None
    specificity_stem_forms: List[str] = field(default_factory=list)
    specificity_stem_paradigm_shape: Optional[Tuple[int, int]] = None
    specificity_stem_scores: Optional[Dict[str, float]] = None
    stem_changed: bool = False  # True if TF-IDF picked a different stem


@dataclass
class IllustrationConstrainedResult:
    """Full Phase 6.0 output."""
    n_concordance_entries: int
    n_folios_with_ids: int
    n_unique_plants: int
    n_medieval_names_resolved: int
    n_unresolvable: int
    tier_distribution: Dict[int, int]
    folios: List[Dict]
    herbal_a_folios: int
    herbal_a_with_ids: int
    coverage: float
    gate_passed: bool
    verdict: str


# ---------------------------------------------------------------------------
# Concordance parsing
# ---------------------------------------------------------------------------

def parse_concordance(csv_path: Optional[str] = None) -> Dict[str, List[Dict]]:
    """
    Parse the concordance CSV into {folio: [identification_dicts]}.

    Each entry has keys: folio, linnaean_name, common_name, source.
    """
    if csv_path is None:
        csv_path = os.path.join(
            _data_dir('reference/voynich_plant'),
            'Voynich_Herbal_Multi-Source_Identification_Concordance.csv',
        )

    result: Dict[str, List[Dict]] = defaultdict(list)

    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            folio = row['Folio'].strip()
            entry = {
                'folio': folio,
                'linnaean_name': row['Proposed Botanical Identification'].strip(),
                'common_name': row['Common Name'].strip(),
                'source': row['Principal Researcher / Source'].strip(),
            }
            result[folio].append(entry)

    return dict(result)


def load_medieval_names(json_path: Optional[str] = None) -> Dict[str, Dict]:
    """
    Load the Linnaean -> medieval Latin mapping.

    Returns {linnaean_name: {medieval_name, medieval_stem, declension, ...}}.
    """
    if json_path is None:
        json_path = os.path.join(
            _data_dir('reference/voynich_plant'),
            'medieval_latin_names.json',
        )

    with open(json_path, encoding='utf-8') as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Tier classification
# ---------------------------------------------------------------------------

def _extract_genus(linnaean_name: str) -> str:
    """Extract genus from a Linnaean binomial (first word)."""
    parts = linnaean_name.strip().split()
    return parts[0] if parts else linnaean_name


def classify_identification_tier(identifications: List[Dict]) -> Tuple[int, Optional[str]]:
    """
    Classify folio identification confidence tier.

    Tier 1 (consensus):  2+ sources agree on genus.
    Tier 2 (single-high): Only 1 source but from a credible researcher
                          (Bax, Tucker & Janick, Sherwood).
    Tier 3 (contested):  Multiple sources disagree on genus, or single
                         low-confidence source.

    Returns (tier, consensus_genus_or_None).
    """
    if not identifications:
        return 3, None

    # Extract genera from each identification
    genera = [_extract_genus(ident['linnaean_name']) for ident in identifications]
    genus_counts = Counter(genera)

    # Single identification
    if len(identifications) == 1:
        source = identifications[0]['source']
        high_confidence_sources = {
            'Stephen Bax', 'Tucker & Janick', 'Janick & Tucker',
            'Edith Sherwood',
        }
        if source in high_confidence_sources:
            return 2, genera[0]
        return 3, genera[0]

    # Multiple identifications — check genus agreement
    most_common_genus, most_common_count = genus_counts.most_common(1)[0]

    if most_common_count >= 2:
        # 2+ sources agree on genus -> Tier 1
        return 1, most_common_genus

    # All different genera
    if len(genus_counts) == len(identifications):
        return 3, None

    # Partial agreement but not enough
    return 3, None


# ---------------------------------------------------------------------------
# Dominant stem extraction
# ---------------------------------------------------------------------------

def extract_folio_dominant_stem(
    page: VoynichPage,
    token_counts: Optional[Dict[str, int]] = None,
) -> Tuple[Optional[str], List[str], Optional[Tuple[int, int]], int, Dict[str, int]]:
    """
    Find the dominant stem on a folio page.

    Returns (dominant_stem, forms, paradigm_shape, stem_token_count, all_stems_dict).
    """
    tokens = page.all_tokens
    if not tokens:
        return None, [], None, 0, {}

    # Decompose all tokens
    decomps: List[MorphemeDecomposition] = []
    for token in tokens:
        d = decompose_token_morphemes(token)
        if d.stem:
            decomps.append(d)

    if not decomps:
        return None, [], None, 0, {}

    # Build per-token counts if not provided
    if token_counts is None:
        token_counts = Counter(tokens)

    # Group by stem
    stem_groups: Dict[str, List[MorphemeDecomposition]] = defaultdict(list)
    for d in decomps:
        stem_groups[d.stem].append(d)

    # Score each stem by total token count
    stem_scores: Dict[str, int] = {}
    for stem, group in stem_groups.items():
        total = sum(token_counts.get(d.token, 1) for d in group)
        stem_scores[stem] = total

    # Find dominant stem
    if not stem_scores:
        return None, [], None, 0, {}

    dominant_stem = max(stem_scores, key=stem_scores.get)
    dominant_group = stem_groups[dominant_stem]

    forms = sorted(set(d.token for d in dominant_group))
    prefixes = set(d.prefix for d in dominant_group if d.prefix)
    suffixes = set(d.suffix for d in dominant_group if d.suffix)
    paradigm_shape = (len(prefixes), len(suffixes))
    stem_count = stem_scores[dominant_stem]

    return dominant_stem, forms, paradigm_shape, stem_count, stem_scores


# ---------------------------------------------------------------------------
# Phase 6.1: Corpus-wide stem statistics for TF-IDF extraction
# ---------------------------------------------------------------------------

@dataclass
class CorpusStemStats:
    """Corpus-wide stem frequency and document frequency statistics."""
    # stem -> total corpus count
    corpus_freq: Dict[str, int]
    # stem -> number of folios it appears on
    doc_freq: Dict[str, int]
    # Total number of folios
    n_folios: int
    # folio -> {stem: count}
    folio_stem_counts: Dict[str, Dict[str, int]]
    # folio -> total token count
    folio_token_counts: Dict[str, int]
    # Total corpus tokens
    total_tokens: int


def build_corpus_stem_stats(
    corpus: 'VoynichCorpus',
    section: str = 'herbal_a',
) -> CorpusStemStats:
    """
    Build corpus-wide stem statistics for TF-IDF computation.

    Decomposes all tokens in the specified section, computes per-folio
    and corpus-wide stem frequencies and document frequencies.
    """
    corpus_freq: Dict[str, int] = Counter()
    doc_freq: Dict[str, int] = Counter()
    folio_stem_counts: Dict[str, Dict[str, int]] = {}
    folio_token_counts: Dict[str, int] = {}
    total_tokens = 0

    pages = [p for p in corpus.pages.values() if p.section == section]

    for page in pages:
        tokens = page.all_tokens
        folio_token_counts[page.folio] = len(tokens)
        total_tokens += len(tokens)

        stem_counts: Dict[str, int] = Counter()
        for token in tokens:
            d = decompose_token_morphemes(token)
            if d.stem:
                stem_counts[d.stem] += 1

        folio_stem_counts[page.folio] = dict(stem_counts)

        for stem, count in stem_counts.items():
            corpus_freq[stem] += count
            doc_freq[stem] += 1

    return CorpusStemStats(
        corpus_freq=dict(corpus_freq),
        doc_freq=dict(doc_freq),
        n_folios=len(pages),
        folio_stem_counts=folio_stem_counts,
        folio_token_counts=folio_token_counts,
        total_tokens=total_tokens,
    )


def compute_stem_specificity(
    stem: str,
    folio: str,
    stats: CorpusStemStats,
) -> StemSpecificityScore:
    """
    Compute all four specificity metrics for a stem on a folio.

    Returns a StemSpecificityScore with TF-IDF, specificity ratio,
    exclusivity ratio, and PMI.
    """
    tf = stats.folio_stem_counts.get(folio, {}).get(stem, 0)
    cf = stats.corpus_freq.get(stem, 0)
    df = stats.doc_freq.get(stem, 0)
    n = stats.n_folios
    folio_tokens = stats.folio_token_counts.get(folio, 1)

    # TF-IDF: tf(S,F) * log(N / df(S))
    tfidf = tf * math.log(n / max(df, 1)) if tf > 0 else 0.0

    # Specificity ratio: tf / cf (fraction of total occurrences on this folio)
    spec_ratio = tf / max(cf, 1)

    # Exclusivity: tf / (cf - tf + 1) (this folio vs all others)
    exclusivity = tf / max(cf - tf + 1, 1)

    # PMI: log(P(S,F) / (P(S) * P(F)))
    p_sf = tf / max(stats.total_tokens, 1)
    p_s = cf / max(stats.total_tokens, 1)
    p_f = folio_tokens / max(stats.total_tokens, 1)
    if p_s > 0 and p_f > 0 and p_sf > 0:
        pmi = math.log(p_sf / (p_s * p_f))
    else:
        pmi = 0.0

    return StemSpecificityScore(
        stem=stem,
        folio=folio,
        tf=tf,
        cf=cf,
        df=df,
        tfidf=round(tfidf, 4),
        specificity_ratio=round(spec_ratio, 4),
        exclusivity=round(exclusivity, 4),
        pmi=round(pmi, 4),
    )


def extract_folio_specific_stem(
    page: 'VoynichPage',
    stats: CorpusStemStats,
    metric: str = 'tfidf',
    min_tf: int = 2,
) -> Tuple[Optional[str], List[str], Optional[Tuple[int, int]], int,
           Dict[str, float], Optional[StemSpecificityScore]]:
    """
    Find the most folio-specific stem using TF-IDF or other specificity metric.

    Unlike extract_folio_dominant_stem (frequency-based), this selects stems
    that are disproportionately concentrated on this folio — the signature
    of a plant name that appears only where that plant is depicted.

    Parameters:
        page: VoynichPage to analyze
        stats: Pre-computed corpus-wide stem statistics
        metric: Which specificity metric to rank by.
                One of 'tfidf', 'specificity', 'exclusivity', 'pmi'.
        min_tf: Minimum term frequency on this folio to be considered.

    Returns:
        (stem, forms, paradigm_shape, stem_token_count,
         top_scores_dict, best_score_obj)
    """
    folio = page.folio
    stem_counts = stats.folio_stem_counts.get(folio, {})
    if not stem_counts:
        return None, [], None, 0, {}, None

    # Compute specificity for all stems on this folio
    scores: List[StemSpecificityScore] = []
    for stem, count in stem_counts.items():
        if count < min_tf:
            continue
        score = compute_stem_specificity(stem, folio, stats)
        scores.append(score)

    if not scores:
        # Fall back to any stem with tf >= 1
        for stem, count in stem_counts.items():
            score = compute_stem_specificity(stem, folio, stats)
            scores.append(score)

    if not scores:
        return None, [], None, 0, {}, None

    # Rank by chosen metric
    metric_key = {
        'tfidf': lambda s: s.tfidf,
        'specificity': lambda s: s.specificity_ratio,
        'exclusivity': lambda s: s.exclusivity,
        'pmi': lambda s: s.pmi,
    }.get(metric, lambda s: s.tfidf)

    scores.sort(key=metric_key, reverse=True)
    best = scores[0]

    # Build forms and paradigm shape for the winning stem
    tokens = page.all_tokens
    decomps = [decompose_token_morphemes(t) for t in tokens]
    stem_decomps = [d for d in decomps if d.stem == best.stem]
    forms = sorted(set(d.token for d in stem_decomps))
    prefixes = set(d.prefix for d in stem_decomps if d.prefix)
    suffixes = set(d.suffix for d in stem_decomps if d.suffix)
    paradigm_shape = (len(prefixes), len(suffixes))

    # Build top scores dict for reporting
    top_dict: Dict[str, float] = {}
    for s in scores[:10]:
        top_dict[s.stem] = getattr(s, metric.replace('specificity', 'specificity_ratio'))

    return (best.stem, forms, paradigm_shape, best.tf,
            top_dict, best)


# ---------------------------------------------------------------------------
# Build folio identification sets
# ---------------------------------------------------------------------------

def build_folio_identification_sets(
    concordance: Dict[str, List[Dict]],
    medieval_names: Dict[str, Dict],
    corpus: VoynichCorpus,
    stem_stats: Optional[CorpusStemStats] = None,
    use_tfidf: bool = False,
) -> List[FolioIdentificationSet]:
    """
    Combine concordance + medieval names + corpus analysis into per-folio sets.

    Only includes folios that exist in herbal_a section (f1-f56, Language A).

    If use_tfidf=True, also computes specificity-based stems and replaces
    the dominant_stem with the most folio-specific stem (Phase 6.1 Fix A).
    """
    # Get herbal_a page set
    herbal_pages = {
        p.folio: p
        for p in corpus.pages.values()
        if p.section == 'herbal_a'
    }

    # Build corpus stem stats if using TF-IDF and not provided
    if use_tfidf and stem_stats is None:
        stem_stats = build_corpus_stem_stats(corpus, section='herbal_a')

    results: List[FolioIdentificationSet] = []

    for folio, raw_ids in sorted(concordance.items()):
        # Check if this folio exists in herbal_a
        page = herbal_pages.get(folio)
        if page is None:
            continue

        # Build PlantIdentification objects
        plant_ids: List[PlantIdentification] = []
        for raw in raw_ids:
            linnaean = raw['linnaean_name']
            med_info = medieval_names.get(linnaean, {})

            medieval_name = med_info.get('medieval_name')
            medieval_stem = med_info.get('medieval_stem')
            declension = med_info.get('declension')
            alt_names = med_info.get('alternate_names', [])

            # Compute alternate stems from alternate names
            alt_stems = []
            for alt in alt_names:
                if alt:
                    alt_decl = infer_declension(alt)
                    alt_stem = extract_latin_stem(alt, alt_decl)
                    alt_stems.append(alt_stem)

            pid = PlantIdentification(
                folio=folio,
                linnaean_name=linnaean,
                common_name=raw['common_name'],
                source=raw['source'],
                medieval_name=medieval_name,
                medieval_stem=medieval_stem,
                declension=declension,
                alternate_stems=alt_stems,
            )
            plant_ids.append(pid)

        # Classify tier
        tier, consensus_genus = classify_identification_tier(raw_ids)

        # Extract dominant stem (frequency-based — original method)
        page_token_counts = Counter(page.all_tokens)
        dom_stem, dom_forms, dom_shape, dom_count, all_stems = \
            extract_folio_dominant_stem(page, page_token_counts)

        n_sources = len(set(pid.source for pid in plant_ids))

        # Phase 6.1: TF-IDF specificity-based stem extraction
        spec_stem = None
        spec_forms: List[str] = []
        spec_shape = None
        spec_scores = None
        stem_changed = False

        if use_tfidf and stem_stats is not None:
            (spec_stem, spec_forms, spec_shape, spec_count,
             spec_top, spec_best) = extract_folio_specific_stem(
                page, stem_stats, metric='tfidf', min_tf=2,
            )
            if spec_stem is not None:
                spec_scores = spec_top
                stem_changed = spec_stem != dom_stem
                # Replace dominant stem with specificity stem
                dom_stem = spec_stem
                dom_forms = spec_forms
                dom_shape = spec_shape
                dom_count = spec_count
                # Update all_stems to include specificity scores
                all_stems = {s: c for s, c in
                             sorted(stem_stats.folio_stem_counts.get(
                                 folio, {}).items(),
                                    key=lambda x: x[1], reverse=True)[:10]}

        fset = FolioIdentificationSet(
            folio=folio,
            identifications=plant_ids,
            n_sources=n_sources,
            consensus_genus=consensus_genus,
            tier=tier,
            dominant_stem=dom_stem,
            dominant_stem_forms=dom_forms,
            dominant_stem_paradigm_shape=dom_shape,
            dominant_stem_token_count=dom_count,
            all_stems=dict(sorted(all_stems.items(),
                                  key=lambda x: x[1], reverse=True)[:10]),
            token_count=len(page.all_tokens),
            specificity_stem=spec_stem,
            specificity_stem_forms=spec_forms,
            specificity_stem_paradigm_shape=spec_shape,
            specificity_stem_scores=spec_scores,
            stem_changed=stem_changed,
        )
        results.append(fset)

    return results


# ---------------------------------------------------------------------------
# Gate check
# ---------------------------------------------------------------------------

def _check_gate(
    name: str, value: float, threshold: float, direction: str = 'greater',
) -> Tuple[bool, str]:
    """Check a single gate condition."""
    if direction == 'greater':
        passed = value > threshold
        op = '>'
    else:
        passed = value < threshold
        op = '<'
    status = 'PASSED' if passed else 'FAILED'
    return passed, f"  Gate [{name}]: {value:.4f} {op} {threshold} -> {status}"


# ---------------------------------------------------------------------------
# JSON serialization helper
# ---------------------------------------------------------------------------

def _convert(obj):
    """Convert numpy/special types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _convert(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_convert(v) for v in obj]
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _plant_id_to_dict(pid: PlantIdentification) -> Dict:
    """Convert PlantIdentification to serializable dict."""
    return {
        'folio': pid.folio,
        'linnaean_name': pid.linnaean_name,
        'common_name': pid.common_name,
        'source': pid.source,
        'medieval_name': pid.medieval_name,
        'medieval_stem': pid.medieval_stem,
        'declension': pid.declension,
        'alternate_stems': pid.alternate_stems,
    }


def _folio_set_to_dict(fset: FolioIdentificationSet) -> Dict:
    """Convert FolioIdentificationSet to serializable dict."""
    d = {
        'folio': fset.folio,
        'identifications': [_plant_id_to_dict(p) for p in fset.identifications],
        'n_sources': fset.n_sources,
        'consensus_genus': fset.consensus_genus,
        'tier': fset.tier,
        'dominant_stem': fset.dominant_stem,
        'dominant_stem_forms': fset.dominant_stem_forms,
        'dominant_stem_paradigm_shape': list(fset.dominant_stem_paradigm_shape)
            if fset.dominant_stem_paradigm_shape else None,
        'dominant_stem_token_count': fset.dominant_stem_token_count,
        'all_stems': fset.all_stems,
        'token_count': fset.token_count,
    }
    # Phase 6.1 fields
    if fset.specificity_stem is not None:
        d['specificity_stem'] = fset.specificity_stem
        d['specificity_stem_forms'] = fset.specificity_stem_forms
        d['specificity_stem_paradigm_shape'] = (
            list(fset.specificity_stem_paradigm_shape)
            if fset.specificity_stem_paradigm_shape else None
        )
        d['specificity_stem_scores'] = fset.specificity_stem_scores
        d['stem_changed'] = fset.stem_changed
    return d


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_illustration_constrained(use_tfidf: bool = False) -> Dict:
    """
    Run Phase 6.0: Illustration-Constrained Setup.

    1. Load corpus
    2. Parse concordance CSV
    3. Load medieval name mappings
    4. Build folio identification sets
    5. Report tier distribution
    6. Gate: >= 8 Tier 1+2 folios with resolved medieval names
    7. Save results/illustration_constrained.json

    If use_tfidf=True, replaces frequency-based dominant stems with
    TF-IDF specificity-based stems (Phase 6.1 Fix A) and reports
    the diagnostic comparison.
    """
    print("=" * 70)
    if use_tfidf:
        print("Phase 6.1: Illustration-Constrained Setup (TF-IDF stems)")
    else:
        print("Phase 6.0: Illustration-Constrained Setup")
    print("=" * 70)

    # 1. Load corpus
    print("\n  Loading corpus...")
    corpus = load_corpus(verbose=False)
    herbal_a_pages = [
        p for p in corpus.pages.values() if p.section == 'herbal_a'
    ]
    print(f"    Herbal A pages: {len(herbal_a_pages)}")

    # 2. Parse concordance
    print("\n  6.0a: Parsing concordance CSV")
    concordance = parse_concordance()
    n_entries = sum(len(ids) for ids in concordance.values())
    n_folios = len(concordance)
    n_unique_plants = len(set(
        ident['linnaean_name']
        for ids in concordance.values()
        for ident in ids
    ))
    print(f"    Concordance entries: {n_entries}")
    print(f"    Unique folios: {n_folios}")
    print(f"    Unique plant species: {n_unique_plants}")

    # 3. Load medieval name mappings
    print("\n    Loading medieval Latin name mappings...")
    medieval_names = load_medieval_names()
    n_resolved = sum(1 for v in medieval_names.values()
                     if v.get('medieval_name') is not None)
    n_unresolvable = sum(1 for v in medieval_names.values()
                         if v.get('medieval_name') is None)
    print(f"    Resolved medieval names: {n_resolved}")
    print(f"    Unresolvable (New World etc.): {n_unresolvable}")

    # Build corpus stem stats if using TF-IDF
    stem_stats = None
    if use_tfidf:
        print("\n  6.1a: Computing corpus-wide stem statistics...")
        stem_stats = build_corpus_stem_stats(corpus, section='herbal_a')
        print(f"    Unique stems in corpus: {len(stem_stats.corpus_freq)}")
        print(f"    Total folios: {stem_stats.n_folios}")
        print(f"    Total tokens: {stem_stats.total_tokens}")

    # 4. Build folio identification sets
    print("\n  6.0b: Building folio identification sets")
    folio_sets = build_folio_identification_sets(
        concordance, medieval_names, corpus,
        stem_stats=stem_stats, use_tfidf=use_tfidf,
    )
    print(f"    Herbal A folios with identifications: {len(folio_sets)}")

    # 5. Tier distribution
    print("\n  6.0c: Tier classification")
    tier_dist: Dict[int, int] = Counter(fs.tier for fs in folio_sets)
    for tier in sorted(tier_dist):
        tier_labels = {1: 'Consensus', 2: 'Single-high', 3: 'Contested'}
        label = tier_labels.get(tier, f'Tier {tier}')
        print(f"    Tier {tier} ({label}): {tier_dist[tier]} folios")

    # Count Tier 1+2 folios with resolved medieval names
    tier12_resolved = sum(
        1 for fs in folio_sets
        if fs.tier <= 2
        and any(p.medieval_name for p in fs.identifications)
    )
    print(f"\n    Tier 1+2 with medieval names: {tier12_resolved}")

    # Show some examples
    print("\n    Example folio identification sets:")
    for fs in folio_sets[:8]:
        med_names = [p.medieval_name for p in fs.identifications
                     if p.medieval_name]
        med_str = ', '.join(med_names[:2]) if med_names else '(no medieval name)'
        changed = ' [CHANGED]' if fs.stem_changed else ''
        print(f"      {fs.folio} [T{fs.tier}]: "
              f"stem='{fs.dominant_stem}' ({fs.dominant_stem_token_count}x), "
              f"IDs={med_str}{changed}")

    # Phase 6.1: TF-IDF diagnostic comparison
    tfidf_diagnostic = None
    if use_tfidf:
        print("\n  6.1b: TF-IDF Diagnostic Comparison")
        print("  " + "-" * 66)

        # Also build frequency-based sets for comparison
        freq_sets = build_folio_identification_sets(
            concordance, medieval_names, corpus,
            stem_stats=None, use_tfidf=False,
        )
        freq_index = {fs.folio: fs for fs in freq_sets}

        n_changed = sum(1 for fs in folio_sets if fs.stem_changed)
        n_daiin_old = sum(1 for fs in freq_sets
                          if fs.dominant_stem == 'daiin')
        n_daiin_new = sum(1 for fs in folio_sets
                          if fs.dominant_stem == 'daiin')

        # Compute mean specificity ratios
        old_spec_ratios = []
        new_spec_ratios = []
        for fs in folio_sets:
            freq_fs = freq_index.get(fs.folio)
            if freq_fs and stem_stats:
                old_score = compute_stem_specificity(
                    freq_fs.dominant_stem or '', fs.folio, stem_stats)
                old_spec_ratios.append(old_score.specificity_ratio)
                new_score = compute_stem_specificity(
                    fs.dominant_stem or '', fs.folio, stem_stats)
                new_spec_ratios.append(new_score.specificity_ratio)

        mean_old = np.mean(old_spec_ratios) if old_spec_ratios else 0.0
        mean_new = np.mean(new_spec_ratios) if new_spec_ratios else 0.0

        print(f"    Folios that changed dominant stem: {n_changed}/{len(folio_sets)}")
        print(f"    'daiin' as dominant (old/freq): {n_daiin_old}")
        print(f"    'daiin' as dominant (new/tfidf): {n_daiin_new}")
        print(f"    Mean specificity ratio (old): {mean_old:.4f}")
        print(f"    Mean specificity ratio (new): {mean_new:.4f}")

        # Show per-folio changes
        print(f"\n    Per-folio comparison:")
        print(f"    {'Folio':<8s} {'Old stem':<12s} {'New stem':<12s} "
              f"{'Old spec':<10s} {'New spec':<10s} {'Changed'}")
        print(f"    {'─'*8} {'─'*12} {'─'*12} {'─'*10} {'─'*10} {'─'*7}")
        for fs in folio_sets:
            freq_fs = freq_index.get(fs.folio)
            if freq_fs is None:
                continue
            old_stem = freq_fs.dominant_stem or '—'
            new_stem = fs.dominant_stem or '—'
            old_s = compute_stem_specificity(
                freq_fs.dominant_stem or '', fs.folio, stem_stats
            ) if stem_stats else None
            new_s = compute_stem_specificity(
                fs.dominant_stem or '', fs.folio, stem_stats
            ) if stem_stats else None
            old_r = f"{old_s.specificity_ratio:.4f}" if old_s else "—"
            new_r = f"{new_s.specificity_ratio:.4f}" if new_s else "—"
            ch = "YES" if fs.stem_changed else ""
            print(f"    {fs.folio:<8s} {old_stem:<12s} {new_stem:<12s} "
                  f"{old_r:<10s} {new_r:<10s} {ch}")

        tfidf_diagnostic = {
            'n_changed': n_changed,
            'n_total': len(folio_sets),
            'daiin_count_old': n_daiin_old,
            'daiin_count_new': n_daiin_new,
            'mean_specificity_old': round(float(mean_old), 4),
            'mean_specificity_new': round(float(mean_new), 4),
        }

    # Coverage
    herbal_a_count = len(herbal_a_pages)
    coverage = len(folio_sets) / herbal_a_count if herbal_a_count > 0 else 0.0
    print(f"\n    Coverage: {len(folio_sets)}/{herbal_a_count} "
          f"herbal A folios ({coverage:.1%})")

    # 6. Gate
    gate_ok, gate_msg = _check_gate(
        'tier12_medieval_names', float(tier12_resolved), 7.0, 'greater',
    )
    print(f"\n{gate_msg}")
    if gate_ok:
        verdict = 'sufficient_anchors'
    else:
        verdict = 'insufficient_anchors'
    print(f"  Verdict: {verdict}")

    # Build result
    result = IllustrationConstrainedResult(
        n_concordance_entries=n_entries,
        n_folios_with_ids=len(folio_sets),
        n_unique_plants=n_unique_plants,
        n_medieval_names_resolved=n_resolved,
        n_unresolvable=n_unresolvable,
        tier_distribution=dict(tier_dist),
        folios=[_folio_set_to_dict(fs) for fs in folio_sets],
        herbal_a_folios=herbal_a_count,
        herbal_a_with_ids=len(folio_sets),
        coverage=round(coverage, 4),
        gate_passed=gate_ok,
        verdict=verdict,
    )

    # Add TF-IDF diagnostic to result
    result_dict = _convert(asdict(result))
    if tfidf_diagnostic:
        result_dict['tfidf_diagnostic'] = tfidf_diagnostic

    # Save
    out_path = os.path.join(_results_dir(), 'illustration_constrained.json')
    with open(out_path, 'w') as f:
        json.dump(result_dict, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")

    return result_dict
