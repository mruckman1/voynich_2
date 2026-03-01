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
# Build folio identification sets
# ---------------------------------------------------------------------------

def build_folio_identification_sets(
    concordance: Dict[str, List[Dict]],
    medieval_names: Dict[str, Dict],
    corpus: VoynichCorpus,
) -> List[FolioIdentificationSet]:
    """
    Combine concordance + medieval names + corpus analysis into per-folio sets.

    Only includes folios that exist in herbal_a section (f1-f56, Language A).
    """
    # Get herbal_a page set
    herbal_pages = {
        p.folio: p
        for p in corpus.pages.values()
        if p.section == 'herbal_a'
    }

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

        # Extract dominant stem (using per-page token counts)
        page_token_counts = Counter(page.all_tokens)
        dom_stem, dom_forms, dom_shape, dom_count, all_stems = \
            extract_folio_dominant_stem(page, page_token_counts)

        n_sources = len(set(pid.source for pid in plant_ids))

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
    return {
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


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_illustration_constrained() -> Dict:
    """
    Run Phase 6.0: Illustration-Constrained Setup.

    1. Load corpus
    2. Parse concordance CSV
    3. Load medieval name mappings
    4. Build folio identification sets
    5. Report tier distribution
    6. Gate: >= 8 Tier 1+2 folios with resolved medieval names
    7. Save results/illustration_constrained.json
    """
    print("=" * 70)
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

    # 4. Build folio identification sets
    print("\n  6.0b: Building folio identification sets")
    folio_sets = build_folio_identification_sets(
        concordance, medieval_names, corpus,
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
        print(f"      {fs.folio} [T{fs.tier}]: "
              f"stem='{fs.dominant_stem}' ({fs.dominant_stem_token_count}x), "
              f"IDs={med_str}")

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

    # Save
    out_path = os.path.join(_results_dir(), 'illustration_constrained.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")

    return _convert(asdict(result))
