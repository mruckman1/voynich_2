"""
Phase 33.10 – Long Crib Targets
==================================
Identify the 3-5 folios with the longest, most distinctive plant names and the
most confident botanical identifications, to use as cribs for the long crib CSP
attack.

For each target folio, extract label candidates ranked by TF-IDF specificity and
filter by syllabic length compatibility with the plant name.

Dependency chain:
    data/reference/voynich_plant/medieval_latin_names.json  (optional)
    results/consensus_plants.json   (Phase 31.1, optional)
    results/modifier_integrate.json (Phase 16 modifiers)
        -> results/long_crib_targets.json  (this step)
"""

import json
import math
import os
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from voynich.core._paths import results_dir as _results_dir, data_dir as _data_dir
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    load_corpus,
    token_to_triples,
    tokenize_eva_chars,
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
# Hardcoded plant identifications (primary source)
# ---------------------------------------------------------------------------

PLANT_IDENTIFICATIONS = [
    {
        'folio': 'f11r',
        'genus': 'Rosmarinus officinalis',
        'medieval_latin': 'rosmarinus',
        'syllables': ['ros', 'ma', 'ri', 'nus'],
        'confidence': 'B',  # single source
        'sources': ['General Botanical List'],
    },
    {
        'folio': 'f33r',
        'genus': 'Papaver somniferum',
        'medieval_latin': 'papaver',
        'syllables': ['pa', 'pa', 'ver'],  # Note repeated syllable
        'confidence': 'B',
        'sources': ['Bax'],
    },
    {
        'folio': 'f9v',
        'genus': 'Viola',
        'medieval_latin': 'viola',
        'syllables': ['vi', 'o', 'la'],
        'confidence': 'A',  # 3 independent sources
        'sources': ['Tucker/Talbert', 'Sherwood', 'General'],
    },
    {
        'folio': 'f37v',
        'genus': 'Anagallis arvensis',
        'medieval_latin': 'anagallis',
        'syllables': ['a', 'na', 'gal', 'lis'],
        'confidence': 'B',
        'sources': ['Sherwood'],
    },
    {
        'folio': 'f47v',
        'genus': 'Pulmonaria',
        'medieval_latin': 'pulmonaria',
        'syllables': ['pul', 'mo', 'na', 'ri', 'a'],
        'confidence': 'B',
        'sources': ['Sherwood'],
    },
    {
        'folio': 'f2r',
        'genus': 'Centaurea cyanus',
        'medieval_latin': 'centaurea',
        'syllables': ['cen', 'ta', 'u', 're', 'a'],
        'confidence': 'B',
        'sources': ['Tucker/Talbert'],
    },
    {
        'folio': 'f4v',
        'genus': 'Helleborus',
        'medieval_latin': 'helleborus',
        'syllables': ['hel', 'le', 'bo', 'rus'],
        'confidence': 'B',
        'sources': ['General'],
    },
    {
        'folio': 'f6r',
        'genus': 'Calendula',
        'medieval_latin': 'calendula',
        'syllables': ['ca', 'len', 'du', 'la'],
        'confidence': 'B',
        'sources': ['Phase 24 analysis'],
    },
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PlantCrib:
    folio: str
    genus: str
    medieval_latin: str
    syllables: List[str]
    n_syllables: int
    confidence: str
    sources: List[str]
    crib_value: float


@dataclass
class LabelCandidate:
    eva_token: str
    eva_chars: List[str]
    syllabic_chars: List[str]  # after removing modifiers
    n_syllabic: int
    tfidf_score: float
    folio_count: int
    corpus_count: int
    compatible: bool  # n_syllabic >= n_syllables of plant name


@dataclass
class FolioTarget:
    folio: str
    plant: Dict  # PlantCrib as dict
    label_candidates: List[Dict]  # LabelCandidate as dict
    n_compatible: int


@dataclass
class LongCribTargetsResult:
    n_plants: int
    ranked_plants: List[Dict]  # PlantCrib list
    n_target_folios: int
    folio_targets: List[Dict]  # FolioTarget list
    total_compatible_candidates: int
    verdict: str  # 'TARGETS_FOUND' or 'NO_TARGETS'
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Plant identification loading
# ---------------------------------------------------------------------------

_CONFIDENCE_WEIGHT = {'A': 3.0, 'B': 2.0, 'C': 1.0}


def _load_hardcoded_plants() -> List[PlantCrib]:
    """Build PlantCrib entries from the hardcoded identification list."""
    cribs = []
    for entry in PLANT_IDENTIFICATIONS:
        syllables = entry['syllables']
        conf = entry.get('confidence', 'C')
        cribs.append(PlantCrib(
            folio=entry['folio'],
            genus=entry['genus'],
            medieval_latin=entry['medieval_latin'],
            syllables=syllables,
            n_syllables=len(syllables),
            confidence=conf,
            sources=entry.get('sources', []),
            crib_value=len(syllables) * _CONFIDENCE_WEIGHT.get(conf, 1.0),
        ))
    return cribs


def _load_consensus_plants(rd: str) -> List[PlantCrib]:
    """Load additional plant identifications from consensus_plants.json if available."""
    path = os.path.join(rd, 'consensus_plants.json')
    if not os.path.exists(path):
        return []

    with open(path) as f:
        data = json.load(f)

    cribs = []
    # Process all folios that have medieval name data
    all_folios = data.get('all_folios', [])
    tier_a = data.get('tier_a_folios', [])
    tier_b = data.get('tier_b_folios', [])

    # Combine tier_a and tier_b (these have full medieval name info)
    detailed_folios = tier_a + tier_b

    for entry in detailed_folios:
        folio = entry.get('folio', '')
        consensus = entry.get('consensus', {})
        genus = consensus.get('genus', '')
        tier = consensus.get('tier', 'C')
        sources = consensus.get('sources', [])
        med_names = entry.get('medieval_names', [])

        if not med_names or not genus:
            continue

        # Use the first medieval name entry
        mn = med_names[0]
        medieval_name = mn.get('medieval_name', '')
        syllabified = mn.get('syllabified', [])

        if not medieval_name or not syllabified:
            continue

        conf = tier if tier in _CONFIDENCE_WEIGHT else 'C'
        cribs.append(PlantCrib(
            folio=folio,
            genus=genus,
            medieval_latin=medieval_name,
            syllables=syllabified,
            n_syllables=len(syllabified),
            confidence=conf,
            sources=sources,
            crib_value=len(syllabified) * _CONFIDENCE_WEIGHT.get(conf, 1.0),
        ))

    return cribs


def _load_medieval_names_file() -> Dict[str, Dict]:
    """Load medieval_latin_names.json for syllable info if available."""
    try:
        path = os.path.join(str(_data_dir('reference')), 'voynich_plant',
                            'medieval_latin_names.json')
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _merge_plant_identifications(rd: str) -> List[PlantCrib]:
    """Merge hardcoded + consensus plants, deduplicating by folio.

    Hardcoded entries take priority.  Consensus entries for folios not
    already in the hardcoded list are appended.
    """
    hardcoded = _load_hardcoded_plants()
    consensus = _load_consensus_plants(rd)

    seen_folios: Set[str] = set()
    merged: List[PlantCrib] = []

    # Hardcoded first (primary)
    for crib in hardcoded:
        if crib.folio not in seen_folios:
            seen_folios.add(crib.folio)
            merged.append(crib)

    # Consensus supplements
    for crib in consensus:
        if crib.folio not in seen_folios:
            seen_folios.add(crib.folio)
            merged.append(crib)

    return merged


# ---------------------------------------------------------------------------
# TF-IDF label candidate extraction
# ---------------------------------------------------------------------------

def _compute_tfidf(
    target_folio: str,
    corpus,
    modifier_chars: Set[str],
    n_syllables: int,
    top_k: int = 5,
) -> List[LabelCandidate]:
    """Extract top-k label candidates from a folio ranked by TF-IDF.

    Parameters
    ----------
    target_folio : str
        Folio identifier (e.g. 'f9v').
    corpus : VoynichCorpus
        Loaded corpus.
    modifier_chars : set
        EVA chars classified as modifiers.
    n_syllables : int
        Number of syllables in the plant name.
    top_k : int
        Number of top candidates to return.
    """
    # Get tokens for target folio
    page = corpus.pages.get(target_folio)
    if page is None:
        return []

    folio_tokens = page.all_tokens
    if not folio_tokens:
        return []

    # Count tokens on this folio
    folio_counter = Counter(folio_tokens)
    total_folio = len(folio_tokens)

    # Count which folios contain each token (for IDF)
    n_folios = len(corpus.pages)
    token_folio_count: Dict[str, int] = defaultdict(int)
    token_corpus_count: Dict[str, int] = Counter()

    for fid, pg in corpus.pages.items():
        pg_tokens = pg.all_tokens
        token_corpus_count.update(pg_tokens)
        seen_on_page: Set[str] = set(pg_tokens)
        for tok in seen_on_page:
            token_folio_count[tok] += 1

    # Compute TF-IDF for each unique token on this folio
    tfidf_scores: Dict[str, float] = {}
    for tok, count in folio_counter.items():
        tf = count / total_folio
        df = token_folio_count.get(tok, 1)
        idf = math.log(n_folios / df) if df > 0 else 0.0
        tfidf_scores[tok] = tf * idf

    # Rank by TF-IDF
    ranked = sorted(tfidf_scores.items(), key=lambda x: -x[1])

    # Build LabelCandidate entries for top-k
    candidates = []
    for tok, score in ranked[:top_k]:
        eva_chars = tokenize_eva_chars(tok)
        syllabic = [ch for ch in eva_chars if ch not in modifier_chars]
        n_syl = len(syllabic)
        compatible = n_syl >= n_syllables

        candidates.append(LabelCandidate(
            eva_token=tok,
            eva_chars=eva_chars,
            syllabic_chars=syllabic,
            n_syllabic=n_syl,
            tfidf_score=round(score, 6),
            folio_count=folio_counter.get(tok, 0),
            corpus_count=token_corpus_count.get(tok, 0),
            compatible=compatible,
        ))

    return candidates


# ---------------------------------------------------------------------------
# Load modifier chars
# ---------------------------------------------------------------------------

def _load_modifier_chars(rd: str) -> Set[str]:
    """Load modifier character set from modifier_integrate.json."""
    path = os.path.join(rd, 'modifier_integrate.json')
    if not os.path.exists(path):
        # Fallback: hardcoded from Phase 16 results
        return {'h', 'iin', 'b', 'ckh', 'i', 'iiin', 'u', 'aiin',
                'al', 'ar', 'dy', 'ey', 'm', 'n', 'or'}
    with open(path) as f:
        data = json.load(f)
    return set(data.get('modifier_chars', []))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_long_crib_targets() -> None:
    """Step 33.10: Identify target folios for long crib CSP attack."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE 33.10: Long Crib Targets")
    print("=" * 70)

    rd = str(_results_dir())

    # ── 1. Load plant identifications ──
    print("\n  1. Loading plant identifications ...")
    all_plants = _merge_plant_identifications(rd)
    n_hardcoded = len(_load_hardcoded_plants())
    n_consensus_extra = len(all_plants) - n_hardcoded
    print(f"     {n_hardcoded} hardcoded + {n_consensus_extra} from consensus "
          f"= {len(all_plants)} total plants")

    # ── 2. Rank by crib_value ──
    print("\n  2. Ranking by crib_value (n_syllables * confidence_weight) ...")
    ranked = sorted(all_plants, key=lambda p: -p.crib_value)
    for i, p in enumerate(ranked[:10]):
        print(f"     {i+1}. {p.folio} {p.medieval_latin} "
              f"({p.n_syllables} syl, conf={p.confidence}, "
              f"value={p.crib_value:.1f})")

    # ── 3. Load corpus and modifiers ──
    print("\n  3. Loading corpus and modifier set ...")
    corpus = load_corpus(verbose=False)
    modifier_chars = _load_modifier_chars(rd)
    n_folios = len(corpus.pages)
    print(f"     Corpus: {n_folios} folios")
    print(f"     Modifier chars ({len(modifier_chars)}): "
          f"{sorted(modifier_chars)}")

    # ── 4. For each target, extract label candidates ──
    print("\n  4. Extracting label candidates by TF-IDF ...")

    folio_targets: List[FolioTarget] = []
    total_compatible = 0

    for plant in ranked:
        folio = plant.folio

        # Check folio exists in corpus
        if folio not in corpus.pages:
            print(f"     {folio}: NOT FOUND in corpus, skipping")
            continue

        candidates = _compute_tfidf(
            target_folio=folio,
            corpus=corpus,
            modifier_chars=modifier_chars,
            n_syllables=plant.n_syllables,
            top_k=5,
        )

        n_compat = sum(1 for c in candidates if c.compatible)
        total_compatible += n_compat

        target = FolioTarget(
            folio=folio,
            plant=_convert(asdict(plant)),
            label_candidates=[_convert(asdict(c)) for c in candidates],
            n_compatible=n_compat,
        )
        folio_targets.append(target)

        compat_str = f"{n_compat}/{len(candidates)} compatible"
        top_tok = candidates[0].eva_token if candidates else '?'
        top_score = candidates[0].tfidf_score if candidates else 0.0
        print(f"     {folio} ({plant.medieval_latin}, {plant.n_syllables} syl): "
              f"top={top_tok} (tfidf={top_score:.4f}), {compat_str}")

    # ── 5. Determine verdict ──
    n_with_compatible = sum(1 for ft in folio_targets if ft.n_compatible > 0)
    verdict = 'TARGETS_FOUND' if n_with_compatible >= 3 else 'NO_TARGETS'

    print(f"\n  5. Verdict: {verdict}")
    print(f"     {n_with_compatible} folios have compatible label candidates")
    print(f"     {total_compatible} total compatible candidates across all folios")

    # ── 6. Save results ──
    result = LongCribTargetsResult(
        n_plants=len(all_plants),
        ranked_plants=[_convert(asdict(p)) for p in ranked],
        n_target_folios=len(folio_targets),
        folio_targets=[_convert(asdict(ft)) for ft in folio_targets],
        total_compatible_candidates=total_compatible,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'long_crib_targets.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2)

    elapsed = time.time() - t0
    print(f"\n  Result: {verdict} — {len(all_plants)} plants ranked, "
          f"{n_with_compatible} folios with compatible labels, "
          f"{total_compatible} total compatible candidates")
    print(f"  Saved -> {out_path}")
    print(f"  Completed in {elapsed:.1f}s")
