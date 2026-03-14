"""
Phase 52 Track B: Word Identification Validation
=================================================
Validate the word catalog against four independent tests:
1. Null shuffling (assignment permutation)
2. Botanical cross-reference
3. Morphological paradigms
4. Signal adjacency enrichment

Dependency chain:
    word_catalog.json          (Track A)
    signal_bigrams.json        (Step 29.1)
    combined_refine.json       (Step 15)
    modifier_integrate.json    (Step 16)
    triple_tiers.json          (Step 44)
    consensus_plants.json      (Step 31)
        -> word_validation.json (this step)
"""

from __future__ import annotations

import json
import math
import os
import random
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
    _search_dict,
)
from voynich.phases.suffix_calibration import SIGNAL_WORDS_SET


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
class Validation:
    name: str
    description: str
    passed: bool
    value: float
    threshold: float


@dataclass
class WordValidationResult:
    # Null test
    null_test: Dict
    # Botanical
    n_botanical_matches: int
    botanical_matches: List[Dict]
    # Paradigms
    n_paradigms: int
    paradigms: List[Dict]
    # Signal adjacency
    signal_adjacency_z: float
    signal_adjacency_enrichment: float
    # Validation battery
    validations: List[Dict]
    n_passed: int
    n_total: int
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Test 1: Null test — shuffled assignments
# ---------------------------------------------------------------------------

def _null_bridge_search(
    token_evas: List[str],
    token_decoded: List[str],
    token_folios: List[str],
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    modifier_chars: Set[str],
    confirmed_triples: Set[str],
    pharma_dict: Set[str],
    rng: random.Random,
    sample_rate: float = 0.5,
) -> int:
    """Run a single null bridge search with shuffled assignment.

    Returns count of (EVA_type, matched_word) pairs with n_folios >= 2.
    Uses distance-1 only and samples signal anchors for speed.
    """
    n_tokens = len(token_evas)

    # Shuffle assignment values (swap syllables between triples)
    keys = list(assignment.keys())
    values = list(assignment.values())
    rng.shuffle(values)
    null_assignment = dict(zip(keys, values))

    # Find signal positions, sample a fraction
    signal_positions = [i for i in range(n_tokens)
                        if token_decoded[i] in SIGNAL_WORDS_SET]
    n_sample = max(1, int(len(signal_positions) * sample_rate))
    sampled = set(rng.sample(signal_positions, n_sample))

    seen: Set[int] = set()
    pair_folios: Dict[Tuple[str, str], Set[str]] = defaultdict(set)

    for sig_idx in sampled:
        anchor_word = token_decoded[sig_idx]
        for offset, position in [(-1, 'before'), (1, 'after')]:
            nbr_idx = sig_idx + offset
            if nbr_idx < 0 or nbr_idx >= n_tokens:
                continue
            if token_decoded[nbr_idx] in SIGNAL_WORDS_SET:
                continue
            if nbr_idx in seen:
                continue
            seen.add(nbr_idx)

            dark_eva = token_evas[nbr_idx]
            pattern, details = _build_partial_decode(
                dark_eva, null_assignment, eva_to_triple,
                modifier_chars, confirmed_triples,
            )

            n_conf = sum(1 for _, _, _, c in details if c)
            n_free = sum(1 for _, _, _, c in details if not c)
            if n_conf < 1 or n_free < 1 or n_free > 3:
                continue

            matches = _search_dict(pattern, pharma_dict)
            for mword in matches:
                pair_folios[(dark_eva, mword)].add(token_folios[nbr_idx])

    # Count pairs with >= 2 folios, scaled by sample rate
    n_multi_folio = sum(1 for folios in pair_folios.values()
                        if len(folios) >= 2)
    return int(n_multi_folio / sample_rate)


# ---------------------------------------------------------------------------
# Test 2: Botanical cross-reference
# ---------------------------------------------------------------------------

def _botanical_cross_reference(
    catalog: List[Dict],
    consensus_plants: Dict,
) -> List[Dict]:
    """Check if catalog words match plant names on corresponding folios."""
    matches = []

    # Build folio → plant name mapping from consensus_plants
    folio_plants: Dict[str, List[str]] = {}
    for tier_key in ['tier_a_folios', 'tier_b_folios']:
        for entry in consensus_plants.get(tier_key, []):
            folio = entry.get('folio', '')
            names = []
            for name_field in ['medieval_name', 'latin_name', 'genus']:
                val = entry.get(name_field, '')
                if val:
                    names.append(val.lower())
            if folio and names:
                folio_plants.setdefault(folio, []).extend(names)

    for wid in catalog:
        if wid.get('tier') not in ('T1', 'T2'):
            continue
        latin = wid['latin_word'].lower()
        for folio in wid.get('folios', []):
            plant_names = folio_plants.get(folio, [])
            for pname in plant_names:
                # Check substring match in either direction
                if latin in pname or pname in latin:
                    matches.append({
                        'eva_type': wid['eva_type'],
                        'latin_word': wid['latin_word'],
                        'folio': folio,
                        'plant_name': pname,
                        'match_type': 'substring',
                    })
                # Check stem overlap (first 4+ chars)
                elif len(latin) >= 4 and len(pname) >= 4:
                    if latin[:4] == pname[:4]:
                        matches.append({
                            'eva_type': wid['eva_type'],
                            'latin_word': wid['latin_word'],
                            'folio': folio,
                            'plant_name': pname,
                            'match_type': 'stem_overlap',
                        })

    return matches


# ---------------------------------------------------------------------------
# Test 3: Morphological paradigms
# ---------------------------------------------------------------------------

LATIN_ENDINGS = [
    'ione', 'onis', 'ibus', 'orum', 'arum',
    'em', 'is', 'um', 'us', 'ae', 'am', 'as',
    'es', 'ei', 'ia', 'ii',
    'i', 'o', 'e', 'a',
]


def _extract_stem(word: str) -> str:
    """Extract approximate Latin stem by stripping common endings."""
    for ending in LATIN_ENDINGS:
        if word.endswith(ending) and len(word) - len(ending) >= 4:
            return word[:-len(ending)]
    return word


def _find_paradigms(catalog: List[Dict]) -> List[Dict]:
    """Find morphological paradigms — different EVA types mapping to
    different inflections of the same Latin stem."""
    stems: Dict[str, List[Dict]] = defaultdict(list)

    for wid in catalog:
        if wid.get('tier') in ('T1', 'T2', 'T3'):
            stem = _extract_stem(wid['latin_word'])
            if len(stem) >= 4:
                stems[stem].append(wid)

    paradigms = []
    for stem, entries in stems.items():
        # Require at least 2 entries with DIFFERENT EVA types
        eva_types = set(e['eva_type'] for e in entries)
        latin_words = set(e['latin_word'] for e in entries)
        if len(eva_types) >= 2 and len(latin_words) >= 2:
            paradigms.append({
                'stem': stem,
                'n_forms': len(latin_words),
                'latin_words': sorted(latin_words),
                'eva_types': sorted(eva_types),
                'tiers': [e['tier'] for e in entries],
            })

    return sorted(paradigms, key=lambda p: p['n_forms'], reverse=True)


# ---------------------------------------------------------------------------
# Test 4: Signal adjacency enrichment
# ---------------------------------------------------------------------------

def _signal_adjacency_enrichment(
    catalog: List[Dict],
    token_evas: List[str],
    token_decoded: List[str],
) -> Tuple[float, float]:
    """Compute enrichment of catalog EVA types near signal words.

    Returns (z_score, enrichment_ratio).
    """
    n_tokens = len(token_evas)

    # Signal positions
    signal_pos = set()
    for i in range(n_tokens):
        if token_decoded[i] in SIGNAL_WORDS_SET:
            signal_pos.add(i)

    def near_signal(idx: int, max_dist: int = 2) -> bool:
        for d in range(1, max_dist + 1):
            if (idx - d) in signal_pos or (idx + d) in signal_pos:
                return True
        return False

    # Baseline: fraction of ALL non-signal tokens near a signal token
    n_non_signal = 0
    n_near_baseline = 0
    for i in range(n_tokens):
        if i not in signal_pos:
            n_non_signal += 1
            if near_signal(i):
                n_near_baseline += 1

    baseline_rate = n_near_baseline / max(n_non_signal, 1)

    # Catalog rate: fraction of catalog EVA type occurrences near signal
    catalog_eva = set()
    for wid in catalog:
        if wid.get('tier') in ('T1', 'T2'):
            catalog_eva.add(wid['eva_type'])

    n_cat = 0
    n_cat_near = 0
    for i in range(n_tokens):
        if token_evas[i] in catalog_eva and i not in signal_pos:
            n_cat += 1
            if near_signal(i):
                n_cat_near += 1

    cat_rate = n_cat_near / max(n_cat, 1)

    # Enrichment
    enrichment = cat_rate / max(baseline_rate, 1e-6)

    # Z-score (binomial approximation)
    if n_cat > 0 and baseline_rate > 0:
        expected = n_cat * baseline_rate
        std = math.sqrt(n_cat * baseline_rate * (1 - baseline_rate))
        z = (n_cat_near - expected) / max(std, 1e-6)
    else:
        z = 0.0

    return z, enrichment


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_word_validation() -> None:
    """Phase 52 Track B: Validate word identifications."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE 52 TRACK B: Word Identification Validation")
    print("=" * 70)

    rd = _results_dir()

    # ── Load catalog ─────────────────────────────────────────────────
    print("\n  B.1  Loading catalog...")
    catalog_data = _safe_load(os.path.join(rd, 'word_catalog.json'))
    if not catalog_data:
        print("  *** word_catalog.json not found — run Track A first ***")
        return

    catalog = catalog_data.get('single_token_ids', [])
    n_t1 = catalog_data.get('n_tier1', 0)
    n_t2 = catalog_data.get('n_tier2', 0)
    real_count = n_t1 + n_t2
    print(f"       {len(catalog)} catalog entries (T1={n_t1}, T2={n_t2})")

    # ── Load inputs for null test ────────────────────────────────────
    print("\n  B.2  Loading inputs for null test...")

    with open(os.path.join(rd, 'signal_bigrams.json')) as f:
        bigram_data = json.load(f)
    token_evas = bigram_data['token_evas']
    token_decoded = bigram_data['token_decoded']
    token_folios = bigram_data['token_folios']

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
    # Note: LANDSCAPE_CONFIRMED excluded to match Phase 51B behavior

    eva_to_triple = build_eva_to_triple_lookup()
    pharma_dict = _build_pharma_dict()

    # ── Test 1: Null test ────────────────────────────────────────────
    print("\n  B.3  Null test (50 shuffled assignments)...")

    rng = random.Random(42)
    null_counts: List[int] = []

    for trial in range(50):
        nc = _null_bridge_search(
            token_evas, token_decoded, token_folios,
            assignment, eva_to_triple, modifier_chars,
            confirmed_triples, pharma_dict, rng,
            sample_rate=0.5,
        )
        null_counts.append(nc)
        if (trial + 1) % 10 == 0:
            print(f"       Trial {trial + 1}/50 done")

    null_mean = sum(null_counts) / len(null_counts)
    null_std = math.sqrt(sum((x - null_mean) ** 2 for x in null_counts)
                         / len(null_counts)) if null_counts else 1.0
    selectivity = real_count / max(null_mean, 1e-6)
    z_score = (real_count - null_mean) / max(null_std, 1e-6)

    null_test = {
        'real_count': real_count,
        'null_mean': round(null_mean, 2),
        'null_std': round(null_std, 2),
        'selectivity': round(selectivity, 4),
        'z_score': round(z_score, 2),
        'n_iterations': 50,
    }

    print(f"       Real T1+T2: {real_count}")
    print(f"       Null mean: {null_mean:.1f} (std={null_std:.1f})")
    print(f"       Selectivity: {selectivity:.2f}×")
    print(f"       Z-score: {z_score:.2f}")

    # ── Test 2: Botanical cross-reference ────────────────────────────
    print("\n  B.4  Botanical cross-reference...")

    consensus_plants = _safe_load(os.path.join(rd, 'consensus_plants.json'))
    botanical_matches = _botanical_cross_reference(catalog, consensus_plants)
    print(f"       Botanical matches: {len(botanical_matches)}")
    for bm in botanical_matches[:5]:
        print(f"         {bm['eva_type']} → {bm['latin_word']} "
              f"on {bm['folio']} (plant: {bm['plant_name']})")

    # ── Test 3: Morphological paradigms ──────────────────────────────
    print("\n  B.5  Morphological paradigms...")

    paradigms = _find_paradigms(catalog)
    print(f"       Paradigms found: {len(paradigms)}")
    for p in paradigms[:5]:
        print(f"         stem={p['stem']}: {p['latin_words']}")

    # ── Test 4: Signal adjacency enrichment ──────────────────────────
    print("\n  B.6  Signal adjacency enrichment...")

    adj_z, adj_enrichment = _signal_adjacency_enrichment(
        catalog, token_evas, token_decoded,
    )
    print(f"       Enrichment: {adj_enrichment:.2f}×")
    print(f"       Z-score: {adj_z:.2f}")

    # ── Validation battery ───────────────────────────────────────────
    print("\n  B.7  Validation battery...")

    validations = [
        Validation('V1_catalog_size', 'T1+T2 identifications >= 20',
                    real_count >= 20, float(real_count), 20.0),
        Validation('V3_null_selectivity', 'Selectivity > 1.5x',
                    selectivity > 1.5, selectivity, 1.5),
        Validation('V4_paradigms', 'Morphological paradigms >= 3',
                    len(paradigms) >= 3, float(len(paradigms)), 3.0),
        Validation('V5_botanical', 'Botanical matches >= 1',
                    len(botanical_matches) >= 1,
                    float(len(botanical_matches)), 1.0),
    ]

    n_passed = sum(1 for v in validations if v.passed)
    for v in validations:
        status = 'PASS' if v.passed else 'FAIL'
        print(f"       {v.name}: {v.value:.1f} vs {v.threshold:.1f} → {status}")

    # ── Save ─────────────────────────────────────────────────────────
    runtime = round(time.time() - t0, 2)

    result = WordValidationResult(
        null_test=null_test,
        n_botanical_matches=len(botanical_matches),
        botanical_matches=botanical_matches[:20],
        n_paradigms=len(paradigms),
        paradigms=paradigms[:20],
        signal_adjacency_z=round(adj_z, 4),
        signal_adjacency_enrichment=round(adj_enrichment, 4),
        validations=[asdict(v) for v in validations],
        n_passed=n_passed,
        n_total=len(validations),
        runtime_seconds=runtime,
    )

    out_path = _save_json(rd, 'word_validation.json', asdict(result))
    print(f"\n  Saved → {out_path}")
    print(f"  Completed in {runtime:.1f}s")
