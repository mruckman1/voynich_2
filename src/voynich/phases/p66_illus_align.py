"""
Phase 66, Track 4: Illustration-Text Alignment
================================================
For folios with plant identifications, check if CVC-decoded tokens
contain syllable sequences from that plant's Latin name.
Null: 1000 random permutations of folio-to-plant assignments.

Dependency chain:
    results/consensus_plants.json     (Phase 31)
    results/combined_refine.json      (Phase 15)
        -> results/p66_illus_align.json
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import build_eva_to_triple_lookup, load_corpus, tokenize_eva_chars
from voynich.core.stats import syllabify_latin
from voynich.phases.corrected_coda import build_coda_table_v2, decode_corpus_cvc_v2


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
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class IllusAlignResult:
    phase: str = "66"
    step: str = "66.4"
    experiment: str = "illustration_alignment"
    n_folios_tested: int = 0
    n_folios_with_hits: int = 0
    total_hits: int = 0
    null_mean_hits: float = 0.0
    null_std_hits: float = 0.0
    enrichment: float = 0.0
    z_score: float = 0.0
    p_value: float = 1.0
    per_folio: List[Dict] = field(default_factory=list)
    i1_enrichment: bool = False    # enrichment > 1.5
    i2_z_score: bool = False       # z > 2.0
    gates_passed: int = 0
    gate_passed: bool = False
    verdict: str = ""
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_folio_plant_mappings(plants_data: Dict) -> List[Dict]:
    """Extract folio-to-plant-name mappings from consensus_plants.json."""
    mappings = []
    for tier_key in ['tier_a_folios', 'tier_b_folios']:
        entries = plants_data.get(tier_key, [])
        for entry in entries:
            folio = entry.get('folio', '')
            if not folio:
                continue
            # Collect all medieval/latin names for this folio
            names = []
            for mn in entry.get('medieval_names', []):
                for name_field in ['medieval_name', 'genus', 'linnaean_name']:
                    n = mn.get(name_field, '')
                    if n:
                        names.append(n.lower())
                for alt in mn.get('alternate_names', []):
                    if alt:
                        names.append(alt.lower())
            # Deduplicate
            names = list(dict.fromkeys(names))
            if names:
                mappings.append({'folio': folio, 'names': names})
    return mappings


def _syllable_bigrams(word: str) -> List[str]:
    """Return syllable bigrams from a Latin word."""
    syls = syllabify_latin(word)
    if len(syls) < 2:
        return syls  # single-syllable word: return the syllable itself
    return [syls[i] + syls[i + 1] for i in range(len(syls) - 1)]


def _count_hits(
    folio_decoded_map: Dict[str, str],
    folio_plant_pairs: List[Tuple[str, List[str]]],
) -> int:
    """Count syllable-bigram hits across all folio-plant pairs."""
    total = 0
    for folio, names in folio_plant_pairs:
        decoded_text = folio_decoded_map.get(folio, '')
        if not decoded_text:
            continue
        decoded_lower = decoded_text.lower()
        for name in names:
            # Split multi-word names and collect bigrams from each word
            for word in name.split():
                bigrams = _syllable_bigrams(word)
                for bg in bigrams:
                    if bg in decoded_lower:
                        total += 1
    return total


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_illus_align() -> IllusAlignResult:
    t0 = time.time()
    rd = str(_results_dir())
    result = IllusAlignResult()

    print("=" * 70)
    print("Phase 66, Track 4: Illustration-Text Alignment")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load consensus plants
    # ------------------------------------------------------------------
    plants_path = os.path.join(rd, 'consensus_plants.json')
    plants_data = _safe_load(plants_path)
    if not plants_data:
        print("[WARN] consensus_plants.json not found — cannot proceed.")
        result.verdict = "INSUFFICIENT_DATA"
        result.runtime_seconds = round(time.time() - t0, 2)
        _save_json(rd, 'p66_illus_align.json', result)
        return result

    mappings = _extract_folio_plant_mappings(plants_data)
    print(f"  Folio-plant mappings extracted: {len(mappings)}")
    if not mappings:
        print("[WARN] No folio-plant mappings found.")
        result.verdict = "INSUFFICIENT_DATA"
        result.runtime_seconds = round(time.time() - t0, 2)
        _save_json(rd, 'p66_illus_align.json', result)
        return result

    # ------------------------------------------------------------------
    # 2. Load corpus and CVC decode all folio tokens
    # ------------------------------------------------------------------
    print("  Loading corpus and decoding tokens ...")
    corpus = load_corpus(verbose=False)
    assignment = _safe_load(os.path.join(rd, 'combined_refine.json')).get(
        'best_assignment', {}
    )
    eva_to_triple = build_eva_to_triple_lookup()
    coda_table = build_coda_table_v2()

    # Build decoded text per folio
    folio_decoded_map: Dict[str, str] = {}
    for folio_id, page in corpus.pages.items():
        tokens = page.all_tokens
        if not tokens:
            continue
        decoded = decode_corpus_cvc_v2(tokens, assignment, eva_to_triple, coda_table)
        # Filter out empty decodes
        decoded_clean = [d for d in decoded if d and d != '???']
        folio_decoded_map[folio_id] = ' '.join(decoded_clean)

    print(f"  Decoded folios: {len(folio_decoded_map)}")

    # ------------------------------------------------------------------
    # 3. Build folio-plant pairs and count real hits
    # ------------------------------------------------------------------
    folio_plant_pairs: List[Tuple[str, List[str]]] = []
    per_folio_details: List[Dict] = []

    for m in mappings:
        folio = m['folio']
        names = m['names']
        folio_plant_pairs.append((folio, names))

    result.n_folios_tested = len(folio_plant_pairs)

    # Count hits per folio for detail
    for folio, names in folio_plant_pairs:
        decoded_text = folio_decoded_map.get(folio, '')
        decoded_lower = decoded_text.lower()
        folio_hits = 0
        hit_details = []
        for name in names:
            for word in name.split():
                bigrams = _syllable_bigrams(word)
                for bg in bigrams:
                    if bg in decoded_lower:
                        folio_hits += 1
                        hit_details.append({'name': name, 'bigram': bg})
        per_folio_details.append({
            'folio': folio,
            'plant_names': names,
            'hits': folio_hits,
            'hit_details': hit_details[:10],  # cap detail output
            'decoded_preview': decoded_text[:120],
        })

    real_hits = _count_hits(folio_decoded_map, folio_plant_pairs)
    result.total_hits = real_hits
    result.n_folios_with_hits = sum(1 for d in per_folio_details if d['hits'] > 0)
    result.per_folio = per_folio_details

    print(f"  Real hits: {real_hits} across {result.n_folios_with_hits}/{result.n_folios_tested} folios")

    # ------------------------------------------------------------------
    # 4. Null: 1000 permutations of folio-to-plant assignments
    # ------------------------------------------------------------------
    print("  Running 1000-trial permutation null ...")
    rng = random.Random(42)
    folios_list = [fp[0] for fp in folio_plant_pairs]
    names_list = [fp[1] for fp in folio_plant_pairs]
    null_hits = []

    for _ in range(1000):
        shuffled_names = names_list[:]
        rng.shuffle(shuffled_names)
        perm_pairs = list(zip(folios_list, shuffled_names))
        h = _count_hits(folio_decoded_map, perm_pairs)
        null_hits.append(h)

    null_arr = np.array(null_hits, dtype=float)
    null_mean = float(np.mean(null_arr))
    null_std = float(np.std(null_arr))

    result.null_mean_hits = round(null_mean, 4)
    result.null_std_hits = round(null_std, 4)

    # Enrichment
    if null_mean > 0:
        result.enrichment = round(real_hits / null_mean, 4)
    else:
        result.enrichment = float(real_hits) if real_hits > 0 else 0.0

    # Z-score
    if null_std > 0:
        result.z_score = round((real_hits - null_mean) / null_std, 4)
    else:
        result.z_score = 0.0

    # P-value (one-tailed)
    n_ge = int(np.sum(null_arr >= real_hits))
    result.p_value = round((n_ge + 1) / (1000 + 1), 6)

    print(f"  Null mean: {result.null_mean_hits}, std: {result.null_std_hits}")
    print(f"  Enrichment: {result.enrichment}x, z-score: {result.z_score}, p: {result.p_value}")

    # ------------------------------------------------------------------
    # 5. Gate evaluation
    # ------------------------------------------------------------------
    result.i1_enrichment = result.enrichment > 1.5
    result.i2_z_score = result.z_score > 2.0
    result.gates_passed = sum([result.i1_enrichment, result.i2_z_score])
    result.gate_passed = result.gates_passed >= 2

    if result.gate_passed:
        result.verdict = "ALIGNMENT_SIGNIFICANT"
    elif result.gates_passed == 1:
        result.verdict = "ALIGNMENT_MARGINAL"
    else:
        result.verdict = "ALIGNMENT_NOT_FOUND"

    # ------------------------------------------------------------------
    # 6. Print summary and save
    # ------------------------------------------------------------------
    result.runtime_seconds = round(time.time() - t0, 2)

    print()
    print("-" * 50)
    print(f"  I1 enrichment > 1.5x:  {result.i1_enrichment}  ({result.enrichment}x)")
    print(f"  I2 z-score > 2.0:      {result.i2_z_score}  (z={result.z_score})")
    print(f"  Gates passed:          {result.gates_passed}/2")
    print(f"  Verdict:               {result.verdict}")
    print(f"  Runtime:               {result.runtime_seconds}s")
    print("-" * 50)

    path = _save_json(rd, 'p66_illus_align.json', result)
    print(f"  Saved: {path}")
    return result
