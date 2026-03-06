"""
Phase B.1 -- Triple Overlap Analysis
======================================
Compare Voynich stroke triples to Tironian reference triples.

For each of the 25 attested Voynich triples (from stroke_features.json),
compute cosine similarity against every reference sign in the master
paleographic reference.  Classify matches as EXACT (>= 0.95), NEAR
(>= 0.80), POSSIBLE (>= 0.65), or NONE (< 0.65).

A null baseline is computed by shuffling stroke types across 100 random
triple assignments and measuring expected overlap, yielding a selectivity
ratio that quantifies how much better the real overlap is than chance.

Gates:
    B.1a: >= 40% of Voynich signs have EXACT or NEAR matches
    B.1c: selectivity > 1.5

Dependency chain:
    results/stroke_features.json
    data/reference/paleographic/master_reference.json
        -> triple_overlap.json (this step)
"""

import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from voynich.core._paths import results_dir as _results_dir, data_dir as _data_dir
from voynich.core.reference import load_master_reference
from voynich.core.stats import cosine_similarity_triples


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
class TripleOverlapResult:
    """Full triple overlap analysis between Voynich and Tironian reference."""
    n_voynich_triples: int
    n_reference_signs: int
    n_exact: int
    n_near: int
    n_possible: int
    n_none: int
    exact_triples: List[str]
    near_triples: List[Dict]
    overlap_rate: float
    jaccard_similarity: float
    null_mean_overlap: float
    null_std_overlap: float
    selectivity: float
    gate_b1a_passed: bool
    gate_b1c_passed: bool
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def _parse_triple_key(triple_key: str) -> Dict[str, str]:
    """Parse 'first_stroke,last_stroke,glyph_class' into a dict."""
    parts = triple_key.split(',')
    if len(parts) == 3:
        return {
            'first_stroke': parts[0],
            'last_stroke': parts[1],
            'glyph_class': parts[2],
        }
    return {'first_stroke': '', 'last_stroke': '', 'glyph_class': ''}


def _extract_reference_triples(master_ref: Dict) -> List[Dict]:
    """Extract all signs from master reference that have stroke decomposition."""
    signs = []
    for sign in master_ref.get('all_signs', []):
        fs = sign.get('first_stroke', '')
        ls = sign.get('last_stroke', '')
        gc = sign.get('glyph_class', '')
        if fs and ls and gc:
            sign_copy = dict(sign)
            sign_copy['_triple'] = {
                'first_stroke': fs,
                'last_stroke': ls,
                'glyph_class': gc,
            }
            sign_copy['_triple_key'] = f"{fs},{ls},{gc}"
            signs.append(sign_copy)
    return signs


def _classify_match(similarity: float) -> str:
    """Classify a similarity score into match tier."""
    if similarity >= 0.95:
        return 'EXACT'
    elif similarity >= 0.80:
        return 'NEAR'
    elif similarity >= 0.65:
        return 'POSSIBLE'
    else:
        return 'NONE'


def _compute_null_overlap(
    voynich_triple_dicts: List[Dict[str, str]],
    ref_triples: List[Dict],
    n_permutations: int = 100,
) -> Tuple[float, float]:
    """Compute null baseline overlap by shuffling stroke types.

    For each permutation, randomly reassign stroke types within each
    component (first_stroke, last_stroke, glyph_class) independently,
    then measure overlap rate against reference.
    """
    rng = random.Random(42)

    # Collect all stroke types used in Voynich triples
    first_strokes = [t['first_stroke'] for t in voynich_triple_dicts]
    last_strokes = [t['last_stroke'] for t in voynich_triple_dicts]
    glyph_classes = [t['glyph_class'] for t in voynich_triple_dicts]

    null_overlaps: List[float] = []

    for _ in range(n_permutations):
        shuffled_first = first_strokes[:]
        shuffled_last = last_strokes[:]
        shuffled_class = glyph_classes[:]
        rng.shuffle(shuffled_first)
        rng.shuffle(shuffled_last)
        rng.shuffle(shuffled_class)

        n_match = 0
        for i in range(len(voynich_triple_dicts)):
            rand_triple = {
                'first_stroke': shuffled_first[i],
                'last_stroke': shuffled_last[i],
                'glyph_class': shuffled_class[i],
            }
            best_sim = 0.0
            for ref_sign in ref_triples:
                sim = cosine_similarity_triples(rand_triple, ref_sign['_triple'])
                if sim > best_sim:
                    best_sim = sim
            if best_sim >= 0.80:
                n_match += 1

        overlap_rate = n_match / len(voynich_triple_dicts) if voynich_triple_dicts else 0.0
        null_overlaps.append(overlap_rate)

    if not null_overlaps:
        return 0.0, 0.0

    mean_overlap = sum(null_overlaps) / len(null_overlaps)
    variance = sum((x - mean_overlap) ** 2 for x in null_overlaps) / len(null_overlaps)
    std_overlap = math.sqrt(variance)

    return mean_overlap, std_overlap


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_triple_overlap() -> None:
    """Phase B.1: Compare Voynich stroke triples to Tironian reference triples."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE B.1: Triple Overlap Analysis")
    print("=" * 70)

    rd = _results_dir()

    # ---- Step 1: Load stroke_features.json ----
    print("\n  1. Loading stroke features (25 attested triples) ...")
    sf_path = os.path.join(rd, 'stroke_features.json')
    if not os.path.exists(sf_path):
        print("      [ERROR] stroke_features.json not found. Run stroke-features first.")
        return

    with open(sf_path) as f:
        sf_data = json.load(f)

    attested_triples = sf_data.get('attested_triples', [])
    voynich_triple_keys: List[str] = []
    voynich_triple_dicts: List[Dict[str, str]] = []

    for t in attested_triples:
        tk = t.get('triple_key', '')
        voynich_triple_keys.append(tk)
        voynich_triple_dicts.append(_parse_triple_key(tk))

    print(f"      {len(voynich_triple_keys)} Voynich triples loaded")

    # ---- Step 2: Load master reference ----
    print("\n  2. Loading master paleographic reference ...")
    master_ref = load_master_reference()

    if master_ref is None:
        print("      [WARN] master_reference.json not found.")
        print("      Creating result with empty reference data.")
        ref_signs = []
    else:
        ref_signs = _extract_reference_triples(master_ref)
        print(f"      {len(ref_signs)} reference signs with stroke decomposition")

    # ---- Step 3: Compute pairwise similarities ----
    print("\n  3. Computing cosine similarity for each Voynich triple vs reference ...")
    exact_triples: List[str] = []
    near_triples: List[Dict] = []
    possible_triples: List[str] = []
    none_triples: List[str] = []

    for i, v_triple in enumerate(voynich_triple_dicts):
        v_key = voynich_triple_keys[i]
        best_sim = 0.0
        best_ref_id = ''
        best_ref_triple = ''

        for ref_sign in ref_signs:
            sim = cosine_similarity_triples(v_triple, ref_sign['_triple'])
            if sim > best_sim:
                best_sim = sim
                best_ref_id = ref_sign.get('sign_id', ref_sign.get('id', ''))
                best_ref_triple = ref_sign['_triple_key']

        match_type = _classify_match(best_sim)
        if match_type == 'EXACT':
            exact_triples.append(v_key)
        elif match_type == 'NEAR':
            near_triples.append({
                'voynich_triple': v_key,
                'best_ref_id': best_ref_id,
                'best_ref_triple': best_ref_triple,
                'similarity': round(best_sim, 4),
            })
        elif match_type == 'POSSIBLE':
            possible_triples.append(v_key)
        else:
            none_triples.append(v_key)

    n_exact = len(exact_triples)
    n_near = len(near_triples)
    n_possible = len(possible_triples)
    n_none = len(none_triples)
    n_total = len(voynich_triple_keys)

    print(f"      EXACT (>= 0.95): {n_exact}")
    print(f"      NEAR  (>= 0.80): {n_near}")
    print(f"      POSSIBLE (>= 0.65): {n_possible}")
    print(f"      NONE  (< 0.65): {n_none}")

    # ---- Step 4: Overlap rate ----
    overlap_rate = (n_exact + n_near) / n_total if n_total > 0 else 0.0
    print(f"\n  4. Overlap rate (EXACT + NEAR): {overlap_rate:.4f} ({overlap_rate:.1%})")

    # ---- Step 5: Jaccard similarity ----
    print("\n  5. Computing Jaccard similarity on triple keys ...")
    voynich_key_set = set(voynich_triple_keys)
    ref_key_set = set(s['_triple_key'] for s in ref_signs)
    intersection = voynich_key_set & ref_key_set
    union = voynich_key_set | ref_key_set
    jaccard = len(intersection) / len(union) if union else 0.0
    print(f"      |Voynich| = {len(voynich_key_set)}, |Reference| = {len(ref_key_set)}")
    print(f"      |Intersection| = {len(intersection)}, |Union| = {len(union)}")
    print(f"      Jaccard similarity: {jaccard:.4f}")

    # ---- Step 6: Null baseline ----
    print("\n  6. Computing null baseline (100 random triple shuffles) ...")
    null_mean, null_std = _compute_null_overlap(
        voynich_triple_dicts, ref_signs, n_permutations=100,
    )
    print(f"      Null mean overlap: {null_mean:.4f} +/- {null_std:.4f}")

    # ---- Step 7: Selectivity ----
    selectivity = overlap_rate / max(null_mean, 0.001)
    print(f"\n  7. Selectivity: {selectivity:.2f}x (real / null)")

    # ---- Step 8: Gate B.1a ----
    gate_b1a = overlap_rate >= 0.40
    print(f"\n  8. Gate B.1a (>= 40% EXACT+NEAR): "
          f"{'PASS' if gate_b1a else 'FAIL'} ({overlap_rate:.1%})")

    # ---- Step 9: Gate B.1c ----
    gate_b1c = selectivity > 1.5
    print(f"\n  9. Gate B.1c (selectivity > 1.5): "
          f"{'PASS' if gate_b1c else 'FAIL'} ({selectivity:.2f}x)")

    # ---- Verdict ----
    if gate_b1a and gate_b1c:
        verdict = (
            f"PASS: {overlap_rate:.1%} of Voynich triples match Tironian signs "
            f"(EXACT+NEAR), selectivity {selectivity:.2f}x over null. "
            f"Strong structural correspondence."
        )
    elif gate_b1a or gate_b1c:
        verdict = (
            f"PARTIAL: overlap {overlap_rate:.1%} "
            f"(gate B.1a {'PASS' if gate_b1a else 'FAIL'}), "
            f"selectivity {selectivity:.2f}x "
            f"(gate B.1c {'PASS' if gate_b1c else 'FAIL'}). "
            f"Mixed evidence for Tironian correspondence."
        )
    else:
        verdict = (
            f"FAIL: overlap {overlap_rate:.1%} < 40% and "
            f"selectivity {selectivity:.2f}x <= 1.5. "
            f"Voynich triples do not resemble Tironian signs structurally."
        )

    print(f"\n  Verdict: {verdict}")

    # ---- Save ----
    result = TripleOverlapResult(
        n_voynich_triples=n_total,
        n_reference_signs=len(ref_signs),
        n_exact=n_exact,
        n_near=n_near,
        n_possible=n_possible,
        n_none=n_none,
        exact_triples=exact_triples,
        near_triples=near_triples,
        overlap_rate=round(overlap_rate, 4),
        jaccard_similarity=round(jaccard, 4),
        null_mean_overlap=round(null_mean, 4),
        null_std_overlap=round(null_std, 4),
        selectivity=round(selectivity, 2),
        gate_b1a_passed=gate_b1a,
        gate_b1c_passed=gate_b1c,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'triple_overlap.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)

    print(f"\n  Results saved -> {out_path}")
