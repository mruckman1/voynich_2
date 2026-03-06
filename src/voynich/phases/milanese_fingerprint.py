"""
Phase D.1 – Milanese Cipher Fingerprint Comparison
====================================================
Compare Voynich encoding structure to Milanese diplomatic cipher keys.
Examines sign inventory size, homophone presence, and syllabic sign usage
to quantify structural similarity between the Voynich and known 15th-century
Milanese ciphers.

This is exploratory — no formal gate, but reports findings that inform
the broader historical investigation.

Dependency chain:
    data/reference/milanese/milanese_cipher_keys.json
    results/stroke_features.json (Phase 14.2 — Voynich sign inventory)
        → milanese_fingerprint.json (this step)
"""

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


from voynich.core._paths import results_dir as _results_dir
from voynich.core.reference import load_milanese_reference


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
class MilaneseFingerprint:
    """Result of comparing Voynich structure to Milanese diplomatic ciphers."""
    n_ciphers_analyzed: int
    cipher_comparisons: List[Dict]  # cipher_id, cipher_name, date,
                                    # inventory_size, has_homophones,
                                    # has_syllabic, structural_similarity
    voynich_inventory_size: int     # 25 triples or 44 EVA chars
    most_similar_cipher: str
    best_similarity: float
    mean_similarity: float
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Comparison logic
# ---------------------------------------------------------------------------

def _compute_structural_similarity(
    cipher: Dict[str, Any],
    voynich_inventory_size: int,
    voynich_has_homophones: bool,
    voynich_syllabic_ratio: float,
) -> float:
    """Compute a weighted structural similarity score in [0, 1].

    Components (weighted average):
      0.35 * sign inventory size ratio (closer to 1.0 = more similar)
      0.35 * homophone presence match (1.0 if both have or both lack)
      0.30 * syllabic sign ratio closeness
    """
    # --- Sign inventory size ratio ---
    cipher_size = cipher.get('inventory_size', 0)
    if cipher_size > 0 and voynich_inventory_size > 0:
        ratio = min(cipher_size, voynich_inventory_size) / max(cipher_size, voynich_inventory_size)
    else:
        ratio = 0.0

    # --- Homophone presence match ---
    cipher_homophones = cipher.get('has_homophones', False)
    homophone_match = 1.0 if cipher_homophones == voynich_has_homophones else 0.0

    # --- Syllabic sign ratio closeness ---
    cipher_syllabic_ratio = cipher.get('syllabic_ratio', 0.0)
    syllabic_closeness = 1.0 - min(abs(cipher_syllabic_ratio - voynich_syllabic_ratio), 1.0)

    # Weighted average
    similarity = 0.35 * ratio + 0.35 * homophone_match + 0.30 * syllabic_closeness
    return round(similarity, 4)


def _extract_cipher_features(cipher: Dict[str, Any]) -> Dict[str, Any]:
    """Extract or infer structural features from a cipher entry."""
    signs = cipher.get('signs', cipher.get('symbols', []))
    inventory_size = cipher.get('inventory_size', len(signs) if isinstance(signs, list) else 0)

    # Check for homophones: multiple signs mapping to the same plaintext value
    has_homophones = cipher.get('has_homophones', False)
    if not has_homophones and isinstance(signs, list):
        plaintext_values = [s.get('plaintext', s.get('value', '')) for s in signs if isinstance(s, dict)]
        if plaintext_values:
            from collections import Counter
            value_counts = Counter(plaintext_values)
            has_homophones = any(c > 1 for c in value_counts.values())

    # Check for syllabic signs
    has_syllabic = cipher.get('has_syllabic', False)
    n_syllabic = 0
    if isinstance(signs, list):
        for s in signs:
            if isinstance(s, dict):
                stype = s.get('type', s.get('sign_type', ''))
                ptext = s.get('plaintext', s.get('value', ''))
                if stype == 'syllabic' or (isinstance(ptext, str) and len(ptext) > 1 and ptext.isalpha()):
                    n_syllabic += 1
                    has_syllabic = True

    syllabic_ratio = n_syllabic / inventory_size if inventory_size > 0 else 0.0

    # Check for nulls
    has_nulls = cipher.get('has_nulls', False)
    if not has_nulls and isinstance(signs, list):
        for s in signs:
            if isinstance(s, dict):
                stype = s.get('type', s.get('sign_type', ''))
                ptext = s.get('plaintext', s.get('value', ''))
                if stype == 'null' or ptext in ('null', 'nihil', '', None):
                    has_nulls = True
                    break

    return {
        'inventory_size': inventory_size,
        'has_homophones': has_homophones,
        'has_syllabic': has_syllabic,
        'has_nulls': has_nulls,
        'syllabic_ratio': round(syllabic_ratio, 4),
        'n_syllabic': n_syllabic,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_milanese_fingerprint() -> None:
    """Phase D.1: compare Voynich encoding structure to Milanese diplomatic ciphers."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE D.1: Milanese Cipher Fingerprint Comparison")
    print("=" * 70)

    rd = _results_dir()

    # ─── Step 1: Load Milanese cipher keys ───
    print("\n  1. Loading Milanese cipher keys ...")
    milanese_ciphers = load_milanese_reference()

    if not milanese_ciphers:
        # ─── Step 2: No data available — save minimal result ───
        print("     WARNING: No Milanese cipher data available.")
        print("     Expected: data/reference/milanese/milanese_cipher_keys.json")
        print("     Saving minimal result with gate_passed=False equivalent.")

        result = MilaneseFingerprint(
            n_ciphers_analyzed=0,
            cipher_comparisons=[],
            voynich_inventory_size=0,
            most_similar_cipher="N/A",
            best_similarity=0.0,
            mean_similarity=0.0,
            verdict=(
                "NO DATA: Milanese cipher key file not found. "
                "Cannot perform structural comparison. "
                "Place cipher data in data/reference/milanese/milanese_cipher_keys.json."
            ),
            runtime_seconds=round(time.time() - t0, 2),
        )

        out_path = os.path.join(rd, 'milanese_fingerprint.json')
        with open(out_path, 'w') as f:
            json.dump(_convert(result), f, indent=2)
        print(f"\n  -> {out_path}")
        return

    print(f"     {len(milanese_ciphers)} cipher(s) loaded")

    # ─── Step 3: Load stroke_features.json for Voynich sign inventory ───
    print("\n  3. Loading Voynich sign inventory from stroke_features.json ...")
    sf_path = os.path.join(rd, 'stroke_features.json')
    voynich_inventory_size = 25  # default: 25 attested triples
    voynich_n_collision_triples = 0

    if os.path.exists(sf_path):
        with open(sf_path) as f:
            sf_data = json.load(f)
        voynich_inventory_size = sf_data.get('n_attested_triples', 25)
        voynich_n_collision_triples = sf_data.get('n_collision_triples', 0)
        print(f"     Voynich: {voynich_inventory_size} attested triples "
              f"({voynich_n_collision_triples} collisions from 44 EVA glyphs)")
    else:
        print(f"     stroke_features.json not found, using default: {voynich_inventory_size} triples")

    # Voynich structural properties (known from Phase 14 analysis)
    # Multiple EVA glyphs map to the same triple -> homophone-like behaviour
    voynich_has_homophones = voynich_n_collision_triples > 0
    # The feature model treats each triple as a syllable -> high syllabic ratio
    voynich_syllabic_ratio = 1.0  # all triples are syllabic in the feature model

    # ─── Step 4: Compare each Milanese cipher to Voynich ───
    print("\n  4. Comparing cipher structures ...")
    comparisons: List[Dict] = []

    for idx, cipher in enumerate(milanese_ciphers):
        cipher_id = cipher.get('id', cipher.get('cipher_id', f'cipher_{idx+1}'))
        cipher_name = cipher.get('name', cipher.get('cipher_name', f'Cipher {idx+1}'))
        cipher_date = cipher.get('date', cipher.get('year', 'unknown'))

        features = _extract_cipher_features(cipher)

        similarity = _compute_structural_similarity(
            features,
            voynich_inventory_size,
            voynich_has_homophones,
            voynich_syllabic_ratio,
        )

        comparison = {
            'cipher_id': str(cipher_id),
            'cipher_name': str(cipher_name),
            'date': str(cipher_date),
            'inventory_size': features['inventory_size'],
            'has_homophones': features['has_homophones'],
            'has_syllabic': features['has_syllabic'],
            'has_nulls': features['has_nulls'],
            'syllabic_ratio': features['syllabic_ratio'],
            'n_syllabic': features['n_syllabic'],
            'structural_similarity': similarity,
        }
        comparisons.append(comparison)

        print(f"     {cipher_name:<35} inv={features['inventory_size']:>3}  "
              f"homo={'Y' if features['has_homophones'] else 'N'}  "
              f"syll={'Y' if features['has_syllabic'] else 'N'}  "
              f"null={'Y' if features['has_nulls'] else 'N'}  "
              f"sim={similarity:.3f}")

    # ─── Step 5: Compute overall statistics ───
    print("\n  5. Computing aggregate similarity statistics ...")
    similarities = [c['structural_similarity'] for c in comparisons]
    mean_sim = sum(similarities) / len(similarities) if similarities else 0.0

    # Sort by similarity descending
    comparisons.sort(key=lambda c: c['structural_similarity'], reverse=True)
    best = comparisons[0] if comparisons else None
    best_similarity = best['structural_similarity'] if best else 0.0
    most_similar = best['cipher_name'] if best else 'N/A'

    print(f"     Mean similarity: {mean_sim:.4f}")
    print(f"     Best match: {most_similar} (similarity={best_similarity:.4f})")

    # ─── Step 6: Identify most similar cipher ───
    print(f"\n  6. Most similar cipher: {most_similar}")
    if best:
        print(f"     Date: {best['date']}")
        print(f"     Inventory size: {best['inventory_size']} "
              f"(Voynich: {voynich_inventory_size})")
        print(f"     Homophones: {'Yes' if best['has_homophones'] else 'No'}")
        print(f"     Syllabic signs: {'Yes' if best['has_syllabic'] else 'No'}")

    # ─── Step 7: Build verdict ───
    if best_similarity >= 0.7:
        verdict = (
            f"STRONG MATCH: Best similarity {best_similarity:.3f} with {most_similar}. "
            f"Voynich encoding structure closely resembles Milanese diplomatic ciphers. "
            f"Mean similarity across {len(comparisons)} ciphers: {mean_sim:.3f}."
        )
    elif best_similarity >= 0.4:
        verdict = (
            f"MODERATE MATCH: Best similarity {best_similarity:.3f} with {most_similar}. "
            f"Some structural overlap with Milanese ciphers but not conclusive. "
            f"Mean similarity: {mean_sim:.3f}."
        )
    else:
        verdict = (
            f"WEAK MATCH: Best similarity {best_similarity:.3f} with {most_similar}. "
            f"Voynich encoding structure does not closely resemble Milanese diplomatic ciphers. "
            f"Mean similarity: {mean_sim:.3f}."
        )

    print(f"\n  Verdict: {verdict}")

    # ─── Save ───
    result = MilaneseFingerprint(
        n_ciphers_analyzed=len(comparisons),
        cipher_comparisons=comparisons,
        voynich_inventory_size=voynich_inventory_size,
        most_similar_cipher=most_similar,
        best_similarity=best_similarity,
        mean_similarity=round(mean_sim, 4),
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'milanese_fingerprint.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)

    print(f"\n  -> {out_path}")
