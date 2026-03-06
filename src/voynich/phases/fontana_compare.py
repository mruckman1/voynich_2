"""
Phase B.5 -- Fontana Cipher Structural Comparison
===================================================
Structural comparison of Voynich signs to Fontana cipher signs.

Groups both Fontana and Voynich signs into families by shared first_stroke,
then compares the family structures: count, members per family, and
differentiation method (last_stroke vs glyph_class variation).

Also computes triple-level similarity (EXACT and NEAR) and coverage
metrics in both directions (Fontana->Voynich and Voynich->Fontana).

This is exploratory — no formal gate.  Results inform whether the
Fontana cipher (a known 15th-century Italian cipher) shares structural
properties with the Voynich manuscript's sign system.

Dependency chain:
    results/stroke_features.json
    data/reference/fontana/fontana_signs.json
        -> fontana_compare.json (this step)
"""

import json
import os
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from voynich.core._paths import results_dir as _results_dir, data_dir as _data_dir
from voynich.core.reference import detect_sign_families, load_fontana_reference
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
class FontanaCompareResult:
    """Structural comparison between Voynich and Fontana cipher signs."""
    n_fontana_signs: int
    n_fontana_families: int
    n_voynich_families: int
    family_count_ratio: float
    n_exact_triple_matches: int
    n_near_triple_matches: int
    fontana_coverage: float
    voynich_coverage: float
    structural_similarity_score: float
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


def _build_voynich_families(
    attested_triples: List[Dict],
) -> List[Dict[str, Any]]:
    """Group Voynich attested triples into families by shared first_stroke."""
    by_first: Dict[str, List[Dict]] = defaultdict(list)

    for t in attested_triples:
        tk = t.get('triple_key', '')
        parts = tk.split(',')
        if len(parts) >= 1:
            first_stroke = parts[0]
            by_first[first_stroke].append(t)

    families: List[Dict[str, Any]] = []
    for idx, (fs, members) in enumerate(sorted(by_first.items()), 1):
        if len(members) < 2:
            continue
        triple_keys = sorted(set(m.get('triple_key', '') for m in members))
        families.append({
            'family_id': f'V_FAM_{idx:02d}',
            'common_first_stroke': fs,
            'n_members': len(members),
            'triple_keys': triple_keys,
            'n_distinct_triples': len(triple_keys),
        })

    return families


def _compute_triple_matches(
    voynich_triples: List[Tuple[str, Dict[str, str]]],
    fontana_signs: List[Dict],
) -> Tuple[int, int, List[str], List[str]]:
    """Compute EXACT and NEAR triple matches between Voynich and Fontana.

    Returns (n_exact, n_near, fontana_matched_ids, voynich_matched_keys).
    """
    n_exact = 0
    n_near = 0
    fontana_matched: set = set()
    voynich_matched: set = set()

    for v_key, v_triple in voynich_triples:
        for f_sign in fontana_signs:
            fs = f_sign.get('first_stroke', '')
            ls = f_sign.get('last_stroke', '')
            gc = f_sign.get('glyph_class', '')
            if not (fs and ls and gc):
                continue

            f_triple = {'first_stroke': fs, 'last_stroke': ls, 'glyph_class': gc}
            sim = cosine_similarity_triples(v_triple, f_triple)

            f_id = f_sign.get('sign_id', f_sign.get('id', '?'))

            if sim >= 0.95:
                n_exact += 1
                fontana_matched.add(str(f_id))
                voynich_matched.add(v_key)
            elif sim >= 0.80:
                n_near += 1
                fontana_matched.add(str(f_id))
                voynich_matched.add(v_key)

    return n_exact, n_near, sorted(fontana_matched), sorted(voynich_matched)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_fontana_compare() -> None:
    """Phase B.5: Structural comparison of Voynich signs to Fontana cipher signs."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE B.5: Fontana Cipher Structural Comparison")
    print("=" * 70)

    rd = _results_dir()

    # ---- Step 1: Load stroke_features.json (25 Voynich triples) ----
    print("\n  1. Loading stroke features (25 Voynich triples) ...")
    sf_path = os.path.join(rd, 'stroke_features.json')
    if not os.path.exists(sf_path):
        print("      [ERROR] stroke_features.json not found. Run stroke-features first.")
        return

    with open(sf_path) as f:
        sf_data = json.load(f)

    attested_triples = sf_data.get('attested_triples', [])
    voynich_triples: List[Tuple[str, Dict[str, str]]] = []
    for t in attested_triples:
        tk = t.get('triple_key', '')
        voynich_triples.append((tk, _parse_triple_key(tk)))

    print(f"      {len(voynich_triples)} Voynich triples loaded")

    # ---- Step 2: Load Fontana signs ----
    print("\n  2. Loading Fontana cipher signs ...")
    fontana_signs = load_fontana_reference()
    n_fontana = len(fontana_signs)
    print(f"      {n_fontana} Fontana signs loaded")

    # ---- Step 3: Group Fontana signs into families ----
    print("\n  3. Grouping Fontana signs into families (by shared first_stroke) ...")
    fontana_families = detect_sign_families(fontana_signs)
    n_fontana_families = len(fontana_families)
    print(f"      {n_fontana_families} Fontana families detected")

    for fam in fontana_families:
        triples_str = ', '.join(fam.get('triple_keys', [])[:3])
        extra = ''
        if len(fam.get('triple_keys', [])) > 3:
            extra = f' ... (+{len(fam["triple_keys"]) - 3})'
        print(f"      {fam['family_id']}: {fam['common_first_stroke']} "
              f"({fam['n_members']} members, "
              f"{fam.get('n_distinct_triples', 0)} distinct triples: "
              f"{triples_str}{extra})")

    # ---- Step 4: Group Voynich triples into families ----
    print("\n  4. Grouping Voynich triples into families (by shared first_stroke) ...")
    voynich_families = _build_voynich_families(attested_triples)
    n_voynich_families = len(voynich_families)
    print(f"      {n_voynich_families} Voynich families detected")

    for fam in voynich_families:
        triples_str = ', '.join(fam.get('triple_keys', [])[:3])
        extra = ''
        if len(fam.get('triple_keys', [])) > 3:
            extra = f' ... (+{len(fam["triple_keys"]) - 3})'
        print(f"      {fam['family_id']}: {fam['common_first_stroke']} "
              f"({fam['n_members']} members, "
              f"{fam['n_distinct_triples']} distinct triples: "
              f"{triples_str}{extra})")

    # ---- Step 5: Compare family structures ----
    print("\n  5. Comparing family structures ...")
    family_count_ratio = (
        n_fontana_families / n_voynich_families
        if n_voynich_families > 0 else 0.0
    )
    print(f"      Fontana families: {n_fontana_families}")
    print(f"      Voynich families: {n_voynich_families}")
    print(f"      Family count ratio (Fontana/Voynich): {family_count_ratio:.2f}")

    # Compare family first_stroke distributions
    fontana_first_strokes = set(
        fam['common_first_stroke'] for fam in fontana_families
    )
    voynich_first_strokes = set(
        fam['common_first_stroke'] for fam in voynich_families
    )
    shared_first_strokes = fontana_first_strokes & voynich_first_strokes
    print(f"      Shared first_stroke types: {sorted(shared_first_strokes)}")

    # ---- Step 6: Compute triple-level similarities ----
    print("\n  6. Computing triple-level matches ...")
    n_exact, n_near, fontana_matched, voynich_matched = _compute_triple_matches(
        voynich_triples, fontana_signs,
    )
    print(f"      EXACT matches (>= 0.95): {n_exact}")
    print(f"      NEAR matches  (>= 0.80): {n_near}")

    # ---- Step 7: Coverage metrics ----
    fontana_coverage = (
        len(fontana_matched) / n_fontana if n_fontana > 0 else 0.0
    )
    voynich_coverage = (
        len(voynich_matched) / len(voynich_triples) if voynich_triples else 0.0
    )

    print(f"\n  7. Coverage metrics:")
    print(f"      Fontana coverage (fraction matched by Voynich): "
          f"{fontana_coverage:.4f} ({fontana_coverage:.1%})")
    print(f"      Voynich coverage (fraction matched by Fontana): "
          f"{voynich_coverage:.4f} ({voynich_coverage:.1%})")

    # ---- Structural similarity score ----
    # Weighted composite: 40% family structure match, 30% Fontana coverage, 30% Voynich coverage
    family_structure_score = (
        len(shared_first_strokes) /
        max(len(fontana_first_strokes | voynich_first_strokes), 1)
    )
    structural_similarity = (
        0.40 * family_structure_score +
        0.30 * fontana_coverage +
        0.30 * voynich_coverage
    )

    print(f"\n      Family structure Jaccard: {family_structure_score:.4f}")
    print(f"      Structural similarity score: {structural_similarity:.4f}")

    # ---- Verdict (exploratory, no formal gate) ----
    if n_fontana == 0:
        verdict = (
            f"INCONCLUSIVE: No Fontana signs loaded. "
            f"Cannot compare structural patterns."
        )
    elif structural_similarity >= 0.4:
        verdict = (
            f"HIGH SIMILARITY: Structural similarity {structural_similarity:.3f} "
            f"(Fontana coverage {fontana_coverage:.1%}, "
            f"Voynich coverage {voynich_coverage:.1%}). "
            f"{n_fontana_families} Fontana families vs {n_voynich_families} Voynich families "
            f"(ratio {family_count_ratio:.2f}). "
            f"Fontana cipher shares significant structural overlap with Voynich signs."
        )
    elif structural_similarity >= 0.2:
        verdict = (
            f"MODERATE SIMILARITY: Structural similarity {structural_similarity:.3f} "
            f"(Fontana coverage {fontana_coverage:.1%}, "
            f"Voynich coverage {voynich_coverage:.1%}). "
            f"Some shared family structure ({len(shared_first_strokes)} common "
            f"first_stroke types) but limited triple-level overlap."
        )
    else:
        verdict = (
            f"LOW SIMILARITY: Structural similarity {structural_similarity:.3f}. "
            f"Fontana cipher signs differ structurally from Voynich signs. "
            f"Only {n_exact} exact + {n_near} near matches out of "
            f"{n_fontana} x {len(voynich_triples)} comparisons."
        )

    print(f"\n  Verdict: {verdict}")

    # ---- Save ----
    result = FontanaCompareResult(
        n_fontana_signs=n_fontana,
        n_fontana_families=n_fontana_families,
        n_voynich_families=n_voynich_families,
        family_count_ratio=round(family_count_ratio, 4),
        n_exact_triple_matches=n_exact,
        n_near_triple_matches=n_near,
        fontana_coverage=round(fontana_coverage, 4),
        voynich_coverage=round(voynich_coverage, 4),
        structural_similarity_score=round(structural_similarity, 4),
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'fontana_compare.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)

    print(f"\n  Results saved -> {out_path}")
