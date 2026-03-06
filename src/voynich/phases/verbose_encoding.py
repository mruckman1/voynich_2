"""
Phase D.3 – Verbose Encoding Assessment
=========================================
Assess whether the Voynich uses verbose (one-to-many) encoding by examining
variant forms.  If multiple distinct signs map to the same phonetic value,
the encoding is inherently verbose — each plaintext unit has multiple
cipher representations, inflating the apparent entropy.

Compares:
  - Master paleographic reference: how many distinct signs map to each
    Latin syllable/value?
  - Voynich feature model: how many EVA glyphs map to each unique
    (first_stroke, last_stroke, glyph_class) triple?

If both show >1.5 variants per unit, the encoding is inherently verbose
and explains the entropy floor observed in Phase 9.5.

Dependency chain:
    data/reference/paleographic/master_reference.json
    results/stroke_features.json (Phase 14.2)
        -> verbose_encoding.json (this step)
"""

import json
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from voynich.core._paths import results_dir as _results_dir
from voynich.core.reference import EVA_VISUAL_COMPONENTS, load_master_reference


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
class VerboseEncodingResult:
    """Result of verbose encoding assessment."""
    n_tironian_signs: int
    n_unique_latin_values: int
    tironian_variants_per_value: float
    max_variants: int
    max_variants_value: str
    voynich_glyphs_per_triple: float    # 44/25 = 1.76
    voynich_collisions: List[Dict]      # triple_key, n_glyphs, glyphs
    both_verbose: bool                  # both > 1.5
    entropy_floor_explained: bool
    interpretation: str
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def _analyse_reference_variants(
    master_ref: Dict[str, Any],
) -> Dict[str, Any]:
    """Analyse variant forms in the master paleographic reference.

    For each latin_value (syllable or word) that appears in the reference,
    count how many DISTINCT signs (different triple_keys or sign_ids) map
    to it.

    Returns a summary dict with variant statistics.
    """
    # The master_reference may have various structures; extract sign-to-value
    # mappings from whichever keys are available.
    signs = master_ref.get('signs', [])
    if not signs:
        # Try alternative structures
        signs = master_ref.get('entries', [])
    if not signs:
        # Maybe it's a flat mapping of sign_id -> value
        if isinstance(master_ref, dict) and 'signs' not in master_ref:
            # Treat top-level keys as sign_ids
            signs = []
            for key, val in master_ref.items():
                if isinstance(val, dict):
                    entry = dict(val)
                    entry.setdefault('sign_id', key)
                    signs.append(entry)
                elif isinstance(val, str):
                    signs.append({'sign_id': key, 'latin_value': val})

    # Build value -> set of distinct sign identifiers
    value_to_signs: Dict[str, set] = {}
    n_total_signs = 0

    for sign in signs:
        if not isinstance(sign, dict):
            continue

        # Get the Latin value this sign represents
        latin_value = sign.get('latin_value', sign.get('value',
                      sign.get('plaintext', sign.get('syllable', ''))))
        if not latin_value or not isinstance(latin_value, str):
            continue

        # Get a unique identifier for this sign
        sign_id = sign.get('sign_id', sign.get('id',
                  sign.get('triple_key', sign.get('glyph', ''))))
        if not sign_id:
            sign_id = str(sign)

        latin_value = latin_value.strip().lower()
        if latin_value:
            value_to_signs.setdefault(latin_value, set()).add(sign_id)
            n_total_signs += 1

    if not value_to_signs:
        return {
            'n_signs': n_total_signs,
            'n_values': 0,
            'variants_per_value': 0.0,
            'max_variants': 0,
            'max_variants_value': '',
            'value_distribution': {},
        }

    # Compute statistics
    variant_counts = {v: len(sids) for v, sids in value_to_signs.items()}
    n_values = len(variant_counts)
    mean_variants = sum(variant_counts.values()) / n_values if n_values > 0 else 0.0

    max_value = max(variant_counts, key=variant_counts.get) if variant_counts else ''
    max_count = variant_counts.get(max_value, 0)

    # Distribution: how many values have 1 variant, 2 variants, etc.
    count_dist: Counter = Counter(variant_counts.values())

    return {
        'n_signs': n_total_signs,
        'n_values': n_values,
        'variants_per_value': round(mean_variants, 4),
        'max_variants': max_count,
        'max_variants_value': max_value,
        'value_distribution': dict(count_dist),
    }


def _analyse_voynich_collisions() -> Dict[str, Any]:
    """Analyse Voynich EVA glyph to triple collisions.

    Multiple EVA glyphs can map to the same (first_stroke, last_stroke,
    glyph_class) triple. This is a form of one-to-many encoding at the
    glyph level — the same phonetic value can be written with different
    visual forms.

    Returns collision statistics and the list of collision triples.
    """
    # Build triple -> list of EVA glyphs
    triple_to_glyphs: Dict[str, List[str]] = {}
    n_total_glyphs = 0

    for glyph, comp in EVA_VISUAL_COMPONENTS.items():
        triple_key = (
            comp['first_stroke'] + ','
            + comp['last_stroke'] + ','
            + comp['glyph_class']
        )
        triple_to_glyphs.setdefault(triple_key, []).append(glyph)
        n_total_glyphs += 1

    n_triples = len(triple_to_glyphs)
    glyphs_per_triple = n_total_glyphs / n_triples if n_triples > 0 else 0.0

    # Find collision triples (more than 1 glyph)
    collisions: List[Dict] = []
    for triple_key in sorted(triple_to_glyphs.keys()):
        glyphs = triple_to_glyphs[triple_key]
        if len(glyphs) > 1:
            collisions.append({
                'triple_key': triple_key,
                'n_glyphs': len(glyphs),
                'glyphs': sorted(glyphs),
            })

    return {
        'n_total_glyphs': n_total_glyphs,
        'n_triples': n_triples,
        'glyphs_per_triple': round(glyphs_per_triple, 4),
        'collisions': collisions,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_verbose_encoding() -> None:
    """Phase D.3: assess whether the Voynich uses verbose encoding."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE D.3: Verbose Encoding Assessment")
    print("=" * 70)

    rd = _results_dir()

    # ─── Step 1: Load master_reference.json ───
    print("\n  1. Loading master paleographic reference ...")
    master_ref = load_master_reference()

    if master_ref is None:
        print("     WARNING: master_reference.json not found.")
        print("     Expected: data/reference/paleographic/master_reference.json")
        print("     Will analyse Voynich collisions only.")
        ref_stats = {
            'n_signs': 0, 'n_values': 0, 'variants_per_value': 0.0,
            'max_variants': 0, 'max_variants_value': '',
            'value_distribution': {},
        }
    else:
        print("     Master reference loaded")

        # ─── Step 2: Count variants per Latin value ───
        print("\n  2. Analysing variant forms per Latin value ...")
        ref_stats = _analyse_reference_variants(master_ref)
        print(f"     Total signs in reference: {ref_stats['n_signs']}")
        print(f"     Unique Latin values: {ref_stats['n_values']}")
        print(f"     Average variants per value: {ref_stats['variants_per_value']:.3f}")
        print(f"     Max variants: {ref_stats['max_variants']} (value: '{ref_stats['max_variants_value']}')")

        if ref_stats.get('value_distribution'):
            print("     Distribution of variant counts:")
            for n_variants, n_values in sorted(ref_stats['value_distribution'].items()):
                print(f"       {n_variants} variant(s): {n_values} value(s)")

    # ─── Step 3: Check if average > 1.5 (verbose) ───
    tironian_verbose = ref_stats['variants_per_value'] > 1.5
    if ref_stats['n_values'] > 0:
        print(f"\n  3. Reference verbose? "
              f"{'YES' if tironian_verbose else 'NO'} "
              f"({ref_stats['variants_per_value']:.3f} variants/value, threshold 1.5)")
    else:
        print("\n  3. Reference data unavailable for verbosity check")

    # ─── Step 4: Also load stroke_features.json for context ───
    print("\n  4. Loading stroke_features.json for Voynich triple context ...")
    sf_path = os.path.join(rd, 'stroke_features.json')
    if os.path.exists(sf_path):
        with open(sf_path) as f:
            sf_data = json.load(f)
        sf_n_triples = sf_data.get('n_attested_triples', 0)
        sf_n_collisions = sf_data.get('n_collision_triples', 0)
        print(f"     stroke_features.json: {sf_n_triples} triples, "
              f"{sf_n_collisions} collisions")
    else:
        print("     stroke_features.json not found, computing from EVA_VISUAL_COMPONENTS")

    # ─── Step 5: Analyse Voynich glyph-to-triple collisions ───
    print("\n  5. Analysing Voynich glyph-to-triple collisions ...")
    voynich_stats = _analyse_voynich_collisions()
    glyphs_per_triple = voynich_stats['glyphs_per_triple']
    collisions = voynich_stats['collisions']

    print(f"     Total EVA glyphs: {voynich_stats['n_total_glyphs']}")
    print(f"     Unique triples: {voynich_stats['n_triples']}")
    print(f"     Glyphs per triple: {glyphs_per_triple:.3f}")
    print(f"     Collision triples ({len(collisions)}):")
    for col in collisions:
        print(f"       {col['triple_key']}: {col['n_glyphs']} glyphs -> "
              f"{', '.join(col['glyphs'])}")

    # ─── Step 6: Compute glyph-per-triple ratio for Voynich ───
    voynich_verbose = glyphs_per_triple > 1.5
    print(f"\n  6. Voynich verbose? "
          f"{'YES' if voynich_verbose else 'NO'} "
          f"({glyphs_per_triple:.3f} glyphs/triple, threshold 1.5)")

    # ─── Step 7: Compare both systems ───
    print("\n  7. Cross-system comparison ...")
    both_verbose = tironian_verbose and voynich_verbose
    if both_verbose:
        print("     BOTH systems show verbose (one-to-many) encoding.")
    elif voynich_verbose and not tironian_verbose:
        print("     Voynich is verbose but reference is not.")
    elif tironian_verbose and not voynich_verbose:
        print("     Reference is verbose but Voynich is not.")
    else:
        print("     Neither system shows verbose encoding.")

    # ─── Step 8: Entropy floor explanation ───
    print("\n  8. Entropy floor implication ...")
    entropy_floor_explained = voynich_verbose  # verbose encoding inflates entropy
    if entropy_floor_explained:
        explanation = (
            f"The Voynich glyph-to-triple ratio of {glyphs_per_triple:.2f} "
            f"means multiple visual forms encode the same phonetic value. "
            f"This one-to-many mapping inflates per-character entropy above "
            f"what natural language would show, explaining the elevated "
            f"entropy floor observed in Phase 9.5. "
            f"Even a perfect decoding would retain some excess entropy "
            f"because the cipher alphabet is larger than the plaintext alphabet."
        )
    else:
        explanation = (
            f"The Voynich glyph-to-triple ratio of {glyphs_per_triple:.2f} "
            f"is below the verbose threshold (1.5). "
            f"The entropy floor may have other causes: imperfect decoding, "
            f"additional encoding layers, or non-Latin source language."
        )
    print(f"     {explanation}")

    # Build interpretation string
    if both_verbose:
        interpretation = (
            f"Both the paleographic reference ({ref_stats['variants_per_value']:.2f} "
            f"variants/value) and Voynich ({glyphs_per_triple:.2f} glyphs/triple) "
            f"exhibit verbose encoding. This is a shared structural feature "
            f"consistent with medieval notational systems that use multiple "
            f"sign variants for the same sound value."
        )
    elif voynich_verbose:
        interpretation = (
            f"Voynich shows verbose encoding ({glyphs_per_triple:.2f} glyphs/triple) "
            f"even without strong reference confirmation. The {len(collisions)} "
            f"collision triples each have 2+ distinct EVA glyphs mapping to "
            f"the same phonetic slot."
        )
    elif ref_stats['n_values'] > 0:
        interpretation = (
            f"Neither system shows strong verbose encoding above threshold. "
            f"Reference: {ref_stats['variants_per_value']:.2f} variants/value. "
            f"Voynich: {glyphs_per_triple:.2f} glyphs/triple."
        )
    else:
        interpretation = (
            f"Reference data unavailable. Voynich glyph-to-triple ratio is "
            f"{glyphs_per_triple:.2f} ({len(collisions)} collision triples). "
            f"Cannot fully assess cross-system verbosity without reference comparison."
        )

    # Verdict
    if both_verbose:
        verdict = (
            f"VERBOSE ENCODING CONFIRMED: Both Voynich ({glyphs_per_triple:.2f}x) and "
            f"reference ({ref_stats['variants_per_value']:.2f}x) exceed the 1.5x "
            f"verbosity threshold. This explains the entropy floor from Phase 9.5."
        )
    elif voynich_verbose:
        verdict = (
            f"VOYNICH VERBOSE: {glyphs_per_triple:.2f} glyphs per triple exceeds 1.5x. "
            f"The Voynich uses one-to-many encoding. Reference data "
            f"{'unavailable' if ref_stats['n_values'] == 0 else 'below threshold'}."
        )
    else:
        verdict = (
            f"NOT CLEARLY VERBOSE: Voynich glyphs/triple = {glyphs_per_triple:.2f} "
            f"(threshold 1.5). Entropy floor may require alternative explanations."
        )

    print(f"\n  Verdict: {verdict}")

    # ─── Save ───
    result = VerboseEncodingResult(
        n_tironian_signs=ref_stats['n_signs'],
        n_unique_latin_values=ref_stats['n_values'],
        tironian_variants_per_value=ref_stats['variants_per_value'],
        max_variants=ref_stats['max_variants'],
        max_variants_value=ref_stats['max_variants_value'],
        voynich_glyphs_per_triple=glyphs_per_triple,
        voynich_collisions=collisions,
        both_verbose=both_verbose,
        entropy_floor_explained=entropy_floor_explained,
        interpretation=interpretation,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'verbose_encoding.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)

    print(f"\n  -> {out_path}")
