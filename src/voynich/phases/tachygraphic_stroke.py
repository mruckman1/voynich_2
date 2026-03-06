"""
Phase 19.5 – Tachygraphic Stroke-Rule Test
============================================
Test whether EVA character construction follows the systematic
stroke-modification rules documented in Italian syllabic tachygraphy
(Costamagna) and the Bobbio tradition.

Dependency chain:
    EVA_VISUAL_COMPONENTS (reference.py)
    combined_refine.json  (Phase 15 best assignment)
    corpus
        → tachygraphic_stroke.json
"""

import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    load_corpus,
    tokenize_eva_chars,
)
from voynich.core.reference import EVA_VISUAL_COMPONENTS
from voynich.core.stats import (
    bootstrap_ci,
    coefficient_of_variation,
)


# ---------------------------------------------------------------------------
# JSON serialiser
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
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SignFamily:
    glyph_class: str
    members: List[str]
    n_members: int
    modification_dimension: str  # 'first_stroke', 'last_stroke', 'both', 'none'
    first_stroke_values: List[str]
    last_stroke_values: List[str]
    phonetic_regularity: Dict[str, float]  # consonant_entropy, vowel_entropy
    colless_index: float


@dataclass
class TachygraphicStrokeResult:
    # Sign family analysis
    n_families: int
    n_chars_covered: int
    sign_families: List[Dict[str, Any]]
    # Modification dimension summary
    n_first_stroke_varying: int
    n_last_stroke_varying: int
    n_both_varying: int
    n_none_varying: int
    # Phonetic regularity
    mean_consonant_entropy: float
    mean_vowel_entropy: float
    regularity_ratio: float  # lower = more regular
    # Fontana rotation test
    n_rotational_families: int
    rotational_families: List[str]
    # Colless balance
    mean_colless: float
    reference_colless: float  # from random groupings
    # Family size statistics
    mean_family_size: float
    min_family_size: int
    max_family_size: int
    historical_range: str  # "4-8 members per family"
    # Null test
    null_mean_entropy: float
    null_std_entropy: float
    real_entropy: float
    null_selectivity: float
    # Gate
    gate_passed: bool
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path: str) -> Optional[Dict]:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _build_sign_families() -> Dict[str, List[Dict[str, str]]]:
    """Group EVA characters by glyph_class to form sign families."""
    families: Dict[str, List[Dict[str, str]]] = defaultdict(list)

    for eva_char, components in EVA_VISUAL_COMPONENTS.items():
        gc = components.get('glyph_class', 'unknown')
        families[gc].append({
            'eva_char': eva_char,
            'first_stroke': components.get('first_stroke', ''),
            'last_stroke': components.get('last_stroke', ''),
            'glyph_class': gc,
        })

    return dict(families)


def _identify_modification_dimension(
    members: List[Dict[str, str]],
) -> str:
    """
    Determine which stroke feature varies within a family.
    Returns 'first_stroke', 'last_stroke', 'both', or 'none'.
    """
    if len(members) <= 1:
        return 'none'

    first_strokes = set(m['first_stroke'] for m in members)
    last_strokes = set(m['last_stroke'] for m in members)

    first_varies = len(first_strokes) > 1
    last_varies = len(last_strokes) > 1

    if first_varies and last_varies:
        return 'both'
    elif first_varies:
        return 'first_stroke'
    elif last_varies:
        return 'last_stroke'
    else:
        return 'none'


def _entropy(values: List[str]) -> float:
    """Shannon entropy of a list of categorical values."""
    if not values:
        return 0.0
    counts = Counter(values)
    total = len(values)
    return -sum(
        (c / total) * math.log2(c / total)
        for c in counts.values() if c > 0
    )


def _test_phonetic_regularity(
    family_members: List[Dict[str, str]],
    assignment: Dict[str, str],
) -> Dict[str, float]:
    """
    Within a sign family, compute entropy of consonant and vowel assignments.
    In tachygraphic systems, one dimension should have near-zero entropy.
    """
    consonants = []
    vowels = []

    for member in family_members:
        triple_key = f"{member['first_stroke']},{member['last_stroke']},{member['glyph_class']}"
        syllable = assignment.get(triple_key, '')

        if len(syllable) >= 2:
            consonants.append(syllable[0])
            vowels.append(syllable[1])
        elif len(syllable) == 1:
            # Pure vowel or consonant
            vowels.append(syllable)
            consonants.append('')

    c_entropy = _entropy(consonants) if consonants else 0.0
    v_entropy = _entropy(vowels) if vowels else 0.0

    return {
        'consonant_entropy': round(c_entropy, 4),
        'vowel_entropy': round(v_entropy, 4),
        'min_entropy': round(min(c_entropy, v_entropy), 4),
        'n_consonant_values': len(set(consonants)),
        'n_vowel_values': len(set(vowels)),
    }


def _test_fontana_rotation(
    families: Dict[str, List[Dict[str, str]]],
) -> List[str]:
    """
    Check for families where members are related by rotation/reflection.
    Fontana's cipher generates b/d/p/q from the same base by rotation.
    """
    rotational = []

    # Define stroke types that could be rotational variants
    rotation_pairs = {
        ('vertical', 'ascender'), ('ascender', 'vertical'),
        ('loop', 'open_curve'), ('open_curve', 'loop'),
        ('sigmoid', 'connector'), ('connector', 'sigmoid'),
    }

    for gc, members in families.items():
        if len(members) < 2:
            continue

        # Check if first_stroke values are rotation-related
        first_strokes = [m['first_stroke'] for m in members]
        n_rotation_related = 0
        for i in range(len(first_strokes)):
            for j in range(i + 1, len(first_strokes)):
                pair = (first_strokes[i], first_strokes[j])
                if pair in rotation_pairs:
                    n_rotation_related += 1

        if n_rotation_related >= 1:
            rotational.append(gc)

    return rotational


def _colless_index(sizes: List[int]) -> float:
    """
    Colless imbalance index adapted for multi-child trees.
    For a family with n members, compute imbalance as the CV of member
    frequencies. 0 = perfectly balanced.
    """
    if len(sizes) <= 1:
        return 0.0
    return coefficient_of_variation([float(s) for s in sizes])


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_tachygraphic_stroke() -> None:
    """Phase 19.5: Tachygraphic stroke-rule test."""
    t0 = time.time()
    rd = str(_results_dir())

    print("=" * 60)
    print("Phase 19.5: Tachygraphic Stroke-Rule Test")
    print("=" * 60)

    # ── 1. Build sign families ────────────────────────────────────────
    print("\n  1. Building sign families from EVA_VISUAL_COMPONENTS …")

    families = _build_sign_families()
    n_chars = sum(len(members) for members in families.values())
    print(f"    {len(families)} families covering {n_chars} EVA characters")

    for gc, members in sorted(families.items(), key=lambda x: len(x[1]), reverse=True):
        chars = [m['eva_char'] for m in members]
        print(f"      {gc:12s}: {len(members)} members — {', '.join(chars[:8])}")

    # ── 2. Load phoneme assignment ────────────────────────────────────
    print("\n  2. Loading Phase 15 best phoneme assignment …")

    refine_data = _load_json(os.path.join(rd, 'combined_refine.json'))
    assignment = {}
    if refine_data:
        for key in ['best_assignment', 'assignment', 'latin_assignment', 'best_latin_assignment']:
            if key in refine_data:
                assignment = refine_data[key]
                break
    print(f"    {len(assignment)} triple→syllable mappings loaded")

    # ── 3. Analyze each family ────────────────────────────────────────
    print("\n  3. Analyzing modification dimensions and phonetic regularity …")

    sign_family_results: List[SignFamily] = []
    n_first = n_last = n_both = n_none = 0
    all_min_entropies = []

    corpus = load_corpus(verbose=False)
    tokens = corpus.get_tokens()

    # Get char frequencies for Colless
    char_freq = Counter()
    for tok in tokens:
        for ch in tokenize_eva_chars(tok):
            char_freq[ch] += 1

    for gc, members in sorted(families.items()):
        mod_dim = _identify_modification_dimension(members)
        phonetic = _test_phonetic_regularity(members, assignment)

        first_strokes = sorted(set(m['first_stroke'] for m in members))
        last_strokes = sorted(set(m['last_stroke'] for m in members))

        # Family Colless (based on member frequencies)
        member_freqs = [char_freq.get(m['eva_char'], 0) for m in members]
        colless = _colless_index(member_freqs) if len(member_freqs) > 1 else 0.0

        sf = SignFamily(
            glyph_class=gc,
            members=[m['eva_char'] for m in members],
            n_members=len(members),
            modification_dimension=mod_dim,
            first_stroke_values=first_strokes,
            last_stroke_values=last_strokes,
            phonetic_regularity=phonetic,
            colless_index=round(colless, 4),
        )
        sign_family_results.append(sf)

        if mod_dim == 'first_stroke':
            n_first += 1
        elif mod_dim == 'last_stroke':
            n_last += 1
        elif mod_dim == 'both':
            n_both += 1
        else:
            n_none += 1

        all_min_entropies.append(phonetic['min_entropy'])

        print(f"    {gc:12s}: dim={mod_dim:12s}  C_H={phonetic['consonant_entropy']:.3f}  V_H={phonetic['vowel_entropy']:.3f}  colless={colless:.3f}")

    # ── 4. Aggregate metrics ─────────────────────────────────────────
    print("\n  4. Aggregating metrics …")

    c_entropies = [sf.phonetic_regularity['consonant_entropy'] for sf in sign_family_results if sf.n_members > 1]
    v_entropies = [sf.phonetic_regularity['vowel_entropy'] for sf in sign_family_results if sf.n_members > 1]

    mean_c_h = float(np.mean(c_entropies)) if c_entropies else 0.0
    mean_v_h = float(np.mean(v_entropies)) if v_entropies else 0.0
    regularity_ratio = min(mean_c_h, mean_v_h) / max(mean_c_h, mean_v_h) if max(mean_c_h, mean_v_h) > 0 else 1.0

    family_sizes = [sf.n_members for sf in sign_family_results]
    mean_size = float(np.mean(family_sizes)) if family_sizes else 0
    min_size = min(family_sizes) if family_sizes else 0
    max_size = max(family_sizes) if family_sizes else 0

    mean_colless = float(np.mean([sf.colless_index for sf in sign_family_results])) if sign_family_results else 0

    print(f"    Mean consonant entropy: {mean_c_h:.4f}")
    print(f"    Mean vowel entropy:     {mean_v_h:.4f}")
    print(f"    Regularity ratio:       {regularity_ratio:.4f}")
    print(f"    Mean family size:       {mean_size:.1f} (range {min_size}–{max_size})")
    print(f"    Mean Colless index:     {mean_colless:.4f}")

    # ── 5. Fontana rotation test ─────────────────────────────────────
    print("\n  5. Fontana rotation test …")

    rotational = _test_fontana_rotation(families)
    print(f"    {len(rotational)} families with rotational symmetry: {rotational}")

    # ── 6. Null baseline ─────────────────────────────────────────────
    print("\n  6. Null baseline (100 random family groupings) …")

    rng = random.Random(42)
    all_chars = list(EVA_VISUAL_COMPONENTS.keys())
    real_sizes = [len(m) for m in families.values()]

    null_entropies = []
    for trial in range(100):
        shuffled = list(all_chars)
        rng.shuffle(shuffled)
        # Create families with same sizes as real
        idx = 0
        trial_min_entropies = []
        for size in real_sizes:
            if idx + size > len(shuffled):
                break
            fake_members = []
            for i in range(size):
                ch = shuffled[idx + i]
                comp = EVA_VISUAL_COMPONENTS.get(ch, {})
                fake_members.append({
                    'eva_char': ch,
                    'first_stroke': comp.get('first_stroke', ''),
                    'last_stroke': comp.get('last_stroke', ''),
                    'glyph_class': comp.get('glyph_class', ''),
                })
            idx += size

            phonetic = _test_phonetic_regularity(fake_members, assignment)
            trial_min_entropies.append(phonetic['min_entropy'])

        if trial_min_entropies:
            null_entropies.append(float(np.mean(trial_min_entropies)))

    null_mean = float(np.mean(null_entropies)) if null_entropies else 0.0
    null_std = float(np.std(null_entropies)) if null_entropies else 0.0
    real_mean_min_entropy = float(np.mean(all_min_entropies)) if all_min_entropies else 0.0

    # Selectivity: lower entropy = more tachygraphic
    # Real families should have LOWER entropy than null
    null_sel = null_mean / real_mean_min_entropy if real_mean_min_entropy > 0 else 0.0

    print(f"    Null mean min-entropy: {null_mean:.4f} ± {null_std:.4f}")
    print(f"    Real mean min-entropy: {real_mean_min_entropy:.4f}")
    print(f"    Selectivity: {null_sel:.2f}×")

    # ── 7. Gate ──────────────────────────────────────────────────────
    # Tachygraphic: real families have lower phonetic entropy than null
    gate_passed = bool(null_sel >= 1.5)

    if gate_passed and regularity_ratio < 0.7:
        verdict_label = "TACHYGRAPHIC"
    elif null_sel >= 1.2:
        verdict_label = "PARTIAL"
    else:
        verdict_label = "NON-TACHYGRAPHIC"

    verdict = (
        f"{verdict_label}: selectivity={null_sel:.2f}×, "
        f"regularity={regularity_ratio:.3f}, "
        f"dim: first={n_first}, last={n_last}, both={n_both}, none={n_none}"
    )

    print(f"\n  Gate: {'PASS' if gate_passed else 'FAIL'}")
    print(f"  {verdict}")

    # ── 8. Save ──────────────────────────────────────────────────────
    # Compute reference Colless from null
    ref_colless = float(np.mean(null_entropies)) if null_entropies else 0.0

    result = TachygraphicStrokeResult(
        n_families=len(families),
        n_chars_covered=n_chars,
        sign_families=[_convert(asdict(sf)) for sf in sign_family_results],
        n_first_stroke_varying=n_first,
        n_last_stroke_varying=n_last,
        n_both_varying=n_both,
        n_none_varying=n_none,
        mean_consonant_entropy=round(mean_c_h, 4),
        mean_vowel_entropy=round(mean_v_h, 4),
        regularity_ratio=round(regularity_ratio, 4),
        n_rotational_families=len(rotational),
        rotational_families=rotational,
        mean_colless=round(mean_colless, 4),
        reference_colless=round(ref_colless, 4),
        mean_family_size=round(mean_size, 2),
        min_family_size=min_size,
        max_family_size=max_size,
        historical_range="4-8 members per family (Tironian/Costamagna)",
        null_mean_entropy=round(null_mean, 4),
        null_std_entropy=round(null_std, 4),
        real_entropy=round(real_mean_min_entropy, 4),
        null_selectivity=round(null_sel, 4),
        gate_passed=gate_passed,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'tachygraphic_stroke.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)
    print(f"\n    → {out_path}")
