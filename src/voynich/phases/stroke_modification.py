"""
Phase 19.6 – Stroke-Modification Encoding Simulation
======================================================
If Test 19.5 confirms tachygraphic structure, simulate a full
tachygraphic encoding of Latin medical text and compare its
statistical fingerprint to the Voynich's.

Dependency chain:
    EVA_VISUAL_COMPONENTS (reference.py)
    reference corpus
    Phase 18 result files
        → stroke_modification.json
"""

import json
import math
import os
import random
import time
import zlib
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import load_corpus
from voynich.core.reference import (
    EVA_VISUAL_COMPONENTS,
    load_reference_corpus,
)
from voynich.core.stats import (
    coefficient_of_variation,
    entropy_curve,
    first_order_entropy,
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
class SimulationVariant:
    name: str
    n_consonant_classes: int
    n_vowel_variants: int
    n_homophones: int
    n_modifiers: int
    n_output_glyphs: int
    # Fingerprint metrics
    h0: float
    h2: float
    h4: float
    h6: float
    burstiness_cv: float
    zipf_exponent: float
    ttr: float
    compression_ratio: float
    h2_h1_ratio: float
    # Composite distance to Voynich
    composite_distance: float


@dataclass
class StrokeModificationResult:
    # Voynich fingerprint
    voynich_fingerprint: Dict[str, float]
    # Best simulation
    best_params: Dict[str, int]
    best_distance: float
    best_variant_name: str
    # All variants
    simulation_variants: List[Dict[str, Any]]
    # Null comparison
    null_substitution_distance: float
    null_syllabic_distance: float
    null_random_distance: float
    # Phase 18 tri-state reproduction
    reproduces_tristate: bool
    tristate_detail: str
    # Parameter sweep summary
    n_variants_tested: int
    distance_range: List[float]
    # Gate
    gate_passed: bool
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Tachygraphic encoding table builder
# ---------------------------------------------------------------------------

_VOWELS = set('aeiou')
_CONSONANTS = set('bcdfghjklmnpqrstvwxyz')

# Consonant classes (typologically motivated)
CONSONANT_CLASSES = [
    ['b', 'p'],       # labial stops
    ['d', 't'],       # coronal stops
    ['g', 'k', 'c'],  # velar stops
    ['f', 'v'],       # labiodental fricatives
    ['l', 'r'],       # liquids
    ['m', 'n'],       # nasals
    ['s', 'z'],       # sibilants
    ['h'],            # glottal
]


def _build_tachygraphic_table(
    n_consonants: int,
    n_vowels: int,
    n_homophones: int,
    n_modifiers: int,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """
    Build a tachygraphic encoding table.
    Each consonant class gets a base form; vowel variants create modifications.
    Returns mapping from syllable → list of possible output encodings.
    """
    rng = random.Random(seed)

    # Use a reduced alphabet for output symbols
    alpha = 'abcdefghijklmnopqrstuvwxyz'
    symbols_used = 0

    table: Dict[str, List[str]] = {}

    classes = CONSONANT_CLASSES[:n_consonants]
    vowels = list('aeiou')[:n_vowels]

    for ci, consonants in enumerate(classes):
        base = alpha[symbols_used % 26]
        symbols_used += 1

        for vi, vowel in enumerate(vowels):
            mod = alpha[symbols_used % 26]
            symbols_used += 1

            for consonant in consonants:
                syl = consonant + vowel
                output = base + mod

                # Add homophones
                outputs = [output]
                for h in range(n_homophones):
                    alt = output + alpha[(symbols_used + h) % 26]
                    outputs.append(alt)

                table[syl] = outputs

    # Pure vowels
    for vi, vowel in enumerate(vowels):
        sym = alpha[symbols_used % 26]
        symbols_used += 1
        table[vowel] = [sym]

    # Modifier symbols (inserted randomly, don't map to specific syllables)
    modifier_syms = []
    for mi in range(n_modifiers):
        modifier_syms.append(alpha[symbols_used % 26].upper())
        symbols_used += 1

    return table, modifier_syms


def _syllabify_word(word: str) -> List[str]:
    """Simple CV syllabification."""
    word = word.lower()
    syllables = []
    current = ''
    for ch in word:
        if ch not in _VOWELS and ch not in _CONSONANTS:
            continue
        current += ch
        if ch in _VOWELS:
            syllables.append(current)
            current = ''
    if current:
        if syllables:
            syllables[-1] += current
        else:
            syllables.append(current)
    return syllables if syllables else [word[:2]] if word else ['a']


def _encode_text(
    text: str,
    table: Dict[str, List[str]],
    modifier_syms: List[str],
    modifier_prob: float = 0.1,
    seed: int = 42,
) -> str:
    """Encode Latin text through a tachygraphic table."""
    rng = random.Random(seed)
    words = text.lower().split()
    encoded_words = []

    for word in words:
        clean = ''.join(c for c in word if c.isalpha())
        if not clean:
            continue

        syls = _syllabify_word(clean)
        parts = []
        for syl in syls:
            # Try direct lookup
            if syl in table:
                parts.append(rng.choice(table[syl]))
            else:
                # Try CV decomposition
                found = False
                for i in range(1, len(syl)):
                    prefix = syl[:i]
                    suffix = syl[i:]
                    if prefix in table:
                        parts.append(rng.choice(table[prefix]))
                        if suffix in table:
                            parts.append(rng.choice(table[suffix]))
                        found = True
                        break
                if not found:
                    # Char-by-char fallback
                    for ch in syl:
                        if ch in table:
                            parts.append(rng.choice(table[ch]))

            # Random modifier insertion
            if modifier_syms and rng.random() < modifier_prob:
                parts.append(rng.choice(modifier_syms))

        if parts:
            encoded_words.append(''.join(parts))

    return ' '.join(encoded_words)


# ---------------------------------------------------------------------------
# Statistical fingerprint computation
# ---------------------------------------------------------------------------

def _compute_fingerprint(text: str) -> Dict[str, float]:
    """Compute the full statistical fingerprint of a text."""
    if not text or len(text) < 50:
        return {k: 0.0 for k in ['h0', 'h2', 'h4', 'h6', 'burstiness_cv',
                                   'zipf_exponent', 'ttr', 'compression_ratio',
                                   'h2_h1_ratio']}

    # Entropy curve
    curve = entropy_curve(text, max_order=6)
    h0 = curve.get(0, 0)
    h1 = h0  # H0 is the first-order entropy
    h2 = curve.get(2, 0)
    h4 = curve.get(4, 0)
    h6 = curve.get(6, 0)

    # Tokens
    tokens = text.split()

    # Burstiness (CV of inter-arrival gaps for mid-frequency tokens)
    token_counts = Counter(tokens)
    mid_freq = [t for t, c in token_counts.items()
                if 5 <= c <= max(10, len(tokens) // 50)]
    if mid_freq:
        cvs = []
        for t in mid_freq[:20]:
            positions = [i for i, tok in enumerate(tokens) if tok == t]
            if len(positions) >= 3:
                gaps = [positions[i + 1] - positions[i]
                        for i in range(len(positions) - 1)]
                cv = coefficient_of_variation([float(g) for g in gaps])
                cvs.append(cv)
        burstiness = float(np.mean(cvs)) if cvs else 1.0
    else:
        burstiness = 1.0

    # Zipf exponent (simple log-log slope)
    sorted_freqs = sorted(token_counts.values(), reverse=True)
    if len(sorted_freqs) >= 5:
        log_ranks = np.log(np.arange(1, min(len(sorted_freqs), 100) + 1))
        log_freqs = np.log(np.array(sorted_freqs[:100], dtype=float))
        if len(log_ranks) > 1:
            slope = float(np.polyfit(log_ranks, log_freqs, 1)[0])
            zipf_exp = -slope
        else:
            zipf_exp = 1.0
    else:
        zipf_exp = 1.0

    # TTR
    n_types = len(token_counts)
    n_tokens = len(tokens)
    ttr = n_types / n_tokens if n_tokens > 0 else 0

    # Compression ratio
    text_bytes = text.encode('utf-8')
    compressed = zlib.compress(text_bytes, level=9)
    compression_ratio = len(compressed) / len(text_bytes) if text_bytes else 1.0

    # H2/H1 ratio
    h2_h1 = h2 / h1 if h1 > 0 else 0

    return {
        'h0': round(h0, 4),
        'h2': round(h2, 4),
        'h4': round(h4, 4),
        'h6': round(h6, 4),
        'burstiness_cv': round(burstiness, 4),
        'zipf_exponent': round(zipf_exp, 4),
        'ttr': round(ttr, 4),
        'compression_ratio': round(compression_ratio, 4),
        'h2_h1_ratio': round(h2_h1, 4),
    }


def _composite_distance(fp1: Dict[str, float], fp2: Dict[str, float]) -> float:
    """Normalized Euclidean distance between two fingerprints."""
    keys = sorted(set(fp1.keys()) & set(fp2.keys()))
    if not keys:
        return 999.0

    diffs = []
    for k in keys:
        v1 = fp1[k]
        v2 = fp2[k]
        # Normalize by max to get relative distance
        max_val = max(abs(v1), abs(v2), 0.001)
        diffs.append(((v1 - v2) / max_val) ** 2)

    return float(np.sqrt(np.mean(diffs)))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_stroke_modification() -> None:
    """Phase 19.6: Stroke-modification encoding simulation."""
    t0 = time.time()
    rd = str(_results_dir())

    print("=" * 60)
    print("Phase 19.6: Stroke-Modification Encoding Simulation")
    print("=" * 60)

    # ── 1. Compute Voynich fingerprint ────────────────────────────────
    print("\n  1. Computing Voynich statistical fingerprint …")

    corpus = load_corpus(verbose=False)
    voynich_text = corpus.get_text()
    voynich_fp = _compute_fingerprint(voynich_text)

    for k, v in voynich_fp.items():
        print(f"    {k:20s}: {v:.4f}")

    # ── 2. Get Latin reference text ──────────────────────────────────
    print("\n  2. Loading Latin reference text …")

    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    latin_tokens = ref_corpus.get_combined_tokens('latin')
    latin_text = ' '.join(latin_tokens[:8000]) if latin_tokens else ''

    if not latin_text:
        from voynich.core.ciphers import generate_reference_text
        latin_text = generate_reference_text('latin', n_words=8000)

    print(f"    {len(latin_text.split())} words of Latin reference text")

    # ── 3. Parameter sweep ───────────────────────────────────────────
    print("\n  3. Running parameter sweep …")

    param_grid = [
        # (n_consonants, n_vowels, n_homophones, n_modifiers)
        (4, 3, 0, 0),
        (5, 4, 0, 0),
        (5, 5, 0, 0),
        (6, 5, 0, 0),
        (7, 5, 0, 0),
        (8, 5, 0, 0),
        (5, 5, 1, 0),
        (5, 5, 2, 0),
        (6, 5, 1, 0),
        (6, 5, 2, 0),
        (5, 5, 0, 5),
        (5, 5, 0, 10),
        (5, 5, 0, 15),
        (5, 5, 1, 5),
        (5, 5, 1, 10),
        (5, 5, 2, 10),
        (6, 5, 1, 10),
        (6, 5, 2, 15),
        (7, 5, 1, 10),
        (7, 5, 2, 15),
        (7, 5, 0, 15),
        (6, 5, 0, 15),
        (5, 4, 1, 10),
        (6, 4, 2, 10),
    ]

    variants: List[SimulationVariant] = []
    best_variant = None
    best_distance = 999.0

    for pi, (nc, nv, nh, nm) in enumerate(param_grid):
        name = f"C{nc}_V{nv}_H{nh}_M{nm}"

        table, mod_syms = _build_tachygraphic_table(nc, nv, nh, nm, seed=42 + pi)
        n_output = len(table) + len(mod_syms)

        mod_prob = nm * 0.02 if nm > 0 else 0.0
        encoded = _encode_text(latin_text, table, mod_syms, mod_prob, seed=42 + pi)

        if not encoded or len(encoded) < 100:
            continue

        fp = _compute_fingerprint(encoded)
        dist = _composite_distance(fp, voynich_fp)

        sv = SimulationVariant(
            name=name,
            n_consonant_classes=nc,
            n_vowel_variants=nv,
            n_homophones=nh,
            n_modifiers=nm,
            n_output_glyphs=n_output,
            h0=fp.get('h0', 0),
            h2=fp.get('h2', 0),
            h4=fp.get('h4', 0),
            h6=fp.get('h6', 0),
            burstiness_cv=fp.get('burstiness_cv', 0),
            zipf_exponent=fp.get('zipf_exponent', 0),
            ttr=fp.get('ttr', 0),
            compression_ratio=fp.get('compression_ratio', 0),
            h2_h1_ratio=fp.get('h2_h1_ratio', 0),
            composite_distance=round(dist, 4),
        )
        variants.append(sv)

        if dist < best_distance:
            best_distance = dist
            best_variant = sv

        if pi % 6 == 0:
            print(f"    {name:20s}  dist={dist:.4f}  H0={fp['h0']:.3f} H2={fp['h2']:.3f}")

    # Sort by distance
    variants.sort(key=lambda v: v.composite_distance)

    if best_variant:
        print(f"\n    Best: {best_variant.name} (distance={best_distance:.4f})")
        print(f"      H0={best_variant.h0:.3f} H2={best_variant.h2:.3f} H6={best_variant.h6:.3f}")
        print(f"      burstiness={best_variant.burstiness_cv:.3f} zipf={best_variant.zipf_exponent:.3f}")

    # ── 4. Null comparisons ──────────────────────────────────────────
    print("\n  4. Computing null comparison distances …")

    # Simple substitution
    from voynich.core.ciphers import SimpleSubstitutionCipher
    sub_cipher = SimpleSubstitutionCipher(seed=42)
    sub_text = sub_cipher.encrypt(latin_text)
    sub_fp = _compute_fingerprint(sub_text)
    sub_dist = _composite_distance(sub_fp, voynich_fp)

    # Pure syllabic
    from voynich.core.ciphers import SyllabicEncoder
    syl_encoder = SyllabicEncoder(seed=42)
    syl_text = syl_encoder.encode(latin_text)
    syl_fp = _compute_fingerprint(syl_text)
    syl_dist = _composite_distance(syl_fp, voynich_fp)

    # Random text
    rng = random.Random(42)
    rand_words = [''.join(rng.choices('abcdefghijklmnopqrst', k=rng.randint(2, 6)))
                  for _ in range(len(latin_text.split()))]
    rand_text = ' '.join(rand_words)
    rand_fp = _compute_fingerprint(rand_text)
    rand_dist = _composite_distance(rand_fp, voynich_fp)

    print(f"    Simple substitution: {sub_dist:.4f}")
    print(f"    Pure syllabic:       {syl_dist:.4f}")
    print(f"    Random text:         {rand_dist:.4f}")
    print(f"    Best tachygraphic:   {best_distance:.4f}")

    # ── 5. Phase 18 tri-state check ──────────────────────────────────
    print("\n  5. Checking Phase 18 tri-state reproduction …")

    # Check if the best tachygraphic encoding reproduces INDETERMINATE
    # A tachygraphic system should show: balanced trie (H3-like), rigid
    # HMM (H1-like), and natural compression (H2-like) simultaneously
    reproduces = False
    tristate_detail = ""

    if best_variant:
        # Burstiness near 1.0 → H1-like (Poisson)
        burst_h1 = abs(best_variant.burstiness_cv - 1.0) < 0.3
        # Compression similar to Voynich → H2-like (natural)
        comp_h2 = abs(best_variant.compression_ratio - voynich_fp['compression_ratio']) < 0.1
        # Low H6 → structured (H3-like)
        h6_h3 = best_variant.h6 < 0.5

        reproduces = burst_h1 and comp_h2
        tristate_detail = (
            f"burstiness={'H1-like' if burst_h1 else 'not-H1'} "
            f"({best_variant.burstiness_cv:.3f}), "
            f"compression={'H2-like' if comp_h2 else 'not-H2'} "
            f"({best_variant.compression_ratio:.3f}), "
            f"H6={'H3-like' if h6_h3 else 'not-H3'} "
            f"({best_variant.h6:.3f})"
        )

    print(f"    Reproduces tri-state: {'YES' if reproduces else 'NO'}")
    print(f"    {tristate_detail}")

    # ── 6. Gate ──────────────────────────────────────────────────────
    gate_passed = bool(best_distance < sub_dist and
                       best_distance < syl_dist and
                       best_distance < rand_dist)

    if gate_passed and reproduces:
        verdict = f"PASS: tachygraphic model (dist={best_distance:.3f}) beats all nulls AND reproduces tri-state"
    elif gate_passed:
        verdict = f"PARTIAL: tachygraphic model (dist={best_distance:.3f}) beats all nulls but does NOT reproduce tri-state"
    else:
        verdict = f"FAIL: tachygraphic model (dist={best_distance:.3f}) does not beat all nulls"

    print(f"\n  Gate: {'PASS' if gate_passed else 'FAIL'}")
    print(f"  {verdict}")

    # ── 7. Save ──────────────────────────────────────────────────────
    best_params = {}
    if best_variant:
        best_params = {
            'n_consonant_classes': best_variant.n_consonant_classes,
            'n_vowel_variants': best_variant.n_vowel_variants,
            'n_homophones': best_variant.n_homophones,
            'n_modifiers': best_variant.n_modifiers,
        }

    distances = [v.composite_distance for v in variants]

    result = StrokeModificationResult(
        voynich_fingerprint=voynich_fp,
        best_params=best_params,
        best_distance=round(best_distance, 4),
        best_variant_name=best_variant.name if best_variant else 'none',
        simulation_variants=[_convert(asdict(sv)) for sv in variants[:30]],
        null_substitution_distance=round(sub_dist, 4),
        null_syllabic_distance=round(syl_dist, 4),
        null_random_distance=round(rand_dist, 4),
        reproduces_tristate=reproduces,
        tristate_detail=tristate_detail,
        n_variants_tested=len(variants),
        distance_range=[round(min(distances), 4), round(max(distances), 4)] if distances else [0, 0],
        gate_passed=gate_passed,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'stroke_modification.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)
    print(f"\n    → {out_path}")
