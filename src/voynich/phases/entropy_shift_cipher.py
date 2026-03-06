"""
Phase 19.2 – Cipher Mechanism Entropy Shift Identification
===========================================================
The Voynich Language A entropy curve runs parallel to Latin at r≈0.999
with a specific upward shift.  Different cipher mechanisms produce
different shift profiles.  This test identifies which historical cipher
type reproduces the observed shift.

Dependency chain:
    corpus
    reference corpus
    ciphers.py encoders
        → entropy_shift_cipher.json
"""

import json
import math
import os
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from voynich.core._paths import results_dir as _results_dir
from voynich.core.ciphers import (
    AbbreviationEncoder,
    HomophonicCipher,
    NomenclatorCipher,
    NullInsertionEncoder,
    SimpleSubstitutionCipher,
    SyllabicEncoder,
    VigenereCipher,
    generate_reference_text,
)
from voynich.core.corpus import load_corpus
from voynich.core.reference import load_reference_corpus
from voynich.core.stats import (
    cosine_similarity,
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
class MechanismProfile:
    name: str
    n_instantiations: int
    mean_shift_vector: List[float]
    std_shift_vector: List[float]
    cosine_similarity: float
    euclidean_distance: float
    ci_lower: float
    ci_upper: float


@dataclass
class EntropyShiftResult:
    # Voynich profile
    voynich_entropy_curve: Dict[str, float]
    # Latin profile
    latin_entropy_curve: Dict[str, float]
    # Observed shift
    observed_shift_vector: List[float]
    # Per-mechanism results
    mechanism_profiles: List[Dict[str, Any]]
    # Ranking
    cipher_ranking: List[Dict[str, Any]]
    best_match_cipher: str
    best_match_cosine: float
    best_match_euclidean: float
    second_best_cipher: str
    second_best_cosine: float
    # Discrimination
    top2_ci_overlap: bool
    discrimination_verdict: str
    # Null comparison
    null_shift_cosine: float
    # Gate
    gate_passed: bool
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Custom cipher mechanisms (not in ciphers.py)
# ---------------------------------------------------------------------------

_VOWELS = set('aeiou')
_CONSONANTS = set('bcdfghjklmnpqrstvwxyz')


def _syllabify_simple(word: str) -> List[str]:
    """Split a word into CV-ish syllables."""
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
    return syllables if syllables else [word]


class SyllabicModifierEncoder:
    """
    Syllabic base where modifier characters alter adjacent syllable values.
    Models the Phase 16 hypothesis: some characters modify rather than
    produce independent syllables.
    """

    def __init__(self, n_modifiers: int = 5, seed: int = 42):
        self.rng = random.Random(seed)
        self.syllable_map: Dict[str, str] = {}
        self._next_id = 0
        self.n_modifiers = n_modifiers
        self.name = 'syllabic_modifier'

        # Build modifier symbols
        mod_alpha = 'ABCDEFGHIJ'
        self.modifier_symbols = [mod_alpha[i] for i in range(n_modifiers)]

        # Modifier insertion probability
        self.mod_prob = 0.15

    def _get_symbol(self, syllable: str) -> str:
        if syllable not in self.syllable_map:
            alpha = 'abcdefghijklmnopqrst'
            idx = self._next_id
            s1 = alpha[idx % len(alpha)]
            s2 = alpha[(idx // len(alpha)) % len(alpha)]
            self.syllable_map[syllable] = s1 + s2
            self._next_id += 1
        return self.syllable_map[syllable]

    def encode(self, plaintext: str) -> str:
        words = plaintext.lower().split()
        encoded = []
        for word in words:
            syls = _syllabify_simple(word)
            parts = []
            for syl in syls:
                parts.append(self._get_symbol(syl))
                # Randomly insert modifier
                if self.rng.random() < self.mod_prob:
                    parts.append(self.rng.choice(self.modifier_symbols))
            encoded.append(''.join(parts))
        return ' '.join(encoded)


class TachygraphicEncoder:
    """
    Syllabic base where signs are constructed from stroke components
    with systematic modification rules — the Costamagna model.
    Base forms represent consonant classes; modifications encode vowels.
    """

    def __init__(self, n_bases: int = 7, n_modifications: int = 5, seed: int = 42):
        self.rng = random.Random(seed)
        self.n_bases = n_bases
        self.n_modifications = n_modifications
        self.name = 'tachygraphic'

        # Build systematic encoding: consonant→base, vowel→modification
        consonant_classes = [
            ['b', 'p'], ['d', 't'], ['g', 'k', 'c'],
            ['f', 'v'], ['l', 'r'], ['m', 'n'],
            ['s', 'z'],
        ][:n_bases]

        vowel_mods = ['a', 'e', 'i', 'o', 'u'][:n_modifications]

        # Build encoding table: syllable → encoded form
        self.table: Dict[str, str] = {}
        alpha = 'abcdefghijklmnopqrstuvwxyz'
        for bi, consonants in enumerate(consonant_classes):
            base_char = alpha[bi * 2]  # Base form
            for vi, vowel in enumerate(vowel_mods):
                mod_char = alpha[bi * 2 + 1] if vi % 2 == 0 else alpha[20 + vi % 6] if 20 + vi % 6 < 26 else alpha[vi % 20]
                for consonant in consonants:
                    syl = consonant + vowel
                    self.table[syl] = base_char + mod_char

        # Pure vowels
        for vi, vowel in enumerate(vowel_mods):
            self.table[vowel] = alpha[14 + vi] if 14 + vi < 26 else alpha[vi]

    def encode(self, plaintext: str) -> str:
        words = plaintext.lower().split()
        encoded = []
        for word in words:
            syls = _syllabify_simple(word)
            parts = []
            for syl in syls:
                if syl in self.table:
                    parts.append(self.table[syl])
                else:
                    # Fallback: encode char by char
                    for ch in syl:
                        if ch + 'a' in self.table:
                            parts.append(self.table[ch + 'a'][:1])
                        elif ch in self.table:
                            parts.append(self.table[ch])
                        else:
                            parts.append(ch)
            encoded.append(''.join(parts))
        return ' '.join(encoded)


# ---------------------------------------------------------------------------
# Cipher mechanism registry
# ---------------------------------------------------------------------------

def _build_cipher_instances(seed: int) -> List:
    """Build one instance of each cipher mechanism with the given seed."""
    return [
        SimpleSubstitutionCipher(seed=seed),
        VigenereCipher(key_length=5, seed=seed),
        HomophonicCipher(seed=seed),
        NomenclatorCipher(seed=seed),
        SyllabicEncoder(seed=seed),
        NullInsertionEncoder(seed=seed),
        AbbreviationEncoder(mode='heavy', seed=seed),
        SyllabicModifierEncoder(n_modifiers=5, seed=seed),
        TachygraphicEncoder(n_bases=7, n_modifications=5, seed=seed),
    ]


def _encode_and_curve(cipher, plaintext: str, max_order: int) -> Dict[int, float]:
    """Encode plaintext with cipher and compute entropy curve."""
    if hasattr(cipher, 'encrypt'):
        encoded = cipher.encrypt(plaintext)
    elif hasattr(cipher, 'encode'):
        encoded = cipher.encode(plaintext)
    else:
        return {}

    if not encoded or len(encoded) < 50:
        return {}

    return entropy_curve(encoded, max_order=max_order)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_entropy_shift() -> None:
    """Phase 19.2: Cipher mechanism entropy shift identification."""
    t0 = time.time()
    rd = str(_results_dir())
    max_order = 6

    print("=" * 60)
    print("Phase 19.2: Cipher Mechanism Entropy Shift Identification")
    print("=" * 60)

    # ── 1. Compute Voynich entropy curve ─────────────────────────────
    print("\n  1. Computing Voynich entropy curve …")

    corpus = load_corpus(verbose=False)
    voynich_text = corpus.get_text()
    voynich_curve = entropy_curve(voynich_text, max_order=max_order)

    print(f"    Voynich H0={voynich_curve.get(0, 0):.3f}, "
          f"H2={voynich_curve.get(2, 0):.3f}, "
          f"H4={voynich_curve.get(4, 0):.3f}, "
          f"H6={voynich_curve.get(6, 0):.3f}")

    # ── 2. Compute Latin reference entropy curve ─────────────────────
    print("\n  2. Computing Latin reference entropy curve …")

    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    latin_tokens = ref_corpus.get_combined_tokens('latin')
    latin_text = ' '.join(latin_tokens[:5000]) if latin_tokens else generate_reference_text('latin', n_words=5000)

    latin_curve = entropy_curve(latin_text, max_order=max_order)

    print(f"    Latin  H0={latin_curve.get(0, 0):.3f}, "
          f"H2={latin_curve.get(2, 0):.3f}, "
          f"H4={latin_curve.get(4, 0):.3f}, "
          f"H6={latin_curve.get(6, 0):.3f}")

    # ── 3. Compute observed shift vector ─────────────────────────────
    print("\n  3. Computing shift vector …")

    orders = list(range(max_order + 1))
    observed_shift = np.array([
        voynich_curve.get(k, 0) - latin_curve.get(k, 0)
        for k in orders
    ])

    print(f"    Shift: {[f'{s:.3f}' for s in observed_shift]}")

    # ── 4. Apply each mechanism to Latin ─────────────────────────────
    print("\n  4. Simulating 9 cipher mechanisms (20 instantiations each) …")

    n_instantiations = 20
    mechanism_names = [
        'simple_substitution', 'polyalphabetic', 'homophonic',
        'nomenclator', 'syllabic', 'null_insertion',
        'abbreviation_heavy', 'syllabic_modifier', 'tachygraphic',
    ]

    all_profiles: List[MechanismProfile] = []

    for mech_idx, mech_name in enumerate(mechanism_names):
        print(f"\n    ── {mech_name} ──")

        shift_vectors = []
        for inst in range(n_instantiations):
            seed = 1000 + mech_idx * 100 + inst
            ciphers = _build_cipher_instances(seed)
            cipher = ciphers[mech_idx]

            mech_curve = _encode_and_curve(cipher, latin_text, max_order)
            if not mech_curve:
                continue

            shift = np.array([
                mech_curve.get(k, 0) - latin_curve.get(k, 0)
                for k in orders
            ])
            shift_vectors.append(shift)

        if not shift_vectors:
            print(f"      [SKIP] No valid encodings")
            continue

        shifts = np.array(shift_vectors)
        mean_shift = shifts.mean(axis=0)
        std_shift = shifts.std(axis=0)

        # Compare to observed Voynich shift
        cos_sim = float(cosine_similarity(mean_shift, observed_shift))
        euc_dist = float(np.linalg.norm(mean_shift - observed_shift))

        # Bootstrap CI on cosine similarity
        cos_sims = [
            float(cosine_similarity(sv, observed_shift))
            for sv in shift_vectors
        ]
        ci_lower = float(np.percentile(cos_sims, 2.5))
        ci_upper = float(np.percentile(cos_sims, 97.5))

        profile = MechanismProfile(
            name=mech_name,
            n_instantiations=len(shift_vectors),
            mean_shift_vector=[round(float(v), 4) for v in mean_shift],
            std_shift_vector=[round(float(v), 4) for v in std_shift],
            cosine_similarity=round(cos_sim, 4),
            euclidean_distance=round(euc_dist, 4),
            ci_lower=round(ci_lower, 4),
            ci_upper=round(ci_upper, 4),
        )
        all_profiles.append(profile)

        print(f"      Cosine sim: {cos_sim:.4f} [{ci_lower:.3f}, {ci_upper:.3f}]")
        print(f"      Euclidean dist: {euc_dist:.4f}")

    # ── 5. Rank mechanisms ───────────────────────────────────────────
    print("\n  5. Ranking mechanisms by cosine similarity …")

    ranking = sorted(all_profiles, key=lambda p: p.cosine_similarity, reverse=True)
    cipher_ranking = []
    for i, prof in enumerate(ranking):
        cipher_ranking.append({
            'rank': i + 1,
            'name': prof.name,
            'cosine_similarity': prof.cosine_similarity,
            'euclidean_distance': prof.euclidean_distance,
        })
        print(f"    {i + 1}. {prof.name:25s} cos={prof.cosine_similarity:.4f}  euc={prof.euclidean_distance:.4f}")

    best = ranking[0] if ranking else None
    second = ranking[1] if len(ranking) > 1 else None

    best_name = best.name if best else 'none'
    best_cos = best.cosine_similarity if best else 0.0
    best_euc = best.euclidean_distance if best else 999.0
    second_name = second.name if second else 'none'
    second_cos = second.cosine_similarity if second else 0.0

    # ── 6. Discrimination test ───────────────────────────────────────
    print("\n  6. Discrimination test …")

    if best and second:
        ci_overlap = best.ci_lower <= second.ci_upper and second.ci_lower <= best.ci_upper
        if not ci_overlap:
            disc_verdict = f"DISCRIMINATED: {best_name} clearly beats {second_name}"
        else:
            disc_verdict = f"DEGENERATE: {best_name} and {second_name} CIs overlap"
    else:
        ci_overlap = True
        disc_verdict = "INSUFFICIENT: fewer than 2 mechanisms tested"

    print(f"    {disc_verdict}")

    # ── 7. Null control ──────────────────────────────────────────────
    print("\n  7. Null control (shuffled Voynich) …")

    rng = random.Random(42)
    shuffled_chars = list(voynich_text)
    rng.shuffle(shuffled_chars)
    shuffled_text = ''.join(shuffled_chars)
    shuffled_curve = entropy_curve(shuffled_text, max_order=max_order)
    null_shift = np.array([
        shuffled_curve.get(k, 0) - latin_curve.get(k, 0)
        for k in orders
    ])
    null_cos = float(cosine_similarity(null_shift, observed_shift))
    print(f"    Null cosine: {null_cos:.4f} (vs best: {best_cos:.4f})")

    # ── 8. Gate ──────────────────────────────────────────────────────
    gate_passed = bool(best_cos > 0.8 and best_cos > null_cos + 0.1)

    if gate_passed:
        verdict = f"PASS: {best_name} matches Voynich shift (cos={best_cos:.3f})"
    elif best_cos > 0.5:
        verdict = f"PARTIAL: {best_name} moderate match (cos={best_cos:.3f})"
    else:
        verdict = f"FAIL: no mechanism closely reproduces Voynich shift"

    print(f"\n  Gate: {'PASS' if gate_passed else 'FAIL'}")
    print(f"  {verdict}")

    # ── 9. Save ──────────────────────────────────────────────────────
    result = EntropyShiftResult(
        voynich_entropy_curve={str(k): round(v, 4) for k, v in voynich_curve.items()},
        latin_entropy_curve={str(k): round(v, 4) for k, v in latin_curve.items()},
        observed_shift_vector=[round(float(v), 4) for v in observed_shift],
        mechanism_profiles=[_convert(asdict(p)) for p in all_profiles],
        cipher_ranking=cipher_ranking,
        best_match_cipher=best_name,
        best_match_cosine=round(best_cos, 4),
        best_match_euclidean=round(best_euc, 4),
        second_best_cipher=second_name,
        second_best_cosine=round(second_cos, 4),
        top2_ci_overlap=ci_overlap,
        discrimination_verdict=disc_verdict,
        null_shift_cosine=round(null_cos, 4),
        gate_passed=gate_passed,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'entropy_shift_cipher.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)
    print(f"\n    → {out_path}")
