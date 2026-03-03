"""
Phase 14.4 – Calibration on Synthetic Feature-Based Abugida
=============================================================
Validates the feature-level CSP on a known encoding before running on the
real Voynich data.

Method
------
1. Build a known ``true_feature_mapping``: triple_key -> CV syllable (one per
   attested triple, chosen to resemble a plausible Latin phonemic mapping).
2. Encode a Latin reference text by: syllabify each word -> look up which
   triple maps to that syllable -> pick the representative EVA glyph for
   that triple.
3. Run the Phase 14.3 feature CSP on the encoded tokens.
4. Measure recovery accuracy: what fraction of triple assignments did the
   CSP recover correctly?
5. Add noise (randomly substitute 20% of glyphs) and re-run to test
   robustness.

This calibrates the dict_hit expectation: if the CSP recovers the correct
mapping on clean synthetic data, we can estimate how much noise discounts
real performance.

Dependency chain:
    stroke_features.json  (attested triple list)
        → feature_calibrate.json  (this step)
"""

import json
import os
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    load_corpus,
    tokenize_eva_chars,
)
from voynich.core.reference import (
    EVA_VISUAL_COMPONENTS,
    build_cv_syllable_table,
    build_triple_phoneme_hypotheses,
    load_reference_corpus,
)
from voynich.core.stats import build_ngram_lm, syllabify_latin
from voynich.phases.csp_constraints import (
    AnchorConstraint,
    build_phoneme_inventory,
)
from voynich.phases.csp_solver import (
    _convert,
    decode_corpus,
    decode_token,
)
from voynich.phases.feature_csp import (
    FeatureVariable,
    build_feature_variables,
    initialise_feature_domains,
    run_feature_csp_for_language,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_true_mapping(
    variables: List[FeatureVariable],
    target_syllables: List[str],
) -> Dict[str, str]:
    """Assign one target syllable per FeatureVariable, by frequency rank.

    The highest-frequency triple gets the most common syllable, etc.
    This mimics a plausible encoding where common phonemes appear in
    common positions.
    """
    sorted_vars = sorted(variables, key=lambda v: v.frequency, reverse=True)
    mapping: Dict[str, str] = {}
    for i, var in enumerate(sorted_vars):
        syl_idx = i % len(target_syllables)
        mapping[var.cell_key] = target_syllables[syl_idx]
    return mapping


def _encode_latin_with_feature_mapping(
    latin_tokens: List[str],
    true_mapping: Dict[str, str],
    eva_to_triple: Dict[str, str],
    triple_to_glyphs: Dict[str, List[str]],
    max_tokens: int = 3000,
) -> Tuple[List[str], int]:
    """Encode Latin words as synthetic EVA tokens using the true feature mapping.

    For each syllable in a Latin word, find which triple maps to that syllable
    and pick the representative (first) glyph.  Concatenates glyphs for
    each word to produce a synthetic EVA token.

    Returns (encoded_tokens, n_encoded).
    """
    # Invert mapping: syllable -> list of triple_keys that map to it
    syl_to_triples: Dict[str, List[str]] = {}
    for triple_key, syl in true_mapping.items():
        if syl not in syl_to_triples:
            syl_to_triples[syl] = []
        syl_to_triples[syl].append(triple_key)

    # Build syllable -> representative EVA glyph lookup
    syl_to_glyph: Dict[str, str] = {}
    for syl, triples in syl_to_triples.items():
        # Pick the first triple (highest-frequency order)
        for triple_key in triples:
            glyphs = triple_to_glyphs.get(triple_key, [])
            if glyphs:
                syl_to_glyph[syl] = glyphs[0]
                break

    all_known_syls = set(true_mapping.values())
    vowels = set('aeiou')
    encoded: List[str] = []

    for word in latin_tokens[:max_tokens]:
        syls = syllabify_latin(word)
        if not syls:
            continue
        glyph_seq: List[str] = []
        for syl in syls:
            syl_lower = syl.lower()
            # Try to find the syllable directly
            if syl_lower in syl_to_glyph:
                glyph_seq.append(syl_to_glyph[syl_lower])
                continue
            # Extract CV pattern and try that
            onset = ''
            rest = syl_lower
            while rest and rest[0] not in vowels:
                onset += rest[0]
                rest = rest[1:]
            nucleus = rest[0] if rest else 'a'
            cv = (onset[-1] if onset else '') + nucleus
            if cv in syl_to_glyph:
                glyph_seq.append(syl_to_glyph[cv])
            elif nucleus in syl_to_glyph:
                glyph_seq.append(syl_to_glyph[nucleus])
        if glyph_seq:
            encoded.append(''.join(glyph_seq))

    return encoded, len(encoded)


def _add_noise(
    tokens: List[str],
    all_glyphs: List[str],
    noise_rate: float = 0.2,
    seed: int = 77,
) -> List[str]:
    """Randomly substitute ~noise_rate fraction of EVA characters."""
    rng = random.Random(seed)
    noisy: List[str] = []
    for token in tokens:
        chars = list(token)  # raw char-level substitution
        new_chars: List[str] = []
        for ch in chars:
            if rng.random() < noise_rate and all_glyphs:
                new_chars.append(rng.choice(all_glyphs))
            else:
                new_chars.append(ch)
        noisy.append(''.join(new_chars))
    return noisy


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CalibrationResult:
    """Results of synthetic abugida calibration run."""
    n_triples: int
    n_tokens_encoded: int
    n_tokens_noisy: int
    true_mapping: Dict[str, str]
    recovered_mapping: Dict[str, str]      # from noise-free CSP
    recovery_accuracy: float               # fraction of triples correctly recovered
    noise_free_dict_hit: float
    noisy_dict_hit: float
    robustness_ratio: float                # noisy / noise_free
    clean_selectivity: float
    noisy_selectivity: float
    expected_voynich_ceiling: float        # calibrated expectation
    gate_passed: bool                      # recovery_accuracy >= 0.5 AND selectivity >= 1.5
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_feature_calibrate() -> None:
    """Step 14.4: synthetic abugida calibration of the feature CSP."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE 14.4: Feature CSP Calibration on Synthetic Abugida")
    print("=" * 70)

    rd = _results_dir()

    # Load reference corpora (Latin only for calibration)
    ref_corpus = load_reference_corpus(verbose=False)
    latin_tokens = ref_corpus.get_combined_tokens('latin')
    if not latin_tokens:
        print("  [SKIP] No Latin reference corpus available")
        return

    print(f"\n  Latin reference tokens: {len(latin_tokens)}")

    # Build triple lookup and get all attested triples
    eva_to_triple = build_eva_to_triple_lookup()

    # Collect triple -> glyphs mapping
    triple_to_glyphs: Dict[str, List[str]] = {}
    for glyph, triple_key in eva_to_triple.items():
        if triple_key not in triple_to_glyphs:
            triple_to_glyphs[triple_key] = []
        triple_to_glyphs[triple_key].append(glyph)

    # Build inventory for Latin
    inventory = build_phoneme_inventory('latin', ref_corpus)
    lm = build_ngram_lm(latin_tokens[:5000], order=3, smoothing=0.01)
    ref_word_set = set(w.lower() for w in latin_tokens if len(w) >= 2)

    # Latin CV syllables: the first len(triples) most common ones
    lat_syls = build_cv_syllable_table('latin')
    known_cv = [
        'ra', 'te', 'cu', 'na', 'li', 'me', 'tu', 'si',
        'sa', 'ni', 'de', 'pa', 'bo', 'vi', 'ca', 'ro',
        'su', 'mi', 're', 'ta', 'lo', 'ne', 'fi', 'pi',
    ]

    # Build feature variables (no glyph freq from Voynich — use uniform)
    uniform_freq: Counter = Counter({glyph: 1 for glyph in eva_to_triple.keys()})
    hypothesis_map = build_triple_phoneme_hypotheses('latin', lat_syls)
    variables = build_feature_variables(
        eva_to_triple, uniform_freq, inventory, hypothesis_map
    )
    variables = initialise_feature_domains(
        variables, inventory, hypothesis_map, anchors=[]
    )

    print(f"\n  Feature variables: {len(variables)}")

    # Build true mapping: triple_key -> syllable (frequency-rank order)
    true_mapping = _build_true_mapping(variables, known_cv)

    print(f"\n  True mapping ({len(true_mapping)} triples):")
    for triple_key, syl in sorted(true_mapping.items(), key=lambda x: x[1]):
        glyphs = ', '.join(triple_to_glyphs.get(triple_key, ['?'])[:3])
        print(f"    {triple_key:<40} -> {syl}  (glyphs: {glyphs})")

    # Encode Latin text using the true mapping
    encoded_tokens, n_encoded = _encode_latin_with_feature_mapping(
        latin_tokens, true_mapping, eva_to_triple, triple_to_glyphs,
        max_tokens=3000,
    )
    print(f"\n  Encoded {n_encoded} tokens from Latin corpus")
    if encoded_tokens:
        print(f"  Sample tokens: {encoded_tokens[:6]}")

    # Verify round-trip
    print("\n  Round-trip verification (first 5):")
    for tok in encoded_tokens[:5]:
        decoded = decode_token(tok, true_mapping, eva_to_triple)
        print(f"    {tok:20s} -> {decoded}")

    # Run noise-free CSP
    print("\n  Running noise-free feature CSP...")
    clean_result = run_feature_csp_for_language(
        language='latin',
        variables=variables,
        lm=lm,
        voynich_tokens=encoded_tokens,
        eva_to_triple=eva_to_triple,
        anchors=[],
        inventory=inventory,
        ref_word_set=ref_word_set,
        beam_width=80,
    )
    print(f"  Noise-free dict_hit: {clean_result.best_dict_hit:.3f}  selectivity: {clean_result.best_selectivity:.2f}x")

    # Compute recovery accuracy
    recovered = clean_result.best_assignment
    n_correct = sum(
        1 for k in true_mapping
        if k in recovered and recovered[k] == true_mapping[k]
    )
    recovery_acc = n_correct / len(true_mapping) if true_mapping else 0.0
    print(f"  Recovery accuracy: {n_correct}/{len(true_mapping)} triples = {recovery_acc:.1%}")

    # Run with 20% noise
    all_glyphs_list = list(set(''.join(eva_to_triple.keys())))
    noisy_tokens = _add_noise(encoded_tokens, all_glyphs_list, noise_rate=0.2)
    print("\n  Running noisy (20% substitution) feature CSP...")

    # Re-build variables with fresh domains for the noisy run
    variables_noisy = build_feature_variables(
        eva_to_triple, uniform_freq, inventory, hypothesis_map
    )
    variables_noisy = initialise_feature_domains(
        variables_noisy, inventory, hypothesis_map, anchors=[]
    )

    noisy_result = run_feature_csp_for_language(
        language='latin',
        variables=variables_noisy,
        lm=lm,
        voynich_tokens=noisy_tokens,
        eva_to_triple=eva_to_triple,
        anchors=[],
        inventory=inventory,
        ref_word_set=ref_word_set,
        beam_width=80,
    )
    print(f"  Noisy dict_hit: {noisy_result.best_dict_hit:.3f}  selectivity: {noisy_result.best_selectivity:.2f}x")

    robustness = (
        noisy_result.best_dict_hit / clean_result.best_dict_hit
        if clean_result.best_dict_hit > 0 else 0.0
    )

    # Calibrated expectation for real Voynich
    expected_ceiling = clean_result.best_dict_hit * 0.5  # Voynich likely noisier than 20%
    gate_passed = recovery_acc >= 0.5 and clean_result.best_selectivity >= 1.5

    if gate_passed:
        verdict = (
            f"CALIBRATION PASS: Recovery {recovery_acc:.0%}, selectivity {clean_result.best_selectivity:.2f}x. "
            f"Feature CSP works on known feature-based encoding. "
            f"Expected Voynich dict_hit upper bound: ~{expected_ceiling:.0%}."
        )
    else:
        verdict = (
            f"CALIBRATION PARTIAL: Recovery {recovery_acc:.0%}, selectivity {clean_result.best_selectivity:.2f}x. "
            f"Feature CSP may not reliably decode real feature-based scripts. "
            f"Consider data-driven fallback (Step 14.7)."
        )

    print(f"\n  ── Calibration Summary ──")
    print(f"  Recovery accuracy:         {recovery_acc:.1%}")
    print(f"  Noise-free dict_hit:       {clean_result.best_dict_hit:.3f}")
    print(f"  Noisy dict_hit (20%):      {noisy_result.best_dict_hit:.3f}")
    print(f"  Robustness ratio:          {robustness:.2f}")
    print(f"  Expected Voynich ceiling:  ~{expected_ceiling:.0%}")
    print(f"  Gate: {'PASS' if gate_passed else 'FAIL'}")

    result = CalibrationResult(
        n_triples=len(variables),
        n_tokens_encoded=n_encoded,
        n_tokens_noisy=len(noisy_tokens),
        true_mapping=true_mapping,
        recovered_mapping=recovered,
        recovery_accuracy=recovery_acc,
        noise_free_dict_hit=clean_result.best_dict_hit,
        noisy_dict_hit=noisy_result.best_dict_hit,
        robustness_ratio=robustness,
        clean_selectivity=clean_result.best_selectivity,
        noisy_selectivity=noisy_result.best_selectivity,
        expected_voynich_ceiling=expected_ceiling,
        gate_passed=gate_passed,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'feature_calibrate.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)
    print(f"\n  Results saved → {out_path}")
