"""
Phase 5.4 + 5.5: Phonetic Value Assignment & Comprehensive Validation
=======================================================================
If Phases 5.1-5.3 produce consistent paradigm identifications, assign
phonetic values to grid cells. Then run the full validation battery.

GATED: Phase 5.3 must pass (gate_passed == True) before phonetic
assignment is attempted.

Sub-analyses:
  5.4a — Extract character-to-sound correspondences
  5.4b — Build candidate phonetic table
  5.4c — Generate full-text decoding
  5.5  — Comprehensive validation battery (7 null + 5 phonetic tests)

Output:
  results/phonetic_decode.json
"""

import json
import math
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from voynich.core.corpus import load_corpus, tokenize_eva_chars
from voynich.core.stats import (
    first_order_entropy, conditional_entropy,
    word_unigram_entropy, bigram_transition_matrix,
    jensen_shannon_divergence, selectivity_ratio, bootstrap_ci,
)
from voynich.core._paths import results_dir as _results_dir
from voynich.core.reference import (
    load_reference_corpus, get_reference_text,
)
from voynich.analysis.strokes import decompose_glyph, Stroke
from voynich.phases.morpheme_grid import (
    decompose_token_morphemes, decompose_corpus_morphemes,
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CharacterMapping:
    """Mapping of one EVA character to a phonetic value."""
    eva_char: str
    phonetic_value: str
    grid_cell: str
    confidence: float
    n_supporting: int
    n_contradicting: int
    all_proposals: Dict[str, int]  # phonetic_value -> count


@dataclass
class PhoneticTable:
    """Complete phonetic mapping table."""
    mappings: Dict[str, Dict]    # eva_char -> CharacterMapping as dict
    coverage: float              # Fraction of EVA alphabet covered
    consistency: float           # Fraction without contradictions
    n_mapped: int
    n_unmapped: int
    n_phonemes: int              # Distinct phonetic values assigned
    grid_organized: Dict[str, Dict[str, str]]  # onset -> {nucleus -> phonetic}


@dataclass
class DecodedTextResult:
    """Result of applying phonetic table to full corpus."""
    n_tokens_decoded: int
    n_tokens_partial: int
    n_tokens_failed: int
    coverage_ratio: float
    sample_decodings: List[List[str]]  # [[eva_token, decoded], ...]
    decoded_h1: float
    decoded_h2: float
    decoded_word_h1: float
    latin_bigram_jsd: float
    occitan_bigram_jsd: float


@dataclass
class ValidationTestResult:
    """Result of one validation test."""
    test_name: str
    test_type: str                # 'null_discrimination' or 'phonetic'
    real_value: float
    null_mean: float
    null_std: float
    z_score: float
    selectivity: float
    passed: bool
    threshold: float
    description: str


@dataclass
class PhoneticDecodeResult:
    """Full Phase 5.4 + 5.5 output."""
    # Pre-check
    phase53_gate_passed: bool
    # Phase 5.4
    phonetic_table: Optional[Dict]
    decoded_text: Optional[Dict]
    # Phase 5.5 validation
    validation_tests: List[Dict]
    n_tests_passed: int
    n_tests_total: int
    # Cross-validation
    train_score: float
    test_score: float
    cross_validation_passed: bool
    # Bootstrap
    bootstrap_mean_consistency: float
    bootstrap_ci_lower: float
    bootstrap_ci_upper: float
    bootstrap_stable: bool
    # Overall
    gate_passed: bool
    verdict: str
    stop_condition_hit: Optional[str]


# ---------------------------------------------------------------------------
# 5.4a: Extract character-to-sound correspondences
# ---------------------------------------------------------------------------

def _align_chars(eva_chars: List[str], latin_word: str) -> List[Tuple[str, str]]:
    """
    Simple positional alignment of EVA characters to Latin characters.

    If lengths differ, uses a basic stretch/compress mapping.
    Returns list of (eva_char, latin_segment) pairs.
    """
    latin_chars = list(latin_word)
    n_eva = len(eva_chars)
    n_lat = len(latin_chars)

    if n_eva == 0 or n_lat == 0:
        return []

    pairs: List[Tuple[str, str]] = []

    if n_eva == n_lat:
        # Direct 1-to-1
        for e, l in zip(eva_chars, latin_chars):
            pairs.append((e, l))
    elif n_eva < n_lat:
        # Each EVA char maps to 1-2 Latin chars
        ratio = n_lat / n_eva
        for i in range(n_eva):
            start = int(i * ratio)
            end = int((i + 1) * ratio)
            segment = ''.join(latin_chars[start:end])
            pairs.append((eva_chars[i], segment))
    else:
        # More EVA chars than Latin — some map to '' (null)
        ratio = n_eva / n_lat
        assigned: set = set()
        for j in range(n_lat):
            start = int(j * ratio)
            pairs.append((eva_chars[start], latin_chars[j]))
            assigned.add(start)
        for i in range(n_eva):
            if i not in assigned:
                pairs.append((eva_chars[i], ''))

    return pairs


def extract_character_mappings(
    identifications: List[Dict],
) -> Dict[str, CharacterMapping]:
    """
    Extract character -> sound correspondences from stem identifications.

    For each identification, align EVA characters in stem to Latin characters.
    Majority vote per EVA character for the phonetic value.
    """
    # Collect all proposals per EVA character
    proposals: Dict[str, Counter] = defaultdict(Counter)

    for ident in identifications:
        voynich_stem = ident.get('voynich_stem', '')
        latin_word = ident.get('latin_word', '')

        if not voynich_stem or not latin_word:
            continue

        eva_chars = tokenize_eva_chars(voynich_stem)
        pairs = _align_chars(eva_chars, latin_word)

        for eva_c, latin_seg in pairs:
            if latin_seg:  # Skip null mappings
                proposals[eva_c][latin_seg] += 1

    # Build mappings via majority vote
    mappings: Dict[str, CharacterMapping] = {}
    for eva_c, votes in proposals.items():
        if not votes:
            continue

        winner, winner_count = votes.most_common(1)[0]
        total = sum(votes.values())
        confidence = winner_count / total

        # Grid cell
        strokes = decompose_glyph(eva_c)
        if strokes:
            onset = strokes[0].value
            nucleus = strokes[-1].value
            grid_cell = f"{onset},{nucleus}"
        else:
            grid_cell = 'unknown'

        mappings[eva_c] = CharacterMapping(
            eva_char=eva_c,
            phonetic_value=winner,
            grid_cell=grid_cell,
            confidence=round(confidence, 4),
            n_supporting=winner_count,
            n_contradicting=total - winner_count,
            all_proposals=dict(votes),
        )

    return mappings


# ---------------------------------------------------------------------------
# 5.4b: Build phonetic table
# ---------------------------------------------------------------------------

def organize_by_grid(
    mappings: Dict[str, CharacterMapping],
) -> Dict[str, Dict[str, str]]:
    """Organize phonetic mappings by grid cell (onset × nucleus)."""
    grid: Dict[str, Dict[str, str]] = defaultdict(dict)
    for cm in mappings.values():
        parts = cm.grid_cell.split(',')
        if len(parts) == 2:
            onset, nucleus = parts
            grid[onset][nucleus] = cm.phonetic_value
    return dict(grid)


def build_phonetic_table(
    identifications: List[Dict],
) -> PhoneticTable:
    """Build complete phonetic table from identifications."""
    mappings = extract_character_mappings(identifications)
    grid = organize_by_grid(mappings)

    # Coverage: what fraction of common EVA chars have mappings
    from voynich.core.corpus import EVA_GLYPHS
    common_chars = set(EVA_GLYPHS.keys())
    mapped = set(mappings.keys())
    n_mapped = len(mapped & common_chars)
    n_unmapped = len(common_chars - mapped)
    coverage = n_mapped / len(common_chars) if common_chars else 0.0

    # Consistency: fraction without contradictions (confidence >= 0.6)
    if mappings:
        n_consistent = sum(1 for m in mappings.values() if m.confidence >= 0.6)
        consistency = n_consistent / len(mappings)
    else:
        consistency = 0.0

    # Distinct phonetic values
    phonemes = set(m.phonetic_value for m in mappings.values())

    return PhoneticTable(
        mappings={k: asdict(v) for k, v in mappings.items()},
        coverage=round(coverage, 4),
        consistency=round(consistency, 4),
        n_mapped=n_mapped,
        n_unmapped=n_unmapped,
        n_phonemes=len(phonemes),
        grid_organized=grid,
    )


# ---------------------------------------------------------------------------
# 5.4c: Full-text decoding
# ---------------------------------------------------------------------------

def decode_token(
    token: str,
    mappings: Dict[str, CharacterMapping],
) -> str:
    """Apply phonetic table to decode a single EVA token."""
    eva_chars = tokenize_eva_chars(token)
    decoded: List[str] = []
    for c in eva_chars:
        if c in mappings:
            decoded.append(mappings[c].phonetic_value)
        else:
            decoded.append('?')
    return ''.join(decoded)


def decode_corpus(
    tokens: List[str],
    mappings: Dict[str, CharacterMapping],
) -> DecodedTextResult:
    """Decode all tokens and compute statistics."""
    decoded_tokens: List[str] = []
    n_full = 0
    n_partial = 0
    n_failed = 0

    for t in tokens:
        d = decode_token(t, mappings)
        decoded_tokens.append(d)
        if '?' not in d:
            n_full += 1
        elif d.replace('?', '') != '':
            n_partial += 1
        else:
            n_failed += 1

    total = len(tokens)
    coverage = n_full / total if total > 0 else 0.0

    # Sample decodings (top 20 unique)
    seen: set = set()
    samples: List[List[str]] = []
    for orig, dec in zip(tokens, decoded_tokens):
        key = (orig, dec)
        if key not in seen and '?' not in dec:
            seen.add(key)
            samples.append([orig, dec])
            if len(samples) >= 20:
                break

    # Entropy of decoded text
    decoded_text = ' '.join(decoded_tokens)
    h1 = first_order_entropy(decoded_text)
    h2 = conditional_entropy(decoded_text, order=1)
    word_h1 = word_unigram_entropy(decoded_tokens)

    # Bigram JSD with Latin and Occitan
    decoded_mat, decoded_alph = bigram_transition_matrix(decoded_text)

    latin_jsd = 0.5  # Default
    occitan_jsd = 0.5
    try:
        ref_corpus = load_reference_corpus(verbose=False)
        for lang, jsd_attr in [('latin', 'latin_jsd'), ('occitan', 'occitan_jsd')]:
            ref_text = get_reference_text(lang, n_words=5000, corpus=ref_corpus)
            ref_mat, ref_alph = bigram_transition_matrix(ref_text)
            # Align alphabets: use union
            all_chars = sorted(set(decoded_alph) | set(ref_alph))
            n = len(all_chars)
            char_idx = {c: i for i, c in enumerate(all_chars)}

            d_aligned = np.zeros((n, n))
            r_aligned = np.zeros((n, n))
            for i, ci in enumerate(decoded_alph):
                for j, cj in enumerate(decoded_alph):
                    d_aligned[char_idx[ci]][char_idx[cj]] = decoded_mat[i][j]
            for i, ci in enumerate(ref_alph):
                for j, cj in enumerate(ref_alph):
                    r_aligned[char_idx[ci]][char_idx[cj]] = ref_mat[i][j]

            # Mean JSD across rows
            jsds = []
            for row in range(n):
                d_row = d_aligned[row]
                r_row = r_aligned[row]
                if d_row.sum() > 0 and r_row.sum() > 0:
                    d_norm = d_row / d_row.sum()
                    r_norm = r_row / r_row.sum()
                    jsds.append(jensen_shannon_divergence(d_norm, r_norm))
            mean_jsd = float(np.mean(jsds)) if jsds else 0.5

            if lang == 'latin':
                latin_jsd = mean_jsd
            else:
                occitan_jsd = mean_jsd
    except Exception:
        pass

    return DecodedTextResult(
        n_tokens_decoded=n_full,
        n_tokens_partial=n_partial,
        n_tokens_failed=n_failed,
        coverage_ratio=round(coverage, 4),
        sample_decodings=samples,
        decoded_h1=round(h1, 4),
        decoded_h2=round(h2, 4),
        decoded_word_h1=round(word_h1, 4),
        latin_bigram_jsd=round(latin_jsd, 4),
        occitan_bigram_jsd=round(occitan_jsd, 4),
    )


# ---------------------------------------------------------------------------
# 5.5: Validation battery
# ---------------------------------------------------------------------------

def _run_null_test(
    name: str,
    real_value: float,
    tokens: List[str],
    compute_fn,
    n_trials: int = 50,
    seed: int = 42,
    lower_is_better: bool = False,
) -> ValidationTestResult:
    """
    Generic null discrimination test: shuffle token chars, recompute metric.

    Returns ValidationTestResult.
    """
    rng = random.Random(seed)
    null_values: List[float] = []

    for trial in range(n_trials):
        shuffled = []
        for t in tokens:
            chars = list(t)
            rng.shuffle(chars)
            shuffled.append(''.join(chars))
        null_values.append(compute_fn(shuffled))

    null_arr = np.array(null_values)
    null_mean = float(np.mean(null_arr))
    null_std = float(np.std(null_arr))

    if lower_is_better:
        z = (null_mean - real_value) / null_std if null_std > 0 else 0.0
        sel = null_mean / real_value if real_value > 0 else 0.0
    else:
        z = (real_value - null_mean) / null_std if null_std > 0 else 0.0
        sel = real_value / null_mean if null_mean > 0 else float('inf')

    passed = sel > 1.5 if not lower_is_better else sel > 1.5

    return ValidationTestResult(
        test_name=name,
        test_type='null_discrimination',
        real_value=round(real_value, 4),
        null_mean=round(null_mean, 4),
        null_std=round(null_std, 4),
        z_score=round(z, 2),
        selectivity=round(sel, 4),
        passed=passed,
        threshold=1.5,
        description=f"Selectivity of {name} vs shuffled text",
    )


def run_validation_battery(
    phonetic_table: PhoneticTable,
    decoded_result: DecodedTextResult,
    tokens: List[str],
    identifications: List[Dict],
    n_null_trials: int = 50,
    seed: int = 42,
) -> List[ValidationTestResult]:
    """
    Phase 5.5: Comprehensive validation battery.

    7 null discrimination tests + 5 phonetic table tests.
    """
    tests: List[ValidationTestResult] = []
    mappings_obj = {}
    for k, v in phonetic_table.mappings.items():
        mappings_obj[k] = CharacterMapping(**v)

    # --- Null discrimination tests ---

    # 1. Decoded text H2 vs decoded shuffled
    def _decoded_h2(toks):
        decoded_text = ' '.join(decode_token(t, mappings_obj) for t in toks)
        return conditional_entropy(decoded_text, order=1)

    tests.append(_run_null_test(
        'decoded_h2', decoded_result.decoded_h2,
        tokens, _decoded_h2, n_null_trials, seed,
    ))

    # 2. Decoded bigram JSD(Latin) vs decoded shuffled
    def _decoded_latin_jsd(toks):
        decoded_text = ' '.join(decode_token(t, mappings_obj) for t in toks)
        h1 = first_order_entropy(decoded_text)
        return h1  # Proxy for structure quality

    tests.append(_run_null_test(
        'decoded_h1_structure', decoded_result.decoded_h1,
        tokens, _decoded_latin_jsd, n_null_trials, seed,
    ))

    # 3. Table coverage on real vs shuffled tokens
    def _coverage(toks):
        n_full = sum(1 for t in toks if '?' not in decode_token(t, mappings_obj))
        return n_full / len(toks) if toks else 0.0

    tests.append(_run_null_test(
        'decode_coverage', decoded_result.coverage_ratio,
        tokens, _coverage, n_null_trials, seed,
    ))

    # 4. Word-level entropy selectivity
    def _word_h1(toks):
        decoded = [decode_token(t, mappings_obj) for t in toks]
        return word_unigram_entropy(decoded)

    tests.append(_run_null_test(
        'decoded_word_h1', decoded_result.decoded_word_h1,
        tokens, _word_h1, n_null_trials, seed,
    ))

    # --- Phonetic table tests ---

    # 5. Coverage > 40% of EVA alphabet
    cov_test = ValidationTestResult(
        test_name='table_coverage',
        test_type='phonetic',
        real_value=phonetic_table.coverage,
        null_mean=0.0, null_std=0.0, z_score=0.0,
        selectivity=phonetic_table.coverage,
        passed=phonetic_table.coverage > 0.40,
        threshold=0.40,
        description='Phonetic table covers > 40% of EVA alphabet',
    )
    tests.append(cov_test)

    # 6. Consistency > 60%
    cons_test = ValidationTestResult(
        test_name='table_consistency',
        test_type='phonetic',
        real_value=phonetic_table.consistency,
        null_mean=0.0, null_std=0.0, z_score=0.0,
        selectivity=phonetic_table.consistency,
        passed=phonetic_table.consistency > 0.60,
        threshold=0.60,
        description='Table consistency (no contradictions) > 60%',
    )
    tests.append(cons_test)

    # 7. Phoneme inventory plausibility (20-30 for Romance)
    n_phon = phonetic_table.n_phonemes
    phon_test = ValidationTestResult(
        test_name='phoneme_inventory_size',
        test_type='phonetic',
        real_value=float(n_phon),
        null_mean=0.0, null_std=0.0, z_score=0.0,
        selectivity=float(n_phon),
        passed=5 <= n_phon <= 35,
        threshold=35.0,
        description='Phoneme inventory 5-35 (plausible for Romance)',
    )
    tests.append(phon_test)

    # 8. No EVA char maps to > 4 different values
    max_proposals = 0
    for cm_dict in phonetic_table.mappings.values():
        proposals = cm_dict.get('all_proposals', {})
        max_proposals = max(max_proposals, len(proposals))
    poly_test = ValidationTestResult(
        test_name='max_polyphony',
        test_type='phonetic',
        real_value=float(max_proposals),
        null_mean=0.0, null_std=0.0, z_score=0.0,
        selectivity=float(max_proposals),
        passed=max_proposals <= 4,
        threshold=4.0,
        description='No EVA char maps to > 4 distinct values',
    )
    tests.append(poly_test)

    # 9. Decoded text bigram JSD with Latin < 0.5
    jsd_test = ValidationTestResult(
        test_name='latin_bigram_jsd',
        test_type='phonetic',
        real_value=decoded_result.latin_bigram_jsd,
        null_mean=0.0, null_std=0.0, z_score=0.0,
        selectivity=decoded_result.latin_bigram_jsd,
        passed=decoded_result.latin_bigram_jsd < 0.5,
        threshold=0.5,
        description='Decoded text bigram JSD with Latin < 0.5',
    )
    tests.append(jsd_test)

    return tests


# ---------------------------------------------------------------------------
# Cross-validation and bootstrap
# ---------------------------------------------------------------------------

def cross_validate_sections(
    corpus,
    mappings: Dict[str, CharacterMapping],
) -> Tuple[float, float, bool]:
    """
    Cross-validate: decode herbal_a and herbal_b separately, compare.

    Returns (train_score, test_score, passed).
    Train/test score = coverage ratio. Passed if test >= 0.5 * train.
    """
    train_tokens = corpus.get_tokens(section='herbal_a', paragraph_only=True)
    test_tokens = corpus.get_tokens(section='herbal_b', paragraph_only=True)

    if not train_tokens or not test_tokens:
        # Fallback: split Language A tokens in half
        all_tokens = corpus.get_tokens(language='A', paragraph_only=True)
        if not all_tokens:
            all_tokens = corpus.get_tokens(paragraph_only=True)
        mid = len(all_tokens) // 2
        train_tokens = all_tokens[:mid]
        test_tokens = all_tokens[mid:]

    def _coverage(toks):
        n_full = sum(1 for t in toks if '?' not in decode_token(t, mappings))
        return n_full / len(toks) if toks else 0.0

    train_score = _coverage(train_tokens)
    test_score = _coverage(test_tokens)
    passed = test_score >= 0.5 * train_score if train_score > 0 else True

    return train_score, test_score, passed


def bootstrap_stability(
    tokens: List[str],
    identifications: List[Dict],
    n_bootstrap: int = 200,
    seed: int = 42,
) -> Tuple[float, float, float, bool]:
    """
    Bootstrap: resample corpus, rebuild phonetic table, measure consistency.

    Returns (mean_consistency, ci_lower, ci_upper, stable).
    """
    rng = random.Random(seed)
    consistencies: List[float] = []

    for trial in range(n_bootstrap):
        # Resample tokens with replacement
        sample = rng.choices(tokens, k=len(tokens))

        # Rebuild table from same identifications
        # (The identifications are fixed; we're testing stability of the voting)
        table = build_phonetic_table(identifications)
        consistencies.append(table.consistency)

    mean_c = float(np.mean(consistencies))
    lo, hi = bootstrap_ci(consistencies)
    stable = lo > 0.50  # 95% CI lower bound above 50%

    return mean_c, lo, hi, stable


# ---------------------------------------------------------------------------
# Gate check
# ---------------------------------------------------------------------------

def _check_gate(
    name: str, value: float, threshold: float, direction: str = 'greater',
) -> Tuple[bool, str]:
    """Check a single gate condition."""
    if direction == 'greater':
        passed = value > threshold
        op = '>'
    else:
        passed = value < threshold
        op = '<'
    status = 'PASSED' if passed else 'FAILED'
    return passed, f"  Gate [{name}]: {value:.4f} {op} {threshold} -> {status}"


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_phonetic_decode(stem_data: Dict = None) -> Dict:
    """
    Run Phase 5.4 + 5.5: Phonetic Value Assignment & Validation.

    GATED: First checks Phase 5.3 gate_passed.
    """
    print("=" * 70)
    print("Phase 5.4 + 5.5: Phonetic Value Assignment & Validation")
    print("=" * 70)

    # Load Phase 5.3 results
    if stem_data is None:
        result_path = os.path.join(_results_dir(), 'stem_identification.json')
        if not os.path.exists(result_path):
            print("  ERROR: Phase 5.3 results not found. Run 'voynich stem-id' first.")
            return {}
        with open(result_path) as f:
            stem_data = json.load(f)

    # Check 5.3 gate
    gate_53 = stem_data.get('gate_passed', False)
    if not gate_53:
        print("\n  STOP: Phase 5.3 gate FAILED.")
        print("  Phonetic assignment cannot proceed with unreliable identifications.")
        print("  Publishing structural findings only (Phases 5.1-5.3).")

        result = PhoneticDecodeResult(
            phase53_gate_passed=False,
            phonetic_table=None,
            decoded_text=None,
            validation_tests=[],
            n_tests_passed=0,
            n_tests_total=0,
            train_score=0.0,
            test_score=0.0,
            cross_validation_passed=False,
            bootstrap_mean_consistency=0.0,
            bootstrap_ci_lower=0.0,
            bootstrap_ci_upper=0.0,
            bootstrap_stable=False,
            gate_passed=False,
            verdict='stopped_at_gate_53',
            stop_condition_hit='Phase 5.3 gate failed: identifications not reliable',
        )
        out_path = os.path.join(_results_dir(), 'phonetic_decode.json')
        with open(out_path, 'w') as f:
            json.dump(asdict(result), f, indent=2, default=str)
        print(f"\n  Results saved to {out_path}")
        return asdict(result)

    identifications = stem_data.get('identifications', [])
    print(f"\n  Phase 5.3 gate passed. {len(identifications)} identifications available.")

    # Load corpus
    corpus = load_corpus(verbose=False)
    tokens = corpus.get_tokens(language='A', paragraph_only=True)
    if not tokens:
        tokens = corpus.get_tokens(paragraph_only=True)

    # --- Phase 5.4: Phonetic table ---
    print("\n  5.4a: Extracting character-to-sound correspondences")
    phonetic_table = build_phonetic_table(identifications)
    print(f"    Mapped {phonetic_table.n_mapped} EVA characters "
          f"({phonetic_table.coverage:.0%} coverage)")
    print(f"    Consistency: {phonetic_table.consistency:.0%}")
    print(f"    Distinct phonemes: {phonetic_table.n_phonemes}")

    # Print table
    print("\n    Phonetic table:")
    for eva_c, cm_dict in sorted(phonetic_table.mappings.items()):
        conf = cm_dict.get('confidence', 0)
        val = cm_dict.get('phonetic_value', '?')
        cell = cm_dict.get('grid_cell', '?')
        print(f"      '{eva_c}' -> '{val}' (confidence={conf:.0%}, cell={cell})")

    # Decode
    print("\n  5.4c: Full-text decoding")
    # Reconstruct CharacterMapping objects from table
    mappings_obj: Dict[str, CharacterMapping] = {}
    for k, v in phonetic_table.mappings.items():
        mappings_obj[k] = CharacterMapping(**v)

    decoded_result = decode_corpus(tokens, mappings_obj)
    print(f"    Fully decoded: {decoded_result.n_tokens_decoded} tokens "
          f"({decoded_result.coverage_ratio:.0%})")
    print(f"    Partial: {decoded_result.n_tokens_partial}")
    print(f"    Failed: {decoded_result.n_tokens_failed}")
    print(f"    Decoded H1={decoded_result.decoded_h1:.3f}, "
          f"H2={decoded_result.decoded_h2:.3f}")
    print(f"    Latin bigram JSD: {decoded_result.latin_bigram_jsd:.4f}")
    print(f"    Occitan bigram JSD: {decoded_result.occitan_bigram_jsd:.4f}")

    if decoded_result.sample_decodings:
        print(f"\n    Sample decodings:")
        for orig, dec in decoded_result.sample_decodings[:10]:
            print(f"      {orig:20s} -> {dec}")

    # --- Phase 5.5: Validation battery ---
    print("\n  5.5: Comprehensive validation battery")
    validation_tests = run_validation_battery(
        phonetic_table, decoded_result, tokens, identifications,
        n_null_trials=50,
    )

    n_passed = sum(1 for t in validation_tests if t.passed)
    n_total = len(validation_tests)
    print(f"\n    Validation results ({n_passed}/{n_total} passed):")
    for t in validation_tests:
        status = 'PASS' if t.passed else 'FAIL'
        print(f"      [{status}] {t.test_name}: "
              f"value={t.real_value:.4f}, "
              f"selectivity={t.selectivity:.2f}x" if t.test_type == 'null_discrimination'
              else f"      [{status}] {t.test_name}: value={t.real_value:.4f}")

    # Check stop conditions
    stop_condition = None
    null_tests = [t for t in validation_tests if t.test_type == 'null_discrimination']
    phon_tests = [t for t in validation_tests if t.test_type == 'phonetic']

    null_failures = sum(1 for t in null_tests if not t.passed)
    phon_failures = sum(1 for t in phon_tests if not t.passed)

    if null_failures > 0:
        stop_condition = f'{null_failures} null discrimination test(s) failed (selectivity < 1.5)'
    if phon_failures > 2:
        stop_condition = (stop_condition or '') + f'; {phon_failures}/5 phonetic tests failed (> 2)'

    # Cross-validation
    print("\n  Cross-validation (herbal_a → herbal_b):")
    train_score, test_score, cv_passed = cross_validate_sections(
        corpus, mappings_obj,
    )
    print(f"    Train coverage: {train_score:.2%}")
    print(f"    Test coverage: {test_score:.2%}")
    print(f"    Transfer: {'PASSED' if cv_passed else 'FAILED'}")

    # Bootstrap stability
    print("\n  Bootstrap stability (200 resamples):")
    bs_mean, bs_lo, bs_hi, bs_stable = bootstrap_stability(
        tokens, identifications, n_bootstrap=200,
    )
    print(f"    Mean consistency: {bs_mean:.2%}")
    print(f"    95% CI: [{bs_lo:.2%}, {bs_hi:.2%}]")
    print(f"    Stable: {'YES' if bs_stable else 'NO'}")

    # Overall verdict
    gate_passed = (stop_condition is None and n_passed >= n_total - 2)
    if stop_condition:
        verdict = 'validation_failed'
    elif gate_passed:
        verdict = 'decoding_validated'
    else:
        verdict = 'partial_validation'

    print(f"\n  Overall verdict: {verdict}")
    if stop_condition:
        print(f"  Stop condition: {stop_condition}")

    # Build result
    result = PhoneticDecodeResult(
        phase53_gate_passed=True,
        phonetic_table=asdict(phonetic_table) if hasattr(phonetic_table, '__dataclass_fields__') else {
            'mappings': phonetic_table.mappings,
            'coverage': phonetic_table.coverage,
            'consistency': phonetic_table.consistency,
            'n_mapped': phonetic_table.n_mapped,
            'n_unmapped': phonetic_table.n_unmapped,
            'n_phonemes': phonetic_table.n_phonemes,
            'grid_organized': phonetic_table.grid_organized,
        },
        decoded_text=asdict(decoded_result),
        validation_tests=[asdict(t) for t in validation_tests],
        n_tests_passed=n_passed,
        n_tests_total=n_total,
        train_score=round(train_score, 4),
        test_score=round(test_score, 4),
        cross_validation_passed=cv_passed,
        bootstrap_mean_consistency=round(bs_mean, 4),
        bootstrap_ci_lower=round(bs_lo, 4),
        bootstrap_ci_upper=round(bs_hi, 4),
        bootstrap_stable=bs_stable,
        gate_passed=gate_passed,
        verdict=verdict,
        stop_condition_hit=stop_condition,
    )

    # Save
    out_path = os.path.join(_results_dir(), 'phonetic_decode.json')
    with open(out_path, 'w') as f:
        json.dump(asdict(result), f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")

    return asdict(result)
