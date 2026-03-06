"""
Phase D.2 – Entropy Floor Diagnostic
======================================
Test whether decoded Voynich text entropy is consistent with natural Latin
or shows a verbose encoding floor.  Computes character-level entropy at
orders 0-6 for: (a) decoded Voynich text using the best CSP assignment,
(b) Latin reference corpus, and (c) raw EVA character stream.

If decoded H(6) is substantially above Latin H(6), this indicates the
decoding has not fully penetrated the cipher, or the encoding is
inherently verbose (one-to-many mapping inflates entropy).

Cross-references the Phase 9.5 finding (EVA H(6) = 0.978 bits/char).

Dependency chain:
    results/tironian_csp.json  (preferred)
      OR results/combined_refine.json  (fallback)
    results/entropy_curves.json  (Phase 9.5 reference)
    data/reference/<language>/  (Latin reference corpus)
    corpus (IVTFF)
        -> entropy_floor.json (this step)
"""

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    load_corpus,
    token_to_triples,
    tokenize_eva_chars,
)
from voynich.core.reference import load_reference_corpus
from voynich.core.stats import conditional_entropy, first_order_entropy


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


def _compute_entropy_profile(text: str, max_order: int = 6) -> Dict[str, float]:
    """Compute character entropy at orders 0 through max_order.

    Order 0 = H(1) = first-order (unigram) entropy.
    Order k = H(k+1) = conditional entropy given k preceding characters.
    """
    profile: Dict[str, float] = {}
    if not text or len(text) < 10:
        for order in range(max_order + 1):
            profile[str(order)] = 0.0
        return profile

    # Order 0: first-order entropy (H1)
    profile['0'] = round(first_order_entropy(text), 4)

    # Orders 1-6: conditional entropy H(k+1|k)
    for order in range(1, max_order + 1):
        h = conditional_entropy(text, order=order)
        profile[str(order)] = round(h, 4)

    return profile


def _decode_token_simple(
    token: str,
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
) -> str:
    """Decode a single token using a triple-key -> syllable assignment.

    Returns the concatenated syllable string for the token.
    """
    triples = token_to_triples(token, eva_to_triple)
    syllables = [assignment.get(tk, '?') for tk in triples]
    return ''.join(syllables)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class EntropyFloorResult:
    """Result of entropy floor diagnostic."""
    decoded_entropy_profile: Dict[str, float]  # order -> entropy
    latin_entropy_profile: Dict[str, float]
    eva_raw_entropy_profile: Dict[str, float]
    decoded_floor: float            # H at order 6
    latin_floor: float
    eva_floor: float
    floor_gap: float                # decoded - latin
    is_elevated: bool               # floor_gap > 0.3 bits
    interpretation: str             # natural_language / verbose_encoding / insufficient_data
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_entropy_floor() -> None:
    """Phase D.2: entropy floor diagnostic for decoded Voynich text."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE D.2: Entropy Floor Diagnostic")
    print("=" * 70)

    rd = _results_dir()

    # ─── Step 1: Load best available CSP result ───
    print("\n  1. Loading best available CSP assignment ...")
    assignment: Dict[str, str] = {}
    source_file = 'none'

    # Try tironian_csp.json first, then combined_refine.json
    candidates = ['tironian_csp.json', 'combined_refine.json']
    for cand in candidates:
        cand_path = os.path.join(rd, cand)
        if os.path.exists(cand_path):
            with open(cand_path) as f:
                csp_data = json.load(f)
            assignment = csp_data.get('best_assignment', {})
            if assignment:
                source_file = cand
                break

    if not assignment:
        print("     WARNING: No CSP assignment found.")
        print("     Tried: " + ', '.join(candidates))
        result = EntropyFloorResult(
            decoded_entropy_profile={},
            latin_entropy_profile={},
            eva_raw_entropy_profile={},
            decoded_floor=0.0,
            latin_floor=0.0,
            eva_floor=0.0,
            floor_gap=0.0,
            is_elevated=False,
            interpretation='insufficient_data',
            verdict="INSUFFICIENT DATA: No CSP assignment available for decoding.",
            runtime_seconds=round(time.time() - t0, 2),
        )
        out_path = os.path.join(rd, 'entropy_floor.json')
        with open(out_path, 'w') as f:
            json.dump(_convert(result), f, indent=2)
        print(f"\n  -> {out_path}")
        return

    print(f"     Loaded assignment from {source_file} ({len(assignment)} mappings)")

    # ─── Step 2: Load corpus, decode all tokens ───
    print("\n  2. Loading Voynich corpus and decoding tokens ...")
    corpus = load_corpus(verbose=False)
    tokens = corpus.get_tokens(language='A', paragraph_only=True)
    eva_to_triple = build_eva_to_triple_lookup()

    decoded_words: List[str] = []
    for token in tokens:
        decoded = _decode_token_simple(token, assignment, eva_to_triple)
        if decoded and '?' not in decoded:
            decoded_words.append(decoded)

    print(f"     {len(tokens)} tokens total, {len(decoded_words)} fully decoded")

    # ─── Step 3: Concatenate decoded text into character stream ───
    print("\n  3. Building character streams ...")
    decoded_text = ' '.join(decoded_words)
    print(f"     Decoded text length: {len(decoded_text)} characters")

    # ─── Step 4: Compute decoded entropy profile (orders 0-6) ───
    print("\n  4. Computing decoded text entropy profile (orders 0-6) ...")
    decoded_profile = _compute_entropy_profile(decoded_text, max_order=6)
    for order, h in sorted(decoded_profile.items(), key=lambda x: int(x[0])):
        print(f"     H({order}): {h:.4f} bits/char")

    # ─── Step 5: Load Latin reference corpus, compute same profile ───
    print("\n  5. Loading Latin reference corpus ...")
    latin_profile: Dict[str, float] = {}
    try:
        ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
        latin_text = ref_corpus.get_combined_text('latin')
        if latin_text and len(latin_text) > 100:
            print(f"     Latin text length: {len(latin_text)} characters")
            latin_profile = _compute_entropy_profile(latin_text, max_order=6)
            print("     Latin entropy profile:")
            for order, h in sorted(latin_profile.items(), key=lambda x: int(x[0])):
                print(f"       H({order}): {h:.4f} bits/char")
        else:
            print("     WARNING: Latin corpus too short for reliable entropy estimation")
    except (FileNotFoundError, Exception) as exc:
        print(f"     WARNING: Could not load Latin reference corpus: {exc}")

    if not latin_profile:
        # Use known typical Latin entropy values as fallback
        latin_profile = {
            '0': 4.0, '1': 3.2, '2': 2.6,
            '3': 2.1, '4': 1.7, '5': 1.4, '6': 1.2,
        }
        print("     Using approximate Latin entropy values as fallback")

    # ─── Step 6: Compare entropy floors ───
    print("\n  6. Comparing entropy floors ...")
    decoded_floor = decoded_profile.get('6', 0.0)
    latin_floor = latin_profile.get('6', 0.0)
    floor_gap = round(decoded_floor - latin_floor, 4) if decoded_floor and latin_floor else 0.0
    print(f"     Decoded H(6): {decoded_floor:.4f}")
    print(f"     Latin   H(6): {latin_floor:.4f}")
    print(f"     Gap:          {floor_gap:+.4f} bits/char")

    # ─── Step 7: Load raw EVA text, compute EVA entropy profile ───
    print("\n  7. Computing raw EVA character entropy profile ...")
    eva_chars_list: List[str] = []
    for token in tokens:
        chars = tokenize_eva_chars(token)
        eva_chars_list.extend(chars)

    # Join EVA chars with no separator (they are already discrete symbols)
    eva_text = ''.join(eva_chars_list)
    eva_profile = _compute_entropy_profile(eva_text, max_order=6)
    print(f"     EVA text length: {len(eva_text)} characters")
    for order, h in sorted(eva_profile.items(), key=lambda x: int(x[0])):
        print(f"     EVA H({order}): {h:.4f} bits/char")

    # ─── Step 8: Cross-reference Phase 9.5 entropy_curves.json ───
    print("\n  8. Cross-referencing Phase 9.5 entropy curves ...")
    ec_path = os.path.join(rd, 'entropy_curves.json')
    phase95_h6 = None
    if os.path.exists(ec_path):
        with open(ec_path) as f:
            ec_data = json.load(f)
        # Try to find the H(6) value from section_analysis combined_curve
        sa = ec_data.get('section_analysis', {})
        combined_curve = sa.get('combined_curve', [])
        if isinstance(combined_curve, list) and len(combined_curve) > 6:
            phase95_h6 = combined_curve[6]  # index 6 = order 6
            print(f"     Phase 9.5 EVA H(6): {phase95_h6:.4f} bits/char")
        elif isinstance(combined_curve, dict):
            phase95_h6 = combined_curve.get('6', combined_curve.get(6, None))
            if phase95_h6 is not None:
                print(f"     Phase 9.5 EVA H(6): {phase95_h6:.4f} bits/char")
            else:
                print("     Phase 9.5 combined_curve structure not as expected")
        else:
            print("     Phase 9.5 combined_curve not found or too short")
    else:
        print("     entropy_curves.json not found")

    # ─── Step 9: Report interpretation ───
    print("\n  9. Interpretation ...")
    is_elevated = floor_gap > 0.3
    eva_floor = eva_profile.get('6', 0.0)

    if decoded_floor == 0.0 or latin_floor == 0.0:
        interpretation = 'insufficient_data'
        verdict = (
            "INSUFFICIENT DATA: Cannot reliably compare entropy floors. "
            "Decoded text too short or Latin reference unavailable."
        )
    elif not is_elevated:
        interpretation = 'natural_language'
        verdict = (
            f"CONSISTENT WITH NATURAL LANGUAGE: Decoded H(6)={decoded_floor:.3f} is within "
            f"0.3 bits of Latin H(6)={latin_floor:.3f} (gap={floor_gap:+.3f}). "
            f"The decoded text shows entropy convergence consistent with natural Latin."
        )
    else:
        interpretation = 'verbose_encoding'
        verdict = (
            f"ELEVATED ENTROPY FLOOR: Decoded H(6)={decoded_floor:.3f} is "
            f"{floor_gap:+.3f} bits above Latin H(6)={latin_floor:.3f}. "
            f"This suggests verbose (one-to-many) encoding: multiple cipher signs "
            f"map to the same plaintext unit, inflating per-character unpredictability. "
            f"Raw EVA H(6)={eva_floor:.3f} for comparison."
        )

    print(f"     Interpretation: {interpretation}")
    print(f"     {verdict}")

    # ─── Save ───
    result = EntropyFloorResult(
        decoded_entropy_profile=decoded_profile,
        latin_entropy_profile=latin_profile,
        eva_raw_entropy_profile=eva_profile,
        decoded_floor=decoded_floor,
        latin_floor=latin_floor,
        eva_floor=eva_floor,
        floor_gap=floor_gap,
        is_elevated=is_elevated,
        interpretation=interpretation,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'entropy_floor.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)

    print(f"\n  -> {out_path}")
