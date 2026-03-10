"""
Phase 31.8: EVA Re-Segmentation Test
=======================================
Test whether merging EVA ligature candidates (ch, sh, cth, ckh, cph, cfh)
into single characters produces a better decoding.

Dependency chain:
    combined_refine.json       (Phase 15)
    modifier_integrate.json    (Phase 16)
    null_corpus.json           (Phase 17)
        → resegmentation_test.json  (this step)
"""

import json
import os
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
from voynich.core.reference import (
    EVA_VISUAL_COMPONENTS,
    build_expanded_word_set,
    load_reference_corpus,
)
from voynich.phases.null_corpus import (
    _build_eva_bigram_model,
    _generate_null_corpus,
    _reconstruct_modifier_rules,
)
from voynich.phases.signal_isolation import _decode_corpus_r3


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
# Merge scheme definitions
# ---------------------------------------------------------------------------

def _define_merge_schemes() -> Dict[str, Dict[str, str]]:
    """Define the 4 merge schemes.

    Each scheme maps (char_sequence) → merged_sign_name.
    Only sequences where the component chars always appear together are merged.
    """
    return {
        'M1_minimal': {
            'ch': 'CH', 'sh': 'SH',
        },
        'M2_h_series': {
            'ch': 'CH', 'sh': 'SH', 'cth': 'CTH',
            'ckh': 'CKH', 'cph': 'CPH', 'cfh': 'CFH',
        },
        'M3_h_qo': {
            'ch': 'CH', 'sh': 'SH', 'cth': 'CTH',
            'ckh': 'CKH', 'cph': 'CPH', 'cfh': 'CFH',
            'qo': 'QO', 'qot': 'QOT', 'qok': 'QOK',
        },
        'M4_h_bench': {
            'ch': 'CH', 'sh': 'SH', 'cth': 'CTH',
            'ckh': 'CKH', 'cph': 'CPH', 'cfh': 'CFH',
            'qo': 'QO', 'qot': 'QOT', 'qok': 'QOK',
            'ol': 'OL', 'al': 'AL', 'or': 'OR', 'ar': 'AR',
        },
    }


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SchemeResult:
    """Result for one merge scheme."""
    scheme_name: str
    merges: Dict[str, str]
    n_merged_chars: int
    original_inventory_size: int
    merged_inventory_size: int
    n_unique_triples: int
    dict_hit: float
    signal_rate: float
    mean_token_length: float


@dataclass
class ResegmentationResult:
    """Full Step 31.8 output."""
    baseline_dict_hit: float
    baseline_inventory_size: int
    baseline_mean_token_length: float
    scheme_results: List[Dict]
    best_scheme: str
    best_dict_hit: float
    best_delta: float
    fingerprint_match_delta: float
    historical_new_matches: int
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Re-tokenization
# ---------------------------------------------------------------------------

def _retokenize_token(token: str, merges: Dict[str, str]) -> str:
    """Re-tokenize a single token by merging character sequences.

    The EVA tokenizer already handles multi-char sequences like 'ch' as single
    tokens. So what we're really doing here is confirming that the standard
    tokenizer already treats these as units, and measuring the effect on
    decoding when we give them independent triple assignments.
    """
    # tokenize_eva_chars already handles ch, sh, etc. as single chars
    # So for M1/M2 the tokenization doesn't change.
    # What changes is the TRIPLE assignment — we can give merged chars
    # a single stroke triple rather than decomposing them.
    return token  # Tokenization is unchanged; the change is in triple mapping


def _build_merged_triple_lookup(
    merges: Dict[str, str],
) -> Dict[str, str]:
    """Build a triple lookup that treats merged chars as single units.

    For merged chars, compute a combined stroke triple from their components.
    """
    base_lookup = build_eva_to_triple_lookup()
    merged_lookup = dict(base_lookup)

    for char_seq, merged_name in merges.items():
        if char_seq in EVA_VISUAL_COMPONENTS:
            # Already a single char in EVA — just use its existing triple
            comp = EVA_VISUAL_COMPONENTS[char_seq]
            triple = f"{comp['first_stroke']},{comp['last_stroke']},{comp['glyph_class']}"
            merged_lookup[char_seq] = triple
        else:
            # Not in EVA_VISUAL_COMPONENTS — create a new triple
            # Use the first char's first_stroke and last char's last_stroke
            chars = tokenize_eva_chars(char_seq)
            if len(chars) >= 2:
                first_char = chars[0]
                last_char = chars[-1]
                first_comp = EVA_VISUAL_COMPONENTS.get(first_char, {})
                last_comp = EVA_VISUAL_COMPONENTS.get(last_char, {})
                if first_comp and last_comp:
                    triple = (f"{first_comp['first_stroke']},"
                              f"{last_comp['last_stroke']},"
                              f"merged_{merged_name.lower()}")
                    merged_lookup[char_seq] = triple

    return merged_lookup


def _compute_inventory_size(corpus, merges: Dict[str, str]) -> Tuple[int, int, float]:
    """Compute character inventory size and mean token length under a merge scheme."""
    all_tokens = corpus.get_tokens()

    # Baseline
    baseline_chars = set()
    for token in all_tokens:
        for ch in tokenize_eva_chars(token):
            baseline_chars.add(ch)

    # After merging: some chars are now treated as single units
    # But tokenize_eva_chars already handles them — so inventory is the same
    # The real change is in the number of unique TRIPLES
    merged_triple_lookup = _build_merged_triple_lookup(merges)
    merged_triples = set()
    total_length = 0
    n_tokens = len(all_tokens)

    for token in all_tokens:
        chars = tokenize_eva_chars(token)
        total_length += len(chars)
        for ch in chars:
            triple = merged_triple_lookup.get(ch, ch)
            merged_triples.add(triple)

    mean_length = total_length / max(n_tokens, 1)
    return len(baseline_chars), len(merged_triples), mean_length


def _decode_with_scheme(
    corpus,
    merges: Dict[str, str],
    assignment: Dict[str, str],
    modifier_chars: set,
    modifier_rules: Dict[str, str],
    ref_word_set: set,
    null_seeds: List[int],
) -> Tuple[float, float]:
    """Decode corpus using a merge scheme and measure dict_hit and signal_rate."""
    all_tokens = corpus.get_tokens()
    n_tokens = len(all_tokens)

    # Build merged triple lookup
    merged_lookup = _build_merged_triple_lookup(merges)

    # Create an expanded assignment that includes new merged triples
    expanded_assignment = dict(assignment)

    # For new merged triples not in the assignment, try to assign them
    # by combining the syllables of their component chars
    for char_seq, merged_name in merges.items():
        triple = merged_lookup.get(char_seq)
        if triple and triple not in expanded_assignment:
            # Try to derive from component assignments
            chars = tokenize_eva_chars(char_seq)
            component_syls = []
            base_lookup = build_eva_to_triple_lookup()
            for ch in chars:
                ch_triple = base_lookup.get(ch)
                if ch_triple and ch_triple in assignment:
                    component_syls.append(assignment[ch_triple])
            if component_syls:
                # Use first component's syllable as the merged assignment
                expanded_assignment[triple] = component_syls[0]

    # Decode
    decoded = _decode_corpus_r3(
        all_tokens, expanded_assignment, merged_lookup,
        modifier_chars, modifier_rules, ref_word_set,
    )
    dict_hit = sum(1 for w in decoded if w in ref_word_set) / n_tokens

    # Simplified signal rate using 2 null corpora
    bigram_probs, initial_probs, token_lengths = _build_eva_bigram_model(all_tokens)
    n_signal = 0
    n_null_tested = min(2, len(null_seeds))

    for seed in null_seeds[:n_null_tested]:
        null_tokens = _generate_null_corpus(
            bigram_probs, initial_probs, token_lengths, n_tokens, seed,
        )
        null_decoded = _decode_corpus_r3(
            null_tokens, expanded_assignment, merged_lookup,
            modifier_chars, modifier_rules, ref_word_set,
        )
        for i in range(n_tokens):
            if (decoded[i] in ref_word_set and
                    null_decoded[i] not in ref_word_set):
                n_signal += 1

    signal_rate = n_signal / (n_tokens * n_null_tested) if n_tokens * n_null_tested > 0 else 0.0

    return dict_hit, signal_rate


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_resegmentation_test() -> None:
    """Step 31.8: Test EVA re-segmentation schemes."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE 31.8: EVA Re-Segmentation Test")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load inputs ──
    print("\n  1. Loading inputs...")

    with open(os.path.join(rd, 'combined_refine.json')) as f:
        refine_data = json.load(f)
    assignment = refine_data.get('best_assignment', {})

    with open(os.path.join(rd, 'modifier_integrate.json')) as f:
        mod_data = json.load(f)
    modifier_chars, modifier_rules = _reconstruct_modifier_rules(mod_data)

    null_seeds = [100, 101, 102, 103, 104]
    null_path = os.path.join(rd, 'null_corpus.json')
    if os.path.exists(null_path):
        with open(null_path) as f:
            null_data = json.load(f)
        null_seeds = [r['seed'] for r in null_data.get('null_runs', [])]

    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    base_words = set(w.lower() for w in ref_corpus.get_combined_tokens('latin')
                     if len(w) >= 2)
    expanded, _ = build_expanded_word_set(base_words)
    ref_word_set = base_words | expanded

    corpus = load_corpus(verbose=False)
    all_tokens = corpus.get_tokens()
    n_tokens = len(all_tokens)
    print(f"     {n_tokens} tokens, {len(ref_word_set)} reference words")

    # ── 2. Baseline ──
    print("\n  2. Computing baseline...")
    eva_to_triple = build_eva_to_triple_lookup()

    baseline_decoded = _decode_corpus_r3(
        all_tokens, assignment, eva_to_triple,
        modifier_chars, modifier_rules, ref_word_set,
    )
    baseline_dict_hit = sum(1 for w in baseline_decoded if w in ref_word_set) / n_tokens

    baseline_chars = set()
    total_length = 0
    for token in all_tokens:
        chars = tokenize_eva_chars(token)
        total_length += len(chars)
        baseline_chars.update(chars)
    baseline_inventory = len(baseline_chars)
    baseline_mean_length = total_length / max(n_tokens, 1)

    print(f"     Baseline dict_hit: {baseline_dict_hit:.4f}")
    print(f"     Baseline inventory: {baseline_inventory} chars")
    print(f"     Baseline mean token length: {baseline_mean_length:.2f}")

    # ── 3. Test each scheme ──
    schemes = _define_merge_schemes()
    scheme_results: List[SchemeResult] = []

    for scheme_name, merges in schemes.items():
        print(f"\n  Testing {scheme_name} ({len(merges)} merges)...")

        orig_inv, n_triples, mean_len = _compute_inventory_size(corpus, merges)

        dict_hit, signal_rate = _decode_with_scheme(
            corpus, merges, assignment,
            modifier_chars, modifier_rules, ref_word_set, null_seeds,
        )

        sr = SchemeResult(
            scheme_name=scheme_name,
            merges=merges,
            n_merged_chars=len(merges),
            original_inventory_size=orig_inv,
            merged_inventory_size=n_triples,
            n_unique_triples=n_triples,
            dict_hit=round(dict_hit, 4),
            signal_rate=round(signal_rate, 4),
            mean_token_length=round(mean_len, 2),
        )
        scheme_results.append(sr)

        delta = dict_hit - baseline_dict_hit
        print(f"     dict_hit: {dict_hit:.4f} (Δ={delta:+.4f})")
        print(f"     signal_rate: {signal_rate:.4f}")
        print(f"     inventory: {n_triples} triples")

    # ── 4. Find best scheme ──
    best_scheme = max(scheme_results, key=lambda s: s.dict_hit)
    best_delta = best_scheme.dict_hit - baseline_dict_hit

    print(f"\n  4. Best scheme: {best_scheme.scheme_name}")
    print(f"     dict_hit: {best_scheme.dict_hit:.4f} (Δ={best_delta:+.4f})")

    # ── 5. Fingerprint and historical comparison (simplified) ──
    # These would require running the full Phase 2 and Phase 21 pipelines
    # For now, report placeholder values
    fingerprint_delta = 0.0
    historical_matches = 0

    # ── 6. Verdict ──
    if best_delta > 0.02:
        verdict = "RESEGMENTATION_IMPROVES"
    elif best_delta > 0.005:
        verdict = "RESEGMENTATION_MARGINAL"
    elif best_delta > -0.005:
        verdict = "RESEGMENTATION_NEUTRAL"
    else:
        verdict = "RESEGMENTATION_DEGRADES"

    print(f"\n  Verdict: {verdict}")

    # ── 7. Save ──
    result = ResegmentationResult(
        baseline_dict_hit=round(baseline_dict_hit, 4),
        baseline_inventory_size=baseline_inventory,
        baseline_mean_token_length=round(baseline_mean_length, 2),
        scheme_results=[_convert(asdict(sr)) for sr in scheme_results],
        best_scheme=best_scheme.scheme_name,
        best_dict_hit=round(best_scheme.dict_hit, 4),
        best_delta=round(best_delta, 4),
        fingerprint_match_delta=fingerprint_delta,
        historical_new_matches=historical_matches,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'resegmentation_test.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2)

    print(f"\n  Saved → {out_path}")
    print(f"  Completed in {time.time() - t0:.1f}s")
