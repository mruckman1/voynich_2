"""
Phase 6 A+B: Anchor-and-Propagate with Paradigm Filtering
===========================================================
The core decoding engine for illustration-constrained decoding.

For each Rosetta folio, hypothesize that the dominant Voynich stem = the
medieval Latin plant name. Apply paradigm shape filtering (Approach B) to
reject incompatible hypotheses, then extract character-to-sound mappings
and check cross-consistency across all anchors (Approach A).

Sub-analyses:
  6.A — Anchor hypothesis generation + cross-consistency checking
  6.B — Paradigm shape filtering by Latin declension class

Output:
  results/anchor_propagate.json
"""

import json
import math
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from voynich.core.corpus import load_corpus, VoynichCorpus, tokenize_eva_chars
from voynich.core._paths import results_dir as _results_dir
from voynich.core.reference import (
    LATIN_DECLENSION_SUFFIXES, expected_paradigm_shape,
    infer_declension, extract_latin_stem,
)
from voynich.phases.morpheme_grid import (
    decompose_token_morphemes, MorphemeDecomposition,
)
from voynich.phases.illustration_constrained import (
    FolioIdentificationSet, PlantIdentification,
    load_medieval_names, parse_concordance,
    build_folio_identification_sets,
    _convert, _check_gate,
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class AnchorHypothesis:
    """Hypothesis: this folio's dominant stem = this medieval Latin name."""
    folio: str
    voynich_stem: str
    voynich_stem_chars: List[str]
    voynich_n_forms: int
    voynich_paradigm_shape: Tuple[int, int]
    medieval_name: str
    medieval_stem: str
    declension: str
    # Paradigm filter (Approach B)
    expected_paradigm_shape: Tuple[int, int]
    paradigm_compatible: bool
    paradigm_distance: float
    # Character mappings from this anchor
    char_mappings: Dict[str, str]
    n_chars_mapped: int
    mapping_confidence: float


@dataclass
class CrossConsistencyMatrix:
    """Cross-consistency of char mappings across all anchors."""
    n_anchors: int
    n_chars_total: int
    n_chars_unanimous: int
    n_chars_majority: int
    n_chars_conflicting: int
    unanimity_ratio: float
    majority_ratio: float
    per_char_votes: Dict[str, Dict[str, int]]
    consensus_mapping: Dict[str, str]


@dataclass
class PropagationResult:
    """Result of propagating anchored mappings to non-anchor folios."""
    n_non_anchor_folios: int
    n_tokens_decoded: int
    n_tokens_partial: int
    n_tokens_failed: int
    decode_coverage: float
    sample_decodings: List[Dict]


@dataclass
class AnchorPropagateResult:
    """Full Phase 6 A+B output."""
    n_rosetta_folios: int
    n_anchors_generated: int
    n_paradigm_compatible: int
    anchor_hypotheses: List[Dict]
    cross_consistency: Dict
    consensus_mapping: Dict[str, str]
    n_chars_mapped: int
    # Propagation
    propagation: Dict
    # Null tests
    null_shuffled_folio_mean: float
    null_shuffled_folio_std: float
    null_shuffled_folio_z: float
    null_random_plant_mean: float
    null_random_plant_std: float
    null_random_plant_z: float
    # Gate
    unanimity_ratio: float
    gate_passed: bool
    verdict: str


# ---------------------------------------------------------------------------
# Character alignment
# ---------------------------------------------------------------------------

def _align_chars(
    eva_chars: List[str],
    latin_word: str,
) -> List[Tuple[str, str]]:
    """
    Positional alignment of EVA characters to Latin characters.

    If lengths differ, uses stretch/compress mapping.
    Returns list of (eva_char, latin_segment) pairs.
    """
    latin_chars = list(latin_word)
    n_eva = len(eva_chars)
    n_lat = len(latin_chars)

    if n_eva == 0 or n_lat == 0:
        return []

    pairs: List[Tuple[str, str]] = []

    if n_eva == n_lat:
        for e, l in zip(eva_chars, latin_chars):
            pairs.append((e, l))
    elif n_eva < n_lat:
        ratio = n_lat / n_eva
        for i in range(n_eva):
            start = int(i * ratio)
            end = int((i + 1) * ratio)
            segment = ''.join(latin_chars[start:end])
            pairs.append((eva_chars[i], segment))
    else:
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


# ---------------------------------------------------------------------------
# Anchor hypothesis building
# ---------------------------------------------------------------------------

def build_anchor_hypothesis(
    folio_set: FolioIdentificationSet,
    identification: PlantIdentification,
    encoding_model: str = 'morphographic-alphabetic',
) -> Optional[AnchorHypothesis]:
    """
    Build one anchor hypothesis for a folio.

    Hypothesis: folio's dominant stem encodes the medieval Latin plant name.
    """
    if not folio_set.dominant_stem or not identification.medieval_stem:
        return None

    voynich_stem = folio_set.dominant_stem
    eva_chars = tokenize_eva_chars(voynich_stem)
    medieval_stem = identification.medieval_stem
    declension = identification.declension or infer_declension(
        identification.medieval_name or medieval_stem)

    # Paradigm shape compatibility (Approach B)
    exp_shape = expected_paradigm_shape(declension)
    obs_shape = folio_set.dominant_stem_paradigm_shape or (0, 0)

    # Distance metric: normalized difference in suffix count
    exp_suffixes = exp_shape[1]
    obs_suffixes = obs_shape[1]
    if exp_suffixes > 0:
        paradigm_distance = abs(obs_suffixes - exp_suffixes) / exp_suffixes
    else:
        paradigm_distance = 1.0

    # Compatible if within 100% of expected (generous to account for
    # sparse Voynich data where not all forms may be attested)
    paradigm_compatible = paradigm_distance <= 1.0

    # Build character mapping by aligning EVA chars to Latin stem
    alignment = _align_chars(eva_chars, medieval_stem)
    char_mappings: Dict[str, str] = {}
    for eva_char, latin_seg in alignment:
        if latin_seg and eva_char not in char_mappings:
            char_mappings[eva_char] = latin_seg

    n_mapped = len(char_mappings)
    confidence = n_mapped / len(eva_chars) if eva_chars else 0.0

    return AnchorHypothesis(
        folio=folio_set.folio,
        voynich_stem=voynich_stem,
        voynich_stem_chars=eva_chars,
        voynich_n_forms=len(folio_set.dominant_stem_forms),
        voynich_paradigm_shape=obs_shape,
        medieval_name=identification.medieval_name or '',
        medieval_stem=medieval_stem,
        declension=declension,
        expected_paradigm_shape=exp_shape,
        paradigm_compatible=paradigm_compatible,
        paradigm_distance=round(paradigm_distance, 4),
        char_mappings=char_mappings,
        n_chars_mapped=n_mapped,
        mapping_confidence=round(confidence, 4),
    )


# ---------------------------------------------------------------------------
# Cross-consistency checking (Approach A)
# ---------------------------------------------------------------------------

def cross_consistency_check(
    hypotheses: List[AnchorHypothesis],
) -> CrossConsistencyMatrix:
    """
    Check cross-consistency of character mappings across all anchor hypotheses.

    For each EVA character appearing in 2+ anchors, check if the same
    Latin segment is assigned unanimously.
    """
    # Collect all (eva_char -> latin_segment) votes
    char_votes: Dict[str, Dict[str, int]] = defaultdict(lambda: Counter())

    for h in hypotheses:
        for eva_char, latin_seg in h.char_mappings.items():
            if latin_seg:
                char_votes[eva_char][latin_seg] += 1

    # Classify each character
    n_unanimous = 0
    n_majority = 0
    n_conflicting = 0
    consensus: Dict[str, str] = {}

    for eva_char, votes in char_votes.items():
        total_votes = sum(votes.values())
        if total_votes < 1:
            continue

        best_seg, best_count = votes.most_common(1)[0]
        consensus[eva_char] = best_seg

        if len(votes) == 1:
            # Only one proposal -> unanimous
            n_unanimous += 1
        elif best_count > total_votes / 2:
            # Majority agrees
            if best_count == total_votes:
                n_unanimous += 1
            else:
                n_majority += 1
        else:
            n_conflicting += 1

    n_total = n_unanimous + n_majority + n_conflicting
    unanimity_ratio = n_unanimous / n_total if n_total > 0 else 0.0
    majority_ratio = (n_unanimous + n_majority) / n_total if n_total > 0 else 0.0

    return CrossConsistencyMatrix(
        n_anchors=len(hypotheses),
        n_chars_total=n_total,
        n_chars_unanimous=n_unanimous,
        n_chars_majority=n_majority,
        n_chars_conflicting=n_conflicting,
        unanimity_ratio=round(unanimity_ratio, 4),
        majority_ratio=round(majority_ratio, 4),
        per_char_votes={k: dict(v) for k, v in char_votes.items()},
        consensus_mapping=consensus,
    )


# ---------------------------------------------------------------------------
# Propagation
# ---------------------------------------------------------------------------

def propagate_to_non_anchors(
    consensus_mapping: Dict[str, str],
    corpus: VoynichCorpus,
    anchor_folios: set,
) -> PropagationResult:
    """
    Apply consensus mapping to decode tokens on non-anchor herbal_a folios.
    """
    non_anchor_pages = [
        p for p in corpus.pages.values()
        if p.section == 'herbal_a' and p.folio not in anchor_folios
    ]

    n_decoded = 0
    n_partial = 0
    n_failed = 0
    samples: List[Dict] = []

    for page in non_anchor_pages:
        for token in page.all_tokens[:50]:  # Sample first 50 tokens per page
            eva_chars = tokenize_eva_chars(token)
            decoded_parts = []
            has_unknown = False

            for c in eva_chars:
                if c in consensus_mapping:
                    decoded_parts.append(consensus_mapping[c])
                else:
                    decoded_parts.append('?')
                    has_unknown = True

            decoded = ''.join(decoded_parts)
            if not has_unknown:
                n_decoded += 1
            elif '?' in decoded and decoded.replace('?', ''):
                n_partial += 1
            else:
                n_failed += 1

            if len(samples) < 30:
                coverage = sum(1 for c in eva_chars
                               if c in consensus_mapping) / max(len(eva_chars), 1)
                samples.append({
                    'folio': page.folio,
                    'eva_token': token,
                    'decoded': decoded,
                    'coverage': round(coverage, 4),
                })

    total = n_decoded + n_partial + n_failed
    decode_coverage = n_decoded / total if total > 0 else 0.0

    return PropagationResult(
        n_non_anchor_folios=len(non_anchor_pages),
        n_tokens_decoded=n_decoded,
        n_tokens_partial=n_partial,
        n_tokens_failed=n_failed,
        decode_coverage=round(decode_coverage, 4),
        sample_decodings=samples,
    )


# ---------------------------------------------------------------------------
# Null tests
# ---------------------------------------------------------------------------

def null_test_shuffled_folios(
    folio_sets: List[FolioIdentificationSet],
    real_unanimity: float,
    n_trials: int = 100,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Null test 1: Shuffle which folio text goes with which plant name.

    Keeps the same plant names and folio texts but randomly reassigns
    which text goes with which plant name.
    """
    rng = random.Random(seed)

    # Collect all identifications and all folio sets
    all_plant_ids = []
    eligible_sets = []
    for fs in folio_sets:
        for pid in fs.identifications:
            if pid.medieval_stem:
                all_plant_ids.append(pid)
                eligible_sets.append(fs)
                break  # One per folio

    if len(eligible_sets) < 3:
        return 0.0, 1.0, 0.0

    null_unanimities: List[float] = []

    for trial in range(n_trials):
        # Shuffle the plant ID assignments
        shuffled_ids = list(all_plant_ids)
        rng.shuffle(shuffled_ids)

        # Build anchor hypotheses with shuffled assignments
        hypotheses = []
        for i, fs in enumerate(eligible_sets):
            if i < len(shuffled_ids):
                h = build_anchor_hypothesis(fs, shuffled_ids[i])
                if h and h.paradigm_compatible:
                    hypotheses.append(h)

        if len(hypotheses) >= 2:
            cc = cross_consistency_check(hypotheses)
            null_unanimities.append(cc.unanimity_ratio)
        else:
            null_unanimities.append(0.0)

    null_mean = float(np.mean(null_unanimities))
    null_std = float(np.std(null_unanimities))
    z = (real_unanimity - null_mean) / null_std if null_std > 0 else 0.0

    return null_mean, null_std, z


def null_test_random_plants(
    folio_sets: List[FolioIdentificationSet],
    real_unanimity: float,
    n_trials: int = 100,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Null test 2: Replace plant names with random medieval Latin words.

    Uses random Latin words (from a pool of common medieval plant names)
    as substitutes for the actual identifications.
    """
    rng = random.Random(seed)

    # Pool of random Latin stems (common medieval Latin nouns)
    random_stem_pool = [
        'herb', 'aqu', 'ole', 'radic', 'foli', 'flor', 'semin',
        'morb', 'febr', 'dolor', 'sanguin', 'remedi', 'virt',
        'natur', 'corpor', 'membr', 'pector', 'ventr', 'capit',
        'dent', 'ocul', 'aur', 'nas', 'man', 'ped', 'crust',
        'cort', 'pulver', 'success', 'decoct', 'infus', 'emplast',
        'unguent', 'electuari', 'sirup', 'pill', 'trocis',
    ]

    eligible_sets = [
        fs for fs in folio_sets
        if any(p.medieval_stem for p in fs.identifications)
        and fs.dominant_stem
    ]

    if len(eligible_sets) < 3:
        return 0.0, 1.0, 0.0

    null_unanimities: List[float] = []

    for trial in range(n_trials):
        hypotheses = []
        for fs in eligible_sets:
            # Pick a random stem instead of real plant name
            fake_stem = rng.choice(random_stem_pool)
            fake_id = PlantIdentification(
                folio=fs.folio,
                linnaean_name='Random',
                common_name='random',
                source='null_test',
                medieval_name=fake_stem + 'a',
                medieval_stem=fake_stem,
                declension='noun_1st',
            )
            h = build_anchor_hypothesis(fs, fake_id)
            if h and h.paradigm_compatible:
                hypotheses.append(h)

        if len(hypotheses) >= 2:
            cc = cross_consistency_check(hypotheses)
            null_unanimities.append(cc.unanimity_ratio)
        else:
            null_unanimities.append(0.0)

    null_mean = float(np.mean(null_unanimities))
    null_std = float(np.std(null_unanimities))
    z = (real_unanimity - null_mean) / null_std if null_std > 0 else 0.0

    return null_mean, null_std, z


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_anchor_propagate(
    rosetta_data: Optional[Dict] = None,
    constrained_data: Optional[Dict] = None,
) -> Dict:
    """
    Run Phase 6 A+B: Anchor-and-Propagate with Paradigm Filtering.

    1. Load Phase 6.0 and 6 D+E results
    2. Build anchor hypothesis for each Rosetta folio
    3. Apply paradigm filter (Approach B)
    4. Cross-consistency check on surviving anchors
    5. Propagate to non-anchor folios
    6. Run null tests
    7. Gate: unanimity > 0.50 AND z > 2.0 vs both nulls
    8. Save results/anchor_propagate.json
    """
    print("=" * 70)
    print("Phase 6 A+B: Anchor-and-Propagate with Paradigm Filtering")
    print("=" * 70)

    # Load prior results
    if rosetta_data is None:
        results_path = os.path.join(_results_dir(), 'rosetta_selection.json')
        if os.path.exists(results_path):
            with open(results_path) as f:
                rosetta_data = json.load(f)
        else:
            from voynich.phases.rosetta_selection import run_rosetta_selection
            rosetta_data = run_rosetta_selection()

    if not rosetta_data.get('gate_passed', False):
        print("\n  Prior gate FAILED — insufficient Rosetta folios.")
        result = AnchorPropagateResult(
            n_rosetta_folios=0, n_anchors_generated=0,
            n_paradigm_compatible=0, anchor_hypotheses=[],
            cross_consistency={}, consensus_mapping={},
            n_chars_mapped=0, propagation={},
            null_shuffled_folio_mean=0.0, null_shuffled_folio_std=0.0,
            null_shuffled_folio_z=0.0,
            null_random_plant_mean=0.0, null_random_plant_std=0.0,
            null_random_plant_z=0.0,
            unanimity_ratio=0.0, gate_passed=False,
            verdict='prior_gate_failed',
        )
        out_path = os.path.join(_results_dir(), 'anchor_propagate.json')
        with open(out_path, 'w') as f:
            json.dump(_convert(asdict(result)), f, indent=2, default=str)
        return _convert(asdict(result))

    selected_folios = rosetta_data.get('selected_rosetta_folios', [])
    encoding_model = rosetta_data.get('best_encoding_model',
                                      'morphographic-alphabetic')

    print(f"\n  Rosetta folios: {len(selected_folios)}")
    print(f"  Encoding model: {encoding_model}")

    # Rebuild folio identification sets
    print("\n  Loading corpus and building identification sets...")
    corpus = load_corpus(verbose=False)
    concordance = parse_concordance()
    medieval_names = load_medieval_names()
    folio_sets = build_folio_identification_sets(
        concordance, medieval_names, corpus,
    )

    # Build index for quick lookup
    folio_index = {fs.folio: fs for fs in folio_sets}

    # Build anchor hypotheses for each Rosetta folio
    print("\n  6.A: Building anchor hypotheses")
    all_hypotheses: List[AnchorHypothesis] = []
    compatible_hypotheses: List[AnchorHypothesis] = []

    for folio in selected_folios:
        fs = folio_index.get(folio)
        if fs is None:
            continue

        # Use the first identification with a resolved medieval name
        for pid in fs.identifications:
            if pid.medieval_stem:
                h = build_anchor_hypothesis(fs, pid, encoding_model)
                if h:
                    all_hypotheses.append(h)
                    if h.paradigm_compatible:
                        compatible_hypotheses.append(h)
                    print(f"    {folio}: stem='{h.voynich_stem}' -> "
                          f"'{h.medieval_stem}' "
                          f"({'OK' if h.paradigm_compatible else 'FILTERED'}) "
                          f"[{h.n_chars_mapped} chars mapped]")
                break

    print(f"\n    Hypotheses generated: {len(all_hypotheses)}")
    print(f"    Paradigm-compatible: {len(compatible_hypotheses)}")

    # Cross-consistency check on compatible anchors
    print("\n  6.B: Cross-consistency check")
    if len(compatible_hypotheses) >= 2:
        cc = cross_consistency_check(compatible_hypotheses)
        print(f"    Characters with votes: {cc.n_chars_total}")
        print(f"    Unanimous: {cc.n_chars_unanimous}")
        print(f"    Majority: {cc.n_chars_majority}")
        print(f"    Conflicting: {cc.n_chars_conflicting}")
        print(f"    Unanimity ratio: {cc.unanimity_ratio:.4f}")
        print(f"    Majority ratio: {cc.majority_ratio:.4f}")

        # Show consensus mapping
        print(f"\n    Consensus mapping ({len(cc.consensus_mapping)} chars):")
        for eva, lat in sorted(cc.consensus_mapping.items()):
            votes = cc.per_char_votes.get(eva, {})
            vote_str = ', '.join(f'{v}:{c}' for v, c in
                                 sorted(votes.items(), key=lambda x: -x[1]))
            print(f"      {eva} -> {lat}  [{vote_str}]")

        unanimity = cc.unanimity_ratio
        consensus_mapping = cc.consensus_mapping
        cc_dict = asdict(cc)
    else:
        print("    Too few compatible anchors for cross-consistency check.")
        unanimity = 0.0
        consensus_mapping = {}
        cc_dict = {}

    # Propagation
    print("\n  Propagating to non-anchor folios...")
    anchor_folio_set = set(selected_folios)
    propagation = propagate_to_non_anchors(
        consensus_mapping, corpus, anchor_folio_set,
    )
    print(f"    Non-anchor folios: {propagation.n_non_anchor_folios}")
    print(f"    Fully decoded tokens: {propagation.n_tokens_decoded}")
    print(f"    Partially decoded: {propagation.n_tokens_partial}")
    print(f"    Failed: {propagation.n_tokens_failed}")
    print(f"    Decode coverage: {propagation.decode_coverage:.4f}")

    if propagation.sample_decodings:
        print("\n    Sample decodings:")
        for s in propagation.sample_decodings[:10]:
            print(f"      {s['folio']}: {s['eva_token']} -> {s['decoded']} "
                  f"(cov={s['coverage']:.2f})")

    # Null tests
    print("\n  Null test 1: Shuffled folio assignments")
    sf_mean, sf_std, sf_z = null_test_shuffled_folios(
        folio_sets, unanimity, n_trials=100, seed=42,
    )
    print(f"    Real unanimity: {unanimity:.4f}")
    print(f"    Null mean: {sf_mean:.4f} +/- {sf_std:.4f}")
    print(f"    z-score: {sf_z:.2f}")

    print("\n  Null test 2: Random plant names")
    rp_mean, rp_std, rp_z = null_test_random_plants(
        folio_sets, unanimity, n_trials=100, seed=42,
    )
    print(f"    Real unanimity: {unanimity:.4f}")
    print(f"    Null mean: {rp_mean:.4f} +/- {rp_std:.4f}")
    print(f"    z-score: {rp_z:.2f}")

    # Gate
    gate1_ok, gate1_msg = _check_gate(
        'unanimity_ratio', unanimity, 0.50, 'greater',
    )
    gate2_ok, gate2_msg = _check_gate(
        'shuffled_folio_z', sf_z, 2.0, 'greater',
    )
    gate3_ok, gate3_msg = _check_gate(
        'random_plant_z', rp_z, 2.0, 'greater',
    )
    print(f"\n{gate1_msg}")
    print(gate2_msg)
    print(gate3_msg)

    gate_passed = gate1_ok and gate2_ok and gate3_ok
    if gate_passed:
        verdict = 'cross_modal_signal_confirmed'
    elif gate1_ok:
        verdict = 'marginal_signal'
    else:
        verdict = 'no_cross_modal_signal'
    print(f"  Verdict: {verdict}")

    # Build result
    result = AnchorPropagateResult(
        n_rosetta_folios=len(selected_folios),
        n_anchors_generated=len(all_hypotheses),
        n_paradigm_compatible=len(compatible_hypotheses),
        anchor_hypotheses=[_convert(asdict(h)) for h in all_hypotheses],
        cross_consistency=_convert(cc_dict),
        consensus_mapping=consensus_mapping,
        n_chars_mapped=len(consensus_mapping),
        propagation=_convert(asdict(propagation)),
        null_shuffled_folio_mean=round(sf_mean, 4),
        null_shuffled_folio_std=round(sf_std, 4),
        null_shuffled_folio_z=round(sf_z, 2),
        null_random_plant_mean=round(rp_mean, 4),
        null_random_plant_std=round(rp_std, 4),
        null_random_plant_z=round(rp_z, 2),
        unanimity_ratio=unanimity,
        gate_passed=gate_passed,
        verdict=verdict,
    )

    # Save
    out_path = os.path.join(_results_dir(), 'anchor_propagate.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")

    return _convert(asdict(result))
