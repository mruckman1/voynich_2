"""
Phase 6.1 Fix C: Encoding Model Diagnosis
===========================================
Per-anchor encoding model fit analysis, segmentation sensitivity testing,
and hybrid model detection.

The encoding model determines how to segment a Voynich stem into
character-to-sound units. If the model is wrong, even correct anchor
identifications produce inconsistent mappings.

Sub-analyses:
  C.1 — Per-anchor encoding model fit
  C.2 — Model consensus analysis
  C.3 — Segmentation sensitivity test
  C.4 — Hybrid model test (different models for different word lengths)

Output:
  results/encoding_diagnosis.json
"""

import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from voynich.core.corpus import load_corpus, VoynichCorpus, tokenize_eva_chars
from voynich.core._paths import results_dir as _results_dir
from voynich.core.reference import infer_declension
from voynich.phases.illustration_constrained import (
    FolioIdentificationSet, PlantIdentification,
    load_medieval_names, parse_concordance,
    build_folio_identification_sets,
    _convert, _check_gate,
)
from voynich.phases.anchor_propagate import (
    build_anchor_hypothesis, cross_consistency_check,
    AnchorHypothesis, CrossConsistencyMatrix,
    _align_chars,
)
from voynich.phases.morpheme_grid import decompose_token_morphemes
from voynich.analysis.strokes import decompose_glyph


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PerAnchorModelFit:
    """Encoding model fit for one anchor under one model."""
    folio: str
    plant_name: str
    eva_stem: str
    medieval_stem: str
    observed_chars: int
    expected_units: int
    fit: float  # |observed - expected| / expected; 0 = perfect


@dataclass
class ModelConsensus:
    """Consensus analysis for one encoding model."""
    model_name: str
    description: str
    n_good_fit: int  # anchors with fit < 0.3
    n_total: int
    mean_fit: float
    per_anchor: List[Dict]


@dataclass
class SegmentationResult:
    """Result of testing one segmentation approach."""
    segmentation_name: str
    description: str
    unanimity: float
    n_chars_mapped: int
    delta_from_baseline: float


@dataclass
class HybridModelResult:
    """Result of hybrid model test for one word-length category."""
    category: str
    n_anchors: int
    best_model: str
    best_fit: float
    anchors: List[str]


@dataclass
class EncodingDiagnosisResult:
    """Full encoding diagnosis output."""
    # C.1: Per-anchor model fit
    model_fits: List[Dict]  # One entry per model
    # C.2: Model consensus
    winner: str
    winner_good_fit: int
    winner_total: int
    clear_winner: bool
    # C.3: Segmentation sensitivity
    segmentation_results: List[Dict]
    best_segmentation: str
    best_segmentation_unanimity: float
    # C.4: Hybrid model test
    hybrid_results: List[Dict]
    hybrid_evidence: bool
    # Gate
    verdict: str


# ---------------------------------------------------------------------------
# Syllable counting
# ---------------------------------------------------------------------------

def _syllable_count(word: str) -> int:
    """Estimate syllable count for a Latin word (vowel-cluster heuristic)."""
    vowels = set('aeiouy')
    count = 0
    in_vowel = False
    for c in word.lower():
        if c in vowels:
            if not in_vowel:
                count += 1
            in_vowel = True
        else:
            in_vowel = False
    return max(1, count)


def _phoneme_count(word: str) -> int:
    """Rough phoneme count for a Latin word."""
    # Latin is roughly phonemic — each letter is ~1 phoneme,
    # except digraphs (ph, ch, th, qu) which are ~1 phoneme each
    word_l = word.lower()
    count = 0
    i = 0
    digraphs = {'ph', 'ch', 'th', 'qu', 'ae', 'oe'}
    while i < len(word_l):
        if i + 1 < len(word_l) and word_l[i:i+2] in digraphs:
            count += 1
            i += 2
        else:
            count += 1
            i += 1
    return max(1, count)


def _stroke_count(eva_stem: str) -> int:
    """Count total strokes in an EVA stem."""
    chars = tokenize_eva_chars(eva_stem)
    total = 0
    for c in chars:
        strokes = decompose_glyph(c)
        total += len(strokes) if strokes else 1
    return max(1, total)


# ---------------------------------------------------------------------------
# C.1: Per-anchor encoding model fit
# ---------------------------------------------------------------------------

def compute_per_anchor_model_fits(
    hypotheses: List[AnchorHypothesis],
) -> Dict[str, ModelConsensus]:
    """
    Test 4 encoding models against each anchor individually.

    Returns {model_name: ModelConsensus}.
    """
    models = {
        'morphographic-syllabic': {
            'description': '1 EVA char = 1 Latin syllable',
            'expected_fn': lambda stem: _syllable_count(stem),
        },
        'morphographic-alphabetic': {
            'description': '1 EVA char = 1 Latin letter',
            'expected_fn': lambda stem: len(stem),
        },
        'morphographic-abbreviated': {
            'description': 'EVA chars = abbreviated Latin (half letter count)',
            'expected_fn': lambda stem: max(2, len(stem) // 2),
        },
        'mixed': {
            'description': 'Hybrid encoding (~70% of letter count)',
            'expected_fn': lambda stem: max(2, int(len(stem) * 0.7)),
        },
    }

    results: Dict[str, ModelConsensus] = {}

    for model_name, model_info in models.items():
        per_anchor: List[PerAnchorModelFit] = []
        for h in hypotheses:
            observed = len(h.voynich_stem_chars)
            expected = model_info['expected_fn'](h.medieval_stem)
            fit = abs(observed - expected) / max(expected, 1)

            per_anchor.append(PerAnchorModelFit(
                folio=h.folio,
                plant_name=h.medieval_name,
                eva_stem=h.voynich_stem,
                medieval_stem=h.medieval_stem,
                observed_chars=observed,
                expected_units=expected,
                fit=round(fit, 4),
            ))

        n_good = sum(1 for a in per_anchor if a.fit < 0.3)
        mean_fit = float(np.mean([a.fit for a in per_anchor])) if per_anchor else 999.0

        results[model_name] = ModelConsensus(
            model_name=model_name,
            description=model_info['description'],
            n_good_fit=n_good,
            n_total=len(per_anchor),
            mean_fit=round(mean_fit, 4),
            per_anchor=[asdict(a) for a in per_anchor],
        )

    return results


# ---------------------------------------------------------------------------
# C.2: Model consensus analysis
# ---------------------------------------------------------------------------

def analyze_model_consensus(
    model_fits: Dict[str, ModelConsensus],
) -> Tuple[str, bool]:
    """
    Determine the winning model and whether there's a clear winner.

    Returns (winner_name, is_clear_winner).
    A clear winner has > 6/N anchors fitting well.
    """
    # Rank by n_good_fit descending, then by mean_fit ascending
    ranked = sorted(
        model_fits.values(),
        key=lambda m: (-m.n_good_fit, m.mean_fit),
    )

    winner = ranked[0]
    n_total = winner.n_total

    # Clear winner: >75% of anchors fit well
    clear = winner.n_good_fit > n_total * 0.75 if n_total > 0 else False

    return winner.model_name, clear


# ---------------------------------------------------------------------------
# C.3: Segmentation sensitivity test
# ---------------------------------------------------------------------------

def _segment_greedy_left(eva_chars: List[str], n_segments: int) -> List[List[str]]:
    """Each character = one segment (standard)."""
    return [[c] for c in eva_chars]


def _segment_balanced(eva_chars: List[str], n_segments: int) -> List[List[str]]:
    """Distribute characters as evenly as possible across segments."""
    if n_segments <= 0 or not eva_chars:
        return [[c] for c in eva_chars]
    n_segments = min(n_segments, len(eva_chars))
    base_size = len(eva_chars) // n_segments
    remainder = len(eva_chars) % n_segments
    segments = []
    idx = 0
    for i in range(n_segments):
        size = base_size + (1 if i < remainder else 0)
        segments.append(eva_chars[idx:idx + size])
        idx += size
    return segments


def _segment_greedy_right(eva_chars: List[str], n_segments: int) -> List[List[str]]:
    """Last segment gets extra characters."""
    if n_segments <= 0 or not eva_chars:
        return [[c] for c in eva_chars]
    n_segments = min(n_segments, len(eva_chars))
    if n_segments == len(eva_chars):
        return [[c] for c in eva_chars]
    # First n-1 segments get 1 char each, last gets the rest
    segments = [[c] for c in eva_chars[:n_segments - 1]]
    segments.append(eva_chars[n_segments - 1:])
    return segments


def _build_segmented_mapping(
    hypothesis: AnchorHypothesis,
    segments: List[List[str]],
    latin_word: str,
    model_name: str,
) -> Dict[str, str]:
    """
    Build character-to-sound mappings under a given segmentation.

    Maps each segment of EVA chars to a corresponding segment of the
    Latin word based on the encoding model.
    """
    if model_name == 'morphographic-syllabic':
        # Split Latin into syllables
        latin_parts = _split_into_syllables(latin_word, len(segments))
    else:
        # Split Latin into equal-ish chunks
        latin_parts = _split_into_chunks(latin_word, len(segments))

    mappings: Dict[str, str] = {}
    for seg, lat_part in zip(segments, latin_parts):
        # The primary EVA char of this segment maps to the Latin part
        primary_char = seg[0]
        if primary_char not in mappings and lat_part:
            mappings[primary_char] = lat_part

    return mappings


def _split_into_syllables(word: str, n: int) -> List[str]:
    """Split a word into approximately n syllable-like chunks."""
    vowels = set('aeiouy')
    boundaries = [0]

    # Find syllable boundaries (before each vowel cluster after the first)
    i = 0
    while i < len(word):
        if word[i].lower() in vowels:
            # Skip the vowel cluster
            while i < len(word) and word[i].lower() in vowels:
                i += 1
            if i < len(word):
                boundaries.append(i)
        else:
            i += 1

    # If we have more boundaries than needed, merge some
    while len(boundaries) > n:
        # Find the two closest boundaries and merge
        min_gap = float('inf')
        min_idx = 0
        for j in range(1, len(boundaries)):
            gap = boundaries[j] - boundaries[j - 1]
            if gap < min_gap:
                min_gap = gap
                min_idx = j
        boundaries.pop(min_idx)

    # If we have fewer boundaries than needed, split longest chunks
    while len(boundaries) < n and len(boundaries) < len(word):
        # Find longest chunk and split it
        max_len = 0
        max_idx = 0
        for j in range(len(boundaries)):
            end = boundaries[j + 1] if j + 1 < len(boundaries) else len(word)
            chunk_len = end - boundaries[j]
            if chunk_len > max_len:
                max_len = chunk_len
                max_idx = j
        end = (boundaries[max_idx + 1]
               if max_idx + 1 < len(boundaries) else len(word))
        mid = boundaries[max_idx] + (end - boundaries[max_idx]) // 2
        boundaries.insert(max_idx + 1, mid)

    # Build chunks
    parts = []
    for j in range(len(boundaries)):
        start = boundaries[j]
        end = boundaries[j + 1] if j + 1 < len(boundaries) else len(word)
        parts.append(word[start:end])

    return parts


def _split_into_chunks(word: str, n: int) -> List[str]:
    """Split a word into n roughly equal chunks."""
    if n <= 0 or not word:
        return [word] if word else []
    n = min(n, len(word))
    chunk_size = len(word) / n
    parts = []
    for i in range(n):
        start = int(i * chunk_size)
        end = int((i + 1) * chunk_size)
        parts.append(word[start:end])
    return parts


def test_segmentation_sensitivity(
    hypotheses: List[AnchorHypothesis],
    winning_model: str,
    baseline_unanimity: float,
) -> List[SegmentationResult]:
    """
    Test how sensitive consistency results are to segmentation choices.

    Under the winning model, try multiple segmentation strategies and
    check which produces the highest unanimity.
    """
    strategies = {
        'greedy_left': ('Each EVA char = one segment (standard)', _segment_greedy_left),
        'balanced': ('Characters distributed evenly across segments', _segment_balanced),
        'greedy_right': ('Last segment gets extra characters', _segment_greedy_right),
    }

    results: List[SegmentationResult] = []

    for seg_name, (description, seg_fn) in strategies.items():
        # Rebuild all hypotheses with this segmentation
        modified_hypotheses: List[AnchorHypothesis] = []

        for h in hypotheses:
            eva_chars = h.voynich_stem_chars
            medieval_stem = h.medieval_stem

            if winning_model == 'morphographic-syllabic':
                n_segments = _syllable_count(medieval_stem)
            elif winning_model == 'morphographic-alphabetic':
                n_segments = len(medieval_stem)
            elif winning_model == 'morphographic-abbreviated':
                n_segments = max(2, len(medieval_stem) // 2)
            else:
                n_segments = max(2, int(len(medieval_stem) * 0.7))

            segments = seg_fn(eva_chars, n_segments)
            new_mappings = _build_segmented_mapping(
                h, segments, medieval_stem, winning_model,
            )

            # Create a modified hypothesis with new mappings
            modified_h = AnchorHypothesis(
                folio=h.folio,
                voynich_stem=h.voynich_stem,
                voynich_stem_chars=h.voynich_stem_chars,
                voynich_n_forms=h.voynich_n_forms,
                voynich_paradigm_shape=h.voynich_paradigm_shape,
                medieval_name=h.medieval_name,
                medieval_stem=h.medieval_stem,
                declension=h.declension,
                expected_paradigm_shape=h.expected_paradigm_shape,
                paradigm_compatible=h.paradigm_compatible,
                paradigm_distance=h.paradigm_distance,
                char_mappings=new_mappings,
                n_chars_mapped=len(new_mappings),
                mapping_confidence=len(new_mappings) / max(len(eva_chars), 1),
            )
            modified_hypotheses.append(modified_h)

        # Check cross-consistency
        if len(modified_hypotheses) >= 2:
            cc = cross_consistency_check(modified_hypotheses)
            unanimity = cc.unanimity_ratio
            n_chars = cc.n_chars_total
        else:
            unanimity = 0.0
            n_chars = 0

        results.append(SegmentationResult(
            segmentation_name=seg_name,
            description=description,
            unanimity=round(unanimity, 4),
            n_chars_mapped=n_chars,
            delta_from_baseline=round(unanimity - baseline_unanimity, 4),
        ))

    return results


# ---------------------------------------------------------------------------
# C.4: Hybrid model test
# ---------------------------------------------------------------------------

def test_hybrid_model(
    hypotheses: List[AnchorHypothesis],
    model_fits: Dict[str, ModelConsensus],
) -> Tuple[List[HybridModelResult], bool]:
    """
    Test whether different word lengths fit different encoding models.

    Separates anchors into short (1-2 syllables), medium (3),
    and long (4+) plant names, then finds the best model per category.
    """
    # Categorize anchors by Latin name syllable count
    categories: Dict[str, List[AnchorHypothesis]] = {
        'short': [],   # 1-2 syllables
        'medium': [],  # 3 syllables
        'long': [],    # 4+ syllables
    }

    for h in hypotheses:
        sylls = _syllable_count(h.medieval_stem)
        if sylls <= 2:
            categories['short'].append(h)
        elif sylls == 3:
            categories['medium'].append(h)
        else:
            categories['long'].append(h)

    results: List[HybridModelResult] = []
    model_names = list(model_fits.keys())

    for cat_name, cat_hypotheses in categories.items():
        if not cat_hypotheses:
            results.append(HybridModelResult(
                category=cat_name,
                n_anchors=0,
                best_model='none',
                best_fit=999.0,
                anchors=[],
            ))
            continue

        # For each model, compute mean fit for just this category's anchors
        best_model = 'none'
        best_fit = 999.0

        for model_name, consensus in model_fits.items():
            cat_folios = {h.folio for h in cat_hypotheses}
            cat_fits = [
                a['fit'] for a in consensus.per_anchor
                if a['folio'] in cat_folios
            ]
            if cat_fits:
                mean_fit = float(np.mean(cat_fits))
                if mean_fit < best_fit:
                    best_fit = mean_fit
                    best_model = model_name

        results.append(HybridModelResult(
            category=cat_name,
            n_anchors=len(cat_hypotheses),
            best_model=best_model,
            best_fit=round(best_fit, 4),
            anchors=[h.folio for h in cat_hypotheses],
        ))

    # Hybrid evidence: different categories prefer different models
    active_results = [r for r in results if r.n_anchors > 0]
    models_used = set(r.best_model for r in active_results)
    hybrid_evidence = len(models_used) > 1 and len(active_results) >= 2

    return results, hybrid_evidence


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_encoding_diagnosis(
    anchor_data: Optional[Dict] = None,
    use_tfidf: bool = False,
) -> Dict:
    """
    Run Phase 6.1 Fix C: Encoding Model Diagnosis.

    1. Load anchor-propagate results and rebuild hypotheses
    2. C.1: Per-anchor encoding model fit
    3. C.2: Model consensus analysis
    4. C.3: Segmentation sensitivity test
    5. C.4: Hybrid model test
    6. Report verdict
    7. Save results/encoding_diagnosis.json
    """
    print("=" * 70)
    print("Phase 6.1 Fix C: Encoding Model Diagnosis")
    print("=" * 70)

    # Load prior results
    if anchor_data is None:
        results_path = os.path.join(_results_dir(), 'anchor_propagate.json')
        if os.path.exists(results_path):
            with open(results_path) as f:
                anchor_data = json.load(f)
        else:
            print("\n  No anchor_propagate.json found. Run 'voynich anchor' first.")
            return {}

    baseline_unanimity = anchor_data.get('unanimity_ratio', 0.0)
    print(f"\n  Baseline unanimity: {baseline_unanimity:.4f}")

    # Rebuild hypotheses
    print("\n  Loading corpus and rebuilding anchor hypotheses...")
    corpus = load_corpus(verbose=False)
    concordance = parse_concordance()
    medieval_names = load_medieval_names()
    folio_sets = build_folio_identification_sets(
        concordance, medieval_names, corpus,
        use_tfidf=use_tfidf,
    )
    folio_index = {fs.folio: fs for fs in folio_sets}

    hypotheses: List[AnchorHypothesis] = []
    stored = anchor_data.get('anchor_hypotheses', [])
    for h_dict in stored:
        if not h_dict.get('paradigm_compatible', False):
            continue
        folio = h_dict['folio']
        fs = folio_index.get(folio)
        if fs is None:
            continue
        for pid in fs.identifications:
            if pid.medieval_stem == h_dict.get('medieval_stem'):
                h = build_anchor_hypothesis(fs, pid)
                if h and h.paradigm_compatible:
                    hypotheses.append(h)
                break

    print(f"  Reconstructed anchors: {len(hypotheses)}")

    if len(hypotheses) < 3:
        print("  Too few anchors for diagnosis.")
        return {'verdict': 'insufficient_anchors'}

    # C.1: Per-anchor encoding model fit
    print("\n  C.1: Per-Anchor Encoding Model Fit")
    print("  " + "─" * 66)
    model_fits = compute_per_anchor_model_fits(hypotheses)

    for model_name, consensus in model_fits.items():
        print(f"\n    {model_name} ({consensus.description}):")
        print(f"      Good fit (<0.3): {consensus.n_good_fit}/{consensus.n_total}")
        print(f"      Mean fit: {consensus.mean_fit:.4f}")
        print(f"\n      {'Anchor':<8s} {'Plant':<14s} {'EVA stem':<10s} "
              f"{'Obs':<5s} {'Exp':<5s} {'Fit':<8s}")
        print(f"      {'─'*8} {'─'*14} {'─'*10} {'─'*5} {'─'*5} {'─'*8}")
        for a in consensus.per_anchor:
            fit_str = f"{a['fit']:.3f}"
            marker = " *" if a['fit'] < 0.3 else ""
            print(f"      {a['folio']:<8s} {a['medieval_stem'][:14]:<14s} "
                  f"{a['eva_stem'][:10]:<10s} {a['observed_chars']:<5d} "
                  f"{a['expected_units']:<5d} {fit_str}{marker}")

    # C.2: Model consensus analysis
    print("\n  C.2: Model Consensus Analysis")
    print("  " + "─" * 66)
    winner, clear_winner = analyze_model_consensus(model_fits)

    print(f"\n    Model ranking:")
    for name, consensus in sorted(model_fits.items(),
                                   key=lambda x: (-x[1].n_good_fit,
                                                  x[1].mean_fit)):
        marker = " << WINNER" if name == winner else ""
        print(f"      {name:<30s} {consensus.n_good_fit}/{consensus.n_total} "
              f"good fit, mean={consensus.mean_fit:.3f}{marker}")

    status = "CLEAR WINNER" if clear_winner else "NO CLEAR WINNER"
    print(f"\n    Winner: {winner} ({status})")

    # C.3: Segmentation sensitivity test
    print("\n  C.3: Segmentation Sensitivity Test")
    print("  " + "─" * 66)
    seg_results = test_segmentation_sensitivity(
        hypotheses, winner, baseline_unanimity,
    )

    print(f"\n    {'Segmentation':<18s} {'Unanimity':<12s} {'Chars':<8s} {'Δ'}")
    print(f"    {'─'*18} {'─'*12} {'─'*8} {'─'*10}")
    for sr in sorted(seg_results, key=lambda x: -x.unanimity):
        print(f"    {sr.segmentation_name:<18s} {sr.unanimity:<12.4f} "
              f"{sr.n_chars_mapped:<8d} {sr.delta_from_baseline:+.4f}")

    best_seg = max(seg_results, key=lambda x: x.unanimity)
    print(f"\n    Best segmentation: {best_seg.segmentation_name} "
          f"(unanimity {best_seg.unanimity:.4f})")

    # C.4: Hybrid model test
    print("\n  C.4: Hybrid Model Test")
    print("  " + "─" * 66)
    hybrid_results, hybrid_evidence = test_hybrid_model(hypotheses, model_fits)

    for hr in hybrid_results:
        if hr.n_anchors > 0:
            print(f"    {hr.category} names ({hr.n_anchors} anchors): "
                  f"best={hr.best_model}, fit={hr.best_fit:.3f}")
            print(f"      Anchors: {', '.join(hr.anchors)}")
        else:
            print(f"    {hr.category} names: (no anchors)")

    if hybrid_evidence:
        print(f"\n    Hybrid evidence: YES — different models for different lengths")
    else:
        print(f"\n    Hybrid evidence: NO — single model fits all lengths")

    # Verdict
    if clear_winner and best_seg.unanimity > baseline_unanimity:
        verdict = 'model_identified'
    elif clear_winner:
        verdict = 'model_identified_no_segmentation_improvement'
    elif hybrid_evidence:
        verdict = 'hybrid_model_suggested'
    else:
        verdict = 'no_clear_model'
    print(f"\n  VERDICT: {verdict}")

    # Build result
    result = EncodingDiagnosisResult(
        model_fits=[_convert(asdict(m)) for m in model_fits.values()],
        winner=winner,
        winner_good_fit=model_fits[winner].n_good_fit,
        winner_total=model_fits[winner].n_total,
        clear_winner=clear_winner,
        segmentation_results=[_convert(asdict(sr)) for sr in seg_results],
        best_segmentation=best_seg.segmentation_name,
        best_segmentation_unanimity=round(best_seg.unanimity, 4),
        hybrid_results=[_convert(asdict(hr)) for hr in hybrid_results],
        hybrid_evidence=hybrid_evidence,
        verdict=verdict,
    )

    # Save
    out_path = os.path.join(_results_dir(), 'encoding_diagnosis.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")

    return _convert(asdict(result))
