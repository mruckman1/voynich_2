"""
Phase 6 D+E: Rosetta Folio Selection & Encoding Model Test
============================================================
Score herbal folios for "Rosetta" suitability — how useful each folio is
as an anchor for cross-modal decoding — and test which encoding model
best fits the observed token structure.

Sub-analyses:
  6.D — Five-criteria scoring and greedy Rosetta folio selection
  6.E — Encoding model selection (syllabic, alphabetic, abbreviated, mixed)

Output:
  results/rosetta_selection.json
"""

import json
import math
import os
from collections import Counter
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from voynich.core.corpus import load_corpus, VoynichCorpus, tokenize_eva_chars
from voynich.core._paths import results_dir as _results_dir
from voynich.core.reference import LATIN_DECLENSION_SUFFIXES
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
class FolioScore:
    """Rosetta suitability score for one folio."""
    folio: str
    tier: int
    id_confidence: float
    name_distinctiveness: float
    dominant_stem_clarity: float
    char_coverage: float
    char_novelty: float
    combined_score: float
    medieval_name: str
    medieval_stem: str
    dominant_stem: str


@dataclass
class EncodingModelResult:
    """Test result for one encoding model against selected anchors."""
    model_name: str
    description: str
    expected_stem_length: float
    observed_mean_stem_length: float
    length_ratio: float
    consistency_score: float
    n_folios_tested: int


@dataclass
class RosettaSelectionResult:
    """Full Phase 6 D+E output."""
    n_candidates: int
    folio_scores: List[Dict]
    selected_rosetta_folios: List[str]
    n_selected: int
    eva_chars_covered: int
    eva_chars_total: int
    char_coverage_ratio: float
    # Encoding model test
    encoding_models_tested: List[Dict]
    best_encoding_model: str
    encoding_confidence: float
    # Gate
    gate_passed: bool
    verdict: str


# ---------------------------------------------------------------------------
# Scoring functions (Approach D)
# ---------------------------------------------------------------------------

def score_id_confidence(folio_set: FolioIdentificationSet) -> float:
    """
    Score identification confidence.

    Tier 1 -> 1.0, Tier 2 -> 0.6, Tier 3 -> 0.2.
    Bonus +0.1 if the identification comes from a dedicated researcher.
    """
    base = {1: 1.0, 2: 0.6, 3: 0.2}.get(folio_set.tier, 0.1)

    # Bonus for well-studied folios
    high_sources = {'Stephen Bax', 'Tucker & Janick', 'Janick & Tucker',
                    'Edith Sherwood'}
    for pid in folio_set.identifications:
        if pid.source in high_sources:
            base = min(1.0, base + 0.1)
            break

    return base


def score_name_distinctiveness(
    medieval_stem: str,
    all_stems: List[str],
) -> float:
    """
    Score how distinctive the medieval name's character set is.

    High score = few characters shared with other candidate stems.
    """
    if not medieval_stem or not all_stems:
        return 0.0

    my_chars = set(medieval_stem)
    if not my_chars:
        return 0.0

    # Count how many other stems share each character
    other_stems = [s for s in all_stems if s != medieval_stem and s]
    if not other_stems:
        return 1.0

    char_frequency: Counter = Counter()
    for s in other_stems:
        for c in set(s):
            char_frequency[c] += 1

    # Distinctiveness = fraction of chars that appear in fewer than 50% of stems
    threshold = len(other_stems) * 0.5
    n_distinctive = sum(1 for c in my_chars
                        if char_frequency.get(c, 0) < threshold)
    return n_distinctive / len(my_chars)


def score_dominant_stem_clarity(
    folio_set: FolioIdentificationSet,
) -> float:
    """
    Score how clearly one stem dominates the folio.

    clarity = tokens_with_dominant_stem / total_tokens.
    """
    if folio_set.token_count == 0 or folio_set.dominant_stem_token_count == 0:
        return 0.0
    return folio_set.dominant_stem_token_count / folio_set.token_count


def score_char_coverage(
    page_tokens: List[str],
    eva_alphabet: set,
) -> float:
    """
    Score character coverage: fraction of EVA alphabet on this folio.
    """
    if not eva_alphabet:
        return 0.0
    folio_chars: set = set()
    for token in page_tokens:
        for c in tokenize_eva_chars(token):
            folio_chars.add(c)
    return len(folio_chars & eva_alphabet) / len(eva_alphabet)


def score_char_novelty(
    page_tokens: List[str],
    already_selected_chars: set,
) -> float:
    """
    Score character novelty relative to already-selected folios.

    novelty = |chars_on_folio - already_selected| / |chars_on_folio|.
    """
    folio_chars: set = set()
    for token in page_tokens:
        for c in tokenize_eva_chars(token):
            folio_chars.add(c)
    if not folio_chars:
        return 0.0
    novel = folio_chars - already_selected_chars
    return len(novel) / len(folio_chars)


# ---------------------------------------------------------------------------
# Greedy Rosetta selection
# ---------------------------------------------------------------------------

def select_rosetta_folios(
    folio_sets: List[FolioIdentificationSet],
    corpus: VoynichCorpus,
    target_count: int = 10,
) -> Tuple[List[FolioScore], List[str]]:
    """
    Greedy selection of 8-12 Rosetta folios maximizing combined score
    + character coverage.

    Only Tier 1+2 folios with resolved medieval names are candidates.
    """
    # Build EVA alphabet from all herbal_a tokens
    all_tokens = corpus.get_tokens(section='herbal_a', paragraph_only=True)
    eva_alphabet: set = set()
    for token in all_tokens:
        for c in tokenize_eva_chars(token):
            eva_alphabet.add(c)

    # Filter to eligible candidates
    candidates: List[FolioIdentificationSet] = []
    for fs in folio_sets:
        if fs.tier > 2:
            continue
        # Must have at least one resolved medieval name
        if not any(p.medieval_name for p in fs.identifications):
            continue
        if not fs.dominant_stem:
            continue
        candidates.append(fs)

    if not candidates:
        return [], []

    # Collect all medieval stems for distinctiveness scoring
    all_med_stems = []
    for fs in candidates:
        for pid in fs.identifications:
            if pid.medieval_stem:
                all_med_stems.append(pid.medieval_stem)

    # Get page tokens for each candidate
    candidate_tokens: Dict[str, List[str]] = {}
    for fs in candidates:
        page = corpus.pages.get(fs.folio)
        if page:
            candidate_tokens[fs.folio] = page.all_tokens
        else:
            candidate_tokens[fs.folio] = []

    # Score criteria 1-4 (independent of selection order)
    pre_scores: Dict[str, Dict[str, float]] = {}
    for fs in candidates:
        # Use the first resolved medieval name
        med_stem = None
        med_name = None
        for pid in fs.identifications:
            if pid.medieval_stem:
                med_stem = pid.medieval_stem
                med_name = pid.medieval_name
                break

        pre_scores[fs.folio] = {
            'id_confidence': score_id_confidence(fs),
            'name_distinctiveness': score_name_distinctiveness(
                med_stem or '', all_med_stems,
            ),
            'dominant_stem_clarity': score_dominant_stem_clarity(fs),
            'char_coverage': score_char_coverage(
                candidate_tokens.get(fs.folio, []), eva_alphabet,
            ),
            'medieval_name': med_name or '',
            'medieval_stem': med_stem or '',
        }

    # Greedy selection with novelty recomputation
    selected: List[str] = []
    selected_chars: set = set()
    all_scores: List[FolioScore] = []

    # Sort candidates by base score (avg of criteria 1-4)
    def base_score(folio: str) -> float:
        s = pre_scores[folio]
        return (s['id_confidence'] + s['name_distinctiveness'] +
                s['dominant_stem_clarity'] + s['char_coverage']) / 4.0

    remaining = sorted([fs.folio for fs in candidates],
                       key=base_score, reverse=True)

    while remaining and len(selected) < target_count:
        best_folio = None
        best_combined = -1.0
        best_novelty = 0.0

        for folio in remaining:
            novelty = score_char_novelty(
                candidate_tokens.get(folio, []),
                selected_chars,
            )
            s = pre_scores[folio]
            combined = (s['id_confidence'] + s['name_distinctiveness'] +
                        s['dominant_stem_clarity'] + s['char_coverage'] +
                        novelty) / 5.0

            if combined > best_combined:
                best_combined = combined
                best_folio = folio
                best_novelty = novelty

        if best_folio is None or (len(selected) >= 8 and best_novelty < 0.05):
            break

        selected.append(best_folio)
        remaining.remove(best_folio)

        # Update selected chars
        for token in candidate_tokens.get(best_folio, []):
            for c in tokenize_eva_chars(token):
                selected_chars.add(c)

        # Find the matching FolioIdentificationSet
        fs_match = next(fs for fs in candidates if fs.folio == best_folio)
        s = pre_scores[best_folio]

        all_scores.append(FolioScore(
            folio=best_folio,
            tier=fs_match.tier,
            id_confidence=round(s['id_confidence'], 4),
            name_distinctiveness=round(s['name_distinctiveness'], 4),
            dominant_stem_clarity=round(s['dominant_stem_clarity'], 4),
            char_coverage=round(s['char_coverage'], 4),
            char_novelty=round(best_novelty, 4),
            combined_score=round(best_combined, 4),
            medieval_name=s['medieval_name'],
            medieval_stem=s['medieval_stem'],
            dominant_stem=fs_match.dominant_stem or '',
        ))

    return all_scores, selected


# ---------------------------------------------------------------------------
# Encoding model testing (Approach E)
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


def test_encoding_models(
    selected_folios: List[str],
    folio_sets: List[FolioIdentificationSet],
    corpus: VoynichCorpus,
) -> List[EncodingModelResult]:
    """
    Approach E: Test 4 encoding models against selected Rosetta folios.

    Compares expected vs observed dominant stem length under each model.
    """
    # Collect (voynich_stem_len, latin_name) pairs for selected folios
    pairs: List[Tuple[int, str]] = []
    for folio in selected_folios:
        fs = next((f for f in folio_sets if f.folio == folio), None)
        if fs is None or not fs.dominant_stem:
            continue
        med_name = None
        for pid in fs.identifications:
            if pid.medieval_name:
                med_name = pid.medieval_name
                break
        if med_name is None:
            continue

        eva_stem_chars = tokenize_eva_chars(fs.dominant_stem)
        pairs.append((len(eva_stem_chars), med_name))

    if not pairs:
        return []

    models: List[EncodingModelResult] = []

    # Model 1: Morphographic-syllabic (1 EVA char = 1 syllable)
    expected_syl = [_syllable_count(name) for _, name in pairs]
    observed = [n for n, _ in pairs]
    ratio_syl = np.mean(observed) / np.mean(expected_syl) if np.mean(expected_syl) > 0 else 0
    consistency_syl = 1.0 - np.std([o / max(e, 1) for o, e in zip(observed, expected_syl)])
    models.append(EncodingModelResult(
        model_name='morphographic-syllabic',
        description='Each EVA character encodes one syllable of Latin',
        expected_stem_length=round(float(np.mean(expected_syl)), 2),
        observed_mean_stem_length=round(float(np.mean(observed)), 2),
        length_ratio=round(float(ratio_syl), 4),
        consistency_score=round(float(max(0, consistency_syl)), 4),
        n_folios_tested=len(pairs),
    ))

    # Model 2: Morphographic-alphabetic (1 EVA char = 1 Latin letter)
    expected_alpha = [len(name) for _, name in pairs]
    ratio_alpha = np.mean(observed) / np.mean(expected_alpha) if np.mean(expected_alpha) > 0 else 0
    consistency_alpha = 1.0 - np.std([o / max(e, 1) for o, e in zip(observed, expected_alpha)])
    models.append(EncodingModelResult(
        model_name='morphographic-alphabetic',
        description='Each EVA character encodes one Latin letter',
        expected_stem_length=round(float(np.mean(expected_alpha)), 2),
        observed_mean_stem_length=round(float(np.mean(observed)), 2),
        length_ratio=round(float(ratio_alpha), 4),
        consistency_score=round(float(max(0, consistency_alpha)), 4),
        n_folios_tested=len(pairs),
    ))

    # Model 3: Morphographic-abbreviated (EVA chars < Latin letters)
    expected_abbrev = [max(2, len(name) // 2) for _, name in pairs]
    ratio_abbrev = np.mean(observed) / np.mean(expected_abbrev) if np.mean(expected_abbrev) > 0 else 0
    consistency_abbrev = 1.0 - np.std([o / max(e, 1) for o, e in zip(observed, expected_abbrev)])
    models.append(EncodingModelResult(
        model_name='morphographic-abbreviated',
        description='Each EVA character encodes an abbreviated Latin segment',
        expected_stem_length=round(float(np.mean(expected_abbrev)), 2),
        observed_mean_stem_length=round(float(np.mean(observed)), 2),
        length_ratio=round(float(ratio_abbrev), 4),
        consistency_score=round(float(max(0, consistency_abbrev)), 4),
        n_folios_tested=len(pairs),
    ))

    # Model 4: Mixed (moderate expectation)
    expected_mixed = [max(2, int(len(name) * 0.7)) for _, name in pairs]
    ratio_mixed = np.mean(observed) / np.mean(expected_mixed) if np.mean(expected_mixed) > 0 else 0
    consistency_mixed = 1.0 - np.std([o / max(e, 1) for o, e in zip(observed, expected_mixed)])
    models.append(EncodingModelResult(
        model_name='mixed',
        description='Hybrid encoding with variable granularity',
        expected_stem_length=round(float(np.mean(expected_mixed)), 2),
        observed_mean_stem_length=round(float(np.mean(observed)), 2),
        length_ratio=round(float(ratio_mixed), 4),
        consistency_score=round(float(max(0, consistency_mixed)), 4),
        n_folios_tested=len(pairs),
    ))

    return models


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_rosetta_selection(
    constrained_data: Optional[Dict] = None,
    use_tfidf: bool = False,
) -> Dict:
    """
    Run Phase 6 D+E: Rosetta Folio Selection + Encoding Model Test.

    1. Load Phase 6.0 results (or run if not available)
    2. Score all candidate folios
    3. Greedy-select Rosetta folios
    4. Test encoding models
    5. Gate: >= 8 selected folios with combined_score > 0.5
    6. Save results/rosetta_selection.json
    """
    print("=" * 70)
    print("Phase 6 D+E: Rosetta Folio Selection & Encoding Model Test")
    print("=" * 70)

    # Load prior results or recompute
    if constrained_data is None:
        results_path = os.path.join(
            _results_dir(), 'illustration_constrained.json')
        if os.path.exists(results_path):
            with open(results_path) as f:
                constrained_data = json.load(f)
        else:
            from voynich.phases.illustration_constrained import \
                run_illustration_constrained
            constrained_data = run_illustration_constrained()

    # Check prior gate
    if not constrained_data.get('gate_passed', False):
        print("\n  Prior gate FAILED — insufficient anchors.")
        print("  Cannot proceed with Rosetta selection.")
        result = RosettaSelectionResult(
            n_candidates=0, folio_scores=[], selected_rosetta_folios=[],
            n_selected=0, eva_chars_covered=0, eva_chars_total=0,
            char_coverage_ratio=0.0, encoding_models_tested=[],
            best_encoding_model='none', encoding_confidence=0.0,
            gate_passed=False, verdict='prior_gate_failed',
        )
        out_path = os.path.join(_results_dir(), 'rosetta_selection.json')
        with open(out_path, 'w') as f:
            json.dump(_convert(asdict(result)), f, indent=2, default=str)
        return _convert(asdict(result))

    # Rebuild folio sets from stored data
    print("\n  Rebuilding folio identification sets...")
    corpus = load_corpus(verbose=False)
    concordance = parse_concordance()
    medieval_names = load_medieval_names()
    folio_sets = build_folio_identification_sets(
        concordance, medieval_names, corpus,
        use_tfidf=use_tfidf,
    )

    # Select Rosetta folios
    print("\n  6.D: Scoring and selecting Rosetta folios")
    all_scores, selected = select_rosetta_folios(folio_sets, corpus)

    print(f"    Candidates evaluated: {len([fs for fs in folio_sets if fs.tier <= 2])}")
    print(f"    Rosetta folios selected: {len(selected)}")

    if all_scores:
        print("\n    Selected folios (ranked):")
        for i, sc in enumerate(all_scores, 1):
            print(f"      {i:2d}. {sc.folio} [T{sc.tier}] "
                  f"score={sc.combined_score:.3f} "
                  f"stem='{sc.dominant_stem}' -> '{sc.medieval_name}'")

    # Compute character coverage of selected folios
    all_herbal_tokens = corpus.get_tokens(
        section='herbal_a', paragraph_only=True)
    eva_alphabet: set = set()
    for token in all_herbal_tokens:
        for c in tokenize_eva_chars(token):
            eva_alphabet.add(c)

    selected_chars: set = set()
    for folio in selected:
        page = corpus.pages.get(folio)
        if page:
            for token in page.all_tokens:
                for c in tokenize_eva_chars(token):
                    selected_chars.add(c)

    char_coverage_ratio = (len(selected_chars & eva_alphabet) /
                           len(eva_alphabet) if eva_alphabet else 0.0)
    print(f"\n    EVA character coverage: "
          f"{len(selected_chars & eva_alphabet)}/{len(eva_alphabet)} "
          f"({char_coverage_ratio:.1%})")

    # Test encoding models
    print("\n  6.E: Encoding model test")
    encoding_results = test_encoding_models(selected, folio_sets, corpus)

    if encoding_results:
        print("\n    Encoding model results:")
        for er in encoding_results:
            print(f"      {er.model_name:<28s} "
                  f"expected={er.expected_stem_length:.1f} "
                  f"observed={er.observed_mean_stem_length:.1f} "
                  f"ratio={er.length_ratio:.3f} "
                  f"consistency={er.consistency_score:.3f}")

        # Best model = closest length_ratio to 1.0, then highest consistency
        best = min(encoding_results,
                   key=lambda r: (abs(r.length_ratio - 1.0),
                                  -r.consistency_score))
        print(f"\n    Best encoding model: {best.model_name} "
              f"(ratio={best.length_ratio:.3f})")
        best_name = best.model_name
        best_confidence = best.consistency_score
    else:
        best_name = 'undetermined'
        best_confidence = 0.0
        print("    No encoding models tested (insufficient data).")

    # Gate
    gate_ok, gate_msg = _check_gate(
        'rosetta_folio_count', float(len(selected)), 7.0, 'greater',
    )
    print(f"\n{gate_msg}")
    verdict = 'rosetta_set_ready' if gate_ok else 'insufficient_rosetta_folios'
    print(f"  Verdict: {verdict}")

    # Build result
    result = RosettaSelectionResult(
        n_candidates=len([fs for fs in folio_sets if fs.tier <= 2]),
        folio_scores=[asdict(s) for s in all_scores],
        selected_rosetta_folios=selected,
        n_selected=len(selected),
        eva_chars_covered=len(selected_chars & eva_alphabet),
        eva_chars_total=len(eva_alphabet),
        char_coverage_ratio=round(char_coverage_ratio, 4),
        encoding_models_tested=[asdict(r) for r in encoding_results],
        best_encoding_model=best_name,
        encoding_confidence=round(best_confidence, 4),
        gate_passed=gate_ok,
        verdict=verdict,
    )

    # Save
    out_path = os.path.join(_results_dir(), 'rosetta_selection.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")

    return _convert(asdict(result))
