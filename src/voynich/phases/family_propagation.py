"""
Phase 28.3 – Family Propagation
=================================
For each unconfirmed triple (~9 of 25), generates typologically valid
alternative syllable candidates using PHONEME_PLACE_MAP × PHONEME_NUCLEUS_MAP,
scores each by dict_hit on a corpus sample, and reports recommendations.

Does NOT automatically change the table — recommendations only.

Dependency chain:
    crib_extraction.json     (Step 28.1 — confirmed triples)
    crib_consistency.json    (Step 28.2 — inconsistent triples)
    feature_csp.json         (Phase 14 assignment)
        → family_propagation.json  (this step)
"""

import json
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    load_corpus,
    tokenize_eva_chars,
)
from voynich.core.reference import (
    EVA_VISUAL_COMPONENTS,
    PHONEME_NUCLEUS_MAP,
    PHONEME_PLACE_MAP,
    build_expanded_word_set,
    load_reference_corpus,
)
from voynich.phases.csp_solver import decode_token


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
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PropagationEntry:
    triple: str
    eva_glyphs: List[str]
    current_syllable: str
    is_confirmed: bool
    is_family_consistent: bool
    candidate_syllables: List[str]
    candidate_dict_hits: List[float]   # dict_hit for each candidate
    best_alternative: Optional[str]
    best_alternative_dict_hit: float
    current_dict_hit: float
    dict_hit_delta: float
    recommendation: str                # 'confirm', 'investigate', 'correct'


@dataclass
class FamilyPropagationResult:
    n_confirmed_triples: int
    n_unconfirmed_triples: int
    n_inconsistent: int
    propagation_entries: List[Dict]
    n_corrections_recommended: int
    best_correction_triple: Optional[str]
    best_correction_syllable: Optional[str]
    estimated_dict_hit_delta: float
    gate_passed: bool
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def _enumerate_candidates(
    triple: str,
    current_syllable: str,
) -> List[str]:
    """Generate all typologically valid CV syllables for a triple."""
    parts = triple.split(',')
    if len(parts) != 3:
        return []
    first_stroke, last_stroke, _ = parts

    onsets = PHONEME_PLACE_MAP.get(first_stroke, [])
    nuclei = PHONEME_NUCLEUS_MAP.get(last_stroke, [])

    candidates = set()
    for onset in onsets:
        for nucleus in nuclei:
            syl = onset + nucleus
            if len(syl) >= 2:  # valid CV syllable
                candidates.add(syl)
    # Also include pure vowels
    for nucleus in nuclei:
        if len(nucleus) == 1:
            candidates.add(nucleus)

    # Remove current so we only test alternatives
    candidates.discard(current_syllable)
    return sorted(candidates)


def _score_candidate(
    assignment: Dict[str, str],
    triple: str,
    candidate: str,
    tokens_sample: List[str],
    eva_to_triple: Dict[str, str],
    ref_word_set: set,
) -> float:
    """Score a candidate syllable by dict_hit on a token sample."""
    test_assignment = dict(assignment)
    test_assignment[triple] = candidate
    hits = 0
    for token in tokens_sample:
        decoded = decode_token(token, test_assignment, eva_to_triple)
        if decoded.lower() in ref_word_set:
            hits += 1
    return hits / len(tokens_sample) if tokens_sample else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_family_propagation() -> None:
    """Step 28.3: Family propagation for unconfirmed triples."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE 28.3: Family Propagation")
    print("=" * 70)

    rd = _results_dir()
    eva_to_triple = build_eva_to_triple_lookup()

    # ── 1. Load inputs ──
    print("\n  1. Loading inputs …")

    # Crib extraction — confirmed triples
    crib_path = os.path.join(rd, 'crib_extraction.json')
    if not os.path.exists(crib_path):
        print("  [SKIP] crib_extraction.json not found — run crib-extract first")
        return
    with open(crib_path) as f:
        crib_data = json.load(f)
    confirmed_triples = set(crib_data.get('all_triples_covered', []))
    unconfirmed_triples = crib_data.get('all_triples_unconfirmed', [])

    # Consistency — inconsistent triples
    consist_path = os.path.join(rd, 'crib_consistency.json')
    inconsistent_set: Set[str] = set()
    if os.path.exists(consist_path):
        with open(consist_path) as f:
            consist_data = json.load(f)
        for entry in consist_data.get('inconsistent_triples', []):
            inconsistent_set.add(entry.get('triple', ''))

    # Phase 14 assignment
    csp_path = os.path.join(rd, 'feature_csp.json')
    if not os.path.exists(csp_path):
        print("  [SKIP] feature_csp.json not found")
        return
    with open(csp_path) as f:
        csp_data = json.load(f)
    assignment = (csp_data.get('language_results', {})
                  .get('latin', {}).get('best_assignment', {}))
    if not assignment:
        assignment = csp_data.get('best_assignment', {})

    print(f"     Confirmed triples: {len(confirmed_triples)}")
    print(f"     Unconfirmed triples: {len(unconfirmed_triples)}")
    print(f"     Inconsistent triples: {len(inconsistent_set)}")

    # ── 2. Build reference word set ──
    print("\n  2. Building reference word set …")
    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    base_words = set(w.lower() for w in ref_corpus.get_combined_tokens('latin')
                     if len(w) >= 2)
    expanded, _ = build_expanded_word_set(base_words)
    ref_word_set = base_words | expanded
    print(f"     {len(ref_word_set)} reference words")

    # ── 3. Sample corpus tokens ──
    print("\n  3. Sampling corpus tokens …")
    corpus = load_corpus(verbose=False)
    all_tokens = corpus.get_tokens()
    sample_tokens = all_tokens[:2000]
    print(f"     Using {len(sample_tokens)} tokens for scoring")

    # Current baseline dict_hit
    baseline_hits = sum(
        1 for t in sample_tokens
        if decode_token(t, assignment, eva_to_triple).lower() in ref_word_set
    )
    baseline_dict_hit = baseline_hits / len(sample_tokens)
    print(f"     Baseline dict_hit on sample: {baseline_dict_hit:.3f}")

    # ── 4. Build EVA glyph → triple lookup (reverse) ──
    triple_to_glyphs: Dict[str, List[str]] = {}
    for glyph, components in EVA_VISUAL_COMPONENTS.items():
        tk = (components['first_stroke'] + ',' +
              components['last_stroke'] + ',' +
              components['glyph_class'])
        triple_to_glyphs.setdefault(tk, []).append(glyph)

    # ── 5. Score candidates for each unconfirmed triple ──
    print("\n  4. Scoring candidates for unconfirmed triples …")
    entries: List[PropagationEntry] = []

    # Process all triples (confirmed and unconfirmed)
    for triple in sorted(assignment.keys()):
        current_syl = assignment[triple]
        is_confirmed = triple in confirmed_triples
        is_consistent = triple not in inconsistent_set
        glyphs = triple_to_glyphs.get(triple, [])

        # Only enumerate and score alternatives for unconfirmed or inconsistent
        if is_confirmed and is_consistent:
            entries.append(PropagationEntry(
                triple=triple,
                eva_glyphs=sorted(glyphs),
                current_syllable=current_syl,
                is_confirmed=True,
                is_family_consistent=is_consistent,
                candidate_syllables=[],
                candidate_dict_hits=[],
                best_alternative=None,
                best_alternative_dict_hit=0.0,
                current_dict_hit=baseline_dict_hit,
                dict_hit_delta=0.0,
                recommendation='confirm',
            ))
            continue

        candidates = _enumerate_candidates(triple, current_syl)
        # Score each candidate (limit to top 10 by typological priority)
        candidates = candidates[:10]

        scores = []
        for cand in candidates:
            score = _score_candidate(
                assignment, triple, cand, sample_tokens,
                eva_to_triple, ref_word_set,
            )
            scores.append(score)

        # Find best alternative
        best_alt = None
        best_alt_score = 0.0
        for cand, score in zip(candidates, scores):
            if score > best_alt_score:
                best_alt_score = score
                best_alt = cand

        delta = best_alt_score - baseline_dict_hit if best_alt else 0.0

        # Recommendation
        if not is_consistent and best_alt and delta > 0.005:
            recommendation = 'correct'
        elif not is_confirmed and best_alt and delta > 0.01:
            recommendation = 'investigate'
        else:
            recommendation = 'confirm'

        tag = '!' if recommendation == 'correct' else ('?' if recommendation == 'investigate' else ' ')
        print(f"    {tag} {triple} = '{current_syl}'  "
              f"confirmed={is_confirmed}  consistent={is_consistent}  "
              f"best_alt={best_alt} Δ={delta:+.4f}")

        entries.append(PropagationEntry(
            triple=triple,
            eva_glyphs=sorted(glyphs),
            current_syllable=current_syl,
            is_confirmed=is_confirmed,
            is_family_consistent=is_consistent,
            candidate_syllables=candidates,
            candidate_dict_hits=[round(s, 4) for s in scores],
            best_alternative=best_alt,
            best_alternative_dict_hit=round(best_alt_score, 4),
            current_dict_hit=round(baseline_dict_hit, 4),
            dict_hit_delta=round(delta, 4),
            recommendation=recommendation,
        ))

    n_corrections = sum(1 for e in entries if e.recommendation == 'correct')
    best_correction = max(
        (e for e in entries if e.recommendation == 'correct'),
        key=lambda e: e.dict_hit_delta,
        default=None,
    )

    # ── 6. Gate and verdict ──
    gate_passed = True  # propagation always passes; it's advisory
    verdict = (
        f"ADVISORY: {n_corrections} correction(s) recommended. "
        f"Best: {best_correction.triple} → '{best_correction.best_alternative}' "
        f"(Δ={best_correction.dict_hit_delta:+.4f})"
        if best_correction
        else f"ADVISORY: No corrections recommended. "
             f"All {len(confirmed_triples)} confirmed triples are consistent."
    )
    print(f"\n  {verdict}")

    # ── 7. Save ──
    result = FamilyPropagationResult(
        n_confirmed_triples=len(confirmed_triples),
        n_unconfirmed_triples=len(unconfirmed_triples),
        n_inconsistent=len(inconsistent_set),
        propagation_entries=[_convert(asdict(e)) for e in entries],
        n_corrections_recommended=n_corrections,
        best_correction_triple=(best_correction.triple
                                if best_correction else None),
        best_correction_syllable=(best_correction.best_alternative
                                  if best_correction else None),
        estimated_dict_hit_delta=(round(best_correction.dict_hit_delta, 4)
                                  if best_correction else 0.0),
        gate_passed=gate_passed,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'family_propagation.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)
    print(f"\n  → {out_path}")
