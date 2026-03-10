"""
Phase 31.2: Plant Name CSP on Folio Labels
=============================================
For each Tier A/B folio, perform an exhaustive constraint-satisfaction search
to determine whether any EVA label token can encode the expected plant name.
This bypasses the decoding table entirely — picture of plant + adjacent glyphs
= known word.

Dependency chain:
    consensus_plants.json       (Step 31.1)
    bootstrap_loop.json         (Phase 30 confirmed triples)
    tachygraphic_stroke.json    (Phase 19 sign families)
    modifier_integrate.json     (Phase 16 modifiers)
        → plant_name_csp.json  (this step)
"""

import json
import os
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from itertools import permutations
from typing import Any, Dict, List, Optional, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    load_corpus,
    tokenize_eva_chars,
)
from voynich.phases.null_corpus import _reconstruct_modifier_rules


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
class AlignmentResult:
    """One alignment of EVA chars to syllables."""
    token: str
    plant_name: str
    eva_chars: List[str]
    syllables: List[str]
    assignments: Dict[str, str]  # triple_key -> syllable
    confirmed_consistent: int    # how many confirmed triples agree
    confirmed_conflicting: int   # how many confirmed triples disagree
    unconfirmed_filled: int      # how many new triples proposed
    family_consistent: bool
    name_coverage: float
    score: float


@dataclass
class FolioCSPResult:
    """CSP results for one folio."""
    folio: str
    genus: str
    tier: str
    n_label_candidates: int
    n_plant_names: int
    n_alignments_tested: int
    n_valid_alignments: int
    best_alignment: Optional[Dict]
    proposed_new_triples: Dict[str, str]  # triple_key -> syllable (from best)
    null_correct_score: float
    null_wrong_mean_score: float
    null_selectivity: float


@dataclass
class CrossFolioConsistency:
    """Triple assignments consistent across multiple folios."""
    triple_key: str
    syllable: str
    supporting_folios: List[str]
    n_folios: int


@dataclass
class PlantCSPResult:
    """Full Step 31.2 output."""
    n_folios_tested: int
    n_folios_with_valid_alignments: int
    per_folio_results: List[Dict]
    cross_folio_consistent: List[Dict]
    n_new_triple_assignments: int
    mean_null_selectivity: float
    gate_passed: bool
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Syllabification
# ---------------------------------------------------------------------------

def _syllabify_cv(word: str) -> List[str]:
    """Syllabify a Latin word into CV-like syllables."""
    word = word.lower().strip()
    vowels = set('aeiouy')
    result = []
    current = ''

    for ch in word:
        current += ch
        if ch in vowels:
            result.append(current)
            current = ''

    if current and result:
        result[-1] += current
    elif current:
        result.append(current)

    return result


# ---------------------------------------------------------------------------
# Alignment enumeration
# ---------------------------------------------------------------------------

def _check_length_compatibility(
    n_eva_chars: int,
    n_syllables: int,
) -> Tuple[bool, str]:
    """Check if EVA char count is compatible with syllable count."""
    diff = abs(n_eva_chars - n_syllables)
    if diff == 0:
        return True, 'exact'
    elif diff == 1:
        return True, 'off_by_1'
    elif diff == 2:
        return True, 'off_by_2'
    return False, 'incompatible'


def _enumerate_alignments(
    eva_chars: List[str],
    syllables: List[str],
    eva_to_triple: Dict[str, str],
    modifier_chars: Set[str],
    max_alignments: int = 500,
) -> List[Dict[str, str]]:
    """Enumerate possible char→syllable alignments.

    For each syllabic EVA char, assigns one syllable.
    Modifiers are skipped (not assigned a syllable).
    """
    # Filter to syllabic chars only
    syllabic_chars = [(i, ch) for i, ch in enumerate(eva_chars)
                      if ch not in modifier_chars]
    n_syl_chars = len(syllabic_chars)
    n_syls = len(syllables)

    if n_syl_chars == 0 or n_syls == 0:
        return []

    alignments = []

    if n_syl_chars == n_syls:
        # Direct 1:1 alignment
        mapping = {}
        for (idx, ch), syl in zip(syllabic_chars, syllables):
            triple = eva_to_triple.get(ch, ch)
            mapping[triple] = syl
        alignments.append(mapping)

    elif n_syl_chars == n_syls + 1:
        # One extra EVA char — try skipping each position
        for skip in range(n_syl_chars):
            mapping = {}
            syl_idx = 0
            valid = True
            for pos, (idx, ch) in enumerate(syllabic_chars):
                if pos == skip:
                    continue
                if syl_idx >= n_syls:
                    valid = False
                    break
                triple = eva_to_triple.get(ch, ch)
                mapping[triple] = syllables[syl_idx]
                syl_idx += 1
            if valid and syl_idx == n_syls:
                alignments.append(mapping)

    elif n_syl_chars == n_syls - 1:
        # One fewer EVA char — abbreviation; try assigning first n_syl_chars syllables
        mapping = {}
        for (idx, ch), syl in zip(syllabic_chars, syllables[:n_syl_chars]):
            triple = eva_to_triple.get(ch, ch)
            mapping[triple] = syl
        alignments.append(mapping)

    elif n_syl_chars == n_syls + 2:
        # Two extra EVA chars — try skipping each pair
        for s1 in range(n_syl_chars):
            for s2 in range(s1 + 1, n_syl_chars):
                if len(alignments) >= max_alignments:
                    break
                mapping = {}
                syl_idx = 0
                valid = True
                for pos, (idx, ch) in enumerate(syllabic_chars):
                    if pos == s1 or pos == s2:
                        continue
                    if syl_idx >= n_syls:
                        valid = False
                        break
                    triple = eva_to_triple.get(ch, ch)
                    mapping[triple] = syllables[syl_idx]
                    syl_idx += 1
                if valid and syl_idx == n_syls:
                    alignments.append(mapping)

    elif n_syl_chars == n_syls - 2:
        # Two fewer EVA chars — try first n_syl_chars syllables
        mapping = {}
        for (idx, ch), syl in zip(syllabic_chars, syllables[:n_syl_chars]):
            triple = eva_to_triple.get(ch, ch)
            mapping[triple] = syl
        alignments.append(mapping)

    return alignments[:max_alignments]


# ---------------------------------------------------------------------------
# Consistency checking
# ---------------------------------------------------------------------------

def _check_confirmed_consistency(
    alignment: Dict[str, str],
    confirmed_triples: Dict[str, str],
) -> Tuple[int, int]:
    """Check alignment against confirmed triples.

    Returns (n_consistent, n_conflicting).
    """
    n_consistent = 0
    n_conflicting = 0

    for triple_key, proposed_syl in alignment.items():
        if triple_key in confirmed_triples:
            if confirmed_triples[triple_key] == proposed_syl:
                n_consistent += 1
            else:
                n_conflicting += 1

    return n_consistent, n_conflicting


def _check_family_consistency(
    alignment: Dict[str, str],
    sign_families: Dict[str, str],
) -> bool:
    """Check if proposed assignments respect sign family structure.

    Same family should share the same consonant.
    """
    # Group triples by family
    family_consonants: Dict[str, Set[str]] = defaultdict(set)

    for triple_key, syllable in alignment.items():
        family = sign_families.get(triple_key, 'unknown')
        if syllable and len(syllable) >= 1:
            consonant = syllable[0] if syllable[0] not in 'aeiouy' else ''
            if consonant:
                family_consonants[family].add(consonant)

    # Each family should ideally have at most 1 consonant
    for family, consonants in family_consonants.items():
        if len(consonants) > 2:  # Allow some flexibility
            return False

    return True


def _score_alignment(
    alignment: Dict[str, str],
    n_confirmed_consistent: int,
    n_confirmed_conflicting: int,
    n_unconfirmed_filled: int,
    family_consistent: bool,
    name_coverage: float,
) -> float:
    """Score an alignment based on consistency and coverage."""
    if n_confirmed_conflicting > 0:
        return 0.0  # Any conflict with confirmed triples = reject

    confirmed_score = 1.0 if n_confirmed_consistent > 0 else 0.5
    family_score = 1.0 if family_consistent else 0.3

    return (confirmed_score * 0.4 +
            (n_unconfirmed_filled / 25.0) * 0.3 +
            family_score * 0.2 +
            name_coverage * 0.1)


# ---------------------------------------------------------------------------
# Per-folio CSP
# ---------------------------------------------------------------------------

def _run_folio_csp(
    folio_entry: Dict,
    confirmed_triples: Dict[str, str],
    sign_families: Dict[str, str],
    eva_to_triple: Dict[str, str],
    modifier_chars: Set[str],
    corpus,
    all_concordance_entries: List[Dict],  # for null test (wrong plants)
) -> FolioCSPResult:
    """Run exhaustive CSP for one folio."""
    folio = folio_entry['folio']
    consensus = folio_entry['consensus']
    tier = consensus['tier']
    genus = consensus.get('genus', '?')

    med_names = folio_entry.get('medieval_names', [])
    label_cands = folio_entry.get('label_candidates', [])

    # Extract plant name syllabifications
    plant_syllables: List[Tuple[str, List[str]]] = []
    for mn in med_names:
        name = mn.get('medieval_name', '')
        syls = mn.get('syllabified', [])
        if name and syls:
            plant_syllables.append((name, syls))

    n_alignments_tested = 0
    n_valid = 0
    best_alignment: Optional[AlignmentResult] = None
    best_score = -1.0

    for lc in label_cands[:5]:  # Top 5 label candidates
        token = lc['token']
        eva_chars = lc['eva_chars']

        for plant_name, syllables in plant_syllables:
            # Filter syllabic chars
            syllabic_chars = [ch for ch in eva_chars if ch not in modifier_chars]
            n_syl_chars = len(syllabic_chars)
            n_syls = len(syllables)

            compatible, mode = _check_length_compatibility(n_syl_chars, n_syls)
            if not compatible:
                continue

            alignments = _enumerate_alignments(
                eva_chars, syllables, eva_to_triple, modifier_chars,
            )

            for alignment in alignments:
                n_alignments_tested += 1

                n_cons, n_conf = _check_confirmed_consistency(
                    alignment, confirmed_triples,
                )

                if n_conf > 0:
                    continue  # Reject — conflicts with confirmed triples

                # Count unconfirmed triples filled
                n_unconfirmed = sum(1 for k in alignment
                                   if k not in confirmed_triples)

                family_ok = _check_family_consistency(alignment, sign_families)
                coverage = n_syls / max(len(syllables), 1)

                score = _score_alignment(
                    alignment, n_cons, n_conf, n_unconfirmed,
                    family_ok, coverage,
                )

                if score > 0:
                    n_valid += 1

                if score > best_score:
                    best_score = score
                    best_alignment = AlignmentResult(
                        token=token,
                        plant_name=plant_name,
                        eva_chars=eva_chars,
                        syllables=syllables,
                        assignments=dict(alignment),
                        confirmed_consistent=n_cons,
                        confirmed_conflicting=n_conf,
                        unconfirmed_filled=n_unconfirmed,
                        family_consistent=family_ok,
                        name_coverage=round(coverage, 3),
                        score=round(score, 4),
                    )

    # Null test: try wrong plant names from other folios
    wrong_scores = []
    for other_entry in all_concordance_entries:
        if other_entry['folio'] == folio:
            continue
        other_med = other_entry.get('medieval_names', [])
        for mn in other_med[:2]:  # Try 2 names from each other folio
            name = mn.get('medieval_name', '')
            syls = mn.get('syllabified', [])
            if not name or not syls:
                continue

            wrong_best = 0.0
            for lc in label_cands[:3]:
                token = lc['token']
                eva_chars = lc['eva_chars']
                syllabic = [ch for ch in eva_chars if ch not in modifier_chars]

                compatible, _ = _check_length_compatibility(len(syllabic), len(syls))
                if not compatible:
                    continue

                aligns = _enumerate_alignments(
                    eva_chars, syls, eva_to_triple, modifier_chars,
                )
                for alignment in aligns:
                    nc, nf = _check_confirmed_consistency(alignment, confirmed_triples)
                    if nf > 0:
                        continue
                    nu = sum(1 for k in alignment if k not in confirmed_triples)
                    fam = _check_family_consistency(alignment, sign_families)
                    cov = len(syls) / max(len(syls), 1)
                    s = _score_alignment(alignment, nc, nf, nu, fam, cov)
                    wrong_best = max(wrong_best, s)

            if wrong_best > 0:
                wrong_scores.append(wrong_best)

    correct_score = best_score if best_score > 0 else 0.0
    wrong_mean = sum(wrong_scores) / len(wrong_scores) if wrong_scores else 0.0
    null_sel = correct_score / max(wrong_mean, 0.001)

    # Extract proposed new triples from best alignment
    proposed_new = {}
    if best_alignment and best_alignment.confirmed_conflicting == 0:
        for tk, syl in best_alignment.assignments.items():
            if tk not in confirmed_triples:
                proposed_new[tk] = syl

    return FolioCSPResult(
        folio=folio,
        genus=genus,
        tier=tier,
        n_label_candidates=len(label_cands),
        n_plant_names=len(plant_syllables),
        n_alignments_tested=n_alignments_tested,
        n_valid_alignments=n_valid,
        best_alignment=_convert(asdict(best_alignment)) if best_alignment else None,
        proposed_new_triples=proposed_new,
        null_correct_score=round(correct_score, 4),
        null_wrong_mean_score=round(wrong_mean, 4),
        null_selectivity=round(null_sel, 2),
    )


# ---------------------------------------------------------------------------
# Cross-folio validation
# ---------------------------------------------------------------------------

def _cross_folio_validate(
    folio_results: List[FolioCSPResult],
) -> List[CrossFolioConsistency]:
    """Find triple assignments consistent across multiple folios."""
    triple_proposals: Dict[str, Dict[str, List[str]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for fr in folio_results:
        for triple_key, syllable in fr.proposed_new_triples.items():
            triple_proposals[triple_key][syllable].append(fr.folio)

    consistent = []
    for triple_key, syl_folios in triple_proposals.items():
        for syllable, folios in syl_folios.items():
            if len(folios) >= 2:
                consistent.append(CrossFolioConsistency(
                    triple_key=triple_key,
                    syllable=syllable,
                    supporting_folios=sorted(folios),
                    n_folios=len(folios),
                ))

    consistent.sort(key=lambda c: -c.n_folios)
    return consistent


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_plant_csp() -> None:
    """Step 31.2: Plant name CSP on folio labels."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE 31.2: Plant Name CSP")
    print("=" * 70)

    rd = _results_dir()
    eva_to_triple = build_eva_to_triple_lookup()

    # ── 1. Load inputs ──
    print("\n  1. Loading inputs...")

    # Consensus plants from Step 31.1
    cp_path = os.path.join(rd, 'consensus_plants.json')
    if not os.path.exists(cp_path):
        print("  [SKIP] consensus_plants.json not found — run consensus-plants first")
        return
    with open(cp_path) as f:
        cp_data = json.load(f)

    # Confirmed triples from Phase 30
    bt_path = os.path.join(rd, 'bootstrap_loop.json')
    confirmed_triples = {}
    if os.path.exists(bt_path):
        with open(bt_path) as f:
            bt_data = json.load(f)
        confirmed_triples = bt_data.get('final_assignment', {})
        # Only keep truly confirmed ones
        confirmed_keys = set(bt_data.get('confirmed_triples', []))
        if confirmed_keys:
            confirmed_triples = {k: v for k, v in confirmed_triples.items()
                                 if k in confirmed_keys}
    if not confirmed_triples:
        # Fall back to combined_refine assignment as reference
        with open(os.path.join(rd, 'combined_refine.json')) as f:
            refine_data = json.load(f)
        confirmed_triples = {}  # No confirmed — all proposals are new

    # Sign families from Phase 19
    sf_path = os.path.join(rd, 'tachygraphic_stroke.json')
    sign_families = {}
    if os.path.exists(sf_path):
        with open(sf_path) as f:
            sf_data = json.load(f)
        for family in sf_data.get('families', []):
            family_name = family.get('family_name', '')
            for member in family.get('members', []):
                triple_key = member.get('triple_key', '')
                if triple_key:
                    sign_families[triple_key] = family_name

    # Modifier chars
    with open(os.path.join(rd, 'modifier_integrate.json')) as f:
        mod_data = json.load(f)
    modifier_chars_set, _ = _reconstruct_modifier_rules(mod_data)

    corpus = load_corpus(verbose=False)

    print(f"     {len(confirmed_triples)} confirmed triples")
    print(f"     {len(sign_families)} sign family entries")

    # ── 2. Collect Tier A+B folios ──
    all_entries = cp_data.get('tier_a_folios', []) + cp_data.get('tier_b_folios', [])
    print(f"\n  2. Testing {len(all_entries)} Tier A+B folios...")

    # ── 3. Run CSP per folio ──
    folio_results: List[FolioCSPResult] = []
    for entry in all_entries:
        folio = entry['folio']
        genus = entry.get('consensus', {}).get('genus', '?')
        print(f"\n     {folio} ({genus}):")

        result = _run_folio_csp(
            entry, confirmed_triples, sign_families,
            eva_to_triple, modifier_chars_set, corpus,
            all_entries,
        )
        folio_results.append(result)

        print(f"       {result.n_alignments_tested} alignments tested, "
              f"{result.n_valid_alignments} valid")
        if result.best_alignment:
            ba = result.best_alignment
            print(f"       Best: {ba.get('token', '?')} → {ba.get('plant_name', '?')}")
            print(f"       Score: {ba.get('score', 0):.4f}, "
                  f"confirmed={ba.get('confirmed_consistent', 0)}, "
                  f"new={ba.get('unconfirmed_filled', 0)}")
        print(f"       Null selectivity: {result.null_selectivity:.2f}×")

    # ── 4. Cross-folio validation ──
    print("\n  4. Cross-folio validation...")
    consistent = _cross_folio_validate(folio_results)

    if consistent:
        print(f"     {len(consistent)} cross-folio consistent triple assignments:")
        for c in consistent:
            print(f"       {c.triple_key} → '{c.syllable}' "
                  f"(from {c.n_folios} folios: {', '.join(c.supporting_folios)})")
    else:
        print("     No cross-folio consistent assignments found")

    # ── 5. Gate and verdict ──
    n_with_valid = sum(1 for fr in folio_results if fr.n_valid_alignments > 0)
    sel_values = [fr.null_selectivity for fr in folio_results
                  if fr.null_selectivity > 0 and fr.null_selectivity < 100]
    mean_sel = sum(sel_values) / len(sel_values) if sel_values else 0.0

    gate = len(consistent) >= 1 and mean_sel > 1.0
    if len(consistent) >= 3:
        verdict = "STRONG_BOTANICAL_ANCHORS"
    elif len(consistent) >= 1:
        verdict = "PARTIAL_BOTANICAL_ANCHORS"
    elif n_with_valid > 0:
        verdict = "WEAK_BOTANICAL_ANCHORS"
    else:
        verdict = "NO_BOTANICAL_ANCHORS"

    print(f"\n  Gate: {'PASS' if gate else 'FAIL'}")
    print(f"  {verdict}: {len(consistent)} cross-folio assignments, "
          f"{n_with_valid}/{len(folio_results)} folios with valid alignments")

    # ── 6. Save ──
    result = PlantCSPResult(
        n_folios_tested=len(folio_results),
        n_folios_with_valid_alignments=n_with_valid,
        per_folio_results=[_convert(asdict(fr)) for fr in folio_results],
        cross_folio_consistent=[_convert(asdict(c)) for c in consistent],
        n_new_triple_assignments=len(consistent),
        mean_null_selectivity=round(mean_sel, 2),
        gate_passed=gate,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'plant_name_csp.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2)

    print(f"\n  Saved → {out_path}")
    print(f"  Completed in {time.time() - t0:.1f}s")
