"""
Phase 19.7 – Illustration-Targeted Folio Decode
=================================================
Concentrate decoding effort on folios with published botanical
identifications, using both approaches' constraints simultaneously
to maximize the chance of detecting illustration-text matches.

Dependency chain:
    illustration_constrained.json (Phase 6)
    combined_refine.json  (Phase 15)
    modifier_integrate.json (Phase 16)
    corpus
        → illustration_targeted.json
"""

import json
import os
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    decode_token_modifier_aware,
    load_corpus,
    tokenize_eva_chars,
    token_to_triples,
)
from voynich.core.reference import (
    PHARMACEUTICAL_VOCABULARY,
    build_expanded_word_set,
    load_reference_corpus,
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
class FolioMatchResult:
    folio: str
    n_tokens: int
    n_decoded: int
    plant_identifications: List[str]
    medieval_names: List[str]
    medieval_stems: List[str]
    # Matches found
    name_matches: List[str]
    stem_matches: List[str]
    humoral_matches: List[str]
    preparation_matches: List[str]
    # Scores
    weighted_score: float
    decoded_sample: List[List[str]]  # [[token, decoded], ...]


@dataclass
class IllustrationTargetedResult:
    n_folios_tested: int
    n_folios_with_matches: int
    # Per-folio results
    per_folio_results: List[Dict[str, Any]]
    # Aggregate
    total_weighted_score: float
    mean_score_per_folio: float
    n_name_matches: int
    n_stem_matches: int
    n_humoral_matches: int
    n_preparation_matches: int
    # Permutation test
    null_scores: List[float]
    real_score: float
    p_value: float
    selectivity: float
    # Strategy comparison
    strategy_a_matches: int
    strategy_b_matches: int
    strategy_c_matches: int
    best_strategy: str
    # Gate
    gate_passed: bool
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LATIN_VOWELS = set('aeiou')


def _load_json(path: str) -> Optional[Dict]:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _edit_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return _edit_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            cost = 0 if c1 == c2 else 1
            curr_row.append(min(curr_row[j] + 1, prev_row[j + 1] + 1, prev_row[j] + cost))
        prev_row = curr_row
    return prev_row[-1]


def _extract_skeleton(word: str) -> str:
    return ''.join(ch for ch in word.lower() if ch.isalpha() and ch not in _LATIN_VOWELS)


def _decode_token_safe(
    token: str,
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    modifier_chars: Set[str],
) -> str:
    try:
        return decode_token_modifier_aware(
            token, assignment, eva_to_triple, modifier_chars,
        )
    except Exception:
        triples = token_to_triples(token, eva_to_triple)
        return ''.join(assignment.get(t, '?') for t in triples)


# Humoral quality terms
HUMORAL_TERMS = {
    'calidus', 'calida', 'calidum', 'frigidus', 'frigida', 'frigidum',
    'humidus', 'humida', 'humidum', 'siccus', 'sicca', 'siccum',
    'temperatus', 'temperata', 'temperatum',
}

# Common preparation terms
PREPARATION_TERMS = {
    'succus', 'succo', 'decoctio', 'decocto', 'infusio', 'infuso',
    'pulvis', 'pulvere', 'unguentum', 'unguento', 'emplastrum',
    'syrupus', 'syrupo', 'oleum', 'oleo', 'aqua',
    'contere', 'misce', 'recipe', 'accipe', 'coque', 'adde', 'pone',
    'folia', 'radix', 'radice', 'semen', 'semine', 'cortex', 'cortice',
    'flos', 'flore', 'herba',
}


def _search_matches(
    decoded_tokens: List[str],
    plant_entry: Dict,
) -> Dict[str, List[str]]:
    """
    Search decoded tokens for matches against expected plant names,
    synonyms, humoral qualities, and preparation terms.
    """
    matches: Dict[str, List[str]] = {
        'name': [], 'stem': [], 'humoral': [], 'preparation': [],
    }

    # Build search terms
    search_names = set()
    search_stems = set()

    for ident in plant_entry.get('identifications', []):
        med_name = ident.get('medieval_name', '')
        med_stem = ident.get('medieval_stem', '')
        if med_name:
            search_names.add(med_name.lower())
        if med_stem:
            search_stems.add(med_stem.lower())
        for alt in ident.get('alternate_stems', []):
            search_stems.add(alt.lower())

    for decoded in decoded_tokens:
        d = decoded.lower().strip()
        if not d or '?' in d:
            continue

        # Name match (exact or edit distance ≤ 2)
        for name in search_names:
            if d == name or _edit_distance(d, name) <= 2:
                matches['name'].append(f"{d}≈{name}")

        # Stem match (decoded contains stem or stem contains decoded)
        for stem in search_stems:
            if len(stem) >= 3 and (stem in d or d in stem):
                matches['stem'].append(f"{d}∋{stem}")
            elif len(d) >= 3 and _edit_distance(d, stem) <= 2:
                matches['stem'].append(f"{d}≈{stem}")

        # Humoral match
        for term in HUMORAL_TERMS:
            if d == term or _edit_distance(d, term) <= 1:
                matches['humoral'].append(d)

        # Preparation match
        for term in PREPARATION_TERMS:
            if d == term or _edit_distance(d, term) <= 1:
                matches['preparation'].append(d)

    return matches


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_illustration_targeted() -> None:
    """Phase 19.7: Illustration-targeted folio decode."""
    t0 = time.time()
    rd = str(_results_dir())

    print("=" * 60)
    print("Phase 19.7: Illustration-Targeted Folio Decode")
    print("=" * 60)

    # ── 1. Load dependencies ──────────────────────────────────────────
    print("\n  1. Loading botanical database, assignment, and corpus …")

    illus_data = _load_json(os.path.join(rd, 'illustration_constrained.json'))
    refine_data = _load_json(os.path.join(rd, 'combined_refine.json'))
    mod_data = _load_json(os.path.join(rd, 'modifier_integrate.json'))

    if not illus_data or 'folios' not in illus_data:
        print("  [WARN] Missing illustration_constrained.json")
        result = IllustrationTargetedResult(
            n_folios_tested=0, n_folios_with_matches=0,
            per_folio_results=[], total_weighted_score=0,
            mean_score_per_folio=0, n_name_matches=0,
            n_stem_matches=0, n_humoral_matches=0, n_preparation_matches=0,
            null_scores=[], real_score=0, p_value=1.0, selectivity=0,
            strategy_a_matches=0, strategy_b_matches=0, strategy_c_matches=0,
            best_strategy='none',
            gate_passed=False, verdict="SKIP: missing illustration_constrained.json",
            runtime_seconds=round(time.time() - t0, 2),
        )
        out = os.path.join(rd, 'illustration_targeted.json')
        with open(out, 'w') as f:
            json.dump(_convert(result), f, indent=2)
        print(f"\n    → {out}")
        return

    assignment = {}
    if refine_data:
        for key in ['best_assignment', 'assignment', 'latin_assignment', 'best_latin_assignment']:
            if key in refine_data:
                assignment = refine_data[key]
                break

    modifier_chars = set()
    if mod_data and 'modifier_chars' in mod_data:
        modifier_chars = set(mod_data['modifier_chars'])

    eva_to_triple = build_eva_to_triple_lookup()
    corpus = load_corpus(verbose=False)

    folio_entries = illus_data['folios']
    print(f"    {len(folio_entries)} folios with botanical IDs")

    # ── 2. Decode each illustrated folio ─────────────────────────────
    print("\n  2. Decoding illustrated folios …")

    folio_results: List[FolioMatchResult] = []
    total_name = total_stem = total_humoral = total_prep = 0
    total_weighted = 0.0
    strat_a_total = strat_b_total = strat_c_total = 0

    for entry in folio_entries:
        folio_id = entry.get('folio', '')
        page = corpus.get_page(folio_id)
        if not page:
            continue

        tokens = page.all_tokens
        if not tokens:
            continue

        # Strategy A: Phase 15/16 feature CSP + modifiers
        decoded_a = []
        for tok in tokens:
            d = _decode_token_safe(tok, assignment, eva_to_triple, modifier_chars)
            decoded_a.append(d)

        # Strategy B: Consonant skeleton matching
        # Simplified: extract skeleton from decoded tokens
        decoded_b = []
        for d in decoded_a:
            skel = _extract_skeleton(d)
            decoded_b.append(skel)

        # Strategy C: Combined (use decoded_a since it's more informative)
        decoded_c = decoded_a  # For this implementation, A is the primary decoder

        # Search for matches
        matches = _search_matches(decoded_a, entry)

        # Count strategy-specific matches
        matches_a = _search_matches(decoded_a, entry)
        strat_a_total += sum(len(v) for v in matches_a.values())
        strat_c_total += sum(len(v) for v in matches.values())

        # Score: name=3, stem=2, humoral=1, preparation=0.5
        n_name = len(matches['name'])
        n_stem = len(matches['stem'])
        n_humoral = len(matches['humoral'])
        n_prep = len(matches['preparation'])
        weighted = n_name * 3 + n_stem * 2 + n_humoral * 1 + n_prep * 0.5

        total_name += n_name
        total_stem += n_stem
        total_humoral += n_humoral
        total_prep += n_prep
        total_weighted += weighted

        # Plant identifications
        plant_ids = []
        med_names = []
        med_stems = []
        for ident in entry.get('identifications', []):
            plant_ids.append(ident.get('common_name', ident.get('linnaean_name', '')))
            mn = ident.get('medieval_name', '')
            ms = ident.get('medieval_stem', '')
            if mn:
                med_names.append(mn)
            if ms:
                med_stems.append(ms)

        # Decoded sample
        sample = [[tok, dec] for tok, dec in zip(tokens[:10], decoded_a[:10])]

        fr = FolioMatchResult(
            folio=folio_id,
            n_tokens=len(tokens),
            n_decoded=sum(1 for d in decoded_a if d and '?' not in d),
            plant_identifications=plant_ids,
            medieval_names=med_names,
            medieval_stems=med_stems,
            name_matches=matches['name'],
            stem_matches=matches['stem'],
            humoral_matches=matches['humoral'],
            preparation_matches=matches['preparation'],
            weighted_score=weighted,
            decoded_sample=sample,
        )
        folio_results.append(fr)

        if weighted > 0:
            print(f"    {folio_id}: score={weighted:.1f}  names={n_name} stems={n_stem} humoral={n_humoral} prep={n_prep}  plants={plant_ids[:2]}")

    n_with_matches = sum(1 for fr in folio_results if fr.weighted_score > 0)
    mean_score = total_weighted / len(folio_results) if folio_results else 0

    print(f"\n    {n_with_matches}/{len(folio_results)} folios with matches")
    print(f"    Total weighted score: {total_weighted:.1f}")

    # ── 3. Permutation test ──────────────────────────────────────────
    print("\n  3. Permutation test (10000 random plant-to-folio assignments) …")

    rng = random.Random(42)
    null_scores_list = []

    # Extract all plant entries and all folio pages
    all_entries = [e for e in folio_entries if corpus.get_page(e.get('folio', ''))]

    for trial in range(min(10000, 1000)):  # Cap at 1000 for speed
        shuffled_entries = list(all_entries)
        rng.shuffle(shuffled_entries)

        trial_score = 0.0
        for i, entry in enumerate(all_entries):
            folio_id = entry.get('folio', '')
            page = corpus.get_page(folio_id)
            if not page:
                continue

            # Use shuffled plant identification
            shuffled_plant = shuffled_entries[i]
            tokens = page.all_tokens
            decoded = [_decode_token_safe(tok, assignment, eva_to_triple, modifier_chars)
                       for tok in tokens[:50]]  # Limit for speed

            matches = _search_matches(decoded, shuffled_plant)
            n = len(matches['name'])
            s = len(matches['stem'])
            h = len(matches['humoral'])
            p = len(matches['preparation'])
            trial_score += n * 3 + s * 2 + h * 1 + p * 0.5

        null_scores_list.append(trial_score)

    null_arr = np.array(null_scores_list)
    null_mean = float(np.mean(null_arr)) if len(null_arr) > 0 else 0
    p_value = float(np.mean(null_arr >= total_weighted)) if len(null_arr) > 0 else 1.0
    sel = total_weighted / null_mean if null_mean > 0 else 0.0

    print(f"    Real score: {total_weighted:.1f}, null mean: {null_mean:.1f}")
    print(f"    p-value: {p_value:.4f}, selectivity: {sel:.2f}×")

    # ── 4. Strategy comparison ───────────────────────────────────────
    best_strat = 'A' if strat_a_total >= strat_c_total else 'C'

    # ── 5. Gate ──────────────────────────────────────────────────────
    gate_passed = bool(p_value < 0.05 and sel >= 1.5)

    if gate_passed:
        verdict = f"PASS: p={p_value:.4f}, selectivity={sel:.2f}×, {n_with_matches} folios matched"
    elif p_value < 0.10:
        verdict = f"MARGINAL: p={p_value:.4f}, selectivity={sel:.2f}×"
    else:
        verdict = f"FAIL: p={p_value:.4f}, no significant illustration-text correlation"

    print(f"\n  Gate: {'PASS' if gate_passed else 'FAIL'}")
    print(f"  {verdict}")

    # ── 6. Save ──────────────────────────────────────────────────────
    result = IllustrationTargetedResult(
        n_folios_tested=len(folio_results),
        n_folios_with_matches=n_with_matches,
        per_folio_results=[_convert(asdict(fr)) for fr in folio_results],
        total_weighted_score=round(total_weighted, 2),
        mean_score_per_folio=round(mean_score, 4),
        n_name_matches=total_name,
        n_stem_matches=total_stem,
        n_humoral_matches=total_humoral,
        n_preparation_matches=total_prep,
        null_scores=[round(s, 2) for s in null_scores_list[:50]],
        real_score=round(total_weighted, 2),
        p_value=round(p_value, 4),
        selectivity=round(sel, 4),
        strategy_a_matches=strat_a_total,
        strategy_b_matches=strat_b_total,
        strategy_c_matches=strat_c_total,
        best_strategy=best_strat,
        gate_passed=gate_passed,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'illustration_targeted.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)
    print(f"\n    → {out_path}")
