"""
Phase 31.4: Botanical Signal Validation
==========================================
If plant propagation produced an expanded table, validate it by checking
whether newly decoded text contains domain-appropriate botanical vocabulary
on the correct folios.

Dependency chain:
    plant_name_propagate.json  (Step 31.3)
    consensus_plants.json      (Step 31.1)
    combined_refine.json       (Phase 15)
    modifier_integrate.json    (Phase 16)
        → botanical_signal.json  (this step)
"""

import json
import os
import random
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    load_corpus,
)
from voynich.core.reference import (
    build_expanded_word_set,
    load_reference_corpus,
)
from voynich.phases.null_corpus import _reconstruct_modifier_rules
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


# Botanical vocabulary categories
PREPARATION_TERMS = {
    'coque', 'tere', 'misce', 'cola', 'destilla', 'bibe',
    'contere', 'incide', 'dissolve', 'adde', 'pone', 'recipe',
    'fac', 'da', 'cura', 'sana', 'lava',
}

PLANT_PART_TERMS = {
    'radix', 'radice', 'folia', 'folium', 'flos', 'flore',
    'semen', 'semine', 'cortex', 'cortice', 'herba', 'herbe',
    'succus', 'succo', 'ramus', 'ramo',
}

HUMORAL_TERMS = {
    'calidus', 'calida', 'calidum', 'frigidus', 'frigida', 'frigidum',
    'siccus', 'sicca', 'siccum', 'humidus', 'humida', 'humidum',
    'temperatus', 'temperata', 'temperatum',
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FolioVocabMatch:
    """Vocabulary matches for one folio."""
    folio: str
    genus: str
    expected_plant_names: List[str]
    plant_name_hits: List[str]
    humoral_hits: List[str]
    plant_part_hits: List[str]
    preparation_hits: List[str]
    total_domain_hits: int
    n_tokens: int
    domain_hit_rate: float


@dataclass
class BotanicalSignalResult:
    """Full Step 31.4 output."""
    n_folios_tested: int
    per_folio_matches: List[Dict]
    total_plant_name_hits: int
    total_domain_hits: int
    mean_domain_hit_rate: float
    permutation_p_value: float
    n_permutations: int
    best_folio: str
    best_folio_domain_hits: int
    annotated_passage: str
    gate_passed: bool
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def _edit_distance(a: str, b: str) -> int:
    """Simple Levenshtein edit distance."""
    if len(a) < len(b):
        return _edit_distance(b, a)
    if len(b) == 0:
        return len(a)

    prev_row = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr_row = [i + 1]
        for j, cb in enumerate(b):
            cost = 0 if ca == cb else 1
            curr_row.append(min(
                curr_row[j] + 1,
                prev_row[j + 1] + 1,
                prev_row[j] + cost,
            ))
        prev_row = curr_row

    return prev_row[-1]


def _search_plant_vocabulary(
    decoded_words: List[str],
    expected_names: List[str],
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """Search decoded text for plant-related vocabulary."""
    plant_hits = []
    humoral_hits = []
    part_hits = []
    prep_hits = []

    word_set = set(decoded_words)

    # Plant name matches (exact or edit distance ≤ 2)
    for word in word_set:
        for name in expected_names:
            if word == name or _edit_distance(word, name) <= 2:
                plant_hits.append(f"{word}≈{name}")
                break

    # Humoral, plant-part, preparation matches
    for word in word_set:
        if word in HUMORAL_TERMS:
            humoral_hits.append(word)
        if word in PLANT_PART_TERMS:
            part_hits.append(word)
        if word in PREPARATION_TERMS:
            prep_hits.append(word)

    return plant_hits, humoral_hits, part_hits, prep_hits


def _decode_folio(
    folio: str,
    corpus,
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    modifier_chars: set,
    modifier_rules: Dict[str, str],
    ref_word_set: set,
) -> List[str]:
    """Decode all tokens on a folio."""
    page = corpus.pages.get(folio)
    if not page:
        return []

    tokens = page.all_tokens
    return _decode_corpus_r3(
        tokens, assignment, eva_to_triple,
        modifier_chars, modifier_rules, ref_word_set,
    )


def _permutation_test(
    folio_matches: List[FolioVocabMatch],
    n_perms: int = 1000,
) -> float:
    """Permutation test: does the real folio-plant assignment produce more matches?"""
    if not folio_matches:
        return 1.0

    real_total = sum(fm.total_domain_hits for fm in folio_matches)

    # Permute: randomly reassign plant names to folios
    rng = random.Random(42)
    n_better = 0
    all_hits_lists = [fm.total_domain_hits for fm in folio_matches]

    for _ in range(n_perms):
        shuffled = list(all_hits_lists)
        rng.shuffle(shuffled)
        perm_total = sum(shuffled)
        if perm_total >= real_total:
            n_better += 1

    return (n_better + 1) / (n_perms + 1)


def _annotated_transliteration(
    folio: str,
    corpus,
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    modifier_chars: set,
    modifier_rules: Dict[str, str],
    ref_word_set: set,
    confirmed_words: Set[str],
    plant_words: Set[str],
) -> str:
    """Produce annotated transliteration of a folio."""
    page = corpus.pages.get(folio)
    if not page:
        return ""

    tokens = page.all_tokens
    decoded = _decode_corpus_r3(
        tokens, assignment, eva_to_triple,
        modifier_chars, modifier_rules, ref_word_set,
    )

    parts = []
    for token, word in zip(tokens, decoded):
        if word in plant_words:
            parts.append(f"[PLANT:{word}]")
        elif word in confirmed_words:
            parts.append(f"[CONFIRMED:{word}]")
        elif word in ref_word_set:
            parts.append(f"[HIT:{word}]")
        else:
            parts.append(f"[MISS:{word}]")

    return ' '.join(parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_botanical_signal() -> None:
    """Step 31.4: Validate botanical signal on expanded table."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE 31.4: Botanical Signal Validation")
    print("=" * 70)

    rd = _results_dir()
    eva_to_triple = build_eva_to_triple_lookup()

    # ── 1. Load inputs ──
    print("\n  1. Loading inputs...")

    # Propagation results
    prop_path = os.path.join(rd, 'plant_name_propagate.json')
    if not os.path.exists(prop_path):
        print("  [SKIP] plant_name_propagate.json not found — run plant-prop first")
        return
    with open(prop_path) as f:
        prop_data = json.load(f)

    # Consensus plants
    cp_path = os.path.join(rd, 'consensus_plants.json')
    with open(cp_path) as f:
        cp_data = json.load(f)

    # Assignment (use expanded if available, else base)
    with open(os.path.join(rd, 'combined_refine.json')) as f:
        refine_data = json.load(f)
    assignment = refine_data.get('best_assignment', {})

    # Apply plant-derived assignments if any
    plant_new = prop_data.get('plant_new_assignments', {})
    if plant_new:
        assignment.update(plant_new)

    with open(os.path.join(rd, 'modifier_integrate.json')) as f:
        mod_data = json.load(f)
    modifier_chars, modifier_rules = _reconstruct_modifier_rules(mod_data)

    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    base_words = set(w.lower() for w in ref_corpus.get_combined_tokens('latin')
                     if len(w) >= 2)
    expanded, _ = build_expanded_word_set(base_words)
    ref_word_set = base_words | expanded

    corpus = load_corpus(verbose=False)
    print(f"     Assignment: {len(assignment)} triples "
          f"({len(plant_new)} plant-derived)")

    # ── 2. Decode Tier A+B folios ──
    print("\n  2. Decoding Tier A+B folio texts...")
    tier_ab = cp_data.get('tier_a_folios', []) + cp_data.get('tier_b_folios', [])

    folio_matches: List[FolioVocabMatch] = []
    confirmed_words = {'bene', 'de', 'sero', 'sene', 'raro', 'dine',
                       'cola', 'codi', 'dico', 'ci'}  # Phase 30 signal words

    for entry in tier_ab:
        folio = entry['folio']
        genus = entry.get('consensus', {}).get('genus', '?')
        med_names = entry.get('medieval_names', [])
        expected_names = []
        for mn in med_names:
            name = mn.get('medieval_name', '')
            if name:
                expected_names.append(name.lower())
            for alt in mn.get('alternate_names', []):
                if alt:
                    expected_names.append(alt.lower())

        decoded = _decode_folio(
            folio, corpus, assignment, eva_to_triple,
            modifier_chars, modifier_rules, ref_word_set,
        )

        plant_hits, humoral_hits, part_hits, prep_hits = _search_plant_vocabulary(
            decoded, expected_names,
        )

        total = len(plant_hits) + len(humoral_hits) + len(part_hits) + len(prep_hits)
        n_tokens = len(decoded)
        rate = total / max(n_tokens, 1)

        match = FolioVocabMatch(
            folio=folio,
            genus=genus,
            expected_plant_names=expected_names,
            plant_name_hits=plant_hits,
            humoral_hits=humoral_hits,
            plant_part_hits=part_hits,
            preparation_hits=prep_hits,
            total_domain_hits=total,
            n_tokens=n_tokens,
            domain_hit_rate=round(rate, 4),
        )
        folio_matches.append(match)

        print(f"     {folio} ({genus}): {total} domain hits "
              f"(plant={len(plant_hits)}, humoral={len(humoral_hits)}, "
              f"parts={len(part_hits)}, prep={len(prep_hits)})")

    # ── 3. Permutation test ──
    print("\n  3. Permutation test (1000 permutations)...")
    p_value = _permutation_test(folio_matches, n_perms=1000)
    print(f"     p-value: {p_value:.4f}")

    # ── 4. Annotated passage from best folio ──
    best_folio_match = max(folio_matches, key=lambda m: m.total_domain_hits) if folio_matches else None
    best_folio = best_folio_match.folio if best_folio_match else ''
    best_hits = best_folio_match.total_domain_hits if best_folio_match else 0

    plant_word_set = set()
    if best_folio_match:
        for ph in best_folio_match.plant_name_hits:
            plant_word_set.add(ph.split('≈')[0])

    annotated = ''
    if best_folio:
        print(f"\n  4. Annotated passage from {best_folio}...")
        annotated = _annotated_transliteration(
            best_folio, corpus, assignment, eva_to_triple,
            modifier_chars, modifier_rules, ref_word_set,
            confirmed_words, plant_word_set,
        )
        # Print first 200 chars
        print(f"     {annotated[:200]}...")

    # ── 5. Summary ──
    total_plant = sum(len(fm.plant_name_hits) for fm in folio_matches)
    total_domain = sum(fm.total_domain_hits for fm in folio_matches)
    rates = [fm.domain_hit_rate for fm in folio_matches]
    mean_rate = sum(rates) / len(rates) if rates else 0.0

    gate = p_value < 0.05
    if gate and total_plant > 0:
        verdict = "BOTANICAL_SIGNAL_CONFIRMED"
    elif total_domain > 0:
        verdict = "BOTANICAL_VOCABULARY_FOUND"
    else:
        verdict = "NO_BOTANICAL_SIGNAL"

    print(f"\n  Gate: {'PASS' if gate else 'FAIL'} (p={p_value:.4f})")
    print(f"  Verdict: {verdict}")
    print(f"  Plant name hits: {total_plant}, Total domain hits: {total_domain}")

    # ── 6. Save ──
    result = BotanicalSignalResult(
        n_folios_tested=len(folio_matches),
        per_folio_matches=[_convert(asdict(fm)) for fm in folio_matches],
        total_plant_name_hits=total_plant,
        total_domain_hits=total_domain,
        mean_domain_hit_rate=round(mean_rate, 4),
        permutation_p_value=round(p_value, 4),
        n_permutations=1000,
        best_folio=best_folio,
        best_folio_domain_hits=best_hits,
        annotated_passage=annotated[:2000],
        gate_passed=gate,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'botanical_signal.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2)

    print(f"\n  Saved → {out_path}")
    print(f"  Completed in {time.time() - t0:.1f}s")
