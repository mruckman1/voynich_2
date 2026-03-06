"""
Phase 20.3 – Constrained Tachygraphic Grid Solve
==================================================
CSP-solve the full tachygraphic table at EVA-character granularity, seeded
by anchors from Step 20.1 and family constraints from Step 20.2.  Reuses
the beam_search() infrastructure from csp_solver.py via duck-typing.

Dependency chain:
    tachy_anchors.json + tachy_families.json + combined_refine.json
    + modifier_integrate.json
        → tachy_grid_solve.json
"""

import json
import math
import os
import random
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
from voynich.core.stats import (
    build_ngram_lm,
    cross_entropy_lm,
    jensen_shannon_divergence,
    selectivity_ratio,
    syllabify_latin,
)
from voynich.phases.csp_constraints import (
    AnchorConstraint,
    PhonemeInventory,
    build_phoneme_inventory,
)
from voynich.phases.csp_solver import (
    CSPAssignment,
    CSPVariable,
    beam_search,
    decode_corpus,
    score_assignment_full,
)


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
class TachyVariable:
    """One EVA character needing a syllable assignment.

    Duck-types to CSPVariable: has cell_key, cv_label, eva_glyphs,
    frequency, domain.
    """
    cell_key: str           # EVA char name
    cv_label: str           # same as cell_key
    eva_glyphs: List[str]   # [cell_key]
    frequency: int          # corpus frequency
    domain: List[str] = field(default_factory=list)
    # Metadata
    glyph_class: str = ''
    first_stroke: str = ''
    last_stroke: str = ''
    is_anchored: bool = False
    anchor_syllable: str = ''
    anchor_tier: int = 0


@dataclass
class TachyGridSolveResult:
    n_variables: int
    n_anchored: int
    n_family_constrained: int
    domain_sizes_initial: Dict[str, int]
    domain_sizes_after_prune: Dict[str, int]
    best_assignment: Dict[str, str]     # EVA char → syllable
    best_dict_hit: float
    best_cross_entropy: float
    top_assignments: List[Dict]
    tachygraphic_table: Dict[str, Dict]
    null_ce_mean: float
    null_ce_std: float
    null_selectivity: float
    stability_agreement: float
    stable: bool
    phase16_baseline_dict_hit: float
    gate_passed: bool
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _load_json(rd: str, fname: str) -> Dict:
    path = os.path.join(rd, fname)
    if not os.path.exists(path):
        print(f"    [WARN] {fname} not found")
        return {}
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Variable building
# ---------------------------------------------------------------------------

def _build_tachy_variables(
    modifier_chars: Set[str],
    char_freqs: Counter,
    anchors_data: Dict,
    families_data: Dict,
    inventory: PhonemeInventory,
) -> List[TachyVariable]:
    """Build one TachyVariable per syllabic EVA character."""
    # All EVA chars
    all_eva = set(EVA_VISUAL_COMPONENTS.keys())
    syllabic_chars = sorted(all_eva - modifier_chars)

    # Build anchor lookup: char → (syllable, tier)
    anchor_map: Dict[str, Tuple[str, int]] = {}
    for a in anchors_data.get('char_anchors', []):
        anchor_map[a['eva_char']] = (a['syllable'], a.get('tier', 3))

    # Build family preliminary table: char → syllable
    family_table = families_data.get('preliminary_table', {})

    # Full CV syllable inventory
    all_syllables = list(inventory.cv_syllables)

    variables = []
    for ch in syllabic_chars:
        comp = EVA_VISUAL_COMPONENTS.get(ch, {})
        freq = char_freqs.get(ch, 0)

        # Determine domain
        if ch in anchor_map and anchor_map[ch][1] == 1:
            # Tier 1 hard anchor — single-value domain
            domain = [anchor_map[ch][0]]
            is_anchored = True
            anchor_syl = anchor_map[ch][0]
            anchor_tier = 1
        elif ch in family_table:
            # Family assignment provides the primary candidate
            fam_syl = family_table[ch]
            # Domain: family syllable + all syllables sharing same consonant
            consonant = ''
            vowels_set = set('aeiou')
            for c in fam_syl:
                if c not in vowels_set:
                    consonant += c
                else:
                    break
            # Build domain from same consonant class
            if consonant:
                same_onset = [s for s in all_syllables
                              if s.startswith(consonant)]
                # Also include the family syllable itself
                domain = list(set(same_onset + [fam_syl]))
            else:
                domain = list(all_syllables)

            is_anchored = ch in anchor_map
            anchor_syl = anchor_map.get(ch, ('', 0))[0]
            anchor_tier = anchor_map.get(ch, ('', 0))[1] if is_anchored else 0
        else:
            # No family or anchor info — full domain
            domain = list(all_syllables)
            is_anchored = ch in anchor_map
            anchor_syl = anchor_map.get(ch, ('', 0))[0]
            anchor_tier = anchor_map.get(ch, ('', 0))[1] if is_anchored else 0

        # Tier 2 anchor: boost but don't restrict
        if is_anchored and anchor_tier == 2 and anchor_syl not in domain:
            domain.append(anchor_syl)

        # Sort domain by frequency rank in inventory
        freq_rank = {s: i for i, s in enumerate(inventory.frequency_ranked)}
        domain.sort(key=lambda s: freq_rank.get(s, len(freq_rank)))

        variables.append(TachyVariable(
            cell_key=ch,
            cv_label=ch,
            eva_glyphs=[ch],
            frequency=freq,
            domain=domain,
            glyph_class=comp.get('glyph_class', ''),
            first_stroke=comp.get('first_stroke', ''),
            last_stroke=comp.get('last_stroke', ''),
            is_anchored=is_anchored,
            anchor_syllable=anchor_syl,
            anchor_tier=anchor_tier,
        ))

    # Sort by frequency descending (MRV will reorder during search)
    variables.sort(key=lambda v: -v.frequency)
    return variables


def _build_identity_eva_to_cell(syllabic_chars: Set[str]) -> Dict[str, str]:
    """Identity mapping: each EVA char maps to itself as cell_key."""
    return {ch: ch for ch in syllabic_chars}


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _score_dict_hit(
    assignment: Dict[str, str],
    tokens: List[str],
    eva_to_cell: Dict[str, str],
    modifier_chars: Set[str],
    ref_word_set: set,
    max_tokens: int = 2000,
) -> float:
    """Score dict hit rate using modifier-aware decode."""
    hits = 0
    total = 0
    for token in tokens[:max_tokens]:
        chars = tokenize_eva_chars(token)
        syllables = []
        for ch in chars:
            if ch in modifier_chars:
                continue
            cell = eva_to_cell.get(ch)
            if cell and cell in assignment:
                syllables.append(assignment[cell])
        decoded = ''.join(syllables)
        if decoded and decoded in ref_word_set:
            hits += 1
        total += 1
    return hits / total if total else 0.0


def _score_cross_entropy_simple(
    assignment: Dict[str, str],
    tokens: List[str],
    eva_to_cell: Dict[str, str],
    modifier_chars: Set[str],
    lm: Dict,
    max_tokens: int = 2000,
) -> float:
    """Simple cross-entropy estimation of decoded text."""
    decoded_parts = []
    for token in tokens[:max_tokens]:
        chars = tokenize_eva_chars(token)
        syllables = []
        for ch in chars:
            if ch in modifier_chars:
                continue
            cell = eva_to_cell.get(ch)
            if cell and cell in assignment:
                syllables.append(assignment[cell])
        decoded_parts.append(''.join(syllables))

    decoded_text = ' '.join(decoded_parts)
    if not decoded_text.strip():
        return 99.0
    return cross_entropy_lm(list(decoded_text), lm)


# ---------------------------------------------------------------------------
# Simulated annealing local optimisation
# ---------------------------------------------------------------------------

def _local_optimize(
    assignment: Dict[str, str],
    variables: List[TachyVariable],
    tokens: List[str],
    eva_to_cell: Dict[str, str],
    modifier_chars: Set[str],
    ref_word_set: set,
    lm: Dict,
    n_iter: int = 30000,
    seed: int = 42,
) -> Tuple[Dict[str, str], float]:
    """Simulated annealing over single-character swaps."""
    rng = random.Random(seed)
    current = dict(assignment)
    # Only optimise non-anchored variables
    free_vars = [v for v in variables if not (v.is_anchored and v.anchor_tier == 1)]
    if not free_vars:
        return current, _score_dict_hit(current, tokens, eva_to_cell,
                                         modifier_chars, ref_word_set)

    current_score = _score_dict_hit(current, tokens, eva_to_cell,
                                     modifier_chars, ref_word_set)
    best = dict(current)
    best_score = current_score

    temp = 1.0
    cooling = 0.99995

    for i in range(n_iter):
        # Pick random free variable and random domain value
        var = rng.choice(free_vars)
        if len(var.domain) < 2:
            continue
        old_val = current.get(var.cell_key, var.domain[0])
        new_val = rng.choice(var.domain)
        if new_val == old_val:
            continue

        current[var.cell_key] = new_val
        new_score = _score_dict_hit(current, tokens[:500], eva_to_cell,
                                     modifier_chars, ref_word_set)

        delta = new_score - current_score
        if delta > 0 or (temp > 0 and rng.random() < math.exp(delta / max(temp, 1e-10))):
            current_score = new_score
            if new_score > best_score:
                best = dict(current)
                best_score = new_score
        else:
            current[var.cell_key] = old_val

        temp *= cooling

    return best, best_score


# ---------------------------------------------------------------------------
# Null baseline
# ---------------------------------------------------------------------------

def _null_baseline(
    variables: List[TachyVariable],
    tokens: List[str],
    eva_to_cell: Dict[str, str],
    modifier_chars: Set[str],
    ref_word_set: set,
    n_trials: int = 10,
) -> Tuple[float, float]:
    """Random assignment baseline for dict_hit."""
    scores = []
    rng = random.Random(42)
    for trial in range(n_trials):
        assignment = {}
        for v in variables:
            assignment[v.cell_key] = rng.choice(v.domain) if v.domain else '?'
        score = _score_dict_hit(assignment, tokens[:1000], eva_to_cell,
                                 modifier_chars, ref_word_set)
        scores.append(score)
    return float(np.mean(scores)), float(np.std(scores))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_tachy_grid_solve() -> None:
    """Step 20.3: CSP-solve the tachygraphic table at char granularity."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE 20.3: Constrained Tachygraphic Grid Solve")
    print("=" * 70)

    rd = _results_dir()

    # ─── 1. Load dependencies ───
    print("\n  1. Loading dependencies …")
    anchors_data = _load_json(rd, 'tachy_anchors.json')
    families_data = _load_json(rd, 'tachy_families.json')
    modifier_data = _load_json(rd, 'modifier_integrate.json')
    modifier_chars = set(modifier_data.get('modifier_chars', []))

    # Corpus
    corpus = load_corpus(verbose=False)
    tokens = corpus.get_tokens()

    # Reference words
    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    base_words = set(
        w.lower() for w in ref_corpus.get_combined_tokens('latin')
        if len(w) >= 2
    )
    expanded_words, _ = build_expanded_word_set(base_words)
    ref_word_set = base_words | expanded_words

    # Phoneme inventory
    inventory = build_phoneme_inventory('latin', ref_corpus)

    # Language model (character trigram)
    ref_text = ' '.join(ref_corpus.get_combined_tokens('latin'))
    lm = build_ngram_lm(list(ref_text.lower()), order=3)

    # Char frequencies
    char_freqs: Counter = Counter()
    for token in tokens:
        for ch in tokenize_eva_chars(token):
            char_freqs[ch] += 1

    print(f"      Tokens: {len(tokens)}")
    print(f"      Reference words: {len(ref_word_set)}")
    print(f"      CV syllables: {len(inventory.cv_syllables)}")

    # ─── 2. Build variables ───
    print("\n  2. Building TachyVariables …")
    variables = _build_tachy_variables(
        modifier_chars, char_freqs, anchors_data, families_data, inventory,
    )
    eva_to_cell = _build_identity_eva_to_cell(
        set(v.cell_key for v in variables)
    )

    n_anchored = sum(1 for v in variables if v.is_anchored)
    n_family = sum(1 for v in variables
                   if v.cell_key in families_data.get('preliminary_table', {}))
    domain_sizes_initial = {v.cell_key: len(v.domain) for v in variables}

    print(f"      Variables: {len(variables)}")
    print(f"      Anchored: {n_anchored}")
    print(f"      Family-constrained: {n_family}")
    print(f"      Mean domain size: {np.mean(list(domain_sizes_initial.values())):.1f}")

    # ─── 3. Build anchor constraints for beam_search ───
    print("\n  3. Building anchor constraints …")
    anchor_constraints: List[AnchorConstraint] = []
    for a in anchors_data.get('char_anchors', []):
        if a.get('tier', 3) <= 2:
            anchor_constraints.append(AnchorConstraint(
                folio='anchor',
                voynich_stem=a['eva_char'],
                voynich_cells=[a['eva_char']],
                target_word=a['syllable'],
                target_syllables=[a['syllable']],
                weight=5.0 if a['tier'] == 1 else 2.0,
            ))
    print(f"      Anchor constraints: {len(anchor_constraints)}")

    # ─── 4. Beam search with restarts ───
    print("\n  4. Running beam search (5 restarts) …")
    all_solutions: List[CSPAssignment] = []

    for restart in range(5):
        print(f"      Restart {restart + 1}/5 …", end=' ', flush=True)
        solutions = beam_search(
            variables=variables,
            lm=lm,
            voynich_tokens=tokens[:2000],
            eva_to_cell=eva_to_cell,
            anchors=anchor_constraints,
            inventory=inventory,
            ref_word_set=ref_word_set,
            beam_width=100,
            max_solutions=5,
            seed=42 + restart * 1000,
        )
        if solutions:
            best = solutions[0]
            print(f"dict_hit={best.dict_hit_rate:.3f} "
                  f"CE={best.cross_entropy:.3f}")
            all_solutions.extend(solutions)
        else:
            print("no solutions")

    # Sort by dict_hit (descending)
    all_solutions.sort(key=lambda s: -s.dict_hit_rate)

    # ─── 5. Local optimisation on top-3 ───
    print("\n  5. Local optimisation (SA) on top-3 …")
    top_results: List[Tuple[Dict[str, str], float]] = []

    for i, sol in enumerate(all_solutions[:3]):
        print(f"      Solution {i + 1}: initial dict_hit={sol.dict_hit_rate:.3f}")
        improved, improved_score = _local_optimize(
            sol.mapping, variables, tokens, eva_to_cell,
            modifier_chars, ref_word_set, lm,
            n_iter=30000, seed=42 + i,
        )
        print(f"        → after SA: dict_hit={improved_score:.3f}")
        top_results.append((improved, improved_score))

    # If no solutions at all, use family table as fallback
    if not top_results:
        print("      [WARN] No beam search solutions — using family table")
        fallback = families_data.get('preliminary_table', {})
        fallback_score = _score_dict_hit(
            fallback, tokens, eva_to_cell, modifier_chars, ref_word_set)
        top_results.append((fallback, fallback_score))

    top_results.sort(key=lambda x: -x[1])
    best_assignment, best_dict_hit = top_results[0]

    # Score best with full scoring
    domain_sizes_after = {v.cell_key: len(v.domain) for v in variables}

    # ─── 6. Cross-entropy for best ───
    print("\n  6. Scoring best assignment …")
    best_ce = _score_cross_entropy_simple(
        best_assignment, tokens, eva_to_cell, modifier_chars, lm,
    )
    print(f"      Best dict_hit: {best_dict_hit:.3f}")
    print(f"      Best CE: {best_ce:.4f}")

    # ─── 7. Null baseline ───
    print("\n  7. Null baseline …")
    null_mean, null_std = _null_baseline(
        variables, tokens, eva_to_cell, modifier_chars, ref_word_set,
    )
    null_sel = best_dict_hit / null_mean if null_mean > 0 else float('inf')
    print(f"      Null dict_hit: {null_mean:.3f} ± {null_std:.3f}")
    print(f"      Selectivity: {null_sel:.2f}×")

    # ─── 8. Stability test ───
    print("\n  8. Stability test …")
    top_mappings = [r[0] for r in top_results[:3]]
    if len(top_mappings) >= 2:
        agreements = []
        keys = sorted(set().union(*[m.keys() for m in top_mappings]))
        for i in range(len(top_mappings)):
            for j in range(i + 1, len(top_mappings)):
                agree = sum(1 for k in keys
                            if top_mappings[i].get(k) == top_mappings[j].get(k))
                agreements.append(agree / len(keys) if keys else 0)
        stability_agreement = float(np.mean(agreements))
    else:
        stability_agreement = 1.0
    stable = stability_agreement >= 0.6
    print(f"      Pairwise agreement: {stability_agreement:.1%}")
    print(f"      Stable: {stable}")

    # ─── 9. Build tachygraphic table ───
    print("\n  9. Building tachygraphic table …")
    tachy_table: Dict[str, Dict] = {}
    for v in variables:
        ch = v.cell_key
        syl = best_assignment.get(ch, '?')
        tachy_table[ch] = {
            'syllable': syl,
            'glyph_class': v.glyph_class,
            'first_stroke': v.first_stroke,
            'last_stroke': v.last_stroke,
            'is_anchored': v.is_anchored,
            'anchor_tier': v.anchor_tier,
            'family_proposed': families_data.get('preliminary_table', {}).get(ch, ''),
        }
        print(f"      {ch:8s} → {syl:4s}  [{v.glyph_class}]"
              f"{'  *anchor*' if v.is_anchored else ''}")

    # Phase 16 baseline
    phase16_data = _load_json(rd, 'modifier_integrate.json')
    phase16_dict_hit = 0.516  # known from Phase 16
    for s in phase16_data.get('strategy_results', []):
        if s.get('strategy') == 'R3_combined':
            phase16_dict_hit = s.get('dict_hit', 0.516)

    # ─── 10. Gate ───
    gate_passed = null_sel > 1.3 and best_dict_hit > 0.10
    if gate_passed:
        verdict = (f"PASS: dict_hit={best_dict_hit:.1%} "
                   f"(null selectivity={null_sel:.2f}×). "
                   f"CE={best_ce:.4f}. "
                   f"Phase 16 baseline={phase16_dict_hit:.1%}.")
    else:
        verdict = (f"FAIL: dict_hit={best_dict_hit:.1%}, "
                   f"null selectivity={null_sel:.2f}× (need >1.3×).")

    print(f"\n  10. Gate: {verdict}")

    # ─── 11. Save ───
    top_serialised = []
    for mapping, score in top_results[:3]:
        top_serialised.append({
            'mapping': mapping,
            'dict_hit': score,
        })

    result = TachyGridSolveResult(
        n_variables=len(variables),
        n_anchored=n_anchored,
        n_family_constrained=n_family,
        domain_sizes_initial=domain_sizes_initial,
        domain_sizes_after_prune=domain_sizes_after,
        best_assignment=best_assignment,
        best_dict_hit=best_dict_hit,
        best_cross_entropy=best_ce,
        top_assignments=top_serialised,
        tachygraphic_table=tachy_table,
        null_ce_mean=null_mean,
        null_ce_std=null_std,
        null_selectivity=null_sel,
        stability_agreement=stability_agreement,
        stable=stable,
        phase16_baseline_dict_hit=phase16_dict_hit,
        gate_passed=gate_passed,
        verdict=verdict,
        runtime_seconds=time.time() - t0,
    )

    out_path = os.path.join(rd, 'tachy_grid_solve.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)

    print(f"\n  → {out_path}")
