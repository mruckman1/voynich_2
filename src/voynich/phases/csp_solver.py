"""
Phase 11 – Core CSP engine for phonetic decoding
==================================================
Constraint propagation (AC-3) and beam-search solver that maps 14 Voynich
grid cells to CV syllables in a target Romance language.
"""

import copy
import json
import os
import random
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_cell_lookup,
    load_corpus,
    token_to_grid_cells,
    tokenize_eva_chars,
)
from voynich.core.reference import (
    build_cv_syllable_table,
    load_reference_corpus,
)

from voynich.phases.csp_constraints import (
    AnchorConstraint,
    VerbConstraint,
    PhonemeInventory,
    build_anchor_constraints,
    build_phoneme_inventory,
    check_phonotactic_legality,
    composite_score,
    prune_by_frequency,
    prune_by_inventory,
    prune_by_phonotactics,
    score_anchor_match,
    score_cross_entropy,
    score_verb_consistency,
    score_word_validity,
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CSPVariable:
    """One grid cell that needs a phonetic assignment."""
    cell_key: str
    cv_label: str
    eva_glyphs: List[str]
    frequency: int
    domain: List[str] = field(default_factory=list)


@dataclass
class CSPAssignment:
    """A complete mapping from grid cells to CV syllables with scores."""
    mapping: Dict[str, str]
    score: float = 0.0
    cross_entropy: float = 0.0
    word_validity: float = 0.0
    dict_hit_rate: float = 0.0
    anchor_penalty: float = 0.0
    anchor_match_count: int = 0
    verb_penalty: float = 0.0
    verb_match_count: int = 0
    relaxation_level: int = 0
    decoded_sample: List[Any] = field(default_factory=list)


@dataclass
class CSPResult:
    """Full CSP solution output."""
    language: str
    n_variables: int
    domain_sizes_initial: Dict[str, int]
    domain_sizes_after_propagation: Dict[str, int]
    n_candidates_explored: int
    best_assignment: Dict
    top_k_assignments: List[Dict]
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _convert(obj: Any) -> Any:
    """Convert dataclass/numpy types to JSON-serialisable form."""
    if hasattr(obj, '__dataclass_fields__'):
        return {k: _convert(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _convert(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_convert(item) for item in obj]
    if isinstance(obj, set):
        return sorted(_convert(item) for item in obj)
    if isinstance(obj, float) and (obj != obj):  # NaN
        return None
    if isinstance(obj, (bool, int, float, str, type(None))):
        return obj
    return str(obj)


# ---------------------------------------------------------------------------
# CSP variable construction
# ---------------------------------------------------------------------------

def build_csp_variables(cv_labels: Dict) -> List[CSPVariable]:
    """Initialise 14 CSP variables from ``cv_labels.json`` data."""
    variables: List[CSPVariable] = []
    for cell_key, info in cv_labels.items():
        variables.append(CSPVariable(
            cell_key=cell_key,
            cv_label=info.get('cv_label', ''),
            eva_glyphs=info.get('glyphs', []),
            frequency=info.get('frequency', 0),
        ))
    # Sort by frequency descending (highest-frequency first)
    variables.sort(key=lambda v: v.frequency, reverse=True)
    return variables


def initialise_domains(
    variables: List[CSPVariable],
    inventory: PhonemeInventory,
    cell_frequencies: Dict[str, int],
    anchors: List[AnchorConstraint],
    frequency_slack: int = 3,
) -> List[CSPVariable]:
    """Apply Layers 1–3 + 5 to initialise and prune variable domains.

    1. Inventory constraint  (Layer 1)
    2. Frequency matching    (Layer 2)
    3. Phonotactic legality  (Layer 3)
    4. Anchor soft-fixing    (Layer 5 partial)
    """
    # Start: every cell gets the full CV syllable list
    cell_domains: Dict[str, List[str]] = {
        v.cell_key: list(inventory.cv_syllables) for v in variables
    }

    # Layer 1: inventory
    cell_domains = prune_by_inventory(cell_domains, inventory)

    # Layer 2: frequency
    cell_domains = prune_by_frequency(
        cell_domains, cell_frequencies, inventory, slack=frequency_slack,
    )

    # Layer 3: phonotactics
    cell_domains = prune_by_phonotactics(cell_domains, inventory)

    # Layer 5 (partial): anchor hints expand domains to ensure anchor-
    # suggested syllables are always searchable.  We ADD hints rather
    # than intersecting, so the CE scorer (not the prior) decides.
    anchor_hints = _anchor_hints(anchors, inventory)
    legal_cv = set(inventory.cv_syllables)
    for cell_key, hint_syls in anchor_hints.items():
        if cell_key in cell_domains:
            existing = set(cell_domains[cell_key])
            for syl in hint_syls:
                if syl in legal_cv:
                    existing.add(syl)
            cell_domains[cell_key] = list(existing)

    # Write back to variables
    for v in variables:
        v.domain = cell_domains.get(v.cell_key, list(inventory.cv_syllables))

    return variables


def _anchor_hints(
    anchors: List[AnchorConstraint],
    inventory: PhonemeInventory,
) -> Dict[str, List[str]]:
    """Derive soft hints from anchor constraints.

    When a Voynich stem has the same number of cells as the target has
    syllables, we can directly hint each cell.  Otherwise we skip (the
    beam search will handle mismatches via Layer 5 scoring).
    """
    hints: Dict[str, List[str]] = {}  # cell_key -> candidate syllables
    cv_set = set(inventory.cv_syllables)

    for anchor in anchors:
        if len(anchor.voynich_cells) != len(anchor.target_syllables):
            continue  # length mismatch — can't directly align

        for cell_key, target_syl in zip(
            anchor.voynich_cells, anchor.target_syllables
        ):
            # Map target syllable to nearest CV pattern
            syl = target_syl.lower()
            candidates: List[str] = []
            if syl in cv_set:
                candidates.append(syl)
            else:
                # Try just first consonant + first vowel
                vowels = set(inventory.vowels)
                onset = ''
                rest = syl
                while rest and rest[0] not in vowels:
                    onset += rest[0]
                    rest = rest[1:]
                nucleus = rest[0] if rest else ''
                cv = onset + nucleus
                if cv in cv_set:
                    candidates.append(cv)
                elif onset and nucleus:
                    # Try just last consonant + nucleus
                    cv2 = onset[-1] + nucleus
                    if cv2 in cv_set:
                        candidates.append(cv2)
                if not candidates and nucleus and nucleus in cv_set:
                    candidates.append(nucleus)

            if candidates:
                if cell_key in hints:
                    hints[cell_key].extend(candidates)
                else:
                    hints[cell_key] = list(candidates)

    # Deduplicate
    for k in hints:
        hints[k] = list(set(hints[k]))

    return hints


# ---------------------------------------------------------------------------
# AC-3 constraint propagation
# ---------------------------------------------------------------------------

def ac3_propagate(
    variables: List[CSPVariable],
) -> Tuple[bool, List[CSPVariable]]:
    """Arc Consistency 3: prune domains so every value has a compatible partner.

    The main binary constraint is **all-different**: no two cells should
    map to the same syllable.  This is a soft constraint (some cells may
    legitimately share a common syllable like 'a' or 'e'), so we only
    enforce it among cells with very small domains (≤3 values) to avoid
    over-pruning.

    Returns (solvable, variables).
    """
    var_map = {v.cell_key: v for v in variables}
    changed = True

    while changed:
        changed = False
        # If any cell has a single-value domain, remove it from others
        for v in variables:
            if len(v.domain) == 1:
                fixed_val = v.domain[0]
                for other in variables:
                    if other.cell_key == v.cell_key:
                        continue
                    if fixed_val in other.domain and len(other.domain) > 1:
                        other.domain.remove(fixed_val)
                        changed = True
                        if len(other.domain) == 0:
                            return False, variables

    return True, variables


# ---------------------------------------------------------------------------
# Token decoding
# ---------------------------------------------------------------------------

def decode_token(
    token: str,
    assignment: Dict[str, str],
    eva_to_cell: Dict[str, str],
    context_rules: Optional[Dict[str, Dict[str, str]]] = None,
) -> str:
    """Decode a single EVA token to a phonetic string.

    If context_rules is provided, applies position/adjacency-dependent reading
    rules on top of the fixed assignment.  context_rules format:
        {cell_key: {'word_initial': 'syl', 'word_final': 'syl', 'after_vowel': 'syl', ...}}
    """
    chars = tokenize_eva_chars(token)
    cells = [eva_to_cell.get(ch) for ch in chars]
    n = len(cells)
    parts: List[str] = []

    for ci, cell in enumerate(cells):
        if not cell:
            parts.append('?')
            continue

        syl = assignment.get(cell, '?')

        if context_rules and cell in context_rules:
            # Determine context for this cell position
            if n == 1:
                ctx = 'word_initial'
            elif ci == 0:
                ctx = 'word_initial'
            elif ci == n - 1:
                ctx = 'word_final'
            else:
                # Medial: check predecessor cell type
                pred_cell = cells[ci - 1]
                if pred_cell:
                    # Classify predecessor as vowel-dominant or not
                    pred_parts = pred_cell.split(',')
                    nucleus = pred_parts[1] if len(pred_parts) > 1 else ''
                    if 'loop' in nucleus or 'tail' in nucleus or 'descender' in nucleus:
                        ctx = 'after_vowel'
                    else:
                        ctx = 'before_vowel'
                else:
                    ctx = 'default'

            cell_ctx_rules = context_rules[cell]
            if ctx in cell_ctx_rules:
                syl = cell_ctx_rules[ctx]
            elif 'default' in cell_ctx_rules:
                syl = cell_ctx_rules['default']

        parts.append(syl)
    return ''.join(parts)


def decode_corpus(
    tokens: List[str],
    assignment: Dict[str, str],
    eva_to_cell: Dict[str, str],
    max_tokens: int = 2000,
) -> List[str]:
    """Decode a list of Voynich tokens."""
    return [
        decode_token(t, assignment, eva_to_cell)
        for t in tokens[:max_tokens]
    ]


# ---------------------------------------------------------------------------
# Assignment scoring
# ---------------------------------------------------------------------------

def score_assignment_full(
    assignment: Dict[str, str],
    lm: Dict,
    voynich_tokens: List[str],
    eva_to_cell: Dict[str, str],
    anchors: List[AnchorConstraint],
    inventory: PhonemeInventory,
    ref_word_set: Optional[set] = None,
    verb_constraints: Optional[List[VerbConstraint]] = None,
    relaxation_level: int = 0,
    max_tokens: int = 2000,
) -> CSPAssignment:
    """Fully score a complete assignment."""
    from voynich.phases.csp_constraints import score_dict_hit_rate

    # Cross-entropy (Layer 6) — use correctly-formatted text
    ce = score_cross_entropy(
        assignment, lm, voynich_tokens, eva_to_cell, max_tokens=max_tokens,
    )

    # Decode tokens for word validity and dict hit
    decoded = decode_corpus(voynich_tokens, assignment, eva_to_cell, max_tokens)

    # Word validity (Layer 4)
    validity = score_word_validity(decoded, inventory)

    # Dictionary hit rate
    if ref_word_set:
        dict_hit = score_dict_hit_rate(
            assignment, voynich_tokens, eva_to_cell, ref_word_set, max_tokens,
        )
    else:
        dict_hit = 0.0

    # Anchor match (Layer 5)
    anchor_pen, anchor_n = score_anchor_match(assignment, anchors, eva_to_cell)

    # Verb consistency (Layer 7)
    verb_pen = 0.0
    verb_n = 0
    if verb_constraints:
        verb_pen, verb_n = score_verb_consistency(assignment, verb_constraints, eva_to_cell)

    # Diversity: count distinct syllables assigned
    n_distinct = len(set(assignment.values()))
    n_cells = len(assignment)

    # Composite — dict hit rate improves the score
    score = composite_score(
        ce, validity, anchor_pen, anchor_n, len(anchors),
        verb_penalty=verb_pen,
        n_distinct_syllables=n_distinct,
        n_cells=n_cells,
    ) - dict_hit  # lower score is better; more dict hits = lower score

    # Decoded sample (first 20)
    sample = list(zip(voynich_tokens[:20], decoded[:20]))

    return CSPAssignment(
        mapping=dict(assignment),
        score=score,
        cross_entropy=ce,
        word_validity=validity,
        dict_hit_rate=dict_hit,
        anchor_penalty=anchor_pen,
        anchor_match_count=anchor_n,
        verb_penalty=verb_pen,
        verb_match_count=verb_n,
        relaxation_level=relaxation_level,
        decoded_sample=sample,
    )


# ---------------------------------------------------------------------------
# Beam search solver
# ---------------------------------------------------------------------------

def _partial_score(
    partial: Dict[str, str],
    lm: Dict,
    tokens: List[str],
    eva_to_cell: Dict[str, str],
    max_tokens: int = 500,
) -> float:
    """Quick cross-entropy estimate for a partial assignment.

    Only decodes tokens where ALL cells in the token are assigned.
    Falls back to a neutral score when no tokens are fully decodable.
    """
    from voynich.core.stats import cross_entropy_lm

    decoded_words: List[str] = []
    assigned_cells = set(partial.keys())

    for token in tokens[:max_tokens]:
        chars = tokenize_eva_chars(token)
        cells = [eva_to_cell.get(ch) for ch in chars]
        # Only decode tokens where every mapped cell is already assigned
        if all(c is None or c in assigned_cells for c in cells):
            parts = []
            for ch in chars:
                cell = eva_to_cell.get(ch)
                if cell and cell in partial:
                    parts.append(partial[cell])
            if parts:
                decoded_words.append(''.join(parts))

    if not decoded_words:
        return 10.0  # neutral — no info yet

    decoded_text = '_' + '_'.join(decoded_words) + '_'
    return cross_entropy_lm(decoded_text, lm, per_char=True)


def beam_search(
    variables: List[CSPVariable],
    lm: Dict,
    voynich_tokens: List[str],
    eva_to_cell: Dict[str, str],
    anchors: List[AnchorConstraint],
    inventory: PhonemeInventory,
    ref_word_set: Optional[set] = None,
    verb_constraints: Optional[List[VerbConstraint]] = None,
    relaxation_level: int = 0,
    beam_width: int = 50,
    max_solutions: int = 20,
    seed: int = 42,
) -> List[CSPAssignment]:
    """Beam search over CSP variable assignments.

    Variables are ordered by MRV (minimum remaining values = smallest
    domain first), unless *verb_constraints* are present in which case
    verb-constrained cells are prioritised first.

    Anchor hints (length-matched Rosetta folios) add a score bonus so
    anchor-consistent assignments are not pruned before final scoring.
    Verb-aligned assignments get an additional VERB_BONUS discount.
    """
    rng = random.Random(seed)

    # Pre-compute anchor hints for bonus scoring during search
    anchor_hint_set: Dict[str, set] = {}
    if anchors:
        for cell_key, hint_syls in _anchor_hints(anchors, inventory).items():
            anchor_hint_set[cell_key] = set(hint_syls)

    # Pre-compute verb hints: cell_key -> set of target syllables
    verb_hint_set: Dict[str, set] = {}
    if verb_constraints:
        for vc in verb_constraints:
            for cell_key, target_syl in zip(vc.voynich_cells, vc.latin_syllables):
                if cell_key not in verb_hint_set:
                    verb_hint_set[cell_key] = set()
                verb_hint_set[cell_key].add(target_syl)

    ANCHOR_BONUS = 0.4  # score reduction per anchor-aligned cell assignment
    VERB_BONUS = 0.5    # score reduction per verb-aligned cell assignment

    # Order variables: verb-constrained cells first, then MRV
    verb_cell_keys = set(verb_hint_set.keys())
    verb_vars = sorted(
        [v for v in variables if v.cell_key in verb_cell_keys],
        key=lambda v: len(v.domain),
    )
    other_vars = sorted(
        [v for v in variables if v.cell_key not in verb_cell_keys],
        key=lambda v: len(v.domain),
    )
    ordered = verb_vars + other_vars

    # Initialise beam with empty assignment
    beam: List[Tuple[Dict[str, str], float]] = [({}, 10.0)]

    for step, var in enumerate(ordered):
        next_beam: List[Tuple[Dict[str, str], float]] = []

        for partial, partial_sc in beam:
            for value in var.domain:
                new_partial = {**partial, var.cell_key: value}

                # Quick forward check: don't assign a value already used
                # by a single-domain (fixed) variable
                used_by_fixed = set()
                for v2 in ordered:
                    if v2.cell_key != var.cell_key and len(v2.domain) == 1:
                        if v2.cell_key in partial:
                            used_by_fixed.add(partial[v2.cell_key])
                if value in used_by_fixed:
                    continue

                # Score every 3 steps or on the last step to save time
                if step % 3 == 0 or step == len(ordered) - 1:
                    sc = _partial_score(
                        new_partial, lm, voynich_tokens, eva_to_cell,
                        max_tokens=300,
                    )
                else:
                    sc = partial_sc  # carry forward

                # Anchor hint bonus: if this cell→value is anchor-aligned,
                # apply a score discount so the candidate isn't pruned.
                if var.cell_key in anchor_hint_set and value in anchor_hint_set[var.cell_key]:
                    sc -= ANCHOR_BONUS

                # Verb hint bonus
                if var.cell_key in verb_hint_set and value in verb_hint_set[var.cell_key]:
                    sc -= VERB_BONUS

                next_beam.append((new_partial, sc))

        # Prune to beam width
        next_beam.sort(key=lambda x: x[1])
        beam = next_beam[:beam_width]

        if not beam:
            break

    # Score all final complete assignments
    results: List[CSPAssignment] = []
    for mapping, _ in beam:
        result = score_assignment_full(
            mapping, lm, voynich_tokens, eva_to_cell,
            anchors, inventory,
            ref_word_set=ref_word_set,
            verb_constraints=verb_constraints,
            relaxation_level=relaxation_level,
        )
        results.append(result)

    results.sort(key=lambda r: r.score)
    return results[:max_solutions]


# ---------------------------------------------------------------------------
# Sanity test: synthetic Voynich → CSP recovery
# ---------------------------------------------------------------------------

def run_csp_solver_test() -> Dict:
    """V1 sanity check: use the REAL grid cells and a known CV mapping.

    1. Load the real 14-cell grid from cv_labels.json
    2. Create a known (fixed) mapping: cell → CV syllable
    3. Encode Latin words through the mapping (syllabify → CV → representative glyph)
    4. Run the CSP solver on the encoded tokens
    5. Check whether the solver recovers the known mapping

    This tests the full pipeline end-to-end with real EVA glyphs.
    """
    from voynich.core.stats import build_ngram_lm, syllabify_latin

    print("=" * 70)
    print("PHASE 11.0: CSP Solver Sanity Test")
    print("=" * 70)

    # Load real grid
    rd = _results_dir()
    cv_path = os.path.join(rd, 'cv_labels.json')
    if not os.path.exists(cv_path):
        print("  [SKIP] cv_labels.json not found")
        return {'verdict': 'skipped', 'reason': 'no_cv_labels'}

    with open(cv_path) as f:
        cv_labels = json.load(f)

    # Load Latin reference corpus
    ref_corpus = load_reference_corpus(verbose=False)
    latin_tokens = ref_corpus.get_combined_tokens('latin')
    if not latin_tokens:
        print("  [SKIP] No Latin reference corpus available")
        return {'verdict': 'skipped', 'reason': 'no_latin_corpus'}

    # Build the real EVA-to-cell lookup
    eva_to_cell = build_eva_to_cell_lookup(cv_labels)

    # Sort cells by frequency (highest first) and create a known mapping
    # using the 14 most common Latin CV syllables
    cells_by_freq = sorted(
        cv_labels.items(),
        key=lambda x: x[1].get('frequency', 0),
        reverse=True,
    )

    # Known CV syllables — one per cell, frequency-ordered
    known_cv = [
        'ra', 'te', 'cu', 'na', 'li', 'me', 'tu', 'si',
        'sa', 'ni', 'de', 'pa', 'bo', 'vi',
    ]
    true_mapping: Dict[str, str] = {}
    for i, (cell_key, info) in enumerate(cells_by_freq):
        if i < len(known_cv):
            true_mapping[cell_key] = known_cv[i]

    # Pick one representative glyph per cell (first glyph)
    cell_to_glyph: Dict[str, str] = {}
    for cell_key, info in cv_labels.items():
        glyphs = info.get('glyphs', [])
        if glyphs:
            cell_to_glyph[cell_key] = glyphs[0]

    # Reverse: CV syllable → representative glyph
    cv_to_glyph: Dict[str, str] = {}
    for cell_key, cv in true_mapping.items():
        glyph = cell_to_glyph.get(cell_key)
        if glyph:
            cv_to_glyph[cv] = glyph

    print(f"\n  True mapping ({len(true_mapping)} cells):")
    for cell_key, cv in sorted(true_mapping.items(), key=lambda x: x[1]):
        glyph = cell_to_glyph.get(cell_key, '?')
        label = cv_labels.get(cell_key, {}).get('cv_label', '?')
        print(f"    {label} ({glyph}) → {cv}")

    # Encode Latin words as sequences of real EVA glyphs
    cv_set = set(known_cv)
    vowels = set('aeiou')
    encoded_tokens: List[str] = []
    for word in latin_tokens[:3000]:
        syls = syllabify_latin(word)
        if not syls:
            continue
        glyph_seq: List[str] = []
        for syl in syls:
            syl_lower = syl.lower()
            # Extract CV pattern
            onset = ''
            rest = syl_lower
            while rest and rest[0] not in vowels:
                onset += rest[0]
                rest = rest[1:]
            nucleus = rest[0] if rest else 'a'
            # Try full onset + nucleus
            cv = (onset[-1] if onset else '') + nucleus
            if cv not in cv_set:
                # Try just vowel
                cv = nucleus
                if cv not in cv_set:
                    cv = 'ra'  # fallback to most common
            glyph = cv_to_glyph.get(cv)
            if glyph:
                glyph_seq.append(glyph)
        if glyph_seq:
            # Concatenate EVA glyphs (no separator — EVA tokenizer handles it)
            encoded_tokens.append(''.join(glyph_seq))

    print(f"\n  Encoded {len(encoded_tokens)} tokens")
    if encoded_tokens:
        print(f"  Samples: {encoded_tokens[:8]}")

    # Verify round-trip on a few tokens
    print(f"\n  Round-trip verification:")
    for tok in encoded_tokens[:5]:
        decoded = decode_token(tok, true_mapping, eva_to_cell)
        print(f"    {tok:20s} → {decoded}")

    # Build LM from Latin token list (NOT a joined string — pass the list directly)
    lm = build_ngram_lm(latin_tokens[:5000], order=3, smoothing=0.01)

    # Build CSP variables
    variables = build_csp_variables(cv_labels)
    cell_frequencies = {v.cell_key: v.frequency for v in variables}

    # Build inventory
    inventory = build_phoneme_inventory('latin', ref_corpus)

    # Initialise domains
    variables = initialise_domains(
        variables, inventory, cell_frequencies, anchors=[],
    )

    print(f"\n  Domain sizes after Layers 1-3:")
    for v in variables:
        has_true = '✓' if true_mapping.get(v.cell_key) in v.domain else '✗'
        print(f"    {v.cv_label}: {len(v.domain):3d} candidates  "
              f"true={true_mapping.get(v.cell_key, '?'):4s} {has_true}")

    # AC-3
    solvable, variables = ac3_propagate(variables)
    print(f"\n  AC-3 solvable: {solvable}")

    # Run beam search
    print(f"\n  Running beam search (width=30)...")
    t0 = time.time()

    assignments = beam_search(
        variables, lm, encoded_tokens[:500], eva_to_cell,
        anchors=[], inventory=inventory,
        beam_width=30, max_solutions=10,
    )

    elapsed = time.time() - t0
    print(f"  Beam search completed in {elapsed:.1f}s")
    print(f"  Found {len(assignments)} candidate assignments")

    # Check recovery accuracy
    if assignments:
        best = assignments[0]
        n_correct = 0
        for cell_key, true_cv in true_mapping.items():
            if best.mapping.get(cell_key) == true_cv:
                n_correct += 1
        accuracy = n_correct / len(true_mapping)
        print(f"\n  Recovery accuracy: {n_correct}/{len(true_mapping)} = {accuracy:.1%}")
        print(f"  Best CE: {best.cross_entropy:.4f}")
        print(f"  Best composite score: {best.score:.4f}")
        print(f"  Word validity: {best.word_validity:.4f}")

        # Show mapping comparison
        print(f"\n  Mapping comparison:")
        for cell_key in sorted(true_mapping.keys()):
            true_cv = true_mapping[cell_key]
            pred_cv = best.mapping.get(cell_key, '?')
            match = '✓' if true_cv == pred_cv else '✗'
            label = cv_labels.get(cell_key, {}).get('cv_label', '?')
            print(f"    {label}: true={true_cv:4s}  pred={pred_cv:4s}  {match}")
    else:
        accuracy = 0.0
        print("\n  No assignments found!")

    # Score the true mapping
    true_result = score_assignment_full(
        true_mapping, lm, encoded_tokens[:500], eva_to_cell,
        anchors=[], inventory=inventory,
    )
    print(f"\n  True mapping scores:")
    print(f"    CE: {true_result.cross_entropy:.4f}")
    print(f"    Composite: {true_result.score:.4f}")
    if assignments:
        print(f"    CSP best CE: {assignments[0].cross_entropy:.4f}")
        print(f"    CSP best is {'better' if assignments[0].cross_entropy < true_result.cross_entropy else 'worse'} than true")

    # Random baseline: the key sanity check is that the TRUE mapping scores
    # significantly better than random assignments on the encoded corpus.
    rng = random.Random(42)
    rand_cells = list(true_mapping.keys())
    rand_sylls = list(inventory.cv_syllables)
    rand_ces: List[float] = []
    for _ in range(100):
        mapping = {}
        rng.shuffle(rand_sylls)
        for i, ck in enumerate(rand_cells):
            mapping[ck] = rand_sylls[i % len(rand_sylls)]
        ce = score_cross_entropy(mapping, lm, encoded_tokens[:500], eva_to_cell, max_tokens=500)
        rand_ces.append(ce)
    rand_mean = sum(rand_ces) / len(rand_ces)
    selectivity = rand_mean / true_result.cross_entropy if true_result.cross_entropy > 0 else 0.0
    print(f"\n  Random baseline mean CE: {rand_mean:.4f}")
    print(f"  Selectivity (random/true): {selectivity:.4f}")

    # Pass if the true mapping is clearly better than random (selectivity ≥ 1.3)
    # OR if any direct cell recoveries were made.
    passed = selectivity >= 1.3 or accuracy >= 0.2

    # Save result
    result = {
        'test': 'csp_solver_sanity',
        'n_cells': len(true_mapping),
        'n_encoded_tokens': len(encoded_tokens),
        'recovery_accuracy': accuracy,
        'recovery_target': 0.2,
        'random_baseline_mean_ce': rand_mean,
        'true_mapping_ce': true_result.cross_entropy,
        'selectivity': selectivity,
        'passed': passed,
        'true_mapping_score': true_result.score,
        'best_csp_ce': assignments[0].cross_entropy if assignments else None,
        'best_csp_score': assignments[0].score if assignments else None,
        'n_assignments_found': len(assignments),
        'elapsed_seconds': elapsed,
    }

    with open(os.path.join(rd, 'csp_solver_test.json'), 'w') as f:
        json.dump(_convert(result), f, indent=2)

    print(f"\n  Sanity test {'PASSED' if result['passed'] else 'FAILED'}")
    print(f"  Results saved to results/csp_solver_test.json")

    return result
