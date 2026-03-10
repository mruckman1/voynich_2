"""
Phase 34.5 – Slot-Conditioned CSP Variables
=============================================
Creates position-conditioned CSP variables where the same stroke triple
receives different phoneme assignments depending on its morphological slot
(PREFIX, ROOT, or SUFFIX).

Rationale:
  Phase 33 found that 18/25 triples receive conflicting recommendations from
  different correction approaches.  A single global assignment per triple
  cannot simultaneously satisfy prefix, root, and suffix contexts.  By forking
  triples into slot-specific variables, each slot can receive its own domain:
  - PREFIX vars: CV syllables (same as Phase 14)
  - ROOT vars: CV syllables
  - SUFFIX vars: Latin inflectional endings

Algorithm:
  1. Decompose every token with decompose_token_morphemes()
  2. Tag each EVA char with its morphological slot
  3. For each of the 25 stroke triples, count how often it appears in each slot
  4. Fork variables for multi-slot triples (>=10% in >=2 slots)
  5. Assign slot-specific domains

Dependency chain:
    combined_refine.json      (Phase 15 assignment)
    morpheme_grid.json        (morpheme decomposition rules)
    feature_csp.json          (base 25 feature variables)
        -> slot_variables.json  (this step)
"""

import json
import os
import time
from collections import Counter, defaultdict
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
    build_cv_syllable_table,
)
from voynich.phases.morpheme_grid import (
    decompose_token_morphemes,
    KNOWN_PREFIXES,
    KNOWN_SUFFIXES,
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
# Constants
# ---------------------------------------------------------------------------

# Latin inflectional endings used as suffix variable domains
LATIN_SUFFIX_ENDINGS = [
    'a', 'ae', 'am', 'arum', 'as',
    'e', 'i', 'is', 'o', 'um', 'us',
    'unt', 'tur', 're', 'ri',
]

# Minimum fraction of a triple's occurrences in a slot for it to be
# considered "active" in that slot.
SLOT_THRESHOLD = 0.10


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TripleSlotProfile:
    """Per-triple distribution across morphological slots."""
    triple_key: str
    eva_glyphs: List[str]
    total_count: int
    prefix_count: int
    root_count: int
    suffix_count: int
    prefix_frac: float
    root_frac: float
    suffix_frac: float
    n_active_slots: int
    is_forked: bool
    forked_slots: List[str]


@dataclass
class ForkedVariable:
    """A slot-specific variable forked from a base triple."""
    variable_key: str     # e.g. "ascender,descender,suffix@ROOT"
    base_triple: str      # the original triple_key
    slot: str             # PREFIX, ROOT, or SUFFIX
    eva_glyphs: List[str]
    frequency: int        # count of this triple in this slot
    domain: List[str]     # candidate syllables for this slot


@dataclass
class SlotVariableResult:
    """Full Step 34.5 output."""
    n_base_triples: int
    n_forked_variables: int
    n_prefix_vars: int
    n_root_vars: int
    n_suffix_vars: int
    n_unforked: int
    slot_profiles: List[Dict]
    forked_variable_definitions: List[Dict]
    # Corpus-level morpheme stats
    n_tokens: int
    n_with_prefix: int
    n_with_suffix: int
    n_stem_only: int
    pct_with_prefix: float
    pct_with_suffix: float
    # Domain sizes
    cv_syllable_count: int
    suffix_domain_count: int
    mean_domain_size: float
    # Comparison to Phase 14
    phase14_n_variables: int
    variable_expansion_ratio: float
    # Phase 33 orthogonality
    n_conflicting_triples_phase33: int
    n_conflicting_now_forked: int
    orthogonality_resolution_rate: float
    gate_passed: bool
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Slot tagging
# ---------------------------------------------------------------------------

def _tag_chars_with_slots(
    token: str,
) -> List[Tuple[str, str]]:
    """Tag each EVA character in a token with its morphological slot.

    Returns list of (eva_char, slot) pairs where slot is one of
    'PREFIX', 'ROOT', 'SUFFIX'.
    """
    decomp = decompose_token_morphemes(token)
    tagged: List[Tuple[str, str]] = []
    for ch in decomp.prefix_glyphs:
        tagged.append((ch, 'PREFIX'))
    for ch in decomp.stem_glyphs:
        tagged.append((ch, 'ROOT'))
    for ch in decomp.suffix_glyphs:
        tagged.append((ch, 'SUFFIX'))
    return tagged


def _compute_slot_profiles(
    all_tokens: List[str],
    eva_to_triple: Dict[str, str],
) -> Tuple[List[TripleSlotProfile], Dict[str, Dict[str, int]]]:
    """Compute per-triple slot distributions across the corpus.

    Returns (profiles, triple_slot_counts) where triple_slot_counts is
    {triple_key: {'PREFIX': n, 'ROOT': n, 'SUFFIX': n}}.
    """
    # Count triple occurrences per slot
    triple_slot_counts: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {'PREFIX': 0, 'ROOT': 0, 'SUFFIX': 0}
    )
    # Track which glyphs belong to each triple
    triple_to_glyphs: Dict[str, Set[str]] = defaultdict(set)

    for token in all_tokens:
        tagged = _tag_chars_with_slots(token)
        for eva_char, slot in tagged:
            triple_key = eva_to_triple.get(eva_char)
            if triple_key is None:
                continue
            triple_slot_counts[triple_key][slot] += 1
            triple_to_glyphs[triple_key].add(eva_char)

    profiles: List[TripleSlotProfile] = []
    for triple_key in sorted(triple_slot_counts.keys()):
        counts = triple_slot_counts[triple_key]
        total = counts['PREFIX'] + counts['ROOT'] + counts['SUFFIX']
        if total == 0:
            continue

        pfx_frac = counts['PREFIX'] / total
        root_frac = counts['ROOT'] / total
        sfx_frac = counts['SUFFIX'] / total

        # Count active slots (fraction >= threshold)
        active_slots = []
        if pfx_frac >= SLOT_THRESHOLD:
            active_slots.append('PREFIX')
        if root_frac >= SLOT_THRESHOLD:
            active_slots.append('ROOT')
        if sfx_frac >= SLOT_THRESHOLD:
            active_slots.append('SUFFIX')

        is_forked = len(active_slots) >= 2

        profiles.append(TripleSlotProfile(
            triple_key=triple_key,
            eva_glyphs=sorted(triple_to_glyphs[triple_key]),
            total_count=total,
            prefix_count=counts['PREFIX'],
            root_count=counts['ROOT'],
            suffix_count=counts['SUFFIX'],
            prefix_frac=round(pfx_frac, 4),
            root_frac=round(root_frac, 4),
            suffix_frac=round(sfx_frac, 4),
            n_active_slots=len(active_slots),
            is_forked=is_forked,
            forked_slots=active_slots,
        ))

    # Ensure dict values are plain dicts (not defaultdict lambdas)
    triple_slot_counts_clean = {k: dict(v) for k, v in triple_slot_counts.items()}
    return profiles, triple_slot_counts_clean


# ---------------------------------------------------------------------------
# Variable forking
# ---------------------------------------------------------------------------

def _build_forked_variables(
    profiles: List[TripleSlotProfile],
    cv_syllables: List[str],
    triple_slot_counts: Dict[str, Dict[str, int]],
) -> List[ForkedVariable]:
    """Build slot-specific variables from triple profiles.

    For each triple:
    - If it is forked (active in >=2 slots), create one variable per active slot
    - If it is not forked (active in only 1 slot), create a single variable
      with the slot label of its dominant slot

    Domain assignment:
    - PREFIX: CV syllables
    - ROOT: CV syllables
    - SUFFIX: Latin inflectional endings
    """
    variables: List[ForkedVariable] = []

    for profile in profiles:
        if profile.is_forked:
            # Create one variable per active slot
            for slot in profile.forked_slots:
                var_key = f"{profile.triple_key}@{slot}"
                counts = triple_slot_counts.get(profile.triple_key, {})
                freq = counts.get(slot, 0)

                if slot == 'SUFFIX':
                    domain = list(LATIN_SUFFIX_ENDINGS)
                else:
                    domain = list(cv_syllables)

                variables.append(ForkedVariable(
                    variable_key=var_key,
                    base_triple=profile.triple_key,
                    slot=slot,
                    eva_glyphs=profile.eva_glyphs,
                    frequency=freq,
                    domain=domain,
                ))
        else:
            # Single variable — use dominant slot
            counts = triple_slot_counts.get(profile.triple_key, {})
            dominant_slot = max(
                ['PREFIX', 'ROOT', 'SUFFIX'],
                key=lambda s: counts.get(s, 0),
            )
            var_key = f"{profile.triple_key}@{dominant_slot}"

            if dominant_slot == 'SUFFIX':
                domain = list(LATIN_SUFFIX_ENDINGS)
            else:
                domain = list(cv_syllables)

            variables.append(ForkedVariable(
                variable_key=var_key,
                base_triple=profile.triple_key,
                slot=dominant_slot,
                eva_glyphs=profile.eva_glyphs,
                frequency=profile.total_count,
                domain=domain,
            ))

    # Sort by frequency descending
    variables.sort(key=lambda v: -v.frequency)
    return variables


# ---------------------------------------------------------------------------
# Phase 33 orthogonality check
# ---------------------------------------------------------------------------

def _check_orthogonality(
    profiles: List[TripleSlotProfile],
    rd: str,
) -> Tuple[int, int, float]:
    """Check how many Phase 33 conflicting triples are now forked.

    A triple is "conflicting" in Phase 33 if different approaches
    (signal, perplexity, suffix) recommended different syllables.

    Returns (n_conflicting, n_now_forked, resolution_rate).
    """
    phase33_path = os.path.join(rd, 'phase33_integrate.json')
    if not os.path.exists(phase33_path):
        return 0, 0, 0.0

    with open(phase33_path) as f:
        phase33_data = json.load(f)

    consensus = phase33_data.get('triple_consensus', [])
    forked_triples = {p.triple_key for p in profiles if p.is_forked}

    n_conflicting = 0
    n_now_forked = 0

    for tc in consensus:
        triple_key = tc.get('triple_key', '')
        phase15_syl = tc.get('phase15_syllable', '')

        # Check if any approach disagrees with Phase 15
        approach_syls = set()
        for key in ('signal_syllable', 'ppl_syllable', 'suffix_syllable',
                     'crib_syllable', 'distrib_syllable'):
            val = tc.get(key, '')
            if val and val != phase15_syl:
                approach_syls.add(val)

        if approach_syls:
            n_conflicting += 1
            if triple_key in forked_triples:
                n_now_forked += 1

    resolution_rate = n_now_forked / n_conflicting if n_conflicting > 0 else 0.0
    return n_conflicting, n_now_forked, resolution_rate


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_slot_variables() -> None:
    """Step 34.5: Build position-conditioned CSP variables."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE 34.5: Slot-Conditioned CSP Variables")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load corpus ──
    print("\n  1. Loading corpus ...")
    corpus = load_corpus(verbose=False)
    all_tokens = corpus.get_tokens()
    n_tokens = len(all_tokens)
    print(f"     {n_tokens} tokens")

    # ── 2. Build triple lookup ──
    print("\n  2. Building triple lookup ...")
    eva_to_triple = build_eva_to_triple_lookup()
    n_base_triples = len(set(eva_to_triple.values()))
    print(f"     {n_base_triples} base triples from {len(eva_to_triple)} EVA glyphs")

    # ── 3. Compute morpheme stats ──
    print("\n  3. Computing morpheme decomposition stats ...")
    n_with_prefix = 0
    n_with_suffix = 0
    n_stem_only = 0

    for token in all_tokens:
        decomp = decompose_token_morphemes(token)
        has_pfx = bool(decomp.prefix)
        has_sfx = bool(decomp.suffix)
        if has_pfx:
            n_with_prefix += 1
        if has_sfx:
            n_with_suffix += 1
        if not has_pfx and not has_sfx:
            n_stem_only += 1

    pct_pfx = round(100.0 * n_with_prefix / n_tokens, 1) if n_tokens > 0 else 0.0
    pct_sfx = round(100.0 * n_with_suffix / n_tokens, 1) if n_tokens > 0 else 0.0
    print(f"     With prefix: {n_with_prefix} ({pct_pfx}%)")
    print(f"     With suffix: {n_with_suffix} ({pct_sfx}%)")
    print(f"     Stem only:   {n_stem_only}")

    # ── 4. Compute slot profiles ──
    print("\n  4. Computing per-triple slot profiles ...")
    profiles, triple_slot_counts = _compute_slot_profiles(all_tokens, eva_to_triple)
    n_forked = sum(1 for p in profiles if p.is_forked)
    print(f"     {len(profiles)} triples profiled")
    print(f"     {n_forked} triples active in >=2 slots (will be forked)")

    for p in profiles:
        tag = '*' if p.is_forked else ' '
        print(f"    {tag} {p.triple_key:40s}  PFX={p.prefix_frac:.2f}  "
              f"ROOT={p.root_frac:.2f}  SFX={p.suffix_frac:.2f}  "
              f"slots={p.n_active_slots}")

    # ── 5. Build forked variables ──
    print("\n  5. Building forked variables ...")
    cv_syllables = build_cv_syllable_table('latin')
    variables = _build_forked_variables(profiles, cv_syllables, triple_slot_counts)

    n_prefix_vars = sum(1 for v in variables if v.slot == 'PREFIX')
    n_root_vars = sum(1 for v in variables if v.slot == 'ROOT')
    n_suffix_vars = sum(1 for v in variables if v.slot == 'SUFFIX')
    n_unforked = sum(1 for p in profiles if not p.is_forked)

    domain_sizes = [len(v.domain) for v in variables]
    mean_domain = sum(domain_sizes) / len(domain_sizes) if domain_sizes else 0.0

    print(f"     Total variables: {len(variables)}")
    print(f"     PREFIX vars: {n_prefix_vars}")
    print(f"     ROOT vars:   {n_root_vars}")
    print(f"     SUFFIX vars: {n_suffix_vars}")
    print(f"     Unforked:    {n_unforked}")
    print(f"     CV syllables: {len(cv_syllables)}")
    print(f"     Suffix endings: {len(LATIN_SUFFIX_ENDINGS)}")
    print(f"     Mean domain size: {mean_domain:.1f}")

    for v in variables[:10]:
        print(f"       {v.variable_key:50s}  freq={v.frequency:5d}  "
              f"|domain|={len(v.domain)}")

    # ── 6. Phase 33 orthogonality check ──
    print("\n  6. Checking Phase 33 orthogonality resolution ...")
    n_conflict, n_resolved, resolution_rate = _check_orthogonality(profiles, rd)
    print(f"     Conflicting triples (Phase 33): {n_conflict}")
    print(f"     Now forked: {n_resolved}")
    print(f"     Resolution rate: {resolution_rate:.1%}")

    # ── 7. Gate and verdict ──
    expansion_ratio = len(variables) / n_base_triples if n_base_triples > 0 else 0.0

    # Gate: we need a meaningful number of forked variables (>= 5)
    # and expansion ratio between 1.2x and 2.5x (not too many, not too few)
    gate_passed = (
        len(variables) >= 30
        and n_forked >= 5
        and expansion_ratio >= 1.2
    )

    if gate_passed:
        verdict = (
            f"FORKED: {len(variables)} slot-conditioned variables from "
            f"{n_base_triples} base triples ({expansion_ratio:.2f}x expansion). "
            f"{n_forked} triples forked across slots. "
            f"Phase 33 resolution: {n_resolved}/{n_conflict} conflicting triples "
            f"({resolution_rate:.0%})."
        )
    else:
        verdict = (
            f"INSUFFICIENT: Only {n_forked} forked triples "
            f"({len(variables)} total vars, {expansion_ratio:.2f}x expansion). "
            f"Slot-conditioning may not provide enough additional degrees of freedom."
        )

    print(f"\n  Gate: {'PASS' if gate_passed else 'FAIL'}")
    print(f"  {verdict}")

    # ── 8. Save ──
    result = SlotVariableResult(
        n_base_triples=n_base_triples,
        n_forked_variables=len(variables),
        n_prefix_vars=n_prefix_vars,
        n_root_vars=n_root_vars,
        n_suffix_vars=n_suffix_vars,
        n_unforked=n_unforked,
        slot_profiles=[_convert(asdict(p)) for p in profiles],
        forked_variable_definitions=[_convert(asdict(v)) for v in variables],
        n_tokens=n_tokens,
        n_with_prefix=n_with_prefix,
        n_with_suffix=n_with_suffix,
        n_stem_only=n_stem_only,
        pct_with_prefix=pct_pfx,
        pct_with_suffix=pct_sfx,
        cv_syllable_count=len(cv_syllables),
        suffix_domain_count=len(LATIN_SUFFIX_ENDINGS),
        mean_domain_size=round(mean_domain, 1),
        phase14_n_variables=25,
        variable_expansion_ratio=round(expansion_ratio, 2),
        n_conflicting_triples_phase33=n_conflict,
        n_conflicting_now_forked=n_resolved,
        orthogonality_resolution_rate=round(resolution_rate, 4),
        gate_passed=gate_passed,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'slot_variables.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)
    print(f"\n  -> {out_path}")
