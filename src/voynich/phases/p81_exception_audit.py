"""
Phase 81: Exception Degrees of Freedom Audit (Reviewer 3.9)
============================================================
Quantifies the degrees of freedom introduced by each exception category
(compounds, allographs, modifiers, wildcards) and documents the independent
criteria that motivate each classification.

The reviewer's concern: each category absorbs a problem the model can't solve,
creating unfalsifiability. This phase shows the categories are constrained by
independent criteria (visual, distributional, positional) rather than post-hoc.

Output: results/p81_exception_audit.json
"""

import json
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Set

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    load_corpus,
    tokenize_eva_chars,
)
from voynich.core.reference import EVA_VISUAL_COMPONENTS


# ---------------------------------------------------------------------------
# JSON helpers
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
    if isinstance(obj, set):
        return sorted(obj)
    if isinstance(obj, (bool, int, float, str, type(None))):
        return obj
    return str(obj)


def _safe_load(path: str) -> Dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _save_json(rd: str, filename: str, data: Any) -> str:
    path = os.path.join(rd, filename)
    with open(path, 'w') as f:
        json.dump(_convert(data), f, indent=2)
    return path


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CompoundAudit:
    """Audit of compound character classifications."""
    compounds: List[str]         # EVA compound chars (e.g. ['qo'])
    n_compounds: int
    dof_added: int               # degrees of freedom consumed
    criterion: str               # what motivated the classification
    q_without_o_rate: float      # fraction of 'q' not followed by 'o'
    tokens_affected: int         # corpus tokens containing a compound
    fraction_affected: float


@dataclass
class AllographGroup:
    """One group of allographic variants."""
    triple: str
    glyphs: List[str]
    n_glyphs: int
    shared_features: Dict[str, str]  # the triple components they share
    differentiating_features: List[str]  # what distinguishes them


@dataclass
class AllographAudit:
    """Audit of allograph classifications."""
    groups: List[AllographGroup]
    n_groups: int                # number of groups with >1 member
    n_merges: int                # total glyphs merged (= sum of group sizes - n_groups)
    dof_added: int               # DOF consumed
    criterion: str
    tokens_affected: int
    fraction_affected: float


@dataclass
class ModifierPositionalProfile:
    """Positional stats for one modifier character."""
    eva_char: str
    n_occurrences: int
    frac_initial: float
    frac_medial: float
    frac_final: float
    frac_singleton: float


@dataclass
class ModifierAudit:
    """Audit of modifier character classifications."""
    modifier_chars: List[str]
    n_modifiers: int
    dof_added: int
    criterion: str
    profiles: List[ModifierPositionalProfile]
    mean_initial_rate_modifiers: float
    mean_initial_rate_syllabic: float
    positional_separation: float  # how different are modifier vs syllabic positions
    tokens_affected: int
    fraction_affected: float


@dataclass
class WildcardAudit:
    """Audit of unresolved (wildcard) triples."""
    unresolved_triples: List[str]
    n_unresolved: int
    n_total_triples: int
    dof_remaining: int
    constraints_from_identifications: int  # from Phase 80
    effective_free_parameters: int


@dataclass
class FalsifiabilityTest:
    """Summary of falsifiability analysis."""
    total_dof: int
    total_constraints: int
    over_determined: bool         # constraints > DOF?
    falsifying_predictions: List[str]


@dataclass
class ExceptionAuditResult:
    phase: str = "81"
    experiment: str = "exception_audit"
    compounds: CompoundAudit = None
    allographs: AllographAudit = None
    modifiers: ModifierAudit = None
    wildcards: WildcardAudit = None
    falsifiability: FalsifiabilityTest = None
    total_dof: int = 0
    total_chars_in_alphabet: int = 0
    total_chars_as_syllabic: int = 0
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Core analysis functions
# ---------------------------------------------------------------------------

def _audit_compounds(corpus, eva_to_triple) -> CompoundAudit:
    """Audit compound character classifications."""
    # qo is the only compound — check q-without-o rate
    all_tokens = []
    for _, page in corpus.pages.items():
        all_tokens.extend(page.all_tokens)

    q_total = 0
    q_without_o = 0
    tokens_with_compound = 0

    for tok in all_tokens:
        chars = tokenize_eva_chars(tok)
        has_compound = False
        for i, ch in enumerate(chars):
            if ch in ('qo', 'qok', 'qot'):
                has_compound = True
            elif ch == 'q':
                q_total += 1
                # Check if next char is 'o'
                if i + 1 < len(chars) and chars[i + 1] == 'o':
                    pass  # q followed by o, but tokenizer should have caught 'qo'
                else:
                    q_without_o += 1
        if has_compound:
            tokens_with_compound += 1
            # Also count q in compound form
            for ch in chars:
                if ch.startswith('qo'):
                    q_total += 1

    q_rate = q_without_o / q_total if q_total > 0 else 0.0

    return CompoundAudit(
        compounds=['qo', 'qok', 'qot'],
        n_compounds=3,
        dof_added=1,  # 1 decision: treat qo as compound
        criterion="Distributional: EVA 'q' appears without 'o' in only "
                  f"{q_rate:.1%} of occurrences. The tokenizer treats qo/qok/qot "
                  "as single characters because they share visual and distributional "
                  "properties distinct from q alone.",
        q_without_o_rate=round(q_rate, 4),
        tokens_affected=tokens_with_compound,
        fraction_affected=round(tokens_with_compound / len(all_tokens), 4)
            if all_tokens else 0.0,
    )


def _audit_allographs(corpus, eva_to_triple) -> AllographAudit:
    """Audit allograph classifications."""
    # Group glyphs by triple
    triple_to_glyphs: Dict[str, List[str]] = {}
    for glyph, comp in EVA_VISUAL_COMPONENTS.items():
        triple = f"{comp['first_stroke']},{comp['last_stroke']},{comp['glyph_class']}"
        if triple not in triple_to_glyphs:
            triple_to_glyphs[triple] = []
        triple_to_glyphs[triple].append(glyph)

    groups = []
    n_merges = 0
    for triple, glyphs in sorted(triple_to_glyphs.items()):
        if len(glyphs) <= 1:
            continue
        comp = EVA_VISUAL_COMPONENTS[glyphs[0]]
        shared = {
            'first_stroke': comp['first_stroke'],
            'last_stroke': comp['last_stroke'],
            'glyph_class': comp['glyph_class'],
        }
        # What differs between the glyphs in this group?
        diffs = []
        for g in glyphs:
            c = EVA_VISUAL_COMPONENTS[g]
            # The triple is the same, so differences are in other aspects
            # (stroke count, presence of serifs, etc.) — captured in glyph name
            diffs.append(g)

        groups.append(AllographGroup(
            triple=triple,
            glyphs=glyphs,
            n_glyphs=len(glyphs),
            shared_features=shared,
            differentiating_features=diffs,
        ))
        n_merges += len(glyphs) - 1

    # Count affected tokens
    all_tokens = []
    for _, page in corpus.pages.items():
        all_tokens.extend(page.all_tokens)

    # An allograph group affects tokens where different members appear
    multi_glyph_triples = {g.triple for g in groups}
    tokens_affected = 0
    for tok in all_tokens:
        chars = tokenize_eva_chars(tok)
        for ch in chars:
            if ch in eva_to_triple and eva_to_triple[ch] in multi_glyph_triples:
                tokens_affected += 1
                break

    return AllographAudit(
        groups=groups,
        n_groups=len(groups),
        n_merges=n_merges,
        dof_added=0,  # Allographs ADD no DOF — they REDUCE the alphabet
        criterion="Visual: glyphs sharing identical stroke-triple decomposition "
                  "(first_stroke, last_stroke, glyph_class) receive the same "
                  "syllable assignment. This is determined by the visual feature "
                  "model, not by decoding results. The triple decomposition was "
                  "defined BEFORE any phonetic assignments were made.",
        tokens_affected=tokens_affected,
        fraction_affected=round(tokens_affected / len(all_tokens), 4)
            if all_tokens else 0.0,
    )


def _audit_modifiers(corpus, eva_to_triple, rd) -> ModifierAudit:
    """Audit modifier character classifications with positional evidence."""
    mod_data = _safe_load(os.path.join(rd, 'modifier_integrate.json'))
    modifier_chars_list = mod_data.get('modifier_chars', [])
    modifier_set = set(modifier_chars_list)

    all_tokens = []
    for _, page in corpus.pages.items():
        all_tokens.extend(page.all_tokens)

    # Compute positional profiles for ALL chars
    char_positions: Dict[str, Dict[str, int]] = {}
    for tok in all_tokens:
        chars = tokenize_eva_chars(tok)
        for idx, ch in enumerate(chars):
            if ch not in char_positions:
                char_positions[ch] = {'initial': 0, 'medial': 0, 'final': 0,
                                       'singleton': 0, 'total': 0}
            char_positions[ch]['total'] += 1
            if len(chars) == 1:
                char_positions[ch]['singleton'] += 1
            elif idx == 0:
                char_positions[ch]['initial'] += 1
            elif idx == len(chars) - 1:
                char_positions[ch]['final'] += 1
            else:
                char_positions[ch]['medial'] += 1

    # Build profiles for modifiers
    profiles = []
    for ch in sorted(modifier_chars_list):
        pos = char_positions.get(ch, {'initial': 0, 'medial': 0, 'final': 0,
                                       'singleton': 0, 'total': 0})
        total = pos['total']
        if total == 0:
            continue
        profiles.append(ModifierPositionalProfile(
            eva_char=ch,
            n_occurrences=total,
            frac_initial=round(pos['initial'] / total, 4),
            frac_medial=round(pos['medial'] / total, 4),
            frac_final=round(pos['final'] / total, 4),
            frac_singleton=round(pos['singleton'] / total, 4),
        ))

    # Compare modifier vs syllabic initial rates
    mod_initial_rates = []
    syl_initial_rates = []
    for ch, pos in char_positions.items():
        total = pos['total']
        if total < 10:
            continue
        rate = pos['initial'] / total
        if ch in modifier_set:
            mod_initial_rates.append(rate)
        elif ch in eva_to_triple:
            syl_initial_rates.append(rate)

    mean_mod_initial = (sum(mod_initial_rates) / len(mod_initial_rates)
                        if mod_initial_rates else 0.0)
    mean_syl_initial = (sum(syl_initial_rates) / len(syl_initial_rates)
                        if syl_initial_rates else 0.0)

    # Count affected tokens
    tokens_with_modifier = 0
    for tok in all_tokens:
        chars = tokenize_eva_chars(tok)
        if any(ch in modifier_set for ch in chars):
            tokens_with_modifier += 1

    return ModifierAudit(
        modifier_chars=sorted(modifier_chars_list),
        n_modifiers=len(modifier_chars_list),
        dof_added=15,  # 15 chars reclassified
        criterion="Positional distribution: modifier characters appear "
                  f"word-initially at {mean_mod_initial:.1%} vs syllabic chars "
                  f"at {mean_syl_initial:.1%}. Phase 16 classified these using "
                  "distributional criteria (never appear as standalone words, "
                  "predominantly non-initial position) BEFORE any CVC coda "
                  "analysis was performed.",
        profiles=profiles,
        mean_initial_rate_modifiers=round(mean_mod_initial, 4),
        mean_initial_rate_syllabic=round(mean_syl_initial, 4),
        positional_separation=round(mean_syl_initial - mean_mod_initial, 4),
        tokens_affected=tokens_with_modifier,
        fraction_affected=round(tokens_with_modifier / len(all_tokens), 4)
            if all_tokens else 0.0,
    )


def _audit_wildcards(rd) -> WildcardAudit:
    """Audit unresolved triple wildcards."""
    tier_data = _safe_load(os.path.join(rd, 'triple_tiers.json'))
    assignment_data = _safe_load(os.path.join(rd, 'combined_refine.json'))
    assignment = assignment_data.get('best_assignment', {})

    confirmed_keys: Set[str] = set()
    if tier_data and 'tiers' in tier_data:
        tiers = tier_data['tiers']
        if isinstance(tiers, dict):
            for entry in tiers.get('CONFIRMED', []):
                confirmed_keys.add(entry.get('triple_key', ''))

    unresolved = [k for k in assignment if k not in confirmed_keys]

    # Check Phase 80 for constraints
    p80_data = _safe_load(os.path.join(rd, 'p80_wildcard_consistency.json'))
    n_constrained = p80_data.get('n_triples_observed', 0)

    return WildcardAudit(
        unresolved_triples=sorted(unresolved),
        n_unresolved=len(unresolved),
        n_total_triples=len(assignment),
        dof_remaining=len(unresolved),
        constraints_from_identifications=n_constrained,
        effective_free_parameters=max(0, len(unresolved) - n_constrained),
    )


def _falsifiability_analysis(
    compounds: CompoundAudit,
    allographs: AllographAudit,
    modifiers: ModifierAudit,
    wildcards: WildcardAudit,
    n_identifications: int,
) -> FalsifiabilityTest:
    """Assess whether the model is falsifiable."""
    total_dof = (compounds.dof_added + allographs.dof_added
                 + modifiers.dof_added + wildcards.dof_remaining)

    # Constraints: each word identification constrains the assignment table
    # Positional constraints: each confirmed triple has a fixed value
    # The 301 fully-decoded identifications each independently confirm
    # that the confirmed triple assignments produce dictionary words
    n_confirmed_triples = wildcards.n_total_triples - wildcards.n_unresolved
    total_constraints = n_identifications + n_confirmed_triples

    predictions = [
        "If EVA 'ch' (assigned to 'co') appeared word-finally at >20% rate, "
        "this would contradict Italian syllable distributions (co is word-initial).",
        "If the 15 modifier characters appeared word-initially at rates comparable "
        "to syllabic characters, the modifier classification would be falsified.",
        "If randomly shuffling the assignment table produced equally many "
        "or more signal words, the table would be indistinguishable from noise "
        "(tested: p=0.001, Table 3 in paper).",
        "If the 22 word-level identifications produced different Latin words "
        "under a different assignment table with equal frequency, the identifications "
        "would be table-independent (tested: p=0.009, Section 6.3 in paper).",
        "If the coda mapping {hook->n, sigmoid->s, vertical->t} were shuffled, "
        "grammatical distribution should diverge from pharmaceutical Latin "
        "(tested: p<0.002, Section 7 in paper).",
    ]

    return FalsifiabilityTest(
        total_dof=total_dof,
        total_constraints=total_constraints,
        over_determined=(total_constraints > total_dof),
        falsifying_predictions=predictions,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_exception_audit():
    """Phase 81: Quantify degrees of freedom for each exception category."""
    t0 = time.time()
    rd = _results_dir()
    print("Phase 81: Exception Degrees of Freedom Audit")
    print("=" * 60)

    corpus = load_corpus(verbose=False)
    eva_to_triple = build_eva_to_triple_lookup()

    # Run each audit
    compounds = _audit_compounds(corpus, eva_to_triple)
    allographs = _audit_allographs(corpus, eva_to_triple)
    modifiers = _audit_modifiers(corpus, eva_to_triple, rd)
    wildcards = _audit_wildcards(rd)

    # Get identification count from p75_t1
    t1_data = _safe_load(os.path.join(rd, 'p75_t1.json'))
    n_ids = len(t1_data.get('identifications', []))

    falsifiability = _falsifiability_analysis(
        compounds, allographs, modifiers, wildcards, n_ids)

    # Print summary
    print("\n--- Compounds ---")
    print(f"  Characters: {compounds.compounds}")
    print(f"  DOF: {compounds.dof_added}")
    print(f"  q-without-o rate: {compounds.q_without_o_rate:.1%}")
    print(f"  Tokens affected: {compounds.tokens_affected} ({compounds.fraction_affected:.1%})")
    print(f"  Criterion: {compounds.criterion[:100]}...")

    print("\n--- Allographs ---")
    print(f"  Groups: {allographs.n_groups} (with {allographs.n_merges} glyphs merged)")
    print(f"  DOF: {allographs.dof_added} (allographs REDUCE alphabet, not expand it)")
    for g in allographs.groups:
        print(f"    {g.triple}: {g.glyphs}")
    print(f"  Tokens affected: {allographs.tokens_affected} ({allographs.fraction_affected:.1%})")
    print(f"  Criterion: {allographs.criterion[:100]}...")

    print("\n--- Modifiers ---")
    print(f"  Characters: {modifiers.modifier_chars}")
    print(f"  DOF: {modifiers.dof_added}")
    print(f"  Mean word-initial rate: modifiers={modifiers.mean_initial_rate_modifiers:.1%} "
          f"vs syllabic={modifiers.mean_initial_rate_syllabic:.1%}")
    print(f"  Positional separation: {modifiers.positional_separation:.1%}")
    print(f"  Tokens affected: {modifiers.tokens_affected} ({modifiers.fraction_affected:.1%})")

    print("\n--- Wildcards (unresolved triples) ---")
    print(f"  Unresolved: {wildcards.n_unresolved}/{wildcards.n_total_triples}")
    print(f"  Constrained by identifications: {wildcards.constraints_from_identifications}")
    print(f"  Effective free parameters: {wildcards.effective_free_parameters}")
    print(f"  Triples: {wildcards.unresolved_triples}")

    print("\n--- Falsifiability ---")
    print(f"  Total DOF: {falsifiability.total_dof}")
    print(f"  Total constraints: {falsifiability.total_constraints}")
    print(f"  Over-determined: {falsifiability.over_determined}")
    print(f"  Falsifying predictions:")
    for p in falsifiability.falsifying_predictions:
        print(f"    - {p[:100]}...")

    # Count total chars
    total_chars = len(EVA_VISUAL_COMPONENTS)
    syllabic_chars = len(eva_to_triple)

    result = ExceptionAuditResult(
        compounds=compounds,
        allographs=allographs,
        modifiers=modifiers,
        wildcards=wildcards,
        falsifiability=falsifiability,
        total_dof=falsifiability.total_dof,
        total_chars_in_alphabet=total_chars,
        total_chars_as_syllabic=syllabic_chars,
        runtime_seconds=round(time.time() - t0, 2),
    )

    path = _save_json(rd, 'p81_exception_audit.json', result)
    print(f"\n  Saved -> {path}")
    print(f"  Runtime: {result.runtime_seconds:.1f}s")
    return result
