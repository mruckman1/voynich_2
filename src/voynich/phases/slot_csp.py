"""
Phase 34.6 – Slot-Conditioned CSP Solve and Decode
====================================================
Solves the position-conditioned CSP from Step 34.5 and decodes the full
corpus.  Each token is decomposed into prefix/root/suffix, each EVA char
is mapped to its slot-specific variable, and the concatenated syllables
form the decoded word.

Algorithm:
  1. Load slot variables from step 34.5
  2. For each slot group (PREFIX, ROOT, SUFFIX), run coordinate descent
     to optimise the assignment within that group
  3. For each token: decompose -> decode prefix/root/suffix through
     respective assignments -> assemble decoded word
  4. Compare dict-hit to Phase 16 baseline (43.6%)
  5. Test orthogonality resolution: does slot-conditioning satisfy
     Phase 33's conflicting recommendations?

Dependency chain:
    slot_variables.json       (Step 34.5)
    combined_refine.json      (Phase 15 assignment — baseline)
    modifier_integrate.json   (Phase 16 modifiers)
        -> slot_csp.json  (this step)
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
    tokenize_eva_chars,
)
from voynich.core.reference import (
    build_cv_syllable_table,
    build_expanded_word_set,
    load_reference_corpus,
)
from voynich.phases.morpheme_grid import (
    decompose_token_morphemes,
    KNOWN_PREFIXES,
    KNOWN_SUFFIXES,
)
from voynich.phases.null_corpus import _reconstruct_modifier_rules
from voynich.phases.signal_isolation import _decode_corpus_r3
from voynich.phases.slot_variables import LATIN_SUFFIX_ENDINGS


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
class SlotAssignment:
    """A mapping from slot-variable keys to syllables."""
    prefix_assignment: Dict[str, str]   # triple_key -> syllable for PREFIX vars
    root_assignment: Dict[str, str]     # triple_key -> syllable for ROOT vars
    suffix_assignment: Dict[str, str]   # triple_key -> syllable for SUFFIX vars
    combined_assignment: Dict[str, str] # variable_key -> syllable (all)


@dataclass
class SlotCSPResult:
    """Full Step 34.6 output."""
    # Assignment
    n_variables: int
    n_prefix_vars: int
    n_root_vars: int
    n_suffix_vars: int
    slot_assignment: Dict[str, str]
    prefix_assignment: Dict[str, str]
    root_assignment: Dict[str, str]
    suffix_assignment: Dict[str, str]
    # Decode results
    n_tokens: int
    dict_hit_rate: float
    n_dict_hits: int
    # Baseline comparison
    phase16_baseline_dict_hit: float
    delta_vs_phase16: float
    phase15_assignment_dict_hit: float
    delta_vs_phase15: float
    # Per-slot breakdown
    prefix_decode_hit_rate: float
    root_decode_hit_rate: float
    suffix_decode_hit_rate: float
    # Phase 33 orthogonality
    n_conflicting_triples: int
    n_orthogonal_resolved: int
    # Decoded samples
    decoded_sample: List[Dict]
    # Convergence
    n_iterations: int
    convergence_trace: List[float]
    gate_passed: bool
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Slot-aware decoding
# ---------------------------------------------------------------------------

def _decode_token_slotted(
    token: str,
    prefix_map: Dict[str, str],
    root_map: Dict[str, str],
    suffix_map: Dict[str, str],
    eva_to_triple: Dict[str, str],
) -> str:
    """Decode a single token through slot-conditioned assignments.

    1. Decompose the token into prefix/root/suffix EVA chars
    2. Map each EVA char to its triple_key
    3. Look up the syllable from the slot-appropriate assignment
    4. Concatenate all syllables
    """
    decomp = decompose_token_morphemes(token)
    parts: List[str] = []

    # Decode prefix chars
    for ch in decomp.prefix_glyphs:
        triple_key = eva_to_triple.get(ch)
        if triple_key and triple_key in prefix_map:
            parts.append(prefix_map[triple_key])
        elif triple_key and triple_key in root_map:
            # Fallback: use root assignment if no prefix-specific one
            parts.append(root_map[triple_key])

    # Decode root/stem chars
    for ch in decomp.stem_glyphs:
        triple_key = eva_to_triple.get(ch)
        if triple_key and triple_key in root_map:
            parts.append(root_map[triple_key])

    # Decode suffix chars
    for ch in decomp.suffix_glyphs:
        triple_key = eva_to_triple.get(ch)
        if triple_key and triple_key in suffix_map:
            parts.append(suffix_map[triple_key])
        elif triple_key and triple_key in root_map:
            # Fallback: use root assignment if no suffix-specific one
            parts.append(root_map[triple_key])

    return ''.join(parts).lower()


def _decode_corpus_slotted(
    tokens: List[str],
    prefix_map: Dict[str, str],
    root_map: Dict[str, str],
    suffix_map: Dict[str, str],
    eva_to_triple: Dict[str, str],
) -> List[str]:
    """Decode all tokens through slot-conditioned assignments."""
    return [
        _decode_token_slotted(t, prefix_map, root_map, suffix_map, eva_to_triple)
        for t in tokens
    ]


# ---------------------------------------------------------------------------
# Coordinate descent optimisation
# ---------------------------------------------------------------------------

def _compute_dict_hit(
    decoded: List[str],
    ref_word_set: set,
) -> float:
    """Compute dict-hit rate for a list of decoded words."""
    if not decoded:
        return 0.0
    hits = sum(1 for w in decoded if w in ref_word_set)
    return hits / len(decoded)


def _seed_from_phase15(
    slot_vars: List[Dict],
    phase15_assignment: Dict[str, str],
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    """Seed slot assignments from the Phase 15 global assignment.

    For each forked variable, the Phase 15 syllable for the base triple
    is used as the initial assignment.
    """
    prefix_map: Dict[str, str] = {}
    root_map: Dict[str, str] = {}
    suffix_map: Dict[str, str] = {}

    for var_def in slot_vars:
        base_triple = var_def['base_triple']
        slot = var_def['slot']
        syl = phase15_assignment.get(base_triple, '')

        if slot == 'PREFIX':
            prefix_map[base_triple] = syl
        elif slot == 'ROOT':
            root_map[base_triple] = syl
        elif slot == 'SUFFIX':
            # Phase 15 assigns CV syllables; for suffix vars, find the
            # closest inflectional ending or keep as-is
            if syl in LATIN_SUFFIX_ENDINGS:
                suffix_map[base_triple] = syl
            else:
                # Default to the most common Latin ending
                suffix_map[base_triple] = 'a'

    return prefix_map, root_map, suffix_map


def _coordinate_descent(
    all_tokens: List[str],
    slot_vars: List[Dict],
    prefix_map: Dict[str, str],
    root_map: Dict[str, str],
    suffix_map: Dict[str, str],
    eva_to_triple: Dict[str, str],
    ref_word_set: set,
    max_iterations: int = 5,
    seed: int = 42,
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str], List[float], int]:
    """Coordinate descent over slot assignments.

    Uses a subsample of tokens for fast candidate evaluation, then
    validates on the full corpus.

    Returns (prefix_map, root_map, suffix_map, convergence_trace, n_iters).
    """
    rng = random.Random(seed)

    # Build per-slot variable lists with domains
    prefix_vars = [v for v in slot_vars if v['slot'] == 'PREFIX']
    root_vars = [v for v in slot_vars if v['slot'] == 'ROOT']
    suffix_vars = [v for v in slot_vars if v['slot'] == 'SUFFIX']

    # Use a subsample for fast evaluation (2000 tokens)
    subsample_size = min(2000, len(all_tokens))
    subsample_indices = rng.sample(range(len(all_tokens)), subsample_size)
    subsample_tokens = [all_tokens[i] for i in subsample_indices]

    # Initial dict hit (on full corpus)
    decoded = _decode_corpus_slotted(all_tokens, prefix_map, root_map, suffix_map, eva_to_triple)
    best_hit = _compute_dict_hit(decoded, ref_word_set)
    trace = [best_hit]

    print(f"       Initial dict_hit: {best_hit:.4f} (using {subsample_size}-token subsample for search)")

    for iteration in range(max_iterations):
        improved = False

        # Shuffle variable order each iteration for better exploration
        all_vars = list(prefix_vars) + list(root_vars) + list(suffix_vars)
        rng.shuffle(all_vars)

        for var_def in all_vars:
            base_triple = var_def['base_triple']
            slot = var_def['slot']
            domain = var_def['domain']

            if not domain:
                continue

            # Pick the right map
            if slot == 'PREFIX':
                current_map = prefix_map
            elif slot == 'ROOT':
                current_map = root_map
            else:
                current_map = suffix_map

            current_syl = current_map.get(base_triple, '')
            # Evaluate current on subsample
            sub_decoded = _decode_corpus_slotted(
                subsample_tokens, prefix_map, root_map, suffix_map, eva_to_triple,
            )
            best_local_hit = _compute_dict_hit(sub_decoded, ref_word_set)
            best_syl = current_syl

            # Limit domain to top-15 by frequency in reference
            candidates = domain[:15] if len(domain) > 15 else domain

            for candidate_syl in candidates:
                if candidate_syl == current_syl:
                    continue
                # Try this assignment on subsample only
                current_map[base_triple] = candidate_syl
                sub_decoded = _decode_corpus_slotted(
                    subsample_tokens, prefix_map, root_map, suffix_map, eva_to_triple,
                )
                hit = _compute_dict_hit(sub_decoded, ref_word_set)
                if hit > best_local_hit:
                    best_local_hit = hit
                    best_syl = candidate_syl
                    improved = True

            # Set best value
            current_map[base_triple] = best_syl

        # Evaluate on full corpus after each iteration
        decoded = _decode_corpus_slotted(all_tokens, prefix_map, root_map, suffix_map, eva_to_triple)
        best_hit = _compute_dict_hit(decoded, ref_word_set)
        trace.append(best_hit)
        print(f"       Iteration {iteration + 1}: dict_hit = {best_hit:.4f}")

        if not improved:
            print(f"       Converged after {iteration + 1} iterations")
            break

    return prefix_map, root_map, suffix_map, trace, len(trace) - 1


# ---------------------------------------------------------------------------
# Phase 33 orthogonality resolution
# ---------------------------------------------------------------------------

def _check_orthogonality_resolution(
    prefix_map: Dict[str, str],
    root_map: Dict[str, str],
    suffix_map: Dict[str, str],
    phase15_assignment: Dict[str, str],
    rd: str,
) -> Tuple[int, int]:
    """Check whether slot-conditioning resolves Phase 33 conflicts.

    A conflict is "resolved" if the slot assignments differ from each other,
    meaning each slot can satisfy its own approach's recommendation without
    contradicting the others.

    Returns (n_conflicting, n_resolved).
    """
    phase33_path = os.path.join(rd, 'phase33_integrate.json')
    if not os.path.exists(phase33_path):
        return 0, 0

    with open(phase33_path) as f:
        phase33_data = json.load(f)

    consensus = phase33_data.get('triple_consensus', [])
    n_conflicting = 0
    n_resolved = 0

    for tc in consensus:
        triple_key = tc.get('triple_key', '')
        phase15_syl = tc.get('phase15_syllable', '')

        # Check if any approach disagrees
        approach_syls = set()
        for key in ('signal_syllable', 'ppl_syllable', 'suffix_syllable'):
            val = tc.get(key, '')
            if val and val != phase15_syl:
                approach_syls.add(val)

        if not approach_syls:
            continue

        n_conflicting += 1

        # Check if slot-conditioning resolved it: do PREFIX, ROOT, SUFFIX
        # have different assignments for this triple?
        slot_syls = set()
        if triple_key in prefix_map:
            slot_syls.add(('PREFIX', prefix_map[triple_key]))
        if triple_key in root_map:
            slot_syls.add(('ROOT', root_map[triple_key]))
        if triple_key in suffix_map:
            slot_syls.add(('SUFFIX', suffix_map[triple_key]))

        # Resolved if at least 2 slots have different syllables
        unique_syls = {s for _, s in slot_syls if s}
        if len(unique_syls) >= 2:
            n_resolved += 1

    return n_conflicting, n_resolved


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_slot_csp() -> None:
    """Step 34.6: Slot-conditioned CSP solve and decode."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE 34.6: Slot-Conditioned CSP Solve and Decode")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load slot variables ──
    print("\n  1. Loading slot variables ...")
    sv_path = os.path.join(rd, 'slot_variables.json')
    if not os.path.exists(sv_path):
        print("  [SKIP] slot_variables.json not found -- run slot-vars first")
        return
    with open(sv_path) as f:
        sv_data = json.load(f)

    slot_vars = sv_data.get('forked_variable_definitions', [])
    if not slot_vars:
        print("  [SKIP] No forked variable definitions found")
        return

    n_prefix = sum(1 for v in slot_vars if v['slot'] == 'PREFIX')
    n_root = sum(1 for v in slot_vars if v['slot'] == 'ROOT')
    n_suffix = sum(1 for v in slot_vars if v['slot'] == 'SUFFIX')
    print(f"     {len(slot_vars)} slot variables: {n_prefix} PREFIX, "
          f"{n_root} ROOT, {n_suffix} SUFFIX")

    # ── 2. Load Phase 15 baseline assignment ──
    print("\n  2. Loading Phase 15 baseline assignment ...")
    refine_path = os.path.join(rd, 'combined_refine.json')
    if not os.path.exists(refine_path):
        print("  [SKIP] combined_refine.json not found")
        return
    with open(refine_path) as f:
        refine_data = json.load(f)
    phase15_assignment = refine_data.get('best_assignment', {})
    print(f"     {len(phase15_assignment)} triple assignments loaded")

    # ── 3. Load modifiers ──
    print("\n  3. Loading modifier rules ...")
    mod_path = os.path.join(rd, 'modifier_integrate.json')
    if not os.path.exists(mod_path):
        print("  [SKIP] modifier_integrate.json not found")
        return
    with open(mod_path) as f:
        mod_data = json.load(f)
    modifier_chars, modifier_rules = _reconstruct_modifier_rules(mod_data)
    print(f"     {len(modifier_chars)} modifier chars")

    # ── 4. Build reference word set ──
    print("\n  4. Building reference word set ...")
    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    base_words = set(
        w.lower() for w in ref_corpus.get_combined_tokens('latin')
        if len(w) >= 2
    )
    expanded, _ = build_expanded_word_set(base_words)
    ref_word_set = base_words | expanded
    print(f"     {len(ref_word_set)} reference words")

    # ── 5. Load corpus ──
    print("\n  5. Loading corpus ...")
    corpus = load_corpus(verbose=False)
    all_tokens = corpus.get_tokens()
    n_tokens = len(all_tokens)
    print(f"     {n_tokens} tokens")

    # ── 6. Compute Phase 15 baseline (R3 decode) ──
    print("\n  6. Computing Phase 15 baseline (R3 decode) ...")
    eva_to_triple = build_eva_to_triple_lookup()
    phase15_decoded = _decode_corpus_r3(
        all_tokens, phase15_assignment, eva_to_triple,
        modifier_chars, modifier_rules, ref_word_set,
    )
    phase15_hit_rate = _compute_dict_hit(phase15_decoded, ref_word_set)
    print(f"     Phase 15/16 R3 dict_hit: {phase15_hit_rate:.4f}")

    # ── 7. Seed slot assignments from Phase 15 ──
    print("\n  7. Seeding slot assignments from Phase 15 ...")
    prefix_map, root_map, suffix_map = _seed_from_phase15(
        slot_vars, phase15_assignment,
    )
    print(f"     PREFIX map: {len(prefix_map)} entries")
    print(f"     ROOT map:   {len(root_map)} entries")
    print(f"     SUFFIX map: {len(suffix_map)} entries")

    # ── 8. Coordinate descent optimisation ──
    print("\n  8. Running coordinate descent ...")
    prefix_map, root_map, suffix_map, trace, n_iters = _coordinate_descent(
        all_tokens, slot_vars,
        prefix_map, root_map, suffix_map,
        eva_to_triple, ref_word_set,
        max_iterations=20, seed=42,
    )

    # ── 9. Final decode ──
    print("\n  9. Final decode ...")
    decoded = _decode_corpus_slotted(
        all_tokens, prefix_map, root_map, suffix_map, eva_to_triple,
    )
    dict_hit_rate = _compute_dict_hit(decoded, ref_word_set)
    n_dict_hits = sum(1 for w in decoded if w in ref_word_set)
    delta_vs_phase16 = dict_hit_rate - 0.436
    delta_vs_phase15 = dict_hit_rate - phase15_hit_rate
    print(f"     Slot CSP dict_hit: {dict_hit_rate:.4f} ({n_dict_hits}/{n_tokens})")
    print(f"     vs Phase 16 baseline (43.6%): {delta_vs_phase16:+.4f}")
    print(f"     vs Phase 15 R3:               {delta_vs_phase15:+.4f}")

    # ── 10. Per-slot decode breakdown ──
    print("\n  10. Per-slot decode breakdown ...")
    # Measure how tokens with prefix/root/suffix individually contribute
    pfx_hits = 0
    pfx_total = 0
    root_hits = 0
    root_total = 0
    sfx_hits = 0
    sfx_total = 0

    for token, dec_word in zip(all_tokens, decoded):
        decomp = decompose_token_morphemes(token)
        is_hit = dec_word in ref_word_set
        if decomp.prefix:
            pfx_total += 1
            if is_hit:
                pfx_hits += 1
        if decomp.stem:
            root_total += 1
            if is_hit:
                root_hits += 1
        if decomp.suffix:
            sfx_total += 1
            if is_hit:
                sfx_hits += 1

    pfx_hit_rate = pfx_hits / pfx_total if pfx_total > 0 else 0.0
    root_hit_rate = root_hits / root_total if root_total > 0 else 0.0
    sfx_hit_rate = sfx_hits / sfx_total if sfx_total > 0 else 0.0

    print(f"     PREFIX tokens: {pfx_hits}/{pfx_total} ({pfx_hit_rate:.1%})")
    print(f"     ROOT tokens:   {root_hits}/{root_total} ({root_hit_rate:.1%})")
    print(f"     SUFFIX tokens: {sfx_hits}/{sfx_total} ({sfx_hit_rate:.1%})")

    # ── 11. Phase 33 orthogonality ──
    print("\n  11. Phase 33 orthogonality resolution ...")
    n_conflict, n_resolved = _check_orthogonality_resolution(
        prefix_map, root_map, suffix_map, phase15_assignment, rd,
    )
    print(f"     Conflicting triples: {n_conflict}")
    print(f"     Resolved by forking: {n_resolved}")

    # ── 12. Decoded samples ──
    print("\n  12. Decoded samples ...")
    decoded_sample: List[Dict] = []
    sample_indices = list(range(min(50, n_tokens)))
    for idx in sample_indices:
        token = all_tokens[idx]
        dec = decoded[idx]
        is_hit = dec in ref_word_set
        decomp = decompose_token_morphemes(token)
        decoded_sample.append({
            'token': token,
            'decoded': dec,
            'dict_hit': is_hit,
            'prefix': decomp.prefix,
            'stem': decomp.stem,
            'suffix': decomp.suffix,
        })
        if idx < 10:
            tag = '+' if is_hit else ' '
            print(f"    {tag} {token:20s} -> {dec:15s} "
                  f"(pfx={decomp.prefix or '-'} "
                  f"root={decomp.stem or '-'} "
                  f"sfx={decomp.suffix or '-'})")

    # ── 13. Gate and verdict ──
    gate_passed = dict_hit_rate > phase15_hit_rate and dict_hit_rate > 0.436

    if dict_hit_rate > 0.50:
        verdict = (
            f"IMPROVEMENT: {dict_hit_rate:.1%} dict_hit ({delta_vs_phase16:+.1%} vs "
            f"Phase 16). Slot-conditioning provides meaningful lift. "
            f"{n_resolved}/{n_conflict} Phase 33 conflicts resolved."
        )
    elif dict_hit_rate > phase15_hit_rate:
        verdict = (
            f"MARGINAL: {dict_hit_rate:.1%} dict_hit ({delta_vs_phase15:+.1%} vs "
            f"Phase 15 R3). Small improvement from slot-conditioning. "
            f"{n_resolved}/{n_conflict} Phase 33 conflicts resolved."
        )
    else:
        verdict = (
            f"NO_IMPROVEMENT: {dict_hit_rate:.1%} dict_hit ({delta_vs_phase15:+.1%} vs "
            f"Phase 15 R3). Slot-conditioning does not improve decoding. "
            f"Position-conditioned encoding hypothesis not supported."
        )

    print(f"\n  Gate: {'PASS' if gate_passed else 'FAIL'}")
    print(f"  {verdict}")

    # ── 14. Build combined assignment dict ──
    combined_assignment: Dict[str, str] = {}
    for triple, syl in prefix_map.items():
        combined_assignment[f"{triple}@PREFIX"] = syl
    for triple, syl in root_map.items():
        combined_assignment[f"{triple}@ROOT"] = syl
    for triple, syl in suffix_map.items():
        combined_assignment[f"{triple}@SUFFIX"] = syl

    # ── 15. Save ──
    result = SlotCSPResult(
        n_variables=len(slot_vars),
        n_prefix_vars=n_prefix,
        n_root_vars=n_root,
        n_suffix_vars=n_suffix,
        slot_assignment=combined_assignment,
        prefix_assignment=prefix_map,
        root_assignment=root_map,
        suffix_assignment=suffix_map,
        n_tokens=n_tokens,
        dict_hit_rate=round(dict_hit_rate, 6),
        n_dict_hits=n_dict_hits,
        phase16_baseline_dict_hit=0.436,
        delta_vs_phase16=round(delta_vs_phase16, 6),
        phase15_assignment_dict_hit=round(phase15_hit_rate, 6),
        delta_vs_phase15=round(delta_vs_phase15, 6),
        prefix_decode_hit_rate=round(pfx_hit_rate, 4),
        root_decode_hit_rate=round(root_hit_rate, 4),
        suffix_decode_hit_rate=round(sfx_hit_rate, 4),
        n_conflicting_triples=n_conflict,
        n_orthogonal_resolved=n_resolved,
        decoded_sample=decoded_sample,
        n_iterations=n_iters,
        convergence_trace=[round(t, 6) for t in trace],
        gate_passed=gate_passed,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'slot_csp.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)
    print(f"\n  -> {out_path}")
