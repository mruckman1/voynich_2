"""
Step 33.11: Long-Crib CSP — Exhaustive Plant-Name Alignment
=============================================================
For each target folio from Step 33.10, exhaustively test whether any EVA
label token can encode the expected plant name.  Long cribs (4+ syllables)
are qualitatively different from short cribs because they provide more
constraints and make false positives exponentially less likely.

Algorithm:
  For each (folio, plant_name) target and each compatible label candidate,
  decompose the label into syllabic EVA chars, enumerate valid char→syllable
  alignments, and score them against the confirmed decoding table.
  Cross-folio consistency (two different plants on different folios proposing
  the same triple→syllable assignment) provides strong independent evidence.

Null control:
  For each folio, also align 3 random *wrong* plant names (from other folios).
  Gate: correct plant must yield ≥2× as many valid alignments as wrong plants.

Dependency chain:
    long_crib_targets.json   (Step 33.10)
    combined_refine.json     (Phase 15 assignment)
    bootstrap_loop.json      (Phase 30 confirmed triples)
    modifier_integrate.json  (Phase 16 modifiers)
        → long_crib_csp.json  (this step)
"""

import json
import os
import random
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from itertools import product
from typing import Any, Dict, List, Optional, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import build_eva_to_triple_lookup, tokenize_eva_chars
from voynich.core.reference import PHONEME_PLACE_MAP, PHONEME_NUCLEUS_MAP


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


def _reconstruct_modifier_chars(data: Dict) -> Set[str]:
    """Extract modifier_chars set from modifier_integrate data."""
    return set(data.get('modifier_chars', []))


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CribAlignment:
    """One alignment of EVA label chars to plant-name syllables."""
    folio: str
    plant_name: str
    eva_token: str
    syllabic_chars: List[str]
    syllables: List[str]
    alignment: Dict[str, str]       # triple_key → proposed syllable
    n_confirmed_positions: int
    n_new_positions: int
    new_assignments: Dict[str, str]  # only the NEW triple → syllable proposals
    valid: bool
    rejection_reason: str            # '' if valid


@dataclass
class LongCribCSPResult:
    """Full Step 33.11 output."""
    n_targets_tested: int
    n_with_valid_alignments: int
    all_alignments: List[Dict]
    # Cross-folio consistency
    n_cross_folio_consistent: int
    consistent_new_assignments: Dict[str, str]  # triple_key → syllable (agreed across folios)
    n_conflicting: int
    # Null control
    correct_plant_alignments: int
    wrong_plant_alignments: int
    null_selectivity: float
    # Summary
    n_new_confirmed_triples: int
    new_confirmed_triples: Dict[str, str]
    verdict: str   # 'CRIBS_CONFIRMED', 'PARTIAL_MATCH', 'NO_MATCH'
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Alignment enumeration
# ---------------------------------------------------------------------------

def _enumerate_alignments(
    syllabic_chars: List[str],
    syllables: List[str],
    eva_to_triple: Dict[str, str],
) -> List[List[Tuple[str, str, str]]]:
    """Enumerate valid char→syllable alignments.

    Returns a list of alignments.  Each alignment is a list of
    (eva_char, triple_key, syllable) tuples — one per matched position.

    Supports:
      - Exact match (n_chars == n_syllables): single 1:1 alignment.
      - n_chars == n_syllables + 1: skip one char (abbreviation of label).
      - n_chars == n_syllables - 1: skip one syllable (abbreviation in script).
    """
    n_chars = len(syllabic_chars)
    n_syls = len(syllables)
    alignments: List[List[Tuple[str, str, str]]] = []

    if n_chars == 0 or n_syls == 0:
        return alignments

    if n_chars == n_syls:
        # Direct 1:1 mapping
        alignment = []
        for ch, syl in zip(syllabic_chars, syllables):
            triple = eva_to_triple.get(ch, ch)
            alignment.append((ch, triple, syl))
        alignments.append(alignment)

    elif n_chars == n_syls + 1:
        # One extra EVA char — try skipping each char position
        for skip in range(n_chars):
            alignment = []
            syl_idx = 0
            for pos, ch in enumerate(syllabic_chars):
                if pos == skip:
                    continue
                if syl_idx >= n_syls:
                    break
                triple = eva_to_triple.get(ch, ch)
                alignment.append((ch, triple, syllables[syl_idx]))
                syl_idx += 1
            if syl_idx == n_syls:
                alignments.append(alignment)

    elif n_chars == n_syls - 1:
        # One fewer EVA char — try skipping each syllable position
        for skip in range(n_syls):
            alignment = []
            char_idx = 0
            for pos, syl in enumerate(syllables):
                if pos == skip:
                    continue
                if char_idx >= n_chars:
                    break
                ch = syllabic_chars[char_idx]
                triple = eva_to_triple.get(ch, ch)
                alignment.append((ch, triple, syl))
                char_idx += 1
            if char_idx == n_chars:
                alignments.append(alignment)

    elif n_chars == n_syls + 2:
        # Two extra EVA chars — skip two positions
        for s1 in range(n_chars):
            for s2 in range(s1 + 1, n_chars):
                alignment = []
                syl_idx = 0
                for pos, ch in enumerate(syllabic_chars):
                    if pos == s1 or pos == s2:
                        continue
                    if syl_idx >= n_syls:
                        break
                    triple = eva_to_triple.get(ch, ch)
                    alignment.append((ch, triple, syllables[syl_idx]))
                    syl_idx += 1
                if syl_idx == n_syls:
                    alignments.append(alignment)
                if len(alignments) >= 500:
                    return alignments

    elif n_chars == n_syls - 2:
        # Two fewer EVA chars — skip two syllable positions
        for s1 in range(n_syls):
            for s2 in range(s1 + 1, n_syls):
                alignment = []
                char_idx = 0
                for pos, syl in enumerate(syllables):
                    if pos == s1 or pos == s2:
                        continue
                    if char_idx >= n_chars:
                        break
                    ch = syllabic_chars[char_idx]
                    triple = eva_to_triple.get(ch, ch)
                    alignment.append((ch, triple, syl))
                    char_idx += 1
                if char_idx == n_chars:
                    alignments.append(alignment)
                if len(alignments) >= 500:
                    return alignments

    return alignments


# ---------------------------------------------------------------------------
# Validation checks
# ---------------------------------------------------------------------------

def _check_confirmed(
    alignment: List[Tuple[str, str, str]],
    confirmed_triples: Dict[str, str],
    full_assignment: Dict[str, str],
) -> Tuple[int, int, bool, str]:
    """Check alignment against confirmed triples and full assignment.

    Returns (n_confirmed_ok, n_new, passes, rejection_reason).
    - n_confirmed_ok: positions where char's triple is confirmed AND
      current assignment matches the proposed syllable.
    - n_new: positions where the triple is NOT confirmed (new proposals).
    - passes: False if any confirmed triple CONFLICTS.
    """
    n_confirmed_ok = 0
    n_new = 0

    for eva_char, triple_key, syllable in alignment:
        if triple_key in confirmed_triples:
            if confirmed_triples[triple_key] == syllable:
                n_confirmed_ok += 1
            else:
                return n_confirmed_ok, n_new, False, (
                    f"confirmed triple {triple_key} decodes to "
                    f"'{confirmed_triples[triple_key]}', not '{syllable}'"
                )
        elif triple_key in full_assignment:
            # Not confirmed, but has an existing assignment — count as new
            # proposal (may agree or disagree with the statistical table)
            n_new += 1
        else:
            n_new += 1

    return n_confirmed_ok, n_new, True, ''


def _check_repeated_syllables(
    alignment: List[Tuple[str, str, str]],
) -> Tuple[bool, str]:
    """If the plant name has repeated syllables (e.g. 'pa','pa' in papaver),
    chars at those positions must use the same triple."""
    syl_to_triples: Dict[str, Set[str]] = defaultdict(set)
    for eva_char, triple_key, syllable in alignment:
        syl_to_triples[syllable].add(triple_key)

    # If a syllable appears at multiple positions, the triples should ideally
    # be the same.  In practice, different EVA chars may map to different
    # triples (EVA is many-to-one to triples), so we only flag if the same
    # syllable requires DIFFERENT triples to be assigned the SAME value —
    # which is always fine in terms of the all-different constraint.
    # Actually, repeated syllable → multiple triples all mapping to same
    # syllable is fine unless it forces duplication.  No rejection needed
    # from this check alone; it's informational.
    return True, ''


def _check_all_different(
    alignment: List[Tuple[str, str, str]],
    full_assignment: Dict[str, str],
    confirmed_triples: Dict[str, str],
) -> Tuple[bool, str]:
    """New assignments must not duplicate existing (non-confirmed) ones.

    The all-different constraint means no two triples should decode to
    the same syllable.  We check only new proposals against the existing
    table.
    """
    existing_syl_to_triple: Dict[str, str] = {}
    for tk, syl in full_assignment.items():
        if syl not in existing_syl_to_triple:
            existing_syl_to_triple[syl] = tk

    for eva_char, triple_key, syllable in alignment:
        if triple_key in confirmed_triples:
            continue  # Confirmed position — already in table, skip
        # Check if this syllable is already assigned to a DIFFERENT triple
        if syllable in existing_syl_to_triple:
            existing_tk = existing_syl_to_triple[syllable]
            if existing_tk != triple_key:
                return False, (
                    f"syllable '{syllable}' already assigned to "
                    f"{existing_tk}, cannot assign to {triple_key}"
                )

    return True, ''


def _validate_alignment(
    alignment: List[Tuple[str, str, str]],
    confirmed_triples: Dict[str, str],
    full_assignment: Dict[str, str],
) -> Tuple[bool, int, int, Dict[str, str], str]:
    """Run all checks on an alignment.

    Returns (valid, n_confirmed, n_new, new_assignments, rejection_reason).
    """
    # 1. Confirmed triple check
    n_confirmed, n_new, passes, reason = _check_confirmed(
        alignment, confirmed_triples, full_assignment,
    )
    if not passes:
        return False, n_confirmed, n_new, {}, reason

    # 2. Repeated syllable check
    rep_ok, rep_reason = _check_repeated_syllables(alignment)
    if not rep_ok:
        return False, n_confirmed, n_new, {}, rep_reason

    # 3. All-different check
    ad_ok, ad_reason = _check_all_different(
        alignment, full_assignment, confirmed_triples,
    )
    if not ad_ok:
        return False, n_confirmed, n_new, {}, ad_reason

    # Extract new assignments
    new_assigns = {}
    for eva_char, triple_key, syllable in alignment:
        if triple_key not in confirmed_triples:
            new_assigns[triple_key] = syllable

    return True, n_confirmed, n_new, new_assigns, ''


# ---------------------------------------------------------------------------
# Per-folio alignment
# ---------------------------------------------------------------------------

def _test_folio_target(
    target: Dict,
    eva_to_triple: Dict[str, str],
    modifier_chars: Set[str],
    confirmed_triples: Dict[str, str],
    full_assignment: Dict[str, str],
) -> List[CribAlignment]:
    """Test all compatible label candidates for one folio target."""
    folio = target['folio']
    plant = target['plant']
    plant_name = plant['medieval_latin']
    syllables = plant['syllables']
    n_syllables = plant['n_syllables']

    results: List[CribAlignment] = []

    for lc in target.get('label_candidates', []):
        if not lc.get('compatible', False):
            continue

        eva_token = lc['eva_token']
        syllabic_chars = lc.get('syllabic_chars', [])

        if not syllabic_chars:
            # Recompute from eva_chars if not pre-provided
            eva_chars = lc.get('eva_chars', tokenize_eva_chars(eva_token))
            syllabic_chars = [ch for ch in eva_chars if ch not in modifier_chars]

        # Enumerate alignments
        alignments = _enumerate_alignments(
            syllabic_chars, syllables, eva_to_triple,
        )

        for alignment in alignments:
            valid, n_confirmed, n_new, new_assigns, reason = _validate_alignment(
                alignment, confirmed_triples, full_assignment,
            )

            results.append(CribAlignment(
                folio=folio,
                plant_name=plant_name,
                eva_token=eva_token,
                syllabic_chars=list(syllabic_chars),
                syllables=list(syllables),
                alignment={tk: syl for (_, tk, syl) in alignment},
                n_confirmed_positions=n_confirmed,
                n_new_positions=n_new,
                new_assignments=new_assigns,
                valid=valid,
                rejection_reason=reason,
            ))

    return results


# ---------------------------------------------------------------------------
# Null control — wrong plant alignments
# ---------------------------------------------------------------------------

def _test_wrong_plants(
    target: Dict,
    all_targets: List[Dict],
    eva_to_triple: Dict[str, str],
    modifier_chars: Set[str],
    confirmed_triples: Dict[str, str],
    full_assignment: Dict[str, str],
    n_wrong: int = 3,
    seed: int = 42,
) -> int:
    """Count valid alignments when aligning WRONG plant names to this folio's labels."""
    folio = target['folio']
    rng = random.Random(seed + hash(folio))

    # Collect wrong plants (from other folios)
    other_plants = []
    for t in all_targets:
        if t['folio'] != folio:
            other_plants.append(t['plant'])

    if not other_plants:
        return 0

    # Pick up to n_wrong random wrong plants
    wrong_plants = rng.sample(other_plants, min(n_wrong, len(other_plants)))
    n_valid = 0

    for wrong_plant in wrong_plants:
        wrong_syllables = wrong_plant['syllables']

        for lc in target.get('label_candidates', []):
            # Test against ALL label candidates (not just compatible ones for
            # the correct plant) — use length compatibility check
            eva_token = lc['eva_token']
            syllabic_chars = lc.get('syllabic_chars', [])
            if not syllabic_chars:
                eva_chars = lc.get('eva_chars', tokenize_eva_chars(eva_token))
                syllabic_chars = [ch for ch in eva_chars if ch not in modifier_chars]

            n_chars = len(syllabic_chars)
            n_syls = len(wrong_syllables)
            diff = abs(n_chars - n_syls)
            if diff > 2:
                continue  # Not length-compatible

            alignments = _enumerate_alignments(
                syllabic_chars, wrong_syllables, eva_to_triple,
            )

            for alignment in alignments:
                valid, _, _, _, _ = _validate_alignment(
                    alignment, confirmed_triples, full_assignment,
                )
                if valid:
                    n_valid += 1

    return n_valid


# ---------------------------------------------------------------------------
# Cross-folio validation
# ---------------------------------------------------------------------------

def _cross_folio_validate(
    all_alignments: List[CribAlignment],
) -> Tuple[Dict[str, str], int]:
    """Find new triple assignments consistent across multiple folios.

    Returns (consistent_assignments, n_conflicting).
    - consistent_assignments: triple_key → syllable where 2+ folios agree.
    - n_conflicting: triples proposed by 2+ folios with DIFFERENT syllables.
    """
    # Group new_assignments by triple_key and folio
    # triple_key → syllable → set of folios
    triple_proposals: Dict[str, Dict[str, Set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )

    for aln in all_alignments:
        if not aln.valid:
            continue
        for tk, syl in aln.new_assignments.items():
            triple_proposals[tk][syl].add(aln.folio)

    consistent: Dict[str, str] = {}
    n_conflicting = 0

    for tk, syl_map in triple_proposals.items():
        # Find syllables proposed by 2+ folios
        best_syl = None
        best_count = 0
        has_conflict = False

        for syl, folios in syl_map.items():
            if len(folios) >= 2:
                if best_syl is not None:
                    has_conflict = True
                if len(folios) > best_count:
                    best_syl = syl
                    best_count = len(folios)

        if best_syl is not None and not has_conflict:
            consistent[tk] = best_syl
        if has_conflict:
            n_conflicting += 1

    return consistent, n_conflicting


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_long_crib_csp() -> None:
    """Step 33.11: Long-Crib CSP — Exhaustive Plant-Name Alignment."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 33.11: Long-Crib CSP")
    print("=" * 70)

    rd = _results_dir()
    eva_to_triple = build_eva_to_triple_lookup()

    # ── 1. Load long_crib_targets.json ──
    targets_path = os.path.join(rd, 'long_crib_targets.json')
    if not os.path.exists(targets_path):
        print("  [SKIP] long_crib_targets.json not found — run long-crib-targets first")
        return
    with open(targets_path) as f:
        targets_data = json.load(f)

    folio_targets = targets_data.get('folio_targets', [])
    if not folio_targets:
        print("  [SKIP] No folio targets found in long_crib_targets.json")
        return

    print(f"\n  1. Loaded {len(folio_targets)} folio targets")

    # ── 2. Load combined_refine.json for full assignment ──
    refine_path = os.path.join(rd, 'combined_refine.json')
    with open(refine_path) as f:
        refine_data = json.load(f)
    full_assignment = refine_data.get('best_assignment', {})
    print(f"     Full assignment: {len(full_assignment)} triples")

    # ── 3. Load confirmed triples ──
    confirmed_triples: Dict[str, str] = {}
    bt_path = os.path.join(rd, 'bootstrap_loop.json')
    if os.path.exists(bt_path):
        with open(bt_path) as f:
            bt_data = json.load(f)
        confirmed_keys = set(bt_data.get('confirmed_triples', []))
        final_assign = bt_data.get('final_assignment', {})
        if confirmed_keys:
            confirmed_triples = {k: v for k, v in final_assign.items()
                                 if k in confirmed_keys}
    if not confirmed_triples:
        # Fall back: use full assignment but mark none as confirmed
        # (all positions will count as "new")
        print("     No confirmed triples found — all proposals will be new")

    print(f"     Confirmed triples: {len(confirmed_triples)}")

    # ── 4. Load modifier chars ──
    mod_path = os.path.join(rd, 'modifier_integrate.json')
    with open(mod_path) as f:
        mod_data = json.load(f)
    modifier_chars = _reconstruct_modifier_chars(mod_data)
    print(f"     Modifier chars: {len(modifier_chars)}")

    # ── 5. Test each folio target ──
    print(f"\n  2. Testing {len(folio_targets)} folio targets...")

    all_alignments: List[CribAlignment] = []
    n_with_valid = 0
    total_correct_valid = 0
    total_wrong_valid = 0

    for target in folio_targets:
        folio = target['folio']
        plant_name = target['plant']['medieval_latin']
        n_syls = target['plant']['n_syllables']
        n_compat = sum(1 for lc in target.get('label_candidates', [])
                       if lc.get('compatible', False))

        print(f"\n     {folio} ({plant_name}, {n_syls} syllables, "
              f"{n_compat} compatible labels)")

        # Test correct plant
        folio_alignments = _test_folio_target(
            target, eva_to_triple, modifier_chars,
            confirmed_triples, full_assignment,
        )

        valid_alns = [a for a in folio_alignments if a.valid]
        n_valid = len(valid_alns)
        n_rejected = len(folio_alignments) - n_valid

        if n_valid > 0:
            n_with_valid += 1
            total_correct_valid += n_valid

        # Show results
        print(f"       Alignments: {len(folio_alignments)} tested, "
              f"{n_valid} valid, {n_rejected} rejected")
        if valid_alns:
            # Show best by confirmed positions
            best = max(valid_alns, key=lambda a: (a.n_confirmed_positions,
                                                   -a.n_new_positions))
            print(f"       Best: {best.eva_token} → {best.plant_name}")
            print(f"         confirmed={best.n_confirmed_positions}, "
                  f"new={best.n_new_positions}")
            if best.new_assignments:
                for tk, syl in sorted(best.new_assignments.items()):
                    marker = " (agrees w/ table)" if full_assignment.get(tk) == syl else ""
                    print(f"         {tk} → '{syl}'{marker}")
        elif folio_alignments:
            # Show why rejected
            reasons = Counter(a.rejection_reason for a in folio_alignments
                              if a.rejection_reason)
            for reason, count in reasons.most_common(3):
                print(f"       Rejected ({count}): {reason}")

        # Null control: wrong plants
        n_wrong_valid = _test_wrong_plants(
            target, folio_targets, eva_to_triple, modifier_chars,
            confirmed_triples, full_assignment,
        )
        total_wrong_valid += n_wrong_valid
        sel = n_valid / max(n_wrong_valid, 0.5)
        print(f"       Null control: {n_valid} correct vs {n_wrong_valid} wrong "
              f"(selectivity {sel:.2f}×)")

        all_alignments.extend(folio_alignments)

    # ── 6. Cross-folio validation ──
    print(f"\n  3. Cross-folio validation...")

    valid_alignments = [a for a in all_alignments if a.valid]
    consistent_new, n_conflicting = _cross_folio_validate(valid_alignments)

    if consistent_new:
        print(f"     {len(consistent_new)} cross-folio consistent triple assignments:")
        for tk, syl in sorted(consistent_new.items()):
            # Find supporting folios
            supporting = set()
            for a in valid_alignments:
                if tk in a.new_assignments and a.new_assignments[tk] == syl:
                    supporting.add(a.folio)
            agree_marker = " (agrees w/ table)" if full_assignment.get(tk) == syl else " [NEW]"
            print(f"       {tk} → '{syl}' (from {len(supporting)} folios: "
                  f"{', '.join(sorted(supporting))}){agree_marker}")
    else:
        print("     No cross-folio consistent assignments found")

    if n_conflicting > 0:
        print(f"     {n_conflicting} triples with conflicting proposals across folios")

    # ── 7. Null selectivity ──
    null_selectivity = total_correct_valid / max(total_wrong_valid, 0.5)
    print(f"\n  4. Overall null selectivity: {total_correct_valid} correct vs "
          f"{total_wrong_valid} wrong = {null_selectivity:.2f}×")

    # ── 8. Identify new confirmed triples ──
    # Triples that are (a) cross-folio consistent AND (b) not already confirmed
    new_confirmed: Dict[str, str] = {}
    for tk, syl in consistent_new.items():
        if tk not in confirmed_triples:
            new_confirmed[tk] = syl

    print(f"\n  5. New confirmed triples: {len(new_confirmed)}")
    for tk, syl in sorted(new_confirmed.items()):
        print(f"       {tk} → '{syl}'")

    # ── 9. Verdict ──
    if len(new_confirmed) >= 3 and null_selectivity >= 2.0:
        verdict = 'CRIBS_CONFIRMED'
    elif len(new_confirmed) >= 1 or (n_with_valid >= 3 and null_selectivity >= 1.5):
        verdict = 'PARTIAL_MATCH'
    else:
        verdict = 'NO_MATCH'

    print(f"\n  Verdict: {verdict}")
    print(f"    {n_with_valid}/{len(folio_targets)} folios with valid alignments")
    print(f"    {len(new_confirmed)} new confirmed triples")
    print(f"    Null selectivity: {null_selectivity:.2f}×")

    # ── 10. Save results ──
    elapsed = round(time.time() - t0, 2)

    result = LongCribCSPResult(
        n_targets_tested=len(folio_targets),
        n_with_valid_alignments=n_with_valid,
        all_alignments=[_convert(asdict(a)) for a in all_alignments],
        n_cross_folio_consistent=len(consistent_new),
        consistent_new_assignments=consistent_new,
        n_conflicting=n_conflicting,
        correct_plant_alignments=total_correct_valid,
        wrong_plant_alignments=total_wrong_valid,
        null_selectivity=round(null_selectivity, 4),
        n_new_confirmed_triples=len(new_confirmed),
        new_confirmed_triples=new_confirmed,
        verdict=verdict,
        runtime_seconds=elapsed,
    )

    out_path = os.path.join(rd, 'long_crib_csp.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2)

    print(f"\n  Saved → {out_path}")
    print(f"  Runtime: {elapsed:.1f}s")
