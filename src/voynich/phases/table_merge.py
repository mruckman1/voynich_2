"""
Phase 22.3 – Table Merge (table-merge)
========================================
Merges first-syllable (22.1), Fontana phonetic (22.2), cross-approach
anchors (19.8), and Phase 15 fallbacks into a single decoding table.

Evidence priority hierarchy (highest to lowest):
  1. First-syllable + Fontana AGREE → highest
  2. Cross-approach anchor confirmed → high
  3. Fontana phonetic alone → high
  4. First-syllable alone → medium
  5. Phase 15 triple-level assignment → low
  6. Family propagation from Priority 1-3 → medium
  7. Unassigned → "?"

Produces TWO merged tables: Mode A (strict CV) and Mode B (CVC).

Dependency chain:
    first_syllable_table.json (22.1) + fontana_phonetic.json (22.2)
    + cross_approach.json (19.8) + combined_refine.json (15.4)
    + modifier_integrate.json (16)
        → merged_table.json (this step)
"""

import json
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.reference import EVA_VISUAL_COMPONENTS


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


def _load_json(path: str) -> Optional[Dict]:
    import os
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class MergedEntry:
    eva_char: str
    syllable_a: str          # Mode A (strict CV)
    syllable_b: str          # Mode B (CVC)
    priority: int            # 1-7 per hierarchy
    source: str              # which evidence source
    is_modifier: bool
    modifier_function: str   # strip/alter/silent
    confidence: str
    conflicts: List[str] = field(default_factory=list)


@dataclass
class MergedTableResult:
    timestamp: str
    n_eva_chars: int
    n_assigned_a: int
    n_assigned_b: int
    n_priority_1: int        # first-syl + fontana agree
    n_priority_2: int        # cross-approach anchor
    n_priority_3: int        # fontana alone
    n_priority_4: int        # first-syl alone
    n_priority_5: int        # phase 15 fallback
    n_priority_6: int        # family propagation
    n_priority_7: int        # unassigned
    n_modifiers: int
    mode_a_table: List[Dict[str, Any]]
    mode_b_table: List[Dict[str, Any]]
    edit_distance_phase15: float
    edit_distance_phase21: float
    n_conflicts: int


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_table_merge() -> Dict[str, Any]:
    """Merge all evidence sources into decoding tables."""
    t0 = time.time()
    rdir = _results_dir()

    # --- Load inputs ---
    first_syl = _load_json(str(rdir / "first_syllable_table.json")) or {}
    fontana = _load_json(str(rdir / "fontana_phonetic.json")) or {}
    cross = _load_json(str(rdir / "cross_approach.json")) or {}
    refine = _load_json(str(rdir / "combined_refine.json")) or {}
    mod_data = _load_json(str(rdir / "modifier_integrate.json")) or {}
    paleo = _load_json(str(rdir / "paleo_table.json")) or {}

    modifier_chars = set(mod_data.get('modifier_chars', []))

    # --- Build lookups ---
    # First-syllable lookup (22.1)
    first_syl_a: Dict[str, str] = {}  # eva_char → CV syllable
    first_syl_b: Dict[str, str] = {}  # eva_char → CVC syllable
    first_syl_conf: Dict[str, str] = {}
    for entry in first_syl.get('mode_a_table', []):
        ec = entry.get('eva_char', '')
        first_syl_a[ec] = entry.get('first_syllable_cv', '')
        first_syl_b[ec] = entry.get('first_syllable_cvc', '')
        first_syl_conf[ec] = entry.get('confidence', '')

    # Fontana phonetic lookup (22.2)
    fontana_syl: Dict[str, str] = {}
    fontana_conf: Dict[str, str] = {}
    for hyp in fontana.get('hypotheses', []):
        ec = hyp.get('eva_char', '')
        fontana_syl[ec] = hyp.get('hypothesized_syllable', '')
        fontana_conf[ec] = hyp.get('confidence', '')

    # Cross-approach anchors (19.8)
    # Build anchor hints: which EVA chars consistently decode to which syllable
    anchor_hints: Dict[str, str] = {}
    for pw in cross.get('per_word_results', []):
        if not (pw.get('exact_match') or pw.get('edit2_match')):
            continue
        best = pw.get('best_decoded', '')
        if not best:
            continue
        # The best_decoded is the full word — not directly a per-char mapping
        # Store the Latin word's first syllable as a hint for validation
        from voynich.phases.first_syllable import _extract_first_cv
        latin_word = pw.get('latin_word', '')
        if latin_word:
            anchor_hints[latin_word] = _extract_first_cv(latin_word)

    # Phase 15 triple-level assignment
    phase15 = refine.get('best_assignment', {})
    eva_to_triple: Dict[str, str] = {}
    for eva_ch, comp in EVA_VISUAL_COMPONENTS.items():
        triple_key = f"{comp['first_stroke']},{comp['last_stroke']},{comp['glyph_class']}"
        eva_to_triple[eva_ch] = triple_key

    # Phase 21 paleo table
    paleo_syl: Dict[str, str] = {}
    for entry in paleo.get('table', []):
        ec = entry.get('eva_char', '')
        syl = entry.get('latin_syllable', '')
        if ec and syl:
            paleo_syl[ec] = syl

    # Modifier classification from Phase 16
    mod_classifications = {}
    for cls in mod_data.get('classifications', []):
        ec = cls.get('eva_char', '')
        if ec:
            mod_classifications[ec] = cls

    # --- Merge with priority hierarchy ---
    all_eva_chars = sorted(set(
        list(first_syl_a.keys()) +
        list(fontana_syl.keys()) +
        list(eva_to_triple.keys())
    ))

    merged: List[MergedEntry] = []
    priority_counts = {i: 0 for i in range(1, 8)}

    for eva_ch in all_eva_chars:
        is_mod = eva_ch in modifier_chars
        fs_a = first_syl_a.get(eva_ch, '')
        fs_b = first_syl_b.get(eva_ch, '')
        fn_syl = fontana_syl.get(eva_ch, '')
        fs_conf = first_syl_conf.get(eva_ch, '')
        fn_conf = fontana_conf.get(eva_ch, '')

        # Phase 15 fallback
        triple = eva_to_triple.get(eva_ch, '')
        p15_syl = phase15.get(triple, '')

        # Modifier function
        mod_func = 'syllabic'
        if is_mod:
            cls_info = mod_classifications.get(eva_ch, {})
            mod_func = 'strip'  # Default R3 strategy for modifiers

        conflicts: List[str] = []

        # Priority 1: first-syllable + fontana AGREE
        if fs_a and fn_syl and fs_a == fn_syl:
            syl_a = fs_a
            syl_b = fs_b
            priority = 1
            source = 'first_syl+fontana'
            confidence = 'high'

        # Priority 2: check if this char's assignment matches a cross-approach anchor
        elif fs_a and fs_a in anchor_hints.values():
            syl_a = fs_a
            syl_b = fs_b
            priority = 2
            source = 'first_syl+anchor'
            confidence = 'high'

        # Priority 3: Fontana alone (with good confidence)
        elif fn_syl and fn_conf in ('high', 'medium'):
            syl_a = fn_syl
            syl_b = fn_syl
            priority = 3
            source = 'fontana'
            confidence = fn_conf
            if fs_a and fs_a != fn_syl:
                conflicts.append(f'first_syl={fs_a}')

        # Priority 4: First-syllable alone
        elif fs_a and fs_conf not in ('none', 'fallback_p15', 'fallback_family'):
            syl_a = fs_a
            syl_b = fs_b
            priority = 4
            source = 'first_syllable'
            confidence = fs_conf
            if fn_syl and fn_syl != fs_a:
                conflicts.append(f'fontana={fn_syl}')

        # Priority 5: Phase 15 triple-level
        elif p15_syl:
            syl_a = p15_syl
            syl_b = p15_syl
            priority = 5
            source = 'phase15'
            confidence = 'low'

        # Priority 6: family propagation (from first_syl fallbacks)
        elif fs_a and fs_conf in ('fallback_family', 'fallback_p15'):
            syl_a = fs_a
            syl_b = fs_b
            priority = 6
            source = fs_conf
            confidence = 'low'

        # Priority 7: unassigned
        else:
            syl_a = ''
            syl_b = ''
            priority = 7
            source = 'unassigned'
            confidence = 'none'

        # For modifiers, clear syllable assignment (they modify, not decode)
        if is_mod:
            # Keep the syllable for reference but mark modifier
            pass

        priority_counts[priority] = priority_counts.get(priority, 0) + 1

        merged.append(MergedEntry(
            eva_char=eva_ch,
            syllable_a=syl_a,
            syllable_b=syl_b,
            priority=priority,
            source=source,
            is_modifier=is_mod,
            modifier_function=mod_func,
            confidence=confidence,
            conflicts=conflicts,
        ))

    # --- Family coherence post-processing ---
    # If 3/4+ members of a family agree on consonant, override minority
    family_groups: Dict[str, List[MergedEntry]] = {}
    for entry in merged:
        comp = EVA_VISUAL_COMPONENTS.get(entry.eva_char, {})
        gc = comp.get('glyph_class', '')
        family_groups.setdefault(gc, []).append(entry)

    for gc, members in family_groups.items():
        non_mod = [m for m in members if not m.is_modifier and m.syllable_a]
        if len(non_mod) < 3:
            continue
        # Extract consonant onsets
        onset_counts: Counter = Counter()
        for m in non_mod:
            onset = ''
            for ch in m.syllable_a:
                if ch in 'aeiou':
                    break
                onset += ch
            onset_counts[onset] += 1

        if not onset_counts:
            continue
        dominant_onset, dom_count = onset_counts.most_common(1)[0]
        agreement_frac = dom_count / len(non_mod)

        # If >= 75% agree and an outlier has priority > 3, override
        if agreement_frac >= 0.75:
            for m in non_mod:
                onset = ''
                for ch in m.syllable_a:
                    if ch in 'aeiou':
                        break
                    onset += ch
                if onset != dominant_onset and m.priority > 3:
                    # Replace consonant
                    vowel = m.syllable_a[len(onset):] if len(onset) < len(m.syllable_a) else 'e'
                    old_syl = m.syllable_a
                    m.syllable_a = dominant_onset + vowel
                    m.syllable_b = dominant_onset + vowel
                    m.conflicts.append(f'family_override:{old_syl}→{m.syllable_a}')

    # --- Compute edit distances ---
    def _edit_distance_tables(merged_table: List[MergedEntry], ref: Dict[str, str]) -> float:
        """Fraction of chars where merged disagrees with reference."""
        n_compared = 0
        n_differ = 0
        for entry in merged_table:
            if entry.is_modifier:
                continue
            triple = eva_to_triple.get(entry.eva_char, '')
            ref_syl = ref.get(triple, '') or ref.get(entry.eva_char, '')
            if not ref_syl or not entry.syllable_a:
                continue
            n_compared += 1
            if entry.syllable_a != ref_syl:
                n_differ += 1
        return n_differ / max(n_compared, 1)

    ed_p15 = _edit_distance_tables(merged, phase15)
    ed_p21 = _edit_distance_tables(merged, paleo_syl)

    # --- Build result ---
    n_assigned_a = sum(1 for m in merged if m.syllable_a and not m.is_modifier)
    n_assigned_b = sum(1 for m in merged if m.syllable_b and not m.is_modifier)
    n_conflicts = sum(1 for m in merged if m.conflicts)

    result = MergedTableResult(
        timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
        n_eva_chars=len(merged),
        n_assigned_a=n_assigned_a,
        n_assigned_b=n_assigned_b,
        n_priority_1=priority_counts[1],
        n_priority_2=priority_counts[2],
        n_priority_3=priority_counts[3],
        n_priority_4=priority_counts[4],
        n_priority_5=priority_counts[5],
        n_priority_6=priority_counts[6],
        n_priority_7=priority_counts[7],
        n_modifiers=sum(1 for m in merged if m.is_modifier),
        mode_a_table=[_convert(asdict(m)) for m in merged],
        mode_b_table=[_convert(asdict(m)) for m in merged],
        edit_distance_phase15=round(ed_p15, 4),
        edit_distance_phase21=round(ed_p21, 4),
        n_conflicts=n_conflicts,
    )

    out_path = rdir / "merged_table.json"
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t0
    print(f"table-merge: {len(merged)} chars, A={n_assigned_a} B={n_assigned_b}, "
          f"P1={priority_counts[1]} P2={priority_counts[2]} P3={priority_counts[3]} "
          f"P4={priority_counts[4]} P5={priority_counts[5]}, "
          f"conflicts={n_conflicts}, ed_p15={ed_p15:.2f} ({elapsed:.1f}s)")

    return _convert(asdict(result))
