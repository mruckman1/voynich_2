"""
Phase 23.3 – Bench Family Split Analysis (bench-split)
=======================================================
Phase 22 mapped all 24 bench-class EVA characters to Fontana's circle
family, producing consonant "q" for all of them — a degenerate mapping.
This step subdivides the bench class by secondary stroke features
(first_stroke, last_stroke), matches each subgroup to the most
structurally similar Fontana family, and derives revised phonetic
assignments.  Compares against Phase 16's statistical assignments.

Dependency chain:
    combined_refine.json (Phase 15 best_assignment)
    fontana_families.json (Phase 21)
    fontana_phonetic.json (Phase 22.2 family summary)
        → bench_split.json (this step)
"""

import json
import os
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.reference import (
    EVA_VISUAL_COMPONENTS,
    PHONEME_NUCLEUS_MAP,
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


def _load_json(path: str) -> Optional[Dict]:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Stroke-to-Fontana family similarity
# ---------------------------------------------------------------------------

# Maps Voynich first_stroke types to structurally similar Fontana base forms
# (ordered by similarity, best first)
_STROKE_TO_FONTANA = {
    'loop': ['circle', 'closed_loop', 'oval', 'open_curve_right'],
    'open_curve': ['open_curve_right', 'open_curve_left', 'curve', 'circle'],
    'sigmoid': ['curve', 'open_curve_left', 'diagonal_left', 'hook'],
    'connector': ['horizontal_stroke', 'angle', 'diagonal_right'],
    'ascender': ['vertical_stroke', 'ascender', 'ascender_variant'],
    'vertical': ['vertical_stroke', 'minim', 'diagonal_right'],
    'crossbar': ['horizontal_stroke', 'crossbar', 'angle'],
    'descender': ['vertical_stroke', 'diagonal_left'],
    'hook': ['hook', 'curve', 'open_curve_right'],
    'tail': ['diagonal_left', 'diagonal_right', 'hook'],
    'plume': ['diagonal_right', 'vertical_stroke'],
}

# Maps Voynich last_stroke to a vowel preference (used when PHONEME_NUCLEUS_MAP
# doesn't have a mapping)
_LAST_STROKE_VOWEL_FALLBACK = {
    'loop': 'a',
    'tail': 'a',
    'sigmoid': 'e',
    'vertical': 'i',
    'descender': 'o',
    'hook': 'i',
    'connector': 'e',
    'open_curve': 'a',
    'crossbar': 'e',
    'ascender': 'e',
    'plume': 'u',
}


def _get_vowel_for_last_stroke(last_stroke: str) -> str:
    """Derive vowel from last stroke, using PHONEME_NUCLEUS_MAP first."""
    candidates = PHONEME_NUCLEUS_MAP.get(last_stroke, [])
    if candidates:
        return candidates[0]  # first candidate = best match
    return _LAST_STROKE_VOWEL_FALLBACK.get(last_stroke, 'a')


def _score_family_match(
    subgroup_first_stroke: str,
    fontana_base_form: str,
) -> float:
    """Score how well a Voynich first_stroke matches a Fontana family."""
    candidates = _STROKE_TO_FONTANA.get(subgroup_first_stroke, [])
    if fontana_base_form in candidates:
        idx = candidates.index(fontana_base_form)
        return 1.0 - (idx * 0.2)  # 1.0 for best, 0.8 for second, etc.
    return 0.0


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class BenchSubgroup:
    subgroup_id: str
    first_stroke: str
    last_stroke: str
    eva_chars: List[str]
    n_chars: int
    triple_key: str
    phase16_syllable: str
    fontana_family_original: str
    fontana_family_proposed: str
    fontana_match_score: float
    proposed_consonant: str
    proposed_vowel: str
    proposed_syllable: str
    agrees_with_phase16: bool


@dataclass
class BenchSplitResult:
    timestamp: str
    n_bench_chars: int
    n_subgroups: int
    subgroups: List[Dict]
    n_agree_phase16: int
    agreement_rate_phase16: float
    fontana_families_used: List[str]
    revised_bench_table: Dict[str, str]
    gate_passed: bool
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_bench_split() -> Dict[str, Any]:
    """Step 23.3: Bench family split analysis."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE 23.3: Bench Family Split Analysis")
    print("=" * 70)

    rdir = _results_dir()

    # Load Phase 16 assignment
    combined = _load_json(str(rdir / "combined_refine.json")) or {}
    phase16_assignment = combined.get("best_assignment", {})

    # Load Fontana families
    fontana_data = _load_json(str(rdir / "fontana_families.json")) or {}
    fontana_families = fontana_data.get("families", [])
    family_base_forms = [f['base_form'] for f in fontana_families]
    print(f"  Fontana families: {len(fontana_families)}")

    # Load Fontana phonetic data (for consonant-per-family)
    fontana_phon = _load_json(str(rdir / "fontana_phonetic.json")) or {}
    family_summary = fontana_phon.get("fontana_family_summary", [])

    # Build family → consonant mapping
    family_consonants: Dict[str, List[str]] = {}
    for entry in family_summary:
        bf = entry.get('base_form', '')
        cons = entry.get('consonants', [])
        if bf:
            family_consonants[bf] = cons

    # Identify bench-class EVA chars
    bench_chars = []
    for eva_char, comp in EVA_VISUAL_COMPONENTS.items():
        if comp['glyph_class'] == 'bench':
            bench_chars.append(eva_char)
    print(f"  Bench-class EVA chars: {len(bench_chars)}")

    # Subdivide by (first_stroke, last_stroke)
    subgroup_map: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for eva_char in bench_chars:
        comp = EVA_VISUAL_COMPONENTS[eva_char]
        key = (comp['first_stroke'], comp['last_stroke'])
        subgroup_map[key].append(eva_char)

    print(f"  Subgroups (by first_stroke, last_stroke): {len(subgroup_map)}")

    # Process each subgroup
    subgroups = []
    n_agree = 0
    revised_table: Dict[str, str] = {}
    families_used: Set[str] = set()

    for (fs, ls), chars in sorted(subgroup_map.items()):
        subgroup_id = f"{fs}_{ls}"
        triple_key = f"{fs},{ls},bench"

        # Phase 16 syllable for this triple
        p16_syl = phase16_assignment.get(triple_key, '')

        # Score all Fontana families for this subgroup
        best_family = 'circle'
        best_score = 0.0
        for bf in family_base_forms:
            score = _score_family_match(fs, bf)
            if score > best_score:
                best_score = score
                best_family = bf

        # Get consonant from best-matched family
        cons_list = family_consonants.get(best_family, [])
        proposed_consonant = cons_list[0] if cons_list else 'q'

        # Get vowel from last stroke
        proposed_vowel = _get_vowel_for_last_stroke(ls)

        # Build proposed syllable
        if proposed_consonant in ('a', 'e', 'i', 'o', 'u'):
            # Family encodes vowels — use pure vowel
            proposed_syllable = proposed_vowel
        else:
            proposed_syllable = proposed_consonant + proposed_vowel

        agrees = proposed_syllable == p16_syl
        if agrees:
            n_agree += 1
        families_used.add(best_family)

        # Add to revised table for each EVA char in subgroup
        for c in chars:
            revised_table[c] = proposed_syllable

        sg = BenchSubgroup(
            subgroup_id=subgroup_id,
            first_stroke=fs,
            last_stroke=ls,
            eva_chars=sorted(chars),
            n_chars=len(chars),
            triple_key=triple_key,
            phase16_syllable=p16_syl,
            fontana_family_original='circle',
            fontana_family_proposed=best_family,
            fontana_match_score=round(best_score, 2),
            proposed_consonant=proposed_consonant,
            proposed_vowel=proposed_vowel,
            proposed_syllable=proposed_syllable,
            agrees_with_phase16=agrees,
        )
        subgroups.append(_convert(asdict(sg)))

    n_subgroups = len(subgroups)
    agreement_rate = n_agree / n_subgroups if n_subgroups > 0 else 0.0

    # Gate
    gate_passed = agreement_rate > 0.3
    if agreement_rate > 0.5:
        verdict = "STRONG IMPROVEMENT — split mapping aligns with Phase 16"
    elif agreement_rate > 0.3:
        verdict = "MODERATE IMPROVEMENT — partial alignment with Phase 16"
    elif agreement_rate > 0.1:
        verdict = "WEAK IMPROVEMENT — minimal alignment"
    else:
        verdict = "NO IMPROVEMENT — split mapping does not match Phase 16"

    elapsed = time.time() - t0

    result = BenchSplitResult(
        timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
        n_bench_chars=len(bench_chars),
        n_subgroups=n_subgroups,
        subgroups=subgroups,
        n_agree_phase16=n_agree,
        agreement_rate_phase16=round(agreement_rate, 4),
        fontana_families_used=sorted(families_used),
        revised_bench_table=revised_table,
        gate_passed=gate_passed,
        verdict=verdict,
        runtime_seconds=round(elapsed, 1),
    )

    out_path = rdir / "bench_split.json"
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2, ensure_ascii=False)

    print(f"\n  Subgroups: {n_subgroups}")
    print(f"  Fontana families used: {sorted(families_used)}")
    print(f"  Agreement with Phase 16: {n_agree}/{n_subgroups}"
          f" ({agreement_rate:.1%})")
    print(f"  Verdict: {verdict}")
    print(f"  → {out_path} ({elapsed:.1f}s)")

    return _convert(asdict(result))
