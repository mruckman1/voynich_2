"""
Phase 23.2 – Historical Inversion Mapping (hist-invert)
========================================================
For each of Phase 16's 25 triple→syllable assignments, searches the
5,199-sign master reference for historical signs whose Latin value starts
with that syllable.  Tests whether the delta between Phase 16 (statistical)
and Phase 22 (historical) assignments follows a systematic pattern —
vowel rotation, consonant class swap, family rotation, etc.

Dependency chain:
    combined_refine.json (Phase 15 best_assignment)
    merged_table.json (Phase 22 EVA→syllable)
    master_reference.json (5,199 historical signs)
        → historical_inversion.json (this step)
"""

import json
import math
import os
import random
import re
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.reference import EVA_VISUAL_COMPONENTS, load_master_reference


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


# Copied from first_syllable.py to avoid internal import issues
_LATIN_VOWELS = set('aeiouy')


def _clean_latin_value(raw: str) -> str:
    """Clean a historical Latin value for syllabification."""
    if not raw:
        return ''
    m = re.match(r'^\([^)]+\)\s+(.+)$', raw)
    if m:
        raw = m.group(1)
    m = re.match(r'^[^\s=]+\s*=\s*(.+)$', raw)
    if m:
        raw = m.group(1)
    raw = re.sub(r'\(([^)]+)\)', r'\1', raw)
    raw = re.sub(r'\[sup:([^\]]+)\]', r'\1', raw)
    raw = raw.split()[0] if raw.strip() else ''
    raw = re.sub(r'[^a-zA-Z]', '', raw)
    return raw.lower()


def _extract_first_cv(word: str) -> str:
    """Extract first CV syllable (consonant cluster + one vowel)."""
    if not word:
        return ''
    word = word.lower()
    parts: List[str] = []
    i = 0
    while i < len(word) and word[i] not in _LATIN_VOWELS:
        parts.append(word[i])
        i += 1
    if i < len(word) and word[i] in _LATIN_VOWELS:
        parts.append(word[i])
    elif not parts:
        return '?'
    return ''.join(parts)


# ---------------------------------------------------------------------------
# EVA → triple mapping helpers
# ---------------------------------------------------------------------------

def _build_triple_to_eva_chars() -> Dict[str, List[str]]:
    """Build mapping from triple_key → list of EVA chars sharing that triple."""
    result: Dict[str, List[str]] = defaultdict(list)
    for eva_char, comp in EVA_VISUAL_COMPONENTS.items():
        tk = f"{comp['first_stroke']},{comp['last_stroke']},{comp['glyph_class']}"
        result[tk].append(eva_char)
    return dict(result)


def _convert_phase22_to_triple_level(
    mode_a_table: List[Dict],
    triple_to_eva: Dict[str, List[str]],
) -> Dict[str, str]:
    """Convert Phase 22's EVA-level table to triple-level by majority vote."""
    # Build EVA char → syllable lookup from mode_a_table
    eva_to_syl: Dict[str, str] = {}
    for entry in mode_a_table:
        eva_char = entry.get('eva_char', '')
        syl = entry.get('syllable_a', '')
        if eva_char and syl and not entry.get('is_modifier', False):
            eva_to_syl[eva_char] = syl

    # For each triple, gather syllables from all EVA chars, take majority vote
    triple_table: Dict[str, str] = {}
    for tk, chars in triple_to_eva.items():
        syls = [eva_to_syl[c] for c in chars if c in eva_to_syl]
        if syls:
            counts = Counter(syls)
            triple_table[tk] = counts.most_common(1)[0][0]

    return triple_table


# ---------------------------------------------------------------------------
# Agreement classification
# ---------------------------------------------------------------------------

def _classify_agreement(syl_a: str, syl_b: str) -> str:
    """Classify agreement between two CV syllables."""
    if not syl_a or not syl_b:
        return 'unrelated'
    if syl_a == syl_b:
        return 'exact'
    # Extract consonant (onset) and vowel (nucleus)
    onset_a = syl_a[:-1] if len(syl_a) > 1 else ''
    vowel_a = syl_a[-1] if syl_a else ''
    onset_b = syl_b[:-1] if len(syl_b) > 1 else ''
    vowel_b = syl_b[-1] if syl_b else ''
    if onset_a == onset_b and onset_a:
        return 'same_consonant'
    if vowel_a == vowel_b and vowel_a in _LATIN_VOWELS:
        return 'same_vowel'
    return 'unrelated'


# ---------------------------------------------------------------------------
# Pattern testing
# ---------------------------------------------------------------------------

_VOWELS_ORDERED = ['a', 'e', 'i', 'o', 'u']

_CONSONANT_CLASSES = {
    'stops': ['p', 'b', 't', 'd', 'c', 'g', 'k'],
    'fricatives': ['f', 'v', 's', 'z', 'h'],
    'nasals': ['m', 'n'],
    'liquids': ['l', 'r'],
}


def _apply_vowel_rotation(syl: str, shift: int) -> str:
    """Rotate the vowel in a CV syllable by `shift` positions."""
    if not syl:
        return syl
    vowel = syl[-1] if syl[-1] in _LATIN_VOWELS else ''
    if not vowel:
        return syl
    idx = _VOWELS_ORDERED.index(vowel) if vowel in _VOWELS_ORDERED else -1
    if idx < 0:
        return syl
    new_vowel = _VOWELS_ORDERED[(idx + shift) % len(_VOWELS_ORDERED)]
    return syl[:-1] + new_vowel


def _apply_consonant_class_swap(syl: str, class_a: str, class_b: str) -> str:
    """Swap consonants between two articulatory classes."""
    if len(syl) < 2:
        return syl
    onset = syl[:-1]
    vowel = syl[-1]
    a_members = _CONSONANT_CLASSES.get(class_a, [])
    b_members = _CONSONANT_CLASSES.get(class_b, [])
    if onset in a_members:
        idx = a_members.index(onset)
        new_onset = b_members[idx % len(b_members)] if b_members else onset
        return new_onset + vowel
    if onset in b_members:
        idx = b_members.index(onset)
        new_onset = a_members[idx % len(a_members)] if a_members else onset
        return new_onset + vowel
    return syl


def _run_pattern_test(
    phase22_triples: Dict[str, str],
    phase16_triples: Dict[str, str],
    transform_fn,
    pattern_name: str,
) -> Dict[str, Any]:
    """Apply a transform to Phase 22 syllables and count matches with Phase 16."""
    common_keys = set(phase22_triples) & set(phase16_triples)
    if not common_keys:
        return {'pattern_name': pattern_name, 'n_agreements': 0,
                'n_total': 0, 'agreement_rate': 0.0, 'p_value': 1.0}

    n_agree = 0
    for tk in common_keys:
        transformed = transform_fn(phase22_triples[tk])
        if transformed == phase16_triples[tk]:
            n_agree += 1

    n_total = len(common_keys)
    rate = n_agree / n_total if n_total > 0 else 0.0

    # p-value via binomial test (chance of agreement = 1/75 per trial)
    p_chance = 1.0 / 75.0
    # P(X >= n_agree) under binomial(n_total, p_chance)
    if n_agree == 0:
        p_value = 1.0
    else:
        # Use normal approximation for binomial
        mu = n_total * p_chance
        sigma = math.sqrt(n_total * p_chance * (1 - p_chance))
        if sigma > 0:
            z = (n_agree - 0.5 - mu) / sigma
            # One-sided p-value: P(Z >= z)
            p_value = 0.5 * math.erfc(z / math.sqrt(2))
        else:
            p_value = 0.0 if n_agree > mu else 1.0

    return {
        'pattern_name': pattern_name,
        'n_agreements': n_agree,
        'n_total': n_total,
        'agreement_rate': round(rate, 4),
        'p_value': round(p_value, 6),
    }


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class TripleInversion:
    triple_key: str
    phase16_syllable: str
    phase22_syllable: str
    n_historical_matches: int
    historical_sources: Dict[str, int]
    agreement_type: str
    eva_chars_in_triple: List[str]


@dataclass
class InversionResult:
    timestamp: str
    n_triples: int
    n_master_signs: int
    inversions: List[Dict]
    n_exact_match: int
    n_same_consonant: int
    n_same_vowel: int
    n_unrelated: int
    pattern_tests: List[Dict]
    best_pattern: str
    best_pattern_agreement: float
    consonant_confusion: Dict[str, str]
    vowel_confusion: Dict[str, str]
    gate_passed: bool
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_historical_inversion() -> Dict[str, Any]:
    """Step 23.2: Historical inversion mapping."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE 23.2: Historical Inversion Mapping")
    print("=" * 70)

    rdir = _results_dir()

    # Load Phase 16 assignment
    combined = _load_json(str(rdir / "combined_refine.json")) or {}
    phase16_assignment = combined.get("best_assignment", {})
    print(f"  Phase 16 assignment: {len(phase16_assignment)} triples")

    # Load Phase 22 merged table
    merged = _load_json(str(rdir / "merged_table.json")) or {}
    mode_a_table = merged.get("mode_a_table", [])
    print(f"  Phase 22 table: {len(mode_a_table)} EVA chars")

    # Load master reference
    master_data = load_master_reference()
    if master_data is None:
        master_signs = []
    else:
        master_signs = master_data.get('all_signs', master_data.get('signs', []))
    n_master = len(master_signs)
    print(f"  Master reference: {n_master} signs")

    # Build triple→EVA mapping
    triple_to_eva = _build_triple_to_eva_chars()

    # Convert Phase 22 to triple level
    phase22_triples = _convert_phase22_to_triple_level(mode_a_table, triple_to_eva)
    print(f"  Phase 22 → triple level: {len(phase22_triples)} triples")

    # --- Per-triple inversion ---
    inversions = []
    agreement_counts = Counter()

    for tk, p16_syl in sorted(phase16_assignment.items()):
        p22_syl = phase22_triples.get(tk, '')
        eva_chars = triple_to_eva.get(tk, [])

        # Search master reference for signs matching p16 syllable
        source_counts: Dict[str, int] = Counter()
        n_matches = 0
        for sign in master_signs:
            lv = sign.get('latin_value', '')
            cleaned = _clean_latin_value(lv)
            first_cv = _extract_first_cv(cleaned)
            if first_cv == p16_syl:
                n_matches += 1
                src = sign.get('source', 'unknown')
                source_counts[src] += 1

        agreement = _classify_agreement(p16_syl, p22_syl)
        agreement_counts[agreement] += 1

        inv = TripleInversion(
            triple_key=tk,
            phase16_syllable=p16_syl,
            phase22_syllable=p22_syl,
            n_historical_matches=n_matches,
            historical_sources=dict(source_counts),
            agreement_type=agreement,
            eva_chars_in_triple=eva_chars,
        )
        inversions.append(_convert(asdict(inv)))

    print(f"  Agreement: exact={agreement_counts['exact']}, "
          f"same_C={agreement_counts['same_consonant']}, "
          f"same_V={agreement_counts['same_vowel']}, "
          f"unrelated={agreement_counts['unrelated']}")

    # --- Systematic pattern tests ---
    pattern_tests = []

    # Identity (baseline)
    pattern_tests.append(
        _run_pattern_test(phase22_triples, phase16_assignment,
                          lambda s: s, 'identity')
    )

    # Vowel rotations (shifts 1-4)
    for shift in range(1, 5):
        pattern_tests.append(
            _run_pattern_test(
                phase22_triples, phase16_assignment,
                lambda s, sh=shift: _apply_vowel_rotation(s, sh),
                f'vowel_rotation_{shift}',
            )
        )

    # Consonant class swaps
    class_names = list(_CONSONANT_CLASSES.keys())
    for i in range(len(class_names)):
        for j in range(i + 1, len(class_names)):
            ca, cb = class_names[i], class_names[j]
            pattern_tests.append(
                _run_pattern_test(
                    phase22_triples, phase16_assignment,
                    lambda s, a=ca, b=cb: _apply_consonant_class_swap(s, a, b),
                    f'swap_{ca}_{cb}',
                )
            )

    # Consonant frequency-order shifts (shift by 1, 2, 3)
    all_consonants = sorted(set(
        s[:-1] for s in phase16_assignment.values() if len(s) > 1
    ))
    for shift in range(1, 4):
        def _shift_consonant(syl, sh=shift, cons=all_consonants):
            if len(syl) < 2:
                return syl
            onset = syl[:-1]
            vowel = syl[-1]
            if onset in cons:
                idx = cons.index(onset)
                new_onset = cons[(idx + sh) % len(cons)]
                return new_onset + vowel
            return syl

        pattern_tests.append(
            _run_pattern_test(
                phase22_triples, phase16_assignment,
                _shift_consonant,
                f'consonant_freq_shift_{shift}',
            )
        )

    # Random permutation baseline (100 trials)
    rng = random.Random(42)
    all_syls = list(set(phase16_assignment.values()))
    random_agreements = []
    for _ in range(100):
        rng.shuffle(all_syls)
        perm_map = {}
        p22_syls_list = list(phase22_triples.values())
        for idx, syl in enumerate(p22_syls_list):
            perm_map[syl] = all_syls[idx % len(all_syls)]

        n_agree = 0
        for tk in set(phase22_triples) & set(phase16_assignment):
            p22 = phase22_triples[tk]
            mapped = perm_map.get(p22, p22)
            if mapped == phase16_assignment[tk]:
                n_agree += 1
        random_agreements.append(n_agree)

    random_mean = sum(random_agreements) / len(random_agreements)
    pattern_tests.append({
        'pattern_name': 'random_permutation',
        'n_agreements': round(random_mean, 1),
        'n_total': len(set(phase22_triples) & set(phase16_assignment)),
        'agreement_rate': round(
            random_mean / max(len(set(phase22_triples) & set(phase16_assignment)), 1),
            4
        ),
        'p_value': 1.0,
    })

    # Find best pattern
    best = max(pattern_tests, key=lambda p: p['agreement_rate'])
    best_pattern = best['pattern_name']
    best_agreement = best['agreement_rate']

    # --- Confusion matrices ---
    consonant_confusion: Dict[str, str] = {}
    vowel_confusion: Dict[str, str] = {}
    for tk in set(phase22_triples) & set(phase16_assignment):
        p16 = phase16_assignment[tk]
        p22 = phase22_triples[tk]
        if len(p16) >= 1 and len(p22) >= 1:
            c16 = p16[:-1] if len(p16) > 1 else ''
            v16 = p16[-1] if p16[-1] in _LATIN_VOWELS else ''
            c22 = p22[:-1] if len(p22) > 1 else ''
            v22 = p22[-1] if p22[-1] in _LATIN_VOWELS else ''
            if c22:
                consonant_confusion[c22] = c16
            if v22:
                vowel_confusion[v22] = v16

    # Gate
    gate_passed = best_agreement > 0.3
    if best_agreement > 0.5:
        verdict = "STRONG SYSTEMATIC PATTERN"
    elif best_agreement > 0.3:
        verdict = "WEAK SYSTEMATIC PATTERN"
    else:
        verdict = "NO SYSTEMATIC PATTERN"

    elapsed = time.time() - t0

    result = InversionResult(
        timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
        n_triples=len(phase16_assignment),
        n_master_signs=n_master,
        inversions=inversions,
        n_exact_match=agreement_counts.get('exact', 0),
        n_same_consonant=agreement_counts.get('same_consonant', 0),
        n_same_vowel=agreement_counts.get('same_vowel', 0),
        n_unrelated=agreement_counts.get('unrelated', 0),
        pattern_tests=pattern_tests,
        best_pattern=best_pattern,
        best_pattern_agreement=round(best_agreement, 4),
        consonant_confusion=consonant_confusion,
        vowel_confusion=vowel_confusion,
        gate_passed=gate_passed,
        verdict=verdict,
        runtime_seconds=round(elapsed, 1),
    )

    out_path = rdir / "historical_inversion.json"
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2, ensure_ascii=False)

    print(f"\n  Best pattern: {best_pattern} ({best_agreement:.1%} agreement)")
    print(f"  Verdict: {verdict}")
    print(f"  → {out_path} ({elapsed:.1f}s)")

    return _convert(asdict(result))
