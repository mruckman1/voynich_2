"""
Step 24.15 – Voynich Word Grammar Exploitation
===============================================
Exploit the rigid positional rules of EVA characters to constrain
decoding independently of any phonetic table.

Dependency chain:
    combined_refine.json (Phase 15)
    modifier_integrate.json (Phase 16)
        → token_grammar.json (this step)
"""

import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    decode_token_modifier_aware,
    load_corpus,
    tokenize_eva_chars,
)
from voynich.core.reference import (
    EVA_VISUAL_COMPONENTS,
    build_cv_syllable_table,
    build_expanded_word_set,
    load_reference_corpus,
)
from voynich.phases.csp_solver import decode_token


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
class EVAPositionalProfile:
    eva_char: str
    p_initial: float
    p_medial: float
    p_final: float
    p_solo: float
    total_occurrences: int
    position_class: str  # "initial_only", "final_only", "medial_only", "mixed"


@dataclass
class PositionalViolation:
    triple_key: str
    eva_char: str
    eva_position_class: str
    assigned_syllable: str
    syllable_position_class: str
    compatibility_score: float
    suggested_replacement: str
    suggestion_score: float


@dataclass
class TokenGrammarResult:
    timestamp: str
    # EVA profiles
    n_eva_chars_profiled: int
    n_initial_only: int
    n_final_only: int
    n_medial_only: int
    n_mixed: int
    eva_profiles: List[Dict]
    # Latin syllable profiles
    n_syllables_profiled: int
    n_initial_dominant: int
    n_final_dominant: int
    n_medial_dominant: int
    # Phase 16 violations
    n_violations: int
    violations: List[Dict]
    violation_rate: float  # fraction of assignments that violate
    # Corrective filter
    n_corrections_proposed: int
    corrections: List[Dict]
    # Gallows analysis
    gallows_chars: List[str]
    gallows_initial_rate: float  # fraction at line/paragraph start
    gallows_syllables: List[Dict]  # {char, syllable, position_class}
    # Verdict
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Step 1 – EVA character positional profiles
# ---------------------------------------------------------------------------

def _build_eva_positional_profiles(
    corpus_tokens: List[str],
) -> Dict[str, Dict[str, float]]:
    """Compute positional distribution for each EVA character within tokens.

    Returns dict of eva_char -> {p_initial, p_medial, p_final, p_solo, total}.
    """
    profiles: Dict[str, Counter] = defaultdict(Counter)

    for token in corpus_tokens:
        chars = tokenize_eva_chars(token)
        if not chars:
            continue
        for i, ch in enumerate(chars):
            if len(chars) == 1:
                profiles[ch]['solo'] += 1
            elif i == 0:
                profiles[ch]['initial'] += 1
            elif i == len(chars) - 1:
                profiles[ch]['final'] += 1
            else:
                profiles[ch]['medial'] += 1

    result: Dict[str, Dict[str, float]] = {}
    for ch, counts in profiles.items():
        total = sum(counts.values())
        result[ch] = {
            'p_initial': counts['initial'] / total if total else 0.0,
            'p_medial': counts['medial'] / total if total else 0.0,
            'p_final': counts['final'] / total if total else 0.0,
            'p_solo': counts['solo'] / total if total else 0.0,
            'total': total,
        }
    return result


def _classify_eva_position(profile: Dict[str, float]) -> str:
    """Classify an EVA character into a positional class.

    - initial_only:  P(initial) + P(solo) > 0.8
    - final_only:    P(final)   + P(solo) > 0.8
    - medial_only:   P(medial)  > 0.8
    - mixed:         none of the above
    """
    if profile['p_initial'] + profile['p_solo'] > 0.8:
        return 'initial_only'
    if profile['p_final'] + profile['p_solo'] > 0.8:
        return 'final_only'
    if profile['p_medial'] > 0.8:
        return 'medial_only'
    return 'mixed'


# ---------------------------------------------------------------------------
# Step 2 – Latin syllable positional profiles
# ---------------------------------------------------------------------------

def _syllabify_latin(word: str) -> List[str]:
    """Simple vowel-based syllabification for a Latin word."""
    vowels = set('aeiou')
    syllables: List[str] = []
    current = ''
    for ch in word.lower():
        current += ch
        if ch in vowels:
            syllables.append(current)
            current = ''
    if current:
        if syllables:
            syllables[-1] += current
        else:
            syllables.append(current)
    return syllables


def _build_latin_syllable_positions(
    ref_words: List[str],
) -> Dict[str, Dict[str, float]]:
    """Compute positional distribution for Latin syllables within words.

    Returns dict of syllable -> {p_initial, p_medial, p_final, p_solo, total}.
    """
    profiles: Dict[str, Counter] = defaultdict(Counter)

    for word in ref_words:
        syls = _syllabify_latin(word)
        if not syls:
            continue
        for i, syl in enumerate(syls):
            if len(syls) == 1:
                profiles[syl]['solo'] += 1
            elif i == 0:
                profiles[syl]['initial'] += 1
            elif i == len(syls) - 1:
                profiles[syl]['final'] += 1
            else:
                profiles[syl]['medial'] += 1

    result: Dict[str, Dict[str, float]] = {}
    for syl, counts in profiles.items():
        total = sum(counts.values())
        result[syl] = {
            'p_initial': counts['initial'] / total if total else 0.0,
            'p_medial': counts['medial'] / total if total else 0.0,
            'p_final': counts['final'] / total if total else 0.0,
            'p_solo': counts['solo'] / total if total else 0.0,
            'total': total,
        }
    return result


def _classify_syllable_position(profile: Dict[str, float]) -> str:
    """Classify a Latin syllable into a positional class.

    Uses relaxed thresholds (0.6) compared to EVA classification (0.8)
    because Latin syllables have more distributional spread.

    - initial_dominant:  P(initial) + P(solo) > 0.6
    - final_dominant:    P(final)   + P(solo) > 0.6
    - medial_dominant:   P(medial)  > 0.6
    - mixed:             none of the above
    """
    if profile['p_initial'] + profile['p_solo'] > 0.6:
        return 'initial_dominant'
    if profile['p_final'] + profile['p_solo'] > 0.6:
        return 'final_dominant'
    if profile['p_medial'] > 0.6:
        return 'medial_dominant'
    return 'mixed'


# ---------------------------------------------------------------------------
# Step 3 – Positional compatibility matrix (cosine similarity)
# ---------------------------------------------------------------------------

def _cosine_similarity(
    vec_a: Tuple[float, ...],
    vec_b: Tuple[float, ...],
) -> float:
    """Cosine similarity between two vectors."""
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    mag_a = math.sqrt(sum(a * a for a in vec_a))
    mag_b = math.sqrt(sum(b * b for b in vec_b))
    if mag_a < 1e-12 or mag_b < 1e-12:
        return 0.0
    return dot / (mag_a * mag_b)


def _position_vector(profile: Dict[str, float]) -> Tuple[float, ...]:
    """Extract [p_initial, p_medial, p_final, p_solo] as a tuple."""
    return (
        profile['p_initial'],
        profile['p_medial'],
        profile['p_final'],
        profile['p_solo'],
    )


def _compute_compatibility(
    eva_profile: Dict[str, float],
    syl_profile: Dict[str, float],
) -> float:
    """Cosine similarity between an EVA char's and a syllable's positional
    distribution vectors."""
    return _cosine_similarity(
        _position_vector(eva_profile),
        _position_vector(syl_profile),
    )


# ---------------------------------------------------------------------------
# Step 4 – Check Phase 16 violations
# ---------------------------------------------------------------------------

def _check_violations(
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    eva_profiles: Dict[str, Dict[str, float]],
    syl_profiles: Dict[str, Dict[str, float]],
) -> List[PositionalViolation]:
    """Identify Phase 16 triple->syllable assignments that violate
    positional compatibility.

    A violation occurs when an EVA character's strong positional preference
    contradicts the assigned syllable's positional profile:
      - EVA char with P(initial)+P(solo) > 0.8 assigned a syllable with
        P(final)+P(solo) > 0.6
      - EVA char with P(final)+P(solo) > 0.8 assigned a syllable with
        P(initial)+P(solo) > 0.6
    """
    violations: List[PositionalViolation] = []

    # Build reverse map: triple_key -> list of EVA chars
    triple_to_chars: Dict[str, List[str]] = defaultdict(list)
    for ch, tk in eva_to_triple.items():
        triple_to_chars[tk].append(ch)

    for triple_key, syllable in assignment.items():
        syl_prof = syl_profiles.get(syllable)
        if syl_prof is None:
            continue
        syl_class = _classify_syllable_position(syl_prof)

        for ch in triple_to_chars.get(triple_key, []):
            eva_prof = eva_profiles.get(ch)
            if eva_prof is None:
                continue
            eva_class = _classify_eva_position(eva_prof)

            # Check for contradictions
            is_violation = False
            if eva_class == 'initial_only' and syl_class == 'final_dominant':
                is_violation = True
            elif eva_class == 'final_only' and syl_class == 'initial_dominant':
                is_violation = True
            elif eva_class == 'medial_only' and syl_class in (
                'initial_dominant', 'final_dominant',
            ):
                is_violation = True

            if is_violation:
                compat = _compute_compatibility(eva_prof, syl_prof)
                violations.append(PositionalViolation(
                    triple_key=triple_key,
                    eva_char=ch,
                    eva_position_class=eva_class,
                    assigned_syllable=syllable,
                    syllable_position_class=syl_class,
                    compatibility_score=compat,
                    suggested_replacement='',
                    suggestion_score=0.0,
                ))

    return violations


# ---------------------------------------------------------------------------
# Step 5 – Corrective filter
# ---------------------------------------------------------------------------

def _propose_corrections(
    violations: List[PositionalViolation],
    eva_profiles: Dict[str, Dict[str, float]],
    syl_profiles: Dict[str, Dict[str, float]],
    candidate_syllables: List[str],
) -> List[PositionalViolation]:
    """For each violation, find the most compatible syllable replacement.

    Searches all candidate syllables for the one with the highest cosine
    similarity to the violated EVA character's positional vector.
    """
    corrected: List[PositionalViolation] = []

    for viol in violations:
        eva_prof = eva_profiles.get(viol.eva_char)
        if eva_prof is None:
            corrected.append(viol)
            continue

        eva_vec = _position_vector(eva_prof)
        best_syl = ''
        best_score = -1.0

        for syl in candidate_syllables:
            sp = syl_profiles.get(syl)
            if sp is None:
                continue
            # Only consider syllables that occur at least a few times
            if sp.get('total', 0) < 5:
                continue
            score = _cosine_similarity(eva_vec, _position_vector(sp))
            if score > best_score:
                best_score = score
                best_syl = syl

        corrected.append(PositionalViolation(
            triple_key=viol.triple_key,
            eva_char=viol.eva_char,
            eva_position_class=viol.eva_position_class,
            assigned_syllable=viol.assigned_syllable,
            syllable_position_class=viol.syllable_position_class,
            compatibility_score=viol.compatibility_score,
            suggested_replacement=best_syl,
            suggestion_score=best_score,
        ))

    return corrected


# ---------------------------------------------------------------------------
# Step 6 – Gallows analysis
# ---------------------------------------------------------------------------

# Gallows characters: the large decorative characters and their compounds
GALLOWS_CHARS = ['t', 'k', 'p', 'f', 'cth', 'ckh', 'cph', 'cfh']


def _gallows_analysis(
    corpus,
    eva_to_triple: Dict[str, str],
    assignment: Dict[str, str],
    syl_profiles: Dict[str, Dict[str, float]],
) -> Tuple[float, List[Dict]]:
    """Analyse gallows characters for line/paragraph-initial bias.

    Returns (initial_rate, syllable_info_list).

    initial_rate: fraction of gallows-containing tokens that are the first
                  token on their line/paragraph.
    syllable_info_list: for each gallows char, its assigned syllable and
                        positional class.
    """
    gallows_set = set(GALLOWS_CHARS)
    total_gallows_tokens = 0
    initial_gallows_tokens = 0

    # Walk through loci (lines) to detect first-token positions
    for page in corpus.pages.values():
        for locus in page.loci:
            if not locus.locus_type.startswith('P'):
                continue
            text = locus.clean_text
            if not text:
                continue
            line_tokens = text.split()
            for ti, tok in enumerate(line_tokens):
                chars = tokenize_eva_chars(tok)
                has_gallows = any(ch in gallows_set for ch in chars)
                if has_gallows:
                    total_gallows_tokens += 1
                    if ti == 0:
                        initial_gallows_tokens += 1

    initial_rate = (
        initial_gallows_tokens / total_gallows_tokens
        if total_gallows_tokens > 0
        else 0.0
    )

    # Syllable info for each gallows character
    syllable_info: List[Dict] = []
    for gch in GALLOWS_CHARS:
        tk = eva_to_triple.get(gch)
        if tk is None:
            syllable_info.append({
                'char': gch,
                'syllable': '?',
                'position_class': 'unknown',
            })
            continue
        syl = assignment.get(tk, '?')
        sp = syl_profiles.get(syl)
        pos_class = _classify_syllable_position(sp) if sp else 'unknown'
        syllable_info.append({
            'char': gch,
            'syllable': syl,
            'position_class': pos_class,
        })

    return initial_rate, syllable_info


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_token_grammar() -> None:
    """Step 24.15: Voynich Word Grammar Exploitation."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 24.15: Voynich Word Grammar Exploitation")
    print("=" * 70)

    rd = _results_dir()

    # ─── Load Phase 15 / Phase 16 results ───────────────────────────
    print("\n  1. Loading dependency results ...")

    refine_path = os.path.join(rd, 'combined_refine.json')
    if not os.path.exists(refine_path):
        print("  [SKIP] combined_refine.json not found — run combined-refine first")
        return
    with open(refine_path) as f:
        refine_data = json.load(f)
    assignment = refine_data.get('best_assignment', {})

    mod_path = os.path.join(rd, 'modifier_integrate.json')
    modifier_chars: Set[str] = set()
    if os.path.exists(mod_path):
        with open(mod_path) as f:
            mod_data = json.load(f)
        modifier_chars = set(mod_data.get('modifier_chars', []))
    print(f"     Phase 15 assignment: {len(assignment)} triples")
    print(f"     Phase 16 modifiers:  {len(modifier_chars)} chars")

    # ─── Load corpus ────────────────────────────────────────────────
    print("\n  2. Loading corpus ...")
    corpus = load_corpus(verbose=False)
    tokens = corpus.get_tokens()
    print(f"     {len(tokens):,} tokens")

    # ─── Load reference corpus ──────────────────────────────────────
    print("\n  3. Loading Latin reference corpus ...")
    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    ref_words = ref_corpus.get_combined_tokens('latin')
    print(f"     {len(ref_words):,} Latin reference words")

    # ─── Build EVA-to-triple lookup ─────────────────────────────────
    eva_to_triple = build_eva_to_triple_lookup()

    # ─── Step 1: EVA positional profiles ────────────────────────────
    print("\n  4. Building EVA character positional profiles ...")
    eva_profiles = _build_eva_positional_profiles(tokens)

    # Classify
    eva_profile_records: List[EVAPositionalProfile] = []
    class_counts: Counter = Counter()
    for ch, prof in sorted(eva_profiles.items(), key=lambda x: -x[1]['total']):
        pos_class = _classify_eva_position(prof)
        class_counts[pos_class] += 1
        eva_profile_records.append(EVAPositionalProfile(
            eva_char=ch,
            p_initial=round(prof['p_initial'], 4),
            p_medial=round(prof['p_medial'], 4),
            p_final=round(prof['p_final'], 4),
            p_solo=round(prof['p_solo'], 4),
            total_occurrences=int(prof['total']),
            position_class=pos_class,
        ))

    n_initial_only = class_counts['initial_only']
    n_final_only = class_counts['final_only']
    n_medial_only = class_counts['medial_only']
    n_mixed = class_counts['mixed']

    print(f"     {len(eva_profiles)} EVA chars profiled")
    print(f"       initial_only: {n_initial_only}")
    print(f"       final_only:   {n_final_only}")
    print(f"       medial_only:  {n_medial_only}")
    print(f"       mixed:        {n_mixed}")

    # Print top 5 by frequency in each non-empty class
    for cls_name in ('initial_only', 'final_only', 'medial_only'):
        members = [r for r in eva_profile_records if r.position_class == cls_name]
        if members:
            top = sorted(members, key=lambda r: -r.total_occurrences)[:5]
            names = ', '.join(f"{r.eva_char}({r.total_occurrences})" for r in top)
            print(f"       top {cls_name}: {names}")

    # ─── Step 2: Latin syllable positional profiles ─────────────────
    print("\n  5. Building Latin syllable positional profiles ...")
    syl_profiles = _build_latin_syllable_positions(ref_words)
    syl_classes: Counter = Counter()
    for syl, prof in syl_profiles.items():
        syl_classes[_classify_syllable_position(prof)] += 1

    n_syl_initial = syl_classes.get('initial_dominant', 0)
    n_syl_final = syl_classes.get('final_dominant', 0)
    n_syl_medial = syl_classes.get('medial_dominant', 0)
    n_syl_mixed = syl_classes.get('mixed', 0)
    print(f"     {len(syl_profiles)} syllables profiled")
    print(f"       initial_dominant: {n_syl_initial}")
    print(f"       final_dominant:   {n_syl_final}")
    print(f"       medial_dominant:  {n_syl_medial}")
    print(f"       mixed:            {n_syl_mixed}")

    # ─── Step 3: Compatibility (computed during violation check) ─────
    # (cosine similarity is computed per-pair in steps 4 & 5)

    # ─── Step 4: Check Phase 16 violations ──────────────────────────
    print("\n  6. Checking Phase 16 positional violations ...")
    violations = _check_violations(
        assignment, eva_to_triple, eva_profiles, syl_profiles,
    )
    n_total_pairs = 0
    triple_to_chars: Dict[str, List[str]] = defaultdict(list)
    for ch, tk in eva_to_triple.items():
        triple_to_chars[tk].append(ch)
    for tk in assignment:
        n_total_pairs += len(triple_to_chars.get(tk, []))
    violation_rate = (
        len(violations) / n_total_pairs if n_total_pairs > 0 else 0.0
    )

    print(f"     {len(violations)} violations in {n_total_pairs} (char, syllable) pairs")
    print(f"     violation rate: {violation_rate:.1%}")
    for v in violations[:10]:
        print(
            f"       {v.eva_char} ({v.eva_position_class}) "
            f"→ {v.assigned_syllable} ({v.syllable_position_class}) "
            f"compat={v.compatibility_score:.3f}"
        )

    # ─── Step 5: Corrective filter ──────────────────────────────────
    print("\n  7. Proposing corrections for violations ...")
    candidate_syllables = list(syl_profiles.keys())
    corrected_violations = _propose_corrections(
        violations, eva_profiles, syl_profiles, candidate_syllables,
    )
    n_corrections = sum(
        1 for v in corrected_violations
        if v.suggested_replacement and v.suggested_replacement != v.assigned_syllable
    )
    print(f"     {n_corrections} corrections proposed")
    for v in corrected_violations[:10]:
        if v.suggested_replacement and v.suggested_replacement != v.assigned_syllable:
            print(
                f"       {v.eva_char}: {v.assigned_syllable} → "
                f"{v.suggested_replacement} (score {v.suggestion_score:.3f})"
            )

    corrections_list = [
        asdict(v) for v in corrected_violations
        if v.suggested_replacement and v.suggested_replacement != v.assigned_syllable
    ]

    # ─── Step 6: Gallows analysis ───────────────────────────────────
    print("\n  8. Gallows character analysis ...")
    gallows_initial_rate, gallows_syllable_info = _gallows_analysis(
        corpus, eva_to_triple, assignment, syl_profiles,
    )
    print(f"     gallows initial rate: {gallows_initial_rate:.1%}")
    for gi in gallows_syllable_info:
        print(
            f"       {gi['char']:4s} → {gi['syllable']:6s} "
            f"({gi['position_class']})"
        )

    # ─── Verdict ────────────────────────────────────────────────────
    if len(violations) == 0:
        verdict = (
            "NO VIOLATIONS — Phase 16 assignments are fully compatible "
            "with EVA positional grammar."
        )
    elif violation_rate < 0.1:
        verdict = (
            f"LOW VIOLATION RATE ({violation_rate:.1%}) — "
            f"{len(violations)} minor positional mismatches found. "
            f"{n_corrections} corrections proposed."
        )
    elif violation_rate < 0.3:
        verdict = (
            f"MODERATE VIOLATION RATE ({violation_rate:.1%}) — "
            f"{len(violations)} positional mismatches. "
            f"{n_corrections} corrections proposed. "
            "EVA word grammar provides useful constraints."
        )
    else:
        verdict = (
            f"HIGH VIOLATION RATE ({violation_rate:.1%}) — "
            f"{len(violations)} positional mismatches. "
            "Phase 16 assignments substantially conflict with "
            "EVA positional grammar."
        )

    runtime = time.time() - t0

    print(f"\n  Verdict: {verdict}")
    print(f"  Runtime: {runtime:.1f}s")

    # ─── Build result ───────────────────────────────────────────────
    result = TokenGrammarResult(
        timestamp=time.strftime('%Y-%m-%dT%H:%M:%S'),
        n_eva_chars_profiled=len(eva_profiles),
        n_initial_only=n_initial_only,
        n_final_only=n_final_only,
        n_medial_only=n_medial_only,
        n_mixed=n_mixed,
        eva_profiles=[asdict(r) for r in eva_profile_records],
        n_syllables_profiled=len(syl_profiles),
        n_initial_dominant=n_syl_initial,
        n_final_dominant=n_syl_final,
        n_medial_dominant=n_syl_medial,
        n_violations=len(violations),
        violations=[asdict(v) for v in corrected_violations],
        violation_rate=round(violation_rate, 4),
        n_corrections_proposed=n_corrections,
        corrections=corrections_list,
        gallows_chars=GALLOWS_CHARS,
        gallows_initial_rate=round(gallows_initial_rate, 4),
        gallows_syllables=gallows_syllable_info,
        verdict=verdict,
        runtime_seconds=round(runtime, 2),
    )

    # ─── Save ───────────────────────────────────────────────────────
    out_path = os.path.join(rd, 'token_grammar.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)
    print(f"\n  Saved → {out_path}")
    print("=" * 70)
