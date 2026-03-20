"""
Phase 57, Step 1-2: Coda Marker Table + CVC Decode Function
============================================================
Maps EVA modifier characters to Costamagna coda consonants and provides
a CVC-aware decode function.

Costamagna's 5 coda rules (from costamagna_1953_catalog.json):
  m = two dots added to the base sign
  n = one dot added to the base sign
  r = descender stroke or dot below
  s = curve appended to the sign
  t = crossbar through or above the sign

Phase 16's 15 modifier characters (from modifier_integrate.json):
  Each has a last_stroke feature from EVA_VISUAL_COMPONENTS.

The mapping: last_stroke -> coda consonant.

Dependency chain:
    results/modifier_integrate.json
    EVA_VISUAL_COMPONENTS (reference.py)
    results/combined_refine.json
        -> results/coda_table.json
"""

import json
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import build_eva_to_triple_lookup, tokenize_eva_chars
from voynich.core.reference import EVA_VISUAL_COMPONENTS


# ---------------------------------------------------------------------------
# JSON helpers (same pattern as other phase modules)
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
class CodaTable:
    """Maps stroke types to coda consonants."""
    variant: str                          # 'primary' or 'alternate'
    stroke_to_coda: Dict[str, str]        # last_stroke -> coda consonant
    eva_modifiers: Dict[str, str]         # EVA char -> last_stroke
    modifier_confidence: Dict[str, str]   # EVA char -> MODIFIER/AMBIGUOUS
    n_modifier: int
    n_ambiguous_as_coda: int
    ambiguous_chars: List[str]            # AMBIGUOUS chars that can act as coda


@dataclass
class CvcDecodeResult:
    """Decode result for a single token."""
    token: str
    eva_chars: List[str]
    char_roles: List[str]        # SYLLABIC / CODA_MARKER per char
    decoded_cv: str              # old CV-only decode (strip modifiers)
    decoded_cvc: str             # new CVC decode


@dataclass
class CodaTableResult:
    """Full Step 57.1 output."""
    phase: str = "57"
    step: str = "57.1"
    experiment: str = "coda_table"
    primary: Optional[CodaTable] = None
    alternate: Optional[CodaTable] = None
    modifier_char_details: List[Dict[str, Any]] = field(default_factory=list)
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Stroke-to-coda mappings
# ---------------------------------------------------------------------------

# Primary mapping derived from Phase 56 Q6
STROKE_TO_CODA_PRIMARY = {
    'hook':       'n',    # Costamagna: "one dot" -> nasal
    'descender':  'r',    # Costamagna: "vertical descender"
    'sigmoid':    's',    # Costamagna: "curve"
    'vertical':   't',    # Costamagna: "crossbar"
    'connector':  'l',    # Not in Costamagna's 5 basic codas -- tentative
}

# Alternate: vertical -> m (Costamagna "two dots" for m vs "crossbar" for t)
STROKE_TO_CODA_ALTERNATE = {
    'hook':       'n',
    'descender':  'r',
    'sigmoid':    's',
    'vertical':   'm',    # ALTERNATIVE: "two dots" instead of "crossbar"
    'connector':  'l',
}

# Simple gallows (always SYLLABIC regardless of modifier classification)
SIMPLE_GALLOWS = {'k', 't', 'p', 'f'}


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def build_coda_table(variant: str = 'primary') -> CodaTable:
    """Build a coda marker table from Phase 16 modifier classifications.

    Parameters
    ----------
    variant : str
        'primary' (vertical->t) or 'alternate' (vertical->m)

    Returns
    -------
    CodaTable
    """
    rd = str(_results_dir())
    mod_data = _safe_load(os.path.join(rd, 'modifier_integrate.json'))
    if not mod_data:
        raise FileNotFoundError("results/modifier_integrate.json not found")

    stroke_map = (STROKE_TO_CODA_PRIMARY if variant == 'primary'
                  else STROKE_TO_CODA_ALTERNATE)

    # Build EVA modifier -> last_stroke lookup from EVA_VISUAL_COMPONENTS
    eva_modifiers: Dict[str, str] = {}
    modifier_confidence: Dict[str, str] = {}

    # Process MODIFIER chars (high confidence)
    for cls in mod_data.get('classifications', []):
        eva_char = cls['eva_char']
        classification = cls['final_classification']

        # Skip simple gallows -- they are always syllabic
        if eva_char in SIMPLE_GALLOWS:
            continue

        comp = EVA_VISUAL_COMPONENTS.get(eva_char)
        if comp is None:
            continue

        last_stroke = comp['last_stroke']

        if classification == 'modifier':
            eva_modifiers[eva_char] = last_stroke
            modifier_confidence[eva_char] = 'MODIFIER'
        elif classification == 'ambiguous':
            # Track ambiguous chars -- they'll be context-dependent in decode
            eva_modifiers[eva_char] = last_stroke
            modifier_confidence[eva_char] = 'AMBIGUOUS'

    # Count
    n_mod = sum(1 for v in modifier_confidence.values() if v == 'MODIFIER')
    ambig_chars = [ch for ch, v in modifier_confidence.items()
                   if v == 'AMBIGUOUS' and ch not in SIMPLE_GALLOWS]

    return CodaTable(
        variant=variant,
        stroke_to_coda=dict(stroke_map),
        eva_modifiers=eva_modifiers,
        modifier_confidence=modifier_confidence,
        n_modifier=n_mod,
        n_ambiguous_as_coda=len(ambig_chars),
        ambiguous_chars=ambig_chars,
    )


def get_coda(eva_char: str, coda_table: CodaTable) -> Optional[str]:
    """Return the coda consonant for an EVA modifier character.

    Returns None if the character is not in the coda table.
    """
    last_stroke = coda_table.eva_modifiers.get(eva_char)
    if last_stroke is None:
        return None
    return coda_table.stroke_to_coda.get(last_stroke)


def classify_token_chars(
    eva_chars: List[str],
    coda_table: CodaTable,
) -> List[Tuple[str, str]]:
    """Classify each EVA character in a token as SYLLABIC or CODA_MARKER.

    Rules:
    1. The first character of a token is always SYLLABIC (words don't start
       with a coda).
    2. Simple gallows (k, t, p, f) are always SYLLABIC.
    3. Characters classified as MODIFIER by Phase 16 -> CODA_MARKER.
    4. AMBIGUOUS characters -> CODA_MARKER if they follow a SYLLABIC char;
       SYLLABIC otherwise (conservative default).
    5. All other characters -> SYLLABIC.

    Returns list of (role, eva_char) tuples.
    """
    classified: List[Tuple[str, str]] = []

    for idx, char in enumerate(eva_chars):
        # Rule 1: first char is always syllabic
        if idx == 0:
            classified.append(('SYLLABIC', char))
            continue

        # Rule 2: simple gallows are always syllabic
        if char in SIMPLE_GALLOWS:
            classified.append(('SYLLABIC', char))
            continue

        conf = coda_table.modifier_confidence.get(char)

        # Rule 3: MODIFIER -> CODA_MARKER
        if conf == 'MODIFIER':
            classified.append(('CODA_MARKER', char))
            continue

        # Rule 4: AMBIGUOUS -> CODA_MARKER only if the char has a valid
        # coda stroke AND follows a SYLLABIC char.  Otherwise SYLLABIC.
        if conf == 'AMBIGUOUS':
            last_stroke = coda_table.eva_modifiers.get(char)
            has_valid_coda = (last_stroke is not None
                              and last_stroke in coda_table.stroke_to_coda)
            if has_valid_coda and classified and classified[-1][0] == 'SYLLABIC':
                classified.append(('CODA_MARKER', char))
            else:
                classified.append(('SYLLABIC', char))
            continue

        # Rule 5: everything else -> SYLLABIC
        classified.append(('SYLLABIC', char))

    return classified


def decode_token_cvc(
    token: str,
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    coda_table: CodaTable,
) -> CvcDecodeResult:
    """Decode an EVA token using CVC coda rules.

    Algorithm:
    1. Tokenize into EVA characters.
    2. Classify each as SYLLABIC or CODA_MARKER.
    3. SYLLABIC chars -> look up CV syllable from assignment table.
    4. CODA_MARKER chars -> append coda consonant to preceding CV syllable.
    5. Concatenate result.

    Also computes the old CV-only decode (strip modifiers) for comparison.
    """
    eva_chars = tokenize_eva_chars(token)
    if not eva_chars:
        return CvcDecodeResult(
            token=token, eva_chars=[], char_roles=[],
            decoded_cv='', decoded_cvc='',
        )

    classified = classify_token_chars(eva_chars, coda_table)
    roles = [role for role, _ in classified]

    # Build CVC output
    output_parts: List[Tuple[str, str]] = []  # (type, value)
    for role, char in classified:
        if role == 'SYLLABIC':
            triple = eva_to_triple.get(char)
            syl = assignment.get(triple, '?') if triple else '?'
            output_parts.append(('CV', syl))
        elif role == 'CODA_MARKER':
            coda = get_coda(char, coda_table)
            if coda and output_parts and output_parts[-1][0] in ('CV', 'CVC'):
                # Append coda to preceding syllable
                prev_type, prev_val = output_parts[-1]
                output_parts[-1] = ('CVC', prev_val + coda)
            elif coda:
                # Orphaned coda (no preceding CV) -- standalone consonant
                output_parts.append(('ORPHAN', coda))

    decoded_cvc = ''.join(val for _, val in output_parts)

    # CV-only decode (strip modifiers)
    cv_parts = []
    for role, char in classified:
        if role == 'SYLLABIC':
            triple = eva_to_triple.get(char)
            syl = assignment.get(triple, '?') if triple else '?'
            cv_parts.append(syl)
    decoded_cv = ''.join(cv_parts)

    return CvcDecodeResult(
        token=token,
        eva_chars=eva_chars,
        char_roles=roles,
        decoded_cv=decoded_cv,
        decoded_cvc=decoded_cvc,
    )


def decode_corpus_cvc(
    tokens: List[str],
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    coda_table: CodaTable,
) -> List[str]:
    """Decode a list of EVA tokens using CVC rules.

    Returns list of decoded strings (one per token).
    """
    return [
        decode_token_cvc(tok, assignment, eva_to_triple, coda_table).decoded_cvc
        for tok in tokens
    ]


def decode_corpus_cv_strip(
    tokens: List[str],
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    coda_table: CodaTable,
) -> List[str]:
    """Decode tokens using CV-only (strip modifiers).

    Returns list of decoded strings.
    """
    return [
        decode_token_cvc(tok, assignment, eva_to_triple, coda_table).decoded_cv
        for tok in tokens
    ]


# ---------------------------------------------------------------------------
# CLI entry point: run_coda_table  (Step 57.1)
# ---------------------------------------------------------------------------

def run_coda_table():
    """Build and display both coda table variants. Save to results/coda_table.json."""
    t0 = time.time()
    print("=" * 70)
    print("Phase 57, Step 1: Coda Marker Table")
    print("=" * 70)

    rd = str(_results_dir())

    # Build both variants
    primary = build_coda_table('primary')
    alternate = build_coda_table('alternate')

    # Build modifier details for display
    details = []
    for char, last_stroke in sorted(primary.eva_modifiers.items()):
        conf = primary.modifier_confidence.get(char, '?')
        coda_p = primary.stroke_to_coda.get(last_stroke, '?')
        coda_a = alternate.stroke_to_coda.get(last_stroke, '?')
        details.append({
            'eva_char': char,
            'last_stroke': last_stroke,
            'confidence': conf,
            'coda_primary': coda_p,
            'coda_alternate': coda_a,
        })

    result = CodaTableResult(
        primary=primary,
        alternate=alternate,
        modifier_char_details=details,
        runtime_seconds=time.time() - t0,
    )

    # Print summary
    print(f"\nStroke-to-Coda Mapping (primary: vertical->t):")
    print(f"  {'Stroke':<14} {'Coda':<6} {'Count':<6}")
    print(f"  {'-'*14} {'-'*6} {'-'*6}")
    stroke_counts = Counter(primary.eva_modifiers.values())
    for stroke, coda in sorted(primary.stroke_to_coda.items()):
        cnt = stroke_counts.get(stroke, 0)
        print(f"  {stroke:<14} {coda:<6} {cnt:<6}")

    print(f"\nModifier Characters ({primary.n_modifier} MODIFIER + "
          f"{primary.n_ambiguous_as_coda} AMBIGUOUS):")
    print(f"  {'EVA char':<10} {'Last stroke':<14} {'Conf':<12} "
          f"{'Coda(p)':<8} {'Coda(a)':<8}")
    print(f"  {'-'*10} {'-'*14} {'-'*12} {'-'*8} {'-'*8}")
    for d in details:
        print(f"  {d['eva_char']:<10} {d['last_stroke']:<14} "
              f"{d['confidence']:<12} {d['coda_primary']:<8} "
              f"{d['coda_alternate']:<8}")

    print(f"\nAlternate differs on: vertical -> m (instead of t)")
    print(f"  Affected chars: "
          f"{[d['eva_char'] for d in details if d['last_stroke'] == 'vertical']}")

    # Quick decode demo on "daiin"
    assignment_data = _safe_load(os.path.join(rd, 'combined_refine.json'))
    assignment = assignment_data.get('best_assignment', {})
    eva_to_triple = build_eva_to_triple_lookup()

    demo_tokens = ['daiin', 'chedy', 'shedy', 'qokeedy', 'dain', 'ol']
    print(f"\nDecode Demo (primary variant):")
    print(f"  {'Token':<12} {'CV decode':<12} {'CVC decode':<12} {'Roles'}")
    print(f"  {'-'*12} {'-'*12} {'-'*12} {'-'*30}")
    for tok in demo_tokens:
        r = decode_token_cvc(tok, assignment, eva_to_triple, primary)
        roles_str = ','.join(f"{ch}={role[0]}" for (role, ch)
                            in zip(r.char_roles, r.eva_chars))
        print(f"  {tok:<12} {r.decoded_cv:<12} {r.decoded_cvc:<12} {roles_str}")

    # Save
    path = _save_json(rd, 'coda_table.json', result)
    print(f"\nSaved: {path}")
    print(f"Step 57.1 completed in {time.time() - t0:.1f}s")
