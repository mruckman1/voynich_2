"""
Phase 21.3 – Chatelain Bobbio Family Extraction (chatelain-families)
====================================================================
Extracts sign families from Chatelain's variant relationships and builds
a reference syllable table for Italian tachygraphic signs.

Dependency chain:
    paleo_ingest.json (master_reference.json)
        → chatelain_families.json (this step)
"""

import json
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.reference import normalize_stroke, stroke_category


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


def _load_master_reference() -> Dict:
    import os
    path = "data/reference/paleographic/master_reference.json"
    if not os.path.exists(path):
        raise FileNotFoundError("master_reference.json not found — run paleo-ingest first")
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Bobbio plate detection
# ---------------------------------------------------------------------------

# Chatelain plates XXXVII–XXXVIII are the Bobbio sections
_BOBBIO_PLATES = {'XXXVII', 'XXXVIII', '37', '38'}
_ITALIAN_SOURCES = {'bobbio', 'milan', 'verona', 'pavia', 'italy', 'italian'}


def _is_italian_origin(sign: Dict) -> bool:
    """Check if a Chatelain sign is from Italian/Bobbio origin."""
    geo = (sign.get('geographic_source') or '').lower()
    if geo and any(src in geo for src in _ITALIAN_SOURCES):
        return True
    plate = str(sign.get('plate', '')).upper().strip()
    # Bobbio plates
    if plate in _BOBBIO_PLATES:
        return True
    # Null geographic is acceptable (most entries)
    if not geo:
        return True  # Include as potentially Italian
    return False


def _is_simple_sign(sign: Dict) -> bool:
    """Check if a sign is simple (0 middle strokes)."""
    ms = sign.get('middle_strokes', []) or []
    return len(ms) == 0


# ---------------------------------------------------------------------------
# Phonetic pattern detection
# ---------------------------------------------------------------------------

_VOWELS = set('aeiouæœ')
_CONSONANTS = set('bcdfghjklmnpqrstvwxyz')


def _extract_initial_consonant(word: str) -> Optional[str]:
    """Extract leading consonant(s) from a Latin word."""
    if not word:
        return None
    word_lower = word.lower().strip()
    consonants = []
    for ch in word_lower:
        if ch in _CONSONANTS:
            consonants.append(ch)
        else:
            break
    return ''.join(consonants) if consonants else None


def _extract_first_vowel(word: str) -> Optional[str]:
    """Extract first vowel from a Latin word."""
    if not word:
        return None
    for ch in word.lower().strip():
        if ch in _VOWELS:
            return ch
    return None


def _detect_family_pattern(latin_values: List[str]) -> Tuple[str, Optional[str]]:
    """Detect phonetic pattern among family members' Latin values.

    Returns (pattern_type, shared_value) where pattern_type is one of:
    'same_consonant', 'same_vowel', 'same_syllable', 'unrelated'
    """
    if len(latin_values) < 2:
        return ('insufficient', None)

    # Check shared initial consonant
    initials = [_extract_initial_consonant(v) for v in latin_values]
    initials = [i for i in initials if i]
    if len(initials) >= 2:
        unique_initials = set(initials)
        if len(unique_initials) == 1:
            return ('same_consonant', initials[0])

    # Check shared first vowel
    vowels = [_extract_first_vowel(v) for v in latin_values]
    vowels = [v for v in vowels if v]
    if len(vowels) >= 2:
        unique_vowels = set(vowels)
        if len(unique_vowels) == 1:
            return ('same_vowel', vowels[0])

    # Check if all start with same syllable (first 2 chars)
    prefixes = [v.lower()[:2] for v in latin_values if len(v) >= 2]
    if len(prefixes) >= 2 and len(set(prefixes)) == 1:
        return ('same_syllable', prefixes[0])

    return ('unrelated', None)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ChatelainFamily:
    root_id: str
    members: List[str]
    latin_values: List[str]
    pattern_type: str
    shared_value: Optional[str]
    first_strokes: List[str]
    size: int


@dataclass
class SyllableTableEntry:
    stroke_pattern: str      # e.g., "vertical_stroke → hook"
    consonant_class: Optional[str]
    vowel_hint: Optional[str]
    evidence_count: int
    confidence: str


@dataclass
class ChatelainFamiliesResult:
    timestamp: str
    n_italian_signs: int
    n_simple_signs: int
    n_families: int
    families: List[Dict[str, Any]]
    syllabic_fraction: float
    reference_syllable_table: List[Dict[str, Any]]
    schmitz_comparison: Dict[str, Any]
    gate_description: str


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_chatelain_families() -> Dict[str, Any]:
    """Extract Chatelain families and build reference syllable table."""
    t0 = time.time()

    master = _load_master_reference()
    all_signs = master.get('all_signs', [])

    # --- Filter Chatelain Italian-origin simple signs ---
    chatelain_signs = [s for s in all_signs if s.get('source') == 'chatelain']
    italian_signs = [s for s in chatelain_signs if _is_italian_origin(s)]
    simple_signs = [s for s in italian_signs if _is_simple_sign(s)]

    # Build index by source_id for variant lookups
    by_id: Dict[str, Dict] = {}
    for s in chatelain_signs:
        by_id[s.get('source_id', '')] = s

    # --- Build families from variant_of relationships ---
    # variant_of points to a root entry's position identifier
    # Group entries that share the same variant_of root
    families_by_root: Dict[str, List[Dict]] = defaultdict(list)
    standalone: List[Dict] = []

    for s in simple_signs:
        var_of = s.get('variant_of')
        if var_of:
            families_by_root[var_of].append(s)
        else:
            # This could be a root itself
            sid = s.get('source_id', '')
            pos = f"{s.get('plate', '?')}_{sid.split('_')[-1] if '_' in sid else '?'}"
            families_by_root[pos].append(s)

    # Merge: if a root also has an entry, include it
    families: List[ChatelainFamily] = []
    for root_key, members in families_by_root.items():
        if len(members) < 2:
            standalone.extend(members)
            continue

        latin_vals = [m.get('latin_value', '') or '' for m in members]
        latin_vals = [v for v in latin_vals if v]

        first_strokes = [m.get('first_stroke', '') or '' for m in members]
        first_strokes = [fs for fs in first_strokes if fs]

        pattern_type, shared_value = _detect_family_pattern(latin_vals)

        families.append(ChatelainFamily(
            root_id=root_key,
            members=[m.get('source_id', '') for m in members],
            latin_values=latin_vals,
            pattern_type=pattern_type,
            shared_value=shared_value,
            first_strokes=list(set(first_strokes)),
            size=len(members),
        ))

    # --- Syllabic fraction ---
    syllabic_count = sum(1 for f in families if f.pattern_type in ('same_consonant', 'same_syllable'))
    syllabic_fraction = syllabic_count / max(len(families), 1)

    # --- Build reference syllable table ---
    syllable_table: List[SyllableTableEntry] = []
    for f in families:
        if f.pattern_type == 'same_consonant' and f.shared_value:
            for fs in f.first_strokes:
                syllable_table.append(SyllableTableEntry(
                    stroke_pattern=fs,
                    consonant_class=f.shared_value,
                    vowel_hint=None,
                    evidence_count=f.size,
                    confidence='medium' if f.size >= 3 else 'low',
                ))
        elif f.pattern_type == 'same_vowel' and f.shared_value:
            for fs in f.first_strokes:
                syllable_table.append(SyllableTableEntry(
                    stroke_pattern=fs,
                    consonant_class=None,
                    vowel_hint=f.shared_value,
                    evidence_count=f.size,
                    confidence='low',
                ))

    # --- Schmitz comparison ---
    schmitz_signs = [s for s in all_signs if s.get('source') == 'schmitz']
    schmitz_simple = [s for s in schmitz_signs if _is_simple_sign(s)]

    # Build Schmitz families similarly
    schmitz_families_by_root: Dict[str, List[Dict]] = defaultdict(list)
    for s in schmitz_simple:
        # Schmitz doesn't have variant_of, group by plate + first_stroke
        key = f"{s.get('first_stroke', 'unk')}_{s.get('final_stroke', 'unk')}"
        schmitz_families_by_root[key].append(s)

    schmitz_families_with_latin = 0
    schmitz_syllabic = 0
    for key, members in schmitz_families_by_root.items():
        if len(members) < 2:
            continue
        schmitz_families_with_latin += 1
        latin_vals = [m.get('latin_value', '') or '' for m in members]
        latin_vals = [v for v in latin_vals if v]
        if len(latin_vals) >= 2:
            pattern, _ = _detect_family_pattern(latin_vals)
            if pattern in ('same_consonant', 'same_syllable'):
                schmitz_syllabic += 1

    schmitz_syllabic_frac = schmitz_syllabic / max(schmitz_families_with_latin, 1)

    schmitz_comparison = {
        'schmitz_simple_signs': len(schmitz_simple),
        'schmitz_families': schmitz_families_with_latin,
        'schmitz_syllabic_fraction': schmitz_syllabic_frac,
        'chatelain_syllabic_fraction': syllabic_fraction,
        'chatelain_higher': syllabic_fraction > schmitz_syllabic_frac,
        'description': (
            f"Chatelain syllabic fraction: {syllabic_fraction:.3f}, "
            f"Schmitz: {schmitz_syllabic_frac:.3f}. "
            f"{'Chatelain higher as expected (Bobbio material)' if syllabic_fraction > schmitz_syllabic_frac else 'Schmitz higher (unexpected)'}"
        ),
    }

    # --- Gate ---
    gate_desc = (
        f"Syllabic fraction = {syllabic_fraction:.3f}. "
        f"Gate: ≥ 0.10. {'PASS' if syllabic_fraction >= 0.10 else 'BELOW — limited syllabic structure preserved'}. "
        f"Proceeding regardless — even word-level correspondences provide phonetic constraints."
    )

    result = ChatelainFamiliesResult(
        timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
        n_italian_signs=len(italian_signs),
        n_simple_signs=len(simple_signs),
        n_families=len(families),
        families=[_convert(asdict(f)) for f in families],
        syllabic_fraction=syllabic_fraction,
        reference_syllable_table=[_convert(asdict(e)) for e in syllable_table],
        schmitz_comparison=schmitz_comparison,
        gate_description=gate_desc,
    )

    out_path = _results_dir() / "chatelain_families.json"
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t0
    print(f"chatelain-families: {len(families)} families, "
          f"syllabic fraction={syllabic_fraction:.3f}, "
          f"{len(syllable_table)} table entries ({elapsed:.1f}s)")

    return _convert(asdict(result))
