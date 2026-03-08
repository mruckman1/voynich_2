"""
Phase 22.1 – First-Syllable Extraction (first-syl)
====================================================
Derives syllable-level values from word-level historical Tironian matches
by extracting the first CV syllable of each matched Latin word.

Hypothesis: in the Italian syllabic tachygraphic tradition, word-level
signs were repurposed as syllable signs. The syllabic value = the FIRST
CV SYLLABLE of the word that sign most commonly abbreviated.

Dependency chain:
    eva_stroke_compare.json (21.4) + tachygraphic_stroke.json (19.5)
    + cross_approach.json (19.8) + combined_refine.json (15.4)
    + modifier_integrate.json (16)
        → first_syllable_table.json (this step)
"""

import json
import re
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
# Latin syllabification
# ---------------------------------------------------------------------------

_LATIN_VOWELS = set('aeiouy')
_LATIN_CONSONANTS = set('bcdfghjklmnpqrstvwxz')


def _clean_latin_value(raw: str) -> str:
    """Clean a historical Latin value for syllabification.

    Strips parenthetical abbreviation markers like (imss), (c), (adh).
    Takes the first word from multi-word values.
    Strips numeric values and Tironian numeral notations.
    """
    if not raw:
        return ''

    # Strip leading abbreviation markers: "(imss) in millesimo..."
    # Pattern: value starts with (xxx) followed by the expansion
    m = re.match(r'^\([^)]+\)\s+(.+)$', raw)
    if m:
        raw = m.group(1)

    # Strip "N = X" notation like "l = al", "q = qua", "9 = ci"
    m = re.match(r'^[^\s=]+\s*=\s*(.+)$', raw)
    if m:
        raw = m.group(1)

    # Strip parenthetical expansions: "a(ger)" → "ager", "f(elix)" → "felix"
    raw = re.sub(r'\(([^)]+)\)', r'\1', raw)

    # Strip [sup:...] superscript markers: "zo[sup:ci]" → "zoci"
    raw = re.sub(r'\[sup:([^\]]+)\]', r'\1', raw)

    # Take first word only
    raw = raw.split()[0] if raw.strip() else ''

    # Strip non-alpha
    raw = re.sub(r'[^a-zA-Z]', '', raw)

    return raw.lower()


def _extract_first_cv(word: str) -> str:
    """Extract first CV syllable (strict: consonant cluster + one vowel).

    Examples:
        sub → su, codice → co, denarius → de, se → se,
        adhuc → a (vowel-initial), heredibus → he, o → o
    """
    if not word:
        return ''
    word = word.lower()

    parts: List[str] = []
    i = 0

    # Onset: collect initial consonants
    while i < len(word) and word[i] not in _LATIN_VOWELS:
        parts.append(word[i])
        i += 1

    # Nucleus: collect first vowel only
    if i < len(word) and word[i] in _LATIN_VOWELS:
        parts.append(word[i])
    elif not parts:
        return '?'

    return ''.join(parts)


def _extract_first_cvc(word: str) -> str:
    """Extract first CVC syllable (allow coda consonants before next vowel).

    Examples:
        sub → sub, codice → co, denarius → de,
        adhuc → ad, heredibus → he, ipsius → ip
    """
    if not word:
        return ''
    word = word.lower()

    parts: List[str] = []
    i = 0

    # Onset: collect initial consonants
    while i < len(word) and word[i] not in _LATIN_VOWELS:
        parts.append(word[i])
        i += 1

    # Nucleus: collect first vowel
    if i < len(word) and word[i] in _LATIN_VOWELS:
        parts.append(word[i])
        i += 1
    elif not parts:
        return '?'
    else:
        return ''.join(parts)

    # Coda: collect consonants until next vowel (maximal onset for next syl)
    # Latin maximal onset: leave at least one consonant for next syllable
    # if cluster between vowels. But for CVC extraction we take all codas.
    coda_start = i
    while i < len(word) and word[i] not in _LATIN_VOWELS:
        i += 1

    # If we reached end of word or next char is vowel with no consonants, done
    coda_len = i - coda_start
    if coda_len > 0:
        if i >= len(word):
            # End of word: take all remaining consonants
            parts.extend(word[coda_start:i])
        else:
            # Before next vowel: apply maximal onset — leave one consonant
            # for the next syllable's onset (if there's more than one)
            take = max(1, coda_len - 1) if coda_len > 1 else coda_len
            parts.extend(word[coda_start:coda_start + take])

    return ''.join(parts)


def _syllabify_latin_word(word: str) -> List[str]:
    """Full Latin syllabification (maximal onset principle).

    Returns list of syllables. Used for analysis, not directly for decoding.
    """
    if not word:
        return []
    word = word.lower()

    syllables: List[str] = []
    current: List[str] = []

    i = 0
    while i < len(word):
        ch = word[i]
        if ch in _LATIN_VOWELS:
            current.append(ch)
            # Look ahead: if consonant cluster followed by vowel,
            # split before last consonant of cluster (maximal onset)
            j = i + 1
            while j < len(word) and word[j] not in _LATIN_VOWELS:
                j += 1
            n_cons = j - (i + 1)
            if n_cons > 0 and j < len(word):
                # Leave at least 1 consonant for next onset
                take_coda = max(0, n_cons - 1)
                current.extend(word[i + 1:i + 1 + take_coda])
                syllables.append(''.join(current))
                current = []
                i = i + 1 + take_coda
            elif n_cons > 0 and j >= len(word):
                # End of word: all consonants are coda
                current.extend(word[i + 1:j])
                syllables.append(''.join(current))
                current = []
                i = j
            else:
                # Next char is vowel or end
                syllables.append(''.join(current))
                current = []
                i += 1
        else:
            current.append(ch)
            i += 1

    if current:
        if syllables:
            syllables[-1] += ''.join(current)
        else:
            syllables.append(''.join(current))

    return syllables


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class FirstSyllableEntry:
    eva_char: str
    historical_word: str
    historical_source: str
    first_syllable_cv: str        # Mode A
    first_syllable_cvc: str       # Mode B
    match_level: str              # exact/near/partial/none
    match_score: float
    sign_family: str              # from Phase 19.5
    is_modifier: bool             # from Phase 16
    confidence: str               # high/medium/low
    all_candidates: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class FirstSyllableResult:
    timestamp: str
    n_eva_chars: int
    n_with_historical: int
    n_with_first_syl: int
    n_modifiers_skipped: int
    mode_a_table: List[Dict[str, Any]]
    mode_b_table: List[Dict[str, Any]]
    family_consonant_agreement: float
    family_vowel_agreement: float
    family_details: List[Dict[str, Any]]
    anchor_compatible: int
    anchor_total: int
    anchor_details: List[Dict[str, Any]]
    n_from_phase15_fallback: int
    n_from_family_propagation: int
    n_unmatched: int


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_first_syllable() -> Dict[str, Any]:
    """Extract first syllables from historical word-level matches."""
    t0 = time.time()
    rdir = _results_dir()

    # --- Load inputs ---
    stroke_data = _load_json(str(rdir / "eva_stroke_compare.json")) or {}
    per_char = stroke_data.get('per_char_matches', [])

    family_data = _load_json(str(rdir / "tachygraphic_stroke.json")) or {}
    sign_families = family_data.get('sign_families', [])

    cross_data = _load_json(str(rdir / "cross_approach.json")) or {}
    per_word = cross_data.get('per_word_results', [])

    refine_data = _load_json(str(rdir / "combined_refine.json")) or {}
    phase15_assignment = refine_data.get('best_assignment', {})

    mod_data = _load_json(str(rdir / "modifier_integrate.json")) or {}
    modifier_chars = set(mod_data.get('modifier_chars', []))

    # --- Build family lookup ---
    char_to_family: Dict[str, str] = {}
    for fam in sign_families:
        glyph_class = fam.get('glyph_class', '')
        for member in fam.get('members', []):
            char_to_family[member] = glyph_class

    # --- Build triple lookup for Phase 15 fallback ---
    triple_to_syl: Dict[str, str] = phase15_assignment

    # EVA char → triple key (from EVA_VISUAL_COMPONENTS)
    eva_to_triple: Dict[str, str] = {}
    for eva_ch, comp in EVA_VISUAL_COMPONENTS.items():
        triple_key = f"{comp['first_stroke']},{comp['last_stroke']},{comp['glyph_class']}"
        eva_to_triple[eva_ch] = triple_key

    # --- Process each EVA char ---
    entries: List[FirstSyllableEntry] = []
    n_modifiers_skipped = 0

    for char_info in per_char:
        eva_ch = char_info.get('eva_char', '')
        glyph_class = char_info.get('eva_glyph_class', '')
        candidates = char_info.get('top_candidates', [])
        best_level = char_info.get('best_match_level', 'none')
        best_score = char_info.get('best_match_score', 0.0)
        best_source = char_info.get('best_match_source', '')

        is_mod = eva_ch in modifier_chars
        family = char_to_family.get(eva_ch, glyph_class)

        # Find best Latin word to extract syllable from
        best_word = ''
        best_word_source = best_source
        best_word_level = best_level
        best_word_score = best_score
        all_cands: List[Dict[str, str]] = []

        for cand in candidates:
            raw_val = cand.get('latin_value', '')
            cleaned = _clean_latin_value(raw_val)
            level = cand.get('match_level', 'none')
            score = cand.get('best_score', 0.0)
            sources = cand.get('sources', [])

            if cleaned:
                all_cands.append({
                    'raw': raw_val,
                    'cleaned': cleaned,
                    'first_cv': _extract_first_cv(cleaned),
                    'first_cvc': _extract_first_cvc(cleaned),
                    'level': level,
                    'score': str(score),
                    'source': sources[0] if sources else '',
                })

            if cleaned and not best_word:
                best_word = cleaned
                best_word_source = sources[0] if sources else best_source
                best_word_level = level
                best_word_score = score

        # Extract first syllables
        first_cv = _extract_first_cv(best_word) if best_word else ''
        first_cvc = _extract_first_cvc(best_word) if best_word else ''

        # Confidence
        if best_word_level == 'exact' and best_word_score >= 0.9:
            confidence = 'high'
        elif best_word_level in ('exact', 'near'):
            confidence = 'medium'
        elif best_word_level == 'partial':
            confidence = 'low'
        else:
            confidence = 'none'

        if is_mod:
            n_modifiers_skipped += 1

        entries.append(FirstSyllableEntry(
            eva_char=eva_ch,
            historical_word=best_word,
            historical_source=best_word_source,
            first_syllable_cv=first_cv,
            first_syllable_cvc=first_cvc,
            match_level=best_word_level,
            match_score=best_word_score,
            sign_family=family,
            is_modifier=is_mod,
            confidence=confidence,
            all_candidates=all_cands,
        ))

    # --- Family consistency check ---
    family_groups: Dict[str, List[FirstSyllableEntry]] = {}
    for entry in entries:
        if entry.is_modifier or not entry.first_syllable_cv:
            continue
        family_groups.setdefault(entry.sign_family, []).append(entry)

    family_details: List[Dict[str, Any]] = []
    total_consonant_agree = 0
    total_vowel_agree = 0
    n_families_checked = 0

    for fam_name in sorted(family_groups.keys()):
        members = family_groups[fam_name]
        if len(members) < 2:
            continue

        # Extract onsets and nuclei from Mode A syllables
        onsets: List[str] = []
        nuclei: List[str] = []
        for m in members:
            syl = m.first_syllable_cv
            if not syl:
                continue
            # Split into onset + nucleus
            onset_chars = []
            nucleus_char = ''
            for ch in syl:
                if ch in _LATIN_VOWELS and not nucleus_char:
                    nucleus_char = ch
                elif not nucleus_char:
                    onset_chars.append(ch)
            onsets.append(''.join(onset_chars))
            nuclei.append(nucleus_char)

        # Agreement: do members share the same onset consonant?
        if onsets:
            most_common_onset = Counter(onsets).most_common(1)[0]
            consonant_agree = most_common_onset[1] / len(onsets)
        else:
            consonant_agree = 0.0

        if nuclei:
            most_common_nucleus = Counter(nuclei).most_common(1)[0]
            vowel_agree = most_common_nucleus[1] / len(nuclei)
        else:
            vowel_agree = 0.0

        total_consonant_agree += consonant_agree
        total_vowel_agree += vowel_agree
        n_families_checked += 1

        family_details.append({
            'family': fam_name,
            'n_members': len(members),
            'onsets': dict(Counter(onsets)),
            'nuclei': dict(Counter(nuclei)),
            'consonant_agreement': round(consonant_agree, 3),
            'vowel_agreement': round(vowel_agree, 3),
            'member_syllables': [
                {'eva': m.eva_char, 'cv': m.first_syllable_cv, 'cvc': m.first_syllable_cvc}
                for m in members
            ],
        })

    avg_consonant = total_consonant_agree / max(n_families_checked, 1)
    avg_vowel = total_vowel_agree / max(n_families_checked, 1)

    # --- Cross-reference with anchors ---
    # Anchors from Phase 19.8 that had matches
    anchor_details: List[Dict[str, Any]] = []
    n_compatible = 0
    n_anchor_total = 0

    for pw in per_word:
        latin_word = pw.get('latin_word', '')
        best_decoded = pw.get('best_decoded', '')
        has_match = pw.get('edit2_match', False) or pw.get('exact_match', False)
        if not has_match or not best_decoded:
            continue

        n_anchor_total += 1
        # Check if our first-syllable assignments are compatible
        # "compatible" = decoded from Phase 19.8 starts with a syllable we assigned
        compatible = False
        note = ''

        # Find what our table would produce for the first EVA char of
        # one of the matching tokens
        tokens = pw.get('voynich_tokens', [])
        if tokens:
            # Just check if the anchor word's first syllable appears in our table
            anchor_first_cv = _extract_first_cv(latin_word)
            anchor_first_cvc = _extract_first_cvc(latin_word)

            for entry in entries:
                if entry.first_syllable_cv == anchor_first_cv and not entry.is_modifier:
                    compatible = True
                    note = f'{entry.eva_char}→{entry.first_syllable_cv} matches anchor {latin_word}'
                    break

        if compatible:
            n_compatible += 1

        anchor_details.append({
            'latin_word': latin_word,
            'best_decoded': best_decoded,
            'compatible': compatible,
            'note': note,
        })

    # --- Fill unmatched characters ---
    n_phase15_fallback = 0
    n_family_propagation = 0
    n_unmatched = 0

    for entry in entries:
        if entry.first_syllable_cv or entry.is_modifier:
            continue

        # Try Phase 15 fallback
        triple_key = eva_to_triple.get(entry.eva_char, '')
        if triple_key and triple_key in triple_to_syl:
            p15_syl = triple_to_syl[triple_key]
            entry.first_syllable_cv = p15_syl
            entry.first_syllable_cvc = p15_syl
            entry.confidence = 'fallback_p15'
            entry.historical_source = 'phase15'
            n_phase15_fallback += 1
            continue

        # Try family propagation: if another non-modifier member of same
        # family has a high-confidence assignment, propagate its consonant
        family = entry.sign_family
        fam_members = family_groups.get(family, [])
        best_fam_syl = ''
        for fm in fam_members:
            if fm.eva_char != entry.eva_char and fm.first_syllable_cv and fm.confidence in ('high', 'medium'):
                best_fam_syl = fm.first_syllable_cv
                break

        if best_fam_syl:
            entry.first_syllable_cv = best_fam_syl
            entry.first_syllable_cvc = best_fam_syl
            entry.confidence = 'fallback_family'
            entry.historical_source = 'family_propagation'
            n_family_propagation += 1
        else:
            n_unmatched += 1

    # --- Build output tables ---
    n_with_historical = sum(1 for e in entries if e.historical_word and e.historical_source != 'phase15')
    n_with_first_syl = sum(1 for e in entries if e.first_syllable_cv)

    mode_a = [_convert(asdict(e)) for e in entries]
    # Mode B is same entries but primary field is first_syllable_cvc
    mode_b = [_convert(asdict(e)) for e in entries]

    result = FirstSyllableResult(
        timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
        n_eva_chars=len(entries),
        n_with_historical=n_with_historical,
        n_with_first_syl=n_with_first_syl,
        n_modifiers_skipped=n_modifiers_skipped,
        mode_a_table=mode_a,
        mode_b_table=mode_b,
        family_consonant_agreement=round(avg_consonant, 4),
        family_vowel_agreement=round(avg_vowel, 4),
        family_details=family_details,
        anchor_compatible=n_compatible,
        anchor_total=n_anchor_total,
        anchor_details=anchor_details,
        n_from_phase15_fallback=n_phase15_fallback,
        n_from_family_propagation=n_family_propagation,
        n_unmatched=n_unmatched,
    )

    out_path = rdir / "first_syllable_table.json"
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t0
    print(f"first-syl: {len(entries)} EVA chars, {n_with_first_syl} with first-syllable, "
          f"{n_modifiers_skipped} modifiers, family_C_agree={avg_consonant:.1%}, "
          f"anchors={n_compatible}/{n_anchor_total} ({elapsed:.1f}s)")

    return _convert(asdict(result))
