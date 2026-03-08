"""
Phase 22.2 – Fontana Phonetic Mapping (fontana-phon)
=====================================================
Maps Fontana cipher key (BSB + BNF) onto Voynich EVA characters via
Phase 19.5/21.2 structural correspondences.

Key insight: Fontana is ALPHABETIC (one sign = one letter). If the Voynich
uses the same construction logic but is SYLLABIC (one sign = one CV syllable):
- The CONSONANT of a Voynich family = the consonant from the matched Fontana family
- The VOWEL within a Voynich family = follows the same directional convention
  as Fontana's vowels (tick_up=a, tick_right=e, tick_down=i, tick_left=o, tick_northeast=u)

Dependency chain:
    Fontana BSB/BNF JSONs + fontana_families.json (21.2)
    + tachygraphic_stroke.json (19.5) + eva_stroke_compare.json (21.4)
        → fontana_phonetic.json (this step)
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
# Fontana key consolidation
# ---------------------------------------------------------------------------

# Vowel convention from Fontana circle family
FONTANA_VOWEL_MAP = {
    'tick_up': 'a',
    'tick_right': 'e',
    'tick_down': 'i',
    'tick_left': 'o',
    'tick_northeast': 'u',
}

# Structural correspondence: Voynich glyph_class → Fontana base_form(s)
# Based on Phase 21.2 structural matching (14.81× selectivity)
VOYNICH_TO_FONTANA_FAMILY = {
    'bench': ['circle', 'open_curve_right', 'open_curve_left', 'horizontal_stroke'],
    'minim': ['vertical_stroke'],
    'gallows': ['vertical_stroke'],  # gallows = vertical with crossbar/loop additions
    'compound': ['vertical_stroke'],  # compound = vertical combinations
    'suffix': ['diagonal_right', 'hook', 'angle'],
    'rare': ['diagonal_right', 'hook'],
}

# Voynich stroke features → Fontana modification types
# Maps the stroke-level features that distinguish members within a family
VOYNICH_STROKE_TO_FONTANA_MOD = {
    # Bench family sub-variations by first_stroke
    'loop': 'circle_variant',       # o, a, e — loop-based bench chars
    'open_curve': 'open_curve',     # c, h, ch, etc.
    'sigmoid': 'curve_variant',     # s, sh
    'connector': 'minimal',         # b, j, u
    # Last-stroke variations (the differentiator within sub-groups)
    'tail': 'tick_up',              # a
    'vertical': 'tick_down',        # l
    'hook': 'tick_northeast',       # n, aiin
    'descender': 'tick_left',       # ey, y
    'ascender': 'tick_right',       # g
    'crossbar': 'crossbar',         # t, f, x
    'plume': 'loop_added',          # p
    'sigmoid': 'curve_variant',     # r
}


def _load_fontana_key(bsb_path: str, bnf_path: str) -> List[Dict]:
    """Load and consolidate Fontana cipher key from BSB and BNF sources."""
    combined: Dict[str, Dict] = {}  # keyed by (base_form, added_feature, letter_value)

    for path, source in [(bsb_path, 'bsb'), (bnf_path, 'bnf')]:
        data = _load_json(path)
        if not data:
            continue
        for entry in data.get('entries', []):
            for sign in entry.get('cipher_signs', []):
                lv = sign.get('letter_value', '')
                if not lv:
                    continue
                bf = sign.get('base_form', '')
                af = sign.get('added_feature', '')
                key = f"{bf}|{af}|{lv}"
                if key not in combined:
                    combined[key] = {
                        'base_form': bf,
                        'added_feature': af,
                        'letter_value': lv,
                        'sources': [source],
                        'sign_ids': [sign.get('sign_id', '')],
                        'confidence': sign.get('confidence', 'medium'),
                    }
                else:
                    if source not in combined[key]['sources']:
                        combined[key]['sources'].append(source)
                    sid = sign.get('sign_id', '')
                    if sid not in combined[key]['sign_ids']:
                        combined[key]['sign_ids'].append(sid)

    return list(combined.values())


def _build_fontana_family_consonants(fontana_key: List[Dict]) -> Dict[str, List[str]]:
    """For each Fontana base_form, identify which consonant letters it encodes."""
    vowels = set('aeiou')
    family_consonants: Dict[str, List[str]] = {}

    for entry in fontana_key:
        bf = entry['base_form']
        lv = entry['letter_value'].lower()
        if lv not in vowels and len(lv) == 1:
            family_consonants.setdefault(bf, []).append(lv)

    # Deduplicate and count
    result = {}
    for bf, consonants in family_consonants.items():
        counts = Counter(consonants)
        result[bf] = [c for c, _ in counts.most_common()]

    return result


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class FontanaPhoneticHypothesis:
    eva_char: str
    voynich_family: str
    fontana_family_match: str
    fontana_consonant: str
    fontana_vowel: str
    hypothesized_syllable: str
    evidence_chain: str
    confidence: str
    is_modifier: bool


@dataclass
class FontanaPhoneticResult:
    timestamp: str
    n_fontana_signs: int
    n_bsb_only: int
    n_bnf_only: int
    n_both: int
    fontana_family_summary: List[Dict[str, Any]]
    n_eva_chars_mapped: int
    n_syllables_derived: int
    hypotheses: List[Dict[str, Any]]
    agreement_with_first_syl: int
    disagreement_with_first_syl: int
    agreement_details: List[Dict[str, str]]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_fontana_phonetic() -> Dict[str, Any]:
    """Derive syllable hypotheses from Fontana cipher key."""
    t0 = time.time()
    rdir = _results_dir()

    # --- Locate Fontana JSONs ---
    import os
    archive_base = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))), 'archive', '2Translate')

    bsb_path = os.path.join(archive_base,
        'Fontana_Bellicorum_Instrumentorum_Liber_bsb_DONE',
        'Fontana_Bellicorum_Instrumentorum_Liber_extracted.json')

    # Find BNF path (long filename)
    bnf_dir = os.path.join(archive_base, 'Fontana_Secretum_de_Thesauro_bnf')
    bnf_path = ''
    if os.path.isdir(bnf_dir):
        for fname in os.listdir(bnf_dir):
            if fname.endswith('_extracted.json'):
                bnf_path = os.path.join(bnf_dir, fname)
                break

    # --- Load Fontana key ---
    fontana_key = _load_fontana_key(bsb_path, bnf_path)
    n_bsb = sum(1 for e in fontana_key if e['sources'] == ['bsb'])
    n_bnf = sum(1 for e in fontana_key if e['sources'] == ['bnf'])
    n_both = sum(1 for e in fontana_key if len(e['sources']) > 1)

    # --- Build Fontana family consonant map ---
    family_consonants = _build_fontana_family_consonants(fontana_key)

    # --- Fontana family summary ---
    family_summary_data: Dict[str, Dict] = {}
    for entry in fontana_key:
        bf = entry['base_form']
        if bf not in family_summary_data:
            family_summary_data[bf] = {'letters': [], 'features': set()}
        family_summary_data[bf]['letters'].append(entry['letter_value'])
        family_summary_data[bf]['features'].add(entry['added_feature'])

    family_summary = []
    for bf in sorted(family_summary_data):
        info = family_summary_data[bf]
        letter_counts = dict(Counter(info['letters']).most_common())
        family_summary.append({
            'base_form': bf,
            'letters': letter_counts,
            'features': sorted(info['features']),
            'consonants': family_consonants.get(bf, []),
        })

    # --- Load Voynich data ---
    family_data = _load_json(str(rdir / "tachygraphic_stroke.json")) or {}
    sign_families = family_data.get('sign_families', [])

    mod_data = _load_json(str(rdir / "modifier_integrate.json")) or {}
    modifier_chars = set(mod_data.get('modifier_chars', []))

    # --- Load first_syllable_table for comparison ---
    first_syl_data = _load_json(str(rdir / "first_syllable_table.json")) or {}
    first_syl_table = first_syl_data.get('mode_a_table', [])
    first_syl_lookup: Dict[str, str] = {}
    for entry in first_syl_table:
        ec = entry.get('eva_char', '')
        syl = entry.get('first_syllable_cv', '')
        if ec and syl:
            first_syl_lookup[ec] = syl

    # --- Map each EVA char to Fontana-derived syllable ---
    hypotheses: List[FontanaPhoneticHypothesis] = []

    for fam in sign_families:
        glyph_class = fam.get('glyph_class', '')
        members = fam.get('members', [])
        fontana_families = VOYNICH_TO_FONTANA_FAMILY.get(glyph_class, [])

        # Find the best-matching Fontana consonant for this family
        family_consonant = ''
        matched_fontana_family = ''
        for ff in fontana_families:
            consonants = family_consonants.get(ff, [])
            if consonants:
                family_consonant = consonants[0]  # Most frequent
                matched_fontana_family = ff
                break

        for eva_ch in members:
            is_mod = eva_ch in modifier_chars
            comp = EVA_VISUAL_COMPONENTS.get(eva_ch, {})
            last_stroke = comp.get('last_stroke', '')
            first_stroke = comp.get('first_stroke', '')

            # Derive vowel from last_stroke → Fontana tick direction mapping
            fontana_mod = VOYNICH_STROKE_TO_FONTANA_MOD.get(last_stroke, '')
            vowel = FONTANA_VOWEL_MAP.get(fontana_mod, '')

            # If no vowel from last_stroke, try first_stroke
            if not vowel:
                fontana_mod_first = VOYNICH_STROKE_TO_FONTANA_MOD.get(first_stroke, '')
                vowel = FONTANA_VOWEL_MAP.get(fontana_mod_first, '')

            # Special handling for specific known patterns
            # Gallows: each gallows char has a different crossbar/loop combination
            if glyph_class == 'gallows':
                # k: ascender,ascender → vertical + no addition (Fontana: l or minimal)
                # t: ascender,crossbar → vertical + crossbar (Fontana: t)
                # p: ascender,plume → vertical + loop (Fontana: p or b)
                # f: ascender,crossbar → vertical + crossbar (Fontana: f)
                gallows_consonant_map = {
                    'k': 'k',  # vertical+ascender combination
                    't': 't',  # vertical+crossbar → t in Fontana
                    'p': 'p',  # vertical+plume(loop) → p in Fontana
                    'f': 'f',  # vertical+crossbar variant → f in Fontana
                }
                if eva_ch in gallows_consonant_map:
                    family_consonant = gallows_consonant_map[eva_ch]

            # Build syllable hypothesis
            if family_consonant and vowel:
                syllable = family_consonant + vowel
                confidence = 'high' if matched_fontana_family else 'medium'
            elif family_consonant:
                syllable = family_consonant + 'e'  # Default vowel
                confidence = 'low'
            elif vowel:
                syllable = vowel  # Pure vowel
                confidence = 'low'
            else:
                syllable = ''
                confidence = 'none'

            evidence = (f"voynich_{glyph_class}→fontana_{matched_fontana_family}; "
                       f"consonant={family_consonant}; "
                       f"last_stroke={last_stroke}→vowel={vowel}")

            hypotheses.append(FontanaPhoneticHypothesis(
                eva_char=eva_ch,
                voynich_family=glyph_class,
                fontana_family_match=matched_fontana_family,
                fontana_consonant=family_consonant,
                fontana_vowel=vowel,
                hypothesized_syllable=syllable,
                evidence_chain=evidence,
                confidence=confidence,
                is_modifier=is_mod,
            ))

    # --- Compare with first-syllable table ---
    n_agree = 0
    n_disagree = 0
    agreement_details: List[Dict[str, str]] = []

    for hyp in hypotheses:
        if hyp.is_modifier or not hyp.hypothesized_syllable:
            continue
        first_syl = first_syl_lookup.get(hyp.eva_char, '')
        if not first_syl:
            continue

        agree = hyp.hypothesized_syllable == first_syl
        if agree:
            n_agree += 1
        else:
            n_disagree += 1

        agreement_details.append({
            'eva_char': hyp.eva_char,
            'fontana_syllable': hyp.hypothesized_syllable,
            'first_syllable': first_syl,
            'agree': 'yes' if agree else 'no',
        })

    # --- Build result ---
    n_mapped = sum(1 for h in hypotheses if h.hypothesized_syllable)
    n_with_syl = sum(1 for h in hypotheses if h.hypothesized_syllable and not h.is_modifier)

    result = FontanaPhoneticResult(
        timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
        n_fontana_signs=len(fontana_key),
        n_bsb_only=n_bsb,
        n_bnf_only=n_bnf,
        n_both=n_both,
        fontana_family_summary=family_summary,
        n_eva_chars_mapped=n_mapped,
        n_syllables_derived=n_with_syl,
        hypotheses=[_convert(asdict(h)) for h in hypotheses],
        agreement_with_first_syl=n_agree,
        disagreement_with_first_syl=n_disagree,
        agreement_details=agreement_details,
    )

    out_path = rdir / "fontana_phonetic.json"
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t0
    print(f"fontana-phon: {len(fontana_key)} Fontana signs, "
          f"{n_with_syl} syllables derived, "
          f"agree/disagree with first-syl: {n_agree}/{n_disagree} ({elapsed:.1f}s)")

    return _convert(asdict(result))
