"""
Phase 20.2 – Sign Family to Syllable Family Mapping
=====================================================
Map Phase 19.5's stroke-based sign families to Latin syllable families using
anchors from Step 20.1 and the tachygraphic principle that within a family
one phonetic dimension is fixed.

Dependency chain:
    tachy_anchors.json + tachygraphic_stroke.json + stroke_modification.json
    + modifier_integrate.json
        → tachy_families.json
"""

import json
import os
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import build_eva_to_triple_lookup, load_corpus, tokenize_eva_chars
from voynich.core.reference import (
    EVA_VISUAL_COMPONENTS,
    PHONEME_NUCLEUS_MAP,
    PHONEME_PLACE_MAP,
    build_expanded_word_set,
    load_reference_corpus,
)
from voynich.core.stats import jensen_shannon_divergence

import numpy as np


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
class SubFamily:
    name: str                       # e.g. "bench_loop"
    parent_class: str               # e.g. "bench"
    first_stroke: str               # shared first_stroke value
    modification_dimension: str     # "last_stroke" for sub-families
    members: List[str]              # EVA chars (syllabic only)
    all_members: List[str]          # including modifiers
    consonant_candidates: List[str] # from PHONEME_PLACE_MAP
    assigned_consonant: str         # resolved consonant
    vowel_assignments: Dict[str, str]  # eva_char → vowel
    syllable_assignments: Dict[str, str]  # eva_char → full syllable
    evidence_sources: Dict[str, str]  # eva_char → "anchor"/"family"/"freq"
    anchor_based: bool              # True if consonant from anchor


@dataclass
class TachyFamiliesResult:
    n_families_original: int        # 6
    n_subfamilies: int              # after bench/rare sub-segmentation
    subfamilies: List[Dict]
    preliminary_table: Dict[str, str]  # eva_char → syllable
    evidence_provenance: Dict[str, str]  # eva_char → evidence source
    n_chars_covered: int
    n_consonant_classes_used: int
    consonant_class_diversity: float  # 0-1
    frequency_jsd: float            # table freq vs Latin freq
    gate_passed: bool
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _load_sign_families(rd: str) -> List[Dict]:
    path = os.path.join(rd, 'tachygraphic_stroke.json')
    if not os.path.exists(path):
        print("    [WARN] tachygraphic_stroke.json not found")
        return []
    with open(path) as f:
        data = json.load(f)
    return data.get('sign_families', [])


def _load_anchors(rd: str) -> Dict[str, Dict]:
    """Load Tier 1/2 anchors from Step 20.1.  Returns dict: char → info."""
    path = os.path.join(rd, 'tachy_anchors.json')
    if not os.path.exists(path):
        print("    [WARN] tachy_anchors.json not found")
        return {}
    with open(path) as f:
        data = json.load(f)
    anchors = {}
    for a in data.get('char_anchors', []):
        if a.get('tier', 3) <= 2:
            anchors[a['eva_char']] = a
    return anchors


def _load_modifier_chars(rd: str) -> Set[str]:
    path = os.path.join(rd, 'modifier_integrate.json')
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        data = json.load(f)
    return set(data.get('modifier_chars', []))


# ---------------------------------------------------------------------------
# Sub-segmentation
# ---------------------------------------------------------------------------

def _subsegment_family(
    family: Dict,
    modifier_chars: Set[str],
) -> List[SubFamily]:
    """Sub-segment a family by first_stroke if modification_dimension='both'.
    Families with dimension 'last_stroke' or 'first_stroke' stay as one."""
    glyph_class = family['glyph_class']
    members = family['members']
    mod_dim = family['modification_dimension']

    # Split into syllabic and modifier members
    syllabic = [m for m in members if m not in modifier_chars]
    all_members_by_stroke: Dict[str, List[str]] = defaultdict(list)
    syllabic_by_stroke: Dict[str, List[str]] = defaultdict(list)

    for m in members:
        comp = EVA_VISUAL_COMPONENTS.get(m, {})
        fs = comp.get('first_stroke', 'unknown')
        all_members_by_stroke[fs].append(m)
        if m not in modifier_chars:
            syllabic_by_stroke[fs].append(m)

    if mod_dim == 'both' and len(syllabic_by_stroke) > 1:
        # Sub-segment by first_stroke
        subfamilies = []
        for fs, syls in sorted(syllabic_by_stroke.items()):
            if not syls:
                continue
            cons = PHONEME_PLACE_MAP.get(fs, ['?'])
            subfamilies.append(SubFamily(
                name=f"{glyph_class}_{fs}",
                parent_class=glyph_class,
                first_stroke=fs,
                modification_dimension='last_stroke',  # within sub-family
                members=syls,
                all_members=all_members_by_stroke.get(fs, syls),
                consonant_candidates=list(cons),
                assigned_consonant='',
                vowel_assignments={},
                syllable_assignments={},
                evidence_sources={},
                anchor_based=False,
            ))
        return subfamilies
    else:
        # Single sub-family
        fs_values = family.get('first_stroke_values', [])
        fs = fs_values[0] if len(fs_values) == 1 else (fs_values[0] if fs_values else 'unknown')
        cons = PHONEME_PLACE_MAP.get(fs, ['?'])
        return [SubFamily(
            name=glyph_class,
            parent_class=glyph_class,
            first_stroke=fs,
            modification_dimension=mod_dim,
            members=syllabic,
            all_members=members,
            consonant_candidates=list(cons),
            assigned_consonant='',
            vowel_assignments={},
            syllable_assignments={},
            evidence_sources={},
            anchor_based=False,
        )]


# ---------------------------------------------------------------------------
# Consonant and vowel assignment
# ---------------------------------------------------------------------------

def _extract_consonant(syllable: str) -> str:
    """Extract onset consonant(s) from a CV syllable."""
    vowels = set('aeiou')
    onset = []
    for c in syllable:
        if c in vowels:
            break
        onset.append(c)
    return ''.join(onset) if onset else ''


def _extract_vowel(syllable: str) -> str:
    """Extract nucleus vowel from a CV syllable."""
    vowels = set('aeiou')
    for c in syllable:
        if c in vowels:
            return c
    return ''


def _assign_consonant_to_subfamily(
    sf: SubFamily,
    anchors: Dict[str, Dict],
    latin_consonant_freq: Dict[str, float],
) -> None:
    """Assign a consonant class to the sub-family using anchors or frequency."""
    # Check if any member has an anchor
    for m in sf.members:
        if m in anchors:
            syl = anchors[m]['syllable']
            cons = _extract_consonant(syl)
            if cons:
                sf.assigned_consonant = cons
                sf.anchor_based = True
                return

    # No anchor — use frequency matching
    # Pick the most likely consonant from candidates based on Latin frequency
    best_cons = ''
    best_freq = -1.0
    for c in sf.consonant_candidates:
        freq = latin_consonant_freq.get(c, 0.0)
        if freq > best_freq:
            best_freq = freq
            best_cons = c
    sf.assigned_consonant = best_cons or (sf.consonant_candidates[0] if sf.consonant_candidates else '?')


def _assign_vowels_within_subfamily(
    sf: SubFamily,
    anchors: Dict[str, Dict],
    char_freqs: Counter,
) -> None:
    """Assign vowels to members within a sub-family."""
    LATIN_VOWEL_ORDER = ['a', 'e', 'i', 'o', 'u']

    # Pre-assign from anchors
    assigned: Dict[str, str] = {}
    for m in sf.members:
        if m in anchors:
            vowel = _extract_vowel(anchors[m]['syllable'])
            if vowel:
                assigned[m] = vowel
                sf.evidence_sources[m] = 'anchor'

    # Remaining members: assign by frequency ordering
    # Sort remaining by corpus frequency (descending)
    remaining = [m for m in sf.members if m not in assigned]
    remaining.sort(key=lambda m: char_freqs.get(m, 0), reverse=True)

    # Available vowels (remove those already anchored)
    used_vowels = set(assigned.values())
    available = [v for v in LATIN_VOWEL_ORDER if v not in used_vowels]

    for i, m in enumerate(remaining):
        if i < len(available):
            assigned[m] = available[i]
        elif LATIN_VOWEL_ORDER:
            # Wrap around or reuse least-used vowel
            assigned[m] = LATIN_VOWEL_ORDER[i % len(LATIN_VOWEL_ORDER)]
        sf.evidence_sources.setdefault(m, 'frequency')

    sf.vowel_assignments = assigned

    # Build full syllable assignments
    cons = sf.assigned_consonant
    for m, vowel in assigned.items():
        sf.syllable_assignments[m] = cons + vowel
        if m in anchors:
            # Prefer the actual anchored syllable
            sf.syllable_assignments[m] = anchors[m]['syllable']
            sf.evidence_sources[m] = 'anchor'


# ---------------------------------------------------------------------------
# Latin frequency data
# ---------------------------------------------------------------------------

def _build_latin_consonant_freq(ref_word_set: set) -> Dict[str, float]:
    """Estimate consonant onset frequency from Latin reference words."""
    onset_counts: Counter = Counter()
    total = 0
    vowels = set('aeiou')

    for word in ref_word_set:
        if not word:
            continue
        onset = []
        for c in word.lower():
            if c in vowels:
                break
            if c.isalpha():
                onset.append(c)
        if onset:
            onset_counts[''.join(onset)] += 1
        total += 1

    freq = {}
    for cons, count in onset_counts.items():
        freq[cons] = count / total if total else 0.0
    return freq


def _build_latin_syllable_freq(ref_word_set: set) -> Dict[str, float]:
    """Estimate CV syllable frequency from Latin reference words."""
    from voynich.core.stats import syllabify_latin
    syl_counts: Counter = Counter()
    total = 0
    for word in list(ref_word_set)[:10000]:
        try:
            syls = syllabify_latin(word)
            for s in syls:
                syl_counts[s.lower()] += 1
                total += 1
        except Exception:
            continue
    return {s: c / total for s, c in syl_counts.items()} if total else {}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_tachy_families() -> None:
    """Step 20.2: Map sign families to syllable families."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE 20.2: Sign Family → Syllable Family Mapping")
    print("=" * 70)

    rd = _results_dir()

    # ─── 1. Load dependencies ───
    print("\n  1. Loading dependencies …")
    families = _load_sign_families(rd)
    anchors = _load_anchors(rd)
    modifier_chars = _load_modifier_chars(rd)

    print(f"      Sign families: {len(families)}")
    print(f"      Tier 1/2 anchors: {len(anchors)}")

    # Build Latin frequency data
    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    base_words = set(
        w.lower() for w in ref_corpus.get_combined_tokens('latin')
        if len(w) >= 2
    )
    expanded_words, _ = build_expanded_word_set(base_words)
    ref_word_set = base_words | expanded_words
    latin_consonant_freq = _build_latin_consonant_freq(ref_word_set)

    # Build EVA char corpus frequencies
    corpus = load_corpus(verbose=False)
    tokens = corpus.get_tokens()
    char_freqs: Counter = Counter()
    for token in tokens:
        for ch in tokenize_eva_chars(token):
            char_freqs[ch] += 1

    # ─── 2. Sub-segment families ───
    print("\n  2. Sub-segmenting families …")
    all_subfamilies: List[SubFamily] = []
    for fam in families:
        subs = _subsegment_family(fam, modifier_chars)
        all_subfamilies.extend(subs)
        if len(subs) > 1:
            print(f"      {fam['glyph_class']}: {fam['n_members']} members "
                  f"→ {len(subs)} sub-families")
        else:
            n_syl = len(subs[0].members)
            print(f"      {fam['glyph_class']}: {n_syl} syllabic members")

    print(f"      Total sub-families: {len(all_subfamilies)}")

    # ─── 3. Assign consonants ───
    print("\n  3. Assigning consonant classes …")
    for sf in all_subfamilies:
        _assign_consonant_to_subfamily(sf, anchors, latin_consonant_freq)
        src = "anchor" if sf.anchor_based else "frequency"
        print(f"      {sf.name:20s}: consonant='{sf.assigned_consonant}' "
              f"({src}, {len(sf.members)} members)")

    # ─── 4. Assign vowels within each sub-family ───
    print("\n  4. Assigning vowels within sub-families …")
    for sf in all_subfamilies:
        _assign_vowels_within_subfamily(sf, anchors, char_freqs)

    # ─── 5. Build preliminary table ───
    print("\n  5. Building preliminary table …")
    preliminary_table: Dict[str, str] = {}
    evidence_provenance: Dict[str, str] = {}

    for sf in all_subfamilies:
        for m, syl in sf.syllable_assignments.items():
            preliminary_table[m] = syl
            evidence_provenance[m] = sf.evidence_sources.get(m, 'family')

    print(f"      Chars mapped: {len(preliminary_table)}")

    # Print table
    for ch in sorted(preliminary_table.keys()):
        syl = preliminary_table[ch]
        src = evidence_provenance.get(ch, '?')
        comp = EVA_VISUAL_COMPONENTS.get(ch, {})
        gc = comp.get('glyph_class', '?')
        print(f"        {ch:8s} → {syl:4s}  [{src:9s}]  {gc}")

    # ─── 6. Consistency checks ───
    print("\n  6. Consistency checks …")

    # Check consonant class diversity
    consonant_classes = set(sf.assigned_consonant for sf in all_subfamilies
                           if sf.assigned_consonant)
    n_classes = len(consonant_classes)
    # Expected: 5 from C5_V4 model
    diversity = n_classes / 5.0 if n_classes <= 5 else 1.0
    print(f"      Distinct consonant classes: {n_classes} "
          f"(expected ~5, diversity={diversity:.2f})")

    # Check for duplicate syllable assignments
    syl_counts = Counter(preliminary_table.values())
    duplicates = {s: c for s, c in syl_counts.items() if c > 1}
    n_dups = len(duplicates)
    print(f"      Duplicate syllable assignments: {n_dups}")
    for syl, count in sorted(duplicates.items(), key=lambda x: -x[1])[:5]:
        chars = [ch for ch, s in preliminary_table.items() if s == syl]
        print(f"        '{syl}' assigned to: {chars}")

    # Frequency JSD
    latin_syl_freq = _build_latin_syllable_freq(ref_word_set)
    if latin_syl_freq and preliminary_table:
        # Build mapped frequency distribution
        all_syls = sorted(set(list(latin_syl_freq.keys())
                              + list(preliminary_table.values())))
        mapped_freq: Counter = Counter()
        total_freq = 0
        for ch, syl in preliminary_table.items():
            f = char_freqs.get(ch, 0)
            mapped_freq[syl] += f
            total_freq += f
        p = np.array([mapped_freq.get(s, 0) / max(total_freq, 1) for s in all_syls])
        q = np.array([latin_syl_freq.get(s, 0) for s in all_syls])
        # Normalise
        p_sum = p.sum()
        q_sum = q.sum()
        if p_sum > 0 and q_sum > 0:
            p = p / p_sum
            q = q / q_sum
            freq_jsd = float(jensen_shannon_divergence(p, q))
        else:
            freq_jsd = 1.0
    else:
        freq_jsd = 1.0
    print(f"      Frequency JSD (mapped vs Latin): {freq_jsd:.4f}")

    # ─── 7. Gate ───
    gate_passed = n_classes >= 3 and len(preliminary_table) >= 15
    if gate_passed:
        verdict = (f"PASS: {len(preliminary_table)} chars mapped across "
                   f"{n_classes} consonant classes. "
                   f"JSD={freq_jsd:.4f}.")
    else:
        verdict = (f"FAIL: {len(preliminary_table)} chars mapped, "
                   f"{n_classes} classes (need ≥3 classes, ≥15 chars).")

    print(f"\n  7. Gate: {verdict}")

    # ─── 8. Save ───
    result = TachyFamiliesResult(
        n_families_original=len(families),
        n_subfamilies=len(all_subfamilies),
        subfamilies=[asdict(sf) for sf in all_subfamilies],
        preliminary_table=preliminary_table,
        evidence_provenance=evidence_provenance,
        n_chars_covered=len(preliminary_table),
        n_consonant_classes_used=n_classes,
        consonant_class_diversity=diversity,
        frequency_jsd=freq_jsd,
        gate_passed=gate_passed,
        verdict=verdict,
        runtime_seconds=time.time() - t0,
    )

    out_path = os.path.join(rd, 'tachy_families.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)

    print(f"\n  → {out_path}")
