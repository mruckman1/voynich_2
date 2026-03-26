"""
Phase 61, Track D: Zodiac Labels Under CVC Decode
===================================================
Re-decodes all 299 zodiac labels (f70v-f73v) through the corrected CVC
pipeline and matches against month/zodiac names in 6 languages.  Compares
folio-selectivity to Phase 26's CV result (NO_SIGNAL).

Dependency chain:
    results/zodiac_map.json           (Phase 26.1)
    results/combined_refine.json      (Phase 15)
    results/modifier_integrate.json   (Phase 16)
        -> results/phase61_zodiac_cvc.json
"""

import json
import os
import random
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import build_eva_to_triple_lookup, load_corpus, tokenize_eva_chars
from voynich.core.reference import MONTH_NAMES_MULTI, ZODIAC_NAMES_MULTI
from voynich.phases.corrected_coda import (
    build_coda_table_v2,
    decode_token_cvc_v2,
)
from voynich.phases.zodiac_map import FOLIO_ZODIAC_MAP


# ---------------------------------------------------------------------------
# JSON helpers
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
# Levenshtein edit distance
# ---------------------------------------------------------------------------

def _edit_distance(a: str, b: str) -> int:
    """Standard Levenshtein edit distance."""
    if len(a) < len(b):
        return _edit_distance(b, a)
    if len(b) == 0:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            cost = 0 if ca == cb else 1
            curr.append(min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost))
        prev = curr
    return prev[len(b)]


# ---------------------------------------------------------------------------
# Expanded name tables with CVC-compatible variants
# ---------------------------------------------------------------------------

def _build_expanded_month_names() -> Dict[str, Dict[int, List[str]]]:
    """Build month names with CVC variants (coda endings)."""
    expanded: Dict[str, Dict[int, List[str]]] = {}
    for lang, names in MONTH_NAMES_MULTI.items():
        expanded[lang] = {}
        for idx, name in enumerate(names):
            month_num = idx + 1
            forms = [name.lower()]
            # Add truncated and CVC variants
            if len(name) > 4:
                forms.append(name[:4].lower())   # first 4 chars
                forms.append(name[:5].lower())   # first 5 chars
            # Add coda variants (-en, -on, -er)
            stem = name.lower().rstrip('aeiouy')
            if stem and len(stem) >= 3:
                for suffix in ('en', 'on', 'er', 'in'):
                    variant = stem + suffix
                    if variant != name.lower() and len(variant) >= 4:
                        forms.append(variant)
            expanded[lang][month_num] = list(set(forms))
    return expanded


def _build_expanded_zodiac_names() -> Dict[str, Dict[str, List[str]]]:
    """Build zodiac names with CVC variants."""
    signs = [
        'aries', 'taurus', 'gemini', 'cancer', 'leo', 'virgo',
        'libra', 'scorpio', 'sagittarius', 'capricornus', 'aquarius', 'pisces',
    ]
    expanded: Dict[str, Dict[str, List[str]]] = {}
    for lang, names in ZODIAC_NAMES_MULTI.items():
        expanded[lang] = {}
        for idx, name in enumerate(names):
            sign = signs[idx]
            forms = [name.lower()]
            if len(name) > 4:
                forms.append(name[:4].lower())
                forms.append(name[:5].lower())
            stem = name.lower().rstrip('aeiouy')
            if stem and len(stem) >= 3:
                for suffix in ('en', 'on', 'er', 'in'):
                    variant = stem + suffix
                    if variant != name.lower() and len(variant) >= 4:
                        forms.append(variant)
            expanded[lang][sign] = list(set(forms))
    return expanded


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ZodiacMatch:
    label_eva: str
    label_cvc: str
    folio: str
    language: str
    match_type: str       # 'month' or 'zodiac'
    target: str
    target_key: str       # month_num or sign name
    ed: int
    correct_folio: bool


@dataclass
class ZodiacCvcResult:
    phase: str = "61"
    step: str = "61.4"
    experiment: str = "zodiac_cvc"
    n_labels_decoded: int = 0
    n_unique_cvc: int = 0
    n_matches_ed2: int = 0
    n_matches_ed1: int = 0
    n_correct_folio: int = 0
    correct_rate: float = 0.0
    null_correct_mean: float = 0.0
    null_correct_std: float = 0.0
    z_score: float = 0.0
    selectivity: float = 0.0
    # Per-language breakdown
    language_breakdown: Dict[str, int] = field(default_factory=dict)
    # Comparison to Phase 26
    phase26_n_matches: int = 109
    phase26_selectivity: float = 0.10
    improvement: bool = False
    # Top matches
    top_correct_matches: List[Dict[str, Any]] = field(default_factory=list)
    top_ed1_matches: List[Dict[str, Any]] = field(default_factory=list)
    all_matches: List[Dict[str, Any]] = field(default_factory=list)
    # Gates
    g1_enough_matches: bool = False    # >= 5 matches at ED <= 2
    g2_folio_selective: bool = False   # selectivity > 1.5
    g3_language_consist: bool = False  # >= 2 correct-folio from same lang
    g4_beats_phase26: bool = False     # selectivity > 0.10
    g5_strong_hit: bool = False        # >= 1 match ED<=1 on correct folio
    gates_passed: int = 0
    gate_passed: bool = False
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def _decode_zodiac_labels(
    corpus,
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    coda_table,
) -> List[Dict[str, Any]]:
    """Decode all zodiac section labels using corrected CVC pipeline."""
    decoded_labels = []

    for folio_id, info in FOLIO_ZODIAC_MAP.items():
        page = corpus.get_page(folio_id)
        if page is None:
            continue

        for locus in page.loci:
            text = locus.clean_text
            if not text or not text.strip():
                continue

            for token in text.split():
                token = token.strip()
                if not token:
                    continue

                result = decode_token_cvc_v2(
                    token, assignment, eva_to_triple, coda_table,
                )
                cvc_str = result.decoded_cvc if hasattr(result, 'decoded_cvc') else str(result)

                decoded_labels.append({
                    'eva': token,
                    'cvc': cvc_str.lower(),
                    'folio': folio_id,
                    'sign': info['sign'],
                    'month_idx': info['month_idx'],
                })

    return decoded_labels


def _match_labels(
    decoded_labels: List[Dict[str, Any]],
    month_names: Dict[str, Dict[int, List[str]]],
    zodiac_names: Dict[str, Dict[str, List[str]]],
    max_ed: int = 2,
) -> List[ZodiacMatch]:
    """Match decoded labels against all month and zodiac names."""
    matches: List[ZodiacMatch] = []

    for label in decoded_labels:
        cvc = label['cvc']
        if not cvc or cvc == '?' or len(cvc) < 2:
            continue

        folio = label['folio']
        expected_month = label['month_idx']
        expected_sign = label['sign']

        # Match against month names
        for lang, month_dict in month_names.items():
            for month_num, forms in month_dict.items():
                for form in forms:
                    ed = _edit_distance(cvc, form)
                    if ed <= max_ed:
                        correct = (month_num == expected_month)
                        matches.append(ZodiacMatch(
                            label_eva=label['eva'],
                            label_cvc=cvc,
                            folio=folio,
                            language=lang,
                            match_type='month',
                            target=form,
                            target_key=str(month_num),
                            ed=ed,
                            correct_folio=correct,
                        ))

        # Match against zodiac names
        for lang, sign_dict in zodiac_names.items():
            for sign, forms in sign_dict.items():
                for form in forms:
                    ed = _edit_distance(cvc, form)
                    if ed <= max_ed:
                        correct = (sign == expected_sign)
                        matches.append(ZodiacMatch(
                            label_eva=label['eva'],
                            label_cvc=cvc,
                            folio=folio,
                            language=lang,
                            match_type='zodiac',
                            target=form,
                            target_key=sign,
                            ed=ed,
                            correct_folio=correct,
                        ))

    return matches


def _folio_selectivity_test(
    matches: List[ZodiacMatch],
    n_perms: int = 1000,
) -> Dict[str, Any]:
    """Test whether correct-folio matches are above chance."""
    if not matches:
        return {'correct_rate': 0.0, 'null_mean': 0.0, 'null_std': 0.0,
                'z': 0.0, 'selectivity': 0.0}

    n_correct = sum(1 for m in matches if m.correct_folio)
    real_rate = n_correct / len(matches)

    # Null: shuffle the correct_folio flags
    rng = random.Random(42)
    null_rates = []
    flags = [m.correct_folio for m in matches]
    for _ in range(n_perms):
        rng.shuffle(flags)
        null_rates.append(sum(flags) / len(flags))

    null_mean = sum(null_rates) / len(null_rates) if null_rates else 0.0
    null_var = (sum((r - null_mean) ** 2 for r in null_rates)
                / len(null_rates) if null_rates else 0.0)
    null_std = null_var ** 0.5

    z = (real_rate - null_mean) / null_std if null_std > 0 else 0.0
    selectivity = real_rate / null_mean if null_mean > 0 else 0.0

    return {
        'n_correct': n_correct,
        'n_total': len(matches),
        'correct_rate': round(real_rate, 4),
        'null_mean': round(null_mean, 4),
        'null_std': round(null_std, 4),
        'z': round(z, 2),
        'selectivity': round(selectivity, 2),
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_zodiac_cvc():
    t0 = time.time()
    print("=" * 70)
    print("Phase 61, Track D: Zodiac Labels Under CVC Decode")
    print("=" * 70)

    rd = str(_results_dir())

    # Load data
    print("\n  Loading data ...")
    eva_to_triple = build_eva_to_triple_lookup()

    refine_data = _safe_load(os.path.join(rd, 'combined_refine.json'))
    assignment = refine_data.get('best_assignment', {})

    coda_table = build_coda_table_v2()
    corpus = load_corpus(verbose=False)

    # Build expanded name tables
    month_names = _build_expanded_month_names()
    zodiac_names = _build_expanded_zodiac_names()

    n_month_forms = sum(len(forms) for d in month_names.values() for forms in d.values())
    n_zodiac_forms = sum(len(forms) for d in zodiac_names.values() for forms in d.values())
    print(f"  Month name forms: {n_month_forms}")
    print(f"  Zodiac name forms: {n_zodiac_forms}")

    # Decode all zodiac labels
    print("\n  1. Decoding zodiac labels with corrected CVC ...")
    decoded_labels = _decode_zodiac_labels(corpus, assignment, eva_to_triple, coda_table)
    unique_cvc = set(l['cvc'] for l in decoded_labels)
    print(f"     {len(decoded_labels)} labels decoded, {len(unique_cvc)} unique CVC forms")

    # Match against names
    print("\n  2. Matching against month/zodiac names (ED <= 2) ...")
    matches = _match_labels(decoded_labels, month_names, zodiac_names, max_ed=2)
    correct = [m for m in matches if m.correct_folio]
    ed1_matches = [m for m in matches if m.ed <= 1]
    ed1_correct = [m for m in ed1_matches if m.correct_folio]

    print(f"     {len(matches)} total matches at ED <= 2")
    print(f"     {len(correct)} on correct folio")
    print(f"     {len(ed1_matches)} at ED <= 1 ({len(ed1_correct)} on correct folio)")

    # Language breakdown
    lang_counts: Dict[str, int] = Counter()
    for m in correct:
        lang_counts[m.language] += 1
    print(f"     Correct-folio by language: {dict(lang_counts)}")

    # Folio selectivity test
    print("\n  3. Folio selectivity permutation test (1000 perms) ...")
    sel_result = _folio_selectivity_test(matches)
    print(f"     Correct rate: {sel_result['correct_rate']:.4f}")
    print(f"     Null mean:    {sel_result['null_mean']:.4f}")
    print(f"     z-score:      {sel_result['z']:.2f}")
    print(f"     Selectivity:  {sel_result['selectivity']:.2f}×")

    # Check for language consistency among correct-folio matches
    correct_lang_counts = Counter(m.language for m in correct)
    best_lang_count = max(correct_lang_counts.values()) if correct_lang_counts else 0

    # Gates
    g1 = len(matches) >= 5
    g2 = sel_result['selectivity'] > 1.5
    g3 = best_lang_count >= 2
    g4 = sel_result['selectivity'] > 0.10
    g5 = len(ed1_correct) >= 1
    gates = sum([g1, g2, g3, g4, g5])

    result = ZodiacCvcResult(
        n_labels_decoded=len(decoded_labels),
        n_unique_cvc=len(unique_cvc),
        n_matches_ed2=len(matches),
        n_matches_ed1=len(ed1_matches),
        n_correct_folio=len(correct),
        correct_rate=sel_result['correct_rate'],
        null_correct_mean=sel_result['null_mean'],
        null_correct_std=sel_result['null_std'],
        z_score=sel_result['z'],
        selectivity=sel_result['selectivity'],
        language_breakdown=dict(lang_counts),
        improvement=sel_result['selectivity'] > 0.10,
        top_correct_matches=[_convert(asdict(m)) for m in sorted(correct, key=lambda x: x.ed)[:10]],
        top_ed1_matches=[_convert(asdict(m)) for m in sorted(ed1_matches, key=lambda x: x.ed)[:10]],
        all_matches=[_convert(asdict(m)) for m in matches[:200]],
        g1_enough_matches=g1,
        g2_folio_selective=g2,
        g3_language_consist=g3,
        g4_beats_phase26=g4,
        g5_strong_hit=g5,
        gates_passed=gates,
        gate_passed=gates >= 3,
        runtime_seconds=round(time.time() - t0, 1),
    )

    path = _save_json(rd, 'phase61_zodiac_cvc.json', result)

    # Summary
    print("\n" + "=" * 70)
    print("  TRACK D SUMMARY: Zodiac CVC Re-Decode")
    print("=" * 70)
    print(f"  Labels decoded:     {result.n_labels_decoded}")
    print(f"  Matches (ED<=2):    {result.n_matches_ed2}")
    print(f"  Matches (ED<=1):    {result.n_matches_ed1}")
    print(f"  Correct folio:      {result.n_correct_folio}")
    print(f"  Selectivity:        {result.selectivity:.2f}×")
    print(f"  z-score:            {result.z_score:.2f}")
    print(f"  Phase 26 sel:       {result.phase26_selectivity:.2f}×")
    print(f"  Improvement:        {result.improvement}")
    print(f"\n  Gates: {gates}/5 passed")
    print(f"    G1 (>=5 matches):       {g1}")
    print(f"    G2 (sel > 1.5):         {g2}")
    print(f"    G3 (lang consistency):  {g3}")
    print(f"    G4 (beats Phase 26):    {g4}")
    print(f"    G5 (ED<=1 correct):     {g5}")
    if correct:
        print(f"\n  Top correct-folio matches:")
        for m in sorted(correct, key=lambda x: x.ed)[:5]:
            print(f"    {m.label_eva} -> {m.label_cvc} ~ {m.target} "
                  f"({m.language}, {m.match_type}, ED={m.ed}, {m.folio})")
    print(f"\n  Saved: {path}")
    print(f"  Runtime: {result.runtime_seconds:.1f}s")
