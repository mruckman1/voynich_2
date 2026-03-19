"""
Phase 54.8 – Zodiac Label Dialect Decode
=========================================
Decode zodiac-section labels using the Phase 16 assignment table and match
them against month names and zodiac sign names in five northern-Italian
dialects (Venetian, Lombard, Ligurian, Emilian, Tuscan).

Dependency chain:
    zodiac_map.json          (Step 26.1 — label catalog)
    combined_refine.json     (Phase 15 — best assignment)
    modifier_integrate.json  (Phase 16 — modifier classification)
        → phase54_zodiac.json   (this step)
"""

import json
import os
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Any, Set, Tuple, Optional

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    decode_token_modifier_aware,
    tokenize_eva_chars,
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


def _edit_distance(a: str, b: str) -> int:
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if a[i-1] == b[j-1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j-1])
            prev = temp
    return dp[n]


# ---------------------------------------------------------------------------
# Folio-to-zodiac mapping (from Phase 26)
# ---------------------------------------------------------------------------

FOLIO_ZODIAC_MAP = {
    'f70v1': {'sign': 'aries', 'month': 4, 'month_name': 'april'},
    'f70v2': {'sign': 'pisces', 'month': 3, 'month_name': 'march'},
    'f71r':  {'sign': 'aries', 'month': 4, 'month_name': 'april'},
    'f71v':  {'sign': 'taurus', 'month': 5, 'month_name': 'may'},
    'f72r1': {'sign': 'gemini', 'month': 6, 'month_name': 'june'},
    'f72r2': {'sign': 'cancer', 'month': 7, 'month_name': 'july'},
    'f72r3': {'sign': 'leo', 'month': 8, 'month_name': 'august'},
    'f72v1': {'sign': 'virgo', 'month': 9, 'month_name': 'september'},
    'f72v2': {'sign': 'libra', 'month': 10, 'month_name': 'october'},
    'f72v3': {'sign': 'scorpio', 'month': 11, 'month_name': 'november'},
    'f73r':  {'sign': 'sagittarius', 'month': 12, 'month_name': 'december'},
    'f73v':  {'sign': 'capricorn', 'month': 1, 'month_name': 'january'},
}


# ---------------------------------------------------------------------------
# Dialect month and zodiac name tables
# ---------------------------------------------------------------------------

DIALECT_MONTHS = {
    'venetian': {
        3: ['marso', 'marz', 'marzo'],
        4: ['avrile', 'avril', 'aprile'],
        5: ['mazo', 'magio', 'maggio'],
        6: ['zugno', 'giugno'],
        7: ['lugio', 'luio', 'luglio'],
        8: ['agosto', 'avosto'],
        9: ['setenbre', 'setenbrio', 'settembre'],
        10: ['otobre', 'otubrio', 'ottobre'],
        11: ['novenbre', 'novenbrio', 'novembre'],
        12: ['desenbre', 'desenbrio', 'dicembre'],
        1: ['zenaro', 'zener', 'gennaio'],
        2: ['fevraro', 'febrer', 'febbraio'],
    },
    'lombard': {
        3: ['mars', 'marzo'],
        4: ['avril', 'aprile'],
        5: ['magg', 'maggio'],
        6: ['giugn', 'giugno'],
        7: ['luij', 'luglio'],
        8: ['agost', 'agosto'],
        9: ['setember', 'settembre'],
        10: ['otober', 'ottobre'],
        11: ['november', 'novembre'],
        12: ['desember', 'dicembre'],
        1: ['genar', 'gennaio'],
        2: ['febrar', 'febbraio'],
    },
    'ligurian': {
        3: ['marso', 'marzo'],
        4: ['arvî', 'aprile'],
        5: ['mazzo', 'maggio'],
        6: ['zugno', 'giugno'],
        7: ['luggio', 'luglio'],
        8: ['agosto'],
        9: ['settembre'],
        10: ['ottobre'],
        11: ['novembre'],
        12: ['dicembre'],
        1: ['zena', 'gennaio'],
        2: ['febraro', 'febbraio'],
    },
    'emilian': {
        3: ['mêrs', 'marz', 'marzo'],
        4: ['avrîl', 'april', 'aprile'],
        5: ['mâg', 'magg', 'maggio'],
        6: ['zogn', 'giugno'],
        7: ['lój', 'luglio'],
        8: ['agóst', 'agosto'],
        9: ['setémber', 'settembre'],
        10: ['otóber', 'ottobre'],
        11: ['novémber', 'novembre'],
        12: ['dicémber', 'dicembre'],
        1: ['znêr', 'genar', 'gennaio'],
        2: ['febrêr', 'febbraio'],
    },
    'tuscan': {
        3: ['marzo'],
        4: ['aprile'],
        5: ['maggio'],
        6: ['giugno'],
        7: ['luglio'],
        8: ['agosto'],
        9: ['settembre'],
        10: ['ottobre'],
        11: ['novembre'],
        12: ['dicembre'],
        1: ['gennaio'],
        2: ['febbraio'],
    },
}

DIALECT_ZODIAC = {
    'venetian': {
        'aries': ['montone', 'ariete', 'arzelin'],
        'taurus': ['toro', 'tauro'],
        'gemini': ['gemeli', 'zemeli', 'zugni'],
        'cancer': ['granzo', 'cangro', 'cancro'],
        'leo': ['lion', 'leon', 'leo'],
        'virgo': ['verzene', 'vergine'],
        'libra': ['balanza', 'libra'],
        'scorpio': ['scorpion', 'escorpion', 'scorpio'],
        'sagittarius': ['sagitario', 'saetario', 'sagittario'],
        'capricorn': ['capricorno'],
        'aquarius': ['aquario'],
        'pisces': ['pesse', 'pessi', 'pissi', 'pesci'],
    },
    'lombard': {
        'aries': ['montun', 'ariete'],
        'taurus': ['tor', 'toro'],
        'gemini': ['gemej', 'gemelli'],
        'cancer': ['cancro', 'granc'],
        'leo': ['leon', 'leo'],
        'virgo': ['vergen', 'vergine'],
        'libra': ['bilancia', 'libra'],
        'scorpio': ['scorpion', 'scorpio'],
        'sagittarius': ['sagitari', 'sagittario'],
        'capricorn': ['capricorno'],
        'aquarius': ['aquario'],
        'pisces': ['pess', 'pesci'],
    },
    'ligurian': {
        'aries': ['monton', 'ariete'],
        'taurus': ['toro'],
        'gemini': ['zemelli', 'gemelli'],
        'cancer': ['granzo', 'cancro'],
        'leo': ['lion', 'leo'],
        'virgo': ['verzene', 'vergine'],
        'libra': ['bilansa', 'libra'],
        'scorpio': ['scorpion', 'scorpio'],
        'sagittarius': ['sagitario', 'sagittario'],
        'capricorn': ['capricorno'],
        'aquarius': ['aquario'],
        'pisces': ['pesse', 'pesci'],
    },
    'emilian': {
        'aries': ['munton', 'ariete'],
        'taurus': ['tor', 'toro'],
        'gemini': ['zemei', 'gemelli'],
        'cancer': ['cancro'],
        'leo': ['leon', 'leo'],
        'virgo': ['vergine'],
        'libra': ['bilancia', 'libra'],
        'scorpio': ['scorpion', 'scorpio'],
        'sagittarius': ['sagitari', 'sagittario'],
        'capricorn': ['capricorno'],
        'aquarius': ['aquario'],
        'pisces': ['pess', 'pesci'],
    },
    'tuscan': {
        'aries': ['ariete', 'montone'],
        'taurus': ['toro'],
        'gemini': ['gemelli'],
        'cancer': ['cancro'],
        'leo': ['leone', 'leo'],
        'virgo': ['vergine'],
        'libra': ['bilancia', 'libra'],
        'scorpio': ['scorpione', 'scorpio'],
        'sagittarius': ['sagittario'],
        'capricorn': ['capricorno'],
        'aquarius': ['acquario'],
        'pisces': ['pesci'],
    },
}

DIALECTS = ['venetian', 'lombard', 'ligurian', 'emilian', 'tuscan']


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ZodiacDialectResult:
    phase: str                          # "54.8"
    experiment: str                     # "zodiac_dialect_decode"
    n_labels_decoded: int
    n_matches: int
    n_correct_folio: int
    n_incorrect_folio: int
    folio_selectivity: float
    consistency_rate: float
    matches_sample: List[Dict]          # first 30 matches
    per_dialect_match_counts: Dict[str, int]
    dialect_scores: Dict[str, float]
    null_mean: float
    null_std: float
    z_score: float
    selectivity: float
    gates: Dict[str, bool]
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Null-test helpers
# ---------------------------------------------------------------------------

VOWELS = 'aeiou'
CONSONANTS = 'bcdfglmnprstv'


def _random_italian_word(length: int, rng: random.Random) -> str:
    word = []
    for i in range(length):
        if i % 2 == 0:
            word.append(rng.choice(CONSONANTS))
        else:
            word.append(rng.choice(VOWELS))
    return ''.join(word)


def _count_null_matches_for_word(fake: str) -> int:
    """Count how many dialect month/zodiac forms match *fake* at ED <= 2."""
    count = 0
    for dialect in DIALECTS:
        for month_num, forms in DIALECT_MONTHS[dialect].items():
            for form in forms:
                if _edit_distance(fake, form) <= 2:
                    count += 1
        for sign_key, forms in DIALECT_ZODIAC[dialect].items():
            for form in forms:
                if _edit_distance(fake, form) <= 2:
                    count += 1
    return count


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_zodiac_dialect() -> None:
    t0 = time.time()
    print("=" * 70)
    print("PHASE 54.8: Zodiac Label Dialect Decode")
    print("=" * 70)

    rd = _results_dir()

    # ------------------------------------------------------------------
    # Step 1: Load zodiac labels
    # ------------------------------------------------------------------
    print("\n  1. Loading zodiac labels ...")

    zodiac_labels: List[Dict] = []
    zodiac_path = os.path.join(rd, 'zodiac_map.json')

    if os.path.exists(zodiac_path):
        with open(zodiac_path) as f:
            zdata = json.load(f)

        # Phase 26 stores folio_map as a list of folio dicts, each with
        # 'labels' containing ZodiacLabel dicts with 'eva_text'.
        if 'folio_map' in zdata:
            for folio_entry in zdata['folio_map']:
                folio_id = folio_entry.get('folio', '')
                for lab in folio_entry.get('labels', []):
                    eva_text = lab.get('eva_text', '')
                    if eva_text.strip():
                        tokens = [t.strip() for t in eva_text.split() if t.strip()]
                        zodiac_labels.append({'folio': folio_id, 'tokens': tokens})
        elif 'labels' in zdata:
            for lab in zdata['labels']:
                zodiac_labels.append({
                    'folio': lab.get('folio', ''),
                    'tokens': lab.get('tokens', lab.get('eva_tokens', [])),
                })
        elif 'folio_labels' in zdata:
            for folio, labels in zdata['folio_labels'].items():
                for lab in labels:
                    zodiac_labels.append({
                        'folio': folio,
                        'tokens': lab if isinstance(lab, list) else [lab],
                    })

    # Fallback: load corpus directly
    if not zodiac_labels:
        print("      (zodiac_map.json not usable, loading corpus directly)")
        from voynich.core.corpus import load_corpus
        corpus = load_corpus(verbose=False)
        for folio_id in FOLIO_ZODIAC_MAP:
            page = corpus.get_page(folio_id)
            if page:
                for locus in page.loci:
                    for token in locus.clean_text.split():
                        if token.strip():
                            zodiac_labels.append({
                                'folio': folio_id,
                                'tokens': [token.strip()],
                            })

    print(f"      {len(zodiac_labels)} zodiac labels loaded")

    # ------------------------------------------------------------------
    # Step 2: Load assignment table and decode labels
    # ------------------------------------------------------------------
    print("\n  2. Decoding labels with Phase 16 assignment ...")

    with open(os.path.join(rd, 'combined_refine.json')) as f:
        assignment = json.load(f)['best_assignment']

    with open(os.path.join(rd, 'modifier_integrate.json')) as f:
        mod_data = json.load(f)

    modifier_chars: Set[str] = set(mod_data.get('modifier_chars', []))
    modifier_rules: Dict[str, str] = {}
    for c in mod_data.get('classifications', []):
        if c.get('final_classification') == 'modifier':
            modifier_rules[c['eva_char']] = c.get('modifier_type', 'silent')

    eva_to_triple = build_eva_to_triple_lookup()

    decoded_labels: List[Dict] = []
    for lab in zodiac_labels:
        folio = lab['folio']
        decoded_tokens = []
        for t in lab['tokens']:
            d = decode_token_modifier_aware(
                t, assignment, eva_to_triple, modifier_chars,
                modifier_rules=modifier_rules,
            )
            decoded_tokens.append(d.lower())

        decoded_concat = ''.join(decoded_tokens)
        decoded_spaced = ' '.join(decoded_tokens)
        decoded_labels.append({
            'folio': folio,
            'eva_tokens': lab['tokens'],
            'decoded_tokens': decoded_tokens,
            'decoded_concat': decoded_concat,
            'decoded_spaced': decoded_spaced,
        })

    print(f"      {len(decoded_labels)} labels decoded")

    # ------------------------------------------------------------------
    # Step 3: Match against dialect month/zodiac names (ED <= 2)
    # ------------------------------------------------------------------
    print("\n  3. Matching decoded labels against dialect names ...")

    matches: List[Dict] = []
    for lab in decoded_labels:
        folio = lab['folio']
        folio_info = FOLIO_ZODIAC_MAP.get(folio)

        # Build candidate text forms: concatenated, spaced, individual tokens
        text_forms = [lab['decoded_concat'], lab['decoded_spaced']] + lab['decoded_tokens']

        for text_form in text_forms:
            if len(text_form) < 3:
                continue

            for dialect in DIALECTS:
                # Check month names
                if folio_info:
                    for month_num, forms in DIALECT_MONTHS[dialect].items():
                        for form in forms:
                            ed = _edit_distance(text_form, form)
                            if ed <= 2:
                                correct_folio = (folio_info['month'] == month_num)
                                matches.append({
                                    'label': text_form,
                                    'folio': folio,
                                    'dialect': dialect,
                                    'match_type': 'month',
                                    'month': month_num,
                                    'form': form,
                                    'ed': ed,
                                    'correct_folio': correct_folio,
                                })

                # Check zodiac names
                if folio_info:
                    sign = folio_info.get('sign')
                    for sign_key, forms in DIALECT_ZODIAC[dialect].items():
                        for form in forms:
                            ed = _edit_distance(text_form, form)
                            if ed <= 2:
                                correct_folio = (sign == sign_key) if sign else False
                                matches.append({
                                    'label': text_form,
                                    'folio': folio,
                                    'dialect': dialect,
                                    'match_type': 'zodiac',
                                    'sign': sign_key,
                                    'form': form,
                                    'ed': ed,
                                    'correct_folio': correct_folio,
                                })

    print(f"      {len(matches)} matches found (ED <= 2)")

    # ------------------------------------------------------------------
    # Step 4: Folio-correctness test
    # ------------------------------------------------------------------
    print("\n  4. Folio-correctness analysis ...")

    correct_matches = [m for m in matches if m['correct_folio']]
    incorrect_matches = [m for m in matches if not m['correct_folio']]
    folio_selectivity = len(correct_matches) / max(1, len(incorrect_matches))

    print(f"      Correct-folio matches:   {len(correct_matches)}")
    print(f"      Incorrect-folio matches: {len(incorrect_matches)}")
    print(f"      Folio selectivity:       {folio_selectivity:.3f}")

    # ------------------------------------------------------------------
    # Step 5: Cross-label consistency
    # ------------------------------------------------------------------
    print("\n  5. Cross-label dialect consistency ...")

    folio_dialect_counts: Dict[str, Counter] = defaultdict(Counter)
    for m in matches:
        folio_dialect_counts[m['folio']][m['dialect']] += 1

    consistent_folios = 0
    total_folios_with_matches = 0
    for folio, counts in folio_dialect_counts.items():
        if sum(counts.values()) >= 2:
            total_folios_with_matches += 1
            top = counts.most_common(1)[0]
            if top[1] / sum(counts.values()) >= 0.5:
                consistent_folios += 1

    consistency_rate = consistent_folios / max(1, total_folios_with_matches)
    print(f"      Folios with >= 2 matches: {total_folios_with_matches}")
    print(f"      Consistent folios:        {consistent_folios}")
    print(f"      Consistency rate:         {consistency_rate:.3f}")

    # ------------------------------------------------------------------
    # Step 6: Dialect scoring
    # ------------------------------------------------------------------
    print("\n  6. Dialect scoring ...")

    dialect_match_counts = Counter(m['dialect'] for m in matches)
    dialect_correct_counts = Counter(m['dialect'] for m in correct_matches)

    total_matches = sum(dialect_match_counts.values())
    dialect_scores: Dict[str, float] = {}
    for d in DIALECTS:
        raw = dialect_match_counts.get(d, 0) / max(1, total_matches)
        if correct_matches:
            correct_bonus = dialect_correct_counts.get(d, 0) * 0.5
            dialect_scores[d] = raw + correct_bonus / max(1, len(correct_matches))
        else:
            dialect_scores[d] = raw

    for d in sorted(dialect_scores, key=dialect_scores.get, reverse=True):
        cnt = dialect_match_counts.get(d, 0)
        corr = dialect_correct_counts.get(d, 0)
        print(f"      {d:12s}  matches={cnt:3d}  correct={corr:2d}  score={dialect_scores[d]:.4f}")

    # ------------------------------------------------------------------
    # Step 7: Null test (1000 iterations)
    # ------------------------------------------------------------------
    print("\n  7. Running null test (1000 iterations, seed=42) ...")

    rng = random.Random(42)

    # For efficiency: only check decoded_concat form per label, and sample
    # up to 50 labels per null trial.
    concat_lengths = [len(lab['decoded_concat']) for lab in decoded_labels
                      if len(lab['decoded_concat']) >= 3]
    n_sample = min(50, len(concat_lengths))

    null_match_counts: List[int] = []
    for trial in range(1000):
        # Sample label lengths (with replacement if needed)
        sampled_lengths = [rng.choice(concat_lengths) for _ in range(n_sample)] \
            if concat_lengths else []
        null_matches = 0
        for length in sampled_lengths:
            fake = _random_italian_word(length, rng)
            null_matches += _count_null_matches_for_word(fake)
        null_match_counts.append(null_matches)

    null_mean = sum(null_match_counts) / len(null_match_counts)
    null_std = (sum((c - null_mean) ** 2 for c in null_match_counts)
                / len(null_match_counts)) ** 0.5

    # Compute comparable real count on same basis: decoded_concat only,
    # sampled same n_sample labels.  For fairness, count all decoded_concat
    # matches (no sampling for the real data — use the full set scaled).
    real_concat_matches = 0
    for lab in decoded_labels:
        text_form = lab['decoded_concat']
        if len(text_form) < 3:
            continue
        real_concat_matches += _count_null_matches_for_word(text_form)

    # Scale real to same sample size as null for fair comparison
    n_eligible = sum(1 for lab in decoded_labels if len(lab['decoded_concat']) >= 3)
    if n_eligible > 0:
        real_scaled = real_concat_matches * n_sample / n_eligible
    else:
        real_scaled = 0.0

    z_score = (real_scaled - null_mean) / null_std if null_std > 0 else 0.0
    selectivity = real_scaled / null_mean if null_mean > 0 else float('inf')

    print(f"      Real matches (concat-only, scaled): {real_scaled:.1f}")
    print(f"      Null mean:      {null_mean:.2f}")
    print(f"      Null std:       {null_std:.2f}")
    print(f"      z-score:        {z_score:.2f}")
    print(f"      Selectivity:    {selectivity:.2f}x")

    # ------------------------------------------------------------------
    # Step 8: Gates
    # ------------------------------------------------------------------
    print("\n  8. Gate evaluation ...")

    gates = {
        'G1_min_matches': len(matches) >= 3,
        'G2_folio_selectivity': folio_selectivity >= 2.0,
        'G3_dialect_cluster': any(v >= 2 for v in dialect_match_counts.values()),
        'G4_null_selectivity': selectivity >= 1.5,
    }

    n_passed = sum(gates.values())
    if n_passed == 4:
        verdict = 'DIALECT_IDENTIFIED'
    elif n_passed >= 2:
        verdict = 'WEAK_SIGNAL'
    else:
        verdict = 'NO_SIGNAL'

    for g, v in gates.items():
        status = "PASS" if v else "FAIL"
        print(f"      {g:30s} {status}")
    print(f"\n      Verdict: {verdict} ({n_passed}/4 gates)")

    # ------------------------------------------------------------------
    # Build and save result
    # ------------------------------------------------------------------
    runtime = round(time.time() - t0, 2)

    result = ZodiacDialectResult(
        phase='54.8',
        experiment='zodiac_dialect_decode',
        n_labels_decoded=len(decoded_labels),
        n_matches=len(matches),
        n_correct_folio=len(correct_matches),
        n_incorrect_folio=len(incorrect_matches),
        folio_selectivity=round(folio_selectivity, 4),
        consistency_rate=round(consistency_rate, 4),
        matches_sample=matches[:30],
        per_dialect_match_counts=dict(dialect_match_counts),
        dialect_scores={k: round(v, 4) for k, v in dialect_scores.items()},
        null_mean=round(null_mean, 4),
        null_std=round(null_std, 4),
        z_score=round(z_score, 4),
        selectivity=round(selectivity, 4),
        gates=gates,
        verdict=verdict,
        runtime_seconds=runtime,
    )

    out_path = os.path.join(rd, 'phase54_zodiac.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2)

    print(f"\n  -> {out_path}  ({runtime:.1f}s)")
    print("=" * 70)
