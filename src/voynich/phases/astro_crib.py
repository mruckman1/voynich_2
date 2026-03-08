"""
Step 26.3 – Full Zodiac Description Crib Search
================================================
Beyond month names, test whether the zodiac text contains the standard
medieval astrological descriptions associated with each sign: element,
quality, ruling planet, body part, and element cycling.

Dependency chain:
    zodiac_map.json (Step 26.1)
    combined_refine.json (Phase 15)
    modifier_integrate.json (Phase 16)
    month_crib.json (Step 26.2)
        → astro_crib.json
"""

import json
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
    ZODIAC_PROPERTIES,
    build_expanded_word_set,
    load_reference_corpus,
)


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
# Astrological vocabulary by domain
# ---------------------------------------------------------------------------

ELEMENT_VOCAB = {
    'ignis': ['ignis', 'igneus', 'ignea', 'fuoco', 'foco'],
    'terra': ['terra', 'terreus', 'terrea', 'terre'],
    'aer': ['aer', 'aereus', 'aerea', 'aria', 'aire'],
    'aqua': ['aqua', 'aqueus', 'aquea', 'acqua', 'eau'],
}

QUALITY_VOCAB = {
    'calidus': ['calidus', 'calida', 'calidum', 'caldo', 'calda', 'chaud'],
    'frigidus': ['frigidus', 'frigida', 'frigidum', 'freddo', 'fredda', 'froid'],
    'siccus': ['siccus', 'sicca', 'siccum', 'secco', 'secca', 'sec'],
    'humidus': ['humidus', 'humida', 'humidum', 'umido', 'umida', 'humide'],
}

PLANET_VOCAB = {
    'mars': ['mars', 'martis', 'marte'],
    'venus': ['venus', 'veneris', 'venere'],
    'mercurius': ['mercurius', 'mercurii', 'mercurio'],
    'luna': ['luna', 'lunae', 'lune'],
    'sol': ['sol', 'solis', 'sole'],
    'iupiter': ['iupiter', 'iovis', 'iuppiter', 'giove', 'jupiter'],
    'saturnus': ['saturnus', 'saturni', 'saturno'],
}

BODY_VOCAB = {
    'caput': ['caput', 'capitis', 'capo', 'testa'],
    'collum': ['collum', 'colli', 'collo'],
    'bracchia': ['bracchia', 'brachia', 'brachium', 'braccio', 'braccia'],
    'pectus': ['pectus', 'pectoris', 'petto'],
    'cor': ['cor', 'cordis', 'cuore'],
    'venter': ['venter', 'ventris', 'ventre'],
    'renes': ['renes', 'renum', 'reni', 'rene'],
    'genitalia': ['genitalia', 'genitalium', 'pudenda'],
    'femora': ['femora', 'femur', 'femoris', 'coscia'],
    'genua': ['genua', 'genu', 'genuum', 'ginocchio'],
    'crura': ['crura', 'crus', 'cruris', 'gamba'],
    'pedes': ['pedes', 'pes', 'pedis', 'piede', 'piedi'],
}

# Element cycling: fire→earth→air→water repeating
ELEMENT_CYCLE = ['ignis', 'terra', 'aer', 'aqua']

# Planet → which signs it rules
PLANET_SIGNS = {
    'mars': ['aries', 'scorpio'],
    'venus': ['taurus', 'libra'],
    'mercurius': ['gemini', 'virgo'],
    'luna': ['cancer'],
    'sol': ['leo'],
    'iupiter': ['sagittarius', 'pisces'],
    'saturnus': ['capricornus', 'aquarius'],
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class VocabHit:
    folio: str
    zodiac_sign: str
    domain: str
    expected_term: str
    found_variant: str
    decoded_word: str
    on_correct_folio: bool


@dataclass
class PlanetCribResult:
    planet: str
    expected_signs: List[str]
    expected_folios: List[str]
    found_on_folios: List[str]
    n_correct: int
    n_incorrect: int


@dataclass
class AstroCribResult:
    timestamp: str
    # Vocabulary presence
    vocab_hits: List[Dict]
    n_vocab_hits: int
    vocab_by_domain: Dict[str, Dict]  # domain -> {total, correct_folio, wrong_folio}
    n_domains_with_hits: int
    correct_folio_rate: float
    wrong_folio_rate: float
    vocab_selectivity: float
    # Planet cribs
    planet_results: List[Dict]
    n_planet_matches: int
    # Body part cribs
    body_hits: List[Dict]
    n_body_correct: int
    # Element cycling
    element_cycle_score: float
    element_cycle_detail: List[Dict]
    # Null control
    null_vocab_rate: float
    null_vocab_std: float
    # Verdict
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _decode_folio_text(
    page,
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    modifier_chars: Set[str],
) -> List[str]:
    """Decode all tokens on a folio page into phonetic words."""
    decoded = []
    if page is None:
        return decoded
    tokens = page.all_tokens
    for token in tokens:
        dec = decode_token_modifier_aware(
            token, assignment, eva_to_triple, modifier_chars
        )
        decoded.append(dec.lower())
    return decoded


def _search_vocab_in_decoded(
    decoded_words: List[str],
    vocab_variants: List[str],
) -> List[str]:
    """Search for vocabulary terms in decoded word list.

    Returns list of matched variants (substring matching in individual words
    and in concatenated adjacent pairs).
    """
    found = []
    for variant in vocab_variants:
        variant_lower = variant.lower()
        # Direct word match
        for dw in decoded_words:
            if variant_lower == dw or variant_lower in dw:
                found.append(variant)
                break
        else:
            # Adjacent pair concatenation match
            for i in range(len(decoded_words) - 1):
                pair = decoded_words[i] + decoded_words[i + 1]
                if variant_lower in pair:
                    found.append(variant)
                    break
    return found


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_astro_crib() -> None:
    t0 = time.time()
    print("=" * 70)
    print("STEP 26.3: Full Zodiac Description Crib Search")
    print("=" * 70)

    rd = _results_dir()

    # Load dependencies
    zodiac_data = _load_json(os.path.join(rd, 'zodiac_map.json'))
    if not zodiac_data:
        print("  [SKIP] zodiac_map.json not found — run zodiac-map first")
        return

    refine_data = _load_json(os.path.join(rd, 'combined_refine.json'))
    if not refine_data:
        print("  [SKIP] combined_refine.json not found")
        return

    mod_data = _load_json(os.path.join(rd, 'modifier_integrate.json'))
    if not mod_data:
        print("  [SKIP] modifier_integrate.json not found")
        return

    assignment = refine_data.get('best_assignment', {})
    modifier_chars = set(mod_data.get('modifier_chars', []))

    corpus = load_corpus(verbose=False)
    eva_to_triple = build_eva_to_triple_lookup()

    folio_map = zodiac_data.get('folio_map', [])

    # Build folio -> sign mapping
    folio_sign: Dict[str, str] = {}
    for finfo in folio_map:
        folio_sign[finfo['folio']] = finfo['zodiac_sign']

    # Decode all zodiac folios
    print(f"\n  1. Decoding {len(folio_map)} zodiac folios ...")
    decoded_folios: Dict[str, List[str]] = {}
    for finfo in folio_map:
        folio = finfo['folio']
        page = corpus.get_page(folio)
        decoded_folios[folio] = _decode_folio_text(
            page, assignment, eva_to_triple, modifier_chars
        )
        n_words = len(decoded_folios[folio])
        print(f"      {folio}: {n_words} decoded words")

    # -------------------------------------------------------------------
    # Vocabulary presence test
    # -------------------------------------------------------------------
    print(f"\n  2. Vocabulary presence test (4 domains) ...")

    all_vocab_hits: List[VocabHit] = []
    domain_stats: Dict[str, Dict] = {}

    domains = {
        'element': ELEMENT_VOCAB,
        'quality': QUALITY_VOCAB,
        'planet': PLANET_VOCAB,
        'body': BODY_VOCAB,
    }

    for domain_name, domain_vocab in domains.items():
        correct = 0
        wrong = 0
        total = 0

        for finfo in folio_map:
            folio = finfo['folio']
            sign = finfo['zodiac_sign']
            words = decoded_folios.get(folio, [])

            if sign not in ZODIAC_PROPERTIES:
                continue

            props = ZODIAC_PROPERTIES[sign]

            # Determine expected term for this domain
            if domain_name == 'element':
                expected_key = props['element']
            elif domain_name == 'quality':
                # Quality has two terms, check both
                qual_parts = props['quality'].split()
                for qp in qual_parts:
                    if qp in domain_vocab:
                        variants = domain_vocab[qp]
                        found = _search_vocab_in_decoded(words, variants)
                        for fv in found:
                            on_correct = True  # quality applies to correct sign
                            all_vocab_hits.append(VocabHit(
                                folio=folio, zodiac_sign=sign,
                                domain=domain_name, expected_term=qp,
                                found_variant=fv, decoded_word=fv,
                                on_correct_folio=on_correct,
                            ))
                            total += 1
                            correct += 1
                continue
            elif domain_name == 'planet':
                expected_key = props['planet']
            elif domain_name == 'body':
                expected_key = props['body']
            else:
                continue

            if expected_key not in domain_vocab:
                continue

            variants = domain_vocab[expected_key]
            found_on_correct = _search_vocab_in_decoded(words, variants)

            for fv in found_on_correct:
                all_vocab_hits.append(VocabHit(
                    folio=folio, zodiac_sign=sign,
                    domain=domain_name, expected_term=expected_key,
                    found_variant=fv, decoded_word=fv,
                    on_correct_folio=True,
                ))
                correct += 1
                total += 1

            # Also check if this term appears on WRONG folios
            for other_finfo in folio_map:
                other_folio = other_finfo['folio']
                if other_folio == folio:
                    continue
                other_words = decoded_folios.get(other_folio, [])
                found_on_wrong = _search_vocab_in_decoded(other_words, variants)
                for fv in found_on_wrong:
                    all_vocab_hits.append(VocabHit(
                        folio=other_folio, zodiac_sign=other_finfo['zodiac_sign'],
                        domain=domain_name, expected_term=expected_key,
                        found_variant=fv, decoded_word=fv,
                        on_correct_folio=False,
                    ))
                    wrong += 1
                    total += 1

        domain_stats[domain_name] = {
            'total': total,
            'correct_folio': correct,
            'wrong_folio': wrong,
            'rate_correct': round(correct / max(total, 1), 4),
        }
        print(f"      {domain_name:10s}: correct={correct}, wrong={wrong}, total={total}")

    n_domains_with_hits = sum(
        1 for ds in domain_stats.values() if ds['correct_folio'] > 0
    )
    total_correct = sum(ds['correct_folio'] for ds in domain_stats.values())
    total_wrong = sum(ds['wrong_folio'] for ds in domain_stats.values())
    total_all = sum(ds['total'] for ds in domain_stats.values())
    correct_rate = total_correct / max(total_all, 1)
    wrong_rate = total_wrong / max(total_all, 1)

    # -------------------------------------------------------------------
    # Planet name crib
    # -------------------------------------------------------------------
    print(f"\n  3. Planet name crib (7 planets × 2 ruling signs each) ...")

    planet_results: List[PlanetCribResult] = []
    n_planet_matches = 0

    for planet, expected_signs in PLANET_SIGNS.items():
        expected_folios = [
            f for f, s in folio_sign.items() if s in expected_signs
        ]
        variants = PLANET_VOCAB.get(planet, [planet])

        found_folios = []
        for folio, words in decoded_folios.items():
            found = _search_vocab_in_decoded(words, variants)
            if found:
                found_folios.append(folio)

        n_correct = len(set(found_folios) & set(expected_folios))
        n_incorrect = len(set(found_folios) - set(expected_folios))

        planet_results.append(PlanetCribResult(
            planet=planet,
            expected_signs=expected_signs,
            expected_folios=expected_folios,
            found_on_folios=found_folios,
            n_correct=n_correct,
            n_incorrect=n_incorrect,
        ))

        if n_correct > 0:
            n_planet_matches += 1

        status = "MATCH" if n_correct > 0 else "no match"
        print(f"      {planet:12s}: expected on {expected_folios}, "
              f"found on {found_folios} → {status}")

    # -------------------------------------------------------------------
    # Body part crib
    # -------------------------------------------------------------------
    print(f"\n  4. Body part crib (12 body parts × 1 sign each) ...")

    body_hits_correct = 0
    body_hits_list: List[Dict] = []

    for sign, props in ZODIAC_PROPERTIES.items():
        body_key = props['body']
        if body_key not in BODY_VOCAB:
            continue
        variants = BODY_VOCAB[body_key]

        # Find folios for this sign
        sign_folios = [f for f, s in folio_sign.items() if s == sign]

        for folio in sign_folios:
            words = decoded_folios.get(folio, [])
            found = _search_vocab_in_decoded(words, variants)
            if found:
                body_hits_correct += 1
                body_hits_list.append({
                    'sign': sign,
                    'body_part': body_key,
                    'folio': folio,
                    'found_variants': found,
                })
                print(f"      {sign:14s} ({body_key:12s}) → found on {folio}: {found}")

    if body_hits_correct == 0:
        print(f"      No body part matches found on correct folios")

    # -------------------------------------------------------------------
    # Element cycling test
    # -------------------------------------------------------------------
    print(f"\n  5. Element cycling test (period-4 pattern) ...")

    # Order folios by zodiac sequence
    sign_order = [
        'pisces', 'aries', 'aries', 'taurus', 'taurus',
        'gemini', 'cancer', 'leo', 'virgo', 'libra',
        'scorpio', 'sagittarius',
    ]  # matches folio_map order

    cycle_detail: List[Dict] = []
    n_correct_element = 0
    n_total_tested = 0

    for finfo in folio_map:
        folio = finfo['folio']
        sign = finfo['zodiac_sign']
        if sign not in ZODIAC_PROPERTIES:
            continue

        expected_element = ZODIAC_PROPERTIES[sign]['element']
        expected_variants = ELEMENT_VOCAB.get(expected_element, [expected_element])

        words = decoded_folios.get(folio, [])
        found = _search_vocab_in_decoded(words, expected_variants)

        is_correct = len(found) > 0
        if is_correct:
            n_correct_element += 1
        n_total_tested += 1

        cycle_detail.append({
            'folio': folio,
            'sign': sign,
            'expected_element': expected_element,
            'found': found,
            'correct': is_correct,
        })

    cycle_score = n_correct_element / max(n_total_tested, 1)
    print(f"      Element cycling: {n_correct_element}/{n_total_tested} correct "
          f"({cycle_score:.1%})")

    # -------------------------------------------------------------------
    # Null control: shuffle zodiac properties
    # -------------------------------------------------------------------
    print(f"\n  6. Null control (50 random property shuffles) ...")

    rng = random.Random(42)
    null_rates: List[float] = []

    signs_list = list(ZODIAC_PROPERTIES.keys())

    for _ in range(50):
        shuffled_signs = signs_list[:]
        rng.shuffle(shuffled_signs)
        sign_remap = dict(zip(signs_list, shuffled_signs))

        n_null_hits = 0
        n_null_total = 0

        for finfo in folio_map:
            folio = finfo['folio']
            sign = finfo['zodiac_sign']
            remapped = sign_remap.get(sign, sign)
            if remapped not in ZODIAC_PROPERTIES:
                continue

            props = ZODIAC_PROPERTIES[remapped]
            words = decoded_folios.get(folio, [])

            for domain_name, domain_vocab in domains.items():
                if domain_name == 'quality':
                    continue
                if domain_name == 'element':
                    key = props['element']
                elif domain_name == 'planet':
                    key = props['planet']
                elif domain_name == 'body':
                    key = props['body']
                else:
                    continue

                if key in domain_vocab:
                    found = _search_vocab_in_decoded(words, domain_vocab[key])
                    if found:
                        n_null_hits += 1
                    n_null_total += 1

        null_rates.append(n_null_hits / max(n_null_total, 1))

    null_mean = sum(null_rates) / len(null_rates) if null_rates else 0
    null_std = (sum((r - null_mean) ** 2 for r in null_rates) / len(null_rates)) ** 0.5 if null_rates else 0
    vocab_selectivity = correct_rate / null_mean if null_mean > 0 else float('inf')

    print(f"      Null vocab hit rate: {null_mean:.4f} ± {null_std:.4f}")
    print(f"      Real correct rate:   {correct_rate:.4f}")
    print(f"      Vocab selectivity:   {vocab_selectivity:.2f}×")

    # -------------------------------------------------------------------
    # Verdict
    # -------------------------------------------------------------------
    findings = []
    if n_domains_with_hits >= 2:
        findings.append(f"{n_domains_with_hits}/4 domains with hits")
    if n_planet_matches >= 2:
        findings.append(f"{n_planet_matches} planet name matches")
    if body_hits_correct >= 3:
        findings.append(f"{body_hits_correct} body part matches")
    if cycle_score > 0.3:
        findings.append(f"element cycling {cycle_score:.0%}")

    if findings:
        verdict = f"SIGNAL: {'; '.join(findings)}. Selectivity {vocab_selectivity:.2f}×."
    else:
        verdict = (f"WEAK: {n_domains_with_hits}/4 domains, "
                   f"{n_planet_matches} planets, "
                   f"{body_hits_correct} body parts, "
                   f"cycle {cycle_score:.0%}.")

    print(f"\n  7. Verdict: {verdict}")

    result = AstroCribResult(
        timestamp=time.strftime('%Y-%m-%dT%H:%M:%S'),
        vocab_hits=[_convert(asdict(vh)) for vh in all_vocab_hits],
        n_vocab_hits=len(all_vocab_hits),
        vocab_by_domain=domain_stats,
        n_domains_with_hits=n_domains_with_hits,
        correct_folio_rate=round(correct_rate, 4),
        wrong_folio_rate=round(wrong_rate, 4),
        vocab_selectivity=round(vocab_selectivity, 4),
        planet_results=[_convert(asdict(pr)) for pr in planet_results],
        n_planet_matches=n_planet_matches,
        body_hits=body_hits_list,
        n_body_correct=body_hits_correct,
        element_cycle_score=round(cycle_score, 4),
        element_cycle_detail=cycle_detail,
        null_vocab_rate=round(null_mean, 4),
        null_vocab_std=round(null_std, 4),
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'astro_crib.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2)

    print(f"\n  → {out_path}")
