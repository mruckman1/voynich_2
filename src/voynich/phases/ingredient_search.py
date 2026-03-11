"""
Step 41.11 – Ingredient Search in Content Zones
=================================================
Match candidate ingredient tokens from CONTENT zones against Venetian
pharmaceutical references: the Anonimo Veneziano culinary/pharma text,
a curated medieval ingredient list, and the expanded Venetian word set.

Dependency chain:
    inter_formula_tokens.json  (Step 41.10)
    venetian_forms.json        (Step 40.5)
    data/reference/italian/anonimo_veneziano.txt
        → ingredient_search.json  (this step)
"""

import json
import os
import re
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple

from voynich.core._paths import results_dir as _results_dir, data_dir as _data_dir


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _convert(obj: Any) -> Any:
    if hasattr(obj, '__dataclass_fields__'):
        from dataclasses import asdict
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


def _safe_load(path: str) -> Dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


# ---------------------------------------------------------------------------
# Constants: medieval pharmaceutical ingredients
# ---------------------------------------------------------------------------

MEDIEVAL_INGREDIENTS: Dict[str, str] = {
    'aqua': 'water', 'vino': 'wine', 'aceto': 'vinegar',
    'miele': 'honey', 'olio': 'oil', 'sale': 'salt',
    'pevere': 'pepper', 'canela': 'cinnamon', 'zenzero': 'ginger',
    'rosa': 'rose', 'salvia': 'sage', 'menta': 'mint',
    'radice': 'root', 'foglia': 'leaf', 'fiore': 'flower',
    'seme': 'seed', 'scorza': 'bark', 'erba': 'herb',
    'cera': 'wax', 'grasso': 'fat', 'ovo': 'egg',
    'alume': 'alum', 'zucaro': 'sugar', 'latte': 'milk',
    # Venetian/medieval variants
    'aqua': 'water', 'vin': 'wine', 'aseo': 'vinegar',
    'miel': 'honey', 'ogio': 'oil', 'sal': 'salt',
    'pever': 'pepper', 'cenamo': 'cinnamon', 'zenzevro': 'ginger',
    'roxa': 'rose', 'menta': 'mint', 'savia': 'sage',
    'raixa': 'root', 'foia': 'leaf', 'fior': 'flower',
    'semenza': 'seed', 'scorza': 'bark', 'herba': 'herb',
    'zera': 'wax', 'graso': 'fat', 'ovo': 'egg',
    'zucharo': 'sugar', 'lacte': 'milk', 'late': 'milk',
    # Common recipe ingredients from Anonimo Veneziano
    'mandole': 'almonds', 'amido': 'starch', 'pignoli': 'pine nuts',
    'garofali': 'cloves', 'zafarano': 'saffron', 'lardo': 'lard',
    'agresta': 'verjuice', 'specie': 'spices', 'cipole': 'onions',
    'polastri': 'chickens', 'carne': 'meat', 'pesse': 'fish',
    'riso': 'rice', 'farina': 'flour', 'buro': 'butter',
    'persemolo': 'parsley', 'petrosello': 'parsley', 'noce': 'walnut',
    'melegette': 'grains of paradise',
}


# ---------------------------------------------------------------------------
# Core: reference building
# ---------------------------------------------------------------------------

def _load_anonimo_vocab(data_dir: str) -> Set[str]:
    """Load and tokenize the Anonimo Veneziano text."""
    path = os.path.join(data_dir, 'reference', 'italian', 'anonimo_veneziano.txt')
    if not os.path.exists(path):
        return set()
    with open(path, encoding='utf-8', errors='replace') as f:
        text = f.read()
    # Extract words: lowercase, strip punctuation
    words = set()
    for token in re.findall(r"[a-zA-ZàèéìòùÀÈÉÌÒÙ']+", text):
        w = token.lower().strip("'")
        if len(w) >= 2:
            words.add(w)
    return words


def _edit_distance_1(word: str) -> Set[str]:
    """Generate all strings at edit distance 1 from the given word."""
    letters = 'abcdefghijklmnopqrstuvwxyz'
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = {a + b[1:] for a, b in splits if b}
    transposes = {a + b[1] + b[0] + b[2:] for a, b in splits if len(b) > 1}
    replaces = {a + c + b[1:] for a, b in splits if b for c in letters}
    inserts = {a + c + b for a, b in splits for c in letters}
    return deletes | transposes | replaces | inserts


def _match_candidate(
    decoded: str,
    medieval_ingredients: Dict[str, str],
    anonimo_vocab: Set[str],
    venetian_extended: Set[str],
) -> Dict:
    """Match a single decoded token against all reference sets."""
    result: Dict[str, Any] = {
        'decoded': decoded,
        'exact_medieval': None,
        'exact_anonimo': False,
        'exact_venetian': False,
        'ed1_medieval': [],
        'ed1_anonimo': [],
        'ed1_venetian': [],
        'best_match': None,
        'best_match_source': None,
        'best_english': None,
    }

    # Exact matches
    if decoded in medieval_ingredients:
        result['exact_medieval'] = medieval_ingredients[decoded]
        result['best_match'] = decoded
        result['best_match_source'] = 'medieval_ingredient'
        result['best_english'] = medieval_ingredients[decoded]
        return result

    if decoded in anonimo_vocab:
        result['exact_anonimo'] = True
        result['best_match'] = decoded
        result['best_match_source'] = 'anonimo_veneziano'
        # Check if also a known ingredient
        if decoded in medieval_ingredients:
            result['best_english'] = medieval_ingredients[decoded]
        return result

    if decoded in venetian_extended:
        result['exact_venetian'] = True
        result['best_match'] = decoded
        result['best_match_source'] = 'venetian_extended'
        return result

    # Edit-distance-1 matches
    ed1_variants = _edit_distance_1(decoded)

    # Check medieval ingredients at ed1
    for variant in ed1_variants:
        if variant in medieval_ingredients:
            result['ed1_medieval'].append({
                'variant': variant,
                'english': medieval_ingredients[variant],
            })

    # Check anonimo at ed1
    anonimo_ed1 = sorted(ed1_variants & anonimo_vocab)[:5]
    result['ed1_anonimo'] = anonimo_ed1

    # Check venetian extended at ed1
    venetian_ed1 = sorted(ed1_variants & venetian_extended)[:5]
    result['ed1_venetian'] = venetian_ed1

    # Pick best match
    if result['ed1_medieval']:
        best = result['ed1_medieval'][0]
        result['best_match'] = best['variant']
        result['best_match_source'] = 'medieval_ingredient_ed1'
        result['best_english'] = best['english']
    elif result['ed1_anonimo']:
        result['best_match'] = result['ed1_anonimo'][0]
        result['best_match_source'] = 'anonimo_veneziano_ed1'
    elif result['ed1_venetian']:
        result['best_match'] = result['ed1_venetian'][0]
        result['best_match_source'] = 'venetian_extended_ed1'

    return result


def _compare_ingredients_across_zones(
    zone_matches: List[Dict],
) -> Dict:
    """Compare ingredient matches across content zones."""
    # Collect all matched ingredients per zone
    zone_ingredients: List[Tuple[int, List[str]]] = []
    for zm in zone_matches:
        ingredients = []
        for m in zm.get('matches', []):
            if m.get('best_match'):
                ingredients.append(m['best_match'])
        zone_ingredients.append((zm['zone_id'], ingredients))

    # Count ingredient frequencies across zones
    ingredient_freq: Counter = Counter()
    for _, ingredients in zone_ingredients:
        for ing in ingredients:
            ingredient_freq[ing] += 1

    # Find ingredients appearing in multiple zones
    cross_zone = {ing: count for ing, count in ingredient_freq.items() if count >= 2}

    # Find zone-unique ingredients
    all_ingredients = set()
    for _, ingredients in zone_ingredients:
        all_ingredients.update(ingredients)

    zone_unique = {}
    for zid, ingredients in zone_ingredients:
        others = set()
        for zid2, ing2 in zone_ingredients:
            if zid2 != zid:
                others.update(ing2)
        unique = sorted(set(ingredients) - others)
        if unique:
            zone_unique[str(zid)] = unique

    return {
        'n_unique_ingredients': len(all_ingredients),
        'ingredient_frequency': dict(ingredient_freq.most_common()),
        'cross_zone_ingredients': cross_zone,
        'zone_unique_ingredients': zone_unique,
        'n_zones_with_matches': sum(
            1 for _, ings in zone_ingredients if ings
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_ingredient_search() -> None:
    """Step 41.11: Search for ingredient matches in content zones."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 41.11: Ingredient Search in Content Zones")
    print("=" * 70)

    rd = _results_dir()
    dd = _data_dir()

    # ── 1. Load inputs ──
    print("\n  1. Loading inputs ...")
    inter_data = _safe_load(os.path.join(rd, 'inter_formula_tokens.json'))
    ven_forms = _safe_load(os.path.join(rd, 'venetian_forms.json'))

    if not inter_data:
        print("    ERROR: inter_formula_tokens.json not found.")
        output = {
            'error': 'inter_formula_tokens.json not found',
            'runtime_seconds': 0.0,
        }
        out_path = os.path.join(rd, 'ingredient_search.json')
        with open(out_path, 'w') as f:
            json.dump(output, f, indent=2)
        return

    ingredient_candidates = inter_data.get('ingredient_candidates', [])
    print(f"    Ingredient candidate zones: {len(ingredient_candidates)}")

    # ── 2. Build reference sets ──
    print("\n  2. Building reference sets ...")

    # Anonimo Veneziano
    anonimo_vocab = _load_anonimo_vocab(dd)
    print(f"    Anonimo Veneziano vocab: {len(anonimo_vocab)} words")

    # Medieval ingredients
    print(f"    Medieval ingredient list: {len(MEDIEVAL_INGREDIENTS)} entries")

    # Venetian extended set
    venetian_extended: Set[str] = set()
    ven_ext_list = ven_forms.get('venetian_extended_set', [])
    if isinstance(ven_ext_list, list):
        venetian_extended = set(ven_ext_list)
    elif isinstance(ven_ext_list, dict):
        venetian_extended = set(ven_ext_list.keys())
    # Also try the venetian_words field
    if not venetian_extended:
        ven_words = ven_forms.get('venetian_words', [])
        if isinstance(ven_words, list):
            venetian_extended = set(ven_words)
    # Also build from generated forms
    gen_forms = ven_forms.get('generated_forms', {})
    if isinstance(gen_forms, dict):
        for forms in gen_forms.values():
            if isinstance(forms, list):
                venetian_extended.update(forms)
    print(f"    Venetian extended set: {len(venetian_extended)} words")

    # ── 3. Match ingredient candidates ──
    print("\n  3. Matching ingredient candidates ...")
    zone_match_results = []
    total_exact = 0
    total_ed1 = 0
    total_no_match = 0

    for ic in ingredient_candidates:
        zone_id = ic['zone_id']
        all_decoded = ic.get('all_zone_decoded', [])
        head_decoded = [h['decoded'] for h in ic.get('head_tokens', [])]

        # Match all tokens in the zone, but prioritize head tokens
        matches = []
        for decoded in all_decoded:
            if not decoded or len(decoded) < 2:
                continue
            m = _match_candidate(
                decoded, MEDIEVAL_INGREDIENTS, anonimo_vocab, venetian_extended,
            )
            matches.append(m)

            if m.get('exact_medieval') or m.get('exact_anonimo') or m.get('exact_venetian'):
                total_exact += 1
            elif m.get('best_match'):
                total_ed1 += 1
            else:
                total_no_match += 1

        # Head token matches (first 3 tokens, potential ingredient names)
        head_matches = []
        for decoded in head_decoded:
            if not decoded or len(decoded) < 2:
                continue
            hm = _match_candidate(
                decoded, MEDIEVAL_INGREDIENTS, anonimo_vocab, venetian_extended,
            )
            head_matches.append(hm)

        zone_match_results.append({
            'zone_id': zone_id,
            'ingredient_candidate': ic.get('ingredient_candidate', ''),
            'n_tokens': len(all_decoded),
            'matches': matches,
            'head_matches': head_matches,
            'n_exact': sum(
                1 for m in matches
                if m.get('exact_medieval') or m.get('exact_anonimo')
                or m.get('exact_venetian')
            ),
            'n_ed1': sum(
                1 for m in matches
                if m.get('best_match') and not (
                    m.get('exact_medieval') or m.get('exact_anonimo')
                    or m.get('exact_venetian')
                )
            ),
        })

        # Print summary for this zone
        n_exact = zone_match_results[-1]['n_exact']
        n_ed1 = zone_match_results[-1]['n_ed1']
        cand = ic.get('ingredient_candidate', '???')
        print(f"    Zone {zone_id:2d}: candidate='{cand}', "
              f"{n_exact} exact, {n_ed1} ed1, "
              f"{len(all_decoded)} total tokens")

        # Show interesting matches
        for m in matches:
            if m.get('best_match'):
                src = m.get('best_match_source', '')
                eng = m.get('best_english', '')
                eng_str = f" = {eng}" if eng else ''
                print(f"      {m['decoded']:15s} -> {m['best_match']}"
                      f" ({src}){eng_str}")

    # ── 4. Cross-zone comparison ──
    print("\n  4. Comparing ingredients across zones ...")
    cross_zone = _compare_ingredients_across_zones(zone_match_results)
    print(f"    Unique ingredients found: {cross_zone['n_unique_ingredients']}")
    print(f"    Zones with matches: {cross_zone['n_zones_with_matches']}")
    print(f"    Cross-zone ingredients: {len(cross_zone['cross_zone_ingredients'])}")

    for ing, count in sorted(cross_zone['cross_zone_ingredients'].items(),
                              key=lambda x: -x[1])[:10]:
        eng = MEDIEVAL_INGREDIENTS.get(ing, '')
        eng_str = f" ({eng})" if eng else ''
        print(f"      {ing}{eng_str}: {count} zones")

    # ── 5. Summary statistics ──
    print("\n  5. Summary ...")
    all_matched_words = set()
    all_medieval_hits = []
    all_anonimo_hits = []
    for zm in zone_match_results:
        for m in zm['matches']:
            if m.get('exact_medieval'):
                all_medieval_hits.append(m['decoded'])
                all_matched_words.add(m['decoded'])
            if m.get('exact_anonimo'):
                all_anonimo_hits.append(m['decoded'])
                all_matched_words.add(m['decoded'])
            if m.get('exact_venetian'):
                all_matched_words.add(m['decoded'])

    print(f"    Total exact matches: {total_exact}")
    print(f"    Total ed1 matches:   {total_ed1}")
    print(f"    Total no match:      {total_no_match}")
    print(f"    Medieval ingredient hits: {len(all_medieval_hits)}")
    print(f"    Anonimo vocab hits: {len(all_anonimo_hits)}")

    # ── 6. Save ──
    elapsed = time.time() - t0

    output = {
        'n_candidate_zones': len(ingredient_candidates),
        'n_medieval_ingredients': len(MEDIEVAL_INGREDIENTS),
        'n_anonimo_vocab': len(anonimo_vocab),
        'n_venetian_extended': len(venetian_extended),
        'total_exact_matches': total_exact,
        'total_ed1_matches': total_ed1,
        'total_no_match': total_no_match,
        'zone_match_results': zone_match_results,
        'cross_zone_comparison': cross_zone,
        'medieval_ingredient_hits': sorted(set(all_medieval_hits)),
        'anonimo_vocab_hits': sorted(set(all_anonimo_hits)),
        'all_matched_words': sorted(all_matched_words),
        'interpretation': (
            f"Searched {len(ingredient_candidates)} content zones. "
            f"{total_exact} exact matches, {total_ed1} edit-distance-1 matches. "
            f"{cross_zone['n_unique_ingredients']} unique ingredients found. "
            f"{cross_zone['n_zones_with_matches']}/{len(ingredient_candidates)} zones "
            f"have at least one ingredient match."
        ),
        'runtime_seconds': round(elapsed, 1),
    }

    out_path = os.path.join(rd, 'ingredient_search.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(output), f, indent=2)
    print(f"\n  Saved -> {out_path} ({elapsed:.1f}s)")
