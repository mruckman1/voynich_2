"""
Phase 65, Step 5: Recipe Template-Constrained Segmentation
===========================================================
Use pharmaceutical recipe templates and vocabulary to segment
recipe character streams via dictionary-constrained DP.

Dependency chain:
    results/p65_decoded_stream.json  (Step 65.1)
    results/cvc_recipes.json         (Phase 59)
        -> results/p65_recipe_segment.json
"""

import json
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from voynich.core._paths import results_dir as _results_dir
from voynich.core.reference import (
    PHARMACEUTICAL_VOCABULARY,
    load_reference_corpus,
)


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
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class RecipeSegmentResult:
    phase: str = "65"
    step: str = "65.5"
    experiment: str = "recipe_segment"
    # Dictionary info
    pharma_dict_size: int = 0
    # Results
    n_recipes: int = 0
    n_matched: int = 0
    match_rate: float = 0.0
    mean_coverage: float = 0.0
    mean_word_length: float = 0.0
    mean_per_slot_ed: float = 0.0
    n_fully_readable: int = 0
    n_distinct_ingredients: int = 0
    n_template_types: int = 0
    template_distribution: Dict[str, int] = field(default_factory=dict)
    ingredient_distribution: List[Dict] = field(default_factory=list)
    top_recipes: List[Dict] = field(default_factory=list)
    sample_segmentations: List[Dict] = field(default_factory=list)
    # Null comparison
    null_coverage: float = 0.0
    selectivity: float = 0.0
    # Gates
    g1_match_rate: bool = False
    g2_per_slot_ed: bool = False
    g3_ingredients: bool = False
    g4_template_types: bool = False
    g5_fully_readable: bool = False
    gates_passed: int = 0
    gate_passed: bool = False
    verdict: str = ""
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Edit distance
# ---------------------------------------------------------------------------

def _edit_distance(a: str, b: str) -> int:
    """Levenshtein edit distance."""
    if len(a) < len(b):
        return _edit_distance(b, a)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            cost = 0 if ca == cb else 1
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + cost))
        prev = curr
    return prev[-1]


# ---------------------------------------------------------------------------
# Pharmaceutical dictionary
# ---------------------------------------------------------------------------

def _build_pharma_dict() -> Set[str]:
    """Build a comprehensive pharmaceutical dictionary for recipe matching."""
    words: Set[str] = set()

    # From PHARMACEUTICAL_VOCABULARY
    for category, word_list in PHARMACEUTICAL_VOCABULARY.items():
        for w in word_list:
            words.add(w.lower())

    # Common Latin pharmaceutical words and their inflections
    extra_words = [
        # Verbs (imperative + indicative)
        'cola', 'colar', 'colas', 'coletur', 'colatur',
        'tere', 'teres', 'teratur', 'terantur',
        'misce', 'misces', 'misceantur',
        'recipe', 'accipe', 'accipias',
        'coque', 'coques', 'coquatur', 'decoque',
        'adde', 'addas', 'addatur',
        'pone', 'ponas', 'ponatur',
        'bibe', 'bibat', 'bibatur',
        'fac', 'fiat', 'fiant',
        'da', 'detur', 'dentur',
        'solve', 'solvatur',
        # Ingredients (various cases)
        'senna', 'sennam', 'sennae',
        'corallum', 'coralli', 'corallis',
        'radicem', 'radicis', 'radice', 'radicum',
        'stercora', 'stercoris', 'stercus',
        'mel', 'melle', 'mellis',
        'aqua', 'aquam', 'aquae',
        'cera', 'ceram', 'cerae',
        'sal', 'salem', 'salis',
        'oleum', 'olei', 'oleo',
        'vinum', 'vini', 'vino',
        'acetum', 'aceti', 'aceto',
        # Prepositions
        'in', 'cum', 'con', 'per', 'de', 'ad', 'ex', 'pro', 'super',
        # Quantities
        'dragmam', 'unciam', 'libram', 'partem', 'dimidium',
        'tres', 'duas', 'unam', 'duas',
        # Qualities
        'calida', 'frigida', 'sicca', 'humida',
        'subtiliter', 'bene', 'fortiter', 'diligenter',
        # Body
        'caput', 'cor', 'stomachum', 'oculus',
        # Structure words
        'et', 'est', 'sunt', 'habet', 'valet',
        'contra', 'prodest', 'iuvat',
        # Compound prefixes
        'dia', 'anti',
        # Common in recipes
        'pulvis', 'emplastrum', 'unguentum', 'sirupus',
        'decoctum', 'infusum', 'electuarium', 'pilula',
        'pannum', 'setam', 'lacte', 'brodio', 'iure',
    ]
    for w in extra_words:
        words.add(w.lower())

    # Also add from Latin reference corpus (most frequent)
    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    freq = Counter(w.lower() for w in ref_corpus.get_combined_tokens('latin')
                   if len(w) >= 2 and w.isalpha())
    for w, _ in freq.most_common(5000):
        words.add(w)

    return words


# ---------------------------------------------------------------------------
# Recipe templates
# ---------------------------------------------------------------------------

RECIPE_TEMPLATES = [
    {
        'name': 'simple_preparation',
        'slots': [
            {'role': 'VERB', 'min_len': 2, 'max_len': 8},
            {'role': 'INGREDIENT', 'min_len': 3, 'max_len': 12},
            {'role': 'PREP', 'min_len': 2, 'max_len': 4},
            {'role': 'MEDIUM', 'min_len': 3, 'max_len': 8},
        ],
    },
    {
        'name': 'grinding_recipe',
        'slots': [
            {'role': 'VERB', 'min_len': 3, 'max_len': 8},
            {'role': 'INGREDIENT', 'min_len': 3, 'max_len': 12},
            {'role': 'QUALIFIER', 'min_len': 3, 'max_len': 10},
        ],
    },
    {
        'name': 'mixture',
        'slots': [
            {'role': 'VERB', 'min_len': 3, 'max_len': 8},
            {'role': 'INGREDIENT_1', 'min_len': 3, 'max_len': 12},
            {'role': 'PREP', 'min_len': 2, 'max_len': 4},
            {'role': 'INGREDIENT_2', 'min_len': 3, 'max_len': 12},
        ],
    },
    {
        'name': 'dosage',
        'slots': [
            {'role': 'VERB', 'min_len': 2, 'max_len': 7},
            {'role': 'QUANTITY', 'min_len': 2, 'max_len': 8},
            {'role': 'PREP', 'min_len': 2, 'max_len': 4},
            {'role': 'VEHICLE', 'min_len': 3, 'max_len': 8},
        ],
    },
    {
        'name': 'property_statement',
        'slots': [
            {'role': 'SUBJECT', 'min_len': 3, 'max_len': 10},
            {'role': 'VERB', 'min_len': 2, 'max_len': 6},
            {'role': 'QUALITY', 'min_len': 4, 'max_len': 10},
        ],
    },
    {
        'name': 'compound_naming',
        'slots': [
            {'role': 'PREFIX', 'min_len': 2, 'max_len': 4},
            {'role': 'BASE', 'min_len': 4, 'max_len': 10},
        ],
    },
]


# ---------------------------------------------------------------------------
# Dictionary-constrained DP segmentation
# ---------------------------------------------------------------------------

def dp_segment(
    stream: str, dictionary: Set[str],
    min_word_len: int = 2, max_word_len: int = 12,
    max_ed: int = 2,
) -> Tuple[List[str], float, float]:
    """DP segmentation maximizing dictionary coverage.

    For each position, try all word lengths. If the substring matches
    a dictionary word (within edit distance max_ed), score it by
    (word_length - edit_distance). Otherwise penalize.

    Returns (words, coverage_fraction, mean_ed).
    """
    n = len(stream)
    if n == 0:
        return [], 0.0, 0.0

    # Pre-build a trie-like lookup: for each length, collect dict words
    dict_by_len: Dict[int, Set[str]] = {}
    for w in dictionary:
        L = len(w)
        if min_word_len <= L <= max_word_len:
            dict_by_len.setdefault(L, set()).add(w)

    INF = float('inf')
    dp_score = [-INF] * (n + 1)
    dp_back = [-1] * (n + 1)
    dp_match = [None] * (n + 1)
    dp_score[0] = 0.0

    for i in range(n):
        if dp_score[i] == -INF:
            continue
        for L in range(min_word_len, min(max_word_len, n - i) + 1):
            j = i + L
            word = stream[i:j]

            # Find best dictionary match
            best_ed = INF
            best_match = None

            # Check exact match first
            if word in dictionary:
                best_ed = 0
                best_match = word
            else:
                # Check words of same length ±1
                for check_len in [L, L - 1, L + 1]:
                    if check_len < min_word_len:
                        continue
                    candidates = dict_by_len.get(check_len, set())
                    # Sample to limit computation
                    check_list = list(candidates)[:200]
                    for cand in check_list:
                        ed = _edit_distance(word, cand)
                        if ed < best_ed:
                            best_ed = ed
                            best_match = cand
                            if ed == 0:
                                break
                    if best_ed == 0:
                        break

            if best_ed <= max_ed:
                score = L - best_ed  # reward coverage, penalize edits
            else:
                score = -L * 0.5  # penalty for unmatched substring
                best_match = None
                best_ed = L

            total = dp_score[i] + score
            if total > dp_score[j]:
                dp_score[j] = total
                dp_back[j] = i
                dp_match[j] = (word, best_match, best_ed)

    # Backtrack
    words = []
    matches = []
    pos = n
    while pos > 0:
        prev = dp_back[pos]
        if prev < 0:
            words.append(stream[:pos])
            matches.append((stream[:pos], None, pos))
            break
        info = dp_match[pos]
        if info:
            words.append(info[0])  # actual substring
            matches.append(info)
        pos = prev

    words.reverse()
    matches.reverse()

    # Coverage = fraction of chars covered by matched words
    covered = sum(len(m[0]) for m in matches if m[1] is not None)
    coverage = covered / n if n > 0 else 0.0

    eds = [m[2] for m in matches if m[1] is not None]
    mean_ed = float(np.mean(eds)) if eds else float('inf')

    return words, coverage, mean_ed


def template_segment(
    stream: str, dictionary: Set[str], templates: List[Dict],
    max_ed: int = 2,
) -> Optional[Dict]:
    """Try all templates on a recipe stream, return best match."""
    best_result = None
    best_score = -float('inf')

    for template in templates:
        slots = template['slots']
        min_total = sum(s['min_len'] for s in slots)
        max_total = sum(s['max_len'] for s in slots)

        if len(stream) < min_total * 0.5 or len(stream) > max_total * 2:
            continue

        # Try segmentation matching this template
        result = _try_template(stream, slots, dictionary, max_ed)
        if result and result['score'] > best_score:
            best_score = result['score']
            best_result = {
                'template': template['name'],
                'segments': result['segments'],
                'score': result['score'],
                'coverage': result['coverage'],
            }

    return best_result


def _try_template(
    stream: str, slots: List[Dict], dictionary: Set[str], max_ed: int,
) -> Optional[Dict]:
    """Try to segment stream according to template slots."""
    n = len(stream)
    n_slots = len(slots)

    best_segments = None
    best_score = -float('inf')

    def _recurse(pos: int, slot_idx: int, segments: List[Dict], score: float):
        nonlocal best_segments, best_score

        if slot_idx >= n_slots:
            if pos >= n * 0.6:  # consumed enough
                if score > best_score:
                    best_score = score
                    best_segments = list(segments)
            return

        slot = slots[slot_idx]
        for L in range(slot['min_len'], min(slot['max_len'] + 1, n - pos + 1)):
            word = stream[pos:pos + L]

            # Find closest dictionary match
            if word in dictionary:
                ed = 0
                match = word
            else:
                # Quick check: only a few candidates
                match = None
                ed = max_ed + 1
                for cand in list(dictionary)[:500]:
                    if abs(len(cand) - L) > 2:
                        continue
                    d = _edit_distance(word, cand)
                    if d < ed:
                        ed = d
                        match = cand
                        if d == 0:
                            break

            if ed > max_ed:
                continue

            seg = {
                'role': slot['role'],
                'word': word,
                'match': match,
                'ed': ed,
            }
            segments.append(seg)
            _recurse(pos + L, slot_idx + 1, segments, score + (L - ed))
            segments.pop()

    _recurse(0, 0, [], 0.0)

    if best_segments is None:
        return None

    covered = sum(len(s['word']) for s in best_segments if s['match'])
    coverage = covered / n if n > 0 else 0.0

    return {
        'segments': best_segments,
        'score': best_score,
        'coverage': coverage,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_recipe_segment():
    """Phase 65.5: Recipe template-constrained segmentation."""
    t0 = time.time()
    rd = str(_results_dir())
    print("=" * 70)
    print("Phase 65, Step 5: Recipe Template-Constrained Segmentation")
    print("=" * 70)

    stream_data = _safe_load(os.path.join(rd, 'p65_decoded_stream.json'))
    if not stream_data:
        print("  ERROR: p65_decoded_stream.json not found.")
        return None

    recipe_streams = stream_data.get('recipe_streams', [])
    print(f"  Recipe streams: {len(recipe_streams)}")

    # Build pharmaceutical dictionary
    print("  Building pharmaceutical dictionary...")
    pharma_dict = _build_pharma_dict()
    print(f"  Dictionary: {len(pharma_dict)} words")

    # Process each recipe with both DP and template approaches
    print("\n  Segmenting recipes...")
    all_results: List[Dict] = []
    all_ingredients: List[str] = []
    all_eds: List[float] = []
    all_coverages: List[float] = []
    template_counts: Counter = Counter()

    for i, recipe_info in enumerate(recipe_streams):
        stream = recipe_info.get('text', '')
        if not stream or len(stream) < 4:
            continue

        # DP segmentation (unconstrained, just dictionary)
        dp_words, dp_coverage, dp_mean_ed = dp_segment(
            stream, pharma_dict, max_ed=2)

        # Template segmentation
        tmpl_result = template_segment(
            stream, pharma_dict, RECIPE_TEMPLATES, max_ed=2)

        # Use whichever is better
        if tmpl_result and tmpl_result['coverage'] > dp_coverage:
            best_method = 'template'
            best_coverage = tmpl_result['coverage']
            best_words = [s['word'] for s in tmpl_result['segments']]
            best_template = tmpl_result['template']
            best_eds = [s['ed'] for s in tmpl_result['segments']]

            # Extract ingredients
            for seg in tmpl_result['segments']:
                if 'INGREDIENT' in seg['role'] and seg['match']:
                    all_ingredients.append(seg['match'])

            template_counts[best_template] += 1
        else:
            best_method = 'dp'
            best_coverage = dp_coverage
            best_words = dp_words
            best_template = None
            best_eds = []

        all_coverages.append(best_coverage)
        if best_eds:
            all_eds.extend(best_eds)

        all_results.append({
            'recipe_idx': i,
            'stream_length': len(stream),
            'method': best_method,
            'template': best_template,
            'coverage': round(best_coverage, 3),
            'words': best_words[:20],  # limit for JSON
            'n_words': len(best_words),
            'matched': best_coverage > 0.3,
        })

        if (i + 1) % 50 == 0:
            print(f"    Processed {i + 1}/{len(recipe_streams)}...")

    # Aggregate
    n_matched = sum(1 for r in all_results if r['matched'])
    match_rate = n_matched / len(all_results) if all_results else 0.0
    mean_coverage = float(np.mean(all_coverages)) if all_coverages else 0.0
    mean_ed = float(np.mean(all_eds)) if all_eds else float('inf')

    all_word_lengths = [len(w) for r in all_results for w in r['words']]
    mean_wl = float(np.mean(all_word_lengths)) if all_word_lengths else 0.0

    # Unique ingredients
    ingredient_counter = Counter(all_ingredients)
    n_distinct_ingredients = len(ingredient_counter)

    # Fully readable (all words matched, coverage > 80%)
    n_fully_readable = sum(1 for r in all_results if r['coverage'] > 0.8)

    # Null comparison: random segmentation of same recipes
    rng = np.random.default_rng(42)
    null_coverages = []
    for recipe_info in recipe_streams[:50]:  # sample 50
        stream = recipe_info.get('text', '')
        if not stream or len(stream) < 4:
            continue
        n_words_expected = max(1, len(stream) // 5)
        rb = sorted(rng.choice(max(1, len(stream) - 1), size=min(n_words_expected, len(stream) - 1),
                               replace=False).tolist())
        rand_words = []
        prev = 0
        for b in rb:
            if b > prev:
                rand_words.append(stream[prev:b])
            prev = b
        if prev < len(stream):
            rand_words.append(stream[prev:])
        matched_chars = sum(len(w) for w in rand_words if w in pharma_dict)
        null_coverages.append(matched_chars / len(stream) if stream else 0.0)
    null_coverage = float(np.mean(null_coverages)) if null_coverages else 0.0
    selectivity = mean_coverage / null_coverage if null_coverage > 0 else float('inf')

    # Top recipes by coverage
    sorted_results = sorted(all_results, key=lambda r: r['coverage'], reverse=True)

    # Gates
    g1 = match_rate >= 0.30
    g2 = mean_ed <= 2.0
    g3 = n_distinct_ingredients >= 5
    g4 = len(template_counts) >= 2
    g5 = n_fully_readable >= 1
    gates_passed = sum([g1, g2, g3, g4, g5])

    verdict = "RECIPE_PASS" if gates_passed >= 3 else (
        "RECIPE_PARTIAL" if gates_passed >= 2 else "RECIPE_FAIL")

    print(f"\n  Results:")
    print(f"  Recipes processed: {len(all_results)}")
    print(f"  Matched (>30% coverage): {n_matched} ({match_rate:.1%})")
    print(f"  Mean coverage: {mean_coverage:.3f}")
    print(f"  Mean per-slot ED: {mean_ed:.2f}")
    print(f"  Fully readable (>80%): {n_fully_readable}")
    print(f"  Distinct ingredients: {n_distinct_ingredients}")
    print(f"  Template types: {dict(template_counts)}")
    print(f"  Null coverage: {null_coverage:.3f}, selectivity: {selectivity:.2f}x")
    print(f"\n  Gates: R1(match)={'PASS' if g1 else 'FAIL'} "
          f"R2(ED)={'PASS' if g2 else 'FAIL'} "
          f"R3(ingredients)={'PASS' if g3 else 'FAIL'} "
          f"R4(templates)={'PASS' if g4 else 'FAIL'} "
          f"R5(readable)={'PASS' if g5 else 'FAIL'}")
    print(f"  Verdict: {verdict} ({gates_passed}/5)")

    # Top 5 recipes
    print("\n  Top recipes by coverage:")
    for r in sorted_results[:5]:
        print(f"    Recipe {r['recipe_idx']}: coverage={r['coverage']:.2f}, "
              f"method={r['method']}, words={' '.join(r['words'][:8])}")

    result = RecipeSegmentResult(
        pharma_dict_size=len(pharma_dict),
        n_recipes=len(all_results),
        n_matched=n_matched,
        match_rate=round(match_rate, 4),
        mean_coverage=round(mean_coverage, 4),
        mean_word_length=round(mean_wl, 2),
        mean_per_slot_ed=round(mean_ed, 2) if mean_ed != float('inf') else 99.0,
        n_fully_readable=n_fully_readable,
        n_distinct_ingredients=n_distinct_ingredients,
        n_template_types=len(template_counts),
        template_distribution=dict(template_counts),
        ingredient_distribution=[{'ingredient': w, 'count': c}
                                 for w, c in ingredient_counter.most_common(20)],
        top_recipes=[r for r in sorted_results[:20]],
        sample_segmentations=[{
            'recipe_idx': r['recipe_idx'],
            'words': r['words'][:15],
            'coverage': r['coverage'],
            'template': r['template'],
        } for r in sorted_results[:10]],
        null_coverage=round(null_coverage, 4),
        selectivity=round(selectivity, 3),
        g1_match_rate=g1,
        g2_per_slot_ed=g2,
        g3_ingredients=g3,
        g4_template_types=g4,
        g5_fully_readable=g5,
        gates_passed=gates_passed,
        gate_passed=gates_passed >= 3,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    _save_json(rd, 'p65_recipe_segment.json', asdict(result))
    print(f"\n  Runtime: {result.runtime_seconds:.1f}s")
    return result
