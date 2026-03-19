"""
Phase 54.3: Article and Pronoun System Matching
================================================
Test signal function words against complete morphological paradigms for
five northern-Italian dialects (Venetian, Lombard, Ligurian, Emilian,
Tuscan).  Compute per-dialect raw, weighted, and composite scores; run
a CV-inventory null test to quantify selectivity.

Output:
  results/phase54_articles.json
"""

import json
import os
import random
import time
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Any, Set, Tuple

from voynich.core._paths import results_dir as _results_dir


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
# Signal function words to test
# ---------------------------------------------------------------------------

SIGNAL_FUNCTION_WORDS: Dict[str, List[str]] = {
    'la': ['article_f_sg'],
    'li': ['article_pl'],
    'di': ['preposition_of'],
    'de': ['preposition_of'],
    'co': ['preposition_with'],
    'se': ['conjunction_if', 'pronoun_refl'],
    'si': ['pronoun_refl'],
    'ci': ['pronoun_ci'],
    'ne': ['conjunction_not', 'pronoun_ne'],
    'te': ['pronoun_2sg'],
    'ti': ['pronoun_2sg'],
    'tu': ['pronoun_2sg'],
    'ha': ['auxiliary_has'],
    'fa': ['auxiliary_does'],
}


# ---------------------------------------------------------------------------
# Dialect paradigms (Rohlfs 1949-54)
# ---------------------------------------------------------------------------

DIALECT_PARADIGMS: Dict[str, Dict[str, Dict[str, List[str]]]] = {
    'venetian': {
        'articles': {'m_sg': ['el', 'lo'], 'f_sg': ['la'], 'pl': ['i', 'li', 'le']},
        'prepositions': {'of': ['de'], 'from': ['da'], 'with': ['co', 'con'], 'in': ['in'], 'on': ['su', 'sora']},
        'pronouns': {'1sg': ['mi'], '2sg': ['ti', 'te'], '3sg_m': ['el', 'lu'], '3sg_f': ['la', 'ela'],
                     'refl': ['se'], 'ci': ['ghe', 'ge', 'ce'], 'ne': ['ne']},
        'auxiliaries': {'has': ['ha', 'a'], 'is': ['xe', 'e'], 'does': ['fa']},
        'conjunctions': {'and': ['e'], 'or': ['o'], 'if': ['se'], 'not': ['no', 'ne']},
    },
    'lombard': {
        'articles': {'m_sg': ['el', 'ol'], 'f_sg': ['la'], 'pl': ['i', 'li']},
        'prepositions': {'of': ['de'], 'from': ['da'], 'with': ['co', 'con'], 'in': ['in'], 'on': ['su']},
        'pronouns': {'1sg': ['mi'], '2sg': ['ti', 'te'], '3sg_m': ['lu', 'el'],
                     'refl': ['se'], 'ci': ['ghe', 'ce'], 'ne': ['ne']},
        'auxiliaries': {'has': ['ha', 'a'], 'is': ['e'], 'does': ['fa']},
        'conjunctions': {'and': ['e'], 'or': ['o'], 'if': ['se'], 'not': ['no', 'ne']},
    },
    'ligurian': {
        'articles': {'m_sg': ['o', 'u'], 'f_sg': ['a'], 'pl': ['i', 'e']},
        'prepositions': {'of': ['de', 'di'], 'from': ['da'], 'with': ['con', 'co'], 'in': ['in'], 'on': ['in su']},
        'pronouns': {'1sg': ['mi'], '2sg': ['ti', 'te'], '3sg_m': ['lu'],
                     'refl': ['se'], 'ci': ['ghe', 'ce'], 'ne': ['ne']},
        'auxiliaries': {'has': ['ha', 'a'], 'is': ['e', 'xe'], 'does': ['fa', 'fai']},
        'conjunctions': {'and': ['e'], 'or': ['o'], 'if': ['se'], 'not': ['no', 'ne']},
    },
    'emilian': {
        'articles': {'m_sg': ['al', 'el'], 'f_sg': ['la'], 'pl': ['i', 'al']},
        'prepositions': {'of': ['ed', 'ad', 'de'], 'from': ['da'], 'with': ['con'], 'in': ['in'], 'on': ['su']},
        'pronouns': {'1sg': ['me'], '2sg': ['te'], '3sg_m': ['al', 'lu'],
                     'refl': ['se', 'as'], 'ci': ['ghe', 'ce'], 'ne': ['ne']},
        'auxiliaries': {'has': ['ha', 'a'], 'is': ['e'], 'does': ['fa']},
        'conjunctions': {'and': ['e', 'ed'], 'or': ['o'], 'if': ['se'], 'not': ['no', 'ne']},
    },
    'tuscan': {
        'articles': {'m_sg': ['il', 'lo'], 'f_sg': ['la'], 'pl': ['i', 'gli', 'le']},
        'prepositions': {'of': ['di'], 'from': ['da'], 'with': ['con'], 'in': ['in'], 'on': ['su', 'sopra']},
        'pronouns': {'1sg': ['io'], '2sg': ['tu', 'te'], '3sg_m': ['egli', 'lui'],
                     'refl': ['si'], 'ci': ['ci'], 'ne': ['ne']},
        'auxiliaries': {'has': ['ha'], 'is': ['\u00e8'], 'does': ['fa']},
        'conjunctions': {'and': ['e', 'et'], 'or': ['o'], 'if': ['se'], 'not': ['non', 'no']},
    },
}


# ---------------------------------------------------------------------------
# CV inventory for null test
# ---------------------------------------------------------------------------

CV_INVENTORY = [
    'ba', 'be', 'bi', 'bo', 'bu',
    'ca', 'ce', 'ci', 'co', 'cu',
    'da', 'de', 'di', 'do', 'du',
    'fa', 'fe',
    'ga', 'ge', 'gi',
    'ha', 'hi',
    'la', 'li',
    'mi',
    'ne', 'ni', 'no', 'nu',
    'ra', 're', 'ri', 'ro', 'ru',
    'sa', 'se', 'si', 'so', 'su',
    'ta', 'te', 'ti', 'to', 'tu',
]


# ---------------------------------------------------------------------------
# Category mapping
# ---------------------------------------------------------------------------

# Maps SIGNAL_FUNCTION_WORDS category tags -> (paradigm_group, paradigm_key)
_CATEGORY_MAP: Dict[str, Tuple[str, str]] = {
    'article_f_sg':    ('articles', 'f_sg'),
    'article_pl':      ('articles', 'pl'),
    'preposition_of':  ('prepositions', 'of'),
    'preposition_with': ('prepositions', 'with'),
    'conjunction_if':  ('conjunctions', 'if'),
    'pronoun_refl':    ('pronouns', 'refl'),
    'pronoun_2sg':     ('pronouns', '2sg'),
    'pronoun_ci':      ('pronouns', 'ci'),
    'pronoun_ne':      ('pronouns', 'ne'),
    'conjunction_not': ('conjunctions', 'not'),
    'auxiliary_has':   ('auxiliaries', 'has'),
    'auxiliary_does':  ('auxiliaries', 'does'),
}

# Map paradigm groups to the 5 functional super-categories
_GROUP_TO_SUPERCATEGORY: Dict[str, str] = {
    'articles': 'articles',
    'prepositions': 'prepositions',
    'pronouns': 'pronouns',
    'auxiliaries': 'auxiliaries',
    'conjunctions': 'conjunctions',
}


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class ArticleResult:
    phase: str                          # "54.3"
    experiment: str                     # "article_pronoun_system"
    n_function_words_tested: int
    signal_function_words: List[Dict]   # word, categories, weight
    per_dialect_scores: Dict[str, Dict] # dialect -> {raw, weighted, coverage, composite}
    ranking: List[Dict]                 # [{dialect, composite}, ...]
    top_discriminants: List[Dict]       # [{word, eliminates, supports}, ...]
    null_mean: float
    null_std: float
    z_score: float
    selectivity: float
    gates: Dict[str, bool]
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def _word_matches_dialect(word: str, categories: List[str],
                          paradigm: Dict[str, Dict[str, List[str]]]) -> bool:
    """Return True if *word* appears in any of the paradigm slots
    indicated by *categories*."""
    for cat in categories:
        mapping = _CATEGORY_MAP.get(cat)
        if mapping is None:
            continue
        group, key = mapping
        forms = paradigm.get(group, {}).get(key, [])
        if word in forms:
            return True
    return False


def _score_word_set(word_set: Dict[str, List[str]],
                    dialects: Dict[str, Dict]) -> Dict[str, Dict]:
    """Score each dialect against a word set.

    Returns dict  dialect -> {raw, weighted, coverage, composite}.
    """
    n_total = len(word_set)
    if n_total == 0:
        return {d: {'raw': 0.0, 'weighted': 0.0, 'coverage': 0,
                     'composite': 0.0} for d in dialects}

    # Step 2 + 3: per-word matches and diagnostic weights
    word_dialect_matches: Dict[str, Set[str]] = {}
    for word, cats in word_set.items():
        matches: Set[str] = set()
        for dname, paradigm in dialects.items():
            if _word_matches_dialect(word, cats, paradigm):
                matches.add(dname)
        word_dialect_matches[word] = matches

    weights: Dict[str, float] = {}
    for word, matches in word_dialect_matches.items():
        n_match = len(matches)
        weights[word] = 1.0 / n_match if n_match > 0 else 0.0

    total_weight = sum(weights.values())

    scores: Dict[str, Dict] = {}
    for dname in dialects:
        matched_words = [w for w, m in word_dialect_matches.items() if dname in m]
        raw = len(matched_words) / n_total if n_total else 0.0
        weighted = (sum(weights[w] for w in matched_words) / total_weight
                    if total_weight > 0 else 0.0)

        # Category coverage: which super-categories have >= 1 match?
        covered_cats: Set[str] = set()
        for w in matched_words:
            for cat in word_set[w]:
                mapping = _CATEGORY_MAP.get(cat)
                if mapping:
                    covered_cats.add(_GROUP_TO_SUPERCATEGORY[mapping[0]])
        coverage = len(covered_cats)

        composite = 0.5 * weighted + 0.3 * (coverage / 5.0) + 0.2 * raw
        scores[dname] = {
            'raw': round(raw, 4),
            'weighted': round(weighted, 4),
            'coverage': coverage,
            'composite': round(composite, 4),
        }
    return scores


def _best_composite(scores: Dict[str, Dict]) -> float:
    return max(s['composite'] for s in scores.values()) if scores else 0.0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_articles() -> None:
    t0 = time.time()

    print("=" * 70)
    print("PHASE 54.3: Article and Pronoun System Matching")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 1: Load signal words, keep only those in SIGNAL_FUNCTION_WORDS
    # ------------------------------------------------------------------
    rdir = _results_dir()

    signal_words_found: Set[str] = set()

    path_10k = os.path.join(rdir, "signal_10k.json")
    if os.path.exists(path_10k):
        with open(path_10k) as f:
            data_10k = json.load(f)
        for entry in data_10k.get("word_signals", []):
            w = entry.get("word", "")
            if w in SIGNAL_FUNCTION_WORDS:
                signal_words_found.add(w)
        print(f"  signal_10k.json: {len(data_10k.get('word_signals', []))} words, "
              f"{sum(1 for e in data_10k.get('word_signals', []) if e.get('word','') in SIGNAL_FUNCTION_WORDS)} function words")
    else:
        print(f"  WARNING: {path_10k} not found")

    path_it = os.path.join(rdir, "italian_signal.json")
    if os.path.exists(path_it):
        with open(path_it) as f:
            data_it = json.load(f)
        for entry in data_it.get("italian_signal_words", []):
            w = entry.get("word", "")
            if w in SIGNAL_FUNCTION_WORDS:
                signal_words_found.add(w)
        print(f"  italian_signal.json: {len(data_it.get('italian_signal_words', []))} words, "
              f"{sum(1 for e in data_it.get('italian_signal_words', []) if e.get('word','') in SIGNAL_FUNCTION_WORDS)} function words")
    else:
        print(f"  WARNING: {path_it} not found")

    # Build the tested word set (only those actually found as signal)
    tested_words: Dict[str, List[str]] = {
        w: cats for w, cats in SIGNAL_FUNCTION_WORDS.items()
        if w in signal_words_found
    }
    n_tested = len(tested_words)
    print(f"\n  Function words testable: {n_tested} / {len(SIGNAL_FUNCTION_WORDS)}")

    # ------------------------------------------------------------------
    # Step 2-3: Per-word dialect matches and diagnostic weights
    # ------------------------------------------------------------------
    word_info: List[Dict] = []
    for word, cats in sorted(tested_words.items()):
        matching_dialects = []
        for dname, paradigm in DIALECT_PARADIGMS.items():
            if _word_matches_dialect(word, cats, paradigm):
                matching_dialects.append(dname)
        n_d = len(matching_dialects)
        weight = 1.0 / n_d if n_d > 0 else 0.0
        word_info.append({
            'word': word,
            'categories': cats,
            'matching_dialects': sorted(matching_dialects),
            'n_dialects': n_d,
            'weight': round(weight, 4),
        })

    # ------------------------------------------------------------------
    # Step 4-6: Score each dialect
    # ------------------------------------------------------------------
    per_dialect = _score_word_set(tested_words, DIALECT_PARADIGMS)

    # ------------------------------------------------------------------
    # Step 7: Identify top discriminants
    # ------------------------------------------------------------------
    all_dialects = set(DIALECT_PARADIGMS.keys())
    top_discriminants: List[Dict] = []
    for wi in word_info:
        supports = set(wi['matching_dialects'])
        eliminates = sorted(all_dialects - supports)
        if eliminates:  # only include words that actually discriminate
            top_discriminants.append({
                'word': wi['word'],
                'eliminates': eliminates,
                'supports': sorted(supports),
                'weight': wi['weight'],
            })
    # Sort by weight descending (most discriminating first)
    top_discriminants.sort(key=lambda x: -x['weight'])

    # ------------------------------------------------------------------
    # Step 10: Ranking
    # ------------------------------------------------------------------
    ranking = sorted(
        [{'dialect': d, 'composite': s['composite']} for d, s in per_dialect.items()],
        key=lambda x: -x['composite'],
    )

    # ------------------------------------------------------------------
    # Step 8: Null test
    # ------------------------------------------------------------------
    print("\n  Running null test (1000 iterations)...")
    rng = random.Random(42)
    null_best_scores: List[float] = []
    n_null = 1000

    for _ in range(n_null):
        # Replace each signal function word with a random CV syllable
        null_words: Dict[str, List[str]] = {}
        used: Set[str] = set()
        for word, cats in tested_words.items():
            replacement = rng.choice(CV_INVENTORY)
            # Allow duplicates — each trial is independent
            null_words[replacement] = cats
        null_scores = _score_word_set(null_words, DIALECT_PARADIGMS)
        null_best_scores.append(_best_composite(null_scores))

    null_mean = sum(null_best_scores) / len(null_best_scores)
    null_var = sum((x - null_mean) ** 2 for x in null_best_scores) / len(null_best_scores)
    null_std = null_var ** 0.5

    real_best = ranking[0]['composite'] if ranking else 0.0
    z_score = (real_best - null_mean) / null_std if null_std > 0 else 0.0
    selectivity = real_best / null_mean if null_mean > 0 else 0.0

    # ------------------------------------------------------------------
    # Step 9: Gates
    # ------------------------------------------------------------------
    second_best = ranking[1]['composite'] if len(ranking) > 1 else 0.0
    top_coverage = per_dialect.get(ranking[0]['dialect'], {}).get('coverage', 0) if ranking else 0

    gates = {
        'G1_enough_words': n_tested >= 8,
        'G2_separation': (real_best - second_best) >= 0.10,
        'G3_coverage': top_coverage >= 4,
        'G4_null_selectivity': selectivity >= 1.5,
    }
    n_pass = sum(gates.values())
    if n_pass == 4:
        verdict = "STRONG_MATCH"
    elif n_pass >= 3:
        verdict = "MODERATE_MATCH"
    elif n_pass >= 2:
        verdict = "WEAK_MATCH"
    else:
        verdict = "NO_MATCH"

    # ------------------------------------------------------------------
    # Console output
    # ------------------------------------------------------------------
    print(f"\n  {'Dialect':<12} {'Raw':>6} {'Weighted':>9} {'Cov':>4} {'Composite':>10}")
    print("  " + "-" * 45)
    for r in ranking:
        d = r['dialect']
        s = per_dialect[d]
        marker = " <-- BEST" if d == ranking[0]['dialect'] else ""
        print(f"  {d:<12} {s['raw']:>6.3f} {s['weighted']:>9.4f} {s['coverage']:>4d} {s['composite']:>10.4f}{marker}")

    print(f"\n  Top discriminants:")
    for td in top_discriminants[:8]:
        print(f"    {td['word']:<4} eliminates {td['eliminates']}, "
              f"supports {td['supports']}  (w={td['weight']:.2f})")

    print(f"\n  Null test: mean={null_mean:.4f}, std={null_std:.4f}")
    print(f"  Real best: {real_best:.4f}  z={z_score:.2f}  selectivity={selectivity:.2f}x")

    print(f"\n  Gates:")
    for gname, gval in gates.items():
        status = "PASS" if gval else "FAIL"
        print(f"    {gname}: {status}")

    print(f"\n  Verdict: {verdict}")
    print(f"  Top dialect: {ranking[0]['dialect']} ({ranking[0]['composite']:.4f})")

    # ------------------------------------------------------------------
    # Build result and save
    # ------------------------------------------------------------------
    elapsed = round(time.time() - t0, 2)

    result = ArticleResult(
        phase="54.3",
        experiment="article_pronoun_system",
        n_function_words_tested=n_tested,
        signal_function_words=word_info,
        per_dialect_scores=per_dialect,
        ranking=ranking,
        top_discriminants=top_discriminants,
        null_mean=round(null_mean, 6),
        null_std=round(null_std, 6),
        z_score=round(z_score, 4),
        selectivity=round(selectivity, 4),
        gates=gates,
        verdict=verdict,
        runtime_seconds=elapsed,
    )

    out_path = os.path.join(rdir, "phase54_articles.json")
    with open(out_path, "w") as f:
        json.dump(_convert(result), f, indent=2)
    print(f"\n  Saved: {out_path}")
    print(f"  Runtime: {elapsed:.1f}s")
