"""
Phase 54.7 – Simulated Macaronic Text Comparison
===================================================
Generates synthetic macaronic pharmaceutical text in each of 5 Northern
Italian dialects, then compares distributional properties against the
ACTUAL decoded Voynich text.  Six metrics are computed for each simulated
corpus and the real decoded Voynich: function-word frequency, mean word
length, character bigram entropy, content/function ratio, type-token ratio,
and word-length distribution.  Dialects are ranked by composite distance
(weighted JSD + scalar differences).

A 1000-iteration null test establishes whether the best-matching dialect
is significantly closer than random gibberish.

Dependency chain:
    combined_refine.json       (Phase 15 — best assignment)
    modifier_integrate.json    (Phase 16 — modifier chars + rules)
        → phase54_dialect_sim.json   (this step)
"""

import json
import math
import os
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    load_corpus,
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


# ---------------------------------------------------------------------------
# Recipe templates
# ---------------------------------------------------------------------------

LATIN_RECIPES = [
    "recipe de radice et cola cum aqua bene et da de sero",
    "accipe sene et tere bene cola et da dose de sera",
    "recipe de herba et cola cum sero bene et bene",
    "tere radice et misce cum aqua cola bene de sero",
    "accipe de coralli et tere bene cola cum sero",
    "recipe sene et cola de aqua da bene de sera",
    "tere de radice cola cum aqua et da de sero",
    "accipe herba et cola bene tere cum sero de sera",
    "recipe de sene cola bene et da cum aqua de sero",
    "tere bene de radice et cola cum sero da dose",
    "accipe de coralli tere et cola cum aqua bene",
    "recipe herba et tere bene cola de sero cum aqua",
    "da de sene cola bene et tere cum sero de sera",
    "accipe radice et cola de aqua tere bene de sero",
    "recipe de herba tere bene cola cum sero da dose",
    "cola de sene et da bene cum aqua de sero tere",
    "accipe de radice cola bene et tere cum sero da",
    "recipe coralli et tere cola de aqua bene de sera",
    "tere de sene cola bene cum sero et da de aqua",
    "accipe herba cola de sero et tere bene da dose",
]


# ---------------------------------------------------------------------------
# Dialect substitution and phonological rule tables
# ---------------------------------------------------------------------------

DIALECT_FUNCTION_SUBS = {
    'venetian': {
        'et': 'e', 'cum': 'co', 'de': 'de', 'da': 'da',
        'in': 'in', 'bene': 'ben', 'aqua': 'aqua',
    },
    'lombard': {
        'et': 'e', 'cum': 'co', 'de': 'de', 'da': 'da',
        'in': 'in', 'bene': 'ben', 'aqua': 'aqua',
    },
    'ligurian': {
        'et': 'e', 'cum': 'con', 'de': 'de', 'da': 'da',
        'in': 'in', 'bene': 'ben', 'aqua': 'aegua',
    },
    'emilian': {
        'et': 'e', 'cum': 'con', 'de': 'de', 'da': 'da',
        'in': 'in', 'bene': 'ben', 'aqua': 'aqua',
    },
    'tuscan': {
        'et': 'e', 'cum': 'con', 'de': 'di', 'da': 'da',
        'in': 'in', 'bene': 'bene', 'aqua': 'acqua',
    },
}

DIALECT_PHONOLOGICAL_RULES = {
    'venetian': [
        ('ll', 'l'),   # degemination
        ('nn', 'n'),
        ('ss', 's'),
        ('tt', 't'),
    ],
    'lombard': [
        ('ll', 'l'),
        ('nn', 'n'),
        ('ss', 's'),
        ('tt', 't'),
    ],
    'ligurian': [
        ('ll', 'l'),
        ('nn', 'n'),
        ('ss', 's'),
    ],
    'emilian': [
        ('ll', 'l'),
        ('nn', 'n'),
        ('ss', 's'),
        ('tt', 't'),
    ],
    'tuscan': [
        # No degemination - Tuscan preserves geminates
    ],
}


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------

def _translate_recipe(recipe: str, dialect: str) -> str:
    """Translate a Latin recipe template into a dialect variant."""
    words = recipe.split()
    subs = DIALECT_FUNCTION_SUBS[dialect]
    rules = DIALECT_PHONOLOGICAL_RULES[dialect]
    translated = []
    for word in words:
        # Apply function word substitution
        w = subs.get(word, word)
        # Apply phonological rules
        for old, new in rules:
            w = w.replace(old, new)
        translated.append(w)
    return ' '.join(translated)


# ---------------------------------------------------------------------------
# Distributional metrics
# ---------------------------------------------------------------------------

FUNCTION_WORDS = {
    'di', 'de', 'co', 'con', 'se', 'si', 'ci', 'ne', 'la', 'li',
    'e', 'et', 'in', 'da', 'a', 'ha', 'fa', 'te', 'ti', 'tu',
    'ben', 'bene', 'cum', 'su',
}


def _compute_metrics(tokens: List[str]) -> Dict[str, Any]:
    """Compute 6 distributional metrics for a token list."""
    if not tokens:
        return {
            'func_freq': {},
            'mean_length': 0,
            'bigram_h2': 0,
            'content_ratio': 0,
            'ttr_1k': 0,
            'length_dist': [],
        }

    # 1. Function word frequency distribution
    func_counts = Counter(t for t in tokens if t in FUNCTION_WORDS)
    total_func = sum(func_counts.values())
    func_freq = (
        {w: c / total_func for w, c in func_counts.items()}
        if total_func > 0
        else {}
    )

    # 2. Mean word length
    mean_length = sum(len(t) for t in tokens) / len(tokens)

    # 3. Character bigram entropy
    bigrams: Counter = Counter()
    for t in tokens:
        for i in range(len(t) - 1):
            bigrams[t[i:i + 2]] += 1
    total_bi = sum(bigrams.values())
    if total_bi > 0:
        probs = [c / total_bi for c in bigrams.values()]
        bigram_h2 = -sum(p * math.log2(p) for p in probs if p > 0)
    else:
        bigram_h2 = 0

    # 4. Content / function word ratio
    n_func = sum(1 for t in tokens if t in FUNCTION_WORDS)
    content_ratio = (len(tokens) - n_func) / len(tokens) if tokens else 0

    # 5. Type-token ratio at 1000 tokens
    subset = tokens[:1000]
    ttr_1k = len(set(subset)) / len(subset) if subset else 0

    # 6. Word length distribution (as histogram, lengths 1-10)
    length_hist = [0] * 10
    for t in tokens:
        idx = min(len(t), 10) - 1
        length_hist[idx] += 1
    total = sum(length_hist)
    length_dist = [c / total for c in length_hist] if total > 0 else length_hist

    return {
        'func_freq': func_freq,
        'mean_length': mean_length,
        'bigram_h2': bigram_h2,
        'content_ratio': content_ratio,
        'ttr_1k': ttr_1k,
        'length_dist': length_dist,
    }


# ---------------------------------------------------------------------------
# Jensen-Shannon divergence
# ---------------------------------------------------------------------------

def _jsd(p: Dict[str, float], q: Dict[str, float]) -> float:
    """Jensen-Shannon divergence between two distributions (as dicts)."""
    all_keys = set(p.keys()) | set(q.keys())
    if not all_keys:
        return 0.0
    pp = [p.get(k, 0.0) for k in all_keys]
    qq = [q.get(k, 0.0) for k in all_keys]
    # Normalize
    sp, sq = sum(pp), sum(qq)
    if sp > 0:
        pp = [x / sp for x in pp]
    if sq > 0:
        qq = [x / sq for x in qq]
    m = [(a + b) / 2 for a, b in zip(pp, qq)]

    def kl(a: List[float], b: List[float]) -> float:
        return sum(
            ai * math.log2(ai / bi)
            for ai, bi in zip(a, b)
            if ai > 0 and bi > 0
        )

    return (kl(pp, m) + kl(qq, m)) / 2


# ---------------------------------------------------------------------------
# Composite distance
# ---------------------------------------------------------------------------

def _composite_distance(
    real_metrics: Dict[str, Any],
    sim_metrics: Dict[str, Any],
) -> float:
    """Weighted distance between real and simulated metrics."""
    d = 0.0
    # Function word JSD (weight 0.30)
    d += 0.30 * _jsd(real_metrics['func_freq'], sim_metrics['func_freq'])
    # Mean length difference (weight 0.15)
    d += 0.15 * abs(
        real_metrics['mean_length'] - sim_metrics['mean_length']
    ) / max(real_metrics['mean_length'], 1)
    # Bigram entropy difference (weight 0.20)
    d += 0.20 * abs(
        real_metrics['bigram_h2'] - sim_metrics['bigram_h2']
    ) / max(real_metrics['bigram_h2'], 1)
    # Content ratio difference (weight 0.15)
    d += 0.15 * abs(
        real_metrics['content_ratio'] - sim_metrics['content_ratio']
    )
    # TTR difference (weight 0.10)
    d += 0.10 * abs(real_metrics['ttr_1k'] - sim_metrics['ttr_1k'])
    # Length distribution JSD (weight 0.10)
    lp = {str(i): v for i, v in enumerate(real_metrics['length_dist'])}
    lq = {str(i): v for i, v in enumerate(sim_metrics['length_dist'])}
    d += 0.10 * _jsd(lp, lq)
    return d


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class DialectSimResult:
    phase: str                                   # "54.7"
    experiment: str                               # "simulated_macaronic_comparison"
    n_real_tokens: int
    n_simulated_tokens_per_dialect: int
    real_metrics: Dict[str, Any]
    per_dialect_metrics: Dict[str, Dict[str, Any]]
    per_dialect_distances: Dict[str, float]
    ranking: List[Dict]                           # [{dialect, distance, score}]
    dialect_scores: Dict[str, float]
    null_mean_distance: float
    null_std_distance: float
    z_score: float
    selectivity: float
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_dialect_sim() -> None:
    t0 = time.time()

    print("=" * 70)
    print("PHASE 54.7: Simulated Macaronic Text Comparison")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Load corpus and decode real Voynich
    # ------------------------------------------------------------------
    corpus = load_corpus(verbose=False)
    rd = _results_dir()

    # Load assignment + modifiers
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

    # Decode real Voynich
    print("\n  Decoding real Voynich corpus ...")
    real_decoded: List[str] = []
    for folio_id, page in corpus.pages.items():
        for token in page.paragraph_text.split():
            if token.strip():
                d = decode_token_modifier_aware(
                    token.strip(),
                    assignment,
                    eva_to_triple,
                    modifier_chars,
                    modifier_rules=modifier_rules,
                )
                real_decoded.append(d.lower())

    print(f"    {len(real_decoded)} tokens decoded")

    # ------------------------------------------------------------------
    # Generate simulated dialect corpora
    # ------------------------------------------------------------------
    print("\n  Generating simulated dialect corpora ...")
    dialects = list(DIALECT_FUNCTION_SUBS.keys())
    dialect_corpora: Dict[str, List[str]] = {}
    for dialect in dialects:
        tokens: List[str] = []
        for recipe in LATIN_RECIPES:
            translated = _translate_recipe(recipe, dialect)
            tokens.extend(translated.split())
        dialect_corpora[dialect] = tokens
        print(f"    {dialect}: {len(tokens)} tokens")

    # ------------------------------------------------------------------
    # Compute metrics
    # ------------------------------------------------------------------
    print("\n  Computing distributional metrics ...")
    real_metrics = _compute_metrics(real_decoded)
    print(f"    Real Voynich: mean_len={real_metrics['mean_length']:.2f}, "
          f"bigram_H2={real_metrics['bigram_h2']:.2f}, "
          f"content_ratio={real_metrics['content_ratio']:.3f}, "
          f"TTR@1k={real_metrics['ttr_1k']:.3f}")

    per_dialect_metrics: Dict[str, Dict[str, Any]] = {}
    for dialect in dialects:
        m = _compute_metrics(dialect_corpora[dialect])
        per_dialect_metrics[dialect] = m
        print(f"    {dialect}: mean_len={m['mean_length']:.2f}, "
              f"bigram_H2={m['bigram_h2']:.2f}, "
              f"content_ratio={m['content_ratio']:.3f}, "
              f"TTR@1k={m['ttr_1k']:.3f}")

    # ------------------------------------------------------------------
    # Compute distances and rank
    # ------------------------------------------------------------------
    print("\n  Computing composite distances ...")
    dialect_distances: Dict[str, float] = {}
    for dialect in dialects:
        dist = _composite_distance(real_metrics, per_dialect_metrics[dialect])
        dialect_distances[dialect] = dist
        print(f"    {dialect}: distance = {dist:.6f}")

    # Convert distances to scores (smaller distance = higher score)
    max_dist = max(dialect_distances.values())
    min_dist = min(dialect_distances.values())
    range_dist = max_dist - min_dist if max_dist > min_dist else 1.0
    dialect_scores: Dict[str, float] = {}
    for dialect, dist in dialect_distances.items():
        dialect_scores[dialect] = 1.0 - (dist - min_dist) / range_dist

    # Build ranking (sorted by ascending distance)
    ranking_items = sorted(dialect_distances.items(), key=lambda x: x[1])
    ranking: List[Dict] = []
    for dialect, dist in ranking_items:
        ranking.append({
            'dialect': dialect,
            'distance': round(dist, 6),
            'score': round(dialect_scores[dialect], 4),
        })

    print("\n  Ranking (closest first):")
    for i, entry in enumerate(ranking):
        print(f"    {i+1}. {entry['dialect']}: "
              f"dist={entry['distance']:.6f}, score={entry['score']:.4f}")

    best_dialect = ranking[0]['dialect']
    best_distance = ranking[0]['distance']

    # ------------------------------------------------------------------
    # Null test: 1000 random-word corpora
    # ------------------------------------------------------------------
    print("\n  Running null test (1000 iterations) ...")
    rng = random.Random(42)
    # Determine token count from one dialect corpus
    n_sim_tokens = len(dialect_corpora[dialects[0]])
    null_distances: List[float] = []

    for trial in range(1000):
        # Generate random-word text with same length as recipes
        null_tokens: List[str] = []
        for _ in range(n_sim_tokens):
            wlen = rng.randint(2, 5)
            word = ''.join(rng.choice('abcdefghilmnopqrstuvz') for _ in range(wlen))
            null_tokens.append(word)
        null_metrics = _compute_metrics(null_tokens)
        null_dist = _composite_distance(real_metrics, null_metrics)
        null_distances.append(null_dist)

    null_mean = sum(null_distances) / len(null_distances)
    null_std = (
        sum((d - null_mean) ** 2 for d in null_distances) / len(null_distances)
    ) ** 0.5

    z_score = (null_mean - best_distance) / null_std if null_std > 0 else 0.0
    selectivity = null_mean / best_distance if best_distance > 0 else 0.0

    print(f"    Null mean distance: {null_mean:.6f}")
    print(f"    Null std distance:  {null_std:.6f}")
    print(f"    Best dialect distance: {best_distance:.6f}")
    print(f"    z-score: {z_score:.2f}")
    print(f"    Selectivity: {selectivity:.2f}x")

    # ------------------------------------------------------------------
    # Verdict
    # ------------------------------------------------------------------
    if z_score >= 3.0:
        verdict = f"SIGNIFICANT_{best_dialect.upper()}"
    elif z_score >= 2.0:
        verdict = f"SUGGESTIVE_{best_dialect.upper()}"
    else:
        verdict = "NO_SIGNAL"

    print(f"\n  Verdict: {verdict}")

    # ------------------------------------------------------------------
    # Build and save result
    # ------------------------------------------------------------------
    runtime = round(time.time() - t0, 2)

    result = DialectSimResult(
        phase="54.7",
        experiment="simulated_macaronic_comparison",
        n_real_tokens=len(real_decoded),
        n_simulated_tokens_per_dialect=n_sim_tokens,
        real_metrics=real_metrics,
        per_dialect_metrics=per_dialect_metrics,
        per_dialect_distances={k: round(v, 6) for k, v in dialect_distances.items()},
        ranking=ranking,
        dialect_scores={k: round(v, 4) for k, v in dialect_scores.items()},
        null_mean_distance=round(null_mean, 6),
        null_std_distance=round(null_std, 6),
        z_score=round(z_score, 4),
        selectivity=round(selectivity, 4),
        verdict=verdict,
        runtime_seconds=runtime,
    )

    out_path = os.path.join(rd, 'phase54_dialect_sim.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2)

    print(f"\n  Saved -> {out_path}")
    print(f"  Completed in {runtime:.1f}s")
