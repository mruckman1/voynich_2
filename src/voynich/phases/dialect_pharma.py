"""
Phase 54.4: Pharmaceutical Terminology Regionalization
======================================================
Test decoded vocabulary against characteristic pharmaceutical term lists
from three major Italian medical-school traditions (Salerno, Bologna,
Padua/Venice).  Score each tradition by exact and edit-distance-1 matches,
derive dialect affinity weights, and run a permutation null test to check
whether the observed clustering exceeds chance.

Output:
  results/phase54_pharma_region.json
"""

import json
import os
import random
import time
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Any, Set

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
# Pharmaceutical tradition reference data
# ---------------------------------------------------------------------------

PHARMA_TRADITIONS: Dict[str, Dict[str, Any]] = {
    'salerno': {
        'source': 'Circa Instans (Platearius, 12th c.)',
        'characteristic_terms': [
            'recipe', 'accipe', 'misce', 'cola', 'destilla', 'tere',
            'simplicium', 'compositum', 'electuarium', 'syrupus',
            'dosis', 'pondus', 'manipulus', 'fasciculus',
            'sene', 'senna', 'raso', 'bene', 'commune',
        ],
        'naming_pattern': 'latin_conservative',
        'dia_compounds': 'moderate',
    },
    'bologna': {
        'source': 'Taddeo Alderotti Consilia (13th c.)',
        'characteristic_terms': [
            'ordinatio', 'consilium', 'practica', 'regimen',
            'complexio', 'qualitas', 'gradus', 'virtus',
            'decoctio', 'infusio', 'extractio',
            'bevanda', 'sciroppo', 'empiastro',
            'ratione', 'commune', 'secundi',
        ],
        'naming_pattern': 'latin_vernacular_mixed',
        'dia_compounds': 'high',
    },
    'padua_venice': {
        'source': "Pietro d'Abano Conciliator + Antidotarium",
        'characteristic_terms': [
            'antidotum', 'experimentum', 'probatum',
            'olio', 'aqua', 'aceto', 'vino',
            'zenzero', 'zafferano', 'cannella',
            'stercora', 'corallum', 'coralli', 'semen', 'radix',
            'ratione', 'secundi', 'commune', 'codex',
            'diasene', 'diasenna', 'radicom',
            'sero', 'cola', 'tere', 'codi',
        ],
        'naming_pattern': 'arabic_influenced_venetian',
        'dia_compounds': 'very_high',
    },
}

# Arabic pharmaceutical loan words to check
ARABIC_PHARMACEUTICAL_LOANS = [
    'zenzero', 'zafferano', 'cannella', 'sciroppo', 'alambicco',
    'alcool', 'elixir', 'talco', 'borrace', 'canfora',
    'benzoe', 'sandalo', 'aloe', 'tamarindo', 'sena',
]

# Tradition -> dialect affinity weights
TRADITION_DIALECT_MAP: Dict[str, Dict[str, float]] = {
    'salerno': {'venetian': 0.3, 'lombard': 0.3, 'ligurian': 0.3, 'emilian': 0.3, 'tuscan': 0.5},
    'bologna': {'venetian': 0.3, 'lombard': 0.3, 'ligurian': 0.2, 'emilian': 0.8, 'tuscan': 0.2},
    'padua_venice': {'venetian': 0.9, 'lombard': 0.5, 'ligurian': 0.3, 'emilian': 0.3, 'tuscan': 0.1},
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class PharmaRegionResult:
    phase: str                              # "54.4"
    experiment: str                         # "pharma_regionalization"
    n_decoded_words: int
    tradition_scores: Dict[str, float]
    tradition_details: Dict[str, List[Dict]]
    tradition_ranking: List[Dict]           # [{tradition, score}, ...]
    dia_words: List[str]
    arabic_matches: List[Dict]
    dialect_scores: Dict[str, float]
    null_mean: float
    null_std: float
    z_score: float
    selectivity: float
    gates: Dict[str, bool]
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_pharma_region() -> None:
    t0 = time.time()

    print("=" * 70)
    print("PHASE 54.4: Pharmaceutical Terminology Regionalization")
    print("=" * 70)

    rd = _results_dir()

    # ------------------------------------------------------------------
    # Step 1: Load all decoded vocabulary
    # ------------------------------------------------------------------
    print("\n--- Step 1: Loading decoded vocabulary ---")

    # Signal words (Latin 10K)
    with open(os.path.join(rd, 'signal_10k.json')) as f:
        sig_data = json.load(f)
    signal_words: Set[str] = set(
        w['word'] for w in sig_data['word_signals']
        if w.get('is_genuine_signal')
    )

    # Italian signal
    with open(os.path.join(rd, 'italian_signal.json')) as f:
        it_data = json.load(f)
    for w in it_data.get('italian_only_signals', it_data.get('italian_signal_words', [])):
        word = w['word'] if isinstance(w, dict) else w
        signal_words.add(word)

    # T1 identifications from word catalog
    with open(os.path.join(rd, 'word_catalog.json')) as f:
        cat_data = json.load(f)
    t1_words: Set[str] = set()
    for entry in cat_data.get('single_token_ids', []):
        if entry.get('tier') == 'T1':
            t1_words.add(entry['latin_word'])

    all_decoded = signal_words | t1_words
    print(f"  Signal words: {len(signal_words)}")
    print(f"  T1 catalog words: {len(t1_words)}")
    print(f"  Combined unique: {len(all_decoded)}")

    # ------------------------------------------------------------------
    # Step 2: Score each tradition
    # ------------------------------------------------------------------
    print("\n--- Step 2: Scoring pharmaceutical traditions ---")

    tradition_scores: Dict[str, float] = {}
    tradition_details: Dict[str, List[Dict]] = {}

    for name, tradition in PHARMA_TRADITIONS.items():
        matches: List[Dict] = []
        for term in tradition['characteristic_terms']:
            if term in all_decoded:
                matches.append({
                    'term': term,
                    'match_type': 'exact',
                    'score': 1.0,
                })
            elif any(_edit_distance(term, w) <= 1 for w in all_decoded):
                best = min(all_decoded, key=lambda w: _edit_distance(term, w))
                matches.append({
                    'term': term,
                    'match_type': 'ed1',
                    'matched_to': best,
                    'score': 0.5,
                })
            else:
                matches.append({
                    'term': term,
                    'match_type': 'none',
                    'score': 0.0,
                })

        total = sum(m['score'] for m in matches)
        tradition_scores[name] = total / len(tradition['characteristic_terms'])
        tradition_details[name] = matches

        n_exact = sum(1 for m in matches if m['match_type'] == 'exact')
        n_ed1 = sum(1 for m in matches if m['match_type'] == 'ed1')
        print(f"  {name:15s}: score={tradition_scores[name]:.3f}  "
              f"(exact={n_exact}, ed1={n_ed1}, "
              f"total={len(tradition['characteristic_terms'])})")

    # Ranking
    tradition_ranking = sorted(
        [{'tradition': k, 'score': v} for k, v in tradition_scores.items()],
        key=lambda x: x['score'],
        reverse=True,
    )
    print(f"\n  Ranking: {' > '.join(r['tradition'] for r in tradition_ranking)}")

    # ------------------------------------------------------------------
    # Step 3: Arabic-influence marker test
    # ------------------------------------------------------------------
    print("\n--- Step 3: Arabic pharmaceutical loan words ---")

    dia_words = sorted(w for w in all_decoded if w.startswith('dia'))
    print(f"  dia- prefixed words: {dia_words}")

    arabic_matches: List[Dict] = []
    for loan in ARABIC_PHARMACEUTICAL_LOANS:
        for w in all_decoded:
            if _edit_distance(w, loan) <= 1:
                arabic_matches.append({
                    'loan': loan,
                    'decoded': w,
                    'ed': _edit_distance(w, loan),
                })
                break
    print(f"  Arabic loan matches (ed<=1): {len(arabic_matches)}")
    for am in arabic_matches:
        print(f"    {am['loan']} ~ {am['decoded']} (ed={am['ed']})")

    # ------------------------------------------------------------------
    # Step 4: Map traditions to dialect scores
    # ------------------------------------------------------------------
    print("\n--- Step 4: Dialect affinity scores ---")

    dialect_scores: Dict[str, float] = {
        d: 0.0 for d in ['venetian', 'lombard', 'ligurian', 'emilian', 'tuscan']
    }
    total_weight = sum(tradition_scores.values())
    if total_weight > 0:
        for trad, score in tradition_scores.items():
            for dialect, affinity in TRADITION_DIALECT_MAP[trad].items():
                dialect_scores[dialect] += score * affinity / total_weight

    for dialect in sorted(dialect_scores, key=dialect_scores.get, reverse=True):
        print(f"  {dialect:12s}: {dialect_scores[dialect]:.4f}")

    # ------------------------------------------------------------------
    # Step 5: Null test (1000 iterations)
    # ------------------------------------------------------------------
    print("\n--- Step 5: Permutation null test (1000 trials) ---")

    rng = random.Random(42)
    all_terms: List[str] = []
    for trad in PHARMA_TRADITIONS.values():
        all_terms.extend(trad['characteristic_terms'])

    null_best_scores: List[float] = []
    for _ in range(1000):
        shuffled = list(all_terms)
        rng.shuffle(shuffled)
        # Split back into 3 groups of original sizes
        idx = 0
        null_tradition_scores: Dict[str, float] = {}
        for name, tradition in PHARMA_TRADITIONS.items():
            n = len(tradition['characteristic_terms'])
            null_terms = shuffled[idx:idx + n]
            idx += n
            match_count = sum(
                1 for t in null_terms
                if t in all_decoded
                or any(_edit_distance(t, w) <= 1 for w in all_decoded)
            )
            null_tradition_scores[name] = match_count / n
        null_best_scores.append(max(null_tradition_scores.values()))

    null_mean = sum(null_best_scores) / len(null_best_scores)
    null_std = (
        sum((s - null_mean) ** 2 for s in null_best_scores) / len(null_best_scores)
    ) ** 0.5
    real_best = max(tradition_scores.values())
    z_score = (real_best - null_mean) / null_std if null_std > 0 else 0.0
    selectivity = real_best / null_mean if null_mean > 0 else float('inf')

    print(f"  Real best tradition score: {real_best:.4f}")
    print(f"  Null mean: {null_mean:.4f}  std: {null_std:.4f}")
    print(f"  z-score: {z_score:.2f}")
    print(f"  Selectivity: {selectivity:.2f}x")

    # ------------------------------------------------------------------
    # Step 6: Gates and verdict
    # ------------------------------------------------------------------
    print("\n--- Step 6: Gates ---")

    # G1: >= 5 tradition-specific terms matched across all traditions
    total_matches = sum(
        1 for name in PHARMA_TRADITIONS
        for m in tradition_details[name]
        if m['match_type'] in ('exact', 'ed1')
    )
    g1 = total_matches >= 5

    # G2: top tradition >= 2x second-best
    sorted_scores = sorted(tradition_scores.values(), reverse=True)
    g2 = (
        sorted_scores[0] >= 2 * sorted_scores[1]
        if len(sorted_scores) >= 2 and sorted_scores[1] > 0
        else False
    )

    # G3: null selectivity >= 1.5
    g3 = selectivity >= 1.5

    gates = {'G1_min_matches': g1, 'G2_tradition_dominance': g2, 'G3_selectivity': g3}
    n_passed = sum(gates.values())

    if n_passed == 3:
        verdict = 'STRONG_REGIONAL_SIGNAL'
    elif n_passed == 2:
        verdict = 'MODERATE_REGIONAL_SIGNAL'
    elif n_passed == 1:
        verdict = 'WEAK_REGIONAL_SIGNAL'
    else:
        verdict = 'NO_REGIONAL_SIGNAL'

    print(f"  G1 (>= 5 term matches):     {g1}  ({total_matches} matched)")
    print(f"  G2 (top >= 2x second):       {g2}  "
          f"({sorted_scores[0]:.3f} vs {sorted_scores[1]:.3f})")
    print(f"  G3 (selectivity >= 1.5):     {g3}  ({selectivity:.2f}x)")
    print(f"\n  Verdict: {verdict}")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    runtime = time.time() - t0

    result = PharmaRegionResult(
        phase='54.4',
        experiment='pharma_regionalization',
        n_decoded_words=len(all_decoded),
        tradition_scores=tradition_scores,
        tradition_details=tradition_details,
        tradition_ranking=tradition_ranking,
        dia_words=dia_words,
        arabic_matches=arabic_matches,
        dialect_scores=dialect_scores,
        null_mean=null_mean,
        null_std=null_std,
        z_score=z_score,
        selectivity=selectivity,
        gates=gates,
        verdict=verdict,
        runtime_seconds=round(runtime, 2),
    )

    out_path = os.path.join(rd, 'phase54_pharma_region.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)

    print(f"\n  Saved: {out_path}")
    print(f"  Runtime: {runtime:.1f}s")
    print("=" * 70)
