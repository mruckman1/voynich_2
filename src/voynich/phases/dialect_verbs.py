"""
Phase 54.6: Verb Morphology Deep Dive
======================================
Test decoded signal words that form verb paradigms against five
northern-Italian dialects (Venetian, Lombard, Ligurian, Emilian,
Tuscan).  Score each dialect by edit-distance match, check paradigm
coherence, analyse variant distribution across scribal hands and
manuscript sections, and run a null permutation test.

Output:
  results/phase54_verb_morph.json
"""

import json
import os
import random
import time
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Any, Set, Tuple, Optional

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


def _chi2_test(table):
    """2D contingency table chi-squared test. table = list of lists."""
    rows = len(table)
    cols = len(table[0])
    row_sums = [sum(row) for row in table]
    col_sums = [sum(table[r][c] for r in range(rows)) for c in range(cols)]
    total = sum(row_sums)
    if total == 0:
        return 0.0, 1.0
    chi2 = 0.0
    for r in range(rows):
        for c in range(cols):
            expected = row_sums[r] * col_sums[c] / total
            if expected > 0:
                chi2 += (table[r][c] - expected) ** 2 / expected
    df = max(1, (rows - 1) * (cols - 1))
    p = _chi2_survival(chi2, df)
    return chi2, p


def _chi2_survival(x, k):
    """Approximate chi-squared survival function P(X > x) for k degrees of freedom."""
    if x <= 0:
        return 1.0
    # Use normal approximation for chi-squared
    z = ((x / k) ** (1/3) - (1 - 2/(9*k))) / math.sqrt(2/(9*k))
    # Standard normal CDF approximation
    p = 0.5 * (1.0 + math.erf(-z / math.sqrt(2)))
    return max(0.0, min(1.0, p))


# ---------------------------------------------------------------------------
# Verb paradigm data — decoded signal words
# ---------------------------------------------------------------------------

VERB_PARADIGM_MAP = {
    'dire': {
        'dise': {'slot': '3sg_pres_ind', 'dialect_forms': {
            'venetian': ['dise'], 'lombard': ['dis'], 'ligurian': ['dixe'],
            'emilian': ['dis'], 'tuscan': ['dice']
        }},
        'dice': {'slot': '3sg_pres_ind', 'dialect_forms': {
            'venetian': ['dise'], 'lombard': ['dis'], 'ligurian': ['dixe'],
            'emilian': ['dis'], 'tuscan': ['dice']
        }},
        'dico': {'slot': '1sg_pres_ind', 'dialect_forms': {
            'venetian': ['digo'], 'lombard': ['disi'], 'ligurian': ['digo'],
            'emilian': ['dig'], 'tuscan': ['dico']
        }},
        'dicu': {'slot': '1sg_pres_ind_archaic', 'dialect_forms': {
            'venetian': ['digo'], 'lombard': ['disi'], 'ligurian': ['digo'],
            'emilian': ['dig'], 'tuscan': ['dico']
        }},
        'diga': {'slot': '3sg_pres_subj', 'dialect_forms': {
            'venetian': ['diga'], 'lombard': ['diga'], 'ligurian': ['diga'],
            'emilian': ['diga'], 'tuscan': ['dica']
        }},
    },
    'dare': {
        'dedi': {'slot': '1sg_perf', 'dialect_forms': {
            'venetian': ['dedi', 'diedi'], 'lombard': ['dedi'], 'ligurian': ['deti'],
            'emilian': ['det'], 'tuscan': ['diedi']
        }},
        'dido': {'slot': '1sg_perf_var', 'dialect_forms': {
            'venetian': ['didi'], 'lombard': ['dedi'], 'ligurian': ['deti'],
            'emilian': ['det'], 'tuscan': ['diedi']
        }},
        'dere': {'slot': 'infinitive', 'dialect_forms': {
            'venetian': ['dar'], 'lombard': ['da'], 'ligurian': ['da'],
            'emilian': ['dar'], 'tuscan': ['dare']
        }},
    },
}

DIALECTS = ['venetian', 'lombard', 'ligurian', 'emilian', 'tuscan']

# Variant forms to track in distribution analysis
VARIANT_FORMS = ['dise', 'dice']


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class VerbMorphResult:
    phase: str
    experiment: str
    forms_analyzed: List[Dict]
    per_dialect_verb_scores: Dict[str, float]
    paradigm_coherence: Dict[str, bool]
    variant_distribution: Dict
    dialect_scores: Dict[str, float]
    null_mean: float
    null_std: float
    z_score: float
    selectivity: float
    gates: Dict[str, bool]
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Step 1: Score each dialect by form match
# ---------------------------------------------------------------------------

def _score_dialect(dialect: str) -> Tuple[float, List[Dict]]:
    """Compute mean edit-distance match score for a dialect across all verb forms.

    Returns (score, form_details).
    """
    form_details = []
    scores = []
    for verb, forms in VERB_PARADIGM_MAP.items():
        for decoded_form, info in forms.items():
            expected_forms = info['dialect_forms'].get(dialect, [])
            if not expected_forms:
                continue
            min_ed = min(_edit_distance(decoded_form, exp) for exp in expected_forms)
            max_len = max(len(decoded_form), max(len(exp) for exp in expected_forms))
            if max_len == 0:
                score = 1.0
            else:
                score = 1.0 - min_ed / max_len
            scores.append(score)
            best_exp = min(expected_forms, key=lambda exp: _edit_distance(decoded_form, exp))
            form_details.append({
                'decoded': decoded_form,
                'verb': verb,
                'slot': info['slot'],
                'best_dialect': dialect,
                'best_expected': best_exp,
                'ed': min_ed,
                'score': score,
            })
    if not scores:
        return 0.0, form_details
    return sum(scores) / len(scores), form_details


# ---------------------------------------------------------------------------
# Step 2: Paradigm coherence
# ---------------------------------------------------------------------------

def _check_paradigm_coherence(dialect: str) -> bool:
    """Check if ALL attested forms have min_ed <= 1 to at least one expected form."""
    for verb, forms in VERB_PARADIGM_MAP.items():
        for decoded_form, info in forms.items():
            expected_forms = info['dialect_forms'].get(dialect, [])
            if not expected_forms:
                return False
            min_ed = min(_edit_distance(decoded_form, exp) for exp in expected_forms)
            if min_ed > 1:
                return False
    return True


# ---------------------------------------------------------------------------
# Step 3 & 4: Variant distribution across hands and sections
# ---------------------------------------------------------------------------

def _analyse_variant_distribution(corpus, assignment, modifier_chars,
                                  modifier_rules, eva_to_triple):
    """Decode all tokens. Track 'dise' and 'dice' by hand and section.

    Returns dict with by_hand, hand_chi2, hand_p, by_section, section_chi2,
    section_p.
    """
    # Accumulate counts by hand and section
    hand_counts = defaultdict(lambda: Counter())   # hand -> Counter{'dise': n, 'dice': m}
    section_counts = defaultdict(lambda: Counter())  # section -> Counter

    for folio_id, page in corpus.pages.items():
        hand = page.hand
        section = page.section
        tokens = page.all_tokens
        for token in tokens:
            decoded = decode_token_modifier_aware(
                token, assignment, eva_to_triple, modifier_chars,
                modifier_rules=modifier_rules,
            ).lower()
            if decoded in VARIANT_FORMS:
                hand_counts[hand][decoded] += 1
                section_counts[section][decoded] += 1

    # Build hand contingency table (rows = sorted hands, cols = [dise, dice])
    all_hands = sorted(hand_counts.keys())
    hand_table = []
    by_hand = {}
    for h in all_hands:
        row = [hand_counts[h].get('dise', 0), hand_counts[h].get('dice', 0)]
        hand_table.append(row)
        by_hand[str(h)] = {'dise': row[0], 'dice': row[1]}

    if hand_table:
        hand_chi2, hand_p = _chi2_test(hand_table)
    else:
        hand_chi2, hand_p = 0.0, 1.0

    # Build section contingency table
    all_sections = sorted(section_counts.keys())
    section_table = []
    by_section = {}
    for s in all_sections:
        row = [section_counts[s].get('dise', 0), section_counts[s].get('dice', 0)]
        section_table.append(row)
        by_section[s] = {'dise': row[0], 'dice': row[1]}

    if section_table:
        section_chi2, section_p = _chi2_test(section_table)
    else:
        section_chi2, section_p = 0.0, 1.0

    return {
        'by_hand': by_hand,
        'hand_chi2': hand_chi2,
        'hand_p': hand_p,
        'by_section': by_section,
        'section_chi2': section_chi2,
        'section_p': section_p,
    }


# ---------------------------------------------------------------------------
# Step 6: Null test
# ---------------------------------------------------------------------------

def _build_slot_pool() -> Dict[str, List[str]]:
    """Pool all expected forms across all dialects for each slot."""
    slot_pool: Dict[str, Set[str]] = defaultdict(set)
    for verb, forms in VERB_PARADIGM_MAP.items():
        for decoded_form, info in forms.items():
            slot = info['slot']
            for dialect, expected in info['dialect_forms'].items():
                for exp in expected:
                    slot_pool[slot].add(exp)
    return {slot: list(forms) for slot, forms in slot_pool.items()}


def _null_score_trial(slot_pool: Dict[str, List[str]], rng: random.Random) -> float:
    """Score a single null trial: randomly assign forms, compute best dialect score."""
    # Collect all form entries with their slots
    all_entries = []
    for verb, forms in VERB_PARADIGM_MAP.items():
        for decoded_form, info in forms.items():
            all_entries.append((decoded_form, info['slot'], info['dialect_forms']))

    # Randomly assign each form a random expected form from the slot pool
    random_forms = {}
    for decoded_form, slot, dialect_forms in all_entries:
        pool = slot_pool.get(slot, [decoded_form])
        random_forms[decoded_form] = rng.choice(pool)

    # Score each dialect against the random assignment
    best_score = 0.0
    for dialect in DIALECTS:
        scores = []
        for decoded_form, slot, dialect_forms in all_entries:
            random_form = random_forms[decoded_form]
            expected = dialect_forms.get(dialect, [])
            if not expected:
                continue
            min_ed = min(_edit_distance(random_form, exp) for exp in expected)
            max_len = max(len(random_form), max(len(exp) for exp in expected))
            if max_len == 0:
                s = 1.0
            else:
                s = 1.0 - min_ed / max_len
            scores.append(s)
        if scores:
            dialect_score = sum(scores) / len(scores)
            if dialect_score > best_score:
                best_score = dialect_score
    return best_score


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_verb_morph() -> None:
    t0 = time.time()
    rd = _results_dir()

    print("=" * 70)
    print("PHASE 54.6: Verb Morphology Deep Dive")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 1: Score each dialect by form match
    # ------------------------------------------------------------------
    print("\n--- Step 1: Per-dialect edit-distance scoring ---")

    per_dialect_verb_scores: Dict[str, float] = {}
    all_form_details: List[Dict] = []
    best_dialect = None
    best_dialect_score = -1.0

    for dialect in DIALECTS:
        score, details = _score_dialect(dialect)
        per_dialect_verb_scores[dialect] = round(score, 4)
        print(f"  {dialect:12s}: score = {score:.4f}")
        if score > best_dialect_score:
            best_dialect_score = score
            best_dialect = dialect

    # Collect form details for the best dialect
    _, best_details = _score_dialect(best_dialect)
    all_form_details = best_details
    n_forms = len(all_form_details)
    print(f"\n  Best dialect: {best_dialect} ({best_dialect_score:.4f})")
    print(f"  Forms analysed: {n_forms}")

    # ------------------------------------------------------------------
    # Step 2: Paradigm coherence
    # ------------------------------------------------------------------
    print("\n--- Step 2: Paradigm coherence test ---")

    paradigm_coherence: Dict[str, bool] = {}
    for dialect in DIALECTS:
        coherent = _check_paradigm_coherence(dialect)
        paradigm_coherence[dialect] = coherent
        tag = "COHERENT" if coherent else "incoherent"
        print(f"  {dialect:12s}: {tag}")

    n_coherent = sum(1 for v in paradigm_coherence.values() if v)
    print(f"\n  Coherent dialects: {n_coherent}/5")

    # ------------------------------------------------------------------
    # Step 3 & 4: Variant distribution across hands and sections
    # ------------------------------------------------------------------
    print("\n--- Step 3-4: Variant distribution (dise/dice) ---")

    corpus = load_corpus(verbose=False)

    # Load assignment table
    refine_path = os.path.join(rd, 'combined_refine.json')
    with open(refine_path) as f:
        refine_data = json.load(f)
    assignment = refine_data['best_assignment']

    # Load modifier data
    mod_path = os.path.join(rd, 'modifier_integrate.json')
    with open(mod_path) as f:
        mod_data = json.load(f)
    modifier_chars = set(mod_data.get('modifier_chars', []))
    modifier_rules = {}
    for c in mod_data.get('classifications', []):
        if c.get('final_classification') == 'modifier':
            modifier_rules[c['eva_char']] = c.get('modifier_type', 'silent')

    eva_to_triple = build_eva_to_triple_lookup()

    variant_dist = _analyse_variant_distribution(
        corpus, assignment, modifier_chars, modifier_rules, eva_to_triple)

    print(f"  By hand:")
    for h, counts in sorted(variant_dist['by_hand'].items()):
        print(f"    Hand {h}: dise={counts['dise']}, dice={counts['dice']}")
    print(f"  Hand chi2={variant_dist['hand_chi2']:.4f}, p={variant_dist['hand_p']:.4f}")

    print(f"  By section:")
    for s, counts in sorted(variant_dist['by_section'].items()):
        total = counts['dise'] + counts['dice']
        if total > 0:
            print(f"    {s:20s}: dise={counts['dise']}, dice={counts['dice']}")
    print(f"  Section chi2={variant_dist['section_chi2']:.4f}, p={variant_dist['section_p']:.4f}")

    # ------------------------------------------------------------------
    # Step 5: Combine per-dialect verb scores
    # ------------------------------------------------------------------
    dialect_scores = dict(per_dialect_verb_scores)

    # ------------------------------------------------------------------
    # Step 6: Null test (1000 iterations)
    # ------------------------------------------------------------------
    print("\n--- Step 6: Null permutation test (1000 trials) ---")

    slot_pool = _build_slot_pool()
    rng = random.Random(42)
    null_scores = []
    for _ in range(1000):
        null_scores.append(_null_score_trial(slot_pool, rng))

    null_mean = sum(null_scores) / len(null_scores)
    null_std = (sum((s - null_mean) ** 2 for s in null_scores) / len(null_scores)) ** 0.5
    if null_std > 0:
        z_score = (best_dialect_score - null_mean) / null_std
    else:
        z_score = 0.0
    if null_mean > 0:
        selectivity = best_dialect_score / null_mean
    else:
        selectivity = 0.0

    print(f"  Observed best score: {best_dialect_score:.4f} ({best_dialect})")
    print(f"  Null mean: {null_mean:.4f}, std: {null_std:.4f}")
    print(f"  z-score: {z_score:.2f}")
    print(f"  Selectivity: {selectivity:.2f}x")

    # ------------------------------------------------------------------
    # Step 7: Gates
    # ------------------------------------------------------------------
    print("\n--- Step 7: Gate evaluation ---")

    g1 = n_forms >= 5
    g2 = best_dialect_score >= 0.7
    g3 = n_coherent <= 2
    g4 = (variant_dist['hand_p'] < 0.10) or (variant_dist['hand_p'] > 0.50)

    gates = {
        'G1_enough_forms': g1,
        'G2_top_score_ge_0.7': g2,
        'G3_le_2_coherent': g3,
        'G4_hand_interpretable': g4,
    }

    all_pass = all(gates.values())
    verdict = "VERB_MORPH_PASS" if all_pass else "VERB_MORPH_FAIL"

    for gname, gpassed in gates.items():
        tag = "PASS" if gpassed else "FAIL"
        print(f"  {gname}: {tag}")
    print(f"\n  Verdict: {verdict}")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    runtime = time.time() - t0

    result = VerbMorphResult(
        phase="54.6",
        experiment="verb_morphology",
        forms_analyzed=all_form_details,
        per_dialect_verb_scores=per_dialect_verb_scores,
        paradigm_coherence=paradigm_coherence,
        variant_distribution=variant_dist,
        dialect_scores=dialect_scores,
        null_mean=round(null_mean, 4),
        null_std=round(null_std, 4),
        z_score=round(z_score, 2),
        selectivity=round(selectivity, 2),
        gates=gates,
        verdict=verdict,
        runtime_seconds=round(runtime, 2),
    )

    out_path = os.path.join(rd, 'phase54_verb_morph.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)

    print(f"\n  Runtime: {runtime:.1f}s")
    print(f"  Saved: {out_path}")
