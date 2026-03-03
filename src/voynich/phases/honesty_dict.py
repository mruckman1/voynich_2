"""
Phase 17.0.1 – Dictionary Tier Control Test
============================================
Scores the Phase 16 modifier-corrected decoded output against three
dictionaries of different sizes to determine how much of the 51.6%
dict_hit is driven by dictionary expansion vs genuine decoding.

Dependency chain:
    modifier_integrate.json  (Phase 16 best result)
    combined_refine.json     (Phase 15 best_assignment)
        → honesty_dict.json  (this step)
"""

import json
import os
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    decode_token_modifier_aware,
    load_corpus,
    tokenize_eva_chars,
)
from voynich.core.reference import (
    build_expanded_word_set,
    load_reference_corpus,
)
from voynich.phases.csp_solver import decode_token


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


def _reconstruct_modifier_rules(data: Dict) -> Tuple[Set[str], Dict[str, str]]:
    """Extract modifier_chars set and modifier_rules dict from modifier_integrate data."""
    modifier_chars = set(data.get('modifier_chars', []))
    modifier_rules: Dict[str, str] = {}
    for c in data.get('classifications', []):
        if c.get('final_classification') == 'modifier':
            modifier_rules[c['eva_char']] = c.get('modifier_type', 'silent')
    return modifier_chars, modifier_rules


def _build_core_dictionary(ref_corpus, base_words: set) -> set:
    """Build a strict core dictionary: high-frequency, length >= 3, no variants."""
    token_counts: Counter = Counter()
    for w in ref_corpus.get_combined_tokens('latin'):
        token_counts[w.lower()] += 1
    core = set()
    for word, count in token_counts.items():
        if count >= 2 and len(word) >= 3 and word in base_words:
            core.add(word)
    return core


def _compute_dict_hit(decoded: List[str], ref_word_set: set) -> float:
    if not decoded:
        return 0.0
    hits = sum(1 for w in decoded if w.lower() in ref_word_set)
    return hits / len(decoded)


def _compute_selectivity(
    dict_hit: float,
    tokens: List[str],
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    ref_word_set: set,
    n_random: int = 5,
) -> float:
    """Selectivity ratio: dict_hit / random_baseline."""
    rng = random.Random(42)
    syllables = list(set(assignment.values()))
    if not syllables:
        return 0.0

    random_hits = []
    for _ in range(n_random):
        rand_assignment = {k: rng.choice(syllables) for k in assignment}
        decoded = [
            decode_token(t, rand_assignment, eva_to_triple)
            for t in tokens[:2000]
        ]
        random_hits.append(_compute_dict_hit(decoded, ref_word_set))

    baseline = sum(random_hits) / len(random_hits) if random_hits else 0.01
    return dict_hit / max(baseline, 0.001)


def _r3_decode(
    tokens: List[str],
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    modifier_chars: Set[str],
    modifier_rules: Dict[str, str],
    ref_word_set: set,
    max_tokens: int = 8000,
) -> List[str]:
    """Reproduce Phase 16 R3 combined decode."""
    decoded = []
    for token in tokens[:max_tokens]:
        alt = decode_token_modifier_aware(
            token, assignment, eva_to_triple, modifier_chars,
            modifier_rules=modifier_rules,
        )
        if alt.lower() in ref_word_set:
            decoded.append(alt)
            continue
        stripped = decode_token_modifier_aware(
            token, assignment, eva_to_triple, modifier_chars,
        )
        if stripped.lower() in ref_word_set:
            decoded.append(stripped)
            continue
        original = decode_token(token, assignment, eva_to_triple)
        decoded.append(original)
    return decoded


def _r1_decode(
    tokens: List[str],
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    modifier_chars: Set[str],
    max_tokens: int = 8000,
) -> List[str]:
    """R1: Strip modifier chars before decoding."""
    return [
        decode_token_modifier_aware(token, assignment, eva_to_triple, modifier_chars)
        for token in tokens[:max_tokens]
    ]


def _naive_decode(
    tokens: List[str],
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    max_tokens: int = 8000,
) -> List[str]:
    """Naive: No modifier handling."""
    return [
        decode_token(token, assignment, eva_to_triple)
        for token in tokens[:max_tokens]
    ]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DictTierResult:
    tier_name: str
    dict_size: int
    dict_hit_rate: float
    n_hits: int
    n_tokens: int
    selectivity: float
    random_baseline: float
    hit_words_sample: List[str]


@dataclass
class HonestyDictResult:
    # Tier results for R3 decode
    tiers: List[Dict]

    original_hit: float
    expanded_hit: float
    core_hit: float

    original_selectivity: float
    expanded_selectivity: float
    core_selectivity: float

    # Cross-strategy comparison (all against original dict)
    r3_original_hit: float
    r1_strip_original_hit: float
    naive_original_hit: float

    # Dictionary sizes
    original_dict_size: int
    expanded_dict_size: int
    core_dict_size: int

    n_triples: int
    n_tokens_tested: int

    gate_passed: bool
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_honesty_dict() -> None:
    """Step 17.0.1: Multi-dictionary scoring control test."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE 17.0.1: Dictionary Tier Control Test")
    print("=" * 70)

    rd = _results_dir()

    # ─── Load Phase 16 results ───
    print("\n  1. Loading Phase 16 results …")
    mod_path = os.path.join(rd, 'modifier_integrate.json')
    if not os.path.exists(mod_path):
        print("  [SKIP] modifier_integrate.json not found")
        return
    with open(mod_path) as f:
        mod_data = json.load(f)
    modifier_chars, modifier_rules = _reconstruct_modifier_rules(mod_data)
    print(f"      {len(modifier_chars)} modifier chars")

    # ─── Load Phase 15 assignment ───
    refine_path = os.path.join(rd, 'combined_refine.json')
    if not os.path.exists(refine_path):
        print("  [SKIP] combined_refine.json not found")
        return
    with open(refine_path) as f:
        refine_data = json.load(f)
    assignment = refine_data.get('best_assignment', {})
    print(f"      {len(assignment)} triple assignments")

    # ─── Load corpus ───
    print("\n  2. Loading corpus …")
    corpus = load_corpus(verbose=False)
    tokens = corpus.get_tokens()
    eva_to_triple = build_eva_to_triple_lookup()
    print(f"      {len(tokens)} tokens")

    # ─── Build three dictionaries ───
    print("\n  3. Building dictionaries …")
    try:
        ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
        base_words = set(
            w.lower() for w in ref_corpus.get_combined_tokens('latin')
            if len(w) >= 2
        )
    except (FileNotFoundError, KeyError):
        base_words = set()

    expanded_words, _ = build_expanded_word_set(base_words)
    expanded_set = base_words | expanded_words
    core_set = _build_core_dictionary(ref_corpus, base_words)

    print(f"      Original:  {len(base_words):>7,} words")
    print(f"      Expanded:  {len(expanded_set):>7,} words")
    print(f"      Core:      {len(core_set):>7,} words")

    # ─── R3 decode (using expanded dict for hit selection, matching Phase 16) ───
    print("\n  4. Decoding all tokens (R3 combined strategy) …")
    r3_decoded = _r3_decode(
        tokens, assignment, eva_to_triple,
        modifier_chars, modifier_rules, expanded_set,
    )
    print(f"      Decoded {len(r3_decoded)} tokens")

    # ─── R1 and naive decode ───
    print("\n  5. Decoding with R1 (strip) and naive (no modifiers) …")
    r1_decoded = _r1_decode(tokens, assignment, eva_to_triple, modifier_chars)
    naive_decoded = _naive_decode(tokens, assignment, eva_to_triple)

    # ─── Score R3 against all 3 dictionaries ───
    print("\n  6. Scoring R3 decoded output against 3 dictionaries …")
    tiers = []
    for tier_name, word_set in [
        ('original', base_words),
        ('expanded', expanded_set),
        ('core', core_set),
    ]:
        hit_rate = _compute_dict_hit(r3_decoded, word_set)
        n_hits = sum(1 for w in r3_decoded if w.lower() in word_set)
        selectivity = _compute_selectivity(
            hit_rate, tokens, assignment, eva_to_triple, word_set,
        )
        # Random baseline
        rng = random.Random(42)
        syllables = list(set(assignment.values()))
        rand_hits = []
        for _ in range(5):
            rand_assign = {k: rng.choice(syllables) for k in assignment}
            rand_decoded = [decode_token(t, rand_assign, eva_to_triple) for t in tokens[:2000]]
            rand_hits.append(_compute_dict_hit(rand_decoded, word_set))
        random_baseline = sum(rand_hits) / len(rand_hits) if rand_hits else 0.01

        # Sample of hit words
        hit_words = sorted(set(w.lower() for w in r3_decoded if w.lower() in word_set))
        hit_sample = hit_words[:20]

        tier = DictTierResult(
            tier_name=tier_name,
            dict_size=len(word_set),
            dict_hit_rate=round(hit_rate, 4),
            n_hits=n_hits,
            n_tokens=len(r3_decoded),
            selectivity=round(selectivity, 2),
            random_baseline=round(random_baseline, 4),
            hit_words_sample=hit_sample,
        )
        tiers.append(tier)
        print(f"      {tier_name:>10}: hit={hit_rate:.1%}, selectivity={selectivity:.2f}×, "
              f"baseline={random_baseline:.1%}, n_hits={n_hits}")

    # ─── Cross-strategy comparison against original dict ───
    print("\n  7. Cross-strategy comparison (all against original dict) …")
    r3_orig = _compute_dict_hit(r3_decoded, base_words)
    r1_orig = _compute_dict_hit(r1_decoded, base_words)
    naive_orig = _compute_dict_hit(naive_decoded, base_words)
    print(f"      R3 combined:  {r3_orig:.1%}")
    print(f"      R1 strip:     {r1_orig:.1%}")
    print(f"      Naive:        {naive_orig:.1%}")

    # ─── Gate ───
    original_hit = tiers[0].dict_hit_rate
    gate_passed = original_hit > 0.25

    print(f"\n  8. Gate: original_hit > 0.25")
    print(f"      original_hit = {original_hit:.1%}")
    print(f"      {'PASS' if gate_passed else 'FAIL'}")

    # ─── Verdict ───
    expanded_hit = tiers[1].dict_hit_rate
    core_hit = tiers[2].dict_hit_rate

    if gate_passed:
        verdict = (
            f"PASS: R3 decoded output scores {original_hit:.1%} against original dict "
            f"(expanded={expanded_hit:.1%}, core={core_hit:.1%}). "
            f"Signal survives dictionary reduction."
        )
    elif original_hit > 0.15:
        verdict = (
            f"MARGINAL: original_hit={original_hit:.1%} (gate=25%). "
            f"Some genuine signal but partially confounded with dictionary expansion. "
            f"Expanded={expanded_hit:.1%}, core={core_hit:.1%}."
        )
    else:
        verdict = (
            f"FAIL: original_hit={original_hit:.1%} (gate=25%). "
            f"The Phase 15-16 gains ({expanded_hit:.1%}) are driven by "
            f"dictionary expansion, not improved decoding."
        )

    print(f"\n  Verdict: {verdict}")

    # ─── Save ───
    result = HonestyDictResult(
        tiers=[_convert(asdict(t)) for t in tiers],
        original_hit=round(tiers[0].dict_hit_rate, 4),
        expanded_hit=round(tiers[1].dict_hit_rate, 4),
        core_hit=round(tiers[2].dict_hit_rate, 4),
        original_selectivity=round(tiers[0].selectivity, 2),
        expanded_selectivity=round(tiers[1].selectivity, 2),
        core_selectivity=round(tiers[2].selectivity, 2),
        r3_original_hit=round(r3_orig, 4),
        r1_strip_original_hit=round(r1_orig, 4),
        naive_original_hit=round(naive_orig, 4),
        original_dict_size=len(base_words),
        expanded_dict_size=len(expanded_set),
        core_dict_size=len(core_set),
        n_triples=len(assignment),
        n_tokens_tested=len(r3_decoded),
        gate_passed=gate_passed,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'honesty_dict.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)
    print(f"\n  → {out_path}")
