"""
Phase 54.5: "Co" Syntactic Validation
=======================================
Test whether the decoded word "co" (hypothesised preposition "with" in
Venetian/Lombard) behaves syntactically like a preposition by checking
whether it is preferentially followed by nouns/content words.  Two null
tests quantify selectivity: (A) position-shuffle and (B) word-shuffle.
Compute dialect scores based on whether prepositional behaviour is
confirmed.

Output:
  results/phase54_co_syntax.json
"""

import json
import os
import random
import time
from collections import Counter
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


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class CoSyntaxResult:
    phase: str                          # "54.5"
    experiment: str                     # "co_syntactic_validation"
    n_co_occurrences: int
    noun_rate: float
    content_rate: float                 # fraction followed by 3+ char words
    signal_rate: float                  # fraction followed by signal words
    null_a_mean: float
    null_a_std: float
    z_a: float
    null_b_mean: float
    null_b_std: float
    z_b: float
    selectivity: float                  # noun_rate / null_a_mean
    found_bigrams: List[str]
    dialect_scores: Dict[str, float]
    co_details_sample: List[Dict]       # first 20 co positions with next word
    gates: Dict[str, bool]
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_co_syntax() -> None:
    t0 = time.time()

    print("=" * 70)
    print("PHASE 54.5: Co Syntactic Validation")
    print("=" * 70)

    rd = _results_dir()

    # ------------------------------------------------------------------
    # Step 1: Load assignment table, modifier data, and signal words
    # ------------------------------------------------------------------

    # Assignment table (T_P15)
    with open(os.path.join(rd, 'combined_refine.json')) as f:
        assignment = json.load(f)['best_assignment']

    # Modifier data
    with open(os.path.join(rd, 'modifier_integrate.json')) as f:
        mod_data = json.load(f)
    modifier_chars = set(mod_data.get('modifier_chars', []))
    modifier_rules: Dict[str, str] = {}
    for c in mod_data.get('classifications', []):
        if c.get('final_classification') == 'modifier':
            modifier_rules[c['eva_char']] = c.get('modifier_type', 'silent')

    # Eva-to-triple lookup
    eva_to_triple = build_eva_to_triple_lookup()

    # Signal words (for classification)
    signal_words: Set[str] = set()

    path_10k = os.path.join(rd, 'signal_10k.json')
    if os.path.exists(path_10k):
        with open(path_10k) as f:
            sig_data = json.load(f)
        for w in sig_data.get('word_signals', []):
            if w.get('is_genuine_signal'):
                signal_words.add(w['word'])
        print(f"  signal_10k.json: {len(sig_data.get('word_signals', []))} words, "
              f"{sum(1 for w in sig_data.get('word_signals', []) if w.get('is_genuine_signal'))} genuine signal")
    else:
        print(f"  WARNING: {path_10k} not found")

    # Italian signal words
    path_it = os.path.join(rd, 'italian_signal.json')
    if os.path.exists(path_it):
        with open(path_it) as f:
            it_data = json.load(f)
        for w in it_data.get('italian_only_signals', it_data.get('italian_signal_words', [])):
            if isinstance(w, dict):
                signal_words.add(w['word'])
            else:
                signal_words.add(w)
        print(f"  italian_signal.json loaded")
    else:
        print(f"  WARNING: {path_it} not found")

    print(f"  Total signal words: {len(signal_words)}")

    # ------------------------------------------------------------------
    # Step 2: Build noun candidates from T1 identifications + signal
    # ------------------------------------------------------------------

    # T1 identifications (all are nouns/content words)
    t1_words: Set[str] = set()
    path_cat = os.path.join(rd, 'word_catalog.json')
    if os.path.exists(path_cat):
        with open(path_cat) as f:
            cat_data = json.load(f)
        for entry in cat_data.get('single_token_ids', []):
            if entry.get('tier') == 'T1':
                t1_words.add(entry['latin_word'])
        print(f"  word_catalog.json: {len(t1_words)} T1 words")
    else:
        print(f"  WARNING: {path_cat} not found")

    # Noun candidates: T1 words + signal content words (3+ chars)
    NOUN_CANDIDATES = t1_words | {
        'sero', 'cola', 'sene', 'codi', 'tere', 'raso', 'cora', 'bela',
        'dice', 'cose', 'bene', 'nera', 'sera', 'sede', 'tela', 'raro',
        'diri', 'sere', 'rati', 'dira',
        'ratione', 'coralli', 'diasene', 'stercora', 'radicom', 'commune',
        'codex', 'secundi', 'rabidi',
    }
    print(f"  Noun candidates: {len(NOUN_CANDIDATES)}")

    # ------------------------------------------------------------------
    # Step 3: Load corpus and decode ALL tokens
    # ------------------------------------------------------------------

    corpus = load_corpus(verbose=False)
    all_tokens: List[str] = []   # EVA tokens
    all_folios: List[str] = []   # parallel folio list
    for folio_id, page in corpus.pages.items():
        tokens = page.paragraph_text.split()
        for t in tokens:
            all_tokens.append(t)
            all_folios.append(folio_id)

    print(f"  Corpus: {len(all_tokens)} tokens across {len(corpus.pages)} folios")

    # Decode all tokens
    all_decoded: List[str] = []
    for token in all_tokens:
        decoded = decode_token_modifier_aware(
            token, assignment, eva_to_triple, modifier_chars,
            modifier_rules=modifier_rules,
        )
        all_decoded.append(decoded.lower())

    print(f"  Decoded {len(all_decoded)} tokens")

    # ------------------------------------------------------------------
    # Step 4: Find all positions where decoded token = "co"
    # ------------------------------------------------------------------

    co_positions: List[Dict] = []
    for i, dec in enumerate(all_decoded):
        if dec == 'co':
            next_dec = all_decoded[i + 1] if i + 1 < len(all_decoded) else None
            co_positions.append({
                'position': i,
                'folio': all_folios[i],
                'next_word': next_dec,
                'next_is_signal': next_dec in signal_words if next_dec else False,
                'next_is_noun': next_dec in NOUN_CANDIDATES if next_dec else False,
                'next_is_content': len(next_dec) >= 3 if next_dec else False,
            })

    n_co = len(co_positions)
    print(f"\n  Found {n_co} 'co' occurrences")

    # ------------------------------------------------------------------
    # Step 5: Compute noun rate, content rate, signal rate
    # ------------------------------------------------------------------

    if n_co > 0:
        noun_rate = sum(1 for cp in co_positions if cp['next_is_noun']) / n_co
        content_rate = sum(1 for cp in co_positions if cp['next_is_content']) / n_co
        signal_rate_val = sum(1 for cp in co_positions if cp['next_is_signal']) / n_co
    else:
        noun_rate = 0.0
        content_rate = 0.0
        signal_rate_val = 0.0

    print(f"  Noun rate (co + NOUN): {noun_rate:.4f}")
    print(f"  Content rate (co + 3+ char): {content_rate:.4f}")
    print(f"  Signal rate (co + SIGNAL): {signal_rate_val:.4f}")

    # ------------------------------------------------------------------
    # Step 6: Null test A — position shuffle (1000 iterations)
    # ------------------------------------------------------------------

    print("\n  Running null test A (position shuffle, 1000 iterations)...")
    rng = random.Random(42)
    null_a_rates: List[float] = []
    n_null = 1000

    for _ in range(n_null):
        n_noun_null = 0
        for _ in range(n_co):
            # Replace following word with random token from all_decoded
            rand_idx = rng.randint(0, len(all_decoded) - 1)
            rand_word = all_decoded[rand_idx]
            if rand_word in NOUN_CANDIDATES:
                n_noun_null += 1
        rate = n_noun_null / n_co if n_co > 0 else 0.0
        null_a_rates.append(rate)

    null_a_mean = sum(null_a_rates) / len(null_a_rates) if null_a_rates else 0.0
    null_a_var = sum((x - null_a_mean) ** 2 for x in null_a_rates) / len(null_a_rates) if null_a_rates else 0.0
    null_a_std = null_a_var ** 0.5

    print(f"  Null A: mean={null_a_mean:.4f}, std={null_a_std:.4f}")

    # ------------------------------------------------------------------
    # Step 7: Null test B — word shuffle (1000 iterations)
    # ------------------------------------------------------------------

    print("  Running null test B (word shuffle, 1000 iterations)...")
    null_b_rates: List[float] = []

    for _ in range(n_null):
        # Pick n_co random positions, check fraction followed by noun
        n_noun_null = 0
        for _ in range(n_co):
            rand_pos = rng.randint(0, len(all_decoded) - 2)  # -2 so next exists
            next_word = all_decoded[rand_pos + 1]
            if next_word in NOUN_CANDIDATES:
                n_noun_null += 1
        rate = n_noun_null / n_co if n_co > 0 else 0.0
        null_b_rates.append(rate)

    null_b_mean = sum(null_b_rates) / len(null_b_rates) if null_b_rates else 0.0
    null_b_var = sum((x - null_b_mean) ** 2 for x in null_b_rates) / len(null_b_rates) if null_b_rates else 0.0
    null_b_std = null_b_var ** 0.5

    print(f"  Null B: mean={null_b_mean:.4f}, std={null_b_std:.4f}")

    # ------------------------------------------------------------------
    # Step 8: Compute z-scores
    # ------------------------------------------------------------------

    z_a = (noun_rate - null_a_mean) / null_a_std if null_a_std > 0 else 0.0
    z_b = (noun_rate - null_b_mean) / null_b_std if null_b_std > 0 else 0.0
    selectivity = noun_rate / null_a_mean if null_a_mean > 0 else 0.0

    print(f"\n  Real noun rate: {noun_rate:.4f}")
    print(f"  z_a (vs position shuffle): {z_a:.2f}")
    print(f"  z_b (vs word shuffle): {z_b:.2f}")
    print(f"  Selectivity: {selectivity:.2f}x")

    # ------------------------------------------------------------------
    # Step 9: Dialect scoring
    # ------------------------------------------------------------------

    if z_a > 2:
        # Co confirmed as preposition — score dialects by usage of "co"
        dialect_scores = {
            'venetian': 0.9,
            'lombard': 0.8,
            'ligurian': 0.6,
            'emilian': 0.2,
            'tuscan': 0.1,
        }
        print("\n  'co' confirmed as preposition (z_a > 2)")
    else:
        # Not confirmed — uninformative
        dialect_scores = {d: 0.5 for d in
                          ['venetian', 'lombard', 'ligurian', 'emilian', 'tuscan']}
        print("\n  'co' NOT confirmed as preposition (z_a <= 2)")

    print(f"  Dialect scores: {dialect_scores}")

    # ------------------------------------------------------------------
    # Step 10: Check for known prepositional bigrams
    # ------------------------------------------------------------------

    CO_EXPECTED_BIGRAMS = [
        'co sero', 'co cola', 'co bene', 'co cora', 'co sene', 'co raso',
    ]
    found_bigrams: List[str] = []
    for cp in co_positions:
        bigram = f"co {cp['next_word']}"
        if bigram in CO_EXPECTED_BIGRAMS:
            found_bigrams.append(bigram)

    found_unique = sorted(set(found_bigrams))
    print(f"  Expected bigrams found: {len(found_unique)} / {len(CO_EXPECTED_BIGRAMS)}")
    for bg in found_unique:
        count = found_bigrams.count(bg)
        print(f"    {bg} (x{count})")

    # ------------------------------------------------------------------
    # Step 11: Gates
    # ------------------------------------------------------------------

    gates = {
        'G1_enough_co': n_co >= 50,
        'G2_noun_rate_above_null_a': z_a > 2.0,
        'G3_noun_rate_above_null_b': z_b > 1.0,
    }
    n_pass = sum(gates.values())

    if n_pass == 3:
        verdict = "CONFIRMED_PREPOSITION"
    elif n_pass >= 2:
        verdict = "PROBABLE_PREPOSITION"
    elif n_pass >= 1:
        verdict = "WEAK_EVIDENCE"
    else:
        verdict = "NO_EVIDENCE"

    print(f"\n  Gates:")
    for gname, gval in gates.items():
        status = "PASS" if gval else "FAIL"
        print(f"    {gname}: {status}")
    print(f"\n  Verdict: {verdict}")

    # ------------------------------------------------------------------
    # Build result and save
    # ------------------------------------------------------------------

    elapsed = round(time.time() - t0, 2)

    # Sample first 20 co positions for output
    co_details_sample = co_positions[:20]

    result = CoSyntaxResult(
        phase="54.5",
        experiment="co_syntactic_validation",
        n_co_occurrences=n_co,
        noun_rate=round(noun_rate, 6),
        content_rate=round(content_rate, 6),
        signal_rate=round(signal_rate_val, 6),
        null_a_mean=round(null_a_mean, 6),
        null_a_std=round(null_a_std, 6),
        z_a=round(z_a, 4),
        null_b_mean=round(null_b_mean, 6),
        null_b_std=round(null_b_std, 6),
        z_b=round(z_b, 4),
        selectivity=round(selectivity, 4),
        found_bigrams=found_unique,
        dialect_scores={k: round(v, 4) for k, v in dialect_scores.items()},
        co_details_sample=co_details_sample,
        gates=gates,
        verdict=verdict,
        runtime_seconds=elapsed,
    )

    out_path = os.path.join(rd, "phase54_co_syntax.json")
    with open(out_path, "w") as f:
        json.dump(_convert(result), f, indent=2)
    print(f"\n  Saved: {out_path}")
    print(f"  Runtime: {elapsed:.1f}s")
