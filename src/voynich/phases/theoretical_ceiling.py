"""
Phase 23.1 – Theoretical Ceiling Analysis (ceiling)
=====================================================
Determines the theoretical maximum dict-hit rate achievable under ANY
syllable assignment for the Voynich token distribution.  Computes the
"oracle ceiling" — fraction of tokens where at least one combination of
CV syllables produces a word in the expanded dictionary — and compares
Phase 16's 51.6% against it to measure efficiency.

Dependency chain:
    combined_refine.json (Phase 15 best_assignment)
    modifier_integrate.json (Phase 16 modifier chars)
        → theoretical_ceiling.json (this step)
"""

import json
import os
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    load_corpus,
    token_to_triples,
    tokenize_eva_chars,
)
from voynich.core.reference import (
    EVA_VISUAL_COMPONENTS,
    build_cv_syllable_table,
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


def _load_json(path: str) -> Optional[Dict]:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Trie for prefix pruning
# ---------------------------------------------------------------------------

class _Trie:
    """Character-level trie for efficient prefix checking."""

    __slots__ = ('children', 'is_end')

    def __init__(self):
        self.children: Dict[str, '_Trie'] = {}
        self.is_end: bool = False

    def insert(self, word: str) -> None:
        node = self
        for ch in word:
            if ch not in node.children:
                node.children[ch] = _Trie()
            node = node.children[ch]
        node.is_end = True

    def has_prefix(self, prefix: str) -> bool:
        node = self
        for ch in prefix:
            if ch not in node.children:
                return False
            node = node.children[ch]
        return True


def _build_trie(word_set: set) -> _Trie:
    """Build a trie from a word set for efficient prefix lookups."""
    trie = _Trie()
    for word in word_set:
        trie.insert(word.lower())
    return trie


# ---------------------------------------------------------------------------
# Oracle ceiling computation
# ---------------------------------------------------------------------------

def _can_hit(
    triple_keys: List[str],
    syllable_table: List[str],
    trie: _Trie,
    ref_word_set: set,
    idx: int = 0,
    prefix: str = '',
) -> bool:
    """Recursively check if ANY syllable assignment for the given triples
    produces a word in the dictionary.  Uses trie-based prefix pruning."""
    if idx == len(triple_keys):
        return prefix in ref_word_set

    for syl in syllable_table:
        new_prefix = prefix + syl
        if not trie.has_prefix(new_prefix):
            continue
        if _can_hit(triple_keys, syllable_table, trie, ref_word_set,
                    idx + 1, new_prefix):
            return True
    return False


def _can_hit_sampled(
    triple_keys: List[str],
    syllable_table: List[str],
    ref_word_set: set,
    rng: random.Random,
    n_samples: int = 10000,
) -> bool:
    """For long tokens, sample random assignments to estimate hittability."""
    for _ in range(n_samples):
        word = ''.join(rng.choice(syllable_table) for _ in triple_keys)
        if word.lower() in ref_word_set:
            return True
    return False


def _strip_modifiers_and_get_triples(
    token: str,
    eva_to_triple: Dict[str, str],
    modifier_chars: Set[str],
) -> List[str]:
    """Tokenize EVA chars, strip modifiers, return remaining triple keys."""
    chars = tokenize_eva_chars(token)
    syllabic = [c for c in chars if c not in modifier_chars]
    if not syllabic:
        return []
    triples = []
    for c in syllabic:
        tk = eva_to_triple.get(c)
        if tk:
            triples.append(tk)
    return triples


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class CeilingResult:
    timestamp: str
    phase16_dict_hit: float
    phase16_selectivity: float
    oracle_ceiling: float
    oracle_n_tokens: int
    oracle_n_hittable: int
    efficiency: float
    mean_triples_per_token: float
    triple_count_distribution: Dict[int, int]
    ceiling_by_length: Dict[int, Dict]
    n_cv_syllables: int
    n_syllables_used: int
    dict_size_base: int
    dict_size_expanded: int
    random_sample_ceiling: float
    gate_passed: bool
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_theoretical_ceiling() -> Dict[str, Any]:
    """Step 23.1: Theoretical ceiling analysis."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE 23.1: Theoretical Ceiling Analysis")
    print("=" * 70)

    rdir = _results_dir()

    # Load Phase 16 assignment
    combined = _load_json(str(rdir / "combined_refine.json")) or {}
    best_assignment = combined.get("best_assignment", {})
    phase16_dict_hit = combined.get("ablation_table", [{}])[1].get("dict_hit", 0.3545)
    # Phase 16 actual is from modifier_integrate
    mod_data = _load_json(str(rdir / "modifier_integrate.json")) or {}
    phase16_dict_hit = mod_data.get("r3_dict_hit", 0.5165)
    modifier_chars = set(mod_data.get("modifier_chars", []))

    # Load corpus
    corpus = load_corpus(verbose=False)
    tokens = corpus.get_tokens()
    print(f"  Corpus: {len(tokens)} tokens")

    # Build triple lookup
    eva_to_triple = build_eva_to_triple_lookup()

    # Build dictionary
    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    base_words = set(
        w.lower() for w in ref_corpus.get_combined_tokens('latin')
        if len(w) >= 2
    )
    expanded_words, _ = build_expanded_word_set(base_words)
    ref_word_set = base_words | expanded_words
    print(f"  Dictionary: {len(base_words)} base + {len(expanded_words)} expanded"
          f" = {len(ref_word_set)} total")

    # Build syllable table
    syllable_table = build_cv_syllable_table('latin')
    n_cv_syllables = len(syllable_table)
    n_syllables_used = len(set(best_assignment.values()))
    print(f"  CV syllables: {n_cv_syllables} total, {n_syllables_used} used by Phase 16")

    # Build trie for prefix pruning
    print("  Building trie from dictionary...")
    trie = _build_trie(ref_word_set)

    # Oracle ceiling computation
    max_tokens = 2000
    sample_tokens = tokens[:max_tokens]
    rng = random.Random(42)

    triple_counts = Counter()
    n_hittable = 0
    hittable_by_length: Dict[int, List[bool]] = {}

    print(f"  Computing oracle ceiling for {len(sample_tokens)} tokens...")
    for i, token in enumerate(sample_tokens):
        if (i + 1) % 500 == 0:
            print(f"    ... {i + 1}/{len(sample_tokens)}")

        triples = _strip_modifiers_and_get_triples(
            token, eva_to_triple, modifier_chars
        )
        n_t = len(triples)
        triple_counts[n_t] += 1

        if n_t not in hittable_by_length:
            hittable_by_length[n_t] = []

        if n_t == 0:
            hittable_by_length[n_t].append(False)
            continue

        # For short tokens, use exact enumeration with trie pruning
        if n_t <= 4:
            hit = _can_hit(triples, syllable_table, trie, ref_word_set)
        else:
            # For long tokens, sample
            hit = _can_hit_sampled(triples, syllable_table, ref_word_set,
                                   rng, n_samples=10000)

        if hit:
            n_hittable += 1
        hittable_by_length[n_t].append(hit)

    oracle_ceiling = n_hittable / max(len(sample_tokens), 1)
    efficiency = phase16_dict_hit / oracle_ceiling if oracle_ceiling > 0 else 0.0

    # Ceiling by length
    ceiling_by_length = {}
    for n_t in sorted(hittable_by_length.keys()):
        hits_list = hittable_by_length[n_t]
        n_tokens_at_len = len(hits_list)
        n_hit = sum(hits_list)
        ceiling_by_length[n_t] = {
            'n_tokens': n_tokens_at_len,
            'n_hittable': n_hit,
            'ceiling': n_hit / n_tokens_at_len if n_tokens_at_len > 0 else 0.0,
        }

    # Mean triples per token
    total_triples = sum(k * v for k, v in triple_counts.items())
    mean_triples = total_triples / max(len(sample_tokens), 1)

    # Random baseline: average dict-hit over 100 random syllable tables
    print("  Computing random baseline (100 random tables)...")
    random_hits_list = []
    for trial in range(100):
        rand_assignment = {}
        for tk in best_assignment:
            rand_assignment[tk] = rng.choice(syllable_table)
        # Score
        hits = 0
        for token in sample_tokens[:500]:
            triples = _strip_modifiers_and_get_triples(
                token, eva_to_triple, modifier_chars
            )
            if not triples:
                continue
            decoded = ''.join(rand_assignment.get(tk, '?') for tk in triples)
            if decoded.lower() in ref_word_set:
                hits += 1
        random_hits_list.append(hits / min(len(sample_tokens), 500))

    random_ceiling = sum(random_hits_list) / len(random_hits_list)

    # Phase 16 selectivity from modifier_integrate
    phase16_sel = mod_data.get("r3_selectivity", 3.40)

    # Gate and verdict
    if efficiency > 0.8:
        verdict = "NEAR OPTIMAL"
    elif efficiency > 0.5:
        verdict = "SIGNIFICANT GAP"
    else:
        verdict = "LARGE UNEXPLAINED GAP"
    gate_passed = True  # informational step, always passes

    elapsed = time.time() - t0

    result = CeilingResult(
        timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
        phase16_dict_hit=round(phase16_dict_hit, 4),
        phase16_selectivity=round(phase16_sel, 2),
        oracle_ceiling=round(oracle_ceiling, 4),
        oracle_n_tokens=len(sample_tokens),
        oracle_n_hittable=n_hittable,
        efficiency=round(efficiency, 4),
        mean_triples_per_token=round(mean_triples, 2),
        triple_count_distribution=dict(sorted(triple_counts.items())),
        ceiling_by_length=ceiling_by_length,
        n_cv_syllables=n_cv_syllables,
        n_syllables_used=n_syllables_used,
        dict_size_base=len(base_words),
        dict_size_expanded=len(ref_word_set),
        random_sample_ceiling=round(random_ceiling, 4),
        gate_passed=gate_passed,
        verdict=verdict,
        runtime_seconds=round(elapsed, 1),
    )

    out_path = rdir / "theoretical_ceiling.json"
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2, ensure_ascii=False)

    print(f"\n  Oracle ceiling: {oracle_ceiling:.1%}")
    print(f"  Phase 16 actual: {phase16_dict_hit:.1%}")
    print(f"  Efficiency: {efficiency:.1%}")
    print(f"  Random baseline: {random_ceiling:.1%}")
    print(f"  Mean triples/token: {mean_triples:.2f}")
    print(f"  Verdict: {verdict}")
    print(f"  → {out_path} ({elapsed:.1f}s)")

    return _convert(asdict(result))
