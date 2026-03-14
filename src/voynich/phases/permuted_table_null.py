"""
Phase 50 Track A – Permuted-Table Null Test
=============================================
Test whether the specific 25-triple assignment table matters, or whether
ANY table + ED1 + char-LM produces ~50% dict-hit.

Dependency chain:
    combined_refine.json        (Phase 15 best table)
    bootstrap_loop.json         (Phase 30 confirmed triples)
    signal_bigrams.json         (Phase 29 parallel arrays)
        -> permuted_table_null.json
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import build_eva_to_triple_lookup, load_corpus, tokenize_eva_chars
from voynich.core.reference import load_reference_corpus
from voynich.core.stats import build_ngram_lm, cross_entropy_lm


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
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return _convert(obj.tolist())
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


def _save_json(rd: str, filename: str, data: Any) -> str:
    path = os.path.join(rd, filename)
    with open(path, 'w') as f:
        json.dump(_convert(data), f, indent=2)
    return path


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class PermutedTableNullResult:
    real_table_hit_rate: float
    partial_permutation_mean: float
    partial_permutation_std: float
    partial_permutation_z: float
    partial_selectivity: float
    full_permutation_mean: float
    full_permutation_std: float
    full_permutation_z: float
    full_selectivity: float
    n_partial_trials: int
    n_full_trials: int
    n_tokens: int
    n_confirmed_triples: int
    n_free_triples: int
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Reference data builders
# ---------------------------------------------------------------------------

def _build_10k_word_set() -> Set[str]:
    """Build a 10K-word reference dictionary from Latin + Italian."""
    ref = load_reference_corpus(languages=['latin', 'italian'], verbose=False)
    all_tokens: List[str] = []
    for lang in ('latin', 'italian'):
        all_tokens.extend(ref.get_combined_tokens(lang))
    freq = Counter(w.lower() for w in all_tokens if len(w) >= 2)
    return {w for w, _ in freq.most_common(10000)}


def _build_char_lm() -> Dict:
    """Build a character-level 5-gram LM from Latin + Italian reference text."""
    ref = load_reference_corpus(languages=['latin', 'italian'], verbose=False)
    all_tokens: List[str] = []
    for lang in ('latin', 'italian'):
        all_tokens.extend(ref.get_combined_tokens(lang))
    tokens = [w.lower() for w in all_tokens if len(w) >= 2 and w.isalpha()]
    return build_ngram_lm(tokens, order=5, smoothing=1.0)


# ---------------------------------------------------------------------------
# ED1 generation
# ---------------------------------------------------------------------------

_ALPHABET = 'abcdefghijklmnopqrstuvwxyz'


def _generate_ed1(word: str) -> Set[str]:
    """Generate all edit-distance-1 variants (substitution, deletion, insertion)."""
    variants: Set[str] = set()
    n = len(word)

    # Substitutions
    for i in range(n):
        for c in _ALPHABET:
            if c != word[i]:
                variants.add(word[:i] + c + word[i + 1:])

    # Deletions
    for i in range(n):
        variants.add(word[:i] + word[i + 1:])

    # Insertions
    for i in range(n + 1):
        for c in _ALPHABET:
            variants.add(word[:i] + c + word[i:])

    return variants


# ---------------------------------------------------------------------------
# Core decode + score pipeline
# ---------------------------------------------------------------------------

def _decode_and_score_ed1(
    token_evas: List[str],
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    char_lm: Dict,
    word_set: Set[str],
) -> List[Tuple[str, bool]]:
    """Decode EVA tokens using assignment, apply ED1 + char-LM rescoring.

    Returns list of (best_word, is_dict_hit) for each token.
    """
    results: List[Tuple[str, bool]] = []

    for token in token_evas:
        # Decode token via assignment table
        chars = tokenize_eva_chars(token)
        syllables: List[str] = []
        for ch in chars:
            triple_key = eva_to_triple.get(ch)
            if triple_key is None:
                continue
            syl = assignment.get(triple_key)
            if syl is not None:
                syllables.append(syl)

        decoded = ''.join(syllables)
        if not decoded:
            results.append(('', False))
            continue

        # Generate ED1 variants
        ed1_variants = _generate_ed1(decoded)
        ed1_variants.add(decoded)  # include original

        # Split into dict-hit and non-dict-hit variants
        dict_hits = [v for v in ed1_variants if v in word_set]

        if dict_hits:
            # Score dict-hit variants with char LM, pick lowest CE
            best_word = min(
                dict_hits,
                key=lambda w: cross_entropy_lm('_' + w + '_', char_lm, per_char=True),
            )
            results.append((best_word, True))
        else:
            # Score all variants, pick lowest CE
            best_word = min(
                ed1_variants,
                key=lambda w: cross_entropy_lm('_' + w + '_', char_lm, per_char=True),
            )
            results.append((best_word, False))

    return results


# ---------------------------------------------------------------------------
# Subsampling
# ---------------------------------------------------------------------------

def _subsample_tokens(
    token_evas: List[str],
    token_folios: List[str],
    n: int = 5000,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[int]]:
    """Stratified subsample: proportional samples from each folio."""
    rng = random.Random(seed)

    # Group indices by folio
    folio_indices: Dict[str, List[int]] = {}
    for i, folio in enumerate(token_folios):
        folio_indices.setdefault(folio, []).append(i)

    total = len(token_evas)
    selected_indices: List[int] = []

    for folio, indices in sorted(folio_indices.items()):
        # Proportional allocation
        k = max(1, round(len(indices) * n / total))
        k = min(k, len(indices))
        selected_indices.extend(rng.sample(indices, k))

    # Trim or pad to exactly n
    if len(selected_indices) > n:
        rng.shuffle(selected_indices)
        selected_indices = selected_indices[:n]
    elif len(selected_indices) < n:
        # Fill remaining from full pool
        remaining = [i for i in range(total) if i not in set(selected_indices)]
        rng.shuffle(remaining)
        selected_indices.extend(remaining[: n - len(selected_indices)])

    selected_indices.sort()

    sub_evas = [token_evas[i] for i in selected_indices]
    sub_folios = [token_folios[i] for i in selected_indices]
    return sub_evas, sub_folios, selected_indices


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_permuted_table_null() -> Dict[str, Any]:
    """Phase 50 Track A: Permuted-Table Null Test."""
    t0 = time.time()
    rd = _results_dir()

    print("=" * 70)
    print("Phase 50 Track A: Permuted-Table Null Test")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("\n--- Step 1: Loading data ---")

    refine_path = os.path.join(rd, 'combined_refine.json')
    refine_data = _safe_load(refine_path)
    assignment: Dict[str, str] = refine_data.get('best_assignment', {})
    print(f"  Assignment table: {len(assignment)} triples")

    boot_path = os.path.join(rd, 'bootstrap_loop.json')
    boot_data = _safe_load(boot_path)
    confirmed_triples: List[str] = boot_data.get('confirmed_triples', [])
    print(f"  Confirmed triples: {len(confirmed_triples)}")

    sig_path = os.path.join(rd, 'signal_bigrams.json')
    sig_data = _safe_load(sig_path)
    token_evas_all: List[str] = sig_data.get('token_evas', [])
    token_folios_all: List[str] = sig_data.get('token_folios', [])
    print(f"  Total tokens from signal_bigrams: {len(token_evas_all)}")

    # Fallback: if signal_bigrams doesn't have tokens, build from corpus
    if not token_evas_all:
        print("  signal_bigrams.json missing token_evas; loading from corpus...")
        corpus = load_corpus(verbose=False)
        for folio, page in corpus.pages.items():
            for token in page.all_tokens:
                token_evas_all.append(token)
                token_folios_all.append(folio)
        print(f"  Loaded {len(token_evas_all)} tokens from corpus")

    # ------------------------------------------------------------------
    # 2. Build infrastructure
    # ------------------------------------------------------------------
    print("\n--- Step 2: Building infrastructure ---")

    eva_to_triple = build_eva_to_triple_lookup()
    print(f"  EVA-to-triple lookup: {len(eva_to_triple)} entries")

    char_lm = _build_char_lm()
    print(f"  Char LM built (order={char_lm.get('order', '?')})")

    word_set = _build_10k_word_set()
    print(f"  10K word set: {len(word_set)} words")

    # ------------------------------------------------------------------
    # 3. Subsample
    # ------------------------------------------------------------------
    print("\n--- Step 3: Subsampling to 5000 tokens ---")

    sub_evas, sub_folios, sub_indices = _subsample_tokens(
        token_evas_all, token_folios_all, n=5000, seed=42,
    )
    n_tokens = len(sub_evas)
    print(f"  Subsampled: {n_tokens} tokens")

    # ------------------------------------------------------------------
    # 4. Real table baseline
    # ------------------------------------------------------------------
    print("\n--- Step 4: Real table baseline ---")

    real_results = _decode_and_score_ed1(
        sub_evas, assignment, eva_to_triple, char_lm, word_set,
    )
    real_hits = sum(1 for _, hit in real_results if hit)
    real_hit_rate = real_hits / n_tokens if n_tokens > 0 else 0.0
    print(f"  Real table hit rate: {real_hit_rate:.4f} ({real_hits}/{n_tokens})")

    # ------------------------------------------------------------------
    # 5. Partial permutation (shuffle free triples only)
    # ------------------------------------------------------------------
    n_trials = 50
    print(f"\n--- Step 5: Partial permutation ({n_trials} trials) ---")

    confirmed_set = set(confirmed_triples)
    free_triples = [k for k in assignment if k not in confirmed_set]
    n_confirmed = len(confirmed_set & set(assignment.keys()))
    n_free = len(free_triples)
    print(f"  Confirmed: {n_confirmed}, Free: {n_free}")

    partial_rates: List[float] = []
    for trial in range(n_trials):
        rng = random.Random(5000 + trial)
        perm_assignment = dict(assignment)

        # Shuffle syllable values among free triples
        free_values = [perm_assignment[k] for k in free_triples]
        rng.shuffle(free_values)
        for k, v in zip(free_triples, free_values):
            perm_assignment[k] = v

        trial_results = _decode_and_score_ed1(
            sub_evas, perm_assignment, eva_to_triple, char_lm, word_set,
        )
        trial_hits = sum(1 for _, hit in trial_results if hit)
        rate = trial_hits / n_tokens if n_tokens > 0 else 0.0
        partial_rates.append(rate)

        if (trial + 1) % 10 == 0:
            print(f"  Trial {trial + 1}/{n_trials}: hit_rate={rate:.2%}")

    partial_mean = float(np.mean(partial_rates))
    partial_std = float(np.std(partial_rates, ddof=1)) if len(partial_rates) > 1 else 0.0
    partial_z = (real_hit_rate - partial_mean) / partial_std if partial_std > 0 else 0.0
    partial_selectivity = real_hit_rate / partial_mean if partial_mean > 0 else float('inf')

    print(f"  Partial permutation: mean={partial_mean:.4f}, std={partial_std:.4f}")
    print(f"  z-score={partial_z:.2f}, selectivity={partial_selectivity:.2f}")

    # ------------------------------------------------------------------
    # 6. Full permutation (shuffle all 25 triples)
    # ------------------------------------------------------------------
    print(f"\n--- Step 6: Full permutation ({n_trials} trials) ---")

    all_triple_keys = list(assignment.keys())
    full_rates: List[float] = []
    for trial in range(n_trials):
        rng = random.Random(5000 + trial)
        perm_assignment = {}

        all_values = [assignment[k] for k in all_triple_keys]
        rng.shuffle(all_values)
        for k, v in zip(all_triple_keys, all_values):
            perm_assignment[k] = v

        trial_results = _decode_and_score_ed1(
            sub_evas, perm_assignment, eva_to_triple, char_lm, word_set,
        )
        trial_hits = sum(1 for _, hit in trial_results if hit)
        rate = trial_hits / n_tokens if n_tokens > 0 else 0.0
        full_rates.append(rate)

        if (trial + 1) % 10 == 0:
            print(f"  Trial {trial + 1}/{n_trials}: hit_rate={rate:.2%}")

    full_mean = float(np.mean(full_rates))
    full_std = float(np.std(full_rates, ddof=1)) if len(full_rates) > 1 else 0.0
    full_z = (real_hit_rate - full_mean) / full_std if full_std > 0 else 0.0
    full_selectivity = real_hit_rate / full_mean if full_mean > 0 else float('inf')

    print(f"  Full permutation: mean={full_mean:.4f}, std={full_std:.4f}")
    print(f"  z-score={full_z:.2f}, selectivity={full_selectivity:.2f}")

    # ------------------------------------------------------------------
    # 7. Verdict
    # ------------------------------------------------------------------
    print("\n--- Step 7: Verdict ---")

    if full_selectivity < 1.1:
        verdict = "ARTIFACT"
    elif full_selectivity < 1.3:
        verdict = "MARGINAL"
    elif partial_selectivity < 1.2:
        verdict = "CONFIRMED_CORE_ONLY"
    else:
        verdict = "GENUINE"

    runtime = time.time() - t0

    print(f"  Verdict: {verdict}")
    print(f"  Runtime: {runtime:.1f}s")

    # ------------------------------------------------------------------
    # 8. Save results
    # ------------------------------------------------------------------
    result = PermutedTableNullResult(
        real_table_hit_rate=real_hit_rate,
        partial_permutation_mean=partial_mean,
        partial_permutation_std=partial_std,
        partial_permutation_z=partial_z,
        partial_selectivity=partial_selectivity,
        full_permutation_mean=full_mean,
        full_permutation_std=full_std,
        full_permutation_z=full_z,
        full_selectivity=full_selectivity,
        n_partial_trials=n_trials,
        n_full_trials=n_trials,
        n_tokens=n_tokens,
        n_confirmed_triples=n_confirmed,
        n_free_triples=n_free,
        verdict=verdict,
        runtime_seconds=runtime,
    )

    out_path = _save_json(rd, 'permuted_table_null.json', asdict(result))
    print(f"\n  Saved: {out_path}")

    print("\n" + "=" * 70)
    print("Phase 50 Track A complete")
    print(f"  Real hit rate:       {real_hit_rate:.4f}")
    print(f"  Partial perm mean:   {partial_mean:.4f} (selectivity {partial_selectivity:.2f}x)")
    print(f"  Full perm mean:      {full_mean:.4f} (selectivity {full_selectivity:.2f}x)")
    print(f"  Verdict:             {verdict}")
    print("=" * 70)

    return asdict(result)
