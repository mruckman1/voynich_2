"""
Phase 47 Track B – Word-Level Surjective Disambiguation
=========================================================
For each token, enumerate alternative decoded words (from MaxSAT candidates),
then select the interpretation maximising word bigram probability with
neighbouring tokens via per-folio Viterbi.

Dependency chain:
    combined_refine.json        (Phase 15 best table)
    canonical_table.json        (Phase 45 tiers)
    maxsat_landscape.json       (Phase 44 per-triple consensus)
    triple_tiers.json           (Phase 45 tier assignments)
    signal_bigrams.json         (Phase 29 parallel arrays)
    modifier_integrate.json     (Phase 16 modifiers)
        -> disamb_lattice.json  (Step 47B.1)
        -> disamb_bigram.json   (Step 47B.2)
        -> disamb_viterbi.json  (Step 47B.3)
        -> disamb_eval.json     (Step 47B.4)
        -> disamb_compare.json  (Step 47B.5)
"""

from __future__ import annotations

import itertools
import json
import math
import os
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

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
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if v != v else v
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
# Step 47B.1 — Per-token decode lattice
# ---------------------------------------------------------------------------

@dataclass
class LatticeResult:
    n_tokens: int
    n_with_alternatives: int
    alternative_rate: float
    mean_lattice_size: float
    max_lattice_size: int
    total_lattice_paths: int
    lattice_size_distribution: Dict[str, int]
    runtime_seconds: float


def _build_triple_candidates(rd: str) -> Dict[str, List[Tuple[str, float]]]:
    """For each triple, return list of (syllable, probability) candidates.

    CONFIRMED triples: 1 candidate (the current assignment).
    LANDSCAPE_CONFIRMED: current + MaxSAT alternatives (top 3).
    GENUINELY_AMBIGUOUS: all MaxSAT candidates with probability >= 0.08.
    """
    tiers_data = _safe_load(os.path.join(rd, 'triple_tiers.json'))
    maxsat = _safe_load(os.path.join(rd, 'maxsat_landscape.json'))
    p15 = _safe_load(os.path.join(rd, 'combined_refine.json')).get(
        'best_assignment', {},
    )

    ptc = maxsat.get('per_triple_consensus', {})

    # Build tier lookup
    tier_lookup: Dict[str, str] = {}
    for tier_name, entries in tiers_data.get('tiers', {}).items():
        for entry in entries:
            tier_lookup[entry['triple_key']] = tier_name

    candidates: Dict[str, List[Tuple[str, float]]] = {}

    for triple_key, current_syl in p15.items():
        tier = tier_lookup.get(triple_key, 'UNKNOWN')

        if tier == 'CONFIRMED':
            candidates[triple_key] = [(current_syl, 1.0)]
        elif tier == 'LANDSCAPE_CONFIRMED':
            cands = [(current_syl, 1.0)]
            ms_consensus = ptc.get(triple_key, {})
            for syl, prob in sorted(ms_consensus.items(), key=lambda x: -x[1]):
                if syl != current_syl and prob >= 0.05:
                    cands.append((syl, prob))
                if len(cands) >= 4:
                    break
            candidates[triple_key] = cands
        elif tier == 'GENUINELY_AMBIGUOUS':
            cands = [(current_syl, 1.0)]
            ms_consensus = ptc.get(triple_key, {})
            for syl, prob in sorted(ms_consensus.items(), key=lambda x: -x[1]):
                if syl != current_syl and prob >= 0.08:
                    cands.append((syl, prob))
            candidates[triple_key] = cands
        else:
            candidates[triple_key] = [(current_syl, 1.0)]

    return candidates


def _enumerate_lattice(
    token: str,
    triple_candidates: Dict[str, List[Tuple[str, float]]],
    eva_to_triple: Dict[str, str],
    modifier_chars: set,
    max_paths: int = 1000,
) -> List[Tuple[str, float]]:
    """Return list of (decoded_word, prior_probability) for a token.

    The decoded word is the concatenation of syllables for non-modifier
    EVA characters.
    """
    chars = tokenize_eva_chars(token)
    triple_slots = []
    for ch in chars:
        if ch in modifier_chars:
            continue
        t = eva_to_triple.get(ch)
        if t:
            triple_slots.append(t)

    if not triple_slots:
        return [('', 1.0)]

    # Get candidate lists for each slot
    slot_candidates = []
    for t in triple_slots:
        cands = triple_candidates.get(t, [('?', 1.0)])
        slot_candidates.append(cands)

    # Estimate total paths
    total = 1
    for sc in slot_candidates:
        total *= len(sc)

    if total <= max_paths:
        # Full enumeration
        results = []
        for combo in itertools.product(*slot_candidates):
            word = ''.join(syl for syl, _ in combo)
            prior = 1.0
            for _, p in combo:
                prior *= p
            results.append((word, prior))
    else:
        # Pruned: take top-max_paths by prior
        # Use greedy expansion from highest-prior candidates
        results = []
        # Sort each slot's candidates by probability (descending)
        sorted_slots = [sorted(sc, key=lambda x: -x[1]) for sc in slot_candidates]

        # Enumerate using itertools.product on truncated lists
        # Truncate each slot to top-k where product of tops <= max_paths
        k_per_slot = max(1, int(max_paths ** (1.0 / len(sorted_slots))))
        truncated = [sc[:k_per_slot] for sc in sorted_slots]
        for combo in itertools.product(*truncated):
            word = ''.join(syl for syl, _ in combo)
            prior = 1.0
            for _, p in combo:
                prior *= p
            results.append((word, prior))
        # Sort by prior descending and take top
        results.sort(key=lambda x: -x[1])
        results = results[:max_paths]

    # Normalize priors
    total_prior = sum(p for _, p in results)
    if total_prior > 0:
        results = [(w, p / total_prior) for w, p in results]

    return results


def run_disamb_lattice() -> None:
    """Step 47B.1: build per-token decode lattice."""
    t0 = time.time()
    print("=" * 70)
    print("STEP 47B.1: Per-Token Decode Lattice")
    print("=" * 70)

    rd = _results_dir()

    # Load data
    sb = _safe_load(os.path.join(rd, 'signal_bigrams.json'))
    token_evas = sb.get('token_evas', [])
    token_decoded = sb.get('token_decoded', [])
    n_tokens = len(token_evas)

    if n_tokens == 0:
        print("  [SKIP] No token data")
        return

    mod_data = _safe_load(os.path.join(rd, 'modifier_integrate.json'))
    modifier_chars = set(mod_data.get('modifier_chars', []))
    eva_to_triple = build_eva_to_triple_lookup()

    print("\n  Building triple candidates from tiers...")
    triple_candidates = _build_triple_candidates(rd)
    n_multi = sum(1 for cands in triple_candidates.values() if len(cands) > 1)
    print(f"  {n_multi} triples with alternatives (out of {len(triple_candidates)})")

    # Build lattice for each token
    print("\n  Enumerating lattices...")
    lattice_sizes = []
    n_with_alt = 0
    total_paths = 0
    size_dist: Counter = Counter()

    # Store compact lattice (only tokens with alternatives)
    lattice_data: Dict[int, List[Tuple[str, float]]] = {}

    for i, token in enumerate(token_evas):
        lattice = _enumerate_lattice(
            token, triple_candidates, eva_to_triple, modifier_chars,
        )
        size = len(lattice)
        lattice_sizes.append(size)
        total_paths += size
        if size > 1:
            n_with_alt += 1
            lattice_data[i] = lattice
        size_dist[size] += 1

    mean_size = sum(lattice_sizes) / n_tokens if n_tokens else 0.0
    max_size = max(lattice_sizes) if lattice_sizes else 0

    print(f"\n  Results:")
    print(f"    Tokens: {n_tokens}")
    print(f"    With alternatives: {n_with_alt} ({n_with_alt/n_tokens:.1%})")
    print(f"    Mean lattice size: {mean_size:.2f}")
    print(f"    Max lattice size: {max_size}")
    print(f"    Total paths: {total_paths}")

    # Save lattice data (store only statistics + compact lattice for tokens with alts)
    result = LatticeResult(
        n_tokens=n_tokens,
        n_with_alternatives=n_with_alt,
        alternative_rate=round(n_with_alt / n_tokens, 4) if n_tokens else 0.0,
        mean_lattice_size=round(mean_size, 2),
        max_lattice_size=max_size,
        total_lattice_paths=total_paths,
        lattice_size_distribution={str(k): v for k, v in sorted(size_dist.items())},
        runtime_seconds=round(time.time() - t0, 2),
    )

    # Also save compact lattice for Viterbi
    out = _convert(asdict(result))
    out['lattice'] = {
        str(idx): [(w, round(p, 6)) for w, p in entries]
        for idx, entries in lattice_data.items()
    }

    out_path = os.path.join(rd, 'disamb_lattice.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved -> {out_path}")


# ---------------------------------------------------------------------------
# Step 47B.2 — Decoded word bigram model
# ---------------------------------------------------------------------------

@dataclass
class BigramModelResult:
    variant: str
    n_tokens_used: int
    vocabulary_size: int
    n_bigrams: int
    coverage: float
    runtime_seconds: float


class WordBigramModel:
    """Simple add-1 smoothed word bigram model."""

    def __init__(self, name: str):
        self.name = name
        self.bigram_counts: Counter = Counter()
        self.unigram_counts: Counter = Counter()
        self.total = 0
        self.vocab_size = 0

    def train(self, word_sequences: List[List[str]]) -> None:
        for seq in word_sequences:
            for w in seq:
                self.unigram_counts[w] += 1
                self.total += 1
            for i in range(len(seq) - 1):
                self.bigram_counts[(seq[i], seq[i + 1])] += 1
        self.vocab_size = len(self.unigram_counts)

    def log_prob_bigram(self, w2: str, w1: str) -> float:
        """Log P(w2 | w1) with add-1 smoothing."""
        count_w1 = self.unigram_counts.get(w1, 0)
        count_bi = self.bigram_counts.get((w1, w2), 0)
        v = max(self.vocab_size, 1)
        p = (count_bi + 1) / (count_w1 + v)
        return math.log(p + 1e-20)

    def log_prob_unigram(self, w: str) -> float:
        """Log P(w) with add-1 smoothing."""
        count = self.unigram_counts.get(w, 0)
        v = max(self.vocab_size, 1)
        p = (count + 1) / (self.total + v)
        return math.log(p + 1e-20)

    def stats(self) -> Dict:
        return {
            'name': self.name,
            'vocabulary_size': self.vocab_size,
            'n_bigrams': len(self.bigram_counts),
            'total_tokens': self.total,
        }


def run_disamb_bigram() -> None:
    """Step 47B.2: build decoded word bigram models (FULL, SIGNAL, GREEN)."""
    t0 = time.time()
    print("=" * 70)
    print("STEP 47B.2: Decoded Word Bigram Models")
    print("=" * 70)

    rd = _results_dir()

    sb = _safe_load(os.path.join(rd, 'signal_bigrams.json'))
    token_folios = sb.get('token_folios', [])
    token_decoded = sb.get('token_decoded', [])
    token_classifications = sb.get('token_classifications', [])
    n_tokens = len(token_decoded)

    if n_tokens == 0:
        print("  [SKIP] No data")
        return

    # Classify tokens for GREEN variant
    # (simplified: use signal + dict hit as proxy for GREEN)
    sig_data = _safe_load(os.path.join(rd, 'signal_10k.json'))
    signal_words_set = {
        w['word'] for w in sig_data.get('word_signals', [])
        if w.get('is_genuine_signal')
    }

    # Group by folio
    folio_sequences: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for i in range(n_tokens):
        folio_sequences[token_folios[i]].append(
            (token_decoded[i], token_classifications[i]),
        )

    # Build three variants
    models = {}
    for variant in ['FULL', 'SIGNAL', 'GREEN']:
        model = WordBigramModel(variant)
        sequences = []
        for folio, tokens in folio_sequences.items():
            if variant == 'FULL':
                seq = [w for w, _ in tokens]
            elif variant == 'SIGNAL':
                seq = [w for w, c in tokens if c == 'SIGNAL']
            else:  # GREEN
                seq = [w for w, c in tokens if c == 'SIGNAL' and w in signal_words_set]
            if len(seq) >= 2:
                sequences.append(seq)
        model.train(sequences)
        models[variant] = model

        n_used = model.total
        print(f"\n  {variant}: {n_used} tokens, {model.vocab_size} vocab, "
              f"{len(model.bigram_counts)} bigrams")

    # Save model statistics (not the full model — too large)
    model_stats = []
    for variant, model in models.items():
        model_stats.append({
            'variant': variant,
            'vocabulary_size': model.vocab_size,
            'n_bigrams': len(model.bigram_counts),
            'total_tokens': model.total,
            'top_bigrams': [
                {'bigram': f'{w1} {w2}', 'count': c}
                for (w1, w2), c in model.bigram_counts.most_common(20)
            ],
        })

    out = {
        'models': model_stats,
        'runtime_seconds': round(time.time() - t0, 2),
    }

    out_path = os.path.join(rd, 'disamb_bigram.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(out), f, indent=2)
    print(f"\n  Saved -> {out_path}")


# ---------------------------------------------------------------------------
# Step 47B.3 — Word-level Viterbi
# ---------------------------------------------------------------------------

@dataclass
class ViterbiResult:
    variant: str
    n_tokens: int
    n_changed: int
    change_rate: float


def _word_viterbi(
    token_decoded: List[str],
    lattice_data: Dict[int, List[Tuple[str, float]]],
    model: WordBigramModel,
    folio_indices: List[int],
    alpha: float = 0.7,
) -> List[str]:
    """Word-level Viterbi over a folio's token sequence.

    For tokens with lattice size 1, pass through directly.
    For tokens with alternatives, select the one maximising bigram probability.
    """
    n = len(folio_indices)
    if n == 0:
        return []

    # Build per-position lattice
    lattices: List[List[Tuple[str, float]]] = []
    for idx in folio_indices:
        if idx in lattice_data:
            lattices.append(lattice_data[idx])
        else:
            lattices.append([(token_decoded[idx], 1.0)])

    # Viterbi DP
    # dp[j] = best log-prob ending at position i with entry j
    dp = [{} for _ in range(n)]
    backptr = [{} for _ in range(n)]

    # Initialize position 0
    for j, (word, prior) in enumerate(lattices[0]):
        dp[0][j] = math.log(prior + 1e-20) + model.log_prob_unigram(word)
        backptr[0][j] = -1

    # Forward pass
    for i in range(1, n):
        for j, (word_j, prior_j) in enumerate(lattices[i]):
            best_score = -float('inf')
            best_prev = 0
            bigram_score = model.log_prob_unigram(word_j)  # fallback

            for k, (word_k, _) in enumerate(lattices[i - 1]):
                if k not in dp[i - 1]:
                    continue
                bi = model.log_prob_bigram(word_j, word_k)
                uni = model.log_prob_unigram(word_j)
                score = dp[i - 1][k] + alpha * bi + (1 - alpha) * uni
                if score > best_score:
                    best_score = score
                    best_prev = k

            dp[i][j] = best_score + math.log(prior_j + 1e-20)
            backptr[i][j] = best_prev

    # Backtrace
    if not dp[n - 1]:
        return [token_decoded[idx] for idx in folio_indices]

    best_final = max(dp[n - 1], key=dp[n - 1].get)
    path = [best_final]
    for i in range(n - 1, 0, -1):
        path.append(backptr[i][path[-1]])
    path.reverse()

    return [lattices[i][path[i]][0] for i in range(n)]


def run_disamb_viterbi() -> None:
    """Step 47B.3: word-level Viterbi disambiguation."""
    t0 = time.time()
    print("=" * 70)
    print("STEP 47B.3: Word-Level Viterbi Disambiguation")
    print("=" * 70)

    rd = _results_dir()

    # Load parallel arrays
    sb = _safe_load(os.path.join(rd, 'signal_bigrams.json'))
    token_folios = sb.get('token_folios', [])
    token_decoded = sb.get('token_decoded', [])
    token_classifications = sb.get('token_classifications', [])
    n_tokens = len(token_decoded)

    if n_tokens == 0:
        print("  [SKIP] No data")
        return

    # Load lattice
    lattice_raw = _safe_load(os.path.join(rd, 'disamb_lattice.json'))
    lattice_entries = lattice_raw.get('lattice', {})
    lattice_data: Dict[int, List[Tuple[str, float]]] = {}
    for idx_str, entries in lattice_entries.items():
        lattice_data[int(idx_str)] = [(w, p) for w, p in entries]

    print(f"\n  {len(lattice_data)} tokens with alternatives")

    # Build bigram models
    sig_data = _safe_load(os.path.join(rd, 'signal_10k.json'))
    signal_words_set = {
        w['word'] for w in sig_data.get('word_signals', [])
        if w.get('is_genuine_signal')
    }

    folio_sequences: Dict[str, List[Tuple[int, str, str]]] = defaultdict(list)
    for i in range(n_tokens):
        folio_sequences[token_folios[i]].append(
            (i, token_decoded[i], token_classifications[i]),
        )

    # Train three models
    models = {}
    for variant in ['FULL', 'SIGNAL', 'GREEN']:
        model = WordBigramModel(variant)
        sequences = []
        for folio, tokens in folio_sequences.items():
            if variant == 'FULL':
                seq = [w for _, w, _ in tokens]
            elif variant == 'SIGNAL':
                seq = [w for _, w, c in tokens if c == 'SIGNAL']
            else:
                seq = [w for _, w, c in tokens if c == 'SIGNAL' and w in signal_words_set]
            if len(seq) >= 2:
                sequences.append(seq)
        model.train(sequences)
        models[variant] = model

    # Run Viterbi for each variant
    variant_results = {}
    for variant, model in models.items():
        print(f"\n  Running {variant} Viterbi...")
        disambiguated = list(token_decoded)  # copy

        for folio, tokens in folio_sequences.items():
            indices = [i for i, _, _ in tokens]
            result_words = _word_viterbi(
                token_decoded, lattice_data, model, indices, alpha=0.7,
            )
            for idx, word in zip(indices, result_words):
                disambiguated[idx] = word

        n_changed = sum(
            1 for i in range(n_tokens)
            if disambiguated[i] != token_decoded[i]
        )
        change_rate = n_changed / n_tokens if n_tokens else 0.0

        print(f"    Changed: {n_changed} ({change_rate:.1%})")

        variant_results[variant] = {
            'n_changed': n_changed,
            'change_rate': round(change_rate, 4),
            'disambiguated': disambiguated,
        }

    # Save (without full disambiguated arrays — just stats + sample)
    out = {
        'n_tokens': n_tokens,
        'variants': {},
    }
    for variant, vr in variant_results.items():
        # Sample first 100 changed tokens
        changed_examples = []
        for i in range(n_tokens):
            if vr['disambiguated'][i] != token_decoded[i]:
                changed_examples.append({
                    'index': i,
                    'folio': token_folios[i],
                    'original': token_decoded[i],
                    'disambiguated': vr['disambiguated'][i],
                })
                if len(changed_examples) >= 100:
                    break

        out['variants'][variant] = {
            'n_changed': vr['n_changed'],
            'change_rate': vr['change_rate'],
            'changed_examples': changed_examples,
            'disambiguated_tokens': vr['disambiguated'],
        }

    out['runtime_seconds'] = round(time.time() - t0, 2)

    out_path = os.path.join(rd, 'disamb_viterbi.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(out), f, indent=2)
    print(f"\n  Saved -> {out_path}")


# ---------------------------------------------------------------------------
# Step 47B.4 — Disambiguation quality evaluation
# ---------------------------------------------------------------------------

@dataclass
class DisambEvalResult:
    variants: List[Dict]
    bedrock_survival: Dict[str, bool]
    runtime_seconds: float


def run_disamb_eval() -> None:
    """Step 47B.4: evaluate disambiguation quality."""
    t0 = time.time()
    print("=" * 70)
    print("STEP 47B.4: Disambiguation Quality Evaluation")
    print("=" * 70)

    rd = _results_dir()

    # Load viterbi results
    viterbi = _safe_load(os.path.join(rd, 'disamb_viterbi.json'))
    variants = viterbi.get('variants', {})
    n_tokens = viterbi.get('n_tokens', 0)

    if n_tokens == 0:
        print("  [SKIP] No data")
        return

    # Load reference dict for dict-hit evaluation
    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    ref_tokens_raw = [
        w.lower() for w in ref_corpus.get_combined_tokens('latin')
        if len(w) >= 2
    ]
    word_freq = Counter(ref_tokens_raw)
    ref_word_set_10k = {w for w, _ in word_freq.most_common(10000)}

    # Load original decoded for comparison
    sb = _safe_load(os.path.join(rd, 'signal_bigrams.json'))
    token_decoded_orig = sb.get('token_decoded', [])
    token_classifications = sb.get('token_classifications', [])

    # Bedrock signal words
    BEDROCK = ['bene', 'codi', 'sero', 'sene', 'de', 'raro', 'dine', 'cola']

    # Original dict-hit
    orig_hits_10k = sum(1 for w in token_decoded_orig if w in ref_word_set_10k)
    orig_rate_10k = orig_hits_10k / n_tokens if n_tokens else 0.0
    print(f"\n  Original dict-hit (10K): {orig_rate_10k:.1%} ({orig_hits_10k})")

    eval_results = []
    bedrock_survival: Dict[str, bool] = {}

    for variant_name, vdata in variants.items():
        disambiguated = vdata.get('disambiguated_tokens', [])
        if not disambiguated:
            continue

        change_rate = vdata.get('change_rate', 0.0)

        # Dict-hit evaluation
        hits_10k = sum(1 for w in disambiguated if w in ref_word_set_10k)
        rate_10k = hits_10k / n_tokens if n_tokens else 0.0
        delta_10k = rate_10k - orig_rate_10k

        # Signal word survival
        orig_bedrock = Counter(
            w for w in token_decoded_orig if w in BEDROCK
        )
        disamb_bedrock = Counter(
            w for w in disambiguated if w in BEDROCK
        )
        survival = all(
            disamb_bedrock.get(w, 0) >= orig_bedrock.get(w, 0) * 0.5
            for w in BEDROCK
        )
        n_survived = sum(
            1 for w in BEDROCK if disamb_bedrock.get(w, 0) > 0
        )

        # Circular improvement: bigram probability comparison
        # (simplified: count bigram hits in disambiguated vs original)
        orig_bigram_pairs = sum(
            1 for i in range(n_tokens - 1)
            if token_decoded_orig[i] in ref_word_set_10k
            and token_decoded_orig[i + 1] in ref_word_set_10k
        )
        disamb_bigram_pairs = sum(
            1 for i in range(n_tokens - 1)
            if disambiguated[i] in ref_word_set_10k
            and disambiguated[i + 1] in ref_word_set_10k
        )

        eval_entry = {
            'variant': variant_name,
            'change_rate': change_rate,
            'dict_hit_10k': round(rate_10k, 4),
            'delta_dict_hit_10k': round(delta_10k, 4),
            'bedrock_survived': n_survived,
            'bedrock_total': len(BEDROCK),
            'all_bedrock_survived': survival,
            'orig_dict_bigram_pairs': orig_bigram_pairs,
            'disamb_dict_bigram_pairs': disamb_bigram_pairs,
            'beneficial': (
                0.05 <= change_rate <= 0.30
                and delta_10k > 0.01
                and survival
            ),
        }
        eval_results.append(eval_entry)
        bedrock_survival[variant_name] = survival

        print(f"\n  {variant_name}:")
        print(f"    Change rate: {change_rate:.1%}")
        print(f"    Dict-hit (10K): {rate_10k:.1%} (delta={delta_10k:+.1%})")
        print(f"    Bedrock survived: {n_survived}/{len(BEDROCK)}")
        print(f"    Beneficial: {eval_entry['beneficial']}")

    result = DisambEvalResult(
        variants=eval_results,
        bedrock_survival=bedrock_survival,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'disamb_eval.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2)
    print(f"\n  Saved -> {out_path}")


# ---------------------------------------------------------------------------
# Step 47B.5 — Comparison and best variant
# ---------------------------------------------------------------------------

@dataclass
class DisambCompareResult:
    best_variant: str
    best_delta_dict_hit: float
    disambiguation_beneficial: bool
    summary: List[Dict]
    verdict: str
    runtime_seconds: float


def run_disamb_compare() -> None:
    """Step 47B.5: compare variants and select best."""
    t0 = time.time()
    print("=" * 70)
    print("STEP 47B.5: Disambiguation Variant Comparison")
    print("=" * 70)

    rd = _results_dir()

    eval_data = _safe_load(os.path.join(rd, 'disamb_eval.json'))
    variants = eval_data.get('variants', [])

    if not variants:
        print("  [SKIP] No evaluation data")
        return

    # Sort by delta_dict_hit
    variants.sort(key=lambda x: -x.get('delta_dict_hit_10k', 0))

    best = variants[0] if variants else {}
    best_variant = best.get('variant', 'none')
    best_delta = best.get('delta_dict_hit_10k', 0)
    beneficial = best.get('beneficial', False)

    if beneficial:
        verdict = f'WORD_LEVEL_IMPROVEMENT ({best_variant}, delta={best_delta:+.1%})'
    else:
        verdict = 'NOT_BENEFICIAL (internal corpus bigram model too noisy)'

    print(f"\n  Ranking:")
    for v in variants:
        marker = ' *' if v.get('beneficial') else ''
        print(f"    {v['variant']:8s}: delta={v.get('delta_dict_hit_10k',0):+.4f}  "
              f"change={v.get('change_rate',0):.1%}  "
              f"bedrock={v.get('bedrock_survived',0)}/8{marker}")

    print(f"\n  Verdict: {verdict}")

    result = DisambCompareResult(
        best_variant=best_variant,
        best_delta_dict_hit=round(best_delta, 4),
        disambiguation_beneficial=beneficial,
        summary=variants,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'disamb_compare.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2)
    print(f"\n  Saved -> {out_path}")


# ---------------------------------------------------------------------------
# Track B orchestrator
# ---------------------------------------------------------------------------

def run_track_b_47() -> None:
    """Run all Track B steps."""
    run_disamb_lattice()
    print()
    run_disamb_bigram()
    print()
    run_disamb_viterbi()
    print()
    run_disamb_eval()
    print()
    run_disamb_compare()
