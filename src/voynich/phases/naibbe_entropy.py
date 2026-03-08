"""
Step 27.2 -- Naibbe Dice Cipher Entropy Shift Test
====================================================
Implement the Naibbe dice cipher with Greshko's 2025 published
parameters and test whether it produces an entropy shift vector
closer to the Voynich than the tachygraphic model (cosine 0.820).

Dependency chain:
    results/entropy_shift_cipher.json
    reference corpus (latin)
    core/ciphers.py
        -> naibbe_entropy.json
"""

from __future__ import annotations

import json
import math
import os
import random
import time
import zlib
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import load_corpus
from voynich.core.reference import load_reference_corpus
from voynich.core.stats import cosine_similarity, entropy_curve, first_order_entropy


# ---------------------------------------------------------------------------
# JSON serialiser
# ---------------------------------------------------------------------------

def _convert(obj: Any) -> Any:
    """Recursively convert dataclasses/numpy/NaN to JSON-safe types."""
    if hasattr(obj, '__dataclass_fields__'):
        return {k: _convert(v) for k, v in asdict(obj).items()}
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, dict):
        return {str(k): _convert(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_convert(item) for item in obj]
    if isinstance(obj, float) and (obj != obj):  # NaN
        return None
    if isinstance(obj, (bool, int, float, str, type(None))):
        return obj
    return str(obj)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class NaibbeGridEntry:
    n_tables: int
    bigram_prob: float
    prefix_prob: float
    suffix_prob: float
    mean_cosine: float
    std_cosine: float


@dataclass
class NaibbeEntropyResult:
    timestamp: str
    # Reference data
    voynich_shift_vector: List[float]
    latin_entropy_curve: Dict[str, float]
    voynich_entropy_curve: Dict[str, float]
    # Greshko default result
    greshko_mean_shift: List[float]
    greshko_cosine: float
    greshko_ci_lower: float
    greshko_ci_upper: float
    greshko_euclidean: float
    # Grid search
    n_grid_configs: int
    n_instantiations_per: int
    grid_top_10: List[Dict]
    best_config: Dict
    best_cosine: float
    best_ci: List[float]
    # Ranking against Phase 19.2
    phase19_ranking: List[Dict]
    updated_ranking: List[Dict]
    naibbe_rank: int
    phase19_best_cipher: str
    phase19_best_cosine: float
    naibbe_vs_tachygraphic: str
    # Discrimination test
    tachy_ci: List[float]
    naibbe_ci: List[float]
    ci_overlap: bool
    discrimination_verdict: str
    # Burstiness cross-check
    burstiness_cv_naibbe: float
    burstiness_cv_voynich: float
    burstiness_consistent: bool
    # Tri-state test
    naibbe_lz_ratio: float
    voynich_lz_ratio: float
    compression_consistent: bool
    naibbe_trie_colless: float
    naibbe_hmm_transition_entropy: float
    tristate_match_count: int
    # Gate
    gate_passed: bool
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Naibbe Dice Cipher
# ---------------------------------------------------------------------------

class NaibbeDiceCipher:
    """
    Naibbe (dice) cipher encoder following Greshko's 2025 parameters.

    The cipher uses multiple substitution tables, with table selection
    optionally conditioned on the previous output token (bigram
    dependence).  Prefixes and suffixes are added stochastically to
    produce Voynich-like word length distributions.

    Parameters
    ----------
    n_tables : int
        Number of random substitution tables (Greshko default: 2).
    bigram_prob : float
        Probability that table selection depends on previous output
        token (Greshko default: 0.20).
    word_len_range : tuple
        (min, max) character length for output words (Greshko default: (3, 6)).
    prefix_prob : float
        Probability of prepending a random prefix (Greshko default: 0.20).
    suffix_prob : float
        Probability of appending a random suffix (Greshko default: 0.30).
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_tables: int = 2,
        bigram_prob: float = 0.20,
        word_len_range: Tuple[int, int] = (3, 6),
        prefix_prob: float = 0.20,
        suffix_prob: float = 0.30,
        seed: int = 42,
    ):
        self.rng = random.Random(seed)
        self.n_tables = n_tables
        self.bigram_prob = bigram_prob
        self.word_len_range = word_len_range
        self.prefix_prob = prefix_prob
        self.suffix_prob = suffix_prob
        self.name = 'naibbe_greshko'

        # Build substitution tables: each maps lowercase letter -> cipher char
        # Use a restricted output alphabet (simulating dice notation / Voynich-like chars)
        self.output_alphabet = list('abcdefghijklmnopqrstuvwxyz')
        self.input_alphabet = list('abcdefghijklmnopqrstuvwxyz')

        self.tables: List[Dict[str, str]] = []
        for _ in range(n_tables):
            shuffled = self.output_alphabet.copy()
            self.rng.shuffle(shuffled)
            table = {inp: out for inp, out in zip(self.input_alphabet, shuffled)}
            self.tables.append(table)

        # Prefix and suffix character sets (small sets for realism)
        self.prefix_chars = list('abcde')
        self.suffix_chars = list('fghij')

    def encode(self, plaintext: str) -> str:
        """Encode plaintext through Naibbe cipher."""
        words = plaintext.lower().split()
        encoded_words: List[str] = []
        prev_table_idx = 0

        for word in words:
            # Table selection
            if encoded_words and self.rng.random() < self.bigram_prob:
                # Bigram-dependent: choose table based on last char of previous output
                prev_last = encoded_words[-1][-1] if encoded_words[-1] else 'a'
                table_idx = ord(prev_last) % self.n_tables
            else:
                table_idx = self.rng.randrange(self.n_tables)

            table = self.tables[table_idx]

            # Character-level substitution
            encoded_chars: List[str] = []
            for ch in word:
                if ch in table:
                    encoded_chars.append(table[ch])
                # Skip non-alphabetic

            if not encoded_chars:
                continue

            result = ''.join(encoded_chars)

            # Add prefix
            if self.rng.random() < self.prefix_prob:
                prefix_len = self.rng.randint(1, 2)
                prefix = ''.join(self.rng.choice(self.prefix_chars) for _ in range(prefix_len))
                result = prefix + result

            # Add suffix
            if self.rng.random() < self.suffix_prob:
                suffix_len = self.rng.randint(1, 2)
                suffix = ''.join(self.rng.choice(self.suffix_chars) for _ in range(suffix_len))
                result = result + suffix

            # Enforce word length range
            min_len, max_len = self.word_len_range
            if len(result) < min_len:
                # Pad with random chars
                while len(result) < min_len:
                    result += self.rng.choice(self.output_alphabet)
            elif len(result) > max_len:
                result = result[:max_len]

            encoded_words.append(result)
            prev_table_idx = table_idx

        return ' '.join(encoded_words)


# ---------------------------------------------------------------------------
# Entropy shift computation
# ---------------------------------------------------------------------------

def _compute_shift_vectors(
    latin_text: str,
    latin_curve: Dict[int, float],
    cipher: NaibbeDiceCipher,
    n_instantiations: int,
    max_order: int,
    base_seed: int,
) -> List[np.ndarray]:
    """Encode Latin with multiple seeds, compute shift vectors."""
    orders = list(range(max_order + 1))
    shift_vectors: List[np.ndarray] = []

    for inst in range(n_instantiations):
        # Create new cipher instance with different seed
        c = NaibbeDiceCipher(
            n_tables=cipher.n_tables,
            bigram_prob=cipher.bigram_prob,
            word_len_range=cipher.word_len_range,
            prefix_prob=cipher.prefix_prob,
            suffix_prob=cipher.suffix_prob,
            seed=base_seed + inst,
        )
        encoded = c.encode(latin_text)
        if not encoded or len(encoded) < 50:
            continue

        enc_curve = entropy_curve(encoded, max_order=max_order)
        shift = np.array([
            enc_curve.get(k, 0) - latin_curve.get(k, 0)
            for k in orders
        ])
        shift_vectors.append(shift)

    return shift_vectors


# ---------------------------------------------------------------------------
# Cross-check: burstiness
# ---------------------------------------------------------------------------

def _compute_burstiness_cv(tokens: List[str], min_occurrences: int = 5,
                           top_n_exclude: int = 20) -> float:
    """
    Compute mean coefficient of variation of inter-arrival gaps.
    Simplified reimplementation of _corpus_cv_stats from burstiness_test.py.
    """
    # Build positions
    positions: Dict[str, List[int]] = {}
    for i, tok in enumerate(tokens):
        positions.setdefault(tok, []).append(i)

    # Exclude top-N most frequent
    freq = sorted(positions.keys(), key=lambda t: len(positions[t]), reverse=True)
    exclude = set(freq[:top_n_exclude])

    cvs: List[float] = []
    for tok, poslist in positions.items():
        if tok in exclude or len(poslist) < min_occurrences:
            continue
        gaps = [poslist[i + 1] - poslist[i] for i in range(len(poslist) - 1)]
        if not gaps:
            continue
        mean_gap = np.mean(gaps)
        if mean_gap > 0:
            cv = float(np.std(gaps) / mean_gap)
            cvs.append(cv)

    return float(np.mean(cvs)) if cvs else 0.0


# ---------------------------------------------------------------------------
# Cross-check: trie Colless index
# ---------------------------------------------------------------------------

def _compute_trie_colless(tokens: List[str], max_depth: int = 8) -> float:
    """
    Compute Colless-like imbalance index for a character trie built from tokens.
    Measures how balanced the trie branching structure is.
    """
    # Build trie as nested dicts
    trie: Dict = {}
    for tok in tokens:
        node = trie
        for ch in tok[:max_depth]:
            if ch not in node:
                node[ch] = {}
            node = node[ch]

    # Compute subtree sizes and Colless imbalance
    def subtree_size(node: Dict) -> int:
        if not node:
            return 1
        return sum(subtree_size(child) for child in node.values())

    def colless_sum(node: Dict) -> Tuple[int, int]:
        """Returns (total_imbalance, n_internal_nodes)."""
        if not node or len(node) < 2:
            return (0, 0)
        child_sizes = [subtree_size(child) for child in node.values()]
        # Colless: sum of |size_i - size_j| for all pairs i < j
        imbalance = 0
        for i in range(len(child_sizes)):
            for j in range(i + 1, len(child_sizes)):
                imbalance += abs(child_sizes[i] - child_sizes[j])

        total_imb = imbalance
        total_n = 1
        for child in node.values():
            ci, cn = colless_sum(child)
            total_imb += ci
            total_n += cn

        return (total_imb, total_n)

    total_imb, total_n = colless_sum(trie)
    return total_imb / total_n if total_n > 0 else 0.0


# ---------------------------------------------------------------------------
# Cross-check: HMM transition entropy (simplified)
# ---------------------------------------------------------------------------

def _compute_hmm_transition_entropy(tokens: List[str], n_states: int = 8) -> float:
    """
    Simplified HMM-like transition entropy estimate.
    Uses character bigram transition matrix entropy as a proxy for
    full HMM training (which is expensive). This captures the key
    property: how predictable are state transitions.
    """
    text = ' '.join(tokens)
    # Build bigram transition counts
    transitions: Dict[str, Counter] = {}
    for i in range(len(text) - 1):
        c1, c2 = text[i], text[i + 1]
        if c1 == ' ' or c2 == ' ':
            continue
        if c1 not in transitions:
            transitions[c1] = Counter()
        transitions[c1][c2] += 1

    # Compute per-context entropy
    entropies: List[float] = []
    for ctx, counts in transitions.items():
        total = sum(counts.values())
        if total < 2:
            continue
        ent = 0.0
        for n in counts.values():
            p = n / total
            if p > 0:
                ent -= p * math.log2(p)
        entropies.append(ent)

    return float(np.mean(entropies)) if entropies else 0.0


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_naibbe_entropy() -> None:
    """Step 27.2: Naibbe dice cipher entropy shift test."""
    t0 = time.time()
    rd = str(_results_dir())
    max_order = 6

    print("=" * 60)
    print("Step 27.2: Naibbe Dice Cipher Entropy Shift Test")
    print("=" * 60)

    # ── 1. Load reference data ────────────────────────────────────────
    print("\n  1. Loading reference data ...")

    # Load Phase 19.2 results
    shift_path = os.path.join(rd, 'entropy_shift_cipher.json')
    with open(shift_path) as f:
        phase19 = json.load(f)

    observed_shift = np.array(phase19['observed_shift_vector'])
    voynich_curve_raw = phase19['voynich_entropy_curve']
    voynich_curve = {int(k): v for k, v in voynich_curve_raw.items()}
    latin_curve_raw = phase19['latin_entropy_curve']
    latin_curve = {int(k): v for k, v in latin_curve_raw.items()}

    phase19_best = phase19['best_match_cipher']
    phase19_best_cos = phase19['best_match_cosine']
    phase19_ranking = phase19['cipher_ranking']

    # Tachygraphic CI from Phase 19.2 profiles
    tachy_ci = [phase19_best_cos, phase19_best_cos]  # default
    for profile in phase19.get('mechanism_profiles', []):
        if profile.get('name') == 'tachygraphic':
            tachy_ci = [profile.get('ci_lower', phase19_best_cos),
                        profile.get('ci_upper', phase19_best_cos)]
            break

    print(f"    Observed shift: {[round(float(v), 3) for v in observed_shift]}")
    print(f"    Phase 19.2 best: {phase19_best} (cos={phase19_best_cos:.4f})")
    print(f"    Tachygraphic CI: [{tachy_ci[0]:.4f}, {tachy_ci[1]:.4f}]")

    # Load Latin text for encoding
    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    latin_tokens = ref_corpus.get_combined_tokens('latin')
    latin_text = ' '.join(latin_tokens[:5000]) if latin_tokens else ''

    # Load Phase 18 reference values
    burstiness_path = os.path.join(rd, 'burstiness_test.json')
    voynich_cv = 1.014  # default
    if os.path.exists(burstiness_path):
        with open(burstiness_path) as f:
            burst = json.load(f)
        voynich_cv = burst.get('voynich_mean_cv', 1.014)

    lz_path = os.path.join(rd, 'lz_complexity.json')
    voynich_lz = 0.3297  # default
    if os.path.exists(lz_path):
        with open(lz_path) as f:
            lz_data = json.load(f)
        voynich_lz = lz_data.get('voynich_asymptotic_zlib', 0.3297)

    hmm_path = os.path.join(rd, 'hmm_pos_induction.json')
    voynich_hmm_ent = 1.006  # default
    if os.path.exists(hmm_path):
        with open(hmm_path) as f:
            hmm_data = json.load(f)
        voynich_hmm_ent = hmm_data.get('transition_entropy_mean', 1.006)

    # ── 2. Greshko default parameters ─────────────────────────────────
    print("\n  2. Testing Greshko default parameters ...")
    print("    n_tables=2, bigram=0.20, word_len=(3,6), prefix=0.20, suffix=0.30")

    greshko_cipher = NaibbeDiceCipher(
        n_tables=2, bigram_prob=0.20, word_len_range=(3, 6),
        prefix_prob=0.20, suffix_prob=0.30, seed=42,
    )

    greshko_shifts = _compute_shift_vectors(
        latin_text, latin_curve, greshko_cipher,
        n_instantiations=20, max_order=max_order, base_seed=5000,
    )

    if greshko_shifts:
        greshko_mean = np.mean(greshko_shifts, axis=0)
        greshko_cos = float(cosine_similarity(greshko_mean, observed_shift))
        cos_sims = [float(cosine_similarity(sv, observed_shift)) for sv in greshko_shifts]
        greshko_ci = [float(np.percentile(cos_sims, 2.5)),
                      float(np.percentile(cos_sims, 97.5))]
        greshko_euc = float(np.linalg.norm(greshko_mean - observed_shift))
    else:
        greshko_mean = np.zeros(max_order + 1)
        greshko_cos = 0.0
        greshko_ci = [0.0, 0.0]
        greshko_euc = 999.0

    print(f"    Greshko cosine: {greshko_cos:.4f} [{greshko_ci[0]:.3f}, {greshko_ci[1]:.3f}]")
    print(f"    Greshko euclidean: {greshko_euc:.4f}")
    print(f"    Mean shift: {[round(float(v), 3) for v in greshko_mean]}")

    # ── 3. Parameter grid search ──────────────────────────────────────
    print("\n  3. Parameter grid search (81 configurations x 5 seeds) ...")

    n_tables_grid = [1, 2, 3]
    bigram_grid = [0.10, 0.20, 0.30]
    prefix_grid = [0.10, 0.20, 0.30]
    suffix_grid = [0.20, 0.30, 0.40]

    grid_results: List[NaibbeGridEntry] = []
    total_configs = len(n_tables_grid) * len(bigram_grid) * len(prefix_grid) * len(suffix_grid)
    config_idx = 0

    for nt in n_tables_grid:
        for bp in bigram_grid:
            for pp in prefix_grid:
                for sp in suffix_grid:
                    config_idx += 1
                    cipher = NaibbeDiceCipher(
                        n_tables=nt, bigram_prob=bp,
                        word_len_range=(3, 6),
                        prefix_prob=pp, suffix_prob=sp,
                        seed=42,
                    )
                    shifts = _compute_shift_vectors(
                        latin_text, latin_curve, cipher,
                        n_instantiations=5, max_order=max_order,
                        base_seed=10000 + config_idx * 100,
                    )

                    if shifts:
                        mean_shift = np.mean(shifts, axis=0)
                        cos_vals = [float(cosine_similarity(sv, observed_shift))
                                    for sv in shifts]
                        mean_cos = float(np.mean(cos_vals))
                        std_cos = float(np.std(cos_vals))
                    else:
                        mean_cos = 0.0
                        std_cos = 0.0

                    grid_results.append(NaibbeGridEntry(
                        n_tables=nt, bigram_prob=bp,
                        prefix_prob=pp, suffix_prob=sp,
                        mean_cosine=round(mean_cos, 4),
                        std_cosine=round(std_cos, 4),
                    ))

                    if config_idx % 20 == 0 or config_idx == total_configs:
                        print(f"    Completed {config_idx}/{total_configs} configs ...")

    # Sort by cosine (descending)
    grid_results.sort(key=lambda g: g.mean_cosine, reverse=True)

    print(f"\n    Top 5 configurations:")
    for i, g in enumerate(grid_results[:5]):
        print(f"      {i + 1}. nt={g.n_tables} bp={g.bigram_prob:.2f} "
              f"pp={g.prefix_prob:.2f} sp={g.suffix_prob:.2f} "
              f"cos={g.mean_cosine:.4f} +/- {g.std_cosine:.4f}")

    best_grid = grid_results[0]
    best_cosine = best_grid.mean_cosine

    # Compute CI for best grid config with 20 instantiations
    best_cipher = NaibbeDiceCipher(
        n_tables=best_grid.n_tables, bigram_prob=best_grid.bigram_prob,
        word_len_range=(3, 6),
        prefix_prob=best_grid.prefix_prob, suffix_prob=best_grid.suffix_prob,
        seed=42,
    )
    best_shifts = _compute_shift_vectors(
        latin_text, latin_curve, best_cipher,
        n_instantiations=20, max_order=max_order, base_seed=50000,
    )
    if best_shifts:
        best_cos_vals = [float(cosine_similarity(sv, observed_shift)) for sv in best_shifts]
        best_ci = [float(np.percentile(best_cos_vals, 2.5)),
                   float(np.percentile(best_cos_vals, 97.5))]
        best_cosine = float(np.mean(best_cos_vals))
    else:
        best_ci = [0.0, 0.0]

    print(f"\n    Best config (20 seeds): cos={best_cosine:.4f} "
          f"[{best_ci[0]:.3f}, {best_ci[1]:.3f}]")

    # ── 4. Rank against Phase 19.2 ───────────────────────────────────
    print("\n  4. Ranking against Phase 19.2 mechanisms ...")

    # Build updated ranking
    naibbe_entry = {
        'rank': 0,
        'name': 'naibbe_greshko',
        'cosine_similarity': round(greshko_cos, 4),
        'ci': [round(v, 4) for v in greshko_ci],
    }
    naibbe_best_entry = {
        'rank': 0,
        'name': 'naibbe_best_grid',
        'cosine_similarity': round(best_cosine, 4),
        'ci': [round(v, 4) for v in best_ci],
    }

    all_entries = []
    for r in phase19_ranking:
        all_entries.append({
            'name': r['name'],
            'cosine_similarity': r['cosine_similarity'],
        })
    all_entries.append({
        'name': 'naibbe_greshko',
        'cosine_similarity': round(greshko_cos, 4),
    })
    if abs(best_cosine - greshko_cos) > 0.01:
        all_entries.append({
            'name': 'naibbe_best_grid',
            'cosine_similarity': round(best_cosine, 4),
        })

    all_entries.sort(key=lambda e: e['cosine_similarity'], reverse=True)
    for i, e in enumerate(all_entries):
        e['rank'] = i + 1

    naibbe_rank = next(
        (e['rank'] for e in all_entries if e['name'] == 'naibbe_greshko'),
        len(all_entries),
    )

    print(f"\n    Updated ranking:")
    for e in all_entries:
        marker = " <--" if 'naibbe' in e['name'] else ""
        print(f"      {e['rank']}. {e['name']:25s} cos={e['cosine_similarity']:.4f}{marker}")

    # Determine verdict vs tachygraphic
    if greshko_cos > phase19_best_cos:
        naibbe_vs = 'NAIBBE_SUPERIOR'
    elif greshko_cos > 0.566:
        naibbe_vs = 'TACHYGRAPHIC_PREFERRED'
    else:
        naibbe_vs = 'TACHYGRAPHIC_CONFIRMED'

    # ── 5. Discrimination test ────────────────────────────────────────
    print("\n  5. Discrimination test (CI overlap) ...")

    ci_overlap = (greshko_ci[0] <= tachy_ci[1] and tachy_ci[0] <= greshko_ci[1])

    if not ci_overlap:
        disc_verdict = f"DISCRIMINATED: tachygraphic [{tachy_ci[0]:.3f}, {tachy_ci[1]:.3f}] vs naibbe [{greshko_ci[0]:.3f}, {greshko_ci[1]:.3f}]"
    else:
        disc_verdict = f"DEGENERATE: CIs overlap — tachygraphic [{tachy_ci[0]:.3f}, {tachy_ci[1]:.3f}] vs naibbe [{greshko_ci[0]:.3f}, {greshko_ci[1]:.3f}]"

    print(f"    {disc_verdict}")

    # ── 6. Burstiness cross-check ─────────────────────────────────────
    print("\n  6. Burstiness cross-check ...")

    # Generate Naibbe text with Greshko defaults for cross-checks
    naibbe_text = greshko_cipher.encode(latin_text)
    naibbe_tokens = naibbe_text.split()

    naibbe_cv = _compute_burstiness_cv(naibbe_tokens)
    burst_consistent = abs(naibbe_cv - voynich_cv) < 0.3  # within 0.3

    print(f"    Naibbe CV: {naibbe_cv:.4f}")
    print(f"    Voynich CV: {voynich_cv:.4f}")
    print(f"    Consistent: {burst_consistent}")

    # ── 7. Tri-state test ─────────────────────────────────────────────
    print("\n  7. Tri-state test (trie + HMM + LZ) ...")

    # LZ compression
    naibbe_compressed = len(zlib.compress(naibbe_text.encode()))
    naibbe_lz = naibbe_compressed / len(naibbe_text.encode()) if naibbe_text else 0.0
    lz_consistent = abs(naibbe_lz - voynich_lz) < 0.15

    # Trie Colless
    naibbe_colless = _compute_trie_colless(naibbe_tokens)

    # HMM transition entropy (proxy)
    naibbe_hmm = _compute_hmm_transition_entropy(naibbe_tokens)

    # Count how many of 3 tests are consistent
    # Compare to Voynich: CV~1.014, LZ~0.330, HMM~1.006
    hmm_consistent = abs(naibbe_hmm - voynich_hmm_ent) < 0.5

    tristate_matches = sum([burst_consistent, lz_consistent, hmm_consistent])

    print(f"    LZ ratio: naibbe={naibbe_lz:.4f} voynich={voynich_lz:.4f} "
          f"consistent={lz_consistent}")
    print(f"    Trie Colless: {naibbe_colless:.2f}")
    print(f"    HMM entropy: naibbe={naibbe_hmm:.4f} voynich={voynich_hmm_ent:.4f} "
          f"consistent={hmm_consistent}")
    print(f"    Tri-state matches: {tristate_matches}/3")

    # ── 8. Gate and verdict ───────────────────────────────────────────
    print("\n  8. Gate and verdict ...")

    gate_passed = greshko_cos > 0.5

    if greshko_cos > phase19_best_cos and not ci_overlap:
        verdict = (f"NAIBBE_SUPERIOR: Naibbe (cos={greshko_cos:.4f}) beats "
                   f"tachygraphic ({phase19_best_cos:.4f}) with non-overlapping CIs")
    elif greshko_cos > phase19_best_cos and ci_overlap:
        verdict = (f"DEGENERATE: Naibbe (cos={greshko_cos:.4f}) slightly beats "
                   f"tachygraphic ({phase19_best_cos:.4f}) but CIs overlap")
    elif ci_overlap:
        verdict = (f"DEGENERATE: Naibbe (cos={greshko_cos:.4f}) and "
                   f"tachygraphic ({phase19_best_cos:.4f}) cannot be separated")
    elif greshko_cos > 0.566:
        verdict = (f"TACHYGRAPHIC_PREFERRED: tachygraphic ({phase19_best_cos:.4f}) "
                   f"beats Naibbe ({greshko_cos:.4f}) with discriminated CIs")
    else:
        verdict = (f"TACHYGRAPHIC_CONFIRMED: Naibbe (cos={greshko_cos:.4f}) ranks "
                   f"below homophonic (0.566), tachygraphic strongly preferred")

    print(f"    Gate: {'PASS' if gate_passed else 'FAIL'}")
    print(f"    Verdict: {verdict}")

    # ── 9. Save ───────────────────────────────────────────────────────
    result = NaibbeEntropyResult(
        timestamp=time.strftime('%Y-%m-%dT%H:%M:%S'),
        voynich_shift_vector=[round(float(v), 4) for v in observed_shift],
        latin_entropy_curve={str(k): round(v, 4) for k, v in latin_curve.items()},
        voynich_entropy_curve={str(k): round(v, 4) for k, v in voynich_curve.items()},
        greshko_mean_shift=[round(float(v), 4) for v in greshko_mean],
        greshko_cosine=round(greshko_cos, 4),
        greshko_ci_lower=round(greshko_ci[0], 4),
        greshko_ci_upper=round(greshko_ci[1], 4),
        greshko_euclidean=round(greshko_euc, 4),
        n_grid_configs=len(grid_results),
        n_instantiations_per=5,
        grid_top_10=[_convert(asdict(g)) for g in grid_results[:10]],
        best_config=_convert(asdict(best_grid)),
        best_cosine=round(best_cosine, 4),
        best_ci=[round(v, 4) for v in best_ci],
        phase19_ranking=phase19_ranking,
        updated_ranking=all_entries,
        naibbe_rank=naibbe_rank,
        phase19_best_cipher=phase19_best,
        phase19_best_cosine=round(phase19_best_cos, 4),
        naibbe_vs_tachygraphic=naibbe_vs,
        tachy_ci=[round(v, 4) for v in tachy_ci],
        naibbe_ci=[round(v, 4) for v in greshko_ci],
        ci_overlap=ci_overlap,
        discrimination_verdict=disc_verdict,
        burstiness_cv_naibbe=round(naibbe_cv, 4),
        burstiness_cv_voynich=round(voynich_cv, 4),
        burstiness_consistent=burst_consistent,
        naibbe_lz_ratio=round(naibbe_lz, 4),
        voynich_lz_ratio=round(voynich_lz, 4),
        compression_consistent=lz_consistent,
        naibbe_trie_colless=round(naibbe_colless, 2),
        naibbe_hmm_transition_entropy=round(naibbe_hmm, 4),
        tristate_match_count=tristate_matches,
        gate_passed=gate_passed,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'naibbe_entropy.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2)
    print(f"\n  -> {out_path}")
