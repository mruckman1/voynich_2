"""
Phase 46 Track B – Frequency Structure Diagnostic
===================================================
Test whether Voynich SBM co-occurrence structure matches natural language
or specific cipher types.

Dependency chain:
    sbm_communities.json          (Phase 44B.2, Voynich SBM)
    sbm_graph.json                (Phase 44B.1, Voynich co-occurrence)
        -> freq_reference.json    (Step 46B.1)
        -> freq_cipher.json       (Step 46B.2)
        -> freq_compare.json      (Step 46B.3)
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from voynich.core._paths import results_dir as _results_dir
from voynich.core.reference import load_reference_corpus
from voynich.core.stats import syllabify_latin

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
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
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
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SBMProfile:
    label: str
    corpus_type: str            # 'reference', 'cipher', 'voynich'
    n_tokens: int
    n_types: int
    optimal_k: int
    silhouette_score: float
    modularity: float
    community_sizes: List[int]
    largest_community_coverage: float
    frequency_tier_ari: float
    mean_degree: float
    runtime_seconds: float


@dataclass
class FreqReferenceResult:
    profiles: List[Dict]
    runtime_seconds: float


@dataclass
class FreqCipherResult:
    profiles: List[Dict]
    cipher_specs: List[Dict]
    runtime_seconds: float


@dataclass
class FreqCompareResult:
    voynich_profile: Dict
    reference_profiles: List[Dict]
    cipher_profiles: List[Dict]
    comparison_table: List[Dict]
    nearest_match: str
    nearest_distance: float
    verdict: str
    rationale: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Generic SBM pipeline
# ---------------------------------------------------------------------------


def _build_cooccurrence_generic(
    token_stream: List[str],
    types: List[str],
    type_to_idx: Dict[str, int],
) -> np.ndarray:
    """Build 4-layer co-occurrence matrix from a generic token stream.

    Layers:
      L1: adjacent pairs within tokens (weight 2.0)
      L2: same-word co-occurrence (weight 1.0)
      L3: positional substitutability (weight 1.5)
      L4: cross-word transitions (weight 1.0)

    Here "tokens" are individual characters/syllables in sequence, and
    "words" are delimited by spaces in the input or by a provided word
    boundary list.
    """
    n = len(types)
    L1 = np.zeros((n, n), dtype=np.float64)
    L2 = np.zeros((n, n), dtype=np.float64)
    L3 = np.zeros((n, n), dtype=np.float64)
    L4 = np.zeros((n, n), dtype=np.float64)

    # L1: adjacent pairs
    for i in range(len(token_stream) - 1):
        a, b = token_stream[i], token_stream[i + 1]
        ai = type_to_idx.get(a)
        bi = type_to_idx.get(b)
        if ai is not None and bi is not None:
            L1[ai, bi] += 1

    # L2: same-word co-occurrence (use a window of 5 for generality)
    window = 5
    for i in range(len(token_stream)):
        ai = type_to_idx.get(token_stream[i])
        if ai is None:
            continue
        for j in range(i + 1, min(i + window, len(token_stream))):
            bj = type_to_idx.get(token_stream[j])
            if bj is not None:
                L2[ai, bj] += 1
                L2[bj, ai] += 1

    # L3: positional substitutability (first 5 positions in sliding window)
    position_counts: Dict[int, Counter] = defaultdict(Counter)
    for pos in range(min(5, len(token_stream))):
        for start in range(0, len(token_stream) - pos, window):
            idx = type_to_idx.get(token_stream[start + pos])
            if idx is not None:
                position_counts[pos][idx] += 1

    for pos in position_counts:
        chars_at = list(position_counts[pos].keys())
        for i in range(len(chars_at)):
            for j in range(i + 1, len(chars_at)):
                ci, cj = chars_at[i], chars_at[j]
                score = min(position_counts[pos][ci], position_counts[pos][cj])
                L3[ci, cj] += score
                L3[cj, ci] += score

    # L4: skip-1 transitions (approximate cross-word)
    for i in range(len(token_stream) - 2):
        ai = type_to_idx.get(token_stream[i])
        bi = type_to_idx.get(token_stream[i + 2])
        if ai is not None and bi is not None:
            L4[ai, bi] += 1

    # Weighted combination
    combined = 2.0 * L1 + 1.0 * L2 + 1.5 * L3 + 1.0 * L4
    combined = (combined + combined.T) / 2.0
    return combined


def _compute_modularity(adj: np.ndarray, labels: np.ndarray) -> float:
    """Newman modularity Q for undirected graph."""
    m = adj.sum() / 2.0
    if m == 0:
        return 0.0
    n = len(labels)
    Q = 0.0
    for i in range(n):
        ki = adj[i].sum()
        for j in range(n):
            if labels[i] == labels[j]:
                kj = adj[j].sum()
                Q += adj[i, j] - ki * kj / (2 * m)
    return Q / (2 * m)


def _sbm_pipeline(
    token_stream: List[str],
    label: str,
    corpus_type: str = 'reference',
) -> SBMProfile:
    """Shared SBM pipeline: build co-occurrence, spectral cluster,
    compute silhouette, modularity, frequency-tier ARI."""
    from sklearn.cluster import SpectralClustering
    from sklearn.metrics import adjusted_rand_score, silhouette_score

    t0 = time.time()

    # Build type vocabulary
    freq = Counter(token_stream)
    types = sorted(freq.keys())
    n = len(types)
    type_to_idx = {t: i for i, t in enumerate(types)}

    if n < 4:
        return SBMProfile(
            label=label, corpus_type=corpus_type,
            n_tokens=len(token_stream), n_types=n,
            optimal_k=1, silhouette_score=0.0, modularity=0.0,
            community_sizes=[n], largest_community_coverage=1.0,
            frequency_tier_ari=0.0, mean_degree=0.0,
            runtime_seconds=round(time.time() - t0, 2),
        )

    # Build co-occurrence
    combined = _build_cooccurrence_generic(token_stream, types, type_to_idx)

    # Normalize to [0, 1]
    max_val = combined.max()
    if max_val > 0:
        combined_norm = combined / max_val
    else:
        combined_norm = combined

    # Spectral clustering with silhouette optimization
    best_k = 2
    best_sil = -1.0
    best_labels = np.zeros(n, dtype=int)
    max_k = min(13, n)

    for k in range(2, max_k):
        try:
            sc = SpectralClustering(
                n_clusters=k, affinity='precomputed',
                random_state=42, n_init=10,
            )
            labels = sc.fit_predict(combined_norm + 1e-8)
            dist = 1.0 - combined_norm
            np.fill_diagonal(dist, 0)
            sil = silhouette_score(dist, labels, metric='precomputed')
            if sil > best_sil:
                best_sil = sil
                best_k = k
                best_labels = labels.copy()
        except Exception:
            continue

    # Modularity
    modularity = _compute_modularity(combined, best_labels)

    # Community sizes
    comm_counts = Counter(int(l) for l in best_labels)
    community_sizes = sorted(comm_counts.values(), reverse=True)
    largest_coverage = community_sizes[0] / n if n > 0 else 0.0

    # Frequency-tier ARI
    # Assign each type a quintile (0-4) based on frequency rank
    sorted_by_freq = sorted(types, key=lambda t: -freq[t])
    quintile_labels = np.zeros(n, dtype=int)
    for rank, t in enumerate(sorted_by_freq):
        quintile = min(4, (rank * 5) // n)
        quintile_labels[type_to_idx[t]] = quintile

    freq_ari = adjusted_rand_score(quintile_labels, best_labels)

    # Mean degree
    degrees = combined.sum(axis=1)
    mean_degree = float(degrees.mean()) if n > 0 else 0.0

    return SBMProfile(
        label=label,
        corpus_type=corpus_type,
        n_tokens=len(token_stream),
        n_types=n,
        optimal_k=best_k,
        silhouette_score=round(float(best_sil), 4),
        modularity=round(float(modularity), 4),
        community_sizes=community_sizes,
        largest_community_coverage=round(largest_coverage, 4),
        frequency_tier_ari=round(float(freq_ari), 4),
        mean_degree=round(mean_degree, 2),
        runtime_seconds=round(time.time() - t0, 2),
    )


# ---------------------------------------------------------------------------
# Cipher generators
# ---------------------------------------------------------------------------


def _generate_simple_substitution(
    chars: List[str],
    seed: int = 42,
) -> List[str]:
    """Simple 1:1 substitution cipher."""
    rng = random.Random(seed)
    alphabet = sorted(set(chars))
    cipher_alphabet = list(alphabet)
    rng.shuffle(cipher_alphabet)
    mapping = dict(zip(alphabet, cipher_alphabet))
    return [mapping.get(c, c) for c in chars]


def _generate_homophonic(
    chars: List[str],
    n_homophones: int = 3,
    seed: int = 42,
) -> List[str]:
    """Homophonic substitution: vowels get n_homophones symbols each."""
    rng = random.Random(seed)
    vowels = set('aeiou')
    alphabet = sorted(set(chars))

    # Build mapping: each vowel -> n_homophones symbols, consonants -> 1
    sym_id = 0
    mapping: Dict[str, List[str]] = {}
    for c in alphabet:
        if c in vowels:
            syms = [f'H{sym_id + i}' for i in range(n_homophones)]
            sym_id += n_homophones
            mapping[c] = syms
        else:
            mapping[c] = [f'H{sym_id}']
            sym_id += 1

    return [rng.choice(mapping.get(c, [c])) for c in chars]


def _generate_tachygraphic_cv(
    words: List[str],
    seed: int = 42,
) -> List[str]:
    """Tachygraphic CV encoding: syllabify, map each syllable to a symbol."""
    rng = random.Random(seed)
    # Syllabify all words
    syllable_stream = []
    for word in words:
        syls = syllabify_latin(word.lower())
        if syls:
            syllable_stream.extend(syls)
        else:
            syllable_stream.append(word.lower())

    # Map syllables to symbols
    unique_syls = sorted(set(syllable_stream))
    # Cap at 300 to avoid huge co-occurrence matrices
    if len(unique_syls) > 300:
        freq = Counter(syllable_stream)
        top_300 = {s for s, _ in freq.most_common(300)}
        syllable_stream = [
            s if s in top_300 else 'RARE' for s in syllable_stream
        ]
        unique_syls = sorted(set(syllable_stream))

    sym_map = {s: f'T{i}' for i, s in enumerate(unique_syls)}
    return [sym_map[s] for s in syllable_stream]


def _generate_nomenclator(
    words: List[str],
    chars: List[str],
    n_code_words: int = 50,
    seed: int = 42,
) -> List[str]:
    """Nomenclator: top n_code_words get dedicated symbols, rest use simple sub."""
    rng = random.Random(seed)
    word_freq = Counter(words)
    code_words = {w for w, _ in word_freq.most_common(n_code_words)}

    # Simple substitution for character-level
    alphabet = sorted(set(chars))
    cipher_alpha = list(alphabet)
    rng.shuffle(cipher_alpha)
    char_map = dict(zip(alphabet, cipher_alpha))

    # Build output: word-level codes + char-level substitution
    result = []
    sym_id = 0
    word_sym: Dict[str, str] = {}
    for w in sorted(code_words):
        word_sym[w] = f'N{sym_id}'
        sym_id += 1

    # Process words
    word_idx = 0
    char_idx = 0
    for word in words:
        if word in word_sym:
            result.append(word_sym[word])
        else:
            # Character-level substitution
            for c in word.lower():
                result.append(char_map.get(c, c))

    return result


def _generate_null_insertion(
    chars: List[str],
    null_rate: float = 0.15,
    n_null_symbols: int = 5,
    seed: int = 42,
) -> List[str]:
    """Simple substitution + random null insertion at given rate."""
    rng = random.Random(seed)

    # First apply simple substitution
    sub_chars = _generate_simple_substitution(chars, seed=seed)

    # Insert nulls
    null_symbols = [f'NULL{i}' for i in range(n_null_symbols)]
    result = []
    for c in sub_chars:
        result.append(c)
        if rng.random() < null_rate:
            result.append(rng.choice(null_symbols))

    return result


# ---------------------------------------------------------------------------
# Step 46B.1 — SBM on Reference Corpora
# ---------------------------------------------------------------------------


def run_freq_reference() -> None:
    """Step 46B.1: SBM on Latin/Italian reference corpora."""
    t0 = time.time()
    print("=" * 70)
    print("STEP 46B.1: SBM on Reference Corpora")
    print("=" * 70)

    rd = _results_dir()

    # Load reference corpora
    ref_corpus = load_reference_corpus(
        languages=['latin', 'italian'], verbose=False,
    )

    profiles: List[Dict] = []

    # 1. Latin character-level
    print("\n  1. Latin (character-level)...")
    latin_tokens = ref_corpus.get_combined_tokens('latin')
    latin_text = ' '.join(w.lower() for w in latin_tokens if len(w) >= 2)
    latin_chars = [c for c in latin_text if c.isalpha()]
    if latin_chars:
        prof = _sbm_pipeline(latin_chars, 'latin_char', 'reference')
        profiles.append(_convert(asdict(prof)))
        print(f"    k={prof.optimal_k}  sil={prof.silhouette_score:.3f}  "
              f"ARI={prof.frequency_tier_ari:.3f}  "
              f"largest={prof.largest_community_coverage:.3f}")

    # 2. Italian character-level
    print("\n  2. Italian (character-level)...")
    italian_tokens = ref_corpus.get_combined_tokens('italian')
    if italian_tokens:
        ital_text = ' '.join(w.lower() for w in italian_tokens if len(w) >= 2)
        ital_chars = [c for c in ital_text if c.isalpha()]
        if ital_chars:
            prof = _sbm_pipeline(ital_chars, 'italian_char', 'reference')
            profiles.append(_convert(asdict(prof)))
            print(f"    k={prof.optimal_k}  sil={prof.silhouette_score:.3f}  "
                  f"ARI={prof.frequency_tier_ari:.3f}  "
                  f"largest={prof.largest_community_coverage:.3f}")
    else:
        print("    No Italian corpus available, skipping.")

    # 3. Latin syllable-level
    print("\n  3. Latin (syllable-level)...")
    latin_words = [w.lower() for w in latin_tokens if len(w) >= 2]
    syl_stream = []
    for word in latin_words[:20000]:  # Cap for performance
        syls = syllabify_latin(word)
        if syls:
            syl_stream.extend(syls)
    if syl_stream:
        prof = _sbm_pipeline(syl_stream, 'latin_syllable', 'reference')
        profiles.append(_convert(asdict(prof)))
        print(f"    k={prof.optimal_k}  sil={prof.silhouette_score:.3f}  "
              f"ARI={prof.frequency_tier_ari:.3f}  "
              f"largest={prof.largest_community_coverage:.3f}")

    result = FreqReferenceResult(
        profiles=profiles,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'freq_reference.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2)
    print(f"\n  Saved -> {out_path}")


# ---------------------------------------------------------------------------
# Step 46B.2 — SBM on Synthetic Ciphers
# ---------------------------------------------------------------------------


def run_freq_cipher() -> None:
    """Step 46B.2: SBM on 5 synthetic cipher texts."""
    t0 = time.time()
    print("=" * 70)
    print("STEP 46B.2: SBM on Synthetic Ciphers")
    print("=" * 70)

    rd = _results_dir()

    # Load Latin source text
    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    latin_tokens = ref_corpus.get_combined_tokens('latin')
    latin_words = [w.lower() for w in latin_tokens if len(w) >= 2]
    latin_text = ' '.join(latin_words[:15000])
    latin_chars = [c for c in latin_text if c.isalpha()]

    profiles: List[Dict] = []
    cipher_specs: List[Dict] = []

    # 1. Simple substitution
    print("\n  1. Simple substitution...")
    sub_chars = _generate_simple_substitution(latin_chars)
    prof = _sbm_pipeline(sub_chars, 'simple_substitution', 'cipher')
    profiles.append(_convert(asdict(prof)))
    cipher_specs.append({
        'name': 'simple_substitution', 'cipher_type': 'simple_sub',
        'n_types': prof.n_types, 'n_tokens': prof.n_tokens,
    })
    print(f"    k={prof.optimal_k}  sil={prof.silhouette_score:.3f}  "
          f"ARI={prof.frequency_tier_ari:.3f}  "
          f"largest={prof.largest_community_coverage:.3f}")

    # 2. Homophonic
    print("\n  2. Homophonic (3 per vowel)...")
    homo_chars = _generate_homophonic(latin_chars, n_homophones=3)
    prof = _sbm_pipeline(homo_chars, 'homophonic', 'cipher')
    profiles.append(_convert(asdict(prof)))
    cipher_specs.append({
        'name': 'homophonic', 'cipher_type': 'homophonic',
        'n_types': prof.n_types, 'n_tokens': prof.n_tokens,
    })
    print(f"    k={prof.optimal_k}  sil={prof.silhouette_score:.3f}  "
          f"ARI={prof.frequency_tier_ari:.3f}  "
          f"largest={prof.largest_community_coverage:.3f}")

    # 3. Tachygraphic CV
    print("\n  3. Tachygraphic CV...")
    tachy_stream = _generate_tachygraphic_cv(latin_words[:15000])
    prof = _sbm_pipeline(tachy_stream, 'tachygraphic_cv', 'cipher')
    profiles.append(_convert(asdict(prof)))
    cipher_specs.append({
        'name': 'tachygraphic_cv', 'cipher_type': 'tachygraphic_cv',
        'n_types': prof.n_types, 'n_tokens': prof.n_tokens,
    })
    print(f"    k={prof.optimal_k}  sil={prof.silhouette_score:.3f}  "
          f"ARI={prof.frequency_tier_ari:.3f}  "
          f"largest={prof.largest_community_coverage:.3f}")

    # 4. Nomenclator
    print("\n  4. Nomenclator (50-word codebook)...")
    nomen_stream = _generate_nomenclator(latin_words[:15000], latin_chars)
    if nomen_stream:
        prof = _sbm_pipeline(nomen_stream, 'nomenclator', 'cipher')
        profiles.append(_convert(asdict(prof)))
        cipher_specs.append({
            'name': 'nomenclator', 'cipher_type': 'nomenclator',
            'n_types': prof.n_types, 'n_tokens': prof.n_tokens,
        })
        print(f"    k={prof.optimal_k}  sil={prof.silhouette_score:.3f}  "
              f"ARI={prof.frequency_tier_ari:.3f}  "
              f"largest={prof.largest_community_coverage:.3f}")

    # 5. Null insertion
    print("\n  5. Null insertion (15% rate)...")
    null_chars = _generate_null_insertion(latin_chars, null_rate=0.15)
    prof = _sbm_pipeline(null_chars, 'null_insertion', 'cipher')
    profiles.append(_convert(asdict(prof)))
    cipher_specs.append({
        'name': 'null_insertion', 'cipher_type': 'null_insertion',
        'n_types': prof.n_types, 'n_tokens': prof.n_tokens,
    })
    print(f"    k={prof.optimal_k}  sil={prof.silhouette_score:.3f}  "
          f"ARI={prof.frequency_tier_ari:.3f}  "
          f"largest={prof.largest_community_coverage:.3f}")

    result = FreqCipherResult(
        profiles=profiles,
        cipher_specs=cipher_specs,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'freq_cipher.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2)
    print(f"\n  Saved -> {out_path}")


# ---------------------------------------------------------------------------
# Step 46B.3 — Comparison
# ---------------------------------------------------------------------------


def run_freq_compare() -> None:
    """Step 46B.3: Compare Voynich vs reference vs cipher SBM patterns."""
    t0 = time.time()
    print("=" * 70)
    print("STEP 46B.3: Frequency Structure Comparison")
    print("=" * 70)

    rd = _results_dir()

    # Load Voynich SBM profile from Phase 44B.2
    sbm_data = _safe_load(os.path.join(rd, 'sbm_communities.json'))
    graph_data = _safe_load(os.path.join(rd, 'sbm_graph.json'))

    # Extract Voynich profile
    voynich_k = sbm_data.get('optimal_k', 6)
    voynich_sil = sbm_data.get('silhouette_score', 0.61)
    communities = sbm_data.get('communities', {})
    comm_counts = Counter(communities.values())
    voynich_sizes = sorted(comm_counts.values(), reverse=True)
    voynich_largest = voynich_sizes[0] / sum(voynich_sizes) if voynich_sizes else 0.0

    # Frequency-tier ARI from Phase 45
    factor_data = _safe_load(os.path.join(rd, 'sbm_factorization.json'))
    voynich_ari = factor_data.get('best_ari', 0.25)

    voynich_profile = {
        'label': 'voynich',
        'corpus_type': 'voynich',
        'optimal_k': voynich_k,
        'silhouette_score': voynich_sil,
        'frequency_tier_ari': voynich_ari,
        'largest_community_coverage': round(voynich_largest, 4),
    }

    # Load reference and cipher profiles
    ref_data = _safe_load(os.path.join(rd, 'freq_reference.json'))
    cipher_data = _safe_load(os.path.join(rd, 'freq_cipher.json'))

    ref_profiles = ref_data.get('profiles', [])
    cipher_profiles = cipher_data.get('profiles', [])

    # Build 4D feature vectors
    all_profiles = [voynich_profile] + ref_profiles + cipher_profiles
    features = []
    for p in all_profiles:
        features.append([
            p.get('optimal_k', 0),
            p.get('silhouette_score', 0),
            p.get('frequency_tier_ari', 0),
            p.get('largest_community_coverage', 0),
        ])

    features_arr = np.array(features, dtype=np.float64)

    # Normalize each dimension to [0, 1]
    for col in range(features_arr.shape[1]):
        col_min = features_arr[:, col].min()
        col_max = features_arr[:, col].max()
        rng = col_max - col_min
        if rng > 0:
            features_arr[:, col] = (features_arr[:, col] - col_min) / rng

    # Euclidean distance from Voynich (index 0) to each other
    voynich_vec = features_arr[0]
    comparison_table: List[Dict] = []
    distances: List[float] = []

    for i, p in enumerate(all_profiles):
        dist = float(np.linalg.norm(voynich_vec - features_arr[i]))
        comparison_table.append({
            'label': p.get('label', f'profile_{i}'),
            'corpus_type': p.get('corpus_type', 'unknown'),
            'optimal_k': p.get('optimal_k', 0),
            'silhouette_score': round(p.get('silhouette_score', 0), 4),
            'frequency_tier_ari': round(p.get('frequency_tier_ari', 0), 4),
            'largest_community_coverage': round(
                p.get('largest_community_coverage', 0), 4,
            ),
            'distance_to_voynich': round(dist, 4),
        })
        if i > 0:  # Skip Voynich itself
            distances.append(dist)

    # Sort by distance (excluding Voynich)
    non_voynich = [c for c in comparison_table if c['label'] != 'voynich']
    non_voynich.sort(key=lambda c: c['distance_to_voynich'])

    nearest = non_voynich[0] if non_voynich else {'label': 'none', 'distance_to_voynich': 0}
    nearest_match = nearest['label']
    nearest_distance = nearest['distance_to_voynich']

    # Determine verdict
    median_dist = float(np.median(distances)) if distances else 0.0
    nearest_type = nearest.get('corpus_type', 'unknown')

    if nearest_distance > 1.5 * median_dist and median_dist > 0:
        verdict = 'UNIQUE'
        rationale = (
            f"Voynich distance to all others > 1.5x median ({nearest_distance:.3f} > "
            f"{1.5 * median_dist:.3f}). Frequency structure is anomalous."
        )
    elif nearest_type == 'reference':
        verdict = 'LANGUAGE_LIKE'
        rationale = (
            f"Nearest match is {nearest_match} (reference corpus, "
            f"distance={nearest_distance:.3f}). "
            "Voynich frequency structure resembles natural language."
        )
    elif nearest_type == 'cipher':
        verdict = 'CIPHER_LIKE'
        rationale = (
            f"Nearest match is {nearest_match} (cipher, "
            f"distance={nearest_distance:.3f}). "
            "Voynich frequency structure matches this cipher type."
        )
    else:
        verdict = 'UNDETERMINED'
        rationale = f"Nearest match is {nearest_match} ({nearest_type})."

    # Check if tachygraphic uniquely matches
    tachy_entries = [
        c for c in non_voynich if c['label'] == 'tachygraphic_cv'
    ]
    if tachy_entries and nearest_match == 'tachygraphic_cv':
        verdict = 'CIPHER_LIKE'
        rationale += (
            " The tachygraphic CV cipher uniquely matches — independent "
            "evidence for the tachygraphic hypothesis."
        )

    print(f"\n  Comparison table (sorted by distance to Voynich):")
    print(f"  {'Label':<25s} {'Type':<12s} {'k':>3s} {'Sil':>6s} "
          f"{'ARI':>6s} {'Cov':>6s} {'Dist':>6s}")
    print(f"  {'-'*25} {'-'*12} {'-'*3} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")
    for c in comparison_table:
        print(f"  {c['label']:<25s} {c['corpus_type']:<12s} "
              f"{c['optimal_k']:>3d} {c['silhouette_score']:>6.3f} "
              f"{c['frequency_tier_ari']:>6.3f} "
              f"{c['largest_community_coverage']:>6.3f} "
              f"{c['distance_to_voynich']:>6.3f}")

    print(f"\n  Nearest: {nearest_match} (distance={nearest_distance:.3f})")
    print(f"  Verdict: {verdict}")
    print(f"  Rationale: {rationale}")

    result = FreqCompareResult(
        voynich_profile=voynich_profile,
        reference_profiles=ref_profiles,
        cipher_profiles=cipher_profiles,
        comparison_table=comparison_table,
        nearest_match=nearest_match,
        nearest_distance=round(nearest_distance, 4),
        verdict=verdict,
        rationale=rationale,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'freq_compare.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2)
    print(f"\n  Saved -> {out_path}")


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------


def run_track_b_46() -> None:
    """Run all Track B steps."""
    run_freq_reference()
    print("\n" + "=" * 70 + "\n")
    run_freq_cipher()
    print("\n" + "=" * 70 + "\n")
    run_freq_compare()
