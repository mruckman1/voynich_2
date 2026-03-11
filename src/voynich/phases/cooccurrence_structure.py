"""
Step 43.8 -- Signal Word Co-occurrence and Section Mapping
============================================================
Analyze how signal words relate to each other positionally and how their
patterns map the manuscript's content sections.

Dependency chain:
    results/signal_positions.json     (Step 43.6)
    results/positional_profiles.json  (Step 43.7)
    results/signal_10k.json           (Phase 36.2: decoded tokens)
        -> cooccurrence_structure.json (this step)
"""

import json
import math
import os
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from voynich.core._paths import results_dir as _results_dir


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BEDROCK_WORDS = ['bene', 'codi', 'sero', 'sene', 'de', 'raro', 'dine', 'cola']


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


def _safe_load(path: str) -> Dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


# ---------------------------------------------------------------------------
# K-means
# ---------------------------------------------------------------------------

def _kmeans(X: np.ndarray, k: int, max_iter: int = 50, seed: int = 42):
    """Simple numpy k-means clustering."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    if k > n:
        k = n
    idx = rng.choice(n, k, replace=False)
    centers = X[idx].copy()
    labels = np.zeros(n, dtype=int)
    for _ in range(max_iter):
        dists = np.linalg.norm(X[:, None] - centers[None], axis=2)
        labels = np.argmin(dists, axis=1)
        new_centers = np.array([
            X[labels == i].mean(axis=0) if (labels == i).any() else centers[i]
            for i in range(k)
        ])
        if np.allclose(centers, new_centers):
            break
        centers = new_centers
    return labels, centers


def _silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
    """Compute mean silhouette coefficient."""
    n = X.shape[0]
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2 or len(unique_labels) >= n:
        return -1.0

    silhouettes = np.zeros(n)
    for i in range(n):
        own_mask = labels == labels[i]
        own_mask[i] = False
        if own_mask.sum() == 0:
            silhouettes[i] = 0.0
            continue
        dists_i = np.linalg.norm(X - X[i], axis=1)
        a_i = dists_i[own_mask].mean()

        b_i = float('inf')
        for lbl in unique_labels:
            if lbl == labels[i]:
                continue
            other_mask = labels == lbl
            if other_mask.sum() == 0:
                continue
            b_candidate = dists_i[other_mask].mean()
            if b_candidate < b_i:
                b_i = b_candidate

        if b_i == float('inf'):
            silhouettes[i] = 0.0
        else:
            silhouettes[i] = (b_i - a_i) / max(a_i, b_i, 1e-12)

    return float(np.mean(silhouettes))


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CooccurrenceResult:
    # Pairwise co-occurrence
    cooccurrence_matrix: List[List[int]]   # 8x8 counts within window
    signal_words: List[str]
    window_size: int
    # PMI
    pmi_matrix: List[List[float]]          # 8x8 PMI values
    significant_pairs: List[Dict]          # [{word1, word2, count, pmi}, ...]
    # Folio clustering
    n_clusters: int
    cluster_assignments: Dict[str, int]    # folio -> cluster_id
    cluster_profiles: List[Dict]           # per-cluster mean signal word rates
    cluster_section_overlap: Dict[str, Dict[str, int]]  # cluster -> {section: count}
    # Sequence patterns
    transition_matrix: List[List[float]]   # 8x8 P(next|current)
    recurring_patterns: List[Dict]         # [{pattern: [w1,w2], count, folios}, ...]
    # Section profiles
    section_signal_rates: Dict[str, Dict[str, float]]  # section -> {word: rate}
    # Summary
    strongest_association: Dict
    weakest_association: Dict
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_cooccurrence_structure() -> None:
    """Step 43.8: Signal Word Co-occurrence and Section Mapping."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 43.8: Signal Word Co-occurrence and Section Mapping")
    print("=" * 70)

    rd = _results_dir()
    window_size = 5
    n_words = len(BEDROCK_WORDS)
    word_to_idx = {w: i for i, w in enumerate(BEDROCK_WORDS)}

    # ------------------------------------------------------------------
    # 1. Load inputs
    # ------------------------------------------------------------------
    print("\n  1. Loading inputs ...")

    sig_pos = _safe_load(os.path.join(rd, 'signal_positions.json'))
    pos_prof = _safe_load(os.path.join(rd, 'positional_profiles.json'))
    sig_10k = _safe_load(os.path.join(rd, 'signal_10k.json'))

    if not sig_10k:
        print("  [SKIP] signal_10k.json not found")
        return

    token_decoded = sig_10k.get('token_decoded', [])
    token_folios = sig_10k.get('token_folios', [])
    n_tokens = len(token_decoded)

    print(f"     signal_positions.json: {'loaded' if sig_pos else 'not found (will compute from signal_10k)'}")
    print(f"     positional_profiles.json: {'loaded' if pos_prof else 'not found (optional)'}")
    print(f"     signal_10k.json: {n_tokens} tokens")

    # Role classification from positional_profiles (optional)
    role_classification = pos_prof.get('role_classification', {})
    if role_classification:
        print(f"     Role classifications: {len(role_classification)} words")

    # Build folio -> section mapping from signal_positions if available,
    # otherwise infer from token_folios
    folio_section_map: Dict[str, str] = {}
    if sig_pos:
        folio_heat_map = sig_pos.get('folio_heat_map', {})
        per_section = sig_pos.get('per_section_summary', {})
        # Build folio->section from per_word_summary sections
        # Actually we need folio->section; try to reconstruct from corpus
        pass

    # We need folio -> section.  Infer from folio naming conventions.
    def _folio_to_section(folio: str) -> str:
        """Heuristic section assignment from folio ID."""
        # Standard Voynich folio-to-section mapping
        f = folio.lower().replace('f', '').rstrip('rv')
        try:
            num = int(''.join(c for c in f if c.isdigit()))
        except ValueError:
            return 'unknown'
        if num <= 56:
            return 'herbal_a'
        elif num <= 67:
            return 'pharmaceutical'
        elif num <= 73:
            return 'zodiac'
        elif num <= 84:
            return 'biological'
        elif num <= 86:
            return 'cosmological'
        elif num <= 102:
            return 'herbal_b'
        elif num <= 116:
            return 'stars'
        else:
            return 'unknown'

    # Build folio metadata
    folio_token_counts: Dict[str, int] = Counter()
    folio_sections: Dict[str, str] = {}
    for i in range(n_tokens):
        fol = token_folios[i]
        folio_token_counts[fol] += 1
        if fol not in folio_sections:
            folio_sections[fol] = _folio_to_section(fol)

    # ------------------------------------------------------------------
    # 2. Co-occurrence matrix (8x8) within sliding window
    # ------------------------------------------------------------------
    print("\n  2. Computing co-occurrence matrix (window={}) ...".format(window_size))

    cooc = np.zeros((n_words, n_words), dtype=int)
    # Also track adjacent (within 3) and same-folio rates
    adjacent_cooc = np.zeros((n_words, n_words), dtype=int)
    same_folio_cooc = np.zeros((n_words, n_words), dtype=int)

    # Word counts for PMI
    word_counts = Counter()
    bedrock_set = set(BEDROCK_WORDS)

    for i in range(n_tokens):
        w_i = token_decoded[i]
        if w_i not in bedrock_set:
            continue
        idx_i = word_to_idx[w_i]
        word_counts[w_i] += 1

        fol_i = token_folios[i]

        # Look ahead within window
        for j in range(i + 1, min(i + window_size + 1, n_tokens)):
            w_j = token_decoded[j]
            if w_j not in bedrock_set:
                continue
            idx_j = word_to_idx[w_j]

            cooc[idx_i, idx_j] += 1
            cooc[idx_j, idx_i] += 1

            if j - i <= 3:
                adjacent_cooc[idx_i, idx_j] += 1
                adjacent_cooc[idx_j, idx_i] += 1

            if token_folios[j] == fol_i:
                same_folio_cooc[idx_i, idx_j] += 1
                same_folio_cooc[idx_j, idx_i] += 1

    cooc_list = cooc.tolist()

    total_cooc = int(cooc.sum()) // 2  # each pair counted twice
    print(f"     Total co-occurrence pairs (within window): {total_cooc}")

    # Show top co-occurring pairs
    pair_list = []
    for a in range(n_words):
        for b in range(a + 1, n_words):
            if cooc[a, b] > 0:
                pair_list.append((BEDROCK_WORDS[a], BEDROCK_WORDS[b], int(cooc[a, b])))
    pair_list.sort(key=lambda x: -x[2])
    for w1, w2, cnt in pair_list[:10]:
        print(f"     {w1:6s} - {w2:6s}: {cnt:5d}")

    # ------------------------------------------------------------------
    # 3. PMI matrix
    # ------------------------------------------------------------------
    print("\n  3. Computing PMI matrix ...")

    pmi = np.full((n_words, n_words), float('nan'))

    # Total valid windows for pair probability
    total_windows = max(n_tokens - window_size, 1)

    for a in range(n_words):
        for b in range(n_words):
            if a == b:
                pmi[a, b] = 0.0
                continue
            p_a = word_counts.get(BEDROCK_WORDS[a], 0) / max(n_tokens, 1)
            p_b = word_counts.get(BEDROCK_WORDS[b], 0) / max(n_tokens, 1)
            p_ab = cooc[a, b] / max(total_windows, 1)

            if p_a > 0 and p_b > 0 and p_ab > 0:
                pmi[a, b] = math.log2(p_ab / (p_a * p_b))
            elif p_ab == 0:
                pmi[a, b] = float('-inf')
            else:
                pmi[a, b] = 0.0

    # Replace -inf and nan with None-safe values for JSON
    pmi_list = []
    for row in pmi:
        pmi_row = []
        for v in row:
            if math.isinf(v) or math.isnan(v):
                pmi_row.append(None)
            else:
                pmi_row.append(round(float(v), 4))
        pmi_list.append(pmi_row)

    # ------------------------------------------------------------------
    # 4. Significant pairs (PMI > 0 AND count > 5)
    # ------------------------------------------------------------------
    print("\n  4. Identifying significant pairs ...")

    significant_pairs: List[Dict] = []
    for a in range(n_words):
        for b in range(a + 1, n_words):
            count_ab = int(cooc[a, b])
            pmi_ab = pmi[a, b]
            if count_ab > 5 and not math.isinf(pmi_ab) and not math.isnan(pmi_ab) and pmi_ab > 0:
                adj_count = int(adjacent_cooc[a, b])
                sf_count = int(same_folio_cooc[a, b])
                significant_pairs.append({
                    'word1': BEDROCK_WORDS[a],
                    'word2': BEDROCK_WORDS[b],
                    'count': count_ab,
                    'pmi': round(float(pmi_ab), 4),
                    'adjacent_count': adj_count,
                    'same_folio_count': sf_count,
                    'adjacent_rate': round(adj_count / max(count_ab, 1), 4),
                    'same_folio_rate': round(sf_count / max(count_ab, 1), 4),
                })

    significant_pairs.sort(key=lambda x: -x['pmi'])
    print(f"     {len(significant_pairs)} significant pairs found")
    for sp in significant_pairs[:10]:
        print(f"     {sp['word1']:6s} - {sp['word2']:6s}: "
              f"PMI={sp['pmi']:+.3f}, count={sp['count']}, "
              f"adj_rate={sp['adjacent_rate']:.2f}, "
              f"same_folio_rate={sp['same_folio_rate']:.2f}")

    # ------------------------------------------------------------------
    # 5. Folio clustering by signal word profile
    # ------------------------------------------------------------------
    print("\n  5. Folio clustering by signal word profile ...")

    # Build per-folio signal word count vectors
    folio_signal_counts: Dict[str, np.ndarray] = defaultdict(lambda: np.zeros(n_words))
    for i in range(n_tokens):
        w = token_decoded[i]
        if w in bedrock_set:
            fol = token_folios[i]
            folio_signal_counts[fol][word_to_idx[w]] += 1

    # Only cluster folios with at least 1 signal word occurrence
    folios_with_signal = sorted(
        fol for fol, vec in folio_signal_counts.items()
        if vec.sum() > 0
    )

    if len(folios_with_signal) < 4:
        print("     Too few folios with signal words for clustering")
        n_clusters = 1
        cluster_assignments: Dict[str, int] = {f: 0 for f in folios_with_signal}
        cluster_profiles: List[Dict] = []
        cluster_section_overlap: Dict[str, Dict[str, int]] = {}
    else:
        # Build feature matrix: normalize by folio token count
        X = np.zeros((len(folios_with_signal), n_words))
        for fi, fol in enumerate(folios_with_signal):
            tc = max(folio_token_counts[fol], 1)
            X[fi] = folio_signal_counts[fol] / tc

        # Try k=4..7, pick best silhouette
        best_k = 4
        best_score = -2.0
        best_labels = None

        for k in range(4, min(8, len(folios_with_signal))):
            labels, centers = _kmeans(X, k)
            score = _silhouette_score(X, labels)
            print(f"     k={k}: silhouette={score:.4f}")
            if score > best_score:
                best_score = score
                best_k = k
                best_labels = labels

        print(f"     Best k={best_k} (silhouette={best_score:.4f})")

        n_clusters = best_k
        cluster_assignments = {
            folios_with_signal[i]: int(best_labels[i])
            for i in range(len(folios_with_signal))
        }

        # Cluster profiles: mean signal word rates per cluster
        cluster_profiles = []
        for c in range(best_k):
            mask = best_labels == c
            if mask.sum() == 0:
                cluster_profiles.append({
                    'cluster': c,
                    'n_folios': 0,
                    'mean_rates': {w: 0.0 for w in BEDROCK_WORDS},
                })
                continue
            mean_rates = X[mask].mean(axis=0)
            cluster_profiles.append({
                'cluster': c,
                'n_folios': int(mask.sum()),
                'mean_rates': {
                    BEDROCK_WORDS[j]: round(float(mean_rates[j]), 6)
                    for j in range(n_words)
                },
            })
            rates_str = ', '.join(
                f'{BEDROCK_WORDS[j]}={mean_rates[j]:.4f}'
                for j in range(n_words) if mean_rates[j] > 0.0001
            )
            print(f"     Cluster {c} ({int(mask.sum())} folios): {rates_str}")

        # Cluster-section overlap
        cluster_section_overlap = defaultdict(lambda: defaultdict(int))
        for fol, cid in cluster_assignments.items():
            sec = folio_sections.get(fol, 'unknown')
            cluster_section_overlap[str(cid)][sec] += 1
        cluster_section_overlap = {
            k: dict(v) for k, v in sorted(cluster_section_overlap.items())
        }

        print("\n     Cluster-section overlap:")
        for cid, sec_counts in sorted(cluster_section_overlap.items()):
            parts = ', '.join(f'{s}={c}' for s, c in sorted(sec_counts.items(), key=lambda x: -x[1]))
            print(f"     Cluster {cid}: {parts}")

    # ------------------------------------------------------------------
    # 6. Signal word sequence patterns and transition matrix
    # ------------------------------------------------------------------
    print("\n  6. Computing signal word sequence patterns ...")

    # Group signal word occurrences by folio in order
    folio_sequences: Dict[str, List[str]] = defaultdict(list)
    for i in range(n_tokens):
        w = token_decoded[i]
        if w in bedrock_set:
            folio_sequences[token_folios[i]].append(w)

    # Transition matrix: count P(next_signal | current_signal)
    trans_counts = np.zeros((n_words, n_words), dtype=int)
    for fol, seq in folio_sequences.items():
        for si in range(len(seq) - 1):
            a = word_to_idx[seq[si]]
            b = word_to_idx[seq[si + 1]]
            trans_counts[a, b] += 1

    # Normalize to probabilities
    trans_matrix = np.zeros((n_words, n_words))
    for a in range(n_words):
        row_sum = trans_counts[a].sum()
        if row_sum > 0:
            trans_matrix[a] = trans_counts[a] / row_sum
    trans_matrix_list = [
        [round(float(v), 4) for v in row]
        for row in trans_matrix
    ]

    # Show top transitions
    print("     Top transitions (P(next|current)):")
    trans_flat = []
    for a in range(n_words):
        for b in range(n_words):
            if trans_matrix[a, b] > 0.05:
                trans_flat.append((BEDROCK_WORDS[a], BEDROCK_WORDS[b],
                                   float(trans_matrix[a, b]), int(trans_counts[a, b])))
    trans_flat.sort(key=lambda x: -x[2])
    for w1, w2, prob, cnt in trans_flat[:12]:
        print(f"     {w1:6s} -> {w2:6s}: P={prob:.3f} (n={cnt})")

    # Recurring bigram patterns across folios
    bigram_folio_map: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for fol, seq in folio_sequences.items():
        if len(seq) < 4:
            continue
        seen_on_folio = set()
        for si in range(len(seq) - 1):
            bg = (seq[si], seq[si + 1])
            if bg not in seen_on_folio:
                bigram_folio_map[bg].append(fol)
                seen_on_folio.add(bg)

    recurring_patterns: List[Dict] = []
    for (w1, w2), folios_list in sorted(bigram_folio_map.items(), key=lambda x: -len(x[1])):
        if len(folios_list) >= 5:
            recurring_patterns.append({
                'pattern': [w1, w2],
                'count': len(folios_list),
                'folios': folios_list[:20],  # cap for JSON size
            })

    recurring_patterns.sort(key=lambda x: -x['count'])
    print(f"\n     {len(recurring_patterns)} recurring bigram patterns (on 5+ folios)")
    for rp in recurring_patterns[:10]:
        print(f"     {rp['pattern'][0]:6s} -> {rp['pattern'][1]:6s}: "
              f"{rp['count']} folios")

    # ------------------------------------------------------------------
    # 7. Section-specific signal profiles
    # ------------------------------------------------------------------
    print("\n  7. Computing section-specific signal profiles ...")

    section_token_totals: Dict[str, int] = Counter()
    section_word_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for i in range(n_tokens):
        fol = token_folios[i]
        sec = folio_sections.get(fol, 'unknown')
        section_token_totals[sec] += 1
        w = token_decoded[i]
        if w in bedrock_set:
            section_word_counts[sec][w] += 1

    section_signal_rates: Dict[str, Dict[str, float]] = {}
    for sec in sorted(section_token_totals.keys()):
        total = section_token_totals[sec]
        rates = {}
        for w in BEDROCK_WORDS:
            cnt = section_word_counts[sec].get(w, 0)
            rates[w] = round(cnt / max(total, 1), 6)
        section_signal_rates[sec] = rates

    # Print section profiles
    print(f"     {'Section':15s} {'tokens':>7s}  " +
          '  '.join(f'{w:>6s}' for w in BEDROCK_WORDS))
    for sec in sorted(section_signal_rates.keys()):
        total = section_token_totals[sec]
        rates = section_signal_rates[sec]
        rates_str = '  '.join(
            f'{rates[w]*1000:6.2f}' for w in BEDROCK_WORDS
        )
        print(f"     {sec:15s} {total:7d}  {rates_str}  (per 1000 tokens)")

    # Identify elevated/depressed rates per word
    print("\n     Elevated/depressed signals by section:")
    global_rates = {}
    for w in BEDROCK_WORDS:
        total_w = sum(section_word_counts[sec].get(w, 0) for sec in section_token_totals)
        global_rates[w] = total_w / max(n_tokens, 1)

    for w in BEDROCK_WORDS:
        gr = global_rates[w]
        if gr == 0:
            continue
        elevated = []
        depressed = []
        for sec in sorted(section_signal_rates.keys()):
            sr = section_signal_rates[sec][w]
            if sr > gr * 1.5:
                elevated.append(f"{sec}({sr/gr:.1f}x)")
            elif sr < gr * 0.5 and section_token_totals[sec] > 100:
                depressed.append(f"{sec}({sr/gr:.1f}x)")
        if elevated or depressed:
            parts = []
            if elevated:
                parts.append("UP: " + ', '.join(elevated))
            if depressed:
                parts.append("DOWN: " + ', '.join(depressed))
            print(f"     {w:8s}: {'; '.join(parts)}")

    # ------------------------------------------------------------------
    # 8. Summary: strongest and weakest associations
    # ------------------------------------------------------------------
    print("\n  8. Summary ...")

    strongest = {'word1': '', 'word2': '', 'pmi': 0.0, 'count': 0}
    weakest = {'word1': '', 'word2': '', 'pmi': 0.0, 'count': 0}

    if significant_pairs:
        strongest = {
            'word1': significant_pairs[0]['word1'],
            'word2': significant_pairs[0]['word2'],
            'pmi': significant_pairs[0]['pmi'],
            'count': significant_pairs[0]['count'],
        }
        # Weakest among significant pairs
        weakest_sp = significant_pairs[-1]
        weakest = {
            'word1': weakest_sp['word1'],
            'word2': weakest_sp['word2'],
            'pmi': weakest_sp['pmi'],
            'count': weakest_sp['count'],
        }
    else:
        # Fall back: find pair with most co-occurrences even if not significant
        best_count = 0
        for a in range(n_words):
            for b in range(a + 1, n_words):
                if int(cooc[a, b]) > best_count:
                    best_count = int(cooc[a, b])
                    pmi_val = pmi[a, b]
                    if math.isinf(pmi_val) or math.isnan(pmi_val):
                        pmi_val = 0.0
                    strongest = {
                        'word1': BEDROCK_WORDS[a],
                        'word2': BEDROCK_WORDS[b],
                        'pmi': round(float(pmi_val), 4),
                        'count': best_count,
                    }

    runtime = round(time.time() - t0, 2)

    print(f"\n     Strongest association: {strongest['word1']} - {strongest['word2']} "
          f"(PMI={strongest['pmi']}, count={strongest['count']})")
    print(f"     Weakest association:   {weakest['word1']} - {weakest['word2']} "
          f"(PMI={weakest['pmi']}, count={weakest['count']})")
    print(f"     Clusters: {n_clusters}")
    print(f"     Recurring signal bigram patterns: {len(recurring_patterns)}")
    print(f"     Sections with profiles: {len(section_signal_rates)}")
    print(f"     Runtime: {runtime}s")

    # ------------------------------------------------------------------
    # 9. Save
    # ------------------------------------------------------------------
    result = CooccurrenceResult(
        cooccurrence_matrix=cooc_list,
        signal_words=list(BEDROCK_WORDS),
        window_size=window_size,
        pmi_matrix=pmi_list,
        significant_pairs=significant_pairs,
        n_clusters=n_clusters,
        cluster_assignments=cluster_assignments,
        cluster_profiles=cluster_profiles,
        cluster_section_overlap=cluster_section_overlap,
        transition_matrix=trans_matrix_list,
        recurring_patterns=recurring_patterns,
        section_signal_rates=section_signal_rates,
        strongest_association=strongest,
        weakest_association=weakest,
        runtime_seconds=runtime,
    )

    out_path = os.path.join(rd, 'cooccurrence_structure.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)
    print(f"\n  -> {out_path}")
