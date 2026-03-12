"""
Phase 47 Track C – Structural Reading and Content Analysis
============================================================
Extract structural content from GREEN+YELLOW tokens (35.5% of corpus)
without changing the table.  N-grams, recipe grammar, topic clustering,
star folio readings, section vocabulary differentiation.

Dependency chain:
    signal_bigrams.json         (Phase 29 parallel arrays)
    signal_10k.json             (Phase 36 signal words at 10K)
    canonical_table.json        (Phase 45 tier annotations)
    modifier_integrate.json     (Phase 16 modifiers)
    complete_lexicon.json       (Phase 41/46 73-word lexicon)
    structural_reading.json     (Phase 43 recipe folios)
    final_annotations.json      (Phase 46 GREEN/YELLOW rates)
        -> read_ngrams.json     (Step 47C.1)
        -> read_recipes.json    (Step 47C.2)
        -> read_topics.json     (Step 47C.3)
        -> read_star_folios.json (Step 47C.4)
        -> read_sections.json   (Step 47C.5)
"""

from __future__ import annotations

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
# Per-token tier classification (replicates final_decode.py logic)
# ---------------------------------------------------------------------------

def _load_tier_context(rd: str) -> Tuple[
    Dict[str, str], set, Dict[str, str], set, set, set,
]:
    """Load everything needed for per-token tier classification.

    Returns (tier_annotations, modifier_chars, modifier_rules,
             signal_word_set, ref_word_set_10k, ref_word_set_131k).
    """
    canon = _safe_load(os.path.join(rd, 'canonical_table.json'))
    tier_annotations = canon.get('tier_annotations', {})

    mod_data = _safe_load(os.path.join(rd, 'modifier_integrate.json'))
    modifier_chars = set(mod_data.get('modifier_chars', []))
    modifier_rules: Dict[str, str] = {}
    for c in mod_data.get('classifications', []):
        if c.get('final_classification') == 'modifier':
            modifier_rules[c['eva_char']] = c.get('modifier_type', 'silent')

    sig_data = _safe_load(os.path.join(rd, 'signal_10k.json'))
    signal_word_set = {
        w['word'] for w in sig_data.get('word_signals', [])
        if w.get('is_genuine_signal')
    }

    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    ref_tokens_raw = [
        w.lower() for w in ref_corpus.get_combined_tokens('latin')
        if len(w) >= 2
    ]
    base_words = set(ref_tokens_raw)
    expanded, _ = build_expanded_word_set(base_words)
    ref_word_set_131k = base_words | expanded

    word_freq = Counter(ref_tokens_raw)
    ref_word_set_10k = {w for w, _ in word_freq.most_common(10000)}

    return (tier_annotations, modifier_chars, modifier_rules,
            signal_word_set, ref_word_set_10k, ref_word_set_131k)


def _classify_per_token(
    token_evas: List[str],
    tier_annotations: Dict[str, str],
    modifier_chars: set,
    modifier_rules: Dict[str, str],
    eva_to_triple: Dict[str, str],
    signal_word_set: set,
    ref_word_set_10k: set,
    ref_word_set_131k: set,
    assignment: Dict[str, str],
) -> List[str]:
    """Return per-token tier: GREEN/YELLOW/ORANGE/RED.

    Replicates final_decode.py lines 557-598.
    """
    tiers: List[str] = []
    for token in token_evas:
        chars = tokenize_eva_chars(token)
        triples_used = []
        for ch in chars:
            if ch not in modifier_chars:
                t = eva_to_triple.get(ch)
                if t:
                    triples_used.append(t)

        triple_tiers = [tier_annotations.get(t, 'UNKNOWN') for t in triples_used]
        all_tier1 = all(t == 'CONFIRMED' for t in triple_tiers) if triple_tiers else False
        all_tier12 = all(
            t in ('CONFIRMED', 'LANDSCAPE_CONFIRMED') for t in triple_tiers
        ) if triple_tiers else False
        has_tier3 = any(t == 'GENUINELY_AMBIGUOUS' for t in triple_tiers)

        alt = decode_token_modifier_aware(
            token, assignment, eva_to_triple, modifier_chars, modifier_rules,
        )
        decoded_word = alt.lower()

        in_10k = decoded_word in ref_word_set_10k
        in_131k = decoded_word in ref_word_set_131k
        is_signal = decoded_word in signal_word_set

        if all_tier1 and in_10k and is_signal:
            tiers.append('GREEN')
        elif all_tier12 and in_131k:
            tiers.append('YELLOW')
        elif (all_tier12 or has_tier3) and in_131k:
            tiers.append('ORANGE')
        else:
            tiers.append('RED')

    return tiers


def _folio_to_section(folio: str) -> str:
    """Heuristic section assignment from folio name."""
    corpus = load_corpus(verbose=False)
    page = corpus.pages.get(folio)
    if page and hasattr(page, 'section'):
        return page.section
    return 'unknown'


def _load_all_folio_sections() -> Dict[str, str]:
    """Return folio -> section for all folios."""
    corpus = load_corpus(verbose=False)
    return {
        folio: (page.section if hasattr(page, 'section') else 'unknown')
        for folio, page in corpus.pages.items()
    }


# ---------------------------------------------------------------------------
# Step 47C.1 — Repeated n-grams
# ---------------------------------------------------------------------------

@dataclass
class NgramResult:
    n_total_tokens: int
    n_green_yellow: int
    green_yellow_rate: float
    ngrams_by_n: Dict[int, List[Dict]]
    n_filtered_ngrams: int
    top_ngrams: List[Dict]
    runtime_seconds: float


def run_read_ngrams() -> None:
    """Step 47C.1: search for repeated n-grams in GREEN+YELLOW tokens."""
    t0 = time.time()
    print("=" * 70)
    print("STEP 47C.1: Repeated Multi-Word Sequences")
    print("=" * 70)

    rd = _results_dir()

    # Load parallel arrays
    sb = _safe_load(os.path.join(rd, 'signal_bigrams.json'))
    token_folios = sb.get('token_folios', [])
    token_evas = sb.get('token_evas', [])
    token_decoded = sb.get('token_decoded', [])
    token_classifications = sb.get('token_classifications', [])
    n_tokens = len(token_decoded)

    if n_tokens == 0:
        print("  [SKIP] No token data in signal_bigrams.json")
        return

    # Load tier context and classify
    print("\n  Classifying per-token tiers...")
    (tier_annotations, modifier_chars, modifier_rules,
     signal_word_set, ref_10k, ref_131k) = _load_tier_context(rd)
    eva_to_triple = build_eva_to_triple_lookup()
    assignment = _safe_load(os.path.join(rd, 'combined_refine.json')).get(
        'best_assignment', {},
    )
    token_tiers = _classify_per_token(
        token_evas, tier_annotations, modifier_chars, modifier_rules,
        eva_to_triple, signal_word_set, ref_10k, ref_131k, assignment,
    )

    n_gy = sum(1 for t in token_tiers if t in ('GREEN', 'YELLOW'))
    print(f"  {n_gy} GREEN+YELLOW tokens ({n_gy/n_tokens:.1%})")

    # Group tokens by folio
    folio_groups: Dict[str, List[int]] = defaultdict(list)
    for i, f in enumerate(token_folios):
        folio_groups[f].append(i)

    # Find n-grams
    print("\n  Searching for repeated n-grams (n=2..7)...")
    ngram_counts: Dict[int, Counter] = {n: Counter() for n in range(2, 8)}
    ngram_folios: Dict[int, Dict[Tuple, List[str]]] = {n: defaultdict(list) for n in range(2, 8)}
    ngram_signal_count: Dict[int, Dict[Tuple, int]] = {n: defaultdict(int) for n in range(2, 8)}

    for folio, indices in folio_groups.items():
        for n in range(2, 8):
            for start in range(len(indices) - n + 1):
                window = indices[start:start + n]
                # All must be GREEN or YELLOW
                if all(token_tiers[i] in ('GREEN', 'YELLOW') for i in window):
                    words = tuple(token_decoded[i] for i in window)
                    ngram_counts[n][words] += 1
                    ngram_folios[n][words].append(folio)
                    # Count SIGNAL tokens in this n-gram
                    n_sig = sum(1 for i in window if token_classifications[i] == 'SIGNAL')
                    ngram_signal_count[n][words] = max(
                        ngram_signal_count[n][words], n_sig,
                    )

    # Filter: count >= 3, signal >= 2
    ngrams_by_n: Dict[int, List[Dict]] = {}
    all_filtered = []
    for n in range(2, 8):
        filtered = []
        for words, count in ngram_counts[n].most_common():
            if count >= 3 and ngram_signal_count[n][words] >= 2:
                entry = {
                    'ngram': ' '.join(words),
                    'n': n,
                    'count': count,
                    'folios': sorted(set(ngram_folios[n][words])),
                    'n_folios': len(set(ngram_folios[n][words])),
                    'signal_tokens': ngram_signal_count[n][words],
                }
                filtered.append(entry)
                all_filtered.append(entry)
        ngrams_by_n[n] = filtered[:20]

    all_filtered.sort(key=lambda x: (-x['count'], -x['n']))
    top_ngrams = all_filtered[:20]

    n_filtered = len(all_filtered)
    print(f"\n  Found {n_filtered} recurring n-grams (count>=3, signal>=2)")
    for n in range(2, 8):
        print(f"    n={n}: {len(ngrams_by_n.get(n, []))} n-grams")

    if top_ngrams:
        print("\n  Top 10:")
        for ng in top_ngrams[:10]:
            print(f"    [{ng['count']}x, {ng['n_folios']} folios] {ng['ngram']}")

    result = NgramResult(
        n_total_tokens=n_tokens,
        n_green_yellow=n_gy,
        green_yellow_rate=round(n_gy / n_tokens, 4) if n_tokens else 0.0,
        ngrams_by_n={k: v for k, v in ngrams_by_n.items()},
        n_filtered_ngrams=n_filtered,
        top_ngrams=top_ngrams,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'read_ngrams.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2)
    print(f"\n  Saved -> {out_path}")


# ---------------------------------------------------------------------------
# Step 47C.2 — Recipe grammar extraction
# ---------------------------------------------------------------------------

@dataclass
class RecipeResult:
    recipe_folios: List[str]
    n_recipes: int
    recipes: List[Dict]
    grammar_positions: Dict[int, List[Tuple[str, int]]]
    runtime_seconds: float


def run_read_recipes() -> None:
    """Step 47C.2: recipe grammar extraction on recipe folios."""
    t0 = time.time()
    print("=" * 70)
    print("STEP 47C.2: Recipe Grammar Extraction")
    print("=" * 70)

    rd = _results_dir()

    # Load recipe folios from Phase 43
    sr = _safe_load(os.path.join(rd, 'structural_reading.json'))
    recipe_folios = sr.get('recipe_folios', [])
    if not recipe_folios:
        # Fallback: identify from folio classifications
        folio_types = sr.get('folio_classifications', {})
        recipe_folios = [f for f, t in folio_types.items() if t == 'recipe']
    if not recipe_folios:
        recipe_folios = []

    # Load parallel arrays and lexicon
    sb = _safe_load(os.path.join(rd, 'signal_bigrams.json'))
    token_folios = sb.get('token_folios', [])
    token_decoded = sb.get('token_decoded', [])
    token_classifications = sb.get('token_classifications', [])
    lexicon = _safe_load(os.path.join(rd, 'complete_lexicon.json'))
    lex_entries = lexicon.get('complete_lexicon', {})

    # Group tokens by folio
    folio_tokens: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for i in range(len(token_folios)):
        folio_tokens[token_folios[i]].append(
            (token_decoded[i], token_classifications[i]),
        )

    # Boundary markers
    BOUNDARY_WORDS = {'cola', 'codi'}

    recipes = []
    for folio in recipe_folios:
        tokens = folio_tokens.get(folio, [])
        if not tokens:
            continue

        # Find boundary positions
        boundaries = [i for i, (w, _) in enumerate(tokens) if w in BOUNDARY_WORDS]
        if not boundaries:
            # Treat entire folio as one recipe
            boundaries = [0]

        # Extract recipes between boundaries
        for b_idx in range(len(boundaries)):
            start = boundaries[b_idx]
            end = boundaries[b_idx + 1] if b_idx + 1 < len(boundaries) else len(tokens)
            recipe_tokens = tokens[start:end]

            decoded_words = [w for w, _ in recipe_tokens]
            signal_words = [w for w, c in recipe_tokens if c == 'SIGNAL']

            # POS tagging from lexicon
            pos_tags = []
            for w, _ in recipe_tokens:
                entry = lex_entries.get(w, {})
                pos_tags.append(entry.get('pos', 'unknown'))

            recipes.append({
                'folio': folio,
                'start': start,
                'end': end,
                'length': len(recipe_tokens),
                'decoded': decoded_words,
                'signal_words': signal_words,
                'pos_tags': pos_tags,
                'boundary_word': decoded_words[0] if decoded_words else '',
            })

    # Build grammar template: positional distributions
    grammar_positions: Dict[int, Counter] = defaultdict(Counter)
    for recipe in recipes:
        for pos, (word, _) in enumerate(zip(recipe['decoded'], recipe.get('pos_tags', []))):
            if pos < 10:
                grammar_positions[pos][word] += 1

    grammar_summary = {
        pos: counter.most_common(5)
        for pos, counter in sorted(grammar_positions.items())
    }

    print(f"\n  Recipe folios: {len(recipe_folios)}")
    print(f"  Recipes extracted: {len(recipes)}")
    if recipes:
        lengths = [r['length'] for r in recipes]
        print(f"  Mean recipe length: {sum(lengths)/len(lengths):.1f} tokens")
        print(f"\n  Grammar (positional word frequency):")
        for pos, words in sorted(grammar_summary.items())[:5]:
            top = ', '.join(f'{w}({c})' for w, c in words[:3])
            print(f"    Position {pos}: {top}")

    result = RecipeResult(
        recipe_folios=recipe_folios,
        n_recipes=len(recipes),
        recipes=recipes[:100],  # cap for JSON size
        grammar_positions=grammar_summary,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'read_recipes.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2)
    print(f"\n  Saved -> {out_path}")


# ---------------------------------------------------------------------------
# Step 47C.3 — Folio-level topic clustering
# ---------------------------------------------------------------------------

@dataclass
class TopicResult:
    n_folios: int
    n_signal_words: int
    optimal_k: int
    silhouette_score: float
    clusters: List[Dict]
    top_cooccurrences: List[Dict]
    section_alignment: Dict[str, Dict[str, int]]
    runtime_seconds: float


def _kmeans_numpy(X: np.ndarray, k: int, max_iter: int = 100, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Simple K-means. Returns (labels, centroids)."""
    rng = np.random.RandomState(seed)
    n = X.shape[0]
    idx = rng.choice(n, size=k, replace=False)
    centroids = X[idx].copy()

    for _ in range(max_iter):
        # Assign
        dists = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
        labels = np.argmin(dists, axis=1)
        # Update
        new_centroids = np.zeros_like(centroids)
        for j in range(k):
            members = X[labels == j]
            if len(members) > 0:
                new_centroids[j] = members.mean(axis=0)
            else:
                new_centroids[j] = centroids[j]
        if np.allclose(new_centroids, centroids):
            break
        centroids = new_centroids
    return labels, centroids


def _silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
    """Simplified silhouette score."""
    n = len(labels)
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return 0.0

    sil = np.zeros(n)
    for i in range(n):
        own_cluster = labels[i]
        own_mask = labels == own_cluster
        other_labels = [l for l in unique_labels if l != own_cluster]

        if own_mask.sum() <= 1:
            sil[i] = 0.0
            continue

        a_i = np.mean(np.linalg.norm(X[own_mask] - X[i], axis=1))
        b_i = float('inf')
        for ol in other_labels:
            other_mask = labels == ol
            if other_mask.sum() > 0:
                d = np.mean(np.linalg.norm(X[other_mask] - X[i], axis=1))
                b_i = min(b_i, d)

        if max(a_i, b_i) > 0:
            sil[i] = (b_i - a_i) / max(a_i, b_i)

    return float(np.mean(sil))


def run_read_topics() -> None:
    """Step 47C.3: folio-level topic clustering from signal words."""
    t0 = time.time()
    print("=" * 70)
    print("STEP 47C.3: Folio-Level Topic Clustering")
    print("=" * 70)

    rd = _results_dir()

    # Load signal words
    sig_data = _safe_load(os.path.join(rd, 'signal_10k.json'))
    signal_words = [
        w['word'] for w in sig_data.get('word_signals', [])
        if w.get('is_genuine_signal')
    ]
    if not signal_words:
        print("  [SKIP] No signal words found")
        return
    signal_word_idx = {w: i for i, w in enumerate(signal_words)}
    n_sw = len(signal_words)

    # Load parallel arrays
    sb = _safe_load(os.path.join(rd, 'signal_bigrams.json'))
    token_folios = sb.get('token_folios', [])
    token_decoded = sb.get('token_decoded', [])

    # Get folio ordering and sections
    folio_sections = _load_all_folio_sections()
    folio_order = list(dict.fromkeys(token_folios))  # preserve insertion order
    n_folios = len(folio_order)

    # Build feature matrix: n_folios x n_signal_words
    folio_token_count: Counter = Counter(token_folios)
    folio_sw_counts: Dict[str, Counter] = defaultdict(Counter)
    for i in range(len(token_folios)):
        w = token_decoded[i]
        if w in signal_word_idx:
            folio_sw_counts[token_folios[i]][w] += 1

    X = np.zeros((n_folios, n_sw))
    for fi, folio in enumerate(folio_order):
        total = folio_token_count[folio] or 1
        for sw, count in folio_sw_counts[folio].items():
            X[fi, signal_word_idx[sw]] = count / total

    # K-means for k=2..8
    print(f"\n  Matrix: {n_folios} folios x {n_sw} signal words")
    best_k = 2
    best_sil = -1.0
    best_labels = None

    for k in range(2, 9):
        labels, centroids = _kmeans_numpy(X, k, seed=42)
        sil = _silhouette_score(X, labels)
        print(f"    k={k}: silhouette={sil:.4f}")
        if sil > best_sil:
            best_sil = sil
            best_k = k
            best_labels = labels

    print(f"\n  Optimal k={best_k} (silhouette={best_sil:.4f})")

    # Build cluster summaries
    clusters = []
    for cid in range(best_k):
        mask = best_labels == cid
        cluster_folios = [folio_order[i] for i in range(n_folios) if mask[i]]
        # Section composition
        section_comp: Counter = Counter()
        for f in cluster_folios:
            section_comp[folio_sections.get(f, 'unknown')] += 1

        # Top signal words (by mean frequency in cluster)
        mean_freqs = X[mask].mean(axis=0) if mask.sum() > 0 else np.zeros(n_sw)
        top_sw_idx = np.argsort(-mean_freqs)[:10]
        top_sw = [(signal_words[i], round(float(mean_freqs[i]), 6)) for i in top_sw_idx if mean_freqs[i] > 0]

        clusters.append({
            'cluster_id': cid,
            'n_folios': int(mask.sum()),
            'folios': cluster_folios,
            'section_composition': dict(section_comp),
            'top_signal_words': top_sw,
        })

    # Co-occurrence PMI (top pairs)
    print("\n  Computing signal word co-occurrence PMI...")
    word_folio_sets: Dict[str, set] = defaultdict(set)
    for i in range(len(token_folios)):
        w = token_decoded[i]
        if w in signal_word_idx:
            word_folio_sets[w].add(token_folios[i])

    cooc_pairs = []
    for i, w1 in enumerate(signal_words):
        for j, w2 in enumerate(signal_words):
            if j <= i:
                continue
            s1 = word_folio_sets.get(w1, set())
            s2 = word_folio_sets.get(w2, set())
            joint = len(s1 & s2)
            if joint < 3:
                continue
            p_joint = joint / n_folios
            p1 = len(s1) / n_folios
            p2 = len(s2) / n_folios
            if p1 > 0 and p2 > 0 and p_joint > 0:
                pmi = math.log2(p_joint / (p1 * p2))
                cooc_pairs.append({
                    'word1': w1, 'word2': w2,
                    'pmi': round(pmi, 4),
                    'joint_folios': joint,
                })

    cooc_pairs.sort(key=lambda x: -x['pmi'])
    top_cooc = cooc_pairs[:20]

    if top_cooc:
        print("  Top co-occurring pairs:")
        for p in top_cooc[:5]:
            print(f"    {p['word1']:8s} + {p['word2']:8s}  PMI={p['pmi']:.2f}  joint={p['joint_folios']}")

    # Section alignment
    section_alignment: Dict[str, Dict[str, int]] = {}
    for section in set(folio_sections.values()):
        section_alignment[section] = {}
        for cid in range(best_k):
            count = sum(1 for f in clusters[cid]['folios'] if folio_sections.get(f) == section)
            section_alignment[section][str(cid)] = count

    result = TopicResult(
        n_folios=n_folios,
        n_signal_words=n_sw,
        optimal_k=best_k,
        silhouette_score=round(best_sil, 4),
        clusters=clusters,
        top_cooccurrences=top_cooc,
        section_alignment=section_alignment,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'read_topics.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2)
    print(f"\n  Saved -> {out_path}")


# ---------------------------------------------------------------------------
# Step 47C.4 — Star folio readings
# ---------------------------------------------------------------------------

@dataclass
class StarFolioResult:
    star_folios: List[Dict]
    total_gloss_attempts: int
    runtime_seconds: float


def run_read_star() -> None:
    """Step 47C.4: detailed readings of top folios."""
    t0 = time.time()
    print("=" * 70)
    print("STEP 47C.4: Star Folio Readings")
    print("=" * 70)

    rd = _results_dir()

    # Load parallel arrays
    sb = _safe_load(os.path.join(rd, 'signal_bigrams.json'))
    token_folios = sb.get('token_folios', [])
    token_evas = sb.get('token_evas', [])
    token_decoded = sb.get('token_decoded', [])
    token_classifications = sb.get('token_classifications', [])

    if not token_folios:
        print("  [SKIP] No data")
        return

    # Load tier context
    (tier_annotations, modifier_chars, modifier_rules,
     signal_word_set, ref_10k, ref_131k) = _load_tier_context(rd)
    eva_to_triple = build_eva_to_triple_lookup()
    assignment = _safe_load(os.path.join(rd, 'combined_refine.json')).get(
        'best_assignment', {},
    )
    token_tiers = _classify_per_token(
        token_evas, tier_annotations, modifier_chars, modifier_rules,
        eva_to_triple, signal_word_set, ref_10k, ref_131k, assignment,
    )

    # Load lexicon
    lexicon = _safe_load(os.path.join(rd, 'complete_lexicon.json'))
    lex_entries = lexicon.get('complete_lexicon', {})

    # Group by folio
    folio_indices: Dict[str, List[int]] = defaultdict(list)
    for i, f in enumerate(token_folios):
        folio_indices[f].append(i)

    # Compute GREEN rate per folio (min 50 tokens)
    folio_green_rate = {}
    for folio, indices in folio_indices.items():
        if len(indices) < 50:
            continue
        n_green = sum(1 for i in indices if token_tiers[i] == 'GREEN')
        folio_green_rate[folio] = n_green / len(indices)

    # Top 5 by GREEN rate
    top_folios = sorted(folio_green_rate, key=folio_green_rate.get, reverse=True)[:5]
    print(f"\n  Top 5 folios by GREEN rate:")
    for f in top_folios:
        print(f"    {f}: {folio_green_rate[f]:.1%} GREEN ({len(folio_indices[f])} tokens)")

    # Detailed readings
    star_readings = []
    total_glosses = 0

    for folio in top_folios:
        indices = folio_indices[folio]
        n_tokens = len(indices)

        # Token annotations
        annotations = []
        for i in indices:
            entry = lex_entries.get(token_decoded[i], {})
            annotations.append({
                'eva': token_evas[i],
                'decoded': token_decoded[i],
                'tier': token_tiers[i],
                'dict_hit': token_decoded[i] in ref_131k,
                'signal': token_classifications[i],
                'lexicon_pos': entry.get('pos', ''),
                'lexicon_gloss': entry.get('english_gloss', ''),
            })

        # Find GREEN+YELLOW runs
        runs = []
        current_run_start = None
        current_run_len = 0
        for j, i in enumerate(indices):
            if token_tiers[i] in ('GREEN', 'YELLOW'):
                if current_run_start is None:
                    current_run_start = j
                    current_run_len = 1
                else:
                    current_run_len += 1
            else:
                if current_run_start is not None and current_run_len >= 3:
                    run_indices = indices[current_run_start:current_run_start + current_run_len]
                    runs.append({
                        'start': current_run_start,
                        'length': current_run_len,
                        'text': ' '.join(token_decoded[i] for i in run_indices),
                    })
                current_run_start = None
                current_run_len = 0

        if current_run_start is not None and current_run_len >= 3:
            run_indices = indices[current_run_start:current_run_start + current_run_len]
            runs.append({
                'start': current_run_start,
                'length': current_run_len,
                'text': ' '.join(token_decoded[i] for i in run_indices),
            })

        longest_run = max((r['length'] for r in runs), default=0)

        # Gloss attempts for runs >= 3
        glosses = []
        for run in runs:
            words = run['text'].split()
            pos_tags = [lex_entries.get(w, {}).get('pos', '?') for w in words]
            english = [lex_entries.get(w, {}).get('english_gloss', w) for w in words]
            glosses.append({
                'fragment': run['text'],
                'length': run['length'],
                'pos_sequence': ' '.join(pos_tags),
                'gloss': ' '.join(english),
            })
            total_glosses += 1

        # Tier distribution
        tier_dist = Counter(token_tiers[i] for i in indices)

        folio_sections = _load_all_folio_sections()

        star_readings.append({
            'folio': folio,
            'section': folio_sections.get(folio, 'unknown'),
            'n_tokens': n_tokens,
            'green_rate': round(tier_dist.get('GREEN', 0) / n_tokens, 4),
            'yellow_rate': round(tier_dist.get('YELLOW', 0) / n_tokens, 4),
            'red_rate': round(tier_dist.get('RED', 0) / n_tokens, 4),
            'longest_gy_run': longest_run,
            'n_runs_gte3': len(runs),
            'runs': runs[:10],
            'glosses': glosses[:10],
            'annotations': annotations[:50],  # first 50 tokens
        })

    result = StarFolioResult(
        star_folios=star_readings,
        total_gloss_attempts=total_glosses,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'read_star_folios.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2)
    print(f"\n  Saved -> {out_path}")


# ---------------------------------------------------------------------------
# Step 47C.5 — Section-level vocabulary differentiation
# ---------------------------------------------------------------------------

@dataclass
class SectionVocabResult:
    sections: List[Dict]
    pairwise_jsd: Dict[str, float]
    section_specific_words: Dict[str, List[str]]
    mean_jsd: float
    runtime_seconds: float


def _jsd(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence."""
    p = p / (p.sum() + 1e-12)
    q = q / (q.sum() + 1e-12)
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log2(p / (m + 1e-12) + 1e-12))
    kl_qm = np.sum(q * np.log2(q / (m + 1e-12) + 1e-12))
    return float(0.5 * kl_pm + 0.5 * kl_qm)


def run_read_sections() -> None:
    """Step 47C.5: section-level vocabulary differentiation."""
    t0 = time.time()
    print("=" * 70)
    print("STEP 47C.5: Section-Level Vocabulary Differentiation")
    print("=" * 70)

    rd = _results_dir()

    # Load parallel arrays
    sb = _safe_load(os.path.join(rd, 'signal_bigrams.json'))
    token_folios = sb.get('token_folios', [])
    token_decoded = sb.get('token_decoded', [])
    n_tokens = len(token_decoded)

    if n_tokens == 0:
        print("  [SKIP] No data")
        return

    folio_sections = _load_all_folio_sections()

    # Group decoded words by section
    section_words: Dict[str, List[str]] = defaultdict(list)
    for i in range(n_tokens):
        folio = token_folios[i]
        section = folio_sections.get(folio, 'unknown')
        section_words[section].append(token_decoded[i])

    section_names = sorted(section_words.keys())
    print(f"\n  Sections: {len(section_names)}")

    # Per-section stats
    sections_summary = []
    section_counters: Dict[str, Counter] = {}
    for section in section_names:
        words = section_words[section]
        counter = Counter(words)
        section_counters[section] = counter
        top_30 = counter.most_common(30)
        sections_summary.append({
            'section': section,
            'n_tokens': len(words),
            'n_unique': len(counter),
            'top_words': [{'word': w, 'count': c} for w, c in top_30],
        })
        print(f"    {section:16s}: {len(words):6d} tokens, {len(counter):5d} unique")

    # Section-specific words: >= 5 in this section, < 2 total elsewhere
    section_specific: Dict[str, List[str]] = {}
    for section in section_names:
        specific = []
        for word, count in section_counters[section].items():
            if count < 5:
                continue
            other_count = sum(
                section_counters[s].get(word, 0)
                for s in section_names if s != section
            )
            if other_count < 2:
                specific.append(word)
        section_specific[section] = sorted(specific)

    for section, words in section_specific.items():
        if words:
            print(f"    {section} specific: {', '.join(words[:5])}")

    # Build vocabulary for JSD
    all_words = sorted(set(token_decoded))
    word_idx = {w: i for i, w in enumerate(all_words)}
    V = len(all_words)

    section_vecs: Dict[str, np.ndarray] = {}
    for section in section_names:
        vec = np.zeros(V)
        for word, count in section_counters[section].items():
            vec[word_idx[word]] = count
        section_vecs[section] = vec

    # Pairwise JSD
    print("\n  Pairwise JSD:")
    pairwise_jsd: Dict[str, float] = {}
    jsd_values = []
    for i, s1 in enumerate(section_names):
        for j, s2 in enumerate(section_names):
            if j <= i:
                continue
            j_val = _jsd(section_vecs[s1], section_vecs[s2])
            key = f"{s1}_vs_{s2}"
            pairwise_jsd[key] = round(j_val, 6)
            jsd_values.append(j_val)
            print(f"    {s1:16s} vs {s2:16s}: JSD={j_val:.4f}")

    mean_jsd = sum(jsd_values) / len(jsd_values) if jsd_values else 0.0
    print(f"\n  Mean JSD: {mean_jsd:.4f}")

    result = SectionVocabResult(
        sections=sections_summary,
        pairwise_jsd=pairwise_jsd,
        section_specific_words=section_specific,
        mean_jsd=round(mean_jsd, 6),
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'read_sections.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2)
    print(f"\n  Saved -> {out_path}")


# ---------------------------------------------------------------------------
# Track C orchestrator
# ---------------------------------------------------------------------------

def run_track_c_47() -> None:
    """Run all Track C steps."""
    run_read_ngrams()
    print()
    run_read_recipes()
    print()
    run_read_topics()
    print()
    run_read_star()
    print()
    run_read_sections()
