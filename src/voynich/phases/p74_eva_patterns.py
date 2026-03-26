"""
Phase 74, Track B1: EVA-Level Pattern Expansion
================================================
Expand the vocabulary of identified words WITHOUT character-level decode,
using distributional patterns at the EVA token level.

Two sub-analyses:
  A. Distributional identifications: unidentified EVA types that appear
     in the same contexts as T1-identified types probably encode related
     meanings (distributional synonymy / morphological variants).
  B. Positional identifications: unidentified types that appear exclusively
     in the same manuscript positions as T1 types (recipe-initial,
     paragraph-initial, etc.) are likely the same functional class.

Dependency chain:
    results/p69_clean_corpus.json        (Phase 69 — T1 catalogue)
    results/combined_refine.json         (Phase 15 — assignment table)
        -> results/p74_patterns.json
"""

import json
import math
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import load_corpus, tokenize_eva_chars


# ---------------------------------------------------------------------------
# JSON helpers
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
    if isinstance(obj, set):
        return sorted(obj)
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
# Cosine similarity between Counter vectors
# ---------------------------------------------------------------------------

def _cosine_similarity_counters(a: Counter, b: Counter) -> float:
    """Compute cosine similarity between two Counter (sparse vector) objects."""
    keys = set(a.keys()) | set(b.keys())
    if not keys:
        return 0.0

    dot = sum(a.get(k, 0) * b.get(k, 0) for k in keys)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# A. Distributional identifications
# ---------------------------------------------------------------------------

def _build_context_vectors(
    all_tokens: List[str],
    folio_boundaries: List[int],
    window: int = 3,
) -> Dict[str, Counter]:
    """Build context vectors (bag of neighboring token types) for each EVA
    token type, respecting folio boundaries."""
    vectors: Dict[str, Counter] = {}

    # Build boundary set for quick lookup
    boundary_set = set(folio_boundaries)

    for i, token in enumerate(all_tokens):
        if token not in vectors:
            vectors[token] = Counter()

        for j in range(max(0, i - window), min(len(all_tokens), i + window + 1)):
            if j == i:
                continue
            # Don't cross folio boundaries
            if any(b > min(i, j) and b <= max(i, j) for b in boundary_set
                   if abs(b - i) <= window):
                continue
            vectors[token][all_tokens[j]] += 1

    return vectors


def _find_distributional_identifications(
    context_vectors: Dict[str, Counter],
    t1_types: Dict[str, Dict],
    type_counts: Counter,
    min_freq: int = 5,
    min_sim: float = 0.30,
) -> List[Dict[str, Any]]:
    """For each unidentified token type (freq >= min_freq), find the most
    similar T1-identified type by context vector cosine similarity."""
    t1_set = set(t1_types.keys())
    frequent_unidentified = [t for t, c in type_counts.items()
                             if c >= min_freq and t not in t1_set]

    identifications = []

    for unid_type in frequent_unidentified:
        unid_vec = context_vectors.get(unid_type)
        if not unid_vec:
            continue

        best_match = None
        best_sim = 0.0

        for id_type in t1_set:
            id_vec = context_vectors.get(id_type)
            if not id_vec:
                continue

            sim = _cosine_similarity_counters(unid_vec, id_vec)
            if sim > best_sim:
                best_sim = sim
                best_match = id_type

        if best_match and best_sim > min_sim:
            t1_info = t1_types[best_match]
            identifications.append({
                'eva_type': unid_type,
                'frequency': type_counts[unid_type],
                'matched_t1_type': best_match,
                'matched_word': t1_info.get('matched_word', '?'),
                'similarity': round(best_sim, 4),
                'source': 'distributional',
            })

    return sorted(identifications, key=lambda x: -x['similarity'])


# ---------------------------------------------------------------------------
# B. Positional identifications
# ---------------------------------------------------------------------------

def _find_positional_identifications(
    corpus,
    all_tokens: List[str],
    t1_types: Dict[str, Dict],
    type_counts: Counter,
) -> List[Dict[str, Any]]:
    """Identify unidentified token types that appear in the same manuscript
    positions as T1 types (paragraph-initial, line-initial, recipe section)."""
    t1_set = set(t1_types.keys())

    # Build position annotations for each token
    positions_by_type: Dict[str, Counter] = {}  # token_type -> Counter of positions

    token_idx = 0
    for folio_id, page in sorted(corpus.pages.items()):
        section = page.section
        for locus in page.loci:
            if not locus.clean_text:
                continue

            locus_tokens = locus.clean_text.split()
            locus_type = locus.locus_type  # P=paragraph, L=label, C=circular

            for local_pos, tok in enumerate(locus_tokens):
                pos_labels = []

                # Line initial
                if local_pos == 0:
                    pos_labels.append('line_initial')

                # Paragraph initial (first token of a P-type locus)
                if local_pos == 0 and locus_type.startswith('P'):
                    pos_labels.append('para_initial')

                # Recipe section
                if section in ('recipes', 'pharmaceutical'):
                    pos_labels.append(f'section_{section}')
                    if local_pos == 0 and locus_type.startswith('P'):
                        pos_labels.append('recipe_initial')

                if tok not in positions_by_type:
                    positions_by_type[tok] = Counter()
                for label in pos_labels:
                    positions_by_type[tok][label] += 1

    # Find unidentified types that concentrate at specific positions
    identifications = []

    for position_type in ['recipe_initial', 'para_initial', 'line_initial']:
        # Which T1 types appear at this position?
        t1_at_pos = {t: positions_by_type.get(t, Counter()).get(position_type, 0)
                     for t in t1_set}
        t1_at_pos = {t: c for t, c in t1_at_pos.items() if c >= 2}

        if not t1_at_pos:
            continue

        # Most common T1 type at this position
        most_common_t1 = max(t1_at_pos, key=t1_at_pos.get)
        t1_info = t1_types[most_common_t1]

        # Which unidentified types also appear at this position?
        for unid_type in type_counts:
            if unid_type in t1_set:
                continue
            pos_count = positions_by_type.get(unid_type, Counter()).get(
                position_type, 0)
            total_count = type_counts[unid_type]

            # Must appear ≥3 times at this position, and ≥30% of total usage
            if pos_count >= 3 and total_count > 0 and pos_count / total_count >= 0.30:
                identifications.append({
                    'eva_type': unid_type,
                    'frequency': total_count,
                    'position': position_type,
                    'frequency_at_position': pos_count,
                    'position_fraction': round(pos_count / total_count, 3),
                    'likely_class': t1_info.get('matched_word', '?'),
                    'reference_t1': most_common_t1,
                    'reference_word': t1_info.get('matched_word', '?'),
                    'source': 'positional',
                })

    # Deduplicate (same type may match multiple positions)
    seen = set()
    deduped = []
    for ident in identifications:
        key = ident['eva_type']
        if key not in seen:
            seen.add(key)
            deduped.append(ident)

    return sorted(deduped, key=lambda x: -x['frequency_at_position'])


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class EVAPatternResult:
    phase: str = "74"
    step: str = "74.B1"
    experiment: str = "eva_pattern_expansion"
    # Distributional
    n_distributional: int = 0
    distributional_ids: List[Dict[str, Any]] = field(default_factory=list)
    mean_similarity: float = 0.0
    # Positional
    n_positional: int = 0
    positional_ids: List[Dict[str, Any]] = field(default_factory=list)
    # Combined
    n_total_new: int = 0
    total_identified_types: int = 0  # T1 + new
    # Gates
    gate_b1_1: bool = False   # ≥50 distributional identifications
    gate_b1_2: bool = False   # ≥10 positional identifications
    gates_passed: int = 0
    gate_passed: bool = False
    verdict: str = "UNKNOWN"
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_eva_patterns():
    """Track B1: EVA-level distributional and positional expansion."""
    t0 = time.time()
    rd = str(_results_dir())

    print("Phase 74.B1 — EVA-Level Pattern Expansion")
    print("=" * 42)

    # --- Load T1 catalogue ---
    print("  Loading T1 catalogue...")
    clean_data = _safe_load(os.path.join(rd, 'p69_clean_corpus.json'))
    t1_catalogue = clean_data.get('t1_catalogue', [])

    # Build T1 lookup: eva_type -> info
    t1_types: Dict[str, Dict] = {}
    for entry in t1_catalogue:
        eva_type = entry.get('eva_type', '')
        if eva_type:
            t1_types[eva_type] = entry

    print(f"  T1 types: {len(t1_types)}")

    # --- Load corpus ---
    print("  Loading corpus...")
    corpus = load_corpus(verbose=False)
    all_tokens = corpus.get_tokens()
    type_counts = Counter(all_tokens)

    print(f"  Total tokens: {len(all_tokens)}, unique types: {len(type_counts)}")
    print(f"  Unidentified types (freq≥5): "
          f"{sum(1 for t, c in type_counts.items() if c >= 5 and t not in t1_types)}")

    # --- Build folio boundaries for context vectors ---
    folio_boundaries = []
    idx = 0
    for folio_id, page in sorted(corpus.pages.items()):
        page_tokens = page.all_tokens
        idx += len(page_tokens)
        folio_boundaries.append(idx)

    # =====================================================================
    # A. Distributional identifications
    # =====================================================================
    print("\n  A. Building context vectors (window=3)...")
    context_vectors = _build_context_vectors(all_tokens, folio_boundaries, window=3)
    print(f"    Context vectors for {len(context_vectors)} types")

    print("  Finding distributional matches...")
    distrib_ids = _find_distributional_identifications(
        context_vectors, t1_types, type_counts, min_freq=5, min_sim=0.30)

    print(f"    Found {len(distrib_ids)} distributional identifications")
    if distrib_ids:
        mean_sim = float(np.mean([d['similarity'] for d in distrib_ids]))
        print(f"    Mean similarity: {mean_sim:.3f}")
        print(f"    Top 10:")
        for d in distrib_ids[:10]:
            print(f"      {d['eva_type']} → {d['matched_word']} "
                  f"(sim={d['similarity']:.3f}, freq={d['frequency']}, "
                  f"via {d['matched_t1_type']})")
    else:
        mean_sim = 0.0

    # =====================================================================
    # B. Positional identifications
    # =====================================================================
    print("\n  B. Finding positional identifications...")
    positional_ids = _find_positional_identifications(
        corpus, all_tokens, t1_types, type_counts)

    print(f"    Found {len(positional_ids)} positional identifications")
    for p in positional_ids[:10]:
        print(f"      {p['eva_type']} at {p['position']} "
              f"({p['frequency_at_position']}× of {p['frequency']})")

    # =====================================================================
    # Combined summary
    # =====================================================================
    # Combine and deduplicate
    all_new_types = set()
    for d in distrib_ids:
        all_new_types.add(d['eva_type'])
    for p in positional_ids:
        all_new_types.add(p['eva_type'])

    n_total_new = len(all_new_types)
    total_identified = len(t1_types) + n_total_new

    print(f"\n  Combined: {n_total_new} new types identified")
    print(f"  Total identified types: {total_identified} "
          f"(T1: {len(t1_types)} + new: {n_total_new})")

    # =====================================================================
    # Gates
    # =====================================================================
    g1 = len(distrib_ids) >= 50
    g2 = len(positional_ids) >= 10

    gates_passed = sum([g1, g2])

    print(f"\n  Gates:")
    print(f"    B1_1 (≥50 distributional): {'PASS' if g1 else 'FAIL'} "
          f"({len(distrib_ids)})")
    print(f"    B1_2 (≥10 positional): {'PASS' if g2 else 'FAIL'} "
          f"({len(positional_ids)})")
    print(f"    Total: {gates_passed}/2")

    # --- Verdict ---
    if g1 and g2:
        verdict = 'PATTERNS_FOUND'
    elif g1 or g2:
        verdict = 'PARTIAL_PATTERNS'
    else:
        verdict = 'NO_PATTERNS'

    print(f"\n  Verdict: {verdict}")

    # --- Build result ---
    # Truncate lists for JSON size
    result = EVAPatternResult(
        n_distributional=len(distrib_ids),
        distributional_ids=distrib_ids[:100],  # Top 100
        mean_similarity=round(mean_sim, 4),
        n_positional=len(positional_ids),
        positional_ids=positional_ids[:50],
        n_total_new=n_total_new,
        total_identified_types=total_identified,
        gate_b1_1=g1,
        gate_b1_2=g2,
        gates_passed=gates_passed,
        gate_passed=gates_passed >= 1,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 1),
    )

    path = _save_json(rd, 'p74_patterns.json', asdict(result))
    print(f"\n  Saved: {path}")
    print(f"  Runtime: {result.runtime_seconds:.1f}s")

    return result
