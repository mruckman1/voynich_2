"""
Phase 62, Investigation 6: Double-Modifier Sequences
=====================================================
Catalog consecutive modifier-modifier pairs within tokens, map each pair
to a coda consonant cluster, and compare the distribution to Latin coda
cluster frequencies.

Dependency chain:
    results/combined_refine.json      (Phase 15)
    results/modifier_integrate.json   (Phase 16)
        -> results/phase62_double_modifier.json
"""

import json
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy import stats as sp_stats

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import load_corpus, tokenize_eva_chars
from voynich.core.reference import EVA_VISUAL_COMPONENTS
from voynich.phases.corrected_coda import (
    build_coda_table_v2,
    classify_token_chars_v2,
)
from voynich.phases.coda_markers import get_coda


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
# Legal Latin coda clusters
# ---------------------------------------------------------------------------

LEGAL_CODA_CLUSTERS = frozenset({
    'ns', 'nt', 'rs', 'rt', 'st', 'rn', 'rm', 'mn',
    'lt', 'ls', 'lm', 'nr', 'ms', 'mt', 'ts', 'tr',
    'rl', 'sl', 'nl', 'tl', 'sn', 'sm',
    'ln', 'lr', 'sr', 'tn',
})

# Latin reference coda cluster frequencies (from pharmaceutical texts)
# Approximate ranked list
LATIN_CLUSTER_RANK = [
    'ns', 'nt', 'rs', 'rt', 'st', 'rn', 'rm', 'lt',
    'mn', 'ls', 'lm', 'ts', 'ms', 'mt', 'nr', 'tr',
]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DoubleModifierResult:
    phase: str = "62"
    step: str = "62.6"
    experiment: str = "double_modifier"
    n_tokens_scanned: int = 0
    n_double_pairs: int = 0
    n_unique_clusters: int = 0
    cluster_distribution: Dict[str, int] = field(default_factory=dict)
    n_legal: int = 0
    n_illegal: int = 0
    attestation_rate: float = 0.0
    top_pairs: List[Dict] = field(default_factory=list)
    top_5_all_legal: bool = False
    rank_correlation: float = 0.0
    # Example tokens per cluster
    cluster_examples: Dict[str, List[str]] = field(default_factory=dict)
    # Gates
    g1_enough_data: bool = False       # >= 500 double-modifier sequences
    g2_top5_legal: bool = False        # top 5 clusters all legal Latin
    g3_rank_corr: bool = False         # rank correlation > 0.3
    gates_passed: int = 0
    gate_passed: bool = False
    verdict: str = ""
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def run_double_modifier():
    """Phase 62.6: Double-modifier sequence analysis."""
    t0 = time.time()
    rd = str(_results_dir())
    print("=" * 70)
    print("Phase 62, Investigation 6: Double-Modifier Sequences")
    print("=" * 70)

    # Load
    coda_table = build_coda_table_v2()
    corpus = load_corpus(verbose=False)
    all_tokens = corpus.get_tokens()

    print(f"  Tokens: {len(all_tokens)}")

    # Scan for double-modifier pairs
    cluster_counts = Counter()
    cluster_examples = {}  # cluster -> [example tokens]

    for token in all_tokens:
        eva_chars = tokenize_eva_chars(token)
        if len(eva_chars) < 2:
            continue

        classified = classify_token_chars_v2(eva_chars, coda_table)

        for i in range(len(classified) - 1):
            role1, char1 = classified[i]
            role2, char2 = classified[i + 1]

            if role1 == 'CODA_MARKER' and role2 == 'CODA_MARKER':
                coda1 = get_coda(char1, coda_table)
                coda2 = get_coda(char2, coda_table)
                if coda1 and coda2:
                    cluster = coda1 + coda2
                    cluster_counts[cluster] += 1
                    if cluster not in cluster_examples:
                        cluster_examples[cluster] = []
                    if len(cluster_examples[cluster]) < 5:
                        cluster_examples[cluster].append(token)

    n_double = sum(cluster_counts.values())
    n_unique = len(cluster_counts)

    # Legal/illegal
    n_legal = sum(c for cl, c in cluster_counts.items() if cl in LEGAL_CODA_CLUSTERS)
    n_illegal = sum(c for cl, c in cluster_counts.items() if cl not in LEGAL_CODA_CLUSTERS)
    att_rate = n_legal / n_double if n_double > 0 else 0.0

    # Top 5
    top5 = cluster_counts.most_common(5)
    top5_all_legal = all(cl in LEGAL_CODA_CLUSTERS for cl, _ in top5) if top5 else False

    # Build top_pairs details
    top_pairs = []
    for cl, count in cluster_counts.most_common(20):
        top_pairs.append({
            'cluster': cl,
            'count': count,
            'legal': cl in LEGAL_CODA_CLUSTERS,
            'examples': cluster_examples.get(cl, [])[:3],
        })

    # Rank correlation with Latin reference
    # Rank Voynich clusters and Latin clusters
    voynich_rank = [cl for cl, _ in cluster_counts.most_common() if cl in LEGAL_CODA_CLUSTERS]
    # Build ranks for common set
    common = [cl for cl in LATIN_CLUSTER_RANK if cl in set(voynich_rank)]
    if len(common) >= 4:
        v_ranks = [voynich_rank.index(cl) for cl in common]
        l_ranks = [LATIN_CLUSTER_RANK.index(cl) for cl in common]
        rank_corr = float(sp_stats.spearmanr(v_ranks, l_ranks).statistic)
        if np.isnan(rank_corr):
            rank_corr = 0.0
    else:
        rank_corr = 0.0

    # Gates
    g1 = n_double >= 500
    g2 = top5_all_legal
    g3 = rank_corr > 0.3
    gates_passed = sum([g1, g2, g3])

    if g1 and g2:
        verdict = "DOUBLE_CODAS_VALID"
    elif g1 and att_rate > 0.5:
        verdict = "PARTIALLY_VALID"
    else:
        verdict = "CLUSTERS_NOT_LATIN"

    result = DoubleModifierResult(
        n_tokens_scanned=len(all_tokens),
        n_double_pairs=n_double,
        n_unique_clusters=n_unique,
        cluster_distribution=dict(cluster_counts.most_common()),
        n_legal=n_legal,
        n_illegal=n_illegal,
        attestation_rate=att_rate,
        top_pairs=top_pairs,
        top_5_all_legal=top5_all_legal,
        rank_correlation=round(rank_corr, 4),
        cluster_examples={k: v[:3] for k, v in cluster_examples.items()},
        g1_enough_data=g1,
        g2_top5_legal=g2,
        g3_rank_corr=g3,
        gates_passed=gates_passed,
        gate_passed=gates_passed >= 2,
        verdict=verdict,
        runtime_seconds=time.time() - t0,
    )

    # Print summary
    print(f"\n  Double-modifier pairs: {n_double}")
    print(f"  Unique clusters: {n_unique}")
    print(f"  Legal: {n_legal} ({att_rate:.1%})  Illegal: {n_illegal}")
    print(f"  Top 5 all legal: {top5_all_legal}")
    print(f"  Top clusters:")
    for p in top_pairs[:10]:
        tag = "LEGAL" if p['legal'] else "ILLEGAL"
        print(f"    {p['cluster']:4s} {p['count']:5d} ({tag})  ex: {p['examples']}")
    print(f"  Rank correlation with Latin: {rank_corr:.3f}")
    print(f"\n  Gates: G1={'PASS' if g1 else 'FAIL'} G2={'PASS' if g2 else 'FAIL'} "
          f"G3={'PASS' if g3 else 'FAIL'} ({gates_passed}/3)")
    print(f"  Verdict: {verdict}")

    path = _save_json(rd, 'phase62_double_modifier.json', asdict(result))
    print(f"\n  Saved: {path}")
    print(f"  Runtime: {result.runtime_seconds:.1f}s")
    return result
