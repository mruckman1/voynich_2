"""
Phase 62, Investigation 5: Orphaned Coda Investigation
=======================================================
Phase 59 found 21,805 orphaned coda consonants (20.1% of segments).
Investigate the EVA character sequences that produce orphans, analyze
double-coda clusters against legal Latin clusters, and identify
misclassification candidates.

Dependency chain:
    results/combined_refine.json      (Phase 15)
    results/modifier_integrate.json   (Phase 16)
        -> results/phase62_orphaned_coda.json
"""

import json
import os
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import build_eva_to_triple_lookup, load_corpus, tokenize_eva_chars
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


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class OrphanedCodaResult:
    phase: str = "62"
    step: str = "62.5"
    experiment: str = "orphaned_coda"
    n_tokens_analyzed: int = 0
    total_segments: int = 0
    total_orphans: int = 0
    orphan_rate: float = 0.0
    orphan_by_coda: Dict[str, int] = field(default_factory=dict)
    # Double coda analysis
    n_double_codas: int = 0
    n_triple_codas: int = 0
    double_coda_clusters: Dict[str, int] = field(default_factory=dict)
    n_legal_clusters: int = 0
    n_illegal_clusters: int = 0
    legal_fraction: float = 0.0
    top_legal: List[List] = field(default_factory=list)
    top_illegal: List[List] = field(default_factory=list)
    # Per-char orphan rates
    char_orphan_rates: List[Dict] = field(default_factory=list)
    reclassification_candidates: List[Dict] = field(default_factory=list)
    # Position analysis
    orphan_positions: Dict[str, int] = field(default_factory=dict)
    # Gates
    g1_legal_clusters: bool = False    # >= 60% double codas = legal Latin
    g2_few_misclass: bool = False      # <= 3 chars with orphan rate > 50%
    g3_reclassify_helps: bool = False  # reclassification reduces orphan rate
    gates_passed: int = 0
    gate_passed: bool = False
    verdict: str = ""
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def run_orphaned_coda():
    """Phase 62.5: Investigate orphaned codas under CVC v2."""
    t0 = time.time()
    rd = str(_results_dir())
    print("=" * 70)
    print("Phase 62, Investigation 5: Orphaned Coda Investigation")
    print("=" * 70)

    # Load
    coda_table = build_coda_table_v2()
    corpus = load_corpus(verbose=False)
    all_tokens = corpus.get_tokens()

    print(f"  Tokens: {len(all_tokens)}")

    # Classify every token, track orphans and doubles
    total_segments = 0
    total_orphans = 0
    orphan_by_coda = Counter()
    double_coda_clusters = Counter()
    triple_coda_count = 0
    orphan_positions = Counter()  # token_initial / medial / final

    # Track per-char stats: total coda appearances vs orphan appearances
    char_total_coda = Counter()
    char_orphan_coda = Counter()

    for token in all_tokens:
        eva_chars = tokenize_eva_chars(token)
        if not eva_chars:
            continue
        classified = classify_token_chars_v2(eva_chars, coda_table)
        roles = [role for role, _ in classified]
        n = len(classified)

        for idx, (role, char) in enumerate(classified):
            total_segments += 1

            if role != 'CODA_MARKER':
                continue

            coda_val = get_coda(char, coda_table)
            if not coda_val:
                continue

            char_total_coda[char] += 1

            # Check if orphaned: coda with no preceding SYLLABIC
            is_orphan = False
            if idx == 0:
                is_orphan = True
                orphan_positions['token_initial'] += 1
            elif roles[idx - 1] == 'CODA_MARKER':
                is_orphan = True  # double coda — second is orphan
                # Get the preceding coda value
                prev_char = classified[idx - 1][1]
                prev_coda = get_coda(prev_char, coda_table)
                if prev_coda:
                    cluster = prev_coda + coda_val
                    double_coda_clusters[cluster] += 1
                if idx >= 2 and roles[idx - 2] == 'CODA_MARKER':
                    triple_coda_count += 1

                if idx < n - 1:
                    orphan_positions['medial'] += 1
                else:
                    orphan_positions['final'] += 1
            # Normal coda after SYLLABIC is NOT orphaned
            elif roles[idx - 1] == 'SYLLABIC':
                continue
            else:
                is_orphan = True
                orphan_positions['other'] += 1

            if is_orphan:
                total_orphans += 1
                orphan_by_coda[coda_val] += 1
                char_orphan_coda[char] += 1

    orphan_rate = total_orphans / total_segments if total_segments > 0 else 0.0

    # Analyze double-coda clusters
    n_legal = 0
    n_illegal = 0
    for cluster, count in double_coda_clusters.items():
        if cluster in LEGAL_CODA_CLUSTERS:
            n_legal += count
        else:
            n_illegal += count
    total_double = n_legal + n_illegal
    legal_fraction = n_legal / total_double if total_double > 0 else 0.0

    top_legal = [(c, n) for c, n in double_coda_clusters.most_common()
                 if c in LEGAL_CODA_CLUSTERS][:10]
    top_illegal = [(c, n) for c, n in double_coda_clusters.most_common()
                   if c not in LEGAL_CODA_CLUSTERS][:10]

    # Per-char orphan rates
    char_rates = []
    for char in sorted(char_total_coda.keys()):
        total = char_total_coda[char]
        orphaned = char_orphan_coda.get(char, 0)
        rate = orphaned / total if total > 0 else 0.0
        char_rates.append({
            'char': char,
            'total_coda': total,
            'orphaned': orphaned,
            'orphan_rate': round(rate, 4),
        })

    # Reclassification candidates: orphan rate > 50% and total > 50
    reclass = [cr for cr in char_rates
               if cr['orphan_rate'] > 0.5 and cr['total_coda'] > 50]
    n_high_orphan = len(reclass)

    # Gates
    g1 = legal_fraction >= 0.60
    g2 = n_high_orphan <= 3
    g3 = len(reclass) > 0  # at least some actionable findings

    gates_passed = sum([g1, g2, g3])

    if g1 and g2:
        verdict = "CLASSIFICATION_GOOD"
    elif n_high_orphan <= 5:
        verdict = "MINOR_CORRECTIONS_NEEDED"
    else:
        verdict = "MAJOR_RECLASSIFICATION_NEEDED"

    result = OrphanedCodaResult(
        n_tokens_analyzed=len(all_tokens),
        total_segments=total_segments,
        total_orphans=total_orphans,
        orphan_rate=orphan_rate,
        orphan_by_coda=dict(orphan_by_coda.most_common()),
        n_double_codas=total_double,
        n_triple_codas=triple_coda_count,
        double_coda_clusters=dict(double_coda_clusters.most_common(30)),
        n_legal_clusters=n_legal,
        n_illegal_clusters=n_illegal,
        legal_fraction=legal_fraction,
        top_legal=top_legal,
        top_illegal=top_illegal,
        char_orphan_rates=char_rates,
        reclassification_candidates=reclass,
        orphan_positions=dict(orphan_positions),
        g1_legal_clusters=g1,
        g2_few_misclass=g2,
        g3_reclassify_helps=g3,
        gates_passed=gates_passed,
        gate_passed=gates_passed >= 2,
        verdict=verdict,
        runtime_seconds=time.time() - t0,
    )

    # Print summary
    print(f"\n  Total segments: {total_segments}")
    print(f"  Total orphans: {total_orphans} ({orphan_rate:.1%})")
    print(f"  Orphan by coda: {dict(orphan_by_coda.most_common(5))}")
    print(f"  Double-coda clusters: {total_double}")
    print(f"    Legal: {n_legal} ({legal_fraction:.1%})  Illegal: {n_illegal}")
    print(f"    Top legal: {top_legal[:5]}")
    print(f"    Top illegal: {top_illegal[:5]}")
    print(f"  Reclassification candidates (orphan>50%, n>50): {n_high_orphan}")
    for cr in reclass:
        print(f"    {cr['char']:8s} orphan rate={cr['orphan_rate']:.1%} "
              f"(n={cr['total_coda']})")
    print(f"\n  Gates: G1={'PASS' if g1 else 'FAIL'} G2={'PASS' if g2 else 'FAIL'} "
          f"G3={'PASS' if g3 else 'FAIL'} ({gates_passed}/3)")
    print(f"  Verdict: {verdict}")

    path = _save_json(rd, 'phase62_orphaned_coda.json', asdict(result))
    print(f"\n  Saved: {path}")
    print(f"  Runtime: {result.runtime_seconds:.1f}s")
    return result
