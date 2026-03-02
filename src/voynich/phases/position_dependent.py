"""
Phase 9.3 — Position-Dependent Encoding Test
=============================================

Rationale
---------
In a polyalphabetic cipher the mapping changes based on position.  The same
token means different things at different positions within a line.  If the
Voynich uses position-dependent encoding, co-occurrence patterns should differ
systematically across positions.

Sub-analyses
------------
9.3a  Split corpus by token position within lines, compare bigram matrices (JSD)
9.3b  Token identity test: same token at different positions
9.3c  Reference comparison (Latin, German)
Null  Randomly shuffle token positions within each line
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import load_corpus
from voynich.core.reference import load_reference_corpus
from voynich.core.stats import (
    bootstrap_ci,
    cosine_similarity,
    jensen_shannon_divergence,
    selectivity_ratio,
    word_transition_matrix,
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PositionalBigramComparison:
    position_a: str
    position_b: str
    jsd: float
    n_tokens_a: int
    n_tokens_b: int


@dataclass
class TokenIdentityTest:
    n_tokens_tested: int
    n_position_dependent: int
    fraction_position_dependent: float
    mean_cosine_across_positions: float
    top_position_dependent: List[Tuple[str, float]]
    top_position_stable: List[Tuple[str, float]]


@dataclass
class ReferencePositionalJSD:
    language: str
    initial_medial_jsd: float
    initial_final_jsd: float
    medial_final_jsd: float
    mean_jsd: float


@dataclass
class PositionDependentResult:
    voynich_positional_jsds: List[Dict]
    voynich_mean_jsd: float
    token_identity_test: Dict
    reference_jsds: List[Dict]
    voynich_vs_reference_ratio: float
    null_shuffled_jsds: List[float]
    null_mean_jsd: float
    position_selectivity: float
    gate_jsd: bool
    gate_selectivity: bool
    gate_passed: bool
    verdict: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _convert(obj: Any) -> Any:
    """Convert dataclass/numpy types to JSON-serializable form."""
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
    if isinstance(obj, (bool, int, float, str, type(None))):
        return obj
    return str(obj)


def _get_lines(corpus, language: str = 'A') -> List[List[str]]:
    """Extract lines of tokens from the corpus (one list per line/paragraph)."""
    pages = corpus.get_pages_by_language(language)
    lines: List[List[str]] = []
    for page in pages:
        for locus in page.loci:
            if locus.locus_type not in ('P', 'L'):
                continue
            text = locus.clean_text
            tokens = text.split()
            if tokens:
                lines.append(tokens)
    return lines


def _split_by_position(
    lines: List[List[str]],
) -> Dict[str, List[str]]:
    """
    Split tokens into three positional groups based on their position within
    each line: initial (first third), medial (middle third), final (last third).
    """
    initial: List[str] = []
    medial: List[str] = []
    final: List[str] = []

    for line in lines:
        n = len(line)
        if n == 0:
            continue
        if n == 1:
            initial.append(line[0])
            continue
        if n == 2:
            initial.append(line[0])
            final.append(line[1])
            continue
        # Split into thirds
        t1 = max(1, n // 3)
        t2 = max(t1 + 1, 2 * n // 3)
        initial.extend(line[:t1])
        medial.extend(line[t1:t2])
        final.extend(line[t2:])

    return {'initial': initial, 'medial': medial, 'final': final}


def _build_transition_dist(tokens: List[str], top_n: int = 200) -> np.ndarray:
    """Build a flattened bigram probability distribution from tokens."""
    if len(tokens) < 10:
        return np.array([1.0])
    mat, vocab = word_transition_matrix(tokens, top_n=top_n)
    flat = mat.flatten()
    total = flat.sum()
    if total > 0:
        flat = flat / total
    return flat


def _compute_jsd_between_positions(
    pos_tokens: Dict[str, List[str]], top_n: int = 200,
) -> List[PositionalBigramComparison]:
    """Compute JSD between all pairs of positional groups."""
    positions = ['initial', 'medial', 'final']
    dists: Dict[str, np.ndarray] = {}
    sizes: Dict[str, int] = {}

    # Build a shared vocabulary across all positions
    all_tokens = []
    for p in positions:
        all_tokens.extend(pos_tokens.get(p, []))
    counts = Counter(all_tokens)
    shared_vocab = [tok for tok, _ in counts.most_common(top_n)]
    vocab_set = set(shared_vocab)
    v = len(shared_vocab)

    for p in positions:
        toks = pos_tokens.get(p, [])
        sizes[p] = len(toks)
        # Build bigram counts in shared vocab space
        mat = np.zeros((v, v), dtype=float)
        tok_to_idx = {t: i for i, t in enumerate(shared_vocab)}
        for i in range(len(toks) - 1):
            a, b = toks[i], toks[i + 1]
            if a in tok_to_idx and b in tok_to_idx:
                mat[tok_to_idx[a], tok_to_idx[b]] += 1
        flat = mat.flatten()
        total = flat.sum()
        if total > 0:
            flat = flat / total
        else:
            flat = np.ones_like(flat) / len(flat)
        dists[p] = flat

    comparisons = []
    for i, pa in enumerate(positions):
        for pb in positions[i + 1:]:
            # Ensure same length
            d_a, d_b = dists[pa], dists[pb]
            jsd = float(jensen_shannon_divergence(d_a, d_b))
            comparisons.append(PositionalBigramComparison(
                position_a=pa, position_b=pb, jsd=jsd,
                n_tokens_a=sizes[pa], n_tokens_b=sizes[pb],
            ))
    return comparisons


def _token_identity_analysis(
    lines: List[List[str]], top_n: int = 100,
) -> TokenIdentityTest:
    """
    For each high-frequency token, compare its co-occurrence vectors
    across initial, medial, and final positions.
    """
    # Gather per-position co-occurrence vectors
    all_tokens = [t for line in lines for t in line]
    freq = Counter(all_tokens)
    top_tokens = [t for t, _ in freq.most_common(top_n)]
    top_set = set(top_tokens)

    # Build co-occurrence counts per (token, position)
    pos_cooc: Dict[str, Dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))

    for line in lines:
        n = len(line)
        if n < 3:
            continue
        t1 = max(1, n // 3)
        t2 = max(t1 + 1, 2 * n // 3)
        for idx, tok in enumerate(line):
            if tok not in top_set:
                continue
            if idx < t1:
                pos = 'initial'
            elif idx < t2:
                pos = 'medial'
            else:
                pos = 'final'
            # Context: neighbors within window of 2
            for d in (-2, -1, 1, 2):
                ni = idx + d
                if 0 <= ni < n:
                    pos_cooc[tok][pos][line[ni]] += 1

    # Compare initial vs final co-occurrence vectors for each token
    results: List[Tuple[str, float]] = []
    for tok in top_tokens:
        vec_init = pos_cooc[tok].get('initial', Counter())
        vec_final = pos_cooc[tok].get('final', Counter())
        if not vec_init or not vec_final:
            continue
        # Build aligned vectors
        all_ctx = set(vec_init.keys()) | set(vec_final.keys())
        v_i = np.array([vec_init.get(c, 0) for c in all_ctx], dtype=float)
        v_f = np.array([vec_final.get(c, 0) for c in all_ctx], dtype=float)
        n_i, n_f = v_i.sum(), v_f.sum()
        if n_i == 0 or n_f == 0:
            continue
        cos = cosine_similarity(v_i, v_f)
        results.append((tok, float(cos)))

    if not results:
        return TokenIdentityTest(
            n_tokens_tested=0, n_position_dependent=0,
            fraction_position_dependent=0.0, mean_cosine_across_positions=0.0,
            top_position_dependent=[], top_position_stable=[],
        )

    results.sort(key=lambda x: x[1])
    n_dep = sum(1 for _, cos in results if cos < 0.3)
    mean_cos = float(np.mean([cos for _, cos in results]))

    return TokenIdentityTest(
        n_tokens_tested=len(results),
        n_position_dependent=n_dep,
        fraction_position_dependent=n_dep / len(results),
        mean_cosine_across_positions=mean_cos,
        top_position_dependent=results[:10],
        top_position_stable=results[-10:][::-1],
    )


def _reference_positional_jsd(
    ref_corpus, language: str, top_n: int = 200,
) -> Optional[ReferencePositionalJSD]:
    """Compute positional JSDs for a reference language."""
    try:
        tokens = ref_corpus.get_combined_tokens(language)
    except Exception:
        return None
    if len(tokens) < 100:
        return None

    # Split into synthetic "lines" of ~8 tokens (typical Voynich line length)
    line_len = 8
    lines = [tokens[i:i + line_len] for i in range(0, len(tokens), line_len)]
    pos_tokens = _split_by_position(lines)
    comparisons = _compute_jsd_between_positions(pos_tokens, top_n=top_n)

    jsd_dict = {}
    for c in comparisons:
        key = f"{c.position_a}_{c.position_b}"
        jsd_dict[key] = c.jsd

    im = jsd_dict.get('initial_medial', 0.0)
    if_ = jsd_dict.get('initial_final', 0.0)
    mf = jsd_dict.get('medial_final', 0.0)
    mean_j = (im + if_ + mf) / 3.0

    return ReferencePositionalJSD(
        language=language,
        initial_medial_jsd=im,
        initial_final_jsd=if_,
        medial_final_jsd=mf,
        mean_jsd=mean_j,
    )


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_position_dependent() -> Dict:
    """
    Phase 9.3: Test whether the Voynich encoding is position-dependent
    (polyalphabetic) by comparing bigram statistics across token positions.
    """
    print("Phase 9.3: Position-Dependent Encoding Test")
    print("=" * 60)

    # --- Load data ---
    corpus = load_corpus(verbose=False)
    ref_corpus = load_reference_corpus(verbose=False)
    lines = _get_lines(corpus, language='A')
    print(f"  Voynich Language A: {len(lines)} lines")

    # ===================================================================
    # 9.3a: Position-split bigram JSDs
    # ===================================================================
    print("\n  9.3a: Position-split bigram matrices ...")
    pos_tokens = _split_by_position(lines)
    for p, toks in pos_tokens.items():
        print(f"    {p}: {len(toks):,} tokens")

    comparisons = _compute_jsd_between_positions(pos_tokens)
    voynich_jsds = [_convert(asdict(c)) for c in comparisons]
    voynich_mean_jsd = float(np.mean([c.jsd for c in comparisons])) if comparisons else 0.0

    for c in comparisons:
        print(f"    JSD({c.position_a} vs {c.position_b}) = {c.jsd:.6f}")
    print(f"    Mean JSD: {voynich_mean_jsd:.6f}")

    # ===================================================================
    # 9.3b: Token identity test
    # ===================================================================
    print("\n  9.3b: Token identity test ...")
    identity_test = _token_identity_analysis(lines, top_n=100)
    print(f"    Tokens tested: {identity_test.n_tokens_tested}")
    print(f"    Position-dependent (cosine < 0.3): {identity_test.n_position_dependent}")
    print(f"    Mean cosine across positions: {identity_test.mean_cosine_across_positions:.3f}")

    if identity_test.top_position_dependent:
        print("    Most position-dependent tokens:")
        for tok, cos in identity_test.top_position_dependent[:5]:
            print(f"      {tok}: cosine={cos:.3f}")
    if identity_test.top_position_stable:
        print("    Most position-stable tokens:")
        for tok, cos in identity_test.top_position_stable[:5]:
            print(f"      {tok}: cosine={cos:.3f}")

    # ===================================================================
    # 9.3c: Reference comparison
    # ===================================================================
    print("\n  9.3c: Reference language comparison ...")
    ref_results: List[ReferencePositionalJSD] = []
    for lang in ('latin', 'german', 'occitan', 'italian'):
        rj = _reference_positional_jsd(ref_corpus, lang)
        if rj:
            ref_results.append(rj)
            print(f"    {lang}: mean JSD = {rj.mean_jsd:.6f}")

    ref_mean_jsds = [r.mean_jsd for r in ref_results]
    overall_ref_mean = float(np.mean(ref_mean_jsds)) if ref_mean_jsds else 1e-6
    voynich_vs_ref = voynich_mean_jsd / overall_ref_mean if overall_ref_mean > 0 else 1.0
    print(f"    Voynich / reference ratio: {voynich_vs_ref:.3f}")

    # ===================================================================
    # Null test: position-shuffled Voynich
    # ===================================================================
    print("\n  Null test: position-shuffled Voynich ...")
    n_null = 50
    null_jsds: List[float] = []
    rng = random.Random(42)

    for trial in range(n_null):
        shuffled_lines = []
        for line in lines:
            sl = line.copy()
            rng.shuffle(sl)
            shuffled_lines.append(sl)
        shuffled_pos = _split_by_position(shuffled_lines)
        shuffled_comps = _compute_jsd_between_positions(shuffled_pos)
        mean_j = float(np.mean([c.jsd for c in shuffled_comps])) if shuffled_comps else 0.0
        null_jsds.append(mean_j)

    null_mean_jsd = float(np.mean(null_jsds)) if null_jsds else 0.0
    position_selectivity = voynich_mean_jsd / null_mean_jsd if null_mean_jsd > 0 else 1.0
    print(f"    Null mean JSD: {null_mean_jsd:.6f}")
    print(f"    Position selectivity: {position_selectivity:.3f}x")

    # ===================================================================
    # Gate
    # ===================================================================
    gate_jsd = voynich_vs_ref > 2.0
    gate_selectivity = position_selectivity >= 1.5
    gate_passed = gate_jsd and gate_selectivity

    if gate_passed:
        verdict = 'position_dependent_encoding_detected'
    elif gate_selectivity and not gate_jsd:
        verdict = 'position_effects_within_natural_language_range'
    elif gate_jsd and not gate_selectivity:
        verdict = 'high_jsd_but_shuffling_matches'
    else:
        verdict = 'no_position_dependent_signal'

    print(f"\n  Gate: jsd_ratio={gate_jsd}  "
          f"selectivity={gate_selectivity}  passed={gate_passed}")
    print(f"  Verdict: {verdict}")

    # ===================================================================
    # Save
    # ===================================================================
    result = PositionDependentResult(
        voynich_positional_jsds=voynich_jsds,
        voynich_mean_jsd=voynich_mean_jsd,
        token_identity_test=_convert(asdict(identity_test)),
        reference_jsds=[_convert(asdict(r)) for r in ref_results],
        voynich_vs_reference_ratio=voynich_vs_ref,
        null_shuffled_jsds=null_jsds,
        null_mean_jsd=null_mean_jsd,
        position_selectivity=position_selectivity,
        gate_jsd=gate_jsd,
        gate_selectivity=gate_selectivity,
        gate_passed=gate_passed,
        verdict=verdict,
    )

    out = _convert(asdict(result))
    with open(_results_dir() / 'position_dependent.json', 'w') as f:
        json.dump(out, f, indent=2)

    print(f"\n  Results saved to results/position_dependent.json")
    return out
