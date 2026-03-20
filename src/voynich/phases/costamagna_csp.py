"""
Phase 58: Costamagna-Constrained CSP
======================================
Uses Costamagna's attested syllable inventory to constrain CSP domains
for the unresolved triples.  Runs greedy hill-climbing with random
restarts to find an assignment that improves on T_P15.

Dependency chain:
    results/phase57_verdict.json
    results/combined_refine.json
    results/triple_tiers.json
    results/modifier_integrate.json
    data/GL.S.III.MISC.12/extraction/syllabary_table.json
        -> results/cost_domains.json     (Step 58.1)
        -> results/cost_reduction.json   (Step 58.2)
        -> results/cost_csp.json         (Step 58.3)
"""

import json
import math
import os
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from voynich.core._paths import data_dir, results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    load_corpus,
    tokenize_eva_chars,
)
from voynich.core.reference import (
    EVA_VISUAL_COMPONENTS,
    build_expanded_word_set,
    load_reference_corpus,
)
from voynich.phases.csp_solver import decode_token
from voynich.phases.null_corpus import (
    _build_eva_bigram_model,
    _generate_null_corpus,
    _reconstruct_modifier_rules,
)


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
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CostamagnaDomain:
    triple_key: str
    domain_type: str       # CONFIRMED / CODA_MARKER / UNRESOLVED
    domain: List[str]      # candidate syllables
    size: int
    source: str


@dataclass
class DomainComparison:
    per_triple: List[Dict[str, Any]]
    n_confirmed: int
    n_coda: int
    n_unresolved: int
    total_search_space_log10: float
    mean_domain_size: float
    phase_14_mean: float


@dataclass
class CspSolution:
    assignment: Dict[str, str]
    score: float
    dict_hit: float
    n_changed_from_tp15: int
    changes: List[Dict[str, str]]


@dataclass
class CostDomainsResult:
    phase: str = "58"
    step: str = "58.1"
    experiment: str = "cost_domains"
    domains: List[CostamagnaDomain] = field(default_factory=list)
    costamagna_cv_count: int = 0
    costamagna_cvc_count: int = 0
    runtime_seconds: float = 0.0


@dataclass
class CostReductionResult:
    phase: str = "58"
    step: str = "58.2"
    experiment: str = "cost_reduction"
    comparison: Optional[DomainComparison] = None
    runtime_seconds: float = 0.0


@dataclass
class CostCspResult:
    phase: str = "58"
    step: str = "58.3"
    experiment: str = "cost_csp"
    n_restarts: int = 0
    n_solutions: int = 0
    best_score: float = 0.0
    best_dict_hit: float = 0.0
    solutions: List[CspSolution] = field(default_factory=list)
    tp15_dict_hit: float = 0.0
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Costamagna syllable loading
# ---------------------------------------------------------------------------

def _load_costamagna_inventory() -> Tuple[Set[str], Set[str], Set[str]]:
    """Load Costamagna syllabary -> (cv_set, cvc_set, all_set)."""
    syl_path = os.path.join(
        str(data_dir('GL.S.III.MISC.12/extraction')),
        'syllabary_table.json',
    )
    if not os.path.exists(syl_path):
        return set(), set(), set()

    with open(syl_path) as f:
        entries = json.load(f)

    cv_set: Set[str] = set()
    cvc_set: Set[str] = set()
    all_set: Set[str] = set()

    for entry in entries:
        syl = entry.get('syllable', '')
        struct = entry.get('structure', '')
        parts = [s.strip() for s in syl.split('-')] if '-' in syl else [syl]
        for part in parts:
            pl = part.lower()
            all_set.add(pl)
            if struct in ('CV', 'VC', 'V'):
                cv_set.add(pl)
            elif struct in ('CVC', 'CCV', 'VCC', 'CVCC'):
                cvc_set.add(pl)
            elif struct == 'shared_sign':
                if len(pl) <= 2:
                    cv_set.add(pl)
                else:
                    cvc_set.add(pl)
            else:
                if len(pl) <= 2:
                    cv_set.add(pl)
                else:
                    cvc_set.add(pl)

    return cv_set, cvc_set, all_set


# ---------------------------------------------------------------------------
# Domain construction (Step 58.1)
# ---------------------------------------------------------------------------

# Costamagna shared-sign pairs
SHARED_SIGNS = {
    'ad': 'at', 'at': 'ad',
    'me': 'mi', 'mi': 'me',
    'ne': 'ni', 'ni': 'ne',
}


def _get_confirmed_triples(rd: str) -> Dict[str, str]:
    """Get confirmed triple assignments from Phase 15 + tiering."""
    refine_data = _safe_load(os.path.join(rd, 'combined_refine.json'))
    assignment = refine_data.get('best_assignment', {})

    # Use triple_tiers.json to identify confirmed triples
    tier_data = _safe_load(os.path.join(rd, 'triple_tiers.json'))
    confirmed_triples: Dict[str, str] = {}

    if tier_data and 'tiers' in tier_data:
        tiers = tier_data['tiers']
        # tiers is a dict: {'CONFIRMED': [...], 'LANDSCAPE_CONFIRMED': [...], ...}
        if isinstance(tiers, dict):
            for entry in tiers.get('CONFIRMED', []):
                triple = entry.get('triple_key', '')
                if triple in assignment:
                    confirmed_triples[triple] = assignment[triple]
            # Also include LANDSCAPE_CONFIRMED as confirmed
            for entry in tiers.get('LANDSCAPE_CONFIRMED', []):
                triple = entry.get('triple_key', '')
                if triple in assignment:
                    confirmed_triples[triple] = assignment[triple]
        elif isinstance(tiers, list):
            for entry in tiers:
                triple = entry.get('triple_key', '')
                tier = entry.get('tier', '')
                if tier in ('CONFIRMED', 'LANDSCAPE_CONFIRMED') and triple in assignment:
                    confirmed_triples[triple] = assignment[triple]

    # If no tiering info, use all assignments as baseline
    if not confirmed_triples:
        confirmed_triples = dict(assignment)

    return confirmed_triples


def _get_coda_triples(rd: str) -> Set[str]:
    """Get triple keys that are coda markers (from Phase 57)."""
    p57_data = _safe_load(os.path.join(rd, 'phase57_verdict.json'))
    coda_triples: Set[str] = set()

    # Check if Phase 57 passed
    if not p57_data or p57_data.get('n_passed', 0) < 3:
        return coda_triples  # Phase 57 didn't pass — don't exclude codas

    # Get modifier triples from modifier_integrate.json
    mod_data = _safe_load(os.path.join(rd, 'modifier_integrate.json'))
    if mod_data:
        for cls in mod_data.get('classifications', []):
            if cls.get('final_classification') == 'modifier':
                triple = cls.get('triple_key', '')
                if triple:
                    coda_triples.add(triple)

    return coda_triples


def build_costamagna_domains(rd: str) -> List[CostamagnaDomain]:
    """Build Costamagna-constrained domains for all triples."""
    cv_set, cvc_set, all_set = _load_costamagna_inventory()
    confirmed = _get_confirmed_triples(rd)
    coda_triples = _get_coda_triples(rd)

    # Get all triples from the assignment
    refine_data = _safe_load(os.path.join(rd, 'combined_refine.json'))
    assignment = refine_data.get('best_assignment', {})

    confirmed_values = set(confirmed.values())
    # Use CV syllables as the base domain for unresolved triples
    base_domain = sorted(cv_set - confirmed_values)

    # Add shared-sign expansions
    expanded = set(base_domain)
    for syl in list(expanded):
        if syl in SHARED_SIGNS:
            partner = SHARED_SIGNS[syl]
            if partner not in confirmed_values:
                expanded.add(partner)
    base_domain = sorted(expanded)

    domains: List[CostamagnaDomain] = []
    for triple_key in sorted(assignment.keys()):
        if triple_key in confirmed:
            domains.append(CostamagnaDomain(
                triple_key=triple_key,
                domain_type='CONFIRMED',
                domain=[confirmed[triple_key]],
                size=1,
                source=f"Phase 15/28 confirmed: {confirmed[triple_key]}",
            ))
        elif triple_key in coda_triples:
            domains.append(CostamagnaDomain(
                triple_key=triple_key,
                domain_type='CODA_MARKER',
                domain=[],
                size=0,
                source="Phase 57 coda classification",
            ))
        else:
            domains.append(CostamagnaDomain(
                triple_key=triple_key,
                domain_type='UNRESOLVED',
                domain=list(base_domain),
                size=len(base_domain),
                source="Costamagna CV inventory",
            ))

    return domains


# ---------------------------------------------------------------------------
# Domain comparison (Step 58.2)
# ---------------------------------------------------------------------------

def _compare_domain_sizes(domains: List[CostamagnaDomain]) -> DomainComparison:
    """Compare Costamagna domain sizes against Phase 11/14 baselines."""
    unresolved = [d for d in domains if d.domain_type == 'UNRESOLVED']
    confirmed = [d for d in domains if d.domain_type == 'CONFIRMED']
    coda = [d for d in domains if d.domain_type == 'CODA_MARKER']

    per_triple = []
    for d in domains:
        per_triple.append({
            'triple': d.triple_key,
            'type': d.domain_type,
            'costamagna_size': d.size,
            'phase_11_size': 75,   # unconstrained CV
            'phase_14_size': 5.2,  # stroke-guided mean
        })

    # Total search space
    if unresolved:
        log_space = sum(math.log10(d.size) for d in unresolved if d.size > 0)
        mean_size = sum(d.size for d in unresolved) / len(unresolved)
    else:
        log_space = 0
        mean_size = 0

    return DomainComparison(
        per_triple=per_triple,
        n_confirmed=len(confirmed),
        n_coda=len(coda),
        n_unresolved=len(unresolved),
        total_search_space_log10=round(log_space, 1),
        mean_domain_size=round(mean_size, 1),
        phase_14_mean=5.2,
    )


# ---------------------------------------------------------------------------
# Greedy hill-climbing CSP (Step 58.3)
# ---------------------------------------------------------------------------

def _compute_dict_hit(
    assignment: Dict[str, str],
    tokens: List[str],
    eva_to_triple: Dict[str, str],
    ref_word_set: Set[str],
    max_tokens: int = 0,
) -> float:
    """Compute dict hit rate for an assignment."""
    toks = tokens[:max_tokens] if max_tokens > 0 else tokens
    hits = 0
    for tok in toks:
        decoded = decode_token(tok, assignment, eva_to_triple)
        if decoded.lower() in ref_word_set:
            hits += 1
    return hits / len(toks) if toks else 0.0


def _greedy_optimize(
    initial_assignment: Dict[str, str],
    domains: List[CostamagnaDomain],
    tokens: List[str],
    eva_to_triple: Dict[str, str],
    ref_word_set: Set[str],
    max_tokens: int = 5000,
    max_iterations: int = 50,
) -> Tuple[Dict[str, str], float]:
    """Greedy hill-climbing: iteratively swap each unresolved triple's value
    to maximize dict_hit."""
    assignment = dict(initial_assignment)
    unresolved = [d for d in domains if d.domain_type == 'UNRESOLVED']

    best_score = _compute_dict_hit(
        assignment, tokens, eva_to_triple, ref_word_set, max_tokens)

    for iteration in range(max_iterations):
        improved = False
        for domain in unresolved:
            triple = domain.triple_key
            current_val = assignment.get(triple, '')
            best_val = current_val
            best_val_score = best_score

            for candidate in domain.domain:
                if candidate == current_val:
                    continue
                assignment[triple] = candidate
                score = _compute_dict_hit(
                    assignment, tokens, eva_to_triple, ref_word_set, max_tokens)
                if score > best_val_score:
                    best_val = candidate
                    best_val_score = score

            if best_val != current_val:
                assignment[triple] = best_val
                best_score = best_val_score
                improved = True
            else:
                assignment[triple] = current_val

        if not improved:
            break

    return assignment, best_score


def _run_csp_hill_climbing(
    domains: List[CostamagnaDomain],
    tp15_assignment: Dict[str, str],
    tokens: List[str],
    eva_to_triple: Dict[str, str],
    ref_word_set: Set[str],
    n_restarts: int = 20,
    max_tokens: int = 5000,
) -> List[CspSolution]:
    """Run greedy hill-climbing with random restarts."""
    unresolved = [d for d in domains if d.domain_type == 'UNRESOLVED']
    confirmed = {d.triple_key: d.domain[0] for d in domains
                 if d.domain_type == 'CONFIRMED'}

    solutions: List[CspSolution] = []
    rng = random.Random(42)

    for restart in range(n_restarts):
        # Initialize: confirmed triples locked, unresolved randomly sampled
        assignment = dict(tp15_assignment)
        for d in unresolved:
            if d.domain:
                assignment[d.triple_key] = rng.choice(d.domain)

        # Override confirmed triples
        assignment.update(confirmed)

        # Optimize
        optimized, score = _greedy_optimize(
            assignment, domains, tokens, eva_to_triple, ref_word_set,
            max_tokens=max_tokens,
        )

        # Compute changes from T_P15
        changes = []
        for d in unresolved:
            triple = d.triple_key
            old_val = tp15_assignment.get(triple, '')
            new_val = optimized.get(triple, '')
            if old_val != new_val:
                changes.append({
                    'triple': triple,
                    'old': old_val,
                    'new': new_val,
                })

        solutions.append(CspSolution(
            assignment=optimized,
            score=round(score, 4),
            dict_hit=round(score, 4),
            n_changed_from_tp15=len(changes),
            changes=changes,
        ))

    # Sort by score (highest first)
    solutions.sort(key=lambda s: -s.score)
    return solutions


# ---------------------------------------------------------------------------
# CLI entry points
# ---------------------------------------------------------------------------

def run_cost_domains():
    """Step 58.1: Build Costamagna-constrained domains."""
    t0 = time.time()
    print("=" * 70)
    print("Phase 58, Step 1: Costamagna-Constrained Domains")
    print("=" * 70)

    rd = str(_results_dir())
    domains = build_costamagna_domains(rd)

    cv_set, cvc_set, _ = _load_costamagna_inventory()

    print(f"\n  Costamagna inventory: {len(cv_set)} CV + {len(cvc_set)} CVC")
    print(f"\n  Domain assignments:")
    for d in domains:
        print(f"    {d.triple_key:<40} {d.domain_type:<12} size={d.size}")

    n_confirmed = sum(1 for d in domains if d.domain_type == 'CONFIRMED')
    n_coda = sum(1 for d in domains if d.domain_type == 'CODA_MARKER')
    n_unresolved = sum(1 for d in domains if d.domain_type == 'UNRESOLVED')
    print(f"\n  CONFIRMED: {n_confirmed}  CODA_MARKER: {n_coda}  UNRESOLVED: {n_unresolved}")

    result = CostDomainsResult(
        domains=domains,
        costamagna_cv_count=len(cv_set),
        costamagna_cvc_count=len(cvc_set),
        runtime_seconds=round(time.time() - t0, 2),
    )
    path = _save_json(rd, 'cost_domains.json', result)
    print(f"\n  Saved: {path}")
    print(f"  Step 58.1 completed in {time.time() - t0:.1f}s")


def run_cost_reduction():
    """Step 58.2: Compare domain sizes across phases."""
    t0 = time.time()
    print("=" * 70)
    print("Phase 58, Step 2: Domain Size Comparison")
    print("=" * 70)

    rd = str(_results_dir())
    domains = build_costamagna_domains(rd)
    comparison = _compare_domain_sizes(domains)

    print(f"\n  Domain sizes:")
    print(f"    CONFIRMED: {comparison.n_confirmed}")
    print(f"    CODA_MARKER: {comparison.n_coda}")
    print(f"    UNRESOLVED: {comparison.n_unresolved}")
    print(f"\n  Mean domain size (unresolved): {comparison.mean_domain_size}")
    print(f"  Phase 14 mean: {comparison.phase_14_mean}")
    print(f"  Total search space: 10^{comparison.total_search_space_log10}")

    print(f"\n  Phase comparison:")
    print(f"    Phase 11 (unconstrained): ~75 per variable")
    print(f"    Phase 14 (stroke-guided): ~5.2 per variable")
    print(f"    Costamagna (this): ~{comparison.mean_domain_size} per variable")

    result = CostReductionResult(
        comparison=comparison,
        runtime_seconds=round(time.time() - t0, 2),
    )
    path = _save_json(rd, 'cost_reduction.json', result)
    print(f"\n  Saved: {path}")
    print(f"  Step 58.2 completed in {time.time() - t0:.1f}s")


def run_cost_csp():
    """Step 58.3: Run CSP with Costamagna domains."""
    t0 = time.time()
    print("=" * 70)
    print("Phase 58, Step 3: Costamagna-Constrained CSP")
    print("=" * 70)

    rd = str(_results_dir())
    eva_to_triple = build_eva_to_triple_lookup()

    # Load T_P15
    refine_data = _safe_load(os.path.join(rd, 'combined_refine.json'))
    tp15 = refine_data.get('best_assignment', {})

    # Load reference
    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    base_words = set(w.lower() for w in ref_corpus.get_combined_tokens('latin')
                     if len(w) >= 2)
    expanded, _ = build_expanded_word_set(base_words)
    ref_word_set = base_words | expanded

    # Load corpus
    corpus = load_corpus(verbose=False)
    all_tokens = corpus.get_tokens()

    # Build domains
    domains = build_costamagna_domains(rd)

    # T_P15 baseline
    tp15_dict_hit = _compute_dict_hit(
        tp15, all_tokens, eva_to_triple, ref_word_set, max_tokens=5000)
    print(f"\n  T_P15 baseline dict_hit: {tp15_dict_hit:.4f}")

    n_unresolved = sum(1 for d in domains if d.domain_type == 'UNRESOLVED')
    print(f"  Unresolved triples: {n_unresolved}")

    if n_unresolved == 0:
        print("  No unresolved triples - nothing to optimize")
        result = CostCspResult(
            tp15_dict_hit=round(tp15_dict_hit, 4),
            runtime_seconds=round(time.time() - t0, 2),
        )
        _save_json(rd, 'cost_csp.json', result)
        return

    # Run hill-climbing
    n_restarts = 20
    print(f"\n  Running {n_restarts} random restarts ...")
    solutions = _run_csp_hill_climbing(
        domains, tp15, all_tokens, eva_to_triple, ref_word_set,
        n_restarts=n_restarts, max_tokens=5000,
    )

    # Print top solutions
    print(f"\n  Top 5 solutions:")
    for i, sol in enumerate(solutions[:5]):
        delta = sol.dict_hit - tp15_dict_hit
        sign = '+' if delta >= 0 else ''
        print(f"    #{i+1}: dict_hit={sol.dict_hit:.4f} ({sign}{delta:.4f} vs T_P15) "
              f"changes={sol.n_changed_from_tp15}")
        if sol.changes:
            for ch in sol.changes[:3]:
                print(f"         {ch['triple']}: {ch['old']} -> {ch['new']}")
            if len(sol.changes) > 3:
                print(f"         ... and {len(sol.changes) - 3} more")

    # Check consensus across solutions
    if solutions:
        best = solutions[0]
        # Check which changes are consistent across top-5
        if len(solutions) >= 5:
            consensus = {}
            for d in [dd for dd in domains if dd.domain_type == 'UNRESOLVED']:
                vals = [s.assignment.get(d.triple_key) for s in solutions[:5]]
                if len(set(vals)) == 1 and vals[0] != tp15.get(d.triple_key):
                    consensus[d.triple_key] = vals[0]
            if consensus:
                print(f"\n  Consensus changes (agree across top 5):")
                for triple, val in consensus.items():
                    old = tp15.get(triple, '?')
                    print(f"    {triple}: {old} -> {val}")
            else:
                print(f"\n  No consensus changes across top 5 solutions")

    result = CostCspResult(
        n_restarts=n_restarts,
        n_solutions=len(solutions),
        best_score=solutions[0].score if solutions else 0.0,
        best_dict_hit=solutions[0].dict_hit if solutions else 0.0,
        solutions=solutions[:10],
        tp15_dict_hit=round(tp15_dict_hit, 4),
        runtime_seconds=round(time.time() - t0, 2),
    )
    path = _save_json(rd, 'cost_csp.json', result)
    print(f"\n  Saved: {path}")
    print(f"  Step 58.3 completed in {time.time() - t0:.1f}s")
