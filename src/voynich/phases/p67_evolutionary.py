"""
Phase 67, Track 4: Composite-Scored Evolutionary Optimization
==============================================================
Treat all 13 unresolved triples as optimization variables.  Use
Track 2 (frequency) and Track 3 (features) outputs to build
constrained candidate domains.  Run an evolutionary algorithm with
dict-hit + bonus as the fitness function.

Dependency chain:
    results/p67_frequency.json        (Track 2)
    results/p67_features.json         (Track 3)
    results/combined_refine.json      (Phase 15)
    results/triple_tiers.json         (Phase 28/53)
    results/modifier_integrate.json   (Phase 16)
    data/GL.S.III.MISC.12/extraction/syllabary_table.json
        -> results/p67_evolutionary.json
"""

import json
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    load_corpus,
    tokenize_eva_chars,
)
from voynich.core.reference import build_expanded_word_set, load_reference_corpus
from voynich.phases.corrected_coda import (
    build_coda_table_v2,
    decode_token_cvc_v2,
)
from voynich.phases.costamagna_csp import _load_costamagna_inventory


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
# Confirmed / unresolved triple separation
# ---------------------------------------------------------------------------

def _get_confirmed_and_unresolved(
    rd: str,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Return (confirmed_12, unresolved_13)."""
    refine_data = _safe_load(os.path.join(rd, 'combined_refine.json'))
    assignment = refine_data.get('best_assignment', {})

    tier_data = _safe_load(os.path.join(rd, 'triple_tiers.json'))
    confirmed_keys: Set[str] = set()

    if tier_data and 'tiers' in tier_data:
        tiers = tier_data['tiers']
        if isinstance(tiers, dict):
            for entry in tiers.get('CONFIRMED', []):
                confirmed_keys.add(entry.get('triple_key', ''))
        elif isinstance(tiers, list):
            for entry in tiers:
                if entry.get('tier', '') == 'CONFIRMED':
                    confirmed_keys.add(entry.get('triple_key', ''))

    if not confirmed_keys:
        return dict(assignment), {}

    confirmed = {k: v for k, v in assignment.items() if k in confirmed_keys}
    unresolved = {k: v for k, v in assignment.items() if k not in confirmed_keys}
    return confirmed, unresolved


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class EvolutionaryResult:
    phase: str = "67"
    step: str = "67.4"
    experiment: str = "evolutionary_optimization"
    # Search space
    n_unresolved: int = 0
    domain_sizes: Dict[str, int] = field(default_factory=dict)
    total_search_space_log10: float = 0.0
    # Evolutionary params
    population_size: int = 200
    n_generations: int = 500
    mutation_rate: float = 0.15
    elite_fraction: float = 0.10
    subsample_size: int = 5000
    # Results
    best_fitness: float = 0.0
    best_assignment: Dict[str, str] = field(default_factory=dict)
    tp15_assignment: Dict[str, str] = field(default_factory=dict)
    # Full-corpus validation
    tp15_dict_hit: float = 0.0
    evo_dict_hit: float = 0.0
    improvement: float = 0.0
    n_changed_from_tp15: int = 0
    changes: List[Dict[str, str]] = field(default_factory=list)
    # Convergence
    convergence_curve: List[Dict[str, float]] = field(default_factory=list)
    # Top-5 consensus
    top5_agreement: int = 0
    # Track 2/3 agreement
    n_agree_freq: int = 0
    n_agree_feat: int = 0
    # Gates
    g1_improvement: bool = False      # E1: evo > T_P15
    g2_significant: bool = False      # E2: improvement > 0.01
    g3_changed: bool = False          # E3: >= 5 changed
    g4_consensus: bool = False        # E4: top-5 agree on >= 8/13
    g5_signal: bool = False           # E5: signal count >= T_P15
    gates_passed: int = 0
    gate_passed: bool = False
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Domain construction
# ---------------------------------------------------------------------------

def _build_domains(
    unresolved: Dict[str, str],
    confirmed_values: Set[str],
    rd: str,
) -> Dict[str, List[str]]:
    """Build candidate domains for each unresolved triple.

    Uses Track 2 (frequency) and Track 3 (features) if available,
    otherwise falls back to full Costamagna CV inventory.
    """
    cv_set, cvc_set, all_set = _load_costamagna_inventory()
    costamagna_all = cv_set | cvc_set
    if not costamagna_all:
        # Fallback: use confirmed syllable values plus common CV syllables
        costamagna_all = set('aeiou')
        for v in 'bcdfglmnprst':
            for u in 'aeiou':
                costamagna_all.add(v + u)

    # Load Track 2 frequency domains
    freq_data = _safe_load(os.path.join(rd, 'p67_frequency.json'))
    freq_domains: Dict[str, List[str]] = {}
    if freq_data and 'domains' in freq_data:
        for d in freq_data['domains']:
            freq_domains[d['triple_key']] = d.get('candidates', [])

    # Load Track 3 feature predictions
    feat_data = _safe_load(os.path.join(rd, 'p67_features.json'))
    feat_predictions: Dict[str, List[str]] = {}
    if feat_data and 'predictions' in feat_data:
        for p in feat_data['predictions']:
            tk = p.get('triple_key', '')
            candidates = []
            # Add predicted syllable
            predicted = p.get('predicted_syllable', '')
            if predicted:
                candidates.append(predicted)
            # Add top candidates from probability distribution
            for tc in p.get('top_candidates', []):
                syl = tc.get('syllable', '')
                if syl and syl not in candidates:
                    candidates.append(syl)
            feat_predictions[tk] = candidates[:5]

    # Build domains per triple
    domains: Dict[str, List[str]] = {}

    for triple_key in unresolved:
        domain_set: Set[str] = set()

        # Start with frequency-matched candidates
        if triple_key in freq_domains and freq_domains[triple_key]:
            domain_set.update(freq_domains[triple_key])

        # Add feature-predicted candidates
        if triple_key in feat_predictions:
            domain_set.update(feat_predictions[triple_key])

        # If no Track 2/3 data, use full Costamagna inventory
        if not domain_set:
            domain_set = set(costamagna_all)

        # Always include the current T_P15 value (so we can find it's still optimal)
        if unresolved[triple_key]:
            domain_set.add(unresolved[triple_key])

        # Note: do NOT exclude confirmed_values — duplicates are allowed
        # (T_P15 has 'di' and 'se' assigned to multiple triples)

        domains[triple_key] = sorted(domain_set)

    return domains


# ---------------------------------------------------------------------------
# Pre-compiled token structures for fast fitness evaluation
# ---------------------------------------------------------------------------

@dataclass
class PrecompiledToken:
    """Pre-decomposed token: list of (type, value) where type is
    'FIXED' (confirmed syllable or coda) or a triple_key (unresolved)."""
    parts: List[Tuple[str, str]]  # (type, value)
    # type = 'FIXED' means value is already known
    # type = triple_key means value comes from assignment[triple_key]


def _precompile_tokens(
    tokens: List[str],
    confirmed: Dict[str, str],
    eva_to_triple: Dict[str, str],
    coda_table,
) -> List[PrecompiledToken]:
    """Pre-decompose tokens so fitness only needs string concatenation + dict lookup."""
    from voynich.phases.corrected_coda import classify_token_chars_v2
    from voynich.phases.coda_markers import get_coda

    confirmed_keys = set(confirmed.keys())
    compiled = []

    for token in tokens:
        eva_chars = tokenize_eva_chars(token)
        if not eva_chars:
            compiled.append(PrecompiledToken(parts=[]))
            continue

        classified = classify_token_chars_v2(eva_chars, coda_table)
        parts: List[Tuple[str, str]] = []

        for role, char in classified:
            if role == 'SYLLABIC':
                triple_key = eva_to_triple.get(char, '')
                if not triple_key:
                    parts.append(('FIXED', '?'))
                elif triple_key in confirmed_keys:
                    # Confirmed: value is fixed
                    parts.append(('FIXED', confirmed[triple_key]))
                else:
                    # Unresolved: value depends on individual
                    parts.append((triple_key, ''))
            elif role == 'CODA_MARKER':
                coda = get_coda(char, coda_table)
                if coda:
                    parts.append(('FIXED', coda))

        compiled.append(PrecompiledToken(parts=parts))

    return compiled


def _fast_fitness(
    individual: Dict[str, str],
    compiled_tokens: List[PrecompiledToken],
    ref_word_set: Set[str],
    freq_predictions: Dict[str, str],
    feat_predictions: Dict[str, str],
) -> float:
    """Fast fitness: string concatenation from pre-compiled tokens."""
    n_hits = 0
    n_total = len(compiled_tokens)

    for ct in compiled_tokens:
        if not ct.parts:
            continue

        has_unknown = False
        chars = []
        for ptype, pval in ct.parts:
            if ptype == 'FIXED':
                if pval == '?':
                    has_unknown = True
                    break
                chars.append(pval)
            else:
                syl = individual.get(ptype, '?')
                if syl == '?':
                    has_unknown = True
                    break
                chars.append(syl)

        if has_unknown:
            continue

        decoded = ''.join(chars)
        if decoded in ref_word_set:
            n_hits += 1

    dict_hit = n_hits / n_total if n_total > 0 else 0.0

    # Bonuses for agreement with Track 2/3
    bonus = 0.0
    for tk, syl in individual.items():
        if tk in freq_predictions and freq_predictions[tk] == syl:
            bonus += 0.005
        if tk in feat_predictions and feat_predictions[tk] == syl:
            bonus += 0.003

    return dict_hit + bonus


# ---------------------------------------------------------------------------
# Evolutionary operators
# ---------------------------------------------------------------------------

def _create_individual(
    domains: Dict[str, List[str]],
    rng: np.random.Generator,
) -> Dict[str, str]:
    """Create a random individual."""
    ind = {}
    for tk, domain in domains.items():
        ind[tk] = domain[rng.integers(0, len(domain))]
    return ind


def _tournament_select(
    population: List[Dict[str, str]],
    fitnesses: List[float],
    rng: np.random.Generator,
    k: int = 5,
) -> Dict[str, str]:
    """Tournament selection."""
    indices = rng.choice(len(population), size=min(k, len(population)), replace=False)
    best = max(indices, key=lambda i: fitnesses[i])
    return dict(population[best])


def _crossover(
    parent1: Dict[str, str],
    parent2: Dict[str, str],
    keys: List[str],
    rng: np.random.Generator,
) -> Dict[str, str]:
    """Uniform crossover."""
    child = {}
    for key in keys:
        if rng.random() < 0.5:
            child[key] = parent1[key]
        else:
            child[key] = parent2[key]
    return child


def _mutate(
    individual: Dict[str, str],
    domains: Dict[str, List[str]],
    rng: np.random.Generator,
    mutation_rate: float = 0.15,
) -> Dict[str, str]:
    """Per-variable mutation."""
    ind = dict(individual)
    for tk, domain in domains.items():
        if rng.random() < mutation_rate:
            ind[tk] = domain[rng.integers(0, len(domain))]
    return ind


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_evo_optimize():
    """Track 4: Evolutionary optimization of unresolved triples."""
    t0 = time.time()
    rd = str(_results_dir())

    pop_size = 200
    n_generations = 500
    mutation_rate = 0.15
    elite_frac = 0.10
    subsample_size = 5000

    print("Phase 67.4 — Evolutionary Optimization")
    print("=" * 42)

    # --- Load data ---
    confirmed, unresolved = _get_confirmed_and_unresolved(rd)
    confirmed_values = set(confirmed.values())
    print(f"  Confirmed: {len(confirmed)}, Unresolved: {len(unresolved)}")

    eva_to_triple = build_eva_to_triple_lookup()
    coda_table = build_coda_table_v2()

    corpus = load_corpus(verbose=False)
    all_tokens = corpus.get_tokens()

    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    base_words = set(w.lower() for w in ref_corpus.get_combined_tokens('latin')
                     if len(w) >= 2)
    expanded, _ = build_expanded_word_set(base_words)
    ref_word_set = base_words | expanded

    # Subsample for fitness evaluation
    subsample = all_tokens[:subsample_size]
    print(f"  Corpus: {len(all_tokens)} tokens (subsample: {len(subsample)})")

    # --- Build domains ---
    print("  Building candidate domains...")
    domains = _build_domains(unresolved, confirmed_values, rd)
    total_space = 1.0
    for tk in sorted(domains.keys()):
        ds = len(domains[tk])
        total_space *= ds
        print(f"    {tk}: {ds} candidates")

    log_space = np.log10(total_space) if total_space > 0 else 0.0
    print(f"  Total search space: 10^{log_space:.1f}")

    # --- Build Track 2/3 prediction maps for bonuses ---
    freq_data = _safe_load(os.path.join(rd, 'p67_frequency.json'))
    freq_preds: Dict[str, str] = {}
    if freq_data and 'domains' in freq_data:
        for d in freq_data['domains']:
            cands = d.get('candidates', [])
            if len(cands) == 1:
                freq_preds[d['triple_key']] = cands[0]

    feat_data = _safe_load(os.path.join(rd, 'p67_features.json'))
    feat_preds: Dict[str, str] = {}
    if feat_data and 'predictions' in feat_data:
        for p in feat_data['predictions']:
            pred = p.get('predicted_syllable', '')
            if pred:
                feat_preds[p['triple_key']] = pred

    # --- Pre-compile subsample tokens for fast fitness ---
    print("  Pre-compiling token structures...")
    compiled = _precompile_tokens(subsample, confirmed, eva_to_triple, coda_table)
    print(f"  Pre-compiled {len(compiled)} tokens")

    # --- T_P15 baseline ---
    tp15_fitness = _fast_fitness(
        unresolved, compiled, ref_word_set, freq_preds, feat_preds)
    print(f"\n  T_P15 baseline fitness: {tp15_fitness:.4f}")

    # --- Evolutionary search ---
    print(f"\n  Running evolutionary search ({pop_size} pop × {n_generations} gen)...")
    rng = np.random.default_rng(42)
    keys = sorted(domains.keys())

    # Initialize population
    population = [_create_individual(domains, rng) for _ in range(pop_size)]
    # Seed one individual with T_P15 values
    population[0] = dict(unresolved)

    best_ever = dict(unresolved)
    best_fitness_ever = tp15_fitness
    convergence = []

    # Track top-5 for consensus
    top5_archive: List[Tuple[float, Dict[str, str]]] = []

    for gen in range(n_generations):
        # Evaluate using fast pre-compiled fitness
        fitnesses = [
            _fast_fitness(ind, compiled, ref_word_set, freq_preds, feat_preds)
            for ind in population
        ]

        # Track best
        gen_best_idx = int(np.argmax(fitnesses))
        gen_best_fitness = fitnesses[gen_best_idx]

        if gen_best_fitness > best_fitness_ever:
            best_fitness_ever = gen_best_fitness
            best_ever = dict(population[gen_best_idx])

        # Update top-5 archive
        for fit, ind in zip(fitnesses, population):
            top5_archive.append((fit, dict(ind)))
        top5_archive.sort(key=lambda x: -x[0])
        top5_archive = top5_archive[:5]

        if gen % 100 == 0 or gen == n_generations - 1:
            mean_fit = float(np.mean(fitnesses))
            print(f"    Gen {gen:4d}: best={gen_best_fitness:.4f}, "
                  f"mean={mean_fit:.4f}, ever={best_fitness_ever:.4f}")
            convergence.append({
                'generation': gen,
                'best': round(gen_best_fitness, 4),
                'mean': round(mean_fit, 4),
                'best_ever': round(best_fitness_ever, 4),
            })

        # Selection + Reproduction
        n_elite = max(1, int(elite_frac * pop_size))
        elite_indices = sorted(range(len(fitnesses)),
                               key=lambda i: fitnesses[i], reverse=True)[:n_elite]
        new_pop = [dict(population[i]) for i in elite_indices]

        while len(new_pop) < pop_size:
            p1 = _tournament_select(population, fitnesses, rng)
            p2 = _tournament_select(population, fitnesses, rng)
            child = _crossover(p1, p2, keys, rng)
            child = _mutate(child, domains, rng, mutation_rate)
            new_pop.append(child)

        population = new_pop[:pop_size]

    # --- Top-5 consensus ---
    top5_consensus = 0
    if len(top5_archive) >= 2:
        for tk in keys:
            vals = [ind[tk] for _, ind in top5_archive]
            most_common = Counter(vals).most_common(1)[0]
            if most_common[1] >= max(2, len(top5_archive) - 1):
                top5_consensus += 1

    # --- Full-corpus validation (using pre-compiled for speed) ---
    print("\n  Validating best solution on full corpus...")
    full_compiled = _precompile_tokens(all_tokens, confirmed, eva_to_triple, coda_table)

    tp15_dict_hit = _fast_fitness(
        unresolved, full_compiled, ref_word_set, {}, {})
    evo_dict_hit = _fast_fitness(
        best_ever, full_compiled, ref_word_set, {}, {})
    improvement = evo_dict_hit - tp15_dict_hit

    print(f"  T_P15 dict_hit: {tp15_dict_hit:.4f}")
    print(f"  Evo dict_hit:   {evo_dict_hit:.4f}")
    print(f"  Improvement:    {improvement:+.4f}")

    # Changes from T_P15
    changes = []
    n_changed = 0
    for tk in keys:
        if best_ever[tk] != unresolved[tk]:
            n_changed += 1
            changes.append({
                'triple_key': tk,
                'tp15': unresolved[tk],
                'evo': best_ever[tk],
            })

    # Track 2/3 agreement
    n_agree_freq = sum(1 for tk in keys
                       if tk in freq_preds and best_ever[tk] == freq_preds[tk])
    n_agree_feat = sum(1 for tk in keys
                       if tk in feat_preds and best_ever[tk] == feat_preds[tk])

    # --- Signal word check ---
    from voynich.phases.suffix_calibration import SIGNAL_WORDS_51
    signal_set = set(SIGNAL_WORDS_51.keys())

    # Count signal words using pre-compiled tokens
    def _count_signal(ind, compiled_toks, sig_set):
        count = 0
        for ct in compiled_toks:
            if not ct.parts:
                continue
            chars = []
            ok = True
            for ptype, pval in ct.parts:
                if ptype == 'FIXED':
                    if pval == '?':
                        ok = False
                        break
                    chars.append(pval)
                else:
                    syl = ind.get(ptype, '?')
                    if syl == '?':
                        ok = False
                        break
                    chars.append(syl)
            if ok and ''.join(chars) in sig_set:
                count += 1
        return count

    tp15_signal = _count_signal(unresolved, full_compiled, signal_set)
    evo_signal = _count_signal(best_ever, full_compiled, signal_set)

    # --- Gates ---
    g1 = evo_dict_hit > tp15_dict_hit
    g2 = improvement > 0.01
    g3 = n_changed >= 5
    g4 = top5_consensus >= 8
    g5 = evo_signal >= tp15_signal
    gates_passed = sum([g1, g2, g3, g4, g5])

    result = EvolutionaryResult(
        n_unresolved=len(unresolved),
        domain_sizes={tk: len(d) for tk, d in domains.items()},
        total_search_space_log10=round(log_space, 1),
        population_size=pop_size,
        n_generations=n_generations,
        mutation_rate=mutation_rate,
        elite_fraction=elite_frac,
        subsample_size=subsample_size,
        best_fitness=round(best_fitness_ever, 4),
        best_assignment=best_ever,
        tp15_assignment=dict(unresolved),
        tp15_dict_hit=round(tp15_dict_hit, 4),
        evo_dict_hit=round(evo_dict_hit, 4),
        improvement=round(improvement, 4),
        n_changed_from_tp15=n_changed,
        changes=changes,
        convergence_curve=convergence,
        top5_agreement=top5_consensus,
        n_agree_freq=n_agree_freq,
        n_agree_feat=n_agree_feat,
        g1_improvement=g1,
        g2_significant=g2,
        g3_changed=g3,
        g4_consensus=g4,
        g5_signal=g5,
        gates_passed=gates_passed,
        gate_passed=gates_passed >= 3,
        runtime_seconds=round(time.time() - t0, 1),
    )

    path = _save_json(rd, 'p67_evolutionary.json', result)

    # --- Summary ---
    print(f"\n  Summary")
    print(f"  -------")
    print(f"  Best fitness:     {best_fitness_ever:.4f} (T_P15: {tp15_fitness:.4f})")
    print(f"  Dict hit:         {evo_dict_hit:.4f} vs {tp15_dict_hit:.4f} "
          f"({'PASS' if g1 else 'FAIL'})")
    print(f"  Improvement:      {improvement:+.4f} "
          f"({'PASS' if g2 else 'FAIL'} > 0.01)")
    print(f"  Triples changed:  {n_changed}/13 ({'PASS' if g3 else 'FAIL'} >= 5)")
    print(f"  Top-5 consensus:  {top5_consensus}/13 ({'PASS' if g4 else 'FAIL'} >= 8)")
    print(f"  Signal tokens:    {evo_signal} vs {tp15_signal} "
          f"({'PASS' if g5 else 'FAIL'})")
    print(f"  Gates: {gates_passed}/5")
    print(f"  Saved: {path}")
    print(f"  Time: {result.runtime_seconds:.1f}s")
