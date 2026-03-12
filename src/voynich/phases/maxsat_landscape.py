"""
Phase 44 – Track A: MaxSAT Landscape Enumeration
====================================================
Encode the 25-triple assignment problem as a Weighted Partial MaxSAT
instance.  Use RC2 to enumerate near-optimal solutions.  Characterise
the solution landscape as FLAT / BASINED / PEAKED.

Dependency chain:
    combined_refine.json       (Phase 15 assignment)
    bootstrap_loop.json        (Phase 30 confirmed triples)
    modifier_integrate.json    (Phase 16 modifiers)
    signal_isolation.json      (Phase 28 signal words)
        -> maxsat_encoding.json    (Step 44A.1)
        -> maxsat_solutions.json   (Step 44A.2)
        -> maxsat_landscape.json   (Step 44A.3)
        -> maxsat_validation.json  (Step 44A.4)
"""

import json
import math
import os
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    decode_token_modifier_aware,
    load_corpus,
    tokenize_eva_chars,
)
from voynich.core.reference import (
    EVA_VISUAL_COMPONENTS,
    PHONEME_NUCLEUS_MAP,
    PHONEME_PLACE_MAP,
    build_expanded_word_set,
    build_triple_phoneme_hypotheses,
    load_reference_corpus,
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
    if isinstance(obj, (bool, int, float, str, type(None))):
        return obj
    return str(obj)


def _safe_load(path: str) -> Dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


# ---------------------------------------------------------------------------
# Modifier reconstruction (same pattern as signal_isolation.py)
# ---------------------------------------------------------------------------

def _reconstruct_modifier_rules(data: Dict) -> Tuple[Set[str], Dict[str, str]]:
    modifier_chars = set(data.get('modifier_chars', []))
    modifier_rules: Dict[str, str] = {}
    for c in data.get('classifications', []):
        if c.get('final_classification') == 'modifier':
            modifier_rules[c['eva_char']] = c.get('modifier_type', 'silent')
    return modifier_chars, modifier_rules


# ---------------------------------------------------------------------------
# Decode helpers
# ---------------------------------------------------------------------------

def _decode_corpus_r3(
    tokens: List[str],
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    modifier_chars: set,
    modifier_rules: Dict[str, str],
    ref_word_set: set,
) -> List[str]:
    """Decode tokens using R3 strategy: try alteration, then strip, then raw."""
    from voynich.phases.csp_solver import decode_token
    decoded = []
    for token in tokens:
        alt = decode_token_modifier_aware(
            token, assignment, eva_to_triple, modifier_chars, modifier_rules,
        )
        if alt.lower() in ref_word_set:
            decoded.append(alt.lower())
            continue
        stripped = decode_token_modifier_aware(
            token, assignment, eva_to_triple, modifier_chars,
        )
        if stripped.lower() in ref_word_set:
            decoded.append(stripped.lower())
            continue
        raw = decode_token(token, assignment, eva_to_triple)
        decoded.append(raw.lower())
    return decoded


def _compute_dict_hit(decoded: List[str], ref_word_set: set) -> float:
    if not decoded:
        return 0.0
    hits = sum(1 for w in decoded if w in ref_word_set)
    return hits / len(decoded)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class EncodingResult:
    n_variables: int
    n_hard_clauses: int
    n_soft_clauses: int
    n_free_triples: int
    n_confirmed_triples: int
    mean_domain_size: float
    total_soft_weight: int
    encoding_time_seconds: float


@dataclass
class SolutionEntry:
    assignment: Dict[str, str]
    cost: int
    dict_hit: float
    hamming_from_p15: int


@dataclass
class SolutionsResult:
    n_optimal: int
    n_near_optimal_1pct: int
    n_near_optimal_5pct: int
    n_near_optimal_10pct: int
    optimal_cost: int
    best_dict_hit: float
    phase15_cost: int
    phase15_rank: int
    enumeration_time_seconds: float


@dataclass
class LandscapeResult:
    classification: str  # FLAT / BASINED / PEAKED
    n_solutions: int
    n_basins: int
    per_triple_consensus: Dict[str, Dict[str, float]]
    n_landscape_confirmed: int
    mean_hamming: float
    phase15_in_optimal: bool
    runtime_seconds: float


@dataclass
class ValidationResult:
    best_maxsat_dict_hit: float
    best_maxsat_assignment: Dict[str, str]
    phase15_dict_hit: float
    delta_dict_hit: float
    n_triples_changed: int
    changed_triples: List[Dict]
    gate_passed: bool
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Step 44A.1 – WCNF Encoding
# ---------------------------------------------------------------------------

def _build_wcnf(
    triple_domains: Dict[str, List[str]],
    confirmed: Dict[str, str],
    free_triples: List[str],
    bigram_pairs: List[Tuple[str, str, int]],
    signal_words: List[Tuple[str, float, List[Tuple[str, ...]]]],
    ref_bigram_probs: Dict[Tuple[str, str], float],
) -> Tuple[Any, Dict[Tuple[str, str], int], int]:
    """Build Weighted Partial MaxSAT instance.

    Returns (wcnf, var_map, n_soft_weight).
    """
    from pysat.formula import WCNF

    wcnf = WCNF()
    var_map: Dict[Tuple[str, str], int] = {}  # (triple_key, syllable) -> var_id
    next_var = 1

    # Build variable map for ALL triples (confirmed get unit clauses later)
    all_triples = sorted(triple_domains.keys())
    for t_key in all_triples:
        for syl in triple_domains[t_key]:
            var_map[(t_key, syl)] = next_var
            next_var += 1

    # ── Hard clause: exactly-one per triple ──
    for t_key in all_triples:
        domain = triple_domains[t_key]
        var_ids = [var_map[(t_key, s)] for s in domain]

        # At-least-one
        wcnf.append(var_ids)

        # At-most-one (pairwise exclusion)
        for i in range(len(var_ids)):
            for j in range(i + 1, len(var_ids)):
                wcnf.append([-var_ids[i], -var_ids[j]])

    # ── Hard clause: confirmed assignments ──
    for t_key, syl in confirmed.items():
        if (t_key, syl) in var_map:
            wcnf.append([var_map[(t_key, syl)]])

    # ── Hard clause: all-different on free triples ──
    # For each pair of free triples and each shared syllable in both domains
    for i in range(len(free_triples)):
        for j in range(i + 1, len(free_triples)):
            t_i = free_triples[i]
            t_j = free_triples[j]
            shared_syls = set(triple_domains[t_i]) & set(triple_domains[t_j])
            for syl in shared_syls:
                v_i = var_map.get((t_i, syl))
                v_j = var_map.get((t_j, syl))
                if v_i and v_j:
                    wcnf.append([-v_i, -v_j])

    # ── Soft clause: bigram plausibility ──
    total_soft_weight = 0
    for t1_key, t2_key, corpus_count in bigram_pairs:
        if t1_key not in triple_domains or t2_key not in triple_domains:
            continue
        for s1 in triple_domains[t1_key]:
            for s2 in triple_domains[t2_key]:
                prob = ref_bigram_probs.get((s1, s2), 0.0)
                if prob > 0:
                    weight = max(1, int(corpus_count * math.log(prob + 1e-10) * -0.1))
                    v1 = var_map.get((t1_key, s1))
                    v2 = var_map.get((t2_key, s2))
                    if v1 and v2:
                        # Reward clause: if both assigned, this is good
                        # Encode as soft clause: (v1 AND v2) contributes weight
                        # In MaxSAT: soft clause [v1, v2] with weight
                        # We want to reward co-occurrence, so penalize NOT having it
                        wcnf.append([-v1, -v2], weight=weight)
                        total_soft_weight += weight

    # ── Soft clause: signal word preservation ──
    for word, sigma, triple_seq_list in signal_words:
        weight = max(1, int(sigma * 100))
        for triple_seq in triple_seq_list:
            # Each triple_seq is ((t_key, syl), (t_key, syl), ...)
            clause_vars = []
            valid = True
            for t_key, syl in triple_seq:
                v = var_map.get((t_key, syl))
                if v is None:
                    valid = False
                    break
                clause_vars.append(v)
            if valid and clause_vars:
                # Penalize NOT having all these assignments
                # If any is negated, pay penalty
                for v in clause_vars:
                    wcnf.append([v], weight=weight)
                    total_soft_weight += weight

    return wcnf, var_map, total_soft_weight


def _extract_assignment(
    model: List[int],
    var_map: Dict[Tuple[str, str], int],
    triple_domains: Dict[str, List[str]],
) -> Dict[str, str]:
    """Extract triple -> syllable assignment from a SAT model."""
    positive_vars = set(v for v in model if v > 0)
    assignment: Dict[str, str] = {}
    for t_key in triple_domains:
        for syl in triple_domains[t_key]:
            v = var_map.get((t_key, syl))
            if v and v in positive_vars:
                assignment[t_key] = syl
                break
    return assignment


def _hamming_distance(a1: Dict[str, str], a2: Dict[str, str],
                      triples: List[str]) -> int:
    return sum(1 for t in triples if a1.get(t) != a2.get(t))


def run_maxsat_encode() -> None:
    """Step 44A.1: Build WCNF instance."""
    t0 = time.time()
    print("=" * 70)
    print("STEP 44A.1: MaxSAT WCNF Encoding")
    print("=" * 70)

    rd = _results_dir()
    eva_to_triple = build_eva_to_triple_lookup()

    # Load inputs
    print("\n  1. Loading inputs ...")
    refine_data = _safe_load(os.path.join(rd, 'combined_refine.json'))
    if not refine_data:
        print("  [SKIP] combined_refine.json not found")
        return
    p15_assignment = refine_data.get('best_assignment', {})

    boot_data = _safe_load(os.path.join(rd, 'bootstrap_loop.json'))
    confirmed_list = boot_data.get('confirmed_triples', [])
    confirmed: Dict[str, str] = {}
    final_assignment = boot_data.get('final_assignment', p15_assignment)
    for t_key in confirmed_list:
        if t_key in final_assignment:
            confirmed[t_key] = final_assignment[t_key]

    # Build triple domains
    print("  2. Building triple domains ...")
    hypotheses = build_triple_phoneme_hypotheses('latin')
    all_triples = sorted(hypotheses.keys())
    free_triples = [t for t in all_triples if t not in confirmed]

    mean_domain = sum(len(hypotheses[t]) for t in all_triples) / max(len(all_triples), 1)
    print(f"     {len(all_triples)} triples, {len(confirmed)} confirmed, "
          f"{len(free_triples)} free, mean domain size {mean_domain:.1f}")

    # Build bigram pairs from corpus
    print("  3. Building corpus bigram pairs ...")
    corpus = load_corpus(verbose=False)
    all_tokens = []
    for _fol, page in corpus.pages.items():
        all_tokens.extend(page.all_tokens)

    bigram_counter: Counter = Counter()
    for token in all_tokens:
        chars = tokenize_eva_chars(token)
        triples = [eva_to_triple.get(ch) for ch in chars]
        triples = [t for t in triples if t]
        for i in range(len(triples) - 1):
            bigram_counter[(triples[i], triples[i + 1])] += 1

    # Top-1000 bigrams
    top_bigrams = bigram_counter.most_common(1000)
    bigram_pairs = [(t1, t2, cnt) for (t1, t2), cnt in top_bigrams]

    # Build reference bigram probabilities
    print("  4. Building reference bigram model ...")
    ref_corpus = load_reference_corpus()
    ref_tokens = ref_corpus.get_combined_tokens('latin')
    ref_text = ' '.join(ref_tokens[:50000])

    # Build syllable-pair frequency from reference
    ref_bigram_probs: Dict[Tuple[str, str], float] = {}
    all_syls_in_domains = set()
    for domain in hypotheses.values():
        all_syls_in_domains.update(domain)

    syl_bigram_count: Counter = Counter()
    total_syl_bigrams = 0
    for word in ref_tokens[:50000]:
        word_lower = word.lower()
        # Extract CV syllable bigrams from word
        syls_in_word = []
        i = 0
        while i < len(word_lower):
            # Try 2-char syllables first, then 1-char
            if i + 1 < len(word_lower) and word_lower[i:i+2] in all_syls_in_domains:
                syls_in_word.append(word_lower[i:i+2])
                i += 2
            elif word_lower[i:i+1] in all_syls_in_domains:
                syls_in_word.append(word_lower[i:i+1])
                i += 1
            else:
                i += 1
        for j in range(len(syls_in_word) - 1):
            syl_bigram_count[(syls_in_word[j], syls_in_word[j + 1])] += 1
            total_syl_bigrams += 1

    if total_syl_bigrams > 0:
        for pair, cnt in syl_bigram_count.items():
            ref_bigram_probs[pair] = cnt / total_syl_bigrams

    # Signal words
    print("  5. Loading signal words ...")
    sig_data = _safe_load(os.path.join(rd, 'signal_isolation.json'))
    signal_words_raw = []
    if sig_data:
        for ws in sig_data.get('word_signals', []):
            if ws.get('is_genuine_signal', False):
                word = ws['word']
                sigma = ws.get('signal_sigma', 1.0)
                # Reconstruct the triple-syllable sequence from the word
                # using the current assignment
                triple_seq = []
                valid = True
                for ch in word:
                    found = False
                    for syl_candidate in all_syls_in_domains:
                        # Simple: look for this syllable in domains
                        pass
                    # Use the assignment to find which triples produce this word
                # Simpler approach: encode signal words by their assignment
                seq = []
                for t_key, syl in final_assignment.items():
                    if syl in word:
                        seq.append((t_key, syl))
                if seq:
                    signal_words_raw.append((word, sigma, [seq]))

    # Build WCNF
    print("  6. Building WCNF instance ...")
    wcnf, var_map, total_soft = _build_wcnf(
        hypotheses, confirmed, free_triples,
        bigram_pairs, signal_words_raw, ref_bigram_probs,
    )

    n_hard = sum(1 for c in wcnf.hard if c)
    n_soft = sum(1 for c in wcnf.soft if c)

    result = EncodingResult(
        n_variables=len(var_map),
        n_hard_clauses=n_hard,
        n_soft_clauses=n_soft,
        n_free_triples=len(free_triples),
        n_confirmed_triples=len(confirmed),
        mean_domain_size=round(mean_domain, 2),
        total_soft_weight=total_soft,
        encoding_time_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'maxsat_encoding.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2)

    print(f"\n  Encoding: {result.n_variables} vars, "
          f"{n_hard} hard clauses, {n_soft} soft clauses")
    print(f"  Saved -> {out_path}")
    print(f"\n  Step 44A.1 completed in {time.time() - t0:.1f}s")


# ---------------------------------------------------------------------------
# Step 44A.2 – Solution Enumeration
# ---------------------------------------------------------------------------

def run_maxsat_solve() -> None:
    """Step 44A.2: Enumerate optimal + near-optimal solutions."""
    t0 = time.time()
    print("=" * 70)
    print("STEP 44A.2: MaxSAT Solution Enumeration")
    print("=" * 70)

    rd = _results_dir()
    eva_to_triple = build_eva_to_triple_lookup()

    # Load inputs
    print("\n  1. Loading inputs ...")
    refine_data = _safe_load(os.path.join(rd, 'combined_refine.json'))
    if not refine_data:
        print("  [SKIP] combined_refine.json not found")
        return
    p15_assignment = refine_data.get('best_assignment', {})

    boot_data = _safe_load(os.path.join(rd, 'bootstrap_loop.json'))
    confirmed_list = boot_data.get('confirmed_triples', [])
    final_assignment = boot_data.get('final_assignment', p15_assignment)
    confirmed: Dict[str, str] = {}
    for t_key in confirmed_list:
        if t_key in final_assignment:
            confirmed[t_key] = final_assignment[t_key]

    # Reload modifier + word set for dict-hit evaluation
    mod_data = _safe_load(os.path.join(rd, 'modifier_integrate.json'))
    modifier_chars, modifier_rules = set(), {}
    if mod_data:
        modifier_chars, modifier_rules = _reconstruct_modifier_rules(mod_data)

    ref_corpus = load_reference_corpus()
    base_words = set(w.lower() for w in ref_corpus.get_combined_tokens('latin'))
    expanded_set, _ = build_expanded_word_set(base_words)

    corpus = load_corpus(verbose=False)
    all_tokens = []
    for _fol, page in corpus.pages.items():
        all_tokens.extend(page.all_tokens)
    sample_tokens = all_tokens[:2000]

    # Rebuild WCNF (same encoding as 44A.1)
    print("  2. Rebuilding WCNF ...")
    hypotheses = build_triple_phoneme_hypotheses('latin')
    all_triples = sorted(hypotheses.keys())
    free_triples = [t for t in all_triples if t not in confirmed]

    # Build bigram pairs
    bigram_counter: Counter = Counter()
    for token in all_tokens:
        chars = tokenize_eva_chars(token)
        triples = [eva_to_triple.get(ch) for ch in chars]
        triples = [t for t in triples if t]
        for i in range(len(triples) - 1):
            bigram_counter[(triples[i], triples[i + 1])] += 1
    top_bigrams = bigram_counter.most_common(1000)
    bigram_pairs = [(t1, t2, cnt) for (t1, t2), cnt in top_bigrams]

    # Reference bigram probs
    ref_tokens = ref_corpus.get_combined_tokens('latin')
    all_syls_in_domains = set()
    for domain in hypotheses.values():
        all_syls_in_domains.update(domain)

    syl_bigram_count: Counter = Counter()
    total_syl_bigrams = 0
    for word in ref_tokens[:50000]:
        word_lower = word.lower()
        syls_in_word = []
        i = 0
        while i < len(word_lower):
            if i + 1 < len(word_lower) and word_lower[i:i+2] in all_syls_in_domains:
                syls_in_word.append(word_lower[i:i+2])
                i += 2
            elif word_lower[i:i+1] in all_syls_in_domains:
                syls_in_word.append(word_lower[i:i+1])
                i += 1
            else:
                i += 1
        for j in range(len(syls_in_word) - 1):
            syl_bigram_count[(syls_in_word[j], syls_in_word[j + 1])] += 1
            total_syl_bigrams += 1

    ref_bigram_probs: Dict[Tuple[str, str], float] = {}
    if total_syl_bigrams > 0:
        for pair, cnt in syl_bigram_count.items():
            ref_bigram_probs[pair] = cnt / total_syl_bigrams

    # Signal words
    sig_data = _safe_load(os.path.join(rd, 'signal_isolation.json'))
    signal_words_raw = []
    if sig_data:
        for ws in sig_data.get('word_signals', []):
            if ws.get('is_genuine_signal', False):
                word = ws['word']
                sigma = ws.get('signal_sigma', 1.0)
                seq = []
                for t_key, syl in final_assignment.items():
                    if syl in word:
                        seq.append((t_key, syl))
                if seq:
                    signal_words_raw.append((word, sigma, [seq]))

    wcnf, var_map, _ = _build_wcnf(
        hypotheses, confirmed, free_triples,
        bigram_pairs, signal_words_raw, ref_bigram_probs,
    )

    # Enumerate solutions
    print("  3. Enumerating solutions (max 500, timeout 30 min) ...")
    from pysat.examples.rc2 import RC2

    solutions: List[SolutionEntry] = []
    optimal_cost = None
    deadline = time.time() + 1800  # 30 minute timeout

    try:
        with RC2(wcnf) as solver:
            for model in solver.enumerate():
                cost = solver.cost
                if optimal_cost is None:
                    optimal_cost = cost
                    print(f"     Optimal cost: {cost}")

                assignment = _extract_assignment(model, var_map, hypotheses)
                # Merge with confirmed
                full_assign = dict(confirmed)
                full_assign.update(assignment)

                # Compute dict-hit on sample
                decoded = _decode_corpus_r3(
                    sample_tokens, full_assign, eva_to_triple,
                    modifier_chars, modifier_rules, expanded_set,
                )
                dh = _compute_dict_hit(decoded, expanded_set)
                hamming = _hamming_distance(full_assign, final_assignment, free_triples)

                solutions.append(SolutionEntry(
                    assignment=full_assign,
                    cost=cost,
                    dict_hit=round(dh, 4),
                    hamming_from_p15=hamming,
                ))

                if len(solutions) >= 500:
                    print("     Reached 500 solution cap")
                    break
                if time.time() > deadline:
                    print("     Timeout reached (30 min)")
                    break
                if len(solutions) % 50 == 0:
                    print(f"     ... {len(solutions)} solutions enumerated")
    except Exception as e:
        print(f"     RC2 error: {e}")

    if optimal_cost is None:
        optimal_cost = 0

    # Count solutions at delta levels
    n_opt = sum(1 for s in solutions if s.cost == optimal_cost)
    threshold_1 = optimal_cost * 1.01 if optimal_cost > 0 else 1
    threshold_5 = optimal_cost * 1.05 if optimal_cost > 0 else 5
    threshold_10 = optimal_cost * 1.10 if optimal_cost > 0 else 10
    n_1pct = sum(1 for s in solutions if s.cost <= threshold_1)
    n_5pct = sum(1 for s in solutions if s.cost <= threshold_5)
    n_10pct = sum(1 for s in solutions if s.cost <= threshold_10)

    # Phase 15 cost: check if P15 assignment satisfies the instance
    p15_full = dict(confirmed)
    p15_full.update({t: final_assignment.get(t, '') for t in free_triples})
    p15_cost = -1  # unknown if not in enumerated set
    p15_rank = -1
    for idx, s in enumerate(sorted(solutions, key=lambda x: x.cost)):
        if s.hamming_from_p15 == 0:
            p15_cost = s.cost
            p15_rank = idx + 1
            break

    best_dh = max((s.dict_hit for s in solutions), default=0.0)

    result = SolutionsResult(
        n_optimal=n_opt,
        n_near_optimal_1pct=n_1pct,
        n_near_optimal_5pct=n_5pct,
        n_near_optimal_10pct=n_10pct,
        optimal_cost=optimal_cost,
        best_dict_hit=round(best_dh, 4),
        phase15_cost=p15_cost,
        phase15_rank=p15_rank,
        enumeration_time_seconds=round(time.time() - t0, 2),
    )

    # Save solutions (store top-100 only to keep file size manageable)
    top_solutions = sorted(solutions, key=lambda x: x.cost)[:100]
    out_data = {
        'summary': _convert(asdict(result)),
        'solutions': [_convert(asdict(s)) for s in top_solutions],
    }
    out_path = os.path.join(rd, 'maxsat_solutions.json')
    with open(out_path, 'w') as f:
        json.dump(out_data, f, indent=2)

    print(f"\n  {len(solutions)} solutions enumerated")
    print(f"  Optimal: {n_opt}, delta-1%: {n_1pct}, delta-5%: {n_5pct}, delta-10%: {n_10pct}")
    print(f"  Best dict-hit: {best_dh:.4f}")
    print(f"  Phase 15 rank: {p15_rank}")
    print(f"  Saved -> {out_path}")
    print(f"\n  Step 44A.2 completed in {time.time() - t0:.1f}s")


# ---------------------------------------------------------------------------
# Step 44A.3 – Landscape Characterization
# ---------------------------------------------------------------------------

def run_maxsat_landscape() -> None:
    """Step 44A.3: Characterize the solution landscape."""
    t0 = time.time()
    print("=" * 70)
    print("STEP 44A.3: MaxSAT Landscape Characterization")
    print("=" * 70)

    rd = _results_dir()

    # Load solutions
    sol_data = _safe_load(os.path.join(rd, 'maxsat_solutions.json'))
    if not sol_data:
        print("  [SKIP] maxsat_solutions.json not found")
        return

    summary = sol_data.get('summary', {})
    solutions = sol_data.get('solutions', [])
    n_1pct = summary.get('n_near_optimal_1pct', 0)

    if not solutions:
        print("  [SKIP] No solutions found")
        return

    # Load free triples list
    boot_data = _safe_load(os.path.join(rd, 'bootstrap_loop.json'))
    refine_data = _safe_load(os.path.join(rd, 'combined_refine.json'))
    p15_assignment = refine_data.get('best_assignment', {})
    final_assignment = boot_data.get('final_assignment', p15_assignment)
    confirmed_list = boot_data.get('confirmed_triples', [])

    hypotheses = build_triple_phoneme_hypotheses('latin')
    all_triples = sorted(hypotheses.keys())
    free_triples = [t for t in all_triples if t not in set(confirmed_list)]

    # ── Per-triple consensus ──
    print("\n  1. Computing per-triple consensus ...")
    per_triple_consensus: Dict[str, Dict[str, float]] = {}
    n_sol = len(solutions)
    for t_key in free_triples:
        syl_counts: Counter = Counter()
        for sol in solutions:
            assign = sol.get('assignment', {})
            syl = assign.get(t_key, '?')
            syl_counts[syl] += 1
        per_triple_consensus[t_key] = {
            syl: round(cnt / n_sol, 4)
            for syl, cnt in syl_counts.most_common()
        }

    n_landscape_confirmed = sum(
        1 for t in free_triples
        if per_triple_consensus.get(t, {}) and
        max(per_triple_consensus[t].values()) > 0.8
    )

    # ── Pairwise Hamming distances ──
    print("  2. Computing Hamming distances ...")
    import numpy as np
    n = min(len(solutions), 100)
    hamming_matrix = np.zeros((n, n), dtype=int)
    for i in range(n):
        a_i = solutions[i].get('assignment', {})
        for j in range(i + 1, n):
            a_j = solutions[j].get('assignment', {})
            dist = sum(1 for t in free_triples if a_i.get(t) != a_j.get(t))
            hamming_matrix[i, j] = dist
            hamming_matrix[j, i] = dist

    mean_hamming = float(np.mean(hamming_matrix[np.triu_indices(n, k=1)])) if n > 1 else 0.0

    # ── Basin detection via DBSCAN ──
    print("  3. Detecting basins (DBSCAN) ...")
    n_basins = 1
    try:
        from sklearn.cluster import DBSCAN
        clustering = DBSCAN(eps=2, min_samples=2, metric='precomputed')
        labels = clustering.fit_predict(hamming_matrix)
        n_basins = len(set(labels) - {-1})
        if n_basins == 0:
            n_basins = 1  # all outliers = 1 diffuse basin
        print(f"     DBSCAN found {n_basins} basins")
    except Exception as e:
        print(f"     DBSCAN failed: {e}, defaulting to 1 basin")

    # ── Landscape classification ──
    if n_1pct > 100:
        classification = "FLAT"
    elif n_1pct < 10:
        classification = "PEAKED"
    else:
        classification = "BASINED"

    # Check if Phase 15 is in optimal set
    p15_in_optimal = summary.get('phase15_rank', -1) == 1

    result = LandscapeResult(
        classification=classification,
        n_solutions=len(solutions),
        n_basins=n_basins,
        per_triple_consensus=per_triple_consensus,
        n_landscape_confirmed=n_landscape_confirmed,
        mean_hamming=round(mean_hamming, 2),
        phase15_in_optimal=p15_in_optimal,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'maxsat_landscape.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2)

    print(f"\n  Classification: {classification}")
    print(f"  Basins: {n_basins}, Mean Hamming: {mean_hamming:.2f}")
    print(f"  Landscape-confirmed triples: {n_landscape_confirmed}/{len(free_triples)}")
    print(f"  Phase 15 in optimal: {p15_in_optimal}")
    print(f"  Saved -> {out_path}")
    print(f"\n  Step 44A.3 completed in {time.time() - t0:.1f}s")


# ---------------------------------------------------------------------------
# Step 44A.4 – Cross-Validation
# ---------------------------------------------------------------------------

def run_maxsat_validate() -> None:
    """Step 44A.4: Validate MaxSAT best vs Phase 15."""
    t0 = time.time()
    print("=" * 70)
    print("STEP 44A.4: MaxSAT Validation")
    print("=" * 70)

    rd = _results_dir()
    eva_to_triple = build_eva_to_triple_lookup()

    # Load solutions
    sol_data = _safe_load(os.path.join(rd, 'maxsat_solutions.json'))
    if not sol_data or not sol_data.get('solutions'):
        print("  [SKIP] maxsat_solutions.json not found or empty")
        return
    solutions = sol_data['solutions']

    # Load Phase 15 baseline
    refine_data = _safe_load(os.path.join(rd, 'combined_refine.json'))
    p15_assignment = refine_data.get('best_assignment', {})
    boot_data = _safe_load(os.path.join(rd, 'bootstrap_loop.json'))
    final_assignment = boot_data.get('final_assignment', p15_assignment)

    # Load modifiers + word set
    mod_data = _safe_load(os.path.join(rd, 'modifier_integrate.json'))
    modifier_chars, modifier_rules = set(), {}
    if mod_data:
        modifier_chars, modifier_rules = _reconstruct_modifier_rules(mod_data)

    ref_corpus = load_reference_corpus()
    base_words = set(w.lower() for w in ref_corpus.get_combined_tokens('latin'))
    expanded_set, _ = build_expanded_word_set(base_words)

    corpus = load_corpus(verbose=False)
    all_tokens = []
    for _fol, page in corpus.pages.items():
        all_tokens.extend(page.all_tokens)

    # Decode Phase 15 baseline
    print("\n  1. Computing Phase 15 baseline dict-hit ...")
    p15_decoded = _decode_corpus_r3(
        all_tokens, final_assignment, eva_to_triple,
        modifier_chars, modifier_rules, expanded_set,
    )
    p15_dh = _compute_dict_hit(p15_decoded, expanded_set)
    print(f"     Phase 15 dict-hit: {p15_dh:.4f}")

    # Find best MaxSAT solution
    best_sol = min(solutions, key=lambda s: s.get('cost', float('inf')))
    best_assign = best_sol.get('assignment', {})

    # Full corpus decode for best
    print("  2. Decoding full corpus with best MaxSAT solution ...")
    best_decoded = _decode_corpus_r3(
        all_tokens, best_assign, eva_to_triple,
        modifier_chars, modifier_rules, expanded_set,
    )
    best_dh = _compute_dict_hit(best_decoded, expanded_set)
    print(f"     Best MaxSAT dict-hit: {best_dh:.4f}")

    # Triple changes
    hypotheses = build_triple_phoneme_hypotheses('latin')
    all_triples = sorted(hypotheses.keys())
    confirmed_list = boot_data.get('confirmed_triples', [])
    free_triples = [t for t in all_triples if t not in set(confirmed_list)]

    changed = []
    for t in free_triples:
        old = final_assignment.get(t, '?')
        new = best_assign.get(t, '?')
        if old != new:
            changed.append({'triple': t, 'old': old, 'new': new})

    delta = best_dh - p15_dh
    gate = best_dh > p15_dh and best_dh > 0.40

    if delta > 0.01:
        verdict = "MAXSAT_IMPROVED"
    elif delta > -0.01:
        verdict = "MAXSAT_COMPARABLE"
    else:
        verdict = "MAXSAT_WORSE"

    result = ValidationResult(
        best_maxsat_dict_hit=round(best_dh, 4),
        best_maxsat_assignment=best_assign,
        phase15_dict_hit=round(p15_dh, 4),
        delta_dict_hit=round(delta, 4),
        n_triples_changed=len(changed),
        changed_triples=changed,
        gate_passed=gate,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'maxsat_validation.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2)

    print(f"\n  Delta dict-hit: {delta:+.4f}")
    print(f"  Triples changed: {len(changed)}")
    print(f"  Verdict: {verdict}")
    print(f"  Saved -> {out_path}")
    print(f"\n  Step 44A.4 completed in {time.time() - t0:.1f}s")


# ---------------------------------------------------------------------------
# Track A runner
# ---------------------------------------------------------------------------

def run_track_a() -> None:
    """Run all Track A steps."""
    run_maxsat_encode()
    print("\n" + "=" * 70 + "\n")
    run_maxsat_solve()
    print("\n" + "=" * 70 + "\n")
    run_maxsat_landscape()
    print("\n" + "=" * 70 + "\n")
    run_maxsat_validate()
