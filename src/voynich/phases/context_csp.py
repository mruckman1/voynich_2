"""
Phase 13.3 – Context-Aware CSP Solver
======================================
Builds and optimizes a context-aware phonetic table by adding position/
adjacency-dependent reading rules on top of the Phase 11 fixed-value
assignment.

Two search modes:
  Version A (rule-constrained): Only allow values extracted as significant
    rules in Phase 13.2.  Small search space (~3^k for k context-sensitive
    cells), exhaustively enumerable.

  Version B (free search): Allow any value in the inventory at each context.
    Larger search space, requires beam search.  May find rules that error
    analysis missed.

Selectivity gate: context-aware CE must be < baseline CE at 1.5× or better.
"""

import json
import os
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import build_eva_to_cell_lookup, load_corpus
from voynich.core.reference import load_reference_corpus, build_cv_syllable_table
from voynich.core.stats import build_ngram_lm, cross_entropy_lm
from voynich.phases.csp_constraints import (
    build_phoneme_inventory,
    score_cross_entropy,
    score_dict_hit_rate,
)
from voynich.phases.csp_solver import (
    _convert,
    decode_token,
)
from voynich.phases.csp_diagnosis import (
    TokenDiagnosis,
    categorize_token,
    _bucket_by_length,
    _get_cells_used,
)
from voynich.phases.error_patterns import (
    _classify_nucleus,
    build_contextualized_diagnoses,
    build_error_catalog,
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ContextAwareAssignment:
    """A phonetic table with per-context values for sensitive cells."""
    base_mapping: Dict[str, str]              # cell → default syllable
    context_rules: Dict[str, Dict[str, str]]  # cell → {context → syllable}
    dict_hit_rate: float
    cross_entropy: float
    selectivity: float                        # vs random assignments
    n_rules_applied: int
    improvement_over_baseline: float


@dataclass
class ContextCSPResult:
    """Full Phase 13.3 output."""
    version_a_result: Optional[Dict]    # Rule-constrained search
    version_b_result: Optional[Dict]    # Free search
    best_version: str                   # 'A', 'B', or 'baseline'
    best_dict_hit: float
    best_ce: float
    best_selectivity: float
    best_context_rules: Dict[str, Dict[str, str]]
    baseline_dict_hit: float
    baseline_ce: float
    improvement: float
    gate_passed: bool
    gate_message: str


# ---------------------------------------------------------------------------
# Decode corpus with context rules
# ---------------------------------------------------------------------------

def decode_corpus_contextual(
    tokens: List[str],
    base_assignment: Dict[str, str],
    context_rules: Dict[str, Dict[str, str]],
    eva_to_cell: Dict[str, str],
    max_tokens: int = 2000,
) -> List[str]:
    """Decode all tokens using context-aware rules."""
    return [
        decode_token(t, base_assignment, eva_to_cell, context_rules)
        for t in tokens[:max_tokens]
    ]


def score_context_assignment(
    base_assignment: Dict[str, str],
    context_rules: Dict[str, Dict[str, str]],
    voynich_tokens: List[str],
    eva_to_cell: Dict[str, str],
    ref_word_set: set,
    lm: Dict,
    max_tokens: int = 2000,
) -> Tuple[float, float]:
    """Score a context-aware assignment. Returns (dict_hit, cross_entropy)."""
    decoded = decode_corpus_contextual(
        voynich_tokens, base_assignment, context_rules, eva_to_cell, max_tokens,
    )
    n_hits = sum(1 for w in decoded if w in ref_word_set)
    n_valid = sum(1 for w in decoded if w and '?' not in w)
    dict_hit = n_hits / max(n_valid, 1)

    # Cross-entropy: average char-level CE over decoded tokens
    ce_vals = []
    for w in decoded:
        if w and '?' not in w and len(w) >= 2:
            try:
                v = cross_entropy_lm(w, lm)
                if v is not None and v < 99.0:
                    ce_vals.append(v)
            except Exception:
                pass
    ce = sum(ce_vals) / len(ce_vals) if ce_vals else 99.0

    return dict_hit, ce


def compute_selectivity(
    base_assignment: Dict[str, str],
    context_rules: Dict[str, Dict[str, str]],
    voynich_tokens: List[str],
    eva_to_cell: Dict[str, str],
    ref_word_set: set,
    lm: Dict,
    inventory: Any,
    n_random: int = 100,
    seed: int = 42,
) -> Tuple[float, float]:
    """Compute selectivity of context-aware assignment vs random context rules.

    Returns (real_dict_hit, selectivity_ratio).
    """
    real_dict_hit, real_ce = score_context_assignment(
        base_assignment, context_rules, voynich_tokens, eva_to_cell,
        ref_word_set, lm,
    )

    # Generate random context-rule assignments
    rng = random.Random(seed)
    cv_syllables = inventory.cv_syllables
    context_sensitive_cells = list(context_rules.keys())
    context_types = ['word_initial', 'word_final', 'after_vowel']

    null_hits: List[float] = []
    for _ in range(n_random):
        null_rules: Dict[str, Dict[str, str]] = {}
        for cell in context_sensitive_cells:
            null_rules[cell] = {}
            for ctx in context_types:
                null_rules[cell][ctx] = rng.choice(cv_syllables)
        null_hit, _ = score_context_assignment(
            base_assignment, null_rules, voynich_tokens, eva_to_cell, ref_word_set, lm, 200,
        )
        null_hits.append(null_hit)

    null_mean = sum(null_hits) / max(len(null_hits), 1)
    selectivity = real_dict_hit / max(null_mean, 1e-9)
    return real_dict_hit, selectivity


# ---------------------------------------------------------------------------
# Version A: Rule-constrained search
# ---------------------------------------------------------------------------

def search_version_a(
    rules: List[Dict],
    base_assignment: Dict[str, str],
    voynich_tokens: List[str],
    eva_to_cell: Dict[str, str],
    ref_word_set: set,
    lm: Dict,
    inventory: Any,
    baseline_dict_hit: float,
) -> Tuple[Dict[str, Dict[str, str]], float, float]:
    """Exhaustive search over rule-constrained context values.

    Each context-sensitive cell can use its extracted rule value or the
    default (base_assignment) value.  For k cells with ≤4 context types each,
    the search space is manageable.

    Returns (best_context_rules, best_dict_hit, best_ce).
    """
    # Group rules by cell
    rules_by_cell: Dict[str, List[Dict]] = defaultdict(list)
    for r in rules:
        rules_by_cell[r['cell_key']].append(r)

    context_sensitive_cells = list(rules_by_cell.keys())
    print(f"  Version A: {len(context_sensitive_cells)} context-sensitive cells")

    if not context_sensitive_cells:
        return {}, baseline_dict_hit, 99.0

    # Build all candidate context-rule sets for each cell
    # Each option is either {} (use base) or {context: corrected}
    cell_options: List[List[Dict[str, str]]] = []
    for cell in context_sensitive_cells:
        options: List[Dict[str, str]] = [{}]  # option 0: use base for all contexts
        for rule in rules_by_cell[cell]:
            options.append({rule['context']: rule['corrected']})
        # Also try combining rules for the same cell
        if len(rules_by_cell[cell]) > 1:
            combined = {r['context']: r['corrected'] for r in rules_by_cell[cell]}
            options.append(combined)
        cell_options.append(options)

    # Count total combinations
    total = 1
    for opts in cell_options:
        total *= len(opts)
    print(f"  Version A: {total} combinations to evaluate")

    best_context_rules: Dict[str, Dict[str, str]] = {}
    best_dict_hit = baseline_dict_hit
    best_ce = 99.0

    if total <= 10000:
        # Exhaustive search
        from itertools import product as iproduct
        evaluated = 0
        for combo in iproduct(*cell_options):
            context_rules: Dict[str, Dict[str, str]] = {}
            for cell, ctx_map in zip(context_sensitive_cells, combo):
                if ctx_map:
                    context_rules[cell] = ctx_map

            dh, ce = score_context_assignment(
                base_assignment, context_rules, voynich_tokens, eva_to_cell,
                ref_word_set, lm, 500,
            )
            if dh > best_dict_hit:
                best_dict_hit = dh
                best_ce = ce
                best_context_rules = context_rules
            evaluated += 1
            if evaluated % 500 == 0:
                print(f"    Evaluated {evaluated}/{total}, best={best_dict_hit:.1%}")
    else:
        # Beam search over combinations
        print(f"  Version A: large space ({total}), using beam search (width=20)")
        beam: List[Tuple[Dict[str, Dict[str, str]], float]] = [({}, baseline_dict_hit)]

        for i, (cell, opts) in enumerate(zip(context_sensitive_cells, cell_options)):
            new_beam: List[Tuple[Dict[str, Dict[str, str]], float]] = []
            for partial_rules, _ in beam:
                for ctx_map in opts:
                    candidate = dict(partial_rules)
                    if ctx_map:
                        candidate[cell] = ctx_map
                    dh, ce = score_context_assignment(
                        base_assignment, candidate, voynich_tokens, eva_to_cell,
                        ref_word_set, lm, 300,
                    )
                    new_beam.append((candidate, dh))
            new_beam.sort(key=lambda x: x[1], reverse=True)
            beam = new_beam[:20]

        best_context_rules, best_dict_hit = beam[0]
        if best_context_rules:
            _, best_ce = score_context_assignment(
                base_assignment, best_context_rules, voynich_tokens, eva_to_cell,
                ref_word_set, lm,
            )

    print(f"  Version A best: dict_hit={best_dict_hit:.1%} (Δ={best_dict_hit - baseline_dict_hit:+.1%})")
    return best_context_rules, best_dict_hit, best_ce


# ---------------------------------------------------------------------------
# Version B: Free search
# ---------------------------------------------------------------------------

def search_version_b(
    context_sensitive_cells: List[str],
    base_assignment: Dict[str, str],
    voynich_tokens: List[str],
    eva_to_cell: Dict[str, str],
    ref_word_set: set,
    lm: Dict,
    inventory: Any,
    baseline_dict_hit: float,
    beam_width: int = 20,
    seed: int = 42,
) -> Tuple[Dict[str, Dict[str, str]], float, float]:
    """Beam search over free context-rule values.

    All cells flagged as context-sensitive can take any value in the inventory
    for each context type.

    Returns (best_context_rules, best_dict_hit, best_ce).
    """
    cv_syllables = inventory.cv_syllables[:20]  # Limit to top-20 most frequent
    context_types = ['word_initial', 'word_final', 'after_vowel']

    if not context_sensitive_cells:
        return {}, baseline_dict_hit, 99.0

    print(f"  Version B: {len(context_sensitive_cells)} cells × "
          f"{len(context_types)} contexts × {len(cv_syllables)} candidates")

    rng = random.Random(seed)

    # Start with rule-constrained best (if available) as initial beam
    # Plus random restarts for diversity
    beam: List[Tuple[Dict[str, Dict[str, str]], float]] = [({}, baseline_dict_hit)]

    for _ in range(5):  # 5 random initializations
        init_rules: Dict[str, Dict[str, str]] = {}
        for cell in context_sensitive_cells:
            init_rules[cell] = {ctx: rng.choice(cv_syllables) for ctx in context_types}
        dh, _ = score_context_assignment(
            base_assignment, init_rules, voynich_tokens, eva_to_cell,
            ref_word_set, lm, 200,
        )
        beam.append((init_rules, dh))

    best_context_rules: Dict[str, Dict[str, str]] = {}
    best_dict_hit = baseline_dict_hit

    # Iterate: for each cell, try all candidates at each context
    max_iterations = 3
    for iteration in range(max_iterations):
        improved = False
        for cell in context_sensitive_cells:
            new_beam: List[Tuple[Dict[str, Dict[str, str]], float]] = []
            for partial_rules, _ in beam[:beam_width]:
                for ctx in context_types:
                    for syl in cv_syllables:
                        candidate = {k: dict(v) for k, v in partial_rules.items()}
                        if cell not in candidate:
                            candidate[cell] = {}
                        candidate[cell][ctx] = syl
                        dh, _ = score_context_assignment(
                            base_assignment, candidate, voynich_tokens, eva_to_cell,
                            ref_word_set, lm, 200,
                        )
                        new_beam.append((candidate, dh))
            new_beam.sort(key=lambda x: x[1], reverse=True)
            if new_beam and new_beam[0][1] > best_dict_hit:
                best_dict_hit = new_beam[0][1]
                best_context_rules = new_beam[0][0]
                improved = True
            beam = new_beam[:beam_width]

        print(f"  Version B iteration {iteration + 1}: best={best_dict_hit:.1%}")
        if not improved:
            break

    if best_context_rules:
        _, best_ce = score_context_assignment(
            base_assignment, best_context_rules, voynich_tokens, eva_to_cell,
            ref_word_set, lm,
        )
    else:
        best_ce = 99.0

    print(f"  Version B best: dict_hit={best_dict_hit:.1%} (Δ={best_dict_hit - baseline_dict_hit:+.1%})")
    return best_context_rules, best_dict_hit, best_ce


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_context_csp() -> Dict:
    """Phase 13.3: Optimize context-aware phonetic table.

    Loads rules from rule_extraction.json and the CSP assignment,
    runs Version A (rule-constrained) and Version B (free search),
    and selects the best result.
    """
    print("=" * 70)
    print("PHASE 13.3: Context-Aware CSP Solver")
    print("=" * 70)

    t0 = time.time()
    rd = _results_dir()

    # ------------------------------------------------------------------
    # 1. Load dependencies
    # ------------------------------------------------------------------
    re_path = os.path.join(rd, 'rule_extraction.json')
    if not os.path.exists(re_path):
        print("  [SKIP] rule_extraction.json not found — run extract-rules first")
        return {'verdict': 'skipped', 'reason': 'no_rule_extraction'}

    with open(re_path) as f:
        re_data = json.load(f)

    rules = re_data.get('rules', [])
    baseline_dict_hit = re_data.get('baseline_dict_hit', 0.0)
    print(f"  Rules loaded: {len(rules)}")
    print(f"  Baseline dict_hit: {baseline_dict_hit:.1%}")

    # Load assignment
    for fname in ('recalibrated_csp.json', 'csp_final.json', 'csp_decode.json'):
        candidate = os.path.join(rd, fname)
        if os.path.exists(candidate):
            with open(candidate) as f:
                decode_data = json.load(f)
            break
    else:
        return {'verdict': 'skipped', 'reason': 'no_csp_result'}

    if 'best_assignment' in decode_data:
        best_assignment: Dict[str, str] = decode_data['best_assignment']
        eva_to_cell_map: Dict[str, str] = decode_data.get('eva_to_cell_mapping', {})
    elif 'language_results' in decode_data:
        lat = decode_data['language_results'].get('latin', {})
        best_assignment = lat.get('best_assignment', {})
        eva_to_cell_map = decode_data.get('eva_to_cell_mapping', {})
    else:
        return {'verdict': 'skipped', 'reason': 'no_assignment'}

    cv_path = os.path.join(rd, 'cv_labels.json')
    with open(cv_path) as f:
        cv_labels = json.load(f)

    corpus = load_corpus(verbose=False)
    ref_corpus = load_reference_corpus(verbose=False)
    corpus_tokens = corpus.get_tokens(language='A', paragraph_only=True)[:1500]
    ref_tokens = ref_corpus.get_combined_tokens('latin')
    ref_word_set: set = set(ref_tokens[:50000])
    inventory = build_phoneme_inventory('latin', ref_corpus)
    lm = build_ngram_lm(ref_tokens[:10000], order=3, smoothing=0.01)

    if eva_to_cell_map:
        eva_to_cell = eva_to_cell_map
    else:
        eva_to_cell = build_eva_to_cell_lookup(cv_labels)

    # ------------------------------------------------------------------
    # 2. Get context-sensitive cells
    # ------------------------------------------------------------------
    ep_path = os.path.join(rd, 'error_patterns.json')
    context_sensitive_cells: List[str] = []
    if os.path.exists(ep_path):
        with open(ep_path) as f:
            ep_data = json.load(f)
        # Cells with significant position or adjacency dependence
        for test in ep_data.get('position_tests', []):
            if test.get('significant', False):
                context_sensitive_cells.append(test['cell_key'])
        for test in ep_data.get('adjacency_tests', []):
            if test.get('significant', False):
                ck = test['cell_key']
                if ck not in context_sensitive_cells:
                    context_sensitive_cells.append(ck)
    # Fallback: use cells from extracted rules
    if not context_sensitive_cells:
        context_sensitive_cells = list({r['cell_key'] for r in rules})
    print(f"  Context-sensitive cells: {len(context_sensitive_cells)}")

    # ------------------------------------------------------------------
    # 3. Version A: Rule-constrained search
    # ------------------------------------------------------------------
    print("\n  Running Version A (rule-constrained)...")
    va_rules, va_dict_hit, va_ce = search_version_a(
        rules, best_assignment, corpus_tokens, eva_to_cell,
        ref_word_set, lm, inventory, baseline_dict_hit,
    )

    # ------------------------------------------------------------------
    # 4. Version B: Free search (only if Version A shows < 15% dict_hit)
    # ------------------------------------------------------------------
    vb_rules: Dict = {}
    vb_dict_hit = baseline_dict_hit
    vb_ce = 99.0

    if va_dict_hit < 0.15:
        print("\n  Version A < 15%, running Version B (free search)...")
        vb_rules, vb_dict_hit, vb_ce = search_version_b(
            context_sensitive_cells, best_assignment, corpus_tokens, eva_to_cell,
            ref_word_set, lm, inventory, baseline_dict_hit,
        )
    else:
        print(f"\n  Version A ≥ 15%, skipping Version B.")

    # ------------------------------------------------------------------
    # 5. Select best result
    # ------------------------------------------------------------------
    if va_dict_hit >= vb_dict_hit and va_dict_hit >= baseline_dict_hit:
        best_version = 'A'
        best_context_rules = va_rules
        best_dict_hit = va_dict_hit
        best_ce = va_ce
    elif vb_dict_hit > baseline_dict_hit:
        best_version = 'B'
        best_context_rules = vb_rules
        best_dict_hit = vb_dict_hit
        best_ce = vb_ce
    else:
        best_version = 'baseline'
        best_context_rules = {}
        best_dict_hit = baseline_dict_hit
        best_ce = 99.0

    improvement = best_dict_hit - baseline_dict_hit
    print(f"\n  Best version: {best_version}")
    print(f"  Best dict_hit: {best_dict_hit:.1%} (baseline {baseline_dict_hit:.1%}, Δ={improvement:+.1%})")

    # Compute selectivity for best result
    if best_context_rules:
        print("  Computing selectivity vs random context rules...")
        _, selectivity = compute_selectivity(
            best_assignment, best_context_rules, corpus_tokens, eva_to_cell,
            ref_word_set, lm, inventory, n_random=50,
        )
    else:
        selectivity = 1.0

    print(f"  Selectivity: {selectivity:.2f}x")

    gate_passed = best_dict_hit >= 0.13 and selectivity >= 1.5
    if gate_passed:
        gate_message = (f"PASS: dict_hit={best_dict_hit:.1%} ≥ 13% AND "
                        f"selectivity={selectivity:.2f}x ≥ 1.5x.")
    elif best_dict_hit >= 0.13:
        gate_message = (f"PARTIAL: dict_hit={best_dict_hit:.1%} ≥ 13% but "
                        f"selectivity={selectivity:.2f}x < 1.5x. Rules may be overfitting.")
    else:
        gate_message = (f"FAIL: dict_hit={best_dict_hit:.1%} < 13%. "
                        "Context rules cannot break the ceiling via this approach.")

    print(f"  Gate 13.3: {'PASS ✓' if gate_passed else 'FAIL ✗'}")

    # ------------------------------------------------------------------
    # 6. Save
    # ------------------------------------------------------------------
    result = ContextCSPResult(
        version_a_result={
            'context_rules': va_rules,
            'dict_hit': round(va_dict_hit, 4),
            'cross_entropy': round(va_ce, 4),
        },
        version_b_result={
            'context_rules': vb_rules,
            'dict_hit': round(vb_dict_hit, 4),
            'cross_entropy': round(vb_ce, 4),
        } if vb_dict_hit > baseline_dict_hit else None,
        best_version=best_version,
        best_dict_hit=round(best_dict_hit, 4),
        best_ce=round(best_ce, 4) if best_ce < 90 else None,
        best_selectivity=round(selectivity, 3),
        best_context_rules=best_context_rules,
        baseline_dict_hit=round(baseline_dict_hit, 4),
        baseline_ce=None,
        improvement=round(improvement, 4),
        gate_passed=gate_passed,
        gate_message=gate_message,
    )

    out_path = os.path.join(rd, 'context_csp.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2)

    elapsed = time.time() - t0
    print(f"\n  Saved to {out_path} ({elapsed:.1f}s)")
    return _convert(asdict(result))
