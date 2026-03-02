"""
Phase 13.2 – Reading Rule Extraction and Formalization
=======================================================
Converts the significant context-dependent error patterns found in Phase 13.1
into formal reading rules of the form:

    CELL → VALUE / CONTEXT

Where CONTEXT is one of:
  - 'word_initial'   (cell appears at position 0 in the token)
  - 'word_final'     (cell appears at the last position in the token)
  - 'after_vowel'    (preceded by a vowel-dominant cell)
  - 'before_vowel'   (followed by a vowel-dominant cell)
  - 'default'        (the fixed-table value, everywhere else)

Each rule is ranked by "power" — the fraction of total near-miss tokens it
converts to exact dictionary hits when applied in isolation.

Rules are then applied greedily (highest power first) to track cumulative
dict_hit improvement.

Gate 13.2: cumulative dict_hit ≥ 15% after applying validated rules.
"""

import json
import os
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import build_eva_to_cell_lookup, load_corpus
from voynich.core.reference import (
    ROMANCE_PHONOLOGICAL_PROCESSES,
    load_reference_corpus,
)
from voynich.phases.csp_constraints import build_phoneme_inventory
from voynich.phases.csp_solver import _convert, decode_token
from voynich.phases.csp_diagnosis import (
    categorize_token,
    _edit_distance,
    _bucket_by_length,
    _nearest_word,
    _get_cells_used,
)
from voynich.phases.error_patterns import (
    ContextualizedDiagnosis,
    ErrorRecord,
    _classify_nucleus,
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ReadingRule:
    """A single context-dependent reading rule."""
    rule_id: str
    cell_key: str
    cv_label: str
    context: str           # 'word_initial'|'word_final'|'after_vowel'|'before_vowel'
    produced: str          # Fixed-table value (wrong in this context)
    corrected: str         # Rule-prescribed correction
    coverage: float        # Fraction of errors in this context fixed by this rule
    n_errors_in_context: int
    power: float           # Fraction of total near-misses converted to hits
    plausibility: str      # 'high'|'moderate'|'low'
    linguistic_basis: str  # Description of the phonological process


@dataclass
class RuleExtractionResult:
    """Full Phase 13.2 output."""
    n_near_miss_tokens: int
    n_cells_with_rules: int
    rules: List[Dict]
    cumulative_dict_hit: List[float]    # [baseline, +rule1, +rule2, ...]
    cumulative_rule_ids: List[str]
    baseline_dict_hit: float
    final_dict_hit: float
    total_improvement: float
    gate_passed: bool
    gate_message: str


# ---------------------------------------------------------------------------
# Rule extraction
# ---------------------------------------------------------------------------

def _context_label(position: str, predecessor: str) -> str:
    """Convert position/predecessor pair to a rule context label."""
    if position == 'initial':
        return 'word_initial'
    if position == 'final':
        return 'word_final'
    # Medial: check predecessor
    if predecessor != 'NONE':
        pred_class = _classify_nucleus(predecessor)
        if pred_class == 'vowel':
            return 'after_vowel'
        else:
            return 'before_vowel'
    return 'default'


def _check_plausibility(produced: str, corrected: str) -> Tuple[str, str]:
    """Look up linguistic plausibility in the ROMANCE_PHONOLOGICAL_PROCESSES table."""
    processes = ROMANCE_PHONOLOGICAL_PROCESSES.get(produced, {})
    if corrected in processes:
        proc = processes[corrected]
        return proc.get('naturality', 'low'), proc.get('description', '')
    # Also check reverse direction
    processes_rev = ROMANCE_PHONOLOGICAL_PROCESSES.get(corrected, {})
    if produced in processes_rev:
        proc = processes_rev[produced]
        return 'moderate', f"Reverse: {proc.get('description', '')}"
    return 'low', 'Not found in Romance phonological process catalogue'


def extract_rules(
    position_tests: List[Dict],
    adjacency_tests: List[Dict],
    error_catalog: List[ErrorRecord],
    assignment: Dict[str, str],
    cv_labels: Dict,
    min_coverage: float = 0.40,
    min_errors: int = 8,
) -> List[ReadingRule]:
    """Extract reading rules from significant context tests.

    For each significant cell-context pair:
    1. Find the majority correction (most common needed phoneme).
    2. Require coverage > min_coverage (40%).
    3. Assess linguistic plausibility.
    """
    rules: List[ReadingRule] = []
    rule_counter = 0

    # Group error catalog by (cell, context)
    errors_by_cell_context: Dict[Tuple[str, str], List[ErrorRecord]] = defaultdict(list)
    for e in error_catalog:
        ctx = _context_label(e.position, e.predecessor)
        if e.needed:
            errors_by_cell_context[(e.cell_key, ctx)].append(e)

    # Extract rules from significant position tests
    for test in position_tests:
        if not test.get('significant', False):
            continue
        cell_key = test['cell_key']
        cv_label = test.get('cv_label', '?')
        produced = assignment.get(cell_key, '?')

        for pos, top_correction in test.get('top_correction_by_position', {}).items():
            ctx = _context_label(pos, 'NONE' if pos == 'initial' else 'vowel')
            errors = errors_by_cell_context.get((cell_key, ctx), [])

            if len(errors) < min_errors:
                continue

            # Count corrections in this context
            correction_counts = Counter(e.needed for e in errors if e.needed)
            total_in_ctx = sum(correction_counts.values())
            if total_in_ctx == 0:
                continue

            best_correction = correction_counts.most_common(1)[0][0]
            best_count = correction_counts.most_common(1)[0][1]
            coverage = best_count / total_in_ctx

            if coverage < min_coverage:
                continue
            if best_correction == produced:
                continue  # No change

            plausibility, basis = _check_plausibility(produced, best_correction)

            rule_counter += 1
            rules.append(ReadingRule(
                rule_id=f"R{rule_counter:02d}_{cv_label}_{ctx[:4]}",
                cell_key=cell_key,
                cv_label=cv_label,
                context=ctx,
                produced=produced,
                corrected=best_correction,
                coverage=round(coverage, 3),
                n_errors_in_context=total_in_ctx,
                power=0.0,  # Computed later
                plausibility=plausibility,
                linguistic_basis=basis,
            ))

    # Extract rules from significant adjacency tests
    for test in adjacency_tests:
        if not test.get('significant', False):
            continue
        cell_key = test['cell_key']
        cv_label = test.get('cv_label', '?')
        produced = assignment.get(cell_key, '?')

        for pred_class, top_correction in test.get('top_correction_by_predecessor', {}).items():
            if pred_class == 'NONE':
                ctx = 'word_initial'
            elif pred_class == 'vowel':
                ctx = 'after_vowel'
            else:
                ctx = 'before_vowel'

            errors = errors_by_cell_context.get((cell_key, ctx), [])
            if len(errors) < min_errors:
                continue

            correction_counts = Counter(e.needed for e in errors if e.needed)
            total_in_ctx = sum(correction_counts.values())
            if total_in_ctx == 0:
                continue

            best_correction = correction_counts.most_common(1)[0][0]
            best_count = correction_counts.most_common(1)[0][1]
            coverage = best_count / total_in_ctx

            if coverage < min_coverage:
                continue
            if best_correction == produced:
                continue

            plausibility, basis = _check_plausibility(produced, best_correction)

            # Check for duplicates
            already_exists = any(
                r.cell_key == cell_key and r.context == ctx and r.corrected == best_correction
                for r in rules
            )
            if already_exists:
                continue

            rule_counter += 1
            rules.append(ReadingRule(
                rule_id=f"R{rule_counter:02d}_{cv_label}_{ctx[:4]}",
                cell_key=cell_key,
                cv_label=cv_label,
                context=ctx,
                produced=produced,
                corrected=best_correction,
                coverage=round(coverage, 3),
                n_errors_in_context=total_in_ctx,
                power=0.0,
                plausibility=plausibility,
                linguistic_basis=basis,
            ))

    return rules


# ---------------------------------------------------------------------------
# Rule power scoring
# ---------------------------------------------------------------------------

def _apply_single_rule_decode(
    token: str,
    cells_used: List[str],
    cell_positions: List[str],
    cell_predecessors: List[str],
    base_assignment: Dict[str, str],
    rule: ReadingRule,
    eva_to_cell: Dict[str, str],
) -> str:
    """Decode a token applying one context rule to the target cell."""
    from voynich.core.corpus import tokenize_eva_chars
    chars = tokenize_eva_chars(token)
    parts: List[str] = []
    for ci, ch in enumerate(chars):
        cell = eva_to_cell.get(ch)
        if not cell:
            parts.append('?')
            continue
        # Determine context for this cell
        pos = cell_positions[ci] if ci < len(cell_positions) else 'medial'
        pred = cell_predecessors[ci] if ci < len(cell_predecessors) else 'NONE'
        ctx = _context_label(pos, pred)

        if cell == rule.cell_key and ctx == rule.context:
            parts.append(rule.corrected)
        else:
            syl = base_assignment.get(cell, '?')
            parts.append(syl)
    return ''.join(parts)


def score_rule_power(
    rule: ReadingRule,
    ctx_diagnoses: List[ContextualizedDiagnosis],
    base_assignment: Dict[str, str],
    eva_to_cell: Dict[str, str],
    ref_word_set: set,
) -> float:
    """Compute how many near-miss tokens this rule converts to dict hits."""
    near_miss_count = sum(1 for d in ctx_diagnoses if d.category == 'NEAR_MISS')
    if near_miss_count == 0:
        return 0.0

    n_converted = 0
    for d in ctx_diagnoses:
        if d.category != 'NEAR_MISS':
            continue
        new_decoded = _apply_single_rule_decode(
            d.voynich_token, d.cells_used, d.cell_positions,
            d.cell_predecessors, base_assignment, rule, eva_to_cell,
        )
        if new_decoded in ref_word_set:
            n_converted += 1

    return n_converted / near_miss_count


# ---------------------------------------------------------------------------
# Greedy rule application
# ---------------------------------------------------------------------------

def apply_rules_greedy(
    rules: List[ReadingRule],
    corpus_tokens: List[str],
    ctx_diagnoses: List[ContextualizedDiagnosis],
    base_assignment: Dict[str, str],
    eva_to_cell: Dict[str, str],
    ref_word_set: set,
    ref_words_by_len: Dict,
    inventory: Any,
    baseline_dict_hit: float,
) -> Tuple[List[ReadingRule], List[float], List[str]]:
    """Apply rules greedily from highest power to lowest.

    Returns (sorted_rules_with_power, cumulative_dict_hits, rule_ids_applied).
    """
    # Score all rules first
    print(f"  Scoring {len(rules)} rules for power...")
    for rule in rules:
        rule.power = round(score_rule_power(
            rule, ctx_diagnoses, base_assignment, eva_to_cell, ref_word_set,
        ), 4)
        print(f"    {rule.rule_id}: {rule.produced}→{rule.corrected} / {rule.context}"
              f"  coverage={rule.coverage:.1%}  power={rule.power:.4f}  [{rule.plausibility}]")

    # Sort by power descending
    rules_sorted = sorted(rules, key=lambda r: r.power, reverse=True)

    # Greedy application: maintain current context_rules dict and re-decode
    current_context_rules: Dict[str, Dict[str, str]] = {}  # cell → {context → corrected}
    cumulative_hits: List[float] = [baseline_dict_hit]
    rule_ids_applied: List[str] = []

    for rule in rules_sorted:
        if rule.power < 0.005:
            print(f"  Stopping: rule power {rule.power:.4f} < 0.005 threshold")
            break

        # Add rule to current set
        if rule.cell_key not in current_context_rules:
            current_context_rules[rule.cell_key] = {}
        current_context_rules[rule.cell_key][rule.context] = rule.corrected

        # Re-score with current rule set applied
        n_hits = 0
        n_decoded = 0
        for d in ctx_diagnoses:
            new_dec = _decode_with_context_rules(
                d.voynich_token, d.cells_used, d.cell_positions, d.cell_predecessors,
                base_assignment, current_context_rules, eva_to_cell,
            )
            if new_dec and '?' not in new_dec:
                n_decoded += 1
                if new_dec in ref_word_set:
                    n_hits += 1

        new_dict_hit = n_hits / max(n_decoded, 1)
        cumulative_hits.append(round(new_dict_hit, 4))
        rule_ids_applied.append(rule.rule_id)
        print(f"  After {rule.rule_id}: dict_hit={new_dict_hit:.1%} "
              f"(Δ={new_dict_hit - cumulative_hits[-2]:+.1%})")

        if new_dict_hit >= 0.25:
            print("  Reached 25% dict_hit target. Stopping.")
            break

    return rules_sorted, cumulative_hits, rule_ids_applied


def _decode_with_context_rules(
    token: str,
    cells_used: List[str],
    cell_positions: List[str],
    cell_predecessors: List[str],
    base_assignment: Dict[str, str],
    context_rules: Dict[str, Dict[str, str]],
    eva_to_cell: Dict[str, str],
) -> str:
    """Decode a token applying a set of context rules."""
    from voynich.core.corpus import tokenize_eva_chars
    chars = tokenize_eva_chars(token)
    parts: List[str] = []
    for ci, ch in enumerate(chars):
        cell = eva_to_cell.get(ch)
        if not cell:
            parts.append('?')
            continue
        pos = cell_positions[ci] if ci < len(cell_positions) else 'medial'
        pred = cell_predecessors[ci] if ci < len(cell_predecessors) else 'NONE'
        ctx = _context_label(pos, pred)

        cell_rules = context_rules.get(cell, {})
        if ctx in cell_rules:
            parts.append(cell_rules[ctx])
        elif 'default' in cell_rules:
            parts.append(cell_rules['default'])
        else:
            syl = base_assignment.get(cell, '?')
            parts.append(syl)
    return ''.join(parts)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_rule_extraction() -> Dict:
    """Phase 13.2: Extract and rank reading rules from error pattern analysis.

    Loads error_patterns.json from Phase 13.1 and the CSP assignment,
    then extracts formal reading rules with coverage and power metrics.
    """
    print("=" * 70)
    print("PHASE 13.2: Rule Extraction and Formalization")
    print("=" * 70)

    t0 = time.time()
    rd = _results_dir()

    # ------------------------------------------------------------------
    # 1. Check gate from Phase 13.1
    # ------------------------------------------------------------------
    ep_path = os.path.join(rd, 'error_patterns.json')
    if not os.path.exists(ep_path):
        print("  [SKIP] error_patterns.json not found — run error-patterns first")
        return {'verdict': 'skipped', 'reason': 'no_error_patterns'}

    with open(ep_path) as f:
        ep_data = json.load(f)

    mi_selectivity = ep_data.get('mi_selectivity', 0.0)
    gate_passed = ep_data.get('gate_passed', False)
    print(f"  Phase 13.1 MI selectivity: {mi_selectivity:.2f}x  gate={'PASS' if gate_passed else 'FAIL'}")

    if mi_selectivity < 1.0:
        print("  [WARNING] MI selectivity < 1.0x — rules likely won't improve dict_hit significantly")
        print("  Proceeding anyway to document baseline behavior")

    # ------------------------------------------------------------------
    # 2. Load CSP assignment and corpus
    # ------------------------------------------------------------------
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
    ref_words_by_len = _bucket_by_length(ref_tokens[:10000], max_per_bucket=60)
    inventory = build_phoneme_inventory('latin', ref_corpus)

    if eva_to_cell_map:
        eva_to_cell = eva_to_cell_map
    else:
        eva_to_cell = build_eva_to_cell_lookup(cv_labels)

    # ------------------------------------------------------------------
    # 3. Reconstruct contextualized diagnoses and error catalog
    # ------------------------------------------------------------------
    from voynich.phases.error_patterns import (
        build_contextualized_diagnoses,
        build_error_catalog,
        TokenDiagnosis,
    )
    from voynich.phases.csp_diagnosis import _get_cells_used

    print("\n  Reconstructing error catalog...")
    raw_diagnoses = []
    for token in corpus_tokens:
        decoded = decode_token(token, best_assignment, eva_to_cell)
        cells_used = _get_cells_used(token, eva_to_cell)
        cat, best_match, best_dist = categorize_token(
            decoded, ref_word_set, ref_words_by_len, inventory,
        )
        raw_diagnoses.append(TokenDiagnosis(
            voynich_token=token,
            decoded=decoded,
            category=cat,
            best_dict_match=best_match,
            best_dict_distance=best_dist,
            cells_used=cells_used,
        ))

    ctx_diagnoses = build_contextualized_diagnoses(raw_diagnoses, best_assignment, eva_to_cell, cv_labels)
    ctx_diagnoses, error_catalog = build_error_catalog(ctx_diagnoses, best_assignment, cv_labels)
    meaningful_errors = [e for e in error_catalog if e.produced and e.needed]
    print(f"  Error records: {len(meaningful_errors)}")

    # ------------------------------------------------------------------
    # 4. Extract rules
    # ------------------------------------------------------------------
    print("\n  Extracting reading rules...")
    position_tests = ep_data.get('position_tests', [])
    adjacency_tests = ep_data.get('adjacency_tests', [])

    rules = extract_rules(
        position_tests, adjacency_tests, meaningful_errors,
        best_assignment, cv_labels,
    )
    print(f"  Rules extracted: {len(rules)}")

    if not rules:
        print("  No rules found with sufficient coverage. Checking raw error data...")
        # Fallback: extract from raw corrections even without significance
        from voynich.phases.error_patterns import PositionTest
        # Use top corrections from error patterns directly
        for pt in position_tests[:5]:
            if pt.get('n_errors', 0) < 5:
                continue
            cell_key = pt['cell_key']
            produced = best_assignment.get(cell_key, '?')
            for pos, correction in pt.get('top_correction_by_position', {}).items():
                if correction and correction != produced:
                    ctx = _context_label(pos, 'NONE' if pos == 'initial' else 'vowel')
                    plausibility, basis = _check_plausibility(produced, correction)
                    n_in_ctx = pt.get('n_errors_by_position', {}).get(pos, 0)
                    if n_in_ctx >= 3:
                        rules.append(ReadingRule(
                            rule_id=f"R_fallback_{pt['cv_label']}_{ctx[:4]}",
                            cell_key=cell_key,
                            cv_label=pt.get('cv_label', '?'),
                            context=ctx,
                            produced=produced,
                            corrected=correction,
                            coverage=0.30,
                            n_errors_in_context=n_in_ctx,
                            power=0.0,
                            plausibility=plausibility,
                            linguistic_basis=basis,
                        ))
        print(f"  Fallback rules: {len(rules)}")

    # ------------------------------------------------------------------
    # 5. Baseline dict_hit
    # ------------------------------------------------------------------
    n_hits = sum(1 for d in ctx_diagnoses if d.category == 'HIT')
    n_decoded = sum(1 for d in ctx_diagnoses if d.decoded and '?' not in d.decoded)
    baseline_dict_hit = n_hits / max(n_decoded, 1)
    n_near_miss = sum(1 for d in ctx_diagnoses if d.category == 'NEAR_MISS')
    print(f"  Baseline dict_hit: {baseline_dict_hit:.1%}  Near-miss: {n_near_miss}")

    # ------------------------------------------------------------------
    # 6. Score rule power and apply greedily
    # ------------------------------------------------------------------
    if rules:
        print("\n  Applying rules greedily...")
        rules_sorted, cumulative_hits, rule_ids_applied = apply_rules_greedy(
            rules, corpus_tokens, ctx_diagnoses, best_assignment,
            eva_to_cell, ref_word_set, ref_words_by_len, inventory,
            baseline_dict_hit,
        )
    else:
        rules_sorted = []
        cumulative_hits = [baseline_dict_hit]
        rule_ids_applied = []

    final_dict_hit = cumulative_hits[-1] if cumulative_hits else baseline_dict_hit
    total_improvement = final_dict_hit - baseline_dict_hit

    gate_passed_13_2 = final_dict_hit >= 0.15
    if gate_passed_13_2:
        gate_message = f"PASS: Cumulative dict_hit {final_dict_hit:.1%} ≥ 15% after applying rules."
    else:
        gate_message = (
            f"FAIL: Cumulative dict_hit {final_dict_hit:.1%} < 15%. "
            "Context rules provide insufficient improvement. "
            "Will attempt free-search CSP in context_csp.py."
        )

    print(f"\n  Final dict_hit after all rules: {final_dict_hit:.1%}")
    print(f"  Total improvement: {total_improvement:+.1%}")
    print(f"  Gate 13.2: {'PASS ✓' if gate_passed_13_2 else 'FAIL ✗'}")
    print(f"  {gate_message}")

    # ------------------------------------------------------------------
    # 7. Print rule summary
    # ------------------------------------------------------------------
    if rules_sorted:
        print(f"\n  Reading rule summary ({len(rules_sorted)} rules):")
        for r in rules_sorted[:10]:
            print(f"    [{r.plausibility[:3].upper()}] {r.rule_id}: "
                  f"{r.cv_label} {r.produced}→{r.corrected} / {r.context}  "
                  f"cov={r.coverage:.0%}  power={r.power:.3f}  "
                  f"| {r.linguistic_basis[:50]}")

    # ------------------------------------------------------------------
    # 8. Save
    # ------------------------------------------------------------------
    result = RuleExtractionResult(
        n_near_miss_tokens=n_near_miss,
        n_cells_with_rules=len({r.cell_key for r in rules_sorted}),
        rules=[_convert(asdict(r)) for r in rules_sorted],
        cumulative_dict_hit=cumulative_hits,
        cumulative_rule_ids=rule_ids_applied,
        baseline_dict_hit=round(baseline_dict_hit, 4),
        final_dict_hit=round(final_dict_hit, 4),
        total_improvement=round(total_improvement, 4),
        gate_passed=gate_passed_13_2,
        gate_message=gate_message,
    )

    out_path = os.path.join(rd, 'rule_extraction.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2)

    elapsed = time.time() - t0
    print(f"\n  Saved to {out_path} ({elapsed:.1f}s)")
    return _convert(asdict(result))
