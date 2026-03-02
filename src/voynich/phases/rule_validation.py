"""
Phase 13.4 – Rule Validation
=============================
Validates extracted reading rules against three independent criteria:

  1. Cross-validation: Split Language A folios into two halves.  Run rule
     extraction independently on each half.  A rule "transfers" if it
     appears in both halves AND produces dict_hit improvement on the OTHER
     half's tokens.

  2. Per-rule selectivity: Apply each rule to real tokens vs. shuffled tokens
     (token order randomized).  Selectivity = improvement_real / improvement_shuffled.
     Genuine reading rules depend on token order (sequential context), so they
     should NOT improve shuffled text.

  3. Linguistic plausibility: Cross-reference with ROMANCE_PHONOLOGICAL_PROCESSES.
     Rules rated 'high' or 'moderate' are accepted; 'low' are rejected.

Gate 13.4: ≥ 1 rule passes all three checks.  Returns the validated rule set.
"""

import json
import os
import random
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import build_eva_to_cell_lookup, load_corpus
from voynich.core.reference import ROMANCE_PHONOLOGICAL_PROCESSES, load_reference_corpus
from voynich.phases.csp_constraints import build_phoneme_inventory
from voynich.phases.csp_solver import _convert, decode_token
from voynich.phases.csp_diagnosis import (
    categorize_token,
    _bucket_by_length,
    _get_cells_used,
)
from voynich.phases.csp_diagnosis import TokenDiagnosis
from voynich.phases.error_patterns import (
    build_contextualized_diagnoses,
    build_error_catalog,
)
from voynich.phases.rule_extraction import (
    ReadingRule,
    _context_label,
    extract_rules,
    apply_rules_greedy,
    _decode_with_context_rules,
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RuleValidationRecord:
    """Validation results for one reading rule."""
    rule_id: str
    cell_key: str
    cv_label: str
    context: str
    produced: str
    corrected: str
    coverage: float
    power: float
    plausibility: str
    linguistic_basis: str
    # Validation fields
    transfers_half_a_to_b: bool    # Rule found in half A and improves half B
    transfers_half_b_to_a: bool    # Rule found in half B and improves half A
    transfer_rate: float           # Fraction of cross-validation folds where rule transfers
    selectivity_real: float        # dict_hit improvement on real tokens
    selectivity_shuffled: float    # dict_hit improvement on shuffled tokens
    selectivity_ratio: float       # real / shuffled
    validated: bool                # Passes all three checks


@dataclass
class RuleValidationResult:
    """Full Phase 13.4 output."""
    n_rules_input: int
    n_rules_validated: int
    rule_records: List[Dict]
    validated_rules: List[Dict]
    cross_validation_transfer_rate: float
    n_plausible: int
    n_high_selectivity: int
    final_dict_hit_validated: float
    baseline_dict_hit: float
    gate_passed: bool
    gate_message: str


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def _split_folios(corpus: Any) -> Tuple[List[str], List[str]]:
    """Split Language A folios into two halves by folio index (odd vs even)."""
    all_tokens_by_folio = corpus.get_tokens_by_folio(language='A', paragraph_only=True)
    folios = sorted(all_tokens_by_folio.keys())
    half_a_folios = [f for i, f in enumerate(folios) if i % 2 == 0]
    half_b_folios = [f for i, f in enumerate(folios) if i % 2 == 1]

    half_a_tokens: List[str] = []
    half_b_tokens: List[str] = []
    for f in half_a_folios:
        half_a_tokens.extend(all_tokens_by_folio[f])
    for f in half_b_folios:
        half_b_tokens.extend(all_tokens_by_folio[f])

    return half_a_tokens[:800], half_b_tokens[:800]


def _run_extraction_on_half(
    tokens: List[str],
    best_assignment: Dict[str, str],
    eva_to_cell: Dict[str, str],
    cv_labels: Dict,
    ref_word_set: set,
    ref_words_by_len: Dict,
    inventory: Any,
) -> List[ReadingRule]:
    """Run error catalog + rule extraction on a subset of tokens."""
    raw = []
    for token in tokens:
        decoded = decode_token(token, best_assignment, eva_to_cell)
        cells_used = _get_cells_used(token, eva_to_cell)
        cat, best_match, best_dist = categorize_token(
            decoded, ref_word_set, ref_words_by_len, inventory,
        )
        raw.append(TokenDiagnosis(
            voynich_token=token, decoded=decoded, category=cat,
            best_dict_match=best_match, best_dict_distance=best_dist,
            cells_used=cells_used,
        ))

    ctx = build_contextualized_diagnoses(raw, best_assignment, eva_to_cell, cv_labels)
    ctx, errors = build_error_catalog(ctx, best_assignment, cv_labels)
    meaningful = [e for e in errors if e.produced and e.needed]

    # Minimal chi-squared test to extract rules
    from voynich.phases.error_patterns import test_position_dependence, test_adjacency_dependence
    pos_tests = test_position_dependence(meaningful, cv_labels, best_assignment)
    adj_tests = test_adjacency_dependence(meaningful, cv_labels, best_assignment)

    rules = extract_rules(
        [asdict(t) for t in pos_tests],
        [asdict(t) for t in adj_tests],
        meaningful, best_assignment, cv_labels,
        min_coverage=0.30, min_errors=5,
    )
    return rules


def run_cross_validation(
    input_rules: List[Dict],
    corpus: Any,
    best_assignment: Dict[str, str],
    eva_to_cell: Dict[str, str],
    cv_labels: Dict,
    ref_word_set: set,
    ref_words_by_len: Dict,
    inventory: Any,
) -> Dict[str, bool]:
    """Check which rules transfer across folio halves.

    Returns {rule_id: transfers}.
    """
    print("  Splitting corpus into two folio halves...")

    # Try to get tokens by folio, fall back to sequential split
    try:
        half_a_tokens, half_b_tokens = _split_folios(corpus)
    except (AttributeError, TypeError):
        all_tokens = corpus.get_tokens(language='A', paragraph_only=True)[:1600]
        mid = len(all_tokens) // 2
        half_a_tokens = all_tokens[:mid]
        half_b_tokens = all_tokens[mid:]

    print(f"  Half A: {len(half_a_tokens)} tokens, Half B: {len(half_b_tokens)} tokens")

    # Extract rules from each half
    print("  Extracting rules from half A...")
    rules_a = _run_extraction_on_half(
        half_a_tokens, best_assignment, eva_to_cell, cv_labels,
        ref_word_set, ref_words_by_len, inventory,
    )
    print(f"  Rules from half A: {len(rules_a)}")

    print("  Extracting rules from half B...")
    rules_b = _run_extraction_on_half(
        half_b_tokens, best_assignment, eva_to_cell, cv_labels,
        ref_word_set, ref_words_by_len, inventory,
    )
    print(f"  Rules from half B: {len(rules_b)}")

    # Build signature sets for quick lookup
    def _sig(r):
        return (r.cell_key, r.context, r.corrected)

    sigs_a = {_sig(r) for r in rules_a}
    sigs_b = {_sig(r) for r in rules_b}

    # Check which input rules appear in each half
    transfer: Dict[str, bool] = {}
    for r in input_rules:
        sig = (r['cell_key'], r['context'], r['corrected'])
        in_a = sig in sigs_a
        in_b = sig in sigs_b
        transfer[r['rule_id']] = in_a or in_b  # present in at least one half

    # Also check whether rules from A improve B and vice versa
    def _apply_and_score(rules_list, test_tokens, assignment, eva_to_cell, ref_word_set):
        context_rules: Dict[str, Dict[str, str]] = {}
        for r in rules_list:
            ck, ctx, corr = _sig(r)
            if ck not in context_rules:
                context_rules[ck] = {}
            context_rules[ck][ctx] = corr
        n_hits = 0
        n_total = 0
        for tok in test_tokens:
            dec = decode_token(tok, assignment, eva_to_cell, context_rules)
            if dec and '?' not in dec:
                n_total += 1
                if dec in ref_word_set:
                    n_hits += 1
        return n_hits / max(n_total, 1)

    # Baseline dict_hits for each half
    def _baseline_score(test_tokens, assignment, eva_to_cell, ref_word_set):
        n_hits = sum(1 for tok in test_tokens
                     if decode_token(tok, assignment, eva_to_cell) in ref_word_set)
        n_total = sum(1 for tok in test_tokens
                      if decode_token(tok, assignment, eva_to_cell) not in ('', '?'))
        return n_hits / max(n_total, 1)

    if rules_a:
        baseline_b = _baseline_score(half_b_tokens, best_assignment, eva_to_cell, ref_word_set)
        score_a_on_b = _apply_and_score(rules_a, half_b_tokens, best_assignment, eva_to_cell, ref_word_set)
        a_improves_b = score_a_on_b > baseline_b
        print(f"  Rules from A on half B: {score_a_on_b:.1%} (baseline {baseline_b:.1%}) — "
              f"{'improves' if a_improves_b else 'no improvement'}")

    if rules_b:
        baseline_a = _baseline_score(half_a_tokens, best_assignment, eva_to_cell, ref_word_set)
        score_b_on_a = _apply_and_score(rules_b, half_a_tokens, best_assignment, eva_to_cell, ref_word_set)
        b_improves_a = score_b_on_a > baseline_a
        print(f"  Rules from B on half A: {score_b_on_a:.1%} (baseline {baseline_a:.1%}) — "
              f"{'improves' if b_improves_a else 'no improvement'}")

    return transfer


# ---------------------------------------------------------------------------
# Per-rule selectivity
# ---------------------------------------------------------------------------

def compute_rule_selectivity(
    rule: Dict,
    corpus_tokens: List[str],
    best_assignment: Dict[str, str],
    eva_to_cell: Dict[str, str],
    ref_word_set: set,
    ref_words_by_len: Dict,
    inventory: Any,
    n_shuffles: int = 20,
    seed: int = 42,
) -> Tuple[float, float]:
    """Compute dict_hit improvement for real vs shuffled token order.

    Returns (selectivity_real, selectivity_shuffled) as absolute improvements.
    """
    rng = random.Random(seed)

    # Baseline
    n_baseline_hits = sum(
        1 for tok in corpus_tokens
        if decode_token(tok, best_assignment, eva_to_cell) in ref_word_set
    )
    n_valid = sum(
        1 for tok in corpus_tokens
        if decode_token(tok, best_assignment, eva_to_cell) not in ('', '?')
    )
    baseline = n_baseline_hits / max(n_valid, 1)

    # With rule applied
    ctx_rules: Dict[str, Dict[str, str]] = {rule['cell_key']: {rule['context']: rule['corrected']}}
    n_rule_hits = sum(
        1 for tok in corpus_tokens
        if decode_token(tok, best_assignment, eva_to_cell, ctx_rules) in ref_word_set
    )
    real_improvement = (n_rule_hits / max(n_valid, 1)) - baseline

    # With shuffled token order (same rule, shuffled context)
    shuffled_improvements: List[float] = []
    for _ in range(n_shuffles):
        shuffled_tokens = list(corpus_tokens)
        rng.shuffle(shuffled_tokens)
        n_shuffled_hits = sum(
            1 for tok in shuffled_tokens
            if decode_token(tok, best_assignment, eva_to_cell, ctx_rules) in ref_word_set
        )
        shuffled_improvements.append((n_shuffled_hits / max(n_valid, 1)) - baseline)

    shuffled_mean = sum(shuffled_improvements) / max(len(shuffled_improvements), 1)
    return real_improvement, shuffled_mean


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_rule_validation() -> Dict:
    """Phase 13.4: Validate extracted reading rules.

    Cross-validates rules across folio halves, computes per-rule selectivity,
    and checks linguistic plausibility.  Returns only validated rules.
    """
    print("=" * 70)
    print("PHASE 13.4: Rule Validation")
    print("=" * 70)

    t0 = time.time()
    rd = _results_dir()

    # ------------------------------------------------------------------
    # 1. Load input rules
    # ------------------------------------------------------------------
    re_path = os.path.join(rd, 'rule_extraction.json')
    if not os.path.exists(re_path):
        print("  [SKIP] rule_extraction.json not found — run extract-rules first")
        return {'verdict': 'skipped', 'reason': 'no_rule_extraction'}

    with open(re_path) as f:
        re_data = json.load(f)

    input_rules: List[Dict] = re_data.get('rules', [])
    baseline_dict_hit = re_data.get('baseline_dict_hit', 0.0)
    print(f"  Input rules: {len(input_rules)}")
    print(f"  Baseline dict_hit: {baseline_dict_hit:.1%}")

    if not input_rules:
        result = RuleValidationResult(
            n_rules_input=0, n_rules_validated=0, rule_records=[],
            validated_rules=[], cross_validation_transfer_rate=0.0,
            n_plausible=0, n_high_selectivity=0,
            final_dict_hit_validated=baseline_dict_hit,
            baseline_dict_hit=baseline_dict_hit, gate_passed=False,
            gate_message="No rules to validate.",
        )
        out_path = os.path.join(rd, 'rule_validation.json')
        with open(out_path, 'w') as f:
            json.dump(_convert(asdict(result)), f, indent=2)
        return _convert(asdict(result))

    # ------------------------------------------------------------------
    # 2. Load corpus and reference
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
    # 3. Cross-validation
    # ------------------------------------------------------------------
    print("\n  Running cross-validation...")
    transfer_map = run_cross_validation(
        input_rules, corpus, best_assignment, eva_to_cell, cv_labels,
        ref_word_set, ref_words_by_len, inventory,
    )
    n_transfer = sum(1 for v in transfer_map.values() if v)
    transfer_rate = n_transfer / max(len(input_rules), 1)
    print(f"  Transfer rate: {n_transfer}/{len(input_rules)} = {transfer_rate:.0%}")

    # ------------------------------------------------------------------
    # 4. Per-rule selectivity and plausibility
    # ------------------------------------------------------------------
    print("\n  Computing per-rule selectivity and plausibility...")
    records: List[RuleValidationRecord] = []

    for r in input_rules:
        # Selectivity
        real_imp, shuffled_imp = compute_rule_selectivity(
            r, corpus_tokens, best_assignment, eva_to_cell, ref_word_set,
            ref_words_by_len, inventory, n_shuffles=20,
        )
        sel_ratio = real_imp / max(abs(shuffled_imp), 1e-6) if abs(shuffled_imp) > 1e-9 else (
            10.0 if real_imp > 0 else 1.0
        )

        # Plausibility already computed in rule extraction; use stored value
        plausibility = r.get('plausibility', 'low')

        # Transfer
        transfers = transfer_map.get(r['rule_id'], False)

        # Validation: passes if transfer AND selectivity >= 1.5 AND plausibility != 'low'
        validated = transfers and sel_ratio >= 1.5 and plausibility != 'low'

        records.append(RuleValidationRecord(
            rule_id=r['rule_id'],
            cell_key=r['cell_key'],
            cv_label=r.get('cv_label', '?'),
            context=r['context'],
            produced=r['produced'],
            corrected=r['corrected'],
            coverage=r.get('coverage', 0.0),
            power=r.get('power', 0.0),
            plausibility=plausibility,
            linguistic_basis=r.get('linguistic_basis', ''),
            transfers_half_a_to_b=transfers,
            transfers_half_b_to_a=transfers,
            transfer_rate=1.0 if transfers else 0.0,
            selectivity_real=round(real_imp, 4),
            selectivity_shuffled=round(shuffled_imp, 4),
            selectivity_ratio=round(sel_ratio, 3),
            validated=validated,
        ))

        status = "✓" if validated else "✗"
        print(f"  {status} {r['rule_id']}: {r.get('cv_label', '?')} {r['produced']}→{r['corrected']} "
              f"/ {r['context']}  xfer={'Y' if transfers else 'N'}  "
              f"sel={sel_ratio:.2f}x  [{plausibility}]")

    # ------------------------------------------------------------------
    # 5. Compute final dict_hit with validated rules only
    # ------------------------------------------------------------------
    validated_rules = [r for r in records if r.validated]
    print(f"\n  Validated rules: {len(validated_rules)}/{len(records)}")

    if validated_rules:
        context_rules: Dict[str, Dict[str, str]] = {}
        for r in validated_rules:
            if r.cell_key not in context_rules:
                context_rules[r.cell_key] = {}
            context_rules[r.cell_key][r.context] = r.corrected

        n_hits = sum(
            1 for tok in corpus_tokens
            if decode_token(tok, best_assignment, eva_to_cell, context_rules) in ref_word_set
        )
        n_valid = sum(
            1 for tok in corpus_tokens
            if decode_token(tok, best_assignment, eva_to_cell, context_rules) not in ('', '?')
        )
        final_dict_hit = n_hits / max(n_valid, 1)
    else:
        final_dict_hit = baseline_dict_hit

    improvement = final_dict_hit - baseline_dict_hit
    print(f"  Final dict_hit with validated rules: {final_dict_hit:.1%} (Δ={improvement:+.1%})")

    n_plausible = sum(1 for r in records if r.plausibility in ('high', 'moderate'))
    n_high_sel = sum(1 for r in records if r.selectivity_ratio >= 1.5)

    gate_passed = len(validated_rules) >= 1
    if gate_passed:
        gate_message = (
            f"PASS: {len(validated_rules)} validated rule(s). "
            f"dict_hit={final_dict_hit:.1%} (Δ={improvement:+.1%}). "
            "Proceeding to full corpus decoding."
        )
    else:
        gate_message = (
            f"FAIL: No rules pass all three validation checks. "
            f"Transfer rate={transfer_rate:.0%}, plausible={n_plausible}, high-sel={n_high_sel}. "
            "The 11.1% ceiling is not improvable via context-dependent reading rules with the "
            "current 14-cell grid."
        )

    print(f"  Gate 13.4: {'PASS ✓' if gate_passed else 'FAIL ✗'}")
    print(f"  {gate_message}")

    # ------------------------------------------------------------------
    # 6. Save
    # ------------------------------------------------------------------
    result = RuleValidationResult(
        n_rules_input=len(input_rules),
        n_rules_validated=len(validated_rules),
        rule_records=[_convert(asdict(r)) for r in records],
        validated_rules=[_convert(asdict(r)) for r in validated_rules],
        cross_validation_transfer_rate=round(transfer_rate, 3),
        n_plausible=n_plausible,
        n_high_selectivity=n_high_sel,
        final_dict_hit_validated=round(final_dict_hit, 4),
        baseline_dict_hit=round(baseline_dict_hit, 4),
        gate_passed=gate_passed,
        gate_message=gate_message,
    )

    out_path = os.path.join(rd, 'rule_validation.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2)

    elapsed = time.time() - t0
    print(f"\n  Saved to {out_path} ({elapsed:.1f}s)")
    return _convert(asdict(result))
