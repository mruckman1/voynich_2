"""
Step 40.6 – CVC-Expanded CSP
==============================
Re-solve the Phase 14 CSP with expanded CVC/CCV syllable domains.

Dependency chain:
    cvc_inventory.json       (Step 40.5)
    combined_refine.json     (Step 15)
    modifier_integrate.json  (Step 16)
        → cvc_csp.json       (this step)
"""

import json
import os
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    load_corpus,
    tokenize_eva_chars,
)
from voynich.core.reference import (
    build_cvc_syllable_table,
    build_triple_phoneme_hypotheses,
    PHONEME_PLACE_MAP,
    PHONEME_NUCLEUS_MAP,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _convert(obj: Any) -> Any:
    if hasattr(obj, '__dataclass_fields__'):
        from dataclasses import asdict
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
# Core: CVC CSP
# ---------------------------------------------------------------------------

def _build_cvc_domains(
    variables,
    cvc_inventory: Dict,
    recommended_level: int,
) -> Dict[str, List[str]]:
    """Build expanded CVC domains for each triple.

    Uses the CVC inventory from Step 40.5, adding CVC/CCV syllables
    to the existing CV domain for each variable.
    """
    # Get the CVC syllable set at the recommended level
    level_data = cvc_inventory.get('by_level', {}).get(str(recommended_level), {})
    cvc_sample = level_data.get('sample', [])

    # Get CVC/CCV syllables from the Anonimo analysis
    top_cvc = [s['syl'] for s in cvc_inventory.get('top_cvc_syllables', [])]
    top_ccv = [s['syl'] for s in cvc_inventory.get('top_ccv_syllables', [])]

    all_cvc_ccv = set(cvc_sample) | set(top_cvc) | set(top_ccv)

    # For each variable, expand domain: existing CV + relevant CVC/CCV
    domains = {}
    for var in variables:
        existing_domain = list(var.domain) if hasattr(var, 'domain') else []
        # Add CVC syllables that share the same onset consonant family
        first_stroke = var.first_stroke if hasattr(var, 'first_stroke') else ''
        onset_consonants = set(PHONEME_PLACE_MAP.get(first_stroke, []))

        expanded = set(existing_domain)
        for syl in all_cvc_ccv:
            if len(syl) >= 2:
                # Check if onset consonant matches
                if syl[0] in onset_consonants or not onset_consonants:
                    expanded.add(syl)

        domains[var.cell_key] = sorted(expanded)

    return domains


def _coordinate_descent_cvc(
    variables,
    cvc_domains: Dict[str, List[str]],
    voynich_tokens: List[str],
    eva_to_triple: Dict[str, str],
    ref_word_set: Set[str],
    best_cv_assignment: Dict[str, str],
    max_iterations: int = 5,
) -> Tuple[Dict[str, str], float]:
    """Coordinate descent: for each triple, try all CVC domain values,
    keep the one that maximizes dict-hit.

    Start from the Phase 15 CV assignment and try to improve.
    """
    from voynich.phases.csp_solver import decode_token

    current = dict(best_cv_assignment)

    def _score(assignment):
        hits = 0
        total = 0
        for token in voynich_tokens:
            decoded = decode_token(token, assignment, eva_to_triple)
            if decoded and len(decoded) >= 2:
                total += 1
                if decoded in ref_word_set:
                    hits += 1
        return hits / total if total > 0 else 0.0

    best_score = _score(current)
    print(f"      Initial score: {best_score:.4f}")

    for iteration in range(max_iterations):
        improved = False
        for var in variables:
            key = var.cell_key
            domain = cvc_domains.get(key, [current.get(key, '')])
            original_val = current.get(key, '')
            best_val = original_val
            best_val_score = best_score

            for candidate in domain:
                if candidate == original_val:
                    continue
                current[key] = candidate
                score = _score(current)
                if score > best_val_score:
                    best_val = candidate
                    best_val_score = score

            current[key] = best_val
            if best_val != original_val:
                improved = True
                best_score = best_val_score
                print(f"      Iter {iteration}: {key} → {best_val} (score {best_score:.4f})")

        if not improved:
            print(f"      Converged at iteration {iteration}")
            break

    return current, best_score


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_cvc_csp() -> None:
    """Step 40.6: CVC-Expanded CSP."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 40.6: CVC-Expanded CSP")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load inputs ──
    print("\n  1. Loading inputs …")
    cvc_inv = _safe_load(os.path.join(rd, 'cvc_inventory.json'))
    refine = _safe_load(os.path.join(rd, 'combined_refine.json'))
    mod_data = _safe_load(os.path.join(rd, 'modifier_integrate.json'))
    merged_dict = _safe_load(os.path.join(rd, 'merged_dict.json'))

    best_cv_assignment = refine.get('best_assignment', {})
    recommended_level = cvc_inv.get('recommended_level', 3)
    print(f"    Phase 15 CV assignment: {len(best_cv_assignment)} triples")
    print(f"    Recommended CVC level: {recommended_level}")

    # Build reference word set
    ref_words = set(merged_dict.get('latin_10k_words', []))
    ref_words.update(merged_dict.get('italian_10k_words', []))
    ven_lex = _safe_load(os.path.join(rd, 'venetian_lexicon.json'))
    for entry in ven_lex.get('supplement_words', []):
        if isinstance(entry, str):
            ref_words.add(entry)
        elif isinstance(entry, dict):
            ref_words.add(entry.get('word', ''))
    # Add Venetian extended set if available
    ven_forms = _safe_load(os.path.join(rd, 'venetian_forms.json'))
    for w in ven_forms.get('venetian_extended_set', []):
        ref_words.add(w)
    ref_words.discard('')
    print(f"    Reference word set: {len(ref_words):,}")

    # ── 2. Load corpus ──
    print("\n  2. Loading corpus …")
    corpus = load_corpus(verbose=False)
    eva_to_triple = build_eva_to_triple_lookup()
    all_tokens = corpus.get_tokens()

    # Use herbal_a subsample for speed (consistent with Phase 14/15)
    herbal_pages = corpus.get_pages_by_section('herbal_a')
    herbal_tokens = []
    for page in herbal_pages:
        herbal_tokens.extend(page.all_tokens)
    subsample = herbal_tokens[:2000] if len(herbal_tokens) > 2000 else herbal_tokens
    print(f"    Full corpus: {len(all_tokens):,} tokens")
    print(f"    Herbal_a subsample: {len(subsample):,} tokens")

    # ── 3. Build feature variables ──
    print("\n  3. Building feature variables …")
    # Build glyph frequencies from subsample
    glyph_freq: Counter = Counter()
    for token in subsample:
        for glyph in tokenize_eva_chars(token):
            glyph_freq[glyph] += 1

    from voynich.phases.feature_csp import build_feature_variables, FeatureVariable
    # PhonemeInventory duck-type: just need .syllables attribute
    class _SimpleInventory:
        def __init__(self, syllables):
            self.syllables = syllables

    # Get CV syllables as baseline
    try:
        cv_table = build_cvc_syllable_table('italian', relaxation_level=0)
        if isinstance(cv_table, dict):
            cv_syls = set()
            for v in cv_table.values():
                if isinstance(v, (list, set)):
                    cv_syls.update(v)
                elif isinstance(v, str):
                    cv_syls.add(v)
        else:
            cv_syls = set(cv_table) if cv_table else set()
    except Exception:
        cv_syls = set()

    inventory = _SimpleInventory(sorted(cv_syls) if cv_syls else ['ba', 'be', 'bi', 'bo',
        'ca', 'ce', 'ci', 'co', 'da', 'de', 'di', 'do', 'fa', 'fe',
        'ga', 'ha', 'hi', 'la', 'le', 'mi', 'ne', 'ni', 'no',
        'ra', 're', 'ri', 'ro', 'se', 'si', 'so', 'te', 'to'])

    variables = build_feature_variables(
        eva_to_triple, glyph_freq, inventory,
    )
    print(f"    Feature variables: {len(variables)}")

    # Set initial domains from CV assignment
    for var in variables:
        cv_syl = best_cv_assignment.get(var.cell_key, '')
        if cv_syl and cv_syl not in var.domain:
            var.domain.append(cv_syl)
        if not var.domain:
            var.domain = list(inventory.syllables)

    # ── 4. Build CVC-expanded domains ──
    print("\n  4. Building CVC-expanded domains …")
    cvc_domains = _build_cvc_domains(variables, cvc_inv, recommended_level)
    for key, dom in cvc_domains.items():
        print(f"    {key}: {len(dom)} candidates")

    # ── 5. Coordinate descent ──
    print("\n  5. Running coordinate descent with CVC domains …")
    cvc_assignment, cvc_score = _coordinate_descent_cvc(
        variables, cvc_domains, subsample, eva_to_triple,
        ref_words, best_cv_assignment,
        max_iterations=3,
    )

    # ── 6. Compare to CV baseline ──
    print("\n  6. Comparison:")
    from voynich.phases.csp_solver import decode_token
    cv_hits = 0
    cv_total = 0
    for token in subsample:
        decoded = decode_token(token, best_cv_assignment, eva_to_triple)
        if decoded and len(decoded) >= 2:
            cv_total += 1
            if decoded in ref_words:
                cv_hits += 1
    cv_score = cv_hits / cv_total if cv_total > 0 else 0.0

    print(f"    CV-only dict-hit: {cv_score:.4f}")
    print(f"    CVC expanded dict-hit: {cvc_score:.4f}")
    print(f"    Delta: {cvc_score - cv_score:+.4f}")

    # Count changed triples
    n_changed = sum(1 for k in best_cv_assignment
                    if cvc_assignment.get(k) != best_cv_assignment.get(k))
    print(f"    Changed triples: {n_changed}/{len(best_cv_assignment)}")

    # ── 7. Save ──
    elapsed = time.time() - t0

    output = {
        'recommended_level': recommended_level,
        'n_variables': len(variables),
        'n_subsample_tokens': len(subsample),
        'cv_dict_hit': round(cv_score, 6),
        'cvc_dict_hit': round(cvc_score, 6),
        'delta': round(cvc_score - cv_score, 6),
        'n_changed_triples': n_changed,
        'best_assignment_cvc': cvc_assignment,
        'best_cv_assignment': best_cv_assignment,
        'changes': {k: {'cv': best_cv_assignment.get(k, ''),
                         'cvc': cvc_assignment.get(k, '')}
                    for k in best_cv_assignment
                    if cvc_assignment.get(k) != best_cv_assignment.get(k)},
        'verdict': ('CVC_IMPROVES' if cvc_score > cv_score + 0.005
                    else 'CVC_NEUTRAL' if abs(cvc_score - cv_score) <= 0.005
                    else 'CVC_DEGRADES'),
        'runtime_seconds': round(elapsed, 1),
    }

    out_path = os.path.join(rd, 'cvc_csp.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(output), f, indent=2)
    print(f"\n  Saved → {out_path} ({elapsed:.1f}s)")
