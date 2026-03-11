"""
Step 40.15 – Predicted Form Search
====================================
Search each botanical folio for EVA tokens matching the predicted
partial form and validate cross-folio.

Dependency chain:
    botanical_predictions.json  (Step 40.14)
    merged_signal.json          (Step 38.3)
    combined_refine.json        (Step 15)
        → botanical_search.json (this step)
"""

import json
import os
import time
from typing import Any, Dict, List, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    load_corpus,
    token_to_triples,
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
# Core: Search and validation
# ---------------------------------------------------------------------------

def _search_folio_tokens(
    folio_tokens: List[str],
    predicted_triples: List[str],
    known_positions: List[int],
    eva_to_triple: Dict[str, str],
) -> List[Dict]:
    """Search folio tokens for matches at known predicted positions."""
    matches = []
    for token in folio_tokens:
        triples = token_to_triples(token, eva_to_triple)
        if not triples:
            continue

        # Check if known positions match predicted triples
        if len(triples) < len(known_positions):
            continue

        n_match = 0
        n_checked = 0
        for pos in known_positions:
            if pos < len(triples) and pos < len(predicted_triples):
                n_checked += 1
                if triples[pos] == predicted_triples[pos]:
                    n_match += 1

        if n_checked > 0 and n_match == n_checked:
            matches.append({
                'token': token,
                'token_triples': triples,
                'n_matched_positions': n_match,
                'n_total_positions': len(triples),
            })

    return matches


def _cross_folio_validate(
    all_matches: List[Dict],
) -> Dict:
    """Check if matches from different folios imply consistent triple assignments."""
    # Collect all implied assignments: triple_key → set of syllables
    implied: Dict[str, Set[str]] = {}
    for m in all_matches:
        for triple, syllable in m.get('implied_assignments', {}).items():
            if triple not in implied:
                implied[triple] = set()
            implied[triple].add(syllable)

    n_consistent = sum(1 for s in implied.values() if len(s) == 1)
    n_conflicting = sum(1 for s in implied.values() if len(s) > 1)

    return {
        'n_triples_tested': len(implied),
        'n_consistent': n_consistent,
        'n_conflicting': n_conflicting,
        'consistency_rate': round(n_consistent / max(len(implied), 1), 4),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_botanical_search() -> None:
    """Step 40.15: Predicted Form Search."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 40.15: Predicted Form Search")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load inputs ──
    print("\n  1. Loading inputs …")
    pred_data = _safe_load(os.path.join(rd, 'botanical_predictions.json'))
    refine = _safe_load(os.path.join(rd, 'combined_refine.json'))

    predictions = pred_data.get('predictions', [])
    assignment = refine.get('best_assignment', {})
    print(f"    Predictions to search: {len(predictions)}")

    # ── 2. Load corpus ──
    print("\n  2. Loading corpus …")
    corpus = load_corpus(verbose=False)
    eva_to_triple = build_eva_to_triple_lookup()

    # Build folio → tokens map
    folio_tokens: Dict[str, List[str]] = {}
    for folio, page in corpus.pages.items():
        folio_tokens[folio] = page.all_tokens
    print(f"    Folios loaded: {len(folio_tokens)}")

    # ── 3. Search for each prediction ──
    print("\n  3. Searching folios …")
    all_results = []
    n_corroborated = 0

    for pred in predictions:
        folio = pred.get('folio', '')
        predicted_triples = pred.get('predicted_triples', [])
        known_positions = pred.get('known_positions', [])

        if not folio or not predicted_triples or not known_positions:
            continue

        tokens = folio_tokens.get(folio, [])
        matches = _search_folio_tokens(
            tokens, predicted_triples, known_positions, eva_to_triple,
        )

        # Also search wrong folios as null control
        wrong_match_count = 0
        n_wrong_tested = 0
        for wrong_folio, wrong_tokens in list(folio_tokens.items())[:20]:
            if wrong_folio == folio:
                continue
            n_wrong_tested += 1
            wrong_matches = _search_folio_tokens(
                wrong_tokens, predicted_triples, known_positions, eva_to_triple,
            )
            wrong_match_count += len(wrong_matches)

        null_rate = wrong_match_count / max(n_wrong_tested, 1)
        correct_rate = len(matches)

        result = {
            'folio': folio,
            'italian_name': pred.get('italian_name', ''),
            'known_fraction': pred.get('known_fraction', 0.0),
            'n_matches_correct_folio': correct_rate,
            'null_match_rate': round(null_rate, 4),
            'selectivity': round(correct_rate / max(null_rate, 0.01), 2),
            'matches': matches[:5],
            'corroborated': correct_rate > 0 and correct_rate > null_rate * 2,
        }
        all_results.append(result)

        if result['corroborated']:
            n_corroborated += 1
            print(f"    {folio}: {pred.get('italian_name', '')} — "
                  f"CORROBORATED ({correct_rate} matches, null {null_rate:.1f})")
        else:
            print(f"    {folio}: {pred.get('italian_name', '')} — "
                  f"not corroborated ({correct_rate} matches, null {null_rate:.1f})")

    # ── 4. Cross-folio validation ──
    print("\n  4. Cross-folio validation …")
    cross_val = _cross_folio_validate(all_results)
    print(f"    Consistent: {cross_val['n_consistent']}")
    print(f"    Conflicting: {cross_val['n_conflicting']}")

    # ── 5. Summary ──
    corr_rate = n_corroborated / max(len(all_results), 1)
    print(f"\n  5. Summary:")
    print(f"    Predictions tested: {len(all_results)}")
    print(f"    Corroborated: {n_corroborated} ({corr_rate:.2%})")

    # ── 6. Save ──
    elapsed = time.time() - t0

    output = {
        'n_predictions_tested': len(all_results),
        'n_corroborated': n_corroborated,
        'corroboration_rate': round(corr_rate, 4),
        'search_results': all_results,
        'cross_folio_validation': cross_val,
        'verdict': ('BOTANICAL_CONFIRMED' if n_corroborated >= 3
                    else 'BOTANICAL_PARTIAL' if n_corroborated >= 1
                    else 'BOTANICAL_UNCONFIRMED'),
        'runtime_seconds': round(elapsed, 1),
    }

    out_path = os.path.join(rd, 'botanical_search.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(output), f, indent=2)
    print(f"\n  Saved → {out_path} ({elapsed:.1f}s)")
