"""
Step 41.15 -- Botanical Predictions V2
========================================
Search botanical folios for EVA tokens matching predicted forms from
Step 41.14.  For each prediction with known_fraction >= 0.5, search
the corresponding folio's tokens, decode them, and check if they match
at the known positions.  Run null controls on wrong folios and check
for cross-folio consistency.

Dependency chain:
    drosera_propagation.json  (Step 41.14)
    combined_refine.json      (Phase 15)
    modifier_integrate.json   (Phase 16)
        -> botanical_predictions_v2.json  (this step)
"""

import json
import os
import random
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    load_corpus,
    token_to_triples,
    tokenize_eva_chars,
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
# Matching logic
# ---------------------------------------------------------------------------

def _match_token_to_prediction(
    token: str,
    predicted_triples: List[str],
    known_positions: List[int],
    eva_to_triple: Dict[str, str],
    modifier_chars: Set[str],
) -> Optional[Dict]:
    """Check if a token matches a prediction at known positions.

    Returns match info dict if matched, None otherwise.
    The token's EVA chars are decomposed into triples (modifier chars
    are skipped).  Then the triple at each known position is compared
    to the predicted triple.
    """
    # Decompose token into EVA chars, filtering modifiers
    chars = tokenize_eva_chars(token)
    syllabic_chars = [ch for ch in chars if ch not in modifier_chars]
    token_triples = [eva_to_triple.get(ch, ch) for ch in syllabic_chars]

    if not token_triples:
        return None

    # Check length compatibility: predicted form has n_syllables positions;
    # token_triples should be close to that length
    n_pred = len(predicted_triples)
    n_tok = len(token_triples)

    if abs(n_tok - n_pred) > 1:
        return None

    # Score: count how many known positions match
    n_matched = 0
    n_checked = 0
    matched_details: List[Dict] = []

    for pos in known_positions:
        if pos >= n_tok or pos >= n_pred:
            continue
        n_checked += 1
        pred_triple = predicted_triples[pos]
        tok_triple = token_triples[pos]
        is_match = (pred_triple == tok_triple)
        if is_match:
            n_matched += 1
        matched_details.append({
            'position': pos,
            'predicted': pred_triple,
            'actual': tok_triple,
            'match': is_match,
        })

    if n_checked == 0:
        return None

    match_score = n_matched / n_checked

    # Only count as a match if ALL known positions match
    if n_matched < n_checked:
        return None

    return {
        'token': token,
        'token_triples': token_triples,
        'n_matched': n_matched,
        'n_checked': n_checked,
        'match_score': round(match_score, 4),
        'matched_details': matched_details,
        'n_token_triples': n_tok,
        'n_predicted_triples': n_pred,
    }


def _search_folio_for_prediction(
    folio_tokens: List[str],
    predicted_triples: List[str],
    known_positions: List[int],
    eva_to_triple: Dict[str, str],
    modifier_chars: Set[str],
) -> List[Dict]:
    """Search all tokens on a folio for matches to a prediction."""
    matches = []
    for idx, token in enumerate(folio_tokens):
        m = _match_token_to_prediction(
            token, predicted_triples, known_positions,
            eva_to_triple, modifier_chars,
        )
        if m is not None:
            m['token_position'] = idx
            matches.append(m)
    return matches


# ---------------------------------------------------------------------------
# Cross-folio validation
# ---------------------------------------------------------------------------

def _extract_implied_assignments(
    matches: List[Dict],
    predicted_triples: List[str],
    known_positions: List[int],
    syllables: List[str],
) -> Dict[str, str]:
    """From matches, extract implied new triple assignments.

    For positions that were UNKNOWN in the prediction but present in
    matched tokens, we can tentatively assign the token's triple to
    the corresponding syllable.
    """
    implied: Dict[str, str] = {}
    unknown_positions = set(range(len(predicted_triples))) - set(known_positions)

    for m in matches:
        tok_triples = m.get('token_triples', [])
        for pos in unknown_positions:
            if pos < len(tok_triples) and pos < len(syllables):
                triple = tok_triples[pos]
                syl = syllables[pos]
                if triple and syl:
                    if triple not in implied:
                        implied[triple] = syl
                    elif implied[triple] != syl:
                        # Conflicting assignment -- mark as ambiguous
                        implied[triple] = f"CONFLICT:{implied[triple]}/{syl}"

    return implied


def _cross_folio_validate(
    all_implied: List[Dict],
) -> Dict:
    """Check if implied assignments from different folios are consistent.

    If two different plants on different folios imply the same new
    triple -> syllable assignment, that is a confirmed discovery.
    """
    # triple_key -> {syllable -> list of (folio, plant_name)}
    triple_evidence: Dict[str, Dict[str, List[Tuple[str, str]]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for entry in all_implied:
        folio = entry.get('folio', '')
        plant = entry.get('plant_name', '')
        for triple, syl in entry.get('implied', {}).items():
            if 'CONFLICT' in str(syl):
                continue
            triple_evidence[triple][syl].append((folio, plant))

    confirmed: List[Dict] = []
    conflicting: List[Dict] = []

    for triple, syl_sources in triple_evidence.items():
        syllables = list(syl_sources.keys())
        if len(syllables) == 1:
            syl = syllables[0]
            sources = syl_sources[syl]
            # Check if evidence comes from at least 2 different folios
            unique_folios = set(f for f, _ in sources)
            if len(unique_folios) >= 2:
                confirmed.append({
                    'triple_key': triple,
                    'syllable': syl,
                    'n_folios': len(unique_folios),
                    'supporting_folios': sorted(unique_folios),
                    'sources': [
                        {'folio': f, 'plant': p} for f, p in sources
                    ],
                })
        elif len(syllables) > 1:
            conflicting.append({
                'triple_key': triple,
                'competing_syllables': {
                    s: [{'folio': f, 'plant': p} for f, p in srcs]
                    for s, srcs in syl_sources.items()
                },
            })

    return {
        'n_triples_with_evidence': len(triple_evidence),
        'n_confirmed': len(confirmed),
        'n_conflicting': len(conflicting),
        'confirmed_assignments': confirmed,
        'conflicting_assignments': conflicting,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_botanical_predictions_v2() -> None:
    """Step 41.15: Search folios for predicted botanical EVA forms."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 41.15: Botanical Predictions V2")
    print("=" * 70)

    rd = _results_dir()

    # -- 1. Load inputs --
    print("\n  1. Loading inputs ...")

    prop_data = _safe_load(os.path.join(rd, 'drosera_propagation.json'))
    if not prop_data or not prop_data.get('predictions'):
        print("     SKIP: drosera_propagation.json not found or empty.")
        print("     Saving minimal output.")
        output = {
            'skip_reason': 'drosera_propagation.json not found or no predictions',
            'n_predictions_tested': 0,
            'search_results': [],
            'runtime_seconds': round(time.time() - t0, 1),
        }
        out_path = os.path.join(rd, 'botanical_predictions_v2.json')
        with open(out_path, 'w') as f:
            json.dump(_convert(output), f, indent=2)
        print(f"\n  Saved -> {out_path}")
        return

    predictions = prop_data.get('predictions', [])

    refine = _safe_load(os.path.join(rd, 'combined_refine.json'))
    assignment = refine.get('best_assignment', {})

    mod_data = _safe_load(os.path.join(rd, 'modifier_integrate.json'))
    modifier_chars: Set[str] = set(mod_data.get('modifier_chars', []))

    print(f"     Predictions loaded: {len(predictions)}")
    print(f"     Assignment triples: {len(assignment)}")
    print(f"     Modifier chars: {len(modifier_chars)}")

    # -- 2. Load corpus --
    print("\n  2. Loading corpus ...")

    corpus = load_corpus(verbose=False)
    eva_to_triple = build_eva_to_triple_lookup()

    folio_tokens: Dict[str, List[str]] = {}
    for folio, page in corpus.pages.items():
        folio_tokens[folio] = page.all_tokens
    all_folios = sorted(folio_tokens.keys())
    print(f"     Folios loaded: {len(folio_tokens)}")

    # -- 3. Filter predictions with known_fraction >= 0.5 --
    print("\n  3. Filtering testable predictions ...")

    testable = [
        p for p in predictions
        if p.get('known_fraction', 0.0) >= 0.5
        and p.get('known_positions')
        and p.get('folio') in folio_tokens
    ]
    print(f"     Testable predictions (known >= 50%): {len(testable)}")

    # -- 4. Search each prediction on its folio --
    print("\n  4. Searching folios for predicted forms ...")

    search_results: List[Dict] = []
    all_implied: List[Dict] = []
    n_with_matches = 0

    # Seed RNG for reproducibility in null control
    rng = random.Random(42)

    for pred in testable:
        folio = pred['folio']
        predicted_triples = pred['predicted_triples']
        known_positions = pred['known_positions']
        italian_name = pred.get('italian_name', '')
        syllables = pred.get('syllables', [])

        tokens = folio_tokens.get(folio, [])

        # Search correct folio
        matches = _search_folio_for_prediction(
            tokens, predicted_triples, known_positions,
            eva_to_triple, modifier_chars,
        )

        # -- Null control: search 3 random wrong folios --
        wrong_folios = [f for f in all_folios if f != folio]
        if len(wrong_folios) > 3:
            wrong_sample = rng.sample(wrong_folios, 3)
        else:
            wrong_sample = wrong_folios

        wrong_match_counts: List[int] = []
        for wf in wrong_sample:
            wrong_tokens = folio_tokens.get(wf, [])
            wrong_matches = _search_folio_for_prediction(
                wrong_tokens, predicted_triples, known_positions,
                eva_to_triple, modifier_chars,
            )
            wrong_match_counts.append(len(wrong_matches))

        mean_wrong = (
            sum(wrong_match_counts) / len(wrong_match_counts)
            if wrong_match_counts else 0.0
        )

        correct_count = len(matches)
        selectivity = (
            correct_count / max(mean_wrong, 0.01)
            if correct_count > 0 else 0.0
        )

        # Extract implied assignments from unknown positions
        implied = _extract_implied_assignments(
            matches, predicted_triples, known_positions, syllables,
        )
        if implied:
            all_implied.append({
                'folio': folio,
                'plant_name': italian_name,
                'implied': implied,
            })

        result = {
            'folio': folio,
            'italian_name': italian_name,
            'known_fraction': pred.get('known_fraction', 0.0),
            'n_syllables': len(syllables),
            'syllables': syllables,
            'predicted_triples': predicted_triples,
            'known_positions': known_positions,
            'n_matches_correct_folio': correct_count,
            'mean_matches_wrong_folios': round(mean_wrong, 2),
            'wrong_folio_counts': wrong_match_counts,
            'selectivity': round(selectivity, 2),
            'matches': matches[:10],  # Cap stored matches
            'implied_new_assignments': implied,
            'corroborated': correct_count > 0 and correct_count > mean_wrong * 2,
        }
        search_results.append(result)

        if correct_count > 0:
            n_with_matches += 1

        status = "MATCH" if correct_count > 0 else "no match"
        corr = " CORROBORATED" if result['corroborated'] else ""
        print(f"     {folio}: {italian_name} -> {status} "
              f"({correct_count} correct, {mean_wrong:.1f} wrong avg)"
              f"{corr}")

    # -- 5. Cross-folio validation --
    print("\n  5. Cross-folio validation ...")

    cross_val = _cross_folio_validate(all_implied)
    n_confirmed = cross_val['n_confirmed']
    n_conflicting = cross_val['n_conflicting']

    print(f"     Triples with cross-folio evidence: "
          f"{cross_val['n_triples_with_evidence']}")
    print(f"     Confirmed (2+ folios agree): {n_confirmed}")
    print(f"     Conflicting: {n_conflicting}")

    for ca in cross_val.get('confirmed_assignments', []):
        print(f"       {ca['triple_key']} -> '{ca['syllable']}' "
              f"(from {ca['n_folios']} folios: "
              f"{', '.join(ca['supporting_folios'])})")

    # -- 6. Summary --
    print("\n  6. Summary ...")

    n_tested = len(search_results)
    n_corroborated = sum(1 for r in search_results if r['corroborated'])
    corr_rate = n_corroborated / max(n_tested, 1)

    total_correct_matches = sum(
        r['n_matches_correct_folio'] for r in search_results
    )
    total_wrong_avg = sum(
        r['mean_matches_wrong_folios'] for r in search_results
    )
    overall_selectivity = (
        total_correct_matches / max(total_wrong_avg, 0.01)
        if total_correct_matches > 0 else 0.0
    )

    print(f"     Predictions tested: {n_tested}")
    print(f"     With matches on correct folio: {n_with_matches}")
    print(f"     Corroborated (correct > 2x wrong): {n_corroborated}")
    print(f"     Corroboration rate: {corr_rate:.2%}")
    print(f"     Overall selectivity: {overall_selectivity:.2f}x")
    print(f"     Cross-folio confirmed assignments: {n_confirmed}")

    # Verdict
    if n_confirmed >= 2 and n_corroborated >= 3:
        verdict = (f"BOTANICAL_V2_CONFIRMED "
                   f"({n_confirmed} cross-folio, {n_corroborated} corroborated)")
    elif n_corroborated >= 1 or n_confirmed >= 1:
        verdict = (f"BOTANICAL_V2_PARTIAL "
                   f"({n_confirmed} cross-folio, {n_corroborated} corroborated)")
    else:
        verdict = "BOTANICAL_V2_UNCONFIRMED"

    print(f"     Verdict: {verdict}")

    # -- 7. Save --
    elapsed = time.time() - t0

    output = {
        'n_predictions_total': len(predictions),
        'n_predictions_tested': n_tested,
        'n_with_matches': n_with_matches,
        'n_corroborated': n_corroborated,
        'corroboration_rate': round(corr_rate, 4),
        'overall_selectivity': round(overall_selectivity, 2),
        'search_results': search_results,
        'cross_folio_validation': cross_val,
        'n_cross_folio_confirmed': n_confirmed,
        'n_cross_folio_conflicting': n_conflicting,
        'all_implied_assignments': all_implied,
        'verdict': verdict,
        'runtime_seconds': round(elapsed, 1),
    }

    out_path = os.path.join(rd, 'botanical_predictions_v2.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(output), f, indent=2)
    print(f"\n  Saved -> {out_path} ({elapsed:.1f}s)")
