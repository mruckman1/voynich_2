"""
Phase 24.3 – Exhaustive Single-Triple Swap with Greedy Accumulation (targeted-swap)
=====================================================================================
For each error triple identified by Step 24.2, tries the single best
candidate syllable, evaluates with the full readability battery, then
greedily accumulates swaps that improve both dict-hit AND bigram
plausibility.

Critical filter: only accept swaps where bigram_plausibility does not
decrease.  Bigram plausibility is the decisive discriminator — random
tables produce ~0% bigram plausibility regardless of dict-hit rate.

Dependency chain:
    error_candidates.json (24.2) + combined_refine.json (Phase 15)
    + modifier_integrate.json (Phase 16)
        → targeted_swap.json (this step)
"""

import json
import os
import random
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
    LATIN_PHRASE_PATTERNS,
    PHARMACEUTICAL_VOCABULARY,
    build_expanded_word_set,
    load_reference_corpus,
)
from voynich.phases.csp_solver import decode_token


# ---------------------------------------------------------------------------
# Helpers
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


def _load_json(path: str) -> Optional[Dict]:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Readability helpers (adapted from readability_22.py / readability_delta.py)
# ---------------------------------------------------------------------------

def _build_ref_bigrams(ref_words: List[str]) -> set:
    return {(ref_words[i].lower(), ref_words[i + 1].lower())
            for i in range(len(ref_words) - 1)}


def _bigram_plausibility(decoded_words: List[str], ref_bigrams: set) -> float:
    if len(decoded_words) < 2:
        return 0.0
    hits = sum(1 for i in range(len(decoded_words) - 1)
               if (decoded_words[i], decoded_words[i + 1]) in ref_bigrams)
    return hits / (len(decoded_words) - 1)


_LATIN_POS_RULES = [
    (lambda w: w.endswith(('are', 'ere', 'ire', 'ari', 'eri', 'iri')), 'VERB'),
    (lambda w: w.endswith(('at', 'et', 'it', 'ant', 'ent', 'unt')), 'VERB'),
    (lambda w: w.endswith(('atur', 'etur', 'itur')), 'VERB'),
    (lambda w: w in ('in', 'de', 'ad', 'ex', 'per', 'cum', 'pro', 'sub',
                      'super', 'contra', 'inter'), 'PREP'),
    (lambda w: w in ('et', 'sed', 'aut', 'vel', 'atque', 'quod', 'quia',
                      'si', 'nec', 'neque'), 'CONJ'),
    (lambda w: w.endswith(('us', 'a', 'um', 'is', 'e', 'ius', 'ior')), 'ADJ'),
    (lambda w: w.endswith(('ae', 'arum', 'orum', 'ibus', 'ium')), 'NOUN'),
    (lambda w: len(w) >= 4, 'NOUN'),
]


def _pos_tag(word: str) -> str:
    w = word.lower()
    for rule, tag in _LATIN_POS_RULES:
        if rule(w):
            return tag
    return 'NOUN'


def _pos_trigram_validity(decoded_words: List[str], ref_pos_trigrams: set) -> float:
    if len(decoded_words) < 3:
        return 0.0
    tags = [_pos_tag(w) for w in decoded_words]
    hits = sum(1 for i in range(len(tags) - 2)
               if (tags[i], tags[i + 1], tags[i + 2]) in ref_pos_trigrams)
    total = len(tags) - 2
    return hits / total if total else 0.0


def _build_ref_pos_trigrams(ref_words: List[str]) -> set:
    tags = [_pos_tag(w.lower()) for w in ref_words]
    return {(tags[i], tags[i + 1], tags[i + 2]) for i in range(len(tags) - 2)}


def _domain_coherence(decoded_words: List[str], pharma_vocab: Dict) -> Dict[str, Dict]:
    word_set = set(w.lower() for w in decoded_words)
    results = {}
    for domain, terms in pharma_vocab.items():
        term_set = set(t.lower() for t in terms)
        hits = word_set & term_set
        results[domain] = {
            'n_terms': len(term_set),
            'n_hits': len(hits),
            'hit_rate': len(hits) / max(len(term_set), 1),
            'matched_terms': sorted(hits),
        }
    return results


def _detect_phrases(decoded_words: List[str], phrase_patterns) -> List[Dict]:
    text = ' '.join(decoded_words)
    hits = []
    for pattern_name, templates in phrase_patterns:
        for template in templates:
            if template.lower() in text:
                idx = text.index(template.lower())
                hits.append({
                    'pattern': pattern_name,
                    'template': template,
                    'position': idx,
                })
    return hits


# ---------------------------------------------------------------------------
# R3 combined decode
# ---------------------------------------------------------------------------

def _decode_r3(
    tokens: List[str],
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    modifier_chars: Set[str],
    modifier_rules: Dict[str, str],
    ref_word_set: set,
) -> List[str]:
    """Decode tokens using R3 combined strategy (alter -> strip -> original)."""
    decoded = []
    for token in tokens:
        # Try alteration first
        alt = decode_token_modifier_aware(
            token, assignment, eva_to_triple, modifier_chars,
            modifier_rules=modifier_rules,
        )
        if alt.lower() in ref_word_set:
            decoded.append(alt)
            continue

        # Try stripping
        stripped = decode_token_modifier_aware(
            token, assignment, eva_to_triple, modifier_chars,
        )
        if stripped.lower() in ref_word_set:
            decoded.append(stripped)
            continue

        # Fall back to original decoding
        original = decode_token(token, assignment, eva_to_triple)
        decoded.append(original)

    return decoded


# ---------------------------------------------------------------------------
# Full readability battery
# ---------------------------------------------------------------------------

def _run_readability(
    decoded_words: List[str],
    ref_word_set: set,
    ref_bigrams: set,
    ref_pos_trigrams: set,
) -> Dict[str, Any]:
    """Run the readability battery and return a flat metrics dict."""
    n_total = len(decoded_words)
    dict_words = [w for w in decoded_words if w.lower() in ref_word_set]
    dict_hit = len(dict_words) / n_total if n_total > 0 else 0.0

    # Use dict-hitting words (in original order) for readability
    analysis_words = [w.lower() for w in dict_words]

    bg = _bigram_plausibility(analysis_words, ref_bigrams)
    pos_val = _pos_trigram_validity(analysis_words, ref_pos_trigrams)

    all_decoded = [w.lower() for w in decoded_words]
    domain_results = _domain_coherence(all_decoded, PHARMACEUTICAL_VOCABULARY)
    n_domains = sum(1 for d in domain_results.values() if d['n_hits'] > 0)

    phrase_hits = _detect_phrases(analysis_words, LATIN_PHRASE_PATTERNS)

    return {
        'dict_hit': dict_hit,
        'bigram_plausibility': bg,
        'pos_validity': pos_val,
        'n_domain_hits': n_domains,
        'n_phrase_hits': len(phrase_hits),
    }


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SwapEvaluation:
    triple_key: str
    old_syllable: str
    new_syllable: str
    dict_hit: float
    dict_hit_delta: float
    bigram_plausibility: float
    bigram_delta: float
    pos_validity: float
    n_domain_hits: int
    n_phrase_hits: int
    accepted: bool
    rejection_reason: str  # "" if accepted


@dataclass
class GreedyStep:
    step: int
    triple_swapped: str
    old_syllable: str
    new_syllable: str
    cumulative_dict_hit: float
    cumulative_bigram: float
    n_total_swaps: int


@dataclass
class TargetedSwapResult:
    timestamp: str
    baseline_dict_hit: float
    baseline_bigram: float
    baseline_pos_validity: float
    baseline_n_domains: int
    baseline_n_phrases: int
    n_candidates_evaluated: int
    n_accepted: int
    n_rejected: int
    swap_evaluations: List[Dict]
    greedy_sequence: List[Dict]
    final_dict_hit: float
    final_bigram: float
    improvement_dict_hit: float
    improvement_bigram: float
    final_assignment: Dict[str, str]  # the corrected table
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_targeted_swap() -> None:
    """Step 24.3: Exhaustive single-triple swap with greedy accumulation."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE 24.3: Exhaustive Single-Triple Swap")
    print("=" * 70)

    rdir = _results_dir()

    # ------------------------------------------------------------------
    # Load error candidates from Step 24.2
    # ------------------------------------------------------------------
    ec_path = str(rdir / "error_candidates.json")
    ec_data = _load_json(ec_path)
    if ec_data is None:
        print(f"  ERROR: {ec_path} not found. Run Step 24.2 first.")
        return
    per_triple = ec_data.get('per_triple_candidates', [])
    print(f"  Loaded {len(per_triple)} error-triple candidates from 24.2")

    # ------------------------------------------------------------------
    # Load Phase 16 pipeline
    # ------------------------------------------------------------------
    combined = _load_json(str(rdir / "combined_refine.json")) or {}
    assignment = dict(combined.get('best_assignment', {}))

    mod_data = _load_json(str(rdir / "modifier_integrate.json")) or {}
    modifier_chars: Set[str] = set(mod_data.get('modifier_chars', []))
    modifier_rules: Dict[str, str] = {}
    for c in mod_data.get('classifications', []):
        if c.get('final_classification') == 'modifier':
            modifier_rules[c['eva_char']] = c.get('modifier_type', 'silent')

    print(f"  Phase 16 assignment: {len(assignment)} triples")
    print(f"  Modifier chars: {len(modifier_chars)}")

    # ------------------------------------------------------------------
    # Build reference word set and readability infrastructure
    # ------------------------------------------------------------------
    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    base_words = set(
        w.lower() for w in ref_corpus.get_combined_tokens('latin')
        if len(w) >= 2
    )
    expanded_words, _ = build_expanded_word_set(base_words)
    ref_word_set = base_words | expanded_words
    print(f"  Reference dictionary: {len(ref_word_set)} words")

    ref_words = [w.lower() for w in ref_corpus.get_combined_tokens('latin')
                 if len(w) >= 2]
    ref_bigrams = _build_ref_bigrams(ref_words[:50000])
    ref_pos_trigrams = _build_ref_pos_trigrams(ref_words[:20000])
    print(f"  Reference bigrams: {len(ref_bigrams)}")

    # ------------------------------------------------------------------
    # Build 5000-token sample (same seed as other phases)
    # ------------------------------------------------------------------
    corpus = load_corpus(verbose=False)
    all_tokens = corpus.get_tokens()
    eva_to_triple = build_eva_to_triple_lookup()

    rng = random.Random(42)
    if len(all_tokens) > 5000:
        sample_indices = sorted(rng.sample(range(len(all_tokens)), 5000))
        sample_tokens = [all_tokens[i] for i in sample_indices]
    else:
        sample_tokens = list(all_tokens)
    print(f"  Sample: {len(sample_tokens)} tokens")

    # ------------------------------------------------------------------
    # Compute Phase 16 baseline readability
    # ------------------------------------------------------------------
    print("\n  Computing Phase 16 baseline...")
    baseline_decoded = _decode_r3(
        sample_tokens, assignment, eva_to_triple,
        modifier_chars, modifier_rules, ref_word_set,
    )
    baseline_metrics = _run_readability(
        baseline_decoded, ref_word_set, ref_bigrams, ref_pos_trigrams,
    )
    baseline_dict_hit = baseline_metrics['dict_hit']
    baseline_bigram = baseline_metrics['bigram_plausibility']
    baseline_pos = baseline_metrics['pos_validity']
    baseline_domains = baseline_metrics['n_domain_hits']
    baseline_phrases = baseline_metrics['n_phrase_hits']

    print(f"  Baseline dict_hit:  {baseline_dict_hit:.4f}")
    print(f"  Baseline bigram:    {baseline_bigram:.6f}")
    print(f"  Baseline POS:       {baseline_pos:.4f}")
    print(f"  Baseline domains:   {baseline_domains}")
    print(f"  Baseline phrases:   {baseline_phrases}")

    # ------------------------------------------------------------------
    # Evaluate each error triple's best candidate swap
    # ------------------------------------------------------------------
    print(f"\n  Evaluating {len(per_triple)} single-triple swaps...")
    swap_evals: List[SwapEvaluation] = []

    for idx, entry in enumerate(per_triple):
        triple_key = entry.get('triple_key', '')
        # Get the best candidate from this entry
        candidates = entry.get('candidates', [])
        if not candidates:
            continue

        # Take the top candidate (first in the sorted list)
        best_cand = candidates[0]
        new_syllable = best_cand.get('syllable', '')
        if not new_syllable or not triple_key:
            continue

        old_syllable = assignment.get(triple_key, '')
        if new_syllable == old_syllable:
            continue  # no change

        # Apply the single swap
        test_assignment = dict(assignment)
        test_assignment[triple_key] = new_syllable

        # Decode full sample with modified assignment
        test_decoded = _decode_r3(
            sample_tokens, test_assignment, eva_to_triple,
            modifier_chars, modifier_rules, ref_word_set,
        )
        test_metrics = _run_readability(
            test_decoded, ref_word_set, ref_bigrams, ref_pos_trigrams,
        )

        dict_hit = test_metrics['dict_hit']
        bigram = test_metrics['bigram_plausibility']
        pos_val = test_metrics['pos_validity']
        n_domains = test_metrics['n_domain_hits']
        n_phrases = test_metrics['n_phrase_hits']

        dict_delta = dict_hit - baseline_dict_hit
        bigram_delta = bigram - baseline_bigram

        # Apply acceptance filter
        accepted = False
        rejection_reason = ""

        if bigram >= baseline_bigram and dict_hit > baseline_dict_hit:
            accepted = True
        elif bigram > baseline_bigram:
            # Accept if bigram improves even if dict_hit stays same
            accepted = True
        else:
            if bigram < baseline_bigram:
                rejection_reason = (
                    f"bigram decreased: {bigram:.6f} < {baseline_bigram:.6f}"
                )
            elif dict_hit <= baseline_dict_hit:
                rejection_reason = (
                    f"dict_hit not improved: {dict_hit:.4f} <= {baseline_dict_hit:.4f} "
                    f"and bigram not improved"
                )

        se = SwapEvaluation(
            triple_key=triple_key,
            old_syllable=old_syllable,
            new_syllable=new_syllable,
            dict_hit=round(dict_hit, 6),
            dict_hit_delta=round(dict_delta, 6),
            bigram_plausibility=round(bigram, 8),
            bigram_delta=round(bigram_delta, 8),
            pos_validity=round(pos_val, 6),
            n_domain_hits=n_domains,
            n_phrase_hits=n_phrases,
            accepted=accepted,
            rejection_reason=rejection_reason,
        )
        swap_evals.append(se)

        status = "ACCEPT" if accepted else "reject"
        print(f"    [{idx + 1}/{len(per_triple)}] {triple_key}: "
              f"{old_syllable}->{new_syllable}  "
              f"dict={dict_hit:.4f}({dict_delta:+.4f}) "
              f"bg={bigram:.6f}({bigram_delta:+.6f})  "
              f"[{status}]")

    n_accepted = sum(1 for se in swap_evals if se.accepted)
    n_rejected = sum(1 for se in swap_evals if not se.accepted)
    print(f"\n  Evaluated: {len(swap_evals)} | "
          f"Accepted: {n_accepted} | Rejected: {n_rejected}")

    # ------------------------------------------------------------------
    # Greedy accumulation
    # ------------------------------------------------------------------
    print("\n  Greedy accumulation...")

    greedy_assignment = dict(assignment)
    greedy_dict_hit = baseline_dict_hit
    greedy_bigram = baseline_bigram
    greedy_sequence: List[GreedyStep] = []
    used_triples: Set[str] = set()

    accepted_swaps = [se for se in swap_evals if se.accepted]

    step_num = 0
    progress = True
    while progress and accepted_swaps:
        progress = False

        # Sort accepted swaps by combined improvement score
        # Weight bigram improvement more heavily since it is the decisive
        # discriminator.  Normalize dict_delta by a reference scale.
        def _combined_score(se: SwapEvaluation) -> float:
            # Re-evaluate this swap against current greedy baseline
            # We use the original deltas for initial sorting, but the
            # actual test below re-decodes against the current table.
            return (se.bigram_delta * 10000.0) + (se.dict_hit_delta * 100.0)

        accepted_swaps.sort(key=_combined_score, reverse=True)

        for se in list(accepted_swaps):
            if se.triple_key in used_triples:
                continue

            # Apply this swap to the current greedy assignment
            test_table = dict(greedy_assignment)
            test_table[se.triple_key] = se.new_syllable

            # Re-decode and re-evaluate against current greedy state
            test_decoded = _decode_r3(
                sample_tokens, test_table, eva_to_triple,
                modifier_chars, modifier_rules, ref_word_set,
            )
            test_metrics = _run_readability(
                test_decoded, ref_word_set, ref_bigrams, ref_pos_trigrams,
            )
            new_dict = test_metrics['dict_hit']
            new_bigram = test_metrics['bigram_plausibility']

            # Accept only if improvement against current greedy baseline
            if (new_bigram >= greedy_bigram and new_dict > greedy_dict_hit) or \
               (new_bigram > greedy_bigram):
                step_num += 1
                old_syl = greedy_assignment.get(se.triple_key, '')
                greedy_assignment[se.triple_key] = se.new_syllable
                greedy_dict_hit = new_dict
                greedy_bigram = new_bigram
                used_triples.add(se.triple_key)

                gs = GreedyStep(
                    step=step_num,
                    triple_swapped=se.triple_key,
                    old_syllable=old_syl,
                    new_syllable=se.new_syllable,
                    cumulative_dict_hit=round(new_dict, 6),
                    cumulative_bigram=round(new_bigram, 8),
                    n_total_swaps=step_num,
                )
                greedy_sequence.append(gs)

                print(f"    Step {step_num}: {se.triple_key} "
                      f"{old_syl}->{se.new_syllable}  "
                      f"dict={new_dict:.4f} bg={new_bigram:.6f}")

                progress = True
                # Remove from candidates and restart the loop with updated
                # table to re-sort remaining swaps.
                accepted_swaps = [
                    s for s in accepted_swaps
                    if s.triple_key not in used_triples
                ]
                break  # restart while loop

    if not greedy_sequence:
        print("    No greedy swaps applied.")

    # ------------------------------------------------------------------
    # Final metrics
    # ------------------------------------------------------------------
    final_dict_hit = greedy_dict_hit
    final_bigram = greedy_bigram
    improvement_dict = final_dict_hit - baseline_dict_hit
    improvement_bigram = final_bigram - baseline_bigram

    print(f"\n  Final dict_hit:  {final_dict_hit:.4f} "
          f"(delta={improvement_dict:+.4f})")
    print(f"  Final bigram:    {final_bigram:.6f} "
          f"(delta={improvement_bigram:+.6f})")
    print(f"  Total swaps applied: {len(greedy_sequence)}")

    # ------------------------------------------------------------------
    # Build and save result
    # ------------------------------------------------------------------
    elapsed = time.time() - t0

    result = TargetedSwapResult(
        timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
        baseline_dict_hit=round(baseline_dict_hit, 6),
        baseline_bigram=round(baseline_bigram, 8),
        baseline_pos_validity=round(baseline_pos, 6),
        baseline_n_domains=baseline_domains,
        baseline_n_phrases=baseline_phrases,
        n_candidates_evaluated=len(swap_evals),
        n_accepted=n_accepted,
        n_rejected=n_rejected,
        swap_evaluations=[_convert(asdict(se)) for se in swap_evals],
        greedy_sequence=[_convert(asdict(gs)) for gs in greedy_sequence],
        final_dict_hit=round(final_dict_hit, 6),
        final_bigram=round(final_bigram, 8),
        improvement_dict_hit=round(improvement_dict, 6),
        improvement_bigram=round(improvement_bigram, 8),
        final_assignment=greedy_assignment,
        runtime_seconds=round(elapsed, 2),
    )

    out_path = rdir / "targeted_swap.json"
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2, ensure_ascii=False)

    print(f"\n  Saved to {out_path} ({elapsed:.1f}s)")
    print("=" * 70)
