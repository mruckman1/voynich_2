"""
Phase 24.2 -- Error Candidate Identification (error-cand)
=========================================================
For each triple classified as ``probably_wrong`` or ``uncertain`` by
Step 24.1 (triple sensitivity analysis), systematically searches the
CV syllable inventory for replacement syllables that improve dict-hit
rate, bigram plausibility, and family coherence.

Dependency chain:
    combined_refine.json     (Phase 15 best_assignment)
    modifier_integrate.json  (Phase 16 modifier chars)
    triple_sensitivity.json  (Step 24.1)
        -> error_candidates.json (this step)
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
    build_cv_syllable_table,
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
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CandidateScore:
    syllable: str
    dict_hit: float
    dict_hit_delta: float
    bigram_plausibility: float
    bigram_delta: float
    family_coherence: float
    combined_score: float


@dataclass
class TripleErrorCandidates:
    triple_key: str
    classification: str  # from 24.1
    current_syllable: str
    n_candidates_tested: int
    best_candidate: str
    best_combined_score: float
    best_dict_hit_delta: float
    best_bigram_delta: float
    top_5_candidates: List[Dict]


@dataclass
class ErrorCandidatesResult:
    timestamp: str
    n_error_triples: int
    n_probably_wrong: int
    n_uncertain: int
    baseline_dict_hit: float
    baseline_bigram: float
    per_triple_candidates: List[Dict]
    best_single_swap: Dict  # triple with highest best_combined_score
    total_candidates_tested: int
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Decode helpers
# ---------------------------------------------------------------------------

def _decode_r3(
    tokens: List[str],
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    modifier_chars: Set[str],
    modifier_rules: Dict[str, str],
    ref_word_set: set,
) -> List[str]:
    """Decode tokens using the R3 combined strategy (alter -> strip -> original)."""
    decoded = []
    for token in tokens:
        alt = decode_token_modifier_aware(
            token, assignment, eva_to_triple, modifier_chars,
            modifier_rules=modifier_rules,
        )
        if alt.lower() in ref_word_set:
            decoded.append(alt)
            continue

        stripped = decode_token_modifier_aware(
            token, assignment, eva_to_triple, modifier_chars,
        )
        if stripped.lower() in ref_word_set:
            decoded.append(stripped)
            continue

        original = decode_token(token, assignment, eva_to_triple)
        decoded.append(original)
    return decoded


def _build_ref_bigrams(ref_words: List[str]) -> set:
    return {(ref_words[i].lower(), ref_words[i + 1].lower())
            for i in range(len(ref_words) - 1)}


def _bigram_plausibility(decoded_words: List[str], ref_bigrams: set) -> float:
    if len(decoded_words) < 2:
        return 0.0
    hits = sum(1 for i in range(len(decoded_words) - 1)
               if (decoded_words[i], decoded_words[i + 1]) in ref_bigrams)
    return hits / (len(decoded_words) - 1)


def _dict_hit_rate(decoded_words: List[str], ref_word_set: set) -> float:
    if not decoded_words:
        return 0.0
    hits = sum(1 for w in decoded_words if w.lower() in ref_word_set)
    return hits / len(decoded_words)


# ---------------------------------------------------------------------------
# Family coherence
# ---------------------------------------------------------------------------

def _build_first_stroke_groups(assignment: Dict[str, str]) -> Dict[str, List[str]]:
    """Group triple keys by their first_stroke component (onset consonant class).

    Returns dict: first_stroke -> list of triple_keys in that group.
    """
    groups: Dict[str, List[str]] = defaultdict(list)
    for triple_key in assignment:
        parts = triple_key.split(',')
        if len(parts) >= 1:
            first_stroke = parts[0]
            groups[first_stroke].append(triple_key)
    return dict(groups)


def _onset_consonant(syllable: str) -> str:
    """Extract the onset consonant(s) from a CV syllable.

    Pure-vowel syllables return empty string.
    """
    vowels = set('aeiou')
    onset = ''
    for ch in syllable.lower():
        if ch in vowels:
            break
        onset += ch
    return onset


def _compute_family_coherence(
    assignment: Dict[str, str],
    first_stroke_groups: Dict[str, List[str]],
) -> float:
    """Fraction of first_stroke groups where all triples share the same onset consonant.

    Groups with only one member are counted as coherent.
    """
    if not first_stroke_groups:
        return 1.0
    n_coherent = 0
    n_groups = 0
    for first_stroke, triple_keys in first_stroke_groups.items():
        assigned = [assignment.get(tk) for tk in triple_keys if tk in assignment]
        if not assigned:
            continue
        n_groups += 1
        onsets = set(_onset_consonant(syl) for syl in assigned)
        if len(onsets) <= 1:
            n_coherent += 1
    return n_coherent / n_groups if n_groups > 0 else 1.0


# ---------------------------------------------------------------------------
# Syllable frequency estimation
# ---------------------------------------------------------------------------

def _estimate_syllable_frequencies(
    ref_corpus_tokens: List[str],
    cv_table: List[str],
) -> Dict[str, float]:
    """Estimate relative frequency of each CV syllable in the reference corpus.

    Walks through reference words, extracts leading CV patterns, and
    normalises to a frequency distribution.
    """
    vowels = set('aeiou')
    counts: Counter = Counter()
    for word in ref_corpus_tokens:
        w = word.lower()
        i = 0
        while i < len(w):
            # Extract onset consonants
            onset = ''
            while i < len(w) and w[i] not in vowels:
                onset += w[i]
                i += 1
            # Extract nucleus vowel
            nucleus = ''
            if i < len(w) and w[i] in vowels:
                nucleus = w[i]
                i += 1
            if nucleus:
                cv = onset + nucleus
                if cv in cv_table:
                    counts[cv] += 1
                elif nucleus in cv_table:
                    # Fall back to pure vowel if onset not in inventory
                    counts[nucleus] += 1
                    i -= len(onset)  # don't skip consonants
                    i += 1
                else:
                    pass
            else:
                i += 1

    total = sum(counts.values()) or 1
    freq = {syl: counts.get(syl, 0) / total for syl in cv_table}
    return freq


def _compute_triple_corpus_frequency(
    triple_key: str,
    eva_to_triple: Dict[str, str],
    token_counter: Counter,
) -> int:
    """Count how many times EVA chars mapping to this triple appear in the corpus."""
    total = 0
    for eva_char, tk in eva_to_triple.items():
        if tk == triple_key:
            total += token_counter.get(eva_char, 0)
    return total


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_error_candidates() -> None:
    """Step 24.2: Error Candidate Identification."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE 24.2: Error Candidate Identification")
    print("=" * 70)

    rdir = _results_dir()

    # ------------------------------------------------------------------
    # 1. Load Step 24.1 results
    # ------------------------------------------------------------------
    sens_path = str(rdir / "triple_sensitivity.json")
    sens_data = _load_json(sens_path)
    if sens_data is None:
        print("  ERROR: triple_sensitivity.json not found. Run Step 24.1 first.")
        return
    sensitivities = sens_data.get('sensitivities', [])
    print(f"  Loaded {len(sensitivities)} triple sensitivities from Step 24.1")

    # ------------------------------------------------------------------
    # 2. Select error candidate triples
    # ------------------------------------------------------------------
    error_triples = [
        s for s in sensitivities
        if s.get('classification') in ('probably_wrong', 'uncertain')
    ]
    n_probably_wrong = sum(
        1 for s in error_triples if s.get('classification') == 'probably_wrong'
    )
    n_uncertain = sum(
        1 for s in error_triples if s.get('classification') == 'uncertain'
    )
    print(f"  Error candidates: {len(error_triples)} "
          f"({n_probably_wrong} probably_wrong, {n_uncertain} uncertain)")

    if not error_triples:
        print("  No error triples found. Nothing to do.")
        result = ErrorCandidatesResult(
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
            n_error_triples=0,
            n_probably_wrong=0,
            n_uncertain=0,
            baseline_dict_hit=0.0,
            baseline_bigram=0.0,
            per_triple_candidates=[],
            best_single_swap={},
            total_candidates_tested=0,
            runtime_seconds=round(time.time() - t0, 1),
        )
        out_path = rdir / "error_candidates.json"
        with open(out_path, 'w') as f:
            json.dump(_convert(asdict(result)), f, indent=2, ensure_ascii=False)
        print(f"  -> {out_path}")
        return

    # ------------------------------------------------------------------
    # 3. Load Phase 16 pipeline
    # ------------------------------------------------------------------
    print("  Loading Phase 16 pipeline...")

    # Assignment
    combined = _load_json(str(rdir / "combined_refine.json")) or {}
    assignment = dict(combined.get("best_assignment", {}))

    # Modifiers
    mod_data = _load_json(str(rdir / "modifier_integrate.json")) or {}
    modifier_chars: Set[str] = set(mod_data.get("modifier_chars", []))
    modifier_rules: Dict[str, str] = {}
    for c in mod_data.get("classifications", []):
        if c.get("final_classification") == "modifier":
            modifier_rules[c["eva_char"]] = c.get("modifier_type", "silent")

    # Reference word set
    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    base_words = set(
        w.lower() for w in ref_corpus.get_combined_tokens('latin')
        if len(w) >= 2
    )
    expanded_words, _ = build_expanded_word_set(base_words)
    ref_word_set = base_words | expanded_words
    print(f"  Dictionary: {len(ref_word_set)} words")

    # Reference bigrams
    ref_words = [w.lower() for w in ref_corpus.get_combined_tokens('latin')
                 if len(w) >= 2]
    ref_bigrams = _build_ref_bigrams(ref_words[:5000])

    # Corpus
    corpus = load_corpus(verbose=False)
    all_tokens = corpus.get_tokens()
    eva_to_triple = build_eva_to_triple_lookup()
    print(f"  Corpus: {len(all_tokens)} tokens")

    # ------------------------------------------------------------------
    # 4. CV syllable inventory
    # ------------------------------------------------------------------
    cv_table = build_cv_syllable_table('latin')
    print(f"  CV syllable inventory: {len(cv_table)} syllables")

    # ------------------------------------------------------------------
    # 5. Build token sample (5000 tokens, seeded)
    # ------------------------------------------------------------------
    rng = random.Random(42)
    if len(all_tokens) > 5000:
        sample_tokens = rng.sample(all_tokens, 5000)
    else:
        sample_tokens = list(all_tokens)
    print(f"  Sample: {len(sample_tokens)} tokens")

    # ------------------------------------------------------------------
    # Precompute corpus-level character frequencies for triple freq estimation
    # ------------------------------------------------------------------
    char_counter: Counter = Counter()
    for token in all_tokens:
        chars = tokenize_eva_chars(token)
        for ch in chars:
            char_counter[ch] += 1

    # Syllable frequency estimates from reference corpus
    syl_freq = _estimate_syllable_frequencies(ref_words[:10000], cv_table)

    # Family coherence groups
    first_stroke_groups = _build_first_stroke_groups(assignment)

    # ------------------------------------------------------------------
    # Compute baseline metrics
    # ------------------------------------------------------------------
    print("  Computing baseline metrics...")
    baseline_decoded = _decode_r3(
        sample_tokens, assignment, eva_to_triple,
        modifier_chars, modifier_rules, ref_word_set,
    )
    baseline_dict_hit = _dict_hit_rate(baseline_decoded, ref_word_set)
    baseline_decoded_lower = [w.lower() for w in baseline_decoded]
    baseline_bigram = _bigram_plausibility(baseline_decoded_lower, ref_bigrams)
    baseline_coherence = _compute_family_coherence(assignment, first_stroke_groups)
    print(f"  Baseline: dict_hit={baseline_dict_hit:.4f}, "
          f"bigram={baseline_bigram:.6f}, coherence={baseline_coherence:.4f}")

    # ------------------------------------------------------------------
    # 6. For each error candidate triple, search replacement syllables
    # ------------------------------------------------------------------
    per_triple_results: List[TripleErrorCandidates] = []
    total_candidates_tested = 0

    for idx, sens_entry in enumerate(error_triples):
        triple_key = sens_entry['triple_key']
        classification = sens_entry['classification']
        current_syl = assignment.get(triple_key, '?')

        print(f"\n  [{idx + 1}/{len(error_triples)}] {triple_key} "
              f"(current={current_syl}, class={classification})")

        # 6a. Determine syllables used by OTHER triples
        used_by_others = set()
        for tk, syl in assignment.items():
            if tk != triple_key:
                used_by_others.add(syl)

        # 6b. Generate candidate replacements
        # Compute triple's corpus frequency
        triple_freq = _compute_triple_corpus_frequency(
            triple_key, eva_to_triple, char_counter,
        )
        total_char_count = sum(char_counter.values()) or 1
        triple_rel_freq = triple_freq / total_char_count

        candidates = []
        for syl in cv_table:
            # All-different: candidate not in used set
            if syl in used_by_others:
                continue
            # Frequency compatibility: within 5x
            syl_ref_freq = syl_freq.get(syl, 0.0)
            if triple_rel_freq > 0 and syl_ref_freq > 0:
                ratio = max(triple_rel_freq, syl_ref_freq) / min(triple_rel_freq, syl_ref_freq)
                if ratio > 5.0:
                    continue
            candidates.append(syl)

        # Cap at top 30 after filtering (prefer those closest in frequency)
        if len(candidates) > 30:
            candidates.sort(
                key=lambda s: abs(syl_freq.get(s, 0.0) - triple_rel_freq)
            )
            candidates = candidates[:30]

        print(f"    {len(candidates)} candidates after filtering")

        # 6c. Score each candidate
        scored: List[CandidateScore] = []
        for cand_syl in candidates:
            # Substitute into assignment
            test_assignment = dict(assignment)
            test_assignment[triple_key] = cand_syl

            # Decode sample
            test_decoded = _decode_r3(
                sample_tokens, test_assignment, eva_to_triple,
                modifier_chars, modifier_rules, ref_word_set,
            )
            test_dict_hit = _dict_hit_rate(test_decoded, ref_word_set)
            test_decoded_lower = [w.lower() for w in test_decoded]
            test_bigram = _bigram_plausibility(test_decoded_lower, ref_bigrams)

            # Family coherence after swap
            test_coherence = _compute_family_coherence(
                test_assignment, first_stroke_groups,
            )

            # Deltas
            dict_hit_delta = test_dict_hit - baseline_dict_hit
            bigram_delta = test_bigram - baseline_bigram

            # 6d. Combined score
            combined = (
                0.5 * dict_hit_delta
                + 0.3 * bigram_delta
                + 0.2 * test_coherence
            )

            scored.append(CandidateScore(
                syllable=cand_syl,
                dict_hit=round(test_dict_hit, 6),
                dict_hit_delta=round(dict_hit_delta, 6),
                bigram_plausibility=round(test_bigram, 6),
                bigram_delta=round(bigram_delta, 6),
                family_coherence=round(test_coherence, 4),
                combined_score=round(combined, 6),
            ))
            total_candidates_tested += 1

        # 6e. Rank candidates
        scored.sort(key=lambda s: -s.combined_score)

        if scored:
            best = scored[0]
            top_5 = [_convert(asdict(s)) for s in scored[:5]]
            print(f"    Best: {best.syllable} "
                  f"(dict_hit_delta={best.dict_hit_delta:+.4f}, "
                  f"bigram_delta={best.bigram_delta:+.6f}, "
                  f"coherence={best.family_coherence:.4f}, "
                  f"combined={best.combined_score:.6f})")
        else:
            best = CandidateScore(
                syllable='', dict_hit=0.0, dict_hit_delta=0.0,
                bigram_plausibility=0.0, bigram_delta=0.0,
                family_coherence=0.0, combined_score=0.0,
            )
            top_5 = []

        per_triple_results.append(TripleErrorCandidates(
            triple_key=triple_key,
            classification=classification,
            current_syllable=current_syl,
            n_candidates_tested=len(scored),
            best_candidate=best.syllable,
            best_combined_score=round(best.combined_score, 6),
            best_dict_hit_delta=round(best.dict_hit_delta, 6),
            best_bigram_delta=round(best.bigram_delta, 6),
            top_5_candidates=top_5,
        ))

    # ------------------------------------------------------------------
    # 7. Assemble and save results
    # ------------------------------------------------------------------
    # Find best single swap across all triples
    if per_triple_results:
        best_swap_entry = max(per_triple_results, key=lambda x: x.best_combined_score)
        best_single_swap = _convert(asdict(best_swap_entry))
    else:
        best_single_swap = {}

    result = ErrorCandidatesResult(
        timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
        n_error_triples=len(error_triples),
        n_probably_wrong=n_probably_wrong,
        n_uncertain=n_uncertain,
        baseline_dict_hit=round(baseline_dict_hit, 6),
        baseline_bigram=round(baseline_bigram, 6),
        per_triple_candidates=[_convert(asdict(r)) for r in per_triple_results],
        best_single_swap=best_single_swap,
        total_candidates_tested=total_candidates_tested,
        runtime_seconds=round(time.time() - t0, 1),
    )

    out_path = rdir / "error_candidates.json"
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t0

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"  Error triples analysed: {len(error_triples)} "
          f"({n_probably_wrong} probably_wrong, {n_uncertain} uncertain)")
    print(f"  Total candidates tested: {total_candidates_tested}")
    print(f"  Baseline dict_hit: {baseline_dict_hit:.4f}")
    print(f"  Baseline bigram: {baseline_bigram:.6f}")
    if per_triple_results:
        print(f"  Best single swap: {best_swap_entry.triple_key} "
              f"-> {best_swap_entry.best_candidate} "
              f"(dict_hit_delta={best_swap_entry.best_dict_hit_delta:+.4f}, "
              f"combined={best_swap_entry.best_combined_score:.6f})")
    print(f"  -> {out_path} ({elapsed:.1f}s)")
