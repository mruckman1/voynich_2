"""
Step 33.15 -- Distributional Mapping Cross-Validation
======================================================
Cross-validates the word-level distributional mapping (Step 33.14) against
character-level findings from Phase 16.  Checks whether distributional
assignments agree with the Phase 16 decode table, proposes new triple
assignments via a reverse Ventris test, and evaluates a hybrid decode that
overlays distributional word matches on top of the character-level decode.

Dependency chain:
    distributional_match.json  (Step 33.14 — optimal_mappings)
    combined_refine.json       (Phase 15 best_assignment)
    modifier_integrate.json    (Phase 16 modifiers)
    null_corpus.json           (Phase 17 null seeds)
    signal_bigrams.json        (Phase 29.1 — baseline bigram z)
        -> distributional_validate.json  (this step)
"""

import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    load_corpus,
    token_to_triples,
    tokenize_eva_chars,
)
from voynich.core.reference import build_expanded_word_set, load_reference_corpus
from voynich.phases.csp_solver import decode_token
from voynich.phases.signal_isolation import _decode_corpus_r3
from voynich.phases.null_corpus import (
    _build_eva_bigram_model,
    _generate_null_corpus,
    _reconstruct_modifier_rules,
)


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


def _edit_distance(s1: str, s2: str) -> int:
    """Levenshtein edit distance."""
    if len(s1) < len(s2):
        return _edit_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    return prev_row[-1]


def _syllabify_approx(word: str, n_syllables: int) -> List[str]:
    """Split a Latin word into approximately n_syllables chunks.

    Uses a simple 2-char-per-syllable approximation.  If the word length
    does not divide evenly, the last chunk absorbs the remainder.
    """
    if n_syllables <= 0 or not word:
        return [word] if word else []
    chunk_size = max(1, len(word) // n_syllables)
    syllables: List[str] = []
    pos = 0
    for i in range(n_syllables):
        if i == n_syllables - 1:
            syllables.append(word[pos:])
        else:
            syllables.append(word[pos:pos + chunk_size])
            pos += chunk_size
    return [s for s in syllables if s]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TokenConvergence:
    eva_token: str
    distributional_word: str
    character_decoded: str
    edit_distance: int
    convergence: str  # 'EXACT', 'PARTIAL', 'WEAK', 'DIVERGENT'
    first_syllable_match: bool


@dataclass
class ReverseVentrisProposal:
    eva_token: str
    latin_word: str
    triple_key: str
    proposed_syllable: str
    consistent_with_confirmed: bool


@dataclass
class DistributionalValidateResult:
    # Convergence analysis
    n_mappings: int
    n_exact: int
    n_partial: int
    n_weak: int
    n_divergent: int
    convergence_details: List[Dict]
    # Reverse Ventris
    n_reverse_proposals: int
    n_consistent: int
    reverse_proposals: List[Dict]
    # Hybrid decode
    hybrid_dict_hit: float
    hybrid_signal_rate: float
    hybrid_bigram_z: float
    baseline_bigram_z: float
    delta_bigram_z: float
    # Verdict
    distributional_valid: bool
    verdict: str  # 'CONVERGENT', 'PARTIAL_CONVERGENT', 'DIVERGENT'
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_distributional_validate() -> None:
    """Step 33.15: Cross-validate distributional mapping against character-level decode."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 33.15: Distributional Mapping Cross-Validation")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load distributional_match.json ──
    print("\n  1. Loading distributional match results ...")

    dist_path = os.path.join(rd, 'distributional_match.json')
    if not os.path.exists(dist_path):
        print("  [SKIP] distributional_match.json not found")
        _save_no_signal(rd, t0)
        return

    with open(dist_path) as f:
        dist_data = json.load(f)

    significant = dist_data.get('significant', False)
    if not significant:
        print("  [SKIP] distributional_match.json reports significant=False")
        _save_no_signal(rd, t0)
        return

    optimal_mappings = dist_data.get('optimal_mappings', [])
    if not optimal_mappings:
        print("  [SKIP] No optimal_mappings in distributional_match.json")
        _save_no_signal(rd, t0)
        return

    print(f"     {len(optimal_mappings)} distributional mappings loaded")

    # ── 2. Load assignment and modifiers ──
    print("\n  2. Loading assignment and modifiers ...")

    refine_path = os.path.join(rd, 'combined_refine.json')
    if not os.path.exists(refine_path):
        print("  [SKIP] combined_refine.json not found")
        return
    with open(refine_path) as f:
        refine_data = json.load(f)
    assignment = refine_data.get('best_assignment', {})

    mod_path = os.path.join(rd, 'modifier_integrate.json')
    if not os.path.exists(mod_path):
        print("  [SKIP] modifier_integrate.json not found")
        return
    with open(mod_path) as f:
        mod_data = json.load(f)
    modifier_chars, modifier_rules = _reconstruct_modifier_rules(mod_data)

    print(f"     Assignment: {len(assignment)} triples")
    print(f"     Modifiers: {len(modifier_chars)} chars")

    # ── 3. Build reference word set and bigrams ──
    print("\n  3. Building reference word set ...")
    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    base_words = set(
        w.lower() for w in ref_corpus.get_combined_tokens('latin')
        if len(w) >= 2
    )
    expanded, _ = build_expanded_word_set(base_words)
    ref_word_set = base_words | expanded
    print(f"     {len(ref_word_set)} reference words")

    ref_tokens_raw = [
        w.lower() for w in ref_corpus.get_combined_tokens('latin')
        if len(w) >= 2
    ]
    ref_bigrams: Set[Tuple[str, str]] = set()
    for i in range(len(ref_tokens_raw) - 1):
        ref_bigrams.add((ref_tokens_raw[i], ref_tokens_raw[i + 1]))
    print(f"     {len(ref_bigrams)} reference bigrams")

    # ── 4. Load corpus and prepare structures ──
    print("\n  4. Loading corpus ...")
    corpus = load_corpus(verbose=False)
    eva_to_triple = build_eva_to_triple_lookup()

    all_tokens: List[str] = []
    token_folios: List[str] = []
    for folio, page in corpus.pages.items():
        for token in page.all_tokens:
            all_tokens.append(token)
            token_folios.append(folio)
    n_tokens = len(all_tokens)
    print(f"     {n_tokens} tokens")

    # ── 5. Cross-validate character-level ──
    print("\n  5. Cross-validating distributional mappings against Phase 16 decode ...")

    convergence_details: List[TokenConvergence] = []
    n_exact = 0
    n_partial = 0
    n_weak = 0
    n_divergent = 0

    for mapping in optimal_mappings:
        eva_tok = mapping.get('eva_token', '')
        latin_word = mapping.get('latin_word', '')
        if not eva_tok or not latin_word:
            continue

        # Decode through Phase 16 table (R3 strategy on single token)
        decoded_list = _decode_corpus_r3(
            [eva_tok], assignment, eva_to_triple,
            modifier_chars, modifier_rules, ref_word_set,
        )
        char_decoded = decoded_list[0] if decoded_list else ''

        dist = _edit_distance(char_decoded, latin_word.lower())
        first_syl_match = (
            len(char_decoded) >= 2
            and len(latin_word) >= 2
            and char_decoded[:2] == latin_word.lower()[:2]
        )

        if dist == 0:
            convergence = 'EXACT'
            n_exact += 1
        elif first_syl_match:
            convergence = 'PARTIAL'
            n_partial += 1
        elif dist <= 2:
            convergence = 'WEAK'
            n_weak += 1
        else:
            convergence = 'DIVERGENT'
            n_divergent += 1

        convergence_details.append(TokenConvergence(
            eva_token=eva_tok,
            distributional_word=latin_word,
            character_decoded=char_decoded,
            edit_distance=dist,
            convergence=convergence,
            first_syllable_match=first_syl_match,
        ))

    n_mappings = len(convergence_details)
    print(f"     {n_mappings} mappings evaluated:")
    print(f"       EXACT:     {n_exact}")
    print(f"       PARTIAL:   {n_partial}")
    print(f"       WEAK:      {n_weak}")
    print(f"       DIVERGENT: {n_divergent}")

    # Show details for non-divergent
    for cd in convergence_details:
        tag = {'EXACT': '=', 'PARTIAL': '~', 'WEAK': '?', 'DIVERGENT': 'X'}[
            cd.convergence
        ]
        print(f"       [{tag}] {cd.eva_token:12s} -> dist={cd.distributional_word:12s} "
              f"char={cd.character_decoded:12s} (ed={cd.edit_distance})")

    # ── 6. Reverse Ventris test ──
    print("\n  6. Reverse Ventris test (proposing triple assignments) ...")

    # Identify confirmed triples from bootstrap if available
    confirmed_triples: Set[str] = set()
    boot_path = os.path.join(rd, 'bootstrap_loop.json')
    if os.path.exists(boot_path):
        with open(boot_path) as f:
            boot_data = json.load(f)
        confirmed_triples = set(boot_data.get('confirmed_triples', []))
    print(f"     {len(confirmed_triples)} confirmed triples from bootstrap")

    reverse_proposals: List[ReverseVentrisProposal] = []

    for mapping in optimal_mappings:
        eva_tok = mapping.get('eva_token', '')
        latin_word = mapping.get('latin_word', '')
        if not eva_tok or not latin_word:
            continue

        # Decompose EVA token into triples
        triples = token_to_triples(eva_tok, eva_to_triple)
        if not triples:
            continue

        # Approximate syllabification of Latin word
        syllables = _syllabify_approx(latin_word.lower(), len(triples))
        if len(syllables) != len(triples):
            continue

        for triple_key, proposed_syl in zip(triples, syllables):
            if not proposed_syl:
                continue

            # Check consistency with confirmed triples
            consistent = True
            if triple_key in confirmed_triples:
                existing = assignment.get(triple_key, '')
                if existing and existing != proposed_syl:
                    consistent = False

            reverse_proposals.append(ReverseVentrisProposal(
                eva_token=eva_tok,
                latin_word=latin_word,
                triple_key=triple_key,
                proposed_syllable=proposed_syl,
                consistent_with_confirmed=consistent,
            ))

    n_reverse = len(reverse_proposals)
    n_consistent = sum(1 for rp in reverse_proposals if rp.consistent_with_confirmed)
    n_inconsistent = n_reverse - n_consistent

    print(f"     {n_reverse} triple assignment proposals")
    print(f"       Consistent with confirmed: {n_consistent}")
    print(f"       Inconsistent: {n_inconsistent}")

    # Show first 20 proposals
    for rp in reverse_proposals[:20]:
        tag = '+' if rp.consistent_with_confirmed else 'X'
        existing = assignment.get(rp.triple_key, '?')
        print(f"       [{tag}] {rp.triple_key}: {existing} -> {rp.proposed_syllable} "
              f"(from {rp.eva_token}={rp.latin_word})")

    # ── 7. Hybrid decode ──
    print("\n  7. Building hybrid decode ...")

    # Build override dict from top-20 significant distributional matches
    override_dict: Dict[str, str] = {}
    for mapping in optimal_mappings[:20]:
        eva_tok = mapping.get('eva_token', '')
        latin_word = mapping.get('latin_word', '')
        if eva_tok and latin_word:
            override_dict[eva_tok] = latin_word.lower()

    print(f"     {len(override_dict)} distributional overrides (top-20)")

    # Decode the corpus: distributional override first, then character-level
    hybrid_decoded: List[str] = []
    n_overridden = 0
    for token in all_tokens:
        if token in override_dict:
            hybrid_decoded.append(override_dict[token])
            n_overridden += 1
        else:
            # Fall through to character-level R3
            dec = _decode_corpus_r3(
                [token], assignment, eva_to_triple,
                modifier_chars, modifier_rules, ref_word_set,
            )
            hybrid_decoded.append(dec[0] if dec else '')

    hybrid_hits = [w in ref_word_set for w in hybrid_decoded]
    hybrid_dict_hit = sum(hybrid_hits) / n_tokens if n_tokens > 0 else 0.0
    print(f"     Overridden tokens: {n_overridden} / {n_tokens} "
          f"({n_overridden / n_tokens:.2%})")
    print(f"     Hybrid dict_hit: {hybrid_dict_hit:.4f}")

    # ── 8. Signal classification for hybrid decode ──
    print("\n  8. Classifying hybrid tokens (SIGNAL pipeline) ...")

    # Load null seeds
    null_seeds = [100, 101, 102, 103, 104]
    null_path = os.path.join(rd, 'null_corpus.json')
    if os.path.exists(null_path):
        with open(null_path) as f:
            null_data = json.load(f)
        null_seeds = [r['seed'] for r in null_data.get('null_runs', [])]

    # Regenerate null corpora and decode (null tokens go through character-level
    # only, since they are synthetic and will not match distributional overrides)
    bigram_probs, initial_probs, token_lengths = _build_eva_bigram_model(
        all_tokens,
    )
    null_hits_list: List[List[bool]] = []
    for i, seed in enumerate(null_seeds):
        print(f"     Null corpus {i + 1}/{len(null_seeds)} (seed={seed}) ...")
        null_tokens = _generate_null_corpus(
            bigram_probs, initial_probs, token_lengths, n_tokens, seed,
        )
        null_decoded = _decode_corpus_r3(
            null_tokens, assignment, eva_to_triple,
            modifier_chars, modifier_rules, ref_word_set,
        )
        null_hits_list.append([w in ref_word_set for w in null_decoded])

    # Classify each token
    classifications: List[str] = []
    for idx in range(n_tokens):
        r_hit = hybrid_hits[idx]
        null_hit_count = sum(1 for nh in null_hits_list if nh[idx])

        if r_hit and null_hit_count <= 1:
            classifications.append('SIGNAL')
        elif r_hit and null_hit_count >= 3:
            classifications.append('SHARED_HIT')
        elif not r_hit and null_hit_count >= 3:
            classifications.append('ANTI_SIGNAL')
        else:
            classifications.append('SHARED_MISS')

    cls_counts = Counter(classifications)
    n_signal = cls_counts.get('SIGNAL', 0)
    hybrid_signal_rate = n_signal / n_tokens if n_tokens > 0 else 0.0

    for cls in ['SIGNAL', 'SHARED_HIT', 'SHARED_MISS', 'ANTI_SIGNAL']:
        print(f"       {cls:14s}: {cls_counts.get(cls, 0):6d}")
    print(f"     Hybrid SIGNAL rate: {hybrid_signal_rate:.4f}")

    # ── 9. Bigram z-score for hybrid decode ──
    print("\n  9. Computing hybrid bigram z-score ...")

    # Find SIGNAL-SIGNAL consecutive pairs within folio boundaries
    n_signal_pairs = 0
    n_bigram_hits = 0
    for i in range(n_tokens - 1):
        if (classifications[i] == 'SIGNAL'
                and classifications[i + 1] == 'SIGNAL'
                and token_folios[i] == token_folios[i + 1]):
            n_signal_pairs += 1
            if (hybrid_decoded[i], hybrid_decoded[i + 1]) in ref_bigrams:
                n_bigram_hits += 1

    bigram_hit_rate = n_bigram_hits / n_signal_pairs if n_signal_pairs > 0 else 0.0
    print(f"     {n_signal_pairs} SIGNAL-SIGNAL pairs, {n_bigram_hits} bigram hits")
    print(f"     Bigram hit rate: {bigram_hit_rate:.6f}")

    # Null permutation test (1000 relabelings)
    rng = random.Random(42)
    indices = list(range(n_tokens))
    null_rates: List[float] = []

    for _ in range(1000):
        fake_signal = set(rng.sample(indices, min(n_signal, n_tokens)))
        n_pairs = 0
        n_hits = 0
        for i in range(n_tokens - 1):
            if (i in fake_signal and (i + 1) in fake_signal
                    and token_folios[i] == token_folios[i + 1]):
                n_pairs += 1
                if (hybrid_decoded[i], hybrid_decoded[i + 1]) in ref_bigrams:
                    n_hits += 1
        rate = n_hits / n_pairs if n_pairs > 0 else 0.0
        null_rates.append(rate)

    null_mean = sum(null_rates) / len(null_rates) if null_rates else 0.0
    null_var = (
        sum((r - null_mean) ** 2 for r in null_rates) / len(null_rates)
        if null_rates else 0.0
    )
    null_std = null_var ** 0.5

    if null_std > 0:
        hybrid_bigram_z = (bigram_hit_rate - null_mean) / null_std
    else:
        hybrid_bigram_z = float('inf') if bigram_hit_rate > null_mean else 0.0

    z_display = (
        round(hybrid_bigram_z, 2)
        if hybrid_bigram_z != float('inf') else 999.0
    )

    print(f"     Null mean: {null_mean:.6f}, std: {null_std:.6f}")
    print(f"     Hybrid bigram z: {z_display}")

    # ── 10. Load baseline for comparison ──
    print("\n  10. Comparing to Phase 29 baseline ...")
    baseline_bigram_z = 6.14
    bg_path = os.path.join(rd, 'signal_bigrams.json')
    if os.path.exists(bg_path):
        with open(bg_path) as f:
            bg_data = json.load(f)
        baseline_bigram_z = bg_data.get('bigram_z_score', 6.14)

    delta_bigram_z = z_display - baseline_bigram_z

    print(f"     Baseline bigram z: {baseline_bigram_z:.2f}")
    print(f"     Hybrid bigram z:   {z_display}")
    print(f"     Delta:             {delta_bigram_z:+.2f}")

    # ── 11. Verdict ──
    print("\n  11. Verdict ...")

    convergence_ratio = (n_exact + n_partial) / n_mappings if n_mappings > 0 else 0.0

    if n_exact >= 3 or convergence_ratio >= 0.5:
        distributional_valid = True
        verdict = 'CONVERGENT'
    elif n_exact >= 1 or (n_exact + n_partial) >= 3:
        distributional_valid = True
        verdict = 'PARTIAL_CONVERGENT'
    else:
        distributional_valid = False
        verdict = 'DIVERGENT'

    # Append key metrics to verdict string
    verdict_detail = (
        f"{verdict}: {n_exact} exact + {n_partial} partial + {n_weak} weak "
        f"+ {n_divergent} divergent out of {n_mappings} mappings; "
        f"hybrid dict_hit={hybrid_dict_hit:.4f}, "
        f"signal_rate={hybrid_signal_rate:.4f}, "
        f"bigram_z={z_display} (delta={delta_bigram_z:+.2f}); "
        f"{n_reverse} reverse proposals ({n_consistent} consistent)"
    )

    print(f"     Convergence ratio: {convergence_ratio:.2f}")
    print(f"     Distributional valid: {distributional_valid}")
    print(f"     {verdict_detail}")

    # ── 12. Save results ──
    result = DistributionalValidateResult(
        n_mappings=n_mappings,
        n_exact=n_exact,
        n_partial=n_partial,
        n_weak=n_weak,
        n_divergent=n_divergent,
        convergence_details=[_convert(asdict(cd)) for cd in convergence_details],
        n_reverse_proposals=n_reverse,
        n_consistent=n_consistent,
        reverse_proposals=[_convert(asdict(rp)) for rp in reverse_proposals],
        hybrid_dict_hit=round(hybrid_dict_hit, 6),
        hybrid_signal_rate=round(hybrid_signal_rate, 6),
        hybrid_bigram_z=z_display,
        baseline_bigram_z=round(baseline_bigram_z, 2),
        delta_bigram_z=round(delta_bigram_z, 2),
        distributional_valid=distributional_valid,
        verdict=verdict_detail,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'distributional_validate.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)
    print(f"\n  -> {out_path}")


def _save_no_signal(rd: str, t0: float) -> None:
    """Save a minimal result when distributional match data is unavailable."""
    result = DistributionalValidateResult(
        n_mappings=0,
        n_exact=0,
        n_partial=0,
        n_weak=0,
        n_divergent=0,
        convergence_details=[],
        n_reverse_proposals=0,
        n_consistent=0,
        reverse_proposals=[],
        hybrid_dict_hit=0.0,
        hybrid_signal_rate=0.0,
        hybrid_bigram_z=0.0,
        baseline_bigram_z=0.0,
        delta_bigram_z=0.0,
        distributional_valid=False,
        verdict='NO_DISTRIBUTIONAL_SIGNAL',
        runtime_seconds=round(time.time() - t0, 2),
    )
    out_path = os.path.join(rd, 'distributional_validate.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)
    print(f"\n  -> {out_path}")
