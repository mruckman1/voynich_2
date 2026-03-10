"""
Step 34.16 – Vowel Pointer Test (Track F)
==========================================
Tests whether high-frequency short EVA tokens function as floating
vowel markers ("vowel pointers") that attach to preceding root tokens
to form complete Latin words.

Candidates: EVA tokens that are <= 2 chars long, appear >= 50 times,
are not confirmed SIGNAL words, and decode to short syllables
containing vowels.

Dependency chain:
    combined_refine.json       (Phase 15 assignment)
    modifier_integrate.json    (Phase 16 modifiers)
    signal_bigrams.json        (Phase 29 token classifications)
        → vowel_pointer_test.json   (this step)
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
    load_corpus,
    tokenize_eva_chars,
)
from voynich.core.reference import (
    EVA_VISUAL_COMPONENTS,
    build_expanded_word_set,
    load_reference_corpus,
)
from voynich.phases.csp_solver import decode_token
from voynich.phases.null_corpus import _reconstruct_modifier_rules
from voynich.phases.signal_isolation import _decode_corpus_r3


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


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LATIN_VOWELS = set('aeiou')

MIN_VP_FREQUENCY = 50
MAX_VP_CHARS = 2


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class VowelPointerCandidate:
    """A candidate vowel pointer token."""
    eva_token: str
    frequency: int
    decoded: str
    has_vowel: bool
    is_signal: bool
    # Attachment test results
    n_attachment_tests: int
    attachment_hit_rate: float      # T+VP concatenation hit rate
    baseline_hit_rate: float        # T alone hit rate
    random_hit_rate: float          # T + random_short hit rate
    attachment_improvement: float   # attachment_hit_rate - baseline_hit_rate
    random_improvement: float       # random_hit_rate - baseline_hit_rate
    is_confirmed_vp: bool           # improvement significant


@dataclass
class VowelDistributionTest:
    """Test: do VPs appear more after consonant-heavy roots?"""
    n_vp_following_consonant_heavy: int
    n_vp_following_other: int
    consonant_heavy_rate: float
    baseline_consonant_heavy_rate: float
    enrichment: float
    is_significant: bool


@dataclass
class SectionUniformityTest:
    """Test: does attachment improvement hold across sections?"""
    section: str
    n_pairs: int
    attachment_hit_rate: float
    baseline_hit_rate: float
    improvement: float


@dataclass
class VowelPointerResult:
    # Candidate identification
    n_candidates: int
    n_confirmed: int
    candidates: List[Dict]
    confirmed_vps: List[str]

    # Attachment test aggregate
    mean_attachment_improvement: float
    mean_random_improvement: float
    attachment_vs_random_z: float

    # Vowel distribution correlation
    vowel_distribution: Dict

    # Section uniformity
    section_uniformity: List[Dict]
    n_sections_improved: int
    n_sections_total: int

    # Baseline comparison
    baseline_dict_hit: float
    vp_attached_dict_hit: float
    delta_dict_hit: float

    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Vowel pointer identification
# ---------------------------------------------------------------------------

def _identify_vp_candidates(
    all_tokens: List[str],
    decoded: List[str],
    token_classifications: List[str],
    ref_word_set: set,
    eva_to_triple: Dict[str, str],
) -> List[VowelPointerCandidate]:
    """Identify candidate vowel pointer tokens.

    Criteria:
    - <= 2 EVA chars long
    - Appears >= 50 times
    - Not classified as SIGNAL
    - Decodes to a string containing at least one vowel
    """
    # Count token frequencies
    token_freqs: Counter = Counter(all_tokens)

    # Build classification lookup: token → most common classification
    token_cls_counts: Dict[str, Counter] = defaultdict(Counter)
    for tok, cls in zip(all_tokens, token_classifications):
        token_cls_counts[tok][cls] += 1

    # Identify candidates
    seen_tokens: Set[str] = set()
    candidates: List[str] = []

    for token, freq in token_freqs.most_common():
        if token in seen_tokens:
            continue
        seen_tokens.add(token)

        eva_chars = tokenize_eva_chars(token)
        if len(eva_chars) > MAX_VP_CHARS:
            continue
        if freq < MIN_VP_FREQUENCY:
            break  # sorted by frequency, so all following are below threshold

        # Check if it is SIGNAL
        cls_counts = token_cls_counts.get(token, Counter())
        dominant_cls = cls_counts.most_common(1)[0][0] if cls_counts else 'SHARED_MISS'
        if dominant_cls == 'SIGNAL':
            continue

        candidates.append(token)

    return candidates


def _decode_single_token(
    token: str,
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    modifier_chars: set,
    modifier_rules: Dict[str, str],
    ref_word_set: set,
) -> str:
    """Decode a single token using R3 strategy."""
    result = _decode_corpus_r3(
        [token], assignment, eva_to_triple,
        modifier_chars, modifier_rules, ref_word_set,
    )
    return result[0]


def _test_attachment(
    vp_token: str,
    vp_decoded: str,
    all_tokens: List[str],
    decoded: List[str],
    token_folios: List[str],
    ref_word_set: set,
    rng: random.Random,
    short_decoded_pool: List[str],
) -> Tuple[int, float, float, float]:
    """Test vowel pointer attachment.

    For each occurrence of vp_token following another token T:
    - Decode T alone -> baseline hit
    - Concatenate decoded_T + decoded_VP -> attachment hit
    - Concatenate decoded_T + random_short -> random hit

    Returns (n_tests, attachment_hit_rate, baseline_hit_rate, random_hit_rate).
    """
    n_tests = 0
    n_attachment_hits = 0
    n_baseline_hits = 0
    n_random_hits = 0

    for i in range(1, len(all_tokens)):
        if all_tokens[i] != vp_token:
            continue
        if token_folios[i] != token_folios[i - 1]:
            continue

        preceding_decoded = decoded[i - 1]
        n_tests += 1

        # Baseline: preceding token alone
        if preceding_decoded in ref_word_set:
            n_baseline_hits += 1

        # Attachment: concatenate
        concatenated = preceding_decoded + vp_decoded
        if concatenated in ref_word_set:
            n_attachment_hits += 1

        # Random control: preceding + random short decoded
        if short_decoded_pool:
            random_short = rng.choice(short_decoded_pool)
            random_cat = preceding_decoded + random_short
            if random_cat in ref_word_set:
                n_random_hits += 1

    if n_tests == 0:
        return 0, 0.0, 0.0, 0.0

    return (
        n_tests,
        n_attachment_hits / n_tests,
        n_baseline_hits / n_tests,
        n_random_hits / n_tests,
    )


# ---------------------------------------------------------------------------
# Vowel distribution correlation
# ---------------------------------------------------------------------------

def _is_consonant_heavy(decoded_word: str) -> bool:
    """Check if a decoded word has more consonants than vowels."""
    vowels = sum(1 for c in decoded_word.lower() if c in LATIN_VOWELS)
    consonants = sum(1 for c in decoded_word.lower()
                     if c.isalpha() and c not in LATIN_VOWELS)
    return consonants > vowels


def _test_vowel_distribution(
    vp_tokens: Set[str],
    all_tokens: List[str],
    decoded: List[str],
    token_folios: List[str],
) -> VowelDistributionTest:
    """Test: do VPs appear more after consonant-heavy roots?"""
    n_vp_after_ch = 0
    n_vp_after_other = 0
    n_all_after_ch = 0
    n_all_after_other = 0

    for i in range(1, len(all_tokens)):
        if token_folios[i] != token_folios[i - 1]:
            continue

        prev_decoded = decoded[i - 1]
        is_ch = _is_consonant_heavy(prev_decoded)

        if is_ch:
            n_all_after_ch += 1
        else:
            n_all_after_other += 1

        if all_tokens[i] in vp_tokens:
            if is_ch:
                n_vp_after_ch += 1
            else:
                n_vp_after_other += 1

    n_vp_total = n_vp_after_ch + n_vp_after_other
    n_all_total = n_all_after_ch + n_all_after_other

    ch_rate = n_vp_after_ch / n_vp_total if n_vp_total > 0 else 0.0
    baseline_ch_rate = n_all_after_ch / n_all_total if n_all_total > 0 else 0.0
    enrichment = ch_rate / baseline_ch_rate if baseline_ch_rate > 0 else 0.0

    return VowelDistributionTest(
        n_vp_following_consonant_heavy=n_vp_after_ch,
        n_vp_following_other=n_vp_after_other,
        consonant_heavy_rate=round(ch_rate, 4),
        baseline_consonant_heavy_rate=round(baseline_ch_rate, 4),
        enrichment=round(enrichment, 4),
        is_significant=enrichment > 1.2,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_vowel_pointer() -> None:
    """Step 34.16: Vowel pointer test."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 34.16: Vowel Pointer Test (Track F)")
    print("=" * 70)

    rd = _results_dir()
    eva_to_triple = build_eva_to_triple_lookup()

    # ── 1. Load inputs ──
    print("\n  1. Loading inputs ...")

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

    # Signal classifications
    token_classifications: List[str] = []
    sig_path = os.path.join(rd, 'signal_bigrams.json')
    if os.path.exists(sig_path):
        with open(sig_path) as f:
            sig_data = json.load(f)
        token_classifications = sig_data.get('token_classifications', [])

    print(f"     Assignment: {len(assignment)} triples")
    print(f"     Modifiers: {len(modifier_chars)} chars")

    # ── 2. Build reference word set ──
    print("\n  2. Building reference word set ...")
    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    base_words = set(
        w.lower() for w in ref_corpus.get_combined_tokens('latin')
        if len(w) >= 2
    )
    expanded, _ = build_expanded_word_set(base_words)
    ref_word_set = base_words | expanded
    print(f"     {len(ref_word_set)} reference words")

    # ── 3. Load and decode corpus ──
    print("\n  3. Loading and decoding corpus ...")
    corpus = load_corpus(verbose=False)

    all_tokens: List[str] = []
    token_folios: List[str] = []
    token_sections: List[str] = []

    for folio, page in corpus.pages.items():
        for token in page.all_tokens:
            all_tokens.append(token)
            token_folios.append(folio)
            token_sections.append(page.section)

    n_tokens = len(all_tokens)

    # Ensure we have classifications for all tokens
    if len(token_classifications) < n_tokens:
        # Pad with SHARED_MISS
        token_classifications.extend(
            ['SHARED_MISS'] * (n_tokens - len(token_classifications))
        )

    decoded = _decode_corpus_r3(
        all_tokens, assignment, eva_to_triple,
        modifier_chars, modifier_rules, ref_word_set,
    )
    dict_hits = [w in ref_word_set for w in decoded]
    baseline_dict_hit = sum(dict_hits) / n_tokens if n_tokens > 0 else 0.0

    print(f"     {n_tokens} tokens, baseline dict_hit = {baseline_dict_hit:.3f}")

    # ── 4. Identify VP candidates ──
    print("\n  4. Identifying vowel pointer candidates ...")
    candidate_tokens = _identify_vp_candidates(
        all_tokens, decoded, token_classifications,
        ref_word_set, eva_to_triple,
    )

    # Decode each candidate
    candidate_decoded: Dict[str, str] = {}
    for tok in candidate_tokens:
        candidate_decoded[tok] = _decode_single_token(
            tok, assignment, eva_to_triple,
            modifier_chars, modifier_rules, ref_word_set,
        )

    # Filter: must decode to something containing a vowel
    vp_candidates: List[str] = []
    for tok in candidate_tokens:
        dec = candidate_decoded[tok]
        has_vowel = any(c in LATIN_VOWELS for c in dec.lower())
        if has_vowel:
            vp_candidates.append(tok)

    print(f"     {len(candidate_tokens)} short frequent tokens")
    print(f"     {len(vp_candidates)} with vowel in decoded form")
    for tok in vp_candidates[:10]:
        freq = Counter(all_tokens)[tok]
        print(f"       '{tok}' (freq={freq}) -> '{candidate_decoded[tok]}'")

    # ── 5. Attachment test ──
    print("\n  5. Running attachment tests ...")
    rng = random.Random(42)

    # Build pool of short decoded forms for random control
    short_decoded_pool = [
        candidate_decoded[t] for t in vp_candidates
        if len(candidate_decoded[t]) <= 3
    ]
    if not short_decoded_pool:
        short_decoded_pool = ['a', 'e', 'i', 'o', 'u']

    token_freqs = Counter(all_tokens)
    vp_results: List[VowelPointerCandidate] = []

    for tok in vp_candidates:
        dec = candidate_decoded[tok]
        freq = token_freqs[tok]

        # Check if this token is SIGNAL
        is_signal = False
        for i, t in enumerate(all_tokens):
            if t == tok and i < len(token_classifications):
                if token_classifications[i] == 'SIGNAL':
                    is_signal = True
                    break

        n_tests, att_rate, base_rate, rand_rate = _test_attachment(
            tok, dec, all_tokens, decoded, token_folios,
            ref_word_set, rng, short_decoded_pool,
        )

        att_improvement = att_rate - base_rate
        rand_improvement = rand_rate - base_rate
        is_confirmed = att_improvement > 0.02 and att_improvement > rand_improvement

        vp_result = VowelPointerCandidate(
            eva_token=tok,
            frequency=freq,
            decoded=dec,
            has_vowel=True,
            is_signal=is_signal,
            n_attachment_tests=n_tests,
            attachment_hit_rate=round(att_rate, 4),
            baseline_hit_rate=round(base_rate, 4),
            random_hit_rate=round(rand_rate, 4),
            attachment_improvement=round(att_improvement, 4),
            random_improvement=round(rand_improvement, 4),
            is_confirmed_vp=is_confirmed,
        )
        vp_results.append(vp_result)

        print(f"     '{tok}' -> '{dec}'  freq={freq}  tests={n_tests}  "
              f"attach={att_rate:.3f} base={base_rate:.3f} "
              f"rand={rand_rate:.3f}  "
              f"{'CONFIRMED' if is_confirmed else 'not confirmed'}")

    confirmed_vps = [r.eva_token for r in vp_results if r.is_confirmed_vp]
    n_confirmed = len(confirmed_vps)
    print(f"\n     Confirmed vowel pointers: {n_confirmed}")

    # ── 6. Aggregate attachment statistics ──
    print("\n  6. Aggregate statistics ...")
    att_improvements = [r.attachment_improvement for r in vp_results
                        if r.n_attachment_tests > 0]
    rand_improvements = [r.random_improvement for r in vp_results
                         if r.n_attachment_tests > 0]

    mean_att_imp = (sum(att_improvements) / len(att_improvements)
                    if att_improvements else 0.0)
    mean_rand_imp = (sum(rand_improvements) / len(rand_improvements)
                     if rand_improvements else 0.0)

    # Compute z-score: attachment vs random improvement
    diff = [a - r for a, r in zip(att_improvements, rand_improvements)]
    mean_diff = sum(diff) / len(diff) if diff else 0.0
    var_diff = (sum((d - mean_diff) ** 2 for d in diff) / len(diff)
                if diff else 0.0)
    std_diff = var_diff ** 0.5
    att_vs_rand_z = mean_diff / (std_diff / len(diff) ** 0.5) \
        if std_diff > 0 and len(diff) > 0 else 0.0

    print(f"     Mean attachment improvement: {mean_att_imp:.4f}")
    print(f"     Mean random improvement: {mean_rand_imp:.4f}")
    print(f"     Attachment vs random z: {att_vs_rand_z:.2f}")

    # ── 7. Vowel distribution correlation ──
    print("\n  7. Vowel distribution correlation ...")
    vp_token_set = set(confirmed_vps) if confirmed_vps else set(
        r.eva_token for r in vp_results
    )
    vowel_dist = _test_vowel_distribution(
        vp_token_set, all_tokens, decoded, token_folios,
    )
    print(f"     VP after consonant-heavy: {vowel_dist.consonant_heavy_rate:.3f}")
    print(f"     Baseline rate: {vowel_dist.baseline_consonant_heavy_rate:.3f}")
    print(f"     Enrichment: {vowel_dist.enrichment:.2f}x "
          f"({'SIGNIFICANT' if vowel_dist.is_significant else 'not significant'})")

    # ── 8. Section uniformity test ──
    print("\n  8. Section uniformity test ...")
    section_sets: Dict[str, List[int]] = defaultdict(list)
    for i, sec in enumerate(token_sections):
        section_sets[sec].append(i)

    section_tests: List[SectionUniformityTest] = []
    n_improved = 0

    for sec in sorted(section_sets.keys()):
        indices = section_sets[sec]
        n_pairs = 0
        n_attach_hits = 0
        n_base_hits = 0

        for i in indices:
            if i == 0 or token_folios[i] != token_folios[i - 1]:
                continue
            if all_tokens[i] not in vp_token_set:
                continue

            prev_dec = decoded[i - 1]
            vp_dec = decoded[i]
            n_pairs += 1

            if prev_dec in ref_word_set:
                n_base_hits += 1
            if (prev_dec + vp_dec) in ref_word_set:
                n_attach_hits += 1

        if n_pairs == 0:
            continue

        att_rate = n_attach_hits / n_pairs
        base_rate = n_base_hits / n_pairs
        improvement = att_rate - base_rate

        section_tests.append(SectionUniformityTest(
            section=sec,
            n_pairs=n_pairs,
            attachment_hit_rate=round(att_rate, 4),
            baseline_hit_rate=round(base_rate, 4),
            improvement=round(improvement, 4),
        ))

        if improvement > 0.005:
            n_improved += 1

        print(f"     {sec:15s}  pairs={n_pairs:4d}  "
              f"attach={att_rate:.3f}  base={base_rate:.3f}  "
              f"delta={improvement:+.3f}")

    n_sections_total = len(section_tests)

    # ── 9. Estimate VP-attached dict_hit ──
    print("\n  9. Estimating VP-attached corpus dict_hit ...")
    vp_attached_hits = 0
    vp_attached_total = 0

    i = 0
    while i < n_tokens:
        if (i < n_tokens - 1
                and all_tokens[i + 1] in vp_token_set
                and token_folios[i] == token_folios[i + 1]):
            # Try concatenation
            concat = decoded[i] + decoded[i + 1]
            if concat in ref_word_set:
                vp_attached_hits += 1
            elif decoded[i] in ref_word_set:
                vp_attached_hits += 1
            vp_attached_total += 1
            i += 2  # consume both tokens
        else:
            if decoded[i] in ref_word_set:
                vp_attached_hits += 1
            vp_attached_total += 1
            i += 1

    vp_dict_hit = (vp_attached_hits / vp_attached_total
                   if vp_attached_total > 0 else 0.0)
    delta = vp_dict_hit - baseline_dict_hit

    print(f"     VP-attached dict_hit: {vp_dict_hit:.3f}")
    print(f"     Baseline dict_hit: {baseline_dict_hit:.3f}")
    print(f"     Delta: {delta:+.3f}")

    # ── 10. Verdict ──
    vp_significant = n_confirmed > 0 and att_vs_rand_z > 2.0
    vp_uniform = n_improved >= n_sections_total * 0.5 if n_sections_total > 0 else False
    vp_improves = delta > 0.005

    verdict = (
        f"{n_confirmed}/{len(vp_results)} confirmed vowel pointers. "
        f"Attachment z = {att_vs_rand_z:.2f} "
        f"({'SIGNIFICANT' if vp_significant else 'NOT significant'}). "
        f"Distribution enrichment = {vowel_dist.enrichment:.2f}x. "
        f"Section uniformity: {n_improved}/{n_sections_total} improved. "
        f"VP dict_hit = {vp_dict_hit:.3f} vs baseline = {baseline_dict_hit:.3f} "
        f"(delta = {delta:+.3f}, {'IMPROVED' if vp_improves else 'NO improvement'})."
    )
    print(f"\n  VERDICT: {verdict}")

    # ── 11. Save ──
    elapsed = round(time.time() - t0, 2)

    result = VowelPointerResult(
        n_candidates=len(vp_results),
        n_confirmed=n_confirmed,
        candidates=[_convert(asdict(r)) for r in vp_results],
        confirmed_vps=confirmed_vps,
        mean_attachment_improvement=round(mean_att_imp, 4),
        mean_random_improvement=round(mean_rand_imp, 4),
        attachment_vs_random_z=round(att_vs_rand_z, 2),
        vowel_distribution=_convert(asdict(vowel_dist)),
        section_uniformity=[_convert(asdict(s)) for s in section_tests],
        n_sections_improved=n_improved,
        n_sections_total=n_sections_total,
        baseline_dict_hit=round(baseline_dict_hit, 4),
        vp_attached_dict_hit=round(vp_dict_hit, 4),
        delta_dict_hit=round(delta, 4),
        verdict=verdict,
        runtime_seconds=elapsed,
    )

    out_path = os.path.join(rd, 'vowel_pointer_test.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)
    print(f"\n  Saved -> {out_path}")
    print(f"  Completed in {elapsed:.1f}s")
