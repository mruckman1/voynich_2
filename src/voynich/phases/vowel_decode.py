"""
Step 34.17 – Vowel-Pointer Re-Decode (Track F)
================================================
Re-decodes the corpus with vowel-pointer attachment rules derived from
Step 34.16.  Confirmed vowel pointers are merged with their preceding
root tokens before decoding.  Also tests combination with Track A
(abjad consonant skeleton + vowel pointers -> full word candidates).

Dependency chain:
    vowel_pointer_test.json     (Step 34.16)
    combined_refine.json        (Phase 15 assignment)
    modifier_integrate.json     (Phase 16 modifiers)
    signal_bigrams.json         (Phase 29 classifications)
    abjad_csp.json              (Step 34.2, optional — Track A combo)
        → vowel_decode.json     (this step)
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
from voynich.phases.null_corpus import (
    _build_eva_bigram_model,
    _generate_null_corpus,
    _reconstruct_modifier_rules,
)
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


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MergedTokenInfo:
    """Information about a merged root+VP token."""
    original_root: str
    original_vp: str
    merged_eva: str
    decoded_root: str
    decoded_vp: str
    decoded_merged: str
    dict_hit: bool


@dataclass
class AbjadCombinationResult:
    """Result of combining abjad consonants with vowel pointers."""
    n_tokens_tested: int
    n_combo_hits: int
    combo_hit_rate: float
    n_abjad_only_hits: int
    abjad_only_rate: float
    n_cv_only_hits: int
    cv_only_rate: float
    improvement_over_abjad: float
    sample_combos: List[Dict]


@dataclass
class FolioVPDecode:
    """Per-folio vowel-pointed decode results."""
    folio: str
    n_tokens: int
    n_merged: int
    merge_rate: float
    dict_hit_rate: float
    baseline_dict_hit_rate: float
    delta: float


@dataclass
class VowelDecodeResult:
    # Merge statistics
    n_tokens_original: int
    n_tokens_after_merge: int
    n_merges: int
    merge_rate: float
    confirmed_vps_used: List[str]

    # Decode results
    vp_dict_hit: float
    baseline_dict_hit: float
    delta_dict_hit: float

    # Signal isolation on VP decode
    n_signal_vp: int
    signal_rate_vp: float
    n_signal_baseline: int
    signal_rate_baseline: float
    signal_delta: float

    # Bigram z
    n_signal_pairs: int
    n_bigram_hits: int
    bigram_hit_rate: float
    bigram_z: float

    # Phase 29 comparison
    phase29_signal_rate: float
    phase29_bigram_z: float
    signal_rate_delta_vs_p29: float
    bigram_z_delta_vs_p29: float

    # Per-folio results (top 20)
    top_folios: List[Dict]

    # Track A combination (abjad + VP)
    abjad_combination: Optional[Dict]

    # Null comparison
    null_dict_hit_mean: float
    null_dict_hit_std: float
    selectivity: float

    # Per-section results
    per_section: List[Dict]

    # Sample merged tokens
    sample_merges: List[Dict]

    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Re-tokenization
# ---------------------------------------------------------------------------

def _merge_vp_tokens(
    tokens: List[str],
    folios: List[str],
    vp_set: Set[str],
) -> Tuple[List[str], List[str], List[Tuple[int, int]], int]:
    """Merge VP tokens with their preceding non-VP tokens.

    Returns (merged_tokens, merged_folios, merge_spans, n_merges).
    merge_spans[i] = (start_idx, end_idx) into original token list.
    """
    merged_tokens: List[str] = []
    merged_folios: List[str] = []
    merge_spans: List[Tuple[int, int]] = []
    n_merges = 0

    i = 0
    n = len(tokens)

    while i < n:
        # Check if next token is a VP on the same folio
        if (i + 1 < n
                and tokens[i + 1] in vp_set
                and folios[i + 1] == folios[i]
                and tokens[i] not in vp_set):
            # Merge: concatenate EVA strings
            combined = tokens[i] + tokens[i + 1]
            merged_tokens.append(combined)
            merged_folios.append(folios[i])
            merge_spans.append((i, i + 1))
            n_merges += 1
            i += 2
        else:
            merged_tokens.append(tokens[i])
            merged_folios.append(folios[i])
            merge_spans.append((i, i))
            i += 1

    return merged_tokens, merged_folios, merge_spans, n_merges


def _decode_merged_corpus(
    merged_tokens: List[str],
    all_tokens: List[str],
    merge_spans: List[Tuple[int, int]],
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    modifier_chars: set,
    modifier_rules: Dict[str, str],
    ref_word_set: set,
    decoded_original: List[str],
) -> List[str]:
    """Decode merged tokens, falling back to best original decode.

    Strategy: decode merged token; if it hits the dict, use it.
    Otherwise fall back to original root decode or concatenation.
    """
    # Batch decode all merged tokens
    merged_decoded = _decode_corpus_r3(
        merged_tokens, assignment, eva_to_triple,
        modifier_chars, modifier_rules, ref_word_set,
    )

    # For each merged token, decide: merged hit vs original
    final_decoded: List[str] = []

    for idx, (tok, dec) in enumerate(zip(merged_tokens, merged_decoded)):
        start, end = merge_spans[idx]

        if dec in ref_word_set:
            final_decoded.append(dec)
        elif start == end:
            # No merge happened — use original
            final_decoded.append(decoded_original[start])
        else:
            # Merge happened but merged decode is not a hit
            # Try concatenation of original decodes
            concat = ''.join(decoded_original[j] for j in range(start, end + 1))
            if concat in ref_word_set:
                final_decoded.append(concat)
            else:
                # Fall back to root decode
                final_decoded.append(decoded_original[start])

    return final_decoded


# ---------------------------------------------------------------------------
# Abjad combination test
# ---------------------------------------------------------------------------

def _test_abjad_combination(
    all_tokens: List[str],
    token_folios: List[str],
    confirmed_vps: Set[str],
    decoded_original: List[str],
    abjad_table: Dict[str, str],
    eva_to_triple: Dict[str, str],
    ref_word_set: set,
) -> AbjadCombinationResult:
    """Combine abjad consonant skeletons with VP vowels.

    For token T followed by VP:
    - Abjad-decode T -> consonant skeleton
    - Decode VP -> vowel(s)
    - Insert vowel(s) into skeleton -> candidate words
    - Check against dict
    """

    def _abjad_decode(token: str) -> str:
        chars = tokenize_eva_chars(token)
        consonants = []
        for ch in chars:
            triple = eva_to_triple.get(ch)
            if triple and triple in abjad_table:
                consonants.append(abjad_table[triple])
        return ''.join(consonants)

    def _insert_vowels(skeleton: str, vowels: str) -> List[str]:
        """Generate candidate words by inserting vowels into skeleton."""
        candidates: List[str] = []
        if not skeleton or not vowels:
            return candidates

        # Try simple insertion after each consonant
        for v in vowels:
            for i in range(len(skeleton) + 1):
                word = skeleton[:i] + v + skeleton[i:]
                candidates.append(word)

        # Try inserting all vowels at each position
        for i in range(len(skeleton) + 1):
            word = skeleton[:i] + vowels + skeleton[i:]
            candidates.append(word)

        # Try interleaving: c1 v1 c2 v2 ...
        if len(skeleton) == len(vowels):
            interleaved = ''.join(
                c + v for c, v in zip(skeleton, vowels)
            )
            candidates.append(interleaved)
        elif len(skeleton) == len(vowels) + 1:
            interleaved = ''.join(
                c + v for c, v in zip(skeleton, vowels)
            ) + skeleton[-1]
            candidates.append(interleaved)

        return candidates

    n_tested = 0
    n_combo_hits = 0
    n_abjad_only = 0
    n_cv_only = 0
    sample_combos: List[Dict] = []

    for i in range(len(all_tokens) - 1):
        if all_tokens[i + 1] not in confirmed_vps:
            continue
        if token_folios[i] != token_folios[i + 1]:
            continue

        n_tested += 1

        root = all_tokens[i]
        vp = all_tokens[i + 1]
        skeleton = _abjad_decode(root)
        vp_decoded = decoded_original[i + 1]
        vowels = ''.join(c for c in vp_decoded.lower() if c in LATIN_VOWELS)

        # Abjad only
        abjad_hit = skeleton in ref_word_set and len(skeleton) >= 2
        if abjad_hit:
            n_abjad_only += 1

        # CV only (Phase 16)
        cv_hit = decoded_original[i] in ref_word_set
        if cv_hit:
            n_cv_only += 1

        # Combination: insert vowels into skeleton
        candidates = _insert_vowels(skeleton, vowels)
        combo_hit = any(c in ref_word_set for c in candidates)
        if combo_hit:
            n_combo_hits += 1

        if len(sample_combos) < 30:
            best_combo = next(
                (c for c in candidates if c in ref_word_set), ''
            )
            sample_combos.append({
                'root': root,
                'vp': vp,
                'skeleton': skeleton,
                'vowels': vowels,
                'combo_word': best_combo,
                'combo_hit': combo_hit,
                'cv_decoded': decoded_original[i],
                'cv_hit': cv_hit,
            })

    if n_tested == 0:
        return AbjadCombinationResult(
            n_tokens_tested=0, n_combo_hits=0, combo_hit_rate=0.0,
            n_abjad_only_hits=0, abjad_only_rate=0.0,
            n_cv_only_hits=0, cv_only_rate=0.0,
            improvement_over_abjad=0.0, sample_combos=[],
        )

    combo_rate = n_combo_hits / n_tested
    abjad_rate = n_abjad_only / n_tested
    cv_rate = n_cv_only / n_tested

    return AbjadCombinationResult(
        n_tokens_tested=n_tested,
        n_combo_hits=n_combo_hits,
        combo_hit_rate=round(combo_rate, 4),
        n_abjad_only_hits=n_abjad_only,
        abjad_only_rate=round(abjad_rate, 4),
        n_cv_only_hits=n_cv_only,
        cv_only_rate=round(cv_rate, 4),
        improvement_over_abjad=round(combo_rate - abjad_rate, 4),
        sample_combos=sample_combos,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_vowel_decode() -> None:
    """Step 34.17: Vowel-pointer re-decode."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 34.17: Vowel-Pointer Re-Decode (Track F)")
    print("=" * 70)

    rd = _results_dir()
    eva_to_triple = build_eva_to_triple_lookup()

    # ── 1. Load inputs ──
    print("\n  1. Loading inputs ...")

    # Vowel pointer test results
    vp_path = os.path.join(rd, 'vowel_pointer_test.json')
    if not os.path.exists(vp_path):
        print("  [SKIP] vowel_pointer_test.json not found -- run vowel-ptr first")
        return
    with open(vp_path) as f:
        vp_data = json.load(f)
    confirmed_vps_list = vp_data.get('confirmed_vps', [])
    if not confirmed_vps_list:
        # Fall back to candidates with positive attachment improvement
        for cand in vp_data.get('candidates', []):
            if cand.get('attachment_improvement', 0) > 0:
                confirmed_vps_list.append(cand['eva_token'])
    if not confirmed_vps_list:
        # Last resort: top 5 candidates
        for cand in vp_data.get('candidates', [])[:5]:
            confirmed_vps_list.append(cand['eva_token'])

    confirmed_vps = set(confirmed_vps_list)

    # Phase 15 assignment
    refine_path = os.path.join(rd, 'combined_refine.json')
    if not os.path.exists(refine_path):
        print("  [SKIP] combined_refine.json not found")
        return
    with open(refine_path) as f:
        refine_data = json.load(f)
    assignment = refine_data.get('best_assignment', {})

    # Phase 16 modifiers
    mod_path = os.path.join(rd, 'modifier_integrate.json')
    if not os.path.exists(mod_path):
        print("  [SKIP] modifier_integrate.json not found")
        return
    with open(mod_path) as f:
        mod_data = json.load(f)
    modifier_chars, modifier_rules = _reconstruct_modifier_rules(mod_data)

    # Signal classifications
    signal_classifications: List[str] = []
    sig_path = os.path.join(rd, 'signal_bigrams.json')
    if os.path.exists(sig_path):
        with open(sig_path) as f:
            sig_data = json.load(f)
        signal_classifications = sig_data.get('token_classifications', [])

    # Abjad table (optional -- for Track A combo)
    abjad_table: Optional[Dict[str, str]] = None
    abjad_path = os.path.join(rd, 'abjad_csp.json')
    if os.path.exists(abjad_path):
        with open(abjad_path) as f:
            abjad_data = json.load(f)
        abjad_table = abjad_data.get('best_assignment', {})

    print(f"     Confirmed VPs: {confirmed_vps_list}")
    print(f"     Assignment: {len(assignment)} triples")
    print(f"     Abjad table: {'loaded' if abjad_table else 'not available'}")

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

    # Build reference bigrams
    ref_tokens_lat = ref_corpus.get_combined_tokens('latin')
    ref_bigrams: Set[Tuple[str, str]] = set()
    for i in range(len(ref_tokens_lat) - 1):
        w1 = ref_tokens_lat[i].lower()
        w2 = ref_tokens_lat[i + 1].lower()
        if w1 in ref_word_set and w2 in ref_word_set:
            ref_bigrams.add((w1, w2))
    print(f"     {len(ref_bigrams)} reference bigrams")

    # ── 3. Load corpus and decode baseline ──
    print("\n  3. Loading corpus and computing baseline ...")
    corpus = load_corpus(verbose=False)

    all_tokens: List[str] = []
    token_folios: List[str] = []
    token_sections: List[str] = []

    for folio, page in corpus.pages.items():
        for token in page.all_tokens:
            all_tokens.append(token)
            token_folios.append(folio)
            token_sections.append(page.section)

    n_original = len(all_tokens)

    decoded_original = _decode_corpus_r3(
        all_tokens, assignment, eva_to_triple,
        modifier_chars, modifier_rules, ref_word_set,
    )
    baseline_hits = [w in ref_word_set for w in decoded_original]
    baseline_dict_hit = sum(baseline_hits) / n_original if n_original > 0 else 0.0

    print(f"     {n_original} tokens, baseline dict_hit = {baseline_dict_hit:.3f}")

    # ── 4. Merge VP tokens ──
    print("\n  4. Merging confirmed VP tokens ...")
    merged_tokens, merged_folios, merge_spans, n_merges = _merge_vp_tokens(
        all_tokens, token_folios, confirmed_vps,
    )

    n_after_merge = len(merged_tokens)
    merge_rate = n_merges / n_original if n_original > 0 else 0.0

    print(f"     Tokens after merge: {n_after_merge} "
          f"({n_merges} merges, rate={merge_rate:.3f})")

    # ── 5. Decode merged corpus ──
    print("\n  5. Decoding merged corpus ...")
    merged_decoded = _decode_merged_corpus(
        merged_tokens, all_tokens, merge_spans,
        assignment, eva_to_triple,
        modifier_chars, modifier_rules,
        ref_word_set, decoded_original,
    )

    vp_hits = [w in ref_word_set for w in merged_decoded]
    vp_dict_hit = sum(vp_hits) / n_after_merge if n_after_merge > 0 else 0.0
    delta_dict_hit = vp_dict_hit - baseline_dict_hit

    print(f"     VP dict_hit: {vp_dict_hit:.3f}")
    print(f"     Delta vs baseline: {delta_dict_hit:+.3f}")

    # Collect sample merges
    sample_merges: List[Dict] = []
    for idx, (tok, dec) in enumerate(zip(merged_tokens, merged_decoded)):
        start, end = merge_spans[idx]
        if start != end and len(sample_merges) < 50:
            sample_merges.append({
                'root': all_tokens[start],
                'vp': all_tokens[end],
                'merged_eva': tok,
                'decoded_root': decoded_original[start],
                'decoded_vp': decoded_original[end],
                'decoded_merged': dec,
                'dict_hit': dec in ref_word_set,
            })

    # ── 6. Signal isolation on VP decode ──
    print("\n  6. Signal isolation on VP decode ...")
    null_seeds = [100, 101, 102, 103, 104]
    null_path = os.path.join(rd, 'null_corpus.json')
    if os.path.exists(null_path):
        with open(null_path) as f:
            null_data = json.load(f)
        null_seeds = [r['seed'] for r in null_data.get('null_runs', [])]

    bigram_probs, initial_probs, token_lengths = _build_eva_bigram_model(
        all_tokens,
    )

    # Generate and decode null corpora
    null_dict_hits: List[float] = []
    null_hits_list: List[List[bool]] = []

    for seed in null_seeds:
        null_tokens = _generate_null_corpus(
            bigram_probs, initial_probs, token_lengths, n_original, seed,
        )
        # Apply same VP merge to null
        null_merged, null_m_folios, null_m_spans, _ = _merge_vp_tokens(
            null_tokens, token_folios, confirmed_vps,
        )
        null_decoded = _decode_corpus_r3(
            null_merged, assignment, eva_to_triple,
            modifier_chars, modifier_rules, ref_word_set,
        )
        null_h = [w in ref_word_set for w in null_decoded]
        null_rate = sum(null_h) / len(null_decoded) if null_decoded else 0.0
        null_dict_hits.append(null_rate)
        null_hits_list.append(null_h)

    null_mean = sum(null_dict_hits) / len(null_dict_hits) if null_dict_hits else 0.0
    null_var = (sum((r - null_mean) ** 2 for r in null_dict_hits) /
                len(null_dict_hits) if null_dict_hits else 0.0)
    null_std = null_var ** 0.5
    selectivity = vp_dict_hit / null_mean if null_mean > 0 else 0.0

    # Signal classification: real hit + majority null miss = SIGNAL
    classifications: List[str] = []
    for idx in range(n_after_merge):
        r_hit = vp_hits[idx]
        null_hit_count = sum(
            1 for nh in null_hits_list
            if idx < len(nh) and nh[idx]
        )
        if r_hit and null_hit_count <= 1:
            classifications.append('SIGNAL')
        elif r_hit and null_hit_count >= 3:
            classifications.append('SHARED_HIT')
        elif not r_hit and null_hit_count >= 3:
            classifications.append('ANTI_SIGNAL')
        else:
            classifications.append('SHARED_MISS')

    n_vp_signal = classifications.count('SIGNAL')
    signal_rate_vp = n_vp_signal / n_after_merge if n_after_merge > 0 else 0.0

    # Baseline signal stats
    n_baseline_signal = 0
    if signal_classifications:
        n_baseline_signal = sum(
            1 for c in signal_classifications if c == 'SIGNAL'
        )
    signal_rate_baseline = (n_baseline_signal / len(signal_classifications)
                            if signal_classifications else 0.0)
    signal_delta = signal_rate_vp - signal_rate_baseline

    print(f"     VP signal rate: {signal_rate_vp:.3f}")
    print(f"     Baseline signal rate: {signal_rate_baseline:.3f}")
    print(f"     Signal delta: {signal_delta:+.3f}")
    print(f"     Null mean: {null_mean:.3f}, selectivity: {selectivity:.2f}x")

    # ── 7. Bigram z-score ──
    print("\n  7. Computing bigram z-score ...")
    signal_pairs: List[Tuple[str, str]] = []
    for i in range(n_after_merge - 1):
        if (classifications[i] == 'SIGNAL'
                and classifications[i + 1] == 'SIGNAL'
                and merged_folios[i] == merged_folios[i + 1]):
            signal_pairs.append((merged_decoded[i], merged_decoded[i + 1]))

    n_bigram_hits = sum(1 for w1, w2 in signal_pairs
                        if (w1, w2) in ref_bigrams)
    bigram_hit_rate = (n_bigram_hits / len(signal_pairs)
                       if signal_pairs else 0.0)

    # Null permutation test
    rng = random.Random(42)
    indices = list(range(n_after_merge))
    null_rates: List[float] = []
    for _ in range(500):
        fake_signal = set(rng.sample(indices,
                                     min(n_vp_signal, n_after_merge)))
        n_pairs = 0
        n_hits = 0
        for i in range(n_after_merge - 1):
            if (i in fake_signal and (i + 1) in fake_signal
                    and merged_folios[i] == merged_folios[i + 1]):
                n_pairs += 1
                if (merged_decoded[i], merged_decoded[i + 1]) in ref_bigrams:
                    n_hits += 1
        rate = n_hits / n_pairs if n_pairs > 0 else 0.0
        null_rates.append(rate)

    null_bg_mean = (sum(null_rates) / len(null_rates)
                    if null_rates else 0.0)
    null_bg_var = (sum((r - null_bg_mean) ** 2 for r in null_rates)
                   / len(null_rates) if null_rates else 0.0)
    null_bg_std = null_bg_var ** 0.5
    bigram_z = ((bigram_hit_rate - null_bg_mean) / null_bg_std
                if null_bg_std > 0 else 0.0)

    print(f"     Signal pairs: {len(signal_pairs)}")
    print(f"     Bigram hits: {n_bigram_hits} ({bigram_hit_rate:.4f})")
    print(f"     Bigram z: {bigram_z:.2f}")

    # ── 8. Phase 29 comparison ──
    print("\n  8. Phase 29 comparison ...")
    phase29_signal_rate = 0.165
    phase29_bigram_z = 6.14
    bg_path = os.path.join(rd, 'signal_bigrams.json')
    if os.path.exists(bg_path):
        with open(bg_path) as f:
            bg = json.load(f)
        phase29_signal_rate = bg.get('signal_rate', 0.165)
        phase29_bigram_z = bg.get('bigram_z_score', 6.14)

    signal_rate_delta_p29 = signal_rate_vp - phase29_signal_rate
    bigram_z_delta_p29 = bigram_z - phase29_bigram_z

    print(f"     Phase 29: SIGNAL={phase29_signal_rate:.3f}, "
          f"z={phase29_bigram_z:.2f}")
    print(f"     VP decode: SIGNAL={signal_rate_vp:.3f}, z={bigram_z:.2f}")

    # ── 9. Per-folio results ──
    print("\n  9. Per-folio VP decode results (top 20) ...")
    folio_tokens: Dict[str, int] = Counter()
    folio_merges: Dict[str, int] = Counter()
    folio_vp_hits: Dict[str, int] = Counter()
    folio_base_hits: Dict[str, int] = Counter()

    for idx, (folio, hit) in enumerate(zip(merged_folios, vp_hits)):
        folio_tokens[folio] += 1
        if hit:
            folio_vp_hits[folio] += 1
        start, end = merge_spans[idx]
        if start != end:
            folio_merges[folio] += 1

    for folio, hit in zip(token_folios, baseline_hits):
        folio_base_hits[folio] += int(hit)

    folio_base_n: Counter = Counter(token_folios)

    folio_results: List[FolioVPDecode] = []
    for folio in sorted(folio_tokens.keys()):
        n_tok = folio_tokens[folio]
        n_mrg = folio_merges.get(folio, 0)
        n_vh = folio_vp_hits.get(folio, 0)
        n_bh = folio_base_hits.get(folio, 0)
        n_base_tok = folio_base_n.get(folio, 1)

        vp_rate = n_vh / n_tok if n_tok > 0 else 0.0
        base_rate = n_bh / n_base_tok if n_base_tok > 0 else 0.0

        folio_results.append(FolioVPDecode(
            folio=folio,
            n_tokens=n_tok,
            n_merged=n_mrg,
            merge_rate=round(n_mrg / n_tok, 4) if n_tok > 0 else 0.0,
            dict_hit_rate=round(vp_rate, 4),
            baseline_dict_hit_rate=round(base_rate, 4),
            delta=round(vp_rate - base_rate, 4),
        ))

    folio_results.sort(key=lambda f: -f.delta)
    for fr in folio_results[:20]:
        print(f"     {fr.folio:8s}  tok={fr.n_tokens:3d}  "
              f"merged={fr.n_merged:2d}  "
              f"vp_hit={fr.dict_hit_rate:.3f}  "
              f"base={fr.baseline_dict_hit_rate:.3f}  "
              f"delta={fr.delta:+.3f}")

    # ── 10. Track A (abjad + VP) combination test ──
    print("\n  10. Track A (abjad + VP) combination test ...")
    abjad_combo: Optional[AbjadCombinationResult] = None
    if abjad_table and confirmed_vps:
        abjad_combo = _test_abjad_combination(
            all_tokens, token_folios, confirmed_vps,
            decoded_original, abjad_table, eva_to_triple, ref_word_set,
        )
        print(f"     Tested: {abjad_combo.n_tokens_tested} root+VP pairs")
        print(f"     Combo hit rate: {abjad_combo.combo_hit_rate:.3f}")
        print(f"     Abjad-only rate: {abjad_combo.abjad_only_rate:.3f}")
        print(f"     CV-only rate: {abjad_combo.cv_only_rate:.3f}")
        print(f"     Improvement over abjad: "
              f"{abjad_combo.improvement_over_abjad:+.3f}")
    else:
        print("     [SKIP] Abjad table not available or no confirmed VPs")

    # ── 11. Per-section results ──
    print("\n  11. Per-section results ...")
    # Build merged sections
    merged_sections: List[str] = []
    for idx in range(len(merge_spans)):
        start, _ = merge_spans[idx]
        if start < len(token_sections):
            merged_sections.append(token_sections[start])
        else:
            merged_sections.append('unknown')

    section_counts: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {'n': 0, 'hits': 0, 'signal': 0}
    )
    for i in range(n_after_merge):
        sec = merged_sections[i] if i < len(merged_sections) else 'unknown'
        section_counts[sec]['n'] += 1
        if vp_hits[i]:
            section_counts[sec]['hits'] += 1
        if classifications[i] == 'SIGNAL':
            section_counts[sec]['signal'] += 1

    per_section: List[Dict] = []
    for sec_name, counts in sorted(section_counts.items()):
        n = counts['n']
        if n < 10:
            continue
        hit_rate = counts['hits'] / n
        sig_rate = counts['signal'] / n
        per_section.append({
            'section': sec_name,
            'n_tokens': n,
            'dict_hit_rate': round(hit_rate, 4),
            'signal_rate': round(sig_rate, 4),
        })
        print(f"     {sec_name:15s}  n={n:5d}  dict_hit={hit_rate:.3f}  "
              f"signal={sig_rate:.3f}")

    # ── 12. Verdict ──
    improved = delta_dict_hit > 0.005
    signal_improved = signal_delta > 0.005
    abjad_synergy = (abjad_combo is not None
                     and abjad_combo.improvement_over_abjad > 0.02)
    bigram_better = bigram_z > phase29_bigram_z

    if bigram_better and signal_rate_vp > phase29_signal_rate:
        verdict = (
            f"VP_DECODE_BETTER: bigram z={bigram_z:.2f} > {phase29_bigram_z:.2f}, "
            f"SIGNAL={signal_rate_vp:.3f} > {phase29_signal_rate:.3f}. "
            f"Vowel-pointed decode beats CV baseline."
        )
    elif improved:
        verdict = (
            f"VP_DECODE_MARGINAL: dict_hit improved by {delta_dict_hit:+.4f} "
            f"but bigram z={bigram_z:.2f} (Phase 29: {phase29_bigram_z:.2f}). "
            f"VP merging helps dict hits but not signal structure."
        )
    else:
        verdict = (
            f"VP_DECODE_NO_IMPROVEMENT: dict_hit delta={delta_dict_hit:+.4f}, "
            f"bigram z={bigram_z:.2f}. VP merging does not improve decoding."
        )

    if abjad_synergy:
        verdict += f" ABJAD_SYNERGY: combo rate {abjad_combo.combo_hit_rate:.3f}."

    print(f"\n  VERDICT: {verdict}")

    # ── 13. Save ──
    elapsed = round(time.time() - t0, 2)

    result = VowelDecodeResult(
        n_tokens_original=n_original,
        n_tokens_after_merge=n_after_merge,
        n_merges=n_merges,
        merge_rate=round(merge_rate, 4),
        confirmed_vps_used=confirmed_vps_list,
        vp_dict_hit=round(vp_dict_hit, 4),
        baseline_dict_hit=round(baseline_dict_hit, 4),
        delta_dict_hit=round(delta_dict_hit, 4),
        n_signal_vp=n_vp_signal,
        signal_rate_vp=round(signal_rate_vp, 4),
        n_signal_baseline=n_baseline_signal,
        signal_rate_baseline=round(signal_rate_baseline, 4),
        signal_delta=round(signal_delta, 4),
        n_signal_pairs=len(signal_pairs),
        n_bigram_hits=n_bigram_hits,
        bigram_hit_rate=round(bigram_hit_rate, 4),
        bigram_z=round(bigram_z, 2),
        phase29_signal_rate=round(phase29_signal_rate, 4),
        phase29_bigram_z=round(phase29_bigram_z, 2),
        signal_rate_delta_vs_p29=round(signal_rate_delta_p29, 4),
        bigram_z_delta_vs_p29=round(bigram_z_delta_p29, 2),
        top_folios=[_convert(asdict(f)) for f in folio_results[:20]],
        abjad_combination=(
            _convert(asdict(abjad_combo)) if abjad_combo else None
        ),
        null_dict_hit_mean=round(null_mean, 4),
        null_dict_hit_std=round(null_std, 4),
        selectivity=round(selectivity, 2),
        per_section=per_section,
        sample_merges=sample_merges[:30],
        verdict=verdict,
        runtime_seconds=elapsed,
    )

    out_path = os.path.join(rd, 'vowel_decode.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)
    print(f"\n  Saved -> {out_path}")
    print(f"  Completed in {elapsed:.1f}s")
