"""
Step 34.2 – Abjad Consonant-Only CSP (Track A)
================================================
Re-runs the phonetic CSP from Phase 14 but mapping each stroke triple
to a single Latin consonant (or consonant cluster) instead of a CV
syllable.  Smaller domain (26 vs 75) = more constrained search.

Dependency chain:
    sigla_dictionary.json      (34.1: combined consonant dict)
    combined_refine.json       (Phase 15: for comparison)
    modifier_integrate.json    (Phase 16: modifiers)
    compound_sign_test.json    (Phase 31: morpheme decomposition)
    tachygraphic_stroke.json   (Phase 19.5: sign families)
    null_corpus.json           (Phase 17: seeds)
        → abjad_csp.json      (this step)
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
    tokenize_eva_chars,
)
from voynich.core.reference import (
    EVA_VISUAL_COMPONENTS,
    build_expanded_word_set,
    load_reference_corpus,
)
from voynich.phases.morpheme_grid import (
    KNOWN_PREFIXES,
    KNOWN_SUFFIXES,
    decompose_token_morphemes,
)
from voynich.phases.null_corpus import (
    _build_eva_bigram_model,
    _generate_null_corpus,
    _reconstruct_modifier_rules,
)
from voynich.phases.signal_isolation import _decode_corpus_r3
from voynich.phases.sigla_dictionary import _strip_vowels


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

CONSONANT_DOMAIN = [
    'b', 'c', 'd', 'f', 'g', 'h', 'l', 'm', 'n', 'p', 'q', 'r',
    's', 't', 'v', 'x', 'z',
    'st', 'pr', 'tr', 'sc', 'sp', 'qu', 'ch', 'ph', 'th',
]

# Approximate frequency rank for medical Latin consonants
LATIN_CONSONANT_FREQ_RANK = [
    'r', 's', 't', 'n', 'c', 'l', 'm', 'd', 'p', 'b', 'f', 'g',
    'v', 'x', 'q', 'h', 'z',
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class AbjadCSPResult:
    # Variable info
    n_variables: int
    n_domain: int
    triple_frequencies: Dict[str, int]

    # CSP solution
    best_assignment: Dict[str, str]   # triple_key → consonant
    best_cross_entropy: float
    n_beam_solutions: int

    # Skeleton matching
    skeleton_hit_rate: float
    n_skeleton_hits: int
    n_root_tokens: int
    sample_decoded_roots: List[Dict]  # first 30

    # Null comparison
    null_skeleton_hit_rates: List[float]
    null_mean: float
    selectivity: float

    # Comparison to Phase 16 CV model
    phase16_selectivity: float
    abjad_vs_cv: str  # 'ABJAD_BETTER' or 'CV_BETTER'

    runtime_seconds: float


# ---------------------------------------------------------------------------
# Root extraction
# ---------------------------------------------------------------------------

def _extract_roots(
    tokens: List[str],
    eva_to_triple: Dict[str, str],
) -> List[Tuple[str, List[str]]]:
    """Extract root portions of each token using morpheme decomposition.

    Returns list of (original_token, root_eva_chars).
    """
    results = []
    for token in tokens:
        decomp = decompose_token_morphemes(token)
        root_chars = decomp.stem_chars if hasattr(decomp, 'stem_chars') else []
        if not root_chars:
            # Fallback: use tokenize_eva_chars on the stem
            stem = decomp.stem if hasattr(decomp, 'stem') else token
            root_chars = tokenize_eva_chars(stem)
        results.append((token, root_chars))
    return results


def _root_to_triples(
    root_chars: List[str],
    eva_to_triple: Dict[str, str],
) -> List[str]:
    """Map EVA root characters to their triple keys."""
    triples = []
    for ch in root_chars:
        triple = eva_to_triple.get(ch)
        if triple:
            triples.append(triple)
    return triples


# ---------------------------------------------------------------------------
# Abjad decode
# ---------------------------------------------------------------------------

def _abjad_decode_roots(
    tokens: List[str],
    eva_to_triple: Dict[str, str],
    abjad_table: Dict[str, str],
) -> List[str]:
    """Decode root portions of tokens through abjad table.

    Returns list of consonant-skeleton strings (one per token).
    """
    decoded = []
    for token in tokens:
        decomp = decompose_token_morphemes(token)
        stem = decomp.stem if hasattr(decomp, 'stem') else token
        stem_chars = tokenize_eva_chars(stem)
        consonants = []
        for ch in stem_chars:
            triple = eva_to_triple.get(ch)
            if triple and triple in abjad_table:
                consonants.append(abjad_table[triple])
        decoded.append(''.join(consonants))
    return decoded


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _score_abjad_assignment(
    assignment: Dict[str, str],
    root_triple_sequences: List[List[str]],
    skeleton_dict: Dict[str, List[str]],
    ref_skeleton_bigrams: Counter,
    n_ref_bigrams: int,
) -> Tuple[float, float, int]:
    """Score an abjad assignment by skeleton matching and cross-entropy.

    Returns (cross_entropy, hit_rate, n_hits).
    """
    # Decode each root sequence
    decoded_skeletons = []
    for triple_seq in root_triple_sequences:
        consonants = ''.join(assignment.get(t, '?') for t in triple_seq)
        decoded_skeletons.append(consonants)

    # Skeleton hit rate
    n_hits = sum(1 for s in decoded_skeletons if s in skeleton_dict and len(s) >= 2)
    hit_rate = n_hits / len(decoded_skeletons) if decoded_skeletons else 0.0

    # Cross-entropy against reference consonant bigrams
    total_log_prob = 0.0
    n_bigrams = 0
    for skeleton in decoded_skeletons:
        if len(skeleton) < 2:
            continue
        for i in range(len(skeleton) - 1):
            bigram = skeleton[i:i+2]
            freq = ref_skeleton_bigrams.get(bigram, 0)
            prob = (freq + 1) / (n_ref_bigrams + 17 * 17)  # Laplace smoothing
            total_log_prob += math.log2(prob)
            n_bigrams += 1

    cross_entropy = -total_log_prob / n_bigrams if n_bigrams > 0 else 20.0

    return cross_entropy, hit_rate, n_hits


def _build_ref_skeleton_bigrams(
    ref_corpus,
) -> Tuple[Counter, int]:
    """Build consonant bigram counts from reference corpus."""
    bigrams: Counter = Counter()
    ref_tokens = ref_corpus.get_combined_tokens('latin')
    for word in ref_tokens:
        skeleton = _strip_vowels(word.lower())
        if len(skeleton) >= 2:
            for i in range(len(skeleton) - 1):
                bigrams[skeleton[i:i+2]] += 1

    total = sum(bigrams.values())
    return bigrams, total


# ---------------------------------------------------------------------------
# Beam search (simplified abjad-specific)
# ---------------------------------------------------------------------------

def _abjad_beam_search(
    triple_keys: List[str],
    triple_freqs: Dict[str, int],
    root_triple_sequences: List[List[str]],
    skeleton_dict: Dict[str, List[str]],
    ref_skeleton_bigrams: Counter,
    n_ref_bigrams: int,
    family_map: Dict[str, str],
    beam_width: int = 30,
    max_solutions: int = 10,
    seed: int = 42,
) -> List[Tuple[Dict[str, str], float, float, int]]:
    """Beam search for abjad assignments.

    Returns list of (assignment, cross_entropy, hit_rate, n_hits).
    """
    rng = random.Random(seed)

    # Sort triples by frequency (MRV — most constrained first)
    sorted_triples = sorted(triple_keys, key=lambda t: triple_freqs.get(t, 0),
                            reverse=True)

    # Beam: list of (partial_assignment, score)
    beam: List[Tuple[Dict[str, str], float]] = [({}, 0.0)]

    for triple in sorted_triples:
        new_beam: List[Tuple[Dict[str, str], float]] = []

        for partial, prev_score in beam:
            # Determine candidate consonants
            used = set(partial.values())
            candidates = [c for c in CONSONANT_DOMAIN if c not in used]

            # If no candidates remain, allow reuse of clusters
            if not candidates:
                candidates = CONSONANT_DOMAIN[:]

            # Frequency hint: prefer consonants matching frequency rank
            freq_rank = len([t for t in sorted_triples
                             if t in partial]) if partial else 0
            if freq_rank < len(LATIN_CONSONANT_FREQ_RANK):
                preferred = LATIN_CONSONANT_FREQ_RANK[freq_rank]
                if preferred in candidates:
                    candidates.remove(preferred)
                    candidates.insert(0, preferred)

            for consonant in candidates[:8]:  # Limit branching
                new_assign = dict(partial)
                new_assign[triple] = consonant

                # Quick score: count how many decoded roots match skeleton dict
                quick_hits = 0
                for seq in root_triple_sequences[:500]:
                    decoded = ''.join(new_assign.get(t, '?') for t in seq)
                    if '?' not in decoded and decoded in skeleton_dict and len(decoded) >= 2:
                        quick_hits += 1

                score = quick_hits
                new_beam.append((new_assign, score))

        # Prune to beam width
        new_beam.sort(key=lambda x: x[1], reverse=True)
        beam = new_beam[:beam_width]

    # Score final candidates fully
    results = []
    for assignment, _ in beam:
        ce, hit_rate, n_hits = _score_abjad_assignment(
            assignment, root_triple_sequences, skeleton_dict,
            ref_skeleton_bigrams, n_ref_bigrams,
        )
        results.append((assignment, ce, hit_rate, n_hits))

    results.sort(key=lambda x: (-x[2], x[1]))  # Best hit rate, lowest CE
    return results[:max_solutions]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_abjad_csp() -> None:
    """Step 34.2: Abjad consonant-only CSP."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 34.2: Abjad Consonant-Only CSP (Track A)")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load sigla dictionary ──
    print("\n  1. Loading sigla dictionary …")
    sigla_path = os.path.join(rd, 'sigla_dictionary.json')
    if not os.path.exists(sigla_path):
        print("  [SKIP] sigla_dictionary.json not found — run sigla-dict first")
        return

    # Build consonant skeleton dictionary from base words
    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    base_words = set(
        w.lower() for w in ref_corpus.get_combined_tokens('latin')
        if len(w) >= 2
    )
    skeleton_dict: Dict[str, List[str]] = defaultdict(list)
    for word in sorted(base_words):
        skel = _strip_vowels(word)
        if len(skel) >= 1:
            skeleton_dict[skel].append(word)
    skeleton_dict = dict(skeleton_dict)
    print(f"     {len(skeleton_dict)} consonant skeletons from {len(base_words)} words")

    # ── 2. Load modifiers + corpus ──
    print("\n  2. Loading corpus + modifiers …")
    with open(os.path.join(rd, 'modifier_integrate.json')) as f:
        mod_data = json.load(f)
    modifier_chars, modifier_rules = _reconstruct_modifier_rules(mod_data)

    corpus = load_corpus(verbose=False)
    eva_to_triple = build_eva_to_triple_lookup()

    all_tokens: List[str] = []
    for _, page in corpus.pages.items():
        for token in page.all_tokens:
            all_tokens.append(token)
    print(f"     {len(all_tokens)} tokens")

    # ── 3. Extract roots and map to triple sequences ──
    print("\n  3. Extracting roots …")
    root_triple_sequences: List[List[str]] = []
    root_strings: List[str] = []
    for token in all_tokens:
        decomp = decompose_token_morphemes(token)
        stem = decomp.stem if hasattr(decomp, 'stem') else token
        stem_chars = tokenize_eva_chars(stem)
        triples = _root_to_triples(stem_chars, eva_to_triple)
        root_triple_sequences.append(triples)
        root_strings.append(stem)

    # Collect all active triples and their frequencies
    triple_freqs: Counter = Counter()
    for seq in root_triple_sequences:
        for t in seq:
            triple_freqs[t] += 1
    triple_keys = sorted(triple_freqs.keys(),
                         key=lambda t: triple_freqs[t], reverse=True)
    print(f"     {len(triple_keys)} active triples, {len(root_triple_sequences)} root sequences")

    # ── 4. Load sign families for constraints ──
    print("\n  4. Loading sign families …")
    family_map: Dict[str, str] = {}
    tachy_path = os.path.join(rd, 'tachygraphic_stroke.json')
    if os.path.exists(tachy_path):
        with open(tachy_path) as f:
            tachy_data = json.load(f)
        for fam in tachy_data.get('families', []):
            fam_name = fam.get('family_name', '')
            for member in fam.get('members', []):
                family_map[member] = fam_name
        print(f"     {len(family_map)} triples in {len(set(family_map.values()))} families")

    # ── 5. Build reference consonant bigrams ──
    print("\n  5. Building reference consonant bigrams …")
    ref_skeleton_bigrams, n_ref_bigrams = _build_ref_skeleton_bigrams(ref_corpus)
    print(f"     {len(ref_skeleton_bigrams)} unique consonant bigrams, {n_ref_bigrams} total")

    # ── 6. Run abjad beam search ──
    print("\n  6. Running abjad beam search …")
    solutions = _abjad_beam_search(
        triple_keys, triple_freqs, root_triple_sequences,
        skeleton_dict, ref_skeleton_bigrams, n_ref_bigrams,
        family_map, beam_width=30, max_solutions=10,
    )

    if not solutions:
        print("  [FAIL] No solutions found")
        return

    best_assign, best_ce, best_hit_rate, best_n_hits = solutions[0]
    print(f"     Best: hit_rate={best_hit_rate:.3f} ({best_n_hits} hits), CE={best_ce:.3f}")
    print(f"     Found {len(solutions)} solutions total")

    # ── 7. Decode roots through best assignment ──
    print("\n  7. Decoding sample roots …")
    sample_decoded = []
    for i, (token, seq) in enumerate(zip(all_tokens[:100], root_triple_sequences[:100])):
        consonants = ''.join(best_assign.get(t, '?') for t in seq)
        matches = skeleton_dict.get(consonants, [])
        sample_decoded.append({
            'token': token,
            'root_triples': seq,
            'consonants': consonants,
            'matches': matches[:5],
            'is_hit': consonants in skeleton_dict and len(consonants) >= 2,
        })

    # ── 8. Null comparison ──
    print("\n  8. Running null comparison …")
    null_seeds = [100, 101, 102, 103, 104]
    null_path = os.path.join(rd, 'null_corpus.json')
    if os.path.exists(null_path):
        with open(null_path) as f:
            null_data = json.load(f)
        null_seeds = [r['seed'] for r in null_data.get('null_runs', [])]

    bigram_probs, initial_probs, token_lengths = _build_eva_bigram_model(all_tokens)
    null_hit_rates: List[float] = []

    for seed in null_seeds:
        null_tokens = _generate_null_corpus(
            bigram_probs, initial_probs, token_lengths, len(all_tokens), seed,
        )
        null_decoded = _abjad_decode_roots(null_tokens, eva_to_triple, best_assign)
        null_hits = sum(1 for s in null_decoded if s in skeleton_dict and len(s) >= 2)
        null_hit_rates.append(null_hits / len(null_decoded) if null_decoded else 0.0)

    null_mean = sum(null_hit_rates) / len(null_hit_rates) if null_hit_rates else 0.0
    selectivity = best_hit_rate / null_mean if null_mean > 0 else float('inf')

    print(f"     Null mean hit rate: {null_mean:.3f}")
    print(f"     Selectivity: {selectivity:.2f}×")

    # ── 9. Compare to Phase 16 ──
    phase16_selectivity = 3.38  # From MEMORY
    verdict = 'ABJAD_BETTER' if selectivity > phase16_selectivity else 'CV_BETTER'
    print(f"     Phase 16 selectivity: {phase16_selectivity:.2f}×")
    print(f"     Verdict: {verdict}")

    elapsed = time.time() - t0

    result = AbjadCSPResult(
        n_variables=len(triple_keys),
        n_domain=len(CONSONANT_DOMAIN),
        triple_frequencies=dict(triple_freqs.most_common(30)),
        best_assignment=best_assign,
        best_cross_entropy=round(best_ce, 4),
        n_beam_solutions=len(solutions),
        skeleton_hit_rate=round(best_hit_rate, 4),
        n_skeleton_hits=best_n_hits,
        n_root_tokens=len(root_triple_sequences),
        sample_decoded_roots=sample_decoded[:30],
        null_skeleton_hit_rates=[round(r, 4) for r in null_hit_rates],
        null_mean=round(null_mean, 4),
        selectivity=round(selectivity, 4),
        phase16_selectivity=phase16_selectivity,
        abjad_vs_cv=verdict,
        runtime_seconds=round(elapsed, 1),
    )

    out_path = os.path.join(rd, 'abjad_csp.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)
    print(f"\n  Saved → {out_path}")
    print(f"\n  Completed in {elapsed:.1f}s")
