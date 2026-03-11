"""
Step 43.3 – Encoding Table Search
===================================
Search the space of encoding tables via simulated annealing to find the
one whose encoded output best matches the Voynich statistical fingerprint.

Dependency chain:
    results/voynich_fingerprint.json  (Step 43.1: target fingerprint)
    results/tachygraphic_encoder.json (Step 43.2: encoder + initial table)
    data/reference/latin/             (Circa Instans)
    data/reference/italian/           (Anonimo Veneziano)
        → encoding_search.json        (this step)
"""

import json
import os
import time
import math
import copy
import random

import numpy as np
from collections import Counter
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import load_corpus, build_eva_to_triple_lookup, tokenize_eva_chars
from voynich.core.reference import EVA_VISUAL_COMPONENTS, load_reference_corpus
from voynich.core.stats import (
    syllabify_latin,
    first_order_entropy,
    conditional_entropy,
    simulated_annealing,
)


# ---------------------------------------------------------------------------
# JSON helpers
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


def _safe_load(path: str) -> Dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class EncodingSearchResult:
    # Best encoding found
    best_cost: float
    best_fingerprint_distance: float
    best_syllable_to_triple: Dict[str, str]
    # Convergence
    n_iterations: int
    n_restarts: int
    convergence_curve: List[float]
    # Null comparison
    null_distances: List[float]
    null_mean: float
    null_std: float
    selectivity_sigma: float
    # Per-language results
    latin_best_cost: float
    latin_best_distance: float
    italian_best_cost: float
    italian_best_distance: float
    best_language: str
    # Per-dimension matching
    per_dimension_match: Dict[str, float]  # dimension_name -> error
    well_matched_dims: List[str]
    poorly_matched_dims: List[str]
    # Gate
    gate_passed: bool
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Simplified fingerprint labels (the ~16 dimensions we optimize against)
# ---------------------------------------------------------------------------

SIMPLIFIED_LABELS = [
    'H1', 'H2',
    'mean_tok_len', 'std_tok_len',
    'ttr',
    'char_freq_0', 'char_freq_1', 'char_freq_2', 'char_freq_3', 'char_freq_4',
    'char_freq_5', 'char_freq_6', 'char_freq_7', 'char_freq_8', 'char_freq_9',
    'zipf_alpha',
]


# ---------------------------------------------------------------------------
# Pre-compute triple_to_eva mapping (deterministic: first EVA char sorted)
# ---------------------------------------------------------------------------

def _build_triple_to_first_eva() -> Dict[str, str]:
    """For each triple_key, return the alphabetically first EVA char.

    This gives a deterministic, fast mapping for fingerprint computation.
    """
    triple_to_chars: Dict[str, List[str]] = {}
    for eva_char, components in EVA_VISUAL_COMPONENTS.items():
        triple_key = (
            f"{components['first_stroke']},"
            f"{components['last_stroke']},"
            f"{components['glyph_class']}"
        )
        triple_to_chars.setdefault(triple_key, []).append(eva_char)
    return {tk: sorted(chars)[0] for tk, chars in triple_to_chars.items()}


def _build_triple_to_all_eva() -> Dict[str, List[str]]:
    """For each triple_key, return all EVA chars sorted."""
    triple_to_chars: Dict[str, List[str]] = {}
    for eva_char, components in EVA_VISUAL_COMPONENTS.items():
        triple_key = (
            f"{components['first_stroke']},"
            f"{components['last_stroke']},"
            f"{components['glyph_class']}"
        )
        triple_to_chars.setdefault(triple_key, []).append(eva_char)
    return {tk: sorted(chars) for tk, chars in triple_to_chars.items()}


def _build_glyph_class_to_triples() -> Dict[str, List[str]]:
    """Map each glyph_class to a list of triple_keys within that class."""
    class_to_triples: Dict[str, List[str]] = {}
    seen = set()
    for components in EVA_VISUAL_COMPONENTS.values():
        triple_key = (
            f"{components['first_stroke']},"
            f"{components['last_stroke']},"
            f"{components['glyph_class']}"
        )
        gc = components['glyph_class']
        if triple_key not in seen:
            class_to_triples.setdefault(gc, []).append(triple_key)
            seen.add(triple_key)
    return class_to_triples


# ---------------------------------------------------------------------------
# Extract Voynich target fingerprint (simplified dimensions)
# ---------------------------------------------------------------------------

def _extract_voynich_target(fp_data: Dict) -> Tuple[np.ndarray, List[str]]:
    """Extract the simplified target fingerprint from voynich_fingerprint.json.

    Returns (target_vector, dimension_labels) where target_vector has ~16
    elements corresponding to SIMPLIFIED_LABELS.
    """
    full_labels = fp_data['fingerprint']['labels']
    full_values = fp_data['fingerprint']['values']
    label_to_val = dict(zip(full_labels, full_values))

    # Also extract from the structured sub-dicts
    char_level = fp_data.get('char_level', {})
    token_level = fp_data.get('token_level', {})

    target = []
    labels_out = []

    # H1, H2 (from entropy curve or char_level)
    entropy_curve = char_level.get('entropy_curve', {})
    if isinstance(entropy_curve, dict):
        h1_val = label_to_val.get('entropy_H1', entropy_curve.get('H1', 0.0))
        h2_val = label_to_val.get('entropy_H2', entropy_curve.get('H2', 0.0))
    else:
        h1_val = label_to_val.get('entropy_H1', entropy_curve[1] if len(entropy_curve) > 1 else 0.0)
        h2_val = label_to_val.get('entropy_H2', entropy_curve[2] if len(entropy_curve) > 2 else 0.0)
    target.append(h1_val)
    labels_out.append('H1')
    target.append(h2_val)
    labels_out.append('H2')

    # Mean / std token length
    mean_tl = label_to_val.get('mean_tok_len', token_level.get('mean_token_length', 0.0))
    std_tl = label_to_val.get('std_tok_len', token_level.get('std_token_length', 0.0))
    target.append(mean_tl)
    labels_out.append('mean_tok_len')
    target.append(std_tl)
    labels_out.append('std_tok_len')

    # Type-token ratio
    ttr_val = label_to_val.get('ttr', token_level.get('type_token_ratio', 0.0))
    target.append(ttr_val)
    labels_out.append('ttr')

    # Top 10 character frequencies (sorted descending)
    char_freqs = char_level.get('char_freqs', {})
    if char_freqs:
        sorted_freqs = sorted(char_freqs.values(), reverse=True)
    else:
        # Fallback: extract from the full fingerprint (char_freq_* labels)
        sorted_freqs = sorted(
            [v for l, v in zip(full_labels, full_values) if l.startswith('char_freq_')],
            reverse=True,
        )
    for i in range(10):
        val = sorted_freqs[i] if i < len(sorted_freqs) else 0.0
        target.append(val)
        labels_out.append(f'char_freq_{i}')

    # Zipf alpha
    zipf_val = label_to_val.get('zipf_exponent', token_level.get('zipf_exponent', 0.0))
    target.append(zipf_val)
    labels_out.append('zipf_alpha')

    return np.array(target, dtype=np.float64), labels_out


# ---------------------------------------------------------------------------
# Pre-syllabify reference text
# ---------------------------------------------------------------------------

def _presyllabify(tokens: List[str]) -> List[List[str]]:
    """Syllabify all tokens (words) from reference text. Pre-compute once."""
    result = []
    for tok in tokens:
        clean = ''.join(c for c in tok.lower() if c.isalpha())
        if not clean:
            continue
        result.append(syllabify_latin(clean))
    return result


# ---------------------------------------------------------------------------
# Fast fingerprint computation for encoded text
# ---------------------------------------------------------------------------

def _compute_simplified_fingerprint(
    encoded_tokens: List[str],
) -> np.ndarray:
    """Compute the simplified fingerprint vector from a list of encoded tokens.

    Returns a numpy array of ~16 dimensions matching SIMPLIFIED_LABELS.
    """
    if not encoded_tokens:
        return np.zeros(len(SIMPLIFIED_LABELS), dtype=np.float64)

    # ── Character-level statistics ──
    # Work on raw character strings (single chars for the encoded EVA-like text)
    all_chars: List[str] = []
    for tok in encoded_tokens:
        all_chars.extend(tok)

    total_chars = len(all_chars)
    if total_chars == 0:
        return np.zeros(len(SIMPLIFIED_LABELS), dtype=np.float64)

    char_counts = Counter(all_chars)
    char_probs = {c: n / total_chars for c, n in char_counts.items()}

    # H1 — first-order entropy
    h1 = -sum(p * math.log2(p) for p in char_probs.values() if p > 0)

    # H2 — conditional entropy order 1 (bigram-based)
    if total_chars > 1:
        bigram_counts: Counter = Counter()
        unigram_ctx: Counter = Counter()
        for tok in encoded_tokens:
            for i in range(len(tok) - 1):
                bigram_counts[(tok[i], tok[i + 1])] += 1
                unigram_ctx[tok[i]] += 1
        total_bi = sum(bigram_counts.values())
        total_ctx = sum(unigram_ctx.values())
        if total_bi > 0 and total_ctx > 0:
            h_joint = -sum(
                (c / total_bi) * math.log2(c / total_bi)
                for c in bigram_counts.values() if c > 0
            )
            h_ctx = -sum(
                (c / total_ctx) * math.log2(c / total_ctx)
                for c in unigram_ctx.values() if c > 0
            )
            h2 = h_joint - h_ctx
        else:
            h2 = 0.0
    else:
        h2 = 0.0

    # ── Token-level statistics ──
    lengths = np.array([len(t) for t in encoded_tokens], dtype=np.float64)
    mean_tok_len = float(np.mean(lengths))
    std_tok_len = float(np.std(lengths))

    n_types = len(set(encoded_tokens))
    n_tokens = len(encoded_tokens)
    ttr = n_types / n_tokens if n_tokens > 0 else 0.0

    # Top 10 character frequencies (sorted descending)
    sorted_freqs = sorted(char_probs.values(), reverse=True)
    top10 = [sorted_freqs[i] if i < len(sorted_freqs) else 0.0 for i in range(10)]

    # Zipf alpha via log-log regression on rank-frequency
    if n_types >= 2:
        tok_counts = Counter(encoded_tokens)
        rank_freq = sorted(tok_counts.values(), reverse=True)
        ranks = np.arange(1, len(rank_freq) + 1, dtype=np.float64)
        freqs_arr = np.array(rank_freq, dtype=np.float64)
        # Log-log regression: log(freq) = -alpha * log(rank) + C
        log_ranks = np.log(ranks)
        log_freqs = np.log(freqs_arr)
        # Least-squares: slope = -alpha
        n = len(log_ranks)
        sum_x = np.sum(log_ranks)
        sum_y = np.sum(log_freqs)
        sum_xy = np.sum(log_ranks * log_freqs)
        sum_x2 = np.sum(log_ranks ** 2)
        denom = n * sum_x2 - sum_x ** 2
        if abs(denom) > 1e-12:
            slope = (n * sum_xy - sum_x * sum_y) / denom
            zipf_alpha = -slope
        else:
            zipf_alpha = 1.0
    else:
        zipf_alpha = 1.0

    # Assemble vector
    vec = [h1, h2, mean_tok_len, std_tok_len, ttr] + top10 + [zipf_alpha]
    return np.array(vec, dtype=np.float64)


# ---------------------------------------------------------------------------
# Fast encoding from syllabified words through state table
# ---------------------------------------------------------------------------

def _encode_syllabified(
    syllabified_words: List[List[str]],
    state: Dict[str, str],
    triple_to_eva: Dict[str, str],
    known_syllables_list: List[str],
) -> List[str]:
    """Encode pre-syllabified words through a syllable->triple state.

    Uses deterministic EVA mapping (triple_to_eva gives first sorted char).
    For unknown syllables, uses a nearest-match fallback based on CV prefix.

    Returns list of encoded token strings.
    """
    tokens: List[str] = []
    known_set = set(state.keys())

    # Pre-build a fallback cache for unknown syllables
    _fallback_cache: Dict[str, Optional[str]] = {}

    for syls in syllabified_words:
        chars: List[str] = []
        for syl in syls:
            if syl in state:
                mapped_syl = syl
            elif syl in _fallback_cache:
                mapped_syl = _fallback_cache[syl]
            else:
                mapped_syl = _find_nearest_syllable(syl, known_set)
                _fallback_cache[syl] = mapped_syl

            if mapped_syl is None:
                continue

            triple_key = state[mapped_syl]
            eva_char = triple_to_eva.get(triple_key)
            if eva_char is not None:
                chars.append(eva_char)

        if chars:
            tokens.append(''.join(chars))

    return tokens


def _find_nearest_syllable(
    syllable: str,
    known: set,
) -> Optional[str]:
    """Find nearest known syllable by simple heuristic (for speed)."""
    if syllable in known:
        return syllable

    vowels = set('aeiou')

    # Try CV truncation
    if len(syllable) >= 2:
        cv = syllable[:2]
        if cv in known:
            return cv

    # Try onset + common vowels
    if len(syllable) >= 1 and syllable[0] not in vowels:
        for v in 'aeio':
            cand = syllable[0] + v
            if cand in known:
                return cand

    # Try just the vowel nucleus if pure vowel syllable
    if len(syllable) >= 1 and syllable[0] in vowels:
        for v in 'aeio':
            if v in known:
                return v

    # Fallback: first known syllable alphabetically closest
    # (Skip expensive edit-distance for speed in inner loop)
    return None


# ---------------------------------------------------------------------------
# Cost function: weighted Euclidean distance to Voynich fingerprint
# ---------------------------------------------------------------------------

def _build_cost_fn(
    syllabified_words: List[List[str]],
    target_vec: np.ndarray,
    triple_to_eva: Dict[str, str],
    known_syllables_list: List[str],
    norm_factors: np.ndarray,
):
    """Return a cost function: state -> float (lower is better).

    The cost is the weighted Euclidean distance between the encoded text's
    simplified fingerprint and the Voynich target, normalized per-dimension.
    """
    def cost_fn(state: Dict[str, str]) -> float:
        tokens = _encode_syllabified(
            syllabified_words, state, triple_to_eva, known_syllables_list,
        )
        if not tokens:
            return 1e6

        enc_vec = _compute_simplified_fingerprint(tokens)
        # Normalized distance
        diff = (enc_vec - target_vec) / norm_factors
        return float(np.sqrt(np.sum(diff ** 2)))

    return cost_fn


# ---------------------------------------------------------------------------
# Proposal function for simulated annealing
# ---------------------------------------------------------------------------

def _build_propose_fn(
    all_triples: List[str],
    class_to_triples: Dict[str, List[str]],
    triple_to_class: Dict[str, str],
):
    """Return a proposal function: (state, rng) -> new_state.

    Mutation types:
      0.5 probability — swap two syllable->triple assignments
      0.3 probability — reassign one syllable to a different triple
      0.2 probability — reassign one syllable to a random triple
                        within the same glyph_class family
    """
    def propose_fn(state: Dict[str, str], rng: random.Random) -> Dict[str, str]:
        new_state = dict(state)
        syllables = list(new_state.keys())
        if len(syllables) < 2:
            return new_state

        r = rng.random()

        if r < 0.5:
            # Swap two syllable->triple assignments
            s1, s2 = rng.sample(syllables, 2)
            new_state[s1], new_state[s2] = new_state[s2], new_state[s1]

        elif r < 0.8:
            # Reassign one syllable to a different triple
            s = rng.choice(syllables)
            current = new_state[s]
            candidates = [t for t in all_triples if t != current]
            if candidates:
                new_state[s] = rng.choice(candidates)

        else:
            # Reassign one syllable to a triple in the same family
            s = rng.choice(syllables)
            current = new_state[s]
            gc = triple_to_class.get(current, '')
            family_triples = class_to_triples.get(gc, all_triples)
            candidates = [t for t in family_triples if t != current]
            if candidates:
                new_state[s] = rng.choice(candidates)
            else:
                # Fall back to any triple
                candidates = [t for t in all_triples if t != current]
                if candidates:
                    new_state[s] = rng.choice(candidates)

        return new_state

    return propose_fn


# ---------------------------------------------------------------------------
# Generate random encoding tables (for null comparison)
# ---------------------------------------------------------------------------

def _random_encoding_table(
    syllables: List[str],
    all_triples: List[str],
    rng: random.Random,
) -> Dict[str, str]:
    """Generate a random syllable->triple mapping."""
    return {syl: rng.choice(all_triples) for syl in syllables}


# ---------------------------------------------------------------------------
# Build initial encoding table from tachygraphic_encoder.json or fallback
# ---------------------------------------------------------------------------

def _build_initial_state(
    encoder_data: Dict,
    combined_data: Dict,
    all_triples: List[str],
    syllabified_words: List[List[str]],
) -> Dict[str, str]:
    """Build the initial syllable->triple state for annealing.

    Prefers the syllable_to_triple from tachygraphic_encoder.json.
    Falls back to combined_refine.json (inverted assignment).
    If neither available, builds a random one seeded from observed syllables.
    """
    # Try tachygraphic encoder output first
    syl_to_triple = encoder_data.get('syllable_to_triple', {})
    if syl_to_triple:
        # Ensure all syllables observed in text have a mapping
        observed_syls = set()
        for word_syls in syllabified_words:
            for s in word_syls:
                observed_syls.add(s)

        state = dict(syl_to_triple)
        rng = random.Random(42)
        for s in observed_syls:
            if s not in state:
                state[s] = rng.choice(all_triples)
        return state

    # Fallback: invert combined_refine assignment (triple->syllable to syllable->triple)
    assignment = combined_data.get('best_assignment', {})
    if assignment:
        inverted: Dict[str, str] = {}
        for triple_key, syllable in assignment.items():
            if syllable not in inverted:
                inverted[syllable] = triple_key

        observed_syls = set()
        for word_syls in syllabified_words:
            for s in word_syls:
                observed_syls.add(s)

        state = dict(inverted)
        rng = random.Random(42)
        for s in observed_syls:
            if s not in state:
                state[s] = rng.choice(all_triples)
        return state

    # Last fallback: random
    observed_syls = set()
    for word_syls in syllabified_words:
        for s in word_syls:
            observed_syls.add(s)

    rng = random.Random(42)
    return {s: rng.choice(all_triples) for s in observed_syls}


# ---------------------------------------------------------------------------
# Run search for one language
# ---------------------------------------------------------------------------

def _run_search_for_language(
    lang_name: str,
    syllabified_words: List[List[str]],
    target_vec: np.ndarray,
    norm_factors: np.ndarray,
    init_state: Dict[str, str],
    all_triples: List[str],
    class_to_triples: Dict[str, List[str]],
    triple_to_class: Dict[str, str],
    triple_to_eva: Dict[str, str],
    max_iter: int = 100_000,
    n_restarts: int = 5,
) -> Tuple[Dict[str, str], float, List[float]]:
    """Run simulated annealing for one language's reference text.

    Returns (best_state, best_cost, convergence_history).
    """
    known_syls = list(init_state.keys())

    cost_fn = _build_cost_fn(
        syllabified_words, target_vec, triple_to_eva, known_syls, norm_factors,
    )
    propose_fn = _build_propose_fn(all_triples, class_to_triples, triple_to_class)

    print(f"    Running SA for {lang_name}: {max_iter} iters x {n_restarts} restarts …")
    best_state, best_cost, history = simulated_annealing(
        cost_fn=cost_fn,
        init_state=init_state,
        propose_fn=propose_fn,
        max_iter=max_iter,
        t_start=1.0,
        t_end=0.001,
        n_restarts=n_restarts,
        seed=42,
        verbose=True,
        checkpoint_interval=10_000,
    )

    return best_state, best_cost, history


# ---------------------------------------------------------------------------
# Null comparison
# ---------------------------------------------------------------------------

def _null_comparison(
    syllabified_words: List[List[str]],
    target_vec: np.ndarray,
    norm_factors: np.ndarray,
    triple_to_eva: Dict[str, str],
    syllables: List[str],
    all_triples: List[str],
    n_null: int = 20,
) -> List[float]:
    """Compute fingerprint distances for random encoding tables."""
    null_distances: List[float] = []
    for i in range(n_null):
        rng = random.Random(1000 + i)
        rand_state = _random_encoding_table(syllables, all_triples, rng)
        known_syls = list(rand_state.keys())

        tokens = _encode_syllabified(
            syllabified_words, rand_state, triple_to_eva, known_syls,
        )
        if not tokens:
            null_distances.append(1e6)
            continue

        enc_vec = _compute_simplified_fingerprint(tokens)
        diff = (enc_vec - target_vec) / norm_factors
        dist = float(np.sqrt(np.sum(diff ** 2)))
        null_distances.append(dist)

    return null_distances


# ---------------------------------------------------------------------------
# Per-dimension matching analysis
# ---------------------------------------------------------------------------

def _per_dimension_analysis(
    best_state: Dict[str, str],
    syllabified_words: List[List[str]],
    target_vec: np.ndarray,
    triple_to_eva: Dict[str, str],
    dim_labels: List[str],
) -> Tuple[Dict[str, float], List[str], List[str]]:
    """Compute per-dimension error and classify well/poorly matched dims."""
    known_syls = list(best_state.keys())
    tokens = _encode_syllabified(
        syllabified_words, best_state, triple_to_eva, known_syls,
    )
    enc_vec = _compute_simplified_fingerprint(tokens)

    per_dim: Dict[str, float] = {}
    well_matched: List[str] = []
    poorly_matched: List[str] = []

    for i, label in enumerate(dim_labels):
        t = target_vec[i]
        e = enc_vec[i]
        if abs(t) > 1e-9:
            error = abs(e - t) / abs(t)
        else:
            error = abs(e - t)
        per_dim[label] = round(error, 4)
        if error < 0.10:
            well_matched.append(label)
        else:
            poorly_matched.append(label)

    return per_dim, well_matched, poorly_matched


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_encoding_search() -> None:
    """Step 43.3: search encoding table space via simulated annealing."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 43.3: Encoding Table Search")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load Voynich fingerprint ──
    print("\n  1. Loading Voynich fingerprint …")
    fp_path = os.path.join(rd, 'voynich_fingerprint.json')
    fp_data = _safe_load(fp_path)
    if not fp_data or 'fingerprint' not in fp_data:
        raise FileNotFoundError(
            f"voynich_fingerprint.json not found or missing 'fingerprint' key at {fp_path}"
        )
    target_vec, dim_labels = _extract_voynich_target(fp_data)
    print(f"     Target fingerprint: {len(target_vec)} dimensions")
    for i, (lab, val) in enumerate(zip(dim_labels, target_vec)):
        if i < 5 or i >= len(dim_labels) - 1:
            print(f"       {lab}: {val:.6f}")
        elif i == 5:
            print(f"       …")

    # Normalization factors: use target values (clamp to avoid div-by-zero)
    norm_factors = np.maximum(np.abs(target_vec), 0.01)

    # ── 2. Load tachygraphic encoder (initial table) ──
    print("\n  2. Loading tachygraphic encoder …")
    enc_path = os.path.join(rd, 'tachygraphic_encoder.json')
    encoder_data = _safe_load(enc_path)
    n_enc_syls = len(encoder_data.get('syllable_to_triple', {}))
    print(f"     Encoder syllable mappings: {n_enc_syls}")

    combined_path = os.path.join(rd, 'combined_refine.json')
    combined_data = _safe_load(combined_path)
    n_combined = len(combined_data.get('best_assignment', {}))
    print(f"     Combined refine assignment: {n_combined} triples")

    # ── 3. Build triple lookups ──
    print("\n  3. Building triple lookups …")
    triple_to_eva = _build_triple_to_first_eva()
    all_triples = sorted(triple_to_eva.keys())
    class_to_triples = _build_glyph_class_to_triples()
    print(f"     Unique triples: {len(all_triples)}")
    print(f"     Glyph classes: {sorted(class_to_triples.keys())}")

    # Build triple -> glyph_class lookup
    triple_to_class: Dict[str, str] = {}
    for gc, triples in class_to_triples.items():
        for tk in triples:
            triple_to_class[tk] = gc

    # ── 4. Load reference texts ──
    print("\n  4. Loading reference texts …")
    try:
        ref_latin = load_reference_corpus(languages=['latin'], verbose=False)
        latin_tokens = ref_latin.get_combined_tokens('latin')
    except Exception:
        latin_tokens = []
    print(f"     Latin tokens: {len(latin_tokens)}")

    try:
        ref_italian = load_reference_corpus(languages=['italian'], verbose=False)
        italian_tokens = ref_italian.get_combined_tokens('italian')
    except Exception:
        italian_tokens = []
    print(f"     Italian tokens: {len(italian_tokens)}")

    # ── 5. Pre-syllabify reference texts ──
    # Use a reasonable subsample for speed (5000 words)
    print("\n  5. Pre-syllabifying reference texts …")
    MAX_WORDS = 5000
    latin_syls = _presyllabify(latin_tokens[:MAX_WORDS]) if latin_tokens else []
    italian_syls = _presyllabify(italian_tokens[:MAX_WORDS]) if italian_tokens else []
    print(f"     Latin syllabified words: {len(latin_syls)}")
    print(f"     Italian syllabified words: {len(italian_syls)}")

    if latin_syls:
        flat = [s for w in latin_syls for s in w]
        print(f"     Latin total syllables: {len(flat)}, unique: {len(set(flat))}")
    if italian_syls:
        flat = [s for w in italian_syls for s in w]
        print(f"     Italian total syllables: {len(flat)}, unique: {len(set(flat))}")

    # ── 6. Build initial states ──
    print("\n  6. Building initial encoding states …")
    latin_init = _build_initial_state(
        encoder_data, combined_data, all_triples, latin_syls,
    ) if latin_syls else {}
    italian_init = _build_initial_state(
        encoder_data, combined_data, all_triples, italian_syls,
    ) if italian_syls else {}
    print(f"     Latin init state: {len(latin_init)} syllables")
    print(f"     Italian init state: {len(italian_init)} syllables")

    # ── 7. Run SA for Latin ──
    latin_best_state = {}
    latin_best_cost = 1e6
    latin_history: List[float] = []

    if latin_syls and latin_init:
        print("\n  7a. Simulated annealing — Latin")
        latin_best_state, latin_best_cost, latin_history = _run_search_for_language(
            'Latin', latin_syls, target_vec, norm_factors,
            latin_init, all_triples, class_to_triples, triple_to_class,
            triple_to_eva,
            max_iter=100_000,
            n_restarts=5,
        )
        print(f"     Latin best cost: {latin_best_cost:.6f}")
    else:
        print("\n  7a. Skipping Latin (no reference text)")

    # ── 8. Run SA for Italian ──
    italian_best_state = {}
    italian_best_cost = 1e6
    italian_history: List[float] = []

    if italian_syls and italian_init:
        print("\n  7b. Simulated annealing — Italian")
        italian_best_state, italian_best_cost, italian_history = _run_search_for_language(
            'Italian', italian_syls, target_vec, norm_factors,
            italian_init, all_triples, class_to_triples, triple_to_class,
            triple_to_eva,
            max_iter=100_000,
            n_restarts=5,
        )
        print(f"     Italian best cost: {italian_best_cost:.6f}")
    else:
        print("\n  7b. Skipping Italian (no reference text)")

    # ── 9. Select overall best ──
    print("\n  8. Selecting best language …")
    if latin_best_cost <= italian_best_cost:
        best_language = 'latin'
        best_state = latin_best_state
        best_cost = latin_best_cost
        best_history = latin_history
        best_syls = latin_syls
    else:
        best_language = 'italian'
        best_state = italian_best_state
        best_cost = italian_best_cost
        best_history = italian_history
        best_syls = italian_syls
    print(f"     Best: {best_language} (cost={best_cost:.6f})")

    # ── 10. Null comparison ──
    print("\n  9. Null comparison (20 random tables) …")
    if best_syls and best_state:
        null_syls = list(best_state.keys())
        null_dists = _null_comparison(
            best_syls, target_vec, norm_factors, triple_to_eva,
            null_syls, all_triples, n_null=20,
        )
        null_mean = float(np.mean(null_dists))
        null_std = float(np.std(null_dists))
        if null_std > 1e-9:
            selectivity_sigma = (null_mean - best_cost) / null_std
        else:
            selectivity_sigma = 0.0
        print(f"     Null mean: {null_mean:.4f}, std: {null_std:.4f}")
        print(f"     Best cost: {best_cost:.4f}")
        print(f"     Selectivity sigma: {selectivity_sigma:.2f}")
    else:
        null_dists = []
        null_mean = 0.0
        null_std = 0.0
        selectivity_sigma = 0.0

    # ── 11. Per-dimension matching ──
    print("\n  10. Per-dimension matching analysis …")
    if best_state and best_syls:
        per_dim, well_matched, poorly_matched = _per_dimension_analysis(
            best_state, best_syls, target_vec, triple_to_eva, dim_labels,
        )
        print(f"     Well matched (<10% error): {len(well_matched)} dims")
        for d in well_matched:
            print(f"       {d}: {per_dim[d]:.4f}")
        print(f"     Poorly matched (>=10% error): {len(poorly_matched)} dims")
        for d in poorly_matched[:5]:
            print(f"       {d}: {per_dim[d]:.4f}")
        if len(poorly_matched) > 5:
            print(f"       … and {len(poorly_matched) - 5} more")
    else:
        per_dim = {}
        well_matched = []
        poorly_matched = []

    # ── 12. Gate ──
    gate_passed = selectivity_sigma > 2.0
    if gate_passed:
        verdict = (
            f"PASS: selectivity_sigma={selectivity_sigma:.2f} > 2.0. "
            f"Best language={best_language}, cost={best_cost:.4f}, "
            f"{len(well_matched)}/{len(dim_labels)} dimensions well-matched."
        )
    else:
        verdict = (
            f"FAIL: selectivity_sigma={selectivity_sigma:.2f} <= 2.0. "
            f"Best language={best_language}, cost={best_cost:.4f}, "
            f"{len(well_matched)}/{len(dim_labels)} dimensions well-matched."
        )

    print(f"\n  GATE: {verdict}")

    # ── 13. Save ──
    elapsed = time.time() - t0

    # Total iterations = max_iter * n_restarts * (1 for latin + 1 for italian if run)
    n_langs_run = (1 if latin_syls else 0) + (1 if italian_syls else 0)
    total_iters = 100_000 * 5 * n_langs_run

    result = EncodingSearchResult(
        best_cost=round(best_cost, 6),
        best_fingerprint_distance=round(best_cost, 6),
        best_syllable_to_triple=best_state if best_state else {},
        n_iterations=total_iters,
        n_restarts=5 * n_langs_run,
        convergence_curve=[round(c, 6) for c in best_history],
        null_distances=[round(d, 6) for d in null_dists],
        null_mean=round(null_mean, 6),
        null_std=round(null_std, 6),
        selectivity_sigma=round(selectivity_sigma, 4),
        latin_best_cost=round(latin_best_cost, 6),
        latin_best_distance=round(latin_best_cost, 6),
        italian_best_cost=round(italian_best_cost, 6),
        italian_best_distance=round(italian_best_cost, 6),
        best_language=best_language,
        per_dimension_match=per_dim,
        well_matched_dims=well_matched,
        poorly_matched_dims=poorly_matched,
        gate_passed=gate_passed,
        verdict=verdict,
        runtime_seconds=round(elapsed, 1),
    )

    out_path = os.path.join(rd, 'encoding_search.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)
    print(f"\n  -> {out_path} ({elapsed:.1f}s)")
