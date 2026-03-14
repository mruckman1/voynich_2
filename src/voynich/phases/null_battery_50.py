"""
Phase 50 Track C – Extended Null Battery
==========================================
Five independent null tests to validate Phase 49/50 results.

Dependency chain:
    combined_refine.json    (Phase 15)
    signal_bigrams.json     (Phase 29)
        -> null_battery_50.json
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import build_eva_to_triple_lookup, load_corpus, tokenize_eva_chars
from voynich.core.reference import load_reference_corpus
from voynich.core.stats import build_ngram_lm, cross_entropy_lm
from scipy.stats import pearsonr


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
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return _convert(obj.tolist())
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


def _save_json(rd: str, filename: str, data: Any) -> str:
    path = os.path.join(rd, filename)
    with open(path, 'w') as f:
        json.dump(_convert(data), f, indent=2)
    return path


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class NullBattery50Result:
    c1_wrong_language: Dict
    c2_length_matched: Dict
    c3_section_specific: Dict
    c4_cross_validated: Dict
    c5_bigram_attribution: Dict
    tests_passed: int
    tests_total: int
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Shared helpers (same as Track A)
# ---------------------------------------------------------------------------

_ALPHABET = 'abcdefghijklmnopqrstuvwxyz'


def _build_10k_word_set() -> Set[str]:
    """Build a 10K-word reference dictionary from Latin + Italian."""
    ref = load_reference_corpus(languages=['latin', 'italian'], verbose=False)
    all_tokens: List[str] = []
    for lang in ('latin', 'italian'):
        all_tokens.extend(ref.get_combined_tokens(lang))
    freq = Counter(w.lower() for w in all_tokens if len(w) >= 2)
    return {w for w, _ in freq.most_common(10000)}


def _build_char_lm() -> Dict:
    """Build a character-level 5-gram LM from Latin + Italian reference text."""
    ref = load_reference_corpus(languages=['latin', 'italian'], verbose=False)
    all_tokens: List[str] = []
    for lang in ('latin', 'italian'):
        all_tokens.extend(ref.get_combined_tokens(lang))
    tokens = [w.lower() for w in all_tokens if len(w) >= 2 and w.isalpha()]
    return build_ngram_lm(tokens, order=5, smoothing=1.0)


def _generate_ed1(word: str) -> Set[str]:
    """Generate all edit-distance-1 variants (substitution, deletion, insertion)."""
    variants: Set[str] = set()
    n = len(word)

    # Substitutions
    for i in range(n):
        for c in _ALPHABET:
            if c != word[i]:
                variants.add(word[:i] + c + word[i + 1:])

    # Deletions
    for i in range(n):
        variants.add(word[:i] + word[i + 1:])

    # Insertions
    for i in range(n + 1):
        for c in _ALPHABET:
            variants.add(word[:i] + c + word[i:])

    return variants


def _decode_token(
    token: str,
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
) -> str:
    """Decode an EVA token to a syllable string via assignment table."""
    chars = tokenize_eva_chars(token)
    syllables: List[str] = []
    for ch in chars:
        triple_key = eva_to_triple.get(ch)
        if triple_key is None:
            continue
        syl = assignment.get(triple_key)
        if syl is not None:
            syllables.append(syl)
    return ''.join(syllables)


def _subsample_tokens(
    token_evas: List[str],
    token_folios: List[str],
    n: int = 5000,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[int]]:
    """Stratified subsample: proportional samples from each folio."""
    rng = random.Random(seed)

    # Group indices by folio
    folio_indices: Dict[str, List[int]] = {}
    for i, folio in enumerate(token_folios):
        folio_indices.setdefault(folio, []).append(i)

    total = len(token_evas)
    selected_indices: List[int] = []

    for folio, indices in sorted(folio_indices.items()):
        k = max(1, round(len(indices) * n / total))
        k = min(k, len(indices))
        selected_indices.extend(rng.sample(indices, k))

    if len(selected_indices) > n:
        rng.shuffle(selected_indices)
        selected_indices = selected_indices[:n]
    elif len(selected_indices) < n:
        remaining = [i for i in range(total) if i not in set(selected_indices)]
        rng.shuffle(remaining)
        selected_indices.extend(remaining[: n - len(selected_indices)])

    selected_indices.sort()

    sub_evas = [token_evas[i] for i in selected_indices]
    sub_folios = [token_folios[i] for i in selected_indices]
    return sub_evas, sub_folios, selected_indices


def _ed1_best(
    word: str,
    char_lm: Dict,
    word_set: Set[str],
    min_len: int = 4,
) -> Tuple[str, bool]:
    """Run the ED1 + char LM scoring pipeline on a single word.

    1. If len(word) >= min_len: generate ED1 variants; else candidates = {word}
    2. Filter to word_set hits
    3. Score with char LM, pick lowest CE
    4. If no dict hit: pick lowest CE overall
    5. Return (best_word, is_hit)
    """
    if len(word) >= min_len:
        candidates = _generate_ed1(word)
        candidates.add(word)
    else:
        candidates = {word}

    dict_hits = [v for v in candidates if v in word_set]

    if dict_hits:
        best = min(
            dict_hits,
            key=lambda w: cross_entropy_lm('_' + w + '_', char_lm, per_char=True),
        )
        return best, True
    else:
        best = min(
            candidates,
            key=lambda w: cross_entropy_lm('_' + w + '_', char_lm, per_char=True),
        )
        return best, False


# ---------------------------------------------------------------------------
# C.1: Wrong-Language LM Test
# ---------------------------------------------------------------------------

def _run_wrong_language_lm(
    token_evas: List[str],
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    word_set: Set[str],
) -> Dict:
    """Build separate char 5-gram LMs from each language and compare dict-hit rates."""
    print("\n  C.1: Wrong-Language LM Test")

    languages = ['latin', 'italian', 'german', 'occitan']
    lang_rates: Dict[str, float] = {}

    # Build Latin+Italian mixed LM (baseline)
    mixed_lm = _build_char_lm()
    mixed_hits = 0
    for token in token_evas:
        decoded = _decode_token(token, assignment, eva_to_triple)
        if not decoded:
            continue
        _, hit = _ed1_best(decoded, mixed_lm, word_set)
        if hit:
            mixed_hits += 1
    latin_italian_rate = mixed_hits / max(len(token_evas), 1)
    lang_rates['latin_italian'] = latin_italian_rate
    print(f"    latin_italian: {latin_italian_rate:.4f}")

    # Build per-language LMs
    for lang in languages:
        try:
            ref = load_reference_corpus(languages=[lang], verbose=False)
            lang_tokens = ref.get_combined_tokens(lang)
            lang_tokens = [w.lower() for w in lang_tokens if len(w) >= 2 and w.isalpha()]
            if len(lang_tokens) < 100:
                print(f"    {lang}: SKIP (too few tokens: {len(lang_tokens)})")
                lang_rates[lang] = 0.0
                continue
            lang_lm = build_ngram_lm(lang_tokens, order=5, smoothing=1.0)
        except Exception as e:
            print(f"    {lang}: SKIP ({e})")
            lang_rates[lang] = 0.0
            continue

        hits = 0
        for token in token_evas:
            decoded = _decode_token(token, assignment, eva_to_triple)
            if not decoded:
                continue
            _, hit = _ed1_best(decoded, lang_lm, word_set)
            if hit:
                hits += 1
        rate = hits / max(len(token_evas), 1)
        lang_rates[lang] = rate
        print(f"    {lang}: {rate:.4f}")

    german_rate = lang_rates.get('german', 0.0)
    gap = latin_italian_rate - german_rate

    if gap > 0.03:
        verdict = "LATIN_CONFIRMED"
    elif gap > 0.01:
        verdict = "LATIN_MARGINAL"
    else:
        verdict = "INDISTINGUISHABLE"

    print(f"    Gap (lat+ita - german): {gap:.4f} -> {verdict}")

    return {
        'lang_rates': lang_rates,
        'latin_italian_rate': latin_italian_rate,
        'german_rate': german_rate,
        'gap': gap,
        'verdict': verdict,
    }


# ---------------------------------------------------------------------------
# C.2: Length-Matched Random
# ---------------------------------------------------------------------------

def _run_length_matched_random(
    token_evas: List[str],
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    char_lm: Dict,
    word_set: Set[str],
    n_trials: int = 20,
) -> Dict:
    """Test whether length-matched random dict words achieve similar hit rate."""
    print("\n  C.2: Length-Matched Random Test")

    # Build length buckets from 10K dict
    length_buckets: Dict[int, List[str]] = defaultdict(list)
    for w in word_set:
        length_buckets[len(w)].append(w)

    # Decode real tokens
    decoded_words: List[str] = []
    for token in token_evas:
        decoded = _decode_token(token, assignment, eva_to_triple)
        decoded_words.append(decoded)

    # Real rate
    real_hits = 0
    for decoded in decoded_words:
        if not decoded:
            continue
        _, hit = _ed1_best(decoded, char_lm, word_set)
        if hit:
            real_hits += 1
    real_rate = real_hits / max(len(decoded_words), 1)
    print(f"    Real rate: {real_rate:.4f}")

    # Random trials
    trial_rates: List[float] = []
    for trial in range(n_trials):
        rng = random.Random(7000 + trial)
        trial_hits = 0
        for decoded in decoded_words:
            if not decoded:
                continue
            length = len(decoded)
            bucket = length_buckets.get(length, [])
            if bucket:
                random_word = rng.choice(bucket)
                # Score both with char LM, pick whichever has lower CE
                ce_decoded = cross_entropy_lm('_' + decoded + '_', char_lm, per_char=True)
                ce_random = cross_entropy_lm('_' + random_word + '_', char_lm, per_char=True)
                chosen = random_word if ce_random < ce_decoded else decoded
                if chosen in word_set:
                    trial_hits += 1
            else:
                if decoded in word_set:
                    trial_hits += 1
        rate = trial_hits / max(len(decoded_words), 1)
        trial_rates.append(rate)

    trial_mean = float(np.mean(trial_rates))
    trial_std = float(np.std(trial_rates, ddof=1)) if len(trial_rates) > 1 else 0.0

    print(f"    Length-matched random: mean={trial_mean:.4f}, std={trial_std:.4f}")
    print(f"    Real exceeds by: {real_rate - trial_mean:.4f}")

    return {
        'real_rate': real_rate,
        'random_mean': trial_mean,
        'random_std': trial_std,
        'n_trials': n_trials,
        'exceeds_2sigma': real_rate > trial_mean + 2 * trial_std,
    }


# ---------------------------------------------------------------------------
# C.3: Section-Specific Analysis
# ---------------------------------------------------------------------------

def _run_section_specific(
    token_evas: List[str],
    token_folios: List[str],
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    char_lm: Dict,
    word_set: Set[str],
) -> Dict:
    """Analyze dict-hit rate per section and check for length confound."""
    print("\n  C.3: Section-Specific Analysis")

    # Load corpus for section mapping
    corpus = load_corpus(verbose=False)
    folio_to_section: Dict[str, str] = {}
    for folio_id, page in corpus.pages.items():
        folio_to_section[folio_id] = page.section or 'unknown'

    # Group tokens by section
    section_tokens: Dict[str, List[str]] = defaultdict(list)
    section_folios_map: Dict[str, List[str]] = defaultdict(list)
    for token, folio in zip(token_evas, token_folios):
        section = folio_to_section.get(folio, 'unknown')
        section_tokens[section].append(token)
        section_folios_map[section].append(folio)

    per_section: Dict[str, Dict] = {}
    section_hit_rates: List[float] = []
    section_mean_lengths: List[float] = []

    for section, tokens in sorted(section_tokens.items()):
        if len(tokens) < 10:
            continue

        hits = 0
        total_length = 0
        n_decoded = 0
        for token in tokens:
            decoded = _decode_token(token, assignment, eva_to_triple)
            if not decoded:
                continue
            n_decoded += 1
            total_length += len(decoded)
            _, hit = _ed1_best(decoded, char_lm, word_set)
            if hit:
                hits += 1

        rate = hits / max(n_decoded, 1)
        mean_len = total_length / max(n_decoded, 1)

        per_section[section] = {
            'n_tokens': len(tokens),
            'n_decoded': n_decoded,
            'hit_rate': round(rate, 4),
            'mean_decoded_length': round(mean_len, 2),
        }
        section_hit_rates.append(rate)
        section_mean_lengths.append(mean_len)

        print(f"    {section}: rate={rate:.4f}, mean_len={mean_len:.2f}, n={n_decoded}")

    # Pearson correlation
    if len(section_hit_rates) >= 3:
        r, p_val = pearsonr(section_hit_rates, section_mean_lengths)
        r = float(r) if not math.isnan(r) else 0.0
        p_val = float(p_val) if not math.isnan(p_val) else 1.0
    else:
        r = 0.0
        p_val = 1.0

    if abs(r) > 0.5:
        verdict = "LENGTH_CONFOUND"
    else:
        verdict = "CONTENT_VARIATION"

    print(f"    Pearson r={r:.4f}, p={p_val:.4f} -> {verdict}")

    return {
        'per_section': per_section,
        'pearson_r': round(r, 4),
        'pearson_p': round(p_val, 4),
        'verdict': verdict,
    }


# ---------------------------------------------------------------------------
# C.4: Cross-Validated LM
# ---------------------------------------------------------------------------

def _run_cross_validated_lm(
    token_evas: List[str],
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    word_set: Set[str],
) -> Dict:
    """Test whether the char LM generalises by splitting reference corpus 50/50."""
    print("\n  C.4: Cross-Validated LM Test")

    # Load all Latin+Italian tokens
    ref = load_reference_corpus(languages=['latin', 'italian'], verbose=False)
    all_ref_tokens: List[str] = []
    for lang in ('latin', 'italian'):
        all_ref_tokens.extend(ref.get_combined_tokens(lang))
    all_ref_tokens = [w.lower() for w in all_ref_tokens if len(w) >= 2 and w.isalpha()]

    # Shuffle with seed=42, split 50/50
    rng = random.Random(42)
    shuffled = list(all_ref_tokens)
    rng.shuffle(shuffled)
    mid = len(shuffled) // 2
    train_tokens = shuffled[:mid]
    test_tokens = shuffled[mid:]

    # Build train-half LM
    train_lm = build_ngram_lm(train_tokens, order=5, smoothing=1.0)

    # Build test-half dictionary (top 10K)
    test_freq = Counter(test_tokens)
    test_dict = {w for w, _ in test_freq.most_common(10000)}

    print(f"    Train tokens: {len(train_tokens)}")
    print(f"    Test dict size: {len(test_dict)}")

    # Decode Voynich with train-half LM, evaluate against test-half dictionary
    cv_hits = 0
    n_decoded = 0
    for token in token_evas:
        decoded = _decode_token(token, assignment, eva_to_triple)
        if not decoded:
            continue
        n_decoded += 1
        _, hit = _ed1_best(decoded, train_lm, test_dict)
        if hit:
            cv_hits += 1
    cv_rate = cv_hits / max(n_decoded, 1)

    # Full (unsplit) rate using original word_set and full char LM
    full_lm = _build_char_lm()
    full_hits = 0
    n_full = 0
    for token in token_evas:
        decoded = _decode_token(token, assignment, eva_to_triple)
        if not decoded:
            continue
        n_full += 1
        _, hit = _ed1_best(decoded, full_lm, word_set)
        if hit:
            full_hits += 1
    full_rate = full_hits / max(n_full, 1)

    ratio = cv_rate / full_rate if full_rate > 0 else 0.0

    if ratio > 0.7:
        verdict = "SIGNAL_TRANSFERS"
    elif ratio < 0.4:
        verdict = "LM_MEMORIZING"
    else:
        verdict = "PARTIAL_TRANSFER"

    print(f"    CV rate: {cv_rate:.4f}")
    print(f"    Full rate: {full_rate:.4f}")
    print(f"    Ratio: {ratio:.4f} -> {verdict}")

    return {
        'cross_validated_rate': round(cv_rate, 4),
        'full_rate': round(full_rate, 4),
        'ratio': round(ratio, 4),
        'n_train': len(train_tokens),
        'n_test_dict': len(test_dict),
        'verdict': verdict,
    }


# ---------------------------------------------------------------------------
# C.5: Bigram Attribution
# ---------------------------------------------------------------------------

def _run_bigram_attribution(
    token_evas: List[str],
    token_folios: List[str],
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    char_lm: Dict,
    word_set: Set[str],
) -> Dict:
    """Classify consecutive-pair bigrams by whether they came from ED1 or raw decode."""
    print("\n  C.5: Bigram Attribution Test")

    # Build reference bigram set from Latin+Italian
    ref = load_reference_corpus(languages=['latin', 'italian'], verbose=False)
    all_ref_tokens: List[str] = []
    for lang in ('latin', 'italian'):
        all_ref_tokens.extend(ref.get_combined_tokens(lang))
    ref_lowered = [w.lower() for w in all_ref_tokens if len(w) >= 2]

    ref_bigrams: Set[Tuple[str, str]] = set()
    for i in range(len(ref_lowered) - 1):
        ref_bigrams.add((ref_lowered[i], ref_lowered[i + 1]))

    # Decode each token RAW and ED1
    raw_words: List[str] = []
    ed1_words: List[str] = []
    ed1_hits: List[bool] = []

    for token in token_evas:
        raw = _decode_token(token, assignment, eva_to_triple)
        raw_words.append(raw)

        if raw:
            best_word, hit = _ed1_best(raw, char_lm, word_set)
            ed1_words.append(best_word)
            ed1_hits.append(hit)
        else:
            ed1_words.append('')
            ed1_hits.append(False)

    # Count consecutive pair bigram hits
    direct_count = 0
    ed1_count = 0
    total_pairs = 0

    for i in range(len(ed1_words) - 1):
        w1_ed1 = ed1_words[i]
        w2_ed1 = ed1_words[i + 1]

        # Both must be dict hits and len >= 3
        if not (ed1_hits[i] and ed1_hits[i + 1]):
            continue
        if len(w1_ed1) < 3 or len(w2_ed1) < 3:
            continue

        total_pairs += 1

        if (w1_ed1, w2_ed1) in ref_bigrams:
            # Classify source
            w1_raw = raw_words[i]
            w2_raw = raw_words[i + 1]
            if w1_ed1 == w1_raw and w2_ed1 == w2_raw:
                direct_count += 1
            else:
                ed1_count += 1

    print(f"    Total qualifying pairs: {total_pairs}")
    print(f"    DIRECT bigram hits: {direct_count}")
    print(f"    ED1 bigram hits: {ed1_count}")

    # Also try loading Viterbi results from word_lm_rescore.json
    rd = _results_dir()
    wlm_path = os.path.join(rd, 'word_lm_rescore.json')
    viterbi_count = 0
    viterbi_available = False
    if os.path.exists(wlm_path):
        wlm_data = _safe_load(wlm_path)
        viterbi_decoded = wlm_data.get('viterbi_decoded', [])
        if viterbi_decoded and len(viterbi_decoded) == len(token_evas):
            viterbi_available = True
            for i in range(len(viterbi_decoded) - 1):
                w1 = viterbi_decoded[i]
                w2 = viterbi_decoded[i + 1]
                if not w1 or not w2:
                    continue
                if len(w1) < 3 or len(w2) < 3:
                    continue
                if w1 in word_set and w2 in word_set:
                    if (w1, w2) in ref_bigrams:
                        viterbi_count += 1
            print(f"    Viterbi bigram hits: {viterbi_count}")

    return {
        'total_qualifying_pairs': total_pairs,
        'direct_bigram_hits': direct_count,
        'ed1_bigram_hits': ed1_count,
        'viterbi_bigram_hits': viterbi_count if viterbi_available else None,
        'viterbi_available': viterbi_available,
        'has_direct': direct_count > 0,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_null_battery_50() -> Dict[str, Any]:
    """Phase 50 Track C: Extended Null Battery (5 subtests)."""
    t0 = time.time()
    rd = _results_dir()

    print("=" * 70)
    print("Phase 50 Track C: Extended Null Battery")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("\n--- Step 1: Loading data ---")

    refine_path = os.path.join(rd, 'combined_refine.json')
    refine_data = _safe_load(refine_path)
    assignment: Dict[str, str] = refine_data.get('best_assignment', {})
    print(f"  Assignment table: {len(assignment)} triples")

    sig_path = os.path.join(rd, 'signal_bigrams.json')
    sig_data = _safe_load(sig_path)
    token_evas_all: List[str] = sig_data.get('token_evas', [])
    token_folios_all: List[str] = sig_data.get('token_folios', [])
    print(f"  Total tokens from signal_bigrams: {len(token_evas_all)}")

    # Fallback: if signal_bigrams doesn't have tokens, build from corpus
    if not token_evas_all:
        print("  signal_bigrams.json missing token_evas; loading from corpus...")
        corpus = load_corpus(verbose=False)
        for folio, page in corpus.pages.items():
            for token in page.all_tokens:
                token_evas_all.append(token)
                token_folios_all.append(folio)
        print(f"  Loaded {len(token_evas_all)} tokens from corpus")

    # ------------------------------------------------------------------
    # 2. Build infrastructure
    # ------------------------------------------------------------------
    print("\n--- Step 2: Building infrastructure ---")

    eva_to_triple = build_eva_to_triple_lookup()
    print(f"  EVA-to-triple lookup: {len(eva_to_triple)} entries")

    char_lm = _build_char_lm()
    print(f"  Char LM built (order={char_lm.get('order', '?')})")

    word_set = _build_10k_word_set()
    print(f"  10K word set: {len(word_set)} words")

    # ------------------------------------------------------------------
    # 3. Subsample
    # ------------------------------------------------------------------
    print("\n--- Step 3: Subsampling to 5000 tokens ---")

    sub_evas, sub_folios, sub_indices = _subsample_tokens(
        token_evas_all, token_folios_all, n=5000, seed=42,
    )
    n_tokens = len(sub_evas)
    print(f"  Subsampled: {n_tokens} tokens")

    # ------------------------------------------------------------------
    # 4. Run all 5 subtests
    # ------------------------------------------------------------------
    print("\n--- Step 4: Running 5 subtests ---")

    c1 = _run_wrong_language_lm(sub_evas, assignment, eva_to_triple, word_set)
    c2 = _run_length_matched_random(sub_evas, assignment, eva_to_triple, char_lm, word_set)
    c3 = _run_section_specific(sub_evas, sub_folios, assignment, eva_to_triple, char_lm, word_set)
    c4 = _run_cross_validated_lm(sub_evas, assignment, eva_to_triple, word_set)
    c5 = _run_bigram_attribution(
        sub_evas, sub_folios, assignment, eva_to_triple, char_lm, word_set,
    )

    # ------------------------------------------------------------------
    # 5. Tally passes
    # ------------------------------------------------------------------
    print("\n--- Step 5: Verdict ---")

    passes: List[Tuple[str, bool]] = []

    # C1 passes if verdict is LATIN_CONFIRMED or LATIN_MARGINAL
    c1_pass = c1['verdict'] in ('LATIN_CONFIRMED', 'LATIN_MARGINAL')
    passes.append(('C1_wrong_language', c1_pass))

    # C2 passes if real_rate > length_matched_mean + 2*std
    c2_pass = c2['exceeds_2sigma']
    passes.append(('C2_length_matched', c2_pass))

    # C3 passes if verdict is CONTENT_VARIATION
    c3_pass = c3['verdict'] == 'CONTENT_VARIATION'
    passes.append(('C3_section_specific', c3_pass))

    # C4 passes if verdict is SIGNAL_TRANSFERS or PARTIAL_TRANSFER
    c4_pass = c4['verdict'] in ('SIGNAL_TRANSFERS', 'PARTIAL_TRANSFER')
    passes.append(('C4_cross_validated', c4_pass))

    # C5 passes if DIRECT > 0 (any direct CC bigrams)
    c5_pass = c5['direct_bigram_hits'] > 0
    passes.append(('C5_bigram_attribution', c5_pass))

    n_passed = sum(1 for _, p in passes if p)
    n_total = len(passes)

    for name, passed in passes:
        marker = 'PASS' if passed else 'FAIL'
        print(f"  {name}: {marker}")

    verdict = f"NULL_BATTERY_{n_passed}/{n_total}"
    print(f"\n  Verdict: {verdict}")

    runtime = time.time() - t0
    print(f"  Runtime: {runtime:.1f}s")

    # ------------------------------------------------------------------
    # 6. Save results
    # ------------------------------------------------------------------
    result = NullBattery50Result(
        c1_wrong_language=c1,
        c2_length_matched=c2,
        c3_section_specific=c3,
        c4_cross_validated=c4,
        c5_bigram_attribution=c5,
        tests_passed=n_passed,
        tests_total=n_total,
        verdict=verdict,
        runtime_seconds=round(runtime, 2),
    )

    out_path = _save_json(rd, 'null_battery_50.json', asdict(result))
    print(f"\n  Saved: {out_path}")

    print("\n" + "=" * 70)
    print("Phase 50 Track C complete")
    print(f"  Tests passed: {n_passed}/{n_total}")
    print(f"  Verdict:      {verdict}")
    print("=" * 70)

    return asdict(result)
