"""
Step 42.2 – Symmetric Z-Score Recomputation
=============================================
For every bigram z-score in the project, recompute with guaranteed
symmetric methodology: both real and null count exact AND relaxed
(edit-distance-1) bigram hits.

Uses the same fix pattern from Phase 41 (venetian_validated.py):
precompute edit-distance-1 partner sets, then count both exact and
relaxed in every null permutation.

Dependency chain:
    bigram_code_audit.json          (Step 42.1 — audit classifications)
    signal_bigrams.json             (Phase 29 signal data)
    combined_bigrams.json           (Phase 35 signal data)
    signal_10k.json                 (Phase 36 signal data)
    merged_signal.json              (Phase 38 signal data)
    corrected_signal.json           (Phase 39.4 signal data)
    amplified_signal.json           (Phase 39.16 signal data)
    venetian_validated.json         (Phase 41 — already recomputed)
    merged_dict.json                (reference bigrams for Phases 38+)
    amplified_dict.json             (calibrated dictionary for 39.16)
        → symmetric_recompute.json  (this step)
"""

import json
import os
import random
import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from voynich.core._paths import results_dir as _results_dir


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────
# Bigram matching (reused from venetian_validated.py)
# ─────────────────────────────────────────────────────────────────

def _edit_distance_1(a: str, b: str) -> bool:
    """Check if two words are within edit distance 1."""
    if abs(len(a) - len(b)) > 1:
        return False
    if a == b:
        return True
    if len(a) == len(b):
        return sum(1 for x, y in zip(a, b) if x != y) == 1
    longer, shorter = (a, b) if len(a) > len(b) else (b, a)
    diffs = 0
    i = j = 0
    while i < len(longer) and j < len(shorter):
        if longer[i] != shorter[j]:
            diffs += 1
            i += 1
        else:
            i += 1
            j += 1
    return diffs + (len(longer) - i) <= 1


def _build_word_index(
    reference_bigrams: Set[Tuple[str, str]],
) -> Dict[str, Set[str]]:
    """Build word→partner index for fast bigram lookup."""
    index: Dict[str, Set[str]] = {}
    for w1, w2 in reference_bigrams:
        if w1 not in index:
            index[w1] = set()
        index[w1].add(w2)
    return index


def _precompute_partner_sets(
    unique_signal_words: Set[str],
    ref_words: Set[str],
) -> Dict[str, Set[str]]:
    """For each signal word, find all reference words within edit distance 1."""
    partners: Dict[str, Set[str]] = {}
    for sw in unique_signal_words:
        if not sw:
            continue
        p = set()
        for rw in ref_words:
            if abs(len(rw) - len(sw)) <= 1 and _edit_distance_1(sw, rw):
                p.add(rw)
        partners[sw] = p
    return partners


def _check_relaxed_pair(
    w1: str,
    w2: str,
    ref_bigrams: Set[Tuple[str, str]],
    word_index: Dict[str, Set[str]],
    partners: Dict[str, Set[str]],
) -> bool:
    """Check if (w1, w2) matches any reference bigram within edit distance 1.

    Does NOT check exact (caller should check exact first).
    """
    w1_partners = partners.get(w1, set())
    w2_partners = partners.get(w2, set())

    # (w1, p2) where p2 is within ed1 of w2
    if w1 in word_index:
        for rp in word_index[w1]:
            if rp in w2_partners:
                return True

    # (p1, w2) or (p1, p2) where p1 is within ed1 of w1
    for p1 in w1_partners:
        if p1 in word_index:
            for rp in word_index[p1]:
                if rp == w2 or rp in w2_partners:
                    return True

    return False


# ─────────────────────────────────────────────────────────────────
# Signal pair extraction
# ─────────────────────────────────────────────────────────────────

def _extract_signal_pairs(
    classifications: List[str],
    decoded: List[str],
    folios: List[str],
) -> Tuple[List[Dict], List[str]]:
    """Extract consecutive SIGNAL-SIGNAL pairs respecting folio boundaries.

    Returns:
        pairs: list of {'w1': str, 'w2': str, 'folio': str, 'pos': int}
        all_signal_words: list of decoded words at SIGNAL positions
    """
    pairs = []
    all_signal_words = []

    for i in range(len(classifications)):
        if classifications[i] == 'SIGNAL':
            all_signal_words.append(decoded[i])

    for i in range(len(classifications) - 1):
        if (classifications[i] == 'SIGNAL'
                and classifications[i + 1] == 'SIGNAL'
                and folios[i] == folios[i + 1]):
            pairs.append({
                'w1': decoded[i],
                'w2': decoded[i + 1],
                'folio': folios[i],
                'pos': i,
            })

    return pairs, all_signal_words


# ─────────────────────────────────────────────────────────────────
# Canonical symmetric z-test
# ─────────────────────────────────────────────────────────────────

def _z_stat(real_val: float, null_vals: List[float]) -> Tuple[float, float, float]:
    """Compute z-score from real value and null distribution."""
    if not null_vals:
        return 0.0, 0.0, 0.001
    n_mean = sum(null_vals) / len(null_vals)
    n_std = (sum((v - n_mean) ** 2 for v in null_vals)
             / len(null_vals)) ** 0.5
    z = (real_val - n_mean) / n_std if n_std > 0.001 else 0.0
    return z, n_mean, n_std


def _symmetric_bigram_z(
    signal_pairs: List[Dict],
    all_signal_words: List[str],
    reference_bigrams: Set[Tuple[str, str]],
    phase_label: str,
    n_permutations: int = 500,
) -> Dict:
    """Canonical symmetric bigram z-test.

    Both real and null count exact and relaxed (edit-distance-1) hits.
    Uses shuffle null model: randomly permute signal word list, count
    consecutive pairs matching reference bigrams.
    """
    # Build indexes
    word_index = _build_word_index(reference_bigrams)
    ref_words: Set[str] = set()
    for w1, w2 in reference_bigrams:
        ref_words.add(w1)
        ref_words.add(w2)

    # Precompute partner sets for all unique signal words
    unique_signal = set(w for w in all_signal_words if w)
    print(f"      [{phase_label}] Precomputing partners for "
          f"{len(unique_signal)} signal words against "
          f"{len(ref_words)} ref words …")
    t_partner = time.time()
    partners = _precompute_partner_sets(unique_signal, ref_words)
    partner_time = time.time() - t_partner
    mean_partners = (sum(len(v) for v in partners.values())
                     / max(len(partners), 1))
    print(f"      [{phase_label}] Partners done ({partner_time:.1f}s). "
          f"Mean partners/word: {mean_partners:.1f}")

    # Count real hits
    real_exact = 0
    real_relaxed = 0
    for pair in signal_pairs:
        w1, w2 = pair['w1'], pair['w2']
        if (w1, w2) in reference_bigrams:
            real_exact += 1
        elif _check_relaxed_pair(w1, w2, reference_bigrams,
                                 word_index, partners):
            real_relaxed += 1

    real_total = real_exact + real_relaxed
    print(f"      [{phase_label}] Real: exact={real_exact}, "
          f"relaxed={real_relaxed}, total={real_total}")

    # Null permutation test — SYMMETRIC: counts both exact AND relaxed
    rng = random.Random(42)
    null_exact_list: List[int] = []
    null_relaxed_list: List[int] = []
    null_total_list: List[int] = []

    n_pairs = len(signal_pairs)
    for perm_i in range(n_permutations):
        if (perm_i + 1) % 100 == 0:
            print(f"      [{phase_label}] Permutation "
                  f"{perm_i + 1}/{n_permutations} …")

        shuffled = list(all_signal_words)
        rng.shuffle(shuffled)

        perm_exact = 0
        perm_relaxed = 0
        for k in range(min(n_pairs, len(shuffled) - 1)):
            w1 = shuffled[k]
            w2 = shuffled[k + 1]
            if not w1 or not w2:
                continue
            if (w1, w2) in reference_bigrams:
                perm_exact += 1
            elif _check_relaxed_pair(w1, w2, reference_bigrams,
                                     word_index, partners):
                perm_relaxed += 1

        null_exact_list.append(perm_exact)
        null_relaxed_list.append(perm_relaxed)
        null_total_list.append(perm_exact + perm_relaxed)

    # Compute z-scores
    z_exact, null_exact_mean, null_exact_std = _z_stat(
        real_exact, null_exact_list)
    z_relaxed, null_relaxed_mean, null_relaxed_std = _z_stat(
        real_relaxed, null_relaxed_list)
    z_total, null_total_mean, null_total_std = _z_stat(
        real_total, null_total_list)

    print(f"      [{phase_label}] z_exact={z_exact:.4f}, "
          f"z_relaxed={z_relaxed:.4f}, z_total={z_total:.4f}")

    return {
        'phase': phase_label,
        'n_signal_pairs': n_pairs,
        'n_signal_words': len(all_signal_words),
        'n_unique_signal_words': len(unique_signal),
        'n_ref_bigrams': len(reference_bigrams),
        'n_ref_words': len(ref_words),
        'real_exact': real_exact,
        'real_relaxed': real_relaxed,
        'real_total': real_total,
        'null_exact_mean': round(null_exact_mean, 4),
        'null_exact_std': round(null_exact_std, 4),
        'null_relaxed_mean': round(null_relaxed_mean, 4),
        'null_relaxed_std': round(null_relaxed_std, 4),
        'null_total_mean': round(null_total_mean, 4),
        'null_total_std': round(null_total_std, 4),
        'z_exact': round(z_exact, 4),
        'z_relaxed': round(z_relaxed, 4),
        'z_total': round(z_total, 4),
        'n_permutations': n_permutations,
        'mean_partners_per_word': round(mean_partners, 2),
        'partner_precompute_seconds': round(partner_time, 1),
    }


# ─────────────────────────────────────────────────────────────────
# Reference bigram builders
# ─────────────────────────────────────────────────────────────────

def _build_latin_ref_bigrams() -> Set[Tuple[str, str]]:
    """Build reference bigrams from Latin corpus (same as Phase 29)."""
    from voynich.core.reference import load_reference_corpus

    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    ref_tokens = [w.lower() for w in ref_corpus.get_combined_tokens('latin')
                  if len(w) >= 2]

    bigrams: Set[Tuple[str, str]] = set()
    for i in range(len(ref_tokens) - 1):
        bigrams.add((ref_tokens[i], ref_tokens[i + 1]))

    return bigrams


def _build_10k_ref_bigrams(rd: str) -> Set[Tuple[str, str]]:
    """Build reference bigrams filtered to 10K Latin vocabulary.

    Phase 36 used the full Latin bigram set but filtered to pairs where
    both words are in the 10K vocabulary.
    """
    from voynich.core.reference import load_reference_corpus

    # Load 10K word list
    merged_dict = _safe_load(os.path.join(rd, 'merged_dict.json'))
    latin_10k = set(merged_dict.get('latin_10k_words', []))

    if not latin_10k:
        print("      WARNING: latin_10k_words not found in merged_dict.json")
        return set()

    # Build bigrams from Latin reference, filtered to 10K vocab
    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    ref_tokens = [w.lower() for w in ref_corpus.get_combined_tokens('latin')
                  if len(w) >= 2]

    bigrams: Set[Tuple[str, str]] = set()
    for i in range(len(ref_tokens) - 1):
        w1, w2 = ref_tokens[i], ref_tokens[i + 1]
        if w1 in latin_10k and w2 in latin_10k:
            bigrams.add((w1, w2))

    return bigrams


def _build_merged_ref_bigrams(rd: str) -> Set[Tuple[str, str]]:
    """Load merged (Latin+Italian) reference bigrams from merged_dict.json."""
    merged_dict = _safe_load(os.path.join(rd, 'merged_dict.json'))
    bigram_list = merged_dict.get('bigram_list', [])

    bigrams: Set[Tuple[str, str]] = set()
    for pair in bigram_list:
        if len(pair) >= 2:
            bigrams.add((pair[0], pair[1]))

    return bigrams


def _build_calibrated_ref_bigrams(rd: str) -> Set[Tuple[str, str]]:
    """Build reference bigrams filtered to calibrated vocabulary.

    Phase 39.16 used the merged bigram set filtered to pairs where both
    words are in the calibrated 1K vocabulary.
    """
    amp_dict = _safe_load(os.path.join(rd, 'amplified_dict.json'))
    cal_words = set(amp_dict.get('calibrated_words', []))

    if not cal_words:
        print("      WARNING: calibrated_words not found")
        return set()

    # Load merged bigrams and filter
    merged_dict = _safe_load(os.path.join(rd, 'merged_dict.json'))
    bigram_list = merged_dict.get('bigram_list', [])

    bigrams: Set[Tuple[str, str]] = set()
    for pair in bigram_list:
        if len(pair) >= 2 and pair[0] in cal_words and pair[1] in cal_words:
            bigrams.add((pair[0], pair[1]))

    return bigrams


# ─────────────────────────────────────────────────────────────────
# Per-phase recomputation
# ─────────────────────────────────────────────────────────────────

def _recompute_phase_29(rd: str) -> Dict:
    """Phase 29: signal_bigrams.py, Latin 131K, z=6.14."""
    data = _safe_load(os.path.join(rd, 'signal_bigrams.json'))
    if not data:
        return {'phase': '29', 'error': 'signal_bigrams.json not found'}

    decoded = data['token_decoded']
    classifications = data['token_classifications']
    folios = data['token_folios']

    ref_bigrams = _build_latin_ref_bigrams()
    print(f"      [Phase 29] {len(ref_bigrams)} Latin reference bigrams")

    pairs, signal_words = _extract_signal_pairs(
        classifications, decoded, folios)
    print(f"      [Phase 29] {len(pairs)} SIGNAL pairs, "
          f"{len(signal_words)} SIGNAL words")

    result = _symmetric_bigram_z(
        pairs, signal_words, ref_bigrams, '29', n_permutations=500)
    result['original_z'] = data.get('bigram_z_score', None)
    result['dictionary'] = 'Latin 131K'
    return result


def _recompute_phase_35(rd: str) -> Dict:
    """Phase 35: combined_bigrams.py, Latin 131K, z=6.88."""
    data = _safe_load(os.path.join(rd, 'combined_bigrams.json'))
    if not data:
        return {'phase': '35', 'error': 'combined_bigrams.json not found'}

    decoded = data.get('token_decoded', [])
    classifications = data.get('token_classifications', [])
    folios = data.get('token_folios', [])

    if not decoded:
        return {'phase': '35', 'error': 'No token data in combined_bigrams.json'}

    ref_bigrams = _build_latin_ref_bigrams()
    print(f"      [Phase 35] {len(ref_bigrams)} Latin reference bigrams")

    pairs, signal_words = _extract_signal_pairs(
        classifications, decoded, folios)
    print(f"      [Phase 35] {len(pairs)} SIGNAL pairs, "
          f"{len(signal_words)} SIGNAL words")

    result = _symmetric_bigram_z(
        pairs, signal_words, ref_bigrams, '35', n_permutations=500)
    result['original_z'] = data.get('bigram_z_score', None)
    result['dictionary'] = 'Latin 131K'
    return result


def _recompute_phase_36(rd: str) -> Dict:
    """Phase 36: bigrams_10k.py, Latin 10K, z=12.66."""
    data_10k = _safe_load(os.path.join(rd, 'signal_10k.json'))
    data_bg = _safe_load(os.path.join(rd, 'bigrams_10k.json'))
    if not data_10k:
        return {'phase': '36', 'error': 'signal_10k.json not found'}

    decoded = data_10k['token_decoded']
    classifications = data_10k['token_classifications']
    folios = data_10k['token_folios']

    ref_bigrams = _build_10k_ref_bigrams(rd)
    print(f"      [Phase 36] {len(ref_bigrams)} 10K-filtered reference bigrams")

    pairs, signal_words = _extract_signal_pairs(
        classifications, decoded, folios)
    print(f"      [Phase 36] {len(pairs)} SIGNAL pairs, "
          f"{len(signal_words)} SIGNAL words")

    result = _symmetric_bigram_z(
        pairs, signal_words, ref_bigrams, '36', n_permutations=500)
    result['original_z'] = data_bg.get('bigram_z', None)
    result['dictionary'] = 'Latin 10K'
    return result


def _recompute_phase_37_6(rd: str) -> Dict:
    """Phase 37.6: concat_bigrams.py, exact-only, z=-6.67.

    This phase used a concatenated token stream and shuffle null.
    Already symmetric (exact-only for both). Include for completeness
    but note: concat_signal.json lacks token_decoded/folios, so we
    cannot recompute with the full methodology. Report as VALIDATED
    based on code audit (both sides used exact-only).
    """
    data = _safe_load(os.path.join(rd, 'concat_bigrams.json'))

    return {
        'phase': '37.6',
        'original_z': data.get('merged_bigram_z', None),
        'dictionary': 'Latin 17K',
        'note': (
            'Already symmetric (exact-only for both real and null). '
            'concat_signal.json lacks full token arrays needed for '
            'recomputation. Validated by code audit only.'
        ),
        'z_exact': data.get('merged_bigram_z', None),
        'z_relaxed': None,
        'z_total': None,
        'code_audit_status': 'VALID',
        'recomputed': False,
    }


def _recompute_phase_38(rd: str) -> Dict:
    """Phase 38: merged_bigrams.py, merged L+I, z=14.37."""
    data_sig = _safe_load(os.path.join(rd, 'merged_signal.json'))
    data_bg = _safe_load(os.path.join(rd, 'merged_bigrams.json'))
    if not data_sig:
        return {'phase': '38', 'error': 'merged_signal.json not found'}

    decoded = data_sig['token_decoded']
    classifications = data_sig['token_classifications']
    folios = data_sig['token_folios']

    ref_bigrams = _build_merged_ref_bigrams(rd)
    print(f"      [Phase 38] {len(ref_bigrams)} merged reference bigrams")

    pairs, signal_words = _extract_signal_pairs(
        classifications, decoded, folios)
    print(f"      [Phase 38] {len(pairs)} SIGNAL pairs, "
          f"{len(signal_words)} SIGNAL words")

    result = _symmetric_bigram_z(
        pairs, signal_words, ref_bigrams, '38', n_permutations=500)
    result['original_z'] = data_bg.get('bigram_z', None)
    result['dictionary'] = 'Merged L+I 19K'
    return result


def _recompute_phase_39_4(rd: str) -> Dict:
    """Phase 39.4: corrected_signal.py, merged, z=11.53."""
    data = _safe_load(os.path.join(rd, 'corrected_signal.json'))
    if not data:
        return {'phase': '39.4', 'error': 'corrected_signal.json not found'}

    decoded = data.get('token_decoded', [])
    classifications = data.get('token_classifications', [])

    if not decoded:
        return {'phase': '39.4', 'error': 'No token data'}

    # corrected_signal.json may lack folios — get from signal_bigrams
    folios = data.get('token_folios', [])
    if not folios:
        sb = _safe_load(os.path.join(rd, 'signal_bigrams.json'))
        folios = sb.get('token_folios', [])

    if len(folios) != len(decoded):
        return {'phase': '39.4',
                'error': f'Folio/decoded length mismatch: {len(folios)} vs {len(decoded)}'}

    ref_bigrams = _build_merged_ref_bigrams(rd)
    print(f"      [Phase 39.4] {len(ref_bigrams)} merged reference bigrams")

    pairs, signal_words = _extract_signal_pairs(
        classifications, decoded, folios)
    print(f"      [Phase 39.4] {len(pairs)} SIGNAL pairs, "
          f"{len(signal_words)} SIGNAL words")

    result = _symmetric_bigram_z(
        pairs, signal_words, ref_bigrams, '39.4', n_permutations=500)
    result['original_z'] = data.get('bigram_z', None)
    result['dictionary'] = 'Merged L+I 19K'
    return result


def _recompute_phase_39_16(rd: str) -> Dict:
    """Phase 39.16: amplified_bigrams.py, calibrated 1K, z=19.89."""
    data_sig = _safe_load(os.path.join(rd, 'amplified_signal.json'))
    data_bg = _safe_load(os.path.join(rd, 'amplified_bigrams.json'))
    if not data_sig:
        return {'phase': '39.16', 'error': 'amplified_signal.json not found'}

    decoded = data_sig['token_decoded']
    classifications = data_sig['token_classifications']
    folios = data_sig['token_folios']

    ref_bigrams = _build_calibrated_ref_bigrams(rd)
    print(f"      [Phase 39.16] {len(ref_bigrams)} calibrated reference bigrams")

    pairs, signal_words = _extract_signal_pairs(
        classifications, decoded, folios)
    print(f"      [Phase 39.16] {len(pairs)} SIGNAL pairs, "
          f"{len(signal_words)} SIGNAL words")

    result = _symmetric_bigram_z(
        pairs, signal_words, ref_bigrams, '39.16', n_permutations=500)
    result['original_z'] = data_bg.get('bigram_z', None)
    result['dictionary'] = 'Calibrated 1K'
    return result


def _load_phase_40(rd: str) -> Dict:
    """Phase 40: Already recomputed in Phase 41. Just load results."""
    ven = _safe_load(os.path.join(rd, 'venetian_validated.json'))
    orig = _safe_load(os.path.join(rd, 'venetian_bigrams.json'))

    if not ven:
        return {'phase': '40', 'error': 'venetian_validated.json not found'}

    return {
        'phase': '40',
        'original_z': orig.get('bigram_z', 319.76),
        'dictionary': 'Venetian 29K',
        'z_exact': ven.get('z_exact', None),
        'z_relaxed': ven.get('z_relaxed', None),
        'z_total': ven.get('z_total', None),
        'real_exact': ven.get('real_exact', None),
        'real_relaxed': ven.get('real_relaxed', None),
        'real_total': ven.get('real_total', None),
        'null_total_mean': ven.get('null_total_mean', None),
        'null_total_std': ven.get('null_total_std', None),
        'n_permutations': ven.get('n_permutations', 500),
        'n_signal_pairs': ven.get('n_signal_pairs', None),
        'note': 'Already recomputed in Phase 41 (venetian_validated.json)',
        'recomputed': True,
        'recomputed_by': 'Phase 41',
    }


# ─────────────────────────────────────────────────────────────────
# Classification
# ─────────────────────────────────────────────────────────────────

def _classify_result(original_z: Optional[float],
                     symmetric_z: Optional[float]) -> str:
    """Classify the change from original to symmetric z-score.

    CONFIRMED:    symmetric z within 20% of original (or both >2 and same sign)
    DEFLATED:     symmetric z significantly lower but still >2.0
    INFLATED:     symmetric z drops below 2.0
    INVALIDATED:  symmetric z ≤0 or original was clearly wrong
    UNCHANGED:    exact same value (code audit only, no recomputation)
    """
    if original_z is None or symmetric_z is None:
        return 'UNKNOWN'

    # Handle negative original z (Phase 37.6)
    if original_z < 0 and symmetric_z < 0:
        return 'CONFIRMED'

    if original_z < 0 or symmetric_z < 0:
        if symmetric_z <= 0:
            return 'INVALIDATED'
        return 'DEFLATED'

    if original_z == 0:
        return 'CONFIRMED' if symmetric_z == 0 else 'UNKNOWN'

    ratio = symmetric_z / original_z if original_z != 0 else 0

    if ratio >= 0.8:
        return 'CONFIRMED'
    elif symmetric_z > 2.0:
        return 'DEFLATED'
    elif symmetric_z <= 0:
        return 'INVALIDATED'
    else:
        return 'INFLATED'


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def run_symmetric_recompute() -> None:
    """Step 42.2: Symmetric recomputation of all bigram z-scores."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 42.2: Symmetric Z-Score Recomputation")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load audit results ──
    print("\n  1. Loading audit results …")
    audit = _safe_load(os.path.join(rd, 'bigram_code_audit.json'))
    if not audit:
        print("    WARNING: bigram_code_audit.json not found. "
              "Running all recomputations anyway.")

    # ── 2. Recompute each phase ──
    print("\n  2. Recomputing z-scores with symmetric methodology …")

    results: List[Dict] = []

    phase_runners = [
        ('29', _recompute_phase_29),
        ('35', _recompute_phase_35),
        ('36', _recompute_phase_36),
        ('37.6', _recompute_phase_37_6),
        ('38', _recompute_phase_38),
        ('39.4', _recompute_phase_39_4),
        ('39.16', _recompute_phase_39_16),
        ('40', _load_phase_40),
    ]

    for label, runner in phase_runners:
        print(f"\n    ── Phase {label} ──")
        t_phase = time.time()
        try:
            result = runner(rd)
            result['runtime_seconds'] = round(time.time() - t_phase, 1)
            results.append(result)

            if 'error' in result:
                print(f"      ERROR: {result['error']}")
            else:
                orig = result.get('original_z', '?')
                z_ex = result.get('z_exact', '?')
                z_tot = result.get('z_total', '?')
                print(f"      Original z: {orig}")
                print(f"      Symmetric z_exact: {z_ex}")
                print(f"      Symmetric z_total: {z_tot}")

        except Exception as e:
            print(f"      ERROR: {e}")
            results.append({
                'phase': label,
                'error': str(e),
                'runtime_seconds': round(time.time() - t_phase, 1),
            })

    # ── 3. Build summary table ──
    print("\n  3. Summary comparison table")
    print(f"    {'Phase':<10s} {'Dict':<15s} {'Orig z':>10s} {'Sym z_ex':>10s} "
          f"{'Sym z_tot':>10s} {'Status':<15s}")
    print("    " + "-" * 75)

    summary_table: List[Dict] = []
    for r in results:
        phase = r.get('phase', '?')
        dictionary = r.get('dictionary', '?')
        orig_z = r.get('original_z')
        z_exact = r.get('z_exact')
        z_total = r.get('z_total')

        # Use z_total for classification — exact matches are too rare
        # (5-17 hits) to produce meaningful z_exact.  The signal is in
        # the relaxed (edit-distance-1) matches, so z_total is the
        # appropriate significance indicator.
        primary_z = z_total if z_total is not None else z_exact
        classification = _classify_result(orig_z, primary_z)

        row = {
            'phase': phase,
            'dictionary': dictionary,
            'original_z': orig_z,
            'symmetric_z_exact': z_exact,
            'symmetric_z_total': z_total,
            'classification': classification,
        }
        summary_table.append(row)

        orig_str = f"{orig_z:.2f}" if orig_z is not None else "N/A"
        zex_str = f"{z_exact:.4f}" if z_exact is not None else "N/A"
        ztot_str = f"{z_total:.4f}" if z_total is not None else "N/A"
        print(f"    {phase:<10s} {dictionary:<15s} {orig_str:>10s} "
              f"{zex_str:>10s} {ztot_str:>10s} {classification:<15s}")

    # ── 4. Overall assessment ──
    print("\n  4. Overall assessment")

    n_confirmed = sum(1 for s in summary_table
                      if s['classification'] == 'CONFIRMED')
    n_deflated = sum(1 for s in summary_table
                     if s['classification'] == 'DEFLATED')
    n_inflated = sum(1 for s in summary_table
                     if s['classification'] == 'INFLATED')
    n_invalidated = sum(1 for s in summary_table
                        if s['classification'] == 'INVALIDATED')

    print(f"    CONFIRMED:   {n_confirmed}")
    print(f"    DEFLATED:    {n_deflated}")
    print(f"    INFLATED:    {n_inflated}")
    print(f"    INVALIDATED: {n_invalidated}")

    # Best surviving z (use z_total — exact matches are too rare)
    valid_z_total = [
        (s['symmetric_z_total'], s['phase'])
        for s in summary_table
        if s['symmetric_z_total'] is not None
        and s['symmetric_z_total'] > 0
        and s['classification'] in ('CONFIRMED', 'DEFLATED')
    ]
    if valid_z_total:
        best_z, best_phase = max(valid_z_total, key=lambda x: x[0])
    else:
        # Fall back: best z_total from any recomputed phase
        all_z_total = [
            (s['symmetric_z_total'], s['phase'])
            for s in summary_table
            if s['symmetric_z_total'] is not None
            and s['symmetric_z_total'] > 0
        ]
        if all_z_total:
            best_z, best_phase = max(all_z_total, key=lambda x: x[0])
        else:
            best_z, best_phase = 0.0, '?'

    print(f"\n    Best surviving z_total: {best_z:.4f} "
          f"(Phase {best_phase})")

    if best_z >= 5.0:
        verdict = 'STRONG_BIGRAM_SIGNAL'
    elif best_z >= 3.0:
        verdict = 'MODERATE_BIGRAM_SIGNAL'
    elif best_z >= 2.0:
        verdict = 'WEAK_BIGRAM_SIGNAL'
    else:
        verdict = 'NO_SIGNIFICANT_BIGRAM_SIGNAL'

    print(f"    VERDICT: {verdict}")

    # ── 5. Save ──
    elapsed = time.time() - t0

    output = {
        'recomputed': [_convert(r) for r in results],
        'summary_table': summary_table,
        'n_phases_recomputed': len(results),
        'n_confirmed': n_confirmed,
        'n_deflated': n_deflated,
        'n_inflated': n_inflated,
        'n_invalidated': n_invalidated,
        'best_surviving_z_total': round(best_z, 4),
        'best_surviving_z_exact': round(
            max((s['symmetric_z_exact'] for s in summary_table
                 if s.get('symmetric_z_exact') is not None
                 and s['symmetric_z_exact'] > 0), default=0.0), 4),
        'best_surviving_phase': best_phase,
        'verdict': verdict,
        'runtime_seconds': round(elapsed, 1),
    }

    out_path = os.path.join(rd, 'symmetric_recompute.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(output), f, indent=2)
    print(f"\n  Saved → {out_path} ({elapsed:.1f}s)")
