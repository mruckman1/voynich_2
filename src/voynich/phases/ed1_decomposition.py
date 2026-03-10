"""
Step 39.1 – ED1 Decomposition of CC Bigrams
=============================================
For each of the 31 content-content bigrams at edit distance 1, identify
the specific reference bigram matched, the character error, and the
EVA character / triple responsible.  Also reconstruct all 91 medical
phrases (merged_context.json only stored 30).

Dependency chain:
    merged_bigrams.json        (Step 38.4)
    merged_dict.json           (Step 38.1)
    merged_signal.json         (Step 38.3)
    decode_10k.json            (Step 36.1)
    modifier_integrate.json    (Step 16)
        → ed1_decomposition.json   (this step)
"""

import json
import os
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    load_corpus,
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
# Edit distance helpers
# ---------------------------------------------------------------------------

def _edit_distance_1(w1: str, w2: str) -> bool:
    """Check if edit distance <= 1."""
    if abs(len(w1) - len(w2)) > 1:
        return False
    if w1 == w2:
        return True
    if len(w1) == len(w2):
        return sum(a != b for a, b in zip(w1, w2)) <= 1
    short, long_ = (w1, w2) if len(w1) < len(w2) else (w2, w1)
    i = j = diffs = 0
    while i < len(short) and j < len(long_):
        if short[i] != long_[j]:
            diffs += 1
            if diffs > 1:
                return False
            j += 1
        else:
            i += 1
            j += 1
    return True


def _compute_edit_operation(decoded: str, reference: str) -> Optional[Dict]:
    """Compute the specific edit operation between decoded and reference.

    Returns None if distance > 1.
    """
    if decoded == reference:
        return None  # exact match, no error

    if len(decoded) == len(reference):
        # Substitution
        diffs = [(i, decoded[i], reference[i])
                 for i in range(len(decoded)) if decoded[i] != reference[i]]
        if len(diffs) == 1:
            pos, dec_ch, ref_ch = diffs[0]
            return {
                'error_type': 'SUBSTITUTION',
                'error_position': pos,
                'decoded_char': dec_ch,
                'reference_char': ref_ch,
            }
        return None

    if len(decoded) == len(reference) + 1:
        # Decoded has an extra char (insertion in decoded = deletion needed)
        for skip in range(len(decoded)):
            trimmed = decoded[:skip] + decoded[skip + 1:]
            if trimmed == reference:
                return {
                    'error_type': 'INSERTION',
                    'error_position': skip,
                    'decoded_char': decoded[skip],
                    'reference_char': '',
                }
        return None

    if len(decoded) + 1 == len(reference):
        # Decoded missing a char (deletion in decoded = insertion needed)
        for skip in range(len(reference)):
            trimmed = reference[:skip] + reference[skip + 1:]
            if trimmed == decoded:
                return {
                    'error_type': 'DELETION',
                    'error_position': skip,
                    'decoded_char': '',
                    'reference_char': reference[skip],
                }
        return None

    return None


def _is_vowel_error(edit_op: Dict) -> bool:
    """Check if the edit operation is a vowel substitution."""
    if edit_op['error_type'] != 'SUBSTITUTION':
        return False
    vowels = set('aeiou')
    return edit_op['decoded_char'] in vowels and edit_op['reference_char'] in vowels


# ---------------------------------------------------------------------------
# Reference bigram matching
# ---------------------------------------------------------------------------

def _find_reference_matches(
    w1: str,
    w2: str,
    bigram_set: Set[Tuple[str, str]],
    all_ref_words: Set[str],
) -> List[Dict]:
    """Find all reference bigrams within ED1 of (w1, w2)."""
    # Build candidate reference words for w1 and w2
    w1_candidates = {rw for rw in all_ref_words if _edit_distance_1(w1, rw)}
    w2_candidates = {rw for rw in all_ref_words if _edit_distance_1(w2, rw)}

    matches = []
    for rw1 in w1_candidates:
        for rw2 in w2_candidates:
            if (rw1, rw2) in bigram_set:
                op1 = _compute_edit_operation(w1, rw1)
                op2 = _compute_edit_operation(w2, rw2)
                matches.append({
                    'ref_w1': rw1,
                    'ref_w2': rw2,
                    'w1_exact': w1 == rw1,
                    'w2_exact': w2 == rw2,
                    'w1_edit': op1,
                    'w2_edit': op2,
                    'w1_vowel_error': _is_vowel_error(op1) if op1 else False,
                    'w2_vowel_error': _is_vowel_error(op2) if op2 else False,
                })
    return matches


# ---------------------------------------------------------------------------
# Medical phrase reconstruction
# ---------------------------------------------------------------------------

def _reconstruct_medical_phrases(
    decoded_lower: List[str],
    token_folios: List[str],
    classifications: List[str],
    merged_dict: Set[str],
    latin_10k: Set[str],
    italian_10k: Set[str],
    signal_words: Set[str],
) -> List[Dict]:
    """Reconstruct all medical phrases (merged_context stored only 30)."""
    # Build chains (same logic as merged_context._build_chains_merged)
    n = len(decoded_lower)
    chains = []
    i = 0
    while i < n:
        if decoded_lower[i] in merged_dict:
            folio = token_folios[i] if i < len(token_folios) else 'unknown'
            chain = []
            j = i
            while (j < n and
                   decoded_lower[j] in merged_dict and
                   j < len(token_folios) and
                   token_folios[j] == folio):
                w = decoded_lower[j]
                src = ('SHARED' if w in latin_10k and w in italian_10k
                       else 'LATIN_ONLY' if w in latin_10k
                       else 'ITALIAN_ONLY')
                chain.append({
                    'word': w,
                    'source': src,
                    'is_signal': w in signal_words,
                    'position': j,
                })
                j += 1

            has_signal = any(e['is_signal'] for e in chain)
            if len(chain) >= 3 and has_signal:
                sources = Counter(e['source'] for e in chain)
                chains.append({
                    'folio': folio,
                    'start_pos': i,
                    'length': len(chain),
                    'words': [e['word'] for e in chain],
                    'sources': [e['source'] for e in chain],
                    'positions': [e['position'] for e in chain],
                    'n_signal': sum(1 for e in chain if e['is_signal']),
                    'language_mix': dict(sources),
                    'is_macaronic': len(sources) > 1 and 'ITALIAN_ONLY' in sources,
                })
            i = j
        else:
            i += 1

    chains.sort(key=lambda x: x['length'], reverse=True)

    # Medical phrase detection (same logic as merged_context._medical_collocations)
    pharma_verbs = {'cola', 'recipe', 'misce', 'coque', 'dice', 'cura',
                    'sana', 'bibe', 'beni'}
    body_parts = {'cora', 'core', 'corpo', 'carne', 'ossa', 'pede',
                  'manu', 'dente', 'naso'}
    ingredients = {'rosa', 'sale', 'vino', 'olio', 'bene', 'sene',
                   'calce', 'suco'}
    qualities = {'bela', 'bona', 'calida', 'frigida', 'sicca',
                 'dulce', 'rara', 'nova'}

    medical_phrases = []
    for chain in chains:
        words = set(chain['words'])
        has_verb = bool(words & pharma_verbs)
        has_body = bool(words & body_parts)
        has_ingredient = bool(words & ingredients)
        has_quality = bool(words & qualities)

        n_medical = sum([has_verb, has_body, has_ingredient, has_quality])
        if n_medical >= 2:
            medical_phrases.append({
                'folio': chain['folio'],
                'start_pos': chain['start_pos'],
                'words': chain['words'],
                'positions': chain['positions'],
                'sources': chain['sources'],
                'has_verb': has_verb,
                'has_body_part': has_body,
                'has_ingredient': has_ingredient,
                'has_quality': has_quality,
                'n_medical_types': n_medical,
                'is_macaronic': chain.get('is_macaronic', False),
            })

    medical_phrases.sort(key=lambda x: x['n_medical_types'], reverse=True)
    return medical_phrases


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_ed1_decomposition() -> None:
    """Step 39.1: ED1 Decomposition of CC Bigrams."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 39.1: ED1 Decomposition of CC Bigrams")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load inputs ──
    print("\n  1. Loading inputs …")
    bigram_data = _safe_load(os.path.join(rd, 'merged_bigrams.json'))
    dict_data = _safe_load(os.path.join(rd, 'merged_dict.json'))
    signal_data = _safe_load(os.path.join(rd, 'merged_signal.json'))
    decode_data = _safe_load(os.path.join(rd, 'decode_10k.json'))
    mod_data = _safe_load(os.path.join(rd, 'modifier_integrate.json'))

    catalog = bigram_data.get('bigram_catalog', [])
    bigram_list = dict_data.get('bigram_list', [])
    bigram_set = {(b[0], b[1]) for b in bigram_list}

    # Build set of all reference words that appear in bigrams
    all_ref_words: Set[str] = set()
    for b in bigram_list:
        all_ref_words.add(b[0])
        all_ref_words.add(b[1])

    latin_10k = set(dict_data.get('latin_10k_words', []))
    italian_10k = set(dict_data.get('italian_10k_words', []))
    merged_dict = set(dict_data.get('merged_words', []))

    token_decoded = signal_data.get('token_decoded', [])
    token_folios = signal_data.get('token_folios', [])
    classifications = signal_data.get('token_classifications', [])
    word_signals = signal_data.get('word_signals', [])
    signal_words = {w['word'] for w in word_signals if w.get('is_genuine_signal')}

    token_evas = decode_data.get('token_evas', [])

    # Reconstruct modifier chars
    modifier_set = set()
    for entry in mod_data.get('classifications', []):
        if isinstance(entry, dict) and entry.get('final_class') == 'modifier':
            modifier_set.add(entry.get('eva_char', ''))
        elif isinstance(entry, str):
            modifier_set.add(entry)
    # Fallback: use known modifier chars from Phase 16
    if not modifier_set:
        modifier_set = {'h', 'iin', 'b', 'ckh', 'i', 'iiin', 'u', 'aiin',
                        'al', 'ar', 'dy', 'ey', 'm', 'n', 'or'}

    print(f"     Catalog entries: {len(catalog)}")
    print(f"     Reference bigrams: {len(bigram_set)}")
    print(f"     Reference words: {len(all_ref_words)}")
    print(f"     Signal words: {len(signal_words)}")
    print(f"     Modifier chars: {len(modifier_set)}")

    # ── 2. Filter CC bigrams ──
    print("\n  2. Filtering content-content bigrams …")
    cc_entries = [e for e in catalog if e.get('content_type') == 'content-content']
    print(f"     CC bigram entries: {len(cc_entries)}")

    # Unique word pairs
    unique_pairs = set()
    for e in cc_entries:
        unique_pairs.add((e['w1'], e['w2']))
    print(f"     Unique CC word pairs: {len(unique_pairs)}")

    # ── 3. Find reference matches for each CC bigram ──
    print("\n  3. Finding reference matches for CC bigrams …")

    # Pre-filter: for each unique word pair, find reference matches once
    pair_ref_cache: Dict[Tuple[str, str], List[Dict]] = {}
    for pair in unique_pairs:
        w1, w2 = pair
        matches = _find_reference_matches(w1, w2, bigram_set, all_ref_words)
        pair_ref_cache[pair] = matches

    enriched_cc = []
    n_with_vowel_error = 0

    for entry in cc_entries:
        w1, w2 = entry['w1'], entry['w2']
        ref_matches = pair_ref_cache.get((w1, w2), [])

        # Determine if any match involves a vowel error
        has_vowel_error = any(
            m.get('w1_vowel_error') or m.get('w2_vowel_error')
            for m in ref_matches
        )
        if has_vowel_error:
            n_with_vowel_error += 1

        # Get EVA token at this position
        pos = entry.get('position', -1)
        eva_token_w1 = token_evas[pos] if 0 <= pos < len(token_evas) else ''
        eva_token_w2 = token_evas[pos + 1] if 0 <= pos + 1 < len(token_evas) else ''

        enriched_cc.append({
            'folio': entry['folio'],
            'position': pos,
            'w1': w1,
            'w2': w2,
            'w1_source': entry.get('w1_source', ''),
            'w2_source': entry.get('w2_source', ''),
            'eva_token_w1': eva_token_w1,
            'eva_token_w2': eva_token_w2,
            'reference_matches': ref_matches,
            'n_reference_matches': len(ref_matches),
            'has_vowel_error': has_vowel_error,
        })

    print(f"     CC entries with reference matches: "
          f"{sum(1 for e in enriched_cc if e['n_reference_matches'] > 0)}")
    print(f"     CC entries with vowel errors: {n_with_vowel_error}")

    # ── 4. Summary by unique pair ──
    print("\n  4. Summary by unique word pair …")
    pair_summary = []
    for pair in sorted(unique_pairs):
        w1, w2 = pair
        count = sum(1 for e in cc_entries if e['w1'] == w1 and e['w2'] == w2)
        ref_matches = pair_ref_cache.get(pair, [])
        has_vowel = any(m.get('w1_vowel_error') or m.get('w2_vowel_error')
                       for m in ref_matches)
        pair_summary.append({
            'w1': w1, 'w2': w2,
            'count': count,
            'n_reference_matches': len(ref_matches),
            'has_vowel_error': has_vowel,
            'reference_matches': ref_matches[:5],  # top 5
        })
        print(f"     {w1} {w2}: {count}× → {len(ref_matches)} ref matches"
              f" (vowel_err={has_vowel})")

    # ── 5. Reconstruct all medical phrases ──
    print("\n  5. Reconstructing medical phrases …")
    # Need decoded_lower as list
    decoded_lower = [w.lower() if isinstance(w, str) else '' for w in token_decoded]

    medical_phrases_full = _reconstruct_medical_phrases(
        decoded_lower, token_folios, classifications,
        merged_dict, latin_10k, italian_10k, signal_words,
    )
    print(f"     Medical phrases reconstructed: {len(medical_phrases_full)}")

    # ── 6. Save ──
    elapsed = time.time() - t0

    output = {
        'n_cc_entries': len(enriched_cc),
        'n_unique_pairs': len(unique_pairs),
        'n_with_vowel_error': n_with_vowel_error,
        'n_with_reference_match': sum(
            1 for e in enriched_cc if e['n_reference_matches'] > 0
        ),
        'cc_entries': enriched_cc,
        'pair_summary': pair_summary,
        'n_medical_phrases': len(medical_phrases_full),
        'medical_phrases_full': medical_phrases_full,
        'verdict': (
            f"{len(enriched_cc)} CC bigrams decomposed, "
            f"{len(unique_pairs)} unique pairs, "
            f"{n_with_vowel_error} with vowel errors, "
            f"{len(medical_phrases_full)} medical phrases reconstructed."
        ),
        'runtime_seconds': round(elapsed, 1),
    }

    out_path = os.path.join(rd, 'ed1_decomposition.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(output), f, indent=2)
    print(f"\n  Saved → {out_path} ({elapsed:.1f}s)")
