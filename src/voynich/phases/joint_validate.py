"""
Step 37.9 – Joint Swap Validation
====================================
Validate the best joint swap on the full corpus and held-out data.

Dependency chain:
    joint_swap.json            (Step 37.8)
    combined_refine.json       (Phase 15)
    modifier_integrate.json    (Phase 16)
    signal_10k.json            (Step 36.2)
    decode_10k.json            (Step 36.1)
    consonant_grouping.json    (Step 37.1)
        → joint_validate.json  (this step)
"""

import json
import os
import random
import time
from collections import Counter
from typing import Any, Dict, List, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    decode_token_modifier_aware,
)
from voynich.core.reference import load_reference_corpus


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


_FUNCTION_WORDS = {
    'de', 'in', 'ad', 'et', 'cum', 'per', 'pro', 'ex', 'ab', 'sub',
    'non', 'si', 'ut', 'sed', 'ne', 'ac', 'at', 'an', 'se', 'te',
    'me', 'nos', 'iam', 'est', 'sunt', 'hoc', 'id', 'ea', 'is',
    'di', 'du', 'ce', 'ci', 'co', 'cu', 'bi', 'bo', 'be', 'da',
    'la', 'le', 'li', 'lo', 'ni', 'no', 'nu', 'ra', 're', 'ri',
    'ro', 'sa', 'so', 'su', 'ta', 'ti', 'to',
}


def _is_content_word(word: str) -> bool:
    return word.lower() not in _FUNCTION_WORDS and len(word) >= 4


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_joint_validate() -> None:
    """Step 37.9: Joint Swap Validation."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 37.9: Joint Swap Validation")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load inputs ──
    print("\n  1. Loading inputs …")
    swap_data = _safe_load(os.path.join(rd, 'joint_swap.json'))
    refine_data = _safe_load(os.path.join(rd, 'combined_refine.json'))
    mod_data = _safe_load(os.path.join(rd, 'modifier_integrate.json'))
    signal_data = _safe_load(os.path.join(rd, 'signal_10k.json'))
    cg_data = _safe_load(os.path.join(rd, 'consonant_grouping.json'))

    original_assignment = refine_data.get('best_assignment', {})
    corrected_table = swap_data.get('accumulated_table', original_assignment)
    swap_sequence = swap_data.get('swap_sequence', [])
    modifier_chars = set(mod_data.get('modifier_chars', []))
    modifier_rules: Dict[str, str] = {}
    for c in mod_data.get('classifications', []):
        if c.get('eva_char') in modifier_chars:
            modifier_rules[c['eva_char']] = 'silent'

    token_evas = signal_data.get('token_evas', [])
    token_folios = signal_data.get('token_folios', [])

    eva_to_triple = build_eva_to_triple_lookup()

    print(f"     {len(swap_sequence)} swaps to validate")

    # Build dictionary + bigrams
    ref = load_reference_corpus(languages=['latin'], verbose=False)
    ref_tokens = [w.lower() for w in ref.get_combined_tokens('latin') if len(w) >= 2]
    word_freq = Counter(ref_tokens)
    dict_10k = set(w for w, _ in word_freq.most_common(10000))

    ref_bigrams = set()
    for i in range(len(ref_tokens) - 1):
        if ref_tokens[i] in dict_10k and ref_tokens[i + 1] in dict_10k:
            ref_bigrams.add((ref_tokens[i], ref_tokens[i + 1]))

    # ── 2. Full corpus decode (original vs corrected) ──
    print("  2. Full corpus decode …")

    def _full_decode(table: Dict[str, str]) -> List[str]:
        decoded = []
        for eva in token_evas:
            d = decode_token_modifier_aware(
                eva, table, eva_to_triple, modifier_chars, modifier_rules)
            decoded.append(d.lower())
        return decoded

    orig_decoded = _full_decode(original_assignment)
    corr_decoded = _full_decode(corrected_table)

    orig_hits = sum(1 for w in orig_decoded if w in dict_10k)
    corr_hits = sum(1 for w in corr_decoded if w in dict_10k)
    n = len(token_evas)

    orig_hit_rate = orig_hits / n if n > 0 else 0.0
    corr_hit_rate = corr_hits / n if n > 0 else 0.0

    orig_content = sum(1 for w in orig_decoded if w in dict_10k and _is_content_word(w))
    corr_content = sum(1 for w in corr_decoded if w in dict_10k and _is_content_word(w))

    print(f"     Original: hit={orig_hit_rate:.3%}, content={orig_content}")
    print(f"     Corrected: hit={corr_hit_rate:.3%}, content={corr_content}")

    # ── 3. Signal isolation on corrected ──
    print("  3. Signal isolation comparison …")
    # Simplified: count SIGNAL-classified tokens that still hit
    signal_cls = signal_data.get('token_classifications', [])
    orig_signal_hits = sum(1 for i in range(min(n, len(signal_cls)))
                          if signal_cls[i] == 'SIGNAL' and orig_decoded[i] in dict_10k)
    corr_signal_hits = sum(1 for i in range(min(n, len(signal_cls)))
                          if signal_cls[i] == 'SIGNAL' and corr_decoded[i] in dict_10k)

    n_signal = signal_cls.count('SIGNAL')
    orig_signal_rate = orig_signal_hits / n_signal if n_signal > 0 else 0.0
    corr_signal_rate = corr_signal_hits / n_signal if n_signal > 0 else 0.0

    print(f"     SIGNAL token hit rate (orig):  {orig_signal_rate:.3%}")
    print(f"     SIGNAL token hit rate (corr):  {corr_signal_rate:.3%}")

    # ── 4. Bigram z-score ──
    print("  4. Bigram z-score …")

    def _bigram_z(decoded: List[str], folios: List[str]) -> Tuple[float, int, int]:
        """Compute bigram z and content-content count."""
        exact = 0
        cc = 0
        for i in range(len(decoded) - 1):
            if i < len(folios) - 1 and folios[i] != folios[i + 1]:
                continue
            w1, w2 = decoded[i], decoded[i + 1]
            if w1 in dict_10k and w2 in dict_10k:
                if (w1, w2) in ref_bigrams:
                    exact += 1
                    if _is_content_word(w1) and _is_content_word(w2):
                        cc += 1

        # Null estimate via shuffled positions
        rng = random.Random(42)
        null_counts = []
        signal_words = [w for w in decoded if w in dict_10k]
        for _ in range(100):
            shuffled = list(signal_words)
            rng.shuffle(shuffled)
            null_hits = sum(1 for j in range(len(shuffled) - 1)
                          if (shuffled[j], shuffled[j + 1]) in ref_bigrams)
            null_counts.append(null_hits)
        null_mean = sum(null_counts) / len(null_counts)
        null_var = sum((c - null_mean) ** 2 for c in null_counts) / len(null_counts)
        null_std = null_var ** 0.5
        z = ((exact - null_mean) / null_std if null_std > 0
             else (10.0 if exact > null_mean else 0.0))
        return z, exact, cc

    orig_z, orig_exact, orig_cc = _bigram_z(orig_decoded, token_folios)
    corr_z, corr_exact, corr_cc = _bigram_z(corr_decoded, token_folios)

    print(f"     Original: z={orig_z:.2f}, exact={orig_exact}, CC={orig_cc}")
    print(f"     Corrected: z={corr_z:.2f}, exact={corr_exact}, CC={corr_cc}")
    z_maintained = corr_z >= 10.0

    # ── 5. Held-out validation ──
    print("  5. Held-out validation (50/50 split) …")
    mid = n // 2
    train_evas = token_evas[:mid]
    test_evas = token_evas[mid:]

    train_corr = [decode_token_modifier_aware(
        e, corrected_table, eva_to_triple, modifier_chars, modifier_rules).lower()
        for e in train_evas]
    test_corr = [decode_token_modifier_aware(
        e, corrected_table, eva_to_triple, modifier_chars, modifier_rules).lower()
        for e in test_evas]

    train_hit = sum(1 for w in train_corr if w in dict_10k) / len(train_corr) if train_corr else 0.0
    test_hit = sum(1 for w in test_corr if w in dict_10k) / len(test_corr) if test_corr else 0.0
    train_content = sum(1 for w in train_corr if w in dict_10k and _is_content_word(w))
    test_content = sum(1 for w in test_corr if w in dict_10k and _is_content_word(w))

    print(f"     Train: hit={train_hit:.3%}, content={train_content}")
    print(f"     Test:  hit={test_hit:.3%}, content={test_content}")
    generalizes = test_hit >= train_hit * 0.9  # Within 10%

    # ── 6. Cross-reference with consonant grouping ──
    print("  6. Cross-reference with consonant grouping …")
    # Check if swaps change only vowels (consonant preserved)
    vowel_only = 0
    consonant_changed = 0
    for swap in swap_sequence:
        orig_s1 = original_assignment.get(swap['triple1'], '')
        orig_s2 = original_assignment.get(swap['triple2'], '')
        new_s1 = swap['s1']
        new_s2 = swap['s2']

        # Extract consonant onsets
        def _onset(s):
            vowels = set('aeiou')
            onset = ''
            for ch in s:
                if ch in vowels:
                    break
                onset += ch
            return onset

        if _onset(orig_s1) == _onset(new_s1):
            vowel_only += 1
        else:
            consonant_changed += 1
        if _onset(orig_s2) == _onset(new_s2):
            vowel_only += 1
        else:
            consonant_changed += 1

    print(f"     Vowel-only changes: {vowel_only}")
    print(f"     Consonant changes:  {consonant_changed}")

    # ── 7. Content bigram context ──
    print("  7. Content-content bigram context …")
    cc_contexts = []
    for i in range(len(corr_decoded) - 1):
        if i >= len(token_folios) - 1:
            break
        if token_folios[i] != token_folios[i + 1]:
            continue
        w1, w2 = corr_decoded[i], corr_decoded[i + 1]
        if (w1 in dict_10k and w2 in dict_10k and
                _is_content_word(w1) and _is_content_word(w2) and
                (w1, w2) in ref_bigrams):
            # Get context window
            start = max(0, i - 3)
            end = min(len(corr_decoded), i + 5)
            context_words = corr_decoded[start:end]
            cc_contexts.append({
                'word1': w1,
                'word2': w2,
                'folio': token_folios[i],
                'position': i,
                'context': ' '.join(context_words),
            })

    print(f"     {len(cc_contexts)} content-content bigrams with context")
    for ctx in cc_contexts[:5]:
        print(f"       {ctx['folio']}: \"{ctx['context']}\" [{ctx['word1']} {ctx['word2']}]")

    # ── 8. Save ──
    elapsed = time.time() - t0

    output = {
        'n_swaps': len(swap_sequence),
        'swap_sequence': swap_sequence,
        'original_hit_rate': round(orig_hit_rate, 4),
        'corrected_hit_rate': round(corr_hit_rate, 4),
        'delta_hit': round(corr_hit_rate - orig_hit_rate, 4),
        'original_content_words': orig_content,
        'corrected_content_words': corr_content,
        'original_bigram_z': round(orig_z, 2),
        'corrected_bigram_z': round(corr_z, 2),
        'z_maintained': z_maintained,
        'original_cc_bigrams': orig_cc,
        'corrected_cc_bigrams': corr_cc,
        'train_hit': round(train_hit, 4),
        'test_hit': round(test_hit, 4),
        'train_content': train_content,
        'test_content': test_content,
        'generalizes': generalizes,
        'vowel_only_changes': vowel_only,
        'consonant_changes': consonant_changed,
        'content_bigram_contexts': cc_contexts[:50],
        'corrected_table': corrected_table,
        'verdict': (
            f"Validation: hit={corr_hit_rate:.3%} "
            f"(Δ={corr_hit_rate - orig_hit_rate:+.3%}), "
            f"z={corr_z:.2f} ({'MAINTAINED' if z_maintained else 'DROPPED'}), "
            f"CC={corr_cc}, "
            f"held-out {'GENERALIZES' if generalizes else 'OVERFITS'}. "
            f"Vowel-only={vowel_only}, consonant={consonant_changed}."
        ),
        'runtime_seconds': round(elapsed, 1),
    }

    out_path = os.path.join(rd, 'joint_validate.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(output), f, indent=2)
    print(f"\n  Saved → {out_path} ({elapsed:.1f}s)")
