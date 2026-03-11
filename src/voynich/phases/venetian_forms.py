"""
Step 40.1 – Venetian Phonological Form Inventory
==================================================
Build a comprehensive inventory of how Latin/Italian medical words appear
in 15th-century Venetian dialect by applying documented sound changes.

Dependency chain:
    merged_dict.json         (Step 38.1)
    venetian_lexicon.json    (Step 39.11)
    data/reference/italian/anonimo_veneziano.txt
        → venetian_forms.json  (this step)
"""

import json
import os
import re
import time
from collections import Counter
from typing import Any, Dict, List, Set, Tuple

from voynich.core._paths import data_dir as _data_dir, results_dir as _results_dir
from voynich.core.reference import (
    VENETIAN_SOUND_CHANGES,
    apply_venetian_sound_changes,
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


def _tokenize_text(text: str) -> List[str]:
    """Tokenize text into lowercase words."""
    return re.findall(r'[a-zA-ZàèéìòùÀÈÉÌÒÙ]+', text.lower())


def _deaccent(word: str) -> str:
    """Remove accents for matching."""
    accent_map = {
        'à': 'a', 'è': 'e', 'é': 'e', 'ì': 'i', 'ò': 'o', 'ù': 'u',
    }
    return ''.join(accent_map.get(ch, ch) for ch in word)


# ---------------------------------------------------------------------------
# Core: Venetian form generation
# ---------------------------------------------------------------------------

def _build_venetian_medical_dict(
    base_words: Set[str],
    sound_changes: List[Tuple[str, str, str]],
) -> Tuple[Dict[str, List[Dict]], Dict[str, int]]:
    """Apply Venetian sound changes to all base words.

    Returns:
        venetian_forms: dict mapping venetian_form -> [{origin, rule}]
        rule_counts: dict mapping rule_name -> count of words affected
    """
    venetian_forms: Dict[str, List[Dict]] = {}
    rule_counts: Counter = Counter()

    for word in sorted(base_words):
        variants = apply_venetian_sound_changes(word)
        for variant, rules in variants.items():
            if variant not in venetian_forms:
                venetian_forms[variant] = []
            for rule in rules:
                venetian_forms[variant].append({
                    'origin': word,
                    'rule': rule,
                })
                rule_counts[rule] += 1

    return venetian_forms, dict(rule_counts)


def _load_anonimo_vocab() -> Set[str]:
    """Load vocabulary from the Anonimo Veneziano text."""
    anonimo_path = os.path.join(_data_dir(), 'reference', 'italian',
                                'anonimo_veneziano.txt')
    if not os.path.exists(anonimo_path):
        return set()
    with open(anonimo_path) as f:
        text = f.read()
    tokens = _tokenize_text(text)
    # Deduplicate with deaccented forms
    vocab = set()
    for t in tokens:
        vocab.add(t)
        da = _deaccent(t)
        if da != t:
            vocab.add(da)
    return vocab


def _cross_reference_anonimo(
    generated_forms: Dict[str, List[Dict]],
    anonimo_vocab: Set[str],
) -> Dict[str, str]:
    """Annotate generated forms: 'attested' if in Anonimo, else 'predicted'."""
    status = {}
    for form in generated_forms:
        if form in anonimo_vocab:
            status[form] = 'attested'
        else:
            status[form] = 'predicted'
    return status


def _test_specific_words(
    target_words: List[str],
    venetian_forms: Dict[str, List[Dict]],
    anonimo_vocab: Set[str],
    base_words: Set[str],
) -> List[Dict]:
    """Test whether specific decoded words are valid Venetian forms."""
    results = []
    for word in target_words:
        entry = {
            'word': word,
            'in_base_dict': word in base_words,
            'is_venetian_form': word in venetian_forms,
            'in_anonimo': word in anonimo_vocab,
            'origins': [],
        }
        if word in venetian_forms:
            entry['origins'] = venetian_forms[word]
        results.append(entry)
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_venetian_forms() -> None:
    """Step 40.1: Venetian Phonological Form Inventory."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 40.1: Venetian Phonological Form Inventory")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load inputs ──
    print("\n  1. Loading inputs …")
    merged_dict = _safe_load(os.path.join(rd, 'merged_dict.json'))
    ven_lex = _safe_load(os.path.join(rd, 'venetian_lexicon.json'))

    # Build base word set from merged dictionary
    latin_words = set(merged_dict.get('latin_10k_words', []))
    italian_words = set(merged_dict.get('italian_10k_words', []))
    ven_supplement = set(ven_lex.get('venetian_words', []))
    # Also include supplement_words (deaccented forms)
    for entry in ven_lex.get('supplement_words', []):
        if isinstance(entry, str):
            ven_supplement.add(entry)
        elif isinstance(entry, dict):
            ven_supplement.add(entry.get('word', ''))

    base_words = latin_words | italian_words
    print(f"    Base dictionary: {len(base_words):,} words "
          f"(Latin {len(latin_words):,} + Italian {len(italian_words):,})")
    print(f"    Venetian supplement: {len(ven_supplement):,} words")

    # ── 2. Load Anonimo Veneziano vocabulary ──
    print("\n  2. Loading Anonimo Veneziano vocabulary …")
    anonimo_vocab = _load_anonimo_vocab()
    print(f"    Anonimo vocabulary: {len(anonimo_vocab):,} unique forms")

    # ── 3. Apply Venetian sound changes ──
    print("\n  3. Applying Venetian sound changes …")
    venetian_forms, rule_counts = _build_venetian_medical_dict(
        base_words, VENETIAN_SOUND_CHANGES,
    )
    print(f"    Generated {len(venetian_forms):,} Venetian variant forms")
    print(f"    Rule application counts:")
    for rule, count in sorted(rule_counts.items(), key=lambda x: -x[1]):
        print(f"      {rule}: {count:,}")

    # ── 4. Cross-reference with Anonimo ──
    print("\n  4. Cross-referencing with Anonimo Veneziano …")
    attestation = _cross_reference_anonimo(venetian_forms, anonimo_vocab)
    n_attested = sum(1 for s in attestation.values() if s == 'attested')
    n_predicted = sum(1 for s in attestation.values() if s == 'predicted')
    print(f"    Attested in Anonimo: {n_attested:,}")
    print(f"    Predicted (not attested): {n_predicted:,}")

    # ── 5. Build extended Venetian word set ──
    print("\n  5. Building extended Venetian word set …")
    # The extended set = base_words + Venetian forms + Anonimo vocab + supplement
    venetian_extended_set = set(base_words)
    venetian_extended_set.update(venetian_forms.keys())
    venetian_extended_set.update(anonimo_vocab)
    venetian_extended_set.update(ven_supplement)
    # Remove empty strings
    venetian_extended_set.discard('')
    print(f"    Extended Venetian set: {len(venetian_extended_set):,} words")
    n_new = len(venetian_extended_set) - len(base_words)
    print(f"    New forms added: {n_new:,}")

    # ── 6. Test specific signal words ──
    print("\n  6. Testing key signal words …")
    # Load signal words for targeted testing
    merged_signal = _safe_load(os.path.join(rd, 'merged_signal.json'))
    signal_words = [w['word'] for w in merged_signal.get('word_signals', [])
                    if w.get('is_genuine_signal')]
    specific_tests = _test_specific_words(
        signal_words[:20],  # test top signal words
        venetian_forms,
        anonimo_vocab,
        base_words,
    )
    n_venetian_signal = sum(1 for t in specific_tests if t['is_venetian_form'])
    print(f"    Signal words that are Venetian forms: {n_venetian_signal}/{len(specific_tests)}")

    # ── 7. Build provenance map (venetian_form -> origin word) ──
    provenance_map = {}
    for form, origins in venetian_forms.items():
        provenance_map[form] = origins[0]['origin'] if origins else ''

    # ── 8. Save ──
    elapsed = time.time() - t0

    output = {
        'n_base_words': len(base_words),
        'n_latin': len(latin_words),
        'n_italian': len(italian_words),
        'n_venetian_supplement': len(ven_supplement),
        'n_anonimo_vocab': len(anonimo_vocab),
        'n_venetian_forms_generated': len(venetian_forms),
        'n_attested': n_attested,
        'n_predicted': n_predicted,
        'n_extended_set': len(venetian_extended_set),
        'n_new_forms': n_new,
        'rule_counts': rule_counts,
        'attestation_summary': {
            'attested': n_attested,
            'predicted': n_predicted,
        },
        'specific_word_tests': specific_tests,
        'n_signal_venetian_forms': n_venetian_signal,
        # Store the extended set as sorted list for downstream steps
        'venetian_extended_set': sorted(venetian_extended_set),
        'provenance_map': provenance_map,
        'runtime_seconds': round(elapsed, 1),
    }

    out_path = os.path.join(rd, 'venetian_forms.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(output), f, indent=2)
    print(f"\n  Saved → {out_path} ({elapsed:.1f}s)")
