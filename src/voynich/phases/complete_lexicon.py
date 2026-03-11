"""
Step 41.8 – Complete Signal Word Lexicon
=========================================
Assemble the definitive glossed lexicon for all 73 signal words by
merging original glosses (Step 40.9), dictionary search results
(Step 41.6), and context disambiguations (Step 41.7).

Dependency chain:
    syllable_lexicon.json            (Step 40.9 — 28 original glosses)
    venetian_dictionary_search.json  (Step 41.6 — new glosses)
    context_disambiguation.json      (Step 41.7 — disambiguated meanings)
    venetian_confirmed.json          (Step 41.4 — validated σ-scores)
    venetian_forms.json              (Step 40.1 — Venetian extended set)
    data/reference/italian/anonimo_veneziano.txt
        → complete_lexicon.json  (this step)
"""

import json
import os
import re
import time
from collections import Counter
from typing import Any, Dict, List, Set

from voynich.core._paths import data_dir as _data_dir, results_dir as _results_dir


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
# Anonimo Veneziano loader
# ---------------------------------------------------------------------------

def _load_anonimo_vocab(data_dir: str) -> Set[str]:
    """Extract unique lowercased words from the Anonimo Veneziano text."""
    path = os.path.join(data_dir, 'reference', 'italian', 'anonimo_veneziano.txt')
    if not os.path.exists(path):
        return set()
    with open(path, encoding='utf-8', errors='replace') as f:
        text = f.read()
    return set(re.findall(r'[a-z]+', text.lower()))


# ---------------------------------------------------------------------------
# Merge logic
# ---------------------------------------------------------------------------

def _build_complete_entry(
    word: str,
    syl_entry: Dict,
    confirmed_entry: Dict,
    dict_search_match: Dict,
    disambiguation: Dict,
    word_signal: Dict,
) -> Dict:
    """Build a single definitive lexicon entry for one signal word."""

    # Start with original gloss if available
    original_gloss = syl_entry.get('english_gloss', '???')
    original_pos = syl_entry.get('part_of_speech', 'unknown')
    original_domain = syl_entry.get('medical_domain', 'unknown')
    original_latin = syl_entry.get('latin_equivalent', '')
    original_venetian = syl_entry.get('venetian_form', '')

    # Determine identification method
    if original_gloss and original_gloss != '???':
        identification_method = 'original_gloss'
        english_gloss = original_gloss
        pos = original_pos
        domain = original_domain
        latin_equiv = original_latin
        venetian_word = original_venetian
        confidence = 'HIGH'
    elif dict_search_match:
        # Use dictionary search result
        method = dict_search_match.get('method', 'unknown')
        matched_word = dict_search_match.get('matched_word', '')
        identification_method = f'dict_search:{method}'
        english_gloss = f'cf. {matched_word}'
        pos = 'unknown'
        domain = 'unknown'
        latin_equiv = ''
        venetian_word = matched_word
        confidence = dict_search_match.get('confidence', 'LOW')
    else:
        identification_method = 'unidentified'
        english_gloss = ''
        pos = 'unknown'
        domain = 'unknown'
        latin_equiv = ''
        venetian_word = ''
        confidence = 'NONE'

    # Apply disambiguation if this word was disambiguated
    if disambiguation and disambiguation.get('verdict') in ('STRONG', 'WEAK'):
        primary = disambiguation.get('primary_meaning', '')
        if primary:
            english_gloss = primary
            identification_method += '+disambiguated'
            # Try to extract domain from the meaning string
            primary_lower = primary.lower()
            if 'anatomical' in primary_lower:
                domain = 'anatomical'
            elif 'pharmaceutical' in primary_lower:
                domain = 'pharmaceutical'
            elif 'botanical' in primary_lower:
                domain = 'botanical'

    # Get sigma and corpus frequency
    sigma = word_signal.get('sigma', syl_entry.get('sigma', 0.0))
    real_count = word_signal.get('real_count', 0)

    # Confidence from sigma if not already set from gloss
    if confidence == 'NONE' or confidence == 'LOW':
        if sigma >= 20:
            confidence = 'MEDIUM'  # high sigma but no gloss → medium
        elif sigma >= 10:
            confidence = 'LOW'

    return {
        'decoded': word,
        'sigma': sigma,
        'venetian_word': venetian_word,
        'latin_equiv': latin_equiv,
        'english_gloss': english_gloss,
        'pos': pos,
        'domain': domain,
        'confidence': confidence,
        'identification_method': identification_method,
        'corpus_frequency': real_count,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_complete_lexicon() -> None:
    """Step 41.8: Assemble complete signal word lexicon."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 41.8: Complete Signal Word Lexicon")
    print("=" * 70)

    rd = _results_dir()
    dd = _data_dir()

    # ── 1. Load inputs ──
    print("\n  1. Loading inputs …")
    syl_lex = _safe_load(os.path.join(rd, 'syllable_lexicon.json'))
    dict_search = _safe_load(os.path.join(rd, 'venetian_dictionary_search.json'))
    disambiguation = _safe_load(os.path.join(rd, 'context_disambiguation.json'))
    ven_confirmed = _safe_load(os.path.join(rd, 'venetian_confirmed.json'))
    merged_signal = _safe_load(os.path.join(rd, 'merged_signal.json'))

    lexicon_entries = syl_lex.get('syllable_lexicon', {})
    new_glosses = dict_search.get('new_glosses', {})
    disambiguations = disambiguation.get('disambiguations', {})
    confirmed_vocab = ven_confirmed.get('vocabulary', [])
    word_signals = merged_signal.get('word_signals', [])

    print(f"    Original lexicon entries: {len(lexicon_entries)}")
    print(f"    New glosses from dict search: {len(new_glosses)}")
    print(f"    Disambiguations: {len(disambiguations)}")
    print(f"    Confirmed vocabulary: {len(confirmed_vocab)}")

    # Build confirmed lookup
    confirmed_lookup: Dict[str, Dict] = {}
    for entry in confirmed_vocab:
        confirmed_lookup[entry.get('decoded', '')] = entry

    # Build word_signal lookup
    ws_lookup: Dict[str, Dict] = {}
    for ws in word_signals:
        ws_lookup[ws.get('word', '')] = ws

    # ── 2. Merge into complete lexicon ──
    print("\n  2. Merging into complete lexicon …")
    complete_lexicon: List[Dict] = []

    for word in sorted(lexicon_entries.keys()):
        syl_entry = lexicon_entries.get(word, {})
        confirmed_entry = confirmed_lookup.get(word, {})
        dict_match = new_glosses.get(word, {})
        disambig = disambiguations.get(word, {})
        ws = ws_lookup.get(word, {})

        entry = _build_complete_entry(
            word, syl_entry, confirmed_entry, dict_match, disambig, ws,
        )
        complete_lexicon.append(entry)

    # Sort by sigma descending
    complete_lexicon.sort(key=lambda x: -x['sigma'])
    print(f"    Complete lexicon entries: {len(complete_lexicon)}")

    # ── 3. Compute stats ──
    print("\n  3. Computing statistics …")
    n_glossed = sum(1 for e in complete_lexicon if e['english_gloss'])
    n_unglossed = len(complete_lexicon) - n_glossed

    # POS distribution
    pos_counts: Counter = Counter()
    for entry in complete_lexicon:
        pos = entry.get('pos', 'unknown')
        primary_pos = pos.split('/')[0]
        pos_counts[primary_pos] += 1

    # Domain distribution
    domain_counts: Counter = Counter()
    for entry in complete_lexicon:
        domain_counts[entry.get('domain', 'unknown')] += 1

    # Confidence distribution
    conf_counts: Counter = Counter()
    for entry in complete_lexicon:
        conf_counts[entry.get('confidence', 'NONE')] += 1

    # Identification method distribution
    method_counts: Counter = Counter()
    for entry in complete_lexicon:
        method = entry.get('identification_method', 'unknown')
        # Simplify for counting
        base_method = method.split('+')[0].split(':')[0]
        method_counts[base_method] += 1

    print(f"    Glossed: {n_glossed}")
    print(f"    Unglossed remaining: {n_unglossed}")

    print(f"\n    POS distribution:")
    for pos, count in pos_counts.most_common():
        print(f"      {pos}: {count}")

    print(f"\n    Domain distribution:")
    for domain, count in domain_counts.most_common():
        print(f"      {domain}: {count}")

    print(f"\n    Confidence distribution:")
    for conf, count in conf_counts.most_common():
        print(f"      {conf}: {count}")

    # ── 4. Anonimo Veneziano overlap ──
    print("\n  4. Anonimo Veneziano overlap …")
    anonimo_vocab = _load_anonimo_vocab(dd)
    n_in_anonimo = 0
    anonimo_matches: List[str] = []
    for entry in complete_lexicon:
        decoded = entry['decoded']
        if decoded in anonimo_vocab:
            n_in_anonimo += 1
            anonimo_matches.append(decoded)

    anonimo_overlap_rate = n_in_anonimo / len(complete_lexicon) if complete_lexicon else 0.0
    print(f"    Glossed words in Anonimo: {n_in_anonimo}/{len(complete_lexicon)} "
          f"({anonimo_overlap_rate:.1%})")
    if anonimo_matches:
        print(f"    Matches: {', '.join(anonimo_matches[:20])}")
        if len(anonimo_matches) > 20:
            print(f"    … and {len(anonimo_matches) - 20} more")

    # ── 5. Print complete lexicon table ──
    print(f"\n  5. Complete lexicon (sorted by σ):")
    print(f"    {'#':>3s} {'Decoded':10s} {'σ':>8s} {'Freq':>6s} "
          f"{'Gloss':25s} {'Domain':15s} {'Conf':8s}")
    print(f"    {'—' * 80}")
    for i, entry in enumerate(complete_lexicon[:40], 1):
        gloss = entry['english_gloss'] or '—'
        if len(gloss) > 25:
            gloss = gloss[:22] + '…'
        print(f"    {i:3d} {entry['decoded']:10s} {entry['sigma']:8.1f} "
              f"{entry['corpus_frequency']:6d} {gloss:25s} "
              f"{entry['domain']:15s} {entry['confidence']:8s}")
    if len(complete_lexicon) > 40:
        print(f"    … and {len(complete_lexicon) - 40} more")

    # ── 6. Save ──
    elapsed = time.time() - t0

    output = {
        'n_total': len(complete_lexicon),
        'n_glossed': n_glossed,
        'n_unglossed_remaining': n_unglossed,
        'pos_distribution': dict(pos_counts),
        'domain_distribution': dict(domain_counts),
        'confidence_distribution': dict(conf_counts),
        'identification_methods': dict(method_counts),
        'anonimo_overlap': {
            'n_in_anonimo': n_in_anonimo,
            'overlap_rate': round(anonimo_overlap_rate, 4),
            'matched_words': anonimo_matches,
        },
        'complete_lexicon': {e['decoded']: e for e in complete_lexicon},
        'lexicon_list': complete_lexicon,
        'runtime_seconds': round(elapsed, 1),
    }

    out_path = os.path.join(rd, 'complete_lexicon.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(output), f, indent=2)
    print(f"\n  Saved → {out_path} ({elapsed:.1f}s)")
