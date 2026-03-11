"""
Step 40.9 – Signal Word Syllable Lexicon
==========================================
Build a comprehensive lexicon translating each signal word to its most
likely Venetian meaning, with Latin equivalents and English glosses.

Dependency chain:
    merged_signal.json       (Step 38.3)
    merged_bigrams.json      (Step 38.4)
    venetian_lexicon.json    (Step 39.11)
    venetian_forms.json      (Step 40.1)
        → syllable_lexicon.json  (this step)
"""

import json
import os
import re
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Set

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
# Gloss table: signal words → English meanings
# ---------------------------------------------------------------------------
# This is the core interpretive act of Phase 40 Track C: assigning meanings
# to the confirmed signal words based on Latin/Italian/Venetian dictionaries
# and pharmaceutical/medical context.

SIGNAL_WORD_GLOSSES = {
    # Function words (Latin/Italian shared)
    'de': {'gloss': 'of/from', 'pos': 'prep', 'domain': 'function', 'latin': 'de', 'venetian': 'de'},
    'di': {'gloss': 'of', 'pos': 'prep', 'domain': 'function', 'latin': 'de', 'venetian': 'di'},
    'se': {'gloss': 'if/self', 'pos': 'conj/pron', 'domain': 'function', 'latin': 'se/si', 'venetian': 'se'},
    'ne': {'gloss': 'not/nor', 'pos': 'adv/conj', 'domain': 'function', 'latin': 'ne', 'venetian': 'ne'},
    'si': {'gloss': 'thus/yes', 'pos': 'adv', 'domain': 'function', 'latin': 'sic', 'venetian': 'si'},
    'la': {'gloss': 'the (f.)', 'pos': 'art', 'domain': 'function', 'latin': 'illa', 'venetian': 'la'},
    'le': {'gloss': 'the (f.pl.)', 'pos': 'art', 'domain': 'function', 'latin': 'illae', 'venetian': 'le'},
    'co': {'gloss': 'with', 'pos': 'prep', 'domain': 'function', 'latin': 'cum', 'venetian': 'co/con'},
    'du': {'gloss': 'two/of the', 'pos': 'num/prep', 'domain': 'function', 'latin': 'duo', 'venetian': 'du'},
    'ce': {'gloss': 'here/this', 'pos': 'pron', 'domain': 'function', 'latin': 'hic', 'venetian': 'ce'},
    'ni': {'gloss': 'nor/nothing', 'pos': 'conj', 'domain': 'function', 'latin': 'nec', 'venetian': 'ni'},
    'te': {'gloss': 'you/thee', 'pos': 'pron', 'domain': 'function', 'latin': 'te', 'venetian': 'te'},
    'mi': {'gloss': 'me/my', 'pos': 'pron', 'domain': 'function', 'latin': 'mihi', 'venetian': 'mi'},
    'bi': {'gloss': 'twice/two', 'pos': 'prefix', 'domain': 'function', 'latin': 'bis', 'venetian': 'bi'},
    'do': {'gloss': 'give/two', 'pos': 'verb/num', 'domain': 'function', 'latin': 'do/duo', 'venetian': 'do'},
    # Medical/pharmaceutical vocabulary
    'bene': {'gloss': 'well/good', 'pos': 'adv/adj', 'domain': 'quality', 'latin': 'bene', 'venetian': 'ben'},
    'cola': {'gloss': 'strain (v.)', 'pos': 'verb', 'domain': 'pharmaceutical', 'latin': 'colare', 'venetian': 'cola'},
    'cora': {'gloss': 'heart', 'pos': 'noun', 'domain': 'anatomical', 'latin': 'cor', 'venetian': 'cora/cor'},
    'sene': {'gloss': 'without/senna', 'pos': 'prep/noun', 'domain': 'botanical', 'latin': 'sine/senna', 'venetian': 'sene'},
    'sero': {'gloss': 'serum/late', 'pos': 'noun/adv', 'domain': 'pharmaceutical', 'latin': 'serum', 'venetian': 'sero'},
    'raro': {'gloss': 'rare/thin', 'pos': 'adj', 'domain': 'quality', 'latin': 'rarus', 'venetian': 'raro'},
    'dine': {'gloss': 'before meal', 'pos': 'noun', 'domain': 'pharmaceutical', 'latin': 'ante cenam', 'venetian': 'dine'},
    'codi': {'gloss': 'codex/tail', 'pos': 'noun', 'domain': 'general', 'latin': 'codex/cauda', 'venetian': 'codi'},
    'bela': {'gloss': 'beautiful', 'pos': 'adj', 'domain': 'quality', 'latin': 'bella', 'venetian': 'bela'},
    'rado': {'gloss': 'scraped/root', 'pos': 'verb/noun', 'domain': 'pharmaceutical', 'latin': 'radere/radix', 'venetian': 'rado'},
    'rosa': {'gloss': 'rose', 'pos': 'noun', 'domain': 'botanical', 'latin': 'rosa', 'venetian': 'rosa'},
    'dise': {'gloss': 'says', 'pos': 'verb', 'domain': 'general', 'latin': 'dicit', 'venetian': 'dise'},
    'dose': {'gloss': 'dose/sweet', 'pos': 'noun/adj', 'domain': 'pharmaceutical', 'latin': 'dosis/dulcis', 'venetian': 'dose'},
    # Preparation verbs (Venetian forms)
    'fa': {'gloss': 'makes/does', 'pos': 'verb', 'domain': 'pharmaceutical', 'latin': 'facit', 'venetian': 'fa'},
    'ha': {'gloss': 'has', 'pos': 'verb', 'domain': 'general', 'latin': 'habet', 'venetian': 'ha'},
    'hi': {'gloss': 'there/to him', 'pos': 'adv/pron', 'domain': 'function', 'latin': 'ibi/illi', 'venetian': 'ghe/li'},
    'ga': {'gloss': 'has (dial.)', 'pos': 'verb', 'domain': 'general', 'latin': 'habet', 'venetian': 'ga'},
    'ra': {'gloss': '(syllable)', 'pos': 'syllable', 'domain': 'general', 'latin': '-', 'venetian': '-'},
    'ro': {'gloss': '(syllable)', 'pos': 'syllable', 'domain': 'general', 'latin': '-', 'venetian': '-'},
    'be': {'gloss': 'well/drink', 'pos': 'adv/verb', 'domain': 'pharmaceutical', 'latin': 'bene/bibere', 'venetian': 'be'},
    'ba': {'gloss': '(syllable)', 'pos': 'syllable', 'domain': 'general', 'latin': '-', 'venetian': '-'},
    'fe': {'gloss': 'faith/bile', 'pos': 'noun', 'domain': 'anatomical', 'latin': 'fides/fel', 'venetian': 'fe'},
}

# Concatenated pair lexicon
CONCAT_LEXICON = {
    'bene': {'gloss': 'well/good', 'parts': ['be', 'ne'], 'domain': 'quality'},
    'sene': {'gloss': 'without/senna', 'parts': ['se', 'ne'], 'domain': 'botanical'},
    'cola': {'gloss': 'strain', 'parts': ['co', 'la'], 'domain': 'pharmaceutical'},
    'cora': {'gloss': 'heart', 'parts': ['co', 'ra'], 'domain': 'anatomical'},
    'dise': {'gloss': 'says', 'parts': ['di', 'se'], 'domain': 'general'},
    'dice': {'gloss': 'says', 'parts': ['di', 'ce'], 'domain': 'general'},
    'radi': {'gloss': 'root', 'parts': ['ra', 'di'], 'domain': 'botanical'},
    'sero': {'gloss': 'serum', 'parts': ['se', 'ro'], 'domain': 'pharmaceutical'},
    'bela': {'gloss': 'beautiful', 'parts': ['be', 'la'], 'domain': 'quality'},
    'dose': {'gloss': 'dose', 'parts': ['do', 'se'], 'domain': 'pharmaceutical'},
    'codi': {'gloss': 'codex/tail', 'parts': ['co', 'di'], 'domain': 'general'},
    'dine': {'gloss': 'before meal', 'parts': ['di', 'ne'], 'domain': 'pharmaceutical'},
    'rosa': {'gloss': 'rose', 'parts': ['ro', 'se'], 'domain': 'botanical'},
    'rado': {'gloss': 'scraped/root', 'parts': ['ra', 'do'], 'domain': 'pharmaceutical'},
}


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def _build_signal_lexicon(
    signal_words: List[Dict],
    gloss_table: Dict,
) -> List[Dict]:
    """Build lexicon entries for each signal word."""
    lexicon = []
    for sw in signal_words:
        word = sw.get('word', '')
        sigma = sw.get('sigma', 0.0)

        gloss_entry = gloss_table.get(word, {})

        if sigma >= 20:
            confidence = 'HIGH'
        elif sigma >= 5:
            confidence = 'MEDIUM'
        elif sigma >= 2:
            confidence = 'LOW'
        else:
            confidence = 'MINIMAL'

        lexicon.append({
            'decoded': word,
            'english_gloss': gloss_entry.get('gloss', '???'),
            'part_of_speech': gloss_entry.get('pos', 'unknown'),
            'medical_domain': gloss_entry.get('domain', 'unknown'),
            'latin_equivalent': gloss_entry.get('latin', ''),
            'venetian_form': gloss_entry.get('venetian', ''),
            'sigma': sigma,
            'confidence': confidence,
            'source': sw.get('source', ''),
        })

    return lexicon


def _pos_distribution(lexicon: List[Dict]) -> Dict[str, int]:
    """Count POS distribution of signal words."""
    pos_counts: Counter = Counter()
    for entry in lexicon:
        pos = entry.get('part_of_speech', 'unknown')
        # Take first POS if multiple (e.g., 'verb/noun')
        primary = pos.split('/')[0]
        pos_counts[primary] += 1
    return dict(pos_counts.most_common())


def _anonimo_pos_distribution() -> Dict[str, float]:
    """Approximate POS distribution from Venetian recipe text.

    Based on analysis of the Anonimo Veneziano's recipe structure:
    imperative verbs, ingredient nouns, prepositions, conjunctions.
    """
    return {
        'verb': 0.20,
        'noun': 0.30,
        'prep': 0.15,
        'adj': 0.10,
        'adv': 0.08,
        'conj': 0.07,
        'art': 0.05,
        'pron': 0.05,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_syllable_lexicon() -> None:
    """Step 40.9: Signal Word Syllable Lexicon."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 40.9: Signal Word Syllable Lexicon")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load inputs ──
    print("\n  1. Loading inputs …")
    merged_signal = _safe_load(os.path.join(rd, 'merged_signal.json'))
    merged_bigrams = _safe_load(os.path.join(rd, 'merged_bigrams.json'))
    ven_lex = _safe_load(os.path.join(rd, 'venetian_lexicon.json'))

    signal_words = merged_signal.get('word_signals', [])
    print(f"    Signal words: {len(signal_words)}")

    # ── 2. Build lexicon ──
    print("\n  2. Building signal word lexicon …")
    lexicon = _build_signal_lexicon(signal_words, SIGNAL_WORD_GLOSSES)

    n_glossed = sum(1 for e in lexicon if e['english_gloss'] != '???')
    n_high = sum(1 for e in lexicon if e['confidence'] == 'HIGH')
    n_medium = sum(1 for e in lexicon if e['confidence'] == 'MEDIUM')
    n_low = sum(1 for e in lexicon if e['confidence'] == 'LOW')
    print(f"    Glossed: {n_glossed}/{len(lexicon)}")
    print(f"    HIGH confidence: {n_high}")
    print(f"    MEDIUM confidence: {n_medium}")
    print(f"    LOW confidence: {n_low}")

    # ── 3. POS distribution ──
    print("\n  3. POS distribution:")
    pos_dist = _pos_distribution(lexicon)
    anonimo_pos = _anonimo_pos_distribution()
    for pos, count in pos_dist.items():
        frac = count / len(lexicon) if lexicon else 0
        expected = anonimo_pos.get(pos, 0.0)
        print(f"    {pos}: {count} ({frac:.2f}) [expected {expected:.2f}]")

    # ── 4. Concatenated pair lexicon ──
    print("\n  4. Concatenated pair analysis:")
    concat_entries = []
    for word, info in CONCAT_LEXICON.items():
        # Check if both parts are signal words
        parts_in_signal = all(
            any(sw.get('word') == p for sw in signal_words)
            for p in info['parts']
        )
        concat_entries.append({
            'concatenated': word,
            'parts': info['parts'],
            'gloss': info['gloss'],
            'domain': info['domain'],
            'both_parts_signal': parts_in_signal,
        })
        if parts_in_signal:
            print(f"    {' + '.join(info['parts'])} = {word} ({info['gloss']})")

    # ── 5. Domain summary ──
    print("\n  5. Domain summary:")
    from collections import Counter as C2
    domain_counts = C2(e['medical_domain'] for e in lexicon)
    for domain, count in domain_counts.most_common():
        print(f"    {domain}: {count}")

    # ── 6. Save ──
    elapsed = time.time() - t0

    output = {
        'n_signal_words': len(signal_words),
        'n_glossed': n_glossed,
        'n_high_confidence': n_high,
        'n_medium_confidence': n_medium,
        'n_low_confidence': n_low,
        'syllable_lexicon': {e['decoded']: e for e in lexicon},
        'pos_distribution': pos_dist,
        'anonimo_pos_expected': anonimo_pos,
        'concat_lexicon': concat_entries,
        'domain_counts': dict(domain_counts),
        'runtime_seconds': round(elapsed, 1),
    }

    out_path = os.path.join(rd, 'syllable_lexicon.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(output), f, indent=2)
    print(f"\n  Saved → {out_path} ({elapsed:.1f}s)")
