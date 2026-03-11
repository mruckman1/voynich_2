"""
Step 41.4 – Definitive Venetian Signal Vocabulary
==================================================
Produce the project's final confirmed signal vocabulary with proper
Venetian null validation — annotated with glosses, domains, and
vocabulary coherence assessment.

Dependency chain:
    venetian_signal_proper.json  (Step 41.3 — proper σ-scores)
    syllable_lexicon.json        (Step 40.9 — 28 glossed words)
    venetian_forms.json          (Step 40.1 — Venetian set)
        → venetian_confirmed.json  (this step)
"""

import json
import os
import time
from collections import Counter
from typing import Any, Dict, List, Set

from voynich.core._paths import results_dir as _results_dir


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
# Gloss table from Phase 40.9
# ---------------------------------------------------------------------------

SIGNAL_WORD_GLOSSES = {
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


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def _build_confirmed_vocabulary(
    word_signals: List[Dict],
    gloss_table: Dict,
) -> List[Dict]:
    """Build confirmed vocabulary list with glosses and confidence."""
    confirmed = []

    for ws in word_signals:
        if not ws.get('is_genuine_signal', False):
            continue

        word = ws['word']
        sigma = ws['sigma']
        gloss_entry = gloss_table.get(word, {})

        if sigma >= 20:
            confidence = 'HIGH'
        elif sigma >= 5:
            confidence = 'MEDIUM'
        else:
            confidence = 'LOW'

        entry = {
            'decoded': word,
            'sigma': sigma,
            'selectivity': ws.get('selectivity', 0.0),
            'real_count': ws.get('real_count', 0),
            'confidence': confidence,
            'glossed': bool(gloss_entry),
            'english_gloss': gloss_entry.get('gloss', ''),
            'part_of_speech': gloss_entry.get('pos', ''),
            'domain': gloss_entry.get('domain', 'unknown'),
            'latin_equivalent': gloss_entry.get('latin', ''),
            'venetian_form': gloss_entry.get('venetian', ''),
        }
        confirmed.append(entry)

    return confirmed


def _vocabulary_coherence(confirmed: List[Dict]) -> Dict:
    """Assess whether the vocabulary is coherent with a pharmaceutical text."""
    domain_counts: Counter = Counter()
    pos_counts: Counter = Counter()

    for entry in confirmed:
        domain = entry.get('domain', 'unknown')
        pos = entry.get('part_of_speech', 'unknown')
        if not pos:
            pos = 'unknown'
        domain_counts[domain] += 1
        # Count primary POS only
        pos_primary = pos.split('/')[0]
        pos_counts[pos_primary] += 1

    n_total = len(confirmed)
    n_glossed = sum(1 for e in confirmed if e['glossed'])
    n_function = domain_counts.get('function', 0)
    n_pharma = domain_counts.get('pharmaceutical', 0)
    n_botanical = domain_counts.get('botanical', 0)
    n_anatomical = domain_counts.get('anatomical', 0)
    n_quality = domain_counts.get('quality', 0)
    n_medical = n_pharma + n_botanical + n_anatomical + n_quality

    # A pharmaceutical recipe text should have:
    # ~30-50% function words, ~30-50% medical terms, ~10-20% general
    medical_fraction = n_medical / n_total if n_total > 0 else 0.0
    function_fraction = n_function / n_total if n_total > 0 else 0.0

    coherent = (medical_fraction > 0.20 and function_fraction > 0.20
                and function_fraction < 0.70)

    return {
        'n_total': n_total,
        'n_glossed': n_glossed,
        'gloss_rate': round(n_glossed / n_total, 4) if n_total > 0 else 0.0,
        'domain_counts': dict(domain_counts),
        'pos_counts': dict(pos_counts),
        'medical_fraction': round(medical_fraction, 4),
        'function_fraction': round(function_fraction, 4),
        'coherent_with_pharmaceutical': coherent,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_venetian_confirmed() -> None:
    """Step 41.4: Produce definitive confirmed Venetian signal vocabulary."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 41.4: Definitive Venetian Signal Vocabulary")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load inputs ──
    print("\n  1. Loading inputs …")

    sig_proper = _safe_load(os.path.join(rd, 'venetian_signal_proper.json'))
    if not sig_proper:
        print("  [SKIP] venetian_signal_proper.json not found")
        return

    word_signals = sig_proper.get('word_signals', [])
    print(f"    Word signals loaded: {len(word_signals)}")

    # ── 2. Build confirmed vocabulary ──
    print("\n  2. Building confirmed vocabulary …")
    confirmed = _build_confirmed_vocabulary(word_signals, SIGNAL_WORD_GLOSSES)
    print(f"    Confirmed signal words: {len(confirmed)}")
    n_glossed = sum(1 for e in confirmed if e['glossed'])
    n_unglossed = len(confirmed) - n_glossed
    print(f"    Glossed: {n_glossed}")
    print(f"    Unglossed: {n_unglossed}")

    # ── 3. Vocabulary coherence ──
    print("\n  3. Assessing vocabulary coherence …")
    coherence = _vocabulary_coherence(confirmed)
    print(f"    Domain distribution:")
    for domain, count in sorted(coherence['domain_counts'].items(),
                                key=lambda x: -x[1]):
        print(f"      {domain}: {count}")
    print(f"    Medical fraction: {coherence['medical_fraction']:.4f}")
    print(f"    Function fraction: {coherence['function_fraction']:.4f}")
    print(f"    Coherent with pharmaceutical text: "
          f"{coherence['coherent_with_pharmaceutical']}")

    # ── 4. Print vocabulary table ──
    print("\n  4. Confirmed vocabulary (sorted by σ):")
    print(f"    {'#':>3s} {'Decoded':10s} {'σ':>8s} {'Count':>6s} "
          f"{'Conf':6s} {'Gloss':20s} {'Domain':15s}")
    print(f"    {'—' * 72}")
    for i, entry in enumerate(confirmed[:30], 1):
        gloss = entry['english_gloss'] or '—'
        print(f"    {i:3d} {entry['decoded']:10s} {entry['sigma']:8.1f} "
              f"{entry['real_count']:6d} {entry['confidence']:6s} "
              f"{gloss:20s} {entry['domain']:15s}")
    if len(confirmed) > 30:
        print(f"    … and {len(confirmed) - 30} more")

    # ── 5. Summary ──
    print("\n  5. Summary:")
    n_high = sum(1 for e in confirmed if e['confidence'] == 'HIGH')
    n_med = sum(1 for e in confirmed if e['confidence'] == 'MEDIUM')
    n_low = sum(1 for e in confirmed if e['confidence'] == 'LOW')
    print(f"    HIGH confidence (σ≥20): {n_high}")
    print(f"    MEDIUM confidence (5≤σ<20): {n_med}")
    print(f"    LOW confidence (2<σ<5): {n_low}")

    # ── 6. Save ──
    elapsed = time.time() - t0

    output = {
        'n_confirmed': len(confirmed),
        'n_glossed': n_glossed,
        'n_unglossed': n_unglossed,
        'n_high_confidence': n_high,
        'n_medium_confidence': n_med,
        'n_low_confidence': n_low,
        'vocabulary': confirmed,
        'coherence': coherence,
        'runtime_seconds': round(elapsed, 1),
    }

    out_path = os.path.join(rd, 'venetian_confirmed.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(output), f, indent=2)
    print(f"\n  Saved → {out_path} ({elapsed:.1f}s)")
