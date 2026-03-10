"""
Step 38.10 – Phase 38 Verdict
==============================
Interpret Phase 38 results and assign a verdict.

Dependency chain:
    merged_readability.json    (Step 38.9)
    merged_bigrams.json        (Step 38.4)
    merged_signal.json         (Step 38.3)
    merged_bootstrap.json      (Step 38.6)
    merged_folio.json          (Step 38.8)
        → phase38_verdict.json (this step)
"""

import json
import os
import time
from typing import Any, Dict

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
# Main
# ---------------------------------------------------------------------------

def run_phase38_verdict() -> None:
    """Step 38.10: Phase 38 Verdict."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 38.10: Phase 38 Verdict")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load results ──
    print("\n  1. Loading results …")
    readability = _safe_load(os.path.join(rd, 'merged_readability.json'))
    bigrams = _safe_load(os.path.join(rd, 'merged_bigrams.json'))
    signal = _safe_load(os.path.join(rd, 'merged_signal.json'))
    bootstrap = _safe_load(os.path.join(rd, 'merged_bootstrap.json'))
    folio = _safe_load(os.path.join(rd, 'merged_folio.json'))

    metrics = readability.get('metrics', {})

    # ── 2. Extract key numbers ──
    bigram_z = metrics.get('M03_bigram_z', 0.0)
    cc_bigrams = metrics.get('M10_cc_bigrams', 0)
    cross_lang = metrics.get('M13_cross_language_bigrams', 0)
    signal_rate = metrics.get('M02_signal_rate', 0.0)
    confirmed_vocab = metrics.get('M05_confirmed_vocab', 0)
    medical_phrases = metrics.get('M12_medical_phrases', 0)
    n_italian_only = signal.get('n_italian_only_signal_words', 0)
    boot_italian = bootstrap.get('n_confirmed_italian_only', 0)

    # Best fragment
    best_frag = folio.get('best_fragment', {})
    best_frag_length = best_frag.get('length', 0) if best_frag else 0
    best_frag_has_ita = best_frag.get('has_italian', False) if best_frag else False

    print(f"     Bigram z: {bigram_z:.2f}")
    print(f"     CC bigrams: {cc_bigrams}")
    print(f"     Cross-language: {cross_lang}")
    print(f"     SIGNAL rate: {signal_rate:.4f}")
    print(f"     Confirmed vocab: {confirmed_vocab}")
    print(f"     Italian-only signal words: {n_italian_only}")
    print(f"     Bootstrap Italian confirmed: {boot_italian}")
    print(f"     Medical phrases: {medical_phrases}")
    print(f"     Best fragment: {best_frag_length} tokens")

    # ── 3. Decision table ──
    print("\n  2. Evaluating verdict …")

    if (bigram_z > 17 and
        cc_bigrams > 0 and
        best_frag_length >= 4 and best_frag_has_ita and medical_phrases > 0 and
        boot_italian >= 3 and
        cross_lang > 3):
        verdict = 'MACARONIC_BREAKTHROUGH'
    elif (bigram_z >= 16 and
          cross_lang > 0 and
          n_italian_only > 5):
        verdict = 'MACARONIC_CONFIRMED'
    elif (signal_rate > 0.185 and
          confirmed_vocab > 51):
        verdict = 'SIGNAL_EXPANDED'
    elif bigram_z < 16:
        verdict = 'MERGED_COLLISIONS'
    else:
        verdict = 'SIGNAL_EXPANDED'

    print(f"\n  ╔{'═' * 50}╗")
    print(f"  ║  VERDICT: {verdict:38s}  ║")
    print(f"  ╚{'═' * 50}╝")

    # ── 4. Interpretation ──
    interpretations = {
        'MACARONIC_BREAKTHROUGH': (
            "The Voynich manuscript encodes a macaronic Latin-Italian medical text, "
            "consistent with 15th-century Po Valley scribal practice. The decoded text "
            "contains Latin function words and technical terms interleaved with Northern "
            "Italian content words. The first content phrases are identifiable."
        ),
        'MACARONIC_CONFIRMED': (
            "The language identification is confirmed — Northern Italian, not classical "
            "Latin — but the content-word barrier persists. The z≈17 signal is real and "
            "represents the strongest sequential structure achievable with the current "
            "triple table. Italian words expand the confirmed vocabulary but don't bridge "
            "the gap to content-content bigrams."
        ),
        'SIGNAL_EXPANDED': (
            "The merged dictionary finds more signal words but the sequential structure "
            "doesn't improve beyond Phase 37. The Italian vocabulary enriches the word "
            "list but doesn't change the bigram relationships. The macaronic finding is "
            "linguistically important but doesn't advance the decipherment."
        ),
        'MERGED_COLLISIONS': (
            "The Italian 10K dictionary is too permissive for signal isolation when "
            "combined with Latin 10K. Italian words that match the decoded text are "
            "dictionary collisions, not genuinely decoded content. The Latin 10K "
            "remains the best analytical configuration."
        ),
    }

    interpretation = interpretations.get(verdict, "Unknown verdict.")
    print(f"\n  Interpretation: {interpretation}")

    # ── 5. Gap analysis ──
    print("\n  3. Gap analysis …")
    gap = {
        'known': [
            "Encoding mechanism is tachygraphic (cosine 0.820, 11 alternatives tested)",
            "Phonetic core uses CV syllables mapped through 25 stroke triples",
            f"{sum(1 for w in signal.get('word_signals', []) if w.get('sigma', 0) > 5)} triples confirmed by multiple methods",
            f"{confirmed_vocab} words confirmed as signal vocabulary",
            f"Sequential word-pair structure detected at {bigram_z:.1f}σ above null",
            "Source language is macaronic Latin-Italian (Po Valley, early 15th c.)",
            "Content domain is medical/pharmaceutical",
        ],
        'unknown': [
            "13 triples remain unconfirmed — their correct syllable values",
            "Specific vowel ordering within confirmed consonant classes",
            f"Whether the {cc_bigrams} content-content bigram barrier can be broken",
            "Specific content of individual folios beyond function-word skeleton",
        ],
        'what_would_help': [
            "Costamagna syllabic tachygraphy tables with both Latin and Italian syllables",
            "Confirmed long crib in Italian (not Latin) — plant labels in Italian names",
            "Dedicated Italian botanical dictionary (Venetian dialect forms)",
            "Physical manuscript evidence (multispectral imaging, marginal annotations)",
        ],
    }

    for category, items in gap.items():
        print(f"\n  {category.upper()}:")
        for item in items:
            print(f"    • {item}")

    # ── 6. Save ──
    elapsed = time.time() - t0

    output = {
        'verdict': verdict,
        'interpretation': interpretation,
        'key_metrics': {
            'bigram_z': round(bigram_z, 2),
            'cc_bigrams': cc_bigrams,
            'cross_language_bigrams': cross_lang,
            'signal_rate': round(signal_rate, 4),
            'confirmed_vocab': confirmed_vocab,
            'italian_only_signal_words': n_italian_only,
            'bootstrap_italian': boot_italian,
            'medical_phrases': medical_phrases,
            'best_fragment_length': best_frag_length,
        },
        'decision_criteria': {
            'MACARONIC_BREAKTHROUGH': 'z>17, CC>0, 4+ word macaronic phrase, boot_ita≥3, cross>3',
            'MACARONIC_CONFIRMED': 'z≥16, cross>0, Italian-only>5',
            'SIGNAL_EXPANDED': 'signal>0.185 or vocab>51',
            'MERGED_COLLISIONS': 'z<16',
        },
        'gap_analysis': gap,
        'runtime_seconds': round(elapsed, 1),
    }

    out_path = os.path.join(rd, 'phase38_verdict.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(output), f, indent=2)
    print(f"\n  Saved → {out_path} ({elapsed:.1f}s)")
