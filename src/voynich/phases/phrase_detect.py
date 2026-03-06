"""
Phase C.3 -- Phrase Detection on Tironian-Informed Decode
==========================================================
Scan the Tironian-CSP decoded output for multi-word Latin sequences
that form valid pharmaceutical, prepositional, adjectival, genitive,
or conjunction collocations.  This is the PRIMARY SUCCESS CRITERION
for Phase C: if genuine multi-word phrases appear above chance, the
decoding is linguistically meaningful.

Null control: 10 random assignments from the Phase 14 phonotactic
domain pool are decoded and phrase-scanned to establish a baseline.

Dependency chain:
    results/tironian_csp.json  (Phase C.1-C.2)
        -> phrase_detect.json  (this step)
"""

import json
import os
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    load_corpus,
    tokenize_eva_chars,
)
from voynich.core.reference import (
    EVA_VISUAL_COMPONENTS,
    build_cv_syllable_table,
    load_reference_corpus,
)
from voynich.core.stats import compute_phrase_selectivity
from voynich.phases.csp_constraints import build_phoneme_inventory
from voynich.phases.csp_solver import decode_corpus, decode_token


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
    if isinstance(obj, float) and (obj != obj):
        return None
    if isinstance(obj, (bool, int, float, str, type(None))):
        return obj
    return str(obj)


# ---------------------------------------------------------------------------
# Word lists for phrase pattern detection
# ---------------------------------------------------------------------------

_IMPERATIVES = {
    'recipe', 'accipe', 'misce', 'cola', 'tera', 'solve',
    'distilla', 'lava', 'coque', 'adde', 'tere', 'cape',
    'pone', 'da', 'fac', 'applica', 'bibe',
}

_PREPOSITIONS = {
    'cum', 'in', 'ad', 'de', 'per', 'pro', 'sub',
    'super', 'ante', 'ex', 'ab', 'sine',
}

_ADJECTIVES = {
    # masc/fem/neut forms included
    'calidus', 'calida', 'calidum',
    'frigidus', 'frigida', 'frigidum',
    'siccus', 'sicca', 'siccum',
    'humidus', 'humida', 'humidum',
    'niger', 'nigra', 'nigrum',
    'albus', 'alba', 'album',
    'dulcis', 'dulce',
    'amarus', 'amara', 'amarum',
    'mollis', 'molle',
    'viridis', 'viride',
    'ruber', 'rubra', 'rubrum',
}

_NOUNS = {
    'aqua', 'radix', 'folia', 'flos', 'semen', 'cortex',
    'vinum', 'oleum', 'herba', 'pulvis', 'mel', 'acetum',
    'succus', 'rosa', 'viola', 'salvia', 'menta', 'ruta',
    'caulis', 'fructus', 'folium', 'flores', 'semina',
}

_GENITIVE_NOUNS = {
    'viole', 'rose', 'salvie', 'mente', 'rute',
    'vini', 'olei', 'mellis', 'aceti',
    'radicis', 'floris', 'seminis', 'corticis',
    'aque', 'herbe', 'pulveris',
}

_CONJUNCTIONS = {'et', 'ac', 'atque', 'vel', 'aut', 'nec', 'sed'}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PhraseDetectResult:
    """Phase C.3: Phrase detection results."""
    n_phrases_detected: int
    phrases: List[Dict]
    pattern_type_counts: Dict[str, int]
    null_phrase_counts: List[int]
    phrase_selectivity: float
    p_value: float
    gate_passed: bool
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Phrase scanning
# ---------------------------------------------------------------------------

def _scan_phrases(
    decoded_tokens: List[str],
    folio_tokens: Optional[List[Tuple[str, str]]] = None,
) -> Tuple[List[Dict], Dict[str, int]]:
    """Scan adjacent decoded tokens for Latin phrase patterns.

    Parameters
    ----------
    decoded_tokens : list of str
        Decoded word list (one per Voynich token).
    folio_tokens : list of (folio, token) pairs, optional
        For position tracking.

    Returns
    -------
    (phrases, pattern_type_counts)
    """
    phrases: List[Dict] = []
    counts: Dict[str, int] = {
        'pharmaceutical': 0,
        'prepositional': 0,
        'adjectival': 0,
        'genitive': 0,
        'conjunction': 0,
    }

    n = len(decoded_tokens)
    for i in range(n - 1):
        w1 = decoded_tokens[i].lower()
        w2 = decoded_tokens[i + 1].lower()

        folio = ''
        if folio_tokens and i < len(folio_tokens):
            folio = folio_tokens[i][0]

        pattern = None

        # Pharmaceutical: imperative + noun
        if w1 in _IMPERATIVES and w2 in _NOUNS:
            pattern = 'pharmaceutical'

        # Prepositional: preposition + noun
        elif w1 in _PREPOSITIONS and w2 in _NOUNS:
            pattern = 'prepositional'

        # Adjectival: adjective + noun (or noun + adjective)
        elif (w1 in _ADJECTIVES and w2 in _NOUNS) or \
             (w1 in _NOUNS and w2 in _ADJECTIVES):
            pattern = 'adjectival'

        # Genitive: noun + genitive noun
        elif w1 in _NOUNS and w2 in _GENITIVE_NOUNS:
            pattern = 'genitive'

        # Conjunction: conjunction + any known word
        elif w1 in _CONJUNCTIONS and (
            w2 in _NOUNS or w2 in _ADJECTIVES or w2 in _IMPERATIVES
            or w2 in _PREPOSITIONS
        ):
            pattern = 'conjunction'

        if pattern:
            phrases.append({
                'position': i,
                'decoded_words': [w1, w2],
                'pattern_type': pattern,
                'folio': folio,
            })
            counts[pattern] += 1

    return phrases, counts


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_phrase_detect() -> None:
    """Phase C.3: Phrase detection on Tironian-informed decode."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE C.3: Phrase Detection (Primary Success Criterion)")
    print("=" * 70)

    rd = _results_dir()

    # ------------------------------------------------------------------ 1
    print("\n  1. Loading tironian_csp.json for best assignment ...")
    tir_path = os.path.join(rd, 'tironian_csp.json')
    if not os.path.exists(tir_path):
        print("    [SKIP] tironian_csp.json not found -- run tironian-csp first")
        return

    with open(tir_path) as f:
        tir_data = json.load(f)
    best_assignment = tir_data.get('best_assignment', {})
    if not best_assignment:
        print("    [SKIP] No assignment found in tironian_csp.json")
        return
    print(f"    Loaded assignment with {len(best_assignment)} mappings")

    # ------------------------------------------------------------------ 2
    print("\n  2. Loading corpus and decoding all Language A tokens ...")
    corpus = load_corpus(verbose=False)
    tokens = corpus.get_tokens(language='A', paragraph_only=True)
    if not tokens:
        print("    [SKIP] No Language A tokens found")
        return

    eva_to_triple = build_eva_to_triple_lookup()
    decoded_all = decode_corpus(tokens, best_assignment, eva_to_triple, max_tokens=len(tokens))
    print(f"    Decoded {len(decoded_all)} tokens")

    # ------------------------------------------------------------------ 3-5
    print("\n  3. Scanning for multi-word Latin phrase patterns ...")
    print("    Pattern types: pharmaceutical, prepositional, adjectival, genitive, conjunction")

    phrases, pattern_counts = _scan_phrases(decoded_all)
    n_phrases = len(phrases)

    print(f"\n  4. Results:")
    print(f"    Total phrases detected: {n_phrases}")
    for ptype, count in sorted(pattern_counts.items(), key=lambda x: -x[1]):
        print(f"      {ptype:<20} {count}")

    if phrases:
        print(f"\n  5. Detected phrases (first 20):")
        for ph in phrases[:20]:
            print(f"    pos={ph['position']:>5}  "
                  f"{' '.join(ph['decoded_words']):<25}  "
                  f"[{ph['pattern_type']}]")

    # ------------------------------------------------------------------ 6
    print("\n  6. Running null baseline (10 random assignments) ...")
    rng = random.Random(42)

    # Load stroke feature data to get variable domains
    sf_path = os.path.join(rd, 'stroke_features.json')
    if os.path.exists(sf_path):
        with open(sf_path) as f:
            sf_data = json.load(f)
        triple_keys = [t['triple_key'] for t in sf_data.get('triples', [])]
    else:
        triple_keys = list(best_assignment.keys())

    # Build domain from phonotactic inventory
    ref_corpus = load_reference_corpus(verbose=False)
    inventory = build_phoneme_inventory('latin', ref_corpus)
    all_syls = list(inventory.cv_syllables)

    null_phrase_counts: List[int] = []
    for trial in range(10):
        rand_map = {tk: rng.choice(all_syls) for tk in triple_keys}
        rand_decoded = decode_corpus(tokens, rand_map, eva_to_triple, max_tokens=len(tokens))
        rand_phrases, _ = _scan_phrases(rand_decoded)
        null_phrase_counts.append(len(rand_phrases))
        print(f"    Trial {trial + 1:>2}: {len(rand_phrases)} phrases")

    # ------------------------------------------------------------------ 7
    print("\n  7. Computing phrase selectivity ...")
    sel_result = compute_phrase_selectivity(n_phrases, null_phrase_counts)
    phrase_selectivity = sel_result.get('selectivity', 0.0)
    p_value = sel_result.get('p_value', 1.0)

    print(f"    Real phrases:    {n_phrases}")
    null_mean = sel_result.get('null_mean', 0.0)
    null_std = sel_result.get('null_std', 0.0)
    print(f"    Null mean:       {null_mean:.1f} +/- {null_std:.1f}")
    print(f"    Selectivity:     {phrase_selectivity:.2f}x")
    print(f"    p-value:         {p_value:.4f}")

    # ------------------------------------------------------------------ 8
    # Gate: n_phrases >= 3 AND phrase_selectivity > 2.0
    gate_passed = n_phrases >= 3 and phrase_selectivity > 2.0

    if gate_passed:
        verdict = (
            f"PASS: {n_phrases} phrases detected ({phrase_selectivity:.2f}x selectivity, "
            f"p={p_value:.4f}). Multi-word Latin collocations appear above chance. "
            f"Dominant pattern: {max(pattern_counts, key=pattern_counts.get) if any(v > 0 for v in pattern_counts.values()) else 'none'}."
        )
    else:
        reasons = []
        if n_phrases < 3:
            reasons.append(f"only {n_phrases} phrases (need >= 3)")
        if phrase_selectivity <= 2.0:
            reasons.append(f"selectivity {phrase_selectivity:.2f}x (need > 2.0)")
        verdict = (
            f"FAIL: {'; '.join(reasons)}. "
            f"Decoded text does not show statistically significant phrase structure."
        )

    print(f"\n  Gate: {'PASS' if gate_passed else 'FAIL'}")
    print(f"  Verdict: {verdict}")

    # Save
    result = PhraseDetectResult(
        n_phrases_detected=n_phrases,
        phrases=phrases,
        pattern_type_counts=pattern_counts,
        null_phrase_counts=null_phrase_counts,
        phrase_selectivity=round(phrase_selectivity, 4),
        p_value=round(p_value, 4),
        gate_passed=gate_passed,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'phrase_detect.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)

    print(f"\n  Results saved -> {out_path}")
