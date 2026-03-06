"""
Phase 20.5 – Tachygraphic Readability Assessment
=================================================
Assess whether decoded text is recognisable Latin by testing bigram
plausibility, POS validity, domain coherence, and phrase patterns — all
compared against null baselines.

Dependency chain:
    tachy_decode.json + Latin reference corpus
        → tachy_readability.json
"""

import json
import os
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.reference import (
    LATIN_PHRASE_PATTERNS,
    PHARMACEUTICAL_VOCABULARY,
    build_expanded_word_set,
    load_reference_corpus,
)
from voynich.core.stats import (
    build_ngram_lm,
    cross_entropy_lm,
    selectivity_ratio,
)


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
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TachyReadabilityResult:
    # Bigram analysis
    bigram_plausibility: float
    bigram_null_expected: float
    bigram_selectivity: float
    # Cross-entropy
    cross_entropy_decoded: float
    cross_entropy_ref: float
    ce_ratio: float
    # POS analysis
    pos_distribution: Dict[str, int]
    pos_trigram_validity: float
    pos_null_validity: float
    pos_selectivity: float
    # Domain coherence
    domain_hits: Dict[str, Dict]
    n_domains_with_hits: int
    # Phrase patterns
    phrase_hits: List[Dict]
    n_phrase_hits: int
    # Overall
    n_tests: int
    n_tests_passing: int
    test_results: List[Dict]
    gate_passed: bool
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _load_json(rd: str, fname: str) -> Dict:
    path = os.path.join(rd, fname)
    if not os.path.exists(path):
        print(f"    [WARN] {fname} not found")
        return {}
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Test 1: Bigram plausibility
# ---------------------------------------------------------------------------

def _bigram_plausibility(
    decoded_words: List[str],
    ref_bigrams: set,
) -> float:
    """Fraction of consecutive decoded word pairs found in reference bigrams."""
    if len(decoded_words) < 2:
        return 0.0
    hits = 0
    total = 0
    for i in range(len(decoded_words) - 1):
        bg = (decoded_words[i], decoded_words[i + 1])
        if bg in ref_bigrams:
            hits += 1
        total += 1
    return hits / total if total else 0.0


def _build_ref_bigrams(ref_words: List[str]) -> set:
    """Build set of word bigrams from reference corpus."""
    bigrams = set()
    for i in range(len(ref_words) - 1):
        bigrams.add((ref_words[i].lower(), ref_words[i + 1].lower()))
    return bigrams


# ---------------------------------------------------------------------------
# Test 2: POS sequence validity
# ---------------------------------------------------------------------------

# Simple Latin POS tagger based on word endings
_LATIN_POS_RULES = [
    # Verbs
    (lambda w: w.endswith(('are', 'ere', 'ire', 'ari', 'eri', 'iri')), 'VERB'),
    (lambda w: w.endswith(('at', 'et', 'it', 'ant', 'ent', 'unt')), 'VERB'),
    (lambda w: w.endswith(('atur', 'etur', 'itur')), 'VERB'),
    # Prepositions
    (lambda w: w in ('in', 'de', 'ad', 'ex', 'per', 'cum', 'pro', 'sub',
                      'super', 'contra', 'inter'), 'PREP'),
    # Conjunctions
    (lambda w: w in ('et', 'sed', 'aut', 'vel', 'atque', 'quod', 'quia',
                      'si', 'nec', 'neque'), 'CONJ'),
    # Adjectives
    (lambda w: w.endswith(('us', 'a', 'um', 'is', 'e', 'ius', 'ior')), 'ADJ'),
    # Nouns (default for longer words)
    (lambda w: w.endswith(('ae', 'arum', 'orum', 'ibus', 'ium')), 'NOUN'),
    (lambda w: len(w) >= 4, 'NOUN'),
]


def _pos_tag(word: str) -> str:
    """Simple rule-based Latin POS tag."""
    w = word.lower()
    for rule, tag in _LATIN_POS_RULES:
        if rule(w):
            return tag
    return 'NOUN'  # default


def _pos_trigram_validity(
    decoded_words: List[str],
    ref_pos_trigrams: set,
) -> float:
    """Fraction of POS trigrams in decoded text that match reference."""
    if len(decoded_words) < 3:
        return 0.0
    tags = [_pos_tag(w) for w in decoded_words]
    hits = 0
    total = 0
    for i in range(len(tags) - 2):
        tri = (tags[i], tags[i + 1], tags[i + 2])
        if tri in ref_pos_trigrams:
            hits += 1
        total += 1
    return hits / total if total else 0.0


def _build_ref_pos_trigrams(ref_words: List[str]) -> set:
    """Build POS trigram set from reference words."""
    tags = [_pos_tag(w.lower()) for w in ref_words]
    trigrams = set()
    for i in range(len(tags) - 2):
        trigrams.add((tags[i], tags[i + 1], tags[i + 2]))
    return trigrams


# ---------------------------------------------------------------------------
# Test 3: Domain coherence
# ---------------------------------------------------------------------------

def _domain_coherence(
    decoded_words: List[str],
    pharma_vocab: Dict[str, List[str]],
) -> Dict[str, Dict]:
    """Per-domain hit rates for pharmaceutical vocabulary categories."""
    word_set = set(decoded_words)
    results = {}
    for domain, terms in pharma_vocab.items():
        term_set = set(t.lower() for t in terms)
        hits = word_set & term_set
        results[domain] = {
            'n_terms': len(term_set),
            'n_hits': len(hits),
            'hit_rate': len(hits) / len(term_set) if term_set else 0.0,
            'matched_terms': sorted(hits),
        }
    return results


# ---------------------------------------------------------------------------
# Test 4: Phrase patterns
# ---------------------------------------------------------------------------

def _detect_phrases(
    decoded_words: List[str],
    phrase_patterns: List[Tuple[str, List[str]]],
) -> List[Dict]:
    """Detect known Latin medical phrases in decoded text."""
    hits = []
    text = ' '.join(decoded_words)

    for pattern_name, templates in phrase_patterns:
        for template in templates:
            template_lower = template.lower()
            if template_lower in text:
                # Find position
                idx = text.index(template_lower)
                hits.append({
                    'pattern': pattern_name,
                    'template': template,
                    'position': idx,
                    'match_type': 'exact',
                })

    return hits


# ---------------------------------------------------------------------------
# Null comparison
# ---------------------------------------------------------------------------

def _null_bigram_plausibility(
    ref_word_set: set,
    ref_bigrams: set,
    n_words: int,
    n_trials: int = 10,
) -> float:
    """Expected bigram plausibility for random dictionary words."""
    import random
    rng = random.Random(42)
    word_list = sorted(ref_word_set)
    if not word_list:
        return 0.0

    scores = []
    for _ in range(n_trials):
        sample = [rng.choice(word_list) for _ in range(n_words)]
        score = _bigram_plausibility(sample, ref_bigrams)
        scores.append(score)
    return sum(scores) / len(scores)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_tachy_readability() -> None:
    """Step 20.5: Readability assessment of tachygraphic decode."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE 20.5: Tachygraphic Readability Assessment")
    print("=" * 70)

    rd = _results_dir()

    # ─── 1. Load decoded text ───
    print("\n  1. Loading decoded text …")
    decode_data = _load_json(rd, 'tachy_decode.json')

    # Extract decoded words from the sample and top words
    decoded_words: List[str] = []
    for entry in decode_data.get('decoded_sample', []):
        if len(entry) >= 2:
            decoded = entry[1]
            if decoded and decoded != '?':
                decoded_words.append(decoded.lower())

    # Also get the full top decoded words
    top_words = decode_data.get('top_decoded_words', [])

    # Build reference
    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    ref_tokens = ref_corpus.get_combined_tokens('latin')
    ref_words = [w.lower() for w in ref_tokens if len(w) >= 2]
    base_words = set(ref_words)
    expanded_words, _ = build_expanded_word_set(base_words)
    ref_word_set = base_words | expanded_words

    # Filter decoded to dict hits for meaningful analysis
    dict_decoded = [w for w in decoded_words if w in ref_word_set]
    print(f"      Decoded words (sample): {len(decoded_words)}")
    print(f"      Dict-hitting words: {len(dict_decoded)}")

    # ─── 2. Test 1: Bigram plausibility ───
    print("\n  2. Test 1: Bigram plausibility …")
    ref_bigrams = _build_ref_bigrams(ref_words[:50000])
    bg_plausibility = _bigram_plausibility(dict_decoded, ref_bigrams)
    bg_null = _null_bigram_plausibility(
        ref_word_set, ref_bigrams, len(dict_decoded))
    bg_selectivity = bg_plausibility / bg_null if bg_null > 0 else float('inf')
    print(f"      Bigram plausibility: {bg_plausibility:.3f}")
    print(f"      Null expected: {bg_null:.3f}")
    print(f"      Selectivity: {bg_selectivity:.2f}×")

    # ─── 3. Test 2: Cross-entropy ───
    print("\n  3. Test 2: Cross-entropy …")
    ref_text = ' '.join(ref_words[:20000])
    lm = build_ngram_lm(list(ref_text), order=3)
    decoded_text = ' '.join(dict_decoded)
    ce_decoded = cross_entropy_lm(list(decoded_text), lm) if decoded_text else 99.0
    ce_ref = cross_entropy_lm(list(ref_text[:len(decoded_text)]), lm) if decoded_text else 1.0
    ce_ratio = ce_decoded / ce_ref if ce_ref > 0 else float('inf')
    print(f"      CE (decoded): {ce_decoded:.4f}")
    print(f"      CE (reference): {ce_ref:.4f}")
    print(f"      Ratio: {ce_ratio:.2f}")

    # ─── 4. Test 3: POS trigram validity ───
    print("\n  4. Test 3: POS trigram validity …")
    ref_pos_trigrams = _build_ref_pos_trigrams(ref_words[:20000])
    pos_validity = _pos_trigram_validity(dict_decoded, ref_pos_trigrams)

    # Null: random dict words
    import random as _rng
    _rng.seed(42)
    null_words = [_rng.choice(sorted(ref_word_set)) for _ in range(len(dict_decoded))]
    pos_null = _pos_trigram_validity(null_words, ref_pos_trigrams)
    pos_sel = pos_validity / pos_null if pos_null > 0 else float('inf')
    print(f"      POS validity: {pos_validity:.3f}")
    print(f"      Null validity: {pos_null:.3f}")
    print(f"      Selectivity: {pos_sel:.2f}×")

    # POS distribution
    pos_dist = Counter(_pos_tag(w) for w in dict_decoded)
    print(f"      POS distribution: {dict(pos_dist)}")

    # ─── 5. Test 4: Domain coherence ───
    print("\n  5. Test 4: Domain coherence …")
    domain_results = _domain_coherence(
        [w for w, _ in top_words],
        PHARMACEUTICAL_VOCABULARY,
    )
    n_domains_with_hits = sum(
        1 for d in domain_results.values() if d['n_hits'] > 0
    )
    for domain, info in domain_results.items():
        if info['n_hits'] > 0:
            print(f"      {domain}: {info['n_hits']} hits — "
                  f"{info['matched_terms'][:5]}")
    print(f"      Domains with hits: {n_domains_with_hits}")

    # ─── 6. Test 5: Phrase patterns ───
    print("\n  6. Test 5: Phrase patterns …")
    phrase_hits = _detect_phrases(dict_decoded, LATIN_PHRASE_PATTERNS)
    n_phrase_hits = len(phrase_hits)
    for ph in phrase_hits[:10]:
        print(f"      [{ph['pattern']}] '{ph['template']}' at pos {ph['position']}")
    print(f"      Total phrase hits: {n_phrase_hits}")

    # ─── 7. Compile test results ───
    print("\n  7. Compiling results …")
    tests = [
        {'name': 'bigram_plausibility', 'value': bg_plausibility,
         'threshold': 0.01, 'passed': bg_selectivity > 1.5,
         'selectivity': bg_selectivity},
        {'name': 'cross_entropy_ratio', 'value': ce_ratio,
         'threshold': 3.0, 'passed': ce_ratio < 3.0,
         'selectivity': 1.0 / ce_ratio if ce_ratio > 0 else 0.0},
        {'name': 'pos_validity', 'value': pos_validity,
         'threshold': 0.1, 'passed': pos_sel > 1.3,
         'selectivity': pos_sel},
        {'name': 'domain_coherence', 'value': float(n_domains_with_hits),
         'threshold': 3.0, 'passed': n_domains_with_hits >= 3,
         'selectivity': float(n_domains_with_hits)},
        {'name': 'phrase_detection', 'value': float(n_phrase_hits),
         'threshold': 1.0, 'passed': n_phrase_hits >= 1,
         'selectivity': float(n_phrase_hits)},
    ]

    n_passing = sum(1 for t in tests if t['passed'])
    gate_passed = n_passing >= 3

    for t in tests:
        status = 'PASS' if t['passed'] else 'FAIL'
        print(f"      {t['name']:25s}: {status} "
              f"(value={t['value']:.3f}, sel={t['selectivity']:.2f})")

    if gate_passed:
        verdict = (f"PASS: {n_passing}/5 readability tests pass. "
                   f"Bigram={bg_plausibility:.3f}, "
                   f"phrases={n_phrase_hits}.")
    else:
        verdict = (f"FAIL: Only {n_passing}/5 tests pass (need ≥3). "
                   f"Bigram={bg_plausibility:.3f}.")

    print(f"\n  8. Gate: {verdict}")

    # ─── 8. Save ───
    result = TachyReadabilityResult(
        bigram_plausibility=bg_plausibility,
        bigram_null_expected=bg_null,
        bigram_selectivity=bg_selectivity,
        cross_entropy_decoded=ce_decoded,
        cross_entropy_ref=ce_ref,
        ce_ratio=ce_ratio,
        pos_distribution=dict(pos_dist),
        pos_trigram_validity=pos_validity,
        pos_null_validity=pos_null,
        pos_selectivity=pos_sel,
        domain_hits=domain_results,
        n_domains_with_hits=n_domains_with_hits,
        phrase_hits=phrase_hits,
        n_phrase_hits=n_phrase_hits,
        n_tests=len(tests),
        n_tests_passing=n_passing,
        test_results=tests,
        gate_passed=gate_passed,
        verdict=verdict,
        runtime_seconds=time.time() - t0,
    )

    out_path = os.path.join(rd, 'tachy_readability.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)

    print(f"\n  → {out_path}")
