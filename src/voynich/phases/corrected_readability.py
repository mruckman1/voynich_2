"""
Step 24.7 – Readability Battery on Corrected Decode
=====================================================
Runs the full 5-test readability battery on the corrected decode output
from Step 24.6.  Compares to Phase 16 readability baselines and null
distributions to assess whether the corrected table produces more
linguistically coherent output.

Dependency chain:
    corrected_decode.json (Step 24.6)
    readability_22.json or readability_delta.json (Phase 22/23)
        → corrected_readability.json (this step)
"""

import json
import os
import random
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
from voynich.core.stats import build_ngram_lm, cross_entropy_lm


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


def _load_json(path: str) -> Optional[Dict]:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Readability tests (adapted from readability_22.py)
# ---------------------------------------------------------------------------

def _build_ref_bigrams(ref_words: List[str]) -> set:
    return {(ref_words[i].lower(), ref_words[i + 1].lower())
            for i in range(len(ref_words) - 1)}


def _bigram_plausibility(decoded_words: List[str], ref_bigrams: set) -> float:
    if len(decoded_words) < 2:
        return 0.0
    hits = sum(1 for i in range(len(decoded_words) - 1)
               if (decoded_words[i], decoded_words[i + 1]) in ref_bigrams)
    return hits / (len(decoded_words) - 1)


_LATIN_POS_RULES = [
    (lambda w: w.endswith(('are', 'ere', 'ire', 'ari', 'eri', 'iri')), 'VERB'),
    (lambda w: w.endswith(('at', 'et', 'it', 'ant', 'ent', 'unt')), 'VERB'),
    (lambda w: w.endswith(('atur', 'etur', 'itur')), 'VERB'),
    (lambda w: w in ('in', 'de', 'ad', 'ex', 'per', 'cum', 'pro', 'sub',
                      'super', 'contra', 'inter'), 'PREP'),
    (lambda w: w in ('et', 'sed', 'aut', 'vel', 'atque', 'quod', 'quia',
                      'si', 'nec', 'neque'), 'CONJ'),
    (lambda w: w.endswith(('us', 'a', 'um', 'is', 'e', 'ius', 'ior')), 'ADJ'),
    (lambda w: w.endswith(('ae', 'arum', 'orum', 'ibus', 'ium')), 'NOUN'),
    (lambda w: len(w) >= 4, 'NOUN'),
]


def _pos_tag(word: str) -> str:
    w = word.lower()
    for rule, tag in _LATIN_POS_RULES:
        if rule(w):
            return tag
    return 'NOUN'


def _pos_trigram_validity(decoded_words: List[str], ref_pos_trigrams: set) -> float:
    if len(decoded_words) < 3:
        return 0.0
    tags = [_pos_tag(w) for w in decoded_words]
    hits = sum(1 for i in range(len(tags) - 2)
               if (tags[i], tags[i + 1], tags[i + 2]) in ref_pos_trigrams)
    total = len(tags) - 2
    return hits / total if total else 0.0


def _build_ref_pos_trigrams(ref_words: List[str]) -> set:
    tags = [_pos_tag(w.lower()) for w in ref_words]
    return {(tags[i], tags[i + 1], tags[i + 2]) for i in range(len(tags) - 2)}


def _domain_coherence(decoded_words: List[str], pharma_vocab: Dict) -> Dict[str, Dict]:
    word_set = set(w.lower() for w in decoded_words)
    results = {}
    for domain, terms in pharma_vocab.items():
        term_set = set(t.lower() for t in terms)
        hits = word_set & term_set
        results[domain] = {
            'n_terms': len(term_set),
            'n_hits': len(hits),
            'hit_rate': len(hits) / max(len(term_set), 1),
            'matched_terms': sorted(hits),
        }
    return results


def _detect_phrases(decoded_words: List[str], phrase_patterns) -> List[Dict]:
    text = ' '.join(decoded_words)
    hits = []
    for pattern_name, templates in phrase_patterns:
        for template in templates:
            if template.lower() in text:
                idx = text.index(template.lower())
                hits.append({
                    'pattern': pattern_name,
                    'template': template,
                    'position': idx,
                })
    return hits


def _null_bigram_plausibility(ref_word_set: set, ref_bigrams: set,
                               n_words: int, n_trials: int = 10) -> float:
    rng = random.Random(42)
    word_list = sorted(ref_word_set)
    if not word_list or n_words < 2:
        return 0.0
    scores = []
    for _ in range(n_trials):
        sample = [rng.choice(word_list) for _ in range(n_words)]
        scores.append(_bigram_plausibility(sample, ref_bigrams))
    return sum(scores) / len(scores)


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class CorrectedReadabilityResult:
    timestamp: str
    table_source: str
    # Readability tests
    bigram_plausibility: float
    bigram_null: float
    bigram_selectivity: float
    cross_entropy_decoded: float
    cross_entropy_ref: float
    ce_ratio: float
    pos_trigram_validity: float
    pos_null_validity: float
    pos_selectivity: float
    domain_hits: Dict[str, Dict]
    n_domains_with_hits: int
    phrase_hits: List[Dict]
    n_phrase_hits: int
    # Test summary
    n_tests_passing: int
    test_results: List[Dict]
    # Comparison
    phase16_bigram: float
    phase16_n_tests: int
    improvement_over_phase16: bool
    # Verdict
    gate_passed: bool
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_corrected_readability() -> None:
    """Step 24.7: Readability battery on corrected decode."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 24.7: Readability Battery on Corrected Decode")
    print("=" * 70)

    rdir = _results_dir()

    # ─── 1. Load corrected decode results ───
    print("\n  1. Loading corrected decode results …")
    t1 = time.time()

    decode_data = _load_json(str(rdir / "corrected_decode.json"))
    if decode_data is None:
        print("  [SKIP] corrected_decode.json not found — run corrected-decode first")
        return

    table_source = decode_data.get("table_source", "unknown")
    decoded_sample = decode_data.get("decoded_sample", [])
    dict_hit_rate = decode_data.get("dict_hit_rate", 0.0)

    # Extract decoded words from the sample
    decoded_words_all = [entry[1].lower() for entry in decoded_sample
                         if len(entry) >= 2 and entry[1] and '?' not in entry[1]]

    print(f"     Table source: {table_source}")
    print(f"     Decoded sample size: {len(decoded_words_all)} words")
    print(f"     Dict-hit rate: {dict_hit_rate:.1%}")
    print(f"     ({time.time() - t1:.1f}s)")

    # ─── 2. Build reference resources ───
    print("\n  2. Building reference resources …")
    t2 = time.time()

    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    ref_tokens = ref_corpus.get_combined_tokens('latin')
    ref_words = [w.lower() for w in ref_tokens if len(w) >= 2]

    base_words = set(ref_words)
    expanded_words, _ = build_expanded_word_set(base_words)
    ref_word_set = base_words | expanded_words

    ref_bigrams = _build_ref_bigrams(ref_words[:50000])
    ref_pos_trigrams = _build_ref_pos_trigrams(ref_words[:20000])

    # Build LM for cross-entropy
    ref_text = ' '.join(ref_words[:20000])
    lm = build_ngram_lm(list(ref_text), order=3)
    ce_ref = cross_entropy_lm(list(ref_text[:5000]), lm)

    print(f"     {len(ref_word_set)} reference words")
    print(f"     {len(ref_bigrams)} reference bigrams")
    print(f"     {len(ref_pos_trigrams)} POS trigrams")
    print(f"     Reference cross-entropy: {ce_ref:.4f}")
    print(f"     ({time.time() - t2:.1f}s)")

    # ─── 3. Filter to dict-hitting words for analysis ───
    print("\n  3. Preparing analysis words …")
    t3 = time.time()

    dict_words = [w for w in decoded_words_all if w in ref_word_set]
    # Use dict-hitting words for readability analysis (more meaningful)
    analysis_words = dict_words if dict_words else decoded_words_all

    print(f"     Total decoded words: {len(decoded_words_all)}")
    print(f"     Dict-hitting words: {len(dict_words)}")
    print(f"     Analysis set size: {len(analysis_words)}")
    print(f"     ({time.time() - t3:.1f}s)")

    # ─── 4. Test 1: Bigram plausibility ───
    print("\n  4. Test 1: Bigram plausibility …")
    t4 = time.time()

    bg = _bigram_plausibility(analysis_words, ref_bigrams)
    bg_null = _null_bigram_plausibility(ref_word_set, ref_bigrams, len(analysis_words))
    bg_sel = bg / bg_null if bg_null > 0 else (float('inf') if bg > 0 else 0.0)
    bg_pass = bg_sel > 1.5

    print(f"     Bigram plausibility: {bg:.4f}")
    print(f"     Null baseline:      {bg_null:.4f}")
    print(f"     Selectivity:        {bg_sel:.2f}x")
    print(f"     Pass (>1.5x):       {'PASS' if bg_pass else 'FAIL'}")
    print(f"     ({time.time() - t4:.1f}s)")

    # ─── 5. Test 2: Cross-entropy ratio ───
    print("\n  5. Test 2: Cross-entropy ratio …")
    t5 = time.time()

    decoded_text = ' '.join(analysis_words)
    ce_dec = cross_entropy_lm(list(decoded_text), lm) if decoded_text else 99.0
    ce_ratio = ce_dec / ce_ref if ce_ref > 0 else float('inf')
    ce_pass = ce_ratio < 3.0

    print(f"     Decoded CE:    {ce_dec:.4f}")
    print(f"     Reference CE:  {ce_ref:.4f}")
    print(f"     Ratio:         {ce_ratio:.2f}")
    print(f"     Pass (<3.0):   {'PASS' if ce_pass else 'FAIL'}")
    print(f"     ({time.time() - t5:.1f}s)")

    # ─── 6. Test 3: POS trigram validity ───
    print("\n  6. Test 3: POS trigram validity …")
    t6 = time.time()

    pos_val = _pos_trigram_validity(analysis_words, ref_pos_trigrams)
    rng = random.Random(42)
    word_list = sorted(ref_word_set)
    null_words = [rng.choice(word_list) for _ in range(max(len(analysis_words), 10))]
    pos_null = _pos_trigram_validity(null_words, ref_pos_trigrams)
    pos_sel = pos_val / pos_null if pos_null > 0 else (float('inf') if pos_val > 0 else 0.0)
    pos_pass = pos_sel > 1.3

    print(f"     POS trigram validity: {pos_val:.4f}")
    print(f"     Null baseline:       {pos_null:.4f}")
    print(f"     Selectivity:         {pos_sel:.2f}x")
    print(f"     Pass (>1.3x):        {'PASS' if pos_pass else 'FAIL'}")
    print(f"     ({time.time() - t6:.1f}s)")

    # ─── 7. Test 4: Domain coherence ───
    print("\n  7. Test 4: Domain coherence …")
    t7 = time.time()

    domain_results = _domain_coherence(decoded_words_all, PHARMACEUTICAL_VOCABULARY)
    n_domains = sum(1 for d in domain_results.values() if d['n_hits'] > 0)
    domain_pass = n_domains >= 3

    for dname, dinfo in sorted(domain_results.items()):
        if dinfo['n_hits'] > 0:
            print(f"     {dname:<20} {dinfo['n_hits']} hits: {dinfo['matched_terms'][:5]}")
    print(f"     Domains with hits: {n_domains}")
    print(f"     Pass (>=3):        {'PASS' if domain_pass else 'FAIL'}")
    print(f"     ({time.time() - t7:.1f}s)")

    # ─── 8. Test 5: Phrase detection ───
    print("\n  8. Test 5: Phrase detection …")
    t8 = time.time()

    phrase_hits = _detect_phrases(analysis_words, LATIN_PHRASE_PATTERNS)
    n_phrases = len(phrase_hits)
    phrase_pass = n_phrases >= 1

    if phrase_hits:
        for ph in phrase_hits[:5]:
            print(f"     Found: '{ph['template']}' (pattern: {ph['pattern']})")
    else:
        print(f"     No phrases detected")
    print(f"     Phrases found: {n_phrases}")
    print(f"     Pass (>=1):    {'PASS' if phrase_pass else 'FAIL'}")
    print(f"     ({time.time() - t8:.1f}s)")

    # ─── 9. Compile test results ───
    print("\n  9. Compiling test results …")
    t9 = time.time()

    tests = [
        {'name': 'bigram_plausibility', 'value': round(bg, 4),
         'null': round(bg_null, 4), 'selectivity': round(bg_sel, 2),
         'passed': bg_pass},
        {'name': 'cross_entropy_ratio', 'value': round(ce_ratio, 2),
         'threshold': 3.0, 'selectivity': round(1.0 / ce_ratio if ce_ratio > 0 else 0, 2),
         'passed': ce_pass},
        {'name': 'pos_validity', 'value': round(pos_val, 4),
         'null': round(pos_null, 4), 'selectivity': round(pos_sel, 2),
         'passed': pos_pass},
        {'name': 'domain_coherence', 'value': n_domains,
         'threshold': 3, 'selectivity': n_domains,
         'passed': domain_pass},
        {'name': 'phrase_detection', 'value': n_phrases,
         'threshold': 1, 'selectivity': n_phrases,
         'passed': phrase_pass},
    ]
    n_passing = sum(1 for t in tests if t['passed'])

    print(f"     Tests passing: {n_passing}/5")
    print(f"     ({time.time() - t9:.1f}s)")

    # ─── 10. Comparison with Phase 16 ───
    print("\n  10. Comparing to Phase 16 readability …")
    t10 = time.time()

    phase16_bigram = 0.0
    phase16_n_tests = 0

    # Try readability_22.json first (has Phase 16 context from readability_delta)
    read_delta = _load_json(str(rdir / "readability_delta.json"))
    if read_delta is not None:
        p16_profile = read_delta.get("phase16_profile", {})
        phase16_bigram = p16_profile.get("bigram_plausibility", 0.0)
        phase16_n_tests = p16_profile.get("n_tests_passing", 0)
        print(f"     Phase 16 (from readability_delta): bigram={phase16_bigram:.4f}, tests={phase16_n_tests}/5")
    else:
        read_22 = _load_json(str(rdir / "readability_22.json"))
        if read_22 is not None:
            # Use mode_a as Phase 16 reference
            mode_a = read_22.get("mode_a", {})
            phase16_bigram = mode_a.get("bigram_plausibility", 0.0)
            phase16_n_tests = mode_a.get("n_tests_passing", 0)
            print(f"     Phase 16 (from readability_22): bigram={phase16_bigram:.4f}, tests={phase16_n_tests}/5")
        else:
            print(f"     No Phase 16 readability baseline found")

    improvement = n_passing > phase16_n_tests or (
        n_passing == phase16_n_tests and bg > phase16_bigram
    )

    print(f"     Improvement over Phase 16: {'YES' if improvement else 'NO'}")
    print(f"     ({time.time() - t10:.1f}s)")

    # ─── 11. Null baseline comparison ───
    print("\n  11. Null baseline (5 random shuffles) …")
    t11 = time.time()

    rng2 = random.Random(42)
    null_bgs: List[float] = []
    for trial in range(5):
        shuffled = list(analysis_words)
        rng2.shuffle(shuffled)
        null_bg = _bigram_plausibility(shuffled, ref_bigrams)
        null_bgs.append(null_bg)

    null_mean = sum(null_bgs) / len(null_bgs) if null_bgs else 0.0
    print(f"     Null shuffle bigrams: {[round(b, 4) for b in null_bgs]}")
    print(f"     Null mean: {null_mean:.4f}")
    print(f"     Corrected vs null: {bg:.4f} vs {null_mean:.4f}")
    print(f"     ({time.time() - t11:.1f}s)")

    # ─── Gate and verdict ───
    gate_passed = n_passing >= 3

    verdict_parts = [
        f"Tests: {n_passing}/5",
        f"bigram={bg:.4f} (sel={bg_sel:.2f}x)",
        f"CE ratio={ce_ratio:.2f}",
        f"POS sel={pos_sel:.2f}x",
        f"domains={n_domains}",
        f"phrases={n_phrases}",
        f"vs Phase16: {'better' if improvement else 'same or worse'}",
        f"Gate: {'PASS' if gate_passed else 'FAIL'}",
    ]
    verdict = ' | '.join(verdict_parts)

    elapsed = time.time() - t0

    # ─── Build result ───
    result = CorrectedReadabilityResult(
        timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
        table_source=table_source,
        bigram_plausibility=round(bg, 4),
        bigram_null=round(bg_null, 4),
        bigram_selectivity=round(bg_sel, 2),
        cross_entropy_decoded=round(ce_dec, 4),
        cross_entropy_ref=round(ce_ref, 4),
        ce_ratio=round(ce_ratio, 2),
        pos_trigram_validity=round(pos_val, 4),
        pos_null_validity=round(pos_null, 4),
        pos_selectivity=round(pos_sel, 2),
        domain_hits=domain_results,
        n_domains_with_hits=n_domains,
        phrase_hits=phrase_hits,
        n_phrase_hits=n_phrases,
        n_tests_passing=n_passing,
        test_results=tests,
        phase16_bigram=round(phase16_bigram, 4),
        phase16_n_tests=phase16_n_tests,
        improvement_over_phase16=improvement,
        gate_passed=gate_passed,
        verdict=verdict,
        runtime_seconds=round(elapsed, 2),
    )

    out_path = rdir / "corrected_readability.json"
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2, ensure_ascii=False)

    print(f"\n  SUMMARY")
    print(f"  {'='*60}")
    print(f"  Table source:      {table_source}")
    print(f"  Tests passing:     {n_passing}/5")
    for t in tests:
        status = "PASS" if t['passed'] else "FAIL"
        print(f"    {t['name']:<25} {status}")
    print(f"  Bigram plausib.:   {bg:.4f} (null={bg_null:.4f}, sel={bg_sel:.2f}x)")
    print(f"  CE ratio:          {ce_ratio:.2f}")
    print(f"  Phase 16 bigram:   {phase16_bigram:.4f} ({phase16_n_tests}/5 tests)")
    print(f"  Improvement:       {'YES' if improvement else 'NO'}")
    print(f"  Gate (>=3 tests):  {'PASS' if gate_passed else 'FAIL'}")
    print(f"  → {out_path} ({elapsed:.1f}s)")
