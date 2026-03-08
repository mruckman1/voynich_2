"""
Phase 22.5 – Readability Assessment (read-22)
===============================================
THE CRITICAL TEST. Bigram plausibility is the decisive discriminator:
random tables produce ~0% bigram plausibility regardless of dict-hit rate.
A correct table should produce measurable bigram plausibility because decoded
words appear in the order the original Latin author wrote them.

Runs on BOTH Mode A and Mode B from decode-22 output.

Dependency chain:
    corpus_decode_22.json (22.4) + Latin reference corpus
        → readability_22.json (this step)
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
from voynich.core.stats import (
    build_ngram_lm,
    cross_entropy_lm,
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


def _load_json(path: str) -> Optional[Dict]:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Readability tests (reusing patterns from tachy_readability.py)
# ---------------------------------------------------------------------------

def _bigram_plausibility(decoded_words: List[str], ref_bigrams: set) -> float:
    if len(decoded_words) < 2:
        return 0.0
    hits = sum(1 for i in range(len(decoded_words) - 1)
               if (decoded_words[i], decoded_words[i + 1]) in ref_bigrams)
    return hits / (len(decoded_words) - 1)


def _build_ref_bigrams(ref_words: List[str]) -> set:
    return {(ref_words[i].lower(), ref_words[i + 1].lower())
            for i in range(len(ref_words) - 1)}


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
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ModeReadability:
    mode: str
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
    n_tests_passing: int
    test_results: List[Dict]


@dataclass
class Readability22Result:
    timestamp: str
    mode_a: Dict[str, Any]
    mode_b: Dict[str, Any]
    better_mode: str
    null_bigrams: List[float]
    gate_passed: bool
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Assess one mode
# ---------------------------------------------------------------------------

def _assess_mode(
    mode_data: Dict,
    ref_words: List[str],
    ref_bigrams: set,
    ref_pos_trigrams: set,
    ref_word_set: set,
    lm: Any,
    ce_ref: float,
) -> ModeReadability:
    """Run all readability tests on one mode's decoded output."""

    # Extract decoded words
    decoded_words: List[str] = []
    for entry in mode_data.get('decoded_sample', []):
        d = entry.get('decoded', '')
        if d and '?' not in d:
            decoded_words.append(d.lower())

    # Also use viterbi-segmented words
    viterbi_words: List[str] = []
    for vs in mode_data.get('viterbi_sample', []):
        seg = vs.get('segmented', '')
        viterbi_words.extend(seg.split())

    # Use dict-hitting words for meaningful analysis
    dict_words = [w for w in decoded_words if w in ref_word_set]
    viterbi_dict = [w for w in viterbi_words if w in ref_word_set]

    # Use viterbi words if available, fallback to token-level
    analysis_words = viterbi_dict if len(viterbi_dict) > len(dict_words) else dict_words

    # Test 1: Bigram plausibility
    bg = _bigram_plausibility(analysis_words, ref_bigrams)
    bg_null = _null_bigram_plausibility(ref_word_set, ref_bigrams, len(analysis_words))
    bg_sel = bg / bg_null if bg_null > 0 else (float('inf') if bg > 0 else 0.0)

    # Test 2: Cross-entropy
    decoded_text = ' '.join(analysis_words)
    ce_dec = cross_entropy_lm(list(decoded_text), lm) if decoded_text else 99.0
    ce_ratio = ce_dec / ce_ref if ce_ref > 0 else float('inf')

    # Test 3: POS trigram validity
    pos_val = _pos_trigram_validity(analysis_words, ref_pos_trigrams)
    rng = random.Random(42)
    null_words = [rng.choice(sorted(ref_word_set)) for _ in range(max(len(analysis_words), 10))]
    pos_null = _pos_trigram_validity(null_words, ref_pos_trigrams)
    pos_sel = pos_val / pos_null if pos_null > 0 else (float('inf') if pos_val > 0 else 0.0)

    # Test 4: Domain coherence
    all_decoded = [w.lower() for w in decoded_words]
    domain_results = _domain_coherence(all_decoded, PHARMACEUTICAL_VOCABULARY)
    n_domains = sum(1 for d in domain_results.values() if d['n_hits'] > 0)

    # Test 5: Phrase detection
    phrase_hits = _detect_phrases(analysis_words, LATIN_PHRASE_PATTERNS)

    # Compile tests
    tests = [
        {'name': 'bigram_plausibility', 'value': round(bg, 4),
         'null': round(bg_null, 4), 'selectivity': round(bg_sel, 2),
         'passed': bg_sel > 1.5},
        {'name': 'cross_entropy_ratio', 'value': round(ce_ratio, 2),
         'threshold': 3.0, 'selectivity': round(1.0 / ce_ratio if ce_ratio > 0 else 0, 2),
         'passed': ce_ratio < 3.0},
        {'name': 'pos_validity', 'value': round(pos_val, 4),
         'null': round(pos_null, 4), 'selectivity': round(pos_sel, 2),
         'passed': pos_sel > 1.3},
        {'name': 'domain_coherence', 'value': n_domains,
         'threshold': 3, 'selectivity': n_domains,
         'passed': n_domains >= 3},
        {'name': 'phrase_detection', 'value': len(phrase_hits),
         'threshold': 1, 'selectivity': len(phrase_hits),
         'passed': len(phrase_hits) >= 1},
    ]
    n_passing = sum(1 for t in tests if t['passed'])

    return ModeReadability(
        mode=mode_data.get('mode', '?'),
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
        n_phrase_hits=len(phrase_hits),
        n_tests_passing=n_passing,
        test_results=tests,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_readability_22() -> Dict[str, Any]:
    """Assess readability of Phase 22 decoded text."""
    t0 = time.time()
    rdir = _results_dir()

    # Load decode results
    decode_data = _load_json(str(rdir / "corpus_decode_22.json")) or {}
    mode_a_data = decode_data.get('mode_a', {})
    mode_b_data = decode_data.get('mode_b', {})

    # Build reference
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

    # Assess both modes
    result_a = _assess_mode(mode_a_data, ref_words, ref_bigrams,
                            ref_pos_trigrams, ref_word_set, lm, ce_ref)
    result_b = _assess_mode(mode_b_data, ref_words, ref_bigrams,
                            ref_pos_trigrams, ref_word_set, lm, ce_ref)

    # Null baselines (5 random samples)
    null_bgs: List[float] = []
    rng = random.Random(42)
    word_list = sorted(ref_word_set)
    n_words = max(len([w for w in ref_words[:100] if w in ref_word_set]), 20)
    for _ in range(5):
        sample = [rng.choice(word_list) for _ in range(n_words)]
        null_bgs.append(_bigram_plausibility(sample, ref_bigrams))

    better = 'a' if result_a.bigram_plausibility >= result_b.bigram_plausibility else 'b'
    gate = result_a.n_tests_passing >= 3 or result_b.n_tests_passing >= 3

    verdict_parts = [
        f"Mode A: {result_a.n_tests_passing}/5 tests, bg={result_a.bigram_plausibility:.4f}",
        f"Mode B: {result_b.n_tests_passing}/5 tests, bg={result_b.bigram_plausibility:.4f}",
        f"Null bg mean: {sum(null_bgs)/max(len(null_bgs),1):.4f}",
        f"Better: Mode {better.upper()}",
        f"Gate: {'PASS' if gate else 'FAIL'}",
    ]

    result = Readability22Result(
        timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
        mode_a=_convert(asdict(result_a)),
        mode_b=_convert(asdict(result_b)),
        better_mode=better,
        null_bigrams=[round(b, 4) for b in null_bgs],
        gate_passed=gate,
        verdict=' | '.join(verdict_parts),
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = rdir / "readability_22.json"
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t0
    print(f"read-22: A={result_a.bigram_plausibility:.4f} B={result_b.bigram_plausibility:.4f} "
          f"null={sum(null_bgs)/max(len(null_bgs),1):.4f} "
          f"better={better} gate={'PASS' if gate else 'FAIL'} ({elapsed:.1f}s)")

    return _convert(asdict(result))
