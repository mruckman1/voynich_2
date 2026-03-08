"""
Phase 23.5 – Readability Delta Test (read-delta)
==================================================
Applies the best permutation from Step 23.4 to the corpus and runs the
same 5-test readability battery as Phase 22.  Compares three tables —
Phase 16 (statistical), permuted (bridging), Phase 22 (historical) —
to determine whether the permutation produces more readable output.

Dependency chain:
    combined_refine.json (Phase 15 best_assignment)
    modifier_integrate.json (Phase 16 modifier chars)
    permutation_search.json (23.4 best_table)
    merged_table.json (Phase 22)
        → readability_delta.json (this step)
"""

import json
import os
import random
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    decode_token_modifier_aware,
    load_corpus,
    tokenize_eva_chars,
)
from voynich.core.reference import (
    EVA_VISUAL_COMPONENTS,
    LATIN_PHRASE_PATTERNS,
    PHARMACEUTICAL_VOCABULARY,
    build_expanded_word_set,
    load_reference_corpus,
)
from voynich.core.stats import build_ngram_lm, cross_entropy_lm
from voynich.phases.csp_solver import decode_token


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
# Decode with a triple-level table using R3 strategy
# ---------------------------------------------------------------------------

def _decode_corpus_r3(
    tokens: List[str],
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    modifier_chars: Set[str],
    modifier_rules: Dict[str, str],
    ref_word_set: set,
    max_tokens: int = 2000,
) -> List[str]:
    """Decode corpus using R3 combined strategy (alter → strip → original)."""
    decoded = []
    for token in tokens[:max_tokens]:
        # Try alteration
        alt = decode_token_modifier_aware(
            token, assignment, eva_to_triple, modifier_chars,
            modifier_rules=modifier_rules,
        )
        if alt.lower() in ref_word_set:
            decoded.append(alt)
            continue

        # Try stripping
        stripped = decode_token_modifier_aware(
            token, assignment, eva_to_triple, modifier_chars,
        )
        if stripped.lower() in ref_word_set:
            decoded.append(stripped)
            continue

        # Fall back to original decoding
        original = decode_token(token, assignment, eva_to_triple)
        decoded.append(original)

    return decoded


# ---------------------------------------------------------------------------
# Full readability profile
# ---------------------------------------------------------------------------

def _compute_profile(
    source_name: str,
    decoded_words: List[str],
    ref_word_set: set,
    ref_bigrams: set,
    ref_pos_trigrams: set,
    lm: Any,
    ce_ref: float,
) -> Dict[str, Any]:
    """Compute readability profile for a list of decoded words."""
    # Dict hit
    n_total = len(decoded_words)
    dict_words = [w for w in decoded_words if w.lower() in ref_word_set]
    dict_hit = len(dict_words) / n_total if n_total > 0 else 0.0

    # Selectivity vs random
    rng = random.Random(42)
    word_list = sorted(ref_word_set)
    null_hits = []
    for _ in range(50):
        rand_words = [rng.choice(word_list) for _ in range(max(n_total, 10))]
        null_hit = _bigram_plausibility(rand_words, ref_bigrams)
        null_hits.append(null_hit)

    # Use dict-hitting words for readability analysis
    analysis_words = [w.lower() for w in dict_words]

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
    rng2 = random.Random(42)
    null_words = [rng2.choice(word_list) for _ in range(max(len(analysis_words), 10))]
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
         'threshold': 3.0,
         'passed': ce_ratio < 3.0},
        {'name': 'pos_validity', 'value': round(pos_val, 4),
         'null': round(pos_null, 4), 'selectivity': round(pos_sel, 2),
         'passed': pos_sel > 1.3},
        {'name': 'domain_coherence', 'value': n_domains,
         'threshold': 3,
         'passed': n_domains >= 3},
        {'name': 'phrase_detection', 'value': len(phrase_hits),
         'threshold': 1,
         'passed': len(phrase_hits) >= 1},
    ]
    n_passing = sum(1 for t in tests if t['passed'])

    # Selectivity for dict-hit
    rng3 = random.Random(42)
    null_dict_rates = []
    all_syls = list(set(w[:2] for w in ref_word_set if len(w) >= 2))[:75]
    # Simple null: fraction of random 2-char combos in dict
    for _ in range(50):
        rand_decoded = [''.join(rng3.choice(all_syls) if all_syls else 'xx'
                               for _ in range(3)) for _ in range(100)]
        null_dict_rates.append(
            sum(1 for w in rand_decoded if w.lower() in ref_word_set) / len(rand_decoded)
        )
    null_mean = sum(null_dict_rates) / len(null_dict_rates) if null_dict_rates else 0.001
    selectivity = dict_hit / max(null_mean, 0.001)

    return {
        'source': source_name,
        'dict_hit': round(dict_hit, 4),
        'selectivity': round(selectivity, 2),
        'bigram_plausibility': round(bg, 4),
        'bigram_null': round(bg_null, 4),
        'bigram_selectivity': round(bg_sel, 2),
        'cross_entropy': round(ce_dec, 4) if ce_dec < 99 else None,
        'ce_ratio': round(ce_ratio, 2) if ce_ratio < 100 else None,
        'pos_trigram_validity': round(pos_val, 4),
        'pos_selectivity': round(pos_sel, 2),
        'n_domains_with_hits': n_domains,
        'n_phrase_hits': len(phrase_hits),
        'n_tests_passing': n_passing,
        'test_details': tests,
    }


# ---------------------------------------------------------------------------
# Phase 22 → triple conversion (repeated from permutation_search)
# ---------------------------------------------------------------------------

def _convert_phase22_to_triple_level(
    mode_a_table: List[Dict],
) -> Dict[str, str]:
    eva_to_syl: Dict[str, str] = {}
    for entry in mode_a_table:
        eva_char = entry.get('eva_char', '')
        syl = entry.get('syllable_a', '')
        if eva_char and syl and not entry.get('is_modifier', False):
            eva_to_syl[eva_char] = syl

    triple_to_eva: Dict[str, List[str]] = defaultdict(list)
    for eva_char, comp in EVA_VISUAL_COMPONENTS.items():
        tk = f"{comp['first_stroke']},{comp['last_stroke']},{comp['glyph_class']}"
        triple_to_eva[tk].append(eva_char)

    triple_table: Dict[str, str] = {}
    for tk, chars in triple_to_eva.items():
        syls = [eva_to_syl[c] for c in chars if c in eva_to_syl]
        if syls:
            counts = Counter(syls)
            triple_table[tk] = counts.most_common(1)[0][0]
    return triple_table


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class ReadabilityDeltaResult:
    timestamp: str
    phase16_profile: Dict[str, Any]
    permuted_profile: Dict[str, Any]
    phase22_profile: Dict[str, Any]
    permuted_vs_phase16_dict_hit: float
    permuted_vs_phase16_bigram: float
    permuted_vs_phase22_dict_hit: float
    permuted_vs_phase22_bigram: float
    permutation_type: str
    permutation_description: str
    permuted_table: Dict[str, str]
    phase16_decoded_sample: List[List[str]]
    permuted_decoded_sample: List[List[str]]
    ranking: List[str]
    gate_passed: bool
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_readability_delta() -> Dict[str, Any]:
    """Step 23.5: Readability delta test."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE 23.5: Readability Delta Test")
    print("=" * 70)

    rdir = _results_dir()

    # Load Phase 16 assignment
    combined = _load_json(str(rdir / "combined_refine.json")) or {}
    phase16_assignment = combined.get("best_assignment", {})

    # Load modifier info
    mod_data = _load_json(str(rdir / "modifier_integrate.json")) or {}
    modifier_chars = set(mod_data.get("modifier_chars", []))

    # Build modifier rules from classifications
    modifier_rules: Dict[str, str] = {}
    for cls in mod_data.get("classifications", []):
        if cls.get("final_classification") == "modifier":
            modifier_rules[cls["eva_char"]] = cls.get("modifier_type", "silent")

    # Load permutation search result
    perm_data = _load_json(str(rdir / "permutation_search.json")) or {}
    permuted_table = perm_data.get("best_table", {})
    perm_type = perm_data.get("best_permutation", {}).get("permutation_type", "unknown")
    perm_desc = perm_data.get("best_permutation", {}).get("description", "N/A")

    # Load Phase 22 merged table
    merged = _load_json(str(rdir / "merged_table.json")) or {}
    mode_a_table = merged.get("mode_a_table", [])
    phase22_triples = _convert_phase22_to_triple_level(mode_a_table)

    # Load corpus
    corpus = load_corpus(verbose=False)
    tokens = corpus.get_tokens()
    eva_to_triple = build_eva_to_triple_lookup()
    print(f"  Corpus: {len(tokens)} tokens")

    # Build dictionary
    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    base_words = set(
        w.lower() for w in ref_corpus.get_combined_tokens('latin')
        if len(w) >= 2
    )
    expanded_words, _ = build_expanded_word_set(base_words)
    ref_word_set = base_words | expanded_words

    # Build reference LM and bigrams
    ref_words = [w.lower() for w in ref_corpus.get_combined_tokens('latin')
                 if len(w) >= 2]
    ref_text = ' '.join(ref_words[:5000])
    lm = build_ngram_lm(list(ref_text), order=3)
    ce_ref = cross_entropy_lm(list(ref_text), lm)
    ref_bigrams = _build_ref_bigrams(ref_words[:5000])
    ref_pos_trigrams = _build_ref_pos_trigrams(ref_words[:5000])

    # Decode with each table
    max_tokens = 2000
    print("  Decoding with Phase 16 table...")
    p16_decoded = _decode_corpus_r3(
        tokens, phase16_assignment, eva_to_triple,
        modifier_chars, modifier_rules, ref_word_set, max_tokens,
    )

    print("  Decoding with permuted table...")
    perm_decoded = _decode_corpus_r3(
        tokens, permuted_table, eva_to_triple,
        modifier_chars, modifier_rules, ref_word_set, max_tokens,
    )

    print("  Decoding with Phase 22 table...")
    p22_decoded = _decode_corpus_r3(
        tokens, phase22_triples, eva_to_triple,
        modifier_chars, modifier_rules, ref_word_set, max_tokens,
    )

    # Compute profiles
    print("  Computing readability profiles...")
    p16_profile = _compute_profile(
        'phase16', p16_decoded, ref_word_set, ref_bigrams,
        ref_pos_trigrams, lm, ce_ref,
    )
    perm_profile = _compute_profile(
        'permuted', perm_decoded, ref_word_set, ref_bigrams,
        ref_pos_trigrams, lm, ce_ref,
    )
    p22_profile = _compute_profile(
        'phase22', p22_decoded, ref_word_set, ref_bigrams,
        ref_pos_trigrams, lm, ce_ref,
    )

    # Deltas
    perm_vs_p16_dict = (perm_profile['dict_hit'] or 0) - (p16_profile['dict_hit'] or 0)
    perm_vs_p16_bg = (perm_profile['bigram_plausibility'] or 0) - (p16_profile['bigram_plausibility'] or 0)
    perm_vs_p22_dict = (perm_profile['dict_hit'] or 0) - (p22_profile['dict_hit'] or 0)
    perm_vs_p22_bg = (perm_profile['bigram_plausibility'] or 0) - (p22_profile['bigram_plausibility'] or 0)

    # Decoded samples (first 20 tokens)
    p16_sample = [[tokens[i], p16_decoded[i]] for i in range(min(20, len(p16_decoded)))]
    perm_sample = [[tokens[i], perm_decoded[i]] for i in range(min(20, len(perm_decoded)))]

    # Ranking
    profiles = [
        ('phase16', p16_profile),
        ('permuted', perm_profile),
        ('phase22', p22_profile),
    ]
    profiles.sort(key=lambda x: (
        -x[1]['n_tests_passing'],
        -(x[1]['bigram_plausibility'] or 0),
    ))
    ranking = [p[0] for p in profiles]

    # Gate and verdict
    perm_passing = perm_profile['n_tests_passing']
    p16_passing = p16_profile['n_tests_passing']
    gate_passed = perm_passing >= p16_passing

    perm_bg = perm_profile['bigram_plausibility'] or 0
    p16_bg = p16_profile['bigram_plausibility'] or 0

    if perm_bg > p16_bg and perm_passing >= p16_passing:
        verdict = "PERMUTATION IMPROVES READABILITY"
    elif abs(perm_bg - p16_bg) < 0.01 and perm_passing >= p16_passing:
        verdict = "PERMUTATION NEUTRAL"
    else:
        verdict = "PHASE 16 SUPERIOR"

    elapsed = time.time() - t0

    result = ReadabilityDeltaResult(
        timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
        phase16_profile=p16_profile,
        permuted_profile=perm_profile,
        phase22_profile=p22_profile,
        permuted_vs_phase16_dict_hit=round(perm_vs_p16_dict, 4),
        permuted_vs_phase16_bigram=round(perm_vs_p16_bg, 4),
        permuted_vs_phase22_dict_hit=round(perm_vs_p22_dict, 4),
        permuted_vs_phase22_bigram=round(perm_vs_p22_bg, 4),
        permutation_type=perm_type,
        permutation_description=perm_desc,
        permuted_table=permuted_table,
        phase16_decoded_sample=p16_sample,
        permuted_decoded_sample=perm_sample,
        ranking=ranking,
        gate_passed=gate_passed,
        verdict=verdict,
        runtime_seconds=round(elapsed, 1),
    )

    out_path = rdir / "readability_delta.json"
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2, ensure_ascii=False)

    print(f"\n  Phase 16: dict_hit={p16_profile['dict_hit']:.1%}, "
          f"bigram={p16_profile['bigram_plausibility']:.4f}, "
          f"tests={p16_profile['n_tests_passing']}/5")
    print(f"  Permuted: dict_hit={perm_profile['dict_hit']:.1%}, "
          f"bigram={perm_profile['bigram_plausibility']:.4f}, "
          f"tests={perm_profile['n_tests_passing']}/5")
    print(f"  Phase 22: dict_hit={p22_profile['dict_hit']:.1%}, "
          f"bigram={p22_profile['bigram_plausibility']:.4f}, "
          f"tests={p22_profile['n_tests_passing']}/5")
    print(f"  Ranking: {' > '.join(ranking)}")
    print(f"  Verdict: {verdict}")
    print(f"  → {out_path} ({elapsed:.1f}s)")

    return _convert(asdict(result))
