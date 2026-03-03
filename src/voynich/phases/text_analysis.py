"""
Phase 15.5 – Decoded Text Analysis
====================================
Phrase detection, section-by-section readability, expanded vocabulary
catalog, and comparison to prior Voynich decipherment claims.

Dependency chain:
    combined_refine.json (Step 15.4 – best assignment)
    dict_expansion.json (Step 15.1 – expanded dictionary)
        → text_analysis.json (this step)
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
    decode_token_modifier_aware,
    load_corpus,
    tokenize_eva_chars,
)
from voynich.core.reference import (
    build_cv_syllable_table,
    build_expanded_word_set,
    LATIN_PHRASE_PATTERNS,
    load_reference_corpus,
    PHARMACEUTICAL_VOCABULARY,
)
from voynich.phases.csp_solver import _convert, decode_corpus, decode_token


# ---------------------------------------------------------------------------
# Modifier-aware decoding
# ---------------------------------------------------------------------------

def _decode_modifier_aware(
    tokens: List[str],
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    modifier_chars: set,
    modifier_rules: Dict[str, str],
    ref_word_set: set,
    max_tokens: int = 5000,
) -> List[str]:
    """R3-style combined decode: alteration → stripping → original."""
    decoded: List[str] = []
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
        # Fall back to original
        decoded.append(decode_token(token, assignment, eva_to_triple))
    return decoded


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TextAnalysisResult:
    # 5a: Phrase detection
    phrases_detected: List[Dict]
    n_phrases: int
    phrases_per_100_tokens: float
    random_phrases_per_100: float
    phrase_selectivity: float

    # 5b: Section readability
    section_readability: List[Dict]
    mean_readability: float

    # 5c: Vocabulary catalog
    vocabulary_catalog: List[Dict]
    n_domains_with_hits: int
    total_domain_coverage: float

    # 5d: Prior claims
    prior_claim_matches: List[Dict]
    n_agreements: int

    gate_passed: bool
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Phrase detection
# ---------------------------------------------------------------------------

def _detect_phrases(
    decoded_tokens: List[str],
    ref_word_set: set,
) -> List[Dict]:
    """Scan decoded token sequence for Latin pharmaceutical phrases.

    A phrase is detected when a decoded token matches any keyword in a
    LATIN_PHRASE_PATTERNS group, and an adjacent token (within ±2) also
    matches a keyword in the same or compatible group.
    """
    detections: List[Dict] = []

    # Build keyword -> pattern type lookup
    keyword_to_pattern: Dict[str, List[str]] = {}
    for pattern_type, keywords in LATIN_PHRASE_PATTERNS:
        for kw in keywords:
            if kw not in keyword_to_pattern:
                keyword_to_pattern[kw] = []
            keyword_to_pattern[kw].append(pattern_type)

    for i, token in enumerate(decoded_tokens):
        if token not in keyword_to_pattern:
            continue

        # Check for adjacent keyword within ±2 positions
        patterns = keyword_to_pattern[token]
        context_words = []
        for offset in [-2, -1, 1, 2]:
            j = i + offset
            if 0 <= j < len(decoded_tokens):
                adj = decoded_tokens[j]
                if adj in keyword_to_pattern or adj in ref_word_set:
                    context_words.append((offset, adj))

        if context_words:
            detections.append({
                'position': i,
                'keyword': token,
                'pattern_types': patterns,
                'context': context_words,
            })

    return detections


def _count_random_phrases(
    voynich_tokens: List[str],
    assignment_keys: List[str],
    all_syls: List[str],
    eva_to_triple: Dict[str, str],
    ref_word_set: set,
    n_trials: int = 50,
    max_tokens: int = 500,
    seed: int = 42,
) -> float:
    """Mean phrases per 100 tokens for random assignments."""
    rng = random.Random(seed)
    counts: List[float] = []

    for _ in range(n_trials):
        rand_map = {k: rng.choice(all_syls) for k in assignment_keys}
        decoded = decode_corpus(voynich_tokens, rand_map, eva_to_triple, max_tokens)
        phrases = _detect_phrases(decoded, ref_word_set)
        per_100 = len(phrases) / max(len(decoded), 1) * 100
        counts.append(per_100)

    return sum(counts) / len(counts) if counts else 0.0


# ---------------------------------------------------------------------------
# Section readability
# ---------------------------------------------------------------------------

def _assess_section_readability(
    corpus,
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    ref_word_set: set,
    max_tokens_per_section: int = 500,
    modifier_chars: Optional[set] = None,
    modifier_rules: Optional[Dict[str, str]] = None,
) -> List[Dict]:
    """Decode each section and assess readability."""
    sections = ['herbal_a', 'pharmaceutical', 'astronomical', 'biological']
    results: List[Dict] = []

    for section in sections:
        sect_tokens = corpus.get_tokens(language='A', section=section, paragraph_only=True)
        if not sect_tokens:
            continue

        sample = sect_tokens[:max_tokens_per_section]
        if modifier_chars:
            decoded = _decode_modifier_aware(
                sample, assignment, eva_to_triple,
                modifier_chars, modifier_rules or {}, ref_word_set,
                max_tokens=max_tokens_per_section,
            )
        else:
            decoded = [decode_token(t, assignment, eva_to_triple) for t in sample]

        # Dict hit rate
        hits = sum(1 for w in decoded if w in ref_word_set)
        hit_rate = hits / len(decoded) if decoded else 0.0

        # Phrase detection
        phrases = _detect_phrases(decoded, ref_word_set)

        # Readability score (1-5 scale based on heuristics)
        if hit_rate > 0.30 and len(phrases) > 5:
            score = 4
        elif hit_rate > 0.20 and len(phrases) > 2:
            score = 3
        elif hit_rate > 0.10:
            score = 2
        elif hit_rate > 0.05:
            score = 1
        else:
            score = 1

        # Mark hits in sample
        sample_decoded: List[str] = []
        for w in decoded[:200]:
            if w in ref_word_set:
                sample_decoded.append(f"*{w}*")
            else:
                sample_decoded.append(w)

        results.append({
            'section': section,
            'n_tokens': len(sample),
            'dict_hit_rate': round(hit_rate, 4),
            'n_phrases_detected': len(phrases),
            'readability_score': score,
            'sample_decoded': sample_decoded[:50],
        })

    return results


# ---------------------------------------------------------------------------
# Vocabulary catalog
# ---------------------------------------------------------------------------

def _build_vocabulary_catalog(
    decoded_tokens: List[str],
    ref_word_set: set,
) -> List[Dict]:
    """Check coverage of pharmaceutical vocabulary domains."""
    unique_decoded = set(decoded_tokens)
    catalog: List[Dict] = []

    for domain, expected_words in PHARMACEUTICAL_VOCABULARY.items():
        found = [w for w in expected_words if w in unique_decoded and w in ref_word_set]
        # Also check if decoded tokens contain the word as a substring
        found_partial = []
        for w in expected_words:
            if w not in found:
                for dt in unique_decoded:
                    if w in dt and len(dt) <= len(w) + 2:
                        found_partial.append(w)
                        break

        all_found = sorted(set(found + found_partial))
        coverage = len(all_found) / len(expected_words) if expected_words else 0.0

        catalog.append({
            'domain': domain,
            'expected_words': expected_words,
            'found_words': all_found,
            'coverage': round(coverage, 3),
        })

    return catalog


# ---------------------------------------------------------------------------
# Prior claims comparison
# ---------------------------------------------------------------------------

# Notable prior Voynich reading proposals
_PRIOR_CLAIMS = [
    # (researcher, folio, proposed_word, language)
    ('Bax', 'f68r', 'taurus', 'latin'),
    ('Bax', 'f2v', 'centaurea', 'latin'),
    ('Tucker-Janick', 'f4v', 'capsicum', 'latin'),
    ('Tucker-Janick', 'f11v', 'calendula', 'latin'),
    ('Cheshire', 'f116v', 'palina', 'romance'),
]


def _compare_prior_claims(
    decoded_tokens_by_folio: Dict[str, List[str]],
) -> List[Dict]:
    """Compare decoded output to notable prior reading proposals."""
    matches: List[Dict] = []

    for researcher, folio, proposed, lang in _PRIOR_CLAIMS:
        folio_decoded = decoded_tokens_by_folio.get(folio, [])
        # Check if proposed word appears in decoded tokens for that folio
        found = any(proposed in dt or dt in proposed for dt in folio_decoded if len(dt) >= 3)
        matches.append({
            'researcher': researcher,
            'folio': folio,
            'proposed_word': proposed,
            'found_in_decoding': found,
        })

    return matches


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_text_analysis() -> None:
    """Step 15.5: Decoded text analysis."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE 15.5: Decoded Text Analysis")
    print("=" * 70)

    rd = _results_dir()

    # Load best assignment (prefer combined_refine, fall back to feature_decode)
    best_assignment = None
    source = ''

    cr_path = os.path.join(rd, 'combined_refine.json')
    if os.path.exists(cr_path):
        with open(cr_path) as f:
            cr_data = json.load(f)
        best_assignment = cr_data.get('best_assignment', {})
        if best_assignment:
            source = 'combined_refine.json'

    if not best_assignment:
        fd_path = os.path.join(rd, 'feature_decode.json')
        if os.path.exists(fd_path):
            with open(fd_path) as f:
                fd_data = json.load(f)
            best_assignment = fd_data.get('best_assignment', {})
            source = 'feature_decode.json'

    if not best_assignment:
        print("  [SKIP] No assignment found")
        return

    print(f"  Using assignment from: {source}")

    # Load corpus
    corpus = load_corpus(verbose=False)
    tokens = corpus.get_tokens(language='A', paragraph_only=True)
    if not tokens:
        print("  [SKIP] No Language A tokens found")
        return

    eva_to_triple = build_eva_to_triple_lookup()

    # Build reference word set (expanded if available)
    ref_corpus = load_reference_corpus(verbose=False)
    ref_tokens = ref_corpus.get_combined_tokens('latin')
    original_word_set = set(w.lower() for w in ref_tokens if len(w) >= 2)

    de_path = os.path.join(rd, 'dict_expansion.json')
    if os.path.exists(de_path):
        with open(de_path) as f:
            de_data = json.load(f)
        if de_data.get('gate_passed', False):
            ref_word_set, _ = build_expanded_word_set(original_word_set)
        else:
            ref_word_set = original_word_set
    else:
        ref_word_set = original_word_set

    all_syls = build_cv_syllable_table('latin')

    # Check for Phase 16 modifier-aware decoding
    modifier_chars: Optional[set] = None
    modifier_rules: Optional[Dict[str, str]] = None
    mi_path = os.path.join(rd, 'modifier_integrate.json')
    if os.path.exists(mi_path):
        with open(mi_path) as f:
            mi_data = json.load(f)
        if mi_data.get('r3_final_gate', False):
            modifier_chars = set(mi_data.get('modifier_chars', []))
            # Reconstruct modifier_rules from classifications
            modifier_rules = {}
            for c in mi_data.get('classifications', []):
                if c.get('final_classification') == 'modifier':
                    modifier_rules[c['eva_char']] = c.get('modifier_type', 'silent')
            print(f"  Phase 16 modifier-aware decoding: {len(modifier_chars)} modifiers")

    # Decode full corpus
    if modifier_chars:
        decoded = _decode_modifier_aware(
            tokens, best_assignment, eva_to_triple,
            modifier_chars, modifier_rules or {}, ref_word_set,
        )
    else:
        decoded = decode_corpus(tokens, best_assignment, eva_to_triple, max_tokens=5000)

    # ─── 5a: Phrase detection ───
    print("\n  5a: Phrase detection ...")
    phrases = _detect_phrases(decoded, ref_word_set)
    n_phrases = len(phrases)
    phrases_per_100 = n_phrases / max(len(decoded), 1) * 100

    random_phrases = _count_random_phrases(
        tokens, list(best_assignment.keys()), all_syls,
        eva_to_triple, ref_word_set, n_trials=50,
    )
    phrase_selectivity = phrases_per_100 / max(random_phrases, 0.001)

    print(f"      Phrases detected: {n_phrases}")
    print(f"      Phrases per 100 tokens: {phrases_per_100:.2f}")
    print(f"      Random baseline: {random_phrases:.2f}")
    print(f"      Phrase selectivity: {phrase_selectivity:.2f}x")

    # ─── 5b: Section readability ───
    print("\n  5b: Section readability ...")
    section_readability = _assess_section_readability(
        corpus, best_assignment, eva_to_triple, ref_word_set,
        modifier_chars=modifier_chars, modifier_rules=modifier_rules,
    )
    for sr in section_readability:
        print(f"      {sr['section']}: dict_hit={sr['dict_hit_rate']:.1%}, "
              f"phrases={sr['n_phrases_detected']}, readability={sr['readability_score']}/5")

    mean_readability = (
        sum(sr['readability_score'] for sr in section_readability) /
        len(section_readability)
    ) if section_readability else 0.0

    # ─── 5c: Vocabulary catalog ───
    print("\n  5c: Vocabulary catalog ...")
    vocab_catalog = _build_vocabulary_catalog(decoded, ref_word_set)
    n_domains_with_hits = sum(1 for vc in vocab_catalog if vc['found_words'])
    total_coverage = (
        sum(vc['coverage'] for vc in vocab_catalog) / len(vocab_catalog)
    ) if vocab_catalog else 0.0

    for vc in vocab_catalog:
        print(f"      {vc['domain']}: {len(vc['found_words'])}/{len(vc['expected_words'])} "
              f"({vc['coverage']:.0%})")
        if vc['found_words']:
            print(f"        Found: {vc['found_words']}")

    # ─── 5d: Prior claims comparison ───
    print("\n  5d: Prior claims comparison ...")
    # Build per-folio decoded tokens (simplified: all tokens in one bag)
    decoded_by_folio: Dict[str, List[str]] = {'all': decoded}
    prior_matches = _compare_prior_claims(decoded_by_folio)
    n_agreements = sum(1 for m in prior_matches if m['found_in_decoding'])
    print(f"      Prior claim agreements: {n_agreements}/{len(prior_matches)}")

    # ─── Gate ───
    gate_passed = phrase_selectivity > 1.5 and n_domains_with_hits >= 2

    elapsed = time.time() - t0

    verdict = (
        f"Text analysis: {n_phrases} phrases detected ({phrase_selectivity:.2f}x selectivity), "
        f"{n_domains_with_hits}/6 domains with hits, "
        f"mean readability {mean_readability:.1f}/5."
    )

    result = TextAnalysisResult(
        phrases_detected=phrases[:100],
        n_phrases=n_phrases,
        phrases_per_100_tokens=round(phrases_per_100, 2),
        random_phrases_per_100=round(random_phrases, 2),
        phrase_selectivity=round(phrase_selectivity, 2),
        section_readability=section_readability,
        mean_readability=round(mean_readability, 2),
        vocabulary_catalog=vocab_catalog,
        n_domains_with_hits=n_domains_with_hits,
        total_domain_coverage=round(total_coverage, 3),
        prior_claim_matches=prior_matches,
        n_agreements=n_agreements,
        gate_passed=gate_passed,
        verdict=verdict,
        runtime_seconds=round(elapsed, 2),
    )

    out_path = os.path.join(rd, 'text_analysis.json')
    with open(out_path, 'w') as f:
        json.dump(asdict(result), f, indent=2, default=_convert)

    print(f"\n  Gate: {'PASS' if gate_passed else 'FAIL'}")
    print(f"  {verdict}")
    print(f"\n  → {out_path}")
