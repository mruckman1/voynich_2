"""
Step 41.7 – Context-Based Disambiguation
==========================================
Disambiguate signal words that have multiple candidate meanings by
examining their corpus context (neighbouring signal words within a
±2 token window).

Dependency chain:
    venetian_match.json              (Step 40.2 — decoded tokens + folios)
    syllable_lexicon.json            (Step 40.9 — original glosses)
    venetian_dictionary_search.json  (Step 41.6 — new glosses)
        → context_disambiguation.json  (this step)
"""

import json
import os
import time
from collections import Counter
from typing import Any, Dict, List, Set, Tuple

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
# Ambiguous words: words with 2+ candidate meanings
# ---------------------------------------------------------------------------

AMBIGUOUS_WORDS: Dict[str, List[str]] = {
    'cora': ['heart (anatomical)', 'cure/care (pharmaceutical)'],
    'be': ['well (adverb)', 'drink (verb)'],
    'sene': ['without (preposition)', 'senna (botanical)'],
    'do': ['give (verb)', 'two (numeral)'],
    'dose': ['dose (pharmaceutical)', 'sweet (adjective)'],
    'hi': ['there (adverb)', 'to him (pronoun)'],
    'fe': ['faith (noun)', 'bile (anatomical)'],
    'rado': ['scraped (verb)', 'root (noun)'],
}


# ---------------------------------------------------------------------------
# Domain keyword lists for context scoring
# ---------------------------------------------------------------------------

PHARMACEUTICAL_VERBS = {
    'cola', 'fa', 'dose', 'dise', 'be', 'ha', 'ga',
}

ANATOMICAL_TERMS = {
    'cora', 'fe', 'sene', 'codi',
}

BOTANICAL_TERMS = {
    'rosa', 'sene', 'radi', 'rado',
}

QUALITY_TERMS = {
    'bene', 'bela', 'raro',
}

FUNCTION_WORDS = {
    'de', 'di', 'se', 'ne', 'si', 'la', 'le', 'co', 'du', 'ce',
    'ni', 'te', 'mi', 'bi', 'do',
}

# Map: meaning keyword → domain tag
DOMAIN_KEYWORDS: Dict[str, str] = {}
for _w in PHARMACEUTICAL_VERBS:
    DOMAIN_KEYWORDS[_w] = 'pharmaceutical'
for _w in ANATOMICAL_TERMS:
    DOMAIN_KEYWORDS[_w] = 'anatomical'
for _w in BOTANICAL_TERMS:
    DOMAIN_KEYWORDS[_w] = 'botanical'
for _w in QUALITY_TERMS:
    DOMAIN_KEYWORDS[_w] = 'quality'

# Map: candidate meaning substring → favoured domain
MEANING_DOMAIN_MAP = {
    'anatomical': 'anatomical',
    'pharmaceutical': 'pharmaceutical',
    'botanical': 'botanical',
    'verb': 'pharmaceutical',   # verbs in this corpus are typically recipe instructions
    'adverb': 'quality',
    'preposition': 'function',
    'noun': 'general',
    'numeral': 'function',
    'pronoun': 'function',
    'adjective': 'quality',
}


# ---------------------------------------------------------------------------
# Core: context extraction and scoring
# ---------------------------------------------------------------------------

def _extract_contexts(
    target_word: str,
    decoded_tokens: List[str],
    token_folios: List[str],
    gloss_lookup: Dict[str, str],
    window: int = 2,
) -> List[Dict]:
    """Extract all occurrences of target_word with ±window token context.

    For each occurrence, record the neighbouring tokens and which
    domain keywords appear in the context.
    """
    n = len(decoded_tokens)
    occurrences: List[Dict] = []

    for i in range(n):
        if decoded_tokens[i] != target_word:
            continue

        # Gather context tokens
        context_tokens: List[str] = []
        context_domains: List[str] = []
        for offset in range(-window, window + 1):
            if offset == 0:
                continue
            j = i + offset
            if 0 <= j < n:
                ctx_word = decoded_tokens[j]
                context_tokens.append(ctx_word)
                # Check domain keyword
                domain = DOMAIN_KEYWORDS.get(ctx_word, '')
                if domain:
                    context_domains.append(domain)
                # Also check via gloss lookup
                gloss = gloss_lookup.get(ctx_word, '')
                if gloss:
                    for keyword, dom in MEANING_DOMAIN_MAP.items():
                        if keyword in gloss.lower():
                            context_domains.append(dom)

        folio = token_folios[i] if i < len(token_folios) else ''
        occurrences.append({
            'position': i,
            'folio': folio,
            'context_tokens': context_tokens,
            'context_domains': context_domains,
        })

    return occurrences


def _score_meanings(
    candidates: List[str],
    occurrences: List[Dict],
) -> Dict[str, Dict]:
    """Score each candidate meaning by how many occurrences favour it."""
    scores: Dict[str, int] = {c: 0 for c in candidates}

    for occ in occurrences:
        domains = occ.get('context_domains', [])
        if not domains:
            continue

        domain_counts: Counter = Counter(domains)

        for candidate in candidates:
            # Check if any context domain matches this candidate's domain
            candidate_lower = candidate.lower()
            for keyword, domain in MEANING_DOMAIN_MAP.items():
                if keyword in candidate_lower:
                    if domain_counts.get(domain, 0) > 0:
                        scores[candidate] += 1
                        break

    # Determine primary meaning
    total_scored = sum(scores.values())
    result: Dict[str, Dict] = {}
    for candidate in candidates:
        count = scores[candidate]
        frac = count / total_scored if total_scored > 0 else 0.0
        result[candidate] = {
            'supporting_occurrences': count,
            'fraction': round(frac, 4),
        }

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_context_disambiguation() -> None:
    """Step 41.7: Context-based disambiguation for ambiguous signal words."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 41.7: Context-Based Disambiguation")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load inputs ──
    print("\n  1. Loading inputs …")
    ven_match = _safe_load(os.path.join(rd, 'venetian_match.json'))
    syl_lex = _safe_load(os.path.join(rd, 'syllable_lexicon.json'))
    dict_search = _safe_load(os.path.join(rd, 'venetian_dictionary_search.json'))

    decoded_tokens = ven_match.get('token_decoded', [])
    token_folios = ven_match.get('token_folios', [])

    print(f"    Decoded tokens: {len(decoded_tokens):,}")
    print(f"    Ambiguous words to disambiguate: {len(AMBIGUOUS_WORDS)}")

    # ── 2. Build gloss lookup ──
    print("\n  2. Building gloss lookup …")
    gloss_lookup: Dict[str, str] = {}

    # From syllable lexicon
    lexicon = syl_lex.get('syllable_lexicon', {})
    for word, entry in lexicon.items():
        gloss = entry.get('english_gloss', '')
        if gloss and gloss != '???':
            gloss_lookup[word] = gloss

    # From new dict search glosses
    new_glosses = dict_search.get('new_glosses', {})
    for word, info in new_glosses.items():
        if word not in gloss_lookup:
            matched = info.get('matched_word', '')
            if matched:
                gloss_lookup[word] = f'cf. {matched}'

    print(f"    Gloss lookup entries: {len(gloss_lookup)}")

    # ── 3. Disambiguate each ambiguous word ──
    print("\n  3. Disambiguating …")
    disambiguations: Dict[str, Dict] = {}

    for word, candidates in AMBIGUOUS_WORDS.items():
        # Check word actually appears in corpus
        word_count = sum(1 for t in decoded_tokens if t == word)

        if word_count == 0:
            disambiguations[word] = {
                'candidates': candidates,
                'n_occurrences': 0,
                'primary_meaning': candidates[0],
                'verdict': 'DEFAULT_NO_OCCURRENCES',
                'meaning_scores': {},
            }
            continue

        # Extract all occurrences with context
        occurrences = _extract_contexts(
            word, decoded_tokens, token_folios, gloss_lookup, window=2,
        )

        # Score each candidate meaning
        meaning_scores = _score_meanings(candidates, occurrences)

        # Determine primary meaning (highest support)
        best_candidate = max(candidates, key=lambda c: meaning_scores[c]['supporting_occurrences'])
        best_count = meaning_scores[best_candidate]['supporting_occurrences']

        # Check if there's a clear winner
        second_best_count = 0
        for c in candidates:
            if c != best_candidate:
                second_best_count = max(second_best_count,
                                        meaning_scores[c]['supporting_occurrences'])

        if best_count == 0:
            verdict = 'UNDETERMINED'
        elif best_count > second_best_count * 2:
            verdict = 'STRONG'
        elif best_count > second_best_count:
            verdict = 'WEAK'
        else:
            verdict = 'TIED'

        # Sample contexts for reporting
        sample_contexts: List[Dict] = []
        for occ in occurrences[:5]:
            sample_contexts.append({
                'folio': occ['folio'],
                'context_tokens': occ['context_tokens'],
                'context_domains': occ['context_domains'],
            })

        disambiguations[word] = {
            'candidates': candidates,
            'n_occurrences': word_count,
            'n_scored': len(occurrences),
            'primary_meaning': best_candidate,
            'verdict': verdict,
            'meaning_scores': meaning_scores,
            'sample_contexts': sample_contexts,
        }

    # ── 4. Print results ──
    print(f"\n  4. Disambiguation results:")
    print(f"    {'Word':10s} {'Occ':>5s} {'Primary Meaning':30s} {'Verdict':12s}")
    print(f"    {'—' * 60}")
    for word in sorted(disambiguations.keys()):
        d = disambiguations[word]
        print(f"    {word:10s} {d['n_occurrences']:5d} "
              f"{d['primary_meaning']:30s} {d['verdict']:12s}")

    # Verdict summary
    verdict_counts: Counter = Counter()
    for d in disambiguations.values():
        verdict_counts[d['verdict']] += 1
    print(f"\n    Verdict summary:")
    for v, c in verdict_counts.most_common():
        print(f"      {v}: {c}")

    # ── 5. Save ──
    elapsed = time.time() - t0

    output = {
        'n_ambiguous_words': len(AMBIGUOUS_WORDS),
        'disambiguations': disambiguations,
        'verdict_counts': dict(verdict_counts),
        'n_strong': verdict_counts.get('STRONG', 0),
        'n_weak': verdict_counts.get('WEAK', 0),
        'n_tied': verdict_counts.get('TIED', 0),
        'n_undetermined': verdict_counts.get('UNDETERMINED', 0),
        'runtime_seconds': round(elapsed, 1),
    }

    out_path = os.path.join(rd, 'context_disambiguation.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(output), f, indent=2)
    print(f"\n  Saved → {out_path} ({elapsed:.1f}s)")
