"""
Phase 68, Track 5: Formulaic Pattern Decoding (Hand 4)
========================================================
Match recurring multi-token patterns in the Hand 4 pharmaceutical
section against known Circa Instans formulae.  Fill in unresolved
characters from formula matches.

Dependency chain:
    results/combined_refine.json      (Phase 15)
    results/triple_tiers.json         (Phase 28/53)
        -> results/p68_formulaic.json
"""

import json
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    load_corpus,
    tokenize_eva_chars,
)
from voynich.phases.coda_markers import CodaTable, get_coda
from voynich.phases.corrected_coda import (
    build_coda_table_v2,
    classify_token_chars_v2,
    decode_token_cvc_v2,
)


# ---------------------------------------------------------------------------
# JSON helpers
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
    if isinstance(obj, set):
        return sorted(obj)
    if isinstance(obj, (bool, int, float, str, type(None))):
        return obj
    return str(obj)


def _safe_load(path: str) -> Dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _save_json(rd: str, filename: str, data: Any) -> str:
    path = os.path.join(rd, filename)
    with open(path, 'w') as f:
        json.dump(_convert(data), f, indent=2)
    return path


# ---------------------------------------------------------------------------
# Confirmed / unresolved triple separation
# ---------------------------------------------------------------------------

def _get_confirmed_and_unresolved(
    rd: str,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Return (confirmed_12, unresolved_13)."""
    refine_data = _safe_load(os.path.join(rd, 'combined_refine.json'))
    assignment = refine_data.get('best_assignment', {})

    tier_data = _safe_load(os.path.join(rd, 'triple_tiers.json'))
    confirmed_keys: Set[str] = set()

    if tier_data and 'tiers' in tier_data:
        tiers = tier_data['tiers']
        if isinstance(tiers, dict):
            for entry in tiers.get('CONFIRMED', []):
                confirmed_keys.add(entry.get('triple_key', ''))
        elif isinstance(tiers, list):
            for entry in tiers:
                if entry.get('tier', '') == 'CONFIRMED':
                    confirmed_keys.add(entry.get('triple_key', ''))

    if not confirmed_keys:
        return dict(assignment), {}

    confirmed = {k: v for k, v in assignment.items() if k in confirmed_keys}
    unresolved = {k: v for k, v in assignment.items() if k not in confirmed_keys}
    return confirmed, unresolved


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class FormulaicResult:
    phase: str = "68"
    step: str = "68.5"
    experiment: str = "formulaic_decode"
    n_hand4_tokens: int = 0
    n_recurring_patterns: int = 0
    n_formula_matches: int = 0
    n_triples_constrained: int = 0
    # Pattern and match details
    top_patterns: List[Dict[str, Any]] = field(default_factory=list)
    formula_matches: List[Dict[str, Any]] = field(default_factory=list)
    # Triple constraints
    triple_candidates: Dict[str, str] = field(default_factory=dict)
    triple_details: List[Dict[str, Any]] = field(default_factory=list)
    # Gates
    g1_patterns: bool = False       # FM1: >= 10 recurring patterns
    g2_matches: bool = False        # FM2: >= 3 formula matches score > 0.5
    g3_triples: bool = False        # FM3: >= 2 triples constrained
    gates_passed: int = 0
    gate_passed: bool = False
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Circa Instans formulae (constant)
# ---------------------------------------------------------------------------

CI_FORMULAE = {
    'recipe_X': {
        'latin': 'recipe',
        'decoded': 'recipe',
    },
    'cola_per_pannum': {
        'latin': 'cola per pannum',
        'decoded': 'colaperpannum',
    },
    'tere_in_mortario': {
        'latin': 'tere in mortario',
        'decoded': 'tereinmortario',
    },
    'misce_cum_melle': {
        'latin': 'misce cum melle',
        'decoded': 'miscecummelle',
    },
    'fiat_unguentum': {
        'latin': 'fiat unguentum',
        'decoded': 'fiatunguentum',
    },
    'solve_in_aqua': {
        'latin': 'solve in aqua',
        'decoded': 'solveinaqua',
    },
    'accipe': {
        'latin': 'accipe',
        'decoded': 'accipe',
    },
    'distilla': {
        'latin': 'distilla',
        'decoded': 'distilla',
    },
    'decoctum': {
        'latin': 'decoctum',
        'decoded': 'decoctum',
    },
    'in_nomine': {
        'latin': 'in nomine',
        'decoded': 'innomine',
    },
}


# ---------------------------------------------------------------------------
# Hand 4 / pharmaceutical section tokens
# ---------------------------------------------------------------------------

# Sections associated with Hand 4 and recipe-bearing content
_RECIPE_SECTIONS = {'biological', 'pharmaceutical', 'recipes'}


def _get_hand4_tokens(
    corpus: Any,
    full_assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    coda_table: CodaTable,
) -> List[Dict[str, Any]]:
    """Get tokens from recipe-bearing sections, decoded with full assignment.

    Returns list of dicts with keys: folio, token, decoded, section, idx.
    """
    entries: List[Dict[str, Any]] = []
    global_idx = 0

    for folio, page in corpus.pages.items():
        section = page.section
        tokens = page.all_tokens
        for token in tokens:
            if section in _RECIPE_SECTIONS:
                result = decode_token_cvc_v2(
                    token, full_assignment, eva_to_triple, coda_table)
                decoded = result.decoded_cvc if result.decoded_cvc else ''
                entries.append({
                    'folio': folio,
                    'token': token,
                    'decoded': decoded,
                    'section': section,
                    'idx': global_idx,
                })
            global_idx += 1

    return entries


# ---------------------------------------------------------------------------
# Recurring pattern detection
# ---------------------------------------------------------------------------

def _find_recurring_patterns(
    decoded_words: List[str],
    min_freq: int = 10,
) -> List[Dict[str, Any]]:
    """Count bigrams and trigrams among consecutive decoded words.

    Returns patterns with frequency >= min_freq, sorted by frequency desc.
    """
    bigram_counts: Counter = Counter()
    trigram_counts: Counter = Counter()

    for i in range(len(decoded_words) - 1):
        w1 = decoded_words[i]
        w2 = decoded_words[i + 1]
        if w1 and w2:
            bigram_counts[(w1, w2)] += 1

    for i in range(len(decoded_words) - 2):
        w1 = decoded_words[i]
        w2 = decoded_words[i + 1]
        w3 = decoded_words[i + 2]
        if w1 and w2 and w3:
            trigram_counts[(w1, w2, w3)] += 1

    patterns: List[Dict[str, Any]] = []

    for gram, count in bigram_counts.most_common():
        if count < min_freq:
            break
        patterns.append({
            'ngram': list(gram),
            'joined': ' '.join(gram),
            'n': 2,
            'count': count,
        })

    for gram, count in trigram_counts.most_common():
        if count < min_freq:
            break
        patterns.append({
            'ngram': list(gram),
            'joined': ' '.join(gram),
            'n': 3,
            'count': count,
        })

    patterns.sort(key=lambda p: -p['count'])
    return patterns


# ---------------------------------------------------------------------------
# Skeleton builder
# ---------------------------------------------------------------------------

def _build_confirmed_skeleton(
    decoded_word: str,
    token: str,
    eva_to_triple: Dict[str, str],
    confirmed_keys: Set[str],
    coda_table: CodaTable,
) -> str:
    """Build skeleton marking confirmed chars and '?' for unresolved.

    For each character position in the decoded word, check whether the
    corresponding EVA character maps to a confirmed triple or a coda
    marker.  Confirmed positions keep their decoded character; unresolved
    positions become '?'.
    """
    eva_chars = tokenize_eva_chars(token)
    if not eva_chars:
        return '?' * len(decoded_word) if decoded_word else ''

    classified = classify_token_chars_v2(eva_chars, coda_table)

    # Build a per-syllable skeleton
    skeleton_parts: List[str] = []
    decode_pos = 0

    for role, char in classified:
        if role == 'SYLLABIC':
            triple_key = eva_to_triple.get(char, '')
            # Each syllabic char produces a syllable (typically 2 chars)
            # We need to figure out how many decoded chars this contributes
            if triple_key and triple_key in confirmed_keys:
                # Known — take the decoded characters
                syl_len = 2  # CV syllables are typically 2 chars
                chunk = decoded_word[decode_pos:decode_pos + syl_len]
                skeleton_parts.append(chunk)
                decode_pos += syl_len
            else:
                # Unknown — mark with '?'
                syl_len = 2
                skeleton_parts.append('?' * min(syl_len, max(0, len(decoded_word) - decode_pos)))
                decode_pos += syl_len
        elif role == 'CODA_MARKER':
            coda_val = get_coda(char, coda_table)
            if coda_val:
                skeleton_parts.append(coda_val)
                decode_pos += len(coda_val)

    return ''.join(skeleton_parts)


# ---------------------------------------------------------------------------
# Formula matching
# ---------------------------------------------------------------------------

def _match_formulae(
    patterns: List[Dict[str, Any]],
    formulae: Dict[str, Dict[str, str]],
    decoded_words: List[str],
    tokens: List[str],
    eva_to_triple: Dict[str, str],
    confirmed_keys: Set[str],
    coda_table: CodaTable,
) -> List[Dict[str, Any]]:
    """Match multi-word decoded sequences against CI formulae.

    For each recurring pattern, join into a string, compare to each
    formula's decoded form.  Score = fraction of confirmed characters
    that match.  Keep matches with score > 0.5.  Extract implied values
    for '?' positions from the formula.
    """
    matches: List[Dict[str, Any]] = []

    # Also do sliding-window matching across the full decoded stream
    for formula_key, formula in formulae.items():
        formula_decoded = formula['decoded']
        formula_latin = formula['latin']
        formula_words = formula_latin.split()
        n_words = len(formula_words)

        # Slide through decoded_words
        for i in range(len(decoded_words) - n_words + 1):
            window = decoded_words[i:i + n_words]
            window_tokens = tokens[i:i + n_words]

            if not all(w for w in window):
                continue

            # Build skeleton for this window
            joined_decoded = ''.join(window)
            skeletons = []
            for w, t in zip(window, window_tokens):
                skel = _build_confirmed_skeleton(
                    w, t, eva_to_triple, confirmed_keys, coda_table)
                skeletons.append(skel)
            joined_skeleton = ''.join(skeletons)

            # Score: compare confirmed positions against formula
            if len(joined_skeleton) == 0 or len(formula_decoded) == 0:
                continue

            min_len = min(len(joined_skeleton), len(formula_decoded))
            n_confirmed = 0
            n_match = 0
            implied: List[Dict[str, Any]] = []

            for pos in range(min_len):
                skel_char = joined_skeleton[pos]
                form_char = formula_decoded[pos]
                if skel_char == '?':
                    # Unresolved — record implied value
                    implied.append({
                        'position': pos,
                        'implied_char': form_char,
                    })
                else:
                    n_confirmed += 1
                    if skel_char == form_char:
                        n_match += 1

            score = n_match / n_confirmed if n_confirmed > 0 else 0.0

            if score > 0.5:
                matches.append({
                    'formula_key': formula_key,
                    'formula_latin': formula_latin,
                    'formula_decoded': formula_decoded,
                    'window_decoded': joined_decoded,
                    'window_skeleton': joined_skeleton,
                    'window_words': window,
                    'window_tokens': window_tokens,
                    'score': round(score, 4),
                    'n_confirmed': n_confirmed,
                    'n_match': n_match,
                    'n_implied': len(implied),
                    'implied_positions': implied,
                    'start_idx': i,
                })

    # Deduplicate: keep highest-scoring match per (formula_key, start_idx)
    seen: Set[Tuple[str, int]] = set()
    unique_matches: List[Dict[str, Any]] = []
    for m in sorted(matches, key=lambda x: -x['score']):
        key = (m['formula_key'], m['start_idx'])
        if key not in seen:
            seen.add(key)
            unique_matches.append(m)

    unique_matches.sort(key=lambda x: -x['score'])
    return unique_matches


# ---------------------------------------------------------------------------
# Constraint extraction
# ---------------------------------------------------------------------------

def _extract_constraints(
    formula_matches: List[Dict[str, Any]],
    tokens: List[str],
    eva_to_triple: Dict[str, str],
    confirmed_keys: Set[str],
    coda_table: CodaTable,
    full_assignment: Dict[str, str],
) -> Tuple[Dict[str, str], List[Dict[str, Any]]]:
    """From formula matches, identify which '?' positions correspond to
    which unresolved triples.  Record the formula's character at those
    positions as implied values.

    Returns (triple_candidates, triple_details).
    """
    # Collect votes: triple_key -> Counter of implied syllables
    votes: Dict[str, Counter] = {}

    for match in formula_matches:
        window_tokens = match.get('window_tokens', [])
        formula_decoded = match.get('formula_decoded', '')
        score = match.get('score', 0.0)

        # Walk through each token in the window, figure out which
        # unresolved triples contribute to which positions
        decode_pos = 0

        for token in window_tokens:
            eva_chars = tokenize_eva_chars(token)
            if not eva_chars:
                continue

            classified = classify_token_chars_v2(eva_chars, coda_table)

            for role, char in classified:
                if role == 'SYLLABIC':
                    triple_key = eva_to_triple.get(char, '')
                    syl_len = 2  # CV syllable length

                    if triple_key and triple_key not in confirmed_keys:
                        # This is an unresolved triple — get implied value
                        implied_syl = formula_decoded[decode_pos:decode_pos + syl_len]
                        if implied_syl and len(implied_syl) == syl_len:
                            if triple_key not in votes:
                                votes[triple_key] = Counter()
                            votes[triple_key][implied_syl] += 1

                    decode_pos += syl_len

                elif role == 'CODA_MARKER':
                    coda_val = get_coda(char, coda_table)
                    if coda_val:
                        decode_pos += len(coda_val)

    # Build candidates: pick the most-voted syllable per triple
    triple_candidates: Dict[str, str] = {}
    triple_details: List[Dict[str, Any]] = []

    for triple_key, counter in sorted(votes.items()):
        ranked = counter.most_common()
        best_syl, best_count = ranked[0]
        current = full_assignment.get(triple_key, '')

        detail = {
            'triple_key': triple_key,
            'current_value': current,
            'formula_implied': best_syl,
            'vote_count': best_count,
            'all_votes': dict(counter),
            'changed': best_syl != current,
        }
        triple_details.append(detail)
        triple_candidates[triple_key] = best_syl

    return triple_candidates, triple_details


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_formula_decode():
    """Track 5: Formulaic Pattern Decoding (Hand 4)."""
    t0 = time.time()
    rd = str(_results_dir())

    print("Phase 68.5 — Formulaic Pattern Decoding (Hand 4)")
    print("=" * 50)

    # --- Load data ---
    confirmed, unresolved = _get_confirmed_and_unresolved(rd)
    confirmed_keys = set(confirmed.keys())
    full_assignment = {**confirmed, **unresolved}
    print(f"  Confirmed triples: {len(confirmed)}")
    print(f"  Unresolved triples: {len(unresolved)}")

    eva_to_triple = build_eva_to_triple_lookup()
    coda_table = build_coda_table_v2()

    corpus = load_corpus(verbose=False)

    # --- Step 1: Get Hand 4 / pharmaceutical tokens ---
    print("\n  Collecting recipe-section tokens...")
    hand4_entries = _get_hand4_tokens(
        corpus, full_assignment, eva_to_triple, coda_table)
    n_hand4 = len(hand4_entries)
    print(f"  Recipe-section tokens: {n_hand4}")

    if n_hand4 == 0:
        print("  No recipe-section tokens found; writing empty result.")
        result = FormulaicResult(
            n_hand4_tokens=0,
            runtime_seconds=round(time.time() - t0, 1),
        )
        path = _save_json(rd, 'p68_formulaic.json', result)
        print(f"  Saved: {path}")
        return

    # Extract parallel lists for pattern finding
    decoded_words = [e['decoded'] for e in hand4_entries]
    tokens = [e['token'] for e in hand4_entries]

    section_counts = Counter(e['section'] for e in hand4_entries)
    for sec, cnt in section_counts.most_common():
        print(f"    {sec}: {cnt} tokens")

    # --- Step 2: Find recurring patterns ---
    print("\n  Finding recurring n-gram patterns (min_freq=10)...")
    patterns = _find_recurring_patterns(decoded_words, min_freq=10)
    n_patterns = len(patterns)
    print(f"  Recurring patterns: {n_patterns}")

    for p in patterns[:15]:
        print(f"    [{p['n']}-gram] {p['joined']}  (count={p['count']})")

    # --- Step 3: Match against CI formulae ---
    print("\n  Matching against Circa Instans formulae...")
    formula_matches = _match_formulae(
        patterns, CI_FORMULAE, decoded_words, tokens,
        eva_to_triple, confirmed_keys, coda_table)
    n_matches = len(formula_matches)
    print(f"  Formula matches (score > 0.5): {n_matches}")

    for m in formula_matches[:10]:
        print(f"    {m['formula_key']}: score={m['score']:.3f} "
              f"confirmed={m['n_confirmed']} matched={m['n_match']} "
              f"implied={m['n_implied']}")
        print(f"      decoded:  {m['window_decoded']}")
        print(f"      skeleton: {m['window_skeleton']}")

    # --- Step 4: Extract constraints ---
    print("\n  Extracting triple constraints from formula matches...")
    triple_candidates, triple_details = _extract_constraints(
        formula_matches, tokens, eva_to_triple, confirmed_keys,
        coda_table, full_assignment)
    n_constrained = len(triple_candidates)
    print(f"  Triples constrained: {n_constrained}")

    for detail in triple_details:
        marker = " *" if detail['changed'] else ""
        print(f"    {detail['triple_key']}: {detail['current_value']} -> "
              f"{detail['formula_implied']} (votes={detail['vote_count']}){marker}")

    # --- Gates ---
    g1 = n_patterns >= 10
    g2 = sum(1 for m in formula_matches if m['score'] > 0.5) >= 3
    g3 = n_constrained >= 2
    gates_passed = sum([g1, g2, g3])

    result = FormulaicResult(
        n_hand4_tokens=n_hand4,
        n_recurring_patterns=n_patterns,
        n_formula_matches=n_matches,
        n_triples_constrained=n_constrained,
        top_patterns=patterns[:50],
        formula_matches=formula_matches[:50],
        triple_candidates=triple_candidates,
        triple_details=triple_details,
        g1_patterns=g1,
        g2_matches=g2,
        g3_triples=g3,
        gates_passed=gates_passed,
        gate_passed=gates_passed >= 2,
        runtime_seconds=round(time.time() - t0, 1),
    )

    path = _save_json(rd, 'p68_formulaic.json', result)

    # --- Summary ---
    print(f"\n  Summary")
    print(f"  -------")
    print(f"  Hand 4 tokens:       {n_hand4}")
    print(f"  Recurring patterns:  {n_patterns} ({'PASS' if g1 else 'FAIL'} >= 10)")
    print(f"  Formula matches:     {n_matches} ({'PASS' if g2 else 'FAIL'} >= 3 with score > 0.5)")
    print(f"  Triples constrained: {n_constrained} ({'PASS' if g3 else 'FAIL'} >= 2)")
    print(f"  Gates: {gates_passed}/3")
    print(f"  Saved: {path}")
    print(f"  Time: {result.runtime_seconds:.1f}s")
