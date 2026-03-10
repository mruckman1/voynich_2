"""
Step 39.13 -- Venetian Pharmaceutical Phrases
================================================
Search for Venetian pharmaceutical phrases in decoded text.  Extract
recipe templates from Anonimo Veneziano structure.  Analyze f57v
specifically for Venetian verb "fa".

Dependency chain:
    venetian_decode.json       (Step 39.12)
    venetian_lexicon.json      (Step 39.11)
    merged_signal.json         (Step 38.3)
    decode_10k.json            (Step 36.1)
        -> venetian_phrases.json  (this step)
"""

import json
import os
import time
from collections import Counter, defaultdict
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
# Venetian recipe templates
# ---------------------------------------------------------------------------

# Anonimo Veneziano recipe patterns
VENETIAN_TEMPLATES = [
    # template_name, [function_word_slots], description
    ('toy_ingredient_e_verb',
     ['toy', None, 'e', None],
     'Take [ingredient] and [verb]'),
    ('fa_verb_ingredient',
     ['fa', None, None],
     'Make [verb] [ingredient]'),
    ('meti_in_ingredient',
     ['meti', 'in', None],
     'Put in [ingredient]'),
    ('cola_per_ingredient',
     ['cola', 'per', None],
     'Strain through [ingredient]'),
    ('pesta_ingredient',
     ['pesta', None],
     'Grind [ingredient]'),
    ('boli_con_ingredient',
     ['boli', 'con', None],
     'Boil with [ingredient]'),
    ('destempera_con_ingredient',
     ['destempera', 'con', None],
     'Dissolve with [ingredient]'),
    ('toy_la_ingredient',
     ['toy', 'la', None],
     'Take the [ingredient]'),
]


# ---------------------------------------------------------------------------
# Phrase detection
# ---------------------------------------------------------------------------

def _match_template(
    tokens: List[str],
    start: int,
    template_slots: List,
    function_words: Set[str],
    ingredients: Set[str],
    verbs: Set[str],
) -> Tuple[bool, Dict]:
    """Try to match a recipe template starting at position `start`."""
    n_slots = len(template_slots)
    if start + n_slots > len(tokens):
        return False, {}

    matched_words = []
    for j, slot in enumerate(template_slots):
        word = tokens[start + j]
        if slot is not None:
            # Fixed word slot
            if word != slot:
                return False, {}
            matched_words.append(word)
        else:
            # Wildcard -- should be ingredient or verb
            if word in ingredients or word in verbs or len(word) >= 3:
                matched_words.append(word)
            else:
                return False, {}

    return True, {
        'position': start,
        'matched_words': matched_words,
    }


def _search_venetian_phrases(
    decoded_tokens: List[str],
    token_folios: List[str],
    function_words: Set[str],
    ingredients: Set[str],
    verbs: Set[str],
) -> List[Dict]:
    """Search for Venetian recipe template matches in decoded corpus."""
    matches = []

    for tpl_name, tpl_slots, tpl_desc in VENETIAN_TEMPLATES:
        for i in range(len(decoded_tokens)):
            ok, info = _match_template(
                decoded_tokens, i, tpl_slots,
                function_words, ingredients, verbs)
            if ok:
                folio = token_folios[i] if i < len(token_folios) else 'unknown'
                matches.append({
                    'template': tpl_name,
                    'description': tpl_desc,
                    'folio': folio,
                    'position': i,
                    'matched_words': info['matched_words'],
                    'context': decoded_tokens[max(0, i - 2):i + len(tpl_slots) + 2],
                })

    return matches


# ---------------------------------------------------------------------------
# F57v analysis
# ---------------------------------------------------------------------------

def _analyze_f57v(
    decoded_tokens: List[str],
    token_folios: List[str],
    ingredients: Set[str],
    verbs: Set[str],
) -> Dict:
    """Analyze f57v specifically for Venetian verb 'fa' and context."""
    f57v_positions = [i for i, f in enumerate(token_folios) if f == 'f57v']
    if not f57v_positions:
        return {
            'n_tokens': 0,
            'fa_occurrences': 0,
            'fa_contexts': [],
            'verdict': 'f57v not found in corpus',
        }

    f57v_tokens = [decoded_tokens[i] for i in f57v_positions]
    n_f57v = len(f57v_tokens)

    # Find 'fa' occurrences
    fa_contexts = []
    fa_count = 0
    for idx, pos in enumerate(f57v_positions):
        if decoded_tokens[pos] == 'fa':
            fa_count += 1
            # Get following tokens
            following = []
            for k in range(1, 4):
                next_pos = pos + k
                if next_pos < len(decoded_tokens) and token_folios[next_pos] == 'f57v':
                    w = decoded_tokens[next_pos]
                    following.append({
                        'word': w,
                        'in_ingredients': w in ingredients,
                        'in_verbs': w in verbs,
                    })

            # Get preceding tokens
            preceding = []
            for k in range(1, 3):
                prev_pos = pos - k
                if prev_pos >= 0 and token_folios[prev_pos] == 'f57v':
                    preceding.insert(0, decoded_tokens[prev_pos])

            fa_contexts.append({
                'position': pos,
                'preceding': preceding,
                'following': following,
            })

    # Count ingredient matches on f57v
    ingredient_matches = [w for w in f57v_tokens if w in ingredients]
    verb_matches = [w for w in f57v_tokens if w in verbs]

    return {
        'n_tokens': n_f57v,
        'fa_occurrences': fa_count,
        'fa_contexts': fa_contexts,
        'n_ingredient_matches': len(ingredient_matches),
        'ingredient_matches': list(set(ingredient_matches)),
        'n_verb_matches': len(verb_matches),
        'verb_matches': list(set(verb_matches)),
        'verdict': (f"f57v: {fa_count} 'fa' occurrences, "
                    f"{len(ingredient_matches)} ingredient matches, "
                    f"{len(verb_matches)} verb matches"),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_venetian_phrases() -> None:
    """Step 39.13: Venetian Pharmaceutical Phrases."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 39.13: Venetian Pharmaceutical Phrases")
    print("=" * 70)

    rd = _results_dir()

    # -- 1. Load inputs --
    print("\n  1. Loading inputs ...")

    venetian_data = _safe_load(os.path.join(rd, 'venetian_lexicon.json'))
    decode_data = _safe_load(os.path.join(rd, 'decode_10k.json'))
    signal_data = _safe_load(os.path.join(rd, 'merged_signal.json'))
    ven_decode_data = _safe_load(os.path.join(rd, 'venetian_decode.json'))

    # Venetian vocabulary
    prep_verbs: Set[str] = set()
    for v in venetian_data.get('preparation_verbs', []):
        prep_verbs.add(v['word'])

    prep_ingredients: Set[str] = set()
    for v in venetian_data.get('preparation_ingredients', []):
        prep_ingredients.add(v['word'])

    prep_containers: Set[str] = set()
    for v in venetian_data.get('preparation_containers', []):
        prep_containers.add(v['word'])

    function_words = {'de', 'in', 'se', 'ne', 'ad', 'la', 'le', 'di',
                      'da', 'si', 'e', 'et', 'con', 'per', 'el',
                      'toy', 'toi', 'fa', 'far'}

    # Decoded corpus
    token_decoded = decode_data.get('token_decoded', [])
    token_folios = decode_data.get('token_folios', [])
    decoded_lower = [w.lower() for w in token_decoded]
    n_tokens = len(decoded_lower)

    # Signal classifications
    classifications = signal_data.get('token_classifications', [])

    print(f"     {n_tokens} tokens")
    print(f"     Preparation verbs: {len(prep_verbs)}")
    print(f"     Preparation ingredients: {len(prep_ingredients)}")
    print(f"     Preparation containers: {len(prep_containers)}")

    # -- 2. Search for Venetian recipe templates --
    print("\n  2. Searching for Venetian recipe templates ...")

    template_matches = _search_venetian_phrases(
        decoded_lower, token_folios,
        function_words, prep_ingredients, prep_verbs)

    n_template_matches = len(template_matches)
    print(f"     Template matches: {n_template_matches}")

    if template_matches:
        template_counts = Counter(m['template'] for m in template_matches)
        for tpl, count in template_counts.most_common():
            print(f"       {tpl}: {count}")
        print("     Top matches:")
        for m in template_matches[:10]:
            print(f"       [{m['folio']}] {m['template']}: "
                  f"{' '.join(m['matched_words'])}")

    # -- 3. Search for general Venetian phrases --
    print("\n  3. Searching for general Venetian phrases ...")

    venetian_phrases: List[Dict] = []
    ven_function = {'toy', 'toi', 'fa', 'far', 'meti', 'mete',
                    'meschola', 'boli', 'cola', 'pesta', 'el'}

    for i in range(n_tokens - 1):
        if i >= len(token_folios) - 1:
            break
        if token_folios[i] != token_folios[i + 1]:
            continue
        w1, w2 = decoded_lower[i], decoded_lower[i + 1]
        if w1 in ven_function or w2 in ven_function:
            if w1 in ven_function and (w2 in prep_ingredients or
                                        w2 in prep_containers or
                                        len(w2) >= 3):
                venetian_phrases.append({
                    'folio': token_folios[i],
                    'position': i,
                    'words': [w1, w2],
                    'type': 'verb_object',
                })
            elif w2 in ven_function and (w1 in prep_ingredients or
                                          len(w1) >= 3):
                venetian_phrases.append({
                    'folio': token_folios[i],
                    'position': i,
                    'words': [w1, w2],
                    'type': 'object_verb',
                })

    n_venetian_phrases = len(venetian_phrases)
    print(f"     Venetian phrase candidates: {n_venetian_phrases}")

    if venetian_phrases:
        for p in venetian_phrases[:10]:
            print(f"       [{p['folio']}] {' '.join(p['words'])} ({p['type']})")

    # -- 4. F57v analysis --
    print("\n  4. F57v analysis (Venetian verb 'fa') ...")

    f57v_analysis = _analyze_f57v(
        decoded_lower, token_folios,
        prep_ingredients, prep_verbs)

    print(f"     {f57v_analysis['verdict']}")
    if f57v_analysis.get('fa_contexts'):
        for ctx in f57v_analysis['fa_contexts'][:5]:
            following_words = [f['word'] for f in ctx.get('following', [])]
            print(f"       pos {ctx['position']}: "
                  f"... {' '.join(ctx.get('preceding', []))} FA "
                  f"{' '.join(following_words)} ...")

    # -- 5. Score against Anonimo reference patterns --
    print("\n  5. Scoring against Anonimo reference patterns ...")

    # Count how many template matches use SIGNAL tokens
    n_signal_template = 0
    for m in template_matches:
        pos = m['position']
        n_slots = len(m['matched_words'])
        if pos + n_slots <= len(classifications):
            signal_in_match = sum(
                1 for j in range(n_slots)
                if pos + j < len(classifications)
                and classifications[pos + j] == 'SIGNAL')
            if signal_in_match >= 2:
                n_signal_template += 1

    print(f"     Template matches with >= 2 SIGNAL tokens: {n_signal_template}")

    # -- 6. Verdict --
    if n_template_matches >= 5 and n_signal_template >= 2:
        verdict = (f"VENETIAN_PHRASES_FOUND: {n_template_matches} template "
                   f"matches, {n_signal_template} with SIGNAL tokens")
    elif n_template_matches >= 1:
        verdict = (f"PARTIAL_VENETIAN_PHRASES: {n_template_matches} template "
                   f"matches, {n_signal_template} with SIGNAL")
    elif n_venetian_phrases >= 5:
        verdict = (f"VENETIAN_FUNCTION_WORDS: {n_venetian_phrases} function "
                   f"word phrases, 0 full templates")
    else:
        verdict = (f"NO_VENETIAN_PHRASES: {n_template_matches} templates, "
                   f"{n_venetian_phrases} function word phrases")

    elapsed = time.time() - t0

    output = {
        'n_venetian_phrases': n_venetian_phrases,
        'n_template_matches': n_template_matches,
        'f57v_analysis': f57v_analysis,
        'top_phrase_matches': template_matches[:50],
        'venetian_phrases': venetian_phrases[:50],
        'fa_context': f57v_analysis.get('fa_contexts', []),
        'n_signal_template_matches': n_signal_template,
        'verdict': verdict,
        'runtime_seconds': round(elapsed, 1),
    }

    out_path = os.path.join(rd, 'venetian_phrases.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(output), f, indent=2)
    print(f"\n  Saved -> {out_path} ({elapsed:.1f}s)")
