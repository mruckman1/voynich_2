"""
Step 38.5 – Macaronic Context Analysis
=======================================
Run context exploitation on the merged signal vocabulary. With Italian
content words now visible, context around confirmed signal words should
reveal macaronic collocational patterns.

Dependency chain:
    merged_signal.json         (Step 38.3)
    merged_decode.json         (Step 38.2)
    merged_dict.json           (Step 38.1)
    decode_10k.json            (Step 36.1)
        → merged_context.json  (this step)
"""

import json
import math
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
# Core functions
# ---------------------------------------------------------------------------

def _merged_context_analysis(
    signal_words: Set[str],
    decoded_lower: List[str],
    token_folios: List[str],
    merged_dict: Set[str],
    classifications: List[str],
    window: int = 2,
) -> Dict[str, List[Dict]]:
    """Extract context windows around signal words."""
    contexts: Dict[str, List[Dict]] = defaultdict(list)
    n = len(decoded_lower)

    for i in range(n):
        w = decoded_lower[i]
        if w not in signal_words:
            continue
        folio = token_folios[i] if i < len(token_folios) else 'unknown'

        left = []
        for j in range(max(0, i - window), i):
            if token_folios[j] == folio:
                left.append({
                    'word': decoded_lower[j],
                    'in_merged': decoded_lower[j] in merged_dict,
                    'classification': classifications[j] if j < len(classifications) else 'UNKNOWN',
                })

        right = []
        for j in range(i + 1, min(n, i + window + 1)):
            if j < len(token_folios) and token_folios[j] == folio:
                right.append({
                    'word': decoded_lower[j],
                    'in_merged': decoded_lower[j] in merged_dict,
                    'classification': classifications[j] if j < len(classifications) else 'UNKNOWN',
                })

        contexts[w].append({
            'position': i,
            'folio': folio,
            'left': left,
            'right': right,
        })

    return dict(contexts)


def _compute_pmi(
    signal_words: Set[str],
    decoded_lower: List[str],
    token_folios: List[str],
    merged_dict: Set[str],
    classifications: List[str],
    window: int = 2,
) -> List[Dict]:
    """Compute PMI between signal words and their neighbors."""
    n = len(decoded_lower)
    word_count = Counter(decoded_lower)
    pair_count: Counter = Counter()
    total_pairs = 0

    for i in range(n):
        if decoded_lower[i] not in signal_words:
            continue
        folio = token_folios[i] if i < len(token_folios) else 'unknown'
        for j in range(max(0, i - window), min(n, i + window + 1)):
            if j == i:
                continue
            if j < len(token_folios) and token_folios[j] == folio:
                if decoded_lower[j] in merged_dict:
                    pair_count[(decoded_lower[i], decoded_lower[j])] += 1
                    total_pairs += 1

    if total_pairs == 0:
        return []

    pmi_results = []
    for (w1, w2), count in pair_count.most_common(200):
        p_pair = count / total_pairs
        p_w1 = word_count[w1] / n
        p_w2 = word_count[w2] / n
        if p_w1 > 0 and p_w2 > 0:
            pmi = math.log2(p_pair / (p_w1 * p_w2)) if p_pair > 0 else -10.0
            if pmi > 0.5 and count >= 3:
                pmi_results.append({
                    'w1': w1,
                    'w2': w2,
                    'count': count,
                    'pmi': round(pmi, 3),
                })

    pmi_results.sort(key=lambda x: x['pmi'], reverse=True)
    return pmi_results


def _language_aware_context(
    signal_word: str,
    context_entries: List[Dict],
    latin_10k: Set[str],
    italian_10k: Set[str],
) -> Dict[str, Any]:
    """Compute language composition of context around a signal word."""
    left_sources = Counter()
    right_sources = Counter()

    for entry in context_entries:
        for ctx in entry.get('left', []):
            w = ctx['word']
            if ctx['in_merged']:
                if w in latin_10k and w in italian_10k:
                    left_sources['SHARED'] += 1
                elif w in latin_10k:
                    left_sources['LATIN_ONLY'] += 1
                elif w in italian_10k:
                    left_sources['ITALIAN_ONLY'] += 1
            else:
                left_sources['MISS'] += 1

        for ctx in entry.get('right', []):
            w = ctx['word']
            if ctx['in_merged']:
                if w in latin_10k and w in italian_10k:
                    right_sources['SHARED'] += 1
                elif w in latin_10k:
                    right_sources['LATIN_ONLY'] += 1
                elif w in italian_10k:
                    right_sources['ITALIAN_ONLY'] += 1
            else:
                right_sources['MISS'] += 1

    total_left = sum(left_sources.values())
    total_right = sum(right_sources.values())

    return {
        'word': signal_word,
        'n_occurrences': len(context_entries),
        'left_composition': {k: round(v / total_left, 3) if total_left else 0.0
                            for k, v in left_sources.items()},
        'right_composition': {k: round(v / total_right, 3) if total_right else 0.0
                             for k, v in right_sources.items()},
    }


def _build_chains_merged(
    decoded_lower: List[str],
    token_folios: List[str],
    classifications: List[str],
    merged_dict: Set[str],
    latin_10k: Set[str],
    italian_10k: Set[str],
    signal_words: Set[str],
    min_length: int = 3,
) -> List[Dict]:
    """Build chains of consecutive merged-dict hits containing signal words."""
    n = len(decoded_lower)
    chains = []
    i = 0

    while i < n:
        if decoded_lower[i] in merged_dict:
            folio = token_folios[i] if i < len(token_folios) else 'unknown'
            chain = []
            j = i
            while (j < n and
                   decoded_lower[j] in merged_dict and
                   j < len(token_folios) and
                   token_folios[j] == folio):
                w = decoded_lower[j]
                src = ('SHARED' if w in latin_10k and w in italian_10k
                       else 'LATIN_ONLY' if w in latin_10k
                       else 'ITALIAN_ONLY')
                chain.append({
                    'word': w,
                    'source': src,
                    'is_signal': w in signal_words,
                    'classification': classifications[j] if j < len(classifications) else 'UNKNOWN',
                })
                j += 1

            has_signal = any(e['is_signal'] for e in chain)
            if len(chain) >= min_length and has_signal:
                sources = Counter(e['source'] for e in chain)
                chains.append({
                    'folio': folio,
                    'start_pos': i,
                    'length': len(chain),
                    'words': [e['word'] for e in chain],
                    'sources': [e['source'] for e in chain],
                    'n_signal': sum(1 for e in chain if e['is_signal']),
                    'language_mix': dict(sources),
                    'is_macaronic': len(sources) > 1 and 'ITALIAN_ONLY' in sources,
                })
            i = j
        else:
            i += 1

    chains.sort(key=lambda x: x['length'], reverse=True)
    return chains


def _identify_candidates_merged(
    pmi_results: List[Dict],
    signal_words: Set[str],
    merged_dict: Set[str],
    classifications: List[str],
    decoded_lower: List[str],
) -> List[Dict]:
    """Identify new crib candidates from PMI neighbors."""
    # Candidates: neighbors of ≥2 signal words, in merged dict, PMI > 0.5
    neighbor_of = defaultdict(set)
    for entry in pmi_results:
        if entry['w1'] in signal_words:
            neighbor_of[entry['w2']].add(entry['w1'])
        if entry['w2'] in signal_words:
            neighbor_of[entry['w1']].add(entry['w2'])

    candidates = []
    word_counts = Counter(decoded_lower)
    for word, signal_neighbors in neighbor_of.items():
        if len(signal_neighbors) >= 2 and word in merged_dict and word not in signal_words:
            candidates.append({
                'word': word,
                'freq': word_counts[word],
                'n_signal_neighbors': len(signal_neighbors),
                'signal_neighbors': sorted(signal_neighbors),
            })

    candidates.sort(key=lambda x: x['n_signal_neighbors'], reverse=True)
    return candidates


def _medical_collocations(
    chains: List[Dict],
    signal_words: Set[str],
) -> List[Dict]:
    """Identify medical collocation patterns in chains."""
    # Medical vocabulary (Latin + Italian)
    pharma_verbs = {'cola', 'recipe', 'misce', 'coque', 'dice', 'cura',
                    'sana', 'bibe', 'beni'}
    body_parts = {'cora', 'core', 'corpo', 'carne', 'ossa', 'pede',
                  'manu', 'dente', 'naso'}
    ingredients = {'rosa', 'sale', 'vino', 'olio', 'bene', 'sene',
                   'calce', 'suco'}
    qualities = {'bela', 'bona', 'calida', 'frigida', 'sicca',
                 'dulce', 'rara', 'nova'}

    medical_phrases = []
    for chain in chains:
        words = set(chain['words'])
        has_verb = bool(words & pharma_verbs)
        has_body = bool(words & body_parts)
        has_ingredient = bool(words & ingredients)
        has_quality = bool(words & qualities)

        n_medical = sum([has_verb, has_body, has_ingredient, has_quality])
        if n_medical >= 2:
            medical_phrases.append({
                'folio': chain['folio'],
                'words': chain['words'],
                'sources': chain['sources'],
                'has_verb': has_verb,
                'has_body_part': has_body,
                'has_ingredient': has_ingredient,
                'has_quality': has_quality,
                'n_medical_types': n_medical,
                'is_macaronic': chain.get('is_macaronic', False),
            })

    medical_phrases.sort(key=lambda x: x['n_medical_types'], reverse=True)
    return medical_phrases


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_merged_context() -> None:
    """Step 38.5: Macaronic Context Analysis."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 38.5: Macaronic Context Analysis")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load inputs ──
    print("\n  1. Loading inputs …")
    signal_data = _safe_load(os.path.join(rd, 'merged_signal.json'))
    dict_data = _safe_load(os.path.join(rd, 'merged_dict.json'))
    decode_data = _safe_load(os.path.join(rd, 'decode_10k.json'))

    classifications = signal_data.get('token_classifications', [])
    decoded_lower = signal_data.get('token_decoded', [])
    token_folios = signal_data.get('token_folios', [])

    merged_dict = set(dict_data.get('merged_words', []))
    latin_10k = set(dict_data.get('latin_10k_words', []))
    italian_10k = set(dict_data.get('italian_10k_words', []))

    # Build signal word set
    word_signals = signal_data.get('word_signals', [])
    signal_words = set(w['word'] for w in word_signals)

    print(f"     {len(decoded_lower)} tokens, {len(signal_words)} signal words")

    # ── 2. Context extraction ──
    print("  2. Extracting context windows …")
    contexts = _merged_context_analysis(
        signal_words, decoded_lower, token_folios,
        merged_dict, classifications,
    )
    print(f"     Context for {len(contexts)} signal words")

    # ── 3. PMI computation ──
    print("  3. Computing PMI …")
    pmi_results = _compute_pmi(
        signal_words, decoded_lower, token_folios,
        merged_dict, classifications,
    )
    print(f"     {len(pmi_results)} PMI pairs (>0.5, count≥3)")
    if pmi_results:
        print("     Top PMI pairs:")
        for p in pmi_results[:10]:
            print(f"       {p['w1']:>8s} — {p['w2']:<8s}  "
                  f"PMI={p['pmi']:>6.3f}  count={p['count']}")

    # ── 4. Language-aware context ──
    print("  4. Language-aware context patterns …")
    lang_contexts = []
    for word in sorted(signal_words):
        if word in contexts:
            lc = _language_aware_context(
                word, contexts[word], latin_10k, italian_10k,
            )
            lang_contexts.append(lc)

    # Summarize
    ita_left_heavy = sum(
        1 for lc in lang_contexts
        if lc['left_composition'].get('ITALIAN_ONLY', 0) > 0.3
    )
    ita_right_heavy = sum(
        1 for lc in lang_contexts
        if lc['right_composition'].get('ITALIAN_ONLY', 0) > 0.3
    )
    print(f"     Words with >30% Italian left context: {ita_left_heavy}")
    print(f"     Words with >30% Italian right context: {ita_right_heavy}")

    # ── 5. New crib candidates ──
    print("  5. New crib candidates …")
    candidates = _identify_candidates_merged(
        pmi_results, signal_words, merged_dict, classifications, decoded_lower,
    )
    print(f"     {len(candidates)} new crib candidates")
    for c in candidates[:10]:
        print(f"       {c['word']:>10s}  freq={c['freq']:>3d}  "
              f"neighbors={c['signal_neighbors'][:5]}")

    # ── 6. Build chains ──
    print("  6. Building chains …")
    chains = _build_chains_merged(
        decoded_lower, token_folios, classifications,
        merged_dict, latin_10k, italian_10k, signal_words,
    )
    macaronic_chains = [c for c in chains if c.get('is_macaronic')]

    print(f"     {len(chains)} chains of ≥3 dict hits with signal word")
    print(f"     {len(macaronic_chains)} macaronic chains (mixed language)")
    if chains:
        print(f"     Longest chain: {chains[0]['length']} tokens on {chains[0]['folio']}")
        print(f"       Words: {' '.join(chains[0]['words'][:10])}")

    # ── 7. Medical collocations ──
    print("  7. Medical collocation patterns …")
    medical = _medical_collocations(chains, signal_words)
    print(f"     {len(medical)} candidate medical phrases (≥2 medical types)")
    for m in medical[:5]:
        print(f"       {m['folio']}: {' '.join(m['words'][:8])}  "
              f"(verb={m['has_verb']}, body={m['has_body_part']}, "
              f"ingr={m['has_ingredient']}, qual={m['has_quality']})")

    # ── 8. Confirmed-confirmed pairs with language tags ──
    print("  8. Confirmed-confirmed pairs …")
    confirmed_pairs = defaultdict(int)
    for i in range(len(decoded_lower) - 1):
        if i >= len(token_folios) - 1:
            break
        if token_folios[i] != token_folios[i + 1]:
            continue
        w1, w2 = decoded_lower[i], decoded_lower[i + 1]
        if w1 in signal_words and w2 in signal_words:
            w1_src = ('SHARED' if w1 in latin_10k and w1 in italian_10k
                      else 'LATIN_ONLY' if w1 in latin_10k
                      else 'ITALIAN_ONLY')
            w2_src = ('SHARED' if w2 in latin_10k and w2 in italian_10k
                      else 'LATIN_ONLY' if w2 in latin_10k
                      else 'ITALIAN_ONLY')
            pair_type = f"{w1_src}-{w2_src}"
            confirmed_pairs[pair_type] += 1

    for pt, count in sorted(confirmed_pairs.items(), key=lambda x: -x[1]):
        print(f"       {pt}: {count}")

    # ── 9. Save ──
    elapsed = time.time() - t0

    output = {
        'n_signal_words': len(signal_words),
        'n_pmi_pairs': len(pmi_results),
        'pmi_pairs': pmi_results[:100],
        'n_candidates': len(candidates),
        'candidates': candidates[:50],
        'n_chains': len(chains),
        'n_macaronic_chains': len(macaronic_chains),
        'longest_chain_length': chains[0]['length'] if chains else 0,
        'chains': chains[:50],
        'macaronic_chains': macaronic_chains[:30],
        'n_medical_phrases': len(medical),
        'medical_phrases': medical[:30],
        'confirmed_pair_types': dict(confirmed_pairs),
        'language_context_summary': {
            'n_ita_left_heavy': ita_left_heavy,
            'n_ita_right_heavy': ita_right_heavy,
        },
        'verdict': (
            f"{len(pmi_results)} PMI pairs, {len(candidates)} crib candidates, "
            f"{len(chains)} chains (longest={chains[0]['length'] if chains else 0}), "
            f"{len(macaronic_chains)} macaronic, "
            f"{len(medical)} medical phrases."
        ),
        'runtime_seconds': round(elapsed, 1),
    }

    out_path = os.path.join(rd, 'merged_context.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(output), f, indent=2)
    print(f"\n  Saved → {out_path} ({elapsed:.1f}s)")
