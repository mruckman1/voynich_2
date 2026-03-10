"""
Step 36.4 – Context Analysis at 10K
=====================================
PMI-based context exploitation using the 10K signal vocabulary.
Identifies new crib candidates, builds chains, and tests for medical
collocational patterns in the 10K signal landscape.

Dependency chain:
    signal_10k.json           (Step 36.2)
    decode_10k.json           (Step 36.1)
        → context_10k.json   (this step)
"""

import json
import math
import os
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.reference import (
    build_expanded_word_set,
    load_reference_corpus,
)
from voynich.phases.dict_calibration import _build_dict_variants


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


# POS heuristic (same as signal_context.py)
_SUFFIX_POS = [
    ('ntur', 'VERB'), ('tur', 'VERB'), ('nt', 'VERB'),
    ('mus', 'VERB'), ('tis', 'VERB'),
    ('are', 'VERB'), ('ere', 'VERB'), ('ire', 'VERB'),
    ('ans', 'VERB'), ('ens', 'VERB'),
    ('orum', 'GEN_PL'), ('arum', 'GEN_PL'),
    ('ibus', 'DAT_ABL_PL'),
    ('ium', 'GEN_PL'), ('uum', 'GEN_PL'),
    ('um', 'NOUN_ACC'), ('am', 'NOUN_ACC'), ('em', 'NOUN_ACC'),
    ('us', 'NOUN_NOM'), ('er', 'NOUN_NOM'),
    ('ae', 'GEN_DAT'), ('is', 'GEN_DAT'),
    ('os', 'NOUN_ACC_PL'), ('as', 'NOUN_ACC_PL'), ('es', 'NOUN_NOM_PL'),
    ('a', 'NOUN_NOM'),
    ('i', 'GEN_DAT'),
    ('o', 'ABL_DAT'),
    ('e', 'ABL'),
]
_PREPOSITIONS = {'de', 'in', 'ad', 'cum', 'per', 'pro', 'sub', 'ex', 'ab'}
_CONJUNCTIONS = {'et', 'vel', 'aut', 'sed', 'si', 'ne', 'ut'}


def _suffix_pos_heuristic(word: str) -> str:
    if word in _PREPOSITIONS:
        return 'PREP'
    if word in _CONJUNCTIONS:
        return 'CONJ'
    for suffix, pos in _SUFFIX_POS:
        if word.endswith(suffix) and len(word) > len(suffix):
            return pos
    return 'UNKNOWN'


# Medical collocation patterns
_MEDICAL_PATTERNS = {
    'prep_noun': {
        'description': 'Preposition + noun (de/in/cum + N)',
        'triggers': {'de', 'in', 'cum', 'per', 'ad', 'ex', 'ab'},
    },
    'recipe_noun': {
        'description': 'Imperative + noun (recipe/coque/misce + N)',
        'triggers': {'recipe', 'coque', 'misce', 'cola', 'tere', 'adde', 'pone'},
    },
    'adj_noun': {
        'description': 'Adjective + noun or noun + adjective',
        'adj_suffixes': ('us', 'a', 'um', 'is', 'e', 'es', 'ia'),
    },
    'et_conjunction': {
        'description': 'et/vel + content word',
        'triggers': {'et', 'vel', 'aut'},
    },
}


# ---------------------------------------------------------------------------
# Context extraction
# ---------------------------------------------------------------------------

def _extract_context_windows(
    signal_words: List[str],
    decoded: List[str],
    classifications: List[str],
    folios: List[str],
    dict_10k: set,
    window: int = 2,
) -> List[Dict]:
    """Extract ±window context for each signal word at SIGNAL positions."""
    n = len(decoded)
    word_freq = Counter(decoded)
    total_tokens = n

    # Adjacency pair counts
    pair_freq: Counter = Counter()
    for i in range(n - 1):
        if folios[i] == folios[i + 1]:
            pair_freq[(decoded[i], decoded[i + 1])] += 1
    total_pairs = sum(pair_freq.values())

    windows = []
    for sw in signal_words:
        left_counts: Counter = Counter()
        right_counts: Counter = Counter()
        n_occ = 0

        for i in range(n):
            if decoded[i] == sw and classifications[i] == 'SIGNAL':
                n_occ += 1
                for d in range(1, window + 1):
                    if i - d >= 0 and folios[i - d] == folios[i]:
                        left_counts[decoded[i - d]] += 1
                    if i + d < n and folios[i + d] == folios[i]:
                        right_counts[decoded[i + d]] += 1

        if n_occ == 0:
            windows.append({
                'signal_word': sw, 'n_occurrences': 0,
                'top_left': [], 'top_right': [],
                'context_dict_hit_rate': 0.0,
            })
            continue

        def _make_neighbors(counts, direction):
            neighbors = []
            for word, count in counts.most_common(15):
                if direction == 'left':
                    p_pair = pair_freq.get((word, sw), 0) / total_pairs if total_pairs > 0 else 0
                else:
                    p_pair = pair_freq.get((sw, word), 0) / total_pairs if total_pairs > 0 else 0
                p_w = word_freq[word] / total_tokens
                p_sw = word_freq[sw] / total_tokens
                pmi = math.log2(p_pair / (p_w * p_sw)) if p_pair > 0 and p_w > 0 and p_sw > 0 else 0.0
                neighbors.append({
                    'word': word, 'count': count,
                    'pmi': round(pmi, 3),
                    'is_dict_hit': word in dict_10k,
                    'pos_tag': _suffix_pos_heuristic(word),
                })
            return neighbors

        top_left = _make_neighbors(left_counts, 'left')
        top_right = _make_neighbors(right_counts, 'right')

        all_neighbors = list(left_counts.elements()) + list(right_counts.elements())
        ctx_hits = sum(1 for w in all_neighbors if w in dict_10k)
        ctx_rate = ctx_hits / len(all_neighbors) if all_neighbors else 0.0

        windows.append({
            'signal_word': sw,
            'n_occurrences': n_occ,
            'top_left': top_left,
            'top_right': top_right,
            'context_dict_hit_rate': round(ctx_rate, 4),
        })

    return windows


def _identify_new_cribs(
    context_windows: List[Dict],
    dict_10k: set,
    signal_word_set: set,
    min_associations: int = 2,
    min_pmi: float = 0.5,
) -> List[Dict]:
    """Identify new crib candidates from context patterns."""
    word_evidence: Dict[str, List[Tuple[str, float, int]]] = defaultdict(list)

    for cw in context_windows:
        for neighbor in cw.get('top_left', []) + cw.get('top_right', []):
            word = neighbor['word']
            if word in signal_word_set:
                continue
            if neighbor['is_dict_hit']:
                word_evidence[word].append((
                    cw['signal_word'], neighbor['pmi'], neighbor['count'],
                ))

    cribs = []
    for word, evidence_list in word_evidence.items():
        assoc_sws = set(e[0] for e in evidence_list)
        if len(assoc_sws) < min_associations:
            continue
        mean_pmi = sum(e[1] for e in evidence_list) / len(evidence_list)
        if mean_pmi < min_pmi:
            continue
        total_count = sum(e[2] for e in evidence_list)
        cribs.append({
            'word': word,
            'evidence': ', '.join(f'{sw}(PMI={pmi:.1f})' for sw, pmi, _ in evidence_list),
            'total_count': total_count,
            'n_signal_word_associations': len(assoc_sws),
            'mean_pmi': round(mean_pmi, 3),
        })

    cribs.sort(key=lambda c: (-c['n_signal_word_associations'], -c['mean_pmi']))
    return cribs


def _find_chains(
    decoded: List[str],
    classifications: List[str],
    folios: List[str],
    hits_10k: List[bool],
    min_length: int = 3,
) -> List[Dict]:
    """Find maximal consecutive runs of 10K-dict-hit tokens containing ≥1 SIGNAL."""
    n = len(decoded)
    chains = []

    i = 0
    while i < n:
        if not hits_10k[i]:
            i += 1
            continue
        start = i
        folio = folios[i]
        while i < n and hits_10k[i] and folios[i] == folio:
            i += 1
        end = i

        length = end - start
        if length < min_length:
            continue

        words = decoded[start:end]
        n_sig = sum(1 for j in range(start, end) if classifications[j] == 'SIGNAL')
        if n_sig == 0:
            continue

        chains.append({
            'words': words,
            'folio': folio,
            'start_idx': start,
            'length': length,
            'n_signal': n_sig,
        })

    chains.sort(key=lambda c: (-c['length'], -c['n_signal']))
    return chains


def _confirmed_pairs(
    signal_words_set: set,
    decoded: List[str],
    classifications: List[str],
    folios: List[str],
) -> List[Dict]:
    """Find adjacent pairs where both words are independently confirmed signal words."""
    pairs = []
    n = len(decoded)
    for i in range(n - 1):
        if (decoded[i] in signal_words_set
                and decoded[i + 1] in signal_words_set
                and classifications[i] == 'SIGNAL'
                and classifications[i + 1] == 'SIGNAL'
                and folios[i] == folios[i + 1]):
            pairs.append({
                'word1': decoded[i],
                'word2': decoded[i + 1],
                'folio': folios[i],
                'position': i,
            })
    return pairs


def _medical_collocation_test(
    signal_words_set: set,
    decoded: List[str],
    classifications: List[str],
    folios: List[str],
    dict_10k: set,
) -> Dict:
    """Test for medical collocational patterns among SIGNAL tokens."""
    n = len(decoded)
    pattern_hits: Dict[str, List[Dict]] = defaultdict(list)

    for i in range(n - 1):
        if folios[i] != folios[i + 1]:
            continue
        if classifications[i] != 'SIGNAL' and classifications[i + 1] != 'SIGNAL':
            continue

        w1, w2 = decoded[i], decoded[i + 1]

        # prep + noun
        if w1 in _MEDICAL_PATTERNS['prep_noun']['triggers'] and w2 in dict_10k:
            pattern_hits['prep_noun'].append({
                'folio': folios[i], 'position': i,
                'pattern': f'{w1} {w2}',
            })

        # recipe + noun
        if w1 in _MEDICAL_PATTERNS['recipe_noun']['triggers'] and w2 in dict_10k:
            pattern_hits['recipe_noun'].append({
                'folio': folios[i], 'position': i,
                'pattern': f'{w1} {w2}',
            })

        # et/vel + content
        if w1 in _MEDICAL_PATTERNS['et_conjunction']['triggers'] and w2 in dict_10k:
            pattern_hits['et_conjunction'].append({
                'folio': folios[i], 'position': i,
                'pattern': f'{w1} {w2}',
            })

    return {
        name: {
            'count': len(hits),
            'examples': hits[:10],
        }
        for name, hits in pattern_hits.items()
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_context_10k() -> None:
    """Step 36.4: Context analysis at 10K dictionary."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 36.4: Context Analysis at 10K")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load inputs ──
    print("\n  1. Loading inputs …")
    with open(os.path.join(rd, 'signal_10k.json')) as f:
        sig_data = json.load(f)

    token_folios = sig_data['token_folios']
    token_decoded = sig_data['token_decoded']
    classifications = sig_data['token_classifications']
    hits_10k = sig_data['token_hits_10k']
    n_tokens = sig_data['n_tokens']

    # Get signal word list
    signal_words = [
        w['word'] for w in sig_data['word_signals']
        if w['is_genuine_signal']
    ]
    signal_words_set = set(signal_words)
    print(f"     {n_tokens} tokens, {len(signal_words)} signal words")

    # Build 10K dict
    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    base_words = set(
        w.lower() for w in ref_corpus.get_combined_tokens('latin')
        if len(w) >= 2
    )
    dict_variants = _build_dict_variants(base_words, ref_corpus, [10000])
    dict_10k = dict_variants[0][1]

    # ── 2. Extract context windows ──
    print("  2. Extracting context windows (±2 tokens) …")
    context_windows = _extract_context_windows(
        signal_words, token_decoded, classifications,
        token_folios, dict_10k, window=2,
    )
    print(f"     {len(context_windows)} windows extracted")

    # ── 3. Identify new crib candidates ──
    print("  3. Identifying new crib candidates …")
    new_cribs = _identify_new_cribs(
        context_windows, dict_10k, signal_words_set,
    )
    print(f"     {len(new_cribs)} new crib candidates")
    for nc in new_cribs[:10]:
        print(f"       {nc['word']:<12s} assoc={nc['n_signal_word_associations']}"
              f"  PMI={nc['mean_pmi']:.2f}  count={nc['total_count']}")

    # ── 4. Build chains ──
    print("  4. Building chains (≥3 consecutive 10K-dict hits with SIGNAL) …")
    chains = _find_chains(
        token_decoded, classifications, token_folios, hits_10k,
    )
    longest = max((c['length'] for c in chains), default=0)
    print(f"     {len(chains)} chains found, longest = {longest}")
    for ch in chains[:5]:
        print(f"       {ch['folio']} len={ch['length']} sig={ch['n_signal']}"
              f"  [{' '.join(ch['words'])}]")

    # ── 5. Confirmed-confirmed pairs ──
    print("  5. Finding confirmed-confirmed pairs …")
    conf_pairs = _confirmed_pairs(
        signal_words_set, token_decoded, classifications, token_folios,
    )
    print(f"     {len(conf_pairs)} confirmed-confirmed pairs")
    for cp in conf_pairs[:10]:
        print(f"       {cp['folio']} pos={cp['position']}: {cp['word1']} {cp['word2']}")

    # ── 6. Medical collocation test ──
    print("  6. Medical collocation test …")
    medical_results = _medical_collocation_test(
        signal_words_set, token_decoded, classifications,
        token_folios, dict_10k,
    )
    for pattern, data in medical_results.items():
        print(f"     {pattern}: {data['count']} hits")
        for ex in data['examples'][:3]:
            print(f"       {ex['folio']}: {ex['pattern']}")

    # ── 7. Save ──
    elapsed = time.time() - t0

    output = {
        'signal_words': signal_words,
        'context_windows': context_windows,
        'new_crib_candidates': new_cribs,
        'n_new_crib_candidates': len(new_cribs),
        'chain_candidates': chains[:100],
        'n_chains_found': len(chains),
        'longest_chain': longest,
        'confirmed_confirmed_pairs': conf_pairs,
        'n_confirmed_pairs': len(conf_pairs),
        'medical_collocations': medical_results,
        'n_medical_total': sum(d['count'] for d in medical_results.values()),
        'runtime_seconds': round(elapsed, 1),
    }

    out_path = os.path.join(rd, 'context_10k.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved → {out_path}")

    # ── Summary ──
    print("\n" + "=" * 70)
    print("CONTEXT 10K SUMMARY")
    print("=" * 70)
    print(f"\n  Signal words analyzed: {len(signal_words)}")
    print(f"  New crib candidates: {len(new_cribs)}")
    print(f"  Chains (≥3): {len(chains)}, longest = {longest}")
    print(f"  Confirmed-confirmed pairs: {len(conf_pairs)}")
    print(f"  Medical collocations: {sum(d['count'] for d in medical_results.values())}")
    print(f"\n  Runtime: {elapsed:.1f}s")
