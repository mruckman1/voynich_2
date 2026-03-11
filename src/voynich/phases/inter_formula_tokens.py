"""
Step 41.10 – Inter-Formula Content Token Analysis
===================================================
Deep analysis of tokens in CONTENT zones between the repeating formula
occurrences on f57v.  Builds per-zone vocabulary profiles, identifies
content-specific vocabulary, and extracts ingredient candidates.

Dependency chain:
    formula_segmentation.json  (Step 41.9)
    syllable_lexicon.json      (Step 40.9)
    f57v_reading.json          (Step 40.11)
        → inter_formula_tokens.json  (this step)
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
# Core: content token analysis
# ---------------------------------------------------------------------------

def _extract_zone_tokens(
    zone: Dict,
    line_by_line: List[Dict],
) -> List[Dict]:
    """Extract full token records for a zone using position range."""
    start = zone['start']
    end = zone['end']
    tokens = []
    for i in range(start, min(end, len(line_by_line))):
        t = line_by_line[i]
        tokens.append({
            'position': t['position'],
            'decoded': t['decoded'],
            'classification': t['classification'],
            'english_gloss': t['english_gloss'],
            'pos': t.get('pos', ''),
            'domain': t.get('domain', ''),
        })
    return tokens


def _build_vocabulary_profile(
    content_tokens: List[Dict],
    formula_tokens: List[Dict],
) -> Dict:
    """Compare vocabulary between content and formula zones."""
    content_words = Counter(t['decoded'] for t in content_tokens)
    formula_words = Counter(t['decoded'] for t in formula_tokens)

    content_only = {}
    for word, count in content_words.items():
        if word not in formula_words:
            content_only[word] = count

    formula_only = {}
    for word, count in formula_words.items():
        if word not in content_words:
            formula_only[word] = count

    shared = {}
    for word in content_words:
        if word in formula_words:
            shared[word] = {
                'content_count': content_words[word],
                'formula_count': formula_words[word],
            }

    return {
        'n_content_types': len(content_words),
        'n_formula_types': len(formula_words),
        'n_content_only': len(content_only),
        'n_formula_only': len(formula_only),
        'n_shared': len(shared),
        'content_only_words': dict(sorted(content_only.items(),
                                          key=lambda x: -x[1])),
        'formula_only_words': dict(sorted(formula_only.items(),
                                          key=lambda x: -x[1])),
        'shared_words': shared,
    }


def _cluster_by_zone(
    zones: List[Dict],
    line_by_line: List[Dict],
) -> Dict:
    """For each content zone, find unique and shared words."""
    content_zones = [z for z in zones if z['zone_type'] == 'CONTENT']
    if not content_zones:
        return {'n_content_zones': 0, 'zone_clusters': []}

    zone_vocabs: List[Tuple[int, Set[str]]] = []
    zone_token_lists: List[Tuple[int, List[str]]] = []
    for z in content_zones:
        tokens = _extract_zone_tokens(z, line_by_line)
        vocab = set(t['decoded'] for t in tokens)
        zone_vocabs.append((z['zone_id'], vocab))
        zone_token_lists.append((z['zone_id'], [t['decoded'] for t in tokens]))

    # Compute shared across all content zones
    all_content_words: Set[str] = set()
    for _, vocab in zone_vocabs:
        all_content_words |= vocab

    word_zone_count: Counter = Counter()
    for _, vocab in zone_vocabs:
        for w in vocab:
            word_zone_count[w] += 1

    clusters = []
    for zid, vocab in zone_vocabs:
        others = set()
        for zid2, vocab2 in zone_vocabs:
            if zid2 != zid:
                others |= vocab2
        unique = sorted(vocab - others)
        shared_with_others = sorted(vocab & others)

        # Find the zone's token list for ordering
        token_list = []
        for zid2, tl in zone_token_lists:
            if zid2 == zid:
                token_list = tl
                break

        clusters.append({
            'zone_id': zid,
            'n_tokens': len(token_list),
            'n_types': len(vocab),
            'unique_words': unique,
            'n_unique': len(unique),
            'shared_words': shared_with_others,
            'n_shared': len(shared_with_others),
            'token_sequence': token_list,
        })

    return {
        'n_content_zones': len(content_zones),
        'zone_clusters': clusters,
        'words_in_multiple_zones': sorted(
            w for w, c in word_zone_count.items() if c >= 2
        ),
    }


def _identify_ingredient_candidates(
    zones: List[Dict],
    line_by_line: List[Dict],
    lexicon: Dict[str, Dict],
    n_head: int = 3,
) -> List[Dict]:
    """Extract first n_head tokens of each CONTENT zone as potential
    ingredient names.  Also flag the last token as a potential
    quantity/quality modifier."""
    content_zones = [z for z in zones if z['zone_type'] == 'CONTENT']
    candidates = []

    for z in content_zones:
        tokens = _extract_zone_tokens(z, line_by_line)
        if not tokens:
            continue

        head_tokens = tokens[:n_head]
        tail_tokens = tokens[-min(2, len(tokens)):]

        head_entries = []
        for t in head_tokens:
            entry = lexicon.get(t['decoded'], {})
            head_entries.append({
                'position': t['position'],
                'decoded': t['decoded'],
                'classification': t['classification'],
                'english_gloss': t['english_gloss'],
                'lexicon_gloss': entry.get('english_gloss', ''),
                'pos': entry.get('part_of_speech', t.get('pos', '')),
                'domain': entry.get('medical_domain', t.get('domain', '')),
                'is_function_word': entry.get('part_of_speech', '') in (
                    'prep', 'art', 'conj', 'conj/pron', 'adv/conj',
                ),
            })

        tail_entries = []
        for t in tail_tokens:
            entry = lexicon.get(t['decoded'], {})
            tail_entries.append({
                'position': t['position'],
                'decoded': t['decoded'],
                'classification': t['classification'],
                'english_gloss': t['english_gloss'],
                'lexicon_gloss': entry.get('english_gloss', ''),
            })

        # The best ingredient candidate is the first non-function-word
        ingredient_candidate = None
        for he in head_entries:
            if not he['is_function_word'] and he['decoded'] not in ('', '___'):
                ingredient_candidate = he['decoded']
                break
        if ingredient_candidate is None and head_entries:
            ingredient_candidate = head_entries[0]['decoded']

        candidates.append({
            'zone_id': z['zone_id'],
            'zone_start': z['start'],
            'zone_end': z['end'],
            'n_tokens': z['n_tokens'],
            'head_tokens': head_entries,
            'tail_tokens': tail_entries,
            'ingredient_candidate': ingredient_candidate,
            'all_zone_decoded': z.get('decoded_tokens', []),
        })

    return candidates


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_inter_formula_tokens() -> None:
    """Step 41.10: Analyze inter-formula content tokens."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 41.10: Inter-Formula Content Token Analysis")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load inputs ──
    print("\n  1. Loading inputs ...")
    seg_data = _safe_load(os.path.join(rd, 'formula_segmentation.json'))
    f57v_data = _safe_load(os.path.join(rd, 'f57v_reading.json'))
    lex_data = _safe_load(os.path.join(rd, 'syllable_lexicon.json'))

    if not seg_data or not f57v_data:
        missing = []
        if not seg_data:
            missing.append('formula_segmentation.json')
        if not f57v_data:
            missing.append('f57v_reading.json')
        print(f"    ERROR: Missing dependencies: {', '.join(missing)}")
        output = {
            'error': f"Missing: {', '.join(missing)}",
            'runtime_seconds': 0.0,
        }
        out_path = os.path.join(rd, 'inter_formula_tokens.json')
        with open(out_path, 'w') as f:
            json.dump(output, f, indent=2)
        return

    zones = seg_data.get('zones', [])
    line_by_line = f57v_data.get('line_by_line', [])
    lexicon = lex_data.get('syllable_lexicon', {})

    print(f"    Zones: {len(zones)}")
    print(f"    Tokens: {len(line_by_line)}")
    print(f"    Lexicon: {len(lexicon)} entries")

    # ── 2. Extract content zone tokens ──
    print("\n  2. Extracting content zone tokens ...")
    content_zones = [z for z in zones if z['zone_type'] == 'CONTENT']
    formula_zones = [z for z in zones if z['zone_type'] == 'FORMULA']

    all_content_tokens = []
    all_formula_tokens = []
    for z in content_zones:
        all_content_tokens.extend(_extract_zone_tokens(z, line_by_line))
    for z in formula_zones:
        all_formula_tokens.extend(_extract_zone_tokens(z, line_by_line))

    print(f"    Content tokens: {len(all_content_tokens)}")
    print(f"    Formula tokens: {len(all_formula_tokens)}")

    # Per-token summary
    content_token_records = []
    for t in all_content_tokens:
        is_glossed = t['english_gloss'] not in ('___', '???')
        content_token_records.append({
            'position': t['position'],
            'decoded': t['decoded'],
            'classification': t['classification'],
            'is_glossed': is_glossed,
            'english_gloss': t['english_gloss'],
        })

    n_glossed_content = sum(1 for r in content_token_records if r['is_glossed'])
    n_signal_content = sum(
        1 for r in content_token_records if r['classification'] == 'SIGNAL'
    )
    print(f"    Content glossed: {n_glossed_content}/{len(content_token_records)}")
    print(f"    Content SIGNAL:  {n_signal_content}/{len(content_token_records)}")

    # ── 3. Build vocabulary profile ──
    print("\n  3. Building vocabulary profile ...")
    vocab_profile = _build_vocabulary_profile(all_content_tokens, all_formula_tokens)
    print(f"    Content-only types: {vocab_profile['n_content_only']}")
    print(f"    Formula-only types: {vocab_profile['n_formula_only']}")
    print(f"    Shared types: {vocab_profile['n_shared']}")

    # Show top content-only words
    top_content_only = list(vocab_profile['content_only_words'].items())[:10]
    for word, count in top_content_only:
        entry = lexicon.get(word, {})
        gloss = entry.get('english_gloss', '???')
        print(f"      Content-only: {word:20s} (x{count}) = {gloss}")

    # ── 4. Cluster by zone ──
    print("\n  4. Clustering tokens by zone ...")
    zone_clusters = _cluster_by_zone(zones, line_by_line)
    print(f"    Content zones: {zone_clusters['n_content_zones']}")
    print(f"    Words in 2+ zones: {len(zone_clusters.get('words_in_multiple_zones', []))}")

    for cl in zone_clusters.get('zone_clusters', []):
        print(f"    Zone {cl['zone_id']:2d}: {cl['n_tokens']} tok, "
              f"{cl['n_types']} types, {cl['n_unique']} unique, "
              f"{cl['n_shared']} shared")
        if cl['unique_words']:
            print(f"      Unique: {', '.join(cl['unique_words'][:8])}")

    # ── 5. Identify ingredient candidates ──
    print("\n  5. Identifying ingredient candidates ...")
    ingredient_candidates = _identify_ingredient_candidates(
        zones, line_by_line, lexicon, n_head=3,
    )
    print(f"    Candidate zones: {len(ingredient_candidates)}")

    for ic in ingredient_candidates:
        head_words = [h['decoded'] for h in ic['head_tokens']]
        head_glosses = [h['english_gloss'] for h in ic['head_tokens']]
        cand = ic['ingredient_candidate'] or '(none)'
        print(f"    Zone {ic['zone_id']:2d}: head={' '.join(head_words)}, "
              f"glosses={' | '.join(head_glosses)}, candidate={cand}")

    # ── 6. Content domain distribution ──
    print("\n  6. Content domain distribution ...")
    domain_counts: Counter = Counter()
    pos_counts: Counter = Counter()
    for t in all_content_tokens:
        entry = lexicon.get(t['decoded'], {})
        domain = entry.get('medical_domain', 'unknown')
        pos = entry.get('part_of_speech', 'unknown')
        if t['classification'] == 'SIGNAL':
            domain_counts[domain] += 1
            pos_counts[pos] += 1

    for domain, count in domain_counts.most_common(10):
        print(f"      {domain}: {count}")

    # ── 7. Save ──
    elapsed = time.time() - t0

    output = {
        'n_content_tokens': len(all_content_tokens),
        'n_formula_tokens': len(all_formula_tokens),
        'n_glossed_content': n_glossed_content,
        'n_signal_content': n_signal_content,
        'content_signal_rate': round(
            n_signal_content / max(len(all_content_tokens), 1), 4
        ),
        'content_token_records': content_token_records,
        'vocabulary_profile': vocab_profile,
        'zone_clusters': zone_clusters,
        'ingredient_candidates': ingredient_candidates,
        'content_domain_distribution': dict(domain_counts),
        'content_pos_distribution': dict(pos_counts),
        'interpretation': (
            f"{len(all_content_tokens)} content tokens across "
            f"{len(content_zones)} zones. "
            f"{n_glossed_content} glossed ({n_glossed_content * 100 // max(len(all_content_tokens), 1)}%). "
            f"{vocab_profile['n_content_only']} content-only word types. "
            f"{len(ingredient_candidates)} ingredient candidate zones identified."
        ),
        'runtime_seconds': round(elapsed, 1),
    }

    out_path = os.path.join(rd, 'inter_formula_tokens.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(output), f, indent=2)
    print(f"\n  Saved -> {out_path} ({elapsed:.1f}s)")
