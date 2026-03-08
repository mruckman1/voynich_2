"""
Step 26.6 – Zodiac-Propagated Full Decode
==========================================
Decode the full corpus with the zodiac-merged table and measure readability.
Compare zodiac folios vs non-zodiac folios vs Phase 16 baseline.

Dependency chain:
    zodiac_table.json (Step 26.5)
    modifier_integrate.json (Phase 16)
        → zodiac_decode.json
"""

import json
import os
import random
import time
from collections import defaultdict
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
    build_expanded_word_set,
    load_reference_corpus,
)
from voynich.core.stats import bigram_transition_matrix, compare_bigram_matrices


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


# Zodiac folio IDs
ZODIAC_FOLIOS = {
    'f70v2', 'f70v1', 'f71r', 'f71v', 'f72r1', 'f72r2', 'f72r3',
    'f72v3', 'f72v2', 'f72v1', 'f73r', 'f73v',
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FolioStats:
    folio: str
    section: str
    is_zodiac: bool
    n_tokens: int
    n_dict_hits: int
    dict_hit_rate: float
    sample_decoded: List[str]
    longest_consecutive: int


@dataclass
class ZodiacDecodeResult:
    timestamp: str
    # Full corpus
    corpus_n_tokens: int
    corpus_dict_hit: float
    phase16_dict_hit: float
    improvement: float
    # Null baseline
    null_mean: float
    null_std: float
    selectivity: float
    # Section breakdown
    section_stats: Dict[str, Dict]
    # Zodiac vs non-zodiac
    zodiac_dict_hit: float
    zodiac_n_tokens: int
    non_zodiac_dict_hit: float
    non_zodiac_n_tokens: int
    # Bigram plausibility
    zodiac_bigram_jsd: float
    corpus_bigram_jsd: float
    # Per-folio (zodiac folios detailed)
    zodiac_folio_stats: List[Dict]
    # Best passages
    best_passages: List[Dict]
    longest_consecutive: int
    best_passage_folio: str
    # Verdict
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _consecutive_hits(hits_mask: List[bool]) -> int:
    """Find longest run of True in a boolean list."""
    max_run = 0
    current = 0
    for h in hits_mask:
        if h:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 0
    return max_run


def _find_passages(
    decoded_words: List[str],
    hits_mask: List[bool],
    min_length: int = 3,
) -> List[Dict]:
    """Find consecutive-hit passages of length >= min_length."""
    passages = []
    start = None
    for i, h in enumerate(hits_mask):
        if h:
            if start is None:
                start = i
        else:
            if start is not None:
                length = i - start
                if length >= min_length:
                    passages.append({
                        'start': start,
                        'length': length,
                        'words': decoded_words[start:i],
                    })
            start = None
    # Final passage
    if start is not None:
        length = len(hits_mask) - start
        if length >= min_length:
            passages.append({
                'start': start,
                'length': length,
                'words': decoded_words[start:],
            })
    return passages


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_zodiac_decode() -> None:
    t0 = time.time()
    print("=" * 70)
    print("STEP 26.6: Zodiac-Propagated Full Decode")
    print("=" * 70)

    rd = _results_dir()

    # Load dependencies
    table_data = _load_json(os.path.join(rd, 'zodiac_table.json'))
    if not table_data:
        print("  [SKIP] zodiac_table.json not found — run zodiac-tab first")
        return

    mod_data = _load_json(os.path.join(rd, 'modifier_integrate.json'))
    if not mod_data:
        print("  [SKIP] modifier_integrate.json not found")
        return

    refine_data = _load_json(os.path.join(rd, 'combined_refine.json'))
    phase16_assignment = refine_data.get('best_assignment', {}) if refine_data else {}

    merged = table_data.get('merged_assignment', {})
    modifier_chars = set(mod_data.get('modifier_chars', []))

    corpus = load_corpus(verbose=False)
    eva_to_triple = build_eva_to_triple_lookup()

    # Build expanded word set
    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    base_words = set()
    for text in ref_corpus.get_texts('latin'):
        base_words.update(w.lower() for w in text.tokens if len(w) >= 2)
    expanded_words, _ = build_expanded_word_set(base_words)

    # Build Latin bigram model for plausibility
    latin_text = ref_corpus.get_combined_text('latin')
    lat_bigram_mat, lat_bigram_alph = bigram_transition_matrix(latin_text)

    # -------------------------------------------------------------------
    # Full corpus decode
    # -------------------------------------------------------------------
    print(f"\n  1. Decoding full corpus ...")

    section_tokens: Dict[str, List[str]] = defaultdict(list)
    section_decoded_m: Dict[str, List[str]] = defaultdict(list)
    section_decoded_p: Dict[str, List[str]] = defaultdict(list)
    section_hits_m: Dict[str, List[bool]] = defaultdict(list)
    section_hits_p: Dict[str, List[bool]] = defaultdict(list)

    zodiac_folio_stats: List[FolioStats] = []

    for page in corpus.pages.values():
        folio = page.folio
        section = page.section
        is_zodiac = folio in ZODIAC_FOLIOS
        tokens = page.all_tokens

        folio_decoded_m = []
        folio_hits_m = []

        for token in tokens:
            # Merged table decode
            dec_m = decode_token_modifier_aware(
                token, merged, eva_to_triple, modifier_chars
            )
            hit_m = dec_m.lower() in expanded_words
            section_tokens[section].append(token)
            section_decoded_m[section].append(dec_m)
            section_hits_m[section].append(hit_m)
            folio_decoded_m.append(dec_m)
            folio_hits_m.append(hit_m)

            # Phase 16 decode
            dec_p = decode_token_modifier_aware(
                token, phase16_assignment, eva_to_triple, modifier_chars
            )
            hit_p = dec_p.lower() in expanded_words
            section_decoded_p[section].append(dec_p)
            section_hits_p[section].append(hit_p)

        if is_zodiac and tokens:
            n_hits = sum(folio_hits_m)
            consec = _consecutive_hits(folio_hits_m)
            zodiac_folio_stats.append(FolioStats(
                folio=folio,
                section=section,
                is_zodiac=True,
                n_tokens=len(tokens),
                n_dict_hits=n_hits,
                dict_hit_rate=round(n_hits / len(tokens), 4) if tokens else 0,
                sample_decoded=folio_decoded_m[:20],
                longest_consecutive=consec,
            ))

    # Compute corpus-level stats
    all_hits_m = []
    all_hits_p = []
    for section in section_hits_m:
        all_hits_m.extend(section_hits_m[section])
        all_hits_p.extend(section_hits_p[section])

    corpus_n = len(all_hits_m)
    corpus_hit = sum(all_hits_m) / corpus_n if corpus_n else 0
    p16_hit = sum(all_hits_p) / corpus_n if corpus_n else 0

    print(f"      Corpus tokens: {corpus_n}")
    print(f"      Merged table:  {corpus_hit:.1%}")
    print(f"      Phase 16:      {p16_hit:.1%}")
    print(f"      Improvement:   {corpus_hit - p16_hit:+.1%}")

    # -------------------------------------------------------------------
    # Section breakdown
    # -------------------------------------------------------------------
    print(f"\n  2. Section breakdown:")

    section_stats: Dict[str, Dict] = {}
    for section in sorted(section_hits_m.keys()):
        m_hits = section_hits_m[section]
        p_hits = section_hits_p[section]
        n = len(m_hits)
        m_rate = sum(m_hits) / n if n else 0
        p_rate = sum(p_hits) / n if n else 0
        section_stats[section] = {
            'n_tokens': n,
            'merged_dict_hit': round(m_rate, 4),
            'phase16_dict_hit': round(p_rate, 4),
            'delta': round(m_rate - p_rate, 4),
        }
        flag = " ★" if section == 'astronomical' else ""
        print(f"      {section:16s}: merged={m_rate:.1%}, Phase16={p_rate:.1%}, "
              f"Δ={m_rate - p_rate:+.1%}, n={n}{flag}")

    # -------------------------------------------------------------------
    # Zodiac vs non-zodiac
    # -------------------------------------------------------------------
    zodiac_hits = []
    non_zodiac_hits = []
    for page in corpus.pages.values():
        tokens = page.all_tokens
        for token in tokens:
            dec = decode_token_modifier_aware(
                token, merged, eva_to_triple, modifier_chars
            )
            hit = dec.lower() in expanded_words
            if page.folio in ZODIAC_FOLIOS:
                zodiac_hits.append(hit)
            else:
                non_zodiac_hits.append(hit)

    zodiac_rate = sum(zodiac_hits) / len(zodiac_hits) if zodiac_hits else 0
    non_zodiac_rate = sum(non_zodiac_hits) / len(non_zodiac_hits) if non_zodiac_hits else 0

    print(f"\n  3. Zodiac vs non-zodiac:")
    print(f"      Zodiac:     {zodiac_rate:.1%} ({len(zodiac_hits)} tokens)")
    print(f"      Non-zodiac: {non_zodiac_rate:.1%} ({len(non_zodiac_hits)} tokens)")

    # -------------------------------------------------------------------
    # Bigram plausibility
    # -------------------------------------------------------------------
    print(f"\n  4. Bigram plausibility ...")

    zodiac_decoded_text = ' '.join(
        d for section in section_decoded_m
        if section == 'astronomical'
        for d in section_decoded_m[section]
    )
    corpus_decoded_text = ' '.join(
        d for section in section_decoded_m
        for d in section_decoded_m[section]
    )

    if zodiac_decoded_text:
        z_mat, z_alph = bigram_transition_matrix(zodiac_decoded_text)
        zodiac_jsd = compare_bigram_matrices(z_mat, lat_bigram_mat, z_alph, lat_bigram_alph)
    else:
        zodiac_jsd = 1.0

    if corpus_decoded_text:
        c_mat, c_alph = bigram_transition_matrix(corpus_decoded_text)
        corpus_jsd = compare_bigram_matrices(c_mat, lat_bigram_mat, c_alph, lat_bigram_alph)
    else:
        corpus_jsd = 1.0

    print(f"      Zodiac JSD from Latin: {zodiac_jsd:.4f} (lower = more Latin-like)")
    print(f"      Corpus JSD from Latin: {corpus_jsd:.4f}")

    # -------------------------------------------------------------------
    # Null baseline (20 random permutations)
    # -------------------------------------------------------------------
    print(f"\n  5. Null baseline (20 random permutation tables) ...")

    rng = random.Random(42)
    null_hits: List[float] = []
    all_syllables = list(set(merged.values()))
    all_triples = list(merged.keys())
    all_tokens = corpus.get_tokens()

    for _ in range(20):
        shuffled_syls = all_syllables[:]
        rng.shuffle(shuffled_syls)
        # Create null table by shuffling syllable assignments
        null_table = {}
        for i, triple in enumerate(all_triples):
            null_table[triple] = shuffled_syls[i % len(shuffled_syls)]

        n_null_hits = 0
        for token in all_tokens[:5000]:
            dec = decode_token_modifier_aware(
                token, null_table, eva_to_triple, modifier_chars
            )
            if dec.lower() in expanded_words:
                n_null_hits += 1
        null_hits.append(n_null_hits / 5000)

    null_mean = sum(null_hits) / len(null_hits) if null_hits else 0
    null_std = (sum((h - null_mean) ** 2 for h in null_hits) / len(null_hits)) ** 0.5 if null_hits else 0
    selectivity = corpus_hit / null_mean if null_mean > 0 else float('inf')

    print(f"      Null mean: {null_mean:.4f} ± {null_std:.4f}")
    print(f"      Real:      {corpus_hit:.4f}")
    print(f"      Selectivity: {selectivity:.2f}×")

    # -------------------------------------------------------------------
    # Best passages
    # -------------------------------------------------------------------
    print(f"\n  6. Best decoded passages (zodiac folios) ...")

    all_passages: List[Dict] = []
    for fs in zodiac_folio_stats:
        # Rebuild hit mask for this folio
        page = corpus.get_page(fs.folio)
        if not page:
            continue
        tokens = page.all_tokens
        decoded_words = []
        hit_mask = []
        for token in tokens:
            dec = decode_token_modifier_aware(
                token, merged, eva_to_triple, modifier_chars
            )
            decoded_words.append(dec)
            hit_mask.append(dec.lower() in expanded_words)

        passages = _find_passages(decoded_words, hit_mask, min_length=3)
        for p in passages:
            p['folio'] = fs.folio
            all_passages.append(p)

    all_passages.sort(key=lambda x: x['length'], reverse=True)
    best_5 = all_passages[:5]

    for p in best_5:
        words_str = ' '.join(p['words'][:10])
        print(f"      {p['folio']}: {p['length']} consecutive → {words_str}")

    longest = all_passages[0]['length'] if all_passages else 0
    best_folio = all_passages[0]['folio'] if all_passages else ''

    # Verdict
    if corpus_hit > p16_hit + 0.01 and selectivity > 2.0:
        verdict = (f"IMPROVED: {corpus_hit:.1%} dict_hit ({selectivity:.2f}×), "
                   f"up from Phase16 {p16_hit:.1%}. "
                   f"Zodiac: {zodiac_rate:.1%}.")
    elif corpus_hit >= p16_hit - 0.01:
        verdict = (f"NEUTRAL: {corpus_hit:.1%} ≈ Phase16 {p16_hit:.1%}. "
                   f"Selectivity {selectivity:.2f}×.")
    else:
        verdict = (f"REGRESSED: {corpus_hit:.1%} < Phase16 {p16_hit:.1%}. "
                   f"Zodiac changes degraded performance.")

    print(f"\n  7. Verdict: {verdict}")

    result = ZodiacDecodeResult(
        timestamp=time.strftime('%Y-%m-%dT%H:%M:%S'),
        corpus_n_tokens=corpus_n,
        corpus_dict_hit=round(corpus_hit, 4),
        phase16_dict_hit=round(p16_hit, 4),
        improvement=round(corpus_hit - p16_hit, 4),
        null_mean=round(null_mean, 4),
        null_std=round(null_std, 4),
        selectivity=round(selectivity, 4),
        section_stats=section_stats,
        zodiac_dict_hit=round(zodiac_rate, 4),
        zodiac_n_tokens=len(zodiac_hits),
        non_zodiac_dict_hit=round(non_zodiac_rate, 4),
        non_zodiac_n_tokens=len(non_zodiac_hits),
        zodiac_bigram_jsd=round(zodiac_jsd, 4),
        corpus_bigram_jsd=round(corpus_jsd, 4),
        zodiac_folio_stats=[_convert(asdict(fs)) for fs in zodiac_folio_stats],
        best_passages=best_5,
        longest_consecutive=longest,
        best_passage_folio=best_folio,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'zodiac_decode.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2)

    print(f"\n  → {out_path}")
