"""
Step 37.10 – f57v EVA Token Diversity
=======================================
Determine whether f57v's repetitive decode reflects repetitive EVA input
(same tokens repeating) or diverse EVA input collapsing to few decoded
syllables (many different tokens all decoding to the same syllable).

Dependency chain:
    signal_10k.json            (Step 36.2)
    decode_10k.json            (Step 36.1)
    combined_refine.json       (Phase 15)
        → f57v_eva_analysis.json   (this step)
"""

import json
import os
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    load_corpus,
    token_to_triples,
    tokenize_eva_chars,
)


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
# Main
# ---------------------------------------------------------------------------

def run_f57v_eva() -> None:
    """Step 37.10: f57v EVA Token Diversity."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 37.10: f57v EVA Token Diversity")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load inputs ──
    print("\n  1. Loading inputs …")
    signal_data = _safe_load(os.path.join(rd, 'signal_10k.json'))
    refine_data = _safe_load(os.path.join(rd, 'combined_refine.json'))

    token_folios = signal_data.get('token_folios', [])
    token_evas = signal_data.get('token_evas', [])
    token_decoded = signal_data.get('token_decoded', [])
    token_classifications = signal_data.get('token_classifications', [])
    assignment = refine_data.get('best_assignment', {})

    eva_to_triple = build_eva_to_triple_lookup()

    # Extract f57v tokens
    f57v_positions = [i for i, f in enumerate(token_folios) if f == 'f57v']
    n_f57v = len(f57v_positions)
    print(f"     {n_f57v} tokens on f57v")

    # Build f57v token list
    f57v_tokens = []
    for pos in f57v_positions:
        eva = token_evas[pos]
        dec = token_decoded[pos]
        cls = token_classifications[pos]
        chars = tokenize_eva_chars(eva)
        triples = token_to_triples(eva, eva_to_triple)
        f57v_tokens.append({
            'position': pos,
            'eva_original': eva,
            'eva_chars': chars,
            'decoded': dec,
            'signal_10k': cls == 'SIGNAL',
            'classification': cls,
            'char_triples': triples,
        })

    # ── 2. Token-level diversity metrics ──
    print("  2. Computing diversity metrics …")
    eva_types = set(t['eva_original'] for t in f57v_tokens)
    decoded_types = set(t['decoded'] for t in f57v_tokens)

    token_ttr = len(eva_types) / n_f57v if n_f57v > 0 else 0.0
    decoded_ttr = len(decoded_types) / n_f57v if n_f57v > 0 else 0.0
    compression_ratio = decoded_ttr / token_ttr if token_ttr > 0 else 0.0

    print(f"     Unique EVA tokens: {len(eva_types)} (TTR={token_ttr:.3f})")
    print(f"     Unique decoded:    {len(decoded_types)} (TTR={decoded_ttr:.3f})")
    print(f"     Compression ratio: {compression_ratio:.3f}")
    print(f"     {'→ TABLE COLLAPSE: decode loses diversity' if compression_ratio < 0.7 else '→ Diversity preserved'}")

    # ── 3. Per-decoded-word diversity ──
    print("  3. Per-decoded-word analysis …")
    decoded_to_evas: Dict[str, List[str]] = defaultdict(list)
    decoded_to_triples: Dict[str, List[List[str]]] = defaultdict(list)
    for t in f57v_tokens:
        decoded_to_evas[t['decoded']].append(t['eva_original'])
        decoded_to_triples[t['decoded']].append(t['char_triples'])

    word_diversity = []
    for dec_word in sorted(decoded_to_evas.keys(),
                           key=lambda w: len(decoded_to_evas[w]), reverse=True):
        eva_list = decoded_to_evas[dec_word]
        if len(eva_list) < 3:
            continue
        unique_evas = set(eva_list)
        # Check if all instances use the same triples
        triple_sets = [tuple(t) for t in decoded_to_triples[dec_word]]
        unique_triples = set(triple_sets)

        word_diversity.append({
            'decoded_word': dec_word,
            'total_occurrences': len(eva_list),
            'n_unique_eva_tokens': len(unique_evas),
            'unique_eva_tokens': sorted(unique_evas),
            'n_unique_triple_patterns': len(unique_triples),
            'same_triples': len(unique_triples) == 1,
        })

    print("     Words with ≥3 occurrences:")
    for wd in word_diversity[:15]:
        same = "SAME" if wd['same_triples'] else "DIFF"
        print(f"       {wd['decoded_word']:<12s} ×{wd['total_occurrences']:>3d}  "
              f"{wd['n_unique_eva_tokens']} unique EVA  "
              f"{wd['n_unique_triple_patterns']} triple patterns [{same}]")

    # ── 4. Repetition structure analysis ──
    print("  4. Repetition structure analysis …")
    decoded_seq = [t['decoded'] for t in f57v_tokens]

    # Find repeated multi-token patterns (n-grams for n=2,3,4)
    repeated_blocks = {}
    for n in (2, 3, 4):
        ngrams: Dict[str, int] = Counter()
        for i in range(len(decoded_seq) - n + 1):
            gram = ' '.join(decoded_seq[i:i + n])
            ngrams[gram] += 1
        repeated = {gram: cnt for gram, cnt in ngrams.items() if cnt >= 2}
        repeated_blocks[n] = sorted(repeated.items(), key=lambda x: x[1],
                                    reverse=True)[:10]

    for n in (2, 3, 4):
        blocks = repeated_blocks[n]
        if blocks:
            print(f"     {n}-gram repeats (top 5):")
            for gram, cnt in blocks[:5]:
                print(f"       \"{gram}\" ×{cnt}")

    # Check periodicity via autocorrelation on decoded word IDs
    word_to_id = {w: i for i, w in enumerate(sorted(decoded_types))}
    id_seq = [word_to_id[d] for d in decoded_seq]
    max_lag = min(20, n_f57v // 2)
    autocorrs = []
    mean_id = sum(id_seq) / len(id_seq)
    var_id = sum((x - mean_id) ** 2 for x in id_seq) / len(id_seq)

    for lag in range(1, max_lag + 1):
        if var_id == 0:
            autocorrs.append({'lag': lag, 'autocorrelation': 0.0})
            continue
        cov = sum((id_seq[i] - mean_id) * (id_seq[i + lag] - mean_id)
                  for i in range(len(id_seq) - lag)) / (len(id_seq) - lag)
        autocorrs.append({
            'lag': lag,
            'autocorrelation': round(cov / var_id, 4),
        })

    peak_lag = max(autocorrs, key=lambda a: a['autocorrelation'])
    has_periodicity = peak_lag['autocorrelation'] > 0.3

    print(f"     Peak autocorrelation: lag={peak_lag['lag']}, "
          f"r={peak_lag['autocorrelation']:.4f}")
    print(f"     {'→ PERIODIC structure detected' if has_periodicity else '→ No strong periodicity'}")

    # ── 5. f57v section identification ──
    print("  5. Section identification …")
    corpus = load_corpus(verbose=False)
    f57v_page = corpus.get_page('f57v') if hasattr(corpus, 'get_page') else None
    section = ''
    if f57v_page:
        section = getattr(f57v_page, 'section', '')
        illustration = getattr(f57v_page, 'illustration', '')
        print(f"     Section: {section}")
        print(f"     Illustration: {illustration}")
    else:
        # Try pages dict
        if hasattr(corpus, 'pages') and 'f57v' in corpus.pages:
            page = corpus.pages['f57v']
            section = getattr(page, 'section', '')
            print(f"     Section: {section}")

    # ── 6. Comparison folio (lowest SIGNAL rate) ──
    print("  6. Comparison folio analysis …")
    top_folios = signal_data.get('top_signal_folios', [])
    # Find a folio with low SIGNAL rate that has enough tokens
    all_folio_set = set(token_folios)
    folio_token_counts = Counter(token_folios)
    low_signal_folio = None
    low_signal_rate = 1.0

    # Build SIGNAL rate per folio
    folio_signal_count: Dict[str, int] = Counter()
    for i, f in enumerate(token_folios):
        if token_classifications[i] == 'SIGNAL':
            folio_signal_count[f] += 1

    for fol in all_folio_set:
        n_tok = folio_token_counts[fol]
        if n_tok < 50:  # Need enough tokens
            continue
        s_rate = folio_signal_count.get(fol, 0) / n_tok
        if s_rate < low_signal_rate:
            low_signal_rate = s_rate
            low_signal_folio = fol

    comparison = {}
    if low_signal_folio:
        comp_positions = [i for i, f in enumerate(token_folios) if f == low_signal_folio]
        comp_evas = set(token_evas[i] for i in comp_positions)
        comp_decoded = set(token_decoded[i] for i in comp_positions)
        comp_n = len(comp_positions)
        comp_eva_ttr = len(comp_evas) / comp_n if comp_n > 0 else 0.0
        comp_dec_ttr = len(comp_decoded) / comp_n if comp_n > 0 else 0.0
        comp_compression = comp_dec_ttr / comp_eva_ttr if comp_eva_ttr > 0 else 0.0

        comparison = {
            'folio': low_signal_folio,
            'n_tokens': comp_n,
            'signal_rate': round(low_signal_rate, 4),
            'n_unique_eva': len(comp_evas),
            'n_unique_decoded': len(comp_decoded),
            'token_ttr': round(comp_eva_ttr, 3),
            'decoded_ttr': round(comp_dec_ttr, 3),
            'compression_ratio': round(comp_compression, 3),
        }

        print(f"     Comparison: {low_signal_folio} "
              f"(signal={low_signal_rate:.3f}, {comp_n} tokens)")
        print(f"     Comparison EVA TTR:     {comp_eva_ttr:.3f}")
        print(f"     Comparison decoded TTR: {comp_dec_ttr:.3f}")
        print(f"     Comparison compression: {comp_compression:.3f}")

    # ── 7. Verdict ──
    # Is the repetition from table collapse or genuine?
    if compression_ratio < 0.7:
        diversity_verdict = "TABLE_COLLAPSE"
        explanation = ("Diverse EVA tokens collapse to few decoded syllables — "
                       "the triple table loses information on f57v")
    elif compression_ratio < 0.9:
        diversity_verdict = "MODERATE_COLLAPSE"
        explanation = ("Some diversity loss in decoding, but partly genuine repetition")
    else:
        diversity_verdict = "GENUINE_REPETITION"
        explanation = ("EVA tokens are themselves repetitive — "
                       "f57v contains genuinely repetitive text (e.g., recipes)")

    print(f"\n  Verdict: {diversity_verdict}")
    print(f"  {explanation}")

    # ── 8. Save ──
    elapsed = time.time() - t0
    output = {
        'n_f57v_tokens': n_f57v,
        'n_unique_eva_tokens': len(eva_types),
        'n_unique_decoded': len(decoded_types),
        'token_ttr': round(token_ttr, 3),
        'decoded_ttr': round(decoded_ttr, 3),
        'compression_ratio': round(compression_ratio, 3),
        'word_diversity': word_diversity,
        'repeated_blocks': {str(n): [(g, c) for g, c in blocks]
                           for n, blocks in repeated_blocks.items()},
        'autocorrelations': autocorrs,
        'peak_autocorrelation': peak_lag,
        'has_periodicity': has_periodicity,
        'section': section,
        'comparison_folio': comparison,
        'diversity_verdict': diversity_verdict,
        'f57v_decoded_sequence': decoded_seq,
        'verdict': (
            f"f57v: {diversity_verdict}. "
            f"Compression={compression_ratio:.3f} "
            f"(EVA TTR={token_ttr:.3f}, decoded TTR={decoded_ttr:.3f}). "
            f"{explanation}"
        ),
        'runtime_seconds': round(elapsed, 1),
    }

    out_path = os.path.join(rd, 'f57v_eva_analysis.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(output), f, indent=2)
    print(f"\n  Saved → {out_path} ({elapsed:.1f}s)")
