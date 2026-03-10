"""
Step 35.7 – Combined Folio Examination
========================================
Produce annotated transliterations of the top SIGNAL folios under the
combined spatial+10K model.  Tags include determinative identity,
ligature status, and SIGNAL classification.

Dependency chain:
    combined_bigrams.json      (Step 35.4 — per-token cache)
    combined_signal.json       (Step 35.3 — signal words + folios)
    spatial_preprocess.json    (Step 35.1 — determinatives, strategies)
    signal_bigrams.json        (Phase 29 — for side-by-side comparison)
        → combined_folio.json  (this step)
"""

import json
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

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


# ---------------------------------------------------------------------------
# Medical Latin templates
# ---------------------------------------------------------------------------

_MEDICAL_VERBS = {'recipe', 'coque', 'cola', 'misce', 'fac', 'da', 'adde',
                  'pone', 'tere', 'solve', 'distilla', 'fiat'}
_PREPOSITIONS = {'de', 'in', 'ad', 'cum', 'per', 'pro', 'sub', 'ex', 'ab'}
_CONJUNCTIONS = {'et', 'vel', 'aut', 'sed', 'si', 'ne', 'ut'}


def _parse_latin_fragment(words: List[str], ref_word_set: set) -> Dict:
    """Attempt to parse a sequence of Latin words as a phrase."""
    n = len(words)
    parse_score = 0.0
    structure = "unknown"
    medical = False

    all_hit = all(w in ref_word_set for w in words)
    if all_hit:
        parse_score += 0.3

    if n >= 2 and words[0] in _PREPOSITIONS:
        structure = "prepositional_phrase"
        parse_score += 0.3

    if n >= 2 and words[0] in _MEDICAL_VERBS:
        structure = "medical_formula"
        parse_score += 0.4
        medical = True

    if n >= 3:
        for i in range(1, n - 1):
            if words[i] in _CONJUNCTIONS:
                structure = "coordinated"
                parse_score += 0.2
                break

    if n >= 4:
        parse_score += 0.1
    if n >= 6:
        parse_score += 0.1

    return {
        'words': words,
        'length': n,
        'all_dict_hits': all_hit,
        'structure': structure,
        'parse_score': round(min(parse_score, 1.0), 3),
        'medical_plausibility': medical,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_combined_folio() -> None:
    """Step 35.7: Combined spatial+10K folio examination."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 35.7: Combined Folio Examination")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load inputs ──
    print("\n  1. Loading inputs ...")
    with open(os.path.join(rd, 'combined_bigrams.json')) as f:
        cb = json.load(f)
    with open(os.path.join(rd, 'combined_signal.json')) as f:
        cs = json.load(f)

    decoded = cb['token_decoded']
    classifications = cb['token_classifications']
    folios = cb['token_folios']
    evas = cb['token_evas']
    dict_hits = cb['token_dict_hits']
    n_tokens = cb['n_tokens']

    signal_words = set(ws['word'] for ws in cs['word_signals']
                       if ws.get('is_genuine', False))

    # Load spatial info (determinatives, strategies)
    strategies: List[str] = [''] * n_tokens
    determinatives: List[Optional[str]] = [None] * n_tokens
    sp_path = os.path.join(rd, 'spatial_preprocess.json')
    if os.path.exists(sp_path):
        with open(sp_path) as f:
            sp = json.load(f)
        strategies = sp.get('token_strategies', strategies)
        determinatives = sp.get('token_determinatives', determinatives)

    # Build 10K word set
    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    base_words = set(
        w.lower() for w in ref_corpus.get_combined_tokens('latin')
        if len(w) >= 2
    )
    variants = _build_dict_variants(base_words, ref_corpus, [10000])
    dict_10k = variants[0][1]

    # Load Phase 29 for side-by-side
    phase29_decoded: Optional[List[str]] = None
    p29_path = os.path.join(rd, 'signal_bigrams.json')
    if os.path.exists(p29_path):
        with open(p29_path) as f:
            p29 = json.load(f)
        phase29_decoded = p29.get('token_decoded')

    # ── 2. Select top SIGNAL folios ──
    print("\n  2. Selecting top SIGNAL folios ...")
    top_folios = cs.get('top_signal_folios', [])[:4]
    if not top_folios:
        folio_total: Counter = Counter()
        folio_signal: Counter = Counter()
        for i in range(n_tokens):
            folio_total[folios[i]] += 1
            if classifications[i] == 'SIGNAL':
                folio_signal[folios[i]] += 1
        ranked = sorted(folio_total.keys(),
                        key=lambda f: folio_signal.get(f, 0) / max(folio_total[f], 1),
                        reverse=True)
        top_folios = [{'folio': f, 'n_tokens': folio_total[f],
                       'n_signal': folio_signal.get(f, 0)} for f in ranked[:4]]

    target_folios = [tf['folio'] for tf in top_folios]
    print(f"     Top folios: {target_folios}")

    # ── 3. Annotate each folio ──
    print("\n  3. Annotating top folios ...")
    annotated_folios = []

    for folio_id in target_folios:
        indices = [i for i in range(n_tokens) if folios[i] == folio_id]
        if not indices:
            continue

        annotations = []
        for i in indices:
            tag = classifications[i]
            if decoded[i] in signal_words:
                tag = 'CONFIRMED'

            det = determinatives[i] if i < len(determinatives) else None
            strat = strategies[i] if i < len(strategies) else ''

            annotations.append({
                'pos': i,
                'eva': evas[i],
                'decoded': decoded[i],
                'tag': tag,
                'dict_hit': dict_hits[i],
                'determinative': det,
                'strategy': strat,
            })

        # Build text representation with spatial tags
        text_lines = []
        for ann in annotations:
            parts = []
            if ann['determinative']:
                parts.append(f"[DET:{ann['determinative']}]")
            if ann['strategy'] == 'intersecting_kept':
                parts.append("[LIG]")
            parts.append(f"[{ann['tag']}:{ann['decoded']}]")
            text_lines.append(' '.join(parts))

        # SIGNAL runs
        signal_runs = []
        j = 0
        while j < len(indices):
            if (classifications[indices[j]] in ('SIGNAL', 'CONFIRMED') and
                    dict_hits[indices[j]]):
                run_start = j
                while (j < len(indices) and
                       dict_hits[indices[j]] and
                       classifications[indices[j]] in (
                           'SIGNAL', 'SHARED_HIT', 'CONFIRMED')):
                    j += 1
                run_len = j - run_start
                if run_len >= 2:
                    run_words = [decoded[indices[k]] for k in range(run_start, j)]
                    n_sig = sum(1 for k in range(run_start, j)
                                if classifications[indices[k]] in (
                                    'SIGNAL', 'CONFIRMED'))
                    signal_runs.append({
                        'start': run_start,
                        'length': run_len,
                        'words': run_words,
                        'n_signal': n_sig,
                        'parse': _parse_latin_fragment(run_words, dict_10k),
                    })
            else:
                j += 1

        signal_runs.sort(key=lambda r: (-r['length'], -r['parse']['parse_score']))

        # Side-by-side with Phase 29
        side_by_side = None
        if phase29_decoded and len(phase29_decoded) == n_tokens:
            p29_words = [phase29_decoded[i] for i in indices]
            p35_words = [decoded[i] for i in indices]
            changed = sum(1 for a, b in zip(p29_words, p35_words) if a != b)
            side_by_side = {
                'phase29_sample': p29_words[:30],
                'phase35_sample': p35_words[:30],
                'n_changed': changed,
                'n_total': len(indices),
                'change_rate': round(changed / len(indices), 4) if indices else 0.0,
            }

        n_sig = sum(1 for ann in annotations
                    if ann['tag'] in ('SIGNAL', 'CONFIRMED'))
        annotated_folios.append({
            'folio': folio_id,
            'n_tokens': len(indices),
            'n_signal': n_sig,
            'signal_rate': round(n_sig / len(indices), 4) if indices else 0.0,
            'annotations': annotations[:50],
            'text_lines': text_lines[:50],
            'signal_runs': signal_runs[:10],
            'side_by_side': side_by_side,
        })

        print(f"     {folio_id}: {len(indices)} tokens, {n_sig} SIGNAL, "
              f"{len(signal_runs)} runs")
        for sr in signal_runs[:3]:
            print(f"       run len={sr['length']}: {' '.join(sr['words'][:8])}")

    # ── 4. Best fragment ──
    print("\n  4. Selecting best fragment ...")
    all_runs = []
    for af in annotated_folios:
        for sr in af.get('signal_runs', []):
            all_runs.append({
                'folio': af['folio'],
                **sr,
            })

    all_runs.sort(key=lambda r: (-r['parse']['parse_score'], -r['length']))

    best_fragment = all_runs[0] if all_runs else None
    if best_fragment:
        print(f"     Best: {best_fragment['folio']} "
              f"len={best_fragment['length']} "
              f"score={best_fragment['parse']['parse_score']:.3f}")
        print(f"       {' '.join(best_fragment['words'])}")
    else:
        print("     No fragments found")

    # ── 5. Save ──
    print("\n  5. Saving combined_folio.json ...")
    output = {
        'top_folios': target_folios,
        'annotated_folios': annotated_folios,
        'best_fragment': _convert(best_fragment) if best_fragment else None,
        'n_total_runs': len(all_runs),
        'top_runs': [_convert(r) for r in all_runs[:20]],
        'runtime_seconds': round(time.time() - t0, 1),
    }

    with open(os.path.join(rd, 'combined_folio.json'), 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n  Step 35.7 completed in {time.time() - t0:.1f}s")
