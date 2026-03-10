"""
Step 38.8 – Macaronic Folio Examination
========================================
Produce annotated transliterations of the top SIGNAL folios with language
tagging, showing where Latin and Italian words appear.

Dependency chain:
    merged_signal.json         (Step 38.3)
    merged_decode.json         (Step 38.2)
    merged_dict.json           (Step 38.1)
    decode_10k.json            (Step 36.1)
    signal_10k.json            (Step 36.2 — for side-by-side comparison)
    f57v_eva_analysis.json     (Step 37.10 — for f57v check)
        → merged_folio.json    (this step)
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
# Core functions
# ---------------------------------------------------------------------------

def _annotate_folio(
    folio_tokens: List[Tuple[int, str]],
    decoded_lower: List[str],
    classifications: List[str],
    merged_dict: Set[str],
    latin_10k: Set[str],
    italian_10k: Set[str],
    signal_words: Set[str],
) -> List[Dict]:
    """Annotate tokens on a folio with language tags."""
    annotated = []
    for idx, _ in folio_tokens:
        w = decoded_lower[idx]
        cls = classifications[idx] if idx < len(classifications) else 'UNKNOWN'

        if cls == 'SIGNAL' and w in signal_words:
            if w in latin_10k and w in italian_10k:
                tag = 'SHARED'
            elif w in latin_10k:
                tag = 'LAT'
            elif w in italian_10k:
                tag = 'ITA'
            else:
                tag = 'SIGNAL'
        elif w in merged_dict:
            tag = 'SIGNAL'  # In merged dict but not individually confirmed
        else:
            tag = 'MISS'

        annotated.append({
            'position': idx,
            'decoded': w,
            'tag': tag,
            'classification': cls,
        })
    return annotated


def _signal_runs(
    annotated: List[Dict],
) -> List[Dict]:
    """Extract maximal consecutive SIGNAL/LAT/ITA/SHARED runs."""
    runs = []
    current_run = []
    signal_tags = {'SIGNAL', 'LAT', 'ITA', 'SHARED'}

    for entry in annotated:
        if entry['tag'] in signal_tags:
            current_run.append(entry)
        else:
            if len(current_run) >= 2:
                tags = Counter(e['tag'] for e in current_run)
                runs.append({
                    'length': len(current_run),
                    'words': [e['decoded'] for e in current_run],
                    'tags': [e['tag'] for e in current_run],
                    'start_pos': current_run[0]['position'],
                    'language_mix': dict(tags),
                    'has_italian': 'ITA' in tags,
                    'is_macaronic': len(set(tags.keys()) - {'SIGNAL'}) > 1,
                })
            current_run = []

    if len(current_run) >= 2:
        tags = Counter(e['tag'] for e in current_run)
        runs.append({
            'length': len(current_run),
            'words': [e['decoded'] for e in current_run],
            'tags': [e['tag'] for e in current_run],
            'start_pos': current_run[0]['position'],
            'language_mix': dict(tags),
            'has_italian': 'ITA' in tags,
            'is_macaronic': len(set(tags.keys()) - {'SIGNAL'}) > 1,
        })

    runs.sort(key=lambda x: x['length'], reverse=True)
    return runs


def _medical_phrase_check(
    run: Dict,
) -> Dict[str, bool]:
    """Check if a run contains medical phrase elements."""
    words = set(run['words'])
    pharma_verbs = {'cola', 'recipe', 'misce', 'coque', 'dice', 'cura',
                    'sana', 'bibe', 'beni'}
    body_parts = {'cora', 'core', 'corpo', 'carne', 'ossa', 'pede',
                  'manu', 'dente', 'naso'}
    ingredients = {'rosa', 'sale', 'vino', 'olio', 'bene', 'sene',
                   'calce', 'suco'}
    qualities = {'bela', 'bona', 'calida', 'frigida', 'sicca',
                 'dulce', 'rara', 'nova'}

    return {
        'has_verb': bool(words & pharma_verbs),
        'has_body_part': bool(words & body_parts),
        'has_ingredient': bool(words & ingredients),
        'has_quality': bool(words & qualities),
    }


def _f57v_venetian_check(
    f57v_tokens: List[Dict],
    italian_10k: Set[str],
) -> Dict[str, Any]:
    """Check f57v unique words against Italian vocabulary."""
    f57v_words = set(e['decoded'] for e in f57v_tokens)
    # Italian verb forms from common verbs
    venetian_verbs = {'fa', 'ha', 'va', 'sa', 'da', 'di', 'se', 'ne',
                      'me', 'te', 'lo', 'la', 'li', 'le', 'no', 'si'}
    matches = f57v_words & italian_10k
    verb_matches = f57v_words & venetian_verbs

    return {
        'n_f57v_unique': len(f57v_words),
        'n_italian_matches': len(matches),
        'italian_matches': sorted(matches),
        'n_venetian_verb_forms': len(verb_matches),
        'venetian_verb_forms': sorted(verb_matches),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_merged_folio() -> None:
    """Step 38.8: Macaronic Folio Examination."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 38.8: Macaronic Folio Examination")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load inputs ──
    print("\n  1. Loading inputs …")
    signal_data = _safe_load(os.path.join(rd, 'merged_signal.json'))
    dict_data = _safe_load(os.path.join(rd, 'merged_dict.json'))
    signal_10k = _safe_load(os.path.join(rd, 'signal_10k.json'))

    classifications = signal_data.get('token_classifications', [])
    decoded_lower = signal_data.get('token_decoded', [])
    token_folios = signal_data.get('token_folios', [])

    merged_dict = set(dict_data.get('merged_words', []))
    latin_10k = set(dict_data.get('latin_10k_words', []))
    italian_10k = set(dict_data.get('italian_10k_words', []))

    signal_words = set(w['word'] for w in signal_data.get('word_signals', []))
    folio_ranking = signal_data.get('folio_ranking', [])

    print(f"     {len(decoded_lower)} tokens, {len(signal_words)} signal words")

    # ── 2. Select top folios ──
    print("  2. Selecting top SIGNAL folios …")
    top_folios = [fr['folio'] for fr in folio_ranking[:4]]
    print(f"     Top folios: {top_folios}")

    # Build folio → token index mapping
    folio_tokens: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
    for i, folio in enumerate(token_folios):
        folio_tokens[folio].append((i, decoded_lower[i]))

    # ── 3. Annotate each folio ──
    print("  3. Annotating folios …")
    folio_results = []

    for folio in top_folios:
        tokens = folio_tokens.get(folio, [])
        if not tokens:
            continue

        annotated = _annotate_folio(
            tokens, decoded_lower, classifications,
            merged_dict, latin_10k, italian_10k, signal_words,
        )

        # Language flow
        tag_counts = Counter(e['tag'] for e in annotated)
        n_total = len(annotated)

        # Signal runs
        runs = _signal_runs(annotated)
        macaronic_runs = [r for r in runs if r.get('is_macaronic')]

        # Annotated text (first 50 tokens)
        text_preview = ' '.join(
            f"[{e['tag']}:{e['decoded']}]" for e in annotated[:50]
        )

        # Medical phrases
        medical_runs = []
        for run in runs:
            check = _medical_phrase_check(run)
            n_medical = sum(check.values())
            if n_medical >= 2:
                medical_runs.append({
                    'words': run['words'],
                    'tags': run['tags'],
                    **check,
                })

        folio_results.append({
            'folio': folio,
            'n_tokens': n_total,
            'tag_counts': dict(tag_counts),
            'signal_rate': round(
                sum(1 for e in annotated if e['tag'] != 'MISS') / n_total, 4
            ) if n_total else 0.0,
            'n_lat': tag_counts.get('LAT', 0),
            'n_ita': tag_counts.get('ITA', 0),
            'n_shared': tag_counts.get('SHARED', 0),
            'n_miss': tag_counts.get('MISS', 0),
            'text_preview': text_preview,
            'n_runs': len(runs),
            'n_macaronic_runs': len(macaronic_runs),
            'longest_run': runs[0]['length'] if runs else 0,
            'runs': runs[:15],
            'macaronic_runs': macaronic_runs[:10],
            'n_medical': len(medical_runs),
            'medical_runs': medical_runs[:10],
        })

        print(f"\n     {folio}:")
        print(f"       Tokens: {n_total}")
        print(f"       LAT: {tag_counts.get('LAT', 0)}, "
              f"ITA: {tag_counts.get('ITA', 0)}, "
              f"SHARED: {tag_counts.get('SHARED', 0)}, "
              f"MISS: {tag_counts.get('MISS', 0)}")
        print(f"       Runs: {len(runs)} (longest={runs[0]['length'] if runs else 0})")
        print(f"       Macaronic runs: {len(macaronic_runs)}")
        if runs:
            print(f"       Best run: {' '.join(runs[0]['words'][:8])}")

    # ── 4. f57v check ──
    print("\n  4. f57v Venetian vocabulary check …")
    f57v_tokens = folio_tokens.get('f57v', [])
    f57v_annotated = _annotate_folio(
        f57v_tokens, decoded_lower, classifications,
        merged_dict, latin_10k, italian_10k, signal_words,
    ) if f57v_tokens else []

    f57v_result = _f57v_venetian_check(f57v_annotated, italian_10k)
    print(f"     f57v unique words: {f57v_result['n_f57v_unique']}")
    print(f"     Italian matches: {f57v_result['n_italian_matches']}")
    print(f"     Venetian verb forms: {f57v_result['venetian_verb_forms']}")

    # ── 5. Best macaronic fragment ──
    print("\n  5. Best macaronic fragment …")
    all_runs = []
    for fr in folio_results:
        for run in fr.get('runs', []):
            run['folio'] = fr['folio']
            all_runs.append(run)

    # Score: length × (number of language types) × (has Italian)
    for run in all_runs:
        n_lang_types = len(set(run['tags']) - {'SIGNAL', 'MISS'})
        run['score'] = run['length'] * max(n_lang_types, 1) * (2 if run.get('has_italian') else 1)

    all_runs.sort(key=lambda x: x['score'], reverse=True)
    best_fragment = all_runs[0] if all_runs else None

    if best_fragment:
        print(f"     Best fragment ({best_fragment['folio']}, "
              f"length={best_fragment['length']}, score={best_fragment['score']}):")
        tagged = ' '.join(f"[{t}:{w}]" for w, t in
                         zip(best_fragment['words'], best_fragment['tags']))
        print(f"       {tagged}")

    # ── 6. Side-by-side comparison ──
    print("\n  6. Side-by-side: Latin-only vs Macaronic …")
    latin_classifications = signal_10k.get('token_classifications', [])
    comparison_folio = top_folios[0] if top_folios else None

    side_by_side = None
    if comparison_folio and latin_classifications:
        tokens = folio_tokens.get(comparison_folio, [])
        latin_signal = sum(1 for idx, _ in tokens
                          if idx < len(latin_classifications) and
                          latin_classifications[idx] == 'SIGNAL')
        merged_signal = sum(1 for idx, _ in tokens
                           if idx < len(classifications) and
                           classifications[idx] == 'SIGNAL')
        side_by_side = {
            'folio': comparison_folio,
            'latin_signal_count': latin_signal,
            'merged_signal_count': merged_signal,
            'latin_signal_rate': round(latin_signal / len(tokens), 4) if tokens else 0.0,
            'merged_signal_rate': round(merged_signal / len(tokens), 4) if tokens else 0.0,
        }
        print(f"     {comparison_folio}: Latin SIGNAL={latin_signal}, "
              f"Merged SIGNAL={merged_signal}")

    # ── 7. Save ──
    elapsed = time.time() - t0

    output = {
        'top_folios': top_folios,
        'folio_results': folio_results,
        'f57v_venetian': f57v_result,
        'best_fragment': best_fragment,
        'side_by_side': side_by_side,
        'verdict': (
            f"Top folios: {', '.join(top_folios)}. "
            f"Best fragment: {best_fragment['length'] if best_fragment else 0} tokens "
            f"on {best_fragment['folio'] if best_fragment else 'none'}. "
            f"f57v Italian matches: {f57v_result['n_italian_matches']}."
        ),
        'runtime_seconds': round(elapsed, 1),
    }

    out_path = os.path.join(rd, 'merged_folio.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(output), f, indent=2)
    print(f"\n  Saved → {out_path} ({elapsed:.1f}s)")
