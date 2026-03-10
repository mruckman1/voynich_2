"""
Step 36.6 – Folio Examination at 10K
======================================
Produces annotated transliterations of the top SIGNAL folios using the
10K classification.  These are the human-readable outputs at the project's
strongest signal level (z=13.12).

Dependency chain:
    signal_10k.json           (Step 36.2)
    bigrams_10k.json          (Step 36.3)
    bootstrap_10k.json        (Step 36.5)
    decode_10k.json           (Step 36.1)
        → folio_10k.json     (this step)
"""

import json
import os
import time
from collections import defaultdict
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


# Latin grammar parse heuristic
_FUNCTION_WORDS = {
    'de', 'in', 'ad', 'cum', 'per', 'pro', 'sub', 'ex', 'ab',
    'et', 'vel', 'aut', 'sed', 'si', 'ne', 'ut', 'non', 'que',
    'a', 'e', 'se', 'te', 'me',
}

_VERB_ENDINGS = ('are', 'ere', 'ire', 'tur', 'nt', 'mus', 'tis')
_NOUN_ENDINGS = ('us', 'um', 'am', 'em', 'ae', 'is', 'os', 'as', 'es', 'a', 'o', 'i', 'e')


def _parse_latin_score(words: List[str]) -> float:
    """Heuristic Latin parse score for a word sequence (0.0–1.0)."""
    if not words:
        return 0.0

    score = 0.0
    for i, w in enumerate(words):
        # Function word before content word
        if w in _FUNCTION_WORDS:
            score += 0.3
        # Verb ending
        elif any(w.endswith(e) and len(w) > len(e) + 1 for e in _VERB_ENDINGS):
            score += 0.2
        # Noun ending
        elif any(w.endswith(e) and len(w) > len(e) for e in _NOUN_ENDINGS):
            score += 0.15
        else:
            score += 0.05

        # Bonus: prep + noun pattern
        if i > 0 and words[i - 1] in _FUNCTION_WORDS and w not in _FUNCTION_WORDS:
            score += 0.2

    return min(score / len(words), 1.0)


def _is_medical_template(words: List[str]) -> bool:
    """Check if word sequence matches a medical formula pattern."""
    text = ' '.join(words)
    medical_triggers = [
        'recipe', 'coque', 'misce', 'cola', 'tere', 'adde',
        'herba', 'radix', 'folium', 'aqua', 'calida', 'frigida',
    ]
    return any(t in text for t in medical_triggers)


# ---------------------------------------------------------------------------
# Annotation
# ---------------------------------------------------------------------------

def _annotate_folio(
    folio: str,
    token_folios: List[str],
    token_decoded: List[str],
    token_evas: List[str],
    classifications: List[str],
    hits_10k: List[bool],
    hits_131k: List[bool],
    signal_words_10k: Set[str],
    bootstrap_words: Set[str],
) -> List[Dict]:
    """Annotate each token on a folio with its 10K classification tag."""
    tokens = []
    for i in range(len(token_folios)):
        if token_folios[i] != folio:
            continue

        word = token_decoded[i]
        cl = classifications[i]
        h10k = hits_10k[i]
        h131k = hits_131k[i] if i < len(hits_131k) else False

        if word in signal_words_10k and cl == 'SIGNAL':
            tag = 'CONFIRMED-10K'
        elif word in bootstrap_words:
            tag = 'BOOT-10K'
        elif cl == 'SIGNAL':
            tag = 'SIGNAL-10K'
        elif not h10k and h131k:
            tag = 'HIT-131K-ONLY'
        elif not h10k:
            tag = 'MISS-10K'
        else:
            tag = cl  # SHARED_HIT, SHARED_MISS, ANTI_SIGNAL

        tokens.append({
            'idx': i,
            'eva': token_evas[i],
            'decoded': word,
            'tag': tag,
            'classification': cl,
            'hit_10k': h10k,
        })
    return tokens


def _find_signal_runs(annotated: List[Dict]) -> List[Dict]:
    """Find maximal consecutive SIGNAL runs on a folio."""
    runs = []
    i = 0
    while i < len(annotated):
        if annotated[i]['classification'] != 'SIGNAL':
            i += 1
            continue
        start = i
        while i < len(annotated) and annotated[i]['classification'] == 'SIGNAL':
            i += 1
        length = i - start
        words = [annotated[j]['decoded'] for j in range(start, i)]
        runs.append({
            'start_idx': annotated[start]['idx'],
            'length': length,
            'words': words,
            'text': ' '.join(words),
            'parse_score': round(_parse_latin_score(words), 3),
            'is_medical': _is_medical_template(words),
        })
    runs.sort(key=lambda r: -r['length'])
    return runs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_folio_10k() -> None:
    """Step 36.6: Folio examination at 10K dictionary."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 36.6: Folio Examination at 10K")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load inputs ──
    print("\n  1. Loading inputs …")
    with open(os.path.join(rd, 'signal_10k.json')) as f:
        sig_data = json.load(f)

    token_folios = sig_data['token_folios']
    token_decoded = sig_data['token_decoded']
    token_evas = sig_data['token_evas']
    classifications = sig_data['token_classifications']
    hits_10k = sig_data['token_hits_10k']

    signal_words_10k = set(
        w['word'] for w in sig_data.get('word_signals', [])
        if w.get('is_genuine_signal', False)
    )

    # 131K hits for HIT-131K-ONLY tag
    with open(os.path.join(rd, 'decode_10k.json')) as f:
        decode_data = json.load(f)
    hits_131k = decode_data.get('token_hits_131k', [])

    # Bootstrap words
    bootstrap_words: Set[str] = set()
    boot_path = os.path.join(rd, 'bootstrap_10k.json')
    if os.path.exists(boot_path):
        with open(boot_path) as f:
            boot_data = json.load(f)
        bootstrap_words = set(boot_data.get('accepted_words', []))

    # Top folios
    top_folios_data = sig_data.get('top_signal_folios', [])[:4]
    top_folios = [f['folio'] for f in top_folios_data]
    print(f"     Top 4 SIGNAL folios: {top_folios}")

    # Bigram catalog
    bigram_catalog = []
    bg_path = os.path.join(rd, 'bigrams_10k.json')
    if os.path.exists(bg_path):
        with open(bg_path) as f:
            bg_data = json.load(f)
        bigram_catalog = bg_data.get('bigram_catalog', [])

    # Phase 29 data for side-by-side comparison
    phase29_classifications = []
    p29_path = os.path.join(rd, 'signal_bigrams.json')
    if os.path.exists(p29_path):
        with open(p29_path) as f:
            p29_data = json.load(f)
        phase29_classifications = p29_data.get('token_classifications', [])

    # ── 2. Annotate each folio ──
    print("  2. Annotating top folios …")
    folio_results = []

    for folio in top_folios:
        print(f"\n     --- {folio} ---")

        annotated = _annotate_folio(
            folio, token_folios, token_decoded, token_evas,
            classifications, hits_10k, hits_131k,
            signal_words_10k, bootstrap_words,
        )

        if not annotated:
            continue

        # Tag counts
        tag_counts = defaultdict(int)
        for a in annotated:
            tag_counts[a['tag']] += 1

        print(f"     {len(annotated)} tokens")
        for tag, count in sorted(tag_counts.items()):
            print(f"       {tag}: {count}")

        # Signal runs
        runs = _find_signal_runs(annotated)
        print(f"     SIGNAL runs: {len(runs)}")
        for r in runs[:5]:
            print(f"       len={r['length']} parse={r['parse_score']:.2f}"
                  f"  [{r['text']}]")

        # Annotated text (first 60 tokens)
        annotated_text = ' '.join(
            f"[{a['tag']}:{a['decoded']}]"
            for a in annotated[:60]
        )

        # Bigram matches on this folio
        folio_bigrams = [b for b in bigram_catalog if b['folio'] == folio]

        # Side-by-side: Phase 29 (131K) vs Phase 36 (10K)
        side_by_side = []
        if phase29_classifications:
            for a in annotated[:40]:
                idx = a['idx']
                p29_cl = phase29_classifications[idx] if idx < len(phase29_classifications) else '?'
                side_by_side.append({
                    'idx': idx,
                    'decoded': a['decoded'],
                    'tag_10k': a['tag'],
                    'class_131k': p29_cl,
                    'class_10k': a['classification'],
                })

        folio_results.append({
            'folio': folio,
            'n_tokens': len(annotated),
            'tag_counts': dict(tag_counts),
            'signal_runs': runs,
            'longest_run': runs[0]['length'] if runs else 0,
            'best_run_text': runs[0]['text'] if runs else '',
            'best_run_parse': runs[0]['parse_score'] if runs else 0.0,
            'annotated_sample': annotated_text,
            'bigram_matches': folio_bigrams,
            'side_by_side': side_by_side[:40],
        })

    # ── 3. Best fragment across all folios ──
    print("\n  3. Selecting best fragment …")
    all_runs = []
    for fr in folio_results:
        for r in fr['signal_runs']:
            if r['length'] >= 3:
                all_runs.append({**r, 'folio': fr['folio']})
    all_runs.sort(key=lambda r: (-r['parse_score'], -r['length']))

    best_fragment = all_runs[0] if all_runs else None
    if best_fragment:
        print(f"     Best: {best_fragment['folio']} len={best_fragment['length']}"
              f"  parse={best_fragment['parse_score']:.2f}")
        print(f"     Text: {best_fragment['text']}")
    else:
        print("     No fragments of length ≥ 3")

    # ── 4. Save ──
    elapsed = time.time() - t0

    output = {
        'top_folios': top_folios,
        'folio_results': folio_results,
        'best_fragment': best_fragment,
        'total_signal_runs': sum(len(fr['signal_runs']) for fr in folio_results),
        'longest_run_overall': max(
            (fr['longest_run'] for fr in folio_results), default=0,
        ),
        'n_parseable_fragments': sum(
            1 for r in all_runs if r['parse_score'] >= 0.3
        ),
        'runtime_seconds': round(elapsed, 1),
    }

    out_path = os.path.join(rd, 'folio_10k.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(output), f, indent=2)
    print(f"\n  Saved → {out_path}")

    # ── Summary ──
    print("\n" + "=" * 70)
    print("FOLIO 10K SUMMARY")
    print("=" * 70)
    for fr in folio_results:
        print(f"\n  {fr['folio']}: {fr['n_tokens']} tokens, "
              f"{fr['tag_counts'].get('SIGNAL-10K', 0) + fr['tag_counts'].get('CONFIRMED-10K', 0)} SIGNAL, "
              f"longest run = {fr['longest_run']}")
        if fr['best_run_text']:
            print(f"    Best: [{fr['best_run_text']}] (parse={fr['best_run_parse']:.2f})")
    if best_fragment:
        print(f"\n  BEST FRAGMENT: {best_fragment['folio']}")
        print(f"    {best_fragment['text']}")
        print(f"    parse={best_fragment['parse_score']:.2f}  len={best_fragment['length']}")
    print(f"\n  Runtime: {elapsed:.1f}s")
