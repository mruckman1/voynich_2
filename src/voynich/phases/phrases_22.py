"""
Phase 22.6 – Phrase Detection and Botanical Cross-Check (phrases-22)
=====================================================================
Sliding window phrase detection on Viterbi-segmented text.
Scores against Latin trigram model + dict hit rate + medical formula
templates. Cross-checks against botanical folios.

Dependency chain:
    corpus_decode_22.json (22.4) + readability_22.json (22.5)
    + Latin reference corpus
        → phrases_22.json (this step)
"""

import json
import math
import os
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.reference import (
    LATIN_PHRASE_PATTERNS,
    PHARMACEUTICAL_VOCABULARY,
    build_expanded_word_set,
    load_reference_corpus,
)


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


def _load_json(path: str) -> Optional[Dict]:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Phrase detection
# ---------------------------------------------------------------------------

# Medical formula templates
_MEDICAL_TEMPLATES = [
    'recipe', 'coque in aqua', 'misce cum', 'fac emplastrum',
    'est calida', 'est frigida', 'valet contra', 'in primo gradu',
    'in secundo gradu', 'in tertio gradu', 'folia eius',
    'radix eius', 'semen eius', 'succus eius', 'herba est',
    'aqua rosae', 'oleum', 'cum melle', 'bibat', 'comedat',
    'per dies', 'mane et sero', 'ante cibum', 'post cibum',
]

# Botanical vocabulary
_BOTANICAL_TERMS = {
    'herba', 'folia', 'radix', 'semen', 'flos', 'cortex', 'succus',
    'rosa', 'salvia', 'mentha', 'absinthium', 'plantago', 'urtica',
    'calendula', 'camomilla', 'lavandula', 'thymus', 'anethum',
    'petroselinum', 'apium', 'ruta', 'artemisia', 'verbena',
    'betonica', 'centaurea', 'gentiana', 'malva', 'viola',
}

# Herbal folios (f1-f56, f87-f102)
_HERBAL_FOLIOS = set(f'f{n}' for n in list(range(1, 57)) + list(range(87, 103)))


def _sliding_window_phrases(
    words: List[str],
    ref_word_set: set,
    ref_trigrams: set,
    min_len: int = 3,
    max_len: int = 8,
) -> List[Dict]:
    """Detect phrase-like sequences via sliding window."""
    hits: List[Dict] = []

    for start in range(len(words)):
        for length in range(min_len, min(max_len + 1, len(words) - start + 1)):
            window = words[start:start + length]
            phrase = ' '.join(window)

            # Score: dict hit rate
            dict_hits = sum(1 for w in window if w.lower() in ref_word_set)
            dict_rate = dict_hits / length

            # Score: trigram coverage
            if length >= 3:
                tri_hits = sum(
                    1 for i in range(length - 2)
                    if (window[i], window[i + 1], window[i + 2]) in ref_trigrams
                )
                tri_rate = tri_hits / (length - 2)
            else:
                tri_rate = 0.0

            # Score: medical template match
            template_match = any(t in phrase.lower() for t in _MEDICAL_TEMPLATES)

            # Combined score
            score = dict_rate * 0.4 + tri_rate * 0.4 + (0.2 if template_match else 0.0)

            if score >= 0.6 and dict_rate >= 0.5:
                hits.append({
                    'phrase': phrase,
                    'start': start,
                    'length': length,
                    'dict_rate': round(dict_rate, 3),
                    'trigram_rate': round(tri_rate, 3),
                    'template_match': template_match,
                    'score': round(score, 3),
                })

    # Deduplicate: keep highest-scoring overlapping phrases
    if not hits:
        return hits

    hits.sort(key=lambda x: -x['score'])
    kept: List[Dict] = []
    used_positions: set = set()
    for h in hits:
        positions = set(range(h['start'], h['start'] + h['length']))
        if not positions & used_positions:
            kept.append(h)
            used_positions |= positions

    return sorted(kept, key=lambda x: x['start'])


def _botanical_cross_check(
    per_folio_words: Dict[str, List[str]],
    ref_word_set: set,
) -> Dict[str, Any]:
    """Check if herbal folios have more botanical vocabulary than others."""
    herbal_botanical = 0
    herbal_total = 0
    other_botanical = 0
    other_total = 0

    for folio, words in per_folio_words.items():
        is_herbal = any(folio.startswith(hf) for hf in _HERBAL_FOLIOS)
        dict_words = [w for w in words if w.lower() in ref_word_set]
        botanical_hits = sum(1 for w in dict_words if w.lower() in _BOTANICAL_TERMS)

        if is_herbal:
            herbal_botanical += botanical_hits
            herbal_total += len(dict_words)
        else:
            other_botanical += botanical_hits
            other_total += len(dict_words)

    herbal_rate = herbal_botanical / max(herbal_total, 1)
    other_rate = other_botanical / max(other_total, 1)
    selectivity = herbal_rate / max(other_rate, 0.001)

    # Permutation test
    rng = random.Random(42)
    all_folios = list(per_folio_words.keys())
    n_herbal = sum(1 for f in all_folios if any(f.startswith(hf) for hf in _HERBAL_FOLIOS))
    null_sels: List[float] = []
    for _ in range(1000):
        rng.shuffle(all_folios)
        shuffled_herbal = set(all_folios[:n_herbal])
        sh_bot = 0
        sh_tot = 0
        so_bot = 0
        so_tot = 0
        for folio, words in per_folio_words.items():
            dict_words = [w for w in words if w.lower() in ref_word_set]
            bot = sum(1 for w in dict_words if w.lower() in _BOTANICAL_TERMS)
            if folio in shuffled_herbal:
                sh_bot += bot
                sh_tot += len(dict_words)
            else:
                so_bot += bot
                so_tot += len(dict_words)
        ns = (sh_bot / max(sh_tot, 1)) / max(so_bot / max(so_tot, 1), 0.001)
        null_sels.append(ns)

    p_value = sum(1 for ns in null_sels if ns >= selectivity) / len(null_sels)

    return {
        'herbal_botanical_rate': round(herbal_rate, 4),
        'other_botanical_rate': round(other_rate, 4),
        'selectivity': round(selectivity, 2),
        'p_value': round(p_value, 4),
        'herbal_botanical_count': herbal_botanical,
        'other_botanical_count': other_botanical,
    }


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Phrases22Result:
    timestamp: str
    n_phrases_detected: int
    top_phrases: List[Dict]
    botanical_cross_check: Dict[str, Any]
    per_section_phrases: Dict[str, int]
    template_hits: List[Dict]
    n_template_hits: int
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_phrases_22() -> Dict[str, Any]:
    """Phrase detection and botanical cross-check."""
    t0 = time.time()
    rdir = _results_dir()

    # Load decode results
    decode_data = _load_json(str(rdir / "corpus_decode_22.json")) or {}
    readability = _load_json(str(rdir / "readability_22.json")) or {}
    better_mode = readability.get('better_mode', 'a')

    mode_data = decode_data.get(f'mode_{better_mode}', {})

    # Build reference
    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    ref_tokens = ref_corpus.get_combined_tokens('latin')
    ref_words = [w.lower() for w in ref_tokens if len(w) >= 2]
    base_words = set(ref_words)
    expanded_words, _ = build_expanded_word_set(base_words)
    ref_word_set = base_words | expanded_words

    # Build reference trigrams (word-level)
    ref_trigrams: set = set()
    for i in range(len(ref_words) - 2):
        ref_trigrams.add((ref_words[i], ref_words[i + 1], ref_words[i + 2]))

    # Extract Viterbi-segmented words per folio
    per_folio_words: Dict[str, List[str]] = {}
    all_viterbi_words: List[str] = []

    for vs in mode_data.get('viterbi_sample', []):
        folio = vs.get('folio', '')
        seg = vs.get('segmented', '')
        words = seg.split() if seg else []
        per_folio_words[folio] = words
        all_viterbi_words.extend(words)

    # Also extract from decoded sample
    decoded_words: List[str] = []
    for entry in mode_data.get('decoded_sample', []):
        d = entry.get('decoded', '')
        if d and '?' not in d:
            decoded_words.append(d.lower())

    # Use whichever has more data
    analysis_words = all_viterbi_words if len(all_viterbi_words) > len(decoded_words) else decoded_words

    # Sliding window phrase detection
    phrases = _sliding_window_phrases(analysis_words, ref_word_set, ref_trigrams)

    # Direct template matching
    text = ' '.join(analysis_words)
    template_hits: List[Dict] = []
    for template in _MEDICAL_TEMPLATES:
        if template.lower() in text.lower():
            idx = text.lower().index(template.lower())
            template_hits.append({
                'template': template,
                'position': idx,
                'context': text[max(0, idx - 20):idx + len(template) + 20],
            })

    # Also check LATIN_PHRASE_PATTERNS
    for pattern_name, templates in LATIN_PHRASE_PATTERNS:
        for template in templates:
            if template.lower() in text.lower():
                template_hits.append({
                    'template': template,
                    'pattern': pattern_name,
                    'position': text.lower().index(template.lower()),
                })

    # Per-section phrase counts
    per_section_phrases: Dict[str, int] = {}
    for entry in mode_data.get('decoded_sample', []):
        section = entry.get('section', 'unknown')
        per_section_phrases.setdefault(section, 0)

    for p in phrases:
        # Approximate section from position
        per_section_phrases['detected'] = per_section_phrases.get('detected', 0) + 1

    # Botanical cross-check
    botanical = _botanical_cross_check(per_folio_words, ref_word_set)

    result = Phrases22Result(
        timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
        n_phrases_detected=len(phrases),
        top_phrases=phrases[:30],
        botanical_cross_check=botanical,
        per_section_phrases=per_section_phrases,
        template_hits=template_hits,
        n_template_hits=len(template_hits),
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = rdir / "phrases_22.json"
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t0
    print(f"phrases-22: {len(phrases)} phrases, {len(template_hits)} template hits, "
          f"botanical sel={botanical.get('selectivity', 0):.2f}× "
          f"p={botanical.get('p_value', 1):.3f} ({elapsed:.1f}s)")

    return _convert(asdict(result))
