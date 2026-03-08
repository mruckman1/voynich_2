"""
Phase 22.4 – Corpus Decode with Phase 22 Table (decode-22)
===========================================================
Applies the merged table (22.3) to the full Voynich corpus.
Decodes tokens, runs Viterbi word segmentation, scores against
both dictionaries (17K original, 131K expanded).

Runs BOTH Mode A (strict CV) and Mode B (CVC) tables.

Dependency chain:
    merged_table.json (22.3) + corpus + dictionaries
        → corpus_decode_22.json (this step)
"""

import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    load_corpus,
    tokenize_eva_chars,
)
from voynich.core.reference import (
    build_expanded_word_set,
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
    import os
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _build_word_model(ref_word_set: set) -> Dict[str, float]:
    """Build unigram log-prob model for Viterbi segmentation."""
    total = len(ref_word_set)
    if total == 0:
        return {}
    log_total = math.log(total)
    return {w: -log_total for w in ref_word_set}


def _viterbi_segment(decoded: str, word_model: Dict[str, float],
                     max_word_len: int = 15) -> List[str]:
    """Segment decoded string into Latin words using Viterbi DP."""
    n = len(decoded)
    if n == 0:
        return []

    cost = [float('inf')] * (n + 1)
    cost[0] = 0.0
    backptr = [0] * (n + 1)
    default_cost = 30.0

    for i in range(1, n + 1):
        for j in range(max(0, i - max_word_len), i):
            candidate = decoded[j:i]
            if candidate in word_model:
                c = cost[j] + (-word_model[candidate])
            elif len(candidate) <= 2:
                c = cost[j] + default_cost
            else:
                c = cost[j] + default_cost + len(candidate)
            if c < cost[i]:
                cost[i] = c
                backptr[i] = j

    words: List[str] = []
    i = n
    while i > 0:
        j = backptr[i]
        words.append(decoded[j:i])
        i = j
    words.reverse()
    return words


def _classify_folio(folio_str: str) -> str:
    """Classify folio into manuscript section."""
    import re
    m = re.match(r'f(\d+)', folio_str.lower())
    if not m:
        return 'unknown'
    num = int(m.group(1))
    if 1 <= num <= 56:
        return 'herbal_a'
    elif 57 <= num <= 66:
        return 'pharmaceutical'
    elif 67 <= num <= 73:
        return 'astronomical'
    elif 75 <= num <= 84:
        return 'biological'
    elif 85 <= num <= 86:
        return 'cosmological'
    elif 87 <= num <= 102:
        return 'herbal_b'
    elif 103 <= num <= 116:
        return 'stars'
    else:
        return 'unknown'


# ---------------------------------------------------------------------------
# Decode functions
# ---------------------------------------------------------------------------

def _decode_token_22(
    token: str,
    char_to_syl: Dict[str, str],
    modifier_chars: Set[str],
    mode: str = 'a',
) -> Tuple[str, str]:
    """Decode a single EVA token using Phase 22 merged table.

    Uses R3 combined strategy for modifiers: try strip, alter, original.
    Returns (decoded_string, confidence_tag).
    """
    chars = tokenize_eva_chars(token)
    parts: List[str] = []
    n_mapped = 0
    n_unknown = 0

    for ch in chars:
        if ch in modifier_chars:
            continue  # R3 strip strategy
        syl = char_to_syl.get(ch, '')
        if syl:
            parts.append(syl)
            n_mapped += 1
        else:
            parts.append('?')
            n_unknown += 1

    decoded = ''.join(parts)

    if n_unknown == 0 and n_mapped > 0:
        conf = 'high'
    elif n_mapped > n_unknown:
        conf = 'mixed'
    elif n_mapped > 0:
        conf = 'low'
    else:
        conf = 'none'

    return decoded, conf


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class FolioDecoded22:
    folio: str
    section: str
    n_tokens: int
    n_dict_hits: int
    dict_hit_rate: float
    n_viterbi_words: int
    n_viterbi_dict_hits: int
    viterbi_dict_rate: float
    decoded_sample: List[Dict[str, str]]


@dataclass
class Decode22Result:
    timestamp: str
    mode: str   # 'a' or 'b'
    n_tokens_total: int
    n_tokens_decoded: int
    dict_hit_rate_original: float
    dict_hit_rate_expanded: float
    n_viterbi_words: int
    viterbi_dict_rate: float
    n_high_conf: int
    high_conf_dict_rate: float
    per_section: Dict[str, Dict[str, Any]]
    decoded_sample: List[Dict[str, str]]
    word_frequency_top30: Dict[str, int]
    viterbi_sample: List[Dict[str, str]]


@dataclass
class CorpusDecode22Result:
    timestamp: str
    mode_a: Dict[str, Any]
    mode_b: Dict[str, Any]
    mode_a_dict_hit: float
    mode_b_dict_hit: float
    better_mode: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Main decode for one mode
# ---------------------------------------------------------------------------

def _run_one_mode(
    mode: str,
    table_entries: List[Dict],
    corpus,
    original_dict: set,
    expanded_dict: set,
    word_model: Dict[str, float],
) -> Decode22Result:
    """Decode corpus with one mode (A or B)."""

    # Build char→syllable map
    char_to_syl: Dict[str, str] = {}
    modifier_chars: Set[str] = set()

    syl_field = 'syllable_a' if mode == 'a' else 'syllable_b'
    for entry in table_entries:
        ec = entry.get('eva_char', '')
        if entry.get('is_modifier'):
            modifier_chars.add(ec)
            continue
        syl = entry.get(syl_field, '')
        if syl:
            char_to_syl[ec] = syl

    # Decode all tokens
    all_decoded: List[Tuple[str, str, str, str]] = []  # (token, decoded, conf, section)
    per_section: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)
    word_freq: Counter = Counter()
    all_viterbi_words: List[str] = []
    viterbi_samples: List[Dict[str, str]] = []

    for folio_id, page in corpus.pages.items():
        section = page.section or _classify_folio(folio_id)
        text = page.paragraph_text if hasattr(page, 'paragraph_text') else page.all_text
        if not text:
            continue

        folio_decoded_parts: List[str] = []

        for group in text.split():
            if group.startswith('<') or group.startswith('{'):
                continue
            sub_tokens = [t.strip() for t in group.split('.') if t.strip()]
            for token in sub_tokens:
                decoded, conf = _decode_token_22(token, char_to_syl, modifier_chars, mode)
                all_decoded.append((token, decoded, conf, section))
                per_section[section].append((token, decoded, conf))
                if '?' not in decoded and decoded:
                    word_freq[decoded] += 1
                    folio_decoded_parts.append(decoded)

        # Viterbi segmentation on concatenated folio text
        folio_text = ''.join(folio_decoded_parts)
        if folio_text and word_model:
            words = _viterbi_segment(folio_text, word_model)
            all_viterbi_words.extend(words)
            if len(viterbi_samples) < 20:
                viterbi_samples.append({
                    'folio': folio_id,
                    'raw': folio_text[:200],
                    'segmented': ' '.join(words[:30]),
                })

    # Score
    n_total = len(all_decoded)
    n_decoded = sum(1 for _, d, _, _ in all_decoded if d and '?' not in d)

    orig_hits = sum(1 for _, d, _, _ in all_decoded
                    if d and '?' not in d and d.lower() in original_dict)
    orig_rate = orig_hits / max(n_total, 1)

    exp_hits = sum(1 for _, d, _, _ in all_decoded
                   if d and '?' not in d and d.lower() in expanded_dict)
    exp_rate = exp_hits / max(n_total, 1)

    high_conf = [(t, d, c, s) for t, d, c, s in all_decoded if c == 'high']
    n_high = len(high_conf)
    high_hits = sum(1 for _, d, _, _ in high_conf
                    if d and '?' not in d and d.lower() in expanded_dict)
    high_rate = high_hits / max(n_high, 1)

    # Viterbi dict hits
    viterbi_dict_hits = sum(1 for w in all_viterbi_words if w.lower() in expanded_dict)
    viterbi_rate = viterbi_dict_hits / max(len(all_viterbi_words), 1)

    # Per-section summary
    section_summary: Dict[str, Dict[str, Any]] = {}
    for section in sorted(per_section.keys()):
        items = per_section[section]
        n_tok = len(items)
        n_hits = sum(1 for _, d, _ in items
                     if d and '?' not in d and d.lower() in expanded_dict)
        section_summary[section] = {
            'n_tokens': n_tok,
            'n_dict_hits': n_hits,
            'dict_hit_rate': round(n_hits / max(n_tok, 1), 4),
        }

    # Samples
    samples: List[Dict[str, str]] = []
    for t, d, c, s in all_decoded[:100]:
        in_dict = 'yes' if (d and '?' not in d and d.lower() in expanded_dict) else 'no'
        samples.append({
            'eva_token': t,
            'decoded': d,
            'confidence': c,
            'section': s,
            'in_dict': in_dict,
        })

    return Decode22Result(
        timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
        mode=mode,
        n_tokens_total=n_total,
        n_tokens_decoded=n_decoded,
        dict_hit_rate_original=round(orig_rate, 4),
        dict_hit_rate_expanded=round(exp_rate, 4),
        n_viterbi_words=len(all_viterbi_words),
        viterbi_dict_rate=round(viterbi_rate, 4),
        n_high_conf=n_high,
        high_conf_dict_rate=round(high_rate, 4),
        per_section=section_summary,
        decoded_sample=samples,
        word_frequency_top30=dict(word_freq.most_common(30)),
        viterbi_sample=viterbi_samples,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_decode_22() -> Dict[str, Any]:
    """Decode full corpus with Phase 22 merged table (both modes)."""
    t0 = time.time()
    rdir = _results_dir()

    # Load merged table
    merged = _load_json(str(rdir / "merged_table.json")) or {}
    table_a = merged.get('mode_a_table', [])
    table_b = merged.get('mode_b_table', [])

    # Load dictionaries
    try:
        from voynich.core.reference import LATIN_WORD_SET
        original_dict = LATIN_WORD_SET
    except ImportError:
        original_dict = set()

    try:
        expanded_dict, _ = build_expanded_word_set('latin')
    except Exception:
        expanded_dict = original_dict

    if not expanded_dict:
        expanded_dict = original_dict

    # Build word model for Viterbi
    word_model = _build_word_model(expanded_dict)

    # Load corpus
    corpus = load_corpus()

    # Run both modes
    result_a = _run_one_mode('a', table_a, corpus, original_dict, expanded_dict, word_model)
    result_b = _run_one_mode('b', table_b, corpus, original_dict, expanded_dict, word_model)

    better = 'a' if result_a.dict_hit_rate_expanded >= result_b.dict_hit_rate_expanded else 'b'

    result = CorpusDecode22Result(
        timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
        mode_a=_convert(asdict(result_a)),
        mode_b=_convert(asdict(result_b)),
        mode_a_dict_hit=result_a.dict_hit_rate_expanded,
        mode_b_dict_hit=result_b.dict_hit_rate_expanded,
        better_mode=better,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = rdir / "corpus_decode_22.json"
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t0
    print(f"decode-22: A={result_a.dict_hit_rate_expanded:.1%} "
          f"B={result_b.dict_hit_rate_expanded:.1%} "
          f"viterbi_A={result_a.viterbi_dict_rate:.1%} "
          f"better={better} ({elapsed:.1f}s)")

    return _convert(asdict(result))
