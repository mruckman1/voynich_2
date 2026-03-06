"""
Phase 20.4 – Full Corpus Tachygraphic Decode
=============================================
Apply the best tachygraphic table from Step 20.3 to the full corpus, produce
decoded text with per-folio statistics and section analysis.

Dependency chain:
    tachy_grid_solve.json + modifier_integrate.json + corpus
        → tachy_decode.json
"""

import json
import math
import os
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    decode_token_modifier_aware,
    load_corpus,
    tokenize_eva_chars,
)
from voynich.core.reference import (
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


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FolioDecoded:
    folio: str
    section: str
    n_tokens: int
    n_dict_hits: int
    dict_hit_rate: float
    decoded_sample: List[List[str]]   # [(voynich, decoded), ...]


@dataclass
class TachyDecodeResult:
    n_tokens_total: int
    n_tokens_decoded: int
    n_dict_hits: int
    dict_hit_rate: float
    expanded_dict_hit_rate: float
    mean_syllables_per_token: float
    per_section: Dict[str, Dict]
    per_folio_summary: List[Dict]
    decoded_sample: List[List[str]]
    top_decoded_words: List[List]     # [(word, count), ...]
    word_segmentation_sample: List[List[str]]  # [(decoded, segmented), ...]
    phase16_dict_hit: float
    improvement: float
    gate_passed: bool
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _load_json(rd: str, fname: str) -> Dict:
    path = os.path.join(rd, fname)
    if not os.path.exists(path):
        print(f"    [WARN] {fname} not found")
        return {}
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------

def _decode_token_tachy(
    token: str,
    char_assignment: Dict[str, str],
    modifier_chars: Set[str],
) -> str:
    """Decode EVA token using char-level tachygraphic table.

    Modifier chars are silently stripped.  Non-modifier chars are looked up
    in the assignment.  Unknown chars produce '?'.
    """
    chars = tokenize_eva_chars(token)
    parts = []
    for ch in chars:
        if ch in modifier_chars:
            continue  # strip modifier
        if ch in char_assignment:
            parts.append(char_assignment[ch])
        else:
            parts.append('?')
    return ''.join(parts)


def _decode_token_r3_combined(
    token: str,
    char_assignment: Dict[str, str],
    modifier_chars: Set[str],
    ref_word_set: set,
) -> str:
    """R3 combined strategy: try alteration, then strip, then original.

    Pick whichever produces a dict hit.
    """
    # Strategy 1: strip modifiers
    stripped = _decode_token_tachy(token, char_assignment, modifier_chars)

    if stripped and stripped in ref_word_set:
        return stripped

    # Strategy 2: keep modifier chars (treat as syllabic)
    chars = tokenize_eva_chars(token)
    all_parts = []
    for ch in chars:
        if ch in char_assignment:
            all_parts.append(char_assignment[ch])
        # Modifier chars won't be in char_assignment, so skip
    with_mods = ''.join(all_parts)
    if with_mods and with_mods in ref_word_set:
        return with_mods

    # Default: stripped version
    return stripped


# ---------------------------------------------------------------------------
# Viterbi word segmentation
# ---------------------------------------------------------------------------

def _build_word_model(ref_word_set: set) -> Dict[str, float]:
    """Build unigram log-prob model for Viterbi segmentation."""
    total = len(ref_word_set)
    if total == 0:
        return {}
    log_total = math.log(total)
    return {w: -log_total for w in ref_word_set}  # uniform prior


def _viterbi_segment(decoded: str, word_model: Dict[str, float],
                     max_word_len: int = 15) -> List[str]:
    """Segment decoded string into Latin words using Viterbi DP."""
    n = len(decoded)
    if n == 0:
        return []

    # cost[i] = best log-prob for decoded[:i]
    cost = [float('inf')] * (n + 1)
    cost[0] = 0.0
    backptr = [0] * (n + 1)
    default_cost = 30.0  # penalty for unknown word

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

    # Trace back
    words = []
    i = n
    while i > 0:
        j = backptr[i]
        words.append(decoded[j:i])
        i = j
    words.reverse()
    return words


# ---------------------------------------------------------------------------
# Section classification
# ---------------------------------------------------------------------------

# Approximate folio→section mapping
_SECTION_RANGES = {
    'herbal_a': (1, 57),
    'astronomical': (67, 73),
    'biological': (75, 84),
    'cosmological': (85, 86),
    'pharmaceutical': (87, 102),
    'herbal_b': (100, 116),
    'recipes': (103, 116),
}


def _classify_folio(folio_str: str) -> str:
    """Classify folio into manuscript section by number."""
    # Extract numeric part
    digits = ''.join(c for c in folio_str if c.isdigit())
    if not digits:
        return 'unknown'
    num = int(digits)
    for section, (lo, hi) in _SECTION_RANGES.items():
        if lo <= num <= hi:
            return section
    return 'unknown'


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_tachy_decode() -> None:
    """Step 20.4: Decode full corpus using tachygraphic table."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE 20.4: Full Corpus Tachygraphic Decode")
    print("=" * 70)

    rd = _results_dir()

    # ─── 1. Load dependencies ───
    print("\n  1. Loading dependencies …")
    grid_data = _load_json(rd, 'tachy_grid_solve.json')
    modifier_data = _load_json(rd, 'modifier_integrate.json')
    modifier_chars = set(modifier_data.get('modifier_chars', []))

    char_assignment = grid_data.get('best_assignment', {})
    print(f"      Table entries: {len(char_assignment)}")
    print(f"      Modifier chars: {len(modifier_chars)}")

    # Corpus
    corpus = load_corpus(verbose=False)
    tokens = corpus.get_tokens()
    folios = corpus.get_folios() if hasattr(corpus, 'get_folios') else []

    # Reference words
    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    base_words = set(
        w.lower() for w in ref_corpus.get_combined_tokens('latin')
        if len(w) >= 2
    )
    expanded_words, _ = build_expanded_word_set(base_words)
    ref_word_set = base_words | expanded_words

    print(f"      Corpus tokens: {len(tokens)}")
    print(f"      Reference words: {len(ref_word_set)}")

    # ─── 2. Decode all tokens ───
    print("\n  2. Decoding corpus …")
    decoded_tokens: List[str] = []
    dict_hits_base = 0
    dict_hits_expanded = 0
    total_syllables = 0
    total_decoded = 0

    for token in tokens:
        decoded = _decode_token_r3_combined(
            token, char_assignment, modifier_chars, ref_word_set,
        )
        decoded_tokens.append(decoded)

        if decoded and decoded != '?':
            total_decoded += 1
            # Count syllables (each char maps to ~1 syllable)
            chars = tokenize_eva_chars(token)
            n_syl = sum(1 for ch in chars if ch not in modifier_chars
                        and ch in char_assignment)
            total_syllables += n_syl

            if decoded in base_words:
                dict_hits_base += 1
            if decoded in ref_word_set:
                dict_hits_expanded += 1

    n_total = len(tokens)
    dict_hit_rate = dict_hits_base / n_total if n_total else 0.0
    expanded_hit_rate = dict_hits_expanded / n_total if n_total else 0.0
    mean_syl = total_syllables / total_decoded if total_decoded else 0.0

    print(f"      Decoded: {total_decoded}/{n_total}")
    print(f"      Base dict_hit: {dict_hit_rate:.1%}")
    print(f"      Expanded dict_hit: {expanded_hit_rate:.1%}")
    print(f"      Mean syllables/token: {mean_syl:.2f}")

    # ─── 3. Per-folio analysis ───
    print("\n  3. Per-folio analysis …")
    # Build folio→token mapping from corpus
    folio_tokens: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

    if hasattr(corpus, 'lines'):
        for line in corpus.lines:
            folio = getattr(line, 'folio', 'unknown')
            for i, token in enumerate(getattr(line, 'tokens', [])):
                idx = sum(len(getattr(l, 'tokens', [])) for l in corpus.lines
                          if l is not line and corpus.lines.index(l) < corpus.lines.index(line)) + i
                if idx < len(decoded_tokens):
                    folio_tokens[folio].append((token, decoded_tokens[idx]))
    else:
        # Fallback: assign all tokens to a single folio
        for i, (token, decoded) in enumerate(zip(tokens, decoded_tokens)):
            folio_tokens['all'].append((token, decoded))

    per_folio_list: List[Dict] = []
    section_stats: Dict[str, Dict] = defaultdict(lambda: {
        'n_tokens': 0, 'n_hits': 0, 'decoded_words': Counter()
    })

    for folio, pairs in sorted(folio_tokens.items()):
        section = _classify_folio(folio)
        n_tok = len(pairs)
        n_hits = sum(1 for _, d in pairs if d in ref_word_set)
        hit_rate = n_hits / n_tok if n_tok else 0.0

        sample = [(v, d) for v, d in pairs[:10]]
        per_folio_list.append({
            'folio': folio,
            'section': section,
            'n_tokens': n_tok,
            'n_dict_hits': n_hits,
            'dict_hit_rate': hit_rate,
            'sample': sample,
        })

        section_stats[section]['n_tokens'] += n_tok
        section_stats[section]['n_hits'] += n_hits
        for _, d in pairs:
            if d in ref_word_set:
                section_stats[section]['decoded_words'][d] += 1

    # Finalise section stats
    per_section: Dict[str, Dict] = {}
    for section, stats in section_stats.items():
        n = stats['n_tokens']
        per_section[section] = {
            'n_tokens': n,
            'n_hits': stats['n_hits'],
            'dict_hit_rate': stats['n_hits'] / n if n else 0.0,
            'top_words': stats['decoded_words'].most_common(20),
        }
        print(f"      {section:20s}: {stats['n_hits']}/{n} "
              f"({stats['n_hits'] / n:.1%})" if n else f"      {section}: 0 tokens")

    # ─── 4. Top decoded words ───
    print("\n  4. Top decoded words …")
    word_counts: Counter = Counter()
    for decoded in decoded_tokens:
        if decoded and decoded in ref_word_set:
            word_counts[decoded] += 1
    top_words = word_counts.most_common(30)
    for word, count in top_words[:15]:
        print(f"      {word:15s}  {count}")

    # ─── 5. Viterbi segmentation sample ───
    print("\n  5. Viterbi word segmentation (sample) …")
    word_model = _build_word_model(ref_word_set)
    seg_sample: List[List[str]] = []
    for i, decoded in enumerate(decoded_tokens[:100]):
        if decoded and len(decoded) >= 4 and decoded not in ref_word_set:
            words = _viterbi_segment(decoded, word_model)
            if words and any(w in ref_word_set for w in words):
                seg_sample.append([decoded, ' '.join(words)])
                if len(seg_sample) <= 10:
                    print(f"      {decoded} → {' '.join(words)}")
        if len(seg_sample) >= 30:
            break

    # ─── 6. Decoded sample ───
    decoded_sample: List[List[str]] = []
    for token, decoded in zip(tokens[:50], decoded_tokens[:50]):
        decoded_sample.append([token, decoded])

    # ─── 7. Gate ───
    phase16_dict_hit = 0.516
    improvement = expanded_hit_rate - phase16_dict_hit
    gate_passed = expanded_hit_rate > 0.10
    if gate_passed:
        verdict = (f"PASS: expanded dict_hit={expanded_hit_rate:.1%} "
                   f"(Phase 16={phase16_dict_hit:.1%}, "
                   f"delta={improvement:+.1%}). "
                   f"Mean {mean_syl:.1f} syl/token.")
    else:
        verdict = (f"FAIL: expanded dict_hit={expanded_hit_rate:.1%} "
                   f"(below minimum threshold).")

    print(f"\n  6. Gate: {verdict}")

    # ─── 8. Save ───
    result = TachyDecodeResult(
        n_tokens_total=n_total,
        n_tokens_decoded=total_decoded,
        n_dict_hits=dict_hits_expanded,
        dict_hit_rate=dict_hit_rate,
        expanded_dict_hit_rate=expanded_hit_rate,
        mean_syllables_per_token=mean_syl,
        per_section=per_section,
        per_folio_summary=per_folio_list[:50],  # first 50 folios
        decoded_sample=decoded_sample,
        top_decoded_words=top_words,
        word_segmentation_sample=seg_sample,
        phase16_dict_hit=phase16_dict_hit,
        improvement=improvement,
        gate_passed=gate_passed,
        verdict=verdict,
        runtime_seconds=time.time() - t0,
    )

    out_path = os.path.join(rd, 'tachy_decode.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)

    print(f"\n  → {out_path}")
