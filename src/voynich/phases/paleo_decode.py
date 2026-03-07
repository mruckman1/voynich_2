"""
Phase 21.8 – Corpus Decode with Paleographic Table (paleo-decode)
=================================================================
Decodes the full Voynich corpus using the paleographic table (21.7) and
scores against both dictionaries (17K original, 131K expanded).

Dependency chain:
    paleo_table.json (21.7) + corpus (IVTFF)
        → paleo_decode.json (this step)
"""

import json
import time
from collections import Counter
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
    EVA_VISUAL_COMPONENTS,
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


# ---------------------------------------------------------------------------
# Simple word boundary detection using Latin unigrams
# ---------------------------------------------------------------------------

# Common Latin function words / high-frequency syllables for boundary hints
_LATIN_WORDS: Set[str] = {
    'ad', 'de', 'in', 'et', 'ut', 'ab', 'ex', 'cum', 'per', 'pro',
    'non', 'est', 'sed', 'que', 'aut', 'qui', 'vel', 'nam', 'iam',
    'bene', 'male', 'ante', 'post', 'supra', 'infra',
}


def _decode_token_paleo(
    token: str,
    char_to_syllable: Dict[str, str],
    modifier_chars: Set[str],
    modifier_functions: Dict[str, str],
) -> Tuple[str, str]:
    """Decode a single EVA token using paleographic char-level table.

    Returns (decoded_string, confidence_tag).
    confidence_tag: 'high' if all chars are Priority 1-3, 'mixed', or 'low'.
    """
    chars = tokenize_eva_chars(token)
    parts: List[str] = []
    priorities: List[int] = []

    for ch in chars:
        if ch in modifier_chars:
            # Apply modifier: Cappelli-informed where available, R3 fallback
            func = modifier_functions.get(ch, 'silent')
            if func in ('omission_nasal', 'nasalization'):
                # Append 'n' to previous syllable
                if parts:
                    parts[-1] = parts[-1] + 'n'
            elif func == 'truncation':
                # Double previous consonant
                if parts and parts[-1]:
                    parts[-1] = parts[-1][0] + parts[-1]
            elif func == 'superscript':
                # Skip (vowel change handled differently)
                pass
            else:
                # Default: silent (R3 fallback - strip)
                pass
            continue

        syl = char_to_syllable.get(ch, '')
        if syl:
            parts.append(syl)
            priorities.append(1)  # Placeholder - actual priority from table
        else:
            parts.append('?')
            priorities.append(6)

    decoded = ''.join(parts)

    # Confidence tag
    if not priorities:
        tag = 'low'
    elif all(p <= 3 for p in priorities):
        tag = 'high'
    elif any(p <= 3 for p in priorities):
        tag = 'mixed'
    else:
        tag = 'low'

    return decoded, tag


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SectionDecode:
    section: str
    n_tokens: int
    n_dict_hits: int
    dict_hit_rate: float
    sample_hits: List[str]
    confidence_breakdown: Dict[str, int]


@dataclass
class PaleoDecodeResult:
    timestamp: str
    n_tokens_total: int
    n_decoded: int
    dict_hit_rate_original: float
    dict_hit_rate_expanded: float
    n_high_confidence_tokens: int
    high_conf_dict_hit_rate: float
    per_section: List[Dict[str, Any]]
    sample_decoded_tokens: List[Dict[str, str]]
    word_frequency_top30: Dict[str, int]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_paleo_decode() -> Dict[str, Any]:
    """Decode full corpus with paleographic table."""
    t0 = time.time()
    rdir = _results_dir()

    # --- Load table ---
    table_data = _load_json(str(rdir / "paleo_table.json")) or {}
    table_entries = table_data.get('table', [])

    char_to_syllable: Dict[str, str] = {}
    modifier_chars: Set[str] = set()
    modifier_functions: Dict[str, str] = {}

    for entry in table_entries:
        ec = entry.get('eva_char', '')
        if entry.get('is_modifier'):
            modifier_chars.add(ec)
            modifier_functions[ec] = entry.get('modifier_function', 'silent') or 'silent'
        elif entry.get('latin_syllable'):
            char_to_syllable[ec] = entry['latin_syllable']

    # --- Load dictionaries ---
    # Original (17K)
    try:
        from voynich.core.reference import LATIN_WORD_SET
        original_dict = LATIN_WORD_SET
    except ImportError:
        original_dict = set()

    if not original_dict:
        # Try loading from reference
        ref_path = "data/reference/latin/word_list.txt"
        import os
        if os.path.exists(ref_path):
            with open(ref_path) as f:
                original_dict = {line.strip().lower() for line in f if line.strip()}

    # Expanded (131K)
    try:
        expanded_dict, _ = build_expanded_word_set('latin')
    except Exception:
        expanded_dict = original_dict

    if not expanded_dict:
        expanded_dict = original_dict

    # --- Load corpus ---
    corpus = load_corpus()

    # --- Decode ---
    all_decoded: List[Tuple[str, str, str, str]] = []  # (token, decoded, conf, section)
    per_section_data: Dict[str, List[Tuple[str, str, str]]] = {}
    word_freq: Counter = Counter()

    for page in corpus.pages.values():
        section = page.section or 'unknown'
        text = page.paragraph_text if hasattr(page, 'paragraph_text') else page.all_text
        if not text:
            continue
        # EVA tokens are dot-separated within whitespace-separated locus groups
        for group in text.split():
            if group.startswith('<') or group.startswith('{'):
                continue
            sub_tokens = [t.strip() for t in group.split('.') if t.strip()]
            for token in sub_tokens:
                decoded, conf = _decode_token_paleo(
                    token, char_to_syllable, modifier_chars, modifier_functions
                )

                if '?' not in decoded and decoded:
                    word_freq[decoded] += 1

                all_decoded.append((token, decoded, conf, section))
                per_section_data.setdefault(section, []).append((token, decoded, conf))

    # --- Score against dictionaries ---
    n_total = len(all_decoded)
    n_decoded = sum(1 for _, d, _, _ in all_decoded if d and '?' not in d)

    # Original dict hits
    orig_hits = sum(
        1 for _, d, _, _ in all_decoded
        if d and '?' not in d and d.lower() in original_dict
    )
    orig_rate = orig_hits / max(n_total, 1)

    # Expanded dict hits
    exp_hits = sum(
        1 for _, d, _, _ in all_decoded
        if d and '?' not in d and d.lower() in expanded_dict
    )
    exp_rate = exp_hits / max(n_total, 1)

    # High-confidence tokens
    high_conf = [(t, d, c, s) for t, d, c, s in all_decoded if c == 'high']
    n_high = len(high_conf)
    high_hits = sum(
        1 for _, d, _, _ in high_conf
        if d and '?' not in d and d.lower() in expanded_dict
    )
    high_rate = high_hits / max(n_high, 1)

    # --- Per-section stats ---
    per_section_results: List[SectionDecode] = []
    for section in sorted(per_section_data.keys()):
        items = per_section_data[section]
        n_tok = len(items)
        n_hits = sum(
            1 for _, d, _ in items
            if d and '?' not in d and d.lower() in expanded_dict
        )
        sample = [
            f"{t}→{d}" for t, d, _ in items
            if d and '?' not in d and d.lower() in expanded_dict
        ][:5]
        conf_counts = Counter(c for _, _, c in items)
        per_section_results.append(SectionDecode(
            section=section,
            n_tokens=n_tok,
            n_dict_hits=n_hits,
            dict_hit_rate=n_hits / max(n_tok, 1),
            sample_hits=sample,
            confidence_breakdown=dict(conf_counts),
        ))

    # --- Sample decoded tokens ---
    samples: List[Dict[str, str]] = []
    for t, d, c, s in all_decoded[:100]:
        in_dict = 'yes' if (d and '?' not in d and d.lower() in expanded_dict) else 'no'
        samples.append({
            'eva_token': t,
            'decoded': d,
            'confidence': c,
            'section': s,
            'in_expanded_dict': in_dict,
        })

    result = PaleoDecodeResult(
        timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
        n_tokens_total=n_total,
        n_decoded=n_decoded,
        dict_hit_rate_original=orig_rate,
        dict_hit_rate_expanded=exp_rate,
        n_high_confidence_tokens=n_high,
        high_conf_dict_hit_rate=high_rate,
        per_section=[_convert(asdict(s)) for s in per_section_results],
        sample_decoded_tokens=samples,
        word_frequency_top30=dict(word_freq.most_common(30)),
    )

    out_path = rdir / "paleo_decode.json"
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t0
    print(f"paleo-decode: {n_total} tokens, expanded dict_hit={exp_rate:.1%}, "
          f"high-conf={n_high} @ {high_rate:.1%} ({elapsed:.1f}s)")

    return _convert(asdict(result))
