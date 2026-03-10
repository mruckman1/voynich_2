"""
Phase 33.8: Suffix Grammar Mapping
=====================================
Determine which Latin grammatical class each Voynich suffix correlates with,
using SIGNAL words as ground truth.  Maps EVA suffixes to Latin endings
(nominative -us, accusative -um, etc.) to build a suffix-to-grammar table.

Dependency chain:
    signal_bigrams.json        (Step 29.1 — per-token decoded + classifications)
    signal_isolation.json      (Step 28.4 — genuine signal words)
    compound_sign_test.json    (Step 31.6 — suffix_distribution, suffix_profiles)
    combined_refine.json       (Step 15   — best_assignment)
        → suffix_grammar.json     (this step)
"""

import json
import math
import os
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import build_eva_to_triple_lookup, load_corpus, tokenize_eva_chars


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
# Latin ending / POS tables
# ---------------------------------------------------------------------------

# Noun endings (longest first for greedy matching)
LATIN_NOUN_ENDINGS = {
    '-orum': 'gen_pl_2', '-arum': 'gen_pl_1',
    '-ibus': 'dat_abl_pl',
    '-ium': 'gen_pl_3',
    '-us': 'nom_sg_2', '-um': 'acc_sg_2n', '-am': 'acc_sg_1',
    '-em': 'acc_sg_3', '-os': 'acc_pl_2', '-as': 'acc_pl_1',
    '-es': 'nom_pl_3',
    '-a': 'nom_sg_1', '-i': 'gen_sg_2',
    '-is': 'gen_sg_3', '-ae': 'gen_sg_1',
    '-o': 'dat_abl_sg_2',
}

# Verb endings (longest first)
LATIN_VERB_ENDINGS = {
    '-ntur': '3pl_pass', '-tur': '3sg_pass',
    '-mus': '1pl', '-tis': '2pl',
    '-nt': '3pl',
    '-re': 'inf', '-ri': 'inf_pass',
    '-ns': 'pres_part',
    '-t': '3sg', '-s': '2sg',
}

# Adjective endings (same forms as noun but contextually different)
LATIN_ADJ_ENDINGS = {'-us': 'masc', '-a': 'fem', '-um': 'neut', '-is': 'gen_dat'}

# Known uninflected particles
_PARTICLES = {
    'in', 'de', 'ad', 'cum', 'per', 'pro', 'sub', 'ex', 'ab',
    'et', 'vel', 'aut', 'sed', 'si', 'ne', 'ut', 'non', 'iam',
    'sic', 'ita', 'tunc', 'ergo', 'ibi', 'ubi', 'nunc',
}

# Expected case frequency rank in Latin medical text
_EXPECTED_CASE_RANK = [
    'nom_sg', 'acc_sg', 'abl_sg', 'gen_sg', 'dat_sg',
    'nom_pl', 'acc_pl', 'gen_pl', 'dat_abl_pl',
]


# ---------------------------------------------------------------------------
# Classification helpers
# ---------------------------------------------------------------------------

def _classify_latin_ending(decoded_word: str) -> Tuple[str, str]:
    """Classify a decoded Latin word by its ending.

    Returns (POS, ending) where POS is NOUN, VERB, PARTICLE, or UNCLEAR.
    Longest-match-first to avoid partial matches.
    """
    w = decoded_word.lower()

    # Particles are uninflected — check first
    if w in _PARTICLES:
        return 'PARTICLE', ''

    # Verb endings (check before nouns because -nt, -tur are unambiguous)
    for ending in sorted(LATIN_VERB_ENDINGS.keys(), key=lambda x: -len(x)):
        bare = ending.lstrip('-')
        if w.endswith(bare) and len(w) > len(bare):
            return 'VERB', ending

    # Noun endings
    for ending in sorted(LATIN_NOUN_ENDINGS.keys(), key=lambda x: -len(x)):
        bare = ending.lstrip('-')
        if w.endswith(bare) and len(w) > len(bare):
            return 'NOUN', ending

    return 'UNCLEAR', ''


def _detect_suffix(token_eva_chars: List[str], known_suffixes: Set[str]) -> Tuple[Optional[str], List[str]]:
    """Check if the last EVA character(s) of a token match a known suffix.

    Returns (suffix, root_chars).  If no suffix found, returns (None, original_chars).
    """
    if not token_eva_chars:
        return None, token_eva_chars
    last_char = token_eva_chars[-1]
    if last_char in known_suffixes:
        return last_char, token_eva_chars[:-1]
    return None, token_eva_chars


def _entropy(counter: Counter) -> float:
    """Shannon entropy of a count distribution (log-base-2)."""
    total = sum(counter.values())
    if total == 0:
        return 0.0
    h = 0.0
    for count in counter.values():
        if count > 0:
            p = count / total
            h -= p * math.log2(p)
    return h


def _concentration(counter: Counter) -> float:
    """Confidence = 1 - (entropy / max_entropy).  1.0 = perfectly concentrated."""
    n_classes = len(counter)
    if n_classes <= 1:
        return 1.0
    max_ent = math.log2(n_classes)
    if max_ent == 0:
        return 1.0
    return max(0.0, 1.0 - _entropy(counter) / max_ent)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SuffixGrammar:
    suffix: str
    n_tokens: int
    n_signal_tokens: int
    n_anti_signal_tokens: int
    signal_fraction: float
    dominant_pos: str          # NOUN, VERB, PARTICLE, UNCLEAR
    dominant_ending: str       # most common Latin ending (e.g. '-us')
    secondary_ending: str      # second most common
    ending_distribution: Dict[str, int]
    pos_distribution: Dict[str, int]
    confidence: float          # concentration of the POS distribution


@dataclass
class SuffixGrammarResult:
    n_suffix_types: int
    suffix_grammars: List[Dict]
    # Paradigm analysis
    paradigm_table: Dict[str, str]   # suffix -> case/form label
    paradigm_coherence: float
    n_noun_suffixes: int
    n_verb_suffixes: int
    n_unclear_suffixes: int
    # Signal analysis
    signal_suffix_enrichment: Dict[str, float]  # suffix -> signal_rate / overall
    # Verdict
    verdict: str   # PARADIGM_FOUND, WEAK_PARADIGM, NO_PARADIGM
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def _build_suffix_token_map(
    token_evas: List[str],
    token_decoded: List[str],
    token_classifications: List[str],
    token_dict_hits: List[bool],
    known_suffixes: Set[str],
) -> Dict[str, List[Dict]]:
    """Map each known suffix to its token occurrences.

    Returns { suffix: [ {decoded, classification, dict_hit}, ... ] }
    """
    suffix_map: Dict[str, List[Dict]] = defaultdict(list)
    n = len(token_evas)

    for i in range(n):
        eva_chars = tokenize_eva_chars(token_evas[i])
        suffix, root_chars = _detect_suffix(eva_chars, known_suffixes)
        if suffix is None:
            continue
        suffix_map[suffix].append({
            'decoded': token_decoded[i],
            'classification': token_classifications[i],
            'dict_hit': token_dict_hits[i],
        })

    return dict(suffix_map)


def _analyze_suffix(
    suffix: str,
    entries: List[Dict],
    overall_signal_rate: float,
) -> SuffixGrammar:
    """Compute grammar profile for a single suffix."""
    n = len(entries)

    # Signal / anti-signal counts
    n_signal = sum(1 for e in entries if e['classification'] == 'SIGNAL')
    n_anti = sum(1 for e in entries if e['classification'] == 'ANTI_SIGNAL')
    signal_frac = n_signal / n if n > 0 else 0.0

    # Only classify dict-hit tokens
    pos_counter: Counter = Counter()
    ending_counter: Counter = Counter()

    for e in entries:
        if not e['dict_hit']:
            continue
        pos, ending = _classify_latin_ending(e['decoded'])
        pos_counter[pos] += 1
        if ending:
            ending_counter[ending] += 1

    # Dominant POS
    if pos_counter:
        dominant_pos = pos_counter.most_common(1)[0][0]
    else:
        dominant_pos = 'UNCLEAR'

    # Top two endings
    top_endings = ending_counter.most_common(2)
    dominant_ending = top_endings[0][0] if len(top_endings) >= 1 else ''
    secondary_ending = top_endings[1][0] if len(top_endings) >= 2 else ''

    # Confidence (concentration of POS distribution)
    confidence = _concentration(pos_counter)

    return SuffixGrammar(
        suffix=suffix,
        n_tokens=n,
        n_signal_tokens=n_signal,
        n_anti_signal_tokens=n_anti,
        signal_fraction=round(signal_frac, 4),
        dominant_pos=dominant_pos,
        dominant_ending=dominant_ending,
        secondary_ending=secondary_ending,
        ending_distribution=dict(ending_counter.most_common()),
        pos_distribution=dict(pos_counter.most_common()),
        confidence=round(confidence, 4),
    )


def _build_paradigm_table(
    grammars: List[SuffixGrammar],
) -> Tuple[Dict[str, str], float]:
    """Build suffix -> grammatical case mapping and score paradigm coherence.

    Steps:
    1. Select noun-class suffixes.
    2. Rank by frequency within noun class.
    3. Compare rank order to expected Latin medical-text case frequency.
    4. Score coherence as Spearman-like rank correlation.
    """
    # Collect noun-class suffixes with their dominant ending
    noun_suffixes = [
        (g.suffix, g.dominant_ending, g.n_tokens)
        for g in grammars
        if g.dominant_pos == 'NOUN' and g.dominant_ending
    ]

    # Build paradigm table
    paradigm: Dict[str, str] = {}
    for suffix, ending, _ in noun_suffixes:
        # Map ending to a case category
        case_label = LATIN_NOUN_ENDINGS.get(ending, ending)
        paradigm[suffix] = case_label

    # Also add verb-class suffixes
    for g in grammars:
        if g.dominant_pos == 'VERB' and g.dominant_ending:
            form_label = LATIN_VERB_ENDINGS.get(g.dominant_ending, g.dominant_ending)
            paradigm[g.suffix] = form_label

    if not noun_suffixes:
        return paradigm, 0.0

    # Score paradigm coherence: do the most frequent noun-class suffixes
    # correspond to the most frequent Latin cases?
    # Sort noun suffixes by descending token count
    noun_suffixes_sorted = sorted(noun_suffixes, key=lambda x: -x[2])

    # Map each ending to a broad case group for ranking
    _case_group = {
        'nom_sg_1': 'nom_sg', 'nom_sg_2': 'nom_sg',
        'acc_sg_2n': 'acc_sg', 'acc_sg_1': 'acc_sg', 'acc_sg_3': 'acc_sg',
        'dat_abl_sg_2': 'abl_sg',
        'gen_sg_1': 'gen_sg', 'gen_sg_2': 'gen_sg', 'gen_sg_3': 'gen_sg',
        'nom_pl_3': 'nom_pl', 'acc_pl_1': 'acc_pl', 'acc_pl_2': 'acc_pl',
        'gen_pl_1': 'gen_pl', 'gen_pl_2': 'gen_pl', 'gen_pl_3': 'gen_pl',
        'dat_abl_pl': 'dat_abl_pl',
    }

    # Actual rank (by frequency in Voynich)
    actual_order = []
    for suffix, ending, _ in noun_suffixes_sorted:
        case_label = LATIN_NOUN_ENDINGS.get(ending, ending)
        group = _case_group.get(case_label, case_label)
        if group not in actual_order:
            actual_order.append(group)

    # Compare to expected rank
    expected_lookup = {case: rank for rank, case in enumerate(_EXPECTED_CASE_RANK)}
    n_correct_order = 0
    n_pairs = 0
    for i in range(len(actual_order)):
        for j in range(i + 1, len(actual_order)):
            a_rank = expected_lookup.get(actual_order[i], 99)
            b_rank = expected_lookup.get(actual_order[j], 99)
            if a_rank < b_rank:
                n_correct_order += 1
            n_pairs += 1

    coherence = n_correct_order / n_pairs if n_pairs > 0 else 0.0
    return paradigm, round(coherence, 4)


def _signal_enrichment(
    grammars: List[SuffixGrammar],
    overall_signal_rate: float,
) -> Dict[str, float]:
    """Compute signal-rate enrichment: suffix signal_rate / overall rate."""
    enrichment: Dict[str, float] = {}
    for g in grammars:
        if overall_signal_rate > 0 and g.n_tokens >= 10:
            enrichment[g.suffix] = round(g.signal_fraction / overall_signal_rate, 3)
    return enrichment


def _signal_suffix_ground_truth(
    signal_words: List[str],
    token_evas: List[str],
    token_decoded: List[str],
    token_classifications: List[str],
    known_suffixes: Set[str],
) -> Dict[str, List[str]]:
    """For each genuine signal word, find which EVA suffixes appear on its tokens.

    Returns { signal_word: [suffix1, suffix2, ...] }
    """
    sw_set = set(signal_words)
    mapping: Dict[str, List[str]] = defaultdict(list)
    n = len(token_evas)

    for i in range(n):
        decoded = token_decoded[i]
        if decoded not in sw_set:
            continue
        if token_classifications[i] != 'SIGNAL':
            continue
        eva_chars = tokenize_eva_chars(token_evas[i])
        suffix, _ = _detect_suffix(eva_chars, known_suffixes)
        if suffix is not None:
            mapping[decoded].append(suffix)

    # Deduplicate and convert to dict
    return {w: sorted(set(slist)) for w, slist in mapping.items()}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_suffix_grammar() -> None:
    """Step 33.8: Suffix Grammar Mapping."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE 33.8: Suffix Grammar Mapping")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load inputs ──────────────────────────────────────────────────
    print("\n  1. Loading inputs...")

    with open(os.path.join(rd, 'signal_bigrams.json')) as f:
        bigram_data = json.load(f)
    token_evas = bigram_data['token_evas']
    token_decoded = bigram_data['token_decoded']
    token_classifications = bigram_data['token_classifications']
    token_dict_hits = bigram_data['token_dict_hits']
    n_tokens = len(token_evas)

    with open(os.path.join(rd, 'signal_isolation.json')) as f:
        signal_data = json.load(f)
    signal_words = [
        w['word'] for w in signal_data.get('word_signals', [])
        if w.get('is_genuine_signal', False)
    ]

    with open(os.path.join(rd, 'compound_sign_test.json')) as f:
        compound_data = json.load(f)
    suffix_dist = compound_data.get('decomp_stats', {}).get('suffix_distribution', {})
    known_suffixes = set(suffix_dist.keys())

    overall_signal_rate = sum(
        1 for c in token_classifications if c == 'SIGNAL'
    ) / max(n_tokens, 1)

    print(f"     {n_tokens} tokens, {len(signal_words)} signal words, "
          f"{len(known_suffixes)} known suffixes")
    print(f"     Overall SIGNAL rate: {overall_signal_rate:.4f}")

    # ── 2. Signal-word suffix ground truth ──────────────────────────────
    print("\n  2. Signal-word suffix ground truth...")

    sw_suffix_map = _signal_suffix_ground_truth(
        signal_words, token_evas, token_decoded,
        token_classifications, known_suffixes,
    )
    for sw, slist in sorted(sw_suffix_map.items()):
        print(f"     {sw:10s} → suffixes: {', '.join(slist) if slist else '(none)'}")
    n_sw_with_suffix = sum(1 for v in sw_suffix_map.values() if v)
    print(f"     {n_sw_with_suffix}/{len(signal_words)} signal words appear "
          f"with a suffix")

    # ── 3. Build suffix → token map ─────────────────────────────────────
    print("\n  3. Building suffix → token map...")

    suffix_token_map = _build_suffix_token_map(
        token_evas, token_decoded, token_classifications,
        token_dict_hits, known_suffixes,
    )
    for sfx in sorted(suffix_token_map.keys(), key=lambda s: -len(suffix_token_map[s])):
        n = len(suffix_token_map[sfx])
        n_hit = sum(1 for e in suffix_token_map[sfx] if e['dict_hit'])
        print(f"     {sfx:6s}: {n:6d} tokens, {n_hit:5d} dict_hits "
              f"({n_hit / n:.1%})")

    # ── 4. Classify each suffix ─────────────────────────────────────────
    print("\n  4. Classifying suffixes by Latin POS / ending...")

    grammars: List[SuffixGrammar] = []
    for sfx, entries in sorted(suffix_token_map.items(),
                               key=lambda x: -len(x[1])):
        g = _analyze_suffix(sfx, entries, overall_signal_rate)
        grammars.append(g)
        pos_str = ', '.join(f'{k}={v}' for k, v in g.pos_distribution.items())
        end_str = g.dominant_ending if g.dominant_ending else '(none)'
        print(f"     {sfx:6s}: POS={g.dominant_pos:8s}  ending={end_str:8s}  "
              f"conf={g.confidence:.3f}  signal={g.signal_fraction:.3f}  "
              f"[{pos_str}]")

    # ── 5. Paradigm table construction ──────────────────────────────────
    print("\n  5. Building paradigm table...")

    paradigm_table, paradigm_coherence = _build_paradigm_table(grammars)
    n_noun = sum(1 for g in grammars if g.dominant_pos == 'NOUN')
    n_verb = sum(1 for g in grammars if g.dominant_pos == 'VERB')
    n_unclear = sum(1 for g in grammars
                    if g.dominant_pos not in ('NOUN', 'VERB', 'PARTICLE'))

    print(f"     Noun suffixes: {n_noun}")
    print(f"     Verb suffixes: {n_verb}")
    print(f"     Unclear suffixes: {n_unclear}")
    print(f"     Paradigm coherence: {paradigm_coherence:.4f}")

    if paradigm_table:
        print("     Paradigm mapping:")
        for sfx, case in sorted(paradigm_table.items()):
            count = suffix_dist.get(sfx, 0)
            print(f"       {sfx:6s} → {case:20s}  (n={count})")

    # ── 6. Signal enrichment ────────────────────────────────────────────
    print("\n  6. Signal enrichment by suffix...")

    enrichment = _signal_enrichment(grammars, overall_signal_rate)
    for sfx, ratio in sorted(enrichment.items(), key=lambda x: -x[1]):
        print(f"     {sfx:6s}: {ratio:.3f}x")

    # ── 7. Verdict ──────────────────────────────────────────────────────

    # Criteria:
    #   PARADIGM_FOUND: coherence >= 0.5 AND n_noun >= 3 AND n_verb >= 1
    #   WEAK_PARADIGM: coherence >= 0.3 AND n_noun >= 2
    #   NO_PARADIGM: otherwise
    if paradigm_coherence >= 0.5 and n_noun >= 3 and n_verb >= 1:
        verdict = 'PARADIGM_FOUND'
    elif paradigm_coherence >= 0.3 and n_noun >= 2:
        verdict = 'WEAK_PARADIGM'
    else:
        verdict = 'NO_PARADIGM'

    print(f"\n  Verdict: {verdict}")
    print(f"     Coherence={paradigm_coherence:.4f}, "
          f"noun_suf={n_noun}, verb_suf={n_verb}, unclear={n_unclear}")

    # ── 8. Save ─────────────────────────────────────────────────────────
    runtime = round(time.time() - t0, 2)

    result = SuffixGrammarResult(
        n_suffix_types=len(grammars),
        suffix_grammars=[_convert(asdict(g)) for g in grammars],
        paradigm_table=paradigm_table,
        paradigm_coherence=paradigm_coherence,
        n_noun_suffixes=n_noun,
        n_verb_suffixes=n_verb,
        n_unclear_suffixes=n_unclear,
        signal_suffix_enrichment=enrichment,
        verdict=verdict,
        runtime_seconds=runtime,
    )

    out_path = os.path.join(rd, 'suffix_grammar.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2)

    print(f"\n  Saved → {out_path}")
    print(f"  Completed in {runtime:.1f}s")
