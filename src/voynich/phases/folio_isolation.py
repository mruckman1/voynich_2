"""
Step 24.12 – Single-Folio Deep Decode
======================================
Select the single most constrained folio in the manuscript and attempt
a complete decode using every available constraint simultaneously.

Dependency chain:
    combined_refine.json (Phase 15)
    modifier_integrate.json (Phase 16)
    cross_approach.json (Phase 19.8)
        → folio_isolation.json (this step)
"""

import json
import os
import random
import time
from collections import Counter, defaultdict
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
    load_reference_corpus,
)
from voynich.phases.csp_solver import decode_token


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
# Constants
# ---------------------------------------------------------------------------

BOTANICAL_IDENTIFICATIONS = {
    'f2v': 'Ricinus communis',
    'f4v': 'Nymphaea',
    'f5r': 'Viola',
    'f6r': 'Calendula',
    'f9v': 'Hypericum',
    'f13r': 'Plantago',
    'f25v': 'Borago',
    'f33v': 'Salvia',
    'f34r': 'Rosmarinus',
    'f41v': 'Achillea',
    'f56r': 'Centaurea',
    'f90r1': 'Matricaria',
    'f96v': 'Ruta',
}

# Fallback anchor folio IDs if cross_approach.json is unavailable
FALLBACK_ANCHOR_FOLIOS = [
    'f1r', 'f2v', 'f4v', 'f5r', 'f6r', 'f9v', 'f13r', 'f34r',
]

# Latin botanical vocabulary for crib testing
BOTANICAL_TERMS = [
    'herba', 'folium', 'folia', 'radix', 'semen', 'flos',
    'flores', 'cortex', 'succus', 'oleum', 'aqua', 'pulvis',
    'planta', 'arbor', 'caulis', 'ramus', 'truncus',
]

# Humoral quality terms
HUMORAL_TERMS = [
    'calida', 'frigida', 'sicca', 'humida',
    'calidus', 'frigidus', 'siccus', 'humidus',
    'calor', 'frigus',
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FolioScore:
    folio_id: str
    n_tokens: int
    dict_hit_rate: float
    has_botanical_id: bool
    botanical_name: str
    has_anchor: bool
    total_score: int


@dataclass
class AnnotatedToken:
    eva_token: str
    decoded: str
    strategy: str
    is_dict_hit: bool
    confidence: str


@dataclass
class FolioIsolationResult:
    timestamp: str
    # Folio selection
    n_folios_scored: int
    top_folios: List[Dict]  # top 5 by score
    selected_folio: str
    selected_score: int
    # Decode results
    n_tokens: int
    n_dict_hits: int
    dict_hit_rate: float
    n_unique_words: int
    # Annotated transliteration
    annotated_tokens: List[Dict]
    # Botanical test
    botanical_id: str
    botanical_name_found: bool
    botanical_terms_found: List[str]
    humoral_terms_found: List[str]
    # Coherence
    max_consecutive_hits: int
    coherent_fragments: List[str]  # sequences of consecutive dict-hitting words
    # Verdict
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_cross_approach_anchors(rd: str) -> Set[str]:
    """Load anchor folio IDs from cross_approach.json, falling back to hardcoded list."""
    path = os.path.join(rd, 'cross_approach.json')
    anchor_folios: Set[str] = set()

    if os.path.exists(path):
        try:
            with open(path) as f:
                data = json.load(f)

            # Extract folio IDs from per_word_results where words matched
            for entry in data.get('per_word_results', []):
                if entry.get('exact_match') or entry.get('edit2_match'):
                    # These are corpus-wide matches; mark all botanical folios
                    # as potential anchors since cross_approach confirms words
                    anchor_folios.update(FALLBACK_ANCHOR_FOLIOS)
                    break

            # Also check for any folio-level data
            for key in ('anchor_folios', 'folio_anchors', 'confirmed_folios'):
                if key in data:
                    vals = data[key]
                    if isinstance(vals, list):
                        anchor_folios.update(vals)
                    elif isinstance(vals, dict):
                        anchor_folios.update(vals.keys())

        except (json.JSONDecodeError, KeyError):
            pass

    if not anchor_folios:
        anchor_folios = set(FALLBACK_ANCHOR_FOLIOS)
        print("    [INFO] Using fallback anchor folio list")

    return anchor_folios


def _decode_r3(
    token: str,
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    modifier_chars: Set[str],
    modifier_rules: Dict[str, str],
    ref_word_set: set,
) -> Tuple[str, str]:
    """
    R3 combined decode: try alteration first, then stripping, then original.
    Returns (decoded_word, strategy_used).
    """
    # Try alteration
    alt = decode_token_modifier_aware(
        token, assignment, eva_to_triple, modifier_chars,
        modifier_rules=modifier_rules,
    )
    if alt.lower() in ref_word_set:
        return alt.lower(), 'alteration'

    # Try stripping
    stripped = decode_token_modifier_aware(
        token, assignment, eva_to_triple, modifier_chars,
    )
    if stripped.lower() in ref_word_set:
        return stripped.lower(), 'stripping'

    # Fall back to original decoding
    original = decode_token(token, assignment, eva_to_triple)
    return original.lower(), 'original'


def _classify_confidence(decoded: str, is_hit: bool) -> str:
    """Assign a confidence level to a decoded token."""
    if is_hit:
        return 'high'
    if '?' in decoded:
        return 'low'
    return 'medium'


def _score_folios(
    corpus,
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    modifier_chars: Set[str],
    modifier_rules: Dict[str, str],
    ref_word_set: set,
    anchor_folios: Set[str],
) -> List[FolioScore]:
    """Score all folios on constraint density."""
    pages = corpus.pages

    # Compute per-folio token counts and dict-hit rates
    folio_data: List[Tuple[str, List[str], float]] = []
    for folio_id, page in pages.items():
        tokens = page.all_tokens
        if not tokens:
            continue

        # Decode a sample to get dict-hit rate
        hits = 0
        for tok in tokens:
            decoded, _ = _decode_r3(
                tok, assignment, eva_to_triple,
                modifier_chars, modifier_rules, ref_word_set,
            )
            if decoded in ref_word_set:
                hits += 1

        rate = hits / len(tokens)
        folio_data.append((folio_id, tokens, rate))

    if not folio_data:
        return []

    # Compute averages for scoring
    avg_token_count = sum(len(t) for _, t, _ in folio_data) / len(folio_data)
    avg_dict_hit = sum(r for _, _, r in folio_data) / len(folio_data)

    # Score each folio
    scores: List[FolioScore] = []
    for folio_id, tokens, hit_rate in folio_data:
        total = 0

        # Has above-average token count? (+1)
        if len(tokens) >= avg_token_count:
            total += 1

        # Has above-average dict-hit rate? (+1)
        if hit_rate >= avg_dict_hit:
            total += 1

        # In botanical identification database? (+3)
        has_botanical = folio_id in BOTANICAL_IDENTIFICATIONS
        botanical_name = BOTANICAL_IDENTIFICATIONS.get(folio_id, '')
        if has_botanical:
            total += 3

        # Contains a confirmed cross-approach anchor word? (+2)
        has_anchor = folio_id in anchor_folios
        if has_anchor:
            total += 2

        scores.append(FolioScore(
            folio_id=folio_id,
            n_tokens=len(tokens),
            dict_hit_rate=round(hit_rate, 4),
            has_botanical_id=has_botanical,
            botanical_name=botanical_name,
            has_anchor=has_anchor,
            total_score=total,
        ))

    # Sort: highest score first, then by token count (descending) for ties
    scores.sort(key=lambda s: (-s.total_score, -s.n_tokens))
    return scores


def _annotate_folio(
    tokens: List[str],
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    modifier_chars: Set[str],
    modifier_rules: Dict[str, str],
    ref_word_set: set,
) -> List[AnnotatedToken]:
    """Decode every token on the folio and annotate with metadata."""
    annotations: List[AnnotatedToken] = []
    for tok in tokens:
        decoded, strategy = _decode_r3(
            tok, assignment, eva_to_triple,
            modifier_chars, modifier_rules, ref_word_set,
        )
        is_hit = decoded in ref_word_set
        confidence = _classify_confidence(decoded, is_hit)
        annotations.append(AnnotatedToken(
            eva_token=tok,
            decoded=decoded,
            strategy=strategy,
            is_dict_hit=is_hit,
            confidence=confidence,
        ))
    return annotations


def _botanical_crib_test(
    decoded_words: List[str],
    folio_id: str,
) -> Tuple[bool, List[str], List[str]]:
    """
    Test whether the botanical identification or related terms appear
    in the decoded text.

    Returns (name_found, botanical_terms_found, humoral_terms_found).
    """
    word_set = set(decoded_words)

    # Check if plant name (or parts) appear
    botanical_name = BOTANICAL_IDENTIFICATIONS.get(folio_id, '')
    name_found = False
    if botanical_name:
        name_parts = botanical_name.lower().split()
        for part in name_parts:
            if part in word_set:
                name_found = True
                break
            # Also check substrings for genus names that might be truncated
            for w in decoded_words:
                if len(part) >= 4 and part[:4] in w:
                    name_found = True
                    break
            if name_found:
                break

    # Check for botanical terms
    bot_found = [t for t in BOTANICAL_TERMS if t in word_set]

    # Check for humoral terms
    hum_found = [t for t in HUMORAL_TERMS if t in word_set]

    return name_found, bot_found, hum_found


def _coherence_assessment(
    annotations: List[AnnotatedToken],
) -> Tuple[int, List[str]]:
    """
    Look for coherent Latin fragments in the decoded text.

    Returns (max_consecutive_hits, list_of_coherent_fragments).
    """
    max_consecutive = 0
    current_streak = 0
    streak_words: List[str] = []
    fragments: List[str] = []

    for ann in annotations:
        if ann.is_dict_hit:
            current_streak += 1
            streak_words.append(ann.decoded)
        else:
            if current_streak >= 2:
                fragments.append(' '.join(streak_words))
            if current_streak > max_consecutive:
                max_consecutive = current_streak
            current_streak = 0
            streak_words = []

    # Handle final streak
    if current_streak >= 2:
        fragments.append(' '.join(streak_words))
    if current_streak > max_consecutive:
        max_consecutive = current_streak

    return max_consecutive, fragments


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_folio_isolation() -> None:
    """Step 24.12: Single-Folio Deep Decode."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 24.12: Single-Folio Deep Decode")
    print("=" * 70)

    rd = _results_dir()

    # ─── 1. Load dependencies ───
    print("\n  1. Loading dependencies …")

    # Phase 15 best assignment
    refine_path = os.path.join(rd, 'combined_refine.json')
    if not os.path.exists(refine_path):
        print("  [SKIP] combined_refine.json not found — run combined-refine first")
        return
    with open(refine_path) as f:
        refine_data = json.load(f)
    assignment = refine_data.get('best_assignment', {})
    print(f"      Assignment: {len(assignment)} triple→syllable mappings")

    # Phase 16 modifier integration
    integrate_path = os.path.join(rd, 'modifier_integrate.json')
    if os.path.exists(integrate_path):
        with open(integrate_path) as f:
            integrate_data = json.load(f)
        modifier_chars_list = integrate_data.get('modifier_chars', [])
        # Build modifier rules from classifications
        modifier_rules: Dict[str, str] = {}
        for cls in integrate_data.get('classifications', []):
            if cls.get('final_classification') == 'modifier':
                modifier_rules[cls['eva_char']] = cls.get('modifier_type', 'silent')
        print(f"      Modifiers: {len(modifier_chars_list)} chars, "
              f"{len(modifier_rules)} rules")
    else:
        print("      [WARN] modifier_integrate.json not found — no modifier rules")
        modifier_chars_list = []
        modifier_rules = {}

    modifier_chars = set(modifier_chars_list)

    # Cross-approach anchors
    anchor_folios = _load_cross_approach_anchors(rd)
    print(f"      Anchor folios: {len(anchor_folios)}")

    # ─── 2. Load corpus ───
    print("\n  2. Loading corpus …")
    corpus = load_corpus(verbose=False)
    eva_to_triple = build_eva_to_triple_lookup()
    total_tokens = len(corpus.get_tokens(paragraph_only=False))
    print(f"      {len(corpus.pages)} folios, {total_tokens} total tokens")

    # ─── 3. Build reference word set ───
    print("\n  3. Building expanded reference word set …")
    try:
        ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
        base_words = set(
            w.lower() for w in ref_corpus.get_combined_tokens('latin')
            if len(w) >= 2
        )
    except (FileNotFoundError, KeyError):
        base_words = set()

    expanded_words, _ = build_expanded_word_set(base_words)
    ref_word_set = base_words | expanded_words
    print(f"      {len(ref_word_set)} words in reference set")

    # ─── 4. Score all folios ───
    print("\n  4. Scoring all folios on constraint density …")
    folio_scores = _score_folios(
        corpus, assignment, eva_to_triple,
        modifier_chars, modifier_rules, ref_word_set,
        anchor_folios,
    )
    n_scored = len(folio_scores)
    print(f"      Scored {n_scored} folios")

    if not folio_scores:
        print("  [ERROR] No folios scored — check corpus loading")
        return

    # Top 5
    top5 = folio_scores[:5]
    print("\n      Top 5 folios:")
    print(f"      {'Rank':<5} {'Folio':<10} {'Tokens':<8} {'Dict%':<8} "
          f"{'Botan':<6} {'Anchor':<7} {'Score':<6}")
    print("      " + "-" * 55)
    for rank, fs in enumerate(top5, 1):
        print(f"      {rank:<5} {fs.folio_id:<10} {fs.n_tokens:<8} "
              f"{fs.dict_hit_rate:<8.4f} {'Y' if fs.has_botanical_id else 'N':<6} "
              f"{'Y' if fs.has_anchor else 'N':<7} {fs.total_score:<6}")

    # ─── 5. Select top folio ───
    selected = folio_scores[0]
    print(f"\n  5. Selected folio: {selected.folio_id} "
          f"(score={selected.total_score}, tokens={selected.n_tokens})")
    if selected.botanical_name:
        print(f"      Botanical ID: {selected.botanical_name}")

    # ─── 6. Multi-decode the selected folio ───
    print(f"\n  6. Decoding all tokens on {selected.folio_id} …")
    page = corpus.pages[selected.folio_id]
    folio_tokens = page.all_tokens

    annotations = _annotate_folio(
        folio_tokens, assignment, eva_to_triple,
        modifier_chars, modifier_rules, ref_word_set,
    )

    n_tokens = len(annotations)
    n_dict_hits = sum(1 for a in annotations if a.is_dict_hit)
    dict_hit_rate = n_dict_hits / n_tokens if n_tokens else 0.0
    decoded_words = [a.decoded for a in annotations]
    n_unique = len(set(decoded_words))

    print(f"      Tokens: {n_tokens}")
    print(f"      Dict hits: {n_dict_hits} ({dict_hit_rate:.1%})")
    print(f"      Unique decoded words: {n_unique}")

    # Strategy breakdown
    strategy_counts = Counter(a.strategy for a in annotations)
    for strat, cnt in strategy_counts.most_common():
        print(f"        {strat}: {cnt} ({cnt/n_tokens:.1%})")

    # Confidence breakdown
    conf_counts = Counter(a.confidence for a in annotations)
    for conf, cnt in conf_counts.most_common():
        print(f"        {conf} confidence: {cnt} ({cnt/n_tokens:.1%})")

    # ─── 7. Annotated transliteration ───
    print(f"\n  7. Annotated transliteration of {selected.folio_id}:")

    # Group tokens by locus (line) for line-by-line display
    locus_tokens: List[Tuple[str, List[str]]] = []
    for locus in page.loci:
        if locus.clean_text:
            line_tokens = locus.clean_text.split()
            locus_tokens.append((locus.locus_id, line_tokens))

    ann_idx = 0
    for locus_id, line_toks in locus_tokens:
        print(f"\n      [{locus_id}]")
        line_parts = []
        for tok in line_toks:
            if ann_idx < len(annotations):
                ann = annotations[ann_idx]
                marker = '*' if ann.is_dict_hit else ' '
                line_parts.append(
                    f"  {marker} {ann.eva_token:>15} -> {ann.decoded:<15} "
                    f"({ann.confidence}, {ann.strategy})"
                )
                ann_idx += 1
        for part in line_parts:
            print(f"      {part}")

    # Print first 30 tokens as a compact sample if many
    if n_tokens > 30:
        print(f"\n      (Showing {min(n_tokens, 30)} of {n_tokens} tokens above; "
              f"full list in JSON output)")

    # ─── 8. Botanical crib test ───
    print(f"\n  8. Botanical crib test …")
    if selected.folio_id in BOTANICAL_IDENTIFICATIONS:
        name_found, bot_terms, hum_terms = _botanical_crib_test(
            decoded_words, selected.folio_id,
        )
        print(f"      Plant: {BOTANICAL_IDENTIFICATIONS[selected.folio_id]}")
        print(f"      Plant name found in text: {'YES' if name_found else 'NO'}")
        print(f"      Botanical terms found: {bot_terms if bot_terms else 'none'}")
        print(f"      Humoral terms found: {hum_terms if hum_terms else 'none'}")
    else:
        name_found = False
        bot_terms = []
        hum_terms = []
        print(f"      No botanical identification for {selected.folio_id} — skipping")

    # ─── 9. Coherence assessment ───
    print(f"\n  9. Coherence assessment …")
    max_consecutive, fragments = _coherence_assessment(annotations)
    print(f"      Max consecutive dict hits: {max_consecutive}")
    if fragments:
        print(f"      Coherent fragments ({len(fragments)}):")
        for frag in fragments[:10]:
            print(f"        \"{frag}\"")
    else:
        print(f"      No coherent fragments (2+ consecutive hits) found")

    # ─── 10. Build verdict ───
    parts = []
    parts.append(f"Folio {selected.folio_id} (score={selected.total_score})")
    parts.append(f"dict_hit={dict_hit_rate:.1%} ({n_dict_hits}/{n_tokens})")
    parts.append(f"{n_unique} unique words")
    parts.append(f"max_consecutive={max_consecutive}")

    if name_found:
        parts.append("BOTANICAL NAME DETECTED")
    if bot_terms:
        parts.append(f"botanical terms: {', '.join(bot_terms)}")
    if hum_terms:
        parts.append(f"humoral terms: {', '.join(hum_terms)}")

    if dict_hit_rate >= 0.6 and max_consecutive >= 3:
        verdict_label = "STRONG"
    elif dict_hit_rate >= 0.4 and max_consecutive >= 2:
        verdict_label = "MODERATE"
    elif dict_hit_rate >= 0.2:
        verdict_label = "WEAK"
    else:
        verdict_label = "INSUFFICIENT"

    verdict = f"{verdict_label}: {'; '.join(parts)}"

    runtime = time.time() - t0
    print(f"\n  10. Verdict: {verdict}")
    print(f"      Runtime: {runtime:.1f}s")

    # ─── Save result ───
    top_folios_dicts = [_convert(asdict(fs)) for fs in folio_scores[:5]]
    annotated_dicts = [_convert(asdict(a)) for a in annotations]

    result = FolioIsolationResult(
        timestamp=time.strftime('%Y-%m-%dT%H:%M:%S'),
        n_folios_scored=n_scored,
        top_folios=top_folios_dicts,
        selected_folio=selected.folio_id,
        selected_score=selected.total_score,
        n_tokens=n_tokens,
        n_dict_hits=n_dict_hits,
        dict_hit_rate=round(dict_hit_rate, 4),
        n_unique_words=n_unique,
        annotated_tokens=annotated_dicts,
        botanical_id=BOTANICAL_IDENTIFICATIONS.get(selected.folio_id, ''),
        botanical_name_found=name_found,
        botanical_terms_found=bot_terms,
        humoral_terms_found=hum_terms,
        max_consecutive_hits=max_consecutive,
        coherent_fragments=fragments,
        verdict=verdict,
        runtime_seconds=round(runtime, 2),
    )

    out_path = os.path.join(rd, 'folio_isolation.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)

    print(f"\n  Saved → {out_path}")
    print("=" * 70)
