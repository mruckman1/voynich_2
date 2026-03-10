"""
Step 39.5 – Phrase-Level Crib Extraction
=========================================
From the 91 medical phrases (in ed1_decomposition.json), classify each
token as CONFIRMED (signal word), CONTENT_HIT (dict match), or MISS.
Find MISS tokens flanked by >=2 CONFIRMED words and generate candidate
words for MISS positions based on medical context.  Align candidates to
EVA chars/triples to extract proposed corrections.

Dependency chain:
    ed1_decomposition.json     (Step 39.1)
    merged_signal.json         (Step 38.3)
    merged_dict.json           (Step 38.1)
    targeted_vowel_fix.json    (Step 39.3)
        → phrase_cribs.json    (this step)
"""

import json
import os
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
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
# Token classification
# ---------------------------------------------------------------------------

def _classify_token(
    word: str,
    signal_words: Set[str],
    merged_dict: Set[str],
) -> str:
    """Classify a decoded word as CONFIRMED, CONTENT_HIT, or MISS."""
    if word in signal_words:
        return 'CONFIRMED'
    if word in merged_dict:
        return 'CONTENT_HIT'
    return 'MISS'


# ---------------------------------------------------------------------------
# Flanked-miss detection
# ---------------------------------------------------------------------------

def _find_flanked_misses(
    phrase: Dict,
    signal_words: Set[str],
    merged_dict: Set[str],
) -> List[Dict]:
    """Find MISS tokens flanked by >=2 CONFIRMED words on either side."""
    words = phrase['words']
    positions = phrase.get('positions', list(range(len(words))))
    classifications = [_classify_token(w, signal_words, merged_dict)
                       for w in words]

    flanked = []
    for i, (w, cls) in enumerate(zip(words, classifications)):
        if cls != 'MISS':
            continue

        # Count CONFIRMED tokens to the left
        left_confirmed = 0
        for j in range(i - 1, -1, -1):
            if classifications[j] == 'CONFIRMED':
                left_confirmed += 1
            else:
                break

        # Count CONFIRMED tokens to the right
        right_confirmed = 0
        for j in range(i + 1, len(classifications)):
            if classifications[j] == 'CONFIRMED':
                right_confirmed += 1
            else:
                break

        total_flank = left_confirmed + right_confirmed
        if total_flank >= 2:
            # Collect flanking words for context
            left_words = [words[j] for j in range(max(0, i - left_confirmed), i)]
            right_words = [words[j] for j in range(i + 1, min(len(words),
                                                               i + 1 + right_confirmed))]
            flanked.append({
                'miss_word': w,
                'miss_index_in_phrase': i,
                'position': positions[i] if i < len(positions) else -1,
                'left_confirmed': left_words,
                'right_confirmed': right_words,
                'left_count': left_confirmed,
                'right_count': right_confirmed,
                'folio': phrase.get('folio', 'unknown'),
            })

    return flanked


# ---------------------------------------------------------------------------
# Candidate generation from medical context
# ---------------------------------------------------------------------------

def _suggest_candidates(
    flanked_miss: Dict,
    phrase_words: List[str],
) -> List[str]:
    """Suggest candidate words for a flanked MISS based on medical context."""
    pharma_verbs = {'cola', 'recipe', 'misce', 'coque', 'dice', 'cura',
                    'sana', 'bibe', 'beni'}
    body_parts = {'cora', 'core', 'corpo', 'carne', 'ossa', 'pede',
                  'manu', 'dente', 'naso'}
    ingredients = {'rosa', 'sale', 'vino', 'olio', 'bene', 'sene',
                   'calce', 'suco'}
    qualities = {'bela', 'bona', 'calida', 'frigida', 'sicca',
                 'dulce', 'rara', 'nova'}

    left = set(flanked_miss.get('left_confirmed', []))
    right = set(flanked_miss.get('right_confirmed', []))
    all_context = left | right
    phrase_set = set(phrase_words)

    candidates = []

    # Between pharma_verb + body_part: suggest prepositions
    if (all_context & pharma_verbs) and (all_context & body_parts):
        candidates.extend(['per', 'in', 'de', 'con', 'cum', 'ad', 'super'])

    # Between ingredient + quality: suggest verbs
    if (all_context & ingredients) and (all_context & qualities):
        candidates.extend(['est', 'fa', 'sit', 'fit'])

    # Between body_part + quality: suggest linking words
    if (all_context & body_parts) and (all_context & qualities):
        candidates.extend(['est', 'et', 'cum', 'de', 'in'])

    # Between pharma_verb + ingredient: suggest prepositions/articles
    if (all_context & pharma_verbs) and (all_context & ingredients):
        candidates.extend(['de', 'in', 'cum', 'et', 'la', 'le'])

    # Between ingredient + ingredient: suggest conjunctions
    if len(all_context & ingredients) >= 2:
        candidates.extend(['et', 'e', 'cum', 'con'])

    # General fallback: common function words
    if not candidates:
        candidates.extend(['de', 'in', 'et', 'per', 'cum', 'ad',
                           'est', 'e', 'la', 'con'])

    # Deduplicate preserving order
    seen: Set[str] = set()
    unique = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique.append(c)

    return unique


# ---------------------------------------------------------------------------
# Alignment to EVA triples
# ---------------------------------------------------------------------------

def _try_align_candidate(
    candidate: str,
    eva_token: str,
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    confirmed_triples: Set[str],
) -> Optional[Dict]:
    """Try to align a candidate word with an EVA token's triples.

    Returns alignment info if successful, None if the candidate cannot
    be aligned without conflicting with confirmed triples.
    """
    triples = token_to_triples(eva_token, eva_to_triple)
    if not triples:
        return None

    # Build syllable sequence from candidate
    # Simple: split candidate into 2-char syllables (CV model)
    syllables = []
    i = 0
    while i < len(candidate):
        if i + 1 < len(candidate):
            syllables.append(candidate[i:i + 2])
            i += 2
        else:
            syllables.append(candidate[i])
            i += 1

    if len(syllables) != len(triples):
        return None

    # Check for conflicts with confirmed triples
    proposed = {}
    conflicts = []
    for triple_key, syl in zip(triples, syllables):
        current = assignment.get(triple_key, '')
        if triple_key in confirmed_triples and current != syl:
            conflicts.append({
                'triple_key': triple_key,
                'current': current,
                'proposed': syl,
            })
        proposed[triple_key] = syl

    if conflicts:
        return None

    return {
        'candidate': candidate,
        'eva_token': eva_token,
        'triples': triples,
        'syllables': syllables,
        'proposed_changes': {k: v for k, v in proposed.items()
                            if assignment.get(k, '') != v},
    }


# ---------------------------------------------------------------------------
# Cross-phrase consistency
# ---------------------------------------------------------------------------

def _cross_phrase_consistency(
    all_alignments: List[Dict],
) -> List[Dict]:
    """Find triple corrections proposed by multiple phrases."""
    # Count how many phrases propose each (triple_key, syllable) pair
    proposal_count: Counter = Counter()
    proposal_sources: Dict[Tuple[str, str], List[str]] = defaultdict(list)

    for alignment in all_alignments:
        folio = alignment.get('folio', 'unknown')
        for triple_key, syl in alignment.get('proposed_changes', {}).items():
            proposal_count[(triple_key, syl)] += 1
            proposal_sources[(triple_key, syl)].append(folio)

    consistent = []
    for (triple_key, syl), count in proposal_count.most_common():
        if count >= 2:
            consistent.append({
                'triple_key': triple_key,
                'proposed_syllable': syl,
                'n_supporting_phrases': count,
                'supporting_folios': proposal_sources[(triple_key, syl)],
            })

    return consistent


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_phrase_cribs() -> None:
    """Step 39.5: Phrase-Level Crib Extraction."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 39.5: Phrase-Level Crib Extraction")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load inputs ──
    print("\n  1. Loading inputs …")
    ed1_data = _safe_load(os.path.join(rd, 'ed1_decomposition.json'))
    signal_data = _safe_load(os.path.join(rd, 'merged_signal.json'))
    dict_data = _safe_load(os.path.join(rd, 'merged_dict.json'))
    vowel_data = _safe_load(os.path.join(rd, 'targeted_vowel_fix.json'))
    refine_data = _safe_load(os.path.join(rd, 'combined_refine.json'))

    medical_phrases = ed1_data.get('medical_phrases_full', [])
    merged_words = set(dict_data.get('merged_words', []))

    word_signals = signal_data.get('word_signals', [])
    signal_words = {w['word'] for w in word_signals
                    if w.get('is_genuine_signal')}

    # Corrected assignment: prefer targeted_vowel_fix, fallback to combined_refine
    assignment = dict(vowel_data.get('corrected_assignment', {}))
    if not assignment:
        assignment = dict(refine_data.get('best_assignment', {}))

    # Token-level decoded words and EVA tokens
    token_decoded = signal_data.get('token_decoded', [])
    token_evas_all = _safe_load(os.path.join(rd, 'decode_10k.json')).get(
        'token_evas', [])

    eva_to_triple = build_eva_to_triple_lookup()

    # Build confirmed triples (triples with high-confidence assignments)
    confirmed_triples: Set[str] = set()
    crib_data = _safe_load(os.path.join(rd, 'crib_extraction.json'))
    for crib in crib_data.get('cribs', []):
        if crib.get('tier') in ('Tier-1', 'Tier-2', 1, 2):
            for triple_key in crib.get('triple_keys', []):
                confirmed_triples.add(triple_key)

    print(f"     Medical phrases: {len(medical_phrases)}")
    print(f"     Signal words: {len(signal_words)}")
    print(f"     Merged dict size: {len(merged_words)}")
    print(f"     Assignment entries: {len(assignment)}")
    print(f"     Confirmed triples: {len(confirmed_triples)}")

    # ── 2. Classify tokens in each phrase ──
    print("\n  2. Classifying tokens in medical phrases …")
    annotated_phrases = []
    total_confirmed = 0
    total_content_hit = 0
    total_miss = 0

    for phrase in medical_phrases:
        words = phrase.get('words', [])
        classifications = [_classify_token(w, signal_words, merged_words)
                           for w in words]
        n_conf = classifications.count('CONFIRMED')
        n_hit = classifications.count('CONTENT_HIT')
        n_miss = classifications.count('MISS')
        total_confirmed += n_conf
        total_content_hit += n_hit
        total_miss += n_miss

        annotated_phrases.append({
            'folio': phrase.get('folio', 'unknown'),
            'words': words,
            'classifications': classifications,
            'positions': phrase.get('positions', []),
            'n_confirmed': n_conf,
            'n_content_hit': n_hit,
            'n_miss': n_miss,
        })

    total_tokens = total_confirmed + total_content_hit + total_miss
    print(f"     Total phrase tokens: {total_tokens}")
    print(f"     CONFIRMED: {total_confirmed} "
          f"({100 * total_confirmed / max(total_tokens, 1):.1f}%)")
    print(f"     CONTENT_HIT: {total_content_hit} "
          f"({100 * total_content_hit / max(total_tokens, 1):.1f}%)")
    print(f"     MISS: {total_miss} "
          f"({100 * total_miss / max(total_tokens, 1):.1f}%)")

    # ── 3. Find flanked misses ──
    print("\n  3. Finding flanked MISS tokens …")
    all_flanked: List[Dict] = []

    for phrase in medical_phrases:
        flanked = _find_flanked_misses(phrase, signal_words, merged_words)
        all_flanked.extend(flanked)

    print(f"     Flanked MISS tokens: {len(all_flanked)}")
    for fm in all_flanked[:10]:
        print(f"       {fm['folio']}: '{fm['miss_word']}' flanked by "
              f"{fm['left_confirmed']} | {fm['right_confirmed']}")

    # ── 4. Generate candidates for flanked misses ──
    print("\n  4. Generating candidates for flanked misses …")
    candidates_generated = 0
    flanked_with_candidates: List[Dict] = []

    for fm in all_flanked:
        # Find the phrase this miss belongs to
        phrase_words = []
        for phrase in medical_phrases:
            positions = phrase.get('positions', [])
            if fm['position'] in positions:
                phrase_words = phrase.get('words', [])
                break

        candidates = _suggest_candidates(fm, phrase_words)
        candidates_generated += len(candidates)

        fm_entry = dict(fm)
        fm_entry['candidates'] = candidates
        fm_entry['n_candidates'] = len(candidates)
        flanked_with_candidates.append(fm_entry)

    print(f"     Total candidates generated: {candidates_generated}")

    # ── 5. Align candidates to EVA triples ──
    print("\n  5. Aligning candidates to EVA triples …")
    all_alignments: List[Dict] = []

    for fm in flanked_with_candidates:
        pos = fm.get('position', -1)
        if pos < 0 or pos >= len(token_evas_all):
            continue

        eva_token = token_evas_all[pos]
        for candidate in fm.get('candidates', []):
            alignment = _try_align_candidate(
                candidate, eva_token, assignment,
                eva_to_triple, confirmed_triples,
            )
            if alignment is not None:
                alignment['folio'] = fm.get('folio', 'unknown')
                alignment['miss_word'] = fm.get('miss_word', '')
                alignment['left_confirmed'] = fm.get('left_confirmed', [])
                alignment['right_confirmed'] = fm.get('right_confirmed', [])
                all_alignments.append(alignment)

    print(f"     Successful alignments: {len(all_alignments)}")
    for al in all_alignments[:10]:
        changes = al.get('proposed_changes', {})
        print(f"       {al['folio']}: '{al['candidate']}' → "
              f"{len(changes)} proposed change(s)")

    # ── 6. Cross-phrase consistency ──
    print("\n  6. Cross-phrase consistency check …")
    proposed_corrections = _cross_phrase_consistency(all_alignments)

    print(f"     Multi-phrase corrections: {len(proposed_corrections)}")
    for pc in proposed_corrections[:10]:
        print(f"       {pc['triple_key']} → '{pc['proposed_syllable']}' "
              f"({pc['n_supporting_phrases']} phrases, "
              f"folios: {pc['supporting_folios'][:3]})")

    # ── 7. Build single-source corrections (from alignments) ──
    print("\n  7. Collecting all proposed corrections …")
    # Include both multi-phrase and single-phrase corrections
    single_source: List[Dict] = []
    multi_keys = {pc['triple_key'] for pc in proposed_corrections}

    for al in all_alignments:
        for triple_key, syl in al.get('proposed_changes', {}).items():
            if triple_key not in multi_keys:
                single_source.append({
                    'triple_key': triple_key,
                    'proposed_syllable': syl,
                    'n_supporting_phrases': 1,
                    'source': 'single_phrase',
                    'folio': al.get('folio', 'unknown'),
                    'candidate': al.get('candidate', ''),
                })

    all_corrections = proposed_corrections + single_source
    print(f"     Total proposed corrections: {len(all_corrections)}")

    # ── 8. Save ──
    elapsed = time.time() - t0

    verdict_parts = [
        f"{len(medical_phrases)} medical phrases analyzed",
        f"{len(all_flanked)} flanked misses found",
        f"{candidates_generated} candidates generated",
        f"{len(all_alignments)} alignments found",
        f"{len(proposed_corrections)} multi-phrase corrections",
    ]

    output = {
        'n_medical_phrases': len(medical_phrases),
        'n_flanked_misses': len(all_flanked),
        'n_candidates_generated': candidates_generated,
        'n_alignments_found': len(all_alignments),
        'proposed_corrections': all_corrections,
        'multi_phrase_corrections': proposed_corrections,
        'phrase_annotations': annotated_phrases[:100],
        'flanked_misses': flanked_with_candidates[:50],
        'alignments': all_alignments[:100],
        'token_classification_summary': {
            'total_tokens': total_tokens,
            'n_confirmed': total_confirmed,
            'n_content_hit': total_content_hit,
            'n_miss': total_miss,
        },
        'verdict': '. '.join(verdict_parts) + '.',
        'runtime_seconds': round(elapsed, 1),
    }

    out_path = os.path.join(rd, 'phrase_cribs.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(output), f, indent=2)
    print(f"\n  Saved → {out_path} ({elapsed:.1f}s)")
