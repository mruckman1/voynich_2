"""
Step 26.2 – Multi-Language Month Name Crib Analysis
====================================================
For every zodiac folio, test whether any Voynichese label or text sequence
encodes the expected month name, in each of the candidate languages.

Dependency chain:
    zodiac_map.json (Step 26.1)
    combined_refine.json (Phase 15)
    modifier_integrate.json (Phase 16)
        → month_crib.json
"""

import itertools
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
    token_to_triples,
    tokenize_eva_chars,
)
from voynich.core.reference import (
    MONTH_NAMES_MULTI,
    build_expanded_word_set,
    generate_medieval_variants,
    load_reference_corpus,
)


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
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MonthMatch:
    folio: str
    zodiac_sign: str
    language: str
    month_name: str
    label_text: str
    label_locus_id: str
    decoded_text: str
    agreement_rate: float
    match_type: str       # 'exact', 'close', 'partial', 'none'
    syllable_detail: List[Dict]  # per-syllable comparison


@dataclass
class CSPSolution:
    folio: str
    zodiac_sign: str
    language: str
    month_name: str
    label_text: str
    label_triples: List[str]
    assignment: Dict[str, str]  # triple -> syllable
    decoded: str


@dataclass
class MonthCribResult:
    timestamp: str
    n_languages: int
    n_folios: int
    # Per-language summary
    language_scores: List[Dict]
    best_language: str
    best_language_score: float
    # Forward matches (Phase 16 decode → compare to month)
    forward_matches: List[Dict]
    n_forward_exact: int
    n_forward_close: int
    n_forward_partial: int
    # Table-independent CSP solutions
    csp_solutions: List[Dict]
    n_csp_solutions: int
    best_csp_language: str
    # Cross-folio consistency
    consistent_assignments: Dict[str, str]  # triple -> syllable (agreed across folios)
    n_consistent: int
    conflicting_triples: List[Dict]
    # Null control
    null_mean_score: float
    null_std_score: float
    real_score: float
    selectivity_ratio: float
    # Verdict
    gate_passed: bool
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VOWELS = set('aeiou')


def _syllabify_word(word: str) -> List[str]:
    """Simple CV syllabification for Latin/Romance words."""
    word = word.lower().strip()
    if not word:
        return []

    vowel_positions = [i for i, ch in enumerate(word) if ch in _VOWELS]
    if not vowel_positions:
        return [word] if word else []

    syllables = []
    start = 0
    for vi in range(len(vowel_positions)):
        vpos = vowel_positions[vi]
        if vi < len(vowel_positions) - 1:
            next_vpos = vowel_positions[vi + 1]
            consonant_span = next_vpos - vpos - 1
            if consonant_span <= 0:
                end = vpos + 1
            elif consonant_span == 1:
                end = vpos + 1
            else:
                end = next_vpos - 1
            syllables.append(word[start:end])
            start = end
        else:
            syllables.append(word[start:])
    return syllables


def _build_reverse_table(assignment: Dict[str, str]) -> Dict[str, List[str]]:
    """Build syllable -> list of triples reverse lookup."""
    reverse: Dict[str, List[str]] = defaultdict(list)
    for triple, syllable in assignment.items():
        reverse[syllable].append(triple)
    return dict(reverse)


def _get_label_triples(
    label_text: str,
    eva_to_triple: Dict[str, str],
    modifier_chars: Set[str],
) -> List[str]:
    """Extract syllabic triples from a label (excluding modifiers)."""
    triples = []
    for token in label_text.split():
        token = token.strip()
        if not token:
            continue
        chars = tokenize_eva_chars(token)
        for ch in chars:
            if ch in modifier_chars:
                continue
            triple = eva_to_triple.get(ch)
            if triple:
                triples.append(triple)
    return triples


def _agreement_score(decoded: str, target: str) -> Tuple[float, str, List[Dict]]:
    """Compute agreement between decoded text and target month name.

    Returns (score, match_type, syllable_detail).
    """
    decoded_lower = decoded.lower().replace('?', '')
    target_lower = target.lower()

    # Exact match
    if decoded_lower == target_lower:
        return 1.0, 'exact', []

    # Close match (edit distance 1)
    if len(decoded_lower) > 0 and len(target_lower) > 0:
        if abs(len(decoded_lower) - len(target_lower)) <= 1:
            diffs = 0
            for a, b in zip(decoded_lower, target_lower):
                if a != b:
                    diffs += 1
            diffs += abs(len(decoded_lower) - len(target_lower))
            if diffs <= 1:
                return 0.8, 'close', [{'diff': diffs}]

    # Partial match (syllable comparison)
    dec_syls = _syllabify_word(decoded_lower)
    tgt_syls = _syllabify_word(target_lower)
    if dec_syls and tgt_syls:
        matches = 0
        detail = []
        for i, (d, t) in enumerate(zip(dec_syls, tgt_syls)):
            if d == t:
                matches += 1
                detail.append({'pos': i, 'decoded': d, 'target': t, 'match': True})
            else:
                detail.append({'pos': i, 'decoded': d, 'target': t, 'match': False})
        max_len = max(len(dec_syls), len(tgt_syls))
        rate = matches / max_len if max_len > 0 else 0
        if rate >= 0.5:
            return round(rate, 4), 'partial', detail

    # Substring containment
    if target_lower in decoded_lower or decoded_lower in target_lower:
        shorter = min(len(decoded_lower), len(target_lower))
        longer = max(len(decoded_lower), len(target_lower))
        return round(shorter / longer, 4) * 0.6, 'partial', []

    return 0.0, 'none', []


def _build_month_variants(base_names: Dict[str, List[str]]) -> Dict[str, List[Tuple[str, str]]]:
    """Build month variant table: language -> [(month_name, variant)]."""
    result: Dict[str, List[Tuple[str, str]]] = {}
    for lang, names in base_names.items():
        variants = []
        for name in names:
            variants.append((name, name))
            # Add medieval variants
            med = generate_medieval_variants(name)
            for var_word in med:
                if var_word != name:
                    variants.append((name, var_word))
        result[lang] = variants
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_month_crib() -> None:
    t0 = time.time()
    print("=" * 70)
    print("STEP 26.2: Multi-Language Month Name Crib Analysis")
    print("=" * 70)

    rd = _results_dir()

    # Load dependencies
    zodiac_data = _load_json(os.path.join(rd, 'zodiac_map.json'))
    if not zodiac_data:
        print("  [SKIP] zodiac_map.json not found — run zodiac-map first")
        return

    refine_data = _load_json(os.path.join(rd, 'combined_refine.json'))
    if not refine_data:
        print("  [SKIP] combined_refine.json not found — run combined-refine first")
        return

    mod_data = _load_json(os.path.join(rd, 'modifier_integrate.json'))
    if not mod_data:
        print("  [SKIP] modifier_integrate.json not found — run mod-integrate first")
        return

    assignment = refine_data.get('best_assignment', {})
    modifier_chars = set(mod_data.get('modifier_chars', []))

    corpus = load_corpus(verbose=False)
    eva_to_triple = build_eva_to_triple_lookup()
    reverse_table = _build_reverse_table(assignment)

    # Get all CV syllables in current assignment as the domain
    all_syllables = sorted(set(assignment.values()))

    # Build month variants
    month_variants = _build_month_variants(MONTH_NAMES_MULTI)

    folio_map = zodiac_data.get('folio_map', [])

    print(f"\n  1. Testing {len(MONTH_NAMES_MULTI)} languages across "
          f"{len(folio_map)} zodiac folios ...")

    # -----------------------------------------------------------------------
    # Forward test: decode labels via Phase 16, compare to expected month
    # -----------------------------------------------------------------------
    all_forward_matches: List[MonthMatch] = []
    lang_best_scores: Dict[str, List[float]] = defaultdict(list)

    for finfo in folio_map:
        folio = finfo['folio']
        sign = finfo['zodiac_sign']
        month_idx = finfo['month_index']  # 1-based

        page = corpus.get_page(folio)
        if page is None:
            continue

        # Collect all labels and text on this folio
        all_labels = finfo.get('labels', []) + finfo.get('radial_labels', [])

        for lang, names in MONTH_NAMES_MULTI.items():
            target_month = names[month_idx - 1]  # 0-indexed list

            best_match = None
            best_score = 0.0

            for label_info in all_labels:
                label_text = label_info.get('eva_text', '')
                if not label_text.strip():
                    continue

                # Decode each token in label
                decoded_parts = []
                for token in label_text.split():
                    token = token.strip()
                    if not token:
                        continue
                    dec = decode_token_modifier_aware(
                        token, assignment, eva_to_triple, modifier_chars
                    )
                    decoded_parts.append(dec)

                decoded = ''.join(decoded_parts)
                score, match_type, detail = _agreement_score(decoded, target_month)

                if score > best_score:
                    best_score = score
                    best_match = MonthMatch(
                        folio=folio,
                        zodiac_sign=sign,
                        language=lang,
                        month_name=target_month,
                        label_text=label_text,
                        label_locus_id=label_info.get('locus_id', ''),
                        decoded_text=decoded,
                        agreement_rate=round(score, 4),
                        match_type=match_type,
                        syllable_detail=detail,
                    )

            # Also check circular text
            for circ_info in finfo.get('circular_texts', []):
                circ_text = circ_info.get('eva_text', '')
                if not circ_text.strip():
                    continue
                # Decode entire circular text
                decoded_parts = []
                for token in circ_text.split():
                    token = token.strip()
                    if not token:
                        continue
                    dec = decode_token_modifier_aware(
                        token, assignment, eva_to_triple, modifier_chars
                    )
                    decoded_parts.append(dec)

                # Search for month name as substring in decoded circular text
                full_decoded = ' '.join(decoded_parts)
                cat_decoded = ''.join(decoded_parts)

                for variant_base, variant in month_variants.get(lang, []):
                    if variant_base != target_month:
                        continue
                    if variant.lower() in cat_decoded.lower():
                        score = 0.9
                        if score > best_score:
                            best_score = score
                            best_match = MonthMatch(
                                folio=folio,
                                zodiac_sign=sign,
                                language=lang,
                                month_name=target_month,
                                label_text=circ_text[:60] + '...',
                                label_locus_id=circ_info.get('locus_id', ''),
                                decoded_text=cat_decoded[:60],
                                agreement_rate=0.9,
                                match_type='substring',
                                syllable_detail=[{'found': variant}],
                            )

            if best_match and best_score > 0:
                all_forward_matches.append(best_match)
                lang_best_scores[lang].append(best_score)

    # Summarize forward matches
    n_exact = sum(1 for m in all_forward_matches if m.match_type == 'exact')
    n_close = sum(1 for m in all_forward_matches if m.match_type == 'close')
    n_partial = sum(1 for m in all_forward_matches if m.match_type == 'partial')

    print(f"\n  2. Forward test results:")
    print(f"      Total matches: {len(all_forward_matches)} "
          f"(exact={n_exact}, close={n_close}, partial={n_partial})")

    language_scores = []
    for lang in sorted(MONTH_NAMES_MULTI.keys()):
        scores = lang_best_scores.get(lang, [])
        mean_s = sum(scores) / len(scores) if scores else 0
        max_s = max(scores) if scores else 0
        language_scores.append({
            'language': lang,
            'mean_agreement': round(mean_s, 4),
            'max_agreement': round(max_s, 4),
            'n_matches': len(scores),
            'n_above_half': sum(1 for s in scores if s >= 0.5),
        })
        print(f"      {lang:12s}: mean={mean_s:.3f}, max={max_s:.3f}, "
              f"n≥0.5={sum(1 for s in scores if s >= 0.5)}")

    best_lang = max(language_scores, key=lambda x: x['mean_agreement'])

    # -----------------------------------------------------------------------
    # Table-independent CSP: for short labels, try all assignments
    # -----------------------------------------------------------------------
    print(f"\n  3. Table-independent CSP on short labels (≤4 syllabic triples) ...")

    csp_solutions: List[CSPSolution] = []
    MAX_TRIPLES_FOR_CSP = 4

    for finfo in folio_map:
        folio = finfo['folio']
        sign = finfo['zodiac_sign']
        month_idx = finfo['month_index']

        all_labels = finfo.get('labels', []) + finfo.get('radial_labels', [])

        for label_info in all_labels:
            label_text = label_info.get('eva_text', '')
            if not label_text.strip():
                continue

            triples = _get_label_triples(label_text, eva_to_triple, modifier_chars)
            if not triples or len(triples) > MAX_TRIPLES_FOR_CSP:
                continue

            n_triples = len(triples)

            # For each language, try to find an assignment that produces
            # any month name for this folio
            for lang, names in MONTH_NAMES_MULTI.items():
                target_month = names[month_idx - 1]
                target_syls = _syllabify_word(target_month)

                if len(target_syls) != n_triples:
                    continue  # syllable count must match triple count

                # Direct assignment: triple[i] -> target_syls[i]
                candidate_assignment = {}
                valid = True
                for i, (triple, syl) in enumerate(zip(triples, target_syls)):
                    if triple in candidate_assignment:
                        if candidate_assignment[triple] != syl:
                            valid = False
                            break
                    candidate_assignment[triple] = syl

                if valid:
                    decoded = ''.join(target_syls)
                    csp_solutions.append(CSPSolution(
                        folio=folio,
                        zodiac_sign=sign,
                        language=lang,
                        month_name=target_month,
                        label_text=label_text,
                        label_triples=triples,
                        assignment=candidate_assignment,
                        decoded=decoded,
                    ))

    print(f"      Found {len(csp_solutions)} CSP solutions")
    if csp_solutions:
        csp_lang_counts = Counter(s.language for s in csp_solutions)
        best_csp_lang = csp_lang_counts.most_common(1)[0][0]
        for lang, cnt in csp_lang_counts.most_common():
            print(f"        {lang}: {cnt} solutions")
    else:
        best_csp_lang = ''

    # -----------------------------------------------------------------------
    # Cross-folio consistency
    # -----------------------------------------------------------------------
    print(f"\n  4. Cross-folio consistency check ...")

    # Collect all triple -> syllable assignments from CSP solutions, grouped by folio
    folio_assignments: Dict[str, Dict[str, str]] = defaultdict(dict)
    for sol in csp_solutions:
        for triple, syl in sol.assignment.items():
            key = f"{sol.folio}:{triple}"
            folio_assignments[sol.language]
            if sol.language not in folio_assignments:
                folio_assignments[sol.language] = {}
            folio_assignments[sol.language][key] = syl

    # Check consistency: same triple across different folios
    consistent_assignments: Dict[str, str] = {}
    conflicting: List[Dict] = []

    triple_evidence: Dict[str, Dict[str, Set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for sol in csp_solutions:
        for triple, syl in sol.assignment.items():
            triple_evidence[triple][syl].add(sol.folio)

    for triple, syl_map in triple_evidence.items():
        if len(syl_map) == 1:
            syl = list(syl_map.keys())[0]
            folios = list(syl_map.values())[0]
            if len(folios) >= 2:
                consistent_assignments[triple] = syl
        elif len(syl_map) > 1:
            conflicting.append({
                'triple': triple,
                'assignments': {s: sorted(list(fs)) for s, fs in syl_map.items()},
            })

    print(f"      Consistent (≥2 folios): {len(consistent_assignments)}")
    print(f"      Conflicting:            {len(conflicting)}")
    for triple, syl in consistent_assignments.items():
        print(f"        {triple} → {syl}")

    # -----------------------------------------------------------------------
    # Null control: shuffle month assignments
    # -----------------------------------------------------------------------
    print(f"\n  5. Null control (100 random month-folio permutations) ...")

    rng = random.Random(42)
    null_scores: List[float] = []

    for perm_i in range(100):
        # Shuffle month indices across folios
        month_indices = [fi['month_index'] for fi in folio_map]
        rng.shuffle(month_indices)

        perm_scores = []
        for fi_idx, finfo in enumerate(folio_map):
            folio = finfo['folio']
            wrong_month_idx = month_indices[fi_idx]

            all_labels = finfo.get('labels', []) + finfo.get('radial_labels', [])

            for lang, names in MONTH_NAMES_MULTI.items():
                target_month = names[wrong_month_idx - 1]

                for label_info in all_labels:
                    label_text = label_info.get('eva_text', '')
                    if not label_text.strip():
                        continue

                    decoded_parts = []
                    for token in label_text.split():
                        token = token.strip()
                        if not token:
                            continue
                        dec = decode_token_modifier_aware(
                            token, assignment, eva_to_triple, modifier_chars
                        )
                        decoded_parts.append(dec)

                    decoded = ''.join(decoded_parts)
                    score, _, _ = _agreement_score(decoded, target_month)
                    perm_scores.append(score)

        null_scores.append(max(perm_scores) if perm_scores else 0)

    null_mean = sum(null_scores) / len(null_scores) if null_scores else 0
    null_std = (sum((s - null_mean) ** 2 for s in null_scores) / len(null_scores)) ** 0.5 if null_scores else 0

    real_best = best_lang['max_agreement']
    selectivity = real_best / null_mean if null_mean > 0 else float('inf')

    print(f"      Null mean: {null_mean:.4f} ± {null_std:.4f}")
    print(f"      Real best: {real_best:.4f}")
    print(f"      Selectivity: {selectivity:.2f}×")

    gate = selectivity > 1.5
    if gate:
        verdict = (f"PASS: Month crib selectivity {selectivity:.2f}× > 1.5. "
                   f"Best language: {best_lang['language']} "
                   f"(mean={best_lang['mean_agreement']:.3f}). "
                   f"{len(consistent_assignments)} consistent assignments.")
    else:
        verdict = (f"MARGINAL: Month crib selectivity {selectivity:.2f}× "
                   f"(threshold 1.5). Best language: {best_lang['language']}. "
                   f"{len(csp_solutions)} CSP solutions found.")

    print(f"\n  6. Verdict: {verdict}")

    result = MonthCribResult(
        timestamp=time.strftime('%Y-%m-%dT%H:%M:%S'),
        n_languages=len(MONTH_NAMES_MULTI),
        n_folios=len(folio_map),
        language_scores=language_scores,
        best_language=best_lang['language'],
        best_language_score=best_lang['mean_agreement'],
        forward_matches=[_convert(asdict(m)) for m in all_forward_matches],
        n_forward_exact=n_exact,
        n_forward_close=n_close,
        n_forward_partial=n_partial,
        csp_solutions=[_convert(asdict(s)) for s in csp_solutions],
        n_csp_solutions=len(csp_solutions),
        best_csp_language=best_csp_lang,
        consistent_assignments=consistent_assignments,
        n_consistent=len(consistent_assignments),
        conflicting_triples=conflicting,
        null_mean_score=round(null_mean, 4),
        null_std_score=round(null_std, 4),
        real_score=round(real_best, 4),
        selectivity_ratio=round(selectivity, 4),
        gate_passed=gate,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'month_crib.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2)

    print(f"\n  → {out_path}")
