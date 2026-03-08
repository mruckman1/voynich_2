"""
Step 26.4 – Per-Label Exhaustive CSP Decode
===========================================
For each Voynichese label on a zodiac folio, attempt exhaustive decode
by enumerating all possible syllable assignments for short labels.

Dependency chain:
    zodiac_map.json (Step 26.1)
    combined_refine.json (Phase 15)
    modifier_integrate.json (Phase 16)
        → label_decode.json
"""

import itertools
import json
import os
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
    build_expanded_word_set,
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
class LabelResult:
    folio: str
    zodiac_sign: str
    locus_id: str
    eva_text: str
    n_syllabic_triples: int
    # Phase 16 decode
    phase16_decode: str
    phase16_is_dict_hit: bool
    # CSP results (only for short labels)
    csp_attempted: bool
    n_candidates_tested: int
    n_dict_hits: int
    top_decodings: List[Dict]  # [{decoded, syllables, triples, is_dict_hit}]
    best_csp_decode: str
    best_csp_is_dict_hit: bool


@dataclass
class LabelDecodeResult:
    timestamp: str
    n_labels: int
    n_csp_attempted: int
    n_csp_with_hits: int
    # Phase 16 baseline
    n_phase16_hits: int
    phase16_hit_rate: float
    # CSP results
    n_csp_hits_total: int
    csp_hit_rate: float
    improvement_over_phase16: float
    # Per-label details
    label_results: List[Dict]
    # Derived assignments from CSP dict hits
    derived_assignments: Dict[str, str]
    n_derived: int
    # Agreement with Phase 16
    n_agree: int
    n_disagree: int
    agreement_rate: float
    # Verdict
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_syllabic_triples(
    text: str,
    eva_to_triple: Dict[str, str],
    modifier_chars: Set[str],
) -> List[Tuple[str, str]]:
    """Get (eva_char, triple_key) pairs for syllabic characters in text."""
    result = []
    for token in text.split():
        token = token.strip()
        if not token:
            continue
        chars = tokenize_eva_chars(token)
        for ch in chars:
            if ch in modifier_chars:
                continue
            triple = eva_to_triple.get(ch)
            if triple:
                result.append((ch, triple))
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_label_decode() -> None:
    t0 = time.time()
    print("=" * 70)
    print("STEP 26.4: Per-Label Exhaustive CSP Decode")
    print("=" * 70)

    rd = _results_dir()

    # Load dependencies
    zodiac_data = _load_json(os.path.join(rd, 'zodiac_map.json'))
    if not zodiac_data:
        print("  [SKIP] zodiac_map.json not found — run zodiac-map first")
        return

    refine_data = _load_json(os.path.join(rd, 'combined_refine.json'))
    if not refine_data:
        print("  [SKIP] combined_refine.json not found")
        return

    mod_data = _load_json(os.path.join(rd, 'modifier_integrate.json'))
    if not mod_data:
        print("  [SKIP] modifier_integrate.json not found")
        return

    assignment = refine_data.get('best_assignment', {})
    modifier_chars = set(mod_data.get('modifier_chars', []))

    corpus = load_corpus(verbose=False)
    eva_to_triple = build_eva_to_triple_lookup()

    # Build expanded word set for dict-hit checking
    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    base_words = set()
    for text in ref_corpus.get_texts('latin'):
        base_words.update(w.lower() for w in text.tokens if len(w) >= 2)
    expanded_words, _ = build_expanded_word_set(base_words)

    # Get all unique syllables in the assignment as the CSP domain
    all_syllables = sorted(set(assignment.values()))
    print(f"\n  CSP domain: {len(all_syllables)} syllables")

    folio_map = zodiac_data.get('folio_map', [])

    MAX_TRIPLES = 3  # Exhaustive CSP only for very short labels
    TOP_K = 10

    print(f"\n  1. Processing labels across {len(folio_map)} zodiac folios ...")
    print(f"     (exhaustive CSP for labels with ≤{MAX_TRIPLES} syllabic triples)")

    all_label_results: List[LabelResult] = []
    n_csp_attempted = 0
    n_csp_with_hits = 0
    n_phase16_hits = 0
    total_labels = 0

    for finfo in folio_map:
        folio = finfo['folio']
        sign = finfo['zodiac_sign']

        # Process both Lz labels and Ri radial labels
        all_labels = finfo.get('labels', []) + finfo.get('radial_labels', [])

        for label_info in all_labels:
            eva_text = label_info.get('eva_text', '')
            if not eva_text.strip():
                continue

            total_labels += 1
            locus_id = label_info.get('locus_id', '')

            # Phase 16 decode
            decoded_parts = []
            for token in eva_text.split():
                token = token.strip()
                if not token:
                    continue
                dec = decode_token_modifier_aware(
                    token, assignment, eva_to_triple, modifier_chars
                )
                decoded_parts.append(dec)

            phase16_decoded = ''.join(decoded_parts)
            phase16_hit = phase16_decoded.lower() in expanded_words
            if phase16_hit:
                n_phase16_hits += 1

            # Get syllabic triples
            syl_triples = _get_syllabic_triples(eva_text, eva_to_triple, modifier_chars)
            n_syl = len(syl_triples)

            # CSP: enumerate all assignments for short labels
            csp_attempted = n_syl <= MAX_TRIPLES and n_syl > 0
            n_tested = 0
            top_decodings: List[Dict] = []
            best_csp = ''
            best_csp_hit = False

            if csp_attempted:
                n_csp_attempted += 1
                triple_keys = [t[1] for t in syl_triples]

                # Get unique triples and their positions
                unique_triples = sorted(set(triple_keys))
                n_unique = len(unique_triples)

                # Enumerate all possible assignments of syllables to unique triples
                dict_hit_decodings = []

                for combo in itertools.product(all_syllables, repeat=n_unique):
                    n_tested += 1
                    # Build candidate assignment
                    candidate = dict(zip(unique_triples, combo))
                    # Decode: map each triple position to its syllable
                    decoded_syls = [candidate[tk] for tk in triple_keys]
                    decoded_word = ''.join(decoded_syls)

                    is_hit = decoded_word.lower() in expanded_words
                    if is_hit:
                        dict_hit_decodings.append({
                            'decoded': decoded_word,
                            'syllables': decoded_syls,
                            'triples': triple_keys,
                            'assignment': dict(candidate),
                            'is_dict_hit': True,
                        })

                # Sort by word length (prefer reasonable length) and take top-K
                dict_hit_decodings.sort(key=lambda x: abs(len(x['decoded']) - 5))
                top_decodings = dict_hit_decodings[:TOP_K]

                if dict_hit_decodings:
                    n_csp_with_hits += 1
                    best_csp = dict_hit_decodings[0]['decoded']
                    best_csp_hit = True

            lr = LabelResult(
                folio=folio,
                zodiac_sign=sign,
                locus_id=locus_id,
                eva_text=eva_text,
                n_syllabic_triples=n_syl,
                phase16_decode=phase16_decoded,
                phase16_is_dict_hit=phase16_hit,
                csp_attempted=csp_attempted,
                n_candidates_tested=n_tested,
                n_dict_hits=len(top_decodings),
                top_decodings=top_decodings,
                best_csp_decode=best_csp,
                best_csp_is_dict_hit=best_csp_hit,
            )
            all_label_results.append(lr)

    print(f"\n  2. Results:")
    print(f"      Total labels processed: {total_labels}")
    print(f"      Phase 16 dict hits:     {n_phase16_hits}/{total_labels} "
          f"({n_phase16_hits / max(total_labels, 1):.1%})")
    print(f"      CSP attempted:          {n_csp_attempted}")
    print(f"      CSP with dict hits:     {n_csp_with_hits}/{n_csp_attempted}")

    # Show some examples
    print(f"\n  3. Sample CSP dict-hit decodings:")
    shown = 0
    for lr in all_label_results:
        if lr.best_csp_is_dict_hit and shown < 15:
            print(f"      {lr.folio:8s} | {lr.eva_text:20s} → "
                  f"Phase16: {lr.phase16_decode:12s} | "
                  f"CSP: {lr.best_csp_decode:12s} "
                  f"(from {lr.n_dict_hits} hits)")
            shown += 1

    # -----------------------------------------------------------------------
    # Derive assignments from CSP dict hits
    # -----------------------------------------------------------------------
    print(f"\n  4. Deriving triple→syllable assignments from CSP hits ...")

    # Collect all triple → syllable assignments from CSP solutions
    triple_syl_evidence: Dict[str, Counter] = defaultdict(Counter)

    for lr in all_label_results:
        for dec in lr.top_decodings:
            asgn = dec.get('assignment', {})
            for triple, syl in asgn.items():
                triple_syl_evidence[triple][syl] += 1

    # Only keep assignments where one syllable dominates (≥ 2 supporting labels)
    derived_assignments: Dict[str, str] = {}
    for triple, syl_counts in triple_syl_evidence.items():
        if syl_counts:
            best_syl, best_count = syl_counts.most_common(1)[0]
            total = sum(syl_counts.values())
            if best_count >= 2 and best_count / total >= 0.5:
                derived_assignments[triple] = best_syl

    print(f"      Derived {len(derived_assignments)} assignments from CSP")
    for triple, syl in sorted(derived_assignments.items()):
        print(f"        {triple} → {syl}")

    # Compare with Phase 16
    n_agree = 0
    n_disagree = 0
    for triple, syl in derived_assignments.items():
        phase16_syl = assignment.get(triple, '')
        if syl == phase16_syl:
            n_agree += 1
        else:
            n_disagree += 1
            print(f"        DIFFERS: {triple} → CSP: {syl} vs Phase16: {phase16_syl}")

    agreement_rate = n_agree / max(n_agree + n_disagree, 1)

    print(f"\n  5. Phase 16 agreement:")
    print(f"      Agree: {n_agree}, Disagree: {n_disagree}, "
          f"Rate: {agreement_rate:.1%}")

    # Verdict
    phase16_rate = n_phase16_hits / max(total_labels, 1)
    csp_rate = n_csp_with_hits / max(n_csp_attempted, 1)
    improvement = csp_rate - phase16_rate

    if n_csp_with_hits > n_phase16_hits:
        verdict = (f"CSP IMPROVEMENT: {n_csp_with_hits} CSP hits vs "
                   f"{n_phase16_hits} Phase 16 hits. "
                   f"{len(derived_assignments)} derived assignments.")
    elif len(derived_assignments) > 0:
        verdict = (f"ASSIGNMENTS DERIVED: {len(derived_assignments)} triple→syllable "
                   f"pairs from CSP. Agreement with Phase 16: {agreement_rate:.0%}.")
    else:
        verdict = (f"MINIMAL: CSP produced {n_csp_with_hits} hits, "
                   f"Phase 16 produced {n_phase16_hits}. "
                   f"No strong derived assignments.")

    print(f"\n  6. Verdict: {verdict}")

    result = LabelDecodeResult(
        timestamp=time.strftime('%Y-%m-%dT%H:%M:%S'),
        n_labels=total_labels,
        n_csp_attempted=n_csp_attempted,
        n_csp_with_hits=n_csp_with_hits,
        n_phase16_hits=n_phase16_hits,
        phase16_hit_rate=round(phase16_rate, 4),
        n_csp_hits_total=sum(lr.n_dict_hits for lr in all_label_results),
        csp_hit_rate=round(csp_rate, 4),
        improvement_over_phase16=round(improvement, 4),
        label_results=[_convert(asdict(lr)) for lr in all_label_results],
        derived_assignments=derived_assignments,
        n_derived=len(derived_assignments),
        n_agree=n_agree,
        n_disagree=n_disagree,
        agreement_rate=round(agreement_rate, 4),
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'label_decode.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2)

    print(f"\n  → {out_path}")
