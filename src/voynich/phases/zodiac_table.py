"""
Step 26.5 – Zodiac-Derived Assignment Table
============================================
Assemble all character assignments derived from the zodiac analysis
(month cribs, astrological cribs, label CSP) into a tiered table and
merge with Phase 16.

Dependency chain:
    month_crib.json (Step 26.2)
    astro_crib.json (Step 26.3)
    label_decode.json (Step 26.4)
    combined_refine.json (Phase 15)
    modifier_integrate.json (Phase 16)
        → zodiac_table.json
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
class TieredAssignment:
    triple_key: str
    syllable: str
    tier: str        # tier1_confirmed, tier2_crib, tier3_phase16
    source: str      # month_crib, label_csp, astro_crib, phase16
    n_evidence: int
    confidence: float


@dataclass
class ZodiacTableResult:
    timestamp: str
    n_tier1: int
    n_tier2: int
    n_tier3: int
    n_total: int
    tiered_assignments: List[Dict]
    merged_assignment: Dict[str, str]
    # Changes from Phase 16
    n_changed: int
    changed_triples: List[Dict]
    # Quick decode test
    zodiac_dict_hit: float
    zodiac_n_tokens: int
    phase16_zodiac_dict_hit: float
    # Corpus sample test
    sample_dict_hit: float
    phase16_sample_dict_hit: float
    # Verdict
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_zodiac_table() -> None:
    t0 = time.time()
    print("=" * 70)
    print("STEP 26.5: Zodiac-Derived Assignment Table")
    print("=" * 70)

    rd = _results_dir()

    # Load all dependencies
    month_data = _load_json(os.path.join(rd, 'month_crib.json'))
    astro_data = _load_json(os.path.join(rd, 'astro_crib.json'))
    label_data = _load_json(os.path.join(rd, 'label_decode.json'))
    refine_data = _load_json(os.path.join(rd, 'combined_refine.json'))
    mod_data = _load_json(os.path.join(rd, 'modifier_integrate.json'))
    zodiac_data = _load_json(os.path.join(rd, 'zodiac_map.json'))

    if not refine_data:
        print("  [SKIP] combined_refine.json not found")
        return
    if not mod_data:
        print("  [SKIP] modifier_integrate.json not found")
        return

    phase16_assignment = refine_data.get('best_assignment', {})
    modifier_chars = set(mod_data.get('modifier_chars', []))

    corpus = load_corpus(verbose=False)
    eva_to_triple = build_eva_to_triple_lookup()

    # Build expanded word set
    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    base_words = set()
    for text in ref_corpus.get_texts('latin'):
        base_words.update(w.lower() for w in text.tokens if len(w) >= 2)
    expanded_words, _ = build_expanded_word_set(base_words)

    # -------------------------------------------------------------------
    # Collect assignments from all zodiac steps
    # -------------------------------------------------------------------
    print(f"\n  1. Collecting zodiac-derived assignments ...")

    # Evidence: triple -> [(syllable, source, weight)]
    # Sources with cross-folio validation get high weight (tier1 eligible).
    # Individual CSP solutions without cross-folio check are tier2 at best.
    evidence: Dict[str, List[Tuple[str, str, float]]] = defaultdict(list)

    # Track which triples have cross-folio confirmed evidence
    confirmed_triples: Set[str] = set()

    # Tier 1 candidates: from month_crib consistent_assignments (≥2 folios)
    if month_data:
        consistent = month_data.get('consistent_assignments', {})
        for triple, syl in consistent.items():
            evidence[triple].append((syl, 'month_crib_consistent', 5.0))
            confirmed_triples.add(triple)
        print(f"      Month crib consistent: {len(consistent)} assignments")

        # Collect from CSP solutions: deduplicate per (triple, syl, folio)
        # to avoid inflating weight from many solutions on the same folio.
        csp_seen: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
        for sol in month_data.get('csp_solutions', []):
            folio = sol.get('folio', '')
            asgn = sol.get('assignment', {})
            for triple, syl in asgn.items():
                if triple not in consistent:
                    csp_seen[(triple, syl)].add(folio)

        for (triple, syl), folios in csp_seen.items():
            # Weight by number of distinct folios, capped at 1.0
            weight = min(len(folios) * 0.3, 1.0)
            evidence[triple].append((syl, 'month_crib_csp', weight))
        print(f"      Month crib CSP: {len(csp_seen)} unique (triple, syl) pairs")

    # From label_decode derived assignments (cross-label consistent)
    if label_data:
        derived = label_data.get('derived_assignments', {})
        for triple, syl in derived.items():
            evidence[triple].append((syl, 'label_csp', 2.0))
        print(f"      Label CSP derived: {len(derived)} assignments")

    # From astro_crib (lower confidence — broad vocabulary matching)
    # We don't extract per-triple assignments from astro_crib as it works
    # at the decoded-word level, not at the triple level.
    if astro_data:
        print(f"      Astro crib: provides domain-level evidence only")

    # -------------------------------------------------------------------
    # Build tiered table
    # -------------------------------------------------------------------
    print(f"\n  2. Building tiered assignment table ...")

    tiered: List[TieredAssignment] = []
    merged: Dict[str, str] = {}

    for triple in sorted(phase16_assignment.keys()):
        ev_list = evidence.get(triple, [])

        if ev_list:
            # Aggregate by syllable
            syl_weight: Dict[str, float] = defaultdict(float)
            syl_sources: Dict[str, List[str]] = defaultdict(list)
            for syl, src, weight in ev_list:
                syl_weight[syl] += weight
                syl_sources[syl].append(src)

            best_syl = max(syl_weight, key=syl_weight.get)
            best_weight = syl_weight[best_syl]
            sources = syl_sources[best_syl]

            # Tier 1 requires cross-folio confirmed evidence
            has_confirmed = any(
                s == 'month_crib_consistent' for s in sources
            )
            if has_confirmed and triple in confirmed_triples:
                tier = 'tier1_confirmed'
            elif best_weight >= 2.0:
                tier = 'tier2_crib'
            else:
                # Low-weight evidence: don't override Phase 16
                tier = 'tier3_phase16'
                best_syl = phase16_assignment[triple]

            confidence = min(best_weight / 5.0, 1.0)
            n_ev = len(sources)
            source = sources[0] if sources else 'phase16'

            tiered.append(TieredAssignment(
                triple_key=triple,
                syllable=best_syl,
                tier=tier,
                source=source,
                n_evidence=n_ev,
                confidence=round(confidence, 3),
            ))
            merged[triple] = best_syl
        else:
            # No zodiac evidence — keep Phase 16
            tiered.append(TieredAssignment(
                triple_key=triple,
                syllable=phase16_assignment[triple],
                tier='tier3_phase16',
                source='phase16',
                n_evidence=0,
                confidence=0.0,
            ))
            merged[triple] = phase16_assignment[triple]

    n_t1 = sum(1 for t in tiered if t.tier == 'tier1_confirmed')
    n_t2 = sum(1 for t in tiered if t.tier == 'tier2_crib')
    n_t3 = sum(1 for t in tiered if t.tier == 'tier3_phase16')

    print(f"      Tier 1 (confirmed): {n_t1}")
    print(f"      Tier 2 (crib):      {n_t2}")
    print(f"      Tier 3 (Phase 16):  {n_t3}")
    print(f"      Total:              {len(tiered)}")

    # Identify changes from Phase 16
    changed: List[Dict] = []
    for ta in tiered:
        p16_syl = phase16_assignment.get(ta.triple_key, '')
        if ta.syllable != p16_syl and ta.tier != 'tier3_phase16':
            changed.append({
                'triple': ta.triple_key,
                'phase16_syllable': p16_syl,
                'zodiac_syllable': ta.syllable,
                'tier': ta.tier,
                'source': ta.source,
            })

    print(f"\n  3. Changes from Phase 16: {len(changed)}")
    for ch in changed:
        print(f"      {ch['triple']}: {ch['phase16_syllable']} → "
              f"{ch['zodiac_syllable']} ({ch['tier']})")

    # -------------------------------------------------------------------
    # Quick decode test on zodiac folios
    # -------------------------------------------------------------------
    print(f"\n  4. Quick decode test on zodiac folios ...")

    if zodiac_data:
        folio_map = zodiac_data.get('folio_map', [])
    else:
        folio_map = []

    zodiac_tokens: List[str] = []
    for finfo in folio_map:
        page = corpus.get_page(finfo['folio'])
        if page:
            zodiac_tokens.extend(page.all_tokens)

    if zodiac_tokens:
        # Decode with merged table
        merged_hits = 0
        for token in zodiac_tokens:
            dec = decode_token_modifier_aware(
                token, merged, eva_to_triple, modifier_chars
            )
            if dec.lower() in expanded_words:
                merged_hits += 1
        zodiac_hit = merged_hits / len(zodiac_tokens)

        # Decode with Phase 16 table
        p16_hits = 0
        for token in zodiac_tokens:
            dec = decode_token_modifier_aware(
                token, phase16_assignment, eva_to_triple, modifier_chars
            )
            if dec.lower() in expanded_words:
                p16_hits += 1
        p16_zodiac_hit = p16_hits / len(zodiac_tokens)
    else:
        zodiac_hit = 0.0
        p16_zodiac_hit = 0.0

    print(f"      Zodiac tokens: {len(zodiac_tokens)}")
    print(f"      Merged table:  {zodiac_hit:.1%}")
    print(f"      Phase 16:      {p16_zodiac_hit:.1%}")

    # Also test on a corpus sample
    print(f"\n  5. Corpus sample decode test ...")
    all_tokens = corpus.get_tokens()
    sample = all_tokens[:5000]

    merged_sample_hits = 0
    p16_sample_hits = 0
    for token in sample:
        dec_m = decode_token_modifier_aware(
            token, merged, eva_to_triple, modifier_chars
        )
        if dec_m.lower() in expanded_words:
            merged_sample_hits += 1

        dec_p = decode_token_modifier_aware(
            token, phase16_assignment, eva_to_triple, modifier_chars
        )
        if dec_p.lower() in expanded_words:
            p16_sample_hits += 1

    sample_hit = merged_sample_hits / len(sample) if sample else 0
    p16_sample_hit = p16_sample_hits / len(sample) if sample else 0

    print(f"      Sample size:   {len(sample)}")
    print(f"      Merged table:  {sample_hit:.1%}")
    print(f"      Phase 16:      {p16_sample_hit:.1%}")

    # Verdict
    if n_t1 >= 3 and zodiac_hit > p16_zodiac_hit:
        verdict = (f"IMPROVED: {n_t1} tier-1 assignments. "
                   f"Zodiac: {zodiac_hit:.1%} vs Phase16 {p16_zodiac_hit:.1%}. "
                   f"Corpus: {sample_hit:.1%} vs {p16_sample_hit:.1%}.")
    elif n_t1 + n_t2 >= 3:
        verdict = (f"DERIVED: {n_t1} tier-1 + {n_t2} tier-2 assignments. "
                   f"Zodiac: {zodiac_hit:.1%}, Corpus: {sample_hit:.1%}.")
    elif len(changed) > 0:
        verdict = (f"MARGINAL: {len(changed)} changes from Phase 16. "
                   f"No clear improvement.")
    else:
        verdict = "NO CHANGE: Zodiac analysis produced no new assignments."

    print(f"\n  6. Verdict: {verdict}")

    result = ZodiacTableResult(
        timestamp=time.strftime('%Y-%m-%dT%H:%M:%S'),
        n_tier1=n_t1,
        n_tier2=n_t2,
        n_tier3=n_t3,
        n_total=len(tiered),
        tiered_assignments=[_convert(asdict(ta)) for ta in tiered],
        merged_assignment=merged,
        n_changed=len(changed),
        changed_triples=changed,
        zodiac_dict_hit=round(zodiac_hit, 4),
        zodiac_n_tokens=len(zodiac_tokens),
        phase16_zodiac_dict_hit=round(p16_zodiac_hit, 4),
        sample_dict_hit=round(sample_hit, 4),
        phase16_sample_dict_hit=round(p16_sample_hit, 4),
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'zodiac_table.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2)

    print(f"\n  → {out_path}")
