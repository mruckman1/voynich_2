"""
Phase 28.6 – Ventris Table Assembly
======================================
Assembles the final confidence-tiered assignment table by combining:
  - Phase 15 baseline (combined_refine)
  - Step 28.1 crib confirmations (tier 1+2)
  - Step 28.2 family consistency annotations
  - Step 28.3 propagation corrections (if any improve dict_hit)
  - Step 28.4 signal filtering (downgrade ARTIFACT-only triples)

Each of the 25 triples gets a confidence tier and full provenance chain.

Dependency chain:
    crib_extraction.json      (Step 28.1)
    crib_consistency.json     (Step 28.2)
    family_propagation.json   (Step 28.3)
    signal_isolation.json     (Step 28.4)
    combined_refine.json      (Phase 15)
        → ventris_table.json  (this step)
"""

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.reference import EVA_VISUAL_COMPONENTS


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
class VentrisTableEntry:
    triple: str
    eva_glyphs: List[str]
    syllable: str
    tier: int
    evidence_sources: List[str]
    is_family_consistent: bool
    crib_words_confirming: List[str]
    was_corrected: bool
    correction_source: str


@dataclass
class VentrisTableResult:
    n_tier1: int
    n_tier2: int
    n_tier3: int
    assignments: List[Dict]
    merged_assignment: Dict[str, str]
    n_changed_vs_phase16: int
    changed_triples: List[Dict]
    scientific_note: str
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_ventris_table() -> None:
    """Step 28.6: Assemble confidence-tiered Ventris table."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE 28.6: Ventris Table Assembly")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load all inputs ──
    print("\n  1. Loading inputs …")

    # Phase 15 baseline
    refine_path = os.path.join(rd, 'combined_refine.json')
    if not os.path.exists(refine_path):
        print("  [SKIP] combined_refine.json not found")
        return
    with open(refine_path) as f:
        refine_data = json.load(f)
    baseline = refine_data.get('best_assignment', {})

    # Crib extraction
    crib_path = os.path.join(rd, 'crib_extraction.json')
    crib_data = {}
    if os.path.exists(crib_path):
        with open(crib_path) as f:
            crib_data = json.load(f)

    confirmed_triples = set(crib_data.get('all_triples_covered', []))

    # Build triple → confirming crib words
    triple_to_cribs: Dict[str, List[str]] = {}
    triple_to_sources: Dict[str, Set[str]] = {}
    for crib in crib_data.get('cribs', []):
        word = crib.get('word', '')
        sources = crib.get('sources', [])
        for alignment in crib.get('alignments', []):
            triple = alignment.get('triple_key', '')
            if triple:
                triple_to_cribs.setdefault(triple, []).append(word)
                triple_to_sources.setdefault(triple, set()).update(sources)

    # Consistency
    consist_path = os.path.join(rd, 'crib_consistency.json')
    inconsistent_set: Set[str] = set()
    if os.path.exists(consist_path):
        with open(consist_path) as f:
            consist_data = json.load(f)
        for entry in consist_data.get('inconsistent_triples', []):
            inconsistent_set.add(entry.get('triple', ''))

    # Family propagation
    prop_path = os.path.join(rd, 'family_propagation.json')
    corrections: Dict[str, str] = {}
    if os.path.exists(prop_path):
        with open(prop_path) as f:
            prop_data = json.load(f)
        for entry in prop_data.get('propagation_entries', []):
            if entry.get('recommendation') == 'correct':
                triple = entry.get('triple', '')
                new_syl = entry.get('best_alternative', '')
                delta = entry.get('dict_hit_delta', 0.0)
                if triple and new_syl and delta > 0:
                    corrections[triple] = new_syl

    # Signal isolation — check for artifact-only cribs
    signal_path = os.path.join(rd, 'signal_isolation.json')
    artifact_words: Set[str] = set()
    genuine_words: Set[str] = set()
    if os.path.exists(signal_path):
        with open(signal_path) as f:
            signal_data = json.load(f)
        for ws in signal_data.get('word_signals', []):
            if ws.get('is_genuine_signal'):
                genuine_words.add(ws['word'])
            else:
                artifact_words.add(ws['word'])

    print(f"     Baseline: {len(baseline)} triples")
    print(f"     Confirmed: {len(confirmed_triples)} triples")
    print(f"     Corrections recommended: {len(corrections)}")
    print(f"     Genuine signal words: {len(genuine_words)}")
    print(f"     Artifact words: {len(artifact_words)}")

    # ── 2. Build EVA glyph → triple reverse map ──
    triple_to_glyphs: Dict[str, List[str]] = {}
    for glyph, components in EVA_VISUAL_COMPONENTS.items():
        tk = (components['first_stroke'] + ',' +
              components['last_stroke'] + ',' +
              components['glyph_class'])
        triple_to_glyphs.setdefault(tk, []).append(glyph)

    # ── 3. Assemble tiered table ──
    print("\n  2. Assembling tiered table …")
    entries: List[VentrisTableEntry] = []
    merged: Dict[str, str] = {}
    changed: List[Dict] = []

    for triple in sorted(baseline.keys()):
        current_syl = baseline[triple]
        glyphs = sorted(triple_to_glyphs.get(triple, []))
        confirming_cribs = triple_to_cribs.get(triple, [])
        sources = triple_to_sources.get(triple, set())
        is_consistent = triple not in inconsistent_set

        # Determine tier
        # Tier 1: cross-source confirmed (Phase 14 + Phase 19.x or Phase 26)
        n_independent_pipelines = len({
            'phase14' if 'phase14' in sources else None,
            'phase19' if any(s.startswith('phase19') for s in sources) else None,
            'phase26' if any(s.startswith('phase26') for s in sources) else None,
        } - {None})

        if n_independent_pipelines >= 2 and is_consistent:
            tier = 1
        elif triple in confirmed_triples and is_consistent:
            # Check if all confirming cribs are artifacts
            real_cribs = [w for w in confirming_cribs
                          if w in genuine_words or w not in artifact_words]
            if real_cribs:
                tier = 2
            else:
                tier = 3  # downgrade: confirmed only by artifact words
        else:
            tier = 3

        # Apply correction if recommended
        was_corrected = False
        correction_source = ''
        final_syl = current_syl
        if triple in corrections:
            final_syl = corrections[triple]
            was_corrected = True
            correction_source = 'family_propagation'
            changed.append({
                'triple': triple,
                'old_syllable': current_syl,
                'new_syllable': final_syl,
                'source': 'family_propagation',
            })

        merged[triple] = final_syl

        entries.append(VentrisTableEntry(
            triple=triple,
            eva_glyphs=glyphs,
            syllable=final_syl,
            tier=tier,
            evidence_sources=sorted(sources),
            is_family_consistent=is_consistent,
            crib_words_confirming=sorted(set(confirming_cribs)),
            was_corrected=was_corrected,
            correction_source=correction_source,
        ))

    n_tier1 = sum(1 for e in entries if e.tier == 1)
    n_tier2 = sum(1 for e in entries if e.tier == 2)
    n_tier3 = sum(1 for e in entries if e.tier == 3)

    # ── 4. Report ──
    print(f"\n  Tier breakdown:")
    print(f"    Tier 1 (cross-source): {n_tier1}")
    print(f"    Tier 2 (Phase 14 confirmed): {n_tier2}")
    print(f"    Tier 3 (unconfirmed): {n_tier3}")
    print(f"    Corrections applied: {len(changed)}")

    for e in entries:
        tag = {1: '★', 2: '●', 3: '○'}[e.tier]
        corr = ' [CORRECTED]' if e.was_corrected else ''
        print(f"    {tag} {e.triple:40s} = '{e.syllable}' "
              f"tier={e.tier}{corr}")

    scientific_note = (
        f"Cross-source crib pool = {n_tier1} triples from "
        f"{len(genuine_words)} genuine-signal words. This is insufficient "
        f"to derive new triple assignments via Ventris propagation. "
        f"Phase 28's contribution is a confidence-tiered table "
        f"({n_tier1} tier-1, {n_tier2} tier-2, {n_tier3} tier-3 triples) "
        f"and {len(changed)} correction(s), not new phoneme assignments."
    )

    verdict = (
        f"TABLE_CORRECTED: {len(changed)} triple(s) corrected, "
        f"{n_tier1}+{n_tier2}+{n_tier3} tier distribution"
        if changed
        else f"TABLE_TIERED: {n_tier1}+{n_tier2}+{n_tier3} tier distribution, "
             f"0 corrections (Phase 15 table unchanged)"
    )
    print(f"\n  {verdict}")

    # ── 5. Save ──
    result = VentrisTableResult(
        n_tier1=n_tier1,
        n_tier2=n_tier2,
        n_tier3=n_tier3,
        assignments=[_convert(asdict(e)) for e in entries],
        merged_assignment=merged,
        n_changed_vs_phase16=len(changed),
        changed_triples=changed,
        scientific_note=scientific_note,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'ventris_table.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)
    print(f"\n  → {out_path}")
