"""
Step 41.13 -- Botanical Data Fix
==================================
Fix upstream data format issues that caused Phase 40's Track D to produce
null results.  Inspect all botanical data sources, report their structure,
and build a unified folio -> plant mapping.

Dependency chain:
    italian_botanical_csp.json   (Step 39.9)
    consensus_plants.json        (Phase 31.1)
    italian_plant_names.json     (Step 39.8)
    medieval_latin_names.json    (data/reference/voynich_plant/)
    Voynich_Herbal_Multi-Source_Identification_Concordance.csv
        -> botanical_data_fix.json  (this step)
"""

import csv
import json
import os
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from voynich.core._paths import results_dir as _results_dir, data_dir as _data_dir
from voynich.core.corpus import build_eva_to_triple_lookup


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
# Data source inspection
# ---------------------------------------------------------------------------

def _inspect_data_source(name: str, data: Any) -> Dict:
    """Report the structure and content summary of a loaded data source."""
    report: Dict[str, Any] = {
        'name': name,
        'loaded': data is not None and data != {},
        'type': type(data).__name__,
    }

    if isinstance(data, dict):
        report['top_keys'] = list(data.keys())
        report['n_top_keys'] = len(data)
        # Summarise list-valued keys
        for k, v in data.items():
            if isinstance(v, list):
                report[f'{k}_count'] = len(v)
    elif isinstance(data, list):
        report['n_entries'] = len(data)
    else:
        report['value_preview'] = str(data)[:200]

    return report


def _load_concordance_csv(csv_path: str) -> List[Dict]:
    """Load the multi-source identification concordance CSV."""
    rows: List[Dict] = []
    if not os.path.exists(csv_path):
        return rows
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


# ---------------------------------------------------------------------------
# Unified mapping builder
# ---------------------------------------------------------------------------

def _build_unified_plant_map(
    italian_plant_names: Dict,
    consensus_plants: Dict,
    medieval_latin_names: Dict,
    concordance_rows: List[Dict],
) -> Dict[str, Dict]:
    """Build a unified mapping: folio -> {latin_name, italian_name, common_name,
    medieval_latin, syllabified, sources, tier}.

    Merges data from all four sources, preferring the most specific data.
    """
    unified: Dict[str, Dict] = {}

    # --- Source 1: Concordance CSV (broadest, lowest priority) ---
    for row in concordance_rows:
        folio = row.get('Folio', '').strip()
        if not folio:
            continue
        if folio not in unified:
            unified[folio] = {
                'latin_name': None,
                'italian_names': [],
                'common_name': None,
                'medieval_latin': None,
                'syllabified': {},
                'sources': [],
                'tier': None,
                'genus': None,
            }
        latin = row.get('Proposed Botanical Identification', '').strip()
        common = row.get('Common Name', '').strip()
        researcher = row.get('Principal Researcher / Source', '').strip()
        if latin and not unified[folio]['latin_name']:
            unified[folio]['latin_name'] = latin
        if common and not unified[folio]['common_name']:
            unified[folio]['common_name'] = common
        if researcher and researcher not in unified[folio]['sources']:
            unified[folio]['sources'].append(researcher)

    # --- Source 2: Medieval Latin names (keyed by Linnaean binomial) ---
    # Build a reverse lookup: binomial -> medieval data
    binomial_to_medieval: Dict[str, Dict] = {}
    for binomial, med_data in medieval_latin_names.items():
        binomial_to_medieval[binomial] = med_data

    # --- Source 3: Consensus plants (has tier, genus, medieval_names) ---
    # Merge tier_a and tier_b folios
    for tier_list_key in ('tier_a_folios', 'tier_b_folios'):
        for entry in consensus_plants.get(tier_list_key, []):
            folio = entry.get('folio', '')
            if not folio:
                continue
            if folio not in unified:
                unified[folio] = {
                    'latin_name': None,
                    'italian_names': [],
                    'common_name': None,
                    'medieval_latin': None,
                    'syllabified': {},
                    'sources': [],
                    'tier': None,
                    'genus': None,
                }

            consensus = entry.get('consensus', {})
            tier = consensus.get('tier', entry.get('tier', ''))
            genus = consensus.get('genus', entry.get('genus', ''))

            if tier:
                unified[folio]['tier'] = tier
            if genus:
                unified[folio]['genus'] = genus

            # Medieval names from consensus_plants
            for med in entry.get('medieval_names', []):
                med_name = med.get('medieval_name', '')
                if med_name and not unified[folio]['medieval_latin']:
                    unified[folio]['medieval_latin'] = med_name
                syl = med.get('syllabified', [])
                if syl and med_name:
                    unified[folio]['syllabified'][med_name] = syl
                linnaean = med.get('linnaean_name', '')
                if linnaean and not unified[folio]['latin_name']:
                    unified[folio]['latin_name'] = linnaean

    # Also scan all_folios for tier info
    for entry in consensus_plants.get('all_folios', []):
        folio = entry.get('folio', '')
        tier = entry.get('tier', '')
        genus = entry.get('genus', '')
        if folio and folio in unified:
            if tier and not unified[folio]['tier']:
                unified[folio]['tier'] = tier
            if genus and not unified[folio]['genus']:
                unified[folio]['genus'] = genus

    # --- Source 4: Italian plant names (highest priority for Italian names) ---
    for pentry in italian_plant_names.get('plant_name_table', []):
        folio = pentry.get('folio', '')
        if not folio:
            continue
        if folio not in unified:
            unified[folio] = {
                'latin_name': None,
                'italian_names': [],
                'common_name': None,
                'medieval_latin': None,
                'syllabified': {},
                'sources': [],
                'tier': None,
                'genus': None,
            }
        italian = pentry.get('italian_names', [])
        venetian = pentry.get('venetian_names', [])
        syl = pentry.get('syllabified', {})
        latin_bin = pentry.get('latin_binomial', '')
        med_lat = pentry.get('medieval_latin', None)

        # Merge Italian names (avoid duplicates)
        for name in italian + venetian:
            if name and name not in unified[folio]['italian_names']:
                unified[folio]['italian_names'].append(name)

        # Merge syllabifications
        for word, syls in syl.items():
            if word not in unified[folio]['syllabified']:
                unified[folio]['syllabified'][word] = syls

        if latin_bin and not unified[folio]['latin_name']:
            unified[folio]['latin_name'] = latin_bin
        if med_lat and not unified[folio]['medieval_latin']:
            unified[folio]['medieval_latin'] = med_lat

    # --- Cross-reference: try to fill medieval_latin from medieval_latin_names.json ---
    for folio, entry in unified.items():
        if entry['medieval_latin']:
            continue
        latin = entry.get('latin_name', '')
        if latin and latin in binomial_to_medieval:
            med = binomial_to_medieval[latin]
            entry['medieval_latin'] = med.get('medieval_name', '')
            for alt in med.get('alternate_names', []):
                if alt not in entry.get('italian_names', []):
                    pass  # These are Latin alternates, not Italian

    return unified


# ---------------------------------------------------------------------------
# Alignment constraint extraction
# ---------------------------------------------------------------------------

def _extract_alignment_triples(
    alignments: List[Dict],
    eva_to_triple: Dict[str, str],
) -> List[Dict]:
    """Extract triple constraints from italian_botanical_csp alignments.

    Each alignment has: folio, token, plant_name, triple_keys, syllables,
    assignments (triple_key -> syllable), score, etc.
    """
    results = []
    for aln in alignments:
        folio = aln.get('folio', '')
        plant = aln.get('plant_name', '')
        score = aln.get('score', 0.0)
        assignments = aln.get('assignments', {})
        triple_keys = aln.get('triple_keys', [])
        syllables = aln.get('syllables', [])

        if not assignments:
            continue

        for triple_key, syllable in assignments.items():
            results.append({
                'triple_key': triple_key,
                'syllable': syllable,
                'source_folio': folio,
                'source_plant': plant,
                'alignment_score': score,
            })

    return results


# ---------------------------------------------------------------------------
# Format issue detection
# ---------------------------------------------------------------------------

def _detect_format_issues(
    italian_csp: Dict,
    drosera_constraints: Dict,
    botanical_predictions: Dict,
    italian_plant_names: Dict,
) -> List[Dict]:
    """Detect format issues that caused Track D Phase 40 to produce null results."""
    issues: List[Dict] = []

    # Issue 1: drosera_constraints.json has verdict=NO_DATA because
    # it looked for folio_results/valid_alignments keys that don't exist
    # in italian_botanical_csp.json (which uses top-level 'alignments' key)
    if drosera_constraints.get('verdict') == 'NO_DATA':
        # Check what drosera_constraints was looking for vs what exists
        has_folio_results = 'folio_results' in italian_csp
        has_alignments = 'alignments' in italian_csp
        issues.append({
            'issue': 'drosera_constraints_no_data',
            'description': (
                'drosera_constraints.py looked for folio_results/valid_alignments '
                'but italian_botanical_csp.json uses top-level alignments key'
            ),
            'italian_csp_has_folio_results': has_folio_results,
            'italian_csp_has_alignments': has_alignments,
            'italian_csp_n_alignments': len(italian_csp.get('alignments', [])),
            'fix': 'Read alignments from top-level key instead of folio_results',
        })

    # Issue 2: botanical_predictions.json had 0 predictions because
    # plant_list building failed (looked for folio_plants/plants keys
    # but italian_plant_names uses plant_name_table)
    if botanical_predictions.get('n_predictions_total', 0) == 0:
        pnt = italian_plant_names.get('plant_name_table', [])
        n_with_italian = sum(
            1 for p in pnt
            if p.get('italian_names') or p.get('venetian_names')
        )
        issues.append({
            'issue': 'botanical_predictions_empty',
            'description': (
                'botanical_predictions.py looked for folio_plants/plants keys '
                'but italian_plant_names.json uses plant_name_table; also tried '
                'extracting italian_name from entries but the key is italian_names (list)'
            ),
            'plant_name_table_count': len(pnt),
            'entries_with_italian_names': n_with_italian,
            'fix': 'Use plant_name_table key and iterate italian_names list',
        })

    # Issue 3: The alignment assignments in italian_botanical_csp use
    # triple_key -> syllable mapping, but drosera_constraints looked for
    # triple_assignments/mapping keys
    for aln in italian_csp.get('alignments', []):
        if 'assignments' in aln and ('triple_assignments' not in aln
                                      and 'mapping' not in aln):
            issues.append({
                'issue': 'alignment_key_mismatch',
                'description': (
                    'italian_botanical_csp alignments use "assignments" key '
                    'for triple->syllable mapping, but drosera_constraints.py '
                    'looked for "triple_assignments" or "mapping"'
                ),
                'actual_key': 'assignments',
                'fix': 'Read from assignments key',
            })
            break

    if not issues:
        issues.append({
            'issue': 'none_detected',
            'description': 'No format issues detected - upstream data appears consistent',
        })

    return issues


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_botanical_data_fix() -> None:
    """Step 41.13: Fix botanical pipeline data formats."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 41.13: Botanical Data Fix")
    print("=" * 70)

    rd = _results_dir()
    dd = _data_dir()
    plant_data_dir = os.path.join(dd, 'reference', 'voynich_plant')

    # -- 1. Load all data sources --
    print("\n  1. Loading all botanical data sources ...")

    italian_csp_path = os.path.join(rd, 'italian_botanical_csp.json')
    italian_csp = _safe_load(italian_csp_path)
    if not italian_csp:
        print("     SKIP: italian_botanical_csp.json not found")

    consensus_path = os.path.join(rd, 'consensus_plants.json')
    consensus_plants = _safe_load(consensus_path)
    if not consensus_plants:
        print("     SKIP: consensus_plants.json not found")

    italian_names_path = os.path.join(rd, 'italian_plant_names.json')
    italian_plant_names = _safe_load(italian_names_path)
    if not italian_plant_names:
        print("     SKIP: italian_plant_names.json not found")

    medieval_path = os.path.join(plant_data_dir, 'medieval_latin_names.json')
    medieval_latin_names = _safe_load(medieval_path)
    if not medieval_latin_names:
        print("     SKIP: medieval_latin_names.json not found")

    csv_path = os.path.join(
        plant_data_dir,
        'Voynich_Herbal_Multi-Source_Identification_Concordance.csv',
    )
    concordance_rows = _load_concordance_csv(csv_path)
    if not concordance_rows:
        print("     SKIP: Concordance CSV not found")

    drosera_path = os.path.join(rd, 'drosera_constraints.json')
    drosera_constraints = _safe_load(drosera_path)

    bot_pred_path = os.path.join(rd, 'botanical_predictions.json')
    botanical_predictions = _safe_load(bot_pred_path)

    # -- 2. Inspect each source --
    print("\n  2. Inspecting data source structures ...")

    inspections = []
    for name, data in [
        ('italian_botanical_csp', italian_csp),
        ('consensus_plants', consensus_plants),
        ('italian_plant_names', italian_plant_names),
        ('medieval_latin_names', medieval_latin_names),
        ('concordance_csv', concordance_rows),
        ('drosera_constraints', drosera_constraints),
        ('botanical_predictions', botanical_predictions),
    ]:
        report = _inspect_data_source(name, data)
        inspections.append(report)
        loaded = report.get('loaded', False)
        print(f"     {name}: {'LOADED' if loaded else 'EMPTY/MISSING'}")
        if isinstance(data, dict) and data:
            print(f"       keys: {list(data.keys())[:8]}")
        elif isinstance(data, list):
            print(f"       {len(data)} entries")

    # -- 3. Detect format issues --
    print("\n  3. Detecting format issues ...")

    issues = _detect_format_issues(
        italian_csp, drosera_constraints,
        botanical_predictions, italian_plant_names,
    )
    for issue in issues:
        desc = issue.get('description', '')
        fix = issue.get('fix', 'N/A')
        print(f"     ISSUE: {issue['issue']}")
        print(f"       {desc}")
        print(f"       FIX: {fix}")

    # -- 4. Build unified plant mapping --
    print("\n  4. Building unified plant mapping ...")

    unified_map = _build_unified_plant_map(
        italian_plant_names, consensus_plants,
        medieval_latin_names, concordance_rows,
    )

    n_total = len(unified_map)
    n_with_italian = sum(
        1 for v in unified_map.values() if v.get('italian_names')
    )
    n_with_medieval = sum(
        1 for v in unified_map.values() if v.get('medieval_latin')
    )
    n_with_tier = sum(
        1 for v in unified_map.values()
        if v.get('tier') in ('A', 'B')
    )
    n_with_syllabified = sum(
        1 for v in unified_map.values() if v.get('syllabified')
    )

    print(f"     Total folios in unified map: {n_total}")
    print(f"     With Italian names: {n_with_italian}")
    print(f"     With medieval Latin: {n_with_medieval}")
    print(f"     With tier A/B: {n_with_tier}")
    print(f"     With syllabified forms: {n_with_syllabified}")

    # Show top entries
    for folio in sorted(unified_map.keys())[:5]:
        entry = unified_map[folio]
        it_names = entry.get('italian_names', [])
        tier = entry.get('tier', '?')
        print(f"     {folio} (Tier {tier}): "
              f"IT={it_names[:2] if it_names else 'none'} "
              f"med={entry.get('medieval_latin', 'none')}")

    # -- 5. Extract alignment constraints from italian_botanical_csp --
    print("\n  5. Extracting alignment constraints ...")

    eva_to_triple = build_eva_to_triple_lookup()
    alignments = italian_csp.get('alignments', [])
    alignment_constraints = _extract_alignment_triples(alignments, eva_to_triple)

    # Deduplicate by (triple_key, syllable)
    seen: Set[Tuple[str, str]] = set()
    unique_constraints: List[Dict] = []
    for c in alignment_constraints:
        key = (c['triple_key'], c['syllable'])
        if key not in seen:
            seen.add(key)
            unique_constraints.append(c)

    print(f"     Total alignment constraints: {len(alignment_constraints)}")
    print(f"     Unique (triple, syllable) pairs: {len(unique_constraints)}")
    for c in unique_constraints[:5]:
        print(f"       {c['triple_key']} -> '{c['syllable']}' "
              f"(from {c['source_folio']}/{c['source_plant']}, "
              f"score={c['alignment_score']})")

    # -- 6. Report which triples are constrained --
    print("\n  6. Triple coverage from alignments ...")

    constrained_triples: Dict[str, List[str]] = defaultdict(list)
    for c in unique_constraints:
        constrained_triples[c['triple_key']].append(c['syllable'])

    n_constrained = len(constrained_triples)
    print(f"     Triples constrained by alignments: {n_constrained}/25")
    for tk, syls in sorted(constrained_triples.items()):
        print(f"       {tk} -> {syls}")

    # -- 7. Save --
    elapsed = time.time() - t0

    output = {
        'data_source_inspections': inspections,
        'format_issues': issues,
        'n_issues_detected': sum(
            1 for i in issues if i['issue'] != 'none_detected'
        ),
        'unified_plant_map': unified_map,
        'n_folios_unified': n_total,
        'n_with_italian_names': n_with_italian,
        'n_with_medieval_latin': n_with_medieval,
        'n_with_tier_ab': n_with_tier,
        'n_with_syllabified': n_with_syllabified,
        'alignment_constraints': unique_constraints,
        'n_alignment_constraints': len(unique_constraints),
        'constrained_triples': {
            k: v for k, v in sorted(constrained_triples.items())
        },
        'n_constrained_triples': n_constrained,
        'runtime_seconds': round(elapsed, 1),
    }

    out_path = os.path.join(rd, 'botanical_data_fix.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(output), f, indent=2)
    print(f"\n  Saved -> {out_path} ({elapsed:.1f}s)")
