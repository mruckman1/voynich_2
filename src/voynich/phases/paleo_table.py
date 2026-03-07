"""
Phase 21.7 – Paleographic Table Assembly (paleo-table)
======================================================
Combines all evidence into a single decoding table with evidence provenance
and quality metrics.

Dependency chain:
    family_to_syllable.json (21.5) + cappelli_modifier.json (21.6)
    + combined_refine.json (Phase 15) + phase20_integrate.json (Phase 20)
        → paleo_table.json (this step)
"""

import json
import math
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

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


def _load_json(path: str) -> Optional[Dict]:
    import os
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _jsd(p: Dict[str, float], q: Dict[str, float]) -> float:
    """Jensen-Shannon divergence between two distributions."""
    all_keys = set(p.keys()) | set(q.keys())
    if not all_keys:
        return 0.0
    # Build uniform distributions over all keys
    pp = {k: p.get(k, 0.0) for k in all_keys}
    qq = {k: q.get(k, 0.0) for k in all_keys}
    # Normalize
    sp = sum(pp.values()) or 1.0
    sq = sum(qq.values()) or 1.0
    pp = {k: v / sp for k, v in pp.items()}
    qq = {k: v / sq for k, v in qq.items()}
    m = {k: (pp[k] + qq[k]) / 2 for k in all_keys}
    jsd = 0.0
    for k in all_keys:
        if pp[k] > 0 and m[k] > 0:
            jsd += pp[k] * math.log2(pp[k] / m[k])
        if qq[k] > 0 and m[k] > 0:
            jsd += qq[k] * math.log2(qq[k] / m[k])
    return jsd / 2


# ---------------------------------------------------------------------------
# Latin syllable frequency reference
# ---------------------------------------------------------------------------

# Approximate Latin syllable frequencies (top 30)
_LATIN_SYLLABLE_FREQ: Dict[str, float] = {
    'de': 0.06, 'in': 0.05, 'et': 0.04, 'ad': 0.03, 'te': 0.03,
    'ti': 0.03, 'ta': 0.025, 'ne': 0.025, 'ni': 0.02, 'na': 0.02,
    'se': 0.02, 'si': 0.02, 'sa': 0.02, 're': 0.02, 'ri': 0.02,
    'ra': 0.02, 'di': 0.02, 'da': 0.015, 'be': 0.015, 'bi': 0.01,
    'ba': 0.01, 'me': 0.015, 'mi': 0.01, 'ma': 0.01, 'le': 0.01,
    'li': 0.01, 'la': 0.01, 'pe': 0.01, 'pi': 0.01, 'pa': 0.01,
}


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TableEntry:
    eva_char: str
    latin_syllable: Optional[str]
    is_modifier: bool
    modifier_function: Optional[str]
    evidence_priority: int
    evidence_sources: List[str]
    confidence: str


@dataclass
class PaleoTableResult:
    timestamp: str
    table_size: int
    coverage_total: int
    coverage_high_conf: int
    coverage_priority_1_3: int
    n_modifiers: int
    n_homophones: int
    frequency_jsd: float
    edit_distance_phase20: float
    edit_distance_phase15: float
    table: List[Dict[str, Any]]
    quality_metrics: Dict[str, Any]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_paleo_table() -> Dict[str, Any]:
    """Assemble paleographic decoding table."""
    t0 = time.time()
    rdir = _results_dir()

    # --- Load upstream ---
    fts = _load_json(str(rdir / "family_to_syllable.json")) or {}
    cap_mod = _load_json(str(rdir / "cappelli_modifier.json")) or {}
    p20 = _load_json(str(rdir / "phase20_integrate.json")) or {}
    p15 = _load_json(str(rdir / "combined_refine.json")) or {}

    # Build assignment lookup from 21.5
    assign_by_char: Dict[str, Dict] = {}
    for a in fts.get('assignments', []):
        assign_by_char[a.get('eva_char', '')] = a

    # Build modifier lookup from 21.6
    modifier_chars_set: set = set()
    modifier_functions: Dict[str, str] = {}
    for fa in cap_mod.get('functional_assignments', []):
        mc = fa.get('modifier_char', '')
        modifier_chars_set.add(mc)
        modifier_functions[mc] = fa.get('predicted_cappelli_function', 'unknown')

    # Fallback: Phase 16 modifier list
    mod_int = _load_json(str(rdir / "modifier_integrate.json")) or {}
    for key in ['modifier_chars', 'confirmed_modifiers', 'modifiers']:
        mods = mod_int.get(key, [])
        if mods:
            for m in mods:
                if isinstance(m, str):
                    modifier_chars_set.add(m)
                elif isinstance(m, dict):
                    mc = m.get('eva_char', m.get('char', ''))
                    if mc:
                        modifier_chars_set.add(mc)
            break

    # Phase 20 table for comparison
    p20_table: Dict[str, str] = {}
    for entry in p20.get('decoding_table', p20.get('table', [])):
        if isinstance(entry, dict):
            ec = entry.get('eva_char', '')
            ls = entry.get('latin_syllable', entry.get('value', ''))
            if ec and ls:
                p20_table[ec] = ls

    # Phase 15 triple assignments for comparison
    p15_table: Dict[str, str] = {}
    for ta in p15.get('best_assignments', p15.get('assignments', [])):
        if isinstance(ta, dict):
            tk = ta.get('triple_key', '')
            syl = ta.get('syllable', ta.get('value', ''))
            if tk and syl:
                # Map triple_key back to EVA chars
                for ec, comp in EVA_VISUAL_COMPONENTS.items():
                    etk = f"{comp['first_stroke']},{comp['last_stroke']},{comp['glyph_class']}"
                    if etk == tk:
                        p15_table[ec] = syl

    # --- Build table ---
    table: List[TableEntry] = []

    for eva_char in EVA_VISUAL_COMPONENTS:
        assignment = assign_by_char.get(eva_char, {})
        is_mod = eva_char in modifier_chars_set
        mod_func = modifier_functions.get(eva_char) if is_mod else None

        latin_syl = assignment.get('latin_syllable')
        priority = assignment.get('priority', 6)
        evidence = assignment.get('evidence', [])
        confidence = assignment.get('confidence', 'unassigned')

        table.append(TableEntry(
            eva_char=eva_char,
            latin_syllable=latin_syl if not is_mod else None,
            is_modifier=is_mod,
            modifier_function=mod_func,
            evidence_priority=priority,
            evidence_sources=evidence,
            confidence=confidence,
        ))

    # --- Quality metrics ---
    total_assigned = sum(1 for t in table if t.latin_syllable or t.is_modifier)
    high_conf = sum(1 for t in table if t.confidence == 'high')
    priority_1_3 = sum(1 for t in table if t.evidence_priority <= 3)
    n_mods = sum(1 for t in table if t.is_modifier)

    # Homophone count: syllable values used more than once
    syl_counts = Counter(t.latin_syllable for t in table if t.latin_syllable)
    n_homophones = sum(1 for c in syl_counts.values() if c > 1)

    # Frequency JSD
    syl_freq: Dict[str, float] = {}
    total_syl = sum(syl_counts.values()) or 1
    for s, c in syl_counts.items():
        syl_freq[s] = c / total_syl
    freq_jsd = _jsd(syl_freq, _LATIN_SYLLABLE_FREQ)

    # Edit distance to Phase 20 and Phase 15 tables
    def _table_edit_distance(t1: Dict[str, str], t2: Dict[str, str]) -> float:
        all_chars = set(t1.keys()) | set(t2.keys())
        if not all_chars:
            return 0.0
        matches = sum(1 for c in all_chars if t1.get(c) == t2.get(c) and t1.get(c))
        return 1.0 - matches / len(all_chars)

    my_table = {t.eva_char: t.latin_syllable for t in table if t.latin_syllable}
    ed_p20 = _table_edit_distance(my_table, p20_table)
    ed_p15 = _table_edit_distance(my_table, p15_table)

    result = PaleoTableResult(
        timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
        table_size=len(table),
        coverage_total=total_assigned,
        coverage_high_conf=high_conf,
        coverage_priority_1_3=priority_1_3,
        n_modifiers=n_mods,
        n_homophones=n_homophones,
        frequency_jsd=freq_jsd,
        edit_distance_phase20=ed_p20,
        edit_distance_phase15=ed_p15,
        table=[_convert(asdict(t)) for t in table],
        quality_metrics={
            'coverage_fraction': total_assigned / max(len(table), 1),
            'high_conf_fraction': high_conf / max(len(table), 1),
            'priority_1_3_fraction': priority_1_3 / max(len(table), 1),
            'homophone_fraction': n_homophones / max(len(syl_counts), 1),
        },
    )

    out_path = rdir / "paleo_table.json"
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t0
    print(f"paleo-table: {total_assigned}/{len(table)} assigned, "
          f"P1-3={priority_1_3}, homophones={n_homophones}, "
          f"JSD={freq_jsd:.3f} ({elapsed:.1f}s)")

    return _convert(asdict(result))
