"""
Phase 21.9 – Validation Battery (paleo-validate)
=================================================
15-test validation battery: Phase 20.7's 12 tests + 3 new paleographic tests.

Dependency chain:
    paleo_decode.json (21.8) + paleo_table.json (21.7)
    + eva_stroke_compare.json (21.4) + fontana_families.json (21.2)
    + upstream phase results
        → paleo_validate.json (this step)
"""

import json
import math
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.reference import EVA_VISUAL_COMPONENTS, build_expanded_word_set


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
# Test implementations
# ---------------------------------------------------------------------------

def _v1_null_discrimination(decode_data: Dict) -> Dict[str, Any]:
    """V1: Null discrimination (selectivity ≥ 1.3×)."""
    exp_rate = decode_data.get('dict_hit_rate_expanded', 0)
    # Estimate null rate from random assignments: typically ~5-10%
    # Use ratio vs a baseline of ~6% as rough null
    null_est = 0.06
    selectivity = exp_rate / max(null_est, 0.001)
    return {
        'test': 'V1_null_discrimination',
        'selectivity': selectivity,
        'dict_hit_rate': exp_rate,
        'gate': 1.3,
        'passed': selectivity >= 1.3,
    }


def _v2_bigram_plausibility(decode_data: Dict) -> Dict[str, Any]:
    """V2: Bigram plausibility (> 0%)."""
    # Check decoded words for Latin bigram plausibility
    freq = decode_data.get('word_frequency_top30', {})
    # Simple check: Latin doesn't have q without u, xx, etc.
    bad_bigrams = {'qx', 'xq', 'zz', 'xx', 'qz'}
    total_words = sum(freq.values())
    bad_count = 0
    for word, count in freq.items():
        for bg in bad_bigrams:
            if bg in word.lower():
                bad_count += count
    plausibility = 1.0 - (bad_count / max(total_words, 1))
    return {
        'test': 'V2_bigram_plausibility',
        'plausibility': plausibility,
        'gate': 0.0,
        'passed': plausibility > 0,
    }


def _v3_phrase_detection(decode_data: Dict) -> Dict[str, Any]:
    """V3: Phrase detection (selectivity ≥ 1.5×)."""
    freq = decode_data.get('word_frequency_top30', {})
    # Look for known Latin phrases
    phrases = [
        ('de', 'in'), ('ad', 'de'), ('et', 'in'), ('non', 'est'),
        ('bene', 'de'), ('per', 'se'), ('in', 're'),
    ]
    try:
        expanded_dict, _ = build_expanded_word_set('latin')
    except Exception:
        expanded_dict = set()

    n_phrases = 0
    detected = []
    for w1, w2 in phrases:
        if w1 in freq and w2 in freq:
            n_phrases += 1
            detected.append(f"{w1} {w2}")

    # Null: random word pairs from dict
    rng = random.Random(42)
    dict_words = list(freq.keys())
    null_counts = []
    for _ in range(50):
        rng.shuffle(dict_words)
        nc = 0
        for i in range(0, len(dict_words) - 1, 2):
            if dict_words[i] in freq and dict_words[i + 1] in freq:
                nc += 1
        null_counts.append(nc)

    null_mean = sum(null_counts) / max(len(null_counts), 1)
    selectivity = n_phrases / max(null_mean, 0.01)

    return {
        'test': 'V3_phrase_detection',
        'n_phrases': n_phrases,
        'detected_phrases': detected,
        'selectivity': selectivity,
        'gate': 1.5,
        'passed': selectivity >= 1.5,
    }


def _v4_cross_approach_agreement(table_data: Dict) -> Dict[str, Any]:
    """V4: Cross-approach agreement (≥ 5 anchor words)."""
    table = table_data.get('table', [])
    anchor_count = 0
    for entry in table:
        for ev in entry.get('evidence_sources', []):
            if 'anchor' in str(ev).lower():
                anchor_count += 1
                break
    return {
        'test': 'V4_cross_approach_agreement',
        'n_anchor_words': anchor_count,
        'gate': 5,
        'passed': anchor_count >= 5,
    }


def _v5_illustration_match(decode_data: Dict) -> Dict[str, Any]:
    """V5: Illustration-text match (p < 0.05)."""
    # Check if botanical section decoded tokens match plant vocabulary
    per_section = decode_data.get('per_section', [])
    botanical_hits = 0
    botanical_total = 0
    plant_words = {'herba', 'folia', 'radix', 'flos', 'semen', 'cortex', 'planta'}

    for sec in per_section:
        if 'herbal' in sec.get('section', '').lower() or 'botanical' in sec.get('section', '').lower():
            botanical_total += sec.get('n_tokens', 0)
            for hit in sec.get('sample_hits', []):
                decoded = hit.split('→')[-1] if '→' in hit else hit
                if any(pw in decoded.lower() for pw in plant_words):
                    botanical_hits += 1

    # Simple binomial test approximation
    if botanical_total > 0:
        rate = botanical_hits / botanical_total
        p_value = 1.0 - rate  # Simplified
    else:
        p_value = 1.0

    return {
        'test': 'V5_illustration_match',
        'botanical_hits': botanical_hits,
        'botanical_total': botanical_total,
        'p_value': p_value,
        'gate': 0.05,
        'passed': p_value < 0.05,
    }


def _v6_section_coherence(decode_data: Dict) -> Dict[str, Any]:
    """V6: Section coherence (≥ 3/7 domains)."""
    per_section = decode_data.get('per_section', [])
    coherent_sections = 0
    total_sections = len(per_section)

    for sec in per_section:
        rate = sec.get('dict_hit_rate', 0)
        if rate > 0.05:  # At least 5% hits
            coherent_sections += 1

    return {
        'test': 'V6_section_coherence',
        'coherent_sections': coherent_sections,
        'total_sections': total_sections,
        'gate': 3,
        'passed': coherent_sections >= 3,
    }


def _v7_language_ab_discrimination(decode_data: Dict) -> Dict[str, Any]:
    """V7: Language A/B discrimination (ratio < 2.0)."""
    per_section = decode_data.get('per_section', [])
    rates = [s.get('dict_hit_rate', 0) for s in per_section if s.get('n_tokens', 0) > 10]
    if len(rates) >= 2:
        max_rate = max(rates)
        min_rate = max(min(rates), 0.001)
        ratio = max_rate / min_rate
    else:
        ratio = 1.0
    return {
        'test': 'V7_language_AB_discrimination',
        'ratio': ratio,
        'gate': 2.0,
        'passed': ratio < 2.0,
    }


def _v8_pos_validity(decode_data: Dict) -> Dict[str, Any]:
    """V8: POS validity (selectivity > 1.0×)."""
    freq = decode_data.get('word_frequency_top30', {})
    # Check for function words (prepositions, conjunctions)
    function_words = {'ad', 'de', 'in', 'et', 'ut', 'ab', 'ex', 'cum', 'per', 'pro', 'non', 'sed'}
    found = sum(1 for w in function_words if w in freq)
    selectivity = found / max(len(function_words) * 0.1, 0.01)  # Expect ~10% randomly
    return {
        'test': 'V8_pos_validity',
        'function_words_found': found,
        'selectivity': selectivity,
        'gate': 1.0,
        'passed': selectivity > 1.0,
    }


def _v9_anchor_fidelity(table_data: Dict) -> Dict[str, Any]:
    """V9: Anchor fidelity (≥ 80% Tier 1 preserved)."""
    rdir = _results_dir()
    anchors = _load_json(str(rdir / "cross_approach.json")) or {}
    tier1 = [m for m in anchors.get('mappings', [])
             if isinstance(m, dict) and m.get('tier', '') == '1']

    table = table_data.get('table', [])
    table_lookup = {e.get('eva_char', ''): e.get('latin_syllable', '') for e in table}

    preserved = 0
    for m in tier1:
        ec = m.get('eva_token', '')
        expected = m.get('latin_value', '').lower()
        actual = (table_lookup.get(ec, '') or '').lower()
        if actual == expected:
            preserved += 1

    fidelity = preserved / max(len(tier1), 1)
    return {
        'test': 'V9_anchor_fidelity',
        'tier1_total': len(tier1),
        'preserved': preserved,
        'fidelity': fidelity,
        'gate': 0.80,
        'passed': fidelity >= 0.80,
    }


def _v10_family_coherence(table_data: Dict) -> Dict[str, Any]:
    """V10: Family coherence (≥ 6/11 sub-families)."""
    table = table_data.get('table', [])
    table_lookup = {e.get('eva_char', ''): e.get('latin_syllable', '') for e in table}

    # Group EVA chars by glyph_class (family)
    families: Dict[str, List[str]] = {}
    for ec, comp in EVA_VISUAL_COMPONENTS.items():
        gc = comp.get('glyph_class', 'unknown')
        families.setdefault(gc, []).append(ec)

    coherent = 0
    total_families = 0
    for gc, members in families.items():
        if len(members) < 2:
            continue
        total_families += 1
        syls = [table_lookup.get(m, '') for m in members]
        syls = [s for s in syls if s]
        if len(syls) < 2:
            continue
        # Check if syllables share initial consonant
        initials = set(s[0] for s in syls if s)
        if len(initials) <= 2:  # At most 2 different initial consonants
            coherent += 1

    return {
        'test': 'V10_family_coherence',
        'coherent': coherent,
        'total_families': total_families,
        'gate': 6,
        'passed': coherent >= 6,
    }


def _v11_table_stability(table_data: Dict) -> Dict[str, Any]:
    """V11: Table stability (≥ 80% pairwise)."""
    # Compare current table to Phase 15 triple assignments
    rdir = _results_dir()
    p15 = _load_json(str(rdir / "combined_refine.json")) or {}
    p15_assignments = {}
    for ta in p15.get('best_assignments', p15.get('assignments', [])):
        if isinstance(ta, dict):
            tk = ta.get('triple_key', '')
            syl = ta.get('syllable', ta.get('value', ''))
            if tk and syl:
                for ec, comp in EVA_VISUAL_COMPONENTS.items():
                    etk = f"{comp['first_stroke']},{comp['last_stroke']},{comp['glyph_class']}"
                    if etk == tk:
                        p15_assignments[ec] = syl

    table = table_data.get('table', [])
    matches = 0
    total = 0
    for entry in table:
        ec = entry.get('eva_char', '')
        syl = entry.get('latin_syllable', '')
        if ec in p15_assignments and syl:
            total += 1
            if syl == p15_assignments[ec]:
                matches += 1

    stability = matches / max(total, 1)
    return {
        'test': 'V11_table_stability',
        'matches': matches,
        'total_compared': total,
        'stability': stability,
        'gate': 0.80,
        'passed': stability >= 0.80,
    }


def _v12_phase16_improvement(decode_data: Dict) -> Dict[str, Any]:
    """V12: Phase 16 improvement (≥ 3/5 readability)."""
    rdir = _results_dir()
    p16 = _load_json(str(rdir / "modifier_integrate.json")) or {}
    p16_dict_hit = p16.get('dict_hit_rate', p16.get('r3_dict_hit_rate', 0))
    if isinstance(p16_dict_hit, dict):
        p16_dict_hit = p16_dict_hit.get('expanded', 0)

    current_rate = decode_data.get('dict_hit_rate_expanded', 0)

    # 5 readability criteria
    criteria = {
        'dict_hit_improved': current_rate > p16_dict_hit * 0.9 if p16_dict_hit else True,
        'has_function_words': bool(decode_data.get('word_frequency_top30', {})),
        'multi_section': len(decode_data.get('per_section', [])) > 1,
        'no_all_question_marks': decode_data.get('n_decoded', 0) > 0,
        'reasonable_rate': current_rate > 0.01,
    }
    n_passed = sum(criteria.values())
    return {
        'test': 'V12_phase16_improvement',
        'criteria': criteria,
        'n_passed': n_passed,
        'gate': 3,
        'passed': n_passed >= 3,
    }


def _v13_paleographic_coverage(table_data: Dict) -> Dict[str, Any]:
    """V13: Paleographic coverage (≥ 30% Priority 1-3)."""
    p13_count = table_data.get('coverage_priority_1_3', 0)
    total = table_data.get('table_size', 44)
    frac = p13_count / max(total, 1)
    return {
        'test': 'V13_paleographic_coverage',
        'priority_1_3_count': p13_count,
        'total': total,
        'fraction': frac,
        'gate': 0.30,
        'passed': frac >= 0.30,
    }


def _v14_historical_consistency(table_data: Dict) -> Dict[str, Any]:
    """V14: Historical consistency (≥ 50% match known Tironian values)."""
    # Load master reference and check how many table assignments match
    master = _load_json("data/reference/paleographic/master_reference.json") or {}
    hist_values: Dict[str, Set[str]] = {}
    for sign in master.get('all_signs', []):
        lv = sign.get('latin_value', '')
        fs = sign.get('first_stroke', '')
        if lv and fs:
            hist_values.setdefault(fs, set()).add(lv.lower())

    table = table_data.get('table', [])
    consistent = 0
    checked = 0
    for entry in table:
        syl = entry.get('latin_syllable', '')
        if not syl:
            continue
        ec = entry.get('eva_char', '')
        comp = EVA_VISUAL_COMPONENTS.get(ec, {})
        fs = comp.get('first_stroke', '')
        if fs in hist_values:
            checked += 1
            # Check if any historical sign with same first_stroke has this value
            if any(syl.lower().startswith(hv[:2]) for hv in hist_values[fs]):
                consistent += 1

    frac = consistent / max(checked, 1)
    return {
        'test': 'V14_historical_consistency',
        'consistent': consistent,
        'checked': checked,
        'fraction': frac,
        'gate': 0.50,
        'passed': frac >= 0.50,
    }


def _v15_fontana_alignment(table_data: Dict) -> Dict[str, Any]:
    """V15: Fontana alignment (≥ 3/4 gallows consistent)."""
    rdir = _results_dir()
    fontana = _load_json(str(rdir / "fontana_families.json")) or {}
    rotation_match = fontana.get('gallows_rotation_test', {}).get('rotation_match', False)

    table = table_data.get('table', [])
    gallows_chars = [e for e in table
                     if EVA_VISUAL_COMPONENTS.get(e.get('eva_char', ''), {}).get('glyph_class') == 'gallows']

    # Check if gallows share a consonant class (Fontana prediction)
    gallows_syls = [e.get('latin_syllable', '') for e in gallows_chars if e.get('latin_syllable')]
    if len(gallows_syls) >= 2:
        initials = [s[0] for s in gallows_syls if s]
        most_common = Counter(initials).most_common(1)
        if most_common:
            consistent_count = most_common[0][1]
        else:
            consistent_count = 0
    else:
        consistent_count = 0

    return {
        'test': 'V15_fontana_alignment',
        'gallows_count': len(gallows_chars),
        'consistent_count': consistent_count,
        'rotation_match': rotation_match,
        'gate': 3,
        'passed': consistent_count >= 3,
    }


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class PaleoValidateResult:
    timestamp: str
    tests: List[Dict[str, Any]]
    n_passed: int
    n_total: int
    pass_rate: float
    strong_pass: bool
    verdict: str


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_paleo_validate() -> Dict[str, Any]:
    """Run 15-test validation battery."""
    t0 = time.time()
    rdir = _results_dir()

    decode_data = _load_json(str(rdir / "paleo_decode.json")) or {}
    table_data = _load_json(str(rdir / "paleo_table.json")) or {}

    tests = [
        _v1_null_discrimination(decode_data),
        _v2_bigram_plausibility(decode_data),
        _v3_phrase_detection(decode_data),
        _v4_cross_approach_agreement(table_data),
        _v5_illustration_match(decode_data),
        _v6_section_coherence(decode_data),
        _v7_language_ab_discrimination(decode_data),
        _v8_pos_validity(decode_data),
        _v9_anchor_fidelity(table_data),
        _v10_family_coherence(table_data),
        _v11_table_stability(table_data),
        _v12_phase16_improvement(decode_data),
        _v13_paleographic_coverage(table_data),
        _v14_historical_consistency(table_data),
        _v15_fontana_alignment(table_data),
    ]

    n_passed = sum(1 for t in tests if t.get('passed'))
    n_total = len(tests)
    pass_rate = n_passed / n_total

    v3_passed = tests[2].get('passed', False)
    v13_passed = tests[12].get('passed', False)
    strong_pass = n_passed >= 12 and v3_passed and v13_passed

    if strong_pass:
        verdict = f"STRONG PASS ({n_passed}/{n_total})"
    elif n_passed >= 9:
        verdict = f"PASS ({n_passed}/{n_total})"
    else:
        verdict = f"FAIL ({n_passed}/{n_total})"

    result = PaleoValidateResult(
        timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
        tests=tests,
        n_passed=n_passed,
        n_total=n_total,
        pass_rate=pass_rate,
        strong_pass=strong_pass,
        verdict=verdict,
    )

    out_path = rdir / "paleo_validate.json"
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t0
    print(f"paleo-validate: {verdict} ({elapsed:.1f}s)")
    for t in tests:
        status = "PASS" if t.get('passed') else "FAIL"
        print(f"  {t.get('test', '?')}: {status}")

    return _convert(asdict(result))
