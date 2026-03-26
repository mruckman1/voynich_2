"""
Phase 70, Track 2: Morphological Paradigm Mapping
===================================================
Extract full paradigms from the 223 T1-identified words, map coda consonants
(n, r, s, t) to Latin case endings, and map EVA suffix characters to Latin
case distinctions.

Dependency chain:
    results/p69_clean_corpus.json        (Step 0: t1_catalogue)
    results/p69_t1_network.json          (Track 4: paradigms)
    results/combined_refine.json         (Phase 15: best_assignment)
        -> results/phase70_paradigms.json
"""

import json
import os
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    load_corpus,
    tokenize_eva_chars,
)
from voynich.phases.corrected_coda import (
    build_coda_table_v2,
    classify_token_chars_v2,
    decode_token_cvc_v2,
)
from voynich.phases.suffix_grammar import (
    LATIN_NOUN_ENDINGS,
    LATIN_VERB_ENDINGS,
    _classify_latin_ending,
)


# ---------------------------------------------------------------------------
# JSON helpers
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
    if isinstance(obj, set):
        return sorted(obj)
    if isinstance(obj, (bool, int, float, str, type(None))):
        return obj
    return str(obj)


def _safe_load(path: str) -> Dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _save_json(rd: str, filename: str, data: Any) -> str:
    path = os.path.join(rd, filename)
    with open(path, 'w') as f:
        json.dump(_convert(data), f, indent=2)
    return path


# ---------------------------------------------------------------------------
# Known Latin roots for paradigm validation
# ---------------------------------------------------------------------------

# Roots recognized in pharmaceutical Latin (stem → meaning)
_KNOWN_ROOTS: Dict[str, str] = {
    'cor': 'heart', 'cora': 'heart',
    'rad': 'root', 'radic': 'root',
    'herb': 'herb', 'herba': 'herb',
    'aqu': 'water',
    'sen': 'senna', 'senn': 'senna', 'sene': 'senna',
    'cor': 'heart', 'cord': 'heart',
    'bel': 'beautiful', 'bela': 'beautiful',
    'ser': 'serum/evening', 'sero': 'serum',
    'din': 'daily', 'di': 'of',
    'col': 'strain', 'cola': 'strain',
    'ben': 'good/well', 'bene': 'good',
    'dic': 'say', 'dico': 'I say', 'dice': 'says',
    'cer': 'wax', 'cera': 'wax',
    'ros': 'rose', 'rosa': 'rose',
    'ole': 'oil',
    'sal': 'salt/sage',
    'vin': 'wine',
    'rut': 'rue',
    'ter': 'grind',
    'misc': 'mix',
    'coqu': 'cook',
    'dec': 'decoction/decorate', 'deco': 'decorate',
    'rat': 'reason', 'rati': 'reason',
    'commun': 'common',
    'secund': 'second',
    'sterco': 'dung',
}


def _extract_full_paradigms(
    t1_catalogue: List[Dict],
) -> List[Dict[str, Any]]:
    """Group T1 words by shared prefix to form morphological paradigms.

    Each paradigm = a group of T1 words sharing a Latin root,
    with different case/number endings.
    """
    # Build word → info map
    word_info: Dict[str, Dict] = {}
    for entry in t1_catalogue:
        w = entry.get('matched_word', '')
        if w and len(w) >= 2:
            word_info[w] = entry

    all_words = sorted(word_info.keys())

    # Group by shared prefix (minimum 2 chars, at least 2 members)
    # Use longest-prefix-first strategy
    paradigms: List[Dict[str, Any]] = []
    assigned: Set[str] = set()

    for prefix_len in range(5, 1, -1):  # try longer prefixes first
        prefix_groups: Dict[str, List[str]] = defaultdict(list)
        for w in all_words:
            if w in assigned:
                continue
            if len(w) >= prefix_len:
                prefix = w[:prefix_len]
                prefix_groups[prefix].append(w)

        for prefix, members in sorted(prefix_groups.items(),
                                       key=lambda x: -len(x[1])):
            if len(members) < 2:
                continue

            # Check if any member is already assigned
            unassigned = [m for m in members if m not in assigned]
            if len(unassigned) < 2:
                continue

            # Look up root meaning
            root_meaning = '?'
            for root_candidate in [prefix, prefix[:3], prefix[:2]]:
                if root_candidate in _KNOWN_ROOTS:
                    root_meaning = _KNOWN_ROOTS[root_candidate]
                    break

            # Analyze each member
            family_members = []
            for word in sorted(unassigned):
                ending = word[len(prefix):]
                pos, case_ending = _classify_latin_ending(word)
                info = word_info.get(word, {})

                family_members.append({
                    'decoded': word,
                    'ending': ending if ending else '∅',
                    'pos': pos,
                    'case_ending': case_ending,
                    'eva_type': info.get('eva_type', ''),
                    'frequency': info.get('frequency', 0),
                    'n_folios': info.get('n_folios', 0),
                    'tier': info.get('tier', ''),
                })

            paradigms.append({
                'root': prefix,
                'meaning': root_meaning,
                'n_forms': len(family_members),
                'members': family_members,
            })

            for m in unassigned:
                assigned.add(m)

    # Sort by number of forms descending
    paradigms.sort(key=lambda p: -p['n_forms'])
    return paradigms


def _map_coda_to_case(
    paradigms: List[Dict],
    all_tokens: List[str],
    cvc_decoded: List[str],
    coda_table: Any,
    eva_to_triple: Dict[str, str],
) -> Dict[str, Dict[str, Any]]:
    """Map coda consonants (n, r, s, t) to Latin case endings.

    For each paradigm member, find the coda consonant (last char if it's
    n/r/s/t) and the Latin case ending. Build frequency matrix.
    """
    coda_case: Dict[str, Counter] = {
        'n': Counter(), 'r': Counter(), 's': Counter(),
        't': Counter(), 'm': Counter(),
    }

    for paradigm in paradigms:
        for member in paradigm['members']:
            word = member['decoded']
            case = member['case_ending']
            if not word or not case:
                continue

            # What's the last character of the decoded word?
            last_char = word[-1] if word else ''
            if last_char in coda_case:
                case_label = f"{member['pos']}_{case}" if case else member['pos']
                coda_case[last_char][case_label] += 1

    # Build summary
    coda_case_results: Dict[str, Dict[str, Any]] = {}
    for coda, case_counts in coda_case.items():
        total = sum(case_counts.values())
        if total == 0:
            continue

        top_case, top_count = case_counts.most_common(1)[0]
        coda_case_results[coda] = {
            'total_observations': total,
            'case_distribution': dict(case_counts.most_common()),
            'dominant_case': top_case,
            'dominance_fraction': top_count / total,
            'is_consistent': top_count / total > 0.40,
        }

    return coda_case_results


def _map_eva_suffix_to_case(
    paradigms: List[Dict],
    all_tokens: List[str],
) -> Dict[str, Dict[str, Any]]:
    """Compare EVA tokens of paradigm members to find which EVA chars
    encode case distinctions.

    For each pair of paradigm members that differ in case, find the
    differing EVA suffix characters.
    """
    # Build decoded → EVA token map
    # (one decoded word may come from multiple EVA tokens)
    decoded_to_eva: Dict[str, Set[str]] = defaultdict(set)
    for token in all_tokens:
        # We don't have the decoded form here without re-decoding,
        # so we use a simpler approach: match T1 catalogue entries
        pass  # will use paradigm member's eva_type directly

    suffix_observations: Dict[str, List[Dict]] = defaultdict(list)

    for paradigm in paradigms:
        members = paradigm['members']
        for i, member_a in enumerate(members):
            for j, member_b in enumerate(members):
                if j <= i:
                    continue

                eva_a = member_a.get('eva_type', '')
                eva_b = member_b.get('eva_type', '')
                if not eva_a or not eva_b or eva_a == eva_b:
                    continue

                chars_a = tokenize_eva_chars(eva_a)
                chars_b = tokenize_eva_chars(eva_b)

                # Find common prefix
                common_len = 0
                for k in range(min(len(chars_a), len(chars_b))):
                    if chars_a[k] == chars_b[k]:
                        common_len += 1
                    else:
                        break

                suffix_a = tuple(chars_a[common_len:])
                suffix_b = tuple(chars_b[common_len:])

                if suffix_a and member_a.get('case_ending'):
                    key = ','.join(suffix_a)
                    suffix_observations[key].append({
                        'decoded_ending': member_a['ending'],
                        'case': member_a['case_ending'],
                        'pos': member_a['pos'],
                        'from_paradigm': paradigm['root'],
                    })

                if suffix_b and member_b.get('case_ending'):
                    key = ','.join(suffix_b)
                    suffix_observations[key].append({
                        'decoded_ending': member_b['ending'],
                        'case': member_b['case_ending'],
                        'pos': member_b['pos'],
                        'from_paradigm': paradigm['root'],
                    })

    # Aggregate
    suffix_summary: Dict[str, Dict[str, Any]] = {}
    for suffix, observations in suffix_observations.items():
        case_counts = Counter(obs['case'] for obs in observations)
        total = sum(case_counts.values())

        if total < 2:
            continue

        top_case, top_count = case_counts.most_common(1)[0]

        suffix_summary[suffix] = {
            'n_observations': total,
            'case_distribution': dict(case_counts.most_common()),
            'dominant_case': top_case,
            'dominance_fraction': top_count / total,
            'examples': observations[:5],
        }

    return suffix_summary


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class ParadigmMapResult:
    phase: str = "70"
    step: str = "70.2"
    experiment: str = "paradigm_mapping"
    # Paradigm stats
    n_paradigms: int = 0
    n_paradigms_with_3plus: int = 0
    n_paradigms_multi_case: int = 0
    # Coda → case mapping
    coda_case_map: Dict[str, Any] = field(default_factory=dict)
    n_consistent_codas: int = 0
    # EVA suffix → case mapping
    suffix_case_map: Dict[str, Any] = field(default_factory=dict)
    n_suffix_mappings: int = 0
    # Paradigm details
    paradigm_details: List[Dict] = field(default_factory=list)
    # Gates
    gate_m1: bool = False  # >= 30 paradigms with 3+ forms
    gate_m2: bool = False  # >= 2 codas show > 40% case dominance
    gate_m3: bool = False  # >= 5 EVA suffix→case mappings with 3+ observations
    gate_m4: bool = False  # paradigm roots match known roots in >= 60%
    gates_passed: int = 0
    gate_passed: bool = False
    verdict: str = "UNKNOWN"
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_paradigm_map():
    """Track 2: Extract paradigms and map coda/suffix to Latin cases."""
    t0 = time.time()
    rd = str(_results_dir())

    print("Phase 70.2 — Morphological Paradigm Mapping")
    print("=" * 46)

    # --- Load dependencies ---
    clean_data = _safe_load(os.path.join(rd, 'p69_clean_corpus.json'))
    t1_catalogue = clean_data.get('t1_catalogue', [])
    clean_decoded = clean_data.get('clean_decoded', [])
    print(f"  T1 catalogue entries: {len(t1_catalogue)}")

    refine_data = _safe_load(os.path.join(rd, 'combined_refine.json'))
    assignment = refine_data.get('best_assignment', {})

    eva_to_triple = build_eva_to_triple_lookup()
    coda_table = build_coda_table_v2()

    corpus = load_corpus(verbose=False)
    all_tokens = corpus.get_tokens()

    # --- Step 2.1: Extract paradigms ---
    print("\n  Extracting morphological paradigms...")
    paradigms = _extract_full_paradigms(t1_catalogue)
    n_3plus = sum(1 for p in paradigms if p['n_forms'] >= 3)
    print(f"    Total paradigms: {len(paradigms)}")
    print(f"    Paradigms with 3+ forms: {n_3plus}")

    # Show top 10
    for p in paradigms[:10]:
        members_str = ', '.join(m['decoded'] for m in p['members'][:5])
        print(f"      {p['root']} ({p['meaning']}): {members_str} [{p['n_forms']} forms]")

    # --- Step 2.2: Coda → case mapping ---
    print("\n  Mapping coda consonants to Latin cases...")

    # Decode all tokens for coda analysis
    cvc_decoded = []
    for token in all_tokens:
        try:
            result = decode_token_cvc_v2(token, assignment, eva_to_triple, coda_table)
            cvc_decoded.append(result.decoded_cvc)
        except Exception:
            cvc_decoded.append('')

    coda_case_map = _map_coda_to_case(
        paradigms, all_tokens, cvc_decoded, coda_table, eva_to_triple)

    n_consistent = sum(1 for v in coda_case_map.values() if v.get('is_consistent'))
    print(f"    Codas with data: {len(coda_case_map)}")
    print(f"    Consistent codas (>40% dominance): {n_consistent}")

    for coda, info in sorted(coda_case_map.items()):
        print(f"      -{coda}: {info['dominant_case']} "
              f"({info['dominance_fraction']:.0%} of {info['total_observations']})")

    # --- Step 2.3: EVA suffix → case mapping ---
    print("\n  Mapping EVA suffixes to case distinctions...")
    suffix_map = _map_eva_suffix_to_case(paradigms, all_tokens)
    n_suf_with_3 = sum(1 for v in suffix_map.values() if v['n_observations'] >= 3)
    print(f"    Suffix mappings found: {len(suffix_map)}")
    print(f"    With 3+ observations: {n_suf_with_3}")

    for suffix, info in sorted(suffix_map.items(),
                                key=lambda x: -x[1]['n_observations'])[:10]:
        print(f"      EVA [{suffix}] → {info['dominant_case']} "
              f"({info['dominance_fraction']:.0%} of {info['n_observations']})")

    # --- Step 2.4: Paradigm root validation ---
    n_known_root = sum(1 for p in paradigms if p['meaning'] != '?')
    root_match_rate = n_known_root / len(paradigms) if paradigms else 0.0
    print(f"\n  Known root match rate: {root_match_rate:.1%} ({n_known_root}/{len(paradigms)})")

    # Count paradigms with multiple distinct case endings
    n_multi_case = 0
    for p in paradigms:
        cases = set(m['case_ending'] for m in p['members'] if m['case_ending'])
        if len(cases) >= 2:
            n_multi_case += 1

    # --- Gates ---
    g1 = n_3plus >= 30
    g2 = n_consistent >= 2
    g3 = n_suf_with_3 >= 5
    g4 = root_match_rate >= 0.60

    gates_passed = sum([g1, g2, g3, g4])

    print(f"\n  Gates: {gates_passed}/4")
    print(f"    M1 (≥30 paradigms 3+ forms): {'PASS' if g1 else 'FAIL'} ({n_3plus})")
    print(f"    M2 (≥2 consistent codas): {'PASS' if g2 else 'FAIL'} ({n_consistent})")
    print(f"    M3 (≥5 suffix mappings 3+ obs): {'PASS' if g3 else 'FAIL'} ({n_suf_with_3})")
    print(f"    M4 (≥60% known roots): {'PASS' if g4 else 'FAIL'} ({root_match_rate:.1%})")

    if gates_passed >= 3:
        verdict = 'PARADIGMS_MAPPED'
    elif gates_passed >= 1:
        verdict = 'PARTIAL_MAPPING'
    else:
        verdict = 'INSUFFICIENT_DATA'

    print(f"\n  Verdict: {verdict}")

    # --- Build result ---
    result = ParadigmMapResult(
        n_paradigms=len(paradigms),
        n_paradigms_with_3plus=n_3plus,
        n_paradigms_multi_case=n_multi_case,
        coda_case_map=coda_case_map,
        n_consistent_codas=n_consistent,
        suffix_case_map=suffix_map,
        n_suffix_mappings=n_suf_with_3,
        paradigm_details=paradigms[:50],  # top 50 paradigms
        gate_m1=g1,
        gate_m2=g2,
        gate_m3=g3,
        gate_m4=g4,
        gates_passed=gates_passed,
        gate_passed=gates_passed >= 3,
        verdict=verdict,
        runtime_seconds=time.time() - t0,
    )

    out = _save_json(rd, 'phase70_paradigms.json', asdict(result))
    print(f"\n  Saved: {out}")
    print(f"  Runtime: {result.runtime_seconds:.1f}s")

    return result
