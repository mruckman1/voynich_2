"""
Phase 71, Track 2: Root-Level Paradigm Identification
=====================================================
Expand paradigm discovery from Phase 69's 49 (T1-only) to ALL clean decoded
types. Build a root dictionary mapping decoded prefixes to Latin root meanings.
Classify roots by pharmaceutical category.

Dependency chain:
    results/combined_refine.json         (Phase 15: best_assignment)
    results/p69_clean_corpus.json        (T1 catalogue, clean_decoded)
    results/phase70_paradigms.json       (Phase 70 Track 2: paradigm data)
        -> results/phase71_root_identification.json
"""

import json
import os
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import build_eva_to_triple_lookup, load_corpus, tokenize_eva_chars
from voynich.core.reference import PHARMACEUTICAL_VOCABULARY, build_expanded_word_set, load_reference_corpus
from voynich.phases.corrected_coda import build_coda_table_v2, decode_token_cvc_v2
from voynich.phases.suffix_calibration import SIGNAL_WORDS_51
from voynich.phases.suffix_grammar import _classify_latin_ending


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
# Known Latin roots (extended from p70_paradigm_map._KNOWN_ROOTS)
# ---------------------------------------------------------------------------

_KNOWN_ROOTS: Dict[str, str] = {
    'cor': 'heart', 'cora': 'heart', 'cord': 'heart',
    'rad': 'root', 'radic': 'root',
    'herb': 'herb', 'herba': 'herb',
    'aqu': 'water',
    'sen': 'senna', 'senn': 'senna', 'sene': 'senna',
    'bel': 'beautiful', 'bela': 'beautiful',
    'ser': 'serum/evening', 'sero': 'serum',
    'din': 'daily', 'di': 'of',
    'col': 'strain', 'cola': 'strain',
    'ben': 'good/well', 'bene': 'good',
    'dic': 'say', 'dico': 'I say', 'dice': 'says',
    'cer': 'wax', 'cera': 'wax',
    'ros': 'rose', 'rosa': 'rose',
    'ole': 'oil', 'oleu': 'oil',
    'sal': 'salt/sage', 'salv': 'sage',
    'vin': 'wine',
    'rut': 'rue',
    'ter': 'grind',
    'misc': 'mix',
    'coqu': 'cook',
    'dec': 'decoction', 'deco': 'decorate',
    'rat': 'reason', 'rati': 'reason',
    'commun': 'common',
    'secund': 'second',
    'sterco': 'dung',
    'ne': 'not/nor',
    'se': 'self/if',
    'cu': 'with',
    'cone': 'with',
    'bon': 'good',
    'ner': 'nerve',
    'cor': 'heart',
    'mel': 'honey',
    'rub': 'red',
    'nig': 'black',
    'alb': 'white',
    'sicc': 'dry',
    'cal': 'hot',
    'frig': 'cold',
    'dur': 'hard',
    'mol': 'soft',
    'fort': 'strong',
    'gra': 'heavy',
    'lev': 'light',
    'solv': 'dissolve',
    'pon': 'put/place',
    'fac': 'make',
    'add': 'add',
    'coc': 'cook',
    'bull': 'boil',
    'lav': 'wash',
    'semin': 'seed',
    'foli': 'leaf',
    'flor': 'flower',
    'cortic': 'bark',
    'fruct': 'fruit',
    'succus': 'juice',
    'pulv': 'powder',
}


# ---------------------------------------------------------------------------
# Pharmaceutical classification keywords
# ---------------------------------------------------------------------------

_PHARMA_KEYWORDS = {
    'INGREDIENT': [
        'senna', 'coral', 'root', 'herb', 'flower', 'seed', 'bark', 'leaf',
        'honey', 'wax', 'salt', 'oil', 'water', 'wine', 'dung', 'rose',
        'vinegar', 'juice', 'powder', 'fruit', 'sage',
    ],
    'PREPARATION': [
        'strain', 'grind', 'mix', 'cook', 'add', 'dissolve', 'take', 'make',
        'boil', 'put', 'wash', 'dry',
    ],
    'BODY_PART': [
        'heart', 'nerve', 'bone', 'skin', 'eye', 'head', 'stomach',
    ],
    'QUALITY': [
        'good', 'well', 'beautiful', 'black', 'white', 'red', 'hot', 'cold',
        'dry', 'wet', 'strong', 'soft', 'hard', 'heavy', 'light',
    ],
    'FUNCTION': [
        'of', 'with', 'not', 'nor', 'and', 'or', 'if', 'from', 'to',
        'self', 'the',
    ],
    'QUANTITY': [
        'half', 'third', 'six', 'two', 'ounce', 'dram', 'pound', 'daily',
        'second', 'common',
    ],
}


def _classify_pharma(meaning: str) -> str:
    """Classify a root meaning into a pharmaceutical category."""
    if not meaning or meaning == '?':
        return 'UNKNOWN'

    ml = meaning.lower()
    for category, keywords in _PHARMA_KEYWORDS.items():
        if any(kw in ml for kw in keywords):
            return category
    return 'OTHER'


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def _discover_all_paradigms(
    decoded_types: List[str],
    t1_catalogue: List[Dict],
    expanded_dict: Set[str],
    min_family_size: int = 3,
    max_root_len: int = 6,
    min_root_len: int = 2,
) -> List[Dict[str, Any]]:
    """Find ALL morphological paradigms in the clean decoded vocabulary.

    A paradigm = set of decoded word types sharing a prefix (root)
    that differ only in suffix (case/conjugation ending).
    """
    all_types = sorted(set(d for d in decoded_types if d and len(d) >= 3))

    # Build T1 word set for tagging
    t1_words = set(e.get('matched_word', '') for e in t1_catalogue if e.get('matched_word'))

    paradigms = []
    used_words: Set[str] = set()

    for root_len in range(max_root_len, min_root_len - 1, -1):
        prefix_groups: Dict[str, List[str]] = defaultdict(list)
        for word in all_types:
            if word in used_words:
                continue
            if len(word) < root_len + 1:
                continue
            prefix = word[:root_len]
            prefix_groups[prefix].append(word)

        for prefix, members in sorted(prefix_groups.items(),
                                       key=lambda x: -len(x[1])):
            unassigned = [m for m in members if m not in used_words]
            if len(unassigned) < min_family_size:
                continue

            # Require different suffixes
            suffixes = set(w[root_len:] for w in unassigned)
            if len(suffixes) < min_family_size:
                continue

            # Look up root meaning
            root_meaning = '?'
            for candidate in [prefix, prefix[:4], prefix[:3], prefix[:2]]:
                if candidate in _KNOWN_ROOTS:
                    root_meaning = _KNOWN_ROOTS[candidate]
                    break

            member_details = []
            for word in sorted(unassigned):
                suffix = word[root_len:]
                pos, case_ending = _classify_latin_ending(word)
                freq = decoded_types.count(word)

                member_details.append({
                    'decoded': word,
                    'suffix': suffix if suffix else '∅',
                    'pos': pos,
                    'case_ending': case_ending,
                    'frequency': freq,
                    'in_dictionary': word in expanded_dict,
                    'is_t1': word in t1_words,
                })

            paradigms.append({
                'root': prefix,
                'root_length': root_len,
                'meaning': root_meaning,
                'n_forms': len(member_details),
                'known_root': root_meaning != '?',
                'members': member_details,
            })

            for w in unassigned:
                used_words.add(w)

    paradigms.sort(key=lambda p: -p['n_forms'])
    return paradigms


def _build_root_dictionary(
    paradigms: List[Dict],
    t1_catalogue: List[Dict],
    expanded_dict: Set[str],
) -> Dict[str, Dict[str, Any]]:
    """Build dictionary mapping roots to meanings and paradigm info."""
    # Build T1 lookup
    t1_lookup: Dict[str, Dict] = {}
    for entry in t1_catalogue:
        w = entry.get('matched_word', '')
        if w:
            t1_lookup[w] = entry

    # Signal word lookup
    signal_lookup = {w: info.get('gloss', w) for w, info in SIGNAL_WORDS_51.items()}

    root_dict: Dict[str, Dict[str, Any]] = {}

    for paradigm in paradigms:
        root = paradigm['root']
        meaning = paradigm['meaning']
        source = 'known_root' if meaning != '?' else 'unknown'

        # Try T1
        if meaning == '?':
            for m in paradigm['members']:
                if m['is_t1'] and m['decoded'] in t1_lookup:
                    t1_entry = t1_lookup[m['decoded']]
                    meaning = t1_entry.get('gloss', m['decoded'])
                    source = 'T1'
                    break

        # Try signal words
        if meaning == '?':
            for m in paradigm['members']:
                if m['decoded'] in signal_lookup:
                    meaning = signal_lookup[m['decoded']]
                    source = 'signal'
                    break

        # Try pharma vocabulary
        if meaning == '?':
            for cat, words in PHARMACEUTICAL_VOCABULARY.items():
                for w in words:
                    if w.lower().startswith(root) or root.startswith(w.lower()[:3]):
                        meaning = f"{w.lower()} ({cat})"
                        source = 'pharma_vocab'
                        break
                if meaning != '?':
                    break

        pharma_class = _classify_pharma(meaning)

        root_dict[root] = {
            'meaning': meaning,
            'source': source,
            'pharma_class': pharma_class,
            'n_forms': paradigm['n_forms'],
            'total_frequency': sum(m['frequency'] for m in paradigm['members']),
            'n_dict_hits': sum(1 for m in paradigm['members'] if m['in_dictionary']),
            'n_t1': sum(1 for m in paradigm['members'] if m['is_t1']),
            'example_forms': {
                m['decoded']: m['suffix']
                for m in paradigm['members'][:5]
            },
        }

    return root_dict


def _compute_root_coverage(
    root_dict: Dict[str, Dict],
    decoded_types: List[str],
) -> Dict[str, Any]:
    """How much of the corpus is covered by identified roots?"""
    total = len(decoded_types)
    known_tokens = 0
    unknown_tokens = 0
    paradigm_tokens = 0

    for root, info in root_dict.items():
        for form in info['example_forms']:
            count = decoded_types.count(form)
            paradigm_tokens += count
            if info['meaning'] != '?':
                known_tokens += count
            else:
                unknown_tokens += count

    return {
        'total_tokens': total,
        'paradigm_tokens': paradigm_tokens,
        'paradigm_coverage': paradigm_tokens / total if total > 0 else 0.0,
        'known_root_tokens': known_tokens,
        'known_root_coverage': known_tokens / total if total > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class RootIdentificationResult:
    phase: str = "71"
    step: str = "71.2"
    experiment: str = "root_identification"
    # Paradigm stats
    n_paradigms: int = 0
    n_paradigms_3plus: int = 0
    n_roots_identified: int = 0
    n_roots_unknown: int = 0
    identified_fraction: float = 0.0
    # Pharma classification
    pharma_distribution: Dict[str, int] = field(default_factory=dict)
    n_ingredient_roots: int = 0
    n_preparation_roots: int = 0
    # Coverage
    paradigm_coverage: float = 0.0
    known_root_coverage: float = 0.0
    # Top paradigms (for JSON)
    top_paradigms: List[Dict] = field(default_factory=list)
    # Root dictionary (top entries)
    root_dictionary_sample: List[Dict] = field(default_factory=list)
    # Gates
    gate_r1: bool = False  # >= 80 paradigms with 3+ forms
    gate_r2: bool = False  # >= 30% roots identified
    gate_r3: bool = False  # >= 20 INGREDIENT roots
    gate_r4: bool = False  # >= 5 PREPARATION verb roots
    gate_r5: bool = False  # paradigm coverage > 30% of clean corpus
    gates_passed: int = 0
    gate_passed: bool = False
    verdict: str = "UNKNOWN"
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_root_id():
    """Track 2: Root-level paradigm identification."""
    t0 = time.time()
    rd = str(_results_dir())

    print("Phase 71.2 — Root-Level Paradigm Identification")
    print("=" * 49)

    # --- Load dependencies ---
    clean_data = _safe_load(os.path.join(rd, 'p69_clean_corpus.json'))
    t1_catalogue = clean_data.get('t1_catalogue', [])
    clean_decoded = clean_data.get('clean_decoded', [])
    print(f"  T1 catalogue: {len(t1_catalogue)} entries")
    print(f"  Clean decoded tokens: {len(clean_decoded)}")

    refine_data = _safe_load(os.path.join(rd, 'combined_refine.json'))
    assignment = refine_data.get('best_assignment', {})

    eva_to_triple = build_eva_to_triple_lookup()
    coda_table = build_coda_table_v2()

    # Build expanded dict
    pharma_data = _safe_load(os.path.join(rd, 'phase70_pharma_dict.json'))
    if pharma_data.get('combined_word_list'):
        expanded_dict = set(pharma_data['combined_word_list'])
        print(f"  Using Track 70.1 dict: {len(expanded_dict)} words")
    else:
        print("  Building base expanded dict...")
        ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
        base_words = set(w.lower() for w in ref_corpus.get_combined_tokens('latin')
                         if len(w) >= 2)
        expanded_dict, _ = build_expanded_word_set(base_words)
        expanded_dict = base_words | expanded_dict
        print(f"  Base expanded dict: {len(expanded_dict)} words")

    # If no clean_decoded from Phase 69, decode the full corpus
    if not clean_decoded:
        print("  No clean_decoded available, decoding full corpus...")
        corpus = load_corpus(verbose=False)
        all_tokens = corpus.get_tokens()
        clean_decoded = []
        for token in all_tokens:
            try:
                result = decode_token_cvc_v2(token, assignment, eva_to_triple, coda_table)
                clean_decoded.append(result.decoded_cvc)
            except Exception:
                clean_decoded.append('')

    # --- Discover paradigms ---
    print("\n  Discovering paradigms from all clean decoded types...")
    paradigms = _discover_all_paradigms(
        clean_decoded, t1_catalogue, expanded_dict,
        min_family_size=3, max_root_len=6, min_root_len=2)

    n_3plus = sum(1 for p in paradigms if p['n_forms'] >= 3)
    print(f"  Total paradigms: {len(paradigms)}")
    print(f"  With 3+ forms: {n_3plus}")

    for p in paradigms[:15]:
        members = ', '.join(m['decoded'] for m in p['members'][:4])
        tag = f" ({p['meaning']})" if p['meaning'] != '?' else ""
        print(f"    {p['root']}{tag}: {members} [{p['n_forms']} forms]")

    # --- Build root dictionary ---
    print("\n  Building root dictionary...")
    root_dict = _build_root_dictionary(paradigms, t1_catalogue, expanded_dict)

    n_identified = sum(1 for r in root_dict.values() if r['meaning'] != '?')
    n_unknown = sum(1 for r in root_dict.values() if r['meaning'] == '?')
    id_frac = n_identified / len(root_dict) if root_dict else 0.0
    print(f"  Roots identified: {n_identified}/{len(root_dict)} ({id_frac:.1%})")

    # --- Pharmaceutical classification ---
    pharma_dist = Counter(r['pharma_class'] for r in root_dict.values())
    n_ingredient = pharma_dist.get('INGREDIENT', 0)
    n_preparation = pharma_dist.get('PREPARATION', 0)

    print("\n  Pharmaceutical classification:")
    for cat, count in pharma_dist.most_common():
        print(f"    {cat}: {count}")

    # --- Coverage ---
    print("\n  Computing coverage...")
    coverage = _compute_root_coverage(root_dict, clean_decoded)
    print(f"  Paradigm coverage: {coverage['paradigm_coverage']:.1%}")
    print(f"  Known root coverage: {coverage['known_root_coverage']:.1%}")

    # --- Gates ---
    g1 = n_3plus >= 80
    g2 = id_frac >= 0.30
    g3 = n_ingredient >= 20
    g4 = n_preparation >= 5
    g5 = coverage['paradigm_coverage'] > 0.30

    gates_passed = sum([g1, g2, g3, g4, g5])

    print(f"\n  Gates: {gates_passed}/5")
    print(f"    R1 (≥80 paradigms 3+ forms): {'PASS' if g1 else 'FAIL'} ({n_3plus})")
    print(f"    R2 (≥30% roots identified): {'PASS' if g2 else 'FAIL'} ({id_frac:.1%})")
    print(f"    R3 (≥20 INGREDIENT roots): {'PASS' if g3 else 'FAIL'} ({n_ingredient})")
    print(f"    R4 (≥5 PREPARATION roots): {'PASS' if g4 else 'FAIL'} ({n_preparation})")
    print(f"    R5 (paradigm coverage >30%): {'PASS' if g5 else 'FAIL'} "
          f"({coverage['paradigm_coverage']:.1%})")

    if gates_passed >= 4:
        verdict = 'ROOTS_IDENTIFIED'
    elif gates_passed >= 2:
        verdict = 'PARTIAL_IDENTIFICATION'
    else:
        verdict = 'INSUFFICIENT_ROOTS'

    print(f"\n  Verdict: {verdict}")

    # --- Build result ---
    # Root dictionary sample for JSON (top 100 by frequency)
    root_sample = sorted(
        [{'root': r, **info} for r, info in root_dict.items()],
        key=lambda x: -x.get('total_frequency', 0),
    )[:100]

    result = RootIdentificationResult(
        n_paradigms=len(paradigms),
        n_paradigms_3plus=n_3plus,
        n_roots_identified=n_identified,
        n_roots_unknown=n_unknown,
        identified_fraction=id_frac,
        pharma_distribution=dict(pharma_dist),
        n_ingredient_roots=n_ingredient,
        n_preparation_roots=n_preparation,
        paradigm_coverage=coverage['paradigm_coverage'],
        known_root_coverage=coverage['known_root_coverage'],
        top_paradigms=paradigms[:50],
        root_dictionary_sample=root_sample,
        gate_r1=g1,
        gate_r2=g2,
        gate_r3=g3,
        gate_r4=g4,
        gate_r5=g5,
        gates_passed=gates_passed,
        gate_passed=gates_passed >= 4,
        verdict=verdict,
        runtime_seconds=time.time() - t0,
    )

    out = _save_json(rd, 'phase71_root_identification.json', asdict(result))
    print(f"\n  Saved: {out}")
    print(f"  Runtime: {result.runtime_seconds:.1f}s")

    return result
