"""
Phase 61, Track C: Costamagna Combination Rules
=================================================
Tests whether the corrected CVC decoded output respects Costamagna's
documented sign combination and syllable sequencing rules.  Compares
violation rates of real decoded text against null corpora and Latin
reference text.

Dependency chain:
    data/GL.S.III.MISC.12/extraction/costamagna_1953_catalog.json
    data/GL.S.III.MISC.12/extraction/syllabary_table.json
    results/corrected_coda.json       (Phase 60A)
    results/combined_refine.json      (Phase 15)
    results/modifier_integrate.json   (Phase 16)
    results/null_corpus.json          (Phase 17)
        -> results/phase61_costamagna_sequences.json
"""

import json
import os
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from voynich.core._paths import data_dir, results_dir as _results_dir
from voynich.core.corpus import build_eva_to_triple_lookup, load_corpus
from voynich.core.reference import build_expanded_word_set, load_reference_corpus
from voynich.phases.corrected_coda import (
    build_coda_table_v2,
    decode_corpus_cvc_v2,
)
from voynich.phases.cvc_coda_signal import (
    _build_folio_list,
    _load_shared_data,
)
from voynich.phases.cvc_segmentation import (
    _load_segmentation_inventory,
    segment_decoded_word,
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
# Phonotactic helpers
# ---------------------------------------------------------------------------

VOWELS = set('aeiou')
CONSONANTS = set('bcdfghjklmnpqrstvwxyz')

# Legal Latin consonant clusters at syllable boundaries (coda+onset)
# From Costamagna: compound consonants involve l, n, r, s + other consonant
LEGAL_CLUSTERS = {
    # s + stop/nasal/liquid
    ('s', 'c'), ('s', 'p'), ('s', 't'), ('s', 'k'), ('s', 'n'),
    ('s', 'm'), ('s', 'l'), ('s', 'r'), ('s', 'f'),
    # liquid/nasal + stop
    ('n', 'c'), ('n', 'd'), ('n', 'f'), ('n', 'g'), ('n', 's'),
    ('n', 't'), ('n', 'p'), ('n', 'b'), ('n', 'v'),
    ('l', 'c'), ('l', 'd'), ('l', 'f'), ('l', 'g'), ('l', 's'),
    ('l', 't'), ('l', 'p'), ('l', 'b'), ('l', 'v'), ('l', 'm'),
    ('r', 'c'), ('r', 'd'), ('r', 'f'), ('r', 'g'), ('r', 's'),
    ('r', 't'), ('r', 'p'), ('r', 'b'), ('r', 'v'), ('r', 'm'),
    ('r', 'n'), ('r', 'l'),
    ('m', 'b'), ('m', 'p'), ('m', 'n'),
    # stop + liquid (onset clusters, allowed across boundary)
    ('t', 'r'), ('d', 'r'), ('p', 'r'), ('b', 'r'),
    ('c', 'r'), ('g', 'r'), ('f', 'r'),
    ('t', 'l'), ('p', 'l'), ('b', 'l'), ('c', 'l'), ('g', 'l'), ('f', 'l'),
    # Geminate allowed
    ('l', 'l'), ('n', 'n'), ('r', 'r'), ('s', 's'), ('t', 't'),
}

# Sigla (whole-word signs) that must appear as standalone tokens
TIRONIAN_SIGLA = {'atque', 'super', 'supra', 'est', 'qui', 'que', 'quod'}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ConstraintResult:
    name: str
    description: str
    n_tested: int
    n_violations: int
    violation_rate: float
    examples: List[str] = field(default_factory=list)


@dataclass
class NullComparison:
    constraint_name: str
    real_rate: float
    null_mean_rate: float
    null_std_rate: float
    z_score: float
    selectivity: float         # null_rate / real_rate (higher = better)
    latin_rate: float


@dataclass
class CostSequenceResult:
    phase: str = "61"
    step: str = "61.3"
    experiment: str = "costamagna_sequences"
    n_constraints_extracted: int = 0
    n_constraints_testable: int = 0
    constraint_results: List[Dict[str, Any]] = field(default_factory=list)
    null_comparisons: List[Dict[str, Any]] = field(default_factory=list)
    n_real_lower: int = 0          # constraints where real < null
    best_selectivity: float = 0.0
    latin_comparison: Dict[str, float] = field(default_factory=dict)
    # Gates
    g1_enough_constraints: bool = False   # >= 5 testable constraints
    g2_real_lower: bool = False           # real < null on >= 3 types
    g3_selectivity: bool = False          # sel >= 1.3 on >= 1
    g4_near_latin: bool = False           # real <= 2x latin on >= 1
    gates_passed: int = 0
    gate_passed: bool = False
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Constraint extraction from catalog
# ---------------------------------------------------------------------------

def _load_catalog() -> Dict:
    """Load Costamagna 1953 catalog."""
    cat_path = os.path.join(
        str(data_dir('GL.S.III.MISC.12/extraction')),
        'costamagna_1953_catalog.json',
    )
    if not os.path.exists(cat_path):
        print(f"  WARNING: catalog not found at {cat_path}")
        return {}
    with open(cat_path) as f:
        return json.load(f)


def _extract_constraints(catalog: Dict) -> Dict[str, Any]:
    """Extract testable sequence constraints from the catalog."""
    constraints = {}

    combo = catalog.get('combination_rules', {})
    word_form = catalog.get('word_formation', {})

    # Constraint 1: Coda-onset cluster legality
    # From syllable_final_consonants: codas are m, n, r, s, t
    # From compound_consonant_examples: legal clusters
    coda_consonants = set()
    sfc = combo.get('syllable_final_consonants', {})
    for rule in sfc.get('rules', []):
        coda_consonants.add(rule['consonant'])
    constraints['coda_consonants'] = coda_consonants or {'m', 'n', 'r', 's', 't'}

    # Constraint 2: Attested open syllable types
    open_examples = combo.get('open_syllables', {}).get('examples', [])
    constraints['open_syllable_examples'] = set(s.lower() for s in open_examples)

    # Constraint 3: Attested closed syllable types
    closed_examples = combo.get('closed_syllables_and_compound_consonants', {})
    closed_syls = set(s.lower() for s in closed_examples.get('closed_syllable_examples', []))
    compound_cons = set(s.lower() for s in closed_examples.get('compound_consonant_examples', []))
    constraints['closed_syllable_examples'] = closed_syls
    constraints['compound_examples'] = compound_cons

    # Constraint 4: Connection methods
    methods = combo.get('sign_connection_methods', [])
    constraints['connection_methods'] = [m['method'] for m in methods]

    # Constraint 5: Sigla
    sigla = word_form.get('sigla', {}).get('tironian_sigla', [])
    constraints['sigla'] = set(s.lower() for s in sigla)

    # Constraint 6: Word formation examples (for initial syllable types)
    wf_examples = word_form.get('examples', [])
    initial_syls = set()
    for ex in wf_examples:
        syls = ex.get('syllables', [])
        if syls:
            initial_syls.add(syls[0].lower())
    constraints['word_initial_examples'] = initial_syls

    return constraints


# ---------------------------------------------------------------------------
# Constraint testing
# ---------------------------------------------------------------------------

def _get_final_consonant(syllable: str) -> str:
    """Get the final consonant of a syllable (empty if open)."""
    if not syllable:
        return ''
    last = syllable[-1].lower()
    return last if last in CONSONANTS else ''


def _get_initial_consonant(syllable: str) -> str:
    """Get the initial consonant(s) of a syllable."""
    if not syllable:
        return ''
    first = syllable[0].lower()
    return first if first in CONSONANTS else ''


def _segment_decoded_tokens(
    decoded_tokens: List[str],
    costamagna_inv: Set[str],
) -> List[List[Dict[str, Any]]]:
    """Segment each decoded token into Costamagna syllables."""
    segmented = []
    for word in decoded_tokens:
        if not word or word == '?':
            segmented.append([])
            continue
        segs = segment_decoded_word(word, costamagna_inv)
        segmented.append(segs)
    return segmented


def _test_coda_onset_legality(
    segmented_tokens: List[List[Dict[str, Any]]],
) -> ConstraintResult:
    """Test whether coda+onset clusters at syllable boundaries are legal."""
    n_tested = 0
    n_violations = 0
    examples = []

    for segs in segmented_tokens:
        if len(segs) < 2:
            continue
        for i in range(len(segs) - 1):
            s1 = segs[i]['text']
            s2 = segs[i + 1]['text']
            coda = _get_final_consonant(s1)
            onset = _get_initial_consonant(s2)
            if coda and onset:
                n_tested += 1
                if (coda, onset) not in LEGAL_CLUSTERS:
                    n_violations += 1
                    if len(examples) < 10:
                        examples.append(f"{s1}|{s2} ({coda}+{onset})")

    rate = n_violations / n_tested if n_tested > 0 else 0.0
    return ConstraintResult(
        name='coda_onset_legality',
        description='Consonant cluster at syllable boundary is legal Latin',
        n_tested=n_tested,
        n_violations=n_violations,
        violation_rate=round(rate, 4),
        examples=examples,
    )


def _test_open_closed_ratio(
    segmented_tokens: List[List[Dict[str, Any]]],
) -> ConstraintResult:
    """Test open/closed syllable ratio (Latin ~65-75% open)."""
    n_open = 0
    n_closed = 0

    for segs in segmented_tokens:
        for seg in segs:
            s = seg['text']
            if not s:
                continue
            if s[-1].lower() in VOWELS:
                n_open += 1
            else:
                n_closed += 1

    total = n_open + n_closed
    if total == 0:
        return ConstraintResult(
            name='open_closed_ratio',
            description='Open syllable fraction (Latin target: 65-75%)',
            n_tested=0, n_violations=0, violation_rate=0.0,
        )

    open_frac = n_open / total
    # Violation: deviation from Latin range [0.55, 0.85]
    if open_frac < 0.55 or open_frac > 0.85:
        violation_rate = abs(open_frac - 0.70)  # distance from ideal
    else:
        violation_rate = 0.0

    return ConstraintResult(
        name='open_closed_ratio',
        description=f'Open syllable fraction: {open_frac:.3f} (Latin target: 0.65-0.75)',
        n_tested=total,
        n_violations=int(violation_rate > 0) * total,  # binary: in/out of range
        violation_rate=round(violation_rate, 4),
        examples=[f"open={n_open}, closed={n_closed}, frac={open_frac:.3f}"],
    )


def _test_coda_consonant_inventory(
    segmented_tokens: List[List[Dict[str, Any]]],
    allowed_codas: Set[str],
) -> ConstraintResult:
    """Test whether coda consonants are from Costamagna's set {m,n,r,s,t}."""
    n_tested = 0
    n_violations = 0
    examples = []

    for segs in segmented_tokens:
        for seg in segs:
            s = seg['text']
            if not s:
                continue
            final = s[-1].lower()
            if final in CONSONANTS:
                n_tested += 1
                if final not in allowed_codas:
                    n_violations += 1
                    if len(examples) < 10:
                        examples.append(f"{s} (coda={final})")

    rate = n_violations / n_tested if n_tested > 0 else 0.0
    return ConstraintResult(
        name='coda_inventory',
        description='Coda consonant is from Costamagna set {m,n,r,s,t}',
        n_tested=n_tested,
        n_violations=n_violations,
        violation_rate=round(rate, 4),
        examples=examples,
    )


def _test_word_initial_consonant(
    segmented_tokens: List[List[Dict[str, Any]]],
) -> ConstraintResult:
    """Test whether word-initial consonants are legal Latin onsets."""
    # Legal Latin word-initial onsets
    legal_onsets = set('bcdfghlmnpqrstvz')
    # Legal clusters
    legal_initial_clusters = {
        'br', 'bl', 'cr', 'cl', 'dr', 'fl', 'fr', 'gl', 'gr',
        'pl', 'pr', 'sc', 'sp', 'st', 'str', 'tr',
        'qu', 'sq', 'squ',
    }

    n_tested = 0
    n_violations = 0
    examples = []

    for segs in segmented_tokens:
        if not segs:
            continue
        first_seg = segs[0]['text']
        if not first_seg:
            continue

        # Extract initial consonant(s)
        onset = ''
        for ch in first_seg:
            if ch.lower() in CONSONANTS:
                onset += ch.lower()
            else:
                break

        if not onset:
            continue  # vowel-initial, always legal

        n_tested += 1
        if len(onset) == 1:
            if onset not in legal_onsets:
                n_violations += 1
                if len(examples) < 10:
                    examples.append(f"{first_seg} (onset={onset})")
        else:
            if onset not in legal_initial_clusters:
                n_violations += 1
                if len(examples) < 10:
                    examples.append(f"{first_seg} (onset={onset})")

    rate = n_violations / n_tested if n_tested > 0 else 0.0
    return ConstraintResult(
        name='word_initial_onset',
        description='Word-initial consonant(s) are legal Latin onsets',
        n_tested=n_tested,
        n_violations=n_violations,
        violation_rate=round(rate, 4),
        examples=examples,
    )


def _test_vowel_hiatus(
    segmented_tokens: List[List[Dict[str, Any]]],
) -> ConstraintResult:
    """Test for vowel hiatus (consecutive vowel-initial syllables within a word)."""
    n_tested = 0
    n_violations = 0
    examples = []

    for segs in segmented_tokens:
        if len(segs) < 2:
            continue
        for i in range(len(segs) - 1):
            s1 = segs[i]['text']
            s2 = segs[i + 1]['text']
            if not s1 or not s2:
                continue
            # Check if s1 ends with vowel AND s2 starts with vowel (hiatus)
            if s1[-1].lower() in VOWELS and s2[0].lower() in VOWELS:
                n_tested += 1
                n_violations += 1  # hiatus is a violation
                if len(examples) < 10:
                    examples.append(f"{s1}|{s2}")
            elif s1[-1].lower() in CONSONANTS or s2[0].lower() in CONSONANTS:
                n_tested += 1  # non-hiatus pair tested and passed

    rate = n_violations / n_tested if n_tested > 0 else 0.0
    return ConstraintResult(
        name='vowel_hiatus',
        description='Vowel hiatus (V|V at syllable boundary) — should be rare',
        n_tested=n_tested,
        n_violations=n_violations,
        violation_rate=round(rate, 4),
        examples=examples,
    )


def _test_syllable_length_distribution(
    segmented_tokens: List[List[Dict[str, Any]]],
) -> ConstraintResult:
    """Test that syllable lengths are in Latin range (typically 1-4 chars)."""
    n_tested = 0
    n_violations = 0
    examples = []

    for segs in segmented_tokens:
        for seg in segs:
            s = seg['text']
            if not s:
                continue
            n_tested += 1
            if len(s) > 4 or len(s) < 1:
                n_violations += 1
                if len(examples) < 10:
                    examples.append(f"{s} (len={len(s)})")

    rate = n_violations / n_tested if n_tested > 0 else 0.0
    return ConstraintResult(
        name='syllable_length',
        description='Syllable length in range 1-4 chars',
        n_tested=n_tested,
        n_violations=n_violations,
        violation_rate=round(rate, 4),
        examples=examples,
    )


def _test_attestation_in_catalog(
    segmented_tokens: List[List[Dict[str, Any]]],
    costamagna_inv: Set[str],
) -> ConstraintResult:
    """Test whether syllables are attested in Costamagna's inventory."""
    n_tested = 0
    n_violations = 0
    examples = []

    for segs in segmented_tokens:
        for seg in segs:
            s = seg['text'].lower()
            if not s:
                continue
            n_tested += 1
            if s not in costamagna_inv:
                n_violations += 1
                if len(examples) < 10:
                    examples.append(s)

    rate = n_violations / n_tested if n_tested > 0 else 0.0
    return ConstraintResult(
        name='catalog_attestation',
        description='Syllable attested in Costamagna catalog',
        n_tested=n_tested,
        n_violations=n_violations,
        violation_rate=round(rate, 4),
        examples=examples,
    )


def _run_all_constraints(
    decoded_tokens: List[str],
    costamagna_inv: Set[str],
    coda_consonants: Set[str],
) -> List[ConstraintResult]:
    """Run all constraint tests on a decoded token list."""
    segmented = _segment_decoded_tokens(decoded_tokens, costamagna_inv)

    return [
        _test_coda_onset_legality(segmented),
        _test_open_closed_ratio(segmented),
        _test_coda_consonant_inventory(segmented, coda_consonants),
        _test_word_initial_consonant(segmented),
        _test_vowel_hiatus(segmented),
        _test_syllable_length_distribution(segmented),
        _test_attestation_in_catalog(segmented, costamagna_inv),
    ]


# ---------------------------------------------------------------------------
# Latin reference syllabification
# ---------------------------------------------------------------------------

def _syllabify_latin_tokens(
    latin_tokens: List[str],
    costamagna_inv: Set[str],
) -> List[str]:
    """Simple Latin token list for comparison — just use tokens as-is."""
    # We pass raw Latin tokens through segmentation to get a fair comparison
    return [t.lower() for t in latin_tokens if len(t) >= 2]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_cost_sequences():
    t0 = time.time()
    print("=" * 70)
    print("Phase 61, Track C: Costamagna Combination Rules")
    print("=" * 70)

    rd = str(_results_dir())

    # Load data
    print("\n  Loading data ...")
    data = _load_shared_data()

    all_tokens = data['all_tokens']
    assignment = data['assignment']
    eva_to_triple = data['eva_to_triple']
    null_token_lists = data['null_token_lists']

    coda_table = build_coda_table_v2()

    # Load Costamagna inventory
    costamagna_inv, syl_to_struct = _load_segmentation_inventory()
    print(f"  Costamagna inventory: {len(costamagna_inv)} syllables")

    # Load catalog constraints
    print("\n  1. Extracting constraints from Costamagna catalog ...")
    catalog = _load_catalog()
    constraints_data = _extract_constraints(catalog)
    coda_consonants = constraints_data.get('coda_consonants', {'m', 'n', 'r', 's', 't'})
    print(f"     Coda consonant set: {sorted(coda_consonants)}")
    print(f"     Open syllable examples: {len(constraints_data.get('open_syllable_examples', set()))}")
    print(f"     Closed syllable examples: {len(constraints_data.get('closed_syllable_examples', set()))}")
    print(f"     Sigla: {len(constraints_data.get('sigla', set()))}")

    # Decode real corpus
    print("\n  2. Decoding real corpus with corrected CVC ...")
    real_decoded = decode_corpus_cvc_v2(
        all_tokens, assignment, eva_to_triple, coda_table,
    )
    print(f"     {len(real_decoded)} tokens decoded")

    # Run constraints on real
    print("\n  3. Testing constraints on real decoded corpus ...")
    real_results = _run_all_constraints(real_decoded, costamagna_inv, coda_consonants)
    for cr in real_results:
        print(f"     {cr.name}: {cr.n_violations}/{cr.n_tested} "
              f"violations ({cr.violation_rate:.4f})")

    # Decode null corpora and test
    print("\n  4. Testing constraints on null corpora ...")
    null_constraint_rates: Dict[str, List[float]] = defaultdict(list)
    for idx, null_tokens in enumerate(null_token_lists):
        null_decoded = decode_corpus_cvc_v2(
            null_tokens, assignment, eva_to_triple, coda_table,
        )
        null_results = _run_all_constraints(null_decoded, costamagna_inv, coda_consonants)
        for cr in null_results:
            null_constraint_rates[cr.name].append(cr.violation_rate)
        if idx == 0:
            print(f"     Null corpus 0: {len(null_decoded)} tokens")

    # Load Latin reference and test
    print("\n  5. Testing constraints on Latin reference ...")
    try:
        ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
        latin_tokens = ref_corpus.get_combined_tokens('latin')[:len(all_tokens)]
        latin_decoded = _syllabify_latin_tokens(latin_tokens, costamagna_inv)
        latin_results = _run_all_constraints(latin_decoded, costamagna_inv, coda_consonants)
        latin_rates = {cr.name: cr.violation_rate for cr in latin_results}
        print(f"     Latin tokens: {len(latin_decoded)}")
    except Exception as e:
        print(f"     Latin reference unavailable: {e}")
        latin_rates = {}

    # Compute null comparisons
    print("\n  6. Computing null comparisons ...")
    comparisons: List[NullComparison] = []
    n_real_lower = 0
    best_sel = 0.0

    for cr in real_results:
        null_rates = null_constraint_rates.get(cr.name, [])
        if not null_rates:
            continue

        null_mean = sum(null_rates) / len(null_rates)
        null_var = sum((r - null_mean) ** 2 for r in null_rates) / len(null_rates)
        null_std = null_var ** 0.5

        z = (cr.violation_rate - null_mean) / null_std if null_std > 0 else 0.0
        sel = null_mean / cr.violation_rate if cr.violation_rate > 0 else (
            999.0 if null_mean > 0 else 1.0)
        lat = latin_rates.get(cr.name, -1.0)

        if cr.violation_rate < null_mean:
            n_real_lower += 1
        if sel > best_sel:
            best_sel = sel

        comp = NullComparison(
            constraint_name=cr.name,
            real_rate=cr.violation_rate,
            null_mean_rate=round(null_mean, 4),
            null_std_rate=round(null_std, 4),
            z_score=round(z, 2),
            selectivity=round(sel, 2),
            latin_rate=round(lat, 4) if lat >= 0 else -1.0,
        )
        comparisons.append(comp)
        print(f"     {cr.name}: real={cr.violation_rate:.4f}, "
              f"null={null_mean:.4f}, sel={sel:.2f}×, z={z:.2f}"
              + (f", latin={lat:.4f}" if lat >= 0 else ""))

    # Check how many constraints have real <= 2x latin
    n_near_latin = 0
    for comp in comparisons:
        if comp.latin_rate >= 0 and comp.real_rate <= 2 * comp.latin_rate + 0.001:
            n_near_latin += 1

    # Gates
    n_testable = len([cr for cr in real_results if cr.n_tested > 0])
    g1 = n_testable >= 5
    g2 = n_real_lower >= 3
    g3 = best_sel >= 1.3
    g4 = n_near_latin >= 1
    gates = sum([g1, g2, g3, g4])

    result = CostSequenceResult(
        n_constraints_extracted=len(real_results),
        n_constraints_testable=n_testable,
        constraint_results=[_convert(asdict(cr)) for cr in real_results],
        null_comparisons=[_convert(asdict(c)) for c in comparisons],
        n_real_lower=n_real_lower,
        best_selectivity=round(best_sel, 2),
        latin_comparison=latin_rates,
        g1_enough_constraints=g1,
        g2_real_lower=g2,
        g3_selectivity=g3,
        g4_near_latin=g4,
        gates_passed=gates,
        gate_passed=gates >= 3,
        runtime_seconds=round(time.time() - t0, 1),
    )

    path = _save_json(rd, 'phase61_costamagna_sequences.json', result)

    # Summary
    print("\n" + "=" * 70)
    print("  TRACK C SUMMARY: Costamagna Combination Rules")
    print("=" * 70)
    print(f"  Constraints extracted: {result.n_constraints_extracted}")
    print(f"  Testable:             {result.n_constraints_testable}")
    print(f"  Real < Null:          {result.n_real_lower}/{len(comparisons)}")
    print(f"  Best selectivity:     {result.best_selectivity:.2f}×")
    print(f"\n  Gates: {gates}/4 passed")
    print(f"    G1 (>=5 testable):    {g1}")
    print(f"    G2 (real<null >=3):   {g2}")
    print(f"    G3 (sel >= 1.3):      {g3}")
    print(f"    G4 (near Latin):      {g4}")
    print(f"\n  Saved: {path}")
    print(f"  Runtime: {result.runtime_seconds:.1f}s")
