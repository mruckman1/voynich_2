"""
Phase 7.5 Step 3: Verb Identification
======================================
Build a 15x10 compatibility score matrix between Voynich verb candidates
and Latin pharmaceutical imperatives. Find optimal assignment via Hungarian
algorithm. Check cross-consistency with Phase 6.1 character mappings.

Sub-analyses:
  3a — Profile the 15 Voynich verb candidates
  3b — Build 6-criteria compatibility matrix
  3c — Hungarian optimal assignment
  3d — Cross-consistency with Phase 6.1 character mappings
  3e — Null test: random verb targets

Output:
  results/verb_identification.json
"""

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from voynich.core.corpus import load_corpus, VoynichCorpus
from voynich.core.stats import (
    cosine_similarity, selectivity_ratio,
    hungarian_assignment, rank_correlation,
)
from voynich.core._paths import results_dir as _results_dir
from voynich.core.reference import LATIN_PHARMACEUTICAL_IMPERATIVES
from voynich.phases.morpheme_grid import (
    decompose_token_morphemes, KNOWN_SUFFIXES,
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class VoynichVerbProfile:
    """Profile of one Voynich verb candidate."""
    stem: str
    frequency: int
    frequency_rank: int
    n_suffix_types: int
    n_forms: int
    position_0_pct: float
    position_distribution: List[float]  # fraction at each position (0..7)
    section_distribution: Dict[str, int]
    top_cooccurring_nouns: List[str]
    stem_char_length: int


@dataclass
class VerbAssignment:
    """One assignment from the Hungarian algorithm solution."""
    voynich_stem: str
    latin_verb: str
    latin_meaning: str
    total_score: float
    criterion_scores: Dict[str, float]
    is_confident: bool


@dataclass
class VerbIdentificationResult:
    """Full Phase 7.5 Step 3 output."""
    n_voynich_verbs: int
    n_latin_imperatives: int
    # Voynich verb profiles
    voynich_verb_profiles: List[Dict]
    # Compatibility matrix
    compatibility_matrix: List[List[float]]
    voynich_stems: List[str]
    latin_verbs: List[str]
    # Optimal assignment
    assignments: List[Dict]
    mean_assignment_score: float
    best_total_score: float
    second_best_total_score: float
    assignment_gap: float
    n_confident_assignments: int
    # Cross-consistency with Phase 6.1
    n_char_mappings_tested: int
    n_char_mappings_consistent: int
    char_consistency_rate: float
    # Null test
    null_mean_score: float
    null_std_score: float
    assignment_selectivity: float
    # Gate
    gate_passed: bool
    verdict: str


# ---------------------------------------------------------------------------
# 3a — Profile Voynich verb candidates
# ---------------------------------------------------------------------------

def profile_voynich_verbs(
    segments: list,
    verb_stems: List[str],
    noun_stems: List[str],
    corpus: VoynichCorpus,
) -> List[VoynichVerbProfile]:
    """
    Profile each Voynich verb candidate: frequency, paradigm, position
    distribution, section distribution, co-occurring nouns.
    """
    verb_set = set(verb_stems)
    noun_set = set(noun_stems)

    # Global stem frequency
    stem_freq: Counter = Counter()
    for page in corpus.pages.values():
        for tok in page.all_tokens:
            d = decompose_token_morphemes(tok)
            stem = d.stem if d.stem else tok
            stem_freq[stem] += 1

    # Position distribution (positions 0..7+)
    max_pos = 8
    stem_pos_counts: Dict[str, List[int]] = defaultdict(lambda: [0] * max_pos)
    stem_total: Counter = Counter()

    # Section distribution
    stem_section: Dict[str, Counter] = defaultdict(Counter)
    for page in corpus.pages.values():
        for tok in page.all_tokens:
            d = decompose_token_morphemes(tok)
            stem = d.stem if d.stem else tok
            if stem in verb_set:
                stem_section[stem][page.section] += 1

    # Position counting within segments
    for seg in segments:
        n = len(seg.stems)
        for i, stem in enumerate(seg.stems):
            if stem in verb_set:
                pos = min(i, max_pos - 1)
                stem_pos_counts[stem][pos] += 1
                stem_total[stem] += 1

    # Co-occurring nouns: for each verb, which nouns appear in the same segment
    stem_cooc_nouns: Dict[str, Counter] = defaultdict(Counter)
    for seg in segments:
        seg_verbs = [s for s in seg.stems if s in verb_set]
        seg_nouns = [s for s in seg.stems if s in noun_set]
        for v in seg_verbs:
            for n in seg_nouns:
                stem_cooc_nouns[v][n] += 1

    # Paradigm: suffix types
    stem_suffix_types: Dict[str, set] = defaultdict(set)
    stem_forms: Dict[str, set] = defaultdict(set)
    for page in corpus.pages.values():
        for tok in page.all_tokens:
            d = decompose_token_morphemes(tok)
            stem = d.stem if d.stem else tok
            if stem in verb_set:
                if d.suffix:
                    stem_suffix_types[stem].add(d.suffix)
                stem_forms[stem].add(tok)

    # Build profiles, ranked by frequency
    ranked = sorted(verb_stems, key=lambda s: stem_freq.get(s, 0), reverse=True)
    profiles = []
    for rank, stem in enumerate(ranked, 1):
        total = stem_total.get(stem, 0)
        pos_dist = stem_pos_counts.get(stem, [0] * max_pos)
        pos_pcts = [p / max(total, 1) for p in pos_dist]
        pos0_pct = pos_pcts[0] * 100

        top_nouns = [n for n, _ in stem_cooc_nouns.get(stem, Counter()).most_common(10)]

        # Stem length in EVA characters
        from voynich.core.corpus import tokenize_eva_chars
        stem_chars = tokenize_eva_chars(stem)

        profiles.append(VoynichVerbProfile(
            stem=stem,
            frequency=stem_freq.get(stem, 0),
            frequency_rank=rank,
            n_suffix_types=len(stem_suffix_types.get(stem, set())),
            n_forms=len(stem_forms.get(stem, set())),
            position_0_pct=pos0_pct,
            position_distribution=[float(p) for p in pos_pcts],
            section_distribution=dict(stem_section.get(stem, Counter())),
            top_cooccurring_nouns=top_nouns,
            stem_char_length=len(stem_chars),
        ))

    return profiles


# ---------------------------------------------------------------------------
# 3b — Compatibility matrix
# ---------------------------------------------------------------------------

def build_compatibility_matrix(
    voynich_profiles: List[VoynichVerbProfile],
    latin_imperatives: List[Dict],
    char_profiles: List[Dict],
    noun_subclusters: Optional[List[Dict]],
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Build compatibility score matrix (n_voynich x n_latin), 6 criteria per pair.
    """
    n_v = len(voynich_profiles)
    n_l = len(latin_imperatives)
    matrix = np.zeros((n_v, n_l))

    v_stems = [vp.stem for vp in voynich_profiles]
    l_verbs = [li['verb'] for li in latin_imperatives]

    # Build a map from subcluster label to set of stems
    subcluster_stems: Dict[str, set] = {}
    if noun_subclusters:
        for sc in noun_subclusters:
            label = sc.get('label', '')
            stems = sc.get('top_stems', [])
            subcluster_stems[label] = set(stems)

    # Build high-unanimity character map from Phase 6.1
    char_map: Dict[str, str] = {}
    for cp in char_profiles:
        if cp.get('classification') == 'high' and cp.get('unanimity', 0) > 0.8:
            char_map[cp['eva_char']] = cp['majority_value']

    max_rank = max(n_v, n_l)

    for i, vp in enumerate(voynich_profiles):
        for j, li in enumerate(latin_imperatives):
            scores: Dict[str, float] = {}

            # 1. Frequency rank proximity
            rank_dist = abs(vp.frequency_rank - li['frequency_rank'])
            scores['frequency_rank'] = 1.0 - rank_dist / max(max_rank, 1)

            # 2. Paradigm form count match
            expected_forms = len(li.get('imperative_forms', []))
            if expected_forms > 0 and vp.n_forms > 0:
                form_ratio = min(vp.n_forms, expected_forms) / max(vp.n_forms, expected_forms)
                scores['paradigm_match'] = form_ratio
            else:
                scores['paradigm_match'] = 0.5

            # 3. Stem length compatibility
            v_len = vp.stem_char_length
            l_len = li.get('n_chars', 4)
            if max(v_len, l_len) > 0:
                scores['stem_length'] = 1.0 - abs(v_len - l_len) / max(v_len, l_len)
            else:
                scores['stem_length'] = 0.5

            # 4. Positional profile: both should be segment-initial
            scores['positional'] = vp.position_0_pct / 100.0

            # 5. Object noun compatibility
            if noun_subclusters and vp.top_cooccurring_nouns:
                expected_objects = li.get('typical_objects', [])
                expected_stems: set = set()
                for domain in expected_objects:
                    expected_stems |= subcluster_stems.get(domain, set())

                if expected_stems:
                    overlap = len(set(vp.top_cooccurring_nouns) & expected_stems)
                    scores['object_compat'] = overlap / len(vp.top_cooccurring_nouns)
                else:
                    scores['object_compat'] = 0.5
            else:
                scores['object_compat'] = 0.5

            # 6. Character mapping consistency
            if char_map:
                from voynich.core.corpus import tokenize_eva_chars
                v_chars = tokenize_eva_chars(vp.stem)
                l_stem = li.get('stem', '')
                n_testable = 0
                n_consistent = 0
                for ci, vc in enumerate(v_chars):
                    if vc in char_map and ci < len(l_stem):
                        n_testable += 1
                        if char_map[vc] == l_stem[ci]:
                            n_consistent += 1
                scores['char_mapping'] = (
                    n_consistent / n_testable if n_testable > 0 else 0.5
                )
            else:
                scores['char_mapping'] = 0.5

            # Weighted average (equal weights)
            matrix[i, j] = sum(scores.values()) / len(scores)

    return matrix, v_stems, l_verbs


# ---------------------------------------------------------------------------
# 3c — Hungarian optimal assignment
# ---------------------------------------------------------------------------

def run_hungarian_assignment(
    matrix: np.ndarray,
    voynich_stems: List[str],
    latin_verbs: List[str],
    latin_imperatives: List[Dict],
    confidence_threshold: float = 0.5,
) -> Tuple[List[VerbAssignment], float]:
    """
    Find optimal assignment maximizing total compatibility.

    Returns (assignments, second_best_total).
    """
    row_ind, col_ind, total_score = hungarian_assignment(matrix)

    assignments = []
    for ri, ci in zip(row_ind, col_ind):
        if ci < len(latin_imperatives):
            li = latin_imperatives[ci]
            score = float(matrix[ri, ci])
            assignments.append(VerbAssignment(
                voynich_stem=voynich_stems[ri],
                latin_verb=latin_verbs[ci],
                latin_meaning=li.get('meaning', ''),
                total_score=score,
                criterion_scores={},
                is_confident=score >= confidence_threshold,
            ))

    # Second-best: mask out best assignment, re-run
    masked = matrix.copy()
    for ri, ci in zip(row_ind, col_ind):
        masked[ri, ci] = -1.0
    _, _, second_total = hungarian_assignment(masked)

    return assignments, second_total


# ---------------------------------------------------------------------------
# 3d — Cross-consistency with Phase 6.1
# ---------------------------------------------------------------------------

def check_verb_char_consistency(
    assignments: List[VerbAssignment],
    char_profiles: List[Dict],
) -> Tuple[int, int, float]:
    """
    For each verb assignment, extract implied character mappings and
    check against Phase 6.1's high-unanimity mappings.

    Returns (n_consistent, n_tested, consistency_rate).
    """
    # Build consensus map
    consensus: Dict[str, str] = {}
    for cp in char_profiles:
        if cp.get('classification') == 'high' and cp.get('unanimity', 0) > 0.8:
            consensus[cp['eva_char']] = cp['majority_value']

    if not consensus:
        return 0, 0, 0.0

    from voynich.core.corpus import tokenize_eva_chars

    n_tested = 0
    n_consistent = 0

    for asgn in assignments:
        v_chars = tokenize_eva_chars(asgn.voynich_stem)
        # Get Latin stem for this verb
        l_stem = ''
        for li in LATIN_PHARMACEUTICAL_IMPERATIVES:
            if li['verb'] == asgn.latin_verb:
                l_stem = li['stem']
                break

        if not l_stem:
            continue

        # Check character-by-character
        for ci, vc in enumerate(v_chars):
            if vc in consensus and ci < len(l_stem):
                n_tested += 1
                if consensus[vc] == l_stem[ci]:
                    n_consistent += 1

    rate = n_consistent / n_tested if n_tested > 0 else 0.0
    return n_consistent, n_tested, rate


# ---------------------------------------------------------------------------
# Null test
# ---------------------------------------------------------------------------

def _null_test_assignment(
    matrix: np.ndarray,
    real_total_score: float,
    n_trials: int = 100,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Null test: independently shuffle each column of the compatibility matrix,
    breaking the row-column correspondence, then re-run Hungarian assignment.

    Returns (null_mean, null_std, selectivity).
    """
    rng = np.random.RandomState(seed)
    null_scores = []
    for _ in range(n_trials):
        shuffled = matrix.copy()
        # Shuffle each column independently — this breaks the pairing
        # between verb profiles and Latin verb properties
        for col in range(shuffled.shape[1]):
            rng.shuffle(shuffled[:, col])
        _, _, total = hungarian_assignment(shuffled)
        null_scores.append(total)

    null_arr = np.array(null_scores)
    null_mean = float(np.mean(null_arr))
    null_std = float(np.std(null_arr))
    # Higher total score is better
    sel = real_total_score / null_mean if null_mean > 1e-10 else float('inf')
    return null_mean, null_std, sel


# ---------------------------------------------------------------------------
# JSON conversion
# ---------------------------------------------------------------------------

def _convert(obj):
    """Convert dataclass/numpy types to JSON-serializable form."""
    if hasattr(obj, '__dataclass_fields__'):
        return {k: _convert(v) for k, v in asdict(obj).items()}
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, dict):
        return {str(k): _convert(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_convert(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_verb_identification() -> Dict:
    """
    Phase 7.5 Step 3: Verb Identification.

    Profiles 15 Voynich verb candidates, builds a compatibility matrix
    against 10 Latin pharmaceutical imperatives, finds optimal assignment
    via Hungarian algorithm, and checks cross-consistency with Phase 6.1
    character mappings.
    """
    print("Phase 7.5 Step 3: Verb Identification")
    print("=" * 70)

    corpus = load_corpus(verbose=False)

    # Rebuild verb/noun candidate lists
    print("\n  Loading positional slot data...")
    from voynich.phases.positional_slots import (
        segment_voynich_pharmaceutical, classify_stems_by_position,
    )
    segments = segment_voynich_pharmaceutical(corpus)
    verb_stems, noun_stems, _ = classify_stems_by_position(segments)
    print(f"  {len(verb_stems)} verb candidates, {len(noun_stems)} noun candidates")

    # Profile Voynich verbs
    print("\n  Profiling Voynich verb candidates...")
    profiles = profile_voynich_verbs(segments, verb_stems, noun_stems, corpus)
    for vp in profiles:
        print(f"    {vp.stem:<12s} freq={vp.frequency:<5d} rank={vp.frequency_rank} "
              f"forms={vp.n_forms} pos0={vp.position_0_pct:.0f}% "
              f"chars={vp.stem_char_length}")

    # Load Phase 6.1 character profiles
    print("\n  Loading Phase 6.1 character profiles...")
    char_profiles = []
    diag_path = _results_dir() / 'anchor_diagnosis.json'
    if diag_path.exists():
        with open(diag_path) as f:
            diag = json.load(f)
        char_profiles = diag.get('char_profiles', [])
        n_high = sum(1 for cp in char_profiles if cp.get('classification') == 'high')
        print(f"  Loaded {len(char_profiles)} profiles ({n_high} high-unanimity)")
    else:
        print("  WARNING: anchor_diagnosis.json not found. "
              "Character consistency check will be limited.")

    # Load noun subclusters (from Step 2)
    print("\n  Loading noun subclusters...")
    noun_subclusters = None
    sc_path = _results_dir() / 'noun_subclusters.json'
    if sc_path.exists():
        with open(sc_path) as f:
            sc_data = json.load(f)
        noun_subclusters = sc_data.get('subclusters', [])
        print(f"  Loaded {len(noun_subclusters)} subclusters")
    else:
        print("  WARNING: noun_subclusters.json not found. "
              "Object compatibility will use defaults.")

    # Build compatibility matrix
    print("\n  Building compatibility matrix...")
    latin_imperatives = LATIN_PHARMACEUTICAL_IMPERATIVES
    matrix, v_stems, l_verbs = build_compatibility_matrix(
        profiles, latin_imperatives, char_profiles, noun_subclusters,
    )
    print(f"  Matrix shape: {matrix.shape[0]} x {matrix.shape[1]}")
    print(f"  Score range: [{matrix.min():.3f}, {matrix.max():.3f}]")

    # Hungarian assignment
    print("\n  Running Hungarian assignment...")
    assignments, second_total = run_hungarian_assignment(
        matrix, v_stems, l_verbs, latin_imperatives,
    )

    total_score = sum(a.total_score for a in assignments)
    n_confident = sum(1 for a in assignments if a.is_confident)
    gap = (total_score - second_total) / total_score if total_score > 0 else 0.0

    print(f"\n  {'Voynich':<12s} {'Latin':<12s} {'Meaning':<10s} "
          f"{'Score':<8s} {'Confident'}")
    print(f"  {'─' * 12} {'─' * 12} {'─' * 10} {'─' * 8} {'─' * 9}")
    for a in sorted(assignments, key=lambda x: -x.total_score):
        conf = 'YES' if a.is_confident else 'no'
        print(f"  {a.voynich_stem:<12s} {a.latin_verb:<12s} "
              f"{a.latin_meaning:<10s} {a.total_score:<8.3f} {conf}")

    print(f"\n  Total score: {total_score:.3f}")
    print(f"  Second-best total: {second_total:.3f}")
    print(f"  Gap: {gap:.3f}")
    print(f"  Confident assignments: {n_confident}/{len(assignments)}")

    # Cross-consistency check
    print("\n  Checking character mapping consistency...")
    n_cons, n_test, cons_rate = check_verb_char_consistency(
        assignments, char_profiles,
    )
    print(f"  Tested: {n_test}, Consistent: {n_cons}, Rate: {cons_rate:.3f}")

    # Null test
    print("\n  Running null test (100 trials)...")
    null_mean, null_std, asgn_sel = _null_test_assignment(
        matrix, total_score, n_trials=100,
    )
    print(f"  Null mean score: {null_mean:.3f} +/- {null_std:.3f}")
    print(f"  Assignment selectivity: {asgn_sel:.2f}x")

    # Gate
    gate_passed = asgn_sel > 1.5 and cons_rate > 0.5
    if gate_passed:
        verdict = 'verb_identification_significant'
    elif asgn_sel > 1.5:
        verdict = 'verb_matching_good_char_consistency_low'
    elif cons_rate > 0.5:
        verdict = 'char_consistency_good_matching_weak'
    else:
        verdict = 'verb_identification_not_significant'

    print(f"\n  Gate: {'PASS' if gate_passed else 'FAIL'}")
    print(f"  Verdict: {verdict}")

    mean_score = total_score / len(assignments) if assignments else 0.0

    result = VerbIdentificationResult(
        n_voynich_verbs=len(profiles),
        n_latin_imperatives=len(latin_imperatives),
        voynich_verb_profiles=[_convert(asdict(vp)) for vp in profiles],
        compatibility_matrix=matrix.tolist(),
        voynich_stems=v_stems,
        latin_verbs=l_verbs,
        assignments=[_convert(asdict(a)) for a in assignments],
        mean_assignment_score=mean_score,
        best_total_score=total_score,
        second_best_total_score=second_total,
        assignment_gap=gap,
        n_confident_assignments=n_confident,
        n_char_mappings_tested=n_test,
        n_char_mappings_consistent=n_cons,
        char_consistency_rate=cons_rate,
        null_mean_score=null_mean,
        null_std_score=null_std,
        assignment_selectivity=asgn_sel,
        gate_passed=gate_passed,
        verdict=verdict,
    )

    out = _convert(asdict(result))
    out_path = _results_dir() / 'verb_identification.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {out_path}")
    return out
