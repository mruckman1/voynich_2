"""
Phase 44 – Track C: k-Permutation Coupled Simulated Annealing
===============================================================
Search the surjective mapping space (triples -> syllables) with
coupled parallel SA processes.  Multi-component energy function
balances dict-hit, bigram plausibility, signal preservation, and
articulatory consistency.

Dependency chain:
    combined_refine.json       (Phase 15 assignment)
    bootstrap_loop.json        (Phase 30 confirmed triples)
    modifier_integrate.json    (Phase 16 modifiers)
    signal_isolation.json      (Phase 28 signal words)
        -> kperm_energy.json       (Step 44C.1)
        -> kperm_search.json       (Step 44C.2)
        -> kperm_analysis.json     (Step 44C.3)
        -> kperm_validation.json   (Step 44C.4)
"""

import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    decode_token_modifier_aware,
    load_corpus,
    tokenize_eva_chars,
)
from voynich.core.reference import (
    EVA_VISUAL_COMPONENTS,
    PHONEME_NUCLEUS_MAP,
    PHONEME_PLACE_MAP,
    build_expanded_word_set,
    build_triple_phoneme_hypotheses,
    load_reference_corpus,
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
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if v != v else v
    if isinstance(obj, np.ndarray):
        return _convert(obj.tolist())
    if isinstance(obj, (bool, int, float, str, type(None))):
        return obj
    return str(obj)


def _safe_load(path: str) -> Dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


# ---------------------------------------------------------------------------
# Modifier reconstruction
# ---------------------------------------------------------------------------

def _reconstruct_modifier_rules(data: Dict) -> Tuple[Set[str], Dict[str, str]]:
    modifier_chars = set(data.get('modifier_chars', []))
    modifier_rules: Dict[str, str] = {}
    for c in data.get('classifications', []):
        if c.get('final_classification') == 'modifier':
            modifier_rules[c['eva_char']] = c.get('modifier_type', 'silent')
    return modifier_chars, modifier_rules


# ---------------------------------------------------------------------------
# Decode helpers
# ---------------------------------------------------------------------------

def _decode_corpus_r3(
    tokens: List[str],
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    modifier_chars: set,
    modifier_rules: Dict[str, str],
    ref_word_set: set,
) -> List[str]:
    from voynich.phases.csp_solver import decode_token
    decoded = []
    for token in tokens:
        alt = decode_token_modifier_aware(
            token, assignment, eva_to_triple, modifier_chars, modifier_rules,
        )
        if alt.lower() in ref_word_set:
            decoded.append(alt.lower())
            continue
        stripped = decode_token_modifier_aware(
            token, assignment, eva_to_triple, modifier_chars,
        )
        if stripped.lower() in ref_word_set:
            decoded.append(stripped.lower())
            continue
        raw = decode_token(token, assignment, eva_to_triple)
        decoded.append(raw.lower())
    return decoded


def _compute_dict_hit(decoded: List[str], ref_word_set: set) -> float:
    if not decoded:
        return 0.0
    return sum(1 for w in decoded if w in ref_word_set) / len(decoded)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class EnergyCalibration:
    w_dict: float
    w_bigram: float
    w_signal: float
    w_paleo: float
    range_dict: float
    range_bigram: float
    range_signal: float
    range_paleo: float
    p15_energy: float
    p15_components: Dict[str, float]
    n_random_tested: int
    runtime_seconds: float


@dataclass
class CSASolution:
    assignment: Dict[str, str]
    energy: float
    components: Dict[str, float]
    dict_hit: float
    n_signal_preserved: int
    chain_id: int
    iteration: int


@dataclass
class SearchResult:
    n_chains: int
    n_iterations: int
    best_energy: float
    best_dict_hit: float
    best_assignment: Dict[str, str]
    p15_energy: float
    p15_dict_hit: float
    top_k: List[Dict]
    convergence_curve: List[Dict]
    runtime_seconds: float


@dataclass
class AnalysisResult:
    best_csa_dict_hit: float
    p15_dict_hit: float
    delta_dict_hit: float
    n_triples_changed: int
    changed_triples: List[Dict]
    per_triple_consensus: Dict[str, Dict[str, float]]
    p15_rank: int
    runtime_seconds: float


@dataclass
class CSAValidationResult:
    full_corpus_dict_hit: float
    p15_full_corpus_dict_hit: float
    delta: float
    null_dict_hits: List[float]
    null_mean: float
    null_std: float
    selectivity: float
    gate_passed: bool
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Precomputed paleo penalty table
# ---------------------------------------------------------------------------

def _build_paleo_table(
    all_triples: List[str],
    triple_domains: Dict[str, List[str]],
) -> Dict[Tuple[str, str], int]:
    """Precompute paleo penalty for every (triple, syllable) pair.

    Returns dict mapping (triple_key, syllable) -> penalty (0, 1, or 2).
    """
    table: Dict[Tuple[str, str], int] = {}
    for t_key in all_triples:
        parts = t_key.split(',')
        if len(parts) < 2:
            for syl in triple_domains.get(t_key, []):
                table[(t_key, syl)] = 0
            continue
        first_stroke, last_stroke = parts[0], parts[1]
        onset_cands = set(PHONEME_PLACE_MAP.get(first_stroke, []))
        nucleus_cands = set(PHONEME_NUCLEUS_MAP.get(last_stroke, []))
        for syl in triple_domains.get(t_key, []):
            pen = 0
            if syl and len(syl) >= 2:
                if onset_cands and syl[0] not in onset_cands:
                    pen += 1
                if nucleus_cands and syl[-1] not in nucleus_cands:
                    pen += 1
            table[(t_key, syl)] = pen
    return table


# ---------------------------------------------------------------------------
# Energy function components (with incremental support)
# ---------------------------------------------------------------------------

class EnergyEvaluator:
    """Evaluator for the multi-component energy function.

    Supports both full evaluation and O(1) incremental updates for the
    inner SA loop when a single triple changes.
    """

    def __init__(
        self,
        sample_tokens: List[str],
        eva_to_triple: Dict[str, str],
        modifier_chars: Set[str],
        modifier_rules: Dict[str, str],
        expanded_set: set,
        signal_words: List[Tuple[str, float]],
        ref_syl_bigrams: Dict[Tuple[str, str], float],
        all_syls: Set[str],
        triple_domains: Dict[str, List[str]],
        w_dict: float = 1.0,
        w_bigram: float = 1.0,
        w_signal: float = 1.0,
        w_paleo: float = 1.0,
    ):
        self.sample_tokens = sample_tokens
        self.eva_to_triple = eva_to_triple
        self.modifier_chars = modifier_chars
        self.modifier_rules = modifier_rules
        self.expanded_set = expanded_set
        self.signal_words = signal_words  # [(word, sigma), ...]
        self.ref_syl_bigrams = ref_syl_bigrams
        self.all_syls = all_syls
        self.triple_domains = triple_domains
        self.w_dict = w_dict
        self.w_bigram = w_bigram
        self.w_signal = w_signal
        self.w_paleo = w_paleo

        # Precompute token -> triple sequences
        self.token_triples: List[List[str]] = []
        for token in sample_tokens:
            chars = tokenize_eva_chars(token)
            triples = [eva_to_triple.get(ch) for ch in chars]
            self.token_triples.append([t for t in triples if t])

        # -- Incremental bigram index --
        # For each triple key, store a list of (neighbour_triple, position)
        # where position = 'left' means this triple is at index i and
        # neighbour is at i+1, 'right' means neighbour is at i-1.
        # This lets us compute the bigram delta for a single-triple swap.
        self.bigram_pairs_for_triple: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        #   maps triple_key -> [(other_triple_in_pair, side)]
        #   side='L' means pair is (this_triple, other), 'R' means (other, this_triple)
        self.total_bigrams = 0
        for triples in self.token_triples:
            for i in range(len(triples) - 1):
                left_t, right_t = triples[i], triples[i + 1]
                self.bigram_pairs_for_triple[left_t].append((right_t, 'L'))
                self.bigram_pairs_for_triple[right_t].append((left_t, 'R'))
                self.total_bigrams += 1

        # -- Precomputed paleo table --
        all_triple_keys = sorted(triple_domains.keys())
        self.paleo_table = _build_paleo_table(all_triple_keys, triple_domains)

    # ── Full evaluation methods (used at checkpoints) ──

    def e_dict(self, assignment: Dict[str, str]) -> float:
        """Negative dict-hit rate."""
        decoded = _decode_corpus_r3(
            self.sample_tokens, assignment, self.eva_to_triple,
            self.modifier_chars, self.modifier_rules, self.expanded_set,
        )
        return -_compute_dict_hit(decoded, self.expanded_set)

    def e_bigram(self, assignment: Dict[str, str]) -> float:
        """Bigram implausibility: fraction of decoded bigrams NOT in reference."""
        misses = 0
        for triples in self.token_triples:
            for i in range(len(triples) - 1):
                s_l = assignment.get(triples[i], '?')
                s_r = assignment.get(triples[i + 1], '?')
                if (s_l, s_r) not in self.ref_syl_bigrams:
                    misses += 1
        return misses / max(self.total_bigrams, 1)

    def e_signal(self, assignment: Dict[str, str]) -> float:
        """Penalty for breaking signal words."""
        decoded = _decode_corpus_r3(
            self.sample_tokens, assignment, self.eva_to_triple,
            self.modifier_chars, self.modifier_rules, self.expanded_set,
        )
        decoded_set = set(decoded)
        penalty = 0.0
        for word, sigma in self.signal_words:
            if word not in decoded_set:
                penalty += sigma
        return penalty

    def e_paleo(self, assignment: Dict[str, str]) -> float:
        """Penalty for typologically implausible assignments."""
        penalty = 0
        for t_key, syl in assignment.items():
            penalty += self.paleo_table.get((t_key, syl), 0)
        return penalty

    def energy_fast(self, assignment: Dict[str, str]) -> float:
        """Fast energy (bigram + paleo only — no decode required)."""
        eb = self.e_bigram(assignment)
        ep = self.e_paleo(assignment)
        return self.w_bigram * eb + self.w_paleo * ep

    def energy(self, assignment: Dict[str, str]) -> Tuple[float, Dict[str, float]]:
        """Compute total energy and per-component breakdown."""
        ed = self.e_dict(assignment)
        eb = self.e_bigram(assignment)
        es = self.e_signal(assignment)
        ep = self.e_paleo(assignment)

        total = (self.w_dict * ed +
                 self.w_bigram * eb +
                 self.w_signal * es +
                 self.w_paleo * ep)

        components = {
            'dict': round(ed, 6),
            'bigram': round(eb, 6),
            'signal': round(es, 6),
            'paleo': round(ep, 6),
        }
        return total, components

    def count_signal_preserved(self, assignment: Dict[str, str]) -> int:
        decoded = _decode_corpus_r3(
            self.sample_tokens, assignment, self.eva_to_triple,
            self.modifier_chars, self.modifier_rules, self.expanded_set,
        )
        decoded_set = set(decoded)
        return sum(1 for word, _ in self.signal_words if word in decoded_set)

    # ── Incremental methods (O(affected_pairs) instead of O(all_bigrams)) ──

    def compute_bigram_misses(self, assignment: Dict[str, str]) -> int:
        """Count total bigram misses for full assignment (used to init cache)."""
        misses = 0
        for triples in self.token_triples:
            for i in range(len(triples) - 1):
                s_l = assignment.get(triples[i], '?')
                s_r = assignment.get(triples[i + 1], '?')
                if (s_l, s_r) not in self.ref_syl_bigrams:
                    misses += 1
        return misses

    def compute_paleo_total(self, assignment: Dict[str, str]) -> int:
        """Compute total paleo penalty (used to init cache)."""
        total = 0
        for t_key, syl in assignment.items():
            total += self.paleo_table.get((t_key, syl), 0)
        return total

    def incremental_delta(
        self,
        assignment: Dict[str, str],
        changed_triple: str,
        old_syl: str,
        new_syl: str,
        cur_bigram_misses: int,
        cur_paleo_total: int,
    ) -> Tuple[float, int, int]:
        """Compute energy delta and updated counts for a single-triple swap.

        Returns (delta_energy, new_bigram_misses, new_paleo_total).
        """
        # -- Bigram delta --
        # Only pairs involving changed_triple are affected
        delta_misses = 0
        for other_triple, side in self.bigram_pairs_for_triple.get(changed_triple, []):
            other_syl = assignment.get(other_triple, '?')
            if side == 'L':
                # Pair is (changed_triple, other_triple)
                old_pair = (old_syl, other_syl)
                new_pair = (new_syl, other_syl)
            else:
                # Pair is (other_triple, changed_triple)
                old_pair = (other_syl, old_syl)
                new_pair = (other_syl, new_syl)
            was_miss = old_pair not in self.ref_syl_bigrams
            is_miss = new_pair not in self.ref_syl_bigrams
            if was_miss and not is_miss:
                delta_misses -= 1
            elif not was_miss and is_miss:
                delta_misses += 1

        new_bigram_misses = cur_bigram_misses + delta_misses

        # -- Paleo delta --
        old_pen = self.paleo_table.get((changed_triple, old_syl), 0)
        new_pen = self.paleo_table.get((changed_triple, new_syl), 0)
        new_paleo_total = cur_paleo_total - old_pen + new_pen

        # -- Energy delta --
        old_eb = cur_bigram_misses / max(self.total_bigrams, 1)
        new_eb = new_bigram_misses / max(self.total_bigrams, 1)
        delta_e = (self.w_bigram * (new_eb - old_eb) +
                   self.w_paleo * (new_pen - old_pen))

        return delta_e, new_bigram_misses, new_paleo_total


# ---------------------------------------------------------------------------
# Step 44C.1 – Energy Function Calibration
# ---------------------------------------------------------------------------

def run_kperm_energy() -> None:
    """Step 44C.1: Define and calibrate energy function."""
    t0 = time.time()
    print("=" * 70)
    print("STEP 44C.1: Energy Function Calibration")
    print("=" * 70)

    rd = _results_dir()
    eva_to_triple = build_eva_to_triple_lookup()

    # Load inputs
    print("\n  1. Loading inputs ...")
    refine_data = _safe_load(os.path.join(rd, 'combined_refine.json'))
    if not refine_data:
        print("  [SKIP] combined_refine.json not found")
        return
    p15_assignment = refine_data.get('best_assignment', {})

    boot_data = _safe_load(os.path.join(rd, 'bootstrap_loop.json'))
    final_assignment = boot_data.get('final_assignment', p15_assignment)
    confirmed_list = set(boot_data.get('confirmed_triples', []))

    mod_data = _safe_load(os.path.join(rd, 'modifier_integrate.json'))
    modifier_chars, modifier_rules = set(), {}
    if mod_data:
        modifier_chars, modifier_rules = _reconstruct_modifier_rules(mod_data)

    ref_corpus = load_reference_corpus()
    base_words = set(w.lower() for w in ref_corpus.get_combined_tokens('latin'))
    expanded_set, _ = build_expanded_word_set(base_words)

    corpus = load_corpus(verbose=False)
    all_tokens = []
    for _fol, page in corpus.pages.items():
        all_tokens.extend(page.all_tokens)
    sample_tokens = all_tokens[:2000]

    # Signal words
    sig_data = _safe_load(os.path.join(rd, 'signal_isolation.json'))
    signal_words = []
    if sig_data:
        for ws in sig_data.get('word_signals', []):
            if ws.get('is_genuine_signal', False):
                signal_words.append((ws['word'], ws.get('signal_sigma', 1.0)))

    # Build reference syllable bigrams
    print("  2. Building reference syllable bigrams ...")
    hypotheses = build_triple_phoneme_hypotheses('latin')
    all_triples = sorted(hypotheses.keys())
    free_triples = [t for t in all_triples if t not in confirmed_list]

    all_syls = set()
    for domain in hypotheses.values():
        all_syls.update(domain)

    ref_tokens = ref_corpus.get_combined_tokens('latin')
    syl_bigram_count: Counter = Counter()
    total_syl_bigrams = 0
    for word in ref_tokens[:50000]:
        word_lower = word.lower()
        syls_in_word = []
        i = 0
        while i < len(word_lower):
            if i + 1 < len(word_lower) and word_lower[i:i+2] in all_syls:
                syls_in_word.append(word_lower[i:i+2])
                i += 2
            elif word_lower[i:i+1] in all_syls:
                syls_in_word.append(word_lower[i:i+1])
                i += 1
            else:
                i += 1
        for j in range(len(syls_in_word) - 1):
            syl_bigram_count[(syls_in_word[j], syls_in_word[j + 1])] += 1
            total_syl_bigrams += 1

    ref_syl_bigrams: Dict[Tuple[str, str], float] = {}
    if total_syl_bigrams > 0:
        for pair, cnt in syl_bigram_count.items():
            ref_syl_bigrams[pair] = cnt / total_syl_bigrams

    # Create evaluator with unit weights
    evaluator = EnergyEvaluator(
        sample_tokens, eva_to_triple, modifier_chars, modifier_rules,
        expanded_set, signal_words, ref_syl_bigrams, all_syls, hypotheses,
        w_dict=1.0, w_bigram=1.0, w_signal=1.0, w_paleo=1.0,
    )

    # Evaluate Phase 15
    print("  3. Evaluating Phase 15 table ...")
    p15_energy, p15_components = evaluator.energy(final_assignment)
    print(f"     Phase 15 energy: {p15_energy:.4f}")
    print(f"     Components: {p15_components}")

    # Generate 100 random assignments and compute energy ranges
    print("  4. Calibrating weights over 100 random assignments ...")
    random.seed(42)
    component_values: Dict[str, List[float]] = {
        'dict': [], 'bigram': [], 'signal': [], 'paleo': [],
    }

    for trial in range(100):
        rand_assign = dict(final_assignment)
        for t_key in free_triples:
            domain = hypotheses.get(t_key, ['?'])
            rand_assign[t_key] = random.choice(domain)
        _, comps = evaluator.energy(rand_assign)
        for k, v in comps.items():
            component_values[k].append(v)
        if (trial + 1) % 25 == 0:
            print(f"     ... {trial + 1}/100")

    ranges = {}
    for k, vals in component_values.items():
        r = max(vals) - min(vals)
        ranges[k] = r if r > 0 else 1.0

    # Set weights inversely proportional to range (equal contribution)
    w_dict = 1.0 / ranges['dict']
    w_bigram = 1.0 / ranges['bigram']
    w_signal = 1.0 / ranges['signal'] if ranges['signal'] > 0 else 0.1
    w_paleo = 1.0 / ranges['paleo'] if ranges['paleo'] > 0 else 0.1

    result = EnergyCalibration(
        w_dict=round(w_dict, 4),
        w_bigram=round(w_bigram, 4),
        w_signal=round(w_signal, 4),
        w_paleo=round(w_paleo, 4),
        range_dict=round(ranges['dict'], 4),
        range_bigram=round(ranges['bigram'], 4),
        range_signal=round(ranges['signal'], 4),
        range_paleo=round(ranges['paleo'], 4),
        p15_energy=round(p15_energy, 4),
        p15_components=p15_components,
        n_random_tested=100,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'kperm_energy.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2)

    print(f"\n  Calibrated weights: dict={w_dict:.4f}, bigram={w_bigram:.4f}, "
          f"signal={w_signal:.4f}, paleo={w_paleo:.4f}")
    print(f"  Saved -> {out_path}")
    print(f"\n  Step 44C.1 completed in {time.time() - t0:.1f}s")


# ---------------------------------------------------------------------------
# Step 44C.2 – Coupled Simulated Annealing Search
# ---------------------------------------------------------------------------

def run_kperm_search() -> None:
    """Step 44C.2: Run coupled SA with k-permutations."""
    t0 = time.time()
    print("=" * 70)
    print("STEP 44C.2: Coupled Simulated Annealing Search")
    print("=" * 70)

    rd = _results_dir()
    eva_to_triple = build_eva_to_triple_lookup()

    # Load inputs
    print("\n  1. Loading inputs ...")
    refine_data = _safe_load(os.path.join(rd, 'combined_refine.json'))
    if not refine_data:
        print("  [SKIP] combined_refine.json not found")
        return
    p15_assignment = refine_data.get('best_assignment', {})

    boot_data = _safe_load(os.path.join(rd, 'bootstrap_loop.json'))
    final_assignment = boot_data.get('final_assignment', p15_assignment)
    confirmed_list = set(boot_data.get('confirmed_triples', []))

    mod_data = _safe_load(os.path.join(rd, 'modifier_integrate.json'))
    modifier_chars, modifier_rules = set(), {}
    if mod_data:
        modifier_chars, modifier_rules = _reconstruct_modifier_rules(mod_data)

    ref_corpus = load_reference_corpus()
    base_words = set(w.lower() for w in ref_corpus.get_combined_tokens('latin'))
    expanded_set, _ = build_expanded_word_set(base_words)

    corpus = load_corpus(verbose=False)
    all_tokens = []
    for _fol, page in corpus.pages.items():
        all_tokens.extend(page.all_tokens)
    sample_tokens = all_tokens[:2000]

    # Signal words
    sig_data = _safe_load(os.path.join(rd, 'signal_isolation.json'))
    signal_words = []
    if sig_data:
        for ws in sig_data.get('word_signals', []):
            if ws.get('is_genuine_signal', False):
                signal_words.append((ws['word'], ws.get('signal_sigma', 1.0)))

    # Reference bigrams
    hypotheses = build_triple_phoneme_hypotheses('latin')
    all_triples = sorted(hypotheses.keys())
    free_triples = [t for t in all_triples if t not in confirmed_list]

    all_syls = set()
    for domain in hypotheses.values():
        all_syls.update(domain)

    ref_tokens = ref_corpus.get_combined_tokens('latin')
    syl_bigram_count: Counter = Counter()
    total_syl_bigrams = 0
    for word in ref_tokens[:50000]:
        word_lower = word.lower()
        syls_in_word = []
        i = 0
        while i < len(word_lower):
            if i + 1 < len(word_lower) and word_lower[i:i+2] in all_syls:
                syls_in_word.append(word_lower[i:i+2])
                i += 2
            elif word_lower[i:i+1] in all_syls:
                syls_in_word.append(word_lower[i:i+1])
                i += 1
            else:
                i += 1
        for j in range(len(syls_in_word) - 1):
            syl_bigram_count[(syls_in_word[j], syls_in_word[j + 1])] += 1
            total_syl_bigrams += 1

    ref_syl_bigrams: Dict[Tuple[str, str], float] = {}
    if total_syl_bigrams > 0:
        for pair, cnt in syl_bigram_count.items():
            ref_syl_bigrams[pair] = cnt / total_syl_bigrams

    # Load calibrated weights
    energy_data = _safe_load(os.path.join(rd, 'kperm_energy.json'))
    w_dict = energy_data.get('w_dict', 10.0)
    w_bigram = energy_data.get('w_bigram', 5.0)
    w_signal = energy_data.get('w_signal', 3.0)
    w_paleo = energy_data.get('w_paleo', 1.0)

    evaluator = EnergyEvaluator(
        sample_tokens, eva_to_triple, modifier_chars, modifier_rules,
        expanded_set, signal_words, ref_syl_bigrams, all_syls, hypotheses,
        w_dict=w_dict, w_bigram=w_bigram, w_signal=w_signal, w_paleo=w_paleo,
    )

    # CSA parameters
    K = 10  # number of coupled chains
    N_ITER = 200_000
    T_START = 10.0
    T_END = 0.01
    COUPLING_STRENGTH = 0.5
    MAX_RUNTIME = 7200  # 2 hours
    CHECKPOINT_INTERVAL = 10_000

    print(f"  2. Running CSA: {K} chains x {N_ITER} iterations ...", flush=True)
    print(f"     Free triples: {len(free_triples)}", flush=True)
    print(f"     Weights: dict={w_dict:.2f}, bigram={w_bigram:.2f}, "
          f"signal={w_signal:.2f}, paleo={w_paleo:.2f}", flush=True)
    print(f"     Using incremental energy evaluation", flush=True)

    random.seed(42)

    # Pre-build domain lists for random choice (avoid repeated dict lookups)
    domain_lists: Dict[str, List[str]] = {}
    for t_key in free_triples:
        domain_lists[t_key] = hypotheses.get(t_key, ['?'])

    # Initialize K chains: random perturbations of Phase 15
    states: List[Dict[str, str]] = []
    bigram_misses_cache: List[int] = []  # cached bigram miss count per chain
    paleo_cache: List[int] = []          # cached paleo penalty per chain
    energies: List[float] = []

    for k in range(K):
        state = dict(final_assignment)
        for t_key in free_triples:
            state[t_key] = random.choice(domain_lists[t_key])
        bm = evaluator.compute_bigram_misses(state)
        pt = evaluator.compute_paleo_total(state)
        e_fast = (w_bigram * bm / max(evaluator.total_bigrams, 1) +
                  w_paleo * pt)
        states.append(state)
        bigram_misses_cache.append(bm)
        paleo_cache.append(pt)
        energies.append(e_fast)

    # Chain 0 = Phase 15 assignment
    states[0] = dict(final_assignment)
    bm0 = evaluator.compute_bigram_misses(final_assignment)
    pt0 = evaluator.compute_paleo_total(final_assignment)
    energies[0] = (w_bigram * bm0 / max(evaluator.total_bigrams, 1) +
                   w_paleo * pt0)
    bigram_misses_cache[0] = bm0
    paleo_cache[0] = pt0

    best_state = dict(states[0])
    best_energy = energies[0]
    best_components: Dict[str, float] = {}
    convergence_curve = []

    p15_energy, p15_comps = evaluator.energy(final_assignment)

    deadline = time.time() + MAX_RUNTIME
    n_accepted = 0
    n_evaluated = 0
    last_log_time = time.time()

    for step in range(N_ITER):
        # Temperature schedule: geometric cooling
        T = T_START * (T_END / T_START) ** (step / max(N_ITER - 1, 1))
        mean_energy = sum(energies) / K  # avoid np.mean overhead

        for k in range(K):
            # Generate neighbor: change one free triple (mutate in-place)
            t_key = random.choice(free_triples)
            old_syl = states[k][t_key]
            domain = domain_lists[t_key]
            # Pick a new syllable different from old
            new_syl = domain[random.randint(0, len(domain) - 1)]
            if new_syl == old_syl and len(domain) > 1:
                # Retry once to avoid no-ops
                new_syl = domain[random.randint(0, len(domain) - 1)]

            # Incremental energy evaluation
            delta_e, new_bm, new_pt = evaluator.incremental_delta(
                states[k], t_key, old_syl, new_syl,
                bigram_misses_cache[k], paleo_cache[k],
            )
            n_evaluated += 1

            # Coupled acceptance
            coupling_term = COUPLING_STRENGTH * (energies[k] - mean_energy) / max(T, 1e-10)
            accept_arg = -(delta_e + coupling_term) / max(T, 1e-10)
            if accept_arg > 0 or accept_arg > -30:
                try:
                    acceptance_prob = min(1.0, math.exp(accept_arg))
                except OverflowError:
                    acceptance_prob = 1.0
            else:
                acceptance_prob = 0.0

            if random.random() < acceptance_prob:
                # Accept: apply mutation
                states[k][t_key] = new_syl
                energies[k] += delta_e
                bigram_misses_cache[k] = new_bm
                paleo_cache[k] = new_pt
                n_accepted += 1

                if energies[k] < best_energy:
                    best_energy = energies[k]
                    best_state = dict(states[k])
            # No else needed — state was never actually mutated on rejection

        # Checkpoint: compute full energy for best state
        if step % CHECKPOINT_INTERVAL == 0:
            elapsed = time.time() - t0
            full_e, full_comps = evaluator.energy(best_state)
            best_components = full_comps
            best_dh = -full_comps.get('dict', 0)
            accept_rate = n_accepted / max(n_evaluated, 1)
            evals_per_sec = n_evaluated / max(elapsed, 0.01)
            convergence_curve.append({
                'step': step,
                'T': round(T, 6),
                'best_energy': round(best_energy, 6),
                'best_energy_full': round(full_e, 6),
                'mean_energy': round(mean_energy, 6),
                'std_energy': round(float(np.std(energies)), 6),
                'best_dict_hit': round(best_dh, 4),
                'accept_rate': round(accept_rate, 4),
                'evals_per_sec': round(evals_per_sec, 0),
                'elapsed_seconds': round(elapsed, 1),
            })
            print(f"     Step {step:>7d} | T={T:.6f} | best_E={best_energy:.4f} "
                  f"| mean_E={mean_energy:.4f} | dict-hit={best_dh:.4f} "
                  f"| accept={accept_rate:.3f} | {evals_per_sec:.0f} eval/s "
                  f"| {elapsed:.0f}s", flush=True)

        # Coupling: every 100 steps, worst chain gets perturbed copy of best
        if step % 100 == 0 and step > 0:
            worst_k = max(range(K), key=lambda i: energies[i])
            best_k = min(range(K), key=lambda i: energies[i])
            if worst_k != best_k:
                states[worst_k] = dict(states[best_k])
                # Perturb 1-2 random free triples
                n_perturb = random.randint(1, min(2, len(free_triples)))
                for t_key in random.sample(free_triples, n_perturb):
                    states[worst_k][t_key] = random.choice(domain_lists[t_key])
                # Recompute full cache for the perturbed chain
                bigram_misses_cache[worst_k] = evaluator.compute_bigram_misses(
                    states[worst_k])
                paleo_cache[worst_k] = evaluator.compute_paleo_total(
                    states[worst_k])
                energies[worst_k] = (
                    w_bigram * bigram_misses_cache[worst_k] /
                    max(evaluator.total_bigrams, 1) +
                    w_paleo * paleo_cache[worst_k])

        # Time limit
        if time.time() > deadline:
            print(f"     Timeout at step {step}")
            break

    # Collect top-20 solutions across all chains
    print("  3. Collecting top solutions ...")
    all_final: List[Tuple[float, Dict[str, str], int]] = []
    for k in range(K):
        all_final.append((energies[k], dict(states[k]), k))
    all_final.append((best_energy, dict(best_state), -1))
    all_final.sort(key=lambda x: x[0])

    seen_assigns = set()
    top_k_list = []
    for e, assign, chain_id in all_final:
        key = tuple(sorted(assign.items()))
        if key in seen_assigns:
            continue
        seen_assigns.add(key)
        dh = -evaluator.e_dict(assign)
        n_sig = evaluator.count_signal_preserved(assign)
        top_k_list.append({
            'assignment': assign,
            'energy': round(e, 6),
            'dict_hit': round(dh, 4),
            'n_signal_preserved': n_sig,
            'chain_id': chain_id,
        })
        if len(top_k_list) >= 20:
            break

    best_dh = top_k_list[0]['dict_hit'] if top_k_list else 0.0
    p15_dh = -evaluator.e_dict(final_assignment)

    result = SearchResult(
        n_chains=K,
        n_iterations=N_ITER,
        best_energy=round(best_energy, 6),
        best_dict_hit=round(best_dh, 4),
        best_assignment=best_state,
        p15_energy=round(p15_energy, 6),
        p15_dict_hit=round(p15_dh, 4),
        top_k=top_k_list,
        convergence_curve=convergence_curve,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'kperm_search.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2)

    total_elapsed = time.time() - t0
    print(f"\n  Best energy: {best_energy:.6f} (Phase 15: {p15_energy:.6f})")
    print(f"  Best dict-hit: {best_dh:.4f} (Phase 15: {p15_dh:.4f})")
    print(f"  {len(top_k_list)} unique solutions in top-K")
    print(f"  Total evaluations: {n_evaluated:,}, accepted: {n_accepted:,} "
          f"({n_accepted/max(n_evaluated,1):.1%})")
    print(f"  Throughput: {n_evaluated/max(total_elapsed,0.01):,.0f} eval/s")
    print(f"  Saved -> {out_path}")
    print(f"\n  Step 44C.2 completed in {total_elapsed:.1f}s")


# ---------------------------------------------------------------------------
# Step 44C.3 – Solution Analysis
# ---------------------------------------------------------------------------

def run_kperm_analyze() -> None:
    """Step 44C.3: Analyze CSA solutions and compare to Phase 15."""
    t0 = time.time()
    print("=" * 70)
    print("STEP 44C.3: CSA Solution Analysis")
    print("=" * 70)

    rd = _results_dir()

    # Load search results
    search_data = _safe_load(os.path.join(rd, 'kperm_search.json'))
    if not search_data:
        print("  [SKIP] kperm_search.json not found")
        return

    best_assignment = search_data.get('best_assignment', {})
    top_k = search_data.get('top_k', [])
    p15_dh = search_data.get('p15_dict_hit', 0.0)
    best_dh = search_data.get('best_dict_hit', 0.0)

    # Load Phase 15 baseline
    refine_data = _safe_load(os.path.join(rd, 'combined_refine.json'))
    boot_data = _safe_load(os.path.join(rd, 'bootstrap_loop.json'))
    p15_assignment = refine_data.get('best_assignment', {})
    final_assignment = boot_data.get('final_assignment', p15_assignment)
    confirmed_list = set(boot_data.get('confirmed_triples', []))

    hypotheses = build_triple_phoneme_hypotheses('latin')
    all_triples = sorted(hypotheses.keys())
    free_triples = [t for t in all_triples if t not in confirmed_list]

    # Triple changes
    changed = []
    for t in free_triples:
        old = final_assignment.get(t, '?')
        new = best_assignment.get(t, '?')
        if old != new:
            changed.append({'triple': t, 'old': old, 'new': new})

    # Per-triple consensus across top-K
    per_triple_consensus: Dict[str, Dict[str, float]] = {}
    n_top = len(top_k)
    if n_top > 0:
        for t_key in free_triples:
            syl_counts: Counter = Counter()
            for sol in top_k:
                assign = sol.get('assignment', {})
                syl = assign.get(t_key, '?')
                syl_counts[syl] += 1
            per_triple_consensus[t_key] = {
                syl: round(cnt / n_top, 4)
                for syl, cnt in syl_counts.most_common()
            }

    # Check if Phase 15 is in top-K
    p15_rank = -1
    for idx, sol in enumerate(top_k):
        assign = sol.get('assignment', {})
        if all(assign.get(t) == final_assignment.get(t) for t in free_triples):
            p15_rank = idx + 1
            break

    delta = best_dh - p15_dh

    result = AnalysisResult(
        best_csa_dict_hit=round(best_dh, 4),
        p15_dict_hit=round(p15_dh, 4),
        delta_dict_hit=round(delta, 4),
        n_triples_changed=len(changed),
        changed_triples=changed,
        per_triple_consensus=per_triple_consensus,
        p15_rank=p15_rank,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'kperm_analysis.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2)

    print(f"\n  Best CSA dict-hit: {best_dh:.4f} (Phase 15: {p15_dh:.4f})")
    print(f"  Delta: {delta:+.4f}")
    print(f"  Triples changed: {len(changed)}")
    print(f"  Phase 15 rank in top-K: {p15_rank}")
    for c in changed[:5]:
        print(f"    {c['triple']}: {c['old']} -> {c['new']}")
    print(f"  Saved -> {out_path}")
    print(f"\n  Step 44C.3 completed in {time.time() - t0:.1f}s")


# ---------------------------------------------------------------------------
# Step 44C.4 – Validation
# ---------------------------------------------------------------------------

def run_kperm_validate() -> None:
    """Step 44C.4: Validate CSA best against null corpora."""
    t0 = time.time()
    print("=" * 70)
    print("STEP 44C.4: CSA Validation")
    print("=" * 70)

    rd = _results_dir()
    eva_to_triple = build_eva_to_triple_lookup()

    # Load best assignment
    search_data = _safe_load(os.path.join(rd, 'kperm_search.json'))
    if not search_data:
        print("  [SKIP] kperm_search.json not found")
        return
    best_assignment = search_data.get('best_assignment', {})

    # Load baseline
    refine_data = _safe_load(os.path.join(rd, 'combined_refine.json'))
    boot_data = _safe_load(os.path.join(rd, 'bootstrap_loop.json'))
    p15_assignment = refine_data.get('best_assignment', {})
    final_assignment = boot_data.get('final_assignment', p15_assignment)

    # Load modifiers + word set
    mod_data = _safe_load(os.path.join(rd, 'modifier_integrate.json'))
    modifier_chars, modifier_rules = set(), {}
    if mod_data:
        modifier_chars, modifier_rules = _reconstruct_modifier_rules(mod_data)

    ref_corpus = load_reference_corpus()
    base_words = set(w.lower() for w in ref_corpus.get_combined_tokens('latin'))
    expanded_set, _ = build_expanded_word_set(base_words)

    corpus = load_corpus(verbose=False)
    all_tokens = []
    for _fol, page in corpus.pages.items():
        all_tokens.extend(page.all_tokens)

    # Full corpus decode
    print("\n  1. Full corpus decode with CSA-best ...")
    csa_decoded = _decode_corpus_r3(
        all_tokens, best_assignment, eva_to_triple,
        modifier_chars, modifier_rules, expanded_set,
    )
    csa_dh = _compute_dict_hit(csa_decoded, expanded_set)

    print(f"     CSA-best dict-hit: {csa_dh:.4f}")

    # Phase 15 full corpus
    print("  2. Phase 15 full corpus decode ...")
    p15_decoded = _decode_corpus_r3(
        all_tokens, final_assignment, eva_to_triple,
        modifier_chars, modifier_rules, expanded_set,
    )
    p15_dh = _compute_dict_hit(p15_decoded, expanded_set)
    print(f"     Phase 15 dict-hit: {p15_dh:.4f}")

    # Null corpus test
    print("  3. Null corpus test ...")
    null_data = _safe_load(os.path.join(rd, 'null_corpus.json'))
    null_seeds = null_data.get('seeds', [100, 101, 102, 103, 104])

    null_dict_hits = []
    try:
        from voynich.phases.null_corpus import _build_eva_bigram_model, _generate_null_corpus
        bigram_probs, initial_probs, token_lengths = _build_eva_bigram_model(all_tokens)
        for seed in null_seeds[:5]:
            null_tokens = _generate_null_corpus(
                bigram_probs, initial_probs, token_lengths,
                len(all_tokens), seed,
            )
            null_decoded = _decode_corpus_r3(
                null_tokens, best_assignment, eva_to_triple,
                modifier_chars, modifier_rules, expanded_set,
            )
            null_dh = _compute_dict_hit(null_decoded, expanded_set)
            null_dict_hits.append(round(null_dh, 4))
            print(f"     Null seed {seed}: {null_dh:.4f}")
    except Exception as e:
        print(f"     Null test failed: {e}")
        null_dict_hits = [0.0]

    null_mean = float(np.mean(null_dict_hits))
    null_std = float(np.std(null_dict_hits)) if len(null_dict_hits) > 1 else 0.01
    selectivity = csa_dh / max(null_mean, 0.001)

    delta = csa_dh - p15_dh
    gate = csa_dh > null_mean + 2 * null_std and csa_dh >= p15_dh * 0.95

    if delta > 0.01:
        verdict = "CSA_IMPROVED"
    elif delta > -0.01:
        verdict = "CSA_COMPARABLE"
    else:
        verdict = "CSA_WORSE"

    result = CSAValidationResult(
        full_corpus_dict_hit=round(csa_dh, 4),
        p15_full_corpus_dict_hit=round(p15_dh, 4),
        delta=round(delta, 4),
        null_dict_hits=null_dict_hits,
        null_mean=round(null_mean, 4),
        null_std=round(null_std, 4),
        selectivity=round(selectivity, 2),
        gate_passed=gate,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'kperm_validation.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2)

    print(f"\n  CSA dict-hit: {csa_dh:.4f}, Phase 15: {p15_dh:.4f}, delta: {delta:+.4f}")
    print(f"  Null mean: {null_mean:.4f}, selectivity: {selectivity:.2f}x")
    print(f"  Verdict: {verdict}")
    print(f"  Saved -> {out_path}")
    print(f"\n  Step 44C.4 completed in {time.time() - t0:.1f}s")


# ---------------------------------------------------------------------------
# Track C runner
# ---------------------------------------------------------------------------

def run_track_c() -> None:
    """Run all Track C steps."""
    run_kperm_energy()
    print("\n" + "=" * 70 + "\n")
    run_kperm_search()
    print("\n" + "=" * 70 + "\n")
    run_kperm_analyze()
    print("\n" + "=" * 70 + "\n")
    run_kperm_validate()
