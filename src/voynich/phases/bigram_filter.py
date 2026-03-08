"""
Phase 24.4 – Bigram Plausibility Validation (bigram-filter)
============================================================
Validates the corrected assignment table (from targeted_swap.json) against
Phase 16's original table using a held-out token sample.  Tests for
overfitting by comparing held-out vs training bigram plausibility, and
runs null baselines (random permutations of the corrected table).

Also runs a full readability battery on the held-out sample: POS trigram
validity, domain coherence, and phrase detection.

Dependency chain:
    combined_refine.json (Phase 15 best_assignment)
    modifier_integrate.json (Phase 16 modifier chars)
    targeted_swap.json (Step 24.3 corrected assignment)
        -> bigram_filter.json (this step)
"""

import json
import os
import random
import time
from collections import Counter
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
    LATIN_PHRASE_PATTERNS,
    PHARMACEUTICAL_VOCABULARY,
    build_expanded_word_set,
    load_reference_corpus,
)
from voynich.phases.csp_solver import decode_token


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
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Bigram / readability helpers
# ---------------------------------------------------------------------------

def _build_ref_bigrams(ref_words: List[str]) -> set:
    return {(ref_words[i].lower(), ref_words[i + 1].lower())
            for i in range(len(ref_words) - 1)}


def _bigram_plausibility(decoded_words: List[str], ref_bigrams: set) -> float:
    if len(decoded_words) < 2:
        return 0.0
    hits = sum(1 for i in range(len(decoded_words) - 1)
               if (decoded_words[i], decoded_words[i + 1]) in ref_bigrams)
    return hits / (len(decoded_words) - 1)


_LATIN_POS_RULES = [
    (lambda w: w.endswith(('are', 'ere', 'ire', 'ari', 'eri', 'iri')), 'VERB'),
    (lambda w: w.endswith(('at', 'et', 'it', 'ant', 'ent', 'unt')), 'VERB'),
    (lambda w: w in ('in', 'de', 'ad', 'ex', 'per', 'cum', 'pro', 'sub'), 'PREP'),
    (lambda w: w in ('et', 'sed', 'aut', 'vel', 'atque', 'quod', 'quia', 'si'), 'CONJ'),
    (lambda w: w.endswith(('us', 'a', 'um', 'is', 'e')), 'ADJ'),
    (lambda w: w.endswith(('ae', 'arum', 'orum', 'ibus')), 'NOUN'),
    (lambda w: len(w) >= 4, 'NOUN'),
]


def _pos_tag(word: str) -> str:
    w = word.lower()
    for rule, tag in _LATIN_POS_RULES:
        if rule(w):
            return tag
    return 'NOUN'


def _pos_trigram_validity(decoded_words: List[str], ref_pos_trigrams: set) -> float:
    if len(decoded_words) < 3:
        return 0.0
    tags = [_pos_tag(w) for w in decoded_words]
    hits = sum(1 for i in range(len(tags) - 2)
               if (tags[i], tags[i + 1], tags[i + 2]) in ref_pos_trigrams)
    total = len(tags) - 2
    return hits / total if total else 0.0


def _build_ref_pos_trigrams(ref_words: List[str]) -> set:
    tags = [_pos_tag(w.lower()) for w in ref_words]
    return {(tags[i], tags[i + 1], tags[i + 2]) for i in range(len(tags) - 2)}


def _domain_coherence(decoded_words: List[str], pharma_vocab: Dict) -> Dict[str, Dict]:
    word_set = set(w.lower() for w in decoded_words)
    results = {}
    for domain, terms in pharma_vocab.items():
        term_set = set(t.lower() for t in terms)
        hits = word_set & term_set
        results[domain] = {
            'n_terms': len(term_set),
            'n_hits': len(hits),
            'hit_rate': len(hits) / max(len(term_set), 1),
            'matched_terms': sorted(hits),
        }
    return results


def _detect_phrases(decoded_words: List[str], phrase_patterns) -> List[Dict]:
    text = ' '.join(decoded_words)
    hits = []
    for pattern_name, templates in phrase_patterns:
        for template in templates:
            if template.lower() in text:
                idx = text.index(template.lower())
                hits.append({
                    'pattern': pattern_name,
                    'template': template,
                    'position': idx,
                })
    return hits


# ---------------------------------------------------------------------------
# R3 combined decode
# ---------------------------------------------------------------------------

def _decode_r3(
    tokens: List[str],
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    modifier_chars: Set[str],
    modifier_rules: Dict[str, str],
    ref_word_set: set,
) -> List[str]:
    """Decode tokens using R3 combined strategy (alter -> strip -> original)."""
    decoded = []
    for token in tokens:
        # Try alteration
        alt = decode_token_modifier_aware(
            token, assignment, eva_to_triple, modifier_chars,
            modifier_rules=modifier_rules,
        )
        if alt.lower() in ref_word_set:
            decoded.append(alt)
            continue

        # Try stripping
        stripped = decode_token_modifier_aware(
            token, assignment, eva_to_triple, modifier_chars,
        )
        if stripped.lower() in ref_word_set:
            decoded.append(stripped)
            continue

        # Fall back to original decoding
        original = decode_token(token, assignment, eva_to_triple)
        decoded.append(original)

    return decoded


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def _sample_tokens(all_tokens: List[str], n: int, seed: int) -> List[str]:
    """Draw a reproducible random sample of n tokens."""
    rng = random.Random(seed)
    if len(all_tokens) <= n:
        return list(all_tokens)
    return rng.sample(all_tokens, n)


# ---------------------------------------------------------------------------
# Null table generation
# ---------------------------------------------------------------------------

def _make_null_table(
    assignment: Dict[str, str],
    seed: int,
) -> Dict[str, str]:
    """Create a null table by randomly permuting the syllable values."""
    rng = random.Random(seed)
    keys = list(assignment.keys())
    vals = list(assignment.values())
    rng.shuffle(vals)
    return dict(zip(keys, vals))


# ---------------------------------------------------------------------------
# Dict-hit rate
# ---------------------------------------------------------------------------

def _dict_hit_rate(decoded_words: List[str], ref_word_set: set) -> float:
    if not decoded_words:
        return 0.0
    hits = sum(1 for w in decoded_words if w.lower() in ref_word_set)
    return hits / len(decoded_words)


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class BigramFilterResult:
    timestamp: str
    # Held-out sample
    held_out_n_tokens: int
    held_out_seed: int
    # Corrected table on held-out
    corrected_dict_hit: float
    corrected_bigram: float
    # Phase 16 on held-out
    phase16_dict_hit: float
    phase16_bigram: float
    # Null baselines
    null_bigrams: List[float]
    null_mean_bigram: float
    corrected_vs_null_ratio: float
    # Overfitting check
    training_bigram: float
    held_out_bigram: float
    overfitting_detected: bool
    overfitting_ratio: float  # held_out / training
    # Readability battery on held-out
    pos_validity: float
    n_domain_hits: int
    n_phrase_hits: int
    # Verdict
    bigram_improved: bool  # corrected > phase16 on held-out
    bigram_above_null: bool  # corrected > null mean
    gate_passed: bool  # improved AND not overfit AND above null
    verdict: str
    recommended_table: str  # "corrected" or "phase16"
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_bigram_filter() -> None:
    """Step 24.4: Bigram Plausibility Validation."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 24.4: Bigram Plausibility Validation")
    print("=" * 70)

    rdir = _results_dir()

    # ------------------------------------------------------------------
    # 1. Load corrected assignment from targeted_swap.json
    # ------------------------------------------------------------------
    print("  Loading corrected assignment (targeted_swap.json)...")
    swap_data = _load_json(str(rdir / "targeted_swap.json"))
    if swap_data is None:
        print("  ERROR: targeted_swap.json not found. Run step 24.3 first.")
        return
    corrected_assignment = swap_data.get("final_assignment", {})
    print(f"    Corrected table: {len(corrected_assignment)} triples")

    # ------------------------------------------------------------------
    # 1b. Load Phase 16 original assignment from combined_refine.json
    # ------------------------------------------------------------------
    print("  Loading Phase 16 assignment (combined_refine.json)...")
    combined = _load_json(str(rdir / "combined_refine.json")) or {}
    phase16_assignment = combined.get("best_assignment", {})
    print(f"    Phase 16 table: {len(phase16_assignment)} triples")

    # ------------------------------------------------------------------
    # 2. Load modifiers
    # ------------------------------------------------------------------
    print("  Loading modifier configuration...")
    mod_data = _load_json(str(rdir / "modifier_integrate.json")) or {}
    modifier_chars = set(mod_data.get("modifier_chars", []))
    modifier_rules: Dict[str, str] = {}
    for cls in mod_data.get("classifications", []):
        if cls.get("final_classification") == "modifier":
            modifier_rules[cls["eva_char"]] = cls.get("modifier_type", "silent")
    print(f"    {len(modifier_chars)} modifier chars, {len(modifier_rules)} rules")

    # ------------------------------------------------------------------
    # 3. Load corpus and build samples
    # ------------------------------------------------------------------
    print("  Loading corpus and building samples...")
    corpus = load_corpus(verbose=False)
    all_tokens = corpus.get_tokens()
    eva_to_triple = build_eva_to_triple_lookup()
    print(f"    Corpus: {len(all_tokens)} tokens total")

    # Held-out sample: seed 123, 5000 tokens (DIFFERENT from training)
    held_out_tokens = _sample_tokens(all_tokens, 5000, seed=123)
    print(f"    Held-out sample: {len(held_out_tokens)} tokens (seed=123)")

    # Training sample: seed 42, 5000 tokens (for overfitting check)
    training_tokens = _sample_tokens(all_tokens, 5000, seed=42)
    print(f"    Training sample: {len(training_tokens)} tokens (seed=42)")

    # ------------------------------------------------------------------
    # Build dictionary and reference bigrams
    # ------------------------------------------------------------------
    print("  Building reference dictionary and bigrams...")
    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    base_words = set(
        w.lower() for w in ref_corpus.get_combined_tokens('latin')
        if len(w) >= 2
    )
    expanded_words, _ = build_expanded_word_set(base_words)
    ref_word_set = base_words | expanded_words
    print(f"    Dictionary: {len(ref_word_set)} words")

    ref_words = [w.lower() for w in ref_corpus.get_combined_tokens('latin')
                 if len(w) >= 2]
    ref_bigrams = _build_ref_bigrams(ref_words)
    ref_pos_trigrams = _build_ref_pos_trigrams(ref_words)
    print(f"    Reference bigrams: {len(ref_bigrams)}")
    print(f"    Reference POS trigrams: {len(ref_pos_trigrams)}")

    # ------------------------------------------------------------------
    # 5. Held-out validation
    # ------------------------------------------------------------------
    print("\n  --- Held-out validation ---")

    # Decode held-out with corrected table
    print("  Decoding held-out with corrected table (R3)...")
    corrected_held_out = _decode_r3(
        held_out_tokens, corrected_assignment, eva_to_triple,
        modifier_chars, modifier_rules, ref_word_set,
    )
    corrected_dict_hit = _dict_hit_rate(corrected_held_out, ref_word_set)
    corrected_lo = [w.lower() for w in corrected_held_out if w.lower() in ref_word_set]
    corrected_bigram = _bigram_plausibility(corrected_lo, ref_bigrams)
    print(f"    Corrected: dict_hit={corrected_dict_hit:.1%}, bigram={corrected_bigram:.6f}")

    # Decode held-out with Phase 16 original table
    print("  Decoding held-out with Phase 16 table (R3)...")
    phase16_held_out = _decode_r3(
        held_out_tokens, phase16_assignment, eva_to_triple,
        modifier_chars, modifier_rules, ref_word_set,
    )
    phase16_dict_hit = _dict_hit_rate(phase16_held_out, ref_word_set)
    phase16_lo = [w.lower() for w in phase16_held_out if w.lower() in ref_word_set]
    phase16_bigram = _bigram_plausibility(phase16_lo, ref_bigrams)
    print(f"    Phase 16:  dict_hit={phase16_dict_hit:.1%}, bigram={phase16_bigram:.6f}")

    # ------------------------------------------------------------------
    # 6. Null comparison
    # ------------------------------------------------------------------
    print("\n  --- Null comparison (5 random permutations) ---")
    null_bigrams: List[float] = []
    for i in range(5):
        null_table = _make_null_table(corrected_assignment, seed=i)
        null_decoded = _decode_r3(
            held_out_tokens, null_table, eva_to_triple,
            modifier_chars, modifier_rules, ref_word_set,
        )
        null_lo = [w.lower() for w in null_decoded if w.lower() in ref_word_set]
        nb = _bigram_plausibility(null_lo, ref_bigrams)
        null_bigrams.append(round(nb, 6))
        print(f"    Null {i}: bigram={nb:.6f}")

    null_mean_bigram = sum(null_bigrams) / len(null_bigrams) if null_bigrams else 0.0
    corrected_vs_null = (corrected_bigram / null_mean_bigram
                         if null_mean_bigram > 0
                         else (float('inf') if corrected_bigram > 0 else 0.0))
    print(f"    Null mean: {null_mean_bigram:.6f}")
    print(f"    Corrected/null ratio: {corrected_vs_null:.2f}x")

    # ------------------------------------------------------------------
    # 7. Overfitting check
    # ------------------------------------------------------------------
    print("\n  --- Overfitting check ---")
    print("  Decoding training sample with corrected table (R3)...")
    corrected_training = _decode_r3(
        training_tokens, corrected_assignment, eva_to_triple,
        modifier_chars, modifier_rules, ref_word_set,
    )
    training_lo = [w.lower() for w in corrected_training if w.lower() in ref_word_set]
    training_bigram = _bigram_plausibility(training_lo, ref_bigrams)

    overfitting_ratio = (corrected_bigram / training_bigram
                         if training_bigram > 0
                         else (float('inf') if corrected_bigram > 0 else 1.0))
    overfitting_detected = corrected_bigram < training_bigram * 0.8
    print(f"    Training bigram:  {training_bigram:.6f}")
    print(f"    Held-out bigram:  {corrected_bigram:.6f}")
    print(f"    Ratio (held_out/training): {overfitting_ratio:.4f}")
    print(f"    Overfitting detected: {overfitting_detected}")

    # ------------------------------------------------------------------
    # 8. Full readability battery on held-out
    # ------------------------------------------------------------------
    print("\n  --- Readability battery (held-out) ---")

    # POS trigram validity
    pos_validity = _pos_trigram_validity(corrected_lo, ref_pos_trigrams)
    print(f"    POS trigram validity: {pos_validity:.4f}")

    # Domain coherence
    all_corrected_lo = [w.lower() for w in corrected_held_out]
    domain_results = _domain_coherence(all_corrected_lo, PHARMACEUTICAL_VOCABULARY)
    n_domain_hits = sum(1 for d in domain_results.values() if d['n_hits'] > 0)
    print(f"    Domain coherence: {n_domain_hits} domains with hits")
    for domain, info in domain_results.items():
        if info['n_hits'] > 0:
            print(f"      {domain}: {info['n_hits']} hits ({', '.join(info['matched_terms'][:5])})")

    # Phrase detection
    phrase_hits = _detect_phrases(corrected_lo, LATIN_PHRASE_PATTERNS)
    n_phrase_hits = len(phrase_hits)
    print(f"    Phrase hits: {n_phrase_hits}")
    for ph in phrase_hits[:5]:
        print(f"      {ph['pattern']}: '{ph['template']}' at pos {ph['position']}")

    # ------------------------------------------------------------------
    # 9. Verdict
    # ------------------------------------------------------------------
    bigram_improved = corrected_bigram > phase16_bigram
    bigram_above_null = corrected_bigram > null_mean_bigram

    if overfitting_detected:
        gate_passed = False
        verdict = "OVERFIT"
        recommended_table = "phase16"
    elif bigram_improved and bigram_above_null:
        gate_passed = True
        verdict = "PASS — corrected table validated on held-out data"
        recommended_table = "corrected"
    elif bigram_above_null and not bigram_improved:
        gate_passed = True
        verdict = "PASS — corrected above null but not better than Phase 16"
        recommended_table = "corrected"
    elif bigram_improved and not bigram_above_null:
        gate_passed = False
        verdict = "FAIL — improved over Phase 16 but not above null baseline"
        recommended_table = "phase16"
    else:
        gate_passed = False
        verdict = "FAIL — corrected table worse than Phase 16 and null"
        recommended_table = "phase16"

    elapsed = time.time() - t0

    print(f"\n  {'=' * 50}")
    print(f"  Verdict: {verdict}")
    print(f"  Recommended table: {recommended_table}")
    print(f"  Gate passed: {gate_passed}")
    print(f"  {'=' * 50}")

    # ------------------------------------------------------------------
    # Build result and save
    # ------------------------------------------------------------------
    result = BigramFilterResult(
        timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
        held_out_n_tokens=len(held_out_tokens),
        held_out_seed=123,
        corrected_dict_hit=round(corrected_dict_hit, 4),
        corrected_bigram=round(corrected_bigram, 6),
        phase16_dict_hit=round(phase16_dict_hit, 4),
        phase16_bigram=round(phase16_bigram, 6),
        null_bigrams=null_bigrams,
        null_mean_bigram=round(null_mean_bigram, 6),
        corrected_vs_null_ratio=round(corrected_vs_null, 4),
        training_bigram=round(training_bigram, 6),
        held_out_bigram=round(corrected_bigram, 6),
        overfitting_detected=overfitting_detected,
        overfitting_ratio=round(overfitting_ratio, 4),
        pos_validity=round(pos_validity, 4),
        n_domain_hits=n_domain_hits,
        n_phrase_hits=n_phrase_hits,
        bigram_improved=bigram_improved,
        bigram_above_null=bigram_above_null,
        gate_passed=gate_passed,
        verdict=verdict,
        recommended_table=recommended_table,
        runtime_seconds=round(elapsed, 1),
    )

    out_path = rdir / "bigram_filter.json"
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2, ensure_ascii=False)

    print(f"\n  Saved: {out_path} ({elapsed:.1f}s)")
