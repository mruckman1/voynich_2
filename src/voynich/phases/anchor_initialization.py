"""
Step 43.11 – Anchor Initialization
=====================================
Fix the HMM's hidden states at positions where the 8 bedrock signal words
provide known syllable values, creating "anchor constraints" for the
Baum-Welch training.

Dependency chain:
    results/hmm_architecture.json    (Step 43.10: model structure)
    results/signal_positions.json    (Step 43.6: signal word positions)
    results/combined_refine.json     (Phase 15: 25-triple table)
    results/modifier_integrate.json  (Phase 16: modifier handling)
    data/corpus/                     (EVA transcription)
        → anchor_initialization.json (this step)
"""

import json
import os
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    decode_token_modifier_aware,
    load_corpus,
    tokenize_eva_chars,
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
    if isinstance(obj, (bool, int, float, str, type(None))):
        return obj
    return str(obj)


def _safe_load(path: str) -> Dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


# ---------------------------------------------------------------------------
# Bedrock signal words and their syllable decompositions
# ---------------------------------------------------------------------------

BEDROCK_WORDS: Dict[str, List[str]] = {
    'de':   ['de'],
    'bene': ['be', 'ne'],
    'cola': ['co', 'la'],
    'sene': ['se', 'ne'],
    'sero': ['se', 'ro'],
    'codi': ['co', 'di'],
    'raro': ['ra', 'ro'],
    'dine': ['di', 'ne'],
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class AnchorInitResult:
    # Anchor summary
    n_anchored_tokens: int
    n_anchored_char_positions: int
    n_total_char_positions: int
    anchor_rate: float
    # Per-word anchor counts
    per_word_counts: Dict[str, int]
    # Anchor consistency
    n_consistent_chars: int
    n_inconsistent_chars: int
    consistency_rate: float
    consistency_details: List[Dict]
    # Anchor list (token-level, not char-level to keep JSON small)
    n_anchor_entries: int
    anchor_sample: List[Dict]
    # Constraint method
    constraint_method: str
    anchor_weight: float
    # State mapping
    n_states_anchored: int
    anchored_states: List[str]
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def _build_state_index(state_labels: List[str]) -> Dict[str, int]:
    """Map syllable label to state index."""
    return {label: i for i, label in enumerate(state_labels)}


def _find_anchored_positions(
    corpus,
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    modifier_chars: Set[str],
    state_index: Dict[str, int],
) -> Tuple[List[Dict], Dict[str, int], int]:
    """Find all token positions where signal words anchor the hidden state.

    Returns:
        anchors: list of anchor dicts {folio, token_idx, word, syllables,
                 char_positions: [(global_char_pos, state_idx), ...]}
        per_word_counts: word → count
        total_chars: total character positions in corpus
    """
    anchors: List[Dict] = []
    per_word_counts: Dict[str, int] = Counter()
    total_chars = 0
    global_char_pos = 0

    for folio_id, page in corpus.pages.items():
        tokens = page.all_tokens
        for tok_idx, token in enumerate(tokens):
            # Count characters in this token
            eva_chars = tokenize_eva_chars(token)
            n_chars = len(eva_chars)

            # Decode this token
            decoded = decode_token_modifier_aware(
                token, assignment, eva_to_triple, modifier_chars
            )

            if decoded in BEDROCK_WORDS:
                syllables = BEDROCK_WORDS[decoded]
                # Map each syllable to its state index
                char_positions = []

                # Build the syllable-to-char mapping for this token
                # Each non-modifier EVA char maps to one syllable
                syl_idx = 0
                for ci, ch in enumerate(eva_chars):
                    if ch in modifier_chars:
                        continue
                    if syl_idx < len(syllables):
                        syl = syllables[syl_idx]
                        si = state_index.get(syl)
                        if si is not None:
                            char_positions.append({
                                'global_pos': global_char_pos + ci,
                                'state_idx': si,
                                'syllable': syl,
                                'eva_char': ch,
                            })
                        syl_idx += 1

                if char_positions:
                    anchors.append({
                        'folio': folio_id,
                        'token_idx': tok_idx,
                        'word': decoded,
                        'syllables': syllables,
                        'char_positions': char_positions,
                    })
                    per_word_counts[decoded] += 1

            global_char_pos += n_chars
            total_chars += n_chars

    return anchors, dict(per_word_counts), total_chars


def _check_anchor_consistency(
    anchors: List[Dict],
) -> Tuple[int, int, List[Dict]]:
    """Check whether the same EVA char always anchors to the same state.

    Returns (n_consistent, n_inconsistent, details).
    """
    # Map EVA char → set of (syllable) assignments from anchors
    char_assignments: Dict[str, Counter] = defaultdict(Counter)

    for anchor in anchors:
        for cp in anchor['char_positions']:
            char_assignments[cp['eva_char']][cp['syllable']] += 1

    n_consistent = 0
    n_inconsistent = 0
    details = []

    for eva_char, syl_counts in sorted(char_assignments.items()):
        n_syls = len(syl_counts)
        total = sum(syl_counts.values())
        dominant = syl_counts.most_common(1)[0]

        entry = {
            'eva_char': eva_char,
            'n_assignments': n_syls,
            'total_anchored': total,
            'dominant_syllable': dominant[0],
            'dominant_count': dominant[1],
            'dominant_rate': round(dominant[1] / total, 4) if total > 0 else 0,
            'consistent': n_syls == 1,
        }

        if n_syls == 1:
            n_consistent += 1
        else:
            n_inconsistent += 1
            entry['alternatives'] = dict(syl_counts)

        details.append(entry)

    return n_consistent, n_inconsistent, details


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_anchor_initialization() -> None:
    """Step 43.11: fix signal word positions as anchor constraints."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 43.11: Anchor Initialization")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load HMM architecture ──
    print("\n  1. Loading HMM architecture …")
    hmm_arch = _safe_load(os.path.join(rd, 'hmm_architecture.json'))
    state_labels = hmm_arch.get('state_labels', [])
    obs_vocab = hmm_arch.get('observation_vocab', [])
    K = hmm_arch.get('n_hidden_states', len(state_labels))
    V = hmm_arch.get('n_observation_types', len(obs_vocab))
    print(f"     K={K} states, V={V} observations")

    state_index = _build_state_index(state_labels)

    # Check bedrock syllables are in state set
    missing = []
    for word, syls in BEDROCK_WORDS.items():
        for s in syls:
            if s not in state_index:
                missing.append((word, s))
    if missing:
        print(f"     WARNING: {len(missing)} syllables missing from state set: {missing}")
    else:
        print(f"     All bedrock syllables found in state set ✓")

    # ── 2. Load Phase 15 assignment and modifiers ──
    print("\n  2. Loading assignment and modifiers …")
    combined = _safe_load(os.path.join(rd, 'combined_refine.json'))
    assignment = combined.get('best_assignment', {})

    mod_data = _safe_load(os.path.join(rd, 'modifier_integrate.json'))
    modifier_chars = set(mod_data.get('modifier_chars', []))
    print(f"     Assignment: {len(assignment)} triples")
    print(f"     Modifier chars: {len(modifier_chars)}")

    # ── 3. Load corpus and find anchors ──
    print("\n  3. Finding anchor positions …")
    corpus = load_corpus(verbose=False)
    eva_to_triple = build_eva_to_triple_lookup()

    anchors, per_word_counts, total_chars = _find_anchored_positions(
        corpus, assignment, eva_to_triple, modifier_chars, state_index
    )

    n_anchored_tokens = len(anchors)
    n_anchored_chars = sum(
        len(a['char_positions']) for a in anchors
    )
    anchor_rate = n_anchored_chars / total_chars if total_chars > 0 else 0.0

    print(f"     Total character positions: {total_chars:,}")
    print(f"     Anchored tokens: {n_anchored_tokens:,}")
    print(f"     Anchored char positions: {n_anchored_chars:,}")
    print(f"     Anchor rate: {anchor_rate:.1%}")

    print("\n     Per-word counts:")
    for word in sorted(per_word_counts.keys()):
        print(f"       {word:6s}: {per_word_counts[word]:5d}")

    # ── 4. Consistency check ──
    print("\n  4. Checking anchor consistency …")
    n_consistent, n_inconsistent, consistency_details = _check_anchor_consistency(anchors)
    consistency_rate = (
        n_consistent / (n_consistent + n_inconsistent)
        if (n_consistent + n_inconsistent) > 0 else 0.0
    )
    print(f"     Consistent EVA chars: {n_consistent}")
    print(f"     Inconsistent EVA chars: {n_inconsistent}")
    print(f"     Consistency rate: {consistency_rate:.1%}")

    if n_inconsistent > 0:
        print("\n     Inconsistent chars:")
        for d in consistency_details:
            if not d['consistent']:
                print(f"       '{d['eva_char']}': {d.get('alternatives', {})} "
                      f"(dominant: {d['dominant_syllable']} at {d['dominant_rate']:.0%})")

    # ── 5. Identify anchored states ──
    anchored_states_set = set()
    for word, syls in BEDROCK_WORDS.items():
        if word in per_word_counts:
            for s in syls:
                if s in state_index:
                    anchored_states_set.add(s)
    anchored_states = sorted(anchored_states_set)
    print(f"\n  5. Anchored states: {len(anchored_states)}")
    print(f"     {anchored_states}")

    # ── 6. Summary ──
    print("\n  6. Summary")
    print(f"     Constraint method: hard_clamp")
    print(f"     At anchored positions during Baum-Welch E-step:")
    print(f"       gamma[t, k] = 1.0 for anchored state k")
    print(f"       gamma[t, k'] = 0.0 for all other states k'")
    print(f"     This provides {n_anchored_chars:,} supervised data points "
          f"out of {total_chars:,} total ({anchor_rate:.1%})")

    # ── 7. Save ──
    elapsed = time.time() - t0

    # Keep anchor sample small for JSON
    anchor_sample = []
    for a in anchors[:50]:
        anchor_sample.append({
            'folio': a['folio'],
            'token_idx': a['token_idx'],
            'word': a['word'],
            'syllables': a['syllables'],
            'n_char_anchors': len(a['char_positions']),
        })

    result = AnchorInitResult(
        n_anchored_tokens=n_anchored_tokens,
        n_anchored_char_positions=n_anchored_chars,
        n_total_char_positions=total_chars,
        anchor_rate=round(anchor_rate, 6),
        per_word_counts=per_word_counts,
        n_consistent_chars=n_consistent,
        n_inconsistent_chars=n_inconsistent,
        consistency_rate=round(consistency_rate, 4),
        consistency_details=consistency_details,
        n_anchor_entries=n_anchored_tokens,
        anchor_sample=anchor_sample,
        constraint_method='hard_clamp',
        anchor_weight=1.0,
        n_states_anchored=len(anchored_states),
        anchored_states=anchored_states,
        runtime_seconds=round(elapsed, 1),
    )

    out_path = os.path.join(rd, 'anchor_initialization.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)
    print(f"\n  → {out_path} ({elapsed:.1f}s)")
