"""
Phase 18.3 – Prefix Trie Topology & Branching Imbalance
=========================================================

Discriminates H3 (Taxonomic / Philosophical Language) from H1 / H2 by
analysing the shape of the vocabulary's character-level prefix tree.

  H3 (Taxonomic)  → balanced, shallow trie with regular branching
  H1 (Hoax)       → trie topology matches a Cardan Grille null
  H2 (Cipher)     → deep, imbalanced trie (like natural language)

Uses the Colless imbalance index (phylogenetics metric) generalised
to non-binary trees.

Dependency chain:
    (none — reads corpus directly)
        -> trie_topology.json
"""

import json
import math
import os
import time
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import load_corpus, tokenize_eva_chars
from voynich.core.reference import load_reference_corpus


# ---------------------------------------------------------------------------
# JSON serialiser
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


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class TrieTopologyResult:
    n_types: int
    trie_depth_max: int
    trie_total_nodes: int
    trie_leaf_count: int
    trie_colless_index: float
    trie_branching_by_depth: Dict[str, float]    # depth -> mean branching
    latin_n_types: Optional[int]
    latin_colless_index: Optional[float]
    latin_branching_by_depth: Optional[Dict[str, float]]
    occitan_colless_index: Optional[float]
    cardan_n_types: int
    cardan_colless_index: float
    cardan_branching_by_depth: Dict[str, float]
    voynich_vs_cardan_colless_delta: float
    voynich_vs_latin_colless_delta: Optional[float]
    depth_1_branching: float
    hypothesis_support: Dict[str, float]
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Trie data structure
# ---------------------------------------------------------------------------

class TrieNode:
    __slots__ = ('children', 'is_terminal', '_subtree_leaves')

    def __init__(self) -> None:
        self.children: Dict[str, 'TrieNode'] = {}
        self.is_terminal: bool = False
        self._subtree_leaves: Optional[int] = None


class Trie:
    def __init__(self) -> None:
        self.root = TrieNode()

    def insert(self, chars: List[str]) -> None:
        node = self.root
        for ch in chars:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        node.is_terminal = True

    # -- metrics -----------------------------------------------------------

    def _count_subtree_leaves(self, node: TrieNode) -> int:
        """Number of terminal (leaf) nodes in the subtree rooted at *node*."""
        if node._subtree_leaves is not None:
            return node._subtree_leaves
        if not node.children:
            node._subtree_leaves = 1 if node.is_terminal else 0
            return node._subtree_leaves
        total = 0
        for child in node.children.values():
            total += self._count_subtree_leaves(child)
        # A terminal internal node also counts as a leaf
        if node.is_terminal:
            total += 1
        node._subtree_leaves = total
        return total

    def compute_metrics(self) -> Dict[str, Any]:
        """Return all trie metrics as a dict."""
        self._count_subtree_leaves(self.root)

        total_nodes = 0
        leaf_count = 0
        max_depth = 0
        depth_branching: Dict[int, List[int]] = defaultdict(list)
        colless_sum = 0.0
        colless_nodes = 0

        # BFS
        queue: deque = deque()
        queue.append((self.root, 0))
        while queue:
            node, depth = queue.popleft()
            total_nodes += 1
            if not node.children:
                leaf_count += 1
                max_depth = max(max_depth, depth)
                continue
            # Count as leaf if terminal (even though it has children)
            if node.is_terminal and not node.children:
                leaf_count += 1

            k = len(node.children)
            depth_branching[depth].append(k)

            # Colless contribution: generalised for k-ary node
            if k >= 2:
                sizes = [self._count_subtree_leaves(c) for c in node.children.values()]
                # sum of |s_i - s_j| for all pairs / C(k, 2)
                pair_sum = 0.0
                for i in range(len(sizes)):
                    for j in range(i + 1, len(sizes)):
                        pair_sum += abs(sizes[i] - sizes[j])
                n_pairs = k * (k - 1) / 2
                colless_sum += pair_sum / n_pairs
                colless_nodes += 1

            for child in node.children.values():
                queue.append((child, depth + 1))

        # Count terminal leaves that are also interior
        # (already counted in BFS leaf_count logic via "not node.children")
        # Recount more carefully
        leaf_count = self._count_subtree_leaves(self.root)

        normaliser = max(leaf_count - 1, 1)
        colless_index = colless_sum / normaliser

        branching_by_depth: Dict[int, float] = {}
        for d, factors in sorted(depth_branching.items()):
            branching_by_depth[d] = float(np.mean(factors))

        return {
            'total_nodes': total_nodes,
            'leaf_count': leaf_count,
            'max_depth': max_depth,
            'colless_index': round(colless_index, 4),
            'branching_by_depth': {str(d): round(v, 4) for d, v in branching_by_depth.items()},
            'depth_1_branching': branching_by_depth.get(0, 0.0),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


def _build_trie_from_tokens(tokens: List[str], use_eva: bool = True) -> Tuple[Trie, int]:
    """Build a character-level trie from unique token types.

    If *use_eva* is True, decompose tokens via ``tokenize_eva_chars``
    (for Voynich / Cardan Grille).  Otherwise use raw characters (for
    Latin / Occitan).

    Returns (trie, n_types).
    """
    types = set(tokens)
    trie = Trie()
    for tok in types:
        if use_eva:
            chars = tokenize_eva_chars(tok)
        else:
            chars = list(tok.lower())
        if chars:
            trie.insert(chars)
    return trie, len(types)


def _generate_cardan_grille_tokens(
    tokens: List[str],
    n_tokens: int = 5000,
    seed: int = 42,
) -> List[str]:
    """Generate a Cardan Grille null vocabulary by randomly recombining
    EVA characters according to unigram frequencies and word-length
    distribution from the real corpus.
    """
    rng = np.random.default_rng(seed)

    char_counts: Counter = Counter()
    length_counts: Counter = Counter()
    for tok in tokens:
        chars = tokenize_eva_chars(tok)
        for ch in chars:
            char_counts[ch] += 1
        length_counts[len(chars)] += 1

    chars_list = list(char_counts.keys())
    char_probs = np.array([char_counts[c] for c in chars_list], dtype=float)
    char_probs /= char_probs.sum()

    lengths = list(length_counts.keys())
    length_probs = np.array([length_counts[l] for l in lengths], dtype=float)
    length_probs /= length_probs.sum()

    fake_tokens: List[str] = []
    for _ in range(n_tokens):
        wlen = rng.choice(lengths, p=length_probs)
        chars = rng.choice(chars_list, size=wlen, p=char_probs)
        fake_tokens.append(''.join(chars))

    return fake_tokens


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_trie_topology() -> None:
    """Phase 18.3: prefix trie topology analysis."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE 18.3: Prefix Trie Topology & Branching Imbalance")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Voynich trie ───────────────────────────────────────────────
    print("\n  1. Building Voynich trie …")
    corpus = load_corpus(verbose=False)
    tokens_a = corpus.get_tokens(language='A', paragraph_only=True)
    v_trie, v_n_types = _build_trie_from_tokens(tokens_a, use_eva=True)
    v_metrics = v_trie.compute_metrics()
    print(f"     {v_n_types} types  |  depth={v_metrics['max_depth']}  |  "
          f"Colless={v_metrics['colless_index']:.4f}  |  "
          f"branching@0={v_metrics['depth_1_branching']:.2f}")

    # ── 2. Cardan Grille null trie ────────────────────────────────────
    print("\n  2. Building Cardan Grille null trie …")
    cardan_tokens = _generate_cardan_grille_tokens(tokens_a, n_tokens=5000)
    c_trie, c_n_types = _build_trie_from_tokens(cardan_tokens, use_eva=True)
    c_metrics = c_trie.compute_metrics()
    print(f"     {c_n_types} types  |  depth={c_metrics['max_depth']}  |  "
          f"Colless={c_metrics['colless_index']:.4f}")

    # ── 3. Latin trie ─────────────────────────────────────────────────
    print("\n  3. Building Latin trie …")
    latin_n_types: Optional[int] = None
    latin_colless: Optional[float] = None
    latin_branching: Optional[Dict[str, float]] = None
    try:
        ref = load_reference_corpus(languages=['latin'], verbose=False)
        lat_tokens = ref.get_combined_tokens('latin')
        if lat_tokens:
            l_trie, latin_n_types = _build_trie_from_tokens(lat_tokens, use_eva=False)
            l_metrics = l_trie.compute_metrics()
            latin_colless = l_metrics['colless_index']
            latin_branching = l_metrics['branching_by_depth']
            print(f"     {latin_n_types} types  |  Colless={latin_colless:.4f}")
    except Exception as e:
        print(f"     WARNING: Latin unavailable ({e})")

    # ── 4. Occitan trie ───────────────────────────────────────────────
    print("\n  4. Building Occitan trie …")
    occitan_colless: Optional[float] = None
    try:
        ref_oc = load_reference_corpus(languages=['occitan'], verbose=False)
        oc_tokens = ref_oc.get_combined_tokens('occitan')
        if oc_tokens:
            o_trie, _ = _build_trie_from_tokens(oc_tokens, use_eva=False)
            o_metrics = o_trie.compute_metrics()
            occitan_colless = o_metrics['colless_index']
            print(f"     Colless={occitan_colless:.4f}")
    except Exception:
        print("     WARNING: Occitan unavailable")

    # ── 5. Deltas ─────────────────────────────────────────────────────
    v_colless = v_metrics['colless_index']
    c_colless = c_metrics['colless_index']
    vs_cardan_delta = round(v_colless - c_colless, 4)
    vs_latin_delta = round(v_colless - latin_colless, 4) if latin_colless is not None else None

    print(f"\n  5. Voynich − Cardan Colless delta = {vs_cardan_delta:+.4f}")
    if vs_latin_delta is not None:
        print(f"     Voynich − Latin  Colless delta = {vs_latin_delta:+.4f}")

    # ── 6. Hypothesis scoring ─────────────────────────────────────────
    print("\n  6. Scoring hypotheses …")
    # H3: low Colless (balanced) → taxonomic
    h3 = _sigmoid(-(v_colless - 0.3) / 0.2)
    # H1: similar to Cardan null
    h1 = _sigmoid(-abs(vs_cardan_delta) / 0.15)
    # H2: high Colless (like natural language)
    ref_colless = latin_colless if latin_colless is not None else 0.6
    h2 = _sigmoid((v_colless - ref_colless + 0.2) / 0.25)

    total = h1 + h2 + h3
    if total > 0:
        h1, h2, h3 = h1 / total, h2 / total, h3 / total

    hypothesis_support = {'H1': round(h1, 4), 'H2': round(h2, 4), 'H3': round(h3, 4)}
    print(f"     H1={h1:.3f}  H2={h2:.3f}  H3={h3:.3f}")

    # ── Verdict ───────────────────────────────────────────────────────
    if v_colless < 0.25:
        verdict = (f"BALANCED TRIE: Colless = {v_colless:.4f} is very low — vocabulary "
                   "is unnaturally regular, consistent with H3 (taxonomic language).")
    elif abs(vs_cardan_delta) < 0.1:
        verdict = (f"CARDAN-LIKE: Voynich Colless ({v_colless:.4f}) matches Cardan Grille "
                   f"null ({c_colless:.4f}), delta = {vs_cardan_delta:+.4f}. "
                   "Consistent with H1 (hoax).")
    elif latin_colless is not None and abs(v_colless - latin_colless) < 0.15:
        verdict = (f"NATURAL-LIKE: Voynich Colless ({v_colless:.4f}) matches Latin "
                   f"({latin_colless:.4f}). Consistent with H2 (cipher over natural language).")
    else:
        verdict = (f"MIXED: Voynich Colless = {v_colless:.4f}, Cardan = {c_colless:.4f}, "
                   f"Latin = {latin_colless}. No clear match.")

    print(f"\n  Verdict: {verdict}")

    # ── Save ──────────────────────────────────────────────────────────
    result = TrieTopologyResult(
        n_types=v_n_types,
        trie_depth_max=v_metrics['max_depth'],
        trie_total_nodes=v_metrics['total_nodes'],
        trie_leaf_count=v_metrics['leaf_count'],
        trie_colless_index=v_colless,
        trie_branching_by_depth=v_metrics['branching_by_depth'],
        latin_n_types=latin_n_types,
        latin_colless_index=latin_colless,
        latin_branching_by_depth=latin_branching,
        occitan_colless_index=occitan_colless,
        cardan_n_types=c_n_types,
        cardan_colless_index=c_colless,
        cardan_branching_by_depth=c_metrics['branching_by_depth'],
        voynich_vs_cardan_colless_delta=vs_cardan_delta,
        voynich_vs_latin_colless_delta=vs_latin_delta,
        depth_1_branching=v_metrics['depth_1_branching'],
        hypothesis_support=hypothesis_support,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'trie_topology.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)

    print(f"\n  -> {out_path}")
