"""
Phase B.4 -- Cappelli Abbreviation Matching
=============================================
Quick-match Voynich stroke-feature patterns against Cappelli abbreviation
sign entries that have standalone stroke decompositions.

For each Cappelli entry with has_standalone_sign=true (meaning it has a
stroke-level decomposition), compute cosine_similarity_triples against
each of the 25 Voynich triples.  Record matches at EXACT (>= 0.95) and
NEAR (>= 0.80) thresholds.

This is exploratory — no formal gate.  Results inform whether medieval
abbreviation systems share structural DNA with Voynich sign construction.

Dependency chain:
    results/stroke_features.json
    data/reference/cappelli/cappelli_entries.json
        -> cappelli_match.json (this step)
"""

import json
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from voynich.core._paths import results_dir as _results_dir, data_dir as _data_dir
from voynich.core.reference import load_cappelli_reference
from voynich.core.stats import cosine_similarity_triples


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


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CappelliMatchResult:
    """Results of matching Voynich triples against Cappelli abbreviations."""
    n_cappelli_total: int
    n_cappelli_with_strokes: int
    n_exact_matches: int
    n_near_matches: int
    matches: List[Dict]
    category_distribution: Dict[str, int]
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def _parse_triple_key(triple_key: str) -> Dict[str, str]:
    """Parse 'first_stroke,last_stroke,glyph_class' into a dict."""
    parts = triple_key.split(',')
    if len(parts) == 3:
        return {
            'first_stroke': parts[0],
            'last_stroke': parts[1],
            'glyph_class': parts[2],
        }
    return {'first_stroke': '', 'last_stroke': '', 'glyph_class': ''}


def _entry_has_strokes(entry: Dict) -> bool:
    """Check if a Cappelli entry has stroke decomposition data."""
    if entry.get('has_standalone_sign', False):
        return True
    # Also accept entries with explicit stroke fields
    if (entry.get('first_stroke') and entry.get('last_stroke')
            and entry.get('glyph_class')):
        return True
    return False


def _get_entry_triple(entry: Dict) -> Optional[Dict[str, str]]:
    """Extract stroke triple from a Cappelli entry."""
    fs = entry.get('first_stroke', '')
    ls = entry.get('last_stroke', '')
    gc = entry.get('glyph_class', '')
    if fs and ls and gc:
        return {'first_stroke': fs, 'last_stroke': ls, 'glyph_class': gc}
    return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_cappelli_match() -> None:
    """Phase B.4: Match Voynich triples against Cappelli abbreviation signs."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE B.4: Cappelli Abbreviation Matching")
    print("=" * 70)

    rd = _results_dir()

    # ---- Step 1: Load stroke_features.json (25 Voynich triples) ----
    print("\n  1. Loading stroke features (25 Voynich triples) ...")
    sf_path = os.path.join(rd, 'stroke_features.json')
    if not os.path.exists(sf_path):
        print("      [ERROR] stroke_features.json not found. Run stroke-features first.")
        return

    with open(sf_path) as f:
        sf_data = json.load(f)

    attested_triples = sf_data.get('attested_triples', [])
    voynich_triples: List[Tuple[str, Dict[str, str]]] = []
    for t in attested_triples:
        tk = t.get('triple_key', '')
        voynich_triples.append((tk, _parse_triple_key(tk)))

    print(f"      {len(voynich_triples)} Voynich triples loaded")

    # ---- Step 2: Load Cappelli entries ----
    print("\n  2. Loading Cappelli reference entries ...")
    cappelli_entries = load_cappelli_reference()
    n_total = len(cappelli_entries)
    print(f"      {n_total} total Cappelli entries loaded")

    # ---- Step 3: Filter to entries with stroke decomposition ----
    print("\n  3. Filtering to entries with has_standalone_sign=true ...")
    stroke_entries = [e for e in cappelli_entries if _entry_has_strokes(e)]
    n_with_strokes = len(stroke_entries)
    print(f"      {n_with_strokes} entries with stroke decomposition")

    # ---- Step 4: Compute cosine similarities ----
    print("\n  4. Computing cosine similarities ...")
    all_matches: List[Dict] = []
    n_exact = 0
    n_near = 0

    for entry in stroke_entries:
        entry_triple = _get_entry_triple(entry)
        if entry_triple is None:
            continue

        entry_id = entry.get('entry_id', entry.get('id', entry.get('abbreviation', '?')))
        latin_expansion = entry.get('latin_expansion', entry.get('expansion', ''))
        category = entry.get('category', entry.get('domain', 'unknown'))
        century = entry.get('century', '')

        best_sim = 0.0
        best_voynich_triple = ''

        for v_key, v_triple in voynich_triples:
            sim = cosine_similarity_triples(v_triple, entry_triple)
            if sim > best_sim:
                best_sim = sim
                best_voynich_triple = v_key

        if best_sim >= 0.95:
            match_type = 'EXACT'
            n_exact += 1
        elif best_sim >= 0.80:
            match_type = 'NEAR'
            n_near += 1
        else:
            continue  # Skip non-matches

        all_matches.append({
            'cappelli_entry_id': str(entry_id),
            'voynich_triple': best_voynich_triple,
            'similarity': round(best_sim, 4),
            'latin_expansion': latin_expansion,
            'category': category,
            'century': str(century),
            'match_type': match_type,
        })

    print(f"      EXACT matches (>= 0.95): {n_exact}")
    print(f"      NEAR matches  (>= 0.80): {n_near}")
    print(f"      Total matches: {len(all_matches)}")

    # ---- Step 5: Category distribution ----
    print("\n  5. Matched abbreviation category distribution ...")
    category_counts: Counter = Counter()
    for m in all_matches:
        cat = m.get('category', 'unknown')
        category_counts[cat] += 1

    category_distribution = dict(category_counts.most_common())
    for cat, count in category_counts.most_common(10):
        print(f"      {cat:<30} {count}")

    # ---- Step 6: Century distribution ----
    print("\n  6. Century distribution of matches ...")
    century_counts: Counter = Counter()
    for m in all_matches:
        cent = m.get('century', 'unknown')
        if cent:
            century_counts[cent] += 1
    for cent, count in century_counts.most_common(10):
        print(f"      {cent:<15} {count}")

    # ---- Step 7: Report top matches ----
    if all_matches:
        # Sort by similarity descending
        sorted_matches = sorted(all_matches, key=lambda m: m['similarity'], reverse=True)
        print(f"\n  7. Top 15 matches:")
        print(f"      {'Cappelli ID':<20} {'Voynich Triple':<35} "
              f"{'Sim':>5} {'Latin':<20} {'Category'}")
        print("      " + "-" * 95)
        for m in sorted_matches[:15]:
            print(f"      {m['cappelli_entry_id']:<20} {m['voynich_triple']:<35} "
                  f"{m['similarity']:>5.3f} {m['latin_expansion']:<20} "
                  f"{m['category']}")

    # ---- Verdict (exploratory, no formal gate) ----
    if n_with_strokes == 0:
        verdict = (
            f"INCONCLUSIVE: No Cappelli entries had stroke decomposition. "
            f"Cannot compare structural patterns."
        )
    elif n_exact + n_near == 0:
        verdict = (
            f"NO MATCHES: 0/{n_with_strokes} Cappelli entries matched any "
            f"Voynich triple at >= 0.80 similarity. "
            f"Voynich signs differ structurally from known abbreviation marks."
        )
    else:
        match_rate = (n_exact + n_near) / n_with_strokes
        verdict = (
            f"EXPLORATORY: {n_exact} exact + {n_near} near matches out of "
            f"{n_with_strokes} Cappelli entries with strokes ({match_rate:.1%}). "
            f"Top categories: {', '.join(f'{c}({n})' for c, n in category_counts.most_common(3))}. "
            f"Abbreviation overlap suggests shared scribal tradition."
        )

    print(f"\n  Verdict: {verdict}")

    # ---- Save ----
    result = CappelliMatchResult(
        n_cappelli_total=n_total,
        n_cappelli_with_strokes=n_with_strokes,
        n_exact_matches=n_exact,
        n_near_matches=n_near,
        matches=all_matches,
        category_distribution=category_distribution,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'cappelli_match.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)

    print(f"\n  Results saved -> {out_path}")
