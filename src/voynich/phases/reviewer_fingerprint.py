"""
Reviewer Analysis 3: 6th-Ranked Cosine Similarity Gap
=====================================================

Tests whether there is a meaningful gap between the top-5 Latin
fingerprint matches and the next-best non-Latin profile.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

from voynich.core._paths import results_dir as _results_dir


def _convert(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _convert(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_convert(item) for item in obj]
    if isinstance(obj, float) and (obj != obj):
        return None
    if isinstance(obj, (bool, int, float, str, type(None))):
        return obj
    return str(obj)


ROMANCE_LANGUAGES = {"latin", "occitan", "italian", "spanish"}


def _classify_language_family(lang: str) -> str:
    lang_lower = lang.lower()
    if lang_lower in ROMANCE_LANGUAGES:
        return "Romance"
    if lang_lower in {"german"}:
        return "Germanic"
    if lang_lower in {"hebrew", "arabic"}:
        return "Semitic"
    return "Other"


def run_reviewer_fingerprint() -> None:
    t0 = time.time()
    rd = _results_dir()

    # Load fingerprint rankings
    rankings_path = os.path.join(rd, "match_rankings.json")
    if not os.path.exists(rankings_path):
        print("ERROR: match_rankings.json not found")
        return

    with open(rankings_path) as f:
        rankings_raw = json.load(f)

    # Sort descending by similarity (should already be sorted)
    rankings = sorted(rankings_raw, key=lambda x: x["similarity"], reverse=True)

    # Build top-10 with annotations
    top_10 = []
    for i, entry in enumerate(rankings[:10]):
        lang = entry["language"]
        top_10.append({
            "rank": i + 1,
            "profile": f"{lang}+{entry['encoding']}",
            "language": lang,
            "encoding": entry["encoding"],
            "family": _classify_language_family(lang),
            "cosine": entry["similarity"],
        })

    # Top-5 range
    top_5_scores = [e["cosine"] for e in top_10[:5]]
    top_5_min = min(top_5_scores)
    top_5_max = max(top_5_scores)

    # Gaps
    rank_6 = top_10[5] if len(top_10) > 5 else None
    rank_10 = top_10[9] if len(top_10) > 9 else None
    gap_5_to_6 = top_5_min - rank_6["cosine"] if rank_6 else 0.0
    gap_5_to_10 = top_5_min - rank_10["cosine"] if rank_10 else 0.0

    # Highest-ranked non-Romance
    highest_non_romance = None
    for i, entry in enumerate(rankings):
        lang = entry["language"]
        if lang.lower() not in ROMANCE_LANGUAGES:
            highest_non_romance = {
                "rank": i + 1,
                "profile": f"{lang}+{entry['encoding']}",
                "language": lang,
                "family": _classify_language_family(lang),
                "cosine": entry["similarity"],
            }
            break

    # Romance vs non-Romance in top 20
    top_20 = rankings[:20]
    romance_scores = [e["similarity"] for e in top_20
                      if e["language"].lower() in ROMANCE_LANGUAGES]
    non_romance_scores = [e["similarity"] for e in top_20
                          if e["language"].lower() not in ROMANCE_LANGUAGES]
    mean_romance = sum(romance_scores) / len(romance_scores) if romance_scores else 0.0
    mean_non_romance = (sum(non_romance_scores) / len(non_romance_scores)
                        if non_romance_scores else 0.0)

    # Check discrimination of top 5
    top_5_languages = set(e["language"].lower() for e in top_10[:5])
    all_latin = all(lang in ROMANCE_LANGUAGES for lang in top_5_languages)
    discriminates = gap_5_to_6 > 0.005

    # Interpretation
    if gap_5_to_6 > 0.02:
        interpretation = (
            "Clear discrimination — top-5 Latin cluster separates from "
            "alternatives by {:.4f} cosine.".format(gap_5_to_6)
        )
    elif gap_5_to_6 > 0.005:
        interpretation = (
            "Modest discrimination — Latin is favored but not decisively "
            "(gap {:.4f}).".format(gap_5_to_6)
        )
    else:
        interpretation = (
            "Weak discrimination — fingerprint does not reliably distinguish "
            "Latin from close alternatives. The cosine similarity {:.4f} "
            "reflects generic European text properties rather than "
            "Latin-specific structure. The language identification rests on "
            "the 4 independent convergent methods (signal isolation, "
            "size-matched OT, SBM profiling, n-gram analysis) rather than "
            "on fingerprint matching alone.".format(top_5_max)
        )

    result = {
        "test": "fingerprint_gap",
        "top_10_rankings": top_10,
        "top_5_range": {"min": top_5_min, "max": top_5_max},
        "top_5_all_romance": all_latin,
        "rank_6": {
            "profile": rank_6["profile"] if rank_6 else None,
            "language": rank_6["language"] if rank_6 else None,
            "cosine": rank_6["cosine"] if rank_6 else None,
        },
        "rank_10": {
            "profile": rank_10["profile"] if rank_10 else None,
            "language": rank_10["language"] if rank_10 else None,
            "cosine": rank_10["cosine"] if rank_10 else None,
        },
        "gap_5_to_6": gap_5_to_6,
        "gap_5_to_10": gap_5_to_10,
        "highest_non_romance": highest_non_romance,
        "romance_vs_non_romance": {
            "n_romance_in_top20": len(romance_scores),
            "n_non_romance_in_top20": len(non_romance_scores),
            "mean_romance": mean_romance,
            "mean_non_romance": mean_non_romance,
            "gap": mean_romance - mean_non_romance,
        },
        "discriminates": discriminates,
        "interpretation": interpretation,
        "runtime_seconds": time.time() - t0,
    }

    # Print summary
    print("=" * 70)
    print("REVIEWER ANALYSIS 3: Fingerprint Cosine Similarity Gap")
    print("=" * 70)
    print("\nTop 10 fingerprint matches:")
    for e in top_10:
        marker = " <--" if e["rank"] == 6 else ""
        print(f"  {e['rank']:2d}. {e['profile']:40s}  {e['cosine']:.6f}  "
              f"[{e['family']}]{marker}")
    print(f"\nTop-5 range: {top_5_min:.6f} – {top_5_max:.6f}")
    print(f"Gap rank 5→6: {gap_5_to_6:.6f}")
    print(f"Gap rank 5→10: {gap_5_to_10:.6f}")
    if highest_non_romance:
        print(f"Highest non-Romance: rank {highest_non_romance['rank']}, "
              f"{highest_non_romance['profile']} ({highest_non_romance['cosine']:.6f})")
    print(f"\nRomance mean (top 20): {mean_romance:.6f}")
    print(f"Non-Romance mean (top 20): {mean_non_romance:.6f}")
    print(f"Discriminates: {discriminates}")
    print(f"\n{interpretation}")

    # Save
    out_path = os.path.join(rd, "reviewer_fingerprint.json")
    with open(out_path, "w") as f:
        json.dump(_convert(result), f, indent=2)
    print(f"\nSaved to {out_path}")
