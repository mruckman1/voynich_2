"""
Reviewer Integration: Summary of all three reviewer response analyses.
======================================================================

Loads results from the three independent analyses and produces a
unified summary with paper-ready text snippets.
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


def run_reviewer_integrate() -> None:
    t0 = time.time()
    rd = _results_dir()

    print("=" * 70)
    print("REVIEWER INTEGRATION: Combined Summary")
    print("=" * 70)

    analyses = {}

    # ── Load Analysis 1 (permutation) ──
    perm_path = os.path.join(rd, "reviewer_permutation.json")
    perm_snippet = "(not yet run)"
    perm_verdict = "PENDING"
    if os.path.exists(perm_path):
        with open(perm_path) as f:
            perm = json.load(f)
        analyses["permutation"] = perm
        perm_verdict = perm.get("verdict", "UNKNOWN")
        opt_a = perm.get("option_a", {})
        rt = opt_a.get("real_table", {})
        pv = opt_a.get("p_values", {})
        zs = opt_a.get("z_scores", {})
        nd = opt_a.get("null_distribution", {})
        ns_nd = nd.get("n_signal_words", {})
        perm_snippet = (
            f"1000 random syllabary tables (Option A, "
            f"inventory={opt_a.get('inventory_size', '?')}) produced "
            f"mean {ns_nd.get('mean', '?'):.1f} signal words "
            f"(std {ns_nd.get('std', '?'):.1f}) vs T_P15's "
            f"{rt.get('n_signal_words', '?')} "
            f"(z={zs.get('n_signal_words', '?'):.2f}, "
            f"p={pv.get('n_signal_ge_real', '?'):.4f}). "
            f"Mean selectivity: null "
            f"{nd.get('mean_selectivity', {}).get('mean', '?'):.2f}x "
            f"vs real {rt.get('mean_selectivity', '?'):.2f}x "
            f"(z={zs.get('mean_selectivity', '?'):.2f}, "
            f"p={pv.get('mean_sel_ge_real', '?'):.4f}). "
            f"Verdict: {perm_verdict}."
        )
    else:
        print("  WARNING: reviewer_permutation.json not found")

    # ── Load Analysis 2 (rabidi) ──
    rab_path = os.path.join(rd, "reviewer_rabidi.json")
    rab_snippet = "(not yet run)"
    rab_verdict = "PENDING"
    if os.path.exists(rab_path):
        with open(rab_path) as f:
            rab = json.load(f)
        analyses["rabidi"] = rab
        rab_verdict = rab.get("verdict", "UNKNOWN")
        wr = rab.get("with_rabidi", {})
        wo = rab.get("without_rabidi", {})
        impact = rab.get("impact", {})
        rab_snippet = (
            f"Removing all {rab.get('rabidi_entries', {}).get('n_entries', '?')} "
            f"rabidi identifications reduces the T1 catalog to "
            f"{wo.get('n_entries', '?')} entries across "
            f"{wo.get('n_distinct_words', '?')} distinct words. "
            f"Corpus coverage: {wo.get('corpus_coverage', 0):.1%} "
            f"(vs {wr.get('corpus_coverage', 0):.1%}). "
            f"CI overlap: {wo.get('ci_overlap', 0):.1%} "
            f"(vs {wr.get('ci_overlap', 0):.1%}). "
            f"Paradigms: {rab.get('paradigm_impact', {}).get('paradigms_without_rabidi', '?')} "
            f"of {rab.get('paradigm_impact', {}).get('total_paradigms', '?')} unaffected. "
            f"Verdict: {rab_verdict}."
        )
    else:
        print("  WARNING: reviewer_rabidi.json not found")

    # ── Load Analysis 3 (fingerprint) ──
    fp_path = os.path.join(rd, "reviewer_fingerprint.json")
    fp_snippet = "(not yet run)"
    fp_verdict = "PENDING"
    if os.path.exists(fp_path):
        with open(fp_path) as f:
            fp = json.load(f)
        analyses["fingerprint"] = fp
        fp_verdict = "DISCRIMINATES" if fp.get("discriminates") else "WEAK"
        r6 = fp.get("rank_6", {})
        t5 = fp.get("top_5_range", {})
        fp_snippet = (
            f"The 6th-ranked profile ({r6.get('profile', '?')}) scores "
            f"cosine = {r6.get('cosine', 0):.6f}, representing a gap of "
            f"{fp.get('gap_5_to_6', 0):.6f} below the top-5 cluster "
            f"({t5.get('min', 0):.6f}–{t5.get('max', 0):.6f}). "
            f"{fp.get('interpretation', '')}"
        )
    else:
        print("  WARNING: reviewer_fingerprint.json not found")

    # ── Summary table ──
    result = {
        "test": "reviewer_integration",
        "verdicts": {
            "permutation": perm_verdict,
            "rabidi": rab_verdict,
            "fingerprint": fp_verdict,
        },
        "paper_snippets": {
            "permutation": perm_snippet,
            "rabidi": rab_snippet,
            "fingerprint": fp_snippet,
        },
        "n_analyses_completed": sum(
            1 for v in [perm_verdict, rab_verdict, fp_verdict]
            if v != "PENDING"
        ),
        "runtime_seconds": time.time() - t0,
    }

    # ── Print ──
    print("\n--- Analysis 1: Random Syllabary Permutation ---")
    print(f"  Verdict: {perm_verdict}")
    print(f"  {perm_snippet}")

    print("\n--- Analysis 2: Rabidi Sensitivity ---")
    print(f"  Verdict: {rab_verdict}")
    print(f"  {rab_snippet}")

    print("\n--- Analysis 3: Fingerprint Gap ---")
    print(f"  Verdict: {fp_verdict}")
    print(f"  {fp_snippet}")

    # ── Save ──
    out_path = os.path.join(rd, "reviewer_integrate.json")
    with open(out_path, "w") as f:
        json.dump(_convert(result), f, indent=2)
    print(f"\nSaved to {out_path}")
