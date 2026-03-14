"""
Phase 52 Integration: Word-Level Identification Catalog
=======================================================
Combine results from Track A (word catalog), Track B (validation),
and Track C (structural reading) to produce overall verdict.

Dependency chain:
    word_catalog.json          (Track A)
    word_validation.json       (Track B)
    word_reading.json          (Track C)
        -> phase52_integrate.json
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from voynich.core._paths import results_dir as _results_dir


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
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Phase52Validation:
    name: str
    description: str
    passed: bool
    value: float
    threshold: float


@dataclass
class Phase52IntegrateResult:
    # Track A
    n_catalog_entries: int
    n_tier1: int
    n_tier2: int
    n_tier3: int
    catalog_coverage: float
    # Track B
    null_selectivity: float
    null_z_score: float
    n_paradigms: int
    n_botanical_matches: int
    signal_enrichment: float
    # Track C
    best_folio: str
    best_coverage: float
    longest_run: int
    longest_run_folio: str
    longest_run_text: str
    overall_coverage: float
    circa_instans_overlap: float
    # Validation battery
    validations: List[Dict]
    n_passed: int
    n_total: int
    # Verdict
    verdict: str
    gate_passed: bool
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_phase52_integrate() -> None:
    """Phase 52 Integration: verdict from all three tracks."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE 52 INTEGRATION: Word-Level Identification Catalog")
    print("=" * 70)

    rd = _results_dir()

    # ── Load track results ───────────────────────────────────────────
    print("\n  Loading track results...")

    catalog = _safe_load(os.path.join(rd, 'word_catalog.json'))
    validation = _safe_load(os.path.join(rd, 'word_validation.json'))
    reading = _safe_load(os.path.join(rd, 'word_reading.json'))

    # Extract metrics
    n_t1 = catalog.get('n_tier1', 0)
    n_t2 = catalog.get('n_tier2', 0)
    n_t3 = catalog.get('n_tier3', 0)
    n_catalog = n_t1 + n_t2
    cat_coverage = catalog.get('corpus_coverage', 0.0)

    null_test = validation.get('null_test', {})
    selectivity = null_test.get('selectivity', 0.0)
    null_z = null_test.get('z_score', 0.0)
    n_paradigms = validation.get('n_paradigms', 0)
    n_botanical = validation.get('n_botanical_matches', 0)
    sig_enrichment = validation.get('signal_adjacency_enrichment', 0.0)

    best_folio = reading.get('best_folio', '')
    best_cov = reading.get('best_folio_coverage', 0.0)
    longest_run = reading.get('longest_run_length', 0)
    longest_folio = reading.get('longest_run_folio', '')
    longest_text = reading.get('longest_run_text', '')
    overall_cov = reading.get('overall_coverage', 0.0)
    circa_overlap = reading.get('circa_instans_overlap', 0.0)

    print(f"       Catalog: T1={n_t1}, T2={n_t2}, T3={n_t3}")
    print(f"       Coverage: {cat_coverage:.1%}")
    print(f"       Null selectivity: {selectivity:.2f}×")
    print(f"       Paradigms: {n_paradigms}")
    print(f"       Botanical: {n_botanical}")
    print(f"       Longest run: {longest_run}")
    print(f"       Circa Instans overlap: {circa_overlap:.1%}")

    # ── Validation battery ───────────────────────────────────────────
    print("\n  Validation battery...")

    validations = [
        Phase52Validation(
            'V1_catalog_size', 'T1+T2 identifications >= 20',
            n_catalog >= 20, float(n_catalog), 20.0,
        ),
        Phase52Validation(
            'V2_coverage', 'Corpus coverage >= 30%',
            overall_cov >= 0.30, overall_cov, 0.30,
        ),
        Phase52Validation(
            'V3_null_selectivity', 'Null selectivity > 1.5x',
            selectivity > 1.5, selectivity, 1.5,
        ),
        Phase52Validation(
            'V4_paradigms', 'Morphological paradigms >= 3',
            n_paradigms >= 3, float(n_paradigms), 3.0,
        ),
        Phase52Validation(
            'V5_botanical', 'Botanical matches >= 1',
            n_botanical >= 1, float(n_botanical), 1.0,
        ),
        Phase52Validation(
            'V6_longest_run', 'Longest readable run >= 8',
            longest_run >= 8, float(longest_run), 8.0,
        ),
        Phase52Validation(
            'V7_circa_instans', 'Circa Instans overlap >= 10%',
            circa_overlap >= 0.10, circa_overlap, 0.10,
        ),
    ]

    n_passed = sum(1 for v in validations if v.passed)
    for v in validations:
        status = 'PASS' if v.passed else 'FAIL'
        print(f"       {v.name}: {v.value:.2f} vs {v.threshold:.2f} → {status}")

    print(f"\n       Passed: {n_passed} / {len(validations)}")

    # ── Verdict ──────────────────────────────────────────────────────
    v_map = {v.name: v for v in validations}

    if (v_map['V1_catalog_size'].passed and v_map['V3_null_selectivity'].passed):
        if v_map['V6_longest_run'].passed:
            verdict = 'READABLE_PASSAGES'
        elif v_map['V2_coverage'].passed and v_map['V4_paradigms'].passed:
            verdict = 'VOCABULARY_EXPANDED'
        else:
            verdict = 'CATALOG_VALID'
    elif n_catalog >= 5:
        verdict = 'CATALOG_MARGINAL'
    else:
        verdict = 'CATALOG_EMPTY'

    gate_passed = verdict in ('READABLE_PASSAGES', 'VOCABULARY_EXPANDED',
                               'CATALOG_VALID')

    print(f"\n  VERDICT: {verdict}")
    print(f"  Gate: {'PASS' if gate_passed else 'FAIL'}")

    # ── Save ─────────────────────────────────────────────────────────
    runtime = round(time.time() - t0, 2)

    result = Phase52IntegrateResult(
        n_catalog_entries=n_catalog,
        n_tier1=n_t1,
        n_tier2=n_t2,
        n_tier3=n_t3,
        catalog_coverage=cat_coverage,
        null_selectivity=selectivity,
        null_z_score=null_z,
        n_paradigms=n_paradigms,
        n_botanical_matches=n_botanical,
        signal_enrichment=sig_enrichment,
        best_folio=best_folio,
        best_coverage=best_cov,
        longest_run=longest_run,
        longest_run_folio=longest_folio,
        longest_run_text=longest_text[:300],
        overall_coverage=overall_cov,
        circa_instans_overlap=circa_overlap,
        validations=[asdict(v) for v in validations],
        n_passed=n_passed,
        n_total=len(validations),
        verdict=verdict,
        gate_passed=gate_passed,
        runtime_seconds=runtime,
    )

    out_path = _save_json(rd, 'phase52_integrate.json', asdict(result))
    print(f"\n  Saved → {out_path}")
    print(f"  Completed in {runtime:.1f}s")


def run_phase52() -> None:
    """Run all Phase 52 tracks + integration."""
    from voynich.phases.word_catalog import run_word_catalog
    from voynich.phases.word_validation import run_word_validation
    from voynich.phases.word_reading import run_word_reading

    run_word_catalog()
    print()
    run_word_validation()
    print()
    run_word_reading()
    print()
    run_phase52_integrate()
