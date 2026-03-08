"""
Step 25.3 – Phase 25 Combined Verdict
======================================
Integrate boustrophedon re-ordering and f6r examination results into a
combined decision about the manuscript's reading direction and decode quality.

Dependency chain:
    boustrophedon_decode.json (Step 25.1)
    f6r_manual.json (Step 25.2)
        → phase25_verdict.json (this step)
"""

import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict

from voynich.core._paths import results_dir as _results_dir


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
# Decision matrix
# ---------------------------------------------------------------------------

DECISION_MATRIX = {
    # (boustrophedon_verdict, f6r_verdict) -> (action, description)
    ('CONFIRMED', 'READABLE_LATIN'): (
        'BREAKTHROUGH',
        'Expand f6r method to all herbal folios with boustrophedon; '
        'write paper with decoded passage.',
    ),
    ('CONFIRMED', 'PARTIAL_LATIN'): (
        'BREAKTHROUGH',
        'Expand f6r method to all herbal folios with boustrophedon; '
        'write paper with decoded passage.',
    ),
    ('CONFIRMED', 'DOMAIN_MATCH'): (
        'DIRECTION_FINDING',
        'Boustrophedon is real but decode accuracy still insufficient; '
        'include direction finding in paper.',
    ),
    ('CONFIRMED', 'GIBBERISH'): (
        'DIRECTION_FINDING',
        'Boustrophedon is real but decode accuracy still insufficient; '
        'include direction finding in paper.',
    ),
    ('SUGGESTIVE', 'READABLE_LATIN'): (
        'FOLIO_DECODE',
        'f6r is decodable without direction correction; '
        'focus on expanding folio-by-folio decode.',
    ),
    ('SUGGESTIVE', 'PARTIAL_LATIN'): (
        'FOLIO_DECODE',
        'f6r shows partial Latin content; '
        'focus on expanding folio-by-folio decode.',
    ),
    ('SUGGESTIVE', 'DOMAIN_MATCH'): (
        'STRUCTURAL_ONLY',
        'Both tests show weak positive signals; '
        'paper claims structural identification with partial domain evidence.',
    ),
    ('SUGGESTIVE', 'GIBBERISH'): (
        'STRUCTURAL_ONLY',
        'Both tests negative; '
        'write paper with structural findings only.',
    ),
    ('NOT_CONFIRMED', 'READABLE_LATIN'): (
        'FOLIO_DECODE',
        'f6r is decodable without direction correction; '
        'focus on expanding folio-by-folio decode.',
    ),
    ('NOT_CONFIRMED', 'PARTIAL_LATIN'): (
        'FOLIO_DECODE',
        'f6r shows partial Latin content; '
        'focus on expanding folio-by-folio decode.',
    ),
    ('NOT_CONFIRMED', 'DOMAIN_MATCH'): (
        'STRUCTURAL_ONLY',
        'Direction test negative, domain match only; '
        'paper claims structural identification with domain evidence.',
    ),
    ('NOT_CONFIRMED', 'GIBBERISH'): (
        'STRUCTURAL_ONLY',
        'Both tests negative; '
        'write paper with structural findings only.',
    ),
}


@dataclass
class Phase25VerdictResult:
    timestamp: str
    # Input verdicts
    boustrophedon_verdict: str
    boustrophedon_detail: str
    f6r_verdict: str
    f6r_detail: str
    # Combined
    combined_action: str
    combined_description: str
    # Paper readiness
    paper_claim: str
    paper_includes_decoded_passage: bool
    paper_includes_direction_finding: bool
    next_steps: str
    # Overall
    verdict: str
    runtime_seconds: float


def run_phase25_verdict() -> None:
    """Step 25.3: Combined Phase 25 verdict."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 25.3: Phase 25 Combined Verdict")
    print("=" * 70)

    rdir = _results_dir()

    # ── 1. Load results ──────────────────────────────────────────────────
    print("\n  1. Loading results …")

    boustro_path = rdir / "boustrophedon_decode.json"
    f6r_path = rdir / "f6r_manual.json"

    if not os.path.exists(boustro_path):
        print(f"    [SKIP] {boustro_path} not found — run boustro first")
        return
    if not os.path.exists(f6r_path):
        print(f"    [SKIP] {f6r_path} not found — run f6r-exam first")
        return

    with open(boustro_path) as f:
        boustro_data = json.load(f)
    with open(f6r_path) as f:
        f6r_data = json.load(f)

    boustro_verdict = boustro_data.get('boustrophedon_verdict', 'NOT_CONFIRMED')
    boustro_detail = boustro_data.get('verdict', '')
    f6r_verdict = f6r_data.get('f6r_verdict', 'GIBBERISH')
    f6r_detail = f6r_data.get('verdict', '')

    print(f"    Boustrophedon: {boustro_verdict}")
    print(f"    f6r: {f6r_verdict}")

    # ── 2. Apply decision matrix ─────────────────────────────────────────
    print("\n  2. Applying decision matrix …")

    key = (boustro_verdict, f6r_verdict)
    action, description = DECISION_MATRIX.get(
        key,
        ('STRUCTURAL_ONLY', 'Unknown combination; defaulting to structural findings only.'),
    )

    print(f"    Combined action: {action}")
    print(f"    {description}")

    # ── 3. Paper readiness ───────────────────────────────────────────────
    print("\n  3. Paper readiness assessment …")

    includes_passage = action == 'BREAKTHROUGH'
    includes_direction = action in ('BREAKTHROUGH', 'DIRECTION_FINDING')

    if action == 'BREAKTHROUGH':
        paper_claim = (
            "The paper includes a decoded passage from folio f6r and "
            "a confirmed boustrophedon reading direction as concrete "
            "results beyond structural identification."
        )
        next_steps = (
            "1. Expand f6r decode method to all herbal folios with boustrophedon ordering. "
            "2. Validate decoded passages against known botanical descriptions. "
            "3. Write paper with decoded passage as primary result."
        )
    elif action == 'DIRECTION_FINDING':
        paper_claim = (
            "The paper includes a confirmed reading direction finding "
            "(boustrophedon in specific sections) as a structural result. "
            "Full decipherment requires improved decode accuracy."
        )
        next_steps = (
            "1. Include direction finding in paper. "
            "2. Re-run all readability tests on boustrophedon-reordered text. "
            "3. Investigate decode accuracy bottleneck."
        )
    elif action == 'FOLIO_DECODE':
        paper_claim = (
            "The paper includes partial decode of folio f6r showing "
            "Latin medical content consistent with Calendula description. "
            "Focus on expanding folio-by-folio decode."
        )
        next_steps = (
            "1. Expand detailed decode to top 5 herbal folios. "
            "2. Build plant-specific vocabularies for each identified folio. "
            "3. Write paper with partial decode results."
        )
    else:
        paper_claim = (
            "The paper claims structural identification (cipher type, "
            "source language, content domain) with precise diagnosis of "
            "remaining gap and names specific next steps."
        )
        next_steps = (
            "1. Write paper with structural findings only. "
            "2. Next steps: Costamagna material, archival access, "
            "multispectral imaging for damaged folios."
        )

    print(f"    Paper claim: {paper_claim}")
    print(f"    Includes decoded passage: {includes_passage}")
    print(f"    Includes direction finding: {includes_direction}")
    print(f"    Next steps: {next_steps}")

    elapsed = time.time() - t0

    # ── 4. Overall verdict ───────────────────────────────────────────────
    verdict = (
        f"{action}: boustrophedon={boustro_verdict}, f6r={f6r_verdict}. "
        f"{description}"
    )
    print(f"\n  4. Overall verdict: {verdict}")

    # ── 5. Save ──────────────────────────────────────────────────────────
    result = Phase25VerdictResult(
        timestamp=time.strftime('%Y-%m-%dT%H:%M:%S'),
        boustrophedon_verdict=boustro_verdict,
        boustrophedon_detail=boustro_detail,
        f6r_verdict=f6r_verdict,
        f6r_detail=f6r_detail,
        combined_action=action,
        combined_description=description,
        paper_claim=paper_claim,
        paper_includes_decoded_passage=includes_passage,
        paper_includes_direction_finding=includes_direction,
        next_steps=next_steps,
        verdict=verdict,
        runtime_seconds=round(elapsed, 2),
    )

    out_path = rdir / "phase25_verdict.json"
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2, ensure_ascii=False)

    print(f"\n  → {out_path} ({elapsed:.1f}s)")
