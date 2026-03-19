"""
Phase 55A.3 – Extended Entropy Shift Ranking
=============================================
Adds Schinner and Rugg-Taylor Cardan grille to the Phase 19.2 mechanism
ranking and verifies tachygraphy remains the uniquely positive cosine match.

Dependency chain:
    results/entropy_shift_cipher.json   (existing ranking + Latin baseline)
    results/phase55_schinner_gen.json   (Schinner cosines)
    results/phase55_cardan_gen.json     (Cardan cosines)
        → results/phase55_entropy_extended.json
"""

import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from voynich.core._paths import results_dir as _results_dir


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


def _save_json(rd: str, filename: str, data: Any) -> str:
    path = os.path.join(rd, filename)
    with open(path, 'w') as f:
        json.dump(_convert(data), f, indent=2)
    return path


# ---------------------------------------------------------------------------
# CI overlap check
# ---------------------------------------------------------------------------

def _cis_overlap(lo1: float, hi1: float, lo2: float, hi2: float) -> bool:
    """Return True if two confidence intervals overlap."""
    return lo1 <= hi2 and lo2 <= hi1


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_entropy_extended() -> None:
    """Phase 55A.3: Extended entropy shift ranking with Schinner + Cardan."""
    t0 = time.time()
    rd = str(_results_dir())

    print("=" * 70)
    print("PHASE 55A.3: Extended Entropy Shift Ranking")
    print("=" * 70)

    # ── 1. Load existing Phase 19.2 results ─────────────────────────────
    print("\n  1. Loading Phase 19.2 ranking …")

    phase19 = _safe_load(os.path.join(rd, 'entropy_shift_cipher.json'))
    if not phase19:
        raise FileNotFoundError("entropy_shift_cipher.json not found — run phase19 first")

    existing_ranking = phase19.get('cipher_ranking', [])

    # Build a lookup from mechanism_profiles for CI data
    profiles_by_name = {
        mp.get('name', ''): mp
        for mp in phase19.get('mechanism_profiles', [])
    }

    tachygraphy_entry = None
    for r in existing_ranking:
        if r.get('name', '') == 'tachygraphic':
            mp = profiles_by_name.get('tachygraphic', {})
            tachygraphy_entry = {
                'name': 'tachygraphic',
                'cosine': r.get('cosine_similarity', r.get('cosine', 0.0)),
                'ci_lower': mp.get('ci_lower', 0.0),
                'ci_upper': mp.get('ci_upper', 0.0),
            }
            break

    if tachygraphy_entry is None:
        # Fallback to mechanism_profiles
        mp = profiles_by_name.get('tachygraphic', {})
        if mp:
            tachygraphy_entry = {
                'name': 'tachygraphic',
                'cosine': mp.get('cosine_similarity', 0.0),
                'ci_lower': mp.get('ci_lower', 0.0),
                'ci_upper': mp.get('ci_upper', 0.0),
            }

    print(f"    Existing ranking: {len(existing_ranking)} mechanisms")
    if tachygraphy_entry:
        print(f"    Tachygraphy: cosine={tachygraphy_entry['cosine']:.4f}")

    # ── 2. Load new generators ───────────────────────────────────────────
    print("\n  2. Loading Schinner + Cardan results …")

    schinner_data = _safe_load(os.path.join(rd, 'phase55_schinner_gen.json'))
    cardan_data = _safe_load(os.path.join(rd, 'phase55_cardan_gen.json'))

    if not schinner_data:
        raise FileNotFoundError("phase55_schinner_gen.json not found — run schinner-gen first")
    if not cardan_data:
        raise FileNotFoundError("phase55_cardan_gen.json not found — run cardan-gen first")

    # Extract variant entries
    new_entries: List[Dict] = []

    for variant_name, vdata in schinner_data.get('variants', {}).items():
        entry = {
            'name': variant_name,
            'cosine': vdata.get('mean_cosine', 0.0),
            'ci_lower': vdata.get('ci_lower', 0.0),
            'ci_upper': vdata.get('ci_upper', 0.0),
            'std': vdata.get('std_cosine', 0.0),
            'n_seeds': vdata.get('n_seeds', 20),
        }
        new_entries.append(entry)
        print(f"    {variant_name}: cosine={entry['cosine']:.4f}  "
              f"CI=[{entry['ci_lower']:.4f}, {entry['ci_upper']:.4f}]")

    for variant_name, vdata in cardan_data.get('variants', {}).items():
        entry = {
            'name': variant_name,
            'cosine': vdata.get('mean_cosine', 0.0),
            'ci_lower': vdata.get('ci_lower', 0.0),
            'ci_upper': vdata.get('ci_upper', 0.0),
            'std': vdata.get('std_cosine', 0.0),
            'n_seeds': vdata.get('n_seeds', 20),
        }
        new_entries.append(entry)
        print(f"    {variant_name}: cosine={entry['cosine']:.4f}  "
              f"CI=[{entry['ci_lower']:.4f}, {entry['ci_upper']:.4f}]")

    # ── 3. Merge and re-rank ─────────────────────────────────────────────
    print("\n  3. Building updated ranking …")

    # Normalise existing entries to consistent format
    all_entries: List[Dict] = []
    for r in existing_ranking:
        name = r.get('name', r.get('cipher', 'unknown'))
        mp = profiles_by_name.get(name, {})
        all_entries.append({
            'name': name,
            'cosine': r.get('cosine_similarity', r.get('cosine', 0.0)),
            'ci_lower': mp.get('ci_lower', r.get('ci_lower', 0.0)),
            'ci_upper': mp.get('ci_upper', r.get('ci_upper', 0.0)),
            'std': mp.get('std_cosine', r.get('std', 0.0)),
            'source': 'phase19',
        })

    for e in new_entries:
        e['source'] = 'phase55'
        all_entries.append(e)

    # Sort by cosine descending
    all_entries.sort(key=lambda x: x['cosine'], reverse=True)

    for rank, entry in enumerate(all_entries, 1):
        entry['rank'] = rank

    print(f"    Total mechanisms in updated ranking: {len(all_entries)}")
    for entry in all_entries[:5]:
        print(f"      #{entry['rank']:2d}  {entry['name']:<30s}  cos={entry['cosine']:+.4f}")
    if len(all_entries) > 5:
        print(f"      …")
        for entry in all_entries[-3:]:
            print(f"      #{entry['rank']:2d}  {entry['name']:<30s}  cos={entry['cosine']:+.4f}")

    # ── 4. Gates ─────────────────────────────────────────────────────────
    print("\n  4. Computing gates …")

    # Find tachygraphy in merged ranking
    tachy = next((e for e in all_entries if e['name'] == 'tachygraphic'), None)
    tachy_lo = tachy['ci_lower'] if tachy else 0.0
    tachy_hi = tachy['ci_upper'] if tachy else 1.0
    tachy_cos = tachy['cosine'] if tachy else 0.0

    def _find(name: str) -> Optional[Dict]:
        return next((e for e in all_entries if e['name'] == name), None)

    schinner_simple = _find('schinner_simple')
    schinner_pos = _find('schinner_positional')
    cardan_3 = _find('cardan_3hole')
    cardan_4 = _find('cardan_4hole')

    def _gate_below(entry: Optional[Dict], label: str) -> bool:
        """Check that entry's CI is strictly below tachygraphy's CI (upper bound of entry < lower bound of tachy)."""
        if entry is None:
            print(f"    {label}: MISSING (False)")
            return False
        # entry must be below tachygraphy: entry.ci_upper < tachy.ci_lower
        below = entry['ci_upper'] < tachy_lo
        direction = 'BELOW' if below else ('ABOVE' if entry['ci_lower'] > tachy_hi else 'OVERLAPPING')
        passed = below
        print(f"    {label}: {'PASS' if passed else 'FAIL'} [{direction}] "
              f"(tachy [{tachy_lo:.4f},{tachy_hi:.4f}] vs "
              f"entry [{entry['ci_lower']:.4f},{entry['ci_upper']:.4f}])")
        return passed

    g1 = _gate_below(schinner_simple, "G1 schinner_simple CI below tachygraphy")
    g2 = _gate_below(schinner_pos, "G2 schinner_positional CI below tachygraphy")
    g3 = _gate_below(cardan_3, "G3 cardan_3hole CI below tachygraphy")
    g4 = _gate_below(cardan_4, "G4 cardan_4hole CI below tachygraphy")
    g5 = (all_entries[0]['name'] == 'tachygraphic') if all_entries else False
    print(f"    G5 tachygraphy rank 1: {'PASS' if g5 else 'FAIL'}")

    gates = {'G1': g1, 'G2': g2, 'G3': g3, 'G4': g4, 'G5': g5}
    n_passed = sum(gates.values())

    # ── 5. Verdict ───────────────────────────────────────────────────────
    # Check if Schinner is above tachygraphy (the failure mode where trained-on-Voynich model wins)
    schinner_above = (
        schinner_simple is not None and schinner_simple['ci_lower'] > tachy_hi or
        schinner_pos is not None and schinner_pos['ci_lower'] > tachy_hi
    )

    if g5 and n_passed >= 4:
        verdict = 'TACHYGRAPHY_UNIQUE'
    elif g5 and n_passed >= 2:
        if not g1 or not g2:
            verdict = 'SCHINNER_OVERLAPS'
        else:
            verdict = 'CARDAN_OVERLAPS'
    elif g5:
        verdict = 'TACHYGRAPHY_RANK1_PARTIAL'
    elif schinner_above:
        verdict = 'SCHINNER_ABOVE_TACHYGRAPHY'
    else:
        verdict = 'TACHYGRAPHY_DISPLACED'

    print(f"\n  VERDICT: {verdict}  ({n_passed}/5 gates passed)")

    # ── 6. Tachygraphy uniquely positive? ────────────────────────────────
    positive_mechanisms = [e for e in all_entries if e['cosine'] > 0]
    tachygraphy_uniquely_positive = (
        len(positive_mechanisms) == 1 and
        positive_mechanisms[0]['name'] == 'tachygraphic'
    )
    print(f"  Tachygraphy uniquely positive cosine: {tachygraphy_uniquely_positive}")
    print(f"  Mechanisms with positive cosine: {[e['name'] for e in positive_mechanisms]}")

    # ── 7. Save ──────────────────────────────────────────────────────────
    runtime = round(time.time() - t0, 2)
    output = {
        'phase': '55A.3',
        'experiment': 'entropy_shift_extended',
        'voynich_shift': phase19.get('observed_shift_vector', []),
        'updated_ranking': all_entries,
        'n_mechanisms_total': len(all_entries),
        'tachygraphy_rank': next((e['rank'] for e in all_entries if e['name'] == 'tachygraphic'), None),
        'tachygraphy_cosine': tachy_cos,
        'tachygraphy_uniquely_positive': tachygraphy_uniquely_positive,
        'schinner_above_tachygraphy': schinner_above,
        'positive_cosine_mechanisms': [e['name'] for e in positive_mechanisms],
        'new_mechanisms': {e['name']: {
            'cosine': e['cosine'],
            'ci': [e['ci_lower'], e['ci_upper']],
            'n_seeds': e.get('n_seeds', 20),
        } for e in new_entries},
        'gates': gates,
        'n_gates_passed': n_passed,
        'verdict': verdict,
        'runtime_seconds': runtime,
    }

    out_path = _save_json(rd, 'phase55_entropy_extended.json', output)
    print(f"\n  Saved → {out_path}")
    print(f"  Completed in {runtime:.1f}s")
