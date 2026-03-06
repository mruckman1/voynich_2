"""
Phase A.3a -- Reference Source Validation
==========================================
Validate all user-prepared paleographic JSON files against their schemas.

Checks each expected data file for existence, JSON validity, required fields,
stroke vocabulary conformance, and duplicate sign_ids.  Produces a detailed
per-file validation report and a gate decision (all_valid).

Dependency chain:
    data/reference/tironian/schmitz_plates.json
    data/reference/tironian/chatelain_bobbio.json
    data/reference/cappelli/cappelli_entries.json
    data/reference/costamagna/costamagna_signs.json
    data/reference/ligature/ligature_observations.json
    data/reference/fontana/fontana_signs.json
    data/reference/milanese/milanese_cipher_keys.json
        -> ref_validate.json (this step)
"""

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

from voynich.core._paths import data_dir as _data_dir, results_dir as _results_dir
from voynich.core.reference import (
    VALID_FIRST_STROKES,
    VALID_GLYPH_CLASSES,
    VALID_LAST_STROKES,
    validate_stroke_fields,
)


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
class FileReport:
    """Validation report for a single reference data file."""
    source_type: str
    file_path: str
    file_exists: bool
    n_entries: int
    n_errors: int
    errors: List[str]
    n_warnings: int
    warnings: List[str]


@dataclass
class RefValidateResult:
    reports: List[Dict]
    n_sources_found: int
    n_sources_valid: int
    total_signs: int
    total_errors: int
    all_valid: bool
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Expected file definitions
# ---------------------------------------------------------------------------

EXPECTED_FILES = [
    ('schmitz',    'data/reference/tironian/schmitz_plates.json'),
    ('chatelain',  'data/reference/tironian/chatelain_bobbio.json'),
    ('cappelli',   'data/reference/cappelli/cappelli_entries.json'),
    ('costamagna', 'data/reference/costamagna/costamagna_signs.json'),
    ('ligature',   'data/reference/ligature/ligature_observations.json'),
    ('fontana',    'data/reference/fontana/fontana_signs.json'),
    ('milanese',   'data/reference/milanese/milanese_cipher_keys.json'),
]


# ---------------------------------------------------------------------------
# Per-source validators
# ---------------------------------------------------------------------------

def _validate_tironian(data: Dict, source_type: str) -> FileReport:
    """Validate schmitz_plates.json or chatelain_bobbio.json."""
    errors: List[str] = []
    warnings: List[str] = []

    signs = data.get('signs')
    if signs is None:
        errors.append("Missing top-level 'signs' array")
        return FileReport(
            source_type=source_type, file_path='', file_exists=True,
            n_entries=0, n_errors=len(errors), errors=errors,
            n_warnings=0, warnings=warnings,
        )

    if not isinstance(signs, list):
        errors.append("'signs' is not a list")
        return FileReport(
            source_type=source_type, file_path='', file_exists=True,
            n_entries=0, n_errors=len(errors), errors=errors,
            n_warnings=0, warnings=warnings,
        )

    required_fields = [
        'sign_id', 'latin_expansion', 'first_stroke', 'last_stroke',
        'glyph_class', 'triple_key', 'confidence',
    ]
    seen_ids: set = set()

    for i, entry in enumerate(signs):
        # Required fields
        for fld in required_fields:
            if fld not in entry:
                errors.append(f"signs[{i}]: missing required field '{fld}'")

        # Duplicate sign_id
        sid = entry.get('sign_id', '')
        if sid:
            if sid in seen_ids:
                errors.append(f"signs[{i}]: duplicate sign_id '{sid}'")
            seen_ids.add(sid)

        # Stroke vocabulary
        stroke_errs = validate_stroke_fields(entry)
        errors.extend(stroke_errs)

    return FileReport(
        source_type=source_type, file_path='', file_exists=True,
        n_entries=len(signs), n_errors=len(errors), errors=errors,
        n_warnings=len(warnings), warnings=warnings,
    )


def _validate_cappelli(data: Dict) -> FileReport:
    """Validate cappelli_entries.json."""
    errors: List[str] = []
    warnings: List[str] = []

    entries = data.get('entries')
    if entries is None:
        errors.append("Missing top-level 'entries' array")
        return FileReport(
            source_type='cappelli', file_path='', file_exists=True,
            n_entries=0, n_errors=len(errors), errors=errors,
            n_warnings=0, warnings=warnings,
        )

    if not isinstance(entries, list):
        errors.append("'entries' is not a list")
        return FileReport(
            source_type='cappelli', file_path='', file_exists=True,
            n_entries=0, n_errors=len(errors), errors=errors,
            n_warnings=0, warnings=warnings,
        )

    required_fields = ['entry_id', 'latin_expansion', 'category', 'confidence']
    seen_ids: set = set()

    for i, entry in enumerate(entries):
        for fld in required_fields:
            if fld not in entry:
                errors.append(f"entries[{i}]: missing required field '{fld}'")

        eid = entry.get('entry_id', '')
        if eid:
            if eid in seen_ids:
                errors.append(f"entries[{i}]: duplicate entry_id '{eid}'")
            seen_ids.add(eid)

        # If has_standalone_sign is true, stroke fields are required
        if entry.get('has_standalone_sign'):
            stroke_fields = ['first_stroke', 'last_stroke', 'glyph_class', 'triple_key']
            for sfld in stroke_fields:
                if sfld not in entry:
                    errors.append(
                        f"entries[{i}] ({eid}): has_standalone_sign=true "
                        f"but missing '{sfld}'"
                    )
            stroke_errs = validate_stroke_fields(entry)
            errors.extend(stroke_errs)

    return FileReport(
        source_type='cappelli', file_path='', file_exists=True,
        n_entries=len(entries), n_errors=len(errors), errors=errors,
        n_warnings=len(warnings), warnings=warnings,
    )


def _validate_costamagna(data: Dict) -> FileReport:
    """Validate costamagna_signs.json."""
    errors: List[str] = []
    warnings: List[str] = []

    families = data.get('sign_families')
    if families is None:
        errors.append("Missing top-level 'sign_families' array")
        return FileReport(
            source_type='costamagna', file_path='', file_exists=True,
            n_entries=0, n_errors=len(errors), errors=errors,
            n_warnings=0, warnings=warnings,
        )

    if not isinstance(families, list):
        errors.append("'sign_families' is not a list")
        return FileReport(
            source_type='costamagna', file_path='', file_exists=True,
            n_entries=0, n_errors=len(errors), errors=errors,
            n_warnings=0, warnings=warnings,
        )

    total_members = 0
    seen_ids: set = set()

    for fi, family in enumerate(families):
        if 'family_id' not in family:
            errors.append(f"sign_families[{fi}]: missing 'family_id'")
        if 'members' not in family:
            errors.append(f"sign_families[{fi}]: missing 'members' array")
            continue

        members = family.get('members', [])
        if not isinstance(members, list):
            errors.append(f"sign_families[{fi}]: 'members' is not a list")
            continue

        for mi, member in enumerate(members):
            total_members += 1

            required = [
                'sign_id', 'syllable_value', 'word_position',
                'first_stroke', 'last_stroke', 'glyph_class',
            ]
            for fld in required:
                if fld not in member:
                    fam_id = family.get('family_id', '?')
                    errors.append(
                        f"sign_families[{fi}].members[{mi}] "
                        f"(family={fam_id}): missing '{fld}'"
                    )

            sid = member.get('sign_id', '')
            if sid:
                if sid in seen_ids:
                    errors.append(
                        f"sign_families[{fi}].members[{mi}]: "
                        f"duplicate sign_id '{sid}'"
                    )
                seen_ids.add(sid)

            stroke_errs = validate_stroke_fields(member)
            errors.extend(stroke_errs)

    return FileReport(
        source_type='costamagna', file_path='', file_exists=True,
        n_entries=total_members, n_errors=len(errors), errors=errors,
        n_warnings=len(warnings), warnings=warnings,
    )


def _validate_ligature(data: Dict) -> FileReport:
    """Validate ligature_observations.json."""
    errors: List[str] = []
    warnings: List[str] = []

    pair_summaries = data.get('pair_summaries')
    if pair_summaries is None:
        errors.append("Missing top-level 'pair_summaries' array")
        return FileReport(
            source_type='ligature', file_path='', file_exists=True,
            n_entries=0, n_errors=len(errors), errors=errors,
            n_warnings=0, warnings=warnings,
        )

    if not isinstance(pair_summaries, list):
        errors.append("'pair_summaries' is not a list")
        return FileReport(
            source_type='ligature', file_path='', file_exists=True,
            n_entries=0, n_errors=len(errors), errors=errors,
            n_warnings=0, warnings=warnings,
        )

    required_fields = [
        'eva_pair', 'n_examined', 'connection_rate', 'proposed_as_ligature',
    ]

    for i, ps in enumerate(pair_summaries):
        for fld in required_fields:
            if fld not in ps:
                errors.append(f"pair_summaries[{i}]: missing '{fld}'")

        # Verify count consistency: n_connected + n_separated + n_ambiguous == n_examined
        n_examined = ps.get('n_examined', 0)
        n_connected = ps.get('n_connected', 0)
        n_separated = ps.get('n_separated', 0)
        n_ambiguous = ps.get('n_ambiguous', 0)
        computed_total = n_connected + n_separated + n_ambiguous
        if n_examined > 0 and computed_total != n_examined:
            eva_pair = ps.get('eva_pair', '?')
            errors.append(
                f"pair_summaries[{i}] ({eva_pair}): count mismatch — "
                f"n_connected({n_connected}) + n_separated({n_separated}) + "
                f"n_ambiguous({n_ambiguous}) = {computed_total} != "
                f"n_examined({n_examined})"
            )

    return FileReport(
        source_type='ligature', file_path='', file_exists=True,
        n_entries=len(pair_summaries), n_errors=len(errors), errors=errors,
        n_warnings=len(warnings), warnings=warnings,
    )


def _validate_fontana(data: Dict) -> FileReport:
    """Validate fontana_signs.json."""
    errors: List[str] = []
    warnings: List[str] = []

    signs = data.get('signs')
    if signs is None:
        errors.append("Missing top-level 'signs' array")
        return FileReport(
            source_type='fontana', file_path='', file_exists=True,
            n_entries=0, n_errors=len(errors), errors=errors,
            n_warnings=0, warnings=warnings,
        )

    if not isinstance(signs, list):
        errors.append("'signs' is not a list")
        return FileReport(
            source_type='fontana', file_path='', file_exists=True,
            n_entries=0, n_errors=len(errors), errors=errors,
            n_warnings=0, warnings=warnings,
        )

    required_fields = [
        'sign_id', 'plaintext_value', 'sign_category',
        'first_stroke', 'last_stroke', 'glyph_class',
    ]
    seen_ids: set = set()

    for i, entry in enumerate(signs):
        for fld in required_fields:
            if fld not in entry:
                errors.append(f"signs[{i}]: missing required field '{fld}'")

        sid = entry.get('sign_id', '')
        if sid:
            if sid in seen_ids:
                errors.append(f"signs[{i}]: duplicate sign_id '{sid}'")
            seen_ids.add(sid)

        stroke_errs = validate_stroke_fields(entry)
        errors.extend(stroke_errs)

    return FileReport(
        source_type='fontana', file_path='', file_exists=True,
        n_entries=len(signs), n_errors=len(errors), errors=errors,
        n_warnings=len(warnings), warnings=warnings,
    )


def _validate_milanese(data: Dict) -> FileReport:
    """Validate milanese_cipher_keys.json."""
    errors: List[str] = []
    warnings: List[str] = []

    ciphers = data.get('ciphers')
    if ciphers is None:
        errors.append("Missing top-level 'ciphers' array")
        return FileReport(
            source_type='milanese', file_path='', file_exists=True,
            n_entries=0, n_errors=len(errors), errors=errors,
            n_warnings=0, warnings=warnings,
        )

    if not isinstance(ciphers, list):
        errors.append("'ciphers' is not a list")
        return FileReport(
            source_type='milanese', file_path='', file_exists=True,
            n_entries=0, n_errors=len(errors), errors=errors,
            n_warnings=0, warnings=warnings,
        )

    total_signs = 0
    for ci, cipher in enumerate(ciphers):
        if 'cipher_id' not in cipher:
            errors.append(f"ciphers[{ci}]: missing 'cipher_id'")

        cipher_signs = cipher.get('signs')
        if cipher_signs is None:
            errors.append(
                f"ciphers[{ci}] ({cipher.get('cipher_id', '?')}): "
                f"missing 'signs' array"
            )
        elif isinstance(cipher_signs, list):
            total_signs += len(cipher_signs)
        else:
            errors.append(
                f"ciphers[{ci}] ({cipher.get('cipher_id', '?')}): "
                f"'signs' is not a list"
            )

    return FileReport(
        source_type='milanese', file_path='', file_exists=True,
        n_entries=total_signs, n_errors=len(errors), errors=errors,
        n_warnings=len(warnings), warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_VALIDATORS = {
    'schmitz':    lambda d: _validate_tironian(d, 'schmitz'),
    'chatelain':  lambda d: _validate_tironian(d, 'chatelain'),
    'cappelli':   _validate_cappelli,
    'costamagna': _validate_costamagna,
    'ligature':   _validate_ligature,
    'fontana':    _validate_fontana,
    'milanese':   _validate_milanese,
}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_ref_validate() -> None:
    """Phase A.3a: Validate all user-prepared paleographic reference files."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE A.3a: Reference Source Validation")
    print("=" * 70)

    rd = _results_dir()
    dd = _data_dir()

    # ─── Step 1: Check existence of each expected data file ───
    print("\n  1. Checking existence of expected data files ...")
    reports: List[FileReport] = []
    n_found = 0

    for source_type, rel_path in EXPECTED_FILES:
        full_path = os.path.join(str(dd), *rel_path.split('/')[1:])
        exists = os.path.isfile(full_path)
        status = "FOUND" if exists else "MISSING"
        print(f"      [{status:>7}] {rel_path}")

        if not exists:
            reports.append(FileReport(
                source_type=source_type, file_path=rel_path,
                file_exists=False, n_entries=0, n_errors=0, errors=[],
                n_warnings=0, warnings=[],
            ))
            continue

        n_found += 1

        # ─── Step 2: Parse and validate ───
        try:
            with open(full_path) as f:
                data = json.load(f)
        except json.JSONDecodeError as exc:
            reports.append(FileReport(
                source_type=source_type, file_path=rel_path,
                file_exists=True, n_entries=0, n_errors=1,
                errors=[f"Invalid JSON: {exc}"],
                n_warnings=0, warnings=[],
            ))
            continue

        validator = _VALIDATORS.get(source_type)
        if validator is None:
            reports.append(FileReport(
                source_type=source_type, file_path=rel_path,
                file_exists=True, n_entries=0, n_errors=1,
                errors=[f"No validator defined for source_type '{source_type}'"],
                n_warnings=0, warnings=[],
            ))
            continue

        report = validator(data)
        report.file_path = rel_path
        reports.append(report)

    # ─── Step 3: Print detailed report per file ───
    print(f"\n  2. Validation results ({n_found}/{len(EXPECTED_FILES)} files found):")

    total_signs = 0
    total_errors = 0
    n_valid = 0

    for rpt in reports:
        if not rpt.file_exists:
            print(f"\n      [{rpt.source_type}] {rpt.file_path}")
            print(f"        Status: NOT FOUND (skipped)")
            continue

        total_signs += rpt.n_entries
        total_errors += rpt.n_errors
        is_valid = rpt.n_errors == 0
        if is_valid:
            n_valid += 1

        status_label = "VALID" if is_valid else "INVALID"
        print(f"\n      [{rpt.source_type}] {rpt.file_path}")
        print(f"        Status: {status_label}")
        print(f"        Entries: {rpt.n_entries}")
        print(f"        Errors:  {rpt.n_errors}")

        if rpt.errors:
            for err in rpt.errors[:10]:
                print(f"          - {err}")
            if len(rpt.errors) > 10:
                print(f"          ... and {len(rpt.errors) - 10} more errors")

        if rpt.n_warnings > 0:
            print(f"        Warnings: {rpt.n_warnings}")
            for warn in rpt.warnings[:5]:
                print(f"          - {warn}")

    # ─── Step 4: Gate decision ───
    # all_valid = all found files pass validation (files that don't exist
    # are not counted as failures -- they are simply absent sources)
    all_valid = (n_found > 0) and (n_valid == n_found)

    if all_valid:
        verdict = (
            f"PASS: All {n_found} found source files are valid. "
            f"{total_signs} total signs across {n_found} sources, "
            f"0 errors."
        )
    else:
        verdict = (
            f"FAIL: {n_found - n_valid} of {n_found} found source files "
            f"have validation errors ({total_errors} total errors). "
            f"Fix errors before merging."
        )

    print(f"\n  3. Summary:")
    print(f"      Sources found:  {n_found} / {len(EXPECTED_FILES)}")
    print(f"      Sources valid:  {n_valid} / {n_found}")
    print(f"      Total signs:    {total_signs}")
    print(f"      Total errors:   {total_errors}")

    print(f"\n  Gate: {'PASS' if all_valid else 'FAIL'}")
    print(f"  {verdict}")

    # ─── Save ───
    result = RefValidateResult(
        reports=[_convert(asdict(r)) for r in reports],
        n_sources_found=n_found,
        n_sources_valid=n_valid,
        total_signs=total_signs,
        total_errors=total_errors,
        all_valid=all_valid,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(str(rd), 'ref_validate.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)

    print(f"\n  -> {out_path}")
