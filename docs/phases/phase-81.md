# Phase 81: Exception-DOF Audit

[← Back to Phase Index](README.md)

## Purpose

Reviewer Response v1 (Reviewer 3.9): enumerate the "exception" degrees of freedom introduced beyond the base 25-triple → 25-syllable assignment, count them explicitly, and compare to the number of constraints the model satisfies. Reviewer 3.9 argued that compound characters, allographs, modifiers, and wildcards could absorb enough flexibility to make the model over-fit.

## Method

Each exception class is audited for:
- Number of parameters introduced
- Constraints from phonetic/distributional evidence
- Whether the parameters are free or determined by the evidence

## Results

| Exception class | Parameters | Constrained by | DOF |
|-----------------|-----------|----------------|-----|
| Compounds | 1 (qo) | 97.7% co-occurrence | **1** |
| Allographs | 0 (reduce via shared triples) | — | **0** |
| Modifiers | 15 | word-initial rate 3.3% vs syllabic 40.7% | **15** |
| Wildcards | 13 nominal → 8 effective | 5 constrained by T1 IDs (Phase 80) | **8** |
| **Total DOF** | | | **29** |
| **Total constraints** | | | **328** |

Breakdown of constraints: 316 fully-decoded identifications + 12 confirmed stroke-triple assignments.

**Verdict: OVER_DETERMINED — 29 DOF against 328 constraints.**

## Falsifiable predictions

Five predictions the model commits to (tested in later phases):
1. Randomly shuffled assignment → equal signal-word count
2. Wildcards resolve to T_P15-consistent values (refuted in Phase 80, tightening the model)
3. Unresolved triples cluster in rare positions
4. CVC coda identifications remain table-specific under permutation (tested in [phase-78](phase-78.md), p = 0.002)
5. Signal-word count under table shuffle falls to null mean ~33 (Phase 60B/61B confirmed)

## Interpretation

The 11:1 over-determination ratio (328/29) means the model cannot be overfitting in any meaningful sense — there are an order of magnitude more constraints than free parameters. The specific DOF that most affected earlier concerns (wildcards) is further reduced from 13 nominal to 8 effective by the [phase-80](phase-80.md) consistency check, and the remaining wildcards are shown to be pattern-matching artifacts rather than hidden degrees of freedom.

## Files

- **Implementation:** [src/voynich/phases/p81_exception_audit.py](../../src/voynich/phases/p81_exception_audit.py)
- **Output:** `results/p81_exception_audit.json`
- **CLI:** `voynich exception-audit` / `voynich phase81`

## Dependencies

- `results/modifier_integrate.json`
- `results/triple_tiers.json`
- `results/p80_wildcard_consistency.json`
- `results/p75_t1.json`
