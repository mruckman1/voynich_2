# Phase 85: German-Optimised Table (Circularity Test)

[← Back to Phase Index](README.md)

## Purpose

Reviewer Response v2: address the "circularity" objection — perhaps T_P15 works for Latin only because the CSP was allowed to optimise for it. Does an equivalent search for a German-optimal table (TPG) produce the same quality of signal?

## Method

Build TPG by running the same CSP pipeline (beam search + 5 iterations) against a German reference corpus, with equivalent computational budget. Then apply both T_P15 and TPG to the Voynich corpus and compare on the three paper-level diagnostics (signal count, dict-hit, coherence).

## Results

| Model | Dict-hit | Signal | Verb paradigm | Function kit | Pharma | Coherence |
|-------|----------|--------|---------------|--------------|--------|-----------|
| **T_P15** (Latin) | 43.6% | **28 words** | ✓ | ✓ | ✓ | **PASS (3/3)** |
| **TPG** (German) | **22.4%** | 20 words | ✗ (0) | ✗ | ✗ | **FAIL (0/3)** |

Equal optimisation effort. German has fewer signal words, worse dict-hit, and fails all three coherence gates.

**Verdict: LATIN_SUPERIOR — circularity objection refuted.**

## Interpretation

If the pipeline were language-agnostic (the circularity worry), TPG should match or exceed T_P15's performance against German as T_P15 matches against Latin. Instead, TPG is substantially worse on every metric — it finds 28% fewer signal words *and* fails the coherence tests T_P15 passes. The pipeline's success is not a general artifact of CSP optimisation: it is specific to Latin as the target language.

## Files

- **Implementation:** [src/voynich/phases/p85_german_optimized.py](../../src/voynich/phases/p85_german_optimized.py)
- **Output:** `results/p85_german_optimized.json`
- **CLI:** `voynich german-optimized` / `voynich phase85`

## Dependencies

- `results/combined_refine.json`
- `results/cv_labels.json`
