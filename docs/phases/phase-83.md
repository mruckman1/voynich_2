# Phase 83: Cross-Language Signal Comparison

[← Back to Phase Index](README.md)

## Purpose

Reviewer Response v1 (Reviewer 3.10): test whether the T_P15 assignment's signal is specific to Latin, or whether it would produce comparable signal against any Romance/IE dictionary. Reviewer 3.10 asked whether German, Italian, or even Hebrew reference corpora would show similar counts, which would undermine the Latin-specific claim.

## Method

Apply the identical T_P15 assignment to generate decoded Voynich tokens, then match against four 10K-word reference dictionaries: Latin, Italian, German, Hebrew. For each language, measure signal word count and the three coherence criteria from the paper:
- **Verb paradigm** — ≥3 conjugations of a single verb
- **Function kit** — ≥4/5 Romance function-word categories covered
- **Pharma register** — ≥3 medical/pharmaceutical terms

## Results

| Language | Signal words | Verb paradigm | Function kit | Pharma | Coherence |
|----------|--------------|---------------|--------------|--------|-----------|
| Latin (paper) | 56 | ✓ (5 *dire* forms) | ✓ (4/5) | ✓ (cola/sero/codi/sene/tere/raso) | **PASS** (p=0.011) |
| German (TPG) | **15** | ✗ (0) | **✓ (4/5)** | ✗ (0) | FAIL (1/3) |
| Italian | 8 | — | — | — | FAIL |
| Hebrew | 0 | — | — | — | FAIL (sel 0.74×) |

**Verdict: WEAKLY_DISCRIMINATING — coherence is the discriminator, not raw count.**

## Interpretation

German's 15 signal words (vs Latin's 23 on the same 10K-dict basis) suggested at first glance that the assignment might work for any Germanic-alphabet language. But the coherence criteria separate signal from noise:

- German passes function-kit only (conjunctions, articles, pronouns that happen to be short strings)
- German fails verb-paradigm — zero multi-form conjugation sets
- German fails pharma-register — zero medical terms

The paper's Latin result passes all three coherence criteria simultaneously at p = 0.011, which is the genuinely discriminating metric. Hebrew is cleanly eliminated at selectivity 0.74× (below random). Italian is marginal but directionally consistent with the paper's macaronic-mixing finding from [phase-54](phase-54.md).

**Key lesson:** raw signal-word counts are not the right diagnostic; the coherence of *what kind* of word appears is.

## Files

- **Implementation:** [src/voynich/phases/p83_language_signal.py](../../src/voynich/phases/p83_language_signal.py)
- **Output:** `results/p83_language_signal.json`
- **CLI:** `voynich lang-signal` / `voynich phase83`

## Dependencies

- `results/combined_refine.json`
- `data/reference/{latin,italian,german,hebrew}/*.txt`
