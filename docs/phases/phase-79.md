# Phase 79: Known-Properties Stress Test

[← Back to Phase Index](README.md)

## Purpose

Reviewer Response v1 (Reviewer 3.3): test the tachygraphic model against seven well-documented Voynichese "known properties" from the secondary literature. For each, classify the verdict as EXPLAINED, PARTIAL, or LIMITATION. This closes the gap between the paper's selectivity-based evidence and the descriptive statistics that earlier researchers (Currier, Tiltman, Stallings, Lindemann-Bowern, Timm-Schinner) took as defining the manuscript.

## Method

Seven sub-tests, each producing a `SubTestResult` dataclass with an observed value, a model prediction, and a verdict:

1. **QO pairing** — co-occurrence rate of EVA `q` and `o`
2. **Frequency-connectivity correlation** (Timm & Schinner 2020) — Spearman ρ between log(type frequency) and ED-1 neighbor count
3. **Positional restrictions** — distribution of characters over word-initial and word-final positions
4. **Self-similar words** — rate of tokens with consecutive repeated characters/sequences
5. **Conditional entropy** — H₂ of the manuscript vs. Latin at character and syllable levels
6. **Two-part word structure** — prefix/suffix mutual information and cross-combination rate (Tiltman)
7. **Inventory sufficiency** — coverage of Latin by 21 confirmed CV syllables + CVC codas

## Results

| # | Property | Observed | Verdict |
|---|----------|----------|---------|
| 1 | QO pairing | 97.7% co-occurrence; 1 DOF | EXPLAINED |
| 2 | Freq-connectivity | Spearman ρ = 0.618 | EXPLAINED |
| 3 | Positional restrictions | Top 4 final chars are coda markers | PARTIAL |
| 4 | Self-similar words | 10.25% (Latin reference 9.69%) | PARTIAL |
| 5 | Conditional entropy | Voynich H₂ = 3.39 vs Latin char 3.47 / syl 4.14 | PARTIAL |
| 6 | Two-part structure | MI = 1.808; 86% cross-combination | PARTIAL |
| 7 | Inventory sufficiency | 21 CV × 4 codas = 84 effective; 14.4% Latin | LIMITATION |

**Verdict: 2 EXPLAINED, 4 PARTIAL, 1 LIMITATION** — the tachygraphic model systematically accounts for or partially predicts every known property, with inventory sufficiency the sole structural limitation (addressed in [phase-84](phase-84.md)).

## Interpretation

The two EXPLAINED properties (QO pairing, freq-connectivity) are specific predictions of the CV-syllable-with-modifiers model, not post-hoc rationalisations. The four PARTIAL properties are consistent with the model but also compatible with other mechanisms — they distinguish the Voynich from pure natural language but do not uniquely identify tachygraphy. The LIMITATION (inventory) is the theoretical ceiling: with 21 confirmed syllables the model can only cover 14.4% of reference Latin.

## Files

- **Implementation:** [src/voynich/phases/p79_known_properties.py](../../src/voynich/phases/p79_known_properties.py)
- **Output:** `results/p79_known_properties.json`
- **CLI:** `voynich known-props` / `voynich phase79`

## Dependencies

- `results/combined_refine.json` (Phase 15 best_assignment)
