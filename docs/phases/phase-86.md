# Phase 86: Self-Similar Word Analysis

[← Back to Phase Index](README.md)

## Purpose

Reviewer Response v2: decompose the 10.25% "self-similar word" rate in the Voynich — tokens containing consecutive repeated characters or sequences (e.g., `ee`, `dd`, `dydy`, `olol`) — to determine whether it is structurally anomalous or within the natural range for any short-alphabet writing system.

## Method

Classify every self-similar token in the Voynich into three bins:
- **Consecutive-char artifacts** — doubled single characters (`ee`, `dd`)
- **Short sequence repeats** — doubled 2-char sequences (`dydy`)
- **Triple-or-longer repeats** — `dydydy`, `olololol`

Compare rates against:
- Latin reference corpus under identical metric
- Tachygraphic simulation (synthetic cipher output)

## Results

| Category | Voynich | Latin ref | Tachy sim |
|----------|---------|-----------|-----------|
| Total self-similar | **10.25%** | 9.69% | 9.75% |
| Consecutive-char artifacts (%) | 99.7% | — | — |
| Short sequence repeats (%) | 0.3% | — | — |
| Triple+ repeats (%) | **0.0%** | — | — |

Specific examples:
- `dydydy`: 2 total corpus occurrences (not a productive pattern)
- `olol`: 15 occurrences → `nene` under T_P15 (attested Latin/Italian form)
- `oror`: 8 occurrences → decodes identically (allographic equivalence)

**Verdict: MOSTLY_EXPLAINED** — 99.7% of the rate is ordinary doubled characters, statistically indistinguishable from Latin and from tachygraphic simulation. No genuine triple-repeat phenomenon exists.

## Interpretation

The "self-similar words" observation from Timm-Schinner was over-interpreted. The 10.25% rate is dominated by unremarkable doubled characters (the same `ee`/`dd`/`ss` patterns that occur in any orthography). True reduplication — triple-or-longer repeats that would indicate a productive morphological pattern — is absent. The handful of short-sequence repeats decode to attested Latin/Italian forms under the model.

## Files

- **Implementation:** [src/voynich/phases/p86_self_similar_decode.py](../../src/voynich/phases/p86_self_similar_decode.py)
- **Output:** `results/p86_self_similar_decode.json`
- **CLI:** `voynich self-similar` / `voynich phase86`

## Dependencies

- `results/combined_refine.json`
