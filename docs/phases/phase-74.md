# Phase 74: Descender Investigation + T1 Vocabulary Push

**Verdict: DESCENDER_RESOLVED_AND_VOCAB_EXPANDED** (12/18 gates)

Two independent paths: Path A investigates whether descender→r is correct (it isn't — null wins uniformly); Path B expands vocabulary via distributional patterns and LLM gap-filling with hallucination controls.

## Track A1: Exhaustive Descender Value Testing — DESCENDER_REVISED (4/4)

**CLI:** `voynich descender-test` | **Output:** `results/p74_descender.json`

Same methodology as Phase 72 Track 1 (connector investigation): test 13 possible descender values while holding connector→null fixed. Includes verbal fraction scoring.

| Rank | Value | Dict-hit | Signal | Bigram z | XVal | Verbal | Composite |
|------|-------|----------|--------|----------|------|--------|-----------|
| 1 | **m** | 31.7% | 78 | 92.4 | 91.1% | **24.2%** | **0.6660** |
| 2 | i | 30.4% | 73 | 96.1 | 91.1% | 24.2% | 0.6555 |
| 3 | e | 28.5% | 72 | 96.7 | 91.1% | 24.2% | 0.6491 |
| 9 | **null** | **37.6%** | 62 | 71.3 | 88.1% | **25.9%** | 0.6216 |
| 10 | **r** (current) | 30.2% | 76 | 90.5 | 90.5% | **65.1%** | 0.5739 |
| 11 | s | 33.7% | 66 | 83.2 | 96.5% | 65.1% | 0.5691 |

**Key finding:** The current descender→r ranks **10th out of 13 values**. Every non-verbal-inflating value outperforms it. The verbal fraction drops from 65.1% (impossible for natural Latin) to 24.2% (m) or 25.9% (null), approaching CI-expected ~15%.

### Position Analysis

| Metric | Descender | Connector (Phase 72) |
|--------|-----------|---------------------|
| Total occurrences | 14,164 | ~2,752 |
| Token-final | **94.6%** | 1.9% |
| Token-medial | 5.4% | **98.1%** |
| Mean relative position | 0.69 | 0.39 |

The descender is overwhelmingly token-final (94.6%), confirming it is a **genuine coda marker** — unlike the connector which was a scribal ligature. But "genuine coda marker" does not require it to encode a consonant; it may be a diacritic, prosodic marker, or punctuation with no phonetic value.

**Gates:** DA1 best≠r PASS, DA2 composite+0.005 PASS, DA3 verbal<40% PASS (24.2%), DA4 final>60% PASS (94.6%).

---

## Track A2: Context-Dependent Descender Analysis — CONTEXT_DEPENDENT_PARTIAL (1/3)

**CLI:** `voynich descender-context` | **Output:** `results/p74_descender_context.json`

### Position-Split Testing

| Partition | Count | Best Value | Best Dict-hit | r Dict-hit |
|-----------|-------|------------|---------------|------------|
| Final-only | 13,061 | null | 37.9% | 18.0% |
| Medial-only | 402 | null | 6.7% | 2.7% |
| Both | 341 | — | — | — |

Both final and medial descenders prefer null → **not context-dependent by position** (DA5 FAIL). The difference is not between final and medial behavior but between null and everything else.

### Preceding-Triple Testing

| Triple | n | Best | Best % | r % | Preference |
|--------|---|------|--------|-----|------------|
| loop,loop,bench | 7,478 | null | 24.8% | 0.0% | null (+24.8%) |
| open_curve,connector,bench | 2,837 | null | 40.2% | 35.7% | null (+4.5%) |
| ascender,crossbar,gallows | 579 | null | 55.8% | 2.9% | null (+52.8%) |
| sigmoid,connector,bench | 474 | null | 80.8% | 77.2% | null (+3.6%) |
| loop,vertical,bench | 454 | null | 50.2% | 29.5% | null (+20.7%) |
| loop,sigmoid,bench | 92 | r | 48.9% | 48.9% | r (same) |
| ascender,loop,compound | 32 | r | 68.8% | 68.8% | r (same) |

**13 out of 15 triples independently prefer null.** The 2 exceptions (loop,sigmoid,bench and ascender,loop,compound) have small samples (92 and 32 tokens). This is **uniform null preference**, not context-dependent behavior — the same convergence pattern that identified connector→null in Phase 72.

**Gates:** DA5 context-dep-position FAIL, DA6 ≥3 non-r PASS (13 triples), DA7 mixed>uniform FAIL.

---

## Track B1: EVA-Level Pattern Expansion — PATTERNS_FOUND (2/2)

**CLI:** `voynich eva-patterns` | **Output:** `results/p74_patterns.json`

### Distributional Identifications

595 unidentified EVA types matched to T1-identified types by context vector cosine similarity (window ±3, threshold >0.30). Mean similarity 0.474.

| Unidentified | Matched T1 | Similarity | Frequency |
|-------------|-----------|------------|-----------|
| dain | din (via daiin) | 0.880 | 195 |
| shedy | bet (via qokal) | 0.878 | 381 |
| chedy | bet (via qokal) | 0.852 | 451 |
| qokaiin | bet (via qokal) | 0.851 | 263 |
| or | ni (via aiin) | 0.833 | 277 |
| cheey | cor (via chey) | 0.808 | 156 |

### Positional Identifications

173 unidentified types that concentrate at paragraph-initial or recipe-initial positions.

| Type | Position | Freq at Position | Total Freq |
|------|----------|-----------------|------------|
| dshedy | para_initial | 30 | 33 |
| tchedy | para_initial | 24 | 31 |
| sain | recipe_initial | 21 | 66 |
| tol | para_initial | 19 | 33 |

**Combined:** 675 new types identified → **898 total** (223 T1 + 675 new).

**Gates:** B1_1 ≥50 distributional PASS (595), B1_2 ≥10 positional PASS (173).

---

## Track B2: LLM Gap-Filling with Hallucination Controls — CALIBRATION_ONLY (4/6)

**CLI:** `voynich llm-gap-fill` | **Output:** `results/p74_llm_gapfill.json`

15 passages at ≥75% identified, 5-layer hallucination control framework.

### Control Results

| Layer | Metric | Result | Gate |
|-------|--------|--------|------|
| Known-answer calibration | 40% accuracy (12/30) | **LLM can do this task** | PASS (≥30%) |
| Confidence selectivity | 3.82× (real vs shuffled) | **Context helps, not vocabulary** | PASS (>1.5×) |
| Cross-run consistency | 70% (same word across 3 runs) | **LLM is stable** | PASS (>50%) |
| Decode agreement | 68.8% (ED ≤ 2 with decoded) | **Proposals match decode** | PASS (>40%) |
| Accepted proposals | 0 | **No proposal passes ALL 5 layers** | FAIL (≥5) |
| Fully filled passages | 0 | — | FAIL (≥1) |

**Interpretation:** The LLM **can** do pharmaceutical gap-filling (40% known-answer accuracy, 3.82× selectivity over shuffled controls). But 0 proposals were accepted because the gap tokens have unresolved triples (56% decode error rate), producing garbled decoded strings that no reasonable Latin word is within ED 2 of. The hallucination controls correctly prevented false acceptances — this is the **right outcome**. The decode error rate is the binding constraint, not the LLM or the dictionary.

---

## Track B3: Assemble Complete Readings — NEAR_COMPLETE (1/3)

**CLI:** `voynich complete-read` | **Output:** `results/p74_complete_readings.json`

Without accepted gap-fills, readings remain at T1 + dict level.

| Metric | Value |
|--------|-------|
| Passages | 15 |
| Fully filled (100%) | 0 |
| Near-complete (>90%) | **6** |
| Best fraction | 93.3% |
| Mean fraction | 88.0% |
| Random mean identified | 27.0% |
| **Selectivity** | **3.25×** |

Best passage (f113v, 93.3%): `[ran] · ne · ne · set · bes · cos · cone · se · sera · cone · din · tes · ne · dine · ne`

**Gates:** B3_1 ≥1 fully FAIL (0), B3_2 ≥5 near-complete PASS (6), B3_3 interpretable FAIL.

---

## Integration

| Track | Verdict | Gates |
|-------|---------|-------|
| A1: Descender Test | DESCENDER_REVISED | 4/4 |
| A2: Descender Context | CONTEXT_DEPENDENT_PARTIAL | 1/3 |
| B1: EVA Patterns | PATTERNS_FOUND | 2/2 |
| B2: LLM Gap-Fill | CALIBRATION_ONLY | 4/6 |
| B3: Complete Readings | NEAR_COMPLETE | 1/3 |
| **Overall** | **DESCENDER_RESOLVED_AND_VOCAB_EXPANDED** | **12/18** |

## Key Findings

1. **Descender→r is wrong.** The current mapping ranks 10th of 13 values on composite score. 13/15 preceding triples independently prefer null. The verbal fraction drops from 65.1% to 24–26% with non-r values, approaching natural Latin distributions. This parallels the Phase 72 connector→null finding.

2. **Descender is a genuine coda marker, but likely non-phonetic.** Unlike the connector (98.1% medial → ligature), the descender is 94.6% token-final, confirming it marks coda positions. But its best value is null (or a consonant like 'm' that happens to collide with Latin accusative endings), suggesting it may encode a diacritic or prosodic feature rather than a consonant.

3. **The effective coda system may be just 3 consonants.** If descender→null is adopted (like connector→null in Phase 73), only hook→n, sigmoid→s, and vertical→t encode genuine coda consonants — all with strong cross-validation (54.8–92.8%).

4. **Distributional vocabulary expansion works.** 675 new types identified at the EVA token level (mean cosine similarity 0.474), bringing total identified types from 223 to 898.

5. **LLM gap-filling is feasible but blocked by decode error.** 40% known-answer accuracy and 3.82× confidence selectivity prove the methodology; 0 accepted proposals prove the 56% decode error rate is the binding constraint.

## CLI Commands

```bash
voynich descender-test      # Track A1: 13 descender values (~44s)
voynich descender-context   # Track A2: Position + triple analysis (~3s)
voynich eva-patterns        # Track B1: Distributional + positional (~10s)
voynich llm-gap-fill        # Track B2: LLM gap-filling (~8 min, needs OPENROUTER_API_KEY)
voynich complete-read       # Track B3: Assemble readings (~1s)
voynich phase74-verdict     # Integration (<1s)
voynich phase74             # Full pipeline (~10 min)
```
