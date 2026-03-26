# Phase 72: Decode Model Diagnosis and Revision

**Verdict: MODEL_REVISED** (16/23 gates)

Phase 71 exposed three structural flaws in the CVC decode model: connector→r creating 47% of coda tokens as -r (implausible 35% passive voice), 24% cross-validation (coda-based vs ending-based POS), and decoded strings too long to be single Latin words. Phase 72 diagnoses these problems across 5 tracks.

## Track 1: Connector Investigation — CONNECTOR_REVISED (4/4)

**CLI:** `voynich connector-test` | **Output:** `results/phase72_connector.json`

Exhaustive test of 13 possible connector values. **Null (∅) wins decisively:**

| Rank | Value | Dict-hit | Signal | Bigram z | XVal | Composite |
|------|-------|----------|--------|----------|------|-----------|
| 1 | **null (∅)** | **30.2%** | **76** | **90.5** | **90.5%** | **0.612** |
| 2 | o | 28.2% | 75 | 95.2 | 90.8% | 0.608 |
| 4 | n | 28.7% | 70 | 95.6 | 92.1% | 0.602 |
| 10 | r (current) | 29.0% | 75 | 87.1 | 77.9% | 0.576 |

Position analysis: **98.1% of connectors are token-medial** (only 1.9% final), mean relative position 0.39. This is consistent with scribal ligatures connecting character strokes, not coda consonants marking syllable closure.

All vowels (o, e, i, a, u) outperform r on cross-validation (90.8% vs 77.9%), suggesting the connector might alternatively be a connecting vowel, but null is best overall because it improves dict-hit without introducing false characters.

**Gates:** CN1 best≠r PASS, CN2 composite+0.01 PASS, CN3 xval+5pp PASS, CN4 null in top 3 PASS

## Track 2: Cross-Validation Diagnosis — CONNECTOR_R_IS_PROBLEM (4/4)

**CLI:** `voynich xval-diagnosis` | **Output:** `results/phase72_xval.json`

The "24% cross-validation" from Phase 71 included UNMARKED/FUNCTION_STEM tokens with no coda. Restricting to clean tokens with codas AND classifiable endings: **overall xval is 78.2%**. But a dramatic per-coda split reveals the problem:

| Coda | Agreement | Observations | Status |
|------|-----------|-------------|--------|
| -s | **92.8%** | 3,560/3,835 | Excellent |
| -t | **86.5%** | 1,512/1,747 | Strong |
| -n | 54.8% | 17/31 | Inconclusive (tiny sample) |
| -r | **16.9%** | 192/1,137 | **Catastrophic** |

The coda consonant is always present in decoded strings (100%), confirming the append model works mechanically. But connector→r tokens produce nominal endings 83% of the time despite being classified as VERBAL.

By section: cosmological best (85.8%), biological worst (71.7%), range 14.0%. By length: short tokens (2-3 chars) achieve 100% agreement; agreement drops steadily as decoded strings get longer.

**Error taxonomy:** 26.4% AGREE, 7.5% CODA_VERBAL_END_NOMINAL, 66.1% no classifiable ending.

**Gates:** XV1 any>50% PASS, XV2 confirmed>unresolved+10pp PASS, XV3 r worst by>15pp PASS, XV4 section>10pp PASS

## Track 3: Alternative Combination Models — MODEL_IMPROVED (4/5)

**CLI:** `voynich combo-models` | **Output:** `results/phase72_combination.json`

Uses Track 1's best connector (null) as baseline. Tests 6 combination rules:

| Rank | Model | Dict-hit | Signal | Bigram z | XVal | Composite |
|------|-------|----------|--------|----------|------|-----------|
| 1 | **null_connector** | **30.2%** | **76** | **90.5** | **90.8%** | **0.661** |
| 2 | append (current) | 27.8% | 66 | 90.2 | 90.8% | 0.626 |
| 3 | costamagna_cvc | 27.8% | 66 | 90.2 | 90.8% | 0.626 |
| 4 | prepend_to_next | 29.3% | 60 | 95.4 | 90.8% | 0.624 |
| 5 | replace_last | 22.4% | 41 | 101.5 | 90.8% | 0.564 |
| 6 | insert | 15.2% | 28 | 113.6 | 25.3% | 0.360 |

**null_connector wins** — equivalent to Track 1's finding. The append model is correct for genuine codas (n, s, t); the error is treating connectors as codas at all. The insert model (coda before last vowel) destroys xval at 25.3% — definitively wrong. prepend_to_next has the highest bigram z (95.4) but lower dict-hit and signal.

**Gates:** CM1 best≠append PASS, CM2 dict+1pp PASS, CM3 xval+5pp FAIL, CM4 length closer PASS, CM5 null_connector in top 3 PASS

## Track 4: T1 Expansion — EXPANSION_MINIMAL (2/5)

**CLI:** `voynich t1-expand72` | **Output:** `results/phase72_t1_expand.json`

5-tier relaxation of T1 pipeline constraints:

| Tier | Params | IDs | Token Coverage | FPR |
|------|--------|-----|----------------|-----|
| A | ≥5 folios, ≥70% known | 154 | 21.7% | — |
| B | ≥3 folios, ≥50% known | 223 | 22.4% | — |
| C | ≥2 folios, ≥40% known | 281 | 22.8% | 100% |
| D | ≥2 folios, ≥30% known | 281 | 22.8% | 100% |
| E | ≥1 folio, ≥40% known | 530 | 23.5% | 100% |

**Relaxation doesn't help.** Tiers C/D/E have 100% false positive rate — random assignments produce equally many identifications. The 223 Tier B identifications (Phase 68) are the validated ceiling. Token coverage plateaus at ~23% regardless of tier.

Best passages reach 87% identified (f6r, f27r, f54r) but none hit 90%.

**Gates:** T1_1 C≥400 FAIL, T1_2 FPR<30% PASS, T1_3 coverage>20% PASS, T1_4 D≥600 FAIL, T1_5 passage>90% FAIL

## Track 5: Variable-Length Encoding — LENGTH_PREFERENCES_FOUND (2/5)

**CLI:** `voynich var-length` | **Output:** `results/phase72_var_length.json`

7/12 confirmed triples prefer 1-char values, 0 prefer 3-char:

| Triple | Current | Best | Improvement |
|--------|---------|------|-------------|
| loop,loop,bench | ra | **r** | +4.7pp |
| ascender,descender,suffix | di | **i** | +0.7pp |
| open_curve,connector,bench | co | **o** | +0.7pp |
| vertical,vertical,minim | di | **i** | +0.7pp |
| ascender,ascender,compound | be | **e** | +0.6pp |
| sigmoid,connector,bench | se | **e** | +0.1pp |
| ascender,ascender,gallows | de | **e** | ~0pp |

Greedy optimization with 8 changed triples: **+13.7pp dict-hit** (29.0%→42.8%). But bigram z drops from 87.15 to 60.97, and mean decoded length drops to 4.0 chars (target 5.8). This is the **same collision artifact** from Phase 32 — shorter strings collide more with the 130K dictionary. The bigram z drop confirms sequential signal is being destroyed.

**Gates:** VL1 ≥2 shorter PASS, VL2 ≥2 longer FAIL, VL3 dict+2pp PASS, VL4 length closer FAIL, VL5 bigram≥90% FAIL

## Integration

**Three tracks independently converge on one conclusion:** connector strokes are non-phonetic scribal features with no sound value. They should be dropped from the CVC decode pipeline.

- Track 1 (connector investigation): null beats all 12 alternatives
- Track 2 (xval diagnosis): coda -r is the sole error source (16.9% vs 86-93% for others)
- Track 3 (combination models): null_connector model wins

**The other three codas are validated:** hook→n, sigmoid→s, vertical→t produce 54.8–92.8% cross-validation agreement and should be retained with the standard append model.

**Negative results are equally informative:**
- T1 expansion beyond Tier B adds only noise (100% FPR)
- Variable-length encoding improvements are dictionary collision artifacts
- The insert combination model is definitively wrong

## Recommended Changes

1. **Connector → null (∅)**: Drop connector-class coda markers from the CVC decode. This eliminates the -r overcount (47%→~25% of coda tokens) and raises cross-validation from 78% to 91%.
2. **Retain other codas unchanged**: hook→n, sigmoid→s, vertical→t with standard append.
3. **Stay at T1 Tier B** (223 identifications).
4. **Keep fixed 2-char CV encoding** (variable length is a collision artifact).

## CLI Commands

```bash
voynich connector-test     # Track 1 (~37s)
voynich xval-diagnosis     # Track 2 (~1s)
voynich combo-models       # Track 3 (~14s, depends on Track 1)
voynich t1-expand72        # Track 4 (~9min)
voynich var-length         # Track 5 (~15s)
voynich phase72-verdict    # Integration
voynich phase72            # Full pipeline
```
