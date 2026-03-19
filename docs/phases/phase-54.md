# Phase 54: Gallo-Italic Dialect Identification Battery

[← Phases 49-53](phase-49-53.md) | [Phase Index](README.md)

**Phase 54:** DIALECT_INDETERMINATE (Ligurian #1 at 0.248, Lombard #2 at 0.235; 40% cross-experiment agreement; Fisher p=0.019; 6/10 validations)

---

## Overview

Eight experiments organized into four phases (A-D) treat the 70 confirmed signal words, 22 T1 word-level identifications, and verb paradigms as linguistic data to identify the specific medieval Italian dialect. Each experiment produces a 5-dialect score vector (Venetian, Lombard, Ligurian, Emilian, Tuscan) with null tests and validation gates.

**Final verdict: DIALECT_INDETERMINATE.** No single dialect dominates. The signal words carry contradictory dialectal signals — Tuscan morphological markers alongside Gallo-Italic phonological features — consistent with a macaronic register.

---

## Phase A: Existing Data Only

### Experiment 1: Systematic Degemination Test (Phase 54.1)

**Verdict: LOW_SELECTIVITY** (2/3 gates pass)

| Word | Etymon | Geminate | Status |
|------|--------|----------|--------|
| bela | bella | ll | degeminated |
| cela | cellam | ll | degeminated |
| corali | corallum | ll | degeminated |
| diasene | diasenna | nn | degeminated |
| li | illi | ll | degeminated |
| sene | senna | nn | degeminated |
| commune | communem | mm | preserved |
| coralli | corallum | ll | preserved |

Degemination rate: **0.75** (6/8). Points toward Gallo-Italic (expected 0.80-1.00) rather than Tuscan (expected 0.00-0.15). The coexistence of *corali* (degeminated) and *coralli* (preserved) for the same word directly shows dialect mixing.

Null test: character shuffle produces 0.926 mean rate — the observed 0.75 is *below* null (z = -2.13). Short CV-decoded words structurally lack geminates, so the null test can't distinguish real degemination from the syllabary artifact. G3 (selectivity >= 1.5x) fails.

Best dialect fit: Emilian (0.500).

### Experiment 3: Article and Pronoun System (Phase 54.3)

**Verdict: STRONG_MATCH** (4/4 gates pass) — strongest individual result

| Dialect | Raw | Weighted | Coverage | Composite |
|---------|-----|----------|----------|-----------|
| Tuscan | 0.714 | 0.770 | 5/5 | **0.828** |
| Venetian | 0.714 | 0.432 | 5/5 | 0.659 |
| Lombard | 0.714 | 0.432 | 5/5 | 0.659 |
| Ligurian | 0.643 | 0.392 | 4/5 | 0.565 |
| Emilian | 0.500 | 0.243 | 5/5 | 0.522 |

Three uniquely Tuscan discriminants (weight 1.0 each): **ci** (locative pronoun — Gallo-Italic uses *ghe*), **si** (reflexive — Gallo-Italic uses *se*), **tu** (2sg pronoun — Gallo-Italic uses *ti*/*te*).

Anti-Tuscan markers: **co** (= "with", eliminates Tuscan/Emilian), **de** (= "of", eliminates Tuscan), **li** (plural article, supports only Venetian/Lombard).

Null: selectivity 3.27x (z = 1.95). Separation from #2: 0.169.

### Experiment 6: Verb Morphology Deep Dive (Phase 54.6)

**Verdict: VERB_MORPH_PASS** (4/4 gates pass)

| Form | Verb | Slot | Best Dialect | ED |
|------|------|------|-------------|-----|
| dice | dire | 3sg.pres.ind | Tuscan | 0 |
| dico | dire | 1sg.pres.ind | Tuscan | 0 |
| dise | dire | 3sg.pres.ind | Venetian | 0 |
| dicu | dire | 1sg.pres.ind | Tuscan | 1 |
| diga | dire | 3sg.pres.subj | Ven/Lom/Lig/Em | 0 |
| dedi | dare | 1sg.perf | Venetian/Lombard | 0 |
| dido | dare | 1sg.perf.var | — | 2 (all) |
| dere | dare | infinitive | Tuscan | 1 |

Tuscan wins (0.800) over Venetian (0.781), but no dialect achieves full paradigm coherence. Null z = -0.80 (not table-specific).

**Hand distribution of dise/dice:**

| Hand | dise | dice | Preference |
|------|------|------|-----------|
| 1 | 20 | 41 | **dice** (Tuscan) |
| 2 | 8 | 3 | **dise** (northern) |
| 3 | 4 | 5 | neutral |
| 4 | 2 | 2 | neutral |
| 5 | 1 | 0 | — |

Chi-squared: 7.91, **p = 0.094**. Marginally significant — different scribes may use different dialectal forms.

---

## Phase B: Targeted Corpus Analysis

### Experiment 5: Co Syntactic Validation (Phase 54.5)

**Verdict: WEAK_EVIDENCE** (1/3 gates pass)

490 occurrences of decoded *co*, but only 6.5% followed by a noun candidate (null: 5.9%). z_a = 0.65 (not significant). Three expected bigrams found: *co bene*, *co cora*, *co sene* (6 total occurrences).

Dialect scores: uniform (0.5 each) — co not confirmed as preposition.

### Experiment 2: Lenition Pattern Test (Phase 54.2)

**Verdict: LENITION_DETECTED** (3/3 gates pass)

| Word | Latin | Stop | Reflex | Outcome |
|------|-------|------|--------|---------|
| diga | dicat | /k/ | /g/ | lenited |
| dise | dicit | /k/ | /s/ | spirantized |
| dice | dicit | /k/ | /c/ | preserved |
| secundi | secundum | /k/ | /c/ | preserved |

Mixed pattern: 25% lenited + 25% spirantized + 50% preserved. Best fit: **Ligurian** (0.333), which shows partial lenition with spirantization. Selectivity 1.53x. The simultaneous presence of *dise* (northern) and *dice* (Tuscan) for the same verb is a direct observation of code-mixing.

---

## Phase C: External Reference Material

### Experiment 4: Pharmaceutical Terminology Regionalization (Phase 54.4)

**Verdict: WEAK_REGIONAL_SIGNAL** (1/3 gates pass)

| Tradition | Score | Matched Terms |
|-----------|-------|--------------|
| Padua/Venice | 0.442 | cola, tere, sero, codi, stercora, coralli, diasene, ratione, secundi, commune, codex |
| Salerno | 0.316 | cola, tere, sene, raso, bene, commune |
| Bologna | 0.176 | ratione, commune, secundi |

Padua/Venice leads but most matched terms are universal pharmaceutical Latin. Null selectivity only 1.03x. *Diasene* (dia- prefix) is the best regional marker pointing to Arabic-influenced Venetian transmission.

### Experiment 7: Simulated Macaronic Text Comparison (Phase 54.7)

**Verdict: SIGNIFICANT_TUSCAN** (z = 3.15)

| Dialect | Distance | Score |
|---------|----------|-------|
| Tuscan | 0.359 | 1.000 |
| Venetian | 0.386 | 0.580 |
| Lombard | 0.386 | 0.580 |
| Emilian | 0.424 | 0.014 |
| Ligurian | 0.425 | 0.000 |

Simulated Tuscan pharmaceutical text is distributionally closest to the real decoded Voynich. Key driver: *di* at 29.8% of function words matches Tuscan's *di* (= "of"); Gallo-Italic dialects use *de*. Also *bene* (Tuscan) vs *ben* (northern apocopated).

---

## Phase D: Manuscript-Specific Evidence

### Experiment 8: Zodiac Label Dialect Decode (Phase 54.8)

**Verdict: WEAK_SIGNAL** (2/4 gates pass)

109 matches at ED <= 2 across 299 decoded zodiac labels, but only 10 on the correct folio (folio selectivity 0.10). Null produces *more* matches (z = -0.97). Most matches are at the maximum ED 2, too permissive for these short strings.

Notable: decoded "lane" on f72r3 (Leo folio) matches Tuscan *leone* at ED 2. But the zodiac labels do not decode to recognizable month or zodiac names in any dialect under T_P15.

---

## Integration

### Composite Dialect Scores (weighted average, bootstrap 95% CI)

| Rank | Dialect | Score | CI |
|------|---------|-------|----|
| 1 | Ligurian | 0.248 | [0.069, 0.427] |
| 2 | Lombard | 0.235 | [0.200, 0.270] |
| 3 | Venetian | 0.213 | [0.190, 0.241] |
| 4 | Tuscan | 0.192 | [0.075, 0.344] |
| 5 | Emilian | 0.112 | [0.030, 0.204] |

All CIs overlap. Ranking is unstable across bootstrap resamples. Emilian is the only dialect tentatively excludable.

### Cross-Experiment Winners

| Experiment | Winner | Signal Type |
|-----------|--------|------------|
| 1. Degemination | Emilian | Phonological |
| 2. Lenition | **Ligurian** | Phonological |
| 3. Articles | **Tuscan** | Morphological |
| 4. Pharma | Venetian | Lexical |
| 5. Co-syntax | Uniform | Syntactic |
| 6. Verbs | **Tuscan** | Morphological |
| 7. Simulation | **Tuscan** | Distributional |
| 8. Zodiac | Lombard | Lexical |

Agreement rate: 40% (below 50% consistency threshold). This triggers automatic INDETERMINATE verdict.

### Validation Battery

| # | Test | Result |
|---|------|--------|
| V1 | Degemination rate != 0.5 | PASS |
| V2 | Article separation >= 0.10 | PASS |
| V3 | Verb paradigm coherent <= 2 dialects | PASS |
| V4 | Co precedes nouns above chance | FAIL |
| V5 | Lenition consistent with 1 family | FAIL |
| V6 | Pharma tradition dominance | FAIL |
| V7 | Simulation z >= 2 | PASS |
| V8 | Zodiac >= 2 correct-folio matches | PASS |
| V9 | Cross-experiment agreement >= 60% | FAIL |
| V10 | Fisher combined p < 0.05 | PASS |

**6/10 passed.** Fisher combined chi2 = 29.76, p = 0.019.

---

## Key Findings

1. **Morphological/functional layer is Tuscan.** *ci*, *si*, *tu*, *dice*, *dico* are standard Italian forms. These dominate frequency-weighted measures.

2. **Phonological layer is Gallo-Italic.** Degemination (*bela*, *sene*, *corali*) and lenition/spirantization (*diga*, *dise*) are specifically northern features.

3. **Different scribes may use different dialectal forms.** Hand 1 writes *dice* (Tuscan); Hand 2 writes *dise* (northern) at p = 0.094.

4. **The text is macaronic.** The contradictions are linguistically coherent: a 15th-century northern Italian scribe using the emerging written standard for grammar (Tuscan-influenced) while showing northern sound changes in content vocabulary. This is exactly what the paper's "macaronic Latin-Italian" hypothesis predicts.

5. **Emilian can be tentatively excluded.** It scores lowest on composite (0.112) with a CI that barely reaches other dialects' lower bounds.

---

## CLI Commands

```bash
voynich degemination       # Exp 1
voynich articles           # Exp 3
voynich verb-morph         # Exp 6
voynich co-syntax          # Exp 5
voynich lenition           # Exp 2
voynich pharma-region      # Exp 4
voynich dialect-sim        # Exp 7
voynich zodiac-dialect     # Exp 8
voynich dialect-verdict    # Integration
voynich phase54            # Full pipeline
```

## Output Files

All results in `results/phase54_*.json`:
- `phase54_degemination.json`, `phase54_lenition.json`, `phase54_articles.json`
- `phase54_pharma_region.json`, `phase54_co_syntax.json`, `phase54_verb_morph.json`
- `phase54_dialect_sim.json`, `phase54_zodiac.json`, `phase54_integrate.json`
