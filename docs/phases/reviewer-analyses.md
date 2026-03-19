# Reviewer Response Analyses

[← Phases 49-53](phase-49-53.md) | [Phase Index](README.md)

**Key results:**
- Permutation test: p=0.001 (signal count), p=0.011 (linguistic coherence)
- Word-level permutation: z=4.63, p=0.009
- Rabidi sensitivity: ROBUST (removal has negligible impact)
- Fingerprint cosine gap: WEAK (top-5 to rank-6 gap only 0.0003)
- Within-family entropy: p=0.070 (directionally correct but not significant)

---

Targeted analyses addressing specific reviewer concerns about the headline results. These are post-hoc validations, not numbered phases -- they don't modify the assignment table or produce new decodings.

- **Files**: `reviewer_permutation.py`, `reviewer_rabidi.py`, `reviewer_fingerprint.py`, `reviewer_integrate.py`, `word_permutation_null.py`
- **CLI**: `reviewer-perm`, `reviewer-rabidi`, `reviewer-fingerprint`, `reviewer-all`, `word-perm-null`
- **Results**: `reviewer_permutation.json`, `reviewer_rabidi.json`, `reviewer_fingerprint.json`, `reviewer_integrate.json`, `word_permutation_null.json`

## Analysis 1: Random Syllabary Permutation Test -- SIGNAL_MARGINAL

**Question**: Does ANY random CV syllable assignment produce ~5.5x selectivity for its top-hitting decoded words, or is 5.5x specific to the T_P15 table?

Phase 50A tested whether shuffling syllables AMONG the 25 existing triples matters (answer: barely, 1.10x). This test is fundamentally different: it asks whether the CHOICE of syllables matters -- whether "di,se,ne,co..." is special compared to "pa,ku,vo,ri..." drawn from the same phonological inventory.

**Method**: 1,000 random assignment tables generated for two inventories -- Option A (21 syllables from T_P15's vocabulary) and Option B (202 syllables from 2-char merged-dictionary words + full CV grid). Each random table decoded the real Voynich corpus AND 5 null corpora with that same random table, computing per-word sigma = (real_count - null_mean) / null_std. Dictionary: merged Latin 10K + Italian 10K (19,363 words).

**Results (Option A -- tighter null, same 21-syllable inventory):**

| Metric | T_P15 (real) | Random null (mean +/- std) | z-score | p-value |
|--------|-------------|--------------------------|---------|---------|
| Signal word count | **56** | 32.7 +/- 7.8 | **+3.00** | **0.001** |
| Mean selectivity | 3.81x | 3.43x +/- 0.59 | +0.65 | 0.259 |
| Selectivity CV | 0.850 | 0.89 +/- 0.16 | +0.35 | -- |

**Results (Option B -- broader null, 202-syllable inventory):**

| Metric | T_P15 (real) | Random null (mean +/- std) | z-score | p-value |
|--------|-------------|--------------------------|---------|---------|
| Signal word count | **56** | 22.6 +/- 4.8 | **+6.93** | **< 0.00001** |
| Mean selectivity | 3.81x | 3.61x +/- 0.47 | +0.41 | 0.341 |

**Verdict: SIGNAL_MARGINAL.** The result splits cleanly along two dimensions:

1. **Signal word COUNT is table-specific (p = 0.001).** T_P15 produces 56 signal words -- far more than the typical random table's ~33 (Option A) or ~23 (Option B). The real assignment table genuinely finds more words where real Voynich differs from null corpora.

2. **Per-word selectivity magnitude is NOT table-specific (p = 0.26).** The 3.81x mean selectivity of T_P15's signal words is not significantly above what random tables produce (3.43x). Random tables find fewer signal words, but those that pass sigma > 2.0 tend to have individually higher selectivity -- because with fewer words clearing the threshold, the survivors are stronger outliers.

## Analysis 1b: Signal Word Coherence Check -- COHERENCE_UNCOMMON

**Question**: Do any of the 1,000 random trials produce signal words that form coherent linguistic structure -- Italian verb paradigms, complete function-word inventories, or pharmaceutical register terms?

**Method**: Same 1,000 Option A trials. For each trial's signal word set, scored three coherence tests: (1) verb paradigm -- >=3 conjugated forms of any single Italian verb; (2) function-word kit -- items from >=4 of 5 Romance clause categories; (3) pharmaceutical register -- >=3 terms from the Circa Instans tradition.

**Results:**

| Coherence test | T_P15 | Random trials (1000) | p-value |
|----------------|-------|---------------------|---------|
| Verb paradigm (>=3 forms) | **Yes** (dire: 5 forms) | 69/1000 (6.9%) | 0.069 |
| Function-word kit (>=4/5 cats) | **Yes** (5/5 categories) | 745/1000 (74.5%) | 0.745 |
| Pharma register (>=3 terms) | **Yes** (8 terms) | 147/1000 (14.7%) | 0.147 |
| **All three simultaneously** | **Yes (3/3)** | **11/1000 (1.1%)** | **0.011** |

Coherence distribution: 0/3 = 201 trials, 1/3 = 648, 2/3 = 140, **3/3 = 11**.

**Verdict: COHERENCE_UNCOMMON.** Only 1.1% of random trials achieved 3/3 coherence (p = 0.011). The function-word kit alone is easy (74.5%). The pharmaceutical register is moderately rare (14.7%). The verb paradigm is uncommon (6.9%) -- having >=3 conjugated forms of a single Italian verb among ~33 signal words requires specific alignment. Having all three simultaneously is rare: T_P15's signal words form Italian verb conjugations (dire: 5 forms) AND a complete clause kit (5/5 categories) AND pharmaceutical terminology (8 terms) -- a combination only 11 random tables out of 1,000 reproduce.

## Analysis 1c: Within-Family Phonetic Entropy -- FAMILY_ARTIFACT

**Question**: Is the low within-family phonetic entropy (minim family: 0.592 bits, selectivity 1.61x vs null) a genuine property of T_P15, or an artifact of the stroke-feature model?

Phase 19.5's null test shuffled characters across families (testing whether the family *grouping* matters). This test keeps families fixed and shuffles syllable *assignments* (testing whether the specific syllable values matter).

**Results:**

| Family | T_P15 | Random (mean +/- std) | z | p |
|--------|-------|---------------------|---|---|
| bench (24) | 1.864 | 1.758 +/- 0.263 | +0.40 | 0.633 |
| compound (3) | 0.918 | 1.078 +/- 0.441 | -0.36 | 0.653 |
| gallows (4) | 0.811 | 1.007 +/- 0.431 | -0.45 | 0.476 |
| minim (7) | 0.592 | 0.989 +/- 0.400 | -0.99 | 0.284 |
| rare (3) | 0.918 | 1.072 +/- 0.423 | -0.36 | 0.675 |
| suffix (3) | 0.000 | 0.650 +/- 0.418 | -1.56 | 0.292 |
| **OVERALL** | **0.851** | **1.092 +/- 0.162** | **-1.49** | **0.070** |

Selectivity (random/real): **1.28x** (Phase 19.5 reported 1.61x using character-shuffle null).

**Verdict: FAMILY_ARTIFACT.** Random syllable assignments to the same visual families produce within-family entropy of 1.092 bits vs T_P15's 0.851 (p = 0.070 -- marginal, not significant at 0.05). No individual family is significant. The minim family (the showcase at 0.592 bits) has p = 0.284. The within-family phonetic regularity is partially an artifact of the feature model's structure: with only 21 syllables distributed across 25 triples in 6 families, random assignments also produce some within-family regularity through pigeonhole effects.

## Analysis 1d: Word-Level Permutation Test -- GENUINE

**Question**: Are the 22 T1 content-vocabulary identifications (ratione, coralli, diasene, etc.) specific to T_P15, or does ANY random assignment table produce comparable T1 counts through the same bridge search pipeline?

Analysis 1 tested signal word counts. This test goes further downstream: for each random table, it identifies that table's own signal words, determines its own confirmed triples, runs the full bridge search with those table-specific anchors, and applies the same scoring/tiering criteria.

**Method**: 1,000 fully-permuted assignment tables (all 25 triple values shuffled). Each table independently: (1) decoded real corpus + 5 null corpora, (2) identified its own signal words and confirmed triples, (3) ran bridge search, (4) scored confidence and applied T1/T2/T3 tiering.

**Results:**

| Metric | T_P15 (real) | Random null (mean +/- std) | z-score | p-value |
|--------|-------------|--------------------------|---------|---------|
| T1 count | **22** | 1.5 +/- 4.4 | **+4.63** | **0.009** |
| Distinct T1 words | **9** | 0.8 +/- 1.9 | **+4.30** | **0.012** |
| CI overlap | 0.889 | 0.220 +/- 0.400 | +1.67 | -- |
| Mean folio spread | 10.1 | 2.4 +/- 4.9 | +1.58 | -- |

T1 count distribution: 744 trials (74.4%) produced **zero** T1 identifications. Only 49 trials (4.9%) reached T1 >= 10. Nine trials (0.9%) reached or exceeded 22.

The infrastructure was NOT the bottleneck: random tables averaged 58.6 +/- 10.5 signal words and confirmed 17.4 +/- 1.2 triples. The bottleneck is **specificity** -- T_P15's partial-decode patterns uniquely match pharmaceutical Latin words recurring across 3+ folios.

**Word-level specificity**: 2 of T_P15's 9 distinct T1 words (radicom, stercora) were **never** produced by any of 1,000 random tables. The remaining 7 appeared rarely: diasene (29/1000), secundi (16/1000), rabidi (8/1000), coralli (7/1000), ratione (5/1000), codex (3/1000), commune (1/1000).

**Verdict: GENUINE.** The 22 T1 word-level identifications are table-specific at p = 0.009 (z = 4.63). 74.4% of random tables produce zero T1, and only 0.9% match T_P15's count.

## Analysis 2: Rabidi Sensitivity -- ROBUST

**Question**: Are the 22 T1 word-level identifications robust to removing *rabidi* (5 of 22 entries, appearing on 60 folios)?

**Results:**

| Metric | All 22 T1 | Without rabidi (17) | Delta |
|--------|-----------|---------------------|-------|
| Distinct Latin words | 9 | 8 | -1 |
| EVA types | 22 | 17 | -5 |
| Folios covered | 133 | 102 | -31 |
| Corpus coverage | 32.0% | 31.6% | -0.4% |
| Circa Instans overlap | 88.9% | 87.5% | -1.4% |
| Morphological paradigms | 20 | 20 | 0 |

**Verdict: ROBUST.** Coverage drops by only 0.4 percentage points. CI overlap remains high (87.5%). All 20 morphological paradigms preserved. The remaining 8 words form a coherent pharmaceutical vocabulary: ratione (method), stercora (medicinal dung), coralli (corals), diasene (senna compound), radicom (root), commune (common), codex (manuscript), secundi (second).

## Analysis 3: Fingerprint Cosine Gap -- WEAK

**Question**: Is there a meaningful gap between the top-5 Latin fingerprint matches and the next-best non-Latin profile?

**Results (top 10):**

| Rank | Profile | Family | Cosine |
|------|---------|--------|--------|
| 1 | latin+simple_substitution | Romance | 0.9857 |
| 2 | latin+raw | Romance | 0.9854 |
| 3 | latin+abbreviation_light | Romance | 0.9853 |
| 4 | occitan+raw | Romance | 0.9852 |
| 5 | latin+nomenclator | Romance | 0.9848 |
| **6** | **occitan+abbreviation_light** | **Romance** | **0.9845** |
| 7 | occitan+simple_substitution | Romance | 0.9839 |
| 8 | occitan+null_insertion | Romance | 0.9832 |
| 9 | german+null_insertion | Germanic | 0.9828 |
| 10 | hebrew+null_insertion | Semitic | 0.9827 |

**Gap rank 5->6: 0.000345 cosine.** Gap rank 5->10: 0.002108. Romance mean (top 20): 0.9832 vs non-Romance mean: 0.9825 (gap = 0.0008). First non-Romance profile at rank 9.

**Verdict: WEAK discrimination.** The gap of 0.000345 is well below the 0.005 threshold. All top-8 profiles are Romance. The fingerprint correctly identifies the Romance language *family* but cannot distinguish Latin from Occitan. The language identification rests on 4 independent methods that DO discriminate: signal word isolation (Phase 36), size-matched OT (Phase 50D), SBM profiling (Phase 46B), and character n-gram analysis (Phase 50D).

## Integration Summary

| Analysis | Verdict | Key Number | Paper Implication |
|----------|---------|------------|-------------------|
| 1. Permutation | **SIGNAL_MARGINAL** | p(n_signal)=0.001, p(mean_sel)=0.26 | T_P15 finds more signal words than random (56 vs 33), but per-word selectivity magnitude is structural |
| 1b. Coherence | **COHERENCE_UNCOMMON** | **11/1000 (1.1%, p=0.011)** | T_P15's signal words form Italian verb paradigms + function kit + pharma register simultaneously -- only 1.1% of random tables do |
| 1c. Family entropy | **FAMILY_ARTIFACT** | p=0.070, sel=1.28x | Within-family phonetic regularity is partially an artifact; tachygraphic argument rests on entropy shift + permutation test instead |
| 1d. Word-level perm | **GENUINE** | **z=4.63, p=0.009** | 22 T1 content-vocabulary IDs are table-specific; 74.4% of random tables produce zero T1; 2/9 distinct words never appear in any of 1,000 random trials |
| 2. Rabidi | **ROBUST** | Coverage delta = -0.4% | Removing rabidi does not materially affect the catalog |
| 3. Fingerprint | **WEAK** | Gap = 0.000345 | Fingerprint identifies Romance family, not Latin specifically |

---

[← Phases 49-53](phase-49-53.md) | [Phase Index](README.md)
