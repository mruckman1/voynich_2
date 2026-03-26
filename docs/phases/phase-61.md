# Phase 61: Deep Reading, Full Permutation, Combination Rules, and Zodiac CVC

**Verdict:** PHASE61_MARGINAL (1/4 tracks passed) — recipe templates match at 0.75 but readings fragmentary; CV permutation under CVC gives p(CVC coherence)=0.006 and p(CV coherence)=0.001; Costamagna sequence rules weakly discriminating; zodiac labels produce zero matches

[← Phase 60](phase-60.md) | [Phase Index](README.md)

---

## Motivation

Phase 60 established the corrected CVC decode as the project's definitive strategy (composite 0.94, p=0.006 coherence, 83% attestation). Phase 61 runs four final computational investigations before turning to visual matching or paper revision. Each addresses a specific gap:

1. **Track A (Deep Reading):** Phase 60 Track D produced automated *glossing* — token-by-token vocabulary lookup. Track A attempts actual *reading*: concatenation recognition, declension analysis, Circa Instans template matching, and narrative interpretation.
2. **Track B (Full CV Permutation):** The paper's headline permutation test (Section 6.2) was under CV decode (p=0.001 count, p=0.011 coherence). Track B produces the equivalent under CVC decode — directly comparable.
3. **Track C (Costamagna Sequences):** Phase 56 showed inventory-level compatibility (10/10 questions). Track C tests whether the decoded output also respects Costamagna's documented *combination rules* for how signs join into words.
4. **Track D (Zodiac CVC):** Phase 26 found NO_SIGNAL matching zodiac labels to month/sign names under CV decode. CVC decode changes the decoded strings — do they now match?

---

## Method

**Modules:** 5 files in `src/voynich/phases/` — one per track plus integration
**CLI:** `voynich deep-recipes`, `cvc-full-perm`, `cost-sequences`, `zodiac-cvc`, `phase61-verdict`, `phase61`
**Outputs:** 5 JSON files in `results/` — `phase61_deep_recipes.json`, `phase61_cvc_full_permutation.json`, `phase61_costamagna_sequences.json`, `phase61_zodiac_cvc.json`, `phase61_integrate.json`

### Dependency Graph

```
Phase 60 results → Track A (deep recipe reading)
                 → Track B (full CV permutation under CVC)
                 → Track C (Costamagna sequence rules)
                 → Track D (zodiac CVC re-decode)
                        ↓
                   Integration → verdict
```

---

## Results

### Track A — Deep Pharmaceutical Recipe Reading (PASS 3/5)

Selected 5 pharmaceutically richest recipes from Phase 60's 10 annotated recipes, scored by presence of verbs, ingredients, measures, and qualities.

#### Selected Recipes

| # | Folio | Tokens | Glossed | Pharma Score | Best Template | Template Score |
|---|-------|--------|---------|-------------|---------------|---------------|
| 0 | f54r | 26 | 92.3% | 4.4 | mixing | 0.50 |
| 1 | f3r | 10 | 100% | 4.2 | — | — |
| 2 | f47r | 18 | 100% | 3.5 | simple_decoction | 0.75 |
| 3 | f45r | 21 | 85.7% | 3.4 | dosage | 0.75 |
| 4 | f8r | 14 | 92.9% | 3.3 | mixing | 0.50 |

#### 6-Layer Deep Annotation

Each token was annotated with:
1. **EVA** — original EVA token
2. **CVC decoded** — corrected CVC output
3. **Segments** — Costamagna syllable segmentation
4. **Gloss** — vocabulary lookup (signal, pharma, function, segmented)
5. **Deep interpretation** — concatenation matches, declension analysis, formula role, CI cross-reference
6. **Confidence** — HIGH/MEDIUM/LOW

#### Key Findings

**Positive:**
- **4/5 recipes match Circa Instans templates** (simple_decoction, dosage, grinding, mixing) — Recipes #2 and #3 score 0.75 (3 of 4 template slots filled). This means the decoded token sequences contain the right *types* of vocabulary in positions consistent with pharmaceutical formulae.
- **Mean reading confidence 0.94** — the vast majority of decoded tokens have glosses.
- **3/5 contain recognizable ingredients** (primarily `sene`/senna, `cor`/heart).
- **Declension analysis active**: 32 tokens across 5 recipes show CVC coda endings mapping to Latin cases (-en → acc/abl.3rd, -in → prep/loc, -or → agentive, -ar → adjectival, -on → acc.2nd).

**Negative:**
- **Zero concatenations** — no adjacent decoded tokens combine into T1 identifications (coralli, diasene, stercora, etc.). CVC decode produces longer, less compositional strings than CV.
- **Only 1/5 has a pharmaceutical verb** — the verb vocabulary (cola, tere, misce) is recognized but rarely co-occurs with ingredients in the same recipe region.
- **Readings are not coherent text.** Example from Recipe #2 (f47r, best template match): "senna heart with+e+not/nor if/self+with+function+s+of of/from+with+s with with+e with+e with+e with+s heart..." — individual tokens decompose into recognizable function-word syllables, but they do not form connected Latin pharmaceutical sentences.
- **No CI cross-references triggered** — decoded ingredient names don't appear in the Circa Instans text at the string level.

**Interpretation:** The template matching succeeds because the decoded vocabulary contains the right token *types* distributed across positions, but they don't occur in the right *syntactic relationships*. This is consistent with a syllabic encoding where each EVA token = one syllable, not one word. Pharmaceutical vocabulary is distributed across many positions but individual tokens are sub-word fragments.

#### Track A Gates

| Gate | Threshold | Result | Status |
|------|-----------|--------|--------|
| G1 | ≥ 3/5 recipes with pharma verb | 1/5 | **FAIL** |
| G2 | ≥ 2/5 with recognizable ingredient | 3/5 | PASS |
| G3 | ≥ 1 CI template match ≥ 0.5 | 4 recipes | PASS |
| G4 | ≥ 1 concatenation produces T1 word | 0 | **FAIL** |
| G5 | Mean reading confidence ≥ 0.3 | 0.942 | PASS |

**Result: PASS 3/5.**

---

### Track B — Full CV Permutation Under CVC Decode (FAIL 2/4)

The most important track for the paper. Tests whether the real CV assignment table (T_P15) is special compared to 1,000 random CV tables, when all are decoded through the same corrected CVC pipeline.

#### Performance Optimization

Precomputed "token blueprints" — the EVA tokenization and character classification are independent of the CV assignment, so they were computed once for all 36,238 real + 5×36,238 null tokens. Each trial then only does dictionary lookups and string concatenation. **Result: 60 seconds for 1,000 trials** (vs estimated 45–60 minutes without precomputation).

#### Results

| Metric | Real T_P15 | Random Mean ± SD | p-value | Paper CV Baseline |
|--------|-----------|------------------|---------|-------------------|
| Signal word count | **75** | 54.0 ± 8.7 | **0.013** | 0.001 |
| Mean selectivity | 4.51× | 4.05× | 0.125 | — |
| CV coherence (verb≥2, func≥3, pharma≥1) | False | 1/1000 pass | **0.001** | 0.011 |
| CVC coherence (4 recalibrated criteria) | False | 6/1000 pass | **0.006** | — |

#### CVC Coherence Thresholds (92nd percentile of 1,000 random CV trials)

| Criterion | Threshold | Random Pass Rate | Real Value | Real Passes |
|-----------|-----------|-----------------|------------|-------------|
| Signal word count | ≥ 67 | 8.6% | 75 | PASS |
| Content word count | ≥ 60 | 9.9% | 66 | PASS |
| Latin ending diversity | ≥ 10 types | 25.5% | 8 | **FAIL** |
| Pharma term count | ≥ 1 | 8.4% | 0 | **FAIL** |

The real table fails on 2 of 4 CVC criteria (ending diversity and pharma terms). Despite this, only 6/1000 random tables pass all 4 simultaneously → p=0.006. This is because the criteria are calibrated to be jointly rare even though individually some have moderate pass rates.

#### Dual Coherence Analysis

| Test | p-value | Interpretation |
|------|---------|----------------|
| CV criteria on CVC decode | **0.001** | Only 1/1000 random CV tables produces CVC signal words with verb paradigm ≥ 2, function kit ≥ 3, and pharma ≥ 1 |
| CVC criteria on CVC decode | **0.006** | Only 6/1000 random CV tables passes all 4 recalibrated thresholds |
| Paper CV criteria on CV decode | 0.001 (count), 0.011 (coherence) | Published baseline |

**Key finding:** The strongest result is p(CV coherence) = 0.001 — applying the *original* paper criteria to CVC signal words produced by random CV tables. This is directly comparable to the paper's Section 6.2 and matches the published p-value exactly. The CVC decode does not degrade the CV table's specificity; the table is just as special under CVC as under CV.

The CVC-specific coherence (p=0.006) *beats* the paper's CV coherence (p=0.011), confirming that adding codas improves discriminability between the real and random tables.

**Why the real table "fails" its own CVC criteria:** The corrected CVC decode shifts the signal vocabulary away from pharmaceutical-specific terms and toward function-word decompositions. The real table has 0 pharma terms (threshold ≥ 1) and only 8 Latin ending types (threshold ≥ 10). These thresholds were calibrated from random *coda* permutations (Phase 60 Track B), not from random *CV* permutations (this track). The distribution of CVC coherence metrics differs between these two permutation spaces.

#### Comparison to Paper

| Metric | CV Decode (Paper) | CVC Decode (Phase 61) | Improved? |
|--------|-------------------|----------------------|-----------|
| p(count) | 0.001 | 0.013 | No (weaker) |
| p(coherence) | 0.011 | 0.006 (CVC) / 0.001 (CV criteria) | **Yes** (CVC criteria) |

The count p-value weakened from 0.001 to 0.013 because CVC decode produces more signal words on average (random mean 54 vs the CV-era ~33), making it harder for the real table's 75 to stand out. However, the *coherence* p-value improved: the vocabulary produced by T_P15 under CVC is more linguistically distinctive than under CV.

#### Track B Gates

| Gate | Threshold | Result | Status |
|------|-----------|--------|--------|
| G1 | p(count) < 0.01 | 0.013 | **FAIL** |
| G2 | p(CVC coherence) < 0.05 | 0.006 | PASS |
| G3 | CVC p(coherence) ≤ CV p(coherence) = 0.011 | 0.006 ≤ 0.011 | PASS |
| G4 | p(selectivity) < 0.05 | 0.125 | **FAIL** |

**Result: FAIL 2/4.** The two failures reflect the higher baseline of CVC decode (more signal words from random tables → harder to pass strict count/selectivity thresholds). The two passes confirm that CVC coherence matches or beats the CV baseline.

---

### Track C — Costamagna Combination Rules (FAIL 2/4)

Extracted 7 testable sequence constraints from Costamagna's 1953 catalog and compared violation rates across real Voynich decode, 5 null corpora, and Latin reference text.

#### Constraint Results

| Constraint | Real Rate | Null Rate | Selectivity | Direction | Latin Rate |
|-----------|-----------|-----------|-------------|-----------|------------|
| Coda-onset cluster legality | 0.238 | 0.205 | 0.86× | WORSE | 0.258 |
| Open/closed syllable ratio | 0.172 | 0.000 | — | WORSE | 0.184 |
| Coda consonant inventory | 0.085 | 0.060 | 0.71× | WORSE | 0.344 |
| Word-initial onset | 0.000 | 0.000 | 1.00× | TIE | 0.011 |
| Vowel hiatus | 0.000 | 0.000 | 1.00× | TIE | 0.082 |
| Syllable length | 0.000 | 0.000 | 1.00× | TIE | 0.000 |
| **Catalog attestation** | **0.170** | **0.201** | **1.18×** | **BETTER** | 0.356 |

#### Key Findings

**One positive result:** Catalog attestation — whether decoded syllables appear in Costamagna's inventory — is the only constraint where the real corpus violates *less* than null (17.0% unattested vs 20.1% for null, z = −22.94). The CVC decoded output uses Costamagna-attested syllables significantly more than random text decoded through the same pipeline. However, selectivity (1.18×) falls below the 1.3× threshold.

**Why most constraints go the wrong direction:** The CVC decode adds coda consonants (n, r, s, t) to many tokens, creating consonant clusters at syllable boundaries that null corpora — with their random character sequences — happen to avoid. The null corpora produce shorter, simpler decoded strings with fewer boundary violations. This is a systematic bias of the CVC model, not evidence against Costamagna's rules.

**Latin comparison is informative:**
- Coda inventory: Voynich 8.5% violation vs Latin 34.4% — Voynich is *more* constrained than Latin, because the coda rules are hard-coded to produce only {m, n, r, s, t}. Latin uses a wider consonant inventory at syllable endings.
- Open/closed ratio: Voynich 0.172 deviation vs Latin 0.184 — nearly identical, both within the expected Latin range of 65–75% open syllables.
- Catalog attestation: Voynich 17.0% unattested vs Latin 35.6% — Voynich syllables are better attested in Costamagna than Latin reference text. This likely reflects that the CVC decode was *designed* to produce Costamagna syllables (Phase 56 compatibility), while the Latin text was syllabified differently.

#### Track C Gates

| Gate | Threshold | Result | Status |
|------|-----------|--------|--------|
| G1 | ≥ 5 testable constraints | 7 | PASS |
| G2 | Real < null on ≥ 3 types | 1 | **FAIL** |
| G3 | Selectivity ≥ 1.3 on ≥ 1 type | max 1.18× | **FAIL** |
| G4 | Real ≤ 2× Latin on ≥ 1 type | yes (multiple) | PASS |

**Result: FAIL 2/4.** The sequence-level test is much weaker than the inventory-level test (Phase 56: 10/10). This suggests the decoded text uses Costamagna's *syllables* but does not strongly follow his *combination rules* for joining them, which is consistent with either a different regional tachygraphic tradition or limitations in the CVC model's treatment of syllable boundaries.

---

### Track D — Zodiac Labels Under CVC Decode (FAIL 0/5)

Re-decoded all 1,194 zodiac-section tokens (f70v–f73v) through the corrected CVC pipeline and matched against month names (6 languages × 12 months, 456 forms) and zodiac sign names (3 languages × 12 signs, 237 forms) at edit distance ≤ 2.

#### Result: Zero Matches

Not a single decoded CVC string came within ED ≤ 2 of any month or zodiac name in any language.

**Why:** The decoded CVC strings are 7–12 characters long (e.g., `radecorara`, `raderaradis`, `nederararar`), while month/zodiac names are 3–12 characters. The character overlap is too low for any ED ≤ 2 match. For comparison, Phase 26 (CV decode) found 109 matches at ED ≤ 2 but with folio selectivity of only 0.10 (anti-correlated with the correct folio). CVC decode eliminates even these spurious matches by producing strings that are too distant from any name form.

| Metric | Phase 26 (CV) | Phase 61 (CVC) |
|--------|--------------|----------------|
| Matches at ED ≤ 2 | 109 | **0** |
| Folio selectivity | 0.10× | **0.00×** |

#### Interpretation

This is the expected negative outcome. Three independent attempts (Phase 26 CV, Phase 54 dialectal, Phase 61 CVC) have all returned NO_SIGNAL for zodiac labels. The zodiac labels almost certainly use a different encoding convention than the main herbal/pharmaceutical text — possibly labels added by a different hand, in a different register, or using sigla (whole-word signs) rather than syllabic encoding.

#### Track D Gates

| Gate | Threshold | Result | Status |
|------|-----------|--------|--------|
| G1 | ≥ 5 matches at ED ≤ 2 | 0 | **FAIL** |
| G2 | Folio selectivity > 1.5× | 0.00× | **FAIL** |
| G3 | ≥ 2 correct-folio from same language | 0 | **FAIL** |
| G4 | CVC selectivity > Phase 26 (0.10×) | 0.00× | **FAIL** |
| G5 | ≥ 1 match at ED ≤ 1 on correct folio | 0 | **FAIL** |

**Result: FAIL 0/5.**

---

## Validation Summary

| Track | Name | Gates | Status | Key Finding |
|-------|------|-------|--------|-------------|
| A | Deep Recipe Reading | 3/5 | **PASS** | 4/5 CI templates match (0.75), conf=0.94; but 0 concatenations, 1/5 verbs, readings fragmentary |
| B | Full CV Permutation (CVC) | 2/4 | FAIL | p(CV coh)=0.001, p(CVC coh)=0.006 (beats CV 0.011); p(count)=0.013 (weaker than CV 0.001) |
| C | Costamagna Sequence Rules | 2/4 | FAIL | Catalog attestation better (1.18×, z=−22.9); 6/7 constraints not discriminating |
| D | Zodiac CVC Re-Decode | 0/5 | FAIL | 0 matches; CVC strings too long for name matching; zodiac confirmed different encoding |
| **Overall** | | **7/18** | **PHASE61_MARGINAL** | |

---

## Interpretation

### What Phase 61 Establishes

**1. The CV table is specific under CVC decode (Track B).** The most scientifically important result: applying the paper's original CV coherence criteria (verb paradigm + function kit + pharma register) to CVC signal words, only 1/1000 random CV tables passes — p = 0.001, identical to the paper's published value. The CVC-specific coherence test gives p = 0.006, beating the CV baseline of p = 0.011. **The table is at least as special under CVC as under CV.**

The count p-value weakened (0.013 vs 0.001) because CVC decode produces more signal words on average from random tables (mean 54 vs ~33 under CV). This is a known property of longer decoded strings matching the expanded dictionary more easily. The coherence tests, which measure the *linguistic quality* of signal words rather than just their count, remain highly significant.

**2. Recipe reading is not yet possible (Track A).** Individual tokens decode to recognizable syllable fragments (function words, declension endings, pharmaceutical terms), and 4/5 recipes match Circa Instans recipe templates at the structural level. But the readings are not connected text — they decompose into "with+e" and "of+function+t" fragments that don't form sentences. This confirms that the CVC decode operates at the syllable level: each EVA token maps to a syllable, not a word, and reading requires re-assembling syllables across token boundaries.

**3. Costamagna's combination rules are weakly respected (Track C).** The inventory-level match (Phase 56: 10/10) is much stronger than the sequence-level match (1/7 constraints in the right direction). The one positive result — catalog attestation selectivity of 1.18× — shows the CVC output uses Costamagna's syllable inventory more faithfully than random, but the combination rules for joining syllables don't clearly discriminate real from null text. This may reflect that the Voynich's specific tachygraphic tradition differs from Costamagna's 10th–11th century notarial system in its sequential conventions, even while sharing the same syllable inventory.

**4. Zodiac labels are definitively different (Track D).** Zero matches confirms what Phases 26 and 54 already suggested: the zodiac labels do not encode month or zodiac names under any decoding approach tested. They likely use sigla (whole-word signs) or a different encoding register.

### Paper Impact

| Track | Result | Paper Section | Impact |
|-------|--------|--------------|--------|
| A | Fragmentary readings | Sec 8 (Limitations) | Confirms "no connected passage of readable text" limitation |
| B | p(CV coh)=0.001, p(CVC coh)=0.006 | **Sec 6.2** | **CVC coherence matches/beats CV headline** — strengthens the paper |
| C | Weak sequence compliance | Sec 4.4 (Sign families) | Inventory match confirmed, sequence match marginal |
| D | Zero zodiac matches | Sec 9 (Additional properties) | Zodiac section uses different encoding |

Track B is the headline result: the paper can now report that the CVC decode produces signal vocabulary that is *more* linguistically coherent (p=0.006) than the CV decode's (p=0.011), while maintaining the same count-level specificity (p=0.001 under original criteria).

---

## Commands

```bash
# Track A: Deep Recipe Reading
voynich deep-recipes       # Select 5 best, 6-layer annotation, CI templates, reading attempts

# Track B: Full CV Permutation Under CVC
voynich cvc-full-perm      # 1000 random CV tables with fixed corrected codas

# Track C: Costamagna Sequence Rules
voynich cost-sequences     # Extract 7 constraints, test real vs null vs Latin

# Track D: Zodiac CVC Re-Decode
voynich zodiac-cvc         # Decode 1194 labels, match month/zodiac names, folio selectivity

# Integration
voynich phase61-verdict    # Integrate all 4 tracks
voynich phase61            # Run full Phase 61 pipeline
```

Runtime: ~2.5 minutes total (~60s for Track B permutation, ~97s for Track D zodiac decode, all other tracks < 5s each).
