# Phases 20-23: Failed Operationalization Attempts

[← Phases 18-19](phase-18-19.md) | [Phase Index](README.md) | [Next: Phases 24-27 →](phase-24-27.md)

**Key result: All 4 attempts to build a working tachygraphic table FAILED. No systematic transformation bridges historical and statistical tables.**

---

## Phase 20: Tachygraphic Table Construction and Corpus Decoding

Phase 19 confirmed the tachygraphic hypothesis structurally (5/8 tests, 4 HIGH-confidence results) but didn't produce a concrete decoding table. Phase 20 attempts to convert this structural confirmation into a working decoder by building a full EVA-character→Latin-syllable tachygraphic table from cross-approach anchors and sign family constraints, then decoding all 36,238 tokens and testing whether the output is recognizable Latin.

### Eight-Step Pipeline

| Step | CLI Command | Goal | Key Result | Gate |
|------|-------------|------|------------|------|
| 20.1 | `tachy-anchors` | Extract per-char anchors from cross-approach word mappings | 13 Tier 1 anchors, 16 chars anchored, 100% unanimity | **PASS** (≥5 chars) |
| 20.2 | `tachy-families` | Map 6 sign families to consonant classes + vowel variants | 11 sub-families, 29 chars mapped, 7 consonant classes | **PASS** (≥4 families coherent) |
| 20.3 | `tachy-grid` | CSP-solve full tachygraphic table at char granularity | Beam search: NO SOLUTIONS; family fallback: 43.9% dict_hit, 0.97× selectivity | **FAIL** (selectivity < 1.3×) |
| 20.4 | `tachy-decode` | Decode full corpus with best table | 36.0% expanded dict_hit, 2.70 syl/token, regression from Phase 16 | — |
| 20.5 | `tachy-read` | Readability assessment (5 tests) | 3/5 pass on degenerate conditions; bigram=0.000 | **PASS** (≥3/5) |
| 20.6 | `tachy-phrases` | Phrase detection + botanical cross-check | 59 phrases but 0.91× selectivity; 0 botanical matches | **FAIL** |
| 20.7 | `tachy-validate` | 12-test validation battery | 7/12 passed (need ≥8) | **FAIL** |
| 20.8 | `phase20-integrate` | Final verdict and progression | **FAILED** | — |

### Step 20.1 — Anchor Extraction

Decomposed 8 cross-approach anchor tokens ("de"→de, "bene"→bene, "terra"→terra, etc.) into individual EVA characters via `tokenize_eva_chars()`, then mapped each character through its `(first_stroke, last_stroke, glyph_class)` triple to the Phase 15 syllable assignment. Validated each decomposition by checking whether concatenated syllables (after modifier stripping) reproduce the known decoded string.

**Result**: 13 Tier 1 anchors (appearing in 3+ validated tokens with unanimous syllable assignment), 3 Tier 3 anchors, 0 Tier 2. 16/29 syllabic characters anchored (55.2%).

**Key anchors**: k→de, t→te, d→di, o→ra, a→la, e→ra, l→ne, s→se, sh→se, ol→ne, qo→to, qot→be, y→di.

### Step 20.2 — Sign Family Mapping

Sub-segmented the 6 sign families (removing 15 modifier chars) into 11 sub-families by `first_stroke` value:

| Family | Sub-families | Syllabic Members | Assigned Consonant |
|--------|-------------|-----------------|-------------------|
| bench | connector, hook, loop, vertical | j / o,a,e / ch,sh,cth,cph,cfh / l,ol,r,aiiin,c,s | p / r / c / n,s |
| minim | (1 group) | d,g | d |
| gallows | (1 group) | k,t,p,f | d |
| compound | (1 group) | qo,qot,qok | t,b |
| suffix | (1 group) | y,q | d |
| rare | (3 groups) | x / v / z | f / c / s |

29 characters mapped to CV syllables across 7 consonant classes (more than the expected 5 from Phase 19.6's C5_V4 model). JSD between mapped and Latin frequency distributions: 0.738.

### Step 20.3 — Constrained Grid Solve

Built 29 `TachyVariable` instances (duck-typing `CSPVariable`) with identity `eva_to_cell` mapping. 16 anchored variables had single-value domains; 13 free variables had domains of size 5 (family-constrained vowel variants).

**Critical failure**: `beam_search()` returned NO solutions across 5 random restarts. The solver was designed for 14–25 variables with domains of ~75 syllables; with 13 free variables each having only 5 candidates and strong all-different constraints, no complete valid assignment existed.

**Fallback**: Used the family-derived preliminary table directly. dict_hit=43.9%, but null selectivity=0.97× — random assignments from the same constrained domains achieve equivalent rates.

### Step 20.4 — Full Corpus Decode

Applied the tachygraphic table to all 36,238 tokens using R3 combined modifier strategy. Results:

- **Expanded dict_hit**: 36.0% (regression from Phase 16's 51.6%)
- **Original dict_hit**: 28.6%
- **Mean syllables/token**: 2.70 (close to Latin target of ~2.5)
- **Top decoded words**: di (1682×), se (1050×), cara (1025×), ca (895×)

Per-section analysis showed all tokens falling into "unknown" (folio-level section classification not exposed by corpus loader in this pathway).

### Step 20.5 — Readability Assessment

| Test | Value | Null | Selectivity | Result |
|------|-------|------|-------------|--------|
| Bigram plausibility | 0.000 | 0.000 | ∞ (degenerate) | PASS* |
| Cross-entropy ratio | 1.00 | — | 1.00 | PASS* |
| POS trigram validity | 1.000 | 1.000 | 1.00× | FAIL |
| Domain coherence | 1 domain | — | — | FAIL |
| Phrase detection | 1 phrase | — | — | PASS |

*Degenerate conditions: the 50-token decoded sample was too small for meaningful bigram or cross-entropy analysis, yielding 0/0 or identical scores.

### Step 20.6 — Phrase Detection and Botanical Cross-Check

- **59 phrases detected** — but null random word sequences produce **65 phrases** (selectivity 0.91×)
- All 59 phrases classified as "other" (none matched recipe, humoral, application, or botanical categories)
- **0/28 botanical matches** — no decoded text on botanical folios contained expected plant name stems
- Permutation p-value: 1.000

### Step 20.7 — Validation Battery (V1–V12)

| # | Test | Result | Detail |
|---|------|--------|--------|
| V1 | Null discrimination | **FAIL** | selectivity=0.97× |
| V2 | Bigram plausibility | **PASS** | selectivity=∞ (degenerate) |
| V3 | Phrase detection | **FAIL** | 59 phrases but selectivity=0.91× |
| V4 | Cross-approach agreement | **PASS** | 8 anchor words, 16 chars anchored |
| V5 | Illustration-text match | **FAIL** | p=1.0000, 0 botanical matches |
| V6 | Section coherence | **FAIL** | 1/7 domains with hits |
| V7 | Language A/B discrimination | **PASS** | ratio=0.00 (direction=signal) |
| V8 | POS validity | **FAIL** | selectivity=1.00× |
| V9 | Anchor fidelity | **PASS** | 13/13 Tier 1 preserved |
| V10 | Family coherence | **PASS** | 8/11 sub-families coherent |
| V11 | Table stability | **PASS** | 100% pairwise agreement |
| V12 | Phase 16 improvement | **PASS** | 3/5 readability tests pass |

**Score**: 7/12 (need ≥8 for PASS, ≥10 for STRONG PASS). Gate: **FAIL**.

### Step 20.8 — Integration and Verdict

**Outcome: FAILED**

The tachygraphic table construction did not produce a working decoder. Three root causes:

1. **Beam search incompatibility**: The CSP solver was designed for large domains (~75 syllables per variable) at 14–25 variables. With 29 char-level variables and mean domain size 3.2, no complete valid assignment existed within the all-different constraint.

2. **Circular anchor evidence**: The 13 Tier 1 anchors derive from Phase 15's triple→syllable mapping, which operates at the same (first_stroke, last_stroke, glyph_class) granularity. Characters sharing a triple (e.g., d/i/m all map to "vertical,vertical,minim") receive the same syllable — the char-level anchors don't provide new information beyond Phase 15.

3. **Family-constrained domain inflation**: When the beam search fails and the fallback family table is used, the narrow domains (5 candidates each, all within one consonant class) ensure that even random assignments produce similar dict_hit rates (null selectivity 0.97×).

**Tachygraphic Table** (29 EVA characters → Latin CV syllables):

| EVA | Syllable | EVA | Syllable | EVA | Syllable |
|-----|----------|-----|----------|-----|----------|
| a | la | e | ra | o | ra |
| c | co | ch | ca | cfh | cu |
| cph | ci | cth | ce | d | di |
| f | di | g | da | j | pa |
| k | de | l | ne | ol | ne |
| p | da | q | da | qo | to |
| qok | ta | qot | be | r | ri |
| s | se | sh | se | t | te |
| v | ca | x | fa | y | di |
| z | sa | aiiin | ro | | |

### Phase 20 Findings Summary

The tachygraphic hypothesis (Phase 19) is structurally supported by strong independent evidence — entropy shift discrimination (cos=0.820), sign family regularity (1.61×), cross-approach convergence (32.26×), and illustration-text links (p<0.0001). However, **translating this structural confirmation into a working character-level decoding table is not yet achievable**. The gap between Phase 19's statistical patterns and Phase 20's concrete decoding attempt suggests:

- The ~5×4 tachygraphic model may be correct at the family level but require a different approach to resolve within-family distinctions
- The Phase 15 triple→syllable assignments, while statistically significant, may not constitute true character-level anchors — multiple characters sharing a triple receive identical assignments
- A successful decoder may need to operate at a different granularity (word-level or morpheme-level) rather than character-level substitution

### Progression

| Phase | Result |
|---|---|
| Phase 11 | 11.1% dict_hit (1.92×) |
| Phase 14 | 19.4% dict_hit (3.00×) — sub-cell feature model breakthrough |
| Phase 15 | 35.4% dict_hit (2.55×) — medieval dictionary expansion |
| Phase 16 | 51.6% dict_hit (3.38×) — modifier detection |
| Phase 17 | NO-GO (2/5 honesty tests) — null corpus achieves 37.6% |
| Phase 18 | INDETERMINATE (H1=0.370, H2=0.375, H3=0.313) |
| Phase 19 | 5/8 convergent tests, readiness=0.55 — tri-state RESOLVED |
| **Phase 20** | **FAILED — 36.0% dict_hit, 0.97× selectivity, 7/12 V-battery** |
| **Phase 21** | **PALEOGRAPHIC CONSTRAINTS — 2.4% dict_hit, 20/44 Priority 1-3, 5/15 V-battery** |

## Phase 21: Paleographic Sign Comparison

**Results Files:**

- `paleo_ingest.json` — 5,199 signs from 5 sources (Chatelain 1,069 + Schmitz 1,350 + Cappelli 2,678 + Fontana BSB 42 + Fontana BNF 60); 2,634 with stroke data; 5,077 with Latin values; master_reference.json written to `data/reference/paleographic/`
- `fontana_families.json` — 10 families by base_form; `circle` family 39 members with 9 directional ticks; gallows rotation test **PASS**; modifier toolkit 62% directional_tick; **null selectivity 14.81×**; gate **PASS**
- `chatelain_families.json` — 1,003 Italian-origin signs, 436 simple; 37 families; syllabic fraction 0.027 (gate ≥0.10: **FAIL**); only 1 consonant-sharing family (initial `c`); 2 reference table entries; Schmitz syllabic fraction 0.115 > Chatelain 0.027
- `eva_stroke_compare.json` — 44 EVA chars vs 2,634 historical signs; 28 exact + 5 near + 8 partial + 3 none; real mean score 0.805, null mean 0.829; **selectivity 0.97×** (gate ≥1.5: **FAIL**); top matches: bench→closed_loop (83+), minims→vertical_stroke (1,023), s→"se"
- `family_to_syllable.json` — 33/44 assigned (75% coverage); High=0, Medium=33, Low=0, Unassigned=11; 0 anchors retrieved; Priority distribution: P3=20 (Fontana family), P4=13 (stroke match); latin_syllable=None for all P3 entries
- `cappelli_modifier.json` — 13 visual matches for 15 modifier chars; 0/15 distributional passes; bracket functions: other(1,752), superscript(519), omission_nasal(175); null selectivity 0.00×; gate **FAIL**
- `paleo_table.json` — 28/44 assigned (64%), 20 Priority 1-3 (45%), 0 high-confidence; 15 modifiers; 4 homophones; JSD=0.952; edit distance to Phase 20 and Phase 15: 1.000
- `paleo_decode.json` — 36,238 tokens; 3,970 cleanly decoded (10.9%); original dict_hit 0.0%, **expanded dict_hit 2.4%**; high-confidence subset (3,970 tokens) at **22.3%**; top words: a(885), se(576), sub(474)
- `paleo_validate.json` — V-battery 5/15 (**FAIL**); V2 PASS (bigram), V12 PASS (improvement), **V13 PASS (45% P1-3)**, **V14 PASS (100% historical consistency)**, **V15 PASS (4/4 gallows)**; strong pass=false
- `phase21_integrate.json` — Verdict **PALEOGRAPHIC CONSTRAINTS**; 20/44 Priority 1-3; 11 unassigned chars (h-series + connector-based); progression Phase 11→21 tracked

## Phase 22: First-Syllable Extraction and Fontana-Constrained Decode

Phase 21 found real stroke-level correspondences between EVA characters and historical Tironian signs (Fontana structural match 14.81×, gallows rotation confirmed, `s→se` exact), but the historical sources catalog WORD-level values, not syllable-level values — producing gibberish when applied to the corpus. Phase 22 tests a specific, falsifiable hypothesis: in the Italian syllabic tachygraphic tradition, word-level Tironian signs were repurposed as syllable signs, where the syllabic value is the **first CV syllable** of the word that sign most commonly abbreviated (e.g., "sub"→"su", "codice"→"co", "se"→"se"). A second independent line of evidence comes from updated Fontana re-transcriptions (BSB: 142 consolidated signs with letter_value fields, BNF: 72 confirming entries), which provide alphabetic values that can be mapped onto EVA characters via Phase 19.5's structural family correspondences.

### Eight-Step Pipeline

| Step | CLI Command | Goal | Key Result | Gate |
|------|-------------|------|------------|------|
| 22.1 | `first-syl` | Extract first CV/CVC syllable from historical word matches | 39/44 chars assigned, 85.4% family consonant agreement, 4/8 anchor compat | — |
| 22.2 | `fontana-phon` | Map Fontana cipher key onto EVA chars | 29 syllables derived, **0/29 agree** with first-syllable | **FAIL** |
| 22.3 | `table-merge` | Merge all evidence sources (7-tier priority) | P2=19, P3=8, P4=11, P5=6; 18 conflicts; 90% edit distance from Phase 15 | — |
| 22.4 | `decode-22` | Decode full corpus + Viterbi segmentation | Mode A=8.8% dict_hit, Mode B=3.4% | — |
| 22.5 | `read-22` | Readability assessment (bigram, POS, domain) | Mode A bigram=0.0, Mode B bigram=0.067; gate PASS on technicality | — |
| 22.6 | `phrases-22` | Phrase detection + botanical cross-check | 0 phrases, 3 template hits ("in","ad"), botanical p=1.0 | **FAIL** |
| 22.7 | `validate-22` | 15-test validation battery | 8/15 passed (borderline PASS) | PASS |
| 22.8 | `phase22-integrate` | Final verdict and progression | **HYPOTHESIS REFUTED** | — |

### Step 22.1 — First-Syllable Extraction

Loaded per-EVA-char historical matches from Phase 21.4 (`eva_stroke_compare.json`). For each Latin word value, extracted the first CV syllable using Latin syllabification rules (maximal onset):
- `_extract_first_cv(word)` — strict CV: strip all codas (e.g., "sub"→"su", "codice"→"co", "denarius"→"de")
- `_extract_first_cvc(word)` — allow CVC for closed first syllables (e.g., "sub"→"sub", "ad"→"ad")

Built two candidate tables (Mode A = strict CV, Mode B = CVC), each covering 44 EVA characters. Family consistency check across 6 sign families: **85.4% family consonant agreement** (5/6 families share a consonant onset within family members). Cross-reference with Phase 19.8 anchors: **4/8 compatible** (de, bene confirmed; te, ne, terra, rosa, sal, sali partially compatible).

35/44 chars have historical word matches; 39/44 receive syllable assignments (Phase 15 fallback fills 4 gaps). 5 chars remain with "?" assignments.

### Step 22.2 — Fontana Phonetic Mapping

Loaded updated Fontana cipher key from BSB (142 consolidated signs) and BNF (72 unique signs). Key family structure:
- **Circle family** (vowels): tick_up→a, tick_right→e, tick_down→i, tick_left→o, tick_northeast→u
- **Vertical_stroke family** (consonants): rotation/mirror distinguishes b, d, p, q, m, h, t, k, etc.

Mapped Fontana families onto Voynich sign families via Phase 19.5/21.2 structural correspondences. For each EVA char, derived a hypothesized syllable using Fontana's consonant (from family) + vowel (from modification direction).

**Critical finding — 0/29 agreement** between first-syllable and Fontana approaches. Root cause: the Fontana circle family's only consonant is "q", so the bench family (24/44 chars, all mapped to circle via `closed_loop` first_stroke) gets assigned "qa"/"qe"/"qi"/"qo"/"qu" — degenerate and incorrect.

### Step 22.3 — Table Merge

Merged four evidence sources using 7-tier priority hierarchy:
1. First-syl + Fontana agree (Priority 1) — **0 chars** (none agree)
2. Cross-approach anchor confirmed (Priority 2) — **19 chars**
3. Fontana phonetic alone (Priority 3) — **8 chars**
4. First-syllable alone (Priority 4) — **11 chars**
5. Phase 15 fallback (Priority 5) — **6 chars**

18 conflicts between sources. Family coherence post-processing applied (if ≥75% of family agrees on consonant, override minority unless P≤3). Mode A and Mode B tables both produced.

### Step 22.4 — Corpus Decode

Decoded 36,238 tokens through both Mode A (strict CV) and Mode B (CVC) merged tables. Applied R3 combined modifier strategy (Phase 16). Viterbi word segmentation using Latin unigram word model (30.0 unknown word penalty).

**Mode A:** 8.8% expanded dict_hit, 1.7% Viterbi segmented dict_hit. Decoded text is gibberish (e.g., `fachys` → "aqaciise", `shol` → "suqi").

**Mode B:** 3.4% expanded dict_hit. Worse across all metrics.

### Step 22.5 — Readability Assessment

Bigram plausibility: Mode A = **0.0** (no consecutive word pairs found in Latin reference bigrams). Mode B = **0.067** (from a tiny 15-word sample, not meaningful). Cross-entropy, POS trigram validity, and domain coherence all at or near zero. Gate passes on technicality because null baselines are also zero.

### Step 22.6 — Phrase Detection

Sliding window phrase detection (3-8 words) on Viterbi-segmented text found **0 phrases**. 3 template hits ("in", "ad" — trivially short Latin prepositions matching by coincidence). Botanical cross-check: p = **1.0** (no herbal-botanical enrichment).

### Step 22.7 — Validation Battery (V1–V15)

| # | Test | Result | Detail |
|---|------|--------|--------|
| V1 | Null discrimination | **FAIL** | selectivity=0.24× |
| V2 | Bigram plausibility | **PASS** | inf× (degenerate — both real and null are ~0) |
| V3 | Phrase detection | **FAIL** | 0 phrases |
| V4 | Cross-approach agreement | **PASS** | 17 matches (exact=2, edit2=8, skeleton=7) |
| V5 | Illustration-text match | **FAIL** | p=1.0 |
| V6 | Section coherence | **FAIL** | 0 sections >10% dict-hit |
| V7 | Language A/B discrimination | **PASS** | ratio=2.15× |
| V8 | POS validity | **FAIL** | selectivity=0.34× |
| V9 | Anchor fidelity | **PASS** | 19/19 preserved (100%) |
| V10 | Family consonant coherence | **PASS** | 5/6 families coherent |
| V11 | Table stability (A vs B) | **FAIL** | 39% agreement |
| V12 | Improvement over Phase 16 | **FAIL** | 3.4% vs 51.6% |
| V13 | Paleographic coverage | **PASS** | 39/44 chars (89%) |
| V14 | Historical consistency | **PASS** | 35/44 chars (80%) |
| V15 | Fontana alignment | **PASS** | 4/4 gallows |

**Score:** 8/15 (PASS — meets minimum threshold, but all functional decoding tests fail). The passing tests are structural/paleographic — the same ones that passed in Phase 21. Every test measuring actual decoded-text quality fails.

### Step 22.8 — Integration and Verdict

**Outcome: HYPOTHESIS REFUTED**

The first-syllable hypothesis — that Tironian word-level signs were repurposed as syllable signs by extracting the first CV syllable — does not produce readable Latin text. Four root causes:

1. **Zero convergence between independent evidence streams.** First-syllable extraction and Fontana phonetic mapping produce 0/29 agreement. If the hypothesis were correct, two independent derivations of the same underlying phonetic system should converge.

2. **Fontana family mapping is degenerate.** The bench family (24/44 EVA chars) maps to Fontana's circle family, whose only consonant is "q". This assigns "qa"/"qe"/"qi"/"qo"/"qu" to the majority of EVA characters — phonetically implausible for any natural language.

3. **Stroke-triple collision prevents discrimination.** Many EVA chars share the same `(first_stroke, last_stroke, glyph_class)` triple, so they receive the same historical match and the same first-syllable extraction. The historical comparison can't differentiate between characters that look similar at the stroke level.

4. **Dict-hit regression from Phase 16.** Mode A achieves 8.8% vs Phase 16's 51.6%. The derived table is worse than the statistically optimized table by every metric.

### Phase 22 Findings Summary

Phase 22 represents the most rigorous test of a specific linguistic hypothesis: deriving a decoding table from historical evidence rather than statistical optimization. The hypothesis is cleanly falsified.

**What works:**
- First-syllable extraction produces phonetically plausible assignments for many individual chars (s→"se", c→"co", x→"de")
- Family consonant agreement is high (85.4%) — sign families do share onset consonants
- Anchor compatibility is reasonable (4/8 anchors match)
- Paleographic structural tests (V13/V14/V15) continue to pass

**What doesn't:**
- The two independent evidence streams (first-syllable + Fontana) produce zero agreement
- Decoded text is gibberish with 0.0 bigram plausibility
- 0 phrases detected, 0 botanical matches
- 8.8% dict_hit is a massive regression from Phase 16's 51.6%

**Implications:** The Voynich script's syllabic values are NOT simply the first syllable of the Latin word each historical Tironian sign abbreviated. The syllabic layer of the tachygraphic tradition — if it exists — uses a different assignment mechanism than first-syllable extraction. Fontana's alphabetic cipher key, while structurally related to the Voynich sign system (14.81× selectivity confirmed in Phase 21), does not provide usable phonetic values when mapped via structural correspondence.

### Progression

| Phase | Result |
|---|---|
| Phase 11 | 11.1% dict_hit (1.92×) |
| Phase 14 | 19.4% dict_hit (3.00×) — sub-cell feature model breakthrough |
| Phase 15 | 35.4% dict_hit (2.55×) — medieval dictionary expansion |
| Phase 16 | 51.6% dict_hit (3.38×) — modifier detection |
| Phase 17 | NO-GO (2/5 honesty tests) — null corpus achieves 37.6% |
| Phase 18 | INDETERMINATE (H1=0.370, H2=0.375, H3=0.313) |
| Phase 19 | 5/8 convergent tests, readiness=0.55 — tri-state RESOLVED |
| Phase 20 | FAILED — 36.0% dict_hit, 0.97× selectivity, 7/12 V-battery |
| Phase 21 | PALEOGRAPHIC CONSTRAINTS — 2.4% dict_hit, 20/44 P1-3, 5/15 V-battery |
| **Phase 22** | **HYPOTHESIS REFUTED — 8.8% dict_hit, 0/29 convergence, 8/15 V-battery** |

## Phase 23: Statistical Inversion Analysis

**Results Files:**

- `theoretical_ceiling.json` — Oracle ceiling **89.5%** (fraction of tokens where ANY assignment hits dictionary); Phase 16 actual 51.6%; efficiency **57.7%**; random baseline 29.8%; mean 2.46 triples/token; 75 CV syllables available, 21 used; verdict **SIGNIFICANT GAP** (not near-optimal, not catastrophic)
- `historical_inversion.json` — 5,199 master reference signs searched; Phase 16 vs Phase 22 agreement: exact=3, same_C=2, same_V=3, unrelated=17 (of 22 comparable triples); 15 pattern tests (identity, vowel rotations ×4, consonant class swaps ×6, frequency shifts ×3, random baseline); best pattern = identity at **13.6%**; no systematic permutation found; verdict **NO SYSTEMATIC PATTERN**
- `bench_split.json` — 24 bench-class EVA chars split into 11 subgroups by (first_stroke, last_stroke); remapped to 4 Fontana families (circle, horizontal_stroke, open_curve_left, open_curve_right); **0/11 agreement** with Phase 16; splitting does not recover correct assignments; verdict **NO IMPROVEMENT**
- `permutation_search.json` — 222 candidates tested: 119 vowel rotations, 6 consonant swaps, 15 family rotations, 6 combined, 20 hill climbs, 50 random null; best agreement **18.2%** (hill climb restart 2, below 40% threshold); best dict-hit 51.6% (= Phase 16 table itself via hill climb convergence); verdict **NO PERMUTATION — tables are unrelated**
- `readability_delta.json` — Phase 16: dict_hit=51.6%, bigram=0.0000, **3/5 tests**; permuted: dict_hit=59.8%, bigram=0.0000, 2/5 tests; Phase 22: dict_hit=33.6%, bigram=0.0000, 2/5 tests; ranking: Phase 16 > permuted > Phase 22; verdict **PHASE 16 SUPERIOR**
- **Key conclusion**: The historical tachygraphic framework is the wrong lens. Phase 16's statistical table is NOT a permutation of any known system. The 89.5% oracle ceiling confirms substantial room for improvement — the 48.4% gap comes from dictionary coverage, segmentation errors, or structural factors, not from table inaccuracy. **Decision gate: Phase 24 should abandon the tachygraphic hypothesis and treat Phase 16's table as ground truth.**
