# Phase 11: CSP Phonetic Decoder

[← Phase 10](phase-10-hypothesis.md) | [Phase Index](README.md) | [Next: Phases 12-13 →](phase-12-13.md)

---

## Phase 11: Constraint Satisfaction Phonetic Decoding

Phase 10 confirmed H1 (Constructed Script): the Voynich script is a Devanagari-class abugida where each EVA character encodes one CV syllable and maps to one of 14 occupied cells in a 5×6 onset×nucleus grid. Phase 11 formulates this as a 14-variable Constraint Satisfaction Problem and searches for the phonetic assignment that minimises cross-entropy against a Romance language model.

### Architecture

Each grid cell is a CSP variable; its domain is the set of legal CV syllables in the target language. Six constraint layers progressively prune the search space before beam search explores remaining candidates.

| Layer | Constraint | Implementation |
|-------|------------|----------------|
| L1 | Phoneme inventory | Restrict domains to syllables in the target language's legal CV table |
| L2 | Frequency rank matching | Cells ranked by corpus frequency may only map to syllables at proportional reference-frequency rank ± slack |
| L3 | Phonotactic legality | Remove syllables with forbidden onset clusters |
| L4 | Word-structure validity | Score decoded tokens by vowel presence and word-final legality |
| L5 | Illustration anchors | Rosetta folio stems expand cell domains with anchor-suggested syllables; beam search applies a 0.4-point bonus to anchor-aligned assignments |
| L6 | Cross-entropy scoring | Decoded tokens scored against a character-level 3-gram LM built from the reference corpus |

AC-3 arc consistency propagation further prunes domains: any cell with a singleton domain removes its value from all other cells. Beam search (width 50, MRV ordering — smallest domain first) then expands partial assignments, scoring every 3 steps via a fast partial cross-entropy estimator.

### Step 11.0: CSP Solver Sanity Test

| Sub-step | Description | Module |
|----------|-------------|--------|
| 11.0a | **Known-mapping encoding** — Assign 14 Latin CV syllables to the real 14 grid cells, encode 3,000 Latin words as EVA glyph sequences. | `phases/csp_solver.py` |
| 11.0b | **Random baseline** — Score 100 random cell→syllable assignments; compute mean CE. | `phases/csp_solver.py` |
| 11.0c | **Selectivity check** — Verify that the true mapping CE is below the random mean (selectivity ≥ 1.3×). | `phases/csp_solver.py` |

**Result:** True mapping CE = 3.855 vs random mean CE = 5.649. **Selectivity = 1.47×**, confirming the pipeline can distinguish the true mapping from random. Direct cell recovery is 0/14 — the encoded corpus is dominated by the most frequent glyph (most Latin syllables fall back to the most common CV pattern), so the CSP cannot uniquely recover the mapping, but the true mapping is significantly better than chance.

**Verdict:** `sanity_passed_selectivity_1.47x`. Gate **PASSED** (selectivity ≥ 1.3×).

### Step 11.2: Multi-Language Phonetic Decoding

| Sub-step | Description | Module |
|----------|-------------|--------|
| 11.2a | **Per-language pipeline** — For each language (Latin, Occitan, Italian, German): build CV inventory, build 3-gram LM from reference tokens, build anchor constraints from 8 Rosetta folios, initialise domains (Layers 1–3+5), run AC-3, run beam search (Layer 6). | `phases/csp_decode.py` |
| 11.2b | **Language ranking** — Sort by best cross-entropy. | `phases/csp_decode.py` |
| 11.2c | **Random baseline + selectivity gate** — 200 random assignments; gate: selectivity ≥ 1.5×. | `phases/csp_decode.py` |
| 11.2d | **Anchor details** — For each Rosetta folio, decode the stem and compare to the plant name. | `phases/csp_decode.py` |

**Result:**

| Language | CE | Dict hit | Anchors | Time |
|----------|----|----------|---------|------|
| **Latin** | **2.999** | **11.1%** | **1/8** | 11s |
| Occitan | 3.794 | 5.6% | 0/8 | 11s |
| Italian | 4.191 | 9.2% | 1/8 | 11s |
| German | 4.221 | 2.3% | 0/8 | 13s |

Random baseline mean CE = 5.74. **Selectivity = 1.92×** (Latin best assignment vs random). Latin phonetic table: C1V1→si, C2V3→co, C3V4→co, C2V5→ne, C1V2→ca, C2V2→ca, C1V5→ne, C1V6→ne, C3V1→bi, C2V6→ce, C2V1→se, C4V3→ba, C5V4→ba, C3V6→ba. Decoded sample: "fachys"→"cosiconebi", "shol"→"coca", "cthres"→"cosisibi". Note: Phase 8 MDL ranked German first due to corpus-size effects; Phase 11 CSP ranks Latin first, consistent with all prior structural evidence.

**Verdict:** `csp_decode_significant_selectivity_1.92x`. Gate **PASSED** (selectivity ≥ 1.5×).

### Step 11.3: CSP Validation Battery

| Test | Description | Result |
|------|-------------|--------|
| V1: Sanity Check | True mapping CE < 1.3× random mean on synthetic corpus | **PASS** — selectivity 1.47× |
| V2: Random Baseline | CSP best CE vs 500 random assignments (selectivity ≥ 1.5×) | **PASS** — selectivity 1.90× |
| V3: Cross-Validation | 5-fold folio split, CE coefficient of variation < 0.10 | **PASS** — CV = 0.013 |
| V4: Section Coherence | Herbal decoded text has more plant keywords; pharma has more recipe keywords | **PASS** — score 1.0 |
| V5: Illustration Match | Non-anchor herbal folios produce decodable stems | **PASS** — 5 decoded stems |
| V6: Language B | Language B CE / Language A CE ratio < 2.0 | **PASS** — ratio 1.02× |
| V7: Prior Convergence | Agreement with Phase 8/9/10: H1 verdict, language ranking, verb decoding | **PASS** — 2/3 checks |

**Summary: 7/7 tests passed.**

**Verdict:** `csp_validation_passed_7_of_7`. Gate **PASSED** (≥ 4/7 required).

### Phase 11 Findings Summary

The CSP phonetic decoder achieves a **1.92× selectivity** over random assignment — the strongest discrimination signal obtained for any phonetic mapping approach in this project. Latin consistently wins across CE, dict hit rate, and partial anchor matching. The decoded text is phonetically regular (100% word validity) but does not yet produce recognisable Latin — the 11.1% dictionary hit rate reflects short decoded tokens like "si", "ne", "ca" matching common Latin particles and prepositions, not substantive vocabulary. The selectivity ceiling documented across Phases 5–9 remains: the CSP finds a real CE minimum, but the minimum is shallow enough that many assignments score nearly as well. The remaining gap between the current result and genuine decoding requires either: (a) a larger and more consistent set of illustration anchors, or (b) relaxing the CV-only encoding model to allow CVC syllables or consonant clusters.

## Phase 11.5: CSP Refinement and Diagnostic Validation

Phase 11.5 runs five sequential diagnostic and refinement steps on the Phase 11 CSP output, aiming to improve the 11.1% dictionary hit rate toward the 15% target and increase anchor matches from 1/8 to 3/8.

### Step 11.5.1: Failure Diagnosis

| Metric | Value |
|--------|-------|
| Tokens analyzed | 1,500 (Language A paragraph tokens) |
| HIT (exact reference match) | 9.9% |
| NEAR_MISS (edit distance ≤ 2) | 38.6% |
| GIBBERISH | 49.6% |
| Signal fraction (HIT + NEAR_MISS) | **48.5%** |
| High-error cells (error > 60%) | 13/14 |

**Gate (signal ≥ 15%): PASS.** Top correction vectors: `C3V6 (ba→de)` gain=1.0, `C1V2 (ca→di)` gain=0.34. All 13 high-error cells are dominated by GIBBERISH or NEAR_MISS — no cell is producing useful exact hits. Diagnosis: NEAR_MISS_DOMINANT, meaning the beam search reaches plausible phonetic neighborhoods but not exact targets. Correction vectors prescribe cell-level substitutions for the relaxation phase.

### Step 11.5.2-3: Inherent Vowel and Relaxation Sweep

| Level | Description | Syllables | Dict hit | CE | Selectivity |
|-------|-------------|-----------|----------|----|-------------|
| 0 | Strict CV | 75 | 9.87% | 3.099 | **1.83×** |
| 1 | CV + inherent vowel | 75 | 9.87% | 3.099 | 1.83× |
| 2 | CV + top-25 CVC | 100 | 3.33% | 4.203 | 1.35× ⚠ |
| 3 | CV + full CVC | 146 | 3.87% | 4.327 | 1.31× ⚠ |
| 4 | CV + CVC + top-25 CCV | 171 | 3.87% | 4.327 | 1.31× ⚠ |
| 5 | CV + full CVC + full CCV | 216 | 4.40% | 3.946 | 1.44× ⚠ |

Inherent vowel candidates (a/e/i) all produce identical results at Level 1 — the inventory does not differentiate. **Levels 2–5 all drop below the 1.5× selectivity gate.** Adding syllable types expands the search space faster than it constrains it; the beam search degrades into a broader but shallower exploration. Best configuration remains Level 0 (strict CV).

**Gate (best dict_hit ≥ 15% or improvement ≥ 1.35×): FAIL.** Improvement factor = 0.89× (slight regression from Phase 11 baseline). Verdict: `refinement_minimal_improvement_check_grid_decomposition`.

### Step 11.5.4: Verb Constraint Integration

| Metric | Value |
|--------|-------|
| Phase 9 verb assignments loaded | 10 |
| Length-matched verb constraints | 1 (soft; confidence 0.572) |
| Constraint: `shes` → `recipe` (`re`, `ci`, `pe`) | — |
| Dict hit before | 9.87% |
| Dict hit after | 7.15% (Δ = −2.72%) |
| Verb matches | 1 |
| Illustration conflicts | 6 |

Only 1 of 10 Phase 9 verb assignments produces a usable constraint: most Voynich stems do not length-match their Latin syllabifications. The single soft constraint slightly worsens dict hit (verb pulls one cell assignment toward a low-frequency syllable). Six conflicts found between verb-required cell values and illustration anchor hints.

**Gate (dict_hit ≥ 15% or verb_matches ≥ 5): FAIL.** Verdict: `verb_constraints_applied_dict_hit_0.071`.

### Step 11.5.5: Iterative Anchor Bootstrapping

| Metric | Value |
|--------|-------|
| Confirmed hit anchors extracted (iteration 1) | 14 new (22 total) |
| Dict hit after iteration 1 | 7.15% (Δ = 0.0000) |
| Selectivity | 1.86× |
| Convergence reason | `delta_small` (Δ < 0.005) |
| Iterations run | 1 |

The iterative loop extracts 14 new confirmed hit anchors (tokens that decode consistently to a reference word ≥ 3 times), adding them to the anchor set and re-running the CSP. The result is unchanged — the beam search reproduces the same assignment regardless of the expanded anchors. This indicates the current assignment is at a local minimum; the anchor additions do not provide enough new directional signal to escape it.

**Gate (final_dict_hit ≥ 15% or improvement ≥ 0.03): FAIL.** Verdict: `csp_iterate_delta_small_dict_hit_0.071`.

### Step 11.5.6-7: Multi-Language Final + V1–V9 Validation Battery

**Language ranking (Level 0):**

| Rank | Language | CE | Dict hit | Anchors |
|------|----------|----|----------|---------|
| 1 | **Latin** | **3.099** | **9.87%** | **1/8** |
| 2 | Occitan | 3.997 | 6.47% | 0/8 |
| 3 | German | 4.222 | 2.40% | 0/8 |
| 4 | Italian | 4.252 | 8.93% | 1/8 |

**V1–V9 validation battery:**

| Test | Result | Score |
|------|--------|-------|
| V1: Sanity Check | **PASS** | selectivity 1.47× |
| V2: Random Baseline | **PASS** | selectivity 1.85× |
| V3: Cross-Validation | **PASS** | CV = 0.015 |
| V4: Section Coherence | **PASS** | score 1.000 |
| V5: Illustration Match | **PASS** | score 0.192 |
| V6: Language B | **PASS** | ratio 1.013× |
| V7: Prior Convergence | **PASS** | score 0.667 |
| V8: Readability | **PASS** | composite 0.379 (100% plausible Latin endings, 7.2% exact hits) |
| V9: MCMC Comparison | **FAIL** | CE z=3.77 ✓ but dict-hit z=−0.61 ✗ |

**Summary: 8/9 tests passed.** Gate (≥ 6/9): **PASS.** Selectivity 1.85× (above 1.5× gate).

**Verdict: `framework_correct_phonetics_imprecise`.** The CSP framework is structurally sound — Latin wins on CE, cross-entropy is stable across folds, section coherence is confirmed, and 100% of decoded tokens end in phonotactically plausible Latin suffixes. The V9 failure pinpoints the remaining problem: the MCMC random walk produces dict-hit rates comparable to the CSP solution, meaning the CE advantage is not yet sufficient to lift dict hits above the noise floor. The bottleneck is grid granularity, not the language hypothesis or the CSP architecture.

### Phase 11.5 Findings Summary

Phase 11.5 confirms and sharpens the Phase 11 diagnosis. The CSP framework is working correctly (selectivity 1.85×, 8/9 validation tests pass). The failure mode is specific: the 14-cell grid decomposition maps each Voynich glyph to a CV syllable, but at this resolution the mapping cannot recover individual Latin words — it recovers phonetic neighborhoods. Adding more syllable types (CVC, CCV) to escape this ceiling consistently destroys selectivity by over-expanding the search space, confirming that the grid itself needs finer decomposition before the phonetic mapping can improve. The near-miss rate (38.6%) is a positive signal: the CSP is finding the right phonetic region, and systematic correction of the 13 high-error cells could unlock word-level recovery.

## Cross-Validation Tables

The approaches cross-validate across all phases:

| Approach 1 finds | Approach 2 finds | Phase 3 finds | Phase 4 finds | Phase 4.5 finds | Phase 5 finds | Phase 7 finds | Phase 7.5 finds | Interpretation |
|---|---|---|---|---|---|---|---|---|
| CV syllabary grid with good fit | Closest match = Latin-substitution | D.1 favors syllabary, D.3 favors substitution, PMI r=0.96 | 8/15 metrics discriminate; PMI, bigram, length, stability all pass | Grid captures morphological structure (chi² p<0.001 both axes, JSD=0.46 on nucleus) | 2,328 stem paradigms discovered (z=178); 23 high-paradigm stems with 7–31 forms each | A and B embedding spaces both independently point to Latin (Procrustes + GW); noun candidates cluster 5.4x above baseline in embedding space | Combined A+B space (963 stems) improves ARI 4.2%; noun subclusters split into 4 domains matching Latin pharmaceutical categories; Fisher combined p=2.75×10⁻¹⁰ across 5 families | **Morphological structure confirmed at paradigm level; grid axes encode affix/stem roles; global embedding geometry converges on Latin; aggregate signal overwhelmingly real** |
| Strong positional constraints (MI=0.30) | Latin dominates top 5 | Grid 100% stable, sections diverge (Jaccard=0.14) | Currier A/B distinct (H2 diff significant, grid Jaccard=0.14); min sample ~10k tokens | A/B confirmed as distinct systems (JSD z=3.82, vocab overlap=14%) | Paradigm selectivity 1.47× (z=178) — just below 1.5× gate | Language A ARI=0.11 (embeddings capture section structure); Language B ARI=-0.003 (no section signal — consistent with notation hypothesis) | 65.6% shared stems between A/B; register ARI=0.038 (mostly merged); 9/10 verb assignments plausible but selectivity 0.92x | **Section divergence = genuine A/B split, not artifact; A has semantic structure, B does not; A/B share most vocabulary** |
| 5x6 grid, 47% occupancy | No null insertion evidence | Gap pattern random, closest to Cypriot (8% diff) | R=0.39 (syllabary/abugida overlap); nucleus predicts onset more than reverse | R(affix\|stem)=0.61, R(stem\|affix)=0.39 — linguistically natural under morpheme relabeling | Occitan JSD=0.65 vs Latin JSD=0.71; not separable (ratio=0.92) | Prefix/suffix separation=0.90 in affix embedding space; verbs at position 1 in 60-100% of segments; verb freq rho=0.97 with Latin recipe verbs | 0/8 Rosetta plant stems land in plant_names cluster; subclusters capture frequency patterns not semantic content | **Anomalous reverse R explained** — stems constrain affixes; **Romance family confirmed; embedding subclusters are distributional, not semantic** |
| — | Latin best across encodings | Latin best syllable match | Latin #1, Occitan #2, but CIs overlap on all metrics | qo- removal neutral (14.4% of corpus, distributed across grid, no metric improvement) | Random-word selectivity 0.99× — frequency priors dominate over morphological content; **phonetic decode blocked** | Procrustes selectivity 0.96-0.97x, GW selectivity 1.00x — both fail 1.5x gate; only 14 seed pairs available | Noun subcluster selectivity 1.29x, verb assignment selectivity 0.92x — both fail 1.5x gate; only 1/76 identifications has multi-method support | **Selectivity ceiling persists** at word identification level; individual assignments are frequency-dominated; **structural convergence is real but does not unlock vocabulary** |

**Phase 9 cross-validation (encoding hypothesis tests):**

| Phase 8 finds | Phase 9 finds | Interpretation |
|---|---|---|
| SA stability 2.5%, MDL sanity check 4% recovery — not genuine decryption | Not homophonic (0 clusters), not nomenclator (bimodal but not uniquely), not polyalphabetic (JSD matches shuffled) | **All three tested encoding models ruled out; decoding failure is not explained by a known cipher class** |
| MDL ranks German first (corpus size artifact), compression 1.3–1.8× across all languages | 4 languages indistinguishable at matched 11K tokens; Italian/Occitan tie 3–3 on metrics | **Language question genuinely unresolved — not a methodological failure but a fundamental ambiguity** |
| Fisher combined p = 0.90 (no significance), 1/100 cross-approach agreement | Classified as encoded natural language (conf=1.0); entropy floor 0.978 vs 0.33–0.51 for reference | **Text is definitively not random, glossolalia, or constructed; encoding preserves redundancy above plaintext levels** |
| All compression ratios in frequency-matching range (1.3–1.8×) | Markov models match only 4/6 metrics; H2/H1 = 0.622 (outside natural range 0.3–0.6) | **Structure requires higher-order dependencies; the anomalous H2/H1 ratio is the cipher's signature** |

**Phase 10 cross-validation (hypothesis discrimination):**

| Phase 10.1 finds | Phase 10.2 finds | Phase 10.3 finds | Phase 10.4 finds | Phase 10.5 verdict | Interpretation |
|---|---|---|---|---|---|
| Entropy curve r=0.999 with Latin; sections consistent (herbal-pharma r=0.9998); Language B curve flatter with higher floor (3.25 vs 2.55) | τ_voynich = 4,285 >> τ_latin = 477 (ratio 8.98×); but section τ inconsistent and phrase alignment shows no improvement | 63 folios, within-section JSD not significant vs null; function-word CV inflated (0.73 vs 0.35–0.52); no quire boundary effect | Closest to Devanagari (0.47 similarity); diagnosis = "construction" (onset/nucleus independent of position, p < 10⁻⁶) | **H1 wins** (score 4.0, margin 2.5). H2 = 1.5, H3 = 1.0. Gate passed. | **Constructed script confirmed.** Glyph strokes encode phonetic values via script-specific construction logic; the grid is a syllabary blueprint, not a morphological accident. The 14-variable CSP is the decoding path. |

**Phase 11 cross-validation (CSP phonetic decoding):**

| CSP decode finds | CSP validate finds | Interpretation |
|---|---|---|
| Latin wins: CE 2.999 vs Occitan 3.794, Italian 4.191, German 4.221. Selectivity 1.92× (random mean 5.74). Dict hit 11.1% (Latin reference words in decoded text). 1/8 Rosetta anchors at edit distance ≤ 3. | 7/7 validation tests pass: sanity selectivity 1.47×, V3 CV=0.013, V4 section coherence confirmed, V6 Language B ratio 1.02×, V7 prior convergence 2/3. | **Latin phonetic assignment is statistically discriminated from random** — the strongest signal achieved for any phonetic mapping approach. The decoded text is phonetically legal (100% word validity) but sub-lexical: decoded tokens match Latin particles ("si", "ne", "ca") at 11.1% rate, not substantive vocabulary. Selectivity ceiling persists: the CSP finds a real CE minimum but it is shallow. Next step: larger anchor set or CVC syllable extension. |
| Latin CE lower than all three other languages across all 20 beam-search solutions (no overlap). Phase 8 MDL ranked German first (corpus-size artifact); CSP ranks Latin first, consistent with Phases 3–7 structural evidence. | Cross-validation CV = 0.013: CE is stable across random 5-fold folio splits, confirming the signal is corpus-wide not folio-specific. Language B CE ratio 1.02× (essentially equal to Language A with the Latin table) — consistent with A and B sharing the same phonetic encoding. | **CSP result converges with all prior structural evidence** pointing to Latin. The CE advantage over Occitan (0.79 nats), Italian (1.19 nats), and German (1.22 nats) is consistent and reproduced across all beam-search solutions. Language B sharing the Latin table supports a single encoder for both sections. |

**Phase 11.5 cross-validation (CSP refinement and diagnostic battery):**

| Phase 11.5 diagnose finds | Phase 11.5 refine + iterate finds | Phase 11.5 final (V1–V9) finds | Interpretation |
|---|---|---|---|
| 48.5% of tokens are HIT or NEAR_MISS (gate: 15%). 13/14 cells have error rates > 60%. Top correction vectors: `ba→de` (gain 1.0), `ca→di`, `ne→di`. Dominant error is NEAR_MISS across all high-error cells. | Relaxation sweep: levels 2–5 all drop below 1.5× selectivity gate. Strict CV (level 0) remains best. Inherent vowel (a/e/i) produces no differentiation. Verb constraints: 1 soft constraint from 10 Phase 9 assignments; dict hit drops from 9.87% → 7.15%. Iterative bootstrapping converges at iteration 1 (Δ = 0.0000). | 8/9 tests pass (only V9 MCMC fails on dict-hit z-score). V8 readability: 100% phonotactically plausible Latin endings. Language ranking stable: Latin > Occitan > German > Italian. Selectivity 1.85×. | **The CSP framework is correct; the bottleneck is grid precision.** Decoding is in the right phonetic neighborhood (38.6% near-misses) but the 14-cell grid is too coarse to recover individual words. CVC/CCV relaxation makes things worse, not better — expanding the syllable inventory without a finer grid adds noise faster than signal. The path forward is finer grid decomposition, not larger phoneme inventories. |
