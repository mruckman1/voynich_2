[← Phases 4-5](phase-04-05.md) | [Phase Index](README.md) | [Next: Phases 8-9 →](phase-08-09.md)

# Phases 6-7: Illustration-Constrained Decoding & Distributional Semantics

## Phase 6: Illustration-Constrained Decoding

Phase 5 hit a "selectivity ceiling" — frequency-matched random Latin words scored as well as real medical vocabulary (selectivity 0.99x). Phase 6 breaks this ceiling by inverting the approach: instead of decode-then-validate, it uses botanical illustration identifications as cross-modal constraints that pin specific Latin plant names to specific folios, then checks whether a consistent character-to-sound mapping emerges across multiple anchor folios.

The pipeline: **E** (encoding model test) → **D** (Rosetta folio selection) → **B** (paradigm filtering) → **A** (anchor-and-propagate) → **C** (competitive ID resolution) → **Validation**

### Phase 6.0: Illustration-Constrained Setup

Parses the multi-source plant identification concordance and maps Linnaean binomials to medieval Latin names.

| Component | Description | Module |
|-----------|-------------|--------|
| 6.0a: Concordance parsing | Parse 70-entry multi-source concordance CSV (Bax, Tucker & Janick, Sherwood, etc.), group by folio | `phases/illustration_constrained.py` |
| 6.0b: Medieval name mapping | Map Linnaean binomials to medieval Latin equivalents with declension metadata (69 plants, 63 resolved) | `phases/illustration_constrained.py` |
| 6.0c: Tier classification | Tier 1 (genus consensus across sources), Tier 2 (single high-confidence), Tier 3 (contested) | `phases/illustration_constrained.py` |
| 6.0d: Dominant stem extraction | Extract the most frequent morpheme stem from each folio's tokens | `phases/illustration_constrained.py` |

### Phase 6 D+E: Rosetta Folio Selection

Scores folios on 5 criteria and selects the best set for anchor testing, then tests encoding models.

| Component | Description | Module |
|-----------|-------------|--------|
| D.1: Folio scoring | Score each folio on: ID confidence, name distinctiveness, dominant stem clarity, EVA char coverage, char novelty | `phases/rosetta_selection.py` |
| D.2: Greedy selection | Select folios maximizing combined score + character diversity across the set | `phases/rosetta_selection.py` |
| E.1: Encoding model test | Test 4 models (syllabic, alphabetic, abbreviated, mixed) by comparing expected vs observed token structure | `phases/rosetta_selection.py` |

### Phase 6 A+B: Anchor-and-Propagate

The core decoding engine. Hypothesizes that each Rosetta folio's dominant stem = the medieval Latin plant name, applies paradigm filtering, then checks cross-consistency of character mappings.

| Component | Description | Module |
|-----------|-------------|--------|
| A.1: Anchor hypotheses | For each Rosetta folio, align EVA chars of dominant stem to Latin chars of medieval name | `phases/anchor_propagate.py` |
| B.1: Paradigm filtering | Reject hypotheses where Voynich paradigm shape is incompatible with Latin declension class | `phases/anchor_propagate.py` |
| A.2: Cross-consistency | For each EVA char appearing in 2+ anchors, check if the same Latin mapping is assigned unanimously | `phases/anchor_propagate.py` |
| A.3: Propagation | Apply consensus mapping to decode all non-anchor herbal_a tokens | `phases/anchor_propagate.py` |
| A.4: Null tests | (1) Shuffle which folio text goes with which plant name; (2) Replace plant names with random Circa Instans words | `phases/anchor_propagate.py` |

### Phase 6 C: Competitive ID Resolution

Beam search over competing plant identifications for contested folios.

| Component | Description | Module |
|-----------|-------------|--------|
| C.1: Contested enumeration | Identify Tier 2+3 folios with multiple candidate IDs | `phases/competitive_id.py` |
| C.2: Beam search | Greedy beam search (width=10), add one contested folio at a time, keep top states by unanimity × log(n_chars+1) | `phases/competitive_id.py` |

### Phase 6 Validation

Full validation battery with null tests, leave-one-out, train/test split, and bootstrap stability.

| Component | Description | Module |
|-----------|-------------|--------|
| V.1: Three null tests | Shuffled tokens, shuffled characters, random plant names — each must show selectivity > 1.5x | `phases/illustration_validate.py` |
| V.2: Leave-one-out | Remove each anchor, rebuild mapping, check stability | `phases/illustration_validate.py` |
| V.3: Train/test split | 60/40 split, test generalization of character mapping | `phases/illustration_validate.py` |
| V.4: Bootstrap stability | Resample anchors 200x, verify unanimity CI width | `phases/illustration_validate.py` |
| V.5: Stop conditions | Hard stop (<0.20 or all nulls fail), soft stop (0.20–0.50), green light (>0.50 + all nulls >1.5x) | `phases/illustration_validate.py` |

**Gate structure:** illustration_constrained (>=8 Tier 1+2 folios) → rosetta_selection (>=8 folios, score >0.5) → anchor_propagate (unanimity >0.50, z >2.0) → competitive_id (separation >0.05) → validation (stop conditions)

## Phase 6.1: TF-IDF Stem Extraction and Diagnostic Investigation

Phase 6 produced z-scores of 32.0 and 6.75 (real signal exists) but unanimity of only 0.40 (below 0.50 threshold). Root cause: the dominant stem heuristic selected the most frequent stem per folio, picking up corpus-wide function words ("daiin") rather than folio-specific plant names. Phase 6.1 applies three fixes.

### Fix A: TF-IDF Stem Extraction

Replaces frequency-based stem selection with specificity-based selection. For each stem on each folio, computes four metrics: TF-IDF, specificity ratio (tf/cf), exclusivity ratio (this_folio/other_folios), and PMI. The stem with highest TF-IDF score becomes the dominant stem.

| Component | Description | Module |
|-----------|-------------|--------|
| A.1: Corpus stem statistics | Compute corpus-wide term frequency, document frequency for all stems across herbal_a | `phases/illustration_constrained.py` |
| A.2: Per-folio specificity | Four metrics per stem per folio: TF-IDF, specificity ratio, exclusivity, PMI | `phases/illustration_constrained.py` |
| A.3: Diagnostic comparison | Compare old (frequency) vs new (TF-IDF) dominant stems, report changes | `phases/illustration_constrained.py` |

### Fix B: Anchor-Level Inconsistency Diagnosis

Diagnoses which specific anchors and character mappings cause inconsistencies.

| Component | Description | Module |
|-----------|-------------|--------|
| B.1: Per-anchor profiling | For each anchor, count consistent vs conflicting character-reuse instances | `phases/anchor_diagnosis.py` |
| B.2: Poison detection | Leave-one-anchor-out unanimity; poison = removal improves unanimity by >0.05 | `phases/anchor_diagnosis.py` |
| B.3: Per-character profiling | For each EVA char in 2+ anchors, classify unanimity as high/medium/low | `phases/anchor_diagnosis.py` |
| B.4: Iterative pruning | Remove worst poison anchors until unanimity > 0.50 or 5 anchors remain | `phases/anchor_diagnosis.py` |

### Fix C: Encoding Model Diagnosis

Tests which encoding model best fits the anchor data per-anchor, with segmentation sensitivity.

| Component | Description | Module |
|-----------|-------------|--------|
| C.1: Per-anchor model fit | Test 4 models (syllabic, alphabetic, abbreviated, mixed) per anchor | `phases/encoding_diagnosis.py` |
| C.2: Model consensus | Rank models by count of good-fit anchors (fit < 0.3) | `phases/encoding_diagnosis.py` |
| C.3: Segmentation sensitivity | Test 3 segmentation strategies under winning model | `phases/encoding_diagnosis.py` |
| C.4: Hybrid model test | Check if short/medium/long names fit different models | `phases/encoding_diagnosis.py` |

## Phase 7: Morpheme Distributional Semantics & Positional Slot Analysis

Phases 5-6 hit a selectivity ceiling: individual token-to-word mappings can be satisfied by combinatorial abundance. Phase 7 attacks from two orthogonal angles that use the *entire corpus* rather than small anchor sets, testing global structural properties that don't require individual token identification.

**Approach 8** asks whether the geometric structure of Voynich stem embeddings matches a specific language's vocabulary space (via Procrustes and Gromov-Wasserstein alignment). **Approach 9** asks whether Voynich pharmaceutical text follows Latin recipe positional structure (verb-initial slots, noun/ingredient positions).

If both independently point to the same language, that's convergent evidence stronger than either alone.

### Approach 8: Morpheme-Level Distributional Semantics

Builds PPMI + SVD embedding spaces for Voynich stems, aligns them against Latin and Occitan reference embeddings.

| Sub-phase | Description | Module |
|-----------|-------------|--------|
| 8.1 | **Build Voynich embeddings** — Decompose tokens to stems via morpheme decomposition, build co-occurrence matrix (window=2), PPMI with smoothing (alpha=0.75), SVD to 50 dimensions. Separate spaces for Language A (412 stems, 8,652 tokens) and Language B (714 stems, 19,133 tokens). Validate via k-means ARI against section labels. | `phases/distributional.py` |
| 8.2 | **Build reference embeddings** — Same PPMI+SVD pipeline on Latin (Circa Instans + De Viribus Herbarum, 3,269 stems) and Occitan (Regime du Corps, 1,719 stems) with heuristic suffix stripping. | `phases/distributional.py` |
| 8.3 | **Alignment** — Procrustes rotation using 14 seed pairs from Phase 5.3 stem identifications. Gromov-Wasserstein structural comparison (no seeds needed, top-100 stems). Both A and B aligned independently against all references. | `phases/distributional.py` |
| 8.4 | **Affix embeddings** — Build affix-stem co-occurrence matrix, PPMI+SVD to 20 dims. Test whether prefixes and suffixes cluster separately. | `phases/distributional.py` |
| 8.5 | **Cluster correspondence** — Cluster both Voynich and reference spaces (k=5), compare cluster size distributions via rank correlation. | `phases/distributional.py` |
| 8.6 | **Null tests** — Shuffle seed pair assignments for Procrustes; random orthogonal rotation for GW. Selectivity = null_mean / real_residual (higher is better). Gate: > 1.5x. | `phases/distributional.py` |

### Approach 9: Pharmaceutical Positional Slot Analysis

Tests whether Voynich pharmaceutical text follows rigid positional slot structure like Latin recipe texts.

| Sub-phase | Description | Module |
|-----------|-------------|--------|
| 9.1 | **Latin recipe structure** — Segment Circa Instans into 1,234 recipes using verb-initial markers (recipe, accipe, misce, etc.). Label each token's word class (verb/noun/connector/other). Compute slot entropy by position — expect position 1 to be dominated by verbs. | `phases/positional_slots.py` |
| 9.2 | **Voynich pharmaceutical analysis** — Segment Language B pharmaceutical/recipes sections into 1,458 segments using paragraph breaks. Decompose tokens, record stem and paradigm class at each position. Compute MI(paradigm, position). | `phases/positional_slots.py` |
| 9.3 | **Cross-validate position x paradigm** — Build contingency table (position-class x paradigm-class). Cohen's kappa and chi-squared. Compare to Latin contingency table. Gate: kappa > 0.3, selectivity > 1.5x. | `phases/positional_slots.py` |
| 9.4 | **Verb identification** — Stems appearing at position 1 in >= 60% of their occurrences with verb-like paradigm shape. Rank by frequency, compare to Latin recipe verb ranking via Spearman correlation. | `phases/positional_slots.py` |
| 9.5 | **Ingredient identification** — Post-verb medial stems with noun-like paradigms. Cross-reference with herbal folio occurrence to identify plant names that bridge herbal and pharmaceutical sections. | `phases/positional_slots.py` |

### Approach Integration

Cross-validates Approaches 8 and 9 at multiple convergence points.

| Test | Description | Module |
|------|-------------|--------|
| Verb coherence | Do Approach 9 verb candidates cluster together in Approach 8 embedding space? (mean pairwise cosine sim vs 100 random samples) | `phases/approach_integration.py` |
| Noun coherence | Do noun candidates cluster together? | `phases/approach_integration.py` |
| Slot-embedding kappa | Cohen's kappa between k-means cluster labels and positional-slot labels (verb/noun/other) | `phases/approach_integration.py` |
| Joint null test | Shuffle both slot labels and embedding cluster assignments, recompute kappa. Selectivity > 1.5x. | `phases/approach_integration.py` |

## Phase 7.5: Exploiting the Noun Coherence Bridge

Phase 7 produced one metric clearing the 1.5x selectivity threshold: **noun embedding coherence at 5.38x**. Phase 7.5 exploits this bridge to attempt vocabulary identification through converging constraints from embeddings (Phase 7/8), positional slots (Phase 9), illustration anchors (Phase 6.1), and morpheme decomposition (Phase 4.5).

Five steps build incrementally, each gated by selectivity tests. The convergence scoring step then applies Fisher's combined probability test across all independent evidence families from the entire pipeline.

### Step 1: Combined A+B Corpus Embeddings

Merges Language A and Language B tokens into a single PPMI+SVD embedding space to maximize vocabulary coverage and test register structure.

| Sub-step | Description | Module |
|----------|-------------|--------|
| 1.1 | **Build combined space** — Concatenate all A+B tokens (28,459 total), sweep dimensions (50/75/100), select best by section ARI. Tags each stem with source language (A-only: 63, B-only: 268, shared: 632). | `phases/distributional.py` |
| 1.2 | **Register structure** — ARI between k-means clusters and A/B language labels (0.038). Mean cosine separation between A-stem and B-stem centroids (0.317). Low ARI = registers mostly merged in combined space. | `phases/distributional.py` |
| 1.3 | **Reference alignment** — Procrustes + GW alignment against Latin and Occitan using existing seed pairs. | `phases/distributional.py` |

**Result:** 963 combined stems, best ARI = 0.115 at 100 dims (4.2% improvement over A-only 0.111). 65.6% shared stems between A and B. Gate **PASS** (combined ARI > null). Procrustes/GW selectivity still below 1.5x (0.98x / 1.00x).

### Step 2: Noun Subcluster Analysis

Clusters the 443 noun candidates (identified by positional slot analysis) using 5 distributional features, then labels subclusters by semantic domain.

| Sub-step | Description | Module |
|----------|-------------|--------|
| 2.1 | **Feature extraction** — For each noun: TF-IDF folio specificity, section entropy, verb co-occurrence rate, paradigm suffix richness, log frequency. Normalized to [0,1]. | `phases/noun_subclusters.py` |
| 2.2 | **Clustering** — K-means sweep k=3..8, optimal k selected by silhouette score. | `phases/noun_subclusters.py` |
| 2.3 | **Labeling** — Heuristic assignment: high TF-IDF + low entropy → plant_names, high freq + low entropy → preparations, high verb co-occurrence → plant_parts, high paradigm richness → qualities. | `phases/noun_subclusters.py` |
| 2.4 | **Latin domain matching** — Compare cluster sizes to `LATIN_PHARMACEUTICAL_DOMAINS` reference table (plant_names: 15, preparations: 9, plant_parts: 8, body_parts: 8, qualities: 4). | `phases/noun_subclusters.py` |
| 2.5 | **Null test** — Shuffle features per-column, recluster 100x, compare silhouette. | `phases/noun_subclusters.py` |

**Result:** 4 optimal subclusters — plant_parts (39 stems), qualities (196), preparations (107), plant_names (101). Silhouette = 0.303. Null mean = 0.234. Selectivity = **1.29x** — gate **FAIL** (below 1.5x). Clusters are distributional but not significantly tighter than shuffled features.

### Step 3: Verb Identification

Matches 15 Voynich verb candidates against 10 Latin pharmaceutical imperatives using Hungarian optimal assignment on a multi-criteria compatibility matrix.

| Sub-step | Description | Module |
|----------|-------------|--------|
| 3.1 | **Verb profiling** — For each of 15 verb candidates: frequency, paradigm form count, position-1 concentration %, section distribution, co-occurring nouns, stem character length. | `phases/verb_identification.py` |
| 3.2 | **Compatibility matrix** — 15x10 matrix with 6 criteria per pair: frequency rank proximity, paradigm match, stem length compatibility, positional profile similarity, object noun compatibility (via subclusters), character mapping consistency (via Phase 6.1). | `phases/verb_identification.py` |
| 3.3 | **Hungarian assignment** — Optimal 1:1 matching. Best total score = 6.28, second-best gap = 1.4%. | `phases/verb_identification.py` |
| 3.4 | **Cross-consistency** — Check implied character mappings against Phase 6.1 high-unanimity consensus. | `phases/verb_identification.py` |
| 3.5 | **Null test** — Shuffle compatibility matrix columns independently 100x, re-assign. | `phases/verb_identification.py` |

**Result:** 9/10 confident assignments (top match: tshod→adde at 0.789). Selectivity = **0.92x** — gate **FAIL**. The null test scores higher than real because the matrix has low variance across columns, meaning many different assignments achieve similar scores. Individual verb identifications are plausible but not discriminative.

Top assignments: pchedar→misce (0.706), pcheor→coque (0.656), polche→pone (0.658), psheo→cola (0.672).

### Step 4: Illustration-Embedding Bridge

Tests whether Rosetta folio plant stems (from Phase 6) land in the plant_names embedding subcluster, then attempts three-way convergent anchor expansion.

| Sub-step | Description | Module |
|----------|-------------|--------|
| 4.1 | **Locate Rosetta stems** — For each of 8 Rosetta folio dominant stems, compute cosine distance to each subcluster centroid. | `phases/embedding_bridge.py` |
| 4.2 | **Anchor expansion** — For each of 112 herbal folios, check if TF-IDF top stem falls in plant_names cluster AND is a noun candidate by position. Three-way convergence = illustration + cluster + positional class. | `phases/embedding_bridge.py` |
| 4.3 | **Null test** — Random stems' plant-cluster hit rate (mean 20.4%). | `phases/embedding_bridge.py` |

**Result:** 0/8 Rosetta stems land in plant_names cluster — they scatter across preparations (5), qualities (2), plant_parts (1). Gate **FAIL**. The subclusters do not align with illustration-based plant identity, suggesting the embedding features capture distributional frequency patterns rather than semantic content.

### Step 5: Convergence Scoring

Aggregates all selectivity tests across the entire pipeline using Fisher's combined probability method, and cross-references multi-method vocabulary identifications.

| Sub-step | Description | Module |
|----------|-------------|--------|
| 5.1 | **Compile scores** — Harvest selectivity ratios and p-values from 7 independent evidence families: morpheme grid, distributional embeddings, illustration anchors, positional slots, noun coherence, verb identification, embedding bridge. | `phases/convergence_score.py` |
| 5.2 | **Fisher's test** — chi² = -2·Σln(p_i) tested against chi²(2k). | `phases/convergence_score.py` |
| 5.3 | **Convergent IDs** — Cross-reference all stems with identifications from 2+ independent methods. | `phases/convergence_score.py` |

**Result:** 10 metrics across 5 independent families. Fisher combined chi² = 65.88 (df=10), **p = 2.75x10^-10** — the aggregate signal is overwhelmingly real. Driven by morpheme grid z-scores (>500), noun embedding coherence (5.38x), verb frequency rho (0.97), and anchor unanimity (5.83x). 76 convergent identifications found, but only 1 stem (tol/viola) has multi-method support from 2+ independent sources.

**Verdict:** The Voynich manuscript's structural properties (morpheme decomposition, embedding geometry, positional slots) are real and converge on a Latin pharmaceutical text model. But individual word identification remains blocked: the selectivity ceiling prevents discriminating correct assignments from frequency-matched alternatives.

---
[← Phases 4-5](phase-04-05.md) | [Phase Index](README.md) | [Next: Phases 8-9 →](phase-08-09.md)
