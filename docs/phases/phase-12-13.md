# Phases 12-13: Grid Recalibration & Context Rules

[← Phase 11](phase-11-csp.md) | [Phase Index](README.md) | [Next: Phase 14 →](phase-14-features.md)

---

## Phase 12: Grid Recalibration

Phase 12 systematically tests all structural explanations for the 11.1% dict_hit ceiling.

| Step | Description | Module |
|------|-------------|--------|
| 12.1–12.2 | Correction vector bias detection; stroke-based character move proposal; co-occurrence validation of proposed moves | `grid_recalibrate.py` |
| 12.4 | Stroke-alignment audit of all 44 EVA glyphs; stroke-based and hybrid grid construction | `grid_alternatives.py` |
| 12.5 | PMI-guided digraph/ligature decomposition; 6 variant sweep (sh, qo, aiin ligature re-splits) | `token_decomposition.py` |
| 12.3+12.6 | Iterative CSP re-solve on all grid variants; V1–V10 validation battery including vocabulary catalog (V10) and progression tracking (V11) | `recalibrated_csp.py` |

### Phase 12 Findings Summary

Phase 12 returns a definitive negative result on three independent structural explanations: (1) stroke analysis shows all 44 EVA glyphs are correctly placed — 0 misaligned characters; (2) 6 token decomposition variants all degrade dict_hit; (3) the Phase 11.5 correction vector bias (60% pointing to "di") is a statistical artifact, not a genuine grid error. The CSP re-solve on the original grid with marginal recalibration reaches **dict_hit = 11.15%, selectivity 1.85×**. V1–V8 all pass. The 11.1% ceiling is confirmed as inherent to the 14-cell CV model, not an addressable grid error.

## Phase 13: Context-Dependent Reading Rules

Phase 13 tests whether the ceiling can be broken by context-sensitive phonetic rules — values that depend on word position or adjacent cells — without changing the grid itself.

| Step | Description | Module |
|------|-------------|--------|
| 13.1 | Needleman-Wunsch alignment of near-miss tokens to nearest dict words; per-cell error catalog with position + adjacency tags; chi-squared tests; MI(correction, context) gate vs 100 shuffles | `error_patterns.py` |
| 13.6 | Cell conflation analysis (how many phonemes each cell must encode); medieval Latin dictionary expansion test; null MI test (alternative explanations) | `null_context.py` |
| 13.2 | Rule formalization from significant cell-context pairs; coverage and power scoring; greedy accumulation with cumulative dict_hit tracking | `rule_extraction.py` |
| 13.3 | Context-aware CSP: Version A (only rule-extracted values) exhaustive search over 256 combinations; Version B (any inventory value) beam search with width 20 × 3 iterations | `context_csp.py` |
| 13.4 | Folio-split cross-validation (odd/even halves); per-rule selectivity vs shuffled-token baseline; linguistic plausibility check against ROMANCE_PHONOLOGICAL_PROCESSES catalogue | `rule_validation.py` |
| 13.5 | Full corpus decoding with validated rules; section text samples; Language B test; vocabulary catalog; V1–V11 validation battery with progression tracking | `context_decode.py` |

### Phase 13 Key Results

| Step | Metric | Value | Gate |
|------|--------|-------|------|
| 13.1 MI gate | MI selectivity (errors vs shuffled) | **20.11×** | PASS (≥ 1.5×) |
| 13.1 Position tests | Cells with significant position dependence | 5/14 (p < 0.0001) | — |
| 13.6 Null tests | Cell conflation severity | 7/14 cells need > 2 phonemes | MODERATE |
| 13.6 Null tests | Near-misses fixed by dict expansion | 6% | MINOR |
| 13.2 Rule extraction | Rules extracted | 8 | — |
| 13.2 Rule extraction | Best single rule (C1V2 ca→t / word_final) | +2.0% dict_hit (9.9%→11.9%) | FAIL (< 15%) |
| 13.3 Version A | Rule-constrained exhaustive (256 combos) | 12.4% dict_hit (+2.5%) | FAIL (< 15%) |
| 13.3 Version B | Free-search beam (20-wide, 3 iterations) | **38.5% dict_hit (+28.6%)** | — |
| 13.4 Cross-validation | Rules validated (all 3 checks) | **0/8** | FAIL |
| 13.4 Cross-validation | Version B held-out performance | 5.5% (vs 9.5% baseline) — overfitting | — |
| 13.5 Full corpus | dict_hit with validated rules | **11.43%** | — |
| 13.5 Full corpus | Selectivity | 1.86× | — |
| 13.5 Validation | V1–V9 battery | 7/9 pass | — |
| 13.5 Progression | Phase 11 → 11.5 → 12 → 13 | 11.1% → 9.87% → 11.15% → **11.43%** | — |

### Phase 13 Findings Summary

Phase 13 produces two distinct conclusions. First, the positive: context-dependent error structure in the near-miss tokens is **real and extremely strong** (MI selectivity 20.11×, 5/14 cells significant by chi-squared). The errors are not random. Near-miss tokens fail in systematic ways that depend on word position — predominantly word-final devoicing (ca→t, si→c at word boundaries) and pre-vowel nasal assimilation (si→m, ci→m before vowels). This is exactly the class of variation predicted by Latin phonotactics.

Second, the negative: none of these rules generalize. The cross-validation transfer rate is 100% (every rule recurs in both corpus halves), but 0/8 rules pass the selectivity gate — applying them to held-out data does not improve dict_hit and in some cases reduces it. The free-search CSP (Version B) achieves 38.5% on its training tokens, but this is the most extreme overfitting seen in any phase: 5 cells × 3 context slots × free inventory choices give enough degrees of freedom to memorize phonetic patterns rather than decode them.

The combined interpretation is: the 14-cell grid does contain real phonetic context-dependence (the MI signal is genuine), but the grid is **too coarse to isolate it as addressable rules**. Each cell conflates too many phonemes (average 4–5 in high-error cells) for any single context rule to cover the majority of cases. The structural ceiling confirmed across Phases 11–13 requires a representation with more than 14 cells — either additional onset/nucleus splits (targeting a ~28–30 cell grid) or a featural/abugida model where position within the cell encodes phonetic context directly.

## Cross-Validation Tables

**Phase 12 cross-validation (grid recalibration):**

| Grid recalibration finds | Stroke audit finds | Decomposition sweep finds | Interpretation |
|---|---|---|---|
| Correction vector bias 60% toward "di". After de-biasing: 0 actionable character moves. Recalibrated grid unchanged from original. | 44/44 EVA glyphs correctly placed by stroke analysis. 0 misaligned characters. No hybrid grid outperforms original. | 6 decomposition variants tested (sh re-split, qo collapse, aiin ligature, etc.). All 6 degrade dict_hit. Best variant = original. | **The ceiling is not caused by structural errors in the grid.** The EVA character placement is correct. The bottleneck is the coarseness of the CV model at 14 cells, not any fixable character assignment. |
| CSP re-solve on original: dict_hit = 11.15%, selectivity 1.85×, V1–V8 all pass. | V10 vocabulary catalog: 13 confirmed Latin hits, 7 function words. Progression: Phase 11 11.1% → Phase 12 11.15%. | V11 progression: marginal +0.05% improvement across 3 iterations of recalibration. | **The 11.1% ceiling is structural.** No grid manipulation approach can lift it. Phase 13 tests context-dependent reading rules as the final structural explanation. |

**Phase 13 cross-validation (context-dependent reading rules):**

| Error pattern analysis finds | Rule extraction + CSP finds | Cross-validation finds | Interpretation |
|---|---|---|---|
| MI selectivity 20.11× (threshold 1.5×). 5/14 cells with chi-squared p < 0.0001. Dominant patterns: word-final devoicing (ca→t, si→c), pre-vowel nasal assimilation (si→m, ci→m). | 8 rules extracted. Best single rule +2.0% (ca→t word-final). Version A (rule-constrained) reaches 12.4%. Version B free-search reaches 38.5%. | 0/8 rules pass all three gates (transfer, selectivity ≥ 1.5×, plausibility). Version B on held-out half: 5.5% (worse than baseline 9.5%). Transfer rate 100% but selectivity 1.00× — rules recur but do not improve held-out dict_hit. | **Context-dependence in the error signal is genuine (20.11× MI), but the grid is too coarse to isolate it as actionable rules.** The free-search CSP overfits with 5 × 3 free parameters. The ceiling requires finer grid resolution (≥ 28 cells or abugida model), not more context variables. |
| Null hypothesis tests: cell conflation moderate (7/14 cells need > 2 phonemes, avg 4–5 phonemes/high-error cell), dictionary expansion explains only 6% of near-misses. | Full corpus: 11.43% dict_hit, 1.86× selectivity, 7/9 validation tests pass. Progression: 11.1% → 9.87% → 11.15% → **11.43%**. | V11 confirmed: all improvements since Phase 11 are within 0.5% — the ceiling is robust across all three post-Phase-11 approaches (relaxation, grid recalibration, context rules). | **The 11.1% ceiling is confirmed across three independent attack vectors.** It is structural, not incidental. Next steps require a fundamentally different phonological representation. |
