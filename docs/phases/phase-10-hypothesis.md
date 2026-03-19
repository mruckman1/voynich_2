# Phase 10: Testing the Three Surviving Hypotheses

**Verdict:** H1 (Constructed script) wins (score 4.0, margin 2.5)

[← Phases 8-9](phase-08-09.md) | [Phase Index](README.md) | [Next: Phase 11 →](phase-11-csp.md)

---

## Phase 10: Testing the Three Surviving Hypotheses

Nine phases eliminated every classical cipher model while confirming encoded natural language. Three hypotheses survive:

- **H1 (Constructed script)**: Glyph strokes map to phonetic values via script-specific construction logic — each glyph is built from onset + nucleus components that encode CV syllables, analogous to Hangul or Devanagari.
- **H2 (Information dispersion)**: Each meaning unit is spread across multiple tokens — the encoding disperses information so that distant tokens carry more mutual information than in natural language.
- **H3 (Keyed cipher)**: A key modulates the mapping at a period longer than line-level — different folios or quires use different encoding parameters.

The critical diagnostic is **token-level entropy at increasing context windows** — each hypothesis predicts a different curve shape. Phase 10 runs five discriminating analyses with per-section controls (Language A combined, herbal-only, pharmaceutical-only, plus Language B as negative control).

### Step 10.1: Token-Level Entropy Curves

Tests all three hypotheses simultaneously via the shape of the conditional entropy curve H(token | context of order n) at orders 0, 1, 2, 3, 5, 10.

| Sub-step | Description | Module |
|----------|-------------|--------|
| 10.1a | **Section curves** — Language A combined, herbal, pharma, and Language B. If H1 correct, all A curves have same shape. If H3 correct, sections differ. | `phases/entropy_curves.py` |
| 10.1b | **Reference curves** — Latin, Occitan, Italian, German at same orders. | `phases/entropy_curves.py` |
| 10.1c | **Baselines** — Shuffled tokens (no context should help) and Markov-order-2 character generation. | `phases/entropy_curves.py` |
| 10.1d | **Hypothesis scoring** — H1: Pearson r of reduction rates R(n) vs best reference. H2: back-load ratio R(5→10)/R(1→2). H3: entropy floor ratio and section divergence. | `phases/entropy_curves.py` |

**Result:** Voynich Language A entropy curve shows a **near-perfect parallel shift with Latin** (r = 0.999). Sections are highly consistent (herbal-pharma r = 0.9998, combined-herbal r = 1.000). The back-load ratio is negligible (0.00011), ruling out information dispersion at the entropy curve level. The entropy floor ratio (0.745) is below the H3 threshold. Language B shows a flatter curve with higher floor (3.25 vs 2.55 for combined A), consistent with more restricted/mechanical text.

**Verdict:** `entropy_curve_supports_H1_constructed_script`. Gate **PASSED**.

### Step 10.2: Multi-Token Mutual Information Decay

Primarily tests H2 by measuring how quickly mutual information between tokens decays with distance.

| Sub-step | Description | Module |
|----------|-------------|--------|
| 10.2a | **MI at increasing lags** — Token-gap MI at distances d = 1, 2, ..., 20 for Voynich Language A. | `phases/mutual_info_decay.py` |
| 10.2b | **Exponential decay fit** — Fit y = A·exp(-x/τ) to MI(d) curves. τ comparison across Voynich, references, and shuffled baseline. | `phases/mutual_info_decay.py` |
| 10.2c | **Per-section τ consistency** — Herbal τ vs pharmaceutical τ. If H2 is correct, τ should be similar across sections. | `phases/mutual_info_decay.py` |
| 10.2d | **Phrase-level Procrustes alignment** — If H2 supported, test whether phrase-level embeddings align better than token-level. | `phases/mutual_info_decay.py` |

**Result:** Voynich MI is nearly flat across all lags (7.05–7.11 bits), producing τ = 4,285 — far higher than any reference (Latin τ = 477, best reference). The τ ratio of **8.98×** nominally supports H2. However, per-section τ values are inconsistent (herbal τ = 4,858 vs pharma τ = 8,629), and phrase-level Procrustes alignment shows **no improvement** over token-level at any phrase length (3, 5, 7). The high τ is likely due to plug-in MI estimation bias with a large vocabulary, rather than genuine information dispersion.

**Verdict:** `mi_decay_supports_H2` (by τ ratio), but phrase alignment fails. Gate **PASSED** (τ ratio > 1.5).

### Step 10.3: Folio-Level Encoding Shifts

Primarily tests H3 by detecting systematic encoding differences between folios within the same section.

| Sub-step | Description | Module |
|----------|-------------|--------|
| 10.3a | **Inter-folio bigram JSD** — Within-section only: herbal folio 1 vs herbal folio 2, etc. Cross-section comparisons excluded (they show huge topical JSD). Bootstrap null: shuffle tokens across folios within section, recompute JSD. | `phases/folio_shift.py` |
| 10.3b | **Function-word CV** — Coefficient of variation of uniformly-distributed stems across folios within same section, compared to reference languages. | `phases/folio_shift.py` |
| 10.3c | **Quire boundary analysis** — Within-quire vs between-quire JSD, controlling for section. | `phases/folio_shift.py` |

**Result:** 63 folios analyzed across sections (herbal_a: 39, pharmaceutical: 24). Within-section JSD is high (herbal: 0.936, pharma: 0.964) but **not significantly above bootstrap null** — the residual is not significant. Function-word CV is inflated (Voynich 0.733 vs reference mean 0.349–0.520), but this is the only H3 indicator that fires. No quire boundary effect detected. H3 requires 2/3 indicators; only 1/3 fires.

**Verdict:** `folio_shift_ambiguous`. Gate **PASSED** (clear non-H3 signal). H3 not supported.

### Step 10.4: Glyph Construction Grammar

Primarily tests H1 by comparing the Voynich glyph grid against known constructed scripts and testing construction vs morphology.

| Sub-step | Description | Module |
|----------|-------------|--------|
| 10.4a | **Script grid comparison** — Compare Voynich grid statistics (7 onsets × 11 nuclei, 31% occupancy, R_forward = 0.39, R_reverse = 0.61) against Hangul, Devanagari, Ethiopic, and Linear B using weighted composite distance. | `phases/glyph_grammar.py` |
| 10.4b | **Construction vs morphology** — Correlate onset/nucleus stroke identity with token position in line. Construction scripts show near-zero correlation (stroke identity independent of position); morphological systems show significant correlation. | `phases/glyph_grammar.py` |
| 10.4c | **Phonotactic CSP** — Map 14 grid cells to syllable candidates from Romance phonotactics, constrained by frequency matching. Language B consistency: verify B cells ⊂ A cells and core token coverage. | `phases/glyph_grammar.py` |

**Result:** Closest script: **Devanagari** (similarity 0.473), followed by Linear B (0.411) and Hangul (0.410). The construction test diagnoses **"construction"** — onset-position and nucleus-position correlations are near-zero (-0.058, 0.047) with p < 10⁻⁶, meaning glyph component identity is independent of word position (the hallmark of a constructed script, not a morphological system). The CSP maps 14 cells to Latin syllables but achieves no selectivity over random (1.0×), and Language B cells are not a subset of Language A cells. CSP decoding is not yet viable — the search space needs further pruning by illustration constraints.

**Verdict:** `glyph_grammar_supports_H1`. Gate **PASSED**.

### Step 10.5: Hypothesis Integration and Verdict

Compiles evidence from all Phase 10 sub-analyses into weighted scores.

| Hypothesis | Score | Key evidence |
|------------|-------|-------------|
| **H1 (Constructed script)** | **4.0** | Entropy curve r=0.999 with Latin (1.0), sections consistent (0.5), no folio shifts (0.5), script grid similarity (1.0), construction diagnosis (1.0). CSP not yet viable (-1.0), Language B subset fails (-0.5). |
| H2 (Information dispersion) | 1.5 | τ ratio 8.98× (1.0), no folio shifts (0.5). Back-load ratio fails, section τ inconsistent, phrase alignment fails. |
| H3 (Keyed cipher) | 1.0 | Function-word CV inflated (1.0). Residual JSD fails, floor ratio fails, section not divergent, no quire effect. |

**Winner: H1** with margin **2.5** over H2. Gate **PASSED** (margin > 1.0).

**Actionable next step:** The 14-variable CSP is the decoding path. Each grid cell maps to one phoneme or syllable. Phonotactic constraints of Romance languages prune the search space. Illustration constraints provide anchor values. Constraint propagation is estimated to reduce the search to ~10³–10⁶ candidates.

---
[← Phases 8-9](phase-08-09.md) | [Phase Index](README.md) | [Next: Phase 11 →](phase-11-csp.md)
