# Phases 43-48: Optimization Landscape, Structural Reading, Bilingual Cribs

[← Phases 36-42](phase-36-42.md) | [Phase Index](README.md) | [Next: Phases 49-53 →](phase-49-53.md)

**Phase 43:** LATERAL (structural probing positive, inversion and HMM regressed)
**Phase 44:** SCORING_WEAK (landscape FLAT, 500+ near-optimal MaxSAT solutions)
**Phase 45:** FREQUENCY_ARTIFACT (SBM communities are frequency tiers)
**Phase 46:** TABLE_SELECTED_T_P15 (composite 0.985, z_total=61.63 at 10K)
**Phase 47:** READING_ONLY (305 n-grams, 89 recipes, z-audit resolved)
**Phase 48:** CRIB_SUGGESTIVE ("sheey"->"sera" partial match to "cera" at ED=1)

---

## Phase 43: Re-Encoding Inversion, Structural Probing, and Conditional Decoding

**Verdict: LATERAL** -- 1/3 approaches positive. Phase 16 table confirmed as robust local optimum.

### Approach 1: Re-Encoding Inversion -- REGRESSION
Simulated annealing (100K iterations x 5 restarts) to find tables whose encoded plaintext matches the Voynich fingerprint. Italian won (cost=6.87) over Latin (7.08). But inverted table: dict-hit 20.6% with 1.18x selectivity. 0/25 triples agree with Phase 15. Full validation: 1/5 passed.

### Approach 4: Signal Word Structural Probing -- STRUCTURAL_SIGNAL
Mapped 1,320 occurrences of 6 active bedrock words across 211 folios. All 6 non-uniformly distributed (chi-sq p<0.005). Herbal_a contains 44.7% of signal words (vs 26% of tokens). Classified 226 folios by type: DESCRIPTION (64), UNKNOWN (100), RECIPE_COLLECTION (14), FORMULAIC (12), SPARSE (36). Estimated 34 recipes across 14 folios. Signal density declines systematically: herbal_a (7.6%) -> pharmaceutical (5.0%) -> biological (4.3%) -> stars (2.9%).

### Approach 5: Context-Dependent HMM -- REGRESSION
K=100 HMM, V=44 EVA chars. Dict-hit: 11.2% (vs Phase 15's 39.1%). Agreement with Phase 15: 0%. Null corpora produce identical 11.2%. Bigram z=0.00.

## Phase 44: Solution Landscape Enumeration via MaxSAT, Stochastic Block Models, and Coupled Simulated Annealing

**Verdict: SCORING_WEAK** -- landscape is FLAT. 6/8 validations pass.

### Track A: Weighted Partial MaxSAT -- FLAT
327 Boolean variables, 2,751 hard clauses, 46,648 soft clauses. 2 optimal solutions (cost 59,306). At 1% relaxation: 500 solutions (capped). Best dict-hit: 52.7% subsample but full corpus: 41.76% vs Phase 15's 43.63% (delta=-1.88%). 8 triples changed. **MAXSAT_WORSE.**

### Track B: Stochastic Block Model -- STABLE (NO_CONVERGENCE)
4-layer adjacency matrices over 44 EVA characters. k=6 communities. SBM vs stroke triples: ARI=0.002, NMI=0.395. SBM vs sign families: ARI=0.033. But split-half ARI=0.831 -- communities are highly stable. **SBM finds novel distributional structure unrelated to visual features.**

### Track C: Coupled Simulated Annealing -- CSA_WORSE
10 coupled chains x 200,000 iterations (2M evaluations). Throughput: 55,854 eval/s. Best: 48.95% subsample vs Phase 15's 51.65%. Full corpus: 41.09% vs 43.63%. Null selectivity: 1.10x. All 13 free triples changed -- completely different assignment.

### Key Findings
1. **Landscape is flat**: 500+ near-optimal solutions. No alternative outperforms Phase 15.
2. **Phase 15 not special within landscape**: not found among MaxSAT or CSA solutions, yet none outperform it.
3. **Distributional structure real but orthogonal to visual features**: ARI=0.002 with strokes, but 0.83 split-half stability.
4. **Bottleneck is scoring function, not search**: CSA at 56K eval/s thoroughly samples.
5. **Energy and dict-hit anti-correlated at convergence**.

## Phase 45: SBM Community Forensics and Distributional Re-encoding

**Verdict: FREQUENCY_ARTIFACT** -- 4/8 validations pass. Gate FAIL.

### Key Findings
- Community 0 absorbs 28/44 characters (98.5% corpus coverage)
- Frequency-rank vs community Spearman=0.82
- Best labeling: frequency_tier (ARI=0.25), far above onset_consonant (0.03) or vowel (0.04)
- Hybrid decode: +0.03% (negligible)
- Community landscape: FLAT with 12,424 near-optimal solutions

### Triple Confidence Consolidation
- **12 CONFIRMED** (cross-source validated)
- **10 LANDSCAPE_CONFIRMED** (MaxSAT consensus >=60%)
- **3 GENUINELY_AMBIGUOUS** (no consensus, covering only 164 tokens / 0.45%)
- Ambiguity budget: 0.04% total dict-hit range -- resolving ambiguous triples cannot meaningfully change performance

## Phase 46: Final Internal Consolidation

**Verdict: TABLE_SELECTED_T_P15** -- composite score 0.985, z_total=61.63 at 10K. All 6/6 validations pass.

### Track A: Triple Arbitration

8 candidate tables evaluated. T_P15 wins:

| Rank | Table | Composite | z_total (10K) | Signal Survival | Dict Hit (10K) |
|------|-------|-----------|--------------|-----------------|----------------|
| 1 | **T_P15** | **0.985** | 61.63 | 1.000 | 0.216 |
| 2 | T_P15_10K | 0.969 | 59.28 | 1.000 | 0.216 |
| 3 | T_BEST6 | 0.967 | 57.45 | 1.000 | 0.216 |
| 8 | T_CSA | 0.793 | 41.03 | 0.875 | 0.196 |

MaxSAT disagreements are **non-additive**: individual swaps improve z (up to +7.5), but combining 6/8 yields z=57.45 -- lower than unmodified T_P15 (61.63).

### Track B: Frequency Structure Diagnostic

Voynich nearest SBM match: **Italian character-level text** (distance 0.449). Tachygraphic CV cipher second (0.526). Simple substitution (0.786) and homophonic (1.136) far more distant.

### Track C: Definitive Corpus Decode

36,238 tokens, 43.63% dict-hit, 25.74% signal rate, 53 signal words.

| Level | Count | Rate |
|-------|-------|------|
| GREEN | 5,853 | 16.2% |
| YELLOW | 7,009 | 19.3% |
| ORANGE | 23 | 0.1% |
| RED | 23,353 | 64.4% |

Three HIGH-priority gaps remain: external tachygraphy tables, sharper language model, word-level context models.

## Phase 47: Z-Score Audit, Word Disambiguation, Structural Reading, and Sequence Analysis

**Verdict: READING_ONLY** -- 7/8 validations pass.

### Track A: Z-Score Methodology Audit

Resolved 10x z-score discrepancy between Phase 29 (z=6.14) and Phase 46 (z=61.63):

| Factor | Impact on z |
|--------|-------------|
| Dictionary 131K -> 10K only | +8.27 |
| Hit counting exact -> exact+relaxed only | +9.39 |
| Both combined | +53.28 |

Interaction is **superlinear**. Conservative minimum z = 14.78 (exact-only, 10K dict, 500 perms).

### Track B: Word-Level Disambiguation -- NOT_BENEFICIAL

Word-level Viterbi **decreased** dict-hit from 24.0% to 23.6%. The fundamental problem: corpus bigram statistics derived from the same noisy decoding. Circular optimization.

### Track C: Structural Reading

- 305 recurring n-grams (39 at trigram level+)
- 89 recipes extracted with "cola"/"codi" boundaries
- k=2 topic clusters (herbal_a vs everything else)
- 36 gloss attempts on star folios

### Track D: Manuscript Sequence Analysis

Consecutive folios 1.30x more similar than random. **0 anomalous boundaries.** NO_REORDER -- page sequence is consistent with decoded content.

## Phase 48: Marginal Bilingual Crib Exploitation

**Verdict: CRIB_SUGGESTIVE** -- 8/8 validations pass.

### Track A: f116v Decode

IVTFF line: `oror.sheey<!valsch vbren so nim gaf mich o>` (Eastern Bavarian German).

- "sheey"->"sera" (10K dict hit, all Tier 1 triples). PARTIAL_MATCH to scholarly reading "cera" (wax) at edit distance 1.
- "oror"->"nene" (no dict hit, Tier 2 triples).
- 0 new triple assignments derivable.

### Track B: Secondary Marginals

f17r has no Latin-alphabet text in IVTFF. f66r's "muss mel" (Swabian/Alsatian) is visual-only, no machine-readable transcription. Different annotators: f116v = Hand 3 (Eastern Bavarian), f66r = Hand 5 (Swabian/Alsatian).

### Track C: Marci Annotations

f1r annotations attributed to Marci (17th century). Data: **UNAVAILABLE** -- no machine-readable transcription.

### Track D: Crib Propagation

65 cribs collected, **0 accepted** (all break signal words or degrade dict-hit). Bigram z=59.28 (canonical, unchanged). Table unchanged.

### Progression

| Phase | Dict Hit | Selectivity | Key Advance |
|-------|----------|-------------|-------------|
| Phase 11 | 11.1% | 1.92x | CSP phonetic decoder |
| Phase 14 | 19.4% | 3.00x | 25 stroke-feature triples |
| Phase 15 | 35.4% | 2.55x | Medieval dictionary expansion |
| Phase 16 | 43.6% | 3.38x | Modifier detection |
| Phase 29 | 43.6% | 3.38x | Signal bigram z=6.14 |
| Phase 33 | 43.6% | 3.38x | Table confirmed |
| Phase 44 | 43.6% | 3.38x | MaxSAT landscape FLAT |
| Phase 46 | 43.6% | 1.13x | Final consolidation (T_P15) |
| Phase 47 | 43.6% | -- | READING_ONLY |
| Phase 48 | 43.6% | -- | CRIB_SUGGESTIVE |

---

[← Phases 36-42](phase-36-42.md) | [Phase Index](README.md) | [Next: Phases 49-53 →](phase-49-53.md)
