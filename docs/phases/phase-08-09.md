[← Phases 6-7](phase-06-07.md) | [Phase Index](README.md) | [Next: Phase 10 →](phase-10-hypothesis.md)

# Phases 8-9: Cipher-Level Decoding & Fundamental Reassessment

## Phase 8: Bigram Transfer Cryptanalysis & MDL Decoding

Phases 5–7.5 hit a selectivity ceiling (~1.0–1.46x) because they match individual stems to individual words (unigram matching). Phase 8 changes the fundamental unit of analysis with two complementary approaches that exploit higher-order structure:

- **Approach 16 (Bigram Transfer)**: matches stem *pairs* — builds NxN bigram transition matrices for Voynich and target language stems, then uses simulated annealing to find the permutation minimizing Frobenius distance between matrices.
- **Approach 18 (MDL Decoding)**: evaluates *entire candidate decodings* holistically — builds character-level n-gram language models for Latin and Occitan, then finds the stem mapping that minimizes cross-entropy (bits/char) of the decoded text. The best decoding is the most compressible one.

Both operate on the morpheme stem level from Phase 4.5.

### Approach 16: Bigram Transfer Cryptanalysis

| Sub-step | Description | Module |
|----------|-------------|--------|
| 16.1 | **Build bigram matrices** — Stem sequences from Voynich (8,652 tokens, 412 unique), Latin (63,771 tokens), and Occitan (41,779 tokens). Top-100 stems, 100x100 transition probability matrices. | `phases/bigram_transfer.py` |
| 16.2 | **SA permutation search** — For Frobenius distance metric, run 10 restarts x 100K iterations of simulated annealing to find the best stem permutation aligning Voynich→Latin matrices. | `phases/bigram_transfer.py` |
| 16.3 | **Stability analysis** — Pairwise agreement across 10 independent SA restarts. Top-10 consistent mappings with confidence scores. | `phases/bigram_transfer.py` |
| 16.4 | **Validation battery** — 4 null tests (shuffled Voynich, random target matrix, Latin-to-Latin sanity check, Occitan target) + split-half cross-validation by folios. | `phases/bigram_transfer.py` |

**Result:** Selectivity = **1.30x** — gate **FAIL** (below 1.5x threshold). Stability = 0.025 (very low pairwise agreement across restarts). The optimizer reduces Frobenius distance 23% below random baseline, but the signal is not selective enough — many different permutations achieve similar distances. Notably, Occitan fits better than Latin (distance 0.042 vs 0.047).

Top consistent mappings: `ch→et` (conf=0.6), `daiin→eius` (0.6), `che→in` (0.6) — all common function words, consistent with frequency matching rather than genuine decryption.

### Approach 18: Minimum Description Length Decoding

| Sub-step | Description | Module |
|----------|-------------|--------|
| 18.1 | **Build language models** — Character-level trigram and 5-gram LMs for Latin, Occitan, Italian, and German with add-k smoothing. Measure discrimination gap (heldout vs random text). | `phases/mdl_decode.py` |
| 18.2 | **Sanity check** — Encipher Latin stems with a random substitution, attempt recovery via MCMC. Validates the approach works on known ciphers before applying to Voynich. | `phases/mdl_decode.py` |
| 18.3 | **MCMC decoding** — For each target language (Latin, Occitan, Italian, German), run 5 restarts x 100K iterations of simulated annealing with incremental cross-entropy updates. Cost function = bits/char of decoded text under the trigram LM. Language-aware stemmers for each target. | `phases/mdl_decode.py` |
| 18.4 | **Language ranking** — Rank all target languages by cross-entropy and compression ratio. Compare raw CE (affected by corpus size) vs within-language selectivity (normalized for LM quality). | `phases/mdl_decode.py` |
| 18.5 | **Validation battery** — Random mappings baseline, shuffled Voynich, wrong-language check, split-half cross-validation. | `phases/mdl_decode.py` |

**4-Language Ranking (by raw cross-entropy):**

| Rank | Language | CE (bits/char) | Compression | Corpus size |
|------|----------|---------------|-------------|-------------|
| 1 | German | 1.73 | 1.40x | 149K tokens |
| 2 | Occitan | 1.91 | 1.36x | 48K tokens |
| 3 | Italian | 2.17 | 1.77x | 11K tokens |
| 4 | Latin | 2.24 | 1.32x | 74K tokens |

**Result:** Gate **FAILED** — selectivity = **1.40x** (below 1.5x threshold). German wins on raw CE, but this is misleading: German has the largest corpus (149K tokens, 2x Latin), producing the tightest LM (discrimination gap 6.44 bits vs 4.45 for Latin). The optimizer maps frequent Voynich stems to frequent German function words (`ist`, `und`, `mit`, `auch`) — the same frequency-matching behavior seen across all languages. Cross-validation consistency = 0.96.

The **compression ratio** (random CE / best CE) normalizes for LM quality and tells a different story: Italian leads at 1.77x, followed by German 1.40x, Occitan 1.36x, Latin 1.32x. But Italian's high compression is inflated by its tiny corpus (11K tokens), which makes random mappings score worse.

**Critical caveat:** The **sanity check failed** (only 4% recovery accuracy on a known cipher). The optimizer achieves lower CE than the true mapping on the test cipher, meaning it exploits character frequency patterns without recovering actual substitutions. All four languages achieve compression ratios in the 1.3–1.8x range — consistent with frequency matching, not decryption (genuine decryption would produce 3–5x compression).

**Bottom line:** The MDL decoder **cannot discriminate between languages** because it is not actually decrypting — it finds frequency-optimal mappings that work similarly well for any language with a good enough LM. The language question remains unresolved at the MDL level.

### Cipher Validation & Integration

| Sub-step | Description | Module |
|----------|-------------|--------|
| V.1 | **Cross-approach convergence** — Compare mappings from Approaches 16 and 18 (fraction of stems mapped to the same target). | `phases/cipher_validate.py` |
| V.2 | **Prior phase convergence** — Cross-check decoded stems against illustration IDs (Phase 6), verb positions (Phase 7/9), noun clusters (Phase 7/8). | `phases/cipher_validate.py` |
| V.3 | **Seeded decoding** — Initialize Approach 18's MCMC from Approach 16's mapping; measure improvement. | `phases/cipher_validate.py` |
| V.4 | **Combined assessment** — Fisher combined probability across all evidence, confidence level assignment. | `phases/cipher_validate.py` |

**Result:** Overall gate **FAILED**, confidence = **low**. The two approaches agree on only 1% of stem mappings (1/100). Zero prior-phase convergence (0/3 checks passed — no decoded stems match illustration plant IDs, verb patterns, or noun clusters). Seeded decode improves 1.18x (modest). Fisher combined p = 0.90 (no statistical significance).

**Verdict:** `weak_evidence_single_approach_only`. When tested against all four candidate languages (Latin, Occitan, Italian, German), the MDL decoder ranks German first on raw CE — breaking the expected Romance-language pattern. But this reflects corpus size advantage, not linguistic affinity. The compression ratio ranking (Italian > German > Occitan > Latin) is similarly uninformative, driven by corpus size effects. The sanity check failure, zero cross-approach agreement, and zero prior-phase convergence all indicate this is frequency/structural matching, not genuine decryption. The Voynich manuscript is unlikely to be a simple stem-level substitution cipher over any of the four tested languages.

## Phase 9: Fundamental Reassessment

Eight phases. Thirty-two modules. Every structural finding replicates. Every decoding attempt fails. Phase 9 confronts this pattern by asking **why** decoding fails — testing three specific encoding models and two broader diagnostics without assuming the natural-language-cipher model.

### Step 9.2: Nomenclator / Bimodal Frequency Test (Highest Priority)

| Sub-step | Description | Module |
|----------|-------------|--------|
| 9.2a | **Single vs piecewise Zipf** — Fit single and two-segment power laws to the rank-frequency distribution, compare via AIC/BIC. | `phases/nomenclator_test.py` |
| 9.2b | **Reference bimodality** — Same fit on Latin, Occitan, Italian, German. Is Voynich uniquely bimodal? | `phases/nomenclator_test.py` |
| 9.2c | **Segment profiling** — Split vocabulary at breakpoint into high-freq (codebook) and low-freq (spelled-out) segments. Profile character types, morpheme regularity, coverage. | `phases/nomenclator_test.py` |
| 9.2d | **Differential decoding** — Character-level MDL on the low-freq segment only (~20 char types). | `phases/nomenclator_test.py` |
| Null | Markov-generated text bimodality comparison (50 trials). | `phases/nomenclator_test.py` |

**Result:** Voynich IS bimodal (delta_AIC = **-9,991**, strong preference for piecewise model). Breakpoint at rank 1,001 splits into 1,001 high-frequency types (74.4% of corpus) and 2,761 low-frequency types (25.6%). The low-frequency segment has **24 character types** — classical cryptanalysis territory. Exponent gap = 0.914 (segment 1: 0.914, segment 2: 0.000).

However, **all four reference languages are also bimodal** — Latin (delta_AIC = -34,731), Occitan (-20,051), German (-29,485), Italian (-2,981). Bimodality selectivity = **1.24x** vs Markov null. Gate: bimodality=True, selectivity=**FAIL** (1.24x < 1.5x).

**Verdict:** `bimodal_but_not_unique`. The vocabulary does split into two frequency regimes, but this is a property of natural language frequency distributions, not evidence of nomenclator encoding. The 24-character low-frequency segment is interesting but not diagnostic.

### Step 9.1: Homophonic Substitution Test

| Sub-step | Description | Module |
|----------|-------------|--------|
| 9.1a | **Vocabulary inflation** — Compare Voynich stem types vs reference languages (matched for morphological decomposition). | `phases/homophone_test.py` |
| 9.1b | **Distributional clustering** — Build PPMI+SVD embeddings for all Voynich stems, find cosine > 0.8 pairs, single-linkage cluster. | `phases/homophone_test.py` |
| 9.1c | **Merged decoding comparison** — Replace clusters with representatives, compare SA/MDL baselines. | `phases/homophone_test.py` |
| Null | Same clustering on Latin stems (how many false "homophone groups"?). | `phases/homophone_test.py` |

**Result:** Voynich has only **412 stem types** (8,652 tokens, TTR=0.048). Far from inflated — Latin has 3,543 types, Occitan 1,808, German 3,212. Inflation ratios: 0.12–0.82x (Voynich vocabulary is *smaller* than every reference). **Zero pairs** above cosine 0.8 threshold. No distributional clusters found. Vocabulary reduction: 0.0%.

Latin null: 12 clusters found, reduction ratio 0.893 — Latin shows *more* distributional merging than Voynich.

**Verdict:** `no_homophonic_signal`. The Voynich vocabulary is not inflated by homophones. If anything, it is unusually compact relative to reference languages at comparable corpus sizes.

### Step 9.3: Position-Dependent Encoding Test

| Sub-step | Description | Module |
|----------|-------------|--------|
| 9.3a | **Position-split bigrams** — Split tokens by position within lines (initial/medial/final thirds), build word transition matrices, compute pairwise JSD. | `phases/position_dependent.py` |
| 9.3b | **Token identity test** — For each high-frequency token, compare co-occurrence vectors at initial vs final positions. | `phases/position_dependent.py` |
| 9.3c | **Reference comparison** — Same analysis on Latin, German, Occitan, Italian. | `phases/position_dependent.py` |
| Null | Randomly shuffle token positions within each line (50 trials). | `phases/position_dependent.py` |

**Result:** Voynich positional JSD is high (mean 0.842), and 84/100 top tokens show position-dependent behavior (cosine < 0.3). But the **null shuffled Voynich** has essentially identical JSD (0.847) — the position effect comes from vocabulary sparsity, not encoding structure. Reference languages show lower JSDs (Latin 0.495, German 0.238, Occitan 0.409, Italian 0.630). Voynich/reference ratio = 1.90 (below the 2.0 gate). Position selectivity = 0.993x (no signal above shuffled baseline).

**Verdict:** `no_position_dependent_signal`. The high positional JSD is a sparsity artifact. The encoding is not polyalphabetic.

### Step 9.4: Expanded Language Comparison

| Sub-step | Description | Module |
|----------|-------------|--------|
| 9.4a | **Corpus normalization** — Subsample all corpora to 11K tokens (Italian bottleneck) using contiguous chunks. | `phases/language_comparison.py` |
| 9.4b | **Metric matrix** — 6 metrics (H2, H3, Zipf exponent, word length, TTR, bigram JSD) x 4 languages at matched size. | `phases/language_comparison.py` |
| 9.4c | **Language ranking with CIs** — Bootstrap 100 subsamples, compute composite distance to Voynich, rank with 95% CIs. | `phases/language_comparison.py` |
| 9.4d | **Occitan vs Italian head-to-head** — Per-metric comparison with bootstrap CIs. | `phases/language_comparison.py` |

**Result (ranking by composite distance to Voynich):**

| Rank | Language | Distance | 95% CI | Closest on N metrics |
|------|----------|----------|--------|---------------------|
| 1 | Italian | 3.179 | [3.173, 3.186] | 2 (H2, H3) |
| 2 | Occitan | 3.306 | [2.917, 3.456] | 1 (word length) |
| 3 | German | 3.346 | [3.190, 3.440] | 1 (bigram JSD) |
| 4 | Latin | 3.406 | [3.142, 3.827] | 2 (Zipf, TTR) |

CIs overlap for all four languages. **Separation not significant.** Occitan vs Italian head-to-head: **3–3 tie** (Italian closer on H2, H3, bigram JSD; Occitan closer on Zipf, word length, TTR).

**Verdict:** `languages_indistinguishable_at_this_sample_size`. At 11K tokens, none of the six metrics can separate the four candidate languages. The source language question remains open.

### Step 9.5: Text Typology Classification

| Sub-step | Description | Module |
|----------|-------------|--------|
| 9.5a | **Markov generation test** — Train char-level Markov (orders 1–3) on Voynich, generate 30 synthetic texts each, compare 6 metrics within 2σ. | `phases/text_typology.py` |
| 9.5b | **Text type classification** — Rule-based classifier using H2/H1 ratio, TTR, Zipf R², indicators for glossolalia, constructed, natural, and encoded natural language. | `phases/text_typology.py` |
| 9.5c | **Entropy curves** — Conditional entropy at context orders 0–6 for Voynich and all reference languages. DTW curve comparison, decay rates, asymptotic floors. | `phases/text_typology.py` |
| Null | Classify word-shuffled Voynich (should classify as random/glossolalia). | `phases/text_typology.py` |

**Markov generation results:**

| Order | Metrics within 2σ | Sufficient? |
|-------|-------------------|-------------|
| 1 | 2/6 | No |
| 2 | 4/6 | No |
| 3 | 4/6 | No |

No Markov order reproduces >=5/6 Voynich metrics — the structure requires higher-order dependencies than character-level Markov can capture.

**Classification:** H2/H1 = 0.622 (anomalously high — outside the natural language range of 0.3–0.6), Zipf R² = 0.889 (Zipfian), TTR = 0.349 (normal). All three encoded-natural indicators fire: anomalous H2/H1 + Zipfian + normal TTR. Classification: **encoded natural language** (confidence = 1.0).

**Entropy curves:**

| Order | Voynich | Latin | Occitan | Italian | German |
|-------|---------|-------|---------|---------|--------|
| H0 | 3.832 | 4.021 | 4.163 | 4.126 | 4.213 |
| H1 | 2.385 | 3.479 | 3.605 | 3.320 | 3.369 |
| H3 | 1.986 | 2.187 | 2.119 | 1.911 | 1.921 |
| H6 | **0.978** | 0.386 | 0.328 | 0.476 | 0.510 |

The Voynich entropy floor (0.978 bits/char at order 6) is **2–3x higher** than any reference language (0.33–0.51). Decay rate is shallower (-0.390 vs -0.624 to -0.691 for references). The encoding preserves more character-level redundancy than plaintext — consistent with a cipher that doesn't fully exploit the plaintext's predictability. Closest curve: German (DTW = 2.21).

**Verdict:** `classified_as_encoded_natural`. Gate **PASSED** (classification confidence >= 0.7). The text is not glossolalia, not constructed language, not Markov-generated. It encodes natural language through a mechanism that preserves morphological structure but raises the character-level entropy floor above all tested natural languages.

### Phase 9 Decision Tree Outcome

```
Day 1: Is the vocabulary bimodal?
└── YES, but so are all reference languages → not nomenclator-specific

Day 2: Are there distributional homophone groups?
└── NO — zero clusters found, vocab is actually compact

Day 3: Is the encoding position-dependent?
└── NO — positional JSD matches random shuffling

Day 4: Which language wins at matched corpus sizes?
└── NONE — all four indistinguishable (CIs overlap)

Day 5: What kind of thing is this?
└── ENCODED NATURAL LANGUAGE — Markov insufficient (4/6 metrics),
    anomalous H2/H1 ratio, entropy floor 2-3× above plaintext
```

**Bottom line:** The encoding is not homophonic, not nomenclator, not polyalphabetic, and the source language cannot be resolved with available corpus sizes. The text classifies as encoded natural language with an anomalously high entropy floor — the encoding mechanism preserves morphological and distributional structure but introduces character-level redundancy not seen in any tested plaintext. This is consistent with a cipher system that operates at a granularity between character-level and word-level substitution, or one that introduces systematic padding/expansion at the character level.

---
[← Phases 6-7](phase-06-07.md) | [Phase Index](README.md) | [Next: Phase 10 →](phase-10-hypothesis.md)
