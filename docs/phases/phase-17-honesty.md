# Phase 17: Honesty Diagnostics

**Verdict:** NO-GO (2/5 passed; null corpora achieve 37.6% dict-hit)

[← Phases 15-16](phase-15-16.md) | [Phase Index](README.md) | [Next: Phases 18-19 →](phase-18-19.md)

---

## Phase 17 Step 0: Honesty Diagnostics

Before proceeding with word-boundary detection or further refinement, Phase 17 Step 0 applies five independent validation tests to determine whether the Phase 16 headline result (51.6% dict_hit, 3.40× selectivity) reflects genuine Latin decoding or artifacts of dictionary expansion (17K → 131K words) and per-token cherry-picking (R3 combined strategy).

### Five Honesty Tests

| Step | Test | Method | Gate | Result |
|------|------|--------|------|--------|
| 17.0.1 | Dict Control | Score R3 decoded output against original (17K), expanded (131K), and core (7K) dictionaries | original_hit > 25% | **PASS** — 35.5% original, 4.40× selectivity |
| 17.0.2 | Keyword Presence | Check 100 expected Latin medical words against decoded output (exact + ED≤1) | n_relaxed ≥ 20 AND \|ρ\| > 0.3 | **MARGINAL** — 5 exact, 15 relaxed (ρ=−0.821) |
| 17.0.3 | Verb Decode | Decode 15 Phase 9 verb stems, compare to Latin imperatives | n_ed1 ≥ 5 AND \|ρ\| > 0.3 | **FAIL** — 1/15 at ED≤1 |
| 17.0.4 | Null Corpus | Generate 5 synthetic bigram corpora, apply same decode pipeline | null_r3_max < 25% | **FAIL** — null mean 37.6% (max 38.9%) |
| 17.0.5 | Min Words | Test specific tokens with independent evidence (rosetta plants, verbs, astronomical, high-freq) | total_matches ≥ 3 | **PASS** — 8 matches |

### Cross-Strategy Comparison (Test 1)

| Strategy | Original Dict | Expanded Dict | Core Dict |
|----------|--------------|---------------|-----------|
| R3 Combined | 35.5% | 50.1% | 3.7% |
| R1 Strip | — | — | — |
| Naive (no modifiers) | — | — | — |

The 35.5% score against the original dictionary — without the 131K expanded set — demonstrates that some signal survives dictionary reduction, passing the 25% gate with 4.40× selectivity.

### Null Corpus Control (Test 4)

| Corpus | Naive dict_hit | Expanded dict_hit | R3 dict_hit |
|--------|---------------|-------------------|-------------|
| Real Voynich | — | — | 51.6% |
| Null mean (5 corpora) | 24.6% | 33.0% | 37.6% |
| Null max | 26.3% | 34.5% | 38.9% |
| Separation | — | — | 11.7σ |

While the 11.7σ separation between real and null is statistically significant, the null floor of 37.6% is far too high. A genuine cipher should produce near-zero dict_hit when applied to random text with Voynich-like character statistics. The high null floor indicates the Phase 15 phoneme assignment and R3 cherry-picking strategy produce substantial Latin dictionary collisions on *any* structured text.

### Keyword Analysis (Test 2)

Five keywords found as exact decoded tokens: `de`, `si`, `cola`, `tere`, `bene`. An additional 10 found at edit distance ≤1. The frequency-rank correlation is strong (ρ=−0.821, p=0.023) — higher-ranked keywords appear more often — but the total of 15 relaxed matches falls below the 20-keyword gate.

### Integration Verdict

| Metric | Value |
|--------|-------|
| Tests passed | 2/5 (dict_control, minimum_words) |
| Tests failed | 3/5 (keyword_presence, verb_decode, null_corpus) |
| Overall confidence | **suspect** (score = 0.40) |
| Decision | **NO-GO** |
| Strongest evidence | Dict control (35.5% against original dict) |
| Weakest evidence | Null corpus (37.6% null R3 dict_hit) |
| Red flag | Null corpus achieves comparable dict_hit — pipeline finds Latin in structured noise |
| Progression | 11.1% → 19.4% → 35.4% → 51.6% → **NO-GO** |

### Phase 17 Step 0 Findings Summary

The honesty diagnostics reveal that the Phase 16 headline result (51.6% dict_hit) is **substantially confounded**:

1. **Dictionary expansion is the dominant driver**: The 131K expanded dictionary (medieval variants + pharmaceutical inflections) turns short decoded syllable fragments into "matches" — the core 7K dictionary scores only 3.7%.

2. **R3 cherry-picking inflates the metric**: The per-token strategy of trying alteration → stripping → original and picking whichever gets a dictionary hit is fundamentally biased toward false positives.

3. **Null corpora achieve 37.6%**: Synthetic text with Voynich-like character bigram statistics, decoded through the same pipeline, scores nearly as high as real Voynich text. The "genuine signal" is at most ~14 percentage points (51.6% − 37.6%).

4. **Some real signal exists**: The 35.5% score against the original 17K dictionary at 4.40× selectivity, combined with 5 exact keyword matches and 8 minimum viable word matches, suggests the phoneme assignment captures *something* real — but it is far from a genuine decoding.

5. **Verb decode confirms Phase 9 failure**: Only 1/15 verb candidates decode within ED≤1 of any Latin imperative, consistent with Phase 9's own failed gate (0.92× selectivity).

The NO-GO verdict means further refinement of the current approach (word boundary detection, phrase recovery) would be building on an unreliable foundation. A fundamentally different validation strategy — or a different decoding approach entirely — is needed before the pipeline can claim genuine Latin decoding.

---
[← Phases 15-16](phase-15-16.md) | [Phase Index](README.md) | [Next: Phases 18-19 →](phase-18-19.md)
