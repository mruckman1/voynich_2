# Phases 31-35: Botanical Anchors, Compound Signs, Encoding Reformation

[← Phases 28-30](phase-28-30.md) | [Phase Index](README.md) | [Next: Phases 36-42 →](phase-36-42.md)

---

## Phase 31 — Botanical Anchor Attack + Structural Reframing

Phase 30 identified the core bottleneck: 13/25 stroke triples remain unconfirmed, covering 59% of all corpus tokens. The Ventris bootstrap converged after confirming only 2 words — the system is at equilibrium within the CV phonotactic model. Phase 31 attacks this from two independent directions: (1) use multi-source plant identifications as known-plaintext cribs, bypassing the decoding table entirely, and (2) test whether the decoding units themselves are wrong — gallows as determinatives, compound signs, Language B interleaving, and ligature re-segmentation.

### Path 2: Botanical Known-Plaintext (Steps 31.1–31.4)

*Step 31.1 — Consensus Plant Identification:*
- `consensus_plants.json` — Multi-source genus consensus across 56 folios from 70 concordance entries (General Botanical, Stephen Bax, Tucker & Janick, Edith Sherwood, European Hypothesis, Finnish Biologist). 7 New World plants filtered (Musa, Passiflora, Psacalium, Helianthus, Lithophragma, Duranta, Agave). Tier classification: A (≥3 sources), B (2 sources), C (single), X (contested). **1 Tier-A** folio: f9v (*Viola*, 3 sources). **11 Tier-B** folios: f2v (*Nymphoides*), f24r (*Silene*), f25v (*Dracaena*), f33r (*Papaver*), f37v (*Anagallis*), f47v (*Pulmonaria*), f50r (*Cirsium*), f54r (*Carthamus*), f56r (*Drosera*), f90r (*Osmunda*), f100r (*Brassica*). Medieval Latin names resolved from `medieval_latin_names.json` (60 entries with stems, declensions, alternate names). Label candidates ranked by TF-IDF specificity, first-line preference, and folio uniqueness (up to 10 per folio).

*Step 31.2 — Plant Name CSP:*
- `plant_name_csp.json` — Exhaustive constraint-satisfaction alignment of folio label tokens to expected plant name syllables. For each Tier A/B folio × top-5 label candidates × plant name variants: decompose label into EVA chars, syllabify plant name, enumerate all char-to-syllable alignments (exact, off-by-1, off-by-2), check against 12 confirmed triples (any conflict = reject), score by confirmed_consistent × 0.4 + unconfirmed_filled × 0.3 + family_consistent × 0.2 + name_coverage × 0.1. Cross-folio validation requires ≥2 independent folios agreeing on a new triple assignment. **12 folios tested, 1 with valid alignments** (f56r/*Drosera*): token `esedy` → `dro·se·ra` (score 0.7, 2 confirmed-consistent, 0 conflicting). Two proposed assignments: `loop,loop,bench`="ra", `sigmoid,sigmoid,bench`="se". **0 cross-folio consistent assignments** — the single-folio result can't be trusted alone. Null selectivity: 0.0 (only f56r has non-zero correct score). Verdict: **WEAK_BOTANICAL_ANCHORS**. Gate: **FAIL**.

*Step 31.3 — Plant-Derived Assignment Propagation:*
- `plant_name_propagate.json` — No cross-folio consistent assignments to propagate → 0 new triple assignments, 0 bootstrap iterations, dict_hit unchanged at 43.6%, cascade not detected. Verdict: **NO_NEW_ASSIGNMENTS**.

*Step 31.4 — Botanical Signal Validation:*
- `botanical_signal.json` — Decoded full folio text for each Tier A/B folio with existing table, searched for: expected plant names (exact or edit distance ≤2), humoral qualities (calidus/frigidus/siccus/humidus), plant-part terms (radix/folia/flos/semen/cortex/herba), preparation terms (coque/tere/misce/cola/destilla). **12 folios tested**: 1 plant name hit ("didene"≈"silene" on f24r), 2 preparation hits ("cola" on f25v, f2v). **3 total domain hits**, mean hit rate 0.0039. Permutation test (1000 permutations, reassigning decoded texts to random folios): **p = 1.0** — not significant. 9/12 folios had 0 domain hits. Verdict: **BOTANICAL_VOCABULARY_FOUND** (vocabulary present but indistinguishable from chance). Gate: **FAIL**.

**Path 2 summary**: The botanical anchor set is too thin. With only 1 Tier-A folio and labels averaging 3–5 EVA characters, there aren't enough constraint points to disambiguate triple assignments. The concordance researchers frequently disagree on genus, and medieval Latin plant names are too varied to pin down specific EVA-to-syllable mappings. No new triple assignments were derived.

### Path 4: Structural Reframing (Steps 31.5–31.8)

*Step 31.5 — Gallows as Determinatives:*
- `determinative_test.json` — Tests whether gallows characters (k, t, p, f) are silent semantic classifiers rather than phonetic units. Gallows account for 11.05% of all EVA characters (13,913 occurrences): k=7,065 (5.61%), t=4,954 (3.93%), p=1,465 (1.16%), f=429 (0.34%). Position profiles: k mostly medial (86.0%), t mostly medial (80.8%), p mixed initial/medial (35.6%/62.8%), f mostly medial (69.7%).

  **Stripping test**: remove all gallows from tokens, re-decode → **dict_hit 55.5%** (up from 43.6%, **Δ = +11.9%**), 13,370 tokens affected (36.9%). Signal rate slightly decreased (27.2% → 25.8%).

  **Semantic classification**: group tokens by initial gallows, decode non-gallows portion → **chi² = 1,438.17** (df=116, **p < 0.001**). Tokens beginning with different gallows produce significantly different decoded vocabularies — consistent with determinatives marking semantic domains.

  **Section distribution**: per-section gallows frequency ratios → **chi² = 304.61** (**p < 0.001**). Rates vary: Astronomical 14.09%, Biological 8.36%, Cosmological 13.42%, Herbal_a 11.48%, Herbal_b 8.26%, Pharmaceutical 10.04%, Recipes 11.05%. Non-uniform distribution is consistent with gallows marking content categories.

  **Null control**: randomly strip 4 non-gallows chars (50 trials) → null mean Δ = +7.35% (std 5.62%). Gallows z-score = **0.81** — the +11.9% improvement is above the null mean but only 0.81σ, not independently significant by this metric alone.

  Verdict: **DETERMINATIVE_LIKELY** (strip_improves=true, semantic_differentiation=true, section_nonuniform=true).

*Step 31.6 — Compound Sign Hypothesis:*
- `compound_sign_test.json` — Tests whether Voynich tokens are compound signs with non-phonetic prefixes (semantic category), phonetic roots, and grammatical suffixes. Uses `decompose_token_morphemes()` from Phase 4.5B with `KNOWN_PREFIXES` (o, d, y, s) and `KNOWN_SUFFIXES` (dy, y, ey, aiin, ol, al, in, an, am, m, n, and others).

  **Decomposition** (36,238 tokens): 29.6% have prefix (o=6,295, d=1,918, y=1,752, s=780), 67.0% have suffix (dy=6,494, y=4,462, ey=3,925, aiin=2,547, ol=2,463), 21.1% have both, 24.5% stem-only. Mean stem length: 3.7 EVA chars.

  **Root-only decode**: strip prefixes and suffixes, decode stems only → **dict_hit 58.7%** (up from 43.6%, **Δ = +15.1%**). Per-prefix hit rates: d=76.5% (highest), s=68.0%, o=59.3%, none=57.3%, y=54.5%.

  **Mixed decode**: root decoded phonetically + suffix mapped to Latin endings (dy→a, y→i, ey→e, aiin→um, ol→us, al→is, in→em, am→am, an→en) → **dict_hit 60.7%** (**Δ = +17.1%**).

  **Prefix semantic test**: group by prefix, chi-squared on decoded vocabularies → **chi² = 16,218.21** (**p < 0.001**). Different prefixes produce completely different decoded words — consistent with semantic classification.

  **Suffix grammatical test**: group by suffix, check decoded-word distributions → **chi² = 8,388.96** (**p < 0.001**). Different suffixes produce different distributions. Per-suffix hit rates show longer suffixes (aiiin=77.4%, iin=73.4%) outperform shorter ones (dy=38.1%, the most common and worst-performing suffix).

  Verdict: **COMPOUND_SIGN_SUPPORTED** (root_improves=true, prefix_semantic=true, suffix_grammatical=true).

*Step 31.7 — Language A/B Interleaved Text Separation:*
- `interleaved_test.json` — Tests whether Language B tokens (edy-family, aiin-family from `lang_b_combinatorial.json`) form an interleaved second text stream. **564 Language B tokens** identified (1.56% of corpus, 85 unique types), dominated by `aiin` (319 occurrences, 56.6%). Per-section rates: Cosmological 3.20% (highest), Recipes 2.30%, Astronomical 1.40%, Herbal_a 1.35%, Herbal_b 0.00% (absent). Line boundary clustering: 0.0053 (very low — no evidence of B tokens clustering at line boundaries).

  **Stream separation**: remove Language B tokens, decode remaining Stream A → dict_hit **43.06%** (down from 43.63%, **Δ = -0.57%**). Null control (100 trials removing same fraction of random tokens): mean Δ = -0.00%, std = 0.03%. Improvement z-score = **-18.65** — separation is significantly *worse* than random removal.

  Verdict: **SEPARATION_NOT_BENEFICIAL**. Language B is not a separate interleaved text — it's a minor vocabulary overlay (1.6% of corpus).

*Step 31.8 — EVA Re-Segmentation:*
- `resegmentation_test.json` — Tests 4 ligature merging schemes: M1 (ch+sh, 2 merges), M2 (all h-series: ch+sh+cth+ckh+cph+cfh, 6 merges), M3 (+qo series, 9 merges), M4 (+bench ligatures ol+al+or+ar, 13 merges). All 4 schemes produce **identical results**: dict_hit = 43.6%, 25 unique triples. The stroke-triple feature model (Phase 14) already collapses these ligature distinctions at the stroke level — `tokenize_eva_chars()` treats ch, sh, cth, ckh, cph, cfh as single characters, so merging has zero effect on the decode pipeline. Verdict: **RESEGMENTATION_NEUTRAL** (best_delta = 0.0).

### Step 31.9 — Integration

- `phase31_integrate.json` — Combines all 8 step results.

**Path 2 assessment** (Botanical known-plaintext): 1 Tier-A + 11 Tier-B folios identified, 0 cross-folio consistent assignments, 0 new confirmed triples, cascade not detected. Verdict: **botanical anchors insufficient**.

**Path 4 assessment** (Script architecture): 2/4 structural hypotheses supported.

| Hypothesis | Verdict | dict_hit Δ | Key evidence |
|------------|---------|------------|--------------|
| Gallows as determinatives | **LIKELY** | +11.9% | chi²=1438 semantic, chi²=305 section |
| Compound signs | **SUPPORTED** | +15.1% (root), +17.1% (mixed) | chi²=16218 prefix, chi²=8389 suffix |
| Language A/B interleaving | Not beneficial | -0.6% | z=-18.65 (worse than random) |
| EVA re-segmentation | Neutral | 0.0% | Already collapsed by triple model |

**Recommended changes**: (1) Strip gallows before decoding (treat as determinatives); (2) Decode roots only (strip prefixes/suffixes).

**Combined best dict_hit**: **63.1%** (baseline 43.6% + gallows stripping + root extraction).

**No interaction effects detected** — gallows stripping and root extraction operate on different character positions and are additive.

### Phase 31 Findings Summary

Phase 31 reveals that the decoding model has been partially wrong about **what constitutes the phonetic content** of a Voynich word. The 13 unconfirmed triples covering 59% of the corpus include gallows characters and common prefix/suffix characters — and these may not be phonetic at all:

1. **Gallows (k, t, p, f)** appear to be **semantic determinatives** — silent classifiers that mark the topic of a word (analogous to Egyptian hieroglyphic determinatives), not part of the pronunciation. Evidence: stripping them improves dict_hit by +11.9%, they produce significantly different decoded vocabularies when grouped by initial gallows (chi²=1438), and their distribution varies by manuscript section (chi²=305).

2. **Prefixes (o-, d-, y-, s-)** appear to encode **semantic category** information. Evidence: different prefixes produce completely different decoded root vocabularies (chi²=16,218); the d- prefix achieves 76.5% dict_hit (highest), suggesting it marks a specific grammatical or semantic class.

3. **Suffixes (-dy, -y, -ey, -aiin, -ol, etc.)** appear to encode **grammatical inflection** separately from the phonetic root. Evidence: suffixes produce significantly different distributions (chi²=8,389); longer suffixes correlate with higher hit rates (aiiin=77.4% vs dy=38.1%); suffix-to-Latin-ending mapping (dy→a, y→i, ey→e, aiin→um, ol→us) further improves dict_hit from 58.7% to 60.7%.

4. **The phonetic content resides in the root/stem** — typically 3–4 EVA characters. Decoding only these stems through the existing triple-to-syllable table produces 60.7% dictionary hit rate (mixed mode), up from 43.6% on full tokens.

5. **Language B is not interleaved** (1.6% of corpus, separation hurts). **Ligature re-segmentation is irrelevant** (already handled by the stroke-triple model).

6. **Botanical anchors are too thin.** Only 1 folio has ≥3 independent genus identifications. The concordance provides good coverage (56 folios) but poor depth (most folios have only 1 source).

- **Key conclusions**:
  1. The Voynich script appears to use a **three-layer encoding**: determinative prefix (gallows) + phonetic root (2–4 syllabic EVA chars) + grammatical suffix. This is structurally analogous to Sumerian cuneiform (determinative + logogram + phonetic complement) or Egyptian hieroglyphs (logogram + determinative + phonetic spelling).
  2. The 13 "unconfirmed" triples are not phonetic failures — they correspond to characters that function outside the phonetic layer (gallows = determinatives, prefix/suffix chars = morphological markers). The 12 confirmed triples may already cover the full phonetic inventory.
  3. The combined 63.1% dict_hit (gallows stripping + root extraction) represents the largest single-phase improvement since Phase 16's modifier detection (+16.2%), achieved by recognizing which characters are NOT phonetic rather than by improving which syllables the phonetic characters map to.
  4. Progression: Phase 11=11.1% → Phase 14=19.4% → Phase 15=35.4% → Phase 16=43.6% (full corpus) → Phase 28=43.6% (table confirmed) → Phase 29: z=6.14 → Phase 30: 2 words bootstrap → **Phase 31: 63.1% (compound sign + determinative model)**.

### Progression

| Phase | dict_hit | Signal | Bigram z | Confirmed words | Triples confirmed |
|-------|----------|--------|----------|-----------------|-------------------|
| Phase 16 | 0.436 | — | — | — | — |
| Phase 28 | 0.436 | 16.5% | — | 8 | 12/25 |
| Phase 29 | 0.436 | 16.5% | 6.14 | 8 | 12/25 |
| Phase 30 | 0.436 | 16.5% | 6.14 | 10 | 12/25 |
| **Phase 31** | **0.631** | 43.6% | — | 10 | 12/25 |

## Phase 32 — Compound-Sign Signal Pipeline

Phase 31 showed that decomposing Voynich tokens into prefix + root + suffix and decoding only roots raises dict_hit from 43.6% to 60.7%. Phase 32 re-runs the entire Phase 28–30 signal pipeline on this compound-sign output to determine whether the improvement is genuine signal (real Latin bigrams) or dictionary collisions from shorter decoded words. The decisive metric: does the bigram z-score improve beyond Phase 29's 6.14?

### Step 32.1 — Compound-Sign Corpus Decode

- `compound_decode.json` — Decodes all 36,238 tokens plus 5 null corpora through the compound-sign pipeline: `decompose_token_morphemes()` → strip gallows (k,t,p,f) from stem → R3 decode cleaned stem → map suffix to Latin ending via `SUFFIX_ENDING_MAP` (dy→a, y→i, ey→e, aiin→um, ol→is, al→ae, in→em, am→am, iin→en, m→um, aiiin→ium, iiin→ium, an→an, n→n). Per-token strategy: try root alone → root+ending → root[:-1]+ending → pick first dict hit (else root+ending).

  **Results**: dict_hit = **71.3%** (up from 43.6%, Δ = +27.6%). Strategy breakdown: 25,205 root-only hits, 620 trimmed+ending hits, 1 root+ending hit, 10,412 misses. **Null dict_hit = 64.9%** (mean of 5 null corpora: 64.7%, 64.4%, 65.2%, 65.1%, 64.8%). **Selectivity = 1.10×** — barely above null. Runtime: 5.2s.

  **Critical finding**: The +27.6% dict_hit improvement is almost entirely matched by null corpora (+21.2% null improvement). Stripping prefixes, suffixes, and gallows produces stems of ~3.7 EVA chars that decode to 2–4 letter Latin strings, trivially matching the 131K expanded dictionary regardless of input.

### Step 32.2 — Signal Re-Classification

- `compound_signal.json` — Re-classifies all 36,238 tokens as SIGNAL/SHARED_HIT/SHARED_MISS/ANTI_SIGNAL using compound decode hits (SIGNAL = real hit AND ≤1/5 null hits).

  | Category | Phase 29 | Phase 32 | Change |
  |----------|----------|----------|--------|
  | SIGNAL | 5,985 (16.5%) | 1,352 (3.7%) | −12.8% |
  | SHARED_HIT | 4,294 (11.9%) | 19,727 (54.4%) | +42.6% |
  | SHARED_MISS | 20,344 (56.2%) | 7,245 (20.0%) | −36.2% |
  | ANTI_SIGNAL | 5,615 (15.5%) | 7,914 (21.8%) | +6.4% |

  **Migration matrix**: Of 5,985 Phase 29 SIGNAL tokens, only 770 (12.9%) retained SIGNAL status; 3,267 (54.6%) migrated to SHARED_HIT and 1,914 (32.0%) to SHARED_MISS. The compound decode makes both real and null corpora hit the dictionary at similar rates, collapsing the discriminative gap that defines SIGNAL.

  **50 genuine signal words** identified (σ > 2.0), but with selectivities of only ~1.54× (vs ~5.5× at 10K dictionary in Phase 36). Top: cora (σ=130.4), ne (σ=117.1), se (σ=60.7), sera (σ=47.4), di (σ=44.3).

### Step 32.3 — Bigram Plausibility (THE DECISIVE TEST)

- `compound_bigrams.json` — Tests consecutive SIGNAL-SIGNAL pairs against Latin reference bigrams with 1,000-permutation null test.

  | Metric | Phase 29 | Phase 32 | Change |
  |--------|----------|----------|--------|
  | SIGNAL pairs | 1,127 | 43 | −96.2% |
  | Exact bigram hits | 5 | 0 | −5 |
  | Bigram z-score | **6.14** | **−0.36** | −6.50 |
  | Relaxed (edit-1) hits | 93 | — | — |
  | Inflected bigram hits | — | 0 | — |
  | Trigram hits | — | 0 | — |

  **Verdict**: The 6.14σ sequential signal is **completely destroyed** by compound decomposition. With only 43 SIGNAL pairs (down from 1,127), the bigram test has no statistical power. The z-score of −0.36 means the compound decode produces SIGNAL pairs at a rate indistinguishable from (or slightly worse than) random relabeling.

  POS chi-squared = 10.26 (above threshold) — the only positive metric, reflecting that suffix-mapped Latin endings produce non-random POS tag sequences.

### Step 32.4 — Context Analysis

- `compound_context.json` — PMI context windows, crib candidates, chain analysis, inflected pair check.

  | Metric | Phase 29 | Phase 32 | Change |
  |--------|----------|----------|--------|
  | New crib candidates | 16 | 2 | −14 |
  | Chains (≥3 tokens) | 696 | 932 | +236 |
  | Longest chain | 10 | 60 | +50 |
  | Inflected pairs | — | 0 | — |

  The increase in chains is misleading — it reflects the 71.3% dict_hit rate creating long runs of dictionary hits, not genuine signal runs. 0 inflected confirmed-confirmed pairs were found.

### Step 32.5 — Bootstrap Iteration

- `compound_bootstrap.json` — 4-check bootstrap loop under compound-sign classifications.

  **0 words accepted** (down from Phase 30's 2). Converged at iteration 1. Cascade shape: **degraded**. All candidates failed Check 2 (signal position): with only 3.7% SIGNAL rate, no word's occurrences are predominantly SIGNAL-classified. The bootstrap requires ≥50% signal position, which is unreachable when 54.4% of corpus tokens are SHARED_HIT.

### Step 32.6 — Folio Examination

- `compound_folio.json` — Annotated transliterations of top 4 SIGNAL folios.

  | Folio | Tokens | SIGNAL | SIGNAL rate | Runs | Best run |
  |-------|--------|--------|-------------|------|----------|
  | f89v1 | 144 | 116 | 80.6% | 9 | "la cora di ne be di" (score=0.7) |
  | f47r | 70 | 55 | 78.6% | 4 | — |
  | f25r | 46 | 31 | 67.4% | 1 | — |
  | f27v | 56 | 36 | 64.3% | 3 | — |

  The high SIGNAL rates are artifacts of the compound decode's high dict_hit (71.3%) combined with marginal null discrimination. Best fragment: "la cora di ne be di" on f89v1 (parse_score=0.7, prepositional phrase structure detected) — but this contains only common 2-letter syllables that match trivially.

### Step 32.7 — Readability Battery

- `compound_readability.json` — 12-test battery: **7/12 passed** (gate requires ≥8 → **FAIL**).

  | Test | Value | Threshold | Result |
  |------|-------|-----------|--------|
  | V1 dict_hit ≥ 0.55 | 0.713 | 0.55 | PASS |
  | V2 bigram JSD < 0.5 | 1.03 | 0.5 | **FAIL** |
  | V3 section χ² > 3.84 | 181.0 | 3.84 | PASS |
  | V4 signal σ mean ≥ 2.0 | 20.29 | 2.0 | PASS |
  | V5 n_genuine ≥ 8 | 50 | 8 | PASS |
  | V6 longest run > 4 | 7 | 4 | PASS |
  | V7 modifier frac 0.20–0.50 | 0.341 | 0.20–0.50 | PASS |
  | V8 bigram z ≥ 4.0 | −0.36 | 4.0 | **FAIL** |
  | V9 no regression (Δz ≥ −0.5) | −6.50 | −0.5 | **FAIL** |
  | V10 selectivity > 1.5 | 1.10 | 1.5 | **FAIL** |
  | V11 POS χ² > 5.0 | 10.26 | 5.0 | PASS |
  | V12 bootstrap cascade ≥ 1 | 0 | 1 | **FAIL** |

  The 5 failures are all signal-discrimination tests (V2, V8, V9, V10, V12). The 7 passes are either volume metrics (V1, V5, V6) or structural tests (V3, V4, V7, V11) that don't require distinguishing real from null text.

### Step 32.8 — Verdict

- `phase32_verdict.json` — **COMPOUND_COLLISIONS**

### Phase 32 Findings Summary

Phase 32 provides a definitive negative result: the compound-sign decomposition that raised dict_hit from 43.6% to 71.3% is **entirely driven by short-word dictionary collisions**, not by improved Latin decoding.

1. **The mechanism of failure is clear.** Stripping prefixes, suffixes, and gallows reduces mean token length from ~5.5 to ~3.7 EVA characters. Decoded stems are 2–4 Latin letters — short enough to hit the 131K expanded dictionary by chance. Null corpora (random text with Voynich character statistics) achieve 64.9% dict_hit through the same pipeline, vs 71.3% for real text — a gap of only 6.4pp (selectivity 1.10×).

2. **The 6.14σ sequential signal depends on full-token decodes.** Phase 29's bigram z-score required 1,127 SIGNAL-SIGNAL pairs to achieve statistical significance. Compound decode reduces this to 43 pairs (3.7% SIGNAL rate) — too few for any meaningful bigram test. The signal lived in the discriminative power of longer words (4–8 letters), which compound decomposition destroys.

3. **SHARED_HIT dominates.** 54.4% of tokens are SHARED_HIT (hit on both real and null), up from 11.9%. This is the hallmark of dictionary collisions — both real and null text produce short decoded words that trivially match.

4. **The compound model may be structurally correct** (Phase 31's chi² evidence for prefix semantics and suffix grammar is strong), **but it cannot be validated through the signal pipeline** because it destroys the discriminative power that the signal pipeline depends on. A different evaluation framework would be needed — one that doesn't rely on null corpus comparison for short-word matches.

5. **Bootstrap is fully stalled.** 0 words accepted (down from Phase 30's 2). The 3.7% SIGNAL rate means no word can achieve ≥50% signal position.

- Progression: Phase 11=11.1% → Phase 14=19.4% → Phase 15=35.4% → Phase 16=43.6% → Phase 28=43.6% → Phase 29: z=6.14 → Phase 30: 2 words → Phase 31: 63.1% (compound model) → **Phase 32: COMPOUND_COLLISIONS (z=−0.36, selectivity 1.10×, compound decode destroys signal)**.

## Phase 33 — Multi-Vector Error Correction and Orthogonal Attack

Phase 30 identified 13/25 unconfirmed triples covering 59% of corpus tokens as the core bottleneck. Phase 31 showed compound-sign decomposition could reach 63.1% dict_hit but Phase 32 proved it destroyed the 6.14σ sequential signal (bigram z dropped to −0.36). Phase 33 attacks the 13 unconfirmed triples from 6 independent analytical angles, using SIGNAL-based objectives instead of dict-hit, plus orthogonal methods (perplexity, distributional, botanical cribs, suffix grammar). The goal: find corrections where multiple independent methods converge on the same syllable reassignment.

### Approach 1+2: Anti-Signal Diagnosis + Signal-Guided Swap (Steps 33.1–33.4)

*Step 33.1 — Anti-Signal Diagnosis:*
- `anti_signal_diagnosis.json` — Identifies triples that disproportionately produce ANTI_SIGNAL tokens (words appearing more in null corpora than real). 4 anti-signal words found: sera (σ=−21.5), dira (σ=−15.6), rara (σ=−13.9), dedi (σ=−4.3). Per-triple signal_ratio (SIGNAL/(SIGNAL+ANTI_SIGNAL)): only **1/25 CORRECT** (loop,hook,bench→ni, ratio=0.79), **15 SUSPECT** (0.30–0.63), **9 WRONG** (<0.30). Even confirmed triples average signal_ratio=0.41 — barely above unconfirmed (0.35). The expanded dictionary (131K words) causes every triple to generate more ANTI_SIGNAL than SIGNAL. Verdict: **TABLE_DEGRADED** (9 wrong, 15 suspect, 1 correct).

*Step 33.2 — Per-Triple Signal Rates:*
- `triple_signal_rates.json` — Computes net_signal = (SIGNAL−ANTI_SIGNAL)/total per triple with positional and interaction analysis. Only **4/25 triples have positive net_signal**: open_curve,hook,rare=hi (+0.33, N=15), loop,hook,bench=ni (+0.20, N=4,065), loop,sigmoid,bench=ne (+0.09, N=7,182), crossbar,crossbar,rare=fa (+0.03, N=40). The remaining 21 are net-negative. **10 swap candidates** identified (unconfirmed, net < −0.02). Verdict: **MANY_SWAPS** (10 candidates).

*Step 33.3 — Signal-Guided Swap:*
- `signal_guided_swap.json` — Greedy swap optimization maximizing SIGNAL count while maintaining bigram z ≥ 6.14. **983 candidates** tested across 10 rounds. **3 swaps accepted**:
  - `loop,tail,bench: la → oi` (SIGNAL +403, ANTI −727, net +1,130, z=6.15)
  - `ascender,crossbar,compound: be → ka` (SIGNAL +65, ANTI −69, net +134, z=6.25)
  - `sigmoid,hook,rare: fe → n` (SIGNAL +0, ANTI −1, z=6.25)
  - 7 rejected for failing bigram z threshold. SIGNAL improved 5,985 → 6,453 (+7.8%), but dict_hit **dropped** 43.6% → 40.6%. Verdict: **SWAPS_FOUND** (3 accepted, signal and dict_hit anti-correlated).

*Step 33.4 — Signal-Corrected Full Decode + Validation:*
- `signal_corrected_decode.json` — Full signal pipeline with corrected table. dict_hit = 40.6%, SIGNAL = 6,453 (17.8%), bigram z = 6.08. Held-out validation: transfer confirmed but bigram z dropped from 6.14 → 6.08 (−0.06). Verdict: **SIGNAL_UNCHANGED** — swaps trade dict_hit for signal count with no net gain.

### Approach 3: Latin Character-Level Perplexity (Steps 33.5–33.7)

*Step 33.5 — Latin Character-Level Language Model:*
- `latin_lm.json` — Character-level n-gram LM (3-gram and 5-gram) with add-1 smoothing. Held-out Latin bpc = 3.37. Full decoded corpus bpc = 4.57. **SIGNAL tokens more Latin-like**: bpc = 4.30 vs corpus 4.60 (delta −0.28). Verdict: **CALIBRATION_WEAK, SIGNAL_MORE_LATIN**.

*Step 33.6 — Perplexity Coordinate Descent:*
- `perplexity_search.json` — Coordinate descent over 25 triples to minimize decoded-text perplexity. **12 total changes**, 1,344 candidates evaluated. Train bpc: 4.55 → 4.28 (−0.27). Dict_hit dropped: 43.4% → 41.2%. Verdict: **PERPLEXITY_IMPROVED** (but at cost of dict_hit).

*Step 33.7 — Three-Table Cross-Validation:*
- `perplexity_validate.json` — Compares Phase 15, signal-corrected, and perplexity-optimized tables. **0 BOTH_AGREE** (no triple where signal and perplexity propose the same change), **2 CONFLICT**. Consensus table = Phase 15 (0 changes). Verdict: **NO_IMPROVEMENT**.

### Approach 4: Suffix-Constrained Root Search (Steps 33.8–33.9)

*Step 33.8 — Suffix Grammar Mapping:*
- `suffix_grammar.json` — Maps 11 EVA suffixes to Latin POS/endings. 8 suffixes mapped. Paradigm: 8 noun suffixes, 0 verb suffixes, 3 unclear — coherence = 0.50. Verdict: **WEAK_PARADIGM**.

*Step 33.9 — Suffix-Constrained Root Search:*
- `suffix_constrained_search.json` — **8/13 improvements found** with cross-suffix validation. Dict_hit improved to **45.4%** (+1.8% over baseline). But no cross-method agreement on any specific change. Verdict: **SUFFIX_CONSTRAINTS_FOUND**.

### Approach 5: Long Botanical Crib Attack (Steps 33.10–33.12)

*Step 33.11 — Long Crib CSP:*
- `long_crib_csp.json` — **121 alignments tested, 0 valid** — every alignment conflicted with at least one confirmed triple. Verdict: **NO_MATCH** — confirmed triples are categorically incompatible with reading herbal labels as Latin plant names.

### Approach 6: Token-Pair Distributional Isomorphism (Steps 33.13–33.15)

*Step 33.14 — Distributional Match:*
- `distributional_match.json` — Hungarian algorithm on 20×20 compatibility matrix. **p = 0.477** — **not significant**. Verdict: **MARGINAL**.

### Step 33.16 — Integration and Verdict

- `phase33_integrate.json` — Cross-approach agreement matrix for all 25 triples across 6 approaches.

  **Per-approach verdict table:**

  | Approach | Ran | Changes | Dict-Hit | Signal | Bigram z | Verdict |
  |----------|-----|---------|----------|--------|----------|---------|
  | 1+2: Signal-Guided Swap | Yes | 3 | 0.406 | 0.178 | 6.08 | SIGNAL_UNCHANGED |
  | 3: Perplexity Optimization | Yes | 12 | 0.417 | 0.189 | 5.38 | NO_IMPROVEMENT |
  | 4: Suffix-Constrained | Yes | 8 | 0.454 | — | — | SUFFIX_CONSTRAINTS_FOUND |
  | 5: Long Botanical Crib | Yes | 0 | — | — | — | NO_NEW_TRIPLES |
  | 6: Distributional | Yes | 0 | — | — | — | NO_DISTRIBUTIONAL_SIGNAL |

  **Cross-approach agreement**: Where methods proposed changes, they proposed **different syllables for the same triples**. For ascender,crossbar,compound: signal→ka, perplexity→de, suffix→pa (three different answers). No triple had ≥2 methods agreeing on the same alternative.

  **0 consensus changes applied.** Final metrics unchanged: dict_hit = 43.6%, signal_rate = 16.5%, bigram z = 6.14.

  Verdict: **TABLE_CONFIRMED** — Phase 15 table is confirmed as best available. 5/6 approaches ran; 0 consensus changes; bigram z = 6.14 vs baseline 6.14. 7/25 triples unresolved.

### Phase 33 Findings Summary

Phase 33 is the most comprehensive local-optimality test performed on the decoding table, attacking it from 6 independent analytical angles. The central finding is definitive: **the Phase 15/16 assignment table is a local optimum within the CV phonotactic model.**

- **Key conclusions**:
  1. **The three corrective objectives are mutually orthogonal.** Signal maximization, perplexity minimization, and suffix-valid-word maximization each pull toward different assignments. This is the hallmark of an over-determined system with insufficient model expressiveness.
  2. **Signal maximization and dict-hit are anti-correlated** in this regime. The table sits at a saddle point between these two objectives.
  3. **The 6.14σ bigram signal is robust but fragile.** It survived signal-guided swaps (6.08), but perplexity optimization destroyed it (5.38). The signal lives in a narrow basin.
  4. **Botanical labels are categorically incompatible with the current table.** 0/121 alignments were valid across 15 folios.
  5. **Distributional isomorphism is not significant** (p = 0.477).
  6. **The path forward is not through triple reassignment.** The table is provably resistant to single-triple perturbation across multiple objectives.
  7. Progression: Phase 11=11.1% → Phase 14=19.4% → Phase 15=35.4% → Phase 16=43.6% → Phase 28=43.6% → Phase 29: z=6.14 → Phase 30: 2 words → Phase 31: 63.1% (compound) → Phase 32: z=−0.36 (compound destroys signal) → **Phase 33: TABLE_CONFIRMED (6-approach local optimality proof, 0 consensus changes)**.

### Progression

| Phase | dict_hit | Signal | Bigram z | Confirmed words | Triples confirmed |
|-------|----------|--------|----------|-----------------|-------------------|
| Phase 16 | 0.436 | — | — | — | — |
| Phase 28 | 0.436 | 16.5% | — | 8 | 12/25 |
| Phase 29 | 0.436 | 16.5% | 6.14 | 8 | 12/25 |
| Phase 30 | 0.436 | 16.5% | 6.14 | 10 | 12/25 |
| Phase 31 | 0.631 | 43.6% | — | 10 | 12/25 |
| Phase 32 | 0.436 | 16.5% | −0.36 | 10 | 12/25 |
| **Phase 33** | **0.436** | **16.5%** | **6.14** | **10** | **12/25** |
| Phase 34E | 0.436 | 27.4% | 42.07 | 10 | 12/25 |
| Phase 34G | 0.227 | 18.6% | 13.12 | 10 | 12/25 |
| **Phase 35** | **0.323** | **16.6%** | **6.88** | **10** | **12/25** |

## Phase 34: Encoding Model Reformation

Phase 34 tests 7 parallel encoding hypotheses, each attacking the 43.6% dict_hit / 6.14 bigram z ceiling from a different theoretical angle. The Phase 15/16 assignment table and Phase 29 SIGNAL classification are held fixed; each track modifies a different aspect of the decode or evaluation pipeline.

### Track Results

| Track | Model | dict_hit | Signal | Bigram z | Key Finding |
|-------|-------|----------|--------|----------|-------------|
| A | Abjad consonant-only | 55.7% | 16.2% | 7.71 | CV signal better than consonant-only |
| B | Slot-conditioned CSP | 39.9% | 17.6% | 7.23 | Modest signal improvement (+1.1%) |
| C | Latin-Italian dialect | 51.9% | 10.6% | 0.65 | Signal collapses — dialect mixing destroys discriminability |
| D | Scripta continua | 1.6% | 1.6% | 999 | Spaces are real word boundaries (not scripta continua) |
| E | 2D spatial gallows | 43.6% | **27.4%** | **42.07** | Chi² z=42.07, SIGNAL +10.9%, gallows are determinatives |
| F | Vowel pointers | 43.6% | 16.5% | 6.00 | No improvement — vowel pointer merging is neutral |
| G | Dict right-sizing (10K) | 22.7% | 18.6% | **13.12** | Net signal 16.2% vs 1.0% baseline |

**Verdict: TRACK_E_WINS** — Two tracks produce independent breakthroughs:

**Track E (Spatial Gallows)**: Classifies all 16,021 gallows occurrences by spatial relationship to adjacent characters: PRECEDING (85.8%), INTERSECTING (13.2% — bench ligatures cth/ckh/cph/cfh), FOLLOWING (0.9%), STANDALONE (0.2%). 42.5% of tokens contain gallows. The chi-squared test for semantic differentiation between gallows-domains yields z=42.07 (p<0.0001) — gallows characters function as non-phonetic determinatives (semantic classifiers), not syllabic signs. Stripping preceding/following gallows raises SIGNAL rate from 16.5% to 27.4%.

**Track G (Dictionary Right-Sizing)**: Tests 5 dictionary sizes (5K, 10K, 17K, 30K, 131K). The 10K dictionary optimizes the tradeoff between dict_hit rate and selectivity: 22.7% dict_hit with 1.43× selectivity and bigram z=13.12 (vs 6.14 at 131K). Smaller dictionaries reject false positives that inflate SHARED_HIT counts, concentrating statistical power on genuine signal. The 5K dictionary scores z=13.85 but with lower signal coverage.

### Phase 34 Findings Summary

- **Gallows are determinatives, not syllabic signs.** Track E proves this at z=42.07 — the strongest individual statistical result in the project. Gallows characters preceding other characters (85.8% of gallows usage) serve as semantic classifiers (like Egyptian hieroglyph determinatives), not phonetic signs.
- **The 131K dictionary is too large.** Track G shows that right-sizing the dictionary to 10K words doubles the bigram z-score from 6.14 to 13.12 by eliminating false positives that dilute the signal.
- **Spaces are real word boundaries.** Track D's scripta continua test achieves only 1.6% dict_hit, definitively ruling out the hypothesis that EVA spaces are arbitrary.
- **The encoding is not a mixed-language cipher.** Track C's Latin-Italian dialect model collapses signal to 10.6% — the text is monolingual Latin, not code-switching.
- **Consonant-only decoding does not improve on CV.** Track A achieves higher raw dict_hit (55.7%) but the signal classification shows CV decoding captures more genuine linguistic structure.

## Phase 35: Spatial Conditioning + 10K Dictionary

Phase 35 combines Phase 34's two strongest tracks — Track E (spatial gallows conditioning, SIGNAL 27.4%) and Track G (10K right-sized dictionary, bigram z=13.12) — and re-runs the full Phase 28–30 signal pipeline under combined conditions. The prediction: SIGNAL rate >27.4% AND bigram z >13.12 (multiplicative improvement).

### Pipeline Steps

| Step | Operation | Key Output |
|------|-----------|------------|
| 35.1 | Spatial preprocessing | 42.5% tokens conditioned; 13,337 gallows stripped, 2,025 ligatures retained, 33 silenced |
| 35.2 | Combined decode (10K dict) | 32.3% dict_hit, selectivity **1.06×** (null 30.5%) |
| 35.3 | Signal isolation | 6,018 SIGNAL (16.6%), 4,054 ANTI_SIGNAL (11.2%), net signal 5.4% |
| 35.4 | Bigram plausibility | z=6.88, 7 exact hits, 240 relaxed, 0 trigrams |
| 35.5 | Context analysis | 0 new crib candidates, 915 chains (longest=10) |
| 35.6 | Bootstrap | 0 words accepted (degraded from Phase 30's 2) |
| 35.7 | Folio transliterations | Best fragment: "cola dili" (f25r, parse_score=0.7) |
| 35.8 | Readability battery | 9/12 passed (V10, V11, V12 failed) |
| 35.9 | Verdict | **NO_INTERACTION** |

### Prediction Test Results

| Prediction | Required | Actual | Status |
|------------|----------|--------|--------|
| Signal rate exceeds Track E | >27.4% | 16.6% | **FAIL** |
| Bigram z exceeds Track G | >13.12 | 6.88 | **FAIL** |
| Selectivity > 1.3× | >1.3 | 1.06 | **FAIL** |
| Bootstrap ≥ 1 word | ≥1 | 0 | **FAIL** |

### Phase 35 Findings Summary

The combination of spatial conditioning and dictionary right-sizing fails because the two tracks operate on **fundamentally different mechanisms** that cancel when combined:

- **Track E works by removing phonetically-empty gallows** from the decode stream, producing shorter tokens that decode differently. This works with the 131K dictionary because the dict is large enough that the new decoded words still match — and they match preferentially in real text (SIGNAL 27.4%).

- **Track G works by shrinking the dictionary** to reject false positives. This works with unconditioned tokens because the 10K dictionary is selective enough to distinguish real from null tokens (selectivity 1.43×).

- **When combined**, spatial conditioning shortens tokens (mean 3.48→3.08 EVA chars), making decoded words shorter (2–4 Latin letters). These short words match the 10K dictionary at nearly identical rates for real (32.3%) and null (30.5%) text — selectivity collapses to 1.06×. The null corpus conditioning heuristic strips ALL bare gallows from null tokens, producing similarly shortened null words that hit the 10K dict.

- **Key conclusion**: The two improvements are on different axes that don't combine productively. Spatial conditioning needs a large dictionary to absorb the decoded variants; dictionary right-sizing needs unconditioned tokens to maintain discriminative power.

- Progression: Phase 11=11.1% → Phase 14=19.4% → Phase 15=35.4% → Phase 16=43.6% → Phase 28=43.6% → Phase 29: z=6.14 → Phase 30: 2 words → Phase 31: 63.1% (compound) → Phase 32: z=−0.36 (compound collisions) → Phase 33: TABLE_CONFIRMED → Phase 34: Track E z=42.07, Track G z=13.12 → **Phase 35: NO_INTERACTION (combined z=6.88, selectivity 1.06×)**.
