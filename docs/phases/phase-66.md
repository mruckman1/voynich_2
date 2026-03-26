# Phase 66: Multi-Vector Attack with Hallucination Controls

[← Phase Index](README.md)

## Motivation

Phase 65 showed the decoded character stream doesn't segment into Latin words using character-level statistics (0/4 gates, 56% decode error rate). Phase 66 brings twelve approaches organized in four tiers — LLM pharmaceutical reading with rigorous null controls (Tier 1), context/alignment methods (Tier 2), focused corpus analysis (Tier 3), and structural analysis (Tier 4). The critical design principle: every track that uses an LLM or produces interpretive readings includes blind null controls, known-answer calibration, and anchor word verification.

## Validation Framework

All Tier 1 tracks use a five-layer validation framework:

- **V1: Blind Null Controls** — every real passage accompanied by a shuffled control (same characters, random order) and a null control (from bigram-model synthetic Voynich). The LLM receives the same prompt for all three.
- **V2: Signal-Word Calibration** — before reading Voynich, test whether the LLM can identify known signal words in real decoded streams.
- **V3: Anchor Verification** — 70 signal words are statistically validated; the LLM's readings must preserve them.
- **V4: Research-Constrained Prompt** — prompt restricts the LLM to medieval pharmaceutical vocabulary with explicit signal word constraints.
- **V5: Cross-Folio Consistency** — the same decoded sequence must be read the same way on different folios.

## Methods and Results

### Track 1: LLM Pharmaceutical Reading (Step 66.1)

Sends 20 CVC-decoded passages to Gemini 3.1 Pro via OpenRouter with pharmaceutical constraints, alongside 20 shuffled + 20 null controls. Real passages run 3× for consistency measurement. 100 API calls total.

**Calibration**: LLM recovered 67.5% of known signal words from real decoded streams — it can identify signal words that are genuinely present.

**Key finding**: Real dict_hit 78.7% vs shuffled 35.1% (2.24× ratio) — the LLM distinguishes real from shuffled. But null controls scored 65.6% (ratio only 1.20×) — text generated from the Voynich's character bigram model, with no linguistic content, fools the LLM nearly as well as real text.

| Gate | Threshold | Result |
|------|-----------|--------|
| L0: Calibration ≥ 20% | 67.5% | **PASS** |
| L1: ≥ 3 valid readings | ≥ 3 | **PASS** |
| L2: Shuffled ratio ≥ 2× | 2.24× | **PASS** |
| L3: Null ratio ≥ 2× | 1.20× | FAIL |
| L4: Cross-run consistency > 0.5 | 0.0 | FAIL |
| L5: Cross-folio consistency | 0.0 | FAIL |
| L6: Anchor preservation ≥ 70% | 48.4% | FAIL |
| L7: Valid translation | none | FAIL |

**Verdict: CONTROLS_DOMINATE (3/8)**

### Track 2: Reverse Simulation — Viterbi (Step 66.2)

Builds a 1,062-word Latin vocabulary, forward-encodes through the tachygraphic model (only 87 words encodable — 8.2%), trains a word-level bigram LM, and runs Viterbi decoding on 20 real + 20 shuffled + 20 null passages.

**Key finding**: Real passages scored slightly higher than null (-214.75 vs -214.94 log-prob) but *lower* than shuffled (-210.0). Top Viterbi readings are dominated by repetitions of "semine" — the decoder latches onto the single highest-frequency word.

| Gate | Threshold | Result |
|------|-----------|--------|
| R1: Real logprob > shuffled | -214.75 < -210.0 | FAIL |
| R2: Real logprob > null | -214.75 > -214.94 | **PASS** |
| R3: ≥ 5 passages with ≥ 3 words | ≥ 5 | **PASS** |
| R4: Coverage gap > 0.05 | 0.000 | FAIL |

**Verdict: WEAK_SIGNAL (2/4)**

### Track 3: f116v Crib Exploitation (Step 66.3)

CVC-decodes the two Voynichese tokens embedded in Latin/German text on f116v, finds closest pharmaceutical words, and compares edit distances against 100 random null tokens.

- `oror` → CVC decode `nes` (ED=0 exact match in expanded dictionary)
- `sheey` → CVC decode `serar` (ED=1 from `sera`, a confirmed signal word)
- Real mean min ED = 0.50, null mean min ED = 1.06 ± 1.07

| Gate | Threshold | Result |
|------|-----------|--------|
| F1: ≥ 1 token within ED 2 | 2 tokens (ED 0, 1) | **PASS** |
| F2: Real mean ED < null | 0.50 < 1.06 | **PASS** |

**Verdict: CRIB_SUPPORTED (2/2)** — Note: only 2 tokens, very low statistical power (z=0.53, p=0.39).

### Track 4: Illustration-Text Alignment (Step 66.4)

For 12 folios with consensus plant identifications, checks if CVC-decoded tokens contain syllable sequences from the plant's Latin name. Null: 1,000 random permutations of folio-to-plant assignments.

| Gate | Threshold | Result |
|------|-----------|--------|
| I1: Enrichment > 1.5× | 0.78× | FAIL |
| I2: z > 2.0 | -0.47 | FAIL |

**Verdict: ALIGNMENT_NOT_FOUND (0/2)**

### Track 5: CI Parallel Corpus Alignment (Step 66.5)

262 Circa Instans entries syllabified and compared against 226 Voynich folio profiles using character trigram Jaccard overlap. Null: 100 shuffled CI entry labels.

| Gate | Threshold | Result |
|------|-----------|--------|
| P1: Selectivity > 1.5× | 1.0× | FAIL |
| P2: ≥ 5 entries above 95th pctile | 13 | **PASS** |

**Verdict: CI_ALIGNMENT_MARGINAL (1/2)** — overlap is driven by corpus-wide character frequencies, not specific entry-folio pairings.

### Track 6: Fontana Structural Comparison (Step 66.6)

Purely structural comparison of Fontana's tachygraphic system with Voynich sign families.

- Fontana: 10 sign families (dominant: circle 39, vertical_stroke 18, horizontal_stroke 13)
- Voynich: 12 sign families (dominant: loop_bench 12, open_curve_bench 7, vertical_minim 7)
- Both use rotation/directional modification: Fontana has 4 families with 4+ directional tick variants; Voynich has 4 gallows from 1 ascender base differentiated by last-stroke direction
- Fontana dominant modifier type: directional_tick (61.8%)

| Gate | Threshold | Result |
|------|-----------|--------|
| FN1: \|family count diff\| ≤ 2 | diff = 2 | **PASS** |
| FN2: Rotation principle present | both systems | **PASS** |

**Verdict: FONTANA_STRUCTURALLY_SIMILAR (2/2)**

### Track 7: Language A Focus (Step 66.7)

Separate signal isolation on Language A (Currier A, 138 pages, 13,362 tokens).

| Metric | Language A | Full Corpus |
|--------|-----------|-------------|
| Dict hit | 36.0% | 29.0% |
| Signal rate | 9.2% | 8.1% |
| Pharma density | 1.0% | 0.6% |

| Gate | Threshold | Result |
|------|-----------|--------|
| LA1: Signal rate > 10% | 9.2% | FAIL |
| LA2: Pharma density > corpus mean | 1.0% > 0.6% | **PASS** |
| LA3: Dict hit > 40% | 36.0% | FAIL |

**Verdict: LANG_A_MARGINAL (1/3)** — consistent with Phase 62's finding that Language A decodes 14pp better.

### Track 8: Hand 4 Focus (Step 66.8)

Latin ending distribution, recurring multi-token sequences, and signal rate on Hand 4 (biological section, 20 pages, 6,476 tokens).

- Top bigrams: `berar berar` (29×), `serar berar` (27×), `ne serar` (24×), `ne corar` (21×)
- Top endings: `-ar` (2,067×), `-ne` (676×), `-er` (593×), `-in` (583×)

| Gate | Threshold | Result |
|------|-----------|--------|
| H41: Latin ending fraction > 15% | 39.1% | **PASS** |
| H42: Signal rate > 5% | 6.6% | **PASS** |
| H43: ≥ 10 recurring bigrams | 280 | **PASS** |

**Verdict: HAND4_STRUCTURED (3/3)** — highly repetitive bigram patterns consistent with formulaic pharmaceutical text.

### Track 9: Collocational Analysis (Step 66.9)

Decoded token co-occurrence within 5-token windows, t-score significance, and CI ingredient co-occurrence matching. Null: 100 shuffled token orderings.

| Gate | Threshold | Result |
|------|-----------|--------|
| C1: ≥ 50 significant collocations | 3,107 | **PASS** |
| C2: ≥ 5 CI matches | 139 | **PASS** |

**Verdict: COLLOCATIONS_CONFIRMED (2/2)** — but selectivity = 1.00× (shuffled null produces same CI match rate: 138.95 ± 2.48). Collocations are structurally real but not specific to real token ordering.

### Track 10: N-gram Frequency Ranking (Step 66.10)

Frequency-ranked Voynich decoded types matched against syllabified CI word types. 11 exact matches, 212 at ED ≤ 2. Notable exact: `ne`, `cor`. Notable close: `din↔in`, `ben↔bene`, `ser↔per`, `cone↔pone`.

| Gate | Threshold | Result |
|------|-----------|--------|
| N1: Spearman ρ > 0.3 | 0.098 (p=0.157) | FAIL |
| N2: ≥ 10 matched pairs at ED ≤ 2 | 212 | **PASS** |

**Verdict: FREQUENCY_MARGINAL (1/2)** — vocabulary overlap exists but frequency rankings are uncorrelated.

### Track 11: Metrical Analysis (Step 66.11)

5,376 lines across 226 pages analyzed for verse structure.

- Line length: mean 17.1 syllables, std 12.9, CV = 0.754
- Max autocorrelation: |r| = 0.397 at lag 1 (alternating long/short — layout artifact, drops to near-zero by lag 3+)

| Gate | Threshold | Result |
|------|-----------|--------|
| M1: Max autocorrelation < 0.10 | 0.397 | FAIL |
| M2: Line length std > 3.0 | 12.9 | **PASS** |

**Verdict: PROSE_LIKELY (1/2)** — high variability confirms prose; lag-1 alternation is a layout artifact.

### Track 12: Astronomical Deep Dive (Step 66.12)

Astronomical/cosmological (5,080 tokens, 32 pages) vs pharmaceutical (3,542 tokens, 30 pages).

| Metric | Astronomical | Pharmaceutical |
|--------|-------------|----------------|
| Dict hit | 16.1% | 16.9% |
| Entropy H1 | 9.50 bits | 8.93 bits |
| Coda `r` | 34.2% | 29.8% |
| Coda `n` | 11.5% | 13.7% |

| Gate | Threshold | Result |
|------|-----------|--------|
| A1: Coda chi² p < 0.05 | p < 10⁻⁶ (chi²=33.5) | **PASS** |
| A2: Dict hit diff ≥ 10pp | 0.9pp | FAIL |

**Verdict: PARTIAL_DIVERGENCE (1/2)** — same encoding, different vocabulary, same decode quality.

## Integration

| Tier | Tracks | Passed | Failed |
|------|--------|--------|--------|
| 1: LLM/Knowledge | 1, 2, 3 | 2 | 1 |
| 2: Context/Alignment | 4, 5, 6 | 1 | 2 |
| 3: Corpus Analysis | 7, 8, 9, 10 | 2 | 2 |
| 4: Structural | 11, 12 | 0 | 2 |
| **Total** | **12** | **5** | **7** |

**Reading level: CONTROLS_DOMINATE** — Track 1 calibration passed but null controls scored nearly as well as real text.

**Content level: PARTIAL_CONTENT** — structural vocabulary patterns exist but not specific to real token ordering.

## Verdict

**STRUCTURAL_INSIGHT (5/12 gates passed)**

## Key Findings

1. **Hallucination controls are essential and effective.** Track 1's null controls caught the LLM producing 65.6% dict_hit on synthetic noise — without controls, the 78.7% on real text would appear to be a breakthrough. Track 9's shuffled controls show collocational structure is a character-frequency artifact, not word-level content.

2. **The tachygraphic structural hypothesis is independently supported** (Track 6). Fontana's sign families parallel the Voynich's in count (10 vs 12), modification principle (directional rotation), and family-size distribution.

3. **The encoding is real but the decode is too noisy for content recovery.** The 56% error rate from 13 unresolved triples creates a noise floor that defeats every word-level analysis: plant name alignment (Track 4), CI parallel (Track 5), frequency ranking (Track 10), LLM reading (Track 1), and Viterbi decoding (Track 2).

4. **Structural properties are visible above the noise.** Hand 4's formulaic patterns (Track 8), section-level coda divergence (Track 12), Language A's decode advantage (Track 7), and prose structure (Track 11) all operate at the token or character level where the signal-to-noise ratio is adequate.

5. **The bottleneck remains the 13 free triples.** Resolving them requires visual sign comparison that bridges the font-to-manuscript domain gap — the same conclusion as Phases 63, 64, and 65.

## Dependency Chain

```
p66_validation.py (shared V1-V5 framework)
    │
    ├── p66_llm_reading.py ──┐
    ├── p66_reverse_sim.py ──┤ Tier 1 (independent)
    └── p66_f116v_crib.py  ──┘
    │
    ├── p66_illus_align.py ──┐
    ├── p66_parallel_align.py┤ Tier 2 (independent)
    └── p66_fontana.py ──────┘
    │
    ├── p66_lang_a.py ───────┐
    ├── p66_hand4.py ────────┤ Tier 3 (independent)
    ├── p66_collocations.py──┤
    └── p66_ngram_freq.py ───┘
    │
    ├── p66_metrical.py ─────┐
    └── p66_astro_deep.py ───┘ Tier 4 (independent)
    │
    ▼
    p66_integrate.py (verdict)
```

## CLI Commands

```bash
# Tier 1
voynich llm-reading       # Track 1: LLM pharmaceutical reading (needs OPENROUTER_API_KEY)
voynich reverse-sim       # Track 2: Reverse simulation (Viterbi)
voynich f116v-crib        # Track 3: f116v crib exploitation

# Tier 2
voynich illus-align       # Track 4: Illustration-text alignment
voynich parallel-align    # Track 5: CI parallel corpus alignment
voynich fontana-struct    # Track 6: Fontana structural comparison

# Tier 3
voynich lang-a-66         # Track 7: Language A focus
voynich hand4             # Track 8: Hand 4 focus
voynich collocations      # Track 9: Collocational analysis
voynich ngram-freq        # Track 10: N-gram frequency ranking

# Tier 4
voynich metrical          # Track 11: Metrical analysis
voynich astro-deep        # Track 12: Astronomical deep dive

# Integration
voynich phase66-verdict   # Integration and verdict
voynich phase66           # Full pipeline (all 12 tracks + verdict)
```
