# Phase 71: Inflectional Reverse Engineering and Root-Level Identification

[← Phase Index](README.md) · [← Progression](../progression.md)

## Motivation

Phase 70 Track 2 discovered that CVC coda consonants directly encode Latin verbal inflection: coda -s → 2nd person singular (100% of 20 paradigm observations), coda -t → 3rd person singular (82% of 11 obs), coda -n → accusative case, coda -r → passive voice. Phase 71 attempts to scale this insight to the full corpus — classifying every token's grammatical role from its coda markers, building a root dictionary from paradigm groupings, and producing grammatically-annotated passage readings.

## Verdict: MARGINAL (11/16 gates)

The coda-to-grammar mapping, while validated at the paradigm level by Phase 70, **does not scale to corpus-wide grammatical classification**. The -r coda is massively overrepresented (47% of all coda tokens), producing a 57% verbal fraction that is incompatible with any natural Latin text. The null test fails to distinguish the real coda-grammar assignment from random permutations. Section and hand profiles are highly significant, and lexical identification in selected passages genuinely concentrates vocabulary (1.91×), but grammatical templates match random passages at the same rate as selected ones.

## Track 1: Complete Inflectional Catalog (2/5 gates — PARTIAL_INFLECTIONAL)

**File:** `src/voynich/phases/inflectional_catalog.py`
**CLI:** `voynich inflect-catalog`
**Output:** `results/phase71_inflectional_catalog.json`

### Token Classification (36,238 tokens)

| Function | Count | % |
|----------|-------|---|
| VERB_PASSIVE (-r coda) | 12,692 | 35.0% |
| NOUN_ACC (-n coda) | 5,153 | 14.2% |
| VERB_2SG (-s coda) | 4,815 | 13.3% |
| UNMARKED (no coda) | 4,304 | 11.9% |
| VERB_3SG (-t coda) | 2,960 | 8.2% |
| FUNCTION/SHORT_STEM | 2,356 | 6.5% |
| Double coda (various) | 3,695 | 10.2% |
| Multi coda (3+) | 263 | 0.7% |

81.6% of tokens (29,578) have at least one coda marker. 70.7% have a single coda.

### Broad Distribution vs CI Expected

| Category | Voynich | CI Expected | Ratio |
|----------|---------|-------------|-------|
| VERBAL | 57.2% | ~15% | 3.8× |
| NOMINAL | 14.9% | ~35% | 0.4× |
| FUNCTION_STEM | 6.5% | ~30% | 0.2× |
| UNMARKED | 21.4% | ~20% | 1.1× |

### The -r Coda Problem

Coda -r accounts for **16,916 tokens** (47% of all coda tokens), which is ~3× more than any other coda. This results from two stroke types (descender + connector) both mapping to -r after the Phase 59 connector→r correction. If -r encoded only passive voice, 35% of the text would be passive constructions — implausible for any natural language, including pharmaceutical Latin with its heavy passive usage (colatur, teratur, etc.). The -r coda likely encodes a broader phonological class (passive, agent nouns, comparatives, or simply a common coda consonant).

### Per-Coda Breakdown

| Coda | Tokens | Primary Function | Dominance |
|------|--------|-----------------|-----------|
| -r | 16,916 | VERB_PASSIVE | 75% |
| -s | 6,667 | VERB_2SG | 72% |
| -n | 5,792 | NOUN_ACC | 89% |
| -t | 4,448 | VERB_3SG | 67% |

The -n coda shows the strongest single-function dominance (89% NOUN_ACC).

### Double Codas

| Cluster | Count | Note |
|---------|-------|------|
| -rr | 1,352 | Geminate, likely modifier classification issue (cf. Phase 62) |
| -tr | 410 | |
| -sr | 262 | |
| -sn | 247 | Maps to PARTICIPLE (-ns reversed) |
| -ss | 236 | Geminate |
| -st | 203 | Maps to VERB_EST |

### Section Profiles (chi² p = 6.3 × 10⁻²¹³)

| Section | n | Verbal | Nominal |
|---------|---|--------|---------|
| herbal_a | 9,449 | 53.9% | 16.4% |
| pharmaceutical | 3,542 | 52.7% | 12.9% |
| recipes | 10,092 | 57.6% | 17.9% |
| biological | 6,476 | 61.8% | 13.9% |
| astronomical | 2,860 | 60.2% | 7.7% |
| cosmological | 2,220 | 58.5% | 15.6% |

Herbal and pharmaceutical sections have slightly lower verbal fractions and higher nominal fractions, directionally consistent with recipe content (more ingredient nouns). Astronomical sections have the lowest nominal fraction (7.7%).

### Hand Profiles (chi² p = 8.9 × 10⁻²²²)

Hand 1: 50.9% verbal. Hand 2: 61.1% verbal. Hand 3: 57.6% verbal.

### Cross-Validation: 24.0%

When comparing coda-based classification (e.g., "coda -s → VERB_2SG") against `_classify_latin_ending()` on the decoded word (e.g., "decoded word ends in -s → verb"), only 24% of clean tokens agree. This low agreement suggests the coda consonants and the decoded word endings often encode different information.

### Null Validation: NOT SIGNIFICANT (p = 0.26)

500 trials randomly permuting the four coda→grammar assignments. The real mapping's chi-squared distance to CI-expected (0.479) is not significantly different from the null mean (0.517). The effect size is too small to detect with only 4! = 24 possible permutations.

### Gates

| Gate | Threshold | Result | Value |
|------|-----------|--------|-------|
| I1 | ≥5,000 VERBAL | **PASS** | 20,711 |
| I2 | ≥10,000 NOMINAL | FAIL | 5,400 |
| I3 | Null p < 0.05 | FAIL | p=0.26 |
| I4 | Section chi² p < 0.05 | **PASS** | p≈0 |
| I5 | Verbal 10–25% | FAIL | 57.2% |

---

## Track 2: Root-Level Paradigm Identification (4/5 gates — ROOTS_IDENTIFIED)

**File:** `src/voynich/phases/root_identification.py`
**CLI:** `voynich root-id`
**Output:** `results/phase71_root_identification.json`

### Paradigm Discovery

342 paradigms found (all with 3+ forms), expanded from Phase 69's 49 T1-only paradigms to cover all clean decoded vocabulary types.

Top paradigms by size:

| Root | Meaning | Forms | Note |
|------|---------|-------|------|
| radera | root | 117 | Long decoded strings grouped by 6-char prefix |
| corara | heart | 72 | |
| nedera | not/nor | 59 | |
| dicora | I say | 57 | |
| didera | of | 57 | |
| serara | serum/evening | 52 | |
| corade | heart | 49 | |

These are **not genuine Latin inflectional paradigms** — they are long multi-token decoded strings grouped by shared prefix. A real Latin paradigm for "radix" would have ~6-8 forms, not 117. The CVC decoding produces long decoded strings (one per EVA token, which encodes 2-3 syllables each), and short prefixes match huge numbers of these strings.

### Root Identification: 79.5% (272/342)

High identification rate reflects that the `_KNOWN_ROOTS` dictionary matches very short prefixes (2-3 characters). Root "co" matches "heart" and captures every decoded string beginning with "co-".

### Pharmaceutical Classification

| Category | Roots | % |
|----------|-------|---|
| FUNCTION | 135 | 39.5% |
| UNKNOWN | 70 | 20.5% |
| OTHER | 49 | 14.3% |
| BODY_PART | 38 | 11.1% |
| INGREDIENT | 35 | 10.2% |
| QUANTITY | 11 | 3.2% |
| PREPARATION | 2 | 0.6% |
| QUALITY | 2 | 0.6% |

Only **2 PREPARATION roots** — if this were genuine pharmaceutical text, preparation verbs (strain, grind, mix, cook) should be abundant. The dominance of FUNCTION words (39.5%) reflects that the most common decoded roots are short (di, ne, se, co).

### Coverage

- Paradigm coverage: 47.9% of clean corpus tokens
- Known root coverage: 41.7%

### Gates

| Gate | Threshold | Result | Value |
|------|-----------|--------|-------|
| R1 | ≥80 paradigms 3+ forms | **PASS** | 342 |
| R2 | ≥30% roots identified | **PASS** | 79.5% |
| R3 | ≥20 INGREDIENT roots | **PASS** | 35 |
| R4 | ≥5 PREPARATION roots | FAIL | 2 |
| R5 | Paradigm coverage >30% | **PASS** | 47.9% |

---

## Track 3: Grammatically-Annotated Passage Reading (5/6 gates — GRAMMATICAL_READING)

**File:** `src/voynich/phases/grammatical_reading.py`
**CLI:** `voynich gram-read`
**Output:** `results/phase71_grammatical_reading.json`

### Passage Statistics (20 passages, 15 tokens each)

- Mean grammatical coverage: **90.0%** (all 20 above 70%)
- Mean lexical identification: **83.3%** (all 20 above 50%)
- Template matches (score >0.4): **8** passages
- Pharmaceutically interpretable: **2** passages

### Null Controls

| Metric | Real | Random | Selectivity |
|--------|------|--------|-------------|
| Grammatical coverage | 90.0% | 81.7% | 1.10× |
| Lexical identification | 83.3% | 43.7% | **1.91×** |
| Template matches/passage | 1.9 | 2.0 | 0.95× |

**Grammatical coverage is near-universal** (baseline 81.7%) because 81.6% of all tokens have coda markers — the classification is not passage-specific.

**Lexical selectivity is genuine** (1.91×) — selected passages concentrate identified vocabulary compared to random windows, consistent with Phase 70 Track 4's finding (2.29×).

**Template selectivity fails** (0.95×) — the VERBAL→NOMINAL pattern appears so uniformly throughout the corpus that CI grammatical templates match random passages at the same rate. The grammatical structure is too uniform to reveal sentence-level variation.

### Example Passage (f27r, herbal_a)

```
EVA:     chy     shol    chy     daiin   chy     ...
Decoded: cor     sene    cor     din     cor     ...
Root:    co      se      co      di      co      ...
Grammar: PASSIVE UNMARKED PASSIVE ACC    PASSIVE ...
Gloss:   cor     senna   cor     din     cor     ...

Natural: cor · senna · cor · din · cor · [?verb, 3SG] · ber · cos · cos · cor · dicor · deras (r) · coras · heart (r) · dis
```

Note: "cor" (heart, a noun) is labeled VERB_PASSIVE because it ends in -r. The grammatical labels reflect coda presence, not syntactic role.

### Gates

| Gate | Threshold | Result | Value |
|------|-----------|--------|-------|
| G1 | ≥10 passages gram >70% | **PASS** | 20 |
| G2 | ≥5 passages lex >50% | **PASS** | 20 |
| G3 | ≥3 passages template >0.4 | **PASS** | 8 |
| G4 | Template sel >1.3× | FAIL | 0.95× |
| G5 | Lex sel >1.5× | **PASS** | 1.91× |
| G6 | ≥1 interpretable | **PASS** | 2 |

---

## Integration

**File:** `src/voynich/phases/phase71_integrate.py`
**CLI:** `voynich phase71-verdict` / `voynich phase71`

| Track | Verdict | Gates |
|-------|---------|-------|
| 1: Inflectional Catalog | PARTIAL_INFLECTIONAL | 2/5 |
| 2: Root Identification | ROOTS_IDENTIFIED | 4/5 |
| 3: Grammatical Reading | GRAMMATICAL_READING | 5/6 |
| **Overall** | **MARGINAL** | **11/16** |

Verdict is MARGINAL because:
- GRAMMATICAL_READING requires Track 1 null significance — failed (p=0.26)
- GRAMMATICAL_STRUCTURE requires Track 1 ≥4 gates — failed (2/5)
- INFLECTIONAL_CONFIRMED requires Track 1 ≥3 gates — failed (2/5)

---

## Key Findings

### 1. The -r coda is overrepresented (negative finding)

Coda -r accounts for 47% of all coda tokens (16,916/33,823). Two stroke types (descender + connector) both map to -r, creating a 2:1 frequency advantage over -s, -n, and -t. This produces 35% VERB_PASSIVE corpus-wide — far exceeding any natural Latin text. The connector→r correction from Phase 59 may be accurate phonologically but the resulting grammatical distribution is unnatural.

**Implication:** Coda -r likely encodes a broader class than passive voice alone — possibly including agent nouns (-or, -er), comparative adjectives (-ior), and general final-r consonants.

### 2. Grammatical templates are non-discriminating (negative finding)

The VERBAL→NOMINAL pattern (recipe_instruction and passive_instruction templates) appears in nearly every 15-token window because 57% of tokens are classified as verbal. Random passages match CI templates at 2.0 templates/passage vs 1.9 for selected passages. The grammatical classification is too uniform for sentence-level analysis.

### 3. Cross-validation is poor (negative finding)

Only 24% of clean tokens show agreement between coda-based classification (e.g., "coda -s → verb") and ending-based classification (e.g., "decoded word ends in -s → verb"). This suggests the coda markers and decoded word endings encode different information, or that the 56% decode error rate corrupts the decoded endings.

### 4. Section and hand profiles are highly significant (positive finding)

Section chi² p = 6.3 × 10⁻²¹³ and hand chi² p = 8.9 × 10⁻²²². Different manuscript sections genuinely have different coda distributions. Herbal/pharmaceutical sections show lower verbal fractions (52-54%) than astronomical/biological sections (60-62%), which is directionally correct for recipe vs non-recipe content.

### 5. Paradigm structure is real (positive finding)

342 paradigms covering 47.9% of the clean corpus, with 79.5% root identification. While the paradigms are prefix-groupings of decoded strings (not genuine Latin inflectional families), they demonstrate consistent morphological structure in the decoded output.

### 6. Lexical selectivity persists (positive finding)

T1-dense passages concentrate identified vocabulary at 1.91× vs random (consistent with Phase 70's 2.29×). The word-level signal is genuine even though the grammatical-level signal is not.

---

## What This Means for the Project

The coda-to-grammar mapping from Phase 70 Track 2 reflects real observations at the paradigm level (20 -s observations → 100% 2sg, 11 -t observations → 82% 3sg) but these small-sample associations do not generalize to a corpus-wide grammatical skeleton. The fundamental issue is that coda -r is too frequent to encode a single grammatical function — it is a phonological marker (the consonant /r/) that appears in many different grammatical contexts (passive, agent, comparative, deponent verbs, etc.).

The 56% decode error rate remains the binding constraint. Until more triples are resolved, the decoded character stream is too noisy for either word-boundary discovery (Phase 65), LLM reading (Phase 66), computational triple resolution (Phase 67-68), or grammatical analysis (Phase 71) to extract sentence-level structure.

---

## CLI Commands

```bash
voynich inflect-catalog    # Track 1: Inflectional catalog
voynich root-id            # Track 2: Root-level paradigm identification
voynich gram-read          # Track 3: Grammatically-annotated reading
voynich phase71-verdict    # Integration
voynich phase71            # Full pipeline (3.6s)
```

## Runtime

Total: **3.6 seconds** (no API calls, entirely computational).
