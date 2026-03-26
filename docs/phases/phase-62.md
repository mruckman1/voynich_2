# Phase 62: Exhaustive Computational Analysis — Final Pre-Visual Pass

[← Back to Phase Index](README.md)

## Overview

Phase 61 confirmed the syllable-level decode is correct but word assembly remains unsolved. Phase 62 exhausts every remaining computational avenue before turning to visual comparison. Eleven investigations organized in three tiers.

**Overall Verdict: PARTIAL** — 7/11 investigations passed gates. Q1=TOKENS_ARE_SYLLABLES | Q2=MINOR_CORRECTIONS_NEEDED | Q3=SIGNIFICANT_STRUCTURE. Runtime: 39s.

## Tier 1: The Word Boundary Problem — Q1 = TOKENS_ARE_SYLLABLES (2/4 passed)

| Inv | CLI | Result | Gates | Key Finding |
|-----|-----|--------|-------|-------------|
| 1 | `t1-reverse` | **FAIL** (0/3) | ED 3.55 mean | CVC decode of T1 words averages ED 3.55 from Latin targets; 0 exact, 2 within ED≤2; syllabic ratio 0.97 (1:1 char:syllable) |
| 2 | `cross-token` | **PASS** (2/4) | 35.5% cross | Cross-boundary hits exist but z=0.23 vs null — not significant; token adjacency doesn't help word reconstruction |
| 3 | `gallows-initial` | **PASS** (2/3) | 48.8% word-final pred | Gallows predecessors are word-final 48.8%; gallows decode to function words (sigla confirmed); but concat rate 0.56× baseline |
| 4 | `decoded-bigram` | **FAIL** (1/3) | 90.5% NOT_IN_LATIN | Only 19/200 top bigrams appear in Latin; 10 cross-word vs 8 within-word (near-tie); decoded pair frequencies don't match Latin |

**Interpretation:** EVA tokens encode 2-3 syllables each (confirmed by Inv 7). Token boundaries are not word boundaries, but removing them doesn't help either. The word assembly problem is the primary remaining blocker.

## Tier 2: Classification Quality — Q2 = MINOR_CORRECTIONS_NEEDED (2/4 passed)

| Inv | CLI | Result | Gates | Key Finding |
|-----|-----|--------|-------|-------------|
| 5 | `orphaned-coda` | **FAIL** (1/3) | 1.3% orphans | Only 1.3% orphaned (down from 20.1% pre-correction); 48.7% of double codas are legal Latin; **geminates dominate illegals: rr=596, ss=185, tt=54** |
| 6 | `double-mod` | **FAIL** (1/3) | -0.25 rank corr | Same 1,631 pairs; top cluster is illegal rr; rank correlation with Latin is negative; modifier pairs don't match Latin cluster frequencies |
| 7 | `token-length` | **PASS** (2/3) | r=0.983 | Near-perfect correlation EVA→decoded length; 2.48 chars per syllabic char; **49.2% of tokens produce 6+ decoded chars** (multi-syllable) |
| 8 | `syl-entropy` | **PASS** (2/3) | H1 ratio 1.15 | H1 comparable (1.15×); H2 slightly lower (0.89×); TTR 10× higher because Voynich "syllables" are multi-syllable tokens with 6,326 types |

**Interpretation:** Coda classification mostly correct (orphans at 1.3%). The **geminate clusters rr/ss/tt** are the main issue — doubled modifiers likely have a special function (gemination? emphasis?) rather than representing two-consonant codas.

## Tier 3: Corpus-Level Structure — Q3 = SIGNIFICANT_STRUCTURE (3/3 passed)

| Inv | CLI | Result | Gates | Key Finding |
|-----|-----|--------|-------|-------------|
| 9 | `lang-ab-cvc` | **PASS** (2/3) | chi²=221.7 | CVC overlap 17.1% (vs 13.8% EVA). **Lang A: 39% dict hit vs Lang B: 25.5%** — A decodes substantially better. A-exclusive: cola, didi, dido; B-exclusive: bela, code, fa, hi, la |
| 10 | `hand-cvc` | **PASS** (2/3) | chi²=1285 | Coda distribution differs massively across hands. **Hand 4 (pharma): 70.6% Latin endings** vs Hand 3: 48.5%. Hand-exclusive signals: Hand 0=fa/hi, Hand 1=cola/didi/dido, Hand 2=code |
| 11 | `multi-entropy` | **PASS** (2/3) | similarity 0.89 | Character H1 ratio 0.82; syllable 1.15; bigram 1.05; trigram 0.95. **Entropy profile matches Latin at all four levels** — the decoded output has the right statistical shape |

**Interpretation:** A/B distinction is real and amplified under CVC. Language A (herbal_a) decodes 14pp better. Hands show genuinely different profiles. Multi-level entropy confirms the decode produces Latin-compatible statistics.

## Key Conclusions

1. **Tokens = multi-syllable units** (2-3 syllables), not single syllables and not whole words
2. **CVC decode produces Latin-compatible entropy** at all levels (overall similarity 0.89)
3. **Geminate clusters (rr/ss) are the main remaining classification issue**
4. **Language A decodes 14pp better than Language B** — herbal section is best decoded
5. **Hand 4 (pharmaceutical) has highest Latin ending rate** (70.6%)
6. **T1 identifications don't work well under CVC** — mean ED 3.55; the identification mechanism operates at a level CVC doesn't capture directly
7. **Word boundary problem confirmed as primary blocker** — computational approaches exhausted; visual comparison is the next step

## CLI Commands

```bash
# Tier 1
voynich t1-reverse         # Inv 1
voynich cross-token        # Inv 2
voynich gallows-initial    # Inv 3
voynich decoded-bigram     # Inv 4

# Tier 2
voynich orphaned-coda      # Inv 5
voynich double-mod         # Inv 6
voynich token-length       # Inv 7
voynich syl-entropy        # Inv 8

# Tier 3
voynich lang-ab-cvc        # Inv 9
voynich hand-cvc           # Inv 10
voynich multi-entropy      # Inv 11

# Integration
voynich phase62-verdict    # Combine all 11
voynich phase62            # Run full Phase 62
```

## Dependency Graph

All 11 investigations are independent — they can run in any order or in parallel. All depend on:
- `results/combined_refine.json` (Phase 15)
- `results/modifier_integrate.json` (Phase 16)

Integration (`phase62-verdict`) depends on all 11 result JSONs.

## Result Files

| File | Investigation |
|------|---------------|
| `results/phase62_t1_reverse.json` | Inv 1 |
| `results/phase62_cross_token.json` | Inv 2 |
| `results/phase62_gallows_initial.json` | Inv 3 |
| `results/phase62_decoded_bigram.json` | Inv 4 |
| `results/phase62_orphaned_coda.json` | Inv 5 |
| `results/phase62_double_modifier.json` | Inv 6 |
| `results/phase62_token_length.json` | Inv 7 |
| `results/phase62_syllable_entropy.json` | Inv 8 |
| `results/phase62_lang_ab_cvc.json` | Inv 9 |
| `results/phase62_hand_cvc.json` | Inv 10 |
| `results/phase62_multi_entropy.json` | Inv 11 |
| `results/phase62_integrate.json` | Integration |
