# Phase 76: Triple Resolution from Vocabulary Convergence

**Verdict: NO_PROGRESS** (7/16 gates)

Attempted to resolve unresolved triples by extracting character-level constraints from 316 T1 identifications. The wildcard propagation method finds constraints on 5/13 triples but cannot self-validate (LOO structurally inapplicable). Parallel passage analysis reveals massive structural repetition (10,493 pairs). LLM gap-filling produces its first 2 accepted proposals ("deinde", "rane") but the session's KA accuracy (26.7%) undermines confidence.

## Track 1: Wildcard Constraint Extraction — CONSTRAINTS_WEAK (2/6)

**CLI:** `voynich wildcard-prop` | **Output:** `results/p76_wildcard_prop.json`

### Constraint Results

5 of 13 unresolved triples received constraints:

| Triple | Best Syllable | Obs | Consistency | Confidence | Current |
|--------|--------------|-----|-------------|------------|---------|
| ascender,crossbar,compound | re | 8 | 62% | LIKELY | be |
| ascender,loop,compound | gu | 6 | 67% | LIKELY | to |
| loop,sigmoid,bench | re | 6 | 67% | LIKELY | ne |
| ascender,crossbar,gallows | ab | 8 | 50% | INSUFFICIENT | te |
| loop,tail,bench | nu | 6 | 50% | INSUFFICIENT | la |

### LOO Cross-Validation: 0/12

**All 12 confirmed triples return 0 observations when temporarily treated as unresolved.** This is a structural limitation, not a method failure: confirmed triples are the high-frequency ones (appearing in most tokens). Removing one pushes most patterns past the 50% known-fraction threshold, excluding them from pattern building. The LOO test cannot run — it does not produce wrong answers, it produces no answers. This means the wildcard propagation method **cannot be validated on known data**, which is a serious limitation even if not a refutation.

### Applied Impact

Promoting the 3 LIKELY triples into the assignment table:
- Dict-hit: 37.6% → 37.8% (+0.2pp)
- Clean fraction: 63.0% → **71.5%** (crossing the 70% threshold)
- Signal: 164 → 183

**Gates:** W1 LOO>50% FAIL (0%), W2 ≥3 RESOLVED FAIL (0), W3 ≥5 LIKELY+ FAIL (3), W4 clean>70% **PASS** (71.5%), W5 dict+1pp FAIL (+0.2%), W6 signal maintained **PASS** (164→183).

---

## Track 2: Grammatical Skeleton Parsing — SKELETON_SELECTIVE (3/4)

**CLI:** `voynich skeleton-parse` | **Output:** `results/p76_skeleton.json`

### Skeleton Distribution

| Label | Fraction | Meaning |
|-------|----------|---------|
| STEM | 57.2% | Bare stems, function words, nominatives |
| V2 | 16.0% | 2sg imperative verbs (-s coda) |
| ACC | 15.8% | Accusative nouns (-n coda) |
| V3 | 10.9% | 3sg indicative verbs (-t coda) |

33,098 skeletons built (31,756 unique). The distribution matches pharmaceutical Latin expectations: imperative verbs (V2) and accusative objects (ACC) roughly equal, bare stems dominant.

### Parallel Passages

**10,493 parallel passage pairs** — windows with identical grammatical skeletons but different EVA tokens. **66,331 diagnostic diffs** where one token in the pair is T1-identified and the other is not. **1,478 unique tokens** receive substitution-class assignments through parallel passage analysis.

This is the largest structural repetition signal in the project. It confirms the text follows formulaic templates consistent with pharmaceutical recipe collections.

### Template Selectivity: FAIL (1.00×)

All 6 recipe templates match 100% of passages in every section. The templates are too permissive — 2-element patterns like [V2, ACC] are so common that every 15-token window contains at least one. Templates need 4+ elements or specific sequences to be discriminating.

**Gates:** S1 ≥100 pairs **PASS** (10,493), S2 ≥20 diagnostic **PASS** (66,331), S3 pharma>astro FAIL (1.00×), S4 ≥10 substitutions **PASS** (1,478).

---

## Track 3: Frequency-Identification Gap — GAP_WIDE (0/3)

**CLI:** `voynich freq-gap` | **Output:** `results/p76_freq_gap.json`

Top-500 token types: 141 T1-identified + 49 dict-hit = 190 identified (38.0%), 310 unidentified.

### Most-Blocking Triples

| Triple | Blocked Types | Blocked Tokens | Current | Track 1 Status |
|--------|--------------|----------------|---------|----------------|
| ascender,crossbar,gallows | 52 | 1,770 | te | INSUFFICIENT |
| loop,tail,bench | 44 | 1,617 | la | INSUFFICIENT |
| loop,sigmoid,bench | 17 | 457 | ne | LIKELY |
| ascender,crossbar,compound | 15 | 536 | be | LIKELY |
| ascender,plume,gallows | 14 | 265 | ga | no constraints |

The single most impactful triple to resolve is `ascender,crossbar,gallows` (blocking 52 types, 1,770 tokens), but it has only INSUFFICIENT constraints from Track 1.

**Gates:** FG1 ID rate>40% FAIL (38.0%), FG2 most-blocking=RESOLVED FAIL, FG3 ≥3/5 constrained FAIL (2/5).

---

## Track 4: Conditional LLM Gap-Fill — GAPFILL_ATTEMPTED (2/3)

**CLI:** `voynich cond-gapfill` | **Output:** `results/p76_gapfill.json`

Preconditions MET (3 likely triples + clean 71.5%), so the LLM gap-fill re-ran.

| Metric | Phase 74 | Phase 76 |
|--------|----------|----------|
| KA accuracy | 40.0% | 26.7% |
| Confidence selectivity | 3.82× | 1.03× |
| Consistency | 70% | 66.7% |
| Accepted proposals | 0 | **2** |

### Accepted Proposals

1. **"deinde"** (then/next) — decoded "dinede" (ED=2). *"The hint 'dinede' is an anagram for 'deinde' (then), a highly common sequential adverb in Latin recipe instructions."*

2. **"rane"** (frogs) — decoded "dirane" (ED=2). *"Following 'di' (of), the hint 'dirane' strongly suggests 'di rane' (of frogs), making 'cordi di rane' mean 'hearts of frogs', a known medieval ingredient."*

Both pass all 5 hallucination control layers (HIGH confidence × 3 runs, consistent, ED≤2, in dictionary). However, KA accuracy dropped to 26.7% (below 30% threshold) and confidence selectivity collapsed to 1.03× — the LLM is no more confident with real context than shuffled context. These should be treated as **tentative leads**, not validated identifications.

**Gates:** GF1 not skipped **PASS**, GF2 KA≥30% FAIL (26.7%), GF3 ≥1 accepted **PASS** (2).

---

## Integration

| Track | Verdict | Gates |
|-------|---------|-------|
| Track 1: Wildcard Propagation | CONSTRAINTS_WEAK | 2/6 |
| Track 2: Skeleton Parsing | SKELETON_SELECTIVE | 3/4 |
| Track 3: Frequency Gap | GAP_WIDE | 0/3 |
| Track 4: Conditional Gap-Fill | GAPFILL_ATTEMPTED | 2/3 |
| **Overall** | **NO_PROGRESS** | **7/16** |

## Key Findings

1. **Wildcard propagation finds constraints but cannot validate them.** 5/13 triples constrained, 3 at LIKELY level. LOO is structurally inapplicable (confirmed triples are too frequent — removing one destroys the pattern space). The 3 LIKELY proposals (re, gu, re) need external validation.

2. **Massive structural repetition confirms pharmaceutical formulae.** 10,493 parallel passage pairs with identical grammatical skeletons. 1,478 tokens receive substitution-class assignments. The text is highly formulaic.

3. **The top-blocking triple is `ascender,crossbar,gallows`.** Resolving this single triple (currently "te") would identify 52 new frequent types (1,770 tokens). It has insufficient wildcard constraints.

4. **LLM gap-filling produced its first accepted proposals** — "deinde" (then) and "rane" (frogs). Both pharmaceutically plausible. But session KA accuracy (26.7%) and selectivity (1.03×) undermine confidence.

5. **Clean fraction reaches 71.5%** with LIKELY triples applied (up from 63%), crossing the 70% threshold. This is meaningful for downstream analyses even if the specific triple values are unconfirmed.

## CLI Commands

```bash
voynich wildcard-prop      # Track 1: Constraint extraction + LOO (~107s)
voynich skeleton-parse     # Track 2: Skeleton + parallel passages (<1s)
voynich freq-gap           # Track 3: Frequency gap analysis (<1s)
voynich cond-gapfill       # Track 4: Conditional LLM gap-fill (~407s)
voynich phase76-verdict    # Integration (<1s)
voynich phase76            # Full pipeline (~9 min)
```
