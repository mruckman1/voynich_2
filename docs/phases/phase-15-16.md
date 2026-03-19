# Phases 15-16: Feature Refinement & Modifier Detection

**Phase 15:** 35.4% dict-hit (2.55× selectivity)
**Phase 16:** 43.6% full corpus (3.38× on subsample)

[← Phase 14](phase-14-features.md) | [Phase Index](README.md) | [Next: Phase 17 →](phase-17-honesty.md)

---

## Phase 15: Feature Model Refinement

Phase 15 attacks three addressable weaknesses in the Phase 14 result: (1) articulatory inconsistency (V12 FAIL at 30.8%), (2) underdetermined search (4% recovery despite 66.3% calibration ceiling), and (3) dictionary gaps (19.4% vs 66.3% ceiling). Three independent improvements are developed, then combined via ablation study.

| Step | Description | Module |
|------|-------------|--------|
| 15.1 | Medieval Latin dictionary expansion: 26 spelling variation rules (ae→e, vowel interchange, voicing, gemination/degemination, h-loss); pharmaceutical vocabulary (6 domains, 78 terms); Latin inflectional forms (5 noun declensions, 4 verb conjugations, 3 adjective types); near-miss catalog (365 near-misses, 80% insertion category); expanded dict 6,180 → 131K words; selectivity ratio 0.97 | `dict_expansion.py` |
| 15.2 | Articulatory consistency scoring: AC metric = mean onset consistency × mean nucleus consistency; baseline AC = 58.7%; delta grid search (0.0–0.5 AC bonus in beam search scoring); hard articulatory constraints (restrict onset domains by place class); per-onset coordinate descent (fix all but one onset group, exhaustively enumerate); best approach: per-onset descent (28.2% dict_hit, AC = 66.7%) | `articulatory_csp.py` |
| 15.3 | Iterative re-solving with confirmed hits: extract 72 high-confidence dictionary hits as hard CSP constraints; 16/25 triples initially constrained → 18/25 after iteration; split-variable approach (fixed triples excluded from beam search to avoid all-different conflicts); converges at iteration 1 (30.6% dict_hit) | `iterative_hits.py` |
| 15.4 | Combined optimization: 2³ ablation study across dict expansion × AC scoring × hit constraints; dict expansion alone = 35.4% (+16.0%), AC alone = 27.7% (+8.2%), hits alone = 19.4% (−0.1%); no positive synergy (−8.1%); best config: dict expansion only; combined iterative pipeline confirms 35.4% at 2.55× | `combined_refine.py` |
| 15.5 | Decoded text analysis: phrase detection (0 multi-word phrases — decoded tokens are long concatenated syllables, not word-segmented); section readability (herbal_a 35.8%, pharmaceutical 22.6%); vocabulary catalog (3/6 domains: `cola`, `bene`, `ad`/`de`/`in`); prior claims comparison (0/5 matches) | `text_analysis.py` |
| 15.6 | Full V1–V14 validation battery: 11/14 PASS; V12 articulatory consistency 63.5% (PASS); V13 phrase selectivity 0.0× (FAIL — needs word segmentation); V14 domain coverage 3/6 (PASS); progression tracking Phase 11 → 14 → 15 | `phase15_validate.py` |

### Phase 15 Ablation Table

| Config | Dict Expansion | AC Scoring | Hit Constraints | Dict Hit | Selectivity | AC |
|--------|:-:|:-:|:-:|--------|-------------|------|
| baseline | | | | 19.4% | 2.75× | 0.587 |
| **dict** | **x** | | | **35.4%** | **2.61×** | **0.587** |
| ac | | x | | 27.7% | 3.95× | 0.554 |
| hits | | | x | 19.4% | 2.38× | 0.698 |
| dict+ac | x | x | | 31.3% | 2.40× | 0.554 |
| dict+hits | x | | x | 35.4% | 2.70× | 0.587 |
| ac+hits | | x | x | 19.6% | 2.42× | 0.651 |
| dict+ac+hits | x | x | x | 35.4% | 2.70× | 0.635 |

### Phase 15 Key Results

| Step | Metric | Value | Gate |
|------|--------|-------|------|
| 15.1 Dictionary | Expanded dict size | 6,180 → 131,366 | — |
| 15.1 Dictionary | Dict hit (expanded) | **34.9%** (+15.5%) | PASS |
| 15.1 Dictionary | Selectivity ratio | 0.97 (≥ 0.9 gate) | PASS |
| 15.2 AC Scoring | Baseline AC | 58.7% | — |
| 15.2 AC Scoring | Best AC (per-onset descent) | 66.7% | — |
| 15.2 AC Scoring | Best dict_hit (hard constraints) | 27.7% (3.95×) | PASS |
| 15.3 Iterative | Triples constrained | 16 → 18 / 25 | — |
| 15.3 Iterative | Dict hit after iteration | 30.6% | PASS |
| 15.4 Combined | Best config | dict expansion only | — |
| 15.4 Combined | Best dict_hit | **35.4%** (2.55×) | PASS |
| 15.4 Combined | Synergy | −8.1% (no synergy) | — |
| 15.5 Text | Domains with hits | 3/6 | — |
| 15.5 Text | Herbal A section hit rate | 35.8% | — |
| 15.6 Validate | V1–V14 battery | **11/14** PASS | PASS |
| 15.6 Validate | V12 (AC) | 63.5% (≥ 50%) | PASS |
| 15.6 Validate | V14 (domain coverage) | 3/6 (≥ 3) | PASS |
| 15.6 Validate | Progression | 11.1% → 19.4% → **35.4%** | — |

### Phase 15 Findings Summary

Phase 15 nearly doubles the Phase 14 dict_hit rate (19.4% → 35.4%) primarily through dictionary expansion — generating medieval Latin spelling variants and pharmaceutical vocabulary inflections. The key insight is that the Phase 14 phoneme assignment was already finding real Latin words (`sene`, `radi`, `cone`, `sera`) that weren't in the classical Latin reference dictionary due to medieval spelling conventions (ae→e simplification, vowel interchange) and missing inflected forms.

Articulatory consistency improves substantially (30.8% → 63.5%) through per-onset coordinate descent, confirming that the stroke-to-phoneme mapping is becoming more typologically plausible: triples sharing the same `first_stroke` increasingly map to consonants from the same place of articulation (onset consistency 88.3%), and triples sharing the same `last_stroke` map to similar vowels (nucleus consistency 71.9%).

The 2³ ablation study provides a clean decomposition: dictionary expansion alone accounts for the full +16% improvement, while articulatory constraints improve AC but actually reduce dict_hit when combined with expansion (31.3% vs 35.4%). Hit-based iterative re-solving offers no improvement over the expanded-dictionary baseline. The lack of synergy suggests the three interventions compete rather than cooperate — AC constraints restrict the search space in ways that exclude the dict-expansion-optimal assignment.

Decoded text shows recognizable Latin morpheme patterns across sections: `sene-` (senecio/senex), `radi-` (radix), `cone-` (confer/coquere), `sera-` (series), with herbal_a achieving 35.8% dict_hit and pharmaceutical 22.6%. Three of six pharmaceutical vocabulary domains show hits: verbs (`cola`), qualities (`bene`), and function words (`ad`, `de`, `in`). Phrase detection fails because the decoding produces concatenated syllable strings rather than word-segmented output — a known limitation of the syllabary model that word boundary detection could address in a future phase.

## Phase 16: Modifier Detection and Syllable Correction

Phase 16 tests the hypothesis that some EVA characters are **modifiers** — glyphs that alter adjacent syllables rather than producing their own, analogous to Devanagari virama, Arabic shadda, or Thai mai tho. The feature model (Phases 14–15) assigns each EVA character an independent CV syllable, producing ~3.5 syllables per token. Latin medical words average ~2.5 syllables. If modifier characters can be identified and handled correctly, the syllable count should drop into alignment and dictionary hit rate should improve.

### Five Independent Approaches

| Step | Approach | Method | Gate | Result |
|------|----------|--------|------|--------|
| 16.1 (B) | Standalone | Never-solo frequency, positional entropy, adjacency entropy | ≥ 5 candidates | **PASS** — 7 candidates |
| 16.2 (D) | Anomaly | Zipf residuals, obligatory co-occurrence, length correlation | ≥ 3 chars | **PASS** — 30 candidates |
| 16.3 (A) | Distribution | KS test: modifier subsets vs Latin syllable-count distribution | KS < 0.15, mean 2.0–3.0 | **FAIL** — best mean 3.35 |
| 16.4 (E) | Minimal Pairs | Token pairs differing by 1 char; dict-hit preservation | ≥ 5 helpful removals | **PASS** — 2,509 helpful |
| 16.5 (C) | Localization | Padding ratio in decoded dictionary hits | ≥ 3 chars with ratio ≥ 0.6 | **PASS** — 11 candidates |

### Convergent Classification

Characters classified by agreement across the 5 approaches (≥ 3 → MODIFIER, 2 → AMBIGUOUS, ≤ 1 → SYLLABIC):

- **15 MODIFIER** characters identified
- **11 SYLLABIC** characters confirmed
- **18 AMBIGUOUS** characters (2-approach agreement)

### Re-decode Strategies

| Strategy | Description | dict_hit | Selectivity | Mean syl/token |
|----------|-------------|----------|-------------|----------------|
| Baseline (Phase 15) | No modifier handling | 35.4% | 2.55× | ~3.5 |
| R1 Strip | Skip modifier chars before triple mapping | 47.2% | 3.11× | 2.63 |
| R2 Alter | Apply modifier-type-specific rules (vowel_changer, geminator, nasalizer, cluster, silent) | 47.2% | 3.11× | 2.63 |
| **R3 Combined** | Per-token: try alteration → stripping → original | **51.6%** | **3.40×** | **2.63** |

### Phase 16 Key Results

| Metric | Value |
|--------|-------|
| dict_hit improvement | 35.4% → **51.6%** (+16.2%) |
| Selectivity | **3.40×** (vs 2.55× Phase 15) |
| Mean syllables/token | **2.63** (target ~2.5, was ~3.5) |
| Modifier chars identified | 15 (≥ 3-approach agreement) |
| Progression | 11.1% → 19.4% → 35.4% → **51.6%** |

### Phase 16 Findings Summary

Phase 16 confirms the modifier hypothesis: 15 EVA characters function as modifiers rather than independent syllable-bearing glyphs. Removing or transforming these characters during decoding reduces the mean syllables per token from ~3.5 to 2.63 — closely matching the Latin target of ~2.5 — and raises the dictionary hit rate from 35.4% to 51.6% with 3.40× selectivity over random baseline.

The critical architectural insight is that modifier classification must operate at the **EVA character level**, not the triple level. Multiple EVA chars share the same stroke triple (e.g., `d`, `i`, `m` all map to `vertical,vertical,minim`), but `d` appears as a standalone token while `i` and `m` never do. The `decode_token_modifier_aware()` function in `corpus.py` handles this by filtering modifier characters **before** the triple mapping step, allowing characters with identical stroke triples to have different syllabic/modifier roles.

The R3 combined strategy outperforms both pure stripping (R1) and pure alteration (R2) by trying alteration rules first (which may preserve more phonetic information) and falling back to stripping only when alteration doesn't produce a dictionary hit. This +4.4% gap between R3 and R1/R2 suggests that some modifier characters genuinely alter rather than silence the adjacent syllable.

**Phrase detection caveat**: Despite 51.6% dict_hit (53% in herbal_a), re-running the Phase 15.5 phrase detector with modifier-aware decoding finds **zero Latin pharmaceutical phrases** — 0/30 keywords (`recipe`, `aqua`, `folia`, `radix`, `cum`, `et`, `in`, `ad`, etc.) appear anywhere in the decoded output. The high dict_hit is driven by short decoded strings (`di`, `cone`, `se`, `ne`, `de`, `ce`) colliding with the 131K-word expanded dictionary, not by producing recognizable Latin words. The modifier correction fixes syllable count (3.5 → 2.63) but the underlying phoneme assignment still outputs syllable fragments, not word-level Latin. The 51.6% measures dictionary collision rate of a syllable-level decoding, not genuine readability.

---
[← Phase 14](phase-14-features.md) | [Phase Index](README.md) | [Next: Phase 17 →](phase-17-honesty.md)
