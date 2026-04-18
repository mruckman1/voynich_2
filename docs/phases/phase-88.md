---
name: Phase 88 Greshko Generalized Naibbe Results
description: Key finding from Reviewer Response v3 — entropy shift is NOT a specific-enough diagnostic; cross-boundary MI and freq-connectivity are. Informs how to frame the paper's tachygraphic-uniqueness claim.
type: project
originSessionId: db449dd9-5cbc-4e45-b1ae-c05650130fb7
---
**Fact**: Phase 88 implements Greshko's generalized Naibbe (bigram respacing + 6 tables at 5:2:2:2:1:1 + slot-grammar affixes) and tests against three paper diagnostics. Result: 1 of 3 CONFIRMS Greshko.

- **Entropy shift cosine** (Sec 4.2): generalized Naibbe low-H1 mean = +0.983 (full) / +0.982 (B); real Naibbe ciphertexts ~0.999. CONFIRMS Greshko — Section 4.2 "no configuration produces positive cosine" must be revised. Entropy shift is NOT a specific-enough discriminator.
- **Cross-boundary MI ratio** (Sec 4.4): Voynich 1.448, tachygraphic 1.284, Naibbe low-H1 = 1.002, real Naibbe 1.005–1.009. INSUFFICIENT — paper claim holds.
- **Freq-connectivity ρ** (Sec 5.1, Timm & Schinner): Voynich 0.615, Naibbe low-H1 = 0.235, real Naibbe 0.378–0.414. INSUFFICIENT — paper claim holds.

**Why**: Greshko's 2026-04 email correspondence challenged the paper's Section 4.2 cosine-only argument. The three-diagnostic extension (entropy + MI + freq-conn) shows his critique is valid on the entropy axis only; the token-adjacency diagnostics remain tachygraphic-specific. Greshko himself predicted this in his follow-up: "differences in our two approaches would be revealed in the relationships between adjacent tokens."

**How to apply**:
- When discussing the tachygraphic model's distinctiveness in paper context, lead with cross-boundary MI (1.45×) and freq-connectivity (ρ=0.62), not entropy shift alone.
- The entropy shift cosine (0.820) is a weak discriminator — many verbose-substitution ciphers reproduce it. Don't cite it as uniqueness evidence without qualification.
- Natural follow-on: Phase 89 = focused adjacent-token analysis. If a Naibbe variant can also hit 1.45× MI and 0.6 ρ, paper's tachygraphy case falls; if not, tighter discriminator to lead with in revision.
- Data/code: `data/reference/greshko/{nathist_book16,divcom_output_ciphertext,nathist_output_ciphertext}.txt`; phase = `voynich phase88`; outputs = `results/p88_naibbe_generalized.json` + `results/p88_integrate.json`.
