"""
Voynich 2: Stroke-Level Syllabary & Information-Theoretic Fingerprinting
=========================================================================
Two complementary language-agnostic approaches to the Voynich manuscript.

Usage:
    voynich                   # Show corpus summary
    voynich corpus            # Load and summarize the EVA corpus
    voynich reference         # Show reference corpus summary
    voynich strokes           # Approach 1: stroke-level syllabary analysis
    voynich fingerprint       # Approach 2: information-theoretic fingerprinting
    voynich both              # Run both approaches sequentially
    voynich nulls             # Phase 2A: null character identification
    voynich grid              # Phase 2B: syllabary grid refinement
    voynich phase2            # Run both Phase 2 analyses
    voynich degeneracy        # Phase 3D: break substitution vs syllabary degeneracy
    voynich grid-validate     # Phase 3E: validate syllabary grid
    voynich syllable-match    # Phase 3F: syllable-level language matching
    voynich validate-all      # Phase 3G: scholarly validation framework
    voynich phase3            # Run all Phase 3 workstreams
    voynich audit             # Phase 4.1: discriminant audit of Phase 3 results
    voynich section-diagnosis # Phase 4.2: section consistency diagnosis
    voynich abugida           # Phase 4.3: abugida hypothesis test
    voynich multi-language    # Phase 4.4: multi-language comparison
    voynich phase4            # Run all Phase 4 analyses
    voynich lang-a            # Phase 4.5A+C: language A isolation + qo-removal
    voynich morpheme-grid     # Phase 4.5B: morpheme grid reinterpretation
    voynich phase4-5          # Run all Phase 4.5 analyses
    voynich paradigms         # Phase 5.1: paradigm discovery
    voynich paradigm-match    # Phase 5.2: paradigm-to-language matching
    voynich stem-id           # Phase 5.3: frequency-based stem identification
    voynich phonetic          # Phase 5.4+5.5: phonetic decode and validation
    voynich phase5            # Run all Phase 5 analyses
    voynich illustration      # Phase 6.0: illustration-constrained setup
    voynich rosetta           # Phase 6 D+E: Rosetta folio selection
    voynich anchor            # Phase 6 A+B: anchor-and-propagate
    voynich compete           # Phase 6 C: competitive ID resolution
    voynich phase6-validate   # Phase 6 validation battery
    voynich phase6            # Run all Phase 6 analyses
    voynich anchor-diagnosis  # Phase 6.1B: anchor inconsistency diagnosis
    voynich encoding-diagnosis # Phase 6.1C: encoding model diagnosis
    voynich phase6-1          # Run full Phase 6.1 pipeline (TF-IDF + diagnosis)
    voynich embeddings        # Approach 8: morpheme distributional semantics
    voynich slots             # Approach 9: pharmaceutical positional slot analysis
    voynich phase7            # Run full Phase 7 (Approaches 8 + 9 + integration)
    voynich nomenclator       # Phase 9.2: bimodal frequency / nomenclator test
    voynich homophones        # Phase 9.1: homophonic substitution test
    voynich position-dep      # Phase 9.3: position-dependent encoding test
    voynich lang-compare      # Phase 9.4: expanded language comparison
    voynich typology          # Phase 9.5: text typology classification
    voynich phase9            # Run all Phase 9 analyses
    voynich entropy-curves    # Phase 10.1: token-level entropy curves
    voynich mi-decay          # Phase 10.2: mutual information decay
    voynich folio-shift       # Phase 10.3: folio-level encoding shifts
    voynich glyph-grammar     # Phase 10.4: glyph construction grammar
    voynich hypothesis        # Phase 10.5: hypothesis integration & verdict
    voynich phase10           # Run all Phase 10 analyses
    voynich csp-solve         # Phase 11.0: CSP solver sanity test
    voynich csp-decode        # Phase 11.2: multi-language CSP decoding
    voynich csp-validate      # Phase 11.3: CSP validation battery
    voynich phase11           # Run all Phase 11 analyses
    voynich csp-diagnose      # Phase 11.5.1: CSP failure diagnosis
    voynich csp-refine        # Phase 11.5.2-3: inherent vowel + relaxation sweep
    voynich verb-constrain    # Phase 11.5.4: verb-constrained CSP solving
    voynich csp-iterate       # Phase 11.5.5: iterative CSP refinement
    voynich csp-final         # Phase 11.5.6-7: final multi-language + V1-V9
    voynich phase11-5         # Run full Phase 11.5 pipeline
    voynich grid-recal        # Phase 12.1-12.2: grid recalibration from correction vectors
    voynich grid-alt          # Phase 12.4: stroke-based alternative grid construction
    voynich token-decomp      # Phase 12.5: token decomposition variant sweep
    voynich recal-csp         # Phase 12.3+12.6: recalibrated CSP + full validation
    voynich phase12           # Run full Phase 12 pipeline
    voynich error-patterns    # Phase 13.1: near-miss error pattern analysis + MI gate
    voynich null-context      # Phase 13.6: null hypothesis testing (run alongside 13.1)
    voynich extract-rules     # Phase 13.2: context-dependent rule extraction
    voynich context-csp       # Phase 13.3: context-aware CSP solver
    voynich rule-validate     # Phase 13.4: cross-validation + per-rule selectivity
    voynich context-decode    # Phase 13.5: full decoding + V1-V11 battery
    voynich phase13           # Run full Phase 13 pipeline
    voynich cell-analysis     # Phase 14.1: within-cell character distributional analysis
    voynich stroke-features   # Phase 14.2: stroke-to-phoneme feature mapping
    voynich feature-csp       # Phase 14.3: feature-level CSP solver
    voynich feature-calibrate # Phase 14.4: synthetic calibration
    voynich feature-decode    # Phase 14.5-14.6: full decode + V1-V12
    voynich subcell-split     # Phase 14.7: data-driven sub-cell splitting
    voynich phase14           # Run full Phase 14 pipeline
    voynich dict-expand       # Phase 15.1: medieval Latin dictionary expansion
    voynich artic-csp         # Phase 15.2: articulatory consistency scoring
    voynich iter-hits         # Phase 15.3: iterative re-solving with confirmed hits
    voynich combined-refine   # Phase 15.4: combined optimization + ablation
    voynich text-analysis     # Phase 15.5: decoded text analysis + phrase detection
    voynich phase15-validate  # Phase 15.6: full V1-V14 validation battery
    voynich phase15           # Run full Phase 15 pipeline
    voynich mod-standalone    # Phase 16.1: standalone modifier candidate analysis
    voynich mod-anomaly       # Phase 16.2: frequency anomaly modifier detection
    voynich mod-distrib       # Phase 16.3: syllable distribution matching
    voynich mod-pairs         # Phase 16.4: minimal pair modifier evidence
    voynich mod-localize      # Phase 16.5: hit localization padding analysis
    voynich mod-integrate     # Phase 16.6: convergent classification + re-decode
    voynich phase16           # Run full Phase 16 pipeline
    voynich triple-loo        # Phase 24.1: leave-one-out triple sensitivity
    voynich error-id          # Phase 24.2: error candidate identification
    voynich triple-swap       # Phase 24.3: exhaustive single-triple swap
    voynich bigram-val        # Phase 24.4: bigram plausibility validation
    voynich corrected-tab     # Phase 24.5: corrected table assembly
    voynich corrected-decode  # Phase 24.6: full corpus decode with corrected table
    voynich corrected-read    # Phase 24.7: readability battery
    voynich word-bound        # Phase 24.8: word boundary re-analysis
    voynich ligature-test     # Phase 24.9: EVA ligature hypothesis test
    voynich direction         # Phase 24.10: reading direction analysis
    voynich crib-search       # Phase 24.11: known-plaintext crib search
    voynich folio-deep        # Phase 24.12: single-folio deep decode
    voynich section-xfer      # Phase 24.13: section-trained transfer test
    voynich reverse-eng       # Phase 24.14: reverse-engineer from confirmed words
    voynich token-gram        # Phase 24.15: word grammar exploitation
    voynich phase24-integrate # Phase 24.16: combined verdict
    voynich phase24           # Run full Phase 24 pipeline
    voynich boustro           # Step 25.1: boustrophedon re-ordering test
    voynich f6r-exam          # Step 25.2: folio f6r manual examination
    voynich phase25-verdict   # Step 25.3: combined verdict
    voynich phase25           # Run full Phase 25 pipeline
    voynich dict-cal          # Step 34.18: dictionary right-sizing (Track G)
    voynich sigla-dict        # Step 34.1: medieval abbreviation dictionary (Track A)
    voynich abjad-csp         # Step 34.2: abjad consonant-only CSP (Track A)
    voynich sigla-decode      # Step 34.3: sigla-specific decode (Track A)
    voynich abjad-signal      # Step 34.4: abjad signal isolation (Track A)
    voynich slot-vars         # Step 34.5: slot-conditioned variable fork (Track B)
    voynich slot-csp          # Step 34.6: position-conditioned CSP solve (Track B)
    voynich slot-signal       # Step 34.7: slot-conditioned signal isolation (Track B)
    voynich mixed-lm          # Step 34.8: mixed Latin-Italian LM (Track C)
    voynich dialect-decode    # Step 34.9: dialect-conditioned decode (Track C)
    voynich dialect-signal    # Step 34.10: dialect signal isolation (Track C)
    voynich continua          # Step 34.11: space stripping + character stream (Track D)
    voynich reseg-decode      # Step 34.12: Viterbi re-segmentation (Track D)
    voynich reseg-signal      # Step 34.13: re-segmented signal isolation (Track D)
    voynich gallows-geom      # Step 34.14: gallows-bench spatial geometry (Track E)
    voynich spatial-decode    # Step 34.15: spatial-tagged decode (Track E)
    voynich vowel-ptr         # Step 34.16: vowel pointer hypothesis test (Track F)
    voynich vowel-decode      # Step 34.17: vowel-pointed decode (Track F)
    voynich phase34-integrate # Step 34.19: phase 34 integration
    voynich phase34           # Run full Phase 34 pipeline

    # Phase 35: Spatial Conditioning + 10K Dictionary
    voynich spatial-pre       # Step 35.1: spatial gallows preprocessing
    voynich comb-decode       # Step 35.2: combined spatial+10K decode
    voynich comb-signal       # Step 35.3: combined signal isolation
    voynich comb-bigram       # Step 35.4: combined bigram plausibility
    voynich comb-context      # Step 35.5: combined context analysis
    voynich comb-bootstrap    # Step 35.6: combined Ventris bootstrap
    voynich comb-folio        # Step 35.7: combined folio transliterations
    voynich comb-read         # Step 35.8: combined readability battery
    voynich phase35-verdict   # Step 35.9: Phase 35 verdict
    voynich phase35           # Run full Phase 35 pipeline

    # Phase 45: SBM Community Forensics + Distributional Re-encoding
    voynich sbm-profile       # Step 45A.1: per-community distributional profiles
    voynich sbm-position      # Step 45A.2: positional analysis
    voynich sbm-morpheme      # Step 45A.3: morphological role analysis
    voynich sbm-modifier      # Step 45A.4: modifier vs syllabic alignment
    voynich sbm-combinat      # Step 45A.5: community bigram transition matrix
    voynich sbm-factor        # Step 45A.6: C×V factorization hypothesis test
    voynich sbm-signal        # Step 45A.7: signal word decomposition by community
    voynich track-a-45        # Run all Track A steps
    voynich sbm-encode        # Step 45B.1: community-based encoding table
    voynich sbm-csp           # Step 45B.2: CSP decode with community variables
    voynich comm-signal       # Step 45B.3: signal isolation on community decode
    voynich sbm-hybrid        # Step 45B.4: hybrid stroke+community decode
    voynich sbm-landscape     # Step 45B.5: MaxSAT landscape at community granularity
    voynich track-b-45        # Run all Track B steps
    voynich triple-tiers      # Step 45C.1: three-tier confidence partition
    voynich triple-ambig      # Step 45C.2: ambiguous triple characterization
    voynich triple-lock       # Step 45C.3: canonical table assembly
    voynich triple-impact     # Step 45C.4: impact analysis
    voynich track-c-45        # Run all Track C steps
    voynich phase45-integrate # Phase 45 integration verdict
    voynich phase45           # Run full Phase 45 pipeline
"""
import sys
import time


def cmd_corpus():
    """Load and display corpus summary."""
    from voynich.core.corpus import load_corpus
    corpus = load_corpus(verbose=True)
    summary = corpus.summary()
    print("\nCorpus Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    # Show section-level token counts
    print("\nTokens per section:")
    for section in sorted(set(
        p.section for p in corpus.pages.values()
    )):
        tokens = corpus.get_tokens(section=section, paragraph_only=True)
        print(f"  {section:<20s}: {len(tokens):6d} tokens")


def cmd_strokes():
    """Run Approach 1: stroke-level syllabary analysis."""
    from voynich.analysis.strokes import run_stroke_analysis
    t0 = time.time()
    run_stroke_analysis()
    print(f"\nStroke analysis completed in {time.time() - t0:.1f}s")


def cmd_fingerprint():
    """Run Approach 2: information-theoretic fingerprinting."""
    from voynich.analysis.fingerprint import run_fingerprint_analysis
    t0 = time.time()
    run_fingerprint_analysis()
    print(f"\nFingerprint analysis completed in {time.time() - t0:.1f}s")


def cmd_reference():
    """Show reference corpus summary."""
    from voynich.core.reference import load_reference_corpus
    try:
        corpus = load_reference_corpus(verbose=True)
    except FileNotFoundError as e:
        print(f"  {e}")
        return

    print("\nReference Corpus Summary:")
    for lang, info in sorted(corpus.summary().items()):
        print(f"\n  {lang}:")
        print(f"    Texts:  {info['texts']}")
        print(f"    Tokens: {info['total_tokens']:,}")
        for name in info['files']:
            texts = [t for t in corpus.get_texts(lang) if t.name == name]
            if texts:
                print(f"      - {name}: {texts[0].token_count:,} tokens")


def cmd_nulls():
    """Run Phase 2A: null character identification."""
    from voynich.phases.nulls import run_null_analysis
    t0 = time.time()
    run_null_analysis()
    print(f"\nNull analysis completed in {time.time() - t0:.1f}s")


def cmd_grid():
    """Run Phase 2B: syllabary grid refinement."""
    from voynich.phases.grid_refine import run_grid_refinement
    t0 = time.time()
    run_grid_refinement()
    print(f"\nGrid refinement completed in {time.time() - t0:.1f}s")


def cmd_phase2():
    """Run both Phase 2 analyses sequentially."""
    cmd_nulls()
    print("\n" + "=" * 70 + "\n")
    cmd_grid()


def cmd_both():
    """Run both approaches sequentially."""
    cmd_strokes()
    print("\n" + "=" * 70 + "\n")
    cmd_fingerprint()


def cmd_degeneracy():
    """Run Phase 3D: break substitution vs syllabary degeneracy."""
    from voynich.phases.degeneracy import run_degeneracy_analysis
    t0 = time.time()
    run_degeneracy_analysis()
    print(f"\nDegeneracy analysis completed in {time.time() - t0:.1f}s")


def cmd_grid_validate():
    """Run Phase 3E: validate syllabary grid."""
    from voynich.phases.grid_validate import run_grid_validation
    t0 = time.time()
    run_grid_validation()
    print(f"\nGrid validation completed in {time.time() - t0:.1f}s")


def cmd_syllable_match():
    """Run Phase 3F: syllable-level language matching."""
    from voynich.phases.syllable_match import run_syllable_matching
    t0 = time.time()
    run_syllable_matching()
    print(f"\nSyllable matching completed in {time.time() - t0:.1f}s")


def cmd_validate_all():
    """Run Phase 3G: scholarly validation framework."""
    from voynich.phases.scholarly import run_scholarly_validation
    t0 = time.time()
    run_scholarly_validation()
    print(f"\nScholarly validation completed in {time.time() - t0:.1f}s")


def cmd_phase3():
    """Run all Phase 3 workstreams sequentially."""
    cmd_degeneracy()
    print("\n" + "=" * 70 + "\n")
    cmd_grid_validate()
    print("\n" + "=" * 70 + "\n")
    cmd_syllable_match()
    print("\n" + "=" * 70 + "\n")
    cmd_validate_all()


def cmd_audit():
    """Run Phase 4.1: discriminant audit of Phase 3 results."""
    from voynich.phases.discriminant_audit import run_discriminant_audit
    t0 = time.time()
    run_discriminant_audit()
    print(f"\nDiscriminant audit completed in {time.time() - t0:.1f}s")


def cmd_section_diagnosis():
    """Run Phase 4.2: section consistency diagnosis."""
    from voynich.phases.section_diagnosis import run_section_diagnosis
    t0 = time.time()
    run_section_diagnosis()
    print(f"\nSection diagnosis completed in {time.time() - t0:.1f}s")


def cmd_abugida():
    """Run Phase 4.3: abugida hypothesis test."""
    from voynich.phases.abugida_test import run_abugida_test
    t0 = time.time()
    run_abugida_test()
    print(f"\nAbugida test completed in {time.time() - t0:.1f}s")


def cmd_multi_language():
    """Run Phase 4.4: multi-language comparison."""
    from voynich.phases.multi_language import run_multi_language
    t0 = time.time()
    run_multi_language()
    print(f"\nMulti-language comparison completed in {time.time() - t0:.1f}s")


def cmd_phase4():
    """Run all Phase 4 analyses sequentially."""
    cmd_audit()
    print("\n" + "=" * 70 + "\n")
    cmd_section_diagnosis()
    print("\n" + "=" * 70 + "\n")
    cmd_abugida()
    print("\n" + "=" * 70 + "\n")
    cmd_multi_language()


def cmd_lang_a():
    """Run Phase 4.5A+C: Language A isolation and qo-removal."""
    from voynich.phases.language_a_isolation import run_language_a_isolation
    t0 = time.time()
    run_language_a_isolation()
    print(f"\nLanguage A isolation completed in {time.time() - t0:.1f}s")


def cmd_morpheme_grid():
    """Run Phase 4.5B: morpheme grid reinterpretation."""
    from voynich.phases.morpheme_grid import run_morpheme_grid
    t0 = time.time()
    run_morpheme_grid()
    print(f"\nMorpheme grid analysis completed in {time.time() - t0:.1f}s")


def cmd_phase45():
    """Run all Phase 4.5 analyses sequentially."""
    cmd_lang_a()
    print("\n" + "=" * 70 + "\n")
    cmd_morpheme_grid()


def cmd_paradigms():
    """Run Phase 5.1: paradigm discovery."""
    from voynich.phases.paradigm_discovery import run_paradigm_discovery
    t0 = time.time()
    run_paradigm_discovery()
    print(f"\nParadigm discovery completed in {time.time() - t0:.1f}s")


def cmd_paradigm_match():
    """Run Phase 5.2: paradigm-to-language matching."""
    from voynich.phases.paradigm_match import run_paradigm_match
    t0 = time.time()
    run_paradigm_match()
    print(f"\nParadigm matching completed in {time.time() - t0:.1f}s")


def cmd_stem_id():
    """Run Phase 5.3: frequency-based stem identification."""
    from voynich.phases.stem_identification import run_stem_identification
    t0 = time.time()
    run_stem_identification()
    print(f"\nStem identification completed in {time.time() - t0:.1f}s")


def cmd_phonetic():
    """Run Phase 5.4+5.5: phonetic value assignment and validation."""
    from voynich.phases.phonetic_decode import run_phonetic_decode
    t0 = time.time()
    run_phonetic_decode()
    print(f"\nPhonetic decode completed in {time.time() - t0:.1f}s")


def cmd_phase5():
    """Run all Phase 5 analyses sequentially (with gate checking)."""
    cmd_paradigms()
    print("\n" + "=" * 70 + "\n")
    cmd_paradigm_match()
    print("\n" + "=" * 70 + "\n")
    cmd_stem_id()
    print("\n" + "=" * 70 + "\n")
    cmd_phonetic()


def cmd_illustration():
    """Run Phase 6.0: illustration-constrained setup."""
    from voynich.phases.illustration_constrained import run_illustration_constrained
    t0 = time.time()
    run_illustration_constrained()
    print(f"\nIllustration-constrained setup completed in {time.time() - t0:.1f}s")


def cmd_rosetta():
    """Run Phase 6 D+E: Rosetta folio selection and encoding model test."""
    from voynich.phases.rosetta_selection import run_rosetta_selection
    t0 = time.time()
    run_rosetta_selection()
    print(f"\nRosetta selection completed in {time.time() - t0:.1f}s")


def cmd_anchor():
    """Run Phase 6 A+B: anchor-and-propagate with paradigm filtering."""
    from voynich.phases.anchor_propagate import run_anchor_propagate
    t0 = time.time()
    run_anchor_propagate()
    print(f"\nAnchor-and-propagate completed in {time.time() - t0:.1f}s")


def cmd_compete():
    """Run Phase 6 C: competitive ID resolution."""
    from voynich.phases.competitive_id import run_competitive_id
    t0 = time.time()
    run_competitive_id()
    print(f"\nCompetitive ID resolution completed in {time.time() - t0:.1f}s")


def cmd_phase6_validate():
    """Run Phase 6 validation battery."""
    from voynich.phases.illustration_validate import run_illustration_validate
    t0 = time.time()
    run_illustration_validate()
    print(f"\nPhase 6 validation completed in {time.time() - t0:.1f}s")


def cmd_phase6():
    """Run all Phase 6 analyses sequentially (with gate checking)."""
    cmd_illustration()
    print("\n" + "=" * 70 + "\n")
    cmd_rosetta()
    print("\n" + "=" * 70 + "\n")
    cmd_anchor()
    print("\n" + "=" * 70 + "\n")
    cmd_compete()
    print("\n" + "=" * 70 + "\n")
    cmd_phase6_validate()


def cmd_anchor_diagnosis():
    """Run Phase 6.1B: anchor-level inconsistency diagnosis."""
    from voynich.phases.anchor_diagnosis import run_anchor_diagnosis
    t0 = time.time()
    run_anchor_diagnosis()
    print(f"\nAnchor diagnosis completed in {time.time() - t0:.1f}s")


def cmd_encoding_diagnosis():
    """Run Phase 6.1C: encoding model diagnosis."""
    from voynich.phases.encoding_diagnosis import run_encoding_diagnosis
    t0 = time.time()
    run_encoding_diagnosis()
    print(f"\nEncoding diagnosis completed in {time.time() - t0:.1f}s")


def cmd_phase61():
    """Run full Phase 6.1 pipeline: TF-IDF stems + encoding + anchor diagnosis."""
    # Step 1: TF-IDF stem extraction
    from voynich.phases.illustration_constrained import run_illustration_constrained
    t0 = time.time()
    run_illustration_constrained(use_tfidf=True)
    print(f"\nTF-IDF illustration setup completed in {time.time() - t0:.1f}s")

    print("\n" + "=" * 70 + "\n")

    # Step 2: Rosetta selection (uses new stems)
    from voynich.phases.rosetta_selection import run_rosetta_selection
    t0 = time.time()
    run_rosetta_selection(use_tfidf=True)
    print(f"\nRosetta selection completed in {time.time() - t0:.1f}s")

    print("\n" + "=" * 70 + "\n")

    # Step 3: Anchor-propagate with TF-IDF stems
    from voynich.phases.anchor_propagate import run_anchor_propagate
    t0 = time.time()
    run_anchor_propagate(use_tfidf=True)
    print(f"\nAnchor-and-propagate (TF-IDF) completed in {time.time() - t0:.1f}s")

    print("\n" + "=" * 70 + "\n")

    # Step 4: Encoding model diagnosis
    from voynich.phases.encoding_diagnosis import run_encoding_diagnosis
    t0 = time.time()
    run_encoding_diagnosis(use_tfidf=True)
    print(f"\nEncoding diagnosis completed in {time.time() - t0:.1f}s")

    print("\n" + "=" * 70 + "\n")

    # Step 5: Anchor diagnosis
    from voynich.phases.anchor_diagnosis import run_anchor_diagnosis
    t0 = time.time()
    run_anchor_diagnosis(use_tfidf=True)
    print(f"\nAnchor diagnosis completed in {time.time() - t0:.1f}s")

    print("\n" + "=" * 70 + "\n")

    # Step 6: Competitive ID
    from voynich.phases.competitive_id import run_competitive_id
    t0 = time.time()
    run_competitive_id()
    print(f"\nCompetitive ID completed in {time.time() - t0:.1f}s")

    print("\n" + "=" * 70 + "\n")

    # Step 7: Validation battery
    from voynich.phases.illustration_validate import run_illustration_validate
    t0 = time.time()
    run_illustration_validate()
    print(f"\nPhase 6.1 validation completed in {time.time() - t0:.1f}s")


def cmd_embeddings():
    """Run Approach 8: morpheme-level distributional semantics."""
    from voynich.phases.distributional import run_distributional
    t0 = time.time()
    run_distributional()
    print(f"\nDistributional analysis completed in {time.time() - t0:.1f}s")


def cmd_slots():
    """Run Approach 9: pharmaceutical positional slot analysis."""
    from voynich.phases.positional_slots import run_positional_slots
    t0 = time.time()
    run_positional_slots()
    print(f"\nPositional slot analysis completed in {time.time() - t0:.1f}s")


def cmd_phase7():
    """Run full Phase 7: Approaches 8 + 9 + integration."""
    cmd_embeddings()
    print("\n" + "=" * 70 + "\n")
    cmd_slots()
    print("\n" + "=" * 70 + "\n")
    from voynich.phases.approach_integration import run_approach_integration
    t0 = time.time()
    run_approach_integration()
    print(f"\nApproach integration completed in {time.time() - t0:.1f}s")


def cmd_combined_embed():
    """Run Phase 7.5 Step 1: combined A+B corpus embeddings."""
    from voynich.phases.distributional import run_combined_distributional
    t0 = time.time()
    run_combined_distributional()
    print(f"\nCombined embeddings completed in {time.time() - t0:.1f}s")


def cmd_noun_clusters():
    """Run Phase 7.5 Step 2: noun subcluster analysis."""
    from voynich.phases.noun_subclusters import run_noun_subclusters
    t0 = time.time()
    run_noun_subclusters()
    print(f"\nNoun subcluster analysis completed in {time.time() - t0:.1f}s")


def cmd_verb_id():
    """Run Phase 7.5 Step 3: verb identification."""
    from voynich.phases.verb_identification import run_verb_identification
    t0 = time.time()
    run_verb_identification()
    print(f"\nVerb identification completed in {time.time() - t0:.1f}s")


def cmd_embed_bridge():
    """Run Phase 7.5 Step 4: illustration-embedding bridge."""
    from voynich.phases.embedding_bridge import run_embedding_bridge
    t0 = time.time()
    run_embedding_bridge()
    print(f"\nEmbedding bridge completed in {time.time() - t0:.1f}s")


def cmd_convergence():
    """Run Phase 7.5 Step 5: convergence scoring."""
    from voynich.phases.convergence_score import run_convergence_score
    t0 = time.time()
    run_convergence_score()
    print(f"\nConvergence scoring completed in {time.time() - t0:.1f}s")


def cmd_phase75():
    """Run full Phase 7.5: Exploiting the Noun Coherence Bridge."""
    cmd_combined_embed()
    print("\n" + "=" * 70 + "\n")
    cmd_noun_clusters()
    print("\n" + "=" * 70 + "\n")
    cmd_verb_id()
    print("\n" + "=" * 70 + "\n")
    cmd_embed_bridge()
    print("\n" + "=" * 70 + "\n")
    cmd_convergence()


def cmd_bigram_transfer():
    """Run Approach 16: bigram transfer cryptanalysis."""
    from voynich.phases.bigram_transfer import run_bigram_transfer
    t0 = time.time()
    run_bigram_transfer()
    print(f"\nBigram transfer completed in {time.time() - t0:.1f}s")


def cmd_mdl_decode():
    """Run Approach 18: MDL decoding."""
    from voynich.phases.mdl_decode import run_mdl_decode
    t0 = time.time()
    run_mdl_decode()
    print(f"\nMDL decode completed in {time.time() - t0:.1f}s")


def cmd_cipher_validate():
    """Run Phase 8 validation battery."""
    from voynich.phases.cipher_validate import run_cipher_validate
    t0 = time.time()
    run_cipher_validate()
    print(f"\nCipher validation completed in {time.time() - t0:.1f}s")


def cmd_phase8():
    """Run full Phase 8: Approaches 16 + 18 + validation."""
    cmd_bigram_transfer()
    print("\n" + "=" * 70 + "\n")
    cmd_mdl_decode()
    print("\n" + "=" * 70 + "\n")
    cmd_cipher_validate()


def cmd_nomenclator():
    """Run Phase 9.2: nomenclator / bimodal frequency test."""
    from voynich.phases.nomenclator_test import run_nomenclator_test
    t0 = time.time()
    run_nomenclator_test()
    print(f"\nNomenclator test completed in {time.time() - t0:.1f}s")


def cmd_homophones():
    """Run Phase 9.1: homophonic substitution test."""
    from voynich.phases.homophone_test import run_homophone_test
    t0 = time.time()
    run_homophone_test()
    print(f"\nHomophone test completed in {time.time() - t0:.1f}s")


def cmd_position_dep():
    """Run Phase 9.3: position-dependent encoding test."""
    from voynich.phases.position_dependent import run_position_dependent
    t0 = time.time()
    run_position_dependent()
    print(f"\nPosition-dependent test completed in {time.time() - t0:.1f}s")


def cmd_lang_compare():
    """Run Phase 9.4: expanded language comparison."""
    from voynich.phases.language_comparison import run_language_comparison
    t0 = time.time()
    run_language_comparison()
    print(f"\nLanguage comparison completed in {time.time() - t0:.1f}s")


def cmd_typology():
    """Run Phase 9.5: text typology classification."""
    from voynich.phases.text_typology import run_text_typology
    t0 = time.time()
    run_text_typology()
    print(f"\nText typology completed in {time.time() - t0:.1f}s")


def cmd_phase9():
    """Run full Phase 9: alternative encoding hypothesis testing."""
    cmd_nomenclator()
    print("\n" + "=" * 70 + "\n")
    cmd_homophones()
    print("\n" + "=" * 70 + "\n")
    cmd_position_dep()
    print("\n" + "=" * 70 + "\n")
    cmd_lang_compare()
    print("\n" + "=" * 70 + "\n")
    cmd_typology()


def cmd_entropy_curves():
    """Run Phase 10.1: token-level entropy curves."""
    from voynich.phases.entropy_curves import run_entropy_curves
    t0 = time.time()
    run_entropy_curves()
    print(f"\nEntropy curves completed in {time.time() - t0:.1f}s")


def cmd_mi_decay():
    """Run Phase 10.2: mutual information decay."""
    from voynich.phases.mutual_info_decay import run_mutual_info_decay
    t0 = time.time()
    run_mutual_info_decay()
    print(f"\nMI decay completed in {time.time() - t0:.1f}s")


def cmd_folio_shift():
    """Run Phase 10.3: folio-level encoding shifts."""
    from voynich.phases.folio_shift import run_folio_shift
    t0 = time.time()
    run_folio_shift()
    print(f"\nFolio shift test completed in {time.time() - t0:.1f}s")


def cmd_glyph_grammar():
    """Run Phase 10.4: glyph construction grammar."""
    from voynich.phases.glyph_grammar import run_glyph_grammar
    t0 = time.time()
    run_glyph_grammar()
    print(f"\nGlyph grammar test completed in {time.time() - t0:.1f}s")


def cmd_hypothesis():
    """Run Phase 10.5: hypothesis integration and verdict."""
    from voynich.phases.hypothesis_verdict import run_hypothesis_verdict
    t0 = time.time()
    run_hypothesis_verdict()
    print(f"\nHypothesis verdict completed in {time.time() - t0:.1f}s")


def cmd_phase10():
    """Run full Phase 10: three-hypothesis testing."""
    cmd_entropy_curves()
    print("\n" + "=" * 70 + "\n")
    cmd_mi_decay()
    print("\n" + "=" * 70 + "\n")
    cmd_folio_shift()
    print("\n" + "=" * 70 + "\n")
    cmd_glyph_grammar()
    print("\n" + "=" * 70 + "\n")
    cmd_hypothesis()


def cmd_csp_solve():
    """Run Phase 11.0: CSP solver sanity test."""
    from voynich.phases.csp_solver import run_csp_solver_test
    t0 = time.time()
    run_csp_solver_test()
    print(f"\nCSP solver test completed in {time.time() - t0:.1f}s")


def cmd_csp_decode():
    """Run Phase 11.2: multi-language CSP phonetic decoding."""
    from voynich.phases.csp_decode import run_csp_decode
    t0 = time.time()
    run_csp_decode()
    print(f"\nCSP decode completed in {time.time() - t0:.1f}s")


def cmd_csp_validate():
    """Run Phase 11.3: CSP validation battery."""
    from voynich.phases.csp_validate import run_csp_validation_phase
    t0 = time.time()
    run_csp_validation_phase()
    print(f"\nCSP validation completed in {time.time() - t0:.1f}s")


def cmd_phase11():
    """Run full Phase 11: CSP phonetic decoding."""
    cmd_csp_solve()
    print("\n" + "=" * 70 + "\n")
    cmd_csp_decode()
    print("\n" + "=" * 70 + "\n")
    cmd_csp_validate()


def cmd_csp_diagnose():
    """Run Phase 11.5.1: CSP failure diagnosis."""
    from voynich.phases.csp_diagnosis import run_csp_diagnosis
    t0 = time.time()
    run_csp_diagnosis()
    print(f"\nCSP diagnosis completed in {time.time() - t0:.1f}s")


def cmd_csp_refine():
    """Run Phase 11.5.2-3: inherent vowel + relaxation sweep."""
    from voynich.phases.csp_refinement import run_csp_refinement
    t0 = time.time()
    run_csp_refinement()
    print(f"\nCSP refinement completed in {time.time() - t0:.1f}s")


def cmd_verb_constrain():
    """Run Phase 11.5.4: verb-constrained CSP solving."""
    from voynich.phases.verb_constraints import run_verb_constraints
    t0 = time.time()
    run_verb_constraints()
    print(f"\nVerb-constrained CSP completed in {time.time() - t0:.1f}s")


def cmd_csp_iterate():
    """Run Phase 11.5.5: iterative CSP refinement loop."""
    from voynich.phases.csp_iterate import run_csp_iterate
    t0 = time.time()
    run_csp_iterate()
    print(f"\nCSP iterative refinement completed in {time.time() - t0:.1f}s")


def cmd_csp_final():
    """Run Phase 11.5.6-7: final multi-language comparison + V1-V9 validation."""
    from voynich.phases.csp_final import run_csp_final
    t0 = time.time()
    run_csp_final()
    print(f"\nCSP final validation completed in {time.time() - t0:.1f}s")


def cmd_phase115():
    """Run full Phase 11.5: CSP refinement pipeline."""
    cmd_csp_diagnose()
    print("\n" + "=" * 70 + "\n")
    cmd_csp_refine()
    print("\n" + "=" * 70 + "\n")
    cmd_verb_constrain()
    print("\n" + "=" * 70 + "\n")
    cmd_csp_iterate()
    print("\n" + "=" * 70 + "\n")
    cmd_csp_final()


def cmd_grid_recal():
    """Run Phase 12.1-12.2: grid recalibration from correction vectors."""
    from voynich.phases.grid_recalibrate import run_grid_recalibration
    t0 = time.time()
    run_grid_recalibration()
    print(f"\nGrid recalibration completed in {time.time() - t0:.1f}s")


def cmd_grid_alt():
    """Run Phase 12.4: stroke-based alternative grid construction."""
    from voynich.phases.grid_alternatives import run_grid_alternatives
    t0 = time.time()
    run_grid_alternatives()
    print(f"\nGrid alternatives analysis completed in {time.time() - t0:.1f}s")


def cmd_token_decomp():
    """Run Phase 12.5: token decomposition variant sweep."""
    from voynich.phases.token_decomposition import run_token_decomposition
    t0 = time.time()
    run_token_decomposition()
    print(f"\nToken decomposition sweep completed in {time.time() - t0:.1f}s")


def cmd_recal_csp():
    """Run Phase 12.3+12.6: recalibrated CSP solve + full validation."""
    from voynich.phases.recalibrated_csp import run_recalibrated_csp
    t0 = time.time()
    run_recalibrated_csp()
    print(f"\nRecalibrated CSP completed in {time.time() - t0:.1f}s")


def cmd_phase12():
    """Run full Phase 12 pipeline: grid recalibration + alternative grids + decomp variants + final CSP."""
    print("=" * 70)
    print("PHASE 12: Grid Recalibration")
    print("=" * 70)
    cmd_grid_recal()
    print("\n" + "=" * 70 + "\n")
    cmd_grid_alt()
    print("\n" + "=" * 70 + "\n")
    cmd_token_decomp()
    print("\n" + "=" * 70 + "\n")
    cmd_recal_csp()


def cmd_error_patterns():
    """Run Phase 13.1: near-miss error pattern analysis and MI gate."""
    from voynich.phases.error_patterns import run_error_patterns
    t0 = time.time()
    run_error_patterns()
    print(f"\nError pattern analysis completed in {time.time() - t0:.1f}s")


def cmd_null_context():
    """Run Phase 13.6: null hypothesis testing (cell conflation + dict expansion)."""
    from voynich.phases.null_context import run_null_context
    t0 = time.time()
    run_null_context()
    print(f"\nNull context analysis completed in {time.time() - t0:.1f}s")


def cmd_extract_rules():
    """Run Phase 13.2: context-dependent rule extraction and power ranking."""
    from voynich.phases.rule_extraction import run_rule_extraction
    t0 = time.time()
    run_rule_extraction()
    print(f"\nRule extraction completed in {time.time() - t0:.1f}s")


def cmd_context_csp():
    """Run Phase 13.3: context-aware CSP solver (Version A + B)."""
    from voynich.phases.context_csp import run_context_csp
    t0 = time.time()
    run_context_csp()
    print(f"\nContext CSP completed in {time.time() - t0:.1f}s")


def cmd_rule_validate():
    """Run Phase 13.4: cross-validation + per-rule selectivity + plausibility."""
    from voynich.phases.rule_validation import run_rule_validation
    t0 = time.time()
    run_rule_validation()
    print(f"\nRule validation completed in {time.time() - t0:.1f}s")


def cmd_context_decode():
    """Run Phase 13.5: full corpus decoding with validated rules + V1-V11 battery."""
    from voynich.phases.context_decode import run_context_decode
    t0 = time.time()
    run_context_decode()
    print(f"\nContext decode completed in {time.time() - t0:.1f}s")


def cmd_phase13():
    """Run full Phase 13 pipeline: context-dependent reading rules."""
    print("=" * 70)
    print("PHASE 13: Context-Dependent Reading Rules")
    print("=" * 70)
    cmd_error_patterns()
    print("\n" + "=" * 70 + "\n")
    cmd_null_context()
    print("\n" + "=" * 70 + "\n")
    cmd_extract_rules()
    print("\n" + "=" * 70 + "\n")
    cmd_context_csp()
    print("\n" + "=" * 70 + "\n")
    cmd_rule_validate()
    print("\n" + "=" * 70 + "\n")
    cmd_context_decode()


# ---------------------------------------------------------------------------
# Phase 14: Sub-Cell Phonetic Feature Model
# ---------------------------------------------------------------------------

def cmd_cell_analysis():
    """Run Phase 14.1: within-cell character distributional analysis."""
    from voynich.phases.cell_analysis import run_cell_analysis
    t0 = time.time()
    run_cell_analysis()
    print(f"\nCell analysis completed in {time.time() - t0:.1f}s")


def cmd_stroke_features():
    """Run Phase 14.2: stroke feature decomposition and triple enumeration."""
    from voynich.phases.stroke_features import run_stroke_features
    t0 = time.time()
    run_stroke_features()
    print(f"\nStroke features completed in {time.time() - t0:.1f}s")


def cmd_feature_csp():
    """Run Phase 14.3: feature-level CSP solver (25 variables vs 14 cells)."""
    from voynich.phases.feature_csp import run_feature_csp
    t0 = time.time()
    run_feature_csp()
    print(f"\nFeature CSP completed in {time.time() - t0:.1f}s")


def cmd_feature_calibrate():
    """Run Phase 14.4: synthetic abugida calibration of the feature CSP."""
    from voynich.phases.feature_calibrate import run_feature_calibrate
    t0 = time.time()
    run_feature_calibrate()
    print(f"\nFeature calibration completed in {time.time() - t0:.1f}s")


def cmd_feature_decode():
    """Run Phase 14.5-14.6: full Voynich decode + V1-V12 validation battery."""
    from voynich.phases.feature_decode import run_feature_decode
    t0 = time.time()
    run_feature_decode()
    print(f"\nFeature decode completed in {time.time() - t0:.1f}s")


def cmd_subcell_split():
    """Run Phase 14.7: data-driven sub-cell splitting fallback CSP."""
    from voynich.phases.subcell_split import run_subcell_split
    t0 = time.time()
    run_subcell_split()
    print(f"\nSub-cell split completed in {time.time() - t0:.1f}s")


def cmd_phase14():
    """Run full Phase 14 pipeline: sub-cell phonetic feature model."""
    print("=" * 70)
    print("PHASE 14: Sub-Cell Phonetic Feature Model")
    print("=" * 70)
    cmd_cell_analysis()
    print("\n" + "=" * 70 + "\n")
    cmd_stroke_features()
    print("\n" + "=" * 70 + "\n")
    cmd_feature_csp()
    print("\n" + "=" * 70 + "\n")
    cmd_feature_calibrate()
    print("\n" + "=" * 70 + "\n")
    cmd_feature_decode()
    print("\n" + "=" * 70 + "\n")
    cmd_subcell_split()


# ---------------------------------------------------------------------------
# Phase 15: Feature Model Refinement
# ---------------------------------------------------------------------------

def cmd_dict_expand():
    """Run Phase 15.1: medieval Latin dictionary expansion + re-scoring."""
    from voynich.phases.dict_expansion import run_dict_expansion
    t0 = time.time()
    run_dict_expansion()
    print(f"\nDictionary expansion completed in {time.time() - t0:.1f}s")


def cmd_artic_csp():
    """Run Phase 15.2: articulatory consistency CSP scoring."""
    from voynich.phases.articulatory_csp import run_articulatory_csp
    t0 = time.time()
    run_articulatory_csp()
    print(f"\nArticulatory CSP completed in {time.time() - t0:.1f}s")


def cmd_iter_hits():
    """Run Phase 15.3: iterative re-solving with confirmed hits."""
    from voynich.phases.iterative_hits import run_iterative_hits
    t0 = time.time()
    run_iterative_hits()
    print(f"\nIterative hits completed in {time.time() - t0:.1f}s")


def cmd_combined_refine():
    """Run Phase 15.4: combined optimization + ablation study."""
    from voynich.phases.combined_refine import run_combined_refine
    t0 = time.time()
    run_combined_refine()
    print(f"\nCombined refine completed in {time.time() - t0:.1f}s")


def cmd_text_analysis():
    """Run Phase 15.5: decoded text analysis + phrase detection."""
    from voynich.phases.text_analysis import run_text_analysis
    t0 = time.time()
    run_text_analysis()
    print(f"\nText analysis completed in {time.time() - t0:.1f}s")


def cmd_phase15_validate():
    """Run Phase 15.6: full V1-V14 validation battery."""
    from voynich.phases.phase15_validate import run_phase15_validate
    t0 = time.time()
    run_phase15_validate()
    print(f"\nPhase 15 validation completed in {time.time() - t0:.1f}s")


def cmd_phase15():
    """Run full Phase 15 pipeline: feature model refinement."""
    print("=" * 70)
    print("PHASE 15: Feature Model Refinement")
    print("=" * 70)
    cmd_dict_expand()
    print("\n" + "=" * 70 + "\n")
    cmd_artic_csp()
    print("\n" + "=" * 70 + "\n")
    cmd_iter_hits()
    print("\n" + "=" * 70 + "\n")
    cmd_combined_refine()
    print("\n" + "=" * 70 + "\n")
    cmd_text_analysis()
    print("\n" + "=" * 70 + "\n")
    cmd_phase15_validate()


# ---------------------------------------------------------------------------
# Phase 16: Modifier Detection and Syllable Correction
# ---------------------------------------------------------------------------

def cmd_mod_standalone():
    """Run Phase 16.1: standalone modifier candidate analysis."""
    from voynich.phases.modifier_standalone import run_modifier_standalone
    t0 = time.time()
    run_modifier_standalone()
    print(f"\nModifier standalone analysis completed in {time.time() - t0:.1f}s")


def cmd_mod_anomaly():
    """Run Phase 16.2: frequency anomaly modifier detection."""
    from voynich.phases.modifier_anomaly import run_modifier_anomaly
    t0 = time.time()
    run_modifier_anomaly()
    print(f"\nModifier anomaly detection completed in {time.time() - t0:.1f}s")


def cmd_mod_distrib():
    """Run Phase 16.3: syllable distribution matching."""
    from voynich.phases.modifier_distribution import run_modifier_distribution
    t0 = time.time()
    run_modifier_distribution()
    print(f"\nModifier distribution matching completed in {time.time() - t0:.1f}s")


def cmd_mod_pairs():
    """Run Phase 16.4: minimal pair modifier evidence."""
    from voynich.phases.modifier_minimal_pairs import run_modifier_minimal_pairs
    t0 = time.time()
    run_modifier_minimal_pairs()
    print(f"\nMinimal pair analysis completed in {time.time() - t0:.1f}s")


def cmd_mod_localize():
    """Run Phase 16.5: hit localization padding analysis."""
    from voynich.phases.modifier_localize import run_modifier_localize
    t0 = time.time()
    run_modifier_localize()
    print(f"\nHit localization analysis completed in {time.time() - t0:.1f}s")


def cmd_mod_integrate():
    """Run Phase 16.6: convergent classification + re-decode."""
    from voynich.phases.modifier_integrate import run_modifier_integrate
    t0 = time.time()
    run_modifier_integrate()
    print(f"\nModifier integration + re-decode completed in {time.time() - t0:.1f}s")


def cmd_phase16():
    """Run full Phase 16 pipeline: modifier detection + syllable correction."""
    print("=" * 70)
    print("PHASE 16: Modifier Detection and Syllable Correction")
    print("=" * 70)
    cmd_mod_standalone()
    print("\n" + "=" * 70 + "\n")
    cmd_mod_anomaly()
    print("\n" + "=" * 70 + "\n")
    cmd_mod_distrib()
    print("\n" + "=" * 70 + "\n")
    cmd_mod_pairs()
    print("\n" + "=" * 70 + "\n")
    cmd_mod_localize()
    print("\n" + "=" * 70 + "\n")
    cmd_mod_integrate()


# ── Phase 17 Step 0: Honesty Diagnostics ──────────────────────────────

def cmd_honesty_dict():
    """Run Phase 17.0.1: dictionary tier control test."""
    from voynich.phases.honesty_dict import run_honesty_dict
    t0 = time.time()
    run_honesty_dict()
    print(f"\nDictionary honesty test completed in {time.time() - t0:.1f}s")


def cmd_honesty_keywords():
    """Run Phase 17.0.2: keyword presence test."""
    from voynich.phases.honesty_keywords import run_honesty_keywords
    t0 = time.time()
    run_honesty_keywords()
    print(f"\nKeyword presence test completed in {time.time() - t0:.1f}s")


def cmd_honesty_verbs():
    """Run Phase 17.0.3: positional verb decode test."""
    from voynich.phases.honesty_verbs import run_honesty_verbs
    t0 = time.time()
    run_honesty_verbs()
    print(f"\nVerb decode test completed in {time.time() - t0:.1f}s")


def cmd_null_corpus():
    """Run Phase 17.0.4: null corpus control test."""
    from voynich.phases.null_corpus import run_null_corpus
    t0 = time.time()
    run_null_corpus()
    print(f"\nNull corpus control test completed in {time.time() - t0:.1f}s")


def cmd_honesty_words():
    """Run Phase 17.0.5: minimum viable word test."""
    from voynich.phases.honesty_words import run_honesty_words
    t0 = time.time()
    run_honesty_words()
    print(f"\nMinimum viable word test completed in {time.time() - t0:.1f}s")


def cmd_step0_integrate():
    """Run Phase 17.0.6: honesty diagnostics integration."""
    from voynich.phases.step0_integrate import run_step0_integrate
    t0 = time.time()
    run_step0_integrate()
    print(f"\nStep 0 integration completed in {time.time() - t0:.1f}s")


def cmd_step0():
    """Run full Phase 17 Step 0: honesty diagnostics pipeline."""
    print("=" * 70)
    print("PHASE 17 STEP 0: Honesty Diagnostics")
    print("=" * 70)
    cmd_honesty_dict()
    print("\n" + "=" * 70 + "\n")
    cmd_honesty_keywords()
    print("\n" + "=" * 70 + "\n")
    cmd_honesty_verbs()
    print("\n" + "=" * 70 + "\n")
    cmd_null_corpus()
    print("\n" + "=" * 70 + "\n")
    cmd_honesty_words()
    print("\n" + "=" * 70 + "\n")
    cmd_step0_integrate()


# ── Phase A: Paleographic Reference Inventory ─────────────────────────

def cmd_ref_validate():
    """Run Phase A.3a: validate all paleographic reference JSONs."""
    from voynich.phases.ref_validate import run_ref_validate
    t0 = time.time()
    run_ref_validate()
    print(f"\nReference validation completed in {time.time() - t0:.1f}s")


def cmd_ref_merge():
    """Run Phase A.3b: merge validated sources into master reference."""
    from voynich.phases.ref_merge import run_ref_merge
    t0 = time.time()
    run_ref_merge()
    print(f"\nReference merge completed in {time.time() - t0:.1f}s")


def cmd_phaseA():
    """Run full Phase A pipeline: reference inventory."""
    print("=" * 70)
    print("PHASE A: Paleographic Reference Inventory")
    print("=" * 70)
    cmd_ref_validate()
    print("\n" + "=" * 70 + "\n")
    cmd_ref_merge()


# ── Phase B: Structural Comparison ────────────────────────────────────

def cmd_ligature_analysis():
    """Run Phase B.0: ligature analysis on manuscript images."""
    from voynich.phases.ligature_analysis import run_ligature_analysis
    t0 = time.time()
    run_ligature_analysis()
    print(f"\nLigature analysis completed in {time.time() - t0:.1f}s")


def cmd_triple_overlap():
    """Run Phase B.1: triple overlap analysis (Voynich vs Tironian)."""
    from voynich.phases.triple_overlap import run_triple_overlap
    t0 = time.time()
    run_triple_overlap()
    print(f"\nTriple overlap analysis completed in {time.time() - t0:.1f}s")


def cmd_modifier_tironian():
    """Run Phase B.2: modifier character test against Tironian marks."""
    from voynich.phases.modifier_tironian import run_modifier_tironian
    t0 = time.time()
    run_modifier_tironian()
    print(f"\nModifier Tironian test completed in {time.time() - t0:.1f}s")


def cmd_positional_compare():
    """Run Phase B.3: positional constraint comparison."""
    from voynich.phases.positional_compare import run_positional_compare
    t0 = time.time()
    run_positional_compare()
    print(f"\nPositional comparison completed in {time.time() - t0:.1f}s")


def cmd_cappelli_match():
    """Run Phase B.4: Cappelli quick-match."""
    from voynich.phases.cappelli_match import run_cappelli_match
    t0 = time.time()
    run_cappelli_match()
    print(f"\nCappelli match completed in {time.time() - t0:.1f}s")


def cmd_fontana_compare():
    """Run Phase B.5: Fontana structural comparison."""
    from voynich.phases.fontana_compare import run_fontana_compare
    t0 = time.time()
    run_fontana_compare()
    print(f"\nFontana comparison completed in {time.time() - t0:.1f}s")


def cmd_phaseB():
    """Run full Phase B pipeline: structural comparison."""
    print("=" * 70)
    print("PHASE B: Structural Comparison")
    print("=" * 70)
    cmd_ligature_analysis()
    print("\n" + "=" * 70 + "\n")
    cmd_triple_overlap()
    print("\n" + "=" * 70 + "\n")
    cmd_modifier_tironian()
    print("\n" + "=" * 70 + "\n")
    cmd_positional_compare()
    print("\n" + "=" * 70 + "\n")
    cmd_cappelli_match()
    print("\n" + "=" * 70 + "\n")
    cmd_fontana_compare()


# ── Phase C: CSP Re-Solve With Paleographic Priors ───────────────────

def cmd_tironian_csp():
    """Run Phase C.1-C.2: CSP re-solve with Tironian priors."""
    from voynich.phases.tironian_csp import run_tironian_csp
    t0 = time.time()
    run_tironian_csp()
    print(f"\nTironian CSP completed in {time.time() - t0:.1f}s")


def cmd_phrase_detect():
    """Run Phase C.3: phrase detection (primary success criterion)."""
    from voynich.phases.phrase_detect import run_phrase_detect
    t0 = time.time()
    run_phrase_detect()
    print(f"\nPhrase detection completed in {time.time() - t0:.1f}s")


def cmd_modifier_clean():
    """Run Phase C.4: modifier-clean subset test."""
    from voynich.phases.modifier_clean import run_modifier_clean
    t0 = time.time()
    run_modifier_clean()
    print(f"\nModifier-clean test completed in {time.time() - t0:.1f}s")


def cmd_reseg_csp():
    """Run Phase C.5: re-segmented CSP (if ligatures found)."""
    from voynich.phases.reseg_csp import run_reseg_csp
    t0 = time.time()
    run_reseg_csp()
    print(f"\nRe-segmented CSP completed in {time.time() - t0:.1f}s")


def cmd_phaseC_validate():
    """Run Phase C.6: full validation battery (17 tests)."""
    from voynich.phases.phaseC_validate import run_phaseC_validate
    t0 = time.time()
    run_phaseC_validate()
    print(f"\nPhase C validation completed in {time.time() - t0:.1f}s")


def cmd_phaseC():
    """Run full Phase C pipeline: CSP re-solve with Tironian priors."""
    print("=" * 70)
    print("PHASE C: CSP Re-Solve With Paleographic Priors")
    print("=" * 70)
    cmd_tironian_csp()
    print("\n" + "=" * 70 + "\n")
    cmd_phrase_detect()
    print("\n" + "=" * 70 + "\n")
    cmd_modifier_clean()
    print("\n" + "=" * 70 + "\n")
    cmd_reseg_csp()
    print("\n" + "=" * 70 + "\n")
    cmd_phaseC_validate()


# ── Phase D: Parallel Historical Investigation ────────────────────────

def cmd_milanese_fingerprint():
    """Run Phase D.1: Milanese cipher fingerprint comparison."""
    from voynich.phases.milanese_fingerprint import run_milanese_fingerprint
    t0 = time.time()
    run_milanese_fingerprint()
    print(f"\nMilanese fingerprint completed in {time.time() - t0:.1f}s")


def cmd_entropy_floor():
    """Run Phase D.2: entropy floor diagnostic."""
    from voynich.phases.entropy_floor import run_entropy_floor
    t0 = time.time()
    run_entropy_floor()
    print(f"\nEntropy floor diagnostic completed in {time.time() - t0:.1f}s")


def cmd_verbose_encoding():
    """Run Phase D.3: verbose encoding assessment."""
    from voynich.phases.verbose_encoding import run_verbose_encoding
    t0 = time.time()
    run_verbose_encoding()
    print(f"\nVerbose encoding assessment completed in {time.time() - t0:.1f}s")


def cmd_phaseD():
    """Run full Phase D pipeline: parallel historical investigation."""
    print("=" * 70)
    print("PHASE D: Parallel Historical Investigation")
    print("=" * 70)
    cmd_milanese_fingerprint()
    print("\n" + "=" * 70 + "\n")
    cmd_entropy_floor()
    print("\n" + "=" * 70 + "\n")
    cmd_verbose_encoding()


# ── Phase 18: Hypothesis Discrimination Battery ──────────────────────────

def cmd_burstiness():
    """Run Phase 18.1: spatial autocorrelation / burstiness test."""
    from voynich.phases.burstiness_test import run_burstiness_test
    t0 = time.time()
    run_burstiness_test()
    print(f"\nBurstiness test completed in {time.time() - t0:.1f}s")


def cmd_stride_entropy():
    """Run Phase 18.2: stride-entropy decimation test."""
    from voynich.phases.stride_entropy import run_stride_entropy
    t0 = time.time()
    run_stride_entropy()
    print(f"\nStride entropy test completed in {time.time() - t0:.1f}s")


def cmd_trie_topology():
    """Run Phase 18.3: prefix trie topology analysis."""
    from voynich.phases.trie_topology import run_trie_topology
    t0 = time.time()
    run_trie_topology()
    print(f"\nTrie topology analysis completed in {time.time() - t0:.1f}s")


def cmd_hmm_pos():
    """Run Phase 18.4: unsupervised HMM POS induction."""
    from voynich.phases.hmm_pos_induction import run_hmm_pos_induction
    t0 = time.time()
    run_hmm_pos_induction()
    print(f"\nHMM POS induction completed in {time.time() - t0:.1f}s")


def cmd_lz_complexity():
    """Run Phase 18.5: Lempel-Ziv complexity growth curve."""
    from voynich.phases.lz_complexity import run_lz_complexity
    t0 = time.time()
    run_lz_complexity()
    print(f"\nLZ complexity test completed in {time.time() - t0:.1f}s")


def cmd_hyp_discriminate():
    """Run Phase 18.6: final hypothesis discrimination."""
    from voynich.phases.hypothesis_discriminator import run_hypothesis_discriminator
    t0 = time.time()
    run_hypothesis_discriminator()
    print(f"\nHypothesis discrimination completed in {time.time() - t0:.1f}s")


def cmd_phase18():
    """Run full Phase 18 pipeline: hypothesis discrimination battery."""
    print("=" * 70)
    print("PHASE 18: Hypothesis Discrimination Battery")
    print("=" * 70)
    cmd_burstiness()
    print("\n" + "=" * 70 + "\n")
    cmd_stride_entropy()
    print("\n" + "=" * 70 + "\n")
    cmd_trie_topology()
    print("\n" + "=" * 70 + "\n")
    cmd_hmm_pos()
    print("\n" + "=" * 70 + "\n")
    cmd_lz_complexity()
    print("\n" + "=" * 70 + "\n")
    cmd_hyp_discriminate()


# === Phase 19: Convergent Constraint Exploitation ===

def cmd_modifier_validate():
    """Run Phase 19.4: modifier character distributional validation."""
    from voynich.phases.modifier_validation import run_modifier_validation
    t0 = time.time()
    run_modifier_validation()
    print(f"\nModifier validation completed in {time.time() - t0:.1f}s")


def cmd_affix_isolate():
    """Run Phase 19.3: affix layer isolation and independent decoding."""
    from voynich.phases.affix_isolation import run_affix_isolation
    t0 = time.time()
    run_affix_isolation()
    print(f"\nAffix isolation completed in {time.time() - t0:.1f}s")


def cmd_lang_b_attack():
    """Run Phase 19.1: Language B combinatorial attack."""
    from voynich.phases.lang_b_combinatorial import run_lang_b_combinatorial
    t0 = time.time()
    run_lang_b_combinatorial()
    print(f"\nLanguage B combinatorial attack completed in {time.time() - t0:.1f}s")


def cmd_entropy_shift():
    """Run Phase 19.2: cipher mechanism entropy shift identification."""
    from voynich.phases.entropy_shift_cipher import run_entropy_shift
    t0 = time.time()
    run_entropy_shift()
    print(f"\nEntropy shift analysis completed in {time.time() - t0:.1f}s")


def cmd_tachy_stroke():
    """Run Phase 19.5: tachygraphic stroke-rule test."""
    from voynich.phases.tachygraphic_stroke import run_tachygraphic_stroke
    t0 = time.time()
    run_tachygraphic_stroke()
    print(f"\nTachygraphic stroke test completed in {time.time() - t0:.1f}s")


def cmd_cross_validate():
    """Run Phase 19.8: cross-approach bidirectional mapping validation."""
    from voynich.phases.cross_approach import run_cross_approach
    t0 = time.time()
    run_cross_approach()
    print(f"\nCross-approach validation completed in {time.time() - t0:.1f}s")


def cmd_illus_target():
    """Run Phase 19.7: illustration-targeted folio decode."""
    from voynich.phases.illustration_targeted import run_illustration_targeted
    t0 = time.time()
    run_illustration_targeted()
    print(f"\nIllustration-targeted decode completed in {time.time() - t0:.1f}s")


def cmd_stroke_sim():
    """Run Phase 19.6: stroke-modification encoding simulation."""
    from voynich.phases.stroke_modification import run_stroke_modification
    t0 = time.time()
    run_stroke_modification()
    print(f"\nStroke modification simulation completed in {time.time() - t0:.1f}s")


def cmd_phase19_integrate():
    """Run Phase 19.9: Phase 19 integration."""
    from voynich.phases.phase19_integrate import run_phase19_integrate
    t0 = time.time()
    run_phase19_integrate()
    print(f"\nPhase 19 integration completed in {time.time() - t0:.1f}s")


def cmd_phase19():
    """Run full Phase 19 pipeline: convergent constraint exploitation."""
    print("=" * 70)
    print("PHASE 19: Convergent Constraint Exploitation")
    print("=" * 70)
    cmd_modifier_validate()
    print("\n" + "=" * 70 + "\n")
    cmd_affix_isolate()
    print("\n" + "=" * 70 + "\n")
    cmd_lang_b_attack()
    print("\n" + "=" * 70 + "\n")
    cmd_entropy_shift()
    print("\n" + "=" * 70 + "\n")
    cmd_tachy_stroke()
    print("\n" + "=" * 70 + "\n")
    cmd_cross_validate()
    print("\n" + "=" * 70 + "\n")
    cmd_illus_target()
    print("\n" + "=" * 70 + "\n")
    cmd_stroke_sim()
    print("\n" + "=" * 70 + "\n")
    cmd_phase19_integrate()


# ---------------------------------------------------------------------------
# Phase 20: Tachygraphic Table Construction and Corpus Decoding
# ---------------------------------------------------------------------------

def cmd_tachy_anchors():
    """Run Phase 20.1: tachygraphic anchor extraction."""
    from voynich.phases.tachy_anchors import run_tachy_anchors
    t0 = time.time()
    run_tachy_anchors()
    print(f"\nTachygraphic anchors completed in {time.time() - t0:.1f}s")


def cmd_tachy_families():
    """Run Phase 20.2: sign family → syllable family mapping."""
    from voynich.phases.tachy_families import run_tachy_families
    t0 = time.time()
    run_tachy_families()
    print(f"\nTachygraphic families completed in {time.time() - t0:.1f}s")


def cmd_tachy_grid():
    """Run Phase 20.3: constrained tachygraphic grid solve."""
    from voynich.phases.tachy_grid_solve import run_tachy_grid_solve
    t0 = time.time()
    run_tachy_grid_solve()
    print(f"\nTachygraphic grid solve completed in {time.time() - t0:.1f}s")


def cmd_tachy_decode():
    """Run Phase 20.4: full corpus tachygraphic decode."""
    from voynich.phases.tachy_decode import run_tachy_decode
    t0 = time.time()
    run_tachy_decode()
    print(f"\nTachygraphic decode completed in {time.time() - t0:.1f}s")


def cmd_tachy_read():
    """Run Phase 20.5: tachygraphic readability assessment."""
    from voynich.phases.tachy_readability import run_tachy_readability
    t0 = time.time()
    run_tachy_readability()
    print(f"\nTachygraphic readability completed in {time.time() - t0:.1f}s")


def cmd_tachy_phrases():
    """Run Phase 20.6: Latin phrase detection + botanical cross-check."""
    from voynich.phases.tachy_phrases import run_tachy_phrases
    t0 = time.time()
    run_tachy_phrases()
    print(f"\nTachygraphic phrases completed in {time.time() - t0:.1f}s")


def cmd_tachy_validate():
    """Run Phase 20.7: tachygraphic validation battery."""
    from voynich.phases.tachy_validate import run_tachy_validate
    t0 = time.time()
    run_tachy_validate()
    print(f"\nTachygraphic validation completed in {time.time() - t0:.1f}s")


def cmd_phase20_integrate():
    """Run Phase 20.8: Phase 20 integration and verdict."""
    from voynich.phases.phase20_integrate import run_phase20_integrate
    t0 = time.time()
    run_phase20_integrate()
    print(f"\nPhase 20 integration completed in {time.time() - t0:.1f}s")


def cmd_phase20():
    """Run full Phase 20 pipeline: tachygraphic table construction."""
    print("=" * 70)
    print("PHASE 20: Tachygraphic Table Construction and Corpus Decoding")
    print("=" * 70)
    cmd_tachy_anchors()
    print("\n" + "=" * 70 + "\n")
    cmd_tachy_families()
    print("\n" + "=" * 70 + "\n")
    cmd_tachy_grid()
    print("\n" + "=" * 70 + "\n")
    cmd_tachy_decode()
    print("\n" + "=" * 70 + "\n")
    cmd_tachy_read()
    print("\n" + "=" * 70 + "\n")
    cmd_tachy_phrases()
    print("\n" + "=" * 70 + "\n")
    cmd_tachy_validate()
    print("\n" + "=" * 70 + "\n")
    cmd_phase20_integrate()


# ---------------------------------------------------------------------------
# Phase 21: Paleographic Sign Comparison
# ---------------------------------------------------------------------------

def cmd_paleo_ingest():
    """Run Phase 21.1: paleographic source normalization."""
    from voynich.phases.paleo_ingest import run_paleo_ingest
    t0 = time.time()
    run_paleo_ingest()
    print(f"\nPaleographic ingest completed in {time.time() - t0:.1f}s")


def cmd_fontana_families():
    """Run Phase 21.2: Fontana family extraction."""
    from voynich.phases.fontana_families import run_fontana_families
    t0 = time.time()
    run_fontana_families()
    print(f"\nFontana families completed in {time.time() - t0:.1f}s")


def cmd_chatelain_families():
    """Run Phase 21.3: Chatelain Bobbio family extraction."""
    from voynich.phases.chatelain_families import run_chatelain_families
    t0 = time.time()
    run_chatelain_families()
    print(f"\nChatelain families completed in {time.time() - t0:.1f}s")


def cmd_eva_compare():
    """Run Phase 21.4: EVA-to-historical stroke comparison."""
    from voynich.phases.eva_stroke_compare import run_eva_stroke_compare
    t0 = time.time()
    run_eva_stroke_compare()
    print(f"\nEVA stroke comparison completed in {time.time() - t0:.1f}s")


def cmd_family_syllable():
    """Run Phase 21.5: sign family to syllable mapping."""
    from voynich.phases.family_to_syllable import run_family_to_syllable
    t0 = time.time()
    run_family_to_syllable()
    print(f"\nFamily-to-syllable mapping completed in {time.time() - t0:.1f}s")


def cmd_cappelli_mod():
    """Run Phase 21.6: Cappelli modifier identification."""
    from voynich.phases.cappelli_modifier import run_cappelli_modifier
    t0 = time.time()
    run_cappelli_modifier()
    print(f"\nCappelli modifier analysis completed in {time.time() - t0:.1f}s")


def cmd_paleo_table():
    """Run Phase 21.7: paleographic table assembly."""
    from voynich.phases.paleo_table import run_paleo_table
    t0 = time.time()
    run_paleo_table()
    print(f"\nPaleographic table assembled in {time.time() - t0:.1f}s")


def cmd_paleo_decode():
    """Run Phase 21.8: corpus decode with paleographic table."""
    from voynich.phases.paleo_decode import run_paleo_decode
    t0 = time.time()
    run_paleo_decode()
    print(f"\nPaleographic decode completed in {time.time() - t0:.1f}s")


def cmd_paleo_validate():
    """Run Phase 21.9: 15-test validation battery."""
    from voynich.phases.paleo_validate import run_paleo_validate
    t0 = time.time()
    run_paleo_validate()
    print(f"\nPaleographic validation completed in {time.time() - t0:.1f}s")


def cmd_phase21_integrate():
    """Run Phase 21.10: integration and verdict."""
    from voynich.phases.phase21_integrate import run_phase21_integrate
    t0 = time.time()
    run_phase21_integrate()
    print(f"\nPhase 21 integration completed in {time.time() - t0:.1f}s")


def cmd_phase21():
    """Run full Phase 21 pipeline: paleographic sign comparison."""
    print("=" * 70)
    print("PHASE 21: Paleographic Sign Comparison")
    print("=" * 70)
    cmd_paleo_ingest()
    print("\n" + "=" * 70 + "\n")
    cmd_fontana_families()
    print("\n" + "=" * 70 + "\n")
    cmd_chatelain_families()
    print("\n" + "=" * 70 + "\n")
    cmd_eva_compare()
    print("\n" + "=" * 70 + "\n")
    cmd_family_syllable()
    print("\n" + "=" * 70 + "\n")
    cmd_cappelli_mod()
    print("\n" + "=" * 70 + "\n")
    cmd_paleo_table()
    print("\n" + "=" * 70 + "\n")
    cmd_paleo_decode()
    print("\n" + "=" * 70 + "\n")
    cmd_paleo_validate()
    print("\n" + "=" * 70 + "\n")
    cmd_phase21_integrate()


# ── Phase 22: First-Syllable Extraction & Fontana-Constrained Decode ──────

def cmd_first_syl():
    """Run Step 22.1: first-syllable extraction."""
    from voynich.phases.first_syllable import run_first_syllable
    t0 = time.time()
    run_first_syllable()
    print(f"\nStep 22.1 completed in {time.time() - t0:.1f}s")


def cmd_fontana_phon():
    """Run Step 22.2: Fontana phonetic mapping."""
    from voynich.phases.fontana_phonetic import run_fontana_phonetic
    t0 = time.time()
    run_fontana_phonetic()
    print(f"\nStep 22.2 completed in {time.time() - t0:.1f}s")


def cmd_table_merge():
    """Run Step 22.3: merge all evidence into decoding table."""
    from voynich.phases.table_merge import run_table_merge
    t0 = time.time()
    run_table_merge()
    print(f"\nStep 22.3 completed in {time.time() - t0:.1f}s")


def cmd_decode_22():
    """Run Step 22.4: corpus decode with Phase 22 table."""
    from voynich.phases.decode_22 import run_decode_22
    t0 = time.time()
    run_decode_22()
    print(f"\nStep 22.4 completed in {time.time() - t0:.1f}s")


def cmd_read_22():
    """Run Step 22.5: readability assessment (THE CRITICAL TEST)."""
    from voynich.phases.readability_22 import run_readability_22
    t0 = time.time()
    run_readability_22()
    print(f"\nStep 22.5 completed in {time.time() - t0:.1f}s")


def cmd_phrases_22():
    """Run Step 22.6: phrase detection and botanical cross-check."""
    from voynich.phases.phrases_22 import run_phrases_22
    t0 = time.time()
    run_phrases_22()
    print(f"\nStep 22.6 completed in {time.time() - t0:.1f}s")


def cmd_validate_22():
    """Run Step 22.7: 15-test validation battery."""
    from voynich.phases.validate_22 import run_validate_22
    t0 = time.time()
    run_validate_22()
    print(f"\nStep 22.7 completed in {time.time() - t0:.1f}s")


def cmd_phase22_integrate():
    """Run Step 22.8: integration and verdict."""
    from voynich.phases.phase22_integrate import run_phase22_integrate
    t0 = time.time()
    run_phase22_integrate()
    print(f"\nStep 22.8 completed in {time.time() - t0:.1f}s")


def cmd_phase22():
    """Run full Phase 22 pipeline: first-syllable extraction + Fontana decode."""
    print("=" * 70)
    print("PHASE 22: First-Syllable Extraction & Fontana-Constrained Decode")
    print("=" * 70)
    cmd_first_syl()
    print("\n" + "=" * 70 + "\n")
    cmd_fontana_phon()
    print("\n" + "=" * 70 + "\n")
    cmd_table_merge()
    print("\n" + "=" * 70 + "\n")
    cmd_decode_22()
    print("\n" + "=" * 70 + "\n")
    cmd_read_22()
    print("\n" + "=" * 70 + "\n")
    cmd_phrases_22()
    print("\n" + "=" * 70 + "\n")
    cmd_validate_22()
    print("\n" + "=" * 70 + "\n")
    cmd_phase22_integrate()


# ── Phase 23: Statistical Inversion Analysis ──────────────────────────────

def cmd_ceiling():
    """Run Step 23.1: theoretical ceiling analysis."""
    from voynich.phases.theoretical_ceiling import run_theoretical_ceiling
    t0 = time.time()
    run_theoretical_ceiling()
    print(f"\nStep 23.1 completed in {time.time() - t0:.1f}s")


def cmd_hist_invert():
    """Run Step 23.2: historical inversion mapping."""
    from voynich.phases.historical_inversion import run_historical_inversion
    t0 = time.time()
    run_historical_inversion()
    print(f"\nStep 23.2 completed in {time.time() - t0:.1f}s")


def cmd_bench_split():
    """Run Step 23.3: bench family split analysis."""
    from voynich.phases.bench_split import run_bench_split
    t0 = time.time()
    run_bench_split()
    print(f"\nStep 23.3 completed in {time.time() - t0:.1f}s")


def cmd_perm_search():
    """Run Step 23.4: permutation search."""
    from voynich.phases.permutation_search import run_permutation_search
    t0 = time.time()
    run_permutation_search()
    print(f"\nStep 23.4 completed in {time.time() - t0:.1f}s")


def cmd_read_delta():
    """Run Step 23.5: readability delta test."""
    from voynich.phases.readability_delta import run_readability_delta
    t0 = time.time()
    run_readability_delta()
    print(f"\nStep 23.5 completed in {time.time() - t0:.1f}s")


def cmd_phase23():
    """Run full Phase 23 pipeline: statistical inversion analysis."""
    print("=" * 70)
    print("PHASE 23: Statistical Inversion Analysis")
    print("=" * 70)
    cmd_ceiling()
    print("\n" + "=" * 70 + "\n")
    cmd_hist_invert()
    print("\n" + "=" * 70 + "\n")
    cmd_bench_split()
    print("\n" + "=" * 70 + "\n")
    cmd_perm_search()
    print("\n" + "=" * 70 + "\n")
    cmd_read_delta()


# ── Phase 24: Triple Sensitivity & Refinement ──────────────────────────────

def cmd_triple_loo():
    """Run Step 24.1: leave-one-out triple sensitivity analysis."""
    from voynich.phases.triple_sensitivity import run_triple_sensitivity
    t0 = time.time()
    run_triple_sensitivity()
    print(f"\nStep 24.1 completed in {time.time() - t0:.1f}s")


def cmd_error_id():
    """Run Step 24.2: error candidate identification."""
    from voynich.phases.error_candidates import run_error_candidates
    t0 = time.time()
    run_error_candidates()
    print(f"\nStep 24.2 completed in {time.time() - t0:.1f}s")


def cmd_triple_swap():
    """Run Step 24.3: exhaustive single-triple swap."""
    from voynich.phases.targeted_swap import run_targeted_swap
    t0 = time.time()
    run_targeted_swap()
    print(f"\nStep 24.3 completed in {time.time() - t0:.1f}s")


def cmd_bigram_val():
    """Run Step 24.4: bigram plausibility validation."""
    from voynich.phases.bigram_filter import run_bigram_filter
    t0 = time.time()
    run_bigram_filter()
    print(f"\nStep 24.4 completed in {time.time() - t0:.1f}s")


def cmd_corrected_tab():
    """Run Step 24.5: corrected table assembly."""
    from voynich.phases.corrected_table import run_corrected_table
    t0 = time.time()
    run_corrected_table()
    print(f"\nStep 24.5 completed in {time.time() - t0:.1f}s")


def cmd_corrected_decode():
    """Run Step 24.6: full corpus decode with corrected table."""
    from voynich.phases.corrected_decode import run_corrected_decode
    t0 = time.time()
    run_corrected_decode()
    print(f"\nStep 24.6 completed in {time.time() - t0:.1f}s")


def cmd_corrected_read():
    """Run Step 24.7: readability battery on corrected decode."""
    from voynich.phases.corrected_readability import run_corrected_readability
    t0 = time.time()
    run_corrected_readability()
    print(f"\nStep 24.7 completed in {time.time() - t0:.1f}s")


def cmd_word_bound():
    """Run Step 24.8: word boundary re-analysis."""
    from voynich.phases.word_boundary import run_word_boundary
    t0 = time.time()
    run_word_boundary()
    print(f"\nStep 24.8 completed in {time.time() - t0:.1f}s")


def cmd_ligature_test():
    """Run Step 24.9: EVA ligature hypothesis test."""
    from voynich.phases.ligature_test import run_ligature_test
    t0 = time.time()
    run_ligature_test()
    print(f"\nStep 24.9 completed in {time.time() - t0:.1f}s")


def cmd_direction():
    """Run Step 24.10: reading direction analysis."""
    from voynich.phases.directionality import run_directionality
    t0 = time.time()
    run_directionality()
    print(f"\nStep 24.10 completed in {time.time() - t0:.1f}s")


def cmd_crib_search():
    """Run Step 24.11: known-plaintext crib search."""
    from voynich.phases.known_text_search import run_known_text_search
    t0 = time.time()
    run_known_text_search()
    print(f"\nStep 24.11 completed in {time.time() - t0:.1f}s")


def cmd_folio_deep():
    """Run Step 24.12: single-folio deep decode."""
    from voynich.phases.folio_isolation import run_folio_isolation
    t0 = time.time()
    run_folio_isolation()
    print(f"\nStep 24.12 completed in {time.time() - t0:.1f}s")


def cmd_section_xfer():
    """Run Step 24.13: section-trained decode transfer test."""
    from voynich.phases.cross_section import run_cross_section
    t0 = time.time()
    run_cross_section()
    print(f"\nStep 24.13 completed in {time.time() - t0:.1f}s")


def cmd_reverse_eng():
    """Run Step 24.14: reverse-engineer from confirmed words."""
    from voynich.phases.reverse_engineer import run_reverse_engineer
    t0 = time.time()
    run_reverse_engineer()
    print(f"\nStep 24.14 completed in {time.time() - t0:.1f}s")


def cmd_token_gram():
    """Run Step 24.15: Voynich word grammar exploitation."""
    from voynich.phases.token_grammar import run_token_grammar
    t0 = time.time()
    run_token_grammar()
    print(f"\nStep 24.15 completed in {time.time() - t0:.1f}s")


def cmd_phase24_integrate():
    """Run Step 24.16: Phase 24 integration."""
    from voynich.phases.phase24_integrate import run_phase24_integrate
    t0 = time.time()
    run_phase24_integrate()
    print(f"\nStep 24.16 completed in {time.time() - t0:.1f}s")


def cmd_boustro():
    """Run Step 25.1: boustrophedon re-ordering test."""
    from voynich.phases.boustrophedon import run_boustrophedon
    t0 = time.time()
    run_boustrophedon()
    print(f"\nStep 25.1 completed in {time.time() - t0:.1f}s")


def cmd_f6r_exam():
    """Run Step 25.2: folio f6r manual examination."""
    from voynich.phases.f6r_manual import run_f6r_manual
    t0 = time.time()
    run_f6r_manual()
    print(f"\nStep 25.2 completed in {time.time() - t0:.1f}s")


def cmd_phase25_verdict():
    """Run Step 25.3: combined Phase 25 verdict."""
    from voynich.phases.phase25_verdict import run_phase25_verdict
    t0 = time.time()
    run_phase25_verdict()
    print(f"\nStep 25.3 completed in {time.time() - t0:.1f}s")


def cmd_phase25():
    """Run full Phase 25 pipeline: reading direction test + f6r examination."""
    print("=" * 70)
    print("PHASE 25: Reading Direction Test and Folio f6r Examination")
    print("=" * 70)
    cmd_boustro()
    print("\n" + "=" * 70 + "\n")
    cmd_f6r_exam()
    print("\n" + "=" * 70 + "\n")
    cmd_phase25_verdict()


# -----------------------------------------------------------------------
# Phase 26: Zodiac Known-Plaintext Attack
# -----------------------------------------------------------------------

def cmd_zodiac_map():
    """Run Step 26.1: zodiac folio mapping and label catalog."""
    from voynich.phases.zodiac_map import run_zodiac_map
    t0 = time.time()
    run_zodiac_map()
    print(f"\nStep 26.1 completed in {time.time() - t0:.1f}s")


def cmd_month_crib():
    """Run Step 26.2: multi-language month name crib analysis."""
    from voynich.phases.month_crib import run_month_crib
    t0 = time.time()
    run_month_crib()
    print(f"\nStep 26.2 completed in {time.time() - t0:.1f}s")


def cmd_astro_crib():
    """Run Step 26.3: full zodiac description crib search."""
    from voynich.phases.astro_crib import run_astro_crib
    t0 = time.time()
    run_astro_crib()
    print(f"\nStep 26.3 completed in {time.time() - t0:.1f}s")


def cmd_label_decode():
    """Run Step 26.4: per-label exhaustive CSP decode."""
    from voynich.phases.zodiac_label_decode import run_label_decode
    t0 = time.time()
    run_label_decode()
    print(f"\nStep 26.4 completed in {time.time() - t0:.1f}s")


def cmd_zodiac_tab():
    """Run Step 26.5: zodiac-derived assignment table."""
    from voynich.phases.zodiac_table import run_zodiac_table
    t0 = time.time()
    run_zodiac_table()
    print(f"\nStep 26.5 completed in {time.time() - t0:.1f}s")


def cmd_zodiac_decode():
    """Run Step 26.6: full corpus decode with zodiac-merged table."""
    from voynich.phases.zodiac_decode import run_zodiac_decode
    t0 = time.time()
    run_zodiac_decode()
    print(f"\nStep 26.6 completed in {time.time() - t0:.1f}s")


def cmd_phase26_validate():
    """Run Step 26.7: Phase 26 validation battery (V1-V12)."""
    from voynich.phases.phase26_validate import run_phase26_validate
    t0 = time.time()
    run_phase26_validate()
    print(f"\nStep 26.7 completed in {time.time() - t0:.1f}s")


def cmd_phase26_verdict():
    """Run Step 26.8: Phase 26 verdict."""
    from voynich.phases.phase26_verdict import run_phase26_verdict
    t0 = time.time()
    run_phase26_verdict()
    print(f"\nStep 26.8 completed in {time.time() - t0:.1f}s")


def cmd_phase26():
    """Run full Phase 26 pipeline: Zodiac Known-Plaintext Attack."""
    print("=" * 70)
    print("PHASE 26: Zodiac Known-Plaintext Attack")
    print("=" * 70)
    cmd_zodiac_map()
    print("\n" + "=" * 70 + "\n")
    cmd_month_crib()
    print("\n" + "=" * 70 + "\n")
    cmd_astro_crib()
    print("\n" + "=" * 70 + "\n")
    cmd_label_decode()
    print("\n" + "=" * 70 + "\n")
    cmd_zodiac_tab()
    print("\n" + "=" * 70 + "\n")
    cmd_zodiac_decode()
    print("\n" + "=" * 70 + "\n")
    cmd_phase26_validate()
    print("\n" + "=" * 70 + "\n")
    cmd_phase26_verdict()


def cmd_gibberish_test():
    """Run Step 27.1: gibberish typology control test."""
    from voynich.phases.gibberish_typology import run_gibberish_typology
    t0 = time.time()
    run_gibberish_typology()
    print(f"\nStep 27.1 completed in {time.time() - t0:.1f}s")


def cmd_naibbe_test():
    """Run Step 27.2: Naibbe dice cipher entropy test."""
    from voynich.phases.naibbe_entropy import run_naibbe_entropy
    t0 = time.time()
    run_naibbe_entropy()
    print(f"\nStep 27.2 completed in {time.time() - t0:.1f}s")


def cmd_phase27_verdict():
    """Run Step 27.3: Phase 27 peer review verdict."""
    from voynich.phases.phase27_verdict import run_phase27_verdict
    t0 = time.time()
    run_phase27_verdict()
    print(f"\nStep 27.3 completed in {time.time() - t0:.1f}s")


def cmd_phase27():
    """Run full Phase 27 pipeline: Peer Review Controls."""
    print("=" * 70)
    print("PHASE 27: Peer Review Controls")
    print("=" * 70)
    cmd_gibberish_test()
    print("\n" + "=" * 70 + "\n")
    cmd_naibbe_test()
    print("\n" + "=" * 70 + "\n")
    cmd_phase27_verdict()


def cmd_crib_extract():
    """Run Step 28.1: crib extraction from confirmed words."""
    from voynich.phases.crib_extraction import run_crib_extraction
    t0 = time.time()
    run_crib_extraction()
    print(f"\nStep 28.1 completed in {time.time() - t0:.1f}s")


def cmd_crib_consist():
    """Run Step 28.2: internal consistency test."""
    from voynich.phases.consistency_check import run_consistency_check
    t0 = time.time()
    run_consistency_check()
    print(f"\nStep 28.2 completed in {time.time() - t0:.1f}s")


def cmd_family_prop():
    """Run Step 28.3: family propagation."""
    from voynich.phases.family_propagation import run_family_propagation
    t0 = time.time()
    run_family_propagation()
    print(f"\nStep 28.3 completed in {time.time() - t0:.1f}s")


def cmd_signal_iso():
    """Run Step 28.4: signal isolation."""
    from voynich.phases.signal_isolation import run_signal_isolation
    t0 = time.time()
    run_signal_isolation()
    print(f"\nStep 28.4 completed in {time.time() - t0:.1f}s")


def cmd_crib_local():
    """Run Step 28.5: crib localization."""
    from voynich.phases.crib_localization import run_crib_localization
    t0 = time.time()
    run_crib_localization()
    print(f"\nStep 28.5 completed in {time.time() - t0:.1f}s")


def cmd_ventris_tab():
    """Run Step 28.6: Ventris table assembly."""
    from voynich.phases.ventris_table import run_ventris_table
    t0 = time.time()
    run_ventris_table()
    print(f"\nStep 28.6 completed in {time.time() - t0:.1f}s")


def cmd_ventris_decode():
    """Run Step 28.7: Ventris corpus decode."""
    from voynich.phases.ventris_decode import run_ventris_decode
    t0 = time.time()
    run_ventris_decode()
    print(f"\nStep 28.7 completed in {time.time() - t0:.1f}s")


def cmd_ventris_read():
    """Run Step 28.8: Ventris readability battery."""
    from voynich.phases.ventris_readability import run_ventris_readability
    t0 = time.time()
    run_ventris_readability()
    print(f"\nStep 28.8 completed in {time.time() - t0:.1f}s")


def cmd_phase28_verdict():
    """Run Step 28.9: Phase 28 verdict."""
    from voynich.phases.phase28_verdict import run_phase28_verdict
    t0 = time.time()
    run_phase28_verdict()
    print(f"\nStep 28.9 completed in {time.time() - t0:.1f}s")


def cmd_phase28():
    """Run full Phase 28 pipeline: Ventris-Style Crib Propagation."""
    print("=" * 70)
    print("PHASE 28: Ventris-Style Crib Propagation and Signal Isolation")
    print("=" * 70)
    cmd_crib_extract()
    print("\n" + "=" * 70 + "\n")
    cmd_crib_consist()
    print("\n" + "=" * 70 + "\n")
    cmd_family_prop()
    print("\n" + "=" * 70 + "\n")
    cmd_signal_iso()
    print("\n" + "=" * 70 + "\n")
    cmd_crib_local()
    print("\n" + "=" * 70 + "\n")
    cmd_ventris_tab()
    print("\n" + "=" * 70 + "\n")
    cmd_ventris_decode()
    print("\n" + "=" * 70 + "\n")
    cmd_ventris_read()
    print("\n" + "=" * 70 + "\n")
    cmd_phase28_verdict()


def cmd_signal_bigram():
    """Run Step 29.1: signal-filtered bigram plausibility."""
    from voynich.phases.signal_bigrams import run_signal_bigrams
    t0 = time.time()
    run_signal_bigrams()
    print(f"\nStep 29.1 completed in {time.time() - t0:.1f}s")


def cmd_signal_context():
    """Run Step 29.2: context of confirmed signal words."""
    from voynich.phases.signal_context import run_signal_context
    t0 = time.time()
    run_signal_context()
    print(f"\nStep 29.2 completed in {time.time() - t0:.1f}s")


def cmd_signal_folio():
    """Run Step 29.3: signal folio deep examination."""
    from voynich.phases.signal_folio_read import run_signal_folio_read
    t0 = time.time()
    run_signal_folio_read()
    print(f"\nStep 29.3 completed in {time.time() - t0:.1f}s")


def cmd_signal_phrase():
    """Run Step 29.4: signal phrase extraction."""
    from voynich.phases.signal_phrases import run_signal_phrases
    t0 = time.time()
    run_signal_phrases()
    print(f"\nStep 29.4 completed in {time.time() - t0:.1f}s")


def cmd_phase29_verdict():
    """Run Step 29.5: Phase 29 verdict."""
    from voynich.phases.phase29_verdict import run_phase29_verdict
    t0 = time.time()
    run_phase29_verdict()
    print(f"\nStep 29.5 completed in {time.time() - t0:.1f}s")


def cmd_phase29():
    """Run full Phase 29 pipeline: Signal-Filtered Readability."""
    print("=" * 70)
    print("PHASE 29: Signal-Filtered Readability and Context Exploitation")
    print("=" * 70)
    cmd_signal_bigram()
    print("\n" + "=" * 70 + "\n")
    cmd_signal_context()
    print("\n" + "=" * 70 + "\n")
    cmd_signal_folio()
    print("\n" + "=" * 70 + "\n")
    cmd_signal_phrase()
    print("\n" + "=" * 70 + "\n")
    cmd_phase29_verdict()


# ── Phase 30: Iterative Ventris Bootstrap ──

def cmd_bootstrap():
    """Run Step 30.1: Iterative Ventris bootstrap loop."""
    from voynich.phases.bootstrap_loop import run_bootstrap_loop
    t0 = time.time()
    run_bootstrap_loop()
    print(f"\nStep 30.1 completed in {time.time() - t0:.1f}s")


def cmd_boot_signal():
    """Run Step 30.2: Re-isolate signal post-bootstrap."""
    from voynich.phases.bootstrap_signal import run_bootstrap_signal
    t0 = time.time()
    run_bootstrap_signal()
    print(f"\nStep 30.2 completed in {time.time() - t0:.1f}s")


def cmd_boot_bigram():
    """Run Step 30.3: Re-run bigram plausibility post-bootstrap."""
    from voynich.phases.bootstrap_bigrams import run_bootstrap_bigrams
    t0 = time.time()
    run_bootstrap_bigrams()
    print(f"\nStep 30.3 completed in {time.time() - t0:.1f}s")


def cmd_boot_context():
    """Run Step 30.4: Re-run context analysis post-bootstrap."""
    from voynich.phases.bootstrap_context import run_bootstrap_context
    t0 = time.time()
    run_bootstrap_context()
    print(f"\nStep 30.4 completed in {time.time() - t0:.1f}s")


def cmd_boot_folio():
    """Run Step 30.5: Annotated folio examination post-bootstrap."""
    from voynich.phases.bootstrap_folio import run_bootstrap_folio
    t0 = time.time()
    run_bootstrap_folio()
    print(f"\nStep 30.5 completed in {time.time() - t0:.1f}s")


def cmd_boot_read():
    """Run Step 30.6: Full readability battery post-bootstrap."""
    from voynich.phases.bootstrap_readability import run_bootstrap_readability
    t0 = time.time()
    run_bootstrap_readability()
    print(f"\nStep 30.6 completed in {time.time() - t0:.1f}s")


def cmd_phase30_verdict():
    """Run Step 30.7: Phase 30 verdict and convergence analysis."""
    from voynich.phases.phase30_verdict import run_phase30_verdict
    t0 = time.time()
    run_phase30_verdict()
    print(f"\nStep 30.7 completed in {time.time() - t0:.1f}s")


def cmd_phase30():
    """Run full Phase 30 pipeline: Iterative Ventris Bootstrap."""
    print("=" * 70)
    print("PHASE 30: Iterative Ventris Bootstrap")
    print("=" * 70)
    cmd_bootstrap()
    print("\n" + "=" * 70 + "\n")
    cmd_boot_signal()
    print("\n" + "=" * 70 + "\n")
    cmd_boot_bigram()
    print("\n" + "=" * 70 + "\n")
    cmd_boot_context()
    print("\n" + "=" * 70 + "\n")
    cmd_boot_folio()
    print("\n" + "=" * 70 + "\n")
    cmd_boot_read()
    print("\n" + "=" * 70 + "\n")
    cmd_phase30_verdict()


# ── Phase 31: Botanical Anchors + Structural Reframing ──

def cmd_consensus_plants():
    """Run Step 31.1: Consensus plant identification."""
    from voynich.phases.consensus_plants import run_consensus_plants
    t0 = time.time()
    run_consensus_plants()
    print(f"\nStep 31.1 completed in {time.time() - t0:.1f}s")


def cmd_plant_csp():
    """Run Step 31.2: Plant name CSP on folio labels."""
    from voynich.phases.plant_csp import run_plant_csp
    t0 = time.time()
    run_plant_csp()
    print(f"\nStep 31.2 completed in {time.time() - t0:.1f}s")


def cmd_plant_prop():
    """Run Step 31.3: Plant-derived assignment propagation."""
    from voynich.phases.plant_propagate import run_plant_propagate
    t0 = time.time()
    run_plant_propagate()
    print(f"\nStep 31.3 completed in {time.time() - t0:.1f}s")


def cmd_bot_signal():
    """Run Step 31.4: Botanical signal validation."""
    from voynich.phases.botanical_signal import run_botanical_signal
    t0 = time.time()
    run_botanical_signal()
    print(f"\nStep 31.4 completed in {time.time() - t0:.1f}s")


def cmd_determ_test():
    """Run Step 31.5: Gallows as determinatives test."""
    from voynich.phases.determinative_test import run_determinative_test
    t0 = time.time()
    run_determinative_test()
    print(f"\nStep 31.5 completed in {time.time() - t0:.1f}s")


def cmd_compound_test():
    """Run Step 31.6: Compound sign hypothesis test."""
    from voynich.phases.compound_sign_test import run_compound_sign
    t0 = time.time()
    run_compound_sign()
    print(f"\nStep 31.6 completed in {time.time() - t0:.1f}s")


def cmd_interleave_test():
    """Run Step 31.7: Interleaved text separation test."""
    from voynich.phases.interleaved_test import run_interleaved_test
    t0 = time.time()
    run_interleaved_test()
    print(f"\nStep 31.7 completed in {time.time() - t0:.1f}s")


def cmd_reseg_test():
    """Run Step 31.8: EVA re-segmentation test."""
    from voynich.phases.resegmentation_test import run_resegmentation_test
    t0 = time.time()
    run_resegmentation_test()
    print(f"\nStep 31.8 completed in {time.time() - t0:.1f}s")


def cmd_phase31_integrate():
    """Run Step 31.9: Phase 31 integration."""
    from voynich.phases.phase31_integrate import run_phase31_integrate
    t0 = time.time()
    run_phase31_integrate()
    print(f"\nStep 31.9 completed in {time.time() - t0:.1f}s")


def cmd_phase31():
    """Run full Phase 31 pipeline: Botanical Anchors + Structural Reframing."""
    print("=" * 70)
    print("PHASE 31: Botanical Anchors + Structural Reframing")
    print("=" * 70)
    # Path 2: Botanical Anchors
    print("\n" + "=" * 70)
    print("PATH 2: Botanical Anchor Attack")
    print("=" * 70)
    cmd_consensus_plants()
    print("\n" + "=" * 70 + "\n")
    cmd_plant_csp()
    print("\n" + "=" * 70 + "\n")
    cmd_plant_prop()
    print("\n" + "=" * 70 + "\n")
    cmd_bot_signal()
    # Path 4: Structural Reframing
    print("\n" + "=" * 70)
    print("PATH 4: Structural Reframing")
    print("=" * 70)
    cmd_determ_test()
    print("\n" + "=" * 70 + "\n")
    cmd_compound_test()
    print("\n" + "=" * 70 + "\n")
    cmd_interleave_test()
    print("\n" + "=" * 70 + "\n")
    cmd_reseg_test()
    # Integration
    print("\n" + "=" * 70)
    print("INTEGRATION")
    print("=" * 70)
    cmd_phase31_integrate()


    # ── Phase 32: Compound-Sign Signal Pipeline ──

def cmd_comp_decode():
    """Run Step 32.1: Compound-sign corpus decode."""
    from voynich.phases.compound_decode import run_compound_decode
    t0 = time.time()
    run_compound_decode()
    print(f"\nStep 32.1 completed in {time.time() - t0:.1f}s")


def cmd_comp_signal():
    """Run Step 32.2: Compound signal classification."""
    from voynich.phases.compound_signal import run_compound_signal
    t0 = time.time()
    run_compound_signal()
    print(f"\nStep 32.2 completed in {time.time() - t0:.1f}s")


def cmd_comp_bigram():
    """Run Step 32.3: Compound bigram plausibility."""
    from voynich.phases.compound_bigrams import run_compound_bigrams
    t0 = time.time()
    run_compound_bigrams()
    print(f"\nStep 32.3 completed in {time.time() - t0:.1f}s")


def cmd_comp_context():
    """Run Step 32.4: Compound context analysis."""
    from voynich.phases.compound_context import run_compound_context
    t0 = time.time()
    run_compound_context()
    print(f"\nStep 32.4 completed in {time.time() - t0:.1f}s")


def cmd_comp_bootstrap():
    """Run Step 32.5: Compound bootstrap loop."""
    from voynich.phases.compound_bootstrap import run_compound_bootstrap
    t0 = time.time()
    run_compound_bootstrap()
    print(f"\nStep 32.5 completed in {time.time() - t0:.1f}s")


def cmd_comp_folio():
    """Run Step 32.6: Compound folio annotations."""
    from voynich.phases.compound_folio import run_compound_folio
    t0 = time.time()
    run_compound_folio()
    print(f"\nStep 32.6 completed in {time.time() - t0:.1f}s")


def cmd_comp_read():
    """Run Step 32.7: Compound readability battery."""
    from voynich.phases.compound_readability import run_compound_readability
    t0 = time.time()
    run_compound_readability()
    print(f"\nStep 32.7 completed in {time.time() - t0:.1f}s")


def cmd_phase32_verdict():
    """Run Step 32.8: Phase 32 verdict."""
    from voynich.phases.phase32_verdict import run_phase32_verdict
    t0 = time.time()
    run_phase32_verdict()
    print(f"\nStep 32.8 completed in {time.time() - t0:.1f}s")


def cmd_phase32():
    """Run full Phase 32 pipeline: Compound-Sign Signal Pipeline."""
    print("=" * 70)
    print("PHASE 32: Compound-Sign Signal Pipeline")
    print("=" * 70)
    cmd_comp_decode()
    print("\n" + "=" * 70 + "\n")
    cmd_comp_signal()
    print("\n" + "=" * 70 + "\n")
    cmd_comp_bigram()
    print("\n" + "=" * 70 + "\n")
    cmd_comp_context()
    print("\n" + "=" * 70 + "\n")
    cmd_comp_bootstrap()
    print("\n" + "=" * 70 + "\n")
    cmd_comp_folio()
    print("\n" + "=" * 70 + "\n")
    cmd_comp_read()
    print("\n" + "=" * 70 + "\n")
    cmd_phase32_verdict()


    # ── Phase 33: Multi-Vector Error Correction ──

def cmd_anti_diag():
    """Run Step 33.1: Anti-signal diagnosis."""
    from voynich.phases.anti_signal_diagnosis import run_anti_signal_diagnosis
    t0 = time.time()
    run_anti_signal_diagnosis()
    print(f"\nStep 33.1 completed in {time.time() - t0:.1f}s")


def cmd_triple_rates():
    """Run Step 33.2: Per-triple SIGNAL rate analysis."""
    from voynich.phases.triple_signal_rates import run_triple_signal_rates
    t0 = time.time()
    run_triple_signal_rates()
    print(f"\nStep 33.2 completed in {time.time() - t0:.1f}s")


def cmd_signal_swap():
    """Run Step 33.3: SIGNAL-guided triple swap."""
    from voynich.phases.signal_guided_swap import run_signal_guided_swap
    t0 = time.time()
    run_signal_guided_swap()
    print(f"\nStep 33.3 completed in {time.time() - t0:.1f}s")


def cmd_signal_correct():
    """Run Step 33.4: Corrected table decode and validation."""
    from voynich.phases.signal_corrected_decode import run_signal_corrected_decode
    t0 = time.time()
    run_signal_corrected_decode()
    print(f"\nStep 33.4 completed in {time.time() - t0:.1f}s")


def cmd_latin_lm():
    """Run Step 33.5: Latin character language model."""
    from voynich.phases.latin_lm import run_latin_lm
    t0 = time.time()
    run_latin_lm()
    print(f"\nStep 33.5 completed in {time.time() - t0:.1f}s")


def cmd_ppl_search():
    """Run Step 33.6: Perplexity-optimal triple search."""
    from voynich.phases.perplexity_search import run_perplexity_search
    t0 = time.time()
    run_perplexity_search()
    print(f"\nStep 33.6 completed in {time.time() - t0:.1f}s")


def cmd_ppl_validate():
    """Run Step 33.7: Perplexity vs SIGNAL cross-validation."""
    from voynich.phases.perplexity_validate import run_perplexity_validate
    t0 = time.time()
    run_perplexity_validate()
    print(f"\nStep 33.7 completed in {time.time() - t0:.1f}s")


def cmd_suffix_gram():
    """Run Step 33.8: Suffix-to-grammar mapping."""
    from voynich.phases.suffix_grammar import run_suffix_grammar
    t0 = time.time()
    run_suffix_grammar()
    print(f"\nStep 33.8 completed in {time.time() - t0:.1f}s")


def cmd_suffix_search():
    """Run Step 33.9: Suffix-constrained triple search."""
    from voynich.phases.suffix_constrained_search import run_suffix_constrained_search
    t0 = time.time()
    run_suffix_constrained_search()
    print(f"\nStep 33.9 completed in {time.time() - t0:.1f}s")


def cmd_long_crib():
    """Run Step 33.10: Long botanical crib target selection."""
    from voynich.phases.long_crib_targets import run_long_crib_targets
    t0 = time.time()
    run_long_crib_targets()
    print(f"\nStep 33.10 completed in {time.time() - t0:.1f}s")


def cmd_long_csp():
    """Run Step 33.11: Long crib exhaustive alignment."""
    from voynich.phases.long_crib_csp import run_long_crib_csp
    t0 = time.time()
    run_long_crib_csp()
    print(f"\nStep 33.11 completed in {time.time() - t0:.1f}s")


def cmd_long_prop():
    """Run Step 33.12: Long crib assignment propagation."""
    from voynich.phases.long_crib_propagate import run_long_crib_propagate
    t0 = time.time()
    run_long_crib_propagate()
    print(f"\nStep 33.12 completed in {time.time() - t0:.1f}s")


def cmd_pair_freq():
    """Run Step 33.13: Token-pair frequency tables."""
    from voynich.phases.token_pair_freq import run_token_pair_freq
    t0 = time.time()
    run_token_pair_freq()
    print(f"\nStep 33.13 completed in {time.time() - t0:.1f}s")


def cmd_distrib_match():
    """Run Step 33.14: Distributional token-to-word matching."""
    from voynich.phases.distributional_match import run_distributional_match
    t0 = time.time()
    run_distributional_match()
    print(f"\nStep 33.14 completed in {time.time() - t0:.1f}s")


def cmd_distrib_validate():
    """Run Step 33.15: Distributional validation."""
    from voynich.phases.distributional_validate import run_distributional_validate
    t0 = time.time()
    run_distributional_validate()
    print(f"\nStep 33.15 completed in {time.time() - t0:.1f}s")


def cmd_phase33_integrate():
    """Run Step 33.16: Phase 33 integration."""
    from voynich.phases.phase33_integrate import run_phase33_integrate
    t0 = time.time()
    run_phase33_integrate()
    print(f"\nStep 33.16 completed in {time.time() - t0:.1f}s")


def cmd_phase33():
    """Run full Phase 33 pipeline: Multi-Vector Error Correction."""
    print("=" * 70)
    print("PHASE 33: Multi-Vector Error Correction and Orthogonal Attack")
    print("=" * 70)
    # Approach 1+2: Anti-Signal Diagnosis and Signal-Guided Swap
    print("\n" + "=" * 70)
    print("APPROACH 1+2: Anti-Signal Diagnosis and Signal-Guided Swap")
    print("=" * 70)
    cmd_anti_diag()
    print("\n" + "=" * 70 + "\n")
    cmd_triple_rates()
    print("\n" + "=" * 70 + "\n")
    cmd_signal_swap()
    print("\n" + "=" * 70 + "\n")
    cmd_signal_correct()
    # Approach 3: Latin Perplexity Optimization
    print("\n" + "=" * 70)
    print("APPROACH 3: Latin Perplexity Optimization")
    print("=" * 70)
    cmd_latin_lm()
    print("\n" + "=" * 70 + "\n")
    cmd_ppl_search()
    print("\n" + "=" * 70 + "\n")
    cmd_ppl_validate()
    # Approach 4: Suffix-Constrained Root Search
    print("\n" + "=" * 70)
    print("APPROACH 4: Suffix-Constrained Root Search")
    print("=" * 70)
    cmd_suffix_gram()
    print("\n" + "=" * 70 + "\n")
    cmd_suffix_search()
    # Approach 5: Long Botanical Crib Attack
    print("\n" + "=" * 70)
    print("APPROACH 5: Long Botanical Crib Attack")
    print("=" * 70)
    cmd_long_crib()
    print("\n" + "=" * 70 + "\n")
    cmd_long_csp()
    print("\n" + "=" * 70 + "\n")
    cmd_long_prop()
    # Approach 6: Token-Pair Distributional Isomorphism
    print("\n" + "=" * 70)
    print("APPROACH 6: Token-Pair Distributional Isomorphism")
    print("=" * 70)
    cmd_pair_freq()
    print("\n" + "=" * 70 + "\n")
    cmd_distrib_match()
    print("\n" + "=" * 70 + "\n")
    cmd_distrib_validate()
    # Integration
    print("\n" + "=" * 70)
    print("INTEGRATION")
    print("=" * 70)
    cmd_phase33_integrate()


# ── Phase 34: Encoding Model Reformation ──

def cmd_dict_cal():
    """Run Step 34.18: Dictionary right-sizing (Track G)."""
    from voynich.phases.dict_calibration import run_dict_calibration
    t0 = time.time()
    run_dict_calibration()
    print(f"\nStep 34.18 completed in {time.time() - t0:.1f}s")


def cmd_sigla_dict():
    """Run Step 34.1: Medieval abbreviation dictionary (Track A)."""
    from voynich.phases.sigla_dictionary import run_sigla_dictionary
    t0 = time.time()
    run_sigla_dictionary()
    print(f"\nStep 34.1 completed in {time.time() - t0:.1f}s")


def cmd_abjad_csp():
    """Run Step 34.2: Abjad consonant-only CSP (Track A)."""
    from voynich.phases.abjad_csp import run_abjad_csp
    t0 = time.time()
    run_abjad_csp()
    print(f"\nStep 34.2 completed in {time.time() - t0:.1f}s")


def cmd_sigla_decode():
    """Run Step 34.3: Sigla-specific decode (Track A)."""
    from voynich.phases.sigla_decode import run_sigla_decode
    t0 = time.time()
    run_sigla_decode()
    print(f"\nStep 34.3 completed in {time.time() - t0:.1f}s")


def cmd_abjad_signal():
    """Run Step 34.4: Abjad signal isolation (Track A)."""
    from voynich.phases.abjad_signal import run_abjad_signal
    t0 = time.time()
    run_abjad_signal()
    print(f"\nStep 34.4 completed in {time.time() - t0:.1f}s")


def cmd_slot_vars():
    """Run Step 34.5: Slot-conditioned variable fork (Track B)."""
    from voynich.phases.slot_variables import run_slot_variables
    t0 = time.time()
    run_slot_variables()
    print(f"\nStep 34.5 completed in {time.time() - t0:.1f}s")


def cmd_slot_csp():
    """Run Step 34.6: Position-conditioned CSP solve (Track B)."""
    from voynich.phases.slot_csp import run_slot_csp
    t0 = time.time()
    run_slot_csp()
    print(f"\nStep 34.6 completed in {time.time() - t0:.1f}s")


def cmd_slot_signal():
    """Run Step 34.7: Slot-conditioned signal isolation (Track B)."""
    from voynich.phases.slot_signal import run_slot_signal
    t0 = time.time()
    run_slot_signal()
    print(f"\nStep 34.7 completed in {time.time() - t0:.1f}s")


def cmd_mixed_lm():
    """Run Step 34.8: Mixed Latin-Italian language model (Track C)."""
    from voynich.phases.mixed_lm import run_mixed_lm
    t0 = time.time()
    run_mixed_lm()
    print(f"\nStep 34.8 completed in {time.time() - t0:.1f}s")


def cmd_dialect_decode():
    """Run Step 34.9: Dialect-conditioned decode (Track C)."""
    from voynich.phases.dialect_decode import run_dialect_decode
    t0 = time.time()
    run_dialect_decode()
    print(f"\nStep 34.9 completed in {time.time() - t0:.1f}s")


def cmd_dialect_signal():
    """Run Step 34.10: Dialect signal isolation (Track C)."""
    from voynich.phases.dialect_signal import run_dialect_signal
    t0 = time.time()
    run_dialect_signal()
    print(f"\nStep 34.10 completed in {time.time() - t0:.1f}s")


def cmd_continua():
    """Run Step 34.11: Space stripping and character stream (Track D)."""
    from voynich.phases.continua_stream import run_continua_stream
    t0 = time.time()
    run_continua_stream()
    print(f"\nStep 34.11 completed in {time.time() - t0:.1f}s")


def cmd_reseg_decode():
    """Run Step 34.12: Viterbi re-segmentation (Track D)."""
    from voynich.phases.resegment_decode import run_resegment_decode
    t0 = time.time()
    run_resegment_decode()
    print(f"\nStep 34.12 completed in {time.time() - t0:.1f}s")


def cmd_reseg_signal():
    """Run Step 34.13: Re-segmented signal isolation (Track D)."""
    from voynich.phases.resegment_signal import run_resegment_signal
    t0 = time.time()
    run_resegment_signal()
    print(f"\nStep 34.13 completed in {time.time() - t0:.1f}s")


def cmd_gallows_geom():
    """Run Step 34.14: Gallows-bench spatial geometry (Track E)."""
    from voynich.phases.gallows_geometry import run_gallows_geometry
    t0 = time.time()
    run_gallows_geometry()
    print(f"\nStep 34.14 completed in {time.time() - t0:.1f}s")


def cmd_spatial_decode():
    """Run Step 34.15: Spatial-tagged decode (Track E)."""
    from voynich.phases.spatial_decode import run_spatial_decode
    t0 = time.time()
    run_spatial_decode()
    print(f"\nStep 34.15 completed in {time.time() - t0:.1f}s")


def cmd_vowel_ptr():
    """Run Step 34.16: Vowel pointer hypothesis test (Track F)."""
    from voynich.phases.vowel_pointer import run_vowel_pointer
    t0 = time.time()
    run_vowel_pointer()
    print(f"\nStep 34.16 completed in {time.time() - t0:.1f}s")


def cmd_vowel_decode():
    """Run Step 34.17: Vowel-pointed decode (Track F)."""
    from voynich.phases.vowel_decode import run_vowel_decode
    t0 = time.time()
    run_vowel_decode()
    print(f"\nStep 34.17 completed in {time.time() - t0:.1f}s")


def cmd_phase34_integrate():
    """Run Step 34.19: Phase 34 integration."""
    from voynich.phases.phase34_integrate import run_phase34_integrate
    t0 = time.time()
    run_phase34_integrate()
    print(f"\nStep 34.19 completed in {time.time() - t0:.1f}s")


def cmd_phase34():
    """Run full Phase 34 pipeline: Encoding Model Reformation."""
    print("=" * 70)
    print("PHASE 34: Encoding Model Reformation")
    print("=" * 70)
    # Track G: Dictionary Right-Sizing
    print("\n" + "=" * 70)
    print("TRACK G: Dictionary Right-Sizing")
    print("=" * 70)
    cmd_dict_cal()
    # Track A: Abbreviated Latin / Abjad Attack
    print("\n" + "=" * 70)
    print("TRACK A: Abbreviated Latin / Abjad Attack")
    print("=" * 70)
    cmd_sigla_dict()
    print("\n" + "=" * 70 + "\n")
    cmd_abjad_csp()
    print("\n" + "=" * 70 + "\n")
    cmd_sigla_decode()
    print("\n" + "=" * 70 + "\n")
    cmd_abjad_signal()
    # Track B: Position-Conditioned Encoding
    print("\n" + "=" * 70)
    print("TRACK B: Position-Conditioned Encoding")
    print("=" * 70)
    cmd_slot_vars()
    print("\n" + "=" * 70 + "\n")
    cmd_slot_csp()
    print("\n" + "=" * 70 + "\n")
    cmd_slot_signal()
    # Track F: Vowel Pointer / Matres Lectionis
    print("\n" + "=" * 70)
    print("TRACK F: Vowel Pointer / Matres Lectionis")
    print("=" * 70)
    cmd_vowel_ptr()
    print("\n" + "=" * 70 + "\n")
    cmd_vowel_decode()
    # Track C: Code-Switching Language Model
    print("\n" + "=" * 70)
    print("TRACK C: Code-Switching Language Model")
    print("=" * 70)
    cmd_mixed_lm()
    print("\n" + "=" * 70 + "\n")
    cmd_dialect_decode()
    print("\n" + "=" * 70 + "\n")
    cmd_dialect_signal()
    # Track D: Scripta Continua Re-Segmentation
    print("\n" + "=" * 70)
    print("TRACK D: Scripta Continua Re-Segmentation")
    print("=" * 70)
    cmd_continua()
    print("\n" + "=" * 70 + "\n")
    cmd_reseg_decode()
    print("\n" + "=" * 70 + "\n")
    cmd_reseg_signal()
    # Track E: 2D Spatial Encoding
    print("\n" + "=" * 70)
    print("TRACK E: 2D Spatial Encoding")
    print("=" * 70)
    cmd_gallows_geom()
    print("\n" + "=" * 70 + "\n")
    cmd_spatial_decode()
    # Integration
    print("\n" + "=" * 70)
    print("INTEGRATION")
    print("=" * 70)
    cmd_phase34_integrate()


# -----------------------------------------------------------------------
# Phase 35: Spatial Conditioning + 10K Dictionary
# -----------------------------------------------------------------------

def cmd_spatial_pre():
    """Run Step 35.1: Spatial gallows preprocessing."""
    from voynich.phases.spatial_preprocess import run_spatial_preprocess
    t0 = time.time()
    run_spatial_preprocess()
    print(f"\nStep 35.1 completed in {time.time() - t0:.1f}s")


def cmd_comb_decode():
    """Run Step 35.2: Combined spatial+10K corpus decode."""
    from voynich.phases.combined_decode import run_combined_decode
    t0 = time.time()
    run_combined_decode()
    print(f"\nStep 35.2 completed in {time.time() - t0:.1f}s")


def cmd_comb_signal():
    """Run Step 35.3: Combined signal isolation."""
    from voynich.phases.combined_signal import run_combined_signal
    t0 = time.time()
    run_combined_signal()
    print(f"\nStep 35.3 completed in {time.time() - t0:.1f}s")


def cmd_comb_bigram():
    """Run Step 35.4: Combined bigram plausibility."""
    from voynich.phases.combined_bigrams import run_combined_bigrams
    t0 = time.time()
    run_combined_bigrams()
    print(f"\nStep 35.4 completed in {time.time() - t0:.1f}s")


def cmd_comb_context():
    """Run Step 35.5: Combined context analysis."""
    from voynich.phases.combined_context import run_combined_context
    t0 = time.time()
    run_combined_context()
    print(f"\nStep 35.5 completed in {time.time() - t0:.1f}s")


def cmd_comb_bootstrap():
    """Run Step 35.6: Combined Ventris bootstrap."""
    from voynich.phases.combined_bootstrap import run_combined_bootstrap
    t0 = time.time()
    run_combined_bootstrap()
    print(f"\nStep 35.6 completed in {time.time() - t0:.1f}s")


def cmd_comb_folio():
    """Run Step 35.7: Combined folio transliterations."""
    from voynich.phases.combined_folio import run_combined_folio
    t0 = time.time()
    run_combined_folio()
    print(f"\nStep 35.7 completed in {time.time() - t0:.1f}s")


def cmd_comb_read():
    """Run Step 35.8: Combined readability battery."""
    from voynich.phases.combined_readability import run_combined_readability
    t0 = time.time()
    run_combined_readability()
    print(f"\nStep 35.8 completed in {time.time() - t0:.1f}s")


def cmd_phase35_verdict():
    """Run Step 35.9: Phase 35 verdict."""
    from voynich.phases.phase35_verdict import run_phase35_verdict
    t0 = time.time()
    run_phase35_verdict()
    print(f"\nStep 35.9 completed in {time.time() - t0:.1f}s")


def cmd_phase35():
    """Run full Phase 35 pipeline: Spatial Conditioning + 10K Dictionary."""
    print("=" * 70)
    print("PHASE 35: Spatial Conditioning + 10K Dictionary")
    print("=" * 70)
    cmd_spatial_pre()
    print("\n" + "=" * 70 + "\n")
    cmd_comb_decode()
    print("\n" + "=" * 70 + "\n")
    cmd_comb_signal()
    print("\n" + "=" * 70 + "\n")
    cmd_comb_bigram()
    print("\n" + "=" * 70 + "\n")
    cmd_comb_context()
    print("\n" + "=" * 70 + "\n")
    cmd_comb_bootstrap()
    print("\n" + "=" * 70 + "\n")
    cmd_comb_folio()
    print("\n" + "=" * 70 + "\n")
    cmd_comb_read()
    print("\n" + "=" * 70 + "\n")
    cmd_phase35_verdict()


# ---------------------------------------------------------------------------
# Phase 36: Full Signal Pipeline at 10K Dictionary
# ---------------------------------------------------------------------------

def cmd_decode_10k():
    """Run Step 36.1: Phase 16 decode with 10K matching."""
    from voynich.phases.decode_10k import run_decode_10k
    t0 = time.time()
    run_decode_10k()
    print(f"\nStep 36.1 completed in {time.time() - t0:.1f}s")


def cmd_signal_10k():
    """Run Step 36.2: Signal isolation at 10K."""
    from voynich.phases.signal_10k import run_signal_10k
    t0 = time.time()
    run_signal_10k()
    print(f"\nStep 36.2 completed in {time.time() - t0:.1f}s")


def cmd_bigram_10k():
    """Run Step 36.3: Bigram plausibility at 10K."""
    from voynich.phases.bigrams_10k import run_bigrams_10k
    t0 = time.time()
    run_bigrams_10k()
    print(f"\nStep 36.3 completed in {time.time() - t0:.1f}s")


def cmd_context_10k():
    """Run Step 36.4: Context analysis at 10K."""
    from voynich.phases.context_10k import run_context_10k
    t0 = time.time()
    run_context_10k()
    print(f"\nStep 36.4 completed in {time.time() - t0:.1f}s")


def cmd_boot_10k():
    """Run Step 36.5: Ventris bootstrap at 10K."""
    from voynich.phases.bootstrap_10k import run_bootstrap_10k
    t0 = time.time()
    run_bootstrap_10k()
    print(f"\nStep 36.5 completed in {time.time() - t0:.1f}s")


def cmd_folio_10k():
    """Run Step 36.6: Folio examination at 10K."""
    from voynich.phases.folio_10k import run_folio_10k
    t0 = time.time()
    run_folio_10k()
    print(f"\nStep 36.6 completed in {time.time() - t0:.1f}s")


def cmd_read_10k():
    """Run Step 36.7: Full readability battery at 10K."""
    from voynich.phases.readability_10k import run_readability_10k
    t0 = time.time()
    run_readability_10k()
    print(f"\nStep 36.7 completed in {time.time() - t0:.1f}s")


def cmd_phase36_verdict():
    """Run Step 36.8: Phase 36 verdict."""
    from voynich.phases.phase36_verdict import run_phase36_verdict
    t0 = time.time()
    run_phase36_verdict()
    print(f"\nStep 36.8 completed in {time.time() - t0:.1f}s")


def cmd_phase36():
    """Run full Phase 36 pipeline: Full Signal Pipeline at 10K Dictionary."""
    print("=" * 70)
    print("PHASE 36: Full Signal Pipeline at 10K Dictionary")
    print("=" * 70)
    cmd_decode_10k()
    print("\n" + "=" * 70 + "\n")
    cmd_signal_10k()
    print("\n" + "=" * 70 + "\n")
    cmd_bigram_10k()
    print("\n" + "=" * 70 + "\n")
    cmd_context_10k()
    print("\n" + "=" * 70 + "\n")
    cmd_boot_10k()
    print("\n" + "=" * 70 + "\n")
    cmd_folio_10k()
    print("\n" + "=" * 70 + "\n")
    cmd_read_10k()
    print("\n" + "=" * 70 + "\n")
    cmd_phase36_verdict()


# ── Phase 37: Signal Decomposition, Concatenation, and Content Word Recovery ──

def cmd_cons_group():
    """Run Step 37.1: Consonant Onset Grouping."""
    from voynich.phases.consonant_grouping import run_consonant_grouping
    t0 = time.time()
    run_consonant_grouping()
    print(f"\nStep 37.1 completed in {time.time() - t0:.1f}s")


def cmd_cv_corr():
    """Run Step 37.2: Within-Class Selectivity Correlation."""
    from voynich.phases.cv_correlation import run_cv_correlation
    t0 = time.time()
    run_cv_correlation()
    print(f"\nStep 37.2 completed in {time.time() - t0:.1f}s")


def cmd_vowel_conf():
    """Run Step 37.3: Vowel Confusion Matrix."""
    from voynich.phases.vowel_confusion import run_vowel_confusion
    t0 = time.time()
    run_vowel_confusion()
    print(f"\nStep 37.3 completed in {time.time() - t0:.1f}s")


def cmd_pair_concat():
    """Run Step 37.4: Confirmed Pair Concatenation."""
    from voynich.phases.pair_concat import run_pair_concat
    t0 = time.time()
    run_pair_concat()
    print(f"\nStep 37.4 completed in {time.time() - t0:.1f}s")


def cmd_concat_signal():
    """Run Step 37.5: Concatenated Signal Isolation."""
    from voynich.phases.concat_signal import run_concat_signal
    t0 = time.time()
    run_concat_signal()
    print(f"\nStep 37.5 completed in {time.time() - t0:.1f}s")


def cmd_concat_bigram():
    """Run Step 37.6: Concatenated Bigram Test."""
    from voynich.phases.concat_bigrams import run_concat_bigrams
    t0 = time.time()
    run_concat_bigrams()
    print(f"\nStep 37.6 completed in {time.time() - t0:.1f}s")


def cmd_joint_target():
    """Run Step 37.7: Joint Swap Targeting."""
    from voynich.phases.joint_target import run_joint_target
    t0 = time.time()
    run_joint_target()
    print(f"\nStep 37.7 completed in {time.time() - t0:.1f}s")


def cmd_joint_swap():
    """Run Step 37.8: Exhaustive 2-Triple and Targeted 3-Triple Search."""
    from voynich.phases.joint_swap import run_joint_swap
    t0 = time.time()
    run_joint_swap()
    print(f"\nStep 37.8 completed in {time.time() - t0:.1f}s")


def cmd_joint_val():
    """Run Step 37.9: Joint Swap Validation."""
    from voynich.phases.joint_validate import run_joint_validate
    t0 = time.time()
    run_joint_validate()
    print(f"\nStep 37.9 completed in {time.time() - t0:.1f}s")


def cmd_f57v_eva():
    """Run Step 37.10: f57v EVA Token Diversity."""
    from voynich.phases.f57v_eva import run_f57v_eva
    t0 = time.time()
    run_f57v_eva()
    print(f"\nStep 37.10 completed in {time.time() - t0:.1f}s")


def cmd_f57v_struct():
    """Run Step 37.11: f57v Structural Analysis."""
    from voynich.phases.f57v_structure import run_f57v_structure
    t0 = time.time()
    run_f57v_structure()
    print(f"\nStep 37.11 completed in {time.time() - t0:.1f}s")


def cmd_ital_corpus():
    """Run Step 37.12: Italian Reference Corpus."""
    from voynich.phases.italian_corpus import run_italian_corpus
    t0 = time.time()
    run_italian_corpus()
    print(f"\nStep 37.12 completed in {time.time() - t0:.1f}s")


def cmd_ital_10k():
    """Run Step 37.13: Italian 10K Dictionary."""
    from voynich.phases.italian_10k import run_italian_10k
    t0 = time.time()
    run_italian_10k()
    print(f"\nStep 37.13 completed in {time.time() - t0:.1f}s")


def cmd_ital_signal():
    """Run Step 37.14: Italian Signal Pipeline."""
    from voynich.phases.italian_signal import run_italian_signal
    t0 = time.time()
    run_italian_signal()
    print(f"\nStep 37.14 completed in {time.time() - t0:.1f}s")


def cmd_phase37_integrate():
    """Run Step 37.15: Phase 37 Integration."""
    from voynich.phases.phase37_integrate import run_phase37_integrate
    t0 = time.time()
    run_phase37_integrate()
    print(f"\nStep 37.15 completed in {time.time() - t0:.1f}s")


def cmd_phase37():
    """Run full Phase 37 pipeline: Signal Decomposition, Concatenation, and Content Word Recovery."""
    print("=" * 70)
    print("PHASE 37: Signal Decomposition, Concatenation, and Content Word Recovery")
    print("=" * 70)
    # Investigation 1: Consonant-Vowel Decomposition
    cmd_cons_group()
    print("\n" + "=" * 70 + "\n")
    cmd_cv_corr()
    print("\n" + "=" * 70 + "\n")
    cmd_vowel_conf()
    print("\n" + "=" * 70 + "\n")
    # Investigation 2: Signal Pair Concatenation
    cmd_pair_concat()
    print("\n" + "=" * 70 + "\n")
    cmd_concat_signal()
    print("\n" + "=" * 70 + "\n")
    cmd_concat_bigram()
    print("\n" + "=" * 70 + "\n")
    # Investigation 3: Multi-Triple Joint Swap
    cmd_joint_target()
    print("\n" + "=" * 70 + "\n")
    cmd_joint_swap()
    print("\n" + "=" * 70 + "\n")
    cmd_joint_val()
    print("\n" + "=" * 70 + "\n")
    # Investigation 4: f57v Deep Examination
    cmd_f57v_eva()
    print("\n" + "=" * 70 + "\n")
    cmd_f57v_struct()
    print("\n" + "=" * 70 + "\n")
    # Investigation 5: Northern Italian 10K Dictionary
    cmd_ital_corpus()
    print("\n" + "=" * 70 + "\n")
    cmd_ital_10k()
    print("\n" + "=" * 70 + "\n")
    cmd_ital_signal()
    print("\n" + "=" * 70 + "\n")
    # Integration
    cmd_phase37_integrate()


# Phase 38: Macaronic Signal Pipeline
def cmd_merged_dict():
    """Run Step 38.1: Merged Dictionary Construction."""
    from voynich.phases.merged_dict import run_merged_dict
    t0 = time.time()
    run_merged_dict()
    print(f"\nStep 38.1 completed in {time.time() - t0:.1f}s")


def cmd_merged_decode():
    """Run Step 38.2: Merged Dictionary Decode Matching."""
    from voynich.phases.merged_decode import run_merged_decode
    t0 = time.time()
    run_merged_decode()
    print(f"\nStep 38.2 completed in {time.time() - t0:.1f}s")


def cmd_merged_signal():
    """Run Step 38.3: Merged Signal Isolation."""
    from voynich.phases.merged_signal import run_merged_signal
    t0 = time.time()
    run_merged_signal()
    print(f"\nStep 38.3 completed in {time.time() - t0:.1f}s")


def cmd_merged_bigram():
    """Run Step 38.4: Merged Bigram Plausibility."""
    from voynich.phases.merged_bigrams import run_merged_bigrams
    t0 = time.time()
    run_merged_bigrams()
    print(f"\nStep 38.4 completed in {time.time() - t0:.1f}s")


def cmd_merged_context():
    """Run Step 38.5: Macaronic Context Analysis."""
    from voynich.phases.merged_context import run_merged_context
    t0 = time.time()
    run_merged_context()
    print(f"\nStep 38.5 completed in {time.time() - t0:.1f}s")


def cmd_merged_boot():
    """Run Step 38.6: Macaronic Ventris Bootstrap."""
    from voynich.phases.merged_bootstrap import run_merged_bootstrap
    t0 = time.time()
    run_merged_bootstrap()
    print(f"\nStep 38.6 completed in {time.time() - t0:.1f}s")


def cmd_merged_concat():
    """Run Step 38.7: Macaronic Concatenation Test."""
    from voynich.phases.merged_concat import run_merged_concat
    t0 = time.time()
    run_merged_concat()
    print(f"\nStep 38.7 completed in {time.time() - t0:.1f}s")


def cmd_merged_folio():
    """Run Step 38.8: Macaronic Folio Examination."""
    from voynich.phases.merged_folio import run_merged_folio
    t0 = time.time()
    run_merged_folio()
    print(f"\nStep 38.8 completed in {time.time() - t0:.1f}s")


def cmd_merged_read():
    """Run Step 38.9: Full Readability Battery."""
    from voynich.phases.merged_readability import run_merged_readability
    t0 = time.time()
    run_merged_readability()
    print(f"\nStep 38.9 completed in {time.time() - t0:.1f}s")


def cmd_phase38_verdict():
    """Run Step 38.10: Phase 38 Verdict."""
    from voynich.phases.phase38_verdict import run_phase38_verdict
    t0 = time.time()
    run_phase38_verdict()
    print(f"\nStep 38.10 completed in {time.time() - t0:.1f}s")


def cmd_phase38():
    """Run full Phase 38 pipeline: Macaronic Signal Pipeline."""
    print("=" * 70)
    print("PHASE 38: Macaronic Signal Pipeline")
    print("=" * 70)
    cmd_merged_dict()
    print("\n" + "=" * 70 + "\n")
    cmd_merged_decode()
    print("\n" + "=" * 70 + "\n")
    cmd_merged_signal()
    print("\n" + "=" * 70 + "\n")
    cmd_merged_bigram()
    print("\n" + "=" * 70 + "\n")
    cmd_merged_context()
    print("\n" + "=" * 70 + "\n")
    cmd_merged_boot()
    print("\n" + "=" * 70 + "\n")
    cmd_merged_concat()
    print("\n" + "=" * 70 + "\n")
    cmd_merged_folio()
    print("\n" + "=" * 70 + "\n")
    cmd_merged_read()
    print("\n" + "=" * 70 + "\n")
    cmd_phase38_verdict()


# Phase 39: Edit-Distance Bridge, Vowel Recovery, and Macaronic Crib Exploitation
def cmd_ed1_decomp():
    """Run Step 39.1: ED1 Decomposition of CC Bigrams."""
    from voynich.phases.ed1_decomposition import run_ed1_decomposition
    t0 = time.time()
    run_ed1_decomposition()
    print(f"\nStep 39.1 completed in {time.time() - t0:.1f}s")


def cmd_vowel_map():
    """Run Step 39.2: Vowel Error Map."""
    from voynich.phases.vowel_error_map import run_vowel_error_map
    t0 = time.time()
    run_vowel_error_map()
    print(f"\nStep 39.2 completed in {time.time() - t0:.1f}s")


def cmd_vowel_fix():
    """Run Step 39.3: Targeted Vowel Fix."""
    from voynich.phases.targeted_vowel_fix import run_targeted_vowel_fix
    t0 = time.time()
    run_targeted_vowel_fix()
    print(f"\nStep 39.3 completed in {time.time() - t0:.1f}s")


def cmd_corrected_sig():
    """Run Step 39.4: Corrected Table Signal Pipeline."""
    from voynich.phases.corrected_signal import run_corrected_signal
    t0 = time.time()
    run_corrected_signal()
    print(f"\nStep 39.4 completed in {time.time() - t0:.1f}s")


def cmd_phrase_crib():
    """Run Step 39.5: Phrase-Level Crib Extraction."""
    from voynich.phases.phrase_cribs import run_phrase_cribs
    t0 = time.time()
    run_phrase_cribs()
    print(f"\nStep 39.5 completed in {time.time() - t0:.1f}s")


def cmd_phrase_align():
    """Run Step 39.6: Phrase Template Alignment."""
    from voynich.phases.phrase_alignment import run_phrase_alignment
    t0 = time.time()
    run_phrase_alignment()
    print(f"\nStep 39.6 completed in {time.time() - t0:.1f}s")


def cmd_phrase_corr():
    """Run Step 39.7: Phrase-Derived Corrections."""
    from voynich.phases.phrase_corrections import run_phrase_corrections
    t0 = time.time()
    run_phrase_corrections()
    print(f"\nStep 39.7 completed in {time.time() - t0:.1f}s")


def cmd_ital_plant():
    """Run Step 39.8: Italian Plant Name Dictionary."""
    from voynich.phases.italian_plant_names import run_italian_plant_names
    t0 = time.time()
    run_italian_plant_names()
    print(f"\nStep 39.8 completed in {time.time() - t0:.1f}s")


def cmd_ital_bot_csp():
    """Run Step 39.9: Italian Botanical CSP."""
    from voynich.phases.italian_botanical_csp import run_italian_botanical_csp
    t0 = time.time()
    run_italian_botanical_csp()
    print(f"\nStep 39.9 completed in {time.time() - t0:.1f}s")


def cmd_bot_prop():
    """Run Step 39.10: Botanical Propagation."""
    from voynich.phases.botanical_propagate import run_botanical_propagate
    t0 = time.time()
    run_botanical_propagate()
    print(f"\nStep 39.10 completed in {time.time() - t0:.1f}s")


def cmd_venetian_lex():
    """Run Step 39.11: Venetian Lexicon."""
    from voynich.phases.venetian_lexicon import run_venetian_lexicon
    t0 = time.time()
    run_venetian_lexicon()
    print(f"\nStep 39.11 completed in {time.time() - t0:.1f}s")


def cmd_venetian_dec():
    """Run Step 39.12: Venetian Decode Test."""
    from voynich.phases.venetian_decode import run_venetian_decode
    t0 = time.time()
    run_venetian_decode()
    print(f"\nStep 39.12 completed in {time.time() - t0:.1f}s")


def cmd_venetian_phr():
    """Run Step 39.13: Venetian Pharmaceutical Phrases."""
    from voynich.phases.venetian_phrases import run_venetian_phrases
    t0 = time.time()
    run_venetian_phrases()
    print(f"\nStep 39.13 completed in {time.time() - t0:.1f}s")


def cmd_amp_dict():
    """Run Step 39.14: Signal-Calibrated Dictionary."""
    from voynich.phases.amplified_dict import run_amplified_dict
    t0 = time.time()
    run_amplified_dict()
    print(f"\nStep 39.14 completed in {time.time() - t0:.1f}s")


def cmd_amp_signal():
    """Run Step 39.15: Signal Isolation at Calibrated Dictionary."""
    from voynich.phases.amplified_signal import run_amplified_signal
    t0 = time.time()
    run_amplified_signal()
    print(f"\nStep 39.15 completed in {time.time() - t0:.1f}s")


def cmd_amp_bigram():
    """Run Step 39.16: Amplified Bigram Test."""
    from voynich.phases.amplified_bigrams import run_amplified_bigrams
    t0 = time.time()
    run_amplified_bigrams()
    print(f"\nStep 39.16 completed in {time.time() - t0:.1f}s")


def cmd_phase39_integrate():
    """Run Step 39.17: Phase 39 Integration."""
    from voynich.phases.phase39_integrate import run_phase39_integrate
    t0 = time.time()
    run_phase39_integrate()
    print(f"\nStep 39.17 completed in {time.time() - t0:.1f}s")


def cmd_phase39():
    """Run full Phase 39 pipeline: Edit-Distance Bridge and Macaronic Crib Exploitation."""
    print("=" * 70)
    print("PHASE 39: Edit-Distance Bridge, Vowel Recovery, and Macaronic Crib Exploitation")
    print("=" * 70)
    # Track A: Edit-Distance Bridge
    cmd_ed1_decomp()
    print("\n" + "=" * 70 + "\n")
    cmd_vowel_map()
    print("\n" + "=" * 70 + "\n")
    cmd_vowel_fix()
    print("\n" + "=" * 70 + "\n")
    cmd_corrected_sig()
    print("\n" + "=" * 70 + "\n")
    # Track B: Medical Phrase Cribs
    cmd_phrase_crib()
    print("\n" + "=" * 70 + "\n")
    cmd_phrase_align()
    print("\n" + "=" * 70 + "\n")
    cmd_phrase_corr()
    print("\n" + "=" * 70 + "\n")
    # Track C: Italian Botanical
    cmd_ital_plant()
    print("\n" + "=" * 70 + "\n")
    cmd_ital_bot_csp()
    print("\n" + "=" * 70 + "\n")
    cmd_bot_prop()
    print("\n" + "=" * 70 + "\n")
    # Track D: Venetian
    cmd_venetian_lex()
    print("\n" + "=" * 70 + "\n")
    cmd_venetian_dec()
    print("\n" + "=" * 70 + "\n")
    cmd_venetian_phr()
    print("\n" + "=" * 70 + "\n")
    # Track E: Amplified
    cmd_amp_dict()
    print("\n" + "=" * 70 + "\n")
    cmd_amp_signal()
    print("\n" + "=" * 70 + "\n")
    cmd_amp_bigram()
    print("\n" + "=" * 70 + "\n")
    # Integration
    cmd_phase39_integrate()


# ---------------------------------------------------------------------------
# Phase 40: Venetian Reading, CVC Expansion, Folio Decipherment
# ---------------------------------------------------------------------------

# Track A
def cmd_ven_forms():
    """Run Step 40.1: Venetian Phonological Form Inventory."""
    from voynich.phases.venetian_forms import run_venetian_forms
    t0 = time.time()
    run_venetian_forms()
    print(f"\nStep 40.1 completed in {time.time() - t0:.1f}s")


def cmd_ven_match():
    """Run Step 40.2: Venetian Form Matching."""
    from voynich.phases.venetian_match import run_venetian_match
    t0 = time.time()
    run_venetian_match()
    print(f"\nStep 40.2 completed in {time.time() - t0:.1f}s")


def cmd_ven_bigram():
    """Run Step 40.3: Venetian Reference Bigrams."""
    from voynich.phases.venetian_bigrams import run_venetian_bigrams
    t0 = time.time()
    run_venetian_bigrams()
    print(f"\nStep 40.3 completed in {time.time() - t0:.1f}s")


def cmd_cc_reclass():
    """Run Step 40.4: CC Bigram Reclassification."""
    from voynich.phases.cc_reclassify import run_cc_reclassify
    t0 = time.time()
    run_cc_reclassify()
    print(f"\nStep 40.4 completed in {time.time() - t0:.1f}s")


# Track B
def cmd_cvc_inv():
    """Run Step 40.5: CVC/CCV Syllable Inventory."""
    from voynich.phases.cvc_inventory import run_cvc_inventory
    t0 = time.time()
    run_cvc_inventory()
    print(f"\nStep 40.5 completed in {time.time() - t0:.1f}s")


def cmd_cvc_csp():
    """Run Step 40.6: CVC-Expanded CSP."""
    from voynich.phases.cvc_csp import run_cvc_csp
    t0 = time.time()
    run_cvc_csp()
    print(f"\nStep 40.6 completed in {time.time() - t0:.1f}s")


def cmd_cvc_signal():
    """Run Step 40.7: CVC Signal Isolation."""
    from voynich.phases.cvc_signal import run_cvc_signal
    t0 = time.time()
    run_cvc_signal()
    print(f"\nStep 40.7 completed in {time.time() - t0:.1f}s")


def cmd_cvc_bigram():
    """Run Step 40.8: CVC Bigram Test."""
    from voynich.phases.cvc_bigrams import run_cvc_bigrams
    t0 = time.time()
    run_cvc_bigrams()
    print(f"\nStep 40.8 completed in {time.time() - t0:.1f}s")


# Track C
def cmd_syl_lex():
    """Run Step 40.9: Signal Word Syllable Lexicon."""
    from voynich.phases.syllable_lexicon import run_syllable_lexicon
    t0 = time.time()
    run_syllable_lexicon()
    print(f"\nStep 40.9 completed in {time.time() - t0:.1f}s")


def cmd_folio_recon():
    """Run Step 40.10: Folio Text Reconstruction."""
    from voynich.phases.folio_reconstruction import run_folio_reconstruction
    t0 = time.time()
    run_folio_reconstruction()
    print(f"\nStep 40.10 completed in {time.time() - t0:.1f}s")


def cmd_f57v_read():
    """Run Step 40.11: f57v Dedicated Venetian Reading."""
    from voynich.phases.f57v_reading import run_f57v_reading
    t0 = time.time()
    run_f57v_reading()
    print(f"\nStep 40.11 completed in {time.time() - t0:.1f}s")


def cmd_best_read():
    """Run Step 40.12: Best Non-f57v Folio Reading."""
    from voynich.phases.best_folio_reading import run_best_folio_reading
    t0 = time.time()
    run_best_folio_reading()
    print(f"\nStep 40.12 completed in {time.time() - t0:.1f}s")


# Track D
def cmd_drosera_con():
    """Run Step 40.13: Drosera Constraint Extraction."""
    from voynich.phases.drosera_constraints import run_drosera_constraints
    t0 = time.time()
    run_drosera_constraints()
    print(f"\nStep 40.13 completed in {time.time() - t0:.1f}s")


def cmd_bot_pred():
    """Run Step 40.14: Predicted Form Generation."""
    from voynich.phases.botanical_predictions import run_botanical_predictions
    t0 = time.time()
    run_botanical_predictions()
    print(f"\nStep 40.14 completed in {time.time() - t0:.1f}s")


def cmd_bot_search():
    """Run Step 40.15: Predicted Form Search."""
    from voynich.phases.botanical_search import run_botanical_search
    t0 = time.time()
    run_botanical_search()
    print(f"\nStep 40.15 completed in {time.time() - t0:.1f}s")


# Integration
def cmd_phase40_integrate():
    """Run Step 40.16: Phase 40 Integration."""
    from voynich.phases.phase40_integrate import run_phase40_integrate
    t0 = time.time()
    run_phase40_integrate()
    print(f"\nStep 40.16 completed in {time.time() - t0:.1f}s")


def cmd_phase40():
    """Run full Phase 40 pipeline: Venetian Reading, CVC Expansion, Folio Decipherment."""
    print("=" * 70)
    print("PHASE 40: Venetian Reading, CVC Expansion, and Folio-Level Decipherment")
    print("=" * 70)
    # Track A: Venetian Correctness
    print("\n" + "=" * 70)
    print("TRACK A: Venetian Correctness Hypothesis")
    print("=" * 70)
    cmd_ven_forms()
    print("\n" + "=" * 70 + "\n")
    cmd_ven_match()
    print("\n" + "=" * 70 + "\n")
    cmd_ven_bigram()
    print("\n" + "=" * 70 + "\n")
    cmd_cc_reclass()
    # Track B: CVC Expansion
    print("\n" + "=" * 70)
    print("TRACK B: CVC/CCV Syllable Expansion")
    print("=" * 70)
    cmd_cvc_inv()
    print("\n" + "=" * 70 + "\n")
    cmd_cvc_csp()
    print("\n" + "=" * 70 + "\n")
    cmd_cvc_signal()
    print("\n" + "=" * 70 + "\n")
    cmd_cvc_bigram()
    # Track C: Folio Reading
    print("\n" + "=" * 70)
    print("TRACK C: Folio-Level Venetian Reading")
    print("=" * 70)
    cmd_syl_lex()
    print("\n" + "=" * 70 + "\n")
    cmd_folio_recon()
    print("\n" + "=" * 70 + "\n")
    cmd_f57v_read()
    print("\n" + "=" * 70 + "\n")
    cmd_best_read()
    # Track D: Botanical Prediction
    print("\n" + "=" * 70)
    print("TRACK D: Botanical Prediction from Drosera")
    print("=" * 70)
    cmd_drosera_con()
    print("\n" + "=" * 70 + "\n")
    cmd_bot_pred()
    print("\n" + "=" * 70 + "\n")
    cmd_bot_search()
    # Integration
    print("\n" + "=" * 70)
    print("INTEGRATION")
    print("=" * 70)
    cmd_phase40_integrate()


# ---------------------------------------------------------------------------
# Phase 41: Venetian Validation, Lexicon Completion, f57v Reading, Botanical Fix
# ---------------------------------------------------------------------------

# Track A
def cmd_null_ven():
    from voynich.phases.null_venetian_decode import run_null_venetian_decode
    run_null_venetian_decode()

def cmd_ven_valid():
    from voynich.phases.venetian_validated import run_venetian_validated
    run_venetian_validated()

def cmd_ven_sig():
    from voynich.phases.venetian_signal_proper import run_venetian_signal_proper
    run_venetian_signal_proper()

def cmd_ven_confirm():
    from voynich.phases.venetian_confirmed import run_venetian_confirmed
    run_venetian_confirmed()

# Track B
def cmd_ungloss_analyze():
    from voynich.phases.unglossed_analysis import run_unglossed_analysis
    run_unglossed_analysis()

def cmd_ven_dict():
    from voynich.phases.venetian_dict_search import run_venetian_dict_search
    run_venetian_dict_search()

def cmd_context_disamb():
    from voynich.phases.context_disambiguation import run_context_disambiguation
    run_context_disambiguation()

def cmd_full_lex():
    from voynich.phases.complete_lexicon import run_complete_lexicon
    run_complete_lexicon()

# Track C
def cmd_formula_seg():
    from voynich.phases.formula_segmentation import run_formula_segmentation
    run_formula_segmentation()

def cmd_inter_formula():
    from voynich.phases.inter_formula_tokens import run_inter_formula_tokens
    run_inter_formula_tokens()

def cmd_ingred_search():
    from voynich.phases.ingredient_search import run_ingredient_search
    run_ingredient_search()

def cmd_f57v_complete():
    from voynich.phases.f57v_complete_reading import run_f57v_complete_reading
    run_f57v_complete_reading()

# Track D
def cmd_bot_fix():
    from voynich.phases.botanical_data_fix import run_botanical_data_fix
    run_botanical_data_fix()

def cmd_drosera_prop():
    from voynich.phases.drosera_propagation import run_drosera_propagation
    run_drosera_propagation()

def cmd_bot_pred_v2():
    from voynich.phases.botanical_predictions_v2 import run_botanical_predictions_v2
    run_botanical_predictions_v2()

# Integration
def cmd_phase41_integrate():
    from voynich.phases.phase41_integrate import run_phase41_integrate
    run_phase41_integrate()

def cmd_phase41():
    """Run full Phase 41 pipeline: Venetian Validation, Lexicon, f57v, Botanical."""
    print("=" * 70)
    print("PHASE 41: Venetian Validation, Lexicon Completion, and Inter-Formula Content Recovery")
    print("=" * 70)
    # Track A: Venetian Null Validation
    print("\n" + "=" * 70)
    print("TRACK A: Venetian Null Validation")
    print("=" * 70)
    cmd_null_ven()
    print("\n" + "=" * 70 + "\n")
    cmd_ven_valid()
    print("\n" + "=" * 70 + "\n")
    cmd_ven_sig()
    print("\n" + "=" * 70 + "\n")
    cmd_ven_confirm()
    # Track B: Lexicon Completion
    print("\n" + "=" * 70)
    print("TRACK B: Lexicon Completion")
    print("=" * 70)
    cmd_ungloss_analyze()
    print("\n" + "=" * 70 + "\n")
    cmd_ven_dict()
    print("\n" + "=" * 70 + "\n")
    cmd_context_disamb()
    print("\n" + "=" * 70 + "\n")
    cmd_full_lex()
    # Track C: f57v Inter-Formula Content
    print("\n" + "=" * 70)
    print("TRACK C: f57v Inter-Formula Content Recovery")
    print("=" * 70)
    cmd_formula_seg()
    print("\n" + "=" * 70 + "\n")
    cmd_inter_formula()
    print("\n" + "=" * 70 + "\n")
    cmd_ingred_search()
    print("\n" + "=" * 70 + "\n")
    cmd_f57v_complete()
    # Track D: Botanical Pipeline Fix
    print("\n" + "=" * 70)
    print("TRACK D: Botanical Pipeline Fix")
    print("=" * 70)
    cmd_bot_fix()
    print("\n" + "=" * 70 + "\n")
    cmd_drosera_prop()
    print("\n" + "=" * 70 + "\n")
    cmd_bot_pred_v2()
    # Integration
    print("\n" + "=" * 70)
    print("INTEGRATION")
    print("=" * 70)
    cmd_phase41_integrate()


# ── Phase 42: Bigram Audit, Symmetric Revalidation, Ground-Truth ──

def cmd_bigram_audit():
    from voynich.phases.bigram_code_audit import run_bigram_code_audit
    run_bigram_code_audit()

def cmd_sym_recompute():
    from voynich.phases.symmetric_recompute import run_symmetric_recompute
    run_symmetric_recompute()

def cmd_sig_reval():
    from voynich.phases.signal_word_revalidate import run_signal_word_revalidate
    run_signal_word_revalidate()

def cmd_sel_audit():
    from voynich.phases.selectivity_audit import run_selectivity_audit
    run_selectivity_audit()

def cmd_ground_truth():
    from voynich.phases.ground_truth import run_ground_truth
    run_ground_truth()

def cmd_phase42():
    """Run full Phase 42 pipeline: Bigram Audit and Ground Truth."""
    print("=" * 70)
    print("PHASE 42: Bigram Audit, Symmetric Revalidation, Ground-Truth")
    print("=" * 70)
    cmd_bigram_audit()
    print("\n" + "=" * 70 + "\n")
    cmd_sym_recompute()
    print("\n" + "=" * 70 + "\n")
    cmd_sig_reval()
    print("\n" + "=" * 70 + "\n")
    cmd_sel_audit()
    print("\n" + "=" * 70 + "\n")
    cmd_ground_truth()


# Phase 43: Re-Encoding Inversion, Structural Probing, Conditional Decoding

def cmd_vm_finger():
    from voynich.phases.voynich_fingerprint import run_voynich_fingerprint
    run_voynich_fingerprint()

def cmd_tachy_enc():
    from voynich.phases.tachygraphic_encoder import run_tachygraphic_encoder
    run_tachygraphic_encoder()

def cmd_enc_search():
    from voynich.phases.encoding_search import run_encoding_search
    run_encoding_search()

def cmd_inv_decode():
    from voynich.phases.inversion_decode import run_inversion_decode
    run_inversion_decode()

def cmd_inv_validate():
    from voynich.phases.inversion_validate import run_inversion_validate
    run_inversion_validate()

def cmd_sig_pos():
    from voynich.phases.signal_positions import run_signal_positions
    run_signal_positions()

def cmd_pos_profiles():
    from voynich.phases.positional_profiles import run_positional_profiles
    run_positional_profiles()

def cmd_cooccur():
    from voynich.phases.cooccurrence_structure import run_cooccurrence_structure
    run_cooccurrence_structure()

def cmd_struct_read():
    from voynich.phases.structural_reading import run_structural_reading
    run_structural_reading()

def cmd_hmm_arch():
    from voynich.phases.hmm_architecture import run_hmm_architecture
    run_hmm_architecture()

def cmd_anchor_init():
    from voynich.phases.anchor_initialization import run_anchor_initialization
    run_anchor_initialization()

def cmd_bw_train():
    from voynich.phases.baum_welch_training import run_baum_welch_training
    run_baum_welch_training()

def cmd_hmm_decode():
    from voynich.phases.viterbi_decode import run_viterbi_decode
    run_viterbi_decode()

def cmd_hmm_signal():
    from voynich.phases.hmm_signal import run_hmm_signal
    run_hmm_signal()

def cmd_phase43_integrate():
    from voynich.phases.phase43_integrate import run_phase43_integrate
    run_phase43_integrate()

def cmd_phase43():

    """Run full Phase 43 pipeline."""
    print("=" * 70)
    print("PHASE 43: Re-Encoding Inversion, Structural Probing, Conditional Decoding")
    print("=" * 70)
    # Approach 1: Re-Encoding Inversion
    print("\n" + "=" * 70)
    print("APPROACH 1: Re-Encoding Inversion")
    print("=" * 70)
    cmd_vm_finger()
    print("\n" + "=" * 70 + "\n")
    cmd_tachy_enc()
    print("\n" + "=" * 70 + "\n")
    cmd_enc_search()
    print("\n" + "=" * 70 + "\n")
    cmd_inv_decode()
    print("\n" + "=" * 70 + "\n")
    cmd_inv_validate()
    # Approach 4: Signal Word Structural Probing
    print("\n" + "=" * 70)
    print("APPROACH 4: Signal Word Structural Probing")
    print("=" * 70)
    cmd_sig_pos()
    print("\n" + "=" * 70 + "\n")
    cmd_pos_profiles()
    print("\n" + "=" * 70 + "\n")
    cmd_cooccur()
    print("\n" + "=" * 70 + "\n")
    cmd_struct_read()
    # Approach 5: Context-Dependent HMM Decoding
    print("\n" + "=" * 70)
    print("APPROACH 5: Context-Dependent HMM Decoding")
    print("=" * 70)
    cmd_hmm_arch()
    print("\n" + "=" * 70 + "\n")
    cmd_anchor_init()
    print("\n" + "=" * 70 + "\n")
    cmd_bw_train()
    print("\n" + "=" * 70 + "\n")
    cmd_hmm_decode()
    print("\n" + "=" * 70 + "\n")
    cmd_hmm_signal()
    # Integration
    print("\n" + "=" * 70)
    print("INTEGRATION")
    print("=" * 70)
    cmd_phase43_integrate()


# Phase 44: Solution Landscape Enumeration (MaxSAT, SBM, CSA)

# Track A: MaxSAT
def cmd_maxsat_encode():
    from voynich.phases.maxsat_landscape import run_maxsat_encode
    t0 = time.time()
    run_maxsat_encode()
    print(f"\nStep 44A.1 completed in {time.time() - t0:.1f}s")

def cmd_maxsat_solve():
    from voynich.phases.maxsat_landscape import run_maxsat_solve
    t0 = time.time()
    run_maxsat_solve()
    print(f"\nStep 44A.2 completed in {time.time() - t0:.1f}s")

def cmd_maxsat_landscape():
    from voynich.phases.maxsat_landscape import run_maxsat_landscape
    t0 = time.time()
    run_maxsat_landscape()
    print(f"\nStep 44A.3 completed in {time.time() - t0:.1f}s")

def cmd_maxsat_validate():
    from voynich.phases.maxsat_landscape import run_maxsat_validate
    t0 = time.time()
    run_maxsat_validate()
    print(f"\nStep 44A.4 completed in {time.time() - t0:.1f}s")

def cmd_track_a():
    from voynich.phases.maxsat_landscape import run_track_a
    t0 = time.time()
    run_track_a()
    print(f"\nTrack A completed in {time.time() - t0:.1f}s")

# Track B: SBM
def cmd_sbm_graph():
    from voynich.phases.sbm_cooccurrence import run_sbm_graph
    t0 = time.time()
    run_sbm_graph()
    print(f"\nStep 44B.1 completed in {time.time() - t0:.1f}s")

def cmd_sbm_fit():
    from voynich.phases.sbm_cooccurrence import run_sbm_fit
    t0 = time.time()
    run_sbm_fit()
    print(f"\nStep 44B.2 completed in {time.time() - t0:.1f}s")

def cmd_sbm_compare():
    from voynich.phases.sbm_cooccurrence import run_sbm_compare
    t0 = time.time()
    run_sbm_compare()
    print(f"\nStep 44B.3 completed in {time.time() - t0:.1f}s")

def cmd_sbm_predict():
    from voynich.phases.sbm_cooccurrence import run_sbm_predict
    t0 = time.time()
    run_sbm_predict()
    print(f"\nStep 44B.4 completed in {time.time() - t0:.1f}s")

def cmd_sbm_validate():
    from voynich.phases.sbm_cooccurrence import run_sbm_validate
    t0 = time.time()
    run_sbm_validate()
    print(f"\nStep 44B.5 completed in {time.time() - t0:.1f}s")

def cmd_track_b():
    from voynich.phases.sbm_cooccurrence import run_track_b
    t0 = time.time()
    run_track_b()
    print(f"\nTrack B completed in {time.time() - t0:.1f}s")

# Track C: k-Permutation CSA
def cmd_kperm_energy():
    from voynich.phases.kperm_csa import run_kperm_energy
    t0 = time.time()
    run_kperm_energy()
    print(f"\nStep 44C.1 completed in {time.time() - t0:.1f}s")

def cmd_kperm_search():
    from voynich.phases.kperm_csa import run_kperm_search
    t0 = time.time()
    run_kperm_search()
    print(f"\nStep 44C.2 completed in {time.time() - t0:.1f}s")

def cmd_kperm_analyze():
    from voynich.phases.kperm_csa import run_kperm_analyze
    t0 = time.time()
    run_kperm_analyze()
    print(f"\nStep 44C.3 completed in {time.time() - t0:.1f}s")

def cmd_kperm_validate():
    from voynich.phases.kperm_csa import run_kperm_validate
    t0 = time.time()
    run_kperm_validate()
    print(f"\nStep 44C.4 completed in {time.time() - t0:.1f}s")

def cmd_track_c():
    from voynich.phases.kperm_csa import run_track_c
    t0 = time.time()
    run_track_c()
    print(f"\nTrack C completed in {time.time() - t0:.1f}s")

# Phase 44 Integration
def cmd_phase44_integrate():
    from voynich.phases.phase44_integrate import run_phase44_integrate
    t0 = time.time()
    run_phase44_integrate()
    print(f"\nPhase 44 integration completed in {time.time() - t0:.1f}s")

def cmd_phase44():
    """Run full Phase 44 pipeline: MaxSAT, SBM, CSA + integration."""
    print("=" * 70)
    print("PHASE 44: Solution Landscape Enumeration")
    print("=" * 70)
    # Track A: MaxSAT
    print("\n" + "=" * 70)
    print("TRACK A: MaxSAT Landscape Enumeration")
    print("=" * 70)
    cmd_maxsat_encode()
    print("\n" + "=" * 70 + "\n")
    cmd_maxsat_solve()
    print("\n" + "=" * 70 + "\n")
    cmd_maxsat_landscape()
    print("\n" + "=" * 70 + "\n")
    cmd_maxsat_validate()
    # Track B: SBM
    print("\n" + "=" * 70)
    print("TRACK B: Stochastic Block Model")
    print("=" * 70)
    cmd_sbm_graph()
    print("\n" + "=" * 70 + "\n")
    cmd_sbm_fit()
    print("\n" + "=" * 70 + "\n")
    cmd_sbm_compare()
    print("\n" + "=" * 70 + "\n")
    cmd_sbm_predict()
    print("\n" + "=" * 70 + "\n")
    cmd_sbm_validate()
    # Track C: CSA
    print("\n" + "=" * 70)
    print("TRACK C: Coupled Simulated Annealing")
    print("=" * 70)
    cmd_kperm_energy()
    print("\n" + "=" * 70 + "\n")
    cmd_kperm_search()
    print("\n" + "=" * 70 + "\n")
    cmd_kperm_analyze()
    print("\n" + "=" * 70 + "\n")
    cmd_kperm_validate()
    # Integration
    print("\n" + "=" * 70)
    print("INTEGRATION")
    print("=" * 70)
    cmd_phase44_integrate()


# Phase 45: SBM Community Forensics + Distributional Re-encoding

# Track A: SBM Forensics
def cmd_sbm_profile():
    from voynich.phases.sbm_forensics import run_sbm_profile
    t0 = time.time()
    run_sbm_profile()
    print(f"\nStep 45A.1 completed in {time.time() - t0:.1f}s")

def cmd_sbm_position():
    from voynich.phases.sbm_forensics import run_sbm_position
    t0 = time.time()
    run_sbm_position()
    print(f"\nStep 45A.2 completed in {time.time() - t0:.1f}s")

def cmd_sbm_morpheme():
    from voynich.phases.sbm_forensics import run_sbm_morpheme
    t0 = time.time()
    run_sbm_morpheme()
    print(f"\nStep 45A.3 completed in {time.time() - t0:.1f}s")

def cmd_sbm_modifier_45():
    from voynich.phases.sbm_forensics import run_sbm_modifier
    t0 = time.time()
    run_sbm_modifier()
    print(f"\nStep 45A.4 completed in {time.time() - t0:.1f}s")

def cmd_sbm_combinat():
    from voynich.phases.sbm_forensics import run_sbm_combinat
    t0 = time.time()
    run_sbm_combinat()
    print(f"\nStep 45A.5 completed in {time.time() - t0:.1f}s")

def cmd_sbm_factor():
    from voynich.phases.sbm_forensics import run_sbm_factor
    t0 = time.time()
    run_sbm_factor()
    print(f"\nStep 45A.6 completed in {time.time() - t0:.1f}s")

def cmd_sbm_signal_45():
    from voynich.phases.sbm_forensics import run_sbm_signal
    t0 = time.time()
    run_sbm_signal()
    print(f"\nStep 45A.7 completed in {time.time() - t0:.1f}s")

def cmd_track_a_45():
    from voynich.phases.sbm_forensics import run_track_a_45
    t0 = time.time()
    run_track_a_45()
    print(f"\nTrack A (Phase 45) completed in {time.time() - t0:.1f}s")

# Track B: SBM Decode
def cmd_sbm_encode():
    from voynich.phases.sbm_decode import run_sbm_encode
    t0 = time.time()
    run_sbm_encode()
    print(f"\nStep 45B.1 completed in {time.time() - t0:.1f}s")

def cmd_sbm_csp():
    from voynich.phases.sbm_decode import run_sbm_csp
    t0 = time.time()
    run_sbm_csp()
    print(f"\nStep 45B.2 completed in {time.time() - t0:.1f}s")

def cmd_comm_signal():
    from voynich.phases.sbm_decode import run_comm_signal
    t0 = time.time()
    run_comm_signal()
    print(f"\nStep 45B.3 completed in {time.time() - t0:.1f}s")

def cmd_sbm_hybrid():
    from voynich.phases.sbm_decode import run_sbm_hybrid
    t0 = time.time()
    run_sbm_hybrid()
    print(f"\nStep 45B.4 completed in {time.time() - t0:.1f}s")

def cmd_sbm_landscape():
    from voynich.phases.sbm_decode import run_sbm_landscape
    t0 = time.time()
    run_sbm_landscape()
    print(f"\nStep 45B.5 completed in {time.time() - t0:.1f}s")

def cmd_track_b_45():
    from voynich.phases.sbm_decode import run_track_b_45
    t0 = time.time()
    run_track_b_45()
    print(f"\nTrack B (Phase 45) completed in {time.time() - t0:.1f}s")

# Track C: Triple Consolidation
def cmd_triple_tiers():
    from voynich.phases.triple_consolidation import run_triple_tiers
    t0 = time.time()
    run_triple_tiers()
    print(f"\nStep 45C.1 completed in {time.time() - t0:.1f}s")

def cmd_triple_ambig():
    from voynich.phases.triple_consolidation import run_triple_ambig
    t0 = time.time()
    run_triple_ambig()
    print(f"\nStep 45C.2 completed in {time.time() - t0:.1f}s")

def cmd_triple_lock():
    from voynich.phases.triple_consolidation import run_triple_lock
    t0 = time.time()
    run_triple_lock()
    print(f"\nStep 45C.3 completed in {time.time() - t0:.1f}s")

def cmd_triple_impact():
    from voynich.phases.triple_consolidation import run_triple_impact
    t0 = time.time()
    run_triple_impact()
    print(f"\nStep 45C.4 completed in {time.time() - t0:.1f}s")

def cmd_track_c_45():
    from voynich.phases.triple_consolidation import run_track_c_45
    t0 = time.time()
    run_track_c_45()
    print(f"\nTrack C (Phase 45) completed in {time.time() - t0:.1f}s")

# Phase 45 Integration
def cmd_phase45_integrate():
    from voynich.phases.phase45_integrate import run_phase45_integrate
    t0 = time.time()
    run_phase45_integrate()
    print(f"\nPhase 45 integration completed in {time.time() - t0:.1f}s")

def cmd_phase45():
    """Run full Phase 45 pipeline: SBM Forensics, SBM Decode, Triple Consolidation."""
    print("=" * 70)
    print("PHASE 45: SBM Community Forensics and Distributional Re-encoding")
    print("=" * 70)
    # Track A: SBM Forensics
    print("\n" + "=" * 70)
    print("TRACK A: SBM Community Forensics")
    print("=" * 70)
    cmd_track_a_45()
    # Track B: SBM Decode
    print("\n" + "=" * 70)
    print("TRACK B: SBM-Based Re-encoding and Decoding")
    print("=" * 70)
    cmd_track_b_45()
    # Track C: Triple Consolidation
    print("\n" + "=" * 70)
    print("TRACK C: Triple Confidence Consolidation")
    print("=" * 70)
    cmd_track_c_45()
    # Integration
    print("\n" + "=" * 70)
    print("INTEGRATION")
    print("=" * 70)
    cmd_phase45_integrate()


# Phase 46: Final Internal Consolidation
# Track A: Triple Arbitration
def cmd_arb_tables():
    from voynich.phases.triple_arbitration import run_arb_tables
    t0 = time.time()
    run_arb_tables()
    print(f"\nTable assembly completed in {time.time() - t0:.1f}s")

def cmd_arb_bigram():
    from voynich.phases.triple_arbitration import run_arb_bigram
    t0 = time.time()
    run_arb_bigram()
    print(f"\nBigram z computation completed in {time.time() - t0:.1f}s")

def cmd_arb_signal():
    from voynich.phases.triple_arbitration import run_arb_signal
    t0 = time.time()
    run_arb_signal()
    print(f"\nSignal word survival completed in {time.time() - t0:.1f}s")

def cmd_arb_10k():
    from voynich.phases.triple_arbitration import run_arb_10k
    t0 = time.time()
    run_arb_10k()
    print(f"\nDict-hit 10K completed in {time.time() - t0:.1f}s")

def cmd_arb_select():
    from voynich.phases.triple_arbitration import run_arb_select
    t0 = time.time()
    run_arb_select()
    print(f"\nTable selection completed in {time.time() - t0:.1f}s")

def cmd_track_a_46():
    from voynich.phases.triple_arbitration import run_track_a_46
    t0 = time.time()
    run_track_a_46()
    print(f"\nTrack A (Phase 46) completed in {time.time() - t0:.1f}s")

# Track B: Frequency Structure Diagnostic
def cmd_freq_reference():
    from voynich.phases.frequency_diagnostic import run_freq_reference
    t0 = time.time()
    run_freq_reference()
    print(f"\nFrequency reference SBM completed in {time.time() - t0:.1f}s")

def cmd_freq_cipher():
    from voynich.phases.frequency_diagnostic import run_freq_cipher
    t0 = time.time()
    run_freq_cipher()
    print(f"\nFrequency cipher SBM completed in {time.time() - t0:.1f}s")

def cmd_freq_compare():
    from voynich.phases.frequency_diagnostic import run_freq_compare
    t0 = time.time()
    run_freq_compare()
    print(f"\nFrequency comparison completed in {time.time() - t0:.1f}s")

def cmd_track_b_46():
    from voynich.phases.frequency_diagnostic import run_track_b_46
    t0 = time.time()
    run_track_b_46()
    print(f"\nTrack B (Phase 46) completed in {time.time() - t0:.1f}s")

# Track C: Definitive Corpus Decode and Gap Map
def cmd_final_decode():
    from voynich.phases.final_decode import run_final_decode
    t0 = time.time()
    run_final_decode()
    print(f"\nFinal decode completed in {time.time() - t0:.1f}s")

def cmd_final_annotate():
    from voynich.phases.final_decode import run_final_annotate
    t0 = time.time()
    run_final_annotate()
    print(f"\nFinal annotation completed in {time.time() - t0:.1f}s")

def cmd_final_map():
    from voynich.phases.final_decode import run_final_map
    t0 = time.time()
    run_final_map()
    print(f"\nGap map completed in {time.time() - t0:.1f}s")

def cmd_final_summary():
    from voynich.phases.final_decode import run_final_summary
    t0 = time.time()
    run_final_summary()
    print(f"\nProject summary completed in {time.time() - t0:.1f}s")

def cmd_track_c_46():
    from voynich.phases.final_decode import run_track_c_46
    t0 = time.time()
    run_track_c_46()
    print(f"\nTrack C (Phase 46) completed in {time.time() - t0:.1f}s")

# Phase 46 Integration
def cmd_phase46_integrate():
    from voynich.phases.phase46_integrate import run_phase46_integrate
    t0 = time.time()
    run_phase46_integrate()
    print(f"\nPhase 46 integration completed in {time.time() - t0:.1f}s")

def cmd_phase46():
    from voynich.phases.phase46_integrate import run_phase46
    t0 = time.time()
    run_phase46()
    print(f"\nPhase 46 completed in {time.time() - t0:.1f}s")


# Phase 47: Z-Score Audit, Disambiguation, Structural Reading, Sequence
# Track A: Z-Score Audit
def cmd_z_reproduce_42():
    from voynich.phases.zscore_audit import run_z_reproduce_42
    t0 = time.time()
    run_z_reproduce_42()
    print(f"\nStep 47A.1 completed in {time.time() - t0:.1f}s")

def cmd_z_reproduce_46():
    from voynich.phases.zscore_audit import run_z_reproduce_46
    t0 = time.time()
    run_z_reproduce_46()
    print(f"\nStep 47A.2 completed in {time.time() - t0:.1f}s")

def cmd_z_diff():
    from voynich.phases.zscore_audit import run_z_diff
    t0 = time.time()
    run_z_diff()
    print(f"\nStep 47A.3 completed in {time.time() - t0:.1f}s")

def cmd_z_canonical():
    from voynich.phases.zscore_audit import run_z_canonical
    t0 = time.time()
    run_z_canonical()
    print(f"\nStep 47A.4 completed in {time.time() - t0:.1f}s")

def cmd_z_sensitivity():
    from voynich.phases.zscore_audit import run_z_sensitivity
    t0 = time.time()
    run_z_sensitivity()
    print(f"\nStep 47A.5 completed in {time.time() - t0:.1f}s")

def cmd_track_a_47():
    from voynich.phases.zscore_audit import run_track_a_47
    t0 = time.time()
    run_track_a_47()
    print(f"\nTrack A (Phase 47) completed in {time.time() - t0:.1f}s")

# Track B: Word-Level Disambiguation
def cmd_disamb_lattice():
    from voynich.phases.word_disambiguation import run_disamb_lattice
    t0 = time.time()
    run_disamb_lattice()
    print(f"\nStep 47B.1 completed in {time.time() - t0:.1f}s")

def cmd_disamb_bigram():
    from voynich.phases.word_disambiguation import run_disamb_bigram
    t0 = time.time()
    run_disamb_bigram()
    print(f"\nStep 47B.2 completed in {time.time() - t0:.1f}s")

def cmd_disamb_viterbi():
    from voynich.phases.word_disambiguation import run_disamb_viterbi
    t0 = time.time()
    run_disamb_viterbi()
    print(f"\nStep 47B.3 completed in {time.time() - t0:.1f}s")

def cmd_disamb_eval():
    from voynich.phases.word_disambiguation import run_disamb_eval
    t0 = time.time()
    run_disamb_eval()
    print(f"\nStep 47B.4 completed in {time.time() - t0:.1f}s")

def cmd_disamb_compare():
    from voynich.phases.word_disambiguation import run_disamb_compare
    t0 = time.time()
    run_disamb_compare()
    print(f"\nStep 47B.5 completed in {time.time() - t0:.1f}s")

def cmd_track_b_47():
    from voynich.phases.word_disambiguation import run_track_b_47
    t0 = time.time()
    run_track_b_47()
    print(f"\nTrack B (Phase 47) completed in {time.time() - t0:.1f}s")

# Track C: Structural Reading
def cmd_read_ngrams():
    from voynich.phases.structural_reading_47 import run_read_ngrams
    t0 = time.time()
    run_read_ngrams()
    print(f"\nStep 47C.1 completed in {time.time() - t0:.1f}s")

def cmd_read_recipes():
    from voynich.phases.structural_reading_47 import run_read_recipes
    t0 = time.time()
    run_read_recipes()
    print(f"\nStep 47C.2 completed in {time.time() - t0:.1f}s")

def cmd_read_topics():
    from voynich.phases.structural_reading_47 import run_read_topics
    t0 = time.time()
    run_read_topics()
    print(f"\nStep 47C.3 completed in {time.time() - t0:.1f}s")

def cmd_read_star():
    from voynich.phases.structural_reading_47 import run_read_star
    t0 = time.time()
    run_read_star()
    print(f"\nStep 47C.4 completed in {time.time() - t0:.1f}s")

def cmd_read_sections():
    from voynich.phases.structural_reading_47 import run_read_sections
    t0 = time.time()
    run_read_sections()
    print(f"\nStep 47C.5 completed in {time.time() - t0:.1f}s")

def cmd_track_c_47():
    from voynich.phases.structural_reading_47 import run_track_c_47
    t0 = time.time()
    run_track_c_47()
    print(f"\nTrack C (Phase 47) completed in {time.time() - t0:.1f}s")

# Track D: Sequence Analysis
def cmd_seq_overlap():
    from voynich.phases.sequence_analysis import run_seq_overlap
    t0 = time.time()
    run_seq_overlap()
    print(f"\nStep 47D.1 completed in {time.time() - t0:.1f}s")

def cmd_seq_continuity():
    from voynich.phases.sequence_analysis import run_seq_continuity
    t0 = time.time()
    run_seq_continuity()
    print(f"\nStep 47D.2 completed in {time.time() - t0:.1f}s")

def cmd_seq_boundary():
    from voynich.phases.sequence_analysis import run_seq_boundary
    t0 = time.time()
    run_seq_boundary()
    print(f"\nStep 47D.3 completed in {time.time() - t0:.1f}s")

def cmd_seq_reorder():
    from voynich.phases.sequence_analysis import run_seq_reorder
    t0 = time.time()
    run_seq_reorder()
    print(f"\nStep 47D.4 completed in {time.time() - t0:.1f}s")

def cmd_track_d_47():
    from voynich.phases.sequence_analysis import run_track_d_47
    t0 = time.time()
    run_track_d_47()
    print(f"\nTrack D (Phase 47) completed in {time.time() - t0:.1f}s")

# Phase 47 Integration
def cmd_phase47_integrate():
    from voynich.phases.phase47_integrate import run_phase47_integrate
    t0 = time.time()
    run_phase47_integrate()
    print(f"\nPhase 47 integration completed in {time.time() - t0:.1f}s")

def cmd_phase47():
    from voynich.phases.phase47_integrate import run_phase47
    t0 = time.time()
    run_phase47()
    print(f"\nPhase 47 completed in {time.time() - t0:.1f}s")

# Phase 48: Marginal Bilingual Crib Exploitation

def cmd_f116v_transcribe():
    from voynich.phases.marginal_cribs import run_f116v_transcribe
    run_f116v_transcribe()

def cmd_f116v_decode():
    from voynich.phases.marginal_cribs import run_f116v_decode
    run_f116v_decode()

def cmd_f116v_context():
    from voynich.phases.marginal_cribs import run_f116v_context
    run_f116v_context()

def cmd_f116v_match():
    from voynich.phases.marginal_cribs import run_f116v_match
    run_f116v_match()

def cmd_f116v_reverse():
    from voynich.phases.marginal_cribs import run_f116v_reverse
    run_f116v_reverse()

def cmd_track_a_48():
    from voynich.phases.marginal_cribs import run_track_a_48
    run_track_a_48()

def cmd_f17r_extract():
    from voynich.phases.marginal_secondary import run_f17r_extract
    run_f17r_extract()

def cmd_f66r_extract():
    from voynich.phases.marginal_secondary import run_f66r_extract
    run_f66r_extract()

def cmd_margin_decode():
    from voynich.phases.marginal_secondary import run_margin_decode
    run_margin_decode()

def cmd_margin_hand():
    from voynich.phases.marginal_secondary import run_margin_hand
    run_margin_hand()

def cmd_track_b_48():
    from voynich.phases.marginal_secondary import run_track_b_48
    run_track_b_48()

def cmd_marci_source():
    from voynich.phases.marci_annotations import run_marci_source
    run_marci_source()

def cmd_marci_extract():
    from voynich.phases.marci_annotations import run_marci_extract
    run_marci_extract()

def cmd_marci_compare():
    from voynich.phases.marci_annotations import run_marci_compare
    run_marci_compare()

def cmd_marci_test():
    from voynich.phases.marci_annotations import run_marci_test
    run_marci_test()

def cmd_track_c_48():
    from voynich.phases.marci_annotations import run_track_c_48
    run_track_c_48()

def cmd_crib_collect():
    from voynich.phases.bilingual_propagation import run_crib_collect
    run_crib_collect()

def cmd_crib_consistent_48():
    from voynich.phases.bilingual_propagation import run_crib_consistent
    run_crib_consistent()

def cmd_crib_propagate():
    from voynich.phases.bilingual_propagation import run_crib_propagate
    run_crib_propagate()

def cmd_crib_decode_48():
    from voynich.phases.bilingual_propagation import run_crib_decode
    run_crib_decode()

def cmd_crib_validate():
    from voynich.phases.bilingual_propagation import run_crib_validate
    run_crib_validate()

def cmd_track_d_48():
    from voynich.phases.bilingual_propagation import run_track_d_48
    run_track_d_48()

def cmd_phase48_integrate():
    from voynich.phases.phase48_integrate import run_phase48_integrate
    run_phase48_integrate()

def cmd_phase48():
    from voynich.phases.phase48_integrate import run_phase48
    run_phase48()


def cmd_phase24():
    """Run full Phase 24 pipeline: targeted error correction + exploratory analysis."""
    print("=" * 70)
    print("PHASE 24: Targeted Error Correction and Exploratory Analysis")
    print("=" * 70)
    # Part A: Error Correction
    print("\n" + "=" * 70)
    print("PART A: Error Correction")
    print("=" * 70)
    cmd_triple_loo()
    print("\n" + "=" * 70 + "\n")
    cmd_error_id()
    print("\n" + "=" * 70 + "\n")
    cmd_triple_swap()
    print("\n" + "=" * 70 + "\n")
    cmd_bigram_val()
    print("\n" + "=" * 70 + "\n")
    cmd_corrected_tab()
    print("\n" + "=" * 70 + "\n")
    cmd_corrected_decode()
    print("\n" + "=" * 70 + "\n")
    cmd_corrected_read()
    # Part B: Exploratory Analyses
    print("\n" + "=" * 70)
    print("PART B: Exploratory Analyses")
    print("=" * 70)
    cmd_word_bound()
    print("\n" + "=" * 70 + "\n")
    cmd_ligature_test()
    print("\n" + "=" * 70 + "\n")
    cmd_direction()
    print("\n" + "=" * 70 + "\n")
    cmd_crib_search()
    print("\n" + "=" * 70 + "\n")
    cmd_folio_deep()
    print("\n" + "=" * 70 + "\n")
    cmd_section_xfer()
    print("\n" + "=" * 70 + "\n")
    cmd_reverse_eng()
    print("\n" + "=" * 70 + "\n")
    cmd_token_gram()
    # Integration
    print("\n" + "=" * 70)
    print("INTEGRATION")
    print("=" * 70)
    cmd_phase24_integrate()


def main():
    commands = {
        'corpus': cmd_corpus,
        'reference': cmd_reference,
        'strokes': cmd_strokes,
        'fingerprint': cmd_fingerprint,
        'both': cmd_both,
        'nulls': cmd_nulls,
        'grid': cmd_grid,
        'phase2': cmd_phase2,
        'degeneracy': cmd_degeneracy,
        'grid-validate': cmd_grid_validate,
        'syllable-match': cmd_syllable_match,
        'validate-all': cmd_validate_all,
        'phase3': cmd_phase3,
        'audit': cmd_audit,
        'section-diagnosis': cmd_section_diagnosis,
        'abugida': cmd_abugida,
        'multi-language': cmd_multi_language,
        'phase4': cmd_phase4,
        'lang-a': cmd_lang_a,
        'morpheme-grid': cmd_morpheme_grid,
        'phase4-5': cmd_phase45,
        'paradigms': cmd_paradigms,
        'paradigm-match': cmd_paradigm_match,
        'stem-id': cmd_stem_id,
        'phonetic': cmd_phonetic,
        'phase5': cmd_phase5,
        'illustration': cmd_illustration,
        'rosetta': cmd_rosetta,
        'anchor': cmd_anchor,
        'compete': cmd_compete,
        'phase6-validate': cmd_phase6_validate,
        'phase6': cmd_phase6,
        'anchor-diagnosis': cmd_anchor_diagnosis,
        'encoding-diagnosis': cmd_encoding_diagnosis,
        'phase6-1': cmd_phase61,
        'embeddings': cmd_embeddings,
        'slots': cmd_slots,
        'phase7': cmd_phase7,
        'combined-embed': cmd_combined_embed,
        'noun-clusters': cmd_noun_clusters,
        'verb-id': cmd_verb_id,
        'embed-bridge': cmd_embed_bridge,
        'convergence': cmd_convergence,
        'phase7-5': cmd_phase75,
        'bigram-transfer': cmd_bigram_transfer,
        'mdl-decode': cmd_mdl_decode,
        'cipher-validate': cmd_cipher_validate,
        'phase8': cmd_phase8,
        'nomenclator': cmd_nomenclator,
        'homophones': cmd_homophones,
        'position-dep': cmd_position_dep,
        'lang-compare': cmd_lang_compare,
        'typology': cmd_typology,
        'phase9': cmd_phase9,
        'entropy-curves': cmd_entropy_curves,
        'mi-decay': cmd_mi_decay,
        'folio-shift': cmd_folio_shift,
        'glyph-grammar': cmd_glyph_grammar,
        'hypothesis': cmd_hypothesis,
        'phase10': cmd_phase10,
        'csp-solve': cmd_csp_solve,
        'csp-decode': cmd_csp_decode,
        'csp-validate': cmd_csp_validate,
        'phase11': cmd_phase11,
        'csp-diagnose': cmd_csp_diagnose,
        'csp-refine': cmd_csp_refine,
        'verb-constrain': cmd_verb_constrain,
        'csp-iterate': cmd_csp_iterate,
        'csp-final': cmd_csp_final,
        'phase11-5': cmd_phase115,
        'grid-recal': cmd_grid_recal,
        'grid-alt': cmd_grid_alt,
        'token-decomp': cmd_token_decomp,
        'recal-csp': cmd_recal_csp,
        'phase12': cmd_phase12,
        'error-patterns': cmd_error_patterns,
        'null-context': cmd_null_context,
        'extract-rules': cmd_extract_rules,
        'context-csp': cmd_context_csp,
        'rule-validate': cmd_rule_validate,
        'context-decode': cmd_context_decode,
        'phase13': cmd_phase13,
        'cell-analysis': cmd_cell_analysis,
        'stroke-features': cmd_stroke_features,
        'feature-csp': cmd_feature_csp,
        'feature-calibrate': cmd_feature_calibrate,
        'feature-decode': cmd_feature_decode,
        'subcell-split': cmd_subcell_split,
        'phase14': cmd_phase14,
        'dict-expand': cmd_dict_expand,
        'artic-csp': cmd_artic_csp,
        'iter-hits': cmd_iter_hits,
        'combined-refine': cmd_combined_refine,
        'text-analysis': cmd_text_analysis,
        'phase15-validate': cmd_phase15_validate,
        'phase15': cmd_phase15,
        # Phase 16
        'mod-standalone': cmd_mod_standalone,
        'mod-anomaly': cmd_mod_anomaly,
        'mod-distrib': cmd_mod_distrib,
        'mod-pairs': cmd_mod_pairs,
        'mod-localize': cmd_mod_localize,
        'mod-integrate': cmd_mod_integrate,
        'phase16': cmd_phase16,
        # Phase 17 Step 0
        'honesty-dict': cmd_honesty_dict,
        'honesty-keywords': cmd_honesty_keywords,
        'honesty-verbs': cmd_honesty_verbs,
        'null-corpus': cmd_null_corpus,
        'honesty-words': cmd_honesty_words,
        'step0-integrate': cmd_step0_integrate,
        'step0': cmd_step0,
        # Phase A: Paleographic Reference Inventory
        'ref-validate': cmd_ref_validate,
        'ref-merge': cmd_ref_merge,
        'phaseA': cmd_phaseA,
        # Phase B: Structural Comparison
        'ligature-analysis': cmd_ligature_analysis,
        'triple-overlap': cmd_triple_overlap,
        'mod-tironian': cmd_modifier_tironian,
        'pos-compare': cmd_positional_compare,
        'cappelli-match': cmd_cappelli_match,
        'fontana-compare': cmd_fontana_compare,
        'phaseB': cmd_phaseB,
        # Phase C: CSP Re-Solve
        'tir-csp': cmd_tironian_csp,
        'phrase-detect': cmd_phrase_detect,
        'mod-clean': cmd_modifier_clean,
        'reseg-csp': cmd_reseg_csp,
        'phaseC-validate': cmd_phaseC_validate,
        'phaseC': cmd_phaseC,
        # Phase D: Parallel Historical Investigation
        'mil-fingerprint': cmd_milanese_fingerprint,
        'entropy-floor': cmd_entropy_floor,
        'verbose-enc': cmd_verbose_encoding,
        'phaseD': cmd_phaseD,
        # Phase 18: Hypothesis Discrimination Battery
        'burstiness': cmd_burstiness,
        'stride-entropy': cmd_stride_entropy,
        'trie-topology': cmd_trie_topology,
        'hmm-pos': cmd_hmm_pos,
        'lz-complexity': cmd_lz_complexity,
        'hyp-discriminate': cmd_hyp_discriminate,
        'phase18': cmd_phase18,
        # Phase 19: Convergent Constraint Exploitation
        'modifier-validate': cmd_modifier_validate,
        'affix-isolate': cmd_affix_isolate,
        'lang-b-attack': cmd_lang_b_attack,
        'entropy-shift': cmd_entropy_shift,
        'tachy-stroke': cmd_tachy_stroke,
        'cross-validate': cmd_cross_validate,
        'illus-target': cmd_illus_target,
        'stroke-sim': cmd_stroke_sim,
        'phase19-integrate': cmd_phase19_integrate,
        'phase19': cmd_phase19,
        # Phase 20: Tachygraphic Table Construction and Corpus Decoding
        'tachy-anchors': cmd_tachy_anchors,
        'tachy-families': cmd_tachy_families,
        'tachy-grid': cmd_tachy_grid,
        'tachy-decode': cmd_tachy_decode,
        'tachy-read': cmd_tachy_read,
        'tachy-phrases': cmd_tachy_phrases,
        'tachy-validate': cmd_tachy_validate,
        'phase20-integrate': cmd_phase20_integrate,
        'phase20': cmd_phase20,
        # Phase 21: Paleographic Sign Comparison
        'paleo-ingest': cmd_paleo_ingest,
        'fontana-families': cmd_fontana_families,
        'chatelain-families': cmd_chatelain_families,
        'eva-compare': cmd_eva_compare,
        'family-syllable': cmd_family_syllable,
        'cappelli-mod': cmd_cappelli_mod,
        'paleo-table': cmd_paleo_table,
        'paleo-decode': cmd_paleo_decode,
        'paleo-validate': cmd_paleo_validate,
        'phase21-integrate': cmd_phase21_integrate,
        'phase21': cmd_phase21,
        # Phase 22: First-Syllable Extraction & Fontana-Constrained Decode
        'first-syl': cmd_first_syl,
        'fontana-phon': cmd_fontana_phon,
        'table-merge': cmd_table_merge,
        'decode-22': cmd_decode_22,
        'read-22': cmd_read_22,
        'phrases-22': cmd_phrases_22,
        'validate-22': cmd_validate_22,
        'phase22-integrate': cmd_phase22_integrate,
        'phase22': cmd_phase22,
        # Phase 23: Statistical Inversion Analysis
        'ceiling': cmd_ceiling,
        'hist-invert': cmd_hist_invert,
        'bench-split': cmd_bench_split,
        'perm-search': cmd_perm_search,
        'read-delta': cmd_read_delta,
        'phase23': cmd_phase23,
        # Phase 24: Targeted Error Correction and Exploratory Analysis
        'triple-loo': cmd_triple_loo,
        'error-id': cmd_error_id,
        'triple-swap': cmd_triple_swap,
        'bigram-val': cmd_bigram_val,
        'corrected-tab': cmd_corrected_tab,
        'corrected-decode': cmd_corrected_decode,
        'corrected-read': cmd_corrected_read,
        'word-bound': cmd_word_bound,
        'ligature-test': cmd_ligature_test,
        'direction': cmd_direction,
        'crib-search': cmd_crib_search,
        'folio-deep': cmd_folio_deep,
        'section-xfer': cmd_section_xfer,
        'reverse-eng': cmd_reverse_eng,
        'token-gram': cmd_token_gram,
        'phase24-integrate': cmd_phase24_integrate,
        'phase24': cmd_phase24,
        # Phase 25: Reading Direction Test and f6r Examination
        'boustro': cmd_boustro,
        'f6r-exam': cmd_f6r_exam,
        'phase25-verdict': cmd_phase25_verdict,
        'phase25': cmd_phase25,
        # Phase 26: Zodiac Known-Plaintext Attack
        'zodiac-map': cmd_zodiac_map,
        'month-crib': cmd_month_crib,
        'astro-crib': cmd_astro_crib,
        'label-decode': cmd_label_decode,
        'zodiac-tab': cmd_zodiac_tab,
        'zodiac-decode': cmd_zodiac_decode,
        'phase26-validate': cmd_phase26_validate,
        'phase26-verdict': cmd_phase26_verdict,
        'phase26': cmd_phase26,
        # Phase 27: Peer Review Controls
        'gibberish-test': cmd_gibberish_test,
        'naibbe-test': cmd_naibbe_test,
        'phase27-verdict': cmd_phase27_verdict,
        'phase27': cmd_phase27,
        # Phase 28: Ventris-Style Crib Propagation
        'crib-extract': cmd_crib_extract,
        'crib-consist': cmd_crib_consist,
        'family-prop': cmd_family_prop,
        'signal-iso': cmd_signal_iso,
        'crib-local': cmd_crib_local,
        'ventris-tab': cmd_ventris_tab,
        'ventris-decode': cmd_ventris_decode,
        'ventris-read': cmd_ventris_read,
        'phase28-verdict': cmd_phase28_verdict,
        'phase28': cmd_phase28,
        # Phase 29: Signal-Filtered Readability
        'signal-bigram': cmd_signal_bigram,
        'signal-context': cmd_signal_context,
        'signal-folio': cmd_signal_folio,
        'signal-phrase': cmd_signal_phrase,
        'phase29-verdict': cmd_phase29_verdict,
        'phase29': cmd_phase29,
        # Phase 30: Iterative Ventris Bootstrap
        'bootstrap': cmd_bootstrap,
        'boot-signal': cmd_boot_signal,
        'boot-bigram': cmd_boot_bigram,
        'boot-context': cmd_boot_context,
        'boot-folio': cmd_boot_folio,
        'boot-read': cmd_boot_read,
        'phase30-verdict': cmd_phase30_verdict,
        'phase30': cmd_phase30,
        # Phase 31: Botanical Anchors + Structural Reframing
        'consensus-plants': cmd_consensus_plants,
        'plant-csp': cmd_plant_csp,
        'plant-prop': cmd_plant_prop,
        'bot-signal': cmd_bot_signal,
        'determ-test': cmd_determ_test,
        'compound-test': cmd_compound_test,
        'interleave-test': cmd_interleave_test,
        'reseg-test': cmd_reseg_test,
        'phase31-integrate': cmd_phase31_integrate,
        'phase31': cmd_phase31,
        # Phase 32: Compound-Sign Signal Pipeline
        'comp-decode': cmd_comp_decode,
        'comp-signal': cmd_comp_signal,
        'comp-bigram': cmd_comp_bigram,
        'comp-context': cmd_comp_context,
        'comp-bootstrap': cmd_comp_bootstrap,
        'comp-folio': cmd_comp_folio,
        'comp-read': cmd_comp_read,
        'phase32-verdict': cmd_phase32_verdict,
        'phase32': cmd_phase32,
        # Phase 33: Multi-Vector Error Correction
        'anti-diag': cmd_anti_diag,
        'triple-rates': cmd_triple_rates,
        'signal-swap': cmd_signal_swap,
        'signal-correct': cmd_signal_correct,
        'latin-lm': cmd_latin_lm,
        'ppl-search': cmd_ppl_search,
        'ppl-validate': cmd_ppl_validate,
        'suffix-gram': cmd_suffix_gram,
        'suffix-search': cmd_suffix_search,
        'long-crib': cmd_long_crib,
        'long-csp': cmd_long_csp,
        'long-prop': cmd_long_prop,
        'pair-freq': cmd_pair_freq,
        'distrib-match': cmd_distrib_match,
        'distrib-validate': cmd_distrib_validate,
        'phase33-integrate': cmd_phase33_integrate,
        'phase33': cmd_phase33,
        # Phase 34: Encoding Model Reformation
        # Track G
        'dict-cal': cmd_dict_cal,
        # Track A
        'sigla-dict': cmd_sigla_dict,
        'abjad-csp': cmd_abjad_csp,
        'sigla-decode': cmd_sigla_decode,
        'abjad-signal': cmd_abjad_signal,
        # Track B
        'slot-vars': cmd_slot_vars,
        'slot-csp': cmd_slot_csp,
        'slot-signal': cmd_slot_signal,
        # Track C
        'mixed-lm': cmd_mixed_lm,
        'dialect-decode': cmd_dialect_decode,
        'dialect-signal': cmd_dialect_signal,
        # Track D
        'continua': cmd_continua,
        'reseg-decode': cmd_reseg_decode,
        'reseg-signal': cmd_reseg_signal,
        # Track E
        'gallows-geom': cmd_gallows_geom,
        'spatial-decode': cmd_spatial_decode,
        # Track F
        'vowel-ptr': cmd_vowel_ptr,
        'vowel-decode': cmd_vowel_decode,
        # Integration
        'phase34-integrate': cmd_phase34_integrate,
        'phase34': cmd_phase34,
        # Phase 35: Spatial Conditioning + 10K Dictionary
        'spatial-pre': cmd_spatial_pre,
        'comb-decode': cmd_comb_decode,
        'comb-signal': cmd_comb_signal,
        'comb-bigram': cmd_comb_bigram,
        'comb-context': cmd_comb_context,
        'comb-bootstrap': cmd_comb_bootstrap,
        'comb-folio': cmd_comb_folio,
        'comb-read': cmd_comb_read,
        'phase35-verdict': cmd_phase35_verdict,
        'phase35': cmd_phase35,
        # Phase 36: Full Signal Pipeline at 10K Dictionary
        'decode-10k': cmd_decode_10k,
        'signal-10k': cmd_signal_10k,
        'bigram-10k': cmd_bigram_10k,
        'context-10k': cmd_context_10k,
        'boot-10k': cmd_boot_10k,
        'folio-10k': cmd_folio_10k,
        'read-10k': cmd_read_10k,
        'phase36-verdict': cmd_phase36_verdict,
        'phase36': cmd_phase36,
        # Phase 37: Signal Decomposition, Concatenation, and Content Word Recovery
        'cons-group': cmd_cons_group,
        'cv-corr': cmd_cv_corr,
        'vowel-conf': cmd_vowel_conf,
        'pair-concat': cmd_pair_concat,
        'concat-signal': cmd_concat_signal,
        'concat-bigram': cmd_concat_bigram,
        'joint-target': cmd_joint_target,
        'joint-swap': cmd_joint_swap,
        'joint-val': cmd_joint_val,
        'f57v-eva': cmd_f57v_eva,
        'f57v-struct': cmd_f57v_struct,
        'ital-corpus': cmd_ital_corpus,
        'ital-10k': cmd_ital_10k,
        'ital-signal': cmd_ital_signal,
        'phase37-integrate': cmd_phase37_integrate,
        'phase37': cmd_phase37,
        # Phase 38: Macaronic Signal Pipeline
        'merged-dict': cmd_merged_dict,
        'merged-decode': cmd_merged_decode,
        'merged-signal': cmd_merged_signal,
        'merged-bigram': cmd_merged_bigram,
        'merged-context': cmd_merged_context,
        'merged-boot': cmd_merged_boot,
        'merged-concat': cmd_merged_concat,
        'merged-folio': cmd_merged_folio,
        'merged-read': cmd_merged_read,
        'phase38-verdict': cmd_phase38_verdict,
        'phase38': cmd_phase38,
        # Phase 39: Edit-Distance Bridge, Vowel Recovery, Macaronic Crib Exploitation
        # Track A
        'ed1-decomp': cmd_ed1_decomp,
        'vowel-map': cmd_vowel_map,
        'vowel-fix': cmd_vowel_fix,
        'corrected-sig': cmd_corrected_sig,
        # Track B
        'phrase-crib': cmd_phrase_crib,
        'phrase-align': cmd_phrase_align,
        'phrase-corr': cmd_phrase_corr,
        # Track C
        'ital-plant': cmd_ital_plant,
        'ital-bot-csp': cmd_ital_bot_csp,
        'bot-prop': cmd_bot_prop,
        # Track D
        'venetian-lex': cmd_venetian_lex,
        'venetian-dec': cmd_venetian_dec,
        'venetian-phr': cmd_venetian_phr,
        # Track E
        'amp-dict': cmd_amp_dict,
        'amp-signal': cmd_amp_signal,
        'amp-bigram': cmd_amp_bigram,
        # Integration
        'phase39-integrate': cmd_phase39_integrate,
        'phase39': cmd_phase39,
        # Phase 40: Venetian Reading, CVC Expansion, Folio Decipherment
        # Track A
        'ven-forms': cmd_ven_forms,
        'ven-match': cmd_ven_match,
        'ven-bigram': cmd_ven_bigram,
        'cc-reclass': cmd_cc_reclass,
        # Track B
        'cvc-inv': cmd_cvc_inv,
        'cvc-csp': cmd_cvc_csp,
        'cvc-signal': cmd_cvc_signal,
        'cvc-bigram': cmd_cvc_bigram,
        # Track C
        'syl-lex': cmd_syl_lex,
        'folio-recon': cmd_folio_recon,
        'f57v-read': cmd_f57v_read,
        'best-read': cmd_best_read,
        # Track D
        'drosera-con': cmd_drosera_con,
        'bot-pred': cmd_bot_pred,
        'bot-search': cmd_bot_search,
        # Integration
        'phase40-integrate': cmd_phase40_integrate,
        'phase40': cmd_phase40,
        # Phase 41: Venetian Validation, Lexicon Completion, f57v Reading, Botanical Fix
        # Track A
        'null-ven': cmd_null_ven,
        'ven-valid': cmd_ven_valid,
        'ven-sig': cmd_ven_sig,
        'ven-confirm': cmd_ven_confirm,
        # Track B
        'ungloss-analyze': cmd_ungloss_analyze,
        'ven-dict': cmd_ven_dict,
        'context-disamb': cmd_context_disamb,
        'full-lex': cmd_full_lex,
        # Track C
        'formula-seg': cmd_formula_seg,
        'inter-formula': cmd_inter_formula,
        'ingred-search': cmd_ingred_search,
        'f57v-complete': cmd_f57v_complete,
        # Track D
        'bot-fix': cmd_bot_fix,
        'drosera-prop': cmd_drosera_prop,
        'bot-pred-v2': cmd_bot_pred_v2,
        # Integration
        'phase41-integrate': cmd_phase41_integrate,
        'phase41': cmd_phase41,
        # Phase 42: Bigram Audit, Symmetric Revalidation, Ground-Truth
        'bigram-audit': cmd_bigram_audit,
        'sym-recompute': cmd_sym_recompute,
        'sig-reval': cmd_sig_reval,
        'sel-audit': cmd_sel_audit,
        'ground-truth': cmd_ground_truth,
        'phase42': cmd_phase42,
        # Phase 43: Re-Encoding Inversion, Structural Probing, Conditional Decoding
        # Approach 1
        'vm-finger': cmd_vm_finger,
        'tachy-enc': cmd_tachy_enc,
        'enc-search': cmd_enc_search,
        'inv-decode': cmd_inv_decode,
        'inv-validate': cmd_inv_validate,
        # Approach 4
        'sig-pos': cmd_sig_pos,
        'pos-profiles': cmd_pos_profiles,
        'cooccur': cmd_cooccur,
        'struct-read': cmd_struct_read,
        # Approach 5
        'hmm-arch': cmd_hmm_arch,
        'anchor-init': cmd_anchor_init,
        'bw-train': cmd_bw_train,
        'hmm-decode': cmd_hmm_decode,
        'hmm-signal': cmd_hmm_signal,
        # Integration
        'phase43-integrate': cmd_phase43_integrate,
        'phase43': cmd_phase43,
        # Phase 44: Solution Landscape Enumeration
        # Track A: MaxSAT
        'maxsat-encode': cmd_maxsat_encode,
        'maxsat-solve': cmd_maxsat_solve,
        'maxsat-landscape': cmd_maxsat_landscape,
        'maxsat-validate': cmd_maxsat_validate,
        'track-a': cmd_track_a,
        # Track B: SBM
        'sbm-graph': cmd_sbm_graph,
        'sbm-fit': cmd_sbm_fit,
        'sbm-compare': cmd_sbm_compare,
        'sbm-predict': cmd_sbm_predict,
        'sbm-validate': cmd_sbm_validate,
        'track-b': cmd_track_b,
        # Track C: CSA
        'kperm-energy': cmd_kperm_energy,
        'kperm-search': cmd_kperm_search,
        'kperm-analyze': cmd_kperm_analyze,
        'kperm-validate': cmd_kperm_validate,
        'track-c': cmd_track_c,
        # Phase 44 Integration
        'phase44-integrate': cmd_phase44_integrate,
        'phase44': cmd_phase44,
        # Phase 45: SBM Community Forensics
        'sbm-profile': cmd_sbm_profile,
        'sbm-position': cmd_sbm_position,
        'sbm-morpheme': cmd_sbm_morpheme,
        'sbm-modifier': cmd_sbm_modifier_45,
        'sbm-combinat': cmd_sbm_combinat,
        'sbm-factor': cmd_sbm_factor,
        'sbm-signal': cmd_sbm_signal_45,
        'track-a-45': cmd_track_a_45,
        'sbm-encode': cmd_sbm_encode,
        'sbm-csp': cmd_sbm_csp,
        'comm-signal': cmd_comm_signal,
        'sbm-hybrid': cmd_sbm_hybrid,
        'sbm-landscape': cmd_sbm_landscape,
        'track-b-45': cmd_track_b_45,
        'triple-tiers': cmd_triple_tiers,
        'triple-ambig': cmd_triple_ambig,
        'triple-lock': cmd_triple_lock,
        'triple-impact': cmd_triple_impact,
        'track-c-45': cmd_track_c_45,
        'phase45-integrate': cmd_phase45_integrate,
        'phase45': cmd_phase45,
        # Phase 46: Final Internal Consolidation
        # Track A: Triple Arbitration
        'arb-tables': cmd_arb_tables,
        'arb-bigram': cmd_arb_bigram,
        'arb-signal': cmd_arb_signal,
        'arb-10k': cmd_arb_10k,
        'arb-select': cmd_arb_select,
        'track-a-46': cmd_track_a_46,
        # Track B: Frequency Structure Diagnostic
        'freq-reference': cmd_freq_reference,
        'freq-cipher': cmd_freq_cipher,
        'freq-compare': cmd_freq_compare,
        'track-b-46': cmd_track_b_46,
        # Track C: Definitive Corpus Decode and Gap Map
        'final-decode': cmd_final_decode,
        'final-annotate': cmd_final_annotate,
        'final-map': cmd_final_map,
        'final-summary': cmd_final_summary,
        'track-c-46': cmd_track_c_46,
        # Phase 46 Integration
        'phase46-integrate': cmd_phase46_integrate,
        'phase46': cmd_phase46,
        # Phase 47: Z-Score Audit, Disambiguation, Structural Reading, Sequence
        # Track A: Z-Score Audit
        'z-reproduce-42': cmd_z_reproduce_42,
        'z-reproduce-46': cmd_z_reproduce_46,
        'z-diff': cmd_z_diff,
        'z-canonical': cmd_z_canonical,
        'z-sensitivity': cmd_z_sensitivity,
        'track-a-47': cmd_track_a_47,
        # Track B: Word-Level Disambiguation
        'disamb-lattice': cmd_disamb_lattice,
        'disamb-bigram': cmd_disamb_bigram,
        'disamb-viterbi': cmd_disamb_viterbi,
        'disamb-eval': cmd_disamb_eval,
        'disamb-compare': cmd_disamb_compare,
        'track-b-47': cmd_track_b_47,
        # Track C: Structural Reading
        'read-ngrams': cmd_read_ngrams,
        'read-recipes': cmd_read_recipes,
        'read-topics': cmd_read_topics,
        'read-star': cmd_read_star,
        'read-sections': cmd_read_sections,
        'track-c-47': cmd_track_c_47,
        # Track D: Sequence Analysis
        'seq-overlap': cmd_seq_overlap,
        'seq-continuity': cmd_seq_continuity,
        'seq-boundary': cmd_seq_boundary,
        'seq-reorder': cmd_seq_reorder,
        'track-d-47': cmd_track_d_47,
        # Phase 47 Integration
        'phase47-integrate': cmd_phase47_integrate,
        'phase47': cmd_phase47,
        # Phase 48: Marginal Bilingual Crib Exploitation
        # Track A: f116v decode
        'f116v-transcribe': cmd_f116v_transcribe,
        'f116v-decode': cmd_f116v_decode,
        'f116v-context': cmd_f116v_context,
        'f116v-match': cmd_f116v_match,
        'f116v-reverse': cmd_f116v_reverse,
        'track-a-48': cmd_track_a_48,
        # Track B: f17r/f66r marginal
        'f17r-extract': cmd_f17r_extract,
        'f66r-extract': cmd_f66r_extract,
        'margin-decode': cmd_margin_decode,
        'margin-hand': cmd_margin_hand,
        'track-b-48': cmd_track_b_48,
        # Track C: Marci annotations
        'marci-source': cmd_marci_source,
        'marci-extract': cmd_marci_extract,
        'marci-compare': cmd_marci_compare,
        'marci-test': cmd_marci_test,
        'track-c-48': cmd_track_c_48,
        # Track D: Crib propagation
        'crib-collect': cmd_crib_collect,
        'crib-consistent': cmd_crib_consistent_48,
        'crib-propagate': cmd_crib_propagate,
        'crib-decode': cmd_crib_decode_48,
        'crib-validate': cmd_crib_validate,
        'track-d-48': cmd_track_d_48,
        # Phase 48 Integration
        'phase48-integrate': cmd_phase48_integrate,
        'phase48': cmd_phase48,
    }

    if len(sys.argv) < 2:
        cmd_corpus()
        return

    command = sys.argv[1]
    if command in commands:
        commands[command]()
    elif command in ('-h', '--help', 'help'):
        print(__doc__)
    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
