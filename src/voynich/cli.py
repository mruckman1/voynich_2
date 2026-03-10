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
