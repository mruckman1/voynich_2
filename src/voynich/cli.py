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
