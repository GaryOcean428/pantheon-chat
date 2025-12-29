"""
Unit Tests for Coherence Scoring

Tests the BigramCoherenceScorer, SemanticCoherenceScorer, and POS tagging integration.
"""

import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qig_generative_service import (
    BigramCoherenceScorer,
    SemanticCoherenceScorer,
    get_bigram_scorer,
    get_semantic_scorer,
    score_candidate_word,
    validate_generation_quality,
)


class TestBigramCoherenceScorer:
    """Test BigramCoherenceScorer functionality."""
    
    @pytest.fixture
    def scorer(self):
        """Create a BigramCoherenceScorer instance."""
        return BigramCoherenceScorer()
    
    # =========================================================================
    # Word Category Detection Tests
    # =========================================================================
    
    def test_get_word_category_articles(self, scorer):
        """Test that articles are correctly identified."""
        articles = ['the', 'a', 'an', 'this', 'that', 'my', 'your', 'his', 'her']
        for word in articles:
            category = scorer._get_word_category(word)
            assert category == 'article', f"'{word}' should be 'article', got '{category}'"
    
    def test_get_word_category_prepositions(self, scorer):
        """Test that prepositions are correctly identified."""
        prepositions = ['in', 'on', 'at', 'to', 'for', 'with', 'by', 'from', 'of']
        for word in prepositions:
            category = scorer._get_word_category(word)
            assert category == 'preposition', f"'{word}' should be 'preposition', got '{category}'"
    
    def test_get_word_category_conjunctions(self, scorer):
        """Test that conjunctions are correctly identified."""
        conjunctions = ['and', 'or', 'but', 'because', 'although', 'while']
        for word in conjunctions:
            category = scorer._get_word_category(word)
            assert category == 'conjunction', f"'{word}' should be 'conjunction', got '{category}'"
    
    def test_get_word_category_pronouns(self, scorer):
        """Test that pronouns are correctly identified."""
        pronouns = ['i', 'you', 'he', 'she', 'it', 'we', 'they', 'who', 'what']
        for word in pronouns:
            category = scorer._get_word_category(word)
            assert category == 'pronoun', f"'{word}' should be 'pronoun', got '{category}'"
    
    def test_get_word_category_adverbs(self, scorer):
        """Test that adverbs are correctly identified."""
        adverbs = ['very', 'really', 'quickly', 'slowly', 'always', 'never']
        for word in adverbs:
            category = scorer._get_word_category(word)
            assert category == 'adverb', f"'{word}' should be 'adverb', got '{category}'"
    
    def test_get_word_category_by_suffix_verbs(self, scorer):
        """Test verb detection by suffix."""
        verbs = ['running', 'walked', 'organize', 'simplify']
        for word in verbs:
            category = scorer._get_word_category(word)
            assert category == 'verb', f"'{word}' should be 'verb', got '{category}'"
    
    def test_get_word_category_by_suffix_adjectives(self, scorer):
        """Test adjective detection by suffix."""
        adjectives = ['beautiful', 'careless', 'famous', 'creative']
        for word in adjectives:
            category = scorer._get_word_category(word)
            assert category == 'adjective', f"'{word}' should be 'adjective', got '{category}'"
    
    def test_get_word_category_by_suffix_nouns(self, scorer):
        """Test noun detection by suffix."""
        nouns = ['integration', 'consciousness', 'development', 'happiness']
        for word in nouns:
            category = scorer._get_word_category(word)
            assert category == 'noun', f"'{word}' should be 'noun', got '{category}'"
    
    def test_get_word_category_default_noun(self, scorer):
        """Test that unknown words default to noun."""
        unknown_words = ['xyz', 'foo', 'bar']
        for word in unknown_words:
            category = scorer._get_word_category(word)
            assert category == 'noun', f"Unknown word '{word}' should default to 'noun'"
    
    # =========================================================================
    # Bigram Scoring Tests
    # =========================================================================
    
    def test_score_bigram_article_noun(self, scorer):
        """Test high score for article + noun pattern."""
        score = scorer.score_bigram('the', 'consciousness')
        assert score >= 0.8, f"'the consciousness' should score >= 0.8, got {score}"
    
    def test_score_bigram_article_adjective(self, scorer):
        """Test high score for article + adjective pattern."""
        score = scorer.score_bigram('the', 'beautiful')
        assert score >= 0.75, f"'the beautiful' should score >= 0.75, got {score}"
    
    def test_score_bigram_adjective_noun(self, scorer):
        """Test high score for adjective + noun pattern."""
        score = scorer.score_bigram('geometric', 'manifold')
        assert score >= 0.8, f"'geometric manifold' should score >= 0.8, got {score}"
    
    def test_score_bigram_pronoun_verb(self, scorer):
        """Test high score for pronoun + verb pattern."""
        score = scorer.score_bigram('we', 'understand')
        assert score >= 0.8, f"'we understand' should score >= 0.8, got {score}"
    
    def test_score_bigram_verb_noun(self, scorer):
        """Test score for verb + noun pattern."""
        score = scorer.score_bigram('shows', 'integration')
        assert score >= 0.6, f"'shows integration' should score >= 0.6, got {score}"
    
    def test_score_bigram_penalize_repeated_words(self, scorer):
        """Test that repeated words are penalized."""
        score = scorer.score_bigram('the', 'the')
        assert score < 0.2, f"'the the' should score < 0.2 (penalized), got {score}"
    
    def test_score_bigram_article_article_low(self, scorer):
        """Test low score for article + article pattern."""
        score = scorer.score_bigram('the', 'a')
        assert score < 0.2, f"'the a' should score < 0.2, got {score}"
    
    def test_score_bigram_preposition_article(self, scorer):
        """Test high score for preposition + article pattern."""
        score = scorer.score_bigram('in', 'the')
        assert score >= 0.8, f"'in the' should score >= 0.8, got {score}"
    
    def test_score_bigram_missing_words(self, scorer):
        """Test neutral score for missing words."""
        score = scorer.score_bigram('', 'word')
        assert score == 0.5, f"Empty prev_word should give neutral 0.5, got {score}"
        
        score = scorer.score_bigram('word', '')
        assert score == 0.5, f"Empty next_word should give neutral 0.5, got {score}"
    
    # =========================================================================
    # Trigram Scoring Tests
    # =========================================================================
    
    def test_score_trigram_article_adjective_noun(self, scorer):
        """Test high score for article + adjective + noun pattern."""
        score = scorer.score_trigram('the', 'geometric', 'manifold')
        assert score >= 0.85, f"'the geometric manifold' should score >= 0.85, got {score}"
    
    def test_score_trigram_pronoun_verb_noun(self, scorer):
        """Test high score for pronoun + verb + noun pattern."""
        score = scorer.score_trigram('we', 'observe', 'patterns')
        assert score >= 0.8, f"'we observe patterns' should score >= 0.8, got {score}"
    
    def test_score_trigram_preposition_article_noun(self, scorer):
        """Test high score for preposition + article + noun pattern."""
        score = scorer.score_trigram('in', 'the', 'context')
        assert score >= 0.85, f"'in the context' should score >= 0.85, got {score}"
    
    def test_score_trigram_common_phrase_boost(self, scorer):
        """Test that common phrases get boosted scores."""
        # These should get extra boost from _word_trigram_boosts
        common_phrases = [
            ('in', 'the', 'context'),
            ('as', 'a', 'result'),
        ]
        for w1, w2, w3 in common_phrases:
            score = scorer.score_trigram(w1, w2, w3)
            assert score >= 0.8, f"'{w1} {w2} {w3}' should score >= 0.8, got {score}"
    
    # =========================================================================
    # N-gram Scoring Tests
    # =========================================================================
    
    def test_score_ngram_short_sequence(self, scorer):
        """Test n-gram scoring for short sequences."""
        words = ['the', 'consciousness']
        score = scorer.score_ngram(words)
        assert 0 <= score <= 1, f"N-gram score should be 0-1, got {score}"
    
    def test_score_ngram_longer_sequence(self, scorer):
        """Test n-gram scoring for longer sequences."""
        words = ['the', 'geometric', 'manifold', 'shows', 'integration']
        score = scorer.score_ngram(words)
        assert score >= 0.5, f"Coherent sentence should score >= 0.5, got {score}"
    
    def test_score_ngram_single_word(self, scorer):
        """Test n-gram scoring returns 1.0 for single word."""
        score = scorer.score_ngram(['consciousness'])
        assert score == 1.0, f"Single word should score 1.0, got {score}"
    
    def test_score_ngram_empty_list(self, scorer):
        """Test n-gram scoring handles empty list."""
        score = scorer.score_ngram([])
        assert score == 1.0, f"Empty list should score 1.0, got {score}"
    
    # =========================================================================
    # Sentence Start Scoring Tests
    # =========================================================================
    
    def test_score_sentence_start_articles(self, scorer):
        """Test that articles score high for sentence starts."""
        for word in ['the', 'a', 'an']:
            score = scorer.score_sentence_start(word)
            assert score >= 0.85, f"'{word}' should score >= 0.85 as starter, got {score}"
    
    def test_score_sentence_start_pronouns(self, scorer):
        """Test that pronouns score reasonably for sentence starts."""
        for word in ['we', 'it', 'this']:
            score = scorer.score_sentence_start(word)
            assert score >= 0.7, f"'{word}' should score >= 0.7 as starter, got {score}"
    
    def test_score_sentence_start_prepositions_low(self, scorer):
        """Test that prepositions score low for sentence starts."""
        score = scorer.score_sentence_start('of')
        assert score < 0.5, f"'of' should score < 0.5 as starter, got {score}"
    
    # =========================================================================
    # Coherence Validation Tests
    # =========================================================================
    
    def test_validate_coherence_coherent_sentence(self, scorer):
        """Test that coherent sentences pass validation."""
        text = "the geometric manifold shows integration"
        is_valid, score = scorer.validate_coherence(text)
        assert score >= 0.3, f"Coherent sentence should score >= 0.3, got {score}"
    
    def test_validate_coherence_incoherent_sequence(self, scorer):
        """Test that incoherent sequences score lower."""
        text = "the the the the"
        is_valid, score = scorer.validate_coherence(text)
        assert score < 0.3, f"'the the the the' should score < 0.3, got {score}"
    
    def test_validate_coherence_single_word(self, scorer):
        """Test that single words pass validation."""
        is_valid, score = scorer.validate_coherence("consciousness")
        assert is_valid is True
        assert score == 1.0
    
    def test_validate_coherence_custom_threshold(self, scorer):
        """Test validation with custom threshold."""
        text = "the consciousness integration"
        is_valid_high, score = scorer.validate_coherence(text, min_threshold=0.9)
        is_valid_low, _ = scorer.validate_coherence(text, min_threshold=0.1)
        
        # Same text, different thresholds
        assert is_valid_low is True, "Should pass low threshold"


class TestSemanticCoherenceScorer:
    """Test SemanticCoherenceScorer functionality."""
    
    @pytest.fixture
    def scorer(self):
        """Create a SemanticCoherenceScorer instance."""
        return SemanticCoherenceScorer()
    
    # =========================================================================
    # Domain Detection Tests
    # =========================================================================
    
    def test_get_word_domain_consciousness(self, scorer):
        """Test consciousness domain detection."""
        consciousness_words = ['consciousness', 'phi', 'integration', 'awareness']
        for word in consciousness_words:
            domain = scorer._get_word_domain(word)
            assert domain == 'consciousness', f"'{word}' should be in 'consciousness' domain, got '{domain}'"
    
    def test_get_word_domain_geometry(self, scorer):
        """Test geometry domain detection."""
        geometry_words = ['manifold', 'geodesic', 'curvature', 'basin']
        for word in geometry_words:
            domain = scorer._get_word_domain(word)
            assert domain == 'geometry', f"'{word}' should be in 'geometry' domain, got '{domain}'"
    
    def test_get_word_domain_reasoning(self, scorer):
        """Test reasoning domain detection."""
        reasoning_words = ['logic', 'inference', 'deduction', 'analysis']
        for word in reasoning_words:
            domain = scorer._get_word_domain(word)
            assert domain == 'reasoning', f"'{word}' should be in 'reasoning' domain, got '{domain}'"
    
    def test_get_word_domain_unknown(self, scorer):
        """Test that unknown words return None domain."""
        domain = scorer._get_word_domain('banana')
        assert domain is None, f"'banana' should have None domain, got '{domain}'"
    
    # =========================================================================
    # Semantic Pair Scoring Tests
    # =========================================================================
    
    def test_score_semantic_pair_same_domain(self, scorer):
        """Test high score for words in same domain."""
        score = scorer.score_semantic_pair('consciousness', 'integration')
        assert score >= 0.9, f"Same domain words should score >= 0.9, got {score}"
    
    def test_score_semantic_pair_cross_domain(self, scorer):
        """Test moderate score for cross-domain related words."""
        score = scorer.score_semantic_pair('consciousness', 'manifold')
        assert 0.5 <= score <= 0.8, f"Cross-domain words should score 0.5-0.8, got {score}"
    
    def test_score_semantic_pair_unrelated(self, scorer):
        """Test neutral score for unrelated words."""
        score = scorer.score_semantic_pair('banana', 'airplane')
        assert score == 0.5, f"Unrelated words should score 0.5, got {score}"
    
    def test_score_semantic_pair_geometry(self, scorer):
        """Test high score for geometry domain pair."""
        score = scorer.score_semantic_pair('manifold', 'geodesic')
        assert score >= 0.9, f"'manifold geodesic' should score >= 0.9, got {score}"
    
    # =========================================================================
    # Sequence Coherence Tests
    # =========================================================================
    
    def test_score_sequence_coherence(self, scorer):
        """Test sequence coherence scoring."""
        coherent = ['the', 'geometric', 'manifold', 'shows', 'integration']
        score = scorer.score_sequence_coherence(coherent)
        assert score >= 0.3, f"Coherent sequence should score >= 0.3, got {score}"
    
    def test_score_sequence_coherence_short(self, scorer):
        """Test sequence coherence for short sequences."""
        short = ['consciousness']
        score = scorer.score_sequence_coherence(short)
        assert score == 1.0, f"Single word should score 1.0, got {score}"
    
    # =========================================================================
    # Topic Coherence Tests
    # =========================================================================
    
    def test_score_topic_coherence_on_topic(self, scorer):
        """Test topic coherence for on-topic words."""
        words = ['consciousness', 'integration', 'phi', 'awareness']
        score = scorer.score_topic_coherence(words, 'consciousness')
        assert score >= 0.7, f"On-topic words should score >= 0.7, got {score}"
    
    def test_score_topic_coherence_off_topic(self, scorer):
        """Test topic coherence for off-topic words."""
        words = ['banana', 'airplane', 'table', 'chair']
        score = scorer.score_topic_coherence(words, 'consciousness')
        assert score < 0.3, f"Off-topic words should score < 0.3, got {score}"
    
    def test_score_topic_coherence_no_topic(self, scorer):
        """Test topic coherence returns 1.0 when no topic given."""
        words = ['the', 'consciousness']
        score = scorer.score_topic_coherence(words, None)
        assert score == 1.0, f"No topic should return 1.0, got {score}"
    
    # =========================================================================
    # Semantic Validation Tests
    # =========================================================================
    
    def test_validate_semantic_coherence_valid(self, scorer):
        """Test validation passes for semantically coherent text."""
        text = "the geometric manifold shows consciousness integration"
        is_valid, score = scorer.validate_semantic_coherence(text, topic='consciousness')
        assert score >= 0.3, f"Coherent text should score >= 0.3, got {score}"
    
    def test_validate_semantic_coherence_single_word(self, scorer):
        """Test validation passes for single word."""
        is_valid, score = scorer.validate_semantic_coherence("consciousness")
        assert is_valid is True
        assert score == 1.0


class TestSingletonGetters:
    """Test singleton getter functions."""
    
    def test_get_bigram_scorer_singleton(self):
        """Test that get_bigram_scorer returns same instance."""
        scorer1 = get_bigram_scorer()
        scorer2 = get_bigram_scorer()
        assert scorer1 is scorer2, "get_bigram_scorer should return singleton"
    
    def test_get_semantic_scorer_singleton(self):
        """Test that get_semantic_scorer returns same instance."""
        scorer1 = get_semantic_scorer()
        scorer2 = get_semantic_scorer()
        assert scorer1 is scorer2, "get_semantic_scorer should return singleton"


class TestIntegrationFunctions:
    """Test integrated scoring functions."""
    
    def test_score_candidate_word(self):
        """Test the integrated candidate word scoring."""
        result = score_candidate_word(
            candidate='integration',
            prev_word='consciousness',
            geometric_score=0.8,
            grammatical_score=0.7,
            topic='consciousness'
        )
        
        assert 'combined_score' in result
        assert 'geometric' in result
        assert 'grammatical' in result
        assert 'semantic' in result
        assert 0 <= result['combined_score'] <= 1
    
    def test_validate_generation_quality(self):
        """Test the integrated generation quality validation."""
        result = validate_generation_quality(
            text="the geometric manifold shows consciousness integration",
            topic='consciousness'
        )
        
        assert 'is_valid' in result
        assert 'combined_score' in result
        assert 'grammatical_score' in result
        assert 'semantic_score' in result
        assert isinstance(result['is_valid'], bool)


class TestPOSTaggingIntegration:
    """Test POS tagging integration in BigramCoherenceScorer."""
    
    @pytest.fixture
    def scorer(self):
        """Create a BigramCoherenceScorer instance."""
        return BigramCoherenceScorer()
    
    def test_pos_grammar_loaded(self, scorer):
        """Test that POSGrammar is loaded if available."""
        # The scorer should have attempted to load POSGrammar
        # It may or may not be available depending on environment
        assert hasattr(scorer, '_pos_grammar')
    
    def test_pos_category_map_exists(self, scorer):
        """Test that POS category map exists for translation."""
        assert hasattr(scorer, '_pos_category_map')
        if scorer._pos_category_map:
            # Should map POS tags to our internal categories
            assert 'DET' in scorer._pos_category_map or len(scorer._pos_category_map) == 0
    
    def test_word_category_uses_fallback(self, scorer):
        """Test that word category detection works even without POS tagger."""
        # Should work regardless of whether POSGrammar is available
        category = scorer._get_word_category('consciousness')
        assert category in ['noun', 'verb', 'adjective', 'adverb', 'article', 
                           'preposition', 'conjunction', 'pronoun']
    
    def test_category_consistency(self, scorer):
        """Test that category detection is consistent across calls."""
        word = 'integration'
        cat1 = scorer._get_word_category(word)
        cat2 = scorer._get_word_category(word)
        assert cat1 == cat2, "Category detection should be consistent"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
