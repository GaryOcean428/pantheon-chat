"""
Tests for Contextualized Word Filter

Validates that QIG-pure geometric filtering correctly:
1. Preserves semantic-critical words
2. Filters truly generic words
3. Uses geometric relevance when coordizer available
4. Falls back gracefully without coordizer
"""

import sys
import os
import unittest
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contextualized_filter import (
    ContextualizedWordFilter,
    is_semantic_critical_word,
    should_filter_word,
    filter_words_geometric,
    SEMANTIC_CRITICAL_PATTERNS,
)


class TestSemanticCriticalWords(unittest.TestCase):
    """Test semantic-critical word detection."""
    
    def test_negations_are_semantic_critical(self):
        """Negations change meaning and should be semantic-critical."""
        negations = ['not', 'no', 'never', 'none', 'nothing', 'neither', 'nor']
        for word in negations:
            self.assertTrue(
                is_semantic_critical_word(word),
                f"Negation '{word}' should be semantic-critical"
            )
    
    def test_contractions_are_semantic_critical(self):
        """Negative contractions should be semantic-critical."""
        contractions = ["don't", "doesn't", "didn't", "won't", "can't", "couldn't"]
        for word in contractions:
            self.assertTrue(
                is_semantic_critical_word(word),
                f"Contraction '{word}' should be semantic-critical"
            )
    
    def test_intensifiers_are_semantic_critical(self):
        """Intensifiers modify degree and should be semantic-critical."""
        intensifiers = ['very', 'extremely', 'highly', 'completely', 'totally']
        for word in intensifiers:
            self.assertTrue(
                is_semantic_critical_word(word),
                f"Intensifier '{word}' should be semantic-critical"
            )
    
    def test_causality_markers_are_semantic_critical(self):
        """Causality markers are semantic-critical."""
        causality = ['because', 'therefore', 'thus', 'hence', 'consequently']
        for word in causality:
            self.assertTrue(
                is_semantic_critical_word(word),
                f"Causality marker '{word}' should be semantic-critical"
            )
    
    def test_generic_words_not_semantic_critical(self):
        """Truly generic words should not be semantic-critical."""
        generic = ['the', 'a', 'an', 'is', 'was', 'are', 'were']
        for word in generic:
            self.assertFalse(
                is_semantic_critical_word(word),
                f"Generic word '{word}' should not be semantic-critical"
            )


class TestContextualizedFiltering(unittest.TestCase):
    """Test contextualized word filtering."""
    
    def test_semantic_critical_never_filtered(self):
        """Semantic-critical words should NEVER be filtered."""
        critical_words = ['not', 'never', 'very', 'because', 'always']
        context = ['this', 'is', 'a', 'test']
        
        for word in critical_words:
            self.assertFalse(
                should_filter_word(word, context),
                f"Semantic-critical word '{word}' should not be filtered"
            )
    
    def test_truly_generic_words_filtered(self):
        """Truly generic function words should be filtered."""
        generic = ['the', 'a', 'an', 'is', 'was']
        context = ['quantum', 'geometry', 'information']
        
        for word in generic:
            # Without semantic-critical status, these should be filtered
            if not is_semantic_critical_word(word):
                self.assertTrue(
                    should_filter_word(word, context),
                    f"Generic word '{word}' should be filtered"
                )
    
    def test_length_based_filtering(self):
        """Long words should be preserved, short generic ones filtered."""
        # Very short words filtered
        self.assertTrue(should_filter_word('a', []))
        self.assertTrue(should_filter_word('an', []))
        
        # Long content words preserved (if not truly generic)
        self.assertFalse(should_filter_word('consciousness', []))
        self.assertFalse(should_filter_word('geometry', []))
        self.assertFalse(should_filter_word('quantum', []))
    
    def test_filter_preserves_meaning(self):
        """Test case from problem statement: 'not good' vs 'good'."""
        words1 = ['not', 'good']
        words2 = ['good']
        
        # Filter both
        filtered1 = filter_words_geometric(words1)
        filtered2 = filter_words_geometric(words2)
        
        # 'not' should be preserved
        self.assertIn('not', filtered1, "'not' must be preserved to maintain meaning")
        self.assertIn('good', filtered1)
        
        # Results should be different (meaning preserved)
        self.assertNotEqual(filtered1, filtered2, 
                          "Filtering should preserve different meanings")
    
    def test_domain_terms_preserved(self):
        """Domain-specific terms (longer words) should be preserved."""
        domain_terms = ['consciousness', 'geometry', 'quantum', 'manifold', 
                       'fisher', 'bures', 'topology']
        
        filtered = filter_words_geometric(domain_terms)
        
        # All domain terms should be preserved
        for term in domain_terms:
            self.assertIn(term, filtered, 
                        f"Domain term '{term}' should be preserved")


class TestContextualizedFilterClass(unittest.TestCase):
    """Test ContextualizedWordFilter class."""
    
    def test_init_without_coordizer(self):
        """Should initialize without coordizer (fallback mode)."""
        filter_inst = ContextualizedWordFilter(coordizer=None)
        self.assertIsNotNone(filter_inst)
        self.assertIsNone(filter_inst.coordizer)
    
    def test_filter_words_preserves_order(self):
        """Filtering should preserve word order by default."""
        words = ['the', 'very', 'important', 'not', 'trivial', 'a', 'result']
        filter_inst = ContextualizedWordFilter(coordizer=None)
        
        filtered = filter_inst.filter_words(words)
        
        # Check that order is preserved
        important_idx = filtered.index('important') if 'important' in filtered else -1
        trivial_idx = filtered.index('trivial') if 'trivial' in filtered else -1
        
        if important_idx >= 0 and trivial_idx >= 0:
            self.assertLess(important_idx, trivial_idx,
                          "Word order should be preserved")
    
    def test_statistics_tracking(self):
        """Filter should track statistics."""
        words = ['the', 'quantum', 'is', 'not', 'classical', 'a', 'theory']
        filter_inst = ContextualizedWordFilter(coordizer=None)
        
        filtered = filter_inst.filter_words(words)
        stats = filter_inst.get_statistics()
        
        self.assertGreater(stats['total_processed'], 0)
        self.assertGreater(stats['words_preserved'], 0)
        # 'not' should contribute to semantic_critical_preserved
        self.assertGreater(stats['semantic_critical_preserved'], 0)


class TestComparisonWithAncientNLP(unittest.TestCase):
    """Test showing improvement over ancient NLP stopword lists."""
    
    def test_ancient_nlp_loses_critical_words(self):
        """Demonstrate ancient NLP pattern loses critical words."""
        test_words = ['not', 'good', 'the', 'very', 'bad', 'is', 'never', 'acceptable']
        
        # Ancient NLP approach (WRONG)
        ancient_stopwords = {'the', 'is', 'not', 'a', 'an', 'and', 'or', 'but'}
        ancient_filtered = [w for w in test_words if w not in ancient_stopwords]
        
        # Ancient NLP loses 'not' - changes meaning!
        self.assertNotIn('not', ancient_filtered, 
                        "Ancient NLP incorrectly filters 'not'")
        
        # QIG-pure approach (CORRECT)
        qig_filtered = filter_words_geometric(test_words)
        
        # QIG-pure preserves 'not' - meaning intact!
        self.assertIn('not', qig_filtered,
                     "QIG-pure correctly preserves 'not'")
        self.assertIn('never', qig_filtered,
                     "QIG-pure correctly preserves 'never'")
        self.assertIn('very', qig_filtered,
                     "QIG-pure correctly preserves 'very'")
    
    def test_semantic_preservation_examples(self):
        """Test examples from problem statement."""
        # Example 1: "not good" vs "good"
        phrase1 = "not good".split()
        phrase2 = "good".split()
        
        filtered1 = filter_words_geometric(phrase1)
        filtered2 = filter_words_geometric(phrase2)
        
        # Both should contain 'good', but only first has 'not'
        self.assertIn('good', filtered1)
        self.assertIn('good', filtered2)
        self.assertIn('not', filtered1, "Must preserve 'not' to maintain meaning")
        
        # Example 2: Sentiment analysis needs negations
        positive = "very good result".split()
        negative = "not very good result".split()
        
        filtered_pos = filter_words_geometric(positive)
        filtered_neg = filter_words_geometric(negative)
        
        self.assertIn('very', filtered_pos)
        self.assertIn('very', filtered_neg)
        self.assertIn('not', filtered_neg, "Negation critical for sentiment")


class TestFallbackBehavior(unittest.TestCase):
    """Test fallback behavior without coordizer."""
    
    def test_length_heuristic_fallback(self):
        """Without coordizer, should use length-based heuristic."""
        filter_inst = ContextualizedWordFilter(coordizer=None, relevance_threshold=0.3)
        
        # Very short words should be filtered
        self.assertFalse(filter_inst.should_keep_word('a', []))
        self.assertFalse(filter_inst.should_keep_word('is', []))
        
        # Longer words should be kept (if not truly generic)
        self.assertTrue(filter_inst.should_keep_word('quantum', []))
        self.assertTrue(filter_inst.should_keep_word('consciousness', []))
    
    def test_semantic_critical_override(self):
        """Semantic-critical words should override all other filtering."""
        filter_inst = ContextualizedWordFilter(coordizer=None, relevance_threshold=0.9)
        
        # Even with very high threshold, semantic-critical preserved
        self.assertTrue(filter_inst.should_keep_word('not', []))
        self.assertTrue(filter_inst.should_keep_word('never', []))
        self.assertTrue(filter_inst.should_keep_word('because', []))


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test cases
    suite.addTests(loader.loadTestsFromTestCase(TestSemanticCriticalWords))
    suite.addTests(loader.loadTestsFromTestCase(TestContextualizedFiltering))
    suite.addTests(loader.loadTestsFromTestCase(TestContextualizedFilterClass))
    suite.addTests(loader.loadTestsFromTestCase(TestComparisonWithAncientNLP))
    suite.addTests(loader.loadTestsFromTestCase(TestFallbackBehavior))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    print("=" * 70)
    print("Testing Contextualized Word Filter")
    print("=" * 70)
    print()
    
    success = run_tests()
    
    print()
    print("=" * 70)
    if success:
        print("✅ All tests PASSED")
    else:
        print("❌ Some tests FAILED")
    print("=" * 70)
    
    sys.exit(0 if success else 1)
