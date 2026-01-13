"""
Tests for QIG-Pure Geometric Word Relationships

Validates that geometric relationships use:
1. Fisher-Rao distances (not PMI)
2. QFI-weighted attention (not frequency)
3. Ricci curvature for context-dependency
4. No basin modification
"""

import sys
import os
import unittest
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geometric_word_relationships import (
    GeometricWordRelationships,
    GeometricProperties,
    get_geometric_relationships
)


class MockCoordizer:
    """Mock coordizer for testing."""
    
    def __init__(self):
        self.vocab = {
            'quantum': np.array([0.1, 0.2, 0.3, 0.4] + [0.0] * 60, dtype=np.float32),
            'geometry': np.array([0.15, 0.25, 0.25, 0.35] + [0.0] * 60, dtype=np.float32),
            'not': np.array([0.9, 0.05, 0.03, 0.02] + [0.0] * 60, dtype=np.float32),
            'the': np.array([0.5, 0.5, 0.0, 0.0] + [0.0] * 60, dtype=np.float32),
            'consciousness': np.array([0.05, 0.15, 0.35, 0.45] + [0.0] * 60, dtype=np.float32),
        }
        
        # Normalize to unit sphere
        for word in self.vocab:
            self.vocab[word] = self._normalize(self.vocab[word])
        
        self.basin_coords = self.vocab
    
    def _normalize(self, v):
        """Normalize to unit sphere."""
        norm = np.linalg.norm(v)
        return v / (norm + 1e-10) if norm > 0 else v
    
    def get_basin_coords(self, word):
        """Get basin coordinates."""
        return self.vocab.get(word)


class TestGeometricProperties(unittest.TestCase):
    """Test geometric property computations."""
    
    def setUp(self):
        """Create mock coordizer and learner."""
        self.coordizer = MockCoordizer()
        self.learner = GeometricWordRelationships(coordizer=self.coordizer)
    
    def test_qfi_computation(self):
        """Test QFI computation."""
        # Get basins
        quantum_basin = self.coordizer.get_basin_coords('quantum')
        
        # Compute QFI
        qfi = self.learner.compute_qfi(quantum_basin)
        
        # QFI should be in [0, 1] range
        self.assertGreaterEqual(qfi, 0.0)
        self.assertLessEqual(qfi, 1.0)
    
    def test_curvature_computation(self):
        """Test Ricci curvature computation."""
        # Get basin
        not_basin = self.coordizer.get_basin_coords('not')
        
        # Get neighbors
        neighbors = [
            self.coordizer.get_basin_coords('quantum'),
            self.coordizer.get_basin_coords('geometry'),
        ]
        
        # Compute curvature
        curvature = self.learner.compute_ricci_curvature(not_basin, neighbors)
        
        # Curvature should be non-negative
        self.assertGreaterEqual(curvature, 0.0)
    
    def test_specificity_computation(self):
        """Test semantic specificity computation."""
        # Specific concept
        consciousness_basin = self.coordizer.get_basin_coords('consciousness')
        specificity = self.learner.compute_specificity(consciousness_basin)
        
        # Should be in [0, 1] range
        self.assertGreaterEqual(specificity, 0.0)
        self.assertLessEqual(specificity, 1.0)
    
    def test_geometric_role_classification(self):
        """Test geometric role classification."""
        # High QFI, low curvature, high specificity = content_bearing
        role1 = self.learner.classify_geometric_role(qfi=0.8, curvature=0.1, specificity=0.8)
        self.assertEqual(role1, 'content_bearing')
        
        # Low QFI = geometrically_unstable
        role2 = self.learner.classify_geometric_role(qfi=0.2, curvature=0.5, specificity=0.5)
        self.assertEqual(role2, 'geometrically_unstable')
        
        # High curvature = context_critical
        role3 = self.learner.classify_geometric_role(qfi=0.5, curvature=0.7, specificity=0.5)
        self.assertEqual(role3, 'context_critical')
        
        # Low curvature + low specificity = geometric_anchor
        role4 = self.learner.classify_geometric_role(qfi=0.5, curvature=0.1, specificity=0.2)
        self.assertEqual(role4, 'geometric_anchor')


class TestFisherRaoDistances(unittest.TestCase):
    """Test Fisher-Rao distance computations (not PMI)."""
    
    def setUp(self):
        """Create mock coordizer and learner."""
        self.coordizer = MockCoordizer()
        self.learner = GeometricWordRelationships(coordizer=self.coordizer)
    
    def test_get_related_words(self):
        """Test getting related words via Fisher-Rao distances."""
        # Get related words to 'quantum'
        related = self.learner.get_related_words('quantum', top_k=3)
        
        # Should return list of (word, similarity) tuples
        self.assertIsInstance(related, list)
        
        for word, similarity in related:
            self.assertIsInstance(word, str)
            self.assertIsInstance(similarity, float)
            self.assertGreaterEqual(similarity, 0.0)
            self.assertLessEqual(similarity, 1.0)
    
    def test_related_words_not_self(self):
        """Test that word doesn't appear in its own related words."""
        related = self.learner.get_related_words('quantum', top_k=5)
        
        related_words = [w for w, _ in related]
        self.assertNotIn('quantum', related_words)
    
    def test_distance_matrix(self):
        """Test Fisher-Rao distance matrix computation."""
        distances, vocab = self.learner.get_distance_matrix()
        
        # Should be square matrix
        self.assertEqual(distances.shape[0], distances.shape[1])
        self.assertEqual(distances.shape[0], len(vocab))
        
        # Diagonal should be zero (distance to self)
        for i in range(len(vocab)):
            self.assertAlmostEqual(distances[i, i], 0.0, places=5)
        
        # Matrix should be symmetric
        for i in range(len(vocab)):
            for j in range(i+1, len(vocab)):
                self.assertAlmostEqual(distances[i, j], distances[j, i], places=5)


class TestQFIWeightedAttention(unittest.TestCase):
    """Test QFI-weighted attention (not frequency-based)."""
    
    def setUp(self):
        """Create mock coordizer and learner."""
        self.coordizer = MockCoordizer()
        self.learner = GeometricWordRelationships(coordizer=self.coordizer)
    
    def test_attention_weights(self):
        """Test QFI-weighted attention computation."""
        # Query basin
        query_basin = self.coordizer.get_basin_coords('quantum')
        
        # Candidate words
        candidates = ['geometry', 'not', 'the', 'consciousness']
        
        # Compute attention weights
        weights = self.learner.compute_attention_weights(query_basin, candidates)
        
        # Should return dict
        self.assertIsInstance(weights, dict)
        self.assertEqual(len(weights), len(candidates))
        
        # All weights should be positive
        for word, weight in weights.items():
            self.assertGreater(weight, 0.0)
    
    def test_attention_uses_qfi_not_frequency(self):
        """Test that attention uses QFI, not frequency."""
        # This test verifies that GeometricWordRelationships doesn't
        # have any frequency counting or co-occurrence
        
        # Check that learner has NO frequency attributes
        self.assertFalse(hasattr(self.learner, 'cooccurrence'))
        self.assertFalse(hasattr(self.learner, 'word_freq'))
        self.assertFalse(hasattr(self.learner, 'total_pairs'))
        
        # Check that it DOES have QFI-related attributes
        self.assertTrue(hasattr(self.learner, 'compute_qfi'))
        self.assertTrue(hasattr(self.learner, 'compute_ricci_curvature'))


class TestNoBasinModification(unittest.TestCase):
    """Test that basins are NOT modified (frozen invariants)."""
    
    def setUp(self):
        """Create mock coordizer and learner."""
        self.coordizer = MockCoordizer()
        self.learner = GeometricWordRelationships(coordizer=self.coordizer)
    
    def test_no_basin_adjustment_method(self):
        """Test that GeometricWordRelationships has NO basin adjustment."""
        # Should NOT have adjust_basin_coordinates method
        self.assertFalse(hasattr(self.learner, 'adjust_basin_coordinates'))
    
    def test_basins_unchanged_after_operations(self):
        """Test that basins remain unchanged after operations."""
        # Get original basin
        original_basin = self.coordizer.get_basin_coords('quantum').copy()
        
        # Perform operations
        self.learner.get_related_words('quantum', top_k=3)
        self.learner.compute_geometric_properties('quantum')
        
        # Basin should be unchanged
        current_basin = self.coordizer.get_basin_coords('quantum')
        np.testing.assert_array_almost_equal(original_basin, current_basin)


class TestNoPMI(unittest.TestCase):
    """Test that PMI is NOT used."""
    
    def setUp(self):
        """Create mock coordizer and learner."""
        self.coordizer = MockCoordizer()
        self.learner = GeometricWordRelationships(coordizer=self.coordizer)
    
    def test_no_pmi_method(self):
        """Test that GeometricWordRelationships has NO PMI computation."""
        # Should NOT have compute_affinity_matrix or PMI methods
        self.assertFalse(hasattr(self.learner, 'compute_affinity_matrix'))
        self.assertFalse(hasattr(self.learner, 'compute_pmi'))
    
    def test_no_cooccurrence_counting(self):
        """Test that there is NO co-occurrence counting."""
        # Should NOT have tokenize_text, learn_from_text, etc.
        self.assertFalse(hasattr(self.learner, 'tokenize_text'))
        self.assertFalse(hasattr(self.learner, 'learn_from_text'))
        self.assertFalse(hasattr(self.learner, 'learn_from_file'))


class TestGeometricFiltering(unittest.TestCase):
    """Test geometric-based filtering (not frequency-based stopwords)."""
    
    def setUp(self):
        """Create mock coordizer and learner."""
        self.coordizer = MockCoordizer()
        self.learner = GeometricWordRelationships(coordizer=self.coordizer)
    
    def test_context_critical_not_filtered(self):
        """Test that context-critical words are NOT filtered."""
        # Get properties for 'not' (should be context-critical)
        props = self.learner.compute_geometric_properties('not')
        
        # Should not be filtered
        should_filter = self.learner.should_filter_word('not')
        self.assertFalse(should_filter, "'not' should never be filtered (context-critical)")
    
    def test_filtering_uses_qfi_not_frequency(self):
        """Test that filtering uses QFI, not frequency."""
        # This is a design test - geometric filtering should use
        # QFI and curvature, not word frequency
        
        # Method signature should not include frequency parameters
        import inspect
        sig = inspect.signature(self.learner.should_filter_word)
        params = list(sig.parameters.keys())
        
        # Should NOT have frequency or count parameters
        self.assertNotIn('frequency', params)
        self.assertNotIn('count', params)


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test cases
    suite.addTests(loader.loadTestsFromTestCase(TestGeometricProperties))
    suite.addTests(loader.loadTestsFromTestCase(TestFisherRaoDistances))
    suite.addTests(loader.loadTestsFromTestCase(TestQFIWeightedAttention))
    suite.addTests(loader.loadTestsFromTestCase(TestNoBasinModification))
    suite.addTests(loader.loadTestsFromTestCase(TestNoPMI))
    suite.addTests(loader.loadTestsFromTestCase(TestGeometricFiltering))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    print("=" * 70)
    print("Testing QIG-Pure Geometric Word Relationships")
    print("=" * 70)
    print()
    
    success = run_tests()
    
    print()
    print("=" * 70)
    if success:
        print("✅ All tests PASSED")
        print()
        print("Validated:")
        print("  ✅ Fisher-Rao distances (not PMI)")
        print("  ✅ QFI-weighted attention (not frequency)")
        print("  ✅ Ricci curvature for context-dependency")
        print("  ✅ No basin modification (frozen invariants)")
        print("  ✅ Geometric filtering (not stopwords)")
    else:
        print("❌ Some tests FAILED")
    print("=" * 70)
    
    sys.exit(0 if success else 1)
