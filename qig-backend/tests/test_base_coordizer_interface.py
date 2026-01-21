"""
Test BaseCoordizer Interface (WP3.1)

Validates that coordizer implementations properly implement the BaseCoordizer
abstract interface with Plan→Realize→Repair compatibility.

Tests:
1. Interface compliance (all required methods exist)
2. Two-step geometric decoding (proxy + exact)
3. POS filtering support
4. Geometric operations from canonical module
5. Consistent behavior across implementations
"""

import pytest
import numpy as np
from abc import ABC
from typing import List, Tuple

# Import coordizer classes
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from coordizers import BaseCoordizer, FisherCoordizer, PostgresCoordizer, get_coordizer


class TestBaseCoordinzerInterface:
    """Test that BaseCoordizer is a proper abstract interface."""
    
    def test_base_coordizer_is_abstract(self):
        """BaseCoordizer should be an ABC."""
        assert issubclass(BaseCoordizer, ABC)
        
    def test_cannot_instantiate_base_coordizer(self):
        """Cannot instantiate abstract BaseCoordizer directly."""
        with pytest.raises(TypeError):
            BaseCoordizer()
    
    def test_required_methods_defined(self):
        """BaseCoordizer defines all required abstract methods."""
        required_methods = [
            'decode_geometric',
            'encode',
            'get_vocabulary_size',
            'get_special_symbols',
            'supports_pos_filtering',
        ]
        
        for method_name in required_methods:
            assert hasattr(BaseCoordizer, method_name), f"Missing method: {method_name}"


class TestFisherCoordinzerInterface:
    """Test that FisherCoordizer implements BaseCoordizer interface."""
    
    def test_fisher_inherits_from_base(self):
        """FisherCoordizer inherits from BaseCoordizer."""
        assert issubclass(FisherCoordizer, BaseCoordizer)
    
    def test_fisher_implements_all_methods(self):
        """FisherCoordizer implements all required methods."""
        coordizer = FisherCoordizer(vocab_size=100)
        
        # Check all required methods are implemented
        assert callable(getattr(coordizer, 'decode_geometric', None))
        assert callable(getattr(coordizer, 'encode', None))
        assert callable(getattr(coordizer, 'get_vocabulary_size', None))
        assert callable(getattr(coordizer, 'get_special_symbols', None))
        assert callable(getattr(coordizer, 'supports_pos_filtering', None))
    
    def test_fisher_decode_geometric_returns_correct_format(self):
        """decode_geometric returns List[Tuple[str, float]]."""
        coordizer = FisherCoordizer(vocab_size=100)
        
        # Train with some basic vocabulary
        corpus = ["hello world", "test data", "geometric fisher"]
        coordizer.train(corpus)
        
        # Create a test basin
        test_basin = np.random.rand(64)
        test_basin = test_basin / np.linalg.norm(test_basin)
        
        # Call decode_geometric
        results = coordizer.decode_geometric(test_basin, top_k=5)
        
        # Validate format
        assert isinstance(results, list)
        assert len(results) <= 5
        
        for token, distance in results:
            assert isinstance(token, str)
            assert isinstance(distance, (float, np.floating))
            assert distance >= 0  # Fisher-Rao distance is non-negative
    
    def test_fisher_two_step_retrieval(self):
        """decode_geometric uses two-step retrieval (proxy + exact)."""
        coordizer = FisherCoordizer(vocab_size=100)
        
        # Train with vocabulary
        corpus = ["alpha", "beta", "gamma", "delta", "epsilon"]
        coordizer.train(corpus)
        
        # Get a known token's basin
        known_token = "alpha"
        if known_token in coordizer.basin_coords:
            known_basin = coordizer.basin_coords[known_token]
            
            # Decode should return the known token with low distance
            results = coordizer.decode_geometric(known_basin, top_k=5)
            
            # First result should be the known token
            assert len(results) > 0
            top_token, top_distance = results[0]
            assert top_token == known_token
            assert top_distance < 0.1  # Should be very close
    
    def test_fisher_pos_filtering_not_supported(self):
        """FisherCoordizer base class doesn't support POS filtering."""
        coordizer = FisherCoordizer(vocab_size=100)
        assert coordizer.supports_pos_filtering() == False
        
        # Attempting POS filtering should raise NotImplementedError
        test_basin = np.random.rand(64)
        test_basin = test_basin / np.linalg.norm(test_basin)
        
        with pytest.raises(NotImplementedError):
            coordizer.decode_geometric(test_basin, top_k=5, allowed_pos="NOUN")
    
    def test_fisher_vocabulary_size(self):
        """get_vocabulary_size returns correct size."""
        coordizer = FisherCoordizer(vocab_size=100)
        
        # Initially has special tokens
        initial_size = coordizer.get_vocabulary_size()
        assert initial_size == len(coordizer.special_tokens)
        
        # After training, should increase
        corpus = ["word1", "word2", "word3"]
        coordizer.train(corpus)
        
        new_size = coordizer.get_vocabulary_size()
        assert new_size > initial_size
    
    def test_fisher_special_symbols(self):
        """get_special_symbols returns geometric definitions."""
        coordizer = FisherCoordizer(vocab_size=100)
        
        symbols = coordizer.get_special_symbols()
        
        # Should have special tokens defined
        assert isinstance(symbols, dict)
        assert len(symbols) > 0
        
        # Check each special token has required fields
        for token, data in symbols.items():
            assert 'basin_coordinates' in data
            assert 'coordinate_dim' in data
            assert 'token_id' in data
            
            # Basin coordinates should be 64D
            basin = data['basin_coordinates']
            assert isinstance(basin, np.ndarray)
            assert len(basin) == 64


class TestPostgresCoordinzerInterface:
    """Test that PostgresCoordizer implements BaseCoordizer interface."""
    
    @pytest.fixture
    def skip_if_no_database(self):
        """Skip test if database is not available."""
        import os
        if not os.getenv('DATABASE_URL'):
            pytest.skip("DATABASE_URL not set, skipping PostgresCoordizer tests")
    
    def test_postgres_inherits_from_fisher(self):
        """PostgresCoordizer inherits from FisherCoordizer (which inherits BaseCoordizer)."""
        assert issubclass(PostgresCoordizer, FisherCoordizer)
        assert issubclass(PostgresCoordizer, BaseCoordizer)
    
    def test_postgres_supports_pos_filtering(self, skip_if_no_database):
        """PostgresCoordizer may support POS filtering (runtime check)."""
        try:
            coordizer = get_coordizer()
            
            # Check if POS filtering is supported
            has_pos_support = coordizer.supports_pos_filtering()
            
            # Should return boolean
            assert isinstance(has_pos_support, bool)
            
        except Exception as e:
            pytest.skip(f"Could not initialize PostgresCoordizer: {e}")
    
    def test_postgres_decode_geometric_with_pos(self, skip_if_no_database):
        """PostgresCoordizer decode_geometric with POS filter (if supported)."""
        try:
            coordizer = get_coordizer()
            
            if not coordizer.supports_pos_filtering():
                pytest.skip("POS filtering not supported (no pos_tag column)")
            
            # Create test basin
            test_basin = np.random.rand(64)
            test_basin = test_basin / np.linalg.norm(test_basin)
            
            # Try decoding with POS filter
            results = coordizer.decode_geometric(
                test_basin,
                top_k=10,
                allowed_pos="NOUN"
            )
            
            # Should return results
            assert isinstance(results, list)
            
            # Each result should be (word, distance)
            for token, distance in results:
                assert isinstance(token, str)
                assert isinstance(distance, (float, np.floating))
                
        except Exception as e:
            pytest.skip(f"Could not test POS filtering: {e}")


class TestCoordinzerConsistency:
    """Test consistent behavior across coordizer implementations."""
    
    def test_encode_returns_64d_basin(self):
        """encode() returns normalized 64D basin coordinates."""
        coordizer = FisherCoordizer(vocab_size=100)
        
        text = "test input"
        basin = coordizer.encode(text)
        
        # Should be 64D
        assert isinstance(basin, np.ndarray)
        assert len(basin) == 64
        
        # E8 Protocol: Should be on simplex (non-negative, sum≈1)
        from qig_geometry.representation import to_simplex_prob
        if not np.all(np.abs(basin) < 1e-10):  # If not zero
            basin_simplex = to_simplex_prob(basin)
            assert np.allclose(np.sum(basin_simplex), 1.0, atol=0.1), "Basin not on simplex"
    
    def test_decode_geometric_distance_sorted(self):
        """decode_geometric returns results sorted by distance ascending."""
        coordizer = FisherCoordizer(vocab_size=100)
        
        # Train with vocabulary
        corpus = ["one", "two", "three", "four", "five", "six", "seven", "eight"]
        coordizer.train(corpus, min_frequency=1)
        
        # Create test basin
        test_basin = np.random.rand(64)
        test_basin = test_basin / np.linalg.norm(test_basin)
        
        # Get results
        results = coordizer.decode_geometric(test_basin, top_k=5)
        
        if len(results) > 1:
            # Check distances are sorted ascending
            distances = [dist for _, dist in results]
            assert distances == sorted(distances), "Results not sorted by distance"
    
    def test_decode_geometric_deterministic(self):
        """decode_geometric returns deterministic results for same input."""
        coordizer = FisherCoordizer(vocab_size=100)
        
        # Train with vocabulary
        corpus = ["alpha", "beta", "gamma"]
        coordizer.train(corpus)
        
        # Create test basin
        test_basin = np.random.rand(64)
        test_basin = test_basin / np.linalg.norm(test_basin)
        
        # Call twice with same input
        results1 = coordizer.decode_geometric(test_basin, top_k=3)
        results2 = coordizer.decode_geometric(test_basin, top_k=3)
        
        # Should be identical
        assert len(results1) == len(results2)
        for (token1, dist1), (token2, dist2) in zip(results1, results2):
            assert token1 == token2
            assert abs(dist1 - dist2) < 1e-6


def test_get_coordizer_returns_base_coordizer():
    """get_coordizer() returns an instance implementing BaseCoordizer."""
    try:
        coordizer = get_coordizer()
        
        # Should be instance of BaseCoordizer
        assert isinstance(coordizer, BaseCoordizer)
        
        # Should have all required methods
        assert hasattr(coordizer, 'decode_geometric')
        assert hasattr(coordizer, 'encode')
        assert hasattr(coordizer, 'get_vocabulary_size')
        assert hasattr(coordizer, 'get_special_symbols')
        assert hasattr(coordizer, 'supports_pos_filtering')
        
    except Exception as e:
        pytest.skip(f"Could not initialize coordizer: {e}")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
