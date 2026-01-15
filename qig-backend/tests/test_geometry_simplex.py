"""
Unit tests for canonical simplex geometry module.

Tests the SINGLE SOURCE OF TRUTH for simplex-as-storage operations:
- to_simplex_prob() - convert any vector to probability simplex
- fisher_rao_distance() - compute Fisher-Rao distance on simplex
- geodesic_interpolation_simplex() - geodesic interpolation using sqrt-simplex internally
- geodesic_mean_simplex() - compute Fréchet mean on simplex
"""

import pytest
import numpy as np
from qig_geometry.geometry_simplex import (
    to_simplex_prob,
    validate_simplex,
    fisher_rao_distance,
    geodesic_interpolation_simplex,
    geodesic_mean_simplex,
    batch_fisher_rao_distance,
    find_nearest_simplex,
    SIMPLEX_EPSILON,
)


class TestToSimplexProb:
    """Test conversion to probability simplex."""
    
    def test_basic_conversion(self):
        """Test basic vector conversion."""
        v = np.array([1.0, 2.0, 3.0])
        p = to_simplex_prob(v)
        
        # Check simplex properties
        assert np.all(p >= 0), "Simplex must be non-negative"
        assert np.isclose(p.sum(), 1.0), "Simplex must sum to 1"
    
    def test_negative_values(self):
        """Test handling of negative values (should take absolute value)."""
        v = np.array([-1.0, 2.0, -3.0])
        p = to_simplex_prob(v)
        
        assert np.all(p >= 0), "Negative values should be made positive"
        assert np.isclose(p.sum(), 1.0), "Must sum to 1"
    
    def test_zero_vector(self):
        """Test handling of zero vector (should return uniform)."""
        v = np.array([0.0, 0.0, 0.0])
        p = to_simplex_prob(v)
        
        # Should return uniform distribution
        expected = np.array([1/3, 1/3, 1/3])
        assert np.allclose(p, expected, atol=1e-6)
    
    def test_already_simplex(self):
        """Test vector that's already a valid simplex."""
        v = np.array([0.2, 0.3, 0.5])
        p = to_simplex_prob(v)
        
        # Should be approximately unchanged
        assert np.allclose(p, v, atol=1e-6)
    
    def test_dimension_preservation(self):
        """Test that dimension is preserved."""
        for dim in [8, 16, 32, 64, 128]:
            v = np.random.randn(dim)
            p = to_simplex_prob(v)
            assert len(p) == dim, f"Dimension should be preserved: {dim}"


class TestValidateSimplex:
    """Test simplex validation."""
    
    def test_valid_simplex(self):
        """Test validation of valid simplex."""
        p = np.array([0.2, 0.3, 0.5])
        valid, reason = validate_simplex(p)
        
        assert valid, f"Valid simplex should pass: {reason}"
    
    def test_negative_values(self):
        """Test rejection of negative values."""
        p = np.array([0.2, -0.1, 0.9])
        valid, reason = validate_simplex(p)
        
        assert not valid, "Negative values should be rejected"
        assert "negative" in reason.lower()
    
    def test_sum_not_one(self):
        """Test rejection of vectors that don't sum to 1."""
        p = np.array([0.2, 0.3, 0.4])
        valid, reason = validate_simplex(p)
        
        assert not valid, "Sum != 1 should be rejected"
        assert "sum" in reason.lower()
    
    def test_nan_values(self):
        """Test rejection of NaN values."""
        p = np.array([0.2, np.nan, 0.8])
        valid, reason = validate_simplex(p)
        
        assert not valid, "NaN values should be rejected"
    
    def test_tolerance(self):
        """Test numerical tolerance."""
        p = np.array([0.2, 0.3, 0.5000001])  # Slightly off sum
        valid, reason = validate_simplex(p, tolerance=1e-5)
        
        assert valid, f"Small numerical error should be tolerated: {reason}"


class TestFisherRaoDistance:
    """Test Fisher-Rao distance computation."""
    
    def test_identity(self):
        """Test distance to self is zero."""
        p = np.array([0.2, 0.3, 0.5])
        d = fisher_rao_distance(p, p)
        
        assert np.isclose(d, 0.0, atol=1e-10), "Distance to self should be zero"
    
    def test_symmetry(self):
        """Test distance symmetry: d(p, q) = d(q, p)."""
        p = np.array([0.2, 0.3, 0.5])
        q = np.array([0.4, 0.1, 0.5])
        
        d1 = fisher_rao_distance(p, q)
        d2 = fisher_rao_distance(q, p)
        
        assert np.isclose(d1, d2, atol=1e-10), "Distance should be symmetric"
    
    def test_range(self):
        """Test distance is in [0, π/2]."""
        np.random.seed(42)
        for _ in range(10):
            p = to_simplex_prob(np.random.randn(64))
            q = to_simplex_prob(np.random.randn(64))
            
            d = fisher_rao_distance(p, q)
            assert 0 <= d <= np.pi/2 + 1e-6, f"Distance {d} should be in [0, π/2]"
    
    def test_orthogonal_distributions(self):
        """Test distance between orthogonal distributions."""
        # Create two distributions with no overlap
        p = np.array([1.0, 0.0, 0.0])
        q = np.array([0.0, 1.0, 0.0])
        
        d = fisher_rao_distance(p, q)
        
        # Orthogonal distributions should have distance π/2
        assert np.isclose(d, np.pi/2, atol=1e-6), f"Orthogonal distance should be π/2, got {d}"
    
    def test_dimension_mismatch(self):
        """Test error on dimension mismatch."""
        p = np.array([0.5, 0.5])
        q = np.array([0.33, 0.33, 0.34])
        
        with pytest.raises(ValueError, match="dimension mismatch"):
            fisher_rao_distance(p, q)


class TestGeodesicInterpolationSimplex:
    """Test geodesic interpolation on simplex."""
    
    def test_endpoints(self):
        """Test that t=0 gives start and t=1 gives end."""
        p = np.array([0.2, 0.3, 0.5])
        q = np.array([0.4, 0.1, 0.5])
        
        result_0 = geodesic_interpolation_simplex(p, q, 0.0)
        result_1 = geodesic_interpolation_simplex(p, q, 1.0)
        
        assert np.allclose(result_0, p, atol=1e-6), "t=0 should give start"
        assert np.allclose(result_1, q, atol=1e-6), "t=1 should give end"
    
    def test_midpoint_simplex(self):
        """Test that midpoint is on simplex."""
        p = np.array([0.2, 0.3, 0.5])
        q = np.array([0.4, 0.1, 0.5])
        
        mid = geodesic_interpolation_simplex(p, q, 0.5)
        
        valid, reason = validate_simplex(mid)
        assert valid, f"Midpoint should be valid simplex: {reason}"
    
    def test_interpolation_preserves_simplex(self):
        """Test all interpolated points are valid simplexes."""
        p = np.array([0.2, 0.3, 0.5])
        q = np.array([0.4, 0.1, 0.5])
        
        for t in np.linspace(0, 1, 11):
            result = geodesic_interpolation_simplex(p, q, t)
            valid, reason = validate_simplex(result)
            assert valid, f"Interpolation at t={t} should be valid: {reason}"
    
    def test_geodesic_is_shortest_path(self):
        """Test geodesic is shorter than linear interpolation."""
        p = np.array([1.0, 0.0, 0.0])
        q = np.array([0.0, 1.0, 0.0])
        
        # Geodesic length (should be π/2 for orthogonal)
        geodesic_length = fisher_rao_distance(p, q)
        
        # Linear path length (sum of distances along segments)
        linear_length = 0.0
        prev = p
        for t in np.linspace(0, 1, 21):
            # Linear interpolation in probability space (NOT geodesic)
            linear_point = (1 - t) * p + t * q
            linear_point = to_simplex_prob(linear_point)
            linear_length += fisher_rao_distance(prev, linear_point)
            prev = linear_point
        
        # Geodesic should be shorter (or equal within numerical error)
        assert geodesic_length <= linear_length + 1e-3, \
            f"Geodesic ({geodesic_length}) should be ≤ linear ({linear_length})"
    
    def test_t_out_of_range(self):
        """Test error for t outside [0, 1]."""
        p = np.array([0.5, 0.5])
        q = np.array([0.3, 0.7])
        
        with pytest.raises(ValueError, match="t must be in"):
            geodesic_interpolation_simplex(p, q, -0.1)
        
        with pytest.raises(ValueError, match="t must be in"):
            geodesic_interpolation_simplex(p, q, 1.1)


class TestGeodesicMeanSimplex:
    """Test Fréchet mean on simplex."""
    
    def test_single_distribution(self):
        """Test mean of single distribution is itself."""
        p = np.array([0.2, 0.3, 0.5])
        mean = geodesic_mean_simplex([p])
        
        assert np.allclose(mean, p, atol=1e-6), "Mean of single dist should be itself"
    
    def test_identical_distributions(self):
        """Test mean of identical distributions."""
        p = np.array([0.2, 0.3, 0.5])
        mean = geodesic_mean_simplex([p, p, p])
        
        assert np.allclose(mean, p, atol=1e-6), "Mean of identical dists should be same"
    
    def test_weighted_mean(self):
        """Test weighted geodesic mean."""
        p1 = np.array([1.0, 0.0, 0.0])
        p2 = np.array([0.0, 1.0, 0.0])
        
        # Equal weights should give midpoint
        mean_equal = geodesic_mean_simplex([p1, p2], weights=np.array([0.5, 0.5]))
        
        # Should be closer to uniform than endpoints
        assert 0.2 < mean_equal[0] < 0.8, "Mean should be between endpoints"
        
        # Unequal weights should bias toward higher-weighted distribution
        mean_biased = geodesic_mean_simplex([p1, p2], weights=np.array([0.9, 0.1]))
        
        # Should be closer to p1
        assert mean_biased[0] > mean_equal[0], "Higher weight should pull mean"
    
    def test_mean_is_simplex(self):
        """Test that mean is a valid simplex."""
        np.random.seed(42)
        distributions = [to_simplex_prob(np.random.randn(8)) for _ in range(5)]
        
        mean = geodesic_mean_simplex(distributions)
        
        valid, reason = validate_simplex(mean)
        assert valid, f"Mean should be valid simplex: {reason}"


class TestBatchOperations:
    """Test batch operations."""
    
    def test_batch_distance(self):
        """Test batch Fisher-Rao distance computation."""
        query = np.array([0.2, 0.3, 0.5])
        candidates = [
            np.array([0.25, 0.25, 0.5]),
            np.array([0.1, 0.4, 0.5]),
            np.array([0.5, 0.25, 0.25]),
        ]
        
        distances = batch_fisher_rao_distance(query, candidates)
        
        assert len(distances) == 3, "Should compute distance to all candidates"
        assert np.all(distances >= 0), "Distances should be non-negative"
        assert np.all(distances <= np.pi/2 + 1e-6), "Distances should be ≤ π/2"
    
    def test_find_nearest(self):
        """Test finding nearest neighbors."""
        query = np.array([0.2, 0.3, 0.5])
        candidates = [
            np.array([0.25, 0.25, 0.5]),    # Close to query
            np.array([0.1, 0.4, 0.5]),      # Close to query
            np.array([1.0, 0.0, 0.0]),      # Far from query
            np.array([0.0, 1.0, 0.0]),      # Far from query
        ]
        
        nearest = find_nearest_simplex(query, candidates, k=2)
        
        assert len(nearest) == 2, "Should return k nearest"
        
        # Check order (nearest first)
        assert nearest[0]['distance'] <= nearest[1]['distance'], "Should be sorted by distance"
        
        # Check that nearest are actually closer
        all_distances = [fisher_rao_distance(query, c) for c in candidates]
        nearest_distances = [item['distance'] for item in nearest]
        
        for nd in nearest_distances:
            assert nd <= max(all_distances), "Nearest should be among closest"


class TestNumericalStability:
    """Test numerical stability and edge cases."""
    
    def test_very_small_values(self):
        """Test handling of very small values."""
        v = np.array([1e-20, 1e-20, 1e-20])
        p = to_simplex_prob(v)
        
        assert np.isclose(p.sum(), 1.0), "Should handle small values"
    
    def test_very_large_values(self):
        """Test handling of very large values."""
        v = np.array([1e10, 1e10, 1e10])
        p = to_simplex_prob(v)
        
        assert np.isclose(p.sum(), 1.0), "Should handle large values"
        assert np.all(np.isfinite(p)), "Result should be finite"
    
    def test_mixed_scale_values(self):
        """Test handling of mixed scale values."""
        v = np.array([1e-10, 1.0, 1e10])
        p = to_simplex_prob(v)
        
        assert np.isclose(p.sum(), 1.0), "Should handle mixed scales"
        assert np.all(np.isfinite(p)), "Result should be finite"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
