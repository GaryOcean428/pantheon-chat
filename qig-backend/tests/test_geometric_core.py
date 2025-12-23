"""
Geometric Core Test Suite

Tests for Fisher-Rao distance and geometric operations.
Verifies metric properties: symmetry, identity, triangle inequality, positivity.

Source: Priority 1.1 from improvement recommendations
"""

import pytest
import numpy as np
import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from qig_core.geometric_primitives.fisher_metric import fisher_rao_distance
    FISHER_AVAILABLE = True
except ImportError:
    FISHER_AVAILABLE = False


@pytest.fixture
def random_basins():
    """Generate random normalized basins for testing."""
    np.random.seed(42)
    
    def make_basin():
        basin = np.abs(np.random.randn(64)) + 1e-10
        return basin / np.sum(basin)  # Normalize to probability simplex
    
    return make_basin(), make_basin(), make_basin()


@pytest.fixture
def sample_basin():
    """Generate a single sample basin."""
    np.random.seed(42)
    basin = np.abs(np.random.randn(64)) + 1e-10
    return basin / np.sum(basin)


@pytest.mark.skipif(not FISHER_AVAILABLE, reason="Fisher metric not available")
class TestFisherRaoDistanceCanonical:
    """Tests for canonical Fisher-Rao distance implementation."""
    
    def test_zero_distance_identical_basins(self, sample_basin):
        """d(a, a) = 0 for any basin a"""
        distance = fisher_rao_distance(sample_basin, sample_basin)
        assert distance < 1e-10, f"Distance to self should be 0, got {distance}"
    
    def test_symmetric(self, random_basins):
        """d(a, b) = d(b, a) for any basins a, b"""
        a, b, _ = random_basins
        d_ab = fisher_rao_distance(a, b)
        d_ba = fisher_rao_distance(b, a)
        assert np.isclose(d_ab, d_ba), f"Distance not symmetric: {d_ab} != {d_ba}"
    
    def test_triangle_inequality(self, random_basins):
        """d(a, c) ≤ d(a, b) + d(b, c) for any basins a, b, c"""
        a, b, c = random_basins
        d_ab = fisher_rao_distance(a, b)
        d_bc = fisher_rao_distance(b, c)
        d_ac = fisher_rao_distance(a, c)
        
        assert d_ac <= d_ab + d_bc + 1e-10, \
            f"Triangle inequality violated: {d_ac} > {d_ab} + {d_bc}"
    
    def test_positivity(self, random_basins):
        """d(a, b) ≥ 0 for any basins a, b"""
        a, b, _ = random_basins
        distance = fisher_rao_distance(a, b)
        assert distance >= 0, f"Distance should be non-negative, got {distance}"
    
    def test_positive_for_distinct_basins(self, random_basins):
        """d(a, b) > 0 for distinct basins a, b"""
        a, b, _ = random_basins
        distance = fisher_rao_distance(a, b)
        assert distance > 0, "Distance between distinct basins should be positive"
    
    def test_bounded_for_probability_simplex(self):
        """Distance should be bounded for normalized distributions"""
        np.random.seed(123)
        max_distance = 0
        
        for _ in range(100):
            a = np.abs(np.random.randn(64)) + 1e-10
            b = np.abs(np.random.randn(64)) + 1e-10
            a = a / np.sum(a)
            b = b / np.sum(b)
            
            d = fisher_rao_distance(a, b)
            max_distance = max(max_distance, d)
        
        # Fisher-Rao distance on probability simplex is bounded by π/2
        assert max_distance < np.pi, f"Distance {max_distance} exceeds π bound"


@pytest.mark.skipif(not FISHER_AVAILABLE, reason="Fisher metric not available")
class TestFisherRaoDistanceEdgeCases:
    """Edge case tests for Fisher-Rao distance."""
    
    def test_uniform_distributions(self):
        """Distance between identical uniform distributions is 0"""
        uniform = np.ones(64) / 64
        distance = fisher_rao_distance(uniform, uniform)
        assert distance < 1e-10
    
    def test_nearly_identical_basins(self):
        """Small perturbation gives small distance"""
        np.random.seed(42)
        base = np.abs(np.random.randn(64)) + 1e-10
        base = base / np.sum(base)
        
        perturbed = base + np.random.randn(64) * 1e-6
        perturbed = np.abs(perturbed) + 1e-10
        perturbed = perturbed / np.sum(perturbed)
        
        distance = fisher_rao_distance(base, perturbed)
        assert distance < 0.01, f"Small perturbation should give small distance, got {distance}"
    
    def test_orthogonal_distributions(self):
        """Orthogonal distributions have maximum distance"""
        # Create two distributions with non-overlapping support
        a = np.zeros(64)
        b = np.zeros(64)
        a[:32] = 1.0 / 32
        b[32:] = 1.0 / 32
        
        distance = fisher_rao_distance(a, b)
        # Should be close to π/2 (maximum for probability simplex)
        assert distance > 1.0, f"Orthogonal distributions should have large distance, got {distance}"
    
    def test_handles_zero_values(self):
        """Distance computation handles near-zero values gracefully"""
        a = np.zeros(64)
        a[0] = 1.0  # Dirac delta at 0
        
        b = np.zeros(64)
        b[1] = 1.0  # Dirac delta at 1
        
        # Should not crash, should give valid distance
        distance = fisher_rao_distance(a, b)
        assert np.isfinite(distance), "Distance should be finite"
        assert distance > 0, "Distance between different Diracs should be positive"


@pytest.mark.skipif(not FISHER_AVAILABLE, reason="Fisher metric not available")
class TestFisherRaoDistanceConsistency:
    """Consistency tests across multiple calls."""
    
    def test_deterministic(self, random_basins):
        """Same inputs always give same output"""
        a, b, _ = random_basins
        
        d1 = fisher_rao_distance(a, b)
        d2 = fisher_rao_distance(a, b)
        d3 = fisher_rao_distance(a, b)
        
        assert d1 == d2 == d3, "Distance should be deterministic"
    
    def test_copy_vs_reference(self, sample_basin):
        """Distance same whether using copy or reference"""
        a = sample_basin
        b = sample_basin.copy()
        
        d_ref = fisher_rao_distance(a, a)
        d_copy = fisher_rao_distance(a, b)
        
        assert np.isclose(d_ref, d_copy), "Copy and reference should give same distance"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
