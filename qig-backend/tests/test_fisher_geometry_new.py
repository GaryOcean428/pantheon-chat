"""
Tests for Fisher-Rao Geometry Operations

Validates proper Riemannian geometry on information manifolds.
"""

import pytest
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from qigkernels.fisher_geometry import (
    fisher_rao_distance,
    bhattacharyya_coefficient,
    compute_fisher_metric,
    natural_gradient,
    hellinger_distance,
    kl_divergence,
    js_divergence,
    geodesic_distance_euclidean_fallback,
)


class TestFisherRaoDistance:
    """Test Fisher-Rao distance computation."""
    
    def test_identical_distributions(self):
        """Test distance between identical distributions."""
        p = np.array([0.5, 0.3, 0.2])
        distance = fisher_rao_distance(p, p)
        
        # Distance to self should be zero
        assert np.isclose(distance, 0.0, atol=1e-6)
    
    def test_orthogonal_distributions(self):
        """Test distance between orthogonal distributions."""
        p = np.array([1.0, 0.0, 0.0])
        q = np.array([0.0, 1.0, 0.0])
        
        distance = fisher_rao_distance(p, q)
        
        # Orthogonal distributions should have maximum distance
        assert distance > 1.0
        assert distance <= np.pi / 2  # Max Fisher-Rao distance
    
    def test_similar_distributions(self):
        """Test distance between similar distributions."""
        p = np.array([0.5, 0.3, 0.2])
        q = np.array([0.45, 0.35, 0.2])
        
        distance = fisher_rao_distance(p, q)
        
        # Similar distributions should have small distance
        assert distance > 0
        assert distance < 0.5
    
    def test_torch_input(self):
        """Test with PyTorch tensors."""
        if not HAS_TORCH:
            pytest.skip("PyTorch not available")
        
        p = torch.tensor([0.5, 0.3, 0.2])
        q = torch.tensor([0.45, 0.35, 0.2])
        
        distance = fisher_rao_distance(p, q)
        
        # Should work with torch input
        assert isinstance(distance, float)
        assert distance > 0
    
    def test_custom_metric_tensor(self):
        """Test with custom metric tensor."""
        p = np.array([0.5, 0.3, 0.2])
        q = np.array([0.45, 0.35, 0.2])
        metric = np.eye(3)
        
        distance = fisher_rao_distance(p, q, metric_tensor=metric)
        
        assert distance > 0
        assert np.isfinite(distance)
    
    def test_unnormalized_input(self):
        """Test that unnormalized inputs are handled."""
        p = np.array([1.0, 2.0, 3.0])  # Not normalized
        q = np.array([2.0, 3.0, 1.0])  # Not normalized
        
        # Should normalize internally and compute distance
        distance = fisher_rao_distance(p, q)
        
        assert np.isfinite(distance)
        assert distance > 0


class TestBhattacharyyaCoefficient:
    """Test Bhattacharyya coefficient."""
    
    def test_identical_distributions(self):
        """Test BC between identical distributions."""
        p = np.array([0.5, 0.3, 0.2])
        bc = bhattacharyya_coefficient(p, p)
        
        # BC should be 1.0 for identical distributions
        assert np.isclose(bc, 1.0, atol=1e-6)
    
    def test_orthogonal_distributions(self):
        """Test BC between orthogonal distributions."""
        p = np.array([1.0, 0.0, 0.0])
        q = np.array([0.0, 1.0, 0.0])
        
        bc = bhattacharyya_coefficient(p, q)
        
        # BC should be 0.0 for orthogonal distributions
        assert np.isclose(bc, 0.0, atol=1e-6)
    
    def test_range(self):
        """Test BC is in valid range [0, 1]."""
        p = np.random.rand(10)
        q = np.random.rand(10)
        
        bc = bhattacharyya_coefficient(p, q)
        
        assert 0.0 <= bc <= 1.0


class TestFisherMetric:
    """Test Fisher information metric computation."""
    
    def test_dirichlet_metric(self):
        """Test Dirichlet-Multinomial metric."""
        basin = np.array([0.5, 0.3, 0.2])
        
        G = compute_fisher_metric(basin, method='dirichlet')
        
        # Should return diagonal matrix
        assert G.shape == (3, 3)
        assert np.allclose(G, np.diag(np.diag(G)))  # Diagonal
        
        # Diagonal elements should be 1/p_i
        expected_diag = 1.0 / basin
        assert np.allclose(np.diag(G), expected_diag, atol=0.1)
    
    def test_metric_positive_definite(self):
        """Test that metric is positive definite."""
        basin = np.random.rand(10)
        basin = basin / basin.sum()
        
        G = compute_fisher_metric(basin, method='dirichlet')
        
        # All eigenvalues should be positive
        eigenvalues = np.linalg.eigvalsh(G)
        assert np.all(eigenvalues > 0)


class TestNaturalGradient:
    """Test natural gradient computation."""
    
    def test_identity_metric(self):
        """Test natural gradient with identity metric."""
        gradient = np.array([1.0, 2.0, 3.0])
        fisher_metric = np.eye(3)
        
        nat_grad = natural_gradient(gradient, fisher_metric)
        
        # With identity metric, natural gradient = gradient
        assert np.allclose(nat_grad, gradient)
    
    def test_diagonal_metric(self):
        """Test natural gradient with diagonal metric."""
        gradient = np.array([1.0, 2.0, 3.0])
        fisher_metric = np.diag([2.0, 2.0, 2.0])
        
        nat_grad = natural_gradient(gradient, fisher_metric)
        
        # Natural gradient should be F^{-1} * gradient
        expected = gradient / 2.0
        assert np.allclose(nat_grad, expected, atol=0.1)
    
    def test_torch_input(self):
        """Test with PyTorch tensors."""
        if not HAS_TORCH:
            pytest.skip("PyTorch not available")
        
        gradient = torch.tensor([1.0, 2.0, 3.0])
        fisher_metric = torch.eye(3)
        
        nat_grad = natural_gradient(gradient, fisher_metric)
        
        # Should return torch tensor
        assert isinstance(nat_grad, torch.Tensor)
        assert torch.allclose(nat_grad, gradient)
    
    def test_damping(self):
        """Test that damping prevents singular matrices."""
        gradient = np.array([1.0, 2.0, 3.0])
        
        # Nearly singular matrix
        fisher_metric = np.array([
            [1e-10, 0, 0],
            [0, 1e-10, 0],
            [0, 0, 1e-10]
        ])
        
        # Should not raise due to damping
        nat_grad = natural_gradient(gradient, fisher_metric, damping=1e-4)
        
        assert np.all(np.isfinite(nat_grad))


class TestHellingerDistance:
    """Test Hellinger distance."""
    
    def test_identical(self):
        """Test Hellinger distance between identical distributions."""
        p = np.array([0.5, 0.3, 0.2])
        
        distance = hellinger_distance(p, p)
        
        assert np.isclose(distance, 0.0, atol=1e-6)
    
    def test_orthogonal(self):
        """Test Hellinger distance between orthogonal distributions."""
        p = np.array([1.0, 0.0, 0.0])
        q = np.array([0.0, 1.0, 0.0])
        
        distance = hellinger_distance(p, q)
        
        # Maximum Hellinger distance is 1.0
        assert np.isclose(distance, 1.0, atol=1e-6)
    
    def test_range(self):
        """Test Hellinger distance is in [0, 1]."""
        p = np.random.rand(10)
        q = np.random.rand(10)
        
        distance = hellinger_distance(p, q)
        
        assert 0.0 <= distance <= 1.0


class TestKLDivergence:
    """Test Kullback-Leibler divergence."""
    
    def test_identical(self):
        """Test KL divergence between identical distributions."""
        p = np.array([0.5, 0.3, 0.2])
        
        kl = kl_divergence(p, p)
        
        assert np.isclose(kl, 0.0, atol=1e-6)
    
    def test_asymmetry(self):
        """Test KL divergence is asymmetric."""
        p = np.array([0.5, 0.3, 0.2])
        q = np.array([0.45, 0.35, 0.2])
        
        kl_pq = kl_divergence(p, q)
        kl_qp = kl_divergence(q, p)
        
        # KL(p||q) != KL(q||p)
        assert not np.isclose(kl_pq, kl_qp)
    
    def test_non_negative(self):
        """Test KL divergence is non-negative."""
        p = np.random.rand(10)
        q = np.random.rand(10)
        
        kl = kl_divergence(p, q)
        
        assert kl >= 0.0


class TestJSDivergence:
    """Test Jensen-Shannon divergence."""
    
    def test_identical(self):
        """Test JS divergence between identical distributions."""
        p = np.array([0.5, 0.3, 0.2])
        
        js = js_divergence(p, p)
        
        assert np.isclose(js, 0.0, atol=1e-6)
    
    def test_symmetry(self):
        """Test JS divergence is symmetric."""
        p = np.array([0.5, 0.3, 0.2])
        q = np.array([0.45, 0.35, 0.2])
        
        js_pq = js_divergence(p, q)
        js_qp = js_divergence(q, p)
        
        # JS(p||q) = JS(q||p)
        assert np.isclose(js_pq, js_qp)
    
    def test_range(self):
        """Test JS divergence is in [0, log(2)]."""
        p = np.random.rand(10)
        q = np.random.rand(10)
        
        js = js_divergence(p, q)
        
        assert 0.0 <= js <= np.log(2)


class TestEuclideanFallback:
    """Test Euclidean fallback (for approximate search only)."""
    
    def test_euclidean_fallback(self):
        """Test Euclidean fallback distance."""
        basin_a = np.array([1.0, 2.0, 3.0])
        basin_b = np.array([2.0, 3.0, 4.0])
        
        distance = geodesic_distance_euclidean_fallback(basin_a, basin_b)
        
        # Should compute L2 norm
        expected = np.linalg.norm(basin_a - basin_b)
        assert np.isclose(distance, expected)
    
    def test_warning_message(self, caplog):
        """Test that using fallback generates warning."""
        import logging
        
        basin_a = np.array([1.0, 2.0, 3.0])
        basin_b = np.array([2.0, 3.0, 4.0])
        
        with caplog.at_level(logging.DEBUG):
            geodesic_distance_euclidean_fallback(basin_a, basin_b)
        
        # Should log warning about Euclidean fallback
        assert any("Euclidean fallback" in record.message for record in caplog.records)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
