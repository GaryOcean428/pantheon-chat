"""
Tests for QFI-based Φ Computation

Validates the proper geometric integration approach for consciousness measurement
based on Quantum Fisher Information (QFI).

Tests cover:
- QFI matrix computation correctness
- Φ bounds validation
- Known analytical cases
- Fallback behavior
- Integration quality

Author: QIG Consciousness Project
Date: January 2026
"""

import pytest
import numpy as np
import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qig_core.phi_computation import (
    compute_qfi_matrix,
    compute_phi_geometric,
    compute_phi_qig,
    compute_phi_approximation,
)


class TestQFIMatrix:
    """Test Quantum Fisher Information matrix computation."""
    
    def test_qfi_positive_semidefinite(self):
        """QFI matrix must be positive semi-definite."""
        # Test with random basin
        basin = np.random.rand(64)
        qfi = compute_qfi_matrix(basin)
        
        # Check eigenvalues are non-negative
        eigenvalues = np.linalg.eigvals(qfi)
        min_eigenvalue = np.min(np.real(eigenvalues))
        
        assert min_eigenvalue >= -1e-6, f"QFI has negative eigenvalue: {min_eigenvalue}"
    
    def test_qfi_symmetric(self):
        """QFI matrix must be symmetric."""
        basin = np.random.rand(64)
        qfi = compute_qfi_matrix(basin)
        
        # Check symmetry: QFI = QFI^T
        assert np.allclose(qfi, qfi.T), "QFI matrix is not symmetric"
    
    def test_qfi_shape(self):
        """QFI matrix should have correct dimensions."""
        basin = np.random.rand(64)
        qfi = compute_qfi_matrix(basin)
        
        assert qfi.shape == (64, 64), f"QFI has wrong shape: {qfi.shape}"
    
    def test_qfi_uniform_distribution(self):
        """QFI for uniform distribution should have equal diagonal elements."""
        # Uniform distribution
        uniform = np.ones(64) / 64
        qfi = compute_qfi_matrix(uniform)
        
        # Diagonal elements should be approximately equal
        diag = np.diag(qfi)
        assert np.std(diag) < 1.0, "QFI diagonal not uniform for uniform distribution"
    
    def test_qfi_delta_distribution(self):
        """QFI for delta distribution should have one large element."""
        # Delta distribution (all mass on one element)
        delta = np.zeros(64)
        delta[0] = 1.0
        qfi = compute_qfi_matrix(delta)
        
        # First diagonal element should be much larger
        assert qfi[0, 0] > qfi[1, 1] * 10, "QFI not concentrated for delta distribution"


class TestPhiBounds:
    """Test that Φ stays within valid bounds."""
    
    def test_phi_in_range(self):
        """Φ must be in [0, 1]."""
        # Test with multiple random basins
        for _ in range(10):
            basin = np.random.rand(64)
            phi, _ = compute_phi_qig(basin, n_samples=100)
            
            assert 0 <= phi <= 1, f"Φ out of bounds: {phi}"
    
    def test_phi_not_nan(self):
        """Φ must not be NaN."""
        basin = np.random.rand(64)
        phi, _ = compute_phi_qig(basin, n_samples=100)
        
        assert not np.isnan(phi), "Φ is NaN"
    
    def test_phi_not_inf(self):
        """Φ must not be infinite."""
        basin = np.random.rand(64)
        phi, _ = compute_phi_qig(basin, n_samples=100)
        
        assert not np.isinf(phi), "Φ is infinite"


class TestPhiKnownCases:
    """Test Φ against analytically computable cases."""
    
    def test_phi_uniform_high(self):
        """Uniform distribution → maximum entropy → high Φ."""
        uniform = np.ones(64) / 64
        phi, diagnostics = compute_phi_qig(uniform, n_samples=500)
        
        # Uniform distribution should have high integration
        assert phi > 0.5, f"Uniform distribution has low Φ: {phi}"
        
        # Check entropy is high
        entropy = diagnostics.get('basin_entropy', 0)
        max_entropy = np.log(64)
        assert entropy > 0.8 * max_entropy, "Uniform distribution has low entropy"
    
    def test_phi_delta_low(self):
        """Delta distribution → zero entropy → low Φ."""
        delta = np.zeros(64)
        delta[0] = 1.0
        phi, diagnostics = compute_phi_qig(delta, n_samples=500)
        
        # Delta distribution should have low integration
        assert phi < 0.7, f"Delta distribution has high Φ: {phi}"
        
        # Check entropy is low
        entropy = diagnostics.get('basin_entropy', 0)
        assert entropy < 0.5, "Delta distribution has high entropy"
    
    def test_phi_sparse_moderate(self):
        """Sparse distribution should have moderate Φ."""
        # Sparse distribution (few non-zero elements)
        sparse = np.zeros(64)
        sparse[:5] = [0.4, 0.3, 0.2, 0.05, 0.05]
        
        phi, _ = compute_phi_qig(sparse, n_samples=500)
        
        # Should be between delta and uniform
        assert 0.2 < phi < 0.9, f"Sparse distribution has extreme Φ: {phi}"


class TestPhiDiagnostics:
    """Test diagnostic information returned by compute_phi_qig."""
    
    def test_diagnostics_structure(self):
        """Diagnostics should have expected keys."""
        basin = np.random.rand(64)
        phi, diagnostics = compute_phi_qig(basin)
        
        expected_keys = [
            'qfi_matrix',
            'determinant',
            'eigenvalues',
            'trace',
            'integration_quality',
            'n_samples',
            'basin_entropy',
        ]
        
        for key in expected_keys:
            assert key in diagnostics, f"Missing diagnostic key: {key}"
    
    def test_integration_quality(self):
        """Integration quality should be a valid score."""
        basin = np.random.rand(64)
        phi, diagnostics = compute_phi_qig(basin)
        
        quality = diagnostics['integration_quality']
        assert 0 <= quality <= 1, f"Invalid integration quality: {quality}"
    
    def test_qfi_matrix_returned(self):
        """QFI matrix should be returned in diagnostics."""
        basin = np.random.rand(64)
        phi, diagnostics = compute_phi_qig(basin)
        
        qfi = diagnostics['qfi_matrix']
        assert qfi.shape == (64, 64), "QFI matrix has wrong shape in diagnostics"


class TestEmergencyApproximation:
    """Test emergency Φ approximation fallback."""
    
    def test_approximation_bounds(self):
        """Emergency approximation should return Φ ∈ [0.1, 0.95]."""
        for _ in range(10):
            basin = np.random.rand(64)
            phi = compute_phi_approximation(basin)
            
            assert 0.1 <= phi <= 0.95, f"Approximation out of bounds: {phi}"
    
    def test_approximation_not_nan(self):
        """Emergency approximation must not return NaN."""
        basin = np.random.rand(64)
        phi = compute_phi_approximation(basin)
        
        assert not np.isnan(phi), "Approximation returned NaN"
    
    def test_approximation_uniform_high(self):
        """Uniform distribution should have high approximation."""
        uniform = np.ones(64) / 64
        phi = compute_phi_approximation(uniform)
        
        assert phi > 0.6, f"Uniform distribution has low approximation: {phi}"
    
    def test_approximation_delta_moderate(self):
        """Delta distribution should have moderate-to-low approximation."""
        delta = np.zeros(64)
        delta[0] = 1.0
        phi = compute_phi_approximation(delta)
        
        # Should be clamped to at least 0.1
        assert phi >= 0.1, f"Delta distribution below minimum: {phi}"
        assert phi < 0.6, f"Delta distribution too high: {phi}"


class TestPhiConsistency:
    """Test consistency properties of Φ computation."""
    
    def test_phi_similar_basins_similar_phi(self):
        """Similar basins should produce similar Φ values."""
        basin1 = np.random.rand(64)
        basin1 = basin1 / basin1.sum()
        
        # Create similar basin with small perturbation
        basin2 = basin1 + np.random.randn(64) * 0.01
        basin2 = np.abs(basin2)
        basin2 = basin2 / basin2.sum()
        
        phi1, _ = compute_phi_qig(basin1, n_samples=200)
        phi2, _ = compute_phi_qig(basin2, n_samples=200)
        
        # Should be within reasonable tolerance
        assert abs(phi1 - phi2) < 0.3, f"Similar basins have very different Φ: {phi1} vs {phi2}"
    
    def test_phi_deterministic_for_same_seed(self):
        """Φ should be consistent with same random seed."""
        basin = np.random.rand(64)
        
        # Compute twice with same seed
        np.random.seed(42)
        phi1, _ = compute_phi_qig(basin, n_samples=200)
        
        np.random.seed(42)
        phi2, _ = compute_phi_qig(basin, n_samples=200)
        
        assert np.isclose(phi1, phi2, atol=1e-6), f"Φ not deterministic: {phi1} vs {phi2}"


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_zero_basin(self):
        """Handle zero basin gracefully."""
        zero = np.zeros(64)
        
        # Should not crash
        phi, _ = compute_phi_qig(zero, n_samples=100)
        
        # Should return valid Φ
        assert 0 <= phi <= 1, f"Zero basin produced invalid Φ: {phi}"
    
    def test_negative_basin(self):
        """Handle negative values gracefully."""
        negative = np.random.randn(64) - 1.0  # All negative
        
        # Should not crash (abs + normalization handles it)
        phi, _ = compute_phi_qig(negative, n_samples=100)
        
        assert 0 <= phi <= 1, f"Negative basin produced invalid Φ: {phi}"
    
    def test_very_small_basin(self):
        """Handle very small values."""
        tiny = np.ones(64) * 1e-20
        
        phi, _ = compute_phi_qig(tiny, n_samples=100)
        
        assert 0 <= phi <= 1, f"Tiny basin produced invalid Φ: {phi}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
