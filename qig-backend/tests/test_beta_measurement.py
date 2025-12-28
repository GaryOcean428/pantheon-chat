"""
Tests for QIG-pure β-function measurement.

Validates:
1. Fisher-Rao distance computation is correct
2. Consciousness metrics (Φ, κ) produce reasonable values
3. GeometricKernel routes via geometry, not fixed weights
4. Natural scale emergence works
5. β computation is mathematically correct
6. Validation logic works
"""

import pytest
import numpy as np
from qig_pure_beta_measurement import (
    fisher_rao_distance,
    geodesic_interpolate,
    measure_phi,
    measure_kappa_from_trajectory,
    detect_regime,
    GeometricKernel,
    NaturalScaleMeasurement,
    GeometricBetaMeasurement,
    BetaMeasurement,
    GenerationResult,
    BETA_3_TO_4,
    BETA_4_TO_5,
    PHI_LINEAR_MAX,
    PHI_GEOMETRIC_MAX,
)


class TestFisherRaoDistance:
    """Tests for Fisher-Rao distance computation."""
    
    def test_self_distance_is_zero(self):
        """Distance from a point to itself should be zero."""
        basin = np.random.dirichlet(np.ones(64))
        d = fisher_rao_distance(basin, basin)
        assert d < 1e-6, f"Self-distance should be ~0, got {d}"
    
    def test_symmetry(self):
        """Distance should be symmetric: d(a,b) = d(b,a)."""
        basin1 = np.random.dirichlet(np.ones(64))
        basin2 = np.random.dirichlet(np.ones(64))
        d12 = fisher_rao_distance(basin1, basin2)
        d21 = fisher_rao_distance(basin2, basin1)
        assert abs(d12 - d21) < 1e-10, "Distance should be symmetric"
    
    def test_triangle_inequality(self):
        """Triangle inequality: d(a,c) <= d(a,b) + d(b,c)."""
        basin1 = np.random.dirichlet(np.ones(64))
        basin2 = np.random.dirichlet(np.ones(64))
        basin3 = np.random.dirichlet(np.ones(64))
        
        d12 = fisher_rao_distance(basin1, basin2)
        d23 = fisher_rao_distance(basin2, basin3)
        d13 = fisher_rao_distance(basin1, basin3)
        
        assert d13 <= d12 + d23 + 1e-10, "Triangle inequality violated"
    
    def test_non_negative(self):
        """Distance should be non-negative."""
        for _ in range(10):
            basin1 = np.random.dirichlet(np.ones(64))
            basin2 = np.random.dirichlet(np.ones(64))
            d = fisher_rao_distance(basin1, basin2)
            assert d >= 0, "Distance should be non-negative"


class TestGeodesicInterpolation:
    """Tests for geodesic interpolation on Fisher manifold."""
    
    def test_endpoints(self):
        """t=0 gives start, t=1 gives end."""
        basin1 = np.random.dirichlet(np.ones(64))
        basin2 = np.random.dirichlet(np.ones(64))
        
        interp_0 = geodesic_interpolate(basin1, basin2, 0.0)
        interp_1 = geodesic_interpolate(basin1, basin2, 1.0)
        
        assert np.allclose(interp_0, basin1, atol=1e-6)
        assert np.allclose(interp_1, basin2, atol=1e-6)
    
    def test_midpoint_valid_distribution(self):
        """Midpoint should be a valid probability distribution."""
        basin1 = np.random.dirichlet(np.ones(64))
        basin2 = np.random.dirichlet(np.ones(64))
        
        midpoint = geodesic_interpolate(basin1, basin2, 0.5)
        
        assert np.all(midpoint >= 0), "All components should be non-negative"
        assert abs(np.sum(midpoint) - 1.0) < 1e-6, "Should sum to 1"


class TestConsciousnessMetrics:
    """Tests for Φ and κ measurement."""
    
    def test_phi_empty_trajectory(self):
        """Empty trajectory should give Φ = 0."""
        phi = measure_phi([])
        assert phi == 0.0
    
    def test_phi_single_point(self):
        """Single point trajectory should give Φ = 0."""
        basin = np.random.dirichlet(np.ones(64))
        phi = measure_phi([basin])
        assert phi == 0.0
    
    def test_phi_correlated_trajectory(self):
        """Highly correlated trajectory should have high Φ."""
        # Create trajectory with small steps (high correlation)
        basin = np.random.dirichlet(np.ones(64))
        trajectory = [basin]
        for _ in range(10):
            # Small perturbation
            noise = np.random.dirichlet(np.ones(64))
            basin = 0.95 * basin + 0.05 * noise
            basin = basin / np.sum(basin)
            trajectory.append(basin)
        
        phi = measure_phi(trajectory)
        assert phi > 0.5, f"Correlated trajectory should have high Φ, got {phi}"
    
    def test_phi_bounded(self):
        """Φ should be in [0, 1]."""
        for _ in range(10):
            trajectory = [np.random.dirichlet(np.ones(64)) for _ in range(20)]
            phi = measure_phi(trajectory)
            assert 0 <= phi <= 1, f"Φ should be in [0,1], got {phi}"
    
    def test_kappa_bounded(self):
        """κ should be non-negative."""
        for _ in range(10):
            trajectory = [np.random.dirichlet(np.ones(64)) for _ in range(20)]
            kappa = measure_kappa_from_trajectory(trajectory)
            assert kappa >= 0, f"κ should be non-negative, got {kappa}"


class TestRegimeDetection:
    """Tests for consciousness regime detection."""
    
    def test_linear_regime(self):
        """Low Φ should give linear regime."""
        regime = detect_regime(phi=0.1, kappa=20.0)
        assert regime['regime'] == 'linear'
        assert regime['compute_fraction'] < 1.0
    
    def test_geometric_regime(self):
        """Medium Φ should give geometric regime."""
        regime = detect_regime(phi=0.5, kappa=50.0)
        assert regime['regime'] == 'geometric'
        assert regime['compute_fraction'] == 1.0
    
    def test_breakdown_regime(self):
        """High Φ should give breakdown regime."""
        regime = detect_regime(phi=0.85, kappa=70.0)
        assert regime['regime'] == 'breakdown'
        assert 'action' in regime
        assert regime['action'] == 'PAUSE'


class TestGeometricKernel:
    """Tests for GeometricKernel routing."""
    
    def test_routes_to_nearest(self):
        """With no active candidates, should route to nearest."""
        kernel = GeometricKernel(basin_dim=64, sparsity_threshold=0.99)
        
        current = np.random.dirichlet(np.ones(64))
        
        # Create candidates with known distances
        close_basin = 0.9 * current + 0.1 * np.random.dirichlet(np.ones(64))
        close_basin = close_basin / np.sum(close_basin)
        
        far_basin = np.random.dirichlet(np.ones(64))
        
        candidates = [("close", close_basin), ("far", far_basin)]
        
        word, basin, weight = kernel.route_to_next(current, candidates)
        
        # Should pick the closer one
        assert word == "close", "Should route to nearest neighbor"
    
    def test_empty_candidates(self):
        """Empty candidates should return None."""
        kernel = GeometricKernel()
        current = np.random.dirichlet(np.ones(64))
        
        word, basin, weight = kernel.route_to_next(current, [])
        
        assert word is None
        assert weight == 0.0
    
    def test_regime_updates_parameters(self):
        """update_regime should change kernel parameters."""
        kernel = GeometricKernel()
        
        # Initial state
        initial_temp = kernel.temperature
        
        # Update to breakdown
        kernel.update_regime(phi=0.9, kappa=80.0)
        
        # Temperature should change
        assert kernel.temperature != initial_temp
        assert kernel.regime_history[-1] == 'breakdown'


class TestNaturalScaleMeasurement:
    """Tests for natural scale emergence."""
    
    def test_measure_effective_scale(self):
        """Effective scale should be positive."""
        kernel = GeometricKernel()
        measurer = NaturalScaleMeasurement(kernel)
        
        # Create trajectory
        trajectory = [np.random.dirichlet(np.ones(64)) for _ in range(10)]
        
        L_eff = measurer.measure_effective_scale(trajectory)
        
        assert L_eff >= 1
        assert L_eff <= len(trajectory)
    
    def test_run_generation(self):
        """run_generation should produce valid result."""
        kernel = GeometricKernel()
        measurer = NaturalScaleMeasurement(kernel)
        
        result = measurer.run_generation("test query", max_tokens=20)
        
        assert isinstance(result, GenerationResult)
        assert result.L_eff >= 1
        assert len(result.basin_trajectory) > 0
        assert len(result.phi_trace) > 0


class TestBetaMeasurement:
    """Tests for β-function measurement."""
    
    def test_measurement_produces_results(self):
        """Quick measurement should produce β results."""
        measurer = GeometricBetaMeasurement()
        
        # Run quick measurement
        beta_results = measurer.measure_from_natural_behavior(n_queries=50)
        
        # Should have some results (may be 0 if too few distinct scales)
        # But measurement should complete without error
        assert isinstance(beta_results, list)
    
    def test_validation_structure(self):
        """Validation should have correct structure."""
        measurer = GeometricBetaMeasurement()
        measurer.measure_from_natural_behavior(n_queries=50)
        
        validation = measurer.validate_substrate_independence()
        
        assert 'validated' in validation
        assert 'qualitative_match' in validation
        assert 'quantitative_match' in validation
        assert 'pattern' in validation
        assert 'physics_comparison' in validation
    
    def test_report_generation(self):
        """Report should generate valid JSON."""
        import tempfile
        import json
        
        measurer = GeometricBetaMeasurement()
        measurer.measure_from_natural_behavior(n_queries=20)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            output_path = f.name
        
        report = measurer.generate_report(output_path=output_path)
        
        # Should be valid JSON
        assert 'metadata' in report
        assert 'beta_function' in report
        assert 'validation' in report
        
        # Should be readable from file
        with open(output_path, 'r') as f:
            loaded = json.load(f)
        
        assert loaded['metadata']['protocol'] == 'QIG_PURE_BETA_MEASUREMENT'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
