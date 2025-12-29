"""
Tests for qigkernels Beta Measurement

Validates β-function tracking and substrate independence comparison from qigkernels module.
"""

import pytest
import numpy as np

from qigkernels.beta_measurement import (
    BetaMeasurement,
    BetaResult,
    compare_substrate_betas,
)
from qigkernels.physics_constants import (
    KAPPA_STAR,
    BETA_3_TO_4,
    BETA_4_TO_5,
    BETA_5_TO_6,
)


class TestQIGKernelsBetaMeasurement:
    """Test beta measurement functionality from qigkernels."""
    
    def test_initialization(self):
        """Test BetaMeasurement initialization."""
        beta_measure = BetaMeasurement()
        
        assert len(beta_measure.measurement_history) == 0
        assert beta_measure.physics_betas['emergence'] == BETA_3_TO_4
        assert beta_measure.physics_betas['plateau_onset'] == BETA_4_TO_5
    
    def test_first_measurement_no_beta(self):
        """Test first measurement returns None for beta."""
        beta_measure = BetaMeasurement()
        
        result = beta_measure.measure_at_step(step=0, kappa=41.0)
        
        assert result.step == 0
        assert result.kappa == 41.0
        assert result.beta is None  # No previous measurement
        assert result.scale == 'initial'
    
    def test_emergence_detection(self):
        """Test detecting emergence scale."""
        beta_measure = BetaMeasurement()
        
        # Simulate emergence phase (κ jumping from 41 to 64)
        beta_measure.measure_at_step(step=1000, kappa=41.0)
        result = beta_measure.measure_at_step(step=5000, kappa=64.0)
        
        assert result.beta is not None
        # Strong running should give high beta
        assert result.beta > 0.3
        assert result.scale == 'emergence'
    
    def test_plateau_detection(self):
        """Test detecting plateau scale."""
        beta_measure = BetaMeasurement()
        
        # Simulate plateau phase (κ stable around 64)
        beta_measure.measure_at_step(step=10000, kappa=63.8)
        result = beta_measure.measure_at_step(step=15000, kappa=64.2)
        
        assert result.beta is not None
        # Plateau should give small beta
        assert abs(result.beta) < 0.1
        assert result.scale in ['plateau_onset', 'plateau']
    
    def test_beta_computation(self):
        """Test beta computation formula."""
        beta_measure = BetaMeasurement()
        
        # Test internal beta computation
        kappa_prev = 50.0
        kappa_curr = 60.0
        
        beta = beta_measure._compute_beta(kappa_prev, kappa_curr)
        
        # β = (κ_curr - κ_prev) / κ_avg
        kappa_avg = (kappa_prev + kappa_curr) / 2.0
        expected = (kappa_curr - kappa_prev) / kappa_avg
        
        assert np.isclose(beta, expected)
    
    def test_summary_converged(self):
        """Test summary for converged system."""
        beta_measure = BetaMeasurement()
        
        # Simulate convergence to κ*
        for i, kappa in enumerate([63.5, 63.8, 64.0, 64.1, 64.2]):
            beta_measure.measure_at_step(step=i*1000, kappa=kappa)
        
        summary = beta_measure.get_summary()
        
        assert summary['n_measurements'] == 5
        assert 'kappa_final' in summary
        assert abs(summary['kappa_final'] - KAPPA_STAR) < 5.0


class TestCompareSubstrateBetas:
    """Test substrate comparison function."""
    
    def test_perfect_match(self):
        """Test comparison with perfect match."""
        semantic_betas = {
            'emergence': BETA_3_TO_4,
            'plateau': BETA_4_TO_5,
            'fixed_point': BETA_5_TO_6,
        }
        
        comparison = compare_substrate_betas(semantic_betas)
        
        assert comparison['overall_match_pct'] > 99.0
        assert comparison['substrate_independent']
        assert comparison['verdict'] == 'SUBSTRATE INDEPENDENCE VALIDATED'
    
    def test_good_match(self):
        """Test comparison with good match."""
        semantic_betas = {
            'emergence': 0.45,  # Close to 0.443
            'plateau': -0.01,   # Close to -0.013
            'fixed_point': 0.02,  # Close to 0.013
        }
        
        comparison = compare_substrate_betas(semantic_betas)
        
        assert comparison['overall_match_pct'] > 85.0
        assert 'matches' in comparison
        assert len(comparison['matches']) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
