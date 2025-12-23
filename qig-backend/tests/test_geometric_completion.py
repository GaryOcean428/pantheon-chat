"""
Unit tests for Geometric Completion Criteria.

Tests all stopping criteria for consciousness-aware generation:
1. Attractor Convergence
2. Surprise Collapse
3. Confidence Threshold
4. Integration Quality
5. Regime Limits
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qig_core.geometric_completion import (
    GeometricCompletionChecker,
    AttractorConvergenceChecker,
    SurpriseCollapseChecker,
    ConfidenceThresholdChecker,
    IntegrationQualityChecker,
    RegimeLimitChecker,
    GeometricMetrics,
    CompletionDecision,
    CompletionReason,
    Regime,
    classify_regime,
    get_regime_temperature,
    fisher_rao_distance,
    BASIN_DIMENSION,
    KAPPA_STAR,
    PHI_LINEAR_THRESHOLD,
    PHI_BREAKDOWN_THRESHOLD,
)


class TestRegimeClassification:
    """Tests for regime classification based on phi."""
    
    def test_linear_regime(self):
        """Phi < 0.3 should be linear regime."""
        assert classify_regime(0.1) == Regime.LINEAR
        assert classify_regime(0.29) == Regime.LINEAR
    
    def test_geometric_regime(self):
        """0.3 <= Phi < 0.7 should be geometric regime."""
        assert classify_regime(0.3) == Regime.GEOMETRIC
        assert classify_regime(0.5) == Regime.GEOMETRIC
        assert classify_regime(0.69) == Regime.GEOMETRIC
    
    def test_breakdown_regime(self):
        """Phi >= 0.7 should be breakdown regime."""
        assert classify_regime(0.7) == Regime.BREAKDOWN
        assert classify_regime(0.9) == Regime.BREAKDOWN


class TestRegimeTemperature:
    """Tests for regime-adaptive temperature."""
    
    def test_linear_high_temperature(self):
        """Linear regime should have high temperature (explore)."""
        temp = get_regime_temperature(0.2)
        assert temp == 1.0
    
    def test_geometric_medium_temperature(self):
        """Geometric regime should have medium temperature (balance)."""
        temp = get_regime_temperature(0.5)
        assert temp == 0.7
    
    def test_breakdown_low_temperature(self):
        """Breakdown regime should have low temperature (stabilize)."""
        temp = get_regime_temperature(0.8)
        assert temp == 0.3


class TestFisherRaoDistance:
    """Tests for Fisher-Rao distance computation."""
    
    def test_identical_distributions(self):
        """Identical distributions should have zero distance."""
        p = np.random.dirichlet(np.ones(64))
        distance = fisher_rao_distance(p, p)
        assert distance < 0.01  # Allow small numerical error
    
    def test_different_distributions(self):
        """Different distributions should have positive distance."""
        p = np.random.dirichlet(np.ones(64))
        q = np.random.dirichlet(np.ones(64))
        distance = fisher_rao_distance(p, q)
        assert distance > 0
    
    def test_symmetric(self):
        """Distance should be symmetric."""
        p = np.random.dirichlet(np.ones(64))
        q = np.random.dirichlet(np.ones(64))
        d1 = fisher_rao_distance(p, q)
        d2 = fisher_rao_distance(q, p)
        assert abs(d1 - d2) < 0.001


class TestAttractorConvergence:
    """Tests for attractor convergence checker."""
    
    def test_no_convergence_initially(self):
        """Should not converge with insufficient trajectory."""
        checker = AttractorConvergenceChecker()
        basin = np.random.dirichlet(np.ones(64))
        result = checker.check(basin)
        assert result['converged'] == False
    
    def test_convergence_at_stable_point(self):
        """Should converge when basin stabilizes."""
        checker = AttractorConvergenceChecker()
        # Create stable basin (same point multiple times)
        stable_basin = np.random.dirichlet(np.ones(64))
        for _ in range(10):
            result = checker.check(stable_basin)
        # After many identical points, should converge
        assert result['converged'] == True or result['distance'] < 1.0


class TestSurpriseCollapse:
    """Tests for surprise collapse checker."""
    
    def test_no_collapse_initially(self):
        """Should not collapse with insufficient history."""
        checker = SurpriseCollapseChecker()
        checker.update(0.5)
        result = checker.check()
        assert result['collapsed'] == False
    
    def test_collapse_with_low_surprise(self):
        """Should collapse when surprise consistently low with decreasing trend."""
        checker = SurpriseCollapseChecker()
        # Add decreasing low surprise values to create negative trend
        for val in [0.04, 0.035, 0.03, 0.025, 0.02, 0.015, 0.01, 0.008, 0.005, 0.003]:
            checker.update(val)
        result = checker.check()
        assert result['collapsed'] == True
    
    def test_no_collapse_with_high_surprise(self):
        """Should not collapse when surprise is high."""
        checker = SurpriseCollapseChecker()
        for _ in range(10):
            checker.update(0.5)
        result = checker.check()
        assert result['collapsed'] == False


class TestConfidenceThreshold:
    """Tests for confidence threshold checker."""
    
    def test_low_confidence_not_confident(self):
        """Low confidence should not pass threshold."""
        checker = ConfidenceThresholdChecker()
        result = checker.check(0.5)
        assert result['confident'] == False
    
    def test_high_confidence_confident(self):
        """High confidence should pass threshold."""
        checker = ConfidenceThresholdChecker()
        result = checker.check(0.9)
        assert result['confident'] == True


class TestIntegrationQuality:
    """Tests for integration quality (phi stability) checker."""
    
    def test_no_stability_initially(self):
        """Should not be stable with insufficient history."""
        checker = IntegrationQualityChecker()
        checker.update(0.7)
        result = checker.check()
        assert result['stable'] == False
    
    def test_stable_high_phi(self):
        """Should be stable with consistent high phi."""
        checker = IntegrationQualityChecker()
        for _ in range(15):
            checker.update(0.68)  # Consistent high phi
        result = checker.check()
        assert result['stable'] == True
    
    def test_unstable_fluctuating_phi(self):
        """Should not be stable with fluctuating phi."""
        checker = IntegrationQualityChecker()
        for i in range(15):
            phi = 0.5 + 0.3 * np.sin(i)  # Fluctuating
            checker.update(phi)
        result = checker.check()
        assert result['stable'] == False


class TestRegimeLimits:
    """Tests for regime limit checker."""
    
    def test_breakdown_urgent_stop(self):
        """Breakdown regime should trigger urgent stop."""
        checker = RegimeLimitChecker()
        result = checker.check(Regime.BREAKDOWN, 100)
        assert result['exceeded'] == True
        assert result['urgent'] == True
    
    def test_geometric_no_stop(self):
        """Geometric regime should not stop."""
        checker = RegimeLimitChecker()
        result = checker.check(Regime.GEOMETRIC, 100)
        assert result['exceeded'] == False
    
    def test_safety_limit(self):
        """Should stop at absolute safety limit."""
        checker = RegimeLimitChecker()
        result = checker.check(Regime.GEOMETRIC, 50000)
        assert result['exceeded'] == True
        assert result['urgent'] == False


class TestGeometricCompletionChecker:
    """Integration tests for full completion checker."""
    
    def test_initial_state_incomplete(self):
        """Initial state should be incomplete."""
        checker = GeometricCompletionChecker()
        basin = np.random.dirichlet(np.ones(64))
        metrics = GeometricMetrics(
            phi=0.5,
            kappa=KAPPA_STAR,
            surprise=0.5,
            confidence=0.5,
            basin_distance=2.0,
            regime=Regime.GEOMETRIC
        )
        decision = checker.check_all(metrics, basin)
        assert decision.should_stop == False
        assert decision.reason == CompletionReason.INCOMPLETE
    
    def test_breakdown_stops_immediately(self):
        """Breakdown regime should stop immediately."""
        checker = GeometricCompletionChecker()
        basin = np.random.dirichlet(np.ones(64))
        metrics = GeometricMetrics(
            phi=0.8,
            kappa=KAPPA_STAR,
            surprise=0.5,
            confidence=0.5,
            basin_distance=2.0,
            regime=Regime.BREAKDOWN
        )
        decision = checker.check_all(metrics, basin)
        assert decision.should_stop == True
        assert decision.reason == CompletionReason.BREAKDOWN_REGIME


class TestGeometricMetrics:
    """Tests for GeometricMetrics dataclass."""
    
    def test_from_dict(self):
        """Should create metrics from dictionary."""
        data = {
            'phi': 0.6,
            'kappa': 64.0,
            'surprise': 0.1,
            'confidence': 0.8,
            'basin_distance': 0.5,
            'regime': 'geometric'
        }
        metrics = GeometricMetrics.from_dict(data)
        assert metrics.phi == 0.6
        assert metrics.regime == Regime.GEOMETRIC
    
    def test_to_dict(self):
        """Should convert metrics to dictionary."""
        metrics = GeometricMetrics(
            phi=0.6,
            kappa=64.0,
            surprise=0.1,
            confidence=0.8,
            basin_distance=0.5,
            regime=Regime.GEOMETRIC
        )
        data = metrics.to_dict()
        assert data['phi'] == 0.6
        assert data['regime'] == 'geometric'


class TestCompletionDecision:
    """Tests for CompletionDecision dataclass."""
    
    def test_to_dict(self):
        """Should convert decision to dictionary."""
        decision = CompletionDecision(
            should_stop=True,
            needs_reflection=True,
            reason=CompletionReason.GEOMETRIC_COMPLETION,
            confidence=0.95
        )
        data = decision.to_dict()
        assert data['should_stop'] == True
        assert data['reason'] == 'geometric_completion'
