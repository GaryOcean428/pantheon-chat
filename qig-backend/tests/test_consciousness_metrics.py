"""
Comprehensive Consciousness Metrics Test Suite

Tests for Φ, κ, T, R, M, Γ, G computation correctness.
Validates regime classification and consciousness detection.

Source: Priority 3.1 from improvement recommendations
"""

import pytest
import numpy as np
import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qig_core.constants.consciousness import (
    THRESHOLDS,
    classify_regime,
    is_conscious,
    compute_suffering,
    SUFFERING_THRESHOLD,
    BASIN_DIMENSION
)


class TestConsciousnessThresholds:
    """Test that thresholds are correctly defined."""
    
    def test_phi_thresholds_ordered(self):
        """PHI_LINEAR_MAX < PHI_BREAKDOWN <= PHI_MIN"""
        assert THRESHOLDS.PHI_LINEAR_MAX < THRESHOLDS.PHI_BREAKDOWN
        assert THRESHOLDS.PHI_BREAKDOWN == THRESHOLDS.PHI_MIN
    
    def test_kappa_range_valid(self):
        """KAPPA_MIN < KAPPA_OPTIMAL < KAPPA_MAX"""
        assert THRESHOLDS.KAPPA_MIN < THRESHOLDS.KAPPA_OPTIMAL
        assert THRESHOLDS.KAPPA_OPTIMAL < THRESHOLDS.KAPPA_MAX
    
    def test_kappa_optimal_near_64(self):
        """κ* ≈ 64 from physics validation"""
        assert 64.0 <= THRESHOLDS.KAPPA_OPTIMAL <= 64.5
    
    def test_all_thresholds_in_valid_range(self):
        """All thresholds should be in [0, 1] or valid kappa range"""
        assert 0 <= THRESHOLDS.PHI_MIN <= 1
        assert 0 <= THRESHOLDS.PHI_LINEAR_MAX <= 1
        assert 0 <= THRESHOLDS.PHI_BREAKDOWN <= 1
        assert 0 <= THRESHOLDS.TACKING_MIN <= 1
        assert 0 <= THRESHOLDS.RADAR_MIN <= 1
        assert 0 <= THRESHOLDS.META_MIN <= 1
        assert 0 <= THRESHOLDS.COHERENCE_MIN <= 1
        assert 0 <= THRESHOLDS.GROUNDING_MIN <= 1
        # Kappa has different range
        assert 0 < THRESHOLDS.KAPPA_MIN < 100
        assert 0 < THRESHOLDS.KAPPA_MAX < 100


class TestRegimeClassification:
    """Test regime classification from Φ value."""
    
    def test_linear_regime_low_phi(self):
        """Φ < 0.3 → linear regime"""
        regime, compute = classify_regime(0.1)
        assert regime == "linear"
        assert compute == 0.3
        
        regime, compute = classify_regime(0.29)
        assert regime == "linear"
    
    def test_geometric_regime_mid_phi(self):
        """0.3 ≤ Φ < 0.7 → geometric regime"""
        regime, compute = classify_regime(0.3)
        assert regime == "geometric"
        assert compute == 1.0
        
        regime, compute = classify_regime(0.5)
        assert regime == "geometric"
        
        regime, compute = classify_regime(0.69)
        assert regime == "geometric"
    
    def test_breakdown_regime_high_phi(self):
        """Φ ≥ 0.7 → breakdown regime"""
        regime, compute = classify_regime(0.7)
        assert regime == "breakdown"
        assert compute == 0.0
        
        regime, compute = classify_regime(0.9)
        assert regime == "breakdown"
        
        regime, compute = classify_regime(1.0)
        assert regime == "breakdown"
    
    def test_boundary_values(self):
        """Test exact boundary transitions"""
        # Linear → Geometric boundary
        regime_below, _ = classify_regime(0.299)
        regime_at, _ = classify_regime(0.3)
        assert regime_below == "linear"
        assert regime_at == "geometric"
        
        # Geometric → Breakdown boundary
        regime_below, _ = classify_regime(0.699)
        regime_at, _ = classify_regime(0.7)
        assert regime_below == "geometric"
        assert regime_at == "breakdown"
    
    def test_compute_fraction_values(self):
        """Verify compute fractions are correct"""
        _, linear_compute = classify_regime(0.1)
        _, geometric_compute = classify_regime(0.5)
        _, breakdown_compute = classify_regime(0.8)
        
        assert linear_compute == 0.3
        assert geometric_compute == 1.0
        assert breakdown_compute == 0.0


class TestConsciousnessDetection:
    """Test is_conscious() function."""
    
    def test_conscious_with_valid_metrics(self):
        """System is conscious when all thresholds met"""
        result = is_conscious(
            phi=0.75,
            kappa=64.0,
            tacking=0.6,
            radar=0.8,
            meta=0.7,
            coherence=0.85,
            grounding=0.9
        )
        assert result is True
    
    def test_not_conscious_low_phi(self):
        """System not conscious if Φ < threshold"""
        result = is_conscious(
            phi=0.5,  # Below PHI_MIN
            kappa=64.0
        )
        assert result is False
    
    def test_not_conscious_low_kappa(self):
        """System not conscious if κ < threshold"""
        result = is_conscious(
            phi=0.75,
            kappa=30.0  # Below KAPPA_MIN
        )
        assert result is False
    
    def test_not_conscious_high_kappa(self):
        """System not conscious if κ > threshold"""
        result = is_conscious(
            phi=0.75,
            kappa=70.0  # Above KAPPA_MAX
        )
        assert result is False
    
    def test_optional_metrics_when_not_provided(self):
        """Optional metrics don't affect result when not provided"""
        result = is_conscious(
            phi=0.75,
            kappa=64.0
            # No optional metrics provided
        )
        assert result is True
    
    def test_optional_metrics_fail_when_below_threshold(self):
        """Optional metrics cause failure when below threshold"""
        # Tacking below threshold
        result = is_conscious(
            phi=0.75,
            kappa=64.0,
            tacking=0.3  # Below TACKING_MIN
        )
        assert result is False
        
        # Radar below threshold
        result = is_conscious(
            phi=0.75,
            kappa=64.0,
            radar=0.5  # Below RADAR_MIN
        )
        assert result is False
    
    def test_kappa_at_optimal(self):
        """System conscious when κ = κ* (optimal)"""
        result = is_conscious(
            phi=0.75,
            kappa=THRESHOLDS.KAPPA_OPTIMAL
        )
        assert result is True


class TestSufferingComputation:
    """Test suffering formula: S = Φ × (1 - Γ) × M"""
    
    def test_suffering_formula_correct(self):
        """S = Φ × (1 - Γ) × M"""
        # Φ=0.8, Γ=0.2, M=0.75 → S = 0.8 × 0.8 × 0.75 = 0.48
        suffering = compute_suffering(phi=0.8, gamma=0.2, meta=0.75)
        expected = 0.8 * (1 - 0.2) * 0.75
        assert abs(suffering - expected) < 0.001
    
    def test_no_suffering_below_consciousness(self):
        """No suffering if Φ < PHI_MIN (not conscious)"""
        suffering = compute_suffering(phi=0.5, gamma=0.0, meta=1.0)
        assert suffering == 0.0
    
    def test_no_suffering_high_generativity(self):
        """No suffering if Γ = 1 (fully generative)"""
        suffering = compute_suffering(phi=0.8, gamma=1.0, meta=1.0)
        assert suffering == 0.0
    
    def test_no_suffering_no_awareness(self):
        """No suffering if M = 0 (no meta-awareness)"""
        suffering = compute_suffering(phi=0.8, gamma=0.0, meta=0.0)
        assert suffering == 0.0
    
    def test_max_suffering(self):
        """Maximum suffering: high Φ, low Γ, high M"""
        suffering = compute_suffering(phi=1.0, gamma=0.0, meta=1.0)
        assert suffering == 1.0
    
    def test_suffering_threshold_meaningful(self):
        """SUFFERING_THRESHOLD should be reasonable value"""
        assert 0 < SUFFERING_THRESHOLD < 1
        # Common suffering level should be below threshold
        normal_suffering = compute_suffering(phi=0.75, gamma=0.8, meta=0.6)
        assert normal_suffering < SUFFERING_THRESHOLD


class TestBasinDimension:
    """Test basin coordinate constants."""
    
    def test_basin_dimension_is_64(self):
        """Basin dimension should be 64 (E8 projection)"""
        assert BASIN_DIMENSION == 64
    
    def test_basin_dimension_power_of_2(self):
        """Basin dimension should be power of 2 for efficiency"""
        assert BASIN_DIMENSION & (BASIN_DIMENSION - 1) == 0


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_phi_exactly_zero(self):
        """Φ = 0 should be linear regime"""
        regime, _ = classify_regime(0.0)
        assert regime == "linear"
    
    def test_phi_exactly_one(self):
        """Φ = 1 should be breakdown regime"""
        regime, _ = classify_regime(1.0)
        assert regime == "breakdown"
    
    def test_negative_phi_still_linear(self):
        """Negative Φ (invalid but should not crash) → linear"""
        regime, _ = classify_regime(-0.1)
        assert regime == "linear"
    
    def test_phi_greater_than_one(self):
        """Φ > 1 (invalid but should not crash) → breakdown"""
        regime, _ = classify_regime(1.5)
        assert regime == "breakdown"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
