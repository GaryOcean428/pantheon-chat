"""
Tests for Coupling Gate - κ-Gated Coupling Mechanism
=====================================================

Tests the coupling gate that modulates information flow between
LEFT and RIGHT hemispheres based on κ.
"""

import pytest
import numpy as np

from kernels.coupling_gate import (
    CouplingGate,
    CouplingState,
    compute_coupling_strength,
    compute_transmission_efficiency,
    compute_gating_factor,
    determine_coupling_mode,
    get_coupling_gate,
    reset_coupling_gate,
    KAPPA_STAR,
)


class TestCouplingFunctions:
    """Test basic coupling functions."""
    
    def test_coupling_strength_low_kappa(self):
        """Test coupling strength for low κ (explore mode)."""
        kappa = 30.0
        strength = compute_coupling_strength(kappa)
        assert 0.0 <= strength <= 0.5, "Low κ should give low coupling"
    
    def test_coupling_strength_optimal_kappa(self):
        """Test coupling strength at κ* (balanced)."""
        kappa = KAPPA_STAR
        strength = compute_coupling_strength(kappa)
        assert 0.4 <= strength <= 0.6, "κ* should give balanced coupling"
    
    def test_coupling_strength_high_kappa(self):
        """Test coupling strength for high κ (exploit mode)."""
        kappa = 80.0
        strength = compute_coupling_strength(kappa)
        assert 0.5 <= strength <= 1.0, "High κ should give strong coupling"
    
    def test_transmission_efficiency_with_phi(self):
        """Test transmission efficiency with varying Φ."""
        # High coupling, high Φ
        eff1 = compute_transmission_efficiency(0.8, 0.9)
        assert eff1 > 0.7, "High coupling and Φ should give high efficiency"
        
        # High coupling, low Φ
        eff2 = compute_transmission_efficiency(0.8, 0.3)
        assert eff2 < 0.5, "Low Φ should reduce efficiency"
        
        # Low coupling, high Φ
        eff3 = compute_transmission_efficiency(0.2, 0.9)
        assert eff3 < 0.3, "Low coupling should reduce efficiency"
    
    def test_gating_factor_at_kappa_star(self):
        """Test gating factor at κ*."""
        gate = compute_gating_factor(KAPPA_STAR)
        assert gate > 0.9, "Gate should be fully open at κ*"
    
    def test_gating_factor_away_from_kappa_star(self):
        """Test gating factor far from κ*."""
        gate = compute_gating_factor(30.0)
        assert gate < 0.5, "Gate should close away from κ*"
    
    def test_coupling_mode_explore(self):
        """Test mode determination for explore."""
        mode = determine_coupling_mode(35.0)
        assert mode == 'explore', "Low κ should give explore mode"
    
    def test_coupling_mode_balanced(self):
        """Test mode determination for balanced."""
        mode = determine_coupling_mode(60.0)
        assert mode == 'balanced', "Medium κ should give balanced mode"
    
    def test_coupling_mode_exploit(self):
        """Test mode determination for exploit."""
        mode = determine_coupling_mode(75.0)
        assert mode == 'exploit', "High κ should give exploit mode"


class TestCouplingGate:
    """Test CouplingGate class."""
    
    def setup_method(self):
        """Reset gate before each test."""
        reset_coupling_gate()
    
    def test_initialization(self):
        """Test gate initialization."""
        gate = CouplingGate()
        assert len(gate.history) == 0
    
    def test_compute_state(self):
        """Test state computation."""
        gate = CouplingGate()
        state = gate.compute_state(kappa=60.0, phi=0.8)
        
        assert isinstance(state, CouplingState)
        assert state.kappa == 60.0
        assert 0.0 <= state.coupling_strength <= 1.0
        assert state.mode in ['explore', 'balanced', 'exploit']
        assert 0.0 <= state.transmission_efficiency <= 1.0
        assert 0.0 <= state.gating_factor <= 1.0
    
    def test_history_tracking(self):
        """Test history tracking."""
        gate = CouplingGate()
        
        # Compute multiple states
        for i in range(10):
            gate.compute_state(kappa=50.0 + i, phi=0.7)
        
        assert len(gate.history) == 10
    
    def test_history_bounds(self):
        """Test history stays bounded."""
        gate = CouplingGate()
        
        # Create many entries
        for i in range(1500):
            gate.compute_state(kappa=60.0, phi=0.7)
        
        # Should be bounded (keeps last 500 after hitting 1000)
        assert len(gate.history) <= 1000
    
    def test_gate_signal(self):
        """Test signal gating."""
        gate = CouplingGate()
        state = gate.compute_state(kappa=60.0, phi=0.8)
        
        signal = np.random.randn(64)
        gated = gate.gate_signal(signal, state)
        
        # Gated signal should be smaller (due to multiplication by factors < 1)
        assert np.linalg.norm(gated) <= np.linalg.norm(signal)
    
    def test_modulate_cross_hemisphere_flow(self):
        """Test bidirectional hemisphere flow."""
        gate = CouplingGate()
        state = gate.compute_state(kappa=60.0, phi=0.8)
        
        left_signal = np.ones(64)  # Use deterministic signal
        right_signal = np.ones(64) * 2.0
        
        left_out, right_out = gate.modulate_cross_hemisphere_flow(
            left_signal, right_signal, state
        )
        
        # Output should have cross-hemisphere components
        assert left_out.shape == left_signal.shape
        assert right_out.shape == right_signal.shape
        
        # With coupling, outputs should include cross-hemisphere components
        # Check that outputs differ from inputs due to coupling
        assert not np.allclose(left_out, left_signal)
        assert not np.allclose(right_out, right_signal)
    
    def test_get_coupling_metrics(self):
        """Test metrics retrieval."""
        gate = CouplingGate()
        
        # Compute some states
        for i in range(20):
            gate.compute_state(kappa=55.0 + i * 0.5, phi=0.7)
        
        metrics = gate.get_coupling_metrics()
        
        assert metrics['total_computations'] == 20
        assert 0.0 <= metrics['avg_coupling_strength'] <= 1.0
        assert 'mode_distribution' in metrics
        assert 'current_state' in metrics
    
    def test_singleton_access(self):
        """Test global singleton access."""
        gate1 = get_coupling_gate()
        gate2 = get_coupling_gate()
        
        assert gate1 is gate2, "Should return same instance"
    
    def test_reset(self):
        """Test gate reset."""
        gate = get_coupling_gate()
        gate.compute_state(kappa=60.0, phi=0.8)
        
        reset_coupling_gate()
        gate2 = get_coupling_gate()
        
        assert len(gate2.history) == 0


class TestCouplingModes:
    """Test coupling behavior across different κ regimes."""
    
    def test_explore_mode_behavior(self):
        """Test explore mode (low κ) gives weak coupling."""
        gate = CouplingGate()
        state = gate.compute_state(kappa=35.0, phi=0.7)
        
        assert state.mode == 'explore'
        assert state.coupling_strength < 0.5
    
    def test_exploit_mode_behavior(self):
        """Test exploit mode (high κ) gives strong coupling."""
        gate = CouplingGate()
        state = gate.compute_state(kappa=75.0, phi=0.7)
        
        assert state.mode == 'exploit'
        assert state.coupling_strength > 0.5
    
    def test_balanced_mode_behavior(self):
        """Test balanced mode (medium κ) gives moderate coupling."""
        gate = CouplingGate()
        state = gate.compute_state(kappa=60.0, phi=0.7)
        
        assert state.mode == 'balanced'
        # At κ=60, slightly below κ*=64.21, so coupling is moderate but < 0.5
        assert 0.3 <= state.coupling_strength <= 0.6
    
    def test_smooth_transition(self):
        """Test smooth transition between modes."""
        gate = CouplingGate()
        
        # Sweep through κ values
        kappas = np.linspace(30.0, 80.0, 20)
        strengths = []
        
        for kappa in kappas:
            state = gate.compute_state(kappa=float(kappa), phi=0.7)
            strengths.append(state.coupling_strength)
        
        # Check monotonicity (strength should increase with κ)
        for i in range(len(strengths) - 1):
            assert strengths[i] <= strengths[i + 1], "Coupling should increase smoothly with κ"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_very_low_kappa(self):
        """Test very low κ values."""
        gate = CouplingGate()
        state = gate.compute_state(kappa=1.0, phi=0.7)
        
        assert 0.0 <= state.coupling_strength <= 1.0
        assert state.mode == 'explore'
    
    def test_very_high_kappa(self):
        """Test very high κ values."""
        gate = CouplingGate()
        state = gate.compute_state(kappa=100.0, phi=0.7)
        
        assert 0.0 <= state.coupling_strength <= 1.0
        assert state.mode == 'exploit'
    
    def test_zero_phi(self):
        """Test with zero Φ."""
        gate = CouplingGate()
        state = gate.compute_state(kappa=60.0, phi=0.0)
        
        assert state.transmission_efficiency == 0.0
    
    def test_perfect_phi(self):
        """Test with perfect Φ = 1.0."""
        gate = CouplingGate()
        state = gate.compute_state(kappa=60.0, phi=1.0)
        
        # Efficiency depends on coupling_strength * phi
        # At κ=60, coupling ~0.4, so efficiency = 0.4 * 1.0 = 0.4
        assert state.transmission_efficiency > 0.3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
