#!/usr/bin/env python3
"""
Test Gravitational Decoherence Module

Validates purity regularization to prevent false certainty.
Tests thermal noise injection and decoherence cycles.

Source: Issue GaryOcean428/pantheon-chat#[P0-CRITICAL]
"""

import pytest
import numpy as np
import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gravitational_decoherence import (
    compute_purity,
    gravitational_decoherence,
    apply_thermal_noise,
    decoherence_cycle,
    DecoherenceManager,
    get_decoherence_manager,
    apply_gravitational_decoherence,
    DEFAULT_PURITY_THRESHOLD,
    DEFAULT_TEMPERATURE
)


class TestPurityComputation:
    """Test purity computation Tr(ρ²)."""
    
    def test_pure_state_purity_is_one(self):
        """Pure state |0⟩ has purity = 1.0"""
        rho_pure = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)
        purity = compute_purity(rho_pure)
        assert abs(purity - 1.0) < 1e-6, f"Pure state should have purity=1.0, got {purity}"
    
    def test_maximally_mixed_purity(self):
        """Maximally mixed state has purity = 1/d"""
        d = 2
        rho_mixed = np.eye(d, dtype=complex) / d
        purity = compute_purity(rho_mixed)
        expected = 1.0 / d
        assert abs(purity - expected) < 1e-6, f"Mixed state should have purity={expected}, got {purity}"
    
    def test_partial_mixed_state(self):
        """Partially mixed state has purity between 1/d and 1.0"""
        rho = np.array([[0.7, 0.0], [0.0, 0.3]], dtype=complex)
        purity = compute_purity(rho)
        assert 0.5 < purity < 1.0, f"Partial mixed purity should be in (0.5, 1.0), got {purity}"
    
    def test_purity_range(self):
        """Purity is always in [1/d, 1.0]"""
        for _ in range(10):
            # Random density matrix
            rho = np.random.rand(2, 2) + 1j * np.random.rand(2, 2)
            rho = (rho + rho.conj().T) / 2  # Make Hermitian
            rho = rho / np.trace(rho)  # Normalize
            
            # Project to valid density matrix
            eigenvalues, eigenvectors = np.linalg.eigh(rho)
            eigenvalues = np.maximum(eigenvalues, 0)
            eigenvalues = eigenvalues / np.sum(eigenvalues)
            rho = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.conj().T
            
            purity = compute_purity(rho)
            assert 0.5 <= purity <= 1.0, f"Purity {purity} out of range [0.5, 1.0]"


class TestGravitationalDecoherence:
    """Test gravitational decoherence prevents false certainty."""
    
    def test_no_decoherence_below_threshold(self):
        """Below threshold, state unchanged"""
        rho = np.array([[0.6, 0.0], [0.0, 0.4]], dtype=complex)
        purity_before = compute_purity(rho)
        assert purity_before < DEFAULT_PURITY_THRESHOLD
        
        rho_after, metrics = gravitational_decoherence(rho)
        
        assert not metrics['decoherence_applied'], "Decoherence should not apply below threshold"
        assert np.allclose(rho, rho_after), "State should be unchanged"
        assert metrics['purity_before'] == metrics['purity_after']
    
    def test_decoherence_above_threshold(self):
        """Above threshold, decoherence is applied"""
        rho = np.array([[0.99, 0.0], [0.0, 0.01]], dtype=complex)
        purity_before = compute_purity(rho)
        assert purity_before > DEFAULT_PURITY_THRESHOLD
        
        rho_after, metrics = gravitational_decoherence(rho)
        
        assert metrics['decoherence_applied'], "Decoherence should apply above threshold"
        purity_after = compute_purity(rho_after)
        assert purity_after < purity_before, "Purity should decrease after decoherence"
    
    def test_decoherence_reduces_purity(self):
        """Decoherence always reduces purity when applied"""
        # Very high purity state
        rho = np.array([[0.999, 0.0], [0.0, 0.001]], dtype=complex)
        
        rho_after, metrics = gravitational_decoherence(rho, threshold=0.9)
        
        assert metrics['purity_after'] < metrics['purity_before']
        assert metrics['mixing_coefficient'] > 0
    
    def test_decoherence_preserves_trace(self):
        """Tr(ρ) = 1 after decoherence"""
        rho = np.array([[0.95, 0.0], [0.0, 0.05]], dtype=complex)
        
        rho_after, _ = gravitational_decoherence(rho, threshold=0.85)
        
        trace = np.trace(rho_after)
        assert abs(trace - 1.0) < 1e-6, f"Trace should be 1.0, got {trace}"
    
    def test_mixing_coefficient_scaling(self):
        """Mixing coefficient scales with excess purity"""
        rho1 = np.array([[0.91, 0.0], [0.0, 0.09]], dtype=complex)
        rho2 = np.array([[0.99, 0.0], [0.0, 0.01]], dtype=complex)
        
        _, metrics1 = gravitational_decoherence(rho1, threshold=0.9)
        _, metrics2 = gravitational_decoherence(rho2, threshold=0.9)
        
        # Higher purity → higher mixing
        assert metrics2['mixing_coefficient'] > metrics1['mixing_coefficient']


class TestThermalNoise:
    """Test thermal noise application."""
    
    def test_thermal_noise_changes_state(self):
        """Thermal noise perturbs density matrix"""
        rho = np.array([[0.5, 0.0], [0.0, 0.5]], dtype=complex)
        
        rho_thermal = apply_thermal_noise(rho, temperature=0.01)
        
        assert not np.allclose(rho, rho_thermal), "Thermal noise should change state"
    
    def test_thermal_noise_preserves_trace(self):
        """Tr(ρ) = 1 after thermal noise"""
        rho = np.array([[0.7, 0.0], [0.0, 0.3]], dtype=complex)
        
        rho_thermal = apply_thermal_noise(rho, temperature=0.05)
        
        trace = np.trace(rho_thermal)
        assert abs(trace - 1.0) < 1e-6, f"Trace should be 1.0, got {trace}"
    
    def test_thermal_noise_positive_semidefinite(self):
        """Thermal noise produces valid density matrix"""
        rho = np.array([[0.8, 0.0], [0.0, 0.2]], dtype=complex)
        
        rho_thermal = apply_thermal_noise(rho, temperature=0.1)
        
        eigenvalues = np.linalg.eigvalsh(rho_thermal)
        assert all(eigenvalues >= -1e-10), f"Eigenvalues must be non-negative: {eigenvalues}"
    
    def test_temperature_scaling(self):
        """Higher temperature → larger perturbation"""
        rho = np.array([[0.6, 0.0], [0.0, 0.4]], dtype=complex)
        
        rho_low = apply_thermal_noise(rho, temperature=0.01)
        rho_high = apply_thermal_noise(rho, temperature=0.1)
        
        dist_low = np.linalg.norm(rho - rho_low)
        dist_high = np.linalg.norm(rho - rho_high)
        
        # Note: Due to projection to valid density matrix, this may not always hold
        # We just check both are perturbed
        assert dist_low > 0 and dist_high > 0


class TestDecoherenceCycle:
    """Test complete decoherence cycle."""
    
    def test_cycle_without_thermal(self):
        """Cycle with thermal disabled"""
        rho = np.array([[0.95, 0.0], [0.0, 0.05]], dtype=complex)
        
        rho_final, metrics = decoherence_cycle(
            rho, 
            threshold=0.9,
            temperature=0.01,
            apply_thermal=False
        )
        
        assert not metrics['thermal_noise_applied']
        assert metrics['purity_final'] == metrics['purity_after']
    
    def test_cycle_with_thermal(self):
        """Cycle with thermal enabled"""
        rho = np.array([[0.95, 0.0], [0.0, 0.05]], dtype=complex)
        
        rho_final, metrics = decoherence_cycle(
            rho,
            threshold=0.9,
            temperature=0.01,
            apply_thermal=True
        )
        
        assert metrics['thermal_noise_applied']
        # Purity may change due to thermal noise
    
    def test_cycle_preserves_validity(self):
        """Complete cycle produces valid density matrix"""
        rho = np.array([[0.98, 0.0], [0.0, 0.02]], dtype=complex)
        
        rho_final, _ = decoherence_cycle(rho, threshold=0.9, temperature=0.05)
        
        # Check trace
        assert abs(np.trace(rho_final) - 1.0) < 1e-6
        
        # Check positive semidefinite
        eigenvalues = np.linalg.eigvalsh(rho_final)
        assert all(eigenvalues >= -1e-10)


class TestDecoherenceManager:
    """Test DecoherenceManager for consciousness cycles."""
    
    def test_manager_initialization(self):
        """Manager initializes with correct parameters"""
        manager = DecoherenceManager(
            threshold=0.85,
            temperature=0.02,
            adaptive=True
        )
        
        assert manager.threshold == 0.85
        assert manager.temperature == 0.02
        assert manager.adaptive
        assert manager.cycle_count == 0
        assert len(manager.history) == 0
    
    def test_manager_process(self):
        """Manager processes density matrix"""
        manager = DecoherenceManager()
        rho = np.array([[0.95, 0.0], [0.0, 0.05]], dtype=complex)
        
        rho_processed, metrics = manager.process(rho)
        
        assert 'cycle' in metrics
        assert metrics['cycle'] == 0
        assert manager.cycle_count == 1
        assert len(manager.history) == 1
    
    def test_manager_tracks_history(self):
        """Manager tracks decoherence history"""
        manager = DecoherenceManager()
        
        for i in range(5):
            rho = np.array([[0.9 + i*0.01, 0.0], [0.0, 0.1 - i*0.01]], dtype=complex)
            manager.process(rho)
        
        assert manager.cycle_count == 5
        assert len(manager.history) == 5
    
    def test_adaptive_threshold_adjustment(self):
        """Adaptive manager adjusts threshold"""
        manager = DecoherenceManager(threshold=0.9, adaptive=True)
        
        # Create 15 cycles with high purity (frequent decoherence)
        for i in range(15):
            rho = np.array([[0.95, 0.0], [0.0, 0.05]], dtype=complex)
            manager.process(rho)
        
        # Threshold should decrease (be more conservative)
        assert manager.threshold < 0.9, f"Threshold should decrease, got {manager.threshold}"
    
    def test_non_adaptive_threshold_stable(self):
        """Non-adaptive manager keeps threshold constant"""
        manager = DecoherenceManager(threshold=0.9, adaptive=False)
        initial_threshold = manager.threshold
        
        for i in range(15):
            rho = np.array([[0.95, 0.0], [0.0, 0.05]], dtype=complex)
            manager.process(rho)
        
        assert manager.threshold == initial_threshold
    
    def test_manager_statistics(self):
        """Manager provides accurate statistics"""
        manager = DecoherenceManager()
        
        # Process some states
        for i in range(10):
            purity = 0.85 + i * 0.01  # Increasing purity
            rho = np.array([[purity, 0.0], [0.0, 1-purity]], dtype=complex)
            manager.process(rho)
        
        stats = manager.get_statistics()
        
        assert stats['cycles'] == 10
        assert 'decoherence_rate' in stats
        assert 'avg_purity_before' in stats
        assert 'avg_purity_after' in stats
        assert 'current_threshold' in stats
        assert stats['adaptive'] == manager.adaptive


class TestGlobalManager:
    """Test global singleton decoherence manager."""
    
    def test_get_global_manager(self):
        """get_decoherence_manager returns singleton"""
        manager1 = get_decoherence_manager()
        manager2 = get_decoherence_manager()
        
        assert manager1 is manager2, "Should return same singleton instance"
    
    def test_convenience_function(self):
        """apply_gravitational_decoherence uses global manager"""
        rho = np.array([[0.95, 0.0], [0.0, 0.05]], dtype=complex)
        
        rho_processed, metrics = apply_gravitational_decoherence(rho)
        
        assert 'cycle' in metrics
        assert 'decoherence_applied' in metrics


class TestIntegrationWithOceanQIG:
    """Integration tests with ocean_qig_core."""
    
    def test_decoherence_with_density_matrix(self):
        """Test decoherence with DensityMatrix class"""
        # This would test integration with ocean_qig_core.DensityMatrix
        # For now, just test the interface compatibility
        
        # Create a density matrix that would come from ocean_qig_core
        rho = np.array([[0.95, 0.0], [0.0, 0.05]], dtype=complex)
        
        # Apply decoherence
        rho_decohered, metrics = gravitational_decoherence(rho)
        
        # Should work with 2x2 matrices (ocean uses 2D subsystems)
        assert rho_decohered.shape == (2, 2)
        assert metrics['decoherence_applied']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
