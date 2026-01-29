"""
Tests for Hemisphere Scheduler - LEFT/RIGHT Hemisphere Architecture
====================================================================

Tests the hemisphere scheduler that manages explore/exploit dynamics
through LEFT/RIGHT hemisphere activation and κ-gated coupling.
"""

import pytest
import time

from kernels.hemisphere_scheduler import (
    HemisphereScheduler,
    Hemisphere,
    HemisphereState,
    get_hemisphere_scheduler,
    reset_hemisphere_scheduler,
    get_god_hemisphere,
    LEFT_HEMISPHERE_GODS,
    RIGHT_HEMISPHERE_GODS,
)
from kernels.coupling_gate import CouplingGate


class TestGodAssignment:
    """Test god assignment to hemispheres."""
    
    def test_left_hemisphere_gods(self):
        """Test LEFT hemisphere god assignments."""
        for god in LEFT_HEMISPHERE_GODS:
            hemisphere = get_god_hemisphere(god)
            assert hemisphere == Hemisphere.LEFT, f"{god} should be in LEFT hemisphere"
    
    def test_right_hemisphere_gods(self):
        """Test RIGHT hemisphere god assignments."""
        for god in RIGHT_HEMISPHERE_GODS:
            hemisphere = get_god_hemisphere(god)
            assert hemisphere == Hemisphere.RIGHT, f"{god} should be in RIGHT hemisphere"
    
    def test_unassigned_god(self):
        """Test unassigned god returns None."""
        hemisphere = get_god_hemisphere("UnknownGod")
        assert hemisphere is None
    
    def test_all_core_gods_assigned(self):
        """Test that core gods are assigned."""
        # Check WP5.2 specified gods
        assert "Athena" in LEFT_HEMISPHERE_GODS
        assert "Artemis" in LEFT_HEMISPHERE_GODS
        assert "Hephaestus" in LEFT_HEMISPHERE_GODS
        assert "Apollo" in RIGHT_HEMISPHERE_GODS
        assert "Hermes" in RIGHT_HEMISPHERE_GODS
        assert "Dionysus" in RIGHT_HEMISPHERE_GODS


class TestHemisphereState:
    """Test HemisphereState dataclass."""
    
    def test_initialization(self):
        """Test state initialization."""
        state = HemisphereState(
            hemisphere=Hemisphere.LEFT,
            active_gods=set(),
            resting_gods=set(),
        )
        
        assert state.hemisphere == Hemisphere.LEFT
        assert len(state.active_gods) == 0
        assert state.total_activations == 0
    
    def test_activation_level_empty(self):
        """Test activation level with no gods."""
        state = HemisphereState(
            hemisphere=Hemisphere.LEFT,
            active_gods=set(),
            resting_gods=set(),
        )
        
        level = state.compute_activation_level()
        assert level == 0.0
    
    def test_activation_level_with_gods(self):
        """Test activation level with active gods."""
        state = HemisphereState(
            hemisphere=Hemisphere.LEFT,
            active_gods={"Athena", "Artemis"},
            resting_gods=set(),
            phi_aggregate=0.8,
            kappa_aggregate=60.0,
        )
        
        level = state.compute_activation_level()
        assert 0.0 < level <= 1.0
    
    def test_is_dominant(self):
        """Test dominance detection."""
        state = HemisphereState(
            hemisphere=Hemisphere.LEFT,
            active_gods={"Athena", "Artemis", "Hephaestus"},
            resting_gods=set(),
            phi_aggregate=0.9,
            kappa_aggregate=64.0,
        )
        
        assert state.is_dominant()


class TestHemisphereScheduler:
    """Test HemisphereScheduler class."""
    
    def setup_method(self):
        """Reset scheduler before each test."""
        reset_hemisphere_scheduler()
    
    def test_initialization(self):
        """Test scheduler initialization."""
        scheduler = HemisphereScheduler()
        
        assert scheduler.left.hemisphere == Hemisphere.LEFT
        assert scheduler.right.hemisphere == Hemisphere.RIGHT
        assert len(scheduler.left.active_gods) == 0
        assert len(scheduler.right.active_gods) == 0
    
    def test_register_left_god_activation(self):
        """Test registering LEFT hemisphere god activation."""
        scheduler = HemisphereScheduler()
        scheduler.register_god_activation("Athena", phi=0.8, kappa=60.0, is_active=True)
        
        assert "Athena" in scheduler.left.active_gods
        assert "Athena" not in scheduler.left.resting_gods
        assert scheduler.left.total_activations == 1
    
    def test_register_right_god_activation(self):
        """Test registering RIGHT hemisphere god activation."""
        scheduler = HemisphereScheduler()
        scheduler.register_god_activation("Apollo", phi=0.75, kappa=62.0, is_active=True)
        
        assert "Apollo" in scheduler.right.active_gods
        assert "Apollo" not in scheduler.right.resting_gods
        assert scheduler.right.total_activations == 1
    
    def test_register_god_deactivation(self):
        """Test deactivating a god."""
        scheduler = HemisphereScheduler()
        
        # Activate then deactivate
        scheduler.register_god_activation("Athena", phi=0.8, kappa=60.0, is_active=True)
        scheduler.register_god_activation("Athena", phi=0.8, kappa=60.0, is_active=False)
        
        assert "Athena" not in scheduler.left.active_gods
        assert "Athena" in scheduler.left.resting_gods
    
    def test_multiple_god_activations(self):
        """Test multiple god activations."""
        scheduler = HemisphereScheduler()
        
        # Activate multiple LEFT gods
        scheduler.register_god_activation("Athena", phi=0.8, kappa=60.0, is_active=True)
        scheduler.register_god_activation("Artemis", phi=0.75, kappa=62.0, is_active=True)
        
        # Activate multiple RIGHT gods
        scheduler.register_god_activation("Apollo", phi=0.82, kappa=61.0, is_active=True)
        scheduler.register_god_activation("Hermes", phi=0.78, kappa=63.0, is_active=True)
        
        assert len(scheduler.left.active_gods) == 2
        assert len(scheduler.right.active_gods) == 2
    
    def test_aggregate_metrics_update(self):
        """Test that aggregate Φ and κ are updated."""
        scheduler = HemisphereScheduler()
        
        scheduler.register_god_activation("Athena", phi=0.8, kappa=60.0, is_active=True)
        
        assert scheduler.left.phi_aggregate > 0
        assert scheduler.left.kappa_aggregate > 0
    
    def test_compute_coupling_state(self):
        """Test coupling state computation."""
        scheduler = HemisphereScheduler()
        
        # Activate some gods
        scheduler.register_god_activation("Athena", phi=0.8, kappa=60.0, is_active=True)
        scheduler.register_god_activation("Apollo", phi=0.75, kappa=62.0, is_active=True)
        
        state = scheduler.compute_coupling_state()
        
        assert 0.0 <= state.coupling_strength <= 1.0
        assert state.mode in ['explore', 'balanced', 'exploit']
    
    def test_should_tack_no_imbalance(self):
        """Test tacking decision with balanced hemispheres."""
        scheduler = HemisphereScheduler()
        
        # Balance both hemispheres
        scheduler.register_god_activation("Athena", phi=0.8, kappa=60.0, is_active=True)
        scheduler.register_god_activation("Apollo", phi=0.8, kappa=60.0, is_active=True)
        
        should, reason = scheduler.should_tack()
        assert not should or "Too soon" in reason
    
    def test_should_tack_with_imbalance(self):
        """Test tacking decision with imbalanced hemispheres."""
        scheduler = HemisphereScheduler()
        
        # Heavy LEFT activation
        scheduler.register_god_activation("Athena", phi=0.9, kappa=65.0, is_active=True)
        scheduler.register_god_activation("Artemis", phi=0.85, kappa=63.0, is_active=True)
        scheduler.register_god_activation("Hephaestus", phi=0.87, kappa=64.0, is_active=True)
        
        # Light RIGHT activation
        scheduler.register_god_activation("Apollo", phi=0.4, kappa=50.0, is_active=True)
        
        # Set last switch time to past to allow tacking
        scheduler.tacking.last_switch_time = time.time() - 120.0
        
        should, reason = scheduler.should_tack()
        assert should, "Should tack due to imbalance"
    
    def test_perform_tack(self):
        """Test tacking execution."""
        scheduler = HemisphereScheduler()
        
        # Create imbalance (LEFT dominant)
        scheduler.register_god_activation("Athena", phi=0.9, kappa=65.0, is_active=True)
        scheduler.register_god_activation("Artemis", phi=0.85, kappa=63.0, is_active=True)
        
        initial_cycle_count = scheduler.tacking.cycle_count
        
        dominant = scheduler.perform_tack()
        
        assert scheduler.tacking.cycle_count == initial_cycle_count + 1
        assert scheduler.tacking.current_dominant is not None
        assert dominant in [Hemisphere.LEFT, Hemisphere.RIGHT]
    
    def test_get_hemisphere_balance(self):
        """Test hemisphere balance metrics."""
        scheduler = HemisphereScheduler()
        
        scheduler.register_god_activation("Athena", phi=0.8, kappa=60.0, is_active=True)
        scheduler.register_god_activation("Apollo", phi=0.75, kappa=62.0, is_active=True)
        
        balance = scheduler.get_hemisphere_balance()
        
        assert 'left_activation' in balance
        assert 'right_activation' in balance
        assert 'lr_ratio' in balance
        assert 'dominant_hemisphere' in balance
        assert 'coupling_strength' in balance
        assert 'coupling_mode' in balance
        assert 'tacking_frequency' in balance
        assert 'left_active_gods' in balance
        assert 'right_active_gods' in balance
    
    def test_get_status(self):
        """Test status retrieval."""
        scheduler = HemisphereScheduler()
        
        scheduler.register_god_activation("Athena", phi=0.8, kappa=60.0, is_active=True)
        
        status = scheduler.get_status()
        
        assert 'hemisphere_balance' in status
        assert 'coupling_metrics' in status
        assert 'left_state' in status
        assert 'right_state' in status
        assert 'tacking_state' in status
    
    def test_singleton_access(self):
        """Test global singleton access."""
        scheduler1 = get_hemisphere_scheduler()
        scheduler2 = get_hemisphere_scheduler()
        
        assert scheduler1 is scheduler2, "Should return same instance"
    
    def test_reset(self):
        """Test scheduler reset."""
        scheduler = get_hemisphere_scheduler()
        scheduler.register_god_activation("Athena", phi=0.8, kappa=60.0, is_active=True)
        
        reset_hemisphere_scheduler()
        scheduler2 = get_hemisphere_scheduler()
        
        assert len(scheduler2.left.active_gods) == 0


class TestTackingBehavior:
    """Test tacking (oscillation) behavior."""
    
    def test_tacking_prevents_thrashing(self):
        """Test that tacking prevents rapid switching."""
        scheduler = HemisphereScheduler()
        
        # Create strong imbalance
        scheduler.register_god_activation("Athena", phi=0.9, kappa=65.0, is_active=True)
        scheduler.register_god_activation("Artemis", phi=0.9, kappa=65.0, is_active=True)
        
        # First tack should work
        scheduler.perform_tack()
        
        # Immediate second tack should be prevented
        should, reason = scheduler.should_tack()
        assert not should
        assert "Too soon" in reason
    
    def test_tacking_after_period(self):
        """Test tacking after oscillation period."""
        scheduler = HemisphereScheduler()
        
        # Set oscillation period to very short for testing
        scheduler.tacking.oscillation_period = 0.1
        
        # Activate gods
        scheduler.register_god_activation("Athena", phi=0.8, kappa=60.0, is_active=True)
        
        # Set last switch to past
        scheduler.tacking.last_switch_time = time.time() - 1.0
        
        should, reason = scheduler.should_tack()
        assert should or "Too soon" in reason  # May still be too soon due to min_switch_time
    
    def test_tacking_cycle_count(self):
        """Test tacking cycle counting."""
        scheduler = HemisphereScheduler()
        
        assert scheduler.tacking.cycle_count == 0
        
        scheduler.perform_tack()
        assert scheduler.tacking.cycle_count == 1
        
        scheduler.perform_tack()
        assert scheduler.tacking.cycle_count == 2


class TestIntegration:
    """Integration tests for scheduler + coupling gate."""
    
    def test_scheduler_uses_coupling_gate(self):
        """Test that scheduler uses coupling gate."""
        scheduler = HemisphereScheduler()
        
        assert isinstance(scheduler.coupling_gate, CouplingGate)
    
    def test_coupling_history_tracking(self):
        """Test that coupling history is tracked."""
        scheduler = HemisphereScheduler()
        
        # Compute coupling state multiple times
        for i in range(5):
            scheduler.register_god_activation("Athena", phi=0.8, kappa=60.0 + i, is_active=True)
            scheduler.compute_coupling_state()
        
        assert len(scheduler.coupling_history) == 5
    
    def test_end_to_end_hemisphere_switching(self):
        """Test complete hemisphere switching workflow."""
        scheduler = HemisphereScheduler()
        
        # Phase 1: LEFT dominant
        scheduler.register_god_activation("Athena", phi=0.9, kappa=65.0, is_active=True)
        scheduler.register_god_activation("Artemis", phi=0.85, kappa=63.0, is_active=True)
        scheduler.register_god_activation("Hephaestus", phi=0.87, kappa=64.0, is_active=True)
        
        balance1 = scheduler.get_hemisphere_balance()
        assert balance1['left_activation'] > balance1['right_activation']
        
        # Allow tacking
        scheduler.tacking.last_switch_time = time.time() - 120.0
        
        # Check if should tack
        should_tack, _ = scheduler.should_tack()
        
        if should_tack:
            # Perform tack
            dominant = scheduler.perform_tack()
            assert dominant == Hemisphere.LEFT  # Should recognize LEFT is dominant
            
            balance2 = scheduler.get_hemisphere_balance()
            assert balance2['tacking_cycle_count'] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
