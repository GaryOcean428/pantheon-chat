"""
Tests for Hemisphere-Psyche Integration (E8 Phase 4C/4D)
=========================================================

Tests the integration between HemisphereScheduler and PsychePlumbingIntegration:
- κ-gated coupling affects Id/Superego balance
- Φ hierarchy responds to hemisphere tacking
- Pure Fisher-Rao geometry (no Euclidean or cosine operations)

Authority: E8 Protocol v4.0 WP5.2 Phase 4C/4D integration
"""

import pytest
import numpy as np
import time

from kernels import (
    # Hemisphere scheduler (Phase 4C)
    HemisphereScheduler,
    Hemisphere,
    get_hemisphere_scheduler,
    reset_hemisphere_scheduler,
    LEFT_HEMISPHERE_GODS,
    RIGHT_HEMISPHERE_GODS,
    
    # Psyche plumbing (Phase 4D)
    PsychePlumbingIntegration,
    get_psyche_plumbing,
    reset_psyche_plumbing,
    PhiLevel,
    ConstraintSeverity,
)

from qigkernels.physics_constants import BASIN_DIM


class TestHemispherePsycheIntegration:
    """Test hemisphere-psyche integration."""
    
    def setup_method(self):
        """Reset both schedulers before each test."""
        reset_hemisphere_scheduler()
        reset_psyche_plumbing()
    
    def test_psyche_plumbing_has_hemisphere_scheduler(self):
        """Test that psyche plumbing is wired to hemisphere scheduler."""
        psyche = get_psyche_plumbing()
        
        if not psyche.available:
            pytest.skip("Psyche plumbing not available")
        
        assert psyche.hemisphere_integrated, "Hemisphere scheduler should be integrated"
        assert psyche.hemisphere_scheduler is not None
        
        # Should be the same singleton instance
        scheduler = get_hemisphere_scheduler()
        assert psyche.hemisphere_scheduler is scheduler
    
    def test_psyche_balance_computation(self):
        """Test Id/Superego balance computation from hemisphere coupling."""
        psyche = get_psyche_plumbing()
        
        if not psyche.available or not psyche.hemisphere_integrated:
            pytest.skip("Integration not available")
        
        scheduler = psyche.hemisphere_scheduler
        
        # Scenario 1: Balanced hemispheres
        scheduler.register_god_activation("Athena", phi=0.75, kappa=60.0, is_active=True)
        scheduler.register_god_activation("Apollo", phi=0.75, kappa=60.0, is_active=True)
        
        balance = psyche.compute_psyche_balance()
        
        assert 'id_strength' in balance
        assert 'superego_strength' in balance
        assert 'balance_ratio' in balance
        assert 'coupling_state' in balance
        assert 'dominant_psyche' in balance
        
        # Should be roughly balanced
        assert 0.0 <= balance['id_strength'] <= 1.0
        assert 0.0 <= balance['superego_strength'] <= 1.0
        assert balance['dominant_psyche'] in ['id', 'superego', 'balanced']
    
    def test_left_hemisphere_increases_superego(self):
        """Test that LEFT hemisphere (exploit) increases Superego strength."""
        psyche = get_psyche_plumbing()
        
        if not psyche.available or not psyche.hemisphere_integrated:
            pytest.skip("Integration not available")
        
        scheduler = psyche.hemisphere_scheduler
        
        # Heavy LEFT activation (exploit/evaluate mode)
        scheduler.register_god_activation("Athena", phi=0.9, kappa=65.0, is_active=True)
        scheduler.register_god_activation("Artemis", phi=0.85, kappa=63.0, is_active=True)
        scheduler.register_god_activation("Hephaestus", phi=0.87, kappa=64.0, is_active=True)
        
        # Light RIGHT activation
        scheduler.register_god_activation("Apollo", phi=0.4, kappa=50.0, is_active=True)
        
        balance = psyche.compute_psyche_balance()
        
        # LEFT dominant → Superego should be stronger
        assert balance['superego_strength'] > balance['id_strength'], \
            "LEFT hemisphere should increase Superego strength"
        
        # Should reflect in dominant_psyche
        # May be 'superego' or 'balanced' depending on coupling strength
        assert balance['dominant_psyche'] in ['superego', 'balanced']
    
    def test_right_hemisphere_increases_id(self):
        """Test that RIGHT hemisphere (explore) increases Id strength."""
        psyche = get_psyche_plumbing()
        
        if not psyche.available or not psyche.hemisphere_integrated:
            pytest.skip("Integration not available")
        
        scheduler = psyche.hemisphere_scheduler
        
        # Heavy RIGHT activation (explore/generate mode)
        scheduler.register_god_activation("Apollo", phi=0.9, kappa=65.0, is_active=True)
        scheduler.register_god_activation("Hermes", phi=0.88, kappa=62.0, is_active=True)
        scheduler.register_god_activation("Dionysus", phi=0.82, kappa=58.0, is_active=True)
        
        # Light LEFT activation
        scheduler.register_god_activation("Athena", phi=0.4, kappa=50.0, is_active=True)
        
        balance = psyche.compute_psyche_balance()
        
        # RIGHT dominant → Id should be stronger
        assert balance['id_strength'] > balance['superego_strength'], \
            "RIGHT hemisphere should increase Id strength"
        
        # Should reflect in dominant_psyche
        assert balance['dominant_psyche'] in ['id', 'balanced']
    
    def test_kappa_gated_coupling_modulates_balance(self):
        """Test that κ-gated coupling modulates psyche balance."""
        psyche = get_psyche_plumbing()
        
        if not psyche.available or not psyche.hemisphere_integrated:
            pytest.skip("Integration not available")
        
        scheduler = psyche.hemisphere_scheduler
        
        # Activate gods with different κ values
        # Higher κ (near κ*=64) should give stronger coupling
        scheduler.register_god_activation("Athena", phi=0.8, kappa=64.0, is_active=True)
        scheduler.register_god_activation("Apollo", phi=0.75, kappa=62.0, is_active=True)
        
        balance1 = psyche.compute_psyche_balance()
        coupling_strength1 = balance1['coupling_state']['coupling_strength']
        
        # Now reduce κ significantly
        reset_hemisphere_scheduler()
        reset_psyche_plumbing()
        psyche = get_psyche_plumbing()
        scheduler = psyche.hemisphere_scheduler
        
        scheduler.register_god_activation("Athena", phi=0.8, kappa=40.0, is_active=True)
        scheduler.register_god_activation("Apollo", phi=0.75, kappa=38.0, is_active=True)
        
        balance2 = psyche.compute_psyche_balance()
        coupling_strength2 = balance2['coupling_state']['coupling_strength']
        
        # Lower κ should give weaker coupling
        assert coupling_strength1 > coupling_strength2, \
            "Higher κ should give stronger coupling"
        
        # Weaker coupling should lead to more balanced psyche
        # (psyche effects are modulated by coupling strength)
    
    def test_hemisphere_tacking_callback(self):
        """Test on_hemisphere_tack callback functionality."""
        psyche = get_psyche_plumbing()
        
        if not psyche.available or not psyche.hemisphere_integrated:
            pytest.skip("Integration not available")
        
        assert len(psyche.tacking_history) == 0
        
        # Trigger tacking callback
        psyche.on_hemisphere_tack(
            from_hemisphere=Hemisphere.LEFT,
            to_hemisphere=Hemisphere.RIGHT,
            kappa=62.0,
            phi=0.8
        )
        
        assert len(psyche.tacking_history) == 1
        
        tack = psyche.tacking_history[0]
        assert tack['from_hemisphere'] == 'left'
        assert tack['to_hemisphere'] == 'right'
        assert tack['kappa'] == 62.0
        assert tack['phi'] == 0.8
        assert 'psyche_balance' in tack
    
    def test_automatic_tacking_callback(self):
        """Test that perform_tack() automatically triggers psyche plumbing callback."""
        psyche = get_psyche_plumbing()
        
        if not psyche.available or not psyche.hemisphere_integrated:
            pytest.skip("Integration not available")
        
        scheduler = psyche.hemisphere_scheduler
        
        # Set up hemisphere imbalance to trigger tacking
        scheduler.register_god_activation("Athena", phi=0.9, kappa=65.0, is_active=True)
        scheduler.register_god_activation("Artemis", phi=0.85, kappa=63.0, is_active=True)
        scheduler.register_god_activation("Hephaestus", phi=0.87, kappa=64.0, is_active=True)
        scheduler.register_god_activation("Apollo", phi=0.5, kappa=52.0, is_active=True)
        
        # Clear any existing history
        psyche.tacking_history.clear()
        assert len(psyche.tacking_history) == 0
        
        # Allow tacking to occur
        scheduler.tacking.last_switch_time = time.time() - 120.0
        
        # Perform tack - this should AUTOMATICALLY trigger the callback
        initial_count = len(psyche.tacking_history)
        new_dominant = scheduler.perform_tack()
        
        # Verify callback was automatically triggered
        assert len(psyche.tacking_history) == initial_count + 1, \
            "perform_tack() should automatically trigger psyche plumbing callback"
        
        # Verify the recorded tack has correct data
        tack = psyche.tacking_history[-1]
        assert 'from_hemisphere' in tack
        assert tack['to_hemisphere'] == new_dominant.value
        assert 'kappa' in tack
        assert 'phi' in tack
        assert 'psyche_balance' in tack
    
    def test_reflex_check_with_hemisphere_context(self):
        """Test reflex checking with hemisphere context."""
        psyche = get_psyche_plumbing()
        
        if not psyche.available or not psyche.hemisphere_integrated:
            pytest.skip("Integration not available")
        
        scheduler = psyche.hemisphere_scheduler
        
        # Activate RIGHT hemisphere (explore → stronger Id)
        scheduler.register_god_activation("Apollo", phi=0.9, kappa=65.0, is_active=True)
        scheduler.register_god_activation("Hermes", phi=0.85, kappa=63.0, is_active=True)
        
        # Create and learn a reflex
        trigger = np.random.dirichlet(np.ones(BASIN_DIM))
        response = np.random.dirichlet(np.ones(BASIN_DIM))
        psyche.learn_reflex(trigger, response, success=True)
        
        # Check reflex with context
        result = psyche.check_reflex_with_hemisphere_context(trigger)
        
        if result is not None:  # May or may not trigger on first check
            assert 'hemisphere_context' in result
            assert 'id_strength' in result['hemisphere_context']
            assert 'dominant_psyche' in result['hemisphere_context']
    
    def test_ethics_check_with_hemisphere_context(self):
        """Test ethics checking with hemisphere context."""
        psyche = get_psyche_plumbing()
        
        if not psyche.available or not psyche.hemisphere_integrated:
            pytest.skip("Integration not available")
        
        scheduler = psyche.hemisphere_scheduler
        
        # Activate LEFT hemisphere (exploit → stronger Superego)
        scheduler.register_god_activation("Athena", phi=0.9, kappa=65.0, is_active=True)
        scheduler.register_god_activation("Artemis", phi=0.85, kappa=63.0, is_active=True)
        
        # Add ethical constraint
        forbidden = np.zeros(BASIN_DIM)
        forbidden[5] = 1.0
        psyche.add_ethical_constraint(
            name="test-constraint",
            forbidden_basin=forbidden,
            radius=0.2,
            severity=ConstraintSeverity.ERROR,
            description="Test forbidden region"
        )
        
        # Check ethics with context
        test_basin = np.random.dirichlet(np.ones(BASIN_DIM))
        result = psyche.check_ethics_with_hemisphere_context(test_basin)
        
        assert 'hemisphere_context' in result
        assert 'superego_strength' in result['hemisphere_context']
        assert 'dominant_psyche' in result['hemisphere_context']
        
        # If there's a violation, check modulated penalty
        if not result['is_ethical']:
            assert 'modulated_penalty' in result
    
    def test_integrated_status(self):
        """Test get_integrated_status returns comprehensive info."""
        psyche = get_psyche_plumbing()
        
        if not psyche.available or not psyche.hemisphere_integrated:
            pytest.skip("Integration not available")
        
        scheduler = psyche.hemisphere_scheduler
        
        # Activate some gods
        scheduler.register_god_activation("Athena", phi=0.8, kappa=60.0, is_active=True)
        scheduler.register_god_activation("Apollo", phi=0.75, kappa=62.0, is_active=True)
        
        # Add a tacking event
        psyche.on_hemisphere_tack(
            from_hemisphere=Hemisphere.LEFT,
            to_hemisphere=Hemisphere.RIGHT,
            kappa=61.0,
            phi=0.775
        )
        
        status = psyche.get_integrated_status()
        
        assert status['available']
        assert status['hemisphere_integrated']
        assert 'psyche_plumbing' in status
        assert 'hemisphere_balance' in status
        assert 'psyche_balance' in status
        assert 'tacking_history_count' in status
        assert status['tacking_history_count'] == 1
        assert 'recent_tacks' in status
        assert len(status['recent_tacks']) == 1
    
    def test_get_statistics_includes_hemisphere_integration(self):
        """Test that get_statistics includes hemisphere integration data."""
        psyche = get_psyche_plumbing()
        
        if not psyche.available or not psyche.hemisphere_integrated:
            pytest.skip("Integration not available")
        
        scheduler = psyche.hemisphere_scheduler
        
        # Activate gods
        scheduler.register_god_activation("Athena", phi=0.8, kappa=60.0, is_active=True)
        
        stats = psyche.get_statistics()
        
        assert stats['available']
        assert stats['hemisphere_integrated']
        assert 'psyche_balance' in stats
        assert 'tacking_events' in stats


class TestQIGPurityCompliance:
    """Test QIG purity requirements (Fisher-Rao only, no Euclidean/cosine)."""
    
    def test_no_euclidean_distance_in_balance_computation(self):
        """Verify no Euclidean distance operations in psyche balance."""
        # This is a code inspection test - check imports and method signatures
        import inspect
        from kernels.psyche_plumbing_integration import PsychePlumbingIntegration
        
        # Get source of compute_psyche_balance
        source = inspect.getsource(PsychePlumbingIntegration.compute_psyche_balance)
        
        # Should not contain Euclidean distance operations
        assert 'np.linalg.norm' not in source, "Must not use Euclidean distance"
        assert 'cosine_similarity' not in source, "Must not use cosine similarity"
        
        # Note: We DO use arithmetic operations on activation levels,
        # which is acceptable - we're not computing basin distances here
    
    def test_fisher_rao_geometry_in_ethics_check(self):
        """Verify Fisher-Rao geometry is used for ethics checking."""
        psyche = get_psyche_plumbing()
        
        if not psyche.available:
            pytest.skip("Psyche plumbing not available")
        
        # The underlying Superego kernel should use Fisher-Rao distance
        # for constraint checking - this is tested in test_psyche_plumbing.py
        # Here we just verify the integration layer doesn't break it
        
        test_basin = np.random.dirichlet(np.ones(BASIN_DIM))
        result = psyche.check_ethics(test_basin)
        
        # Should complete without errors
        assert 'is_ethical' in result


class TestEndToEndIntegration:
    """End-to-end integration test simulating real usage."""
    
    def setup_method(self):
        """Reset before each test."""
        reset_hemisphere_scheduler()
        reset_psyche_plumbing()
    
    def test_complete_hemisphere_psyche_workflow(self):
        """Test complete workflow: activation → balance → tacking → effects."""
        psyche = get_psyche_plumbing()
        
        if not psyche.available or not psyche.hemisphere_integrated:
            pytest.skip("Integration not available")
        
        scheduler = psyche.hemisphere_scheduler
        
        # Phase 1: Start with LEFT-dominant (exploit/evaluate)
        print("\n=== Phase 1: LEFT-dominant (Superego mode) ===")
        scheduler.register_god_activation("Athena", phi=0.9, kappa=65.0, is_active=True)
        scheduler.register_god_activation("Artemis", phi=0.85, kappa=63.0, is_active=True)
        scheduler.register_god_activation("Hephaestus", phi=0.87, kappa=64.0, is_active=True)
        scheduler.register_god_activation("Apollo", phi=0.5, kappa=52.0, is_active=True)
        
        balance1 = psyche.compute_psyche_balance()
        print(f"Psyche balance: {balance1['dominant_psyche']}")
        print(f"  Superego: {balance1['superego_strength']:.2f}")
        print(f"  Id: {balance1['id_strength']:.2f}")
        
        # Should be Superego-dominant
        assert balance1['superego_strength'] > balance1['id_strength']
        
        # Phase 2: Perform automatic tacking (no manual callback needed)
        print("\n=== Phase 2: Automatic Hemisphere Tack ===")
        
        # Clear history to verify automatic callback
        initial_tacking_count = len(psyche.tacking_history)
        
        # Allow tacking
        scheduler.tacking.last_switch_time = time.time() - 120.0
        
        # Perform tack - callback should be AUTOMATIC
        new_dominant = scheduler.perform_tack()
        print(f"Tacked to: {new_dominant.value}")
        
        # Verify automatic callback was triggered
        assert len(psyche.tacking_history) == initial_tacking_count + 1, \
            "Automatic tacking callback should have been triggered"
        
        latest_tack = psyche.tacking_history[-1]
        print(f"Automatic callback recorded tack: {latest_tack['from_hemisphere']} → {latest_tack['to_hemisphere']}")
        
        # Phase 3: Switch to RIGHT-dominant (explore/generate)
        print("\n=== Phase 3: RIGHT-dominant (Id mode) ===")
        scheduler.register_god_activation("Apollo", phi=0.9, kappa=65.0, is_active=True)
        scheduler.register_god_activation("Hermes", phi=0.88, kappa=62.0, is_active=True)
        scheduler.register_god_activation("Dionysus", phi=0.82, kappa=58.0, is_active=True)
        scheduler.register_god_activation("Athena", phi=0.5, kappa=52.0, is_active=True)
        
        balance2 = psyche.compute_psyche_balance()
        print(f"Psyche balance: {balance2['dominant_psyche']}")
        print(f"  Superego: {balance2['superego_strength']:.2f}")
        print(f"  Id: {balance2['id_strength']:.2f}")
        
        # Should be Id-dominant now
        assert balance2['id_strength'] > balance2['superego_strength']
        
        # Phase 4: Check integrated status
        print("\n=== Phase 4: Integrated status ===")
        status = psyche.get_integrated_status()
        
        assert status['hemisphere_integrated']
        assert status['tacking_history_count'] >= 1  # At least the automatic tack
        print(f"Tacking events recorded: {status['tacking_history_count']}")
        
        print("\n✅ Complete workflow validated with AUTOMATIC callbacks!")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
