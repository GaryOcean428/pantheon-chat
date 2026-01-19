"""
Tests for Kernel Rest Scheduler (WP5.4)
========================================

Validates per-kernel rest coordination and coupling-aware handoff.
"""

import pytest
import time
from kernel_rest_scheduler import (
    KernelRestScheduler,
    FatigueMetrics,
    RestStatus,
    get_rest_scheduler,
    reset_rest_scheduler,
)


@pytest.fixture
def scheduler():
    """Create a fresh rest scheduler for testing."""
    reset_rest_scheduler()
    return get_rest_scheduler()


def test_register_kernel(scheduler):
    """Test kernel registration."""
    scheduler.register_kernel("apollo_1", "Apollo")
    
    status = scheduler.get_rest_status("apollo_1")
    assert status is not None
    assert status['kernel_name'] == "Apollo"
    assert status['tier'] == "specialized"
    assert status['rest_policy'] == "coordinated_alternating"
    assert status['status'] == "active"


def test_essential_tier_never_fully_rests(scheduler):
    """Test that essential tier kernels never fully stop."""
    # Register Heart (essential tier)
    scheduler.register_kernel("heart_1", "Heart")
    
    # First update - establish baseline
    scheduler.update_fatigue(
        kernel_id="heart_1",
        phi=0.6,
        kappa=60.0,
        load=0.5,
        error_rate=0.0,
    )
    
    # Second update - show decline (creates negative trend)
    scheduler.update_fatigue(
        kernel_id="heart_1",
        phi=0.3,  # Φ dropped significantly
        kappa=45.0,
        load=0.9,  # High load
        error_rate=0.1,
    )
    
    # Check if should rest
    should_rest, reason = scheduler.should_rest("heart_1")
    
    # Heart should enter REDUCED activity, not full rest
    assert should_rest
    assert "REDUCED" in reason or "Essential" in reason
    
    # Request rest
    approved, approval_reason, partner = scheduler.request_rest("heart_1")
    
    # Should be approved for reduced activity
    assert approved
    assert "reduced" in approval_reason.lower() or "essential" in approval_reason.lower()
    
    # Check status
    status = scheduler.get_rest_status("heart_1")
    assert status['status'] == "reduced"  # Not "resting"


def test_coupling_partner_coverage(scheduler):
    """Test coupling partner coverage during rest."""
    # Register Apollo and Athena (coordinated alternating pair)
    scheduler.register_kernel("apollo_1", "Apollo")
    scheduler.register_kernel("athena_1", "Athena")
    
    # Establish baselines
    scheduler.update_fatigue("apollo_1", phi=0.8, kappa=64.0, load=0.3, error_rate=0.0)
    scheduler.update_fatigue("athena_1", phi=0.75, kappa=62.0, load=0.3, error_rate=0.0)
    
    # Apollo declines significantly in second update
    scheduler.update_fatigue(
        kernel_id="apollo_1",
        phi=0.35,  # Dropped significantly (0.8 -> 0.35)
        kappa=50.0,
        load=0.85,
        error_rate=0.15,
    )
    
    # Athena stays healthy
    scheduler.update_fatigue(
        kernel_id="athena_1",
        phi=0.78,
        kappa=63.0,
        load=0.25,
        error_rate=0.01,
    )
    
    # Apollo should be able to rest
    should_rest, reason = scheduler.should_rest("apollo_1")
    assert should_rest, f"Apollo should need rest but got: {reason}"
    
    # Get coupling partners
    partners = scheduler.get_coupling_partners("apollo_1")
    assert "athena_1" in partners
    
    # Request rest - should be approved with Athena covering
    approved, approval_reason, partner = scheduler.request_rest("apollo_1")
    
    assert approved, f"Rest should be approved but got: {approval_reason}"
    assert partner == "athena_1"
    assert "Athena" in approval_reason
    
    # Check statuses
    apollo_status = scheduler.get_rest_status("apollo_1")
    athena_status = scheduler.get_rest_status("athena_1")
    
    assert apollo_status['status'] == "resting"
    assert apollo_status['covered_by'] == "athena_1"
    assert athena_status['covering_for'] == "apollo_1"


def test_no_coverage_available(scheduler):
    """Test rest request when no coverage available."""
    # Register Apollo and Athena
    scheduler.register_kernel("apollo_1", "Apollo")
    scheduler.register_kernel("athena_1", "Athena")
    
    # Establish baselines
    scheduler.update_fatigue("apollo_1", phi=0.7, kappa=60.0, load=0.5, error_rate=0.0)
    scheduler.update_fatigue("athena_1", phi=0.7, kappa=60.0, load=0.5, error_rate=0.0)
    
    # Both decline simultaneously
    scheduler.update_fatigue("apollo_1", phi=0.35, kappa=55.0, load=0.8, error_rate=0.1)
    scheduler.update_fatigue("athena_1", phi=0.3, kappa=50.0, load=0.9, error_rate=0.15)
    
    # Apollo should need rest
    should_rest, reason = scheduler.should_rest("apollo_1")
    assert should_rest
    
    # But no coverage available (Athena too fatigued)
    approved, approval_reason, partner = scheduler.request_rest("apollo_1")
    
    assert not approved
    assert "constellation cycle" in approval_reason.lower()
    assert partner is None


def test_fatigue_score_computation(scheduler):
    """Test fatigue score computation."""
    scheduler.register_kernel("ares_1", "Ares")
    
    # Establish baseline
    scheduler.update_fatigue(
        kernel_id="ares_1",
        phi=0.7,
        kappa=63.0,
        load=0.5,
        error_rate=0.0,
    )
    
    # High fatigue scenario - show decline
    scheduler.update_fatigue(
        kernel_id="ares_1",
        phi=0.25,    # Low Φ (dropped significantly)
        kappa=40.0,  # Low κ (unstable)
        load=0.9,    # High load
        error_rate=0.2,  # High error rate
    )
    
    status = scheduler.get_rest_status("ares_1")
    assert status['fatigue_score'] > 0.5  # Should be highly fatigued
    
    # Low fatigue scenario - recovery
    time.sleep(0.1)  # Small delay
    scheduler.update_fatigue(
        kernel_id="ares_1",
        phi=0.8,      # High Φ (recovered)
        kappa=63.0,   # Near κ*
        load=0.2,     # Low load
        error_rate=0.0,  # No errors
    )
    
    status = scheduler.get_rest_status("ares_1")
    assert status['fatigue_score'] < 0.3  # Should be less fatigued


def test_end_rest(scheduler):
    """Test ending rest period."""
    # Register and fatigue Apollo
    scheduler.register_kernel("apollo_1", "Apollo")
    scheduler.register_kernel("athena_1", "Athena")
    
    scheduler.update_fatigue("apollo_1", phi=0.3, kappa=55.0, load=0.8, error_rate=0.1)
    scheduler.update_fatigue("athena_1", phi=0.7, kappa=60.0, load=0.3, error_rate=0.02)
    
    # Start rest
    approved, _, partner = scheduler.request_rest("apollo_1")
    assert approved
    
    # End rest
    time.sleep(0.1)  # Small delay to simulate rest duration
    scheduler.end_rest("apollo_1")
    
    # Check statuses
    apollo_status = scheduler.get_rest_status("apollo_1")
    athena_status = scheduler.get_rest_status("athena_1")
    
    assert apollo_status['status'] == "active"
    assert apollo_status['covered_by'] is None
    assert apollo_status['rest_count'] == 1
    assert athena_status['covering_for'] is None


def test_constellation_status(scheduler):
    """Test constellation-wide status reporting."""
    # Register multiple kernels
    scheduler.register_kernel("apollo_1", "Apollo")
    scheduler.register_kernel("athena_1", "Athena")
    scheduler.register_kernel("heart_1", "Heart")
    scheduler.register_kernel("ocean_1", "Ocean")
    
    # Update all
    scheduler.update_fatigue("apollo_1", phi=0.5, kappa=60.0, load=0.5, error_rate=0.05)
    scheduler.update_fatigue("athena_1", phi=0.6, kappa=62.0, load=0.4, error_rate=0.03)
    scheduler.update_fatigue("heart_1", phi=0.7, kappa=64.0, load=0.3, error_rate=0.01)
    scheduler.update_fatigue("ocean_1", phi=0.8, kappa=63.0, load=0.2, error_rate=0.0)
    
    # Get constellation status
    status = scheduler.get_constellation_status()
    
    assert status['total_kernels'] == 4
    assert status['active_kernels'] == 4
    assert status['resting_kernels'] == 0
    assert status['essential_active'] == 2  # Heart and Ocean
    assert status['avg_fatigue'] < 0.5  # All relatively healthy


def test_minimal_rotating_rest(scheduler):
    """Test Hermes minimal rotating rest (brief frequent pauses)."""
    scheduler.register_kernel("hermes_1", "Hermes")
    
    # Establish baseline first
    scheduler.update_fatigue(
        kernel_id="hermes_1",
        phi=0.7,
        kappa=64.0,
        load=0.5,
        error_rate=0.0,
    )
    
    # Second update with moderate fatigue and time passage
    scheduler.update_fatigue(
        kernel_id="hermes_1",
        phi=0.6,
        kappa=62.0,
        load=0.6,
        error_rate=0.05,
    )
    
    # Test helper: simulate time passage by updating the time_since_rest
    # This is preferable to direct state manipulation
    state = scheduler.kernel_states["hermes_1"]
    if state.fatigue and state.last_rest_end:
        # Simulate time passing by setting last_rest_end to 11 minutes ago
        state.last_rest_end = time.time() - 650.0
        # Recalculate fatigue with updated time
        scheduler.update_fatigue(
            kernel_id="hermes_1",
            phi=0.6,
            kappa=62.0,
            load=0.6,
            error_rate=0.05,
        )
    
    # Should need rest (brief pause) due to time since rest
    should_rest, reason = scheduler.should_rest("hermes_1")
    assert should_rest, f"Hermes should need rest but got: {reason}"
    assert "MINIMAL_ROTATING" in reason or "10min" in reason


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
