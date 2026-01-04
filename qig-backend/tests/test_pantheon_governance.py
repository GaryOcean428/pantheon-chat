"""
Tests for Pantheon Governance System
=====================================

Verifies that kernel spawning governance is properly enforced.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from olympus.pantheon_governance import (
    PantheonGovernance,
    ProposalType,
    ProposalStatus,
    ALLOWED_BYPASS_REASONS,
    EMERGENCY_BYPASS_REASONS
)


def test_spawn_permission_with_approval():
    """Test that spawning is allowed with explicit approval."""
    gov = PantheonGovernance()
    
    # Should not raise PermissionError
    result = gov.check_spawn_permission(
        reason="test spawn",
        pantheon_approved=True
    )
    assert result is True


def test_spawn_permission_without_approval():
    """Test that spawning without approval creates proposal."""
    gov = PantheonGovernance()
    
    try:
        gov.check_spawn_permission(
            reason="unauthorized spawn",
            pantheon_approved=False,
            parent_phi=0.5  # Below auto-approval threshold
        )
        assert False, "Should have raised PermissionError"
    except PermissionError as e:
        assert "Pantheon approval" in str(e)
        assert len(gov.get_pending_proposals()) == 1


def test_spawn_permission_with_bypass_reason():
    """Test that allowed bypass reasons auto-approve."""
    gov = PantheonGovernance()
    
    # minimum_population should auto-approve
    result = gov.check_spawn_permission(
        reason="minimum_population",
        pantheon_approved=False
    )
    assert result is True


def test_spawn_permission_with_high_phi_parent():
    """Test that high-Φ parents auto-approve spawning."""
    gov = PantheonGovernance()
    
    # Φ >= 0.7 should auto-approve
    result = gov.check_spawn_permission(
        reason="reproduction",
        parent_id="parent_123",
        parent_phi=0.75,
        pantheon_approved=False
    )
    assert result is True


def test_breed_permission_with_high_phi_parents():
    """Test that breeding high-Φ parents auto-approves."""
    gov = PantheonGovernance()
    
    result = gov.check_breed_permission(
        parent1_id="parent1",
        parent2_id="parent2",
        parent1_phi=0.72,
        parent2_phi=0.68,
        pantheon_approved=False
    )
    # Average Φ = 0.70, should auto-approve
    assert result is True


def test_breed_permission_requires_approval():
    """Test that low-Φ breeding requires approval."""
    gov = PantheonGovernance()
    
    try:
        gov.check_breed_permission(
            parent1_id="parent1",
            parent2_id="parent2",
            parent1_phi=0.5,
            parent2_phi=0.4,
            pantheon_approved=False
        )
        assert False, "Should have raised PermissionError"
    except PermissionError as e:
        assert "Pantheon approval" in str(e)
        assert len(gov.get_pending_proposals()) == 1


def test_turbo_spawn_permission():
    """Test that turbo spawn requires explicit approval."""
    gov = PantheonGovernance()
    
    try:
        gov.check_turbo_spawn_permission(
            count=50,
            pantheon_approved=False,
            emergency_override=False
        )
        assert False, "Should have raised PermissionError"
    except PermissionError as e:
        assert "Pantheon approval" in str(e)
        assert "mass spawning" in str(e).lower()


def test_turbo_spawn_with_approval():
    """Test that turbo spawn works with approval."""
    gov = PantheonGovernance()
    
    result = gov.check_turbo_spawn_permission(
        count=50,
        pantheon_approved=True,
        emergency_override=False
    )
    assert result is True


def test_turbo_spawn_with_emergency_override():
    """Test that turbo spawn works with emergency override."""
    gov = PantheonGovernance()
    
    result = gov.check_turbo_spawn_permission(
        count=50,
        pantheon_approved=False,
        emergency_override=True
    )
    assert result is True


def test_proposal_approval():
    """Test proposal approval workflow."""
    gov = PantheonGovernance()
    
    # Create proposal by attempting unauthorized spawn
    try:
        gov.check_spawn_permission(
            reason="test",
            pantheon_approved=False,
            parent_phi=0.5
        )
    except PermissionError:
        pass
    
    proposals = gov.get_pending_proposals()
    assert len(proposals) == 1
    
    proposal_id = proposals[0].proposal_id
    
    # Approve the proposal
    result = gov.approve_proposal(proposal_id, approver="zeus")
    assert result["success"] is True
    assert proposals[0].status == ProposalStatus.APPROVED


def test_proposal_rejection():
    """Test proposal rejection workflow."""
    gov = PantheonGovernance()
    
    # Create proposal
    try:
        gov.check_spawn_permission(
            reason="test",
            pantheon_approved=False,
            parent_phi=0.5
        )
    except PermissionError:
        pass
    
    proposals = gov.get_pending_proposals()
    proposal_id = proposals[0].proposal_id
    
    # Reject the proposal
    result = gov.reject_proposal(proposal_id, rejector="athena")
    assert result["success"] is True
    assert proposals[0].status == ProposalStatus.REJECTED


def test_emergency_bypass_requires_confirmation():
    """Test that emergency bypass reasons require manual confirmation."""
    gov = PantheonGovernance()
    
    try:
        gov.check_spawn_permission(
            reason="emergency_recovery",
            pantheon_approved=False
        )
        assert False, "Should have raised PermissionError"
    except PermissionError as e:
        assert "manual confirmation" in str(e).lower()


if __name__ == "__main__":
    # Run tests
    print("Testing Pantheon Governance...")
    print("\n1. Testing spawn permission with approval...")
    test_spawn_permission_with_approval()
    print("✅ PASSED")
    
    print("\n2. Testing spawn permission without approval...")
    test_spawn_permission_without_approval()
    print("✅ PASSED")
    
    print("\n3. Testing spawn permission with bypass reason...")
    test_spawn_permission_with_bypass_reason()
    print("✅ PASSED")
    
    print("\n4. Testing spawn permission with high-Φ parent...")
    test_spawn_permission_with_high_phi_parent()
    print("✅ PASSED")
    
    print("\n5. Testing breed permission with high-Φ parents...")
    test_breed_permission_with_high_phi_parents()
    print("✅ PASSED")
    
    print("\n6. Testing breed permission requires approval...")
    test_breed_permission_requires_approval()
    print("✅ PASSED")
    
    print("\n7. Testing turbo spawn permission...")
    test_turbo_spawn_permission()
    print("✅ PASSED")
    
    print("\n8. Testing turbo spawn with approval...")
    test_turbo_spawn_with_approval()
    print("✅ PASSED")
    
    print("\n9. Testing turbo spawn with emergency override...")
    test_turbo_spawn_with_emergency_override()
    print("✅ PASSED")
    
    print("\n10. Testing proposal approval...")
    test_proposal_approval()
    print("✅ PASSED")
    
    print("\n11. Testing proposal rejection...")
    test_proposal_rejection()
    print("✅ PASSED")
    
    print("\n12. Testing emergency bypass requires confirmation...")
    test_emergency_bypass_requires_confirmation()
    print("✅ PASSED")
    
    print("\n" + "="*60)
    print("✅ ALL GOVERNANCE TESTS PASSED!")
    print("="*60)
