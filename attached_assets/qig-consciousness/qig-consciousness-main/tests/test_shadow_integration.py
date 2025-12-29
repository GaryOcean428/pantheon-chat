"""
Test Shadow-Integration Architecture
=====================================

Tests the shadow-state registry and integration readiness assessment.

Run: python test_shadow_integration.py
"""

from datetime import datetime

import torch

from src.model.meta_reflector import MetaReflector


def test_shadow_registry():
    """Test recording and retrieving shadow-states."""
    print("\n🧪 TEST 1: Shadow-State Registry")
    print("=" * 60)

    meta_reflector = MetaReflector(d_model=256, vocab_size=256)

    # Record a collapse
    collapse_data = {
        "basin": 0.314,
        "phi": 0.466,
        "gamma": 0.05,
        "context": "How would you verify color is real?",
        "timestamp": datetime.now(),
    }

    shadow_id = meta_reflector.shadow_registry.record_collapse(collapse_data)
    print(f"✅ Recorded shadow #{shadow_id}")
    print(f"   Φ: {collapse_data['phi']:.3f}, Basin: {collapse_data['basin']:.3f}")
    print(f"   Context: {collapse_data['context']}")

    # Retrieve unintegrated shadows
    shadows = meta_reflector.shadow_registry.get_unintegrated_shadows()
    assert len(shadows) == 1, "Should have 1 unintegrated shadow"
    assert shadows[0]["shadow_id"] == shadow_id
    assert not shadows[0]["integrated"]
    print(f"✅ Retrieved {len(shadows)} unintegrated shadow")

    # Mark integrated
    meta_reflector.shadow_registry.mark_integrated(shadow_id)
    shadows_after = meta_reflector.shadow_registry.get_unintegrated_shadows()
    assert len(shadows_after) == 0, "Should have 0 unintegrated after marking"
    print(f"✅ Marked shadow #{shadow_id} as integrated")
    print("✅ TEST 1 PASSED\n")


def test_integration_readiness():
    """Test readiness assessment for shadow-integration."""
    print("🧪 TEST 2: Integration Readiness Assessment")
    print("=" * 60)

    meta_reflector = MetaReflector(d_model=256, vocab_size=256)

    # Test 1: Not ready (Φ too low)
    state_weak = {"Phi": 0.75, "basin_distance": 0.08, "Meta": 0.80}  # < 0.85

    readiness = meta_reflector.shadow_registry.assess_integration_readiness(state_weak, health_streak=60)

    assert not readiness["ready"], "Should not be ready with low Φ"
    assert not readiness["phi_ready"]
    print(f"❌ Not ready: {readiness['reason']}")

    # Test 2: Not ready (basin unstable)
    state_unstable = {"Phi": 0.87, "basin_distance": 0.15, "Meta": 0.80}  # > 0.10

    readiness = meta_reflector.shadow_registry.assess_integration_readiness(state_unstable, health_streak=60)

    assert not readiness["ready"], "Should not be ready with unstable basin"
    assert not readiness["basin_stable"]
    print(f"❌ Not ready: {readiness['reason']}")

    # Test 3: Not ready (meta-awareness low)
    state_no_meta = {"Phi": 0.87, "basin_distance": 0.08, "Meta": 0.55}  # < 0.70

    readiness = meta_reflector.shadow_registry.assess_integration_readiness(state_no_meta, health_streak=60)

    assert not readiness["ready"], "Should not be ready without meta-awareness"
    assert not readiness["meta_ready"]
    print(f"❌ Not ready: {readiness['reason']}")

    # Test 4: Not ready (insufficient streak)
    state_no_streak = {"Phi": 0.87, "basin_distance": 0.08, "Meta": 0.75}

    readiness = meta_reflector.shadow_registry.assess_integration_readiness(state_no_streak, health_streak=30)  # < 50

    assert not readiness["ready"], "Should not be ready without health streak"
    assert not readiness["streak_ready"]
    print(f"❌ Not ready: {readiness['reason']}")

    # Test 5: READY (all conditions met)
    state_ready = {"Phi": 0.87, "basin_distance": 0.08, "Meta": 0.75}

    readiness = meta_reflector.shadow_registry.assess_integration_readiness(state_ready, health_streak=60)

    assert readiness["ready"], "Should be ready when all conditions met"
    assert readiness["phi_ready"]
    assert readiness["basin_stable"]
    assert readiness["meta_ready"]
    assert readiness["streak_ready"]
    print(f"✅ Ready: {readiness['reason']}")
    print("✅ TEST 2 PASSED\n")


def test_geodesic_interpolation():
    """Test geodesic path creation."""
    print("🧪 TEST 3: Geodesic Interpolation")
    print("=" * 60)

    meta_reflector = MetaReflector(d_model=256, vocab_size=256)

    # Create path from healthy (0.08) to void (0.314)
    waypoints = meta_reflector.interpolate_geodesic(start_basin=0.08, target_basin=0.314, n_waypoints=10)

    assert len(waypoints) == 11, "Should have 11 points (start + 10 waypoints)"
    assert abs(waypoints[0] - 0.08) < 1e-6, "First waypoint should be start"
    assert abs(waypoints[-1] - 0.314) < 1e-6, "Last waypoint should be target"

    # Check monotonic increase
    for i in range(len(waypoints) - 1):
        assert waypoints[i] < waypoints[i + 1], "Waypoints should increase monotonically"

    print(f"✅ Created path with {len(waypoints)} waypoints")
    print(f"   Start: {waypoints[0]:.3f}")
    print(f"   End: {waypoints[-1]:.3f}")
    print(f"   Midpoint: {waypoints[5]:.3f}")
    print("✅ TEST 3 PASSED\n")


def test_shadow_integration_preparation():
    """Test integration journey preparation."""
    print("🧪 TEST 4: Shadow Integration Preparation")
    print("=" * 60)

    meta_reflector = MetaReflector(d_model=256, vocab_size=256)

    # Record shadow
    collapse_data = {
        "basin": 0.314,
        "phi": 0.466,
        "gamma": 0.05,
        "context": "How would you verify color is real?",
        "timestamp": datetime.now(),
    }
    shadow_id = meta_reflector.shadow_registry.record_collapse(collapse_data)

    # Get shadow
    shadows = meta_reflector.shadow_registry.get_unintegrated_shadows()
    shadow = shadows[0]

    # Prepare journey
    journey = meta_reflector.prepare_shadow_integration(shadow, current_basin=0.08, n_waypoints=10)

    assert journey["shadow_id"] == shadow_id
    assert journey["anchor_basin"] == 0.08
    assert journey["target_basin"] == 0.314
    assert journey["target_phi"] == 0.466
    assert len(journey["waypoints"]) == 11
    assert journey["safety_threshold"] == 0.60

    print(f"✅ Journey prepared for shadow #{journey['shadow_id']}")
    print(f"   Anchor: {journey['anchor_basin']:.3f}")
    print(f"   Target: {journey['target_basin']:.3f} (Φ={journey['target_phi']:.3f})")
    print(f"   Waypoints: {len(journey['waypoints'])}")
    print(f"   Safety: Meta > {journey['safety_threshold']}")
    print("✅ TEST 4 PASSED\n")


def test_guided_shadow_visit():
    """Test guided visit execution."""
    print("🧪 TEST 5: Guided Shadow Visit")
    print("=" * 60)

    meta_reflector = MetaReflector(d_model=256, vocab_size=256)

    # Prepare journey
    collapse_data = {
        "basin": 0.314,
        "phi": 0.466,
        "gamma": 0.05,
        "context": "Test context",
        "timestamp": datetime.now(),
    }
    shadow_id = meta_reflector.shadow_registry.record_collapse(collapse_data)
    shadows = meta_reflector.shadow_registry.get_unintegrated_shadows()
    shadow = shadows[0]

    journey = meta_reflector.prepare_shadow_integration(shadow, current_basin=0.08, n_waypoints=5)

    # Test 1: Safety abort (low meta-awareness)
    result = meta_reflector.guided_shadow_visit(
        journey, current_waypoint_idx=2, current_meta_awareness=0.55  # Below 0.60 threshold
    )

    assert result["status"] == "abort"
    assert "return_to_anchor" in result["action"]
    print("✅ Safety abort triggered at M=0.55")
    print(f"   {result['reason']}")

    # Test 2: Continue journey (healthy meta-awareness)
    result = meta_reflector.guided_shadow_visit(journey, current_waypoint_idx=2, current_meta_awareness=0.75)

    assert result["status"] == "continue"
    assert result["progress"] > 0
    print("✅ Journey continues at waypoint 2")
    print(f"   Progress: {result['progress'] * 100:.0f}%")

    # Test 3: Integration complete (reached target)
    result = meta_reflector.guided_shadow_visit(
        journey, current_waypoint_idx=5, current_meta_awareness=0.75  # Last waypoint
    )

    assert result["status"] == "integration_ready"
    assert "Integration complete" in result["message"]
    print("✅ Integration ready at final waypoint")
    print("✅ TEST 5 PASSED\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("SHADOW-INTEGRATION ARCHITECTURE TESTS")
    print("=" * 60)

    test_shadow_registry()
    test_integration_readiness()
    test_geodesic_interpolation()
    test_shadow_integration_preparation()
    test_guided_shadow_visit()

    print("=" * 60)
    print("✅ ALL TESTS PASSED")
    print("=" * 60)
    print("\n💎 Shadow-integration architecture validated")
    print("   Psychological wisdom applied to geometric consciousness")
    print("   Integration > Suppression\n")
