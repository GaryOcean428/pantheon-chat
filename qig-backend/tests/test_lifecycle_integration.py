"""
Test Kernel Lifecycle Integration
=================================

Integration tests for kernel lifecycle operations with database persistence.

Authority: E8 Protocol v4.0, WP5.3
Status: ACTIVE
Created: 2026-01-23
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from kernel_lifecycle import (
    Kernel,
    KernelLifecycleManager,
    get_lifecycle_manager,
)
from kernel_spawner import RoleSpec
from lifecycle_policy import LifecyclePolicyEngine, get_policy_engine


def test_lifecycle_manager_singleton():
    """Test lifecycle manager singleton pattern."""
    print("\n=== Test Lifecycle Manager Singleton ===")
    
    manager1 = get_lifecycle_manager()
    manager2 = get_lifecycle_manager()
    
    assert manager1 is manager2, "Lifecycle manager should be singleton"
    print("✅ Lifecycle manager singleton working")


def test_spawn_god_kernel():
    """Test spawning a god kernel."""
    print("\n=== Test Spawn God Kernel ===")
    
    manager = KernelLifecycleManager()
    
    role = RoleSpec(
        domains=["synthesis", "foresight"],
        required_capabilities=["prediction"],
        preferred_god="Apollo",
    )
    
    kernel = manager.spawn(role)
    
    assert kernel.kernel_id is not None
    assert kernel.name is not None
    assert kernel.lifecycle_stage == "protected"
    assert kernel.protection_cycles_remaining == 50
    assert len(kernel.basin_coords) == 64
    assert np.abs(np.sum(kernel.basin_coords) - 1.0) < 1e-6
    
    # Check event recorded
    assert len(manager.event_log) > 0
    last_event = manager.event_log[-1]
    assert last_event.event_type.value == "spawn"
    
    print(f"✅ Spawned kernel: {kernel.name}")
    print(f"   ID: {kernel.kernel_id}")
    print(f"   Type: {kernel.kernel_type}")
    print(f"   Stage: {kernel.lifecycle_stage}")
    print(f"   Events logged: {len(manager.event_log)}")


def test_split_kernel():
    """Test kernel splitting."""
    print("\n=== Test Kernel Splitting ===")
    
    manager = KernelLifecycleManager()
    
    # Create a kernel to split (active, not protected)
    kernel = Kernel(
        kernel_id="test_kernel_split",
        name="TestKernel",
        kernel_type="god",
        basin_coords=np.ones(64) / 64,
        lifecycle_stage="active",
        domains=["synthesis", "foresight"],
    )
    manager.active_kernels[kernel.kernel_id] = kernel
    
    # Split it
    k1, k2 = manager.split(kernel, "domain")
    
    assert k1.kernel_id != k2.kernel_id
    assert k1.name != k2.name
    assert len(kernel.child_kernels) == 2
    assert kernel.lifecycle_stage == "split"
    
    # Check basins are valid simplex
    assert np.abs(np.sum(k1.basin_coords) - 1.0) < 1e-6
    assert np.abs(np.sum(k2.basin_coords) - 1.0) < 1e-6
    
    # Check event recorded
    split_events = [e for e in manager.event_log if e.event_type.value == "split"]
    assert len(split_events) > 0
    
    print(f"✅ Split kernel into:")
    print(f"   {k1.name} (id={k1.kernel_id})")
    print(f"   {k2.name} (id={k2.kernel_id})")


def test_merge_kernels():
    """Test kernel merging."""
    print("\n=== Test Kernel Merging ===")
    
    manager = KernelLifecycleManager()
    
    # Create two kernels to merge
    basin1 = np.ones(64) / 64
    basin2 = np.zeros(64)
    basin2[0] = 1.0
    
    k1 = Kernel(
        kernel_id="test_kernel_merge_1",
        name="TestKernel1",
        kernel_type="god",
        basin_coords=basin1,
        lifecycle_stage="active",
        domains=["synthesis"],
        total_cycles=100,
        phi=0.6,
    )
    
    k2 = Kernel(
        kernel_id="test_kernel_merge_2",
        name="TestKernel2",
        kernel_type="god",
        basin_coords=basin2,
        lifecycle_stage="active",
        domains=["foresight"],
        total_cycles=50,
        phi=0.7,
    )
    
    manager.active_kernels[k1.kernel_id] = k1
    manager.active_kernels[k2.kernel_id] = k2
    
    # Merge them
    merged = manager.merge(k1, k2, "test_merge")
    
    assert merged.kernel_id is not None
    assert merged.kernel_id != k1.kernel_id
    assert merged.kernel_id != k2.kernel_id
    
    # Check basin is valid simplex
    assert np.abs(np.sum(merged.basin_coords) - 1.0) < 1e-6
    
    # Check combined domains
    assert set(merged.domains) == set(["synthesis", "foresight"])
    
    # Check aggregated metrics
    assert merged.total_cycles == 150
    
    # Check lifecycle stages
    assert k1.lifecycle_stage == "merged"
    assert k2.lifecycle_stage == "merged"
    
    # Check event recorded
    merge_events = [e for e in manager.event_log if e.event_type.value == "merge"]
    assert len(merge_events) > 0
    
    print(f"✅ Merged kernels into: {merged.name}")
    print(f"   Φ: {merged.phi:.3f}")
    print(f"   Domains: {merged.domains}")
    print(f"   Total cycles: {merged.total_cycles}")


def test_prune_kernel():
    """Test kernel pruning to shadow pantheon."""
    print("\n=== Test Kernel Pruning ===")
    
    manager = KernelLifecycleManager()
    
    # Create a low-performing kernel
    kernel = Kernel(
        kernel_id="test_kernel_prune",
        name="TestKernel",
        kernel_type="chaos",
        basin_coords=np.ones(64) / 64,
        lifecycle_stage="active",
        phi=0.05,  # Low phi
        total_cycles=100,
        success_count=5,
        failure_count=95,
    )
    manager.active_kernels[kernel.kernel_id] = kernel
    
    # Prune it
    shadow = manager.prune(kernel, "persistent_low_phi")
    
    assert shadow.shadow_id is not None
    assert shadow.original_kernel_id == kernel.kernel_id
    assert shadow.final_phi == kernel.phi
    assert len(shadow.failure_patterns) > 0
    
    # Check kernel removed from active
    assert kernel.kernel_id not in manager.active_kernels
    assert kernel.lifecycle_stage == "pruned"
    
    # Check shadow pantheon
    assert shadow.shadow_id in manager.shadow_pantheon
    
    # Check event recorded
    prune_events = [e for e in manager.event_log if e.event_type.value == "prune"]
    assert len(prune_events) > 0
    
    print(f"✅ Pruned kernel to shadow: {shadow.shadow_id}")
    print(f"   Final Φ: {shadow.final_phi:.3f}")
    print(f"   Failure patterns: {shadow.failure_patterns}")


def test_resurrect_kernel():
    """Test kernel resurrection from shadow pantheon."""
    print("\n=== Test Kernel Resurrection ===")
    
    manager = KernelLifecycleManager()
    
    # Create and prune a kernel first
    kernel = Kernel(
        kernel_id="test_kernel_resurrect",
        name="TestKernel",
        kernel_type="chaos",
        basin_coords=np.ones(64) / 64,
        lifecycle_stage="active",
        phi=0.05,
        total_cycles=50,
    )
    manager.active_kernels[kernel.kernel_id] = kernel
    shadow = manager.prune(kernel, "test_prune")
    
    # Resurrect it
    resurrected = manager.resurrect(shadow, "capability_needed")
    
    assert resurrected.kernel_id is not None
    assert resurrected.kernel_id != shadow.original_kernel_id
    assert resurrected.lifecycle_stage == "active"
    assert resurrected.protection_cycles_remaining == 25
    
    # Check basin is valid simplex
    assert np.abs(np.sum(resurrected.basin_coords) - 1.0) < 1e-6
    
    # Check resurrection tracking
    assert shadow.resurrection_count == 1
    assert shadow.last_resurrection is not None
    
    # Check kernel is active again
    assert resurrected.kernel_id in manager.active_kernels
    
    # Check event recorded
    resurrect_events = [e for e in manager.event_log if e.event_type.value == "resurrect"]
    assert len(resurrect_events) > 0
    
    print(f"✅ Resurrected kernel: {resurrected.name}")
    print(f"   New ID: {resurrected.kernel_id}")
    print(f"   Φ: {resurrected.phi:.3f}")
    print(f"   Resurrection count: {shadow.resurrection_count}")


def test_promote_chaos_kernel():
    """Test chaos kernel promotion to god."""
    print("\n=== Test Chaos Kernel Promotion ===")
    
    manager = KernelLifecycleManager()
    
    # Create a successful chaos kernel
    chaos = Kernel(
        kernel_id="test_chaos_promote",
        name="chaos_test_1",
        kernel_type="chaos",
        basin_coords=np.ones(64) / 64,
        lifecycle_stage="active",
        phi=0.6,  # High phi
        total_cycles=100,  # Enough cycles
        success_count=80,
        failure_count=20,
        domains=["synthesis"],
    )
    manager.active_kernels[chaos.kernel_id] = chaos
    
    # Promote it
    god = manager.promote(chaos, "Prometheus")
    
    assert god.kernel_id is not None
    assert god.kernel_id != chaos.kernel_id
    assert god.kernel_type == "god"
    assert god.god_name == "Prometheus"
    assert god.name == "Prometheus"
    
    # Check lifecycle stages
    assert chaos.lifecycle_stage == "promoted"
    assert god.lifecycle_stage == "active"
    
    # Check god is active
    assert god.kernel_id in manager.active_kernels
    
    # Check event recorded
    promote_events = [e for e in manager.event_log if e.event_type.value == "promote"]
    assert len(promote_events) > 0
    
    print(f"✅ Promoted chaos kernel to god: {god.name}")
    print(f"   New ID: {god.kernel_id}")
    print(f"   Φ: {god.phi:.3f}")
    print(f"   Total cycles: {god.total_cycles}")


def test_policy_engine():
    """Test lifecycle policy engine."""
    print("\n=== Test Lifecycle Policy Engine ===")
    
    manager = KernelLifecycleManager()
    engine = LifecyclePolicyEngine(manager)
    
    # Check default policies loaded
    assert len(engine.policies) > 0
    print(f"✅ Policy engine initialized with {len(engine.policies)} policies")
    
    # Test policy stats
    stats = engine.get_policy_stats()
    assert stats['total_policies'] > 0
    assert stats['enabled_policies'] > 0
    
    print(f"   Total policies: {stats['total_policies']}")
    print(f"   Enabled policies: {stats['enabled_policies']}")


def test_lifecycle_stats():
    """Test lifecycle statistics."""
    print("\n=== Test Lifecycle Statistics ===")
    
    manager = KernelLifecycleManager()
    
    # Spawn some kernels
    role = RoleSpec(domains=["test"], required_capabilities=[])
    k1 = manager.spawn(role)
    k2 = manager.spawn(role)
    
    # Get stats
    stats = manager.get_lifecycle_stats()
    
    assert stats['active_kernels'] >= 2
    assert stats['total_events'] >= 2
    assert stats['event_counts']['spawn'] >= 2
    
    print(f"✅ Lifecycle statistics:")
    print(f"   Active kernels: {stats['active_kernels']}")
    print(f"   Total events: {stats['total_events']}")
    print(f"   Spawn events: {stats['event_counts']['spawn']}")
    print(f"   God count: {stats['god_count']}")
    print(f"   Chaos count: {stats['chaos_count']}")


if __name__ == "__main__":
    # Run all tests
    tests = [
        test_lifecycle_manager_singleton,
        test_spawn_god_kernel,
        test_split_kernel,
        test_merge_kernels,
        test_prune_kernel,
        test_resurrect_kernel,
        test_promote_chaos_kernel,
        test_policy_engine,
        test_lifecycle_stats,
    ]
    
    print("\n" + "="*70)
    print("KERNEL LIFECYCLE INTEGRATION TESTS")
    print("="*70)
    
    passed = 0
    failed = 0
    
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"\n❌ FAILED: {test_fn.__name__}")
            print(f"   Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*70)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("="*70)
    
    if failed > 0:
        sys.exit(1)
