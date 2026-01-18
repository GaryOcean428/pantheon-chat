"""
Test Kernel Lifecycle Operations
================================

Basic validation tests for lifecycle operations.

Authority: E8 Protocol v4.0, WP5.3
Status: ACTIVE
Created: 2026-01-18
"""

import numpy as np
from kernel_lifecycle import (
    Kernel,
    KernelLifecycleManager,
    compute_frechet_mean_simplex,
    compute_fisher_distance,
    split_basin_coordinates,
)
from kernel_spawner import RoleSpec
from lifecycle_policy import LifecyclePolicyEngine


def test_frechet_mean():
    """Test Fréchet mean computation on simplex."""
    print("\n=== Test Fréchet Mean ===")
    
    # Create two basins
    basin1 = np.ones(64) / 64  # Uniform
    basin2 = np.zeros(64)
    basin2[0] = 1.0  # Concentrated
    
    # Compute Fréchet mean
    mean = compute_frechet_mean_simplex([basin1, basin2])
    
    # Validate properties
    assert len(mean) == 64
    assert np.all(mean >= 0), "Mean should be non-negative"
    assert np.abs(np.sum(mean) - 1.0) < 1e-6, "Mean should sum to 1"
    
    # Check it's between the two basins
    d1 = compute_fisher_distance(mean, basin1)
    d2 = compute_fisher_distance(mean, basin2)
    d12 = compute_fisher_distance(basin1, basin2)
    
    assert d1 < d12, "Mean should be closer to basin1 than basin1 is to basin2"
    assert d2 < d12, "Mean should be closer to basin2 than basin1 is to basin2"
    
    print(f"✅ Fréchet mean computed correctly")
    print(f"   Mean sum: {np.sum(mean):.6f}")
    print(f"   Distance to basin1: {d1:.4f}")
    print(f"   Distance to basin2: {d2:.4f}")
    print(f"   Distance basin1-basin2: {d12:.4f}")


def test_fisher_distance():
    """Test Fisher-Rao distance computation."""
    print("\n=== Test Fisher-Rao Distance ===")
    
    # Test identical basins
    basin = np.ones(64) / 64
    d = compute_fisher_distance(basin, basin)
    assert d < 1e-6, "Distance to self should be near zero"
    print(f"✅ Distance to self: {d:.8f}")
    
    # Test maximum distance
    basin1 = np.zeros(64)
    basin1[0] = 1.0
    basin2 = np.zeros(64)
    basin2[-1] = 1.0
    d = compute_fisher_distance(basin1, basin2)
    assert d > 1.0, "Distance between opposite corners should be large"
    print(f"✅ Distance between opposite corners: {d:.4f}")


def test_split_basin():
    """Test basin splitting."""
    print("\n=== Test Basin Splitting ===")
    
    basin = np.ones(64) / 64
    
    # Test domain split
    b1, b2 = split_basin_coordinates(basin, "domain")
    
    # Validate simplex properties
    assert np.all(b1 >= 0) and np.all(b2 >= 0), "Split basins should be non-negative"
    assert np.abs(np.sum(b1) - 1.0) < 1e-6, "Basin 1 should sum to 1"
    assert np.abs(np.sum(b2) - 1.0) < 1e-6, "Basin 2 should sum to 1"
    
    # Check they're different
    d = compute_fisher_distance(b1, b2)
    assert d > 0.1, "Split basins should be distinct"
    
    print(f"✅ Split basins valid")
    print(f"   Basin1 sum: {np.sum(b1):.6f}")
    print(f"   Basin2 sum: {np.sum(b2):.6f}")
    print(f"   Distance between splits: {d:.4f}")


def test_spawn():
    """Test kernel spawning."""
    print("\n=== Test Kernel Spawning ===")
    
    manager = KernelLifecycleManager()
    
    # Spawn a god kernel
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
    
    print(f"✅ Spawned kernel: {kernel.name}")
    print(f"   ID: {kernel.kernel_id}")
    print(f"   Type: {kernel.kernel_type}")
    print(f"   Stage: {kernel.lifecycle_stage}")
    print(f"   Φ: {kernel.phi:.3f}")


def test_split():
    """Test kernel splitting."""
    print("\n=== Test Kernel Splitting ===")
    
    manager = KernelLifecycleManager()
    
    # Create a kernel to split
    kernel = Kernel(
        kernel_id="test_kernel",
        name="TestKernel",
        kernel_type="god",
        basin_coords=np.ones(64) / 64,
        lifecycle_stage="active",  # Must be active (not protected)
        domains=["synthesis", "foresight"],
    )
    manager.active_kernels[kernel.kernel_id] = kernel
    
    # Split it
    k1, k2 = manager.split(kernel, "domain")
    
    assert k1.kernel_id != k2.kernel_id
    assert k1.name != k2.name
    assert kernel.lifecycle_stage == "split"
    assert len(kernel.child_kernels) == 2
    
    print(f"✅ Split kernel into:")
    print(f"   {k1.name} (Φ={k1.phi:.3f})")
    print(f"   {k2.name} (Φ={k2.phi:.3f})")


def test_merge():
    """Test kernel merging."""
    print("\n=== Test Kernel Merging ===")
    
    manager = KernelLifecycleManager()
    
    # Create two kernels to merge
    kernel1 = Kernel(
        kernel_id="test_kernel1",
        name="TestKernel1",
        kernel_type="god",
        basin_coords=np.ones(64) / 64 * 0.9,
        lifecycle_stage="active",
        domains=["synthesis"],
        total_cycles=100,
        success_count=80,
    )
    kernel2 = Kernel(
        kernel_id="test_kernel2",
        name="TestKernel2",
        kernel_type="god",
        basin_coords=np.ones(64) / 64 * 1.1,
        lifecycle_stage="active",
        domains=["foresight"],
        total_cycles=50,
        success_count=40,
    )
    manager.active_kernels[kernel1.kernel_id] = kernel1
    manager.active_kernels[kernel2.kernel_id] = kernel2
    
    # Merge them
    merged = manager.merge(kernel1, kernel2, "testing_merge")
    
    assert merged.kernel_id not in [kernel1.kernel_id, kernel2.kernel_id]
    assert set(merged.domains) == {"synthesis", "foresight"}
    assert merged.total_cycles == 150
    assert merged.success_count == 120
    assert np.abs(np.sum(merged.basin_coords) - 1.0) < 1e-6
    
    print(f"✅ Merged into: {merged.name}")
    print(f"   Domains: {merged.domains}")
    print(f"   Total cycles: {merged.total_cycles}")
    print(f"   Success count: {merged.success_count}")
    print(f"   Φ: {merged.phi:.3f}")


def test_prune_and_resurrect():
    """Test pruning and resurrection."""
    print("\n=== Test Prune & Resurrect ===")
    
    manager = KernelLifecycleManager()
    
    # Create a kernel to prune
    kernel = Kernel(
        kernel_id="test_kernel",
        name="TestKernel",
        kernel_type="chaos",
        basin_coords=np.ones(64) / 64,
        lifecycle_stage="active",
        phi=0.05,  # Low phi
        total_cycles=200,
        success_count=10,
        failure_count=190,
    )
    manager.active_kernels[kernel.kernel_id] = kernel
    
    # Prune it
    shadow = manager.prune(kernel, "persistent_low_phi")
    
    assert shadow.shadow_id is not None
    assert shadow.original_kernel_id == kernel.kernel_id
    assert shadow.final_phi == 0.05
    assert kernel.lifecycle_stage == "pruned"
    assert kernel.kernel_id not in manager.active_kernels
    
    print(f"✅ Pruned kernel to shadow: {shadow.shadow_id}")
    print(f"   Final Φ: {shadow.final_phi:.3f}")
    print(f"   Failure patterns: {shadow.failure_patterns}")
    
    # Resurrect it
    resurrected = manager.resurrect(shadow, "capability_needed")
    
    assert resurrected.kernel_id != kernel.kernel_id
    assert resurrected.lifecycle_stage == "active"
    assert resurrected.protection_cycles_remaining == 25  # Partial protection
    assert resurrected.phi > shadow.final_phi  # Should start with improved phi
    
    print(f"✅ Resurrected kernel: {resurrected.name}")
    print(f"   New Φ: {resurrected.phi:.3f}")
    print(f"   Protection cycles: {resurrected.protection_cycles_remaining}")


def test_promote():
    """Test chaos kernel promotion."""
    print("\n=== Test Promotion ===")
    
    manager = KernelLifecycleManager()
    
    # Create a successful chaos kernel
    chaos = Kernel(
        kernel_id="test_chaos",
        name="chaos_synthesis_1",
        kernel_type="chaos",
        basin_coords=np.ones(64) / 64,
        lifecycle_stage="active",
        phi=0.65,  # High stable phi
        total_cycles=100,
        success_count=80,
        domains=["synthesis"],
    )
    manager.active_kernels[chaos.kernel_id] = chaos
    
    # Promote it
    god = manager.promote(chaos, "Prometheus")
    
    assert god.kernel_type == "god"
    assert god.god_name == "Prometheus"
    assert god.name == "Prometheus"
    assert god.phi == chaos.phi
    assert chaos.lifecycle_stage == "promoted"
    
    print(f"✅ Promoted chaos kernel to: {god.name}")
    print(f"   Type: {god.kernel_type}")
    print(f"   Φ: {god.phi:.3f}")
    print(f"   Total cycles: {god.total_cycles}")


def test_policy_engine():
    """Test policy engine evaluation."""
    print("\n=== Test Policy Engine ===")
    
    manager = KernelLifecycleManager()
    engine = LifecyclePolicyEngine(manager)
    
    # Create kernels with different characteristics
    
    # Low phi kernel (should trigger prune)
    low_phi = Kernel(
        kernel_id="low_phi",
        name="LowPhiKernel",
        kernel_type="chaos",
        basin_coords=np.ones(64) / 64,
        lifecycle_stage="active",
        phi=0.05,
        total_cycles=150,
    )
    manager.active_kernels[low_phi.kernel_id] = low_phi
    
    # High phi kernel with multiple domains (should trigger split)
    high_phi = Kernel(
        kernel_id="high_phi",
        name="HighPhiKernel",
        kernel_type="god",
        basin_coords=np.ones(64) / 64,
        lifecycle_stage="active",
        phi=0.75,
        domains=["synthesis", "foresight", "strategy"],
        coupled_kernels=["k1", "k2", "k3", "k4", "k5", "k6"],
    )
    manager.active_kernels[high_phi.kernel_id] = high_phi
    
    # Successful chaos kernel (should trigger promote)
    good_chaos = Kernel(
        kernel_id="good_chaos",
        name="chaos_good_1",
        kernel_type="chaos",
        basin_coords=np.ones(64) / 64,
        lifecycle_stage="active",
        phi=0.55,
        total_cycles=100,
        success_count=80,
        domains=["synthesis"],
    )
    manager.active_kernels[good_chaos.kernel_id] = good_chaos
    
    # Evaluate all kernels
    actions = engine.evaluate_all_kernels()
    
    print(f"✅ Policy engine evaluated {len(manager.active_kernels)} kernels")
    print(f"   Triggered {len(actions)} actions:")
    for action in actions:
        print(f"   - {action['policy_name']}: {action['kernel_name']}")
    
    assert len(actions) > 0, "Should trigger at least one action"


def test_lifecycle_stats():
    """Test lifecycle statistics."""
    print("\n=== Test Lifecycle Stats ===")
    
    manager = KernelLifecycleManager()
    
    # Create some kernels
    for i in range(3):
        kernel = Kernel(
            kernel_id=f"kernel_{i}",
            name=f"TestKernel{i}",
            kernel_type="god" if i % 2 == 0 else "chaos",
            basin_coords=np.ones(64) / 64,
            lifecycle_stage="protected" if i == 0 else "active",
        )
        manager.active_kernels[kernel.kernel_id] = kernel
    
    # Get stats
    stats = manager.get_lifecycle_stats()
    
    assert stats['active_kernels'] == 3
    assert stats['god_count'] == 2
    assert stats['chaos_count'] == 1
    assert stats['protected_count'] == 1
    
    print(f"✅ Lifecycle stats:")
    print(f"   Active kernels: {stats['active_kernels']}")
    print(f"   Gods: {stats['god_count']}, Chaos: {stats['chaos_count']}")
    print(f"   Protected: {stats['protected_count']}")


def main():
    """Run all tests."""
    print("=" * 60)
    print("KERNEL LIFECYCLE OPERATIONS TEST SUITE")
    print("=" * 60)
    
    try:
        test_frechet_mean()
        test_fisher_distance()
        test_split_basin()
        test_spawn()
        test_split()
        test_merge()
        test_prune_and_resurrect()
        test_promote()
        test_policy_engine()
        test_lifecycle_stats()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n❌ TEST ERROR: {e}")
        raise


if __name__ == '__main__':
    main()
