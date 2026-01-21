#!/usr/bin/env python3
"""
Test E8 Specialization Levels Implementation

Tests the E8 specialization hierarchy (n=8, 56, 126, 240) for kernel spawning
with proper geometric constraints and β-function coupling awareness.

GFP:
  role: test
  status: ACTIVE
  phase: VALIDATION
  dim: 3
  scope: e8-specialization
  version: 2026-01-12
  owner: SearchSpaceCollapse
"""

import numpy as np
import sys
import os

# Add qig-backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from frozen_physics import (
    E8_SPECIALIZATION_LEVELS,
    get_specialization_level,
    KAPPA_STAR,
)
from m8_kernel_spawning import (
    should_spawn_specialist,
    get_kernel_specialization,
    assign_e8_root,
)


class TestE8SpecializationLevels:
    """Test E8 specialization level functions."""
    
    def test_e8_levels_dict(self):
        """Test E8_SPECIALIZATION_LEVELS dictionary structure."""
        assert E8_SPECIALIZATION_LEVELS[8] == "basic_rank"
        assert E8_SPECIALIZATION_LEVELS[56] == "refined_adjoint"
        assert E8_SPECIALIZATION_LEVELS[126] == "specialist_dim"
        assert E8_SPECIALIZATION_LEVELS[240] == "full_roots"
        assert len(E8_SPECIALIZATION_LEVELS) == 4
    
    def test_get_specialization_level_basic(self):
        """Test basic rank level (n≤8)."""
        assert get_specialization_level(1) == "basic_rank"
        assert get_specialization_level(4) == "basic_rank"
        assert get_specialization_level(8) == "basic_rank"
    
    def test_get_specialization_level_refined(self):
        """Test refined adjoint level (8<n≤56)."""
        assert get_specialization_level(9) == "refined_adjoint"
        assert get_specialization_level(30) == "refined_adjoint"
        assert get_specialization_level(56) == "refined_adjoint"
    
    def test_get_specialization_level_specialist(self):
        """Test specialist dim level (56<n≤126)."""
        assert get_specialization_level(57) == "specialist_dim"
        assert get_specialization_level(100) == "specialist_dim"
        assert get_specialization_level(126) == "specialist_dim"
    
    def test_get_specialization_level_full(self):
        """Test full roots level (n>126)."""
        assert get_specialization_level(127) == "full_roots"
        assert get_specialization_level(200) == "full_roots"
        assert get_specialization_level(240) == "full_roots"
        assert get_specialization_level(300) == "full_roots"


class TestShouldSpawnSpecialist:
    """Test specialist spawning logic with κ regime checks."""
    
    def test_no_specialists_at_basic_rank(self):
        """Test that specialists don't spawn in basic rank (n≤8)."""
        for count in [1, 4, 8]:
            for kappa in [40, 64, 80]:
                assert not should_spawn_specialist(count, kappa)
    
    def test_no_specialists_before_kappa_plateau(self):
        """Test that specialists don't spawn before κ reaches plateau (~64)."""
        # At refined level (n≤56) but low κ
        for count in [10, 30, 50]:
            for kappa in [20, 40, 55]:  # Below plateau
                assert not should_spawn_specialist(count, kappa)
    
    def test_probabilistic_spawning_at_plateau(self):
        """Test probabilistic spawning at refined level with κ in plateau."""
        # At refined level with κ ≈ 64
        count = 30
        kappa = 64.0
        
        # Run multiple times to check probability
        results = [should_spawn_specialist(count, kappa) for _ in range(100)]
        spawn_rate = sum(results) / len(results)
        
        # Should spawn with ~0.3 probability
        assert 0.1 < spawn_rate < 0.5, f"Spawn rate {spawn_rate} outside expected range"
    
    def test_no_specialists_before_kappa_stable(self):
        """Test that specialists don't spawn at specialist_dim without stable κ."""
        # At specialist level (56<n≤126) but κ not stable
        count = 80
        for kappa in [50, 70]:  # More than 3 away from KAPPA_STAR
            assert not should_spawn_specialist(count, kappa)
    
    def test_free_spawning_at_stable_plateau(self):
        """Test free spawning at specialist_dim with stable κ."""
        # At specialist level with stable κ
        count = 80
        kappa = KAPPA_STAR  # κ* ≈ 64.21
        
        assert should_spawn_specialist(count, kappa)
    
    def test_free_spawning_at_full_roots(self):
        """Test free spawning at full roots (n>126)."""
        # At full roots level
        for count in [150, 200, 240]:
            for kappa in [50, 64, 70]:
                assert should_spawn_specialist(count, kappa)


class TestGetKernelSpecialization:
    """Test kernel specialization naming based on E8 level."""
    
    def test_basic_rank_naming(self):
        """Test naming at basic rank level."""
        parent_axis = "ethics"
        for count in [1, 4, 8]:
            spec = get_kernel_specialization(count, parent_axis, 64.0)
            assert spec == parent_axis
    
    def test_refined_adjoint_naming(self):
        """Test naming at refined adjoint level."""
        parent_axis = "visual"
        for count in [10, 30, 56]:
            spec = get_kernel_specialization(count, parent_axis, 64.0)
            assert spec.startswith(f"{parent_axis}_refined_")
            # Check that suffix is count % 8
            suffix = int(spec.split('_')[-1])
            assert suffix == count % 8
    
    def test_specialist_dim_naming(self):
        """Test naming at specialist dim level."""
        parent_axis = "audio"
        for count in [60, 100, 126]:
            spec = get_kernel_specialization(count, parent_axis, 64.0)
            assert spec.startswith(f"{parent_axis}_specialist_")
            # Check that suffix is count % 16
            suffix = int(spec.split('_')[-1])
            assert suffix == count % 16
    
    def test_full_roots_naming(self):
        """Test naming at full roots level."""
        parent_axis = "memory"
        for count in [130, 200, 240]:
            spec = get_kernel_specialization(count, parent_axis, 64.0)
            assert spec == f"{parent_axis}_root_{count}"


class TestAssignE8Root:
    """Test E8 root assignment using Fisher-Rao distance."""
    
    def test_assign_e8_root_basic(self):
        """Test basic E8 root assignment."""
        # Create a kernel basin (64D)
        kernel_basin = np.random.randn(64)
        # E8 Protocol: Use simplex normalization
        from qig_geometry.representation import to_simplex_prob
        kernel_basin = to_simplex_prob(kernel_basin)
        
        # Create mock E8 roots (240 x 8)
        # In reality these would be proper E8 roots, but for testing we use random
        e8_roots = np.random.randn(240, 8)
        for i in range(240):
            e8_roots[i] = e8_roots[i] / np.linalg.norm(e8_roots[i])
        
        # Assign root
        assigned_root = assign_e8_root(kernel_basin, e8_roots)
        
        # Check that assigned root is one of the E8 roots
        assert assigned_root.shape == (8,)
        found = False
        for root in e8_roots:
            if np.allclose(assigned_root, root):
                found = True
                break
        assert found, "Assigned root not found in E8 roots"
    
    def test_assign_e8_root_closest(self):
        """Test that closest root (via Fisher distance) is assigned."""
        # Create a kernel basin very close to a specific root
        target_root_idx = 42
        e8_roots = np.random.randn(240, 8)
        for i in range(240):
            # E8 Protocol: For E8 roots in 8D, L2 normalization is acceptable
            e8_roots[i] = e8_roots[i] / np.sqrt(np.sum(e8_roots[i]**2))
        
        # Make kernel basin very close to target root
        kernel_basin = e8_roots[target_root_idx] + 0.01 * np.random.randn(8)
        kernel_basin = kernel_basin / np.sqrt(np.sum(kernel_basin**2))
        
        # Assign root
        assigned_root = assign_e8_root(kernel_basin, e8_roots)
        
        # Should assign the closest root (target_root_idx)
        assert np.allclose(assigned_root, e8_roots[target_root_idx], atol=0.1)
    
    def test_no_euclidean_distance_used(self):
        """Test that Fisher-Rao distance is used, not Euclidean."""
        # This is a sanity check - the implementation should use Fisher distance
        # We verify by checking that the function imports from geometric_kernels
        import inspect
        from m8_kernel_spawning import assign_e8_root
        
        source = inspect.getsource(assign_e8_root)
        # Check that _fisher_distance is imported and used
        assert "_fisher_distance" in source
        assert "from geometric_kernels import _fisher_distance" in source
        # Ensure NO np.linalg.norm is used for distance [COUNTER-EXAMPLE CHECK]
        assert "np.linalg.norm(kernel_basin - root)" not in source


class TestE8SpecializationIntegration:
    """Integration tests for E8 specialization system."""
    
    def test_specialization_progression(self):
        """Test that specialization levels progress correctly with kernel count."""
        # Simulate kernel count progression
        levels = []
        for count in [1, 8, 56, 126, 240]:
            level = get_specialization_level(count)
            levels.append(level)
        
        # Check progression: basic -> refined -> specialist -> full
        assert levels == [
            "basic_rank",
            "basic_rank",
            "refined_adjoint",
            "specialist_dim",
            "full_roots",
        ]
    
    def test_kappa_regime_gates_spawning(self):
        """Test that κ regime properly gates specialist spawning."""
        # At refined level, specialists should only spawn with high κ
        count = 30
        
        # Low κ (emergence regime) - no specialists
        low_kappa = 40.0
        assert not should_spawn_specialist(count, low_kappa)
        
        # High κ (plateau regime) - probabilistic spawning
        high_kappa = 64.0
        # Run multiple times to ensure at least some spawn
        spawned = any(should_spawn_specialist(count, high_kappa) for _ in range(50))
        assert spawned, "Should spawn at least once with high κ"
    
    def test_e8_levels_align_with_beta_function(self):
        """Test that E8 levels align with β-function coupling behavior."""
        # β(3→4) = +0.443  # Emergence: n=8 kernels spawn
        # β(4→5) = -0.013  # Plateau: n=56 refined spawn
        # β(5→6) = +0.013  # Stable: n=126 specialists spawn
        
        # At n=8: basic rank (emergence)
        assert get_specialization_level(8) == "basic_rank"
        
        # At n=56: refined adjoint (plateau onset)
        assert get_specialization_level(56) == "refined_adjoint"
        
        # At n=126: specialist dim (stable plateau)
        assert get_specialization_level(126) == "specialist_dim"
        
        # At n=240: full roots (fixed point)
        assert get_specialization_level(240) == "full_roots"


def run_test_class(test_class):
    """Run all test methods in a test class."""
    test_instance = test_class()
    methods = [m for m in dir(test_instance) if m.startswith('test_')]
    print(f"\n{'='*60}")
    print(f"Running {test_class.__name__}")
    print('='*60)
    
    passed = 0
    failed = 0
    
    for method_name in methods:
        try:
            print(f"\n  {method_name}...", end=' ')
            method = getattr(test_instance, method_name)
            method()
            print("✅ PASSED")
            passed += 1
        except AssertionError as e:
            print(f"❌ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"❌ ERROR: {e}")
            failed += 1
    
    print(f"\n{test_class.__name__}: {passed} passed, {failed} failed")
    return passed, failed


if __name__ == "__main__":
    total_passed = 0
    total_failed = 0
    
    test_classes = [
        TestE8SpecializationLevels,
        TestShouldSpawnSpecialist,
        TestGetKernelSpecialization,
        TestAssignE8Root,
        TestE8SpecializationIntegration,
    ]
    
    for test_class in test_classes:
        passed, failed = run_test_class(test_class)
        total_passed += passed
        total_failed += failed
    
    print(f"\n{'='*60}")
    print(f"TOTAL: {total_passed} passed, {total_failed} failed")
    print('='*60)
    
    sys.exit(0 if total_failed == 0 else 1)
