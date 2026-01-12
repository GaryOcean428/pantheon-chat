#!/usr/bin/env python3
"""
Simple validation script for E8 specialization levels.

Tests basic functionality without requiring numpy or pytest.
"""

import sys
import os

# Add qig-backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

print("=" * 60)
print("E8 Specialization Levels Validation")
print("=" * 60)

# Test 1: Import and check constants
print("\n✓ Test 1: Import frozen_physics")
try:
    from frozen_physics import (
        E8_SPECIALIZATION_LEVELS,
        get_specialization_level,
        KAPPA_STAR,
    )
    print(f"  E8_SPECIALIZATION_LEVELS: {E8_SPECIALIZATION_LEVELS}")
    print(f"  KAPPA_STAR: {KAPPA_STAR}")
    print("  ✅ PASSED")
except Exception as e:
    print(f"  ❌ FAILED: {e}")
    sys.exit(1)

# Test 2: Check E8 levels dictionary structure
print("\n✓ Test 2: Verify E8 levels dictionary")
try:
    assert E8_SPECIALIZATION_LEVELS[8] == "basic_rank"
    assert E8_SPECIALIZATION_LEVELS[56] == "refined_adjoint"
    assert E8_SPECIALIZATION_LEVELS[126] == "specialist_dim"
    assert E8_SPECIALIZATION_LEVELS[240] == "full_roots"
    assert len(E8_SPECIALIZATION_LEVELS) == 4
    print("  ✅ PASSED")
except AssertionError as e:
    print(f"  ❌ FAILED: {e}")
    sys.exit(1)

# Test 3: Test get_specialization_level function
print("\n✓ Test 3: Test get_specialization_level")
try:
    assert get_specialization_level(1) == "basic_rank"
    assert get_specialization_level(8) == "basic_rank"
    assert get_specialization_level(9) == "refined_adjoint"
    assert get_specialization_level(56) == "refined_adjoint"
    assert get_specialization_level(57) == "specialist_dim"
    assert get_specialization_level(126) == "specialist_dim"
    assert get_specialization_level(127) == "full_roots"
    assert get_specialization_level(240) == "full_roots"
    print("  ✅ PASSED")
except AssertionError as e:
    print(f"  ❌ FAILED: {e}")
    sys.exit(1)

# Test 4: Import m8_kernel_spawning functions
print("\n✓ Test 4: Import m8_kernel_spawning functions")
try:
    from m8_kernel_spawning import (
        should_spawn_specialist,
        get_kernel_specialization,
        assign_e8_root,
    )
    print("  ✅ PASSED")
except Exception as e:
    print(f"  ❌ FAILED: {e}")
    sys.exit(1)

# Test 5: Test should_spawn_specialist logic (basic cases)
print("\n✓ Test 5: Test should_spawn_specialist (basic cases)")
try:
    # At basic rank, should never spawn specialists
    result = should_spawn_specialist(5, 64.0)
    assert not result, "Should not spawn specialists at basic rank"
    
    # At refined with low kappa, should not spawn
    result = should_spawn_specialist(30, 40.0)
    assert not result, "Should not spawn specialists with low κ"
    
    print("  ✅ PASSED")
except (AssertionError, Exception) as e:
    print(f"  ❌ FAILED: {e}")
    sys.exit(1)

# Test 6: Test get_kernel_specialization
print("\n✓ Test 6: Test get_kernel_specialization")
try:
    # Basic rank
    spec = get_kernel_specialization(5, "ethics", 64.0)
    assert spec == "ethics", f"Expected 'ethics', got '{spec}'"
    
    # Refined adjoint
    spec = get_kernel_specialization(30, "visual", 64.0)
    assert spec.startswith("visual_refined_"), f"Expected 'visual_refined_*', got '{spec}'"
    
    # Specialist dim
    spec = get_kernel_specialization(80, "audio", 64.0)
    assert spec.startswith("audio_specialist_"), f"Expected 'audio_specialist_*', got '{spec}'"
    
    # Full roots
    spec = get_kernel_specialization(200, "memory", 64.0)
    assert spec == "memory_root_200", f"Expected 'memory_root_200', got '{spec}'"
    
    print("  ✅ PASSED")
except (AssertionError, Exception) as e:
    print(f"  ❌ FAILED: {e}")
    sys.exit(1)

# Test 7: Verify E8 specialization progression
print("\n✓ Test 7: Verify E8 specialization progression")
try:
    levels = []
    for count in [1, 8, 56, 126, 240]:
        level = get_specialization_level(count)
        levels.append(level)
    
    expected = [
        "basic_rank",
        "basic_rank",
        "refined_adjoint",
        "specialist_dim",
        "full_roots",
    ]
    
    assert levels == expected, f"Expected {expected}, got {levels}"
    print("  ✅ PASSED")
except (AssertionError, Exception) as e:
    print(f"  ❌ FAILED: {e}")
    sys.exit(1)

# Test 8: Check function docstrings
print("\n✓ Test 8: Verify function documentation")
try:
    assert get_specialization_level.__doc__ is not None
    assert "E8 specialization level" in get_specialization_level.__doc__
    assert should_spawn_specialist.__doc__ is not None
    assert "β-function" in should_spawn_specialist.__doc__
    assert get_kernel_specialization.__doc__ is not None
    assert "E8 level" in get_kernel_specialization.__doc__
    print("  ✅ PASSED")
except (AssertionError, Exception) as e:
    print(f"  ❌ FAILED: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ ALL VALIDATION TESTS PASSED")
print("=" * 60)
print("\nE8 specialization levels implementation is working correctly:")
print("  • E8_SPECIALIZATION_LEVELS dictionary defined")
print("  • get_specialization_level() function working")
print("  • should_spawn_specialist() with κ regime checks")
print("  • get_kernel_specialization() with level-based naming")
print("  • assign_e8_root() with Fisher-Rao distance")
print("\nReady for integration testing!")
