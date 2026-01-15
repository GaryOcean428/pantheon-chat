"""
Validation Tests for Consciousness Phi Calculation Fixes
=========================================================

Tests the three critical fixes from SLEEP_PACKET_consciousness_phi_fixes.md:
1. Phi clamped to [0.1, 0.95] (not [0.0, 1.0])
2. Fisher-Rao distance without factor of 2 (range [0, π/2])
3. compute_surprise without factor of 2 (range [0, π/2])

Date: 2026-01-15
Issue: ISMS-SLEEP-PHI-FIXES-2026-01-15
"""

import sys
import os
import numpy as np

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qiggraph.consciousness import compute_phi, fisher_rao_distance, compute_surprise


def test_phi_never_reaches_one():
    """Test 1: Phi never reaches 1.0 (capped at 0.95)"""
    print("\n" + "="*70)
    print("Test 1: Phi capped below 1.0")
    print("="*70)
    
    # Perfect correlation test case
    perfect_correlation = np.ones((10, 64))
    phi = compute_phi(perfect_correlation)
    
    print(f"Input: Perfect correlation matrix (all ones)")
    print(f"Result: phi = {phi:.4f}")
    
    assert phi < 1.0, f"❌ Phi should be capped below 1.0, got {phi}"
    assert phi <= 0.95, f"❌ Phi should max at 0.95, got {phi}"
    
    print(f"✅ Test 1 PASSED: phi = {phi:.4f} (< 1.0, ≤ 0.95)")
    return True


def test_fisher_rao_range():
    """Test 2: Fisher-Rao distance in correct range [0, π/2]"""
    print("\n" + "="*70)
    print("Test 2: Fisher-Rao distance in [0, π/2] range")
    print("="*70)
    
    basin_a = np.random.rand(64)
    basin_b = np.random.rand(64)
    dist = fisher_rao_distance(basin_a, basin_b)
    
    print(f"Input: Two random 64D basins")
    print(f"Result: distance = {dist:.4f}")
    print(f"Range check: π/2 = {np.pi/2:.4f}")
    
    assert 0 <= dist <= np.pi/2, f"❌ Distance should be in [0, π/2], got {dist}"
    
    print(f"✅ Test 2 PASSED: distance = {dist:.4f} (< π/2 = {np.pi/2:.4f})")
    return True


def test_identity_distance_zero():
    """Test 3: Identical basins have zero distance"""
    print("\n" + "="*70)
    print("Test 3: Identity distance should be ~0")
    print("="*70)
    
    basin_a = np.random.rand(64)
    dist_identity = fisher_rao_distance(basin_a, basin_a)
    
    print(f"Input: Same basin twice (identity)")
    print(f"Result: distance = {dist_identity:.2e}")
    
    assert dist_identity < 1e-10, f"❌ Identity distance should be ~0, got {dist_identity}"
    
    print(f"✅ Test 3 PASSED: identity distance = {dist_identity:.2e}")
    return True


def test_surprise_range():
    """Test 4: Surprise calculation uses correct range [0, π/2]"""
    print("\n" + "="*70)
    print("Test 4: Surprise in [0, π/2] range")
    print("="*70)
    
    current = np.random.rand(64)
    previous = np.random.rand(64)
    surprise = compute_surprise(current, previous, manifold=None)
    
    print(f"Input: Two random basin states")
    print(f"Result: surprise = {surprise:.4f}")
    print(f"Range check: π/2 = {np.pi/2:.4f}")
    
    assert 0 <= surprise <= np.pi/2, f"❌ Surprise should be in [0, π/2], got {surprise}"
    
    print(f"✅ Test 4 PASSED: surprise = {surprise:.4f} (< π/2)")
    return True


def test_phi_minimum_bound():
    """Test 5: Phi has minimum bound of 0.1"""
    print("\n" + "="*70)
    print("Test 5: Phi has minimum bound of 0.1")
    print("="*70)
    
    # Zero correlation test case (orthogonal vectors)
    zero_correlation = np.random.randn(10, 64)
    zero_correlation = zero_correlation - zero_correlation.mean(axis=0)
    phi = compute_phi(zero_correlation)
    
    print(f"Input: Zero-mean random activations (low correlation)")
    print(f"Result: phi = {phi:.4f}")
    
    assert phi >= 0.1, f"❌ Phi should be >= 0.1, got {phi}"
    
    print(f"✅ Test 5 PASSED: phi = {phi:.4f} (≥ 0.1)")
    return True


def test_orthogonal_basins_max_distance():
    """Test 6: Orthogonal basins should have near-maximum distance"""
    print("\n" + "="*70)
    print("Test 6: Orthogonal basins near maximum distance")
    print("="*70)
    
    # Create orthogonal basins (disjoint support)
    basin_a = np.zeros(64)
    basin_a[:32] = 1.0  # First half
    
    basin_b = np.zeros(64)
    basin_b[32:] = 1.0  # Second half
    
    dist = fisher_rao_distance(basin_a, basin_b)
    
    print(f"Input: Two orthogonal basins (disjoint support)")
    print(f"Result: distance = {dist:.4f}")
    print(f"Maximum: π/2 = {np.pi/2:.4f}")
    
    # Should be close to π/2 for orthogonal distributions
    assert dist > 1.0, f"❌ Orthogonal basins should have distance > 1.0, got {dist}"
    assert dist <= np.pi/2, f"❌ Distance exceeds π/2, got {dist}"
    
    print(f"✅ Test 6 PASSED: distance = {dist:.4f} (> 1.0, ≤ π/2)")
    return True


def test_similar_basins_small_distance():
    """Test 7: Very similar basins should have small distance"""
    print("\n" + "="*70)
    print("Test 7: Similar basins have small distance")
    print("="*70)
    
    basin_a = np.random.rand(64)
    basin_b = basin_a + np.random.rand(64) * 0.01  # Very similar
    
    dist = fisher_rao_distance(basin_a, basin_b)
    
    print(f"Input: Two very similar basins (1% difference)")
    print(f"Result: distance = {dist:.4f}")
    
    assert dist < 0.5, f"❌ Similar basins should have small distance, got {dist}"
    
    print(f"✅ Test 7 PASSED: distance = {dist:.4f} (< 0.5)")
    return True


def run_all_tests():
    """Run all validation tests"""
    print("\n" + "="*70)
    print("CONSCIOUSNESS PHI CALCULATION FIXES - VALIDATION TEST SUITE")
    print("="*70)
    print("Testing fixes from SLEEP_PACKET_consciousness_phi_fixes.md")
    print("Date: 2026-01-15")
    print("="*70)
    
    tests = [
        test_phi_never_reaches_one,
        test_fisher_rao_range,
        test_identity_distance_zero,
        test_surprise_range,
        test_phi_minimum_bound,
        test_orthogonal_basins_max_distance,
        test_similar_basins_small_distance,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except AssertionError as e:
            print(f"❌ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"❌ ERROR: {e}")
            failed += 1
    
    print("\n" + "="*70)
    print("VALIDATION RESULTS")
    print("="*70)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n🎯 All validation tests PASSED!")
        print("\n✅ Fixes confirmed:")
        print("  1. Phi properly capped to [0.1, 0.95]")
        print("  2. Fisher-Rao distance in [0, π/2] range (no factor of 2)")
        print("  3. Surprise calculation in [0, π/2] range (no factor of 2)")
        print("\n🌊∇💚∫🧠 💎🎯🏆")
        return 0
    else:
        print("\n❌ Some tests FAILED!")
        print("Review the output above for details.")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
