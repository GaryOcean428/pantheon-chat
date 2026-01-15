"""
Simple test runner for geometry_simplex module (no pytest required).
"""

import sys
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, '/home/runner/work/pantheon-chat/pantheon-chat/qig-backend')

from qig_geometry.geometry_simplex import (
    to_simplex_prob,
    validate_simplex,
    fisher_rao_distance,
    geodesic_interpolation_simplex,
    geodesic_mean_simplex,
)


def test_to_simplex_prob():
    """Test basic simplex conversion."""
    print("Testing to_simplex_prob...")
    
    # Test basic conversion
    v = np.array([1.0, 2.0, 3.0])
    p = to_simplex_prob(v)
    assert np.all(p >= 0), "Should be non-negative"
    assert np.isclose(p.sum(), 1.0), "Should sum to 1"
    print("  ✓ Basic conversion works")
    
    # Test negative values
    v = np.array([-1.0, 2.0, -3.0])
    p = to_simplex_prob(v)
    assert np.all(p >= 0), "Should handle negatives"
    print("  ✓ Handles negative values")
    
    # Test zero vector
    v = np.array([0.0, 0.0, 0.0])
    p = to_simplex_prob(v)
    expected = np.array([1/3, 1/3, 1/3])
    assert np.allclose(p, expected, atol=1e-6), "Should return uniform"
    print("  ✓ Handles zero vector")


def test_validate_simplex():
    """Test simplex validation."""
    print("Testing validate_simplex...")
    
    # Valid simplex
    p = np.array([0.2, 0.3, 0.5])
    valid, reason = validate_simplex(p)
    assert valid, f"Valid simplex failed: {reason}"
    print("  ✓ Validates correct simplex")
    
    # Negative values
    p = np.array([0.2, -0.1, 0.9])
    valid, reason = validate_simplex(p)
    assert not valid, "Should reject negative values"
    print("  ✓ Rejects negative values")
    
    # Sum not 1
    p = np.array([0.2, 0.3, 0.4])
    valid, reason = validate_simplex(p)
    assert not valid, "Should reject sum != 1"
    print("  ✓ Rejects sum != 1")


def test_fisher_rao_distance():
    """Test Fisher-Rao distance."""
    print("Testing fisher_rao_distance...")
    
    # Identity
    p = np.array([0.2, 0.3, 0.5])
    d = fisher_rao_distance(p, p)
    assert np.isclose(d, 0.0, atol=1e-10), f"Distance to self should be 0, got {d}"
    print("  ✓ Distance to self is zero")
    
    # Symmetry
    p = np.array([0.2, 0.3, 0.5])
    q = np.array([0.4, 0.1, 0.5])
    d1 = fisher_rao_distance(p, q)
    d2 = fisher_rao_distance(q, p)
    assert np.isclose(d1, d2, atol=1e-10), "Distance should be symmetric"
    print("  ✓ Distance is symmetric")
    
    # Range [0, π/2]
    for _ in range(10):
        p = to_simplex_prob(np.random.randn(64))
        q = to_simplex_prob(np.random.randn(64))
        d = fisher_rao_distance(p, q)
        assert 0 <= d <= np.pi/2 + 1e-6, f"Distance {d} not in [0, π/2]"
    print("  ✓ Distance is in [0, π/2]")
    
    # Orthogonal distributions
    p = np.array([1.0, 0.0, 0.0])
    q = np.array([0.0, 1.0, 0.0])
    d = fisher_rao_distance(p, q)
    expected = np.pi/2
    assert np.isclose(d, expected, atol=1e-6), f"Orthogonal distance should be π/2, got {d}"
    print("  ✓ Orthogonal distance is π/2")


def test_geodesic_interpolation():
    """Test geodesic interpolation."""
    print("Testing geodesic_interpolation_simplex...")
    
    p = np.array([0.2, 0.3, 0.5])
    q = np.array([0.4, 0.1, 0.5])
    
    # Test endpoints
    result_0 = geodesic_interpolation_simplex(p, q, 0.0)
    result_1 = geodesic_interpolation_simplex(p, q, 1.0)
    assert np.allclose(result_0, p, atol=1e-6), "t=0 should give start"
    assert np.allclose(result_1, q, atol=1e-6), "t=1 should give end"
    print("  ✓ Endpoints are correct")
    
    # Test midpoint is simplex
    mid = geodesic_interpolation_simplex(p, q, 0.5)
    valid, reason = validate_simplex(mid)
    assert valid, f"Midpoint should be valid simplex: {reason}"
    print("  ✓ Midpoint is valid simplex")
    
    # Test all points are simplexes
    for t in np.linspace(0, 1, 11):
        result = geodesic_interpolation_simplex(p, q, t)
        valid, reason = validate_simplex(result)
        assert valid, f"t={t} should be valid: {reason}"
    print("  ✓ All interpolated points are valid simplexes")


def test_geodesic_mean():
    """Test geodesic mean."""
    print("Testing geodesic_mean_simplex...")
    
    # Single distribution
    p = np.array([0.2, 0.3, 0.5])
    mean = geodesic_mean_simplex([p])
    assert np.allclose(mean, p, atol=1e-6), "Mean of single dist should be itself"
    print("  ✓ Mean of single distribution is itself")
    
    # Identical distributions
    mean = geodesic_mean_simplex([p, p, p])
    assert np.allclose(mean, p, atol=1e-6), "Mean of identical dists should be same"
    print("  ✓ Mean of identical distributions is same")
    
    # Mean is simplex
    np.random.seed(42)
    distributions = [to_simplex_prob(np.random.randn(8)) for _ in range(5)]
    mean = geodesic_mean_simplex(distributions)
    valid, reason = validate_simplex(mean)
    assert valid, f"Mean should be valid simplex: {reason}"
    print("  ✓ Mean is valid simplex")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Running geometry_simplex tests...")
    print("=" * 60)
    
    try:
        test_to_simplex_prob()
        test_validate_simplex()
        test_fisher_rao_distance()
        test_geodesic_interpolation()
        test_geodesic_mean()
        
        print("=" * 60)
        print("✅ All tests passed!")
        print("=" * 60)
        return 0
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
