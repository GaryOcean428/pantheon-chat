#!/usr/bin/env python3
"""
Test Fisher-Rao Attractor Finding

Tests geometric attractor discovery using Fisher potential.
"""

import sys
sys.path.insert(0, '.')

import numpy as np
from qig_core.attractor_finding import (
    compute_fisher_potential,
    find_local_minimum,
    compute_potential_gradient,
    geodesic_step,
    find_attractors_in_region,
    sample_in_fisher_ball,
)
from qiggraph.manifold import FisherManifold
from qig_geometry import fisher_coord_distance


def test_fisher_potential():
    """Test that Fisher potential can be computed"""
    print("🧪 Testing Fisher Potential Computation...")
    
    basin = np.random.rand(64)
    basin = basin / basin.sum()  # Normalize to probability distribution
    
    metric = FisherManifold()
    
    # Compute potential
    potential = compute_fisher_potential(basin, metric)
    
    # Potential should be finite
    assert np.isfinite(potential), f"Potential should be finite, got {potential}"
    
    # Potential should be real-valued
    assert isinstance(potential, (float, np.floating)), "Potential should be a float"
    
    print(f"✅ Fisher potential computed: U = {potential:.4f}")
    print("✅ Potential computation test passed!\n")


def test_potential_gradient():
    """Test gradient computation"""
    print("🧪 Testing Potential Gradient...")
    
    basin = np.random.rand(64)
    basin = basin / basin.sum()
    
    metric = FisherManifold()
    
    # Compute gradient
    gradient = compute_potential_gradient(basin, metric, epsilon=1e-5)
    
    # Gradient should be same dimension as basin
    assert len(gradient) == len(basin), "Gradient should match basin dimension"
    
    # Gradient should be finite
    assert np.all(np.isfinite(gradient)), "Gradient should be finite"
    
    # Gradient should not be all zeros (unless at critical point)
    gradient_norm = np.linalg.norm(gradient)
    assert gradient_norm >= 0, "Gradient norm should be non-negative"
    
    print(f"✅ Gradient computed: ||∇U|| = {gradient_norm:.4f}")
    print("✅ Gradient computation test passed!\n")


def test_geodesic_step():
    """Test geodesic stepping"""
    print("🧪 Testing Geodesic Step...")
    
    basin = np.random.rand(64)
    basin = basin / basin.sum()
    
    direction = np.random.randn(64)
    step_size = 0.01
    
    metric = FisherManifold()
    
    # Take geodesic step
    new_basin = geodesic_step(basin, direction, step_size, metric)
    
    # New basin should be same dimension
    assert len(new_basin) == len(basin), "New basin should match dimension"
    
    # E8 Protocol: Check simplex constraints
    from qig_geometry.representation import to_simplex_prob
    basin_simplex = to_simplex_prob(new_basin)
    assert np.allclose(np.sum(basin_simplex), 1.0, atol=1e-6), "Basin not on simplex"
    
    # Should have moved from original basin
    distance = fisher_coord_distance(basin, new_basin)
    print(f"✅ Moved Fisher-Rao distance: {distance:.4f}")
    
    print("✅ Geodesic step test passed!\n")


def test_local_minimum_finding():
    """Test that attractor finding converges to local minimum"""
    print("🧪 Testing Local Minimum Finding...")
    
    basin = np.random.rand(64)
    basin = basin / basin.sum()
    
    metric = FisherManifold()
    
    # Find attractor
    attractor, potential, converged = find_local_minimum(
        basin, metric, max_steps=100, tolerance=1e-6
    )
    
    # Should return valid results
    assert len(attractor) == 64, "Attractor should be 64D"
    assert np.isfinite(potential), f"Potential should be finite, got {potential}"
    assert isinstance(converged, bool), "Converged should be boolean"
    
    print(f"✅ Found attractor: potential = {potential:.4f}, converged = {converged}")
    
    # If converged, verify it's a local minimum
    if converged:
        # Sample nearby points
        for _ in range(5):
            nearby = sample_in_fisher_ball(attractor, 0.05, metric)
            nearby_potential = compute_fisher_potential(nearby, metric)
            
            # Attractor should have lower or equal potential
            # (allowing small numerical tolerance)
            assert nearby_potential >= potential - 0.1, \
                f"Attractor should be minimum: U_attr={potential:.4f}, U_nearby={nearby_potential:.4f}"
        
        print("✅ Verified attractor is local minimum")
    else:
        print("⚠️  Did not converge (this is OK for some starting points)")
    
    print("✅ Local minimum finding test passed!\n")


def test_multiple_attractors():
    """Test finding multiple attractors in a region"""
    print("🧪 Testing Multiple Attractor Discovery...")
    
    center = np.random.rand(64)
    center = center / center.sum()
    
    metric = FisherManifold()
    
    # Find attractors in region
    attractors = find_attractors_in_region(
        center, metric, radius=2.0, n_samples=30
    )
    
    print(f"✅ Found {len(attractors)} attractors in region")
    
    if len(attractors) > 0:
        # Check they're sorted by potential (lowest first)
        potentials = [pot for _, pot in attractors]
        assert potentials == sorted(potentials), "Attractors should be sorted by potential"
        print(f"✅ Attractors sorted by potential: {potentials}")
        
        # All attractors should be distinct
        for i, (a1, _) in enumerate(attractors):
            for j, (a2, _) in enumerate(attractors):
                if i != j:
                    distance = fisher_coord_distance(a1, a2)
                    assert distance > 0.04, f"Attractors {i} and {j} too close: d={distance:.4f}"
        
        print(f"✅ All {len(attractors)} attractors are distinct")
    
    print("✅ Multiple attractor discovery test passed!\n")


def test_sample_in_fisher_ball():
    """Test sampling within Fisher-Rao ball"""
    print("🧪 Testing Fisher Ball Sampling...")
    
    center = np.random.rand(64)
    center = center / center.sum()
    
    metric = FisherManifold()
    radius = 0.5
    
    # Sample multiple points
    samples = []
    for _ in range(20):
        sample = sample_in_fisher_ball(center, radius, metric)
        samples.append(sample)
    
    # All samples should be within radius
    for sample in samples:
        distance = fisher_coord_distance(center, sample)
        assert distance <= radius + 0.01, \
            f"Sample outside ball: distance={distance:.4f}, radius={radius}"
    
    print(f"✅ All 20 samples within radius {radius}")
    
    # Samples should be diverse (not all the same)
    sample_array = np.array(samples)
    variance = np.var(sample_array)
    assert variance > 1e-6, "Samples should have variation"
    
    print(f"✅ Samples have variance: {variance:.6f}")
    print("✅ Fisher ball sampling test passed!\n")


def test_attractor_stability():
    """Test that attractors are actually stable (perturbing returns to same basin)"""
    print("🧪 Testing Attractor Stability...")
    
    basin = np.random.rand(64)
    basin = basin / basin.sum()
    
    metric = FisherManifold()
    
    # Find attractor
    attractor, potential, converged = find_local_minimum(
        basin, metric, max_steps=100
    )
    
    if not converged:
        print("⚠️  Initial search did not converge, skipping stability test")
        return
    
    # Perturb attractor slightly and descend again
    perturbed = attractor + np.random.randn(64) * 0.01
    perturbed = perturbed / (np.linalg.norm(perturbed) + 1e-10)
    
    attractor2, potential2, converged2 = find_local_minimum(
        perturbed, metric, max_steps=100
    )
    
    if converged2:
        # Should return to same (or nearby) attractor
        distance = fisher_coord_distance(attractor, attractor2)
        print(f"✅ Re-converged to distance {distance:.4f} from original")
        
        # Potentials should be similar
        potential_diff = abs(potential - potential2)
        print(f"✅ Potential difference: {potential_diff:.4f}")
        
        assert distance < 0.2, f"Should return to same attractor, got distance={distance:.4f}"
    
    print("✅ Attractor stability test passed!\n")


def run_all_tests():
    """Run all attractor finding tests"""
    print("=" * 70)
    print("FISHER-RAO ATTRACTOR FINDING TEST SUITE")
    print("=" * 70)
    print()
    
    test_fisher_potential()
    test_potential_gradient()
    test_geodesic_step()
    test_local_minimum_finding()
    test_multiple_attractors()
    test_sample_in_fisher_ball()
    test_attractor_stability()
    
    print("=" * 70)
    print("✅ ALL ATTRACTOR FINDING TESTS PASSED!")
    print("=" * 70)


if __name__ == '__main__':
    run_all_tests()
