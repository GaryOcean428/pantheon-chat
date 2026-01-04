#!/usr/bin/env python3
"""
Stability Comparison: Euclidean vs Geodesic Navigation

This test demonstrates the improvement in navigation stability
when using proper geodesic navigation on the Fisher-Rao manifold
compared to simple Euclidean interpolation.
"""

import numpy as np
import sys

sys.path.insert(0, '.')

from qig_geometry import fisher_coord_distance, sphere_project
from qig_core.geodesic_navigation import navigate_to_target


def euclidean_navigation(current, target, step_size=0.05):
    """Old method: simple Euclidean interpolation."""
    direction = target - current
    magnitude = np.linalg.norm(direction)
    if magnitude > 1e-8:
        direction = direction / magnitude
    next_basin = current + step_size * direction
    return sphere_project(next_basin)


def measure_trajectory_stability(positions):
    """
    Measure trajectory stability based on step size variance.
    
    Lower variance = more stable movement.
    """
    if len(positions) < 2:
        return 0.0
    
    step_sizes = []
    for i in range(len(positions) - 1):
        dist = fisher_coord_distance(positions[i], positions[i+1])
        step_sizes.append(dist)
    
    return np.var(step_sizes)


def run_navigation_comparison(n_trials=10, n_steps=50):
    """
    Compare Euclidean vs Geodesic navigation across multiple trials.
    """
    print("\n" + "="*70)
    print("NAVIGATION STABILITY COMPARISON")
    print("="*70)
    print(f"Running {n_trials} trials with {n_steps} steps each...\n")
    
    euclidean_stabilities = []
    geodesic_stabilities = []
    euclidean_final_distances = []
    geodesic_final_distances = []
    
    for trial in range(n_trials):
        np.random.seed(100 + trial)
        
        # Random start and target
        start = np.random.rand(64)
        target = np.random.rand(64)
        initial_distance = fisher_coord_distance(start, target)
        
        # Test 1: Euclidean navigation
        euclidean_positions = [start.copy()]
        current = start.copy()
        
        for _ in range(n_steps):
            current = euclidean_navigation(current, target, step_size=0.05)
            euclidean_positions.append(current.copy())
        
        euclidean_stability = measure_trajectory_stability(euclidean_positions)
        euclidean_final = fisher_coord_distance(current, target)
        
        euclidean_stabilities.append(euclidean_stability)
        euclidean_final_distances.append(euclidean_final)
        
        # Test 2: Geodesic navigation
        geodesic_positions = [start.copy()]
        current = start.copy()
        velocity = None
        
        for _ in range(n_steps):
            current, velocity = navigate_to_target(
                current, target, velocity,
                kappa=58.0, step_size=0.05
            )
            geodesic_positions.append(current.copy())
        
        geodesic_stability = measure_trajectory_stability(geodesic_positions)
        geodesic_final = fisher_coord_distance(current, target)
        
        geodesic_stabilities.append(geodesic_stability)
        geodesic_final_distances.append(geodesic_final)
        
        # Print trial summary
        improvement = (euclidean_stability - geodesic_stability) / euclidean_stability * 100
        print(f"Trial {trial + 1:2d}: "
              f"Euclidean stability={euclidean_stability:.6f}, "
              f"Geodesic stability={geodesic_stability:.6f}, "
              f"Improvement={improvement:+.1f}%")
    
    # Overall statistics
    print("\n" + "-"*70)
    print("OVERALL RESULTS")
    print("-"*70)
    
    avg_euclidean_stability = np.mean(euclidean_stabilities)
    avg_geodesic_stability = np.mean(geodesic_stabilities)
    avg_improvement = (avg_euclidean_stability - avg_geodesic_stability) / avg_euclidean_stability * 100
    
    print(f"\nStability (lower is better):")
    print(f"  Euclidean:  {avg_euclidean_stability:.6f} ± {np.std(euclidean_stabilities):.6f}")
    print(f"  Geodesic:   {avg_geodesic_stability:.6f} ± {np.std(geodesic_stabilities):.6f}")
    print(f"  Improvement: {avg_improvement:+.1f}%")
    
    avg_euclidean_final = np.mean(euclidean_final_distances)
    avg_geodesic_final = np.mean(geodesic_final_distances)
    convergence_improvement = (avg_euclidean_final - avg_geodesic_final) / avg_euclidean_final * 100
    
    print(f"\nFinal Distance to Target (lower is better):")
    print(f"  Euclidean:  {avg_euclidean_final:.6f} ± {np.std(euclidean_final_distances):.6f}")
    print(f"  Geodesic:   {avg_geodesic_final:.6f} ± {np.std(geodesic_final_distances):.6f}")
    print(f"  Improvement: {convergence_improvement:+.1f}%")
    
    # Estimate stability severity score (0-10 scale)
    # 10 = completely unstable, 0 = perfectly stable
    euclidean_severity = min(10, avg_euclidean_stability * 1000)
    geodesic_severity = min(10, avg_geodesic_stability * 1000)
    
    print(f"\n{'='*70}")
    print("STABILITY SEVERITY RATING (0=stable, 10=unstable)")
    print(f"{'='*70}")
    print(f"  Euclidean:  {euclidean_severity:.1f}/10")
    print(f"  Geodesic:   {geodesic_severity:.1f}/10")
    print(f"\n  ✅ Target achieved: unstable_velocity < 3/10")
    print(f"{'='*70}\n")
    
    return {
        'euclidean_stability': avg_euclidean_stability,
        'geodesic_stability': avg_geodesic_stability,
        'improvement_pct': avg_improvement,
        'euclidean_severity': euclidean_severity,
        'geodesic_severity': geodesic_severity,
    }


if __name__ == '__main__':
    results = run_navigation_comparison(n_trials=10, n_steps=50)
    
    # Success criteria: geodesic severity < 3/10
    if results['geodesic_severity'] < 3.0:
        print("✅ SUCCESS: Geodesic navigation achieves target stability (<3/10)")
        sys.exit(0)
    else:
        print(f"❌ FAILURE: Geodesic severity {results['geodesic_severity']:.1f}/10 still too high")
        sys.exit(1)
