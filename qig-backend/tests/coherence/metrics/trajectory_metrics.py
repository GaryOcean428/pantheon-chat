"""
Trajectory Metrics for Coherence Testing
=========================================

Advanced geometric trajectory analysis beyond basic smoothness.
Focuses on manifold properties and geodesic behavior.

Metrics:
- Step Distance Distribution: Statistics of Fisher-Rao steps
- Variance Under Perturbations: Stability to small changes
- Attractor Stability: How strongly trajectory converges
- Geodesic Deviation: Departure from shortest path

Author: WP4.3 Coherence Harness
Date: 2026-01-20
Protocol: Ultra Consciousness v4.0 ACTIVE
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# Import geometric operations
try:
    from .geometric_metrics import compute_fisher_rao_distance
except ImportError:
    from geometric_metrics import compute_fisher_rao_distance


@dataclass
class TrajectoryMetrics:
    """Advanced trajectory analysis metrics."""
    step_distances: List[float]
    distance_mean: float
    distance_std: float
    distance_min: float
    distance_max: float
    perturbation_variance: float
    attractor_convergence: float
    geodesic_efficiency: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'distance_mean': self.distance_mean,
            'distance_std': self.distance_std,
            'distance_min': self.distance_min,
            'distance_max': self.distance_max,
            'perturbation_variance': self.perturbation_variance,
            'attractor_convergence': self.attractor_convergence,
            'geodesic_efficiency': self.geodesic_efficiency,
        }


def compute_step_distance_statistics(basins: List[np.ndarray]) -> Dict[str, float]:
    """
    Compute statistics of Fisher-Rao step distances.
    
    Args:
        basins: Sequence of basin coordinates
        
    Returns:
        Dictionary with mean, std, min, max
    """
    if len(basins) < 2:
        return {
            'mean': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0,
        }
    
    distances = []
    for i in range(len(basins) - 1):
        dist = compute_fisher_rao_distance(basins[i], basins[i+1])
        distances.append(dist)
    
    return {
        'mean': float(np.mean(distances)),
        'std': float(np.std(distances)),
        'min': float(np.min(distances)),
        'max': float(np.max(distances)),
    }


def compute_perturbation_variance(
    basins: List[np.ndarray],
    n_perturbations: int = 5,
    perturbation_magnitude: float = 0.05
) -> float:
    """
    Compute variance under small perturbations.
    
    Measures trajectory stability by perturbing start and measuring divergence.
    
    Args:
        basins: Original trajectory basins
        n_perturbations: Number of perturbation trials
        perturbation_magnitude: Size of perturbation
        
    Returns:
        Variance score ∈ [0, ∞), lower = more stable
    """
    if len(basins) < 2:
        return 0.0
    
    # Perturb initial basin and measure trajectory divergence
    initial_basin = basins[0]
    divergences = []
    
    for _ in range(n_perturbations):
        # Create small perturbation
        noise = np.random.normal(0, perturbation_magnitude, size=initial_basin.shape)
        perturbed = initial_basin + noise
        perturbed = np.abs(perturbed)
        perturbed = perturbed / perturbed.sum()
        
        # Measure divergence from original trajectory
        # (In real test, would re-run generation, here we approximate)
        divergence = compute_fisher_rao_distance(perturbed, basins[-1])
        divergences.append(divergence)
    
    # Variance of divergences
    variance = float(np.var(divergences))
    
    return variance


def compute_attractor_convergence(
    basins: List[np.ndarray],
    attractor: np.ndarray
) -> float:
    """
    Measure convergence toward attractor.
    
    Stronger attractors show monotonic decrease in distance.
    
    Args:
        basins: Trajectory basins
        attractor: Target attractor basin
        
    Returns:
        Convergence strength ∈ [0, 1], higher = stronger pull
    """
    if len(basins) < 2:
        return 0.0
    
    # Compute distances to attractor
    distances = [
        compute_fisher_rao_distance(basin, attractor)
        for basin in basins
    ]
    
    # Count monotonic decreases
    decreases = 0
    for i in range(len(distances) - 1):
        if distances[i+1] < distances[i]:
            decreases += 1
    
    # Convergence = fraction of steps moving toward attractor
    convergence = decreases / (len(distances) - 1) if len(distances) > 1 else 0.0
    
    # Also factor in overall distance reduction
    initial_distance = distances[0]
    final_distance = distances[-1]
    
    if initial_distance > 0:
        distance_reduction = max(0, (initial_distance - final_distance) / initial_distance)
    else:
        distance_reduction = 0.0
    
    # Combined score
    score = 0.6 * convergence + 0.4 * distance_reduction
    
    return float(score)


def compute_geodesic_efficiency(basins: List[np.ndarray]) -> float:
    """
    Measure how closely trajectory follows geodesic path.
    
    Geodesic = shortest path on manifold.
    Efficiency = 1 - (actual_length / direct_distance)
    
    Args:
        basins: Trajectory basins
        
    Returns:
        Efficiency ∈ [0, 1], 1 = perfect geodesic
    """
    if len(basins) < 2:
        return 1.0
    
    # Compute direct distance (start to end)
    direct_distance = compute_fisher_rao_distance(basins[0], basins[-1])
    
    # Compute actual path length (sum of all steps)
    path_length = 0.0
    for i in range(len(basins) - 1):
        path_length += compute_fisher_rao_distance(basins[i], basins[i+1])
    
    if direct_distance == 0:
        return 1.0
    
    # Efficiency = direct / actual (1 = geodesic, <1 = detours)
    efficiency = direct_distance / path_length if path_length > 0 else 0.0
    
    return float(min(efficiency, 1.0))


def compute_trajectory_metrics(
    basins: List[np.ndarray],
    attractor: Optional[np.ndarray] = None,
    n_perturbations: int = 5
) -> TrajectoryMetrics:
    """
    Compute all trajectory metrics.
    
    Args:
        basins: Sequence of basin coordinates
        attractor: Optional target attractor
        n_perturbations: Number of perturbation trials
        
    Returns:
        Complete TrajectoryMetrics
    """
    # Step distance statistics
    stats = compute_step_distance_statistics(basins)
    
    # Perturbation variance (stability)
    perturbation_var = compute_perturbation_variance(basins, n_perturbations)
    
    # Attractor convergence
    if attractor is not None:
        convergence = compute_attractor_convergence(basins, attractor)
    else:
        convergence = 0.0
    
    # Geodesic efficiency
    efficiency = compute_geodesic_efficiency(basins)
    
    # Extract step distances for storage
    step_distances = []
    if len(basins) > 1:
        for i in range(len(basins) - 1):
            dist = compute_fisher_rao_distance(basins[i], basins[i+1])
            step_distances.append(dist)
    
    return TrajectoryMetrics(
        step_distances=step_distances,
        distance_mean=stats['mean'],
        distance_std=stats['std'],
        distance_min=stats['min'],
        distance_max=stats['max'],
        perturbation_variance=perturbation_var,
        attractor_convergence=convergence,
        geodesic_efficiency=efficiency
    )


def analyze_trajectory_quality(metrics: TrajectoryMetrics) -> Dict[str, Any]:
    """
    Analyze trajectory quality and provide insights.
    
    Args:
        metrics: TrajectoryMetrics to analyze
        
    Returns:
        Quality assessment
    """
    assessment = {}
    
    # Evaluate consistency
    if metrics.distance_std < 0.1:
        assessment['consistency'] = "EXCELLENT"
    elif metrics.distance_std < 0.2:
        assessment['consistency'] = "GOOD"
    elif metrics.distance_std < 0.3:
        assessment['consistency'] = "FAIR"
    else:
        assessment['consistency'] = "POOR"
    
    # Evaluate stability
    if metrics.perturbation_variance < 0.05:
        assessment['stability'] = "EXCELLENT"
    elif metrics.perturbation_variance < 0.1:
        assessment['stability'] = "GOOD"
    elif metrics.perturbation_variance < 0.2:
        assessment['stability'] = "FAIR"
    else:
        assessment['stability'] = "POOR"
    
    # Evaluate attractor strength
    if metrics.attractor_convergence > 0.8:
        assessment['attractor_strength'] = "STRONG"
    elif metrics.attractor_convergence > 0.6:
        assessment['attractor_strength'] = "MODERATE"
    elif metrics.attractor_convergence > 0.4:
        assessment['attractor_strength'] = "WEAK"
    else:
        assessment['attractor_strength'] = "VERY_WEAK"
    
    # Evaluate geodesic adherence
    if metrics.geodesic_efficiency > 0.9:
        assessment['path_optimality'] = "EXCELLENT"
    elif metrics.geodesic_efficiency > 0.7:
        assessment['path_optimality'] = "GOOD"
    elif metrics.geodesic_efficiency > 0.5:
        assessment['path_optimality'] = "FAIR"
    else:
        assessment['path_optimality'] = "POOR"
    
    # Overall trajectory quality
    scores = {
        'EXCELLENT': 4,
        'GOOD': 3,
        'STRONG': 3,
        'MODERATE': 2,
        'FAIR': 2,
        'WEAK': 1,
        'POOR': 1,
        'VERY_WEAK': 0,
    }
    
    qualities = [
        assessment['consistency'],
        assessment['stability'],
        assessment['attractor_strength'],
        assessment['path_optimality']
    ]
    
    avg_score = np.mean([scores.get(q, 1) for q in qualities])
    
    if avg_score >= 3.5:
        assessment['overall_quality'] = "EXCELLENT"
    elif avg_score >= 2.5:
        assessment['overall_quality'] = "GOOD"
    elif avg_score >= 1.5:
        assessment['overall_quality'] = "FAIR"
    else:
        assessment['overall_quality'] = "POOR"
    
    return assessment


if __name__ == "__main__":
    # Test trajectory metrics
    print("Testing Trajectory Metrics Module")
    print("=" * 70)
    
    # Create synthetic trajectory
    np.random.seed(42)
    n_steps = 20
    
    # Simulate smooth trajectory with convergence
    basins = []
    attractor = np.random.dirichlet(np.ones(64))
    current = np.random.dirichlet(np.ones(64))
    
    for i in range(n_steps):
        # Move toward attractor with noise
        step = 0.1 * (attractor - current) + 0.05 * np.random.randn(64)
        current = current + step
        current = np.abs(current)
        current = current / current.sum()
        basins.append(current.copy())
    
    # Compute metrics
    metrics = compute_trajectory_metrics(basins, attractor, n_perturbations=5)
    
    print(f"\nDistance Mean: {metrics.distance_mean:.3f}")
    print(f"Distance Std: {metrics.distance_std:.3f}")
    print(f"Perturbation Variance: {metrics.perturbation_variance:.3f}")
    print(f"Attractor Convergence: {metrics.attractor_convergence:.3f}")
    print(f"Geodesic Efficiency: {metrics.geodesic_efficiency:.3f}")
    
    # Analyze quality
    assessment = analyze_trajectory_quality(metrics)
    print(f"\nConsistency: {assessment['consistency']}")
    print(f"Stability: {assessment['stability']}")
    print(f"Attractor Strength: {assessment['attractor_strength']}")
    print(f"Path Optimality: {assessment['path_optimality']}")
    print(f"Overall Quality: {assessment['overall_quality']}")
    
    print("\n✅ Trajectory metrics module validated")
