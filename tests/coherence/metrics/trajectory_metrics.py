"""
Trajectory Metrics for QIG Coherence Testing

Measures geometric trajectory quality:
- Smoothness: Variance of step distances
- Geodesic deviation: How far from optimal path
- Attractor stability: Convergence behavior
- Variance under perturbation: Robustness

Uses Fisher-Rao distance for all measurements.

Author: QIG Consciousness Project
Date: January 2026
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "qig-backend"))

try:
    from qig_core.consciousness_metrics import fisher_rao_distance
except ImportError:
    def fisher_rao_distance(p: np.ndarray, q: np.ndarray) -> float:
        """Fisher-Rao distance between probability distributions."""
        p_safe = np.abs(p) + 1e-10
        q_safe = np.abs(q) + 1e-10
        p_safe = p_safe / p_safe.sum()
        q_safe = q_safe / q_safe.sum()
        
        bhattacharyya = np.sum(np.sqrt(p_safe * q_safe))
        bhattacharyya = np.clip(bhattacharyya, 0.0, 1.0)
        
        return np.arccos(bhattacharyya)


@dataclass
class TrajectoryMetrics:
    """Container for trajectory quality metrics."""
    smoothness: float              # 1 - variance (0-1, higher better)
    mean_step_distance: float      # Average step size
    step_variance: float           # Variance in step distances
    geodesic_deviation: float      # Distance from ideal geodesic
    attractor_stability: float     # Convergence to attractor (0-1)
    perturbation_variance: float   # Sensitivity to perturbation
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'smoothness': self.smoothness,
            'mean_step_distance': self.mean_step_distance,
            'step_variance': self.step_variance,
            'geodesic_deviation': self.geodesic_deviation,
            'attractor_stability': self.attractor_stability,
            'perturbation_variance': self.perturbation_variance,
        }


def compute_step_distances(basins: List[np.ndarray]) -> List[float]:
    """
    Compute Fisher-Rao distances between consecutive basins.
    
    Args:
        basins: List of basin coordinates
        
    Returns:
        List of step distances
    """
    if len(basins) < 2:
        return []
    
    distances = []
    for i in range(len(basins) - 1):
        dist = fisher_rao_distance(basins[i], basins[i+1])
        distances.append(dist)
    
    return distances


def compute_smoothness(distances: List[float]) -> float:
    """
    Compute trajectory smoothness.
    
    Smoothness = 1 - normalized_variance.
    Higher smoothness = more consistent step sizes.
    
    Args:
        distances: Step distances
        
    Returns:
        Smoothness score (0-1)
    """
    if len(distances) < 2:
        return 1.0
    
    variance = float(np.var(distances))
    mean = float(np.mean(distances))
    
    # Normalize variance by mean to get coefficient of variation
    if mean > 0:
        cv = np.sqrt(variance) / mean
        # Map CV to smoothness (high CV = low smoothness)
        smoothness = 1.0 / (1.0 + cv)
    else:
        smoothness = 1.0
    
    return smoothness


def compute_geodesic_deviation(
    basins: List[np.ndarray],
    start: Optional[np.ndarray] = None,
    end: Optional[np.ndarray] = None
) -> float:
    """
    Compute deviation from ideal geodesic path.
    
    Measures how much the trajectory deviates from the shortest path
    between start and end points on the Fisher manifold.
    
    Args:
        basins: Actual trajectory
        start: Start basin (default: first basin)
        end: End basin (default: last basin)
        
    Returns:
        Mean deviation from geodesic
    """
    if len(basins) < 3:
        return 0.0
    
    if start is None:
        start = basins[0]
    if end is None:
        end = basins[-1]
    
    # Direct geodesic distance
    direct_distance = fisher_rao_distance(start, end)
    
    # Actual path length
    path_length = sum(compute_step_distances(basins))
    
    # Deviation = excess path length
    if direct_distance > 0:
        deviation = (path_length - direct_distance) / direct_distance
    else:
        deviation = 0.0
    
    return float(max(0.0, deviation))


def compute_attractor_stability(
    basins: List[np.ndarray],
    attractor: Optional[np.ndarray] = None
) -> float:
    """
    Measure convergence toward attractor basin.
    
    Stability = 1 if trajectory converges to attractor.
    Stability = 0 if trajectory diverges.
    
    Args:
        basins: Trajectory basins
        attractor: Target attractor (default: last basin)
        
    Returns:
        Stability score (0-1)
    """
    if len(basins) < 3:
        return 1.0
    
    if attractor is None:
        attractor = basins[-1]
    
    # Compute distances to attractor over time
    distances = []
    for basin in basins:
        dist = fisher_rao_distance(basin, attractor)
        distances.append(dist)
    
    # Check if distances are decreasing (convergence)
    if len(distances) < 2:
        return 1.0
    
    # Linear fit to distances
    x = np.arange(len(distances))
    y = np.array(distances)
    
    # Slope of linear fit (negative = converging)
    if len(x) > 1:
        slope = np.polyfit(x, y, 1)[0]
        
        # Stability = how much we're converging (normalized)
        # Negative slope = converging = high stability
        stability = np.clip(-slope, 0.0, 1.0)
    else:
        stability = 0.5
    
    return float(stability)


def compute_perturbation_variance(
    basins: List[np.ndarray],
    noise_level: float = 0.01,
    n_samples: int = 5
) -> float:
    """
    Measure trajectory sensitivity to small perturbations.
    
    Lower variance = more robust trajectory.
    
    Args:
        basins: Original trajectory
        noise_level: Magnitude of perturbation
        n_samples: Number of perturbed samples
        
    Returns:
        Variance in perturbed trajectories
    """
    if len(basins) < 2:
        return 0.0
    
    # Generate perturbed trajectories
    perturbed_distances = []
    
    for _ in range(n_samples):
        perturbed_basins = []
        for basin in basins:
            # Add small noise
            noise = np.random.normal(0, noise_level, size=basin.shape)
            perturbed = basin + noise
            
            # Renormalize to simplex
            perturbed = np.abs(perturbed) + 1e-10
            perturbed = perturbed / perturbed.sum()
            
            perturbed_basins.append(perturbed)
        
        # Compute total path length for perturbed trajectory
        path_length = sum(compute_step_distances(perturbed_basins))
        perturbed_distances.append(path_length)
    
    # Variance in path lengths
    variance = float(np.var(perturbed_distances))
    
    return variance


def compute_trajectory_metrics(
    basins: List[np.ndarray],
    attractor: Optional[np.ndarray] = None,
    measure_perturbation: bool = False
) -> TrajectoryMetrics:
    """
    Compute full trajectory metrics.
    
    Args:
        basins: List of basin coordinates
        attractor: Optional reference attractor
        measure_perturbation: Whether to compute perturbation variance (expensive)
        
    Returns:
        TrajectoryMetrics containing all measurements
    """
    # Step distances
    distances = compute_step_distances(basins)
    
    if distances:
        mean_dist = float(np.mean(distances))
        variance = float(np.var(distances))
    else:
        mean_dist = 0.0
        variance = 0.0
    
    # Smoothness
    smoothness = compute_smoothness(distances)
    
    # Geodesic deviation
    deviation = compute_geodesic_deviation(basins)
    
    # Attractor stability
    stability = compute_attractor_stability(basins, attractor)
    
    # Perturbation variance (optional, expensive)
    if measure_perturbation:
        pert_var = compute_perturbation_variance(basins)
    else:
        pert_var = 0.0
    
    return TrajectoryMetrics(
        smoothness=smoothness,
        mean_step_distance=mean_dist,
        step_variance=variance,
        geodesic_deviation=deviation,
        attractor_stability=stability,
        perturbation_variance=pert_var,
    )


def compare_trajectory_metrics(
    metrics_a: TrajectoryMetrics,
    metrics_b: TrajectoryMetrics
) -> Dict[str, float]:
    """
    Compare two trajectory metric sets.
    
    Returns deltas (positive = B better than A).
    
    Args:
        metrics_a: First metrics (baseline)
        metrics_b: Second metrics (comparison)
        
    Returns:
        Dictionary of metric deltas
    """
    return {
        'smoothness_delta': metrics_b.smoothness - metrics_a.smoothness,
        'deviation_delta': metrics_a.geodesic_deviation - metrics_b.geodesic_deviation,  # Lower better
        'stability_delta': metrics_b.attractor_stability - metrics_a.attractor_stability,
        'variance_delta': metrics_a.step_variance - metrics_b.step_variance,  # Lower better
    }
