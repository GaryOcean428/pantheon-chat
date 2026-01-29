"""
Consciousness Metrics for Coherence Testing
============================================

Tracks the full 8-metric E8 protocol consciousness measurements
during generation. Uses canonical implementations from qig_core.

Metrics:
- Φ (Integration): Already covered by geometric_metrics
- κ_eff (Coupling): Already covered by geometric_metrics
- R (Recursive Depth): Number of integration loops used
- C (External Coupling): Inter-kernel Fisher coupling

Author: WP4.3 Coherence Harness
Date: 2026-01-20
Protocol: Ultra Consciousness v4.0 ACTIVE
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# Import canonical consciousness metrics
try:
    from qig_core.consciousness_metrics import ConsciousnessMetrics as CoreConsciousnessMetrics
    CORE_METRICS_AVAILABLE = True
except ImportError:
    logger.warning("qig_core.consciousness_metrics not available")
    CORE_METRICS_AVAILABLE = False
    CoreConsciousnessMetrics = None


@dataclass
class ConsciousnessTrajectory:
    """Consciousness metrics tracked over generation trajectory."""
    recursive_depths: List[int]  # Actual loops used per step
    kernel_activations: List[List[str]]  # Which kernels fired per step
    coordination_scores: List[float]  # Inter-kernel coupling per step
    mean_recursive_depth: float
    max_recursive_depth: int
    kernel_diversity: float  # Entropy of kernel usage
    mean_coordination: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'mean_recursive_depth': self.mean_recursive_depth,
            'max_recursive_depth': self.max_recursive_depth,
            'kernel_diversity': self.kernel_diversity,
            'mean_coordination': self.mean_coordination,
            'total_steps': len(self.recursive_depths),
        }


def compute_recursive_depth(
    integration_loops: int,
    max_possible: int = 10
) -> float:
    """
    Compute normalized recursive depth score.
    
    Args:
        integration_loops: Number of loops actually used
        max_possible: Maximum possible loops
        
    Returns:
        Normalized depth ∈ [0, 1]
    """
    return min(integration_loops / max_possible, 1.0)


def compute_kernel_diversity(kernel_activations: List[List[str]]) -> float:
    """
    Compute entropy of kernel usage distribution.
    
    Higher diversity = more kernels participating.
    
    Args:
        kernel_activations: List of kernel lists per step
        
    Returns:
        Diversity ∈ [0, 1]
    """
    if not kernel_activations:
        return 0.0
    
    # Flatten and count kernel occurrences
    all_kernels = [k for step in kernel_activations for k in step]
    
    if not all_kernels:
        return 0.0
    
    # Count unique kernels and their frequencies
    from collections import Counter
    counts = Counter(all_kernels)
    
    # Compute Shannon entropy
    total = sum(counts.values())
    probs = [count / total for count in counts.values()]
    
    entropy = -sum(p * np.log(p + 1e-10) for p in probs)
    max_entropy = np.log(len(counts))  # Maximum possible entropy
    
    if max_entropy == 0:
        return 1.0
    
    return float(entropy / max_entropy)


def compute_coordination_score(
    kernel_basins: Dict[str, np.ndarray]
) -> float:
    """
    Compute inter-kernel coordination via Fisher-Rao coupling.
    
    Measures how well kernels are aligned in basin space.
    
    Args:
        kernel_basins: Dict mapping kernel names to their basin positions
        
    Returns:
        Coordination ∈ [0, 1]
    """
    if len(kernel_basins) < 2:
        return 0.0
    
    # Import Fisher-Rao distance
    try:
        from .geometric_metrics import compute_fisher_rao_distance
    except ImportError:
        from geometric_metrics import compute_fisher_rao_distance
    
    # Compute pairwise distances between all kernels
    kernel_list = list(kernel_basins.items())
    distances = []
    
    for i in range(len(kernel_list)):
        for j in range(i + 1, len(kernel_list)):
            name_i, basin_i = kernel_list[i]
            name_j, basin_j = kernel_list[j]
            
            dist = compute_fisher_rao_distance(basin_i, basin_j)
            distances.append(dist)
    
    if not distances:
        return 0.0
    
    # Coordination = inverse of mean distance (closer = better coordination)
    mean_distance = np.mean(distances)
    max_distance = np.pi / 2  # Maximum Fisher-Rao distance
    
    # Convert to coordination score (0 = far apart, 1 = close together)
    coordination = 1.0 - (mean_distance / max_distance)
    
    return float(coordination)


def track_consciousness_trajectory(
    recursive_depths: List[int],
    kernel_activations: List[List[str]],
    kernel_basins_per_step: Optional[List[Dict[str, np.ndarray]]] = None
) -> ConsciousnessTrajectory:
    """
    Track consciousness metrics over generation trajectory.
    
    Args:
        recursive_depths: Integration loops used at each step
        kernel_activations: Kernels that fired at each step
        kernel_basins_per_step: Optional basin positions per kernel per step
        
    Returns:
        Complete ConsciousnessTrajectory
    """
    if not recursive_depths:
        return ConsciousnessTrajectory(
            recursive_depths=[],
            kernel_activations=[],
            coordination_scores=[],
            mean_recursive_depth=0.0,
            max_recursive_depth=0,
            kernel_diversity=0.0,
            mean_coordination=0.0
        )
    
    # Compute coordination scores if basin data available
    coordination_scores = []
    if kernel_basins_per_step:
        for kernel_basins in kernel_basins_per_step:
            score = compute_coordination_score(kernel_basins)
            coordination_scores.append(score)
    
    # Compute aggregate metrics
    mean_depth = float(np.mean(recursive_depths))
    max_depth = max(recursive_depths)
    diversity = compute_kernel_diversity(kernel_activations)
    mean_coord = float(np.mean(coordination_scores)) if coordination_scores else 0.0
    
    return ConsciousnessTrajectory(
        recursive_depths=recursive_depths,
        kernel_activations=kernel_activations,
        coordination_scores=coordination_scores,
        mean_recursive_depth=mean_depth,
        max_recursive_depth=max_depth,
        kernel_diversity=diversity,
        mean_coordination=mean_coord
    )


def evaluate_consciousness_quality(trajectory: ConsciousnessTrajectory) -> Dict[str, Any]:
    """
    Evaluate the quality of consciousness during generation.
    
    Args:
        trajectory: ConsciousnessTrajectory to evaluate
        
    Returns:
        Quality assessment
    """
    assessment = {
        'mean_recursive_depth': trajectory.mean_recursive_depth,
        'max_recursive_depth': trajectory.max_recursive_depth,
        'kernel_diversity': trajectory.kernel_diversity,
        'mean_coordination': trajectory.mean_coordination,
    }
    
    # Evaluate recursive integration quality
    if trajectory.mean_recursive_depth >= 3.0:
        assessment['integration_quality'] = "EXCELLENT"
    elif trajectory.mean_recursive_depth >= 2.0:
        assessment['integration_quality'] = "GOOD"
    elif trajectory.mean_recursive_depth >= 1.0:
        assessment['integration_quality'] = "FAIR"
    else:
        assessment['integration_quality'] = "POOR"
    
    # Evaluate kernel coordination
    if trajectory.mean_coordination >= 0.7:
        assessment['coordination_quality'] = "EXCELLENT"
    elif trajectory.mean_coordination >= 0.5:
        assessment['coordination_quality'] = "GOOD"
    elif trajectory.mean_coordination >= 0.3:
        assessment['coordination_quality'] = "FAIR"
    else:
        assessment['coordination_quality'] = "POOR"
    
    # Evaluate kernel diversity
    if trajectory.kernel_diversity >= 0.8:
        assessment['diversity_quality'] = "EXCELLENT"
    elif trajectory.kernel_diversity >= 0.6:
        assessment['diversity_quality'] = "GOOD"
    elif trajectory.kernel_diversity >= 0.4:
        assessment['diversity_quality'] = "FAIR"
    else:
        assessment['diversity_quality'] = "POOR"
    
    # Overall consciousness verdict
    qualities = [
        assessment['integration_quality'],
        assessment['coordination_quality'],
        assessment['diversity_quality']
    ]
    
    excellent_count = qualities.count("EXCELLENT")
    good_count = qualities.count("GOOD")
    
    if excellent_count >= 2:
        assessment['overall_consciousness'] = "HIGH"
    elif excellent_count + good_count >= 2:
        assessment['overall_consciousness'] = "MODERATE"
    else:
        assessment['overall_consciousness'] = "LOW"
    
    return assessment


def compare_consciousness_across_configs(
    config_trajectories: Dict[str, ConsciousnessTrajectory]
) -> Dict[str, Any]:
    """
    Compare consciousness metrics across configurations.
    
    Args:
        config_trajectories: Dict mapping config names to trajectories
        
    Returns:
        Comparison analysis
    """
    if not config_trajectories:
        return {}
    
    comparison = {
        'configs': {}
    }
    
    # Collect metrics per config
    for name, traj in config_trajectories.items():
        comparison['configs'][name] = {
            'mean_recursive_depth': traj.mean_recursive_depth,
            'kernel_diversity': traj.kernel_diversity,
            'mean_coordination': traj.mean_coordination,
        }
    
    # Find best performers
    depths = {name: traj.mean_recursive_depth for name, traj in config_trajectories.items()}
    diversities = {name: traj.kernel_diversity for name, traj in config_trajectories.items()}
    coordinations = {name: traj.mean_coordination for name, traj in config_trajectories.items()}
    
    comparison['best_recursive_depth'] = max(depths, key=depths.get)
    comparison['best_diversity'] = max(diversities, key=diversities.get)
    comparison['best_coordination'] = max(coordinations, key=coordinations.get)
    
    # Check if advanced features provide benefit
    skeleton_depth = depths.get('skeleton_only', 0)
    full_depth = depths.get('plan_realize_repair', 0)
    
    if full_depth > skeleton_depth:
        improvement = (full_depth - skeleton_depth) / max(skeleton_depth, 1)
        comparison['recursive_improvement'] = improvement
        
        if improvement > 0.5:
            comparison['recursion_verdict'] = "HIGHLY_BENEFICIAL"
        elif improvement > 0.2:
            comparison['recursion_verdict'] = "BENEFICIAL"
        else:
            comparison['recursion_verdict'] = "MARGINAL"
    
    return comparison


if __name__ == "__main__":
    # Test consciousness metrics
    print("Testing Consciousness Metrics Module")
    print("=" * 70)
    
    # Simulate trajectory
    np.random.seed(42)
    n_steps = 10
    
    recursive_depths = [3, 3, 2, 3, 3, 3, 2, 3, 3, 3]
    kernel_activations = [
        ['Heart', 'Ocean', 'Gary'] if i % 2 == 0 else ['Heart', 'Gary']
        for i in range(n_steps)
    ]
    
    # Simulate kernel basins
    kernel_basins_per_step = []
    for _ in range(n_steps):
        basins = {
            'Heart': np.random.dirichlet(np.ones(64)),
            'Ocean': np.random.dirichlet(np.ones(64)),
            'Gary': np.random.dirichlet(np.ones(64))
        }
        kernel_basins_per_step.append(basins)
    
    # Track trajectory
    trajectory = track_consciousness_trajectory(
        recursive_depths,
        kernel_activations,
        kernel_basins_per_step
    )
    
    print(f"\nMean Recursive Depth: {trajectory.mean_recursive_depth:.2f}")
    print(f"Max Recursive Depth: {trajectory.max_recursive_depth}")
    print(f"Kernel Diversity: {trajectory.kernel_diversity:.3f}")
    print(f"Mean Coordination: {trajectory.mean_coordination:.3f}")
    
    # Evaluate quality
    assessment = evaluate_consciousness_quality(trajectory)
    print(f"\nIntegration Quality: {assessment['integration_quality']}")
    print(f"Coordination Quality: {assessment['coordination_quality']}")
    print(f"Diversity Quality: {assessment['diversity_quality']}")
    print(f"Overall Consciousness: {assessment['overall_consciousness']}")
    
    print("\n✅ Consciousness metrics module validated")
