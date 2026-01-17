"""
Foresight Metrics for QIG Coherence Testing

Measures waypoint planning and prediction accuracy:
- Waypoint alignment: Did generation hit planned targets?
- Prediction error: How accurate were basin predictions?
- Foresight quality: Overall planning effectiveness

Uses Fisher-Rao distance for all measurements.

Author: QIG Consciousness Project
Date: January 2026
"""

import numpy as np
from typing import Dict, List, Optional
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
class ForesightMetrics:
    """Container for foresight/planning metrics."""
    waypoint_alignment: float      # How well did we hit targets? (0-1, higher better)
    mean_prediction_error: float   # Average Fisher-Rao error
    max_prediction_error: float    # Worst prediction
    foresight_quality: float       # Overall planning quality (0-1)
    planning_efficiency: float     # Ratio of planned/actual paths
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'waypoint_alignment': self.waypoint_alignment,
            'mean_prediction_error': self.mean_prediction_error,
            'max_prediction_error': self.max_prediction_error,
            'foresight_quality': self.foresight_quality,
            'planning_efficiency': self.planning_efficiency,
        }


def compute_waypoint_alignment(
    actual_basins: List[np.ndarray],
    target_waypoints: List[np.ndarray]
) -> float:
    """
    Compute alignment between actual trajectory and planned waypoints.
    
    Measures how closely generation followed the plan.
    
    Args:
        actual_basins: Actual basins generated
        target_waypoints: Planned target basins
        
    Returns:
        Alignment score (0-1, 1 = perfect alignment)
    """
    if not actual_basins or not target_waypoints:
        return 0.0
    
    # Match lengths (use minimum)
    n = min(len(actual_basins), len(target_waypoints))
    
    # Compute alignment for each position
    alignments = []
    for i in range(n):
        distance = fisher_rao_distance(actual_basins[i], target_waypoints[i])
        # Convert distance to alignment (0 = perfect, π = worst)
        alignment = 1.0 - (distance / np.pi)
        alignments.append(alignment)
    
    return float(np.mean(alignments))


def compute_prediction_errors(
    predicted_basins: List[np.ndarray],
    actual_basins: List[np.ndarray]
) -> Dict[str, float]:
    """
    Compute prediction accuracy metrics.
    
    Measures how well foresight predicted future basins.
    
    Args:
        predicted_basins: Predicted next basins
        actual_basins: Actual basins generated
        
    Returns:
        Dictionary with mean, max, std of prediction errors
    """
    if not predicted_basins or not actual_basins:
        return {
            'mean': 0.0,
            'max': 0.0,
            'std': 0.0,
        }
    
    n = min(len(predicted_basins), len(actual_basins))
    
    errors = []
    for i in range(n):
        error = fisher_rao_distance(predicted_basins[i], actual_basins[i])
        errors.append(error)
    
    return {
        'mean': float(np.mean(errors)),
        'max': float(np.max(errors)),
        'std': float(np.std(errors)),
    }


def compute_planning_efficiency(
    planned_path: List[np.ndarray],
    actual_path: List[np.ndarray]
) -> float:
    """
    Compute ratio of planned path length to actual path length.
    
    Efficiency = 1.0 means perfect execution of plan.
    Efficiency < 1.0 means actual path was longer (detours).
    Efficiency > 1.0 means actual path was shorter (shortcuts).
    
    Args:
        planned_path: Planned trajectory
        actual_path: Actual trajectory
        
    Returns:
        Efficiency ratio
    """
    if len(planned_path) < 2 or len(actual_path) < 2:
        return 1.0
    
    # Compute total path lengths
    planned_length = 0.0
    for i in range(len(planned_path) - 1):
        planned_length += fisher_rao_distance(
            planned_path[i], 
            planned_path[i+1]
        )
    
    actual_length = 0.0
    for i in range(len(actual_path) - 1):
        actual_length += fisher_rao_distance(
            actual_path[i],
            actual_path[i+1]
        )
    
    if actual_length == 0:
        return 1.0
    
    return planned_length / actual_length


def compute_foresight_quality(
    alignment: float,
    mean_error: float,
    efficiency: float
) -> float:
    """
    Compute overall foresight quality score.
    
    Combines alignment, prediction accuracy, and planning efficiency.
    
    Args:
        alignment: Waypoint alignment (0-1)
        mean_error: Mean prediction error (0-π)
        efficiency: Planning efficiency ratio
        
    Returns:
        Foresight quality (0-1, higher better)
    """
    # Normalize error to 0-1 scale
    normalized_error = 1.0 - (mean_error / np.pi)
    
    # Normalize efficiency (cap at 2.0)
    normalized_efficiency = min(efficiency, 2.0) / 2.0
    
    # Weighted combination
    quality = (
        0.5 * alignment +           # 50% alignment
        0.3 * normalized_error +    # 30% prediction accuracy
        0.2 * normalized_efficiency # 20% efficiency
    )
    
    return float(np.clip(quality, 0.0, 1.0))


def compute_foresight_metrics(
    actual_basins: List[np.ndarray],
    target_waypoints: List[np.ndarray],
    predicted_basins: Optional[List[np.ndarray]] = None
) -> ForesightMetrics:
    """
    Compute full foresight metrics.
    
    Args:
        actual_basins: Actual trajectory basins
        target_waypoints: Planned waypoints
        predicted_basins: Optional predicted basins for error calculation
        
    Returns:
        ForesightMetrics containing all measurements
    """
    # Waypoint alignment
    alignment = compute_waypoint_alignment(actual_basins, target_waypoints)
    
    # Prediction errors (if predictions available)
    if predicted_basins:
        errors = compute_prediction_errors(predicted_basins, actual_basins)
        mean_error = errors['mean']
        max_error = errors['max']
    else:
        mean_error = 0.0
        max_error = 0.0
    
    # Planning efficiency
    efficiency = compute_planning_efficiency(target_waypoints, actual_basins)
    
    # Overall foresight quality
    quality = compute_foresight_quality(alignment, mean_error, efficiency)
    
    return ForesightMetrics(
        waypoint_alignment=alignment,
        mean_prediction_error=mean_error,
        max_prediction_error=max_error,
        foresight_quality=quality,
        planning_efficiency=efficiency,
    )


def compare_foresight_metrics(
    metrics_a: ForesightMetrics,
    metrics_b: ForesightMetrics
) -> Dict[str, float]:
    """
    Compare two foresight metric sets.
    
    Returns deltas (positive = B better than A).
    
    Args:
        metrics_a: First metrics (baseline)
        metrics_b: Second metrics (comparison)
        
    Returns:
        Dictionary of metric deltas
    """
    return {
        'alignment_delta': metrics_b.waypoint_alignment - metrics_a.waypoint_alignment,
        'error_delta': metrics_a.mean_prediction_error - metrics_b.mean_prediction_error,  # Lower better
        'quality_delta': metrics_b.foresight_quality - metrics_a.foresight_quality,
        'efficiency_delta': metrics_b.planning_efficiency - metrics_a.planning_efficiency,
    }
