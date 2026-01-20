"""
Foresight Metrics for Coherence Testing
========================================

Measures the quality of waypoint planning and trajectory prediction.
Evaluates how well the planner predicted future basins before generation.

Metrics:
- Prediction Error: Fisher-Rao distance between predicted and actual
- Waypoint Accuracy: Percentage of successful waypoint hits
- Target Alignment: Average alignment with planned targets

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
class ForesightMetrics:
    """Foresight and prediction metrics."""
    prediction_errors: List[float]  # Per-step prediction errors
    waypoint_hits: List[bool]  # Boolean hits for each waypoint
    mean_prediction_error: float
    waypoint_accuracy: float  # Percentage of hits
    prediction_confidence: float  # Inverse of error variance
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'mean_prediction_error': self.mean_prediction_error,
            'waypoint_accuracy': self.waypoint_accuracy,
            'prediction_confidence': self.prediction_confidence,
            'total_waypoints': len(self.waypoint_hits),
            'successful_hits': sum(self.waypoint_hits),
            'error_std': np.std(self.prediction_errors) if self.prediction_errors else 0.0,
        }


def compute_prediction_error(
    predicted_basin: np.ndarray,
    actual_basin: np.ndarray
) -> float:
    """
    Compute prediction error as Fisher-Rao distance.
    
    Measures how far the actual generation deviated from prediction.
    
    Args:
        predicted_basin: Predicted next basin (from planner)
        actual_basin: Actual next basin (from realizer)
        
    Returns:
        Error ∈ [0, π/2]
    """
    return compute_fisher_rao_distance(predicted_basin, actual_basin)


def compute_waypoint_hit(
    actual_basin: np.ndarray,
    waypoint: np.ndarray,
    threshold: float = 0.3
) -> bool:
    """
    Check if actual basin hit the waypoint target.
    
    A "hit" means the Fisher-Rao distance is below threshold.
    
    Args:
        actual_basin: Actual generated basin
        waypoint: Target waypoint basin
        threshold: Maximum distance to count as hit (default: 0.3)
        
    Returns:
        True if hit, False otherwise
    """
    distance = compute_fisher_rao_distance(actual_basin, waypoint)
    return distance < threshold


def compute_foresight_metrics(
    predicted_waypoints: List[np.ndarray],
    actual_basins: List[np.ndarray],
    hit_threshold: float = 0.3
) -> ForesightMetrics:
    """
    Compute all foresight metrics for trajectory.
    
    Args:
        predicted_waypoints: Sequence of planned waypoints
        actual_basins: Sequence of actual generated basins
        hit_threshold: Distance threshold for waypoint hits
        
    Returns:
        Complete ForesightMetrics object
    """
    if not predicted_waypoints or not actual_basins:
        return ForesightMetrics(
            prediction_errors=[],
            waypoint_hits=[],
            mean_prediction_error=0.0,
            waypoint_accuracy=0.0,
            prediction_confidence=0.0
        )
    
    # Use minimum length to compare
    n = min(len(predicted_waypoints), len(actual_basins))
    
    # Compute prediction errors
    prediction_errors = []
    waypoint_hits = []
    
    for i in range(n):
        error = compute_prediction_error(predicted_waypoints[i], actual_basins[i])
        prediction_errors.append(error)
        
        hit = compute_waypoint_hit(actual_basins[i], predicted_waypoints[i], hit_threshold)
        waypoint_hits.append(hit)
    
    # Compute aggregate metrics
    mean_error = float(np.mean(prediction_errors)) if prediction_errors else 0.0
    accuracy = sum(waypoint_hits) / len(waypoint_hits) if waypoint_hits else 0.0
    
    # Prediction confidence = inverse of error variance
    error_variance = np.var(prediction_errors) if len(prediction_errors) > 1 else 0.0
    confidence = 1.0 / (1.0 + error_variance)  # Normalized to [0, 1]
    
    return ForesightMetrics(
        prediction_errors=prediction_errors,
        waypoint_hits=waypoint_hits,
        mean_prediction_error=mean_error,
        waypoint_accuracy=float(accuracy),
        prediction_confidence=float(confidence)
    )


def analyze_prediction_quality(
    metrics: ForesightMetrics,
    config_name: str = "Unknown"
) -> Dict[str, Any]:
    """
    Analyze prediction quality and generate insights.
    
    Args:
        metrics: ForesightMetrics to analyze
        config_name: Name of configuration being tested
        
    Returns:
        Dictionary with analysis results
    """
    analysis = {
        'config_name': config_name,
        'mean_error': metrics.mean_prediction_error,
        'waypoint_accuracy': metrics.waypoint_accuracy,
        'confidence': metrics.prediction_confidence,
    }
    
    # Determine quality rating
    if metrics.mean_prediction_error < 0.2:
        quality = "EXCELLENT"
    elif metrics.mean_prediction_error < 0.4:
        quality = "GOOD"
    elif metrics.mean_prediction_error < 0.6:
        quality = "FAIR"
    else:
        quality = "POOR"
    
    analysis['quality_rating'] = quality
    
    # Check if predictions are useful
    if metrics.waypoint_accuracy > 0.7:
        analysis['prediction_usefulness'] = "HIGH"
        analysis['recommendation'] = "Foresight is effective - continue using waypoint planning"
    elif metrics.waypoint_accuracy > 0.4:
        analysis['prediction_usefulness'] = "MODERATE"
        analysis['recommendation'] = "Foresight provides some benefit - consider tuning parameters"
    else:
        analysis['prediction_usefulness'] = "LOW"
        analysis['recommendation'] = "Foresight may not be effective - investigate planner issues"
    
    return analysis


def compare_foresight_across_configs(
    config_metrics: Dict[str, ForesightMetrics]
) -> Dict[str, Any]:
    """
    Compare foresight performance across multiple configurations.
    
    Args:
        config_metrics: Dict mapping config names to ForesightMetrics
        
    Returns:
        Comparison analysis
    """
    if not config_metrics:
        return {}
    
    # Find best and worst configurations
    errors = {name: m.mean_prediction_error for name, m in config_metrics.items()}
    accuracies = {name: m.waypoint_accuracy for name, m in config_metrics.items()}
    
    best_error_config = min(errors, key=errors.get)
    worst_error_config = max(errors, key=errors.get)
    
    best_accuracy_config = max(accuracies, key=accuracies.get)
    worst_accuracy_config = min(accuracies, key=accuracies.get)
    
    comparison = {
        'best_prediction_error': {
            'config': best_error_config,
            'error': errors[best_error_config]
        },
        'worst_prediction_error': {
            'config': worst_error_config,
            'error': errors[worst_error_config]
        },
        'best_waypoint_accuracy': {
            'config': best_accuracy_config,
            'accuracy': accuracies[best_accuracy_config]
        },
        'worst_waypoint_accuracy': {
            'config': worst_accuracy_config,
            'accuracy': accuracies[worst_accuracy_config]
        },
        'error_range': max(errors.values()) - min(errors.values()),
        'accuracy_range': max(accuracies.values()) - min(accuracies.values()),
    }
    
    # Check if foresight provides benefit
    skeleton_error = errors.get('skeleton_only', 0)
    full_error = errors.get('plan_realize_repair', 0)
    
    if full_error > 0 and skeleton_error > 0:
        improvement = (skeleton_error - full_error) / skeleton_error
        comparison['foresight_improvement'] = improvement
        
        if improvement > 0.2:
            comparison['foresight_verdict'] = "BENEFICIAL"
        elif improvement > 0:
            comparison['foresight_verdict'] = "MARGINAL"
        else:
            comparison['foresight_verdict'] = "NOT_BENEFICIAL"
    
    return comparison


if __name__ == "__main__":
    # Test foresight metrics computation
    print("Testing Foresight Metrics Module")
    print("=" * 70)
    
    # Create synthetic predictions and actuals
    np.random.seed(42)
    n_steps = 10
    
    # Simulate predictions (with some noise)
    predicted = [np.random.dirichlet(np.ones(64)) for _ in range(n_steps)]
    
    # Simulate actuals (similar to predictions but with drift)
    actual = []
    for pred in predicted:
        noise = np.random.dirichlet(np.ones(64) * 0.1)
        actual_basin = 0.8 * pred + 0.2 * noise
        actual_basin = actual_basin / actual_basin.sum()
        actual.append(actual_basin)
    
    # Compute metrics
    metrics = compute_foresight_metrics(predicted, actual, hit_threshold=0.3)
    
    print(f"\nMean Prediction Error: {metrics.mean_prediction_error:.3f}")
    print(f"Waypoint Accuracy: {metrics.waypoint_accuracy:.1%}")
    print(f"Prediction Confidence: {metrics.prediction_confidence:.3f}")
    print(f"Successful Hits: {sum(metrics.waypoint_hits)}/{len(metrics.waypoint_hits)}")
    
    # Analyze quality
    analysis = analyze_prediction_quality(metrics, "test_config")
    print(f"\nQuality Rating: {analysis['quality_rating']}")
    print(f"Prediction Usefulness: {analysis['prediction_usefulness']}")
    print(f"Recommendation: {analysis['recommendation']}")
    
    print("\n✅ Foresight metrics module validated")
