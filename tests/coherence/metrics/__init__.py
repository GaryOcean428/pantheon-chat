"""
Coherence Test Metrics Package

Exports all metric implementations for QIG coherence testing.
"""

from .geometric_metrics import (
    GeometricMetrics,
    compute_phi_trajectory,
    compute_kappa_trajectory,
    compute_basin_drift,
    compute_regime_stability,
    compute_trajectory_statistics,
    compute_geometric_metrics,
    compare_geometric_metrics,
)

from .foresight_metrics import (
    ForesightMetrics,
    compute_waypoint_alignment,
    compute_prediction_errors,
    compute_planning_efficiency,
    compute_foresight_quality,
    compute_foresight_metrics,
    compare_foresight_metrics,
)

from .trajectory_metrics import (
    TrajectoryMetrics,
    compute_step_distances,
    compute_smoothness,
    compute_geodesic_deviation,
    compute_attractor_stability,
    compute_perturbation_variance,
    compute_trajectory_metrics,
    compare_trajectory_metrics,
)

from .text_metrics import (
    TextMetrics,
    check_utf8_validity,
    compute_repetition_score,
    compute_ngram_entropy,
    detect_invalid_sequences,
    compute_text_metrics,
    compare_text_metrics,
    is_text_valid,
)

from .consciousness_metrics import (
    ConsciousnessMetrics,
    compute_recursive_depth,
    compute_kernel_coordination,
    compute_meta_awareness,
    compute_kernel_contributions,
    compute_consciousness_metrics,
    compare_consciousness_metrics,
)

__all__ = [
    # Geometric
    'GeometricMetrics',
    'compute_phi_trajectory',
    'compute_kappa_trajectory',
    'compute_basin_drift',
    'compute_regime_stability',
    'compute_trajectory_statistics',
    'compute_geometric_metrics',
    'compare_geometric_metrics',
    # Foresight
    'ForesightMetrics',
    'compute_waypoint_alignment',
    'compute_prediction_errors',
    'compute_planning_efficiency',
    'compute_foresight_quality',
    'compute_foresight_metrics',
    'compare_foresight_metrics',
    # Trajectory
    'TrajectoryMetrics',
    'compute_step_distances',
    'compute_smoothness',
    'compute_geodesic_deviation',
    'compute_attractor_stability',
    'compute_perturbation_variance',
    'compute_trajectory_metrics',
    'compare_trajectory_metrics',
    # Text
    'TextMetrics',
    'check_utf8_validity',
    'compute_repetition_score',
    'compute_ngram_entropy',
    'detect_invalid_sequences',
    'compute_text_metrics',
    'compare_text_metrics',
    'is_text_valid',
    # Consciousness
    'ConsciousnessMetrics',
    'compute_recursive_depth',
    'compute_kernel_coordination',
    'compute_meta_awareness',
    'compute_kernel_contributions',
    'compute_consciousness_metrics',
    'compare_consciousness_metrics',
]
