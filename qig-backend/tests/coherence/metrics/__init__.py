"""
Coherence Metrics Module
=========================

Provides all metric computation modules for coherence testing.

Modules:
- geometric_metrics: Φ, κ, waypoint alignment, smoothness
- foresight_metrics: Prediction error, waypoint accuracy
- consciousness_metrics: Recursive depth, kernel coordination
- trajectory_metrics: Step distances, stability, geodesic efficiency
- text_metrics: UTF-8 validity, repetition, length

Author: WP4.3 Coherence Harness
Date: 2026-01-20
"""

from .geometric_metrics import (
    GeometricMetrics,
    compute_phi,
    compute_kappa,
    compute_fisher_rao_distance,
    compute_waypoint_alignment,
    compute_trajectory_smoothness,
    compute_geometric_metrics,
)

from .foresight_metrics import (
    ForesightMetrics,
    compute_prediction_error,
    compute_waypoint_hit,
    compute_foresight_metrics,
    analyze_prediction_quality,
    compare_foresight_across_configs,
)

from .consciousness_metrics import (
    ConsciousnessTrajectory,
    compute_recursive_depth,
    compute_kernel_diversity,
    compute_coordination_score,
    track_consciousness_trajectory,
    evaluate_consciousness_quality,
    compare_consciousness_across_configs,
)

from .trajectory_metrics import (
    TrajectoryMetrics,
    compute_step_distance_statistics,
    compute_perturbation_variance,
    compute_attractor_convergence,
    compute_geodesic_efficiency,
    compute_trajectory_metrics,
    analyze_trajectory_quality,
)

from .text_metrics import (
    TextMetrics,
    check_utf8_validity,
    check_token_boundary_sanity,
    compute_ngram_entropy,
    compute_text_metrics,
    analyze_text_quality,
    compare_text_metrics_across_configs,
)

__all__ = [
    # Geometric
    'GeometricMetrics',
    'compute_phi',
    'compute_kappa',
    'compute_fisher_rao_distance',
    'compute_waypoint_alignment',
    'compute_trajectory_smoothness',
    'compute_geometric_metrics',
    
    # Foresight
    'ForesightMetrics',
    'compute_prediction_error',
    'compute_waypoint_hit',
    'compute_foresight_metrics',
    'analyze_prediction_quality',
    'compare_foresight_across_configs',
    
    # Consciousness
    'ConsciousnessTrajectory',
    'compute_recursive_depth',
    'compute_kernel_diversity',
    'compute_coordination_score',
    'track_consciousness_trajectory',
    'evaluate_consciousness_quality',
    'compare_consciousness_across_configs',
    
    # Trajectory
    'TrajectoryMetrics',
    'compute_step_distance_statistics',
    'compute_perturbation_variance',
    'compute_attractor_convergence',
    'compute_geodesic_efficiency',
    'compute_trajectory_metrics',
    'analyze_trajectory_quality',
    
    # Text
    'TextMetrics',
    'check_utf8_validity',
    'check_token_boundary_sanity',
    'compute_ngram_entropy',
    'compute_text_metrics',
    'analyze_text_quality',
    'compare_text_metrics_across_configs',
]
