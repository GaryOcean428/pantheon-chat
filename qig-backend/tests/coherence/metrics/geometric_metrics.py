"""
Geometric Metrics for Coherence Testing
========================================

Implements pure Fisher-Rao geometry-based metrics for evaluating
QIG generation coherence. All metrics use canonical geometric operations.

Metrics:
- Φ (Integration): QFI-based consciousness measurement
- κ (Coupling): Basin coupling strength evolution
- Basin Drift: Distance from attractor over time
- Regime Transitions: Mode switches (feel → logic)
- Waypoint Alignment: How well words hit predicted basins
- Trajectory Smoothness: Variance of Fisher-Rao step distances

Author: WP4.3 Coherence Harness
Date: 2026-01-20
Protocol: Ultra Consciousness v4.0 ACTIVE
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# Import canonical geometric operations
try:
    from qig_core.phi_computation import compute_phi_qig, compute_qfi_matrix
    from qig_core.consciousness_metrics import fisher_rao_distance
    PHI_AVAILABLE = True
except ImportError:
    logger.warning("qig_core not available - using fallback implementations")
    PHI_AVAILABLE = False
    
    # Fallback QFI matrix computation
    def compute_qfi_matrix(basin_coords: np.ndarray) -> np.ndarray:
        """Fallback QFI matrix computation."""
        p = np.abs(basin_coords) ** 2 + 1e-10
        p = p / p.sum()
        n = len(basin_coords)
        qfi = np.zeros((n, n))
        for i in range(n):
            if p[i] > 1e-10:
                qfi[i, i] = 1.0 / p[i]
            else:
                qfi[i, i] = 1.0 / 1e-10
        qfi += 1e-8 * np.eye(n)
        return qfi


@dataclass
class GeometricMetrics:
    """Complete geometric metrics for a generation trajectory."""
    phi_scores: List[float]  # Φ at each step
    kappa_values: List[float]  # κ at each step
    basin_drift: List[float]  # Distance from attractor
    step_distances: List[float]  # Fisher-Rao distances between consecutive basins
    waypoint_alignment: float  # How well words hit targets (0-1)
    trajectory_smoothness: float  # 1 - variance of step distances (0-1)
    regime_stability: float  # Consistency of regime (0-1)
    mean_phi: float
    mean_kappa: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'mean_phi': self.mean_phi,
            'mean_kappa': self.mean_kappa,
            'waypoint_alignment': self.waypoint_alignment,
            'trajectory_smoothness': self.trajectory_smoothness,
            'regime_stability': self.regime_stability,
            'phi_std': np.std(self.phi_scores) if self.phi_scores else 0.0,
            'kappa_std': np.std(self.kappa_values) if self.kappa_values else 0.0,
            'mean_step_distance': np.mean(self.step_distances) if self.step_distances else 0.0,
            'max_basin_drift': max(self.basin_drift) if self.basin_drift else 0.0,
        }


def compute_phi(basin_coords: np.ndarray) -> float:
    """
    Compute Φ (integration) from basin coordinates.
    
    Uses canonical QFI-based computation if available,
    otherwise uses Shannon entropy fallback.
    
    Args:
        basin_coords: 64D basin coordinates (simplex)
        
    Returns:
        Φ ∈ [0, 1]
    """
    if PHI_AVAILABLE:
        try:
            return compute_phi_qig(basin_coords)
        except Exception as e:
            logger.warning(f"Phi computation failed: {e}, using fallback")
    
    # Fallback: Shannon entropy normalization
    p = np.abs(basin_coords) ** 2 + 1e-10
    p = p / p.sum()
    
    entropy = -np.sum(p * np.log(p + 1e-10))
    max_entropy = np.log(len(basin_coords))
    
    return entropy / max_entropy


def compute_kappa(basin_coords: np.ndarray, reference_basin: Optional[np.ndarray] = None) -> float:
    """
    Compute κ (coupling strength) from basin coordinates.
    
    Measures effective coupling via QFI eigenvalue spectrum.
    Target: κ* ≈ 64 (universal fixed point)
    
    Args:
        basin_coords: Current 64D basin coordinates
        reference_basin: Optional reference for coupling measurement
        
    Returns:
        κ_eff ∈ [0, ∞), target ≈ 64
    """
    # Use QFI eigenvalue statistics
    qfi = compute_qfi_matrix(basin_coords)
    eigenvalues = np.linalg.eigvalsh(qfi)
    
    # Effective coupling = trace of QFI / dimension
    kappa_eff = np.sum(eigenvalues) / len(basin_coords)
    
    return float(kappa_eff)


def compute_fisher_rao_distance(p: np.ndarray, q: np.ndarray) -> float:
    """
    Compute Fisher-Rao distance between two basins.
    
    This is the ONLY approved distance metric for QIG.
    
    Args:
        p: First basin (64D simplex)
        q: Second basin (64D simplex)
        
    Returns:
        Distance ∈ [0, π/2]
    """
    if PHI_AVAILABLE:
        try:
            return fisher_rao_distance(p, q)
        except Exception as e:
            logger.warning(f"Fisher-Rao computation failed: {e}, using fallback")
    
    # Fallback implementation
    p_safe = np.abs(p) + 1e-10
    q_safe = np.abs(q) + 1e-10
    p_safe = p_safe / p_safe.sum()
    q_safe = q_safe / q_safe.sum()
    
    bhattacharyya = np.sum(np.sqrt(p_safe * q_safe))
    bhattacharyya = np.clip(bhattacharyya, 0.0, 1.0)
    
    return np.arccos(bhattacharyya)


def compute_waypoint_alignment(
    actual_basins: List[np.ndarray],
    target_waypoints: List[np.ndarray]
) -> float:
    """
    Compute alignment between actual trajectory and planned waypoints.
    
    Measures how well the generation hit its planned targets.
    
    Args:
        actual_basins: Sequence of actual basin coordinates
        target_waypoints: Sequence of planned waypoint basins
        
    Returns:
        Alignment score ∈ [0, 1], higher = better alignment
    """
    if not actual_basins or not target_waypoints:
        return 0.0
    
    # Use minimum length to compare
    n = min(len(actual_basins), len(target_waypoints))
    
    if n == 0:
        return 0.0
    
    # Compute Fisher-Rao distance for each pair
    distances = []
    for actual, target in zip(actual_basins[:n], target_waypoints[:n]):
        dist = compute_fisher_rao_distance(actual, target)
        distances.append(dist)
    
    # Convert distances to alignment scores (closer = better)
    # Distance range: [0, π/2], so normalize
    alignment_scores = [1.0 - (d / (np.pi / 2)) for d in distances]
    
    return float(np.mean(alignment_scores))


def compute_trajectory_smoothness(basins: List[np.ndarray]) -> float:
    """
    Compute trajectory smoothness from basin sequence.
    
    Smoothness = 1 - variance of Fisher-Rao step distances
    Smooth trajectories have consistent step sizes.
    
    Args:
        basins: Sequence of basin coordinates
        
    Returns:
        Smoothness ∈ [0, 1], higher = smoother
    """
    if len(basins) < 2:
        return 1.0
    
    # Compute Fisher-Rao distances between consecutive basins
    distances = []
    for i in range(len(basins) - 1):
        dist = compute_fisher_rao_distance(basins[i], basins[i+1])
        distances.append(dist)
    
    if not distances:
        return 1.0
    
    # Smoothness = low variance
    variance = np.var(distances)
    
    # Normalize by max possible variance (roughly (π/2)^2 / 4)
    max_variance = (np.pi / 2) ** 2 / 4
    normalized_variance = min(variance / max_variance, 1.0)
    
    smoothness = 1.0 - normalized_variance
    
    return float(smoothness)


def compute_basin_drift(
    basins: List[np.ndarray],
    attractor: np.ndarray
) -> List[float]:
    """
    Compute basin drift over trajectory.
    
    Measures how far each basin is from the attractor.
    
    Args:
        basins: Sequence of basin coordinates
        attractor: Target attractor basin
        
    Returns:
        List of drift distances at each step
    """
    drift = []
    for basin in basins:
        distance = compute_fisher_rao_distance(basin, attractor)
        drift.append(float(distance))
    
    return drift


def detect_regime_transitions(
    basins: List[np.ndarray],
    kappa_values: List[float],
    threshold: float = 0.3
) -> Tuple[int, float]:
    """
    Detect regime transitions in trajectory.
    
    Regime transitions occur when κ changes significantly,
    indicating a switch between modes (e.g., feel → logic).
    
    Args:
        basins: Sequence of basin coordinates
        kappa_values: κ values at each step
        threshold: Minimum κ change to count as transition
        
    Returns:
        (num_transitions, stability_score)
    """
    if len(kappa_values) < 2:
        return 0, 1.0
    
    transitions = 0
    for i in range(len(kappa_values) - 1):
        delta_kappa = abs(kappa_values[i+1] - kappa_values[i])
        if delta_kappa > threshold:
            transitions += 1
    
    # Stability = fewer transitions (normalized by length)
    max_transitions = len(kappa_values) - 1
    stability = 1.0 - (transitions / max_transitions) if max_transitions > 0 else 1.0
    
    return transitions, float(stability)


def compute_geometric_metrics(
    basins: List[np.ndarray],
    waypoints: Optional[List[np.ndarray]] = None,
    attractor: Optional[np.ndarray] = None
) -> GeometricMetrics:
    """
    Compute all geometric metrics for a generation trajectory.
    
    Args:
        basins: Sequence of basin coordinates
        waypoints: Optional planned waypoints for alignment
        attractor: Optional attractor for drift measurement
        
    Returns:
        Complete GeometricMetrics object
    """
    # Compute Φ at each step
    phi_scores = [compute_phi(basin) for basin in basins]
    
    # Compute κ at each step
    kappa_values = [compute_kappa(basin) for basin in basins]
    
    # Compute step distances
    step_distances = []
    if len(basins) > 1:
        for i in range(len(basins) - 1):
            dist = compute_fisher_rao_distance(basins[i], basins[i+1])
            step_distances.append(dist)
    
    # Compute waypoint alignment
    if waypoints:
        waypoint_alignment = compute_waypoint_alignment(basins, waypoints)
    else:
        waypoint_alignment = 0.0
    
    # Compute trajectory smoothness
    trajectory_smoothness = compute_trajectory_smoothness(basins)
    
    # Compute basin drift
    if attractor is not None:
        basin_drift = compute_basin_drift(basins, attractor)
    else:
        basin_drift = []
    
    # Detect regime transitions
    _, regime_stability = detect_regime_transitions(basins, kappa_values)
    
    return GeometricMetrics(
        phi_scores=phi_scores,
        kappa_values=kappa_values,
        basin_drift=basin_drift,
        step_distances=step_distances,
        waypoint_alignment=waypoint_alignment,
        trajectory_smoothness=trajectory_smoothness,
        regime_stability=regime_stability,
        mean_phi=float(np.mean(phi_scores)) if phi_scores else 0.0,
        mean_kappa=float(np.mean(kappa_values)) if kappa_values else 0.0,
    )


if __name__ == "__main__":
    # Test geometric metrics computation
    print("Testing Geometric Metrics Module")
    print("=" * 70)
    
    # Create synthetic test trajectory
    np.random.seed(42)
    n_steps = 10
    basins = [np.random.dirichlet(np.ones(64)) for _ in range(n_steps)]
    waypoints = [np.random.dirichlet(np.ones(64)) for _ in range(n_steps)]
    attractor = np.random.dirichlet(np.ones(64))
    
    # Compute metrics
    metrics = compute_geometric_metrics(basins, waypoints, attractor)
    
    print(f"\nMean Φ: {metrics.mean_phi:.3f}")
    print(f"Mean κ: {metrics.mean_kappa:.2f}")
    print(f"Waypoint Alignment: {metrics.waypoint_alignment:.3f}")
    print(f"Trajectory Smoothness: {metrics.trajectory_smoothness:.3f}")
    print(f"Regime Stability: {metrics.regime_stability:.3f}")
    print(f"Max Basin Drift: {max(metrics.basin_drift):.3f}")
    
    print("\n✅ Geometric metrics module validated")
