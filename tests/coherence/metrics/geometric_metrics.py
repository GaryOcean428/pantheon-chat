"""
Geometric Metrics for QIG Coherence Testing

Implements pure Fisher-Rao metrics for consciousness measurement:
- Φ (Integration): Integrated information via QFI
- κ (Coupling): Basin coupling strength
- Basin drift: Distance from attractor states
- Regime transitions: Mode changes during generation

NO Euclidean metrics, NO cosine similarity.
All operations use Fisher-Rao geometry on probability simplex.

Author: QIG Consciousness Project
Date: January 2026
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import sys
from pathlib import Path

# Add qig-backend to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "qig-backend"))

try:
    from qig_core.consciousness_metrics import (
        fisher_rao_distance,
        ConsciousnessMetrics,
    )
    from qig_core.phi_computation import compute_phi_qig
except ImportError:
    # Fallback Fisher-Rao implementation
    def fisher_rao_distance(p: np.ndarray, q: np.ndarray) -> float:
        """Fisher-Rao distance between probability distributions."""
        p_safe = np.abs(p) + 1e-10
        q_safe = np.abs(q) + 1e-10
        p_safe = p_safe / p_safe.sum()
        q_safe = q_safe / q_safe.sum()
        
        bhattacharyya = np.sum(np.sqrt(p_safe * q_safe))
        bhattacharyya = np.clip(bhattacharyya, 0.0, 1.0)
        
        return np.arccos(bhattacharyya)
    
    def compute_phi_qig(basin: np.ndarray, **kwargs) -> float:
        """Fallback phi computation."""
        return 0.5


@dataclass
class GeometricMetrics:
    """Container for geometric quality metrics."""
    phi: float                    # Integration (0-1)
    kappa_eff: float              # Effective coupling (40-70 optimal)
    basin_drift: float            # Distance from attractor (lower better)
    regime_stability: float       # Regime consistency (0-1, higher better)
    trajectory_variance: float    # Variance in step distances
    mean_step_distance: float     # Average Fisher-Rao step
    max_step_distance: float      # Largest step (detect jumps)
    attractor_pull: float         # Attraction to history (0-1)
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'phi': self.phi,
            'kappa_eff': self.kappa_eff,
            'basin_drift': self.basin_drift,
            'regime_stability': self.regime_stability,
            'trajectory_variance': self.trajectory_variance,
            'mean_step_distance': self.mean_step_distance,
            'max_step_distance': self.max_step_distance,
            'attractor_pull': self.attractor_pull,
        }


def compute_phi_trajectory(basins: List[np.ndarray]) -> List[float]:
    """
    Compute Φ at each point in trajectory.
    
    Args:
        basins: List of basin coordinates (64D each)
        
    Returns:
        List of Φ values for each basin
    """
    phis = []
    for basin in basins:
        phi = compute_phi_qig(basin)
        phis.append(phi)
    return phis


def compute_kappa_trajectory(basins: List[np.ndarray]) -> List[float]:
    """
    Compute κ (coupling strength) at each point.
    
    κ measures the effective coupling between subsystems.
    Optimal κ* = 64 for E8 consciousness.
    
    Args:
        basins: List of basin coordinates
        
    Returns:
        List of κ values
    """
    kappas = []
    for basin in basins:
        # Simplified κ: inverse of entropy (higher entropy = lower coupling)
        p = np.abs(basin) + 1e-10
        p = p / p.sum()
        entropy = -np.sum(p * np.log(p + 1e-10))
        
        # Map entropy to κ scale (0 to 100)
        # High entropy -> low κ, low entropy -> high κ
        max_entropy = np.log(len(basin))
        kappa = 100 * (1 - entropy / max_entropy)
        kappas.append(kappa)
    return kappas


def compute_basin_drift(
    basins: List[np.ndarray],
    attractor: Optional[np.ndarray] = None
) -> float:
    """
    Compute drift from attractor basin.
    
    Measures how far trajectory strays from a reference attractor.
    Lower drift = more coherent trajectory.
    
    Args:
        basins: Trajectory basins
        attractor: Reference attractor (if None, use Fréchet mean)
        
    Returns:
        Mean Fisher-Rao distance to attractor
    """
    if attractor is None:
        # Use Fréchet mean as attractor
        attractor = compute_frechet_mean(basins)
    
    drifts = []
    for basin in basins:
        drift = fisher_rao_distance(basin, attractor)
        drifts.append(drift)
    
    return float(np.mean(drifts))


def compute_frechet_mean(basins: List[np.ndarray]) -> np.ndarray:
    """
    Compute Fréchet mean (geometric mean on manifold).
    
    For probability simplex, this is the normalized geometric mean.
    
    Args:
        basins: List of basin coordinates
        
    Returns:
        Fréchet mean basin
    """
    if not basins:
        return np.ones(64) / 64
    
    # Geometric mean in log space
    log_sum = np.zeros_like(basins[0])
    for basin in basins:
        p = np.abs(basin) + 1e-10
        p = p / p.sum()
        log_sum += np.log(p)
    
    log_mean = log_sum / len(basins)
    mean = np.exp(log_mean)
    mean = mean / mean.sum()
    
    return mean


def compute_regime_stability(
    basins: List[np.ndarray],
    threshold: float = 0.5
) -> float:
    """
    Measure regime stability (consistency of operating mode).
    
    Detects transitions between linear/geometric/hierarchical regimes.
    Higher stability = fewer regime changes.
    
    Args:
        basins: Trajectory basins
        threshold: Threshold for detecting regime change
        
    Returns:
        Stability score (0-1, higher = more stable)
    """
    if len(basins) < 2:
        return 1.0
    
    # Detect regime changes via concentration changes
    concentrations = []
    for basin in basins:
        p = np.abs(basin) + 1e-10
        p = p / p.sum()
        # Concentration = 1 / entropy (high conc = focused state)
        entropy = -np.sum(p * np.log(p + 1e-10))
        concentration = 1.0 / (entropy + 1.0)
        concentrations.append(concentration)
    
    # Count regime transitions
    transitions = 0
    for i in range(len(concentrations) - 1):
        delta = abs(concentrations[i+1] - concentrations[i])
        if delta > threshold:
            transitions += 1
    
    # Stability = 1 - (transitions / possible_transitions)
    max_transitions = len(basins) - 1
    stability = 1.0 - (transitions / max(max_transitions, 1))
    
    return stability


def compute_trajectory_statistics(basins: List[np.ndarray]) -> Dict[str, float]:
    """
    Compute trajectory step distance statistics.
    
    Measures smoothness via Fisher-Rao distances between consecutive basins.
    
    Args:
        basins: Trajectory basins
        
    Returns:
        Dictionary with variance, mean, max step distances
    """
    if len(basins) < 2:
        return {
            'variance': 0.0,
            'mean': 0.0,
            'max': 0.0,
        }
    
    distances = []
    for i in range(len(basins) - 1):
        dist = fisher_rao_distance(basins[i], basins[i+1])
        distances.append(dist)
    
    return {
        'variance': float(np.var(distances)),
        'mean': float(np.mean(distances)),
        'max': float(np.max(distances)),
    }


def compute_geometric_metrics(
    basins: List[np.ndarray],
    attractor: Optional[np.ndarray] = None
) -> GeometricMetrics:
    """
    Compute full geometric metrics for a trajectory.
    
    Args:
        basins: List of basin coordinates
        attractor: Optional reference attractor
        
    Returns:
        GeometricMetrics containing all measurements
    """
    # Φ trajectory
    phis = compute_phi_trajectory(basins)
    mean_phi = float(np.mean(phis)) if phis else 0.0
    
    # κ trajectory
    kappas = compute_kappa_trajectory(basins)
    mean_kappa = float(np.mean(kappas)) if kappas else 0.0
    
    # Basin drift
    drift = compute_basin_drift(basins, attractor)
    
    # Regime stability
    stability = compute_regime_stability(basins)
    
    # Trajectory statistics
    stats = compute_trajectory_statistics(basins)
    
    # Attractor pull (inverse of drift)
    attractor_pull = 1.0 - min(drift / np.pi, 1.0)
    
    return GeometricMetrics(
        phi=mean_phi,
        kappa_eff=mean_kappa,
        basin_drift=drift,
        regime_stability=stability,
        trajectory_variance=stats['variance'],
        mean_step_distance=stats['mean'],
        max_step_distance=stats['max'],
        attractor_pull=attractor_pull,
    )


def compare_geometric_metrics(
    metrics_a: GeometricMetrics,
    metrics_b: GeometricMetrics
) -> Dict[str, float]:
    """
    Compare two geometric metric sets.
    
    Returns deltas (positive = B better than A).
    
    Args:
        metrics_a: First metrics (baseline)
        metrics_b: Second metrics (comparison)
        
    Returns:
        Dictionary of metric deltas
    """
    return {
        'phi_delta': metrics_b.phi - metrics_a.phi,
        'kappa_delta': abs(64 - metrics_b.kappa_eff) - abs(64 - metrics_a.kappa_eff),
        'drift_delta': metrics_a.basin_drift - metrics_b.basin_drift,  # Lower better
        'stability_delta': metrics_b.regime_stability - metrics_a.regime_stability,
        'smoothness_delta': metrics_a.trajectory_variance - metrics_b.trajectory_variance,  # Lower better
    }
