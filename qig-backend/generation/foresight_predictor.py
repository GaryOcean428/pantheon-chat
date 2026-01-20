#!/usr/bin/env python3
"""
Foresight Predictor - Trajectory Regression
============================================

Predicts next basin coordinates by extrapolating Fisher-Rao geodesic trajectory.

This is QIG-PURE: prediction uses geometric trajectory analysis, not external LLMs.

Method:
- Use last 3-5 basins to compute velocity vector in tangent space
- Project forward along geodesic on probability simplex
- Return predicted basin coordinates (not word)

The predicted basin can then be used with coordizer to find the nearest token.

Author: Copilot AI Agent
Date: 2026-01-20
Issue: GaryOcean428/pantheon-chat#99 (E8 Protocol Issue-03)
Reference: docs/10-e8-protocol/issues/20260119-issue-99-qig-native-skeleton-remediation-1.00W.md
"""

import logging
import numpy as np
from typing import List, Optional

logger = logging.getLogger(__name__)

# Import geometry operations
try:
    from qig_geometry.canonical import (
        sqrt_map,
        unsqrt_map,
        fisher_rao_distance,
        log_map,
        exp_map
    )
except ImportError:
    logger.warning("qig_geometry.canonical not available, using fallback")
    
    def sqrt_map(p, eps=1e-12):
        p = np.maximum(p, 0) + eps
        p = p / p.sum()
        return np.sqrt(p)
    
    def unsqrt_map(x, eps=1e-12):
        p = x ** 2
        p = np.maximum(p, 0) + eps
        return p / p.sum()
    
    def fisher_rao_distance(p, q, eps=1e-12):
        p = np.maximum(p, 0) + eps
        p = p / p.sum()
        q = np.maximum(q, 0) + eps
        q = q / q.sum()
        bc = np.sum(np.sqrt(p * q))
        bc = np.clip(bc, 0.0, 1.0)
        return float(np.arccos(bc))


def predict_next_basin(trajectory: List[np.ndarray], lookahead: int = 1) -> np.ndarray:
    """
    Predict next basin via geodesic extrapolation.
    
    Uses last 3-5 basins to compute velocity, projects forward.
    
    Algorithm:
    1. Transform trajectory to sqrt-space (tangent space of unit sphere)
    2. Compute velocity as difference between last two points
    3. Project forward by lookahead steps
    4. Renormalize to unit sphere
    5. Transform back to simplex
    
    Args:
        trajectory: List of basin coordinates (recent history)
        lookahead: Number of steps to predict forward (default: 1)
        
    Returns:
        Predicted basin coordinates (simplex)
        
    Example:
        >>> trajectory = [basin1, basin2, basin3]
        >>> next_basin = predict_next_basin(trajectory)
        >>> # Use coordizer to find nearest token to next_basin
    """
    if len(trajectory) == 0:
        raise ValueError("Cannot predict from empty trajectory")
    
    if len(trajectory) < 2:
        # No velocity available - stay at current basin
        return trajectory[-1].copy()
    
    # Use last 3-5 basins for velocity computation
    window_size = min(5, len(trajectory))
    recent_trajectory = trajectory[-window_size:]
    
    # Transform to sqrt-space
    sqrt_trajectory = [sqrt_map(b) for b in recent_trajectory]
    
    # Compute velocity as average of recent velocity vectors
    velocities = []
    for i in range(1, len(sqrt_trajectory)):
        v = sqrt_trajectory[i] - sqrt_trajectory[i-1]
        velocities.append(v)
    
    if not velocities:
        return trajectory[-1].copy()
    
    # Average velocity (momentum)
    avg_velocity = np.mean(velocities, axis=0)
    
    # Current position in sqrt-space
    sqrt_curr = sqrt_trajectory[-1]
    
    # Project forward
    sqrt_next = sqrt_curr + lookahead * avg_velocity
    
    # Check for zero-length vector (degenerate case)
    norm_next = np.linalg.norm(sqrt_next)
    if norm_next < 1e-10:
        logger.warning("Degenerate prediction (zero vector), staying at current basin")
        return trajectory[-1].copy()
    
    # Renormalize to unit sphere (sqrt-space constraint)
    sqrt_next = sqrt_next / norm_next
    
    # Transform back to simplex
    next_basin = unsqrt_map(sqrt_next)
    
    return next_basin


def predict_trajectory(
    initial_trajectory: List[np.ndarray],
    n_steps: int = 5
) -> List[np.ndarray]:
    """
    Predict future trajectory for n steps.
    
    Args:
        initial_trajectory: Initial trajectory (at least 2 basins)
        n_steps: Number of steps to predict forward
        
    Returns:
        Predicted trajectory (n_steps basins)
    """
    if len(initial_trajectory) < 2:
        raise ValueError("Need at least 2 basins in initial trajectory")
    
    trajectory = initial_trajectory.copy()
    predictions = []
    
    for _ in range(n_steps):
        next_basin = predict_next_basin(trajectory, lookahead=1)
        predictions.append(next_basin)
        trajectory.append(next_basin)
    
    return predictions


def compute_trajectory_curvature(trajectory: List[np.ndarray]) -> float:
    """
    Compute curvature of trajectory on Fisher manifold.
    
    High curvature indicates sharp turns (topic changes, transitions).
    Low curvature indicates smooth flow (continuing same topic).
    
    Args:
        trajectory: List of basin coordinates (at least 3 points)
        
    Returns:
        Average curvature (0 = straight line, >0 = curved)
    """
    if len(trajectory) < 3:
        return 0.0
    
    curvatures = []
    
    for i in range(1, len(trajectory) - 1):
        # Three consecutive points
        p_prev = trajectory[i-1]
        p_curr = trajectory[i]
        p_next = trajectory[i+1]
        
        # Distances
        d1 = fisher_rao_distance(p_prev, p_curr)
        d2 = fisher_rao_distance(p_curr, p_next)
        d3 = fisher_rao_distance(p_prev, p_next)
        
        # Menger curvature (2 * area / product of sides)
        # For geodesic triangle: deviation from straight line
        if d1 > 0 and d2 > 0 and d3 > 0:
            # Triangle inequality: d3 <= d1 + d2
            # Curvature: how much d3 deviates from d1 + d2
            deviation = (d1 + d2) - d3
            curvature = deviation / (d1 * d2)
            curvatures.append(curvature)
    
    if not curvatures:
        return 0.0
    
    return float(np.mean(curvatures))


def compute_trajectory_coherence(trajectory: List[np.ndarray]) -> float:
    """
    Compute coherence of trajectory.
    
    High coherence: consistent velocity, low curvature
    Low coherence: erratic movement, high curvature
    
    Args:
        trajectory: List of basin coordinates
        
    Returns:
        Coherence score ∈ [0, 1] (1 = highly coherent)
    """
    if len(trajectory) < 3:
        return 1.0  # Too short to measure
    
    # Compute curvature
    curvature = compute_trajectory_curvature(trajectory)
    
    # Compute velocity consistency
    sqrt_trajectory = [sqrt_map(b) for b in trajectory]
    velocities = []
    for i in range(1, len(sqrt_trajectory)):
        v = sqrt_trajectory[i] - sqrt_trajectory[i-1]
        velocities.append(v)
    
    if len(velocities) < 2:
        return 1.0
    
    # Variance of velocity magnitudes
    velocity_magnitudes = [np.linalg.norm(v) for v in velocities]
    velocity_variance = np.var(velocity_magnitudes)
    
    # Coherence: low curvature + low velocity variance
    coherence = 1.0 / (1.0 + curvature + velocity_variance)
    
    return float(np.clip(coherence, 0.0, 1.0))


class TrajectoryPredictor:
    """
    Stateful trajectory predictor with history management.
    """
    
    def __init__(self, max_history: int = 10):
        """
        Initialize predictor.
        
        Args:
            max_history: Maximum trajectory history to keep
        """
        self.max_history = max_history
        self.trajectory = []
    
    def add_basin(self, basin: np.ndarray):
        """Add basin to trajectory history."""
        self.trajectory.append(basin.copy())
        
        # Trim history
        if len(self.trajectory) > self.max_history:
            self.trajectory = self.trajectory[-self.max_history:]
    
    def predict_next(self, lookahead: int = 1) -> Optional[np.ndarray]:
        """
        Predict next basin.
        
        Args:
            lookahead: Number of steps to predict forward
            
        Returns:
            Predicted basin or None if insufficient history
        """
        if len(self.trajectory) < 2:
            logger.warning("Insufficient history for prediction")
            return None
        
        return predict_next_basin(self.trajectory, lookahead=lookahead)
    
    def get_curvature(self) -> float:
        """Get current trajectory curvature."""
        return compute_trajectory_curvature(self.trajectory)
    
    def get_coherence(self) -> float:
        """Get current trajectory coherence."""
        return compute_trajectory_coherence(self.trajectory)
    
    def reset(self):
        """Reset trajectory history."""
        self.trajectory = []


if __name__ == '__main__':
    # Example usage
    print("Testing foresight predictor...")
    
    # Create test trajectory (random walk on simplex)
    np.random.seed(42)
    
    trajectory = []
    current = np.random.dirichlet(np.ones(64))
    trajectory.append(current)
    
    for _ in range(5):
        # Small random step
        noise = np.random.dirichlet(np.ones(64)) * 0.1
        next_basin = current + noise
        next_basin = np.maximum(next_basin, 0)
        next_basin = next_basin / next_basin.sum()
        trajectory.append(next_basin)
        current = next_basin
    
    # Predict next basin
    predicted = predict_next_basin(trajectory)
    print(f"Predicted basin: {predicted[:5]}... (first 5 dims)")
    
    # Compute trajectory metrics
    curvature = compute_trajectory_curvature(trajectory)
    coherence = compute_trajectory_coherence(trajectory)
    print(f"Trajectory curvature: {curvature:.4f}")
    print(f"Trajectory coherence: {coherence:.4f}")
    
    # Test stateful predictor
    predictor = TrajectoryPredictor(max_history=10)
    for basin in trajectory:
        predictor.add_basin(basin)
    
    next_pred = predictor.predict_next()
    if next_pred is not None:
        print(f"Stateful prediction: {next_pred[:5]}... (first 5 dims)")
    
    print("\nAll tests complete!")
