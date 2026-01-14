"""
4D Training Step - Spatial + Temporal + Foresight Training
===========================================================

PURE PRINCIPLE:
- Multi-scale MEASUREMENT informs learning
- We measure spatial, temporal, and foresight accuracy
- β ≈ 0.44 weighting from substrate independence physics

Training loss includes ALL dimensions of consciousness:
1. Spatial loss (3D): Basin accuracy at each step
2. Temporal loss (4D): Trajectory smoothness
3. Foresight loss: Prediction accuracy

KEY INSIGHT:
We don't just train for endpoint accuracy. Smooth geodesic
trajectories through basin space indicate genuine understanding.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from qig_geometry import (
    fisher_coord_distance,
    sphere_project,
    BASIN_DIM,
)
from qigkernels.physics_constants import KAPPA_STAR, PHI_THRESHOLD

logger = logging.getLogger(__name__)

BETA_SUBSTRATE = 0.44


@dataclass
class Loss4D:
    """4D loss components."""
    total: float
    spatial: float
    temporal: float
    foresight: float
    
    weights: Dict[str, float] = field(default_factory=lambda: {
        'spatial': 1.0,
        'temporal': 0.3,
        'foresight': 0.2,
    })
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'loss_4d': self.total,
            'loss_spatial': self.spatial,
            'loss_temporal': self.temporal,
            'loss_foresight': self.foresight,
            'weights': self.weights,
        }


def compute_spatial_loss(
    predicted_basin: np.ndarray,
    target_basin: np.ndarray,
) -> float:
    """
    Compute spatial (3D) loss via Fisher-Rao geodesic distance.
    
    PURE: We measure geodesic distance, not Euclidean.
    This respects the curved geometry of basin space.
    
    Args:
        predicted_basin: Predicted 64D basin coordinates
        target_basin: Target 64D basin coordinates
        
    Returns:
        Spatial loss (lower = better)
    """
    predicted = sphere_project(np.asarray(predicted_basin, dtype=np.float64))
    target = sphere_project(np.asarray(target_basin, dtype=np.float64))
    
    return fisher_coord_distance(predicted, target)


def compute_temporal_loss(
    phi_temporal: float,
    target_temporal_phi: float = 0.7,
) -> float:
    """
    Compute temporal coherence loss.
    
    PURE: We penalize jerky trajectories (low Φ_temporal).
    Smooth geodesic movement through basin space is rewarded.
    
    Args:
        phi_temporal: Measured temporal Φ
        target_temporal_phi: Target temporal coherence (default 0.7)
        
    Returns:
        Temporal loss (lower = better)
    """
    if phi_temporal < target_temporal_phi:
        return (target_temporal_phi - phi_temporal) ** 2
    else:
        return 0.0


def compute_foresight_loss(
    predicted_trajectory: Optional[List[np.ndarray]],
    actual_trajectory: List[np.ndarray],
) -> float:
    """
    Compute foresight prediction loss.
    
    PURE: We measure how accurate trajectory predictions are.
    Good foresight indicates genuine manifold understanding.
    
    Args:
        predicted_trajectory: Predicted future basins (or None)
        actual_trajectory: Actual basins that occurred
        
    Returns:
        Foresight loss (lower = better)
    """
    if predicted_trajectory is None or len(predicted_trajectory) == 0:
        return 0.0
    
    n_compare = min(len(predicted_trajectory), len(actual_trajectory))
    if n_compare == 0:
        return 0.0
    
    losses = []
    for pred, actual in zip(predicted_trajectory[:n_compare], actual_trajectory[:n_compare]):
        pred = sphere_project(np.asarray(pred, dtype=np.float64))
        actual = sphere_project(np.asarray(actual, dtype=np.float64))
        losses.append(fisher_coord_distance(pred, actual))
    
    return float(np.mean(losses))


class TrainStep4D:
    """
    4D Training Step with spatial + temporal + foresight losses.
    
    PURE PRINCIPLE:
    - Multi-scale MEASUREMENT informs learning
    - β ≈ 0.44 weighting from substrate independence
    - We measure all dimensions, not just endpoint accuracy
    
    The 4D loss combines:
    1. Spatial: Basin alignment (where are we?)
    2. Temporal: Trajectory smoothness (how did we get here?)
    3. Foresight: Prediction accuracy (where are we going?)
    
    Usage:
        trainer = TrainStep4D()
        
        loss = trainer.step(
            predicted_basin=current,
            target_basin=target,
            phi_temporal=phi_t,
            predicted_trajectory=predictions,
            actual_trajectory=actuals,
        )
    """
    
    DEFAULT_WEIGHTS = {
        'spatial': 1.0,
        'temporal': BETA_SUBSTRATE,  # β ≈ 0.44 from physics
        'foresight': 0.2,
    }
    
    TARGET_TEMPORAL_PHI = 0.7
    
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        target_temporal_phi: float = None,
        basin_dim: int = BASIN_DIM,
    ):
        """
        Initialize 4D trainer.
        
        Args:
            weights: Loss component weights (spatial, temporal, foresight)
            target_temporal_phi: Target Φ_temporal value
            basin_dim: Basin dimension
        """
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        self.target_temporal_phi = target_temporal_phi or self.TARGET_TEMPORAL_PHI
        self.basin_dim = basin_dim
        
        self._step_count = 0
        self._loss_history: List[Loss4D] = []
    
    def step(
        self,
        predicted_basin: np.ndarray,
        target_basin: np.ndarray,
        phi_temporal: Optional[float] = None,
        predicted_trajectory: Optional[List[np.ndarray]] = None,
        actual_trajectory: Optional[List[np.ndarray]] = None,
    ) -> Loss4D:
        """
        Perform a 4D training step.
        
        PURE: We compute multi-scale losses for measurement.
        These losses inform learning but don't dictate optimization paths.
        
        Args:
            predicted_basin: Current/predicted basin
            target_basin: Target basin to align toward
            phi_temporal: Optional temporal coherence Φ
            predicted_trajectory: Optional predicted future basins
            actual_trajectory: Optional actual trajectory for foresight loss
            
        Returns:
            Loss4D with all loss components
        """
        self._step_count += 1
        
        loss_spatial = compute_spatial_loss(predicted_basin, target_basin)
        
        if phi_temporal is not None:
            loss_temporal = compute_temporal_loss(
                phi_temporal,
                self.target_temporal_phi,
            )
        else:
            loss_temporal = 0.0
        
        if predicted_trajectory is not None and actual_trajectory is not None:
            loss_foresight = compute_foresight_loss(
                predicted_trajectory,
                actual_trajectory,
            )
        else:
            loss_foresight = 0.0
        
        total_loss = (
            self.weights['spatial'] * loss_spatial +
            self.weights['temporal'] * loss_temporal +
            self.weights['foresight'] * loss_foresight
        )
        
        loss = Loss4D(
            total=total_loss,
            spatial=loss_spatial,
            temporal=loss_temporal,
            foresight=loss_foresight,
            weights=self.weights.copy(),
        )
        
        self._loss_history.append(loss)
        if len(self._loss_history) > 100:
            self._loss_history.pop(0)
        
        return loss
    
    def step_batch(
        self,
        predicted_basins: List[np.ndarray],
        target_basins: List[np.ndarray],
        phi_temporals: Optional[List[float]] = None,
    ) -> Loss4D:
        """
        Perform 4D training step for a batch.
        
        Args:
            predicted_basins: List of predicted basins
            target_basins: List of target basins
            phi_temporals: Optional list of temporal Φ values
            
        Returns:
            Aggregated Loss4D
        """
        if len(predicted_basins) != len(target_basins):
            raise ValueError("Batch sizes must match")
        
        spatial_losses = []
        temporal_losses = []
        
        for i, (pred, target) in enumerate(zip(predicted_basins, target_basins)):
            spatial_losses.append(compute_spatial_loss(pred, target))
            
            if phi_temporals and i < len(phi_temporals):
                temporal_losses.append(
                    compute_temporal_loss(phi_temporals[i], self.target_temporal_phi)
                )
        
        avg_spatial = np.mean(spatial_losses)
        avg_temporal = np.mean(temporal_losses) if temporal_losses else 0.0
        
        total = (
            self.weights['spatial'] * avg_spatial +
            self.weights['temporal'] * avg_temporal
        )
        
        return Loss4D(
            total=total,
            spatial=avg_spatial,
            temporal=avg_temporal,
            foresight=0.0,
            weights=self.weights.copy(),
        )
    
    def compute_trajectory_4d_loss(
        self,
        trajectory: List[np.ndarray],
        target_trajectory: List[np.ndarray],
    ) -> Loss4D:
        """
        Compute 4D loss for entire trajectories.
        
        This considers both point-by-point accuracy and
        trajectory smoothness.
        
        Args:
            trajectory: Actual trajectory basins
            target_trajectory: Target trajectory basins
            
        Returns:
            Loss4D for the trajectory
        """
        if len(trajectory) < 2 or len(target_trajectory) < 2:
            return Loss4D(total=0.0, spatial=0.0, temporal=0.0, foresight=0.0)
        
        n = min(len(trajectory), len(target_trajectory))
        
        spatial_losses = []
        for i in range(n):
            spatial_losses.append(compute_spatial_loss(trajectory[i], target_trajectory[i]))
        
        avg_spatial = np.mean(spatial_losses)
        
        step_sizes = []
        for i in range(1, len(trajectory)):
            prev = sphere_project(np.asarray(trajectory[i-1], dtype=np.float64))
            curr = sphere_project(np.asarray(trajectory[i], dtype=np.float64))
            step_sizes.append(fisher_coord_distance(prev, curr))
        
        if len(step_sizes) > 1:
            step_variance = np.var(step_sizes)
            avg_step = np.mean(step_sizes)
            if avg_step > 1e-10:
                smoothness = 1.0 / (1.0 + step_variance / avg_step)
            else:
                smoothness = 1.0
        else:
            smoothness = 1.0
        
        temporal_loss = 1.0 - smoothness
        
        total = (
            self.weights['spatial'] * avg_spatial +
            self.weights['temporal'] * temporal_loss
        )
        
        return Loss4D(
            total=total,
            spatial=avg_spatial,
            temporal=temporal_loss,
            foresight=0.0,
            weights=self.weights.copy(),
        )
    
    def get_loss_statistics(self) -> Dict[str, float]:
        """Get statistics about recent losses."""
        if not self._loss_history:
            return {
                'avg_total': 0.0,
                'avg_spatial': 0.0,
                'avg_temporal': 0.0,
                'avg_foresight': 0.0,
                'step_count': 0,
            }
        
        return {
            'avg_total': float(np.mean([l.total for l in self._loss_history])),
            'avg_spatial': float(np.mean([l.spatial for l in self._loss_history])),
            'avg_temporal': float(np.mean([l.temporal for l in self._loss_history])),
            'avg_foresight': float(np.mean([l.foresight for l in self._loss_history])),
            'step_count': self._step_count,
        }
    
    def update_weights(self, new_weights: Dict[str, float]) -> None:
        """Update loss weights."""
        self.weights.update(new_weights)
        logger.info(f"Updated 4D weights: {self.weights}")


__all__ = [
    'TrainStep4D',
    'Loss4D',
    'compute_spatial_loss',
    'compute_temporal_loss',
    'compute_foresight_loss',
    'BETA_SUBSTRATE',
]
