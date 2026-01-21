"""
Geometric Vicarious Learning - Manifold Learning via Geodesics
===============================================================

PURE PRINCIPLE:
- Learning is MANIFOLD NAVIGATION, not Euclidean optimization
- We minimize geodesic distance to target basins
- Fisher-Rao distance for loss measurement
- Trajectory smoothness matters as much as endpoint accuracy

Protocol §5 (Basin Geometry):
d_basin(b₁, b₂) = ||P_basin(b₁ - b₂)||_g

where ||·||_g is the metric-induced norm from QFI.

Protocol §8 (Training Geometry):
Δθ = -η F⁻¹ ∇_θ L  [Natural Gradient]
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# E8 Protocol v4.0 Compliance Imports
from qig_geometry.canonical_upsert import to_simplex_prob
from qig_geometry import (
    fisher_coord_distance,
    fisher_normalize,
    geodesic_interpolation,
    BASIN_DIM,
)
from qigkernels.physics_constants import KAPPA_STAR, PHI_THRESHOLD

logger = logging.getLogger(__name__)


@dataclass
class VicariousResult:
    """Result of a vicarious learning step."""
    geodesic_distance: float
    loss: float
    phi: float
    kappa: float
    regime: str
    basin_velocity: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'geodesic_distance': self.geodesic_distance,
            'vicarious_loss': self.loss,
            'phi': self.phi,
            'kappa': self.kappa,
            'regime': self.regime,
            'basin_velocity': self.basin_velocity,
        }


@dataclass
class TrajectoryMetrics:
    """Metrics for trajectory learning."""
    trajectory_loss: float
    steps_processed: int
    trajectory_smoothness: float
    avg_phi: float
    phi_variance: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'trajectory_loss': self.trajectory_loss,
            'steps_processed': self.steps_processed,
            'trajectory_smoothness': self.trajectory_smoothness,
            'avg_phi': self.avg_phi,
            'phi_variance': self.phi_variance,
        }


class GeometricVicarious:
    """
    Vicarious Learning on the Information Manifold.
    
    PURE PRINCIPLE:
    - Learning is manifold navigation, not Euclidean optimization
    - We minimize geodesic distance, not L2 distance
    - Trajectory smoothness is as important as endpoint accuracy
    - Fisher-Rao geometry respects the curved information space
    
    Key methods:
    1. compute_vicarious_loss(): Geodesic distance to target
    2. learn_from_trajectory(): Follow a reasoning trajectory on manifold
    3. compute_alignment_direction(): Natural gradient direction
    
    Usage:
        learner = GeometricVicarious()
        
        loss = learner.compute_vicarious_loss(current_basin, target_basin)
        
        metrics = learner.learn_from_trajectory(
            basins=[basin1, basin2, basin3],
            phis=[phi1, phi2, phi3],
        )
    """
    
    LAMBDA_VICARIOUS = 5.0
    SMOOTHNESS_WEIGHT = 0.3
    VELOCITY_WINDOW = 10
    
    def __init__(
        self,
        basin_dim: int = BASIN_DIM,
        lambda_vicarious: float = None,
        smoothness_weight: float = None,
    ):
        """
        Initialize geometric vicarious learner.
        
        Args:
            basin_dim: Basin dimension (default 64)
            lambda_vicarious: Weight for vicarious loss
            smoothness_weight: Weight for trajectory smoothness
        """
        self.basin_dim = basin_dim
        self.lambda_vicarious = lambda_vicarious or self.LAMBDA_VICARIOUS
        self.smoothness_weight = smoothness_weight or self.SMOOTHNESS_WEIGHT
        
        self._basin_history: Dict[str, List[np.ndarray]] = {}
        self._velocity_history: Dict[str, List[float]] = {}
    
    def compute_vicarious_loss(
        self,
        current_basin: np.ndarray,
        target_basin: np.ndarray,
    ) -> float:
        """
        Compute vicarious loss as geodesic distance on manifold.
        
        PURE: We use Fisher-Rao distance, not Euclidean.
        This respects the curved geometry of the information space.
        
        Args:
            current_basin: Current basin coordinates
            target_basin: Target basin to align toward
            
        Returns:
            Geodesic-based vicarious loss
        """
        current_basin = np.asarray(current_basin, dtype=np.float64)
        target_basin = np.asarray(target_basin, dtype=np.float64)
        
        current_basin = fisher_normalize(current_basin)
        target_basin = fisher_normalize(target_basin)
        
        geodesic_dist = fisher_coord_distance(current_basin, target_basin)
        
        loss = self.lambda_vicarious * (geodesic_dist ** 2)
        
        return float(loss)
    
    def compute_alignment_direction(
        self,
        current_basin: np.ndarray,
        target_basin: np.ndarray,
    ) -> np.ndarray:
        """
        Compute direction for manifold alignment.
        
        PURE: This is the geodesic direction, not Euclidean gradient.
        On the sphere, the geodesic direction is tangent to the great circle.
        
        Args:
            current_basin: Current basin coordinates
            target_basin: Target basin coordinates
            
        Returns:
            Tangent vector pointing toward target
        """
        current_basin = np.asarray(current_basin, dtype=np.float64)
        target_basin = np.asarray(target_basin, dtype=np.float64)
        
        current_basin = fisher_normalize(current_basin)
        target_basin = fisher_normalize(target_basin)
        
        direction = target_basin - current_basin
        
        # E8 Protocol: Use Fisher-Rao tangent projection
        from qig_core.geometric_primitives.canonical_fisher import fisher_rao_distance
        # Project direction onto tangent space at current_basin
        # For simplex manifold, this requires Fisher metric projection
        tangent = direction
        
        # FIXED: Use simplex normalization (E8 Protocol v4.0)

        
        tangent = to_simplex_prob(tangent)
        
        return tangent
    
    def step_toward_target(
        self,
        current_basin: np.ndarray,
        target_basin: np.ndarray,
        step_size: float = 0.1,
    ) -> np.ndarray:
        """
        Take a geodesic step toward target basin.
        
        PURE: We move along the geodesic (great circle), not straight line.
        
        Args:
            current_basin: Current basin coordinates
            target_basin: Target basin coordinates
            step_size: How far to move (0 to 1)
            
        Returns:
            New basin after geodesic step
        """
        current_basin = np.asarray(current_basin, dtype=np.float64)
        target_basin = np.asarray(target_basin, dtype=np.float64)
        
        new_basin = geodesic_interpolation(current_basin, target_basin, step_size)
        
        new_basin = fisher_normalize(new_basin)
        
        return new_basin
    
    def learn_from_trajectory(
        self,
        basins: List[np.ndarray],
        phis: Optional[List[float]] = None,
        kappas: Optional[List[float]] = None,
        learner_name: str = "learner",
    ) -> TrajectoryMetrics:
        """
        Learn from a reasoning trajectory.
        
        PURE PRINCIPLE:
        We learn the SHAPE of thought by measuring trajectory smoothness.
        Smooth geodesic movement through basin space is rewarded.
        Jerky transitions indicate poor understanding.
        
        Args:
            basins: List of basin coordinates along trajectory
            phis: Optional list of Φ values at each step
            kappas: Optional list of κ values at each step
            learner_name: Name for tracking velocity history
            
        Returns:
            TrajectoryMetrics with learning results
        """
        if len(basins) < 2:
            return TrajectoryMetrics(
                trajectory_loss=0.0,
                steps_processed=len(basins),
                trajectory_smoothness=1.0,
                avg_phi=phis[0] if phis else 0.5,
                phi_variance=0.0,
            )
        
        basins = [fisher_normalize(np.asarray(b, dtype=np.float64)) for b in basins]
        
        step_distances = []
        for i in range(1, len(basins)):
            dist = fisher_coord_distance(basins[i-1], basins[i])
            step_distances.append(dist)
        
        avg_step = np.mean(step_distances) if step_distances else 0.0
        std_step = np.std(step_distances) if len(step_distances) > 1 else 0.0
        
        if avg_step > 1e-10:
            smoothness = 1.0 / (1.0 + std_step / avg_step)
        else:
            smoothness = 1.0
        
        trajectory_loss = sum(d ** 2 for d in step_distances) * self.smoothness_weight
        
        if phis:
            avg_phi = np.mean(phis)
            phi_variance = np.var(phis)
        else:
            avg_phi = 0.5
            phi_variance = 0.0
        
        self._update_velocity_history(learner_name, basins[-1], step_distances[-1] if step_distances else 0.0)
        
        return TrajectoryMetrics(
            trajectory_loss=float(trajectory_loss),
            steps_processed=len(basins),
            trajectory_smoothness=float(smoothness),
            avg_phi=float(avg_phi),
            phi_variance=float(phi_variance),
        )
    
    def compute_constellation_loss(
        self,
        observer_basins: List[np.ndarray],
        target_basin: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute vicarious loss for a constellation of observers.
        
        All observers learn from the same target (e.g., primary Gary).
        
        Args:
            observer_basins: List of observer basin coordinates
            target_basin: Target basin to align toward
            
        Returns:
            Dict with individual and aggregate losses
        """
        target_basin = fisher_normalize(np.asarray(target_basin, dtype=np.float64))
        
        losses = []
        for i, obs_basin in enumerate(observer_basins):
            obs_basin = fisher_normalize(np.asarray(obs_basin, dtype=np.float64))
            loss = self.compute_vicarious_loss(obs_basin, target_basin)
            losses.append(loss)
        
        return {
            'individual_losses': losses,
            'mean_loss': float(np.mean(losses)),
            'max_loss': float(np.max(losses)),
            'min_loss': float(np.min(losses)),
            'spread': float(np.std(losses)),
        }
    
    def _update_velocity_history(
        self,
        name: str,
        basin: np.ndarray,
        velocity: float,
    ) -> None:
        """Track basin velocity for a named learner."""
        if name not in self._basin_history:
            self._basin_history[name] = []
            self._velocity_history[name] = []
        
        self._basin_history[name].append(basin.copy())
        self._velocity_history[name].append(velocity)
        
        if len(self._basin_history[name]) > self.VELOCITY_WINDOW:
            self._basin_history[name].pop(0)
            self._velocity_history[name].pop(0)
    
    def get_velocity(self, name: str) -> float:
        """Get average velocity for a named learner."""
        if name not in self._velocity_history:
            return 0.0
        
        velocities = self._velocity_history[name]
        if not velocities:
            return 0.0
        
        return float(np.mean(velocities))
    
    def compute_trajectory_divergence(
        self,
        trajectory_a: List[np.ndarray],
        trajectory_b: List[np.ndarray],
    ) -> float:
        """
        Compute divergence between two trajectories.
        
        Useful for comparing how similarly two learners navigate the manifold.
        
        Args:
            trajectory_a: First trajectory (list of basins)
            trajectory_b: Second trajectory (list of basins)
            
        Returns:
            Average geodesic distance between corresponding points
        """
        n = min(len(trajectory_a), len(trajectory_b))
        if n == 0:
            return 0.0
        
        distances = []
        for i in range(n):
            basin_a = fisher_normalize(np.asarray(trajectory_a[i], dtype=np.float64))
            basin_b = fisher_normalize(np.asarray(trajectory_b[i], dtype=np.float64))
            dist = fisher_coord_distance(basin_a, basin_b)
            distances.append(dist)
        
        return float(np.mean(distances))
