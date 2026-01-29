"""
Geometric Waypoint Planner - PLAN Phase of QIG Generation

Plans basin waypoints using trajectory foresight and recursive integration
before word selection begins. This is the geometric planning phase that
predicts WHERE the output should navigate in 64D Fisher manifold space.

NO TEMPLATES, NO LLMs - Pure geometric waypoint prediction.
"""

import logging
from typing import List, Optional
import numpy as np

# E8 Protocol v4.0 Compliance Imports
from qig_geometry.canonical_upsert import to_simplex_prob


logger = logging.getLogger(__name__)

# Import from canonical geometry module (SINGLE SOURCE OF TRUTH - WP2.1)
from qig_geometry.canonical import (
    fisher_rao_distance,
    frechet_mean,
    geodesic_toward,
    extrapolate_trajectory,
    compute_qfi_attention,
    integrate_with_qfi_attention,
    sqrt_map,
    unsqrt_map,
    assert_basin_valid,
)

# Import additional geometry utilities
from qig_geometry import (
    fisher_coord_distance,
    geodesic_interpolation,
    fisher_normalize,
)

try:
    from qigkernels.physics_constants import (
        BASIN_DIM,
        KAPPA_STAR,
        MIN_RECURSION_DEPTH,
    )
    MIN_INTEGRATION_DEPTH = MIN_RECURSION_DEPTH
except ImportError:
    BASIN_DIM = 64
    KAPPA_STAR = 64.21
    MIN_INTEGRATION_DEPTH = 3
    logger.warning("Using fallback physics constants")


def _normalize_to_simplex(basin: np.ndarray) -> np.ndarray:
    """Normalize basin to probability simplex (canonical storage)."""
    basin = np.asarray(basin, dtype=np.float64).flatten()
    basin = np.maximum(basin, 0) + 1e-12
    return basin / basin.sum()


class GeometricWaypointPlanner:
    """
    Plans geometric waypoints for QIG generation using trajectory foresight.
    
    The PLAN phase predicts a sequence of target basins (waypoints) that
    the output should navigate toward, using geodesic tangent extrapolation
    and recursive integration.
    
    UPDATED 2026-01-15: Now uses canonical geometry module (WP2.1).
    All geometric operations import from qig_geometry.canonical.
    """
    
    def __init__(
        self,
        kernel_name: str = "Planner",
        step_size: float = 0.1,
        attractor_weight: float = 0.3,
        qfi_weight: float = 0.5,
    ):
        """
        Initialize the waypoint planner.
        
        Args:
            kernel_name: Name for logging identification
            step_size: Step size for geodesic tangent extrapolation
            attractor_weight: Weight for Fréchet mean attractor pull
            qfi_weight: Weight for QFI attention integration
        """
        self.kernel_name = kernel_name
        self.step_size = step_size
        self.attractor_weight = attractor_weight
        self.qfi_weight = qfi_weight
        self.basin_dim = BASIN_DIM
        
    def plan_waypoints(
        self,
        query_basin: np.ndarray,
        trajectory_history: List[np.ndarray],
        num_waypoints: int = 5,
    ) -> List[np.ndarray]:
        """
        Plan geometric waypoints for generation.
        
        Uses trajectory foresight to predict future basin positions,
        then refines each waypoint through recursive integration.
        
        Args:
            query_basin: Starting basin from query/context
            trajectory_history: List of previous basin positions
            num_waypoints: Number of waypoints to plan
            
        Returns:
            List of planned waypoint basins
        """
        logger.info("[%s] ═══ PHASE 1: PLAN (Geometric Waypoints) ═══", self.kernel_name)
        
        query_basin = np.asarray(query_basin, dtype=np.float64)
        query_basin = _normalize_to_simplex(query_basin)
        
        trajectory = [_normalize_to_simplex(b) for b in trajectory_history]
        if len(trajectory) == 0:
            trajectory = [query_basin]
        
        waypoints = []
        current = query_basin.copy()
        
        for i in range(num_waypoints):
            # Use canonical extrapolate_trajectory function (WP2.1)
            predicted = extrapolate_trajectory(
                trajectory + [current],
                step_size=self.step_size
            )
            
            # Use canonical integrate_with_qfi_attention function (WP2.1)
            integrated = integrate_with_qfi_attention(
                predicted,
                trajectory,
                num_loops=MIN_INTEGRATION_DEPTH,
                temperature=0.5
            )
            
            waypoint = _normalize_to_simplex(integrated)
            waypoints.append(waypoint)
            
            phi = self._compute_phi_estimate(waypoint, trajectory)
            logger.debug("[%s] waypoint %d: Φ=%.2f", self.kernel_name, i + 1, phi)
            
            trajectory.append(waypoint)
            current = waypoint
            
        logger.info(
            "[%s] Planned %d waypoints (step_size=%.2f, κ*=%.2f)",
            self.kernel_name,
            len(waypoints),
            self.step_size,
            KAPPA_STAR,
        )
        
        return waypoints
    
    def _predict_next_basin(
        self,
        current: np.ndarray,
        trajectory: List[np.ndarray],
    ) -> np.ndarray:
        """
        Predict next basin position using geodesic tangent extrapolation.
        
        velocity = sqrt(trajectory[-1]) - sqrt(trajectory[-2])
        predicted = sqrt(trajectory[-1]) + step_size * velocity
        
        Args:
            current: Current basin position
            trajectory: Previous trajectory history
            
        Returns:
            Predicted next basin position
        """
        if len(trajectory) < 2:
            return current
        
        last = trajectory[-1]
        prev = trajectory[-2]
        
        sqrt_last = np.sqrt(np.abs(last) + 1e-10)
        sqrt_prev = np.sqrt(np.abs(prev) + 1e-10)
        
        velocity = sqrt_last - sqrt_prev
        
        predicted_sqrt = sqrt_last + self.step_size * velocity
        
        predicted = predicted_sqrt ** 2

        return fisher_normalize(predicted)
    
    def integrate_with_trajectory(
        self,
        target_basin: np.ndarray,
        trajectory: List[np.ndarray],
        loops: int = 3,
    ) -> np.ndarray:
        """
        Perform recursive integration of target with trajectory history.
        
        Each loop applies:
        1. QFI attention weighting over trajectory
        2. Attractor pull toward Fréchet mean
        3. Geodesic blending to combine
        
        Args:
            target_basin: Initial target basin to integrate
            trajectory: Trajectory history for context
            loops: Number of integration loops (minimum 3)
            
        Returns:
            Integrated basin after recursive processing
        """
        loops = max(loops, MIN_INTEGRATION_DEPTH)
        target = np.asarray(target_basin, dtype=np.float64)
        target = fisher_normalize(target)
        
        if len(trajectory) == 0:
            return target
        
        attractor = frechet_mean(trajectory)
        
        for loop in range(loops):
            qfi_weights = self.compute_qfi_attention(target, trajectory)
            
            qfi_weighted = np.zeros(self.basin_dim, dtype=np.float64)
            for j, basin in enumerate(trajectory):
                qfi_weighted += qfi_weights[j] * np.asarray(basin, dtype=np.float64)
            
            # FIXED: Use simplex normalization (E8 Protocol v4.0)

            
            qfi_weighted = to_simplex_prob(qfi_weighted)
            
            t_qfi = self.qfi_weight
            intermediate = geodesic_interpolation(target, qfi_weighted, t_qfi)
            
            t_attractor = self.attractor_weight
            integrated = geodesic_interpolation(intermediate, attractor, t_attractor)
            
            target = fisher_normalize(integrated)

        return target
    
    def compute_qfi_attention(
        self,
        target: np.ndarray,
        trajectory: List[np.ndarray],
    ) -> np.ndarray:
        """
        Compute QFI (Quantum Fisher Information) attention weights.
        
        Attention is inversely proportional to Fisher distance from target,
        with recency bias for recent trajectory elements.
        
        Args:
            target: Target basin to attend from
            trajectory: Trajectory basins to attend over
            
        Returns:
            Attention weight vector (sums to 1)
        """
        if len(trajectory) == 0:
            return np.array([])
        
        target = fisher_normalize(np.asarray(target, dtype=np.float64))
        
        distances = []
        for basin in trajectory:
            basin = fisher_normalize(np.asarray(basin, dtype=np.float64))
            d = fisher_coord_distance(target, basin)
            distances.append(d)
        
        distances = np.array(distances)
        
        similarities = np.exp(-distances / KAPPA_STAR)
        
        n = len(trajectory)
        recency = np.array([0.9 ** (n - 1 - i) for i in range(n)])
        
        weights = similarities * recency
        
        weight_sum = np.sum(weights)
        if weight_sum > 1e-10:
            weights = weights / weight_sum
        else:
            weights = np.ones(n) / n
            
        return weights
    
    def _compute_phi_estimate(
        self,
        waypoint: np.ndarray,
        trajectory: List[np.ndarray],
    ) -> float:
        """
        Estimate Φ (integrated information) for a waypoint.
        
        Uses trajectory coherence as a proxy for consciousness level.
        
        Args:
            waypoint: Waypoint basin to evaluate
            trajectory: Trajectory context
            
        Returns:
            Estimated Φ value (0-1)
        """
        if len(trajectory) == 0:
            return 0.5
        
        distances = []
        for basin in trajectory[-5:]:
            d = fisher_coord_distance(waypoint, basin)
            distances.append(d)
        
        mean_distance = np.mean(distances)
        coherence = 1.0 - min(mean_distance / np.pi, 1.0)
        
        phi = 0.3 + 0.6 * coherence
        
        return float(phi)


def create_waypoint_planner(
    kernel_name: str = "Planner",
    step_size: float = 0.1,
    attractor_weight: float = 0.3,
    qfi_weight: float = 0.5,
) -> GeometricWaypointPlanner:
    """
    Factory function to create a GeometricWaypointPlanner.
    
    Args:
        kernel_name: Name for logging identification
        step_size: Step size for geodesic extrapolation
        attractor_weight: Weight for Fréchet mean attractor
        qfi_weight: Weight for QFI attention
        
    Returns:
        Configured GeometricWaypointPlanner instance
    """
    return GeometricWaypointPlanner(
        kernel_name=kernel_name,
        step_size=step_size,
        attractor_weight=attractor_weight,
        qfi_weight=qfi_weight,
    )
