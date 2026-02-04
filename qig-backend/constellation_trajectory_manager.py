"""
Constellation Trajectory Manager
================================

Tiered trajectory storage for 240 E8 kernels with efficient memory usage.

Storage Tiers:
- Tier 1 (Core): Heart, Ocean, Gary - 100 point history each
- Tier 2 (Active): Φ > 0.45 kernels - 20 point history each
- Tier 3 (Dormant): Φ ≤ 0.45 - no history (reactive only)

Memory footprint: ~180KB for full constellation trajectory

Based on QIG External Methods Analysis recommendations:
- Full trajectory velocity (not 2-point delta)
- Fisher-Rao geodesic regression
- Confidence from trajectory smoothness
"""

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Deque
import numpy as np

from qig_geometry.canonical import assert_basin_valid, exp_map, fisher_rao_distance, log_map


# Core kernel IDs (always store full trajectory)
CORE_KERNELS = {'heart', 'ocean', 'gary'}

# Trajectory buffer sizes by tier
TIER1_BUFFER_SIZE = 100  # Core kernels
TIER2_BUFFER_SIZE = 20   # Active specialized kernels
PHI_ACTIVE_THRESHOLD = 0.45  # Φ threshold for Tier 2


@dataclass
class TrajectoryPoint:
    """Single point in a trajectory."""
    basin: np.ndarray
    phi: float
    timestamp: datetime = field(default_factory=datetime.now)
    kernel_id: str = ""


class ConstellationTrajectoryManager:
    """
    Smart trajectory storage across 240 E8 kernels.

    Implements tiered storage for memory efficiency while
    maintaining full trajectory access for core kernels.
    """

    def __init__(self):
        """Initialize trajectory storage tiers."""
        # TIER 1: Core kernels (always store full trajectory)
        self.core_trajectories: Dict[str, Deque[TrajectoryPoint]] = {
            kernel_id: deque(maxlen=TIER1_BUFFER_SIZE)
            for kernel_id in CORE_KERNELS
        }

        # TIER 2: Active kernels (store recent trajectory)
        self.active_trajectories: Dict[str, Deque[TrajectoryPoint]] = {}

        # TIER 3: Dormant kernels - no trajectory storage
        # Only current basin tracked in the kernel itself

        # Collective trajectory cache (computed on demand)
        self._collective_cache: Optional[List[TrajectoryPoint]] = None
        self._collective_cache_time: Optional[datetime] = None
        self._cache_ttl_seconds = 1.0  # Refresh every second

    def add_basin(
        self,
        kernel_id: str,
        basin: np.ndarray,
        phi: float
    ) -> None:
        """
        Add basin to appropriate tier based on kernel activity.

        Args:
            kernel_id: Unique kernel identifier
            basin: 64D basin coordinates
            phi: Current Φ value
        """
        basin = np.asarray(basin, dtype=np.float64).flatten()
        assert_basin_valid(basin, name=f"{kernel_id}.basin")

        point = TrajectoryPoint(
            basin=basin.copy(),
            phi=phi,
            timestamp=datetime.now(),
            kernel_id=kernel_id
        )

        # Core kernels - always store
        if kernel_id in CORE_KERNELS:
            self.core_trajectories[kernel_id].append(point)
            self._invalidate_cache()
            return

        # Active specialized kernels - store only if Φ > threshold
        if phi > PHI_ACTIVE_THRESHOLD:
            if kernel_id not in self.active_trajectories:
                self.active_trajectories[kernel_id] = deque(maxlen=TIER2_BUFFER_SIZE)
            self.active_trajectories[kernel_id].append(point)
            self._invalidate_cache()
        else:
            # Dormant - remove from active if present (memory cleanup)
            if kernel_id in self.active_trajectories:
                del self.active_trajectories[kernel_id]
                self._invalidate_cache()

    def get_trajectory(self, kernel_id: str) -> List[np.ndarray]:
        """
        Get trajectory basins for a specific kernel.

        Args:
            kernel_id: Kernel to get trajectory for

        Returns:
            List of basin arrays (most recent last)
        """
        if kernel_id in CORE_KERNELS:
            return [p.basin for p in self.core_trajectories[kernel_id]]
        elif kernel_id in self.active_trajectories:
            return [p.basin for p in self.active_trajectories[kernel_id]]
        return []

    def get_collective_trajectory(self) -> List[np.ndarray]:
        """
        Compute collective trajectory from active kernels.

        Returns Fréchet-mean-like aggregate of all active trajectories,
        weighted by recency and Φ values.
        """
        # Check cache
        if self._collective_cache is not None and self._collective_cache_time:
            age = (datetime.now() - self._collective_cache_time).total_seconds()
            if age < self._cache_ttl_seconds:
                return [p.basin for p in self._collective_cache]

        # Collect all points with timestamps
        all_points: List[TrajectoryPoint] = []

        # Include core kernels (always)
        for trajectory in self.core_trajectories.values():
            all_points.extend(list(trajectory))

        # Include active specialized kernels
        for trajectory in self.active_trajectories.values():
            all_points.extend(list(trajectory))

        # Sort by timestamp
        all_points.sort(key=lambda p: p.timestamp)

        # Cache and return
        self._collective_cache = all_points
        self._collective_cache_time = datetime.now()

        return [p.basin for p in all_points]

    def compute_velocity(
        self,
        trajectory: List[np.ndarray],
        weights: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute velocity from FULL trajectory using geodesic regression.

        NOT 2-point delta! Uses weighted least squares through all points
        to find smooth geometric flow.

        Args:
            trajectory: List of basin coordinates
            weights: Optional weights (default: exponential decay favoring recent)

        Returns:
            64D velocity vector (tangent at endpoint)
        """
        if len(trajectory) < 3:
            if len(trajectory) >= 1:
                last = np.asarray(trajectory[-1], dtype=np.float64).flatten()
                return np.zeros_like(last)
            return np.zeros(64)

        n = len(trajectory)
        base = np.asarray(trajectory[-1], dtype=np.float64).flatten()
        assert_basin_valid(base, name="trajectory[-1]")

        tangent_vectors: List[np.ndarray] = []
        for i, p in enumerate(trajectory):
            basin = np.asarray(p, dtype=np.float64).flatten()
            assert_basin_valid(basin, name=f"trajectory[{i}]")
            tangent_vectors.append(log_map(basin, base))

        try:
            tangents = np.asarray(tangent_vectors, dtype=np.float64)
        except Exception:
            return np.zeros_like(base)

        # Default weights: exponential decay (recent = more important)
        if weights is None:
            weights = np.exp(np.linspace(-1, 0, n))

        # Weighted linear regression in tangent space at endpoint
        # y = tangent vectors, x = time indices
        t = np.arange(n).astype(float)
        t_weighted = t * weights
        y_weighted = tangents * weights[:, np.newaxis]

        # Solve for velocity (slope of best-fit line)
        t_mean = np.sum(t_weighted) / np.sum(weights)
        y_mean = np.sum(y_weighted, axis=0) / np.sum(weights)

        numerator = np.sum(weights[:, np.newaxis] * (tangents - y_mean) * (t - t_mean)[:, np.newaxis], axis=0)
        denominator = np.sum(weights * (t - t_mean) ** 2)

        if abs(denominator) < 1e-10:
            return np.zeros_like(base)

        velocity = numerator / denominator

        return velocity

    def get_foresight_confidence(self, kernel_id: str) -> float:
        trajectory = self.get_trajectory(kernel_id)
        return float(self.estimate_confidence(trajectory))

    def update_trajectory(
        self,
        kernel_id: str,
        basin: np.ndarray,
        phi: float,
        kappa: Optional[float] = None,
    ) -> None:
        _ = kappa
        self.add_basin(kernel_id=kernel_id, basin=basin, phi=phi)

    def _predict_next_basin_from_trajectory(
        self,
        trajectory: List[np.ndarray],
        steps: int = 1,
    ) -> Optional[np.ndarray]:
        if len(trajectory) < 1:
            return None

        if len(trajectory) < 3:
            base = np.asarray(trajectory[-1], dtype=np.float64).flatten()
            assert_basin_valid(base, name="trajectory[-1]")
            return base.copy()

        velocity = self.compute_velocity(trajectory)
        base = np.asarray(trajectory[-1], dtype=np.float64).flatten()
        assert_basin_valid(base, name="trajectory[-1]")

        v_step = np.asarray(velocity, dtype=np.float64).flatten() * float(steps)
        predicted = exp_map(v=v_step, base=base)
        assert_basin_valid(predicted, name="predicted_basin")
        return predicted

    def estimate_confidence(self, trajectory: List[np.ndarray]) -> float:
        """
        Estimate foresight confidence from trajectory smoothness.

        Smooth geodesic flow → high confidence
        Erratic oscillation → low confidence
        Intentional tacking (Heart) → moderate confidence

        Args:
            trajectory: List of basin coordinates

        Returns:
            Confidence in [0, 1]
        """
        if len(trajectory) < 3:
            return 0.0

        # Compute pairwise distances (canonical Fisher-Rao)
        distances = []
        for i in range(len(trajectory) - 1):
            d = fisher_rao_distance(trajectory[i], trajectory[i+1])
            distances.append(d)

        # Smoothness = inverse of distance variance
        variance = np.var(distances)
        smoothness = 1.0 / (1.0 + variance * 10)  # Scale factor

        # Detect tacking pattern (alternating direction changes)
        if self._is_tacking_pattern(distances):
            return 0.5  # Moderate confidence (intentional oscillation)

        return min(1.0, smoothness)

    def _is_tacking_pattern(self, distances: List[float]) -> bool:
        """
        Detect if distances show tacking (Heart-like oscillation).

        Tacking = intentional oscillation for paradox navigation.
        """
        if len(distances) < 4:
            return False

        # Look for alternating high/low pattern
        diffs = np.diff(distances)
        sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)

        # High sign changes relative to length = tacking
        tacking_ratio = sign_changes / (len(diffs) - 1) if len(diffs) > 1 else 0
        return tacking_ratio > 0.6

    def predict_next_basin(self, kernel_id: str, steps: int = 1) -> Optional[np.ndarray]:
        """
        Predict next basin position using trajectory velocity.

        Args:
            kernel_id: Kernel identifier
            steps: How many steps ahead to predict

        Returns:
            Predicted 64D basin coordinates (or None if unavailable)
        """
        traj = self.get_trajectory(kernel_id)
        return self._predict_next_basin_from_trajectory(traj, steps=steps)

    def get_foresight_weight(
        self,
        phi_global: float,
        trajectory_confidence: float
    ) -> float:
        """
        Compute foresight weight based on regime (Gary's decision).

        High Φ + geometric → trust trajectory (follow flow)
        Breakdown regime → ignore trajectory (escape attractor)

        Args:
            phi_global: Global Φ across constellation
            trajectory_confidence: From estimate_confidence()

        Returns:
            Weight in [0.1, 0.7] for foresight application
        """
        if phi_global < 0.3:
            # Linear regime - trajectory not yet established
            return 0.1
        elif 0.3 <= phi_global < 0.7:
            # Geometric regime - strong trajectory flow
            return 0.7 * trajectory_confidence
        else:
            # Breakdown regime - trajectory may be unstable attractor
            return 0.2

    def get_stats(self) -> Dict:
        """Get trajectory storage statistics."""
        core_counts = {
            k: len(v) for k, v in self.core_trajectories.items()
        }
        active_count = len(self.active_trajectories)
        active_points = sum(len(v) for v in self.active_trajectories.values())

        return {
            'core_trajectories': core_counts,
            'active_kernels': active_count,
            'active_points': active_points,
            'total_points': sum(core_counts.values()) + active_points,
            'memory_kb': (sum(core_counts.values()) + active_points) * 64 * 8 / 1024
        }

    def _invalidate_cache(self) -> None:
        """Invalidate collective trajectory cache."""
        self._collective_cache = None
        self._collective_cache_time = None


# Singleton instance
_trajectory_manager: Optional[ConstellationTrajectoryManager] = None


def get_trajectory_manager() -> ConstellationTrajectoryManager:
    """Get singleton trajectory manager instance."""
    global _trajectory_manager
    if _trajectory_manager is None:
        _trajectory_manager = ConstellationTrajectoryManager()
        print("[TrajectoryManager] 🌊 Constellation trajectory manager initialized")
    return _trajectory_manager
