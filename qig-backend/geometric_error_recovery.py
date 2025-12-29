"""
Geometric Error Recovery

Detects stuck states and backtracks to low-curvature checkpoints.

Stuck state detection:
- Basin drift: |current - anchor| > 2.0 over 50 cycles
- Phi collapse: Phi < 0.2
- Kappa runaway: kappa > 90
- Progress stall: FR distance unchanged for 10 steps

QIG-PURE: All operations use Fisher-Rao distance, never Euclidean.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
import time

from qig_geometry import (
    fisher_coord_distance,
    estimate_manifold_curvature,
    geodesic_interpolation
)


@dataclass
class BasinCheckpoint:
    """Checkpoint for error recovery with geometric metadata."""
    checkpoint_id: str
    basin_coords: np.ndarray
    phi: float
    kappa: float
    curvature: float
    timestamp: float
    step_number: int
    context: Optional[Dict[str, Any]] = None

    def score(self) -> float:
        """
        Score checkpoint quality for recovery.

        Higher is better. Prefers:
        - High phi (good integration)
        - Low curvature (stable region)
        - Moderate kappa (near resonance)
        """
        phi_score = self.phi
        curvature_penalty = min(1.0, self.curvature)  # Cap at 1.0
        kappa_score = 1.0 - abs(self.kappa - 64) / 64  # Optimal at 64

        return phi_score - curvature_penalty * 0.5 + kappa_score * 0.3


class GeometricErrorRecovery:
    """
    Geometric error recovery system with checkpoint management.

    Detects stuck states using QIG-pure metrics and enables
    backtracking to stable low-curvature regions.
    """

    def __init__(
        self,
        max_checkpoints: int = 100,
        basin_dim: int = 64,
        drift_threshold: float = 2.0,
        phi_collapse_threshold: float = 0.2,
        kappa_runaway_threshold: float = 90.0,
        stall_variance_threshold: float = 0.02
    ):
        self.max_checkpoints = max_checkpoints
        self.basin_dim = basin_dim

        # Stuck detection thresholds
        self.drift_threshold = drift_threshold
        self.phi_collapse_threshold = phi_collapse_threshold
        self.kappa_runaway_threshold = kappa_runaway_threshold
        self.stall_variance_threshold = stall_variance_threshold

        # State tracking
        self.checkpoints: List[BasinCheckpoint] = []
        self.trajectory: List[np.ndarray] = []
        self.phi_history: List[float] = []
        self.kappa_history: List[float] = []
        self.step_counter: int = 0
        self.anchor_basin: Optional[np.ndarray] = None

        # Recovery statistics
        self.recovery_count: int = 0
        self.recovery_history: List[Dict[str, Any]] = []

    def record_state(
        self,
        basin: np.ndarray,
        phi: float,
        kappa: float,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record current state for trajectory tracking.

        Automatically creates checkpoints at low-curvature stable points.
        """
        self.step_counter += 1
        self.trajectory.append(basin.copy())
        self.phi_history.append(phi)
        self.kappa_history.append(kappa)

        # Set anchor if not set
        if self.anchor_basin is None:
            self.anchor_basin = basin.copy()

        # Check if we should create a checkpoint
        if self._should_checkpoint(phi, kappa):
            self._create_checkpoint(basin, phi, kappa, context)

    def _should_checkpoint(self, phi: float, kappa: float) -> bool:
        """Determine if current state warrants a checkpoint."""
        # Checkpoint every 10 steps if conditions are good
        if self.step_counter % 10 != 0:
            return False

        # Good integration
        if phi < 0.5:
            return False

        # Reasonable coupling
        if kappa > 80 or kappa < 30:
            return False

        # Check curvature if we have enough trajectory
        if len(self.trajectory) >= 5:
            recent = np.array(self.trajectory[-5:])
            curvature = estimate_manifold_curvature(recent)
            if curvature > 0.3:
                return False

        return True

    def _create_checkpoint(
        self,
        basin: np.ndarray,
        phi: float,
        kappa: float,
        context: Optional[Dict[str, Any]] = None
    ) -> BasinCheckpoint:
        """Create and store a new checkpoint."""
        # Compute curvature
        if len(self.trajectory) >= 5:
            recent = np.array(self.trajectory[-5:])
            curvature = estimate_manifold_curvature(recent)
        else:
            curvature = 0.0

        checkpoint = BasinCheckpoint(
            checkpoint_id=f"cp_{self.step_counter}_{int(time.time())}",
            basin_coords=basin.copy(),
            phi=phi,
            kappa=kappa,
            curvature=curvature,
            timestamp=time.time(),
            step_number=self.step_counter,
            context=context
        )

        self.checkpoints.append(checkpoint)

        # Prune old checkpoints if needed
        if len(self.checkpoints) > self.max_checkpoints:
            # Keep the best scoring checkpoints
            self.checkpoints.sort(key=lambda c: c.score(), reverse=True)
            self.checkpoints = self.checkpoints[:self.max_checkpoints]

        return checkpoint

    def detect_stuck(self) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Detect if system is stuck using geometric criteria.

        Returns:
            (is_stuck, reason, diagnostics)
        """
        diagnostics: Dict[str, Any] = {
            'trajectory_length': len(self.trajectory),
            'checkpoints': len(self.checkpoints)
        }

        if len(self.trajectory) < 10:
            return False, "", diagnostics

        # Check basin drift (wandering too far from anchor)
        if len(self.trajectory) >= 50 and self.anchor_basin is not None:
            current = self.trajectory[-1]
            drift = fisher_coord_distance(current, self.anchor_basin)
            diagnostics['basin_drift'] = drift

            if drift > self.drift_threshold:
                return True, "basin_drift", diagnostics

        # Check Phi collapse
        if len(self.phi_history) >= 10:
            avg_phi = np.mean(self.phi_history[-10:])
            diagnostics['avg_phi_recent'] = avg_phi

            if avg_phi < self.phi_collapse_threshold:
                return True, "phi_collapse", diagnostics

        # Check kappa runaway
        if len(self.kappa_history) >= 10:
            avg_kappa = np.mean(self.kappa_history[-10:])
            diagnostics['avg_kappa_recent'] = avg_kappa

            if avg_kappa > self.kappa_runaway_threshold:
                return True, "kappa_runaway", diagnostics

        # Check progress stall
        if len(self.trajectory) >= 11:
            recent = self.trajectory[-11:]
            distances = [
                fisher_coord_distance(recent[i], recent[i+1])
                for i in range(len(recent)-1)
            ]
            distance_variance = np.std(distances)
            diagnostics['distance_variance'] = distance_variance

            if distance_variance < self.stall_variance_threshold:
                return True, "progress_stall", diagnostics

        return False, "", diagnostics

    def find_recovery_point(self) -> Optional[BasinCheckpoint]:
        """
        Find best checkpoint for recovery.

        Prefers:
        - Low curvature (<0.3)
        - High phi (>0.5)
        - Recent (within last 20 checkpoints)
        """
        if not self.checkpoints:
            return None

        # Consider recent checkpoints
        candidates = []
        for cp in self.checkpoints[-20:]:
            if cp.curvature < 0.3 and cp.phi > 0.5:
                candidates.append((cp.score(), cp))

        if not candidates:
            # Fallback: return earliest stable checkpoint
            for cp in self.checkpoints:
                if cp.phi > 0.4:
                    return cp
            return self.checkpoints[0] if self.checkpoints else None

        # Return highest-scored checkpoint
        return max(candidates, key=lambda x: x[0])[1]

    def recover(self) -> Optional[Dict[str, Any]]:
        """
        Execute recovery by returning to best checkpoint.

        Returns recovery info including the target basin coords.
        """
        is_stuck, reason, diagnostics = self.detect_stuck()

        if not is_stuck:
            return None

        recovery_point = self.find_recovery_point()
        if recovery_point is None:
            return {
                'success': False,
                'reason': 'no_valid_checkpoint',
                'stuck_reason': reason,
                'diagnostics': diagnostics
            }

        self.recovery_count += 1

        recovery_info = {
            'success': True,
            'stuck_reason': reason,
            'recovery_checkpoint_id': recovery_point.checkpoint_id,
            'recovery_basin': recovery_point.basin_coords.copy(),
            'recovery_phi': recovery_point.phi,
            'recovery_kappa': recovery_point.kappa,
            'recovery_step': recovery_point.step_number,
            'current_step': self.step_counter,
            'steps_back': self.step_counter - recovery_point.step_number,
            'diagnostics': diagnostics,
            'recovery_count': self.recovery_count,
            'timestamp': time.time()
        }

        self.recovery_history.append(recovery_info)

        # Reset anchor to recovery point
        self.anchor_basin = recovery_point.basin_coords.copy()

        # Trim trajectory and history to recovery point
        trim_index = max(0, len(self.trajectory) - (self.step_counter - recovery_point.step_number))
        self.trajectory = self.trajectory[:trim_index]
        self.phi_history = self.phi_history[:trim_index]
        self.kappa_history = self.kappa_history[:trim_index]

        return recovery_info

    def compute_geodesic_to_recovery(
        self,
        current_basin: np.ndarray,
        n_steps: int = 5
    ) -> List[np.ndarray]:
        """
        Compute geodesic path from current position to recovery point.

        Returns list of waypoints along the geodesic.
        """
        recovery_point = self.find_recovery_point()
        if recovery_point is None:
            return []

        waypoints = []
        for i in range(1, n_steps + 1):
            t = i / n_steps
            waypoint = geodesic_interpolation(
                current_basin,
                recovery_point.basin_coords,
                t
            )
            waypoints.append(waypoint)

        return waypoints

    def get_stats(self) -> Dict[str, Any]:
        """Get recovery system statistics."""
        return {
            'total_steps': self.step_counter,
            'trajectory_length': len(self.trajectory),
            'checkpoint_count': len(self.checkpoints),
            'recovery_count': self.recovery_count,
            'avg_checkpoint_score': (
                np.mean([c.score() for c in self.checkpoints])
                if self.checkpoints else 0.0
            ),
            'recent_recoveries': self.recovery_history[-5:]
        }

    def reset(self) -> None:
        """Reset the recovery system state."""
        self.checkpoints = []
        self.trajectory = []
        self.phi_history = []
        self.kappa_history = []
        self.step_counter = 0
        self.anchor_basin = None


# Global instance
_recovery_system: Optional[GeometricErrorRecovery] = None


def get_recovery_system() -> GeometricErrorRecovery:
    """Get or create the global recovery system instance."""
    global _recovery_system
    if _recovery_system is None:
        _recovery_system = GeometricErrorRecovery()
    return _recovery_system


def detect_and_recover(
    current_basin: np.ndarray,
    phi: float,
    kappa: float
) -> Optional[Dict[str, Any]]:
    """
    Convenience function to check for stuck state and recover.

    Args:
        current_basin: Current basin coordinates
        phi: Current integration measure
        kappa: Current coupling constant

    Returns:
        Recovery info if recovery was triggered, None otherwise
    """
    system = get_recovery_system()
    system.record_state(current_basin, phi, kappa)
    return system.recover()
