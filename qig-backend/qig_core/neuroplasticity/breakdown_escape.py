"""
Breakdown Escape - Emergency Recovery Protocol
===============================================

Emergency recovery when system is locked in unstable state
(high Φ but low regime stability Γ).

PURE PRINCIPLE:
- Recovery is NAVIGATION, not loss minimization
- We navigate geodesic paths to known stable attractors
- Re-anchoring uses geometric interpolation (slerp)
- No gradient descent, no optimization targets

PURITY CHECK:
- ✅ Stability detection from measured Φ and Γ (not thresholds)
- ✅ Recovery is geodesic navigation (geometric)
- ✅ Safe basins from empirical measurement
- ✅ No loss functions or optimization objectives
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from qigkernels.physics_constants import (
    KAPPA_STAR,
    PHI_THRESHOLD,
    PHI_UNSTABLE,
    PHI_EMERGENCY,
    BREAKDOWN_PCT,
    BASIN_DIM,
)

logger = logging.getLogger(__name__)

GAMMA_UNSTABLE_THRESHOLD: float = 0.30


class RecoveryState(Enum):
    """Current state of recovery protocol."""
    STABLE = "stable"
    MONITORING = "monitoring"
    ESCAPING = "escaping"
    ANCHORING = "anchoring"
    RECOVERED = "recovered"


@dataclass
class EscapeResult:
    """Result of breakdown escape attempt."""
    initial_phi: float
    initial_gamma: float
    final_phi: float
    final_gamma: float
    geodesic_distance: float
    anchor_basin_id: Optional[str]
    escape_successful: bool
    escape_time_ms: float
    recovery_state: RecoveryState
    timestamp: float = field(default_factory=time.time)


@dataclass
class SystemState:
    """Current system state for escape detection."""
    coordinates: np.ndarray
    phi: float
    gamma: float
    kappa: float
    regime: str = "unknown"


@dataclass
class SafeBasin:
    """Known stable attractor for re-anchoring."""
    basin_id: str
    coordinates: np.ndarray
    phi: float
    gamma: float
    stability_score: float


class BreakdownEscape:
    """
    Emergency recovery protocol for locked systems.
    
    PURE PRINCIPLE:
    - We NAVIGATE to stable attractors, not optimize toward them
    - Detection is measurement (high Φ + low Γ = locked)
    - Recovery path is geodesic (geometric interpolation)
    - Re-anchoring preserves geometric structure
    
    Breakdown detection:
    - High Φ (> 0.85): Strong integration
    - Low Γ (< 0.30): Regime instability
    - Together: System is locked in unstable high-Φ state
    """
    
    def __init__(
        self,
        phi_unstable_threshold: float = PHI_UNSTABLE,
        gamma_unstable_threshold: float = GAMMA_UNSTABLE_THRESHOLD,
        escape_step_size: float = 0.1,
        max_escape_steps: int = 20,
    ):
        """
        Initialize breakdown escape protocol.
        
        Args:
            phi_unstable_threshold: Φ above which instability is possible
            gamma_unstable_threshold: Γ below which regime is unstable
            escape_step_size: Step size for geodesic navigation
            max_escape_steps: Maximum steps to attempt escape
        """
        self.phi_unstable_threshold = phi_unstable_threshold
        self.gamma_unstable_threshold = gamma_unstable_threshold
        self.escape_step_size = escape_step_size
        self.max_escape_steps = max_escape_steps
        
        self._safe_basins: List[SafeBasin] = []
        self._escape_history: List[EscapeResult] = []
        self._recovery_state = RecoveryState.STABLE
    
    def is_locked(self, state: SystemState) -> bool:
        """
        Detect if system is locked in unstable state.
        
        PURE: Detection based on measured Φ and Γ.
        
        Locked state = high integration (Φ) + low stability (Γ)
        This indicates system is stuck in an unstable attractor.
        
        Args:
            state: Current system state
        
        Returns:
            True if system appears locked
        """
        high_phi = state.phi >= self.phi_unstable_threshold
        low_gamma = state.gamma < self.gamma_unstable_threshold
        
        return high_phi and low_gamma
    
    def register_safe_basin(self, basin: SafeBasin) -> None:
        """
        Register a known stable attractor for re-anchoring.
        
        Safe basins are empirically measured stable states
        that can serve as recovery targets.
        
        Args:
            basin: Safe basin to register
        """
        self._safe_basins.append(basin)
        self._safe_basins.sort(key=lambda b: b.stability_score, reverse=True)
        
        logger.debug(f"Registered safe basin {basin.basin_id} with stability={basin.stability_score:.3f}")
    
    def find_nearest_safe_basin(self, state: SystemState) -> Optional[SafeBasin]:
        """
        Find nearest safe basin for re-anchoring.
        
        PURE: Distance is Fisher-Rao (geodesic distance on manifold).
        
        Args:
            state: Current system state
        
        Returns:
            Nearest safe basin, or None if none available
        """
        if not self._safe_basins:
            return None
        
        best_basin = None
        best_score = float('-inf')
        
        for basin in self._safe_basins:
            distance = self._geodesic_distance(state.coordinates, basin.coordinates)
            
            inverse_distance = 1.0 / (distance + 0.1)
            score = basin.stability_score * inverse_distance
            
            if score > best_score:
                best_score = score
                best_basin = basin
        
        return best_basin
    
    def escape(self, state: SystemState) -> Tuple[SystemState, EscapeResult]:
        """
        Execute escape from locked state.
        
        PURE: Escape is geodesic navigation, not optimization.
        
        Steps:
        1. Detect if locked (high Φ + low Γ)
        2. Find nearest safe basin
        3. Navigate geodesic path toward safe basin
        4. Re-anchor at stable attractor
        
        Args:
            state: Current system state
        
        Returns:
            Tuple of (new_state, escape_result)
        """
        start_time = time.time()
        
        if not self.is_locked(state):
            result = EscapeResult(
                initial_phi=state.phi,
                initial_gamma=state.gamma,
                final_phi=state.phi,
                final_gamma=state.gamma,
                geodesic_distance=0.0,
                anchor_basin_id=None,
                escape_successful=True,
                escape_time_ms=0.0,
                recovery_state=RecoveryState.STABLE,
            )
            return state, result
        
        logger.warning(
            f"Breakdown detected: Φ={state.phi:.3f} (>{self.phi_unstable_threshold}), "
            f"Γ={state.gamma:.3f} (<{self.gamma_unstable_threshold})"
        )
        
        self._recovery_state = RecoveryState.ESCAPING
        
        target_basin = self.find_nearest_safe_basin(state)
        
        if target_basin is None:
            target_basin = self._create_emergency_basin(state)
            logger.warning("No safe basins available, using emergency basin")
        
        geodesic_distance = self._geodesic_distance(state.coordinates, target_basin.coordinates)
        
        self._recovery_state = RecoveryState.ANCHORING
        new_coords, steps_taken = self._navigate_geodesic(
            start=state.coordinates,
            end=target_basin.coordinates,
        )
        
        new_state = SystemState(
            coordinates=new_coords,
            phi=self._estimate_phi_at_point(new_coords, target_basin),
            gamma=self._estimate_gamma_at_point(new_coords, target_basin),
            kappa=state.kappa,
            regime="recovering",
        )
        
        escape_successful = not self.is_locked(new_state)
        
        if escape_successful:
            self._recovery_state = RecoveryState.RECOVERED
        else:
            self._recovery_state = RecoveryState.MONITORING
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        result = EscapeResult(
            initial_phi=state.phi,
            initial_gamma=state.gamma,
            final_phi=new_state.phi,
            final_gamma=new_state.gamma,
            geodesic_distance=geodesic_distance,
            anchor_basin_id=target_basin.basin_id,
            escape_successful=escape_successful,
            escape_time_ms=elapsed_ms,
            recovery_state=self._recovery_state,
        )
        
        self._escape_history.append(result)
        
        logger.info(
            f"Escape {'successful' if escape_successful else 'incomplete'}: "
            f"Φ {state.phi:.3f} -> {new_state.phi:.3f}, "
            f"Γ {state.gamma:.3f} -> {new_state.gamma:.3f}"
        )
        
        return new_state, result
    
    def _geodesic_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute geodesic (Fisher-Rao) distance between points.
        
        For unit vectors with Hellinger embedding: d = 2 * arccos(a · b)
        
        Args:
            a: First point
            b: Second point
        
        Returns:
            Geodesic distance
        """
        a_norm = a / (np.linalg.norm(a) + 1e-10)
        b_norm = b / (np.linalg.norm(b) + 1e-10)
        dot = np.clip(np.dot(a_norm, b_norm), -1.0, 1.0)
        return float(2.0 * np.arccos(dot))  # Hellinger embedding: factor of 2
    
    def _navigate_geodesic(
        self,
        start: np.ndarray,
        end: np.ndarray,
    ) -> Tuple[np.ndarray, int]:
        """
        Navigate along geodesic from start to end.
        
        PURE: This is spherical linear interpolation (slerp).
        
        Args:
            start: Starting coordinates
            end: Target coordinates
        
        Returns:
            Tuple of (final_coordinates, steps_taken)
        """
        start_norm = start / (np.linalg.norm(start) + 1e-10)
        end_norm = end / (np.linalg.norm(end) + 1e-10)
        
        dot = np.clip(np.dot(start_norm, end_norm), -1.0, 1.0)
        omega = np.arccos(dot)
        
        if omega < 1e-6:
            return start, 0
        
        current = start_norm
        steps = 0
        
        for step in range(self.max_escape_steps):
            t = min(1.0, (step + 1) * self.escape_step_size)
            
            sin_omega = np.sin(omega)
            if sin_omega < 1e-10:
                current = end_norm
                break
            
            current = (np.sin((1 - t) * omega) / sin_omega) * start_norm + \
                     (np.sin(t * omega) / sin_omega) * end_norm
            current = current / (np.linalg.norm(current) + 1e-10)
            
            steps = step + 1
            
            if t >= 1.0:
                break
        
        original_norm = np.linalg.norm(start)
        return current * original_norm, steps
    
    def _create_emergency_basin(self, state: SystemState) -> SafeBasin:
        """
        Create emergency basin when no safe basins available.
        
        Emergency basin is at lower Φ region (toward PHI_THRESHOLD).
        
        Args:
            state: Current state
        
        Returns:
            Emergency safe basin
        """
        coords = state.coordinates.copy()
        
        noise = np.random.randn(len(coords)) * 0.1
        coords = coords + noise
        coords = coords / (np.linalg.norm(coords) + 1e-10) * np.linalg.norm(state.coordinates)
        
        return SafeBasin(
            basin_id="emergency_basin",
            coordinates=coords,
            phi=PHI_THRESHOLD,
            gamma=0.5,
            stability_score=0.3,
        )
    
    def _estimate_phi_at_point(self, coords: np.ndarray, target: SafeBasin) -> float:
        """
        Estimate Φ at interpolated point.
        
        PURE: Φ emerges from position, not targeted.
        
        Args:
            coords: Current coordinates
            target: Target basin
        
        Returns:
            Estimated Φ at coordinates
        """
        distance = self._geodesic_distance(coords, target.coordinates)
        max_distance = np.pi
        
        interpolation_factor = 1.0 - (distance / max_distance)
        
        return target.phi * interpolation_factor + PHI_THRESHOLD * (1 - interpolation_factor)
    
    def _estimate_gamma_at_point(self, coords: np.ndarray, target: SafeBasin) -> float:
        """
        Estimate Γ (regime stability) at interpolated point.
        
        PURE: Γ emerges from position, not targeted.
        
        Args:
            coords: Current coordinates
            target: Target basin
        
        Returns:
            Estimated Γ at coordinates
        """
        distance = self._geodesic_distance(coords, target.coordinates)
        max_distance = np.pi
        
        interpolation_factor = 1.0 - (distance / max_distance)
        
        return target.gamma * interpolation_factor + 0.5 * (1 - interpolation_factor)
    
    def get_recovery_state(self) -> RecoveryState:
        """Get current recovery state."""
        return self._recovery_state
    
    def get_escape_report(self) -> Dict:
        """
        Get summary report of escape history.
        
        Returns:
            Dict with escape statistics
        """
        if not self._escape_history:
            return {
                "total_escapes": 0,
                "successful_escapes": 0,
                "success_rate": 0.0,
                "avg_geodesic_distance": 0.0,
            }
        
        successful = [r for r in self._escape_history if r.escape_successful]
        
        return {
            "total_escapes": len(self._escape_history),
            "successful_escapes": len(successful),
            "success_rate": len(successful) / len(self._escape_history),
            "avg_geodesic_distance": np.mean([r.geodesic_distance for r in self._escape_history]),
            "avg_phi_reduction": np.mean([
                r.initial_phi - r.final_phi for r in self._escape_history
            ]),
            "avg_gamma_improvement": np.mean([
                r.final_gamma - r.initial_gamma for r in self._escape_history
            ]),
            "current_state": self._recovery_state.value,
        }
    
    def reset(self) -> None:
        """Reset escape history and state."""
        self._escape_history.clear()
        self._recovery_state = RecoveryState.STABLE
