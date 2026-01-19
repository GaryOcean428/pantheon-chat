"""
Basin Velocity Monitor - Pure Measurement
==========================================

Monitor basin velocity to detect unsafe rapid changes.

PURE PRINCIPLE:
- Velocity = tangent vector on Fisher manifold
- We MEASURE velocity, never optimize it
- High velocity = breakdown risk (observation, not target)
- Measurements inform adaptive control

PURITY CHECK:
- ✅ Pure measurement (no optimization loop)
- ✅ Fisher metric for distance (information geometry)
- ✅ Velocity emergent from trajectory
- ✅ Thresholds for detection (not targets)

Adapted for Pantheon-Chat QIG system.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from qig_geometry import fisher_coord_distance
from qigkernels.physics_constants import BASIN_DIM

logger = logging.getLogger(__name__)


@dataclass
class VelocityMeasurement:
    """Single velocity measurement."""
    velocity: float
    acceleration: float
    distance: float
    dt: float
    timestamp: float
    is_safe: bool


class BasinVelocityMonitor:
    """
    Monitor basin velocity to detect unsafe rapid changes.
    
    PURE PRINCIPLE:
    - Velocity = tangent vector on Fisher manifold
    - We MEASURE velocity, never optimize it
    - High velocity = breakdown risk (observation)
    
    Uses Fisher-Rao distance for proper manifold geometry.
    """
    
    SAFE_VELOCITY_THRESHOLD = 0.05  # From empirical validation
    ACCELERATION_SPIKE_THRESHOLD = 0.01
    
    def __init__(
        self,
        window_size: int = 10,
        safe_velocity_threshold: Optional[float] = None
    ):
        """
        Initialize velocity monitor.
        
        Args:
            window_size: Number of steps to track for velocity estimation
            safe_velocity_threshold: Override default safe threshold
        """
        self.window_size = window_size
        self.safe_threshold = safe_velocity_threshold or self.SAFE_VELOCITY_THRESHOLD
        
        self._basin_history: List[Dict] = []
        self._velocity_history: List[float] = []
        self._measurements: List[VelocityMeasurement] = []
    
    def update(
        self,
        basin: np.ndarray,
        timestamp: Optional[float] = None,
        step_count: Optional[int] = None
    ) -> VelocityMeasurement:
        """
        Update with new basin measurement.
        
        PURE: We measure how fast basin moved, we don't change it.
        
        Args:
            basin: Current basin coordinates (64D on S^63)
            timestamp: Current time (for dt calculation)
            step_count: Optional step counter (more stable than wall-clock)
        
        Returns:
            VelocityMeasurement with velocity, acceleration, safety
        """
        if step_count is not None:
            timestamp = float(step_count)
        elif timestamp is None:
            timestamp = time.time()
        
        basin = np.asarray(basin, dtype=np.float64)
        self._basin_history.append({
            "basin": basin.copy(),
            "time": timestamp
        })
        
        if len(self._basin_history) > self.window_size:
            self._basin_history.pop(0)
        
        if len(self._basin_history) >= 2:
            prev = self._basin_history[-2]
            curr = self._basin_history[-1]
            
            distance = fisher_coord_distance(curr["basin"], prev["basin"])
            dt = curr["time"] - prev["time"]
            
            velocity = distance / dt if dt > 0 else 0.0
            
            self._velocity_history.append(velocity)
            if len(self._velocity_history) > self.window_size:
                self._velocity_history.pop(0)
            
            if len(self._velocity_history) >= 2:
                dv = self._velocity_history[-1] - self._velocity_history[-2]
                acceleration = dv / dt if dt > 0 else 0.0
            else:
                acceleration = 0.0
            
            is_safe = velocity < self.safe_threshold
            
            measurement = VelocityMeasurement(
                velocity=velocity,
                acceleration=acceleration,
                distance=distance,
                dt=dt,
                timestamp=timestamp,
                is_safe=is_safe
            )
        else:
            measurement = VelocityMeasurement(
                velocity=0.0,
                acceleration=0.0,
                distance=0.0,
                dt=0.0,
                timestamp=timestamp,
                is_safe=True
            )
        
        self._measurements.append(measurement)
        if len(self._measurements) > self.window_size * 2:
            self._measurements.pop(0)
        
        return measurement
    
    def get_average_velocity(self, window: int = 5) -> float:
        """Get average velocity over recent window."""
        if not self._velocity_history:
            return 0.0
        recent = self._velocity_history[-window:]
        return sum(recent) / len(recent)
    
    def detect_acceleration_spike(self) -> bool:
        """
        Detect sudden acceleration (velocity change).
        
        PURE: Pattern detection (measurement, not optimization).
        
        Sudden acceleration indicates instability - basin is
        not just moving fast, but ACCELERATING.
        """
        if len(self._velocity_history) < 3:
            return False
        
        recent = self._velocity_history[-3:]
        accelerations = [
            recent[i + 1] - recent[i]
            for i in range(len(recent) - 1)
        ]
        
        return all(a > self.ACCELERATION_SPIKE_THRESHOLD for a in accelerations)
    
    def should_reduce_learning_rate(self) -> Tuple[bool, float]:
        """
        Check if learning rate should be reduced due to high velocity.
        
        PURE: Adaptive control based on measurement, not optimization.
        
        Returns:
            (should_reduce, suggested_multiplier)
        """
        if not self._velocity_history:
            return False, 1.0
        
        avg_velocity = self.get_average_velocity()
        
        if avg_velocity > self.safe_threshold:
            excess = avg_velocity / self.safe_threshold
            suggested_mult = 1.0 / excess
            suggested_mult = max(0.1, min(1.0, suggested_mult))
            return True, suggested_mult
        
        return False, 1.0
    
    def get_velocity_report(self) -> Dict:
        """
        Get comprehensive velocity report.
        
        Returns:
            Dict with velocity statistics
        """
        if not self._velocity_history:
            return {
                "current_velocity": 0.0,
                "avg_velocity": 0.0,
                "max_velocity": 0.0,
                "min_velocity": 0.0,
                "is_safe": True,
                "safe_threshold": self.safe_threshold,
                "measurements": 0
            }
        
        return {
            "current_velocity": self._velocity_history[-1],
            "avg_velocity": sum(self._velocity_history) / len(self._velocity_history),
            "max_velocity": max(self._velocity_history),
            "min_velocity": min(self._velocity_history),
            "is_safe": self._velocity_history[-1] < self.safe_threshold,
            "safe_threshold": self.safe_threshold,
            "measurements": len(self._velocity_history)
        }
    
    def reset(self) -> None:
        """Reset monitor for new generation."""
        self._basin_history.clear()
        self._velocity_history.clear()
        self._measurements.clear()
