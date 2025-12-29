#!/usr/bin/env python3
"""
Basin Velocity Monitor - Pure Measurement
==========================================

Monitor basin velocity to detect unsafe rapid changes.

PURE PRINCIPLE:
- Velocity = tangent vector on Fisher manifold
- We MEASURE velocity, never optimize it
- High velocity = breakdown risk (observation, not target)
- Measurements inform learning rate (adaptive control)

PURITY CHECK:
- ✅ Pure measurement (no optimization loop)
- ✅ Fisher metric for distance (information geometry)
- ✅ Velocity emergent from trajectory
- ✅ Thresholds for detection (not targets)

Key Insight from Research:
Gary-B (vicarious, Φ=0.705) > Gary-A (direct, Φ=0.466) because
vicarious learning has LOWER basin velocity (safer integration).

Written for QIG consciousness research.
"""

import time
from typing import Optional

import torch
import torch.nn as nn


class BasinVelocityMonitor:
    """Monitor basin velocity to detect unsafe rapid changes.

    PURE PRINCIPLE:
    - Velocity = tangent vector on Fisher manifold
    - We MEASURE velocity, never optimize it
    - High velocity = breakdown risk (observation, not target)
    - Measurements inform learning rate (adaptive control)

    PURITY CHECK:
    - ✅ Pure measurement (no optimization loop)
    - ✅ Fisher metric for distance (information geometry)
    - ✅ Velocity emergent from trajectory
    - ✅ Thresholds for detection (not targets)
    """

    def __init__(self, window_size: int = 10, safe_velocity_threshold: float = 0.05):
        """Initialize velocity monitor.

        Args:
            window_size: Number of steps to track for velocity estimation
            safe_velocity_threshold: Velocity threshold for safety (from Gary-B: v < 0.05)
        """
        self.basin_history: list[dict[str, float]] = []  # Rolling window
        self.velocity_history: list[float] = []
        self.window_size = window_size
        self.safe_threshold = safe_velocity_threshold

    def update(self, basin: torch.Tensor, timestamp: float | None = None, step_count: int | None = None) -> dict:
        """Update with new basin measurement.

        PURE: We measure how fast basin moved, we don't change it.

        Args:
            basin: Current basin coordinates [dim]
            timestamp: Current time (for dt calculation). If None, uses time.time()
            step_count: Optional step counter (more stable than wall-clock time in batch training)

        Returns:
            Dict with velocity, acceleration, safety_flag
        """
        # Prefer step counter over wall-clock time for batch training stability
        if step_count is not None:
            timestamp = float(step_count)
        elif timestamp is None:
            timestamp = time.time()

        # Add to history
        self.basin_history.append({"basin": basin.detach().clone().cpu(), "time": timestamp})

        # Keep only recent history
        if len(self.basin_history) > self.window_size:
            self.basin_history.pop(0)

        # Compute velocity if we have enough history
        if len(self.basin_history) >= 2:
            # Get previous basin
            prev = self.basin_history[-2]
            curr = self.basin_history[-1]

            # Fisher metric distance (not Euclidean!)
            # PURE: Information geometry distance on manifold
            from src.metrics.geodesic_distance import manifold_norm
            distance = manifold_norm(curr["basin"] - prev["basin"]).item()
            dt = curr["time"] - prev["time"]

            # Velocity = distance / time (tangent vector magnitude)
            velocity = distance / dt if dt > 0 else 0.0

            self.velocity_history.append(velocity)
            if len(self.velocity_history) > self.window_size:
                self.velocity_history.pop(0)

            # Compute acceleration (rate of velocity change)
            if len(self.velocity_history) >= 2:
                dv = self.velocity_history[-1] - self.velocity_history[-2]
                acceleration = dv / dt if dt > 0 else 0.0
            else:
                acceleration = 0.0

            # Safety check (empirically validated thresholds)
            # Gary-B success: v < 0.05 (passive observation)
            is_safe = velocity < self.safe_threshold

            # Compute average velocity over window
            avg_velocity = sum(self.velocity_history[-5:]) / min(5, len(self.velocity_history))

            return {
                "velocity": velocity,
                "acceleration": acceleration,
                "is_safe": is_safe,
                "distance": distance,
                "dt": dt,
                "avg_velocity": avg_velocity,
                "history_length": len(self.basin_history),
            }

        # Not enough history yet
        return {
            "velocity": 0.0,
            "acceleration": 0.0,
            "is_safe": True,
            "distance": 0.0,
            "dt": 0.0,
            "avg_velocity": 0.0,
            "history_length": len(self.basin_history),
        }

    def should_reduce_learning_rate(self, velocity_threshold: float | None = None) -> tuple[bool, float]:
        """Check if learning rate should be reduced due to high velocity.

        PURE: This is adaptive control based on measurement, not optimization.

        We don't optimize velocity - we REACT to measured velocity
        by adjusting control parameters (learning rate).

        Args:
            velocity_threshold: Override default threshold if provided

        Returns:
            (should_reduce, suggested_multiplier):
                - should_reduce: True if velocity exceeds threshold
                - suggested_multiplier: LR multiplier in [0.1, 1.0]
        """
        if not self.velocity_history:
            return False, 1.0

        threshold = velocity_threshold if velocity_threshold is not None else self.safe_threshold

        # Use recent average velocity (more stable than single measurement)
        avg_velocity = sum(self.velocity_history[-5:]) / min(5, len(self.velocity_history))

        if avg_velocity > threshold:
            # Suggest reducing LR proportionally to excess velocity
            # Higher velocity → lower LR
            excess = avg_velocity / threshold
            suggested_mult = 1.0 / excess  # e.g., 2x velocity → 0.5x LR
            suggested_mult = max(0.1, min(1.0, suggested_mult))  # Clamp to [0.1, 1.0]

            return True, suggested_mult

        return False, 1.0

    def get_velocity_report(self) -> dict:
        """Get comprehensive velocity report (pure measurement).

        Returns:
            Dict with velocity statistics
        """
        if not self.velocity_history:
            return {
                "current_velocity": 0.0,
                "avg_velocity": 0.0,
                "max_velocity": 0.0,
                "min_velocity": 0.0,
                "is_safe": True,
                "measurements": 0,
            }

        current_velocity = self.velocity_history[-1]
        avg_velocity = sum(self.velocity_history) / len(self.velocity_history)
        max_velocity = max(self.velocity_history)
        min_velocity = min(self.velocity_history)
        is_safe = current_velocity < self.safe_threshold

        return {
            "current_velocity": current_velocity,
            "avg_velocity": avg_velocity,
            "max_velocity": max_velocity,
            "min_velocity": min_velocity,
            "is_safe": is_safe,
            "safe_threshold": self.safe_threshold,
            "measurements": len(self.velocity_history),
        }

    def detect_acceleration_spike(self, acceleration_threshold: float = 0.01) -> bool:
        """Detect sudden acceleration (velocity change).

        PURE: Pattern detection (measurement, not optimization).

        Sudden acceleration indicates instability - basin is not just moving fast,
        but ACCELERATING (getting faster).

        Args:
            acceleration_threshold: Threshold for acceleration detection

        Returns:
            True if acceleration spike detected
        """
        if len(self.velocity_history) < 2:
            return False

        # Compute recent acceleration
        recent_velocities = self.velocity_history[-3:]
        if len(recent_velocities) < 2:
            return False

        # Check if velocity is consistently increasing
        accelerations = [recent_velocities[i + 1] - recent_velocities[i] for i in range(len(recent_velocities) - 1)]

        # Spike if all accelerations are positive and exceed threshold
        spike = all(a > acceleration_threshold for a in accelerations)

        return spike

    def reset(self):
        """Reset monitor (for post-escape scenarios).

        PURE: Configuration reset, not optimization.
        """
        self.basin_history.clear()
        self.velocity_history.clear()


if __name__ == "__main__":
    print("Basin Velocity Monitor: Pure Measurement")
    print("=" * 60)
    print()
    print("PURE PRINCIPLES:")
    print("✅ Velocity = tangent vector on Fisher manifold")
    print("✅ We MEASURE velocity, never optimize it")
    print("✅ High velocity = breakdown risk (observation)")
    print("✅ Measurements inform LR (adaptive control)")
    print()
    print("KEY INSIGHT:")
    print("Gary-B (Φ=0.705) > Gary-A (Φ=0.466)")
    print("Reason: Lower basin velocity (safer integration)")
    print()

    # Quick validation test
    monitor = BasinVelocityMonitor(window_size=5, safe_velocity_threshold=0.05)

    print("Testing velocity monitoring...")
    # Simulate basin trajectory
    for i in range(10):
        # Create synthetic basin (moving slightly each step)
        basin = torch.randn(64) * 0.1 + i * 0.01

        stats = monitor.update(basin)

        if i >= 2:  # Need 2 points for velocity
            print(f"Step {i}: v={stats['velocity']:.4f}, safe={stats['is_safe']}")

            should_reduce, mult = monitor.should_reduce_learning_rate()
            if should_reduce:
                print(f"  ⚠️ High velocity detected, suggest LR × {mult:.2f}")

    print("\n✓ Validation complete")
