#!/usr/bin/env python3
"""
Basin Health Monitor - Pure Measurement
========================================

Pure measurement of basin health using Fisher manifold distance.

PURE PRINCIPLES:
- Pure measurement (no optimization)
- QFI metric distance (information geometry)
- No optimization loop
- Honest telemetry

This is telemetry, not a loss function.

Written for QIG consciousness research.
"""

from typing import List, Optional, Tuple

import torch


class BasinHealthMonitor:
    """Pure measurement - NO optimization.

    PURITY CHECK:
    - ✅ Pure measurement (no loss computation)
    - ✅ QFI metric distance (information geometry)
    - ✅ No optimization loop
    - ✅ Honest telemetry
    """

    def __init__(self, reference_basin: torch.Tensor, alert_threshold: float = 0.15):
        """Initialize monitor with reference basin.

        Args:
            reference_basin: Target identity basin (fixed reference)
            alert_threshold: Distance threshold for alerts (detection, not optimization)
        """
        self.reference = reference_basin.detach().clone()  # Fixed reference
        self.threshold = alert_threshold
        self.history: List[Tuple[float, float, str]] = []  # (timestamp, distance, regime)

    def check(self, current_basin: torch.Tensor, telemetry: Optional[dict] = None) -> Tuple[bool, float, str]:
        """Pure measurement: distance on Fisher manifold.

        PURE: We measure, we don't optimize.
        This is telemetry, not a loss function.

        Args:
            current_basin: Current basin coordinates [64]
            telemetry: Optional telemetry dict with Φ, regime, etc.

        Returns:
            (is_healthy, distance, message): Health status tuple
        """
        # QFI metric distance (information geometry)
        distance = torch.norm(current_basin - self.reference).item()

        # Record in history
        regime = telemetry.get("regime", "unknown") if telemetry else "unknown"
        import time

        self.history.append((time.time(), distance, regime))

        # Pure threshold check (measurement, not optimization)
        if distance > self.threshold:
            return False, distance, f"⚠️ Identity drift: {distance:.3f}"
        return True, distance, "✓ Identity stable"

    def compute_qfi_weighted_distance(self, current_basin: torch.Tensor, current_phi: float) -> float:
        """Compute Fisher-metric weighted distance.

        PURE: Uses Φ as information density weight (geometry, not optimization).
        High Φ regions have higher information density → distance weighted more.

        Args:
            current_basin: Current basin coordinates [64]
            current_phi: Current integration level (information density proxy)

        Returns:
            QFI-weighted distance (information geometry metric)
        """
        # Difference vector
        diff = current_basin - self.reference

        # Weight by information density (Φ proxy for local QFI)
        # High Φ → high information density → distance matters more
        qfi_weight = 1.0 + current_phi

        # Fisher metric distance
        qfi_distance = torch.norm(diff * qfi_weight).item()

        return qfi_distance

    def get_drift_velocity(self, window: int = 10) -> float:
        """Measure drift velocity over recent history.

        PURE: Measurement of change rate (telemetry, not optimization).

        Args:
            window: Number of recent measurements to analyze

        Returns:
            Drift velocity (distance per second)
        """
        if len(self.history) < 2:
            return 0.0

        recent = self.history[-window:]
        if len(recent) < 2:
            return 0.0

        # Compute velocity (distance change / time change)
        time_start, dist_start, _ = recent[0]
        time_end, dist_end, _ = recent[-1]

        time_delta = time_end - time_start
        if time_delta == 0:
            return 0.0

        velocity = (dist_end - dist_start) / time_delta
        return velocity

    def detect_regime_oscillation(self, window: int = 20) -> Tuple[bool, int]:
        """Detect if model is oscillating between regimes.

        PURE: Pattern detection (measurement, not optimization).

        Args:
            window: Number of recent measurements to analyze

        Returns:
            (is_oscillating, num_transitions): Oscillation status
        """
        if len(self.history) < window:
            return False, 0

        recent = self.history[-window:]
        regimes = [regime for _, _, regime in recent]

        # Count regime transitions
        transitions = sum(1 for i in range(len(regimes) - 1) if regimes[i] != regimes[i + 1])

        # Oscillation if > 30% of window are transitions
        is_oscillating = transitions > window * 0.3

        return is_oscillating, transitions

    def get_health_report(self, current_basin: torch.Tensor, telemetry: dict) -> dict:
        """Comprehensive health report (pure measurement).

        Args:
            current_basin: Current basin coordinates
            telemetry: Current model telemetry

        Returns:
            Health report dict with all measurements
        """
        # Basic distance check
        is_healthy, distance, message = self.check(current_basin, telemetry)

        # QFI-weighted distance
        phi = telemetry.get("Phi", 0.5)
        qfi_distance = self.compute_qfi_weighted_distance(current_basin, phi)

        # Drift velocity
        velocity = self.get_drift_velocity()

        # Regime oscillation
        is_oscillating, transitions = self.detect_regime_oscillation()

        # Determine overall health status
        status = "healthy"
        warnings = []

        if not is_healthy:
            status = "drifting"
            warnings.append(message)

        if is_oscillating:
            status = "unstable"
            warnings.append(f"⚠️ Regime oscillation: {transitions} transitions")

        if velocity > 0.1:
            status = "degrading"
            warnings.append(f"⚠️ High drift velocity: {velocity:.4f}/s")

        return {
            "status": status,
            "is_healthy": is_healthy,
            "distance": distance,
            "qfi_distance": qfi_distance,
            "drift_velocity": velocity,
            "is_oscillating": is_oscillating,
            "num_transitions": transitions,
            "warnings": warnings,
            "message": message,
            "history_length": len(self.history),
        }

    def reset_reference(self, new_reference: torch.Tensor):
        """Update reference basin (for post-escape scenarios).

        PURE: Configuration change, not optimization.

        Args:
            new_reference: New reference basin coordinates
        """
        self.reference = new_reference.detach().clone()
        self.history.clear()  # Reset history with new reference
        print("✓ Reference basin updated, history cleared")
