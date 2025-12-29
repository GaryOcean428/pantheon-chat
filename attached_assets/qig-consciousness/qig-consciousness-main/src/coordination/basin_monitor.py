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

from typing import Optional

import torch
import torch.nn as nn


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
        self.history: list[tuple[float, float, str]] = []  # (timestamp, distance, regime)

    def check(self, current_basin: torch.Tensor, telemetry: dict | None = None) -> tuple[bool, float, str]:
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
        from src.metrics.geodesic_distance import manifold_norm
        distance = manifold_norm(current_basin - self.reference).item()

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
        from src.metrics.geodesic_distance import manifold_norm
        qfi_distance = manifold_norm(diff * qfi_weight).item()

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

    def detect_regime_oscillation(self, window: int = 20) -> tuple[bool, int]:
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


class BasinDriftMonitor:
    """
    Basin Drift Monitor with Intervention Triggers.

    GEOMETRIC PURITY: Tracks basin drift and triggers interventions
    when identity preservation is at risk.

    From comprehensive review:
    - Track drift history (last 100 measurements)
    - Detect drift acceleration (>30% increase)
    - Trigger sleep consolidation when drift > 0.15

    Alert Types:
    - drift_threshold: Absolute drift exceeds threshold
    - drift_acceleration: Drift rate increasing rapidly
    - identity_risk: Sustained high drift threatens identity

    Usage:
        monitor = BasinDriftMonitor(alert_threshold=0.15)
        alert = monitor.update(basin_distance, instance_name="gary_a")

        if alert["alert"]:
            if alert["type"] == "drift_threshold":
                trigger_immediate_sleep()
            elif alert["type"] == "drift_acceleration":
                trigger_sleep_consolidation()
    """

    def __init__(
        self,
        alert_threshold: float = 0.15,
        acceleration_threshold: float = 0.30,
        history_size: int = 100,
        velocity_window: int = 10,
    ):
        """
        Initialize drift monitor.

        Args:
            alert_threshold: Absolute drift threshold for alerts (0.15 = identity risk)
            acceleration_threshold: Drift acceleration threshold (0.30 = 30% increase)
            history_size: Number of measurements to track
            velocity_window: Window size for velocity calculations
        """
        self.alert_threshold = alert_threshold
        self.acceleration_threshold = acceleration_threshold
        self.history_size = history_size
        self.velocity_window = velocity_window

        # Per-instance drift history: {name: [(timestamp, drift), ...]}
        self.drift_history: dict[str, list[tuple[float, float]]] = {}

        # Alert cooldown to prevent spam: {name: last_alert_time}
        self._last_alert: dict[str, float] = {}
        self._alert_cooldown = 30.0  # seconds

    def update(
        self,
        basin_distance: float,
        instance_name: str = "default",
    ) -> dict:
        """
        Update drift measurement and check for alerts.

        Args:
            basin_distance: Current basin distance from reference
            instance_name: Name of the instance (e.g., "gary_a", "ocean")

        Returns:
            Alert dict with keys:
            - alert: bool (True if intervention needed)
            - type: str | None (alert type if alert=True)
            - message: str (human-readable message)
            - drift: float (current drift)
            - velocity: float (drift velocity)
            - acceleration: float (drift acceleration)
            - needs_sleep: bool (immediate sleep recommended)
            - needs_consolidation: bool (consolidation recommended)
        """
        import time
        now = time.time()

        # Initialize history for new instance
        if instance_name not in self.drift_history:
            self.drift_history[instance_name] = []
            self._last_alert[instance_name] = 0.0

        # Add measurement
        history = self.drift_history[instance_name]
        history.append((now, basin_distance))

        # Trim to history_size
        if len(history) > self.history_size:
            history.pop(0)

        # Compute velocity and acceleration
        velocity = self._compute_velocity(history)
        acceleration = self._compute_acceleration(history)

        # Check for alerts
        alert_type: str | None = None
        needs_sleep = False
        needs_consolidation = False

        # Check cooldown
        can_alert = (now - self._last_alert[instance_name]) > self._alert_cooldown

        if can_alert:
            # Priority 1: Absolute threshold breach (identity risk)
            if basin_distance > self.alert_threshold:
                alert_type = "drift_threshold"
                needs_sleep = True
                self._last_alert[instance_name] = now

            # Priority 2: Rapid acceleration (30% increase in drift rate)
            elif acceleration > self.acceleration_threshold:
                alert_type = "drift_acceleration"
                needs_consolidation = True
                self._last_alert[instance_name] = now

            # Priority 3: Sustained moderate drift (identity risk over time)
            elif self._sustained_high_drift(history):
                alert_type = "identity_risk"
                needs_consolidation = True
                self._last_alert[instance_name] = now

        # Build alert message
        if alert_type == "drift_threshold":
            message = f"⚠️ [{instance_name}] Basin drift {basin_distance:.3f} exceeds threshold {self.alert_threshold:.3f} - IMMEDIATE SLEEP RECOMMENDED"
        elif alert_type == "drift_acceleration":
            message = f"⚠️ [{instance_name}] Drift accelerating {acceleration:.1%} - CONSOLIDATION RECOMMENDED"
        elif alert_type == "identity_risk":
            message = f"⚠️ [{instance_name}] Sustained drift detected - CONSOLIDATION RECOMMENDED"
        else:
            message = f"✓ [{instance_name}] Basin stable: drift={basin_distance:.3f}, velocity={velocity:.4f}"

        return {
            "alert": alert_type is not None,
            "type": alert_type,
            "message": message,
            "drift": basin_distance,
            "velocity": velocity,
            "acceleration": acceleration,
            "needs_sleep": needs_sleep,
            "needs_consolidation": needs_consolidation,
            "instance": instance_name,
        }

    def _compute_velocity(self, history: list[tuple[float, float]]) -> float:
        """Compute drift velocity (drift change per second)."""
        if len(history) < 2:
            return 0.0

        window = history[-self.velocity_window:]
        if len(window) < 2:
            return 0.0

        t_start, d_start = window[0]
        t_end, d_end = window[-1]

        dt = t_end - t_start
        if dt == 0:
            return 0.0

        return (d_end - d_start) / dt

    def _compute_acceleration(self, history: list[tuple[float, float]]) -> float:
        """
        Compute drift acceleration (relative change in velocity).

        Returns percentage increase in drift rate.
        """
        if len(history) < self.velocity_window * 2:
            return 0.0

        # Split into two windows
        mid = len(history) // 2
        first_half = history[:mid]
        second_half = history[mid:]

        # Compute velocity for each half
        v1 = self._compute_velocity(first_half) if len(first_half) >= 2 else 0.0
        v2 = self._compute_velocity(second_half) if len(second_half) >= 2 else 0.0

        # Compute relative acceleration
        if abs(v1) < 1e-6:
            if abs(v2) < 1e-6:
                return 0.0
            return 1.0  # Went from zero to non-zero

        return (v2 - v1) / abs(v1)

    def _sustained_high_drift(self, history: list[tuple[float, float]], threshold_ratio: float = 0.5) -> bool:
        """
        Check if drift has been above half-threshold for sustained period.

        Args:
            history: Drift history
            threshold_ratio: Fraction of history that must be above threshold

        Returns:
            True if sustained moderate-to-high drift detected
        """
        if len(history) < 20:
            return False

        # Check last 20 measurements
        recent = history[-20:]
        half_threshold = self.alert_threshold * 0.5

        above_half = sum(1 for _, d in recent if d > half_threshold)
        return above_half / len(recent) > threshold_ratio

    def get_all_instance_status(self) -> dict:
        """
        Get drift status for all monitored instances.

        Returns:
            Dict with instance name -> status dict
        """
        status = {}
        for name, history in self.drift_history.items():
            if len(history) == 0:
                status[name] = {"drift": 0.0, "velocity": 0.0, "healthy": True}
            else:
                _, latest_drift = history[-1]
                velocity = self._compute_velocity(history)
                status[name] = {
                    "drift": latest_drift,
                    "velocity": velocity,
                    "healthy": latest_drift < self.alert_threshold,
                    "history_length": len(history),
                }
        return status

    def reset_instance(self, instance_name: str):
        """Reset tracking for an instance (e.g., after sleep consolidation)."""
        if instance_name in self.drift_history:
            self.drift_history[instance_name] = []
            self._last_alert[instance_name] = 0.0
            print(f"✓ Drift monitor reset for {instance_name}")

    def reset_all(self):
        """Reset all tracking."""
        self.drift_history.clear()
        self._last_alert.clear()
        print("✓ Drift monitor reset for all instances")
