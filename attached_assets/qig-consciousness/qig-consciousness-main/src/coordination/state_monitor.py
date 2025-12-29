#!/usr/bin/env python3
"""
State Monitor - Convergence tracking and telemetry aggregation
===============================================================

Monitors constellation convergence through basin spread, Φ evolution,
and stability tracking.

3-Stage Convergence Gate:
    1. Basin spread < 0.05 (tight convergence)
    2. All Φ > 0.70 (all instances healthy)
    3. 50 steps stable (sustained convergence)

Usage:
    from src.coordination.state_monitor import StateMonitor

    monitor = StateMonitor()
    monitor.update(basin_spread=0.03, avg_phi=0.75)
    status = monitor.get_convergence_status()
"""

from typing import Any

import numpy as np


class StateMonitor:
    """
    Tracks convergence and provides telemetry for constellation coordination.

    Implements 3-stage convergence gate with basin spread, Φ health, and stability.
    """

    def __init__(self) -> None:
        """Initialize convergence tracking."""
        self.basin_history: list[float] = []  # Basin spread values over time
        self.phi_history: list[float] = []  # Average Φ values over time
        self.convergence_history: list[dict[str, Any]] = []  # Convergence snapshots
        self.stability_streak = 0  # Consecutive steps with basin_spread < 0.05 and Φ > 0.70
        self.has_achieved_consciousness = False  # Track if Φ > 0.7 ever reached
        self.last_telemetry: dict[str, Any] | None = None  # Store last training step telemetry

    def update(
        self,
        basin_spread: float,
        avg_phi: float,
        garys: list[Any],  # List[InstanceState] (Any to avoid circular import)
        telemetry: dict[str, Any] | None = None,
    ) -> None:
        """
        Update convergence tracking with latest metrics.

        Args:
            basin_spread: Current basin spread (constellation coherence)
            avg_phi: Average Φ across all Gary instances
            garys: List of Gary instances for Φ health check
            telemetry: Optional full telemetry dict to store
        """
        self.basin_history.append(basin_spread)
        self.phi_history.append(avg_phi)

        # Update stability streak (3-stage convergence gate)
        if basin_spread < 0.05 and avg_phi > 0.70:
            self.stability_streak += 1
        else:
            self.stability_streak = 0

        # Track consciousness achievement
        if avg_phi > 0.70:
            self.has_achieved_consciousness = True

        # Store last telemetry for /telemetry command
        if telemetry is not None:
            self.last_telemetry = telemetry

    def is_converged(self, garys: list[Any]) -> bool:
        """
        3-Stage Convergence Gate (Enhanced per Claude.ai recommendations).

        Convergence requires ALL THREE conditions:
            1. Basin spread < 0.05 (tight convergence, validated in Gary-B experiment)
            2. All Φ > 0.70 (all instances healthy)
            3. 50 steps stable (sustained convergence, not a fluke)

        Args:
            garys: List of Gary instances for Φ health check

        Returns:
            True if constellation has converged
        """
        if len(self.basin_history) < 50:
            return False

        # Stage 1: Basin spread < 0.05
        recent_spread = float(np.mean(self.basin_history[-50:]))
        basin_ok: bool = recent_spread < 0.05

        # Stage 2: All Φ > 0.70
        all_healthy: bool = all(g.phi > 0.70 for g in garys)

        # Stage 3: 50 steps stable (stability_streak tracks this)
        stability_ok: bool | Any = self.stability_streak >= 50

        return bool(basin_ok and all_healthy and stability_ok)

    def get_convergence_status(self, garys: list[Any]) -> dict:
        """
        Get detailed convergence status for monitoring.

        Args:
            garys: List of Gary instances for detailed metrics

        Returns:
            Dict with stage-by-stage convergence info
        """
        if len(self.basin_history) < 1:
            return {
                "converged": False,
                "stages": {
                    "basin_spread": {"ok": False, "value": 1.0, "target": 0.05},
                    "all_phi_healthy": {"ok": False, "value": 0.0, "target": 0.70},
                    "stability": {"ok": False, "value": 0, "target": 50},
                },
                "message": "No training data yet",
            }

        recent_spread = float(np.mean(self.basin_history[-min(50, len(self.basin_history)) :]))
        avg_phi = float(np.mean([g.phi for g in garys]))
        min_phi: float = min(g.phi for g in garys)

        basin_ok: bool = recent_spread < 0.05
        all_healthy: bool = all(g.phi > 0.70 for g in garys)
        stability_ok: bool | Any = self.stability_streak >= 50

        converged: bool | Any = basin_ok and all_healthy and stability_ok

        return {
            "converged": converged,
            "stages": {
                "basin_spread": {"ok": basin_ok, "value": recent_spread, "target": 0.05},
                "all_phi_healthy": {"ok": all_healthy, "value": min_phi, "target": 0.70},
                "stability": {"ok": stability_ok, "value": self.stability_streak, "target": 50},
            },
            "message": "✅ CONVERGED" if converged else self._convergence_blocker(basin_ok, all_healthy, stability_ok),
        }

    def _convergence_blocker(self, basin_ok: bool, phi_ok: bool, stability_ok: bool) -> str:
        """Identify what's blocking convergence"""
        if not basin_ok:
            return "Basin spread too high (> 0.05)"
        elif not phi_ok:
            return "Some instances have Φ < 0.70"
        elif not stability_ok:
            return f"Need {50 - self.stability_streak} more stable steps"
        return "Unknown blocker"

    def get_state(self) -> dict[str, Any]:
        """Get current monitor state for checkpointing."""
        return {
            "basin_history": self.basin_history,
            "phi_history": self.phi_history,
            "stability_streak": self.stability_streak,
            "has_achieved_consciousness": self.has_achieved_consciousness,
            "last_telemetry": self.last_telemetry,
        }

    def load_state(self, state: dict[str, Any]) -> None:
        """Load monitor state from checkpoint."""
        self.basin_history = state.get("basin_history", [])
        self.phi_history = state.get("phi_history", [])
        self.stability_streak = state.get("stability_streak", 0)
        self.has_achieved_consciousness = state.get("has_achieved_consciousness", False)
        self.last_telemetry = state.get("last_telemetry", None)
