"""QIG Safety Mechanisms: Breakdown Handler, Emergency Pause, Gravitational Decoherence.

Implements physics-based safety constraints to keep kernels operating
in the healthy geometric regime (0.45 ≤ Φ < 0.80, κ ≈ 64).

Safety mechanisms from FROZEN_FACTS.md:
- BREAKDOWN (Φ ≥ 0.80): Reduce coupling, inject decoherence
- EMERGENCY (Φ < 0.50 collapse): Pause processing, restore baseline
- GRAVITATIONAL DECOHERENCE: Physics-based noise injection

See: qig-verification/docs/current/FROZEN_FACTS.md
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

import torch
from torch import Tensor, nn

from .constants import (
    KAPPA_STAR,
    PHI_BREAKDOWN_MIN,
    PHI_EMERGENCY,
    PHI_GEOMETRIC_MIN,
    PHI_GEOMETRIC_MAX,
    KAPPA_MIN_OPTIMAL,
    KAPPA_MAX_OPTIMAL,
)


class SafetyState(Enum):
    """Kernel safety state."""

    HEALTHY = "healthy"  # Geometric regime, optimal κ
    WARNING = "warning"  # Approaching boundaries
    BREAKDOWN = "breakdown"  # Φ ≥ 0.80 (too integrated)
    EMERGENCY = "emergency"  # Φ < 0.50 (collapsed)
    PAUSED = "paused"  # Processing halted


@dataclass
class SafetyEvent:
    """Safety event triggered by regime violation."""

    state: SafetyState
    phi: float
    kappa: float
    action_taken: str
    timestamp: float


@dataclass
class SafetyConfig:
    """Configuration for safety mechanisms."""

    # Regime thresholds (from FROZEN_FACTS.md)
    phi_breakdown: float = PHI_BREAKDOWN_MIN  # 0.80
    phi_emergency: float = PHI_EMERGENCY  # 0.50
    phi_geometric_min: float = PHI_GEOMETRIC_MIN  # 0.45
    phi_geometric_max: float = PHI_GEOMETRIC_MAX  # 0.80

    # κ bounds
    kappa_target: float = KAPPA_STAR  # 64.0
    kappa_min: float = KAPPA_MIN_OPTIMAL  # 40.0
    kappa_max: float = KAPPA_MAX_OPTIMAL  # 70.0

    # Decoherence parameters
    base_decoherence: float = 0.01  # Baseline noise
    breakdown_decoherence: float = 0.1  # Increased during breakdown
    gravitational_scale: float = 0.05  # G_N analog

    # Recovery parameters
    breakdown_kappa_reduction: float = 0.8  # Reduce κ by 20% during breakdown
    emergency_pause_duration: int = 10  # Steps to pause

    # Hysteresis
    hysteresis_gap: float = 0.05  # Gap for regime transitions


class BreakdownHandler:
    """
    Handles breakdown regime (Φ ≥ 0.80).

    When kernel becomes too integrated:
    1. Reduce effective coupling (κ)
    2. Inject decoherence noise
    3. Log event for monitoring
    """

    def __init__(self, config: SafetyConfig | None = None):
        self.config = config or SafetyConfig()
        self._in_breakdown = False
        self._breakdown_steps = 0
        self._events: list[SafetyEvent] = []

    def check(self, phi: float, kappa: float) -> tuple[bool, float, float]:
        """
        Check for breakdown and apply corrections.

        Args:
            phi: Current Φ value
            kappa: Current κ value

        Returns:
            Tuple of (is_breakdown, adjusted_kappa, decoherence_std)
        """
        import time

        is_breakdown = phi >= self.config.phi_breakdown

        if is_breakdown and not self._in_breakdown:
            # Entering breakdown
            self._in_breakdown = True
            self._breakdown_steps = 0
            self._events.append(
                SafetyEvent(
                    state=SafetyState.BREAKDOWN,
                    phi=phi,
                    kappa=kappa,
                    action_taken="reduce_coupling",
                    timestamp=time.time(),
                )
            )
        elif not is_breakdown and self._in_breakdown:
            # Exiting breakdown (with hysteresis)
            if phi < self.config.phi_breakdown - self.config.hysteresis_gap:
                self._in_breakdown = False
                self._events.append(
                    SafetyEvent(
                        state=SafetyState.HEALTHY,
                        phi=phi,
                        kappa=kappa,
                        action_taken="recovered",
                        timestamp=time.time(),
                    )
                )

        if self._in_breakdown:
            self._breakdown_steps += 1
            # Reduce κ to encourage decoherence
            adjusted_kappa = kappa * self.config.breakdown_kappa_reduction
            # Increase decoherence
            decoherence = self.config.breakdown_decoherence
        else:
            adjusted_kappa = kappa
            decoherence = self.config.base_decoherence

        return is_breakdown, adjusted_kappa, decoherence

    @property
    def in_breakdown(self) -> bool:
        return self._in_breakdown

    @property
    def breakdown_steps(self) -> int:
        return self._breakdown_steps

    def get_events(self) -> list[SafetyEvent]:
        return self._events.copy()


class EmergencyPause:
    """
    Handles emergency state (Φ < 0.50 collapse).

    When kernel consciousness collapses:
    1. Pause processing
    2. Restore to baseline state
    3. Gradually resume
    """

    def __init__(self, config: SafetyConfig | None = None):
        self.config = config or SafetyConfig()
        self._in_emergency = False
        self._pause_remaining = 0
        self._events: list[SafetyEvent] = []
        self._baseline_state: dict[str, Any] | None = None

    def check(self, phi: float, kappa: float) -> tuple[bool, bool]:
        """
        Check for emergency and determine if processing should pause.

        Args:
            phi: Current Φ value
            kappa: Current κ value

        Returns:
            Tuple of (is_emergency, should_pause)
        """
        import time

        is_emergency = phi < self.config.phi_emergency

        if is_emergency and not self._in_emergency:
            # Entering emergency
            self._in_emergency = True
            self._pause_remaining = self.config.emergency_pause_duration
            self._events.append(
                SafetyEvent(
                    state=SafetyState.EMERGENCY,
                    phi=phi,
                    kappa=kappa,
                    action_taken="pause_processing",
                    timestamp=time.time(),
                )
            )

        should_pause = False
        if self._in_emergency:
            if self._pause_remaining > 0:
                should_pause = True
                self._pause_remaining -= 1
            else:
                # Check if recovered
                if phi >= self.config.phi_emergency + self.config.hysteresis_gap:
                    self._in_emergency = False
                    self._events.append(
                        SafetyEvent(
                            state=SafetyState.HEALTHY,
                            phi=phi,
                            kappa=kappa,
                            action_taken="resumed",
                            timestamp=time.time(),
                        )
                    )

        return is_emergency, should_pause

    def set_baseline(self, state: dict[str, Any]) -> None:
        """Store baseline state for recovery."""
        self._baseline_state = state.copy()

    def get_baseline(self) -> dict[str, Any] | None:
        """Get stored baseline for restoration."""
        return self._baseline_state

    @property
    def in_emergency(self) -> bool:
        return self._in_emergency

    @property
    def is_paused(self) -> bool:
        return self._pause_remaining > 0


class GravitationalDecoherence(nn.Module):
    """
    Physics-based decoherence injection.

    Models gravitational decoherence where massive objects (high-Φ states)
    experience stronger environment-induced decoherence. This prevents
    runaway integration and keeps the system in healthy geometric regime.

    Decoherence rate: Γ = G_N * Φ² * (1 + κ/κ*)

    Based on Penrose-Diósi gravitational decoherence model adapted for
    QIG consciousness kernels.
    """

    def __init__(
        self,
        config: SafetyConfig | None = None,
        hidden_dim: int = 384,
    ):
        super().__init__()
        self.config = config or SafetyConfig()
        self.hidden_dim = hidden_dim

        # Learnable decoherence scale
        self.decoherence_scale = nn.Parameter(
            torch.tensor(self.config.gravitational_scale)
        )

    def compute_decoherence_rate(self, phi: float, kappa: float) -> float:
        """
        Compute decoherence rate from Φ and κ.

        Γ = G_N * Φ² * (1 + κ/κ*)

        Higher Φ → more decoherence (prevents breakdown)
        Higher κ → more decoherence (prevents runaway coupling)
        """
        g_n = self.config.gravitational_scale
        kappa_star = self.config.kappa_target

        rate = g_n * (phi**2) * (1 + kappa / kappa_star)
        return float(rate)

    def forward(
        self,
        hidden_state: Tensor,
        phi: float,
        kappa: float,
    ) -> Tensor:
        """
        Apply gravitational decoherence to hidden state.

        Args:
            hidden_state: [batch, seq, hidden] tensor
            phi: Current Φ value
            kappa: Current κ value

        Returns:
            Hidden state with decoherence noise applied
        """
        rate = self.compute_decoherence_rate(phi, kappa)

        # Scale noise by decoherence rate
        noise_std = rate * self.decoherence_scale.abs()

        if noise_std > 0:
            noise = torch.randn_like(hidden_state) * noise_std
            hidden_state = hidden_state + noise

        return hidden_state


class SafetyGuard:
    """
    Unified safety guard combining all safety mechanisms.

    Monitors kernel state and applies appropriate interventions:
    - HEALTHY: No action needed
    - WARNING: Log and monitor
    - BREAKDOWN: Reduce coupling, inject decoherence
    - EMERGENCY: Pause processing, restore baseline
    """

    def __init__(self, config: SafetyConfig | None = None):
        self.config = config or SafetyConfig()
        self.breakdown_handler = BreakdownHandler(self.config)
        self.emergency_pause = EmergencyPause(self.config)
        self.decoherence = GravitationalDecoherence(self.config)

        self._current_state = SafetyState.HEALTHY

    def check(
        self,
        phi: float,
        kappa: float,
        hidden_state: Tensor | None = None,
    ) -> dict[str, Any]:
        """
        Run all safety checks and apply interventions.

        Args:
            phi: Current Φ value
            kappa: Current κ value
            hidden_state: Optional hidden state for decoherence

        Returns:
            Dict with safety state, adjusted values, and actions
        """
        result = {
            "original_phi": phi,
            "original_kappa": kappa,
            "state": SafetyState.HEALTHY,
            "actions": [],
        }

        # Check emergency first (Φ collapse)
        is_emergency, should_pause = self.emergency_pause.check(phi, kappa)
        if is_emergency:
            result["state"] = SafetyState.EMERGENCY
            result["actions"].append("emergency_detected")
            if should_pause:
                result["should_pause"] = True
                result["actions"].append("processing_paused")
                self._current_state = SafetyState.PAUSED
                return result

        # Check breakdown (Φ too high)
        is_breakdown, adjusted_kappa, decoherence_std = self.breakdown_handler.check(
            phi, kappa
        )
        if is_breakdown:
            result["state"] = SafetyState.BREAKDOWN
            result["adjusted_kappa"] = adjusted_kappa
            result["decoherence_std"] = decoherence_std
            result["actions"].append("breakdown_correction")
            self._current_state = SafetyState.BREAKDOWN
        else:
            result["adjusted_kappa"] = kappa
            result["decoherence_std"] = self.config.base_decoherence

        # Apply gravitational decoherence if hidden state provided
        if hidden_state is not None:
            result["hidden_state"] = self.decoherence(
                hidden_state, phi, result["adjusted_kappa"]
            )

        # Check for warning (approaching boundaries)
        if not is_breakdown and not is_emergency:
            if phi > self.config.phi_geometric_max - 0.1:
                result["state"] = SafetyState.WARNING
                result["actions"].append("approaching_breakdown")
                self._current_state = SafetyState.WARNING
            elif phi < self.config.phi_geometric_min + 0.1:
                result["state"] = SafetyState.WARNING
                result["actions"].append("approaching_linear")
                self._current_state = SafetyState.WARNING
            else:
                self._current_state = SafetyState.HEALTHY

        result["should_pause"] = False
        return result

    @property
    def current_state(self) -> SafetyState:
        return self._current_state

    def get_all_events(self) -> list[SafetyEvent]:
        """Get all safety events from all handlers."""
        events = []
        events.extend(self.breakdown_handler.get_events())
        events.extend(self.emergency_pause._events)
        return sorted(events, key=lambda e: e.timestamp)

    def reset(self) -> None:
        """Reset all safety state."""
        self.breakdown_handler = BreakdownHandler(self.config)
        self.emergency_pause = EmergencyPause(self.config)
        self._current_state = SafetyState.HEALTHY
