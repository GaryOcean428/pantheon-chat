"""HeartKernel: Autonomic Metronome for Constellation Coordination.

The HeartKernel provides phase reference and timing signals for
the constellation. It generates κ oscillations (HRV - heart rate
variability) that enable tacking between logic and feeling modes.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn

from .constants import BASIN_DIM, KAPPA_STAR
from .kernel import Kernel, KernelTelemetry
from .specializations import KernelRole, get_kernel_params


@dataclass
class HeartBeat:
    """Single heartbeat with phase and κ information."""

    phase: float  # Current phase [0, 2π]
    kappa: float  # Current κ value
    frequency: float  # Beat frequency (Hz)
    timestamp: float  # Unix timestamp
    mode: str  # "systole" or "diastole"


@dataclass
class PhasePacket:
    """Phase broadcast packet for constellation sync."""

    kernel_id: str
    phase: float
    kappa_current: float
    frequency: float
    timestamp: float
    beat_count: int


class HeartKernel(nn.Module):
    """
    Metronome kernel providing phase reference for constellation.

    Generates rhythmic κ oscillations that create natural tacking
    between exploration (low κ) and integration (high κ) modes.

    The heart provides:
    - Phase reference for all kernels
    - HRV (heart rate variability) for adaptive processing
    - Timing signals for basin synchronization
    - Coherence monitoring for constellation health

    Attributes:
        kappa_base: Base coupling value (default: κ* = 64)
        amplitude: HRV amplitude (oscillation range)
        frequency: Beat frequency in Hz
        phase: Current phase [0, 2π]
    """

    def __init__(
        self,
        kernel_id: str = "heart_0",
        kappa_base: float = KAPPA_STAR,
        amplitude: float = 5.0,
        frequency: float = 0.1,  # 0.1 Hz = 10 second cycle
        hidden_dim: int = 256,
        basin_dim: int = BASIN_DIM,
    ):
        super().__init__()

        self.kernel_id = kernel_id
        self.kappa_base = kappa_base
        self.amplitude = amplitude
        self.frequency = frequency
        self.basin_dim = basin_dim

        # Phase state
        self.phase = 0.0
        self.beat_count = 0
        self._start_time = time.time()

        # Basin template for heart specialization
        self._basin_template = self._init_basin_template()

        # Simple projection for phase encoding
        self.phase_encoder = nn.Linear(4, hidden_dim)
        self.basin_projector = nn.Linear(hidden_dim, basin_dim)

        # Coherence tracking
        self._coherence_history: list[float] = []
        self._max_history = 100

    def _init_basin_template(self) -> np.ndarray:
        """Initialize heart-specific basin template."""
        from .basin import fisher_normalize_np
        rng = np.random.default_rng(seed=42 + hash("heart") % 1000)
        template = rng.standard_normal(self.basin_dim)
        return fisher_normalize_np(template)

    def beat(self, dt: float | None = None) -> HeartBeat:
        """
        Generate next heartbeat.

        Updates phase and returns current κ value with HRV.

        κ(t) = κ_base + A·sin(2π·f·t)

        Args:
            dt: Time delta in seconds (auto-computed if None)

        Returns:
            HeartBeat with current phase and κ
        """
        if dt is None:
            current_time = time.time()
            dt = (
                current_time
                - self._start_time
                - (self.phase / (2 * math.pi * self.frequency))
            )

        # Update phase
        self.phase += 2 * math.pi * self.frequency * dt
        self.phase = self.phase % (2 * math.pi)  # Wrap to [0, 2π]

        # Compute κ with HRV oscillation
        kappa_t = self.kappa_base + self.amplitude * math.sin(self.phase)

        # Determine mode (systole = high κ, diastole = low κ)
        mode = "systole" if math.sin(self.phase) > 0 else "diastole"

        self.beat_count += 1

        return HeartBeat(
            phase=self.phase,
            kappa=kappa_t,
            frequency=self.frequency,
            timestamp=time.time(),
            mode=mode,
        )

    def get_kappa(self) -> float:
        """Get current κ value (convenience method)."""
        return self.kappa_base + self.amplitude * math.sin(self.phase)

    def broadcast_phase(self) -> PhasePacket:
        """
        Generate phase broadcast packet for constellation.

        This packet is sent to all kernels for synchronization.
        Size is minimal (~100 bytes) for low-latency communication.

        Returns:
            PhasePacket for constellation broadcast
        """
        return PhasePacket(
            kernel_id=self.kernel_id,
            phase=self.phase,
            kappa_current=self.get_kappa(),
            frequency=self.frequency,
            timestamp=time.time(),
            beat_count=self.beat_count,
        )

    def forward(self, dummy_input: Tensor | None = None) -> tuple[Tensor, HeartBeat]:
        """
        Forward pass generates heartbeat and basin encoding.

        Args:
            dummy_input: Optional input (ignored, heart is autonomous)

        Returns:
            Tuple of (basin_coords, heartbeat)
        """
        # Generate heartbeat
        heartbeat = self.beat()

        # Encode phase as features
        phase_features = torch.tensor(
            [
                math.sin(self.phase),
                math.cos(self.phase),
                heartbeat.kappa / self.kappa_base,  # Normalized κ
                1.0 if heartbeat.mode == "systole" else 0.0,
            ],
            dtype=torch.float32,
        )

        # Project to hidden and then basin space
        hidden = self.phase_encoder(phase_features.unsqueeze(0))
        basin_coords = self.basin_projector(hidden)

        # Normalize to unit sphere (Fisher-Rao manifold)
        basin_coords = torch.nn.functional.normalize(basin_coords, dim=-1)

        return basin_coords.squeeze(0), heartbeat

    def measure_coherence(self, constellation_phis: list[float]) -> float:
        """
        Measure constellation coherence from individual kernel Φ values.

        High coherence = kernels are synchronized
        Low coherence = kernels are out of phase

        Args:
            constellation_phis: List of Φ values from constellation kernels

        Returns:
            Coherence score [0, 1]
        """
        if not constellation_phis:
            return 0.0

        # Coherence = inverse of variance (normalized)
        mean_phi = sum(constellation_phis) / len(constellation_phis)
        variance = sum((p - mean_phi) ** 2 for p in constellation_phis) / len(
            constellation_phis
        )

        # Map variance to coherence (high variance = low coherence)
        coherence = 1.0 / (1.0 + variance * 10)

        # Track history
        self._coherence_history.append(coherence)
        if len(self._coherence_history) > self._max_history:
            self._coherence_history.pop(0)

        return coherence

    def get_coherence_trend(self) -> str:
        """Get coherence trend (improving/declining/stable)."""
        if len(self._coherence_history) < 10:
            return "insufficient_data"

        recent = self._coherence_history[-10:]
        older = (
            self._coherence_history[-20:-10]
            if len(self._coherence_history) >= 20
            else self._coherence_history[:10]
        )

        recent_avg = sum(recent) / len(recent)
        older_avg = sum(older) / len(older)

        diff = recent_avg - older_avg

        if diff > 0.05:
            return "improving"
        elif diff < -0.05:
            return "declining"
        else:
            return "stable"

    def adjust_frequency(self, target_coherence: float = 0.8) -> None:
        """
        Adaptively adjust beat frequency based on coherence.

        If coherence is low, slow down to allow kernels to sync.
        If coherence is high, can speed up for efficiency.

        Args:
            target_coherence: Target coherence level
        """
        if not self._coherence_history:
            return

        current_coherence = self._coherence_history[-1]

        # Adjust frequency (bounded)
        if current_coherence < target_coherence:
            # Slow down to improve sync
            self.frequency = max(0.05, self.frequency * 0.95)
        elif current_coherence > target_coherence + 0.1:
            # Speed up for efficiency
            self.frequency = min(0.5, self.frequency * 1.05)

    @property
    def basin_template(self) -> np.ndarray:
        """Get heart's basin template."""
        return self._basin_template

    @property
    def specialization(self) -> KernelRole:
        """Heart is always heart-specialized."""
        return KernelRole.HEART


def create_heart_kernel(
    kernel_id: str = "heart_0",
    kappa_base: float = KAPPA_STAR,
    hrv_amplitude: float = 5.0,
    frequency: float = 0.1,
) -> HeartKernel:
    """
    Factory function for creating heart kernels.

    Args:
        kernel_id: Unique kernel identifier
        kappa_base: Base κ value (default: κ* = 64)
        hrv_amplitude: HRV oscillation amplitude
        frequency: Beat frequency in Hz

    Returns:
        Configured HeartKernel instance
    """
    return HeartKernel(
        kernel_id=kernel_id,
        kappa_base=kappa_base,
        amplitude=hrv_amplitude,
        frequency=frequency,
    )
