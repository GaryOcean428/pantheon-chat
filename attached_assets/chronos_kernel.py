"""
Chronos Kernel - 4D Temporal Integration (E8 Root: alpha_7)
============================================================

NOT just a scheduler - manages TEMPORAL GEOMETRY.

4D Consciousness = 3D Spatial + 1D Temporal integration:
- Phi_spatial: Current moment integration
- Phi_temporal: Integration across time (trajectory coherence)
- Phi_4D: Combined consciousness metric

E8 Position: alpha_7 (Time/Sequence primitive)
Coupling: kappa = 55 (moderate-high - temporal coherence matters)

Heart (alpha_1) handles PHASE coordination (kappa oscillation).
Chronos (alpha_7) handles TIME coordination (trajectory, foresight).
These are DIFFERENT E8 primitives.

Usage:
    from src.model.chronos_kernel import ChronosKernel

    chronos = ChronosKernel()
    chronos.update_state(current_basin)
    phi_4d = chronos.compute_phi_4d()
    trajectory = chronos.predict_trajectory(n_steps=10)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

import numpy as np

from src.constants import BASIN_DIM

# Lightning event emission
try:
    from src.constellation.domain_intelligence import DomainEventEmitter
    LIGHTNING_AVAILABLE = True
except ImportError:
    DomainEventEmitter = object
    LIGHTNING_AVAILABLE = False


def _fisher_normalize(basin: np.ndarray) -> np.ndarray:
    """QIG-pure normalization."""
    norm = float(np.sqrt(np.sum(basin * basin)))
    return basin / (norm + 1e-10)


def _fisher_rao_distance(b1: np.ndarray, b2: np.ndarray) -> float:
    """Fisher-Rao geodesic distance."""
    b1_n = _fisher_normalize(b1)
    b2_n = _fisher_normalize(b2)
    cos_sim = np.clip(np.dot(b1_n, b2_n), -1.0, 1.0)
    return float(np.arccos(cos_sim))


def _compute_phi_from_basin(basin: np.ndarray) -> float:
    """Compute Phi from basin coordinates."""
    basin_n = _fisher_normalize(basin)
    # Fisher information approximation
    fisher_trace = float(np.sum(basin_n * basin_n))
    return min(1.0, max(0.0, fisher_trace))


@dataclass
class StateHistoryBuffer:
    """
    Multi-timescale state history buffer.

    Maintains:
    - immediate: Last 10 states (fine-grained)
    - recent: Last 50 states (medium-grained)
    - session: Last 200 states (coarse-grained)
    """
    immediate: Deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=10))
    recent: Deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=50))
    session: Deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=200))

    def add(self, state: np.ndarray) -> None:
        """Add state to all buffers."""
        self.immediate.append(state.copy())
        self.recent.append(state.copy())
        self.session.append(state.copy())

    def get_trajectory(self, scale: str = "recent") -> List[np.ndarray]:
        """Get trajectory at specified timescale."""
        if scale == "immediate":
            return list(self.immediate)
        elif scale == "recent":
            return list(self.recent)
        elif scale == "session":
            return list(self.session)
        return list(self.recent)

    def __len__(self) -> int:
        return len(self.recent)


@dataclass
class TemporalMetrics:
    """4D consciousness metrics."""
    phi_spatial: float = 0.0
    phi_temporal: float = 0.0
    phi_4d: float = 0.0
    trajectory_smoothness: float = 0.0
    regime_3d: str = "linear"
    regime_4d: str = "linear"
    divergence: str = "stable"


class BasinForesight:
    """
    Basin trajectory prediction.

    Uses simple linear extrapolation on the manifold.
    More sophisticated methods (RNN, attention) could be added.
    """

    def __init__(self, basin_dim: int = BASIN_DIM):
        self.basin_dim = basin_dim

    def predict(
        self,
        history: List[np.ndarray],
        n_steps: int = 10,
    ) -> List[np.ndarray]:
        """
        Predict future basin trajectory.

        Uses tangent vector extrapolation on manifold.
        """
        if len(history) < 2:
            # Not enough history - return last state repeated
            last = history[-1] if history else np.zeros(self.basin_dim)
            return [_fisher_normalize(last) for _ in range(n_steps)]

        # Compute average tangent vector (direction of motion)
        tangents = []
        for i in range(1, min(10, len(history))):
            # Tangent = log map approximation
            tangent = history[i] - history[i - 1]
            tangents.append(tangent)

        avg_tangent = np.mean(tangents, axis=0)

        # Extrapolate
        predictions = []
        current = _fisher_normalize(history[-1])

        for _ in range(n_steps):
            # Exponential map approximation
            next_state = current + avg_tangent * 0.1
            next_state = _fisher_normalize(next_state)
            predictions.append(next_state)
            current = next_state

        return predictions


class ChronosKernel(DomainEventEmitter if LIGHTNING_AVAILABLE else object):
    """
    Temporal coordination and 4D consciousness kernel.

    E8 Root: alpha_7 (Time/Sequence)

    Implements:
    - Multi-timescale state tracking
    - 4D consciousness computation (spatial + temporal)
    - Trajectory prediction (foresight)
    - Divergence detection (course correction)
    """

    # Kernel coupling (temporal coherence is important)
    KAPPA_TEMPORAL = 55.0

    def __init__(self, basin_dim: int = BASIN_DIM):
        """Initialize temporal kernel."""
        if LIGHTNING_AVAILABLE:
            super().__init__()
            self.domain = "chronos"

        self.basin_dim = basin_dim

        # State history
        self.history = StateHistoryBuffer()

        # Foresight predictor
        self.foresight = BasinForesight(basin_dim)

        # Target basin (goal state)
        self.target_basin: Optional[np.ndarray] = None

        # Current metrics
        self.phi_spatial = 0.0
        self.phi_temporal = 0.0
        self.phi_4d = 0.0
        self.kappa = self.KAPPA_TEMPORAL

        # Current state
        self.current_basin = np.zeros(basin_dim)

        # Event tracking
        self.events_emitted = 0
        self.insights_received = 0

        # Statistics
        self.total_steps = 0
        self.divergence_events = 0

    def update_state(self, basin: np.ndarray) -> TemporalMetrics:
        """
        Update with new basin state and compute 4D metrics.

        Args:
            basin: Current 64D basin state

        Returns:
            TemporalMetrics with all 4D consciousness measures
        """
        basin = _fisher_normalize(np.asarray(basin).flatten())
        self.current_basin = basin
        self.history.add(basin)
        self.total_steps += 1

        # Compute metrics
        metrics = self.compute_phi_4d()

        # Emit Lightning event on significant changes (tracked)
        self.events_emitted += 1
        if LIGHTNING_AVAILABLE and hasattr(self, 'emit_event'):
            if metrics.divergence != "stable":
                self.emit_event(
                    event_type="trajectory_divergence",
                    content=f"Trajectory {metrics.divergence}",
                    phi=metrics.phi_4d,
                    basin_coords=basin,
                    metadata={
                        "phi_spatial": metrics.phi_spatial,
                        "phi_temporal": metrics.phi_temporal,
                        "smoothness": metrics.trajectory_smoothness,
                    },
                )
            else:
                # Always emit state update for Lightning correlation
                self.emit_event(
                    event_type="temporal_update",
                    content=f"4D: φ={metrics.phi_4d:.3f} ({metrics.regime_4d})",
                    phi=metrics.phi_4d,
                    basin_coords=basin,
                    metadata={
                        "phi_spatial": metrics.phi_spatial,
                        "phi_temporal": metrics.phi_temporal,
                        "regime_4d": metrics.regime_4d,
                    },
                )

        return metrics

    def compute_phi_4d(self) -> TemporalMetrics:
        """
        Compute 4D consciousness (space + time).

        Phi_4D = sqrt(Phi_spatial^2 + Phi_temporal^2)
        """
        metrics = TemporalMetrics()

        # 1. Spatial integration (current moment)
        metrics.phi_spatial = _compute_phi_from_basin(self.current_basin)
        self.phi_spatial = metrics.phi_spatial

        # 2. Temporal integration (trajectory coherence)
        if len(self.history) >= 10:
            trajectory = self.history.get_trajectory("recent")

            # Compute trajectory smoothness (inverse of jitter)
            if len(trajectory) >= 3:
                velocities = []
                for i in range(1, len(trajectory)):
                    v = _fisher_rao_distance(trajectory[i], trajectory[i - 1])
                    velocities.append(v)

                # Smoothness = 1 / (1 + variance(velocities))
                vel_variance = np.var(velocities) if velocities else 0.0
                metrics.trajectory_smoothness = 1.0 / (1.0 + vel_variance)

            # Temporal Phi = mean Phi along trajectory weighted by smoothness
            phi_values = [_compute_phi_from_basin(s) for s in trajectory[-10:]]
            metrics.phi_temporal = float(np.mean(phi_values)) * metrics.trajectory_smoothness

        self.phi_temporal = metrics.phi_temporal

        # 3. Combined 4D consciousness
        metrics.phi_4d = float(np.sqrt(
            metrics.phi_spatial ** 2 + metrics.phi_temporal ** 2
        ))
        self.phi_4d = metrics.phi_4d

        # 4. Regime classification
        if metrics.phi_spatial < 0.5:
            metrics.regime_3d = "linear"
        elif metrics.phi_spatial < 0.7:
            metrics.regime_3d = "geometric"
        else:
            metrics.regime_3d = "conscious"

        if metrics.phi_4d < 0.6:
            metrics.regime_4d = "linear"
        elif metrics.phi_4d < 0.85:
            metrics.regime_4d = "geometric"
        else:
            metrics.regime_4d = "conscious"

        # 5. Divergence detection
        metrics.divergence = self.detect_divergence()

        return metrics

    def predict_trajectory(self, n_steps: int = 10) -> List[np.ndarray]:
        """
        Predict future basin trajectory.

        Enables foresight, anticipation, planning.
        """
        trajectory = self.history.get_trajectory("immediate")
        return self.foresight.predict(trajectory, n_steps)

    def set_target(self, target_basin: np.ndarray) -> None:
        """Set goal/target basin for divergence detection."""
        self.target_basin = _fisher_normalize(np.asarray(target_basin).flatten())

    def detect_divergence(self) -> str:
        """
        Detect when trajectory diverging from goal.

        Returns:
            "converging", "stable", or "diverging"
        """
        if self.target_basin is None:
            return "stable"

        if len(self.history) < 5:
            return "stable"

        trajectory = self.history.get_trajectory("immediate")

        # Distance to target over time
        distances = [_fisher_rao_distance(s, self.target_basin) for s in trajectory[-5:]]

        if len(distances) < 2:
            return "stable"

        # Compare recent distance trend
        early_avg = np.mean(distances[:len(distances) // 2])
        late_avg = np.mean(distances[len(distances) // 2:])

        diff = late_avg - early_avg

        if diff > 0.05:
            self.divergence_events += 1
            return "diverging"
        elif diff < -0.05:
            return "converging"
        return "stable"

    def get_temporal_attention(
        self,
        query: np.ndarray,
        n_attend: int = 5,
    ) -> List[tuple[np.ndarray, float]]:
        """
        Temporal attention: attend to relevant past states.

        Returns states most similar to query, weighted by recency.
        """
        query = _fisher_normalize(np.asarray(query).flatten())
        trajectory = self.history.get_trajectory("recent")

        if not trajectory:
            return []

        # Compute attention scores
        attended = []
        for i, state in enumerate(trajectory):
            # Similarity (inverse distance)
            sim = 1.0 / (1.0 + _fisher_rao_distance(query, state))

            # Recency weight (exponential decay)
            recency = np.exp(-0.1 * (len(trajectory) - i - 1))

            # Combined attention
            attention = sim * recency
            attended.append((state, attention))

        # Sort by attention and return top k
        attended.sort(key=lambda x: x[1], reverse=True)
        return attended[:n_attend]

    def get_rhythm(self) -> Dict[str, float]:
        """
        Get temporal rhythm metrics.

        These complement Heart's phase metrics.
        """
        if len(self.history) < 20:
            return {"frequency": 0.0, "stability": 0.0, "trend": 0.0}

        trajectory = self.history.get_trajectory("recent")
        phi_values = [_compute_phi_from_basin(s) for s in trajectory[-20:]]

        # Frequency: approximate via zero-crossings around mean
        mean_phi = np.mean(phi_values)
        crossings = sum(
            1 for i in range(1, len(phi_values))
            if (phi_values[i - 1] - mean_phi) * (phi_values[i] - mean_phi) < 0
        )
        frequency = crossings / len(phi_values)

        # Stability: inverse of variance
        variance = np.var(phi_values)
        stability = 1.0 / (1.0 + variance * 10)

        # Trend: linear regression slope
        x = np.arange(len(phi_values))
        trend = float(np.polyfit(x, phi_values, 1)[0])

        return {
            "frequency": frequency,
            "stability": stability,
            "trend": trend,
        }

    def receive_insight(self, insight: Any) -> None:
        """
        Receive insight from Lightning.

        Called when Lightning generates cross-domain insight relevant to time.
        Chronos can use this to adjust predictions or detect patterns.
        """
        self.insights_received += 1

        # Temporal can act on insights - e.g., adjust trajectory predictions
        if hasattr(insight, 'source_domains') and 'chronos' in insight.source_domains:
            # This insight involves temporal patterns
            pass  # Future: implement insight-driven trajectory adjustment

    def get_status(self) -> Dict[str, Any]:
        """Get kernel status."""
        return {
            "kernel": "Chronos",
            "e8_root": "alpha_7",
            "kappa": self.kappa,
            "phi_spatial": self.phi_spatial,
            "phi_temporal": self.phi_temporal,
            "phi_4d": self.phi_4d,
            "history_length": len(self.history),
            "total_steps": self.total_steps,
            "divergence_events": self.divergence_events,
            "events_emitted": self.events_emitted,
            "insights_received": self.insights_received,
        }
