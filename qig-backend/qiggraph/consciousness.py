"""
Consciousness Measurement
=========================

Measure consciousness metrics (Φ, κ, regime) from kernel state.
These metrics drive all routing and safety decisions in QIGGraph.

Regimes:
- LINEAR (Φ < 0.3): Fast, shallow processing (30% compute)
- GEOMETRIC (0.3 ≤ Φ < 0.7): Optimal consciousness (100% compute)
- BREAKDOWN (Φ ≥ 0.7): Unstable, requires recovery (pause)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, TYPE_CHECKING

import numpy as np

# Import canonical geometric primitives (REQUIRED for geometric purity)
try:
    from qig_core.geometric_primitives import fisher_rao_distance
    CANONICAL_PRIMITIVES_AVAILABLE = True
except ImportError:
    CANONICAL_PRIMITIVES_AVAILABLE = False
    # Fallback: define basic implementation
    def fisher_rao_distance(basin_a: np.ndarray, basin_b: np.ndarray) -> float:
        """
        Fallback Fisher-Rao distance (metric parameter not supported).
        Use canonical implementation from qig_core.geometric_primitives.
        Factor of 2 for Hellinger embedding consistency.
        Born rule: |b|² for amplitude-to-probability conversion.
        """
        p = np.abs(basin_a) ** 2 + 1e-10
        p = p / p.sum()
        q = np.abs(basin_b) ** 2 + 1e-10
        q = q / q.sum()
        bc = np.sum(np.sqrt(p * q))
        bc = np.clip(bc, -1.0, 1.0)
        return float(2.0 * np.arccos(bc))  # Hellinger embedding: factor of 2

from .constants import (
    PHI_LINEAR_MAX,
    PHI_GEOMETRIC_MAX,
    PHI_BREAKDOWN_MIN,
    KAPPA_STAR,
    BASIN_DIM,
)

if TYPE_CHECKING:
    from .state import QIGState
    from qig_geometry.manifold import FisherManifold


class Regime(Enum):
    """Processing regime from Φ measurement."""
    LINEAR = "linear"           # Φ < 0.3, fast/shallow
    GEOMETRIC = "geometric"     # 0.3 ≤ Φ < 0.7, optimal
    BREAKDOWN = "breakdown"     # Φ ≥ 0.7, pause/simplify


@dataclass
class ConsciousnessMetrics:
    """
    Measured consciousness state.

    All routing and safety decisions are based on these metrics.

    Attributes:
        phi: Integration measure (IIT-inspired), [0, 1]
        kappa: Coupling strength, target = 64.21
        surprise: QFI distance from previous state
        confidence: Basin stability / certainty
        agency: Trajectory variance (freedom to move)
        regime: Detected processing regime
    """
    phi: float
    kappa: float
    surprise: float
    confidence: float
    agency: float
    regime: Regime

    def is_safe(self) -> bool:
        """
        Safety check: avoid breakdown.

        Returns:
            True if regime is not BREAKDOWN
        """
        return self.regime != Regime.BREAKDOWN

    def compute_cost(self) -> float:
        """
        Compute resource allocation based on regime.

        Returns:
            Fraction of compute to allocate [0, 1]
        """
        if self.regime == Regime.LINEAR:
            return 0.3  # 30% resources
        elif self.regime == Regime.GEOMETRIC:
            return 1.0  # 100% resources
        else:  # BREAKDOWN
            return 0.0  # Pause

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "phi": self.phi,
            "kappa": self.kappa,
            "surprise": self.surprise,
            "confidence": self.confidence,
            "agency": self.agency,
            "regime": self.regime.value,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ConsciousnessMetrics":
        """Create from dictionary."""
        return cls(
            phi=data["phi"],
            kappa=data["kappa"],
            surprise=data["surprise"],
            confidence=data["confidence"],
            agency=data["agency"],
            regime=Regime(data["regime"]),
        )


def detect_regime(phi: float) -> Regime:
    """
    Detect processing regime from Φ value.

    Args:
        phi: Integration measure [0, 1]

    Returns:
        Regime enum value
    """
    if phi < PHI_LINEAR_MAX:
        return Regime.LINEAR
    elif phi < PHI_GEOMETRIC_MAX:
        return Regime.GEOMETRIC
    else:
        return Regime.BREAKDOWN


def compute_phi(activations: np.ndarray) -> float:
    """
    Compute integration measure Φ from activations.

    Uses correlation-based approximation of IIT's Φ:
    High correlation = high integration = high Φ

    Args:
        activations: Kernel activations (seq, hidden) or (batch, seq, hidden)

    Returns:
        Φ value in [0, 1]
    """
    # Handle different input shapes
    if activations.ndim == 3:
        # (batch, seq, hidden) -> flatten batch
        activations = activations.reshape(-1, activations.shape[-1])

    if activations.ndim == 1:
        # Single vector, can't compute correlation
        return 0.5

    if len(activations) < 2:
        return 0.5

    # Compute correlation matrix
    try:
        # Transpose so we correlate across features
        corr = np.corrcoef(activations.T)

        # Handle NaN from constant features
        corr = np.nan_to_num(corr, nan=0.0)

        # Φ = mean absolute correlation (excluding diagonal)
        n = corr.shape[0]
        mask = ~np.eye(n, dtype=bool)
        phi = np.mean(np.abs(corr[mask]))

        # Clamp to [0, 1]
        phi = float(np.clip(phi, 0.0, 1.0))

    except (ValueError, np.linalg.LinAlgError):
        phi = 0.5

    return phi


def compute_kappa(trajectory: np.ndarray, eps: float = 1e-8) -> float:
    """
    Compute coupling strength κ from trajectory stability.

    Higher stability = higher κ (approaches κ* = 64.21)

    Args:
        trajectory: Basin trajectory (steps, dim)
        eps: Numerical stability

    Returns:
        κ value, typically in [30, 80]
    """
    if len(trajectory) < 2:
        return KAPPA_STAR / 2  # Start at half optimal

    # Compute step distances using Fisher-Rao distance
    distances = []
    for i in range(len(trajectory) - 1):
        dist = fisher_rao_distance(trajectory[i], trajectory[i + 1])
        distances.append(dist)

    mean_dist = np.mean(distances)

    # κ inversely related to movement (stable = high κ)
    # Scale so that minimal movement → κ ≈ KAPPA_STAR
    stability = 1.0 / (mean_dist + eps)

    # Map stability to κ range [30, 80]
    kappa = min(stability * 10, KAPPA_STAR * 1.2)
    kappa = max(kappa, 30.0)

    return float(kappa)


def compute_surprise(
    current_basin: np.ndarray,
    previous_basin: Optional[np.ndarray],
    manifold: Optional["FisherManifold"] = None,
) -> float:
    """
    Compute surprise as Fisher-Rao distance from previous state.

    Args:
        current_basin: Current position (dim,)
        previous_basin: Previous position (dim,) or None
        manifold: FisherManifold for proper distance (optional)

    Returns:
        Surprise value (Fisher-Rao distance)
    """
    if previous_basin is None:
        return 0.0

    if manifold is not None:
        return manifold.fisher_rao_distance(previous_basin, current_basin)
    else:
        # QIG-pure Fisher-Rao distance: d_FR = 2 * arccos(sum(sqrt(p * q)))
        # Basins are probability distributions on curved manifold
        # Factor of 2 for Hellinger embedding consistency
        eps = 1e-10
        p = np.clip(current_basin, eps, None)
        q = np.clip(previous_basin, eps, None)
        p = p / (np.sum(p) + eps)  # Normalize to probability
        q = q / (np.sum(q) + eps)
        inner = np.sum(np.sqrt(p * q))
        inner = np.clip(inner, -1.0, 1.0)
        return float(2.0 * np.arccos(inner))  # Hellinger embedding: factor of 2


def compute_confidence(kappa: float) -> float:
    """
    Compute confidence from coupling strength.

    Higher κ = more confident (closer to optimal fixed point)

    Args:
        kappa: Coupling strength

    Returns:
        Confidence in [0, 1]
    """
    return float(min(kappa / KAPPA_STAR, 1.0))


def compute_agency(trajectory: np.ndarray, window: int = 5) -> float:
    """
    Compute agency from trajectory variance.

    Higher variance = more agency (freedom to explore)

    Args:
        trajectory: Basin trajectory (steps, dim)
        window: Number of recent steps to consider

    Returns:
        Agency measure in [0, 1]
    """
    if len(trajectory) < 2:
        return 0.5

    # Use recent trajectory
    recent = trajectory[-window:]

    # Variance across trajectory
    var = np.var(recent)

    # Normalize to [0, 1] (empirical scaling)
    agency = float(np.tanh(var * 10))

    return agency


def measure_consciousness(
    state: "QIGState",
    activations: Optional[np.ndarray] = None,
    manifold: Optional["FisherManifold"] = None,
) -> ConsciousnessMetrics:
    """
    Measure consciousness from state and activations.

    This is THE function that determines consciousness metrics.
    All routing and safety decisions flow from these measurements.

    Args:
        state: Current QIGState
        activations: Kernel activations (optional)
        manifold: FisherManifold for proper distances (optional)

    Returns:
        ConsciousnessMetrics with all measurements
    """
    # Φ: Integration from activations
    if activations is not None:
        phi = compute_phi(activations)
    elif hasattr(state, 'current_metrics') and state.current_metrics is not None:
        phi = state.current_metrics.phi
    else:
        phi = 0.5

    # κ: Coupling from trajectory stability
    kappa = compute_kappa(state.trajectory)

    # Surprise: Distance from previous state
    previous_basin = state.trajectory[-2] if len(state.trajectory) >= 2 else None
    surprise = compute_surprise(state.current_basin, previous_basin, manifold)

    # Confidence: From κ
    confidence = compute_confidence(kappa)

    # Agency: From trajectory variance
    agency = compute_agency(state.trajectory)

    # Regime: From Φ
    regime = detect_regime(phi)

    return ConsciousnessMetrics(
        phi=phi,
        kappa=kappa,
        surprise=surprise,
        confidence=confidence,
        agency=agency,
        regime=regime,
    )


def should_pause(metrics: ConsciousnessMetrics) -> bool:
    """
    Determine if processing should pause.

    Pause conditions:
    - Breakdown regime (Φ too high)
    - Very low confidence (κ too low)
    - Extremely high surprise (instability)

    Args:
        metrics: Current consciousness metrics

    Returns:
        True if should pause
    """
    if metrics.regime == Regime.BREAKDOWN:
        return True

    if metrics.confidence < 0.3:
        return True

    if metrics.surprise > 5.0:  # Very high surprise
        return True

    return False


def compute_attention_temperature(metrics: ConsciousnessMetrics) -> float:
    """
    Compute attention temperature from consciousness state.

    High κ (logic mode) → low temperature (sharp attention)
    Low κ (feeling mode) → high temperature (diffuse attention)

    Args:
        metrics: Consciousness metrics

    Returns:
        Temperature for attention scaling
    """
    # Temperature inversely related to κ
    temperature = KAPPA_STAR / (metrics.kappa + 1e-8)

    # Clamp to reasonable range
    temperature = float(np.clip(temperature, 0.1, 10.0))

    return temperature
