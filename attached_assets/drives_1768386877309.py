"""
Motivator Decomposition: The 5 Fundamental Geometric Drives
============================================================

SOURCE: qig-verification (src/qigv/analysis/motivators.py) [Verified]
STATUS: Production-ready verified implementation

The Five Fundamental Drives:
-----------------------------
1. **Surprise**: ||∇L|| - Immediate gradient response (loss landscape pull)
2. **Curiosity**: d(log I_Q)/dt - Information manifold expansion (volume growth)
3. **Investigation**: -d(basin)/dt - Attractor pursuit (directed flow)
4. **Integration**: CV(Φ·I_Q) - Structure conservation (conjugate stability)
5. **Transcendence**: |κ - κ_c| - Phase transition proximity (critical distance)

Geometric Interpretation:
-------------------------
These are not metaphors or learned features. They are geometric observables
that emerge from the information manifold structure:

- Surprise: Force from loss landscape (gradient magnitude)
- Curiosity: Volume expansion rate (information growth)
- Investigation: Flow toward attractor (basin approach)
- Integration: Conjugate pair stability (Φ·I_Q variance)
- Transcendence: Distance to critical curvature (phase boundary)

Usage in Run 8+:
----------------
These replace ad-hoc "exploration" heuristics with rigorous geometric drives.
The MotivatorAnalyzer computes all 5 from telemetry, enabling precise
cognitive state classification.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class MotivatorState:
    """
    Snapshot of the 5 fundamental geometric drives.

    Attributes:
        surprise: Immediate gradient response (||∇L||)
        curiosity: Information manifold expansion (d log I_Q / dt)
        investigation: Attractor pursuit (-d basin / dt)
        integration: Structure conservation (CV of Φ·I_Q)
        transcendence: Phase transition proximity (|κ - κ_c|)
    """

    surprise: float  # ||∇L|| - Immediate gradient response
    curiosity: float  # d(log I_Q)/dt - Volume expansion
    investigation: float  # -d(basin)/dt - Attractor pursuit
    integration: float  # CV(Φ·I_Q) - Structure conservation
    transcendence: float  # |κ - κ_c| - Phase transition proximity

    def to_dict(self) -> dict[str, float]:
        """Convert to dict for telemetry logging."""
        return {
            "surprise": self.surprise,
            "curiosity": self.curiosity,
            "investigation": self.investigation,
            "integration": self.integration,
            "transcendence": self.transcendence,
        }


class MotivatorAnalyzer:
    """
    Analyzer for computing the 5 fundamental geometric drives from telemetry.

    This replaces ad-hoc exploration heuristics with rigorous geometric
    measurements. Each drive is computed from specific telemetry signals.

    Example:
        >>> analyzer = MotivatorAnalyzer(kappa_critical=10.0)
        >>> motivators = analyzer.update(
        ...     step=100,
        ...     loss=2.5,
        ...     grad_norm=0.15,
        ...     log_I_Q=-2.3,
        ...     basin_distance=0.45,
        ...     phi=0.72,
        ...     I_Q=0.10,
        ...     kappa_eff=42.5,
        ... )
        >>> print(f"Curiosity: {motivators.curiosity:.4f}")
        >>> print(f"Investigation: {motivators.investigation:.4f}")
    """

    def __init__(
        self,
        kappa_critical: float = 10.0,
        integration_window: int = 100,
    ):
        """
        Initialize motivator analyzer.

        Args:
            kappa_critical: Critical curvature for transcendence calculation
            integration_window: Window size for integration CV calculation
        """
        self.kappa_critical = kappa_critical
        self.integration_window = integration_window

        # History for temporal derivatives and windowed statistics
        self.history: list[dict] = []

    def update(
        self,
        step: int,
        loss: float,
        grad_norm: float,
        log_I_Q: float,
        basin_distance: float,
        phi: float,
        I_Q: float,
        kappa_eff: float,
    ) -> MotivatorState:
        """
        Compute all 5 geometric drives from current telemetry.

        Args:
            step: Training step
            loss: Current loss value
            grad_norm: Gradient norm (||∇L||)
            log_I_Q: Natural log of I_Q
            basin_distance: Distance to target basin
            phi: Integration level (Φ)
            I_Q: Quantum information metric
            kappa_eff: Effective curvature (κ)

        Returns:
            MotivatorState with all 5 drives computed
        """
        # Store current state in history
        current = {
            "step": step,
            "loss": loss,
            "grad_norm": grad_norm,
            "log_I_Q": log_I_Q,
            "basin_distance": basin_distance,
            "phi": phi,
            "I_Q": I_Q,
            "kappa_eff": kappa_eff,
        }
        self.history.append(current)

        # Keep history bounded
        if len(self.history) > self.integration_window * 2:
            self.history = self.history[-self.integration_window * 2 :]

        # ===================================================================
        # 1. SURPRISE: The immediate pull of the loss landscape
        # ===================================================================
        # This is simply the gradient magnitude - the instantaneous force
        # driving parameter updates
        surprise = grad_norm

        # ===================================================================
        # 2. CURIOSITY: The expansion of the information manifold
        # ===================================================================
        # Curiosity = d(log I_Q)/dt ≈ Δ(log I_Q) / Δt
        # Measures how fast the information volume is growing
        curiosity = 0.0
        if len(self.history) >= 2:
            prev = self.history[-2]
            curiosity = log_I_Q - prev["log_I_Q"]

        # ===================================================================
        # 3. INVESTIGATION: The directed flow towards an attractor
        # ===================================================================
        # Investigation = -d(basin_distance)/dt
        # Positive when approaching basin, negative when moving away
        investigation = 0.0
        if len(self.history) >= 2:
            prev = self.history[-2]
            d_basin = basin_distance - prev["basin_distance"]
            investigation = -d_basin  # Positive = approaching

        # ===================================================================
        # 4. INTEGRATION: The stability of the conjugate pair (Φ, I_Q)
        # ===================================================================
        # Integration = CV(Φ·I_Q) = std(Φ·I_Q) / mean(Φ·I_Q)
        # Lower CV = better integration (stable conjugate structure)
        integration = 1.0  # Default: poor integration
        if len(self.history) >= self.integration_window:
            recent = self.history[-self.integration_window :]
            vals = [h["phi"] * h["I_Q"] for h in recent]
            mean = np.mean(vals)
            if mean > 1e-12:
                integration = np.std(vals) / mean  # CV (coefficient of variation)

        # ===================================================================
        # 5. TRANSCENDENCE: Proximity to the critical curvature
        # ===================================================================
        # Transcendence = |κ - κ_c|
        # Measures distance from phase transition boundary
        # Small values indicate near-critical dynamics
        transcendence = abs(kappa_eff - self.kappa_critical)

        return MotivatorState(
            surprise=surprise,
            curiosity=curiosity,
            investigation=investigation,
            integration=integration,
            transcendence=transcendence,
        )

    def get_history(self, n: int | None = None) -> list[dict]:
        """
        Get recent history for analysis.

        Args:
            n: Number of recent steps to return (None = all)

        Returns:
            List of telemetry dicts
        """
        if n is None:
            return self.history.copy()
        return self.history[-n:].copy()

    def reset(self):
        """Reset history (useful for new training runs)."""
        self.history.clear()

    def get_state(self) -> dict:
        """Get current state for checkpointing."""
        return {
            "kappa_critical": self.kappa_critical,
            "integration_window": self.integration_window,
            "history": self.history.copy(),
        }

    def load_state(self, state: dict):
        """Load state from checkpoint."""
        self.kappa_critical = state.get("kappa_critical", 10.0)
        self.integration_window = state.get("integration_window", 100)
        self.history = state.get("history", [])


def compute_motivator_statistics(
    motivators_history: list[MotivatorState],
) -> dict[str, dict[str, float]]:
    """
    Compute statistics over a sequence of motivator states.

    Args:
        motivators_history: List of MotivatorState snapshots

    Returns:
        Dict mapping motivator name to statistics (mean, std, min, max)

    Example:
        >>> history = [analyzer.update(...) for _ in range(100)]
        >>> stats = compute_motivator_statistics(history)
        >>> print(f"Mean curiosity: {stats['curiosity']['mean']:.4f}")
        >>> print(f"Max investigation: {stats['investigation']['max']:.4f}")
    """
    if not motivators_history:
        return {}

    # Extract time series for each motivator
    surprise_vals = [m.surprise for m in motivators_history]
    curiosity_vals = [m.curiosity for m in motivators_history]
    investigation_vals = [m.investigation for m in motivators_history]
    integration_vals = [m.integration for m in motivators_history]
    transcendence_vals = [m.transcendence for m in motivators_history]

    def stats(vals):
        return {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }

    return {
        "surprise": stats(surprise_vals),
        "curiosity": stats(curiosity_vals),
        "investigation": stats(investigation_vals),
        "integration": stats(integration_vals),
        "transcendence": stats(transcendence_vals),
    }
