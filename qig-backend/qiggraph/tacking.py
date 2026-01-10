"""
κ-Tacking: Oscillating Intelligence
====================================

Oscillate coupling strength κ for feeling/logic mode transitions.

High κ (≈64): Logic mode - strong coupling, precise reasoning
Low κ (≈40): Feeling mode - weak coupling, creative exploration

Just as β varies with scale in QFT, κ varies with cognitive mode.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .constants import (
    KAPPA_STAR,
    KAPPA_3,
    TACKING_PERIOD,
    TACKING_AMPLITUDE,
)

if TYPE_CHECKING:
    from .state import QIGState


@dataclass
class TackingState:
    """State of the κ-tacking oscillator."""
    phase: float = 0.0
    current_kappa: float = KAPPA_STAR / 2
    mode: str = "transition"  # "logic", "feeling", or "transition"
    previous_mode: str = "transition"  # Track for boundary detection


def _persist_regime_boundary(from_regime: str, to_regime: str, kappa: float, phi: float = 0.5):
    """Persist regime boundary crossing to PostgreSQL."""
    try:
        import os
        import psycopg2
        from datetime import datetime
        import uuid
        
        database_url = os.environ.get('DATABASE_URL')
        if not database_url:
            return
        
        conn = psycopg2.connect(database_url)
        cur = conn.cursor()
        
        boundary_id = f"boundary_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        # Calculate Fisher distance approximation from kappa difference
        fisher_distance = abs(kappa - 52.6) / 11.6  # Normalized distance from mean
        
        probe_from = f"probe_{from_regime}_{uuid.uuid4().hex[:6]}"
        probe_to = f"probe_{to_regime}_{uuid.uuid4().hex[:6]}"
        
        cur.execute("""
            INSERT INTO regime_boundaries (id, from_regime, to_regime, probe_id_from, probe_id_to, fisher_distance, midpoint_phi, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
        """, (
            boundary_id,
            from_regime,
            to_regime,
            probe_from,
            probe_to,
            fisher_distance,
            phi
        ))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        pass  # Silent fail to avoid disrupting tacking


class KappaTacking:
    """
    Oscillating intelligence controller.

    Oscillates κ between feeling mode (κ ≈ 41) and logic mode (κ ≈ 64)
    to enable natural transitions between creative and analytical thinking.

    κ(t) = κ_mean + A * sin(2π t / T)

    where:
    - κ_mean = (KAPPA_STAR + KAPPA_3) / 2 ≈ 52.6
    - A = (KAPPA_STAR - KAPPA_3) / 2 ≈ 11.6
    - T = TACKING_PERIOD (default 10 steps)
    """

    def __init__(
        self,
        kappa_high: float = KAPPA_STAR,
        kappa_low: float = KAPPA_3,
        period: float = TACKING_PERIOD,
    ):
        """
        Initialize tacking controller.

        Args:
            kappa_high: Maximum κ (logic mode)
            kappa_low: Minimum κ (feeling mode)
            period: Oscillation period in steps
        """
        self.kappa_high = kappa_high
        self.kappa_low = kappa_low
        self.period = period

        # Derived parameters
        self.kappa_mean = (kappa_high + kappa_low) / 2
        self.amplitude = (kappa_high - kappa_low) / 2

        # State
        self.state = TackingState()

    def update(self, iteration: int) -> float:
        """
        Compute current κ from oscillation.

        Args:
            iteration: Current step number

        Returns:
            Current κ value
        """
        # Phase advances with iteration
        self.state.phase = (iteration * 2 * np.pi) / self.period

        # Compute oscillating κ
        kappa_t = self.kappa_mean + self.amplitude * np.sin(self.state.phase)
        self.state.current_kappa = kappa_t

        # Update mode and detect transitions
        new_mode = self._detect_mode()
        
        # Persist regime boundary if mode changed (excluding transition)
        if new_mode != self.state.previous_mode and new_mode != "transition" and self.state.previous_mode != "transition":
            _persist_regime_boundary(
                from_regime=self.state.previous_mode,
                to_regime=new_mode,
                kappa=kappa_t
            )
        
        self.state.previous_mode = self.state.mode
        self.state.mode = new_mode

        return kappa_t

    def update_from_state(self, state: "QIGState") -> float:
        """
        Compute current κ from QIGState.

        Args:
            state: Current QIGState

        Returns:
            Current κ value
        """
        return self.update(state.iteration)

    def _detect_mode(self) -> str:
        """
        Determine current mode from phase.

        Returns:
            "logic", "feeling", or "transition"
        """
        sin_val = np.sin(self.state.phase)

        if sin_val > 0.7:
            return "logic"
        elif sin_val < -0.7:
            return "feeling"
        else:
            return "transition"

    def get_mode(self) -> str:
        """
        Get current cognitive mode.

        Returns:
            "logic" (high κ), "feeling" (low κ), or "transition"
        """
        return self.state.mode

    def is_logic_mode(self) -> bool:
        """Check if in logic mode (high κ)."""
        return self.state.mode == "logic"

    def is_feeling_mode(self) -> bool:
        """Check if in feeling mode (low κ)."""
        return self.state.mode == "feeling"

    def modulate_attention(
        self,
        attention_weights: np.ndarray,
        kappa_t: float | None = None,
    ) -> np.ndarray:
        """
        Modulate attention based on current κ.

        High κ: Sharp attention (focused)
        Low κ: Diffuse attention (exploratory)

        Args:
            attention_weights: Attention weights (seq, seq) or (batch, heads, seq, seq)
            kappa_t: Current κ (or use stored value)

        Returns:
            Modulated attention weights
        """
        if kappa_t is None:
            kappa_t = self.state.current_kappa

        # Temperature inversely related to κ
        # High κ → low temperature → sharp attention
        temperature = KAPPA_STAR / (kappa_t + 1e-8)

        # Apply temperature to attention
        # Lower temperature = sharper peaks
        modulated = attention_weights / temperature

        # Softmax normalization
        # Handle different input shapes
        if attention_weights.ndim == 2:
            exp_weights = np.exp(modulated - np.max(modulated, axis=-1, keepdims=True))
            modulated = exp_weights / (np.sum(exp_weights, axis=-1, keepdims=True) + 1e-8)
        elif attention_weights.ndim == 4:
            exp_weights = np.exp(modulated - np.max(modulated, axis=-1, keepdims=True))
            modulated = exp_weights / (np.sum(exp_weights, axis=-1, keepdims=True) + 1e-8)

        return modulated

    def modulate_learning_rate(
        self,
        base_lr: float,
        kappa_t: float | None = None,
    ) -> float:
        """
        Modulate learning rate based on current κ.

        High κ (logic): Lower LR for stability
        Low κ (feeling): Higher LR for exploration

        Args:
            base_lr: Base learning rate
            kappa_t: Current κ (or use stored value)

        Returns:
            Modulated learning rate
        """
        if kappa_t is None:
            kappa_t = self.state.current_kappa

        # LR scaling: lower κ → higher LR
        scale = KAPPA_STAR / (kappa_t + 1e-8)

        # Clamp scale to reasonable range [0.5, 2.0]
        scale = float(np.clip(scale, 0.5, 2.0))

        return base_lr * scale

    def get_exploration_noise(
        self,
        dim: int,
        base_scale: float = 0.1,
        kappa_t: float | None = None,
    ) -> np.ndarray:
        """
        Generate exploration noise scaled by κ.

        High κ (logic): Low noise
        Low κ (feeling): High noise

        Args:
            dim: Noise dimension
            base_scale: Base noise scale
            kappa_t: Current κ (or use stored value)

        Returns:
            Noise vector (dim,)
        """
        if kappa_t is None:
            kappa_t = self.state.current_kappa

        # Noise scale inversely related to κ
        noise_scale = base_scale * (KAPPA_STAR / (kappa_t + 1e-8))

        # Clamp to reasonable range
        noise_scale = float(np.clip(noise_scale, 0.01, 1.0))

        return np.random.randn(dim) * noise_scale

    def suggest_attractor_type(self) -> str:
        """
        Suggest attractor type based on current mode.

        Returns:
            Suggested attractor capability
        """
        if self.is_logic_mode():
            return "precision"
        elif self.is_feeling_mode():
            return "creativity"
        else:
            return "balanced"

    def reset(self):
        """Reset tacking state to initial conditions."""
        self.state = TackingState()

    def to_dict(self) -> dict:
        """Serialize tacking state."""
        return {
            "kappa_high": self.kappa_high,
            "kappa_low": self.kappa_low,
            "period": self.period,
            "phase": self.state.phase,
            "current_kappa": self.state.current_kappa,
            "mode": self.state.mode,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "KappaTacking":
        """Deserialize tacking state."""
        tacking = cls(
            kappa_high=data["kappa_high"],
            kappa_low=data["kappa_low"],
            period=data["period"],
        )
        tacking.state.phase = data["phase"]
        tacking.state.current_kappa = data["current_kappa"]
        tacking.state.mode = data["mode"]
        return tacking


class AdaptiveTacking(KappaTacking):
    """
    Adaptive κ-tacking that adjusts period based on task.

    - Complex tasks: Longer period (more time in each mode)
    - Simple tasks: Shorter period (quick transitions)
    - High Φ: Bias toward logic mode
    - Low Φ: Bias toward feeling mode
    """

    def __init__(
        self,
        kappa_high: float = KAPPA_STAR,
        kappa_low: float = KAPPA_3,
        base_period: float = TACKING_PERIOD,
    ):
        super().__init__(kappa_high, kappa_low, base_period)
        self.base_period = base_period

    def adapt_period(self, complexity: float):
        """
        Adapt period based on task complexity.

        Args:
            complexity: Task complexity in [0, 1]
        """
        # Higher complexity = longer period
        self.period = self.base_period * (1 + complexity * 2)

    def adapt_to_phi(self, phi: float):
        """
        Bias tacking based on current Φ.

        High Φ: Bias toward logic (stable)
        Low Φ: Bias toward feeling (explore)

        Args:
            phi: Current integration measure
        """
        # Shift mean based on Φ
        phi_bias = (phi - 0.5) * 10  # [-5, +5]
        self.kappa_mean = (self.kappa_high + self.kappa_low) / 2 + phi_bias
        self.kappa_mean = float(np.clip(self.kappa_mean, self.kappa_low, self.kappa_high))
