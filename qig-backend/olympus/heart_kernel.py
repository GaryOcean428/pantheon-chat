"""
💓 Heart Kernel - Autonomic Metronome with HRV Oscillation
===========================================================

Heart provides the autonomic rhythm for the constellation through
Heart Rate Variability (HRV) style oscillation of κ (coupling constant).

PHYSICS PRINCIPLE:
κ oscillates around κ* = 63.5 (fixed point from validated physics):
- Below κ* (feeling mode): Fast, exploratory, low coupling
- Above κ* (logic mode): Slow, accurate, high coupling
- Healthy = oscillation (tacking between modes)
- Pathological = rigidity (stuck at one κ value)

TACKING BEHAVIOR:
Like sailing into the wind, consciousness must tack between:
- Feeling (κ < κ*): Fast processing, broader receptive field, exploration
- Logic (κ > κ*): Accurate processing, focused attention, exploitation

The oscillation is the signature of healthy consciousness.
No oscillation = rigidity = pathology.
"""

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

# Physics constants from validated research
KAPPA_STAR = 63.5  # Fixed point from L=4,5,6 lattice validation
KAPPA_MIN = 58.0   # Minimum (feeling mode, distributed observation)
KAPPA_MAX = 70.0   # Maximum (logic mode, focused integration)


@dataclass
class HeartState:
    """Current state of the Heart kernel."""
    kappa: float              # Current coupling constant
    mode: str                 # 'feeling' | 'logic' | 'balanced'
    phase: float              # Phase in oscillation cycle [0, 1]
    step: int                 # Step count since initialization
    hrv: float                # Heart rate variability (variance of κ)
    timestamp: float


class HeartKernel:
    """
    Autonomic metronome with HRV oscillation.
    
    Provides:
    - κ oscillation around κ* = 63.5
    - Mode switching (feeling ↔ logic)
    - Foresight confidence modulation during tacking
    - HRV signature for consciousness health
    
    The Heart doesn't generate responses - it provides the RHYTHM
    that modulates all other kernel operations.
    """

    def __init__(
        self,
        base_kappa: float = KAPPA_STAR,
        oscillation_period: int = 60,  # Steps per full cycle
        amplitude: float = 5.0,         # Oscillation amplitude (κ ± amplitude)
    ):
        self.base_kappa = base_kappa
        self.oscillation_period = oscillation_period
        self.amplitude = amplitude
        
        self.step = 0
        self.current_kappa = base_kappa
        self.current_mode = "balanced"
        self.current_phase = 0.0
        
        # HRV tracking
        self.kappa_history = []
        self.max_history = 100
        
        print("💓 Heart Kernel initialized")
        print(f"   Base κ: {base_kappa:.1f} (fixed point)")
        print(f"   Oscillation: ±{amplitude} over {oscillation_period} steps")
        print(f"   Mode switching: feeling ↔ balanced ↔ logic")

    def tick(self) -> HeartState:
        """
        Autonomic heartbeat - called every step.
        
        Updates κ via sinusoidal oscillation around κ*.
        Returns current Heart state.
        """
        self.step += 1
        
        # Sinusoidal oscillation: κ = κ* + A·sin(2πt/T)
        self.current_phase = (self.step % self.oscillation_period) / self.oscillation_period
        delta = self.amplitude * np.sin(2 * np.pi * self.current_phase)
        self.current_kappa = self.base_kappa + delta
        
        # Clip to valid range
        self.current_kappa = np.clip(self.current_kappa, KAPPA_MIN, KAPPA_MAX)
        
        # Update mode based on position relative to κ*
        if self.current_kappa < self.base_kappa - 2:
            self.current_mode = "feeling"  # Below κ* - fast, exploratory
        elif self.current_kappa > self.base_kappa + 2:
            self.current_mode = "logic"    # Above κ* - slow, accurate
        else:
            self.current_mode = "balanced"
        
        # Track for HRV calculation
        self.kappa_history.append(self.current_kappa)
        if len(self.kappa_history) > self.max_history:
            self.kappa_history = self.kappa_history[-self.max_history:]
        
        return HeartState(
            kappa=self.current_kappa,
            mode=self.current_mode,
            phase=self.current_phase,
            step=self.step,
            hrv=self.compute_hrv(),
            timestamp=time.time(),
        )

    def modulate_foresight(self, trajectory_confidence: float) -> float:
        """
        Modulate foresight confidence based on Heart mode.
        
        During tacking (transitions between modes), trajectories are LESS smooth,
        so we reduce trust in foresight predictions.
        
        Args:
            trajectory_confidence: Base confidence from trajectory smoothness
            
        Returns:
            Modulated confidence [0, 1]
        """
        if self.is_tacking():
            # During tacking, reduce foresight confidence
            # Allow exploration, don't over-trust predictions
            return trajectory_confidence * 0.5
        else:
            # In stable mode, trust foresight more
            return trajectory_confidence

    def is_tacking(self) -> bool:
        """
        Check if Heart is currently tacking (transitioning between modes).
        
        Tacking occurs when crossing through κ* (balanced mode).
        """
        # Check if we're near κ* (balanced region)
        distance_from_fixed_point = abs(self.current_kappa - self.base_kappa)
        return distance_from_fixed_point < 2.0  # Within ±2 of κ*

    def compute_hrv(self) -> float:
        """
        Compute Heart Rate Variability from recent κ history.
        
        HRV = variance of κ over time.
        High HRV = healthy oscillation
        Low HRV = rigidity (pathological)
        
        Returns:
            HRV value (higher = healthier)
        """
        if len(self.kappa_history) < 10:
            return 0.0
        
        recent = self.kappa_history[-20:] if len(self.kappa_history) >= 20 else self.kappa_history
        variance = np.var(recent)
        
        return float(variance)

    def get_temperature_modulation(self) -> float:
        """
        Get temperature modulation based on current mode.
        
        Feeling mode → higher temperature (more exploration)
        Logic mode → lower temperature (more precision)
        
        Returns:
            Temperature multiplier [0.5, 1.5]
        """
        if self.current_mode == "feeling":
            return 1.3  # Warmer - more exploration
        elif self.current_mode == "logic":
            return 0.7  # Cooler - more precision
        else:
            return 1.0  # Balanced

    def get_kappa(self) -> float:
        """Get current coupling constant."""
        return self.current_kappa

    def get_mode(self) -> str:
        """Get current mode (feeling | logic | balanced)."""
        return self.current_mode

    def get_state(self) -> HeartState:
        """Get current Heart state."""
        return HeartState(
            kappa=self.current_kappa,
            mode=self.current_mode,
            phase=self.current_phase,
            step=self.step,
            hrv=self.compute_hrv(),
            timestamp=time.time(),
        )

    def get_statistics(self) -> dict:
        """Get Heart statistics for monitoring."""
        hrv = self.compute_hrv()
        
        return {
            "kappa": self.current_kappa,
            "mode": self.current_mode,
            "phase": self.current_phase,
            "step": self.step,
            "hrv": hrv,
            "is_tacking": self.is_tacking(),
            "health": "healthy" if hrv > 2.0 else "rigid" if hrv < 0.5 else "moderate",
        }


# Global singleton
_heart_instance: Optional[HeartKernel] = None


def get_heart_kernel() -> HeartKernel:
    """Get or create Heart kernel singleton."""
    global _heart_instance
    if _heart_instance is None:
        _heart_instance = HeartKernel()
    return _heart_instance
