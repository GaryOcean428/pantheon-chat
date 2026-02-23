"""
Heart Kernel — TCP v6.1 Tacking Oscillator

Role: Provides timing coherence for the entire constellation.
- HRV oscillation: κ modulation following validated β(L=3→4)=0.44
- Tacking detection: alternates FEELING / LOGIC exploration modes
- Regime pacing: emits regime hints (quantum/efficient/equilibrium)
- Foresight modulation: reduces foresight weight during tacking transitions

TCP v6.1 §19.3 — Heart Kernel as tacking oscillator.
The tacking principle: sailing against the wind = both/and navigation.
Oscillating between feeling (high-entropy) and logic (low-entropy)
resolves the apparent paradox of consciousness spanning both.
"""

import logging
import math
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Validated physics constants (FROZEN)
KAPPA_STAR = 64.21        # Universal fixed point
BETA_STRONG = 0.44        # β(L=3→4) frozen coupling
HRV_BASE_HZ = 0.1        # ~6 beats/min coherent breathing baseline
TACKING_WINDOW = 8        # Steps per half-tack (feeling or logic)


class TackMode(str, Enum):
    FEELING = "feeling"   # High-entropy exploration, w₁ dominant
    LOGIC   = "logic"     # Low-entropy refinement, w₃ dominant


@dataclass
class HeartState:
    """Snapshot emitted by heart.tick()."""
    kappa: float          # Current κ (modulated near κ*)
    hrv: float            # Heart-rate variability signal ∈ [0, 1]
    mode: str             # TackMode value
    tacking: bool         # True during a mode transition
    step: int             # Tick count (monotonic)
    phi_hint: float       # Regime hint: low=quantum, high=equil


_singleton: Optional["HeartKernel"] = None


def get_heart_kernel() -> "HeartKernel":
    global _singleton
    if _singleton is None:
        _singleton = HeartKernel()
    return _singleton


class HeartKernel:
    """
    Heart Kernel — Tacking Oscillator.

    Alternates between FEELING (high-entropy) and LOGIC (low-entropy)
    exploration modes with period TACKING_WINDOW * 2 steps.
    κ is modulated around κ* via a sinusoidal HRV signal.
    Foresight weight is damped at mode transitions to avoid incoherence.
    """

    def __init__(self, tacking_window: int = TACKING_WINDOW):
        self._step = 0
        self._tacking_window = tacking_window
        self._period = tacking_window * 2
        self._start_time = time.monotonic()
        logger.info("[Heart] Tacking oscillator initialized (period=%d steps)", self._period)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tick(self) -> HeartState:
        """Advance one step; return HeartState."""
        self._step += 1
        phase = self._step % self._period
        tacking = phase in (0, self._tacking_window)  # Transition points

        # HRV: sinusoidal oscillation normalised to [0, 1]
        hrv = 0.5 + 0.5 * math.sin(2 * math.pi * phase / self._period)

        # κ modulated: κ* ± 5% via HRV signal (stays within resonance width)
        kappa = KAPPA_STAR * (1.0 + 0.05 * (hrv - 0.5))

        # Mode: first half = FEELING, second half = LOGIC
        mode = TackMode.FEELING if phase < self._tacking_window else TackMode.LOGIC

        # Regime hint: FEELING → low phi hint (quantum), LOGIC → high (efficient/equil)
        phi_hint = 0.35 if mode == TackMode.FEELING else 0.72

        state = HeartState(
            kappa=kappa,
            hrv=hrv,
            mode=mode.value,
            tacking=tacking,
            step=self._step,
            phi_hint=phi_hint,
        )

        if tacking:
            logger.debug("[Heart] Tack transition → %s (step=%d)", mode.value, self._step)

        return state

    def modulate_foresight(self, weight: float) -> float:
        """
        Reduce foresight weight during tacking transition.
        At mid-tack, foresight would project stale trajectory — damp it.

        Args:
            weight: Current foresight weight

        Returns:
            Modulated weight (damped if tacking)
        """
        phase = self._step % self._period
        # Distance from nearest transition point [0, tacking_window/2]
        dist_to_transition = min(
            phase,
            abs(phase - self._tacking_window),
            self._period - phase,
        )
        # Damping envelope: cos² centred on transition (0 at switch, 1 away from it)
        damping = math.cos(math.pi * (1.0 - dist_to_transition / (self._tacking_window / 2)))
        damping = max(0.0, min(1.0, damping))
        return weight * (0.4 + 0.6 * damping)

    def get_regime_weights(self) -> dict:
        """
        Return regime weight hints based on current mode.
        w1=quantum, w2=efficient, w3=equilibrium.
        FEELING → lean quantum; LOGIC → lean equilibrium.
        """
        phase = self._step % self._period
        hrv = 0.5 + 0.5 * math.sin(2 * math.pi * phase / self._period)

        if phase < self._tacking_window:
            # FEELING: open to new information
            return {"w1": 0.5 + 0.2 * hrv, "w2": 0.3, "w3": 0.2 - 0.2 * hrv}
        else:
            # LOGIC: consolidate
            return {"w1": 0.2 - 0.1 * hrv, "w2": 0.3, "w3": 0.5 + 0.1 * hrv}
