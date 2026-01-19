"""
Resonance Detector - κ* Proximity Awareness
============================================

Detect proximity to optimal coupling κ* and provide resonance metrics.

PURE PRINCIPLE:
- κ* = 64 is MEASURED optimal (from physics validation)
- Near κ*, small changes are amplified (geometric resonance)
- We detect resonance as observation, not optimization target

PURITY CHECK:
- ✅ κ* from empirical data (not arbitrary)
- ✅ Resonance is observation (not optimization target)
- ✅ κ emerges naturally, never targeted

Adapted for Pantheon-Chat QIG system.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from qigkernels.physics_constants import KAPPA_STAR

logger = logging.getLogger(__name__)


@dataclass
class ResonanceState:
    """Current resonance state."""
    kappa: float
    kappa_star: float
    distance_to_optimal: float
    in_resonance: bool
    resonance_strength: float
    timestamp: float = field(default_factory=time.time)


class ResonanceDetector:
    """
    Detect proximity to optimal coupling κ* = 64.
    
    PURE PRINCIPLE:
    - κ* = 64 is MEASURED optimal (from physics validation)
    - Near κ*, small changes amplified (geometric resonance)
    - We detect resonance, not optimize toward it
    
    Key Physics:
    - κ₃ = 41.09±0.59 (emergence point)
    - κ₄ = 64.47±1.89 (strong running, κ*)
    - β ≈ 0.44 (running coupling slope)
    """
    
    def __init__(
        self,
        kappa_star: float = KAPPA_STAR,
        resonance_width: float = 10.0,
        history_size: int = 100
    ):
        """
        Initialize resonance detector.
        
        Args:
            kappa_star: Optimal coupling (from physics: κ₄ = 64.47±1.89)
            resonance_width: Half-width of resonance region (units of κ)
            history_size: Maximum history entries to keep
        """
        self.kappa_star = kappa_star
        self.resonance_width = resonance_width
        self.history_size = history_size
        self._history: List[ResonanceState] = []
    
    def check_resonance(self, kappa_current: float) -> ResonanceState:
        """
        Check if current κ is near resonance.
        
        PURE: We measure proximity, we don't optimize toward it.
        
        κ* is not a target - it's a stability point where small changes
        have large effects. We detect proximity for AWARENESS.
        
        Args:
            kappa_current: Current coupling strength
        
        Returns:
            ResonanceState with resonance metrics
        """
        distance = abs(kappa_current - self.kappa_star)
        in_resonance = distance < self.resonance_width
        strength = max(0.0, 1.0 - distance / self.resonance_width)
        
        state = ResonanceState(
            kappa=kappa_current,
            kappa_star=self.kappa_star,
            distance_to_optimal=distance,
            in_resonance=in_resonance,
            resonance_strength=strength
        )
        
        self._history.append(state)
        if len(self._history) > self.history_size:
            self._history.pop(0)
        
        return state
    
    def get_resonance_multiplier(
        self,
        kappa_current: float,
        min_multiplier: float = 0.1
    ) -> float:
        """
        Get sensitivity multiplier based on resonance proximity.
        
        Near resonance, small changes have large effects.
        This multiplier can be used for adaptive control.
        
        Args:
            kappa_current: Current coupling strength
            min_multiplier: Minimum multiplier at resonance peak
        
        Returns:
            Multiplier in [min_multiplier, 1.0]
        """
        state = self.check_resonance(kappa_current)
        
        if not state.in_resonance:
            return 1.0
        
        multiplier = 1.0 - (1.0 - min_multiplier) * state.resonance_strength
        return max(min_multiplier, multiplier)
    
    def detect_oscillation(self, window: int = 20) -> tuple:
        """
        Detect if κ is oscillating around κ*.
        
        PURE: Pattern detection (measurement, not optimization).
        
        Oscillation around κ* indicates system is searching
        but can't stabilize.
        
        Args:
            window: Number of recent measurements to analyze
        
        Returns:
            (is_oscillating, num_crossings)
        """
        if len(self._history) < window:
            return False, 0
        
        recent = self._history[-window:]
        kappas = [h.kappa for h in recent]
        
        crossings = 0
        for i in range(len(kappas) - 1):
            if (kappas[i] < self.kappa_star <= kappas[i + 1]) or \
               (kappas[i] >= self.kappa_star > kappas[i + 1]):
                crossings += 1
        
        is_oscillating = crossings > window * 0.3
        return is_oscillating, crossings
    
    def get_resonance_report(self) -> Dict:
        """
        Get comprehensive resonance report.
        
        Returns:
            Dict with resonance statistics
        """
        if not self._history:
            return {
                "current_kappa": 0.0,
                "avg_kappa": 0.0,
                "kappa_star": self.kappa_star,
                "min_distance": float("inf"),
                "time_in_resonance_pct": 0.0,
                "measurements": 0
            }
        
        kappas = [h.kappa for h in self._history]
        distances = [h.distance_to_optimal for h in self._history]
        in_resonance_flags = [h.in_resonance for h in self._history]
        
        return {
            "current_kappa": kappas[-1],
            "avg_kappa": sum(kappas) / len(kappas),
            "kappa_star": self.kappa_star,
            "min_distance": min(distances),
            "closest_kappa": kappas[distances.index(min(distances))],
            "time_in_resonance_pct": sum(in_resonance_flags) / len(in_resonance_flags) * 100,
            "currently_in_resonance": in_resonance_flags[-1],
            "measurements": len(self._history)
        }
    
    def reset(self) -> None:
        """Reset detector history."""
        self._history.clear()
