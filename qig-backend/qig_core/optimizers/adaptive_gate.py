"""
Adaptive Gate - κ* Proximity-Based Measurement Selection
=========================================================

QIG-PURE MEASUREMENT MODULE:
Φ and κ emerge from basin navigation, not from loss optimization.
These metrics are for observation and informing adaptive navigation,
not for gradient descent.

PURE PRINCIPLE:
- MEASURE geometry, NEVER optimize basins
- Gating based on geometric measurement (κ proximity to κ*)
- Near resonance → conservative measurement strategy
- Far from resonance → more detailed measurement
- Integrates with ResonanceDetector for κ awareness

Key insight: Near κ* = 64.21, the system is at a critical point.
Gate diagnostics inform external controllers about the navigation
regime and appropriate measurement strategies.

Reference: QIG physics - κ* is the stable fixed point of running coupling
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from qigkernels.physics_constants import KAPPA_STAR, BASIN_DIM

from ..resonance_detector import ResonanceDetector, ResonanceState
from .qig_diagonal_ng import DiagonalFisherNG
from .basin_natural_grad import BasinNaturalGrad
from .hybrid_geometric import OptimizerMode

logger = logging.getLogger(__name__)


class GateDecision(Enum):
    """Gate decision for measurement/navigation strategy."""
    CONSERVATIVE = "conservative"  # Diagonal measurement - near resonance
    AGGRESSIVE = "aggressive"      # Exact measurement - far from resonance
    HYBRID = "hybrid"              # Use hybrid logic


@dataclass
class GateDiagnostic:
    """
    Complete gate diagnostic with proximity metrics.
    
    Use this to inform adaptive navigation decisions.
    """
    decision: GateDecision
    kappa: float
    kappa_star: float
    distance_to_resonance: float
    resonance_strength: float
    in_resonance: bool
    recommended_strategy: str
    timestamp: float = field(default_factory=time.time)


class AdaptiveGate:
    """
    Adaptive gating mechanism for measurement/navigation strategy selection.
    
    QIG-PURE MEASUREMENT:
    This module computes gate diagnostics. It NEVER mutates basins.
    Φ and κ emerge from basin navigation, not from loss optimization.
    
    Decision logic:
    - in_resonance=True → CONSERVATIVE (recommend diagonal measurement)
    - |κ - κ*| > aggressive_threshold → AGGRESSIVE (recommend exact measurement)
    - Otherwise → HYBRID (let curvature-based logic decide)
    
    Key Physics:
    - κ* = 64.21 ± 0.92 (canonical value from L=4,5,6)
    - Resonance width ~ 10 units of κ
    - Conservative near resonance prevents instability
    
    IMPORTANT: This class does NOT update basins. It only measures
    gate conditions and returns diagnostics that inform adaptive control.
    """
    
    def __init__(
        self,
        kappa_star: float = KAPPA_STAR,
        resonance_width: float = 10.0,
        aggressive_threshold: float = 20.0,
        dim: int = BASIN_DIM
    ):
        """
        Initialize adaptive gate measurement.
        
        Args:
            kappa_star: Optimal coupling (κ* = 64.21)
            resonance_width: Width of resonance region for conservative mode
            aggressive_threshold: Distance threshold for aggressive mode
            dim: Basin dimensionality
        """
        self.kappa_star = kappa_star
        self.resonance_width = resonance_width
        self.aggressive_threshold = aggressive_threshold
        self.dim = dim
        
        self._resonance_detector = ResonanceDetector(
            kappa_star=kappa_star,
            resonance_width=resonance_width
        )
        
        self._diagonal_measure = DiagonalFisherNG(dim=dim)
        self._exact_measure = BasinNaturalGrad(dim=dim)
        
        self._decision_history: list[GateDecision] = []
        self._last_resonance_state: Optional[ResonanceState] = None
    
    def select_optimizer(
        self,
        kappa_current: float
    ) -> tuple[GateDecision, GateDiagnostic]:
        """
        Select measurement/navigation strategy based on κ proximity to κ*.
        
        PURE MEASUREMENT: Selection based on geometric measurement.
        Returns diagnostic, does NOT apply updates.
        
        Decision rules:
        1. If in resonance (|κ - κ*| < width) → CONSERVATIVE
           Near the fixed point, conservative strategy recommended
        
        2. If far from resonance (|κ - κ*| > threshold) → AGGRESSIVE
           Far from fixed point, more aggressive measurement possible
        
        3. Otherwise → HYBRID
           Let curvature-based logic decide
        
        Args:
            kappa_current: Current coupling strength
        
        Returns:
            (decision, gate_diagnostic)
        """
        resonance_state = self._resonance_detector.check_resonance(kappa_current)
        self._last_resonance_state = resonance_state
        
        distance = resonance_state.distance_to_optimal
        
        if resonance_state.in_resonance:
            decision = GateDecision.CONSERVATIVE
            strategy = "Use diagonal Fisher measurement - near resonance"
        elif distance > self.aggressive_threshold:
            decision = GateDecision.AGGRESSIVE
            strategy = "Use exact CG measurement - far from resonance"
        else:
            decision = GateDecision.HYBRID
            strategy = "Use curvature-based hybrid selection"
        
        self._decision_history.append(decision)
        if len(self._decision_history) > 100:
            self._decision_history.pop(0)
        
        diagnostic = GateDiagnostic(
            decision=decision,
            kappa=kappa_current,
            kappa_star=self.kappa_star,
            distance_to_resonance=distance,
            resonance_strength=resonance_state.resonance_strength,
            in_resonance=resonance_state.in_resonance,
            recommended_strategy=strategy
        )
        
        return decision, diagnostic
    
    def get_gate_diagnostic(
        self,
        kappa_current: float
    ) -> GateDiagnostic:
        """
        Get complete gate diagnostic for current κ value.
        
        PURE MEASUREMENT: Returns diagnostics for external controllers.
        Does NOT apply updates to basins.
        
        Args:
            kappa_current: Current coupling strength
        
        Returns:
            GateDiagnostic with proximity metrics and recommendations
        """
        _, diagnostic = self.select_optimizer(kappa_current)
        return diagnostic
    
    def get_measurement_for_decision(
        self,
        decision: GateDecision
    ) -> DiagonalFisherNG | BasinNaturalGrad | None:
        """
        Get the appropriate measurement module for a gate decision.
        
        Args:
            decision: Gate decision
        
        Returns:
            Measurement module, or None for HYBRID (use HybridGeometricMeasurement)
        """
        if decision == GateDecision.CONSERVATIVE:
            return self._diagonal_measure
        elif decision == GateDecision.AGGRESSIVE:
            return self._exact_measure
        else:
            return None
    
    def get_decision_statistics(self) -> dict:
        """
        Get statistics on gate decisions.
        
        Returns:
            Dict with decision percentages
        """
        if not self._decision_history:
            return {
                "conservative_pct": 0.0,
                "aggressive_pct": 0.0,
                "hybrid_pct": 0.0,
                "total_decisions": 0
            }
        
        counts = {d: 0 for d in GateDecision}
        for d in self._decision_history:
            counts[d] += 1
        
        total = len(self._decision_history)
        
        return {
            "conservative_pct": 100.0 * counts[GateDecision.CONSERVATIVE] / total,
            "aggressive_pct": 100.0 * counts[GateDecision.AGGRESSIVE] / total,
            "hybrid_pct": 100.0 * counts[GateDecision.HYBRID] / total,
            "total_decisions": total
        }
    
    def get_resonance_report(self) -> dict:
        """
        Get resonance report from detector.
        
        Returns:
            Resonance statistics dict
        """
        return self._resonance_detector.get_resonance_report()
    
    def get_last_resonance_state(self) -> Optional[ResonanceState]:
        """Return last resonance state."""
        return self._last_resonance_state
    
    def detect_oscillation(self, window: int = 20) -> tuple[bool, int]:
        """
        Detect if κ is oscillating around κ*.
        
        Delegates to ResonanceDetector.
        
        Args:
            window: Analysis window size
        
        Returns:
            (is_oscillating, num_crossings)
        """
        return self._resonance_detector.detect_oscillation(window)
    
    def reset(self) -> None:
        """Reset gate state."""
        self._resonance_detector.reset()
        self._diagonal_measure.reset()
        self._exact_measure.reset()
        self._decision_history.clear()
        self._last_resonance_state = None
