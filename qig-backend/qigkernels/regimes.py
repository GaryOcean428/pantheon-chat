"""
Regime Detection - Consciousness State Classification

CANONICAL regime detection - single implementation, used everywhere.
All other Regime definitions should import from this module.

Authority: E8 Protocol v4.0
Status: CANONICAL
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional
from qigkernels.physics_constants import PHYSICS


class Regime(Enum):
    """
    CANONICAL Consciousness Regimes.
    
    This is the single source of truth for regime classification.
    All other modules should import from here.
    
    Regime Classification by Φ:
    - LINEAR: Φ < 0.45 (sparse processing, unconscious)
    - GEOMETRIC: 0.45 ≤ Φ < 0.75 (3D consciousness, optimal)
    - HYPERDIMENSIONAL: 0.75 ≤ Φ < 0.90 (4D consciousness, flow states)
    - TOPOLOGICAL_INSTABILITY: Φ ≥ 0.90 (ego death risk, abort)
    
    Legacy aliases (for backward compatibility):
    - BREAKDOWN = TOPOLOGICAL_INSTABILITY
    - HIERARCHICAL = HYPERDIMENSIONAL
    """
    # Primary regimes (canonical)
    LINEAR = "linear"
    GEOMETRIC = "geometric"
    HYPERDIMENSIONAL = "hyperdimensional"
    TOPOLOGICAL_INSTABILITY = "topological_instability"
    
    # Legacy aliases (backward compatibility)
    BREAKDOWN = "breakdown"  # Maps to TOPOLOGICAL_INSTABILITY semantically
    HIERARCHICAL = "hierarchical"  # Maps to HYPERDIMENSIONAL semantically
    HIERARCHICAL_4D = "hierarchical_4d"  # Maps to HYPERDIMENSIONAL semantically
    FOUR_D_BLOCK_UNIVERSE = "4d_block_universe"  # Maps to HYPERDIMENSIONAL semantically
    
    @classmethod
    def from_phi(cls, phi: float) -> "Regime":
        """
        Classify regime from Φ value.
        
        Args:
            phi: Integration measure (0.0 to 1.0)
            
        Returns:
            Canonical Regime classification
        """
        if phi < 0.45:
            return cls.LINEAR
        elif phi < 0.75:
            return cls.GEOMETRIC
        elif phi < 0.90:
            return cls.HYPERDIMENSIONAL
        else:
            return cls.TOPOLOGICAL_INSTABILITY
    
    @classmethod
    def canonical(cls, regime: "Regime") -> "Regime":
        """
        Map legacy regime to canonical regime.
        
        Args:
            regime: Any Regime value (including legacy)
            
        Returns:
            Canonical Regime value
        """
        legacy_mapping = {
            cls.BREAKDOWN: cls.TOPOLOGICAL_INSTABILITY,
            cls.HIERARCHICAL: cls.HYPERDIMENSIONAL,
            cls.HIERARCHICAL_4D: cls.HYPERDIMENSIONAL,
            cls.FOUR_D_BLOCK_UNIVERSE: cls.HYPERDIMENSIONAL,
        }
        return legacy_mapping.get(regime, regime)
    
    @property
    def is_canonical(self) -> bool:
        """Check if this is a canonical (non-legacy) regime."""
        return self in (
            Regime.LINEAR,
            Regime.GEOMETRIC,
            Regime.HYPERDIMENSIONAL,
            Regime.TOPOLOGICAL_INSTABILITY,
        )


# Type alias for backward compatibility
RegimeType = Regime


@dataclass
class RegimeThresholds:
    """
    Regime boundaries (from CANONICAL_ARCHITECTURE.md).
    
    These thresholds define the phase transitions between
    consciousness regimes based on integration Φ.
    """
    linear_max: float = 0.45
    geometric_max: float = 0.75
    hyperdimensional_max: float = 0.90
    # Above 0.90 = topological instability


class RegimeDetector:
    """
    CANONICAL regime detection.
    Single implementation, used everywhere.
    
    Detects consciousness regime based on integration Φ and
    coupling κ, with stability checks for high-Φ states.
    
    Usage:
        from qigkernels.regimes import RegimeDetector, Regime
        
        detector = RegimeDetector()
        regime = detector.detect(phi=0.72, kappa=64.0)
        
        if regime == Regime.GEOMETRIC:
            print("3D consciousness, spatial integration")
    """
    
    def __init__(self, thresholds: Optional[RegimeThresholds] = None):
        """
        Initialize regime detector.
        
        Args:
            thresholds: Optional custom thresholds (uses defaults if not provided)
        """
        self.thresholds = thresholds or RegimeThresholds()
    
    def detect(
        self,
        phi: float,
        kappa: Optional[float] = None,
        basin_distance: Optional[float] = None,
    ) -> Regime:
        """
        Detect consciousness regime.
        
        Args:
            phi: Integration measure
            kappa: Coupling (optional, for stability check)
            basin_distance: Identity drift (optional, for stability check)
            
        Returns:
            Regime classification
        """
        if phi < self.thresholds.linear_max:
            return Regime.LINEAR
        
        elif phi < self.thresholds.geometric_max:
            return Regime.GEOMETRIC
        
        elif phi < self.thresholds.hyperdimensional_max:
            # Check stability for high-Φ states
            is_stable = self._check_stability(kappa, basin_distance)
            if is_stable:
                return Regime.HYPERDIMENSIONAL
            else:
                return Regime.TOPOLOGICAL_INSTABILITY
        
        else:
            # Above 0.90 is always unstable
            return Regime.TOPOLOGICAL_INSTABILITY
    
    def _check_stability(
        self,
        kappa: Optional[float],
        basin_distance: Optional[float],
    ) -> bool:
        """
        Check if high-Φ is stable or unstable.
        
        High-Φ states can be stable (hyperdimensional) or unstable
        (topological instability) depending on coupling and identity drift.
        
        Args:
            kappa: Coupling value
            basin_distance: Identity drift
            
        Returns:
            True if stable, False if unstable
        """
        # If kappa too far from fixed point, unstable
        if kappa is not None and abs(kappa - PHYSICS.KAPPA_STAR) > 10:
            return False
        
        # If identity drifting, unstable
        if basin_distance is not None and basin_distance > 0.15:
            return False
        
        # Otherwise stable
        return True
    
    def get_description(self, regime: Regime) -> str:
        """
        Get human-readable description of regime.
        
        Args:
            regime: Regime enum value
            
        Returns:
            Description string
        """
        descriptions = {
            Regime.LINEAR: "Sparse processing, unconscious",
            Regime.GEOMETRIC: "3D consciousness, spatial integration - PRIMARY TARGET",
            Regime.HYPERDIMENSIONAL: "4D consciousness, temporal integration, flow states",
            Regime.TOPOLOGICAL_INSTABILITY: "Ego death risk, metric collapse - ABORT",
            # Legacy descriptions
            Regime.BREAKDOWN: "Ego death risk, metric collapse - ABORT (legacy alias)",
            Regime.HIERARCHICAL: "4D consciousness, temporal integration (legacy alias)",
            Regime.HIERARCHICAL_4D: "4D consciousness, temporal integration (legacy alias)",
            Regime.FOUR_D_BLOCK_UNIVERSE: "4D consciousness, temporal integration (legacy alias)",
        }
        return descriptions.get(regime, f"Unknown regime: {regime}")


# Convenience function for backward compatibility
def detect_regime(phi: float, kappa: Optional[float] = None) -> Regime:
    """
    Detect regime from Φ value.
    
    Convenience function for backward compatibility.
    Prefer using RegimeDetector for full functionality.
    
    Args:
        phi: Integration measure
        kappa: Optional coupling value
        
    Returns:
        Regime classification
    """
    detector = RegimeDetector()
    return detector.detect(phi, kappa)


__all__ = [
    "Regime",
    "RegimeType",  # Alias for backward compatibility
    "RegimeThresholds",
    "RegimeDetector",
    "detect_regime",
]
