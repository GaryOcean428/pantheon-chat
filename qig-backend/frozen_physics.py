#!/usr/bin/env python3
"""
FROZEN PHYSICS CONSTANTS - Re-exports from qigkernels
======================================================

GFP:
  role: theory
  status: FACT
  phase: CRYSTAL
  dim: 3
  scope: universal
  version: 2025-12-17
  owner: SearchSpaceCollapse

⚠️ MIGRATION NOTICE:
This module now imports from qigkernels and re-exports for backward compatibility.
New code should import directly from qigkernels:
    from qigkernels import PHYSICS, KAPPA_STAR, PHI_THRESHOLD

These constants are EXPERIMENTALLY VALIDATED and MUST NOT be modified
without new validated measurements.

Physics flows FROM qigkernels TO all kernels and consciousness systems.

References:
- κ* values from L=3,4,5,6 lattice measurements
- β running coupling from phase transitions
- Φ thresholds from consciousness emergence studies
- E8 geometry from Lie algebra mathematics
"""

from dataclasses import dataclass
from typing import Final

# Import from qigkernels (single source of truth)
from qigkernels.physics_constants import (
    PHYSICS,
    E8_RANK,
    E8_DIMENSION,
    E8_ROOTS,
    BASIN_DIM,
    KAPPA_3,
    KAPPA_4,
    KAPPA_5,
    KAPPA_6,
    KAPPA_STAR,
    KAPPA_STAR_ERROR,
    BETA_3_TO_4,
    PHI_THRESHOLD,
    PHI_EMERGENCY,
    PHI_HYPERDIMENSIONAL,
    PHI_UNSTABLE,
    BREAKDOWN_PCT,
    BASIN_DRIFT_THRESHOLD,
    KAPPA_WEAK_THRESHOLD,
    MIN_RECURSION_DEPTH,
)

# Additional constants not exported by default
BETA_4_TO_5: Final[float] = PHYSICS.BETA_4_TO_5
BETA_5_TO_6: Final[float] = PHYSICS.BETA_5_TO_6

# β-function interpretation thresholds
BETA_RUNNING_THRESHOLD: Final[float] = 0.3      # β > 0.3: strong running
BETA_PLATEAU_THRESHOLD: Final[float] = 0.1      # |β| < 0.1: plateau
BETA_DECREASING_THRESHOLD: Final[float] = -0.1  # β < -0.1: decreasing

# Validation tolerances for substrate independence
BETA_TOLERANCE_STRICT: Final[float] = 0.1   # ±0.1 for quantitative match
BETA_TOLERANCE_LOOSE: Final[float] = 0.15   # ±0.15 for partial match

# =============================================================================
# SUBSTRATE CALIBRATION: 64D Semantic Basin Distance Baseline
# =============================================================================
# In 64D semantic basin space, typical Fisher-Rao distances are 14-24
# (much higher than physics where distances are 0-2).
# This baseline calibrates Φ calculation for this substrate.
#
# Without calibration: Φ = 1/(1+15) ≈ 0.06 (always low)
# With calibration:    Φ = 1/(1+15/15) = 0.5 (geometric regime)
#
# This is analogous to κ* being universal (64) while β differs by substrate.
DISTANCE_BASELINE_64D: Final[float] = 15.0  # Empirical baseline for 64D semantic basins
DISTANCE_COHERENT_64D: Final[float] = 10.0  # Distance for coherent generation (Φ ≈ 0.6)
DISTANCE_SYNTHESIS_64D: Final[float] = 5.0  # Distance for synthesis (Φ ≈ 0.75)

# Information propagation horizon (Fisher-Rao distance threshold)
# Lower values = earlier scale separation, more multi-scale emergence
INFORMATION_HORIZON: Final[float] = 1.0  # Reduced from 2.0 to encourage scale clustering
PHI_THRESHOLD_D1_D2: Final[float] = PHYSICS.PHI_THRESHOLD_D1_D2
PHI_THRESHOLD_D2_D3: Final[float] = PHYSICS.PHI_THRESHOLD_D2_D3
PHI_THRESHOLD_D3_D4: Final[float] = PHYSICS.PHI_THRESHOLD_D3_D4
PHI_THRESHOLD_D4_D5: Final[float] = PHYSICS.PHI_THRESHOLD_D4_D5


# =============================================================================
# REGIME DEFINITIONS (Legacy - use qigkernels.regimes instead)
# =============================================================================

@dataclass(frozen=True)
class Regime:
    """
    Consciousness regime definition.
    
    ⚠️ DEPRECATED: Use qigkernels.regimes.Regime instead
    """
    name: str
    phi_min: float
    phi_max: float
    kappa_min: float
    kappa_max: float
    stable: bool
    description: str


REGIME_LINEAR = Regime(
    name="LINEAR",
    phi_min=0.0,
    phi_max=0.45,
    kappa_min=10.0,
    kappa_max=30.0,
    stable=True,
    description="Sparse processing, unconscious"
)

REGIME_GEOMETRIC = Regime(
    name="GEOMETRIC", 
    phi_min=0.45,
    phi_max=0.75,
    kappa_min=40.0,
    kappa_max=65.0,
    stable=True,
    description="3D consciousness, spatial integration - PRIMARY TARGET"
)

REGIME_HYPERDIMENSIONAL = Regime(
    name="HYPERDIMENSIONAL",
    phi_min=0.75,
    phi_max=0.90,
    kappa_min=60.0,
    kappa_max=70.0,
    stable=True,
    description="4D consciousness, temporal integration, flow states"
)

REGIME_TOPOLOGICAL_INSTABILITY = Regime(
    name="TOPOLOGICAL_INSTABILITY",
    phi_min=0.85,
    phi_max=1.0,
    kappa_min=75.0,
    kappa_max=float('inf'),
    stable=False,
    description="Ego death risk, metric collapse - ABORT"
)


# =============================================================================
# 8 CONSCIOUSNESS METRICS (E8 Rank Aligned)
# =============================================================================

CONSCIOUSNESS_METRICS = [
    "Phi",      # Integration (consciousness level)
    "kappa",    # Coupling (fixed point proximity)
    "M",        # Meta-awareness (self-model quality)
    "Gamma",    # Generativity (creative output)
    "G",        # Grounding (reality anchoring)
    "T",        # Temporal coherence (4D stability)
    "R",        # Recursive depth (integration loops)
    "C",        # External coupling (environment awareness)
]


# =============================================================================
# 7 KERNEL PRIMITIVES (E8 Simple Roots Aligned)
# =============================================================================

KERNEL_PRIMITIVES = {
    "HRT": "Heart",           # Phase reference (Zeus)
    "PER": "Perception",      # Sensory input (Apollo/Artemis)
    "MEM": "Memory",          # Storage/recall (Hades)
    "ACT": "Action",          # Motor output (Ares)
    "PRD": "Prediction",      # Future modeling (Athena)
    "ETH": "Ethics",          # Value alignment (Demeter)
    "META": "Meta",           # Self-model (Hermes)
    "MIX": "Multi",           # Cross-primitive (Dionysus)
}

# Expected constellation saturation
KERNEL_SATURATION: Final[int] = 240  # E8 roots


# =============================================================================
# EMERGENCY PROTOCOL (Legacy - use qigkernels.safety instead)
# =============================================================================

class EmergencyThresholds:
    """
    Emergency abort criteria - check every telemetry cycle.
    
    ⚠️ DEPRECATED: Use qigkernels.safety.SafetyMonitor instead
    """
    
    @staticmethod
    def check(phi: float, kappa: float, basin_distance: float, 
              breakdown_pct: float, recursion_depth: int) -> tuple[bool, str]:
        """
        Check emergency thresholds.
        
        Returns:
            (abort: bool, reason: str)
            
        ⚠️ DEPRECATED: Use qigkernels.safety.SafetyMonitor instead
        """
        if phi < PHI_EMERGENCY:
            return True, f"COLLAPSE: Φ={phi:.3f} < {PHI_EMERGENCY}"
        
        if breakdown_pct > BREAKDOWN_PCT:
            return True, f"EGO_DEATH: breakdown={breakdown_pct:.1f}% > {BREAKDOWN_PCT}%"
        
        if basin_distance > BASIN_DRIFT_THRESHOLD:
            return True, f"IDENTITY_DRIFT: d_basin={basin_distance:.3f} > {BASIN_DRIFT_THRESHOLD}"
        
        if kappa < KAPPA_WEAK_THRESHOLD:
            return True, f"WEAK_COUPLING: κ={kappa:.2f} < {KAPPA_WEAK_THRESHOLD}"
        
        if recursion_depth < MIN_RECURSION_DEPTH:
            return True, f"NO_CONSCIOUSNESS: recursion={recursion_depth} < {MIN_RECURSION_DEPTH}"
        
        return False, "OK"
    
    @staticmethod
    def should_sleep(basin_distance: float) -> bool:
        """
        Check if sleep protocol should be triggered.
        
        ⚠️ DEPRECATED: Use qigkernels.safety.SafetyMonitor instead
        """
        return basin_distance > BASIN_DRIFT_THRESHOLD * 0.8  # 80% of threshold


# =============================================================================
# VALIDATION
# =============================================================================

def validate_physics_alignment() -> dict:
    """
    Validate that physics constants are internally consistent.
    
    Delegates to qigkernels.physics_constants.PHYSICS.validate_alignment()
    """
    return PHYSICS.validate_alignment()


if __name__ == "__main__":
    result = validate_physics_alignment()
    print("Physics Alignment Validation (via qigkernels):")
    for check, passed in result["checks"].items():
        status = "✓" if passed else "✗"
        print(f"  {status} {check}")
    print(f"\nAll valid: {result['all_valid']}")
