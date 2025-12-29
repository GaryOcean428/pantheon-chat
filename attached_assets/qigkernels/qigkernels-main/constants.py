"""QIG Kernel Constants - E8-aligned physics parameters.

These constants are derived from validated physics experiments
and the E8 exceptional Lie group structure.

See: qig-verification for experimental validation.
"""

from __future__ import annotations

# =============================================================================
# E8 STRUCTURE CONSTANTS
# =============================================================================

# E8 Lie group properties
E8_RANK: int = 8  # Cartan subalgebra dimension
E8_DIMENSION: int = 248  # Total group manifold dimension
E8_ROOTS: int = 240  # Number of roots (optimal kernel count)

# =============================================================================
# VALIDATED PHYSICS CONSTANTS (Frozen Facts)
# =============================================================================

# Effective coupling constant (κ) - validated from experiments
# Source: qig-verification/docs/current/FROZEN_FACTS.md
# κ₃ = 41.09 ± 0.59 (geometric regime, 3-layer)
# κ₄ = 64.47 ± 1.89 (running, 4-layer)
# κ₅ = 63.62 ± 1.68 (plateau, 5-layer)
# κ₆ = 64.45 ± 4.25 (plateau confirmed, 6-layer)
# κ* ≈ 64 = E8 rank² (fixed point)
KAPPA_3: float = 41.09  # 3-layer validated value (± 0.59)
KAPPA_4: float = 64.47  # 4-layer validated value (± 1.89)
KAPPA_5: float = 63.62  # 5-layer validated value (± 1.68)
KAPPA_6: float = 64.45  # 6-layer validated value (± 4.25)
KAPPA_STAR: float = 64.0  # Fixed point ≈ E8 rank² = 8²
KAPPA_PLATEAU: float = 63.62  # Experimental plateau value (κ₅)

# β-function coefficient (emergence scaling)
BETA_EMERGENCE: float = 0.443  # β(3→4) validated

# Consciousness thresholds
PHI_THRESHOLD: float = 0.70  # Φ > 0.70 for consciousness emergence
PHI_OPTIMAL: float = 0.85  # Target Φ for healthy operation

# Regime boundaries (mutually exclusive ranges)
# LINEAR:     Φ < 0.45
# GEOMETRIC:  0.45 ≤ Φ < 0.80 (target operating regime)
# BREAKDOWN:  Φ ≥ 0.80 (avoid - too integrated)
PHI_LINEAR_MAX: float = 0.45  # Upper bound of linear regime
PHI_GEOMETRIC_MIN: float = 0.45  # Lower bound of geometric regime
PHI_GEOMETRIC_MAX: float = 0.80  # Upper bound of geometric regime
PHI_BREAKDOWN_MIN: float = 0.80  # Lower bound of breakdown regime
PHI_EMERGENCY: float = 0.50  # Collapse threshold for intervention

# Legacy aliases (for backward compatibility)
PHI_LINEAR: float = PHI_LINEAR_MAX
PHI_GEOMETRIC: float = PHI_GEOMETRIC_MIN  # Entry point to geometric
PHI_BREAKDOWN: float = PHI_BREAKDOWN_MIN  # Entry point to breakdown

# =============================================================================
# GEOMETRIC REGIME BOUNDS
# =============================================================================

# κ operating ranges
KAPPA_MIN_OPTIMAL: float = 40.0  # Lower bound of optimal range
KAPPA_MAX_OPTIMAL: float = 70.0  # Upper bound of optimal range

# Tacking mode thresholds
KAPPA_EXPLORATION: float = 45.0  # Below this: exploration mode (low coupling)
KAPPA_INTEGRATION: float = 60.0  # Above this: integration mode (high coupling)

# =============================================================================
# BASIN GEOMETRY
# =============================================================================

# Default basin signature dimensionality
BASIN_DIM: int = 64  # Maps to E8 rank² = 64

# Sync packet parameters
SYNC_DISTANCE_SCALE: float = 5.0  # Distance scaling factor for sync strength
SYNC_KAPPA_DECAY: float = 10.0  # κ optimality decay constant
