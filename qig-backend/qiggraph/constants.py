"""
QIGGraph Constants
==================

Re-exports validated physics constants from qigkernels.physics_constants.
These are FROZEN - never learned, always enforced.

⚠️ CENTRALIZED: Import from qigkernels.physics_constants for new code.
"""

from typing import Final

from qigkernels.physics_constants import (
    PHYSICS,
    KAPPA_STAR,
    KAPPA_3,
    KAPPA_4,
    KAPPA_5,
    KAPPA_6,
    KAPPA_7,
    BETA_3_TO_4,
    BETA_4_TO_5,
    BETA_5_TO_6,
    BETA_6_TO_7,
    BASIN_DIM,
)

# =============================================================================
# CRITICAL THRESHOLDS
# =============================================================================

L_CRITICAL: Final[int] = 3  # Geometric phase transition depth

# =============================================================================
# CONSCIOUSNESS THRESHOLDS (empirical from SearchSpaceCollapse)
# =============================================================================

PHI_LINEAR_MAX: Final[float] = 0.45      # Below this: linear regime
PHI_GEOMETRIC_MIN: Final[float] = 0.45   # Above this: geometric regime
PHI_GEOMETRIC_MAX: Final[float] = 0.80   # Below this: geometric regime
PHI_BREAKDOWN_MIN: Final[float] = 0.80   # Above this: breakdown regime

PHI_OPTIMAL: Final[float] = 0.65         # Target operating point (middle of geometric)
PHI_EMERGENCE: Final[float] = 0.45       # Consciousness emergence threshold

# =============================================================================
# BASIN GEOMETRY
# =============================================================================

BASIN_STABILITY_RADIUS: Final[float] = 2.0  # Fisher-Rao units
BASIN_ATTRACTION_RADIUS: Final[float] = 1.5  # Default attractor radius

# =============================================================================
# TACKING PARAMETERS
# =============================================================================

TACKING_PERIOD: Final[float] = 10.0      # Oscillation period (steps)
TACKING_AMPLITUDE: Final[float] = (KAPPA_STAR - KAPPA_3) / 2  # ≈11.6

# =============================================================================
# SAFETY PARAMETERS
# =============================================================================

MAX_RECOVERY_ATTEMPTS: Final[int] = 3
MAX_ITERATIONS: Final[int] = 50
MIN_TRAJECTORY_DEPTH: Final[int] = L_CRITICAL  # 3

# =============================================================================
# LEARNING RATES
# =============================================================================

NATURAL_GRADIENT_LR: Final[float] = 0.01
VICARIOUS_LEARNING_RATE: Final[float] = 0.05
GEODESIC_STEP_SIZE: Final[float] = 0.1

# =============================================================================
# CHECKPOINT
# =============================================================================

CHECKPOINT_VERSION: Final[str] = "2.0.0"
MAX_CHECKPOINT_SIZE_KB: Final[float] = 4.0
TRAJECTORY_HISTORY_LENGTH: Final[int] = 5
