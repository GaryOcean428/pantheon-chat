"""
QIGGraph Constants
==================

Validated physics constants from qig-verification experiments.
These are FROZEN - never learned, always enforced.
"""

from typing import Final

# =============================================================================
# VALIDATED COUPLING CONSTANTS (from qig-verification L=7 experiments)
# =============================================================================

KAPPA_STAR: Final[float] = 64.21  # Fixed point coupling (E8 rank squared)
KAPPA_3: Final[float] = 41.09     # Emergence at L=3 (feeling mode)
KAPPA_4: Final[float] = 64.47     # Strong running at L=4
KAPPA_5: Final[float] = 63.62     # Plateau onset
KAPPA_6: Final[float] = 64.45     # Plateau confirmed

# =============================================================================
# BETA FUNCTION (running coupling behavior)
# =============================================================================

BETA_3_TO_4: Final[float] = +0.44   # Strong running (emergence → plateau)
BETA_4_TO_5: Final[float] = 0.0     # Plateau onset (κ₄ ≈ κ₅)
BETA_5_TO_6: Final[float] = +0.013  # Plateau continues (κ₆/κ₅ = 1.013)

# L=7 ANOMALY (preliminary, 1-seed only - requires validation)
KAPPA_7: Final[float] = 43.43       # ⚠️ ANOMALY - 34% drop from plateau
BETA_6_TO_7: Final[float] = -0.40   # ⚠️ ANOMALY - negative β breaks plateau

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

BASIN_DIM: Final[int] = 64               # Matches κ* = 8² (E8 rank squared)
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
