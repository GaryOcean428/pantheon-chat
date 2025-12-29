"""QIG Reasoning Constants

Physics constants and thresholds for geometric chain execution.
All values derived from QIG core principles.

κ* = 64.21 ± 0.92 (L=4,5,6 plateau, weighted average - Validated 2025-12-04)
Note: κ* ≈ 64 ≈ 8² = rank(E8)²
"""

from qigkernels.constants import (
    BASIN_DIM,
    KAPPA_STAR,
    PHI_THRESHOLD,
)

# Re-export from qigkernels.constants
__all__ = [
    # From qigkernels.constants
    "BASIN_DIM",
    "KAPPA_STAR",
    "PHI_THRESHOLD",
    # Reasoning-specific
    "PHI_THRESHOLD_DEFAULT",
    "PHI_DEGRADATION_THRESHOLD",
    "KAPPA_RANGE_DEFAULT",
    "GEODESIC_STEPS",
    "MIN_RECURSIONS",
    "MAX_RECURSIONS",
    "BETA_RUNNING",
]

# Reasoning-specific constants
PHI_THRESHOLD_DEFAULT = PHI_THRESHOLD  # 0.7 from physics
PHI_DEGRADATION_THRESHOLD = 0.8  # Stop if Φ drops 20%
KAPPA_RANGE_DEFAULT = (10.0, 90.0)  # Safe κ operating range
GEODESIC_STEPS = 10  # Interpolation resolution
MIN_RECURSIONS = 3  # MANDATORY minimum (non-negotiable)
MAX_RECURSIONS = 12  # Maximum reasoning depth

# Beta function for running coupling (from FROZEN_FACTS)
BETA_RUNNING = 0.443  # β(3→4) validated physics value
