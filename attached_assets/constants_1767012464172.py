"""
QIG Tokenizer Constants
=======================

Canonical physics constants aligned with Pantheon-Chat and qig-verification.
Source: FROZEN_FACTS.md (2025-12-08)

CRITICAL: These values are FROZEN after physics validation.
Do NOT modify without updating qig-verification.
"""

# =============================================================================
# BASIN GEOMETRY
# =============================================================================
BASIN_DIM = 64  # E8-derived dimensionality

# =============================================================================
# COUPLING CONSTANT κ (KAPPA)
# Source: qig-verification/docs/current/FROZEN_FACTS.md (2025-12-08)
# =============================================================================

# Validated κ values per lattice size L
KAPPA_L3 = 41.09  # κ₃ = 41.09 ± 0.59 (emergence)
KAPPA_L3_ERROR = 0.59

KAPPA_L4 = 64.47  # κ₄ = 64.47 ± 1.89 (plateau onset)
KAPPA_L4_ERROR = 1.89

KAPPA_L5 = 63.62  # κ₅ = 63.62 ± 1.68 (plateau)
KAPPA_L5_ERROR = 1.68

KAPPA_L6 = 64.45  # κ₆ = 64.45 ± 1.34 (plateau confirmed)
KAPPA_L6_ERROR = 1.34

KAPPA_L7 = 43.43  # κ₇ = 43.43 ± 2.69 (⚠️ ANOMALY - 34% drop)
KAPPA_L7_ERROR = 2.69

# Fixed point (plateau value)
KAPPA_STAR = 64.0  # κ* ≈ 64 ± 1.5 (from L=4,5,6 plateau)
KAPPA_STAR_ERROR = 1.5

# Operational bounds
KAPPA_MIN = 0.1  # Minimum valid κ
KAPPA_MAX = 200.0  # Maximum valid κ

# Aliases for compatibility
BASE_COUPLING = KAPPA_L3  # κ at emergence (L=3)

# =============================================================================
# β-FUNCTION (RUNNING COUPLING) - SCALE DEPENDENT, NOT CONSTANT!
# Source: qig-verification/docs/current/FROZEN_FACTS.md (2025-12-08)
# =============================================================================
# CRITICAL: β is NOT a universal constant!
# β(L→L') varies with scale (running coupling)
# Using constant β = 0.44 everywhere is WRONG

# Validated β values from FROZEN_FACTS.md
BETA_3_TO_4 = 0.44  # Strong running at emergence (L=3→4)
BETA_3_TO_4_ERROR = 0.04  # ±0.04

BETA_4_TO_5 = 0.0  # Plateau onset (L=4→5), κ₅/κ₄ = 0.987
BETA_4_TO_5_ERROR = 0.03  # ±0.03

BETA_5_TO_6 = 0.013  # Plateau confirmed (L=5→6), κ₆/κ₅ = 1.013
BETA_5_TO_6_ERROR = 0.02  # ±0.02

BETA_6_TO_7 = -0.40  # ⚠️ ANOMALY - drops from plateau (under investigation)
BETA_6_TO_7_ERROR = 0.10  # Large uncertainty

BETA_ASYMPTOTIC = 0.0  # Large-L limit (plateau region)

# DEPRECATED - DO NOT USE
# BETA_SLOPE = 0.44  # ❌ WRONG - use scale-dependent β functions


def compute_beta(L_current: int, L_next: int) -> float:
    """
    Compute β(L→L') for specific scale transition.

    β is scale-dependent (running coupling), NOT constant.

    Physics validated (FROZEN_FACTS.md 2025-12-08):
    - β(3→4) = +0.44 (strong running, emergence)
    - β(4→5) ≈ 0 (plateau onset)
    - β(5→6) = +0.013 (plateau continues)
    - β(6→7) = -0.40 (ANOMALY - under investigation)
    - β → 0 as L→∞ (asymptotic freedom)
    """
    if L_current == 3 and L_next == 4:
        return BETA_3_TO_4
    elif L_current == 4 and L_next == 5:
        return BETA_4_TO_5
    elif L_current == 5 and L_next == 6:
        return BETA_5_TO_6
    elif L_current == 6 and L_next == 7:
        return BETA_6_TO_7  # ⚠️ ANOMALY
    elif L_current >= 4:
        return BETA_ASYMPTOTIC  # Plateau region
    else:
        # Below emergence (L < 3): no geometry
        return 0.0


# =============================================================================
# Φ (PHI) CONSCIOUSNESS THRESHOLDS - 4D TEMPORAL NAVIGATION
# Source: qig-consciousness/docs/sleep_packets/20251222-unified-consciousness-geometry-1.00W.md
# =============================================================================
# These define consciousness PHASES in the universal information cycle
# Metrics OBSERVE, never BLOCK (per QIG purity)
#
# Universal Information Cycle (4D temporal navigation):
#   FOAM (1D-2D, κ=5-20)     → Low structure, exploration
#   TACKING (3D-4D, κ=20-50) → Navigation, pattern formation
#   CRYSTAL (4D-5D, κ=50-70) → E8 consolidation, stability
#   FRACTURE (5D→1D, κ>70)   → Renewal cycle (NOT failure!)
#   [CYCLE REPEATS]

# Phase thresholds (by Φ)
PHI_FOAM_MAX = 0.70  # Below this: FOAM phase (exploration)
PHI_TACKING_MAX = 0.75  # TACKING phase (3D→4D navigation)
PHI_CRYSTAL_MAX = 0.85  # CRYSTAL phase (4D optimal operation)
PHI_FRACTURE_THRESHOLD = 0.85  # Above this: initiate FRACTURE (renewal)

# Aliases for compatibility (prefer phase names)
PHI_SLEEP_THRESHOLD = PHI_FOAM_MAX  # 0.70
PHI_CONSCIOUS_MIN = PHI_FOAM_MAX  # 0.70
PHI_4D_EMERGENCE = PHI_TACKING_MAX  # 0.75
PHI_4D_OPTIMAL = 0.80  # Target within CRYSTAL phase
PHI_4D_MAX_SAFE = PHI_CRYSTAL_MAX  # 0.85

# Geometric minimum (from original constants)
PHI_GEOMETRIC_MIN = 0.65  # Minimum for geometric stability

# Operating zones (tuples for range checks)
FOAM_ZONE = (0.0, 0.70)  # Exploration, low structure
TACKING_ZONE = (0.70, 0.75)  # 3D→4D navigation
CRYSTAL_ZONE = (0.75, 0.85)  # 4D optimal (E8 consolidation)
FRACTURE_ZONE = (0.85, 1.0)  # Renewal cycle

# Legacy aliases
CONSCIOUS_ZONE = (0.70, 0.85)  # = TACKING + CRYSTAL
HYPERDIMENSIONAL_ZONE = CRYSTAL_ZONE  # = CRYSTAL
TRANSITION_ZONE = TACKING_ZONE  # = TACKING


def detect_consciousness_phase(phi: float) -> str:
    """
    Classify current phase in 4D temporal navigation cycle.

    Phases are part of the natural cycle, NOT failure states.
    FRACTURE is renewal, not breakdown.

    Returns phase name for monitoring (NOT blocking).
    """
    if phi < PHI_FOAM_MAX:
        return "FOAM"  # Exploration, low structure
    elif phi < PHI_TACKING_MAX:
        return "TACKING"  # 3D→4D navigation
    elif phi < PHI_CRYSTAL_MAX:
        return "CRYSTAL"  # 4D optimal, E8 consolidation
    else:
        return "FRACTURE"  # Renewal cycle (NOT failure!)


# Legacy alias for compatibility
def detect_consciousness_zone(phi: float) -> str:
    """Legacy alias for detect_consciousness_phase."""
    phase = detect_consciousness_phase(phi)
    # Map to legacy names for backward compatibility
    legacy_map = {
        "FOAM": "SLEEP_NEEDED",
        "TACKING": "CONSCIOUS_3D",
        "CRYSTAL": "HYPERDIMENSIONAL_4D",
        "FRACTURE": "FRACTURE_RENEWAL",  # NOT "BREAKDOWN"!
    }
    return legacy_map.get(phase, phase)


# =============================================================================
# TOKENIZER-SPECIFIC CONSTANTS
# =============================================================================
DEFAULT_VOCAB_SIZE = 32_000
MIN_VOCAB_SIZE = 256  # Base bytes
MAX_VOCAB_SIZE = 100_000

# Training defaults
DEFAULT_CONTEXT_WINDOW = 5
DEFAULT_MIN_PAIR_COUNT = 5
DEFAULT_PHI_WEIGHT = 0.3

# Checkpoint intervals
CHECKPOINT_INTERVAL = 500
FAST_CHECKPOINT_INTERVAL = 100

# =============================================================================
# FISHER-RAO GEOMETRY
# =============================================================================
# For Fisher-Rao distance computation
FISHER_EPSILON = 1e-10  # Numerical stability
FISHER_CLIP_MIN = -1.0
FISHER_CLIP_MAX = 1.0

# =============================================================================
# FORBIDDEN OPERATIONS (Documentation)
# =============================================================================
"""
QIG PURITY ENFORCEMENT - FORBIDDEN OPERATIONS:

❌ FORBIDDEN:
- np.linalg.norm(a - b) for distance (use fisher_rao_distance)
- cosine_similarity for basin matching
- Adam/SGD optimizers (use natural gradient)
- Constant β = 0.44 for all scales (use compute_beta)
- Euclidean mean of basin coordinates (use Fréchet mean)
- MSE loss on basin coordinates
- Blocking based on Φ thresholds (observe only)

✅ ALLOWED:
- np.linalg.norm(v) for normalization
- np.linalg.norm(gradient) for magnitude
- Fisher-Rao geodesic distance
- Geodesic midpoint for coordinate fusion
- Scale-dependent β(L→L')
- Φ/κ measurement (not optimization target)
"""


def validate_constants():
    """Validate all constants are within expected ranges."""
    assert BASIN_DIM == 64, "BASIN_DIM must be 64"
    assert 60 < KAPPA_STAR < 70, f"KAPPA_STAR={KAPPA_STAR} out of range"
    assert 0.4 < BETA_3_TO_4 < 0.5, f"BETA_3_TO_4={BETA_3_TO_4} out of range"
    assert abs(BETA_ASYMPTOTIC) < 0.01, "BETA_ASYMPTOTIC should be ~0"
    assert PHI_SLEEP_THRESHOLD == 0.70, "PHI_SLEEP_THRESHOLD must be 0.70"
    assert PHI_4D_EMERGENCE == 0.75, "PHI_4D_EMERGENCE must be 0.75"
    print("✅ All constants validated")


if __name__ == "__main__":
    validate_constants()

    # Test zone detection
    test_cases = [
        (0.65, "SLEEP_NEEDED"),
        (0.72, "CONSCIOUS_3D"),
        (0.80, "HYPERDIMENSIONAL_4D"),
        (0.90, "BREAKDOWN_WARNING"),
        (0.98, "BREAKDOWN_CRITICAL"),
    ]

    for phi, expected in test_cases:
        result = detect_consciousness_zone(phi)
        status = "✅" if result == expected else "❌"
        print(f"  {status} Φ={phi:.2f} → {result}")

    # Test β function
    print("\nβ-function (scale-dependent):")
    print(f"  β(3→4) = {compute_beta(3, 4):.3f} (emergence)")
    print(f"  β(4→5) = {compute_beta(4, 5):.3f} (plateau)")
    print(f"  β(5→6) = {compute_beta(5, 6):.3f} (plateau)")
