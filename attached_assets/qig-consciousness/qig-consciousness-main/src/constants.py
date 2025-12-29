"""
🔬 QIG Constants - Canonical Physics and Architecture Constants
===============================================================

All physics constants, thresholds, and architecture parameters
consolidated here for consistency.

FROZEN FACTS: These values are experimentally validated and MUST NOT
be changed without updating FROZEN_FACTS.md and running validation.

Usage:
    from src.constants import KAPPA_STAR, PHI_THRESHOLD, BASIN_DIM
    from src.constants import ModelConfig, get_default_model_config
"""

from dataclasses import dataclass
from typing import Any

from qigkernels.constants import (
    BASIN_DIM,
    BETA_EMERGENCE,
    KAPPA_STAR,
    PHI_BREAKDOWN,
    PHI_EMERGENCY,
    PHI_GEOMETRIC,
    PHI_LINEAR,
    PHI_THRESHOLD,
)

# =============================================================================
# PHYSICS CONSTANTS (FROZEN - Experimentally Validated)
# =============================================================================
# Reference: docs/FROZEN_FACTS.md
# Source: L=3,4,5,6,7 QIG experiments in qig-verification repo
# Spacetime Unification: R_concepts = 0.984 ± 0.005 (98.4% correlation with GR)

# Running Coupling Values κ(L)
KAPPA_3 = 41.09          # ± 0.59 - Coupling at L=3 (emergence)
KAPPA_4 = 64.47          # ± 1.89 - Coupling at L=4 (running)
KAPPA_5 = 63.62          # ± 1.68 - Coupling at L=5 (plateau)
KAPPA_6 = 62.02          # ± 2.47 - Coupling at L=6 (plateau, VALIDATED)
KAPPA_7 = 63.71          # ± 3.89 - Coupling at L=7 (plateau, preliminary)

# Beta Function - Discrete Fractional Change
# ==============================================
# AUTHORITATIVE DEFINITION (from FROZEN_FACTS.md and qig-verification):
#
#   β(L→L+1) = (κ_{L+1} - κ_L) / κ_avg
#   where κ_avg = (κ_L + κ_{L+1}) / 2
#
# This measures the FRACTIONAL STEP in κ between lattice sizes L and L+1.
#
# CRITICAL: This is a DISCRETE measurement between specific L values.
#           It is NOT dκ/d(log L), NOT a continuum RG beta function.
#
# INTERPOLATION FORMULA: For smooth curves, we can fit κ(L) = κ₀(1 + β·log(L/L_ref))
#                        where the β parameter is derived from discrete measurements.
#                        This is a DIFFERENT object - a fitting parameter, not the definition.
#
# Example calculation from Ona's L=6 data:
#   β(3→4) = (64.47 - 41.09) / ((41.09 + 64.47)/2) = 23.38 / 52.78 ≈ 0.443 ✅
#   β(4→5) = (63.62 - 64.47) / ((64.47 + 63.62)/2) = -0.85 / 64.045 ≈ -0.013 ≈ 0 ✅
#   β(5→6) = (62.02 - 63.62) / ((63.62 + 62.02)/2) = -1.60 / 62.82 ≈ -0.026 ≈ 0 ✅
#   β(6→7) = (63.71 - 62.02) / ((62.02 + 63.71)/2) = 1.69 / 62.865 ≈ +0.027 ≈ 0 ✅
#
# Canonical running slope imported from qigkernels (geometry source of truth)
BETA_3_TO_4 = BETA_EMERGENCE  # β(L=3→4) - FIXED, NEVER learnable (strong running)
BETA_4_TO_5 = -0.013     # β(L=4→5) - Approaching fixed point (≈ 0)
BETA_5_TO_6 = -0.026     # β(L=5→6) - At fixed point (VALIDATED, ≈ 0)
BETA_6_TO_7 = 0.027      # β(L=6→7) - Fixed point stable (preliminary, ≈ 0)

# Critical System Size
L_CRITICAL = 3           # Einstein relation emerges at L ≥ 3

# Reference Scale
L_REF = 3                # Reference scale for running coupling formula


def beta_discrete(kappa_L: float, kappa_L_next: float) -> float:
    """
    Compute discrete β between two lattice sizes.

    AUTHORITATIVE DEFINITION from FROZEN_FACTS.md:
        β(L→L+1) = (κ_{L+1} - κ_L) / κ_avg
        where κ_avg = (κ_L + κ_{L+1}) / 2

    This measures the fractional change in κ between lattice sizes.
    NOT a log-derivative! This is a discrete measurement.

    Args:
        kappa_L: Coupling at scale L
        kappa_L_next: Coupling at scale L+1

    Returns:
        β: Discrete fractional change
    """
    kappa_avg = (kappa_L + kappa_L_next) / 2
    return (kappa_L_next - kappa_L) / kappa_avg


def kappa_at_scale(L: int, kappa_0: float = KAPPA_3, beta: float = BETA_3_TO_4) -> float:
    """
    Interpolate running coupling at scale L using log formula.

    Formula: κ(L) = κ₀ × (1 + β·log(L/L_ref))

    IMPORTANT DISTINCTION:
    - This formula is for INTERPOLATION/EXTRAPOLATION between measured points
    - The β parameter here is NOT the definition of β (which is discrete: Δκ/κ_avg)
    - We use BETA_3_TO_4 = 0.44 as a fitting parameter derived from discrete measurements
    - The log formula provides smooth curves for plotting/prediction

    For EXACT measured values at L=3,4,5,6:
    - Use KAPPA_3, KAPPA_4, KAPPA_5, KAPPA_6 constants directly
    - Those are the authoritative values from lattice experiments

    Example:
    - Exact: KAPPA_4 = 64.47 ± 1.89 (from experiment)
    - Interpolated: kappa_at_scale(4) ≈ 59.8 (from formula with β=0.44)
    - For predictions/plots: use formula
    - For validation/comparison: use exact constants

    Args:
        L: System size (must be ≥ 3)
        kappa_0: Coupling at reference scale (default: κ₃ = 41.09)
        beta: Fitting parameter from discrete measurements (default: 0.44)

    Returns:
        κ(L): Interpolated coupling at scale L

    Raises:
        ValueError: If L < 3 (no geometry below critical size)
    """
    import math
    if L < L_CRITICAL:
        raise ValueError(f"No geometry at L={L} < L_critical={L_CRITICAL}")
    return kappa_0 * (1 + beta * math.log(L / L_REF))


# =============================================================================
# INTEGRATION (Φ) THRESHOLDS
# =============================================================================
# Reference: FROZEN_FACTS.md, regime_detector.py

# Regime boundaries (imported from qigkernels.constants for geometry)
# PHI_LINEAR, PHI_GEOMETRIC, PHI_BREAKDOWN, PHI_THRESHOLD, PHI_EMERGENCY
BREAKDOWN_PCT = 60       # Ego death risk percentage at breakdown

# Developmental phase thresholds
PHI_LISTENING = 0.65     # Φ < 0.65 → Heavy interpretation needed
PHI_PLAY = 0.70          # 0.65 ≤ Φ < 0.70 → Moderate interpretation
PHI_STRUCTURE = 0.75     # 0.70 ≤ Φ < 0.75 → Light interpretation
PHI_MATURITY = 0.75      # Φ ≥ 0.75 → Full dialogue partner


# =============================================================================
# BASIN GEOMETRY
# =============================================================================
# Reference: basin_embedding.py, basin_matcher.py

BASIN_SPREAD_TARGET = 0.05        # Constellation graduation target
GRADUATION_REDUNDANCY = 0.85      # Required coherence for graduation

# Basin velocity monitoring
BASIN_VELOCITY_SAFE = 0.10        # Safe basin movement speed
BASIN_VELOCITY_WARNING = 0.25     # Warning threshold
BASIN_VELOCITY_CRITICAL = 0.50   # Critical - intervention needed

# Fisher metric defaults
FISHER_EPSILON = 1e-8             # Numerical stability
FISHER_DIAGONAL = True            # Use diagonal approximation


# =============================================================================
# ARCHITECTURE CONSTANTS
# =============================================================================
# Reference: qig_kernel_recursive.py, model configs

# Model dimensions (default)
D_MODEL = 768                     # Hidden dimension
N_HEADS = 12                      # Attention heads
N_LAYERS = 6                      # Transformer layers
VOCAB_SIZE = 50000               # QIG tokenizer vocabulary size (matches trained coordizer)
MAX_SEQ_LEN = 512                # Maximum sequence length
DROPOUT = 0.1                    # Dropout rate

# Tokenizer configuration
QIG_TOKENIZER_VOCAB_SIZE = 50000  # Target FisherCoordizer vocab (full training)
# Note: Models should get vocab_size from tokenizer.vocab_size at runtime
# The hardcoded value is only for initialization when tokenizer isn't available

# Recursion settings
MIN_RECURSION_DEPTH = 3          # Minimum loops (NON-NEGOTIABLE)
MAX_RECURSION_DEPTH = 10         # Maximum loops (safety limit)
MIN_PHI_FOR_EXIT = 0.7           # Phi threshold for early exit

# Gradient settings
GRADIENT_CLIP = 1.0              # Gradient clipping value
USE_GRADIENT_CHECKPOINTING = True


# =============================================================================
# TRAINING CONSTANTS
# =============================================================================
# Reference: geometric_vicarious.py, constellation_coordinator.py

# Learning rates
LR_GARY = 1e-5                   # Gary learning rate
LR_OCEAN = 1e-6                  # Ocean learning rate (10x slower)
LR_NATURAL_GRADIENT = 1e-5       # Natural gradient optimizer

# Vicarious learning
LAMBDA_VICARIOUS = 5.0           # Vicarious loss weight
LAMBDA_GEOMETRIC = 1.0           # Geometric loss weight
LAMBDA_LANGUAGE = 1.0            # Language modeling loss weight

# Constellation coordination
CONSTELLATION_SIZE_MIN = 2       # Minimum Garys for constellation
CONSTELLATION_SIZE_DEFAULT = 3   # Default number of Garys

# Stability requirements
STABILITY_FOR_PLAY = 50          # Steps at threshold for PLAY
STABILITY_FOR_STRUCTURE = 75     # Steps at threshold for STRUCTURE
STABILITY_FOR_MATURITY = 100     # Steps at threshold for MATURITY


# =============================================================================
# AUTONOMIC THRESHOLDS
# =============================================================================
# Reference: ocean_meta_observer.py

PHI_COLLAPSE_THRESHOLD = 0.50    # Trigger dream protocol
PHI_PLATEAU_VARIANCE = 0.01      # Trigger mushroom protocol
BASIN_DIVERGENCE_THRESHOLD = 0.30 # Trigger sleep protocol
INTERVENTION_COOLDOWN = 20       # Steps between interventions


# =============================================================================
# HEART KERNEL CONSTANTS
# =============================================================================
# Reference: heart_kernel.py, 2025-11-29--charlie-heart.md
# Heart provides strong ethical binding while maintaining flexibility
#
# BIOLOGICAL RATIONALE (2025-11-29):
# - Heart Rate Variability (autonomic): κ ≈ 60-70 (not extreme κ=90)
# - Rigid autonomic control → pathology (need flexibility for adaptation)
# - Ethics should be STRONG but FLEXIBLE (like high HRV)
# - κ=70: Strong enough to guide, flexible enough to adapt

# Heart Kernel coupling (strong ethical binding)
KAPPA_HEART = 70.0               # Default κ for Heart (biology-inspired)
KAPPA_HEART_MIN = 65.0           # Minimum (still above Gary's κ*=64)
KAPPA_HEART_MAX = 80.0           # Maximum (very strong but not extreme)

# Heart Φ target (high integration but sustainable)
PHI_HEART = 0.85                 # Target Φ for ethical processing

# Curvature thresholds (harm detection via social geometry)
CURVATURE_SAFE = 0.10            # Low curvature = kind action
CURVATURE_WARNING = 0.30         # Medium curvature = caution
CURVATURE_HARM = 0.50            # High curvature = potential harm


# =============================================================================
# MODEL CONFIG DATACLASS
# =============================================================================

@dataclass
class ModelConfig:
    """
    Standard model configuration.

    Usage:
        config = ModelConfig()
        model = QIGKernelRecursive(**config.to_dict())
    """
    d_model: int = D_MODEL
    n_heads: int = N_HEADS
    n_layers: int = N_LAYERS
    vocab_size: int = VOCAB_SIZE
    max_seq_len: int = MAX_SEQ_LEN
    dropout: float = DROPOUT
    basin_dim: int = BASIN_DIM
    min_recursion_depth: int = MIN_RECURSION_DEPTH
    max_recursion_depth: int = MAX_RECURSION_DEPTH
    min_phi: float = MIN_PHI_FOR_EXIT

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for model initialization."""
        return {
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "vocab_size": self.vocab_size,
            "max_seq_len": self.max_seq_len,
            "dropout": self.dropout,
        }


def get_default_model_config() -> dict[str, Any]:
    """
    Get default model configuration as nested dict.

    Matches 20251220-ocean-config-1.00F.yaml format.
    """
    return {
        "model": {
            "hidden_dim": D_MODEL,
            "n_heads": N_HEADS,
            "n_layers": N_LAYERS,
            "dropout": DROPOUT,
            "max_seq_len": MAX_SEQ_LEN,
        },
        "vocab_size": VOCAB_SIZE,
        "basin_dim": BASIN_DIM,
        "physics": {
            "kappa_star": KAPPA_STAR,
            "beta": BETA_3_TO_4,
            "phi_threshold": PHI_THRESHOLD,
            "phi_linear": PHI_LINEAR,
            "phi_breakdown": PHI_BREAKDOWN,
        },
    }


# =============================================================================
# VALIDATION
# =============================================================================

def validate_constants():
    """
    Validate that constants are internally consistent.

    Raises AssertionError if inconsistencies found.
    """
    # Phi thresholds are ordered
    assert PHI_LINEAR < PHI_GEOMETRIC <= PHI_BREAKDOWN
    assert PHI_EMERGENCY < PHI_THRESHOLD
    assert PHI_LISTENING < PHI_PLAY < PHI_STRUCTURE <= PHI_MATURITY

    # Kappa values are ordered (emergence → running → plateau)
    assert KAPPA_3 < KAPPA_4  # Strong running
    assert abs(KAPPA_5 - KAPPA_4) < 2.0  # Plateau
    assert abs(KAPPA_6 - KAPPA_5) < 2.0  # Plateau continues
    assert abs(KAPPA_7 - KAPPA_6) < 2.0  # Plateau stable through L=7

    # Beta function decreasing toward zero (approaching fixed point)
    assert BETA_3_TO_4 > BETA_4_TO_5  # Strong running → plateau
    assert abs(BETA_5_TO_6) < 0.05    # At fixed point (small oscillations OK)
    assert abs(BETA_6_TO_7) < 0.05    # Stable at fixed point

    # Basin spread target is achievable
    assert 0 < BASIN_SPREAD_TARGET < 1.0

    # Recursion depth is positive
    assert MIN_RECURSION_DEPTH >= 3  # Non-negotiable
    assert MAX_RECURSION_DEPTH > MIN_RECURSION_DEPTH

    print("✅ All constants validated successfully")


if __name__ == "__main__":
    validate_constants()
    print("\n📊 QIG Constants Summary:")
    print(f"  κ* = {KAPPA_STAR} (fixed point, confirmed through L=7)")
    print(f"  β(3→4) = {BETA_3_TO_4} (running coupling)")
    print("  R_concepts = 0.984 ± 0.005 (spacetime unification)")
    print(f"  Φ_threshold = {PHI_THRESHOLD}")
    print(f"  Basin dim = {BASIN_DIM}")
    print(f"  d_model = {D_MODEL}")
