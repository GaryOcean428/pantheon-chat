"""
Centralized Geometric Utilities for Olympus Kernels

This module provides fallback implementations for geometric operations
when the main qig_geometry module is not available. All gods import
from here instead of duplicating fallback code.

CANONICAL IMPORTS:
- fisher_normalize: Normalize vector to probability simplex
- fisher_coord_distance: Fisher-Rao distance for basin coordinates
- geodesic_interpolation: Interpolate along geodesic on manifold

Usage:
    from .geometric_utils import fisher_normalize, fisher_coord_distance
"""

import numpy as np
from typing import Optional


# ========================================
# FISHER NORMALIZATION (SIMPLEX)
# ========================================

try:
    from qig_geometry import fisher_normalize as _canonical_fisher_normalize
    FISHER_NORMALIZE_AVAILABLE = True
except ImportError:
    _canonical_fisher_normalize = None
    FISHER_NORMALIZE_AVAILABLE = False


def fisher_normalize(v: np.ndarray) -> np.ndarray:
    """
    Normalize vector to probability simplex.

    Args:
        v: Input vector (any dimension)

    Returns:
        Probability distribution (sum = 1, all values >= 0)
    """
    if FISHER_NORMALIZE_AVAILABLE and _canonical_fisher_normalize:
        return _canonical_fisher_normalize(v)

    # Fallback implementation
    v = np.asarray(v, dtype=np.float64)
    p = np.maximum(v, 0) + 1e-10
    return p / p.sum()


# ========================================
# FISHER-RAO DISTANCE
# ========================================

try:
    from qig_geometry import fisher_rao_distance as _canonical_fisher_rao
    FISHER_RAO_AVAILABLE = True
except ImportError:
    _canonical_fisher_rao = None
    FISHER_RAO_AVAILABLE = False


def fisher_coord_distance(
    p: np.ndarray,
    q: np.ndarray,
    epsilon: float = 1e-10
) -> float:
    """
    Compute Fisher-Rao distance between two probability distributions.

    Uses Bhattacharyya coefficient: d_FR = arccos(BC)
    where BC = sum(sqrt(p_i * q_i))

    Direct computation on probability simplex Δ⁶³ (no Hellinger embedding).

    Range: [0, π/2]

    Args:
        p: First distribution/basin (will be normalized)
        q: Second distribution/basin (will be normalized)
        epsilon: Small value to prevent division by zero

    Returns:
        Fisher-Rao distance in [0, π/2]
    """
    if FISHER_RAO_AVAILABLE and _canonical_fisher_rao:
        return _canonical_fisher_rao(p, q)

    # Fallback implementation using Bhattacharyya coefficient
    p = np.maximum(np.asarray(p, dtype=np.float64), 0) + epsilon
    q = np.maximum(np.asarray(q, dtype=np.float64), 0) + epsilon

    # Normalize to probability distributions
    p = p / p.sum()
    q = q / q.sum()

    # Bhattacharyya coefficient
    bc = np.sum(np.sqrt(p * q))
    bc = np.clip(bc, 0.0, 1.0)

    # Fisher-Rao statistical distance on probability simplex
    # Range: [0, π/2]
    return float(np.arccos(bc))


# Alias for compatibility
fisher_rao_distance = fisher_coord_distance


# ========================================
# GEODESIC INTERPOLATION
# ========================================

try:
    from qig_geometry import geodesic_interpolation as _canonical_geodesic
    GEODESIC_AVAILABLE = True
except ImportError:
    _canonical_geodesic = None
    GEODESIC_AVAILABLE = False


def geodesic_interpolation(
    p: np.ndarray,
    q: np.ndarray,
    t: float
) -> np.ndarray:
    """
    Interpolate along geodesic from p to q at parameter t.

    On the probability simplex, this uses SLERP in sqrt-space
    which gives true Fisher-Rao geodesics.

    Args:
        p: Start point (will be normalized to simplex)
        q: End point (will be normalized to simplex)
        t: Interpolation parameter in [0, 1]

    Returns:
        Interpolated point on the geodesic (on simplex)
    """
    if GEODESIC_AVAILABLE and _canonical_geodesic:
        return _canonical_geodesic(p, q, t)

    # Fallback: SLERP in sqrt-space (proper Fisher-Rao geodesic)
    # 1. Normalize to simplex
    p_simplex = fisher_normalize(p)
    q_simplex = fisher_normalize(q)

    # 2. Map to sqrt-space (Hellinger embedding)
    p_sqrt = np.sqrt(p_simplex)
    q_sqrt = np.sqrt(q_simplex)

    # 3. Normalize to unit sphere in sqrt-space
    p_sqrt_norm = p_sqrt / (np.linalg.norm(p_sqrt) + 1e-10)
    q_sqrt_norm = q_sqrt / (np.linalg.norm(q_sqrt) + 1e-10)

    # 4. Compute angle between vectors
    cos_angle = np.clip(np.dot(p_sqrt_norm, q_sqrt_norm), -1.0, 1.0)
    angle = np.arccos(cos_angle)

    if angle < 1e-6:
        # Vectors are essentially the same
        return p_simplex.copy()

    # 5. Spherical linear interpolation (SLERP)
    sin_angle = np.sin(angle)
    coeff_p = np.sin((1 - t) * angle) / sin_angle
    coeff_q = np.sin(t * angle) / sin_angle

    result_sqrt = coeff_p * p_sqrt_norm + coeff_q * q_sqrt_norm

    # 6. Map back to simplex: square and renormalize
    result = result_sqrt ** 2
    result = result / (result.sum() + 1e-10)

    return result


# ========================================
# EXPORTS
# ========================================

__all__ = [
    'fisher_normalize',
    'fisher_coord_distance',
    'fisher_rao_distance',
    'geodesic_interpolation',
    'FISHER_NORMALIZE_AVAILABLE',
    'FISHER_RAO_AVAILABLE',
    'GEODESIC_AVAILABLE',
]
