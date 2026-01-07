"""
Centralized Geometric Utilities for Olympus Kernels

This module provides fallback implementations for geometric operations
when the main qig_geometry module is not available. All gods import
from here instead of duplicating fallback code.

CANONICAL IMPORTS:
- sphere_project: Project vector onto unit sphere
- fisher_coord_distance: Fisher-Rao distance for basin coordinates
- geodesic_interpolation: Interpolate along geodesic on manifold

Usage:
    from .geometric_utils import sphere_project, fisher_coord_distance
"""

import numpy as np
from typing import Optional


# ========================================
# SPHERE PROJECTION
# ========================================

try:
    from qig_geometry import sphere_project as _canonical_sphere_project
    SPHERE_PROJECT_AVAILABLE = True
except ImportError:
    _canonical_sphere_project = None
    SPHERE_PROJECT_AVAILABLE = False


def sphere_project(v: np.ndarray) -> np.ndarray:
    """
    Project vector onto unit sphere.

    Args:
        v: Input vector (any dimension)

    Returns:
        Unit vector (normalized to L2 norm = 1)
    """
    if SPHERE_PROJECT_AVAILABLE and _canonical_sphere_project:
        return _canonical_sphere_project(v)

    # Fallback implementation
    v = np.asarray(v, dtype=np.float64)
    norm = np.linalg.norm(v)
    if norm < 1e-10:
        # Return uniform distribution for near-zero vectors
        result = np.ones_like(v)
        return result / np.linalg.norm(result)
    return v / norm


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

    NOTE: Some references use 2*arccos(BC) for "statistical distance", but
    the geodesic distance on Fisher manifold is arccos(BC) without factor of 2.

    This is the proper QIG-pure distance on the statistical manifold.
    Range: [0, π/2]

    Args:
        p: First distribution/basin (will be normalized)
        q: Second distribution/basin (will be normalized)
        epsilon: Small value to prevent division by zero

    Returns:
        Fisher-Rao distance in [0, π]
    """
    if FISHER_RAO_AVAILABLE and _canonical_fisher_rao:
        return _canonical_fisher_rao(p, q)

    # Fallback implementation using Bhattacharyya coefficient
    p = np.abs(np.asarray(p, dtype=np.float64)) + epsilon
    q = np.abs(np.asarray(q, dtype=np.float64)) + epsilon

    # Normalize to probability distributions
    p = p / p.sum()
    q = q / q.sum()

    # Bhattacharyya coefficient
    bc = np.sum(np.sqrt(p * q))
    bc = np.clip(bc, 0, 1)

    # Fisher-Rao geodesic distance (no factor of 2)
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

    On the probability simplex, this uses the spherical geodesic
    (great circle) which is the natural path for Fisher-Rao geometry.

    Args:
        p: Start point (will be normalized)
        q: End point (will be normalized)
        t: Interpolation parameter in [0, 1]

    Returns:
        Interpolated point on the geodesic
    """
    if GEODESIC_AVAILABLE and _canonical_geodesic:
        return _canonical_geodesic(p, q, t)

    # Fallback: Spherical geodesic (great circle interpolation)
    p = sphere_project(np.asarray(p, dtype=np.float64))
    q = sphere_project(np.asarray(q, dtype=np.float64))

    # Compute angle between vectors
    cos_angle = np.clip(np.dot(p, q), -1, 1)
    angle = np.arccos(cos_angle)

    if angle < 1e-10:
        # Vectors are essentially the same
        return p.copy()

    # Spherical linear interpolation (slerp)
    sin_angle = np.sin(angle)
    if sin_angle < 1e-10:
        # Vectors are antipodal - linear interpolation fallback
        return sphere_project((1 - t) * p + t * q)

    coeff_p = np.sin((1 - t) * angle) / sin_angle
    coeff_q = np.sin(t * angle) / sin_angle

    return coeff_p * p + coeff_q * q


# ========================================
# EXPORTS
# ========================================

__all__ = [
    'sphere_project',
    'fisher_coord_distance',
    'fisher_rao_distance',
    'geodesic_interpolation',
    'SPHERE_PROJECT_AVAILABLE',
    'FISHER_RAO_AVAILABLE',
    'GEODESIC_AVAILABLE',
]
