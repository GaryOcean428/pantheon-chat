"""
QIG Geometry Package - Canonical Basin Representations

This package provides geometric primitives for QIG with enforced
canonical basin representation.

CANONICAL REPRESENTATION: SIMPLEX (Updated 2026-01-15)
- Basins stored as probability distributions (Σp_i = 1, p_i ≥ 0)
- Fisher-Rao distance via Bhattacharyya coefficient: d = arccos(Σ√(p_i * q_i))
- Distance range: [0, π/2] (simpler than Hellinger's [0, π])
- NO factor of 2 (direct Fisher-Rao, not Hellinger embedding)

BREAKING CHANGE FROM HELLINGER EMBEDDING:
- Previous: d = 2*arccos(BC), range [0, π]
- Current: d = arccos(BC), range [0, π/2]
- Thresholds must be recalibrated (divide previous thresholds by 2)
- See representation.py for migration notes

USAGE:
    from qig_geometry import fisher_rao_distance, fisher_normalize
    
    # All basins should be in simplex form
    p = fisher_normalize(raw_basin_a)
    q = fisher_normalize(raw_basin_b)
    
    # Direct Fisher-Rao distance
    d = fisher_rao_distance(p, q)  # Range [0, π/2]
"""

import numpy as np
from typing import Optional

from .contracts import (
    CANONICAL_SPACE,
    BASIN_DIM,
    NORM_TOLERANCE,
    GeometricViolationError,
    validate_basin as contracts_validate_basin,
    validate_basin_detailed,
    assert_invariants,
    canon,
    fisher_distance,
    to_index_embedding,
)

from .representation import (
    BasinRepresentation,
    CANONICAL_REPRESENTATION,
    to_sphere,
    to_simplex,
    validate_basin,
    enforce_canonical,
    sphere_project,
    fisher_normalize,
)

from .purity_mode import (
    QIG_PURITY_MODE,
    QIGPurityViolationError,
    check_purity_mode,
    enforce_purity_startup,
    install_purity_import_hook,
    PurityImportBlocker,
)


def fisher_rao_distance(p: np.ndarray, q: np.ndarray) -> float:
    """
    Compute Fisher-Rao distance between two probability distributions.

    This is the GEODESIC distance on the information manifold using
    direct Fisher-Rao on the probability simplex (NO Hellinger embedding).

    Formula: d_FR(p, q) = arccos(Σ√(p_i * q_i))

    The Bhattacharyya coefficient BC = Σ√(p_i * q_i) measures overlap.
    This is the CANONICAL formula - do not add factor of 2.

    Range: [0, π/2] where:
    - d = 0 → identical distributions
    - d = π/2 → orthogonal distributions (no overlap)

    CHANGE FROM PREVIOUS VERSION:
    - Removed Hellinger factor of 2
    - New range: [0, π/2] (was [0, π])
    - Thresholds must be recalibrated

    Args:
        p: First probability distribution (simplex)
        q: Second probability distribution (simplex)

    Returns:
        Fisher-Rao distance (≥ 0, max π/2)
    
    Examples:
        >>> p = np.array([0.5, 0.3, 0.2])
        >>> q = np.array([0.4, 0.4, 0.2])
        >>> d = fisher_rao_distance(p, q)
        >>> assert 0 <= d <= np.pi/2
    """
    # Ensure non-negative and normalized
    p = np.abs(p) + 1e-10
    p = p / p.sum()

    q = np.abs(q) + 1e-10
    q = q / q.sum()

    # Bhattacharyya coefficient
    bc = np.sum(np.sqrt(p * q))
    bc = np.clip(bc, 0, 1)

    # Direct Fisher-Rao (NO factor of 2)
    return float(np.arccos(bc))


def fisher_coord_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute Fisher-Rao distance between two basin coordinate vectors.

    Coordinates are assumed to be in SIMPLEX representation (probability distributions).
    This function normalizes inputs to simplex then computes Fisher-Rao distance.

    For simplex coordinates: d = arccos(Σ√(a_i * b_i))
    Range: [0, π/2]

    CHANGE FROM PREVIOUS VERSION:
    - Removed Hellinger sphere embedding
    - Removed factor of 2
    - New range: [0, π/2] (was [0, 2π])

    Args:
        a: First basin coordinate vector (will be normalized to simplex)
        b: Second basin coordinate vector (will be normalized to simplex)

    Returns:
        Fisher-Rao distance (0 to π/2)
    
    Examples:
        >>> a = np.array([0.5, 0.3, 0.2])
        >>> b = np.array([0.4, 0.4, 0.2])
        >>> d = fisher_coord_distance(a, b)
        >>> assert 0 <= d <= np.pi/2
    """
    # Normalize to simplex
    a_simplex = fisher_normalize(a)
    b_simplex = fisher_normalize(b)

    # Use direct Fisher-Rao distance
    return fisher_rao_distance(a_simplex, b_simplex)


def fisher_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute Fisher-Rao similarity between two basin coordinates.

    Formula: similarity = 1 - distance/(π/2) = 1 - 2*distance/π

    Range: [0, 1] where:
    - 1 = identical distributions
    - 0 = orthogonal distributions

    CHANGE FROM PREVIOUS VERSION:
    - Max distance is π/2 (was 2π with Hellinger)
    - Similarity formula adjusted accordingly

    Args:
        a: First basin coordinate vector
        b: Second basin coordinate vector

    Returns:
        Similarity score (0 to 1, higher is more similar)
    
    Examples:
        >>> a = np.array([0.5, 0.3, 0.2])
        >>> b = np.array([0.5, 0.3, 0.2])
        >>> s = fisher_similarity(a, b)
        >>> assert np.isclose(s, 1.0)  # Identical
    """
    distance = fisher_coord_distance(a, b)
    return 1.0 - (2.0 * distance / np.pi)


def normalize_basin_dimension(basin: np.ndarray, target_dim: int = 64) -> np.ndarray:
    """Project a basin vector to a target dimension.

    QIG-PURE: preserves geometric validity by projecting to simplex
    after padding/truncation.

    Args:
        basin: 1D basin coordinate vector
        target_dim: desired output dimension (default 64)

    Returns:
        1D basin vector of length target_dim on probability simplex.
    """
    b = np.asarray(basin, dtype=float)
    if b.ndim != 1:
        raise ValueError(f"basin must be 1D, got shape {b.shape}")

    current_dim = int(b.shape[0])
    if current_dim == int(target_dim):
        return fisher_normalize(b)

    if current_dim < int(target_dim):
        result = np.zeros(int(target_dim), dtype=float)
        result[:current_dim] = b
        return fisher_normalize(result)

    result = b[: int(target_dim)].copy()
    return fisher_normalize(result)


def geodesic_interpolation(
    start: np.ndarray,
    end: np.ndarray,
    t: float
) -> np.ndarray:
    """
    Spherical linear interpolation (slerp) along geodesic on simplex.

    Uses SLERP in sqrt-space which gives geodesic on the simplex.

    Args:
        start: Starting point (probability distribution)
        end: Ending point (probability distribution)
        t: Interpolation parameter (0 = start, 1 = end)

    Returns:
        Interpolated point along geodesic (probability distribution)
    
    Examples:
        >>> start = np.array([1.0, 0.0, 0.0])
        >>> end = np.array([0.0, 1.0, 0.0])
        >>> mid = geodesic_interpolation(start, end, 0.5)
        >>> assert np.isclose(mid.sum(), 1.0)
    """
    # Normalize to simplex
    p_start = fisher_normalize(start)
    p_end = fisher_normalize(end)
    
    # SLERP in sqrt space (geodesic on simplex)
    sqrt_start = np.sqrt(p_start)
    sqrt_end = np.sqrt(p_end)
    
    # Compute angle between sqrt vectors
    dot = np.clip(np.dot(sqrt_start, sqrt_end), -1.0, 1.0)
    omega = np.arccos(dot)

    if omega < 1e-6:
        return p_start.copy()

    sin_omega = np.sin(omega)
    a = np.sin((1 - t) * omega) / sin_omega
    b = np.sin(t * omega) / sin_omega

    sqrt_result = a * sqrt_start + b * sqrt_end
    
    # Square to get back to simplex
    result = sqrt_result ** 2
    return result / result.sum()


def estimate_manifold_curvature(
    points: np.ndarray,
    center: Optional[np.ndarray] = None
) -> float:
    """
    Estimate local curvature of the Fisher manifold from sample points.

    QIG-PURE: Uses Fisher-Rao distance on simplex (not Euclidean).

    Args:
        points: Array of shape (N, D) - N probability distributions
        center: Optional center point for curvature estimation

    Returns:
        Estimated curvature (κ)
    """
    if len(points) < 3:
        return 0.0

    if center is None:
        # Compute Fréchet mean (geometric mean on simplex)
        center = fisher_normalize(np.mean(points, axis=0))

    distances = []
    for point in points:
        d = fisher_coord_distance(center, point)
        distances.append(d)

    if not distances:
        return 0.0

    mean_dist = np.mean(distances)
    variance = np.var(distances)

    if mean_dist < 1e-6:
        return 0.0

    return float(variance / (mean_dist + 1e-10))


def basin_magnitude(basin: np.ndarray) -> float:
    """
    Compute a Fisher-Rao appropriate magnitude measure for basin coordinates.
    
    This measures how far the distribution is from uniform (maximum entropy).
    
    Args:
        basin: Basin coordinate vector
        
    Returns:
        Fisher-Rao distance from uniform distribution (≥ 0, max π/2)
    """
    p = fisher_normalize(basin)
    uniform = np.ones_like(p) / len(p)
    return fisher_rao_distance(p, uniform)


def basin_diversity(basin: np.ndarray) -> float:
    """
    Compute diversity (entropy) of basin distribution.
    
    This is an alternative magnitude measure that quantifies information content.
    Higher diversity = more uniform distribution = higher entropy.
    
    Args:
        basin: Basin coordinate vector
        
    Returns:
        Shannon entropy (≥ 0, higher = more diverse)
    """
    p = fisher_normalize(basin)
    p_safe = p + 1e-10
    return float(-np.sum(p_safe * np.log(p_safe)))


__all__ = [
    # Canonical contract (contracts.py) - THE source of truth
    'CANONICAL_SPACE',
    'BASIN_DIM',
    'NORM_TOLERANCE',
    'GeometricViolationError',
    'contracts_validate_basin',
    'validate_basin_detailed',
    'assert_invariants',
    'canon',
    'fisher_distance',
    'to_index_embedding',
    # Representation utilities (representation.py)
    'BasinRepresentation',
    'CANONICAL_REPRESENTATION',
    'to_sphere',
    'to_simplex',
    'validate_basin',
    'enforce_canonical',
    'sphere_project',
    'fisher_normalize',
    # Distance functions - CANONICAL IMPLEMENTATIONS
    'fisher_rao_distance',
    'fisher_coord_distance',
    'fisher_similarity',
    # Dimension and normalization utilities
    'normalize_basin_dimension',
    # Geodesic navigation
    'geodesic_interpolation',
    # Curvature and magnitude utilities
    'estimate_manifold_curvature',
    'basin_magnitude',
    'basin_diversity',
    # Purity mode enforcement (purity_mode.py)
    'QIG_PURITY_MODE',
    'QIGPurityViolationError',
    'check_purity_mode',
    'enforce_purity_startup',
    'install_purity_import_hook',
    'PurityImportBlocker',
]
