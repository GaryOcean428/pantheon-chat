"""
QIG Geometry Package - Canonical Basin Representations

This package provides geometric primitives for QIG with enforced
canonical basin representation.

CANONICAL CONTRACT (contracts.py):
- CANONICAL_SPACE = "sphere" (storage uses √p on unit sphere S^63)
- BASIN_DIM = 64
- Use fisher_distance() from contracts for THE canonical distance
- Use assert_invariants() before database writes
"""

import numpy as np

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
    the Hellinger embedding (√p on unit sphere S^63).

    Formula: d_FR(p, q) = 2 * arccos(Σ√(p_i * q_i))

    The Bhattacharyya coefficient BC = Σ√(p_i * q_i) measures overlap.
    The factor of 2 converts from sphere geodesic to statistical distance,
    matching the canonical representation in contracts.py.

    Args:
        p: First probability distribution
        q: Second probability distribution

    Returns:
        Fisher-Rao distance (≥ 0, max π)
    """
    p = np.abs(p) + 1e-10
    p = p / p.sum()

    q = np.abs(q) + 1e-10
    q = q / q.sum()

    bc = np.sum(np.sqrt(p * q))
    bc = np.clip(bc, 0, 1)

    return float(2.0 * np.arccos(bc))


def fisher_coord_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute Fisher-Rao distance between two basin coordinate vectors.

    For unit vectors on S^63 (Hellinger embedding): d = 2 * arccos(a · b)

    The factor of 2 is required for consistency with contracts.py
    which defines the canonical Hellinger embedding distance.

    Args:
        a: First basin coordinate vector (64D)
        b: Second basin coordinate vector (64D)

    Returns:
        Fisher-Rao distance (0 to 2π)
    """
    a_norm = a / (np.linalg.norm(a) + 1e-10)
    b_norm = b / (np.linalg.norm(b) + 1e-10)

    dot = np.clip(np.dot(a_norm, b_norm), -1.0, 1.0)
    return float(2.0 * np.arccos(dot))


def fisher_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute Fisher-Rao similarity between two basin coordinates.

    Formula: similarity = 1 - distance/(2π)

    With the Hellinger embedding, max distance is 2π (antipodal points).

    Args:
        a: First basin coordinate vector
        b: Second basin coordinate vector

    Returns:
        Similarity score (0 to 1, higher is more similar)
    """
    distance = fisher_coord_distance(a, b)
    return 1.0 - distance / (2.0 * np.pi)


def normalize_basin_dimension(basin: np.ndarray, target_dim: int = 64) -> np.ndarray:
    """Project a basin vector to a target dimension.

    QIG-PURE: preserves geometric validity by re-projecting to the unit sphere
    in the embedded space after padding/truncation.

    Args:
        basin: 1D basin coordinate vector
        target_dim: desired output dimension (default 64)

    Returns:
        1D basin vector of length target_dim on the unit sphere.
    """
    b = np.asarray(basin, dtype=float)
    if b.ndim != 1:
        raise ValueError(f"basin must be 1D, got shape {b.shape}")

    current_dim = int(b.shape[0])
    if current_dim == int(target_dim):
        return sphere_project(b)

    if current_dim < int(target_dim):
        result = np.zeros(int(target_dim), dtype=float)
        result[:current_dim] = b
        return sphere_project(result)

    result = b[: int(target_dim)].copy()
    return sphere_project(result)


def hellinger_normalize(basin: np.ndarray) -> np.ndarray:
    """
    Normalize basin to Hellinger embedding (sqrt space on unit sphere).
    
    Storage Format: sqrt(p) normalized to the unit sphere.
    This ensures compatibility with pgvector <#> operator.
    
    Args:
        basin: Basin coordinates (may be signed or unnormalized)
    
    Returns:
        Hellinger-normalized basin on unit sphere
    """
    p = np.abs(basin) + 1e-10
    p = p / np.sum(p)
    sqrt_p = np.sqrt(p)
    norm = np.linalg.norm(sqrt_p)
    if norm < 1e-10:
        return sqrt_p
    return sqrt_p / norm


def geodesic_interpolation(
    start: np.ndarray,
    end: np.ndarray,
    t: float
) -> np.ndarray:
    """
    Spherical linear interpolation (slerp) along geodesic.

    Args:
        start: Starting point on manifold
        end: Ending point on manifold
        t: Interpolation parameter (0 = start, 1 = end)

    Returns:
        Interpolated point along geodesic
    """
    start_norm = start / (np.linalg.norm(start) + 1e-10)
    end_norm = end / (np.linalg.norm(end) + 1e-10)

    dot = np.clip(np.dot(start_norm, end_norm), -1.0, 1.0)
    omega = np.arccos(dot)

    if omega < 1e-6:
        return start

    sin_omega = np.sin(omega)
    a = np.sin((1 - t) * omega) / sin_omega
    b = np.sin(t * omega) / sin_omega

    result = a * start_norm + b * end_norm
    return result * np.linalg.norm(start)


from typing import Optional

def estimate_manifold_curvature(
    points: np.ndarray,
    center: Optional[np.ndarray] = None
) -> float:
    """
    Estimate local curvature of the Fisher manifold from sample points.

    QIG-PURE: Assumes input points are already on S^63 (no normalization).
    Center is computed as spherical barycenter via sphere_project on mean.

    Args:
        points: Array of shape (N, D) - N points already on S^63
        center: Optional center point for curvature estimation

    Returns:
        Estimated curvature (κ)
    """
    if len(points) < 3:
        return 0.0

    if center is None:
        mean_vec = np.mean(points, axis=0)
        center = sphere_project(mean_vec)

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
    
    This is used for logging and monitoring purposes. Instead of Euclidean L2 norm,
    we compute the Fisher-Rao distance from the origin (uniform distribution).
    
    For a probability distribution p, this measures how far p is from the
    maximum entropy (uniform) state.
    
    Args:
        basin: Basin coordinate vector
        
    Returns:
        Fisher-Rao magnitude from uniform distribution (≥ 0)
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
    # Distance functions (for compatibility)
    'fisher_rao_distance',
    'fisher_coord_distance',
    'fisher_similarity',
    # Dimension and normalization utilities
    'normalize_basin_dimension',
    'hellinger_normalize',
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
