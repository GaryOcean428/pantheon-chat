"""
QIG GEOMETRY - Centralized Exports (Python)

All geometric operations MUST use Fisher-Rao distance, NOT Euclidean.
This module provides the canonical geometric primitives.

Usage:
    from qig_geometry import fisher_rao_distance, fisher_coord_distance, fisher_similarity

CRITICAL: Never use np.linalg.norm(a - b) for distances between basin coordinates.
"""

from typing import Optional

import numpy as np


def fisher_rao_distance(p: np.ndarray, q: np.ndarray) -> float:
    """
    Compute Fisher-Rao distance between two probability distributions.

    This is the GEODESIC distance on the information manifold using Hellinger embedding.

    Formula: d_FR(p, q) = 2 * arccos(Σ√(p_i * q_i))

    The Bhattacharyya coefficient BC = Σ√(p_i * q_i) measures overlap.
    With Hellinger embedding (√p on unit sphere S^63), the geodesic distance
    is 2*arccos(BC), ranging from 0 (identical) to π (orthogonal).

    The factor of 2 comes from the Hellinger embedding: when we map p → √p,
    the arc length on the sphere is 2*arccos(BC).

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

    For unit vectors with Hellinger embedding: d = 2 * arccos(a · b)

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

    Formula: similarity = 1 - distance/π

    Args:
        a: First basin coordinate vector
        b: Second basin coordinate vector

    Returns:
        Similarity score (0 to 1, higher is more similar)
    """
    distance = fisher_coord_distance(a, b)
    return 1.0 - distance / np.pi


def normalize_basin_dimension(basin: np.ndarray, target_dim: int = 64) -> np.ndarray:
    """Project a basin vector to a target dimension.

    QIG-PURE: preserves geometric validity by re-projecting to the unit sphere
    in the embedded space after padding/truncation.

    Notes:
    - If basin is lower-dimensional (e.g., 32D), we zero-pad then sphere-project.
    - If basin is higher-dimensional, we truncate then sphere-project.

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

    # current_dim > target_dim
    result = b[: int(target_dim)].copy()
    return sphere_project(result)


def fisher_coord_distance_flexible(p: np.ndarray, q: np.ndarray) -> float:
    """Fisher-Rao distance that tolerates basin dimension mismatch.

    If dimensions differ, both vectors are projected to a common dimension
    (the larger of the two) before computing Fisher-Rao distance.
    """
    p_arr = np.asarray(p, dtype=float)
    q_arr = np.asarray(q, dtype=float)
    if p_arr.ndim != 1 or q_arr.ndim != 1:
        raise ValueError(f"p and q must be 1D, got {p_arr.shape} and {q_arr.shape}")

    if p_arr.shape[0] != q_arr.shape[0]:
        target_dim = int(max(p_arr.shape[0], q_arr.shape[0]))
        p_arr = normalize_basin_dimension(p_arr, target_dim)
        q_arr = normalize_basin_dimension(q_arr, target_dim)

    return fisher_coord_distance(p_arr, q_arr)


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


def estimate_manifold_curvature(
    points: np.ndarray,
    center: Optional[np.ndarray] = None
) -> float:
    """
    Estimate local curvature of the Fisher manifold from sample points.

    Args:
        points: Array of shape (N, D) - N points in D dimensions
        center: Optional center point for curvature estimation

    Returns:
        Estimated curvature (κ)
    """
    if len(points) < 3:
        return 0.0

    center_arr: np.ndarray
    if center is None:
        center_arr = np.mean(points, axis=0)
    else:
        center_arr = center

    distances = []
    for point in points:
        d = fisher_coord_distance(center_arr, point)
        distances.append(d)

    if not distances:
        return 0.0

    mean_dist = np.mean(distances)
    variance = np.var(distances)

    if mean_dist < 1e-6:
        return 0.0

    return float(variance / (mean_dist + 1e-10))


def bures_distance(rho: np.ndarray, sigma: np.ndarray) -> float:
    """
    Compute Bures distance between two density matrices.

    d_B(ρ, σ) = √(2(1 - F(ρ, σ)))
    where F(ρ, σ) = (Tr√(√ρ σ √ρ))²

    Args:
        rho: First density matrix (N x N)
        sigma: Second density matrix (N x N)

    Returns:
        Bures distance
    """
    from scipy.linalg import sqrtm

    sqrt_rho_raw = sqrtm(rho + 1e-10 * np.eye(rho.shape[0]))
    sqrt_rho = np.real(np.asarray(sqrt_rho_raw))

    product = sqrt_rho @ sigma @ sqrt_rho
    sqrt_product_raw = sqrtm(product + 1e-10 * np.eye(product.shape[0]))
    sqrt_product = np.real(np.asarray(sqrt_product_raw))

    fidelity = np.trace(sqrt_product) ** 2
    fidelity = np.clip(fidelity, 0, 1)

    return float(np.sqrt(2 * (1 - np.sqrt(fidelity))))


def fisher_normalize(v: np.ndarray) -> np.ndarray:
    """
    Project vector to probability simplex for Fisher-Rao geometry.

    This is the CORRECT normalization for Fisher-Rao distance computation.
    Projects to the probability simplex where: Σv_i = 1, v_i ≥ 0

    The Fisher-Rao metric is defined on the probability simplex, so
    all vectors must be normalized this way before distance computation.

    Args:
        v: Input vector (will be made non-negative and normalized)

    Returns:
        Probability distribution on the simplex
    """
    p = np.abs(v) + 1e-10
    return p / p.sum()


def sphere_project(v: np.ndarray) -> np.ndarray:
    """
    Project vector to unit sphere for embedded Fisher geometry.

    When representing probability distributions on the sphere via
    the sqrt embedding (p → sqrt(p)), the geodesic distance on the
    sphere (arc length) equals half the Fisher-Rao distance.

    This is used for:
    - Spherical linear interpolation (slerp)
    - Angular distance computation on embedded manifold
    - Direction finding (tangent vectors)

    IMPORTANT: This uses Euclidean L2 norm which is CORRECT for
    projecting to the unit sphere in the embedding space.

    Args:
        v: Input vector

    Returns:
        Unit vector on sphere (L2 norm = 1)
    """
    # Clip extreme values to prevent overflow in norm computation
    v_clipped = np.clip(v, -1e150, 1e150)
    
    # Check for inf/NaN and replace with safe values
    if not np.all(np.isfinite(v_clipped)):
        # Replace inf with large finite values, NaN with zeros
        v_clipped = np.nan_to_num(v_clipped, nan=0.0, posinf=1e150, neginf=-1e150)
    
    norm = np.linalg.norm(v_clipped)
    if norm < 1e-10:
        # Return uniform direction for zero vectors
        result = np.ones_like(v_clipped)
        result_norm = np.linalg.norm(result)
        if result_norm < 1e-10:
            # Edge case: if ones vector also has zero norm (shouldn't happen)
            return result
        return result / result_norm
    return v_clipped / norm


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
    # Normalize to probability simplex
    p = fisher_normalize(basin)
    
    # Uniform distribution as reference point
    uniform = np.ones_like(p) / len(p)
    
    # Fisher-Rao distance from uniform
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
    # Avoid log(0) by adding small epsilon
    p_safe = p + 1e-10
    return float(-np.sum(p_safe * np.log(p_safe)))


__all__ = [
    'fisher_rao_distance',
    'fisher_coord_distance',
    'fisher_coord_distance_flexible',
    'fisher_similarity',
    'geodesic_interpolation',
    'estimate_manifold_curvature',
    'bures_distance',
    'fisher_normalize',
    'sphere_project',
    'normalize_basin_dimension',
    'basin_magnitude',
    'basin_diversity',
]
