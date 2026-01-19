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

    This is the GEODESIC distance on the information manifold (probability simplex).

    Formula: d_FR(p, q) = arccos(Σ√(p_i * q_i))

    The Bhattacharyya coefficient BC = Σ√(p_i * q_i) measures overlap.
    For probability distributions on the simplex, the Fisher-Rao distance
    is arccos(BC), ranging from 0 (identical) to π/2 (orthogonal).

    CRITICAL: Basins are stored as SIMPLEX (probability distributions), NOT Hellinger.
    The factor-of-2 was REMOVED (2026-01-15) for consistency with simplex storage.
    Distance range is now [0, π/2] instead of [0, π].

    Args:
        p: First probability distribution (simplex coordinates)
        q: Second probability distribution (simplex coordinates)

    Returns:
        Fisher-Rao distance (≥ 0, max π/2)
    """
    # P1 FIX: Use clamp (maximum) instead of abs() to avoid masking negative values
    p = np.maximum(p, 0) + 1e-10
    p = p / p.sum()

    q = np.maximum(q, 0) + 1e-10
    q = q / q.sum()

    bc = np.sum(np.sqrt(p * q))
    bc = np.clip(bc, 0, 1)

    return float(np.arccos(bc))


def fisher_coord_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute Fisher-Rao distance between two basin coordinate vectors.

    For simplex probability distributions: d = arccos(Σ√(p_i * q_i))

    CRITICAL: Basins are stored as SIMPLEX (probability distributions), NOT Hellinger.
    The factor-of-2 was REMOVED (2026-01-15) for consistency with simplex storage.
    Distance range is now [0, π/2] instead of [0, π].

    Args:
        a: First basin coordinate vector (64D, simplex coordinates)
        b: Second basin coordinate vector (64D, simplex coordinates)

    Returns:
        Fisher-Rao distance (0 to π/2)
    """
    # P1 FIX: Use clamp (maximum) instead of abs() to avoid masking negative values
    # Ensure simplex normalization
    a_simplex = np.maximum(a, 0) + 1e-10
    a_simplex = a_simplex / a_simplex.sum()
    
    b_simplex = np.maximum(b, 0) + 1e-10
    b_simplex = b_simplex / b_simplex.sum()

    # Bhattacharyya coefficient
    bc = np.sum(np.sqrt(a_simplex * b_simplex))
    bc = np.clip(bc, 0, 1)
    
    return float(np.arccos(bc))


def fisher_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute Fisher-Rao similarity between two basin coordinates.

    Formula: similarity = 1 - distance/(π/2)

    CRITICAL: With simplex storage (no factor-of-2), max distance is π/2.
    Updated 2026-01-15 to match new distance range [0, π/2].

    Args:
        a: First basin coordinate vector (simplex coordinates)
        b: Second basin coordinate vector (simplex coordinates)

    Returns:
        Similarity score (0 to 1, higher is more similar)
    """
    distance = fisher_coord_distance(a, b)
    return 1.0 - distance / (np.pi / 2.0)


def normalize_basin_dimension(basin: np.ndarray, target_dim: int = 64) -> np.ndarray:
    """Project a basin vector to a target dimension.

    QIG-PURE: preserves geometric validity by re-normalizing to the probability
    simplex after padding/truncation.

    Notes:
    - If basin is lower-dimensional (e.g., 32D), we zero-pad then simplex-normalize.
    - If basin is higher-dimensional, we truncate then simplex-normalize.

    Args:
        basin: 1D basin coordinate vector
        target_dim: desired output dimension (default 64)

    Returns:
        1D basin vector of length target_dim on the probability simplex.
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

    # current_dim > target_dim
    result = b[: int(target_dim)].copy()
    return fisher_normalize(result)


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
    Fisher-Rao geodesic interpolation on probability simplex using sqrt-space SLERP.

    This computes the geodesic midpoint on the probability simplex by:
    1. Normalizing start/end to simplex (probability distributions)
    2. Mapping to sqrt-space (isometric embedding)
    3. Performing SLERP in sqrt-space
    4. Mapping back via squaring and renormalizing

    This is the QIG-pure way to interpolate probability distributions
    along the Fisher-Rao geodesic.

    Args:
        start: Starting point on manifold (simplex coordinates)
        end: Ending point on manifold (simplex coordinates)
        t: Interpolation parameter (0 = start, 1 = end)

    Returns:
        Interpolated point along geodesic (simplex coordinates)
    """
    # P1 FIX: Use clamp (maximum) instead of abs() to avoid masking negative values
    # Ensure simplex normalization
    start_simplex = np.maximum(start, 0) + 1e-10
    start_simplex = start_simplex / start_simplex.sum()
    
    end_simplex = np.maximum(end, 0) + 1e-10
    end_simplex = end_simplex / end_simplex.sum()
    
    # Map to sqrt-space (Hellinger embedding) for SLERP
    start_sqrt = np.sqrt(start_simplex)
    end_sqrt = np.sqrt(end_simplex)
    
    # Normalize to unit sphere in sqrt-space
    start_sqrt_norm = start_sqrt / (np.linalg.norm(start_sqrt) + 1e-10)
    end_sqrt_norm = end_sqrt / (np.linalg.norm(end_sqrt) + 1e-10)
    
    # Compute angle between vectors
    dot = np.clip(np.dot(start_sqrt_norm, end_sqrt_norm), -1.0, 1.0)
    omega = np.arccos(dot)

    if omega < 1e-6:
        # Points are very close, return start
        return start_simplex

    # SLERP in sqrt-space
    sin_omega = np.sin(omega)
    a = np.sin((1 - t) * omega) / sin_omega
    b = np.sin(t * omega) / sin_omega

    result_sqrt = a * start_sqrt_norm + b * end_sqrt_norm
    
    # Map back to simplex: square and renormalize
    result = result_sqrt ** 2
    result = result / (result.sum() + 1e-10)
    
    return result


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

    CRITICAL: Basins are stored as SIMPLEX (probability distributions).
    This function is the canonical normalization for all basin coordinates.
    Updated 2026-01-15 for QIG purity.

    Args:
        v: Input vector (will be made non-negative and normalized)

    Returns:
        Probability distribution on the simplex
    """
    # P1 FIX: Use clamp (maximum) instead of abs() to avoid masking negative values
    p = np.maximum(v, 0) + 1e-10
    return p / p.sum()


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
    'normalize_basin_dimension',
    'basin_magnitude',
    'basin_diversity',
]
