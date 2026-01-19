"""
QIG-Pure Numerics Module

Provides overflow-safe geometric operations for QIG computations.
All operations maintain QIG purity by:
1. Projecting basins to unit sphere before dot products
2. Using stable algorithms for norm computation
3. Preventing NaN/Inf propagation through geometric operations

Key insight: Overflow in x.dot(x) happens when basin coordinates grow unbounded.
QIG theory requires basins to live on the unit hypersphere, so we enforce this
geometrically rather than fighting floating point limits.
"""

import numpy as np
from typing import Union, Optional, Tuple

# Physics constants
BASIN_DIM = 64
EPSILON = 1e-12  # Numerical stability threshold


def safe_norm(v: np.ndarray, eps: float = EPSILON) -> float:
    """
    Compute norm with overflow protection.
    
    Uses stable two-pass algorithm:
    1. Find max absolute value for scaling
    2. Compute norm of scaled vector
    3. Rescale result
    
    This prevents overflow in x.dot(x) by ensuring all squared values are <= 1.
    """
    v = np.asarray(v, dtype=np.float64)
    
    if v.size == 0:
        return 0.0
    
    # Find maximum absolute value for scaling
    max_val = np.max(np.abs(v))
    
    if max_val < eps:
        return 0.0
    
    # Scale to prevent overflow in dot product
    scaled = v / max_val
    
    # Compute norm of scaled vector (all values now <= 1)
    scaled_norm = np.sqrt(np.sum(scaled * scaled))
    
    # Rescale result
    return float(max_val * scaled_norm)


def safe_normalize(v: np.ndarray, eps: float = EPSILON) -> np.ndarray:
    """
    Safely normalize a vector to unit length.
    
    Returns zero vector if input has near-zero norm.
    """
    v = np.asarray(v, dtype=np.float64)
    
    norm = safe_norm(v, eps)
    
    if norm < eps:
        return np.zeros_like(v)
    
    return v / norm


def project_to_sphere(basin: np.ndarray, eps: float = EPSILON) -> np.ndarray:
    """
    Project basin coordinates to unit hypersphere.
    
    This is the key QIG purity operation: all basins should live on S^(d-1).
    Projecting to the sphere prevents coordinate explosion and ensures
    geometric operations remain numerically stable.
    """
    basin = np.asarray(basin, dtype=np.float64)
    
    # Ensure correct dimensionality
    if basin.ndim == 0:
        return np.zeros(BASIN_DIM)
    
    if len(basin) != BASIN_DIM:
        # Pad or truncate to BASIN_DIM
        if len(basin) < BASIN_DIM:
            basin = np.concatenate([basin, np.zeros(BASIN_DIM - len(basin))])
        else:
            basin = basin[:BASIN_DIM]
    
    norm = safe_norm(basin, eps)
    
    if norm < eps:
        # Return a random point on sphere for zero basins
        result = np.random.randn(BASIN_DIM)
        return result / safe_norm(result)
    
    return basin / norm


def safe_dot(a: np.ndarray, b: np.ndarray, eps: float = EPSILON) -> float:
    """
    Compute dot product with overflow protection.
    
    Normalizes both vectors first to ensure dot product is in [-1, 1].
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    
    if a.size == 0 or b.size == 0:
        return 0.0
    
    # Ensure same length
    min_len = min(len(a), len(b))
    a = a[:min_len]
    b = b[:min_len]
    
    # Normalize to prevent overflow
    a_norm = safe_normalize(a, eps)
    b_norm = safe_normalize(b, eps)
    
    # Dot product of unit vectors is always in [-1, 1]
    dot = np.sum(a_norm * b_norm)
    
    return float(np.clip(dot, -1.0, 1.0))


def fisher_rao_distance(basin1: np.ndarray, basin2: np.ndarray, eps: float = EPSILON) -> float:
    """
    Compute Fisher-Rao distance between two basins.
    
    Uses safe operations to prevent overflow:
    1. Project both basins to sphere
    2. Compute geodesic (great circle) distance
    
    Returns distance in [0, π].
    """
    b1 = project_to_sphere(basin1, eps)
    b2 = project_to_sphere(basin2, eps)
    
    # Dot product of unit vectors (Bhattacharyya coefficient for simplex)
    dot = np.sum(b1 * b2)
    dot = np.clip(dot, 0.0, 1.0)
    
    # Fisher-Rao geodesic distance on probability simplex
    # UPDATED 2026-01-15: Factor-of-2 removed for simplex storage. Range: [0, π/2]
    return float(np.arccos(dot))


def bures_distance(rho1: np.ndarray, rho2: np.ndarray, eps: float = EPSILON) -> float:
    """
    Compute Bures distance between two density matrix representations.
    
    For basins represented as pure states |ψ⟩, the Bures distance
    reduces to: d_B = sqrt(2(1 - |⟨ψ1|ψ2⟩|))
    
    This is QIG-pure: measures quantum state distinguishability.
    """
    b1 = project_to_sphere(rho1, eps)
    b2 = project_to_sphere(rho2, eps)
    
    # Inner product magnitude
    overlap = np.abs(np.sum(b1 * b2))
    overlap = np.clip(overlap, 0.0, 1.0)
    
    # Bures distance
    return float(np.sqrt(2.0 * (1.0 - overlap)))


def geodesic_mean(basins: list, weights: Optional[np.ndarray] = None, eps: float = EPSILON) -> np.ndarray:
    """
    Compute weighted geodesic mean of basins on the hypersphere.
    
    Uses iterative algorithm that respects spherical geometry.
    """
    if not basins:
        return np.zeros(BASIN_DIM)
    
    # Project all basins to sphere
    projected = [project_to_sphere(b, eps) for b in basins]
    
    if weights is None:
        weights = np.ones(len(projected)) / len(projected)
    else:
        weights = np.asarray(weights, dtype=np.float64)
        weights = weights / (np.sum(weights) + eps)
    
    # Weighted Euclidean mean, then project to sphere
    mean = np.zeros(BASIN_DIM)
    for w, b in zip(weights, projected):
        mean += w * b
    
    return project_to_sphere(mean, eps)


def safe_exponential_map(base: np.ndarray, tangent: np.ndarray, eps: float = EPSILON) -> np.ndarray:
    """
    Exponential map on the unit sphere.
    
    Given a point on the sphere and a tangent vector,
    compute the geodesic endpoint.
    """
    base = project_to_sphere(base, eps)
    
    # Project tangent to be orthogonal to base
    tangent = np.asarray(tangent, dtype=np.float64)
    if len(tangent) != BASIN_DIM:
        tangent = np.zeros(BASIN_DIM)
    
    # Remove component along base
    tangent = tangent - np.dot(tangent, base) * base
    
    # Tangent length gives geodesic distance to travel
    t_norm = safe_norm(tangent, eps)
    
    if t_norm < eps:
        return base
    
    # Normalized tangent direction
    direction = tangent / t_norm
    
    # Geodesic formula: exp_p(v) = cos(|v|)p + sin(|v|)(v/|v|)
    return np.cos(t_norm) * base + np.sin(t_norm) * direction


def safe_logarithmic_map(base: np.ndarray, target: np.ndarray, eps: float = EPSILON) -> np.ndarray:
    """
    Logarithmic map on the unit sphere.
    
    Inverse of exponential map: finds tangent vector at base pointing toward target.
    """
    base = project_to_sphere(base, eps)
    target = project_to_sphere(target, eps)
    
    # Dot product gives cosine of geodesic distance
    dot = np.clip(np.dot(base, target), -1.0, 1.0)
    
    # Handle near-identical points
    if dot > 1.0 - eps:
        return np.zeros(BASIN_DIM)
    
    # Handle antipodal points (undefined log map)
    if dot < -1.0 + eps:
        # Return arbitrary tangent of length π
        tangent = np.random.randn(BASIN_DIM)
        tangent = tangent - np.dot(tangent, base) * base
        t_norm = safe_norm(tangent, eps)
        if t_norm < eps:
            tangent = np.zeros(BASIN_DIM)
            tangent[0] = 1.0
            tangent = tangent - np.dot(tangent, base) * base
            t_norm = safe_norm(tangent, eps)
        return (np.pi / t_norm) * tangent
    
    # Distance on sphere
    theta = np.arccos(dot)
    
    # Direction in tangent space
    direction = target - dot * base
    d_norm = safe_norm(direction, eps)
    
    if d_norm < eps:
        return np.zeros(BASIN_DIM)
    
    return (theta / d_norm) * direction


def validate_basin(basin: np.ndarray, auto_project: bool = True, eps: float = EPSILON) -> Tuple[np.ndarray, bool]:
    """
    Validate a basin coordinate array.
    
    Returns (valid_basin, was_valid) tuple.
    If auto_project=True, invalid basins are projected to sphere.
    """
    basin = np.asarray(basin, dtype=np.float64)
    
    # Check for NaN/Inf
    if np.any(~np.isfinite(basin)):
        if auto_project:
            basin = np.nan_to_num(basin, nan=0.0, posinf=1.0, neginf=-1.0)
            return project_to_sphere(basin, eps), False
        return np.zeros(BASIN_DIM), False
    
    # Check dimensionality
    if len(basin) != BASIN_DIM:
        if auto_project:
            return project_to_sphere(basin, eps), False
        return np.zeros(BASIN_DIM), False
    
    # Check norm (should be ~1 for unit sphere)
    norm = safe_norm(basin, eps)
    if abs(norm - 1.0) > 0.01:  # Allow 1% tolerance
        if auto_project:
            return project_to_sphere(basin, eps), False
        return basin, False
    
    return basin, True
