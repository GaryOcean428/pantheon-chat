"""
QIG Core Geometric Primitives

Pure geometric operations for QIG - NO external LLM dependencies.
All operations use Fisher-Rao geometry on the statistical manifold.

Note: torch is optional. All operations have numpy fallbacks.
"""

import numpy as np
from typing import List, Tuple, Optional, Union

# Try to import torch, but fall back to numpy-only implementation
TORCH_AVAILABLE = False
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore


def _to_numpy(arr) -> np.ndarray:
    """Convert input to numpy array, handling torch tensors if available."""
    if TORCH_AVAILABLE and torch is not None and isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    return np.asarray(arr, dtype=np.float64)


def _normalize_probability(p: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """Normalize to valid probability distribution."""
    p = np.abs(p) + epsilon
    return p / p.sum()


def fisher_rao_distance(p: Union[np.ndarray, List[float]], q: Union[np.ndarray, List[float]]) -> float:
    """
    Compute Fisher-Rao distance between two probability distributions.
    
    The Fisher-Rao distance is the geodesic distance on the statistical manifold,
    given by: d_FR(p, q) = 2 * arccos(sum(sqrt(p_i * q_i)))
    
    Args:
        p: First probability distribution (basin coordinates)
        q: Second probability distribution (basin coordinates)
        
    Returns:
        Fisher-Rao distance between p and q
    """
    p_arr = _to_numpy(p)
    q_arr = _to_numpy(q)
    
    # Normalize to probability simplex
    p_norm = _normalize_probability(p_arr)
    q_norm = _normalize_probability(q_arr)
    
    # Bhattacharyya coefficient
    bc = np.sum(np.sqrt(p_norm * q_norm))
    bc = np.clip(bc, 0.0, 1.0)
    
    # Fisher-Rao distance
    return float(2.0 * np.arccos(bc))


def fisher_metric_tensor(p: Union[np.ndarray, List[float]]) -> np.ndarray:
    """
    Compute the Fisher information metric tensor at point p.
    
    The Fisher metric is: g_ij = E[(d log p / d theta_i)(d log p / d theta_j)]
    For the probability simplex: g_ij = delta_ij / p_i
    
    Args:
        p: Point on the probability simplex (basin coordinates)
        
    Returns:
        Fisher information metric tensor (diagonal matrix for simplex)
    """
    p_arr = _to_numpy(p)
    p_norm = _normalize_probability(p_arr)
    
    # Fisher metric on probability simplex is diagonal: g_ii = 1/p_i
    return np.diag(1.0 / p_norm)


def geodesic_interpolate(
    p: Union[np.ndarray, List[float]], 
    q: Union[np.ndarray, List[float]], 
    t: float
) -> np.ndarray:
    """
    Interpolate along the geodesic from p to q at parameter t.
    
    Uses the spherical geodesic on the probability simplex (via sqrt transform).
    At t=0 returns p, at t=1 returns q.
    
    Args:
        p: Starting point (probability distribution)
        q: Ending point (probability distribution)
        t: Interpolation parameter in [0, 1]
        
    Returns:
        Point on geodesic at parameter t
    """
    p_arr = _to_numpy(p)
    q_arr = _to_numpy(q)
    
    p_norm = _normalize_probability(p_arr)
    q_norm = _normalize_probability(q_arr)
    
    # Transform to sphere (sqrt coordinates)
    sqrt_p = np.sqrt(p_norm)
    sqrt_q = np.sqrt(q_norm)
    
    # Compute angle between points
    cos_theta = np.clip(np.dot(sqrt_p, sqrt_q), -1.0, 1.0)
    theta = np.arccos(cos_theta)
    
    if theta < 1e-10:
        # Points are essentially the same
        return p_norm
    
    # Spherical linear interpolation (slerp)
    sin_theta = np.sin(theta)
    sqrt_interp = (np.sin((1 - t) * theta) / sin_theta) * sqrt_p + \
                  (np.sin(t * theta) / sin_theta) * sqrt_q
    
    # Transform back to probability simplex
    result = sqrt_interp ** 2
    return result / result.sum()


def basin_to_probability(basin: Union[np.ndarray, List[float]], epsilon: float = 1e-10) -> np.ndarray:
    """
    Convert basin coordinates to a valid probability distribution.
    
    Args:
        basin: Raw basin coordinates
        epsilon: Small value to ensure positivity
        
    Returns:
        Valid probability distribution (sums to 1, all positive)
    """
    arr = _to_numpy(basin)
    return _normalize_probability(arr, epsilon)


def compute_geodesic_path(
    start: Union[np.ndarray, List[float]],
    end: Union[np.ndarray, List[float]],
    num_points: int = 10
) -> List[np.ndarray]:
    """
    Compute a discretized geodesic path from start to end.
    
    Args:
        start: Starting probability distribution
        end: Ending probability distribution
        num_points: Number of points along the path
        
    Returns:
        List of points along the geodesic
    """
    return [geodesic_interpolate(start, end, t) for t in np.linspace(0, 1, num_points)]


def parallel_transport(
    v: Union[np.ndarray, List[float]],
    p: Union[np.ndarray, List[float]],
    q: Union[np.ndarray, List[float]]
) -> np.ndarray:
    """
    Parallel transport a tangent vector v from p to q along the geodesic.
    
    Args:
        v: Tangent vector at p
        p: Starting point
        q: Ending point
        
    Returns:
        Transported vector at q
    """
    v_arr = _to_numpy(v)
    p_arr = _to_numpy(p)
    q_arr = _to_numpy(q)
    
    p_norm = _normalize_probability(p_arr)
    q_norm = _normalize_probability(q_arr)
    
    # Transform to sphere coordinates
    sqrt_p = np.sqrt(p_norm)
    sqrt_q = np.sqrt(q_norm)
    
    # Project v onto tangent space at p
    v_tangent = v_arr - np.dot(v_arr, sqrt_p) * sqrt_p
    
    # Compute rotation axis
    cos_theta = np.clip(np.dot(sqrt_p, sqrt_q), -1.0, 1.0)
    
    if abs(cos_theta) > 1 - 1e-10:
        # Points are nearly identical or antipodal
        return v_tangent
    
    # Gram-Schmidt to get orthonormal basis
    axis = sqrt_q - cos_theta * sqrt_p
    axis = axis / (np.linalg.norm(axis) + 1e-10)
    
    # Rodrigues rotation formula
    theta = np.arccos(cos_theta)
    v_transported = v_tangent * np.cos(theta) + \
                    np.cross(axis, v_tangent) * np.sin(theta) + \
                    axis * np.dot(axis, v_tangent) * (1 - np.cos(theta))
    
    return v_transported
