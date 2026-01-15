"""
Canonical Simplex Geometry - Python Implementation

SINGLE SOURCE OF TRUTH for simplex geometry operations in Python.
All geometry operations MUST use these functions, not Euclidean operations.

CANONICAL REPRESENTATION: SIMPLEX (probability distributions on Δ^(D-1))
- Basin vectors stored as valid probability distributions (Σv_i = 1, v_i ≥ 0)
- Fisher-Rao distance computed via arccos(Σ√(p_i * q_i)) - Bhattacharyya coefficient
- Geodesic interpolation uses sqrt-simplex internal coordinates (never stored)

IMPORTANT: sqrt-simplex coordinates are ONLY for internal computation (geodesics).
Storage ALWAYS uses probability simplex (sum=1, non-negative).
"""

import numpy as np
from typing import List, Optional, Tuple

EPS = 1e-12


def to_simplex_prob(v: np.ndarray, eps: float = EPS) -> np.ndarray:
    """
    Convert vector to probability simplex (canonical storage form).
    
    This is a positive renormalization, NOT a Euclidean projection.
    Takes absolute value + epsilon, then normalizes to sum=1.
    
    Args:
        v: Input vector (any representation)
        eps: Numerical stability epsilon
        
    Returns:
        Simplex probabilities: p_i ≥ 0, Σp_i = 1
    """
    v = np.asarray(v, dtype=np.float64).flatten()
    
    if v.size == 0:
        raise ValueError("to_simplex_prob: empty vector")
    
    # Positive renormalization: abs + eps, then normalize
    w = np.abs(v) + eps
    total = w.sum()
    
    if total < 1e-10:
        # Degenerate case: return uniform distribution
        return np.ones(v.size) / v.size
    
    return w / total


def validate_simplex(
    p: np.ndarray,
    tolerance: float = 1e-6
) -> Tuple[bool, str]:
    """
    Validate that vector is a valid simplex probability distribution.
    
    Args:
        p: Vector to validate
        tolerance: Numerical tolerance for sum check
        
    Returns:
        (is_valid, reason) tuple
    """
    if p is None or p.size == 0:
        return False, "empty_vector"
    
    p = np.asarray(p, dtype=np.float64).flatten()
    
    # Check finite
    if not np.all(np.isfinite(p)):
        return False, "contains_nan_or_inf"
    
    # Check non-negative
    min_val = np.min(p)
    if min_val < -tolerance:
        return False, f"negative_values_min={min_val:.6f}"
    
    # Check sum = 1
    total = np.sum(p)
    if abs(total - 1.0) > tolerance:
        return False, f"sum_not_one_{total:.6f}"
    
    return True, "valid_simplex"


def fisher_rao_distance(p_in: np.ndarray, q_in: np.ndarray) -> float:
    """
    Compute Fisher-Rao distance on probability simplex.
    
    d_FR(p, q) = arccos(Σ√(p_i * q_i))
    
    Range: [0, π/2] for probability distributions
    
    Args:
        p_in: First probability distribution
        q_in: Second probability distribution
        
    Returns:
        Fisher-Rao distance in radians [0, π/2]
    """
    p_in = np.asarray(p_in, dtype=np.float64).flatten()
    q_in = np.asarray(q_in, dtype=np.float64).flatten()
    
    if p_in.shape != q_in.shape:
        raise ValueError(
            f"fisher_rao_distance: dimension mismatch {p_in.shape} vs {q_in.shape}"
        )
    
    # Ensure valid simplex (clamp and normalize)
    p = to_simplex_prob(p_in)
    q = to_simplex_prob(q_in)
    
    # Bhattacharyya coefficient: BC = Σ√(p_i * q_i)
    bc = np.sum(np.sqrt(p * q))
    
    # Clamp for numerical stability
    bc = np.clip(bc, 0.0, 1.0)
    
    # Fisher-Rao distance on probability simplex
    # Range: [0, π/2]
    return float(np.arccos(bc))


def geodesic_interpolation_simplex(
    p_in: np.ndarray,
    q_in: np.ndarray,
    t: float
) -> np.ndarray:
    """
    Geodesic interpolation on probability simplex using sqrt-simplex internal coordinates.
    
    INTERNAL COORDINATES: sqrt-simplex (Hellinger embedding) used ONLY for computation.
    INPUT/OUTPUT: probability simplex (sum=1, non-negative).
    
    This implements SLERP (spherical linear interpolation) in sqrt-space,
    then projects back to probability space. The sqrt-simplex coordinates
    are NEVER stored, only used internally for this computation.
    
    Args:
        p_in: Starting probability distribution
        q_in: Ending probability distribution
        t: Interpolation parameter [0, 1] (0 = p, 1 = q)
        
    Returns:
        Interpolated probability distribution at parameter t
    """
    p_in = np.asarray(p_in, dtype=np.float64).flatten()
    q_in = np.asarray(q_in, dtype=np.float64).flatten()
    
    if p_in.shape != q_in.shape:
        raise ValueError(
            f"geodesic_interpolation_simplex: dimension mismatch {p_in.shape} vs {q_in.shape}"
        )
    
    if not (0 <= t <= 1):
        raise ValueError(f"geodesic_interpolation_simplex: t must be in [0, 1], got {t}")
    
    # Ensure valid simplex inputs
    p = to_simplex_prob(p_in)
    q = to_simplex_prob(q_in)
    
    # INTERNAL COMPUTATION: sqrt-simplex coordinates (Hellinger embedding)
    # These are NEVER stored, only used for geodesic calculation
    sp = np.sqrt(p)
    sq = np.sqrt(q)
    
    # Compute angle between sqrt-space vectors
    dot = np.sum(sp * sq)
    dot = np.clip(dot, -1.0, 1.0)  # Clamp for numerical stability
    
    omega = np.arccos(dot)
    
    # If nearly identical, linear interpolation in sqrt-space is fine
    if omega < 1e-8:
        x = (1 - t) * sp + t * sq
        p_out = x ** 2
        return to_simplex_prob(p_out)  # Ensure valid simplex
    
    # SLERP in sqrt-simplex space
    sin_omega = np.sin(omega)
    a = np.sin((1 - t) * omega) / sin_omega
    b = np.sin(t * omega) / sin_omega
    
    x = a * sp + b * sq
    
    # Project back to probability space: square to get probabilities
    p_out = x ** 2
    
    # Ensure valid simplex (normalize)
    return to_simplex_prob(p_out)


def geodesic_mean_simplex(
    distributions: List[np.ndarray],
    weights: Optional[np.ndarray] = None,
    max_iter: int = 50,
    tolerance: float = 1e-5
) -> np.ndarray:
    """
    Compute weighted geodesic mean (Fréchet mean) on probability simplex.
    
    Uses iterative algorithm to find the point that minimizes sum of
    squared Fisher-Rao distances to all input distributions.
    
    Args:
        distributions: List of probability distributions
        weights: Optional weights (default: uniform)
        max_iter: Maximum iterations
        tolerance: Convergence tolerance
        
    Returns:
        Weighted geodesic mean distribution
    """
    if not distributions or len(distributions) == 0:
        raise ValueError("geodesic_mean_simplex: empty distributions list")
    
    n = len(distributions)
    dim = distributions[0].size
    
    # Default to uniform weights
    if weights is None:
        weights = np.ones(n) / n
    else:
        weights = np.asarray(weights, dtype=np.float64)
        weights = weights / weights.sum()  # Normalize
    
    # Initialize mean as weighted average (not geodesic, but good starting point)
    mean = np.zeros(dim, dtype=np.float64)
    for i, dist in enumerate(distributions):
        p = to_simplex_prob(dist)
        mean += weights[i] * p
    mean = to_simplex_prob(mean)
    
    # Iterative refinement using geodesic interpolation
    for iter_num in range(max_iter):
        update = np.zeros(dim, dtype=np.float64)
        total_weight = 0.0
        
        for i, dist in enumerate(distributions):
            p = to_simplex_prob(dist)
            distance = fisher_rao_distance(mean, p)
            
            if distance < 1e-10:
                continue  # Already at this point
            
            # Geodesic step towards p
            step_size = weights[i]
            intermediate = geodesic_interpolation_simplex(mean, p, step_size)
            
            update += weights[i] * intermediate
            total_weight += weights[i]
        
        if total_weight < 1e-10:
            break
        
        # Normalize update
        new_mean = update / total_weight
        mean_update = to_simplex_prob(new_mean)
        
        # Check convergence
        change = fisher_rao_distance(mean, mean_update)
        mean = mean_update
        
        if change < tolerance:
            break
    
    return mean


def batch_fisher_rao_distance(
    query: np.ndarray,
    candidates: List[np.ndarray]
) -> np.ndarray:
    """
    Batch compute Fisher-Rao distances from a query to multiple candidates.
    
    Args:
        query: Query probability distribution
        candidates: List of candidate distributions
        
    Returns:
        Array of distances
    """
    return np.array([fisher_rao_distance(query, candidate) for candidate in candidates])


def find_nearest_simplex(
    query: np.ndarray,
    candidates: List[np.ndarray],
    k: int = 10
) -> List[Tuple[int, float]]:
    """
    Find k nearest distributions to query using Fisher-Rao distance.
    
    Args:
        query: Query probability distribution
        candidates: List of candidate distributions
        k: Number of nearest neighbors to return
        
    Returns:
        List of (index, distance) tuples sorted by distance
    """
    distances = [
        (i, fisher_rao_distance(query, candidate))
        for i, candidate in enumerate(candidates)
    ]
    distances.sort(key=lambda x: x[1])
    return distances[:k]


# Export constants
SIMPLEX_EPSILON = EPS


__all__ = [
    'to_simplex_prob',
    'validate_simplex',
    'fisher_rao_distance',
    'geodesic_interpolation_simplex',
    'geodesic_mean_simplex',
    'batch_fisher_rao_distance',
    'find_nearest_simplex',
    'SIMPLEX_EPSILON',
]
