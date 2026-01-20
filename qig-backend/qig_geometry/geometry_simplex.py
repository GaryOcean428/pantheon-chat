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
    tolerance: float = 1e-4  # Relaxed from 1e-5 for trajectory approximation
) -> np.ndarray:
    """
    Compute weighted geodesic mean (Fréchet mean) on probability simplex.
    
    Uses iterative algorithm to find the point that minimizes sum of
    squared Fisher-Rao distances to all input distributions.
    
    CONVERGENCE IMPROVEMENTS (2026-01-15):
    - Adaptive step size (starts at 0.5, decays if overshooting)
    - High variance detection (fallback to weighted mean if max_dist > π/6)
    - Relaxed tolerance (1e-4 acceptable for trajectory approximation)
    - Stall detection (early stop if not improving)
    - Reduced log noise (only warn once per process)
    
    Args:
        distributions: List of probability distributions
        weights: Optional weights (default: uniform)
        max_iter: Maximum iterations
        tolerance: Convergence tolerance (default 1e-4)
        
    Returns:
        Weighted geodesic mean distribution
    """
    if not distributions or len(distributions) == 0:
        raise ValueError("geodesic_mean_simplex: empty distributions list")
    
    n = len(distributions)
    dim = distributions[0].size
    
    # Single distribution - return immediately
    if n == 1:
        return to_simplex_prob(distributions[0])
    
    # Default to uniform weights
    if weights is None:
        weights = np.ones(n) / n
    else:
        weights = np.asarray(weights, dtype=np.float64)
        weights = weights / weights.sum()  # Normalize
    
    # Convert all to simplex once
    simplex_dists = [to_simplex_prob(d) for d in distributions]
    
    # Check for high variance (dispersed distributions)
    # If max pairwise distance > π/6, use weighted mean fallback
    max_dist = 0.0
    for i in range(min(n, 5)):  # Sample first 5 pairs for efficiency
        for j in range(i+1, min(n, 5)):
            dist = fisher_rao_distance(simplex_dists[i], simplex_dists[j])
            max_dist = max(max_dist, dist)
    
    # High variance threshold: π/6 (~0.52 radians)
    # This catches highly dispersed cases and prevents unnecessary iterations
    if max_dist > np.pi / 6:
        # Use weighted mean as fallback (faster for dispersed points)
        mean = np.zeros(dim, dtype=np.float64)
        for i, p in enumerate(simplex_dists):
            mean += weights[i] * p
        return to_simplex_prob(mean)
    
    # Initialize mean as weighted average (good starting point for close distributions)
    mean = np.zeros(dim, dtype=np.float64)
    for i, p in enumerate(simplex_dists):
        mean += weights[i] * p
    mean = to_simplex_prob(mean)
    
    # Adaptive iterative refinement
    step_size = 0.5  # Start larger than naive approach
    min_step = 0.01  # Minimum step size before giving up
    stall_count = 0  # Track iterations with minimal progress
    prev_change = float('inf')
    
    for iter_num in range(max_iter):
        update = np.zeros(dim, dtype=np.float64)
        total_weight = 0.0
        
        for i, p in enumerate(simplex_dists):
            distance = fisher_rao_distance(mean, p)
            
            if distance < 1e-10:
                continue  # Already at this point
            
            # Adaptive geodesic step towards p
            adaptive_step = step_size * weights[i]
            intermediate = geodesic_interpolation_simplex(mean, p, min(adaptive_step, 1.0))
            
            update += weights[i] * intermediate
            total_weight += weights[i]
        
        if total_weight < 1e-10:
            break
        
        # Normalize update
        new_mean = update / total_weight
        mean_update = to_simplex_prob(new_mean)
        
        # Check convergence
        change = fisher_rao_distance(mean, mean_update)
        
        if change < tolerance:
            return mean_update  # Converged successfully
        
        # Check for stall (minimal progress)
        if abs(change - prev_change) < 1e-6:
            stall_count += 1
            if stall_count >= 5:
                return mean_update  # Stalled - return best estimate
        else:
            stall_count = 0
        
        # Adaptive step size: reduce if overshooting
        if change > prev_change:
            step_size *= 0.5  # Overshot - reduce step
            if step_size < min_step:
                return mean_update  # Step too small
        
        prev_change = change
        mean = mean_update
    
    # Reached max iterations (should be rare with improvements)
    # Only log once per process to avoid spam
    if not hasattr(geodesic_mean_simplex, '_logged_warning'):
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(
            f"geodesic_mean_simplex: reached {max_iter} iterations "
            f"(final change: {change:.2e}). This may indicate highly dispersed distributions. "
            f"Further occurrences will not be logged."
        )
        geodesic_mean_simplex._logged_warning = True
    
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


def simplex_mean_sqrt_space(distributions: List[np.ndarray], weights: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Closed-form mean on probability simplex using sqrt-space (Hellinger) coordinates.
    
    This is a fast, non-iterative approximation to the Fréchet mean that works well
    for nearby distributions. For widely dispersed distributions, use geodesic_mean_simplex.
    
    Algorithm:
    1. Transform to sqrt-space: sqrt(p_i)
    2. Average in sqrt-space (Euclidean mean)
    3. Transform back: (mean)^2 → simplex
    
    This is the Fréchet mean for Hellinger distance, which approximates
    the Fisher-Rao Fréchet mean for nearby points.
    
    Args:
        distributions: List of probability simplex vectors
        weights: Optional non-negative weights (will be normalized)
        
    Returns:
        Mean simplex vector
        
    Raises:
        ValueError: If distributions list is empty or weights don't match
    """
    if not distributions:
        raise ValueError("simplex_mean_sqrt_space: empty distributions list")
    
    n = len(distributions)
    
    # Default to uniform weights
    if weights is None:
        weights = np.ones(n) / n
    else:
        weights = np.asarray(weights, dtype=np.float64)
        if len(weights) != n:
            raise ValueError(
                f"simplex_mean_sqrt_space: weight count {len(weights)} "
                f"doesn't match distribution count {n}"
            )
        weights = weights / weights.sum()  # Normalize
    
    # Validate inputs as simplices
    simplex_dists = []
    for i, d in enumerate(distributions):
        is_valid, reason = validate_simplex(d)
        if not is_valid:
            # Try to convert
            d_simplex = to_simplex_prob(d)
            is_valid, reason = validate_simplex(d_simplex)
            if not is_valid:
                raise ValueError(
                    f"simplex_mean_sqrt_space: distribution[{i}] invalid simplex: {reason}"
                )
            simplex_dists.append(d_simplex)
        else:
            simplex_dists.append(np.asarray(d, dtype=np.float64).flatten())
    
    # Transform to sqrt-space
    sqrt_distributions = [np.sqrt(p) for p in simplex_dists]
    
    # Weighted average in sqrt-space (Euclidean)
    sqrt_mean = np.zeros_like(sqrt_distributions[0])
    for w, sqrt_p in zip(weights, sqrt_distributions):
        sqrt_mean += w * sqrt_p
    
    # Transform back to simplex
    p_mean = sqrt_mean ** 2
    return to_simplex_prob(p_mean)


def weighted_simplex_mean(distributions: List[np.ndarray], weights: np.ndarray) -> np.ndarray:
    """
    Weighted mean on probability simplex using sqrt-space.
    
    Alias for simplex_mean_sqrt_space with explicit weights parameter.
    Use this when you need weighted averaging; use simplex_mean_sqrt_space
    with weights=None for uniform weighting.
    
    Args:
        distributions: List of probability simplex vectors
        weights: Non-negative weights (will be normalized)
        
    Returns:
        Weighted mean simplex vector
    """
    return simplex_mean_sqrt_space(distributions, weights=weights)


# Export constants
SIMPLEX_EPSILON = EPS


__all__ = [
    'to_simplex_prob',
    'validate_simplex',
    'fisher_rao_distance',
    'geodesic_interpolation_simplex',
    'geodesic_mean_simplex',
    'simplex_mean_sqrt_space',
    'weighted_simplex_mean',
    'batch_fisher_rao_distance',
    'find_nearest_simplex',
    'SIMPLEX_EPSILON',
]
