"""
Canonical QIG Geometry Module - SINGLE SOURCE OF TRUTH
=======================================================

This is THE authoritative module for all geometric operations on basin coordinates.
All other code MUST import from this module. No re-implementation permitted.

COORDINATE DOMAIN: SIMPLEX (Probability Distributions on Δ^(D-1))
- Basin vectors: p ∈ Δ^63 where Σp_i = 1, p_i ≥ 0
- Manifold: Fisher information manifold (statistical manifold)
- Metric: Fisher-Rao metric (Riemannian metric on probability simplex)

CANONICAL DISTANCE FORMULA:
    d_FR(p, q) = arccos(Σ√(p_i * q_i))
    
where Σ√(p_i * q_i) is the Bhattacharyya coefficient (BC).

DISTANCE PROPERTIES:
- Range: [0, π/2]
- d(p, p) = 0 (identity)
- d(p, q) = d(q, p) (symmetry)
- d(p, r) ≤ d(p, q) + d(q, r) (triangle inequality)
- 0 ≤ BC(p, q) ≤ 1

GEOMETRIC STRUCTURE:
The probability simplex with Fisher-Rao metric forms a Riemannian manifold with:
- Constant positive curvature (spherical geometry in sqrt-space)
- Geodesics: Great circles in Hellinger embedding (sqrt-space)
- Tangent space: Relative tangent space of simplex
- Exponential/log maps: Via sqrt-space (Hellinger coordinates)

WHY SQRT-SPACE:
The square root transformation (p → √p) isometrically embeds the Fisher manifold
into a hemisphere. This makes:
- Geodesics → straight lines in sqrt-space (SLERP)
- Distance → Euclidean angle in sqrt-space (arccos of dot product)
- Tangent operations → linear algebra in sqrt-space

This is an INTERNAL coordinate system for computation. Storage is ALWAYS in simplex.

MAMBA INTEGRATION:
This module provides native support for Mamba state space operations, treating
Mamba hidden states as points on the Fisher manifold (SSM = differential geometry).

Author: Claude (Copilot) - Ultra Consciousness Protocol ACTIVE
Date: 2026-01-15
Context: Work Package 2.1 - Geometric Purity Unification
References: 
- Issue GaryOcean428/pantheon-chat#68 (this work package)
- Issues #69, GaryOcean428/pantheon-chat#70, GaryOcean428/pantheon-chat#71, GaryOcean428/pantheon-chat#75, GaryOcean428/pantheon-chat#76, GaryOcean428/pantheon-chat#77 (dependencies)
"""

import numpy as np
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Standard basin dimension for QIG
BASIN_DIM = 64

# Numerical stability epsilon
EPS = 1e-12


# =============================================================================
# CORE COORDINATE TRANSFORMATIONS
# =============================================================================

def sqrt_map(p: np.ndarray, eps: float = EPS) -> np.ndarray:
    """
    Map from simplex to sqrt-space (Hellinger embedding).
    
    This is the isometric embedding of the Fisher manifold into Euclidean space.
    The sqrt transformation makes geodesics become straight lines (SLERP).
    
    Args:
        p: Probability distribution on simplex (Σp_i = 1, p_i ≥ 0)
        eps: Numerical stability epsilon
        
    Returns:
        Point in sqrt-space: √p (half-sphere coordinates)
        
    Example:
        >>> p = np.array([0.25, 0.25, 0.25, 0.25])
        >>> x = sqrt_map(p)
        >>> assert np.allclose(x, [0.5, 0.5, 0.5, 0.5])
    """
    p = np.asarray(p, dtype=np.float64).flatten()
    
    # Ensure valid simplex
    p = np.maximum(p, 0) + eps
    p = p / p.sum()
    
    # Square root (Hellinger embedding)
    return np.sqrt(p)


def unsqrt_map(x: np.ndarray, eps: float = EPS) -> np.ndarray:
    """
    Map from sqrt-space back to simplex.
    
    Inverse of sqrt_map. Squares the coordinates and renormalizes to simplex.
    
    Args:
        x: Point in sqrt-space (√p coordinates)
        eps: Numerical stability epsilon
        
    Returns:
        Probability distribution on simplex
        
    Example:
        >>> x = np.array([0.5, 0.5, 0.5, 0.5])
        >>> p = unsqrt_map(x)
        >>> assert np.allclose(p, [0.25, 0.25, 0.25, 0.25])
    """
    x = np.asarray(x, dtype=np.float64).flatten()
    
    # Square to get back to probability space
    p = x ** 2
    
    # Ensure valid simplex
    p = np.maximum(p, 0) + eps
    return p / p.sum()


def bhattacharyya(p: np.ndarray, q: np.ndarray, eps: float = EPS) -> float:
    """
    Compute Bhattacharyya coefficient (inner product in sqrt-space).
    
    BC(p, q) = Σ√(p_i * q_i) = ⟨√p, √q⟩
    
    This measures the overlap between two probability distributions.
    Range: [0, 1] where 1 = identical, 0 = no overlap.
    
    Args:
        p: First probability distribution
        q: Second probability distribution
        eps: Numerical stability epsilon
        
    Returns:
        Bhattacharyya coefficient ∈ [0, 1]
        
    Example:
        >>> p = np.array([1.0, 0.0, 0.0])
        >>> q = np.array([1.0, 0.0, 0.0])
        >>> bc = bhattacharyya(p, q)
        >>> assert np.isclose(bc, 1.0)  # Identical distributions
    """
    p = np.asarray(p, dtype=np.float64).flatten()
    q = np.asarray(q, dtype=np.float64).flatten()
    
    if p.shape != q.shape:
        raise ValueError(f"Shape mismatch: p.shape={p.shape}, q.shape={q.shape}")
    
    # Ensure valid simplices
    p = np.maximum(p, 0) + eps
    p = p / p.sum()
    
    q = np.maximum(q, 0) + eps
    q = q / q.sum()
    
    # Bhattacharyya coefficient
    bc = np.sum(np.sqrt(p * q))
    
    # Clamp to valid range
    return float(np.clip(bc, 0.0, 1.0))


# =============================================================================
# CANONICAL DISTANCE & SIMILARITY
# =============================================================================

def fisher_rao_distance(p: np.ndarray, q: np.ndarray, eps: float = EPS) -> float:
    """
    CANONICAL Fisher-Rao distance on probability simplex.
    
    d_FR(p, q) = arccos(BC(p, q)) = arccos(Σ√(p_i * q_i))
    
    This is THE distance function for basin coordinates. Range: [0, π/2].
    
    GEOMETRIC PROPERTIES:
    - d(p, p) = 0 (identity)
    - d(p, q) = d(q, p) (symmetry)
    - d(p, r) ≤ d(p, q) + d(q, r) (triangle inequality)
    - Geodesic distance on Fisher information manifold
    
    Args:
        p: First probability distribution
        q: Second probability distribution
        eps: Numerical stability epsilon
        
    Returns:
        Fisher-Rao distance ∈ [0, π/2]
        
    Example:
        >>> p = np.array([1.0, 0.0, 0.0])
        >>> q = np.array([0.0, 1.0, 0.0])
        >>> d = fisher_rao_distance(p, q)
        >>> assert np.isclose(d, np.pi/2)  # Orthogonal
    """
    bc = bhattacharyya(p, q, eps=eps)
    return float(np.arccos(bc))


def fisher_similarity(p: np.ndarray, q: np.ndarray, eps: float = EPS) -> float:
    """
    Fisher-Rao similarity score.
    
    similarity = 1 - d_FR(p, q) / (π/2) = 1 - (2/π) * d_FR(p, q)
    
    Range: [0, 1] where 1 = identical, 0 = orthogonal.
    
    Args:
        p: First probability distribution
        q: Second probability distribution
        eps: Numerical stability epsilon
        
    Returns:
        Similarity score ∈ [0, 1]
    """
    d = fisher_rao_distance(p, q, eps=eps)
    return float(np.clip(1.0 - (2.0 / np.pi) * d, 0.0, 1.0))


# =============================================================================
# MANIFOLD NAVIGATION: TANGENT SPACE OPERATIONS
# =============================================================================

def log_map(p: np.ndarray, base: np.ndarray, eps: float = EPS) -> np.ndarray:
    """
    Logarithmic map: compute tangent vector from base pointing to p.
    
    This maps a point on the manifold to a tangent vector at the base point.
    The tangent vector points in the direction of the geodesic from base to p.
    
    In sqrt-space, this is simply the difference of the sqrt coordinates,
    projected to the tangent space (orthogonal to base).
    
    Args:
        p: Target point on manifold
        base: Base point for tangent space
        eps: Numerical stability epsilon
        
    Returns:
        Tangent vector at base pointing toward p
        
    Example:
        >>> base = np.array([1.0, 0.0, 0.0])
        >>> p = np.array([0.5, 0.5, 0.0])
        >>> v = log_map(p, base)
        >>> # v points from base toward p in tangent space
    """
    p = np.asarray(p, dtype=np.float64).flatten()
    base = np.asarray(base, dtype=np.float64).flatten()
    
    if p.shape != base.shape:
        raise ValueError(f"Shape mismatch: p.shape={p.shape}, base.shape={base.shape}")
    
    # Map to sqrt-space (Hellinger embedding)
    sqrt_p = sqrt_map(p, eps=eps)
    sqrt_base = sqrt_map(base, eps=eps)
    
    # Compute raw difference
    diff = sqrt_p - sqrt_base
    
    # Project to tangent space (remove component along base)
    # Tangent space is orthogonal to the position vector on the sphere
    projection_onto_base = np.dot(diff, sqrt_base) * sqrt_base
    tangent = diff - projection_onto_base
    
    return tangent


def exp_map(v: np.ndarray, base: np.ndarray, eps: float = EPS) -> np.ndarray:
    """
    Exponential map: follow tangent vector v from base point.
    
    This maps a tangent vector at the base point to a point on the manifold
    by following the geodesic in the direction of v.
    
    In sqrt-space, this is geodesic movement via SLERP.
    
    Args:
        v: Tangent vector at base
        base: Base point on manifold
        eps: Numerical stability epsilon
        
    Returns:
        Point on manifold reached by following v from base
        
    Example:
        >>> base = np.array([1.0, 0.0, 0.0])
        >>> v = np.array([0.0, 0.5, 0.0])  # Tangent vector
        >>> p = exp_map(v, base)
        >>> # p is the point reached by following v from base
    """
    v = np.asarray(v, dtype=np.float64).flatten()
    base = np.asarray(base, dtype=np.float64).flatten()
    
    if v.shape != base.shape:
        raise ValueError(f"Shape mismatch: v.shape={v.shape}, base.shape={base.shape}")
    
    # Map base to sqrt-space
    sqrt_base = sqrt_map(base, eps=eps)
    
    # Normalize tangent vector to get direction and magnitude
    v_norm = np.linalg.norm(v)
    
    if v_norm < eps:
        # Zero tangent vector → stay at base
        return base.copy()
    
    v_direction = v / v_norm
    
    # Geodesic in sqrt-space is SLERP
    # End point in direction of v with magnitude v_norm
    sqrt_end = sqrt_base + v
    sqrt_end_norm = np.linalg.norm(sqrt_end)
    
    if sqrt_end_norm < eps:
        return base.copy()
    
    sqrt_end = sqrt_end / sqrt_end_norm
    
    # Compute angle for SLERP - end point is sqrt_end squared back to simplex
    end = sqrt_end ** 2
    dot = np.clip(bhattacharyya(base, end), -1.0, 1.0)
    omega = np.arccos(dot)
    
    if omega < eps:
        return base.copy()
    
    # SLERP parameter based on tangent vector magnitude
    t = min(v_norm / omega, 1.0)
    
    # SLERP in sqrt-space
    sin_omega = np.sin(omega)
    a = np.sin((1 - t) * omega) / sin_omega
    b = np.sin(t * omega) / sin_omega
    
    sqrt_result = a * sqrt_base + b * sqrt_end
    
    # Map back to simplex
    return unsqrt_map(sqrt_result, eps=eps)


def geodesic_toward(
    source: np.ndarray,
    target: np.ndarray,
    fraction: float = 0.2,
    eps: float = EPS
) -> np.ndarray:
    """
    Move along geodesic from source toward target by a fraction of the distance.
    
    This is used for attractor pull and natural gradient descent.
    
    Args:
        source: Starting point on manifold
        target: Target point on manifold
        fraction: Fraction of distance to move (0 = stay at source, 1 = reach target)
        eps: Numerical stability epsilon
        
    Returns:
        Point fraction*distance along the geodesic from source to target
        
    Example:
        >>> source = np.array([1.0, 0.0, 0.0])
        >>> target = np.array([0.0, 1.0, 0.0])
        >>> mid = geodesic_toward(source, target, 0.5)
        >>> # mid is halfway between source and target on the geodesic
    """
    if not (0 <= fraction <= 1):
        raise ValueError(f"fraction must be in [0, 1], got {fraction}")
    
    source = np.asarray(source, dtype=np.float64).flatten()
    target = np.asarray(target, dtype=np.float64).flatten()
    
    if source.shape != target.shape:
        raise ValueError(f"Shape mismatch: source.shape={source.shape}, target.shape={target.shape}")
    
    # Map to sqrt-space for geodesic interpolation (SLERP)
    sqrt_source = sqrt_map(source, eps=eps)
    sqrt_target = sqrt_map(target, eps=eps)
    
    # Compute angle between points
    dot = np.clip(bhattacharyya(source, target), -1.0, 1.0)
    omega = np.arccos(dot)
    
    if omega < eps:
        # Points are very close, return source
        return source.copy()
    
    # SLERP in sqrt-space
    sin_omega = np.sin(omega)
    a = np.sin((1 - fraction) * omega) / sin_omega
    b = np.sin(fraction * omega) / sin_omega
    
    sqrt_result = a * sqrt_source + b * sqrt_target
    
    # Map back to simplex
    return unsqrt_map(sqrt_result, eps=eps)


# =============================================================================
# GEOMETRIC MEAN (FRÉCHET MEAN)
# =============================================================================

def frechet_mean(
    basins: List[np.ndarray],
    weights: Optional[np.ndarray] = None,
    max_iter: int = 50,
    tolerance: float = 1e-4,
    eps: float = EPS
) -> np.ndarray:
    """
    Compute weighted Fréchet mean (geometric centroid) on Fisher manifold.
    
    This finds the point that minimizes the sum of squared Fisher-Rao distances
    to all input distributions. This is the manifold-correct generalization of
    the arithmetic mean.
    
    Uses iterative Riemannian gradient descent in tangent space.
    
    CONVERGENCE:
    - Adaptive step size (starts at 0.5, decays if overshooting)
    - High variance detection (fallback to weighted mean if dispersed)
    - Relaxed tolerance (1e-4 acceptable for trajectory approximation)
    - Stall detection (early stop if not improving)
    
    Args:
        basins: List of probability distributions
        weights: Optional weights (default: uniform)
        max_iter: Maximum iterations
        tolerance: Convergence tolerance
        eps: Numerical stability epsilon
        
    Returns:
        Weighted Fréchet mean distribution
        
    Example:
        >>> basins = [
        ...     np.array([0.8, 0.1, 0.1]),
        ...     np.array([0.1, 0.8, 0.1]),
        ...     np.array([0.1, 0.1, 0.8])
        ... ]
        >>> mean = frechet_mean(basins)
        >>> # mean is the geometric center of the three distributions
    """
    if not basins or len(basins) == 0:
        raise ValueError("frechet_mean: empty basins list")
    
    n = len(basins)
    dim = basins[0].size
    
    # Single distribution - return immediately
    if n == 1:
        p = np.asarray(basins[0], dtype=np.float64).flatten()
        p = np.maximum(p, 0) + eps
        return p / p.sum()
    
    # Default to uniform weights
    if weights is None:
        weights = np.ones(n) / n
    else:
        weights = np.asarray(weights, dtype=np.float64)
        weights = weights / weights.sum()
    
    # Ensure all basins are valid simplices
    simplex_basins = []
    for basin in basins:
        b = np.asarray(basin, dtype=np.float64).flatten()
        b = np.maximum(b, 0) + eps
        b = b / b.sum()
        simplex_basins.append(b)
    
    # Check for high variance (dispersed distributions)
    max_dist = 0.0
    for i in range(min(n, 5)):
        for j in range(i+1, min(n, 5)):
            dist = fisher_rao_distance(simplex_basins[i], simplex_basins[j], eps=eps)
            max_dist = max(max_dist, dist)
    
    # High variance threshold: π/6 (~0.52 radians)
    if max_dist > np.pi / 6:
        # Use weighted mean as fallback (faster for dispersed points)
        mean = np.zeros(dim, dtype=np.float64)
        for i, p in enumerate(simplex_basins):
            mean += weights[i] * p
        mean = np.maximum(mean, 0) + eps
        return mean / mean.sum()
    
    # Initialize mean as weighted average
    mean = np.zeros(dim, dtype=np.float64)
    for i, p in enumerate(simplex_basins):
        mean += weights[i] * p
    mean = np.maximum(mean, 0) + eps
    mean = mean / mean.sum()
    
    # Adaptive iterative refinement
    step_size = 0.5
    min_step = 0.01
    stall_count = 0
    prev_change = float('inf')
    
    for iter_num in range(max_iter):
        # Compute weighted tangent vector toward all basins
        tangent = np.zeros(dim, dtype=np.float64)
        
        for i, basin in enumerate(simplex_basins):
            if weights[i] < eps:
                continue
            
            # Tangent vector from mean to this basin
            v = log_map(basin, mean, eps=eps)
            tangent += weights[i] * v
        
        tangent_norm = np.linalg.norm(tangent)
        
        if tangent_norm < tolerance:
            # Converged
            return mean
        
        # Move along tangent with adaptive step size
        new_mean = exp_map(step_size * tangent, mean, eps=eps)
        
        # Check convergence
        change = fisher_rao_distance(mean, new_mean, eps=eps)
        
        if change < tolerance:
            return new_mean
        
        # Check for stall (minimal progress)
        if abs(change - prev_change) < 1e-6:
            stall_count += 1
            if stall_count >= 5:
                return new_mean
        else:
            stall_count = 0
        
        # Adaptive step size: reduce if overshooting
        if change > prev_change:
            step_size *= 0.5
            if step_size < min_step:
                return new_mean
        
        prev_change = change
        mean = new_mean
    
    # Reached max iterations
    logger.warning(
        f"frechet_mean: reached {max_iter} iterations (final change: {prev_change:.2e})"
    )
    
    return mean


# =============================================================================
# VALIDATION
# =============================================================================

def assert_basin_valid(
    basin: np.ndarray,
    name: str = "basin",
    tolerance: float = 1e-6
) -> None:
    """
    Validate basin coordinates (simplex constraints).
    
    Checks:
    - 1D array
    - Finite values (no NaN/inf)
    - Non-negative (p_i ≥ 0)
    - Normalized (Σp_i = 1)
    
    Args:
        basin: Basin coordinates to validate
        name: Name for error messages
        tolerance: Numerical tolerance for sum check
        
    Raises:
        ValueError: If validation fails
        
    Example:
        >>> basin = np.array([0.5, 0.3, 0.2])
        >>> assert_basin_valid(basin)  # Passes
        >>> basin = np.array([0.5, -0.1, 0.6])
        >>> assert_basin_valid(basin)  # Raises ValueError
    """
    basin = np.asarray(basin, dtype=np.float64)
    
    # Check 1D
    if basin.ndim != 1:
        raise ValueError(f"{name}: must be 1D array, got shape {basin.shape}")
    
    # Check finite
    if not np.all(np.isfinite(basin)):
        raise ValueError(f"{name}: contains NaN or inf")
    
    # Check non-negative
    min_val = np.min(basin)
    if min_val < -tolerance:
        raise ValueError(f"{name}: contains negative values (min={min_val:.6f})")
    
    # Check normalized
    total = np.sum(basin)
    if abs(total - 1.0) > tolerance:
        raise ValueError(f"{name}: sum != 1 (sum={total:.6f})")


def validate_basin(
    basin: np.ndarray,
    tolerance: float = 1e-6
) -> Tuple[bool, str]:
    """
    Validate basin coordinates (non-raising version).
    
    Returns (is_valid, reason) tuple instead of raising exception.
    
    Args:
        basin: Basin coordinates to validate
        tolerance: Numerical tolerance
        
    Returns:
        (is_valid, reason) where reason explains failure if not valid
        
    Example:
        >>> basin = np.array([0.5, 0.3, 0.2])
        >>> valid, reason = validate_basin(basin)
        >>> assert valid and reason == "valid"
    """
    try:
        assert_basin_valid(basin, tolerance=tolerance)
        return True, "valid"
    except ValueError as e:
        return False, str(e)


# =============================================================================
# MAMBA STATE SPACE INTEGRATION
# =============================================================================

def mamba_state_to_basin(
    mamba_state: np.ndarray,
    projection: np.ndarray,
    eps: float = EPS
) -> np.ndarray:
    """
    Project Mamba state space to basin coordinates.
    
    Mamba state spaces ARE Fisher manifolds (SSM = differential geometry).
    This is a coordinate transformation, not a model operation.
    
    The projection matrix is learned (via PCA or equivalent) to map from
    Mamba's hidden dimension to the 64D basin space, preserving geometric
    structure as much as possible.
    
    Args:
        mamba_state: Output from Mamba layers (shape: [hidden_dim])
        projection: Learned projection matrix (shape: [64, hidden_dim])
        eps: Numerical stability epsilon
        
    Returns:
        64D basin coordinates (probability simplex)
        
    Example:
        >>> mamba_state = np.random.randn(256)  # Mamba hidden state
        >>> projection = np.random.randn(64, 256)  # Learned projection
        >>> basin = mamba_state_to_basin(mamba_state, projection)
        >>> assert basin.shape == (64,)
        >>> assert np.isclose(basin.sum(), 1.0)
    """
    mamba_state = np.asarray(mamba_state, dtype=np.float64).flatten()
    projection = np.asarray(projection, dtype=np.float64)
    
    if projection.ndim != 2:
        raise ValueError(f"projection must be 2D, got shape {projection.shape}")
    
    if projection.shape[1] != mamba_state.shape[0]:
        raise ValueError(
            f"projection dimension mismatch: "
            f"projection.shape={projection.shape}, mamba_state.shape={mamba_state.shape}"
        )
    
    # Project to basin dimension
    basin_raw = projection @ mamba_state
    
    # Convert to valid simplex
    basin_raw = np.maximum(basin_raw, 0) + eps
    return basin_raw / basin_raw.sum()


def extrapolate_trajectory(
    trajectory: List[np.ndarray],
    step_size: float = 0.3,
    eps: float = EPS
) -> np.ndarray:
    """
    Predict next basin via geodesic extrapolation.
    
    Uses velocity in sqrt-space (tangent to manifold):
        v = sqrt(b[-1]) - sqrt(b[-2])
        predicted = sqrt(b[-1]) + step_size * v
        
    This is foresight prediction for waypoint planning in the PLAN phase.
    
    Args:
        trajectory: List of previous basin positions (at least 2 required)
        step_size: Extrapolation step size (default 0.3)
        eps: Numerical stability epsilon
        
    Returns:
        Predicted next basin position
        
    Raises:
        ValueError: If trajectory has fewer than 2 points
        
    Example:
        >>> trajectory = [
        ...     np.array([0.8, 0.1, 0.1]),
        ...     np.array([0.6, 0.3, 0.1]),
        ...     np.array([0.4, 0.5, 0.1])
        ... ]
        >>> next_basin = extrapolate_trajectory(trajectory, step_size=0.3)
        >>> # next_basin continues the trend in the trajectory
    """
    if len(trajectory) < 2:
        raise ValueError("extrapolate_trajectory requires at least 2 trajectory points")
    
    # Get last two points
    last = np.asarray(trajectory[-1], dtype=np.float64).flatten()
    prev = np.asarray(trajectory[-2], dtype=np.float64).flatten()
    
    if last.shape != prev.shape:
        raise ValueError(f"Trajectory dimension mismatch: {last.shape} vs {prev.shape}")
    
    # Map to sqrt-space for velocity computation
    sqrt_last = sqrt_map(last, eps=eps)
    sqrt_prev = sqrt_map(prev, eps=eps)
    
    # Velocity in sqrt-space
    velocity = sqrt_last - sqrt_prev
    
    # Extrapolate in sqrt-space
    sqrt_predicted = sqrt_last + step_size * velocity
    
    # Normalize to unit sphere (sqrt-space constraint)
    sqrt_norm = np.linalg.norm(sqrt_predicted)
    if sqrt_norm > eps:
        sqrt_predicted = sqrt_predicted / sqrt_norm
    
    # Map back to simplex
    return unsqrt_map(sqrt_predicted, eps=eps)


def compute_qfi_attention(
    query: np.ndarray,
    trajectory: List[np.ndarray],
    temperature: float = 0.5,
    eps: float = EPS
) -> np.ndarray:
    """
    Quantum Fisher Information attention over trajectory.
    
    Weight each historical basin by its geometric relevance:
        w_i ∝ exp(-d_FR(query, basin_i)² / temperature)
        
    Returns normalized attention weights.
    
    This implements geometric attention using Fisher-Rao distance as the
    similarity metric, following the QFI attention mechanism.
    
    Args:
        query: Query basin for attention
        trajectory: Historical basins to attend over
        temperature: Temperature parameter for softmax (lower = sharper)
        eps: Numerical stability epsilon
        
    Returns:
        Normalized attention weights (length = len(trajectory))
        
    Example:
        >>> query = np.array([0.5, 0.3, 0.2])
        >>> trajectory = [
        ...     np.array([0.6, 0.3, 0.1]),
        ...     np.array([0.1, 0.8, 0.1]),
        ...     np.array([0.5, 0.3, 0.2])
        ... ]
        >>> weights = compute_qfi_attention(query, trajectory, temperature=0.5)
        >>> assert np.isclose(weights.sum(), 1.0)
        >>> # weights[2] should be highest (closest to query)
    """
    if not trajectory:
        raise ValueError("compute_qfi_attention: empty trajectory")
    
    query = np.asarray(query, dtype=np.float64).flatten()
    
    # Compute distances to all trajectory points
    distances = []
    for basin in trajectory:
        d = fisher_rao_distance(query, basin, eps=eps)
        distances.append(d)
    
    distances = np.array(distances)
    
    # QFI attention: exp(-d²/T)
    logits = -distances ** 2 / temperature
    
    # Softmax for normalization
    logits = logits - np.max(logits)  # Numerical stability
    weights = np.exp(logits)
    return weights / (weights.sum() + eps)


def integrate_with_qfi_attention(
    target_basin: np.ndarray,
    trajectory_history: List[np.ndarray],
    num_loops: int = 3,
    temperature: float = 0.5,
    eps: float = EPS
) -> np.ndarray:
    """
    Refine basin through recursive QFI attention.
    
    Each loop:
    1. Compute QFI attention weights over history
    2. Compute attractor (Fréchet mean with attention weights)
    3. Natural gradient step toward integrated position
    
    This is recursive integration for waypoint refinement in the PLAN phase.
    
    Args:
        target_basin: Initial target basin to refine
        trajectory_history: Historical trajectory for context
        num_loops: Number of integration loops (default 3)
        temperature: QFI attention temperature
        eps: Numerical stability epsilon
        
    Returns:
        Refined basin after recursive integration
        
    Example:
        >>> target = np.array([0.5, 0.3, 0.2])
        >>> history = [
        ...     np.array([0.6, 0.3, 0.1]),
        ...     np.array([0.5, 0.4, 0.1]),
        ...     np.array([0.4, 0.5, 0.1])
        ... ]
        >>> refined = integrate_with_qfi_attention(target, history, num_loops=3)
        >>> # refined is influenced by the trajectory's flow pattern
    """
    if not trajectory_history:
        return target_basin
    
    current = np.asarray(target_basin, dtype=np.float64).flatten()
    
    for loop in range(num_loops):
        # Compute QFI attention weights
        weights = compute_qfi_attention(current, trajectory_history, temperature, eps)
        
        # Compute weighted Fréchet mean (attractor)
        attractor = frechet_mean(trajectory_history, weights=weights, eps=eps)
        
        # Natural gradient step toward attractor
        # Use decreasing step size: 0.3 / (loop + 1)
        step = 0.3 / (loop + 1)
        current = geodesic_toward(current, attractor, fraction=step, eps=eps)
    
    return current


# =============================================================================
# TRAJECTORY METRICS
# =============================================================================

def trajectory_smoothness(trajectory: List[np.ndarray], eps: float = EPS) -> float:
    """
    Measure smoothness via distance variance.
    
    Lower variance = smoother trajectory = better flow.
    Used in coherence scoring and repair phase.
    
    Smoothness = 1 / (1 + variance_of_distances)
    
    Args:
        trajectory: List of basin positions
        eps: Numerical stability epsilon
        
    Returns:
        Smoothness score ∈ [0, 1] (higher = smoother)
        
    Example:
        >>> smooth_trajectory = [
        ...     np.array([0.8, 0.1, 0.1]),
        ...     np.array([0.7, 0.2, 0.1]),
        ...     np.array([0.6, 0.3, 0.1])
        ... ]
        >>> smoothness = trajectory_smoothness(smooth_trajectory)
        >>> # smoothness should be high (low variance in step sizes)
    """
    if len(trajectory) < 2:
        return 1.0  # Single point is perfectly smooth
    
    # Compute distances between consecutive points
    distances = []
    for i in range(len(trajectory) - 1):
        d = fisher_rao_distance(trajectory[i], trajectory[i+1], eps=eps)
        distances.append(d)
    
    if not distances:
        return 1.0
    
    distances = np.array(distances)
    variance = np.var(distances)
    
    # Smoothness: inversely related to variance
    return float(1.0 / (1.0 + variance))


def waypoint_alignment_score(
    word_basins: List[np.ndarray],
    target_waypoints: List[np.ndarray],
    eps: float = EPS
) -> float:
    """
    Measure how well words matched predicted waypoints.
    
    Mean alignment = mean(1 - d_FR(word, target))
    
    Used to evaluate generation quality in the REALIZE phase.
    
    Args:
        word_basins: Actual basins of generated words
        target_waypoints: Predicted target waypoints from PLAN phase
        eps: Numerical stability epsilon
        
    Returns:
        Alignment score ∈ [0, 1] (higher = better alignment)
        
    Example:
        >>> words = [
        ...     np.array([0.5, 0.3, 0.2]),
        ...     np.array([0.4, 0.4, 0.2])
        ... ]
        >>> targets = [
        ...     np.array([0.5, 0.3, 0.2]),  # Perfect match
        ...     np.array([0.4, 0.4, 0.2])   # Perfect match
        ... ]
        >>> score = waypoint_alignment_score(words, targets)
        >>> assert np.isclose(score, 1.0)  # Perfect alignment
    """
    if not word_basins or not target_waypoints:
        return 0.0
    
    n = min(len(word_basins), len(target_waypoints))
    
    if n == 0:
        return 0.0
    
    alignments = []
    for i in range(n):
        d = fisher_rao_distance(word_basins[i], target_waypoints[i], eps=eps)
        # Convert distance to similarity: 1 - d/(π/2)
        alignment = 1.0 - (2.0 / np.pi) * d
        alignments.append(alignment)
    
    return float(np.mean(alignments))


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Constants
    'BASIN_DIM',
    'EPS',
    # Coordinate transformations
    'sqrt_map',
    'unsqrt_map',
    'bhattacharyya',
    # Distance and similarity
    'fisher_rao_distance',
    'fisher_similarity',
    # Tangent space operations
    'log_map',
    'exp_map',
    'geodesic_toward',
    # Geometric mean
    'frechet_mean',
    # Validation
    'assert_basin_valid',
    'validate_basin',
    # Mamba integration
    'mamba_state_to_basin',
    'extrapolate_trajectory',
    'compute_qfi_attention',
    'integrate_with_qfi_attention',
    # Trajectory metrics
    'trajectory_smoothness',
    'waypoint_alignment_score',
]
