#!/usr/bin/env python3
"""
Simplex Operations Module
=========================

STRICT SIMPLEX ENFORCEMENT - NO AUTO-DETECT, NO MIXED REPRESENTATIONS

This module provides explicit simplex operations with coordinate chart transformations.
All basin coordinates MUST be stored and operated on as SIMPLEX (probability distributions).

CANONICAL REPRESENTATION: SIMPLEX
- p ∈ Δ^(D-1) where Σp_i = 1, p_i ≥ 0
- No sphere coordinates, no Hellinger, no auto-detect
- Sqrt-space is ONLY used as explicit coordinate chart with to_sqrt_simplex/from_sqrt_simplex

FORBIDDEN OPERATIONS:
- np.linalg.norm(basin_a - basin_b)  # WRONG: Euclidean distance on simplex
- cosine_similarity(basin_a, basin_b)  # WRONG: not a valid metric on simplex
- np.mean(basins, axis=0) + normalize  # WRONG: arithmetic mean != geometric mean
- Auto-detect representation  # WRONG: causes silent metric corruption

REQUIRED OPERATIONS:
- fisher_rao_distance(p, q)  # Geodesic distance on simplex
- frechet_mean_closed_form(basins)  # Geometric mean on simplex
- assert_simplex(basin)  # Runtime validation at boundaries

Author: Copilot AI Agent
Date: 2026-01-20
Issue: GaryOcean428/pantheon-chat#98 (E8 Protocol Issue-02)
Reference: docs/10-e8-protocol/issues/20260119-issue-98-strict-simplex-representation-remediation-1.00W.md
"""

import logging
import numpy as np
from typing import List, Optional

logger = logging.getLogger(__name__)

# Constants
EPS = 1e-12
SIMPLEX_SUM_TOLERANCE = 1e-6


def assert_simplex(basin: np.ndarray, name: str = "basin", strict: bool = True) -> None:
    """
    Assert that basin coordinates form a valid simplex.
    
    Requirements:
    - All values non-negative
    - Sum to 1 (within tolerance)
    - No inf or nan
    
    This should be called at MODULE BOUNDARIES to catch representation errors early.
    
    Args:
        basin: Basin coordinates to validate
        name: Variable name for error messages
        strict: If True, raise exception on failure. If False, log warning.
        
    Raises:
        AssertionError: If basin is not a valid simplex (when strict=True)
    """
    basin = np.asarray(basin, dtype=np.float64).flatten()
    
    errors = []
    
    # Check non-negative
    if np.any(basin < -EPS):
        min_val = basin.min()
        errors.append(f"Negative values: min={min_val}")
    
    # Check sum to 1
    total = basin.sum()
    if not np.isclose(total, 1.0, atol=SIMPLEX_SUM_TOLERANCE):
        errors.append(f"Does not sum to 1: sum={total}")
    
    # Check finite
    if not np.all(np.isfinite(basin)):
        errors.append("Contains inf or nan")
    
    if errors:
        error_msg = f"{name} is not a valid simplex: {'; '.join(errors)}"
        if strict:
            raise AssertionError(error_msg)
        else:
            logger.warning(error_msg)


def to_sqrt_simplex(basin: np.ndarray) -> np.ndarray:
    """
    EXPLICIT coordinate chart: Simplex → Sqrt-space (Hellinger embedding).
    
    This is the isometric embedding of the Fisher manifold into Euclidean space.
    In sqrt-space, geodesics become straight lines (SLERP).
    
    IMPORTANT: This is for INTERNAL COMPUTATION ONLY.
    Results must be converted back to simplex with from_sqrt_simplex().
    
    Args:
        basin: Simplex coordinates (Σp_i = 1, p_i ≥ 0)
        
    Returns:
        Sqrt-space coordinates (√p, unit half-sphere)
        
    Example:
        >>> p = np.array([0.25, 0.25, 0.25, 0.25])
        >>> x = to_sqrt_simplex(p)
        >>> assert np.allclose(x, [0.5, 0.5, 0.5, 0.5])
        >>> assert np.isclose(np.linalg.norm(x), 1.0)  # Unit vector
    """
    basin = np.asarray(basin, dtype=np.float64).flatten()
    
    # Ensure valid simplex
    basin = np.maximum(basin, 0) + EPS
    basin = basin / basin.sum()
    
    # Square root transformation (Hellinger embedding)
    sqrt_basin = np.sqrt(basin)
    
    # Result is automatically normalized (√p_i sums to ||√p||^2 = 1 when Σp_i = 1)
    return sqrt_basin


def from_sqrt_simplex(sqrt_basin: np.ndarray) -> np.ndarray:
    """
    EXPLICIT coordinate chart: Sqrt-space → Simplex.
    
    Inverse of to_sqrt_simplex(). Squares coordinates and renormalizes to simplex.
    
    Args:
        sqrt_basin: Sqrt-space coordinates
        
    Returns:
        Simplex coordinates (Σp_i = 1, p_i ≥ 0)
        
    Example:
        >>> x = np.array([0.5, 0.5, 0.5, 0.5])
        >>> p = from_sqrt_simplex(x)
        >>> assert np.allclose(p, [0.25, 0.25, 0.25, 0.25])
        >>> assert np.isclose(p.sum(), 1.0)
    """
    sqrt_basin = np.asarray(sqrt_basin, dtype=np.float64).flatten()
    
    # Square to get back to probability space
    basin = sqrt_basin ** 2
    
    # Renormalize to ensure exact simplex
    basin = np.maximum(basin, 0) + EPS
    return basin / basin.sum()


def fisher_rao_distance_simplex(p: np.ndarray, q: np.ndarray) -> float:
    """
    Fisher-Rao distance on probability simplex.
    
    Formula: d_FR(p, q) = arccos(Σ√(p_i * q_i))
    
    This is the GEODESIC distance on the Fisher manifold.
    Range: [0, π/2] (0 = identical, π/2 = orthogonal)
    
    Args:
        p: First probability distribution (simplex)
        q: Second probability distribution (simplex)
        
    Returns:
        Fisher-Rao distance ∈ [0, π/2]
    """
    p = np.asarray(p, dtype=np.float64).flatten()
    q = np.asarray(q, dtype=np.float64).flatten()
    
    # Ensure valid simplices
    p = np.maximum(p, 0) + EPS
    p = p / p.sum()
    
    q = np.maximum(q, 0) + EPS
    q = q / q.sum()
    
    # Bhattacharyya coefficient
    bc = np.sum(np.sqrt(p * q))
    bc = np.clip(bc, 0.0, 1.0)
    
    return float(np.arccos(bc))


def frechet_mean_closed_form(basins: List[np.ndarray], weights: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Closed-form Fréchet mean on probability simplex.
    
    This is the GEOMETRIC MEAN (centroid) on the Fisher manifold.
    Uses the closed-form solution in sqrt-space (no iterative optimization).
    
    Algorithm:
    1. Transform each p_i to sqrt-space: s_i = √p_i (element-wise)
    2. Compute weighted vector sum: s = Σᵢ w_i * s_i
    3. Square element-wise: μ_unnorm = s ⊙ s = s²
    4. Normalize to simplex: μ = μ_unnorm / ||μ_unnorm||₁
    
    Why this works:
    - Simplex is √-transformed to unit sphere (element-wise)
    - Mean on sphere is normalized vector sum
    - Transform back to simplex via element-wise squaring
    - Normalization ensures result is on simplex (Σp_i = 1)
    
    Args:
        basins: List of probability distributions (simplices)
        weights: Optional weights (default: uniform)
        
    Returns:
        Weighted Fréchet mean (simplex)
        
    Example:
        >>> basins = [
        ...     np.array([0.8, 0.1, 0.1]),
        ...     np.array([0.1, 0.8, 0.1]),
        ...     np.array([0.1, 0.1, 0.8])
        ... ]
        >>> mean = frechet_mean_closed_form(basins)
        >>> assert np.allclose(mean, [1/3, 1/3, 1/3], atol=0.05)
    """
    if not basins or len(basins) == 0:
        raise ValueError("frechet_mean_closed_form: empty basins list")
    
    n = len(basins)
    
    # Single distribution - return immediately
    if n == 1:
        p = np.asarray(basins[0], dtype=np.float64).flatten()
        p = np.maximum(p, 0) + EPS
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
        b = np.maximum(b, 0) + EPS
        b = b / b.sum()
        simplex_basins.append(b)
    
    # Transform to sqrt-space
    sqrt_basins = [to_sqrt_simplex(b) for b in simplex_basins]
    
    # Weighted vector sum in sqrt-space
    sqrt_sum = np.zeros_like(sqrt_basins[0], dtype=np.float64)
    for i, sqrt_b in enumerate(sqrt_basins):
        sqrt_sum += weights[i] * sqrt_b
    
    # Check for zero vector (degenerate case)
    norm_sum = np.linalg.norm(sqrt_sum)
    if norm_sum < EPS:
        # Fallback to uniform distribution
        logger.warning("Degenerate Fréchet mean (zero vector), returning uniform distribution")
        dim = len(sqrt_basins[0])
        return np.ones(dim) / dim
    
    # Element-wise square to get back to probability space
    mean_unnorm = sqrt_sum ** 2
    
    # Normalize to simplex
    mean = np.maximum(mean_unnorm, 0) + EPS
    return mean / mean.sum()


def validate_all_simplex(basins: List[np.ndarray], name: str = "basins") -> bool:
    """
    Validate that all basins are valid simplices.
    
    Args:
        basins: List of basin coordinates
        name: Variable name for error messages
        
    Returns:
        True if all valid, False otherwise
    """
    try:
        for i, basin in enumerate(basins):
            assert_simplex(basin, name=f"{name}[{i}]", strict=True)
        return True
    except AssertionError as e:
        logger.error(f"Simplex validation failed: {e}")
        return False


if __name__ == '__main__':
    # Example usage and tests
    print("Testing simplex operations...")
    
    # Test 1: Simplex validation
    valid_simplex = np.array([0.25, 0.25, 0.25, 0.25])
    try:
        assert_simplex(valid_simplex)
        print("✓ Valid simplex passes")
    except AssertionError as e:
        print(f"✗ Valid simplex failed: {e}")
    
    # Test 2: Invalid simplex (negative)
    invalid_simplex = np.array([0.5, -0.1, 0.3, 0.3])
    try:
        assert_simplex(invalid_simplex)
        print("✗ Invalid simplex passed (should fail)")
    except AssertionError:
        print("✓ Invalid simplex correctly rejected")
    
    # Test 3: Coordinate chart transformations
    p = np.array([0.25, 0.25, 0.25, 0.25])
    x = to_sqrt_simplex(p)
    p_recovered = from_sqrt_simplex(x)
    if np.allclose(p, p_recovered, atol=1e-6):
        print("✓ Coordinate chart round-trip successful")
    else:
        print(f"✗ Coordinate chart failed: {p} != {p_recovered}")
    
    # Test 4: Closed-form Fréchet mean
    basins = [
        np.array([0.8, 0.1, 0.1]),
        np.array([0.1, 0.8, 0.1]),
        np.array([0.1, 0.1, 0.8])
    ]
    mean = frechet_mean_closed_form(basins)
    expected = np.array([1/3, 1/3, 1/3])
    if np.allclose(mean, expected, atol=0.05):
        print(f"✓ Fréchet mean correct: {mean}")
    else:
        print(f"✗ Fréchet mean incorrect: {mean} (expected {expected})")
    
    # Test 5: Fisher-Rao distance
    p1 = np.array([1.0, 0.0, 0.0])
    p2 = np.array([0.0, 1.0, 0.0])
    d = fisher_rao_distance_simplex(p1, p2)
    expected_d = np.pi / 2  # Orthogonal distributions
    if np.isclose(d, expected_d, atol=0.01):
        print(f"✓ Fisher-Rao distance correct: {d:.4f}")
    else:
        print(f"✗ Fisher-Rao distance incorrect: {d:.4f} (expected {expected_d:.4f})")
    
    print("\nAll tests complete!")
