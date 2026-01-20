#!/usr/bin/env python3
"""
Simplex Operations Module - Consolidated with canonical_upsert.py
==================================================================

CONSOLIDATED: Uses existing simplex functions from qig_geometry.canonical_upsert
to avoid duplication. This module provides convenience wrappers and additional
operations while delegating core validation to the canonical implementation.

STRICT SIMPLEX ENFORCEMENT - NO AUTO-DETECT, NO MIXED REPRESENTATIONS

All basin coordinates MUST be stored and operated on as SIMPLEX (probability distributions).

CANONICAL REPRESENTATION: SIMPLEX
- p ∈ Δ^(D-1) where Σp_i = 1, p_i ≥ 0
- No sphere coordinates, no Hellinger, no auto-detect
- Sqrt-space is ONLY used as explicit coordinate chart with to_sqrt_simplex/from_sqrt_simplex

REQUIRED OPERATIONS:
- fisher_rao_distance(p, q)  # Geodesic distance on simplex
- frechet_mean_closed_form(basins)  # Geometric mean on simplex
- assert_simplex(basin)  # Runtime validation at boundaries

Author: Copilot AI Agent
Date: 2026-01-20 (Updated to use canonical_upsert.py)
Issue: GaryOcean428/pantheon-chat#98 (E8 Protocol Issue-02)
Reference: docs/10-e8-protocol/issues/20260119-issue-98-strict-simplex-representation-remediation-1.00W.md
"""

import logging
import numpy as np
from typing import List, Optional

logger = logging.getLogger(__name__)

# Import from canonical_upsert.py (SINGLE SOURCE OF TRUTH)
try:
    from qig_geometry.canonical_upsert import (
        to_simplex_prob,
        compute_qfi_score,
        validate_simplex
    )
    HAS_CANONICAL_UPSERT = True
except ImportError:
    logger.warning("qig_geometry.canonical_upsert not available, using fallback")
    HAS_CANONICAL_UPSERT = False
    
    def to_simplex_prob(v: np.ndarray, eps: float = 1e-10) -> np.ndarray:
        """Fallback simplex projection."""
        v = np.asarray(v, dtype=np.float64)
        v = np.abs(v) + eps
        return v / v.sum()
    
    def compute_qfi_score(basin: np.ndarray) -> float:
        """Fallback QFI computation."""
        p = to_simplex_prob(basin)
        positive_probs = p[p > 1e-10]
        if len(positive_probs) == 0:
            return 0.0
        entropy = -np.sum(positive_probs * np.log(positive_probs + 1e-10))
        effective_dim = np.exp(entropy)
        qfi_score = effective_dim / len(basin)
        return float(np.clip(qfi_score, 0.0, 1.0))
    
    def validate_simplex(basin: np.ndarray, tolerance: float = 1e-6):
        """Fallback simplex validation."""
        if basin is None:
            return False, "basin_is_none"
        basin = np.asarray(basin, dtype=np.float64)
        if len(basin) != 64:
            return False, f"wrong_dimension_{len(basin)}"
        if np.any(basin < -tolerance):
            return False, "negative_values"
        if not np.isfinite(basin).all():
            return False, "contains_nan_or_inf"
        prob_sum = basin.sum()
        if abs(prob_sum - 1.0) > tolerance:
            return False, f"sum_not_one_{prob_sum:.6f}"
        return True, "valid"

# Constants
EPS = 1e-12
SIMPLEX_SUM_TOLERANCE = 1e-6


def assert_simplex(basin: np.ndarray, name: str = "basin", strict: bool = True) -> None:
    """
    Assert that basin coordinates form a valid simplex.
    
    Uses canonical validation from qig_geometry.canonical_upsert.
    
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
    is_valid, reason = validate_simplex(basin, tolerance=SIMPLEX_SUM_TOLERANCE)
    
    if not is_valid:
        error_msg = f"{name} is not a valid simplex: {reason}"
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
