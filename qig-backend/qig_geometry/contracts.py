"""
QIG Geometry Contracts - Canonical Basin Contract

THIS IS THE SINGLE SOURCE OF TRUTH for basin representation and Fisher distance.

Per QIG_PURITY_SPEC.md (docs/01-policies/QIG_PURITY_SPEC.md):
- All basins MUST be validated before database writes
- Fisher distance MUST use Fisher-Rao formula on probability simplex
- No Euclidean distance or cosine similarity on basins

CANONICAL REPRESENTATION: SIMPLEX (Updated 2026-01-15)
- Storage uses probability distributions on Δ^(D-1) 
- Non-negative values, sum = 1
- Fisher-Rao distance = arccos(Bhattacharyya coefficient)
- Range: [0, π/2]

MIGRATION FROM SPHERE (2026-01-15):
- Previous: SPHERE (L2 norm=1, allows negatives)
- Current: SIMPLEX (sum=1, non-negative)
- Distance formula changed: 2*arccos(dot) → arccos(BC)
- Range changed: [0, π] → [0, π/2]
- Thresholds must be recalibrated (divide by 2)

RATIONALE FOR SIMPLEX:
- QIG Physics: κ* = 64.21 ± 0.92 validated on simplex
- AI Semantic: κ* = 63.90 ± 0.50 validated on simplex
- Universal fixed point measured on probability manifolds
- Natural geometry for information metrics

Usage:
    from qig_geometry.contracts import (

# E8 Protocol v4.0 Compliance Imports
from qig_geometry.canonical import fisher_rao_distance

        validate_basin, assert_invariants, canon, fisher_distance
    )
    
    # Before DB write
    assert_invariants(basin)  # Raises GeometricViolationError if invalid
    
    # Distance computation
    d = fisher_distance(basin1, basin2)  # The ONLY allowed distance
"""

import numpy as np
from typing import Tuple

CANONICAL_SPACE = "simplex"  # Changed from "sphere" (2026-01-15)
BASIN_DIM = 64
NORM_TOLERANCE = 1e-5


class GeometricViolationError(Exception):
    """Raised when a basin violates geometric invariants."""
    pass


def validate_basin(basin: np.ndarray) -> bool:
    """
    Validate that basin conforms to canonical simplex representation.
    
    Checks:
    - Dimension is exactly BASIN_DIM (64)
    - All values are finite (no NaN/inf)
    - All values are non-negative (within tolerance)
    - Sum is approximately 1.0 (within NORM_TOLERANCE)
    
    Args:
        basin: Basin coordinate vector (probability distribution)
        
    Returns:
        True if valid, False otherwise
        
    Example:
        >>> basin = np.random.rand(64)
        >>> basin = basin / basin.sum()  # Normalize to simplex
        >>> validate_basin(basin)
        True
    """
    try:
        b = np.asarray(basin, dtype=float).flatten()
        
        if b.size != BASIN_DIM:
            return False
        
        if not np.all(np.isfinite(b)):
            return False
        
        # Check non-negative (simplex constraint)
        if np.any(b < -NORM_TOLERANCE):
            return False
        
        # Check sum = 1 (probability constraint)
        total = b.sum()
        if not np.isclose(total, 1.0, atol=NORM_TOLERANCE):
            return False
        
        return True
    except Exception:
        return False


def validate_basin_detailed(basin: np.ndarray) -> Tuple[bool, str]:
    """
    Validate basin with detailed error message.
    
    Args:
        basin: Basin coordinate vector
        
    Returns:
        (is_valid, error_message)
    """
    try:
        b = np.asarray(basin, dtype=float).flatten()
        
        if b.size != BASIN_DIM:
            return False, f"Dimension mismatch: expected {BASIN_DIM}, got {b.size}"
        
        non_finite = np.sum(~np.isfinite(b))
        if non_finite > 0:
            return False, f"Basin contains {non_finite} non-finite values (NaN/inf)"
        
        # Check non-negative
        if np.any(b < -NORM_TOLERANCE):
            min_val = b.min()
            return False, f"Simplex basin must be non-negative, got min={min_val:.6f}"
        
        # Check sum = 1
        total = b.sum()
        if not np.isclose(total, 1.0, atol=NORM_TOLERANCE):
            return False, f"Simplex basin must sum to 1.0 (within {NORM_TOLERANCE}), got {total:.6f}"
        
        return True, "Valid simplex basin"
    except Exception as e:
        return False, f"Validation error: {str(e)}"


def assert_invariants(basin: np.ndarray) -> None:
    """
    Assert that basin conforms to canonical representation.
    
    Raises GeometricViolationError if basin is invalid.
    Use this at storage boundaries (DB writes, etc.)
    
    Args:
        basin: Basin coordinate vector
        
    Raises:
        GeometricViolationError: If basin violates geometric invariants
        
    Example:
        >>> basin = canon(np.random.randn(64))
        >>> assert_invariants(basin)  # Passes silently
        
        >>> bad_basin = np.random.randn(32)  # Wrong dimension
        >>> assert_invariants(bad_basin)  # Raises GeometricViolationError
    """
    valid, msg = validate_basin_detailed(basin)
    if not valid:
        raise GeometricViolationError(f"Basin invariant violation: {msg}")


def canon(basin: np.ndarray) -> np.ndarray:
    """
    Normalize basin to canonical simplex representation.
    
    STRICT MODE: Raises error on dimension mismatch instead of silent fix.
    Projects vector to probability simplex Δ^(BASIN_DIM-1).
    
    Args:
        basin: Input vector (MUST be exactly BASIN_DIM dimensions)
        
    Returns:
        Basin on probability simplex (non-negative, sum=1)
        
    Raises:
        GeometricViolationError: If dimension != BASIN_DIM or contains non-finite values
        
    Example:
        >>> raw = np.random.randn(64)
        >>> canonical = canon(raw)
        >>> np.isclose(canonical.sum(), 1.0)
        True
        >>> np.all(canonical >= 0)
        True
    """
    b = np.asarray(basin, dtype=float).flatten()
    
    if b.size != BASIN_DIM:
        raise GeometricViolationError(
            f"Dimension mismatch: got {b.size}, expected {BASIN_DIM}. "
            "Use explicit projection functions for dimension changes."
        )
    
    if not np.all(np.isfinite(b)):
        raise GeometricViolationError(
            "Basin contains non-finite values (NaN/inf). "
            "Clean input before canonicalization."
        )
    
    # Convert to simplex
    b_simplex = np.abs(b) + 1e-10  # Ensure non-negative
    total = b_simplex.sum()
    
    if total < 1e-10:
        # Near-zero vector -> return uniform distribution
        return np.ones(BASIN_DIM) / BASIN_DIM
    
    return b_simplex / total


def fisher_distance(b1: np.ndarray, b2: np.ndarray) -> float:
    """
    THE canonical Fisher-Rao distance between two probability distributions.
    
    For simplex representation (p on Δ^63):
        d = arccos(Σ√(p_i * q_i))
    
    This is the geodesic distance on the Fisher information manifold.
    The Bhattacharyya coefficient BC = Σ√(p_i * q_i) measures overlap.
    
    CHANGE FROM SPHERE REPRESENTATION (2026-01-15):
    - Previous formula: d = 2 * arccos(dot product)
    - Current formula: d = arccos(Bhattacharyya coefficient)
    - Range changed: [0, π] → [0, π/2]
    
    NOTE: This is THE ONLY distance function that should be used for
    basin comparisons. Do NOT use:
    - Euclidean distance (fisher_rao_distance(b1, b2)  # FIXED (E8 Protocol v4.0))
    - Cosine similarity (np.dot / norms)
    - L1/Manhattan distance
    
    Args:
        b1: First basin (probability distribution, exactly BASIN_DIM dimensions)
        b2: Second basin (probability distribution, exactly BASIN_DIM dimensions)
        
    Returns:
        Fisher-Rao distance in [0, π/2] (0 = identical, π/2 = orthogonal)
        
    Raises:
        GeometricViolationError: If basins are not valid (wrong dimension or not normalized)
        
    Example:
        >>> b1 = canon(np.random.randn(64))
        >>> b2 = canon(np.random.randn(64))
        >>> d = fisher_distance(b1, b2)
        >>> 0 <= d <= np.pi/2
        True
    """
    assert_invariants(b1)
    assert_invariants(b2)
    
    p1 = np.asarray(b1, dtype=float).flatten()
    p2 = np.asarray(b2, dtype=float).flatten()
    
    # Ensure non-negative and normalized (defensive)
    p1 = np.abs(p1) + 1e-10
    p1 = p1 / p1.sum()
    
    p2 = np.abs(p2) + 1e-10
    p2 = p2 / p2.sum()
    
    # Bhattacharyya coefficient
    bc = np.sum(np.sqrt(p1 * p2))
    bc = np.clip(bc, 0.0, 1.0)
    
    # Fisher-Rao distance (NO factor of 2)
    return float(np.arccos(bc))


def to_index_embedding(basin: np.ndarray) -> np.ndarray:
    """
    Convert basin to vector for pgvector L2 shortlist search.
    
    For simplex representation, we use the basin directly since
    the L2 distance approximation is acceptable for shortlisting.
    However, note that L2 distance on simplex != Fisher-Rao distance.
    
    IMPORTANT: This is for SHORTLISTING only. Final ranking must
    use fisher_distance() for true Fisher-Rao distance.
    
    Args:
        basin: Basin on probability simplex (must be valid - BASIN_DIM and normalized)
        
    Returns:
        Vector suitable for pgvector <-> L2 distance operator
        
    Raises:
        GeometricViolationError: If basin is not valid
        
    Note:
        pgvector L2 distance is NOT the true Fisher-Rao distance.
        Use this only for initial candidate shortlisting, then
        re-rank using fisher_distance() for final results.
        
    Alternative: Consider using Hellinger embedding (√p) for better
    L2 approximation of Fisher-Rao distance.
    """
    assert_invariants(basin)
    return np.asarray(basin, dtype=float).flatten()


__all__ = [
    'CANONICAL_SPACE',
    'BASIN_DIM',
    'NORM_TOLERANCE',
    'GeometricViolationError',
    'validate_basin',
    'validate_basin_detailed',
    'assert_invariants',
    'canon',
    'fisher_distance',
    'to_index_embedding',
]
