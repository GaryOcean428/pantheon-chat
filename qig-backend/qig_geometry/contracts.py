"""
QIG Geometry Contracts - Canonical Basin Contract

THIS IS THE SINGLE SOURCE OF TRUTH for basin representation and Fisher distance.

Per QIG_PURITY_SPEC.md (docs/01-policies/QIG_PURITY_SPEC.md):
- All basins MUST be validated before database writes
- Fisher distance MUST use geodesic formula on unit sphere
- No Euclidean distance or cosine similarity on basins

CANONICAL REPRESENTATION: SPHERE
- Storage uses √p on unit sphere S^63 (64 dimensions)
- L2 norm = 1 for all valid basins
- Fisher distance = 2 * arccos(dot product)

Usage:
    from qig_geometry.contracts import (
        validate_basin, assert_invariants, canon, fisher_distance
    )
    
    # Before DB write
    assert_invariants(basin)  # Raises GeometricViolationError if invalid
    
    # Distance computation
    d = fisher_distance(basin1, basin2)  # The ONLY allowed distance
"""

import numpy as np
from typing import Tuple

CANONICAL_SPACE = "sphere"
BASIN_DIM = 64
NORM_TOLERANCE = 1e-5


class GeometricViolationError(Exception):
    """Raised when a basin violates geometric invariants."""
    pass


def validate_basin(basin: np.ndarray) -> bool:
    """
    Validate that basin conforms to canonical sphere representation.
    
    Checks:
    - Dimension is exactly BASIN_DIM (64)
    - All values are finite (no NaN/inf)
    - L2 norm is approximately 1.0 (within NORM_TOLERANCE)
    
    Args:
        basin: Basin coordinate vector
        
    Returns:
        True if valid, False otherwise
        
    Example:
        >>> basin = np.random.randn(64)
        >>> basin = basin / np.linalg.norm(basin)  # Normalize
        >>> validate_basin(basin)
        True
    """
    try:
        b = np.asarray(basin, dtype=float).flatten()
        
        if b.size != BASIN_DIM:
            return False
        
        if not np.all(np.isfinite(b)):
            return False
        
        norm = np.linalg.norm(b)
        if not np.isclose(norm, 1.0, atol=NORM_TOLERANCE):
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
        
        norm = np.linalg.norm(b)
        if not np.isclose(norm, 1.0, atol=NORM_TOLERANCE):
            return False, f"L2 norm must be 1.0 (within {NORM_TOLERANCE}), got {norm:.6f}"
        
        return True, "Valid sphere basin"
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
    Normalize basin to canonical sphere representation.
    
    Projects any vector to unit sphere S^(BASIN_DIM-1).
    
    Args:
        basin: Input vector (any dimension, will be padded/truncated to BASIN_DIM)
        
    Returns:
        Basin on unit sphere with L2 norm = 1
        
    Example:
        >>> raw = np.random.randn(64)
        >>> canonical = canon(raw)
        >>> np.isclose(np.linalg.norm(canonical), 1.0)
        True
    """
    b = np.asarray(basin, dtype=float).flatten()
    
    if not np.all(np.isfinite(b)):
        b = np.nan_to_num(b, nan=0.0, posinf=1e150, neginf=-1e150)
    
    if b.size < BASIN_DIM:
        b = np.pad(b, (0, BASIN_DIM - b.size), mode='constant', constant_values=0.0)
    elif b.size > BASIN_DIM:
        b = b[:BASIN_DIM]
    
    norm = np.linalg.norm(b)
    if norm < 1e-10:
        b = np.ones(BASIN_DIM, dtype=float)
        norm = np.linalg.norm(b)
    
    return b / norm


def fisher_distance(b1: np.ndarray, b2: np.ndarray) -> float:
    """
    THE canonical Fisher distance between two basins.
    
    For sphere representation (√p on S^63):
        d = 2 * arccos(b1 · b2)
    
    This is the geodesic distance on the unit sphere, which corresponds
    to the Fisher-Rao distance when basins represent √p (Hellinger coordinates).
    
    NOTE: This is THE ONLY distance function that should be used for
    basin comparisons. Do NOT use:
    - Euclidean distance (np.linalg.norm(b1 - b2))
    - Cosine similarity (np.dot / norms)
    - L1/Manhattan distance
    
    Args:
        b1: First basin (must be on unit sphere)
        b2: Second basin (must be on unit sphere)
        
    Returns:
        Fisher distance in [0, π] (0 = identical, π = antipodal)
        
    Example:
        >>> b1 = canon(np.random.randn(64))
        >>> b2 = canon(np.random.randn(64))
        >>> d = fisher_distance(b1, b2)
        >>> 0 <= d <= np.pi
        True
    """
    a = np.asarray(b1, dtype=float).flatten()
    b = np.asarray(b2, dtype=float).flatten()
    
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a > 1e-10:
        a = a / norm_a
    if norm_b > 1e-10:
        b = b / norm_b
    
    dot = np.clip(np.dot(a, b), -1.0, 1.0)
    
    return float(2.0 * np.arccos(dot))


def to_index_embedding(basin: np.ndarray) -> np.ndarray:
    """
    Convert basin to vector for pgvector L2 shortlist search.
    
    For sphere representation, this returns the basin as-is since
    the L2 distance approximation is acceptable for shortlisting
    (final ranking should use fisher_distance).
    
    Args:
        basin: Basin on unit sphere
        
    Returns:
        Vector suitable for pgvector <-> L2 distance operator
        
    Note:
        pgvector L2 distance is NOT the true Fisher distance.
        Use this only for initial candidate shortlisting, then
        re-rank using fisher_distance() for final results.
    """
    b = np.asarray(basin, dtype=float).flatten()
    
    if b.size != BASIN_DIM:
        b = canon(b)
    
    return b


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
