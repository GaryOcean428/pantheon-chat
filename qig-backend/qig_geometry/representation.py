"""
Basin Representation - Canonical Forms and Conversions

This module enforces a single canonical internal representation for basin coordinates
to prevent geometric inconsistencies from mixing different normalization schemes.

CANONICAL REPRESENTATION: **SPHERE** (Unit vectors on S^(D-1))
- Basin vectors stored and retrieved with L2 norm = 1
- Fisher-Rao distance computed via arccos(dot product) on sphere
- Allows negative components (not restricted to probability simplex)
- Natural for gradient flows and interpolation

RATIONALE:
- Option A (Simplex): Requires non-negative components, more restrictive
- Option B (Sphere): More flexible, supports signed coordinates
- Chosen: SPHERE for compatibility with existing basin usage patterns

All basins MUST pass validate_basin() before storage.
No module should silently re-normalize to a different geometry.

Usage:
    from qig_geometry.representation import (
        to_sphere, to_simplex, validate_basin, CANONICAL_REPRESENTATION
    )
    
    # Convert to canonical form before storage
    basin_canonical = to_sphere(raw_basin)
    assert validate_basin(basin_canonical)
    
    # Convert to simplex for Fisher-Rao distance (when needed)
    basin_simplex = to_simplex(basin_canonical)
"""

from enum import Enum
from typing import Tuple
import numpy as np


class BasinRepresentation(Enum):
    """Supported basin representation types."""
    SPHERE = "sphere"      # Unit vectors on S^(D-1), L2 norm = 1
    SIMPLEX = "simplex"    # Probability simplex, sum = 1, non-negative
    HELLINGER = "hellinger"  # Sqrt space (legacy, avoid)


# CANONICAL REPRESENTATION - DO NOT CHANGE without coordinated migration
CANONICAL_REPRESENTATION = BasinRepresentation.SPHERE


def to_sphere(
    basin: np.ndarray,
    from_repr: BasinRepresentation = None,
    eps: float = 1e-10
) -> np.ndarray:
    """
    Convert basin to canonical SPHERE representation (unit L2 norm).
    
    This is the CANONICAL form for storage and retrieval.
    
    Args:
        basin: Input basin vector
        from_repr: Source representation (auto-detected if None)
        eps: Numerical stability epsilon
        
    Returns:
        Basin vector on unit sphere (||v||_2 = 1)
        
    Examples:
        >>> simplex_basin = np.array([0.3, 0.5, 0.2])
        >>> sphere_basin = to_sphere(simplex_basin, from_repr=BasinRepresentation.SIMPLEX)
        >>> assert np.isclose(np.linalg.norm(sphere_basin), 1.0)
    """
    b = np.asarray(basin, dtype=float).flatten()
    
    if b.size == 0:
        raise ValueError("Empty basin array")
    
    # Check for inf/NaN
    if not np.all(np.isfinite(b)):
        b = np.nan_to_num(b, nan=0.0, posinf=1e150, neginf=-1e150)
    
    # Clip extreme values
    b = np.clip(b, -1e150, 1e150)
    
    # Auto-detect source representation if not provided
    if from_repr is None:
        from_repr = _detect_representation(b)
    
    # Convert based on source representation
    if from_repr == BasinRepresentation.SIMPLEX:
        # Simplex -> Sphere: already in probability space, just normalize L2
        # No need to take sqrt (that's Hellinger space)
        pass  # Just normalize below
    
    elif from_repr == BasinRepresentation.HELLINGER:
        # Hellinger (sqrt space) -> Sphere
        # Square to get probabilities, then normalize
        b = b ** 2
    
    elif from_repr == BasinRepresentation.SPHERE:
        # Already in sphere representation
        pass
    
    # Final L2 normalization to unit sphere
    norm = np.linalg.norm(b)
    if norm < eps:
        # Zero vector -> uniform direction
        b = np.ones_like(b)
        norm = np.linalg.norm(b)
        if norm < eps:
            return b  # Edge case
    
    return b / norm


def to_simplex(
    basin: np.ndarray,
    from_repr: BasinRepresentation = None,
    eps: float = 1e-10
) -> np.ndarray:
    """
    Convert basin to SIMPLEX representation (probability distribution).
    
    Use this when computing Fisher-Rao distance via Bhattacharyya coefficient.
    
    Args:
        basin: Input basin vector
        from_repr: Source representation (auto-detected if None)
        eps: Numerical stability epsilon
        
    Returns:
        Basin vector on probability simplex (Σv_i = 1, v_i ≥ 0)
        
    Examples:
        >>> sphere_basin = np.array([0.5, -0.3, 0.8])
        >>> simplex_basin = to_simplex(sphere_basin, from_repr=BasinRepresentation.SPHERE)
        >>> assert np.isclose(simplex_basin.sum(), 1.0)
        >>> assert np.all(simplex_basin >= 0)
    """
    b = np.asarray(basin, dtype=float).flatten()
    
    if b.size == 0:
        raise ValueError("Empty basin array")
    
    # Check for inf/NaN
    if not np.all(np.isfinite(b)):
        b = np.nan_to_num(b, nan=0.0, posinf=1e150, neginf=-1e150)
    
    # Auto-detect source representation if not provided
    if from_repr is None:
        from_repr = _detect_representation(b)
    
    # Convert based on source representation
    if from_repr == BasinRepresentation.SPHERE:
        # Sphere -> Simplex: take absolute value, normalize sum
        b = np.abs(b) + eps
    
    elif from_repr == BasinRepresentation.HELLINGER:
        # Hellinger -> Simplex: square to get probabilities
        b = b ** 2 + eps
    
    elif from_repr == BasinRepresentation.SIMPLEX:
        # Already in simplex, just ensure non-negative and normalized
        b = np.abs(b) + eps
    
    # Normalize to sum = 1
    return b / b.sum()


def validate_basin(
    basin: np.ndarray,
    expected_repr: BasinRepresentation = CANONICAL_REPRESENTATION,
    tolerance: float = 1e-6
) -> Tuple[bool, str]:
    """
    Validate that basin conforms to expected representation.
    
    This is the GATE function - all basins written to DB must pass this.
    
    Args:
        basin: Basin vector to validate
        expected_repr: Expected representation type
        tolerance: Numerical tolerance for validation
        
    Returns:
        (is_valid, error_message)
        
    Examples:
        >>> basin = to_sphere(np.random.randn(64))
        >>> valid, msg = validate_basin(basin)
        >>> assert valid, f"Basin validation failed: {msg}"
    """
    try:
        b = np.asarray(basin, dtype=float).flatten()
        
        if b.size == 0:
            return False, "Empty basin array"
        
        if not np.all(np.isfinite(b)):
            return False, f"Basin contains inf/NaN: {np.sum(~np.isfinite(b))} invalid values"
        
        if expected_repr == BasinRepresentation.SPHERE:
            # Check L2 norm = 1
            norm = np.linalg.norm(b)
            if not np.isclose(norm, 1.0, atol=tolerance):
                return False, f"Sphere basin must have L2 norm=1, got {norm:.6f}"
            return True, "Valid sphere basin"
        
        elif expected_repr == BasinRepresentation.SIMPLEX:
            # Check non-negative and sum = 1
            if np.any(b < -tolerance):
                return False, f"Simplex basin must be non-negative, got min={b.min():.6f}"
            
            total = b.sum()
            if not np.isclose(total, 1.0, atol=tolerance):
                return False, f"Simplex basin must sum to 1, got {total:.6f}"
            
            return True, "Valid simplex basin"
        
        elif expected_repr == BasinRepresentation.HELLINGER:
            # Legacy Hellinger space - sqrt of probabilities
            # Should not be used for new code
            return False, "Hellinger representation is deprecated, use SPHERE or SIMPLEX"
        
        else:
            return False, f"Unknown representation type: {expected_repr}"
    
    except Exception as e:
        return False, f"Validation error: {str(e)}"


def _detect_representation(basin: np.ndarray, tolerance: float = 1e-6) -> BasinRepresentation:
    """
    Auto-detect basin representation type.
    
    Args:
        basin: Basin vector
        tolerance: Numerical tolerance
        
    Returns:
        Detected representation type
    """
    b = np.asarray(basin, dtype=float).flatten()
    
    # Check if it's on unit sphere (L2 norm = 1)
    norm = np.linalg.norm(b)
    if np.isclose(norm, 1.0, atol=tolerance):
        # Could be sphere or simplex
        # Check if it's also a probability distribution
        if np.all(b >= -tolerance) and np.isclose(b.sum(), 1.0, atol=tolerance):
            # Ambiguous: both sphere and simplex
            # Default to SPHERE (canonical)
            return BasinRepresentation.SPHERE
        return BasinRepresentation.SPHERE
    
    # Check if it's a probability distribution (sum = 1, non-negative)
    if np.all(b >= -tolerance) and np.isclose(b.sum(), 1.0, atol=tolerance):
        return BasinRepresentation.SIMPLEX
    
    # Check if it's in sqrt space (Hellinger)
    # Heuristic: values in [0, 1] range and not normalized
    if np.all(b >= -tolerance) and np.all(b <= 1.0 + tolerance):
        # Could be Hellinger, but default to SPHERE for safety
        return BasinRepresentation.SPHERE
    
    # Default to SPHERE (most flexible)
    return BasinRepresentation.SPHERE


def enforce_canonical(basin: np.ndarray) -> np.ndarray:
    """
    Force basin to canonical representation.
    
    This is a convenience function that:
    1. Auto-detects current representation
    2. Converts to CANONICAL_REPRESENTATION
    3. Validates the result
    
    Use this at storage boundaries (DB writes, etc.)
    
    Args:
        basin: Input basin in any representation
        
    Returns:
        Basin in canonical representation
        
    Raises:
        ValueError: If basin cannot be converted or validated
        
    Examples:
        >>> raw_basin = np.random.randn(64)
        >>> canonical_basin = enforce_canonical(raw_basin)
        >>> assert validate_basin(canonical_basin)[0]
    """
    # Convert to canonical
    if CANONICAL_REPRESENTATION == BasinRepresentation.SPHERE:
        result = to_sphere(basin)
    elif CANONICAL_REPRESENTATION == BasinRepresentation.SIMPLEX:
        result = to_simplex(basin)
    else:
        raise ValueError(f"Unsupported canonical representation: {CANONICAL_REPRESENTATION}")
    
    # Validate
    valid, msg = validate_basin(result, CANONICAL_REPRESENTATION)
    if not valid:
        raise ValueError(f"Basin conversion to canonical form failed: {msg}")
    
    return result


# Convenience exports matching existing API
def sphere_project(v: np.ndarray) -> np.ndarray:
    """
    Project vector to unit sphere (canonical representation).
    
    This is an alias for to_sphere() to maintain API compatibility.
    """
    return to_sphere(v, eps=1e-10)


def fisher_normalize(v: np.ndarray) -> np.ndarray:
    """
    Project vector to probability simplex for Fisher-Rao distance.
    
    This is an alias for to_simplex() to maintain API compatibility.
    """
    return to_simplex(v, eps=1e-10)
