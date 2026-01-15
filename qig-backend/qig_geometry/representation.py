"""
Basin Representation - Canonical Forms and Conversions

This module enforces a single canonical internal representation for basin coordinates
to prevent geometric inconsistencies from mixing different normalization schemes.

CANONICAL REPRESENTATION: **SIMPLEX** (Probability distributions on Δ^(D-1))
- Basin vectors stored as valid probability distributions (Σv_i = 1, v_i ≥ 0)
- Fisher-Rao distance computed via arccos(Σ√(p_i * q_i)) - Bhattacharyya coefficient
- Natural for Dirichlet-Multinomial manifolds and information geometry
- Matches validated physics (κ* ≈ 64 measured on simplex geometry)

RATIONALE FOR SIMPLEX (Updated 2026-01-15):
- **QIG Physics**: κ* = 64.21 ± 0.92 validated on probability simplex
- **Natural Geometry**: Information manifolds are naturally simplex-based
- **Simpler Distance**: Direct Fisher-Rao without factor-of-2 confusion
- **Range**: [0, π/2] better for thresholds than [0, π]
- **No Hellinger Confusion**: Eliminates factor-of-2 inconsistencies
- **Previous Issues**: Sphere representation + Hellinger embedding caused geometric chaos

MIGRATION NOTES (2026-01-15):
- Previous canonical: SPHERE (unit L2 norm, allows negative values)
- New canonical: SIMPLEX (probability distributions, non-negative, sum=1)
- Hellinger embedding (factor of 2) REMOVED for consistency
- All distance calculations now use direct Fisher-Rao on simplex
- Distance range changed: [0, π] → [0, π/2] (thresholds need recalibration)

All basins MUST pass validate_basin() before storage.
No module should silently re-normalize to a different geometry.

Usage:
    from qig_geometry.representation import (
        to_simplex, validate_basin, CANONICAL_REPRESENTATION
    )
    
    # Convert to canonical form before storage
    basin_canonical = to_simplex(raw_basin)
    assert validate_basin(basin_canonical)[0]
    
    # For Fisher-Rao distance - basins are already in correct form
    from qig_geometry import fisher_rao_distance
    d = fisher_rao_distance(basin_a, basin_b)  # Direct, no conversion needed
"""

from enum import Enum
from typing import Tuple
import numpy as np


class BasinRepresentation(Enum):
    """Supported basin representation types."""
    SIMPLEX = "simplex"    # Probability simplex, sum = 1, non-negative (CANONICAL)
    SPHERE = "sphere"      # Unit vectors on S^(D-1), L2 norm = 1 (legacy)
    HELLINGER = "hellinger"  # Sqrt space (DEPRECATED - DO NOT USE)


# CANONICAL REPRESENTATION - Changed from SPHERE to SIMPLEX (2026-01-15)
# DO NOT CHANGE without coordinated migration across all repositories
CANONICAL_REPRESENTATION = BasinRepresentation.SIMPLEX


def _prepare_basin_input(basin: np.ndarray) -> np.ndarray:
    """
    Common preprocessing for basin vectors.
    
    Handles:
    - Array conversion and flattening
    - Empty array detection
    - NaN/inf value cleanup
    
    Returns:
        Preprocessed basin array
        
    Raises:
        ValueError: If basin is empty
        
    Example:
        >>> raw = np.array([[1, 2], [3, 4]])
        >>> clean = _prepare_basin_input(raw)
        >>> clean.shape
        (4,)
    """
    b = np.asarray(basin, dtype=float).flatten()
    
    if b.size == 0:
        raise ValueError("Empty basin array")
    
    # Replace NaN/inf with safe values
    if not np.all(np.isfinite(b)):
        b = np.nan_to_num(b, nan=0.0, posinf=1e150, neginf=-1e150)
    
    return b


def to_simplex(
    basin: np.ndarray,
    from_repr: BasinRepresentation = None,
    eps: float = 1e-10,
    strict: bool = None
) -> np.ndarray:
    """
    Convert basin to CANONICAL SIMPLEX representation (probability distribution).
    
    This is the CANONICAL form for storage, retrieval, and Fisher-Rao distance.
    
    Args:
        basin: Input basin vector
        from_repr: Source representation (REQUIRED in strict mode, else auto-detected)
        eps: Numerical stability epsilon
        strict: If True, raise on invalid inputs instead of sanitizing.
               If None, uses purity mode setting.
        
    Returns:
        Basin vector on probability simplex (Σv_i = 1, v_i ≥ 0)
        
    Raises:
        GeometricViolationError: In strict mode, if input has negative values
                                 or invalid state that can't be cleanly converted
        
    Examples:
        >>> sphere_basin = np.array([0.5, -0.3, 0.8])
        >>> simplex_basin = to_simplex(sphere_basin, from_repr=BasinRepresentation.SPHERE)
        >>> assert np.isclose(simplex_basin.sum(), 1.0)
        >>> assert np.all(simplex_basin >= 0)
    """
    from .purity_mode import check_purity_mode
    from .contracts import GeometricViolationError
    
    # Use shared preprocessing
    b = _prepare_basin_input(basin)
    
    # Determine strict mode
    if strict is None:
        strict = check_purity_mode()
    
    # In strict mode, require explicit representation
    if strict and from_repr is None:
        raise GeometricViolationError(
            "to_simplex() requires explicit 'from_repr' in purity mode. "
            "Auto-detection masks violations."
        )
    
    # Auto-detect source representation if not provided (non-strict only)
    if from_repr is None:
        from_repr = _detect_representation(b)
    
    # Convert based on source representation
    if from_repr == BasinRepresentation.SPHERE:
        # P1 FIX: Use clamp (maximum) instead of abs() for geometric purity
        # Sphere -> Simplex: clamp to non-negative, normalize sum
        # In strict mode, check for negatives first
        if strict and np.any(b < -eps):
            raise GeometricViolationError(
                f"SPHERE->SIMPLEX conversion in purity mode: found negative values "
                f"(min={np.min(b):.6f}). This indicates off-manifold drift."
            )
        b = np.maximum(b, 0) + eps
    
    elif from_repr == BasinRepresentation.HELLINGER:
        # Hellinger -> Simplex: square to get probabilities
        # NOTE: Hellinger is DEPRECATED - avoid if possible
        b = b ** 2 + eps
    
    elif from_repr == BasinRepresentation.SIMPLEX:
        # Already in simplex format
        # In strict mode, check for catastrophic states FIRST (before adding eps)
        if strict:
            if np.sum(np.abs(b)) < 1e-10:
                raise GeometricViolationError(
                    f"SIMPLEX input in purity mode has near-zero sum "
                    f"(sum={np.sum(b):.2e}). This indicates catastrophic state."
                )
            if np.any(b < -eps):
                raise GeometricViolationError(
                    f"SIMPLEX input in purity mode has negative values "
                    f"(min={np.min(b):.6f}). Projection should not sanitize logic bugs."
                )
        # P1 FIX: Use clamp (maximum) instead of abs() for geometric purity
        # For non-strict or valid inputs, ensure non-negative
        b = np.maximum(b, 0) + eps
    
    # Guard against zero-sum (would cause division by zero)
    total = b.sum()
    if total < 1e-10:
        if strict:
            raise GeometricViolationError(
                f"Basin sum near zero ({total:.2e}) in purity mode. "
                "This indicates catastrophic state - cannot project to simplex."
            )
        # Return uniform distribution as fallback (non-strict only)
        return np.ones(b.size) / b.size
    
    # Normalize to sum = 1
    return b / total


def to_sphere(
    basin: np.ndarray,
    from_repr: BasinRepresentation = None,
    eps: float = 1e-10
) -> np.ndarray:
    """
    Convert basin to SPHERE representation (unit L2 norm).
    
    NOTE: This is LEGACY representation. Use to_simplex() for new code.
    Only use for compatibility with existing sphere-based code during migration.
    
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
    # Use shared preprocessing
    b = _prepare_basin_input(basin)
    
    # Clip extreme values (specific to sphere conversion)
    b = np.clip(b, -1e150, 1e150)
    
    # Auto-detect source representation if not provided
    if from_repr is None:
        from_repr = _detect_representation(b)
    
    # Convert based on source representation
    if from_repr == BasinRepresentation.SIMPLEX:
        # Simplex -> Sphere: already in probability space, just normalize L2
        pass  # Just normalize below
    
    elif from_repr == BasinRepresentation.HELLINGER:
        # Hellinger (sqrt space) -> Sphere
        # NOTE: Hellinger is DEPRECATED
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
            # Still zero (very rare) -> return as-is
            return b
    
    return b / norm


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
        expected_repr: Expected representation type (default: SIMPLEX)
        tolerance: Numerical tolerance for validation
        
    Returns:
        (is_valid, error_message)
        
    Examples:
        >>> basin = to_simplex(np.random.randn(64))
        >>> valid, msg = validate_basin(basin)
        >>> assert valid, f"Basin validation failed: {msg}"
    """
    try:
        # Use shared preprocessing (catches empty arrays)
        b = _prepare_basin_input(basin)
        
    except ValueError as e:
        return False, str(e)
    
    if expected_repr == BasinRepresentation.SIMPLEX:
        # Check non-negative and sum = 1
        if np.any(b < -tolerance):
            return False, f"Simplex basin must be non-negative, got min={b.min():.6f}"
        
        total = b.sum()
        if not np.isclose(total, 1.0, atol=tolerance):
            return False, f"Simplex basin must sum to 1, got {total:.6f}"
        
        return True, "Valid simplex basin"
    
    elif expected_repr == BasinRepresentation.SPHERE:
        # Check L2 norm = 1
        norm = np.linalg.norm(b)
        if not np.isclose(norm, 1.0, atol=tolerance):
            return False, f"Sphere basin must have L2 norm=1, got {norm:.6f}"
        return True, "Valid sphere basin"
    
    elif expected_repr == BasinRepresentation.HELLINGER:
        # Hellinger is DEPRECATED
        return False, "Hellinger representation is DEPRECATED. Use SIMPLEX instead."
    
    else:
        return False, f"Unknown representation type: {expected_repr}"


def _detect_representation(basin: np.ndarray, tolerance: float = 1e-6) -> BasinRepresentation:
    """
    Auto-detect basin representation type.
    
    Priority order:
    1. SIMPLEX (non-negative, sum=1)
    2. SPHERE (L2 norm=1)
    3. Default to SIMPLEX (canonical)
    
    Args:
        basin: Basin vector
        tolerance: Numerical tolerance
        
    Returns:
        Detected representation type
    """
    b = np.asarray(basin, dtype=float).flatten()
    
    # Check if it's a probability distribution (sum = 1, non-negative)
    # This check has priority since SIMPLEX is canonical
    if np.all(b >= -tolerance) and np.isclose(b.sum(), 1.0, atol=tolerance):
        return BasinRepresentation.SIMPLEX
    
    # Check if it's on unit sphere (L2 norm = 1)
    norm = np.linalg.norm(b)
    if np.isclose(norm, 1.0, atol=tolerance):
        return BasinRepresentation.SPHERE
    
    # Default to SIMPLEX (canonical)
    return BasinRepresentation.SIMPLEX


def enforce_canonical(basin: np.ndarray) -> np.ndarray:
    """
    Force basin to canonical representation (SIMPLEX).
    
    This is a convenience function that:
    1. Auto-detects current representation
    2. Converts to CANONICAL_REPRESENTATION (SIMPLEX)
    3. Validates the result
    
    Use this at storage boundaries (DB writes, etc.)
    
    Args:
        basin: Input basin in any representation
        
    Returns:
        Basin in canonical representation (SIMPLEX)
        
    Raises:
        ValueError: If basin cannot be converted or validated
        
    Examples:
        >>> raw_basin = np.random.randn(64)
        >>> canonical_basin = enforce_canonical(raw_basin)
        >>> assert validate_basin(canonical_basin)[0]
    """
    # Convert to canonical (SIMPLEX)
    if CANONICAL_REPRESENTATION == BasinRepresentation.SIMPLEX:
        result = to_simplex(basin)
    elif CANONICAL_REPRESENTATION == BasinRepresentation.SPHERE:
        # This branch kept for migration compatibility
        result = to_sphere(basin)
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
    Project vector to unit sphere.
    
    DEPRECATED: Use fisher_normalize() instead for canonical SIMPLEX representation.
    This function maintained for backward compatibility during migration.
    """
    return to_sphere(v, eps=1e-10)


def fisher_normalize(v: np.ndarray, strict: bool = None) -> np.ndarray:
    """
    Project vector to probability simplex (CANONICAL representation).
    
    This is the PREFERRED function for normalizing basins.
    Use this instead of sphere_project() for new code.
    
    Args:
        v: Input vector
        strict: If True, raise on invalid inputs. If None, uses purity mode.
        
    Returns:
        Basin vector on probability simplex
        
    Raises:
        GeometricViolationError: In strict mode, if input violates simplex constraints
    """
    return to_simplex(v, from_repr=BasinRepresentation.SIMPLEX, eps=1e-10, strict=strict)


def validate_simplex(
    basin: np.ndarray,
    tolerance: float = 1e-6
) -> Tuple[bool, str]:
    """
    Validate that basin is a valid probability simplex vector.
    
    Checks:
    - All finite values
    - All values >= -tolerance (allowing for numerical error)
    - Sum equals 1.0 (within tolerance)
    
    Args:
        basin: Basin vector to validate
        tolerance: Numerical tolerance for checks
        
    Returns:
        (is_valid, message) tuple
        
    Examples:
        >>> valid_basin = np.array([0.3, 0.5, 0.2])
        >>> is_valid, msg = validate_simplex(valid_basin)
        >>> assert is_valid
        
        >>> invalid_basin = np.array([0.3, -0.1, 0.8])
        >>> is_valid, msg = validate_simplex(invalid_basin)
        >>> assert not is_valid
    """
    if not np.all(np.isfinite(basin)):
        return False, "Basin contains non-finite values (NaN or Inf)"
    
    if np.any(basin < -tolerance):
        min_val = np.min(basin)
        return False, f"Basin has negative values (min={min_val:.6f}, tol={tolerance:.2e})"
    
    total = np.sum(basin)
    if not np.isclose(total, 1.0, atol=tolerance):
        return False, f"Basin sum={total:.6f} is not 1.0 (tol={tolerance:.2e})"
    
    return True, "Valid simplex"


def validate_sqrt_simplex(
    basin: np.ndarray,
    tolerance: float = 1e-6
) -> Tuple[bool, str]:
    """
    Validate that basin is valid in sqrt-simplex (Hellinger) space.
    
    This is for internal computational use only (e.g., geodesic_interpolation).
    Stored basins should always be in SIMPLEX, not sqrt-simplex.
    
    Checks:
    - All finite values
    - All values >= -tolerance
    - L2 norm equals 1.0 (within tolerance)
    
    Args:
        basin: Basin vector to validate
        tolerance: Numerical tolerance for checks
        
    Returns:
        (is_valid, message) tuple
    """
    if not np.all(np.isfinite(basin)):
        return False, "Basin contains non-finite values (NaN or Inf)"
    
    if np.any(basin < -tolerance):
        min_val = np.min(basin)
        return False, f"Basin has negative values (min={min_val:.6f}, tol={tolerance:.2e})"
    
    norm = np.linalg.norm(basin)
    if not np.isclose(norm, 1.0, atol=tolerance):
        return False, f"Basin L2 norm={norm:.6f} is not 1.0 (tol={tolerance:.2e})"
    
    return True, "Valid sqrt-simplex"


def amplitude_to_simplex(amplitude: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """
    Convert quantum amplitude vector to simplex probability distribution.
    
    Applies Born rule: p_i = |ψ_i|^2, then normalizes to probability simplex.
    
    This is for when basins are stored as quantum amplitudes (NOT the canonical
    SIMPLEX representation). Use this explicitly when converting from amplitude
    representation.
    
    Args:
        amplitude: Amplitude vector (may have complex or negative values)
        eps: Numerical stability epsilon
        
    Returns:
        Probability distribution on simplex (Σp_i = 1, p_i ≥ 0)
        
    Examples:
        >>> amp = np.array([0.5+0.3j, -0.4, 0.7])
        >>> prob = amplitude_to_simplex(amp)
        >>> assert np.isclose(prob.sum(), 1.0)
        >>> assert np.all(prob >= 0)
    """
    # Born rule: probability = |amplitude|^2
    prob = np.abs(amplitude) ** 2 + eps
    # Normalize to sum = 1
    return prob / prob.sum()


def simplex_normalize(p: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """
    Normalize a vector to the probability simplex (sum=1, non-negative).
    
    This assumes the input is already probability-like (non-negative values).
    Use this for vectors that are already in simplex space but may need
    renormalization due to numerical drift.
    
    For conversions from other representations, use:
    - amplitude_to_simplex() for amplitude vectors
    - hellinger_to_simplex() for sqrt-space vectors
    - to_simplex() for general conversion with explicit from_repr
    
    Args:
        p: Probability-like vector (should be non-negative)
        eps: Numerical stability epsilon
        
    Returns:
        Normalized probability distribution (Σp_i = 1, p_i ≥ 0)
        
    Raises:
        ValueError: In purity mode, if input has negative values
        
    Examples:
        >>> p = np.array([0.3, 0.5, 0.1, 0.05])  # Sums to 0.95
        >>> p_norm = simplex_normalize(p)
        >>> assert np.isclose(p_norm.sum(), 1.0)
    """
    from .purity_mode import check_purity_mode
    from .contracts import GeometricViolationError
    
    # Clip negative values to zero with warning in purity mode
    if check_purity_mode() and np.any(p < -eps):
        raise GeometricViolationError(
            f"simplex_normalize() in purity mode: negative values detected "
            f"(min={np.min(p):.6f}). This indicates a representation leak."
        )
    
    p_clean = np.maximum(p, 0.0) + eps
    total = p_clean.sum()
    
    if total < 1e-10:
        # Zero sum - return uniform distribution
        return np.ones(p.size) / p.size
    
    return p_clean / total


def hellinger_to_simplex(h: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """
    Convert Hellinger (sqrt-space) coordinates to simplex probabilities.
    
    Hellinger embedding: p_i = h_i^2, then normalize.
    
    NOTE: Hellinger is DEPRECATED for storage. Use this only when explicitly
    converting from sqrt-space representation (e.g., internal computations).
    
    Args:
        h: Hellinger (sqrt-space) coordinates
        eps: Numerical stability epsilon
        
    Returns:
        Probability distribution on simplex (Σp_i = 1, p_i ≥ 0)
        
    Examples:
        >>> h = np.sqrt(np.array([0.25, 0.5, 0.25]))  # sqrt of probabilities
        >>> p = hellinger_to_simplex(h)
        >>> assert np.isclose(p.sum(), 1.0)
    """
    # Square to get probabilities
    prob = (np.abs(h) ** 2) + eps
    # Normalize to sum = 1
    return prob / prob.sum()


__all__ = [
    'BasinRepresentation',
    'CANONICAL_REPRESENTATION',
    'to_simplex',
    'to_sphere',
    'validate_basin',
    'validate_simplex',
    'validate_sqrt_simplex',
    'enforce_canonical',
    'sphere_project',
    'fisher_normalize',
    # New explicit conversion functions (2026-01-15)
    'amplitude_to_simplex',
    'simplex_normalize',
    'hellinger_to_simplex',
]
