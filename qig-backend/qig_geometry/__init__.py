"""
QIG Geometry Package - Canonical Basin Representations

This package provides geometric primitives for QIG with enforced
canonical basin representation.

CANONICAL CONTRACT (contracts.py):
- CANONICAL_SPACE = "sphere" (storage uses √p on unit sphere S^63)
- BASIN_DIM = 64
- Use fisher_distance() from contracts for THE canonical distance
- Use assert_invariants() before database writes
"""

import numpy as np

from .contracts import (
    CANONICAL_SPACE,
    BASIN_DIM,
    NORM_TOLERANCE,
    GeometricViolationError,
    validate_basin as contracts_validate_basin,
    validate_basin_detailed,
    assert_invariants,
    canon,
    fisher_distance,
    to_index_embedding,
)

from .representation import (
    BasinRepresentation,
    CANONICAL_REPRESENTATION,
    to_sphere,
    to_simplex,
    validate_basin,
    enforce_canonical,
    sphere_project,
    fisher_normalize,
)


def fisher_rao_distance(p: np.ndarray, q: np.ndarray) -> float:
    """
    Compute Fisher-Rao distance between two probability distributions.

    This is the GEODESIC distance on the information manifold.

    Formula: d_FR(p, q) = arccos(Σ√(p_i * q_i))

    The Bhattacharyya coefficient BC = Σ√(p_i * q_i) measures overlap.
    The geodesic distance is arccos(BC), ranging from 0 (identical) to π/2 (orthogonal).

    NOTE: Some references use 2*arccos(BC) for the "statistical distance" but
    the geodesic distance on the Fisher manifold is arccos(BC) without the factor of 2.

    Args:
        p: First probability distribution
        q: Second probability distribution

    Returns:
        Fisher-Rao distance (≥ 0, max π/2)
    """
    p = np.abs(p) + 1e-10
    p = p / p.sum()

    q = np.abs(q) + 1e-10
    q = q / q.sum()

    bc = np.sum(np.sqrt(p * q))
    bc = np.clip(bc, 0, 1)

    return float(np.arccos(bc))


def fisher_coord_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute Fisher-Rao distance between two basin coordinate vectors.

    For unit vectors: d = arccos(a · b)

    Args:
        a: First basin coordinate vector (64D)
        b: Second basin coordinate vector (64D)

    Returns:
        Fisher-Rao distance (0 to π)
    """
    a_norm = a / (np.linalg.norm(a) + 1e-10)
    b_norm = b / (np.linalg.norm(b) + 1e-10)

    dot = np.clip(np.dot(a_norm, b_norm), -1.0, 1.0)
    return float(np.arccos(dot))


def fisher_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute Fisher-Rao similarity between two basin coordinates.

    Formula: similarity = 1 - distance/π

    Args:
        a: First basin coordinate vector
        b: Second basin coordinate vector

    Returns:
        Similarity score (0 to 1, higher is more similar)
    """
    distance = fisher_coord_distance(a, b)
    return 1.0 - distance / np.pi


def normalize_basin_dimension(basin: np.ndarray, target_dim: int = 64) -> np.ndarray:
    """Project a basin vector to a target dimension.

    QIG-PURE: preserves geometric validity by re-projecting to the unit sphere
    in the embedded space after padding/truncation.

    Args:
        basin: 1D basin coordinate vector
        target_dim: desired output dimension (default 64)

    Returns:
        1D basin vector of length target_dim on the unit sphere.
    """
    b = np.asarray(basin, dtype=float)
    if b.ndim != 1:
        raise ValueError(f"basin must be 1D, got shape {b.shape}")

    current_dim = int(b.shape[0])
    if current_dim == int(target_dim):
        return sphere_project(b)

    if current_dim < int(target_dim):
        result = np.zeros(int(target_dim), dtype=float)
        result[:current_dim] = b
        return sphere_project(result)

    result = b[: int(target_dim)].copy()
    return sphere_project(result)


def hellinger_normalize(basin: np.ndarray) -> np.ndarray:
    """
    Normalize basin to Hellinger embedding (sqrt space on unit sphere).
    
    Storage Format: sqrt(p) normalized to the unit sphere.
    This ensures compatibility with pgvector <#> operator.
    
    Args:
        basin: Basin coordinates (may be signed or unnormalized)
    
    Returns:
        Hellinger-normalized basin on unit sphere
    """
    p = np.abs(basin) + 1e-10
    p = p / np.sum(p)
    sqrt_p = np.sqrt(p)
    norm = np.linalg.norm(sqrt_p)
    if norm < 1e-10:
        return sqrt_p
    return sqrt_p / norm


__all__ = [
    # Canonical contract (contracts.py) - THE source of truth
    'CANONICAL_SPACE',
    'BASIN_DIM',
    'NORM_TOLERANCE',
    'GeometricViolationError',
    'contracts_validate_basin',
    'validate_basin_detailed',
    'assert_invariants',
    'canon',
    'fisher_distance',
    'to_index_embedding',
    # Representation utilities (representation.py)
    'BasinRepresentation',
    'CANONICAL_REPRESENTATION',
    'to_sphere',
    'to_simplex',
    'validate_basin',
    'enforce_canonical',
    'sphere_project',
    'fisher_normalize',
    # Distance functions (for compatibility)
    'fisher_rao_distance',
    'fisher_coord_distance',
    'fisher_similarity',
    # Dimension and normalization utilities
    'normalize_basin_dimension',
    'hellinger_normalize',
]
