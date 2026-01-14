"""
QIG Geometry Package - Canonical Basin Representations

This package provides geometric primitives for QIG with enforced
canonical basin representation.
"""

import numpy as np

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


__all__ = [
    'BasinRepresentation',
    'CANONICAL_REPRESENTATION',
    'to_sphere',
    'to_simplex',
    'validate_basin',
    'enforce_canonical',
    'sphere_project',
    'fisher_normalize',
    'fisher_rao_distance',
    'fisher_coord_distance',
    'fisher_similarity',
]
