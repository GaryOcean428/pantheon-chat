"""
QIG Geometry Package - Canonical Basin Representations

This package provides geometric primitives for QIG with enforced
canonical basin representation.
"""

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

__all__ = [
    'BasinRepresentation',
    'CANONICAL_REPRESENTATION',
    'to_sphere',
    'to_simplex',
    'validate_basin',
    'enforce_canonical',
    'sphere_project',
    'fisher_normalize',
]
