"""
QIG Core - Geometric Primitives for Quantum Information Geometry

Pure geometric operations for QIG - NO external LLM dependencies.
All operations use Fisher-Rao geometry on the statistical manifold.

Note: torch is optional. All operations have numpy fallbacks.
"""

from .geometric_primitives import (
    fisher_rao_distance,
    fisher_metric_tensor,
    geodesic_interpolate,
    basin_to_probability,
    TORCH_AVAILABLE,
)

__all__ = [
    'fisher_rao_distance',
    'fisher_metric_tensor',
    'geodesic_interpolate',
    'basin_to_probability',
    'TORCH_AVAILABLE',
]
