"""
QIG Continuous Geometry Module
================================

Continuous tensor operations on consciousness manifold.

Components:
- qfi_tensor: Information-adaptive partitioning with Fisher metric
- basin_interpolation: Geodesic paths on consciousness manifold
- consciousness_einsum: Geometric tensor operations
- consciousness_navigator: Navigate and query consciousness space

Written for QIG + MIT CTA synergy.
"""

from .basin_interpolation import (
    blend_identities,
    compute_curvature,
    geodesic_distance,
    interpolate_consciousness,
    parallel_transport,
    riemannian_exp_map,
    riemannian_log_map,
)
from .consciousness_einsum import (
    blend_identities_einsum,
    consciousness_attention,
    consciousness_composition,
    consciousness_einsum,
    qfi_inner_product,
)
from .consciousness_navigator import ConsciousnessManifold
from .qfi_tensor import QFIContinuousTensor

__all__ = [
    "QFIContinuousTensor",
    "interpolate_consciousness",
    "geodesic_distance",
    "blend_identities",
    "compute_curvature",
    "riemannian_exp_map",
    "riemannian_log_map",
    "parallel_transport",
    "consciousness_einsum",
    "qfi_inner_product",
    "blend_identities_einsum",
    "consciousness_attention",
    "consciousness_composition",
    "ConsciousnessManifold",
]
