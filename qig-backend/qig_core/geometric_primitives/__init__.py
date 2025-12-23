"""
Geometric Primitives Package

Canonical implementations of QIG-pure geometric operations.

IMPORTANT: All geometric distance operations MUST use fisher_rao_distance.
DO NOT use np.linalg.norm() or cosine_similarity() on basin coordinates.
"""

# Import from existing fisher_metric if it exists
try:
    from .fisher_metric import (
        FisherMetric,
        compute_fisher_information,
        bures_distance,
        density_matrix_from_basin,
    )
    FISHER_METRIC_AVAILABLE = True
except ImportError:
    FISHER_METRIC_AVAILABLE = False

# Import canonical Fisher-Rao implementation
from .canonical_fisher import (
    fisher_rao_distance,
    geodesic_interpolate,
    find_nearest_basins,
    validate_basin,
)

__all__ = [
    # Canonical distance (USE THIS)
    'fisher_rao_distance',
    'geodesic_interpolate',
    'find_nearest_basins',
    'validate_basin',
]

# Add optional exports if available
if FISHER_METRIC_AVAILABLE:
    __all__.extend([
        'FisherMetric',
        'compute_fisher_information',
        'bures_distance',
        'density_matrix_from_basin',
    ])
