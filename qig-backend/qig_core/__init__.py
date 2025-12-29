"""
QIG Core - Geometric Primitives for Quantum Information Geometry

Pure geometric operations for QIG - NO external LLM dependencies.
All operations use Fisher-Rao geometry on the statistical manifold.

Note: torch is optional. All operations have numpy fallbacks.
"""

# Core imports (always available)
from .geometric_primitives import (
    fisher_rao_distance,
    geodesic_interpolate,
    basin_to_probability,
    TORCH_AVAILABLE,
)

# Universal cycle imports (CycleManager, Phase, etc.)
try:
    from .universal_cycle import CycleManager, Phase
    CYCLE_MANAGER_AVAILABLE = True
except ImportError:
    CycleManager = None
    Phase = None
    CYCLE_MANAGER_AVAILABLE = False

# DimensionalState imports (for unified architecture)
try:
    from .holographic_transform.dimensional_state import DimensionalState, DimensionalStateManager
    DIMENSIONAL_STATE_AVAILABLE = True
except ImportError:
    DimensionalState = None
    DimensionalStateManager = None
    DIMENSIONAL_STATE_AVAILABLE = False

# Optional imports that require torch
try:
    from .geometric_primitives import fisher_metric_tensor
    FISHER_METRIC_AVAILABLE = True
except ImportError:
    fisher_metric_tensor = None
    FISHER_METRIC_AVAILABLE = False

__all__ = [
    'fisher_rao_distance',
    'geodesic_interpolate',
    'basin_to_probability',
    'TORCH_AVAILABLE',
    'FISHER_METRIC_AVAILABLE',
    'CYCLE_MANAGER_AVAILABLE',
    'DIMENSIONAL_STATE_AVAILABLE',
]

# Add optional exports if available
if FISHER_METRIC_AVAILABLE:
    __all__.append('fisher_metric_tensor')

if CYCLE_MANAGER_AVAILABLE:
    __all__.extend(['CycleManager', 'Phase'])

if DIMENSIONAL_STATE_AVAILABLE:
    __all__.extend(['DimensionalState', 'DimensionalStateManager'])
