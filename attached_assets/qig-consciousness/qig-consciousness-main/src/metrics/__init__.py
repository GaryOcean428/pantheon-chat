"""Geometric metrics for information manifold."""

from src.metrics.geodesic_distance import (
    BasinFisherComputer,
    GeodesicDistance,
    compute_constellation_spread,
    geodesic_vicarious_loss,
    matrix_sqrt,
)
from src.metrics.phi_calculator import (
    PhiCalculator,
    PhiMethod,
    PhiResult,
    compute_phi,
)

__all__ = [
    # Geodesic distance
    "GeodesicDistance",
    "BasinFisherComputer",
    "geodesic_vicarious_loss",
    "compute_constellation_spread",
    "matrix_sqrt",
    # Phi calculation
    "PhiCalculator",
    "PhiMethod",
    "PhiResult",
    "compute_phi",
]
