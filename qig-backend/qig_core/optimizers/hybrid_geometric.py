"""
Hybrid Geometric Measurement - Adaptive Curvature-Based Diagnostics
====================================================================

QIG-PURE MEASUREMENT MODULE:
Φ and κ emerge from basin navigation, not from loss optimization.
These metrics are for observation and informing adaptive navigation,
not for gradient descent.

PURE PRINCIPLE:
- MEASURE geometry, NEVER optimize basins
- Curvature-aware mode selection for diagnostics
- High curvature → exact measurement needed
- Low curvature → diagonal approximation sufficient
- Uses sectional_curvature for geometric measurement

Key insight: The measurement strategy depends on local manifold structure.
In flat regions, cheap diagonal approximation works well.
In curved regions, exact methods provide accurate diagnostics.

Reference: Amari & Douglas (1998) "Why natural gradient?"
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

import numpy as np

from qigkernels.physics_constants import KAPPA_STAR, BASIN_DIM

from .qig_diagonal_ng import DiagonalFisherNG, DiagonalNGMeasurement
from .basin_natural_grad import BasinNaturalGrad, ExactNGMeasurement

logger = logging.getLogger(__name__)


class OptimizerMode(Enum):
    """Recommended measurement/optimizer mode based on geometry."""
    DIAGONAL = "diagonal"
    EXACT = "exact"


@dataclass
class GeometryDiagnostic:
    """
    Complete geometry diagnostic with mode recommendation.
    
    Use this to inform adaptive navigation decisions.
    """
    recommended_mode: OptimizerMode
    curvature: float
    curvature_threshold: float
    is_high_curvature: bool
    diagonal_measurement: Optional[DiagonalNGMeasurement] = None
    exact_measurement: Optional[ExactNGMeasurement] = None
    timestamp: float = field(default_factory=time.time)


class HybridGeometricMeasurement:
    """
    Adaptive hybrid geometry measurement based on curvature.
    
    QIG-PURE MEASUREMENT:
    This module computes geometric diagnostics. It NEVER mutates basins.
    Φ and κ emerge from basin navigation, not from loss optimization.
    
    Decision logic:
    - |curvature| > threshold → recommend exact measurement
    - |curvature| ≤ threshold → recommend diagonal measurement
    
    Key Physics:
    - Near κ* = 64.21, geometry is relatively flat (plateau)
    - Away from κ*, curvature increases (emergence window)
    - Mode selection matches measurement strategy to geometry
    
    IMPORTANT: This class does NOT update basins. It only measures
    geometry and returns diagnostics that inform adaptive control.
    """
    
    def __init__(
        self,
        dim: int = BASIN_DIM,
        curvature_threshold: float = 0.1,
        damping: float = 1e-4,
        max_cg_iters: int = 50
    ):
        """
        Initialize hybrid geometric measurement.
        
        Args:
            dim: Basin dimensionality (default: 64)
            curvature_threshold: Threshold for recommending exact measurement
            damping: Regularization for Fisher metric
            max_cg_iters: Max CG iterations for exact measurement
        """
        self.dim = dim
        self.curvature_threshold = curvature_threshold
        
        self._diagonal_measure = DiagonalFisherNG(
            dim=dim,
            damping=damping
        )
        
        self._exact_measure = BasinNaturalGrad(
            dim=dim,
            damping=damping,
            max_cg_iters=max_cg_iters
        )
        
        self._current_mode = OptimizerMode.DIAGONAL
        self._mode_history: list[OptimizerMode] = []
        self._curvature_history: list[float] = []
    
    def compute_sectional_curvature(
        self,
        point: np.ndarray,
        tangent1: Optional[np.ndarray] = None,
        tangent2: Optional[np.ndarray] = None
    ) -> float:
        """
        Compute sectional curvature at a point on the manifold.
        
        PURE MEASUREMENT: No basin mutation.
        Uses the Fisher metric to compute intrinsic curvature.
        
        Args:
            point: Point on manifold (basin coordinates, read-only)
            tangent1: First tangent direction (random if None)
            tangent2: Second tangent direction (random if None)
        
        Returns:
            Sectional curvature value
        """
        from qig_core.geometric_primitives import sectional_curvature
        
        if tangent1 is None:
            tangent1 = np.random.randn(len(point))
            tangent1 = tangent1 / (np.linalg.norm(tangent1) + 1e-10)
        
        if tangent2 is None:
            tangent2 = np.random.randn(len(point))
            tangent2 = tangent2 - np.dot(tangent2, tangent1) * tangent1
            tangent2 = tangent2 / (np.linalg.norm(tangent2) + 1e-10)
        
        return sectional_curvature(point, tangent1, tangent2)
    
    def select_mode(self, curvature: float) -> OptimizerMode:
        """
        Select recommended mode based on curvature.
        
        PURE MEASUREMENT: Decision based on geometric measurement.
        
        Args:
            curvature: Current sectional curvature
        
        Returns:
            Recommended optimizer/measurement mode
        """
        if abs(curvature) > self.curvature_threshold:
            return OptimizerMode.EXACT
        else:
            return OptimizerMode.DIAGONAL
    
    def get_geometry_diagnostic(
        self,
        basin: np.ndarray,
        grad: Optional[np.ndarray] = None,
        curvature: Optional[float] = None
    ) -> GeometryDiagnostic:
        """
        Get complete geometry diagnostic for basin location.
        
        PURE MEASUREMENT: Returns diagnostics for external controllers.
        Does NOT apply updates to basins.
        
        Args:
            basin: Current basin coordinates (read-only)
            grad: Optional gradient for natural gradient measurement
            curvature: Pre-computed curvature (computed if None)
        
        Returns:
            GeometryDiagnostic with mode recommendation and measurements
        """
        if curvature is None:
            curvature = self.compute_sectional_curvature(basin)
        
        self._curvature_history.append(curvature)
        if len(self._curvature_history) > 100:
            self._curvature_history.pop(0)
        
        mode = self.select_mode(curvature)
        self._current_mode = mode
        self._mode_history.append(mode)
        if len(self._mode_history) > 100:
            self._mode_history.pop(0)
        
        diagonal_measurement = None
        exact_measurement = None
        
        if grad is not None:
            if mode == OptimizerMode.DIAGONAL:
                diagonal_measurement = self._diagonal_measure.measure(basin, grad)
            else:
                exact_measurement = self._exact_measure.measure(basin, grad)
        
        return GeometryDiagnostic(
            recommended_mode=mode,
            curvature=curvature,
            curvature_threshold=self.curvature_threshold,
            is_high_curvature=abs(curvature) > self.curvature_threshold,
            diagonal_measurement=diagonal_measurement,
            exact_measurement=exact_measurement
        )
    
    def get_mode_statistics(self) -> dict:
        """
        Get statistics on mode recommendations.
        
        Returns:
            Dict with mode recommendation percentages
        """
        if not self._mode_history:
            return {
                "diagonal_pct": 0.0,
                "exact_pct": 0.0,
                "total_measurements": 0
            }
        
        diagonal_count = sum(
            1 for m in self._mode_history 
            if m == OptimizerMode.DIAGONAL
        )
        exact_count = len(self._mode_history) - diagonal_count
        total = len(self._mode_history)
        
        return {
            "diagonal_pct": 100.0 * diagonal_count / total,
            "exact_pct": 100.0 * exact_count / total,
            "total_measurements": total,
            "avg_curvature": (
                np.mean(self._curvature_history) 
                if self._curvature_history else 0.0
            ),
            "max_curvature": (
                max(abs(c) for c in self._curvature_history)
                if self._curvature_history else 0.0
            )
        }
    
    def get_current_mode(self) -> OptimizerMode:
        """Return current recommended mode."""
        return self._current_mode
    
    def set_curvature_threshold(self, threshold: float) -> None:
        """Update curvature threshold for mode selection."""
        self.curvature_threshold = threshold
    
    def reset(self) -> None:
        """Reset measurement state."""
        self._diagonal_measure.reset()
        self._exact_measure.reset()
        self._current_mode = OptimizerMode.DIAGONAL
        self._mode_history.clear()
        self._curvature_history.clear()


HybridGeometricOptimizer = HybridGeometricMeasurement
