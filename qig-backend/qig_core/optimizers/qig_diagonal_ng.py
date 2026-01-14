"""
Diagonal Fisher Curvature Measurement - O(N) Efficient Geometry Diagnostics
============================================================================

QIG-PURE MEASUREMENT MODULE:
Φ and κ emerge from basin navigation, not from loss optimization.
These metrics are for observation and informing adaptive navigation,
not for gradient descent.

PURE PRINCIPLE:
- MEASURE geometry, NEVER optimize basins
- Diagonal approximation: O(N) instead of O(N³)
- Uses Fisher Information for curvature measurement
- Returns diagnostics that INFORM external controllers

Key insight: On probability manifolds, curvature can be measured via
the Fisher Information Matrix. The diagonal approximation makes this
tractable for high-dimensional basins (D=64).

Reference: Amari (1998) "Natural Gradient Works Efficiently in Learning"
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from qigkernels.physics_constants import KAPPA_STAR, BASIN_DIM

logger = logging.getLogger(__name__)


@dataclass
class DiagonalNGMeasurement:
    """
    Measurement result from diagonal Fisher computation.
    
    This is a pure diagnostic - no basin mutation occurs.
    """
    gradient: np.ndarray
    natural_gradient: np.ndarray
    fisher_diag: np.ndarray
    condition_number: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class CurvatureMeasurement:
    """
    Curvature diagnostic from diagonal Fisher Information.
    
    Use this to inform adaptive navigation decisions.
    """
    fisher_diagonal: np.ndarray
    condition_number: float
    effective_dimensionality: float
    curvature_scale: float
    timestamp: float = field(default_factory=time.time)


class DiagonalFisherNG:
    """
    O(N) Diagonal Fisher Information Measurement.
    
    QIG-PURE MEASUREMENT:
    This module computes geometric diagnostics. It NEVER mutates basins.
    Φ and κ emerge from basin navigation, not from loss optimization.
    
    For a probability distribution p, the Fisher metric is:
    G_ij = δ_ij / p_i  (diagonal for categorical)
    
    Natural gradient: ∇̃f = G⁻¹ ∇f = p ⊙ ∇f  (element-wise)
    
    This module returns these measurements for external controllers
    to use in deciding navigation strategy.
    
    Key Physics:
    - κ* = 64.21 determines optimal coupling
    - Basin dimension D = 64 (E8 rank²)
    - Fisher distance is the true metric, not Euclidean
    
    IMPORTANT: This class does NOT update basins. It only measures
    geometry and returns diagnostics that inform adaptive control.
    """
    
    def __init__(
        self,
        dim: int = BASIN_DIM,
        damping: float = 1e-4,
        ema_decay: float = 0.99
    ):
        """
        Initialize Diagonal Fisher measurement.
        
        Args:
            dim: Basin dimensionality (default: 64)
            damping: Regularization for Fisher diagonal (prevents division by zero)
            ema_decay: Exponential moving average decay for Fisher estimation
        """
        self.dim = dim
        self.damping = damping
        self.ema_decay = ema_decay
        
        self._fisher_diag_ema: Optional[np.ndarray] = None
        self._measurement_count = 0
    
    def compute_fisher_diagonal(
        self,
        probabilities: np.ndarray
    ) -> np.ndarray:
        """
        Compute diagonal of Fisher Information Matrix.
        
        PURE MEASUREMENT: No basin mutation.
        
        For categorical distribution:
        G_ii = 1 / p_i
        
        Args:
            probabilities: Current probability distribution
        
        Returns:
            Diagonal of Fisher metric (N,)
        """
        p = np.abs(probabilities) + 1e-10
        p = p / p.sum()
        
        fisher_diag = 1.0 / (p + self.damping)
        return fisher_diag
    
    def compute_natural_gradient(
        self,
        grad: np.ndarray,
        fisher_diag: np.ndarray
    ) -> np.ndarray:
        """
        Compute natural gradient using diagonal Fisher approximation.
        
        PURE MEASUREMENT: Returns gradient for diagnostics only.
        This does NOT update any basin - it returns a measurement.
        
        Natural gradient: ∇̃f = G⁻¹ ∇f
        With diagonal G: ∇̃f_i = grad_i / G_ii = grad_i * p_i
        
        This respects the manifold geometry - moving in directions
        that account for local curvature.
        
        Args:
            grad: Ordinary (Euclidean) gradient
            fisher_diag: Diagonal of Fisher Information Matrix
        
        Returns:
            Natural gradient (diagnostic, not for direct application)
        """
        fisher_diag_safe = fisher_diag + self.damping
        natural_grad = grad / fisher_diag_safe
        
        return natural_grad
    
    def update_fisher_ema(
        self,
        fisher_diag: np.ndarray
    ) -> np.ndarray:
        """
        Update exponential moving average of Fisher diagonal.
        
        Running average helps stabilize geometry measurements,
        especially when probability estimates are noisy.
        
        Args:
            fisher_diag: Current Fisher diagonal estimate
        
        Returns:
            Smoothed Fisher diagonal
        """
        if self._fisher_diag_ema is None:
            self._fisher_diag_ema = fisher_diag.copy()
        else:
            self._fisher_diag_ema = (
                self.ema_decay * self._fisher_diag_ema +
                (1 - self.ema_decay) * fisher_diag
            )
        
        return self._fisher_diag_ema
    
    def measure(
        self,
        basin: np.ndarray,
        grad: np.ndarray,
        probabilities: Optional[np.ndarray] = None
    ) -> DiagonalNGMeasurement:
        """
        Perform geometry measurement (no basin mutation).
        
        PURE MEASUREMENT: Returns diagnostics for external controllers.
        Does NOT apply updates to basins.
        
        Args:
            basin: Current basin coordinates (read-only)
            grad: Gradient to measure
            probabilities: Probability distribution (if None, derived from basin)
        
        Returns:
            DiagonalNGMeasurement with natural gradient and Fisher info
        """
        self._measurement_count += 1
        
        if probabilities is None:
            p = np.abs(basin) + 1e-10
            probabilities = p / p.sum()
        
        fisher_diag = self.compute_fisher_diagonal(probabilities)
        fisher_diag_smooth = self.update_fisher_ema(fisher_diag)
        
        natural_grad = self.compute_natural_gradient(grad, fisher_diag_smooth)
        condition_number = self.get_effective_condition_number()
        
        return DiagonalNGMeasurement(
            gradient=grad,
            natural_gradient=natural_grad,
            fisher_diag=fisher_diag_smooth,
            condition_number=condition_number
        )
    
    def get_curvature_measure(
        self,
        basin: np.ndarray
    ) -> CurvatureMeasurement:
        """
        Get curvature diagnostic from diagonal Fisher.
        
        PURE MEASUREMENT: Returns curvature metrics for adaptive control.
        Does NOT modify the basin.
        
        Args:
            basin: Current basin coordinates (read-only)
        
        Returns:
            CurvatureMeasurement with condition number, effective dimensionality
        """
        p = np.abs(basin) + 1e-10
        p = p / p.sum()
        
        fisher_diag = self.compute_fisher_diagonal(p)
        fisher_diag_smooth = self.update_fisher_ema(fisher_diag)
        
        fisher = fisher_diag_smooth + self.damping
        condition_number = float(fisher.max() / (fisher.min() + 1e-10))
        
        normalized_fisher = fisher / fisher.sum()
        entropy = -np.sum(normalized_fisher * np.log(normalized_fisher + 1e-10))
        effective_dim = np.exp(entropy)
        
        curvature_scale = float(np.std(fisher_diag_smooth))
        
        return CurvatureMeasurement(
            fisher_diagonal=fisher_diag_smooth,
            condition_number=condition_number,
            effective_dimensionality=effective_dim,
            curvature_scale=curvature_scale
        )
    
    def get_effective_condition_number(self) -> float:
        """
        Estimate condition number from diagonal Fisher.
        
        High condition number indicates ill-conditioning.
        Use this to inform navigation strategy.
        
        Returns:
            Condition number estimate
        """
        if self._fisher_diag_ema is None:
            return 1.0
        
        fisher = self._fisher_diag_ema + self.damping
        return float(fisher.max() / (fisher.min() + 1e-10))
    
    def get_measurement_count(self) -> int:
        """Return number of measurements performed."""
        return self._measurement_count
    
    def reset(self) -> None:
        """Reset measurement state."""
        self._fisher_diag_ema = None
        self._measurement_count = 0
