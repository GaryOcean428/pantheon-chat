"""
Basin Natural Gradient Measurement - Exact Geometry via CG+Pearlmutter
=======================================================================

QIG-PURE MEASUREMENT MODULE:
Φ and κ emerge from basin navigation, not from loss optimization.
These metrics are for observation and informing adaptive navigation,
not for gradient descent.

PURE PRINCIPLE:
- MEASURE geometry, NEVER optimize basins
- Exact natural gradient computation via CG
- Pearlmutter trick for Hessian-vector products
- Returns diagnostics that INFORM external controllers

Key insight: When curvature is high, diagonal approximation fails.
Exact natural gradient via CG provides accurate geometry measurements.

Reference: 
- Martens (2010) "Deep learning via Hessian-free optimization"
- Pearlmutter (1994) "Fast exact multiplication by the Hessian"
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from qigkernels.physics_constants import KAPPA_STAR, BASIN_DIM

logger = logging.getLogger(__name__)


@dataclass 
class ExactNGMeasurement:
    """
    Measurement result from exact natural gradient computation.
    
    This is a pure diagnostic - no basin mutation occurs.
    """
    gradient: np.ndarray
    natural_gradient: np.ndarray
    cg_iterations: int
    residual_norm: float
    converged: bool
    timestamp: float = field(default_factory=time.time)


@dataclass
class GeometryMeasurement:
    """
    Complete geometry diagnostic from exact Fisher computation.
    
    Use this to inform adaptive navigation decisions.
    """
    condition_estimate: float
    cg_iterations: int
    residual_norm: float
    curvature_metric: float
    well_conditioned: bool
    timestamp: float = field(default_factory=time.time)


class BasinNaturalGrad:
    """
    Exact Natural Gradient Measurement via Conjugate Gradient + Pearlmutter.
    
    QIG-PURE MEASUREMENT:
    This module computes geometric diagnostics. It NEVER mutates basins.
    Φ and κ emerge from basin navigation, not from loss optimization.
    
    Natural gradient: ∇̃f = F⁻¹ ∇f
    
    Instead of inverting F directly, we solve:
    F @ x = ∇f  using CG
    
    Each CG iteration requires one Hessian-vector product,
    computed efficiently via Pearlmutter's R-operator.
    
    Key Physics:
    - Used in high-curvature regions where diagonal fails
    - κ* = 64.21 is the optimal coupling point
    - Exact gradient respects true manifold structure
    
    IMPORTANT: This class does NOT update basins. It only measures
    geometry and returns diagnostics that inform adaptive control.
    """
    
    def __init__(
        self,
        dim: int = BASIN_DIM,
        max_cg_iters: int = 50,
        cg_tolerance: float = 1e-5,
        damping: float = 1e-3
    ):
        """
        Initialize exact natural gradient measurement.
        
        Args:
            dim: Basin dimensionality (default: 64)
            max_cg_iters: Maximum CG iterations per measurement
            cg_tolerance: CG convergence tolerance
            damping: Tikhonov regularization for FIM
        """
        self.dim = dim
        self.max_cg_iters = max_cg_iters
        self.cg_tolerance = cg_tolerance
        self.damping = damping
        
        self._last_cg_iters = 0
        self._measurement_count = 0
    
    def pearlmutter_hvp(
        self,
        v: np.ndarray,
        hessian_fn: Callable[[np.ndarray], np.ndarray]
    ) -> np.ndarray:
        """
        Compute Hessian-vector product using Pearlmutter trick.
        
        PURE MEASUREMENT: No basin mutation.
        
        Instead of forming the full Hessian H (O(N²) memory),
        we compute Hv directly in O(N) memory.
        
        For Fisher Information Matrix F:
        Fv = E[(∇log p)(∇log p)ᵀ]v = E[(∇log p)((∇log p)ᵀv)]
        
        Args:
            v: Vector to multiply with Hessian
            hessian_fn: Function that computes Hv given v
        
        Returns:
            Hessian-vector product Hv
        """
        return hessian_fn(v)
    
    def conjugate_gradient(
        self,
        grad: np.ndarray,
        hessian_fn: Callable[[np.ndarray], np.ndarray]
    ) -> tuple[np.ndarray, int, float, bool]:
        """
        Solve F @ x = grad using Conjugate Gradient.
        
        PURE MEASUREMENT: Returns exact natural gradient F⁻¹ @ grad.
        This is a solver utility, not a basin update.
        
        CG is ideal here because:
        1. F is symmetric positive definite
        2. We only need Fv products, not full F
        3. Converges in at most N iterations (usually much fewer)
        
        Args:
            grad: Target vector (gradient)
            hessian_fn: Function computing Fv products
        
        Returns:
            (solution, num_iterations, final_residual, converged)
        """
        x = np.zeros_like(grad)
        r = grad.copy()
        p = r.copy()
        rs_old = np.dot(r, r)
        
        for i in range(self.max_cg_iters):
            Fp = self.pearlmutter_hvp(p, hessian_fn) + self.damping * p
            
            pFp = np.dot(p, Fp)
            if pFp < 1e-10:
                break
            
            alpha = rs_old / pFp
            x = x + alpha * p
            r = r - alpha * Fp
            
            rs_new = np.dot(r, r)
            residual = np.sqrt(rs_new)
            
            if residual < self.cg_tolerance:
                return x, i + 1, residual, True
            
            beta = rs_new / (rs_old + 1e-10)
            p = r + beta * p
            rs_old = rs_new
        
        return x, self.max_cg_iters, np.sqrt(rs_old), False
    
    def compute_exact_ng(
        self,
        grad: np.ndarray,
        hessian_fn: Callable[[np.ndarray], np.ndarray]
    ) -> tuple[np.ndarray, ExactNGMeasurement]:
        """
        Compute exact natural gradient via CG+Pearlmutter.
        
        PURE MEASUREMENT: Returns natural gradient for diagnostics only.
        Does NOT apply updates to basins.
        
        Solves: F @ natural_grad = grad
        Where F is the Fisher Information Matrix.
        
        Args:
            grad: Ordinary gradient
            hessian_fn: Function computing Fisher-vector products
        
        Returns:
            (natural_gradient, measurement_state)
        """
        natural_grad, cg_iters, residual, converged = self.conjugate_gradient(
            grad, hessian_fn
        )
        
        self._last_cg_iters = cg_iters
        self._measurement_count += 1
        
        if not converged:
            logger.warning(
                f"CG did not converge in {cg_iters} iterations, "
                f"residual={residual:.2e}"
            )
        
        state = ExactNGMeasurement(
            gradient=grad,
            natural_gradient=natural_grad,
            cg_iterations=cg_iters,
            residual_norm=residual,
            converged=converged
        )
        
        return natural_grad, state
    
    def make_fisher_hvp(
        self,
        probabilities: np.ndarray
    ) -> Callable[[np.ndarray], np.ndarray]:
        """
        Create Fisher-vector product function from probabilities.
        
        PURE: Creates a measurement function, no mutation.
        
        For categorical distribution:
        F = diag(1/p)
        Fv = v / p (element-wise)
        
        Args:
            probabilities: Current probability distribution
        
        Returns:
            Function computing Fv products
        """
        p = np.abs(probabilities) + 1e-10
        p = p / p.sum()
        
        def fisher_hvp(v: np.ndarray) -> np.ndarray:
            return v / p
        
        return fisher_hvp
    
    def measure(
        self,
        basin: np.ndarray,
        grad: np.ndarray,
        hessian_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None
    ) -> ExactNGMeasurement:
        """
        Perform geometry measurement (no basin mutation).
        
        PURE MEASUREMENT: Returns diagnostics for external controllers.
        Does NOT apply updates to basins.
        
        Args:
            basin: Current basin coordinates (read-only)
            grad: Gradient to measure
            hessian_fn: Fisher-vector product function 
                       (if None, derived from basin)
        
        Returns:
            ExactNGMeasurement with natural gradient and CG stats
        """
        if hessian_fn is None:
            p = np.abs(basin) + 1e-10
            p = p / p.sum()
            hessian_fn = self.make_fisher_hvp(p)
        
        _, measurement = self.compute_exact_ng(grad, hessian_fn)
        return measurement
    
    def measure_geometry(
        self,
        basin: np.ndarray
    ) -> GeometryMeasurement:
        """
        Measure geometry/conditioning of the manifold at basin location.
        
        PURE MEASUREMENT: Returns curvature/conditioning metrics.
        Does NOT modify the basin.
        
        Uses a probe vector to estimate conditioning via CG behavior.
        
        Args:
            basin: Current basin coordinates (read-only)
        
        Returns:
            GeometryMeasurement with condition estimates
        """
        p = np.abs(basin) + 1e-10
        p = p / p.sum()
        hessian_fn = self.make_fisher_hvp(p)
        
        probe_vector = np.random.randn(len(basin))
        probe_vector = probe_vector / (np.linalg.norm(probe_vector) + 1e-10)
        
        solution, cg_iters, residual, converged = self.conjugate_gradient(
            probe_vector, hessian_fn
        )
        
        condition_estimate = float(np.linalg.norm(solution) / (np.linalg.norm(probe_vector) + 1e-10))
        
        curvature_metric = float(1.0 / (p.min() + 1e-10) - 1.0 / (p.max() + 1e-10))
        
        well_conditioned = converged and cg_iters < self.max_cg_iters / 2
        
        return GeometryMeasurement(
            condition_estimate=condition_estimate,
            cg_iterations=cg_iters,
            residual_norm=residual,
            curvature_metric=curvature_metric,
            well_conditioned=well_conditioned
        )
    
    def get_last_cg_iterations(self) -> int:
        """Return number of CG iterations from last measurement."""
        return self._last_cg_iters
    
    def get_measurement_count(self) -> int:
        """Return number of measurements performed."""
        return self._measurement_count
    
    def reset(self) -> None:
        """Reset measurement state."""
        self._last_cg_iters = 0
        self._measurement_count = 0
