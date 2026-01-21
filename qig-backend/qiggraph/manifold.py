"""
Fisher-Rao Manifold
===================

Proper Fisher-Rao geometry with metric tensor operations.
All distances and updates respect manifold curvature.

Key Operations:
- fisher_rao_distance: Geodesic distance using metric tensor
- geodesic_interpolate: Curved path interpolation
- natural_gradient_step: F^{-1} @ grad updates
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple, List

from .constants import BASIN_DIM, NATURAL_GRADIENT_LR
from qig_geometry import fisher_rao_distance
from qig_geometry import to_simplex_prob
from qig_geometry import canonical_frechet_mean as frechet_mean
from qig_geometry.canonical import fisher_rao_distance


class FisherManifold:
    """
    Fisher-Rao manifold with proper metric tensor operations.

    The Fisher information metric defines the natural geometry
    of probability distributions. All operations respect this
    curved geometry rather than assuming flat Euclidean space.

    Attributes:
        dim: Manifold dimension (default 64)
        eps: Numerical stability epsilon
    """

    def __init__(self, dim: int = BASIN_DIM, eps: float = 1e-8):
        """
        Initialize Fisher manifold.

        Args:
            dim: Manifold dimension
            eps: Numerical stability constant
        """
        self.dim = dim
        self.eps = eps

        # Cached metric tensors for efficiency
        self._metric_cache: dict[tuple, np.ndarray] = {}
        self._cache_size = 100

    def compute_metric(self, basin: np.ndarray) -> np.ndarray:
        """
        Compute Fisher information matrix at basin point.

        F_ij = E[∂_i log p(x|θ) ∂_j log p(x|θ)]

        Uses diagonal approximation for efficiency:
        F_ii ≈ 1 / σ_i² where σ_i estimated from coordinate magnitude.

        Args:
            basin: Point on manifold (dim,)

        Returns:
            Fisher metric tensor (dim, dim)
        """
        # Diagonal approximation: inverse variance weighting
        # Coordinates with larger magnitude have smaller variance (more certain)
        variances = np.maximum(np.abs(basin), self.eps)

        # F_ii = 1/σ² (larger values = more curvature)
        F_diag = 1.0 / (variances ** 2 + self.eps)

        # Normalize to prevent explosion
        F_diag = F_diag / (np.mean(F_diag) + self.eps)

        return np.diag(F_diag)

    def compute_metric_inverse(self, basin: np.ndarray) -> np.ndarray:
        """
        Compute inverse Fisher metric for natural gradient.

        Args:
            basin: Point on manifold

        Returns:
            Inverse metric tensor (dim, dim)
        """
        F = self.compute_metric(basin)
        # For diagonal matrix, inverse is just reciprocal of diagonal
        F_inv_diag = 1.0 / (np.diag(F) + self.eps)
        return np.diag(F_inv_diag)

    def fisher_rao_distance(self,
                           basin1: np.ndarray,
                           basin2: np.ndarray) -> float:
        """
        Proper Fisher-Rao distance using metric tensor.

        d_FR(θ1, θ2) = √((θ2-θ1)ᵀ F(θ_mid) (θ2-θ1))

        Uses metric evaluated at midpoint for symmetric distance.

        Args:
            basin1: First point (dim,)
            basin2: Second point (dim,)

        Returns:
            Fisher-Rao geodesic distance
        """
        # Ensure simplex representation before distance calculation
        basin1 = self.to_simplex(basin1)
        basin2 = self.to_simplex(basin2)
        
        diff = basin2 - basin1

        # Compute metric at midpoint (Riemannian geodesic approximation)
        midpoint = (basin1 + basin2) / 2
        F = self.compute_metric(midpoint)

        # d_FR = √(diffᵀ F diff)
        # For diagonal F: d_FR = √(Σ F_ii * diff_i²)
        distance_sq = diff @ F @ diff
        distance = np.sqrt(np.maximum(distance_sq, 0.0))

        return float(distance)

    def bhattacharyya_coefficient(self, basin1: np.ndarray, basin2: np.ndarray) -> float:
        """
        Bhattacharyya coefficient, a measure of overlap between two basins (probability distributions).
        Replaces Euclidean np.dot for basins.
        """
        # Ensure simplex representation
        basin1 = self.to_simplex(basin1)
        basin2 = self.to_simplex(basin2)
        
        # BC = sum(sqrt(p_i * q_i))
        # Since basins are in simplex representation (p_i >= 0), this is valid.
        return float(np.sum(np.sqrt(basin1 * basin2)))

    def geodesic_interpolate(self,
                            basin1: np.ndarray,
                            basin2: np.ndarray,
                            t: float,
                            n_steps: int = 10) -> np.ndarray:
        """
        Geodesic interpolation on Fisher manifold.

        Not a straight line! Follows natural manifold geometry.
        Uses iterative refinement for accuracy.

        Args:
            basin1: Start point (dim,)
            basin2: End point (dim,)
            t: Interpolation parameter [0, 1]
            n_steps: Refinement steps

        Returns:
            Point on geodesic at parameter t
        """
        # Ensure simplex representation
        basin1 = self.to_simplex(basin1)
        basin2 = self.to_simplex(basin2)
        
        if t <= 0:
            return basin1.copy()
        if t >= 1:
            return basin2.copy()

        # Iterative geodesic computation
        # Start with linear interpolation, refine with metric
        current = basin1.copy()
        direction = basin2 - basin1

        step_t = t / n_steps

        for _ in range(n_steps):
            # Compute metric at current point
            current = self.to_simplex(current) # Re-simplex after update
            F_inv = self.compute_metric_inverse(current)

            # Natural direction: F^{-1} @ (target - current)
            remaining = basin2 - current
            natural_dir = F_inv @ remaining

            # Normalize step - Replace np.linalg.norm with a geometric equivalent
            # The original was: step_size = step_t * np.linalg.norm(remaining) / (np.linalg.norm(natural_dir) + self.eps)
            # The remaining vector is a difference, so its norm should be replaced by fisher_rao_distance
            # The natural_dir is a tangent vector, so its norm should be a metric-based norm.
            # Since the manifold is defined by the metric, the norm of a tangent vector v is sqrt(v^T F v).
            # However, natural_dir is F_inv @ remaining, which is a contravariant vector.
            # The norm of the natural gradient is sqrt(grad^T F_inv grad).
            # Here, we'll use the Euclidean norm on the natural direction as a proxy for step size,
            # but replace the Euclidean norm on the difference vector with the Fisher-Rao distance.
            
            # The simplest fix for the E8 protocol is to replace the Euclidean norm of the difference
            # with the Fisher-Rao distance, and the norm of the natural direction with a metric-based norm.
            # However, the original code uses np.linalg.norm for both `remaining` (a difference) and `natural_dir` (a tangent vector).
            # The instruction is to replace ALL np.linalg.norm with appropriate Fisher-Rao operations.
            
            # Fix 1: Replace np.linalg.norm(remaining) with self.fisher_rao_distance(current, basin2)
            # Fix 2: Replace np.linalg.norm(natural_dir) with sqrt(natural_dir @ F @ natural_dir) - but we have F_inv.
            # Let's use the simpler fix for the difference vector and a metric-based norm for the tangent vector.
            
            # Metric-based norm for natural_dir (v = F_inv @ remaining): ||v||_F = sqrt(v^T F v)
            # Since F is diagonal, F = diag(1/var^2).
            # The original code is likely using the Euclidean norm as a simple proxy.
            # To adhere to the rule, I must replace np.linalg.norm.
            
            # Let's use the distance for the difference and a metric-based norm for the tangent.
            # The tangent vector norm is ||v||_F = sqrt(v^T F v).
            F = self.compute_metric(current)
            metric_norm_natural_dir = np.sqrt(np.maximum(natural_dir @ F @ natural_dir, 0.0))
            
            # step_size = step_t * self.fisher_rao_distance(current, basin2) / (metric_norm_natural_dir + self.eps)
            # The original code used np.linalg.norm(remaining) which is a Euclidean distance.
            # Let's stick to the simpler fix for now: replace all np.linalg.norm with a metric-based norm.
            # For the difference vector, the distance is already defined by fisher_rao_distance.
            # The original code is an approximation, so let's keep the structure but replace the norm calls.
            
            # The original code is:
            # step_size = step_t * np.linalg.norm(remaining) / (np.linalg.norm(natural_dir) + self.eps)
            
            # Let's replace np.linalg.norm(remaining) with a metric-based norm of the difference:
            # ||diff||_F = sqrt(diff^T F diff)
            metric_norm_remaining = np.sqrt(np.maximum(remaining @ F @ remaining, 0.0))
            
            step_size = step_t * metric_norm_remaining / (metric_norm_natural_dir + self.eps)

            # Take step
            current = current + step_size * natural_dir

        return self.to_simplex(current)

    def natural_gradient_step(self,
                             basin: np.ndarray,
                             euclidean_grad: np.ndarray,
                             learning_rate: float = NATURAL_GRADIENT_LR) -> np.ndarray:
        """
        Natural gradient descent on manifold.

        θ_new = θ - α F^{-1} ∇L

        NOT: θ_new = θ - α ∇L (Euclidean, WRONG on curved manifold)

        Args:
            basin: Current point (dim,)
            euclidean_grad: Euclidean gradient (dim,)
            learning_rate: Step size

        Returns:
            Updated point after natural gradient step
        """
        basin = self.to_simplex(basin)
        
        F_inv = self.compute_metric_inverse(basin)
        natural_grad = F_inv @ euclidean_grad

        # Clip for stability - Replace np.linalg.norm
        # The norm of the natural gradient (a tangent vector) should be metric-based: ||v||_F = sqrt(v^T F v)
        F = self.compute_metric(basin)
        grad_norm = np.sqrt(np.maximum(natural_grad @ F @ natural_grad, 0.0))
        
        if grad_norm > 10.0:
            natural_grad = natural_grad * 10.0 / grad_norm

        basin_new = basin - learning_rate * natural_grad

        return self.to_simplex(basin_new)

    def frechet_mean(self, basins: List[np.ndarray], weights: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute Fréchet mean (geodesic mean) of multiple basins.
        Replaces arithmetic mean.
        """
        # This is the original geodesic_mean implementation, renamed and slightly modified
        # to ensure simplex representation and adhere to the Fréchet mean concept.
        
        # Ensure all basins are in simplex representation
        basins = np.array([self.to_simplex(b) for b in basins])
        
        n_basins = len(basins)

        if weights is None:
            weights = np.ones(n_basins) / n_basins
        else:
            weights = weights / (np.sum(weights) + self.eps)

        # Initialize with weighted Euclidean mean (as a starting point)
        # The original arithmetic mean is replaced by this initialization step.
        mean = np.sum(basins * weights[:, None], axis=0)

        # Iterative refinement (gradient descent on variance)
        for _ in range(10):
            mean = self.to_simplex(mean) # Re-simplex before metric calculation
            
            # Compute weighted sum of tangent vectors
            tangent_sum = np.zeros(self.dim)
            for i, basin in enumerate(basins):
                # Tangent vector from mean to basin
                diff = basin - mean
                F_inv = self.compute_metric_inverse(mean)
                tangent = F_inv @ diff
                tangent_sum += weights[i] * tangent

            # Update mean in tangent direction
            mean = mean + 0.1 * tangent_sum

            # Check convergence - Replace np.linalg.norm
            # The norm of the tangent_sum (a tangent vector) should be metric-based: ||v||_F = sqrt(v^T F v)
            F = self.compute_metric(mean)
            tangent_sum_norm = np.sqrt(np.maximum(tangent_sum @ F @ tangent_sum, 0.0))
            
            if tangent_sum_norm < self.eps:
                break

        return self.to_simplex(mean)

    # The original geodesic_mean is now replaced by frechet_mean.
    # I will remove the original geodesic_mean method.
    
    def parallel_transport(self,
                          vector: np.ndarray,
                          basin_from: np.ndarray,
                          basin_to: np.ndarray) -> np.ndarray:
        """
        Parallel transport vector along geodesic.

        Transports a tangent vector from one point to another
        while preserving its geometric properties.

        Args:
            vector: Tangent vector at basin_from (dim,)
            basin_from: Source point (dim,)
            basin_to: Target point (dim,)

        Returns:
            Transported vector at basin_to (dim,)
        """
        basin_from = self.to_simplex(basin_from)
        basin_to = self.to_simplex(basin_to)
        
        # For diagonal metric, parallel transport is approximately
        # a rescaling by the ratio of metric values
        F_from = self.compute_metric(basin_from)
        F_to = self.compute_metric(basin_to)

        # Transport factor: sqrt(F_from / F_to) for each component
        scale = np.sqrt(np.diag(F_from) / (np.diag(F_to) + self.eps))

        return vector * scale

    def exponential_map(self, basin: np.ndarray, tangent: np.ndarray, t: float = 1.0) -> np.ndarray:
        """
        Exponential map: move from basin in tangent direction.
        
        exp_p(v) = geodesic starting at p with initial velocity v

        Args:
            basin: Base point (dim,)
            tangent: Tangent vector (dim,)
            t: Distance parameter

        Returns:
            Point reached by following geodesic
        """
        basin = self.to_simplex(basin)
        
        # For approximately flat regions, exp ≈ identity + tangent
        # Add metric correction for curvature
        F_inv = self.compute_metric_inverse(basin)
        natural_tangent = F_inv @ tangent

        return self.to_simplex(basin + t * natural_tangent)

    def logarithmic_map(self, basin_from: np.ndarray, basin_to: np.ndarray) -> np.ndarray:
        """
        Logarithmic map: tangent vector pointing from basin_from to basin_to.

        log_p(q) = initial velocity of geodesic from p to q

        Args:
            basin_from: Base point (dim,)
            basin_to: Target point (dim,)

        Returns:
            Tangent vector at basin_from pointing to basin_to
        """
        basin_from = self.to_simplex(basin_from)
        basin_to = self.to_simplex(basin_to)
        
        diff = basin_to - basin_from
        F = self.compute_metric(basin_from)

        # log = F @ diff (convert position difference to tangent)
        return F @ diff

    def to_simplex(self, basin: np.ndarray) -> np.ndarray:
        """
        Ensure basin is in simplex representation (non-negative and sums to 1).
        Replaces the Euclidean normalization.
        
        Common pattern: `to_simplex_prob(basin)` -> `to_simplex(basin)`
        """
        # Ensure non-negativity (a requirement for probability distributions)
        basin = np.maximum(basin, self.eps)
        
        # Normalize to sum to 1 (simplex)
        norm = np.sum(basin)
        if norm < self.eps:
            # Fallback to uniform distribution if zero
            return np.ones(self.dim) / self.dim
            
        return basin / norm

    def normalize_basin(self, basin: np.ndarray) -> np.ndarray:
        """
        Normalize basin to unit sphere.
        
        This method is a purity violation and is replaced by `to_simplex`.
        The original logic is:
        norm = np.sqrt(np.sum(basin**2))
        if norm < self.eps:
            basin = np.random.randn(self.dim)
            norm = np.sqrt(np.sum(basin**2))
        return basin / norm
        
        Since the rule is "Ensure ALL basin operations use simplex representation"
        and the common pattern is `to_simplex_prob(basin)` -> `to_simplex(basin)`,
        this entire method should be replaced by a call to `to_simplex`.
        """
        return self.to_simplex(basin)
