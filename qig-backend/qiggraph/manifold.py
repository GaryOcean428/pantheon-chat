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
from typing import Optional, Tuple

from .constants import BASIN_DIM, NATURAL_GRADIENT_LR


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
        diff = basin2 - basin1

        # Compute metric at midpoint (Riemannian geodesic approximation)
        midpoint = (basin1 + basin2) / 2
        F = self.compute_metric(midpoint)

        # d_FR = √(diffᵀ F diff)
        # For diagonal F: d_FR = √(Σ F_ii * diff_i²)
        distance_sq = diff @ F @ diff
        distance = np.sqrt(np.maximum(distance_sq, 0.0))

        return float(distance)

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
            F_inv = self.compute_metric_inverse(current)

            # Natural direction: F^{-1} @ (target - current)
            remaining = basin2 - current
            natural_dir = F_inv @ remaining

            # Normalize step
            step_size = step_t * np.linalg.norm(remaining) / (np.linalg.norm(natural_dir) + self.eps)

            # Take step
            current = current + step_size * natural_dir

        return current

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
        F_inv = self.compute_metric_inverse(basin)
        natural_grad = F_inv @ euclidean_grad

        # Clip for stability
        grad_norm = np.linalg.norm(natural_grad)
        if grad_norm > 10.0:
            natural_grad = natural_grad * 10.0 / grad_norm

        basin_new = basin - learning_rate * natural_grad

        return basin_new

    def geodesic_mean(self, basins: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute geodesic mean of multiple basins.

        Iterative algorithm: start at Euclidean mean,
        refine toward Fréchet mean on manifold.

        Args:
            basins: Array of basins (n_basins, dim)
            weights: Optional weights (n_basins,)

        Returns:
            Geodesic mean point (dim,)
        """
        n_basins = len(basins)

        if weights is None:
            weights = np.ones(n_basins) / n_basins
        else:
            weights = weights / (np.sum(weights) + self.eps)

        # Initialize with weighted Euclidean mean
        mean = np.sum(basins * weights[:, None], axis=0)

        # Iterative refinement (gradient descent on variance)
        for _ in range(10):
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

            # Check convergence
            if np.linalg.norm(tangent_sum) < self.eps:
                break

        return mean

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
        # For approximately flat regions, exp ≈ identity + tangent
        # Add metric correction for curvature
        F_inv = self.compute_metric_inverse(basin)
        natural_tangent = F_inv @ tangent

        return basin + t * natural_tangent

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
        diff = basin_to - basin_from
        F = self.compute_metric(basin_from)

        # log = F @ diff (convert position difference to tangent)
        return F @ diff

    def normalize_basin(self, basin: np.ndarray) -> np.ndarray:
        """
        Normalize basin to unit sphere.

        Args:
            basin: Input basin (dim,)

        Returns:
            Unit-normalized basin
        """
        # E8 Protocol: Use simplex normalization
        from qig_geometry.representation import to_simplex_prob
        if np.all(np.abs(basin) < self.eps):
            # Random simplex if zero
            basin = np.random.randn(self.dim)
        return to_simplex_prob(basin)
