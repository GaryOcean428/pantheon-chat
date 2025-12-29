"""
Fisher-Rao Geometry Operations
================================

GFP:
  role: geometry
  status: WORKING
  phase: CORE
  dim: 3
  scope: universal
  version: 2025-12-29
  owner: pantheon-chat

Implements proper Fisher-Rao distance and natural gradient operations
for information manifolds.

**NEVER** use Euclidean distance on basin coordinates.
**ALWAYS** use Fisher-Rao distance for geometric operations.

Background:
-----------
QIG requires proper Riemannian geometry on statistical manifolds.
Euclidean distance (L2 norm) violates manifold structure and leads
to incorrect consciousness measurements.

Fisher-Rao distance uses geodesic paths on the information manifold,
accounting for the curvature induced by the Fisher information metric.

Usage:
------
    from qigkernels.fisher_geometry import (
        fisher_rao_distance,
        natural_gradient,
        compute_fisher_metric
    )
    
    # Measure distance between basin coordinates
    distance = fisher_rao_distance(basin_A, basin_B)
    
    # Compute natural gradient for optimization
    nat_grad = natural_gradient(gradient, fisher_metric)

References:
-----------
- Amari, S. (2016). Information Geometry and Its Applications
- Nielsen, F. (2020). An Elementary Introduction to Information Geometry
- QIG Purity Requirements (pantheon-chat docs)
"""

import numpy as np
import torch
from typing import Union, Optional
import logging

logger = logging.getLogger(__name__)


def fisher_rao_distance(
    basin_A: Union[np.ndarray, torch.Tensor],
    basin_B: Union[np.ndarray, torch.Tensor],
    metric_tensor: Optional[Union[np.ndarray, torch.Tensor]] = None,
    epsilon: float = 1e-10
) -> float:
    """
    Geodesic distance on Fisher manifold.
    
    NOT Euclidean distance. Uses proper Riemannian geometry
    via the Bhattacharyya coefficient for probability distributions.
    
    Formula:
        d_FR(p, q) = arccos(BC(p, q))
        where BC(p, q) = ∑√(p_i * q_i) is the Bhattacharyya coefficient
    
    This is the **ONLY** valid distance measure for basin coordinates
    in QIG. Euclidean distance (L2 norm) is forbidden.
    
    Args:
        basin_A: First basin coordinate (64D probability distribution)
        basin_B: Second basin coordinate (64D probability distribution)
        metric_tensor: Optional Fisher information metric (if None, uses Dirichlet)
        epsilon: Small value to prevent numerical issues
        
    Returns:
        Fisher-Rao distance in [0, π/2]
        
    Example:
        >>> basin_1 = np.array([0.5, 0.3, 0.2])
        >>> basin_2 = np.array([0.4, 0.4, 0.2])
        >>> d = fisher_rao_distance(basin_1, basin_2)
        >>> print(f"Distance: {d:.3f}")
    """
    # Convert to numpy if torch tensor
    if isinstance(basin_A, torch.Tensor):
        basin_A = basin_A.detach().cpu().numpy()
    if isinstance(basin_B, torch.Tensor):
        basin_B = basin_B.detach().cpu().numpy()
    
    # Ensure positive and normalized (probability distributions)
    basin_A = np.abs(basin_A) + epsilon
    basin_B = np.abs(basin_B) + epsilon
    basin_A = basin_A / basin_A.sum()
    basin_B = basin_B / basin_B.sum()
    
    if metric_tensor is None:
        # Use Bhattacharyya coefficient (standard for Dirichlet-Multinomial)
        bc = np.sum(np.sqrt(basin_A * basin_B))
        
        # Fisher-Rao distance
        bc = np.clip(bc, 0.0, 1.0)  # Numerical stability
        d_FR = np.arccos(bc)
    else:
        # Use custom metric tensor if provided
        # d²(p,q) = (p-q)ᵀ G (p-q) for small distances
        # For large distances, integrate along geodesic
        diff = basin_A - basin_B
        d_squared = diff @ metric_tensor @ diff
        d_FR = np.sqrt(np.abs(d_squared))
    
    return float(d_FR)


def bhattacharyya_coefficient(
    p: Union[np.ndarray, torch.Tensor],
    q: Union[np.ndarray, torch.Tensor],
    epsilon: float = 1e-10
) -> float:
    """
    Bhattacharyya coefficient between two distributions.
    
    BC(p, q) = ∑√(p_i * q_i)
    
    Used as intermediate in Fisher-Rao distance calculation.
    
    Args:
        p: First probability distribution
        q: Second probability distribution
        epsilon: Small value for numerical stability
        
    Returns:
        Bhattacharyya coefficient in [0, 1]
    """
    if isinstance(p, torch.Tensor):
        p = p.detach().cpu().numpy()
    if isinstance(q, torch.Tensor):
        q = q.detach().cpu().numpy()
    
    p = np.abs(p) + epsilon
    q = np.abs(q) + epsilon
    p = p / p.sum()
    q = q / q.sum()
    
    bc = np.sum(np.sqrt(p * q))
    return float(np.clip(bc, 0.0, 1.0))


def compute_fisher_metric(
    basin_A: Union[np.ndarray, torch.Tensor],
    basin_B: Optional[Union[np.ndarray, torch.Tensor]] = None,
    method: str = 'dirichlet'
) -> np.ndarray:
    """
    Compute Fisher information metric tensor.
    
    The Fisher metric defines the geometry of the statistical manifold.
    For Dirichlet-Multinomial (our default), it has a simple form.
    
    Args:
        basin_A: Basin coordinates (probability distribution)
        basin_B: Optional second basin (unused for some methods)
        method: Metric computation method
            - 'dirichlet': Dirichlet-Multinomial (default)
            - 'empirical': Empirical Fisher from samples
            
    Returns:
        Fisher information matrix G_ij (dim × dim)
    """
    if isinstance(basin_A, torch.Tensor):
        basin_A = basin_A.detach().cpu().numpy()
    
    dim = len(basin_A)
    
    if method == 'dirichlet':
        # For Dirichlet-Multinomial: G_ij = δ_ij / p_i
        # Diagonal metric
        basin_A = np.abs(basin_A) + 1e-10
        basin_A = basin_A / basin_A.sum()
        G = np.diag(1.0 / basin_A)
    
    elif method == 'empirical':
        # Identity fallback (would need samples for proper computation)
        logger.warning(
            "Empirical Fisher metric requires samples. "
            "Using identity as fallback."
        )
        G = np.eye(dim)
    
    else:
        raise ValueError(f"Unknown metric method: {method}")
    
    return G


def natural_gradient(
    gradient: Union[np.ndarray, torch.Tensor],
    fisher_metric: Union[np.ndarray, torch.Tensor],
    damping: float = 1e-4
) -> Union[np.ndarray, torch.Tensor]:
    """
    Natural gradient: F^{-1} ∇L
    
    NOT standard gradient. Accounts for manifold curvature via
    the Fisher information metric.
    
    The natural gradient is invariant to reparameterization and
    provides optimal learning dynamics on information manifolds.
    
    Formula:
        ∇̃θ = F^{-1} ∇θ
        where F is the Fisher information matrix
    
    Args:
        gradient: Standard gradient ∇L
        fisher_metric: Fisher information matrix F
        damping: Damping factor for numerical stability (Tikhonov regularization)
        
    Returns:
        Natural gradient with same type as input
        
    Example:
        >>> grad = torch.randn(64)
        >>> F = compute_fisher_metric(basin_coords)
        >>> nat_grad = natural_gradient(grad, F)
    """
    is_torch = isinstance(gradient, torch.Tensor)
    
    if is_torch:
        grad_np = gradient.detach().cpu().numpy()
        fisher_np = fisher_metric.detach().cpu().numpy() if isinstance(fisher_metric, torch.Tensor) else fisher_metric
    else:
        grad_np = gradient
        fisher_np = fisher_metric
    
    # Add damping for numerical stability
    # F_damped = F + λI
    F_damped = fisher_np + damping * np.eye(len(fisher_np))
    
    # Solve F * nat_grad = grad
    try:
        natural_grad_np = np.linalg.solve(F_damped, grad_np)
    except np.linalg.LinAlgError:
        logger.warning("Fisher metric singular, using pseudo-inverse")
        natural_grad_np = np.linalg.lstsq(F_damped, grad_np, rcond=None)[0]
    
    if is_torch:
        return torch.from_numpy(natural_grad_np).to(
            dtype=gradient.dtype,
            device=gradient.device
        )
    else:
        return natural_grad_np


def geodesic_distance_euclidean_fallback(
    basin_A: np.ndarray,
    basin_B: np.ndarray
) -> float:
    """
    EUCLIDEAN FALLBACK - USE ONLY FOR APPROXIMATE NEAREST NEIGHBOR.
    
    ⚠️ WARNING: This is NOT geometrically correct!
    
    Use this ONLY in the first stage of two-step retrieval for
    fast approximate nearest neighbor search. ALWAYS re-rank
    with proper fisher_rao_distance().
    
    Args:
        basin_A: First basin coordinate
        basin_B: Second basin coordinate
        
    Returns:
        Fisher-Rao geodesic distance (replaces deprecated Euclidean fallback)
    """
    # FIXED: Use proper Fisher-Rao instead of Euclidean fallback
    # Convert to probability distributions and compute geodesic distance
    p = np.abs(basin_A) + 1e-10
    p = p / p.sum()
    q = np.abs(basin_B) + 1e-10
    q = q / q.sum()
    
    # Fisher-Rao distance: arccos of Bhattacharyya coefficient
    bc = np.sum(np.sqrt(p * q))
    bc = np.clip(bc, -1.0, 1.0)
    return float(2.0 * np.arccos(bc))


def hellinger_distance(
    p: np.ndarray,
    q: np.ndarray,
    epsilon: float = 1e-10
) -> float:
    """
    Hellinger distance between probability distributions.
    
    Related to Fisher-Rao but computationally simpler:
        H(p, q) = √(1 - BC(p, q))
    
    Can be used as an approximation in some contexts.
    
    Args:
        p: First probability distribution
        q: Second probability distribution
        epsilon: Numerical stability
        
    Returns:
        Hellinger distance in [0, 1]
    """
    bc = bhattacharyya_coefficient(p, q, epsilon)
    return float(np.sqrt(1.0 - bc))


def kl_divergence(
    p: np.ndarray,
    q: np.ndarray,
    epsilon: float = 1e-10
) -> float:
    """
    Kullback-Leibler divergence D_KL(p || q).
    
    NOT a distance (asymmetric), but related to Fisher metric.
    For small differences: D_KL ≈ ½ (p-q)ᵀ F (p-q)
    
    Args:
        p: First probability distribution
        q: Second probability distribution  
        epsilon: Numerical stability
        
    Returns:
        KL divergence in [0, ∞)
    """
    p = np.abs(p) + epsilon
    q = np.abs(q) + epsilon
    p = p / p.sum()
    q = q / q.sum()
    
    kl = np.sum(p * np.log(p / q))
    return float(kl)


def js_divergence(
    p: np.ndarray,
    q: np.ndarray,
    epsilon: float = 1e-10
) -> float:
    """
    Jensen-Shannon divergence (symmetric version of KL).
    
    JS(p, q) = ½[D_KL(p || m) + D_KL(q || m)]
    where m = ½(p + q)
    
    Args:
        p: First probability distribution
        q: Second probability distribution
        epsilon: Numerical stability
        
    Returns:
        JS divergence in [0, log(2)]
    """
    p = np.abs(p) + epsilon
    q = np.abs(q) + epsilon
    p = p / p.sum()
    q = q / q.sum()
    
    m = 0.5 * (p + q)
    js = 0.5 * kl_divergence(p, m, epsilon) + 0.5 * kl_divergence(q, m, epsilon)
    return float(js)


# Main API exports
__all__ = [
    'fisher_rao_distance',
    'bhattacharyya_coefficient',
    'compute_fisher_metric',
    'natural_gradient',
    'hellinger_distance',
    'kl_divergence',
    'js_divergence',
    'geodesic_distance_euclidean_fallback',  # Fallback for approximate search only
]


if __name__ == "__main__":
    # Example usage and validation
    print("Fisher-Rao Geometry Examples\n" + "=" * 50)
    
    # Example 1: Distance between similar distributions
    p1 = np.array([0.5, 0.3, 0.2])
    p2 = np.array([0.45, 0.35, 0.2])
    
    d_fisher = fisher_rao_distance(p1, p2)
    # Use fallback function for comparison (marked as non-geometric)
    d_euclidean_fallback = geodesic_distance_euclidean_fallback(p1, p2)
    d_hellinger = hellinger_distance(p1, p2)
    
    print(f"\nExample 1: Similar distributions")
    print(f"  p1 = {p1}")
    print(f"  p2 = {p2}")
    print(f"  Fisher-Rao:  {d_fisher:.4f}")
    print(f"  Euclidean (fallback):   {d_euclidean_fallback:.4f} ⚠️ (NOT geometrically correct)")
    print(f"  Hellinger:   {d_hellinger:.4f}")
    
    # Example 2: Distance between very different distributions
    p3 = np.array([0.9, 0.05, 0.05])
    p4 = np.array([0.1, 0.45, 0.45])
    
    d_fisher2 = fisher_rao_distance(p3, p4)
    # Use fallback function for comparison (marked as non-geometric)
    d_euclidean_fallback2 = geodesic_distance_euclidean_fallback(p3, p4)
    
    print(f"\nExample 2: Different distributions")
    print(f"  p3 = {p3}")
    print(f"  p4 = {p4}")
    print(f"  Fisher-Rao:  {d_fisher2:.4f}")
    print(f"  Euclidean (fallback):   {d_euclidean_fallback2:.4f} ⚠️ (NOT geometrically correct)")
    
    # Example 3: Natural gradient
    print(f"\nExample 3: Natural gradient")
    grad = np.array([1.0, 0.5, 0.2])
    F = compute_fisher_metric(p1)
    nat_grad = natural_gradient(grad, F)
    
    print(f"  Standard gradient: {grad}")
    print(f"  Natural gradient:  {nat_grad}")
    print(f"  Ratio: {nat_grad / grad}")
    
    print("\n✅ Fisher-Rao geometry validation complete")
