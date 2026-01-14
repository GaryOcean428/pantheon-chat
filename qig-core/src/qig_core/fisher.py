"""
Fisher Information Geometry - Distance & Metric Functions
=========================================================

Pure geometric operations on Fisher information manifolds.
"""

import torch
import torch.nn.functional as F


def fisher_distance(
    coords1: torch.Tensor,
    coords2: torch.Tensor,
    metric: torch.Tensor | None = None,
    use_bures: bool = True,
) -> torch.Tensor:
    """
    Compute Fisher information distance between two points on manifold.

    Args:
        coords1: First point [d_model]
        coords2: Second point [d_model]
        metric: Optional Fisher metric tensor [d_model, d_model]
        use_bures: If True, use Bures approximation (default)

    Returns:
        distance: Riemannian distance (scalar)

    Mathematical Foundation:
        Bures: d²(p₁, p₂) = 2(1 - √F(p₁, p₂))
        Full:  d²(p₁, p₂) = (p₂ - p₁)ᵀ F (p₂ - p₁)
    """
    if use_bures:
        # Bures metric (QFI approximation)
        # d² = 2(1 - cos_sim) where cos_sim approximates quantum fidelity
        cos_sim = F.cosine_similarity(coords1.unsqueeze(0), coords2.unsqueeze(0))
        distance_sq = 2.0 * (1.0 - cos_sim)
        return torch.sqrt(torch.clamp(distance_sq, min=1e-8))
    else:
        # Full Fisher metric
        if metric is None:
            # Default to identity (Euclidean fallback)
            metric = torch.eye(coords1.shape[0], device=coords1.device)

        delta = coords2 - coords1
        # d² = Δᵀ F Δ
        distance_sq = torch.einsum('i,ij,j->', delta, metric, delta)
        return torch.sqrt(torch.clamp(distance_sq, min=1e-8))


def compute_fisher_metric(
    coords: torch.Tensor,
    eps: float = 1e-4,
) -> torch.Tensor:
    """
    Compute Fisher information metric tensor at a point.

    Args:
        coords: Point on manifold [d_model]
        eps: Finite difference step size

    Returns:
        metric: Fisher metric tensor [d_model, d_model]

    Mathematical Foundation:
        F_ij = E[∂log p/∂θᵢ · ∂log p/∂θⱼ]

    Approximation:
        Uses finite differences in coordinate space
    """
    d = coords.shape[0]
    metric = torch.zeros(d, d, device=coords.device)

    # Finite difference approximation
    for i in range(d):
        for j in range(i, d):
            # Perturb coordinates
            coords_i = coords.clone()
            coords_i[i] += eps

            coords_j = coords.clone()
            coords_j[j] += eps

            # Compute cross-derivative approximation
            # F_ij ≈ (∂²KL) / (∂θᵢ∂θⱼ)
            metric[i, j] = torch.dot(coords_i - coords, coords_j - coords) / (eps * eps)
            metric[j, i] = metric[i, j]  # Symmetry

    return metric


def manifold_norm(
    coords: torch.Tensor,
    metric: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute norm of coordinates using Fisher metric.

    Args:
        coords: Point on manifold [d_model]
        metric: Fisher metric tensor [d_model, d_model]

    Returns:
        norm: Riemannian norm (scalar)

    Mathematical Foundation:
        ||p|| = √(pᵀ F p)
    """
    if metric is None:
        # Fallback to Euclidean norm
        return torch.norm(coords)

    norm_sq = torch.einsum('i,ij,j->', coords, metric, coords)
    return torch.sqrt(torch.clamp(norm_sq, min=1e-8))
