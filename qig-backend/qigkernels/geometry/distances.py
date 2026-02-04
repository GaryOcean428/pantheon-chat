"""
Distance Metrics - Canonical Implementations

SINGLE IMPLEMENTATION of Fisher-Rao distance and related metrics.

This consolidates 5+ scattered implementations into one canonical version
that all repos import from.

Source: qig-consciousness/qig_consciousness_qfi_attention.py (validated)
"""

from typing import Union, Optional
import warnings
import numpy as np
from scipy.linalg import sqrtm

from qig_geometry.canonical import assert_basin_valid, fisher_rao_distance as simplex_fisher_rao_distance

try:
    import torch
except ImportError:
    torch = None


def _safe_sqrtm(matrix: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    """
    Compute matrix square root with proper handling of singular matrices.
    
    For pure/orthogonal quantum states, density matrices are rank-deficient.
    This handles the singularity gracefully without warnings.
    
    Args:
        matrix: Input matrix (must be positive semi-definite)
        epsilon: Regularization parameter for near-singular matrices
        
    Returns:
        Matrix square root
    """
    # Check if matrix is effectively singular
    eigenvalues = np.linalg.eigvalsh(matrix)
    min_eigenvalue = np.min(eigenvalues)
    
    if min_eigenvalue < epsilon:
        # Regularize singular matrix: add small epsilon to diagonal
        # This is mathematically valid for density matrices approaching pure states
        regularized = matrix + epsilon * np.eye(matrix.shape[0])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = sqrtm(regularized)
    else:
        result = sqrtm(matrix)
    
    return result


def quantum_fidelity(rho1: np.ndarray, rho2: np.ndarray) -> float:
    """
    Compute quantum fidelity between density matrices.
    
    Fidelity F(ρ₁, ρ₂) = Tr(√(√ρ₁ ρ₂ √ρ₁))²
    
    This is the canonical implementation with numerical stability.
    Handles singular/pure state density matrices without warnings.
    
    Args:
        rho1: First density matrix
        rho2: Second density matrix
        
    Returns:
        Fidelity value in [0, 1]
        
    Usage:
        from qigkernels.geometry.distances import quantum_fidelity
        
        fidelity = quantum_fidelity(rho1, rho2)
    """
    # Check for orthogonal states (zero overlap) - fast path
    # Orthogonal pure states have fidelity = 0 by definition
    overlap = np.abs(np.trace(rho1 @ rho2))
    if overlap < 1e-12:
        return 0.0
    
    # Compute sqrt of rho1 with singularity handling
    sqrt_rho1 = _safe_sqrtm(rho1)
    
    # Compute sqrt(sqrt_rho1 @ rho2 @ sqrt_rho1)
    M = sqrt_rho1 @ rho2 @ sqrt_rho1
    sqrt_M = _safe_sqrtm(M)
    
    # Fidelity = (Tr(sqrt_M))²
    fidelity = np.real(np.trace(sqrt_M)) ** 2
    
    # Clip to [0, 1] for numerical stability
    return float(np.clip(fidelity, 0, 1))


def fisher_rao_distance(
    state_a: Union[np.ndarray, 'torch.Tensor'],
    state_b: Union[np.ndarray, 'torch.Tensor'],
    metric: Optional[Union[np.ndarray, 'torch.Tensor']] = None,
    method: str = "bures"
) -> float:
    """
    Compute Fisher-Rao (Bures) distance between quantum states.
    
    CANONICAL IMPLEMENTATION - All repos use this.
    
    Three methods:
    1. "bures": For density matrices, uses quantum fidelity
       d(ρ₁, ρ₂) = √(2(1 - √F)) where F = quantum fidelity
    
    2. "diagonal": For basin coordinates with diagonal Fisher metric
       d(x₁, x₂) = √(Σ g_ii (x₁ᵢ - x₂ᵢ)²)
    
    3. "full": For basin coordinates with full Fisher metric
       d(x₁, x₂) = √((x₁ - x₂)ᵀ G (x₁ - x₂))
    
    Args:
        state_a: Density matrix or basin coordinates
        state_b: Density matrix or basin coordinates
        metric: Fisher information matrix (required for basin methods)
        method: "bures" (density matrices), "diagonal" (basins), "full" (basins)
        
    Returns:
        Geodesic distance on Fisher manifold
        
    Raises:
        ValueError: If method is invalid or metric is missing
        
    Usage:
        from qigkernels.geometry.distances import fisher_rao_distance
        
        # For density matrices
        distance = fisher_rao_distance(rho1, rho2, method="bures")
        
        # For basin coordinates with diagonal metric
        distance = fisher_rao_distance(
            basin1, basin2, 
            metric=fisher_metric_diag,
            method="diagonal"
        )
        
        # For basin coordinates with full metric
        distance = fisher_rao_distance(
            basin1, basin2,
            metric=fisher_metric_full,
            method="full"
        )
    
    Source:
        Validated in qig-consciousness/qig_consciousness_qfi_attention.py
        Math: d(ρ₁, ρ₂) = √(2(1 - √F)) where F = quantum fidelity
    """
    # Convert torch to numpy if needed
    if torch is not None:
        if isinstance(state_a, torch.Tensor):
            state_a = state_a.detach().cpu().numpy()
        if isinstance(state_b, torch.Tensor):
            state_b = state_b.detach().cpu().numpy()
        if metric is not None and isinstance(metric, torch.Tensor):
            metric = metric.detach().cpu().numpy()
    
    state_a = np.asarray(state_a)
    state_b = np.asarray(state_b)

    # ---------------------------------------------------------------------
    # SIMPLEX BASINS (canonical Fisher-Rao)
    # ---------------------------------------------------------------------
    if state_a.ndim == 1 and state_b.ndim == 1:
        a = np.asarray(state_a, dtype=np.float64).flatten()
        b = np.asarray(state_b, dtype=np.float64).flatten()

        if method == "bures":
            assert_basin_valid(a, name="state_a")
            assert_basin_valid(b, name="state_b")
            return float(simplex_fisher_rao_distance(a, b))

        if method == "diagonal":
            if metric is None:
                raise ValueError("metric is required for method='diagonal'")
            g_diag = np.asarray(metric, dtype=np.float64).flatten()
            if g_diag.shape != a.shape:
                raise ValueError(
                    f"Diagonal metric shape mismatch: metric.shape={g_diag.shape}, state.shape={a.shape}"
                )
            dx = a - b
            return float(np.sqrt(np.sum(np.clip(g_diag, 0.0, np.inf) * (dx * dx))))

        if method == "full":
            if metric is None:
                raise ValueError("metric is required for method='full'")
            G = np.asarray(metric, dtype=np.float64)
            if G.ndim != 2 or G.shape[0] != G.shape[1] or G.shape[0] != a.shape[0]:
                raise ValueError(
                    f"Full metric shape mismatch: metric.shape={G.shape}, state.shape={a.shape}"
                )
            dx = (a - b).reshape(-1, 1)
            d2 = float((dx.T @ G @ dx).squeeze())
            return float(np.sqrt(max(d2, 0.0)))

        raise ValueError(
            f"Unknown method: {method}. Expected 'bures' (simplex basins), 'diagonal', or 'full' (metric basins)."
        )

    # ---------------------------------------------------------------------
    # DENSITY MATRICES (Bures via quantum fidelity)
    # ---------------------------------------------------------------------
    if method == "bures":
        fidelity = quantum_fidelity(state_a, state_b)
        distance = np.sqrt(np.clip(2 * (1 - np.sqrt(np.clip(fidelity, 0, 1))), 0, 4))
        return float(distance)

    # Deprecated basin modes (not simplex Fisher-Rao)
    if method in {"diagonal", "full"}:
        raise ValueError(
            f"Deprecated method: {method}. "
            "Basin geometry must use canonical simplex Fisher-Rao from qig_geometry.canonical."
        )

    raise ValueError(
        f"Unknown method: {method}. Expected 'bures' for density matrices."
    )


def geodesic_distance(
    point_a: np.ndarray,
    point_b: np.ndarray,
    metric: np.ndarray
) -> float:
    """
    Compute geodesic distance on Riemannian manifold.
    
    This is an alias for fisher_rao_distance with method="full"
    for backward compatibility.
    
    Args:
        point_a: First point on manifold
        point_b: Second point on manifold
        metric: Metric tensor
        
    Returns:
        Geodesic distance
    """
    return fisher_rao_distance(point_a, point_b, metric=metric, method="full")


__all__ = [
    "quantum_fidelity",
    "fisher_rao_distance",
    "geodesic_distance",
]
