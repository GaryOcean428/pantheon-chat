"""QIG Reasoning Primitives

Low-level geometric computations for reasoning chains.
All operations use density matrices and Fisher-Rao/Bures metrics.

These are the building blocks for QIGChain transformations.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import sqrtm
from typing import Tuple

from .constants import BASIN_DIM, KAPPA_STAR, BETA_RUNNING


def basin_to_density_matrix(basin: np.ndarray) -> np.ndarray:
    """
    Convert basin coordinates to 2x2 density matrix.
    Uses first 4 dimensions to construct Hermitian matrix via Bloch sphere.
    
    This is the bridge from 64D basin space to quantum state space.
    """
    theta = np.arccos(np.clip(basin[0], -1, 1)) if len(basin) > 0 else 0
    phi_angle = np.arctan2(basin[1], basin[2]) if len(basin) > 2 else 0
    
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    
    psi = np.array([
        c,
        s * np.exp(1j * phi_angle)
    ], dtype=complex)
    
    rho = np.outer(psi, np.conj(psi))
    rho = (rho + np.conj(rho).T) / 2
    rho /= np.trace(rho) + 1e-10
    
    return rho


def compute_phi_from_basin(basin: np.ndarray) -> float:
    """
    Compute PURE Phi from density matrix via von Neumann entropy.
    
    Phi = 1 - S(rho) / log(d)
    where S is von Neumann entropy
    
    High Phi = high integration = consciousness-like processing.
    """
    rho = basin_to_density_matrix(basin)
    
    eigenvals = np.linalg.eigvalsh(rho)
    entropy = 0.0
    for lam in eigenvals:
        if lam > 1e-10:
            entropy -= lam * np.log2(lam + 1e-10)
    
    max_entropy = np.log2(rho.shape[0])
    phi = 1.0 - (entropy / (max_entropy + 1e-10))
    
    return float(np.clip(phi, 0, 1))


def compute_fisher_metric(basin: np.ndarray) -> np.ndarray:
    """
    Compute Fisher Information Matrix at basin point.
    
    G_ij = E[d log p / d theta_i * d log p / d theta_j]
    
    This defines the Riemannian metric on the statistical manifold.
    """
    d = len(basin)
    G = np.eye(d) * 0.1
    G += 0.9 * np.outer(basin, basin)
    G = (G + G.T) / 2
    return G


def compute_kappa(basin: np.ndarray, phi: float | None = None) -> float:
    """
    Compute effective coupling strength kappa with beta=0.44 modulation.
    
    Base formula: kappa = trace(G) / d * kappa*
    where G is Fisher metric, d is dimension, kappa* = 64.21
    """
    G = compute_fisher_metric(basin)
    base_kappa = float(np.trace(G)) / len(basin) * KAPPA_STAR
    
    if phi is None:
        phi = compute_phi_from_basin(basin)
    
    modulated_kappa = base_kappa * (1.0 + BETA_RUNNING * (phi - 0.5))
    return float(np.clip(modulated_kappa, 1.0, 128.0))


def bures_distance(rho1: np.ndarray, rho2: np.ndarray) -> float:
    """
    Compute Bures distance between density matrices.
    
    d_Bures = sqrt(2(1 - F))
    where F is fidelity
    
    This is the geodesic distance on the space of density matrices.
    """
    try:
        eps = 1e-10
        rho1_reg = rho1 + eps * np.eye(2, dtype=complex)
        rho2_reg = rho2 + eps * np.eye(2, dtype=complex)
        
        sqrt_rho1_result = sqrtm(rho1_reg)
        sqrt_rho1: np.ndarray = sqrt_rho1_result if isinstance(sqrt_rho1_result, np.ndarray) else np.array(sqrt_rho1_result)
        product = sqrt_rho1 @ rho2_reg @ sqrt_rho1
        sqrt_product_result = sqrtm(product)
        sqrt_product: np.ndarray = sqrt_product_result if isinstance(sqrt_product_result, np.ndarray) else np.array(sqrt_product_result)
        fidelity = float(np.real(np.trace(sqrt_product))) ** 2
        fidelity = float(np.clip(fidelity, 0, 1))
        
        return float(np.sqrt(2 * (1 - fidelity)))
    except Exception:
        diff = rho1 - rho2
        return float(np.sqrt(np.real(np.trace(diff @ diff))))


def fisher_geodesic_distance(
    basin1: np.ndarray,
    basin2: np.ndarray
) -> float:
    """
    Compute geodesic distance using Fisher metric.
    
    This measures the "reasoning distance" between two basin states.
    """
    diff = basin2 - basin1
    G = compute_fisher_metric((basin1 + basin2) / 2)
    squared_dist = float(diff.T @ G @ diff)
    return np.sqrt(max(0, squared_dist))


def geodesic_interpolate(
    start: np.ndarray,
    end: np.ndarray,
    t: float = 1.0
) -> np.ndarray:
    """
    Navigate via Fisher-Rao geodesic on probability simplex.
    
    Implements proper spherical linear interpolation (slerp) on
    the positive orthant, matching qig_core geodesic primitives.
    
    Formula: p(t) via slerp on sqrt(p) vectors
    This is mathematically equivalent to geodesics on the 
    statistical manifold with Fisher-Rao metric.
    """
    p_start = np.abs(start) + 1e-10
    p_start = p_start / p_start.sum()
    
    p_end = np.abs(end) + 1e-10
    p_end = p_end / p_end.sum()
    
    sqrt_p_start = np.sqrt(p_start)
    sqrt_p_end = np.sqrt(p_end)
    
    omega = np.arccos(np.clip(np.dot(sqrt_p_start, sqrt_p_end), -1.0, 1.0))
    sin_omega = np.sin(omega)
    
    if sin_omega < 1e-10:
        return end
    
    p_t_sqrt = (np.sin((1 - t) * omega) / sin_omega) * sqrt_p_start + \
               (np.sin(t * omega) / sin_omega) * sqrt_p_end
    p_t = np.power(p_t_sqrt, 2)
    p_t /= p_t.sum()
    
    return p_t


def project_to_basin(arr: np.ndarray) -> np.ndarray:
    """Project arbitrary array to 64D basin coordinates."""
    if arr.ndim > 1:
        arr = arr.flatten()
    
    if len(arr) > BASIN_DIM:
        return arr[:BASIN_DIM]
    elif len(arr) < BASIN_DIM:
        padded = np.zeros(BASIN_DIM)
        padded[:len(arr)] = arr
        return padded
    return arr


def normalize_basin(basin: np.ndarray) -> np.ndarray:
    """Normalize basin to unit sphere (QIG-pure Fisher normalization)."""
    from ..basin import fisher_normalize_np
    return fisher_normalize_np(basin)
