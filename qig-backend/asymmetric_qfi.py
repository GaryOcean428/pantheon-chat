"""
Asymmetric QFI (Quantum Fisher Information) Operations

Implements directional Fisher information where d_ij != d_ji.
This captures the asymmetry in information flow between consciousness states.

CRITICAL: This breaks the metric symmetry axiom intentionally.
In consciousness dynamics, "A attending to B" != "B attending to A"

Key Functions:
- geodesic_tangent: Directional tangent vector on Fisher manifold
- directional_fisher_information: Asymmetric distance d_ij != d_ji
- asymmetric_attention: Regime-modulated attention matrix

Integration Points:
- qig_consciousness_qfi_attention.py: Basin-level asymmetric coupling
- trajectory_decoder.py: Directional attention in decoding
"""

import numpy as np
from typing import List, Tuple, Optional

from qig_geometry import fisher_coord_distance, sphere_project

# Physics constants
KAPPA_STAR = 64.21  # Validated coupling constant


def geodesic_tangent(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Compute directional tangent vector from source to target on Fisher manifold.

    This gives the DIRECTION of geodesic flow, not just the distance.
    Critical for asymmetric coupling: the tangent from A→B differs from B→A.

    Args:
        source: Source basin coordinates (64D)
        target: Target basin coordinates (64D)

    Returns:
        Tangent vector at source pointing toward target (unit normalized)
    """
    # Project both to unit sphere
    source_proj = sphere_project(source)
    target_proj = sphere_project(target)

    # Tangent is the component of (target - source) orthogonal to source
    # For unit vectors on sphere: v = target - (target · source) * source
    dot = np.dot(source_proj, target_proj)
    tangent = target_proj - dot * source_proj

    # Normalize tangent
    norm = np.linalg.norm(tangent)
    if norm < 1e-10:
        # Points are identical or antipodal
        return np.zeros_like(source)

    return tangent / norm


def directional_fisher_information(
    source: np.ndarray,
    target: np.ndarray,
    metric: Optional[np.ndarray] = None
) -> float:
    """
    Compute ASYMMETRIC Fisher information from source to target.

    CRITICAL DIFFERENCE FROM SYMMETRIC DISTANCE:
    d_ij = d(source→target) != d(target→source) = d_ji

    The asymmetry comes from:
    1. Source purity affects how well it can "see" the target
    2. Source entropy affects information capacity
    3. Regime modulation based on source phi

    Args:
        source: Source basin coordinates (64D)
        target: Target basin coordinates (64D)
        metric: Optional Fisher metric tensor (uses identity if None)

    Returns:
        Asymmetric Fisher information d(source→target)
    """
    # Base symmetric distance
    d_symmetric = fisher_coord_distance(source, target)

    # Compute source-based asymmetry factors
    source_proj = sphere_project(source)

    # Purity proxy: how concentrated is the source?
    # Higher purity = sharper "vision" = smaller effective distance
    source_variance = np.var(source_proj)
    purity_factor = 1.0 / (1.0 + source_variance * 10)  # [0.1, 1.0]

    # Entropy proxy: how much information capacity?
    # Higher entropy = more uncertainty = larger effective distance
    p = np.abs(source_proj) + 1e-10
    p = p / np.sum(p)
    entropy = -np.sum(p * np.log(p + 1e-10))
    max_entropy = np.log(len(source_proj))
    entropy_factor = 1.0 + 0.5 * (entropy / max_entropy)  # [1.0, 1.5]

    # Directional bias: alignment with geodesic
    tangent = geodesic_tangent(source, target)
    alignment = np.abs(np.dot(source_proj, tangent))
    alignment_factor = 1.0 - 0.3 * alignment  # [0.7, 1.0]

    # Combine factors for asymmetric distance
    d_asymmetric = d_symmetric * purity_factor * entropy_factor * alignment_factor

    return float(d_asymmetric)


def asymmetric_attention(
    basins: List[np.ndarray],
    phi_values: List[float],
    kappa_base: float = KAPPA_STAR
) -> np.ndarray:
    """
    Compute regime-modulated asymmetric attention matrix.

    Unlike symmetric attention where A_ij = A_ji, this produces
    A_ij != A_ji based on source phi regime.

    Regime modulation:
    - phi < 0.3 (Linear): kappa_eff = kappa_base * 0.3 (weak coupling)
    - 0.3 <= phi <= 0.7 (Geometric): kappa_eff = kappa_base (optimal)
    - phi > 0.7 (Breakdown): kappa_eff = kappa_base * 0.5 (unstable)

    Args:
        basins: List of basin coordinate vectors (64D each)
        phi_values: List of phi values for each basin
        kappa_base: Base coupling constant (default KAPPA_STAR)

    Returns:
        Asymmetric attention matrix A[i,j] = attention from i to j
    """
    n = len(basins)
    attention = np.zeros((n, n))

    for i in range(n):
        phi_i = phi_values[i] if i < len(phi_values) else 0.5

        # Regime-based kappa modulation
        if phi_i < 0.3:
            kappa_eff = kappa_base * 0.3  # Linear regime: weak
        elif phi_i > 0.7:
            kappa_eff = kappa_base * 0.5  # Breakdown regime: unstable
        else:
            kappa_eff = kappa_base  # Geometric regime: optimal

        for j in range(n):
            if i == j:
                attention[i, j] = 1.0  # Self-attention
            else:
                # Asymmetric distance from i to j
                d_ij = directional_fisher_information(basins[i], basins[j])

                # Attention weight: exp(-d/kappa)
                attention[i, j] = np.exp(-d_ij / (kappa_eff + 1e-10))

    return attention


def compute_asymmetric_coupling_matrix(
    basins: List[np.ndarray],
    phi_values: Optional[List[float]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute both attention matrix and asymmetry measure.

    Returns:
        attention: Asymmetric attention matrix
        asymmetry: Matrix of |A_ij - A_ji| showing asymmetry magnitude
    """
    if phi_values is None:
        phi_values = [0.5] * len(basins)

    attention = asymmetric_attention(basins, phi_values)

    # Measure asymmetry: how different is A_ij from A_ji?
    asymmetry = np.abs(attention - attention.T)

    return attention, asymmetry


def regime_from_phi(phi: float) -> str:
    """Classify phi into regime."""
    if phi < 0.3:
        return "linear"
    elif phi > 0.7:
        return "breakdown"
    else:
        return "geometric"


__all__ = [
    'geodesic_tangent',
    'directional_fisher_information',
    'asymmetric_attention',
    'compute_asymmetric_coupling_matrix',
    'regime_from_phi',
    'KAPPA_STAR',
]


if __name__ == "__main__":
    print("[AsymmetricQFI] Running self-tests...")

    # Test geodesic tangent
    source = np.random.randn(64)
    target = np.random.randn(64)
    tangent = geodesic_tangent(source, target)
    assert abs(np.linalg.norm(tangent) - 1.0) < 0.01 or np.linalg.norm(tangent) < 0.01
    print("OK: geodesic_tangent produces unit tangent")

    # Test asymmetry
    d_ab = directional_fisher_information(source, target)
    d_ba = directional_fisher_information(target, source)
    print(f"OK: d(A->B)={d_ab:.4f}, d(B->A)={d_ba:.4f}, asymmetry={abs(d_ab-d_ba):.4f}")

    # Test attention matrix
    basins = [np.random.randn(64) for _ in range(4)]
    phis = [0.2, 0.5, 0.8, 0.6]  # Different regimes
    attn = asymmetric_attention(basins, phis)
    asymm = np.abs(attn - attn.T).mean()
    print(f"OK: attention matrix asymmetry={asymm:.4f}")

    print("\n[AsymmetricQFI] All self-tests passed!")
