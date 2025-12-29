"""Basin projection and signature utilities extracted from qig-consciousness and qig-con2.

Clean implementation focusing on geometry without experiment-specific code.

NOTE: This module is dependency-minimal by design.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn

# Default basin signature dimensionality (D-009)
BASIN_DIM: int = 64


class BasinProjector(nn.Module):
    """
    Projects hidden states to a fixed-size basin signature (default 64D).

    This extracts the core basin projection logic from BasinMatcher in both
    qig-consciousness and qig-con2 repositories.
    """

    def __init__(self, hidden_dim: int, signature_dim: int = 64) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.signature_dim = signature_dim
        self.projection = nn.Linear(hidden_dim, signature_dim)

    def forward(self, hidden_state: Tensor) -> Tensor:
        """
        Project hidden state to basin signature.

        Args:
            hidden_state: Hidden states with shape (batch, seq, hidden_dim)

        Returns:
            Basin signatures with shape (batch, signature_dim)
        """
        # Mean-pool over sequence dimension, then project
        pooled = hidden_state.mean(dim=1)  # (batch, hidden_dim)
        signature = self.projection(pooled)  # (batch, signature_dim)
        return signature


def compute_signature(
    projector: BasinProjector,
    hidden_state: Tensor,
) -> Tensor:
    """
    Compute a single basin signature from a batch of hidden states.

    Args:
        projector: BasinProjector instance
        hidden_state: Hidden states with shape (batch, seq, hidden_dim)

    Returns:
        Single signature vector with shape (signature_dim,)
    """
    batch_signatures = projector(hidden_state)  # (batch, signature_dim)
    return batch_signatures.mean(dim=0)  # (signature_dim,)


def basin_distance(
    a: Tensor,
    b: Tensor,
    use_fisher: bool = True,
) -> Tensor:
    """
    Compute Fisher-Rao distance between basin signatures.

    Args:
        a: First basin signature [..., D]
        b: Second basin signature [..., D]
        use_fisher: If True, use Fisher-Rao distance (default).
                   If False, use Euclidean L2 (NOT recommended).

    Returns:
        Distance tensor (scalar or batch)

    Mathematical Foundation:
        Bures: d²(p₁, p₂) = 2(1 - √F(p₁, p₂))
        where F is quantum fidelity approximated by cosine similarity.
        This respects the curved information manifold structure.
    """
    if a is b:
        return torch.zeros((), dtype=a.dtype, device=a.device)
    if use_fisher:
        # Bures approximation: d² = 2(1 - cos_sim)
        # cos_sim ≈ quantum fidelity for normalized coordinates.
        cos_sim = torch.nn.functional.cosine_similarity(
            a.unsqueeze(0) if a.dim() == 1 else a,
            b.unsqueeze(0) if b.dim() == 1 else b,
            dim=-1,
        )
        distance_sq = 2.0 * (1.0 - cos_sim)
        return torch.sqrt(torch.clamp(distance_sq, min=1e-8)).squeeze()

    # Euclidean fallback (VIOLATES geometric purity - use only for debugging)
    return torch.linalg.norm(a - b, dim=-1)


def fisher_spread(
    coords: Tensor,
    centroid: Tensor | None = None,
) -> Tensor:
    """
    Compute Fisher-Rao spread of coordinates around centroid.

    This is the QIG-pure alternative to torch.norm() for measuring
    coordinate dispersion. Uses cosine-based angular distance which
    respects the curved Fisher information manifold.

    Args:
        coords: Coordinates tensor [B, S, D] or [S, D]
        centroid: Optional centroid [B, 1, D] or [1, D]. If None, computed as mean.

    Returns:
        Mean Fisher distance from centroid [B] or scalar
    """
    if centroid is None:
        if coords.dim() == 3:
            centroid = coords.mean(dim=1, keepdim=True)  # [B, 1, D]
        else:
            centroid = coords.mean(dim=0, keepdim=True)  # [1, D]

    # Normalize to unit sphere for Fisher-Rao
    coords_norm = torch.nn.functional.normalize(coords, dim=-1)
    centroid_norm = torch.nn.functional.normalize(centroid, dim=-1)

    # Cosine similarity: cos(θ) where θ is angle on unit sphere
    cos_sim = (coords_norm * centroid_norm).sum(dim=-1)  # [B, S] or [S]

    # Fisher-Rao distance: d = arccos(cos_sim) ≈ sqrt(2(1-cos_sim)) for small angles
    # Using Bures metric approximation which is numerically stable
    distance_sq = 2.0 * (1.0 - cos_sim.clamp(-1, 1))
    distances = torch.sqrt(distance_sq.clamp(min=1e-8))

    # Mean distance across sequence dimension
    if coords.dim() == 3:
        return distances.mean(dim=-1)  # [B]
    return distances.mean()  # scalar


def kappa_from_fisher_spread(spread: Tensor, kappa_star: float = 64.0) -> Tensor:
    """
    Compute coupling κ from Fisher spread.

    Lower spread = tighter clustering = higher κ.
    κ = κ* × (1 - spread × scaling_factor)

    Args:
        spread: Fisher spread value(s)
        kappa_star: Target fixed point (default 64.0 = E8 rank²)

    Returns:
        Coupling strength κ
    """
    # Spread of 0 → κ = κ*, spread → ∞ → κ → 0
    # Scale factor tuned so typical spreads give reasonable κ range
    return kappa_star * (1.0 - spread.clamp(0, 1) * 0.5)


def fisher_normalize_np(arr: "np.ndarray") -> "np.ndarray":
    """
    Normalize numpy array to unit sphere (QIG-pure alternative to arr/np.linalg.norm(arr)).

    Args:
        arr: Input array of any shape, normalized along last axis

    Returns:
        Unit-normalized array
    """
    import numpy as np
    norm = np.sqrt(np.sum(arr * arr, axis=-1, keepdims=True))
    return arr / (norm + 1e-10)


def fisher_distance_np(a: "np.ndarray", b: "np.ndarray") -> float:
    """
    Compute Fisher-Rao distance between two numpy arrays (QIG-pure).

    Uses Bures metric approximation: d² = 2(1 - cos_sim)

    Args:
        a: First array (will be normalized)
        b: Second array (will be normalized)

    Returns:
        Fisher-Rao distance (scalar)
    """
    import numpy as np
    a_norm = fisher_normalize_np(a.flatten())
    b_norm = fisher_normalize_np(b.flatten())
    cos_sim = np.clip(np.dot(a_norm, b_norm), -1.0, 1.0)
    return float(np.sqrt(2.0 * (1.0 - cos_sim)))


def fisher_spread_np(basins: "np.ndarray", centroid: "np.ndarray | None" = None) -> float:
    """
    Compute Fisher-Rao spread of basins around centroid (QIG-pure numpy version).

    Args:
        basins: Array of basins [N, D] or list of 1D arrays
        centroid: Optional centroid. If None, computed as mean.

    Returns:
        Mean Fisher distance from centroid
    """
    import numpy as np
    basins = np.array(basins)
    if basins.ndim == 1:
        basins = basins.reshape(1, -1)

    if centroid is None:
        centroid = basins.mean(axis=0)

    # Normalize all to unit sphere
    basins_norm = fisher_normalize_np(basins)
    centroid_norm = fisher_normalize_np(centroid.reshape(1, -1))

    # Cosine similarities
    cos_sims = np.clip(np.sum(basins_norm * centroid_norm, axis=-1), -1.0, 1.0)
    distances = np.sqrt(2.0 * (1.0 - cos_sims))

    return float(np.mean(distances))


def save_signature(path: str | Path, sig: Tensor) -> None:
    """
    Persist a basin signature to disk.

    Uses the same format as both source repositories for compatibility.

    Args:
        path: File path to save signature
        sig: Basin signature tensor
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Save in numpy format for compatibility
    import numpy as np
    np.savez_compressed(path, signature=sig.detach().cpu().numpy())


def load_signature(path: str | Path) -> Tensor:
    """
    Load a basin signature from disk.

    Args:
        path: File path to load signature from

    Returns:
        Basin signature tensor
    """
    import numpy as np
    data = np.load(Path(path), allow_pickle=False)
    return torch.from_numpy(data["signature"])
