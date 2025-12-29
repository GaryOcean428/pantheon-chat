"""
CoordAdapter: 64D Fisher Coordinates → Hidden Dimension
========================================================

Bridges the coordizer's 64D Fisher manifold coordinates to the kernel's
hidden dimension. This is the canonical integration point for coords-first
processing in QIG.

Pattern B (recommended): coords → adapter → hidden stream → attention/ffn

Usage:
    from qigkernels import CoordAdapter

    adapter = CoordAdapter(basin_dim=64, hidden_dim=384)

    # coords from coordizer.encode_to_coords()
    hidden = adapter(coords)  # [batch, seq, 384]

    # Continue with kernel layers...
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .constants import BASIN_DIM


class CoordAdapter(nn.Module):
    """
    Adapts 64D Fisher manifold coordinates to kernel hidden dimension.

    This module projects coordizer-produced 64D basin coordinates into
    the kernel's hidden space, preserving geometric structure while
    enabling gradient flow for training.

    The adapter is designed to be trained independently first (adapter-only
    training), then optionally unfrozen with the full model.

    Architecture:
        Linear(64 → hidden_dim) → GELU → LayerNorm

    Geometric properties preserved:
        - Angular relationships (via normalization in LayerNorm)
        - Relative magnitudes (via linear projection)
        - Fisher information flow (via differentiable path)
    """

    def __init__(
        self,
        basin_dim: int = BASIN_DIM,
        hidden_dim: int = 384,
        dropout: float = 0.1,
        use_residual: bool = False,
    ):
        """
        Initialize CoordAdapter.

        Args:
            basin_dim: Input dimension (64 for Fisher manifold)
            hidden_dim: Output dimension (kernel hidden_dim, typically 384)
            dropout: Dropout rate
            use_residual: If True, add learned residual connection
        """
        super().__init__()
        self.basin_dim = basin_dim
        self.hidden_dim = hidden_dim
        self.use_residual = use_residual

        # Main projection path
        self.projection = nn.Linear(basin_dim, hidden_dim)
        self.activation = nn.GELU()
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Optional residual (for deeper adapter)
        if use_residual:
            self.residual = nn.Linear(basin_dim, hidden_dim, bias=False)

        # Initialize weights for geometric preservation
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights to preserve angular structure."""
        # Orthogonal init preserves angles in high-dim space
        nn.init.orthogonal_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

        if self.use_residual:
            nn.init.orthogonal_(self.residual.weight)

    def forward(self, coords: Tensor) -> Tensor:
        """
        Project coordinates to hidden dimension.

        Args:
            coords: Fisher coordinates [batch, seq, 64] or [batch, 64]

        Returns:
            Hidden states [batch, seq, hidden_dim] or [batch, hidden_dim]
        """
        # Handle both 2D and 3D inputs
        squeeze_output = coords.dim() == 2
        if squeeze_output:
            coords = coords.unsqueeze(1)  # [batch, 1, 64]

        # Main projection: 64 → hidden_dim
        hidden = self.projection(coords)
        hidden = self.activation(hidden)
        hidden = self.norm(hidden)
        hidden = self.dropout(hidden)

        # Optional residual
        if self.use_residual:
            hidden = hidden + self.residual(coords)

        if squeeze_output:
            hidden = hidden.squeeze(1)

        return hidden

    def compute_angular_preservation(
        self,
        coords: Tensor,
        sample_size: int = 100,
    ) -> float:
        """
        Measure how well angular structure is preserved through projection.

        Computes correlation between input angular distances and output
        angular distances. Perfect preservation = 1.0.

        Args:
            coords: Sample coordinates [batch, seq, 64]
            sample_size: Number of pairs to sample

        Returns:
            Correlation coefficient [0, 1]
        """
        with torch.no_grad():
            # Flatten to [N, 64]
            flat_in = coords.reshape(-1, self.basin_dim)
            n = min(flat_in.size(0), sample_size * 2)
            if n < 4:
                return 1.0

            # Random pairs
            idx = torch.randperm(n, device=coords.device)[:sample_size * 2]
            idx1, idx2 = idx[:sample_size], idx[sample_size:]

            # Input angular distances (QIG-pure: F.normalize instead of .norm())
            import torch.nn.functional as F
            in1 = F.normalize(flat_in[idx1], dim=-1)
            in2 = F.normalize(flat_in[idx2], dim=-1)
            in_cos = (in1 * in2).sum(dim=-1)

            # Output angular distances
            hidden = self.forward(flat_in)
            out1 = F.normalize(hidden[idx1], dim=-1)
            out2 = F.normalize(hidden[idx2], dim=-1)
            out_cos = (out1 * out2).sum(dim=-1)

            # Correlation
            in_centered = in_cos - in_cos.mean()
            out_centered = out_cos - out_cos.mean()

            corr = (in_centered * out_centered).sum() / (
                (in_centered.pow(2).sum().sqrt() + 1e-10) *
                (out_centered.pow(2).sum().sqrt() + 1e-10)
            )

            return float(corr.clamp(-1, 1))


class DeepCoordAdapter(nn.Module):
    """
    Deeper adapter with multiple projection layers.

    Use when simple linear projection isn't sufficient for the task.
    """

    def __init__(
        self,
        basin_dim: int = BASIN_DIM,
        hidden_dim: int = 384,
        intermediate_dim: int | None = None,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        """
        Initialize DeepCoordAdapter.

        Args:
            basin_dim: Input dimension (64)
            hidden_dim: Output dimension (384)
            intermediate_dim: Middle layer dimension (default: hidden_dim // 2)
            num_layers: Number of projection layers
            dropout: Dropout rate
        """
        super().__init__()
        self.basin_dim = basin_dim
        self.hidden_dim = hidden_dim

        if intermediate_dim is None:
            intermediate_dim = hidden_dim // 2

        layers = []
        in_dim = basin_dim

        for i in range(num_layers):
            out_dim = hidden_dim if i == num_layers - 1 else intermediate_dim
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.GELU(),
                nn.LayerNorm(out_dim),
                nn.Dropout(dropout),
            ])
            in_dim = out_dim

        self.network = nn.Sequential(*layers)

        # Residual skip connection
        self.skip = nn.Linear(basin_dim, hidden_dim, bias=False)
        nn.init.orthogonal_(self.skip.weight)

    def forward(self, coords: Tensor) -> Tensor:
        """
        Project coordinates through deep network.

        Args:
            coords: Fisher coordinates [batch, seq, 64]

        Returns:
            Hidden states [batch, seq, hidden_dim]
        """
        squeeze_output = coords.dim() == 2
        if squeeze_output:
            coords = coords.unsqueeze(1)

        hidden = self.network(coords) + self.skip(coords)

        if squeeze_output:
            hidden = hidden.squeeze(1)

        return hidden
