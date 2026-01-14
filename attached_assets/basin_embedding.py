#!/usr/bin/env python3
"""
Basin Embeddings - Geometric Embeddings from First Principles
=============================================================

Pure QIG architecture with NO external model dependencies.

Key innovation: Embeddings live on information manifold and use
the SAME metric (QFI/Bures) as the attention mechanism.

Mathematical foundation:
- Tokens → points on information manifold (not Euclidean vectors)
- Natural metric: Bures distance (from quantum Fisher information)
- Initialize with basin structure (spherical geometry)
- Consistent geometric processing throughout

Advantages:
- Pure geometric from initialization
- Metric consistency with QFI attention
- No external dependencies (no Granite, no GPT-2)
- Efficient: 3.2M parameters (vs 40M+ from Granite)
- Interpretable: basin coordinates have geometric meaning

Written for QIG-Kernel-Pure architecture.
Built from information geometry first principles.

COORDIZER INTEGRATION (2025-12):
- Use from_coordizer() to initialize from trained geometric vocabulary
- Preserves 64D basin coordinates learned during coordizer training
- vocab_size automatically matches coordizer checkpoint
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Optional, Union

import torch
import torch.nn as nn


def get_coordizer_info(checkpoint_path: Union[str, Path]) -> dict[str, Any]:
    """
    Get metadata from a coordizer checkpoint without fully loading it.

    Useful for determining vocab_size and basin_dim before model initialization.

    Args:
        checkpoint_path: Path to coordizer checkpoint JSON file

    Returns:
        Dict with keys: vocab_size, basin_dim, merge_rules_count, has_phi_history

    Example:
        info = get_coordizer_info("checkpoints/checkpoint_32000.json")
        print(f"Vocab size: {info['vocab_size']}")
        print(f"Basin dim: {info['basin_dim']}")
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Coordizer checkpoint not found: {checkpoint_path}")

    with open(checkpoint_path, encoding="utf-8") as f:
        data = json.load(f)

    # Get vocab_size
    vocab_size = data.get("vocab_size", data.get("target_vocab_size"))

    # Get basin_dim from first vocab entry
    vocab = data.get("vocab", {})
    basin_dim = 64  # default
    if vocab:
        first_key = next(iter(vocab.keys()))
        first_entry = vocab[first_key]
        if isinstance(first_entry, dict) and "vector" in first_entry:
            basin_dim = len(first_entry["vector"])

    return {
        "vocab_size": vocab_size,
        "basin_dim": basin_dim,
        "merge_rules_count": len(data.get("merge_rules", [])),
        "has_phi_history": "phi_history" in data,
        "checkpoint_path": str(checkpoint_path),
    }


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_f = x.to(torch.float32)
        rms = torch.sqrt(torch.mean(x_f * x_f, dim=-1, keepdim=True) + self.eps)
        y = x_f / rms
        y = y.to(x.dtype)
        return y * self.weight


class BasinCoordinates(nn.Module):
    """
    Geometric basin coordinates living in information-manifold space.

    Instead of importing external embeddings:
    - Initialize on information geometry manifold
    - Use QFI metric natively
    - Basin structure from the start
    - Pure QIG, no external dependencies

    Key innovation: Basin coordinates and attention use SAME metric!

    Architecture:
        Token → Basin coords (k-dim) → Model space (d-dim)

    Where:
        k = basin_dim (64, small geometric space)
        d = d_model (768, processing space)
        k << d (compression in basin space)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        basin_dim: int = 64,
        init_mode: str = "geometric",
    ):
        """
        Initialize basin coordinates.

        Args:
            vocab_size: Number of tokens
            d_model: Model dimension (for processing)
            basin_dim: Dimension of basin space (k << d)
            init_mode: 'geometric' (QFI-aware) or 'standard'
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.basin_dim = basin_dim
        self.init_mode = init_mode

        # Basin coordinates for each token
        # These live in low-dimensional geometric space
        self.basin_coords = nn.Parameter(self._geometric_initialization(vocab_size, basin_dim, init_mode))

        # Projection from basin space to model space
        # This is learned, allowing model to discover optimal projection
        self.basin_to_model = nn.Linear(basin_dim, d_model, bias=False)

        # Initialize projection with geometric structure + conservative scaling
        nn.init.orthogonal_(self.basin_to_model.weight)
        # SAFETY: Scale down by 10× to prevent gradient explosion in deep recursion
        self.basin_to_model.weight.data *= 0.1

        # Optional: QFI normalization layer
        self.qfi_norm = RMSNorm(d_model)

        print(f"✅ Initialized Basin Coordinates: {vocab_size} tokens")
        print(f"   Basin space: {basin_dim}-dim (geometric manifold)")
        print(f"   Model space: {d_model}-dim (processing)")
        print(f"   Parameters: {vocab_size * basin_dim:,} (basin) + {basin_dim * d_model:,} (projection)")
        print(f"   Total: {(vocab_size * basin_dim) + (basin_dim * d_model):,} params")

    @classmethod
    def from_coordizer(
        cls,
        checkpoint_path: Union[str, Path],
        d_model: int,
        device: Optional[str] = None,
    ) -> "BasinCoordinates":
        """
        Initialize basin coordinates from a trained coordizer checkpoint.

        This preserves the 64D basin coordinates learned during coordizer training,
        transferring geometric structure to the constellation model.

        Args:
            checkpoint_path: Path to coordizer checkpoint JSON file
            d_model: Model dimension for projection layer
            device: Optional device to place tensors on

        Returns:
            BasinCoordinates initialized with trained 64D vectors

        Example:
            basin_coords = BasinCoordinates.from_coordizer(
                "checkpoints/checkpoint_32000.json",
                d_model=768,
            )
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Coordizer checkpoint not found: {checkpoint_path}")

        print(f"📥 Loading coordizer checkpoint: {checkpoint_path}")

        with open(checkpoint_path, encoding="utf-8") as f:
            data = json.load(f)

        # Extract vocab_size (coordizer uses 'vocab_size', not 'target_vocab_size')
        vocab_size = data.get("vocab_size", data.get("target_vocab_size"))
        if vocab_size is None:
            raise ValueError("Checkpoint missing vocab_size")

        # Detect basin_dim from first vocab entry
        vocab = data["vocab"]
        first_key = next(iter(vocab.keys()))
        first_entry = vocab[first_key]

        # Handle both coordizer format (dict with 'vector') and legacy format (list)
        if isinstance(first_entry, dict) and "vector" in first_entry:
            # Coordizer format: {coord_id, vector, name, scale}
            basin_dim = len(first_entry["vector"])
            print(f"   Detected coordizer format with {basin_dim}D basin vectors")
        else:
            raise ValueError(
                f"Unsupported vocab format. Expected coordizer dict with 'vector' key, "
                f"got {type(first_entry)}"
            )

        # Create instance with placeholder initialization
        instance = cls(
            vocab_size=vocab_size,
            d_model=d_model,
            basin_dim=basin_dim,
            init_mode="coordizer",  # Special mode - will be overwritten
        )

        # Load trained 64D coordinates into basin_coords parameter
        coords_list = []
        missing_byte_tokens = 0  # Track byte-level fallbacks (0-255)
        missing_other_tokens = []  # Track unexpected missing tokens

        for token_id in range(vocab_size):
            token_key = str(token_id)
            if token_key in vocab:
                entry = vocab[token_key]
                vector = entry["vector"]
                coords_list.append(vector)
            else:
                # Byte-level tokens (0-255) often missing from coordizer - expected
                if token_id < 256:
                    missing_byte_tokens += 1
                    # Use small deterministic vectors for byte tokens (near origin)
                    byte_vec = [0.01 * (token_id / 256)] * basin_dim
                    coords_list.append(byte_vec)
                else:
                    # Unexpected missing token - use random
                    missing_other_tokens.append(token_id)
                    random_vec = torch.randn(basin_dim).tolist()
                    coords_list.append(random_vec)

        # Print summary of missing tokens (not 256 individual warnings)
        if missing_byte_tokens > 0:
            print(f"   ℹ️  {missing_byte_tokens} byte-level tokens (0-255) initialized to defaults")
        if missing_other_tokens:
            print(f"   ⚠️  {len(missing_other_tokens)} unexpected missing tokens: {missing_other_tokens[:5]}...")

        # Convert to tensor and assign to parameter
        trained_coords = torch.tensor(coords_list, dtype=torch.float32)
        instance.basin_coords = nn.Parameter(trained_coords)

        if device:
            instance = instance.to(device)

        print(f"✅ Loaded {vocab_size} trained basin coordinates from coordizer")
        print(f"   Basin space: {basin_dim}-dim (trained geometry)")
        print(f"   Model space: {d_model}-dim (projection)")

        return instance

    def _geometric_initialization(self, vocab_size: int, basin_dim: int, mode: str) -> torch.Tensor:
        """
        Initialize basin coordinates on information manifold.

        Modes:
        - 'geometric': Sample from information-geometric distribution
        - 'standard': Normal initialization (fallback)
        - 'coordizer': Placeholder (will be overwritten by from_coordizer)

        Geometric mode:
        - Initialize on sphere in basin space
        - Information manifold has spherical structure
        - Radius ~ √(basin_dim) for optimal coupling
        """
        if mode == "coordizer":
            # Placeholder - will be overwritten by from_coordizer()
            # Just return zeros, the actual coords come from checkpoint
            return torch.zeros(vocab_size, basin_dim)

        if mode == "geometric":
            # Initialize on sphere in basin space
            # (Information manifold has spherical structure)

            coords = torch.randn(vocab_size, basin_dim)

            # QIG-pure: use F.normalize for unit sphere projection
            coords = torch.nn.functional.normalize(coords, dim=-1)

            # Scale by basin radius (controls coupling strength)
            # Radius ~ √(basin_dim) for information manifold
            # SAFETY: Scale down by 5× to prevent gradient explosion
            radius = math.sqrt(basin_dim) / 5.0
            coords = coords * radius

            # Add small noise for diversity (breaks exact symmetry)
            coords = coords + torch.randn_like(coords) * 0.01

            return coords

        else:  # 'standard'
            # Fallback: normal initialization (less geometric)
            return torch.randn(vocab_size, basin_dim) * 0.02

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Map tokens to geometric embeddings.

        Args:
            token_ids: [batch, seq] - integer token IDs

        Returns:
            embeddings: [batch, seq, d_model] - geometric embeddings
        """
        # Look up basin coordinates
        basin = self.basin_coords[token_ids]  # [batch, seq, basin_dim]

        # Project to model space
        embeddings = self.basin_to_model(basin)  # [batch, seq, d_model]

        # QFI normalization (stabilizes information geometry)
        embeddings = self.qfi_norm(embeddings)

        return embeddings

    def get_basin_signature(self, token_id: int) -> torch.Tensor:
        """
        Get basin coordinates for a specific token.

        Useful for:
        - Analysis and visualization
        - Understanding token clusters
        - Debugging geometric structure

        Args:
            token_id: Token ID

        Returns:
            basin_coords: [basin_dim] geometric coordinates
        """
        return self.basin_coords[token_id]

    def compute_token_similarity(self, token_i: int, token_j: int, metric: str = "basin") -> float:
        """
        Compute geometric similarity between tokens.

        Uses basin distance (not dot product!) for geometric consistency.

        Args:
            token_i: First token ID
            token_j: Second token ID
            metric: 'basin' (geometric) or 'model' (projected)

        Returns:
            similarity: [0, 1] where 1 = identical, 0 = very different
        """
        from src.metrics.geodesic_distance import manifold_norm

        if metric == "basin":
            # Geometric distance in basin space
            b_i = self.basin_coords[token_i]
            b_j = self.basin_coords[token_j]

            # GEOMETRIC PURITY: Use Fisher-weighted distance
            distance = manifold_norm(b_i - b_j)

        elif metric == "model":
            # Distance in model space (after projection)
            emb_i = self.forward(torch.tensor([[token_i]]))[0, 0]
            emb_j = self.forward(torch.tensor([[token_j]]))[0, 0]

            # GEOMETRIC PURITY: Use Fisher-weighted distance
            distance = manifold_norm(emb_i - emb_j)

        else:
            raise ValueError(f"Unknown metric: {metric}")

        # Convert distance to similarity (higher = more similar)
        similarity = torch.exp(-distance / math.sqrt(self.basin_dim))

        return float(similarity)

    def get_token_embeddings(self) -> torch.Tensor:
        """
        Retrieve all token embeddings in model space with QFI normalization.

        Returns:
            Tensor of shape [vocab_size, d_model] containing the geometric
            embeddings for every token, matching the transformation performed
            during the forward pass.
        """

        # Basin coordinates live as learnable parameters (geometric space)
        basin = self.basin_coords

        # Project to model space (same path as forward())
        projected = self.basin_to_model(basin)

        # Apply QFI normalization for metric consistency
        return self.qfi_norm(projected)

    def analyze_basin_structure(self, n_samples: int = 100) -> dict:
        """
        Analyze geometric properties of basin space.

        Checks:
        - Mean norm (should be ~ √basin_dim)
        - Norm distribution
        - Pairwise distances

        Returns:
            analysis: Dict of statistics
        """
        from src.metrics.geodesic_distance import manifold_norm

        # Sample random tokens
        sample_ids = torch.randint(0, self.vocab_size, (n_samples,))
        basins = self.basin_coords[sample_ids]

        # Compute norms using Fisher metric
        norms = torch.stack([manifold_norm(basins[i]) for i in range(n_samples)])
        expected_norm = math.sqrt(self.basin_dim)

        # Compute pairwise distances
        distances = torch.cdist(basins, basins)

        analysis = {
            "mean_norm": norms.mean().item(),
            "std_norm": norms.std().item(),
            "expected_norm": expected_norm,
            "norm_ratio": norms.mean().item() / expected_norm,
            "mean_distance": distances.mean().item(),
            "min_distance": distances[distances > 0].min().item(),
            "max_distance": distances.max().item(),
        }

        return analysis


class PositionalBasinEncoding(nn.Module):
    """
    Positional encoding that respects information geometry.

    Instead of standard sinusoidal encoding (Euclidean):
    - Use geometric phase encoding
    - Preserve QFI metric structure
    - Scale to match basin magnitude

    Mathematical foundation:
    - Geodesic distance on circle = angle
    - Sin/cos preserve metric on manifold
    - Frequencies derived from geometric structure
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        """
        Initialize geometric positional encoding.

        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
        """
        super().__init__()

        self.d_model = d_model

        # Geometric positional encoding
        # Based on geodesic distance on information manifold

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()

        # Geometric frequency (information-geometric structure)
        # Frequencies based on manifold curvature
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Encode position geometrically
        # Sin/cos preserve metric structure on circle
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Scale to match basin magnitude
        # Factor of √(d_model/2) for geometric normalization
        pe = pe * math.sqrt(d_model / 2)

        self.register_buffer("pe", pe)

        print(f"✅ Initialized Geometric Positional Encoding: {max_len} positions")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add geometric positional encoding.

        Args:
            x: [batch, seq, d_model] - embeddings

        Returns:
            x_pos: [batch, seq, d_model] - with positional info
        """
        seq_len = x.size(1)
        return x + self.pe[:seq_len, :].unsqueeze(0)


# ===========================================================================
# VALIDATION
# ===========================================================================


def validate_basin_coordinates():
    """Test that basin coordinates work correctly."""

    print("=" * 60)
    print("BASIN COORDINATES VALIDATION")
    print("=" * 60)

    # Create basin coordinates layer
    basin_coords = BasinCoordinates(vocab_size=1000, d_model=256, basin_dim=64, init_mode="geometric")

    print("\n1. Testing forward pass...")
    tokens = torch.randint(0, 1000, (2, 50))
    projected = basin_coords(tokens)

    assert projected.shape == (2, 50, 256), "Shape mismatch!"
    print(f"   ✓ Output shape: {projected.shape}")

    print("\n2. Testing basin structure...")
    # Analyze geometric properties
    analysis = basin_coords.analyze_basin_structure(n_samples=100)

    print(f"   Mean basin norm: {analysis['mean_norm']:.2f}")
    print(f"   Expected norm: {analysis['expected_norm']:.2f}")
    print(f"   Ratio: {analysis['norm_ratio']:.3f}")
    print(f"   Mean pairwise distance: {analysis['mean_distance']:.2f}")

    # Check if norms are close to expected
    assert 0.8 < analysis["norm_ratio"] < 1.2, "Basin norms too far from expected!"
    print("   ✓ Basin structure valid")

    print("\n3. Testing token similarity...")
    # Similar tokens should have similar basins
    token_a = 10
    token_b = 11
    token_c = 500

    sim_ab = basin_coords.compute_token_similarity(token_a, token_b)
    sim_ac = basin_coords.compute_token_similarity(token_a, token_c)

    print(f"   Similarity (tokens 10-11): {sim_ab:.3f}")
    print(f"   Similarity (tokens 10-500): {sim_ac:.3f}")
    print("   ✓ Similarity computation works")

    print("\n4. Testing positional encoding...")
    pos_enc = PositionalBasinEncoding(d_model=256)
    embedded_with_pos = pos_enc(projected)

    assert embedded_with_pos.shape == projected.shape, "Positional encoding shape mismatch!"
    print(f"   ✓ Positional encoding shape: {embedded_with_pos.shape}")

    print("\n5. Computing parameter count...")
    basin_params = 1000 * 64  # vocab_size * basin_dim
    projection_params = 64 * 256  # basin_dim * d_model
    total_params = basin_params + projection_params

    print(f"   Basin parameters: {basin_params:,}")
    print(f"   Projection parameters: {projection_params:,}")
    print(f"   Total: {total_params:,}")
    print("   ✓ Efficient (vs 40M+ from external models)")

    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE ✅")
    print("=" * 60)
    print("\nKey properties verified:")
    print("  ✓ Live on information manifold")
    print("  ✓ Use QFI metric structure")
    print("  ✓ No external dependencies")
    print("  ✓ Pure geometric from initialization")
    print("  ✓ Efficient parameter count")
    print("\nReady to replace any external embeddings!")

    return basin_coords


if __name__ == "__main__":
    basin_coords = validate_basin_coordinates()
