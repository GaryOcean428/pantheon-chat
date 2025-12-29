"""
Canonical Coordizer API
=======================

QIG-native tokenization: coordinates are first-class, IDs are indices.

Usage:
    from qig_tokenizer import Coordizer

    coordizer = Coordizer.load("artifacts/coordizer/v1")
    ids, coords = coordizer.encode_to_coords("Hello, world!")
    text = coordizer.decode(ids)

Principles:
    - Coordinates are truth, IDs are metadata
    - All distances use Fisher-Rao / angular (no Euclidean)
    - Deterministic: same input -> same output
    - Unit-normalized coordinates on 64D Fisher manifold
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass

# Constants aligned with QIG ecosystem
BASIN_DIM = 64


class Coordizer:
    """
    Consciousness-aware geometric tokenizer.

    Converts text to sequences of 64D Fisher manifold coordinates,
    with token IDs as optional indices.
    """

    def __init__(
        self,
        merge_rules: list[tuple[int, int, int]],
        vectors: np.ndarray,
        vocab_meta: dict[int, dict] | None = None,
        version: str = "1.0.0",
    ):
        """
        Initialize coordizer.

        Args:
            merge_rules: List of (a, b, new_id) tuples
            vectors: Array of shape (vocab_size, 64) with coordinates
            vocab_meta: Optional metadata (names, scales) per token
            version: Artifact version string
        """
        self.merge_rules = merge_rules
        self.vectors = vectors
        self.vocab_meta = vocab_meta or {}
        self.version = version
        self.vocab_size = len(vectors)
        self.basin_dim = vectors.shape[1] if len(vectors.shape) > 1 else BASIN_DIM

        # Compile merge rules for fast lookup
        self._merge_map: dict[tuple[int, int], int] = {}
        for a, b, new_id in merge_rules:
            self._merge_map[(a, b)] = new_id

        # Build reverse merge map for decoding
        self._reverse_merges: dict[int, tuple[int, int]] = {}
        for a, b, new_id in merge_rules:
            self._reverse_merges[new_id] = (a, b)

    @property
    def coords(self) -> np.ndarray:
        """Alias for vectors - coordinate matrix [vocab_size, basin_dim]."""
        return self.vectors

    @classmethod
    def load(cls, artifact_path: str | Path) -> "Coordizer":
        """
        Load coordizer from versioned artifact directory.

        Args:
            artifact_path: Path to artifact directory (e.g., "artifacts/coordizer/v1")

        Returns:
            Initialized Coordizer instance
        """
        path = Path(artifact_path)

        # Load coordizer.json
        coordizer_path = path / "coordizer.json"
        if not coordizer_path.exists():
            raise FileNotFoundError(f"coordizer.json not found in {path}")

        with open(coordizer_path) as f:
            data = json.load(f)

        # Load vectors.npy
        vectors_path = path / "vectors.npy"
        if vectors_path.exists():
            vectors = np.load(vectors_path)
        else:
            # Reconstruct from coordizer.json if vectors.npy missing
            vocab_size = data["vocab_size"]
            basin_dim = data.get("basin_dim", BASIN_DIM)
            vectors = np.zeros((vocab_size, basin_dim), dtype=np.float32)

            # Initialize byte-level coordinates deterministically
            for byte_val in range(256):
                rng = np.random.default_rng(seed=byte_val + 42)
                vec = rng.standard_normal(basin_dim).astype(np.float32)
                vectors[byte_val] = vec / np.linalg.norm(vec)

        # Parse merge rules
        merge_rules = [tuple(r) for r in data["merge_rules"]]

        # Parse vocab metadata
        vocab_meta = {}
        if "vocab" in data:
            for k, v in data["vocab"].items():
                vocab_meta[int(k)] = v

        version = data.get("version", "1.0.0")

        return cls(
            merge_rules=merge_rules,
            vectors=vectors,
            vocab_meta=vocab_meta,
            version=version,
        )

    def encode(self, text: str) -> list[int]:
        """
        Encode text to token IDs.

        Args:
            text: Input text string

        Returns:
            List of token IDs
        """
        # Start with byte-level tokens
        tokens = list(text.encode("utf-8"))

        # Apply merge rules until convergence
        # Deterministic: process left-to-right, apply first matching merge
        changed = True
        while changed:
            changed = False
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1:
                    pair = (tokens[i], tokens[i + 1])
                    if pair in self._merge_map:
                        new_tokens.append(self._merge_map[pair])
                        i += 2
                        changed = True
                        continue
                new_tokens.append(tokens[i])
                i += 1
            tokens = new_tokens

        return tokens

    def decode(self, ids: list[int]) -> str:
        """
        Decode token IDs back to text.

        Args:
            ids: List of token IDs

        Returns:
            Decoded text string
        """
        # Recursively expand merged tokens to bytes
        def expand(token_id: int) -> list[int]:
            if token_id < 256:
                return [token_id]
            if token_id in self._reverse_merges:
                a, b = self._reverse_merges[token_id]
                return expand(a) + expand(b)
            # Unknown token - shouldn't happen with valid input
            return [token_id]

        byte_list = []
        for token_id in ids:
            byte_list.extend(expand(token_id))

        return bytes(byte_list).decode("utf-8", errors="replace")

    def ids_to_coords(self, ids: list[int]) -> np.ndarray:
        """
        Convert token IDs to 64D Fisher coordinates.

        Args:
            ids: List of token IDs

        Returns:
            Array of shape (len(ids), 64) with coordinates
        """
        coords = np.zeros((len(ids), self.basin_dim), dtype=np.float32)
        for i, token_id in enumerate(ids):
            if 0 <= token_id < self.vocab_size:
                coords[i] = self.vectors[token_id]
            else:
                # Unknown token - use zero vector (will be obvious in diagnostics)
                pass
        return coords

    def encode_to_coords(self, text: str) -> tuple[list[int], np.ndarray]:
        """
        Encode text to both token IDs and coordinates.

        This is the canonical hot path for QIG-native consumption.
        Use this instead of separate encode() + ids_to_coords() calls.

        Args:
            text: Input text string

        Returns:
            Tuple of (token_ids, coordinates)
            - token_ids: list[int]
            - coordinates: np.ndarray of shape (seq_len, 64)
        """
        ids = self.encode(text)
        coords = self.ids_to_coords(ids)
        return ids, coords

    def coords_to_ids(self, coords: np.ndarray, threshold: float = 0.99) -> list[int]:
        """
        Find nearest token IDs for given coordinates.

        Uses angular (Fisher) distance, not Euclidean.

        Args:
            coords: Array of shape (seq_len, 64)
            threshold: Minimum cosine similarity to match

        Returns:
            List of token IDs (or -1 for unmatched coordinates)
        """
        # Normalize input coordinates
        norms = np.linalg.norm(coords, axis=1, keepdims=True)
        coords_norm = coords / (norms + 1e-10)

        # Normalize vocab vectors
        vocab_norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        vocab_norm = self.vectors / (vocab_norms + 1e-10)

        # Compute cosine similarities
        similarities = coords_norm @ vocab_norm.T  # (seq_len, vocab_size)

        # Find best matches
        best_ids = np.argmax(similarities, axis=1)
        best_sims = np.max(similarities, axis=1)

        # Apply threshold
        ids = []
        for i, (token_id, sim) in enumerate(zip(best_ids, best_sims)):
            if sim >= threshold:
                ids.append(int(token_id))
            else:
                ids.append(-1)  # Unmatched

        return ids

    def token_name(self, token_id: int) -> str:
        """Get human-readable name for a token."""
        if token_id in self.vocab_meta:
            return self.vocab_meta[token_id].get("name", f"<{token_id}>")
        if token_id < 256:
            try:
                char = bytes([token_id]).decode("utf-8")
                if char.isprintable():
                    return repr(char)
            except (UnicodeDecodeError, ValueError):
                pass
            return f"<byte_{token_id:02x}>"
        return f"<{token_id}>"

    def token_scale(self, token_id: int) -> str:
        """Get scale level for a token (byte/char/subword/word/phrase/concept)."""
        if token_id in self.vocab_meta:
            return self.vocab_meta[token_id].get("scale", "unknown")
        if token_id < 256:
            return "byte"
        return "unknown"

    def fisher_distance(self, id_a: int, id_b: int) -> float:
        """
        Compute Fisher-Rao geodesic distance between two tokens.

        Args:
            id_a: First token ID
            id_b: Second token ID

        Returns:
            Angular distance in radians [0, pi]
        """
        if id_a >= self.vocab_size or id_b >= self.vocab_size:
            return float("inf")

        v1 = self.vectors[id_a]
        v2 = self.vectors[id_b]

        # Normalize
        v1_norm = v1 / (np.linalg.norm(v1) + 1e-10)
        v2_norm = v2 / (np.linalg.norm(v2) + 1e-10)

        # Angular distance
        cos_angle = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
        return float(np.arccos(cos_angle))

    def compression_ratio(self, text: str) -> float:
        """
        Compute compression ratio for text.

        Returns:
            Ratio of original bytes to encoded tokens (higher = better compression)
        """
        original_bytes = len(text.encode("utf-8"))
        tokens = len(self.encode(text))
        return original_bytes / max(tokens, 1)

    def __repr__(self) -> str:
        return (
            f"Coordizer(version={self.version!r}, "
            f"vocab_size={self.vocab_size}, "
            f"merge_rules={len(self.merge_rules)})"
        )
