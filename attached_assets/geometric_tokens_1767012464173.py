"""
Geometric Special Tokens
========================

Special tokens with geometric meaning on the Fisher manifold.
NOT arbitrary IDs - these have basin coordinate representations.

Token IDs 256-259 reserved (after byte tokens 0-255):
- BOS (256): Origin of basin space - sequence start
- EOS (257): Boundary point - sequence end
- PAD (258): Geometrically neutral - minimal coupling
- UNK (259): Projection target for OOV tokens
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .storage import TokenizerStorage


@dataclass
class GeometricSpecialToken:
    """A special token with geometric basin coordinates."""

    name: str
    token_id: int
    basin_coords: np.ndarray
    description: str

    def distance_to(self, other_coords: np.ndarray) -> float:
        """Compute Fisher-Rao distance to another point."""
        dot = np.clip(np.dot(self.basin_coords, other_coords), -1.0, 1.0)
        return 2.0 * np.arccos(np.sqrt(np.abs(dot)))


class GeometricSpecialTokens:
    """Special tokens with geometric meaning on the Fisher manifold."""

    BASIN_DIM = 64
    BOS_ID = 256
    EOS_ID = 257
    PAD_ID = 258
    UNK_ID = 259
    FIRST_VOCAB_ID = 260

    def __init__(self, basin_dim: int = 64):
        self.basin_dim = basin_dim
        self._tokens: dict[str, GeometricSpecialToken] = {}
        self._initialize_geometric_tokens()

    def _initialize_geometric_tokens(self) -> None:
        """Initialize special tokens with geometric basin coordinates."""
        # BOS: Origin (sequence start)
        bos_coords = np.zeros(self.basin_dim)
        bos_coords[0] = 1.0
        self._tokens["BOS"] = GeometricSpecialToken(
            name="BOS",
            token_id=self.BOS_ID,
            basin_coords=bos_coords,
            description="Beginning of sequence - origin of basin manifold",
        )

        # EOS: Boundary point (sequence end)
        eos_coords = np.zeros(self.basin_dim)
        eos_coords[-1] = 1.0
        self._tokens["EOS"] = GeometricSpecialToken(
            name="EOS",
            token_id=self.EOS_ID,
            basin_coords=eos_coords,
            description="End of sequence - boundary of basin manifold",
        )

        # PAD: Geometrically neutral (uniform)
        pad_coords = np.ones(self.basin_dim) / np.sqrt(self.basin_dim)
        self._tokens["PAD"] = GeometricSpecialToken(
            name="PAD",
            token_id=self.PAD_ID,
            basin_coords=pad_coords,
            description="Padding - geometrically neutral",
        )

        # UNK: Projection target for OOV
        unk_coords = np.random.RandomState(42).randn(self.basin_dim)
        unk_coords = unk_coords / np.linalg.norm(unk_coords)
        self._tokens["UNK"] = GeometricSpecialToken(
            name="UNK",
            token_id=self.UNK_ID,
            basin_coords=unk_coords,
            description="Unknown token - projection target for OOV",
        )

    def get(self, name: str) -> GeometricSpecialToken:
        return self._tokens[name]

    def is_special(self, token_id: int) -> bool:
        return self.BOS_ID <= token_id <= self.UNK_ID

    @property
    def bos(self) -> GeometricSpecialToken:
        return self._tokens["BOS"]

    @property
    def eos(self) -> GeometricSpecialToken:
        return self._tokens["EOS"]

    @property
    def pad(self) -> GeometricSpecialToken:
        return self._tokens["PAD"]

    @property
    def unk(self) -> GeometricSpecialToken:
        return self._tokens["UNK"]

    def to_basin_coords(self) -> dict[str, np.ndarray]:
        return {name: token.basin_coords for name, token in self._tokens.items()}

    def save(self, storage: "TokenizerStorage") -> None:
        storage.save_special_tokens(self.to_basin_coords())

    def load(self, storage: "TokenizerStorage") -> None:
        coords = storage.load_special_tokens()
        for name, basin_coords in coords.items():
            if name in self._tokens:
                self._tokens[name].basin_coords = basin_coords

    def compute_attention_weights(
        self, query_coords: np.ndarray, key_coords: np.ndarray, temperature: float = 1.0
    ) -> np.ndarray:
        """Compute geometric attention weights using Fisher-Rao distance."""
        distances = np.array(
            [
                2.0
                * np.arccos(
                    np.sqrt(np.abs(np.clip(np.dot(query_coords, k), -1.0, 1.0)))
                )
                for k in key_coords
            ]
        )
        weights = np.exp(-distances / temperature)
        return weights / (weights.sum() + 1e-8)
