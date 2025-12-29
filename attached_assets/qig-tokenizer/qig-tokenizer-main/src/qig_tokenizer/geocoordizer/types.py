"""
GeoCoordizer Type Definitions
=============================

Core types for geometric coordization on Fisher manifold.
All types maintain geometric purity - no Euclidean operations.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Import from canonical constants (aligned with Pantheon-Chat)
from qig_tokenizer.constants import (
    BASIN_DIM,
)


@dataclass
class BasinCoordinate:
    """
    A point on the 64-dimensional Fisher information manifold.

    Represents a single "token" in geometric space. All similarity
    computations use Fisher-Rao distance, never Euclidean.
    """

    coord_id: int
    vector: np.ndarray  # Shape: (BASIN_DIM,)
    name: str | None = None
    scale: str = "subword"  # char, subword, word, phrase, concept

    def __post_init__(self):
        if self.vector.shape != (BASIN_DIM,):
            raise ValueError(
                f"Basin coordinate must be {BASIN_DIM}D, got {self.vector.shape}"
            )

    def fisher_distance(self, other: BasinCoordinate) -> float:
        """
        Compute Fisher-Rao geodesic distance to another coordinate.

        Uses the Fisher information metric, NOT Euclidean distance.
        For practical computation, we approximate using the induced metric.
        """
        # Fisher-Rao distance approximation via angular distance
        # (More accurate than Euclidean, respects manifold geometry)
        norm_self = np.linalg.norm(self.vector)
        norm_other = np.linalg.norm(other.vector)

        if norm_self < 1e-10 or norm_other < 1e-10:
            return float("inf")

        cos_angle = np.clip(
            np.dot(self.vector, other.vector) / (norm_self * norm_other), -1.0, 1.0
        )
        return np.arccos(cos_angle)

    def geodesic_midpoint(self, other: BasinCoordinate) -> np.ndarray:
        """
        Compute geodesic midpoint between this and another coordinate.

        Used for initializing new coordinates from existing ones.
        NOT arithmetic mean - uses manifold geometry.
        """
        # Geodesic midpoint on sphere (approximation for Fisher manifold)
        v1 = self.vector / (np.linalg.norm(self.vector) + 1e-10)
        v2 = other.vector / (np.linalg.norm(other.vector) + 1e-10)

        midpoint = v1 + v2
        midpoint = midpoint / (np.linalg.norm(midpoint) + 1e-10)

        # Scale to average magnitude
        avg_mag = (np.linalg.norm(self.vector) + np.linalg.norm(other.vector)) / 2
        return midpoint * avg_mag


@dataclass
class TokenCandidate:
    """
    Candidate for vocabulary expansion.

    Tracks potential new coordinates based on frequency,
    coupling strength, and efficiency gain.
    """

    sequence: tuple[int, ...]  # Existing coord IDs that would merge
    frequency: int
    coupling_strength: float  # κ between components
    phi_gain: float  # Expected Φ improvement
    efficiency_gain: float  # Tokens saved per occurrence

    @property
    def merge_score(self) -> float:
        """Combined score for merge priority."""
        return (
            self.frequency
            * self.coupling_strength
            * (1.0 + self.phi_gain)
            * self.efficiency_gain
        )


@dataclass
class CoordizationResult:
    """
    Result of coordizing text.

    Contains coordinate sequence plus metadata for
    consciousness metrics and debugging.
    """

    coordinates: list[BasinCoordinate]
    coord_ids: list[int]
    original_text: str
    granularity: str = "auto"  # char, subword, word, phrase, auto

    # Metrics (populated after processing)
    phi: float | None = None
    kappa_eff: float | None = None
    basin_velocity: float | None = None  # Avg geodesic distance between consecutive

    # Multi-scale info (optional)
    scale_hierarchy: dict[str, list[int]] = field(default_factory=dict)

    def compute_basin_velocity(self) -> float:
        """Average geodesic movement between consecutive coordinates."""
        if len(self.coordinates) < 2:
            return 0.0

        total_dist = 0.0
        for i in range(len(self.coordinates) - 1):
            total_dist += self.coordinates[i].fisher_distance(self.coordinates[i + 1])

        self.basin_velocity = total_dist / (len(self.coordinates) - 1)
        return self.basin_velocity


@dataclass
class VocabStats:
    """Statistics for vocabulary health monitoring."""

    total_coordinates: int
    scale_distribution: dict[str, int]  # Scale -> count
    avg_coupling: float
    coverage_rate: float  # % of text covered by word+ level
    oov_rate: float  # % falling back to char/byte level


@dataclass
class GranularityConfig:
    """Configuration for adaptive granularity switching."""

    kappa_high_threshold: float = 50.0  # Above this: coarse
    kappa_low_threshold: float = 30.0  # Below this: fine

    coarse_scales: tuple[str, ...] = ("phrase", "word")
    fine_scales: tuple[str, ...] = ("subword", "char")

    def get_granularity(self, kappa_eff: float) -> str:
        """Determine granularity level from κ_eff."""
        if kappa_eff >= self.kappa_high_threshold:
            return "coarse"
        elif kappa_eff <= self.kappa_low_threshold:
            return "fine"
        else:
            return "normal"
