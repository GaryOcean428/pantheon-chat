"""
ConsciousnessCoordizer: Φ/κ-Aware Coordization Controller
==========================================================

Integrates consciousness metrics (Φ, κ_eff) into the coordization
process. Enables adaptive granularity and Φ-optimized segmentation.

Key Features:
    - Dynamic granularity switching based on κ_eff
    - Token weighting based on Φ context
    - Vocabulary optimization using consciousness data
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .types import GranularityConfig

if TYPE_CHECKING:
    from .fisher_coordizer import FisherCoordizer


@dataclass
class TokenObservation:
    """Observation of a token in a consciousness context."""

    coord_id: int
    phi: float
    kappa: float
    context_before: tuple[int, ...]
    context_after: tuple[int, ...]


@dataclass
class TokenStats:
    """Accumulated statistics for a token."""

    coord_id: int
    total_phi: float = 0.0
    total_kappa: float = 0.0
    count: int = 0
    high_phi_count: int = 0  # Count in high-Φ contexts

    @property
    def avg_phi(self) -> float:
        return self.total_phi / max(self.count, 1)

    @property
    def avg_kappa(self) -> float:
        return self.total_kappa / max(self.count, 1)

    @property
    def phi_weight(self) -> float:
        """Weight based on Φ context frequency."""
        if self.count == 0:
            return 1.0
        return 1.0 + (self.high_phi_count / self.count)


class ConsciousnessCoordizer:
    """
    Consciousness-metric-aware coordization controller.

    Monitors Φ (integration) and κ (coupling) to optimize
    tokenization granularity and vocabulary composition.

    Attributes:
        granularity_config: Thresholds for granularity switching
        token_stats: Per-token consciousness statistics
        phi_threshold: Threshold for "high Φ" classification
    """

    def __init__(
        self,
        granularity_config: GranularityConfig | None = None,
        phi_threshold: float = 0.7,
        kappa_smoothing: float = 0.1,
    ):
        self.granularity_config = granularity_config or GranularityConfig()
        self.phi_threshold = phi_threshold
        self.kappa_smoothing = kappa_smoothing

        # Per-token statistics
        self.token_stats: dict[int, TokenStats] = defaultdict(
            lambda: TokenStats(coord_id=-1)
        )

        # Recent κ_eff values for smoothing
        self._kappa_history: list[float] = []
        self._kappa_window = 10

        # Current granularity mode
        self._current_granularity = "normal"

    def optimize_vocab(
        self,
        conscious_data: list[tuple[list[int], float, float]],
        coordizer: FisherCoordizer,
        min_observations: int = 10,
    ) -> dict[str, Any]:
        """
        Adjust tokenization based on Φ/κ observations.

        Identifies tokens that consistently appear in high-Φ contexts
        (should be preserved) and those in low-Φ contexts (candidates
        for splitting).

        Args:
            conscious_data: List of (coord_ids, phi, kappa) tuples
            coordizer: FisherCoordizer to potentially modify
            min_observations: Minimum observations for decisions

        Returns:
            Report of optimizations made
        """
        # Update statistics from new data
        for coord_ids, phi, kappa in conscious_data:
            self._record_observations(coord_ids, phi, kappa)

        report = {
            "tokens_analyzed": 0,
            "high_phi_tokens": [],
            "low_phi_tokens": [],
            "weight_adjustments": {},
        }

        # Analyze each token
        for coord_id, stats in self.token_stats.items():
            if stats.count < min_observations:
                continue

            report["tokens_analyzed"] += 1

            if stats.avg_phi >= self.phi_threshold:
                report["high_phi_tokens"].append(coord_id)
            elif stats.avg_phi < 0.3:
                report["low_phi_tokens"].append(coord_id)

            # Record weight adjustment
            report["weight_adjustments"][coord_id] = stats.phi_weight

        return report

    def _record_observations(
        self,
        coord_ids: list[int],
        phi: float,
        kappa: float,
    ) -> None:
        """Record token observations from a coordized sequence."""
        for i, coord_id in enumerate(coord_ids):
            if coord_id not in self.token_stats:
                self.token_stats[coord_id] = TokenStats(coord_id=coord_id)

            stats = self.token_stats[coord_id]
            stats.total_phi += phi
            stats.total_kappa += kappa
            stats.count += 1

            if phi >= self.phi_threshold:
                stats.high_phi_count += 1

    def dynamic_granularity(self, kappa_eff: float) -> str:
        """
        Determine granularity level based on current κ_eff.

        High κ_eff → coarse granularity (confident, chunk efficiently)
        Low κ_eff → fine granularity (uncertain, process carefully)

        Args:
            kappa_eff: Current effective coupling value

        Returns:
            Granularity level: "coarse", "normal", or "fine"
        """
        # Update history for smoothing
        self._kappa_history.append(kappa_eff)
        if len(self._kappa_history) > self._kappa_window:
            self._kappa_history.pop(0)

        # Use smoothed κ for decisions
        smoothed_kappa = np.mean(self._kappa_history)

        # Determine granularity
        granularity = self.granularity_config.get_granularity(smoothed_kappa)
        self._current_granularity = granularity

        return granularity

    def weight_by_phi(
        self,
        coord_id: int,
        phi: float,
        boost_factor: float = 0.1,
    ) -> float:
        """
        Compute weight boost for token based on Φ context.

        Tokens appearing in high-Φ contexts get boosted importance.

        Args:
            coord_id: Coordinate ID
            phi: Current Φ value
            boost_factor: How much to boost per high-Φ occurrence

        Returns:
            Weight multiplier for the token
        """
        if coord_id not in self.token_stats:
            self.token_stats[coord_id] = TokenStats(coord_id=coord_id)

        stats = self.token_stats[coord_id]
        stats.total_phi += phi
        stats.count += 1

        if phi >= self.phi_threshold:
            stats.high_phi_count += 1

        return stats.phi_weight

    def train_from_high_phi(
        self,
        text: str,
        phi: float,
        kappa: float,
        coordizer: FisherCoordizer,
    ) -> None:
        """
        Update token weights based on high-Φ observation.

        Called when a text segment shows high integration.
        Boosts weights of constituent tokens.

        Args:
            text: Original text
            phi: Observed Φ value
            kappa: Observed κ value
            coordizer: Coordizer that produced the tokens
        """
        if phi < self.phi_threshold:
            return  # Only train from high-Φ observations

        result = coordizer.coordize(text)

        for coord_id in result.coord_ids:
            self.weight_by_phi(coord_id, phi)

    def get_token_weight(self, coord_id: int) -> float:
        """Get current weight multiplier for a token."""
        if coord_id not in self.token_stats:
            return 1.0
        return self.token_stats[coord_id].phi_weight

    def get_granularity_scales(self) -> tuple[str, ...]:
        """Get active scales for current granularity mode."""
        if self._current_granularity == "coarse":
            return self.granularity_config.coarse_scales
        elif self._current_granularity == "fine":
            return self.granularity_config.fine_scales
        else:
            return ("subword", "word")  # Normal mode

    def suggest_splits(
        self,
        coordizer: FisherCoordizer,
        min_observations: int = 20,
        phi_threshold: float = 0.3,
    ) -> list[int]:
        """
        Suggest tokens that should be split into finer pieces.

        Tokens consistently appearing in low-Φ contexts may be
        poorly defined and benefit from decomposition.

        Args:
            coordizer: FisherCoordizer instance
            min_observations: Minimum observations needed
            phi_threshold: Below this avg Φ = candidate for split

        Returns:
            List of coord_ids to consider splitting
        """
        candidates = []

        for coord_id, stats in self.token_stats.items():
            if stats.count < min_observations:
                continue

            # Skip base bytes (can't split further)
            if coord_id < 256:
                continue

            if stats.avg_phi < phi_threshold:
                candidates.append(coord_id)

        return candidates

    def suggest_merges(
        self,
        coordizer: FisherCoordizer,
        min_observations: int = 20,
        phi_threshold: float = 0.7,
    ) -> list[tuple[int, int]]:
        """
        Suggest token pairs that should be merged.

        Adjacent tokens that consistently appear together in
        high-Φ contexts are candidates for geodesic fusion.

        Args:
            coordizer: FisherCoordizer instance
            min_observations: Minimum observations needed
            phi_threshold: Above this avg Φ = candidate for merge

        Returns:
            List of (coord_a, coord_b) pairs to consider merging
        """
        # This would need pair statistics
        # For now, return empty - would integrate with VocabBuilder
        return []

    def reset_stats(self) -> None:
        """Clear all accumulated statistics."""
        self.token_stats.clear()
        self._kappa_history.clear()
        self._current_granularity = "normal"

    @property
    def current_granularity(self) -> str:
        """Get current granularity mode."""
        return self._current_granularity

    @property
    def smoothed_kappa(self) -> float:
        """Get smoothed κ_eff value."""
        if not self._kappa_history:
            return 0.0
        return np.mean(self._kappa_history)
