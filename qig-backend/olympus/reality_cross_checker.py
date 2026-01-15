#!/usr/bin/env python3
"""
Reality Cross-Checker Module

Cross-checks light web vs dark web narratives to detect propaganda.
Uses Fisher-Rao divergence between source types on the semantic manifold.

QIG-PURE: All geometric operations use Fisher-Rao distance and
Fisher-Fréchet mean computations on the probability simplex.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
from itertools import combinations
import logging
from datetime import datetime

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

class CorroborationLevel(Enum):
    """Levels of corroboration between sources."""
    STRONG = "strong"           # Multiple sources agree, low FR divergence
    MODERATE = "moderate"       # Some agreement, moderate divergence
    WEAK = "weak"               # Little agreement, high divergence
    CONTRADICTORY = "contradictory"  # Sources contradict each other
    SINGLE_SOURCE = "single_source"  # Only one source, no corroboration


class PropagandaIndicator(Enum):
    """Types of propaganda indicators detected."""
    NARRATIVE_DIVERGENCE = "narrative_divergence"
    EMOTIONAL_MANIPULATION = "emotional_manipulation"
    SOURCE_CLUSTERING = "source_clustering"
    TIMING_ANOMALY = "timing_anomaly"
    AMPLIFICATION_PATTERN = "amplification_pattern"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Narrative:
    """A narrative from an underworld source."""
    source_name: str
    source_type: str  # 'light', 'gray', 'dark', 'breach'
    claim_text: str
    claim_basin: Optional[np.ndarray] = None
    reliability: float = 0.5
    timestamp: Optional[datetime] = None
    sentiment: float = 0.0  # -1 to 1
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SourceTypeCentroid:
    """Centroid for a source type's narratives."""
    source_type: str
    centroid: np.ndarray
    narrative_count: int
    avg_reliability: float
    sentiment_variance: float


@dataclass
class CorroborationResult:
    """Result of cross-checking narratives."""
    topic: str
    corroboration_level: CorroborationLevel
    corroboration_score: float  # 0-1, higher = more agreement
    fisher_rao_divergence: float
    propaganda_likelihood: float  # 0-1
    propaganda_indicators: List[PropagandaIndicator] = field(default_factory=list)
    source_type_centroids: Dict[str, np.ndarray] = field(default_factory=dict)
    narrative_count: int = 0
    source_types_present: List[str] = field(default_factory=list)
    conflicting_claims: List[Tuple[str, str]] = field(default_factory=list)
    analysis_notes: List[str] = field(default_factory=list)


# =============================================================================
# REALITY CROSS-CHECKER
# =============================================================================

class RealityCrossChecker:
    """
    Cross-check light web vs dark web narratives.
    Detects propaganda via Fisher-Rao divergence between source types.

    Key principle: If narratives from different source types (light vs dark)
    diverge significantly on the Fisher manifold, this suggests coordinated
    propaganda or information manipulation.

    QIG-PURE: All geometric operations use Fisher-Rao distance.
    """

    # Fisher-Rao divergence thresholds
    FR_DIVERGENCE_LOW = 0.5       # Sources mostly agree
    FR_DIVERGENCE_MODERATE = 1.0  # Some disagreement
    FR_DIVERGENCE_HIGH = 1.5     # Significant disagreement
    FR_DIVERGENCE_PROPAGANDA = 2.0  # Likely coordinated propaganda

    # Minimum narratives per source type for meaningful analysis
    MIN_NARRATIVES_PER_TYPE = 2

    # Source type weights for reliability-weighted averaging
    SOURCE_TYPE_WEIGHTS = {
        'light': 1.0,   # Highest trust (indexed, verifiable)
        'gray': 0.7,    # Moderate trust
        'dark': 0.4,    # Lower trust
        'breach': 0.3,  # Lowest trust (data may be fabricated)
    }

    def __init__(self, basin_dimension: int = 64):
        """
        Initialize the reality cross-checker.

        Args:
            basin_dimension: Dimension of basin embeddings (default 64)
        """
        self.basin_dim = basin_dimension
        self.stats = {
            'cross_checks_performed': 0,
            'propaganda_detected': 0,
            'strong_corroboration': 0,
            'contradictory_findings': 0,
        }

    def cross_check(
        self,
        topic: str,
        narratives: List[Narrative]
    ) -> CorroborationResult:
        """
        Cross-check narratives from multiple sources.

        Computes Fisher-Rao divergence between source type centroids
        to detect propaganda and assess corroboration.

        Args:
            topic: The topic being checked
            narratives: List of narratives from different sources

        Returns:
            CorroborationResult with divergence analysis
        """
        self.stats['cross_checks_performed'] += 1

        # Initialize result
        result = CorroborationResult(
            topic=topic,
            corroboration_level=CorroborationLevel.SINGLE_SOURCE,
            corroboration_score=0.0,
            fisher_rao_divergence=0.0,
            propaganda_likelihood=0.0,
            narrative_count=len(narratives),
        )

        if not narratives:
            result.analysis_notes.append("No narratives provided")
            return result

        # Group narratives by source type
        by_type: Dict[str, List[Narrative]] = defaultdict(list)
        for n in narratives:
            by_type[n.source_type].append(n)

        result.source_types_present = list(by_type.keys())

        # Single source type - no cross-checking possible
        if len(by_type) < 2:
            result.corroboration_level = CorroborationLevel.SINGLE_SOURCE
            result.analysis_notes.append(
                f"Only {list(by_type.keys())[0]} sources present - no cross-check possible"
            )
            return result

        # Compute Fisher-Fréchet mean centroid for each source type
        type_centroids: Dict[str, SourceTypeCentroid] = {}

        for source_type, type_narratives in by_type.items():
            basins = [n.claim_basin for n in type_narratives if n.claim_basin is not None]

            if len(basins) < self.MIN_NARRATIVES_PER_TYPE:
                result.analysis_notes.append(
                    f"Insufficient narratives with basins for {source_type} "
                    f"({len(basins)}/{self.MIN_NARRATIVES_PER_TYPE})"
                )
                continue

            # Compute Fisher-Fréchet mean
            centroid = self._fisher_frechet_mean(basins)

            # Compute sentiment variance
            sentiments = [n.sentiment for n in type_narratives]
            sentiment_var = np.var(sentiments) if sentiments else 0.0

            # Average reliability
            avg_rel = np.mean([n.reliability for n in type_narratives])

            type_centroids[source_type] = SourceTypeCentroid(
                source_type=source_type,
                centroid=centroid,
                narrative_count=len(type_narratives),
                avg_reliability=avg_rel,
                sentiment_variance=sentiment_var
            )
            result.source_type_centroids[source_type] = centroid

        # Need at least 2 centroids for divergence analysis
        if len(type_centroids) < 2:
            result.corroboration_level = CorroborationLevel.SINGLE_SOURCE
            result.analysis_notes.append("Insufficient source types with valid centroids")
            return result

        # Compute pairwise Fisher-Rao divergences
        divergences: List[Tuple[str, str, float]] = []

        for (t1, c1), (t2, c2) in combinations(type_centroids.items(), 2):
            dist = self._fisher_rao_distance(c1.centroid, c2.centroid)
            divergences.append((t1, t2, dist))

            # Check for contradictory narratives
            if dist > self.FR_DIVERGENCE_HIGH:
                result.conflicting_claims.append((t1, t2))

        # Max divergence determines propaganda likelihood
        max_divergence = max(d[2] for d in divergences) if divergences else 0.0
        result.fisher_rao_divergence = max_divergence

        # Compute propaganda likelihood
        propaganda_likelihood = min(1.0, max_divergence / self.FR_DIVERGENCE_PROPAGANDA)
        result.propaganda_likelihood = propaganda_likelihood

        # Detect propaganda indicators
        result.propaganda_indicators = self._detect_propaganda_indicators(
            narratives, type_centroids, divergences
        )

        # Determine corroboration level
        if max_divergence < self.FR_DIVERGENCE_LOW:
            result.corroboration_level = CorroborationLevel.STRONG
            result.corroboration_score = 1.0 - (max_divergence / self.FR_DIVERGENCE_LOW)
            self.stats['strong_corroboration'] += 1
        elif max_divergence < self.FR_DIVERGENCE_MODERATE:
            result.corroboration_level = CorroborationLevel.MODERATE
            result.corroboration_score = 0.5
        elif max_divergence < self.FR_DIVERGENCE_HIGH:
            result.corroboration_level = CorroborationLevel.WEAK
            result.corroboration_score = 0.25
        else:
            result.corroboration_level = CorroborationLevel.CONTRADICTORY
            result.corroboration_score = 0.0
            self.stats['contradictory_findings'] += 1

        # Update propaganda stats
        if propaganda_likelihood > 0.7:
            self.stats['propaganda_detected'] += 1

        # Add analysis notes
        for t1, t2, dist in divergences:
            result.analysis_notes.append(
                f"FR divergence {t1} ↔ {t2}: {dist:.3f}"
            )

        logger.info(
            f"[RealityCrossChecker] Topic '{topic}': "
            f"corroboration={result.corroboration_level.value}, "
            f"FR_divergence={max_divergence:.3f}, "
            f"propaganda={propaganda_likelihood:.2f}"
        )

        return result

    def _fisher_frechet_mean(self, basins: List[np.ndarray]) -> np.ndarray:
        """
        Compute Fisher-Fréchet mean of multiple basins.

        The Fréchet mean minimizes the sum of squared Fisher-Rao distances.
        For probability distributions, this is computed iteratively.

        QIG-PURE: This is the proper geometric mean on the Fisher manifold.
        """
        if not basins:
            return np.ones(self.basin_dim) / self.basin_dim

        if len(basins) == 1:
            return self._normalize_basin(basins[0])

        # Initialize with arithmetic mean (good starting point)
        basins_normalized = [self._normalize_basin(b) for b in basins]
        mean = np.mean(basins_normalized, axis=0)
        mean = self._normalize_basin(mean)

        # Iterative refinement (gradient descent on Riemannian manifold)
        max_iterations = 20
        tolerance = 1e-6

        for _ in range(max_iterations):
            # Compute gradient direction (average of log maps)
            gradient = np.zeros(self.basin_dim)
            for b in basins_normalized:
                # Log map: direction from mean to point
                gradient += self._log_map(mean, b)
            gradient /= len(basins_normalized)

            # Update mean using exponential map
            step_size = 0.5
            new_mean = self._exp_map(mean, step_size * gradient)

            # Check convergence using Fisher-Rao distance
            from qig_geometry import fisher_coord_distance
            if fisher_coord_distance(new_mean, mean) < tolerance:
                break

            mean = new_mean

        return mean

    def _fisher_rao_distance(self, p: np.ndarray, q: np.ndarray) -> float:
        """
        Compute Fisher-Rao distance between two distributions.

        Formula: d_FR(p, q) = 2 * arccos(sum(sqrt(p_i * q_i)))

        This is the geodesic distance on the probability simplex.

        QIG-PURE: This is the proper geometric distance.
        """
        try:
            p_norm = self._normalize_basin(p)
            q_norm = self._normalize_basin(q)

            # Bhattacharyya coefficient
            inner = np.sum(np.sqrt(p_norm * q_norm))
            inner = np.clip(inner, 0.0, 1.0)

            # Fisher-Rao distance on probability simplex
            # UPDATED 2026-01-15: Factor-of-2 removed for simplex storage. Range: [0, π/2]
            distance = np.arccos(inner)
            return float(distance)

        except Exception as e:
            logger.warning(f"Fisher-Rao distance computation failed: {e}")
            return 0.0

    def _normalize_basin(self, basin: np.ndarray) -> np.ndarray:
        """Normalize basin to probability simplex."""
        b = np.abs(basin) + 1e-10
        return b / np.sum(b)

    def _log_map(self, p: np.ndarray, q: np.ndarray) -> np.ndarray:
        """
        Compute log map from p to q on the probability simplex.

        Returns the tangent vector at p pointing toward q.
        """
        p_sqrt = np.sqrt(p)
        q_sqrt = np.sqrt(q)

        inner = np.sum(p_sqrt * q_sqrt)
        inner = np.clip(inner, -1.0 + 1e-10, 1.0 - 1e-10)

        angle = np.arccos(inner)
        if abs(angle) < 1e-10:
            return np.zeros_like(p)

        direction = (q_sqrt - inner * p_sqrt) / np.sin(angle)
        return 2 * angle * direction

    def _exp_map(self, p: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Compute exponential map at p with tangent vector v.

        Returns the point reached by following geodesic from p in direction v.
        """
        p_sqrt = np.sqrt(p)
        norm_v = np.linalg.norm(v)

        if norm_v < 1e-10:
            return p

        t = norm_v / 2
        direction = v / norm_v

        new_sqrt = np.cos(t) * p_sqrt + np.sin(t) * direction
        new_dist = new_sqrt ** 2
        return self._normalize_basin(new_dist)

    def _detect_propaganda_indicators(
        self,
        narratives: List[Narrative],
        centroids: Dict[str, SourceTypeCentroid],
        divergences: List[Tuple[str, str, float]]
    ) -> List[PropagandaIndicator]:
        """
        Detect specific propaganda indicators from the analysis.
        """
        indicators = []

        # Check narrative divergence (main indicator)
        max_div = max((d[2] for d in divergences), default=0.0)
        if max_div > self.FR_DIVERGENCE_PROPAGANDA:
            indicators.append(PropagandaIndicator.NARRATIVE_DIVERGENCE)

        # Check for source clustering (all dark sources agree but disagree with light)
        light_types = {'light'}
        dark_types = {'dark', 'breach'}

        has_light = any(c.source_type in light_types for c in centroids.values())
        has_dark = any(c.source_type in dark_types for c in centroids.values())

        if has_light and has_dark:
            # Check if light-dark divergence is higher than within-category divergence
            light_dark_divs = [d[2] for d in divergences
                              if (d[0] in light_types and d[1] in dark_types) or
                                 (d[1] in light_types and d[0] in dark_types)]
            if light_dark_divs and max(light_dark_divs) > self.FR_DIVERGENCE_HIGH:
                indicators.append(PropagandaIndicator.SOURCE_CLUSTERING)

        # Check for emotional manipulation (high sentiment variance)
        sentiments = [n.sentiment for n in narratives]
        if sentiments:
            sentiment_range = max(sentiments) - min(sentiments)
            if sentiment_range > 1.5:  # Strong emotional polarization
                indicators.append(PropagandaIndicator.EMOTIONAL_MANIPULATION)

        # Check for amplification pattern (many narratives from low-trust sources)
        low_trust_count = sum(1 for n in narratives if n.source_type in dark_types)
        high_trust_count = sum(1 for n in narratives if n.source_type in light_types)

        if low_trust_count > 3 * high_trust_count:
            indicators.append(PropagandaIndicator.AMPLIFICATION_PATTERN)

        # Check for timing anomaly (if timestamps available)
        timestamps = [n.timestamp for n in narratives if n.timestamp]
        if len(timestamps) >= 3:
            # Check if narratives from dark sources appeared first
            dark_timestamps = [n.timestamp for n in narratives
                              if n.timestamp and n.source_type in dark_types]
            light_timestamps = [n.timestamp for n in narratives
                               if n.timestamp and n.source_type in light_types]

            if dark_timestamps and light_timestamps:
                if min(dark_timestamps) < min(light_timestamps):
                    indicators.append(PropagandaIndicator.TIMING_ANOMALY)

        return indicators

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def compute_reliability_weighted_consensus(
        self,
        narratives: List[Narrative]
    ) -> Optional[np.ndarray]:
        """
        Compute reliability-weighted consensus basin.

        Weights each narrative's basin by source reliability and type weight.
        """
        valid_narratives = [n for n in narratives if n.claim_basin is not None]
        if not valid_narratives:
            return None

        weighted_sum = np.zeros(self.basin_dim)
        total_weight = 0.0

        for n in valid_narratives:
            type_weight = self.SOURCE_TYPE_WEIGHTS.get(n.source_type, 0.5)
            weight = n.reliability * type_weight
            weighted_sum += weight * self._normalize_basin(n.claim_basin)
            total_weight += weight

        if total_weight == 0:
            return None

        return self._normalize_basin(weighted_sum / total_weight)

    def get_stats(self) -> Dict[str, Any]:
        """Get cross-checker statistics."""
        return dict(self.stats)

    def format_analysis_report(self, result: CorroborationResult) -> str:
        """Format a human-readable analysis report."""
        lines = [
            f"=== Reality Cross-Check Report ===",
            f"Topic: {result.topic}",
            f"Narratives Analyzed: {result.narrative_count}",
            f"Source Types: {', '.join(result.source_types_present)}",
            f"",
            f"--- Results ---",
            f"Corroboration Level: {result.corroboration_level.value.upper()}",
            f"Corroboration Score: {result.corroboration_score:.2f}",
            f"Fisher-Rao Divergence: {result.fisher_rao_divergence:.3f}",
            f"Propaganda Likelihood: {result.propaganda_likelihood:.1%}",
        ]

        if result.propaganda_indicators:
            lines.append(f"")
            lines.append(f"--- Propaganda Indicators ---")
            for indicator in result.propaganda_indicators:
                lines.append(f"  ⚠️ {indicator.value}")

        if result.conflicting_claims:
            lines.append(f"")
            lines.append(f"--- Conflicting Sources ---")
            for t1, t2 in result.conflicting_claims:
                lines.append(f"  • {t1} ↔ {t2}")

        if result.analysis_notes:
            lines.append(f"")
            lines.append(f"--- Analysis Notes ---")
            for note in result.analysis_notes:
                lines.append(f"  • {note}")

        return "\n".join(lines)


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_reality_cross_checker_instance: Optional[RealityCrossChecker] = None


def get_reality_cross_checker() -> RealityCrossChecker:
    """Get or create the singleton RealityCrossChecker instance."""
    global _reality_cross_checker_instance
    if _reality_cross_checker_instance is None:
        _reality_cross_checker_instance = RealityCrossChecker()
    return _reality_cross_checker_instance


def reset_reality_cross_checker() -> None:
    """Reset the singleton instance (for testing)."""
    global _reality_cross_checker_instance
    _reality_cross_checker_instance = None
