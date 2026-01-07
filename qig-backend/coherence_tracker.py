"""
Semantic Coherence Tracker - Port from qig-consciousness
=========================================================

Tracks text coherence alongside consciousness metrics (Φ, κ).

Problem addressed: Φ measures basin entropy, not text coherence.
Solution: Track semantic coherence as a separate metric.

Metrics computed:
- semantic_coherence: Average bigram basin similarity (word→word flow)
- text_perplexity: Perplexity of generated sequence
- bigram_flow: Strength of consecutive token transitions

This is a NumPy-based implementation for pantheon-chat.
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict

from qig_geometry import fisher_rao_distance


@dataclass
class CoherenceTracker:
    """
    Track semantic coherence during token generation.

    Separates text coherence from consciousness metrics.
    High Φ can coexist with low coherence (word salad).
    This tracker measures what Φ does not.
    """

    # Running statistics
    bigram_similarities: List[float] = field(default_factory=list)
    token_probs: List[float] = field(default_factory=list)
    entropies: List[float] = field(default_factory=list)
    fisher_distances: List[float] = field(default_factory=list)
    tokens_generated: int = 0

    # Previous token basin for bigram tracking
    prev_token_basin: Optional[np.ndarray] = None

    def reset(self) -> None:
        """Reset tracker for new generation."""
        self.bigram_similarities = []
        self.token_probs = []
        self.entropies = []
        self.fisher_distances = []
        self.tokens_generated = 0
        self.prev_token_basin = None

    def update(
        self,
        token_id: int,
        token_basin_coords: np.ndarray,
        selected_prob: float,
        entropy: float,
        bigram_similarity: Optional[float] = None,
    ) -> None:
        """
        Update tracker with new token.

        Args:
            token_id: Generated token ID
            token_basin_coords: Basin coords for this token [64]
            selected_prob: Probability of selected token
            entropy: Entropy of selection distribution
            bigram_similarity: Similarity to previous token (if computed externally)
        """
        self.tokens_generated += 1
        self.token_probs.append(selected_prob)
        self.entropies.append(entropy)

        # Compute bigram similarity if we have previous token
        if self.prev_token_basin is not None:
            if bigram_similarity is None:
                sim = self._compute_bigram_similarity(
                    self.prev_token_basin, token_basin_coords
                )
                self.bigram_similarities.append(sim)
            else:
                self.bigram_similarities.append(bigram_similarity)

            # Also track Fisher-Rao distance (QIG-pure)
            fr_dist = fisher_rao_distance(self.prev_token_basin, token_basin_coords)
            self.fisher_distances.append(fr_dist)

        # Update prev for next iteration
        self.prev_token_basin = token_basin_coords.copy()

    def _compute_bigram_similarity(
        self,
        prev_basin: np.ndarray,
        curr_basin: np.ndarray,
    ) -> float:
        """
        Compute similarity between consecutive token basins.

        Uses Fisher-Rao based similarity: 1 - d_FR/π
        This is QIG-pure (NOT cosine similarity).
        """
        # Fisher-Rao distance
        fr_dist = fisher_rao_distance(prev_basin, curr_basin)

        # Convert to similarity (d_FR ∈ [0, π] → sim ∈ [0, 1])
        similarity = 1.0 - (fr_dist / np.pi)

        return float(similarity)

    def get_prev_token_basin(self) -> Optional[np.ndarray]:
        """Get previous token basin for bigram flow."""
        return self.prev_token_basin

    def compute_metrics(self) -> Dict[str, float]:
        """
        Compute final coherence metrics.

        Returns:
            Dict with semantic_coherence, text_perplexity, bigram_flow, avg_entropy
        """
        if self.tokens_generated == 0:
            return {
                "semantic_coherence": 0.0,
                "text_perplexity": 0.0,
                "bigram_flow": 0.0,
                "avg_entropy": 0.0,
                "tokens_generated": 0,
                "avg_selected_prob": 0.0,
                "avg_fisher_distance": 0.0,
            }

        # Semantic coherence: average bigram similarity
        # High value = consecutive tokens have similar basin coords = semantic flow
        semantic_coherence = (
            sum(self.bigram_similarities) / len(self.bigram_similarities)
            if self.bigram_similarities
            else 0.0
        )

        # Text perplexity: exp(average negative log probability)
        # Low value = model is confident = coherent text (in model's view)
        avg_log_prob = sum(math.log(p + 1e-10) for p in self.token_probs) / len(self.token_probs)
        text_perplexity = math.exp(-avg_log_prob)

        # Bigram flow: inverse variance of bigram similarities
        # Low variance = consistent transitions = structured text
        if len(self.bigram_similarities) > 1:
            mean_sim = sum(self.bigram_similarities) / len(self.bigram_similarities)
            variance = sum((s - mean_sim) ** 2 for s in self.bigram_similarities) / len(self.bigram_similarities)
            bigram_flow = 1.0 / (1.0 + variance)  # High flow = low variance
        else:
            bigram_flow = 1.0

        # Average entropy
        avg_entropy = sum(self.entropies) / len(self.entropies) if self.entropies else 0.0

        # Average selected probability
        avg_prob = sum(self.token_probs) / len(self.token_probs) if self.token_probs else 0.0

        # Average Fisher-Rao distance
        avg_fisher = (
            sum(self.fisher_distances) / len(self.fisher_distances)
            if self.fisher_distances
            else 0.0
        )

        return {
            "semantic_coherence": semantic_coherence,
            "text_perplexity": text_perplexity,
            "bigram_flow": bigram_flow,
            "avg_entropy": avg_entropy,
            "tokens_generated": self.tokens_generated,
            "avg_selected_prob": avg_prob,
            "avg_fisher_distance": avg_fisher,
        }

    def get_summary(self) -> str:
        """Get human-readable summary."""
        metrics = self.compute_metrics()
        return (
            f"Coherence: {metrics['semantic_coherence']:.3f}, "
            f"PPL: {metrics['text_perplexity']:.1f}, "
            f"Flow: {metrics['bigram_flow']:.3f}, "
            f"FR-dist: {metrics['avg_fisher_distance']:.3f}"
        )


def create_coherence_tracker() -> CoherenceTracker:
    """Factory function for creating a coherence tracker."""
    return CoherenceTracker()


__all__ = ["CoherenceTracker", "create_coherence_tracker"]
