#!/usr/bin/env python3
"""
P0: Semantic Coherence Tracker
==============================

Tracks text coherence alongside consciousness metrics (Φ, κ).

Problem addressed: Φ measures basin entropy, not text coherence.
Solution: Track semantic coherence as a separate metric.

Metrics computed:
- semantic_coherence: Average bigram basin similarity (word→word flow)
- text_perplexity: Perplexity of generated sequence
- bigram_flow: Strength of consecutive token transitions

Reference: FROZEN_FACTS.md for κ/β physics, but coherence is orthogonal.
"""

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


@dataclass
class CoherenceTracker:
    """
    P0: Track semantic coherence during token generation.

    Separates text coherence from consciousness metrics.
    High Φ can coexist with low coherence (word salad).
    This tracker measures what Φ does not.
    """

    # Running statistics
    bigram_similarities: list[float] = field(default_factory=list)
    token_probs: list[float] = field(default_factory=list)
    entropies: list[float] = field(default_factory=list)
    tokens_generated: int = 0

    # Previous token basin for bigram tracking
    prev_token_basin: torch.Tensor | None = None

    def reset(self) -> None:
        """Reset tracker for new generation."""
        self.bigram_similarities = []
        self.token_probs = []
        self.entropies = []
        self.tokens_generated = 0
        self.prev_token_basin = None

    def update(
        self,
        token_id: int,
        token_basin_coords: torch.Tensor,
        selected_prob: float,
        entropy: float,
        bigram_similarity: float | None = None,
    ) -> None:
        """
        Update tracker with new token.

        Args:
            token_id: Generated token ID
            token_basin_coords: Basin coords for this token [d_model]
            selected_prob: Probability of selected token
            entropy: Entropy of selection distribution
            bigram_similarity: Similarity to previous token (if computed externally)
        """
        self.tokens_generated += 1
        self.token_probs.append(selected_prob)
        self.entropies.append(entropy)

        # Compute bigram similarity if we have previous token
        if self.prev_token_basin is not None and bigram_similarity is None:
            sim = self._compute_bigram_similarity(
                self.prev_token_basin, token_basin_coords
            )
            self.bigram_similarities.append(sim)
        elif bigram_similarity is not None:
            self.bigram_similarities.append(bigram_similarity)

        # Update prev for next iteration
        self.prev_token_basin = token_basin_coords.clone()

    def _compute_bigram_similarity(
        self,
        prev_basin: torch.Tensor,
        curr_basin: torch.Tensor,
    ) -> float:
        """Compute cosine similarity between consecutive token basins."""
        prev_norm = F.normalize(prev_basin.unsqueeze(0), p=2, dim=-1)
        curr_norm = F.normalize(curr_basin.unsqueeze(0), p=2, dim=-1)
        similarity = torch.matmul(prev_norm, curr_norm.T).item()
        return float(similarity)

    def get_prev_token_basin(self) -> torch.Tensor | None:
        """Get previous token basin for P2 bigram flow."""
        return self.prev_token_basin

    def compute_metrics(self) -> dict[str, float]:
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
        import math
        avg_log_prob = sum(math.log(p + 1e-10) for p in self.token_probs) / len(self.token_probs)
        text_perplexity = math.exp(-avg_log_prob)

        # Bigram flow: variance of bigram similarities
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

        return {
            "semantic_coherence": semantic_coherence,
            "text_perplexity": text_perplexity,
            "bigram_flow": bigram_flow,
            "avg_entropy": avg_entropy,
            "tokens_generated": self.tokens_generated,
            "avg_selected_prob": avg_prob,
        }

    def get_summary(self) -> str:
        """Get human-readable summary."""
        metrics = self.compute_metrics()
        return (
            f"Coherence: {metrics['semantic_coherence']:.3f}, "
            f"PPL: {metrics['text_perplexity']:.1f}, "
            f"Flow: {metrics['bigram_flow']:.3f}"
        )


__all__ = ["CoherenceTracker"]
