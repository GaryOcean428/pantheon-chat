"""
Generation module for QIG consciousness.

Provides geometric token sampling that preserves manifold structure.
"""

from src.generation.coherence_tracker import CoherenceTracker  # P0: Semantic coherence
from src.generation.qfi_sampler import (
    QFISampler,
    TraditionalSampler,
    create_sampler,
)
from src.generation.semantic_fisher_metric import (  # P5: Metric warping
    SemanticFisherMetric,
    create_semantic_metric,
)

__all__ = [
    "QFISampler",
    "TraditionalSampler",
    "create_sampler",
    "CoherenceTracker",  # P0: Semantic coherence tracking
    "SemanticFisherMetric",  # P5: Relationship-warped geodesics
    "create_semantic_metric",  # P5: Factory function
]
