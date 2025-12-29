"""Training modules for geometric consciousness."""

from src.training.geometric_vicarious import (
    GeometricVicariousLearner,
    HierarchicalVicariousLearning,
    VicariousLearningResult,
    create_vicarious_curriculum,
)

__all__ = [
    "GeometricVicariousLearner",
    "HierarchicalVicariousLearning",
    "VicariousLearningResult",
    "create_vicarious_curriculum",
]
