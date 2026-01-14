"""
Training module - canonical source for training components.

NOTE: Consolidated training code also in qig-tokenizer repo.
This module maintains the canonical imports for type registry.
"""

# Canonical imports - type registry requires src.training.*
from src.training.geometric_vicarious import GeometricVicariousLearner, VicariousLearningResult
from src.training.identity_reinforcement import build_identity_reinforced_prompt, calibrate_verbosity

__all__ = [
    "GeometricVicariousLearner",
    "VicariousLearningResult",
    "build_identity_reinforced_prompt",
    "calibrate_verbosity",
]
