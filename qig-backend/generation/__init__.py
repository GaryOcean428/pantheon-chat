"""
Generation Module - E8 Phase 3 Unified Pipeline
================================================

This module implements the unified generation pipeline as specified in E8 Protocol v4.0 Phase 3.

Components:
- token_role_learner: Derives geometric roles from Fisher-Rao neighborhoods
- foresight_predictor: Predicts next basin via trajectory regression
- unified_pipeline: Integrates all components for QIG-pure generation

All operations use Fisher-Rao geometry (NO cosine similarity, NO external NLP).
"""

from .token_role_learner import TokenRoleLearner, GeometricRole
from .foresight_predictor import ForesightPredictor
from .unified_pipeline import UnifiedGenerationPipeline

__all__ = [
    'TokenRoleLearner',
    'GeometricRole',
    'ForesightPredictor',
    'UnifiedGenerationPipeline',
]
