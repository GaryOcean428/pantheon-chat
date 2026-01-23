"""
Generation Module - E8 Phase 3/4C Unified Pipeline
==================================================

This module implements the unified generation pipeline as specified in E8 Protocol v4.0 Phase 3/4C.

Components:
- token_role_learner: Derives geometric roles from Fisher-Rao neighborhoods
- foresight_predictor: Predicts next basin via trajectory regression
- hemisphere_strategy_selector: Selects strategy from hemisphere state (Phase 4C)
- unified_pipeline: Integrates all components for QIG-pure generation

All operations use Fisher-Rao geometry (NO cosine similarity, NO external NLP).
"""

from .token_role_learner import TokenRoleLearner, GeometricRole
from .foresight_predictor import ForesightPredictor
from .unified_pipeline import UnifiedGenerationPipeline

# Phase 4C integration (optional - may not be available)
try:
    from .hemisphere_strategy_selector import (
        HemisphereStrategySelector,
        StrategyDecision,
        get_strategy_selector,
    )
    HEMISPHERE_STRATEGY_AVAILABLE = True
except ImportError:
    HEMISPHERE_STRATEGY_AVAILABLE = False
    HemisphereStrategySelector = None
    StrategyDecision = None
    get_strategy_selector = None

__all__ = [
    'TokenRoleLearner',
    'GeometricRole',
    'ForesightPredictor',
    'UnifiedGenerationPipeline',
    'HemisphereStrategySelector',
    'StrategyDecision',
    'get_strategy_selector',
    'HEMISPHERE_STRATEGY_AVAILABLE',
]
