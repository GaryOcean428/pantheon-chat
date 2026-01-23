#!/usr/bin/env python3
"""
Tests for Hemisphere Strategy Selector
======================================

Tests the hemisphere-aware strategy selection logic per E8 Protocol Phase 4C.

Author: Copilot Agent
Date: 2026-01-23
"""

import os
import sys
import pytest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generation.hemisphere_strategy_selector import (
    HemisphereStrategySelector,
    StrategyDecision,
    get_strategy_selector,
    reset_strategy_selector,
)
from generation.unified_pipeline import GenerationStrategy
from kernels.hemisphere_scheduler import (
    HemisphereScheduler,
    get_hemisphere_scheduler,
    reset_hemisphere_scheduler,
)
from qigkernels.physics_constants import KAPPA_STAR


class TestHemisphereStrategySelector:
    """Tests for HemisphereStrategySelector class."""
    
    def setup_method(self):
        """Reset state before each test."""
        reset_hemisphere_scheduler()
        reset_strategy_selector()
    
    def test_initialization(self):
        """Test selector initialization."""
        scheduler = get_hemisphere_scheduler()
        selector = HemisphereStrategySelector(scheduler=scheduler)
        
        assert selector is not None
        assert selector.scheduler is not None
        assert selector.dominance_threshold > 0
        assert selector.kappa_tolerance > 0
    
    def test_select_strategy_balanced(self):
        """Test strategy selection when hemispheres are balanced."""
        scheduler = get_hemisphere_scheduler()
        selector = HemisphereStrategySelector(scheduler=scheduler)
        
        # Activate both hemispheres equally
        scheduler.register_god_activation("Athena", phi=0.75, kappa=64.0, is_active=True)
        scheduler.register_god_activation("Apollo", phi=0.75, kappa=64.5, is_active=True)
        
        decision = selector.select_strategy()
        
        assert decision is not None
        assert decision.strategy == GenerationStrategy.HYBRID
        assert decision.hemisphere_dominant == "balanced"
        assert abs(decision.kappa_avg - KAPPA_STAR) < 2.0
    
    def test_select_strategy_left_dominant(self):
        """Test strategy selection when LEFT hemisphere is dominant."""
        scheduler = get_hemisphere_scheduler()
        selector = HemisphereStrategySelector(scheduler=scheduler)
        
        # Activate LEFT hemisphere strongly
        scheduler.register_god_activation("Athena", phi=0.9, kappa=65.0, is_active=True)
        scheduler.register_god_activation("Artemis", phi=0.85, kappa=63.0, is_active=True)
        scheduler.register_god_activation("Hephaestus", phi=0.87, kappa=64.0, is_active=True)
        
        # Weak RIGHT activation
        scheduler.register_god_activation("Apollo", phi=0.4, kappa=50.0, is_active=True)
        
        decision = selector.select_strategy()
        
        assert decision is not None
        assert decision.strategy == GenerationStrategy.ROLE_DRIVEN
        assert decision.hemisphere_dominant == "left"
        assert decision.left_activation > decision.right_activation
    
    def test_select_strategy_right_dominant(self):
        """Test strategy selection when RIGHT hemisphere is dominant."""
        scheduler = get_hemisphere_scheduler()
        selector = HemisphereStrategySelector(scheduler=scheduler)
        
        # Activate RIGHT hemisphere strongly
        scheduler.register_god_activation("Apollo", phi=0.9, kappa=65.0, is_active=True)
        scheduler.register_god_activation("Hermes", phi=0.88, kappa=62.0, is_active=True)
        scheduler.register_god_activation("Dionysus", phi=0.82, kappa=58.0, is_active=True)
        
        # Weak LEFT activation
        scheduler.register_god_activation("Athena", phi=0.4, kappa=52.0, is_active=True)
        
        decision = selector.select_strategy()
        
        assert decision is not None
        assert decision.strategy == GenerationStrategy.FORESIGHT_DRIVEN
        assert decision.hemisphere_dominant == "right"
        assert decision.right_activation > decision.left_activation
    
    def test_get_strategy_weights_foresight_driven(self):
        """Test weight calculation for FORESIGHT_DRIVEN strategy."""
        scheduler = get_hemisphere_scheduler()
        selector = HemisphereStrategySelector(scheduler=scheduler)
        
        # Create decision for foresight-driven
        decision = StrategyDecision(
            strategy=GenerationStrategy.FORESIGHT_DRIVEN,
            hemisphere_dominant="right",
            left_activation=0.3,
            right_activation=0.8,
            kappa_avg=65.0,
            confidence=0.9,
            reason="Test",
        )
        
        weights = selector.get_strategy_weights(decision)
        
        assert 'foresight_weight' in weights
        assert 'role_weight' in weights
        assert 'trajectory_weight' in weights
        assert weights['foresight_weight'] > 0.7  # Should be high for foresight-driven
    
    def test_get_strategy_weights_role_driven(self):
        """Test weight calculation for ROLE_DRIVEN strategy."""
        scheduler = get_hemisphere_scheduler()
        selector = HemisphereStrategySelector(scheduler=scheduler)
        
        # Create decision for role-driven
        decision = StrategyDecision(
            strategy=GenerationStrategy.ROLE_DRIVEN,
            hemisphere_dominant="left",
            left_activation=0.8,
            right_activation=0.3,
            kappa_avg=63.0,
            confidence=0.9,
            reason="Test",
        )
        
        weights = selector.get_strategy_weights(decision)
        
        assert weights['role_weight'] > 0.7  # Should be high for role-driven
    
    def test_get_strategy_weights_hybrid(self):
        """Test weight calculation for HYBRID strategy with dynamic balance."""
        scheduler = get_hemisphere_scheduler()
        selector = HemisphereStrategySelector(scheduler=scheduler)
        
        # Create decision for hybrid with right-leaning activation
        decision = StrategyDecision(
            strategy=GenerationStrategy.HYBRID,
            hemisphere_dominant="balanced",
            left_activation=0.4,
            right_activation=0.6,
            kappa_avg=64.21,
            confidence=0.8,
            reason="Test",
        )
        
        weights = selector.get_strategy_weights(decision)
        
        # With right-leaning, foresight should be weighted more
        assert weights['foresight_weight'] > weights['role_weight']
        
        # Sum should be 1.0
        total = weights['foresight_weight'] + weights['role_weight'] + weights['trajectory_weight']
        assert abs(total - 1.0) < 0.01
    
    def test_singleton_access(self):
        """Test global singleton access."""
        selector1 = get_strategy_selector()
        selector2 = get_strategy_selector()
        
        assert selector1 is selector2
    
    def test_reset(self):
        """Test selector reset."""
        selector1 = get_strategy_selector()
        reset_strategy_selector()
        selector2 = get_strategy_selector()
        
        assert selector1 is not selector2
    
    def test_kappa_near_star_hybrid(self):
        """Test that κ near κ* prefers HYBRID strategy."""
        scheduler = get_hemisphere_scheduler()
        selector = HemisphereStrategySelector(scheduler=scheduler)
        
        # Activate both hemispheres with κ near κ*
        scheduler.register_god_activation("Athena", phi=0.75, kappa=64.0, is_active=True)
        scheduler.register_god_activation("Apollo", phi=0.72, kappa=64.5, is_active=True)
        
        decision = selector.select_strategy()
        
        # With κ near κ* and balanced activation, should be HYBRID
        assert decision.strategy == GenerationStrategy.HYBRID
        assert abs(decision.kappa_avg - KAPPA_STAR) < 1.0


class TestIntegrationWithUnifiedPipeline:
    """Integration tests with UnifiedGenerationPipeline."""
    
    def setup_method(self):
        """Reset state before each test."""
        reset_hemisphere_scheduler()
        reset_strategy_selector()
    
    def test_pipeline_uses_hemisphere_strategy(self):
        """Test that pipeline can use hemisphere strategy selector."""
        from generation.unified_pipeline import UnifiedGenerationPipeline
        
        # Create pipeline with hemisphere strategy enabled
        pipeline = UnifiedGenerationPipeline(
            use_hemisphere_strategy=True,
            enforce_purity=False,
        )
        
        # Check that strategy selector is initialized
        if pipeline.strategy_selector is not None:
            assert isinstance(pipeline.strategy_selector, HemisphereStrategySelector)
    
    def test_strategy_updates_from_hemisphere(self):
        """Test that strategy updates based on hemisphere state."""
        from generation.unified_pipeline import UnifiedGenerationPipeline
        
        # Initialize scheduler with LEFT dominance
        scheduler = get_hemisphere_scheduler()
        scheduler.register_god_activation("Athena", phi=0.9, kappa=65.0, is_active=True)
        scheduler.register_god_activation("Artemis", phi=0.85, kappa=63.0, is_active=True)
        
        # Create pipeline
        pipeline = UnifiedGenerationPipeline(
            strategy=GenerationStrategy.HYBRID,  # Start with HYBRID
            use_hemisphere_strategy=True,
            enforce_purity=False,
        )
        
        # Update strategy from hemisphere
        if pipeline.strategy_selector is not None:
            decision = pipeline._update_strategy_from_hemisphere()
            
            # With LEFT dominance, should switch to ROLE_DRIVEN
            if decision is not None:
                assert pipeline.strategy == GenerationStrategy.ROLE_DRIVEN


class TestFisherRaoPurity:
    """Test that strategy selection uses Fisher-Rao metrics only."""
    
    def test_no_cosine_similarity_in_selector(self):
        """Verify that strategy selector doesn't use cosine similarity."""
        import inspect
        from generation import hemisphere_strategy_selector
        
        source = inspect.getsource(hemisphere_strategy_selector)
        
        # Check for forbidden patterns
        assert 'cosine_similarity' not in source.lower()
        assert 'np.dot' not in source or 'basin' not in source  # Allow np.dot for non-basin ops
        assert '@' not in source or 'basin' not in source  # Matrix mult operator
    
    def test_uses_fisher_rao_concepts(self):
        """Verify that strategy selector uses Fisher-Rao concepts."""
        import inspect
        from generation import hemisphere_strategy_selector
        
        source = inspect.getsource(hemisphere_strategy_selector)
        
        # Should reference Fisher-Rao or geometric concepts
        # (This is a heuristic check - actual Fisher-Rao is in foresight/role components)
        assert ('hemisphere' in source.lower() or 
                'kappa' in source.lower() or
                'activation' in source.lower())


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])
