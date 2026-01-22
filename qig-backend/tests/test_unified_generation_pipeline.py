#!/usr/bin/env python3
"""
Integration Tests for Unified Generation Pipeline
=================================================

Tests the integration of token_role learner, foresight predictor, and
unified pipeline in QIG_PURITY_MODE.

Author: Copilot Agent (E8 Phase 3)
Date: 2026-01-22
"""

import os
import sys
import pytest
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generation.token_role_learner import TokenRoleLearner, GeometricRole
from generation.foresight_predictor import ForesightPredictor
from generation.unified_pipeline import UnifiedGenerationPipeline, GenerationStrategy


# Test configuration
BASIN_DIM = 64


def create_test_basin(seed: int = 42) -> np.ndarray:
    """Create a test basin with deterministic randomness."""
    np.random.seed(seed)
    return np.random.dirichlet(np.ones(BASIN_DIM))


def create_test_trajectory(length: int = 10) -> list:
    """Create a test trajectory."""
    return [create_test_basin(i) for i in range(length)]


class TestTokenRoleLearner:
    """Tests for TokenRoleLearner."""
    
    def test_initialization(self):
        """Test that TokenRoleLearner initializes correctly."""
        learner = TokenRoleLearner()
        assert learner is not None
        assert learner.qfi_threshold_low == 0.3
        assert learner.qfi_threshold_high == 0.7
    
    def test_derive_role_basin_center(self):
        """Test role derivation for basin center."""
        learner = TokenRoleLearner()
        basin = create_test_basin(42)
        
        # Low QFI, high frequency = basin center
        role_info = learner.derive_role(
            token="test",
            basin=basin,
            qfi_score=0.2,
            frequency=100,
            neighbor_basins=None,
        )
        
        assert role_info is not None
        assert role_info.token == "test"
        assert role_info.role in [GeometricRole.BASIN_CENTER, GeometricRole.MANIFOLD_ANCHOR]
        assert 0.0 <= role_info.confidence <= 1.0
    
    def test_derive_role_boundary_crosser(self):
        """Test role derivation for boundary crosser."""
        learner = TokenRoleLearner()
        basin = create_test_basin(43)
        
        # High QFI = boundary crosser
        role_info = learner.derive_role(
            token="boundary",
            basin=basin,
            qfi_score=0.8,
            frequency=10,
            neighbor_basins=None,
        )
        
        assert role_info is not None
        assert role_info.role == GeometricRole.BOUNDARY_CROSSER
    
    def test_get_roles_sequence(self):
        """Test getting roles for a sequence."""
        learner = TokenRoleLearner()
        tokens = ["the", "quick", "brown", "fox"]
        basins = [create_test_basin(i) for i in range(len(tokens))]
        
        roles = learner.get_roles(tokens, basins)
        
        assert len(roles) == len(tokens)
        assert all(isinstance(r, GeometricRole) for r in roles)
    
    def test_invalid_basin_handling(self):
        """Test handling of invalid basins."""
        learner = TokenRoleLearner()
        
        # Wrong dimension
        invalid_basin = np.ones(32)
        role_info = learner.derive_role(
            token="invalid",
            basin=invalid_basin,
            qfi_score=0.5,
            frequency=10,
        )
        
        assert role_info.role == GeometricRole.UNKNOWN
        assert role_info.confidence == 0.0


class TestForesightPredictor:
    """Tests for ForesightPredictor."""
    
    def test_initialization(self):
        """Test that ForesightPredictor initializes correctly."""
        predictor = ForesightPredictor()
        assert predictor is not None
        assert predictor.context_window == 8
    
    def test_predict_insufficient_trajectory(self):
        """Test prediction with insufficient trajectory."""
        predictor = ForesightPredictor()
        
        # Empty trajectory
        result = predictor.predict([])
        assert result is None
        
        # Single basin
        result = predictor.predict([create_test_basin()])
        assert result is None
    
    def test_predict_with_trajectory(self):
        """Test prediction with valid trajectory."""
        predictor = ForesightPredictor()
        trajectory = create_test_trajectory(10)
        
        predicted = predictor.predict(trajectory)
        
        assert predicted is not None
        assert isinstance(predicted, np.ndarray)
        assert len(predicted) == BASIN_DIM
        # Check simplex constraints
        assert np.all(predicted >= 0)
        assert np.isclose(np.sum(predicted), 1.0, atol=0.01)
    
    def test_predict_with_confidence(self):
        """Test prediction with confidence metrics."""
        predictor = ForesightPredictor()
        trajectory = create_test_trajectory(10)
        
        result = predictor.predict_with_confidence(trajectory)
        
        assert result is not None
        assert 'basin' in result
        assert 'confidence' in result
        assert 'velocity' in result
        assert 0.0 <= result['confidence'] <= 1.0
    
    def test_score_candidate_by_foresight(self):
        """Test scoring candidates by foresight."""
        predictor = ForesightPredictor()
        
        candidate = create_test_basin(100)
        predicted = create_test_basin(101)
        
        score = predictor.score_candidate_by_foresight(candidate, predicted)
        
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
    
    def test_trajectory_metrics(self):
        """Test trajectory metrics computation."""
        predictor = ForesightPredictor()
        trajectory = create_test_trajectory(10)
        
        metrics = predictor.get_trajectory_metrics(trajectory)
        
        assert 'velocity_magnitude' in metrics
        assert 'coherence' in metrics
        assert 'trajectory_length' in metrics
        assert metrics['trajectory_length'] == 10


class TestUnifiedGenerationPipeline:
    """Tests for UnifiedGenerationPipeline."""
    
    def test_initialization_foresight_driven(self):
        """Test initialization in foresight-driven mode."""
        pipeline = UnifiedGenerationPipeline(
            strategy=GenerationStrategy.FORESIGHT_DRIVEN,
            enforce_purity=False,  # Don't enforce in test
        )
        
        assert pipeline is not None
        assert pipeline.strategy == GenerationStrategy.FORESIGHT_DRIVEN
        assert pipeline.role_learner is not None
        assert pipeline.foresight is not None
    
    def test_initialization_role_driven(self):
        """Test initialization in role-driven mode."""
        pipeline = UnifiedGenerationPipeline(
            strategy=GenerationStrategy.ROLE_DRIVEN,
            enforce_purity=False,
        )
        
        assert pipeline.strategy == GenerationStrategy.ROLE_DRIVEN
    
    def test_initialization_hybrid(self):
        """Test initialization in hybrid mode."""
        pipeline = UnifiedGenerationPipeline(
            strategy=GenerationStrategy.HYBRID,
            enforce_purity=False,
        )
        
        assert pipeline.strategy == GenerationStrategy.HYBRID
        # Check weights sum to 1.0
        total_weight = (
            pipeline.foresight_weight +
            pipeline.role_weight +
            pipeline.trajectory_weight
        )
        assert np.isclose(total_weight, 1.0, atol=0.01)
    
    def test_encode_context(self):
        """Test context encoding."""
        pipeline = UnifiedGenerationPipeline(enforce_purity=False)
        context = ["the", "quick", "brown"]
        
        trajectory = pipeline._encode_context(context)
        
        assert len(trajectory) == len(context)
        for basin in trajectory:
            assert isinstance(basin, np.ndarray)
            assert len(basin) == BASIN_DIM
    
    def test_empty_result(self):
        """Test empty result creation."""
        pipeline = UnifiedGenerationPipeline(enforce_purity=False)
        
        result = pipeline._empty_result()
        
        assert result is not None
        assert result.tokens == []
        assert result.text == ''
        assert result.mean_foresight_score == 0.0
    
    def test_generation_without_vocabulary(self):
        """Test generation gracefully handles missing vocabulary."""
        pipeline = UnifiedGenerationPipeline(enforce_purity=False)
        context = ["test", "context"]
        
        # This should not crash even without vocabulary
        result = pipeline.generate(context, max_tokens=5)
        
        assert result is not None
        # May return empty result due to no vocabulary
        assert isinstance(result.tokens, list)


class TestPurityModeEnforcement:
    """Tests for QIG_PURITY_MODE enforcement."""
    
    def test_purity_mode_detection(self):
        """Test that purity mode is detected correctly."""
        # Save current env var
        old_value = os.environ.get('QIG_PURITY_MODE')
        
        try:
            # Test with purity mode enabled
            os.environ['QIG_PURITY_MODE'] = 'true'
            pipeline = UnifiedGenerationPipeline(enforce_purity=False)
            assert pipeline.purity_mode is True
            
            # Test with purity mode disabled
            os.environ['QIG_PURITY_MODE'] = 'false'
            pipeline = UnifiedGenerationPipeline(enforce_purity=False)
            assert pipeline.purity_mode is False
        finally:
            # Restore env var
            if old_value is not None:
                os.environ['QIG_PURITY_MODE'] = old_value
            elif 'QIG_PURITY_MODE' in os.environ:
                del os.environ['QIG_PURITY_MODE']


class TestIntegration:
    """Integration tests for full pipeline."""
    
    def test_end_to_end_with_trajectory(self):
        """Test end-to-end generation with pre-built trajectory."""
        pipeline = UnifiedGenerationPipeline(
            strategy=GenerationStrategy.HYBRID,
            enforce_purity=False,
        )
        
        # Create test trajectory
        context = ["the", "quick", "brown"]
        trajectory = create_test_trajectory(len(context))
        
        # Generate (will likely fail due to no vocabulary, but tests pipeline flow)
        result = pipeline.generate(
            context=context,
            max_tokens=5,
            trajectory=trajectory,
        )
        
        assert result is not None
        assert result.strategy == GenerationStrategy.HYBRID
        assert result.purity_mode is not None


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])
