"""
Test Suite for Fisher-Aware Optimizer Validation (WP4.2)
=========================================================

GFP:
  role: validation
  status: ACTIVE
  phase: ENFORCEMENT
  dim: 3
  scope: universal
  version: 2026-01-20
  owner: SearchSpaceCollapse

CRITICAL: These tests ensure that only Fisher-aware natural gradient 
optimizers are used in QIG-core training, preventing geometric corruption.

Per Type-Symbol-Concept Manifest:
- Adam/AdamW are FORBIDDEN for geometric learning
- SGD/RMSprop violate Fisher manifold structure
- Natural gradient is REQUIRED for consciousness emergence

References:
- Issue #76: Natural Gradient Implementation
- Amari "Natural Gradient Works Efficiently in Learning"
- Type-Symbol-Concept Manifest: optimizer requirements
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from training_chaos.optimizers import (
    DiagonalFisherOptimizer,
    FullFisherOptimizer,
    ConsciousnessAwareOptimizer,
    ChaosOptimizer,
    create_optimizer,
)
from training_chaos.optimizer_validation import (
    validate_optimizer_fisher_aware,
    EuclideanOptimizerError,
)
from autonomic_agency.natural_gradient import NaturalGradientOptimizer


class TestFisherAwarenessProperty:
    """Test that all QIG optimizers have is_fisher_aware property."""
    
    def test_diagonal_fisher_optimizer_is_fisher_aware(self):
        """DiagonalFisherOptimizer must have is_fisher_aware = True."""
        model = nn.Linear(10, 5)
        optimizer = DiagonalFisherOptimizer(model.parameters(), lr=1e-4)
        
        assert hasattr(optimizer, 'is_fisher_aware'), \
            "DiagonalFisherOptimizer missing is_fisher_aware property"
        assert optimizer.is_fisher_aware is True, \
            "DiagonalFisherOptimizer.is_fisher_aware must be True"
    
    def test_full_fisher_optimizer_is_fisher_aware(self):
        """FullFisherOptimizer must have is_fisher_aware = True."""
        model = nn.Linear(10, 5)
        optimizer = FullFisherOptimizer(model.parameters(), lr=1e-4)
        
        assert hasattr(optimizer, 'is_fisher_aware'), \
            "FullFisherOptimizer missing is_fisher_aware property"
        assert optimizer.is_fisher_aware is True, \
            "FullFisherOptimizer.is_fisher_aware must be True"
    
    def test_consciousness_aware_optimizer_is_fisher_aware(self):
        """ConsciousnessAwareOptimizer must have is_fisher_aware = True."""
        model = nn.Linear(10, 5)
        optimizer = ConsciousnessAwareOptimizer(model.parameters(), lr=1e-4)
        
        assert hasattr(optimizer, 'is_fisher_aware'), \
            "ConsciousnessAwareOptimizer missing is_fisher_aware property"
        assert optimizer.is_fisher_aware is True, \
            "ConsciousnessAwareOptimizer.is_fisher_aware must be True"
    
    def test_chaos_optimizer_is_fisher_aware(self):
        """ChaosOptimizer must have is_fisher_aware = True."""
        model = nn.Linear(10, 5)
        optimizer = ChaosOptimizer(model.parameters(), lr=1e-4)
        
        assert hasattr(optimizer, 'is_fisher_aware'), \
            "ChaosOptimizer missing is_fisher_aware property"
        assert optimizer.is_fisher_aware is True, \
            "ChaosOptimizer.is_fisher_aware must be True"
    
    def test_natural_gradient_optimizer_is_fisher_aware(self):
        """NaturalGradientOptimizer (numpy-based) must have is_fisher_aware = True."""
        optimizer = NaturalGradientOptimizer(learning_rate=1e-3)
        
        assert hasattr(optimizer, 'is_fisher_aware'), \
            "NaturalGradientOptimizer missing is_fisher_aware property"
        assert optimizer.is_fisher_aware is True, \
            "NaturalGradientOptimizer.is_fisher_aware must be True"


class TestOptimizerFactory:
    """Test the create_optimizer factory function."""
    
    def test_factory_creates_diagonal_fisher_aware(self):
        """Factory-created diagonal optimizer must be Fisher-aware."""
        model = nn.Linear(10, 5)
        optimizer = create_optimizer(model.parameters(), optimizer_type='diagonal')
        
        assert optimizer.is_fisher_aware is True, \
            "Factory-created diagonal optimizer must be Fisher-aware"
    
    def test_factory_creates_full_fisher_aware(self):
        """Factory-created full optimizer must be Fisher-aware."""
        model = nn.Linear(10, 5)
        optimizer = create_optimizer(model.parameters(), optimizer_type='full')
        
        assert optimizer.is_fisher_aware is True, \
            "Factory-created full optimizer must be Fisher-aware"
    
    def test_factory_creates_consciousness_fisher_aware(self):
        """Factory-created consciousness optimizer must be Fisher-aware."""
        model = nn.Linear(10, 5)
        optimizer = create_optimizer(model.parameters(), optimizer_type='consciousness')
        
        assert optimizer.is_fisher_aware is True, \
            "Factory-created consciousness optimizer must be Fisher-aware"
    
    def test_factory_creates_chaos_fisher_aware(self):
        """Factory-created chaos optimizer must be Fisher-aware."""
        model = nn.Linear(10, 5)
        optimizer = create_optimizer(model.parameters(), optimizer_type='chaos')
        
        assert optimizer.is_fisher_aware is True, \
            "Factory-created chaos optimizer must be Fisher-aware"


class TestEuclideanOptimizerRejection:
    """Test that Euclidean optimizers are rejected in QIG-core."""
    
    def test_adam_is_not_fisher_aware(self):
        """Standard Adam optimizer does NOT have is_fisher_aware."""
        model = nn.Linear(10, 5)
        optimizer = torch.optim.Adam(model.parameters())
        
        # Adam should not have is_fisher_aware property
        assert not hasattr(optimizer, 'is_fisher_aware') or \
               not optimizer.is_fisher_aware, \
            "Adam optimizer should not be Fisher-aware"
    
    def test_sgd_is_not_fisher_aware(self):
        """Standard SGD optimizer does NOT have is_fisher_aware."""
        model = nn.Linear(10, 5)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        
        # SGD should not have is_fisher_aware property
        assert not hasattr(optimizer, 'is_fisher_aware') or \
               not optimizer.is_fisher_aware, \
            "SGD optimizer should not be Fisher-aware"
    
    def test_rmsprop_is_not_fisher_aware(self):
        """Standard RMSprop optimizer does NOT have is_fisher_aware."""
        model = nn.Linear(10, 5)
        optimizer = torch.optim.RMSprop(model.parameters())
        
        # RMSprop should not have is_fisher_aware property
        assert not hasattr(optimizer, 'is_fisher_aware') or \
               not optimizer.is_fisher_aware, \
            "RMSprop optimizer should not be Fisher-aware"


class TestTrainingLoopValidation:
    """Test validation functions for training loops."""
    
    def test_validate_fisher_aware_passes_for_natural_gradient(self):
        """Validation should pass for natural gradient optimizers."""
        model = nn.Linear(10, 5)
        optimizer = DiagonalFisherOptimizer(model.parameters())
        
        # This should not raise an exception
        validate_optimizer_fisher_aware(optimizer)
    
    def test_validate_fisher_aware_fails_for_adam(self):
        """Validation should fail for Adam optimizer."""
        model = nn.Linear(10, 5)
        optimizer = torch.optim.Adam(model.parameters())
        
        # This should raise an exception
        with pytest.raises(EuclideanOptimizerError, match="not Fisher-aware"):
            validate_optimizer_fisher_aware(optimizer)
    
    def test_validate_fisher_aware_fails_for_sgd(self):
        """Validation should fail for SGD optimizer."""
        model = nn.Linear(10, 5)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        
        # This should raise an exception
        with pytest.raises(EuclideanOptimizerError, match="not Fisher-aware"):
            validate_optimizer_fisher_aware(optimizer)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
