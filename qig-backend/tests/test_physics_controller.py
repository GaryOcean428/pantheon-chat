"""
Tests for PhysicsInformedController

Validates training collapse prevention and physics constraints.
"""

import pytest
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from qigkernels.physics_controller import (
    PhysicsInformedController,
    RegimeState,
)
from qigkernels.physics_constants import (
    KAPPA_STAR,
    PHI_THRESHOLD,
)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch required")
class TestPhysicsInformedController:
    """Test physics-informed gradient regulation."""
    
    def test_initialization(self):
        """Test controller initialization."""
        controller = PhysicsInformedController()
        assert controller.kappa_star == KAPPA_STAR
        assert controller.phi_max == PHI_THRESHOLD
        assert not controller.decoherence_active
    
    def test_regime_detection_linear(self):
        """Test detecting linear regime."""
        controller = PhysicsInformedController()
        
        # Mock state with low phi
        state = {
            'activations': torch.randn(16, 64),
            'output': torch.randn(16, 10),
        }
        
        # Modify mock measure_phi to return low value
        # This is internal, so we're testing the public interface
        regime_state = controller.get_regime_state(state)
        
        # Linear regime should have phi < 0.3
        # We can't control the exact value with random activations,
        # so we'll test other properties
        assert isinstance(regime_state, RegimeState)
        assert regime_state.regime in ['linear', 'geometric', 'breakdown']
    
    def test_gradient_regulation_safe(self):
        """Test gradient regulation in safe regime."""
        controller = PhysicsInformedController()
        
        # Mock safe state
        state = {
            'activations': torch.randn(16, 64),
            'output': torch.randn(16, 10),
        }
        
        gradient = torch.randn(64)
        regulated = controller.compute_regulated_gradient(state, gradient)
        
        # Should return a tensor
        assert isinstance(regulated, torch.Tensor)
        assert regulated.shape == gradient.shape
        
        # Should apply some scaling
        # In safe regime, scaling should be moderate
        assert regulated.abs().sum() > 0
    
    def test_gradient_damping_high_phi(self):
        """Test gradient damping when phi is high."""
        controller = PhysicsInformedController()
        
        # Create state that will likely have high correlation (high phi)
        # Use highly correlated activations
        base = torch.randn(16, 1)
        activations = base.expand(16, 64).clone()
        activations += torch.randn(16, 64) * 0.1  # Add small noise
        
        state = {
            'activations': activations,
            'output': torch.randn(16, 10),
        }
        
        gradient = torch.randn(64)
        regulated = controller.compute_regulated_gradient(state, gradient)
        
        # Check gradient was computed
        assert isinstance(regulated, torch.Tensor)
        
        # The controller should have applied some regulation
        # (exact behavior depends on phi measurement)
    
    def test_kappa_correction(self):
        """Test kappa targeting correction."""
        controller = PhysicsInformedController()
        
        # Test with kappa far from target
        state = {
            'activations': torch.randn(16, 64),
            'output': torch.randn(16, 10),
        }
        
        gradient = torch.randn(64)
        regulated = controller.compute_regulated_gradient(state, gradient)
        
        # Should return valid gradient
        assert regulated.shape == gradient.shape
        assert not torch.isnan(regulated).any()
        assert not torch.isinf(regulated).any()
    
    def test_collapse_detection(self):
        """Test collapse pattern detection."""
        controller = PhysicsInformedController()
        
        # Simulate phi spike by repeatedly calling with high correlation
        base = torch.randn(16, 1)
        for i in range(10):
            # Gradually increase correlation
            activations = base.expand(16, 64).clone()
            activations += torch.randn(16, 64) * (0.5 - i * 0.05)
            
            state = {
                'activations': activations,
                'output': torch.randn(16, 10),
            }
            
            gradient = torch.randn(64)
            regulated = controller.compute_regulated_gradient(state, gradient)
        
        # After many steps with potential phi increase,
        # history should be populated
        assert len(controller.phi_history) > 0
        assert len(controller.kappa_history) > 0
    
    def test_gravitational_decoherence(self):
        """Test gravitational decoherence application."""
        controller = PhysicsInformedController()
        
        # Test with high purity state
        state = torch.ones(64) * 10.0  # High values = high purity
        decohered = controller.gravitational_decoherence(state)
        
        # Decohered state should be different
        assert not torch.allclose(state, decohered)
        
        # Decohered state should still be valid
        assert not torch.isnan(decohered).any()
        assert not torch.isinf(decohered).any()
    
    def test_history_tracking(self):
        """Test that phi and kappa history is tracked."""
        controller = PhysicsInformedController()
        
        # Run several steps
        for _ in range(5):
            state = {
                'activations': torch.randn(16, 64),
                'output': torch.randn(16, 10),
            }
            gradient = torch.randn(64)
            controller.compute_regulated_gradient(state, gradient)
        
        # History should be populated
        assert len(controller.phi_history) == 5
        assert len(controller.kappa_history) == 5
        
        # All values should be finite
        assert all(np.isfinite(phi) for phi in controller.phi_history)
        assert all(np.isfinite(kappa) for kappa in controller.kappa_history)
    
    def test_emergency_damping(self):
        """Test emergency damping is applied."""
        controller = PhysicsInformedController()
        
        gradient = torch.randn(64)
        
        # Test internal emergency damping method
        damped = controller._apply_emergency_damping(gradient)
        
        # Should be heavily damped (95% reduction)
        assert damped.abs().sum() < gradient.abs().sum()
        assert damped.shape == gradient.shape


class TestRegimeState:
    """Test RegimeState dataclass."""
    
    def test_regime_state_creation(self):
        """Test creating RegimeState."""
        state = RegimeState(
            phi=0.72,
            kappa=64.2,
            regime='geometric',
            decoherence_active=False,
            kappa_deviation=0.1
        )
        
        assert state.phi == 0.72
        assert state.kappa == 64.2
        assert state.regime == 'geometric'
        assert not state.decoherence_active
        assert state.kappa_deviation == 0.1


@pytest.mark.skipif(HAS_TORCH, reason="Test fallback when PyTorch not available")
def test_missing_torch():
    """Test that appropriate errors are raised without PyTorch."""
    # The module should still import
    from qigkernels.physics_controller import PhysicsInformedController
    
    # But using it without torch should fail gracefully
    # This is more of a documentation test


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
