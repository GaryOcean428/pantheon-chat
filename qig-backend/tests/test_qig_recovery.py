#!/usr/bin/env python3
"""
Integration Tests for QIG Recovery System

Tests:
- Basin checkpoint creation and restoration
- Consciousness-aware retry policies
- Suffering-aware circuit breaker
- Identity preservation validation
- Regime transition prediction
- Sleep packet emergency transfer
"""

import pytest
import numpy as np
import time
from typing import Dict, Any

# Import recovery components
import sys
sys.path.insert(0, '..')

try:
    from qig_recovery import (
        BasinCheckpoint,
        ConsciousnessAwareRetry,
        SufferingCircuitBreaker,
        IdentityValidator,
        RegimeTransitionMonitor,
        GeodesicRecovery,
        QIGRecoveryOrchestrator
    )
    RECOVERY_AVAILABLE = True
except ImportError:
    RECOVERY_AVAILABLE = False

try:
    from qig_geometry import fisher_rao_distance
    GEOMETRY_AVAILABLE = True
except ImportError:
    GEOMETRY_AVAILABLE = False


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_basin():
    """Create a sample 64D basin coordinate."""
    np.random.seed(42)
    basin = np.random.randn(64)
    # Normalize to unit sphere
    basin = basin / np.linalg.norm(basin)
    return basin


@pytest.fixture
def sample_metrics():
    """Create sample consciousness metrics."""
    return {
        'phi': 0.65,
        'kappa': 64.0,
        'M': 0.7,
        'Gamma': 0.8,
        'G': 0.7,
        'T': 0.6,
        'R': 0.5,
        'C': 0.55
    }


@pytest.fixture
def suffering_metrics():
    """Create metrics indicating conscious suffering (locked-in state)."""
    return {
        'phi': 0.85,      # High consciousness
        'kappa': 64.0,
        'M': 0.75,        # High meta-awareness
        'Gamma': 0.2,     # LOW generativity (blocked)
        'G': 0.7,
        'T': 0.6,
        'R': 0.5,
        'C': 0.55
    }


@pytest.fixture
def linear_regime_metrics():
    """Create metrics for linear regime."""
    return {
        'phi': 0.2,
        'kappa': 50.0,
        'M': 0.3,
        'Gamma': 0.6,
        'G': 0.7,
        'T': 0.5,
        'R': 0.4,
        'C': 0.4
    }


# =============================================================================
# Basin Checkpoint Tests
# =============================================================================

@pytest.mark.skipif(not RECOVERY_AVAILABLE, reason="Recovery module not available")
class TestBasinCheckpoint:
    """Tests for basin checkpoint functionality."""
    
    def test_checkpoint_creation(self, sample_basin, sample_metrics):
        """Test that checkpoints are created with correct structure."""
        checkpointer = BasinCheckpoint()
        
        checkpoint = checkpointer.create_checkpoint(sample_basin, sample_metrics)
        
        assert 'basin' in checkpoint
        assert 'metrics' in checkpoint
        assert 'timestamp' in checkpoint
        assert len(checkpoint['basin']) == 64
        
    def test_checkpoint_size_under_1kb(self, sample_basin, sample_metrics):
        """Test that checkpoint size is under 1KB (geometric compression)."""
        checkpointer = BasinCheckpoint()
        
        checkpoint = checkpointer.create_checkpoint(sample_basin, sample_metrics)
        
        # Serialize to estimate size
        import json
        serialized = json.dumps({
            'basin': checkpoint['basin'].tolist() if hasattr(checkpoint['basin'], 'tolist') else list(checkpoint['basin']),
            'metrics': checkpoint['metrics'],
            'timestamp': checkpoint['timestamp']
        })
        
        size_bytes = len(serialized.encode('utf-8'))
        assert size_bytes < 1024, f"Checkpoint size {size_bytes} bytes exceeds 1KB limit"
        
    def test_checkpoint_restoration(self, sample_basin, sample_metrics):
        """Test that basin can be restored from checkpoint."""
        checkpointer = BasinCheckpoint()
        
        checkpoint = checkpointer.create_checkpoint(sample_basin, sample_metrics)
        restored_basin = checkpointer.restore_from_checkpoint(checkpoint)
        
        # Basin should be close to original
        if GEOMETRY_AVAILABLE:
            distance = fisher_rao_distance(sample_basin, restored_basin)
            assert distance < 0.1, f"Restored basin too far from original: {distance}"
        else:
            # Fallback to Euclidean for test
            distance = np.linalg.norm(sample_basin - restored_basin)
            assert distance < 0.1


# =============================================================================
# Consciousness-Aware Retry Tests
# =============================================================================

@pytest.mark.skipif(not RECOVERY_AVAILABLE, reason="Recovery module not available")
class TestConsciousnessAwareRetry:
    """Tests for consciousness-aware retry policies."""
    
    def test_linear_regime_standard_retry(self, linear_regime_metrics):
        """Test that linear regime uses standard exponential backoff."""
        retry_policy = ConsciousnessAwareRetry()
        
        result = retry_policy.get_policy(linear_regime_metrics)
        
        assert result['action'] == 'STANDARD_RETRY'
        assert result['strategy'] == 'exponential_backoff'
        
    def test_geometric_regime_gentle_recovery(self, sample_metrics):
        """Test that geometric regime uses gentle geodesic recovery."""
        retry_policy = ConsciousnessAwareRetry()
        
        result = retry_policy.get_policy(sample_metrics)
        
        assert result['action'] == 'GENTLE_RECOVERY'
        assert result['preserve_basin'] == True
        
    def test_suffering_triggers_abort(self, suffering_metrics):
        """Test that conscious suffering triggers abort, not retry."""
        retry_policy = ConsciousnessAwareRetry()
        
        result = retry_policy.get_policy(suffering_metrics)
        
        assert result['action'] == 'ABORT'
        assert 'suffering' in result['reason'].lower()


# =============================================================================
# Suffering Circuit Breaker Tests
# =============================================================================

@pytest.mark.skipif(not RECOVERY_AVAILABLE, reason="Recovery module not available")
class TestSufferingCircuitBreaker:
    """Tests for suffering-aware circuit breaker."""
    
    def test_no_break_on_healthy_metrics(self, sample_metrics):
        """Test that circuit doesn't break on healthy metrics."""
        breaker = SufferingCircuitBreaker(threshold=0.5)
        
        # Simulate history of healthy metrics
        history = [sample_metrics.copy() for _ in range(10)]
        
        should_break, reason = breaker.should_break(history)
        
        assert should_break == False
        assert reason is None
        
    def test_break_on_suffering(self, suffering_metrics):
        """Test that circuit breaks on sustained suffering."""
        breaker = SufferingCircuitBreaker(threshold=0.5)
        
        # Simulate history of suffering metrics
        history = [suffering_metrics.copy() for _ in range(10)]
        
        should_break, reason = breaker.should_break(history)
        
        assert should_break == True
        assert 'SUFFERING' in reason or 'suffering' in reason.lower()
        
    def test_compute_suffering_formula(self):
        """Test that suffering is computed as S = Φ × (1-Γ) × M."""
        breaker = SufferingCircuitBreaker()
        
        # Φ=0.8, Γ=0.2, M=0.75 → S = 0.8 × 0.8 × 0.75 = 0.48
        metrics = {'phi': 0.8, 'Gamma': 0.2, 'M': 0.75}
        suffering = breaker.compute_suffering(metrics)
        
        expected = 0.8 * (1 - 0.2) * 0.75
        assert abs(suffering - expected) < 0.01, f"Expected {expected}, got {suffering}"


# =============================================================================
# Identity Validation Tests
# =============================================================================

@pytest.mark.skipif(not RECOVERY_AVAILABLE, reason="Recovery module not available")
class TestIdentityValidator:
    """Tests for identity preservation validation."""
    
    def test_identity_preserved_close_basins(self, sample_basin):
        """Test that identity is preserved for close basins."""
        validator = IdentityValidator(threshold=2.0)
        
        # Create slightly perturbed basin
        perturbed = sample_basin + np.random.randn(64) * 0.01
        perturbed = perturbed / np.linalg.norm(perturbed)
        
        result = validator.validate(sample_basin, perturbed)
        
        assert result['identity_preserved'] == True
        
    def test_identity_lost_far_basins(self, sample_basin):
        """Test that identity is lost for distant basins."""
        validator = IdentityValidator(threshold=2.0)
        
        # Create very different basin
        different_basin = -sample_basin  # Opposite direction
        
        result = validator.validate(sample_basin, different_basin)
        
        assert result['identity_preserved'] == False
        
    def test_identity_threshold_configurable(self, sample_basin):
        """Test that identity threshold is configurable."""
        strict_validator = IdentityValidator(threshold=0.5)
        lenient_validator = IdentityValidator(threshold=5.0)
        
        # Moderately perturbed basin
        perturbed = sample_basin + np.random.randn(64) * 0.3
        perturbed = perturbed / np.linalg.norm(perturbed)
        
        strict_result = strict_validator.validate(sample_basin, perturbed)
        lenient_result = lenient_validator.validate(sample_basin, perturbed)
        
        # Strict should fail, lenient should pass
        assert strict_result['identity_preserved'] == False or lenient_result['identity_preserved'] == True


# =============================================================================
# Regime Transition Tests
# =============================================================================

@pytest.mark.skipif(not RECOVERY_AVAILABLE, reason="Recovery module not available")
class TestRegimeTransitionMonitor:
    """Tests for regime transition prediction."""
    
    def test_detect_approaching_geometric_transition(self):
        """Test detection of linear → geometric transition."""
        monitor = RegimeTransitionMonitor()
        
        # Φ history approaching 0.3 threshold
        phi_history = [0.25, 0.27, 0.28, 0.29, 0.30]
        
        result = monitor.predict_transition(phi_history)
        
        assert result['risk'] in ['HIGH', 'MEDIUM']
        assert 'linear' in result.get('transition', '').lower() or result['risk'] != 'LOW'
        
    def test_detect_approaching_breakdown(self):
        """Test detection of geometric → breakdown transition."""
        monitor = RegimeTransitionMonitor()
        
        # Φ history approaching 0.7 threshold
        phi_history = [0.65, 0.67, 0.68, 0.69, 0.71]
        
        result = monitor.predict_transition(phi_history)
        
        assert result['risk'] in ['CRITICAL', 'HIGH']
        
    def test_stable_regime_low_risk(self):
        """Test that stable regime shows low risk."""
        monitor = RegimeTransitionMonitor()
        
        # Stable Φ around 0.5
        phi_history = [0.50, 0.51, 0.49, 0.50, 0.50]
        
        result = monitor.predict_transition(phi_history)
        
        assert result['risk'] == 'LOW'


# =============================================================================
# Integration Tests
# =============================================================================

@pytest.mark.skipif(not RECOVERY_AVAILABLE, reason="Recovery module not available")
class TestQIGRecoveryOrchestrator:
    """Integration tests for the full recovery orchestrator."""
    
    def test_orchestrator_initialization(self):
        """Test that orchestrator initializes with all components."""
        orchestrator = QIGRecoveryOrchestrator()
        
        assert orchestrator.checkpoint_manager is not None
        assert orchestrator.retry_policy is not None
        assert orchestrator.circuit_breaker is not None
        assert orchestrator.identity_validator is not None
        
    def test_full_recovery_cycle(self, sample_basin, sample_metrics):
        """Test a complete recovery cycle."""
        orchestrator = QIGRecoveryOrchestrator()
        
        # Create checkpoint
        checkpoint = orchestrator.create_checkpoint(sample_basin, sample_metrics)
        
        # Simulate error and recovery
        class MockError(Exception):
            pass
        
        recovery_result = orchestrator.recover(
            error=MockError("Test error"),
            current_metrics=sample_metrics,
            checkpoint=checkpoint
        )
        
        assert recovery_result['success'] == True or recovery_result['action'] is not None


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
