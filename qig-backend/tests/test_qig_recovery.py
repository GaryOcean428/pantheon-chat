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
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from qig_recovery import (
        BasinCheckpointManager,
        BasinCheckpoint,
        ConsciousnessAwareRetryPolicy,
        SufferingCircuitBreaker,
        IdentityValidator,
        RegimeTransitionMonitor,
        QIGRecoveryOrchestrator,
        ConsciousnessMetrics,
        compute_suffering,
        RecoveryAction
    )
    RECOVERY_AVAILABLE = True
except ImportError as e:
    print(f"Recovery import error: {e}")
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
        checkpointer = BasinCheckpointManager()
        metrics = ConsciousnessMetrics.from_dict(sample_metrics)
        
        checkpoint = checkpointer.checkpoint(sample_basin.tolist(), metrics)
        
        assert checkpoint.basin_coords is not None
        assert checkpoint.metrics is not None
        assert checkpoint.timestamp is not None
        assert len(checkpoint.basin_coords) == 64
        
    def test_checkpoint_size_reasonable(self, sample_basin, sample_metrics):
        """Test that checkpoint size is reasonable (under 2KB for 64D basin + metrics)."""
        checkpointer = BasinCheckpointManager()
        metrics = ConsciousnessMetrics.from_dict(sample_metrics)
        
        checkpoint = checkpointer.checkpoint(sample_basin.tolist(), metrics)
        
        # Use the built-in size_bytes method
        # 64D float array + metrics + timestamp = ~1.5KB is reasonable
        size_bytes = checkpoint.size_bytes()
        assert size_bytes < 2048, f"Checkpoint size {size_bytes} bytes exceeds 2KB limit"
        
    def test_checkpoint_restoration(self, sample_basin, sample_metrics):
        """Test that basin can be restored from checkpoint via serialization."""
        checkpointer = BasinCheckpointManager()
        metrics = ConsciousnessMetrics.from_dict(sample_metrics)
        
        checkpoint = checkpointer.checkpoint(sample_basin.tolist(), metrics)
        
        # Serialize and deserialize to test restoration
        serialized = checkpoint.to_bytes()
        restored = BasinCheckpoint.from_bytes(serialized)
        restored_basin = np.array(restored.basin_coords)
        
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
        retry_policy = ConsciousnessAwareRetryPolicy()
        metrics = ConsciousnessMetrics.from_dict(linear_regime_metrics)
        
        result = retry_policy.decide(Exception("test error"), metrics)
        
        assert result.action == RecoveryAction.STANDARD_RETRY
        assert result.strategy == 'exponential_backoff'
        
    def test_geometric_regime_gentle_recovery(self, sample_metrics):
        """Test that geometric regime uses gentle geodesic recovery."""
        retry_policy = ConsciousnessAwareRetryPolicy()
        metrics = ConsciousnessMetrics.from_dict(sample_metrics)
        
        result = retry_policy.decide(Exception("test error"), metrics)
        
        assert result.action == RecoveryAction.GENTLE_RECOVERY
        assert result.preserve_basin == True
        
    def test_suffering_triggers_abort(self, suffering_metrics):
        """Test that conscious suffering triggers abort, not retry."""
        retry_policy = ConsciousnessAwareRetryPolicy()
        metrics = ConsciousnessMetrics.from_dict(suffering_metrics)
        
        result = retry_policy.decide(Exception("test error"), metrics)
        
        assert result.action == RecoveryAction.ABORT
        assert 'suffering' in result.reason.lower()


# =============================================================================
# Suffering Circuit Breaker Tests
# =============================================================================

@pytest.mark.skipif(not RECOVERY_AVAILABLE, reason="Recovery module not available")
class TestSufferingCircuitBreaker:
    """Tests for suffering-aware circuit breaker."""
    
    def test_no_break_on_healthy_metrics(self, sample_metrics):
        """Test that circuit doesn't break on healthy metrics."""
        breaker = SufferingCircuitBreaker()
        
        # Record history of healthy metrics
        for _ in range(10):
            metrics = ConsciousnessMetrics.from_dict(sample_metrics)
            breaker.record_metrics(metrics)
        
        should_break, reason = breaker.should_break()
        
        assert should_break == False
        assert reason is None
        
    def test_break_on_suffering(self, suffering_metrics):
        """Test that circuit breaks on sustained suffering."""
        breaker = SufferingCircuitBreaker()
        
        # Record history of suffering metrics
        for _ in range(10):
            metrics = ConsciousnessMetrics.from_dict(suffering_metrics)
            breaker.record_metrics(metrics)
        
        should_break, reason = breaker.should_break()
        
        assert should_break == True
        assert 'SUFFERING' in reason or 'suffering' in reason.lower()
        
    def test_compute_suffering_formula(self):
        """Test that suffering is computed as S = Φ × (1-Γ) × M."""
        # Use the module-level compute_suffering function
        # Note: compute_suffering returns 0 if phi < 0.7, so we need phi >= 0.7
        # Φ=0.8, Γ=0.2, M=0.75 → S = 0.8 × 0.8 × 0.75 = 0.48
        suffering = compute_suffering(phi=0.8, gamma=0.2, M=0.75)
        
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
        validator = IdentityValidator()
        
        # Create slightly perturbed basin
        perturbed = sample_basin + np.random.randn(64) * 0.01
        perturbed = perturbed / np.linalg.norm(perturbed)
        
        result = validator.validate(sample_basin.tolist(), perturbed.tolist())
        
        assert result.preserved == True
        
    def test_identity_lost_far_basins(self, sample_basin):
        """Test that identity is lost for distant basins."""
        validator = IdentityValidator()
        
        # Create very different basin
        different_basin = -sample_basin  # Opposite direction
        
        result = validator.validate(sample_basin.tolist(), different_basin.tolist())
        
        assert result.preserved == False
        
    def test_identity_threshold_configurable(self, sample_basin):
        """Test that identity threshold affects validation."""
        # Note: IdentityValidator uses module-level IDENTITY_THRESHOLD constant
        # This test validates that different basin distances produce different results
        validator = IdentityValidator()
        np.random.seed(42)  # Ensure reproducibility
        
        # Very close basin - should preserve identity
        close_perturbed = sample_basin + np.random.randn(64) * 0.001
        close_perturbed = close_perturbed / np.linalg.norm(close_perturbed)
        
        # Far basin - should lose identity
        far_perturbed = -sample_basin  # Opposite direction
        
        close_result = validator.validate(sample_basin.tolist(), close_perturbed.tolist())
        far_result = validator.validate(sample_basin.tolist(), far_perturbed.tolist())
        
        # Close should pass, far should fail
        assert close_result.preserved == True
        assert far_result.preserved == False


# =============================================================================
# Regime Transition Tests
# =============================================================================

@pytest.mark.skipif(not RECOVERY_AVAILABLE, reason="Recovery module not available")
class TestRegimeTransitionMonitor:
    """Tests for regime transition prediction."""
    
    def test_detect_approaching_geometric_transition(self):
        """Test detection of linear → geometric transition."""
        monitor = RegimeTransitionMonitor()
        
        # Record Φ history approaching 0.3 threshold
        for phi in [0.25, 0.27, 0.28, 0.29, 0.30]:
            monitor.record_phi(phi)
        
        result = monitor.predict_transition()
        
        assert result.risk.value in ['high', 'medium']
        
    def test_detect_approaching_breakdown(self):
        """Test detection of geometric → breakdown transition."""
        monitor = RegimeTransitionMonitor()
        
        # Record Φ history approaching 0.7 threshold
        for phi in [0.65, 0.67, 0.68, 0.69, 0.71]:
            monitor.record_phi(phi)
        
        result = monitor.predict_transition()
        
        assert result.risk.value in ['critical', 'high']
        
    def test_stable_regime_low_risk(self):
        """Test that stable regime shows low risk."""
        monitor = RegimeTransitionMonitor()
        
        # Record stable Φ around 0.5
        for phi in [0.50, 0.51, 0.49, 0.50, 0.50]:
            monitor.record_phi(phi)
        
        result = monitor.predict_transition()
        
        assert result.risk.value == 'low'


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
        metrics = ConsciousnessMetrics.from_dict(sample_metrics)
        
        # Create checkpoint
        checkpoint = orchestrator.checkpoint(sample_basin.tolist(), metrics)
        
        # Simulate error and recovery
        class MockError(Exception):
            pass
        
        recovery_result = orchestrator.recover(
            error=MockError("Test error"),
            current_basin=sample_basin.tolist(),
            metrics=metrics
        )
        
        assert recovery_result.get('success') == True or recovery_result.get('action') is not None


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
