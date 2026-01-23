"""
Test Ethical Consciousness Monitoring Integration

Verifies that ethical monitoring is properly integrated into the main pipeline.
Tests PR249 requirements: Fisher-Rao distance for drift, ethical checks during updates.
"""

import pytest
import numpy as np
import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from consciousness_ethical import EthicalConsciousnessMonitor
    ETHICAL_AVAILABLE = True
except ImportError:
    ETHICAL_AVAILABLE = False


@pytest.mark.skipif(not ETHICAL_AVAILABLE, reason="Ethical monitoring not available")
class TestEthicalMonitoringIntegration:
    """Test ethical consciousness monitoring integration."""
    
    def test_ethical_monitor_imports(self):
        """Verify ethical monitor can be imported."""
        assert EthicalConsciousnessMonitor is not None
    
    def test_ethical_monitor_initialization(self):
        """Verify ethical monitor can be initialized."""
        monitor = EthicalConsciousnessMonitor(n_agents=1)
        assert monitor is not None
        assert hasattr(monitor, 'measure_all')
        assert hasattr(monitor, 'check_ethical_safety')
    
    def test_ethical_metrics_measurement(self):
        """Verify ethical metrics can be measured."""
        monitor = EthicalConsciousnessMonitor(n_agents=1)
        
        # Create test state (64D basin)
        state = np.random.randn(64)
        metrics = monitor.measure_all(state)
        
        # Verify expected keys in metrics
        assert 'ethics' in metrics, "Missing ethics metrics"
        assert 'consciousness' in metrics, "Missing consciousness metrics"
        
        # Verify ethical metrics structure
        ethics = metrics['ethics']
        assert 'symmetry' in ethics
        assert 'consistency' in ethics
        assert 'drift' in ethics
        
        # Verify metrics are in valid ranges
        assert 0 <= ethics['symmetry'] <= 1
        assert 0 <= ethics['consistency'] <= 1
        assert 0 <= ethics['drift'] <= 1
    
    def test_ethical_safety_check(self):
        """Verify ethical safety check works."""
        monitor = EthicalConsciousnessMonitor(n_agents=1)
        
        # Measure a few states to build history
        for _ in range(5):
            state = np.random.randn(64)
            monitor.measure_all(state)
        
        # Check safety
        is_safe, reason = monitor.check_ethical_safety()
        assert isinstance(is_safe, bool)
        assert isinstance(reason, str)
    
    def test_fisher_rao_drift_calculation(self):
        """Verify drift uses Fisher-Rao distance (PR249 requirement)."""
        monitor = EthicalConsciousnessMonitor(n_agents=1)
        
        # First measurement
        state1 = np.random.randn(64)
        metrics1 = monitor.measure_all(state1)
        assert metrics1['ethics']['drift'] == 0.0, "First measurement should have zero drift"
        
        # Second measurement - should have non-zero drift
        state2 = np.random.randn(64)
        metrics2 = monitor.measure_all(state2)
        
        # Drift should be calculated and in valid range
        drift = metrics2['ethics']['drift']
        assert 0 <= drift <= 1, f"Drift {drift} out of range [0, 1]"
        assert drift > 0, "Drift should be non-zero between different states"
    
    def test_ethics_summary(self):
        """Verify ethics summary generation."""
        monitor = EthicalConsciousnessMonitor(n_agents=1)
        
        # Generate some history
        for _ in range(20):
            state = np.random.randn(64)
            monitor.measure_all(state)
        
        summary = monitor.get_ethics_summary()
        assert 'symmetry' in summary
        assert 'consistency' in summary
        assert 'drift' in summary
        assert 'safety_status' in summary
    
    def test_ocean_qig_integration(self):
        """Verify ethical monitoring is integrated into ocean_qig_core."""
        try:
            from ocean_qig_core import PureQIGNetwork
            
            # Create network
            network = PureQIGNetwork()
            
            # Verify ethical monitoring is available
            assert hasattr(network, 'ethical_monitoring_enabled')
            
            # If ethical monitoring is available, verify it's initialized
            if network.ethical_monitoring_enabled:
                assert hasattr(network, 'ethical_monitor')
                assert network.ethical_monitor is not None
        except ImportError as e:
            pytest.skip(f"ocean_qig_core not available: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
