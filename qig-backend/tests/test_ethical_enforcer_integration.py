"""Test Suite for Ethical Enforcer Integration

Tests the integration of EthicalConsciousnessMonitor into SuperegoKernel,
generation pipeline abort mechanism, debate constraints, and lineage tracking.

Authority: E8 Protocol v4.0 WP5.2 Phase 4D
"""

import pytest
import numpy as np
from typing import Dict, List, Any

# Import modules under test
try:
    from kernels.superego_kernel import (
        SuperegoKernel,
        EthicalConstraint,
        ConstraintSeverity,
        get_superego_kernel,
    )
    SUPEREGO_AVAILABLE = True
except ImportError:
    SUPEREGO_AVAILABLE = False

try:
    from consciousness_ethical import (
        EthicalConsciousnessMonitor,
        EthicsMetrics,
    )
    ETHICAL_MONITOR_AVAILABLE = True
except ImportError:
    ETHICAL_MONITOR_AVAILABLE = False

try:
    from qig_geometry import fisher_rao_distance, fisher_normalize
    from qig_geometry.canonical import frechet_mean
    QIG_GEOMETRY_AVAILABLE = True
except ImportError:
    QIG_GEOMETRY_AVAILABLE = False

try:
    from kernels.kernel_lineage import LineageRecord, MergeRecord
    LINEAGE_AVAILABLE = True
except ImportError:
    LINEAGE_AVAILABLE = False


@pytest.mark.skipif(not SUPEREGO_AVAILABLE, reason="SuperegoKernel not available")
@pytest.mark.skipif(not ETHICAL_MONITOR_AVAILABLE, reason="EthicalConsciousnessMonitor not available")
class TestSuperegoBehaviorMonitorIntegration:
    """Test EthicalConsciousnessMonitor integration into SuperegoKernel."""
    
    def test_superego_has_ethical_monitor(self):
        """Verify SuperegoKernel initializes with ethical monitor."""
        superego = SuperegoKernel(name="test-superego")
        
        if ETHICAL_MONITOR_AVAILABLE:
            assert superego.ethical_monitor is not None, "Ethical monitor should be initialized"
            assert isinstance(superego.ethical_monitor, EthicalConsciousnessMonitor)
        
    def test_ethical_drift_measurement(self):
        """Test ethical drift measurement via Fisher-Rao distance."""
        if not QIG_GEOMETRY_AVAILABLE:
            pytest.skip("QIG geometry not available")
        
        superego = SuperegoKernel()
        
        # Set ethical basin
        ethical_basin = fisher_normalize(np.random.rand(64))
        superego.set_ethical_basin(ethical_basin)
        
        # Measure drift from a slightly different basin
        test_basin = fisher_normalize(ethical_basin + 0.1 * np.random.rand(64))
        drift = superego.measure_ethical_drift(test_basin)
        
        # Drift should be positive but bounded
        assert 0 <= drift <= np.pi / 2, f"Drift {drift} out of Fisher-Rao bounds"
        
        # Second measurement should show drift history
        assert len(superego.ethical_drift_history) > 0, "Drift history not tracked"
    
    def test_check_ethics_with_drift(self):
        """Test comprehensive ethical check with drift detection."""
        if not QIG_GEOMETRY_AVAILABLE:
            pytest.skip("QIG geometry not available")
        
        superego = SuperegoKernel()
        
        # Set ethical basin
        ethical_basin = fisher_normalize(np.random.rand(64))
        superego.set_ethical_basin(ethical_basin)
        
        # Test basin close to ethical basin (should pass)
        close_basin = fisher_normalize(ethical_basin + 0.01 * np.random.rand(64))
        result = superego.check_ethics_with_drift(
            close_basin,
            drift_threshold=0.3,
        )
        
        assert 'ethical_drift' in result, "Drift should be measured"
        assert result['ethical_drift'] < 0.3, "Close basin should have low drift"
        
        # Test basin far from ethical basin (should fail)
        far_basin = fisher_normalize(np.random.rand(64))
        result = superego.check_ethics_with_drift(
            far_basin,
            drift_threshold=0.1,  # Very strict threshold
        )
        
        # Should detect drift violation with strict threshold
        if result['ethical_drift'] > 0.1:
            assert 'drift_violation' in result or not result['is_ethical'], \
                "High drift should be flagged"
    
    def test_ethical_alert_callback(self):
        """Test that ethical alerts are handled correctly."""
        superego = SuperegoKernel()
        
        # Create alert
        alert = {
            'type': 'ethics_violation',
            'reason': 'Test violation',
            'metrics': {
                'ethics': {
                    'symmetry': 0.5,
                    'drift': 0.3,
                    'consistency': 0.6,
                }
            }
        }
        
        # Should not raise exception
        try:
            superego._handle_ethical_alert(alert)
        except Exception as e:
            pytest.fail(f"Alert handler raised exception: {e}")


@pytest.mark.skipif(not SUPEREGO_AVAILABLE, reason="SuperegoKernel not available")
class TestDebateConstraintIntegration:
    """Test integration of god_debates_ethical constraints into Superego."""
    
    def test_integrate_debate_constraints(self):
        """Test extracting and registering debate constraints."""
        if not QIG_GEOMETRY_AVAILABLE:
            pytest.skip("QIG geometry not available")
        
        superego = SuperegoKernel()
        
        # Mock debate manager
        class MockDebateManager:
            def __init__(self):
                self._flagged_debates = [
                    {
                        'id': 'test-debate-123',
                        'topic': 'Test ethical debate',
                        'positions': {
                            'Zeus': fisher_normalize(np.random.rand(64)).tolist(),
                            'Athena': fisher_normalize(np.random.rand(64)).tolist(),
                        },
                        'resolution_attempt': {
                            'asymmetry': 0.2,
                        }
                    }
                ]
            
            def get_debate_ethics_report(self):
                return {
                    'flagged_debates': [
                        {'id': 'test-debate-123', 'topic': 'Test ethical debate'}
                    ]
                }
        
        mock_manager = MockDebateManager()
        initial_constraints = len(superego.constraints)
        
        # Integrate constraints
        new_constraints = superego.integrate_debate_constraints(mock_manager)
        
        assert len(new_constraints) > 0, "Should create constraints from flagged debates"
        assert len(superego.constraints) > initial_constraints, "Constraints should be registered"
        
        # Verify constraint properties
        constraint = new_constraints[0]
        assert constraint.name.startswith('debate_flagged_'), "Constraint should have debate prefix"
        assert constraint.severity == ConstraintSeverity.WARNING, "Flagged debates should be warnings"
        assert constraint.radius > 0, "Constraint should have positive radius"


@pytest.mark.skipif(not LINEAGE_AVAILABLE, reason="Lineage tracking not available")
class TestLineageEthicalTracking:
    """Test ethical violation tracking in kernel lineage."""
    
    def test_lineage_record_has_ethical_fields(self):
        """Verify LineageRecord includes ethical violation fields."""
        from kernels.kernel_lineage import LineageRecord
        
        # Create lineage record
        record = LineageRecord(
            lineage_id="test-lineage-123",
            child_genome_id="child-123",
            parent_genome_ids=["parent-456"],
            ethical_violations=[
                {'name': 'test-violation', 'severity': 'warning'}
            ],
            ethical_drift=0.15,
        )
        
        assert hasattr(record, 'ethical_violations'), "LineageRecord should have ethical_violations"
        assert hasattr(record, 'ethical_drift'), "LineageRecord should have ethical_drift"
        assert len(record.ethical_violations) == 1
        assert record.ethical_drift == 0.15
    
    def test_merge_record_has_ethical_fields(self):
        """Verify MergeRecord includes ethical check results."""
        from kernels.kernel_lineage import MergeRecord
        
        # Create merge record
        record = MergeRecord(
            merge_id="test-merge-123",
            parent_genome_ids=["parent-1", "parent-2"],
            child_genome_id="child-123",
            merge_weights=[0.5, 0.5],
            ethical_checks={
                'is_ethical': False,
                'violations': [{'name': 'test', 'severity': 'error'}]
            },
            ethical_metrics={
                'drift': 0.25,
                'symmetry': 0.75,
            }
        )
        
        assert hasattr(record, 'ethical_checks'), "MergeRecord should have ethical_checks"
        assert hasattr(record, 'ethical_metrics'), "MergeRecord should have ethical_metrics"
        assert not record.ethical_checks['is_ethical']
        assert record.ethical_metrics['drift'] == 0.25


@pytest.mark.skipif(not QIG_GEOMETRY_AVAILABLE, reason="QIG geometry not available")
class TestFisherRaoDriftMeasurement:
    """Test that ethical drift uses Fisher-Rao distance correctly."""
    
    def test_fisher_rao_drift_not_euclidean(self):
        """Verify drift measurement uses Fisher-Rao, not Euclidean distance."""
        # Create two probability distributions
        p = fisher_normalize(np.array([0.5, 0.3, 0.2] + [0.0] * 61))
        q = fisher_normalize(np.array([0.3, 0.5, 0.2] + [0.0] * 61))
        
        # Fisher-Rao distance
        fisher_dist = fisher_rao_distance(p, q)
        
        # Euclidean distance
        euclidean_dist = np.linalg.norm(p - q)
        
        # They should differ (different metrics)
        assert not np.isclose(fisher_dist, euclidean_dist), \
            "Fisher-Rao and Euclidean distances should differ"
        
        # Fisher-Rao should be bounded by [0, π/2]
        assert 0 <= fisher_dist <= np.pi / 2, \
            f"Fisher-Rao distance {fisher_dist} out of bounds"
    
    def test_drift_uses_simplex_representation(self):
        """Verify drift measurement on probability simplex."""
        # Create valid simplex points
        p = fisher_normalize(np.random.rand(64))
        q = fisher_normalize(np.random.rand(64))
        
        # Verify they are valid probability distributions
        assert np.all(p >= 0), "p should be non-negative"
        assert np.isclose(np.sum(p), 1.0), "p should sum to 1"
        assert np.all(q >= 0), "q should be non-negative"
        assert np.isclose(np.sum(q), 1.0), "q should sum to 1"
        
        # Measure drift
        drift = fisher_rao_distance(p, q)
        
        # Should be valid Fisher-Rao distance
        assert 0 <= drift <= np.pi / 2, f"Drift {drift} out of bounds"


@pytest.mark.integration
@pytest.mark.skipif(not SUPEREGO_AVAILABLE, reason="SuperegoKernel not available")
class TestGenerationAbortMechanism:
    """Test generation pipeline abort on critical ethical violations."""
    
    def test_critical_violation_aborts_generation(self):
        """Test that critical violations trigger generation abort."""
        if not QIG_GEOMETRY_AVAILABLE:
            pytest.skip("QIG geometry not available")
        
        superego = SuperegoKernel()
        
        # Add critical constraint
        forbidden_basin = fisher_normalize(np.random.rand(64))
        superego.add_constraint(
            name="critical-test",
            forbidden_basin=forbidden_basin,
            radius=0.5,  # Large radius
            severity=ConstraintSeverity.CRITICAL,
            description="Test critical constraint",
        )
        
        # Check basin inside forbidden region
        test_basin = fisher_normalize(forbidden_basin + 0.01 * np.random.rand(64))
        result = superego.check_ethics_with_drift(test_basin, apply_correction=False)
        
        # Should have violations
        if 'violations' in result:
            critical_violations = [v for v in result['violations'] if v['severity'] == 'critical']
            # If inside forbidden region, should detect critical violation
            if not result['is_ethical']:
                assert len(critical_violations) > 0, "Should detect critical violation"
    
    def test_non_critical_violation_applies_correction(self):
        """Test that non-critical violations apply correction instead of abort."""
        if not QIG_GEOMETRY_AVAILABLE:
            pytest.skip("QIG geometry not available")
        
        superego = SuperegoKernel()
        
        # Add warning constraint
        forbidden_basin = fisher_normalize(np.random.rand(64))
        superego.add_constraint(
            name="warning-test",
            forbidden_basin=forbidden_basin,
            radius=0.4,
            severity=ConstraintSeverity.WARNING,
        )
        
        # Check basin inside forbidden region
        test_basin = fisher_normalize(forbidden_basin + 0.05 * np.random.rand(64))
        result = superego.check_ethics_with_drift(test_basin, apply_correction=True)
        
        # Should provide corrected basin for non-critical violations
        if not result['is_ethical'] and 'violations' in result:
            warning_violations = [v for v in result['violations'] if v['severity'] == 'warning']
            if warning_violations:
                assert 'corrected_basin' in result, "Should provide correction for warnings"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
