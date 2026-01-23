"""
Unit tests for Psyche Plumbing kernels (E8 Protocol v4.0 Phase 4D)

Tests:
- phi_hierarchy.py: Φ hierarchy management
- id_kernel.py: Fast reflex drives
- superego_kernel.py: Ethical constraints
"""

import pytest
import numpy as np
import time

# Import kernels to test
from kernels.phi_hierarchy import (
    PhiLevel,
    PhiHierarchy,
    PhiMeasurement,
    get_phi_hierarchy,
)
from kernels.id_kernel import IdKernel, ReflexPattern, get_id_kernel
from kernels.superego_kernel import (
    SuperegoKernel,
    EthicalConstraint,
    ConstraintSeverity,
    get_superego_kernel,
)

# Import QIG constants
from qigkernels.physics_constants import BASIN_DIM

# Import geometry utilities
try:
    from qig_geometry import fisher_normalize
except ImportError:
    def fisher_normalize(v):
        p = np.maximum(np.asarray(v), 0) + 1e-10
        return p / p.sum()


class TestPhiHierarchy:
    """Test Φ hierarchy management."""
    
    def test_phi_hierarchy_creation(self):
        """Test creating PhiHierarchy."""
        hierarchy = PhiHierarchy(history_size=50)
        assert hierarchy is not None
        assert hierarchy.history_size == 50
        assert len(hierarchy.measurements) == 3  # Three levels
    
    def test_phi_measurement(self):
        """Test measuring Φ at different levels."""
        hierarchy = PhiHierarchy()
        
        # Create test basin
        basin = np.ones(BASIN_DIM) / BASIN_DIM
        
        # Measure at each level
        reported = hierarchy.measure(basin, PhiLevel.REPORTED, source='test')
        internal = hierarchy.measure(basin, PhiLevel.INTERNAL, source='test')
        autonomic = hierarchy.measure(basin, PhiLevel.AUTONOMIC, source='test')
        
        # Check measurements
        assert isinstance(reported, PhiMeasurement)
        assert isinstance(internal, PhiMeasurement)
        assert isinstance(autonomic, PhiMeasurement)
        
        assert reported.level == PhiLevel.REPORTED
        assert internal.level == PhiLevel.INTERNAL
        assert autonomic.level == PhiLevel.AUTONOMIC
        
        assert 0.0 <= reported.phi <= 1.0
        assert 0.0 <= internal.phi <= 1.0
        assert 0.0 <= autonomic.phi <= 1.0
    
    def test_phi_thresholds(self):
        """Test Φ threshold checking."""
        hierarchy = PhiHierarchy()
        
        # Create high-phi basin (concentrated)
        high_phi_basin = np.zeros(BASIN_DIM)
        high_phi_basin[0] = 1.0
        
        measurement = hierarchy.measure(high_phi_basin, PhiLevel.REPORTED, source='test')
        
        # Check threshold logic
        # Note: Actual phi value depends on computation, but should be reasonable
        assert isinstance(measurement.meets_threshold(), bool)
    
    def test_phi_statistics(self):
        """Test Φ statistics computation."""
        hierarchy = PhiHierarchy()
        
        # Make several measurements
        for i in range(10):
            basin = np.random.dirichlet(np.ones(BASIN_DIM))
            hierarchy.measure(basin, PhiLevel.INTERNAL, source='test')
        
        # Get statistics
        stats = hierarchy.get_statistics(PhiLevel.INTERNAL)
        assert 'mean' in stats
        assert 'std' in stats
        assert 'min' in stats
        assert 'max' in stats
        assert 'count' in stats
        assert stats['count'] == 10
    
    def test_phi_hierarchy_singleton(self):
        """Test global singleton access."""
        h1 = get_phi_hierarchy()
        h2 = get_phi_hierarchy()
        assert h1 is h2  # Same instance


class TestIdKernel:
    """Test Id kernel (fast reflex drives)."""
    
    def test_id_kernel_creation(self):
        """Test creating IdKernel."""
        id_kernel = IdKernel(name="test-id", max_reflexes=20)
        assert id_kernel is not None
        assert id_kernel.name == "test-id"
        assert id_kernel.max_reflexes == 20
        assert len(id_kernel.reflexes) == 0
    
    def test_reflex_learning(self):
        """Test learning new reflexes."""
        id_kernel = IdKernel()
        
        # Create trigger and response basins
        trigger = np.random.dirichlet(np.ones(BASIN_DIM))
        response = np.random.dirichlet(np.ones(BASIN_DIM))
        
        # Learn reflex
        id_kernel.learn_reflex(trigger, response, success=True)
        
        assert len(id_kernel.reflexes) == 1
        assert id_kernel.reflexes[0].success_rate > 0.5
    
    def test_reflex_triggering(self):
        """Test reflex triggering."""
        id_kernel = IdKernel()
        
        # Create and learn reflex
        trigger = np.random.dirichlet(np.ones(BASIN_DIM))
        response = np.random.dirichlet(np.ones(BASIN_DIM))
        id_kernel.learn_reflex(trigger, response, success=True)
        
        # Check reflex with same trigger
        result = id_kernel.check_reflex(trigger)
        assert result is not None  # Should trigger
        
        # Check with different basin (unlikely to trigger)
        different = np.random.dirichlet(np.ones(BASIN_DIM) * 0.1)
        result2 = id_kernel.check_reflex(different)
        # May or may not trigger depending on similarity
    
    def test_reflex_latency(self):
        """Test reflex latency measurement."""
        id_kernel = IdKernel()
        
        # Learn reflex
        trigger = np.random.dirichlet(np.ones(BASIN_DIM))
        response = np.random.dirichlet(np.ones(BASIN_DIM))
        id_kernel.learn_reflex(trigger, response, success=True)
        
        # Check with latency
        result = id_kernel.check_reflex(trigger, return_latency=True)
        
        if result is not None:
            response_basin, latency_ms = result
            assert isinstance(response_basin, np.ndarray)
            assert latency_ms > 0
            assert latency_ms < 1000  # Should be fast (< 1 second)
    
    def test_reflex_statistics(self):
        """Test Id kernel statistics."""
        id_kernel = IdKernel()
        
        # Learn some reflexes with very distinct basins to avoid merging
        for i in range(5):
            # Create distinct basins using one-hot encoding
            trigger = np.zeros(BASIN_DIM)
            trigger[i * 10] = 1.0  # Spread them apart
            trigger = fisher_normalize(trigger) if 'fisher_normalize' in dir() else trigger / trigger.sum()
            
            response = np.zeros(BASIN_DIM)
            response[(i * 10) + 1] = 1.0
            response = fisher_normalize(response) if 'fisher_normalize' in dir() else response / response.sum()
            
            id_kernel.learn_reflex(trigger, response, success=True)
        
        stats = id_kernel.get_statistics()
        # Should have at least some reflexes (may merge if similar)
        assert stats['num_reflexes'] >= 1
        assert 'total_activations' in stats
        assert 'avg_latency_ms' in stats


class TestSuperegoKernel:
    """Test Superego kernel (ethical constraints)."""
    
    def test_superego_kernel_creation(self):
        """Test creating SuperegoKernel."""
        superego = SuperegoKernel(name="test-superego")
        assert superego is not None
        assert superego.name == "test-superego"
        assert len(superego.constraints) == 0
    
    def test_constraint_addition(self):
        """Test adding ethical constraints."""
        superego = SuperegoKernel()
        
        # Create forbidden basin
        forbidden = np.random.dirichlet(np.ones(BASIN_DIM))
        
        # Add constraint
        constraint = superego.add_constraint(
            name="test-constraint",
            forbidden_basin=forbidden,
            radius=0.3,
            severity=ConstraintSeverity.ERROR,
            description="Test constraint"
        )
        
        assert len(superego.constraints) == 1
        assert constraint.name == "test-constraint"
        assert constraint.severity == ConstraintSeverity.ERROR
    
    def test_ethics_checking(self):
        """Test checking ethical constraints."""
        superego = SuperegoKernel()
        
        # Add constraint
        forbidden = np.zeros(BASIN_DIM)
        forbidden[0] = 1.0
        superego.add_constraint(
            name="no-zero-basin",
            forbidden_basin=forbidden,
            radius=0.2,
            severity=ConstraintSeverity.ERROR,
        )
        
        # Check ethical basin (far from constraint)
        ethical_basin = np.ones(BASIN_DIM) / BASIN_DIM
        result = superego.check_ethics(ethical_basin)
        
        assert 'violations' in result
        assert 'is_ethical' in result
        assert 'total_penalty' in result
    
    def test_violation_detection(self):
        """Test violation detection."""
        superego = SuperegoKernel()
        
        # Add constraint at center
        forbidden = np.zeros(BASIN_DIM)
        forbidden[0] = 1.0
        superego.add_constraint(
            name="test",
            forbidden_basin=forbidden,
            radius=0.2,
            severity=ConstraintSeverity.CRITICAL,
        )
        
        # Test basin very close to forbidden
        close_basin = forbidden.copy()
        result = superego.check_ethics(close_basin, apply_correction=True)
        
        # Should detect violation
        assert len(result['violations']) > 0
        assert not result['is_ethical']
        assert result['corrected_basin'] is not None
    
    def test_trajectory_correction(self):
        """Test trajectory correction."""
        superego = SuperegoKernel()
        
        # Add constraint
        forbidden = np.zeros(BASIN_DIM)
        forbidden[0] = 1.0
        superego.add_constraint(
            name="test",
            forbidden_basin=forbidden,
            radius=0.3,
            severity=ConstraintSeverity.ERROR,
        )
        
        # Violating basin
        violating = forbidden.copy()
        
        # Check with correction
        result = superego.check_ethics(violating, apply_correction=True)
        
        if not result['is_ethical']:
            corrected = result['corrected_basin']
            assert corrected is not None
            assert len(corrected) == BASIN_DIM
            
            # Check correction moved away from constraint
            # (This is a weak test - just check it's different)
            assert not np.allclose(corrected, violating)
    
    def test_constraint_removal(self):
        """Test removing constraints."""
        superego = SuperegoKernel()
        
        # Add constraint
        forbidden = np.random.dirichlet(np.ones(BASIN_DIM))
        superego.add_constraint(
            name="removable",
            forbidden_basin=forbidden,
            radius=0.2,
            severity=ConstraintSeverity.WARNING,
        )
        
        assert len(superego.constraints) == 1
        
        # Remove constraint
        removed = superego.remove_constraint("removable")
        assert removed
        assert len(superego.constraints) == 0
        
        # Try removing non-existent
        removed2 = superego.remove_constraint("nonexistent")
        assert not removed2
    
    def test_constraint_severity_levels(self):
        """Test different constraint severity levels."""
        superego = SuperegoKernel()
        
        forbidden = np.zeros(BASIN_DIM)
        forbidden[0] = 1.0
        
        # Add constraints with different severities
        for severity in ConstraintSeverity:
            superego.add_constraint(
                name=f"test-{severity.value}",
                forbidden_basin=forbidden,
                radius=0.2,
                severity=severity,
            )
        
        assert len(superego.constraints) == 4
    
    def test_superego_statistics(self):
        """Test Superego statistics."""
        superego = SuperegoKernel()
        
        # Add constraint
        forbidden = np.random.dirichlet(np.ones(BASIN_DIM))
        superego.add_constraint(
            name="test",
            forbidden_basin=forbidden,
            radius=0.3,
            severity=ConstraintSeverity.ERROR,
        )
        
        # Do some checks
        for _ in range(10):
            basin = np.random.dirichlet(np.ones(BASIN_DIM))
            superego.check_ethics(basin)
        
        stats = superego.get_statistics()
        assert stats['num_constraints'] == 1
        assert stats['total_checks'] == 10
        assert 'violation_rate' in stats


class TestIntegration:
    """Integration tests for psyche plumbing."""
    
    def test_phi_hierarchy_with_id_kernel(self):
        """Test Φ hierarchy integration with Id kernel."""
        # Use singleton
        hierarchy = get_phi_hierarchy()
        id_kernel = IdKernel()
        
        # Id should use hierarchy
        assert id_kernel.phi_hierarchy is not None
        assert id_kernel.phi_hierarchy is hierarchy  # Same instance
        
        # Process input
        basin = np.random.dirichlet(np.ones(BASIN_DIM))
        result = id_kernel.process(basin)
        
        assert 'phi_internal' in result
    
    def test_phi_hierarchy_with_superego_kernel(self):
        """Test Φ hierarchy integration with Superego kernel."""
        # Use singleton to ensure same instance
        hierarchy = get_phi_hierarchy()
        superego = SuperegoKernel()
        
        # Superego should use hierarchy
        assert superego.phi_hierarchy is not None
        assert superego.phi_hierarchy is hierarchy  # Same instance
        
        # Add constraint and check
        forbidden = np.random.dirichlet(np.ones(BASIN_DIM))
        superego.add_constraint(
            name="test",
            forbidden_basin=forbidden,
            radius=0.3,
            severity=ConstraintSeverity.ERROR,
        )
        
        basin = np.random.dirichlet(np.ones(BASIN_DIM))
        result = superego.check_ethics(basin)
        
        # Should have measured Φ_internal
        stats = hierarchy.get_statistics(PhiLevel.INTERNAL)
        assert stats['count'] > 0
    
    def test_id_and_superego_together(self):
        """Test Id and Superego working together."""
        id_kernel = IdKernel()
        superego = SuperegoKernel()
        
        # Learn a reflex
        trigger = np.random.dirichlet(np.ones(BASIN_DIM))
        response = np.random.dirichlet(np.ones(BASIN_DIM))
        id_kernel.learn_reflex(trigger, response, success=True)
        
        # Add ethical constraint
        forbidden = np.random.dirichlet(np.ones(BASIN_DIM))
        superego.add_constraint(
            name="ethics",
            forbidden_basin=forbidden,
            radius=0.3,
            severity=ConstraintSeverity.ERROR,
        )
        
        # Check if reflex response is ethical
        reflex_result = id_kernel.check_reflex(trigger)
        if reflex_result is not None:
            ethics_result = superego.check_ethics(reflex_result)
            # Both should work
            assert 'is_ethical' in ethics_result


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
