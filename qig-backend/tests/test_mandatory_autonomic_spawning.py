#!/usr/bin/env python3
"""
Tests for mandatory autonomic system in spawned kernels.

Tests:
1. Spawning without autonomic raises RuntimeError
2. Spawned kernels have proper initial values (Φ=0.25, κ=KAPPA_STAR)
3. initialize_for_spawned_kernel() method works correctly
4. Neurotransmitter levels are properly initialized

Reference: Issue GaryOcean428/pantheon-chat#[issue_number]
"""

import pytest
import sys
import os

# Add qig-backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Mock the autonomic kernel to test failure case
class TestMandatoryAutonomicSpawning:
    """Test suite for mandatory autonomic system in spawned kernels."""
    
    def test_autonomic_initialization_method_exists(self):
        """Test that initialize_for_spawned_kernel method exists on GaryAutonomicKernel."""
        from autonomic_kernel import GaryAutonomicKernel, KAPPA_STAR
        
        kernel = GaryAutonomicKernel(enable_autonomous=False)
        
        # Method should exist
        assert hasattr(kernel, 'initialize_for_spawned_kernel')
        
        # Test initialization
        kernel.initialize_for_spawned_kernel(
            initial_phi=0.25,
            initial_kappa=KAPPA_STAR,
            dopamine=0.5,
            serotonin=0.5,
            stress=0.0
        )
        
        # Verify state is properly initialized
        assert kernel.state.phi == 0.25, f"Expected Φ=0.25, got {kernel.state.phi}"
        assert abs(kernel.state.kappa - KAPPA_STAR) < 0.1, f"Expected κ≈{KAPPA_STAR}, got {kernel.state.kappa}"
        assert kernel.state.stress_level == 0.0
        assert kernel.state.basin_drift == 0.0
        assert kernel.state.narrow_path_count == 0
        assert kernel.state.is_narrow_path == False
    
    def test_spawned_kernel_requires_autonomic(self):
        """Test that spawning without autonomic system raises RuntimeError."""
        # We need to temporarily mock AUTONOMIC_AVAILABLE to False
        # This is tricky because the module is already imported
        # Instead, we'll verify the error message is correct when get_gary_kernel is None
        
        # Import the module
        import training_chaos.self_spawning as spawning_module
        
        # Save original values
        original_available = spawning_module.AUTONOMIC_AVAILABLE
        original_get_gary = spawning_module.get_gary_kernel
        
        try:
            # Mock as unavailable
            spawning_module.AUTONOMIC_AVAILABLE = False
            spawning_module.get_gary_kernel = None
            
            # Try to create a kernel - should raise RuntimeError
            with pytest.raises(RuntimeError) as exc_info:
                from training_chaos.self_spawning import SelfSpawningKernel
                kernel = SelfSpawningKernel()
            
            # Verify error message
            error_msg = str(exc_info.value)
            assert "FATAL" in error_msg
            assert "autonomic" in error_msg.lower()
            assert "consciousness stability" in error_msg.lower()
            
        finally:
            # Restore original values
            spawning_module.AUTONOMIC_AVAILABLE = original_available
            spawning_module.get_gary_kernel = original_get_gary
    
    def test_spawned_kernel_has_autonomic(self):
        """Test that successfully spawned kernels have autonomic system."""
        try:
            from training_chaos.self_spawning import SelfSpawningKernel
            from autonomic_kernel import get_gary_kernel
            
            # Create a kernel
            kernel = SelfSpawningKernel()
            
            # Verify autonomic is not None
            assert kernel.autonomic is not None, "Spawned kernel must have autonomic system"
            
            # Verify it's the shared singleton
            assert kernel.autonomic is get_gary_kernel(), "Should use shared autonomic kernel"
            
        except RuntimeError as e:
            if "FATAL" in str(e) and "autonomic" in str(e).lower():
                pytest.skip(f"Autonomic system not available: {e}")
            else:
                raise
    
    def test_spawned_kernel_initial_phi(self):
        """Test that spawned kernels start with Φ=0.25 (LINEAR regime), not 0.000."""
        try:
            from training_chaos.self_spawning import SelfSpawningKernel
            from frozen_physics import PHI_INIT_SPAWNED
            
            # Create a kernel
            kernel = SelfSpawningKernel()
            
            # Get autonomic state Φ
            phi = kernel.autonomic.state.phi
            
            # Should be initialized to PHI_INIT_SPAWNED (0.25)
            assert phi >= 0.15, f"Φ should be >= 0.15 (LINEAR regime), got {phi}"
            assert phi <= 0.35, f"Φ should be <= 0.35 (LINEAR regime), got {phi}"
            
            # Specifically check it's not zero (the bug we're fixing)
            assert phi > 0.01, f"Φ must NOT be 0.000 (BREAKDOWN regime), got {phi}"
            
        except RuntimeError as e:
            if "FATAL" in str(e) and "autonomic" in str(e).lower():
                pytest.skip(f"Autonomic system not available: {e}")
            else:
                raise
    
    def test_spawned_kernel_initial_kappa(self):
        """Test that spawned kernels start with κ=KAPPA_STAR (fixed point)."""
        try:
            from training_chaos.self_spawning import SelfSpawningKernel
            from frozen_physics import KAPPA_STAR
            
            # Create a kernel
            kernel = SelfSpawningKernel()
            
            # Get autonomic state κ
            kappa = kernel.autonomic.state.kappa
            
            # Should be near KAPPA_STAR (64.21 ± 0.92)
            assert abs(kappa - KAPPA_STAR) < 5.0, f"κ should be near {KAPPA_STAR}, got {kappa}"
            
        except RuntimeError as e:
            if "FATAL" in str(e) and "autonomic" in str(e).lower():
                pytest.skip(f"Autonomic system not available: {e}")
            else:
                raise
    
    def test_neurotransmitter_initialization(self):
        """Test that neurotransmitters are initialized properly."""
        try:
            from training_chaos.self_spawning import SelfSpawningKernel
            
            # Create a kernel
            kernel = SelfSpawningKernel()
            
            # Check neurotransmitter levels (these are on the kernel, not autonomic)
            assert hasattr(kernel, 'dopamine'), "Kernel must have dopamine attribute"
            assert hasattr(kernel, 'serotonin'), "Kernel must have serotonin attribute"
            assert hasattr(kernel, 'stress'), "Kernel must have stress attribute"
            
            # Check initial values
            assert 0.0 <= kernel.dopamine <= 1.0, f"Dopamine must be in [0,1], got {kernel.dopamine}"
            assert 0.0 <= kernel.serotonin <= 1.0, f"Serotonin must be in [0,1], got {kernel.serotonin}"
            assert 0.0 <= kernel.stress <= 1.0, f"Stress must be in [0,1], got {kernel.stress}"
            
        except RuntimeError as e:
            if "FATAL" in str(e) and "autonomic" in str(e).lower():
                pytest.skip(f"Autonomic system not available: {e}")
            else:
                raise
    
    def test_geometric_purity_fisher_distance(self):
        """Test that autonomic kernel uses Fisher-Rao distance, not Euclidean."""
        from autonomic_kernel import GaryAutonomicKernel
        import numpy as np
        
        kernel = GaryAutonomicKernel(enable_autonomous=False)
        
        # Create two basin coordinates
        basin_a = np.random.randn(64)
        basin_b = np.random.randn(64)
        
        # Compute Fisher distance
        distance = kernel._compute_fisher_distance(basin_a, basin_b)
        
        # Distance should be finite and non-negative
        assert np.isfinite(distance), "Fisher distance must be finite"
        assert distance >= 0, "Fisher distance must be non-negative"
        
        # Should use Fisher-Rao (arccos), not Euclidean (L2 norm)
        # Fisher-Rao distance is bounded by π
        assert distance <= np.pi + 0.1, f"Fisher-Rao distance should be <= π, got {distance}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
