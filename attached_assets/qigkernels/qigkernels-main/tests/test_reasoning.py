"""Tests for QIG Reasoning Infrastructure

Validates that reasoning is MANDATORY and works correctly.
"""

import numpy as np
import pytest

from qigkernels.reasoning import (
    QIGChain,
    GeometricStep,
    ChainResult,
    ReasoningMode,
    ModeTracker,
    create_reasoning_chain,
    detect_mode,
    get_mode_config,
    compute_phi_from_basin,
    compute_kappa,
    fisher_geodesic_distance,
    geodesic_interpolate,
    BASIN_DIM,
    MIN_RECURSIONS,
)


class TestPrimitives:
    """Test low-level geometric primitives."""
    
    def test_compute_phi(self):
        """Test Φ computation from basin."""
        from qigkernels.basin import fisher_normalize_np
        basin = np.random.randn(BASIN_DIM)
        basin = fisher_normalize_np(basin)

        phi = compute_phi_from_basin(basin)

        assert 0 <= phi <= 1, f"Φ should be in [0, 1], got {phi}"

    def test_compute_kappa(self):
        """Test κ computation from basin."""
        from qigkernels.basin import fisher_normalize_np
        basin = np.random.randn(BASIN_DIM)
        basin = fisher_normalize_np(basin)
        
        kappa = compute_kappa(basin)
        
        assert kappa > 0, f"κ should be positive, got {kappa}"
        assert kappa < 200, f"κ should be reasonable, got {kappa}"
    
    def test_geodesic_interpolate(self):
        """Test geodesic interpolation."""
        start = np.random.randn(BASIN_DIM)
        start = np.abs(start) / np.sum(np.abs(start))  # Probability simplex
        
        end = np.random.randn(BASIN_DIM)
        end = np.abs(end) / np.sum(np.abs(end))
        
        result = geodesic_interpolate(start, end, t=0.5)
        
        assert result.shape == start.shape
        assert np.all(result >= 0), "Result should be non-negative"
    
    def test_fisher_distance(self):
        """Test Fisher geodesic distance."""
        basin1 = np.random.randn(BASIN_DIM)
        basin2 = np.random.randn(BASIN_DIM)
        
        dist = fisher_geodesic_distance(basin1, basin2)
        
        assert dist >= 0, f"Distance should be non-negative, got {dist}"


class TestQIGChain:
    """Test QIGChain execution."""
    
    def test_basic_chain(self):
        """Test simple 3-step chain."""
        # Define simple transformations
        def step1(basin):
            return basin + 0.01 * np.random.randn(*basin.shape)
        
        def step2(basin):
            return basin * 0.99
        
        def step3(basin):
            from qigkernels.basin import fisher_normalize_np
            return fisher_normalize_np(basin)

        chain = QIGChain([
            GeometricStep("perturb", step1),
            GeometricStep("compress", step2),
            GeometricStep("normalize", step3),
        ])

        # Run
        from qigkernels.basin import fisher_normalize_np
        initial = np.random.randn(BASIN_DIM)
        initial = fisher_normalize_np(initial)
        
        result = chain.run(initial)
        
        # Validate
        assert isinstance(result, ChainResult)
        assert len(result.trajectory) == 3, "Should have 3 steps"
        assert result.final_basin is not None
        assert result.final_phi > 0
    
    def test_chain_records_all_steps(self):
        """Test that chain records ALL steps for training."""
        steps_recorded = []
        
        def recording_transform(basin):
            steps_recorded.append(basin.copy())
            return basin
        
        chain = QIGChain([
            GeometricStep("step1", recording_transform),
            GeometricStep("step2", recording_transform),
            GeometricStep("step3", recording_transform),
        ])
        
        initial = np.random.randn(BASIN_DIM)
        result = chain.run(initial)
        
        # All steps should be recorded
        assert len(steps_recorded) == 3
        assert len(result.trajectory) == 3
        
        # Each step should have basin_coords for training
        for step in result.trajectory:
            assert 'basin_coords' in step
            assert 'phi_after' in step
            assert 'kappa_after' in step
    
    def test_create_reasoning_chain(self):
        """Test convenience function for creating chains."""
        identity = lambda x: x
        
        chain = create_reasoning_chain(
            encode_fn=identity,
            integrate_fn=identity,
            refine_fn=identity,
        )
        
        assert len(chain.steps) == 3
        assert chain.steps[0].name == "encode"
        assert chain.steps[1].name == "integrate"
        assert chain.steps[2].name == "refine"


class TestReasoningModes:
    """Test reasoning mode detection and tracking."""
    
    def test_detect_linear_mode(self):
        """Test LINEAR mode detection."""
        mode = detect_mode(0.3)
        assert mode == ReasoningMode.LINEAR
    
    def test_detect_geometric_mode(self):
        """Test GEOMETRIC mode detection."""
        mode = detect_mode(0.6)
        assert mode == ReasoningMode.GEOMETRIC
    
    def test_detect_hyperdimensional_mode(self):
        """Test HYPERDIMENSIONAL mode detection."""
        mode = detect_mode(0.85)
        assert mode == ReasoningMode.HYPERDIMENSIONAL
    
    def test_mode_config(self):
        """Test mode configuration retrieval."""
        config = get_mode_config(ReasoningMode.GEOMETRIC)
        
        assert config.mode == ReasoningMode.GEOMETRIC
        assert config.phi_min == 0.45
        assert config.phi_max == 0.80
        assert config.recursion_depth >= MIN_RECURSIONS
    
    def test_mode_tracker(self):
        """Test mode transition tracking."""
        tracker = ModeTracker()
        
        # Start in linear
        tracker.update(0.3)
        assert tracker.current_mode == ReasoningMode.LINEAR
        
        # Transition to geometric
        transition = tracker.update(0.6)
        assert tracker.current_mode == ReasoningMode.GEOMETRIC
        assert transition is not None
        assert transition.from_mode == ReasoningMode.LINEAR
        assert transition.to_mode == ReasoningMode.GEOMETRIC
        
        # Summary
        summary = tracker.get_summary()
        assert summary['transitions'] == 1


class TestMandatoryReasoning:
    """Test that reasoning is MANDATORY, not optional."""
    
    def test_chain_requires_steps(self):
        """Test that QIGChain requires at least 1 step."""
        with pytest.raises(ValueError):
            QIGChain([])  # Should fail
    
    def test_chain_always_executes_all_steps(self):
        """Test that chain executes ALL steps (no early exit before min)."""
        execution_count = [0]
        
        def counting_transform(basin):
            execution_count[0] += 1
            return basin
        
        chain = QIGChain([
            GeometricStep("step1", counting_transform),
            GeometricStep("step2", counting_transform),
            GeometricStep("step3", counting_transform),
        ], min_steps=3)  # All 3 are mandatory
        
        initial = np.random.randn(BASIN_DIM)
        chain.run(initial)
        
        # All 3 steps must execute
        assert execution_count[0] == 3
    
    def test_trajectory_available_for_training(self):
        """Test that trajectory is available for chain-level training."""
        chain = QIGChain([
            GeometricStep("step1", lambda x: x),
            GeometricStep("step2", lambda x: x),
        ])
        
        result = chain.run(np.random.randn(BASIN_DIM))
        
        # Training needs access to all intermediate states
        assert len(result.trajectory) == 2
        for step in result.trajectory:
            # Each step must have basin for loss computation
            assert 'basin_coords' in step
            assert isinstance(step['basin_coords'], np.ndarray)
            assert step['basin_coords'].shape == (BASIN_DIM,)


if __name__ == "__main__":
    # Run tests
    print("Testing QIG Reasoning Infrastructure...")
    
    # Primitives
    test_prim = TestPrimitives()
    test_prim.test_compute_phi()
    print("✅ compute_phi works")
    test_prim.test_compute_kappa()
    print("✅ compute_kappa works")
    test_prim.test_geodesic_interpolate()
    print("✅ geodesic_interpolate works")
    test_prim.test_fisher_distance()
    print("✅ fisher_distance works")
    
    # Chain
    test_chain = TestQIGChain()
    test_chain.test_basic_chain()
    print("✅ Basic chain works")
    test_chain.test_chain_records_all_steps()
    print("✅ Chain records ALL steps (for training)")
    test_chain.test_create_reasoning_chain()
    print("✅ create_reasoning_chain works")
    
    # Modes
    test_modes = TestReasoningModes()
    test_modes.test_detect_linear_mode()
    test_modes.test_detect_geometric_mode()
    test_modes.test_detect_hyperdimensional_mode()
    print("✅ Mode detection works")
    test_modes.test_mode_config()
    print("✅ Mode config works")
    test_modes.test_mode_tracker()
    print("✅ Mode tracker works")
    
    # Mandatory reasoning
    test_mandatory = TestMandatoryReasoning()
    test_mandatory.test_chain_always_executes_all_steps()
    print("✅ Chain executes ALL steps (mandatory)")
    test_mandatory.test_trajectory_available_for_training()
    print("✅ Trajectory available for training loss")
    
    print("\n" + "=" * 60)
    print("QIG Reasoning Infrastructure: ALL TESTS PASSED")
    print("=" * 60)
    print("\nKey validations:")
    print("  ✅ Reasoning is MANDATORY (no bypass)")
    print("  ✅ ALL chain steps are recorded (for training)")
    print("  ✅ Trajectory has basin_coords (for loss computation)")
    print("  ✅ Mode detection works (Linear/Geometric/Hyper)")
    print("\nReady for integration with qig_chat.py!")
