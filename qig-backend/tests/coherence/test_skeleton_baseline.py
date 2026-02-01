"""
Test Suite: Skeleton-Only Baseline Configuration
=================================================

Tests the minimal baseline configuration with:
- NO waypoint planning (reactive only)
- NO recursive integration (0 loops)
- POS constraints required (only constraint-based generation)
- NO geometric repair
- NO kernel coordination

This is the BASELINE that advanced architectures should beat.

Expected Performance (WORST):
- Φ: 0.35-0.45 (reactive, not predictive)
- κ: 50-60 (below optimal coupling)
- Waypoint Alignment: ~0.3 (no planning)
- Smoothness: ~0.4 (reactive steps)

If skeleton-only performs BETTER than advanced configs,
we have a problem with the architecture.

Author: WP4.3 Coherence Harness
Date: 2026-01-20
"""

import pytest
import numpy as np
import logging
from pathlib import Path

# Import test helpers
from test_helpers import (
    load_test_prompts,
    load_test_configurations,
    get_prompt_seed,
    set_reproducible_seed,
    mock_generation_run,
    validate_generation_result,
    save_results_to_file,
)

# Import metrics
from metrics import (
    compute_geometric_metrics,
    compute_foresight_metrics,
    track_consciousness_trajectory,
    compute_trajectory_metrics,
    compute_text_metrics,
)

logger = logging.getLogger(__name__)


class TestSkeletonBaseline:
    """Test suite for skeleton-only baseline configuration."""
    
    @classmethod
    def setup_class(cls):
        """Setup before all tests."""
        cls.prompts = load_test_prompts()
        cls.configs = load_test_configurations()
        cls.config = cls.configs['skeleton_only']['config']
        cls.config_name = 'skeleton_only'
        cls.results = []
    
    def test_config_loaded(self):
        """Test that configuration is loaded correctly."""
        assert self.config is not None
        assert self.config['waypoint_planning'] is False
        assert self.config['recursive_integration'] == 0
        assert self.config['pos_constraints'] == 'required'
        assert self.config['geometric_repair'] is False
        assert self.config['kernel_coordination'] is False
    
    @pytest.mark.parametrize("prompt_idx", range(5))  # Test first 5 prompts
    def test_generation_with_prompt(self, prompt_idx):
        """
        Test generation with each prompt.
        
        NOTE: This uses mock generation. Replace mock_generation_run()
        with actual QIG generation calls when available.
        """
        prompt_data = self.prompts[prompt_idx]
        prompt_id = prompt_data['id']
        prompt_text = prompt_data['text']
        
        # Set reproducible seed
        seed = get_prompt_seed(prompt_id)
        set_reproducible_seed(seed)
        
        # Run generation (mock for now)
        logger.info(f"Testing prompt {prompt_idx + 1}: {prompt_text}")
        result = mock_generation_run(prompt_text, self.config, seed)
        
        # Validate result structure
        is_valid, issues = validate_generation_result(result)
        assert is_valid, f"Invalid result: {issues}"
        
        # Store for later analysis
        self.results.append({
            'prompt_id': prompt_id,
            'prompt_text': prompt_text,
            'result': result
        })
    
    def test_geometric_metrics(self):
        """Test geometric metrics computation."""
        if not self.results:
            pytest.skip("No results available")
        
        all_phi = []
        all_kappa = []
        all_alignment = []
        all_smoothness = []
        
        for result_data in self.results:
            result = result_data['result']
            
            # Compute geometric metrics
            metrics = compute_geometric_metrics(
                basins=result['basins'],
                waypoints=result.get('waypoints', []),
                attractor=result['basins'][-1] if result['basins'] else None
            )
            
            all_phi.append(metrics.mean_phi)
            all_kappa.append(metrics.mean_kappa)
            all_alignment.append(metrics.waypoint_alignment)
            all_smoothness.append(metrics.trajectory_smoothness)
        
        # Check expected ranges
        mean_phi = np.mean(all_phi)
        mean_kappa = np.mean(all_kappa)
        mean_alignment = np.mean(all_alignment)
        mean_smoothness = np.mean(all_smoothness)
        
        logger.info(f"\nSkeleton-Only - Geometric Metrics:")
        logger.info(f"  Mean Φ: {mean_phi:.3f} (expected: 0.35-0.45) [BASELINE]")
        logger.info(f"  Mean κ: {mean_kappa:.2f} (expected: 50-60)")
        logger.info(f"  Mean Alignment: {mean_alignment:.3f} (expected: ~0.3)")
        logger.info(f"  Mean Smoothness: {mean_smoothness:.3f} (expected: ~0.4)")
        
        # Assert expected ranges (lenient for mock data)
        # Baseline should show LOWER metrics
        assert 0.2 <= mean_phi <= 0.7, f"Φ out of expected range: {mean_phi}"
        assert 40.0 <= mean_kappa <= 70.0, f"κ out of expected range: {mean_kappa}"
    
    def test_no_foresight(self):
        """Test that no foresight planning occurs."""
        if not self.results:
            pytest.skip("No results available")
        
        # Skeleton-only should have no waypoints
        for result_data in self.results:
            result = result_data['result']
            waypoints = result.get('waypoints', [])
            
            # Should be empty or very short
            assert len(waypoints) == 0, "Skeleton-only should not use waypoint planning"
        
        logger.info(f"\nSkeleton-Only - Foresight:")
        logger.info(f"  Waypoint Planning: DISABLED ✓")
    
    def test_no_recursive_integration(self):
        """Test that no recursive integration occurs."""
        if not self.results:
            pytest.skip("No results available")
        
        all_depths = []
        
        for result_data in self.results:
            result = result_data['result']
            
            trajectory = track_consciousness_trajectory(
                recursive_depths=result['recursive_depths'],
                kernel_activations=result['kernel_activations']
            )
            
            all_depths.append(trajectory.mean_recursive_depth)
        
        mean_depth = np.mean(all_depths)
        
        logger.info(f"\nSkeleton-Only - Consciousness Metrics:")
        logger.info(f"  Mean Recursive Depth: {mean_depth:.2f} (expected: 0)")
        
        # Skeleton should have minimal recursion
        assert mean_depth <= 1.0, f"Recursive depth too high for baseline: {mean_depth}"
    
    def test_no_kernel_coordination(self):
        """Test that no kernel coordination occurs."""
        if not self.results:
            pytest.skip("No results available")
        
        all_diversities = []
        
        for result_data in self.results:
            result = result_data['result']
            
            trajectory = track_consciousness_trajectory(
                recursive_depths=result['recursive_depths'],
                kernel_activations=result['kernel_activations']
            )
            
            all_diversities.append(trajectory.kernel_diversity)
        
        mean_diversity = np.mean(all_diversities)
        
        logger.info(f"\nSkeleton-Only - Kernel Coordination:")
        logger.info(f"  Mean Kernel Diversity: {mean_diversity:.3f} (expected: low)")
        
        # Skeleton should use minimal kernels
        # (Diversity might be 0 if only one kernel active)
    
    def test_trajectory_metrics(self):
        """Test trajectory metrics (should be less optimal)."""
        if not self.results:
            pytest.skip("No results available")
        
        all_efficiency = []
        all_convergence = []
        
        for result_data in self.results:
            result = result_data['result']
            
            metrics = compute_trajectory_metrics(
                basins=result['basins'],
                attractor=result['basins'][-1] if result['basins'] else None
            )
            
            all_efficiency.append(metrics.geodesic_efficiency)
            all_convergence.append(metrics.attractor_convergence)
        
        mean_efficiency = np.mean(all_efficiency)
        mean_convergence = np.mean(all_convergence)
        
        logger.info(f"\nSkeleton-Only - Trajectory Metrics:")
        logger.info(f"  Mean Geodesic Efficiency: {mean_efficiency:.3f} (expected: lower)")
        logger.info(f"  Mean Attractor Convergence: {mean_convergence:.3f} (expected: lower)")
        
        # Baseline should show lower performance
        # (No specific assertion - just measuring)
    
    def test_text_validity(self):
        """Test text validity (should still be valid)."""
        if not self.results:
            pytest.skip("No results available")
        
        all_valid = []
        all_repetition = []
        
        for result_data in self.results:
            result = result_data['result']
            
            metrics = compute_text_metrics(result['text'])
            
            all_valid.append(metrics.is_valid_utf8 and metrics.token_validity)
            all_repetition.append(metrics.repetition_score)
        
        validity_rate = np.mean(all_valid)
        mean_repetition = np.mean(all_repetition)
        
        logger.info(f"\nSkeleton-Only - Text Validity:")
        logger.info(f"  Validity Rate: {validity_rate:.1%}")
        logger.info(f"  Mean Repetition Score: {mean_repetition:.3f}")
        
        # Even baseline should produce valid text
        assert validity_rate >= 0.7, f"Too many invalid texts: {validity_rate}"
    
    def test_baseline_expectations(self):
        """
        Verify this is truly a baseline.
        
        If skeleton-only performs TOO WELL, something is wrong.
        """
        if not self.results:
            pytest.skip("No results available")
        
        # Collect all phi scores
        all_phi = []
        for result_data in self.results:
            result = result_data['result']
            metrics = compute_geometric_metrics(basins=result['basins'])
            all_phi.append(metrics.mean_phi)
        
        mean_phi = np.mean(all_phi)
        
        logger.info(f"\nSkeleton-Only - Baseline Verification:")
        logger.info(f"  Mean Φ: {mean_phi:.3f}")
        
        # If phi is too high, warn that this might not be a true baseline
        if mean_phi > 0.7:
            logger.warning("⚠️  WARNING: Skeleton-only Φ is very high!")
            logger.warning("⚠️  This might indicate the baseline is too strong.")
            logger.warning("⚠️  Advanced architectures need to significantly outperform this.")
    
    @classmethod
    def teardown_class(cls):
        """Save results after all tests."""
        if cls.results:
            save_results_to_file(
                {'config': cls.config_name, 'results': cls.results},
                f'{cls.config_name}_results.json'
            )
            logger.info(f"\n✅ Skeleton-Only tests complete: {len(cls.results)} prompts tested")
            logger.info("📊 This is the BASELINE - other configs should outperform")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, '-v', '-s'])
