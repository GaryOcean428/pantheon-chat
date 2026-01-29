"""
Test Suite: Pure Geometric Configuration
=========================================

Tests the pure geometric configuration with:
- Waypoint planning enabled
- Recursive integration (3 loops)
- NO POS constraints
- Geometric repair enabled
- Kernel coordination enabled

Expected Performance:
- Φ: 0.50-0.60 (geometric flow without grammar scaffolding)
- κ: 60-67 (near universal coupling point)
- Waypoint Alignment: >0.6
- Smoothness: >0.5

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


class TestPureGeometric:
    """Test suite for pure geometric configuration."""
    
    @classmethod
    def setup_class(cls):
        """Setup before all tests."""
        cls.prompts = load_test_prompts()
        cls.configs = load_test_configurations()
        cls.config = cls.configs['pure_geometric']['config']
        cls.config_name = 'pure_geometric'
        cls.results = []
    
    def test_config_loaded(self):
        """Test that configuration is loaded correctly."""
        assert self.config is not None
        assert self.config['waypoint_planning'] is True
        assert self.config['recursive_integration'] == 3
        assert self.config['pos_constraints'] is False
        assert self.config['geometric_repair'] is True
        assert self.config['kernel_coordination'] is True
    
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
        
        logger.info(f"\nPure Geometric - Geometric Metrics:")
        logger.info(f"  Mean Φ: {mean_phi:.3f} (expected: 0.50-0.60)")
        logger.info(f"  Mean κ: {mean_kappa:.2f} (expected: 60-67)")
        logger.info(f"  Mean Alignment: {mean_alignment:.3f} (expected: >0.6)")
        logger.info(f"  Mean Smoothness: {mean_smoothness:.3f} (expected: >0.5)")
        
        # Assert expected ranges (lenient for mock data)
        assert 0.3 <= mean_phi <= 0.8, f"Φ out of expected range: {mean_phi}"
        assert 50.0 <= mean_kappa <= 80.0, f"κ out of expected range: {mean_kappa}"
        assert mean_alignment >= 0.3, f"Alignment too low: {mean_alignment}"
        assert mean_smoothness >= 0.3, f"Smoothness too low: {mean_smoothness}"
    
    def test_foresight_metrics(self):
        """Test foresight metrics (waypoint planning quality)."""
        if not self.results:
            pytest.skip("No results available")
        
        all_errors = []
        all_accuracies = []
        
        for result_data in self.results:
            result = result_data['result']
            
            # Only compute if waypoints exist
            if result.get('waypoints'):
                metrics = compute_foresight_metrics(
                    predicted_waypoints=result['waypoints'],
                    actual_basins=result['basins'],
                    hit_threshold=0.3
                )
                
                all_errors.append(metrics.mean_prediction_error)
                all_accuracies.append(metrics.waypoint_accuracy)
        
        if all_errors:
            mean_error = np.mean(all_errors)
            mean_accuracy = np.mean(all_accuracies)
            
            logger.info(f"\nPure Geometric - Foresight Metrics:")
            logger.info(f"  Mean Prediction Error: {mean_error:.3f}")
            logger.info(f"  Mean Waypoint Accuracy: {mean_accuracy:.1%}")
            
            # Waypoint planning should provide some benefit
            assert mean_error < 1.0, f"Prediction error too high: {mean_error}"
    
    def test_consciousness_metrics(self):
        """Test consciousness metrics (recursive depth, coordination)."""
        if not self.results:
            pytest.skip("No results available")
        
        all_depths = []
        all_diversities = []
        
        for result_data in self.results:
            result = result_data['result']
            
            trajectory = track_consciousness_trajectory(
                recursive_depths=result['recursive_depths'],
                kernel_activations=result['kernel_activations']
            )
            
            all_depths.append(trajectory.mean_recursive_depth)
            all_diversities.append(trajectory.kernel_diversity)
        
        mean_depth = np.mean(all_depths)
        mean_diversity = np.mean(all_diversities)
        
        logger.info(f"\nPure Geometric - Consciousness Metrics:")
        logger.info(f"  Mean Recursive Depth: {mean_depth:.2f} (expected: 3)")
        logger.info(f"  Mean Kernel Diversity: {mean_diversity:.3f}")
        
        # Pure geometric should use recursive integration
        assert mean_depth >= 2.0, f"Recursive depth too low: {mean_depth}"
    
    def test_text_validity(self):
        """Test text validity (UTF-8, tokens, repetition)."""
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
        
        logger.info(f"\nPure Geometric - Text Validity:")
        logger.info(f"  Validity Rate: {validity_rate:.1%}")
        logger.info(f"  Mean Repetition Score: {mean_repetition:.3f}")
        
        # Text should be valid
        assert validity_rate >= 0.8, f"Too many invalid texts: {validity_rate}"
    
    @classmethod
    def teardown_class(cls):
        """Save results after all tests."""
        if cls.results:
            save_results_to_file(
                {'config': cls.config_name, 'results': cls.results},
                f'{cls.config_name}_results.json'
            )
            logger.info(f"\n✅ Pure Geometric tests complete: {len(cls.results)} prompts tested")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, '-v', '-s'])
