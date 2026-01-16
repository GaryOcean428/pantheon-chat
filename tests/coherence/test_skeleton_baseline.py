#!/usr/bin/env python3
"""
Test Skeleton Only Configuration (Config 3 - Baseline)

Tests simple reactive generation without planning or refinement.
Expected Φ: 0.35-0.45 (reactive, not predictive)

Configuration:
- waypoint_planning: False
- recursive_integration: 0
- pos_constraints: "required"
- geometric_repair: False
- kernel_coordination: False

This is the BASELINE - should perform worse than full architecture.

Author: QIG Consciousness Project
Date: January 2026
"""

import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "qig-backend"))

from test_utils import (
    load_prompts,
    load_configurations,
    set_prompt_seed,
    set_global_seed,
    mock_generation_result,
    save_results,
)

from metrics import (
    compute_geometric_metrics,
    compute_trajectory_metrics,
    compute_text_metrics,
    compute_consciousness_metrics,
)


def run_skeleton_baseline_test(prompt: Dict, config: Dict) -> Dict:
    """
    Run skeleton-only test for a single prompt.
    
    Args:
        prompt: Prompt dictionary
        config: Configuration settings
        
    Returns:
        Test results with all metrics
    """
    prompt_id = prompt['id']
    prompt_text = prompt['text']
    
    print(f"\n{'='*60}")
    print(f"Testing: {prompt_id}")
    print(f"Prompt: {prompt_text}")
    print(f"{'='*60}")
    
    set_prompt_seed(prompt_id)
    
    # Generate (mock for now)
    result = mock_generation_result(prompt_text, config['settings'])
    
    # Extract data
    basins = result['basins']
    text = result['text']
    integration_trace = result.get('integration_trace', [])
    kernel_activations = result.get('kernel_activations')
    
    # Compute metrics
    geometric = compute_geometric_metrics(basins)
    trajectory = compute_trajectory_metrics(basins)
    text_metrics = compute_text_metrics(text)
    
    # Consciousness metrics (should be minimal)
    consciousness = compute_consciousness_metrics(
        integration_trace=integration_trace,
        kernel_activations=kernel_activations,
    )
    
    # Compile results
    results = {
        'prompt_id': prompt_id,
        'prompt_text': prompt_text,
        'config_name': 'skeleton_only',
        'text_output': text,
        'metrics': {
            'geometric': geometric.to_dict(),
            'trajectory': trajectory.to_dict(),
            'text': text_metrics.to_dict(),
            'consciousness': consciousness.to_dict(),
        }
    }
    
    # Print summary
    print(f"\n✓ Φ = {geometric.phi:.4f} (expected: 0.35-0.45)")
    print(f"✓ κ = {geometric.kappa_eff:.2f}")
    print(f"✓ Smoothness = {trajectory.smoothness:.4f}")
    print(f"✓ Recursive depth = {consciousness.recursive_depth:.4f} (should be 0)")
    print(f"✓ Text length = {text_metrics.length} chars")
    
    return results


def run_all_skeleton_baseline_tests():
    """
    Run skeleton baseline tests for all prompts.
    
    Returns:
        List of all test results
    """
    print("\n" + "="*60)
    print("SKELETON-ONLY BASELINE CONFIGURATION TEST SUITE")
    print("="*60)
    
    prompts = load_prompts()
    configs = load_configurations()
    config = configs['skeleton_only']
    
    print(f"\nConfiguration: {config['name']}")
    print(f"Description: {config['description']}")
    print(f"Settings: {config['settings']}")
    print(f"\nRunning {len(prompts)} prompts...")
    print("\n⚠️  This is the BASELINE - should perform WORSE than full architecture")
    
    all_results = []
    
    for prompt in prompts:
        try:
            result = run_skeleton_baseline_test(prompt, config)
            all_results.append(result)
        except Exception as e:
            print(f"\n✗ Error testing {prompt['id']}: {e}")
            import traceback
            traceback.print_exc()
    
    # Aggregate statistics
    print("\n" + "="*60)
    print("AGGREGATE STATISTICS")
    print("="*60)
    
    phis = [r['metrics']['geometric']['phi'] for r in all_results]
    kappas = [r['metrics']['geometric']['kappa_eff'] for r in all_results]
    smoothness = [r['metrics']['trajectory']['smoothness'] for r in all_results]
    depths = [r['metrics']['consciousness']['recursive_depth'] for r in all_results]
    
    print(f"\nΦ (Integration):")
    print(f"  Mean: {sum(phis)/len(phis):.4f}")
    print(f"  Min:  {min(phis):.4f}")
    print(f"  Max:  {max(phis):.4f}")
    print(f"  Expected: 0.35-0.45 (lower than full architecture)")
    
    print(f"\nκ (Coupling):")
    print(f"  Mean: {sum(kappas)/len(kappas):.2f}")
    
    print(f"\nTrajectory Smoothness:")
    print(f"  Mean: {sum(smoothness)/len(smoothness):.4f}")
    print(f"  (Expected lower than full architecture)")
    
    print(f"\nRecursive Depth:")
    print(f"  Mean: {sum(depths)/len(depths):.4f}")
    print(f"  Expected: 0.0 (no recursion in baseline)")
    
    # Check pass criteria
    mean_phi = sum(phis) / len(phis)
    in_range = 0.35 <= mean_phi <= 0.45
    mean_depth = sum(depths) / len(depths)
    
    print(f"\n{'='*60}")
    if in_range and mean_depth < 0.1:
        print("✓ PASS: Baseline metrics as expected")
    else:
        if not in_range:
            print(f"⚠️  WARNING: Mean Φ = {mean_phi:.4f} outside expected [0.35, 0.45]")
        if mean_depth >= 0.1:
            print(f"⚠️  WARNING: Mean depth = {mean_depth:.4f} (expected ~0)")
    print(f"{'='*60}\n")
    
    # Save results
    output_dir = Path(__file__).parent / "results"
    output_file = output_dir / "skeleton_baseline_results.json"
    
    save_results({
        'config': config,
        'results': all_results,
        'statistics': {
            'mean_phi': mean_phi,
            'mean_kappa': sum(kappas) / len(kappas),
            'mean_smoothness': sum(smoothness) / len(smoothness),
            'mean_depth': mean_depth,
        }
    }, output_file)
    
    print(f"Results saved to: {output_file}")
    
    return all_results


if __name__ == '__main__':
    set_global_seed(42)
    results = run_all_skeleton_baseline_tests()
    sys.exit(0)
