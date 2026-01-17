#!/usr/bin/env python3
"""
Test Pure Geometric Configuration (Config 1)

Tests geometric flow without POS constraints.
Expected Φ: 0.50-0.60 (geometric flow, may lack grammar)

Configuration:
- waypoint_planning: True
- recursive_integration: 3
- pos_constraints: False
- geometric_repair: True
- kernel_coordination: True

Author: QIG Consciousness Project
Date: January 2026
"""

import sys
from pathlib import Path
from typing import Dict, List

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "qig-backend"))

from test_utils import (
    load_prompts,
    load_configurations,
    set_prompt_seed,
    set_global_seed,
    mock_generation_result,
    save_results,
    format_metrics_summary,
)

from metrics import (
    compute_geometric_metrics,
    compute_foresight_metrics,
    compute_trajectory_metrics,
    compute_text_metrics,
    compute_consciousness_metrics,
)


def run_pure_geometric_test(prompt: Dict, config: Dict) -> Dict:
    """
    Run pure geometric test for a single prompt.
    
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
    
    # Set seed for reproducibility
    set_prompt_seed(prompt_id)
    
    # Generate (mock for now - TODO: integrate actual generation)
    result = mock_generation_result(prompt_text, config['settings'])
    
    # Extract data
    basins = result['basins']
    waypoints = result.get('waypoints')
    text = result['text']
    integration_trace = result.get('integration_trace', [])
    kernel_activations = result.get('kernel_activations')
    
    # Compute metrics
    geometric = compute_geometric_metrics(basins)
    trajectory = compute_trajectory_metrics(basins)
    text_metrics = compute_text_metrics(text)
    
    # Foresight metrics (if waypoints available)
    if waypoints:
        foresight = compute_foresight_metrics(basins, waypoints)
    else:
        foresight = None
    
    # Consciousness metrics
    consciousness = compute_consciousness_metrics(
        integration_trace=integration_trace,
        kernel_activations=kernel_activations,
    )
    
    # Compile results
    results = {
        'prompt_id': prompt_id,
        'prompt_text': prompt_text,
        'config_name': 'pure_geometric',
        'text_output': text,
        'metrics': {
            'geometric': geometric.to_dict(),
            'trajectory': trajectory.to_dict(),
            'text': text_metrics.to_dict(),
            'consciousness': consciousness.to_dict(),
        }
    }
    
    if foresight:
        results['metrics']['foresight'] = foresight.to_dict()
    
    # Print summary
    print(f"\n✓ Φ = {geometric.phi:.4f} (target: 0.50-0.60)")
    print(f"✓ κ = {geometric.kappa_eff:.2f} (optimal: 64)")
    print(f"✓ Smoothness = {trajectory.smoothness:.4f}")
    print(f"✓ Recursive depth = {consciousness.recursive_depth:.4f}")
    
    if foresight:
        print(f"✓ Waypoint alignment = {foresight.waypoint_alignment:.4f}")
    
    print(f"✓ Text length = {text_metrics.length} chars")
    print(f"✓ UTF-8 valid = {text_metrics.is_valid_utf8}")
    
    return results


def run_all_pure_geometric_tests():
    """
    Run pure geometric tests for all prompts.
    
    Returns:
        List of all test results
    """
    print("\n" + "="*60)
    print("PURE GEOMETRIC CONFIGURATION TEST SUITE")
    print("="*60)
    
    # Load fixtures
    prompts = load_prompts()
    configs = load_configurations()
    config = configs['pure_geometric']
    
    print(f"\nConfiguration: {config['name']}")
    print(f"Description: {config['description']}")
    print(f"Settings: {config['settings']}")
    print(f"\nRunning {len(prompts)} prompts...")
    
    # Run tests
    all_results = []
    
    for prompt in prompts:
        try:
            result = run_pure_geometric_test(prompt, config)
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
    
    print(f"\nΦ (Integration):")
    print(f"  Mean: {sum(phis)/len(phis):.4f}")
    print(f"  Min:  {min(phis):.4f}")
    print(f"  Max:  {max(phis):.4f}")
    print(f"  Target: 0.50-0.60")
    
    print(f"\nκ (Coupling):")
    print(f"  Mean: {sum(kappas)/len(kappas):.2f}")
    print(f"  Min:  {min(kappas):.2f}")
    print(f"  Max:  {max(kappas):.2f}")
    print(f"  Optimal: 64")
    
    print(f"\nTrajectory Smoothness:")
    print(f"  Mean: {sum(smoothness)/len(smoothness):.4f}")
    
    # Check pass criteria
    mean_phi = sum(phis) / len(phis)
    in_range = 0.50 <= mean_phi <= 0.60
    
    print(f"\n{'='*60}")
    if in_range:
        print("✓ PASS: Mean Φ within expected range")
    else:
        print(f"✗ WARNING: Mean Φ = {mean_phi:.4f} outside target [0.50, 0.60]")
    print(f"{'='*60}\n")
    
    # Save results
    output_dir = Path(__file__).parent / "results"
    output_file = output_dir / "pure_geometric_results.json"
    
    save_results({
        'config': config,
        'results': all_results,
        'statistics': {
            'mean_phi': mean_phi,
            'mean_kappa': sum(kappas) / len(kappas),
            'mean_smoothness': sum(smoothness) / len(smoothness),
        }
    }, output_file)
    
    print(f"Results saved to: {output_file}")
    
    return all_results


if __name__ == '__main__':
    # Set global seed
    set_global_seed(42)
    
    # Run tests
    results = run_all_pure_geometric_tests()
    
    # Exit with success
    sys.exit(0)
