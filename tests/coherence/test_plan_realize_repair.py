#!/usr/bin/env python3
"""
Test Plan→Realize→Repair Configuration (Config 2)

Tests full architecture with waypoint planning, realization, and repair.
Expected Φ: 0.65-0.75 (geometric + grammatical)

Configuration:
- waypoint_planning: True
- recursive_integration: 3
- pos_constraints: "optional"
- geometric_repair: True
- kernel_coordination: True
- repair_iterations: 3
- repair_radius: 0.2

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
    compute_foresight_metrics,
    compute_trajectory_metrics,
    compute_text_metrics,
    compute_consciousness_metrics,
)


def run_plan_realize_repair_test(prompt: Dict, config: Dict) -> Dict:
    """
    Run Plan→Realize→Repair test for a single prompt.
    
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
    
    # Foresight metrics
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
        'config_name': 'plan_realize_repair',
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
    print(f"\n✓ Φ = {geometric.phi:.4f} (target: 0.65-0.75)")
    print(f"✓ κ = {geometric.kappa_eff:.2f} (optimal: 64)")
    print(f"✓ Smoothness = {trajectory.smoothness:.4f}")
    print(f"✓ Recursive depth = {consciousness.recursive_depth:.4f}")
    
    if foresight:
        print(f"✓ Waypoint alignment = {foresight.waypoint_alignment:.4f}")
        print(f"✓ Foresight quality = {foresight.foresight_quality:.4f}")
    
    print(f"✓ Text length = {text_metrics.length} chars")
    print(f"✓ UTF-8 valid = {text_metrics.is_valid_utf8}")
    
    return results


def run_all_plan_realize_repair_tests():
    """
    Run Plan→Realize→Repair tests for all prompts.
    
    Returns:
        List of all test results
    """
    print("\n" + "="*60)
    print("PLAN→REALIZE→REPAIR CONFIGURATION TEST SUITE")
    print("="*60)
    
    prompts = load_prompts()
    configs = load_configurations()
    config = configs['plan_realize_repair']
    
    print(f"\nConfiguration: {config['name']}")
    print(f"Description: {config['description']}")
    print(f"Settings: {config['settings']}")
    print(f"\nRunning {len(prompts)} prompts...")
    
    all_results = []
    
    for prompt in prompts:
        try:
            result = run_plan_realize_repair_test(prompt, config)
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
    
    # Foresight statistics
    alignments = []
    qualities = []
    for r in all_results:
        if 'foresight' in r['metrics']:
            alignments.append(r['metrics']['foresight']['waypoint_alignment'])
            qualities.append(r['metrics']['foresight']['foresight_quality'])
    
    print(f"\nΦ (Integration):")
    print(f"  Mean: {sum(phis)/len(phis):.4f}")
    print(f"  Min:  {min(phis):.4f}")
    print(f"  Max:  {max(phis):.4f}")
    print(f"  Target: 0.65-0.75")
    
    print(f"\nκ (Coupling):")
    print(f"  Mean: {sum(kappas)/len(kappas):.2f}")
    print(f"  Optimal: 64")
    
    print(f"\nTrajectory Smoothness:")
    print(f"  Mean: {sum(smoothness)/len(smoothness):.4f}")
    
    if alignments:
        print(f"\nWaypoint Alignment:")
        print(f"  Mean: {sum(alignments)/len(alignments):.4f}")
        print(f"  Target: >0.70")
    
    if qualities:
        print(f"\nForesight Quality:")
        print(f"  Mean: {sum(qualities)/len(qualities):.4f}")
    
    # Check pass criteria
    mean_phi = sum(phis) / len(phis)
    in_range = 0.65 <= mean_phi <= 0.75
    
    mean_alignment = sum(alignments) / len(alignments) if alignments else 0
    alignment_good = mean_alignment >= 0.70
    
    print(f"\n{'='*60}")
    if in_range and alignment_good:
        print("✓ PASS: Mean Φ and alignment within expected ranges")
    else:
        if not in_range:
            print(f"✗ WARNING: Mean Φ = {mean_phi:.4f} outside target [0.65, 0.75]")
        if not alignment_good:
            print(f"✗ WARNING: Mean alignment = {mean_alignment:.4f} below target 0.70")
    print(f"{'='*60}\n")
    
    # Save results
    output_dir = Path(__file__).parent / "results"
    output_file = output_dir / "plan_realize_repair_results.json"
    
    save_results({
        'config': config,
        'results': all_results,
        'statistics': {
            'mean_phi': mean_phi,
            'mean_kappa': sum(kappas) / len(kappas),
            'mean_smoothness': sum(smoothness) / len(smoothness),
            'mean_alignment': mean_alignment if alignments else 0,
        }
    }, output_file)
    
    print(f"Results saved to: {output_file}")
    
    return all_results


if __name__ == '__main__':
    set_global_seed(42)
    results = run_all_plan_realize_repair_tests()
    sys.exit(0)
