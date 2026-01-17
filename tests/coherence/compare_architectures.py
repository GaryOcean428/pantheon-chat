#!/usr/bin/env python3
"""
Compare Architecture Configurations

Statistical comparison of pure_geometric, plan_realize_repair, and skeleton_only.

Hypothesis: Plan→Realize→Repair should outperform skeleton-only baseline.

Author: QIG Consciousness Project
Date: January 2026
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import statistics

sys.path.insert(0, str(Path(__file__).parent))

from test_utils import load_configurations, save_results


def load_test_results(config_name: str) -> Dict:
    """
    Load test results for a configuration.
    
    Args:
        config_name: Configuration name
        
    Returns:
        Results dictionary
    """
    results_dir = Path(__file__).parent / "results"
    results_file = results_dir / f"{config_name}_results.json"
    
    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")
    
    with open(results_file, 'r') as f:
        return json.load(f)


def extract_metrics(results: Dict) -> Dict[str, List[float]]:
    """
    Extract metric arrays from results.
    
    Args:
        results: Results dictionary
        
    Returns:
        Dictionary of metric name -> values
    """
    metrics = {
        'phi': [],
        'kappa_eff': [],
        'smoothness': [],
        'waypoint_alignment': [],
        'foresight_quality': [],
        'recursive_depth': [],
        'basin_drift': [],
        'regime_stability': [],
    }
    
    for result in results['results']:
        m = result['metrics']
        
        metrics['phi'].append(m['geometric']['phi'])
        metrics['kappa_eff'].append(m['geometric']['kappa_eff'])
        metrics['smoothness'].append(m['trajectory']['smoothness'])
        metrics['basin_drift'].append(m['geometric']['basin_drift'])
        metrics['regime_stability'].append(m['geometric']['regime_stability'])
        metrics['recursive_depth'].append(m['consciousness']['recursive_depth'])
        
        if 'foresight' in m:
            metrics['waypoint_alignment'].append(m['foresight']['waypoint_alignment'])
            metrics['foresight_quality'].append(m['foresight']['foresight_quality'])
    
    return metrics


def compute_statistics(values: List[float]) -> Dict[str, float]:
    """
    Compute statistical summary.
    
    Args:
        values: List of metric values
        
    Returns:
        Dictionary with mean, std, min, max
    """
    if not values:
        return {'mean': 0, 'std': 0, 'min': 0, 'max': 0}
    
    return {
        'mean': statistics.mean(values),
        'std': statistics.stdev(values) if len(values) > 1 else 0,
        'min': min(values),
        'max': max(values),
    }


def compare_configurations(
    config_a: str,
    config_b: str,
    metrics_a: Dict[str, List[float]],
    metrics_b: Dict[str, List[float]]
) -> Dict[str, Dict]:
    """
    Compare two configurations statistically.
    
    Args:
        config_a: First configuration name
        config_b: Second configuration name
        metrics_a: First configuration metrics
        metrics_b: Second configuration metrics
        
    Returns:
        Comparison results
    """
    comparison = {}
    
    for metric_name in metrics_a.keys():
        values_a = metrics_a[metric_name]
        values_b = metrics_b[metric_name]
        
        if not values_a or not values_b:
            continue
        
        stats_a = compute_statistics(values_a)
        stats_b = compute_statistics(values_b)
        
        # Compute delta (positive = B better than A)
        delta = stats_b['mean'] - stats_a['mean']
        
        # Special cases for metrics where lower is better
        if metric_name in ['basin_drift']:
            delta = -delta  # Invert so positive still means B better
        
        # Percent improvement
        if stats_a['mean'] != 0:
            percent_improvement = (delta / abs(stats_a['mean'])) * 100
        else:
            percent_improvement = 0
        
        comparison[metric_name] = {
            f'{config_a}_mean': stats_a['mean'],
            f'{config_b}_mean': stats_b['mean'],
            'delta': delta,
            'percent_improvement': percent_improvement,
            'b_better': delta > 0,
        }
    
    return comparison


def print_comparison_table(comparison: Dict[str, Dict], config_a: str, config_b: str):
    """
    Print comparison table.
    
    Args:
        comparison: Comparison results
        config_a: First configuration name
        config_b: Second configuration name
    """
    print(f"\n{'='*80}")
    print(f"COMPARISON: {config_b.upper()} vs {config_a.upper()}")
    print(f"{'='*80}")
    print(f"\n{'Metric':<25} {config_a:>15} {config_b:>15} {'Delta':>12} {'Better':>8}")
    print("-" * 80)
    
    for metric, data in comparison.items():
        a_val = data[f'{config_a}_mean']
        b_val = data[f'{config_b}_mean']
        delta = data['delta']
        better = '✓' if data['b_better'] else '✗'
        
        print(f"{metric:<25} {a_val:>15.4f} {b_val:>15.4f} {delta:>+12.4f} {better:>8}")


def generate_html_report(
    all_results: Dict[str, Dict],
    comparisons: Dict[str, Dict],
    output_path: Path
):
    """
    Generate HTML report with visualizations.
    
    Args:
        all_results: All configuration results
        comparisons: All comparisons
        output_path: Output HTML file path
    """
    html = """
<!DOCTYPE html>
<html>
<head>
    <title>QIG Coherence Test Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        h1 { color: #333; }
        h2 { color: #666; margin-top: 30px; }
        table { border-collapse: collapse; width: 100%; background: white; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
        th { background-color: #4CAF50; color: white; }
        tr:nth-child(even) { background-color: #f2f2f2; }
        .better { color: green; font-weight: bold; }
        .worse { color: red; }
        .neutral { color: gray; }
        .summary { background: white; padding: 20px; margin: 20px 0; border-radius: 5px; }
        .metric-card { background: white; padding: 15px; margin: 10px 0; border-left: 4px solid #4CAF50; }
    </style>
</head>
<body>
    <h1>QIG Coherence Test Report</h1>
    <p>Generated: <strong>2026-01-16</strong></p>
    
    <div class="summary">
        <h2>Executive Summary</h2>
        <p>This report compares three QIG generation architectures:</p>
        <ul>
            <li><strong>Pure Geometric</strong>: Geometric flow without POS constraints</li>
            <li><strong>Plan→Realize→Repair</strong>: Full architecture with waypoint planning and repair</li>
            <li><strong>Skeleton Only</strong>: Baseline reactive generation</li>
        </ul>
        <p><strong>Hypothesis:</strong> Plan→Realize→Repair should outperform skeleton-only baseline.</p>
    </div>
"""
    
    # Add comparison tables
    for comp_name, comp_data in comparisons.items():
        html += f"\n    <h2>{comp_name.replace('_', ' → ').title()}</h2>\n"
        html += "    <table>\n"
        html += "        <tr><th>Metric</th><th>Config A</th><th>Config B</th><th>Delta</th><th>Better</th></tr>\n"
        
        for metric, data in comp_data.items():
            better_class = 'better' if data['b_better'] else 'worse'
            better_symbol = '✓' if data['b_better'] else '✗'
            
            keys = list(data.keys())
            a_key = [k for k in keys if k.endswith('_mean')][0]
            b_key = [k for k in keys if k.endswith('_mean') and k != a_key][0]
            
            html += f"        <tr>\n"
            html += f"            <td>{metric}</td>\n"
            html += f"            <td>{data[a_key]:.4f}</td>\n"
            html += f"            <td>{data[b_key]:.4f}</td>\n"
            html += f"            <td class='{better_class}'>{data['delta']:+.4f}</td>\n"
            html += f"            <td class='{better_class}'>{better_symbol}</td>\n"
            html += f"        </tr>\n"
        
        html += "    </table>\n"
    
    html += """
</body>
</html>
"""
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(html)


def main():
    """Main comparison routine."""
    print("\n" + "="*80)
    print("QIG ARCHITECTURE COMPARISON")
    print("="*80)
    
    # Load all results
    configs = ['pure_geometric', 'plan_realize_repair', 'skeleton_only']
    all_results = {}
    all_metrics = {}
    
    print("\nLoading results...")
    for config in configs:
        try:
            all_results[config] = load_test_results(config)
            all_metrics[config] = extract_metrics(all_results[config])
            print(f"  ✓ Loaded {config}")
        except FileNotFoundError as e:
            print(f"  ✗ {e}")
            print(f"\n⚠️  Run the individual test scripts first:")
            print(f"     python3 tests/coherence/test_pure_geometric.py")
            print(f"     python3 tests/coherence/test_plan_realize_repair.py")
            print(f"     python3 tests/coherence/test_skeleton_baseline.py")
            return 1
    
    # Compare configurations
    comparisons = {}
    
    # 1. Plan→Realize→Repair vs Skeleton (main hypothesis)
    print("\n" + "="*80)
    print("MAIN HYPOTHESIS TEST")
    print("="*80)
    comp = compare_configurations(
        'skeleton_only',
        'plan_realize_repair',
        all_metrics['skeleton_only'],
        all_metrics['plan_realize_repair']
    )
    comparisons['plan_realize_repair_vs_skeleton'] = comp
    print_comparison_table(comp, 'skeleton_only', 'plan_realize_repair')
    
    # Check if hypothesis holds
    phi_improved = comp['phi']['b_better']
    alignment_exists = 'waypoint_alignment' in comp
    
    if phi_improved and (not alignment_exists or comp['waypoint_alignment']['b_better']):
        print("\n✓ HYPOTHESIS CONFIRMED: Plan→Realize→Repair outperforms skeleton baseline")
    else:
        print("\n✗ HYPOTHESIS FAILED: Plan→Realize→Repair does NOT consistently outperform baseline")
        print("   This indicates a problem with the advanced architecture!")
    
    # 2. Pure Geometric vs Skeleton
    comp = compare_configurations(
        'skeleton_only',
        'pure_geometric',
        all_metrics['skeleton_only'],
        all_metrics['pure_geometric']
    )
    comparisons['pure_geometric_vs_skeleton'] = comp
    print_comparison_table(comp, 'skeleton_only', 'pure_geometric')
    
    # 3. Plan→Realize→Repair vs Pure Geometric
    comp = compare_configurations(
        'pure_geometric',
        'plan_realize_repair',
        all_metrics['pure_geometric'],
        all_metrics['plan_realize_repair']
    )
    comparisons['plan_realize_repair_vs_pure_geometric'] = comp
    print_comparison_table(comp, 'pure_geometric', 'plan_realize_repair')
    
    # Generate HTML report
    report_path = Path(__file__).parent / "results" / "comparison_report.html"
    generate_html_report(all_results, comparisons, report_path)
    print(f"\n✓ HTML report generated: {report_path}")
    
    # Save JSON comparison
    json_path = Path(__file__).parent / "results" / "comparison_results.json"
    save_results(comparisons, json_path)
    print(f"✓ JSON comparison saved: {json_path}")
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80 + "\n")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
